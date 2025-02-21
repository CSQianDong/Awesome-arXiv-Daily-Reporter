# LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention 

**Title (ZH)**: LServe：高效处理长序列的统一稀疏注意力大规模语言模型服务 

**Authors**: Shang Yang, Junxian Guo, Haotian Tang, Qinghao Hu, Guangxuan Xiao, Jiaming Tang, Yujun Lin, Zhijian Liu, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14866)  

**Abstract**: Large language models (LLMs) have shown remarkable potential in processing long sequences, yet efficiently serving these long-context models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）在处理长序列方面展示了显著的潜力，但由于预填充阶段中的注意力机制所导致的二次计算复杂度以及解码阶段中KV缓存的大内存占用，高效地服务于这些长上下文模型仍是一个挑战。为了解决这些问题，我们引入了LServe系统，该系统通过混合稀疏注意力来加速长序列LLM的服务。该方法将预填充和解码中的不同硬件友好型结构化稀疏模式统一到一个框架中，其中对不重要令牌的计算以块为单位进行跳过。LServe展示出静态和动态稀疏性在长上下文LLM注意力机制中的兼容性。这种设计通过结合这些优化而实现了乘法加速。具体来说，在预填充和解码阶段，我们转换了大约一半的注意力头为几乎免费的流式注意力头。此外，我们发现，只需要一个常数数量的KV页面即可保持长上下文能力，与其上下文长度无关。我们还设计了一种分层的KV页面选择策略，该策略基于查询为中心的相似性动态剪枝KV页面。在平均情况下，LServe将LLM预填充加速2.9倍，解码加速1.3-2.1倍，同时保持长上下文准确性。代码已发布于此 <https://> 地址。 

---
# Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning 

**Title (ZH)**: 将大语言模型导向提出 good 问题：临床推理案例研究 

**Authors**: Shuyue Stella Li, Jimin Mun, Faeze Brahman, Jonathan S. Ilgen, Yulia Tsvetkov, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.14860)  

**Abstract**: Large language models (LLMs) often fail to ask effective questions under uncertainty, making them unreliable in domains where proactive information-gathering is essential for decisionmaking. We present ALFA, a framework that improves LLM question-asking by (i) decomposing the notion of a "good" question into a set of theory-grounded attributes (e.g., clarity, relevance), (ii) controllably synthesizing attribute-specific question variations, and (iii) aligning models via preference-based optimization to explicitly learn to ask better questions along these fine-grained attributes. Focusing on clinical reasoning as a case study, we introduce the MediQ-AskDocs dataset, composed of 17k real-world clinical interactions augmented with 80k attribute-specific preference pairs of follow-up questions, as well as a novel expert-annotated interactive healthcare QA task to evaluate question-asking abilities. Models aligned with ALFA reduce diagnostic errors by 56.6% on MediQ-AskDocs compared to SOTA instruction-tuned LLMs, with a question-level win-rate of 64.4% and strong generalizability. Our findings suggest that explicitly guiding question-asking with structured, fine-grained attributes offers a scalable path to improve LLMs, especially in expert application domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不确定性情况下往往不能提出有效的问题，这使得它们在需要主动信息收集以做出决策的领域中不可靠。我们提出了一种名为ALFA的框架，通过（i）将“好问题”的概念分解为一套理论支持的特征（例如，清晰度、相关性），（ii）有控制地生成特定特征的问题变体，以及（iii）通过基于偏好的优化对模型进行对齐，以明确学习在这些细粒度特征上提出更好问题。以临床推理为例，我们介绍了MediQ-AskDocs数据集，该数据集包含17,000个真实的临床互动，并附加了80,000个针对随访问题的具体特征偏好对，以及一项新的专家注释的互动医疗问答任务，用于评估提问能力。与最先进的指令调优的LLM相比，使用ALFA对齐的模型在MediQ-AskDocs上将诊断错误减少了56.6%，问题级别的胜出率为64.4%，且具有较强的泛化能力。我们的研究结果表明，明确地用结构化的细粒度特征引导提问为提高LLM提供了可扩展的路径，特别是在专家应用领域。 

---
# FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling 

**Title (ZH)**: FR-Spec：通过频率排序推测性采样加速大规模词汇语言模型 

**Authors**: Weilin Zhao, Tengyu Pan, Xu Han, Yudi Zhang, Ao Sun, Yuxiang Huang, Kaihuo Zhang, Weilun Zhao, Yuxuan Li, Jianyong Wang, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.14856)  

**Abstract**: Speculative sampling has emerged as an important technique for accelerating the auto-regressive generation process of large language models (LLMs) by utilizing a draft-then-verify mechanism to produce multiple tokens per forward pass. While state-of-the-art speculative sampling methods use only a single layer and a language modeling (LM) head as the draft model to achieve impressive layer compression, their efficiency gains are substantially reduced for large-vocabulary LLMs, such as Llama-3-8B with a vocabulary of 128k tokens. To address this, we present FR-Spec, a frequency-ranked speculative sampling framework that optimizes draft candidate selection through vocabulary space compression. By constraining the draft search to a frequency-prioritized token subset, our method reduces LM Head computation overhead by 75% while ensuring the equivalence of the final output distribution. Experiments across multiple datasets demonstrate an average of 1.12$\times$ speedup over the state-of-the-art speculative sampling method EAGLE-2. 

**Abstract (ZH)**: 投机抽样作为一种重要技术，通过利用先拟后验机制，在每一前向传播过程中生成多个标记，已经成为了加速大规模语言模型（LLMs）的自回归生成过程的关键方法。当前最先进的投机采样方法仅使用一层和一个语言模型（LM）头作为草稿模型，从而实现了显著的层压缩，但其效率提升在具有大词汇量的语言模型（如词汇量达128K的Llama-3-8B）中大幅减少。为解决这一问题，我们提出了一种基于频率排名的投机抽样框架FR-Spec，通过词汇空间压缩优化草稿候选的选择。通过将草稿搜索限定在频率优先的令牌子集中，我们的方法在保持最终输出分布等价性的同时，将LM头的计算开销降低了75%。我们在多个数据集上的实验结果显示，FR-Spec相比最先进的投机采样方法EAGLE-2，平均实现了1.12倍的速度提升。 

---
# Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps 

**Title (ZH)**: 通过反学习推理步骤来衡量思维链的忠实性 

**Authors**: Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasović, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14829)  

**Abstract**: When prompted to think step-by-step, language models (LMs) produce a chain of thought (CoT), a sequence of reasoning steps that the model supposedly used to produce its prediction. However, despite much work on CoT prompting, it is unclear if CoT reasoning is faithful to the models' parameteric beliefs. We introduce a framework for measuring parametric faithfulness of generated reasoning, and propose Faithfulness by Unlearning Reasoning steps (FUR), an instance of this framework. FUR erases information contained in reasoning steps from model parameters. We perform experiments unlearning CoTs of four LMs prompted on four multi-choice question answering (MCQA) datasets. Our experiments show that FUR is frequently able to change the underlying models' prediction by unlearning key steps, indicating when a CoT is parametrically faithful. Further analysis shows that CoTs generated by models post-unlearning support different answers, hinting at a deeper effect of unlearning. Importantly, CoT steps identified as important by FUR do not align well with human notions of plausbility, emphasizing the need for specialized alignment 

**Abstract (ZH)**: 在被提示进行逐步思考时，语言模型（LMs）会产生一个推理链（CoT），即模型据称用于生成其预测的一系列推理步骤。然而，尽管在CoT提示方面做了许多工作，但CoT推理是否忠实于模型的参数化信念尚不明确。我们提出了一种衡量生成推理参数化忠实性的框架，并提出了基于消除推理步骤（FUR）的方法，这是一种该框架的具体应用。FUR方法会从模型参数中删除包含在推理步骤中的信息。我们对四个LMs在四个多项选择题解答（MCQA）数据集上被提示生成的CoT进行了消除学习实验。实验结果显示，FUR经常能够通过消除关键步骤来改变模型的底层预测，这表明CoT可能是参数化的忠实性表现。进一步的分析表明，模型在消除学习后生成的CoT支持不同的答案，暗示了消除学习的更深层次影响。重要的是，FUR识别出的重要CoT步骤与人类对合理性的真实感知并不一致，这强调了需要专门的对齐方法。 

---
# Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning 

**Title (ZH)**: 逻辑-RL：基于规则的强化学习解锁LLM推理能力 

**Authors**: Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14768)  

**Abstract**: Inspired by the success of DeepSeek-R1, we explore the potential of rule-based reinforcement learning (RL) in large reasoning models. To analyze reasoning dynamics, we use synthetic logic puzzles as training data due to their controllable complexity and straightforward answer verification. We make some key technical contributions that lead to effective and stable RL training: a system prompt that emphasizes the thinking and answering process, a stringent format reward function that penalizes outputs for taking shortcuts, and a straightforward training recipe that achieves stable convergence. Our 7B model develops advanced reasoning skills-such as reflection, verification, and summarization-that are absent from the logic corpus. Remarkably, after training on just 5K logic problems, it demonstrates generalization abilities to the challenging math benchmarks AIME and AMC. 

**Abstract (ZH)**: 受DeepSeek-R1成功的启发，我们探索了基于规则的强化学习（RL）在大型推理模型中的潜在价值。为了分析推理动态，我们使用合成逻辑谜题作为训练数据，因为这些数据具有可控的复杂性和直接的答案验证性。我们在以下几个关键技术贡献方面取得了进展，这些贡献使得RL训练既有效又稳定：一个强调思考和回答过程的系统提示，一个严格的格式奖励函数，该函数对采取捷径的行为进行惩罚，以及一个简洁的训练配方，实现了稳定的收敛。我们的7B模型发展出了先进的推理技能，如反思、验证和总结，这些技能在逻辑语料库中是不存在的。值得注意的是，在仅仅训练了5000个逻辑问题之后，该模型展示了对具有挑战性的数学基准测试AIME和AMC的泛化能力。 

---
# Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning 

**Title (ZH)**: 带有可解释推理的逐步医疗声明事实验证系统 

**Authors**: Juraj Vladika, Ivana Hacajová, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2502.14765)  

**Abstract**: Fact verification (FV) aims to assess the veracity of a claim based on relevant evidence. The traditional approach for automated FV includes a three-part pipeline relying on short evidence snippets and encoder-only inference models. More recent approaches leverage the multi-turn nature of LLMs to address FV as a step-by-step problem where questions inquiring additional context are generated and answered until there is enough information to make a decision. This iterative method makes the verification process rational and explainable. While these methods have been tested for encyclopedic claims, exploration on domain-specific and realistic claims is missing. In this work, we apply an iterative FV system on three medical fact-checking datasets and evaluate it with multiple settings, including different LLMs, external web search, and structured reasoning using logic predicates. We demonstrate improvements in the final performance over traditional approaches and the high potential of step-by-step FV systems for domain-specific claims. 

**Abstract (ZH)**: 事实验证（Fact Verification, FV）的目标是基于相关证据评估某一断言的真实性。传统的自动化FV方法依赖于简短的证据片段和仅编码的推理模型，形成了一个三阶段的工作流程。近年来，方法逐渐转向利用大型语言模型（LLM）的多轮对话特性，将FV问题视为逐步解决问题，通过生成并回答询问额外上下文的问题，直到有足够的信息作出判断。这种迭代方法使得验证过程既合理又具有可解释性。尽管这些方法在百科类断言上得到了测试，但对特定领域和真实场景下的断言探索却相对缺失。本文中，我们在三个医疗事实核查数据集上应用了一种迭代的FV系统，并在不同的大型语言模型、外部网络搜索和结构化逻辑推理等多种设置下进行了评估。我们展示了迭代FV系统相较于传统方法在最终性能上的改进，并指出了逐步FV系统在特定领域断言上的高潜力。 

---
# Large Language Models Struggle to Describe the Haystack without Human Help: Human-in-the-loop Evaluation of LLMs 

**Title (ZH)**: 大型语言模型在没有人类帮助的情况下难以描述文档的核心内容：关于大型语言模型的人类在环评估 

**Authors**: Zongxia Li, Lorena Calvo-Bartolomé, Alexander Hoyle, Paiheng Xu, Alden Dima, Juan Francisco Fung, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.14748)  

**Abstract**: A common use of NLP is to facilitate the understanding of large document collections, with a shift from using traditional topic models to Large Language Models. Yet the effectiveness of using LLM for large corpus understanding in real-world applications remains under-explored. This study measures the knowledge users acquire with unsupervised, supervised LLM-based exploratory approaches or traditional topic models on two datasets. While LLM-based methods generate more human-readable topics and show higher average win probabilities than traditional models for data exploration, they produce overly generic topics for domain-specific datasets that do not easily allow users to learn much about the documents. Adding human supervision to the LLM generation process improves data exploration by mitigating hallucination and over-genericity but requires greater human effort. In contrast, traditional. models like Latent Dirichlet Allocation (LDA) remain effective for exploration but are less user-friendly. We show that LLMs struggle to describe the haystack of large corpora without human help, particularly domain-specific data, and face scaling and hallucination limitations due to context length constraints. Dataset available at https://huggingface. co/datasets/zli12321/Bills. 

**Abstract (ZH)**: 自然语言处理（NLP）的一个常见用途是帮助理解大规模文档集合，从使用传统的话题模型转向使用大型语言模型（LLM）。然而，LLM 在实际应用中对大规模语料库的理解效果仍缺乏充分探索。本研究通过在两个数据集上使用无监督和监督的LLM探索方法或传统话题模型，测量用户获取的知识。虽然基于LLM的方法生成了更易于理解的主题，并在数据探索中显示出更高的平均获胜概率，但它们对特定领域的数据集生成的主题过于通用，使用户难以学到很多关于文档的知识。通过在LLM生成过程中添加人工监督，可以减轻幻觉和通用性过强的问题，但需要更多的手工努力。相比之下，传统的模型如潜在狄利克雷分配（LDA）仍然适用于探索，但用户友好性较差。我们表明，没有人工帮助，LLM 在描述大规模语料库（尤其是特定领域的数据）时显得力不从心，并且由于上下文长度限制，面临扩展性和幻觉的挑战。数据集可从 <https://huggingface.co/datasets/zli12321/Bills> 获取。 

---
# SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines 

**Title (ZH)**: SuperGPQA：跨285个研究生学科领域扩展大语言模型评估 

**Authors**: M-A-P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, Kang Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, Kaixing Deng, Shuyue Guo, Shian Jia, Sichao Jiang, Yiyan Liao, Rui Li, Qinrui Li, Sirun Li, Yizhi Li, Yunwen Li, Dehua Ma, Yuansheng Ni, Haoran Que, Qiyao Wang, Zhoufutu Wen, Siwei Wu, Tianshun Xing, Ming Xu, Zhenzhu Yang, Zekun Moore Wang, Junting Zhou, Yuelin Bai, Xingyuan Bu, Chenglin Cai, Liang Chen, Yifan Chen, Chengtuo Cheng, Tianhao Cheng, Keyi Ding, Siming Huang, Yun Huang, Yaoru Li, Yizhe Li, Zhaoqun Li, Tianhao Liang, Chengdong Lin, Hongquan Lin, Yinghao Ma, Zhongyuan Peng, Zifan Peng, Qige Qi, Shi Qiu, Xingwei Qu, Yizhou Tan, Zili Wang, Chenqing Wang, Hao Wang, Yiya Wang, Yubo Wang, Jiajun Xu, Kexin Yang, Ruibin Yuan, Yuanhao Yue, Tianyang Zhan, Chun Zhang, Jingyang Zhang, Xiyue Zhang, Xingjian Zhang, Yue Zhang, Yongchi Zhao, Xiangyu Zheng, Chenghua Zhong, Yang Gao, Zhoujun Li, Dayiheng Liu, Qian Liu, Tianyu Liu, Shiwen Ni, Junran Peng, Yujia Qin, Wenbo Su, Guoyin Wang, Shi Wang, Jian Yang, Min Yang, Meng Cao, Xiang Yue, Zhaoxiang Zhang, Wangchunshu Zhou, Jiaheng Liu, Qunshu Lin, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14739)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学、物理学和计算机科学等主流学术领域中展现出了卓越的能力。然而，人类知识涵盖了超过200个专门学科，远超现有基准的涵盖范围。在许多专门领域中，特别是轻工业、农业以及以服务为导向的学科中，LLMs的能力仍缺乏充分的评估。为了解决这一缺口，我们提出了SuperGPQA这一全面基准，用于评估涵盖285个学科的研究生级知识和推理能力。我们的基准利用了新的人类-LLM协作过滤机制，在基于LLM响应和专家反馈的迭代完善过程中，排除了简单或模棱两可的问题。我们的实验结果揭示了当前最先进LLMs在不同知识领域中的显著改进空间（例如，专注于推理的模型DeepSeek-R1在其SuperGPQA上的最高准确率为61.82%），突显了当前模型能力与人工通用智能之间的巨大差距。此外，我们还提供了大规模注释流程管理的全面见解，涉及超过80位专家注释员和一个交互式的LLM辅助系统，为未来同类规模的研究项目提供了宝贵的指导方法。 

---
# How to Get Your LLM to Generate Challenging Problems for Evaluation 

**Title (ZH)**: 如何让大语言模型生成具有挑战性的评估问题 

**Authors**: Arkil Patel, Siva Reddy, Dzmitry Bahdanau  

**Link**: [PDF](https://arxiv.org/pdf/2502.14678)  

**Abstract**: The pace of evolution of Large Language Models (LLMs) necessitates new approaches for rigorous and comprehensive evaluation. Traditional human annotation is increasingly impracticable due to the complexities and costs involved in generating high-quality, challenging problems. In this work, we introduce CHASE, a unified framework to synthetically generate challenging problems using LLMs without human involvement. For a given task, our approach builds a hard problem in a bottom-up manner from simpler components. Moreover, our framework decomposes the generation process into independently verifiable sub-tasks, thereby ensuring a high level of quality and correctness. We implement CHASE to create evaluation benchmarks across three diverse domains: (1) document-based question answering, (2) repository-level code completion, and (3) math reasoning. The performance of state-of-the-art LLMs on these synthetic benchmarks lies in the range of 40-60% accuracy, thereby demonstrating the effectiveness of our framework at generating challenging problems. We publicly release our benchmarks and code. 

**Abstract (ZH)**: 大型语言模型（LLMs）的发展速度 necessitates 新的方法来进行严谨和全面的评估。传统的手工注释越来越不切实际，因为生成高质量、具有挑战性的问题涉及复杂性和高昂的成本。在本文中，我们引入了CHASE，这是一种无需人工干预即可使用LLMs合成生成具有挑战性问题的统一框架。对于给定的任务，我们的方法从简单的组件自底向上构建一个困难的问题。此外，我们的框架将生成过程分解为可独立验证的子任务，从而确保较高的质量和正确性。我们实现了CHASE，在三个不同的领域创建了评估基准：（1）基于文档的问答，（2）代码仓库级别的代码完成，以及（3）数学推理。最先进的LLMs在这三个合成基准上的性能准确率范围为40-60%，这表明我们的框架在生成具有挑战性的问题方面具有有效性。我们还将我们的基准和代码公开发布。 

---
# Explanations of Deep Language Models Explain Language Representations in the Brain 

**Title (ZH)**: 深度语言模型的解释揭示了大脑中的语言表示 

**Authors**: Maryam Rahimi, Yadollah Yaghoobzadeh, Mohammad Reza Daliri  

**Link**: [PDF](https://arxiv.org/pdf/2502.14671)  

**Abstract**: Recent advances in artificial intelligence have given rise to large language models (LLMs) that not only achieve human-like performance but also share computational principles with the brain's language processing mechanisms. While previous research has primarily focused on aligning LLMs' internal representations with neural activity, we introduce a novel approach that leverages explainable AI (XAI) methods to forge deeper connections between the two domains. Using attribution methods, we quantified how preceding words contribute to an LLM's next-word predictions and employed these explanations to predict fMRI recordings from participants listening to the same narratives. Our findings demonstrate that attribution methods robustly predict brain activity across the language network, surpassing traditional internal representations in early language areas. This alignment is hierarchical: early-layer explanations correspond to the initial stages of language processing in the brain, while later layers align with more advanced stages. Moreover, the layers more influential on LLM next-word prediction$\unicode{x2014}$those with higher attribution scores$\unicode{x2014}$exhibited stronger alignment with neural activity. This work establishes a bidirectional bridge between AI and neuroscience. First, we demonstrate that attribution methods offer a powerful lens for investigating the neural mechanisms of language comprehension, revealing how meaning emerges from preceding context. Second, we propose using brain alignment as a metric to evaluate the validity of attribution methods, providing a framework for assessing their biological plausibility. 

**Abstract (ZH)**: 近年来，人工智能的最新进展催生了大规模语言模型（LLMs），这些模型不仅在性能上达到了类人的水平，还在计算原理上与大脑的语言处理机制相似。虽然之前的研究所主要集中在使LLMs的内部表示与神经活动对齐，但我们引入了一种新的方法，利用可解释的人工智能（XAI）方法，建立了两个领域之间的更深层次联系。通过归因方法，我们量化了前一个词语对LLMs下一个词语预测的贡献，并利用这些解释预测了参与者在听相同叙事时的fMRI记录。我们的发现表明，归因方法能够稳健地预测语言网络中的脑活动，在早期语言区域的表现超过了传统的内部表示。这种对齐具有层次性：早期层次的解释对应于大脑语言处理的初始阶段，而较晚的层次则与更高级的阶段相对应。此外，对LLMs下一个词语预测有较大影响的层次（即具有较高归因分数的层次）与神经活动的对齐更密切。这项工作建立了AI与神经科学之间的双向桥梁。首先，我们展示了归因方法作为一种强有力的探针，用于研究语言理解的神经机制，揭示了意义如何从背景中涌现。其次，我们提出使用脑活动对齐作为评估归因方法有效性的度量指标，从而提供了一个评估其生物合理性框架。 

---
# LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning 

**Title (ZH)**: LIFT：通过长输入微调提高大型语言模型的长上下文理解能力 

**Authors**: Yansheng Mao, Yufei Xu, Jiaqi Li, Fanxu Meng, Haotong Yang, Zilong Zheng, Xiyuan Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14644)  

**Abstract**: Long context understanding remains challenging for large language models due to their limited context windows. This paper presents Long Input Fine-Tuning (LIFT), a novel framework for long-context modeling that can improve the long-context performance of arbitrary (short-context) LLMs by dynamically adapting model parameters based on the long input. Importantly, LIFT, rather than endlessly extending the context window size to accommodate increasingly longer inputs in context, chooses to store and absorb the long input in parameter. By fine-tuning the long input into model parameters, LIFT allows short-context LLMs to answer questions even when the required information is not provided in the context during inference. Furthermore, to enhance LIFT performance while maintaining the original in-context learning (ICL) capabilities, we introduce Gated Memory, a specialized attention adapter that automatically balances long input memorization and ICL. We provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research. 

**Abstract (ZH)**: 长上下文理解仍然是大规模语言模型面临的挑战，因其受限于有限的上下文窗口。本文提出了Long Input Fine-Tuning（LIFT），这是一种新型框架，可以通过动态调整模型参数以适应长输入，从而提升任意短上下文语言模型（LLM）的长上下文性能。重要的是，LIFT 并没有无止境地扩大上下文窗口大小以适应越来越长的输入，而是选择将长输入存储在参数中。通过将长输入微调到模型参数中，LIFT 使短上下文语言模型在推断时即使所需的上下文信息未提供也能回答问题。此外，为了在保持原有上下文学习（ICL）能力的同时提升 LIFT 性能，我们引入了门控记忆（Gated Memory），这是一种专门的注意力适配器，能够自动平衡长输入记忆和 ICL。我们从多个方面对 LIFT 在长上下文理解中的优缺点进行了全面分析，为未来的研究提供了宝贵的方向。 

---
# How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation 

**Title (ZH)**: 大语言模型与数字孪生之间的差距：基于人设的行为链模拟基准研究 

**Authors**: Rui Li, Heming Xia, Xinfeng Yuan, Qingxiu Dong, Lei Sha, Wenjie Li, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2502.14642)  

**Abstract**: Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior. 

**Abstract (ZH)**: 近年来，大语言模型（LLMs）在学术界获得了广泛关注，因其作为人类数字双胞胎的潜力而受到重视。这些数字双胞胎是设计用于模仿个体，并自主完成诸如决策、问题解决和推理等任务的虚拟代理。然而，目前对LLMs的评估主要集中在对话模拟上，而忽视了对人类行为的模拟，这对于数字双胞胎至关重要。为弥补这一空白，我们提出了BehaviorChain，这是一个用于评估LLMs模拟连续人类行为能力的第一个基准。BehaviorChain包含多种多样的高质量行为链，共计涵盖1,001个独特的人格，这些人格之间的行为总数达到15,846种，每个性格都有详细的背景信息和个性资料。在评估过程中，我们将人格资料整合到LLMs中，并运用它们在由BehaviorChain提供的动态场景中逐步推断出上下文合适的行为。全面的评估结果表明，即使是当前最先进的模型，在准确模拟连续人类行为方面也存在困难。 

---
# Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs 

**Title (ZH)**: 大型语言模型能否预测引用意图？关于开放大型语言模型的上下文学习和微调的实验分析 

**Authors**: Paris Koloveas, Serafeim Chatzopoulos, Thanasis Vergoulis, Christos Tryfonopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.14561)  

**Abstract**: This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches that rely on pre-trained models like SciBERT, which require extensive domain-specific pretraining and specialized architectures, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero, one, few, and many-shot prompting to assess performance across scenarios. Our experimental study identifies the top-performing model through extensive experimentation of in-context learning-related parameters, which we fine-tune to further enhance task performance. The results highlight the strengths and limitations of LLMs in recognizing citation intents, providing valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use. 

**Abstract (ZH)**: 本研究探讨了开放式大规模语言模型（LLMs）通过上下文学习和微调来预测引文意图的能力。与依赖于像SciBERT这样的预训练模型的传统方法不同，后者需要大量的领域特定预训练和专门的架构，我们展示了通用语言模型可以借助极少的任务特定数据来适应这一任务。我们使用零样本、单样本、少量样本和多样本提示，评估了五个主要的开放式LLM家族中的十二种模型变体，以评估其在不同场景下的性能。通过广泛实验相关上下文学习参数，我们的研究确定了表现最佳的模型，并进一步对其进行微调以提高任务性能。研究结果突显了LLMs在识别引文意图方面的优势和局限性，为模型选择和提示工程提供了宝贵见解。此外，我们还公开提供了完整的评估框架和模型，以供未来使用。 

---
# LLM-based User Profile Management for Recommender System 

**Title (ZH)**: 基于LLM的用户画像管理在推荐系统中的应用 

**Authors**: Seunghwan Bang, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.14541)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展为推荐系统开辟了新的机会，使其能够在不依赖传统训练的情况下实现零样本推荐。尽管具有潜力，现有的大多数研究依旧主要依赖用户的购买历史，而未能充分利用用户生成的文本数据，如评论和产品描述，这为改进提供了很大的空间。为弥补这一空白，我们提出了PURE，一种新颖的基于LLM的推荐框架，通过系统地从用户评论中提取和总结关键信息来构建和发展演变中的用户画像。PURE由三个核心组件组成：评论提取器用于识别用户偏好和关键产品特征；用户画像更新器用于完善和更新用户画像；以及推荐器使用最新的画像生成个性化推荐。为了评估PURE，我们引入了一项连续的序贯推荐任务，通过随时间添加评论并在预测中逐步更新，以更好地反映现实场景。在亚马逊数据集上的实验结果表明，PURE 在利用长期用户信息的同时管理词汇量限制方面优于现有的基于LLM的方法。 

---
# How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM? 

**Title (ZH)**: 你可以在LoRA适配器中打包多少知识而不损害大语言模型的性能？ 

**Authors**: Sergey Pletenev, Maria Marina, Daniil Moskovskiy, Vasily Konovalov, Pavel Braslavski, Alexander Panchenko, Mikhail Salnikov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14502)  

**Abstract**: The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多任务上的表现受到预训练期间学习的知识以及存储在模型参数中的知识的限制。低秩适应（LoRA）是一种流行的高效训练技术，用于更新或特定领域适应LLMs。在本研究中，我们探讨了如何使用LoRA将新事实融入LLMs，而不损害先前学习的知识。我们使用不同的新知识量对Llama-3.1-8B-instruct进行微调。我们的实验表明，当训练数据包含已知和新知识的混合时，可以获得最佳结果。然而，这种做法仍然可能带来危害，因为在这种微调之后，模型在外部问答基准测试中的性能会下降。当训练数据偏向某些实体时，模型往往会回归到少数过代表的答案。此外，我们发现模型变得更有信心，并且只在少数情况下拒绝提供答案。这些发现强调了基于LoRA的LLM更新可能带来的潜在风险，并突显了训练数据组成和微调参数的重要性，以平衡新知识的整合和通用模型能力。 

---
# MLGym: A New Framework and Benchmark for Advancing AI Research Agents 

**Title (ZH)**: MLGym：一个促进人工智能研究代理发展的新框架与基准 

**Authors**: Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, Dieuwke Hupkes, Ricardo Silveira Cabral, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach, William Yang Wang, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14499)  

**Abstract**: We introduce Meta MLGym and MLGym-Bench, a new framework and benchmark for evaluating and developing LLM agents on AI research tasks. This is the first Gym environment for machine learning (ML) tasks, enabling research on reinforcement learning (RL) algorithms for training such agents. MLGym-bench consists of 13 diverse and open-ended AI research tasks from diverse domains such as computer vision, natural language processing, reinforcement learning, and game theory. Solving these tasks requires real-world AI research skills such as generating new ideas and hypotheses, creating and processing data, implementing ML methods, training models, running experiments, analyzing the results, and iterating through this process to improve on a given task. We evaluate a number of frontier large language models (LLMs) on our benchmarks such as Claude-3.5-Sonnet, Llama-3.1 405B, GPT-4o, o1-preview, and Gemini-1.5 Pro. Our MLGym framework makes it easy to add new tasks, integrate and evaluate models or agents, generate synthetic data at scale, as well as develop new learning algorithms for training agents on AI research tasks. We find that current frontier models can improve on the given baselines, usually by finding better hyperparameters, but do not generate novel hypotheses, algorithms, architectures, or substantial improvements. We open-source our framework and benchmark to facilitate future research in advancing the AI research capabilities of LLM agents. 

**Abstract (ZH)**: 我们介绍了Meta MLGym和MLGym-Bench，这是一个新的框架和基准，用于评估和开发在AI研究任务中工作的LLM代理。这是首个适用于机器学习（ML）任务的Gym环境，允许研究人员对训练此类代理的强化学习（RL）算法进行研究。MLGym-Bench包含来自计算机视觉、自然语言处理、强化学习和博弈论等多个领域的13项多样性和开放性AI研究任务。解决这些问题需要真实世界的AI研究技能，如产生新想法和假设、创建和处理数据、实现ML方法、训练模型、运行实验、分析结果以及在这一过程中的迭代以提高任务表现。我们在基准测试中评估了多个前沿大型语言模型（LLMs），例如Claude-3.5-Sonnet、Llama-3.1 405B、GPT-4o、o1-preview 和 Gemini-1.5 Pro。我们的MLGym框架使新增任务、集成和评估模型或代理、大规模生成合成数据以及为训练AI研究任务中的代理开发新的学习算法变得容易。我们发现，当前的前沿模型通常可以通过找到更好的超参数来改进给定的基本方法，但它们未能生成新的假设、算法、架构或实质性改进。我们开源了我们的框架和基准，以促进未来对提升LLM代理AI研究能力的研究。 

---
# Unshackling Context Length: An Efficient Selective Attention Approach through Query-Key Compression 

**Title (ZH)**: 解除上下文长度限制：一种通过查询-关键子压缩实现的高效选择性注意力方法 

**Authors**: Haoyu Wang, Tong Teng, Tianyu Guo, An Xiao, Duyu Tang, Hanting Chen, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14477)  

**Abstract**: Handling long-context sequences efficiently remains a significant challenge in large language models (LLMs). Existing methods for token selection in sequence extrapolation either employ a permanent eviction strategy or select tokens by chunk, which may lead to the loss of critical information. We propose Efficient Selective Attention (ESA), a novel approach that extends context length by efficiently selecting the most critical tokens at the token level to compute attention. ESA reduces the computational complexity of token selection by compressing query and key vectors into lower-dimensional representations. We evaluate ESA on long sequence benchmarks with maximum lengths up to 256k using open-source LLMs with context lengths of 8k and 32k. ESA outperforms other selective attention methods, especially in tasks requiring the retrieval of multiple pieces of information, achieving comparable performance to full-attention extrapolation methods across various tasks, with superior results in certain tasks. 

**Abstract (ZH)**: 高效处理长上下文序列仍然是大型语言模型（LLMs）中的一个重要挑战。现有序列外推的 token 选择方法要么采用永久驱逐策略，要么按块选择 token，这可能导致关键信息的丢失。我们提出了一种名为高效选择性注意（Efficient Selective Attention, ESA）的新方法，该方法在 token 级别通过高效选择最重要的 token 来计算注意机制，从而扩展上下文长度。ESA 通过压缩查询向量和键向量到低维度表示来减少 token 选择的计算复杂度。我们使用具有 8k 和 32k 上下文长度的开源 LLM，在最长长度达 256k 的长序列基准上评估了 ESA。在多种任务中，ESA 在检索多个信息片段的任务中表现优于其他选择性注意方法，且在各种任务中能达到全注意机制外推方法的可比性能，在某些任务中表现更优。 

---
# Enhancing Smart Environments with Context-Aware Chatbots using Large Language Models 

**Title (ZH)**: 使用大规模语言模型增强基于情境意识的智能环境中的聊天机器人 

**Authors**: Aurora Polo-Rodríguez, Laura Fiorini, Erika Rovini, Filippo Cavallo, Javier Medina-Quero  

**Link**: [PDF](https://arxiv.org/pdf/2502.14469)  

**Abstract**: This work presents a novel architecture for context-aware interactions within smart environments, leveraging Large Language Models (LLMs) to enhance user experiences. Our system integrates user location data obtained through UWB tags and sensor-equipped smart homes with real-time human activity recognition (HAR) to provide a comprehensive understanding of user context. This contextual information is then fed to an LLM-powered chatbot, enabling it to generate personalised interactions and recommendations based on the user's current activity and environment. This approach moves beyond traditional static chatbot interactions by dynamically adapting to the user's real-time situation. A case study conducted from a real-world dataset demonstrates the feasibility and effectiveness of our proposed architecture, showcasing its potential to create more intuitive and helpful interactions within smart homes. The results highlight the significant benefits of integrating LLM with real-time activity and location data to deliver personalised and contextually relevant user experiences. 

**Abstract (ZH)**: 本研究提出了一种新的架构，旨在增强智能环境中的上下文感知交互，通过利用大规模语言模型（LLMs）提升用户体验。我们的系统将通过超宽带（UWB）标签和传感器装备的智能家庭获得的用户位置数据与实时人体活动识别（HAR）集成，从而提供对用户上下文的全面理解。随后，该上下文信息被输送到一个基于LLMs的聊天机器人中，使其能够根据用户的当前活动和环境生成个性化的交互和建议。这种方法超越了传统的静态聊天机器人交互，能够根据用户的实时情况动态调整。通过对真实世界数据集进行的案例研究展示了我们提出的架构的实际可行性和有效性，展示了其在智能家庭中创建更加直观和有用交互的潜力。研究结果突显了将LLMs与实时活动和位置数据相结合以提供个性化和上下文相关的用户体验的巨大优势。 

---
# Optimal word order for non-causal text generation with Large Language Models: the Spanish case 

**Title (ZH)**: 使用大型语言模型进行非因果文本生成的最佳词序：以西班牙语为例 

**Authors**: Andrea Busto-Castiñeira, Silvia García-Méndez, Francisco de Arriba-Pérez, Francisco J. González-Castaño  

**Link**: [PDF](https://arxiv.org/pdf/2502.14451)  

**Abstract**: Natural Language Generation (NLG) popularity has increased owing to the progress in Large Language Models (LLMs), with zero-shot inference capabilities. However, most neural systems utilize decoder-only causal (unidirectional) transformer models, which are effective for English but may reduce the richness of languages with less strict word order, subject omission, or different relative clause attachment preferences. This is the first work that analytically addresses optimal text generation order for non-causal language models. We present a novel Viterbi algorithm-based methodology for maximum likelihood word order estimation. We analyze the non-causal most-likelihood order probability for NLG in Spanish and, then, the probability of generating the same phrases with Spanish causal NLG. This comparative analysis reveals that causal NLG prefers English-like SVO structures. We also analyze the relationship between optimal generation order and causal left-to-right generation order using Spearman's rank correlation. Our results demonstrate that the ideal order predicted by the maximum likelihood estimator is not closely related to the causal order and may be influenced by the syntactic structure of the target sentence. 

**Abstract (ZH)**: 自然语言生成（NLG）由于大型语言模型（LLMs）的进步而日益流行，这些模型具有零样本推理能力。然而，大多数神经系统利用的是解码器仅因果（单向）变换器模型，这种模型对于英语非常有效，但可能会在语序不那么严格、有主语省略或不同的名词性从句附加偏好的语言中降低语言的丰富性。这是首次通过分析性方法解决非因果语言模型的最佳文本生成顺序的工作。我们提出了一种基于维特比算法的新颖方法，用于估计最大似然词序。我们分析了西班牙语中非因果最大似然词序的概率，并随后分析了使用西班牙语因果NLG生成相同短语的概率。这种比较分析揭示了因果NLG倾向于偏好SVO结构。我们还使用斯皮尔曼秩相关分析了最佳生成顺序与因果自左向右生成顺序之间的关系。我们的结果表明，最大似然估计器预测的理想顺序与因果顺序关系不大，且可能受目标句子语法结构的影响。 

---
# PredictaBoard: Benchmarking LLM Score Predictability 

**Title (ZH)**: PredictaBoard: 评估大型语言模型得分可预测性基准测试 

**Authors**: Lorenzo Pacchiardi, Konstantinos Voudouris, Ben Slater, Fernando Martínez-Plumed, José Hernández-Orallo, Lexin Zhou, Wout Schellaert  

**Link**: [PDF](https://arxiv.org/pdf/2502.14445)  

**Abstract**: Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at this https URL 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）拥有令人印象深刻的技能，但它们往往会表现出不可预测的失误，在最基础的常识推理任务中也表现出不一致的成功率。这种不可预测性给确保其安全部署带来了重大挑战，因为识别并操作在可靠的“安全区”内至关重要，以减轻风险。为解决这一问题，我们提出了一种名为PredictaBoard的新型协作基准框架，用于评估评分预测器（称为评估者）的能力，预测LLMs在特定任务实例（即提示）上的错误。PredictaBoard通过考虑不同容忍误差水平下的拒识率来评估LLM评估者的对弈。因此，PredictaBoard促进了开发更好的评估者和使LMs更加可预测的研究，不仅提升平均性能。我们使用基线评估者和最先进的LMs进行了示范性实验。PredictaBoard强调了评估可预测性和性能的必要性，为一种新的安全人工智能系统铺平了道路，在这种系统中，错误不仅被最小化，还能被预见和有效缓解。我们的基准代码可以在以下链接找到：这个 https URL 

---
# Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models 

**Title (ZH)**: 面向大型语言模型真实性验证的词级密度基不确定性量化方法 

**Authors**: Artem Vazhentsev, Lyudmila Rvanova, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Artem Shelmanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14427)  

**Abstract**: Uncertainty quantification (UQ) is a prominent approach for eliciting truthful answers from large language models (LLMs). To date, information-based and consistency-based UQ have been the dominant UQ methods for text generation via LLMs. Density-based methods, despite being very effective for UQ in text classification with encoder-based models, have not been very successful with generative LLMs. In this work, we adapt Mahalanobis Distance (MD) - a well-established UQ technique in classification tasks - for text generation and introduce a new supervised UQ method. Our method extracts token embeddings from multiple layers of LLMs, computes MD scores for each token, and uses linear regression trained on these features to provide robust uncertainty scores. Through extensive experiments on eleven datasets, we demonstrate that our approach substantially improves over existing UQ methods, providing accurate and computationally efficient uncertainty scores for both sequence-level selective generation and claim-level fact-checking tasks. Our method also exhibits strong generalization to out-of-domain data, making it suitable for a wide range of LLM-based applications. 

**Abstract (ZH)**: 不确定性量化（UQ）是一种从大语言模型（LLM）中获取真实答案的重要方法。到目前为止，基于信息的方法和基于一致性的方法是主要用于通过LLM生成文本的UQ方法中的主导方法。尽管基于密度的方法在使用编码器模型的文本分类任务中表现非常有效，但在生成性LLM中并不非常成功。在本研究中，我们借鉴了分类任务中广泛应用于不确定性量化中的马哈拉诺比斯距离（MD），将其应用于文本生成，并引入了一种新的监督不确定性量化方法。该方法从LLM的多个层中提取标记嵌入，为每个标记计算MD得分，并使用基于这些特征训练的线性回归模型提供稳健的不确定性评分。通过在十一个数据集上的广泛实验，我们证明了我们的方法在现有UQ方法的基础上有了显著的提升，能够为序列级选择性生成和断言级事实检查任务提供准确且计算高效的不确定性评分。此外，我们的方法还表现出对领域外数据的强大泛化能力，使其适用于多种基于LLM的应用。 

---
# Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment 

**Title (ZH)**: 利用小型语言模型在教育领域进行论据挖掘：论据组件识别、分类与评估 

**Authors**: Lucile Favero, Juan Antonio Pérez-Ortiz, Tanja Käser, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2502.14389)  

**Abstract**: Argument mining algorithms analyze the argumentative structure of essays, making them a valuable tool for enhancing education by providing targeted feedback on the students' argumentation skills. While current methods often use encoder or encoder-decoder deep learning architectures, decoder-only models remain largely unexplored, offering a promising research direction.
This paper proposes leveraging open-source, small Large Language Models (LLMs) for argument mining through few-shot prompting and fine-tuning. These models' small size and open-source nature ensure accessibility, privacy, and computational efficiency, enabling schools and educators to adopt and deploy them locally. Specifically, we perform three tasks: segmentation of student essays into arguments, classification of the arguments by type, and assessment of their quality. We empirically evaluate the models on the Feedback Prize - Predicting Effective Arguments dataset of grade 6-12 students essays and demonstrate how fine-tuned small LLMs outperform baseline methods in segmenting the essays and determining the argument types while few-shot prompting yields comparable performance to that of the baselines in assessing quality. This work highlights the educational potential of small, open-source LLMs to provide real-time, personalized feedback, enhancing independent learning and writing skills while ensuring low computational cost and privacy. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

论文字义提取算法通过分析作文的论据结构，成为提高教育质量的重要工具，通过针对性地反馈学生的论据能力为学生提供帮助。当前的方法往往采用编码器或编码器-解码器深度学习架构，而完全解码模型尚未得到充分探索，为未来的研究提供了广阔的前景。

本文提出了利用开源的小型大语言模型（LLMs）进行论文字义提取的方法，通过少量示例的提示及微调实现。小型LLM的体积小、开源的特性确保了其可访问性、隐私保护和计算效率，使得学校和教育者能够将其部署在本地环境中。具体而言，本文执行了三项任务：将学生的作文拆分为论据、按类型对论据进行分类，并评估其质量。通过在6-12年级学生的作文数据集（Feedback Prize - Predicting Effective Arguments）上进行实证评估，我们发现微调的小型LLM在拆分作文和确定论据类型方面优于基线方法，而少量示例的提示在评估质量方面与基线方法具有可比性。本研究表明，小型开源LLM能够在实时、个性化反馈方面发挥教育潜力，促进自主学习和写作能力的发展，同时确保低计算成本和隐私保护。

---

这种翻译方式保持了原文的学术风格和结构，并将其转换成了中文，符合学术论文的写作规范。 

---
# Rumor Detection by Multi-task Suffix Learning based on Time-series Dual Sentiments 

**Title (ZH)**: 基于时间序列双情绪的多任务后缀学习谣言检测 

**Authors**: Zhiwei Liu, Kailai Yang, Eduard Hovy, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14383)  

**Abstract**: The widespread dissemination of rumors on social media has a significant impact on people's lives, potentially leading to public panic and fear. Rumors often evoke specific sentiments, resonating with readers and prompting sharing. To effectively detect and track rumors, it is essential to observe the fine-grained sentiments of both source and response message pairs as the rumor evolves over time. However, current rumor detection methods fail to account for this aspect. In this paper, we propose MSuf, the first multi-task suffix learning framework for rumor detection and tracking using time series dual (coupled) sentiments. MSuf includes three modules: (1) an LLM to extract sentiment intensity features and sort them chronologically; (2) a module that fuses the sorted sentiment features with their source text word embeddings to obtain an aligned embedding; (3) two hard prompts are combined with the aligned vector to perform rumor detection and sentiment analysis using one frozen LLM. MSuf effectively enhances the performance of LLMs for rumor detection with only minimal parameter fine-tuning. Evaluating MSuf on four rumor detection benchmarks, we find significant improvements compared to other emotion-based methods. 

**Abstract (ZH)**: 社交媒体上谣言的广泛传播对人们的生活产生了显著影响，可能导致公众恐慌和恐惧。谣言往往引发特定的情感，与读者产生共鸣，并促使他们分享。为了有效地检测和追踪谣言，在谣言演变过程中观察来源和响应消息对的情感细微变化至关重要。然而，当前的谣言检测方法未能考虑这一点。在这篇文章中，我们提出了MSuf，这是一种首次利用时间序列双（耦合）情感进行谣言检测和追踪的多任务后缀学习框架。MSuf包括三个模块：(1) 一个大型语言模型用于提取情感强度特征并按时间顺序排序；(2) 一个模块将排序后的情感特征与它们的源文本词嵌入结合，以获得对齐的嵌入；(3) 将两个硬提示与对齐的向量结合，使用一个冻结的大型语言模型进行谣言检测和情感分析。MSuf仅通过少量参数微调便有效提升了LLM在谣言检测中的性能。在四个谣言检测基准上评估MSuf，我们发现与基于情绪的方法相比取得了显著改进。 

---
# Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests 

**Title (ZH)**: 通过基准测试、游戏和认知测试 triangulate 大规模语言模型的进步 

**Authors**: Filippo Momentè, Alessandro Suglia, Mario Giulianelli, Ambra Ferrari, Alexander Koller, Oliver Lemon, David Schlangen, Raquel Fernández, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14359)  

**Abstract**: We examine three evaluation paradigms: large question-answering benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs. 

**Abstract (ZH)**: 我们考察了三种评估范式：大规模问答基准（例如MMLU和BBH）、互动游戏（例如信号博弈或禁忌词游戏）以及认知测试（例如工作记忆或理论思维测试）。首先，我们研究在区分不同质量的语言模型方面，哪种基准或游戏更为有效。接着，在受人类认知评估启发的基础上，我们编译了一系列针对性强的测验，这些测验衡量了有效语言使用所必需的认知能力，并探讨了这些测验与基准和游戏中模型性能的相关性。我们的分析表明，互动游戏在区分模型方面优于标准基准。因果推理和逻辑推理与静态和互动测验均相关，但在核心执行功能和社交/情感技能方面则呈现出不同，后者与游戏表现的相关性更强。我们建议开发新的互动基准和特定设计用于语言模型的认知任务，这些任务灵感源于评估人类能力。 

---
# SR-LLM: Rethinking the Structured Representation in Large Language Model 

**Title (ZH)**: SR-LLM: 重新审视大型语言模型中的结构化表示 

**Authors**: Jiahuan Zhang, Tianheng Wang, Hanqing Wu, Ziyi Huang, Yulong Wu, Dongbai Chen, Linfeng Song, Yue Zhang, Guozheng Rao, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14352)  

**Abstract**: Structured representations, exemplified by Abstract Meaning Representation (AMR), have long been pivotal in computational linguistics. However, their role remains ambiguous in the Large Language Models (LLMs) era. Initial attempts to integrate structured representation into LLMs via a zero-shot setting yielded inferior performance. We hypothesize that such a decline stems from the structure information being passed into LLMs in a code format unfamiliar to LLMs' training corpora. Consequently, we propose SR-LLM, an innovative framework with two settings to explore a superior way of integrating structured representation with LLMs from training-free and training-dependent perspectives. The former integrates structural information through natural language descriptions in LLM prompts, whereas its counterpart augments the model's inference capability through fine-tuning on linguistically described structured representations. Performance improvements were observed in widely downstream datasets, with particularly notable gains of 3.17% and 12.38% in PAWS. To the best of our knowledge, this work represents the pioneering demonstration that leveraging structural representations can substantially enhance LLMs' inference capability. We hope that our work sheds light and encourages future research to enhance the reasoning and interoperability of LLMs by structure data. 

**Abstract (ZH)**: 结构化表示，以抽象语义表示（AMR）为例，在计算语言学中一直起着关键作用。然而，在大型语言模型（LLMs）时代，它们的作用仍不明确。早期尝试通过零样本设置将结构化表示整合到LLMs中，性能表现较差。我们假设这种下降是由结构信息以LLMs的训练语料库不熟悉的编码格式传递引起的。因此，我们提出SR-LLM，这是一种创新框架，包含两种设置，旨在从无训练和依赖训练的角度探索将结构化表示与LLMs整合的更优方式。前者通过LLMs提示中的自然语言描述来整合结构信息，而后者则通过在语言描述的结构化表示上进行微调来增强模型的推理能力。在广泛的应用下游数据集上观察到了性能改进，特别是在PAWS数据集上的改进尤为显著，达到3.17%和12.38%。据我们所知，这项工作展示了利用结构化表示可以显著提升LLMs的推理能力的首次示例。我们希望我们的工作能为未来研究提供启示，通过结构化数据提高LLMs的推理能力和互操性。 

---
# A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics 

**Title (ZH)**: 基于反馈的多步推理综述：大规模语言模型在数学中的应用 

**Authors**: Ting-Ruen Wei, Haowei Liu, Xuyang Wu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14333)  

**Abstract**: Recent progress in large language models (LLM) found chain-of-thought prompting strategies to improve the reasoning ability of LLMs by encouraging problem solving through multiple steps. Therefore, subsequent research aimed to integrate the multi-step reasoning process into the LLM itself through process rewards as feedback and achieved improvements over prompting strategies. Due to the cost of step-level annotation, some turn to outcome rewards as feedback. Aside from these training-based approaches, training-free techniques leverage frozen LLMs or external tools for feedback at each step to enhance the reasoning process. With the abundance of work in mathematics due to its logical nature, we present a survey of strategies utilizing feedback at the step and outcome levels to enhance multi-step math reasoning for LLMs. As multi-step reasoning emerges a crucial component in scaling LLMs, we hope to establish its foundation for easier understanding and empower further research. 

**Abstract (ZH)**: 近年来，在大型语言模型（LLM）领域取得了进展，发现通过链式思考提示策略可以提升LLM的推理能力，具体做法是通过多步推理鼓励问题解决。因此，后续研究致力于将多步推理过程集成到LLM本身中，并通过过程奖励作为反馈来改进提示策略。由于步骤级标注成本较高，一些研究转向使用结果奖励作为反馈。除了基于训练的方法之外，还有一些无需训练的技术利用冻结的LLM或外部工具在每个步骤提供反馈以增强推理过程。鉴于数学因其逻辑性质而工作量丰富，我们对利用步骤级和结果级反馈以增强LLM的多步数学推理策略进行了综述。鉴于多步推理在扩展LLM中的重要性，我们希望为更易于理解并促进进一步研究奠定基础。 

---
# EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts 

**Title (ZH)**: EpMAN： episodic 记忆注意力机制用于泛化到更长的上下文 

**Authors**: Subhajit Chaudhury, Payel Das, Sarathkrishna Swaminathan, Georgios Kollias, Elliot Nelson, Khushbu Pahwa, Tejaswini Pedapati, Igor Melnyk, Matthew Riemer  

**Link**: [PDF](https://arxiv.org/pdf/2502.14280)  

**Abstract**: Recent advances in Large Language Models (LLMs) have yielded impressive successes on many language tasks. However, efficient processing of long contexts using LLMs remains a significant challenge. We introduce \textbf{EpMAN} -- a method for processing long contexts in an \textit{episodic memory} module while \textit{holistically attending to} semantically relevant context chunks. The output of \textit{episodic attention} is then used to reweigh the decoder's self-attention to the stored KV cache of the context during training and generation. When an LLM decoder is trained using \textbf{EpMAN}, its performance on multiple challenging single-hop long-context recall and question-answering benchmarks is found to be stronger and more robust across the range from 16k to 256k tokens than baseline decoders trained with self-attention, and popular retrieval-augmented generation frameworks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在许多语言任务上取得了令人印象深刻的成功。然而，高效处理长上下文仍然是一个重大挑战。我们提出了**EpMAN** —— 一种在**情景记忆**模块中处理长上下文的方法，同时**整体关注**语义相关的上下文片段。情景注意的输出随后用于在训练和生成过程中调整解码器的自注意权重，使其指向存储在上下文中的KV缓存。当使用**EpMAN** 训练LLM解码器时，其在多个具有挑战性的单跳长上下文回忆和问答基准测试中的表现相比于仅使用自注意训练和流行检索增强生成框架的基线解码器表现出更强且更加稳健，覆盖了从16,000到256,000个标记的范围。 

---
# Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgment 

**Title (ZH)**: 事实还是推测？基于结构化单跳判断评估大型语言模型的医学知识 

**Authors**: Jiaxi Li, Yiwei Wang, Kai Zhang, Yujun Cai, Bryan Hooi, Nanyun Peng, Kai-Wei Chang, Jin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14275)  

**Abstract**: Large language models (LLMs) have been widely adopted in various downstream task domains. However, their ability to directly recall and apply factual medical knowledge remains under-explored. Most existing medical QA benchmarks assess complex reasoning or multi-hop inference, making it difficult to isolate LLMs' inherent medical knowledge from their reasoning capabilities. Given the high-stakes nature of medical applications, where incorrect information can have critical consequences, it is essential to evaluate how well LLMs encode, retain, and recall fundamental medical facts.
To bridge this gap, we introduce the Medical Knowledge Judgment, a dataset specifically designed to measure LLMs' one-hop factual medical knowledge. MKJ is constructed from the Unified Medical Language System (UMLS), a large-scale repository of standardized biomedical vocabularies and knowledge graphs. We frame knowledge assessment as a binary judgment task, requiring LLMs to verify the correctness of medical statements extracted from reliable and structured knowledge sources.
Our experiments reveal that LLMs struggle with factual medical knowledge retention, exhibiting significant performance variance across different semantic categories, particularly for rare medical conditions. Furthermore, LLMs show poor calibration, often being overconfident in incorrect answers. To mitigate these issues, we explore retrieval-augmented generation, demonstrating its effectiveness in improving factual accuracy and reducing uncertainty in medical decision-making. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在各种下游任务领域得到广泛应用。然而，它们直接回忆和应用医学事实知识的能力仍较少被探索。现有的大多数医学问答基准评估的是复杂推理或多跳推理能力，这使得难以将LLMs固有的医学知识与其推理能力区分开来。鉴于医学应用的高度风险性，错误信息可能会导致严重后果，因此评估LLMs编码、存储和回忆基本医学事实的能力至关重要。

为解决这一问题，我们引入了《医学知识判断》（Medical Knowledge Judgment，MKJ）数据集，专门用于测量LLMs的一跳式医学事实知识。MKJ数据集是从统一医学语言系统（UMLS）构建的，UMLS是一个大规模的标准化生物医学词汇库和知识图谱的仓库。我们将知识评估任务设定为二分类判断任务，要求LLMs验证从可靠的结构化知识源中提取的医学声明的正确性。

我们的实验表明，LLMs在医学事实知识的存储方面存在困难，不同语义类别间的性能差异显著，尤其是对于罕见医学状况的表现尤为突出。此外，LLMs的校准效果较差，常常对自己的错误答案过于自信。为了缓解这些问题，我们探索了检索增强生成方法，并证明了这种方法在提高事实准确性、减少医学决策中的不确定性方面的有效性。 

---
# PaperHelper: Knowledge-Based LLM QA Paper Reading Assistant 

**Title (ZH)**: PaperHelper：基于知识的LLM问答式论文阅读辅助系统 

**Authors**: Congrui Yin, Evan Wei, Zhongxing Zhang, Zaifu Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14271)  

**Abstract**: In the paper, we introduce a paper reading assistant, PaperHelper, a potent tool designed to enhance the capabilities of researchers in efficiently browsing and understanding scientific literature. Utilizing the Retrieval-Augmented Generation (RAG) framework, PaperHelper effectively minimizes hallucinations commonly encountered in large language models (LLMs), optimizing the extraction of accurate, high-quality knowledge. The implementation of advanced technologies such as RAFT and RAG Fusion significantly boosts the performance, accuracy, and reliability of the LLMs-based literature review process. Additionally, PaperHelper features a user-friendly interface that facilitates the batch downloading of documents and uses the Mermaid format to illustrate structural relationships between documents. Experimental results demonstrate that PaperHelper, based on a fine-tuned GPT-4 API, achieves an F1 Score of 60.04, with a latency of only 5.8 seconds, outperforming the basic RAG model by 7\% in F1 Score. 

**Abstract (ZH)**: 在本文中，我们引入了一种论文阅读助手——PaperHelper，这是一种强大的工具，旨在增强研究人员高效浏览和理解科学文献的能力。通过利用检索增强生成（RAG）框架，PaperHelper有效减少了大型语言模型（LLMs）中常见的幻觉现象，优化了高质量、高准确性的知识提取。先进的技术如RAFT和RAG融合的实施显著提升了基于LLMs的文献回顾过程的性能、准确性和可靠性。此外，PaperHelper还具备用户友好的界面，便于批量下载文档，并使用Mermaid格式展示文档之间的结构关系。实验结果表明，基于微调的GPT-4 API，PaperHelper实现了F1分数60.04，响应延迟仅为5.8秒，相比基本的RAG模型在F1分数上提高了7%。 

---
# Transfer-Prompting: Enhancing Cross-Task Adaptation in Large Language Models via Dual-Stage Prompts Optimization 

**Title (ZH)**: 迁移提示：通过双重阶段提示优化增强大型语言模型的跨任务适应性 

**Authors**: Yupeng Chang, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14211)  

**Abstract**: Large language models (LLMs) face significant challenges when balancing multiple high-level objectives, such as generating coherent, relevant, and high-quality responses while maintaining efficient task adaptation across diverse tasks. To address these challenges, we introduce Transfer-Prompting, a novel two-stage framework designed to enhance cross-task adaptation in prompt generation. The framework comprises two key components: (1) source prompt construction, which refines the original prompts on source task datasets to generate source prompts with enhanced generalization ability, and (2) target prompt generation, which enhances cross-task adaptation of target prompts by fine-tuning a set of high-scored source prompts on task-specific datasets. In each optimization cycle, a reference LLM generates candidate prompts based on historical prompt-score pairs and task descriptions in our designed reference prompt. These candidate prompts are refined iteratively, while a scorer LLM evaluates their effectiveness using the multi-dimensional metrics designed in the objective prompts evaluator-a novel contribution in this work that provides a holistic evaluation of prompt quality and task performance. This feedback loop facilitates continuous refinement, optimizing both prompt quality and task-specific outcomes. We validate Transfer-Prompting through extensive experiments across 25 LLMs, including 7 foundational models and 18 specialized models, evaluated on 9 diverse datasets. The results demonstrate that Transfer-Prompting significantly improves task-specific performance, highlighting its potential for enhancing cross-task adaptation in LLMs. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在平衡多个高级目标时面临重大挑战，例如生成连贯、相关且高质量的响应，同时保持高效的任务适应性，应用于多样化任务。为应对这些挑战，我们提出了转移提示（Transfer-Prompting），这是一种新颖的两阶段框架，旨在增强提示生成中的跨任务适应性。该框架包含两个关键组件：（1）源提示构建，即将原始提示在源任务数据集上进行细化，生成具有增强泛化能力的源提示；（2）目标提示生成，通过在任务特定数据集上微调得分较高的源提示，增强目标提示的跨任务适应性。在每次优化循环中，参考LLM基于我们设计的参考提示中的历史提示-分数对和任务描述生成候选提示。这些候选提示在迭代过程中不断改进，而评分LLM则使用目标提示评估者（本工作中的一项新贡献）设计的多维度指标评估其有效性，该评估者提供了对提示质量及任务性能的全面评价。这种反馈循环促进了持续的优化，既优化了提示质量又增强了任务特定效果。我们通过涵盖25个LLM（包括7个基础模型和18个专门模型）的广泛实验，评估了Transfer-Prompting在9个不同数据集上的表现。结果表明，Transfer-Prompting显著提高了任务特定性能，突显了其在提升LLM的跨任务适应性方面的潜力。代码可在以下网址获取：[在这里插入网址]。 

---
# NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM 

**Title (ZH)**: NLP-AKG：基于大语言模型的少量示例构建NLP学术知识图谱 

**Authors**: Jiayin Lan, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14192)  

**Abstract**: Large language models (LLMs) have been widely applied in question answering over scientific research papers. To enhance the professionalism and accuracy of responses, many studies employ external knowledge augmentation. However, existing structures of external knowledge in scientific literature often focus solely on either paper entities or domain concepts, neglecting the intrinsic connections between papers through shared domain concepts. This results in less comprehensive and specific answers when addressing questions that combine papers and concepts. To address this, we propose a novel knowledge graph framework that captures deep conceptual relations between academic papers, constructing a relational network via intra-paper semantic elements and inter-paper citation relations. Using a few-shot knowledge graph construction method based on LLM, we develop NLP-AKG, an academic knowledge graph for the NLP domain, by extracting 620,353 entities and 2,271,584 relations from 60,826 papers in ACL Anthology. Based on this, we propose a 'sub-graph community summary' method and validate its effectiveness on three NLP scientific literature question answering datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科研论文的问答任务中得到了广泛应用。为了提升回应的专业性和准确性，许多研究采用了外部知识增强的方法。然而，现有科学文献中的外部知识结构往往仅专注于论文实体或领域概念，忽略了通过共享领域概念论文之间的内在联系。这种做法导致在回答结合论文和概念的问题时，答案不够全面和具体。为解决这一问题，我们提出了一种新的知识图谱框架，能够捕获学术论文之间深层次的概念关系，通过构建基于论文内语义元素和论文间引用关系的交互网络来进行这一过程。我们基于大模型（LLM）采用少量示例的知识图谱构建方法，构建了NLP-AKG——一个用于NLP领域的学术知识图谱，从中提取了来自ACL Anthology的60,826篇论文中的620,353个实体和2,271,584个关系。在此基础上，我们提出了一种“子图社区摘要”方法，并在三个NLP科学文献问答数据集上验证了该方法的有效性。 

---
# QUAD-LLM-MLTC: Large Language Models Ensemble Learning for Healthcare Text Multi-Label Classification 

**Title (ZH)**: QUAD-LLM-MLTC：医疗文本多标签分类的大语言模型集成学习 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14189)  

**Abstract**: The escalating volume of collected healthcare textual data presents a unique challenge for automated Multi-Label Text Classification (MLTC), which is primarily due to the scarcity of annotated texts for training and their nuanced nature. Traditional machine learning models often fail to fully capture the array of expressed topics. However, Large Language Models (LLMs) have demonstrated remarkable effectiveness across numerous Natural Language Processing (NLP) tasks in various domains, which show impressive computational efficiency and suitability for unsupervised learning through prompt engineering. Consequently, these LLMs promise an effective MLTC of medical narratives. However, when dealing with various labels, different prompts can be relevant depending on the topic. To address these challenges, the proposed approach, QUAD-LLM-MLTC, leverages the strengths of four LLMs: GPT-4o, BERT, PEGASUS, and BART. QUAD-LLM-MLTC operates in a sequential pipeline in which BERT extracts key tokens, PEGASUS augments textual data, GPT-4o classifies, and BART provides topics' assignment probabilities, which results in four classifications, all in a 0-shot setting. The outputs are then combined using ensemble learning and processed through a meta-classifier to produce the final MLTC result. The approach is evaluated using three samples of annotated texts, which contrast it with traditional and single-model methods. The results show significant improvements across the majority of the topics in the classification's F1 score and consistency (F1 and Micro-F1 scores of 78.17% and 80.16% with standard deviations of 0.025 and 0.011, respectively). This research advances MLTC using LLMs and provides an efficient and scalable solution to rapidly categorize healthcare-related text data without further training. 

**Abstract (ZH)**: 不断增长的医疗文本数据量为自动多标签文本分类（MLTC）带来了独特挑战，主要原因是对训练数据的注释稀少及其复杂的性质。传统机器学习模型往往难以充分捕捉到表达的各种主题。然而，大型语言模型（LLMs）在多个自然语言处理（NLP）任务中显示出巨大的效果，尤其是在不同领域，它们展示了出色的计算效率和通过提示工程进行无监督学习的适用性。因此，这些LLMs有望有效地实现医疗叙事的MLTC。然而，在处理各种标签时，不同的话题可能需要不同的提示。为应对这些挑战，我们提出了一种名为QUAD-LLM-MLTC的方法，充分利用了四个LLM的优势：GPT-4o、BERT、PEGASUS和BART。QUAD-LLM-MLTC在顺序处理管道中运作，首先由BERT提取关键标记，然后由PEGASUS扩充文本数据，GPT-4o进行分类，BART提供主题的概率分配，从而产生四个分类结果，均在零样本设置下完成。最终的MLTC结果通过集成学习和元分类器生成。该方法使用三个标注数据样本进行评估，与传统方法和单模型方法进行对比。结果显示，在分类F1分数和一致性方面，多数话题均有显著改进（F1和微观F1分数分别为78.17%和80.16%，标准差分别为0.025和0.011）。这项研究通过LLMs推进了MLTC的方法，并提供了一种高效可扩展的解决方案，可以在不进一步训练的情况下快速对相关医疗文本数据进行分类。 

---
# Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction 

**Title (ZH)**: 增强对话代理的心智理论：使信念、欲望和意图对齐以实现类似人类的互动 

**Authors**: Mohammadmahdi Jafari, Devin Yuncheng Hua, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2502.14171)  

**Abstract**: Natural language interaction with agentic Artificial Intelligence (AI), driven by Large Language Models (LLMs), is expected to remain a dominant paradigm in the near future. While humans instinctively align their communication with mental states -- an ability known as Theory of Mind (ToM), current LLM powered systems exhibit significant limitations in this regard. This study examines the extent to which open source language models (LLaMA) can capture and preserve ToM related information and how effectively it contributes to consistent ToM reasoning in generated responses. We further investigate whether explicit manipulation of ToM related components, such as beliefs, desires, and intentions, can enhance response alignment. Experiments on two LLaMA 3 variants demonstrate that incorporating ToM informed alignment improves response quality, achieving win rates of 67 and 63 percent for the 3B and 8B models, respectively. These findings highlight the potential of ToM driven strategies to improve alignment in LLM based conversational agents. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的代理人工智能（AGI）自然语言交互，预计在未来一段时间内仍将是主导范式。尽管人类本能地根据心理状态调整自己的交流方式（这种能力被称为心智理论，即Theory of Mind, ToM），但当前的LLM驱动系统在这方面表现出明显的局限性。本研究考察了开源语言模型（例如LaMA）能否捕捉和保留与ToM相关的信息，以及这些信息如何有效促进生成响应中的持续性ToM推理。此外，我们还研究了显式操纵与ToM相关的成分（如信念、欲望和意图）是否能够提升响应的对齐程度。在两种LaMA 3变体的实验中，将ToM导向的对齐机制纳入模型显著提高了响应质量，3B模型和8B模型的赢率分别为67%和63%。这些发现突显了ToM驱动策略在提高基于LLM的对话代理对齐程度方面的潜力。 

---
# LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems 

**Title (ZH)**: 基于LLM增强的全双工语音对话系统对话管理 

**Authors**: Hao Zhang, Weiwei Li, Rilin Chen, Vinay Kothapally, Meng Yu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14145)  

**Abstract**: Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS. 

**Abstract (ZH)**: 在语音对话系统（Spoken Dialogue Systems, SDS）中实现全双工通信需要实时协调倾听、说话和思考之间的关系。本文提出了一种语义语音活动检测（VAD）模块，作为对话管理器（Dialogue Manager, DM），以高效管理全双工SDS中的轮换。该语义VAD模块采用轻量级（约0.5B）语言模型（LLM），并针对全双工对话数据进行了微调，能够预测四个控制标记以调节轮换和保持发言，区分有意和无意的插话，同时检测查询完成，以应对用户的暂停和犹豫。通过以短时间片处理输入语音，语义VAD能够实现实时决策，而核心对话引擎（Core Dialogue Engine, CDE）仅在生成响应时被激活，从而减少了计算负担。这种设计允许DM独立优化，而无需重新训练CDE，同时平衡交互准确性和推理效率，以实现可扩展的下一代全双工SDS。 

---
# Self-Regularization with Latent Space Explanations for Controllable LLM-based Classification 

**Title (ZH)**: 基于潜在空间解释的自我正则化控制性大型语言模型分类 

**Authors**: Xuansheng Wu, Wenhao Yu, Xiaoming Zhai, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14133)  

**Abstract**: Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impacts of these unintended features toward classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed framework can significantly improve the classifier's generalizability by regularizing those features that are not semantically correlated to each task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. We will release our code and data once accepted. 

**Abstract (ZH)**: 现代文本分类方法强烈依赖于大语言模型（LLM）的上下文嵌入。与人工设计的特征相比，这些嵌入能够自动且有效地为分类模型的训练提供表示形式。然而，这同时也带来了一个挑战：我们失去了手动移除未预期特征的能力，例如敏感信息或与任务无关的特征，以确保符合监管要求或提高分类模型的一般性。这种局限性源于LLM嵌入的不透明性和难以解释性。在本文中，我们提出了一种新的框架来识别和规范LLM潜在空间中的未预期特征。具体而言，我们首先预训练了一个稀疏自编码器（SAE），从LLM潜在空间中提取可解释的特征。为了确保SAE能够捕捉到任务相关的特征，我们进一步针对特定任务的数据集对其进行微调。在训练分类模型时，我们提出了一种简单而有效的正则化方法，通过最小化分类器权重与识别出的未预期特征之间的相似性，来移除这些未预期特征对分类的影响。我们使用三个实际任务对提出的框架进行了评估，包括有毒聊天检测、奖励建模和疾病诊断。结果表明，该框架通过规范化与任务语义无关的特征，能显著提高分类器的一般性。这项工作通过利用可解释的特征来解决LLM潜在空间中的可控制文本分类问题中的泛化能力、公平性和隐私问题，取得了先驱性进展。论文被接受后，我们将发布我们的代码和数据。 

---
# Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above 

**Title (ZH)**: 以下哪个选项最能描述使用大规模语言模型（LLM）进行的选择题评估？A）被迫的 B）有缺陷的 C）可修复的 D）以上全部 

**Authors**: Nishant Balepur, Rachel Rudinger, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.14127)  

**Abstract**: Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing-where LLMs construct and explain answers-better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA-robustness, biases, and unfaithful explanations-showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations. 

**Abstract (ZH)**: 多项选择题回答（MCQA）因其简洁性和接近人类的测试方式而常被用于大语言模型（LLM）评估，但我们认为需要改革这一形式。我们首先揭示了MCQA格式中的缺陷，因为它难以：1) 测试生成能力和主观性；2) 符合LLM的应用场景；3) 完整测试知识。因此，我们主张采用基于人工测试的生成性格式，其中LLM构建并解释答案，更能捕捉用户的需求和知识，并且易于评分。接着，我们展示了即使在MCQA是一种有用的格式时，其数据集也存在泄露、不可回答性、捷径和饱和等问题。在每个问题上，我们提供了源自教育领域的解决方案，比如评估量表来指导MCQ写作；评分方法来限制猜测；以及项目反应理论来构建更难的MCQ。最后，我们讨论了MCQA中LLM的错误，包括鲁棒性、偏见和不忠实的解释，展示了我们之前提出的方法如何更好地衡量或解决这些问题。虽然我们不必放弃MCQA，但我们鼓励在基于教育测试的原则下进一步完善这一任务，推动评估的进步。 

---
# Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach 

**Title (ZH)**: 面向上下文鲁棒性的大语言模型：一种门控表示微调方法 

**Authors**: Shenglai Zeng, Pengfei He, Kai Guo, Tianqi Zheng, Hanqing Lu, Yue Xing, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14100)  

**Abstract**: Large Language Models (LLMs) enhanced with external contexts, such as through retrieval-augmented generation (RAG), often face challenges in handling imperfect evidence. They tend to over-rely on external knowledge, making them vulnerable to misleading and unhelpful contexts. To address this, we propose the concept of context-robust LLMs, which can effectively balance internal knowledge with external context, similar to human cognitive processes. Specifically, context-robust LLMs should rely on external context only when lacking internal knowledge, identify contradictions between internal and external knowledge, and disregard unhelpful contexts. To achieve this goal, we introduce Grft, a lightweight and plug-and-play gated representation fine-tuning approach. Grft consists of two key components: a gating mechanism to detect and filter problematic inputs, and low-rank representation adapters to adjust hidden representations. By training a lightweight intervention function with only 0.0004\% of model size on fewer than 200 examples, Grft can effectively adapt LLMs towards context-robust behaviors. 

**Abstract (ZH)**: 增强外部上下文的大型语言模型（LLMs），如通过检索增强生成（RAG）技术，往往在处理不完美的证据时面临挑战。这类模型倾向于过度依赖外部知识，使其容易受到误导性和无用上下文的影响。为解决这一问题，我们提出了上下文鲁棒大型语言模型的概念，这种模型能够有效地平衡内部知识与外部上下文，类似于人类认知过程。具体来说，上下文鲁棒大型语言模型应该仅在缺乏内部知识时依赖外部上下文，识别内部知识与外部知识之间的矛盾，并忽略无用的上下文。

为了实现这一目标，我们引入了Grft，一种轻量级且即插即用的门控表示微调方法。Grft 包括两个关键组件：一种门控机制用于检测和过滤问题输入，以及低秩表示适配器用于调整隐藏表示。通过在少于200个示例上训练仅占模型大小0.0004%的轻量级干预函数，Grft 能够有效使大型语言模型朝向上下文鲁棒的行为转变。 

---
# Navigating Semantic Relations: Challenges for Language Models in Abstract Common-Sense Reasoning 

**Title (ZH)**: 导航语义关系：语言模型在抽象常识推理中的挑战 

**Authors**: Cole Gawin, Yidan Sun, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.14086)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance in generating human-like text and solving reasoning tasks of moderate complexity, such as question-answering and mathematical problem-solving. However, their capabilities in tasks requiring deeper cognitive skills, such as common-sense understanding and abstract reasoning, remain under-explored. In this paper, we systematically evaluate abstract common-sense reasoning in LLMs using the ConceptNet knowledge graph. We propose two prompting approaches: instruct prompting, where models predict plausible semantic relationships based on provided definitions, and few-shot prompting, where models identify relations using examples as guidance. Our experiments with the gpt-4o-mini model show that in instruct prompting, consistent performance is obtained when ranking multiple relations but with substantial decline when the model is restricted to predicting only one relation. In few-shot prompting, the model's accuracy improves significantly when selecting from five relations rather than the full set, although with notable bias toward certain relations. These results suggest significant gaps still, even in commercially used LLMs' abstract common-sense reasoning abilities, compared to human-level understanding. However, the findings also highlight the promise of careful prompt engineering, based on selective retrieval, for obtaining better performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成类人类文本和解决中等复杂度的推理任务（如问答和数学问题求解）方面取得了显著成果。然而，它们在需要更深层次认知能力的任务（如常识理解和抽象推理）中的能力尚待探索。本文系统地使用ConceptNet知识图谱评估LLMs在抽象常识推理方面的表现。我们提出了两种提示方法：指令提示，模型根据提供的定义预测合乎逻辑的语义关系；以及少样本提示，模型在示例的指导下识别关系。我们的实验使用gpt-4o-mini模型显示，在指令提示中，当对多种关系进行排序时，可以获得一致的表现，但当模型被限制只能预测单一关系时，表现会显著下降。在少样本提示中，当从五个关系中选择时，模型的准确性显著提高，尽管存在明显的偏好某些关系的现象。这些结果表明，即使是商用的大规模语言模型在抽象常识推理方面的能力与人类级别的理解之间仍存在显著差距。然而，研究结果也突显了通过精心设计提示工程（基于选择性检索）来获得更好性能的潜力。 

---
# Prompt-to-Leaderboard 

**Title (ZH)**: "Prompt-to-Leaderboard" 可以翻译为学术规范的中文标题为：“从提示到排行榜”。这个标题通常用于描述一种方法或系统，能够根据给定的提示生成或评估模型的表现，并将其结果提交到排行榜上进行比较和展示。 

**Authors**: Evan Frick, Connor Chen, Joseph Tennyson, Tianle Li, Wei-Lin Chiang, Anastasios N. Angelopoulos, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.14855)  

**Abstract**: Large language model (LLM) evaluations typically rely on aggregated metrics like accuracy or human preference, averaging across users and prompts. This averaging obscures user- and prompt-specific variations in model performance. To address this, we propose Prompt-to-Leaderboard (P2L), a method that produces leaderboards specific to a prompt. The core idea is to train an LLM taking natural language prompts as input to output a vector of Bradley-Terry coefficients which are then used to predict the human preference vote. The resulting prompt-dependent leaderboards allow for unsupervised task-specific evaluation, optimal routing of queries to models, personalization, and automated evaluation of model strengths and weaknesses. Data from Chatbot Arena suggest that P2L better captures the nuanced landscape of language model performance than the averaged leaderboard. Furthermore, our findings suggest that P2L's ability to produce prompt-specific evaluations follows a power law scaling similar to that observed in LLMs themselves. In January 2025, the router we trained based on this methodology achieved the \#1 spot in the Chatbot Arena leaderboard. Our code is available at this GitHub link: this https URL. 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

大型语言模型（LLM）的评估通常依赖于聚合指标，如准确率或人类偏好，这些指标是通过对多个用户和提示进行平均计算得出的。这种平均计算掩盖了不同用户和提示下模型性能的具体差异。为了解决这一问题，我们提出了Prompt-to-Leaderboard（P2L）方法，该方法能够生成针对特定提示的排行榜。核心思路是训练一个LLM，使其以自然语言提示作为输入，输出布拉德利-特里系数向量，然后使用这些系数来预测人类偏好投票。由此产生的提示依赖式排行榜使得无需监督即可进行任务特定评估、优化模型查询路由、个性化以及自动化评估模型的优势和劣势。Chatbot Arena的数据表明，P2L更能捕捉语言模型性能的细微差异，其排行榜表现优于平均排行榜。此外，我们的研究发现表明，P2L生成提示特定评估的能力与LLM本身的幂律缩放类似。在2025年1月，我们基于该方法训练的路由器在Chatbot Arena排行榜上取得了第一名的成绩。我们的代码可在以下GitHub链接访问：this https URL。 

---
# Optimizing Model Selection for Compound AI Systems 

**Title (ZH)**: 优化复合人工智能系统中的模型选择方法 

**Authors**: Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Matei Zaharia, James Zou, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.14815)  

**Abstract**: Compound AI systems that combine multiple LLM calls, such as self-refine and multi-agent-debate, achieve strong performance on many AI tasks. We address a core question in optimizing compound systems: for each LLM call or module in the system, how should one decide which LLM to use? We show that these LLM choices have a large effect on quality, but the search space is exponential. We propose LLMSelector, an efficient framework for model selection in compound systems, which leverages two key empirical insights: (i) end-to-end performance is often monotonic in how well each module performs, with all other modules held fixed, and (ii) per-module performance can be estimated accurately by an LLM. Building upon these insights, LLMSelector iteratively selects one module and allocates to it the model with the highest module-wise performance, as estimated by an LLM, until no further gain is possible. LLMSelector is applicable to any compound system with a bounded number of modules, and its number of API calls scales linearly with the number of modules, achieving high-quality model allocation both empirically and theoretically. Experiments with popular compound systems such as multi-agent debate and self-refine using LLMs such as GPT-4o, Claude 3.5 Sonnet and Gemini 1.5 show that LLMSelector confers 5%-70% accuracy gains compared to using the same LLM for all modules. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

结合多个大语言模型（LLM）调用的复合AI系统，如自我优化和多智能体辩论，在许多AI任务上取得了出色表现。我们探讨了优化复合系统中的核心问题：对于系统中的每个LLM调用或模块，应如何决定使用哪个LLM？我们展示了这些LLM选择对质量有重大影响，但搜索空间呈指数增长。为此，我们提出了LLMSelector，这是一种高效的选择模型框架，利用了两个关键的经验见解：（i）端到端性能通常在固定其他模块情况下，每个模块性能提高而单调增长；（ii）通过LLM可准确估计每个模块的性能。基于这些见解，LLMSelector 逐步选择一个模块，并为该模块分配根据LLM估计表现最佳的模型，直到无法进一步提高性能。LLMSelector 可应用于具有有限模块数量的任何复合系统，其API调用次数随模块数量线性增长，在实验和理论上均实现了高质量的模型分配。使用流行的复合系统如多智能体辩论和自我优化以及LLM如GPT-4o、Claude 3.5 Sonnet和Gemini 1.5的实验表明，与使用同一LLM为所有模块相比，LLMSelector 可提供5%至70%的准确性提升。 

---
# PEARL: Towards Permutation-Resilient LLMs 

**Title (ZH)**: PEARL：朝着抗置换鲁棒的大型语言模型方向发展 

**Authors**: Liang Chen, Li Shen, Yang Deng, Xiaoyan Zhao, Bin Liang, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.14628)  

**Abstract**: The in-context learning (ICL) capability of large language models (LLMs) enables them to perform challenging tasks using provided demonstrations. However, ICL is highly sensitive to the ordering of demonstrations, leading to instability in predictions. This paper shows that this vulnerability can be exploited to design a natural attack - difficult for model providers to detect - that achieves nearly 80% success rate on LLaMA-3 by simply permuting the demonstrations. Existing mitigation methods primarily rely on post-processing and fail to enhance the model's inherent robustness to input permutations, raising concerns about safety and reliability of LLMs. To address this issue, we propose Permutation-resilient learning (PEARL), a novel framework based on distributionally robust optimization (DRO), which optimizes model performance against the worst-case input permutation. Specifically, PEARL consists of a permutation-proposal network (P-Net) and the LLM. The P-Net generates the most challenging permutations by treating it as an optimal transport problem, which is solved using an entropy-constrained Sinkhorn algorithm. Through minimax optimization, the P-Net and the LLM iteratively optimize against each other, progressively improving the LLM's robustness. Experiments on synthetic pre-training and real-world instruction tuning tasks demonstrate that PEARL effectively mitigates permutation attacks and enhances performance. Notably, despite being trained on fewer shots and shorter contexts, PEARL achieves performance gains of up to 40% when scaled to many-shot and long-context scenarios, highlighting its efficiency and generalization capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）的上下文内学习（ICL）能力使它们能够利用提供的示范来完成具有挑战性的任务。然而，ICL 对示范的顺序极为敏感，导致预测结果不稳定。本文表明，这一脆弱性可用于设计一种自然攻击——对于模型供应商难以检测，且能够在通过简单地重新排列示范使得 LLaMA-3 达到接近 80% 的成功率。现有的缓解方法主要依赖于后处理，并未能增强模型对输入排列的固有鲁棒性，这引发了对LLMs 安全性和可靠性的担忧。为解决这一问题，我们提出了一种基于分布鲁棒优化（DRO）的新型框架——Permutation-resilient learning (PEARL)，该框架旨在优化模型在最坏输入排列情况下的性能。具体而言，PEARL 包括一个排列建议网络（P-Net）和 LLM。P-Net 将排列问题视为一个最优传输问题，并通过熵约束的 Sinkhorn 算法进行求解。通过最小极大优化，P-Net 和 LLM 相互优化，逐步提高LLM 的鲁棒性。在合成预训练和现实世界的指令调优任务上的实验表明，PEARL 有效缓解了排列攻击并提高了性能。值得注意的是，尽管在少量示例和较短上下文中进行训练，当扩展到多示例和长上下文场景时，PEARL 能够实现高达 40% 的性能提升，这突显了其高效性和泛化能力。 

---
# ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification 

**Title (ZH)**: ReVISE：通过内在自我验证在测试时学习 refinement 

**Authors**: Hyunseok Lee, Seunghyuk Oh, Jaehyung Kim, Jinwoo Shin, Jihoon Tack  

**Link**: [PDF](https://arxiv.org/pdf/2502.14565)  

**Abstract**: Self-awareness, i.e., the ability to assess and correct one's own generation, is a fundamental aspect of human intelligence, making its replication in large language models (LLMs) an important yet challenging task. Previous works tackle this by employing extensive reinforcement learning or rather relying on large external verifiers. In this work, we propose Refine via Intrinsic Self-Verification (ReVISE), an efficient and effective framework that enables LLMs to self-correct their outputs through self-verification. The core idea of ReVISE is to enable LLMs to verify their reasoning processes and continually rethink reasoning trajectories based on its verification. We introduce a structured curriculum based upon online preference learning to implement this efficiently. Specifically, as ReVISE involves two challenging tasks (i.e., self-verification and reasoning correction), we tackle each task sequentially using curriculum learning, collecting both failed and successful reasoning paths to construct preference pairs for efficient training. During inference, our approach enjoys natural test-time scaling by integrating self-verification and correction capabilities, further enhanced by our proposed confidence-aware decoding mechanism. Our experiments on various reasoning tasks demonstrate that ReVISE achieves efficient self-correction and significantly improves reasoning performance. 

**Abstract (ZH)**: 自我意识，即评估和纠正自己生成内容的能力，是人类智能的一个基本方面，因此在大规模语言模型（LLMs）中复制这一能力是一项既重要又具有挑战性的工作。先前的研究通过广泛使用强化学习或依赖大型外部验证者来解决这一问题。在本研究中，我们提出了一种名为内在自我验证的精炼方法（ReVISE）的有效框架，使LLMs能够通过自我验证来纠正其输出。ReVISE的核心思想是使LLMs能够验证其推理过程，并基于验证结果持续重新思考推理路径。为此，我们引入了一种基于在线偏好学习的结构化课程，以高效地实施这一过程。具体来说，由于ReVISE涉及两个具有挑战性的任务（自我验证和推理纠正），我们通过逐个解决每个任务并收集成功和失败的推理路径来构建偏好对，以实现高效的训练。在推理过程中，我们的方法通过整合自我验证和纠正能力，自然地实现了测试时的规模扩展，并通过我们提出的一种基于信心的解码机制进一步增强。在各种推理任务上的实验表明，ReVISE能够实现高效的自我纠正，并显著提高推理性能。 

---
# Less is More: Improving LLM Alignment via Preference Data Selection 

**Title (ZH)**: 更少即是更多：通过偏好数据选择提升大语言模型一致性 

**Authors**: Xun Deng, Han Zhong, Rui Ai, Fuli Feng, Zheng Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2502.14560)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a promising approach for aligning large language models with human preferences. While prior work mainly extends DPO from the aspect of the objective function, we instead improve DPO from the largely overlooked but critical aspect of data selection. Specifically, we address the issue of parameter shrinkage caused by noisy data by proposing a novel margin-maximization principle for dataset curation in DPO training. To accurately estimate margins for data selection, we propose a dual-margin guided approach that considers both external reward margins and implicit DPO reward margins. Extensive experiments demonstrate that our method reduces computational cost dramatically while improving performance. Remarkably, by using just 10\% of the Ultrafeedback dataset, our approach achieves 3\% to 8\% improvements across various Llama and Mistral series models on the AlpacaEval 2.0 benchmark. Furthermore, our approach seamlessly extends to iterative DPO, yielding a roughly 3\% improvement with 25\% online data, while further reducing training time. These results highlight the potential of data selection strategies for advancing preference optimization. 

**Abstract (ZH)**: 直接偏好优化（DPO）已经成为一种有前景的方法，用于使大规模语言模型与人类偏好对齐。尽管前期的工作主要通过目标函数扩展了DPO，我们则从被忽视但至关重要的数据选择方面改进了DPO。具体而言，我们通过提出一种新颖的边际最大化原则来解决由嘈杂数据引起的参数收缩问题，以改进DPO训练中的数据集管理。为了准确估计用于数据选择的边际，我们提出了一种双边际引导方法，该方法同时考虑外部奖励边际和隐含的DPO奖励边际。广泛的经验研究证明，我们的方法在大幅降低计算成本的同时提高了性能。令人印象深刻的是，仅使用Ultrafeedback数据集的10%，我们的方法在AlpacaEval 2.0基准测试中的Llama和Mistral系列模型上实现了3%到8%的性能提升。此外，我们的方法无缝扩展到了迭代DPO，通过使用25%的在线数据，我们不仅提升了约3%的性能，还进一步降低了训练时间。这些结果突显了数据选择策略在偏好优化中潜在的作用。 

---
# Generative adversarial networks vs large language models: a comparative study on synthetic tabular data generation 

**Title (ZH)**: 生成对抗网络与大规模语言模型：合成表格数据生成的比较研究 

**Authors**: Austin A. Barr, Robert Rozman, Eddie Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14523)  

**Abstract**: We propose a new framework for zero-shot generation of synthetic tabular data. Using the large language model (LLM) GPT-4o and plain-language prompting, we demonstrate the ability to generate high-fidelity tabular data without task-specific fine-tuning or access to real-world data (RWD) for pre-training. To benchmark GPT-4o, we compared the fidelity and privacy of LLM-generated synthetic data against data generated with the conditional tabular generative adversarial network (CTGAN), across three open-access datasets: Iris, Fish Measurements, and Real Estate Valuation. Despite the zero-shot approach, GPT-4o outperformed CTGAN in preserving means, 95% confidence intervals, bivariate correlations, and data privacy of RWD, even at amplified sample sizes. Notably, correlations between parameters were consistently preserved with appropriate direction and strength. However, refinement is necessary to better retain distributional characteristics. These findings highlight the potential of LLMs in tabular data synthesis, offering an accessible alternative to generative adversarial networks and variational autoencoders. 

**Abstract (ZH)**: 我们提出了一种新的框架，用于零样本生成合成表格数据。通过使用大型语言模型（LLM）GPT-4o 和简单的语言提示，我们展示了在无需特定任务的微调或访问真实世界数据（RWD）进行预训练的情况下，生成高保真度表格数据的能力。为了评估 GPT-4o，我们将其生成的合成数据的保真度和隐私性与使用条件性生成对抗网络（CTGAN）生成的合成数据进行了比较，测试了三个开源数据集：Iris、Fish Measurements 和 Real Estate Valuation。尽管采用零样本方法，GPT-4o 在保留均值、95% 置信区间、双变量相关性和 RWD 的数据隐私方面仍然优于 CTGAN，即使在放大样本量的情况下也是如此。值得注意的是，与参数之间的相关性保持了适当的方向和强度。然而，仍需改进以更好地保留数据分布特性。这些发现突显了LLMs在表格数据合成中的潜力，为生成对抗网络和变分自编码器提供了可访问的替代方案。 

---
# Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions 

**Title (ZH)**: 大型语言模型考虑安全性吗？对编程问题回应的实证研究 

**Authors**: Amirali Sajadi, Binh Le, Anh Nguyen, Kostadin Damevski, Preetha Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14202)  

**Abstract**: The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses. 

**Abstract (ZH)**: 在软件开发中广泛应用会话型大语言模型（LLM）引发了对LLM生成内容安全性的新关切。我们通过动机性研究阐述了ChatGPT在志愿提供上下文相关信息方面的潜力，以促进安全编码实践。受此发现的启发，我们对Claude 3、GPT-4和Llama 3三种主流LLM的安全意识程度进行了研究。我们通过向这些LLM提供包含漏洞代码的Stack Overflow问题，评估它们是否仅仅是提供问题的答案，还是同时警告用户有关潜在不安全代码的问题，从而展示其安全意识程度。进一步地，我们评估LLM的响应是否提供了关于漏洞原因、利用方法和潜在修复的信息，以帮助提高用户的意识。我们的研究结果表明，所有三种模型在准确检测和警告用户关于漏洞方面面临困难，在我们数据集中，其检测率为12.6%至40%。我们还注意到，这些LLM更频繁地识别那些与敏感信息暴露和不当输入中立化相关的漏洞类型，而对其他类型，如外部控制文件名或路径的漏洞关注较少。此外，当LLM发出安全警告时，它们提供的关于漏洞原因、利用方法和修复的信息比Stack Overflow的回应要多。最后，我们深入探讨了研究结果的影响，并提出了一种基于命令行接口（CLI）的提示工具，可以用于生成更安全的LLM响应。 

---
# On the logical skills of large language models: evaluations using arbitrarily complex first-order logic problems 

**Title (ZH)**: 大型语言模型的逻辑能力评估：使用任意复杂的一阶逻辑问题 

**Authors**: Shokhrukh Ibragimov, Arnulf Jentzen, Benno Kuckuck  

**Link**: [PDF](https://arxiv.org/pdf/2502.14180)  

**Abstract**: We present a method of generating first-order logic statements whose complexity can be controlled along multiple dimensions. We use this method to automatically create several datasets consisting of questions asking for the truth or falsity of first-order logic statements in Zermelo-Fraenkel set theory. While the resolution of these questions does not require any knowledge beyond basic notation of first-order logic and set theory, it does require a degree of planning and logical reasoning, which can be controlled up to arbitrarily high difficulty by the complexity of the generated statements. Furthermore, we do extensive evaluations of the performance of various large language models, including recent models such as DeepSeek-R1 and OpenAI's o3-mini, on these datasets. All of the datasets along with the code used for generating them, as well as all data from the evaluations is publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了一种生成一阶逻辑命题的方法，其复杂性可以在多个维度上进行控制。我们使用这种方法自动生成了多个数据集，这些数据集包含的问题是询问一阶逻辑命题在策梅洛-弗兰克尔集合理论中的真假。解答这些问题不需要任何超出基本一阶逻辑和集合理论符号的知识，但需要一定程度的计划和逻辑推理能力，而这种能力可以通过生成命题的复杂性来任意调整到很高的难度。此外，我们对包括最近发布的模型DeepSeek-R1和OpenAI的o3-mini在内的多种大型语言模型在这些数据集上的性能进行了广泛评估。所有数据集、生成这些数据集所使用的代码以及评估中的所有数据均可通过以下网址公开访问：[此处插入网址]。 

---
# Giving AI Personalities Leads to More Human-Like Reasoning 

**Title (ZH)**: 赋予AI个性有助于实现更类人的推理 

**Authors**: Animesh Nighojkar, Bekhzodbek Moydinboyev, My Duong, John Licato  

**Link**: [PDF](https://arxiv.org/pdf/2502.14155)  

**Abstract**: In computational cognitive modeling, capturing the full spectrum of human judgment and decision-making processes, beyond just optimal behaviors, is a significant challenge. This study explores whether Large Language Models (LLMs) can emulate the breadth of human reasoning by predicting both intuitive, fast System 1 and deliberate, slow System 2 processes. We investigate the potential of AI to mimic diverse reasoning behaviors across a human population, addressing what we call the {\em full reasoning spectrum problem}. We designed reasoning tasks using a novel generalization of the Natural Language Inference (NLI) format to evaluate LLMs' ability to replicate human reasoning. The questions were crafted to elicit both System 1 and System 2 responses. Human responses were collected through crowd-sourcing and the entire distribution was modeled, rather than just the majority of the answers. We used personality-based prompting inspired by the Big Five personality model to elicit AI responses reflecting specific personality traits, capturing the diversity of human reasoning, and exploring how personality traits influence LLM outputs. Combined with genetic algorithms to optimize the weighting of these prompts, this method was tested alongside traditional machine learning models. The results show that LLMs can mimic human response distributions, with open-source models like Llama and Mistral outperforming proprietary GPT models. Personality-based prompting, especially when optimized with genetic algorithms, significantly enhanced LLMs' ability to predict human response distributions, suggesting that capturing suboptimal, naturalistic reasoning may require modeling techniques incorporating diverse reasoning styles and psychological profiles. The study concludes that personality-based prompting combined with genetic algorithms is promising for enhancing AI's \textit{human-ness} in reasoning. 

**Abstract (ZH)**: 在计算认知模型中，超越最优行为来捕捉人类判断和决策过程的完整谱系是一项重大挑战。本研究探讨大语言模型（LLMs）是否能够通过预测直觉快速的System 1过程和审慎缓慢的System 2过程来模拟人类推理的广度。我们研究了人工智能模仿人类群体中多样推理行为的潜力，解决我们称之为“完整推理谱系问题”的问题。我们使用了一种自然语言推理（NLI）格式的新颖扩展来设计推理任务，以评估LLMs复制人类推理的能力。问题设计得既能引发直觉快速的System 1反应，又能引发审慎缓慢的System 2反应。人类反应是通过众包收集的，并且我们对整个分布进行了建模，而不仅仅是大多数答案的分布。我们借鉴了大五人格模型的启发，使用基于人格的提示来引发反映特定人格特质的AI响应，从而捕捉人类推理的多样性，并探讨人格特质如何影响LLM的输出。结合遗传算法来优化这些提示的权重后，这种方法与传统的机器学习模型进行了测试。结果表明，开源模型如Llama和Mistral的性能优于专有的GPT模型。尤其是使用遗传算法优化的人格基于提示，显著增强了LLMs预测人类反应分布的能力，表明捕捉非理想的、自然性的推理可能需要融合多样推理风格和心理特征的建模技术。本研究得出结论，结合遗传算法的人格基于提示方法对未来增强AI在推理中的“人性化”具有潜力。 

---
# Investigating Non-Transitivity in LLM-as-a-Judge 

**Title (ZH)**: 探究基于语言模型的法官（LLM-as-a-Judge）中非传递性的现象 

**Authors**: Yi Xu, Laura Ruis, Tim Rocktäschel, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2502.14074)  

**Abstract**: Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的自动评估方法正在成为评估LLM代理执行指令能力的标准工具。在这一范式中最常见的方法是使用基线模型进行成对比较，这种方法严格依赖于传递偏好假设的有效性。然而，这一假设的有效性尚未得到充分探讨。在本研究中，我们考察了AlpacaEval框架中的非传递性现象，并分析了这种现象对模型排名的影响。我们发现，LLM评判者表现出非传递性偏好，这使得排名对基线模型的选择非常敏感。为了缓解这一问题，我们展示了采用轮换锦标赛结合布雷德利-特里模型的偏好方法可以产生更可靠的排名。值得注意的是，我们的方法分别将Spearman相关性和肯德尔相关性与Chatbot Arena的匹配提高到了96.4%（从95.0%）和86.3%（从82.1%）。为解决轮换锦标赛的计算成本问题，我们提出了智慧迭代匹配（Swim）锦标赛，通过动态匹配策略同时保留轮换锦标赛的优势并保持计算效率。 

---
# EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations 

**Title (ZH)**: EquivaMap：利用大规模语言模型进行优化模型等价性自动检查 

**Authors**: Haotian Zhai, Connor Lawless, Ellen Vitercik, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14760)  

**Abstract**: A fundamental problem in combinatorial optimization is identifying equivalent formulations, which can lead to more efficient solution strategies and deeper insights into a problem's computational complexity. The need to automatically identify equivalence between problem formulations has grown as optimization copilots--systems that generate problem formulations from natural language descriptions--have proliferated. However, existing approaches to checking formulation equivalence lack grounding, relying on simple heuristics which are insufficient for rigorous validation. Inspired by Karp reductions, in this work we introduce quasi-Karp equivalence, a formal criterion for determining when two optimization formulations are equivalent based on the existence of a mapping between their decision variables. We propose EquivaMap, a framework that leverages large language models to automatically discover such mappings, enabling scalable and reliable equivalence verification. To evaluate our approach, we construct the first open-source dataset of equivalent optimization formulations, generated by applying transformations such as adding slack variables or valid inequalities to existing formulations. Empirically, EquivaMap significantly outperforms existing methods, achieving substantial improvements in correctly identifying formulation equivalence. 

**Abstract (ZH)**: 组合优化中的一个基本问题是识别等价形式，这可以导致更高效的求解策略，并深入理解问题的计算复杂性。随着优化协驾系统（能够从自然语言描述生成问题形式的系统）的普及，自动识别问题形式之间的等价性变得越来越重要。然而，现有的形式等价性检查方法缺乏坚实的理论基础，仅仅依赖简单的启发式方法，这些方法不足以进行严格的验证。受到Karp约简的启发，我们在此工作中引入了准Karp等价性，这是一种基于决策变量之间的映射存在的形式来判断两个优化形式是否等价的正式标准。我们提出了一种名为EquivaMap的框架，利用大型语言模型自动发现这些映射，从而实现大规模且可靠的等价性验证。为评估我们的方法，我们构建了第一个开源的等价优化形式数据集，通过应用诸如增加松弛变量或有效不等式的变换生成这些形式。从实验结果来看，EquivaMap显著优于现有方法，在准确识别形式等价性方面取得了重大改进。 

---
# Plan-over-Graph: Towards Parallelable LLM Agent Schedule 

**Title (ZH)**: 基于图的计划：朝着可并行的LLM代理调度方向 

**Authors**: Shiqi Zhang, Xinbei Ma, Zouying Cao, Zhuosheng Zhang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14563)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional abilities in reasoning for task planning. However, challenges remain under-explored for parallel schedules. This paper introduces a novel paradigm, plan-over-graph, in which the model first decomposes a real-life textual task into executable subtasks and constructs an abstract task graph. The model then understands this task graph as input and generates a plan for parallel execution. To enhance the planning capability of complex, scalable graphs, we design an automated and controllable pipeline to generate synthetic graphs and propose a two-stage training scheme. Experimental results show that our plan-over-graph method significantly improves task performance on both API-based LLMs and trainable open-sourced LLMs. By normalizing complex tasks as graphs, our method naturally supports parallel execution, demonstrating global efficiency. The code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在任务规划推理方面展现出了卓越的能力。然而，对于并行调度的挑战仍存在未探索的空间。本文引入了一种新颖的范式——“计划覆盖图”（Plan-over-Graph），该范式使模型首先将实际文本任务分解为可执行子任务，并构建一个抽象的任务图。然后，模型将理解该任务图作为输入，并生成一个用于并行执行的计划。为了增强处理复杂可扩展图形的规划能力，我们设计了一个自动化且可控的管道来生成合成图形，并提出了一种两阶段训练方案。实验结果表明，我们的“计划覆盖图”方法在基于API的LLMs和可训练的开源LLMs上显著提高了任务性能。通过将复杂的任务规范化为图形表示，该方法自然支持并行执行，从而展示出全局效率。代码和数据已发布在该链接：[此处提供链接]。 

---
# Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning 

**Title (ZH)**: 增强检索过程奖励模型以实现通用数学推理 

**Authors**: Jiachen Zhu, Congmin Zheng, Jianghao Lin, Kounianhua Du, Ying Wen, Yong Yu, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14361)  

**Abstract**: While large language models (LLMs) have significantly advanced mathematical reasoning, Process Reward Models (PRMs) have been developed to evaluate the logical validity of reasoning steps. However, PRMs still struggle with out-of-distribution (OOD) challenges. This paper identifies key OOD issues, including step OOD, caused by differences in reasoning patterns across model types and sizes, and question OOD, which arises from dataset shifts between training data and real-world problems. To address these issues, we introduce Retrieval-Augmented Process Reward Model (RetrievalPRM), a novel framework designed to tackle these OOD issues. By utilizing a two-stage retrieval-enhanced mechanism, RetrievalPRM retrieves semantically similar questions and steps as a warmup, enhancing PRM's ability to evaluate target steps and improving generalization and reasoning consistency across different models and problem types. Our extensive experiments demonstrate that RetrievalPRM outperforms existing baselines across multiple real-world datasets. Our open-source contributions include a retrieval-enhanced dataset, a tuning framework for PRM training, and the RetrievalPRM model, establishing a new standard for PRM performance. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在数学推理方面取得了显著进展，过程奖励模型（PRMs）已经发展起来，用于评估推理步骤的逻辑有效性。然而，PRMs 仍然难以应对分布外（OOD）挑战。本文识别了关键的 OOD 问题，包括由于不同模型类型和规模的推理模式差异导致的步骤 OOD 问题，以及由于训练数据集与实际问题之间的偏移导致的问题 OOD 问题。为了应对这些问题，我们引入了检索增强过程奖励模型（RetrievalPRM），这是一种新的框架，旨在解决这些 OOD 问题。通过利用两阶段的检索增强机制，RetrievalPRM 在预热阶段检索语义上相似的问题和步骤，从而增强 PRM 评估目标步骤的能力，并在不同模型和问题类型之间提高泛化能力和推理一致性。广泛实验表明，RetrievalPRM 在多个实际数据集上优于现有基线。我们的开源贡献包括一个检索增强数据集、一个 PRM 训练的调优框架和 RetrievalPRM 模型，从而确立了 PRM 性能的新标准。 

---
# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis 

**Title (ZH)**: 树状辩论：多角色辩论树促进科学比较分析的批判性思维 

**Authors**: Priyanka Kargupta, Ishika Agarwal, Tal August, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14767)  

**Abstract**: With the exponential growth of research facilitated by modern technology and improved accessibility, scientific discoveries have become increasingly fragmented within and across fields. This makes it challenging to assess the significance, novelty, incremental findings, and equivalent ideas between related works, particularly those from different research communities. Large language models (LLMs) have recently demonstrated strong quantitative and qualitative reasoning abilities, and multi-agent LLM debates have shown promise in handling complex reasoning tasks by exploring diverse perspectives and reasoning paths. Inspired by this, we introduce Tree-of-Debate (ToD), a framework which converts scientific papers into LLM personas that debate their respective novelties. To emphasize structured, critical reasoning rather than focusing solely on outcomes, ToD dynamically constructs a debate tree, enabling fine-grained analysis of independent novelty arguments within scholarly articles. Through experiments on scientific literature across various domains, evaluated by expert researchers, we demonstrate that ToD generates informative arguments, effectively contrasts papers, and supports researchers in their literature review. 

**Abstract (ZH)**: 随着现代技术的发展和获取途径的改进，科学研究在各领域内乃至跨领域内的成果呈指数级增长。这使得评估相关成果之间的意义、新颖性、增量发现和等效观点变得越来越具有挑战性，尤其是来自不同研究社区的作品之间的评估。大型语言模型（LLMs）近期展示了强大的定量和定性推理能力，而多智能体LLM辩论则展示了在处理复杂推理任务方面的潜力，通过探索多样化的视角和推理路径来应对这些挑战。受此启发，我们引入了辩论树（Tree-of-Debate, ToD）框架，该框架将科学论文转化为能够辩论其各自新颖性的LLM角色。ToD通过促进对独立新颖性论点的精细分析，而不仅仅是关注结果，动态构建辩论树，从而强调结构化的批判性推理。通过跨多个学科的科学文献实验，并由专家研究人员评估，我们证明了ToD能够生成具有信息性的论点，有效对比论文，并支持研究人员进行文献综述。 

---
# EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration 

**Title (ZH)**: EAGER-LLM：通过外生行为语义集成增强大型语言模型的推荐能力 

**Authors**: Minjie Hong, Yan Xia, Zehan Wang, Jieming Zhu, Ye Wang, Sihang Cai, Xiaoda Yang, Quanyu Dai, Zhenhua Dong, Zhimeng Zhang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14735)  

**Abstract**: Large language models (LLMs) are increasingly leveraged as foundational backbones in the development of advanced recommender systems, offering enhanced capabilities through their extensive knowledge and reasoning. Existing llm-based recommender systems (RSs) often face challenges due to the significant differences between the linguistic semantics of pre-trained LLMs and the collaborative semantics essential for RSs. These systems use pre-trained linguistic semantics but learn collaborative semantics from scratch via the llm-Backbone. However, LLMs are not designed for recommendations, leading to inefficient collaborative learning, weak result correlations, and poor integration of traditional RS features. To address these challenges, we propose EAGER-LLM, a decoder-only llm-based generative recommendation framework that integrates endogenous and exogenous behavioral and semantic information in a non-intrusive manner. Specifically, we propose 1)dual-source knowledge-rich item indices that integrates indexing sequences for exogenous signals, enabling efficient link-wide processing; 2)non-invasive multiscale alignment reconstruction tasks guide the model toward a deeper understanding of both collaborative and semantic signals; 3)an annealing adapter designed to finely balance the model's recommendation performance with its comprehension capabilities. We demonstrate EAGER-LLM's effectiveness through rigorous testing on three public benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在先进推荐系统的发展中越来越多地被用作基础架构，通过其广泛的知识和推理能力提供了增强的功能。现有的基于LLM的推荐系统（RSs）往往面临挑战，因为预训练LLMs的语义与RSs所需的协作语义之间存在显著差异。这些系统利用预训练的语义，但通过LLM骨干重新学习协作语义。然而，LLMs并非为推荐设计，导致协作学习效率低下、结果相关性弱，传统RS特征的整合也不理想。为了解决这些挑战，我们提出了一种名为EAGER-LLM的解码器仅基于LLM的生成推荐框架，该框架以非侵入性的方式整合了内在和外在的行为和语义信息。具体而言，我们提出1）双源知识丰富的物品索引，不仅整合了对外部信号的索引序列，还使其能够高效地进行全局链接处理；2）非侵入性的多尺度对齐重建任务引导模型更深入地理解协作和语义信号；3）热化适配器，旨在精细平衡模型的推荐性能与理解能力。我们通过在三个公开基准上的严格测试展示了EAGER-LLM的有效性。 

---
# LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework 

**Title (ZH)**: LLM-EvRep：使用自我监督框架学习一种兼容LLM的事件表示 

**Authors**: Zongyou Yu, Qiang Qu, Qian Zhang, Nan Zhang, Xiaoming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14273)  

**Abstract**: Recent advancements in event-based recognition have demonstrated significant promise, yet most existing approaches rely on extensive training, limiting their adaptability for efficient processing of event-driven visual content. Meanwhile, large language models (LLMs) have exhibited remarkable zero-shot capabilities across diverse domains, but their application to event-based visual recognition remains largely unexplored. To bridge this gap, we propose \textbf{LLM-EvGen}, an event representation generator that produces LLM-compatible event representations \textbf{LLM-EvRep}, thereby enhancing the performance of LLMs on event recognition tasks. The generator is trained using a self-supervised framework, aligning the generated representations with semantic consistency and structural fidelity. Comprehensive experiments were conducted on three datasets: N-ImageNet, N-Caltech101, and N-MNIST. The results demonstrate that our method, \textbf{LLM-EvRep}, outperforms the event-to-video method, E2VID, by 15.93\%, 0.82\%, and 50.21\%, respectively, in recognition tasks when evaluated using GPT-4o. 

**Abstract (ZH)**: 近年来，基于事件的识别研究取得了显著的进展，但大多数现有方法依赖于大量的训练，限制了它们在处理事件驱动的视觉内容时的高效适应性。与此同时，大型语言模型（LLMs）在多个领域展示了惊人的零样本能力，但它们在事件驱动的视觉识别中的应用仍然 largely unexplored（未被充分探索）。为了解决这一问题，我们提出了 \textbf{LLM-EvGen}，一种事件表示生成器，它生成与LLM兼容的事件表示 \textbf{LLM-EvRep}，从而增强LLM在事件识别任务中的性能。生成器使用半监督框架进行训练，生成的表示与语义一致性和结构忠实性保持一致。我们在三个数据集——N-ImageNet、N-Caltech101 和 N-MNIST——上进行了全面的实验。结果表明，在使用GPT-4o评估时，我们的方法 \textbf{LLM-EvRep} 在识别任务中的性能分别优于事件到视频的方法 E2VID 15.93%、0.82% 和 50.21%。 

---
# Personalized Education with Generative AI and Digital Twins: VR, RAG, and Zero-Shot Sentiment Analysis for Industry 4.0 Workforce Development 

**Title (ZH)**: 基于生成式AI和数字孪生的个性化教育：面向工业4.0劳动力发展的VR、RAG和零样本情感分析 

**Authors**: Yu-Zheng Lin, Karan Petal, Ahmed H Alhamadah, Sujan Ghimire, Matthew William Redondo, David Rafael Vidal Corona, Jesus Pacheco, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14080)  

**Abstract**: The Fourth Industrial Revolution (4IR) technologies, such as cloud computing, machine learning, and AI, have improved productivity but introduced challenges in workforce training and reskilling. This is critical given existing workforce shortages, especially in marginalized communities like Underrepresented Minorities (URM), who often lack access to quality education. Addressing these challenges, this research presents gAI-PT4I4, a Generative AI-based Personalized Tutor for Industrial 4.0, designed to personalize 4IR experiential learning. gAI-PT4I4 employs sentiment analysis to assess student comprehension, leveraging generative AI and finite automaton to tailor learning experiences. The framework integrates low-fidelity Digital Twins for VR-based training, featuring an Interactive Tutor - a generative AI assistant providing real-time guidance via audio and text. It uses zero-shot sentiment analysis with LLMs and prompt engineering, achieving 86\% accuracy in classifying student-teacher interactions as positive or negative. Additionally, retrieval-augmented generation (RAG) enables personalized learning content grounded in domain-specific knowledge. To adapt training dynamically, finite automaton structures exercises into states of increasing difficulty, requiring 80\% task-performance accuracy for progression. Experimental evaluation with 22 volunteers showed improved accuracy exceeding 80\%, reducing training time. Finally, this paper introduces a Multi-Fidelity Digital Twin model, aligning Digital Twin complexity with Bloom's Taxonomy and Kirkpatrick's model, providing a scalable educational framework. 

**Abstract (ZH)**: 第四次工业革命（4IR）技术，如云计算、机器学习和人工智能，虽然提高了生产力，但也为劳动力培训和再培训带来了挑战。鉴于现有的劳动力短缺问题，尤其是在被边缘化的社区，如代表性不足的少数群体（URM），他们往往缺乏高质量教育机会。为应对这些挑战，本研究提出了一种基于生成式人工智能的个性化导师gAI-PT4I4，旨在个性化4IR体验式学习。gAI-PT4I4利用情感分析评估学生理解程度，结合生成式人工智能和有限自动机以定制学习体验。该框架整合了用于VR培训的低保真数字孪生，其中包括交互式导师——一个生成式人工智能助理，可提供实时音频和文本指导。它使用零样本的情感分析与大规模语言模型（LLM）及提示工程相结合，实现86%的准确性，用于分类学生-教师互动为正面或负面。此外，检索增强生成（RAG）技术能够生成基于专业领域知识的个性化学习内容。为动态适应培训需求，有限自动机结构化训练练习为递增难度的状态，要求任务执行准确率达到80%才能进步。实验评估使用22名志愿者显示，学习准确率超过了80%，从而缩短了培训时间。最后，本文介绍了多保真度数字孪生模型，该模型将数字孪生的复杂性与布卢姆分类法和柯克帕特里克模型相匹配，提供了一个可扩展的教育框架。 

---
# Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems 

**Title (ZH)**: 基于大型语言模型的对话推荐系统中的协作检索方法 

**Authors**: Yaochen Zhu, Chao Wan, Harald Steck, Dawen Liang, Yesu Feng, Nathan Kallus, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14137)  

**Abstract**: Conversational recommender systems (CRS) aim to provide personalized recommendations via interactive dialogues with users. While large language models (LLMs) enhance CRS with their superior understanding of context-aware user preferences, they typically struggle to leverage behavioral data, which have proven to be important for classical collaborative filtering (CF)-based approaches. For this reason, we propose CRAG, Collaborative Retrieval Augmented Generation for LLM-based CRS. To the best of our knowledge, CRAG is the first approach that combines state-of-the-art LLMs with CF for conversational recommendations. Our experiments on two publicly available movie conversational recommendation datasets, i.e., a refined Reddit dataset (which we name Reddit-v2) as well as the Redial dataset, demonstrate the superior item coverage and recommendation performance of CRAG, compared to several CRS baselines. Moreover, we observe that the improvements are mainly due to better recommendation accuracy on recently released movies. The code and data are available at this https URL. 

**Abstract (ZH)**: 对话推荐系统（CRS）旨在通过与用户的交互对话提供个性化推荐。大型语言模型（LLMs）通过对其上下文感知用户偏好的深刻理解，增强了CRS的能力，但通常难以利用行为数据，而这些行为数据对于基于协作过滤（CF）的经典方法来说是非常重要的。鉴于此，我们提出了CRAG方法，即结合了大型语言模型与协作过滤的对话推荐增强生成方法。据我们所知，CRAG是第一个将最先进的大型语言模型与CF相结合以进行对话推荐的方法。我们在两个公开的电影对话推荐数据集上进行了实验，即经过精炼的Reddit数据集（我们命名为Reddit-v2）以及Redial数据集，结果显示，CRAG在项目覆盖范围和推荐性能方面显著优于几种基准对话推荐系统。此外，我们观察到，这些改进主要归因于在最近上映的电影上的推荐准确性提高。代码和数据可从以下链接获取：[这里提供的链接]。 

---
