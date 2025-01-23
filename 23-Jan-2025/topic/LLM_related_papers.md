# Accessible Smart Contracts Verification: Synthesizing Formal Models with Tamed LLMs 

**Title (ZH)**: 可访问的智能合约验证：利用驯化的大型语言模型合成形式模型 

**Authors**: Jan Corazza, Ivan Gavran, Gabriela Moreira, Daniel Neider  

**Link**: [PDF](https://arxiv.org/pdf/2501.12972)  

**Abstract**: When blockchain systems are said to be trustless, what this really means is that all the trust is put into software. Thus, there are strong incentives to ensure blockchain software is correct -- vulnerabilities here cost millions and break businesses. One of the most powerful ways of establishing software correctness is by using formal methods. Approaches based on formal methods, however, induce a significant overhead in terms of time and expertise required to successfully employ them. Our work addresses this critical disadvantage by automating the creation of a formal model -- a mathematical abstraction of the software system -- which is often a core task when employing formal methods. We perform model synthesis in three phases: we first transpile the code into model stubs; then we "fill in the blanks" using a large language model (LLM); finally, we iteratively repair the generated model, on both syntactical and semantical level. In this way, we significantly reduce the amount of time necessary to create formal models and increase accessibility of valuable software verification methods that rely on them. The practical context of our work was reducing the time-to-value of using formal models for correctness audits of smart contracts. 

**Abstract (ZH)**: 当区块链系统被认为是“无信任”的时，这意味着所有的信任都被放在了软件上。因此，确保区块链软件正确的激励非常强大——这里存在的漏洞可能会造成数百万的损失并破坏企业。以形式方法建立软件正确性的最有力方式之一是使用形式化方法。然而，基于形式化方法的方法会带来显著的时间和专业知识方面的负担，才能成功地运用它们。我们的工作通过自动化创建形式模型——即软件系统的数学抽象——来解决这一关键不足，形式模型通常是应用形式化方法的核心任务之一。我们进行模型合成分为三个阶段：首先将代码转换为模型框架；然后使用大型语言模型（LLM）“填充空白”；最后，以语法和语义两个层面迭代修复生成的模型。这样，我们大大减少了创建形式模型所需的时间，并提高了依赖于这些形式模型的宝贵软件验证方法的可获得性。我们工作的实际背景是减少使用形式模型进行智能合约正确性审核所需的时间。 

---
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning 

**Title (ZH)**: DeepSeek-R1：通过强化学习激励大规模语言模型的推理能力 

**Authors**: DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J.L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R.J. Chen, R.L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S.S. Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12948)  

**Abstract**: We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. To support the research community, we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama. 

**Abstract (ZH)**: 我们介绍了我们第一代推理模型DeepSeek-R1-Zero和DeepSeek-R1。DeepSeek-R1-Zero是一种通过大规模强化学习（RL）训练而来的模型，没有预先通过监督微调（SFT）作为预备步骤，显示出卓越的推理能力。通过RL，DeepSeek-R1-Zero自然地展现出许多强大而引人注目的推理行为。然而，它也遇到了可读性差和语言混合等挑战。为了解决这些问题并进一步提升推理性能，我们引入了DeepSeek-R1，该模型在RL之前包含了多阶段训练和冷启动数据。DeepSeek-R1在推理任务上的性能与OpenAI的o1-1217相媲美。为支持科研界，我们开源了DeepSeek-R1-Zero、DeepSeek-R1以及基于Qwen和Llama从DeepSeek-R1中提炼的六个密集模型（1.5B、7B、8B、14B、32B、70B）。 

---
# Open or Closed LLM for Lesser-Resourced Languages? Lessons from Greek 

**Title (ZH)**: 面向较少资源语言的开放型或封闭型大语言模型？来自希腊语的经验教训

解释：这个标题是关于比较开放型大语言模型（open architecture models）和封闭型大语言模型（closed architecture models）在较少资源语言（lesser-resourced languages）上的应用效果，并以希腊语（Greek）为例进行探讨。翻译时保持了原意，并符合中文的表达习惯。 

**Authors**: John Pavlopoulos, Juli Bakagianni, Kanella Pouli, Maria Gavriilidou  

**Link**: [PDF](https://arxiv.org/pdf/2501.12826)  

**Abstract**: Natural Language Processing (NLP) for lesser-resourced languages faces persistent challenges, including limited datasets, inherited biases from high-resource languages, and the need for domain-specific solutions. This study addresses these gaps for Modern Greek through three key contributions. First, we evaluate the performance of open-source (Llama-70b) and closed-source (GPT-4o mini) large language models (LLMs) on seven core NLP tasks with dataset availability, revealing task-specific strengths, weaknesses, and parity in their performance. Second, we expand the scope of Greek NLP by reframing Authorship Attribution as a tool to assess potential data usage by LLMs in pre-training, with high 0-shot accuracy suggesting ethical implications for data provenance. Third, we showcase a legal NLP case study, where a Summarize, Translate, and Embed (STE) methodology outperforms the traditional TF-IDF approach for clustering \emph{long} legal texts. Together, these contributions provide a roadmap to advance NLP in lesser-resourced languages, bridging gaps in model evaluation, task innovation, and real-world impact. 

**Abstract (ZH)**: 少资源语言的自然语言处理（NLP）面临着持续的挑战，包括有限的数据集、从高资源语言继承来的偏见以及对特定领域解决方案的需要。本研究通过三个关键贡献，针对现代希腊语填补了这些空白。首先，我们评估了开源（Llama-70b）和封闭源（GPT-4o mini）大型语言模型（LLMs）在具有数据集支持的七个核心NLP任务上的性能，揭示了它们在不同任务上的优势、劣势以及表现一致性。其次，我们通过将作者归 attribution 重新框架为一种工具，评估LLMs在预训练中的潜在数据使用情况，高零样本准确率暗示了数据来源上的伦理问题。第三，我们展示了法律NLP案例研究，在此研究中，概括、翻译和嵌入（STE）方法在聚类长法律文本方面优于传统的TF-IDF方法。总体而言，这些贡献为推动少资源语言的NLP提供了蓝图，填补了模型评估、任务创新和实际影响方面的空白。 

---
# EvidenceMap: Unleashing the Power of Small Language Models with Evidence Analysis for Biomedical Question Answering 

**Title (ZH)**: 证据地图：通过证据分析释放小型语言模型在生物医学问答中的潜力 

**Authors**: Chang Zong, Jian Wan, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12746)  

**Abstract**: Current LLM-based approaches improve question answering performance by leveraging the internal reasoning abilities of models or incorporating external knowledge. However, when humans address professional problems, it is essential to explicitly analyze the multifaceted relationships from multiple pieces and diverse sources of evidence to achieve better answers. In this study, we propose a novel generative question answering framework for the biomedical domain, named EvidenceMap, which explicitly learns and incorporates evidence analysis with small language models (SLMs). The framework describes an evidence map for each question and fully utilizes an SLM to derive the representation of the supportive evaluation, the logical correlation, and the summarization of the related evidence, which facilitates an analysis-augmented generation with another SLM in an autoregressive way. Extensive experiments have shown that introducing an evidence analysis learning process can significantly outperform larger models and popular LLM reasoning methods. 

**Abstract (ZH)**: 当前基于大规模语言模型（LLM）的方法通过利用模型的内部推理能力和引入外部知识来提高问答性能。然而，当人类应对专业问题时，要实现更佳的答案，必须明确地分析来自多个方面和不同来源的复杂关系。在这项研究中，我们提出了一种新颖的生成式问答框架，名为EvidenceMap，该框架通过小型语言模型（SLM）明确地学习和整合证据分析。该框架为每个问题生成一个证据地图，并充分利用SLM从支持性评估、逻辑关联和相关证据的总结中推导出表示，从而以自回归的方式促进分析增强生成。广泛实验表明，引入证据分析学习过程可以显著优于更大规模的模型和流行的LLM推理方法。 

---
# Leveraging LLMs to Create a Haptic Devices' Recommendation System 

**Title (ZH)**: 利用大型语言模型构建aptic设备推荐系统 

**Authors**: Yang Liu, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2501.12573)  

**Abstract**: Haptic technology has seen significant growth, yet a lack of awareness of existing haptic device design knowledge hinders development. This paper addresses these limitations by leveraging advancements in Large Language Models (LLMs) to develop a haptic agent, focusing specifically on Grounded Force Feedback (GFF) devices recommendation. Our approach involves automating the creation of a structured haptic device database using information from research papers and product specifications. This database enables the recommendation of relevant GFF devices based on user queries. To ensure precise and contextually relevant recommendations, the system employs a dynamic retrieval method that combines both conditional and semantic searches. Benchmarking against the established UEQ and existing haptic device searching tools, the proposed haptic recommendation agent ranks in the top 10\% across all UEQ categories with mean differences favoring the agent in nearly all subscales, and maintains no significant performance bias across different user groups, showcasing superior usability and user satisfaction. 

**Abstract (ZH)**: 触觉技术已经取得了显著的增长，但现有的触觉设备设计知识缺乏认知，限制了其进一步发展。本文通过利用大型语言模型（LLMs）的进步来开发一个触觉代理，重点关注基于地面力反馈（GFF）设备的推荐。我们的方法包括利用研究论文和产品规范中的信息自动化创建一个结构化的触觉设备数据库。该数据库能够根据用户的查询推荐相关GFF设备。为了确保推荐的精确性和相关性，系统采用了一种动态检索方法，结合条件搜索和语义搜索。与现有的用户体验（UEQ）标准和触觉设备搜索工具进行基准测试，提出的触觉推荐代理在所有UEQ类别中的排名均位于前10%以内，平均差异在几乎所有亚量表上均有利于代理，且在不同用户群体中没有显著的性能偏向，这展示了其优越的可用性和用户满意度。 

---
# Human-like conceptual representations emerge from language prediction 

**Title (ZH)**: 人类概念表示源自语言预测 

**Authors**: Ningyu Xu, Qi Zhang, Chao Du, Qiang Luo, Xipeng Qiu, Xuanjing Huang, Menghan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12547)  

**Abstract**: Recent advances in large language models (LLMs) provide a new opportunity to address the long-standing question of how concepts are represented and organized in the mind, which is central to unravelling the nature of human cognition. Here, we reframed the classic reverse dictionary task to simulate human concept inference in context and investigated the emergence of human-like conceptual representations within LLMs. We found that LLMs were able to infer concepts from definitional descriptions and construct representation spaces that converge towards a shared, context-independent structure. These representations effectively predicted human behavioural judgments and aligned well with neural activity patterns in the human brain, offering evidence for biological plausibility. These findings demonstrate that human-like conceptual representations and organization can naturally emerge from language prediction, even without real-world grounding. Our work supports the view that LLMs serve as valuable tools for understanding complex human cognition and paves the way for better alignment between artificial and human intelligence. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）的发展为解决长期以来困扰人们的关于概念在心智中如何表示和组织的问题提供了新的机会，这关乎揭示人类认知的本质。本文中，我们将经典的逆向词典任务重新构想，以模拟人类在上下文中进行概念推理，并考察LLMs中人类类似的概念表示如何自然地涌现。我们发现，LLMs能够从定义性描述中推断概念，并构建收敛于共享、跨上下文无关结构的表示空间。这些表示能够有效预测人类的行为判断，与人类大脑的神经活动模式高度一致，提供了生物学可行性证据。这些发现表明，即使没有现实世界的基础，人类类似的概念表示和组织也能自然地从语言预测中涌现。本研究支持了LLMs作为理解复杂人类认知有价值的工具的观点，并为更好地使人工智能与人类智能对齐铺平了道路。 

---
# Refining Input Guardrails: Enhancing LLM-as-a-Judge Efficiency Through Chain-of-Thought Fine-Tuning and Alignment 

**Title (ZH)**: 优化输入边界：通过链式思考微调和对齐提高LLM作为仲裁者的效率 

**Authors**: Melissa Kazemi Rad, Huy Nghiem, Andy Luo, Sahil Wadhwa, Mohammad Sorower, Stephen Rawls  

**Link**: [PDF](https://arxiv.org/pdf/2501.13080)  

**Abstract**: Large Language Models (LLMs) have demonstrated powerful capabilities that render them valuable in different applications, including conversational AI products. It is paramount to ensure the security and reliability of these products by mitigating their vulnerabilities towards malicious user interactions, which can lead to the exposure of great risks and reputational repercussions. In this work, we present a comprehensive study on the efficacy of fine-tuning and aligning Chain-of-Thought (CoT) responses of different LLMs that serve as input moderation guardrails. We systematically explore various tuning methods by leveraging a small set of training data to adapt these models as proxy defense mechanisms to detect malicious inputs and provide a reasoning for their verdicts, thereby preventing the exploitation of conversational agents. We rigorously evaluate the efficacy and robustness of different tuning strategies to generalize across diverse adversarial and malicious query types. Our experimental results outline the potential of alignment processes tailored to a varied range of harmful input queries, even with constrained data resources. These techniques significantly enhance the safety of conversational AI systems and provide a feasible framework for deploying more secure and trustworthy AI-driven interactions. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示出了强大的能力，使其在不同的应用场景中变得非常有价值，包括对话型人工智能产品。确保这些产品的安全性和可靠性至关重要，这要求通过减轻其对恶意用户交互的脆弱性来加以防范，否则可能会引发严重风险和声誉损失。在本研究中，我们对不同LLMs的链式推理（Chain-of-Thought, CoT）响应进行微调和对齐的有效性进行了全面研究，这些响应可以用作输入规范的防护措施。我们通过利用少量训练数据系统地探索各种调优方法，将这些模型作为代理防御机制，以检测恶意输入并为其判决提供推理依据，从而防止对话代理的滥用。我们严格评估了不同调优策略的效用和鲁棒性，使其能够适应多样化的对抗性及恶意查询类型。实验结果表明，即使在数据资源有限的情况下，针对不同类型有害输入进行对齐的过程也具有潜在的效果。这些技术极大地增强了对话型人工智能系统的安全性，并为部署更安全和可信赖的AI驱动交互提供了可行框架。 

---
# Pairwise RM: Perform Best-of-N Sampling with Knockout Tournament 

**Title (ZH)**: Pairwise RM：最佳N采样 Knockout 赛制算法 

**Authors**: Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13007)  

**Abstract**: Best-of-N (BoN) sampling, a common strategy for test-time scaling of Large Language Models (LLMs), relies on reward models to select the best candidate solution from multiple generations. However, traditional reward models often assign arbitrary and inconsistent scores, limiting their effectiveness. To address this, we propose a Pairwise Reward Model (Pairwise RM) combined with a knockout tournament for BoN sampling. Instead of assigning absolute scores, given one math problem, Pairwise RM evaluates two candidate solutions' correctness simultaneously. This approach eliminates the need for arbitrary scoring and enables cross-validation of solutions through parallel comparison. In the knockout tournament, Pairwise RM conducts pairwise comparisons between candidate solutions and eliminates the incorrect ones iteratively. We construct \ourdataset, a large-scale dataset of 443K pairwise comparisons derived from NumiaMath and annotated using \texttt{gemini-1.5-flash}, and train the Pairwise RM via supervised fine-tuning. Experiments on MATH-500 and the Olympiad Bench demonstrate significant improvements over traditional discriminative reward models. And a 40\% to 60\% relative improvement is achieved on the top 50\% challenging problems. 

**Abstract (ZH)**: Best-of-N（BoN）采样是一种常见的大型语言模型（LLMs）测试时扩展策略，它依赖于奖励模型从多个生成结果中选择最佳候选解决方案。然而，传统的奖励模型通常会分配任意且不一致的分数，从而限制了它们的有效性。为了解决这一问题，我们提出了一种结合淘汰赛机制的配对奖励模型（Pairwise Reward Model，Pairwise RM）。与传统的奖励模型不同，给定一个数学问题时，Pairwise RM同时评估两个候选解决方案的正确性。这种方法消除了任意评分的需要，并通过并行比较实现了解决方案的交叉验证。在淘汰赛中，Pairwise RM 对候选解决方案进行配对比较，并逐步淘汰不正确的解决方案。我们构建了一个名为 \ourdataset 的大型数据集，包含443K个配对比较，这些数据集源自 NumiaMath，并使用 \texttt{gemini-1.5-flash} 进行注释。我们通过监督微调训练了 Pairwise RM。在 MATH-500 和奥林匹克竞赛基准测试中，实验结果显示Pairwise RM 比传统区分性奖励模型有显著改进。并且在最具有挑战性的前50%的问题上，相对改进达到了40%到60%。 

---
# Implicit Causality-biases in humans and LLMs as a tool for benchmarking LLM discourse capabilities 

**Title (ZH)**: 人类和大语言模型中隐含的因果偏见及其作为评估大语言模型 discourse 能力工具的作用 

**Authors**: Florian Kankowski, Torgrim Solstad, Sina Zarriess, Oliver Bott  

**Link**: [PDF](https://arxiv.org/pdf/2501.12980)  

**Abstract**: In this paper, we compare data generated with mono- and multilingual LLMs spanning a range of model sizes with data provided by human participants in an experimental setting investigating well-established discourse biases. Beyond the comparison as such, we aim to develop a benchmark to assess the capabilities of LLMs with discourse biases as a robust proxy for more general discourse understanding capabilities. More specifically, we investigated Implicit Causality verbs, for which psycholinguistic research has found participants to display biases with regard to three phenomena:\ the establishment of (i) coreference relations (Experiment 1), (ii) coherence relations (Experiment 2), and (iii) the use of particular referring expressions (Experiments 3 and 4). With regard to coreference biases we found only the largest monolingual LLM (German Bloom 6.4B) to display more human-like biases. For coherence relation, no LLM displayed the explanation bias usually found for humans. For referring expressions, all LLMs displayed a preference for referring to subject arguments with simpler forms than to objects. However, no bias effect on referring expression was found, as opposed to recent studies investigating human biases. 

**Abstract (ZH)**: 在本文中，我们比较了使用单语和多语言大规模语言模型（LLM）生成的数据与实验条件下人类参与者提供的数据，这些数据涉及已建立的语言偏见。除了直接比较之外，我们还旨在开发一个基准，以评估LLM在语言偏见方面的能力，作为更广泛语言理解能力的稳健代理指标。更为具体地，我们研究了隐含因果动词，这些动词在心理语言学研究中被发现与三个现象有关：（i）指示关系的建立（实验1），（ii）连贯关系的建立（实验2），以及（iii）使用特定称呼表达（实验3和实验4）。

在关于指示关系偏见的研究中，我们仅发现最大的单语大规模语言模型（德语Bloom 6.4B）显示了更类似人类的偏见。对于连贯关系，没有任何一个LLM显示人类通常所表现的解释偏见。对于称呼表达，所有语言模型都更倾向于使用形式更简单的主语作为指称对象，而较少使用对象。然而，关于称呼表达的偏见效应并没有被发现，这与近期研究人类偏见的成果不同。 

---
# OnionEval: An Unified Evaluation of Fact-conflicting Hallucination for Small-Large Language Models 

**Title (ZH)**: OnionEval：小型和大型语言模型事实矛盾幻觉的统一评估 

**Authors**: Chongren Sun, Yuran Li, Di Wu, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2501.12975)  

**Abstract**: Large Language Models (LLMs) are highly capable but require significant computational resources for both training and inference. Within the LLM family, smaller models (those with fewer than 10 billion parameters) also perform well across various tasks. However, these smaller models share similar limitations to their larger counterparts, including the tendency to hallucinate. Despite the existence of many benchmarks to evaluate hallucination in LLMs, few have specifically focused on small LLMs (SLLMs). Additionally, SLLMs show widely varying performance across different benchmarks. In this paper, we introduce OnionEval, a multi-layer structured framework with a specific metric called the context-influence score (CI), designed to effectively assess the fact-conflicting hallucination tendencies of small LLMs across different contextual levels. Our experimental results reveal a key feature of SLLMs: they excel in factual analysis but face challenges with context reasoning. Further investigation shows that a simple Chain-of-Thought strategy can significantly reduce these limitations, improving the practical usefulness of SLLMs in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有高度的效能，但在训练和推理过程中需要大量计算资源。在LLM家族中，那些参数量少于100亿的较小模型也在各种任务中表现出色。然而，这些小型模型在相似的方面也面临着与大型模型相同的局限性，包括虚构事实的倾向。尽管存在许多用于评估LLM虚构事实能力的基准，但很少有研究专门关注小型LLM（SLLMs）。此外，SLLMs在不同基准测试中的表现差异很大。在本文中，我们引入了OnionEval，这是一种多层结构化框架，带有特定指标——上下文影响评分（CI），专门用于评估SLLMs在不同上下文层次上的事实冲突虚构倾向。我们的实验结果揭示了一个关键特征：SLLMs在事实分析方面表现优异，但在上下文推理方面面临挑战。进一步的研究表明，使用简单的逐层推理策略可以显著减少这些局限性，从而提高SLLMs在实际应用中的实用价值。 

---
# FilmAgent: A Multi-Agent Framework for End-to-End Film Automation in Virtual 3D Spaces 

**Title (ZH)**: FilmAgent：面向虚拟3D空间端到端电影自动化的一种多代理框架 

**Authors**: Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12909)  

**Abstract**: Virtual film production requires intricate decision-making processes, including scriptwriting, virtual cinematography, and precise actor positioning and actions. Motivated by recent advances in automated decision-making with language agent-based societies, this paper introduces FilmAgent, a novel LLM-based multi-agent collaborative framework for end-to-end film automation in our constructed 3D virtual spaces. FilmAgent simulates various crew roles, including directors, screenwriters, actors, and cinematographers, and covers key stages of a film production workflow: (1) idea development transforms brainstormed ideas into structured story outlines; (2) scriptwriting elaborates on dialogue and character actions for each scene; (3) cinematography determines the camera setups for each shot. A team of agents collaborates through iterative feedback and revisions, thereby verifying intermediate scripts and reducing hallucinations. We evaluate the generated videos on 15 ideas and 4 key aspects. Human evaluation shows that FilmAgent outperforms all baselines across all aspects and scores 3.98 out of 5 on average, showing the feasibility of multi-agent collaboration in filmmaking. Further analysis reveals that FilmAgent, despite using the less advanced GPT-4o model, surpasses the single-agent o1, showing the advantage of a well-coordinated multi-agent system. Lastly, we discuss the complementary strengths and weaknesses of OpenAI's text-to-video model Sora and our FilmAgent in filmmaking. 

**Abstract (ZH)**: 虚拟电影制作需要复杂的决策过程，包括剧本创作、虚拟摄影以及精准的演员定位和动作。受近年来基于语言代理社会的自动化决策技术进步的启发，本文介绍了FilmAgent，这是一种基于LLM（大语言模型）的多代理协作框架，用于在我们构建的3D虚拟空间中实现从头到尾的电影自动化。FilmAgent模拟了各种剧组角色，包括导演、编剧、演员和摄影师，并覆盖了电影制作工作流程的关键阶段：（1）创意发展将初步的想法转化为结构化的故事情节；（2）剧本创作详细描述每个场景中的对话和角色动作；（3）摄像确定每个镜头的相机设置。多个代理通过迭代反馈和修订协作，从而验证中间脚本并减少幻觉。我们对15个创意和4个关键方面生成的视频进行了评估。人类评估结果显示，FilmAgent在所有方面都优于所有基线，并且平均得分为3.98/5，这表明多代理协作在电影制作中的可行性。进一步分析表明，尽管FilmAgent使用的是更具限制性的GPT-4o模型，但其性能仍优于单一代理的o1系统，这体现了协调良好的多代理系统的优点。最后，我们讨论了OpenAI的文本转视频模型Sora和我们所提出的FilmAgent在电影制作中的互补优势和劣势。 

---
# WisdomBot: Tuning Large Language Models with Artificial Intelligence Knowledge 

**Title (ZH)**: WisdomBot：使用人工智能知识调优大型语言模型 

**Authors**: Jingyuan Chen, Tao Wu, Wei Ji, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12877)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools in natural language processing (NLP), showing a promising future of artificial generated intelligence (AGI). Despite their notable performance in the general domain, LLMs have remained suboptimal in the field of education, owing to the unique challenges presented by this domain, such as the need for more specialized knowledge, the requirement for personalized learning experiences, and the necessity for concise explanations of complex concepts. To address these issues, this paper presents a novel LLM for education named WisdomBot, which combines the power of LLMs with educational theories, enabling their seamless integration into educational contexts. To be specific, we harness self-instructed knowledge concepts and instructions under the guidance of Bloom's Taxonomy as training data. To further enhance the accuracy and professionalism of model's response on factual questions, we introduce two key enhancements during inference, i.e., local knowledge base retrieval augmentation and search engine retrieval augmentation during inference. We substantiate the effectiveness of our approach by applying it to several Chinese LLMs, thereby showcasing that the fine-tuned models can generate more reliable and professional responses. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为自然语言处理（NLP）领域中强大的工具，并展现了人工智能生成（AGI）领域的光明前景。尽管在通用领域中表现出了显著的能力，但LLMs在教育领域仍存在不足，这归因于该领域独有的挑战，如需要更专门的知识、个性化学习体验的要求以及对于复杂概念的简洁解释的必要性。为了应对这些问题，本文提出了一种名为WisdomBot的新颖教育型LLM，将LLM的强大功能与教育理论相结合，使其能够无缝地融入教育环境。具体而言，我们利用基于布鲁姆分类法的自我指导的知识概念和指示进行训练。为了进一步提高模型在事实问题上回答的准确性和专业性，我们在推理过程引入了两项关键增强，即局部知识库检索增强和搜索引擎检索增强。我们通过将该方法应用于几种中文LLM，验证了其有效性，从而展示了微调模型能够生成更可靠和专业的回答。 

---
# LLMs as Repositories of Factual Knowledge: Limitations and Solutions 

**Title (ZH)**: LLMs作为事实性知识的存储库：局限性与解决方案 

**Authors**: Seyed Mahed Mousavi, Simone Alghisi, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12774)  

**Abstract**: LLMs' sources of knowledge are data snapshots containing factual information about entities collected at different timestamps and from different media types (e.g. wikis, social media, etc.). Such unstructured knowledge is subject to change due to updates through time from past to present. Equally important are the inconsistencies and inaccuracies occurring in different information sources. Consequently, the model's knowledge about an entity may be perturbed while training over the sequence of snapshots or at inference time, resulting in inconsistent and inaccurate model performance. In this work, we study the appropriateness of Large Language Models (LLMs) as repositories of factual knowledge. We consider twenty-four state-of-the-art LLMs that are either closed-, partially (weights), or fully (weight and training data) open-source. We evaluate their reliability in responding to time-sensitive factual questions in terms of accuracy and consistency when prompts are perturbed. We further evaluate the effectiveness of state-of-the-art methods to improve LLMs' accuracy and consistency. We then propose "ENtity-Aware Fine-tuning" (ENAF), a soft neurosymbolic approach aimed at providing a structured representation of entities during fine-tuning to improve the model's performance. 

**Abstract (ZH)**: 以下是该论文内容或标题的中文翻译，符合学术规范：

大规模语言模型（LLMs）的知识来源是包含不同时间戳和不同媒体类型（例如维基百科、社交媒体等）中实体事实信息的数据快照。由于随着时间从过去到现在不断的更新，这种无结构的知识会随之发生变化。同样重要的是，在不同信息源中出现的不一致性和不准确性。因此，当模型在一系列数据快照的训练过程中或在推理阶段，其对实体的知识可能会受到影响，导致模型性能的不一致和不准确。在本文中，我们研究了大规模语言模型（LLMs）作为事实知识库的适当性。我们考虑了二十四个先进的LLMs，它们或处于封闭状态，或部分开源（权重），或完全开源（权重和训练数据）。我们评估了它们在响应时间敏感的事实问题方面的可靠性和一致性，特别是在扰动提示时的表现。我们进一步评估了最先进的方法在提高大规模语言模型准确性与一致性的有效性。然后，我们提出了“实体感知微调”（ENAF）方法，这是一种软神经符号方法，旨在在微调过程中提供实体的结构化表示，以改善模型的性能。 

---
# NExtLong: Toward Effective Long-Context Training without Long Documents 

**Title (ZH)**: NNextLong：面向有效的长语境训练，无需长文档 

**Authors**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12766)  

**Abstract**: Large language models (LLMs) with extended context windows have made significant strides yet remain a challenge due to the scarcity of long documents. Existing methods tend to synthesize long-context data but lack a clear mechanism to reinforce the long-range dependency modeling. To address this limitation, we propose NExtLong, a novel framework for synthesizing long-context data through Negative document Extension. NExtLong decomposes a document into multiple meta-chunks and extends the context by interleaving hard negative distractors retrieved from pretraining corpora. This approach compels the model to discriminate long-range dependent context from distracting content, enhancing its ability to model long-range dependencies. Extensive experiments demonstrate that NExtLong achieves significant performance improvements on the HELMET and RULER benchmarks compared to existing long-context synthesis approaches and leading models, which are trained on non-synthetic long documents. These findings highlight NExtLong's ability to reduce reliance on non-synthetic long documents, making it an effective framework for developing advanced long-context LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有扩展的上下文窗口，在生成长文档方面取得了显著进展，但仍面临挑战，原因是长文档的稀缺性。现有的方法倾向于合成长上下文数据，但缺乏明确机制来强化长距离依赖模型。为解决这一局限性，我们提出了一种名为NExtLong的新框架，该框架通过负文档扩展来合成长上下文数据。NExtLong将文档分解为多个超块，并通过交错预训练语料库中检索到的硬负干扰块来扩展上下文。这种方法迫使模型区分相关的长距离依赖上下文与分散的内容，从而增强其建模长距离依赖的能力。大量实验表明，NExtLong在HELMET和RULER基准测试中相比现有长上下文合成方法和基于真实长文档训练的领先模型，实现了显著的性能提升。这些发现突显了NExtLong减少对非合成长文档依赖的能力，使其成为开发高级长上下文LLM的有效框架。 

---
# Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression 

**Title (ZH)**: 通过AI反馈训练对话系统以提高整体对话印象 

**Authors**: Kai Yoshida, Masahiro Mizukami, Seiya Kawano, Canasai Kruengkrai, Hiroaki Sugiyama, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2501.12698)  

**Abstract**: To improve user engagement during conversations with dialogue systems, we must improve individual dialogue responses and dialogue impressions such as consistency, personality, and empathy throughout the entire dialogue. While such dialogue systems have been developing rapidly with the help of large language models (LLMs), reinforcement learning from AI feedback (RLAIF) has attracted attention to align LLM-based dialogue models for such dialogue impressions. In RLAIF, a reward model based on another LLM is used to create a training signal for an LLM-based dialogue model using zero-shot/few-shot prompting techniques. However, evaluating an entire dialogue only by prompting LLMs is challenging. In this study, the supervised fine-tuning (SFT) of LLMs prepared reward models corresponding to 12 metrics related to the impression of the entire dialogue for evaluating dialogue responses. We tuned our dialogue models using the reward model signals as feedback to improve the impression of the system. The results of automatic and human evaluations showed that tuning the dialogue model using our reward model corresponding to dialogue impression improved the evaluation of individual metrics and the naturalness of the dialogue response. 

**Abstract (ZH)**: 为了提高对话系统中用户的参与度，在整个对话过程中，我们需要改进个体对话响应以及对话印象，包括一致性、个性和同理心。借助大规模语言模型（LLMs）的支持，此类对话系统正快速开发，而基于强化学习的来自AI反馈（RLAIF）方法则引起了关注，用于引导基于LLM的对话模型以匹配这些对话印象。在RLAIF中，通过零样本/少样本提示技术，使用另一种LLM构建的奖励模型被用来为基于LLM的对话模型提供训练信号。然而，仅通过提示LLMs来评估整个对话存在挑战。在此项研究中，我们对LLMs进行了有监督微调（SFT），以便基于与12个涉及整体对话印象的指标相对应的奖励模型来评估对话响应。我们使用奖励模型信号作为反馈来改进系统的对话印象。自动和人工评估结果表明，使用与对话印象相对应的奖励模型来调整对话模型可以提升单个指标的评估结果，并提高对话响应的自然度。 

---
