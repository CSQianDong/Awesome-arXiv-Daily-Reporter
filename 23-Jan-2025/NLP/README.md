# Refining Input Guardrails: Enhancing LLM-as-a-Judge Efficiency Through Chain-of-Thought Fine-Tuning and Alignment 

**Title (ZH)**: 优化输入边界：通过链式思考微调和对齐提高LLM作为仲裁者的效率 

**Authors**: Melissa Kazemi Rad, Huy Nghiem, Andy Luo, Sahil Wadhwa, Mohammad Sorower, Stephen Rawls  

**Link**: [PDF](https://arxiv.org/pdf/2501.13080)  

**Abstract**: Large Language Models (LLMs) have demonstrated powerful capabilities that render them valuable in different applications, including conversational AI products. It is paramount to ensure the security and reliability of these products by mitigating their vulnerabilities towards malicious user interactions, which can lead to the exposure of great risks and reputational repercussions. In this work, we present a comprehensive study on the efficacy of fine-tuning and aligning Chain-of-Thought (CoT) responses of different LLMs that serve as input moderation guardrails. We systematically explore various tuning methods by leveraging a small set of training data to adapt these models as proxy defense mechanisms to detect malicious inputs and provide a reasoning for their verdicts, thereby preventing the exploitation of conversational agents. We rigorously evaluate the efficacy and robustness of different tuning strategies to generalize across diverse adversarial and malicious query types. Our experimental results outline the potential of alignment processes tailored to a varied range of harmful input queries, even with constrained data resources. These techniques significantly enhance the safety of conversational AI systems and provide a feasible framework for deploying more secure and trustworthy AI-driven interactions. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示出了强大的能力，使其在不同的应用场景中变得非常有价值，包括对话型人工智能产品。确保这些产品的安全性和可靠性至关重要，这要求通过减轻其对恶意用户交互的脆弱性来加以防范，否则可能会引发严重风险和声誉损失。在本研究中，我们对不同LLMs的链式推理（Chain-of-Thought, CoT）响应进行微调和对齐的有效性进行了全面研究，这些响应可以用作输入规范的防护措施。我们通过利用少量训练数据系统地探索各种调优方法，将这些模型作为代理防御机制，以检测恶意输入并为其判决提供推理依据，从而防止对话代理的滥用。我们严格评估了不同调优策略的效用和鲁棒性，使其能够适应多样化的对抗性及恶意查询类型。实验结果表明，即使在数据资源有限的情况下，针对不同类型有害输入进行对齐的过程也具有潜在的效果。这些技术极大地增强了对话型人工智能系统的安全性，并为部署更安全和可信赖的AI驱动交互提供了可行框架。 

---
# Autonomy-of-Experts Models 

**Title (ZH)**: 专家自主性模型 

**Authors**: Ang Lv, Ruobing Xie, Yining Qian, Songhao Wu, Xingwu Sun, Zhanhui Kang, Di Wang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13074)  

**Abstract**: Mixture-of-Experts (MoE) models mostly use a router to assign tokens to specific expert modules, activating only partial parameters and often outperforming dense models. We argue that the separation between the router's decision-making and the experts' execution is a critical yet overlooked issue, leading to suboptimal expert selection and ineffective learning. To address this, we propose Autonomy-of-Experts (AoE), a novel MoE paradigm in which experts autonomously select themselves to process inputs. AoE is based on the insight that an expert is aware of its own capacity to effectively process a token, an awareness reflected in the scale of its internal activations. In AoE, routers are removed; instead, experts pre-compute internal activations for inputs and are ranked based on their activation norms. Only the top-ranking experts proceed with the forward pass, while the others abort. The overhead of pre-computing activations is reduced through a low-rank weight factorization. This self-evaluating-then-partner-comparing approach ensures improved expert selection and effective learning. We pre-train language models having 700M up to 4B parameters, demonstrating that AoE outperforms traditional MoE models with comparable efficiency. 

**Abstract (ZH)**: Mixture-of-Experts (MoE) 模型主要使用路由器将令牌分配给特定的专家模块，激活的部分参数往往优于密集模型。我们认为，路由器的决策与专家执行之间的分离是关键但被忽视的问题，导致专家选择不优化和学习效果不佳。为了解决这一问题，我们提出了自治专家（Autonomy-of-Experts, AoE）这一新颖的 MoE 架构，在这种架构中，专家能够自主选择处理输入。AoE 的灵感来源于专家对自己处理令牌能力的认知，这种认知体现在其内部激活的规模上。在 AoE 中，移除了路由器；相反，专家预先计算输入的内部激活并根据它们的激活范数进行排名。仅排名靠前的专家进行前向传递，其余专家则终止。通过低秩权重因子分解减少预先计算激活的开销。这种自我评估后与其他专家比较的方法确保了专家选择的优化和有效的学习。我们对从 700M 到 4B 参数的语言模型进行了预训练，结果显示 AoE 在与传统 MoE 模型相当的效率下表现出更优的效果。 

---
# Does Table Source Matter? Benchmarking and Improving Multimodal Scientific Table Understanding and Reasoning 

**Title (ZH)**: 《表格数据源重要吗？跨模态科学表格理解与推理的基准测试及提升》 

**Authors**: Bohao Yang, Yingji Zhang, Dong Liu, André Freitas, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13042)  

**Abstract**: Recent large language models (LLMs) have advanced table understanding capabilities but rely on converting tables into text sequences. While multimodal large language models (MLLMs) enable direct visual processing, they face limitations in handling scientific tables due to fixed input image resolutions and insufficient numerical reasoning capabilities. We present a comprehensive framework for multimodal scientific table understanding and reasoning with dynamic input image resolutions. Our framework consists of three key components: (1) MMSci-Pre, a domain-specific table structure learning dataset of 52K scientific table structure recognition samples, (2) MMSci-Ins, an instruction tuning dataset with 12K samples across three table-based tasks, and (3) MMSci-Eval, a benchmark with 3,114 testing samples specifically designed to evaluate numerical reasoning capabilities. Extensive experiments demonstrate that our domain-specific approach with 52K scientific table images achieves superior performance compared to 150K general-domain tables, highlighting the importance of data quality over quantity. Our proposed table-based MLLMs with dynamic input resolutions show significant improvements in both general table understanding and numerical reasoning capabilities, with strong generalisation to held-out datasets. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在表格理解方面取得了显著进展，但它们依赖于将表格转换为文本序列。而多模态大型语言模型（MLLMs）能够直接处理视觉信息，但由于固定输入图像分辨率和不足的数值推理能力，它们在处理科学表格方面面临限制。我们提出了一种全面的框架，用于处理具有动态输入图像分辨率的多模态科学表格理解和推理。我们的框架包括三个关键组件：（1）MMSci-Pre，一个包含52,000个科学表格结构识别样本的专业领域表格结构学习数据集；（2）MMSci-Ins，一个包含12,000个样本的任务指令调优数据集，涉及三种基于表格的任务；（3）MMSci-Eval，一个包含3,114个测试样本的基准数据集，专门设计用于评估数值推理能力。大量实验表明，我们使用52,000个科学表格图像的专业领域方法优于使用150,000个通用领域的表格，突显了数据质量而非数量的重要性。我们提出的支持动态输入分辨率的基于表格的MLLMs在通用表格理解和数值推理能力方面显示出显著改进，并且对未见过的数据集具有强大的泛化能力。我们将在以下链接公开发布代码和数据：this https URL。 

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
# FlanEC: Exploring Flan-T5 for Post-ASR Error Correction 

**Title (ZH)**: FlanEC：探索 Flan-T5 在后ASR 错误修正中的应用 

**Authors**: Moreno La Quatra, Valerio Mario Salerno, Yu Tsao, Sabato Marco Siniscalchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12979)  

**Abstract**: In this paper, we present an encoder-decoder model leveraging Flan-T5 for post-Automatic Speech Recognition (ASR) Generative Speech Error Correction (GenSEC), and we refer to it as FlanEC. We explore its application within the GenSEC framework to enhance ASR outputs by mapping n-best hypotheses into a single output sentence. By utilizing n-best lists from ASR models, we aim to improve the linguistic correctness, accuracy, and grammaticality of final ASR transcriptions. Specifically, we investigate whether scaling the training data and incorporating diverse datasets can lead to significant improvements in post-ASR error correction. We evaluate FlanEC using the HyPoradise dataset, providing a comprehensive analysis of the model's effectiveness in this domain. Furthermore, we assess the proposed approach under different settings to evaluate model scalability and efficiency, offering valuable insights into the potential of instruction-tuned encoder-decoder models for this task. 

**Abstract (ZH)**: 在本文中，我们提出了一种利用Flan-T5的编码器-解码器模型，用于后自动语音识别（ASR）生成性语音错误校正（GenSEC），并将其称为FlanEC。我们探讨了在GenSEC框架中应用该模型的方法，通过将n-best假设映射为单个输出句子，以提升ASR输出的质量。通过利用ASR模型的n-best列表，我们旨在提高最终ASR转录的语义正确性、准确性和语法正确性。特别是，我们研究了扩大训练数据规模和引入多样化的数据集是否能显著提高后ASR错误校正的效果。我们使用HyPoradise数据集评估了FlanEC，并对模型在该领域的有效性进行了全面分析。此外，我们还在不同的设置下评估了所提出的方法，以评估模型的可扩展性和效率，从而为这种任务提供了关于指令调优编码器-解码器模型潜力的宝贵见解。 

---
# OnionEval: An Unified Evaluation of Fact-conflicting Hallucination for Small-Large Language Models 

**Title (ZH)**: OnionEval：小型和大型语言模型事实矛盾幻觉的统一评估 

**Authors**: Chongren Sun, Yuran Li, Di Wu, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2501.12975)  

**Abstract**: Large Language Models (LLMs) are highly capable but require significant computational resources for both training and inference. Within the LLM family, smaller models (those with fewer than 10 billion parameters) also perform well across various tasks. However, these smaller models share similar limitations to their larger counterparts, including the tendency to hallucinate. Despite the existence of many benchmarks to evaluate hallucination in LLMs, few have specifically focused on small LLMs (SLLMs). Additionally, SLLMs show widely varying performance across different benchmarks. In this paper, we introduce OnionEval, a multi-layer structured framework with a specific metric called the context-influence score (CI), designed to effectively assess the fact-conflicting hallucination tendencies of small LLMs across different contextual levels. Our experimental results reveal a key feature of SLLMs: they excel in factual analysis but face challenges with context reasoning. Further investigation shows that a simple Chain-of-Thought strategy can significantly reduce these limitations, improving the practical usefulness of SLLMs in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有高度的效能，但在训练和推理过程中需要大量计算资源。在LLM家族中，那些参数量少于100亿的较小模型也在各种任务中表现出色。然而，这些小型模型在相似的方面也面临着与大型模型相同的局限性，包括虚构事实的倾向。尽管存在许多用于评估LLM虚构事实能力的基准，但很少有研究专门关注小型LLM（SLLMs）。此外，SLLMs在不同基准测试中的表现差异很大。在本文中，我们引入了OnionEval，这是一种多层结构化框架，带有特定指标——上下文影响评分（CI），专门用于评估SLLMs在不同上下文层次上的事实冲突虚构倾向。我们的实验结果揭示了一个关键特征：SLLMs在事实分析方面表现优异，但在上下文推理方面面临挑战。进一步的研究表明，使用简单的逐层推理策略可以显著减少这些局限性，从而提高SLLMs在实际应用中的实用价值。 

---
# Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference 

**Title (ZH)**: 利用评估器端口实现高效的提示压缩以优化长上下文变换器推理 

**Authors**: Weizhi Fei, Xueyan Niu, Guoqing Xie, Yingqing Liu, Bo Bai, Wei Han  

**Link**: [PDF](https://arxiv.org/pdf/2501.12959)  

**Abstract**: Although applications involving long-context inputs are crucial for the effective utilization of large language models (LLMs), they also result in increased computational costs and reduced performance. To address this challenge, we propose an efficient, training-free prompt compression method that retains key information within compressed prompts. We identify specific attention heads in transformer-based LLMs, which we designate as evaluator heads, that are capable of selecting tokens in long inputs that are most significant for inference. Building on this discovery, we develop EHPC, an Evaluator Head-based Prompt Compression method, which enables LLMs to rapidly "skim through" input prompts by leveraging only the first few layers with evaluator heads during the pre-filling stage, subsequently passing only the important tokens to the model for inference. EHPC achieves state-of-the-art results across two mainstream benchmarks: prompt compression and long-context inference acceleration. Consequently, it effectively reduces the complexity and costs associated with commercial API calls. We further demonstrate that EHPC attains competitive results compared to key-value cache-based acceleration methods, thereby highlighting its potential to enhance the efficiency of LLMs for long-context tasks. 

**Abstract (ZH)**: 虽然涉及长上下文输入的应用对大型语言模型（LLMs）的有效利用至关重要，但这也导致了更高的计算成本和性能降低。为应对这一挑战，我们提出了一种高效的、无需训练的提示压缩方法，该方法在压缩提示中保留了关键信息。我们识别出在基于变换器的LLMs中具有选择功能的关键注意头部，我们将其命名为评估头。基于这一发现，我们开发了EHPC（Evaluator Head-based Prompt Compression）方法，该方法使LLMs可以通过利用预填阶段中的前几层评估头来快速“浏览”输入提示，并仅将最重要的令牌传递给模型进行推理。EHPC在两个主流基准（提示压缩和长上下文推理加速）上均取得了最先进的结果，从而有效地降低了商业API调用相关的复杂性和成本。进一步的实验表明，EHPC在与基于键值缓存的加速方法相比时，也能获得有竞争力的性能，从而突显了其在提高长上下文任务中LLMs效率方面的潜力。 

---
# Multifractal hopscotch in "Hopscotch" by Julio Cortazar 

**Title (ZH)**: 《Julio Cortázar的作品〈跳房子〉中的多重分形跳跃》 

**Authors**: Jakub Dec, Michał Dolina, Stanisław Drożdż, Jarosław Kwapień, Tomasz Stanisz  

**Link**: [PDF](https://arxiv.org/pdf/2501.12955)  

**Abstract**: Punctuation is the main factor introducing correlations in natural language written texts and it crucially impacts their overall effectiveness, expressiveness, and readability. Punctuation marks at the end of sentences are of particular importance as their distribution can determine various complexity features of written natural language. Here, the sentence length variability (SLV) time series representing "Hopscotch" by Julio Cortazar are subjected to quantitative analysis with an attempt to identify their distribution type, long-memory effects, and potential multiscale patterns. The analyzed novel is an important and innovative piece of literature whose essential property is freedom of movement between its building blocks given to a reader by the author. The statistical consequences of this freedom are closely investigated in both the original, Spanish version of the novel, and its translations into English and Polish. Clear evidence of rich multifractality in the SLV dynamics, with a left-sided asymmetry, however, is observed in all three language versions as well as in the versions with differently ordered chapters. 

**Abstract (ZH)**: 标点符号是自然语言书面文本中引入相关性的主要因素，并且对文本的整体有效性、表达力和可读性具有关键影响。句末标点符号尤为重要，因为它们的分布可以决定书面自然语言的各种复杂特征。在此，代表朱利奥·科塔萨尔的作品《 hopscotch 》的句长变异性（SLV）时间序列进行了定量分析，以确定其分布类型、长记忆效应以及潜在的多尺度模式。这部分析的小说是文学中的一个重要且创新的作品，其核心特性是作者给予读者在其构建块之间自由移动的自由。这种自由的统计后果在小说的原始西班牙文版本及其英语和波兰语翻译版本中进行了详细研究。然而，在所有三种语言版本以及章节顺序不同的版本中，都观察到了丰富的多分形特征，且具有左偏非对称性。 

---
# Punctuation patterns in "Finnegans Wake" by James Joyce are largely translation-invariant 

**Title (ZH)**: 詹姆斯·乔伊斯的《芬尼根的苏醒》中的标点模式在很大程度上具有翻译不变性。 

**Authors**: Krzysztof Bartnicki, Stanisław Drożdż, Jarosław Kwapień, Tomasz Stanisz  

**Link**: [PDF](https://arxiv.org/pdf/2501.12954)  

**Abstract**: The complexity characteristics of texts written in natural languages are significantly related to the rules of punctuation. In particular, the distances between punctuation marks measured by the number of words quite universally follow the family of Weibull distributions known from survival analyses. However, the values of two parameters marking specific forms of these distributions distinguish specific languages. This is such a strong constraint that the punctuation distributions of texts translated from the original language into another adopt quantitative characteristics of the target language. All these changes take place within Weibull distributions such that the corresponding hazard functions are always increasing. Recent previous research shows that James Joyce's famous "Finnegans Wake" is subject to such extreme distribution from the Weibull family that the corresponding hazard function is clearly decreasing. At the same time, the distances of sentence ending punctuation marks, determining the variability of sentence length, have an almost perfect multifractal organization, so far to such an extent found nowhere else in the literature. In the present contribution based on several available translations (Dutch, French, German, Polish, Russian) of "Finnegans Wake", it is shown that the punctuation characteristics of this work remain largely translation invariant, contrary to the common cases. These observations may constitute further evidence that "Finnegans Wake" is a translinguistic work in this respect as well, in line with Joyce's original intention. 

**Abstract (ZH)**: 自然语言撰写的文本复杂特性与标点规则密切相关，尤其是通过词数测量的标点符号之间的距离普遍遵循来自生存分析的韦布尔分布。然而，这些分布的具体形式由两个参数标识，而这些参数值在不同的语言中有显著区别。这种限制是如此强烈，以至于从原语言翻译成其他语言的文本的标点分布会体现出目标语言的数量特征。所有这些变化均在韦布尔分布内进行，相应的危险函数始终保持递增。近期的研究表明，詹姆斯·乔伊斯的著名作品《芬尼根的苏生》的分布属于韦布尔家族的极端情况，其相应的危险函数明显呈递减趋势。同时，句子结尾标点符号的距离，决定了句子长度的变异性，这一现象几乎完美地呈现出多重分形组织，这是目前文献中所罕见的。

在本文中，基于《芬尼根的苏生》可用的几种翻译版本（荷兰语、法语、德语、波兰语、俄语），研究显示该作品的标点特征保持很大程度上的翻译不变性，这与通常的情况相反。这些观察结果可能进一步证明，从语言角度来看，《芬尼根的苏生》也具有跨语言特性，这与乔伊斯的原始意图一致。 

---
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning 

**Title (ZH)**: DeepSeek-R1：通过强化学习激励大规模语言模型的推理能力 

**Authors**: DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J.L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R.J. Chen, R.L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S.S. Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12948)  

**Abstract**: We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. To support the research community, we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama. 

**Abstract (ZH)**: 我们介绍了我们第一代推理模型DeepSeek-R1-Zero和DeepSeek-R1。DeepSeek-R1-Zero是一种通过大规模强化学习（RL）训练而来的模型，没有预先通过监督微调（SFT）作为预备步骤，显示出卓越的推理能力。通过RL，DeepSeek-R1-Zero自然地展现出许多强大而引人注目的推理行为。然而，它也遇到了可读性差和语言混合等挑战。为了解决这些问题并进一步提升推理性能，我们引入了DeepSeek-R1，该模型在RL之前包含了多阶段训练和冷启动数据。DeepSeek-R1在推理任务上的性能与OpenAI的o1-1217相媲美。为支持科研界，我们开源了DeepSeek-R1-Zero、DeepSeek-R1以及基于Qwen和Llama从DeepSeek-R1中提炼的六个密集模型（1.5B、7B、8B、14B、32B、70B）。 

---
# Ontology-Enhanced Educational Annotation Activities 

**Title (ZH)**: ontology增强的教育标注活动 

**Authors**: Joaquí Gayoso-Cabada, María Goicoechea-de-Jorge, Mercedes Gómez-Albarrán, Amelia Sanz-Cabrerizo, Antonio Sarasa-Cabezuelo, José-Luis Sierra  

**Link**: [PDF](https://arxiv.org/pdf/2501.12943)  

**Abstract**: Information and communications technology and technology-enhanced learning have unquestionably transformed traditional teaching-learning processes and are positioned as key factors to promote quality education, one of the basic sustainable development goals of the 2030 agenda. Document annotation, which was traditionally carried out with pencil and paper and currently benefits from digital document annotation tools, is a representative example of this transformation. Using document annotation tools, students can enrich the documents with annotations that highlight the most relevant aspects of these documents. As the conceptual complexity of the learning domain increases, the annotation of the documents may require comprehensive domain knowledge and an expert analysis capability that students usually lack. Consequently, a proliferation of irrelevant, incorrect, and/or poorly decontextualized annotations may appear, while other relevant aspects are completely ignored by the students. The main hypothesis proposed by this paper is that the use of a guiding annotation ontology in the annotation activities is a keystone aspect to alleviate these shortcomings. Consequently, comprehension is improved, exhaustive content analysis is promoted, and meta-reflective thinking is developed. To test this hypothesis, we describe our own annotation tool, \@note, which fully implements this ontology-enhanced annotation paradigm, and we provide experimental evidence about how \@note can improve academic performance via a pilot study concerning critical literary annotation. 

**Abstract (ZH)**: 信息技术和增强学习技术无疑已经改变了传统的教与学过程，并被定位为推动优质教育的关键因素，而优质教育正是2030年可持续发展议程的重要组成部分之一。文档注释作为一种传统的纸质手写注释方式，目前已得益于数字文档注释工具的使用，是这一变革的典型代表。通过使用文档注释工具，学生能够为文档添加能够突出展示这些文档最相关方面的标注。随着学习领域概念复杂性的增加，这种文档注释可能需要全面的专业知识和专家分析能力，而这通常是学生所缺乏的。因此，无关、不准确或脱离上下文的标注可能会变得普遍，而其他重要方面则可能完全被学生忽视。本文的主要假设是，在注释活动中使用引导注释本体论是缓解这些问题的关键因素。因此，这将提高理解能力、促进全面的内容分析，并培养反思性思维。为了验证这一假设，我们描述了我们自己开发的注释工具 \@note，该工具全面实现了本体论增强的注释范式，并通过有关批判性文学注释试点研究的实验性证据展示了 \@note 如何通过提高学术表现来改善这一过程。 

---
# FilmAgent: A Multi-Agent Framework for End-to-End Film Automation in Virtual 3D Spaces 

**Title (ZH)**: FilmAgent：面向虚拟3D空间端到端电影自动化的一种多代理框架 

**Authors**: Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12909)  

**Abstract**: Virtual film production requires intricate decision-making processes, including scriptwriting, virtual cinematography, and precise actor positioning and actions. Motivated by recent advances in automated decision-making with language agent-based societies, this paper introduces FilmAgent, a novel LLM-based multi-agent collaborative framework for end-to-end film automation in our constructed 3D virtual spaces. FilmAgent simulates various crew roles, including directors, screenwriters, actors, and cinematographers, and covers key stages of a film production workflow: (1) idea development transforms brainstormed ideas into structured story outlines; (2) scriptwriting elaborates on dialogue and character actions for each scene; (3) cinematography determines the camera setups for each shot. A team of agents collaborates through iterative feedback and revisions, thereby verifying intermediate scripts and reducing hallucinations. We evaluate the generated videos on 15 ideas and 4 key aspects. Human evaluation shows that FilmAgent outperforms all baselines across all aspects and scores 3.98 out of 5 on average, showing the feasibility of multi-agent collaboration in filmmaking. Further analysis reveals that FilmAgent, despite using the less advanced GPT-4o model, surpasses the single-agent o1, showing the advantage of a well-coordinated multi-agent system. Lastly, we discuss the complementary strengths and weaknesses of OpenAI's text-to-video model Sora and our FilmAgent in filmmaking. 

**Abstract (ZH)**: 虚拟电影制作需要复杂的决策过程，包括剧本创作、虚拟摄影以及精准的演员定位和动作。受近年来基于语言代理社会的自动化决策技术进步的启发，本文介绍了FilmAgent，这是一种基于LLM（大语言模型）的多代理协作框架，用于在我们构建的3D虚拟空间中实现从头到尾的电影自动化。FilmAgent模拟了各种剧组角色，包括导演、编剧、演员和摄影师，并覆盖了电影制作工作流程的关键阶段：（1）创意发展将初步的想法转化为结构化的故事情节；（2）剧本创作详细描述每个场景中的对话和角色动作；（3）摄像确定每个镜头的相机设置。多个代理通过迭代反馈和修订协作，从而验证中间脚本并减少幻觉。我们对15个创意和4个关键方面生成的视频进行了评估。人类评估结果显示，FilmAgent在所有方面都优于所有基线，并且平均得分为3.98/5，这表明多代理协作在电影制作中的可行性。进一步分析表明，尽管FilmAgent使用的是更具限制性的GPT-4o模型，但其性能仍优于单一代理的o1系统，这体现了协调良好的多代理系统的优点。最后，我们讨论了OpenAI的文本转视频模型Sora和我们所提出的FilmAgent在电影制作中的互补优势和劣势。 

---
# Architectural Fusion Through Contextual Partitioning in Large Language Models: A Novel Approach to Parameterized Knowledge Integration 

**Title (ZH)**: 在大型语言模型中通过上下文分区进行建筑融合：一种参数化知识集成的新方法 

**Authors**: Offa Kingsleigh, Alfred Abercrombie, David Woolstencroft, Beorhtric Meadowcroft, Marcus Irvin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12901)  

**Abstract**: Contextual Partitioning introduces an innovative approach to enhancing the architectural design of large-scale computational models through the dynamic segmentation of parameters into context-aware regions. This methodology emphasizes the importance of task-specific specialization, achieved through adaptive parameter allocation mechanisms that align with the linguistic features of input data. Experimental evaluations demonstrated substantial improvements in accuracy, perplexity, and contextual coherence across a variety of linguistic tasks, highlighting the adaptability and scalability of the proposed framework. By reducing redundancy and enhancing computational efficiency, Contextual Partitioning not only streamlines model operations but also expands the scope of applications for advanced language processing systems. The approach operates autonomously, requiring no external fine-tuning, thereby addressing a significant limitation in conventional parameter optimization techniques. Empirical results demonstrate the effectiveness of gradient-driven segmentation, enabling models to dynamically recalibrate and specialize in response to task-specific demands. Furthermore, resource utilization metrics reveal notable reductions in memory usage and training times, confirming the efficiency of the approach. Observations from qualitative analyses illustrate improved contextual coherence and logical flow in generated outputs, reinforcing the practical value of this technique. The findings collectively demonstrate the potential for Contextual Partitioning to redefine the scalability and adaptability of computational language architectures in diverse and complex domains. 

**Abstract (ZH)**: 上下文分区通过将参数动态地分割成情境感知区域，为大规模计算模型的架构设计引入了一种创新方法。该方法强调了任务特定专业化的重要性，通过与输入数据语言特征相适应的自适应参数分配机制来实现。实验评估表明，该方法在多种语言任务中显著提高了准确率、困惑度和上下文一致性，突显了该框架的适应性和可扩展性。通过减少冗余并提高计算效率，上下文分区不仅简化了模型操作，还扩展了高级语言处理系统的应用范围。该方法是自主运行的，无需外部微调，从而解决了传统参数优化技术的一个主要局限性。实验结果表明，梯度驱动的分割方法能够有效引导模型动态调整和专业化以满足任务特定需求。此外，资源利用度指标显示了内存使用量和培训时间的显著减少，进一步证实了该方法的效率。定性分析的观察结果表明，生成输出的上下文一致性和逻辑连贯性有所改善，进一步验证了该技术的实际价值。这些发现共同展示了上下文分区在各种复杂领域中重新定义计算语言架构的可扩展性和适应性的潜力。 

---
# Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback 

**Title (ZH)**: 测试时偏好优化：通过迭代文本反馈进行实时对齐 

**Authors**: Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.12895)  

**Abstract**: Large language models (LLMs) demonstrate impressive performance but lack the flexibility to adapt to human preferences quickly without retraining. In this work, we introduce Test-time Preference Optimization (TPO), a framework that aligns LLM outputs with human preferences during inference, removing the need to update model parameters. Rather than relying on purely numerical rewards, TPO translates reward signals into textual critiques and uses them as textual rewards to iteratively refine its response. Evaluations on benchmarks covering instruction following, preference alignment, safety, and mathematics reveal that TPO progressively improves alignment with human preferences. Notably, after only a few TPO steps, the initially unaligned Llama-3.1-70B-SFT model can surpass the aligned counterpart, Llama-3.1-70B-Instruct. Furthermore, TPO scales efficiently with both the search width and depth during inference. Through case studies, we illustrate how TPO exploits the innate capacity of LLM to interpret and act upon reward signals. Our findings establish TPO as a practical, lightweight alternative for test-time preference optimization, achieving alignment on the fly. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出色，但缺乏快速适应人类偏好的灵活性，这需要重新训练模型。本文引入了一种新的框架——推断时偏好优化（TPO），该框架能够使LLM在推理过程中输出结果与人类偏好保持一致，而无需更新模型参数。TPO 不仅依赖于纯粹的数值奖励，而是将奖励信号转化为文本批评，并使用这些文本批评作为反馈，逐步优化响应。在涉及指令跟随、偏好对齐、安全性和数学问题的基准测试中，TPO 的表现显示出对人类偏好的逐步改进。值得注意的是，在仅经过几次TPO步骤后，原本未对齐的Llama-3.1-70B-SFT模型可以超越已对齐的Llama-3.1-70B-Instruct模型。此外，TPO 在推理过程中的搜索宽度和深度方面具有高效的可扩展性。通过案例研究，我们证明了TPO 如何利用LLM固有的解读和响应奖励信号的能力。我们的研究结果将TPO确立为一种实用且轻量级的推断时偏好优化方法，能够在实时中实现对齐。我们已在以下网址提供代码：this https URL。 

---
# WisdomBot: Tuning Large Language Models with Artificial Intelligence Knowledge 

**Title (ZH)**: WisdomBot：使用人工智能知识调优大型语言模型 

**Authors**: Jingyuan Chen, Tao Wu, Wei Ji, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12877)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools in natural language processing (NLP), showing a promising future of artificial generated intelligence (AGI). Despite their notable performance in the general domain, LLMs have remained suboptimal in the field of education, owing to the unique challenges presented by this domain, such as the need for more specialized knowledge, the requirement for personalized learning experiences, and the necessity for concise explanations of complex concepts. To address these issues, this paper presents a novel LLM for education named WisdomBot, which combines the power of LLMs with educational theories, enabling their seamless integration into educational contexts. To be specific, we harness self-instructed knowledge concepts and instructions under the guidance of Bloom's Taxonomy as training data. To further enhance the accuracy and professionalism of model's response on factual questions, we introduce two key enhancements during inference, i.e., local knowledge base retrieval augmentation and search engine retrieval augmentation during inference. We substantiate the effectiveness of our approach by applying it to several Chinese LLMs, thereby showcasing that the fine-tuned models can generate more reliable and professional responses. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为自然语言处理（NLP）领域中强大的工具，并展现了人工智能生成（AGI）领域的光明前景。尽管在通用领域中表现出了显著的能力，但LLMs在教育领域仍存在不足，这归因于该领域独有的挑战，如需要更专门的知识、个性化学习体验的要求以及对于复杂概念的简洁解释的必要性。为了应对这些问题，本文提出了一种名为WisdomBot的新颖教育型LLM，将LLM的强大功能与教育理论相结合，使其能够无缝地融入教育环境。具体而言，我们利用基于布鲁姆分类法的自我指导的知识概念和指示进行训练。为了进一步提高模型在事实问题上回答的准确性和专业性，我们在推理过程引入了两项关键增强，即局部知识库检索增强和搜索引擎检索增强。我们通过将该方法应用于几种中文LLM，验证了其有效性，从而展示了微调模型能够生成更可靠和专业的回答。 

---
# ACEBench: Who Wins the Match Point in Tool Learning? 

**Title (ZH)**: ACEBench: 谁在工具学习中赢得(match point)最终胜利？ 

**Authors**: Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Yuefeng Huang, Xinzhi Wang, Defu Lian, Baoqun Yin, Yasheng Wang, Wu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12851)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in decision-making and reasoning, especially when combined with various tools to effectively solve complex problems. However, existing evaluation systems for assessing LLM function calling capabilities have several limitations: (1) limited evaluation scenarios, lacking assessments in real multi-turn dialogue contexts; (2) narrow evaluation dimensions, lacking detailed assessments for fine-grained function calls; (3) relying on LLMs or real API executions for result evaluation, which introduces significant overhead. To address these issues, we propose a comprehensive evaluation system named ACEBench. This system is meticulously designed to encompass a wide spectrum of function calling scenarios. Moreover, it categorizes these scenarios into three primary types according to the evaluation methodology: Normal, Special, and Agent. Normal evaluates function calls in basic scenarios; Special evaluates function calls in scenarios with vague or incomplete instructions; Agent introduces multi-agent interactions to simulate function calling evaluation in real-world multi-turn interactions. We conducted extensive experiments on ACEBench, analyzing various LLMs in-depth and performing a more granular analysis of error causes across different data types. 

**Abstract (ZH)**: 大型语言模型（LLMs）在决策和推理方面展现出显著潜力，特别是在结合各种工具后，能够有效解决复杂问题。然而，现有的评估系统在评估LLM函数调用能力方面存在一些局限性：（1）评估场景有限，缺乏对真实多轮对话环境的评估；（2）评估维度狭窄，缺乏对细粒度函数调用的详细评估；（3）评估结果依赖于LLM或真实API的执行，这引入了显著的额外开销。为解决这些问题，我们提出了一套全面的评估系统，名为ACEBench。该系统精心设计，涵盖了广泛的函数调用场景，并根据评估方法将其分为三种主要类型：正常、特殊和代理。正常类型评估在基本场景下的函数调用；特殊类型评估在模糊或不完整的指令场景下的函数调用；代理类型引入多智能体交互以模拟真实多轮对话环境中的函数调用评估。我们在ACEBench上进行了广泛的实验，深入分析了各种LLM，并对不同类型数据中的错误原因进行了更详细的分析。 

---
# Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home 

**Title (ZH)**: 没有自我知识的自适应检索？让不确定性回归 

**Authors**: Viktor Moskvoretskii, Maria Lysyuk, Mikhail Salnikov, Nikolay Ivanov, Sergey Pletenev, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2501.12835)  

**Abstract**: Retrieval Augmented Generation (RAG) improves correctness of Question Answering (QA) and addresses hallucinations in Large Language Models (LLMs), yet greatly increase computational costs. Besides, RAG is not always needed as may introduce irrelevant information. Recent adaptive retrieval methods integrate LLMs' intrinsic knowledge with external information appealing to LLM self-knowledge, but they often neglect efficiency evaluations and comparisons with uncertainty estimation techniques. We bridge this gap by conducting a comprehensive analysis of 35 adaptive retrieval methods, including 8 recent approaches and 27 uncertainty estimation techniques, across 6 datasets using 10 metrics for QA performance, self-knowledge, and efficiency. Our findings show that uncertainty estimation techniques often outperform complex pipelines in terms of efficiency and self-knowledge, while maintaining comparable QA performance. 

**Abstract (ZH)**: 检索增强生成（RAG）通过提高问答（QA）的准确性并解决大规模语言模型（LLMs）中的幻觉问题，但也大大增加了计算成本。此外，RAG 并非总有必要，因为它可能会引入无关信息。最近的自适应检索方法结合了LLMs 的内在知识与外部信息，以适应LLMs 的自知之明，但这些方法往往忽视了效率评估，并未与不确定性估计技术进行比较。我们通过基于6个数据集的综合分析来弥补这一差距，该分析涵盖了10种评估指标，对35种自适应检索方法进行了评估，其中包括8种最近的方法和技术，以及27种不确定性估计技术。我们的研究结果表明，不确定性估计技术在效率和自知之明方面通常优于复杂的工作流程，同时保持与QA性能相当的水平。 

---
# Open or Closed LLM for Lesser-Resourced Languages? Lessons from Greek 

**Title (ZH)**: 少资源语言中开放型或封闭型语言模型的选择：来自希腊语的经验教训 

**Authors**: John Pavlopoulos, Juli Bakagianni, Kanella Pouli, Maria Gavriilidou  

**Link**: [PDF](https://arxiv.org/pdf/2501.12826)  

**Abstract**: Natural Language Processing (NLP) for lesser-resourced languages faces persistent challenges, including limited datasets, inherited biases from high-resource languages, and the need for domain-specific solutions. This study addresses these gaps for Modern Greek through three key contributions. First, we evaluate the performance of open-source (Llama-70b) and closed-source (GPT-4o mini) large language models (LLMs) on seven core NLP tasks with dataset availability, revealing task-specific strengths, weaknesses, and parity in their performance. Second, we expand the scope of Greek NLP by reframing Authorship Attribution as a tool to assess potential data usage by LLMs in pre-training, with high 0-shot accuracy suggesting ethical implications for data provenance. Third, we showcase a legal NLP case study, where a Summarize, Translate, and Embed (STE) methodology outperforms the traditional TF-IDF approach for clustering \emph{long} legal texts. Together, these contributions provide a roadmap to advance NLP in lesser-resourced languages, bridging gaps in model evaluation, task innovation, and real-world impact. 

**Abstract (ZH)**: 资源较少语言的自然语言处理（NLP）面临着持续的挑战，包括数据集有限、从高资源语言继承的偏差以及需要领域特定的解决方案。本研究通过三大贡献解决了现代希腊语在这方面的不足。首先，我们评估了开源（Llama-70b）和闭源（GPT-4o mini）大型语言模型（LLMs）在七个核心NLP任务上的性能，基于数据可用性揭示了其在任务特定方面的优势、劣势及表现均衡性。其次，我们扩大了希腊语NLP的应用范围，将作者身份识别重新构想为一种工具，用以评估LLMs在预训练过程中潜在的数据使用情况，高零样本准确率提示了数据来源的伦理问题。第三，我们展示了一个法律NLP案例研究，其中总结、翻译和嵌入（STE）方法在聚类长法律文本方面优于传统的TF-IDF方法。这些贡献共同提供了一份路线图，以促进资源较少语言的NLP发展，填补了模型评估、任务创新和实际影响方面的空白。 

---
# Generation of Standardized E-Learning Contents from Digital Medical Collections 

**Title (ZH)**: 从数字医疗收藏中生成标准化的在线学习内容 

**Authors**: Felix Buendía, Joaquín Gayoso-Cabada, José-Luis Sierra  

**Link**: [PDF](https://arxiv.org/pdf/2501.12794)  

**Abstract**: In this paper, we describe an approach to transforming the huge amount of medical knowledge available in existing online medical collections into standardized learning packages ready to be integrated into the most popular e-learning platforms. The core of our approach is a tool called Clavy, which makes it possible to retrieve pieces of content in medical collections, to transform this content into meaningful learning units, and to export it in the form of standardized learning packages. In addition to describing the approach, we demonstrate its feasibility by applying it to the generation of IMS content packages from MedPix, a popular online database of medical cases in the domain of radiology. 

**Abstract (ZH)**: 在本文中，我们描述了一种将现有在线医疗资料中大量的医疗知识转化为标准化的学习包的方法，这些学习包可以被整合到最受欢迎的在线学习平台上。我们方法的核心是一个名为Clavy的工具，该工具能够从医疗资料集合中检索内容，并将其转化为有意义的学习单元，并以标准化学习包的形式导出。除了描述这种方法之外，我们还通过将其应用于从MedPix（一个流行的放射学领域在线病例数据库）生成IMS内容包来展示其可行性，MedPix是一个在线数据库，其中包含大量医学病例。 

---
# Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana 

**Title (ZH)**: 使用DataMorgana生成多样化问答基准以评估RAG系统的性能 

**Authors**: Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan, Yoelle Maarek  

**Link**: [PDF](https://arxiv.org/pdf/2501.12789)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems, especially in domain-specific contexts, requires benchmarks that address the distinctive requirements of the applicative scenario. Since real data can be hard to obtain, a common strategy is to use LLM-based methods to generate synthetic data. Existing solutions are general purpose: given a document, they generate a question to build a Q&A pair. However, although the generated questions can be individually good, they are typically not diverse enough to reasonably cover the different ways real end-users can interact with the RAG system. We introduce here DataMorgana, a tool for generating highly customizable and diverse synthetic Q&A benchmarks tailored to RAG applications. DataMorgana enables detailed configurations of user and question categories and provides control over their distribution within the benchmark. It uses a lightweight two-stage process, ensuring efficiency and fast iterations, while generating benchmarks that reflect the expected traffic. We conduct a thorough line of experiments, showing quantitatively and qualitatively that DataMorgana surpasses existing tools and approaches in producing lexically, syntactically, and semantically diverse question sets across domain-specific and general-knowledge corpora. DataMorgana will be made available to selected teams in the research community, as first beta testers, in the context of the upcoming SIGIR'2025 LiveRAG challenge to be announced in early February 2025. 

**Abstract (ZH)**: 评估检索增强生成（RAG）系统，尤其是在特定领域的背景下，需要使用能够应对应用场景独特要求的基准测试。由于真实数据可能难以获取，一个常用策略是使用基于大规模语言模型（LLM）的方法生成合成数据。现有的解决方案通常是通用的：给定一篇文档，它们生成一个问题来构建问答对。然而，尽管生成的问题单个来看可能是好的，但通常它们不够多样化，无法合理覆盖真实最终用户与RAG系统交互的各种方式。我们在此介绍DataMorgana，这是一种用于生成高度自定义和多样化合成问答基准的工具，专门针对RAG应用。DataMorgana允许详细配置用户和问题类别，并提供对基准中它们分布的控制。它使用了一个轻量级的两阶段过程，确保高效和快速迭代，同时生成反映预计流量的基准。我们进行了详尽的实验，定量和定性地展示了DataMorgana在生成跨特定领域和通用知识数据库的词汇、语法和语义多样化问题集方面超越了现有工具和方法。DataMorgana将在2025年初春SIGIR'2025 LiveRAG挑战赛中宣布，作为即将公开的第一个测试版本，提供给研究社区的部分团队进行测试。 

---
# Regularization, Semi-supervision, and Supervision for a Plausible Attention-Based Explanation 

**Title (ZH)**: 正则化、半监督学习与监督学习在可信注意力机制解释中的应用 

**Authors**: Duc Hau Nguyen, Cyrielle Mallart, Guillaume Gravier, Pascale Sébillot  

**Link**: [PDF](https://arxiv.org/pdf/2501.12775)  

**Abstract**: Attention mechanism is contributing to the majority of recent advances in machine learning for natural language processing. Additionally, it results in an attention map that shows the proportional influence of each input in its decision. Empirical studies postulate that attention maps can be provided as an explanation for model output. However, it is still questionable to ask whether this explanation helps regular people to understand and accept the model output (the plausibility of the explanation). Recent studies show that attention weights in the RNN encoders are hardly plausible because they spread on input tokens. We thus propose 3 additional constraints to the learning objective function to improve the plausibility of the attention map: regularization to increase the attention weight sparsity, semi-supervision to supervise the map by a heuristic and supervision by human annotation. Results show that all techniques can improve the attention map plausibility at some level. We also observe that specific instructions for human annotation might have a negative effect on classification performance. Beyond the attention map, the result of experiments on text classification tasks also shows that no matter how the constraint brings the gain, the contextualization layer plays a crucial role in finding the right space for finding plausible tokens. 

**Abstract (ZH)**: 注意力机制在自然语言处理的机器学习领域取得了大部分的近期进展，并且它会产生一个注意力图，以显示每个输入在决策中的相对影响力。实证研究认为，注意力图可以作为模型输出的解释。然而，仍有疑问的是这种解释是否有助于普通人在理解并接受模型输出方面提供帮助（解释的可接受性）。最近的研究表明，在循环神经网络（RNN）编码器中的注意力权重往往不够合理，因为它们在输入单元之间分布不均匀。因此，我们提出了三种额外的约束条件来改进注意力图的合理性：通过正则化增加注意力权重的稀疏性，通过启发式方法进行半监督学习，以及通过人工注释进行监督。实验结果表明，这些技术在一定程度上可以提高注意力图的合理性。我们还观察到，特定的人工注释指导可能会对分类性能产生负面影响。除了注意力图之外，文本分类任务的实验结果还显示，无论这些约束如何带来改进，在建模过程中上下文化层（contextualization layer）都起到了至关重要的作用，对于找到合理的候选词空间起到了关键作用。 

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
# EvidenceMap: Unleashing the Power of Small Language Models with Evidence Analysis for Biomedical Question Answering 

**Title (ZH)**: 证据图谱：利用证据分析释放小型语言模型在生物医学问答中的潜力 

**Authors**: Chang Zong, Jian Wan, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12746)  

**Abstract**: Current LLM-based approaches improve question answering performance by leveraging the internal reasoning abilities of models or incorporating external knowledge. However, when humans address professional problems, it is essential to explicitly analyze the multifaceted relationships from multiple pieces and diverse sources of evidence to achieve better answers. In this study, we propose a novel generative question answering framework for the biomedical domain, named EvidenceMap, which explicitly learns and incorporates evidence analysis with small language models (SLMs). The framework describes an evidence map for each question and fully utilizes an SLM to derive the representation of the supportive evaluation, the logical correlation, and the summarization of the related evidence, which facilitates an analysis-augmented generation with another SLM in an autoregressive way. Extensive experiments have shown that introducing an evidence analysis learning process can significantly outperform larger models and popular LLM reasoning methods. 

**Abstract (ZH)**: 当前基于大型语言模型（LLM）的方法通过利用模型的内部推理能力或引入外部知识来提升问答性能。然而，当人类解决专业问题时，明确分析来自多个角度和多种来源的复杂关系对于获得更好的答案至关重要。本研究提出了一种针对生物医学领域的新型生成式问答框架，名为EvidenceMap，该框架明确地利用小型语言模型（SLM）学习和整合证据分析。该框架为每个问题描述一个证据地图，并充分利用SLM来推导支持评价的表示、逻辑关联以及相关证据的总结，从而通过自回归的方式增强分析能力并进行生成。大量实验表明，引入证据分析学习过程可以显著优于较大的模型和流行的LLM推理方法。 

---
# Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression 

**Title (ZH)**: 通过AI反馈训练对话系统以提高整体对话印象 

**Authors**: Kai Yoshida, Masahiro Mizukami, Seiya Kawano, Canasai Kruengkrai, Hiroaki Sugiyama, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2501.12698)  

**Abstract**: To improve user engagement during conversations with dialogue systems, we must improve individual dialogue responses and dialogue impressions such as consistency, personality, and empathy throughout the entire dialogue. While such dialogue systems have been developing rapidly with the help of large language models (LLMs), reinforcement learning from AI feedback (RLAIF) has attracted attention to align LLM-based dialogue models for such dialogue impressions. In RLAIF, a reward model based on another LLM is used to create a training signal for an LLM-based dialogue model using zero-shot/few-shot prompting techniques. However, evaluating an entire dialogue only by prompting LLMs is challenging. In this study, the supervised fine-tuning (SFT) of LLMs prepared reward models corresponding to 12 metrics related to the impression of the entire dialogue for evaluating dialogue responses. We tuned our dialogue models using the reward model signals as feedback to improve the impression of the system. The results of automatic and human evaluations showed that tuning the dialogue model using our reward model corresponding to dialogue impression improved the evaluation of individual metrics and the naturalness of the dialogue response. 

**Abstract (ZH)**: 为了提高对话系统中用户的参与度，在整个对话过程中，我们需要改进个体对话响应以及对话印象，包括一致性、个性和同理心。借助大规模语言模型（LLMs）的支持，此类对话系统正快速开发，而基于强化学习的来自AI反馈（RLAIF）方法则引起了关注，用于引导基于LLM的对话模型以匹配这些对话印象。在RLAIF中，通过零样本/少样本提示技术，使用另一种LLM构建的奖励模型被用来为基于LLM的对话模型提供训练信号。然而，仅通过提示LLMs来评估整个对话存在挑战。在此项研究中，我们对LLMs进行了有监督微调（SFT），以便基于与12个涉及整体对话印象的指标相对应的奖励模型来评估对话响应。我们使用奖励模型信号作为反馈来改进系统的对话印象。自动和人工评估结果表明，使用与对话印象相对应的奖励模型来调整对话模型可以提升单个指标的评估结果，并提高对话响应的自然度。 

---
# Extracting General-use Transformers for Low-resource Languages via Knowledge Distillation 

**Title (ZH)**: 通过知识蒸馏提取适用于低资源语言的一般用途变换器 

**Authors**: Jan Christian Blaise Cruz, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2501.12660)  

**Abstract**: In this paper, we propose the use of simple knowledge distillation to produce smaller and more efficient single-language transformers from Massively Multilingual Transformers (MMTs) to alleviate tradeoffs associated with the use of such in low-resource settings. Using Tagalog as a case study, we show that these smaller single-language models perform on-par with strong baselines in a variety of benchmark tasks in a much more efficient manner. Furthermore, we investigate additional steps during the distillation process that improves the soft-supervision of the target language, and provide a number of analyses and ablations to show the efficacy of the proposed method. 

**Abstract (ZH)**: 在本文中，我们提出使用简单的知识蒸馏来从大规模多语言变压器（MMTs）生成更小且更高效的单语言变压器，以缓解在资源匮乏环境下使用此类模型时所面临的风险。以塔加洛语为例，我们显示这些更小的单语言模型在多种基准任务上能达到与强大基线模型相当的效果，并且以更为高效的方式。此外，我们在蒸馏过程中探索了额外步骤，以改进目标语言的软监督，并通过多种分析和消融实验展示所提出方法的有效性。 

---
# The potential -- and the pitfalls -- of using pre-trained language models as cognitive science theories 

**Title (ZH)**: 将预训练语言模型作为认知科学理论的潜力与挑战

这个翻译既保留了原文的意思，也符合学术写作的规范。其中，“pitfalls”在这里被翻译为“挑战”，在学术语境中更为常见和恰当。 

**Authors**: Raj Sanjay Shah, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2501.12651)  

**Abstract**: Many studies have evaluated the cognitive alignment of Pre-trained Language Models (PLMs), i.e., their correspondence to adult performance across a range of cognitive domains. Recently, the focus has expanded to the developmental alignment of these models: identifying phases during training where improvements in model performance track improvements in children's thinking over development. However, there are many challenges to the use of PLMs as cognitive science theories, including different architectures, different training data modalities and scales, and limited model interpretability. In this paper, we distill lessons learned from treating PLMs, not as engineering artifacts but as cognitive science and developmental science models. We review assumptions used by researchers to map measures of PLM performance to measures of human performance. We identify potential pitfalls of this approach to understanding human thinking, and we end by enumerating criteria for using PLMs as credible accounts of cognition and cognitive development. 

**Abstract (ZH)**: 许多研究评估了预训练语言模型（PLMs）的认知一致性，即它们在各种认知领域的表现与成人表现的一致性。最近，研究的重点已经扩展到这些模型的发展一致性：识别训练过程中模型性能改进与儿童思维发展改进相吻合的阶段。然而，将PLMs用作认知科学研究理论仍面临诸多挑战，包括不同的架构、不同的训练数据类型和规模，以及模型解释性的有限性。本文中，我们从将PLMs视为认知科学和发育科学模型而非工程产物的角度提炼出经验教训。我们回顾了研究人员用来将PLM性能指标映射到人类绩效指标的假设。我们指出了这种方法在理解人类思维方面的潜在陷阱，并提出了使用PLMs作为认知和认知发展可信解释的标准。 

---
# Dynamics of Toxicity in Political Podcasts 

**Title (ZH)**: 政治播客中的毒性动态 

**Authors**: Naquee Rizwan, Nayandeep Deb, Sarthak Roy, Vishwajeet Singh Solanki, Kiran Garimella, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2501.12640)  

**Abstract**: Toxicity in digital media poses significant challenges, yet little attention has been given to its dynamics within the rapidly growing medium of podcasts. This paper addresses this gap by analyzing political podcast data to study the emergence and propagation of toxicity, focusing on conversation chains-structured reply patterns within podcast transcripts. Leveraging state-of-the-art transcription models and advanced conversational analysis techniques, we systematically examine toxic discourse in over 30 popular political podcasts in the United States. Our key contributions include: (1) creating a comprehensive dataset of transcribed and diarized political podcasts, identifying thousands of toxic instances using Google's Perspective API, (2) uncovering concerning trends where a majority of episodes contain at least one toxic instance, (3) introducing toxic conversation chains and analyzing their structural and linguistic properties, revealing characteristics such as longer durations, repetitive patterns, figurative language, and emotional cues tied to anger and annoyance, (4) identifying demand-related words like 'want', 'like', and 'know' as precursors to toxicity, and (5) developing predictive models to anticipate toxicity shifts based on annotated change points. Our findings provide critical insights into podcast toxicity and establish a foundation for future research on real-time monitoring and intervention mechanisms to foster healthier discourse in this influential medium. 

**Abstract (ZH)**: 数字媒体中的毒性问题提出了重大挑战，但在快速增长的播客媒介中，对其动态的关注却相对较少。本文通过分析政治播客数据，探讨了毒性现象的出现和传播，重点关注播客转录文本中的对话链结构回复模式。借助最先进的转录模型和高级对话分析技术，我们在美国30多部热门政治播客中系统地研究了有毒话语。我们的主要贡献包括：（1）创建了一份全面的转录和分音节的政治播客数据集，利用Google的Perspective API识别了数千个有毒实例；（2）揭示了令人担忧的趋势，即大多数期集至少包含一个有毒实例；（3）引入了有毒对话链，并分析了其结构和语言特征，揭示了诸如更长的持续时间、重复模式、隐喻语言和与愤怒和不满相关的情感提示等特点；（4）识别“想要”、“喜欢”和“知道”等需求相关词汇作为毒性发生的前兆；以及（5）开发了预测模型，根据标注的时间变化点预测毒性变化。我们的研究提供了关于播客毒性的重要见解，并为未来研究实时监控和干预机制以促进这一重要媒介中的健康对话奠定了基础。 

---
# Distillation Quantification for Large Language Models 

**Title (ZH)**: 大型语言模型的蒸馏量化评估 

**Authors**: Sunbowen Lee, Junting Zhou, Chang Ao, Kaige Li, Xinrun Du, Sirui He, Jiaheng Liu, Min Yang, Zhoufutu Wen, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2501.12619)  

**Abstract**: Model distillation is a technique for transferring knowledge from large language models (LLMs) to smaller ones, aiming to create resource-efficient yet high-performing models. However, excessive distillation can lead to homogenization, reducing diversity among models and impairing their ability to robustly handle complex or novel tasks. These limitations underscore the need to systematically quantify the distillation process and its impact. In this work, we propose a framework to evaluate and quantify model distillation. Our method addresses two key aspects: (1) Identifying identity cognition contradictions to assess discrepancies in how models perceive and represent identity-related information, and (2) Analyzing multi-granularity response similarities across models to measure the extent of homogenization. Experimental results demonstrate two key insights: (1) Well-known closed-source and open-source LLMs usually exhibit high distillation degrees, except for Claude, Doubao, and Gemini. (2) Base LLMs show higher distillation degrees compared to aligned LLMs. By offering a systematic approach to improve the transparency of LLM data distillation, we call for LLMs with more independent development and more transparent technical reports to improve LLMs' robustness and safety. The code and data are available under this https URL. 

**Abstract (ZH)**: 模型蒸馏是一种将大型语言模型（LLMs）的知识转移到较小模型中的技术，旨在创建高效且高性能的模型。然而，过度的蒸馏可能导致模型同质化，减少模型之间的多样性，并削弱它们处理复杂或新颖任务的能力。这些局限性突显出系统评估和量化蒸馏过程及其影响的必要性。在本研究中，我们提出了一种评估和量化模型蒸馏的方法。该方法主要关注两个关键方面：（1）识别身份认知矛盾，以评估模型在感知和表示与身份相关的信息时的一致性差异；（2）分析模型间的多粒度响应相似性，以衡量同质化程度。实验结果揭示了两个重要洞察：（1）知名的闭源和开源LLMs通常表现出较高的蒸馏程度，但不包括Claude、Doubao和Gemini；（2）基础LLMs相比于对齐的LLMs显示出更高的蒸馏程度。通过提供一种系统的方法来提高LLM数据蒸馏的透明度，我们呼吁进行更多自主开发的LLMs并提供更透明的技术报告，以提高LLM的鲁棒性和安全性。相关代码和数据可在以下链接获取：https://example.com 

---
# T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation 

**Title (ZH)**: T2ISafety：评估图像生成中的公平性、毒性和隐私基准 

**Authors**: Lijun Li, Zhelun Shi, Xuhao Hu, Bowen Dong, Yiran Qin, Xihui Liu, Lu Sheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2501.12612)  

**Abstract**: Text-to-image (T2I) models have rapidly advanced, enabling the generation of high-quality images from text prompts across various domains. However, these models present notable safety concerns, including the risk of generating harmful, biased, or private content. Current research on assessing T2I safety remains in its early stages. While some efforts have been made to evaluate models on specific safety dimensions, many critical risks remain unexplored. To address this gap, we introduce T2ISafety, a safety benchmark that evaluates T2I models across three key domains: toxicity, fairness, and bias. We build a detailed hierarchy of 12 tasks and 44 categories based on these three domains, and meticulously collect 70K corresponding prompts. Based on this taxonomy and prompt set, we build a large-scale T2I dataset with 68K manually annotated images and train an evaluator capable of detecting critical risks that previous work has failed to identify, including risks that even ultra-large proprietary models like GPTs cannot correctly detect. We evaluate 12 prominent diffusion models on T2ISafety and reveal several concerns including persistent issues with racial fairness, a tendency to generate toxic content, and significant variation in privacy protection across the models, even with defense methods like concept erasing. Data and evaluator are released under this https URL. 

**Abstract (ZH)**: 文本到图像（T2I）模型已经迅速发展，使得能够从文本提示生成高质量的图像，覆盖各种领域。然而，这些模型存在显著的安全问题，包括生成有害、偏见或私密内容的风险。目前关于评估T2I安全性的研究仍处于初级阶段。虽然已经有一些努力在特定的安全维度上评估模型，但许多关键风险仍然未被探索。为填补这一空白，我们引入了T2ISafety，这是一个全面的安全基准，评估T2I模型在三个关键领域中的安全性：毒性、公平性和偏见。我们基于这三个领域建立了一个详细的任务层次结构，包括12个任务和44个类别，并仔细收集了70,000个相应的提示。基于这一分类和提示集，我们构建了一个包含68,000张手动标注图像的大规模T2I数据集，并训练了一个能够检测先前工作中未能识别的重要风险的评估器，包括即使是像GPTs这样的大型专用模型也无法正确检测的风险。我们对12种主要的扩散模型进行了T2ISafety评估，并揭示了几种令人担忧的问题，包括种族公平性的问题持续存在、倾向于生成有毒内容、以及在隐私保护方面模型之间存在显著差异，即使使用了如概念擦除等防御方法。数据集和评估器可以在以下链接下载：[https URL]。 

---
# BLR-MoE: Boosted Language-Routing Mixture of Experts for Domain-Robust Multilingual E2E ASR 

**Title (ZH)**: BLR-MoE：增强语言路由混合专家模型以实现领域 robust 的端到端多语言ASR 

**Authors**: Guodong Ma, Wenxuan Wang, Lifeng Zhou, Yuting Yang, Yuke Li, Binbin Du  

**Link**: [PDF](https://arxiv.org/pdf/2501.12602)  

**Abstract**: Recently, the Mixture of Expert (MoE) architecture, such as LR-MoE, is often used to alleviate the impact of language confusion on the multilingual ASR (MASR) task. However, it still faces language confusion issues, especially in mismatched domain scenarios. In this paper, we decouple language confusion in LR-MoE into confusion in self-attention and router. To alleviate the language confusion in self-attention, based on LR-MoE, we propose to apply attention-MoE architecture for MASR. In our new architecture, MoE is utilized not only on feed-forward network (FFN) but also on self-attention. In addition, to improve the robustness of the LID-based router on language confusion, we propose expert pruning and router augmentation methods. Combining the above, we get the boosted language-routing MoE (BLR-MoE) architecture. We verify the effectiveness of the proposed BLR-MoE in a 10,000-hour MASR dataset. 

**Abstract (ZH)**: 近年来，混合专家（MoE）架构，如LR-MoE，常被用于缓解多语言自动语音识别（MASR）任务中的语言混淆问题。然而，它仍然面临着语言混淆的问题，尤其是在领域不匹配的情况下。本文将LR-MoE中的语言混淆分解开，分别对自注意力和路由模块进行处理。为缓解自注意力中的语言混淆，我们基于LR-MoE提出了应用注意力-MoE架构以解决MASR问题。在我们提出的新架构中，MoE不仅用于前向网络（FFN），还用于自注意力模块。此外，为了提高基于LID的路由模块在语言混淆情况下的鲁棒性，我们提出了专家剪枝和路由增强方法。结合以上方法，我们提出了增強语言路由MoE（BLR-MoE）架构。我们还在一个包含10,000小时数据的MASR数据集上验证了所提出的BLR-MoE的有效性。 

---
# O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning 

**Title (ZH)**: O1-剪枝器：针对O1级归纳裁剪的长度协调微调 

**Authors**: Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.12570)  

**Abstract**: Recently, long-thought reasoning LLMs, such as OpenAI's O1, adopt extended reasoning processes similar to how humans ponder over complex problems. This reasoning paradigm significantly enhances the model's problem-solving abilities and has achieved promising results. However, long-thought reasoning process leads to a substantial increase in inference time. A pressing challenge is reducing the inference overhead of long-thought LLMs while ensuring accuracy. In this paper, we experimentally demonstrate that long-thought reasoning models struggle to effectively allocate token budgets based on problem difficulty and reasoning redundancies. To address this, we propose Length-Harmonizing Fine-Tuning (O1-Pruner), aiming at minimizing reasoning overhead while maintaining accuracy. This effective fine-tuning method first estimates the LLM's baseline performance through pre-sampling and then uses RL-style fine-tuning to encourage the model to generate shorter reasoning processes under accuracy constraints. This allows the model to achieve efficient reasoning with lower redundancy while maintaining accuracy. Experiments on various mathematical reasoning benchmarks show that O1-Pruner not only significantly reduces inference overhead but also achieves higher accuracy, providing a novel and promising solution to this challenge. Our code is coming soon at this https URL 

**Abstract (ZH)**: 近年来，长期被认为的推理大规模语言模型（LLM），例如OpenAI的O1，采用了一种类似人类解决复杂问题的扩展推理过程。这种推理范式显著提升了模型的解决问题能力，并取得了令人瞩目的成果。然而，长时间的推理过程导致推理时间大幅增加。一个紧迫的挑战是在保持准确性的前提下减少长期推理LLM的推理开销。在本文中，我们通过实验证明，长时间推理模型难以根据问题难度和推理冗余有效分配令牌预算。为了解决这一问题，我们提出了一种称为Length-Harmonizing Fine-Tuning（O1-Pruner）的有效微调方法，旨在在保持准确性的同时最小化推理开销。该有效微调方法首先通过预采样估计LLM的基本性能，然后使用类强化学习（RL）风格的微调方法，在准确性约束条件下鼓励模型生成更短的推理过程。这使得模型能够在降低冗余的同时保持准确性。在各种数学推理基准测试上的实验表明，O1-Pruner不仅显著减少了推理开销，还提高了准确性，为解决这一挑战提供了新颖而有前景的解决方案。我们的代码将很快发布在以下链接：[这里是链接] 

---
# Human-like conceptual representations emerge from language prediction 

**Title (ZH)**: 人类概念表示源自语言预测 

**Authors**: Ningyu Xu, Qi Zhang, Chao Du, Qiang Luo, Xipeng Qiu, Xuanjing Huang, Menghan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12547)  

**Abstract**: Recent advances in large language models (LLMs) provide a new opportunity to address the long-standing question of how concepts are represented and organized in the mind, which is central to unravelling the nature of human cognition. Here, we reframed the classic reverse dictionary task to simulate human concept inference in context and investigated the emergence of human-like conceptual representations within LLMs. We found that LLMs were able to infer concepts from definitional descriptions and construct representation spaces that converge towards a shared, context-independent structure. These representations effectively predicted human behavioural judgments and aligned well with neural activity patterns in the human brain, offering evidence for biological plausibility. These findings demonstrate that human-like conceptual representations and organization can naturally emerge from language prediction, even without real-world grounding. Our work supports the view that LLMs serve as valuable tools for understanding complex human cognition and paves the way for better alignment between artificial and human intelligence. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）的发展为解决长期以来困扰人们的关于概念在心智中如何表示和组织的问题提供了新的机会，这关乎揭示人类认知的本质。本文中，我们将经典的逆向词典任务重新构想，以模拟人类在上下文中进行概念推理，并考察LLMs中人类类似的概念表示如何自然地涌现。我们发现，LLMs能够从定义性描述中推断概念，并构建收敛于共享、跨上下文无关结构的表示空间。这些表示能够有效预测人类的行为判断，与人类大脑的神经活动模式高度一致，提供了生物学可行性证据。这些发现表明，即使没有现实世界的基础，人类类似的概念表示和组织也能自然地从语言预测中涌现。本研究支持了LLMs作为理解复杂人类认知有价值的工具的观点，并为更好地使人工智能与人类智能对齐铺平了道路。 

---
# Comparative Approaches to Sentiment Analysis Using Datasets in Major European and Arabic Languages 

**Title (ZH)**: 使用主要欧洲和阿拉伯语语言数据集的 sentiment 分析方法比较 

**Authors**: Mikhail Krasitskii, Olga Kolesnikova, Liliana Chanona Hernandez, Grigori Sidorov, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2501.12540)  

**Abstract**: This study explores transformer-based models such as BERT, mBERT, and XLM-R for multi-lingual sentiment analysis across diverse linguistic structures. Key contributions include the identification of XLM-R superior adaptability in morphologically complex languages, achieving accuracy levels above 88%. The work highlights fine-tuning strategies and emphasizes their significance for improving sentiment classification in underrepresented languages. 

**Abstract (ZH)**: 本研究探讨了基于变压器的模型，如BERT、mBERT和XLM-R在多种语言情感分析中的应用，涉及多样化的语言结构。主要贡献包括识别出XLM-R在形态复杂语言中的优越适应性，达到88%以上的准确率。研究强调了微调策略的重要性，并突出了其对于提升少有语言情感分类性能的意义。 

---
# Academic Case Reports Lack Diversity: Assessing the Presence and Diversity of Sociodemographic and Behavioral Factors related with Post COVID-19 Condition 

**Title (ZH)**: 学术案例报告缺乏多样性：评估与新冠后症状相关的社会经济与行为因素的多样性和存在情况 

**Authors**: Juan Andres Medina Florez, Shaina Raza, Rashida Lynn, Zahra Shakeri, Brendan T. Smith, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12538)  

**Abstract**: Understanding the prevalence, disparities, and symptom variations of Post COVID-19 Condition (PCC) for vulnerable populations is crucial to improving care and addressing intersecting inequities. This study aims to develop a comprehensive framework for integrating social determinants of health (SDOH) into PCC research by leveraging NLP techniques to analyze disparities and variations in SDOH representation within PCC case reports. Following construction of a PCC Case Report Corpus, comprising over 7,000 case reports from the LitCOVID repository, a subset of 709 reports were annotated with 26 core SDOH-related entity types using pre-trained named entity recognition (NER) models, human review, and data augmentation to improve quality, diversity and representation of entity types. An NLP pipeline integrating NER, natural language inference (NLI), trigram and frequency analyses was developed to extract and analyze these entities. Both encoder-only transformer models and RNN-based models were assessed for the NER objective.
Fine-tuned encoder-only BERT models outperformed traditional RNN-based models in generalizability to distinct sentence structures and greater class sparsity. Exploratory analysis revealed variability in entity richness, with prevalent entities like condition, age, and access to care, and underrepresentation of sensitive categories like race and housing status. Trigram analysis highlighted frequent co-occurrences among entities, including age, gender, and condition. The NLI objective (entailment and contradiction analysis) showed attributes like "Experienced violence or abuse" and "Has medical insurance" had high entailment rates (82.4%-80.3%), while attributes such as "Is female-identifying," "Is married," and "Has a terminal condition" exhibited high contradiction rates (70.8%-98.5%). 

**Abstract (ZH)**: 了解新型冠状病毒肺炎后综合症状（Post-COVID-19 Condition, PCC）在脆弱群体中的普遍性、差异性和症状变异对于改善护理和解决交叉不平等至关重要。本文旨在通过利用自然语言处理（NLP）技术开发一个综合框架，将社会决定因素（Social Determinants of Health, SDOH）整合到PCC研究中，以分析PCC案例报告中SDOH表示的差异和变异。通过构建一个包含来自LitCOVID存储库的超过7,000份PCC案例报告的PCC案例报告语料库，709份案例报告被注释为26种核心SDOH相关的实体类型，使用预训练的命名实体识别（NER）模型、人工审核和数据增强来提高实体类型的质量和多样性。本文还开发了一种结合了NER、自然语言推理（NLI）、三元组分析和频次分析的NLP管道，以提取和分析这些实体。两种评估了命名实体识别（NER）目标的模型，分别为仅编码器的Transformer模型和基于RNN的模型。

细调后的仅编码器的BERT模型在通用性（特别是在不同的句子结构上）和更稀疏的类别上优于传统的基于RNN的模型。探索性分析显示实体丰富性的变化，常见的实体包括病情、年龄和医疗访问情况，而种族和住房状况等敏感类别则表现出不足。三元组分析则揭示了多个实体的频繁共现，包括年龄、性别和病情。自然语言推理（NLI）目标（推导性和矛盾分析）显示，诸如“经历过暴力或虐待”和“拥有医疗保险”等属性的推导率分别为82.4%至80.3%，而诸如“自我认同为女性”、“已婚”和“患有终末期疾病”等属性的矛盾率分别为70.8%至98.5%。 

---
# Enhancing Privacy in the Early Detection of Sexual Predators Through Federated Learning and Differential Privacy 

**Title (ZH)**: 通过联邦学习和差分隐私提高性犯罪早期检测中的隐私保护 

**Authors**: Khaoula Chehbouni, Martine De Cock, Gilles Caporossi, Afaf Taik, Reihaneh Rabbany, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12537)  

**Abstract**: The increased screen time and isolation caused by the COVID-19 pandemic have led to a significant surge in cases of online grooming, which is the use of strategies by predators to lure children into sexual exploitation. Previous efforts to detect grooming in industry and academia have involved accessing and monitoring private conversations through centrally-trained models or sending private conversations to a global server. In this work, we implement a privacy-preserving pipeline for the early detection of sexual predators. We leverage federated learning and differential privacy in order to create safer online spaces for children while respecting their privacy. We investigate various privacy-preserving implementations and discuss their benefits and shortcomings. Our extensive evaluation using real-world data proves that privacy and utility can coexist with only a slight reduction in utility. 

**Abstract (ZH)**: 新冠肺炎疫情期间增加的屏幕使用时间和孤立状态导致了在线诱骗案件的显著激增，这是一种掠食者使用策略诱骗儿童进行性剥削的现象。行业和学术界之前努力检测在线诱骗涉及通过中央训练模型访问和监控私人对话，或向全球服务器发送私人对话。在本研究中，我们实现了一个保护隐私的早期发现性掠食者的管道。我们利用联邦学习和差异隐私来创建一个更安全的在线空间，同时尊重儿童的隐私。我们探讨了各种保护隐私的实现方法，并讨论了它们的优势和不足之处。使用真实数据进行的广泛评估证明，通过仅轻微降低实用性，隐私性和实用性可以共存。 

---
# A Rate-Distortion Framework for Summarization 

**Title (ZH)**: 一种摘要化率-失真框架 

**Authors**: Enes Arda, Aylin Yener  

**Link**: [PDF](https://arxiv.org/pdf/2501.13100)  

**Abstract**: This paper introduces an information-theoretic framework for text summarization. We define the summarizer rate-distortion function and show that it provides a fundamental lower bound on summarizer performance. We describe an iterative procedure, similar to Blahut-Arimoto algorithm, for computing this function. To handle real-world text datasets, we also propose a practical method that can calculate the summarizer rate-distortion function with limited data. Finally, we empirically confirm our theoretical results by comparing the summarizer rate-distortion function with the performances of different summarizers used in practice. 

**Abstract (ZH)**: 本文介绍了一种信息论框架用于文本摘要。我们定义了摘要器的率失真函数，并展示了该函数为摘要器性能提供了一个基本的下界。我们描述了一种类似于Blahut-Arimoto算法的迭代过程，用于计算此函数。为处理实际的文本数据集，我们还提出了一种实用的方法，能够在有限的数据下计算摘要器的率失真函数。最后，通过将摘要器的率失真函数与实际使用的不同摘要器的性能进行实验比较，验证了我们的理论结果。 

---
# Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at CHI through a Systematic Literature Review 

**Title (ZH)**: 理解LLM在CHI中的演化：通过系统文献综述剖析LLMs对CHI的影响 

**Authors**: Rock Yuren Pang, Hope Schroeder, Kynnedy Simone Smith, Solon Barocas, Ziang Xiao, Emily Tseng, Danielle Bragg  

**Link**: [PDF](https://arxiv.org/pdf/2501.12557)  

**Abstract**: Large language models (LLMs) have been positioned to revolutionize HCI, by reshaping not only the interfaces, design patterns, and sociotechnical systems that we study, but also the research practices we use. To-date, however, there has been little understanding of LLMs' uptake in HCI. We address this gap via a systematic literature review of 153 CHI papers from 2020-24 that engage with LLMs. We taxonomize: (1) domains where LLMs are applied; (2) roles of LLMs in HCI projects; (3) contribution types; and (4) acknowledged limitations and risks. We find LLM work in 10 diverse domains, primarily via empirical and artifact contributions. Authors use LLMs in five distinct roles, including as research tools or simulated users. Still, authors often raise validity and reproducibility concerns, and overwhelmingly study closed models. We outline opportunities to improve HCI research with and on LLMs, and provide guiding questions for researchers to consider the validity and appropriateness of LLM-related work. 

**Abstract (ZH)**: 大规模语言模型（LLMs）被定位为有望彻底改变人机交互（HCI），不仅通过重塑我们研究的界面、设计模式和社会技术系统，也通过重塑我们所使用的研究实践。然而，迄今为止，LLMs在HCI领域的应用理解尚不充分。我们通过系统文献回顾，研究了2020年至2024年间涉及LLMs的153篇CHI论文，填补这一空白。我们进行了分类分析：（1）LLMs应用的领域；（2）LLMs在HCI项目中的角色；（3）贡献类型；（4）承认的局限性和风险。我们发现LLM的工作涉及10个不同的领域，主要通过实证研究和研究成果贡献。作者们在五种不同的角色中使用LLMs，包括作为研究工具或模拟用户。然而，作者们经常提出关于有效性和可重复性的问题，并且大多数研究集中在封闭模型上。我们概述了改进与LLMs相关的HCI研究的机会，并提出了引导性问题，以帮助研究人员考虑这类工作的有效性和适宜性。 

---
# Compositional Instruction Following with Language Models and Reinforcement Learning 

**Title (ZH)**: 使用语言模型和强化学习的组合式指令跟随 

**Authors**: Vanya Cohen, Geraud Nangue Tasse, Nakul Gopalan, Steven James, Matthew Gombolay, Ray Mooney, Benjamin Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2501.12539)  

**Abstract**: Combining reinforcement learning with language grounding is challenging as the agent needs to explore the environment while simultaneously learning multiple language-conditioned tasks. To address this, we introduce a novel method: the compositionally-enabled reinforcement learning language agent (CERLLA). Our method reduces the sample complexity of tasks specified with language by leveraging compositional policy representations and a semantic parser trained using reinforcement learning and in-context learning. We evaluate our approach in an environment requiring function approximation and demonstrate compositional generalization to novel tasks. Our method significantly outperforms the previous best non-compositional baseline in terms of sample complexity on 162 tasks designed to test compositional generalization. Our model attains a higher success rate and learns in fewer steps than the non-compositional baseline. It reaches a success rate equal to an oracle policy's upper-bound performance of 92%. With the same number of environment steps, the baseline only reaches a success rate of 80%. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

将强化学习与语言关联相结合具有挑战性，因为代理需要在同时学习多个语言条件任务的过程中探索环境。为了解决这一问题，我们提出了一种新颖的方法：组合增强学习语言代理（CERLLA，Computationally-Enabled Reinforcement Learning Language Agent）。该方法通过利用组合型政策表示和使用强化学习和上下文学习训练的语义解析器，减少了由语言指定的任务所需的数据量。我们在需要函数逼近的环境中评估了该方法，并证明了其在新颖任务上的组合泛化能力。在设计用于测试组合泛化的162个任务中，我们的方法在样本复杂性方面显著优于先前最佳的非组合基准。我们的模型的成功率更高，学习步骤更少，达到了与先知策略上界性能（92%）相媲美的成功率。相比之下，基准方法仅在相同数量的环境步长中达到了80%的成功率。 

---
# The Journey Matters: Average Parameter Count over Pre-training Unifies Sparse and Dense Scaling Laws 

**Title (ZH)**: 旅程重要：预训练过程中的平均参数量统一了稀疏和密集的扩展规律 

**Authors**: Tian Jin, Ahmed Imtiaz Humayun, Utku Evci, Suvinay Subramanian, Amir Yazdanbakhsh, Dan Alistarh, Gintare Karolina Dziugaite  

**Link**: [PDF](https://arxiv.org/pdf/2501.12486)  

**Abstract**: Pruning eliminates unnecessary parameters in neural networks; it offers a promising solution to the growing computational demands of large language models (LLMs). While many focus on post-training pruning, sparse pre-training--which combines pruning and pre-training into a single phase--provides a simpler alternative. In this work, we present the first systematic exploration of optimal sparse pre-training configurations for LLMs through an examination of 80 unique pruning schedules across different sparsity levels and training durations. We find that initiating pruning at 25% of total training compute and concluding at 75% achieves near-optimal final evaluation loss. These findings provide valuable insights for efficient and effective sparse pre-training of LLMs. Furthermore, we propose a new scaling law that modifies the Chinchilla scaling law to use the average parameter count over pre-training. Through empirical and theoretical validation, we demonstrate that this modified scaling law accurately models evaluation loss for both sparsely and densely pre-trained LLMs, unifying scaling laws across pre-training paradigms. Our findings indicate that while sparse pre-training achieves the same final model quality as dense pre-training for equivalent compute budgets, it provides substantial benefits through reduced model size, enabling significant potential computational savings during inference. 

**Abstract (ZH)**: 剪枝通过消除神经网络中的冗余参数，为大型语言模型（LLMs）不断增长的计算需求提供了有希望的解决方案。尽管很多人关注剪枝后训练，但剪枝预训练——将剪枝和预训练融合为一个阶段——提供了一种更简单的替代方案。在本工作中，我们通过考察80种不同稀疏度和训练时长的独特剪枝计划，进行了第一次系统性的稀疏预训练配置探索。我们发现，在总训练计算量的25%时开始剪枝，在75%时结束剪枝，可实现近似最优的最终评估损失。这些发现为高效而有效的LLMs稀疏预训练提供了宝贵见解。此外，我们提出了一种新的标度法则，该法则修改了Chinchilla标度法则，使用预训练期间的平均参数数量。通过实证和理论验证，我们证明了这种修改后的标度法则可以准确地对稀疏和密集预训练的LLMs的评估损失进行建模，实现了预训练范式中的标度法则统一。我们的研究结果表明，尽管稀疏预训练在等效计算预算下能达到与密集预训练相同最终模型质量，但通过减少模型大小，它提供了显著的计算节省优势，特别是在推断期间。 

---
# Owls are wise and foxes are unfaithful: Uncovering animal stereotypes in vision-language models 

**Title (ZH)**: 猫头鹰明智而狐狸不忠：揭示视觉语言模型中的动物刻板印象 

**Authors**: Tabinda Aman, Mohammad Nadeem, Shahab Saquib Sohail, Mohammad Anas, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2501.12433)  

**Abstract**: Animal stereotypes are deeply embedded in human culture and language. They often shape our perceptions and expectations of various species. Our study investigates how animal stereotypes manifest in vision-language models during the task of image generation. Through targeted prompts, we explore whether DALL-E perpetuates stereotypical representations of animals, such as "owls as wise," "foxes as unfaithful," etc. Our findings reveal significant stereotyped instances where the model consistently generates images aligned with cultural biases. The current work is the first of its kind to examine animal stereotyping in vision-language models systematically and to highlight a critical yet underexplored dimension of bias in AI-generated visual content. 

**Abstract (ZH)**: 动物刻板印象深深植根于人类文化和语言之中，往往塑造我们对各种物种的感知和期望。我们的研究探讨了视觉语言模型在图像生成任务中如何体现动物刻板印象。通过有针对性的提示，我们调查DALL-E是否延续了诸如“猫头鹰代表智慧”、“狐狸代表不忠”等刻板印象的表示。我们的发现揭示了模型在文化偏见方面一致性地生成大量刻板印象图像的现象。本研究是首次系统地考察视觉语言模型中的动物刻板印象，并对AI生成视觉内容中一个尚未充分探讨的偏见维度进行了揭示。 

---
# Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation 

**Title (ZH)**: 分成后再聚合：通过并行调用工具的方法实现高效的学习工具方法 

**Authors**: Dongsheng Zhu, Weixian Shi, Zhengliang Shi, Zhaochun Ren, Shuaiqiang Wang, Lingyong Yan, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12432)  

**Abstract**: Although current Large Language Models (LLMs) exhibit impressive capabilities, performing complex real-world tasks still requires tool learning. Mainstream methods, such as CoT/ReAct, rely on step-by-step tool invocation to interact with external environments, but they are limited in perceptual scope and lack adequate task-planning capability. To address these limitations, other studies introduce the first Search-based Decision Tree (DFSDT), which still suffers from the high computational cost. In this paper, we introduce a novel parallel tool invocation paradigm, DTA-Llama (Divide-Then-Aggregate Llama). First, we transform traditional tree-based tool search paths into Directed Acyclic Graph (DAG) structure, generating a high-quality parallel tool invocation dataset. The DTA-Llama is then trained on the dataset to learn to iteratively divide the current task into several parallel tool invocation sub-tasks and aggregate the invocation results to decide the next actions. Furthermore, we introduce an efficient inference framework inspired by the Process/Threads mechanism when applying the DTA-Llama to practical tasks. Experimental results show that our approach substantially enhances task performance while reducing token consumption and inference time. Llama2-7B, using our method, is comparable to the official parallel function calling method of GPT-3.5. The relevant code, dataset, and model weights are available at this https URL 

**Abstract (ZH)**: 尽管当前的大规模语言模型（LLMs）表现出色，但执行复杂的现实世界任务仍然需要工具学习。主流方法，如CoT/ReAct，依赖于逐步调用工具并与其外部环境进行交互，但这些方法在感知范围上存在局限性，并且缺乏足够的任务规划能力。为了解决这些限制，其他研究引入了第一个基于搜索的决策树（DFSDT），但该方法仍然受到高计算成本的困扰。在本文中，我们提出了一种新的并行工具调用范式——DTA-Llama（Divide-Then-Aggregate Llama）。首先，我们将传统的树状工具搜索路径转换为有向无环图（DAG）结构，生成高质量的并行工具调用数据集。然后，我们利用该数据集训练DTA-Llama，使其学习将当前任务迭代地分解为多个并行工具调用子任务，并聚合这些调用结果以决定下一步行动。此外，我们提出了一个高效的推理框架，借鉴了在实际任务中应用DTA-Llama时的Process/线程机制。实验结果表明，我们的方法显著提升了任务性能，减少了令牌消耗和推理时间。使用我们的方法，Llama2-7B在性能上可与GPT-3.5的官方并行函数调用方法相媲美。相关代码、数据集和模型权重均可通过以下链接获取：[此处链接] 

---
# Modality Interactive Mixture-of-Experts for Fake News Detection 

**Title (ZH)**: 模ality交互混合专家系统在假新闻检测中的应用 

**Authors**: Yifan Liu, Yaokun Liu, Zelin Li, Ruichen Yao, Yang Zhang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12431)  

**Abstract**: The proliferation of fake news on social media platforms disproportionately impacts vulnerable populations, eroding trust, exacerbating inequality, and amplifying harmful narratives. Detecting fake news in multimodal contexts -- where deceptive content combines text and images -- is particularly challenging due to the nuanced interplay between modalities. Existing multimodal fake news detection methods often emphasize cross-modal consistency but ignore the complex interactions between text and visual elements, which may complement, contradict, or independently influence the predicted veracity of a post. To address these challenges, we present Modality Interactive Mixture-of-Experts for Fake News Detection (MIMoE-FND), a novel hierarchical Mixture-of-Experts framework designed to enhance multimodal fake news detection by explicitly modeling modality interactions through an interaction gating mechanism. Our approach models modality interactions by evaluating two key aspects of modality interactions: unimodal prediction agreement and semantic alignment. The hierarchical structure of MIMoE-FND allows for distinct learning pathways tailored to different fusion scenarios, adapting to the unique characteristics of each modality interaction. By tailoring fusion strategies to diverse modality interaction scenarios, MIMoE-FND provides a more robust and nuanced approach to multimodal fake news detection. We evaluate our approach on three real-world benchmarks spanning two languages, demonstrating its superior performance compared to state-of-the-art methods. By enhancing the accuracy and interpretability of fake news detection, MIMoE-FND offers a promising tool to mitigate the spread of misinformation, with the potential to better safeguard vulnerable communities against its harmful effects. 

**Abstract (ZH)**: 社交媒体平台上的假新闻泛滥，对弱势群体产生了不成比例的影响，侵蚀了信任，加剧了不平等，并放大了有害的叙事。在多模态背景下检测假新闻——其中欺骗性内容结合了文本和图像——尤为具有挑战性，因为不同模态之间的交互是微妙的。现有的多模态假新闻检测方法往往强调跨模态的一致性，而忽略了文本和视觉元素之间复杂的相互作用，这些相互作用可能会互补、矛盾或独立影响帖子的真实性的预测。为了解决这些挑战，我们提出了一种新型分层专家混合模型，用于假新闻检测（MIMoE-FND），该模型通过交互门控机制明确建模模态交互，以增强多模态假新闻检测。我们的方法通过评估模态交互的两个关键方面——单一模态预测一致性与语义对齐——来建模模态交互。MIMoE-FND 的分层结构允许根据不同的融合场景制定不同的学习路径，以适应每种模态交互的独特特征。通过根据不同模态交互场景定制融合策略，MIMoE-FND 提供了一种更稳健且细致的方法来检测多模态假新闻。我们使用两种语言的三个实际基准测试评估了该方法，结果显示其性能优于最先进的方法。通过提高假新闻检测的准确性和可解释性，MIMoE-FND 提供了一种有前景的工具来遏制虚假信息的传播，有望更好地保护弱势群体免受其有害影响。 

---
# Scopes of Alignment 

**Title (ZH)**: “Alignment Scope” 

**Authors**: Kush R. Varshney, Zahra Ashktorab, Djallel Bouneffouf, Matthew Riemer, Justin D. Weisz  

**Link**: [PDF](https://arxiv.org/pdf/2501.12405)  

**Abstract**: Much of the research focus on AI alignment seeks to align large language models and other foundation models to the context-less and generic values of helpfulness, harmlessness, and honesty. Frontier model providers also strive to align their models with these values. In this paper, we motivate why we need to move beyond such a limited conception and propose three dimensions for doing so. The first scope of alignment is competence: knowledge, skills, or behaviors the model must possess to be useful for its intended purpose. The second scope of alignment is transience: either semantic or episodic depending on the context of use. The third scope of alignment is audience: either mass, public, small-group, or dyadic. At the end of the paper, we use the proposed framework to position some technologies and workflows that go beyond prevailing notions of alignment. 

**Abstract (ZH)**: 许多关于AI对齐的研究致力于将大型语言模型和其他基础模型与无背景的通用价值准则（如帮助性、无害性和诚实性）对齐。前沿模型供应商也努力使其模型与这些价值准则保持一致。在本文中，我们阐述了为何需要超越这种有限的观点，并提出三种维度来实现这一点。首先，对齐的范畴是能力：模型必须具备的知识、技能或行为，以便实现其预期目的。其次，对齐的范畴是时间性：或者是语义上的，或者是情景性的，这取决于使用时的具体情境。第三，对齐的范畴是受众：或者是大众、公众、小型团体，或者是针对个体之间的互动。在论文结尾部分，我们将所提出的框架用于定位一些超越现有对齐观念的技术和工作流程。 

---
# FinSphere: A Conversational Stock Analysis Agent Equipped with Quantitative Tools based on Real-Time Database 

**Title (ZH)**: FinSphere：一个配有基于实时数据库的量化工具的对话式股票分析代理 

**Authors**: Shijie Han, Changhai Zhou, Yiqing Shen, Tianning Sun, Yuhua Zhou, Xiaoxia Wang, Zhixiao Yang, Jingshu Zhang, Hongguang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12399)  

**Abstract**: Current financial Large Language Models (LLMs) struggle with two critical limitations: a lack of depth in stock analysis, which impedes their ability to generate professional-grade insights, and the absence of objective evaluation metrics to assess the quality of stock analysis reports. To address these challenges, this paper introduces FinSphere, a conversational stock analysis agent, along with three major contributions: (1) Stocksis, a dataset curated by industry experts to enhance LLMs' stock analysis capabilities, (2) AnalyScore, a systematic evaluation framework for assessing stock analysis quality, and (3) FinSphere, an AI agent that can generate high-quality stock analysis reports in response to user queries. Experiments demonstrate that FinSphere achieves superior performance compared to both general and domain-specific LLMs, as well as existing agent-based systems, even when they are enhanced with real-time data access and few-shot guidance. The integrated framework, which combines real-time data feeds, quantitative tools, and an instruction-tuned LLM, yields substantial improvements in both analytical quality and practical applicability for real-world stock analysis. 

**Abstract (ZH)**: 当前的金融大型语言模型（LLMs）面临两个关键限制：在股票分析方面缺乏深度，这妨碍了它们生成专业水准见解的能力，以及缺乏客观的评价标准来评估股票分析报告的质量。为了应对这些挑战，本文介绍了FinSphere，一种对话式股票分析代理，并提出了三项主要贡献：（1）Stocksis，一个由行业专家策划的数据集，旨在增强LLMs的股票分析能力；（2）AnalyzeScore，一个系统化的评价框架，用于评估股票分析报告的质量；（3）FinSphere，一种能够根据用户查询生成高质量股票分析报告的人工智能代理。实验表明，即使在实时数据访问和少量示例引导的情况下，FinSphere 的表现优于通用和领域特定的LLMs，以及现有的基于代理的系统。该集成框架结合了实时数据流、定量工具和指令调优的LLM，显著提高了股票分析的分析质量和实际应用性。 

---
# Explainable Lane Change Prediction for Near-Crash Scenarios Using Knowledge Graph Embeddings and Retrieval Augmented Generation 

**Title (ZH)**: 使用知识图嵌入和检索增强生成进行可解释的变道预测以应对近危机场景 

**Authors**: M. Manzour, A. Ballardini, R. Izquierdo, M. Á. Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2501.11560)  

**Abstract**: Lane-changing maneuvers, particularly those executed abruptly or in risky situations, are a significant cause of road traffic accidents. However, current research mainly focuses on predicting safe lane changes. Furthermore, existing accident datasets are often based on images only and lack comprehensive sensory data. In this work, we focus on predicting risky lane changes using the CRASH dataset (our own collected dataset specifically for risky lane changes), and safe lane changes (using the HighD dataset). Then, we leverage KG and Bayesian inference to predict these maneuvers using linguistic contextual information, enhancing the model's interpretability and transparency. The model achieved a 91.5% f1-score with anticipation time extending to four seconds for risky lane changes, and a 90.0% f1-score for predicting safe lane changes with the same anticipation time. We validate our model by integrating it into a vehicle within the CARLA simulator in scenarios that involve risky lane changes. The model managed to anticipate sudden lane changes, thus providing automated vehicles with further time to plan and execute appropriate safe reactions. Finally, to enhance the explainability of our model, we utilize RAG to provide clear and natural language explanations for the given prediction. 

**Abstract (ZH)**: 换道行为，尤其是那些执行迅速或在风险较高的情况下进行的换道行为，是导致道路交通事故的重要原因之一。然而，当前的研究主要集中在预测安全的换道行为上。此外，现有的事故数据集往往仅基于图像，缺乏全面的感官数据。在本研究中，我们专注于使用CRASH数据集（我们自己收集的专门针对风险换道行为的数据集）和HighD数据集来预测风险换道行为和安全换道行为。然后，我们利用知识图谱（KG）和贝叶斯推理来利用语义上下文信息预测这些行为，这增强了模型的可解释性和透明度。该模型在预测风险换道行为时，当预见时间为四秒时，获得了91.5%的F1分数；在预测安全换道行为时，相同预见时间下的F1分数达到90.0%。我们通过将该模型集成到CARLA模拟器中的车辆中，针对涉及风险换道的场景进行了验证。模型能够预测突然的换道，从而为自动驾驶车辆提供了额外的时间来规划并执行适当的安全反应。最后，为了增强我们模型的可解释性，我们利用RAG（ rouge-adaptive generation）机制提供清晰且自然语言形式的解释，以说明给定的预测结果。 

---
