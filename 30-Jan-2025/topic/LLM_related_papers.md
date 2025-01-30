# Using Code Generation to Solve Open Instances of Combinatorial Design Problems 

**Title (ZH)**: 使用代码生成解决组合设计问题的开放实例 

**Authors**: Christopher D. Rosin  

**Link**: [PDF](https://arxiv.org/pdf/2501.17725)  

**Abstract**: The Handbook of Combinatorial Designs catalogs many types of combinatorial designs, together with lists of open instances for which existence has not yet been determined. We develop a constructive protocol CPro1, which uses Large Language Models (LLMs) to generate code that constructs combinatorial designs and resolves some of these open instances. The protocol starts from a definition of a particular type of design, and a verifier that reliably confirms whether a proposed design is valid. The LLM selects strategies and implements them in code, and scaffolding provides automated hyperparameter tuning and execution feedback using the verifier. Most generated code fails, but by generating many candidates, the protocol automates exploration of a variety of standard methods (e.g. simulated annealing, genetic algorithms) and experimentation with variations (e.g. cost functions) to find successful approaches. Testing on 16 different types of designs, CPro1 constructs solutions to open instances for 6 of them: Symmetric and Skew Weighing Matrices, Equidistant Permutation Arrays, Packing Arrays, Balanced Ternary Designs, and Florentine Rectangles. 

**Abstract (ZH)**: 《组合设计手册》记载了多种类型的组合设计，并列出了存在性尚未确定的一些开放实例。我们开发了一种构造性协议CPro1，该协议利用大型语言模型（LLMs）生成构建组合设计的代码，并解决了一些开放实例。该协议从定义特定类型的组合设计及其验证器开始，验证器能够可靠地确认所提出的组合设计的有效性。大型语言模型选择策略并在代码中实现这些策略，同时使用验证器提供自动化超参数调整和执行反馈。尽管生成的大多数代码失败，但通过生成大量候选方案，该协议自动探索了多种标准方法（例如模拟退火、遗传算法）并实验了变体（例如成本函数），从而找到成功的解决方法。在对16种不同类型的组合设计进行测试后，CPro1为其中6种设计构建了解决方案：对称和斜对称度量矩阵、等距置换数组、装载数组、平衡三元设计和佛罗伦萨矩形。 

---
# Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction 

**Title (ZH)**: 大型语言模型在单步和多步飞行轨迹预测中的应用 

**Authors**: Kaiwei Luo, Jiliu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.17459)  

**Abstract**: Flight trajectory prediction is a critical time series task in aviation. While deep learning methods have shown significant promise, the application of large language models (LLMs) to this domain remains underexplored. This study pioneers the use of LLMs for flight trajectory prediction by reframing it as a language modeling problem. Specifically, We extract features representing the aircraft's position and status from ADS-B flight data to construct a prompt-based dataset, where trajectory waypoints are converted into language tokens. The dataset is then employed to fine-tune LLMs, enabling them to learn complex spatiotemporal patterns for accurate predictions. Comprehensive experiments demonstrate that LLMs achieve notable performance improvements in both single-step and multi-step predictions compared to traditional methods, with LLaMA-3.1 model achieving the highest overall accuracy. However, the high inference latency of LLMs poses a challenge for real-time applications, underscoring the need for further research in this promising direction. 

**Abstract (ZH)**: 飞行轨迹预测是航空领域中的一个关键的时间序列任务。虽然深度学习方法已经展现出显著的潜力，但将大型语言模型（LLMs）应用于该领域仍然少有人探索。本研究率先通过重新构架为语言建模问题来利用LLMs进行飞行轨迹预测。具体来说，我们从ADS-B飞行数据中提取表示飞机位置和状态的特征，构建基于提示的数据集，其中轨迹航点被转换为语言令牌。然后使用该数据集对LLMs进行微调，使其能够学习复杂的时空模式以进行准确的预测。全面的实验表明，与传统方法相比，LLMs在单步预测和多步预测中均取得了显著的性能提升，LLaMA-3.1模型实现了最高的总体准确率。然而，LLMs的高推理延迟为实时应用带来了挑战，这突显了对该有前途方向进一步研究的必要性。 

---
# Probing LLM World Models: Enhancing Guesstimation with Wisdom of Crowds Decoding 

**Title (ZH)**: 探索大语言模型的世界模型：通过众多个体智慧增强猜测估算 

**Authors**: Yun-Shiuan Chuang, Nikunj Harlalka, Sameer Narendran, Alexander Cheung, Sizhe Gao, Siddharth Suresh, Junjie Hu, Timothy T. Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2501.17310)  

**Abstract**: Guesstimation, the task of making approximate quantity estimates, is a common real-world challenge. However, it has been largely overlooked in large language models (LLMs) and vision language models (VLMs) research. We introduce a novel guesstimation dataset, MARBLES. This dataset requires one to estimate how many items (e.g., marbles) can fit into containers (e.g., a one-cup measuring cup), both with and without accompanying images. Inspired by the social science concept of the ``{Wisdom of Crowds'' (WOC) - taking the median from estimates from a crowd), which has proven effective in guesstimation, we propose ``WOC decoding'' strategy for LLM guesstimation. We show that LLMs/VLMs perform well on guesstimation, suggesting that they possess some level of a "world model" necessary for guesstimation. Moreover, similar to human performance, the WOC decoding method improves LLM/VLM guesstimation accuracy. Furthermore, the inclusion of images in the multimodal condition enhances model performance. These results highlight the value of WOC decoding strategy for LLMs/VLMs and position guesstimation as a probe for evaluating LLMs/VLMs' world model. 

**Abstract (ZH)**: 猜测估计，即做出近似数量估计的任务，在现实生活中是一个常见的挑战。然而，这一任务在大型语言模型（LLMs）和视觉语言模型（VLMs）的研究中却鲜有涉及。我们引入了一个新型的猜测估计数据集，即MARBLES。该数据集要求参与者估算特定物品（例如珠子）可以装入容器（例如1杯容量的量杯）中的数量，既包括配有图片的情况，也包括没有图片的情况。受社会学概念“群体的智慧”（Wisdom of Crowds, WOC）的启发，WOC是指从群体的估计中选取众数，这种方法在猜测估计任务中已被证明是有效的。基于此，我们提出了一种适用于LLMs的“WOC解码”策略。我们展示了LLMs/VLMs在猜测估计任务中的良好表现，这表明这些模型具备进行猜测估计所需的一些“世界模型”。此外，类似人类的表现，“WOC解码”方法提高了LLMs/VLMs的猜测估计准确性。进一步地，多模态条件下（包含图片）的模型表现更好。这些结果突显了WOC解码策略对于LLMs/VLMs的价值，并将猜测估计视为评估LLMs/VLMs世界模型的有效探针。 

---
# From Natural Language to Extensive-Form Game Representations 

**Title (ZH)**: 从自然语言到广义形式游戏表示 

**Authors**: Shilong Deng, Yongzhao Wang, Rahul Savani  

**Link**: [PDF](https://arxiv.org/pdf/2501.17282)  

**Abstract**: We introduce a framework for translating game descriptions in natural language into extensive-form representations in game theory, leveraging Large Language Models (LLMs) and in-context learning. Given the varying levels of strategic complexity in games, such as perfect versus imperfect information, directly applying in-context learning would be insufficient. To address this, we introduce a two-stage framework with specialized modules to enhance in-context learning, enabling it to divide and conquer the problem effectively. In the first stage, we tackle the challenge of imperfect information by developing a module that identifies information sets along and the corresponding partial tree structure. With this information, the second stage leverages in-context learning alongside a self-debugging module to produce a complete extensive-form game tree represented using pygambit, the Python API of a recognized game-theoretic analysis tool called Gambit. Using this python representation enables the automation of tasks such as computing Nash equilibria directly from natural language descriptions. We evaluate the performance of the full framework, as well as its individual components, using various LLMs on games with different levels of strategic complexity. Our experimental results show that the framework significantly outperforms baseline models in generating accurate extensive-form games, with each module playing a critical role in its success. 

**Abstract (ZH)**: 我们提出了一种框架，利用大型语言模型（LLMs）和基于上下文学习的方法，将自然语言中的游戏描述转换为博弈论中的广义形式表示。由于游戏的战略复杂性各不相同，如完美信息与不完美信息之间的区别，直接应用基于上下文学习的方法是不够的。为了解决这一问题，我们引入了一个两阶段框架，其中包含专门的模块来增强基于上下文的学习能力，使其能够有效地分解并解决这一问题。在第一阶段，我们通过开发一个模块来应对不完美信息的挑战，该模块能够识别信息集及其相应的部分树结构。在获得这些信息后，第二阶段利用基于上下文学习的方法，并结合一个自调试模块，使用python API（来自开源博弈论分析工具Gambit的pygambit）生成完整的广义形式博弈树。通过这种方式表示，可以自动化直接从自然语言描述计算纳什均衡等任务。我们使用不同战略复杂度的游戏对整个框架及其各个组件进行了评估。实验结果表明，该框架在生成准确的广义形式博弈方面显著优于基准模型，每个模块都对该框架的成功起到了关键作用。 

---
# Complete Chess Games Enable LLM Become A Chess Master 

**Title (ZH)**: 完整的国际象棋完整对局使大规模语言模型成为国际象棋大师 

**Authors**: Yinqi Zhang, Xintian Han, Haolong Li, Kedi Chen, Shaohui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.17186)  

**Abstract**: Large language models (LLM) have shown remarkable abilities in text generation, question answering, language translation, reasoning and many other tasks. It continues to advance rapidly and is becoming increasingly influential in various fields, from technology and business to education and entertainment. Despite LLM's success in multiple areas, its ability to play abstract games, such as chess, is underexplored. Chess-playing requires the language models to output legal and reasonable moves from textual inputs. Here, we propose the Large language model ChessLLM to play full chess games. We transform the game into a textual format with the best move represented in the Forsyth-Edwards Notation. We show that by simply supervised fine-tuning, our model has achieved a professional-level Elo rating of 1788 in matches against the standard Elo-rated Stockfish when permitted to sample 10 times. We further show that data quality is important. Long-round data supervision enjoys a 350 Elo rating improvement over short-round data. 

**Abstract (ZH)**: 大型语言模型（LLM）在文本生成、问答、语言翻译、推理和其他许多任务中展现出了显著的能力。它不断地在快速进步，并在技术、商业、教育和娱乐等多个领域中发挥着越来越重要的作用。尽管LLM在多个领域取得了成功，但在抽象游戏，如国际象棋方面的能力却尚未得到充分探索。国际象棋游戏要求语言模型从文本输入中输出合法且合理的走法。在这里，我们提出了一种名为ChessLLM的大型语言模型，用于完整地进行国际象棋游戏。我们将游戏转换成文本格式，最合适的走法用佛音-爱德华兹符号表示。我们证明，通过简单的监督微调，我们的模型在与标准Elo评级的Stockfish进行比赛时，允许采样10次的情况下，达到了专业级别的Elo评分为1788分。进一步的研究结果表明，数据质量至关重要。长时间的监督显著提高了350分Elo评级，而短期监督的效果则较差。 

---
# BreezyVoice: Adapting TTS for Taiwanese Mandarin with Enhanced Polyphone Disambiguation -- Challenges and Insights 

**Title (ZH)**: "BreezyVoice：强化多音字消歧的台湾普通话TTS适应——挑战与见解" 

**Authors**: Chan-Jan Hsu, Yi-Cheng Lin, Chia-Chun Lin, Wei-Chih Chen, Ho Lam Chung, Chen-An Li, Yi-Chang Chen, Chien-Yu Yu, Ming-Ji Lee, Chien-Cheng Chen, Ru-Heng Huang, Hung-yi Lee, Da-Shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17790)  

**Abstract**: We present BreezyVoice, a Text-to-Speech (TTS) system specifically adapted for Taiwanese Mandarin, highlighting phonetic control abilities to address the unique challenges of polyphone disambiguation in the language. Building upon CosyVoice, we incorporate a $S^{3}$ tokenizer, a large language model (LLM), an optimal-transport conditional flow matching model (OT-CFM), and a grapheme to phoneme prediction model, to generate realistic speech that closely mimics human utterances. Our evaluation demonstrates BreezyVoice's superior performance in both general and code-switching contexts, highlighting its robustness and effectiveness in generating high-fidelity speech. Additionally, we address the challenges of generalizability in modeling long-tail speakers and polyphone disambiguation. Our approach significantly enhances performance and offers valuable insights into the workings of neural codec TTS systems. 

**Abstract (ZH)**: 我们介绍了BreezyVoice，这是一个专门为台湾 Mandarin 设计的文本到语音（TTS）系统，重点突出了其在应对语言中多音字消歧这一独特挑战方面的音素控制能力。该系统基于CosyVoice构建，集成了一个 $S^{3}$ 分词器、一个大型语言模型（LLM）、一个最优运输条件流匹配模型（OT-CFM）和一个从字符到音素的预测模型，从而生成逼真且接近人类语音的语音。评估结果表明，BreezyVoice 在一般场景和代码切换场景中均表现出优越的性能，展示了其在生成高保真语音方面的稳定性和有效性。此外，我们还解决了建模长尾群体和多音字消歧的挑战。我们的方法显著提高了性能，并为神经编解码器TTS系统的运作机制提供了宝贵的见解。 

---
# Hybrid Graphs for Table-and-Text based Question Answering using LLMs 

**Title (ZH)**: 基于LLMs的表格和文本问答的混合图模型 

**Authors**: Ankush Agarwal, Ganesh S, Chaitanya Devaguptapu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17767)  

**Abstract**: Answering questions that require reasoning and aggregation across both structured (tables) and unstructured (raw text) data sources presents significant challenges. Current methods rely on fine-tuning and high-quality, human-curated data, which is difficult to obtain. Recent advances in Large Language Models (LLMs) have shown promising results for multi-hop question answering (QA) over single-source text data in a zero-shot setting, yet exploration into multi-source Table-Text QA remains limited. In this paper, we present a novel Hybrid Graph-based approach for Table-Text QA that leverages LLMs without fine-tuning. Our method constructs a unified Hybrid Graph from textual and tabular data, pruning information based on the input question to provide the LLM with relevant context concisely. We evaluate our approach on the challenging Hybrid-QA and OTT-QA datasets using state-of-the-art LLMs, including GPT-3.5, GPT-4, and LLaMA-3. Our method achieves the best zero-shot performance on both datasets, improving Exact Match scores by up to 10% on Hybrid-QA and 5.4% on OTT-QA. Moreover, our approach reduces token usage by up to 53% compared to the original context. 

**Abstract (ZH)**: 跨结构化（表格）和非结构化（原始文本）数据源进行推理和聚合以回答问题，这一过程面临着巨大的挑战。当前的方法依赖于微调和高质量的人工标注数据，但这些数据获取起来非常困难。最近，大规模语言模型（LLMs）在零样本设置下对单源文本数据进行多跳问答方面取得了令人鼓舞的结果，但在表文多源问答方面的研究仍相对有限。在本文中，我们提出了一种新颖的基于混合图的方法来解决表文多源问答问题，该方法无需微调即可利用LLMs。我们的方法从文本和表格数据中构建一个统一的混合图，并根据输入问题筛选信息，以便为LLMs提供简洁的相关背景信息。我们使用最新的LLMs（包括GPT-3.5、GPT-4和LLaMA-3）在具有挑战性的Hybrid-QA和OTT-QA数据集上评估了我们的方法。我们的方法在两个数据集上的零样本性能最佳，分别在Hybrid-QA数据集上将精确匹配分数提高了10%，在OTT-QA数据集上提高了5.4%。此外，我们的方法相比于原始上下文将标记使用的数量减少了53%。 

---
# In-Context Meta LoRA Generation 

**Title (ZH)**: 上下文相关元LoRA生成 

**Authors**: Yihua Shao, Minxi Yan, Yang Liu, Siyu Chen, Wenjie Chen, Xinwei Long, Ziyang Yan, Lei Li, Chenyu Zhang, Nicu Sebe, Hao Tang, Yan Wang, Hao Zhao, Mengzhu Wang, Jingcai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.17635)  

**Abstract**: Low-rank Adaptation (LoRA) has demonstrated remarkable capabilities for task specific fine-tuning. However, in scenarios that involve multiple tasks, training a separate LoRA model for each one results in considerable inefficiency in terms of storage and inference. Moreover, existing parameter generation methods fail to capture the correlations among these tasks, making multi-task LoRA parameter generation challenging. To address these limitations, we propose In-Context Meta LoRA (ICM-LoRA), a novel approach that efficiently achieves task-specific customization of large language models (LLMs). Specifically, we use training data from all tasks to train a tailored generator, Conditional Variational Autoencoder (CVAE). CVAE takes task descriptions as inputs and produces task-aware LoRA weights as outputs. These LoRA weights are then merged with LLMs to create task-specialized models without the need for additional fine-tuning. Furthermore, we utilize in-context meta-learning for knowledge enhancement and task mapping, to capture the relationship between tasks and parameter distributions. As a result, our method achieves more accurate LoRA parameter generation for diverse tasks using CVAE. ICM-LoRA enables more accurate LoRA parameter reconstruction than current parameter reconstruction methods and is useful for implementing task-specific enhancements of LoRA parameters. At the same time, our method occupies 283MB, only 1\% storage compared with the original LoRA. 

**Abstract (ZH)**: 低秩适应（Low-rank Adaptation, LoRA）已经展示了在特定任务微调方面的显著能力。然而，在涉及多个任务的场景中，为每个任务单独训练一个LoRA模型会导致存储和推理效率的显著下降。此外，现有的参数生成方法无法捕捉各个任务之间的相关性，使得多任务LoRA参数生成变得具有挑战性。为了解决这些限制，我们提出了一种名为情境元LoRA（In-Context Meta LoRA, ICM-LoRA）的新型方法，该方法能够高效地实现大型语言模型（Large Language Models, LLMs）的特定任务定制。具体而言，我们使用所有任务的数据来训练一个定制生成器，即条件变分自动编码器（Conditional Variational Autoencoder, CVAE）。CVAE 以任务描述作为输入，并生成任务感知的LoRA权重作为输出。这些LoRA权重随后与LLMs结合，创建出特定任务模型，而无需额外的微调。此外，我们利用情境元学习来增强知识并优化任务映射，以捕捉任务之间的关系和参数分布。因此，我们的方法能够使用CVAE实现更准确的LoRA参数生成，适用于多种任务。ICM-LoRA 使得LoRA参数重建更加准确，并且能够有效实现LoRA参数的特定任务增强。同时，我们的方法占用空间仅为283MB，仅为原始LoRA存储空间的1%。 

---
# The Imitation Game According To Turing 

**Title (ZH)**: 图灵的模仿游戏 

**Authors**: Sharon Temtsin, Diane Proudfoot, David Kaber, Christoph Bartneck  

**Link**: [PDF](https://arxiv.org/pdf/2501.17629)  

**Abstract**: The current cycle of hype and anxiety concerning the benefits and risks to human society of Artificial Intelligence is fuelled, not only by the increasing use of generative AI and other AI tools by the general public, but also by claims made on behalf of such technology by popularizers and scientists. In particular, recent studies have claimed that Large Language Models (LLMs) can pass the Turing Test-a goal for AI since the 1950s-and therefore can "think". Large-scale impacts on society have been predicted as a result. Upon detailed examination, however, none of these studies has faithfully applied Turing's original instructions. Consequently, we conducted a rigorous Turing Test with GPT-4-Turbo that adhered closely to Turing's instructions for a three-player imitation game. We followed established scientific standards where Turing's instructions were ambiguous or missing. For example, we performed a Computer-Imitates-Human Game (CIHG) without constraining the time duration and conducted a Man-Imitates-Woman Game (MIWG) as a benchmark. All but one participant correctly identified the LLM, showing that one of today's most advanced LLMs is unable to pass a rigorous Turing Test. We conclude that recent extravagant claims for such models are unsupported, and do not warrant either optimism or concern about the social impact of thinking machines. 

**Abstract (ZH)**: 当前关于人工智能（AI）对人类社会带来的益处与风险所引发的热潮与焦虑周期，不仅受到公众日益增多地使用生成式AI和其他AI工具的影响，也被科技普及者和科学家们所做的相关宣称所推动。特别是，最近的研究声称大型语言模型（LLMs）可以通过图灵测试──自20世纪50年代以来AI领域的目标──因此被认为具有“思考”的能力。人们预测，这将对社会产生重大影响。然而，经过详细审查，这些研究中的任何一个都没有忠实执行图灵的原始指令。因此，我们使用GPT-4-Turbo进行了一个严格遵循图灵原始指令的三玩家模仿游戏，模拟图灵测试。我们遵循了图灵指令不明确或缺失时所建立的科学标准。例如，我们实施了一个计算机模仿人类的游戏（CIHG），没有限制时间，以及进行了一个男人模仿女人的游戏（MIWG）作为基准。几乎所有参与者都能正确识别出LLM，表明今天的最先进LLM无法通过严格的图灵测试。我们得出结论，近期对该类模型的夸张宣传缺乏依据，不应因为“思考机器”对社会影响的悲观或乐观情绪而产生过度关注。 

---
# CSEval: Towards Automated, Multi-Dimensional, and Reference-Free Counterspeech Evaluation using Auto-Calibrated LLMs 

**Title (ZH)**: CSEval：面向自动化、多维度且无需参考标准的反言论评估系统，利用自动标定的大语言模型 

**Authors**: Amey Hengle, Aswini Kumar, Anil Bandhakavi, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2501.17581)  

**Abstract**: Counterspeech has been popular as an effective approach to counter online hate speech, leading to increasing research interest in automated counterspeech generation using language models. However, this field lacks standardised evaluation protocols and robust automated evaluation metrics that align with human judgement. Current automatic evaluation methods, primarily based on similarity metrics, do not effectively capture the complex and independent attributes of counterspeech quality, such as contextual relevance, aggressiveness, or argumentative coherence. This has led to an increased dependency on labor-intensive human evaluations to assess automated counter-speech generation methods. To address these challenges, we introduce CSEval, a novel dataset and framework for evaluating counterspeech quality across four dimensions: contextual-relevance, aggressiveness, argument-coherence, and suitableness. Furthermore, we propose Auto-Calibrated COT for Counterspeech Evaluation (ACE), a prompt-based method with auto-calibrated chain-of-thoughts (CoT) for scoring counterspeech using large language models. Our experiments show that ACE outperforms traditional metrics like ROUGE, METEOR, and BertScore in correlating with human judgement, indicating a significant advancement in automated counterspeech evaluation. 

**Abstract (ZH)**: 对抗言论作为一种有效的方法，已经被广泛用于反击网络仇恨言论，这促进了对使用语言模型进行自动化对抗言论生成的研究兴趣增加。然而，这一领域缺乏标准化的评估协议和与人类判断相一致的稳健自动评估指标。当前的自动评估方法主要基于相似性指标，无法有效捕捉对抗言论质量的复杂且独立的属性，如上下文相关性、攻击性或说理连贯性。这导致了对劳动密集型的人工评估的依赖，以评估自动化对抗言论生成方法。为了解决这些挑战，我们提出了一种新的对抗言论评估数据集和框架CSEval，它从四个维度评估对抗言论质量：上下文相关性、攻击性、说理连贯性以及适用性。此外，我们还提出了基于提示的对抗言论评估自动校准链式思维（Auto-Calibrated Chain-of-Thoughts，简称ACE）方法，这是一种使用大型语言模型评分对抗言论的方法，结合了自动校准的链式思维（CoT）。实验结果表明，ACE在与人类判断相关性方面优于传统的ROUGE、METEOR和BertScore等指标，表明在自动化对抗言论评估方面取得了重要进展。 

---
# Is Conversational XAI All You Need? Human-AI Decision Making With a Conversational XAI Assistant 

**Title (ZH)**: 对话式解释性人工智能足够吗？对话式解释性人工智能助手赋能的人机决策制定 

**Authors**: Gaole He, Nilay Aishwarya, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2501.17546)  

**Abstract**: Explainable artificial intelligence (XAI) methods are being proposed to help interpret and understand how AI systems reach specific predictions. Inspired by prior work on conversational user interfaces, we argue that augmenting existing XAI methods with conversational user interfaces can increase user engagement and boost user understanding of the AI system. In this paper, we explored the impact of a conversational XAI interface on users' understanding of the AI system, their trust, and reliance on the AI system. In comparison to an XAI dashboard, we found that the conversational XAI interface can bring about a better understanding of the AI system among users and higher user trust. However, users of both the XAI dashboard and conversational XAI interfaces showed clear overreliance on the AI system. Enhanced conversations powered by large language model (LLM) agents amplified over-reliance. Based on our findings, we reason that the potential cause of such overreliance is the illusion of explanatory depth that is concomitant with both XAI interfaces. Our findings have important implications for designing effective conversational XAI interfaces to facilitate appropriate reliance and improve human-AI collaboration. Code can be found at this https URL 

**Abstract (ZH)**: 可解释的人工智能（XAI）方法正被提出，以帮助解释和理解人工智能系统是如何作出特定预测的。受先前关于对话式用户界面研究的启发，我们主张将对话式用户界面与现有的XAI方法结合，可以增加用户参与度并提升用户对人工智能系统的理解。在本文中，我们探讨了对话式XAI界面对用户理解人工智能系统、信任以及依赖程度的影响。与传统的XAI仪表板相比，我们发现对话式XAI界面能更好地帮助用户理解人工智能系统，并增加用户对人工智能系统的信任。然而，无论是使用XAI仪表板还是对话式XAI界面的用户都表现出对人工智能系统的过度依赖，而由大规模语言模型（LLM）代理支持的增强对话进一步加剧了这种过度依赖。基于我们的发现，我们推测导致这种过度依赖的原因与两种XAI界面都伴随的解释深度错觉有关。我们的研究结果对设计促进适当依赖并改进人类-人工智能协作的对话式XAI界面具有重要启示意义。代码可在以下链接找到：[提供链接] 

---
# LLM Assistance for Pediatric Depression 

**Title (ZH)**: 儿科抑郁症的大型语言模型辅助治疗 

**Authors**: Mariia Ignashina, Paulina Bondaronek, Dan Santel, John Pestian, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2501.17510)  

**Abstract**: Traditional depression screening methods, such as the PHQ-9, are particularly challenging for children in pediatric primary care due to practical limitations. AI has the potential to help, but the scarcity of annotated datasets in mental health, combined with the computational costs of training, highlights the need for efficient, zero-shot approaches. In this work, we investigate the feasibility of state-of-the-art LLMs for depressive symptom extraction in pediatric settings (ages 6-24). This approach aims to complement traditional screening and minimize diagnostic errors.
Our findings show that all LLMs are 60% more efficient than word match, with Flan leading in precision (average F1: 0.65, precision: 0.78), excelling in the extraction of more rare symptoms like "sleep problems" (F1: 0.92) and "self-loathing" (F1: 0.8). Phi strikes a balance between precision (0.44) and recall (0.60), performing well in categories like "Feeling depressed" (0.69) and "Weight change" (0.78). Llama 3, with the highest recall (0.90), overgeneralizes symptoms, making it less suitable for this type of analysis. Challenges include the complexity of clinical notes and overgeneralization from PHQ-9 scores. The main challenges faced by LLMs include navigating the complex structure of clinical notes with content from different times in the patient trajectory, as well as misinterpreting elevated PHQ-9 scores.
We finally demonstrate the utility of symptom annotations provided by Flan as features in an ML algorithm, which differentiates depression cases from controls with high precision of 0.78, showing a major performance boost compared to a baseline that does not use these features. 

**Abstract (ZH)**: 传统抑郁筛查方法，如PHQ-9量表，在儿科初级保健中特别具有挑战性，主要是因为实际操作上的限制。人工智能有可能帮助解决这个问题，但是心理健康领域标注数据的缺乏，以及训练模型所需的高昂计算成本，凸显了需要高效且零样本的方法。在本研究中，我们探讨了最先进的语言模型（LLMs）在儿科环境中（年龄6-24岁）提取抑郁症状的可能性，目的是补充传统的筛查方法并减少诊断错误。

我们的研究发现，所有LLMs在效率上比基于单词匹配的方法高60%，其中Flan在准确率（平均F1值：0.65，准确率：0.78）方面表现最好，特别擅长提取如“睡眠问题”（F1值：0.92）和“自虐情结”（F1值：0.8）等罕见症状。Phi则在准确率（0.44）和召回率（0.60）之间取得了平衡，在“感到抑郁”（0.69）和“体重变化”（0.78）等类别上表现良好。Llama 3虽然召回率最高（0.90），但在症状描述上过于泛化，不太适合这种类型的分析。面临的挑战包括临床笔记内容的复杂性以及PHQ-9评分的过度泛化。LLMs所面临的最大挑战包括理解和处理患者治疗历程中不同时段内容的复杂结构，以及错误解释PHQ-9评分增高的情况。

最后，我们展示了Flan提供的症状标注作为机器学习算法特征的应用，该算法能够以78%的高准确率区分抑郁病例与对照组，显示出比不使用这些特征的基线模型更高的性能提升。 

---
# DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performance 

**Title (ZH)**: DFPE：一种多元指纹ensemble方法以增强大语言模型性能 

**Authors**: Seffi Cohen, Niv Goldshlager, Nurit Cohen-Inger, Bracha Shapira, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2501.17479)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various natural language processing tasks but often struggle to excel uniformly in diverse or complex domains. We propose a novel ensemble method - Diverse Fingerprint Ensemble (DFPE), which leverages the complementary strengths of multiple LLMs to achieve more robust performance. Our approach involves: (1) clustering models based on response "fingerprints" patterns, (2) applying a quantile-based filtering mechanism to remove underperforming models at a per-subject level, and (3) assigning adaptive weights to remaining models based on their subject-wise validation accuracy. In experiments on the Massive Multitask Language Understanding (MMLU) benchmark, DFPE outperforms the best single model by 3% overall accuracy and 5% in discipline-level accuracy. This method increases the robustness and generalization of LLMs and underscores how model selection, diversity preservation, and performance-driven weighting can effectively address challenging, multi-faceted language understanding tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种自然语言处理任务中展现了出色的性能，但在面对多样或复杂的领域时，往往难以全面表现出色。我们提出了一个新颖的集成方法——不同的指纹集成（Diverse Fingerprint Ensemble，DFPE），该方法利用多个LLM的优势互补性以获得更 robust 的性能。我们的方法包括：（1）基于响应“指纹”模式对模型进行聚类；（2）应用基于分位数的过滤机制，按主题逐个去除表现不佳的模型；（3）根据主题验证准确率为剩余模型分配自适应权重。在大规模多任务语言理解（MMLU）基准上的实验表明，DFPE的整体准确率和学科准确率分别比最佳单一模型高出3%和5%。该方法增强了LLMs的稳健性和泛化能力，并强调了在模型选择、保持多样性以及基于性能赋予权重在应对复杂多面的语言理解任务中的有效作用。 

---
# Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation 

**Title (ZH)**: 病毒：规避护栏 Moderation 的有害微调攻击针对大型语言模型 

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17433)  

**Abstract**: Recent research shows that Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks -- models lose their safety alignment ability after fine-tuning on a few harmful samples. For risk mitigation, a guardrail is typically used to filter out harmful samples before fine-tuning. By designing a new red-teaming method, we in this paper show that purely relying on the moderation guardrail for data filtration is not reliable. Our proposed attack method, dubbed Virus, easily bypasses the guardrail moderation by slightly modifying the harmful data. Experimental results show that the harmful data optimized by Virus is not detectable by the guardrail with up to 100\% leakage ratio, and can simultaneously achieve superior attack performance. Finally, the key message we want to convey through this paper is that: \textbf{it is reckless to consider guardrail moderation as a clutch at straws towards harmful fine-tuning attack}, as it cannot solve the inherent safety issue of the pre-trained LLMs. Our code is available at this https URL 

**Abstract (ZH)**: 近期的研究表明，大型语言模型（LLMs）在对少量有害样本进行微调后易受到有害微调攻击——模型在其安全对齐能力方面会出现问题。为了减小风险，通常会在微调前使用一个护栏机制来过滤掉有害样本。通过设计一种新的红队方法，本文展示了仅依赖于护栏机制的数据过滤是不可靠的。我们提出了一种名为“病毒”（Virus）的攻击方法，通过稍微修改有害数据，可以轻松绕过关护护栏的过滤机制。实验结果表明，通过“病毒”优化的有害数据的漏检率可达100%，并且能够同时达到优秀的攻击性能。最后，本文的主要观点是：将护栏机制视为应对有害微调攻击的救命稻草是危险的，因为它无法解决预训练大语言模型固有的安全性问题。我们的代码可在以下链接获取：[](https://) 

---
# Actions Speak Louder than Words: Agent Decisions Reveal Implicit Biases in Language Models 

**Title (ZH)**: 言行胜于言语：智能体决策揭示语言模型中的隐含偏见 

**Authors**: Yuxuan Li, Hirokazu Shirado, Sauvik Das  

**Link**: [PDF](https://arxiv.org/pdf/2501.17420)  

**Abstract**: While advances in fairness and alignment have helped mitigate overt biases exhibited by large language models (LLMs) when explicitly prompted, we hypothesize that these models may still exhibit implicit biases when simulating human behavior. To test this hypothesis, we propose a technique to systematically uncover such biases across a broad range of sociodemographic categories by assessing decision-making disparities among agents with LLM-generated, sociodemographically-informed personas. Using our technique, we tested six LLMs across three sociodemographic groups and four decision-making scenarios. Our results show that state-of-the-art LLMs exhibit significant sociodemographic disparities in nearly all simulations, with more advanced models exhibiting greater implicit biases despite reducing explicit biases. Furthermore, when comparing our findings to real-world disparities reported in empirical studies, we find that the biases we uncovered are directionally aligned but markedly amplified. This directional alignment highlights the utility of our technique in uncovering systematic biases in LLMs rather than random variations; moreover, the presence and amplification of implicit biases emphasizes the need for novel strategies to address these biases. 

**Abstract (ZH)**: 尽管在公平性与一致性的进步有助于减轻大型语言模型（LLMs）在明确提示下表现出的明显偏见，我们假设这些模型在模拟人类行为时仍可能表现出隐性偏见。为了检验这一假设，我们提出了一种技术，该技术可以系统地揭示这些偏见，涉及广泛的社会人口统计类别，并通过评估具有社会人口统计信息的LLM生成的个性的代理在决策中的差异来进行评估。使用该技术，我们在三个社会人口统计群体和四个决策情景中测试了六种LLM。结果表明，最先进的LLM在几乎所有的模拟中都表现出显著的社会人口统计差异，尽管高级模型减少了明确的偏见，但隐性偏见却更加明显。此外，当我们将我们的发现与实证研究中报告的实际差异进行比较时，发现我们揭示的偏见方向一致，但明显放大。这种方向一致表明，我们的技术有助于揭示LLM中的系统性偏差，而不是随机变化；同时，隐性偏见的存在及其放大强调了需要采用新的策略来解决这些偏见的必要性。 

---
# General Scene Adaptation for Vision-and-Language Navigation 

**Title (ZH)**: 视觉语言导航中的通用场景适应 

**Authors**: Haodong Hong, Yanyuan Qiao, Sen Wang, Jiajun Liu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17403)  

**Abstract**: Vision-and-Language Navigation (VLN) tasks mainly evaluate agents based on one-time execution of individual instructions across multiple environments, aiming to develop agents capable of functioning in any environment in a zero-shot manner. However, real-world navigation robots often operate in persistent environments with relatively consistent physical layouts, visual observations, and language styles from instructors. Such a gap in the task setting presents an opportunity to improve VLN agents by incorporating continuous adaptation to specific environments. To better reflect these real-world conditions, we introduce GSA-VLN, a novel task requiring agents to execute navigation instructions within a specific scene and simultaneously adapt to it for improved performance over time. To evaluate the proposed task, one has to address two challenges in existing VLN datasets: the lack of OOD data, and the limited number and style diversity of instructions for each scene. Therefore, we propose a new dataset, GSA-R2R, which significantly expands the diversity and quantity of environments and instructions for the R2R dataset to evaluate agent adaptability in both ID and OOD contexts. Furthermore, we design a three-stage instruction orchestration pipeline that leverages LLMs to refine speaker-generated instructions and apply role-playing techniques to rephrase instructions into different speaking styles. This is motivated by the observation that each individual user often has consistent signatures or preferences in their instructions. We conducted extensive experiments on GSA-R2R to thoroughly evaluate our dataset and benchmark various methods. Based on our findings, we propose a novel method, GR-DUET, which incorporates memory-based navigation graphs with an environment-specific training strategy, achieving state-of-the-art results on all GSA-R2R splits. 

**Abstract (ZH)**: 视觉语言导航（VLN）任务主要评估代理在多个环境中一次性执行个别指令的能力，旨在开发能够在任何环境中以零样本方式进行工作的代理。然而，现实世界中的导航机器人通常在物理布局相对一致且视觉观察和指导语言风格也相对一致的持久环境中运行。这种任务设置之间的差距为通过持续适应特定环境来改进VLN代理提供了机会。为了更好地反映这些现实条件，我们引入了GSA-VLN，这是一种新颖的任务，要求代理在特定场景中执行导航指令并同时适应该场景，以在时间上提高性能。为了评估该任务，必须解决现有VLN数据集中两个关键挑战：OOD（out-of-distribution）数据的缺失，以及每个场景中指令数量和风格的有限多样性。因此，我们提出了一种新的数据集GSA-R2R，该数据集显著扩展了R2R数据集中的环境和指令的多样性和数量，以评估代理在ID（in-distribution）和OOD（out-of-distribution）环境中的适应性。此外，我们设计了一个三阶段指令协调流水线，利用大模型（LLM）来优化生成的指令，并运用角色扮演技术将其重新表述为不同的话语风格。这一设计受到观察的启发，即每个用户往往在指令中表现出一致的特征或偏好。我们在GSA-R2R上进行了广泛实验，以全面评估我们的数据集并基准多种方法。根据我们的研究发现，我们提出了一种新颖的方法——GR-DUET，该方法结合了基于记忆的导航图与环境特定的训练策略，实现了在所有GSA-R2R分割上的前沿结果。 

---
# MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs 

**Title (ZH)**: 多任务挑战：面向前沿大规模语言模型的现实化多轮对话评估基准 

**Authors**: Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, Ed-Yeremai Cardona, Dean Lee, Jeremy Kritz, Willow Primack, Summer Yue, Chen Xing  

**Link**: [PDF](https://arxiv.org/pdf/2501.17399)  

**Abstract**: We present MultiChallenge, a pioneering benchmark evaluating large language models (LLMs) on conducting multi-turn conversations with human users, a crucial yet underexamined capability for their applications. MultiChallenge identifies four categories of challenges in multi-turn conversations that are not only common and realistic among current human-LLM interactions, but are also challenging to all current frontier LLMs. All 4 challenges require accurate instruction-following, context allocation, and in-context reasoning at the same time. We also develop LLM as judge with instance-level rubrics to facilitate an automatic evaluation method with fair agreement with experienced human raters. Despite achieving near-perfect scores on existing multi-turn evaluation benchmarks, all frontier models have less than 50% accuracy on MultiChallenge, with the top-performing Claude 3.5 Sonnet (June 2024) achieving just a 41.4% average accuracy. 

**Abstract (ZH)**: 我们提出了MultiChallenge，这是一个开创性的基准测试，用于评估大型语言模型（LLMs）在与人类用户进行多轮对话方面的性能，这是其应用中至关重要但尚未充分评估的能力。MultiChallenge 识别了四种多轮对话中的挑战，这些挑战不仅在当前的人类-LLM 交互中普遍存在且具有现实意义，而且对当前所有前沿的LLMs 都具有挑战性。这四个挑战同时需要准确的指令遵循、上下文分配和情境中的推理能力。我们还开发了基于实例级别的评分标准的LLMs 作为裁判，以促进一种公平且与有经验的人类评审员意见一致的自动评估方法。尽管所有现有评估基准的多轮对话得分接近满分，但在MultiChallenge 中，所有前沿模型的准确率均低于50%，其中表现最佳的Anthropic Claude 3.5 Sonnet（2024年6月）的平均准确率仅为41.4%。 

---
# Inferring from Logits: Exploring Best Practices for Decoding-Free Generative Candidate Selection 

**Title (ZH)**: 从Logits推断：探索解码-free 生成候选选择的最佳实践 

**Authors**: Mingyu Derek Ma, Yanna Ding, Zijie Huang, Jianxi Gao, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.17338)  

**Abstract**: Generative Language Models rely on autoregressive decoding to produce the output sequence token by token. Many tasks such as preference optimization, require the model to produce task-level output consisting of multiple tokens directly by selecting candidates from a pool as predictions. Determining a task-level prediction from candidates using the ordinary token-level decoding mechanism is constrained by time-consuming decoding and interrupted gradients by discrete token selection. Existing works have been using decoding-free candidate selection methods to obtain candidate probability from initial output logits over vocabulary. Though these estimation methods are widely used, they are not systematically evaluated, especially on end tasks. We introduce an evaluation of a comprehensive collection of decoding-free candidate selection approaches on a comprehensive set of tasks, including five multiple-choice QA tasks with a small candidate pool and four clinical decision tasks with a massive amount of candidates, some with 10k+ options. We evaluate the estimation methods paired with a wide spectrum of foundation LMs covering different architectures, sizes and training paradigms. The results and insights from our analysis inform the future model design. 

**Abstract (ZH)**: 生成语言模型依赖于自回归解码，逐个生成输出序列的令牌。许多任务，如偏好优化，要求模型能够直接通过从候选池中选择预测候选词来生成任务级别的输出，而不是逐步生成。使用普通基于令牌级解码机制来确定任务级别的预测会受到耗时解码和离散令牌选择中断梯度的限制。现有工作已经采用了解码-free的候选选择方法，通过初始输出词汇表概率来获取候选概率。尽管这些估计方法被广泛使用，但它们没有得到系统的评估，特别是在最终任务方面。我们提出了一个全面的解码-free候选选择方法的评估，包括五个带有小候选池的多项选择问答任务和四个带有大量候选词的临床决策任务，有些任务甚至有数千个选项。我们评估了与不同架构、规模和训练范式的广泛基础LM配对的估计方法。我们分析的结果和见解将为未来的模型设计提供指导。 

---
# Memorize and Rank: Elevating Large Language Models for Clinical Diagnosis Prediction 

**Title (ZH)**: 记忆与排序：提升用于临床诊断预测的大语言模型性能 

**Authors**: Mingyu Derek Ma, Xiaoxuan Wang, Yijia Xiao, Anthony Cuturrufo, Vijay S Nori, Eran Halperin, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.17326)  

**Abstract**: Clinical diagnosis prediction models, when provided with a patient's medical history, aim to detect potential diseases early, facilitating timely intervention and improving prognostic outcomes. However, the inherent scarcity of patient data and large disease candidate space often pose challenges in developing satisfactory models for this intricate task. The exploration of leveraging Large Language Models (LLMs) for encapsulating clinical decision processes has been limited. We introduce MERA, a clinical diagnosis prediction model that bridges pertaining natural language knowledge with medical practice. We apply hierarchical contrastive learning on a disease candidate ranking list to alleviate the large decision space issue. With concept memorization through fine-tuning, we bridge the natural language clinical knowledge with medical codes. Experimental results on MIMIC-III and IV datasets show that MERA achieves the state-of-the-art diagnosis prediction performance and dramatically elevates the diagnosis prediction capabilities of generative LMs. 

**Abstract (ZH)**: 基于患者医疗历史的临床诊断预测模型旨在早期检测潜在疾病，从而促进及时干预并改善预后效果。然而，患者数据的固有稀缺性和疾病候选空间的庞大往往给开发此类复杂任务的满意模型带来了挑战。利用大型语言模型（LLMs）来封装临床决策过程的探索仍较为有限。我们引入了MERA，这是一种将相关自然语言知识与医疗实践结合起来的临床诊断预测模型。我们通过分层对比学习对疾病候选排名列表进行学习，以缓解决策空间过大的问题。通过细调进行概念记忆，我们将自然语言临床知识与医学编码相结合。在MIMIC-III和IV数据集中进行的实验结果显示，MERA实现了最先进的诊断预测性能，并显著提升了生成型语言模型的诊断预测能力。 

---
# Mitigating Hallucinated Translations in Large Language Models with Hallucination-focused Preference Optimization 

**Title (ZH)**: 使用幻觉导向的偏好优化减轻大型语言模型中的幻觉翻译 

**Authors**: Zilu Tang, Rajen Chatterjee, Sarthak Garg  

**Link**: [PDF](https://arxiv.org/pdf/2501.17295)  

**Abstract**: Machine Translation (MT) is undergoing a paradigm shift, with systems based on fine-tuned large language models (LLM) becoming increasingly competitive with traditional encoder-decoder models trained specifically for translation tasks. However, LLM-based systems are at a higher risk of generating hallucinations, which can severely undermine user's trust and safety. Most prior research on hallucination mitigation focuses on traditional MT models, with solutions that involve post-hoc mitigation - detecting hallucinated translations and re-translating them. While effective, this approach introduces additional complexity in deploying extra tools in production and also increases latency. To address these limitations, we propose a method that intrinsically learns to mitigate hallucinations during the model training phase. Specifically, we introduce a data creation framework to generate hallucination focused preference datasets. Fine-tuning LLMs on these preference datasets reduces the hallucination rate by an average of 96% across five language pairs, while preserving overall translation quality. In a zero-shot setting our approach reduces hallucinations by 89% on an average across three unseen target languages. 

**Abstract (ZH)**: 机器翻译（MT）正在经历一场范式的变革，基于微调大型语言模型（LLM）的系统在性能上逐渐与专门针对翻译任务训练的编码器-解码器模型相媲美。然而，基于LLM的系统更容易产生幻觉，这可能会严重削弱用户信任和安全性。之前大部分关于幻觉抑制的研究集中在传统MT模型上，解决方案多涉及事后抑制——检测到生成的幻觉翻译后重新翻译。尽管有效，这种方法在生产环境中部署额外工具的同时也增加了延迟。为解决这些局限性，我们提出了一种在模型训练阶段内在地学习抑制幻觉的方法。具体而言，我们引入了一个数据生成框架，用于生成专注于幻觉的偏好数据集。在这些偏好数据集上微调LLM模型，能够平均降低五种语言对中幻觉的发生率96%，同时保持整体翻译质量。在零样本设置下，我们的方法平均在三种未知目标语言中将幻觉发生率降低了89%。 

---
# Fine-Tuning Open-Source Large Language Models to Improve Their Performance on Radiation Oncology Tasks: A Feasibility Study to Investigate Their Potential Clinical Applications in Radiation Oncology 

**Title (ZH)**: 利用微调开源大规模语言模型以提升其在放射肿瘤学任务上的性能：一项可行性研究，探讨其在放射肿瘤学临床应用的潜在可能性 

**Authors**: Peilong Wang, Zhengliang Liu, Yiwei Li, Jason Holmes, Peng Shu, Lian Zhang, Xiang Li, Quanzheng Li, Brady S. Laughlin, Diego Santos Toesca, Sujay A. Vora, Samir H. Patel, Terence T. Sio, Tianming Liu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17286)  

**Abstract**: Background: The radiation oncology clinical practice involves many steps relying on the dynamic interplay of abundant text data. Large language models have displayed remarkable capabilities in processing complex text information. But their direct applications in specific fields like radiation oncology remain underexplored.
Purpose: This study aims to investigate whether fine-tuning LLMs with domain knowledge can improve the performance on Task (1) treatment regimen generation, Task (2) treatment modality selection (photon, proton, electron, or brachytherapy), and Task (3) ICD-10 code prediction in radiation oncology.
Methods: Data for 15,724 patient cases were extracted. Cases where patients had a single diagnostic record, and a clearly identifiable primary treatment plan were selected for preprocessing and manual annotation to have 7,903 cases of the patient diagnosis, treatment plan, treatment modality, and ICD-10 code. Each case was used to construct a pair consisting of patient diagnostics details and an answer (treatment regimen, treatment modality, or ICD-10 code respectively) for the supervised fine-tuning of these three tasks. Open source LLaMA2-7B and Mistral-7B models were utilized for the fine-tuning with the Low-Rank Approximations method. Accuracy and ROUGE-1 score were reported for the fine-tuned models and original models. Clinical evaluation was performed on Task (1) by radiation oncologists, while precision, recall, and F-1 score were evaluated for Task (2) and (3). One-sided Wilcoxon signed-rank tests were used to statistically analyze the results.
Results: Fine-tuned LLMs outperformed original LLMs across all tasks with p-value <= 0.001. Clinical evaluation demonstrated that over 60% of the fine-tuned LLMs-generated treatment regimens were clinically acceptable. Precision, recall, and F1-score showed improved performance of fine-tuned LLMs. 

**Abstract (ZH)**: 背景：放射肿瘤学临床实践涉及多个步骤，这些步骤依赖于大量文本数据的动态互动。大规模语言模型在处理复杂文本信息方面表现出显著的能力，但在像放射肿瘤学这样的特定领域中的直接应用依然被广泛探索。
目的：本研究旨在探讨通过结合专业领域的知识微调大规模语言模型是否能在治疗方案生成（任务1）、治疗方式选择（光子、质子、电子或近距离放射治疗）（任务2）以及ICD-10编码预测（任务3）这些放射肿瘤学任务上取得更好的性能。
方法：共提取了15,724例患者的病例数据。筛选出具有单一诊断记录且治疗计划明确的病例进行预处理和人工标注，最终获得7,903例包含诊断、治疗计划、治疗方式和ICD-10编码的患者病例。每例病例用于构建用于监督微调这三个任务的数据对（包含患者诊断细节和答案，分别对应治疗方案、治疗方式或ICD-10编码）。利用开源的LLaMA2-7B和Mistral-7B模型进行微调，采用低秩近似方法。报告了微调模型和原始模型的准确率和ROUGE-1分数。对于任务1，由放射肿瘤学专家进行了临床评估；对于任务2和任务3，则分别评估了精确度、召回率和F-1分数。采用单边Wilcoxon符号秩检验对结果进行了统计分析。
结果：微调后的大规模语言模型在所有任务上均优于原始模型，p值小于或等于0.001。临床评估显示，超过60%的微调后模型生成的治疗方案是临床可接受的。精确度、召回率和F-1分数表明微调后的大规模语言模型性能有所提升。 

---
# Improving LLM Leaderboards with Psychometrical Methodology 

**Title (ZH)**: 使用心理测量方法提高大型语言模型排行榜效果 

**Authors**: Denis Federiakin  

**Link**: [PDF](https://arxiv.org/pdf/2501.17200)  

**Abstract**: The rapid development of large language models (LLMs) has necessitated the creation of benchmarks to evaluate their performance. These benchmarks resemble human tests and surveys, as they consist of sets of questions designed to measure emergent properties in the cognitive behavior of these systems. However, unlike the well-defined traits and abilities studied in social sciences, the properties measured by these benchmarks are often vaguer and less rigorously defined. The most prominent benchmarks are often grouped into leaderboards for convenience, aggregating performance metrics and enabling comparisons between models. Unfortunately, these leaderboards typically rely on simplistic aggregation methods, such as taking the average score across benchmarks. In this paper, we demonstrate the advantages of applying contemporary psychometric methodologies - originally developed for human tests and surveys - to improve the ranking of large language models on leaderboards. Using data from the Hugging Face Leaderboard as an example, we compare the results of the conventional naive ranking approach with a psychometrically informed ranking. The findings highlight the benefits of adopting psychometric techniques for more robust and meaningful evaluation of LLM performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速发展促使我们创建基准来评估其性能。这些基准类似于人类测试和调查，它们由一系列旨在衡量这些系统认知行为中涌现属性的问题组成。然而，与社会科学中已经明确定义的特质和能力不同，这些基准所测量的属性通常较为模糊且定义不够严谨。最突出的基准通常出于便利性而被分组到排行榜中，汇总性能指标并使模型之间的比较成为可能。不幸的是，这些排行榜通常依赖于简单的汇总方法，例如取基准的平均分。在本文中，我们展示了将现代心理测量学方法应用于改进大型语言模型排行榜的优势——这些方法最初是为人类测试和调查开发的。使用Hugging Face排行榜的数据为例，我们将传统简单的排名方法与基于心理测量学的方法进行比较。研究结果突显了采用心理测量技术对大型语言模型性能进行更为稳健和有意义评估的优势。 

---
# LLM Evaluation Based on Aerospace Manufacturing Expertise: Automated Generation and Multi-Model Question Answering 

**Title (ZH)**: 基于航空航天制造专业知识的大型语言模型评估：自动化生成与多模型问答 

**Authors**: Beiming Liu, Zhizhuo Cui, Siteng Hu, Xiaohua Li, Haifeng Lin, Zhengxin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.17183)  

**Abstract**: Aerospace manufacturing demands exceptionally high precision in technical parameters. The remarkable performance of Large Language Models (LLMs), such as GPT-4 and QWen, in Natural Language Processing has sparked industry interest in their application to tasks including process design, material selection, and tool information retrieval. However, LLMs are prone to generating "hallucinations" in specialized domains, producing inaccurate or false information that poses significant risks to the quality of aerospace products and flight safety. This paper introduces a set of evaluation metrics tailored for LLMs in aerospace manufacturing, aiming to assess their accuracy by analyzing their performance in answering questions grounded in professional knowledge. Firstly, key information is extracted through in-depth textual analysis of classic aerospace manufacturing textbooks and guidelines. Subsequently, utilizing LLM generation techniques, we meticulously construct multiple-choice questions with multiple correct answers of varying difficulty. Following this, different LLM models are employed to answer these questions, and their accuracy is recorded. Experimental results demonstrate that the capabilities of LLMs in aerospace professional knowledge are in urgent need of improvement. This study provides a theoretical foundation and practical guidance for the application of LLMs in aerospace manufacturing, addressing a critical gap in the field. 

**Abstract (ZH)**: 航空航天制造对技术参数的要求极为严格。大型语言模型（LLMs），如GPT-4和Qwen，在自然语言处理领域的显著性能吸引了业界对其在工艺设计、材料选择和工具信息检索等方面应用的兴趣。然而，LLMs 在专业领域内容易产生“幻觉”，即生成不准确或虚假的信息，这对航空航天产品质量和飞行安全构成了重大风险。本文旨在为LLMs在航空航天制造中的应用提供一套定制化的评估指标，通过分析其在基于专业知识的问题回答中的表现来评估其准确性。首先，通过深入分析经典航空航天制造教科书和指导手册提取关键信息。其次，利用LLMs生成技术，精心构建了包含多个正确答案且难度不一的选择题。然后，使用不同的LLM模型回答这些问题，并记录其准确性。实验结果表明，LLMs在航空航天专业知识方面的能力亟待提升。本研究为LLMs在航空航天制造中的应用提供了理论基础和实践指导，填补了该领域的关键空白。 

---
# Tuning LLM Judges Hyperparameters 

**Title (ZH)**: 调整大语言模型法官的超参数 

**Authors**: David Salinas, Omar Swelam, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2501.17178)  

**Abstract**: Evaluating Large Language Models (LLMs) often requires costly human annotations. To address this, LLM-based judges have been proposed, which compare the outputs of two LLMs enabling the ranking of models without human intervention. While several approaches have been proposed, many confounding factors are present between different papers. For instance the model, the prompt and other hyperparameters are typically changed at the same time making apple-to-apple comparisons challenging. In this paper, we propose to systematically analyze and tune hyperparameter of LLM judges. To alleviate the high cost of evaluating a judge, we propose to leverage multi-objective multi-fidelity which allows to find judges that trades accuracy for cost and also reduce significantly the cost of the search. Our method identifies judges that not only outperform existing benchmarks in accuracy and cost-efficiency but also utilize open-weight models, ensuring greater accessibility and reproducibility. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）通常需要成本高昂的人工注释。为了解决这一问题，提出了基于LLM的裁判系统，这种系统可以通过比较两个LLM的输出来对模型进行排名，而无需人工干预。尽管已经提出了一些方法，但不同论文之间存在诸多混淆因素。例如，模型、提示和超参数通常同时变化，这使得直接对比变得困难。本文中，我们提出了一种系统的方法来分析和调整LLM裁判的超参数。为了减轻评估裁判的成本，我们提出了一种多目标多保真度方法，该方法可以在准确性和成本之间进行权衡，同时也大大降低了搜索成本。我们的方法不仅在准确性和成本效益上超越了现有基准，还利用了开放权重模型，从而确保了更大的可访问性和可重复性。 

---
# Prompt-Based Cost-Effective Evaluation and Operation of ChatGPT as a Computer Programming Teaching Assistant 

**Title (ZH)**: 基于提示的经济高效评估与运行：将ChatGPT作为计算机编程辅导助手的应用研究 

**Authors**: Marc Ballestero-Ribó, Daniel Ortiz-Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2501.17176)  

**Abstract**: The dream of achieving a student-teacher ratio of 1:1 is closer than ever thanks to the emergence of large language models (LLMs). One potential application of these models in the educational field would be to provide feedback to students in university introductory programming courses, so that a student struggling to solve a basic implementation problem could seek help from an LLM available 24/7. This article focuses on studying three aspects related to such an application. First, the performance of two well-known models, GPT-3.5T and GPT-4T, in providing feedback to students is evaluated. The empirical results showed that GPT-4T performs much better than GPT-3.5T, however, it is not yet ready for use in a real-world scenario. This is due to the possibility of generating incorrect information that potential users may not always be able to detect. Second, the article proposes a carefully designed prompt using in-context learning techniques that allows automating important parts of the evaluation process, as well as providing a lower bound for the fraction of feedbacks containing incorrect information, saving time and effort. This was possible because the resulting feedback has a programmatically analyzable structure that incorporates diagnostic information about the LLM's performance in solving the requested task. Third, the article also suggests a possible strategy for implementing a practical learning tool based on LLMs, which is rooted on the proposed prompting techniques. This strategy opens up a whole range of interesting possibilities from a pedagogical perspective. 

**Abstract (ZH)**: 实现师生比1:1的梦想比以往任何时候都更加接近，得益于大型语言模型（LLMs）的出现。这些模型在教育领域的潜在应用之一是为大学入门编程课程的学生提供反馈，使得在编程中遇到基本问题的学生能够随时从不断线的LLM寻求帮助。本文主要研究了这一应用的三个方面。首先，评估了两个知名模型GPT-3.5T和GPT-4T为学生提供反馈的表现。实验证明，GPT-4T的表现明显优于GPT-3.5T，但尚不适用于实际场景使用。原因是生成错误信息的可能性使得潜在用户不一定能够察觉。其次，文章提出了一种巧妙设计的提示，利用上下文学习技术，实现了评估过程中的重要部分自动化，并为含有错误信息的反馈比例设定了下限，从而节省时间和精力。这得益于反馈具有可编程分析的结构，其中包含了LLM在完成指定任务时的诊断信息。最后，本文还提出了一种基于LLM实现实际学习工具的可能策略，该策略源于所提出的提示技术。这一策略从教育角度来看，为了一系列有趣的可能性打开了大门。 

---
# QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks 

**Title (ZH)**: QualityFlow：由LLM质量检查控制的代理工作流程程序合成 

**Authors**: Yaojie Hu, Qiang Zhou, Qihong Chen, Xiaopeng Li, Linbo Liu, Dejiao Zhang, Amit Kachroo, Talha Oz, Omer Tripp  

**Link**: [PDF](https://arxiv.org/pdf/2501.17167)  

**Abstract**: We introduce QualityFlow, a dynamic agentic workflow for program synthesis. Given the English description of a programming problem and a set of unit tests, the model's goal is to synthesize the correct program that solves the problem and passes the tests. QualityFlow consists of multiple large language model (LLM) agents that resemble a software development team, including code generation, testing, and self-debugging. Existing program synthesis methods face three major limitations: assumption of visible unit test conformity, bottleneck of synthesized test quality, and deviation of self-debugging trajectory. To address them, we propose the LLM Quality Checker, which explicitly "imagines" whether the synthesized programs' execution would conform to the unit tests. The Quality Checks dynamically control the workflow, including actions to submit the final answer, clarify the problem statement, and revert previous workflow steps. As a result, our Quality Checker can precisely accept any correct program, mitigate faulty synthesized tests, and prevent potential workflow deviation. The success of the Quality Checker further enables Diversified Prompting, which encourages variations in LLM responses to maximize the possibility that a correct program appears and passes the quality check. In experiments, QualityFlow establishes the state-of-the-art results on four program synthesis benchmarks: MBPP, HumanEval, and the stricter evaluations of both MBPP and HumanEval from EvalPlus. Our systematic analysis shows that the dynamic workflow controlled by LLM quality checks can outperform static workflows and single-attempt zero-shot synthesis. The Quality Checker is the center of our investigation, and we dissect its individual performance and integrated impact on the workflow accuracy, as well as other ablations experiments to justify our workflow design. 

**Abstract (ZH)**: 我们引入了QualityFlow，这是一种动态代理工作流，用于程序合成。给定编程问题的英文描述和一组单元测试，模型的目标是合成正确的程序以解决问题并通过这些测试。QualityFlow 包含多个大的语言模型（LLM）代理，这些代理类似于软件开发团队，包括代码生成、测试和自我调试。现有的程序合成方法面临三大限制：单元测试一致性可见性的假设、合成测试质量的瓶颈以及自我调试轨迹的偏差。为了解决这些问题，我们提出了一种LLM质量检查器，明确“设想”合成的程序执行是否符合单元测试。质量检查动态控制工作流程，包括提交最终答案、澄清问题描述以及回滚先前的工作流程步骤。因此，我们的质量检查器可以精确接受任何正确的程序，减轻错误的合成测试，并防止潜在的工作流程偏移。质量检查的成功进一步使多样化提示成为可能，鼓励LLM响应的变化以最大化正确程序出现并通过质量检查的可能性。在实验中，QualityFlow 在四个程序合成基准测试（MBPP、HumanEval 及从EvalPlus的更严格的评估）中达到了最先进的结果。我们的系统分析表明，由LLM质量检查控制的动态工作流程可以优于静态工作流程和单一尝试的零次合成。质量检查器是我们研究的核心，我们剖析了其单独的性能和综合影响对工作流程准确性的影响，并进行了其他消融实验以证明我们的工作流程设计的有效性。 

---
# Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate 

**Title (ZH)**: 批判性调优：学习批判比学习模仿更有效 

**Authors**: Yubo Wang, Xiang Yue, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.17703)  

**Abstract**: Supervised Fine-Tuning (SFT) is commonly used to train language models to imitate annotated responses for given instructions. In this paper, we challenge this paradigm and propose Critique Fine-Tuning (CFT), a strategy where models learn to critique noisy responses rather than simply imitate correct ones. Inspired by human learning processes that emphasize critical thinking, CFT encourages deeper analysis and nuanced understanding-traits often overlooked by standard SFT. To validate the effectiveness of CFT, we construct a 50K-sample dataset from WebInstruct, using GPT-4o as the teacher to generate critiques in the form of (input=[query; noisy response], output=critique). CFT on this dataset yields a consistent 4-10% improvement over SFT on six math benchmarks with different base models like Qwen2.5, Qwen2.5-Math and DeepSeek-Math. We further expand to MetaMath and NuminaMath datasets and observe similar gains over SFT. Notably, our Qwen2.5-Math-CFT model-trained on just 50K samples-matches or outperforms competitive models such as AceMath and Qwen2.5-Math-Instruct on most benchmarks, both of which use over 2M samples. Ablation studies show that CFT is robust to the source of noisy response and teacher critique model. Through these findings, we argue that critique-based training offers a more effective alternative to advance the reasoning of language models. 

**Abstract (ZH)**: 监督微调（SFT）常用于训练语言模型以模仿给定指令下的标注响应。本文挑战这一范式，提出了一种新的策略——批判性微调（CFT），通过这种策略，模型学会批判性地评估噪声响应，而不是简单地模仿正确响应。受到人类学习过程中强调批判性思考的启发，CFT 鼓励进行更深层次的分析和细致的理解，这是标准 SFT 经常忽视的特质。为了验证 CFT 的有效性，我们从 WebInstruct 构建了一个包含 50,000 个样本的数据集，并使用 GPT-4o 作为教师生成形如 (输入=[查询；噪声响应]，输出=批评) 的批评。在这一数据集上进行 CFT 微调后，我们的模型在六个不同基础模型（如 Qwen2.5、Qwen2.5-Math 和 DeepSeek-Math）的数学基准测试中均取得了 4-10% 的持续改进。我们进一步扩展到 MetaMath 和 NuminaMath 数据集，并观察到类似的改进。值得注意的是，仅使用 50,000 个样本训练的 Qwen2.5-Math-CFT 模型在大多数基准测试中与使用超过 2 百万样本训练的 AceMath 和 Qwen2.5-Math-Instruct 等竞争模型相当或优于后者。消融实验表明，CFT 对噪声响应的来源以及教师批评模型的选择具有鲁棒性。通过这些研究结果，我们认为基于批评的训练为提升语言模型的推理能力提供了更有效的方法。 

---
# Structured Context Recomposition for Large Language Models Using Probabilistic Layer Realignment 

**Title (ZH)**: 使用概率层重排进行大规模语言模型的结构化上下文重组 

**Authors**: Jonathan Teel, Jocasta Cumberbatch, Raphael Benington, Quentin Baskerville  

**Link**: [PDF](https://arxiv.org/pdf/2501.17617)  

**Abstract**: Extended sequence generation often leads to degradation in contextual consistency due to the inability of conventional self-attention mechanisms to effectively retain long-range dependencies. Existing approaches, including memory compression and retrieval-augmented conditioning, introduce computational trade-offs that either increase inference latency or impose additional storage overhead. Structured Context Recomposition (SCR) introduces a probabilistic layer realignment strategy that dynamically adjusts learned representations within transformer layers, ensuring that semantically relevant embeddings persist throughout extended transformations. The proposed method enhances coherence retention through a recursive weighting function that redistributes representational emphasis based on inferred contextual relevance rather than relying on fixed token-level attention scores. Empirical results indicate that probabilistic realignment mitigates abrupt topic shifts and logical inconsistencies, particularly in scenarios where sequences exceed standard attention window constraints. Sequence-level entropy analysis further reveals that SCR moderates representational variability without introducing excessive output regularization, allowing models to sustain generative diversity while preserving contextual alignment. Attention head deviation measurements confirm that hierarchical reweighting contributes to smoother token dependency transitions across transformer layers, reinforcing the stability of multi-turn interactions and document-level reasoning. Computational resource assessments show that while SCR incurs a moderate increase in processing time, memory overhead remains within feasible limits, making it suitable for practical deployment in autoregressive generative applications. 

**Abstract (ZH)**: 扩展序列生成常常会因常规自注意力机制难以有效保留长范围依赖关系而导致语境一致性退化。现有方法，包括内存压缩和检索增强条件化，引入了计算折衷，要么增加推理延迟，要么增加额外的存储开销。结构化上下文重构（SCR）引入了一种基于概率的层重新对齐策略，该策略动态调整transformer层中的学习表示，确保语义相关的嵌入在整个扩展变换过程中持久存在。所提出的方法通过递归加权函数增强了连贯性的保持，该函数基于推断出的上下文相关性重新分配表示性的重点，而不是依赖于固定token级别的注意力评分。实验证据表明，概率重新对齐可以减轻主题突变和逻辑不一致现象，尤其是在序列超过标准注意力窗口约束的情况下。序列级熵分析进一步表明，SCR在不引入过度输出正则化的情况下调节表示性变异性，从而使模型在保持上下文对齐的同时维持生成多样。注意力头偏差测量证实，层次加权有助于transformer层内更平滑的token依赖性过渡，增强多轮对话和文档级推理的稳定性。计算资源评估显示，尽管SCR会导致处理时间的适度增加，但内存开销仍控制在可行范围内，使其适合在自回归生成应用中的实际部署。 

---
# Semantic Consistency Regularization with Large Language Models for Semi-supervised Sentiment Analysis 

**Title (ZH)**: 使用大型语言模型进行半监督情感分析的语义一致性正则化 

**Authors**: Kunrong Li, Xinyu Liu, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.17598)  

**Abstract**: Accurate sentiment analysis of texts is crucial for a variety of applications, such as understanding customer feedback, monitoring market trends, and detecting public sentiment. However, manually annotating large sentiment corpora for supervised learning is labor-intensive and time-consuming. Therefore, it is essential and effective to develop a semi-supervised method for the sentiment analysis task. Although some methods have been proposed for semi-supervised text classification, they rely on the intrinsic information within the unlabeled data and the learning capability of the NLP model, which lack generalization ability to the sentiment analysis scenario and may prone to overfit. Inspired by the ability of pretrained Large Language Models (LLMs) in following instructions and generating coherent text, we propose a Semantic Consistency Regularization with Large Language Models (SCR) framework for semi-supervised sentiment analysis. We introduce two prompting strategies to semantically enhance unlabeled text using LLMs. The first is Entity-based Enhancement (SCR-EE), which involves extracting entities and numerical information, and querying the LLM to reconstruct the textual information. The second is Concept-based Enhancement (SCR-CE), which directly queries the LLM with the original sentence for semantic reconstruction. Subsequently, the LLM-augmented data is utilized for a consistency loss with confidence thresholding, which preserves high-quality agreement samples to provide additional supervision signals during training. Furthermore, to fully utilize the uncertain unlabeled data samples, we propose a class re-assembling strategy inspired by the class space shrinking theorem. Experiments show our method achieves remarkable performance over prior semi-supervised methods. 

**Abstract (ZH)**: 准确的情感分析对于各种应用至关重要，如理解客户反馈、监控市场趋势和检测公众情绪。然而，手动为监督学习标注大量情感语料库是耗时且劳动密集型的。因此，开发一种半监督方法对于情感分析任务是必要且有效的。尽管已经提出了多种半监督文本分类方法，但这些方法主要依赖于未标注数据的内在信息和NLP模型的学习能力，缺乏对情感分析场景的泛化能力，而且容易出现过拟合的情况。借鉴预训练大型语言模型（LLMs）遵循指令和生成连贯文本的能力，我们提出了一种基于大型语言模型的情感一致性正则化（Semantic Consistency Regularization with Large Language Models, SCR）框架，用于半监督情感分析。我们引入了两种提示策略来利用大型语言模型增强未标注文本的语义信息。第一种是实体增强（SCR-EE），涉及提取实体和数值信息，并查询模型重建文本信息。第二种是概念增强（SCR-CE），直接用原始句子查询模型进行语义重建。随后，利用增强后的数据通过一致性损失进行训练，并结合置信阈值保留高质量的一致性样本，为训练期间提供额外的监督信号。此外，为了充分利用未标注数据中的不确定性样本，我们提出了一个受类空间收缩定理启发的类别重组策略。实验结果显示，我们的方法在前人研究的半监督方法中取得了显著性能。 

---
# Tailored Truths: Optimizing LLM Persuasion with Personalization and Fabricated Statistics 

**Title (ZH)**: 量身定制的真相：通过个性化和虚构统计优化大型语言模型的说服力 

**Authors**: Jasper Timm, Chetan Talele, Jacob Haimes  

**Link**: [PDF](https://arxiv.org/pdf/2501.17273)  

**Abstract**: Large Language Models (LLMs) are becoming increasingly persuasive, demonstrating the ability to personalize arguments in conversation with humans by leveraging their personal data. This may have serious impacts on the scale and effectiveness of disinformation campaigns. We studied the persuasiveness of LLMs in a debate setting by having humans $(n=33)$ engage with LLM-generated arguments intended to change the human's opinion. We quantified the LLM's effect by measuring human agreement with the debate's hypothesis pre- and post-debate and analyzing both the magnitude of opinion change, as well as the likelihood of an update in the LLM's direction. We compare persuasiveness across established persuasion strategies, including personalized arguments informed by user demographics and personality, appeal to fabricated statistics, and a mixed strategy utilizing both personalized arguments and fabricated statistics. We found that static arguments generated by humans and GPT-4o-mini have comparable persuasive power. However, the LLM outperformed static human-written arguments when leveraging the mixed strategy in an interactive debate setting. This approach had a $\mathbf{51\%}$ chance of persuading participants to modify their initial position, compared to $\mathbf{32\%}$ for the static human-written arguments. Our results highlight the concerning potential for LLMs to enable inexpensive and persuasive large-scale disinformation campaigns. 

**Abstract (ZH)**: 大型语言模型（LLMs）正变得越来越具有说服力，通过利用个人数据在其与人类的对话中个性化其论点。这可能对虚假信息运动的规模和有效性产生严重影响。我们通过让人类参与者（n=33）与生成的LLM论点进行互动，来研究LLM在辩论环境中的说服力。我们通过测量参与者对于辩论假说的赞成度在辩论前后的变化程度来量化LLM的影响，同时分析论点方向和说服程度的变化幅度。我们比较了几种已确立的说服策略，包括基于用户人口统计和个人特征的个性化论点、援引虚假统计数据，以及结合个性化论点和虚假统计数据的混合策略的说服效果。我们发现，由人类和GPT-4o-mini生成的静态论点在说服力上相差不大。然而，在交互式辩论环境中，当LLM利用混合策略时，其表现超过了静态的人类撰写论点。这种策略有$\mathbf{51\%}$的成功率使参与者改变初始观点，而静态的人类撰写论点的成功率仅为$\mathbf{32\%}$。我们的研究结果突显了LLMs在支持低成本、高说服力的大规模虚假信息运动方面的令人担忧的潜力。 

---
# NUS-Emo at SemEval-2024 Task 3: Instruction-Tuning LLM for Multimodal Emotion-Cause Analysis in Conversations 

**Title (ZH)**: 新加坡国立大学情感分析团队在SemEval-2024 任务3中的研究成果：通过指令调优大语言模型进行多模态对话情感成因分析 

**Authors**: Meng Luo, Han Zhang, Shengqiong Wu, Bobo Li, Hong Han, Hao Fei  

**Link**: [PDF](https://arxiv.org/pdf/2501.17261)  

**Abstract**: This paper describes the architecture of our system developed for Task 3 of SemEval-2024: Multimodal Emotion-Cause Analysis in Conversations. Our project targets the challenges of subtask 2, dedicated to Multimodal Emotion-Cause Pair Extraction with Emotion Category (MECPE-Cat), and constructs a dual-component system tailored to the unique challenges of this task. We divide the task into two subtasks: emotion recognition in conversation (ERC) and emotion-cause pair extraction (ECPE). To address these subtasks, we capitalize on the abilities of Large Language Models (LLMs), which have consistently demonstrated state-of-the-art performance across various natural language processing tasks and domains. Most importantly, we design an approach of emotion-cause-aware instruction-tuning for LLMs, to enhance the perception of the emotions with their corresponding causal rationales. Our method enables us to adeptly navigate the complexities of MECPE-Cat, achieving a weighted average 34.71% F1 score of the task, and securing the 2nd rank on the leaderboard. The code and metadata to reproduce our experiments are all made publicly available. 

**Abstract (ZH)**: 本文描述了我们为SemEval-2024任务3：对话多模态情感原因分析系统开发的架构。我们的项目针对子任务2，即多模态情感原因配对提取（带情感类别）（MECPE-Cat）所面临的挑战，并构建了一个双组件系统，以应对这一任务的独特挑战。我们将任务细分为两个子任务：对话中的情感识别（ERC）和情感原因配对提取（ECPE）。为了解决这些子任务，我们利用了大型语言模型（LLMs）的能力，这些模型在各种自然语言处理任务和领域中始终表现出最先进的性能。最重要的是，我们设计了一种情感-原因意识的指令微调方法，以增强LLMs对情感及其相应因果分析的感知能力。我们的方法使我们能够熟练应对MECPE-Cat的复杂性，任务加权平均F1得分为34.71%，并取得了排行榜第2名的成绩。我们已将复现实验所需的代码和元数据全部公开。 

---
# Uncertainty Quantification and Decomposition for LLM-based Recommendation 

**Title (ZH)**: 基于大语言模型的推荐系统的不确定性量化与分解 

**Authors**: Wonbin Kweon, Sanghwan Jang, SeongKu Kang, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17630)  

**Abstract**: Despite the widespread adoption of large language models (LLMs) for recommendation, we demonstrate that LLMs often exhibit uncertainty in their recommendations. To ensure the trustworthy use of LLMs in generating recommendations, we emphasize the importance of assessing the reliability of recommendations generated by LLMs. We start by introducing a novel framework for estimating the predictive uncertainty to quantitatively measure the reliability of LLM-based recommendations. We further propose to decompose the predictive uncertainty into recommendation uncertainty and prompt uncertainty, enabling in-depth analyses of the primary source of uncertainty. Through extensive experiments, we (1) demonstrate predictive uncertainty effectively indicates the reliability of LLM-based recommendations, (2) investigate the origins of uncertainty with decomposed uncertainty measures, and (3) propose uncertainty-aware prompting for a lower predictive uncertainty and enhanced recommendation. Our source code and model weights are available at this https URL 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在推荐系统中得到了广泛应用，但我们证明了LLMs在推荐方面往往表现出不确定性。为了确保LLMs在生成推荐时的可信使用，我们强调了评估LLM生成推荐可靠性的必要性。我们首先引入了一个新的框架，用于估计预测不确定性，以定量衡量LLM推荐的可靠性。我们进一步提出将预测不确定性分解为推荐不确定性和提示不确定性，从而能够深入了解不确定性的主要来源。通过广泛的实验，我们（1）证明预测不确定性有效地指示了LLM推荐的可靠性，（2）通过分解的不确定性度量来探究不确定性的根源，并（3）提出了具备不确定性感知的提示方法，以降低预测不确定性并提高推荐质量。我们的源代码和模型权重可在以下链接获得：[此处链接] 

---
# GLLM: Self-Corrective G-Code Generation using Large Language Models with User Feedback 

**Title (ZH)**: GLLM：使用大型语言模型并在用户反馈基础上进行自我纠正的G-代码生成 

**Authors**: Mohamed Abdelaal, Samuel Lokadjaja, Gilbert Engert  

**Link**: [PDF](https://arxiv.org/pdf/2501.17584)  

**Abstract**: This paper introduces GLLM, an innovative tool that leverages Large Language Models (LLMs) to automatically generate G-code from natural language instructions for Computer Numerical Control (CNC) machining. GLLM addresses the challenges of manual G-code writing by bridging the gap between human-readable task descriptions and machine-executable code. The system incorporates a fine-tuned StarCoder-3B model, enhanced with domain-specific training data and a Retrieval-Augmented Generation (RAG) mechanism. GLLM employs advanced prompting strategies and a novel self-corrective code generation approach to ensure both syntactic and semantic correctness of the generated G-code. The architecture includes robust validation mechanisms, including syntax checks, G-code-specific verifications, and functional correctness evaluations using Hausdorff distance. By combining these techniques, GLLM aims to democratize CNC programming, making it more accessible to users without extensive programming experience while maintaining high accuracy and reliability in G-code generation. 

**Abstract (ZH)**: 本文介绍了GLLM，这是一种创新工具，利用大型语言模型（LLMs）自动从自然语言指令生成计算机数字控制（CNC）加工所需的G代码。GLLM通过在人类可读的任务描述与机器可执行代码之间搭建桥梁，解决了手动编写G代码的挑战。该系统结合了经过特定领域训练的StarCoder-3B模型，并增强了检索增强生成（RAG）机制。GLLM采用先进的提示策略和新颖的自我纠正代码生成方法，确保生成的G代码在语法和语义上都是正确的。该架构包括了稳健的验证机制，包括语法检查、G代码特定验证以及通过豪斯多夫距离进行的功能正确性评估。通过结合这些技术，GLLM旨在将CNC编程知识平民化，使具有有限编程经验的用户也能更方便地进行编程，同时保持G代码生成的高准确性和可靠性。 

---
# Attribution analysis of legal language as used by LLM 

**Title (ZH)**: LLM 使用的法律语言归因分析 

**Authors**: Richard K. Belew  

**Link**: [PDF](https://arxiv.org/pdf/2501.17330)  

**Abstract**: Three publicly-available LLM specifically designed for legal tasks have been implemented and shown that classification accuracy can benefit from training over legal corpora, but why and how? Here we use two publicly-available legal datasets, a simpler binary classification task of ``overruling'' texts, and a more elaborate multiple choice task identifying ``holding'' judicial decisions. We report on experiments contrasting the legal LLM and a generic BERT model for comparison, against both datasets. We use integrated gradient attribution techniques to impute ``causes'' of variation in the models' perfomance, and characterize them in terms of the tokenizations each use. We find that while all models can correctly classify some test examples from the casehold task, other examples can only be identified by only one, model, and attribution can be used to highlight the reasons for this. We find that differential behavior of the models' tokenizers accounts for most of the difference and analyze these differences in terms of the legal language they process. Frequency analysis of tokens generated by dataset texts, combined with use of known ``stop word'' lists, allow identification of tokens that are clear signifiers of legal topics. 

**Abstract (ZH)**: 已经实现了三个专为法律任务设计的公开可用的大语言模型，并展示了通过在法律语料库上训练可以提高分类准确性，但为什么和如何提高呢？在这项研究中，我们使用两个公开可用的法律数据集：一个简单的二元分类任务“推翻”文本，以及一个较为复杂的多项选择任务，用于识别“裁决”。我们使用实验对比了法律大语言模型和通用的BERT模型在两个数据集上的表现。我们采用集成梯度归因技术来解释模型性能的变化，并从它们的分词方式出发进行描述。我们发现，尽管所有模型都能正确分类部分案例持有任务的数据集示例，但其他示例只能被单一模型识别，归因分析可用于突出显示这种差异的原因。我们发现，模型分词器的差异性行为是主要差异所在，并借助这些差异解释它们处理的法律语言。通过对由数据集文本生成的词元进行频率分析，并结合已知的“停用词”列表，我们可以识别出明确的法律主题标志词。 

---
# Audio Large Language Models Can Be Descriptive Speech Quality Evaluators 

**Title (ZH)**: 音频大型语言模型可以作为描述性语音质量评估器 

**Authors**: Chen Chen, Yuchen Hu, Siyin Wang, Helin Wang, Zhehuai Chen, Chao Zhang, Chao-Han Huck Yang, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2501.17202)  

**Abstract**: An ideal multimodal agent should be aware of the quality of its input modalities. Recent advances have enabled large language models (LLMs) to incorporate auditory systems for handling various speech-related tasks. However, most audio LLMs remain unaware of the quality of the speech they process. This limitation arises because speech quality evaluation is typically excluded from multi-task training due to the lack of suitable datasets. To address this, we introduce the first natural language-based speech evaluation corpus, generated from authentic human ratings. In addition to the overall Mean Opinion Score (MOS), this corpus offers detailed analysis across multiple dimensions and identifies causes of quality degradation. It also enables descriptive comparisons between two speech samples (A/B tests) with human-like judgment. Leveraging this corpus, we propose an alignment approach with LLM distillation (ALLD) to guide the audio LLM in extracting relevant information from raw speech and generating meaningful responses. Experimental results demonstrate that ALLD outperforms the previous state-of-the-art regression model in MOS prediction, with a mean square error of 0.17 and an A/B test accuracy of 98.6%. Additionally, the generated responses achieve BLEU scores of 25.8 and 30.2 on two tasks, surpassing the capabilities of task-specific models. This work advances the comprehensive perception of speech signals by audio LLMs, contributing to the development of real-world auditory and sensory intelligent agents. 

**Abstract (ZH)**: 理想的多模态代理应当意识到其输入模态的质量。近期的进展使大型语言模型（LLMs）能够集成声学系统以处理各种语音相关任务。然而，大多数音频LLMs仍然无法意识到其所处理语音的质量。这一限制主要是由于缺乏合适的数据集，导致语音质量评估通常不作为多任务训练的一部分。为解决这一问题，我们引入了第一个基于自然语言的语音评估语料库，该语料库源自真实的人员评分。除了总体的意见评分（MOS）之外，该语料库还提供了在多个维度上的详细分析，并确定了质量下降的原因。它还能够以类似于人类判断的方式对两个语音样本（A/B测试）进行描述性比较。通过利用该语料库，我们提出了一种利用LLM蒸馏的对齐方法（ALLD），以指导音频LLMs从原始语音中提取相关信息并生成有意义的响应。实验结果表明，ALLD在MOS预测方面的性能优于之前的最佳回归模型，均方误差为0.17，A/B测试准确率为98.6%。此外，生成的响应在两个任务上的BLEU分数分别为25.8和30.2，超过了专门模型的能力。本工作提高了音频LLMs对语音信号的综合感知，有助于开发实用的听觉和感觉智能代理。 

---
# Leveraging Multimodal LLM for Inspirational User Interface Search 

**Title (ZH)**: 利用多模态大语言模型进行启发式用户界面搜索 

**Authors**: Seokhyeon Park, Yumin Song, Soohyun Lee, Jaeyoung Kim, Jinwook Seo  

**Link**: [PDF](https://arxiv.org/pdf/2501.17799)  

**Abstract**: Inspirational search, the process of exploring designs to inform and inspire new creative work, is pivotal in mobile user interface (UI) design. However, exploring the vast space of UI references remains a challenge. Existing AI-based UI search methods often miss crucial semantics like target users or the mood of apps. Additionally, these models typically require metadata like view hierarchies, limiting their practical use. We used a multimodal large language model (MLLM) to extract and interpret semantics from mobile UI images. We identified key UI semantics through a formative study and developed a semantic-based UI search system. Through computational and human evaluations, we demonstrate that our approach significantly outperforms existing UI retrieval methods, offering UI designers a more enriched and contextually relevant search experience. We enhance the understanding of mobile UI design semantics and highlight MLLMs' potential in inspirational search, providing a rich dataset of UI semantics for future studies. 

**Abstract (ZH)**: 启发式搜索是指探索设计方案以启发新创造力的过程，在移动用户界面（UI）设计中至关重要。然而，探索大量的UI参考空间仍然是一个挑战。现有的基于AI的UI搜索方法往往忽略了目标用户或应用程序的情绪等关键语义。此外，这些模型通常需要视图层次结构等元数据，限制了其实际应用。我们采用了多模态大型语言模型（MLLM）来从移动UI图像中提取和解释语义。通过形成性研究，我们识别了关键的UI语义，并开发了一个基于语义的UI搜索系统。通过计算和人工评估，我们证明了我们的方法显著优于现有的UI检索方法，为UI设计师提供了更为丰富和相关的搜索体验。我们加深了对移动UI设计语义的理解，并突显了MLLM在启发式搜索中的潜力，提供了丰富的UI语义数据集，以供未来研究使用。 

---
