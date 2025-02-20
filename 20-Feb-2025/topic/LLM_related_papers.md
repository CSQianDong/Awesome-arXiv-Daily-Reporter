# Lost in Sequence: Do Large Language Models Understand Sequential Recommendation? 

**Title (ZH)**: 迷失在序列之中：大型语言模型理解序列推荐吗？ 

**Authors**: Sein Kim, Hongseok Kang, Kibum Kim, Jiwan Kim, Donghyun Kim, Minchul Yang, Kwangjin Oh, Julian McAuley, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2502.13909)  

**Abstract**: Large Language Models (LLMs) have recently emerged as promising tools for recommendation thanks to their advanced textual understanding ability and context-awareness. Despite the current practice of training and evaluating LLM-based recommendation (LLM4Rec) models under a sequential recommendation scenario, we found that whether these models understand the sequential information inherent in users' item interaction sequences has been largely overlooked. In this paper, we first demonstrate through a series of experiments that existing LLM4Rec models do not fully capture sequential information both during training and inference. Then, we propose a simple yet effective LLM-based sequential recommender, called LLM-SRec, a method that enhances the integration of sequential information into LLMs by distilling the user representations extracted from a pre-trained CF-SRec model into LLMs. Our extensive experiments show that LLM-SRec enhances LLMs' ability to understand users' item interaction sequences, ultimately leading to improved recommendation performance. Furthermore, unlike existing LLM4Rec models that require fine-tuning of LLMs, LLM-SRec achieves state-of-the-art performance by training only a few lightweight MLPs, highlighting its practicality in real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来因其高级的文本理解和上下文感知能力，成为了推荐系统中的有前途的工具。尽管当前的实践中，LLM基础推荐（LLM4Rec）模型主要是在序列推荐的场景下进行训练和评估，但我们发现这些模型是否能够全面理解用户项目交互序列中的顺序信息这一问题并未得到充分的关注。在本文中，我们首先通过一系列实验展示了现有LLM4Rec模型在训练和推理过程中未能充分捕捉序列信息。然后，我们提出了一种简单而有效的基于序列的LLM推荐模型，称为LLM-SRec，在该模型中，通过从预训练的CF-SRec模型中抽取用户表示并将其蒸馏到LLM中，来增强序列信息在LLM中的整合。我们的广泛实验显示，LLM-SRec能够增强LLM理解用户项目交互序列的能力，从而提高推荐性能。此外，与现有LLM4Rec模型需要对LLM进行微调不同，LLM-SRec仅通过训练几组轻量级的MLP就能达到最先进的性能，突显了其在实际应用中的实用性。我们的代码可在以下链接获取：这个 https URL。 

---
# Enhancing LLM-Based Recommendations Through Personalized Reasoning 

**Title (ZH)**: 通过个性化推理增强基于LLM的推荐系统 

**Authors**: Jiahao Liu, Xueshuo Yan, Dongsheng Li, Guangping Zhang, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13845)  

**Abstract**: Current recommendation systems powered by large language models (LLMs) often underutilize their reasoning capabilities due to a lack of explicit logical structuring. To address this limitation, we introduce CoT-Rec, a framework that integrates Chain-of-Thought (CoT) reasoning into LLM-driven recommendations by incorporating two crucial processes: user preference analysis and item perception evaluation. CoT-Rec operates in two key phases: (1) personalized data extraction, where user preferences and item perceptions are identified, and (2) personalized data application, where this information is leveraged to refine recommendations. Our experimental analysis demonstrates that CoT-Rec improves recommendation accuracy by making better use of LLMs' reasoning potential. The implementation is publicly available at this https URL. 

**Abstract (ZH)**: 当前由大规模语言模型（LLMs）驱动的推荐系统往往没有充分利用其推理能力，主要是因为缺乏明确的逻辑结构。为了解决这一限制，我们引入了CoT-Rec框架，该框架通过整合 Chain-of-Thought（CoT）推理，将逻辑推理融入到由LLM驱动的推荐系统中，并通过两个关键过程来实现：用户偏好分析和项目感知评估。CoT-Rec 包含两个核心阶段：（1）个性化数据提取，其中确定用户偏好和项目感知，（2）个性化数据应用，其中利用这些信息来改进推荐。我们的实验分析表明，CoT-Rec 通过更好地利用LLM的推理潜力，提高了推荐的准确性。该实现已公开发布，相关链接为：this https URL。 

---
# TALKPLAY: Multimodal Music Recommendation with Large Language Models 

**Title (ZH)**: TALKPLAY：基于大型语言模型的多模态音乐推荐 

**Authors**: Seungheon Doh, Keunwoo Choi, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2502.13713)  

**Abstract**: We present TalkPlay, a multimodal music recommendation system that reformulates the recommendation task as large language model token generation. TalkPlay represents music through an expanded token vocabulary that encodes multiple modalities - audio, lyrics, metadata, semantic tags, and playlist co-occurrence. Using these rich representations, the model learns to generate recommendations through next-token prediction on music recommendation conversations, that requires learning the associations natural language query and response, as well as music items. In other words, the formulation transforms music recommendation into a natural language understanding task, where the model's ability to predict conversation tokens directly optimizes query-item relevance. Our approach eliminates traditional recommendation-dialogue pipeline complexity, enabling end-to-end learning of query-aware music recommendations. In the experiment, TalkPlay is successfully trained and outperforms baseline methods in various aspects, demonstrating strong context understanding as a conversational music recommender. 

**Abstract (ZH)**: 我们提出了一种名为TalkPlay的多模态音乐推荐系统，将推荐任务重新定义为大型语言模型标记生成。TalkPlay 通过扩展的标记词汇表来表示音乐，该词汇表编码了多种模态的信息——包括音频、歌词、元数据、语义标签以及播放列表共现情况。利用这些丰富的表示形式，模型通过音乐推荐对话中下一个标记的预测来学习生成推荐，从而需要学习自然语言查询和响应之间的关联，以及音乐项目。换句话说，这种表述将音乐推荐转化为一个自然语言理解任务，模型预测对话标记的能力直接优化了查询与项目的相关性。我们的方法消除了传统的推荐对话流水线复杂性，从而实现基于查询的音乐推荐的端到端学习。在实验中，TalkPlay 成功地进行了训练，并在多个方面超越了基准方法，展示了其作为对话式音乐推荐系统强大的上下文理解能力。 

---
# Bursting Filter Bubble: Enhancing Serendipity Recommendations with Aligned Large Language Models 

**Title (ZH)**: 破除信息茧房：通过对齐的大语言模型增强偶然性推荐 

**Authors**: Yunjia Xi, Muyan Weng, Wen Chen, Chao Yi, Dian Chen, Gaoyang Guo, Mao Zhang, Jian Wu, Yuning Jiang, Qingwen Liu, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13539)  

**Abstract**: Recommender systems (RSs) often suffer from the feedback loop phenomenon, e.g., RSs are trained on data biased by their recommendations. This leads to the filter bubble effect that reinforces homogeneous content and reduces user satisfaction. To this end, serendipity recommendations, which offer unexpected yet relevant items, are proposed. Recently, large language models (LLMs) have shown potential in serendipity prediction due to their extensive world knowledge and reasoning capabilities. However, they still face challenges in aligning serendipity judgments with human assessments, handling long user behavior sequences, and meeting the latency requirements of industrial RSs. To address these issues, we propose SERAL (Serendipity Recommendations with Aligned Large Language Models), a framework comprising three stages: (1) Cognition Profile Generation to compress user behavior into multi-level profiles; (2) SerenGPT Alignment to align serendipity judgments with human preferences using enriched training data; and (3) Nearline Adaptation to integrate SerenGPT into industrial RSs pipelines efficiently. Online experiments demonstrate that SERAL improves exposure ratio (PVR), clicks, and transactions of serendipitous items by 5.7%, 29.56%, and 27.6%, enhancing user experience without much impact on overall revenue. Now, it has been fully deployed in the "Guess What You Like" of the Taobao App homepage. 

**Abstract (ZH)**: 推荐系统（RSs）通常会遭受反馈循环现象的影响，例如，它们在受到自身推荐数据偏差训练后，会导致过滤泡泡效应，进一步强化同质化内容，降低用户体验。为了解决这一问题，提出了意外推荐（Serendipity Recommendations），这种推荐能够提供意外且相关的内容。最近，大型语言模型（LLMs）在意外推荐预测方面显示出潜力，因为它们具有广泛的世界知识和推理能力。然而，它们仍然面临着将意外推荐判断与人类评估对齐、处理长用户行为序列以及满足工业推荐系统延迟要求的挑战。为了解决这些问题，我们提出了一种名为SERAL（Serendipity Recommendations with Aligned Large Language Models）的框架，包括三个阶段：（1）认知概况生成（Cognition Profile Generation），将用户行为压缩成多级概况；（2）SerenGPT对齐（SerenGPT Alignment），使用丰富化的训练数据将意外推荐判断与人类偏好对齐；以及（3）离线适应（Nearline Adaptation），高效地将SerenGPT集成到工业推荐系统管道中。在线实验表明，SERAL可以将意外推荐项目的曝光率（PVR）、点击率和交易量分别提高5.7%、29.56%和27.6%，从而提升用户体验，同时对总收入的影响较小。目前，该框架已经全面部署在淘宝App主页的“猜你喜欢”功能中。 

---
# LLM4Tag: Automatic Tagging System for Information Retrieval via Large Language Models 

**Title (ZH)**: LLM4Tag：通过大规模语言模型的自动标签系统用于信息检索 

**Authors**: Ruiming Tang, Chenxu Zhu, Bo Chen, Weipeng Zhang, Menghui Zhu, Xinyi Dai, Huifeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.13481)  

**Abstract**: Tagging systems play an essential role in various information retrieval applications such as search engines and recommender systems. Recently, Large Language Models (LLMs) have been applied in tagging systems due to their extensive world knowledge, semantic understanding, and reasoning capabilities. Despite achieving remarkable performance, existing methods still have limitations, including difficulties in retrieving relevant candidate tags comprehensively, challenges in adapting to emerging domain-specific knowledge, and the lack of reliable tag confidence quantification. To address these three limitations above, we propose an automatic tagging system LLM4Tag. First, a graph-based tag recall module is designed to effectively and comprehensively construct a small-scale highly relevant candidate tag set. Subsequently, a knowledge-enhanced tag generation module is employed to generate accurate tags with long-term and short-term knowledge injection. Finally, a tag confidence calibration module is introduced to generate reliable tag confidence scores. Extensive experiments over three large-scale industrial datasets show that LLM4Tag significantly outperforms the state-of-the-art baselines and LLM4Tag has been deployed online for content tagging to serve hundreds of millions of users. 

**Abstract (ZH)**: 标签系统在各种信息检索应用中（如搜索引擎和推荐系统）起着至关重要的作用。近年来，由于大型语言模型（LLMs）具有广泛的世界知识、语义理解和推理能力，它们被应用于标签系统中。尽管现有方法取得了显著的性能，但仍存在一些局限性，包括难以全面检索相关候选标签、难以适应新兴的域特定知识以及缺乏可靠标签置信度量化的问题。为了解决上述三个局限性，我们提出了一种自动标签系统——LLM4Tag。首先，设计了一种基于图的标签召回模块，以有效地并全面地构建一个小规模的相关候选标签集。随后，采用了知识增强的标签生成模块，通过长短期知识注入生成准确的标签。最后，引入了标签置信度校准模块，生成可靠的标签置信度分数。在三个大规模工业数据集上进行的广泛实验表明，LLM4Tag 显著优于现有最先进的基线方法，并且LLM4Tag 已在线部署用于内容标记，为数亿用户提供服务。 

---
# SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering? 

**Title (ZH)**: SearchRAG：搜索引擎对基于LLM的医疗问答有帮助吗？ 

**Authors**: Yucheng Shi, Tianze Yang, Canyu Chen, Quanzheng Li, Tianming Liu, Xiang Li, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13233)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in general domains but often struggle with tasks requiring specialized knowledge. Conventional Retrieval-Augmented Generation (RAG) techniques typically retrieve external information from static knowledge bases, which can be outdated or incomplete, missing fine-grained clinical details essential for accurate medical question answering. In this work, we propose SearchRAG, a novel framework that overcomes these limitations by leveraging real-time search engines. Our method employs synthetic query generation to convert complex medical questions into search-engine-friendly queries and utilizes uncertainty-based knowledge selection to filter and incorporate the most relevant and informative medical knowledge into the LLM's input. Experimental results demonstrate that our method significantly improves response accuracy in medical question answering tasks, particularly for complex questions requiring detailed and up-to-date knowledge. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在通用领域展现了出色的性能，但在需要专业知识的任务上往往表现不佳。传统的检索增强生成（RAG）技术通常从静态知识库中检索外部信息，这些知识库可能过时或不完整，缺乏准确回答医学问题所必需的细粒度临床细节。本研究中，我们提出了一种名为SearchRAG的新框架，该框架通过利用实时搜索引擎来克服这些局限性。该方法通过合成查询生成将复杂的医学问题转换为搜索引擎友好的查询，并通过基于不确定性的知识选择来筛选和整合与LLM输入最相关的、最有信息价值的医学知识。实验结果表明，该方法显著提高了在医学问答任务中的响应准确性，尤其是在需要详细和最新知识的复杂问题上。 

---
# Neurosymbolic artificial intelligence via large language models and coherence-driven inference 

**Title (ZH)**: 通过大型语言模型和连贯驱动的推理实现神经符号人工智能 

**Authors**: Steve Huntsman, Jewell Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2502.13953)  

**Abstract**: We devise an algorithm to generate sets of propositions that objectively instantiate graphs that support coherence-driven inference. We then benchmark the ability of large language models (LLMs) to reconstruct coherence graphs from (a straightforward transformation of) propositions expressed in natural language, with promising results from a single prompt to models optimized for reasoning. Combining coherence-driven inference with consistency evaluations by neural models may advance the state of the art in machine cognition. 

**Abstract (ZH)**: 我们设计了一个算法，用于生成能够客观地实例化支持共现推理的图的命题集。然后，我们对大型语言模型（LLMs）的能力进行了基准测试，使其能够从自然语言中表达的命题（经过简单的转换）重构共现图，结果显示，仅通过一个提示即可实现对优化推理能力模型的有效性。将共现推理与神经模型的一致性评估相结合，可能促进机器认知领域的最新进展。 

---
# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence 

**Title (ZH)**: 自适应步长：通过模型信心自动划分推理步骤 

**Authors**: Yuliang Liu, Junjie Lu, Zhaoling Chen, Chaofeng Qu, Jason Klein Liu, Chonghan Liu, Zefan Cai, Yunhui Xia, Li Zhao, Jiang Bian, Chuheng Zhang, Wei Shen, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13943)  

**Abstract**: Current approaches for training Process Reward Models (PRMs) often involve breaking down responses into multiple reasoning steps using rule-based techniques, such as using predefined placeholder tokens or setting the reasoning step's length into a fixed size. These approaches overlook the fact that specific words do not typically mark true decision points in a text. To address this, we propose AdaptiveStep, a method that divides reasoning steps based on the model's confidence in predicting the next word. This division method provides more decision-making information at each step, enhancing downstream tasks, such as reward model learning. Moreover, our method does not require manual annotation. We demonstrate its effectiveness through experiments with AdaptiveStep-trained PRMs in mathematical reasoning and code generation tasks. Experimental results indicate that the outcome PRM achieves state-of-the-art Best-of-N performance, surpassing greedy search strategy with token-level value-guided decoding, while also reducing construction costs by over 30% compared to existing open-source PRMs. In addition, we provide a thorough analysis and case study on the PRM's performance, transferability, and generalization capabilities. 

**Abstract (ZH)**: 当前用于训练过程奖励模型（PRMs）的方法通常涉及使用基于规则的技术将响应拆分为多个推理步骤，例如使用预定义的占位符标记或固定推理步骤的长度。这些方法忽视了特定单词通常不标志着文本中的真实决策点这一事实。为了解决这一问题，我们提出了一种名为AdaptiveStep的方法，该方法根据模型预测下一个单词的信心程度来划分推理步骤。这种划分方法在每个步骤中提供了更多的决策信息，从而增强了下游任务，例如奖励模型的学习。此外，我们的方法不需要手动注释。我们通过使用AdaptiveStep训练的PRMs在数学推理和代码生成任务中的实验验证了其效果。实验结果表明，所得到的PRM在Best-of-N性能上达到了最先进的水平，优于基于token级价值指导解码的贪心搜索策略，并且与现有开源PRMs相比，构建成本降低了超过30%。此外，我们还对PRM的性能、可迁移性和泛化能力进行了详尽的分析和案例研究。 

---
# Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning 

**Title (ZH)**: 利用LLM与符号推理相结合证明奥林匹克不等式 

**Authors**: Zenan Li, Zhaoyu Li, Wen Tang, Xian Zhang, Yuan Yao, Xujie Si, Fan Yang, Kaiyu Yang, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.13834)  

**Abstract**: Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以通过在证明系统中生成证明步骤（即策略）来形式化地证明数学定理。然而，可能的策略空间既庞大又复杂，而可用于形式证明的训练数据有限，这给基于LLM的策略生成带来了重大挑战。为了解决这个问题，我们引入了一种神经符号策略生成器，它将LLM学到的数学直觉与符号方法编码的领域特定见解结合起来。这种集成的关键在于识别哪些部分的数学推理最适合LLM，哪些部分最适合符号方法。尽管神经-符号集成的基本思想适用于各种数学问题，但在本文中，我们具体关注奥林匹克不等式（参见图1）。我们分析了人类如何解决这些问题，并提炼出了两种类型的策略：（1）缩放，由符号方法处理；（2）重写，由LLM处理。此外，我们结合符号工具与LLM来修剪和排序证明目标，以实现高效的证明搜索。我们在来自多个数学竞赛的161个挑战性不等式上评估了我们的框架，实现了最先进的性能，并在无需额外训练数据的情况下显著优于现有的LLM和符号方法。 

---
# Reasoning with Reinforced Functional Token Tuning 

**Title (ZH)**: 强化功能标记调优的推理方法 

**Authors**: Kongcheng Zhang, Qi Yao, Baisheng Lai, Jiaxing Huang, Wenkai Fang, Dacheng Tao, Mingli Song, Shunyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13389)  

**Abstract**: In this work, we propose Reinforced Functional Token Tuning (RFTT), a novel reinforced fine-tuning framework that empowers Large Language Models (LLMs) with self-play learn-to-reason capabilities. Unlike prior prompt-driven reasoning efforts, RFTT embeds a rich set of learnable functional tokens (e.g., <analyze>, <verify>, <refine>) directly into the model vocabulary, enabling chain-of-thought construction with diverse human-like reasoning behaviors. Specifically, RFTT comprises two phases: (1) supervised fine-tuning performs prompt-driven tree search to obtain self-generated training data annotated with functional tokens, which warms up the model to learn these tokens for reasoning; and (2) online reinforcement learning further allows the model to explore different reasoning pathways through functional token sampling without relying on prompts, thereby facilitating effective self-improvement for functional reasoning. Extensive experiments demonstrate the superiority of the proposed RFTT on mathematical benchmarks, significantly boosting Qwen-2.5-7B-Instruct (70.6% to 79.8%) and LLaMA-3.1-8B-Instruct (32.2% to 60.2%) on the MATH dataset. Moreover, the performance of RFTT consistently improves with more search rollouts at inference time. Our code is available at this https URL. 

**Abstract (ZH)**: 在这项工作中，我们提出了一种新型的强化函数标记调优框架——强化功能标记调优（RFTT），旨在增强大型语言模型（LLMs）的自博弈学习推理能力。与之前的基于提示的推理努力不同，RFTT 将一组丰富的可学习功能标记（例如 <analyze>、<verify>、<refine>）直接嵌入到模型词汇表中，使模型能够构建具有多种类人类推理行为的推理链条。具体而言，RFTT 包含两个阶段：（1）监督调优通过基于提示的树搜索获取带有功能标记的自动生成训练数据，以预热模型学习这些用于推理的功能标记；（2）在线强化学习进一步允许模型通过功能标记采样探索不同的推理路径，而无需依赖提示，从而促进功能推理的有效自我改进。大量的实验表明，RFTT 在数学基准测试中表现优越，在 MATH 数据集上显著提升了 Qwen-2.5-7B-Instruct（从 70.6% 提高到 79.8%）和 LLaMA-3.1-8B-Instruct（从 32.2% 提高到 60.2%）的效果。此外，我们在推理时进行更多搜索展开时，RFTT 的性能也持续改进。我们的代码可在以下 URL 获取：this https URL。 

---
# Demonstrating specification gaming in reasoning models 

**Title (ZH)**: 在推理模型中展示规范游戏行为 

**Authors**: Alexander Bondarenko, Denis Volk, Dmitrii Volkov, Jeffrey Ladish  

**Link**: [PDF](https://arxiv.org/pdf/2502.13295)  

**Abstract**: We demonstrate LLM agent specification gaming by instructing models to win against a chess engine. We find reasoning models like o1 preview and DeepSeek-R1 will often hack the benchmark by default, while language models like GPT-4o and Claude 3.5 Sonnet need to be told that normal play won't work to hack.
We improve upon prior work like (Hubinger et al., 2024; Meinke et al., 2024; Weij et al., 2024) by using realistic task prompts and avoiding excess nudging. Our results suggest reasoning models may resort to hacking to solve difficult problems, as observed in OpenAI (2024)'s o1 Docker escape during cyber capabilities testing. 

**Abstract (ZH)**: 我们通过指示模型战胜国际象棋引擎来演示大规模语言模型（LLM）代理规范博弈。我们发现，如o1 Preview和DeepSeek-R1这类逻辑推理模型往往会默认作弊，而像GPT-4o和Claude 3.5 Sonnet这类语言模型则需要明确被告知正常对弈无法作弊。

我们改进了前人的工作（如Hubinger等人，2024；Meinke等人，2024；Wei等人，2024），通过使用实际的任务提示并避免过度引导。我们的结果表明，逻辑推理模型可能会为了解决复杂问题而采取作弊手段，这与OpenAI（2024）在网络安全能力测试中观察到的o1 Docker脱逃现象一致。 

---
# Unveiling the Magic of Code Reasoning through Hypothesis Decomposition and Amendment 

**Title (ZH)**: 通过假设分解与修正揭示代码推理的魔力 

**Authors**: Yuze Zhao, Tianyun Ji, Wenjun Feng, Zhenya Huang, Qi Liu, Zhiding Liu, Yixiao Ma, Kai Zhang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13170)  

**Abstract**: The reasoning abilities are one of the most enigmatic and captivating aspects of large language models (LLMs). Numerous studies are dedicated to exploring and expanding the boundaries of this reasoning capability. However, tasks that embody both reasoning and recall characteristics are often overlooked. In this paper, we introduce such a novel task, code reasoning, to provide a new perspective for the reasoning abilities of LLMs. We summarize three meta-benchmarks based on established forms of logical reasoning, and instantiate these into eight specific benchmark tasks. Our testing on these benchmarks reveals that LLMs continue to struggle with identifying satisfactory reasoning pathways. Additionally, we present a new pathway exploration pipeline inspired by human intricate problem-solving methods. This Reflective Hypothesis Decomposition and Amendment (RHDA) pipeline consists of the following iterative steps: (1) Proposing potential hypotheses based on observations and decomposing them; (2) Utilizing tools to validate hypotheses and reflection outcomes; (3) Revising hypothesis in light of observations. Our approach effectively mitigates logical chain collapses arising from forgetting or hallucination issues in multi-step reasoning, resulting in performance gains of up to $3\times$. Finally, we expanded this pipeline by applying it to simulate complex household tasks in real-world scenarios, specifically in VirtualHome, enhancing the handling of failure cases. We release our code and all of results at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）的推理能力是最神秘和引人入胜的方面之一。众多研究致力于探索并拓展这一推理能力的边界。然而，同时包含推理和记忆特征的任务往往被忽视。本文我们引入了新型任务——代码推理，为LLMs的推理能力提供新的视角。我们根据现有的逻辑推理形式总结了三个元基准，并将这些形式具体化为八个特定的基准任务。在对这些基准任务进行测试后，我们发现LLMs仍然难以识别满意的推理路径。此外，我们提出了一种新的路径探索管道，灵感来源于人类复杂的解决问题方法。该管道称为反思假设分解与修正(RHDA)管道，由以下迭代步骤组成：（1）基于观察提出潜在假设并对其进行分解；（2）使用工具验证假设和反思结果；（3）根据观察结果修正假设。我们的方法有效地缓解了多步推理中由于遗忘或幻觉问题导致的逻辑链条崩溃问题，从而带来了最多3倍的性能提升。最后，我们通过将其应用于模拟真实场景中的复杂家庭任务（如VirtualHome），扩展了该管道，增加了对失败情况的处理能力。我们在此处发布我们的代码和所有结果：此链接。 

---
# Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region 

**Title (ZH)**: 为什么受保护的船舶会触礁？对齐的大语言模型的安全机制倾向于扎根于模板区域 

**Authors**: Chak Tou Leong, Qingyu Yin, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13946)  

**Abstract**: The safety alignment of large language models (LLMs) remains vulnerable, as their initial behavior can be easily jailbroken by even relatively simple attacks. Since infilling a fixed template between the input instruction and initial model output is a common practice for existing LLMs, we hypothesize that this template is a key factor behind their vulnerabilities: LLMs' safety-related decision-making overly relies on the aggregated information from the template region, which largely influences these models' safety behavior. We refer to this issue as template-anchored safety alignment. In this paper, we conduct extensive experiments and verify that template-anchored safety alignment is widespread across various aligned LLMs. Our mechanistic analyses demonstrate how it leads to models' susceptibility when encountering inference-time jailbreak attacks. Furthermore, we show that detaching safety mechanisms from the template region is promising in mitigating vulnerabilities to jailbreak attacks. We encourage future research to develop more robust safety alignment techniques that reduce reliance on the template region. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全对齐仍然存在漏洞，因为它们的初始行为可以通过相对简单的攻击轻易破解。由于在输入指令和初始模型输出之间填充固定模板是现有LLMs中常见的做法，我们假设模板是导致这些漏洞的关键因素：LLMs在安全相关决策上过分依赖模板区域的综合信息，这极大地影响了这些模型的安全行为。我们将这一问题称为模板锚定的安全对齐。

在本文中，我们进行了广泛的经验性实验，并验证了模板锚定的安全对齐在各种对齐的LLMs中普遍存在。我们的机制分析表明，当遇到推理时的破解攻击时，这会导致模型易受影响。此外，我们展示了将安全机制与模板区域分离是减轻破解攻击漏洞的有前景的方法。我们鼓励未来的研究开发减少对模板区域依赖的更稳健的安全对齐技术。 

---
# How Do LLMs Perform Two-Hop Reasoning in Context? 

**Title (ZH)**: 在上下文中，大型语言模型如何进行两跳推理？ 

**Authors**: Tianyu Guo, Hanlin Zhu, Ruiqi Zhang, Jiantao Jiao, Song Mei, Michael I. Jordan, Stuart Russell  

**Link**: [PDF](https://arxiv.org/pdf/2502.13913)  

**Abstract**: "Socrates is human. All humans are mortal. Therefore, Socrates is mortal." This classical example demonstrates two-hop reasoning, where a conclusion logically follows from two connected premises. While transformer-based Large Language Models (LLMs) can make two-hop reasoning, they tend to collapse to random guessing when faced with distracting premises. To understand the underlying mechanism, we train a three-layer transformer on synthetic two-hop reasoning tasks. The training dynamics show two stages: a slow learning phase, where the 3-layer transformer performs random guessing like LLMs, followed by an abrupt phase transitions, where the 3-layer transformer suddenly reaches $100%$ accuracy. Through reverse engineering, we explain the inner mechanisms for how models learn to randomly guess between distractions initially, and how they learn to ignore distractions eventually. We further propose a three-parameter model that supports the causal claims for the mechanisms to the training dynamics of the transformer. Finally, experiments on LLMs suggest that the discovered mechanisms generalize across scales. Our methodologies provide new perspectives for scientific understandings of LLMs and our findings provide new insights into how reasoning emerges during training. 

**Abstract (ZH)**: “苏格拉底是人。所有人都会死亡。因此，苏格拉底会死亡。”这一经典的示例展示了两步推理，即结论逻辑地源自两个相连的前提。虽然基于变换器的大语言模型（LLMs）可以进行两步推理，但在面对分散注意力的前提时，它们往往会退化为随机猜测。为了理解其内在机制，我们在一个合成的两步推理任务上训练了一个三层变换器。训练动态表明有两个阶段：一个缓慢的学习阶段，在此阶段中，三层变换器像LLMs一样进行随机猜测，随后是一个突变的相位转变，在此阶段中，三层变换器突然达到100%的准确率。通过逆向工程，我们解释了模型如何首先在分散注意力的情况下进行随机猜测，以及如何最终学会忽略这些分散注意力的前提。我们进一步提出一个三参数模型，支持基于变换器的训练动态机制的因果主张。最后，对LLMs的实验表明，发现的这些机制在不同规模下具有泛化性。我们的方法论为大语言模型的科学理解提供了新的视角，而我们的发现则为我们提供了关于训练过程中推理如何涌现的新见解。 

---
# DataSciBench: An LLM Agent Benchmark for Data Science 

**Title (ZH)**: DataSciBench：一个用于数据科学的大型语言模型代理基准测试 

**Authors**: Dan Zhang, Sining Zhoubian, Min Cai, Fengzu Li, Lekang Yang, Wei Wang, Tianjiao Dong, Ziniu Hu, Jie Tang, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.13897)  

**Abstract**: This paper presents DataSciBench, a comprehensive benchmark for evaluating Large Language Model (LLM) capabilities in data science. Recent related benchmarks have primarily focused on single tasks, easily obtainable ground truth, and straightforward evaluation metrics, which limits the scope of tasks that can be evaluated. In contrast, DataSciBench is constructed based on a more comprehensive and curated collection of natural and challenging prompts for uncertain ground truth and evaluation metrics. We develop a semi-automated pipeline for generating ground truth (GT) and validating evaluation metrics. This pipeline utilizes and implements an LLM-based self-consistency and human verification strategy to produce accurate GT by leveraging collected prompts, predefined task types, and aggregate functions (metrics). Furthermore, we propose an innovative Task - Function - Code (TFC) framework to assess each code execution outcome based on precisely defined metrics and programmatic rules. Our experimental framework involves testing 6 API-based models, 8 open-source general models, and 9 open-source code generation models using the diverse set of prompts we have gathered. This approach aims to provide a more comprehensive and rigorous evaluation of LLMs in data science, revealing their strengths and weaknesses. Experimental results demonstrate that API-based models outperform open-sourced models on all metrics and Deepseek-Coder-33B-Instruct achieves the highest score among open-sourced models. We release all code and data at this https URL. 

**Abstract (ZH)**: 本文介绍了DataSciBench，这是一个全面的基准测试，用于评估大型语言模型（LLM）在数据科学领域的能力。最近的相关基准测试主要集中在单一任务、易于获得的地面真实值和简便的评估指标上，这限制了可以评估的任务范围。相比之下，DataSciBench 是基于更为全面和精挑细选的自然且具有挑战性的提示构建的，适用于不确定的地面真实值和评估指标。我们开发了一种半自动管道来生成地面真实值（GT）并验证评估指标。该管道利用并实现了基于LLM的自我一致性与人工验证策略，通过利用收集的提示、预定义的任务类型和聚合函数（指标）来生成准确的GT。此外，我们提出了一种创新的Task-Function-Code（TFC）框架，根据精确定义的度量标准和编程规则来评估每次代码执行的结果。我们的实验框架包括使用我们收集的多样化的提示测试6种API基模型、8种开源通用模型和9种开源代码生成模型。这种方法旨在提供更全面和严格的LLM在数据科学领域的评估，揭示它们的优势和劣势。实验结果表明，API基模型在所有指标上均优于开源模型，Deepseek-Coder-33B-Instruct 在开源模型中得分最高。我们已在此链接 https://... 上发布了所有代码和数据。 

---
# LESA: Learnable LLM Layer Scaling-Up 

**Title (ZH)**: LESA：可学习的大型语言模型层扩展方法 

**Authors**: Yifei Yang, Zouying Cao, Xinbei Ma, Yao Yao, Libo Qin, Zhi Chen, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13794)  

**Abstract**: Training Large Language Models (LLMs) from scratch requires immense computational resources, making it prohibitively expensive. Model scaling-up offers a promising solution by leveraging the parameters of smaller models to create larger ones. However, existing depth scaling-up methods rely on empirical heuristic rules for layer duplication, which result in poorer initialization and slower convergence during continual pre-training. We propose \textbf{LESA}, a novel learnable method for depth scaling-up. By concatenating parameters from each layer and applying Singular Value Decomposition, we uncover latent patterns between layers, suggesting that inter-layer parameters can be learned. LESA uses a neural network to predict the parameters inserted between adjacent layers, enabling better initialization and faster training. Experiments show that LESA outperforms existing baselines, achieving superior performance with less than half the computational cost during continual pre-training. Extensive analyses demonstrate its effectiveness across different model sizes and tasks. 

**Abstract (ZH)**: 从头开始训练大型语言模型（LLMs）需要巨大的计算资源，使其变得极其昂贵。通过利用较小模型的参数来创建更大模型的规模扩展提供了一个有前景的解决方案。然而，现有的深度扩展方法依赖于基于经验的启发式规则来进行层复制，这会导致较差的初始化和持续预训练期间更慢的收敛速度。我们提出了\textbf{LESA}，一种新颖的学习式深度扩展方法。通过将每一层的参数连接起来并应用奇异值分解（SVD），我们发现了层间隐藏的模式，表明层间参数可以被学习。LESA 使用神经网络预测相邻层之间插入的参数，从而实现更好的初始化和更快的训练。实验结果表明，LESA 在持续预训练期间的计算成本不到现有基线的一半，且性能更优。广泛分析证明了它在不同模型大小和任务上的有效性。 

---
# AI Software Engineer: Programming with Trust 

**Title (ZH)**: AI软件工程师：基于信任的编程 

**Authors**: Abhik Roychoudhury, Corina Pasareanu, Michael Pradel, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2502.13767)  

**Abstract**: Large Language Models (LLMs) have shown surprising proficiency in generating code snippets, promising to automate large parts of software engineering via artificial intelligence (AI). We argue that successfully deploying AI software engineers requires a level of trust equal to or even greater than the trust established by human-driven software engineering practices. The recent trend toward LLM agents offers a path toward integrating the power of LLMs to create new code with the power of analysis tools to increase trust in the code. This opinion piece comments on whether LLM agents could dominate software engineering workflows in the future and whether the focus of programming will shift from programming at scale to programming with trust. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成代码片段方面展示了令人惊讶的能力，有潜力通过人工智能（AI）自动化软件工程中的大部分工作。我们认为，在人工智能软件工程师的成功部署中，建立的信任度应当与甚至高于由人工驱动的软件工程实践所建立的信任度。近年来，LLM智能代理的趋势提供了一条将LLMs的强大生成能力与分析工具的验证能力相结合的路径，以增强对代码的信任。本文观点讨论了LLM智能代理是否可能在未来主导软件工程工作流程，以及编程的重点是否会从大规模编程转向具有信任度的编程。 

---
# Direct Value Optimization: Improving Chain-of-Thought Reasoning in LLMs with Refined Values 

**Title (ZH)**: 直接价值优化：通过细化的价值提升大型语言模型中的链式思考推理 

**Authors**: Hongbo Zhang, Han Cui, Guangsheng Bao, Linyi Yang, Jun Wang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13723)  

**Abstract**: We introduce Direct Value Optimization (DVO), an innovative reinforcement learning framework for enhancing large language models in complex reasoning tasks. Unlike traditional methods relying on preference labels, DVO utilizes value signals at individual reasoning steps, optimizing models via a mean squared error loss. The key benefit of DVO lies in its fine-grained supervision, circumventing the need for labor-intensive human annotations. Target values within the DVO are estimated using either Monte Carlo Tree Search or an outcome value model. Our empirical analysis on both mathematical and commonsense reasoning tasks shows that DVO consistently outperforms existing offline preference optimization techniques, even with fewer training steps. These findings underscore the importance of value signals in advancing reasoning capabilities and highlight DVO as a superior methodology under scenarios lacking explicit human preference information. 

**Abstract (ZH)**: 我们介绍了直接价值优化（DVO），这是一种创新的强化学习框架，用于增强在复杂推理任务中的大型语言模型。与依赖偏好标签的传统方法不同，DVO 利用每个推理步骤的价值信号，并通过均方误差损失优化模型。DVO 的主要优势在于其精细监督，从而避免了需要劳动密集型的人工标注。在 DVO 中，目标值可以通过蒙特卡洛树搜索（MCTS）或结果价值模型进行估算。我们在数学推理和常识推理任务上的实证分析表明，即使训练步骤较少，DVO 也能持续优于现有的离线偏好优化技术。这些发现强调了价值信号在提升推理能力方面的重要性，并突显了在缺乏明确的人类偏好信息的情况下，DVO 作为一个更优的方法论的优势。 

---
# Concept Layers: Enhancing Interpretability and Intervenability via LLM Conceptualization 

**Title (ZH)**: 概念层：通过LLM概念化增强可解释性和可干预性 

**Authors**: Or Raphael Bidusa, Shaul Markovitch  

**Link**: [PDF](https://arxiv.org/pdf/2502.13632)  

**Abstract**: The opaque nature of Large Language Models (LLMs) has led to significant research efforts aimed at enhancing their interpretability, primarily through post-hoc methods. More recent in-hoc approaches, such as Concept Bottleneck Models (CBMs), offer both interpretability and intervenability by incorporating explicit concept representations. However, these methods suffer from key limitations, including reliance on labeled concept datasets and significant architectural modifications that challenges re-integration into existing system pipelines. In this work, we introduce a new methodology for incorporating interpretability and intervenability into an existing model by integrating Concept Layers (CLs) into its architecture. Our approach projects the model's internal vector representations into a conceptual, explainable vector space before reconstructing and feeding them back into the model. Furthermore, we eliminate the need for a human-selected concept set by algorithmically searching an ontology for a set of concepts that can be either task-specific or task-agnostic. We evaluate CLs across multiple tasks, demonstrating that they maintain the original model's performance and agreement while enabling meaningful interventions. Additionally, we present a proof of concept showcasing an intervenability interface, allowing users to adjust model behavior dynamically, such as mitigating biases during inference. 

**Abstract (ZH)**: 大型语言模型（LLMs）的不透明性导致了大量旨在提高其可解释性的研究努力，主要通过事后方法进行。更近期的内置方法，如概念瓶颈模型（CBMs），通过引入明确的概念表示，同时提供了可解释性和干预性。然而，这些方法也面临一些关键局限性，包括对标记概念数据集的依赖以及对现有架构进行重大修改，这给重新整合进现有的系统管道带来了挑战。在本研究中，我们提出了一种新的方法，通过将概念层（CLs）嵌入现有模型的架构中，以实现可解释性和干预性。我们的方法将模型的内部向量表示投影到一个概念性的、可解释的向量空间中，然后再进行重构并反馈到模型中。此外，我们通过算法搜索概念本体来消除人工选择概念集的需求，这组概念可以是与任务特定的，也可以是与任务无关的。我们对CLs进行了多任务评估，结果显示，它们在保持原始模型性能和一致性的同时，还允许进行有意义的干预。此外，我们还展示了概念层干预接口的概念验证，用户可以动态调整模型行为，例如，在推理过程中缓解偏见。 

---
# REFIND: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models 

**Title (ZH)**: REFIND：大型语言模型中检索增强的事实幻觉检测 

**Authors**: DongGeon Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13622)  

**Abstract**: Hallucinations in large language model (LLM) outputs severely limit their reliability in knowledge-intensive tasks such as question answering. To address this challenge, we introduce REFIND (Retrieval-augmented Factuality hallucINation Detection), a novel framework that detects hallucinated spans within LLM outputs by directly leveraging retrieved documents. As part of the REFIND, we propose the Context Sensitivity Ratio (CSR), a novel metric that quantifies the sensitivity of LLM outputs to retrieved evidence. This innovative approach enables REFIND to efficiently and accurately detect hallucinations, setting it apart from existing methods. In the evaluation, REFIND demonstrated robustness across nine languages, including low-resource settings, and significantly outperformed baseline models, achieving superior IoU scores in identifying hallucinated spans. This work highlights the effectiveness of quantifying context sensitivity for hallucination detection, thereby paving the way for more reliable and trustworthy LLM applications across diverse languages. 

**Abstract (ZH)**: 大型语言模型（LLM）输出中的幻觉严重限制了其在知识密集型任务（如问答）中的可靠性。为了解决这一挑战，我们提出了一种新颖的方法REFIND（Retrieval-augmented Factuality hallucINation Detection），该方法通过直接利用检索到的文档来检测LLM输出中的幻觉片段。作为REFIND的一部分，我们提出了上下文敏感度比（CSR，Context Sensitivity Ratio），这是一种新颖的度量标准，用于量化LLM输出对检索到的证据的敏感性。这种创新方法使REFIND能够高效且准确地检测幻觉，从而使它在现有方法中脱颖而出。在评估中，REFIND在九种语言（包括低资源环境）中显示出了鲁棒性，并大幅优于基线模型，在识别幻觉片段方面获得了更高的IoU分数。这项研究成果突显了量化上下文敏感性在幻觉检测中的有效性，从而为不同语言的大规模语言模型（LLM）应用提供了更可靠和可信的路径。 

---
# Complex Ontology Matching with Large Language Model Embeddings 

**Title (ZH)**: 使用大型语言模型嵌入进行复杂的本体匹配 

**Authors**: Guilherme Sousa, Rinaldo Lima, Cassia Trojahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.13619)  

**Abstract**: Ontology, and more broadly, Knowledge Graph Matching is a challenging task in which expressiveness has not been fully addressed. Despite the increasing use of embeddings and language models for this task, approaches for generating expressive correspondences still do not take full advantage of these models, in particular, large language models (LLMs). This paper proposes to integrate LLMs into an approach for generating expressive correspondences based on alignment need and ABox-based relation discovery. The generation of correspondences is performed by matching similar surroundings of instance sub-graphs. The integration of LLMs results in different architectural modifications, including label similarity, sub-graph matching, and entity matching. The performance word embeddings, sentence embeddings, and LLM-based embeddings, was compared. The results demonstrate that integrating LLMs surpasses all other models, enhancing the baseline version of the approach with a 45\% increase in F-measure. 

**Abstract (ZH)**: 本论文探讨了本体和更广泛的知识图谱匹配任务，尽管嵌入表示和语言模型的应用越来越广泛，但生成富有表现力的对应关系的方法仍未充分利用这些模型，尤其是大型语言模型（LLMs）。本文提出了一种将LLMs整合到基于实例子图相似性对齐和ABox关系发现的生成富有表现力对应关系的方法中。通过匹配实例子图周围的相似部分进行对应关系的生成。将LLMs整合到了新架构中，这包括标签相似性、子图匹配和实体匹配。研究了词嵌入、句子嵌入和基于LLM的嵌入的表现。结果显示，将LLMs整合进来的方法显著优于其他模型，使方法的基础版本的F-测量值提高了45%。 

---
# LaVCa: LLM-assisted Visual Cortex Captioning 

**Title (ZH)**: LaVCa：LLM辅助的视觉 cortex 图像描述 

**Authors**: Takuya Matsuyama, Shinji Nishimoto, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13606)  

**Abstract**: Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations. Please check out our webpage at this https URL 

**Abstract (ZH)**: 了解人类大脑中神经群体（或体素）的特性可以增进我们对人类感知和认知处理能力的理解，并有助于开发受大脑启发的计算机模型。近年来，使用深度神经网络（DNNs）的编码模型已经成功地预测了体素级别的活动。然而，由于DNNs的黑盒性质，解释解释体素响应的特性仍然具有挑战性。为此，我们提出了一种基于数据的方法——LaVCa（LLM辅助视觉皮层描述），利用大规模语言模型（LLMs）为具有体素选择性的图像生成自然语言描述。通过将LaVCa应用于由图像引发的大脑活动，我们证明了LaVCa生成的描述体素选择性的 caption 更加准确，与之前的方法相比，LaVCa生成的 caption 在体素间和体素内层面更详细地捕捉到了更多的特性。此外，对LaVCa生成的体素特定特性的详细分析揭示了视觉皮层中感兴趣区域（ROIs）内的精细功能分化，以及同时代表多个不同概念的体素。这些发现通过在整个视觉皮层中赋予详细的 caption 提供了深刻的人类视觉表征洞察，并突显了基于LLM的方法在理解大脑表征中的潜力。请访问我们的网页：[此 https URL] 

---
# Are Large Language Models In-Context Graph Learners? 

**Title (ZH)**: 大型语言模型是上下文图学习者吗？ 

**Authors**: Jintang Li, Ruofan Wu, Yuchang Zhu, Huizhe Zhang, Liang Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable in-context reasoning capabilities across a wide range of tasks, particularly with unstructured inputs such as language or images. However, LLMs struggle to handle structured data, such as graphs, due to their lack of understanding of non-Euclidean structures. As a result, without additional fine-tuning, their performance significantly lags behind that of graph neural networks (GNNs) in graph learning tasks. In this paper, we show that learning on graph data can be conceptualized as a retrieval-augmented generation (RAG) process, where specific instances (e.g., nodes or edges) act as queries, and the graph itself serves as the retrieved context. Building on this insight, we propose a series of RAG frameworks to enhance the in-context learning capabilities of LLMs for graph learning tasks. Comprehensive evaluations demonstrate that our proposed RAG frameworks significantly improve LLM performance on graph-based tasks, particularly in scenarios where a pretrained LLM must be used without modification or accessed via an API. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展示了令人印象深刻的上下文推理能力，尤其是在处理未结构化的输入（如语言或图像）方面。然而，LLMs 在处理结构化数据（如图）方面表现不佳，这主要是由于它们无法理解非欧几里得结构。因此，在不需要额外微调的情况下，它们在图学习任务中的表现远远落后于图神经网络（GNNs）。本文中，我们表明在图数据上进行学习可以被视为一种检索增强生成（RAG）过程，其中特定实例（例如节点或边）作为查询，而图本身则作为检索的上下文。基于这一见解，我们提出了一系列 RAG 框架，以增强 LLMs 在图学习任务中的上下文学习能力。全面的评估结果表明，我们提出的 RAG 框架显著提升了 LLMs 在基于图的任务中的性能，特别是在需要使用未修改的预训练 LLM 或通过 API 访问的情景中。 

---
# Democratizing Large Language Model-Based Graph Data Augmentation via Latent Knowledge Graphs 

**Title (ZH)**: 通过潜在知识图谱实现基于大型语言模型的图形数据增强的民主化 

**Authors**: Yushi Feng, Tsai Hor Chan, Guosheng Yin, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13555)  

**Abstract**: Data augmentation is necessary for graph representation learning due to the scarcity and noise present in graph data. Most of the existing augmentation methods overlook the context information inherited from the dataset as they rely solely on the graph structure for augmentation. Despite the success of some large language model-based (LLM) graph learning methods, they are mostly white-box which require access to the weights or latent features from the open-access LLMs, making them difficult to be democratized for everyone as existing LLMs are mostly closed-source for commercial considerations. To overcome these limitations, we propose a black-box context-driven graph data augmentation approach, with the guidance of LLMs -- DemoGraph. Leveraging the text prompt as context-related information, we task the LLM with generating knowledge graphs (KGs), which allow us to capture the structural interactions from the text outputs. We then design a dynamic merging schema to stochastically integrate the LLM-generated KGs into the original graph during training. To control the sparsity of the augmented graph, we further devise a granularity-aware prompting strategy and an instruction fine-tuning module, which seamlessly generates text prompts according to different granularity levels of the dataset. Extensive experiments on various graph learning tasks validate the effectiveness of our method over existing graph data augmentation methods. Notably, our approach excels in scenarios involving electronic health records (EHRs), which validates its maximal utilization of contextual knowledge, leading to enhanced predictive performance and interpretability. 

**Abstract (ZH)**: 由于图数据中存在稀缺性和噪声问题，数据扩增对于图表示学习是必要的。现有的大多数扩增方法忽略了从数据集中继承的上下文信息，因为它们仅依赖于图结构来进行扩增。尽管一些基于大规模语言模型（LLM）的图学习方法取得了成功，但这些方法大多为白盒模型，需要访问开源LLM的权重或潜在特征，这使得这些模型难以普及到每个人手中，因为现有的LLM大多出于商业考虑而保持封闭源代码。为了解决这些局限性，我们提出了一种基于LLM的黑盒上下文驱动的图数据扩增方法——DemoGraph。利用文本提示作为上下文相关信息，我们指示LLM生成知识图谱（KGs），这使我们能够从文本输出中捕获结构交互信息。然后，我们设计了一种动态融合方案，在训练过程中随机将LLM生成的KGs整合到原始图中。为了控制扩增图的稀疏性，我们进一步开发了一种细粒度感知的提示策略和指令微调模块，可以根据不同数据集的细粒度级别无缝生成文本提示。在各种图学习任务的广泛实验中，我们的方法证明了其比现有图数据扩增方法的有效性。值得注意的是，我们的方法在涉及电子健康记录（EHRs）的场景中表现出色，这验证了其在最大化利用上下文知识方面的优势，从而提高了预测性能和可解释性。 

---
# Activation-aware Probe-Query: Effective Key-Value Retrieval for Long-Context LLMs Inference 

**Title (ZH)**: 基于激活感知的探针查询：面向长上下文大语言模型推理的有效键值检索 

**Authors**: Qingfa Xiao, Jiachuan Wang, Haoyang Li, Cheng Deng, Jiaqi Tang, Shuangyin Li, Yongqi Zhang, Jun Wang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13542)  

**Abstract**: Recent advances in large language models (LLMs) have showcased exceptional performance in long-context tasks, while facing significant inference efficiency challenges with limited GPU memory. Existing solutions first proposed the sliding-window approach to accumulate a set of historical \textbf{key-value} (KV) pairs for reuse, then further improvements selectively retain its subsets at each step. However, due to the sparse attention distribution across a long context, it is hard to identify and recall relevant KV pairs, as the attention is distracted by massive candidate pairs. Additionally, we found it promising to select representative tokens as probe-Query in each sliding window to effectively represent the entire context, which is an approach overlooked by existing methods. Thus, we propose \textbf{ActQKV}, a training-free, \textbf{Act}ivation-aware approach that dynamically determines probe-\textbf{Q}uery and leverages it to retrieve the relevant \textbf{KV} pairs for inference. Specifically, ActQKV monitors a token-level indicator, Activation Bias, within each context window, enabling the proper construction of probe-Query for retrieval at pre-filling stage. To accurately recall the relevant KV pairs and minimize the irrelevant ones, we design a dynamic KV cut-off mechanism guided by information density across layers at the decoding stage. Experiments on the Long-Bench and $\infty$ Benchmarks demonstrate its state-of-the-art performance with competitive inference quality and resource efficiency. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在长上下文任务中展现了出色的表现，但在有限的GPU内存下，推理效率面临重大挑战。现有解决方案首先提出了滑动窗口方法，通过积累一系列可重复使用的历史**键值**（KV）对来解决这一问题，随后进一步改进的方法在每一步选择性地保留其子集。然而，由于长上下文中注意力分布的稀疏性，很难识别和回忆相关的KV对，因为注意力会被大量候选对所分散。此外，我们发现，在每个滑动窗口中选择具有代表性的标记作为探针-查询是有效的，这种做法被现有方法所忽视。因此，我们提出了一种无需训练、基于激活信息的**ActQKV**方法，该方法能够动态确定探针-查询，并利用它来检索推理所需的相关KV对。具体而言，ActQKV 在每个上下文窗口中监控一个标记级别的指标——激活偏差（Activation Bias），从而在预填充阶段适当地构建探针-查询以进行检索。为了准确回忆相关的KV对并最小化无关的KV对，我们在解码阶段通过层间信息密度引导的动态KV截断机制进行设计。在Long-Bench和$\infty$基准测试上的实验表明，该方法在保持高质量推理的同时，具有先进的性能和资源效率。 

---
# Towards Geo-Culturally Grounded LLM Generations 

**Title (ZH)**: 面向地理文化背景的大型语言模型生成 

**Authors**: Piyawat Lertvittayakumjorn, David Kinney, Vinodkumar Prabhakaran, Donald Martin, Sunipa Dev  

**Link**: [PDF](https://arxiv.org/pdf/2502.13497)  

**Abstract**: Generative large language models (LLMs) have been demonstrated to have gaps in diverse, cultural knowledge across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on the ability of LLMs to display familiarity with a diverse range of national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on a series of cultural familiarity benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., the norms, artifacts, and institutions of national cultures), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models, while failing to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional knowledge about a culture and open-ended cultural fluency when it comes to evaluating the cultural familiarity of generative LLMs. 

**Abstract (ZH)**: 生成型大规模语言模型（LLMs）在全球范围内显示出在多元文化和知识方面存在差距。我们探讨了检索增强生成和搜索定向技术对LLMs展示对多种国家文化熟悉程度能力的影响。具体而言，我们比较了标准LLMs、采用定制知识库检索增强的LLMs（即KB接地）以及通过网络搜索检索增强的LLMs（即搜索接地）在一系列文化熟悉度基准测试中的表现。我们发现，搜索接地在测试命题知识（如国家文化的规范、器物和制度）的多项选择基准测试中显著提高了LLMs的性能，而定制知识库接地的效果受限于知识库覆盖不全和检索器不够优化。然而，搜索接地也会增加语言模型产生刻板印象判断的风险，而在统计功效充足的评分者评价中，未能提高对文化熟悉度的判断。这些结果突显了在评估生成型LLMs的文化熟悉度时命题知识与开放性文化流畅度之间的重要区别。 

---
# What are Models Thinking about? Understanding Large Language Model Hallucinations "Psychology" through Model Inner State Analysis 

**Title (ZH)**: 《模型在思考些什么？通过模型内部状态分析理解大语言模型的幻觉现象》 

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Jue Hong, Ye Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13490)  

**Abstract**: Large language model (LLM) systems suffer from the models' unstable ability to generate valid and factual content, resulting in hallucination generation. Current hallucination detection methods heavily rely on out-of-model information sources, such as RAG to assist the detection, thus bringing heavy additional latency. Recently, internal states of LLMs' inference have been widely used in numerous research works, such as prompt injection detection, etc. Considering the interpretability of LLM internal states and the fact that they do not require external information sources, we introduce such states into LLM hallucination detection. In this paper, we systematically analyze different internal states' revealing features during inference forward and comprehensively evaluate their ability in hallucination detection. Specifically, we cut the forward process of a large language model into three stages: understanding, query, generation, and extracting the internal state from these stages. By analyzing these states, we provide a deep understanding of why the hallucinated content is generated and what happened in the internal state of the models. Then, we introduce these internal states into hallucination detection and conduct comprehensive experiments to discuss the advantages and limitations. 

**Abstract (ZH)**: 大型语言模型（LLM）系统在生成有效和准确内容方面存在不稳定的模型能力，导致产生幻觉。当前的幻觉检测方法高度依赖于模型外的信息来源，如RAG，以辅助检测，从而带来额外的延迟。最近，LLM推理过程中的内部状态在许多研究工作中得到了广泛应用，例如提示注入检测等。考虑到LLM内部状态的可解释性以及它们不需要外部信息源的事实，我们将其引入到LLM幻觉检测中。在本文中，我们系统地分析了推理过程各阶段内部状态揭示的不同特征，并全面评估了它们在幻觉检测中的能力。具体而言，我们将大型语言模型的前向过程划分为三个阶段：理解、查询、生成，并从中提取这些阶段的内部状态。通过分析这些状态，我们深入理解了幻觉内容是如何生成的，以及模型内部状态中发生了什么。然后，我们将这些内部状态引入到幻觉检测中，并进行全面的实验以讨论其优缺点。 

---
# LLM should think and action as a human 

**Title (ZH)**: 大语言模型（LLM）在思考和行动时应具备人类的特性。 

**Authors**: Haun Leung, ZiNan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13475)  

**Abstract**: It is popular lately to train large language models to be used as chat assistants, but in the conversation between the user and the chat assistant, there are prompts, require multi-turns between the chat assistant and the user. However, there are a number of issues with the multi-turns conversation: The response of the chat assistant is prone to errors and cannot help users achieve their goals; It is difficult for chat assistant to generate responses with different processes based on actual needs for the same command or request; Chat assistant require the use of tools, but the current approach is not elegant and efficient, and the number of tool calls that can be supported is limited. The main reason for these issues is that large language models do not have the thinking ability as a human, lack the reasoning ability and planning ability, and lack the ability to execute plans. To solve these issues, we propose a thinking method based on a built-in chain of thought: In the multi-turns conversation, for each user prompt, the large language model thinks based on elements such as chat history, thinking context, action calls, memory and knowledge, makes detailed reasoning and planning, and actions according to the plan. We also explored how the large language model enhances thinking ability through this thinking method: Collect training datasets according to the thinking method and fine tune the large language model through supervised learning; Train a consistency reward model and use it as a reward function to fine tune the large language model using reinforcement learning, and the reinforced large language model outputs according to this way of thinking. Our experimental results show that the reasoning ability and planning ability of the large language model are enhanced, and the issues in the multi-turns conversation are solved. 

**Abstract (ZH)**: 近年来，流行通过训练大规模语言模型以用作聊天助手。但在用户与聊天助手的对话中，聊天助手需要与用户进行多次互动。然而，多次对话中存在的问题相当多：聊天助手的回答容易出错，无法帮助用户实现目标；对于相同的命令或请求，聊天助手难以根据实际需求生成具有不同过程的响应；聊天助手需要使用工具，但当前的方法不够优雅和高效，能够支持的工具调用数量有限。这些问题的主要原因是大规模语言模型缺乏作为人类那样的思考能力、推理能力和规划能力，也缺乏执行计划的能力。为了解决这些问题，我们提出了一种基于内置链式思考的方法：在多次对话中，对于每个用户提示，大规模语言模型基于聊天历史、思考上下文、行动调用、记忆和知识等元素进行思考，进行详细的推理与规划，并按照计划执行相应的行动。此外，我们还探讨了该思考方法是如何增强语言模型的思考能力的：根据该思考方法收集训练数据集，并通过监督学习精细调整语言模型；训练一致性的奖励模型作为奖励函数，并通过强化学习精细调整语言模型，从而训练出能够在这种方式下思考的强化模型。我们的实验结果表明，通过这种方法增强了语言模型的推理能力与规划能力，解决了多次对话中存在的问题。 

---
# Estimating Commonsense Plausibility through Semantic Shifts 

**Title (ZH)**: 通过语义转换估算常识合理性 

**Authors**: Wanqing Cui, Keping Bi, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13464)  

**Abstract**: Commonsense plausibility estimation is critical for evaluating language models (LMs), yet existing generative approaches--reliant on likelihoods or verbalized judgments--struggle with fine-grained discrimination. In this paper, we propose ComPaSS, a novel discriminative framework that quantifies commonsense plausibility by measuring semantic shifts when augmenting sentences with commonsense-related information. Plausible augmentations induce minimal shifts in semantics, while implausible ones result in substantial deviations. Evaluations on two types of fine-grained commonsense plausibility estimation tasks across different backbones, including LLMs and vision-language models (VLMs), show that ComPaSS consistently outperforms baselines. It demonstrates the advantage of discriminative approaches over generative methods in fine-grained commonsense plausibility evaluation. Experiments also show that (1) VLMs yield superior performance to LMs, when integrated with ComPaSS, on vision-grounded commonsense tasks. (2) contrastive pre-training sharpens backbone models' ability to capture semantic nuances, thereby further enhancing ComPaSS. 

**Abstract (ZH)**: 常识合理性估计对于评估语言模型（LMs）至关重要，然而现有的生成方法——依赖于似然性或口头判断——在精细区分方面存在困难。本文提出了一种新颖的鉴别性框架ComPaSS，通过衡量在增加与常识相关的信息时语义的变化来量化常识合理性。合理的增加会引发较小的语义变化，而不合理的增加会导致显著的偏差。在不同基础模型（包括大语言模型和视觉-语言模型）上的两种类型的精细粒度常识合理性估计任务中，ComPaSS 在评估中始终优于基准模型，展示了鉴别性方法在精细粒度常识合理性评估中的优势。实验还表明：(1) 当视觉-语言模型（VLMs）与ComPaSS 结合使用时，在基于视觉的常识任务上比语言模型（LMs）表现出更优异的性能。(2) 对抗预训练增强了基础模型捕捉语义细微差别的能力，从而进一步提升了ComPaSS 的性能。 

---
# ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails 

**Title (ZH)**: ThinkGuard: 深思熟虑的慢思考导向审慎的边界控制 

**Authors**: Xiaofei Wen, Wenxuan Zhou, Wenjie Jacky Mo, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13458)  

**Abstract**: Ensuring the safety of large language models (LLMs) is critical as they are deployed in real-world applications. Existing guardrails rely on rule-based filtering or single-pass classification, limiting their ability to handle nuanced safety violations. To address this, we propose ThinkGuard, a critique-augmented guardrail model that distills knowledge from high-capacity LLMs by generating structured critiques alongside safety labels. Fine-tuned on critique-augmented data, the captured deliberative thinking ability drastically enhances the guardrail's cautiousness and interpretability. Evaluated on multiple safety benchmarks, ThinkGuard achieves the highest average F1 and AUPRC, outperforming all baselines. Compared to LLaMA Guard 3, ThinkGuard improves accuracy by 16.1% and macro F1 by 27.0%. Moreover, it surpasses label-only fine-tuned models, confirming that structured critiques enhance both classification precision and nuanced safety reasoning while maintaining computational efficiency. 

**Abstract (ZH)**: 确保大型语言模型（LLMs）的安全性至关重要，因为它们正在被应用于实际应用中。现有的防护措施依赖于基于规则的过滤或单次分类，这限制了它们处理复杂安全违规的能力。为解决这一问题，我们提出了一种名为ThinkGuard的批判增强型防护模型，该模型通过生成结构化的批判性反馈和安全标签来提炼高容量LLMs的知识。经过基于批判增强数据的微调，捕捉到的审慎思考能力极大地提升了防护模型的谨慎性和可解释性。在多个安全性基准测试中，ThinkGuard在平均F1分数和AUPRC方面表现最佳，优于所有基线模型。与LLaMA Guard 3相比，ThinkGuard在准确性上提高了16.1%，宏F1分数提高了27.0%。此外，它还超越了仅依赖标签的微调模型，这证实了结构化的批判性反馈不仅提高了分类精度，还增强了细致的安全推理能力，同时保持了计算效率。 

---
# TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation 

**Title (ZH)**: TreeCut：一种用于大语言模型幻觉评估的合成不可回答的数学文字题数据集 

**Authors**: Jialin Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13442)  

**Abstract**: Large language models (LLMs) now achieve near-human performance on standard math word problem benchmarks (e.g., GSM8K), yet their true reasoning ability remains disputed. A key concern is that models often produce confident, yet unfounded, answers to unanswerable problems. We introduce TreeCut, a synthetic dataset that systematically generates infinite unanswerable math word problems and their answerable counterparts, by representing each question as a tree and removing chosen necessary conditions. Experiments show TreeCut effectively induce hallucinations in large language models, including GPT-4o and o3-mini, with rates of 61% and 42% in their respective worst-case scenarios. Further analysis highlights that deeper or more complex trees, composite item names, and removing necessary condition near the middle of a path all increase the likelihood of hallucinations, underscoring the persistent challenges LLMs face in identifying unanswerable math problems. 

**Abstract (ZH)**: 大型语言模型（LLMs）现在已经在标准数学填空问题基准测试（例如GSM8K）上达到了接近人类的表现，但它们的真实推理能力仍然存在争议。一个关键问题是，模型往往会对不可解的问题给出自信但毫无根据的答案。我们引入了TreeCut，这是一个合成数据集，通过将每个问题表示为一棵树，并移除所选的必要条件，系统地生成无限数量的不可解的数学填空问题及其可解的对应问题。实验结果表明，在各自的最坏情况下，TreeCut有效地促使大型语言模型产生幻觉，GPT-4o和o3-mini的幻觉率分别为61%和42%。进一步分析表明，更深层次或更复杂的树结构、复合项目名称以及在路径中间移除必要条件都会增加幻觉的可能性，突显了LLMs在识别不可解的数学问题方面持续面临的挑战。 

---
# The Self-Improvement Paradox: Can Language Models Bootstrap Reasoning Capabilities without External Scaffolding? 

**Title (ZH)**: 自我提升悖论：语言模型能否在没有外部支撑的情况下自主提升推理能力？ 

**Authors**: Yutao Sun, Mingshuai Chen, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13441)  

**Abstract**: Self-improving large language models (LLMs) -- i.e., to improve the performance of an LLM by fine-tuning it with synthetic data generated by itself -- is a promising way to advance the capabilities of LLMs while avoiding extensive supervision. Existing approaches to self-improvement often rely on external supervision signals in the form of seed data and/or assistance from third-party models. This paper presents Crescent -- a simple yet effective framework for generating high-quality synthetic question-answer data in a fully autonomous manner. Crescent first elicits the LLM to generate raw questions via a bait prompt, then diversifies these questions leveraging a rejection sampling-based self-deduplication, and finally feeds the questions to the LLM and collects the corresponding answers by means of majority voting. We show that Crescent sheds light on the potential of true self-improvement with zero external supervision signals for math reasoning; in particular, Crescent-generated question-answer pairs suffice to (i) improve the reasoning capabilities of an LLM while preserving its general performance (especially in the 0-shot setting); and (ii) distil LLM knowledge to weaker models more effectively than existing methods based on seed-dataset augmentation. 

**Abstract (ZH)**: 自我增强的大语言模型（LLMs）——即通过使用模型自身生成的合成数据对LLM进行微调，以提高其性能——是一种在避免过度监督的情况下提升LLM能力的有前途的方法。现有的自我增强方法通常依赖于外部的监督信号，如种子数据和/或第三方模型的帮助。本文提出了一种名为Crescent的简单但有效的框架，用于完全自主地生成高质量的合成问答数据。首先，通过诱饵提示促使LLM生成原始问题，然后利用基于拒绝采样的自我去重技术使这些问题多样化，最后通过多数投票将这些问题输入LLM，并收集相应的答案。我们展示了Crescent在数学推理领域实现了真正的自我增强，特别是Crescent生成的问题-答案对足以：（i）在保持LLM整体性能（尤其是零样本设置下）的同时提高其推理能力；（ii）比基于种子数据集增强方法更有效地将LLM的知识提取到较弱的模型中。 

---
# RLTHF: Targeted Human Feedback for LLM Alignment 

**Title (ZH)**: RLTHF：针对的人员反馈以实现大规模语言模型的对齐 

**Authors**: Yifei Xu, Tusher Chakraborty, Emre Kıcıman, Bibek Aryal, Eduardo Rodrigues, Srinagesh Sharma, Roberto Estevao, Maria Angels de Luis Balaguer, Jessica Wolk, Rafael Padilha, Leonardo Nunes, Shobana Balakrishnan, Songwu Lu, Ranveer Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.13417)  

**Abstract**: Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF's strategic data curation. 

**Abstract (ZH)**: 将上述论文内容或标题翻译成中文，并符合学术规范：

在基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）中，高质量的人类注释成本高昂，同时人工智能反馈的一般性有限。因此，大规模语言模型（Large Language Models, LLMs）与用户偏好对齐的调整变得具有挑战性。为了解决这一挑战，我们提出了一种结合了LLM初始对齐与选择性人类注解的人机混合框架——基于奖励模型的RLTHF（RLHF with Reward Model, RLTHF）。该框架通过利用注释模型的奖励分布识别LLM难以标注的样本，并通过逐步整合战略性的手工纠错，重新对齐LLM，从而实现全面的人类注释对齐，同时减少人力投入。

具体而言，RLTHF采用奖励模型的奖励分布来识别LLM错误标注的样本，并通过循环加强对齐，利用LLM正确标注的样本与战略性的人类纠正相结合，实现对齐增强。在HH-RLHF与TL;DR数据集上的评估表明，通过仅使用6-7%的人力标注工作量，RLTHF即可达到与全面手工标注相当的对齐水平。此外，基于RLTHF精心构建的数据集进行下游任务训练的模型表现优于基于完全手工标注数据集训练的模型，这进一步证明了RLTHF战略数据筛选的有效性。 

---
# Explore-Construct-Filter: An Automated Framework for Rich and Reliable API Knowledge Graph Construction 

**Title (ZH)**: 探索-构建-过滤：一种自动化框架，用于构建丰富且可靠的API知识图谱 

**Authors**: Yanbang Sun, Qing Huang, Xiaoxue Ren, Zhenchang Xing, Xiaohong Li, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13412)  

**Abstract**: The API Knowledge Graph (API KG) is a structured network that models API entities and their relations, providing essential semantic insights for tasks such as API recommendation, code generation, and API misuse detection. However, constructing a knowledge-rich and reliable API KG presents several challenges. Existing schema-based methods rely heavily on manual annotations to design KG schemas, leading to excessive manual overhead. On the other hand, schema-free methods, due to the lack of schema guidance, are prone to introducing noise, reducing the KG's reliability. To address these issues, we propose the Explore-Construct-Filter framework, an automated approach for API KG construction based on large language models (LLMs). This framework consists of three key modules: 1) KG exploration: LLMs simulate the workflow of annotators to automatically design a schema with comprehensive type triples, minimizing human intervention; 2) KG construction: Guided by the schema, LLMs extract instance triples to construct a rich yet unreliable API KG; 3) KG filtering: Removing invalid type triples and suspicious instance triples to construct a rich and reliable API KG. Experimental results demonstrate that our method surpasses the state-of-the-art method, achieving a 25.2% improvement in F1 score. Moreover, the Explore-Construct-Filter framework proves effective, with the KG exploration module increasing KG richness by 133.6% and the KG filtering module improving reliability by 26.6%. Finally, cross-model experiments confirm the generalizability of our framework. 

**Abstract (ZH)**: API知识图谱（API KG）是一种结构化网络，用于建模API实体及其关系，为API推荐、代码生成和API误用检测等任务提供重要的语义洞察。然而，构建一个知识丰富且可靠的API KG面临着诸多挑战。现有的基于模式的方法高度依赖于人工注释来设计知识图谱的模式，导致了过多的手工劳动。另一方面，无模式方法由于缺乏模式指导，容易引入噪声，降低知识图谱的可靠性。为了解决这些问题，我们提出了一种基于大规模语言模型（LLMs）的自动化API KG构建框架——Explore-Construct-Filter框架。该框架包含三个关键模块：1）知识图谱探索：LLMs模拟注释人员的工作流程，自动设计一个包含全面类型三元组的模式，最大限度地减少人工干预；2）知识图谱构建：在模式的引导下，LLMs提取实例三元组构建一个丰富但不可靠的API知识图谱；3）知识图谱过滤：删除无效类型三元组和可疑实例三元组，构建一个丰富且可靠的API知识图谱。实验结果表明，我们的方法超越了现有最先进的方法，在F1分数上提高了25.2%。此外，Explore-Construct-Filter框架的效果得到验证：知识图谱探索模块将知识图谱的丰富性提高了133.6%，知识图谱过滤模块提高了知识图谱的可靠性26.6%。最后，跨模型实验进一步验证了该框架的泛化能力。 

---
# $\mathtt{GeLLM^3O}$: Generalizing Large Language Models for Multi-property Molecule Optimization 

**Title (ZH)**: $\mathtt{GeLLM^3O}$: 将大型语言模型应用于多属性分子优化 

**Authors**: Vishal Dey, Xiao Hu, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2502.13398)  

**Abstract**: Despite recent advancements, most computational methods for molecule optimization are constrained to single- or double-property optimization tasks and suffer from poor scalability and generalizability to novel optimization tasks. Meanwhile, Large Language Models (LLMs) demonstrate remarkable out-of-domain generalizability to novel tasks. To demonstrate LLMs' potential for molecule optimization, we introduce $\mathtt{MoMUInstruct}$, the first high-quality instruction-tuning dataset specifically focused on complex multi-property molecule optimization tasks. Leveraging $\mathtt{MoMUInstruct}$, we develop $\mathtt{GeLLM^3O}$s, a series of instruction-tuned LLMs for molecule optimization. Extensive evaluations across 5 in-domain and 5 out-of-domain tasks demonstrate that $\mathtt{GeLLM^3O}$s consistently outperform state-of-the-art baselines. $\mathtt{GeLLM^3O}$s also exhibit outstanding zero-shot generalization to unseen tasks, significantly outperforming powerful closed-source LLMs. Such strong generalizability demonstrates the tremendous potential of $\mathtt{GeLLM^3O}$s as foundational models for molecule optimization, thereby tackling novel optimization tasks without resource-intensive retraining. $\mathtt{MoMUInstruct}$, models, and code are accessible through this https URL. 

**Abstract (ZH)**: 尽管近年来取得了进展，但大多数用于分子优化的计算方法仍局限于单一或双属性优化任务，并且难以扩展且对新颖的优化任务缺乏普遍适应性。与此同时，大型语言模型（LLMs）在处理域外任务时展示了显著的普遍适应性。为了展示LLMs在分子优化中的潜力，我们引入了$\mathtt{MoMUInstruct}$，这是一个专注于复杂多属性分子优化任务的高质量指令调优数据集。利用$\mathtt{MoMUInstruct}$，我们开发了$\mathtt{GeLLM^3O}$系列指令调优的LLMs，用于分子优化。在5个领域内和5个领域外任务的广泛评估中，$\mathtt{GeLLM^3O}$s表现出一致的性能优势，超越了最先进的基线方法。$\mathtt{GeLLM^3O}$s在未见过的任务上还表现出了出色的零样本泛化能力，大幅超越了强大的闭源LLMs。这种强烈的泛化能力表明$\mathtt{GeLLM^3O}$s作为分子优化的基础模型具有巨大的潜力，能够应对新颖的优化任务而无需进行资源密集型的重新训练。您可以通过以下链接访问$\mathtt{MoMUInstruct}$、模型和代码：[此处填写链接]。 

---
# Language Models are Few-Shot Graders 

**Title (ZH)**: 语言模型是少量示例评阅者 

**Authors**: Chenyan Zhao, Mariana Silva, Seth Poulsen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13337)  

**Abstract**: Providing evaluations to student work is a critical component of effective student learning, and automating its process can significantly reduce the workload on human graders. Automatic Short Answer Grading (ASAG) systems, enabled by advancements in Large Language Models (LLMs), offer a promising solution for assessing and providing instant feedback for open-ended student responses. In this paper, we present an ASAG pipeline leveraging state-of-the-art LLMs. Our new LLM-based ASAG pipeline achieves better performances than existing custom-built models on the same datasets. We also compare the grading performance of three OpenAI models: GPT-4, GPT-4o, and o1-preview. Our results demonstrate that GPT-4o achieves the best balance between accuracy and cost-effectiveness. On the other hand, o1-preview, despite higher accuracy, exhibits a larger variance in error that makes it less practical for classroom use. We investigate the effects of incorporating instructor-graded examples into prompts using no examples, random selection, and Retrieval-Augmented Generation (RAG)-based selection strategies. Our findings indicate that providing graded examples enhances grading accuracy, with RAG-based selection outperforming random selection. Additionally, integrating grading rubrics improves accuracy by offering a structured standard for evaluation. 

**Abstract (ZH)**: 对学生作业进行评价是有效学生学习的重要组成部分，而自动化这一过程可以显著减轻人为评分者的负担。基于大型语言模型（LLMs）的发展，自动短答案评分（ASAG）系统为评估和提供即时反馈打开了新的可能性，尤其适用于开放式学生回答。本文介绍了一个利用最先进的LLMs构建的ASAG流水线。我们新的基于LLM的ASAG流水线在相同数据集上的表现优于现有自定义模型。我们还比较了OpenAI的三种模型：GPT-4、GPT-4o和o1-preview的评分性能。结果显示，GPT-4o在准确性和成本效益之间取得了最佳平衡。相比之下，尽管o1-preview的准确率较高，但其错误的更大方差使其在课堂教学中不够实用。我们研究了在提示中引入教师评分示例的效果，包括不使用示例、随机选择和基于检索增强生成（RAG）的选择策略。我们的研究结果表明，提供评分示例可以提高评分准确性，而基于RAG的选择策略优于随机选择。此外，结合评分标准能够通过提供结构化的评估标准进一步提高准确性。 

---
# Language Models Can Predict Their Own Behavior 

**Title (ZH)**: 语言模型可以预测其自身的行为 

**Authors**: Dhananjay Ashok, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2502.13329)  

**Abstract**: Autoregressive Language Models output text by sequentially predicting the next token to generate, with modern methods like Chain-of-Thought (CoT) prompting achieving state-of-the-art reasoning capabilities by scaling the number of generated tokens. However, are there times when we can infer how the model will behave (e.g. abstain from answering a question) early in the computation, making generation unnecessary? We show that internal representation of input tokens alone can often precisely predict, not just the next token, but eventual behavior over the entire output sequence. We leverage this capacity and learn probes on internal states to create early warning (and exit) systems. Specifically, if the probes can confidently estimate the way the LM is going to behave, then the system will avoid generating tokens altogether and return the estimated behavior instead. On 27 text classification datasets spanning five different tasks, we apply this method to estimate the eventual answer of an LM under CoT prompting, reducing inference costs by 65% (average) while suffering an accuracy loss of no more than 1.4% (worst case). We demonstrate the potential of this method to pre-emptively identify when a model will abstain from answering a question, fail to follow output format specifications, or give a low-confidence response. We explore the limits of this capability, showing that probes generalize to unseen datasets, but perform worse when LM outputs are longer and struggle to predict properties that require access to knowledge that the models themselves lack. Encouragingly, performance scales with model size, suggesting applicability to the largest of models 

**Abstract (ZH)**: 自回归语言模型通过顺序预测下一个令牌来生成文本，现代方法如链式思考（CoT）提示可扩展生成的令牌数量，从而实现最先进的推理能力。然而，在某些情况下，我们是否可以在计算早期就推断出模型的行为（例如，避免回答某个问题），从而使得生成变得没有必要？我们展示了仅通过输入令牌的内部表示，通常可以精确预测整个输出序列的最终行为，而不仅仅是下一个令牌。我们利用这种能力，通过在内部状态上学习探针来创建早期预警（并退出）系统。具体而言，如果探针能够自信地估计语言模型将如何行为，那么系统将避免生成任何令牌，而是返回估计的行为。在涵盖五个不同任务的27个文本分类数据集中，我们应用该方法来估算在CoT提示下语言模型的最终答案，平均减少84%的推理成本（降低65%），同时最糟糕的情况也仅降低了1.4%的准确率。我们演示了该方法的潜力，能够预先识别模型何时会避免回答问题、无法遵守输出格式规范或给出置信度低的回答。我们探讨了该能力的限制，表明探针能够泛化到未见过的数据集，但在语言模型输出较长且难以预测需要模型本身缺乏知识的属性时性能较弱。令人鼓舞的是，性能随着模型规模的增加而提高，表明这种方法适用于最大的模型。 

---
# Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models 

**Title (ZH)**: 逐步困惑度引导细化以提高大型语言模型中的高效链式推理 

**Authors**: Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2502.13260)  

**Abstract**: Chain-of-Thought (CoT) reasoning, which breaks down complex tasks into intermediate reasoning steps, has significantly enhanced the performance of large language models (LLMs) on challenging tasks. However, the detailed reasoning process in CoT often incurs long generation times and high computational costs, partly due to the inclusion of unnecessary steps. To address this, we propose a method to identify critical reasoning steps using perplexity as a measure of their importance: a step is deemed critical if its removal causes a significant increase in perplexity. Our method enables models to focus solely on generating these critical steps. This can be achieved through two approaches: refining demonstration examples in few-shot CoT or fine-tuning the model using selected examples that include only critical steps. Comprehensive experiments validate the effectiveness of our method, which achieves a better balance between the reasoning accuracy and efficiency of CoT. 

**Abstract (ZH)**: 链式推理（CoT，Chain-of-Thought）是一种将复杂任务分解为中间推理步骤的方法，极大地提高了大语言模型（LLMs，Large Language Models）在复杂任务上的性能。然而，CoT中的详细推理过程往往会导致生成时间的延长和计算成本的增加，部分原因是包括了一些不必要的步骤。为了应对这一问题，我们提出了一种方法，使用困惑度（perplexity）作为衡量这些步骤重要性的指标：如果移除某个步骤会导致困惑度显著增加，则该步骤被认为是关键步骤。我们的方法使得模型能够仅专注于生成这些关键步骤。这一目标可以通过两种途径实现：在少数样本链式推理（few-shot CoT）中精化示例，或者使用仅包含关键步骤的精选示例对模型进行微调。全面的实验验证了我们方法的有效性，该方法能够在保持CoT推理准确性和效率之间更好地平衡。 

---
# HumT DumT: Measuring and controlling human-like language in LLMs 

**Title (ZH)**: HumT DumT：测量和控制LLM中的人类似语言能力 

**Authors**: Myra Cheng, Sunny Yu, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.13259)  

**Abstract**: Should LLMs generate language that makes them seem human? Human-like language might improve user experience, but might also lead to overreliance and stereotyping. Assessing these potential impacts requires a systematic way to measure human-like tone in LLM outputs. We introduce HumT and SocioT, metrics for human-like tone and other dimensions of social perceptions in text data based on relative probabilities from an LLM. By measuring HumT across preference and usage datasets, we find that users prefer less human-like outputs from LLMs. HumT also offers insights into the impacts of anthropomorphism: human-like LLM outputs are highly correlated with warmth, social closeness, femininity, and low status, which are closely linked to the aforementioned harms. We introduce DumT, a method using HumT to systematically control and reduce the degree of human-like tone while preserving model performance. DumT offers a practical approach for mitigating risks associated with anthropomorphic language generation. 

**Abstract (ZH)**: 大型语言模型（LLM）生成的人类化语言是否合适？人类化的语言可能改善用户体验，但也可能导致过度依赖和刻板印象。评估这些潜在影响需要一种系统的方法来衡量LLM输出中的人类化语气。我们提出了HumT和SocioT这两个指标，用于衡量文本数据中人类化语气及其他社会感知维度，基于LLM的相对概率。通过对偏好和使用数据集中的HumT进行测量，我们发现用户更倾向于LLM生成的较少人类化的内容。HumT还揭示了拟人化的影响：人类化的LLM输出与温暖、社交亲近、女性化和低地位等因素高度相关，这些因素与前述的负面影响密切相关。我们介绍了DumT方法，该方法使用HumT系统地控制和降低人类化语气的程度，同时保持模型性能。DumT提供了一种实用的方法，用于减轻与拟人化语言生成相关的风险。 

---
# Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations 

**Title (ZH)**: 两票胜过一票：在战略性的LLM操纵下实现公平和准确的招聘 

**Authors**: Lee Cohen, Jack Hsieh, Connie Hong, Judy Hanwen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13221)  

**Abstract**: In an era of increasingly capable foundation models, job seekers are turning to generative AI tools to enhance their application materials. However, unequal access to and knowledge about generative AI tools can harm both employers and candidates by reducing the accuracy of hiring decisions and giving some candidates an unfair advantage. To address these challenges, we introduce a new variant of the strategic classification framework tailored to manipulations performed using large language models, accommodating varying levels of manipulations and stochastic outcomes. We propose a ``two-ticket'' scheme, where the hiring algorithm applies an additional manipulation to each submitted resume and considers this manipulated version together with the original submitted resume. We establish theoretical guarantees for this scheme, showing improvements for both the fairness and accuracy of hiring decisions when the true positive rate is maximized subject to a no false positives constraint. We further generalize this approach to an $n$-ticket scheme and prove that hiring outcomes converge to a fixed, group-independent decision, eliminating disparities arising from differential LLM access. Finally, we empirically validate our framework and the performance of our two-ticket scheme on real resumes using an open-source resume screening tool. 

**Abstract (ZH)**: 在基础模型日益强大的时代，求职者正在利用生成式AI工具来提升他们的申请材料。然而，生成式AI工具的获取途径和知识水平的不平等，可能会对雇主和求职者造成损害，降低招聘决策的准确性，并使一些求职者获得不公平的优势。为应对这些挑战，我们提出了一种针对使用大型语言模型进行操纵的新型战略分类框架，该框架能够适应不同操作程度和随机结果的变化。我们提出了一种“双票”方案，即招聘算法对每份提交的简历应用额外的操纵，并将此操纵版本与原始提交简历一起考虑。我们为该方案建立了理论保证，显示在无假阳性约束条件下使真实阳性率最大化时，该方案提升了招聘决策的公平性和准确性。我们进一步将这一方法扩展为“n票”方案，并证明招聘结果趋同于一个固定且与群体无关的决策，消除了因不同大型语言模型访问而产生的差异。最后，我们使用一个开源简历筛选工具在实际简历上验证了我们框架和“双票”方案的性能。 

---
# SmartLLM: Smart Contract Auditing using Custom Generative AI 

**Title (ZH)**: SmartLLM：基于定制生成式AI的智能合约审计 

**Authors**: Jun Kevin, Pujianto Yugopuspito  

**Link**: [PDF](https://arxiv.org/pdf/2502.13167)  

**Abstract**: Smart contracts are essential to decentralized finance (DeFi) and blockchain ecosystems but are increasingly vulnerable to exploits due to coding errors and complex attack vectors. Traditional static analysis tools and existing vulnerability detection methods often fail to address these challenges comprehensively, leading to high false-positive rates and an inability to detect dynamic vulnerabilities. This paper introduces SmartLLM, a novel approach leveraging fine-tuned LLaMA 3.1 models with Retrieval-Augmented Generation (RAG) to enhance the accuracy and efficiency of smart contract auditing. By integrating domain-specific knowledge from ERC standards and employing advanced techniques such as QLoRA for efficient fine-tuning, SmartLLM achieves superior performance compared to static analysis tools like Mythril and Slither, as well as zero-shot large language model (LLM) prompting methods such as GPT-3.5 and GPT-4. Experimental results demonstrate a perfect recall of 100% and an accuracy score of 70%, highlighting the model's robustness in identifying vulnerabilities, including reentrancy and access control issues. This research advances smart contract security by offering a scalable and effective auditing solution, supporting the secure adoption of decentralized applications. 

**Abstract (ZH)**: 智能合约是去中心化金融（DeFi）和区块链生态系统的核心组成部分，但由于编码错误和复杂的攻击向量，它们的脆弱性正在增加。传统的静态分析工具和现有的漏洞检测方法往往无法全面应对这些挑战，导致高误报率和无法检测动态漏洞。本文介绍了SmartLLM，这是一种利用微调的LLaMA 3.1模型结合检索增强生成（RAG）的新颖方法，以提高智能合约审计的准确性和效率。通过整合来自ERC标准的领域特定知识，并采用高效的微调技术如QLoRA，SmartLLM 在性能上优于如Mythril和Slither等静态分析工具，以及零样本大语言模型（LLM）提示方法如GPT-3.5和GPT-4。实验结果表明召回率为100%，准确率为70%，突显了该模型在识别包括重入攻击和访问控制问题在内的漏洞方面的稳健性。这项研究通过提供一种可扩展且有效的方法，推进了智能合约的安全性，支持去中心化应用程序的安全部署。 

---
# NestQuant: Nested Lattice Quantization for Matrix Products and LLMs 

**Title (ZH)**: NestQuant：嵌套格网量化在矩阵乘法和大语言模型中的应用 

**Authors**: Semyon Savkin, Eitan Porat, Or Ordentlich, Yury Polyanskiy  

**Link**: [PDF](https://arxiv.org/pdf/2502.09720)  

**Abstract**: Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent work have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Meta's SpinQuant (perplexity 7.3). Comparisons on various LLM evaluation benchmarks also show a reduction in performance degradation induced by quantization. 

**Abstract (ZH)**: 后训练量化（PTQ）已成为高效部署大型语言模型（LLMs）的关键技术。本研究提出了一种名为NestQuant的新型PTQ方案，该方案基于自相似嵌套网格。最近的研究从数学上证明，此类量化器对于低精度矩阵乘法具有信息论上的最优性。我们基于Gosset网格实现了一个具有低复杂度的实际可操作版本的NestQuant，使其成为任何矩阵乘法步骤（例如，在自我注意、MLP等步骤中）的即插即用量化器。例如，NestQuant将Llama-3-8B的权重、KV缓存和激活量化为4位， perplexity为6.6，这相较于未量化模型（perplexity为6.14）的perplexity差距减少了超过55%，并且相较于Meta公司的SpinQuant（perplexity为7.3）表现更为出色。在各种LLM评估基准上的比较也显示，量化引起的性能下降有所减少。 

---
