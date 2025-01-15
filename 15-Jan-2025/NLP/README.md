# PokerBench: Training Large Language Models to become Professional Poker Players 

**Title (ZH)**: PokerBench: 训练大型语言模型成为职业筹码争夺者

注释：在学术翻译中，为了使术语更加准确和符合专业惯例，将“Professional Poker Players”翻译为“职业筹码争夺者”。这是因为，在扑克职业领域，“Professional Poker Player”通常指的是专业扑克玩家，而“Professional”在特定的扑克语境中更强调其获得收益的能力。而“筹码争夺者”在中文语境下，更加形象地表达了“Poker Player”在扑克比赛中追求胜利和筹码积累的含义。 

**Authors**: Richard Zhuang, Akshat Gupta, Richard Yang, Aniket Rahane, Zhengyu Li, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2501.08328)  

**Abstract**: We introduce PokerBench - a benchmark for evaluating the poker-playing abilities of large language models (LLMs). As LLMs excel in traditional NLP tasks, their application to complex, strategic games like poker poses a new challenge. Poker, an incomplete information game, demands a multitude of skills such as mathematics, reasoning, planning, strategy, and a deep understanding of game theory and human psychology. This makes Poker the ideal next frontier for large language models. PokerBench consists of a comprehensive compilation of 11,000 most important scenarios, split between pre-flop and post-flop play, developed in collaboration with trained poker players. We evaluate prominent models including GPT-4, ChatGPT 3.5, and various Llama and Gemma series models, finding that all state-of-the-art LLMs underperform in playing optimal poker. However, after fine-tuning, these models show marked improvements. We validate PokerBench by having models with different scores compete with each other, demonstrating that higher scores on PokerBench lead to higher win rates in actual poker games. Through gameplay between our fine-tuned model and GPT-4, we also identify limitations of simple supervised fine-tuning for learning optimal playing strategy, suggesting the need for more advanced methodologies for effectively training language models to excel in games. PokerBench thus presents a unique benchmark for a quick and reliable evaluation of the poker-playing ability of LLMs as well as a comprehensive benchmark to study the progress of LLMs in complex game-playing scenarios. The dataset and code will be made available at: \url{this https URL}. 

**Abstract (ZH)**: 我们介绍了PokerBench——一个用于评估大型语言模型（LLMs）在扑克游戏中的能力的标准工具。作为传统NLP任务中的强者，LLMs在复杂的战略性游戏中，如扑克，的运用提出了新的挑战。扑克是一种不完整信息的游戏，需要多种技能，包括数学、推理、规划、策略和对博弈论及人类心理学的深刻理解。这使得扑克成为大型语言模型的下一个理想挑战领域。PokerBench包括11,000个最重要的场景综合，这些场景被分为摊前和摊后策略开发，并与训练有素的扑克玩家进行了合作。

我们评估了包括GPT-4、ChatGPT 3.5以及各种Llama和Gemma系列模型在内的知名模型，发现所有最先进的LLMs在玩最优扑克时都表现不佳。然而，在进行微调后，这些模型显示出明显的改进。我们通过让不同评分的模型相互竞争，验证了PokerBench，表明PokerBench上的较高分数会在实际扑克游戏中带来更高的胜率。通过我们的微调模型与GPT-4的对局，我们也发现简单的监督微调方法在学习最优策略方面存在局限性，强调了需要更高级的方法来有效训练语言模型以在游戏场景中表现出色。因此，PokerBench为快速可靠地评估LLMs在扑克中的能力提供了一个独特的基准，并为研究LLMs在复杂游戏场景中的进步提供了一个全面的基准。该数据集和代码将在此处获得：\url{this https URL}。 

---
# Exploring Robustness of Multilingual LLMs on Real-World Noisy Data 

**Title (ZH)**: 探索多语言大语言模型在真实世界噪声数据中的鲁棒性 

**Authors**: Amirhossein Aliakbarzadeh, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2501.08322)  

**Abstract**: Large Language Models (LLMs) are trained on Web data that might contain spelling errors made by humans. But do they become robust to similar real-world noise? In this paper, we investigate the effect of real-world spelling mistakes on the performance of 9 language models, with parameters ranging from 0.2B to 13B, in 3 different NLP tasks, namely Natural Language Inference (NLI), Name Entity Recognition (NER), and Intent Classification (IC). We perform our experiments on 6 different languages and build a dictionary of real-world noise for them using the Wikipedia edit history. We show that the performance gap of the studied models on the clean and noisy test data averaged across all the datasets and languages ranges from 2.3 to 4.3 absolute percentage points. In addition, mT5 models, in general, show more robustness compared to BLOOM, Falcon, and BERT-like models. In particular, mT5 (13B), was the most robust on average overall, across the 3 tasks, and in 4 of the 6 languages. 

**Abstract (ZH)**: 大规模语言模型（LLMs）是基于可能包含人类拼写错误的网络数据进行训练的。但它们是否对类似的现实世界噪音具有鲁棒性呢？在本文中，我们探讨了现实世界中的拼写错误对9种不同参数量（从0.2B到13B）的语言模型在3种不同自然语言处理（NLP）任务中的性能影响，具体任务为自然语言推理（NLI）、命名实体识别（NER）和意图分类（IC）。我们在6种不同语言上进行了实验，并使用维基百科的编辑历史构建了针对这些语言的现实世界噪音词典。结果显示，所研究模型在清洁数据和含有噪声数据上的性能差距（在所有数据集和语言上的平均值）范围为2.3到4.3个绝对百分点。此外，mT5模型通常比BLOOM、Falcon和BERT样式的模型更为鲁棒。特别是在3种任务和6种语言中的4种上，mT5（13B）模型表现出了最好的鲁棒性。 

---
# Enhancing Automated Interpretability with Output-Centric Feature Descriptions 

**Title (ZH)**: 增强基于输出的特征描述的自动可解释性 

**Authors**: Yoav Gur-Arieh, Roy Mayan, Chen Agassy, Atticus Geiger, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2501.08319)  

**Abstract**: Automated interpretability pipelines generate natural language descriptions for the concepts represented by features in large language models (LLMs), such as plants or the first word in a sentence. These descriptions are derived using inputs that activate the feature, which may be a dimension or a direction in the model's representation space. However, identifying activating inputs is costly, and the mechanistic role of a feature in model behavior is determined both by how inputs cause a feature to activate and by how feature activation affects outputs. Using steering evaluations, we reveal that current pipelines provide descriptions that fail to capture the causal effect of the feature on outputs. To fix this, we propose efficient, output-centric methods for automatically generating feature descriptions. These methods use the tokens weighted higher after feature stimulation or the highest weight tokens after applying the vocabulary "unembedding" head directly to the feature. Our output-centric descriptions better capture the causal effect of a feature on model outputs than input-centric descriptions, but combining the two leads to the best performance on both input and output evaluations. Lastly, we show that output-centric descriptions can be used to find inputs that activate features previously thought to be "dead". 

**Abstract (ZH)**: 自动可解释性管道生成大型语言模型（LLM）中特征代表的概念（如植物或句首词）的自然语言描述。这些描述是通过使用激活特征的输入生成的，这些输入可能是模型表示空间中的某个维度或方向。然而，识别激活输入的成本较高，特征在模型行为中的机械作用不仅取决于输入如何导致特征激活，还取决于特征激活如何影响输出。通过引导性评估，我们发现现有的管道提供的描述未能捕捉特征对输出的因果影响。为了解决这一问题，我们提出了以输出为中心的自动生成特征描述的有效方法。这些方法利用了特征刺激后加权较高的标记，或者应用词汇表“解嵌入”头部直接对特征进行操作后加权最高的标记。以输出为中心的描述比以输入为中心的描述更好地捕捉了特征对模型输出的因果影响，但结合两者在输入和输出评估中表现出最佳性能。最后，我们展示了以输出为中心的描述可以用于找到激活先前被认为是“无效”的特征的输入。 

---
# MiniMax-01: Scaling Foundation Models with Lightning Attention 

**Title (ZH)**: MiniMax-01: 通过闪电注意力机制扩展基础模型 

**Authors**: MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, Zijia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08313)  

**Abstract**: We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01, which are comparable to top-tier models while offering superior capabilities in processing longer contexts. The core lies in lightning attention and its efficient scaling. To maximize computational capacity, we integrate it with Mixture of Experts (MoE), creating a model with 32 experts and 456 billion total parameters, of which 45.9 billion are activated for each token. We develop an optimized parallel strategy and highly efficient computation-communication overlap techniques for MoE and lightning attention. This approach enables us to conduct efficient training and inference on models with hundreds of billions of parameters across contexts spanning millions of tokens. The context window of MiniMax-Text-01 can reach up to 1 million tokens during training and extrapolate to 4 million tokens during inference at an affordable cost. Our vision-language model, MiniMax-VL-01 is built through continued training with 512 billion vision-language tokens. Experiments on both standard and in-house benchmarks show that our models match the performance of state-of-the-art models like GPT-4o and Claude-3.5-Sonnet while offering 20-32 times longer context window. We publicly release MiniMax-01 at this https URL. 

**Abstract (ZH)**: 我们介绍了MiniMax-01系列，包括MiniMax-Text-01和MiniMax-VL-01，该系列模型在与顶级模型相当的同时，能够在处理更长的上下文方面展现出优越的能力。关键在于快速注意力机制及其高效扩展方式。为了最大化计算能力，我们将这一机制与专家混合（Mixture of Experts, MoE）相结合，创建了一个由32个专家和总共有456亿参数构成的模型，其中每个词元激活45.9亿参数。我们开发了优化的并行策略和高效计算-通信重叠技术，用于MoE和快速注意力机制。这种方法使得我们能够在数十亿参数的模型上高效地进行训练和推理，即使在包含百万级词元的上下文中也能保持高效。在训练过程中，MiniMax-Text-01的上下文窗口可达到100万词元，并且在推理过程中可以扩展到400万词元，同时成本可控。我们的跨模态模型MiniMax-VL-01通过继续训练获得了5120亿个跨模态词元。在标准基准和内部基准上的实验表明，我们的模型在性能上与诸如GPT-4o和Claude-3.5-Sonnet等最先进的模型相当，同时提供20-32倍更长的上下文窗口。我们已将MiniMax-01开源，并可在以下链接获得：[此处链接]。 

---
# Everybody Likes to Sleep: A Computer-Assisted Comparison of Object Naming Data from 30 Languages 

**Title (ZH)**: 每个人都喜欢睡眠：30种语言的物体命名数据的计算机辅助比较 

**Authors**: Alžběta Kučerová, Johann-Mattis List  

**Link**: [PDF](https://arxiv.org/pdf/2501.08312)  

**Abstract**: Object naming - the act of identifying an object with a word or a phrase - is a fundamental skill in interpersonal communication, relevant to many disciplines, such as psycholinguistics, cognitive linguistics, or language and vision research. Object naming datasets, which consist of concept lists with picture pairings, are used to gain insights into how humans access and select names for objects in their surroundings and to study the cognitive processes involved in converting visual stimuli into semantic concepts. Unfortunately, object naming datasets often lack transparency and have a highly idiosyncratic structure. Our study tries to make current object naming data transparent and comparable by using a multilingual, computer-assisted approach that links individual items of object naming lists to unified concepts. Our current sample links 17 object naming datasets that cover 30 languages from 10 different language families. We illustrate how the comparative dataset can be explored by searching for concepts that recur across the majority of datasets and comparing the conceptual spaces of covered object naming datasets with classical basic vocabulary lists from historical linguistics and linguistic typology. Our findings can serve as a basis for enhancing cross-linguistic object naming research and as a guideline for future studies dealing with object naming tasks. 

**Abstract (ZH)**: 物体命名——即用一个词或短语来识别物体的行为——是人际沟通中的一项基本技能，涉及多个学科领域，如心理语言学、认知语言学或语言与视觉研究。物体命名数据集通常是一系列概念列表，配以图片对，这些数据集用于深入了解人类是如何获取并选择周围物体名称的，并研究将视觉刺激转化为语义概念的认知过程。不幸的是，物体命名数据集往往缺乏透明性，且具有高度的个体性。我们的研究旨在通过使用多语言、计算机辅助的方法，将个体物体命名列表中的条目与统一的概念相连接，从而使当前的物体命名数据更加透明和可比较。目前，我们的样本连接了涵盖30种语言（属于10个不同语系）的17个物体命名数据集。我们通过查找在多数数据集中重复出现的概念，并将涵盖的物体命名数据集的概念空间与历史语言学和语言类型学中的经典基本词汇表进行比较，来说明比较数据集的探索方法。我们的发现可以成为提升跨语言物体命名研究的基础，并为未来处理物体命名任务的研究提供指导。 

---
# A Survey on Pedophile Attribution Techniques for Online Platforms 

**Title (ZH)**: 在线平台中关于恋童癖者认定技术的综述 

**Authors**: Hiba Fallatah, Ching Suen, Olga Ormandjieva  

**Link**: [PDF](https://arxiv.org/pdf/2501.08296)  

**Abstract**: Reliance on anonymity in social media has increased its popularity on these platforms among all ages. The availability of public Wi-Fi networks has facilitated a vast variety of online content, including social media applications. Although anonymity and ease of access can be a convenient means of communication for their users, it is difficult to manage and protect its vulnerable users against sexual predators. Using an automated identification system that can attribute predators to their text would make the solution more attainable. In this survey, we provide a review of the methods of pedophile attribution used in social media platforms. We examine the effect of the size of the suspect set and the length of the text on the task of attribution. Moreover, we review the most-used datasets, features, classification techniques and performance measures for attributing sexual predators. We found that few studies have proposed tools to mitigate the risk of online sexual predators, but none of them can provide suspect attribution. Finally, we list several open research problems. 

**Abstract (ZH)**: 社交媒体上的匿名性使用越来越高，使得各个年龄段的用户在这些平台上更加流行。公共Wi-Fi网络的可用性促进了各种在线内容的发展，包括社交媒体应用程序。虽然匿名性和易于访问可以为用户带来便捷的沟通手段，但管理和保护易受性行为害人者侵害的脆弱用户却非常困难。使用自动识别系统对性害人者进行归因可以帮助解决这一问题。本综述文章提供了一个关于社交媒体平台上性害人者归因方法的回顾。我们考察了嫌疑人群的大小和文本长度对归因任务的影响。此外，我们还回顾了用于归因性害人者的最常用数据集、特征、分类技术及性能评估指标。我们发现，虽然很少有研究提出了降低在线性害人者风险的工具，但没有任何研究能够提供嫌疑人的归因。最后，我们列出了若干有待解决的开放性研究问题。 

---
# HALoGEN: Fantastic LLM Hallucinations and Where to Find Them 

**Title (ZH)**: HALoGEN：卓越的大语言模型幻觉及其来源探索 

**Authors**: Abhilasha Ravichander, Shrusti Ghela, David Wadden, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2501.08292)  

**Abstract**: Despite their impressive ability to generate high-quality and fluent text, generative large language models (LLMs) also produce hallucinations: statements that are misaligned with established world knowledge or provided input context. However, measuring hallucination can be challenging, as having humans verify model generations on-the-fly is both expensive and time-consuming. In this work, we release HALoGEN, a comprehensive hallucination benchmark consisting of: (1) 10,923 prompts for generative models spanning nine domains including programming, scientific attribution, and summarization, and (2) automatic high-precision verifiers for each use case that decompose LLM generations into atomic units, and verify each unit against a high-quality knowledge source. We use this framework to evaluate ~150,000 generations from 14 language models, finding that even the best-performing models are riddled with hallucinations (sometimes up to 86% of generated atomic facts depending on the domain). We further define a novel error classification for LLM hallucinations based on whether they likely stem from incorrect recollection of training data (Type A errors), or incorrect knowledge in training data (Type B errors), or are fabrication (Type C errors). We hope our framework provides a foundation to enable the principled study of why generative models hallucinate, and advances the development of trustworthy large language models. 

**Abstract (ZH)**: 尽管生成型大型语言模型（LLMs）具有生成高质量流畅文本的强大能力，但它们也会产生幻觉：即模型生成的陈述与现有的世界知识或提供的上下文不一致。然而，衡量这些幻觉是具有挑战性的，因为让人类实时验证模型生成的内容既昂贵又耗时。在本研究中，我们发布了HALoGEN，这是一个综合性的幻觉基准，包括：（1）涵盖编程、科学引证和总结等九个领域的10,923个用于生成模型的提示；（2）针对每种应用场景提供自动的高精度验证器，将LLM生成的内容分解为原子单元，并使用高质量的知识来源验证每个单元。我们使用该框架评估了来自14个语言模型的约150,000个生成内容，发现即使表现最好的模型也充满了幻觉（有时在某些领域中，幻觉比例高达86%）。此外，我们还定义了一种新的错误分类方法，即根据幻觉是否源自训练数据的错误记忆（类型A错误）、训练数据中的错误知识（类型B错误）或虚构（类型C错误）来分类。我们希望这项框架能够为科学地研究生成模型为什么会产生幻觉提供基础，并促进可信的大规模语言模型的发展。 

---
# AfriHate: A Multilingual Collection of Hate Speech and Abusive Language Datasets for African Languages 

**Title (ZH)**: AfriHate：用于非洲语言的多语种仇恨言论和侮辱性语言数据集集锦 

**Authors**: Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Abinew Ali Ayele, David Ifeoluwa Adelani, Ibrahim Said Ahmad, Saminu Mohammad Aliyu, Nelson Odhiambo Onyango, Lilian D. A. Wanzare, Samuel Rutunda, Lukman Jibril Aliyu, Esubalew Alemneh, Oumaima Hourrane, Hagos Tesfahun Gebremichael, Elyas Abdi Ismail, Meriem Beloucif, Ebrahim Chekol Jibril, Andiswa Bukula, Rooweither Mabuya, Salomey Osei, Abigail Oppong, Tadesse Destaw Belay, Tadesse Kebede Guge, Tesfa Tegegne Asfaw, Chiamaka Ijeoma Chukwuneke, Paul Röttger, Seid Muhie Yimam, Nedjma Ousidhoum  

**Link**: [PDF](https://arxiv.org/pdf/2501.08284)  

**Abstract**: Hate speech and abusive language are global phenomena that need socio-cultural background knowledge to be understood, identified, and moderated. However, in many regions of the Global South, there have been several documented occurrences of (1) absence of moderation and (2) censorship due to the reliance on keyword spotting out of context. Further, high-profile individuals have frequently been at the center of the moderation process, while large and targeted hate speech campaigns against minorities have been overlooked. These limitations are mainly due to the lack of high-quality data in the local languages and the failure to include local communities in the collection, annotation, and moderation processes. To address this issue, we present AfriHate: a multilingual collection of hate speech and abusive language datasets in 15 African languages. Each instance in AfriHate is annotated by native speakers familiar with the local culture. We report the challenges related to the construction of the datasets and present various classification baseline results with and without using LLMs. The datasets, individual annotations, and hate speech and offensive language lexicons are available on this https URL 

**Abstract (ZH)**: 仇恨言论和滥用语言是全球性的现象，需要了解其社会和文化背景才能被理解、识别和管理。然而，在全球南方的许多地区，已经记录了以下两种情况：(1) 缺乏管理措施，(2) 依靠隔靴搔痒的关键词检测导致审查不力。此外，在这一过程中，重要人物经常处于中心位置，而针对少数群体的大规模和有针对性的仇恨言论运动则被忽略。这些问题主要是由于当地语言高质量数据的缺乏以及未能将当地社区纳入数据收集、标注和管理过程所致。为了解决这些问题，我们推出了AfriHate：在15种非洲语言中集成了仇恨言论和滥用语言的数据集。AfriHate中的每个实例都是由熟悉当地文化的手语者标注的。我们报告了构建这些数据集所遇到的挑战，并展示了使用和不使用大规模语言模型的各种分类 baseline 结果。这些数据集、单独的标注和仇恨言论及冒犯性语言词典可在以下网址获取：https://www.yoururl.com 

---
# Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing 

**Title (ZH)**: 探索大语言模型对社会人口统计学条件化同义词的鲁棒性 

**Authors**: Pulkit Arora, Akbar Karimi, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2501.08276)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in various NLP tasks. However, there are concerns about their reliability in different domains of linguistic variations. Many works have proposed robustness evaluation measures for local adversarial attacks, but we need globally robust models unbiased to different language styles. We take a broader approach to explore a wider range of variations across sociodemographic dimensions to perform structured reliability tests on the reasoning capacity of language models. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic styles. The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their reasoning capabilities in real-world, complex language scenarios. We also explore measures such as perplexity, explainability, and ATOMIC performance of paraphrases for fine-grained reliability analysis of LLMs on these sets. We find that demographic-specific paraphrasing significantly impacts the performance of language models, indicating that the subtleties of language variations remain a significant challenge. The code and dataset will be made available for reproducibility and future research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理（NLP）任务中展现了令人印象深刻的性能。然而，它们在不同的语言变体领域中的可靠性引起了关注。许多工作提出了针对局部对抗攻击的鲁棒性评估指标，但我们需要的是对不同语言风格具有全局鲁棒性的模型。我们采取了更为广泛的方法，以探索跨社会人口学维度的更广泛变化，并对其进行结构化的可靠性测试，以评估语言模型的推理能力。我们扩展了SocialIQA数据集，创建了基于社会人口学风格的多样化的同义替换集合。这些评估旨在深入了解LLMs在（a）使用精心设计的提示生成针对社会人口学的同义替换方面的能力；以及（b）在现实世界的复杂语言场景中进行推理的能力。我们还探讨了困惑度、可解释性和原子性能等指标，以进行这些集合上的LLMs细粒度可靠性的分析。我们的研究发现，针对不同社会人口学特征的同义替换显著影响了语言模型的性能，这表明语言变体的细微差别仍然是一个重大挑战。我们将代码和数据集公开，以实现可重复性和未来的研究。 

---
# Comparative Analysis of Efficient Adapter-Based Fine-Tuning of State-of-the-Art Transformer Models 

**Title (ZH)**: 基于高效适配器的细调方法对当今最佳变换器模型的比较分析 

**Authors**: Saad Mashkoor Siddiqui, Mohammad Ali Sheikh, Muhammad Aleem, Kajol R Singh  

**Link**: [PDF](https://arxiv.org/pdf/2501.08271)  

**Abstract**: In this work, we investigate the efficacy of various adapter architectures on supervised binary classification tasks from the SuperGLUE benchmark as well as a supervised multi-class news category classification task from Kaggle. Specifically, we compare classification performance and time complexity of three transformer models, namely DistilBERT, ELECTRA, and BART, using conventional fine-tuning as well as nine state-of-the-art (SoTA) adapter architectures. Our analysis reveals performance differences across adapter architectures, highlighting their ability to achieve comparable or better performance relative to fine-tuning at a fraction of the training time. Similar results are observed on the new classification task, further supporting our findings and demonstrating adapters as efficient and flexible alternatives to fine-tuning. This study provides valuable insights and guidelines for selecting and implementing adapters in diverse natural language processing (NLP) applications. 

**Abstract (ZH)**: 在这项研究中，我们探究了各种适配器架构在SuperGLUE基准的监督二分类任务以及Kaggle的监督多类别新闻类别分类任务中的有效性。具体来说，我们比较了三种变换器模型——DistilBERT、ELECTRA和BART，在传统微调以及九种最先进的（SOTA）适配器架构下的分类性能和时间复杂度。我们的分析揭示了不同适配器架构在性能上的差异，突显了适配器相对于微调能够在更短的训练时间内达到相当或更好的性能。在新的分类任务中也观察到了相似的结果，进一步支持了我们的发现，并证明了适配器作为微调高效且灵活的替代方案的有效性。本研究为在不同自然语言处理（NLP）应用中选择和实施适配器提供了宝贵的洞察和指南。 

---
# Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models 

**Title (ZH)**: 在长上下文环境中引导基于情境检索与推理的大语言模型 

**Authors**: Yifu Qiu, Varun Embar, Yizhe Zhang, Navdeep Jaitly, Shay B. Cohen, Benjamin Han  

**Link**: [PDF](https://arxiv.org/pdf/2501.08248)  

**Abstract**: Recent advancements in long-context language models (LCLMs) promise to transform Retrieval-Augmented Generation (RAG) by simplifying pipelines. With their expanded context windows, LCLMs can process entire knowledge bases and perform retrieval and reasoning directly -- a capability we define as In-Context Retrieval and Reasoning (ICR^2). However, existing benchmarks like LOFT often overestimate LCLM performance by providing overly simplified contexts. To address this, we introduce ICR^2, a benchmark that evaluates LCLMs in more realistic scenarios by including confounding passages retrieved with strong retrievers. We then propose three methods to enhance LCLM performance: (1) retrieve-then-generate fine-tuning, (2) retrieval-attention-probing, which uses attention heads to filter and de-noise long contexts during decoding, and (3) joint retrieval head training alongside the generation head. Our evaluation of five well-known LCLMs on LOFT and ICR^2 demonstrates significant gains with our best approach applied to Mistral-7B: +17 and +15 points by Exact Match on LOFT, and +13 and +2 points on ICR^2, compared to vanilla RAG and supervised fine-tuning, respectively. It even outperforms GPT-4-Turbo on most tasks despite being a much smaller model. 

**Abstract (ZH)**: 近年来，长上下文语言模型（LCLMs）的进步有望通过简化管道来改造检索增强生成（RAG）系统。借助扩展的上下文窗口，LCLMs能够处理整个知识库，并直接进行检索和推理——我们将其定义为上下文检索与推理（ICR^2）的能力。然而，现有的基准测试，例如LOFT，往往由于提供了过于简化的上下文而高估了LCLM的表现。为了解决这一问题，我们引入了ICR^2，这是一个基准测试，它通过包含使用强大检索器检索的干扰段落，在更现实的场景中评估LCLMs的表现。接着，我们提出了三种方法来提升LCLM的表现：(1) 检索-生成微调方法；(2) 检索-注意-探查，这种方法利用注意头在解码过程中筛选和去噪长上下文；(3) 结合生成头部和检索头部的联合训练。我们对LOFT和ICR^2的五个知名LCLMs进行的评估表明，在我们的最佳方法应用于Mistral-7B模型时，与传统的RAG和监督微调相比，在LOFT上的精确匹配分数分别提高了17和15个点，在ICR^2上的分数分别提高了13和2个点。即使与较小的模型GPT-4-Turbo相比，Mistral-7B也表现出色，不过在许多任务中仍然取得了更好的表现。 

---
# ASTRID -- An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems 

**Title (ZH)**: ASTRID —— 一种自动化且可扩展的 TRIaD，用于评估基于RAG的临床问答系统 

**Authors**: Mohita Chowdhury, Yajie Vera He, Aisling Higham, Ernest Lim  

**Link**: [PDF](https://arxiv.org/pdf/2501.08208)  

**Abstract**: Large Language Models (LLMs) have shown impressive potential in clinical question answering (QA), with Retrieval Augmented Generation (RAG) emerging as a leading approach for ensuring the factual accuracy of model responses. However, current automated RAG metrics perform poorly in clinical and conversational use cases. Using clinical human evaluations of responses is expensive, unscalable, and not conducive to the continuous iterative development of RAG systems. To address these challenges, we introduce ASTRID - an Automated and Scalable TRIaD for evaluating clinical QA systems leveraging RAG - consisting of three metrics: Context Relevance (CR), Refusal Accuracy (RA), and Conversational Faithfulness (CF). Our novel evaluation metric, CF, is designed to better capture the faithfulness of a model's response to the knowledge base without penalising conversational elements. To validate our triad, we curate a dataset of over 200 real-world patient questions posed to an LLM-based QA agent during surgical follow-up for cataract surgery - the highest volume operation in the world - augmented with clinician-selected questions for emergency, clinical, and non-clinical out-of-domain scenarios. We demonstrate that CF can predict human ratings of faithfulness better than existing definitions for conversational use cases. Furthermore, we show that evaluation using our triad consisting of CF, RA, and CR exhibits alignment with clinician assessment for inappropriate, harmful, or unhelpful responses. Finally, using nine different LLMs, we demonstrate that the three metrics can closely agree with human evaluations, highlighting the potential of these metrics for use in LLM-driven automated evaluation pipelines. We also publish the prompts and datasets for these experiments, providing valuable resources for further research and development. 

**Abstract (ZH)**: 大型语言模型（LLMs）在临床问答（QA）方面展现了令人印象深刻的潜力，检索增强生成（RAG）方法因其确保模型响应事实准确性的能力而成为领先的方法。然而，现有的自动化RAG评估指标在临床和对话场景中表现不佳。目前，使用临床人工评估响应成本高、无法扩展，也不利于RAG系统的持续迭代开发。为解决这些挑战，我们提出了一种名为ASTRID的自动化和可扩展的RAG评估框架，以评估利用RAG的临床QA系统，该框架包含三个指标：上下文相关性（CR）、拒绝准确性（RA）和对话一致性（CF）。我们提出的新型评估指标CF旨在更好地捕捉模型响应与知识库的一致性，而不惩罚对话元素。为了验证这一指标框架，我们编译了一个包含200多个真实世界患者问题的数据集，这些问题是在进行白内障手术术后随访时提出给基于LLM的QA代理的，其中还包括临床专家精选的问题，以涵盖急诊、临床和非临床领域。我们证明，CF能够比现有定义更好地预测对话场景中的人类一致性评级。此外，我们展示了使用包含CF、RA和CR的三重评估框架与临床评估的一致性，尤其是在不适当、有害或无用的响应方面。最后，我们使用九种不同的LLM进行测试，证明这三个指标与人类评估结果高度一致，突显了这些指标在基于LLM的自动评估管道中的应用潜力。我们还发布了这些实验的提示和数据集，为进一步的研究和开发提供了宝贵的资源。 

---
# ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving 

**Title (ZH)**: 阿斯姆攻击：评估大型语言模型在数学问题求解中对噪声上下文鲁棒性的表现 

**Authors**: Zain Ul Abedin, Shahzeb Qamar, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2501.08203)  

**Abstract**: While Large Language Models (LLMs) have shown impressive capabilities in math problem-solving tasks, their robustness to noisy inputs is not well-studied. In this work, we propose ArithmAttack to examine how robust the LLMs are when they encounter noisy prompts that contain extra noise in the form of punctuation marks. While being easy to implement, ArithmAttack does not cause any information loss since words are not added or deleted from the context. We evaluate the robustness of seven LLMs, including LLama3, Mistral, and Mathstral, on noisy GSM8K and MultiArith datasets. Our experiments suggest that all the studied models show vulnerability to such noise, with more noise leading to poorer performances. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在数学问题解决任务中展示了令人印象深刻的性能，但它们对噪声输入的鲁棒性研究尚不充分。在本研究中，我们提出ArithmAttack，以考察LLMs在遇到包含标点符号形式噪声的噪声提示时的鲁棒性。ArithmAttack 既易于实现，又不会造成信息损失，因为上下文中的单词没有被添加或删除。我们在Noisy GSM8K 和 MultiArith 数据集上对七种LLMs（包括LLama3、Mistral 和 Mathstral）的鲁棒性进行了评估。实验结果表明，所有研究的模型对这种噪声都表现出脆弱性，噪声越大，性能越差。 

---
# OpenCSG Chinese Corpus: A Series of High-quality Chinese Datasets for LLM Training 

**Title (ZH)**: OpenCSG 中文语料库：用于大语言模型训练的一系列高质量中文数据集 

**Authors**: Yijiong Yu, Ziyun Dai, Zekun Wang, Wei Wang, Ran Chen, Ji Pei  

**Link**: [PDF](https://arxiv.org/pdf/2501.08197)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their success heavily relies on the quality of pretraining corpora. For Chinese LLMs, the scarcity of high-quality Chinese datasets presents a significant challenge, often limiting their performance. To address this issue, we propose the OpenCSG Chinese Corpus, a series of high-quality datasets specifically designed for LLM pretraining, post-training, and fine-tuning. This corpus includes Fineweb-edu-chinese, Fineweb-edu-chinese-v2, Cosmopedia-chinese, and Smoltalk-chinese, each with distinct characteristics: Fineweb-edu datasets focus on filtered, high-quality content derived from diverse Chinese web sources; Cosmopedia-chinese provides synthetic, textbook-style data for knowledge-intensive training; and Smoltalk-chinese emphasizes stylistic and diverse chat-format data. The OpenCSG Chinese Corpus is characterized by its high-quality text, diverse coverage across domains, and scalable, reproducible data curation processes. Additionally, we conducted extensive experimental analyses, including evaluations on smaller parameter models, which demonstrated significant performance improvements in tasks such as C-Eval, showcasing the effectiveness of the corpus for training Chinese LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现了卓越的能力，但其成功很大程度上依赖于预训练语料库的质量。对于中文LLMs而言，高质量中文数据集的稀缺性构成了一个重要的挑战，常限制其性能。为解决这一问题，我们提出开源CSG中文语料库（OpenCSG Chinese Corpus），这是一种专门设计用于LLM预训练、后续训练和微调的高质量数据集系列。该语料库包括Fineweb-edu-chinese、Fineweb-edu-chinese-v2、Cosmopedia-chinese和Smoltalk-chinese，每个数据集具有各自的特性：Fineweb-edu数据集重点关注来自多种中文网络来源的过滤高质量内容；Cosmopedia-chinese提供了用于知识密集型训练的合成型、教科书风格数据；而Smoltalk-chinese则侧重于风格多样且形式多样的聊天格式数据。开源CSG中文语料库以高质量文本、跨领域的多样性覆盖和可扩展、可复现的数据编纂流程为特点。此外，我们进行了广泛的实验分析，包括对较小参数模型的评估，这些评估结果显示在C-Eval等任务上取得了显著性能提升，展示了该语料库对中国LLMs训练的有效性。 

---
# A Multi-Modal AI Copilot for Single-Cell Analysis with Instruction Following 

**Title (ZH)**: 单细胞分析中的指令跟随多模态AI协作员 

**Authors**: Yin Fang, Xinle Deng, Kangwei Liu, Ningyu Zhang, Jingyang Qian, Penghui Yang, Xiaohui Fan, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.08187)  

**Abstract**: Large language models excel at interpreting complex natural language instructions, enabling them to perform a wide range of tasks. In the life sciences, single-cell RNA sequencing (scRNA-seq) data serves as the "language of cellular biology", capturing intricate gene expression patterns at the single-cell level. However, interacting with this "language" through conventional tools is often inefficient and unintuitive, posing challenges for researchers. To address these limitations, we present InstructCell, a multi-modal AI copilot that leverages natural language as a medium for more direct and flexible single-cell analysis. We construct a comprehensive multi-modal instruction dataset that pairs text-based instructions with scRNA-seq profiles from diverse tissues and species. Building on this, we develop a multi-modal cell language architecture capable of simultaneously interpreting and processing both modalities. InstructCell empowers researchers to accomplish critical tasks-such as cell type annotation, conditional pseudo-cell generation, and drug sensitivity prediction-using straightforward natural language commands. Extensive evaluations demonstrate that InstructCell consistently meets or exceeds the performance of existing single-cell foundation models, while adapting to diverse experimental conditions. More importantly, InstructCell provides an accessible and intuitive tool for exploring complex single-cell data, lowering technical barriers and enabling deeper biological insights. 

**Abstract (ZH)**: 大型语言模型在解释复杂的自然语言指令方面表现出色，使其能够执行多种多样的任务。在生命科学领域，单细胞RNA测序（scRNA-seq）数据作为“细胞生物学的语言”，捕获了单细胞层面 intricate 的基因表达模式。然而，通过传统工具与这种“语言”互动往往是低效且不直观的，给研究人员带来了挑战。为解决这些局限性，我们提出了一种多模态AI协同助手InstructCell，它利用自然语言作为媒介进行更直接和灵活的单细胞分析。我们构建了一个全面的多模态指令数据集，将基于文本的指令与来自不同组织和物种的scRNA-seq谱型配对。在此基础上，我们开发了一种多模态细胞语言架构，能够同时解释和处理这两种模态。InstructCell使研究人员能够仅使用简单的自然语言命令完成关键任务，如细胞类型注释、条件伪细胞生成以及药物敏感性预测。广泛的评估表明，InstructCell在性能上往往与现有的单细胞基础模型相当甚至更优，且能够适应各种实验条件。更重要的是，InstructCell提供了一个易操作且直观的工具，用于探索复杂的单细胞数据，降低技术门槛并促进更深入的生物学洞察。 

---
# Potential and Perils of Large Language Models as Judges of Unstructured Textual Data 

**Title (ZH)**: 大型语言模型作为无结构文本数据裁判的潜力与风险 

**Authors**: Rewina Bedemariam, Natalie Perez, Sreyoshi Bhaduri, Satya Kapoor, Alex Gil, Elizabeth Conjar, Ikkei Itoku, David Theil, Aman Chadha, Naumaan Nayyar  

**Link**: [PDF](https://arxiv.org/pdf/2501.08167)  

**Abstract**: Rapid advancements in large language models have unlocked remarkable capabilities when it comes to processing and summarizing unstructured text data. This has implications for the analysis of rich, open-ended datasets, such as survey responses, where LLMs hold the promise of efficiently distilling key themes and sentiments. However, as organizations increasingly turn to these powerful AI systems to make sense of textual feedback, a critical question arises, can we trust LLMs to accurately represent the perspectives contained within these text based datasets? While LLMs excel at generating human-like summaries, there is a risk that their outputs may inadvertently diverge from the true substance of the original responses. Discrepancies between the LLM-generated outputs and the actual themes present in the data could lead to flawed decision-making, with far-reaching consequences for organizations. This research investigates the effectiveness of LLMs as judge models to evaluate the thematic alignment of summaries generated by other LLMs. We utilized an Anthropic Claude model to generate thematic summaries from open-ended survey responses, with Amazon's Titan Express, Nova Pro, and Meta's Llama serving as LLM judges. The LLM-as-judge approach was compared to human evaluations using Cohen's kappa, Spearman's rho, and Krippendorff's alpha, validating a scalable alternative to traditional human centric evaluation methods. Our findings reveal that while LLMs as judges offer a scalable solution comparable to human raters, humans may still excel at detecting subtle, context-specific nuances. This research contributes to the growing body of knowledge on AI assisted text analysis. We discuss limitations and provide recommendations for future research, emphasizing the need for careful consideration when generalizing LLM judge models across various contexts and use cases. 

**Abstract (ZH)**: 大规模语言模型的迅速发展解锁了处理和总结非结构化文本数据的惊人能力。这为分析丰富且开放式数据集（如调查回复）带来了重要影响，L大型语言模型（LLMs）有望高效地提炼关键主题和情绪。然而，随着组织越来越多地利用这些强大的人工智能系统来理解文本反馈，一个关键问题应运而生：我们能否信任LLMs准确地代表这些文本数据集中的视角？尽管LLMs在生成人类般自然的摘要方面表现出色，但它们的输出可能存在无意中的偏差，偏离原始回应的真实内容。LLMs生成的输出与实际数据主题之间的差异可能导致决策失误，对组织产生深远影响。本研究调查了LLMs作为评判模型评估由其他LLMs生成的摘要主题一致性的有效性。我们使用Anthropic的Claude模型从开放式调查回复中生成主题摘要，并使用Amazon的Titan Express、Nova Pro和Meta的Llama作为LLMs评判者。LLM评判者的方法与使用科恩κ、斯皮尔曼ρ和克里彭多夫α进行的人类评估进行了比较，验证了一种可扩展的人性化评估方法的替代方案。我们的研究发现，尽管作为评判者的LLMs提供了与人类评分者相比可扩展的解决方案，但人类评分者仍可能在检测细微的、情境特定的差异方面更为出色。本研究为AI辅助文本分析领域提供了更多信息。我们讨论了限制并提供了未来研究的建议，强调在不同背景和应用场景下一般化LLM评判模型时需要谨慎考虑。 

---
# Refusal Behavior in Large Language Models: A Nonlinear Perspective 

**Title (ZH)**: 大型语言模型中的拒绝行为：一种非线性视角 

**Authors**: Fabian Hildebrandt, Andreas Maier, Patrick Krauss, Achim Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2501.08145)  

**Abstract**: Refusal behavior in large language models (LLMs) enables them to decline responding to harmful, unethical, or inappropriate prompts, ensuring alignment with ethical standards. This paper investigates refusal behavior across six LLMs from three architectural families. We challenge the assumption of refusal as a linear phenomenon by employing dimensionality reduction techniques, including PCA, t-SNE, and UMAP. Our results reveal that refusal mechanisms exhibit nonlinear, multidimensional characteristics that vary by model architecture and layer. These findings highlight the need for nonlinear interpretability to improve alignment research and inform safer AI deployment strategies. 

**Abstract (ZH)**: 大型语言模型（LLMs）的拒绝行为使它们能够拒绝回应有害、不道德或不适当的内容，从而确保与伦理标准保持一致。本文探讨了六种来自三种架构家族的LLMs的拒绝行为。我们通过运用降维技术，包括主成分分析（PCA）、t-SNE和UMAP，挑战了拒绝行为是线性现象这一假设。研究结果表明，拒绝机制表现出非线性和多维的特征，这些特征因模型架构和层数的不同而异。这些发现突显了非线性可解释性的重要性，以改善对齐研究，并指导更安全的AI部署策略。 

---
# Consistency of Responses and Continuations Generated by Large Language Models on Social Media 

**Title (ZH)**: 大型语言模型在社交媒体上生成的回答和续写的一致性 

**Authors**: Wenlu Fan, Yuqi Zhu, Chenyang Wang, Bin Wang, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08102)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities in text generation, yet their emotional consistency and semantic coherence in social media contexts remain insufficiently understood. This study investigates how LLMs handle emotional content and maintain semantic relationships through continuation and response tasks using two open-source models: Gemma and Llama. By analyzing climate change discussions from Twitter and Reddit, we examine emotional transitions, intensity patterns, and semantic similarity between human-authored and LLM-generated content. Our findings reveal that while both models maintain high semantic coherence, they exhibit distinct emotional patterns: Gemma shows a tendency toward negative emotion amplification, particularly anger, while maintaining certain positive emotions like optimism. Llama demonstrates superior emotional preservation across a broader spectrum of affects. Both models systematically generate responses with attenuated emotional intensity compared to human-authored content and show a bias toward positive emotions in response tasks. Additionally, both models maintain strong semantic similarity with original texts, though performance varies between continuation and response tasks. These findings provide insights into LLMs' emotional and semantic processing capabilities, with implications for their deployment in social media contexts and human-AI interaction design. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本生成方面表现出显著的能力，但在社交媒体情境下情感一致性和语义连贯性方面仍然缺乏足够理解。本研究通过使用两个开源模型Gemma和Llama，利用Twitter和Reddit上的气候变化讨论数据，探讨LLMs如何处理情感内容并维持语义关系。我们通过续写和回应任务来分析人类和模型生成的内容间的情感转换、情感强度模式及语义相似性。研究发现，尽管两种模型都保持了较高的语义连贯性，但它们在情感模式上表现出不同的特点：Gemma倾向于放大负面情感，尤其是愤怒，同时保持一定比例的正面情感，如乐观；而Llama则在更广泛的范围内表现出优越的情感保留能力。两种模型系统性地生成的情感强度较低的回应内容，并且在回应任务中倾向于积极情感。此外，两种模型在续写和回应任务中都保持了与原始文本较强的语义相似性，但任务表现存在差异。这些发现揭示了LLMs的情感和语义处理能力，对于其在社交媒体情境中的应用和人机交互设计具有重要启示。 

---
# Dynamic Multimodal Sentiment Analysis: Leveraging Cross-Modal Attention for Enabled Classification 

**Title (ZH)**: 动态多模态情感分析：利用跨模态注意力实现enabled分类 

**Authors**: Hui Lee, Singh Suniljit, Yong Siang Ong  

**Link**: [PDF](https://arxiv.org/pdf/2501.08085)  

**Abstract**: This paper explores the development of a multimodal sentiment analysis model that integrates text, audio, and visual data to enhance sentiment classification. The goal is to improve emotion detection by capturing the complex interactions between these modalities, thereby enabling more accurate and nuanced sentiment interpretation. The study evaluates three feature fusion strategies -- late stage fusion, early stage fusion, and multi-headed attention -- within a transformer-based architecture. Experiments were conducted using the CMU-MOSEI dataset, which includes synchronized text, audio, and visual inputs labeled with sentiment scores. Results show that early stage fusion significantly outperforms late stage fusion, achieving an accuracy of 71.87\%, while the multi-headed attention approach offers marginal improvement, reaching 72.39\%. The findings suggest that integrating modalities early in the process enhances sentiment classification, while attention mechanisms may have limited impact within the current framework. Future work will focus on refining feature fusion techniques, incorporating temporal data, and exploring dynamic feature weighting to further improve model performance. 

**Abstract (ZH)**: 本文探讨了一种多模态情感分析模型的发展，该模型整合了文本、音频和视觉数据，以提高情感分类的准确性。目标是通过捕捉这些模态之间的复杂交互来改进情绪检测，从而实现更为准确和细腻的情感解读。研究在基于变压器的架构中评估了三种特征融合策略——后期融合、早期融合和多头注意力机制。实验使用了CMU-MOSEI数据集，该数据集包含了同步的文本、音频和视觉输入，并用情感评分进行标注。结果显示，早期融合显著优于后期融合，准确率达到71.87%，而多头注意力机制仅带来微小的提升，准确率达到了72.39%。研究结果表明，在处理过程中早期整合模态可以增强情感分类的效果，而注意力机制在现有框架中的影响有限。未来的工作将集中在改进特征融合技术、引入时间数据和探索动态特征加权等方面，以进一步提高模型性能。 

---
# Exploring Narrative Clustering in Large Language Models: A Layerwise Analysis of BERT 

**Title (ZH)**: 探索大型语言模型中的叙述聚类：BERT的分层分析 

**Authors**: Awritrojit Banerjee, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2501.08053)  

**Abstract**: This study investigates the internal mechanisms of BERT, a transformer-based large language model, with a focus on its ability to cluster narrative content and authorial style across its layers. Using a dataset of narratives developed via GPT-4, featuring diverse semantic content and stylistic variations, we analyze BERT's layerwise activations to uncover patterns of localized neural processing. Through dimensionality reduction techniques such as Principal Component Analysis (PCA) and Multidimensional Scaling (MDS), we reveal that BERT exhibits strong clustering based on narrative content in its later layers, with progressively compact and distinct clusters. While strong stylistic clustering might occur when narratives are rephrased into different text types (e.g., fables, sci-fi, kids' stories), minimal clustering is observed for authorial style specific to individual writers. These findings highlight BERT's prioritization of semantic content over stylistic features, offering insights into its representational capabilities and processing hierarchy. This study contributes to understanding how transformer models like BERT encode linguistic information, paving the way for future interdisciplinary research in artificial intelligence and cognitive neuroscience. 

**Abstract (ZH)**: 本研究旨在探讨基于转换器的大型语言模型BERT的内部机制，尤其是其在各层中对叙事内容和作者风格进行聚类的能力。我们使用GPT-4生成的数据集进行分析，该数据集包含多样化的语义内容和风格变异。通过分析BERT在各层中的激活状态，揭示局部神经处理的模式。利用主成分分析（PCA）和多维标度（MDS）等降维技术，我们发现BERT在较晚的层中表现出强大的基于叙事内容的聚类，聚类随着时间的推移逐渐紧凑且分明。虽然在将叙事重新表述为不同文本类型（如寓言、科幻小说、儿童故事）时可能会出现强烈的风格聚类，但针对个别作家的作者风格聚类却很少观察到。这些发现突显了BERT在语义内容优先于风格特征处理方面的特点，为理解其表示能力和处理层次结构提供了见解。本研究为理解如BERT这样的转换器模型如何编码语言信息作出了贡献，并为进一步跨学科研究人工智能和认知神经科学铺平了道路。 

---
# READ: Reinforcement-based Adversarial Learning for Text Classification with Limited Labeled Data 

**Title (ZH)**: READ：基于强化学习的有限标注数据条件下文本分类的对抗学习方法 

**Authors**: Rohit Sharma, Shanu Kumar, Avinash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2501.08035)  

**Abstract**: Pre-trained transformer models such as BERT have shown massive gains across many text classification tasks. However, these models usually need enormous labeled data to achieve impressive performances. Obtaining labeled data is often expensive and time-consuming, whereas collecting unlabeled data using some heuristics is relatively much cheaper for any task. Therefore, this paper proposes a method that encapsulates reinforcement learning-based text generation and semi-supervised adversarial learning approaches in a novel way to improve the model's performance. Our method READ, Reinforcement-based Adversarial learning, utilizes an unlabeled dataset to generate diverse synthetic text through reinforcement learning, improving the model's generalization capability using adversarial learning. Our experimental results show that READ outperforms the existing state-of-art methods on multiple datasets. 

**Abstract (ZH)**: 预训练变换器模型，如BERT，在多种文本分类任务中表现出巨大提升。然而，这些模型通常需要大量的标记数据才能取得显著的性能。获取标记数据往往代价高昂且耗时，而通过某些启发式方法收集未标记数据则对于任何任务来说相对便宜得多。因此，本论文提出了一种方法，通过新颖地结合基于强化学习的文本生成和半监督对抗学习方法来提高模型的性能。我们提出的READ（Reinforcement-based Adversarial Learning）方法使用未标记数据集通过强化学习生成多样化的合成文本，并利用对抗学习提高模型的泛化能力。我们的实验结果表明，READ在多个数据集上优于现有最先进的方法。 

---
# TriAdaptLoRA: Brain-Inspired Triangular Adaptive Low-Rank Adaptation for Parameter-Efficient Fine-Tuning 

**Title (ZH)**: TriAdaptLoRA：受脑启发的三角形自适应低秩调整方法，实现参数高效微调 

**Authors**: Yao Liang, Yuwei Wang, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2501.08008)  

**Abstract**: The fine-tuning of Large Language Models (LLMs) is pivotal for achieving optimal performance across diverse downstream tasks. However, while full fine-tuning delivers superior results, it entails significant computational and resource costs. Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA, address these challenges by reducing the number of trainable parameters, but they often struggle with rank adjustment efficiency and task-specific adaptability. We propose Triangular Adaptive Low-Rank Adaptation (TriAdaptLoRA), a novel PEFT framework inspired by neuroscience principles, which dynamically optimizes the allocation of trainable parameters. TriAdaptLoRA introduces three key innovations: 1) a triangular split of transformation matrices into lower and upper triangular components to maximize parameter utilization, 2) a parameter importance metric based on normalized Frobenius norms for efficient adaptation, and 3) an adaptive rank-growth strategy governed by dynamic thresholds, allowing flexible parameter allocation across training steps. Experiments conducted on a variety of natural language understanding and generation tasks demonstrate that TriAdaptLoRA consistently outperforms existing PEFT methods. It achieves superior performance, enhanced stability, and reduced computational overhead, particularly under linear threshold-driven rank growth. These results highlight its efficacy as a scalable and resource-efficient solution for fine-tuning LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的微调对于实现多种下游任务的最优性能至关重要。然而，虽然全微调可以提供最佳的结果，但也会带来显著的计算和资源成本。参数高效微调（PEFT）方法，如LoRA，通过减少可训练参数的数量来应对这些挑战，但它们在秩调整效率和任务特定适应性方面通常存在困难。我们提出了Triangular Adaptive Low-Rank Adaptation（TriAdaptLoRA），这是一种受神经科学原理启发的新颖PEFT框架，能够动态优化可训练参数的分配。TriAdaptLoRA引入了三项关键创新：1）将变换矩阵三角分裂为下三角和上三角分量，以最大化参数利用率；2）基于归一化Frobenius范数的参数重要性度量，以实现高效的适应；3）由动态阈值控制的自适应秩增长策略，允许在训练步骤中灵活分配参数。在各种自然语言理解和生成任务上的实验表明，TriAdaptLoRA在多个方面优于现有的PEFT方法，实现了更优异的性能、更好的稳定性和更低的计算开销，特别是在线性阈值驱动的秩增长情况下。这些结果突显了其作为可扩展且资源高效的LLM微调解决方案的有效性。 

---
# Formalising lexical and syntactic diversity for data sampling in French 

**Title (ZH)**: 正式化法语中的词汇和句法多样性以用于数据采样 

**Authors**: Louis Estève, Manon Scholivet, Agata Savary  

**Link**: [PDF](https://arxiv.org/pdf/2501.08003)  

**Abstract**: Diversity is an important property of datasets and sampling data for diversity is useful in dataset creation. Finding the optimally diverse sample is expensive, we therefore present a heuristic significantly increasing diversity relative to random sampling. We also explore whether different kinds of diversity -- lexical and syntactic -- correlate, with the purpose of sampling for expensive syntactic diversity through inexpensive lexical diversity. We find that correlations fluctuate with different datasets and versions of diversity measures. This shows that an arbitrarily chosen measure may fall short of capturing diversity-related properties of datasets. 

**Abstract (ZH)**: 多样性是数据集的一个重要属性，采样数据以实现多样性在数据集构建中是有用的。找到最优多样性的样本代价较高，因此我们提出了一种启发式方法，该方法显著增加了样本的多样性，相对于随机采样而言。我们还探讨了不同类型多样性——词汇多样性和语法多样性——之间的相关性，目的是通过低成本的词汇多样性来实现高成本的语法多样性采样。我们的研究发现，不同数据集和多样性度量的不同版本之间的相关性有所波动。这表明，任意选择的度量标准可能无法充分捕捉数据集的多样性相关属性。 

---
# "Wait, did you mean the doctor?": Collecting a Dialogue Corpus for Topical Analysis 

**Title (ZH)**: “等等，你是说医生吗？”：收集用于主题分析的对话语料库 

**Authors**: Amandine Decker, Vincent Tourneur, Maxime Amblard, Ellen Breitholtz  

**Link**: [PDF](https://arxiv.org/pdf/2501.07947)  

**Abstract**: Dialogue is at the core of human behaviour and being able to identify the topic at hand is crucial to take part in conversation. Yet, there are few accounts of the topical organisation in casual dialogue and of how people recognise the current topic in the literature. Moreover, analysing topics in dialogue requires conversations long enough to contain several topics and types of topic shifts. Such data is complicated to collect and annotate. In this paper we present a dialogue collection experiment which aims to build a corpus suitable for topical analysis. We will carry out the collection with a messaging tool we developed. 

**Abstract (ZH)**: 对话是人类行为的核心，能够识别当前话题对于参与对话至关重要。然而，在文献中对于非正式对话中的话题组织以及人们如何识别当前话题的研究仍然较少。此外，分析对话中的话题需要有足够的对话长度以包含多种话题及其转换类型。此类数据的收集和标注过程较为复杂。本文我们将介绍一项对话数据收集实验，旨在构建适合话题分析的语料库。我们将使用自行开发的消息工具来进行数据收集。 

---
# GRAPHMOE: Amplifying Cognitive Depth of Mixture-of-Experts Network via Introducing Self-Rethinking Mechanism 

**Title (ZH)**: GRAPHMOE：通过引入自我反思机制增强混合专家网络的认知深度 

**Authors**: Chen Tang, Bo Lv, Zifan Zheng, Bohao Yang, Kun Zhao, Ning Liao, Xiaoxing Wang, Feiyu Xiong, Zhiyu Li, Nayu Liu, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07890)  

**Abstract**: Traditional Mixture-of-Experts (MoE) networks benefit from utilizing multiple smaller expert models as opposed to a single large network. However, these experts typically operate independently, leaving a question open about whether interconnecting these models could enhance the performance of MoE networks. In response, we introduce GRAPHMOE, a novel method aimed at augmenting the cognitive depth of language models via a self-rethinking mechanism constructed on Pseudo GraphMoE networks. GRAPHMOE employs a recurrent routing strategy to simulate iterative thinking steps, thereby facilitating the flow of information among expert nodes. We implement the GRAPHMOE architecture using Low-Rank Adaptation techniques (LoRA) and conduct extensive experiments on various benchmark datasets. The experimental results reveal that GRAPHMOE outperforms other LoRA based models, achieving state-of-the-art (SOTA) performance. Additionally, this study explores a novel recurrent routing strategy that may inspire further advancements in enhancing the reasoning capabilities of language models. 

**Abstract (ZH)**: 传统的混合专家（MoE）网络通过使用多个较小的专家模型而非单一的大网络来获益。然而，这些专家通常独立运行，这引发了一个问题，即是否可以通过连接这些模型来提升MoE网络的性能。为应对这一挑战，我们提出了GRAPHMOE，一种通过构建在伪GraphMoE网络之上的自我反思机制来增加语言模型认知深度的新型方法。GRAPHMOE采用递归路由策略模拟迭代思考步骤，从而促进了专家节点之间的信息流动。我们采用低秩适应技术（LoRA）构建了GRAPHMOE架构，并在各种基准数据集上进行了广泛实验。实验结果表明，GRAPHMOE优于其他基于LoRA的模型，达到了目前的最先进性能（SOTA）。此外，本研究探讨了一种新的递归路由策略，这可能为提升语言模型推理能力的进一步发展提供启示。 

---
# Continual Learning with Embedding Layer Surgery and Task-wise Beam Search using Whisper 

**Title (ZH)**: 使用嵌入层手术和任务导向的束搜索的连续学习——基于Whisper的研究 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2501.07875)  

**Abstract**: Current Multilingual ASR models only support a fraction of the world's languages. Continual Learning (CL) aims to tackle this problem by adding new languages to pre-trained models while avoiding the loss of performance on existing languages, also known as Catastrophic Forgetting (CF). However, existing CL methods overlook the adaptation of the token embedding lookup table at the decoder, despite its significant contribution to CF. We propose Embedding Layer Surgery where separate copies of the token embeddings are created for each new languages, and one of the copies is selected to replace the old languages embeddings when transcribing the corresponding new language. Unfortunately, this approach means LID errors also cause incorrect ASR embedding selection. Our Task-wise Beam Search allows self-correction for such mistakes. By adapting Whisper to 10 hours of data for each of 10 unseen languages from Common Voice, results show that our method reduces the Average WER (AWER) of pre-trained languages from 14.2% to 11.9% compared with Experience Replay, without compromising the AWER of the unseen languages. 

**Abstract (ZH)**: 当前的多语言ASR模型仅支持世界上的一小部分语言。连续学习（CL）旨在通过向预训练模型添加新语言来解决这一问题，同时避免在现有语言上性能下降的问题，即灾难性遗忘（CF）。然而，现有的CL方法忽略了解码器中的嵌入查找表的适应性调整，尽管嵌入查找表对CF有重要贡献。我们提出了一种嵌入层手术方法，其中为每种新语言创建了嵌入的单独副本，并在转录相应新语言时选择一个副本替换旧语言的嵌入。不幸的是，这种方法意味着识别语言错误也会导致ASR嵌入选择错误。我们提出的任务自适应束搜索允许自我修正此类错误。通过将Whisper适应来自Common Voice的10种未见过的语言的数据，每种语言为10小时，结果显示我们的方法在预训练语言上的平均词错误率（AWER）从14.2%降低到11.9%，而未影响未见过语言的AWER。 

---
# ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding 

**Title (ZH)**: ReARTeR：具有可信过程奖励的检索增强推理 

**Authors**: Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao Zhang, Song Yang, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.07861)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems for Large Language Models (LLMs) hold promise in knowledge-intensive tasks but face limitations in complex multi-step reasoning. While recent methods have integrated RAG with chain-of-thought reasoning or test-time search using Process Reward Models (PRMs), these approaches encounter challenges such as a lack of explanations, bias in PRM training data, early-step bias in PRM scores, and insufficient post-training optimization of reasoning potential. To address these issues, we propose Retrieval-Augmented Reasoning through Trustworthy Process Rewarding (ReARTeR), a framework that enhances RAG systems' reasoning capabilities through post-training and test-time scaling. At test time, ReARTeR introduces Trustworthy Process Rewarding via a Process Reward Model for accurate scalar scoring and a Process Explanation Model (PEM) for generating natural language explanations, enabling step refinement. During post-training, it utilizes Monte Carlo Tree Search guided by Trustworthy Process Rewarding to collect high-quality step-level preference data, optimized through Iterative Preference Optimization. ReARTeR addresses three core challenges: (1) misalignment between PRM and PEM, tackled through off-policy preference learning; (2) bias in PRM training data, mitigated by balanced annotation methods and stronger annotations for challenging examples; and (3) early-step bias in PRM, resolved through a temporal-difference-based look-ahead search strategy. Experimental results on multi-step reasoning benchmarks demonstrate significant improvements, underscoring ReARTeR's potential to advance the reasoning capabilities of RAG systems. 

**Abstract (ZH)**: 增强检索的生成（RAG）系统在大型语言模型（LLMs）中的知识密集型任务中展现出潜力，但在复杂多步推理方面面临局限。尽管最近的方法已经将RAG与链式推理或测试时的搜索（使用过程奖励模型PRM）相结合，但这些方法仍然存在诸如缺乏解释、PRM训练数据中的偏差、PRM评分的早期步骤偏差以及推理潜力的测试后训练优化不足等问题。为了解决这些问题，我们提出了一种名为“信赖过程奖励增强推理”（ReARTeR）的框架，该框架通过测试时和测试后训练的放大来增强RAG系统的推理能力。在测试时，ReARTeR通过引入一种过程奖励模型进行准确的标量评分，并通过过程解释模型（PEM）生成自然语言解释，以实现步骤细化。在测试后训练阶段，它利用通过信赖过程奖励指导的蒙特卡罗树搜索收集高质量的步骤偏好数据，并通过迭代偏好优化进行优化。ReARTeR解决的三个核心挑战包括：（1）PRM和PEM之间的不一致，通过离策偏好学习解决；（2）PRM训练数据中的偏差，通过平衡注释方法和对具有挑战性的例子进行更严格的注释来缓解；（3）PRM中的早期步骤偏差，通过基于时差的前瞻搜索策略解决。在多步骤推理基准测试中的实验结果表明，ReARTeR在显著提高RAG系统的推理能力方面具有巨大潜力。 

---
# Optimizing Language Models for Grammatical Acceptability: A Comparative Study of Fine-Tuning Techniques 

**Title (ZH)**: 优化语言模型的语法可接受性：细调技术的比较研究 

**Authors**: Shobhit Ratan, Farley Knight, Ghada Jerfel, Sze Chung Ho  

**Link**: [PDF](https://arxiv.org/pdf/2501.07853)  

**Abstract**: This study explores the fine-tuning (FT) of the Open Pre-trained Transformer (OPT-125M) for grammatical acceptability tasks using the CoLA dataset. By comparing Vanilla-Fine-Tuning (VFT), Pattern-Based-Fine-Tuning (PBFT), and Parameter-Efficient Fine-Tuning techniques (PEFT) like Low-Rank Adaptation (LoRA), we demonstrate significant improvements in computational efficiency while maintaining high accuracy. Our experiments reveal that while VFT achieves the highest accuracy (81.2%), LoRA enhancing FT by reducing memory usage and iteration time by more than 50%, and increases accuracy in PBFT case. Context Distillation (CD), though computationally efficient, underperformed with accuracy around 31%. Our findings contribute to democratizing access to large language models (LLM) by reducing computational barriers. 

**Abstract (ZH)**: 本研究探讨了使用 CoLA 数据集对 Open Pre-trained Transformer (OPT-125M) 进行微调 (FT) 以完成语法正确性任务的方法。通过对比常规微调（VFT）、基于模式的微调（PBFT）以及参数高效微调技术（PEFT，如低秩适应 LoRA），我们展示了在保持高准确度的同时显著提高计算效率。实验结果表明，尽管常规微调 (VFT) 达到了最高的准确率（81.2%），但通过减少内存使用和迭代时间超过 50% 来增强微调的 LoRA 技术，在 PBFT 情况下提高了准确率，而尽管计算效率高，上下文提炼 (CD) 的准确率仅为约 31%。我们的发现有助于通过降低计算门槛来普及大型语言模型 (LLM) 的访问。 

---
# Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning 

**Title (ZH)**: 利用图形进行推理：构架隐含知识以增强LLMs的推理能力 

**Authors**: Haoyu Han, Yaochen Xie, Hui Liu, Xianfeng Tang, Sreyashi Nag, William Headden, Hui Liu, Yang Li, Chen Luo, Shuiwang Ji, Qi He, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07845)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable success across a wide range of tasks; however, they still encounter challenges in reasoning tasks that require understanding and inferring relationships between distinct pieces of information within text sequences. This challenge is particularly pronounced in tasks involving multi-step processes, such as logical reasoning and multi-hop question answering, where understanding implicit relationships between entities and leveraging multi-hop connections in the given context are crucial. Graphs, as fundamental data structures, explicitly represent pairwise relationships between entities, thereby offering the potential to enhance LLMs' reasoning capabilities. External graphs have proven effective in supporting LLMs across multiple tasks. However, in many reasoning tasks, no pre-existing graph structure is provided. Can we structure implicit knowledge derived from context into graphs to assist LLMs in reasoning? In this paper, we propose Reasoning with Graphs (RwG) by first constructing explicit graphs from the context and then leveraging these graphs to enhance LLM reasoning performance on reasoning tasks. Extensive experiments demonstrate the effectiveness of the proposed method in improving both logical reasoning and multi-hop question answering tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中取得了显著的成功；然而，它们在需要理解并推断文本序列中不同信息之间的关系的推理任务中仍然面临挑战。这种挑战在涉及多步骤过程的任务中尤为明显，例如逻辑推理和多跳问答，因为在这些任务中理解实体之间的隐含关系并利用给定上下文中的多跳连接至关重要。图作为一种基本的数据结构，明确地表示实体对之间的关系，从而有可能增强LLMs的推理能力。外部图已经在多个任务中有效地支持了LLMs。然而，在许多推理任务中，并不提供现成的图结构。我们能否将从上下文中推导出的隐含知识结构化为图，以辅助LLMs进行推理？在本文中，我们提出了一种名为“Graph-based Reasoning”（RwG）的方法，即首先从上下文中构建显式的图结构，然后利用这些图来提升LLMs在推理任务中的性能。广泛的实验结果表明，所提出的方法在提高逻辑推理和多跳问答任务的效果方面具有显著的效果。 

---
# Real-time Verification and Refinement of Language Model Text Generation 

**Title (ZH)**: 实时验证和细化语言模型文本生成 

**Authors**: Joonho Ko, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07824)  

**Abstract**: Large language models (LLMs) have shown remarkable performance across a wide range of natural language tasks. However, a critical challenge remains in that they sometimes generate factually incorrect answers. To address this, while many previous work has focused on identifying errors in their generation and further refining them, they are slow in deployment since they are designed to verify the response from LLMs only after their entire generation (from the first to last tokens) is done. Further, we observe that once LLMs generate incorrect tokens early on, there is a higher likelihood that subsequent tokens will also be factually incorrect. To this end, in this work, we propose Streaming-VR (Streaming Verification and Refinement), a novel approach designed to enhance the efficiency of verification and refinement of LLM outputs. Specifically, the proposed Streaming-VR enables on-the-fly verification and correction of tokens as they are being generated, similar to a streaming process, ensuring that each subset of tokens is checked and refined in real-time by another LLM as the LLM constructs its response. Through comprehensive evaluations on multiple datasets, we demonstrate that our approach not only enhances the factual accuracy of LLMs, but also offers a more efficient solution compared to prior refinement methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类自然语言任务中表现出色。然而，一个关键挑战在于它们有时会产生事实错误的答案。为了解决这一问题，尽管许多先前的工作集中在识别LLMs生成中的错误并进一步对其进行修正，但这些方法在部署上通常较慢，因为它们设计成只在LLMs完成其整个生成过程（从第一个到最后一个词元）后才对其进行验证。此外，我们观察到，一旦LLMs在早期生成错误词元，后续词元更有可能也是事实错误的。基于此，本文提出了一种名为Streaming-VR（流式验证与修正）的新颖方法，旨在提高LLMs输出验证与修正的效率。具体而言，Streaming-VR方法能够实现实时验证和修正生成中的词元，类似于流式处理过程，确保每个词元子集在LLMs构建其响应的同时被另一个LLMs实时检查和修正。通过在多个数据集上的全面评估，我们证明了该方法不仅提高了LLMs的事实准确性，而且提供了比先前修正方法更高效的解决方案。 

---
# A Multi-Encoder Frozen-Decoder Approach for Fine-Tuning Large Language Models 

**Title (ZH)**: 一种多编码器冻结解码器方法用于微调大规模语言模型 

**Authors**: Kaustubh D. Dhole  

**Link**: [PDF](https://arxiv.org/pdf/2501.07818)  

**Abstract**: Among parameter-efficient fine-tuning methods, freezing has emerged as a popular strategy for speeding up training, reducing catastrophic forgetting, and improving downstream performance. We investigate the impact of freezing the decoder in a multi-task setup comprising diverse natural language tasks, aiming to reduce deployment overhead and enhance portability to novel tasks. Our experiments, conducted by fine-tuning both individual and multi-task setups on the AlexaTM model, reveal that freezing decoders is highly effective for tasks with natural language outputs and mitigates catastrophic forgetting in multilingual tasks. However, we find that pairing frozen decoders with a larger model can effectively maintain or even enhance performance in structured and QA tasks, making it a viable strategy for a broader range of task types. 

**Abstract (ZH)**: 在参数效率微调方法中，冻结（Freezing）已成为加快训练、减少灾难性遗忘并提高下游性能的一种流行策略。我们探讨了在包含多种自然语言任务的多任务设置中冻结解码器的影响，旨在减少部署开销并增强对新任务的适应性。通过在AlexaTM模型上对单任务和多任务设置进行微调，我们的实验显示，冻结解码器对具有自然语言输出的任务尤为有效，并在多语言任务中减轻了灾难性遗忘的现象。然而，我们发现将冻结的解码器与更大规模的模型结合使用，可以在结构化和问答任务中有效维持甚至提高性能，从而使其成为更广泛任务类型的有效策略。 

---
# Large Language Models for Knowledge Graph Embedding Techniques, Methods, and Challenges: A Survey 

**Title (ZH)**: 大型语言模型在知识图嵌入技术、方法和挑战综述 

**Authors**: Bingchen Liu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.07766)  

**Abstract**: Large Language Models (LLMs) have attracted a lot of attention in various fields due to their superior performance, aiming to train hundreds of millions or more parameters on large amounts of text data to understand and generate natural language. As the superior performance of LLMs becomes apparent, they are increasingly being applied to knowledge graph embedding (KGE) related tasks to improve the processing results. As a deep learning model in the field of Natural Language Processing (NLP), it learns a large amount of textual data to predict the next word or generate content related to a given text. However, LLMs have recently been invoked to varying degrees in different types of KGE related scenarios such as multi-modal KGE and open KGE according to their task characteristics. In this paper, we investigate a wide range of approaches for performing LLMs-related tasks in different types of KGE scenarios. To better compare the various approaches, we summarize each KGE scenario in a classification. In addition to the categorization methods, we provide a tabular overview of the methods and their source code links for a more direct comparison. In the article we also discuss the applications in which the methods are mainly used and suggest several forward-looking directions for the development of this new research area. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其卓越的性能，在多个领域吸引了大量关注，旨在通过在大量文本数据上训练数亿甚至更多的参数来理解并生成自然语言。随着LLMs性能的优势日益明显，它们被越来越多地应用于知识图谱嵌入（KGE）相关的任务中，以改善处理结果。作为一种自然语言处理（NLP）领域的深度学习模型，LLMs学习大量文本数据以预测下一个单词或生成给定文本相关的内容。然而，根据任务特性，LLMs近年来在不同类型的KGE相关场景中被不同程度地应用于多模态KGE和开放性KGE等场景中。本文旨在探索在不同类型的KGE场景中执行LLMs相关任务的广泛方法。为了更好地比较各种方法，我们按类别总结了每种KGE场景。除了分类方法外，我们还提供了关于这些方法的表格综述及源代码链接，以提供更直接的比较。在文章中，我们还讨论了这些方法主要在哪些应用中使用，并提出了一些对未来这一新研究领域的方向性建议。 

---
# Advancing Student Writing Through Automated Syntax Feedback 

**Title (ZH)**: 通过自动句法反馈促进学生写作能力提升 

**Authors**: Kamyar Zeinalipour, Mehak Mehak, Fatemeh Parsamotamed, Marco Maggini, Marco Gori  

**Link**: [PDF](https://arxiv.org/pdf/2501.07740)  

**Abstract**: This study underscores the pivotal role of syntax feedback in augmenting the syntactic proficiency of students. Recognizing the challenges faced by learners in mastering syntactic nuances, we introduce a specialized dataset named Essay-Syntax-Instruct designed to enhance the understanding and application of English syntax among these students. Leveraging the capabilities of Large Language Models (LLMs) such as GPT3.5-Turbo, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, and Mistral-7B-Instruct-v0.2, this work embarks on a comprehensive fine-tuning process tailored to the syntax improvement task. Through meticulous evaluation, we demonstrate that the fine-tuned LLMs exhibit a marked improvement in addressing syntax-related challenges, thereby serving as a potent tool for students to identify and rectify their syntactic errors. The findings not only highlight the effectiveness of the proposed dataset in elevating the performance of LLMs for syntax enhancement but also illuminate a promising path for utilizing advanced language models to support language acquisition efforts. This research contributes to the broader field of language learning technology by showcasing the potential of LLMs in facilitating the linguistic development of Students. 

**Abstract (ZH)**: 本研究强调了语法反馈在提升学生语法能力方面所发挥的关键作用。面对学习者在掌握语法细微差别方面的挑战，我们引入了一个名为Essay-Syntax-Instruct的专门数据集，旨在增强学生对英语语法的理解和应用能力。利用诸如GPT3.5-Turbo、Llama-2-7b-chat-hf、Llama-2-13b-chat-hf和Mistral-7B-Instruct-v0.2等大型语言模型（LLMs）的能力，本研究启动了一项全面的微调过程，专门针对语法改进任务。通过细致的评估，我们证明了微调后的LLMs在处理语法相关挑战方面表现出明显的改善，从而为学生识别和纠正语法错误提供了一种有力的工具。研究结果不仅突显了所提数据集在提升LLMs语法增强性能方面的有效性，还揭示了利用先进语言模型支持语言习得的努力可能带来的前景。本研究通过展示LLMs在促进学生语言发展方面的潜力，为语言学习技术领域做出了贡献。 

---
# Exploring the encoding of linguistic representations in the Fully-Connected Layer of generative CNNs for Speech 

**Title (ZH)**: 探究生成型CNNs的全连接层中语言表示的编码方法在语音中的应用 

**Authors**: Bruno Ferenc Šegedin, Gasper Beguš  

**Link**: [PDF](https://arxiv.org/pdf/2501.07726)  

**Abstract**: Interpretability work on the convolutional layers of CNNs has primarily focused on computer vision, but some studies also explore correspondences between the latent space and the output in the audio domain. However, it has not been thoroughly examined how acoustic and linguistic information is represented in the fully connected (FC) layer that bridges the latent space and convolutional layers. The current study presents the first exploration of how the FC layer of CNNs for speech synthesis encodes linguistically relevant information. We propose two techniques for exploration of the fully connected layer. In Experiment 1, we use weight matrices as inputs into convolutional layers. In Experiment 2, we manipulate the FC layer to explore how symbolic-like representations are encoded in CNNs. We leverage the fact that the FC layer outputs a feature map and that variable-specific weight matrices are temporally structured to (1) demonstrate how the distribution of learned weights varies between latent variables in systematic ways and (2) demonstrate how manipulating the FC layer while holding constant subsequent model parameters affects the output. We ultimately present an FC manipulation that can output a single segment. Using this technique, we show that lexically specific latent codes in generative CNNs (ciwGAN) have shared lexically invariant sublexical representations in the FC-layer weights, showing that ciwGAN encodes lexical information in a linguistically principled manner. 

**Abstract (ZH)**: 卷积神经网络（CNN）的解释性工作主要集中在计算机视觉领域，但一些研究也探索了音频领域的潜在空间与输出之间的对应关系。然而，关于音频领域中声学信息和语言信息是如何在连接潜在空间和卷积层的全连接层（FC层）中表示的，研究尚未得到充分的探讨。本研究是首次探索用于语音合成的CNN的FC层如何编码与语言相关的信息。我们提出了两种探索FC层的技术。在实验1中，我们使用权重矩阵作为卷积层的输入；在实验2中，我们操控FC层，以探索CNN中符号化表示是如何编码的。我们利用FC层输出特征图且变量特定的权重矩阵在时间上结构化的事实，（1）展示了学习到的权重在潜在变量之间的系统性分布差异；（2）展示了在保持后续模型参数不变的情况下操控FC层如何影响输出。最终，我们展示了一种能够输出单一片段的FC层操控方法。使用这种方法，我们展示了生成性CNN（ciwGAN）中的词汇特定潜在代码具有共享的词汇不变的次词汇表示，在FC层权重中，表明ciwGAN以语言学合理的方式编码了词汇信息。 

---
# ESURF: Simple and Effective EDU Segmentation 

**Title (ZH)**: ESURF: 简洁有效的教育单元分割 

**Authors**: Mohammadreza Sediqin, Shlomo Engelson Argamon  

**Link**: [PDF](https://arxiv.org/pdf/2501.07723)  

**Abstract**: Segmenting text into Elemental Discourse Units (EDUs) is a fundamental task in discourse parsing. We present a new simple method for identifying EDU boundaries, and hence segmenting them, based on lexical and character n-gram features, using random forest classification. We show that the method, despite its simplicity, outperforms other methods both for segmentation and within a state of the art discourse parser. This indicates the importance of such features for identifying basic discourse elements, pointing towards potentially more training-efficient methods for discourse analysis. 

**Abstract (ZH)**: 将文本分割为基本语义单元（EDUs）是语篇解析中的一个基本任务。我们提出了一种基于词汇和字符 n-克（n-gram）特征的简单方法，用于识别 EDU 边界并进行分割，使用随机森林分类方法。研究表明，尽管该方法简单，但在分割任务中却优于其他方法，在最先进的语篇解析器中也表现更佳。这表明这类特征对于识别基本语篇元素的重要性，暗示了可能更高效的方法来提高语篇分析的训练效率。 

---
# LLMic: Romanian Foundation Language Model 

**Title (ZH)**: LLMic：罗马尼亚基础语言模型 

**Authors**: Vlad-Andrei Bădoiu, Mihai-Valentin Dumitru, Alexandru M. Gherghescu, Alexandru Agache, Costin Raiciu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07721)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks with commercial models leading the way. While open models usually operate at a smaller scale, they maintain competitiveness through specialization and fine-tuning. However, a significant challenge persists: open models often underperform in low-resource languages due to limited representation in the training corpus. In this paper, we present LLMic, a bilingual foundation language model designed specifically for the Romanian Language. We document the complete process of pretraining a foundation model for a low-resource language, including corpus construction, architecture selection, and hyper-parameter optimization. Our evaluation demonstrates that LLMic can be specialized for tasks in the target language, achieving results comparable to other much larger open models. We show that fine-tuning LLMic for language translation after the initial pretraining phase outperforms existing solutions in English-to-Romanian translation tasks. This opens the path for efficient large-scale processing for the Romanian language community, using the much smaller LLMic model 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各种任务中的表现展现了卓越的能力，商业模型在这方面处于领先地位。虽然开放模型通常规模较小，但通过专门化和微调仍能保持竞争力。然而，一个显著的挑战仍然存在：开放模型在低资源语言中往往表现不佳，因为训练语料中的代表性不足。在本文中，我们介绍了一种专门为罗曼语设计的双语基础语言模型——LLMic。我们详细记录了构建低资源语言基础模型的完整过程，包括语料库构建、架构选择和超参数优化。我们的评估表明，LLMic 可以针对目标语言进行专门化，其性能可与更大规模的开放模型相媲美。我们展示了在初始预训练阶段后，对LLMic进行语言翻译微调，可显著优于现有的英-罗翻译解决方案。这为罗曼语社区的高效大规模处理铺平了道路，使用的是更小规模的LLMic模型。 

---
# Entailed Between the Lines: Incorporating Implication into NLI 

**Title (ZH)**: 《言外之意：将暗示纳入自然语言推理》

这个翻译既保留了原标题的含义，又符合学术文章翻译的规范。原文中的“Entailed Between the Lines”直译为“行间蕴含的”，“Implication”译为“暗示”，NLI通常是指自然语言推理（Natural Language Inference），在翻译中保持了这一术语的英文缩写。 

**Authors**: Shreya Havaldar, Hamidreza Alvari, Alex Fabrikant, John Palowitch, Mohammad Javad Hosseini, Senaka Buthpitiya  

**Link**: [PDF](https://arxiv.org/pdf/2501.07719)  

**Abstract**: Much of human communication depends on implication, conveying meaning beyond literal words to express a wider range of thoughts, intentions, and feelings. For models to better understand and facilitate human communication, they must be responsive to the text's implicit meaning. We focus on Natural Language Inference (NLI), a core tool for many language tasks, and find that state-of-the-art NLI models and datasets struggle to recognize a range of cases where entailment is implied, rather than explicit from the text. We formalize implied entailment as an extension of the NLI task and introduce the Implied NLI dataset (INLI) to help today's LLMs both recognize a broader variety of implied entailments and to distinguish between implicit and explicit entailment. We show how LLMs fine-tuned on INLI understand implied entailment and can generalize this understanding across datasets and domains. 

**Abstract (ZH)**: 人类沟通很大程度上依赖于隐含意义，通过超出字面意义的表达来传达更广泛的思想、意图和情感。为了使模型更好地理解并促进人类沟通，它们必须对文本中的隐含意义作出响应。我们关注自然语言推理（NLI），这是一种应用于许多语言任务的核心工具，并发现最先进的NLI模型和数据集在识别一些隐含蕴含（而非直接显式）的情况时存在困难。我们将隐含蕴含形式化为NLI任务的拓展，并引入了隐含NLI数据集（INLI），以帮助今天的大型语言模型（LLM）识别更广泛种类的隐含蕴含，并区分隐含和显式的蕴含。我们展示了在INLI上Fine-tune的LLM如何理解隐含蕴含，并能在不同数据集和领域中泛化这种理解。 

---
# Benchmarking Abstractive Summarisation: A Dataset of Human-authored Summaries of Norwegian News Articles 

**Title (ZH)**: 基于规范学术表达，以下是对该标题的翻译：

挪威新闻文章的人工摘要数据集上的抽象总结基准测试 

**Authors**: Samia Touileb, Vladislav Mikhailov, Marie Kroka, Lilja Øvrelid, Erik Velldal  

**Link**: [PDF](https://arxiv.org/pdf/2501.07718)  

**Abstract**: We introduce a dataset of high-quality human-authored summaries of news articles in Norwegian. The dataset is intended for benchmarking the abstractive summarisation capabilities of generative language models. Each document in the dataset is provided with three different candidate gold-standard summaries written by native Norwegian speakers, and all summaries are provided in both of the written variants of Norwegian -- Bokmål and Nynorsk. The paper describes details on the data creation effort as well as an evaluation of existing open LLMs for Norwegian on the dataset. We also provide insights from a manual human evaluation, comparing human-authored to model-generated summaries. Our results indicate that the dataset provides a challenging LLM benchmark for Norwegian summarisation capabilities 

**Abstract (ZH)**: 我们介绍了挪威语高质量人工撰写的新聞文章摘要数据集。该数据集旨在评估生成型语言模型的抽象总结能力。数据集中每个文件都提供了由挪威母语者撰写的三种不同候选标准摘要，并且所有的摘要都提供了书面形式的两种变体——标准挪威语（Bokmål）和新挪威语（Nynorsk）版本。本文详细描述了数据集创建的努力过程，并对数据集上现有的开放型大型语言模型进行了评估。我们还提供了一项人工评估的见解，比较了人工撰写的摘要与模型生成的摘要。我们的结果表明，该数据集为挪威语总结能力提供了具有挑战性的大型语言模型基准。 

---
# Enhancing Talent Employment Insights Through Feature Extraction with LLM Finetuning 

**Title (ZH)**: 通过LLM微调进行特征提取以增强人才招聘洞察力 

**Authors**: Karishma Thakrar, Nick Young  

**Link**: [PDF](https://arxiv.org/pdf/2501.07663)  

**Abstract**: This paper explores the application of large language models (LLMs) to extract nuanced and complex job features from unstructured job postings. Using a dataset of 1.2 million job postings provided by AdeptID, we developed a robust pipeline to identify and classify variables such as remote work availability, remuneration structures, educational requirements, and work experience preferences. Our methodology combines semantic chunking, retrieval-augmented generation (RAG), and fine-tuning DistilBERT models to overcome the limitations of traditional parsing tools. By leveraging these techniques, we achieved significant improvements in identifying variables often mislabeled or overlooked, such as non-salary-based compensation and inferred remote work categories. We present a comprehensive evaluation of our fine-tuned models and analyze their strengths, limitations, and potential for scaling. This work highlights the promise of LLMs in labor market analytics, providing a foundation for more accurate and actionable insights into job data. 

**Abstract (ZH)**: 本文探索了大规模语言模型（LLMs）在从无结构的招聘信息中提取复杂和细微的职业特征方面的应用。使用AdeptID提供的120万条招聘信息数据集，我们开发了一个 robust 管道来识别和分类诸如远程工作机会、薪酬结构、教育要求和工作经验偏好等变量。我们的方法结合了语义切块、检索增强生成（RAG）和 DistilBERT 模型微调技术，以克服传统解析工具的限制。通过利用这些技术，我们显著提高了识别那些经常被误标或忽略的变量（如非薪金薪酬和推断的远程工作类别）的能力。我们对微调模型进行了全面评估，并分析了它们的优势、局限性和扩展潜力。本研究突显了LLMs在劳动力市场分析中的潜力，为更准确和实用的工作数据洞察提供了基础。 

---
# GPT as a Monte Carlo Language Tree: A Probabilistic Perspective 

**Title (ZH)**: 将下面的论文内容或标题翻译成中文，可以翻译为：

GPT作为一种蒙特卡洛语言树：一种概率视角

这样翻译既保留了原文的学术性，又符合中文的表达习惯。 

**Authors**: Kun-Peng Ning, Jia-Yu Yao, Yu-Yang Liu, Mu-Nan Ning, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.07641)  

**Abstract**: Large Language Models (LLMs), such as GPT, are considered to learn the latent distributions within large-scale web-crawl datasets and accomplish natural language processing (NLP) tasks by predicting the next token. However, this mechanism of latent distribution modeling lacks quantitative understanding and analysis. In this paper, we propose a novel perspective that any language dataset can be represented by a Monte Carlo Language Tree (abbreviated as ``Data-Tree''), where each node denotes a token, each edge denotes a token transition probability, and each sequence has a unique path. Any GPT-like language model can also be flattened into another Monte Carlo Language Tree (abbreviated as ``GPT-Tree''). Our experiments show that different GPT models trained on the same dataset exhibit significant structural similarity in GPT-Tree visualization, and larger models converge more closely to the Data-Tree. More than 87\% GPT output tokens can be recalled by Data-Tree. These findings may confirm that the reasoning process of LLMs is more likely to be probabilistic pattern-matching rather than formal reasoning, as each model inference seems to find a context pattern with maximum probability from the Data-Tree. Furthermore, we provide deeper insights into issues such as hallucination, Chain-of-Thought (CoT) reasoning, and token bias in LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs），如GPT，被认为是通过预测下一个词来学习大规模网络爬取数据集中的潜在分布，从而完成自然语言处理（NLP）任务。然而，这种潜在分布建模机制缺乏定量的理解和分析。在本文中，我们提出了一个新颖的观点，即任何语言数据集都可以表示为一个蒙特卡洛语言树（简称“Data-Tree”），其中每个节点表示一个词，每条边表示一个词转接概率，每个序列有一个唯一的路径。任何类似于GPT的语言模型也可以展平为另一个蒙特卡洛语言树（简称“GPT-Tree”）。我们的实验表明，相同数据集训练的不同GPT模型在GPT-Tree可视化方面显示出显著的结构相似性，较大的模型更接近于Data-Tree。超过87%的GPT生成的词可以被Data-Tree召回。这些发现可能表明，LLMs的推理过程更可能是概率模式匹配而不是形式推理，因为每个模型的推理似乎是从Data-Tree中找到具有最大概率的上下文模式。此外，我们还深入探讨了LLMs中存在的幻觉、思维链（CoT）推理和词偏见等问题。 

---
# CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation 

**Title (ZH)**: CWEval：功能性和安全性导向的大型语言模型代码生成评估 

**Authors**: Jinjun Peng, Leyi Cui, Kele Huang, Junfeng Yang, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2501.08200)  

**Abstract**: Large Language Models (LLMs) have significantly aided developers by generating or assisting in code writing, enhancing productivity across various tasks. While identifying incorrect code is often straightforward, detecting vulnerabilities in functionally correct code is more challenging, especially for developers with limited security knowledge, which poses considerable security risks of using LLM-generated code and underscores the need for robust evaluation benchmarks that assess both functional correctness and security. Current benchmarks like CyberSecEval and SecurityEval attempt to solve it but are hindered by unclear and impractical specifications, failing to assess both functionality and security accurately. To tackle these deficiencies, we introduce CWEval, a novel outcome-driven evaluation framework designed to enhance the evaluation of secure code generation by LLMs. This framework not only assesses code functionality but also its security simultaneously with high-quality task specifications and outcome-driven test oracles which provides high accuracy. Coupled with CWEval-bench, a multilingual, security-critical coding benchmark, CWEval provides a rigorous empirical security evaluation on LLM-generated code, overcoming previous benchmarks' shortcomings. Through our evaluations, CWEval reveals a notable portion of functional but insecure code produced by LLMs, and shows a serious inaccuracy of previous evaluations, ultimately contributing significantly to the field of secure code generation. We open-source our artifact at: this https URL . 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成或辅助代码撰写方面显著地帮助了开发者，提升了各种任务的生产力。尽管识别错误的代码往往容易，但对于功能正确的代码检测漏洞则更加困难，特别是对于那些缺乏安全知识的开发者而言，这给使用LLM生成的代码带来了较大的安全风险，强调了需要既评估功能正确性又评估安全性的稳健评估基准的重要性。现有的基准如CyberSecEval和SecurityEval试图解决这一问题，但它们受到不明确且不实用的规范限制，未能准确评估功能和安全性。为了应对这些不足，我们引入了CWEval，这是一种新的结果驱动的评估框架，旨在提高对LLM生成代码的评估能力。该框架不仅评估代码的功能性，还同时评估其安全性，具有高质量的任务规范和结果驱动的测试或acles，提供了高精度的评估。结合CWEval-bench，这是一个多语言、关键安全编码基准，CWEval能够对LLM生成的代码进行全面严谨的安全性评估，克服了之前基准的不足。通过我们的评估，CWEval揭示了由LLM生成的大量功能性但不安全的代码，并显示了之前评估的严重不准确，最终对安全代码生成领域做出了重大贡献。我们开源了我们的实现：[该网址]。 

---
# In-situ graph reasoning and knowledge expansion using Graph-PReFLexOR 

**Title (ZH)**: 基于图的原位推理与知识扩展：Graph-PReFLexOR方法 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2501.08120)  

**Abstract**: The pursuit of automated scientific discovery has fueled progress from symbolic logic to modern AI, forging new frontiers in reasoning and pattern recognition. Transformers function as potential systems, where every possible relationship remains latent potentiality until tasks impose constraints, akin to measurement. Yet, refining their sampling requires more than probabilistic selection: solutions must conform to specific structures or rules, ensuring consistency and the invocation of general principles. We present Graph-PReFLexOR (Graph-based Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning), a framework that combines graph reasoning with symbolic abstraction to dynamically expand domain knowledge. Inspired by reinforcement learning, Graph-PReFLexOR defines reasoning as a structured mapping, where tasks yield knowledge graphs, abstract patterns, and ultimately, final answers. Inspired by category theory, it encodes concepts as nodes and their relationships as edges, supporting hierarchical inference and adaptive learning through isomorphic representations. Demonstrations include hypothesis generation, materials design, and creative reasoning, such as discovering relationships between mythological concepts like 'thin places' with materials science. We propose a 'knowledge garden growth' strategy that integrates insights across domains, promoting interdisciplinary connections. Results with a 3-billion-parameter Graph-PReFLexOR model show superior reasoning depth and adaptability, underscoring the potential for transparent, multidisciplinary AI-driven discovery. It lays the groundwork for general autonomous reasoning solutions. 

**Abstract (ZH)**: 自动化的科学发现追求已经推动了从符号逻辑到现代人工智能的进步，开辟了推理和模式识别的新前沿。变换器作为潜在系统，所有可能的关系都处于潜在状态，直到任务施加约束，类似于测量过程。然而，改进它们的采样需要超越概率选择：解决方案必须符合特定的结构或规则，确保一致性和通用原则的激活。我们提出了Graph-PReFLexOR（基于图形的偏好递归语言建模，用于推理的探索优化），该框架结合了图形推理和符号抽象，以动态扩展领域知识。受强化学习的启发，Graph-PReFLexOR 定义推理为结构化的映射，任务生成知识图、抽象模式，最终得出最终答案。受范畴理论的启发，它将概念编码为节点，关系编码为边，通过同构表示支持层次推断和自适应学习。示例包括假设生成、材料设计和创造性的推理，例如发现神话概念（如“纤薄之地”）与材料科学之间的关系。我们提出了一种“知识花园增长”策略，将不同领域的见解整合在一起，促进学科间的联系。使用包含30亿参数的Graph-PReFLexOR模型的试验结果表明，其推理深度和适应性更胜一筹，突显了透明的跨学科人工智能驱动发现的潜力。它为通用自主推理解决方案奠定了基础。 

---
# Optimizing Speech Multi-View Feature Fusion through Conditional Computation 

**Title (ZH)**: 通过条件计算优化语音多视图特征融合 

**Authors**: Weiqiao Shan, Yuhao Zhang, Yuchen Han, Bei Li, Xiaofeng Zhao, Yuang Li, Min Zhang, Hao Yang, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08057)  

**Abstract**: Recent advancements have highlighted the efficacy of self-supervised learning (SSL) features in various speech-related tasks, providing lightweight and versatile multi-view speech representations. However, our study reveals that while SSL features expedite model convergence, they conflict with traditional spectral features like FBanks in terms of update directions. In response, we propose a novel generalized feature fusion framework grounded in conditional computation, featuring a gradient-sensitive gating network and a multi-stage dropout strategy. This framework mitigates feature conflicts and bolsters model robustness to multi-view input features. By integrating SSL and spectral features, our approach accelerates convergence and maintains performance on par with spectral models across multiple speech translation tasks on the MUSTC dataset. 

**Abstract (ZH)**: 近期的研究表明，自监督学习（SSL）特征在各种语音相关任务中具有显著的效果，能够提供轻量级且多视角的语音表示。然而，我们的研究发现，尽管SSL特征可以加速模型的收敛，但它们在更新方向上与传统的谱特征（如FBanks）存在冲突。为解决这一问题，我们提出了一种基于条件计算的新型特征融合框架，该框架包含梯度敏感门控网络和多阶段 Dropout 策略。该框架能够缓解特征之间的冲突，并提升模型对多视角输入特征的鲁棒性。通过融合SSL特征和谱特征，我们的方法在MUSTC数据集上的多个语音翻译任务中加速了收敛，并且在性能上能够与谱特征模型保持一致。 

---
# Gandalf the Red: Adaptive Security for LLMs 

**Title (ZH)**: 标题：Gandalf the Red：面向大语言模型的自适应安全策略 

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Natalie Wu, Mateo Rojas-Carulla  

**Link**: [PDF](https://arxiv.org/pdf/2501.07927)  

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{this https URL}{\texttt{this https URL}}. 

**Abstract (ZH)**: 当前对大语言模型（LLM）应用中对抗性提示攻击防御措施的评估往往忽视了两个关键因素：对抗行为的动态性质以及限制性防御措施对合法用户造成的影响。我们提出了 D-SEC（动态安全实用性威胁模型），该模型明确区分攻击者和合法用户，建模多步交互，并以优化可表达的形式严格描述安全实用性。为了弥补现有评估中的不足，我们引入了 Gandalf，这是一个基于众包的、游戏化的红队平台，旨在生成现实且适应性强的攻击数据集。通过使用 Gandalf，我们收集并发布了一个包含 279,000 条提示攻击的数据集。结合良性用户数据，我们分析了安全性和实用性之间的相互作用，展示了即使不阻止请求，集成在 LLM 中的防御措施（例如系统提示）也会降低用户体验。我们证明了限制应用领域、多层次防御和自适应防御是构建安全且有用的 LLM 应用的有效策略。相关代码可在 \href{this https URL}{this https URL} 获取。 

---
# Exploring Aviation Incident Narratives Using Topic Modeling and Clustering Techniques 

**Title (ZH)**: 使用主题建模和聚类技术探索航空事故叙述 

**Authors**: Aziida Nanyonga, Hassan Wasswa, Ugur Turhan, Keith Joiner, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2501.07924)  

**Abstract**: Aviation safety is a global concern, requiring detailed investigations into incidents to understand contributing factors comprehensively. This study uses the National Transportation Safety Board (NTSB) dataset. It applies advanced natural language processing (NLP) techniques, including Latent Dirichlet Allocation (LDA), Non-Negative Matrix Factorization (NMF), Latent Semantic Analysis (LSA), Probabilistic Latent Semantic Analysis (pLSA), and K-means clustering. The main objectives are identifying latent themes, exploring semantic relationships, assessing probabilistic connections, and cluster incidents based on shared characteristics. This research contributes to aviation safety by providing insights into incident narratives and demonstrating the versatility of NLP and topic modelling techniques in extracting valuable information from complex datasets. The results, including topics identified from various techniques, provide an understanding of recurring themes. Comparative analysis reveals that LDA performed best with a coherence value of 0.597, pLSA of 0.583, LSA of 0.542, and NMF of 0.437. K-means clustering further reveals commonalities and unique insights into incident narratives. In conclusion, this study uncovers latent patterns and thematic structures within incident narratives, offering a comparative analysis of multiple-topic modelling techniques. Future research avenues include exploring temporal patterns, incorporating additional datasets, and developing predictive models for early identification of safety issues. This research lays the groundwork for enhancing the understanding and improvement of aviation safety by utilising the wealth of information embedded in incident narratives. 

**Abstract (ZH)**: 航空安全是全球关注的问题，需要对事故进行全面详细的调查，以全面理解其促成因素。本研究利用美国国家运输安全委员会（NTSB）的数据集，应用先进的自然语言处理（NLP）技术，包括潜在狄利克雷分配（LDA）、非负矩阵分解（NMF）、潜在语义分析（LSA）、概率潜在语义分析（pLSA）和K均值聚类。主要目标是识别潜在主题、探索语义关系、评估概率联系，并基于共享特征对事故进行聚类。本研究通过提供事故叙述的见解，展示了NLP和主题建模技术在提取复杂数据集中的有价值信息方面的多用途性，从而为航空安全做出了贡献。结果显示，从各种技术中识别的主题包括了重复的模式。比较分析表明，LDA表现最佳，其连贯性值为0.597，pLSA为0.583，LSA为0.542，NMF为0.437。K均值聚类进一步揭示了事故叙述中的共同性和独特的见解。综上所述，本研究揭示了事故叙述中的潜在模式和主题结构，并对多种主题建模技术进行了比较分析。未来的研究方向包括探索时间模式、整合额外的数据集以及开发早期识别安全问题的预测模型。这项研究为利用事故叙述中蕴含的丰富信息增进对航空安全的理解和改进奠定了基础。 

---
# Aviation Safety Enhancement via NLP & Deep Learning: Classifying Flight Phases in ATSB Safety Reports 

**Title (ZH)**: 通过自然语言处理与深度学习提升航空安全：澳大利亚交通安全局安全报告中的飞行阶段分类 

**Authors**: Aziida Nanyonga, Hassan Wasswa, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2501.07923)  

**Abstract**: Aviation safety is paramount, demanding precise analysis of safety occurrences during different flight phases. This study employs Natural Language Processing (NLP) and Deep Learning models, including LSTM, CNN, Bidirectional LSTM (BLSTM), and simple Recurrent Neural Networks (sRNN), to classify flight phases in safety reports from the Australian Transport Safety Bureau (ATSB). The models exhibited high accuracy, precision, recall, and F1 scores, with LSTM achieving the highest performance of 87%, 88%, 87%, and 88%, respectively. This performance highlights their effectiveness in automating safety occurrence analysis. The integration of NLP and Deep Learning technologies promises transformative enhancements in aviation safety analysis, enabling targeted safety measures and streamlined report handling. 

**Abstract (ZH)**: 航空安全至关重要，要求对不同飞行阶段的安全事件进行精确分析。本研究采用了自然语言处理（NLP）和深度学习模型，包括长短期记忆网络（LSTM）、卷积神经网络（CNN）、双向长短期记忆网络（BiLSTM）以及简单的循环神经网络（sRNN），对澳大利亚运输安全局（ATSB）的安全报告中的飞行阶段进行分类。这些模型展现了高度的准确率、精确率、召回率和F1分数，其中LSTM分别达到了87%、88%、87%和88%的最佳性能。这一性能突显了这些模型在自动化安全事件分析方面的有效性。NLP和深度学习技术的集成有望在航空安全分析中带来变革性的改进，从而实现针对性的安全措施并简化报告处理流程。 

---
# Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision 

**Title (ZH)**: 在弱监督条件下，迭代标签 refinement 比偏好优化更为重要 

**Authors**: Yaowen Ye, Cassidy Laidlaw, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2501.07886)  

**Abstract**: Language model (LM) post-training relies on two stages of human supervision: task demonstrations for supervised finetuning (SFT), followed by preference comparisons for reinforcement learning from human feedback (RLHF). As LMs become more capable, the tasks they are given become harder to supervise. Will post-training remain effective under unreliable supervision? To test this, we simulate unreliable demonstrations and comparison feedback using small LMs and time-constrained humans. We find that in the presence of unreliable supervision, SFT still retains some effectiveness, but DPO (a common RLHF algorithm) fails to improve the model beyond SFT. To address this, we propose iterative label refinement (ILR) as an alternative to RLHF. ILR improves the SFT data by using comparison feedback to decide whether human demonstrations should be replaced by model-generated alternatives, then retrains the model via SFT on the updated data. SFT+ILR outperforms SFT+DPO on several tasks with unreliable supervision (math, coding, and safe instruction-following). Our findings suggest that as LMs are used for complex tasks where human supervision is unreliable, RLHF may no longer be the best use of human comparison feedback; instead, it is better to direct feedback towards improving the training data rather than continually training the model. Our code and data are available at this https URL. 

**Abstract (ZH)**: 经过训练的语言模型（LM）依赖于两个阶段的人类监督：监督微调（SFT）的任务演示，随后是基于人类反馈的强化学习（RLHF）的偏好比较。随着语言模型能力的提高，它们被赋予的任务变得更加难以进行监督。在不可靠的监督下，后训练方法是否仍然有效？为了检验这一点，我们使用小型语言模型和时间限制下的人类来模拟不可靠的演示和比较反馈。我们发现，在不可靠监督的环境下，SFT仍然保留了一定的有效性，但常见的RLHF算法DPO无法使模型超越SFT。为解决这一问题，我们提出迭代标签精炼（ILR）作为RLHF的替代方法。ILR通过使用比较反馈来决定是否应该用模型生成的替代示例取代人类演示，然后通过对更新后的数据进行SFT重新训练，从而提高SFT数据的质量。在不可靠监督的多个任务（如数学、编程和安全指令执行）中，SFT+ILR的表现优于SFT+DPO。我们的研究结果表明，当语言模型被用于需要人类监督不可靠的复杂任务时，RLHF可能不再是最有效的利用人类比较反馈的方法；相反，更应该将反馈集中在提高训练数据的质量上，而不是不断地训练模型。我们的代码和数据可在以下网址获取：this https URL。 

---
# Social Media Data Mining With Natural Language Processing on Public Dream Contents 

**Title (ZH)**: 使用自然语言处理技术挖掘社交媒体上的公共资源梦境内容 

**Authors**: Howard Hua, Joe Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07839)  

**Abstract**: The COVID-19 pandemic has significantly transformed global lifestyles, enforcing physical isolation and accelerating digital adoption for work, education, and social interaction. This study examines the pandemic's impact on mental health by analyzing dream content shared on the Reddit r/Dreams community. With over 374,000 subscribers, this platform offers a rich dataset for exploring subconscious responses to the pandemic. Using statistical methods, we assess shifts in dream positivity, negativity, and neutrality from the pre-pandemic to post-pandemic era. To enhance our analysis, we fine-tuned the LLaMA 3.1-8B model with labeled data, enabling precise sentiment classification of dream content. Our findings aim to uncover patterns in dream content, providing insights into the psychological effects of the pandemic and its influence on subconscious processes. This research highlights the profound changes in mental landscapes and the role of dreams as indicators of public well-being during unprecedented times. 

**Abstract (ZH)**: 新冠疫情期间，全球生活方式经历了显著的变革，人们被强制实行物理隔离，加速了工作、教育和社交互动的数字化进程。本研究通过分析在Reddit r/Dreams论坛上共享的梦境内容，探讨疫情对心理健康的影响。该平台拥有超过37.4万名订阅者，提供了丰富的大数据集，用于探索疫情期间潜意识的反应。通过统计方法，我们评估了从疫情前到疫情后的梦境中正面、负面和中性情感的变化趋势。为增强分析效果，我们使用标记数据对LLaMA 3.1-8B模型进行了微调，使其能够精确分类梦境内容的情感。我们的研究旨在揭示梦境内容的模式，提供关于疫情心理影响及其对潜意识过程影响的见解。本研究突显了疫情期间心理景观的深刻变化，以及梦境作为公共福祉指标的作用。 

---
# Agent-Centric Projection of Prompting Techniques and Implications for Synthetic Training Data for Large Language Models 

**Title (ZH)**: 基于代理的提示技术投影及对大规模语言模型合成训练数据的影响 

**Authors**: Dhruv Dhamani, Mary Lou Maher  

**Link**: [PDF](https://arxiv.org/pdf/2501.07815)  

**Abstract**: Recent advances in prompting techniques and multi-agent systems for Large Language Models (LLMs) have produced increasingly complex approaches. However, we lack a framework for characterizing and comparing prompting techniques or understanding their relationship to multi-agent LLM systems. This position paper introduces and explains the concepts of linear contexts (a single, continuous sequence of interactions) and non-linear contexts (branching or multi-path) in LLM systems. These concepts enable the development of an agent-centric projection of prompting techniques, a framework that can reveal deep connections between prompting strategies and multi-agent systems. We propose three conjectures based on this framework: (1) results from non-linear prompting techniques can predict outcomes in equivalent multi-agent systems, (2) multi-agent system architectures can be replicated through single-LLM prompting techniques that simulate equivalent interaction patterns, and (3) these equivalences suggest novel approaches for generating synthetic training data. We argue that this perspective enables systematic cross-pollination of research findings between prompting and multi-agent domains, while providing new directions for improving both the design and training of future LLM systems. 

**Abstract (ZH)**: 近年来，针对大型语言模型（LLM）的提示技术与多agent系统方面取得了显著进展，产生了日益复杂的方法。然而，我们缺乏一种框架来表征和比较提示技术，或理解它们与多agent LLM系统之间的关系。本文观点引入并解释了线性上下文（单一、连续的互动序列）和非线性上下文（分支或多路径）的概念在LLM系统中的意义。这些概念使得能够从agent为中心的角度构建提示技术的投影框架，该框架可以揭示提示策略与多agent系统之间的深层次联系。基于这一框架，我们提出了三条假设：（1）非线性提示技术的结果可以预测等效多agent系统的成果；（2）多agent系统架构可以通过模拟等效互动模式的单一LLM提示技术进行复制；（3）这些等效性提出了生成合成训练数据的新型方法。我们认为，这一视角促进了提示技术与多agent领域研究发现的系统性交叉影响，同时还为改进未来LLM系统的设计和训练提供了新的方向。 

---
# Talk to Right Specialists: Routing and Planning in Multi-agent System for Question Answering 

**Title (ZH)**: 与合适的专家对话：多 Agent 系统中的路由与规划在问答中的应用 

**Authors**: Feijie Wu, Zitao Li, Fei Wei, Yaliang Li, Bolin Ding, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07813)  

**Abstract**: Leveraging large language models (LLMs), an agent can utilize retrieval-augmented generation (RAG) techniques to integrate external knowledge and increase the reliability of its responses. Current RAG-based agents integrate single, domain-specific knowledge sources, limiting their ability and leading to hallucinated or inaccurate responses when addressing cross-domain queries. Integrating multiple knowledge bases into a unified RAG-based agent raises significant challenges, including increased retrieval overhead and data sovereignty when sensitive data is involved. In this work, we propose RopMura, a novel multi-agent system that addresses these limitations by incorporating highly efficient routing and planning mechanisms. RopMura features two key components: a router that intelligently selects the most relevant agents based on knowledge boundaries and a planner that decomposes complex multi-hop queries into manageable steps, allowing for coordinating cross-domain responses. Experimental results demonstrate that RopMura effectively handles both single-hop and multi-hop queries, with the routing mechanism enabling precise answers for single-hop queries and the combined routing and planning mechanisms achieving accurate, multi-step resolutions for complex queries. 

**Abstract (ZH)**: 利用大规模语言模型（LLMs），代理可以通过检索增强生成（RAG）技术整合外部知识，从而提高其响应的可靠性。当前基于RAG的代理通常集成单一的领域特异性知识源，这限制了它们的能力，并且在处理跨域查询时可能导致产生不实或不准确的响应。将多个知识库整合到统一的基于RAG的代理中带来了显著的挑战，包括检索开销的增加以及涉及敏感数据时的数据主权问题。在这项工作中，我们提出了RopMura，这是一种新颖的多Agent系统，通过引入高效的路由和规划机制来解决这些限制。RopMura 包含两个关键组件：一个路由器，可以智能地根据知识边界选择最相关的代理，以及一个规划器，可以将复杂的多跳查询分解为可管理的步骤，从而协调跨域响应。实验结果表明，RopMura 有效地处理了一跳和多跳查询，其中路由机制能够为一跳查询提供精确的回答，而结合的路由和规划机制则能够为复杂的查询实现准确的多步骤解决方案。 

---
# Parameter-Inverted Image Pyramid Networks for Visual Perception and Multimodal Understanding 

**Title (ZH)**: 参数倒置图像金字塔网络在视觉感知与多模态理解中的应用 

**Authors**: Zhaokai Wang, Xizhou Zhu, Xue Yang, Gen Luo, Hao Li, Changyao Tian, Wenhan Dou, Junqi Ge, Lewei Lu, Yu Qiao, Jifeng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2501.07783)  

**Abstract**: Image pyramids are widely adopted in top-performing methods to obtain multi-scale features for precise visual perception and understanding. However, current image pyramids use the same large-scale model to process multiple resolutions of images, leading to significant computational cost. To address this challenge, we propose a novel network architecture, called Parameter-Inverted Image Pyramid Networks (PIIP). Specifically, PIIP uses pretrained models (ViTs or CNNs) as branches to process multi-scale images, where images of higher resolutions are processed by smaller network branches to balance computational cost and performance. To integrate information from different spatial scales, we further propose a novel cross-branch feature interaction mechanism. To validate PIIP, we apply it to various perception models and a representative multimodal large language model called LLaVA, and conduct extensive experiments on various tasks such as object detection, segmentation, image classification and multimodal understanding. PIIP achieves superior performance compared to single-branch and existing multi-resolution approaches with lower computational cost. When applied to InternViT-6B, a large-scale vision foundation model, PIIP can improve its performance by 1%-2% on detection and segmentation with only 40%-60% of the original computation, finally achieving 60.0 box AP on MS COCO and 59.7 mIoU on ADE20K. For multimodal understanding, our PIIP-LLaVA achieves 73.0% accuracy on TextVQA and 74.5% on MMBench with only 2.8M training data. Our code is released at this https URL. 

**Abstract (ZH)**: 图像金字塔在许多高性能方法中被广泛应用，用于获取多尺度特征以实现精确的视觉感知和理解。然而，目前的图像金字塔使用同一大型模型处理不同分辨率的图像，导致计算成本显著增加。为解决这一挑战，我们提出了一种新的网络架构，称为参数倒置图像金字塔网络（PIIP）。具体而言，PIIP 利用预训练模型（ViTs 或 CNNs）作为分支来处理多尺度图像，其中高分辨率图像由较小的网络分支处理，以平衡计算成本和性能。为了整合来自不同空间尺度的信息，我们进一步提出了一种新的跨分支特征交互机制。为了验证 PIIP 的有效性，我们将其应用于各种感知模型以及代表性的多模态大型语言模型 LLaVA，并在对象检测、分割、图像分类和多模态理解等不同任务上进行了广泛的实验。与单分支和现有多种分辨率方法相比，PIIP 在较低计算成本的基础上实现了更优的性能。当应用于大规模视觉基础模型 InternViT-6B 时，PIIP 在检测和分割任务上的性能可提高 1%-2%，计算量仅为原始计算量的 40%-60%，最终在 MS COCO 上达到 60.0 的框 AP，在 ADE20K 上达到 59.7 的 mIoU。在多模态理解方面，我们的 PIIP-LLaVA 在 TextVQA 上实现了 73.0% 的准确率，在 MMBench 上实现了 74.5% 的准确率，仅使用了 2.8 百万个训练数据。我们的代码已在此处发布：[this https URL]。 

---
# A Heterogeneous Multimodal Graph Learning Framework for Recognizing User Emotions in Social Networks 

**Title (ZH)**: 一种异构多模态图学习框架，用于识别社交网络中用户情感 

**Authors**: Sree Bhattacharyya, Shuhua Yang, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07746)  

**Abstract**: The rapid expansion of social media platforms has provided unprecedented access to massive amounts of multimodal user-generated content. Comprehending user emotions can provide valuable insights for improving communication and understanding of human behaviors. Despite significant advancements in Affective Computing, the diverse factors influencing user emotions in social networks remain relatively understudied. Moreover, there is a notable lack of deep learning-based methods for predicting user emotions in social networks, which could be addressed by leveraging the extensive multimodal data available. This work presents a novel formulation of personalized emotion prediction in social networks based on heterogeneous graph learning. Building upon this formulation, we design HMG-Emo, a Heterogeneous Multimodal Graph Learning Framework that utilizes deep learning-based features for user emotion recognition. Additionally, we include a dynamic context fusion module in HMG-Emo that is capable of adaptively integrating the different modalities in social media data. Through extensive experiments, we demonstrate the effectiveness of HMG-Emo and verify the superiority of adopting a graph neural network-based approach, which outperforms existing baselines that use rich hand-crafted features. To the best of our knowledge, HMG-Emo is the first multimodal and deep-learning-based approach to predict personalized emotions within online social networks. Our work highlights the significance of exploiting advanced deep learning techniques for less-explored problems in Affective Computing. 

**Abstract (ZH)**: 社交媒体平台的迅速扩展为获取大量多模态用户生成内容提供了前所未有的机会。理解用户情绪可以为改善人际沟通和理解人类行为提供宝贵的见解。尽管在情感计算方面取得了显著进展，但在社交网络中影响用户情绪的多种因素仍相对未被充分研究。此外，对于基于深度学习预测社交网络中用户情绪的方法也存在明显不足，这可以通过利用广泛可用的多模态数据来解决。本文提出了一种基于异构图学习的个性化情绪预测新框架。在此基础上，我们设计了HMG-Emo异构多模态图学习框架，该框架利用深度学习特征进行用户情绪识别。此外，HMG-Emo中还包括一个动态上下文融合模块，该模块能够适应性地整合社交媒体数据中的不同模态。通过广泛实验，我们证明了HMG-Emo的有效性，并验证了基于图神经网络的方法优于使用丰富手工设计特征的现有基准方法。据我们所知，HMG-Emo是第一个基于多模态和深度学习的方法，用于预测在线社交网络中的个性化情绪。我们的研究突显了利用先进深度学习技术解决情感计算中未充分探索问题的重要性。 

---
# A Survey of Early Exit Deep Neural Networks in NLP 

**Title (ZH)**: 早期退出深度神经网络在自然语言处理中的研究综述 

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal  

**Link**: [PDF](https://arxiv.org/pdf/2501.07670)  

**Abstract**: Deep Neural Networks (DNNs) have grown increasingly large in size to achieve state of the art performance across a wide range of tasks. However, their high computational requirements make them less suitable for resource-constrained applications. Also, real-world datasets often consist of a mixture of easy and complex samples, necessitating adaptive inference mechanisms that account for sample difficulty. Early exit strategies offer a promising solution by enabling adaptive inference, where simpler samples are classified using the initial layers of the DNN, thereby accelerating the overall inference process. By attaching classifiers at different layers, early exit methods not only reduce inference latency but also improve the model robustness against adversarial attacks. This paper presents a comprehensive survey of early exit methods and their applications in NLP. 

**Abstract (ZH)**: 深度神经网络（DNNs）在各种任务中已变得越来越大，以实现最先进的性能。然而，它们的高计算需求使得它们不适合资源受限的应用。此外，现实世界的数据集常常包含简单和复杂样本的混合，需要能够适应样本难度的推理机制。提前退出策略提供了一种有希望的解决方案，通过在DNN的初始层对简单样本进行分类，从而加速整个推理过程。通过在不同层附加分类器，提前退出方法不仅减少了推理延迟，还有助于提高模型对对抗攻击的鲁棒性。本文全面综述了提前退出方法及其在自然语言处理（NLP）中的应用。 

---
# Optimize Incompatible Parameters through Compatibility-aware Knowledge Integration 

**Title (ZH)**: 通过兼容性aware知识集成优化不兼容参数 

**Authors**: Zheqi Lv, Keming Ye, Zishu Wei, Qi Tian, Shengyu Zhang, Wenqiao Zhang, Wenjie Wang, Kun Kuang, Tat-Seng Chua, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07596)  

**Abstract**: Deep neural networks have become foundational to advancements in multiple domains, including recommendation systems, natural language processing, and so on. Despite their successes, these models often contain incompatible parameters that can be underutilized or detrimental to model performance, particularly when faced with specific, varying data distributions. Existing research excels in removing such parameters or merging the outputs of multiple different pretrained models. However, the former focuses on efficiency rather than performance, while the latter requires several times more computing and storage resources to support inference. In this paper, we set the goal to explicitly improve these incompatible parameters by leveraging the complementary strengths of different models, thereby directly enhancing the models without any additional parameters. Specifically, we propose Compatibility-aware Knowledge Integration (CKI), which consists of Parameter Compatibility Assessment and Parameter Splicing, which are used to evaluate the knowledge content of multiple models and integrate the knowledge into one model, respectively. The integrated model can be used directly for inference or for further fine-tuning. We conduct extensive experiments on various datasets for recommendation and language tasks, and the results show that Compatibility-aware Knowledge Integration can effectively optimize incompatible parameters under multiple tasks and settings to break through the training limit of the original model without increasing the inference cost. 

**Abstract (ZH)**: 深度神经网络已成为多个领域发展的基础，包括推荐系统和自然语言处理等。尽管取得了成功，这些模型中往往包含不兼容的参数，这些参数可能被未充分利用或对模型性能造成负面影响，尤其是在面对特定、变化的数据分布时。现有研究在移除这些参数或合并多个预训练模型的输出方面表现出色。然而，前者侧重于效率而非性能，而后者需要更多的计算和存储资源来支持推理。在本文中，我们设定的目标是通过利用不同模型的互补优势来显式地改进这些不兼容的参数，从而直接增强模型而不增加任何额外参数。具体而言，我们提出了兼容性感知知识集成（CKI），它包括参数兼容性评估和参数拼接，用于评估多个模型的知识内容并将这些知识整合到一个模型中。整合后的模型可以直接用于推理，也可以进一步微调。我们在推荐和语言任务的各种数据集上进行了广泛的实验，结果表明，兼容性感知知识集成能够有效优化在多种任务和设置下的不兼容参数，从而突破原始模型的训练限制，而无需增加推理成本。 

---
