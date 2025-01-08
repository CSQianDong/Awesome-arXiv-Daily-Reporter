# Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles 

**Title (ZH)**: LLM校准的影响因素：关于响应一致性、损失函数和提示风格的研究 

**Authors**: Yuxi Xia, Pedro Henrique Luz de Araujo, Klim Zaporojets, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2501.03991)  

**Abstract**: Calibration, the alignment between model confidence and prediction accuracy, is critical for the reliable deployment of large language models (LLMs). Existing works neglect to measure the generalization of their methods to other prompt styles and different sizes of LLMs. To address this, we define a controlled experimental setting covering 12 LLMs and four prompt styles. We additionally investigate if incorporating the response agreement of multiple LLMs and an appropriate loss function can improve calibration performance. Concretely, we build Calib-n, a novel framework that trains an auxiliary model for confidence estimation that aggregates responses from multiple LLMs to capture inter-model agreement. To optimize calibration, we integrate focal and AUC surrogate losses alongside binary cross-entropy. Experiments across four datasets demonstrate that both response agreement and focal loss improve calibration from baselines. We find that few-shot prompts are the most effective for auxiliary model-based methods, and auxiliary models demonstrate robust calibration performance across accuracy variations, outperforming LLMs' internal probabilities and verbalized confidences. These insights deepen the understanding of influence factors in LLM calibration, supporting their reliable deployment in diverse applications. 

**Abstract (ZH)**: 校准，即模型的信心与预测准确性之间的对齐，对于大型语言模型（LLMs）的可靠部署至关重要。现有研究在衡量其方法对其他提示风格和其他大小的LLMs的一般化能力方面存在不足。为解决这一问题，我们定义了一个受控实验设置，涵盖了12种LLM和四种提示风格。我们还进一步探讨了是否可以结合多个LLM的回答一致性以及适当的损失函数来提升校准性能。具体而言，我们构建了一个名为Calib-n的新框架，该框架训练一个辅助模型以估计多种LLM的回答汇总后的共识。为了优化校准，我们结合了聚焦损失和AUC代用品损失与二元交叉熵损失。在四个数据集上的实验表明，回答一致性与聚焦损失都能改进校准性能。我们发现，少量示例的提示对基于辅助模型的方法最为有效，而辅助模型在不同准确度变化下的校准性能表现稳定，优于LLM内部概率和口头表述的信心度。这些见解加深了对LLM校准影响因素的理解，支持其在各种应用中的可靠部署。 

---
# Semantically Cohesive Word Grouping in Indian Languages 

**Title (ZH)**: 印度语言中的语义连贯词组聚类 

**Authors**: N J Karthika, Adyasha Patra, Nagasai Saketh Naidu, Arnab Bhattacharya, Ganesh Ramakrishnan, Chaitali Dangarikar  

**Link**: [PDF](https://arxiv.org/pdf/2501.03988)  

**Abstract**: Indian languages are inflectional and agglutinative and typically follow clause-free word order. The structure of sentences across most major Indian languages are similar when their dependency parse trees are considered. While some differences in the parsing structure occur due to peculiarities of a language or its preferred natural way of conveying meaning, several apparent differences are simply due to the granularity of representation of the smallest semantic unit of processing in a sentence. The semantic unit is typically a word, typographically separated by whitespaces. A single whitespace-separated word in one language may correspond to a group of words in another. Hence, grouping of words based on semantics helps unify the parsing structure of parallel sentences across languages and, in the process, morphology. In this work, we propose word grouping as a major preprocessing step for any computational or linguistic processing of sentences for Indian languages. Among Indian languages, since Hindi is one of the least agglutinative, we expect it to benefit the most from word-grouping. Hence, in this paper, we focus on Hindi to study the effects of grouping. We perform quantitative assessment of our proposal with an intrinsic method that perturbs sentences by shuffling words as well as an extrinsic evaluation that verifies the importance of word grouping for the task of Machine Translation (MT) using decomposed prompting. We also qualitatively analyze certain aspects of the syntactic structure of sentences. Our experiments and analyses show that the proposed grouping technique brings uniformity in the syntactic structures, as well as aids underlying NLP tasks. 

**Abstract (ZH)**: 印度语言是屈折语和粘着语，通常遵循无从句的词序。在大多数主要印度语言中，当考虑依赖解析树时，句子的结构在很大程度上是相似的。虽然由于某种语言特有的特点或其自然表达意义的方式，某些解析结构差异会发生，但许多显而易见的差异仅仅是由于句子中最小语义单元表示的粒度所致。这个最小语义单元通常是单词，通过空格分隔。一个语言中空格分隔的单词组可能对应于另一个语言中的多个单词。因此，基于语义分组单词有助于统一不同语言平行句子的解析结构，同时对形态学产生积极影响。在这项工作中，我们提议将单词分组作为对印度语言进行任何计算或语言处理的主要预处理步骤之一。由于印度语言中，印地语是粘着性最弱的语言之一，我们预计它将最受益于单词分组。因此，在本文中，我们将重点研究印地语，以探讨分组的影响。我们通过一种内在方法评估我们的提议，该方法通过洗牌单词来扰乱句子，并通过分解提示进行的外部评估验证单词分组对机器翻译（MT）任务的重要性。我们还对句子的句法结构进行了定性的分析。我们的实验和分析表明，所提出的分组技术在句法结构上带来了一致性，并有助于底层的自然语言处理任务。 

---
# Localizing AI: Evaluating Open-Weight Language Models for Languages of Baltic States 

**Title (ZH)**: 《本地化的人工智能：波罗的海国家开放权重语言模型评价》

这个标题翻译成中文既保留了原文的学术规范，又通俗易懂。如果需要更深入的学术表达，可以进一步调整为：

《本地化的人工智能：波罗的海国家开放权重语言模型的应用研究》

这样更符合正式学术论文标题的表达方式。 

**Authors**: Jurgita Kapočiūtė-Dzikienė, Toms Bergmanis, Mārcis Pinnis  

**Link**: [PDF](https://arxiv.org/pdf/2501.03952)  

**Abstract**: Although large language models (LLMs) have transformed our expectations of modern language technologies, concerns over data privacy often restrict the use of commercially available LLMs hosted outside of EU jurisdictions. This limits their application in governmental, defence, and other data-sensitive sectors. In this work, we evaluate the extent to which locally deployable open-weight LLMs support lesser-spoken languages such as Lithuanian, Latvian, and Estonian. We examine various size and precision variants of the top-performing multilingual open-weight models, Llama~3, Gemma~2, Phi, and NeMo, on machine translation, multiple-choice question answering, and free-form text generation. The results indicate that while certain models like Gemma~2 perform close to the top commercially available models, many LLMs struggle with these languages. Most surprisingly, however, we find that these models, while showing close to state-of-the-art translation performance, are still prone to lexical hallucinations with errors in at least 1 in 20 words for all open-weight multilingual LLMs. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）已经改变了我们对现代语言技术的期望，但数据隐私方面的担忧往往限制了在欧盟管辖范围以外托管的商用LLMs的应用。这限制了它们在政府、国防和其他涉及敏感数据的领域的应用。本研究评估了可本地部署的开放权重LLMs在支持较少使用的语言（如立陶宛语、拉脱维亚语和爱沙尼亚语）方面的适用范围。我们针对多种大小和精度变体的顶级多语言开放权重模型Llama~3、Gemma~2、Phi和NeMo进行了机器翻译、多项选择题作答和自由文本生成任务的评估。结果表明，虽然某些模型如Gemma~2的表现接近商用顶级模型，但许多LLMs在处理这些语言时表现不佳。然而，最令人惊讶的是，尽管这些模型在翻译性能上接近于最先进的水平，几乎所有多语言开放权重LLMs仍然容易出现词汇幻觉错误，至少每20个单词中就有1个错误。 

---
# Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection 

**Title (ZH)**: 不是所有词元都平等：困惑度加权注意力网络在检测AI生成文本中的应用 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2501.03940)  

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅猛发展显著增强了其生成连贯且上下文相关文本的能力，引发了对AI生成内容滥用的关注，使检测这些内容变得至关重要。然而，这一任务仍然极具挑战性，尤其是在未见过的领域或面对不熟悉的LLMs时。利用LLMs的下一个词分布输出提供了一种有理论吸引力的检测方法，因为这些输出汇总了模型在各种语料库上广泛预训练的见解。尽管有其潜力，尝试将这些输出操作化的零样本方法并未取得显著成功。我们推测其中一个问题是，这些方法使用均值来聚合逐个单词的下一个词分布度量，而一些单词自然地更容易或更难预测，应给予不同的权重。基于这一想法，我们提出了困惑度注意加权网络（PAWN），该网络使用LLM的最后隐藏状态和位置，根据序列长度中逐个单词的下一个词分布度量加权序列特征的和。虽然不是零样本方法，但我们的方法可以通过磁盘缓存最后的隐藏状态和下一个词分布度量，极大地减少了训练资源的需求。PAWN在分布内在的表现与最强大的基线方法（微调的语言模型）相当甚至更好，而只需要它们可训练参数的极小部分。此外，我们的模型在未见过的领域和源模型上的泛化能力更强，决策边界在分布转移中表现出较小的变异性。我们的模型还更 robust 对抗攻击，如果骨干具有多语言能力，它在未监督训练中未见过的语言上的泛化表现也令人满意，LLaMA3-1B 在九种语言交叉验证中的平均宏F1分数达到了81.46%。 

---
# AlphaPO -- Reward shape matters for LLM alignment 

**Title (ZH)**: AlphaPO —— 奖励形状对于LLM对齐至关重要 

**Authors**: Aman Gupta, Shao Tang, Qingquan Song, Sirou Zhu, Jiwoo Hong, Ankan Saha, Viral Gupta, Noah Lee, Eunki Kim, Jason Zhu, Natesh Pillai, S. Sathiya Keerthi  

**Link**: [PDF](https://arxiv.org/pdf/2501.03884)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Examples include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably.
In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce AlphaPO, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance. 

**Abstract (ZH)**: 强化学习与人类反馈（RLHF）及其变体在使大型语言模型（LLMs）有效遵循指令和反映人类价值观方面取得了巨大的进步。最近，直接对齐算法（DAAs）的出现通过直接将奖励建模为所学习策略的函数，省略了RLHF中的奖励建模阶段。这些方法包括直接偏好优化（DPO）和简单偏好优化（SimPO）。这些方法通常会遭受似然性置换的现象，即优选响应的概率往往会不必要地减少。

在本文中，我们认为，对于DAAs而言，奖励（函数）的形状很重要。我们提出了AlphaPO，这是一种新的DAAs方法，利用一个$\alpha$参数来帮助改变奖励函数的形状，超越了标准的对数奖励。AlphaPO有助于保持对似然性置换和过度优化的精细控制。与SimPO（目前已知表现最好的DAAs之一）相比，AlphaPO在Mistral-7B和Llama3-8B的指令版本中的对齐性能相对提高了约7%到10%。分析和结果表明了奖励形状的重要性，以及如何系统地改变它以影响训练动态，并提高对齐性能。 

---
# Add Noise, Tasks, or Layers? MaiNLP at the VarDial 2025 Shared Task on Norwegian Dialectal Slot and Intent Detection 

**Title (ZH)**: 增加噪声、任务还是层？MaiNLP在2025年VarDial共享任务中的挪威方言槽位和意图检测

注释：
1. "VarDial" 是一个国际性的评价挑战赛，全称是"Variety and Diligence"。在此处保持原文名称不变。
2. "Shared Task" 在学术论文中通常指的是共享任务或挑战赛，这里也保持原文不变。
3. "MaiNLP" 是一个研究组织或团队的名称，此处也保持不变。
4. "Norwegian Dialectal Slot and Intent Detection" 翻译为“挪威方言槽位和意图检测”，其中“槽位”是指实体识别中的标签或槽，与意图检测结合则是指识别对话中的特定意图或槽位。

最终翻译保证了学术规范和准确性，同时符合中文表达习惯。 

**Authors**: Verena Blaschke, Felicia Körner, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2501.03870)  

**Abstract**: Slot and intent detection (SID) is a classic natural language understanding task. Despite this, research has only more recently begun focusing on SID for dialectal and colloquial varieties. Many approaches for low-resource scenarios have not yet been applied to dialectal SID data, or compared to each other on the same datasets. We participate in the VarDial 2025 shared task on slot and intent detection in Norwegian varieties, and compare multiple set-ups: varying the training data (English, Norwegian, or dialectal Norwegian), injecting character-level noise, training on auxiliary tasks, and applying Layer Swapping, a technique in which layers of models fine-tuned on different datasets are assembled into a model. We find noise injection to be beneficial while the effects of auxiliary tasks are mixed. Though some experimentation was required to successfully assemble a model from layers, it worked surprisingly well; a combination of models trained on English and small amounts of dialectal data produced the most robust slot predictions. Our best models achieve 97.6% intent accuracy and 85.6% slot F1 in the shared task. 

**Abstract (ZH)**: 槽位和意图检测（SID）是经典自然语言理解任务。尽管如此，研究仅在近期才开始关注其在方言和非正式语言变体中的应用。许多针对低资源环境的方法尚未应用于方言SID数据，或在相同的数据集上进行比较。我们参加了2025年VarDial共享任务中的挪威方言槽位和意图检测任务，并比较了多种设置：不同的训练数据（英语、标准挪威语或方言挪威语）、注入字符级别的噪声、进行辅助任务训练，以及应用Layer Swapping技术。该技术涉及组装由不同数据集上微调的模型的层组成的新模型。我们发现注入噪声是有益的，而辅助任务的效果则令人 mixed。尽管需要一些实验来成功组装模型，但效果出乎意料地好；结合英语训练的模型和少量方言数据训练的模型的组合产生了最稳健的槽位预测结果。我们的最佳模型在共享任务中达到了97.6%的意图准确率和85.6%的槽位F1值。 

---
# Improving Dialectal Slot and Intent Detection with Auxiliary Tasks: A Multi-Dialectal Bavarian Case Study 

**Title (ZH)**: 改进方言槽位和意图检测：多方言巴伐利亚案例研究中的辅助任务应用 

**Authors**: Xaver Maria Krückl, Verena Blaschke, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2501.03863)  

**Abstract**: Reliable slot and intent detection (SID) is crucial in natural language understanding for applications like digital assistants. Encoder-only transformer models fine-tuned on high-resource languages generally perform well on SID. However, they struggle with dialectal data, where no standardized form exists and training data is scarce and costly to produce. We explore zero-shot transfer learning for SID, focusing on multiple Bavarian dialects, for which we release a new dataset for the Munich dialect. We evaluate models trained on auxiliary tasks in Bavarian, and compare joint multi-task learning with intermediate-task training. We also compare three types of auxiliary tasks: token-level syntactic tasks, named entity recognition (NER), and language modelling. We find that the included auxiliary tasks have a more positive effect on slot filling than intent classification (with NER having the most positive effect), and that intermediate-task training yields more consistent performance gains. Our best-performing approach improves intent classification performance on Bavarian dialects by 5.1 and slot filling F1 by 8.4 percentage points. 

**Abstract (ZH)**: 可靠的槽位和意图检测（SID）对于数字助手等自然语言理解应用至关重要。仅使用编码器的变压器模型在高资源语言上进行微调通常在SID任务上表现良好。然而，当面临没有标准化形式且缺乏且生产成本高昂的方言数据时，它们的表现会遇到困难。我们探索了零样本迁移学习在SID中的应用，重点研究了多种巴伐利亚方言，并为此发布了一个新的慕尼黑方言数据集。我们评估了在巴伐利亚语辅助任务上训练的模型，并比较了联合多任务学习与中间任务训练的方法。我们还比较了三种类型的辅助任务：标记级别的句法任务、命名实体识别（NER）和语言建模。研究结果表明，包含的辅助任务对槽位填充的影响更为积极，而对意图分类的影响较小（其中NER的影响最大），且中间任务训练能提供更一致的性能提升。我们最佳的方法在巴伐利亚方言上的意图分类性能提高了5.1个百分点，槽位填充F1分数提高了8.4个百分点。 

---
# Progressive Document-level Text Simplification via Large Language Models 

**Title (ZH)**: 通过大型语言模型实现逐步文档级文本简化 

**Authors**: Dengzhao Fang, Jipeng Qiang, Yi Zhu, Yunhao Yuan, Wei Li, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.03857)  

**Abstract**: Research on text simplification has primarily focused on lexical and sentence-level changes. Long document-level simplification (DS) is still relatively unexplored. Large Language Models (LLMs), like ChatGPT, have excelled in many natural language processing tasks. However, their performance on DS tasks is unsatisfactory, as they often treat DS as merely document summarization. For the DS task, the generated long sequences not only must maintain consistency with the original document throughout, but complete moderate simplification operations encompassing discourses, sentences, and word-level simplifications. Human editors employ a hierarchical complexity simplification strategy to simplify documents. This study delves into simulating this strategy through the utilization of a multi-stage collaboration using LLMs. We propose a progressive simplification method (ProgDS) by hierarchically decomposing the task, including the discourse-level, topic-level, and lexical-level simplification. Experimental results demonstrate that ProgDS significantly outperforms existing smaller models or direct prompting with LLMs, advancing the state-of-the-art in the document simplification task. 

**Abstract (ZH)**: 文本简化研究主要集中在词汇和句级变化上。长文档级简化（DS）仍然相对未被充分探索。大型语言模型（LLMs），如ChatGPT，在许多自然语言处理任务中表现出色。然而，在DS任务上的表现不尽如人意，因为它们通常将DS视为文档摘要。对于DS任务而言，生成的长序列不仅需要在整个文档中保持一致性，还需执行涵盖话语、句子和词汇级简化在内的适度简化操作。人类编辑者通过层级复杂性简化策略来简化文档。本研究通过利用多阶段LLMs协作，尝试模拟这一策略。我们通过层级分解任务（包括话语级、主题级和词汇级简化）提出了一种逐步简化方法（ProgDS）。实验结果表明，ProgDS 在文档简化任务中的表现显著优于现有较小的模型或直接使用LLMs进行提示，推动了该领域的技术前沿。 

---
# BabyLMs for isiXhosa: Data-Efficient Language Modelling in a Low-Resource Context 

**Title (ZH)**: 面向伊西乔加语的 BabyLMs：低资源环境下的数据高效语言模型 

**Authors**: Alexis Matzopoulos, Charl Hendriks, Hishaam Mahomed, Francois Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2501.03855)  

**Abstract**: The BabyLM challenge called on participants to develop sample-efficient language models. Submissions were pretrained on a fixed English corpus, limited to the amount of words children are exposed to in development (<100m). The challenge produced new architectures for data-efficient language modelling, which outperformed models trained on trillions of words. This is promising for low-resource languages, where available corpora are limited to much less than 100m words. In this paper, we explore the potential of BabyLMs for low-resource languages, using the isiXhosa language as a case study. We pretrain two BabyLM architectures, ELC-BERT and MLSM, on an isiXhosa corpus. They outperform a vanilla pretrained model on POS tagging and NER, achieving notable gains (+3.2 F1) for the latter. In some instances, the BabyLMs even outperform XLM-R. Our findings show that data-efficient models are viable for low-resource languages, but highlight the continued importance, and lack of, high-quality pretraining data. Finally, we visually analyse how BabyLM architectures encode isiXhosa. 

**Abstract (ZH)**: BabyLM 挑战赛呼吁参赛者开发样本效率高的语言模型。提交的作品在固定英文语料库上预训练，限制在儿童在发展中接触到的词汇量（<100万词）。该挑战产生了新的数据效率高的语言模型架构，这些架构的表现超过了在万亿级词汇上训练的模型。这对于资源稀缺的语言非常有前景，因为可用的语料库通常远少于100万词。在本文中，我们探讨了BabyLMs在资源稀缺语言中的潜在应用，以祖鲁语（isiXhosa）为例进行研究。我们分别在祖鲁语语料库上预训练了两个BabyLM架构——ELC-BERT和MLSM。它们在词性标注和命名实体识别任务上均超过了普通的预训练模型，后者在命名实体识别任务上的改进尤为显著（+3.2 F1）。在某些情况下，BabyLM甚至超越了XLM-R。我们的研究结果表明，数据效率高的模型对于资源稀缺的语言是可行的，但同时也突显了高质量预训练数据的重要性及其稀缺性。最后，我们通过可视化分析探讨了BabyLM架构如何编码祖鲁语。 

---
# TACLR: A Scalable and Efficient Retrieval-based Method for Industrial Product Attribute Value Identification 

**Title (ZH)**: TACLR：一种可扩展且高效的产品属性值检索识别方法 

**Authors**: Yindu Su, Huike Zou, Lin Sun, Ting Zhang, Haiyang Yang, Liyu Chen, David Lo, Qingheng Zhang, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.03835)  

**Abstract**: Product Attribute Value Identification (PAVI) involves identifying attribute values from product profiles, a key task for improving product search, recommendations, and business analytics on e-commerce platforms. However, existing PAVI methods face critical challenges, such as inferring implicit values, handling out-of-distribution (OOD) values, and producing normalized outputs. To address these limitations, we introduce Taxonomy-Aware Contrastive Learning Retrieval (TACLR), the first retrieval-based method for PAVI. TACLR formulates PAVI as an information retrieval task by encoding product profiles and candidate values into embeddings and retrieving values based on their similarity to the item embedding. It leverages contrastive training with taxonomy-aware hard negative sampling and employs adaptive inference with dynamic thresholds. TACLR offers three key advantages: (1) it effectively handles implicit and OOD values while producing normalized outputs; (2) it scales to thousands of categories, tens of thousands of attributes, and millions of values; and (3) it supports efficient inference for high-load industrial scenarios. Extensive experiments on proprietary and public datasets validate the effectiveness and efficiency of TACLR. Moreover, it has been successfully deployed in a real-world e-commerce platform, processing millions of product listings daily while supporting dynamic, large-scale attribute taxonomies. 

**Abstract (ZH)**: 产品属性值识别（PAVI）涉及从产品简介中识别属性值，这是改善电子商务平台上产品搜索、推荐和商业分析的关键任务。然而，现有的PAVI方法面临一些关键挑战，如推断隐含值、处理未见分布（OOD）值以及生成规范化输出。为了解决这些局限性，我们引入了 Taxonomy-Aware Contrastive Learning Retrieval（TACLR），这是一种用于PAVI的第一个检索式方法。TACLR将PAVI表述为一个信息检索任务，通过嵌入产品简介和候选值来生成向量表示，并基于其与项嵌入的相似性检索值。它利用带有分类知识的对比训练和硬负样本采样，并采用自适应推理和动态阈值。TACLR提供了三个主要优势：（1）它有效地处理隐含值和OOD值，并生成规范化输出；（2）它能够扩展到数千个类别、数十万个属性和数百万个值；（3）它支持高负载工业场景的高效推理。广泛实验在私有数据集和公开数据集上验证了TACLR的有效性和效率。此外，它已在实际的电子商务平台中成功部署，并每日处理数百万份产品列表，同时支持动态、大规模的属性分类法。 

---
# Investigating the Impact of Data Selection Strategies on Language Model Performance 

**Title (ZH)**: 探究数据选择策略对语言模型性能的影响 

**Authors**: Jiayao Gu, Liting Chen, Yihong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.03826)  

**Abstract**: Data selection is critical for enhancing the performance of language models, particularly when aligning training datasets with a desired target distribution. This study explores the effects of different data selection methods and feature types on model performance. We evaluate whether selecting data subsets can influence downstream tasks, whether n-gram features improve alignment with target distributions, and whether embedding-based neural features provide complementary benefits. Through comparative experiments using baseline random selection methods and distribution aligned approaches, we provide insights into the interplay between data selection strategies and model training efficacy. All code for this study can be found on \href{this https URL}{github repository}. 

**Abstract (ZH)**: 数据选择对于提升语言模型的性能至关重要，特别是在将训练数据集与期望的目标分布对齐时。本研究探讨了不同数据选择方法和特征类型对模型性能的影响。我们评估了选择数据子集是否能够影响下游任务，n-gram特征是否能够改善与目标分布的对齐程度，以及基于嵌入的神经特征是否提供额外的好处。通过使用基准的随机选择方法和分布对齐方法进行比较实验，我们提供了数据选择策略与模型训练效果之间相互作用的见解。本研究的所有代码可以在 \href{这个github仓库}{github repository} 中找到。 

---
# Unsupervised Speech Segmentation: A General Approach Using Speech Language Models 

**Title (ZH)**: 无监督语音分割：基于语音语言模型的通用方法 

**Authors**: Avishai Elmakies, Omri Abend, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2501.03711)  

**Abstract**: In this paper, we introduce an unsupervised approach for Speech Segmentation, which builds on previously researched approaches, e.g., Speaker Diarization, while being applicable to an inclusive set of acoustic-semantic distinctions, paving a path towards a general Unsupervised Speech Segmentation approach. Unlike traditional speech and audio segmentation, which mainly focuses on spectral changes in the input signal, e.g., phone segmentation, our approach tries to segment the spoken utterance into chunks with differing acoustic-semantic styles, focusing on acoustic-semantic information that does not translate well into text, e.g., emotion or speaker. While most Speech Segmentation tasks only handle one style change, e.g., emotion diarization, our approach tries to handle multiple acoustic-semantic style changes. Leveraging recent advances in Speech Language Models (SLMs), we propose a simple unsupervised method to segment a given speech utterance. We empirically demonstrate the effectiveness of the proposed approach by considering several setups. Results suggest that the proposed method is superior to the evaluated baselines on boundary detection, segment purity, and over-segmentation. Code is available at this https URL. 

**Abstract (ZH)**: 在本文中，我们提出了一种无监督的语音分割方法，该方法基于先前研究的说话人辩识（Speaker Diarization）等方法，适用于广泛的声学-语义区分，从而为通用的无监督语音分割方法开辟了道路。与传统语音和音频分割主要集中在输入信号的频谱变化（例如，音素分割）不同，我们的方法旨在将口头表达划分为具有不同声学-语义样式的片段，重点是难以通过文本表达的声学-语义信息，例如情感或说话者。大多数语音分割任务只能处理一种样式的转换（例如，情绪辩识），而我们的方法旨在处理多种声学-语义样式的转换。利用语音语言模型（SLMs）的最新进展，我们提出了一种简单且无监督的分割方法，以对给定的语音片段进行分割。通过几种设置下的实证研究，我们证明了所提出方法的有效性。结果表明，所提出的方法在边界检测、片段纯净度和过度分割方面优于评估的基本方法。代码可在以下网址获得：https://github.com/username/repo。 

---
# SLAM: Towards Efficient Multilingual Reasoning via Selective Language Alignment 

**Title (ZH)**: SLAM：通过选择性语言对齐实现高效的多语言推理 

**Authors**: Yuchun Fan, Yongyu Mu, Yilin Wang, Lei Huang, Junhao Ruan, Bei Li, Tong Xiao, Shujian Huang, Xiaocheng Feng, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.03681)  

**Abstract**: Despite the significant improvements achieved by large language models (LLMs) in English reasoning tasks, these models continue to struggle with multilingual reasoning. Recent studies leverage a full-parameter and two-stage training paradigm to teach models to first understand non-English questions and then reason. However, this method suffers from both substantial computational resource computing and catastrophic forgetting. The fundamental cause is that, with the primary goal of enhancing multilingual comprehension, an excessive number of irrelevant layers and parameters are tuned during the first stage. Given our findings that the representation learning of languages is merely conducted in lower-level layers, we propose an efficient multilingual reasoning alignment approach that precisely identifies and fine-tunes the layers responsible for handling multilingualism. Experimental results show that our method, SLAM, only tunes 6 layers' feed-forward sub-layers including 6.5-8% of all parameters within 7B and 13B LLMs, achieving superior average performance than all strong baselines across 10 languages. Meanwhile, SLAM only involves one training stage, reducing training time by 4.1-11.9 compared to the two-stage method. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在英语推理任务中取得了显著进步，但这些模型在多语言推理方面仍然存在困难。近期的研究利用全参数和两阶段训练范式，首先教会模型理解非英语问题，然后进行推理。然而，这种方法存在计算资源需求庞大以及灾难性遗忘的问题。根本原因是，在多语言理解为主要目标的情况下，第一阶段中过度调整了无关的多层网络和参数。鉴于我们发现语言表示学习主要在较低层级的层中进行，我们提出了一种高效的多语言推理对齐方法，该方法精确地识别并调整负责处理多语言性的层。实验结果表明，我们的方法SLAM仅在7B和13B LLM中调整了6层的前向子层，占总参数的6.5-8%，在10种语言上实现了优于所有强大基线的平均性能。同时，SLAM仅涉及一个训练阶段，相比两阶段方法减少了4.1-11.9倍的训练时间。 

---
# A Diversity-Enhanced Knowledge Distillation Model for Practical Math Word Problem Solving 

**Title (ZH)**: 一种增强多样性知识精炼模型以解决实际数学文字问题 

**Authors**: Yi Zhang, Guangyou Zhou, Zhiwen Xie, Jinjin Ma, Jimmy Xiangji Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.03670)  

**Abstract**: Math Word Problem (MWP) solving is a critical task in natural language processing, has garnered significant research interest in recent years. Various recent studies heavily rely on Seq2Seq models and their extensions (e.g., Seq2Tree and Graph2Tree) to generate mathematical equations. While effective, these models struggle to generate diverse but counterpart solution equations, limiting their generalization across various math problem scenarios. In this paper, we introduce a novel Diversity-enhanced Knowledge Distillation (DivKD) model for practical MWP solving. Our approach proposes an adaptive diversity distillation method, in which a student model learns diverse equations by selectively transferring high-quality knowledge from a teacher model. Additionally, we design a diversity prior-enhanced student model to better capture the diversity distribution of equations by incorporating a conditional variational auto-encoder. Extensive experiments on {four} MWP benchmark datasets demonstrate that our approach achieves higher answer accuracy than strong baselines while maintaining high efficiency for practical applications. 

**Abstract (ZH)**: 数学单词问题（MWP）求解是自然语言处理中的一个关键任务，在近年来引起了广泛的研究兴趣。近年来，各种研究主要依赖序列到序列（Seq2Seq）模型及其扩展（例如，Seq2Tree和Graph2Tree）来生成数学方程。虽然这些模型在生成方程方面表现出效用，但它们在生成多样但对等的解决方案方程方面能力有限，这限制了它们在各种数学问题场景中的泛化能力。在本文中，我们提出了一种新的增强多样性知识蒸馏（DivKD）模型，以实现实用的MWP求解。我们的方法提出了一种适应性的多样性蒸馏方法，在这种方法中，学生模型通过选择性地从教师模型中传递高质量的知识来学习多样性的方程。此外，我们设计了一种增强多样性先验的学生模型，通过引入条件变分自编码器更好地捕捉方程的多样性分布。在四个MWP基准数据集上的广泛实验表明，我们的方法在保持高效率的同时，实现了比强基线更高的答案准确性。 

---
# KG-TRICK: Unifying Textual and Relational Information Completion of Knowledge for Multilingual Knowledge Graphs 

**Title (ZH)**: KG-TRICK：统一多语言知识图谱中文本和关系信息完成方法 

**Authors**: Zelin Zhou, Simone Conia, Daniel Lee, Min Li, Shenglei Huang, Umar Farooq Minhas, Saloni Potdar, Henry Xiao, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.03560)  

**Abstract**: Multilingual knowledge graphs (KGs) provide high-quality relational and textual information for various NLP applications, but they are often incomplete, especially in non-English languages. Previous research has shown that combining information from KGs in different languages aids either Knowledge Graph Completion (KGC), the task of predicting missing relations between entities, or Knowledge Graph Enhancement (KGE), the task of predicting missing textual information for entities. Although previous efforts have considered KGC and KGE as independent tasks, we hypothesize that they are interdependent and mutually beneficial. To this end, we introduce KG-TRICK, a novel sequence-to-sequence framework that unifies the tasks of textual and relational information completion for multilingual KGs. KG-TRICK demonstrates that: i) it is possible to unify the tasks of KGC and KGE into a single framework, and ii) combining textual information from multiple languages is beneficial to improve the completeness of a KG. As part of our contributions, we also introduce WikiKGE10++, the largest manually-curated benchmark for textual information completion of KGs, which features over 25,000 entities across 10 diverse languages. 

**Abstract (ZH)**: 多语言知识图谱（KGs）为各种自然语言处理（NLP）应用提供了高质量的关系和文本信息，但它们通常存在不完整的情况，尤其是在非英文语言方面。已有研究显示，结合不同语言知识图谱中的信息可以分别促进知识图谱完成（KGC，Knowledge Graph Completion）任务，即预测实体间缺失的关系，或者知识图谱增强（KGE，Knowledge Graph Enhancement）任务，即预测实体缺失的文本信息。尽管先前的努力将KGC和KGE视为独立任务，但我们假设它们之间是相互依赖且互惠互利的。为此，我们提出了一种新的序列到序列框架KG-TRICK，该框架统一了多语言知识图谱中关系和文本信息完成的任务。KG-TRICK表明：i）可以通过单一框架统一KGC和KGE任务；ii）结合多种语言的文本信息有助于提高知识图谱的完整性。作为我们的贡献之一，我们还引入了WikiKGE10++，这是目前最大的手工构建的文本信息完成基准数据集，涵盖了超过25,000个实体，涉及10种不同的语言。 

---
# Beyond Factual Accuracy: Evaluating Coverage of Diverse Factual Information in Long-form Text Generation 

**Title (ZH)**: 超越事实准确性：评估长文生成中多样事实信息覆盖的情况 

**Authors**: Chris Samarinas, Alexander Krubner, Alireza Salemi, Youngwoo Kim, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.03545)  

**Abstract**: This paper presents ICAT, an evaluation framework for measuring coverage of diverse factual information in long-form text generation. ICAT breaks down a long output text into a list of atomic claims and not only verifies each claim through retrieval from a (reliable) knowledge source, but also computes the alignment between the atomic factual claims and various aspects expected to be presented in the output. We study three implementations of the ICAT framework, each with a different assumption on the availability of aspects and alignment method. By adopting data from the diversification task in the TREC Web Track and the ClueWeb corpus, we evaluate the ICAT framework. We demonstrate strong correlation with human judgments and provide comprehensive evaluation across multiple state-of-the-art LLMs. Our framework further offers interpretable and fine-grained analysis of diversity and coverage. Its modular design allows for easy adaptation to different domains and datasets, making it a valuable tool for evaluating the qualitative aspects of long-form responses produced by LLMs. 

**Abstract (ZH)**: 本文介绍了ICAT，一种用于评估长文本生成中多样化事实信息覆盖情况的评价框架。ICAT将长文本输出分解为一系列原子性断言，并不仅通过检索可靠的知识来源来验证每个断言，还计算原子性事实断言与预期在输出中呈现的各种方面的对齐情况。我们研究了ICAT框架的三种实现方式，每种实现方式在方面可用性和对齐方法方面有不同的假设。通过采用TREC Web Track的多样化任务数据和ClueWeb语料库的数据，我们评估了ICAT框架。我们展示了其与人工判断高度相关，并对多种最先进的大型语言模型进行了全面评估。该框架还提供了可解释且细粒度的多样性和覆盖分析，并具有模块化设计，易于适应不同的领域和数据集，使其成为评估大型语言模型生成的长文本响应的定性方面的一个有价值的工具。 

---
# A Sequential Optimal Learning Approach to Automated Prompt Engineering in Large Language Models 

**Title (ZH)**: 面向大型语言模型自动提示工程的序贯最优学习方法 

**Authors**: Shuyang Wang, Somayeh Moazeni, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2501.03508)  

**Abstract**: Designing effective prompts is essential to guiding large language models (LLMs) toward desired responses. Automated prompt engineering aims to reduce reliance on manual effort by streamlining the design, refinement, and optimization of natural language prompts. This paper proposes an optimal learning framework for automated prompt engineering, designed to sequentially identify effective prompt features while efficiently allocating a limited evaluation budget. We introduce a feature-based method to express prompts, which significantly broadens the search space. Bayesian regression is employed to utilize correlations among similar prompts, accelerating the learning process. To efficiently explore the large space of prompt features for a high quality prompt, we adopt the forward-looking Knowledge-Gradient (KG) policy for sequential optimal learning. The KG policy is computed efficiently by solving mixed-integer second-order cone optimization problems, making it scalable and capable of accommodating prompts characterized only through constraints. We demonstrate that our method significantly outperforms a set of benchmark strategies assessed on instruction induction tasks. The results highlight the advantages of using the KG policy for prompt learning given a limited evaluation budget. Our framework provides a solution to deploying automated prompt engineering in a wider range applications where prompt evaluation is costly. 

**Abstract (ZH)**: 设计有效的提示是引导大型语言模型（LLMs）生成所需响应的关键。自动提示工程旨在通过简化提示设计、优化和精炼过程来减少对人工努力的依赖。本文提出了一种优化的学习框架，旨在顺序识别有效的提示特征，并高效地分配有限的评估预算。我们提出了一种基于特征的方法来表示提示，这大大扩展了搜索空间。通过使用贝叶斯回归，我们利用类似提示之间的相关性，加速了学习过程。为了高效地探索提示特征的大空间以生成高质量的提示，我们采用前瞻性知识梯度（KG）策略进行顺序优化学习。KG策略通过求解混合整数二次锥优化问题来高效计算，使其具有可扩展性，并能够处理仅通过约束来描述的提示。我们的方法在指令归纳任务中显著优于一组基准策略。结果表明，使用KG策略进行提示学习具有有限评估预算的优势。我们的框架提供了一种解决方案，可以将自动化提示工程应用到提示评估成本较高的更广泛的应用场景中。 

---
# Can LLMs Design Good Questions Based on Context? 

**Title (ZH)**: 大规模语言模型能否基于上下文设计出好的问题？ 

**Authors**: Yueheng Zhang, Xiaoyuan Liu, Yiyou Sun, Atheer Alharbi, Hend Alzahrani, Basel Alomair, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.03491)  

**Abstract**: This paper evaluates questions generated by LLMs from context, comparing them to human-generated questions across six dimensions. We introduce an automated LLM-based evaluation method, focusing on aspects like question length, type, context coverage, and answerability. Our findings highlight unique characteristics of LLM-generated questions, contributing insights that can support further research in question quality and downstream applications. 

**Abstract (ZH)**: 本文评估了由大型语言模型（LLM）根据上下文生成的问题，并将其与人类生成的问题在六个维度上进行比较。我们引入了一种基于自动LLM的评估方法，重点关注问题长度、类型、上下文覆盖范围和可回答性等方面。我们的研究结果突出了由LLM生成的问题的特殊特征，为今后的问题质量研究及下游应用提供了有价值的见解。 

---
# Women, Infamous, and Exotic Beings: What Honorific Usages in Wikipedia Reveal about the Socio-Cultural Norms 

**Title (ZH)**: 女性、声名狼藉者与奇异之物：维基百科中尊称用法揭示的社造林木规范 

**Authors**: Sourabrata Mukherjee, Soumya Teotia, Sougata Saha, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2501.03479)  

**Abstract**: Honorifics serve as powerful linguistic markers that reflect social hierarchies and cultural values. This paper presents a large-scale, cross-linguistic exploration of usage of honorific pronouns in Bengali and Hindi Wikipedia articles, shedding light on how socio-cultural factors shape language. Using LLM (GPT-4o), we annotated 10, 000 articles of real and fictional beings in each language for several sociodemographic features such as gender, age, fame, and exoticness, and the use of honorifics. We find that across all feature combinations, use of honorifics is consistently more common in Bengali than Hindi. For both languages, the use non-honorific pronouns is more commonly observed for infamous, juvenile, and exotic beings. Notably, we observe a gender bias in use of honorifics in Hindi, with men being more commonly referred to with honorifics than women. 

**Abstract (ZH)**: 敬语是反映社会阶层和文化价值观的强大语言标志。本文呈现了对孟加拉语和印地语维基百科文章中敬称代词使用的大型跨国语言学探索，揭示了社会文化因素如何塑造语言。我们利用大模型（GPT-4o）对每种语言中的10,000篇关于现实和虚构人物的文章进行了注释，并记录了多个社会人口统计特征（如性别、年龄、知名度和异国情调）以及敬语的使用情况。研究发现，在所有特征组合中，孟加拉语中的敬语使用频率均高于印地语。对于两种语言而言，不使用敬语的代词在非著名、少年和异国事物中更为常见。值得注意的是，在印地语中，我们观察到敬语使用的性别偏差，男性比女性更常被用敬语称呼。 

---
# Reading with Intent -- Neutralizing Intent 

**Title (ZH)**: 带有意图的阅读——消除意图 

**Authors**: Benjamin Reichman, Adar Avsian, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2501.03475)  

**Abstract**: Queries to large language models (LLMs) can be divided into two parts: the instruction/question and the accompanying context. The context for retrieval-augmented generation (RAG) systems in most benchmarks comes from Wikipedia or Wikipedia-like texts which are written in a neutral and factual tone. However, when RAG systems retrieve internet-based content, they encounter text with diverse tones and linguistic styles, introducing challenges for downstream tasks. The Reading with Intent task addresses this issue by evaluating how varying tones in context passages affect model performance. Building on prior work that focused on sarcasm, we extend this paradigm by constructing a dataset where context passages are transformed to $11$ distinct emotions using a better synthetic data generation approach. Using this dataset, we train an emotion translation model to systematically adapt passages to specified emotional tones. The human evaluation shows that the LLM fine-tuned to become the emotion-translator benefited from the synthetically generated data. Finally, the emotion-translator is used in the Reading with Intent task to transform the passages to a neutral tone. By neutralizing the passages, it mitigates the challenges posed by sarcastic passages and improves overall results on this task by about $3\%$. 

**Abstract (ZH)**: 大型语言模型（LLMs）的查询可以分为两部分：指令/问题和伴随的背景信息。在大多数基准测试中，RAG（检索增强生成）系统的背景信息来自于维基百科或类似维基百科的文本，这些文本通常以中立和事实性的方式编写。然而，当RAG系统检索互联网内容时，它们会遇到具有各种语气和语言风格的文本，这给下游任务带来了挑战。阅读有目的性的任务通过评估背景段落中不同语气如何影响模型性能来应对这一问题。在此基础上，我们通过使用更好的合成数据生成方法，将背景段落转换为包含11种不同情感的段落，从而扩展了先前专注于讽刺的工作。使用这个数据集，我们训练了一个情感翻译模型，以系统地将段落转换为指定的情感语气。人工评估表明，经过微调以成为情感转换器的LLM受益于合成生成的数据。最后，将情感转换器应用于阅读有目的性的任务，将其背景段落转换为中性语气。通过中性化段落，该方法减少了讽刺段落带来的挑战，并在该任务上提高了约3%的整体结果。 

---
# MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems 

**Title (ZH)**: MTRAG：一种多轮对话基准，用于评估检索增强生成系统 

**Authors**: Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, Marina Danilevsky  

**Link**: [PDF](https://arxiv.org/pdf/2501.03468)  

**Abstract**: Retrieval-augmented generation (RAG) has recently become a very popular task for Large Language Models (LLMs). Evaluating them on multi-turn RAG conversations, where the system is asked to generate a response to a question in the context of a preceding conversation is an important and often overlooked task with several additional challenges. We present MTRAG: an end-to-end human-generated multi-turn RAG benchmark that reflects several real-world properties across diverse dimensions for evaluating the full RAG pipeline. MTRAG contains 110 conversations averaging 7.7 turns each across four domains for a total of 842 tasks. We also explore automation paths via synthetic data and LLM-as-a-Judge evaluation. Our human and automatic evaluations show that even state-of-the-art LLM RAG systems struggle on MTRAG. We demonstrate the need for strong retrieval and generation systems that can handle later turns, unanswerable questions, non-standalone questions, and multiple domains. MTRAG is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）最近已成为大型语言模型（LLMs）的一项非常受欢迎的任务。在多轮RAG对话中评估它们，即系统需要在前一对话的背景下生成对问题的回应，这是一个重要且经常被忽视的任务，其中包含多种额外的挑战。我们提出了MTRAG：一个端到端的人工生成的多轮RAG基准，它在多个维度上反映了多种现实世界的特性，用于全面评估RAG管道。MTRAG包含四个领域共计110个对话，每个对话平均有7.7轮，总共涵盖842项任务。我们还探讨了通过合成数据和LLM作为评判者进行自动化的途径。我们的手工评估和自动评估表明，即使是最先进的LLM RAG系统也难以应对MTRAG中的任务。我们证明了需要强大的检索和生成系统来处理后续轮次、无法回答的问题、非独立的问题以及多个领域。MTRAG可在以下链接获取：[此 https URL](此 https URL)。 

---
# ISSR: Iterative Selection with Self-Review for Vocabulary Test Distractor Generation 

**Title (ZH)**: ISSR：迭代选择与自我审查词汇测试干扰项生成方法 

**Authors**: Yu-Cheng Liu, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2501.03462)  

**Abstract**: Vocabulary acquisition is essential to second language learning, as it underpins all core language skills. Accurate vocabulary assessment is particularly important in standardized exams, where test items evaluate learners' comprehension and contextual use of words. Previous research has explored methods for generating distractors to aid in the design of English vocabulary tests. However, current approaches often rely on lexical databases or predefined rules, and frequently produce distractors that risk invalidating the question by introducing multiple correct options. In this study, we focus on English vocabulary questions from Taiwan's university entrance exams. We analyze student response distributions to gain insights into the characteristics of these test items and provide a reference for future research. Additionally, we identify key limitations in how large language models (LLMs) support teachers in generating distractors for vocabulary test design. To address these challenges, we propose the iterative selection with self-review (ISSR) framework, which makes use of a novel LLM-based self-review mechanism to ensure that the distractors remain valid while offering diverse options. Experimental results show that ISSR achieves promising performance in generating plausible distractors, and the self-review mechanism effectively filters out distractors that could invalidate the question. 

**Abstract (ZH)**: 词汇获取是第二语言学习的基石，因为它是所有核心语言技能的基础。准确的词汇评估在标准化考试中尤为重要，其中测试项目评估的是考生对词汇的理解和在语境中的使用能力。以往的研究探讨了生成干扰项的方法，以辅助设计英语词汇测试题。然而，当前的方法常常依赖于词汇数据库或预定义的规则，往往会产生一些干扰项，这些干扰项有可能导致问题无效化，引入多个正确选项。在本研究中，我们专注于台湾大学入学考试的英语词汇题目。我们分析了学生对题目答案的分布情况，以深入了解这些测试项目的特征，并为未来的研究提供参考。此外，我们还指出了大型语言模型（LLMs）在支持教师生成词汇测试题干扰项方面存在的关键局限性。为应对这些挑战，我们提出了迭代选择与自我审核（ISSR）框架。该框架利用了一种新颖的基于LLM的自我审核机制，以确保干扰项的有效性的同时提供多样化的选项。实验结果表明，ISSR 在生成合理干扰项方面取得了令人鼓舞的性能，而自我审核机制有效地筛选出了可能使问题无效化的干扰项。 

---
# Text to Band Gap: Pre-trained Language Models as Encoders for Semiconductor Band Gap Prediction 

**Title (ZH)**: 文本到带隙：预训练语言模型作为半导体带隙预测的编码器 

**Authors**: Ying-Ting Yeh, Janghoon Ock, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2501.03456)  

**Abstract**: In this study, we explore the use of a transformer-based language model as an encoder to predict the band gaps of semiconductor materials directly from their text descriptions. Quantum chemistry simulations, including Density Functional Theory (DFT), are computationally intensive and time-consuming, which limits their practicality for high-throughput material screening, particularly for complex systems. Shallow machine learning (ML) models, while effective, often require extensive data preprocessing to convert non-numerical material properties into numerical inputs. In contrast, our approach leverages textual data directly, bypassing the need for complex feature engineering. We generate material descriptions in two formats: formatted strings combining features and natural language text generated using the ChatGPT API. We demonstrate that the RoBERTa model, pre-trained on natural language processing tasks, performs effectively as an encoder for prediction tasks. With minimal fine-tuning, it achieves a mean absolute error (MAE) of approximately 0.33 eV, performing better than shallow machine learning models such as Support Vector Regression, Random Forest, and XGBoost. Even when only the linear regression head is trained while keeping the RoBERTa encoder layers frozen, the accuracy remains nearly identical to that of the fully trained model. This demonstrates that the pre-trained RoBERTa encoder is highly adaptable for processing domain-specific text related to material properties, such as the band gap, significantly reducing the need for extensive retraining. This study highlights the potential of transformer-based language models to serve as efficient and versatile encoders for semiconductor materials property prediction tasks. 

**Abstract (ZH)**: 在本研究中，我们探讨了使用基于变压器的语言模型作为编码器，直接从半导体材料的文本描述预测其能隙的方法。量子化学模拟，包括密度泛函理论（DFT），计算密集且耗时，这限制了其在高通量材料筛选中的实用性，尤其是对于复杂系统而言。浅层机器学习（ML）模型虽然有效，但通常需要大量数据预处理，以将非数值材料特性转换为数值输入。相比之下，我们的方法直接利用文本数据，避免了复杂特征工程的需要。我们生成了两种格式的材料描述：一种是结合特征的格式化字符串，另一种是使用ChatGPT API生成的自然语言文本。研究表明，预训练于自然语言处理任务的RoBERTa模型在预测任务中表现有效。通过最少的微调，它实现了约0.33 eV的平均绝对误差（MAE），优于浅层机器学习模型，如支持向量回归、随机森林和XGBoost。即使只训练线性回归头而冻结RoBERTa编码器层，其准确性仍与完全训练的模型相当。这表明预训练的RoBERTa编码器对处理与材料性质相关的领域特定文本（如能隙）具有很高的适应性，显著减少了重新训练的需求。本研究突显了基于变压器的语言模型作为半导体材料性质预测任务中高效且多功能编码器的潜在价值。 

---
# Finding A Voice: Evaluating African American Dialect Generation for Chatbot Technology 

**Title (ZH)**: 寻找声音：评估非裔美国人口音生成技术在聊天机器人中的应用 

**Authors**: Sarah E. Finch, Ellie S. Paek, Sejung Kwon, Ikseon Choi, Jessica Wells, Rasheeta Chandler, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2501.03441)  

**Abstract**: As chatbots become increasingly integrated into everyday tasks, designing systems that accommodate diverse user populations is crucial for fostering trust, engagement, and inclusivity. This study investigates the ability of contemporary Large Language Models (LLMs) to generate African American Vernacular English (AAVE) and evaluates the impact of AAVE usage on user experiences in chatbot applications. We analyze the performance of three LLM families (Llama, GPT, and Claude) in producing AAVE-like utterances at varying dialect intensities and assess user preferences across multiple domains, including healthcare and education. Despite LLMs' proficiency in generating AAVE-like language, findings indicate that AAVE-speaking users prefer Standard American English (SAE) chatbots, with higher levels of AAVE correlating with lower ratings for a variety of characteristics, including chatbot trustworthiness and role appropriateness. These results highlight the complexities of creating inclusive AI systems and underscore the need for further exploration of diversity to enhance human-computer interactions. 

**Abstract (ZH)**: 随着聊天机器人的日常任务集成程度不断提高，设计能够适应多样用户群体的系统对于培养信任、增强参与度和促进包容性至关重要。本研究探讨了当前大型语言模型（LLMs）生成非标准英语（如非洲裔美国人方言英语AAVE）的能力，并评估AAVE在聊天机器人应用中的使用对其用户体验的影响。我们分析了三种LLM家族（Llama、GPT和Claude）在不同方言强度下生成AAVE样态表达的表现，并评估了用户在医疗保健和教育等多个领域的偏好。尽管LLMs在生成AAVE样态语言方面表现出了较高的能力，但研究结果表明，AAVE使用者更偏好标准美国英语(SAE)聊天机器人，且方言的使用程度越高，聊天机器人的可信度和角色适宜性等各方面评价越低。这些结果突显了创建包容性AI系统的复杂性，并强调了进一步探究多样性以增强人机互动的重要性。 

---
# DAMAGE: Detecting Adversarially Modified AI Generated Text 

**Title (ZH)**: DAMAGE: 检测对抗性修改的AI生成文本 

**Authors**: Elyas Masrour, Bradley Emi, Max Spero  

**Link**: [PDF](https://arxiv.org/pdf/2501.03437)  

**Abstract**: AI humanizers are a new class of online software tools meant to paraphrase and rewrite AI-generated text in a way that allows them to evade AI detection software. We study 19 AI humanizer and paraphrasing tools and qualitatively assess their effects and faithfulness in preserving the meaning of the original text. We show that many existing AI detectors fail to detect humanized text. Finally, we demonstrate a robust model that can detect humanized AI text while maintaining a low false positive rate using a data-centric augmentation approach. We attack our own detector, training our own fine-tuned model optimized against our detector's predictions, and show that our detector's cross-humanizer generalization is sufficient to remain robust to this attack. 

**Abstract (ZH)**: AI人性化工具是一类新的在线软件工具，旨在以逃避AI检测软件的方式重新表述和重写AI生成的文本。我们研究了19种AI人性化和改写工具，并对其在保留原始文本意义方面的效果和忠实度进行了定性评估。我们展示了许多现有的AI检测器无法识别人性化文本。最后，我们提出了一种稳健的模型，能够在保持低误报率的情况下检测到人性化处理的AI文本，采用以数据为中心的增强方法。我们攻击了我们自己的检测器，训练了一款针对我们检测器预测进行优化的微调模型，并证明了我们的检测器在面对这种攻击时具有鲁棒性，表明它在跨AI人性化工具的一般化能力是足够的。 

---
# BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations 

**Title (ZH)**: BoundingDocs：一种带有空间注释的文档问答统一数据集 

**Authors**: Simone Giovannini, Fabio Coppini, Andrea Gemelli, Simone Marinai  

**Link**: [PDF](https://arxiv.org/pdf/2501.03403)  

**Abstract**: We present a unified dataset for document Question-Answering (QA), which is obtained combining several public datasets related to Document AI and visually rich document understanding (VRDU). Our main contribution is twofold: on the one hand we reformulate existing Document AI tasks, such as Information Extraction (IE), into a Question-Answering task, making it a suitable resource for training and evaluating Large Language Models; on the other hand, we release the OCR of all the documents and include the exact position of the answer to be found in the document image as a bounding box. Using this dataset, we explore the impact of different prompting techniques (that might include bounding box information) on the performance of open-weight models, identifying the most effective approaches for document comprehension. 

**Abstract (ZH)**: 我们提供了一个统一的文档问答（Document QA）数据集，该数据集通过结合多个与文档AI和视觉丰富文档理解（VRDU）相关的公开数据集而构建。我们的主要贡献包含两个方面：一方面，我们将现有的文档AI任务，如信息抽取（IE），重新表述为问答任务，使其成为训练和评估大型语言模型的一个合适资源；另一方面，我们发布了所有文档的光学字符识别（OCR）结果，并在文档图像中标注了答案的确切位置，表示为边界框。利用此数据集，我们探讨了不同提示技术（可能包括边界框信息）对开放式重量模型性能的影响，确定了最有效的文档理解方法。 

---
# Advanced Machine Learning Techniques for Social Support Detection on Social Media 

**Title (ZH)**: 社交媒体上社交支持检测的先进机器学习技术 

**Authors**: Olga Kolesnikova, Moein Shahiki Tash, Zahra Ahani, Ameeta Agrawal, Raul Monroy, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2501.03370)  

**Abstract**: The widespread use of social media highlights the need to understand its impact, particularly the role of online social support. This study uses a dataset focused on online social support, which includes binary and multiclass classifications of social support content on social media. The classification of social support is divided into three tasks. The first task focuses on distinguishing between supportive and non-supportive. The second task aims to identify whether the support is directed toward an individual or a group. The third task categorizes the specific type of social support, grouping it into categories such as Nation, LGBTQ, Black people, Women, Religion, and Other (if it does not fit into the previously mentioned categories). To address data imbalances in these tasks, we employed K-means clustering for balancing the dataset and compared the results with the original unbalanced data. Using advanced machine learning techniques, including transformers and zero-shot learning approaches with GPT3, GPT4, and GPT4-o, we predict social support levels in various contexts. The effectiveness of the dataset is evaluated using baseline models across different learning approaches, with transformer-based methods demonstrating superior performance. Additionally, we achieved a 0.4\% increase in the macro F1 score for the second task and a 0.7\% increase for the third task, compared to previous work utilizing traditional machine learning with psycholinguistic and unigram-based TF-IDF values. 

**Abstract (ZH)**: 社交媒體的廣泛使用突顯了理解其影響的必要性，特別是線上 SOCIAL SUPPORT 的作用。本研究使用了一個專注於線上 SOCIAL SUPPORT 的資料集，該資料集包括社交媒體上支持內容的二元和多類別分類。 SOCIAL SUPPORT 的分類被分為三個任務。第一個任務著重於區分支持與非支持。第二個任務旨在判別支持是針對個人還是群體。第三個任務則根據具體的社交支持類型進行分類，包括國家、LGBTQ、黑人、女性、宗教和其它（不符合之前提到的类别）。為了應對這些任務中的數據不平衡，我們採用了 K-means 聚類以平衡數據集，並將結果與原始未平衡數據進行比較。我們使用了包括轉換器和零-shot 學習方法（GPT3、GPT4 和 GPT4-o）等先進的機器學習技術，以預測不同情境下的社交支持水平。通過在不同學習方法下使用基線模型評估數據集的效果，轉換器基方法展示了更高的性能。此外，我们在第二個任務中实现了0.4%的宏F1分數提升，在第三个任务中实现了0.7%的提升，相较于之前使用傳統機器學習和心理語言學以及單詞基的TF-IDF值的研究工作。 

---
# Analyzing Bias in Swiss Federal Supreme Court Judgments Using Facebook's Holistic Bias Dataset: Implications for Language Model Training 

**Title (ZH)**: 使用Facebook的综合偏见数据集分析瑞士联邦最高法院判决中的偏见：对语言模型训练的 implications 

**Authors**: Sabine Wehnert, Muhammet Ertas, Ernesto William De Luca  

**Link**: [PDF](https://arxiv.org/pdf/2501.03324)  

**Abstract**: Natural Language Processing (NLP) is vital for computers to process and respond accurately to human language. However, biases in training data can introduce unfairness, especially in predicting legal judgment. This study focuses on analyzing biases within the Swiss Judgment Prediction Dataset (SJP-Dataset). Our aim is to ensure unbiased factual descriptions essential for fair decision making by NLP models in legal contexts. We analyze the dataset using social bias descriptors from the Holistic Bias dataset and employ advanced NLP techniques, including attention visualization, to explore the impact of dispreferred descriptors on model predictions. The study identifies biases and examines their influence on model behavior. Challenges include dataset imbalance and token limits affecting model performance. 

**Abstract (ZH)**: 自然语言处理（NLP）对于计算机准确处理和回应人类语言至关重要。然而，训练数据中的偏见可能会引入不公平性，特别是在预测法律判决方面尤为明显。本研究旨在分析瑞士判决预测数据集（SJP-数据集）中的偏见，确保NLP模型在法律领域进行公正决策时所依赖的无偏客观描述。我们使用了综合偏见数据集中的社会偏见描述符，结合高级NLP技术（如注意力可视化），探索不受欢迎描述符对模型预测的影响。本研究识别了偏见并考察了它们对模型行为的影响。研究中面临的挑战包括数据集不平衡以及词汇量限制对模型性能的影响。 

---
# ADePT: Adaptive Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning 

**Title (ZH)**: ADePT：自适应分解提示调优方法，用于参数高效微调 

**Authors**: Pengwei Tang, Xiaolin Hu, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.03291)  

**Abstract**: Prompt Tuning (PT) enables the adaptation of Pre-trained Large Language Models (PLMs) to downstream tasks by optimizing a small amount of soft virtual tokens, which are prepended to the input token embeddings. Recently, Decomposed Prompt Tuning (DePT) has demonstrated superior adaptation capabilities by decomposing the soft prompt into a shorter soft prompt and a pair of low-rank matrices. The product of the pair of low-rank matrices is added to the input token embeddings to offset them. Additionally, DePT achieves faster inference compared to PT due to the shorter soft prompt. However, in this paper, we find that the position-based token embedding offsets of DePT restricts its ability to generalize across diverse model inputs, and that the shared embedding offsets across many token embeddings result in sub-optimization. To tackle these issues, we introduce \textbf{A}daptive \textbf{De}composed \textbf{P}rompt \textbf{T}uning (ADePT), which is composed of a short soft prompt and a shallow token-shared feed-forward neural network. ADePT utilizes the token-shared feed-forward neural network to learn the embedding offsets for each token, enabling adaptive embedding offsets that vary according to the model input and better optimization of token embedding offsets. This enables ADePT to achieve superior adaptation performance without requiring more inference time or additional trainable parameters compared to vanilla PT and its variants. In comprehensive experiments across 23 natural language processing (NLP) tasks and 4 typical PLMs of different scales, we show that ADePT consistently surpasses the leading parameter-efficient fine-tuning (PEFT) methods, and even outperforms the full fine-tuning baseline in certain scenarios. Code is available at \url{this https URL}. 

**Abstract (ZH)**: 提示调整（PT）通过优化一小部分软虚拟令牌来使预训练大型语言模型（PLMs）适应下游任务，这些软令牌被附加到输入词元嵌入之前。最近，分解提示调整（DePT）通过将软提示分解为较短的软提示和一对低秩矩阵，展示了更出色的适应能力。这两对低秩矩阵的乘积被添加到输入词元嵌入中以抵消它们。此外，由于软提示较短，DePT 较 PT 实现了更快的推理速度。然而，在本文中，我们发现 DePT 的基于位置的词元嵌入偏移限制了其在不同模型输入上的泛化能力，并且众多词元嵌入共享的嵌入偏移导致了次优优化。为了解决这些问题，我们提出了**可**适应的**分**解的**提**示**调**整（ADePT），它由一个短的软提示和一个浅层的共享词元前馈神经网络组成。ADePT 利用共享词元前馈神经网络为每个词元学习嵌入偏移，从而实现根据模型输入变化的适应性嵌入偏移，并优化词元嵌入偏移。这使得 ADePT 在不需要比 vanilla PT 及其变体更多的推理时间和额外的可训练参数的情况下，就能实现更出色的适应性能。我们通过对 23 个自然语言处理（NLP）任务和 4 个不同规模的典型预训练模型（PLMs）进行全面实验，在这些任务中，ADePT 一致超越了领先的小参数效率微调（PEFT）方法，并在某些场景中甚至超越了全量微调基线。完整代码可在<URL>获取。 

---
# HonkaiChat: Companions from Anime that feel alive! 

**Title (ZH)**: Honkai Chat：充满生命力的动画同伴 

**Authors**: Yueze Liu, Yichi Zhang, Shaan Om Patel, Zhaoyang Zhu, Shilong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.03277)  

**Abstract**: Modern conversational agents, including anime-themed chatbots, are frequently reactive and personality-driven but fail to capture the dynamic nature of human interactions. We propose an event-driven dialogue framework to address these limitations by embedding dynamic events in conversation prompts and fine-tuning models on character-specific data. Evaluations on GPT-4 and comparisons with industry-leading baselines demonstrate that event-driven prompts significantly improve conversational engagement and naturalness while reducing hallucinations. This paper explores the application of this approach in creating lifelike chatbot interactions within the context of Honkai: Star Rail, showcasing the potential for dynamic event-based systems to transform role-playing and interactive dialogue. 

**Abstract (ZH)**: 现代对话代理，包括以动漫为主题的聊天机器人，往往具有反应性和个性驱动的特点，但未能捕捉人类互动的动态性。本文提出了一种事件驱动的对话框架来解决这些局限性，通过在对话提示中嵌入动态事件并基于角色特定的数据微调模型。对GPT-4的评估和与行业领先基准的比较表明，事件驱动的提示显著提高了对话的参与度和自然性，同时减少了幻觉。本文探讨了该方法在《 heavensfall：星穹铁道》这一背景下创建逼真聊天机器人交互的应用，展示了基于动态事件系统的潜力，以变革角色扮演和互动对话。 

---
# ComMer: a Framework for Compressing and Merging User Data for Personalization 

**Title (ZH)**: ComMer：一种用于个性化服务的用户数据压缩与合并框架 

**Authors**: Yoel Zeldes, Amir Zait, Ilia Labzovsky, Danny Karmon, Efrat Farkash  

**Link**: [PDF](https://arxiv.org/pdf/2501.03276)  

**Abstract**: Large Language Models (LLMs) excel at a wide range of tasks, but adapting them to new data, particularly for personalized applications, poses significant challenges due to resource and computational constraints. Existing methods either rely on exposing fresh data to the model through the prompt, which is limited by context size and computationally expensive at inference time, or fine-tuning, which incurs substantial training and update costs. In this paper, we introduce ComMer - Compress and Merge - a novel framework that efficiently personalizes LLMs by compressing users' documents into compact representations, which are then merged and fed into a frozen LLM. We evaluate ComMer on two types of personalization tasks - personalized skill learning, using the tweet paraphrasing dataset and the personalized news headline generation dataset from the LaMP benchmark, and knowledge-intensive, using the PerLTQA dataset. Our experiments demonstrate that in constrained inference budget scenarios ComMer achieves superior quality in skill learning tasks, while highlighting limitations in knowledge-intensive settings due to the loss of detailed information. These results offer insights into trade-offs and potential optimizations in multi-document compression for personalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的任务上表现出色，但将它们适应新数据，特别是针对个性化应用时，由于资源和计算限制，面临重大挑战。现有方法要么依赖于通过提示向模型暴露新鲜数据，这种方法受上下文大小限制，并且在推理时计算成本高昂；要么进行微调，这会带来大量的训练和更新成本。在本文中，我们引入了ComMer（压缩和合并）框架，该框架通过将用户文档压缩为紧凑表示，然后合并送入冻结的LLM，来高效地个性化LLMs。我们对ComMer进行了两项个性化任务的评估：一是个人技能学习任务，使用推文重写数据集和LaMP基准中的个性化新闻标题生成数据集；二是知识密集型任务，使用PerLTQA数据集。实验结果表明，在受限的推理预算场景中，ComMer在技能学习任务中能取得更好的质量，但在知识密集型场景中由于丢失了详细信息而表现出局限性。这些结果为我们理解多文档压缩在个性化中的权衡和潜在优化提供了见解。 

---
# LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena 

**Title (ZH)**: 大型语言模型内容审核与用户满意度：来自聊天机器人领域的证据 

**Authors**: Stefan Pasch  

**Link**: [PDF](https://arxiv.org/pdf/2501.03266)  

**Abstract**: LLM safety and ethical alignment are widely discussed, but the impact of content moderation on user satisfaction remains underexplored. To address this, we analyze nearly 50,000 Chatbot Arena response-pairs using a novel fine-tuned RoBERTa model, that we trained on hand-labeled data to disentangle refusals due to ethical concerns from other refusals due to technical disabilities or lack of information. Our findings reveal a significant refusal penalty on content moderation, with users choosing ethical-based refusals roughly one-fourth as often as their preferred LLM response compared to standard responses. However, the context and phrasing play critical roles: refusals on highly sensitive prompts, such as illegal content, achieve higher win rates than less sensitive ethical concerns, and longer responses closely aligned with the prompt perform better. These results emphasize the need for nuanced moderation strategies that balance ethical safeguards with user satisfaction. Moreover, we find that the refusal penalty is notably lower in evaluations using the LLM-as-a-Judge method, highlighting discrepancies between user and automated assessments. 

**Abstract (ZH)**: LLM的安全性和伦理对齐受到广泛讨论，但内容审核对用户满意度的影响尚未充分探讨。为了解决这一问题，我们使用一种新型微调RoBERTa模型分析了近50,000个Chatbot Arena的响应对，该模型是基于手工标注的数据训练而成，以区分由于伦理关切而拒绝的情况与其他由于技术问题或信息不足而拒绝的情况。我们的研究发现，内容审核显著降低了用户满意度，用户选择基于伦理关切的拒绝的比例大约仅为他们首选的LLM响应的一半。然而，上下文和措辞起着关键作用：对于高度敏感的提示（如非法内容），基于伦理关切的拒绝获得了更高的成功率；而对于较不敏感的伦理关切，则表现较差。此外，更长且与提示高度一致的回应表现更好。这些结果强调了需要一种平衡伦理保护与用户满意度的细腻管理策略。另外，我们发现，在使用LLM作为法官的评估方法中，拒绝惩罚明显较低，这揭示了用户评估与自动化评估之间的差异。 

---
# REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models 

**Title (ZH)**: REINFORCE++：一种简单有效的大型语言模型对齐方法 

**Authors**: Jian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.03262)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has emerged as a critical approach for aligning large language models with human preferences, witnessing rapid algorithmic evolution through methods such as Proximal Policy Optimization (PPO), Direct Preference Optimization (DPO), REINFORCE Leave One-Out (RLOO), ReMax, and Group Relative Policy Optimization (GRPO). We present REINFORCE++, an enhanced variant of the classical REINFORCE algorithm that incorporates key optimization techniques from PPO while eliminating the need for a critic network. REINFORCE++ achieves three primary objectives: (1) simplicity (2) enhanced training stability, and (3) reduced computational overhead. Through extensive empirical evaluation, we demonstrate that REINFORCE++ exhibits superior stability compared to GRPO and achieves greater computational efficiency than PPO while maintaining comparable performance. The implementation is available at \url{this https URL}. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）已成为使大规模语言模型与人类偏好对齐的关键方法，通过使用诸如近端策略优化（PPO）、直接偏好优化（DPO）、REINFORCE 留一法（RLOO）、ReMax 及组相对策略优化（GRPO）等方法，观察到了快速的算法演进。我们提出了一种增强的 REINFORCE 变体算法——REINFORCE++，该算法结合了 PPO 的关键优化技术，同时消除了对批评网络的需求。REINFORCE++ 实现了以下三个主要目标：（1）简洁性；（2）增强的训练稳定性；（3）减少的计算开销。通过广泛的实证评估，我们证明了 REINFORCE++ 相较于 GRPO 具有更高的稳定性，并且在保持与 PPO 相当的性能的同时，比 PPO 具有更高的计算效率。该实现可以在 \url{this https URL} 获取。 

---
# Toward Inclusive Educational AI: Auditing Frontier LLMs through a Multiplexity Lens 

**Title (ZH)**: 面向包容性教育人工智能：通过多元性视角审计前沿大语言模型 

**Authors**: Abdullah Mushtaq, Muhammad Rafay Naeem, Muhammad Imran Taj, Ibrahim Ghaznavi, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2501.03259)  

**Abstract**: As large language models (LLMs) like GPT-4 and Llama 3 become integral to educational contexts, concerns are mounting over the cultural biases, power imbalances, and ethical limitations embedded within these technologies. Though generative AI tools aim to enhance learning experiences, they often reflect values rooted in Western, Educated, Industrialized, Rich, and Democratic (WEIRD) cultural paradigms, potentially sidelining diverse global perspectives. This paper proposes a framework to assess and mitigate cultural bias within LLMs through the lens of applied multiplexity. Multiplexity, inspired by Senturk et al. and rooted in Islamic and other wisdom traditions, emphasizes the coexistence of diverse cultural viewpoints, supporting a multi-layered epistemology that integrates both empirical sciences and normative values. Our analysis reveals that LLMs frequently exhibit cultural polarization, with biases appearing in both overt responses and subtle contextual cues. To address inherent biases and incorporate multiplexity in LLMs, we propose two strategies: \textit{Contextually-Implemented Multiplex LLMs}, which embed multiplex principles directly into the system prompt, influencing LLM outputs at a foundational level and independent of individual prompts, and \textit{Multi-Agent System (MAS)-Implemented Multiplex LLMs}, where multiple LLM agents, each representing distinct cultural viewpoints, collaboratively generate a balanced, synthesized response. Our findings demonstrate that as mitigation strategies evolve from contextual prompting to MAS-implementation, cultural inclusivity markedly improves, evidenced by a significant rise in the Perspectives Distribution Score (PDS) and a PDS Entropy increase from 3.25\% at baseline to 98\% with the MAS-Implemented Multiplex LLMs. Sentiment analysis further shows a shift towards positive sentiment across cultures,... 

**Abstract (ZH)**: 随着大型语言模型（LLMs）如GPT-4和Llama 3在教育场景中日益重要，人们对这些技术中存在的文化偏见、权力不均和伦理限制的担忧不断增加。尽管生成式AI工具旨在提升学习体验，但它们通常反映了根植于西方、受过教育、工业化、富裕和民主（WEIRD）文化模式的价值观，可能忽视了全球多样化的视角。本文提出了一种框架，通过应用多重性的视角评估和减轻LLMs中的文化偏见。多重性理念受到Senturk等人以及伊斯兰和其他智慧传统的启发，强调不同文化观点的共存，支持一种多层次的知识论，将实证科学与规范性价值相整合。我们的分析显示，LLMs经常表现出文化极化现象，偏见既体现在明示的回答中，也体现在微妙的上下文提示中。为解决固有的偏见并将在LLMs中引入多重性，我们提出了两种策略：**上下文实施的多重LLMs**，其通过直接将多重性原则嵌入系统提示，从根本上影响LLM的输出，不依赖于个别提示；以及**多层次系统（MAS）实施的多重LLMs**，其中多个LLM代理，各自代表不同的文化视角，协同生成平衡且综合的回应。我们的研究发现，随着缓解策略从上下文提示演进到MAS实现，文化包容性显著提高，这在视见分布得分（PDS）显著上升以及PDS熵从基线的3.25%增加到MAS实施的多重LLMs的98%中得到了体现。进一步的情感分析显示，文化间整体情绪有向积极转变的趋势，…… 

---
# PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides 

**Title (ZH)**: PPTAgent：超越文本生成幻灯片的演示文稿生成与评估 

**Authors**: Hao Zheng, Xinyan Guan, Hao Kong, Jia Zheng, Hongyu Lin, Yaojie Lu, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.03936)  

**Abstract**: Automatically generating presentations from documents is a challenging task that requires balancing content quality, visual design, and structural coherence. Existing methods primarily focus on improving and evaluating the content quality in isolation, often overlooking visual design and structural coherence, which limits their practical applicability. To address these limitations, we propose PPTAgent, which comprehensively improves presentation generation through a two-stage, edit-based approach inspired by human workflows. PPTAgent first analyzes reference presentations to understand their structural patterns and content schemas, then drafts outlines and generates slides through code actions to ensure consistency and alignment. To comprehensively evaluate the quality of generated presentations, we further introduce PPTEval, an evaluation framework that assesses presentations across three dimensions: Content, Design, and Coherence. Experiments show that PPTAgent significantly outperforms traditional automatic presentation generation methods across all three dimensions. The code and data are available at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

从文档自动生成演示文稿是一个具有挑战性的任务，需要平衡内容质量、视觉设计和结构一致性。现有方法主要集中在独立地改进和完善内容质量，往往忽略了视觉设计和结构一致性，这限制了它们的实际应用。为了解决这些局限性，我们提出了一种名为PPTAgent的系统，该系统通过借鉴人类工作流程的两阶段编辑方法，全面提高了演示文稿生成的质量。PPTAgent首先分析参考演示文稿以理解其结构模式和内容架构，然后通过代码操作草拟提纲并生成幻灯片，以确保一致性和对齐。为了全面评估生成的演示文稿质量，我们进一步引入了PPTEval评估框架，该框架从内容、设计和一致性三个维度评估演示文稿。实验结果显示，PPTAgent在所有三个维度上显著优于传统的自动演示文稿生成方法。代码和数据可在以下链接获取：this https URL。 

---
# From Newswire to Nexus: Using text-based actor embeddings and transformer networks to forecast conflict dynamics 

**Title (ZH)**: 从新闻通讯到网络：利用基于文本的行动者嵌入和变换器网络预测冲突动态 

**Authors**: Mihai Croicu, Simon Polichinel von der Maase  

**Link**: [PDF](https://arxiv.org/pdf/2501.03928)  

**Abstract**: This study advances the field of conflict forecasting by using text-based actor embeddings with transformer models to predict dynamic changes in violent conflict patterns at the actor level. More specifically, we combine newswire texts with structured conflict event data and leverage recent advances in Natural Language Processing (NLP) techniques to forecast escalations and de-escalations among conflicting actors, such as governments, militias, separatist movements, and terrorists. This new approach accurately and promptly captures the inherently volatile patterns of violent conflicts, which existing methods have not been able to achieve. To create this framework, we began by curating and annotating a vast international newswire corpus, leveraging hand-labeled event data from the Uppsala Conflict Data Program. By using this hybrid dataset, our models can incorporate the textual context of news sources along with the precision and detail of structured event data. This combination enables us to make both dynamic and granular predictions about conflict developments. We validate our approach through rigorous back-testing against historical events, demonstrating superior out-of-sample predictive power. We find that our approach is quite effective in identifying and predicting phases of conflict escalation and de-escalation, surpassing the capabilities of traditional models. By focusing on actor interactions, our explicit goal is to provide actionable insights to policymakers, humanitarian organizations, and peacekeeping operations in order to enable targeted and effective intervention strategies. 

**Abstract (ZH)**: 本研究通过使用基于文本的行动者嵌入和变换器模型，推动了冲突预测领域的进展，旨在预测暴力冲突模式在行动者层面的动态变化。具体而言，我们结合新闻稿文本和结构化的冲突事件数据，利用自然语言处理（NLP）技术的最新进展，预测冲突各方（如政府、民兵、分离主义运动和恐怖组织）之间的升级和降级。这种新的方法能够准确且及时捕捉暴力冲突固有的易变模式，这是现有方法所无法实现的。为了构建这一框架，我们首先整理和标注了广泛的国际新闻语料库，并借鉴乌普萨拉冲突数据项目的手动标注事件数据。通过使用这种混合数据集，我们的模型可以结合新闻来源的文本背景和结构化事件数据的高度精确和细节。这种结合使我们能够进行动态和精细的冲突发展预测。我们通过严格的回溯测试，将我们的方法与历史事件进行对比验证，展示了优越的外样本预测能力。我们发现，我们的方法在识别和预测冲突升级和降级的阶段方面非常有效，超过了传统模型的能力。通过对行动者互动的关注，我们的明确目标是为政策制定者、人道主义组织和维和行动提供可操作的见解，以便制定针对性和有效的干预策略。 

---
# Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback 

**Title (ZH)**: Dolphin：通过思考、实践和反馈实现的闭环开放式自主研究 

**Authors**: Jiakang Yuan, Xiangchao Yan, Botian Shi, Tao Chen, Wanli Ouyang, Bo Zhang, Lei Bai, Yu Qiao, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.03916)  

**Abstract**: The scientific research paradigm is undergoing a profound transformation owing to the development of Artificial Intelligence (AI). Recent works demonstrate that various AI-assisted research methods can largely improve research efficiency by improving data analysis, accelerating computation, and fostering novel idea generation. To further move towards the ultimate goal (i.e., automatic scientific research), in this paper, we propose Dolphin, the first closed-loop open-ended auto-research framework to further build the entire process of human scientific research. Dolphin can generate research ideas, perform experiments, and get feedback from experimental results to generate higher-quality ideas. More specifically, Dolphin first generates novel ideas based on relevant papers which are ranked by the topic and task attributes. Then, the codes are automatically generated and debugged with the exception-traceback-guided local code structure. Finally, Dolphin automatically analyzes the results of each idea and feeds the results back to the next round of idea generation. Experiments are conducted on the benchmark datasets of different topics and results show that Dolphin can generate novel ideas continuously and complete the experiment in a loop. We highlight that Dolphin can automatically propose methods that are comparable to the state-of-the-art in some tasks such as 2D image classification and 3D point classification. 

**Abstract (ZH)**: 由于人工智能（AI）的发展，科学研究范式正在经历一场深刻的变化。近期的研究表明，各种AI辅助的研究方法可以通过改善数据处理、加速计算和激发新的想法，大幅提高研究效率。为进一步实现终极目标（即自动科学研究），本文提出了一种名为Dolphin的闭环开放框架，以进一步构建整个人类科学研究过程。Dolphin能够生成研究思路、执行实验，并根据实验结果获取反馈以生成更高质量的思路。更具体地来说，Dolphin首先根据主题和任务属性对相关论文进行排序后生成新颖的想法。然后，通过异常跟踪引导的局部代码结构自动生成和调试代码。最后，Dolphin自动分析每个想法的结果，并将结果反馈到下一轮想法生成过程。实验是在不同主题的基准数据集上进行的，结果显示Dolphin可以连续生成新颖的想法并在循环中完成实验。我们强调，Dolphin能够自动提出在一些任务中（如2D图像分类和3D点分类）与最先进的方法相媲美的方法。 

---
# LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token 

**Title (ZH)**: LLaVA-Mini：高效的一视图 token 大规模多模态模型 

**Authors**: Shaolei Zhang, Qingkai Fang, Zhe Yang, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.03895)  

**Abstract**: The advent of real-time large multimodal models (LMMs) like GPT-4o has sparked considerable interest in efficient LMMs. LMM frameworks typically encode visual inputs into vision tokens (continuous representations) and integrate them and textual instructions into the context of large language models (LLMs), where large-scale parameters and numerous context tokens (predominantly vision tokens) result in substantial computational overhead. Previous efforts towards efficient LMMs always focus on replacing the LLM backbone with smaller models, while neglecting the crucial issue of token quantity. In this paper, we introduce LLaVA-Mini, an efficient LMM with minimal vision tokens. To achieve a high compression ratio of vision tokens while preserving visual information, we first analyze how LMMs understand vision tokens and find that most vision tokens only play a crucial role in the early layers of LLM backbone, where they mainly fuse visual information into text tokens. Building on this finding, LLaVA-Mini introduces modality pre-fusion to fuse visual information into text tokens in advance, thereby facilitating the extreme compression of vision tokens fed to LLM backbone into one token. LLaVA-Mini is a unified large multimodal model that can support the understanding of images, high-resolution images, and videos in an efficient manner. Experiments across 11 image-based and 7 video-based benchmarks demonstrate that LLaVA-Mini outperforms LLaVA-v1.5 with just 1 vision token instead of 576. Efficiency analyses reveal that LLaVA-Mini can reduce FLOPs by 77%, deliver low-latency responses within 40 milliseconds, and process over 10,000 frames of video on the GPU hardware with 24GB of memory. 

**Abstract (ZH)**: 实时大型多模态模型（LMMs）如GPT-4o的出现引发了对高效LMMs的广泛关注。LMM框架通常将视觉输入编码为视觉令牌（连续表示），并将这些视觉令牌与文本指令整合到大规模语言模型（LLM）的上下文中，大规模参数和大量上下文令牌（主要为视觉令牌）导致了显著的计算开销。以往针对高效LMMs的努力主要集中在用较小的模型替换LLM骨干，而忽视了令牌数量这一关键问题。本文中，我们介绍了LLaVA-Mini，这是一种具有最少视觉令牌的高效LMM。为了实现视觉令牌的高度压缩同时保留视觉信息，我们首先分析了LMMs如何理解视觉令牌，发现大多数视觉令牌主要在LLM骨干的早期层中起关键作用，在这些层中，它们主要将视觉信息融合到文本令牌中。基于这一发现，LLaVA-Mini 引入了模态预融合技术，在先进阶段将视觉信息融合到文本令牌中，从而使得输入到LLM骨干的视觉令牌极端压缩为一个令牌。LLaVA-Mini 是一种统一的大型多模态模型，能够以高效的方式支持对图像、高分辨率图像和视频的理解。在11个基于图像和7个基于视频的基准测试中，实验结果表明，LLaVA-Mini的表现优于LLaVA-v1.5，仅使用1个视觉令牌而非576个。效率分析表明，LLaVA-Mini 可以将FLOPs 减少77%，在40毫秒内提供低延迟响应，并在配备24GB内存的GPU硬件上处理超过10,000帧视频。 

---
# BERTopic for Topic Modeling of Hindi Short Texts: A Comparative Study 

**Title (ZH)**: 基于BERTopic的印地语短文本主题建模：一种比较研究 

**Authors**: Atharva Mutsaddi, Anvi Jamkhande, Aryan Thakre, Yashodhara Haribhakta  

**Link**: [PDF](https://arxiv.org/pdf/2501.03843)  

**Abstract**: As short text data in native languages like Hindi increasingly appear in modern media, robust methods for topic modeling on such data have gained importance. This study investigates the performance of BERTopic in modeling Hindi short texts, an area that has been under-explored in existing research. Using contextual embeddings, BERTopic can capture semantic relationships in data, making it potentially more effective than traditional models, especially for short and diverse texts. We evaluate BERTopic using 6 different document embedding models and compare its performance against 8 established topic modeling techniques, such as Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), Latent Semantic Indexing (LSI), Additive Regularization of Topic Models (ARTM), Probabilistic Latent Semantic Analysis (PLSA), Embedded Topic Model (ETM), Combined Topic Model (CTM), and Top2Vec. The models are assessed using coherence scores across a range of topic counts. Our results reveal that BERTopic consistently outperforms other models in capturing coherent topics from short Hindi texts. 

**Abstract (ZH)**: 随着如印地语这样的母语短文本数据在现代媒体中越来越多地出现，针对此类数据的主题建模方法变得尤为重要。本研究调查了 BERTopic 在建模印地语短文本方面的性能，这一领域在现有研究中尚未得到充分探索。利用上下文嵌入，BERTopic 可以捕获数据中的语义关系，使其在传统模型尤其是短而多样的文本方面更具有效性。我们使用6种不同的文档嵌入模型评估了 BERTopic，并将其性能与8种已建立的主题建模技术进行了比较，包括隐含狄利克雷分配（LDA）、非负矩阵分解（NMF）、隐含语义索引（LSI）、主题模型正则化加权（ARTM）、概率隐含语义分析（PLSA）、嵌入主题模型（ETM）、组合主题模型（CTM）和Top2Vec。通过不同主题数量范围内的连贯性评分来评估这些模型。我们的结果显示，BERTopic 在捕捉印地语短文本中的连贯主题方面始终优于其他模型。 

---
# Detecting the Undetectable: Assessing the Efficacy of Current Spoof Detection Methods Against Seamless Speech Edits 

**Title (ZH)**: 检测不可检测的：评估当前欺骗检测方法对无缝语音编辑的有效性 

**Authors**: Sung-Feng Huang, Heng-Cheng Kuo, Zhehuai Chen, Xuesong Yang, Chao-Han Huck Yang, Yu Tsao, Yu-Chiang Frank Wang, Hung-yi Lee, Szu-Wei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2501.03805)  

**Abstract**: Neural speech editing advancements have raised concerns about their misuse in spoofing attacks. Traditional partially edited speech corpora primarily focus on cut-and-paste edits, which, while maintaining speaker consistency, often introduce detectable discontinuities. Recent methods, like A\textsuperscript{3}T and Voicebox, improve transitions by leveraging contextual information. To foster spoofing detection research, we introduce the Speech INfilling Edit (SINE) dataset, created with Voicebox. We detailed the process of re-implementing Voicebox training and dataset creation. Subjective evaluations confirm that speech edited using this novel technique is more challenging to detect than conventional cut-and-paste methods. Despite human difficulty, experimental results demonstrate that self-supervised-based detectors can achieve remarkable performance in detection, localization, and generalization across different edit methods. The dataset and related models will be made publicly available. 

**Abstract (ZH)**: 神经语音编辑的进步引起了对其在欺骗攻击中滥用的担忧。传统部分编辑的语音数据集主要关注剪切和粘贴编辑，虽然能够保持说话人的一致性，但往往会引入可检测的不连续性。最近的方法，如A³T和Voicebox，则通过利用上下文信息改进了过渡效果。为了促进欺骗检测研究，我们引入了使用Voicebox创建的Speech INfilling Edit (SINE)数据集。我们详细描述了重新实现Voicebox训练和数据集创建的过程。主观评估证实，使用此新方法编辑的语音比传统的剪切和粘贴方法更难检测。尽管人类难以检测，实验结果表明，基于自监督的检测器可以在检测、定位以及在不同编辑方法之间泛化方面取得显著性能。该数据集及相关模型将公开发布。 

---
# How to Select Pre-Trained Code Models for Reuse? A Learning Perspective 

**Title (ZH)**: 如何选择用于reuse的预训练代码模型？一种学习视角 

**Authors**: Zhangqian Bi, Yao Wan, Zhaoyang Chu, Yufei Hu, Junyi Zhang, Hongyu Zhang, Guandong Xu, Hai Jin  

**Link**: [PDF](https://arxiv.org/pdf/2501.03783)  

**Abstract**: Pre-training a language model and then fine-tuning it has shown to be an efficient and effective technique for a wide range of code intelligence tasks, such as code generation, code summarization, and vulnerability detection. However, pretraining language models on a large-scale code corpus is computationally expensive. Fortunately, many off-the-shelf Pre-trained Code Models (PCMs), such as CodeBERT, CodeT5, CodeGen, and Code Llama, have been released publicly. These models acquire general code understanding and generation capability during pretraining, which enhances their performance on downstream code intelligence tasks. With an increasing number of these public pre-trained models, selecting the most suitable one to reuse for a specific task is essential. In this paper, we systematically investigate the reusability of PCMs. We first explore three intuitive model selection methods that select by size, training data, or brute-force fine-tuning. Experimental results show that these straightforward techniques either perform poorly or suffer high costs. Motivated by these findings, we explore learning-based model selection strategies that utilize pre-trained models without altering their parameters. Specifically, we train proxy models to gauge the performance of pre-trained models, and measure the distribution deviation between a model's latent features and the task's labels, using their closeness as an indicator of model transferability. We conduct experiments on 100 widely-used opensource PCMs for code intelligence tasks, with sizes ranging from 42.5 million to 3 billion parameters. The results demonstrate that learning-based selection methods reduce selection time to 100 seconds, compared to 2,700 hours with brute-force fine-tuning, with less than 6% performance degradation across related tasks. 

**Abstract (ZH)**: 预训练语言模型并对其进行微调已被证明是一种高效且有效的方法，可用于多种代码智能任务，例如代码生成、代码总结和漏洞检测。然而，使用大型代码语料库预训练语言模型在计算上非常昂贵。幸运的是，很多现成的预训练代码模型（PCMs）已经公开发布，例如CodeBERT、CodeT5、CodeGen和Code Llama等。这些模型在预训练过程中获得了通用的代码理解和生成能力，从而增强了其在下游代码智能任务中的表现。随着越来越多的这些公开预训练模型出现，针对特定任务重复使用最适合的模型变得至关重要。本文系统地研究了PCMs的可重用性。我们首先探索了三种直观的模型选择方法，分别依据模型的大小、训练数据或直接微调来选择。实验结果表明，这些简单的技术要么效果不佳，要么成本较高。基于这些发现，我们进一步探索了基于学习的模型选择策略，这些策略可以在无需修改预训练模型参数的情况下利用预训练模型。具体而言，我们训练了代理模型来评估预训练模型的性能，并通过模型潜在特征与任务标签之间的相似度来衡量模型迁移能力的分布偏差。我们在100个广泛使用的开源PCMs上进行了实验，模型的参数量范围从4250万到3亿个。结果显示，基于学习的选择方法将选择时间缩短至100秒，而直接微调则需要2700小时，同时在相关任务中的性能降低了不到6%。 

---
# Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series 

**Title (ZH)**: 上下文对齐：激活并增强大型语言模型在时间序列中的能力 

**Authors**: Yuxiao Hu, Qian Li, Dongxiao Zhang, Jinyue Yan, Yuntian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.03747)  

**Abstract**: Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment but overlook LLMs' inherent strength on natural language processing -- their deep understanding of linguistic logic and structure rather than superficial embedding processing. We propose Context-Alignment, a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by a Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Demonstration examples prompt are employed to construct Demonstration Examples based Context-Alignment (DECA) following DSCA-GNNs framework. DECA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of DECA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provide powerful prior knowledge on context. 

**Abstract (ZH)**: 近年来，利用预训练的大语言模型（LLMs）处理时间序列（TS）任务引起了越来越多的关注，这涉及激活和增强LLMs的能力。许多方法旨在通过字元级别对齐来激活LLMs的能力，但忽略了LLMs在自然语言处理中固有的优势——对语言逻辑和结构的深刻理解，而不是浅层的嵌入处理。我们提出了一种名为Context-Alignment的新范式，通过将时间序列与自然语言环境中的语言组件对齐，使LLMs能够理解和上下文化时间序列数据，从而激活其能力。具体而言，这种上下文对齐包括结构对齐和逻辑对齐，通过应用双尺度上下文对齐图神经网络（DSCA-GNNs）来处理时间序列-语言多模态输入。结构对齐利用双尺度节点描述时间序列-语言的层次结构，使LLMs能够将长时间序列数据视为一个整体的语言组件，同时保留固有的字元特征。逻辑对齐使用有向边来引导逻辑关系，确保上下文语义的一致性。以示范示例为基础，我们构建了遵循DSCA-GNNs框架的基于示范示例上下文对齐（DECA）。DECA可以灵活地集成到预训练LLMs的不同层中，以提高逻辑和结构的意识，从而增强性能。广泛的实验表明DECA的有效性以及上下文对齐的重要性，特别是在少数样本和零样本预测任务中，证实了上下文对齐为上下文提供了强大的先验知识。 

---
# LlaMADRS: Prompting Large Language Models for Interview-Based Depression Assessment 

**Title (ZH)**: LlaMADRS：基于访谈的大语言模型抑郁症评估Prompting方法 

**Authors**: Gaoussou Youssouf Kebe, Jeffrey M. Girard, Einat Liebenthal, Justin Baker, Fernando De la Torre, Louis-Philippe Morency  

**Link**: [PDF](https://arxiv.org/pdf/2501.03624)  

**Abstract**: This study introduces LlaMADRS, a novel framework leveraging open-source Large Language Models (LLMs) to automate depression severity assessment using the Montgomery-Asberg Depression Rating Scale (MADRS). We employ a zero-shot prompting strategy with carefully designed cues to guide the model in interpreting and scoring transcribed clinical interviews. Our approach, tested on 236 real-world interviews from the Context-Adaptive Multimodal Informatics (CAMI) dataset, demonstrates strong correlations with clinician assessments. The Qwen 2.5--72b model achieves near-human level agreement across most MADRS items, with Intraclass Correlation Coefficients (ICC) closely approaching those between human raters. We provide a comprehensive analysis of model performance across different MADRS items, highlighting strengths and current limitations. Our findings suggest that LLMs, with appropriate prompting, can serve as efficient tools for mental health assessment, potentially increasing accessibility in resource-limited settings. However, challenges remain, particularly in assessing symptoms that rely on non-verbal cues, underscoring the need for multimodal approaches in future work. 

**Abstract (ZH)**: 本研究介绍了一种名为LlaMADRS的新框架，该框架利用开源的大规模语言模型（LLMs）自动化使用蒙特戈梅里-阿斯伯格抑郁量表（MADRS）评估抑郁严重程度。我们采用了零样本提示策略，并设计了精巧的触发器来引导模型在解释和评分转录的临床访谈时的思考过程。该方法在Context-Adaptive Multimodal Informatics (CAMI) 数据集中实际进行的236次临床访谈上进行了测试，显示出与临床评估人员的评分具有较强的关联性。Qwen 2.5--72b模型在大多数MADRS项目上达到接近人类水平的一致性，内在一致性相关系数（ICC）与人类评估者之间的一致性高度接近。我们对模型在不同MADRS项目上的性能进行了全面分析，突显了其优势和当前的局限性。研究结果表明，通过适当提示的大规模语言模型可以作为精神健康评估的高效工具，在资源受限的地区具有潜在的应用价值。然而，仍存在一些挑战，特别是在评估依赖非言语线索的症状方面，这强调了未来工作中需要采用多模态方法的重要性。 

---
# Discriminative Representation learning via Attention-Enhanced Contrastive Learning for Short Text Clustering 

**Title (ZH)**: 基于注意力增强对比学习的短文本聚类差异性表示学习 

**Authors**: Zhihao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2501.03584)  

**Abstract**: Contrastive learning has gained significant attention in short text clustering, yet it has an inherent drawback of mistakenly identifying samples from the same category as negatives and then separating them in the feature space (false negative separation), which hinders the generation of superior representations. To generate more discriminative representations for efficient clustering, we propose a novel short text clustering method, called Discriminative Representation learning via \textbf{A}ttention-\textbf{E}nhanced \textbf{C}ontrastive \textbf{L}earning for Short Text Clustering (\textbf{AECL}). The \textbf{AECL} consists of two modules which are the pseudo-label generation module and the contrastive learning module. Both modules build a sample-level attention mechanism to capture similarity relationships between samples and aggregate cross-sample features to generate consistent representations. Then, the former module uses the more discriminative consistent representation to produce reliable supervision information for assist clustering, while the latter module explores similarity relationships and consistent representations optimize the construction of positive samples to perform similarity-guided contrastive learning, effectively addressing the false negative separation issue. Experimental results demonstrate that the proposed \textbf{AECL} outperforms state-of-the-art methods. If the paper is accepted, we will open-source the code. 

**Abstract (ZH)**: 对比学习在短文本聚类中引起了广泛关注，但它有一个固有的缺点，即错误地将同一类别中的样本识别为负样本并在特征空间中将它们分开（假阴性分离），这阻碍了优质表示的生成。为了生成更具区分性的表示以提高聚类效率，我们提出了一种新的短文本聚类方法，称为基于注意增强对比学习的区分性表示学习（AECL）。AECL 包括两个模块，即伪标签生成模块和对比学习模块。这两个模块都构建了样本级别的注意力机制，以捕获样本之间的相似性关系并聚合跨样本特征以生成一致的表示。其中，伪标签生成模块使用更具区分性的一致表示来生成可靠的监督信息以辅助聚类，而对比学习模块则探索相似性关系和一致表示以优化正样本的构建，从而进行以相似性为导向的对比学习，有效地解决了假阴性分离问题。实验结果表明，提出的 AECL 在性能上优于现有的先进方法。如果论文被接受，我们将开源代码。 

---
# From Code to Compliance: Assessing ChatGPT's Utility in Designing an Accessible Webpage -- A Case Study 

**Title (ZH)**: 从代码到合规：评估ChatGPT在设计无障碍网页方面的实用性——一个案例研究 

**Authors**: Ammar Ahmed, Margarida Fresco, Fredrik Forsberg, Hallvard Grotli  

**Link**: [PDF](https://arxiv.org/pdf/2501.03572)  

**Abstract**: Web accessibility ensures that individuals with disabilities can access and interact with digital content without barriers, yet a significant majority of most used websites fail to meet accessibility standards. This study evaluates ChatGPT's (GPT-4o) ability to generate and improve web pages in line with Web Content Accessibility Guidelines (WCAG). While ChatGPT can effectively address accessibility issues when prompted, its default code often lacks compliance, reflecting limitations in its training data and prevailing inaccessible web practices. Automated and manual testing revealed strengths in resolving simple issues but challenges with complex tasks, requiring human oversight and additional iterations. Unlike prior studies, we incorporate manual evaluation, dynamic elements, and use the visual reasoning capability of ChatGPT along with the prompts to fix accessibility issues. Providing screenshots alongside prompts enhances the LLM's ability to address accessibility issues by allowing it to analyze surrounding components, such as determining appropriate contrast colors. We found that effective prompt engineering, such as providing concise, structured feedback and incorporating visual aids, significantly enhances ChatGPT's performance. These findings highlight the potential and limitations of large language models for accessible web development, offering practical guidance for developers to create more inclusive websites. 

**Abstract (ZH)**: 网络无障碍确保了残疾人士可以无障碍地访问和交互数字内容，然而，大多数常用网站未能达到无障碍标准。本研究评估了ChatGPT（GPT-4o）生成和改进符合Web内容无障碍指南（WCAG）的网页的能力。尽管ChatGPT在收到指令时能够有效解决无障碍问题，但其默认代码往往缺乏合规性，这反映了其训练数据和普遍存在的不无障碍网页实践的局限性。自动化和手动测试表明，ChatGPT在解决简单问题方面具有优势，但在复杂任务方面则存在挑战，需要人类监督和额外的迭代。与先前的研究不同，我们在研究中加入了手动评估、动态元素，并利用ChatGPT的视觉推理能力及其提示来修复无障碍问题。提供屏幕截图与提示增强语言模型（LLM）处理无障碍问题的能力，使其能够分析周围组件，例如确定适当的对比度颜色。我们发现，有效的提示工程，如提供简洁且结构化的反馈以及结合视觉辅助，能够显著提高ChatGPT的表现。这些发现突显了大型语言模型在无障碍网页开发中的潜力和局限性，并为开发人员提供实用指导，以创建更加包容性网站。 

---
# Strategic Fusion Optimizes Transformer Compression 

**Title (ZH)**: 战略融合优化变压器压缩 

**Authors**: Md Shoaibur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2501.03273)  

**Abstract**: This study investigates transformer model compression by systematically pruning its layers. We evaluated 14 pruning strategies across nine diverse datasets, including 12 strategies based on different signals obtained from layer activations, mutual information, gradients, weights, and attention. To address the limitations of single-signal strategies, we introduced two fusion strategies, linear regression and random forest, which combine individual strategies (i.e., strategic fusion), for more informed pruning decisions. Additionally, we applied knowledge distillation to mitigate any accuracy loss during layer pruning. Our results reveal that random forest strategic fusion outperforms individual strategies in seven out of nine datasets and achieves near-optimal performance in the other two. The distilled random forest surpasses the original accuracy in six datasets and mitigates accuracy drops in the remaining three. Knowledge distillation also improves the accuracy-to-size ratio by an average factor of 18.84 across all datasets. Supported by mathematical foundations and biological analogies, our findings suggest that strategically combining multiple signals can lead to efficient, high-performing transformer models for resource-constrained applications. 

**Abstract (ZH)**: 本研究通过系统地修剪 transformer 模型的层来探究其压缩方法。我们评估了 14 种修剪策略在九个不同的数据集上的效果，其中包括依据层激活信号、互信息、梯度、权重和注意力的 12 种策略。为了解决单一信号策略的局限性，我们引入了两种融合策略，即线性回归和随机森林，通过结合各种策略（即选择性融合）来做出更加明智的修剪决策。此外，我们还应用了知识蒸馏以减轻在修剪层过程中可能产生的准确率损失。结果显示，在七个数据集中，随机森林选择性融合优于单一策略，而在另外两个数据集中则接近最优性能。蒸馏后的随机森林在六个数据集上的准确率超过了原始模型，并在剩余的三个数据集中减轻了准确率的下降。知识蒸馏还使所有数据集中的准确率和模型大小比值平均提高了 18.84 倍。基于数学基础和生物类比，我们的研究结果表明，有选择地结合多种信号可以为资源受限的应用生成高效、高性能的 transformer 模型。 

---
# Backdoor Token Unlearning: Exposing and Defending Backdoors in Pretrained Language Models 

**Title (ZH)**: 后门标记遗忘：揭示并防御预训练语言模型中的后门攻击 

**Authors**: Peihai Jiang, Xixiang Lyu, Yige Li, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.03272)  

**Abstract**: Supervised fine-tuning has become the predominant method for adapting large pretrained models to downstream tasks. However, recent studies have revealed that these models are vulnerable to backdoor attacks, where even a small number of malicious samples can successfully embed backdoor triggers into the model. While most existing defense methods focus on post-training backdoor defense, efficiently defending against backdoor attacks during training phase remains largely unexplored. To address this gap, we propose a novel defense method called Backdoor Token Unlearning (BTU), which proactively detects and neutralizes trigger tokens during the training stage. Our work is based on two key findings: 1) backdoor learning causes distinctive differences between backdoor token parameters and clean token parameters in word embedding layers, and 2) the success of backdoor attacks heavily depends on backdoor token parameters. The BTU defense leverages these properties to identify aberrant embedding parameters and subsequently removes backdoor behaviors using a fine-grained unlearning technique. Extensive evaluations across three datasets and four types of backdoor attacks demonstrate that BTU effectively defends against these threats while preserving the model's performance on primary tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 监督微调已成为将大型预训练模型适应下游任务的主要方法。然而，最近的研究揭示出这些模型存在后门攻击的脆弱性，即使少量恶意样本也能够成功地将后门触发器嵌入模型中。尽管目前大多数已有的防御方法集中在后训练阶段的后门防御上，但在训练阶段有效防御后门攻击的研究仍然相对缺乏。为填补这一空白，我们提出了一种名为后门标记遗忘（Backdoor Token Unlearning，BTU）的新型防御方法，该方法在训练阶段主动检测并中和触发标记。我们的工作基于两个关键发现：1）后门学习导致了后门标记参数与干净标记参数在词嵌入层中的显著差异；2）后门攻击的成功高度依赖于后门标记参数。BTU防御利用这些特性来识别异常的嵌入参数，并通过精细的遗忘技术来消除后门行为。在三个数据集和四种类型的后门攻击上进行的广泛评估表明，BTU有效抵御这些威胁，同时保持模型在主要任务上的性能。我们的代码可在此处获取：[URL]。 

---
# A Semantically-Aware, Kernel-Enhanced, and Divergence-Rich Paradigm for Direct Preference Optimization 

**Title (ZH)**: 一种语义aware、核增强且富有发散性的直接偏好优化范式 

**Authors**: Amitava Das, Suranjana Trivedy, Danush Khanna, Rajarshi Roy, Gurpreet Singh, Basab Ghosh, Yaswanth Narsupalli, Vinija Jain, Vasu Sharma, Aishwarya Naresh Reganti, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2501.03271)  

**Abstract**: The rapid rise of large language models (LLMs) has unlocked many applications but also underscores the challenge of aligning them with diverse values and preferences. Direct Preference Optimization (DPO) is central to alignment but constrained by fixed divergences and limited feature transformations. We propose DPO-Kernels, which integrates kernel methods to address these issues through four key contributions: (i) Kernelized Representations with polynomial, RBF, Mahalanobis, and spectral kernels for richer transformations, plus a hybrid loss combining embedding-based and probability-based objectives; (ii) Divergence Alternatives (Jensen-Shannon, Hellinger, Renyi, Bhattacharyya, Wasserstein, and f-divergences) for greater stability; (iii) Data-Driven Selection metrics that automatically choose the best kernel-divergence pair; and (iv) a Hierarchical Mixture of Kernels for both local precision and global modeling. Evaluations on 12 datasets demonstrate state-of-the-art performance in factuality, safety, reasoning, and instruction following. Grounded in Heavy-Tailed Self-Regularization, DPO-Kernels maintains robust generalization for LLMs, offering a comprehensive resource for further alignment research. 

**Abstract (ZH)**: 大语言模型（LLMs）的迅速崛起解锁了许多应用，但也突显了将它们与多样化价值观和偏好对齐的挑战。直接偏好优化（DPO）是实现对齐的核心，但受制于固定的差异度量和有限的功能变换。我们提出了DPO-Kernels，通过四种关键贡献整合了核方法来解决这些问题：(i) 核化表示，包括多项式核、RBF核、马氏距离核和谱核，以实现更丰富的变换，同时结合基于嵌入和基于概率的目标的混合损失；(ii) 更广泛的差异度量（Jensen-Shannon、Hellinger、Rényi、Bhattacharyya、Wasserstein和f-差异度量），以提高稳定性；(iii) 基于数据的选择度量，能够自动选择最优的核-差异度量配对；(iv) 层次核混合模型，以实现局部精确性和全局建模的结合。在12个数据集上的评估表明，DPO-Kernels在事实性、安全性、推理能力和指令遵循方面均表现出最先进的性能。基于重尾自我正则化，DPO-Kernels为LLMs提供了稳健的一般化效果，并为未来的对齐研究提供了全面的资源。 

---
# Breaking Through the Spike: Spike Window Decoding for Accelerated and Precise Automatic Speech Recognition 

**Title (ZH)**: 突破尖峰束缚：基于尖峰窗口解码的加速精准自动语音识别 

**Authors**: Wei Zhang, Tian-Hao Zhang, Chao Luo, Hui Zhou, Chao Yang, Xinyuan Qian, Xu-Cheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.03257)  

**Abstract**: Recently, end-to-end automatic speech recognition has become the mainstream approach in both industry and academia. To optimize system performance in specific scenarios, the Weighted Finite-State Transducer (WFST) is extensively used to integrate acoustic and language models, leveraging its capacity to implicitly fuse language models within static graphs, thereby ensuring robust recognition while also facilitating rapid error correction. However, WFST necessitates a frame-by-frame search of CTC posterior probabilities through autoregression, which significantly hampers inference speed. In this work, we thoroughly investigate the spike property of CTC outputs and further propose the conjecture that adjacent frames to non-blank spikes carry semantic information beneficial to the model. Building on this, we propose the Spike Window Decoding algorithm, which greatly improves the inference speed by making the number of frames decoded in WFST linearly related to the number of spiking frames in the CTC output, while guaranteeing the recognition performance. Our method achieves SOTA recognition accuracy with significantly accelerates decoding speed, proven across both AISHELL-1 and large-scale In-House datasets, establishing a pioneering approach for integrating CTC output with WFST. 

**Abstract (ZH)**: 近年来，端到端自动语音识别已成为工业和学术界的主流方法。为了在特定场景中优化系统性能，广泛使用了加权有限状态转换器（WFST）来整合声学模型和语言模型，利用其在静态图中隐式融合语言模型的能力，从而确保稳健的识别性能，并助力快速的错误修正。然而，WFST需要通过自回归方式对CTC后验概率进行帧内搜索，这显著地减慢了推理速度。本工作中，我们深入探讨了CTC输出的尖峰特性，并进一步提出相邻非空尖峰帧携带对模型有益的语义信息的猜想。基于这一发现，我们提出了尖峰窗口解码算法，该算法通过使WFST中解码的帧数与CTC输出中的尖峰帧数呈线性关系，极大地提高了推理速度，同时保证了识别性能。我们的方法在多个数据集上实现了目前的最佳识别准确率，并显著加速了解码速度，包括AISHELL-1和大规模内部数据集，为CTC输出与WFST的整合提供了开创性的方法。 

---
# Bridging Auditory Perception and Language Comprehension through MEG-Driven Encoding Models 

**Title (ZH)**: 通过MEG驱动的编码模型连接听觉感知与语言理解 

**Authors**: Matteo Ciferri, Matteo Ferrante, Nicola Toschi  

**Link**: [PDF](https://arxiv.org/pdf/2501.03246)  

**Abstract**: Understanding the neural mechanisms behind auditory and linguistic processing is key to advancing cognitive neuroscience. In this study, we use Magnetoencephalography (MEG) data to analyze brain responses to spoken language stimuli. We develop two distinct encoding models: an audio-to-MEG encoder, which uses time-frequency decompositions (TFD) and wav2vec2 latent space representations, and a text-to-MEG encoder, which leverages CLIP and GPT-2 embeddings. Both models successfully predict neural activity, demonstrating significant correlations between estimated and observed MEG signals. However, the text-to-MEG model outperforms the audio-based model, achieving higher Pearson Correlation (PC) score. Spatially, we identify that auditory-based embeddings (TFD and wav2vec2) predominantly activate lateral temporal regions, which are responsible for primary auditory processing and the integration of auditory signals. In contrast, textual embeddings (CLIP and GPT-2) primarily engage the frontal cortex, particularly Broca's area, which is associated with higher-order language processing, including semantic integration and language production, especially in the 8-30 Hz frequency range. The strong involvement of these regions suggests that auditory stimuli are processed through more direct sensory pathways, while linguistic information is encoded via networks that integrate meaning and cognitive control. Our results reveal distinct neural pathways for auditory and linguistic information processing, with higher encoding accuracy for text representations in the frontal regions. These insights refine our understanding of the brain's functional architecture in processing auditory and textual information, offering quantitative advancements in the modelling of neural responses to complex language stimuli. 

**Abstract (ZH)**: 理解听觉和语言处理背后的神经机制对于推动认知神经科学的发展至关重要。本研究使用磁共振脑电图（MEG）数据来分析对口头语言刺激的脑部反应。我们开发了两种不同的编码模型：一种是基于音频的MEG编码器，使用时频分解（TFD）和wav2vec2潜在空间表示；另一种是基于文本的MEG编码器，利用CLIP和GPT-2嵌入。两种模型都成功预测了神经活动，显示出估计的MEG信号与观察到的MEG信号之间存在显著的相关性。然而，基于文本的MEG模型在皮尔逊相关系数（PC）方面优于基于音频的模型。从空间上看，我们发现基于听觉的嵌入（TFD和wav2vec2）主要激活侧颞区，这是初级听觉处理和听觉信号整合的责任区域。相比之下，基于文本的嵌入（CLIP和GPT-2）主要涉及额叶皮层，特别是布罗卡区，这是一个与高层次语言处理相关的区域，包括语义整合和语言产生，特别是在8-30 Hz的频率范围内。这些区域的强烈参与表明，听觉刺激通过更直接的感觉通路处理，而语言信息则通过整合意义和认知控制的网络编码。研究结果揭示了听觉和语言信息处理的不同神经通路，文本表示在额叶区域编码更为准确。这些洞见细化了我们对大脑处理听觉和文本信息的功能架构的理解，为建模复杂语言刺激的神经响应提供了定量的进展。 

---
