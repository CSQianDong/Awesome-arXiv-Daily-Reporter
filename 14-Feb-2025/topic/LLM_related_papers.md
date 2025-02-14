# Human-LLM Coevolution: Evidence from Academic Writing 

**Title (ZH)**: 人类-大规模语言模型共进化：来自学术写作的证据 

**Authors**: Mingmeng Geng, Roberto Trotta  

**Link**: [PDF](https://arxiv.org/pdf/2502.09606)  

**Abstract**: With a statistical analysis of arXiv paper abstracts, we report a marked drop in the frequency of several words previously identified as overused by ChatGPT, such as "delve", starting soon after they were pointed out in early 2024. The frequency of certain other words favored by ChatGPT, such as "significant", has instead kept increasing. These phenomena suggest that some authors of academic papers have adapted their use of large language models (LLMs), for example, by selecting outputs or applying modifications to the LLM-generated content. Such coevolution and cooperation of humans and LLMs thus introduce additional challenges to the detection of machine-generated text in real-world scenarios. Estimating the impact of LLMs on academic writing by examining word frequency remains feasible, and more attention should be paid to words that were already frequently employed, including those that have decreased in frequency. 

**Abstract (ZH)**: 通过对arXiv论文摘要的统计分析，我们发现，在ChatGPT指出了某些过度使用的词汇（如“delve”）后的不久，这些词汇在论文摘要中的出现频率显著下降。另一方面，ChatGPT青睐的某些词汇（如“significant”）的出现频率则继续保持上升趋势。这些现象表明，一些学术论文的作者可能已经调整了他们对大语言模型（LLM）的使用方式，例如，通过选择输出或对LLM生成的内容进行修改。这种人类与LLM的共同发展和合作，为在实际场景中检测机器生成的文本带来了额外的挑战。通过分析词频来评估LLM对学术写作的影响仍然是可行的，未来应更加关注那些原本就经常被使用的词汇，包括那些使用频率有所下降的词汇。 

---
# Logical forms complement probability in understanding language model (and human) performance 

**Title (ZH)**: 逻辑形式在理解语言模型（及人类）的表现中补充了概率方法 

**Authors**: Yixuan Wang, Freda Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.09589)  

**Abstract**: With the increasing interest in using large language models (LLMs) for planning in natural language, understanding their behaviors becomes an important research question. This work conducts a systematic investigation of LLMs' ability to perform logical reasoning in natural language. We introduce a controlled dataset of hypothetical and disjunctive syllogisms in propositional and modal logic and use it as the testbed for understanding LLM performance. Our results lead to novel insights in predicting LLM behaviors: in addition to the probability of input (Gonen et al., 2023; McCoy et al., 2024), logical forms should be considered as orthogonal factors. In addition, we show similarities and differences between the logical reasoning performances of humans and LLMs by comparing LLM and human behavioral results. 

**Abstract (ZH)**: 随着对使用大规模语言模型（LLMs）进行自然语言规划的兴趣日益增加，理解它们的行为已成为一个重要的研究问题。本工作系统地探讨了LLMs在自然语言中进行逻辑推理的能力。我们引入了一个受控的数据集，其中包括命题逻辑和模态逻辑中的假设性与析取三段论，并将其用作研究LLMs性能的平台。我们的结果提供了关于预测LLMs行为的新见解：除了输入的概率（Gonen et al., 2023；McCoy et al., 2024）之外，逻辑形式应被视为独立的因素。此外，通过比较LLMs和人类的行为结果，我们展示了人类和LLMs在逻辑推理性能上的相似性和差异。 

---
# Zero-shot generation of synthetic neurosurgical data with large language models 

**Title (ZH)**: 使用大型语言模型进行零样本合成神经外科数据生成 

**Authors**: Austin A. Barr, Eddie Guo, Emre Sezgin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09566)  

**Abstract**: Clinical data is fundamental to advance neurosurgical research, but access is often constrained by data availability, small sample sizes, privacy regulations, and resource-intensive preprocessing and de-identification procedures. Synthetic data offers a potential solution to challenges associated with accessing and using real-world data (RWD). This study aims to evaluate the capability of zero-shot generation of synthetic neurosurgical data with a large language model (LLM), GPT-4o, by benchmarking with the conditional tabular generative adversarial network (CTGAN). Synthetic datasets were compared to real-world neurosurgical data to assess fidelity (means, proportions, distributions, and bivariate correlations), utility (ML classifier performance on RWD), and privacy (duplication of records from RWD). The GPT-4o-generated datasets matched or exceeded CTGAN performance, despite no fine-tuning or access to RWD for pre-training. Datasets demonstrated high univariate and bivariate fidelity to RWD without directly exposing any real patient records, even at amplified sample size. Training an ML classifier on GPT-4o-generated data and testing on RWD for a binary prediction task showed an F1 score (0.706) with comparable performance to training on the CTGAN data (0.705) for predicting postoperative functional status deterioration. GPT-4o demonstrated a promising ability to generate high-fidelity synthetic neurosurgical data. These findings also indicate that data synthesized with GPT-4o can effectively augment clinical data with small sample sizes, and train ML models for prediction of neurosurgical outcomes. Further investigation is necessary to improve the preservation of distributional characteristics and boost classifier performance. 

**Abstract (ZH)**: 临床数据是推进神经外科研究的基础，但访问和使用这些数据常常受到数据可用性、样本量小、隐私法规以及耗时的预处理和去标识化程序的限制。合成数据为解决访问和使用真实世界数据（RWD）相关的挑战提供了一种潜在的解决方案。本研究旨在通过与条件性表格生成对抗网络（CTGAN）进行基准测试，评估大型语言模型（LLM）GPT-4o零样本生成神经外科合成数据的能力。合成数据集与真实的神经外科数据进行了比较，以评估其真实度（均值、比例、分布和双变量相关性）、效用（RWD上的ML分类器性能）和隐私（RWD中患者记录的重复情况）。尽管GPT-4o未经过微调也未访问用于预训练的RWD数据，但生成的的数据集在真实度、双变量真实度方面与CTGAN相当，即使在样本量放大时也未直接暴露任何真实患者的记录。在二分类预测任务中，使用GPT-4o生成的数据训练ML分类器，并在RWD上进行测试，显示的F1分数（0.706）与使用CTGAN数据训练的F1分数（0.705）相当，用于预测术后功能状态恶化。GPT-4o展示了生成高质量合成神经外科数据的潜力。这些发现还表明，使用GPT-4o合成的数据可以有效补充小样本临床数据，并训练机器学习模型以预测神经外科结果。为进一步提高分布特征的保留和增强分类器性能，进一步的研究是必要的。 

---
# Mind the Gap! Choice Independence in Using Multilingual LLMs for Persuasive Co-Writing Tasks in Different Languages 

**Title (ZH)**: 注意差距！在不同语言的多语言大语言模型辅助写作任务中的选择独立性 

**Authors**: Shreyan Biswas, Alexander Erlei, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2502.09532)  

**Abstract**: Recent advances in generative AI have precipitated a proliferation of novel writing assistants. These systems typically rely on multilingual large language models (LLMs), providing globalized workers the ability to revise or create diverse forms of content in different languages. However, there is substantial evidence indicating that the performance of multilingual LLMs varies between languages. Users who employ writing assistance for multiple languages are therefore susceptible to disparate output quality. Importantly, recent research has shown that people tend to generalize algorithmic errors across independent tasks, violating the behavioral axiom of choice independence. In this paper, we analyze whether user utilization of novel writing assistants in a charity advertisement writing task is affected by the AI's performance in a second language. Furthermore, we quantify the extent to which these patterns translate into the persuasiveness of generated charity advertisements, as well as the role of peoples' beliefs about LLM utilization in their donation choices. Our results provide evidence that writers who engage with an LLM-based writing assistant violate choice independence, as prior exposure to a Spanish LLM reduces subsequent utilization of an English LLM. While these patterns do not affect the aggregate persuasiveness of the generated advertisements, people's beliefs about the source of an advertisement (human versus AI) do. In particular, Spanish-speaking female participants who believed that they read an AI-generated advertisement strongly adjusted their donation behavior downwards. Furthermore, people are generally not able to adequately differentiate between human-generated and LLM-generated ads. Our work has important implications for the design, development, integration, and adoption of multilingual LLMs as assistive agents -- particularly in writing tasks. 

**Abstract (ZH)**: 近年来生成式AI的进展催生了各种新型写作助手。这些系统通常依赖于多语言大型语言模型（LLMs），使全球工作者能够用不同的语言修订或创作多样化的内容。然而，有大量证据表明，多语言LLMs在不同语言中的表现存在差异。使用多语言写作助手的用户因此可能会遇到输出质量不一致的问题。最近的研究显示，人们倾向于将算法错误泛化到独立任务中，违背了选择独立性的行为准则。本文通过分析，在慈善广告写作任务中，用户对基于LLM的写作助手的使用是否受到AI在第二语言中的表现影响。我们还量化了这些模式如何影响生成广告的说服力，以及人们对LLM使用信念在捐款选择中的作用。我们的研究结果表明，在先前接触过西班牙语LLM后，使用英语LLM的写作受试者违反了选择独立性。虽然这些模式并不影响生成广告的整体说服力，但人们对广告来源（人类 vs AI）的信念确实有影响。特别是，相信读到的是AI生成广告的西班牙语女性参与者，强烈调整了他们的捐款行为。此外，人们普遍无法准确区分由人类或LLM生成的广告。我们的研究对多语言LLMs作为辅助工具的设计、开发、集成和采用——尤其是写作任务——具有重要启示。 

---
# Improve LLM-based Automatic Essay Scoring with Linguistic Features 

**Title (ZH)**: 基于语言学特征改进基于大语言模型的自动作文评分 

**Authors**: Zhaoyi Joey Hou, Alejandro Ciuba, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.09497)  

**Abstract**: Automatic Essay Scoring (AES) assigns scores to student essays, reducing the grading workload for instructors. Developing a scoring system capable of handling essays across diverse prompts is challenging due to the flexibility and diverse nature of the writing task. Existing methods typically fall into two categories: supervised feature-based approaches and large language model (LLM)-based methods. Supervised feature-based approaches often achieve higher performance but require resource-intensive training. In contrast, LLM-based methods are computationally efficient during inference but tend to suffer from lower performance. This paper combines these approaches by incorporating linguistic features into LLM-based scoring. Experimental results show that this hybrid method outperforms baseline models for both in-domain and out-of-domain writing prompts. 

**Abstract (ZH)**: 自动作文评分（AES）能够为学生的作文分配分数，减轻教师的评分负担。由于写作任务的灵活性和多样性，开发一个能够处理不同主题作文的评分系统具有挑战性。现有的方法通常可以分为两类：监督特征基方法和大规模语言模型（LLM）基方法。监督特征基方法往往能获得更高的性能，但需要资源密集型的训练。相比之下，LLM基方法在推理过程中计算效率高，但在性能上往往会有所不足。本文通过将语言特征引入LLM基评分方法，将这两种方法结合起来。实验结果表明，这种混合方法无论是针对领域内还是领域外的主题，都能优于基线模型。 

---
# Objective quantification of mood states using large language models 

**Title (ZH)**: 使用大规模语言模型对情绪状态进行客观量化 

**Authors**: Jakub Onysk, Quentin Huys  

**Link**: [PDF](https://arxiv.org/pdf/2502.09487)  

**Abstract**: Emotional states influence human behaviour and cognition, leading to diverse thought trajectories. Similarly, Large Language Models (LLMs) showcase an excellent level of response consistency across wide-ranging contexts (prompts). We leverage these parallels to establish a framework for quantifying mental states. Our approach utilises self-report questionnaires that reliably assess these states due to their inherent sensitivity to patterns of co-occurring responses. Specifically, we recruited a large sample of participants (N=422) to investigate how well an LLM (Mistral-7B-OpenOrca) quantifies a heterogenous set of depressive mood states measured with participants' open-ended responses to a depression questionnaire. We show LLM responses to held-out multiple-choice questions, given participants' open-ended answers, correlate strongly (r: 0.52-0.84) with true questionnaire scores, demonstrating LLM's generalisation from mood representations. We explore a link between these representations and factor analysis. Using ridge regression, we find depression-related subspaces within LLM hidden states. We show these subspaces to be predictive of participants' "Depression" and "Somatic & Emotional Distress" factor scores, as well as suicidality severity. Overall, LLMs can provide quantitative measures of mental states. The reliability of these hinges upon how informative the questions we ask participants are. Used correctly, this approach could supplement mental state assessment in a variety of settings. 

**Abstract (ZH)**: 情绪状态会影响人类的行为和认知，从而引导出多样的思维轨迹。类似地，大型语言模型（LLMs）在广泛的情境（提示）中展示了高度一致的响应能力。我们利用这些相似之处来建立一种量化心理状态的框架。我们的方法利用了自我报告问卷的优势，这些问卷能够可靠地评估情绪状态，因为它们对伴随出现的响应模式高度敏感。具体而言，我们招募了422名参与者来研究LLM（Mistral-7B-OpenOrca）如何量化通过参与者对抑郁问卷的开放式回答来衡量的异质性抑郁情绪状态。我们发现，给定参与者开放式答案，LLM对保留下来的选择题问题的响应与真实问卷得分高度相关（相关系数r：0.52-0.84），这表明LLM可以从情绪表征中进行泛化。我们探讨了这些表征与因子分析之间的联系。使用岭回归，我们在LLM的隐藏状态中发现了与抑郁相关的子空间。我们显示这些子空间能够预测参与者的“抑郁”和“躯体与情感痛苦”因子得分，以及自杀严重性。总体而言，LLM可以提供心理状态的量化指标。这些指标的可靠性取决于我们向参与者提问的质量。正确使用这一方法，可以在多种场景中补充心理状态的评估。 

---
# The Multilingual Mind : A Survey of Multilingual Reasoning in Language Models 

**Title (ZH)**: 多语言思维：语言模型中的多语言推理综述 

**Authors**: Akash Ghosh, Debayan Datta, Sriparna Saha, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.09457)  

**Abstract**: While reasoning and multilingual capabilities in Language Models (LMs) have achieved remarkable progress in recent years, their integration into a unified paradigm, multilingual reasoning, is at a nascent stage. Multilingual reasoning requires language models to handle logical reasoning across languages while addressing misalignment, biases, and challenges in low-resource settings. This survey provides the first in-depth review of multilingual reasoning in LMs. In this survey, we provide a systematic overview of existing methods that leverage LMs for multilingual reasoning, specifically outlining the challenges, motivations, and foundational aspects of applying language models to reason across diverse languages. We provide an overview of the standard data resources used for training multilingual reasoning in LMs and the evaluation benchmarks employed to assess their multilingual capabilities. Next, we analyze various state-of-the-art methods and their performance on these benchmarks. Finally, we explore future research opportunities to improve multilingual reasoning in LMs, focusing on enhancing their ability to handle diverse languages and complex reasoning tasks. 

**Abstract (ZH)**: 近年来，语言模型（LMs）在推理和多语言能力方面取得了显著进展，但将这些能力整合到统一的多语言推理框架中仍处于起步阶段。多语言推理需要语言模型在处理不同语言中的逻辑推理时，同时解决语言间的不一致、偏差以及低资源环境中的挑战。本文综述是关于多语言推理在语言模型中的首个深入回顾。在本文中，我们提供了一种系统性的概览，介绍了现有利用语言模型进行多语言推理的方法，特别阐述了将语言模型应用于跨语言推理时所面临的问题、动机及其基础方面。我们回顾了用于训练多语言推理的语言模型的标准数据资源，以及用于评估其多语言能力的评估基准。接着，我们分析了各种最先进的方法及其在这些基准上的表现。最后，我们探讨了未来的研究机会，以提高语言模型在处理复杂推理任务和多种语言方面的能力。 

---
# On multi-token prediction for efficient LLM inference 

**Title (ZH)**: 高效的LLM推理中的多令牌预测方法 

**Authors**: Somesh Mehra, Javier Alonso Garcia, Lukas Mauch  

**Link**: [PDF](https://arxiv.org/pdf/2502.09419)  

**Abstract**: We systematically investigate multi-token prediction (MTP) capabilities within LLMs pre-trained for next-token prediction (NTP). We first show that such models inherently possess MTP capabilities via numerical marginalization over intermediate token probabilities, though performance is data-dependent and improves with model scale. Furthermore, we explore the challenges of integrating MTP heads into frozen LLMs and find that their hidden layers are strongly specialized for NTP, making adaptation non-trivial. Finally, we show that while joint training of MTP heads with the backbone improves performance, it cannot fully overcome this barrier, prompting further research in this direction. Our findings provide a deeper understanding of MTP applied to pretrained LLMs, informing strategies for accelerating inference through parallel token prediction. 

**Abstract (ZH)**: 我们系统地研究了用于后续令牌预测（Next-token Prediction, NTP）预训练的大型语言模型（LLMs）中的多令牌预测（Multi-token Prediction, MTP）能力。我们首先表明，这类模型本质上具备MTP能力，这是因为通过数值方式对中间令牌概率进行边缘化处理。尽管其性能依赖于数据，但随模型规模的增大而提升。此外，我们探讨了将MTP头部集成到冻结的LLM中所面临的挑战，并发现在其隐藏层中，这类模型主要针对NTP进行了高度专业化，使得适应过程变得相当复杂。最后，我们证明通过骨干网络和MTP头部的联合训练可以提高性能，但无法完全克服这一障碍，这促使我们在该方向上开展进一步研究。我们的发现为MTP在预训练LLM中的应用提供了更深入的理解，并为通过并行令牌预测加速推理策略提供了指导。 

---
# SQuARE: Sequential Question Answering Reasoning Engine for Enhanced Chain-of-Thought in Large Language Models 

**Title (ZH)**: SQuARE：增强大型语言模型链式思考的序列问答推理引擎 

**Authors**: Daniel Fleischer, Moshe Berchansky, Gad Markovits, Moshe Wasserblat  

**Link**: [PDF](https://arxiv.org/pdf/2502.09390)  

**Abstract**: In the rapidly evolving field of Natural Language Processing, Large Language Models (LLMs) are tasked with increasingly complex reasoning challenges. Traditional methods like chain-of-thought prompting have shown promise but often fall short in fully leveraging a model's reasoning capabilities. This paper introduces SQuARE (Sequential Question Answering Reasoning Engine), a novel prompting technique designed to improve reasoning through a self-interrogation paradigm. Building upon CoT frameworks, SQuARE prompts models to generate and resolve multiple auxiliary questions before tackling the main query, promoting a more thorough exploration of various aspects of a topic. Our expansive evaluations, conducted with Llama 3 and GPT-4o models across multiple question-answering datasets, demonstrate that SQuARE significantly surpasses traditional CoT prompts and existing rephrase-and-respond methods. By systematically decomposing queries, SQuARE advances LLM capabilities in reasoning tasks. The code is publicly available at this https URL. 

**Abstract (ZH)**: 在自然语言处理这一快速发展的领域中，大型语言模型（LLMs）面临着日益复杂的推理挑战。传统的链式思考提示方法虽然显示出一定的潜力，但往往未能充分利用模型的推理能力。本文引入了SQuARED（顺序问答推理引擎），这是一种新型的提示技术，旨在通过自我质疑的范式提高推理能力。SQuARED建立在链式思考（CoT）框架之上，促使模型生成和解决多个辅助问题后再处理主要问题，从而促进对主题各个方面进行更全面的探索。我们在Llama 3和GPT-4o模型上，通过对多个问答数据集进行广泛评估，证明了SQuARED显著优于传统的链式思考提示和现有的重述并回应方法。通过系统地分解查询，SQuARED推动了LLM在推理任务中的能力。代码已公开，可从以下链接访问：this https URL。 

---
# Truth Knows No Language: Evaluating Truthfulness Beyond English 

**Title (ZH)**: 真理不分语言：超越英语的可信度评估 

**Authors**: Blanca Calvo Figueras, Eneko Sagarzazu, Julen Etxaniz, Jeremy Barnes, Pablo Gamallo, Iria De Dios Flores, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2502.09387)  

**Abstract**: We introduce a professionally translated extension of the TruthfulQA benchmark designed to evaluate truthfulness in Basque, Catalan, Galician, and Spanish. Truthfulness evaluations of large language models (LLMs) have primarily been conducted in English. However, the ability of LLMs to maintain truthfulness across languages remains under-explored. Our study evaluates 12 state-of-the-art open LLMs, comparing base and instruction-tuned models using human evaluation, multiple-choice metrics, and LLM-as-a-Judge scoring. Our findings reveal that, while LLMs perform best in English and worst in Basque (the lowest-resourced language), overall truthfulness discrepancies across languages are smaller than anticipated. Furthermore, we show that LLM-as-a-Judge correlates more closely with human judgments than multiple-choice metrics, and that informativeness plays a critical role in truthfulness assessment. Our results also indicate that machine translation provides a viable approach for extending truthfulness benchmarks to additional languages, offering a scalable alternative to professional translation. Finally, we observe that universal knowledge questions are better handled across languages than context- and time-dependent ones, highlighting the need for truthfulness evaluations that account for cultural and temporal variability. Dataset and code are publicly available under open licenses. 

**Abstract (ZH)**: 我们介绍了一种专业翻译扩展的TruthfulQA基准，旨在评估巴斯克语、卡塔兰语、加利西亚语和西班牙语中的真实性。大型语言模型（LLMs）的真实性评估主要集中在英语上。然而，LLMs在不同语言中保持真实性的能力仍处于研究不足的状态。我们的研究评估了12个最新的开源LLMs，并通过人工评估、多项选择指标和LLM作为裁判评分，比较了基准模型和指令调优模型。研究结果表明，虽然LLMs在英语上的表现最佳，但在巴斯克语（资源最少的语言）上的表现最差，不过不同语言之间的真实性差异比预期的小得多。此外，我们还表明，LLM作为裁判与人类判断的相关性比多项选择指标更紧密，并且信息含量在真实性评估中起着关键作用。研究结果还表明，机器翻译为将真实性基准扩展到其他语言提供了一种可行的方法，相对于专业翻译，这是一种可扩展的替代方案。最后，我们观察到，在处理文化和时间依赖性较强的上下文问题时，LLMs表现得不如处理通用知识问题。这一发现凸显了需要进行考虑到文化与时间差异的真实性评估的需求。该数据集和代码在开放许可下可供公众使用。 

---
# A Judge-free LLM Open-ended Generation Benchmark Based on the Distributional Hypothesis 

**Title (ZH)**: 基于分布假设的无裁判ULLM开放生成基准 

**Authors**: Kentaro Imajo, Masanori Hirano, Shuji Suzuki, Hiroaki Mikami  

**Link**: [PDF](https://arxiv.org/pdf/2502.09316)  

**Abstract**: Evaluating the open-ended text generation of large language models (LLMs) is challenging because of the lack of a clear ground truth and the high cost of human or LLM-based assessments. We propose a novel benchmark that evaluates LLMs using n-gram statistics and rules, without relying on human judgement or LLM-as-a-judge approaches. Using 50 question and reference answer sets, we introduce three new metrics based on n-grams and rules: Fluency, Truthfulness, and Helpfulness. Our benchmark strongly correlates with GPT-4o-based evaluations while requiring significantly fewer computational resources, demonstrating its effectiveness as a scalable alternative for assessing LLMs' open-ended generation capabilities. 

**Abstract (ZH)**: 评估大规模语言模型（LLMs）的开放式文本生成具有挑战性，因为缺乏明确的 ground truth，并且依赖人类或基于LLM的评估成本较高。我们提出了一种新的基准方法，该方法使用n-元统计和规则来评估LLMs，无需依赖人类判断或LLM作为评判者的方法。利用50个问题和参考答案集，我们引入了基于n-元和规则的三个新指标：流畅性、真实性、和有用性。我们的基准与基于GPT-4o的评估高度相关，但所需计算资源显著减少，证明了其作为评估LLMs开放式生成能力的可扩展替代方法的有效性。 

---
# When the LM misunderstood the human chuckled: Analyzing garden path effects in humans and language models 

**Title (ZH)**: 当LM误解了人类的笑声：分析人类与语言模型中的花园路径效应 

**Authors**: Samuel Joseph Amouyal, Aya Meltzer-Asscher, Jonathan Berant  

**Link**: [PDF](https://arxiv.org/pdf/2502.09307)  

**Abstract**: Modern Large Language Models (LLMs) have shown human-like abilities in many language tasks, sparking interest in comparing LLMs' and humans' language processing. In this paper, we conduct a detailed comparison of the two on a sentence comprehension task using garden-path constructions, which are notoriously challenging for humans. Based on psycholinguistic research, we formulate hypotheses on why garden-path sentences are hard, and test these hypotheses on human participants and a large suite of LLMs using comprehension questions. Our findings reveal that both LLMs and humans struggle with specific syntactic complexities, with some models showing high correlation with human comprehension. To complement our findings, we test LLM comprehension of garden-path constructions with paraphrasing and text-to-image generation tasks, and find that the results mirror the sentence comprehension question results, further validating our findings on LLM understanding of these constructions. 

**Abstract (ZH)**: 现代大型语言模型（LLMs）在许多语言任务中展示了类人的能力，引发了将LLMs与人类的语言处理进行比较的兴趣。本文中，我们通过对包含“花园路径句”（garden-path constructions）的句子理解任务进行了详细比较，来探讨LLMs与人类的语言处理差异。基于心理语言学的研究，我们提出了关于为什么“花园路径句”难以理解的假设，并通过使用理解问题对人类参与者和大量的LLMs进行了验证。研究发现，无论是人类还是LLMs，在处理特定的句法复杂性时都会遇到困难，一些模型在理解上的表现与人类高度一致。为进一步验证我们的发现，我们还测试了LLMs对“花园路径句”的理解能力，分别使用改述任务和文本到图像生成任务进行测试，结果与句子理解问题的测试结果一致，进一步验证了LLMs对这些结构的理解能力。 

---
# SparQLe: Speech Queries to Text Translation Through LLMs 

**Title (ZH)**: SparQLe：通过大规模语言模型将语音查询转换为文本翻译 

**Authors**: Amirbek Djanibekov, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.09284)  

**Abstract**: With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that leverages self-supervised speech representations in combination with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English-language data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising solution for various speech understanding applications. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）影响力的增强，人们越来越关注将语音表示与这些模型结合起来，以实现更流畅的多模态处理和语音理解。本研究介绍了一种新颖的方法，利用自监督的语音表示与指令微调的LLMs相结合，用于语音到文本的转换。该提出的方法利用模态适配器，通过英语数据将提取的语音特征与指令微调的LLMs对齐。我们的实验表明，该方法有效地保留了输入语音的语义内容，并充当自监督语音模型与指令微调LLMs之间的有效桥梁，为各种语音理解应用提供了一个有前途的解决方案。 

---
# Thinking beyond the anthropomorphic paradigm benefits LLM research 

**Title (ZH)**: 超越拟人类化范式有利于大语言模型研究 

**Authors**: Lujain Ibrahim, Myra Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09192)  

**Abstract**: Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of computer science research articles from the past decade and present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). This terminology reflects deeper anthropomorphic conceptualizations which shape how we think about and conduct LLM research. We argue these conceptualizations may be limiting, and that challenging them opens up new pathways for understanding and improving LLMs beyond human analogies. To illustrate this, we identify and analyze five core anthropomorphic assumptions shaping prominent methodologies across the LLM development lifecycle, from the assumption that models must use natural language for reasoning tasks to the assumption that model capabilities should be evaluated through human-centric benchmarks. For each assumption, we demonstrate how non-anthropomorphic alternatives can open new directions for research and development. 

**Abstract (ZH)**: 赋予人类特征给技术，即把人的特质属性赋予给技术设备，是一种自动且无意识的反应，即使是技术专家也不例外。在本文中，我们分析了过去十年中数十万篇计算机科学研究文章，并展示了对大规模语言模型（LLMs）研究中赋予人类特征术语的普遍性和增长趋势的实证证据。这些术语反映了更深层的人类中心概念化，这些概念化塑造了我们对LLMs的研究思维和方法。我们认为这些概念化可能具有局限性，挑战这些概念化将开辟超出人类类比的新途径，以理解和改进LLMs。为了说明这一点，我们确定并分析了五项核心的人类中心假设，这些假设塑造了LLMs开发生命周期中的主流方法论，从模型必须使用自然语言进行推理任务到模型能力应该通过以人为主体的基准进行评估。对于每项假设，我们都展示了非人类中心的替代方案如何为研究和开发开辟新的方向。 

---
# RefineCoder: Iterative Improving of Large Language Models via Adaptive Critique Refinement for Code Generation 

**Title (ZH)**: RefineCoder：通过自适应批评细化实现大规模语言模型在代码生成中的逐步改进 

**Authors**: Changzhi Zhou, Xinyu Zhang, Dandan Song, Xiancai Chen, Wanli Gu, Huipeng Ma, Yuhang Tian, Mengdi Zhang, Linmei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09183)  

**Abstract**: Code generation has attracted increasing attention with the rise of Large Language Models (LLMs). Many studies have developed powerful code LLMs by synthesizing code-related instruction data and applying supervised fine-tuning. However, these methods are limited by teacher model distillation and ignore the potential of iterative refinement by self-generated code. In this paper, we propose Adaptive Critique Refinement (ACR), which enables the model to refine itself by self-generated code and external critique, rather than directly imitating the code responses of the teacher model. Concretely, ACR includes a composite scoring system with LLM-as-a-Judge to evaluate the quality of code responses and a selective critique strategy with LLM-as-a-Critic to critique self-generated low-quality code responses. We develop the RefineCoder series by iteratively applying ACR, achieving continuous performance improvement on multiple code generation benchmarks. Compared to the baselines of the same size, our proposed RefineCoder series can achieve comparable or even superior performance using less data. 

**Abstract (ZH)**: 代码生成随着大型语言模型（LLMs）的兴起越来越受到关注。许多研究通过综合代码相关的指令数据并应用监督微调，开发出了强大的代码LLMs。然而，这些方法受限于教师模型蒸馏，并且忽视了通过自动生成代码进行迭代改进的潜力。本文中，我们提出了一种自适应批评精炼（ACR）方法，该方法使模型能够通过自动生成的代码和外部批评来自我精炼，而不是直接模仿教师模型的代码响应。具体而言，ACR 包括一个由LLM作为裁判的综合评分系统，用于评估代码响应的质量，以及一个由LLM作为批评者的选择性批评策略，用于批评自动生成的低质量代码响应。我们通过迭代应用ACR，开发了RefineCoder系列，并在多个代码生成基准测试中实现了持续的性能提升。与相同规模的基线方法相比，我们提出的RefineCoder系列能够在使用更少数据的情况下达到相当甚至更优的性能。 

---
# Improving TCM Question Answering through Tree-Organized Self-Reflective Retrieval with LLMs 

**Title (ZH)**: 通过基于树组织的自我反思检索提升中医问答性能 

**Authors**: Chang Liu, Ying Chang, Jianmin Li, Yiqian Qu, Yu Li, Lingyong Cao, Shuyuan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09156)  

**Abstract**: Objectives: Large language models (LLMs) can harness medical knowledge for intelligent question answering (Q&A), promising support for auxiliary diagnosis and medical talent cultivation. However, there is a deficiency of highly efficient retrieval-augmented generation (RAG) frameworks within the domain of Traditional Chinese Medicine (TCM). Our purpose is to observe the effect of the Tree-Organized Self-Reflective Retrieval (TOSRR) framework on LLMs in TCM Q&A tasks.
Materials and Methods: We introduce the novel approach of knowledge organization, constructing a tree structure knowledge base with hierarchy. At inference time, our self-reflection framework retrieves from this knowledge base, integrating information across chapters. Questions from the TCM Medical Licensing Examination (MLE) and the college Classics Course Exam (CCE) were randomly selected as benchmark datasets.
Results: By coupling with GPT-4, the framework can improve the best performance on the TCM MLE benchmark by 19.85% in absolute accuracy, and improve recall accuracy from 27% to 38% on CCE datasets. In manual evaluation, the framework improves a total of 18.52 points across dimensions of safety, consistency, explainability, compliance, and coherence.
Conclusion: The TOSRR framework can effectively improve LLM's capability in Q&A tasks of TCM. 

**Abstract (ZH)**: 研究目的：大型语言模型（LLM）可以利用医学知识进行智能问答（Q&A），为辅助诊断和医学人才的培养提供支持。然而，在中医药（TCM）领域，高效的检索增强生成（RAG）框架存在不足。本文旨在观察Tree-Organized Self-Reflective Retrieval（TOSRR）框架在TCM问答任务中的效果。

材料与方法：我们引入了一种新型的知识组织方法，构建了具有层次结构的树状知识库。在推理过程中，我们的自省框架从该知识库中检索信息，整合跨章节的信息。我们选择了中医执业医师资格考试（MLE）和大学古典课程考试（CCE）中的随机问题作为基准数据集。

结果：与GPT-4相结合后，该框架在TCM MLE基准测试中的绝对准确率提高了19.85%，在CCE数据集上的召回准确率从27%提高到38%。在手动评估中，该框架在安全性、一致性、可解释性、合规性和连贯性等多个维度上提高了总共18.52分。

结论：TOSRR框架可以有效提高LLM在中医药问答任务中的能力。 

---
# CoSER: Coordinating LLM-Based Persona Simulation of Established Roles 

**Title (ZH)**: CoSER：基于大型语言模型的固定角色个性化模拟协调 

**Authors**: Xintao Wang, Heng Wang, Yifei Zhang, Xinfeng Yuan, Rui Xu, Jen-tse Huang, Siyu Yuan, Haoran Guo, Jiangjie Chen, Wei Wang, Yanghua Xiao, Shuchang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.09082)  

**Abstract**: Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively. 

**Abstract (ZH)**: 角色扮演语言代理（RPLA）已成为大规模语言模型（LLM）的有前途的应用之一。然而，模拟业已确立的人物对RPLA来说是一个挑战性任务，原因是缺乏真实的性格数据集和用于评估此类数据的细腻方法。本文中，我们提出了CoSER，一个包含高质量数据集、公开模型和评估协议的集合，旨在促进有效的RPLA。CoSER数据集涵盖了771部著名书籍中的17,966个角色，提供了真实世界的对话，包括各种对话设置、人物经历和内心想法。基于表演方法论，我们引入了给定情境表演，用于训练和评估角色扮演的LLM，在书籍场景中，LLM依次扮演多个角色。利用我们的数据集，我们开发了CoSER 8B和CoSER 70B，即基于LLaMA-3.1模型的高级公开角色扮演LLM。广泛的实验表明，CoSER数据集对于RPLA的训练、评估和检索具有重要价值。此外，CoSER 70B在我们的评估和三个现有基准（即InCharacter和LifeChoice基准）上表现出最先进的性能，分别达到75.80%和93.47%的准确率，超过了或匹配了GPT-4o的性能。 

---
# An Open Recipe: Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging 

**Title (ZH)**: 一种开放的食谱：通过模型合并将语言特定的大规模语言模型在一天内适配到一个推理模型中 

**Authors**: Kunat Pipatanakul, Pittawat Taveekitworachai, Potsawee Manakul, Kasima Tharnpipitchai  

**Link**: [PDF](https://arxiv.org/pdf/2502.09056)  

**Abstract**: This paper investigates data selection and model merging methodologies aimed at incorporating advanced reasoning capabilities such as those of DeepSeek R1 into language-specific large language models (LLMs), with a particular focus on the Thai LLM. Our goal is to enhance the reasoning capabilities of language-specific LLMs while maintaining their target language abilities. DeepSeek R1 excels in reasoning but primarily benefits high-resource languages such as English and Chinese. However, low-resource languages remain underserved due to the dominance of English-centric training data and model optimizations, which limit performance in these languages. This limitation results in unreliable code-switching and diminished effectiveness on tasks in low-resource languages. Meanwhile, local and regional LLM initiatives have attempted to bridge this gap by developing language-specific LLMs that focus on improving local linguistic fidelity. We demonstrate that, with only publicly available datasets and a computational budget of $120, it is possible to enhance the reasoning capabilities of language-specific LLMs to match the level of DeepSeek R1, without compromising their performance on target language tasks. 

**Abstract (ZH)**: 本文探讨了数据选择和模型融合方法，旨在将先进的推理能力（如DeepSeek R1的能力）融入语言特定的大语言模型（LLMs）中，特别是针对泰语LLMs。我们的目标是在增强语言特定LLMs的推理能力的同时，保持其目标语言的能力。DeepSeek R1在推理方面表现出色，但主要受益于英语和汉语等高资源语言。然而，由于英语为中心的训练数据和模型优化的主导地位，低资源语言仍缺乏支持，这限制了这些语言的性能。这一限制导致在低资源语言中的代码转换不可靠，且在这些语言上的任务效果减弱。与此同时，本地和区域LLMs项目试图通过开发专注于提升本地语言忠实度的语言特定LLMs来弥补这一差距。我们展示了，在仅使用公开可用的数据集和计算预算为120美元的情况下，有可能增强语言特定LLMs的推理能力，使其与DeepSeek R1相当，同时不牺牲其在目标语言任务上的性能。 

---
# Typhoon T1: An Open Thai Reasoning Model 

**Title (ZH)**: 台风 T1：一个开放的泰国推理模型 

**Authors**: Pittawat Taveekitworachai, Potsawee Manakul, Kasima Tharnpipitchai, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2502.09042)  

**Abstract**: This paper introduces Typhoon T1, an open effort to develop an open Thai reasoning model. A reasoning model is a relatively new type of generative model built on top of large language models (LLMs). A reasoning model generates a long chain of thought before arriving at a final answer, an approach found to improve performance on complex tasks. However, details on developing such a model are limited, especially for reasoning models that can generate traces in a low-resource language. Typhoon T1 presents an open effort that dives into the details of developing a reasoning model in a more cost-effective way by leveraging supervised fine-tuning using open datasets, instead of reinforcement learning. This paper shares the details about synthetic data generation and training, as well as our dataset and model weights. Additionally, we provide insights gained from developing a reasoning model that generalizes across domains and is capable of generating reasoning traces in a low-resource language, using Thai as an example. We hope this open effort provides a foundation for further research in this field. 

**Abstract (ZH)**: 本文介绍了Typhoon T1，这是一个开放的努力，旨在开发一个开源的泰语推理模型。推理模型是一种基于大规模语言模型（LLMs）的新类型的生成模型。该模型在得出最终答案之前会生成一条长时间的思考链，这种方法被发现能够提高在复杂任务上的性能。然而，关于此类模型的开发细节较少，特别是一些能够生成低资源语言推理轨迹的推理模型。Typhoon T1 提出了一种更经济有效的开发推理模型的方法，通过利用开放数据集进行监督微调，而不是使用强化学习。本文分享了合成数据生成和训练的详细信息，以及我们的数据集和模型权重。此外，我们还提供了关于在跨领域领域通用并能够生成低资源语言推理轨迹的推理模型开发过程中获得的见解，以泰语为例。我们希望这一开放努力能够为该领域的进一步研究奠定基础。 

---
# Diversity Enhances an LLM's Performance in RAG and Long-context Task 

**Title (ZH)**: 多样性可以提升大语言模型在检索增强和长上下文任务中的性能 

**Authors**: Zhchao Wang, Bin Bi, Yanqi Luo, Sitaram Asur, Claire Na Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09017)  

**Abstract**: The rapid advancements in large language models (LLMs) have highlighted the challenge of context window limitations, primarily due to the quadratic time complexity of the self-attention mechanism (\(O(N^2)\), where \(N\) denotes the context window length). This constraint impacts tasks such as retrieval-augmented generation (RAG) in question answering (Q\&A) and long context summarization. A common approach involves selecting content with the highest similarity to the query; however, this often leads to redundancy and the exclusion of diverse yet relevant information. Building on principles from Maximal Marginal Relevance (MMR) and Farthest Point Sampling (FPS), we integrate diversity into the content selection process. Our findings reveal that incorporating diversity substantially increases the recall of selecting relevant sentences or chunks before LLM-based Q\&A and summarization. These results highlight the importance of maintaining diversity in future LLM applications to further improve summarization and Q\&A outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展凸显了上下文窗口限制所面临的挑战，主要是由于自我注意机制的时间复杂度为二次复杂度（\(O(N^2)\)，其中\(N\)表示上下文窗口的长度）。这一限制影响了诸如问题回答（Q&A）中的检索增强生成（RAG）和长文本摘要等任务。一种常见的方法是选择与查询最相似的内容；然而，这种方法往往会导致冗余，并排除那些虽然相关但多样性的信息。基于最大化边际相关性（MMR）和最远点采样（FPS）的原则，我们在内容选择过程中引入了多样性。我们的研究发现，引入多样性显著提高了在LLM驱动的问题回答和摘要任务中选择相关句子或段落的召回率。这些结果强调了在未来LLM应用中维持多样性的必要性，以进一步提高摘要和问题回答的效果。 

---
# Medicine on the Edge: Comparative Performance Analysis of On-Device LLMs for Clinical Reasoning 

**Title (ZH)**: 边缘医疗：边缘设备上大型语言模型在临床推理中的性能比较分析 

**Authors**: Leon Nissen, Philipp Zagar, Vishnu Ravi, Aydin Zahedivash, Lara Marie Reimer, Stephan Jonas, Oliver Aalami, Paul Schmiedmayer  

**Link**: [PDF](https://arxiv.org/pdf/2502.08954)  

**Abstract**: The deployment of Large Language Models (LLM) on mobile devices offers significant potential for medical applications, enhancing privacy, security, and cost-efficiency by eliminating reliance on cloud-based services and keeping sensitive health data local. However, the performance and accuracy of on-device LLMs in real-world medical contexts remain underexplored. In this study, we benchmark publicly available on-device LLMs using the AMEGA dataset, evaluating accuracy, computational efficiency, and thermal limitation across various mobile devices. Our results indicate that compact general-purpose models like Phi-3 Mini achieve a strong balance between speed and accuracy, while medically fine-tuned models such as Med42 and Aloe attain the highest accuracy. Notably, deploying LLMs on older devices remains feasible, with memory constraints posing a greater challenge than raw processing power. Our study underscores the potential of on-device LLMs for healthcare while emphasizing the need for more efficient inference and models tailored to real-world clinical reasoning. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

在移动设备上部署大规模语言模型（LLM）为医疗应用提供了巨大潜力，通过消除对云服务的依赖，保护敏感健康数据的隐私、安全并提高成本效益。然而，实世界医疗情境下设备端LLM的性能和准确性尚未得到充分探索。本研究使用AMEGA数据集对公开可用的设备端LLM进行基准测试，评估其在各种移动设备上的准确性、计算效率和热限制。结果显示，紧凑型通用模型如Phi-3 Mini在速度和准确性之间实现了良好的平衡，而经过医学微调的模型如Med42和Aloe则实现了最高的准确性。值得注意的是，部署LLM在较旧的设备上仍然可行，内存约束是更大的挑战，而不是原始计算能力。本研究强调了设备端LLM在医疗保健领域中的潜力，同时也突出了需要更高效的推理算法和更适合实际临床推理的模型的必要性。 

---
# Beyond the Singular: The Essential Role of Multiple Generations in Effective Benchmark Evaluation and Analysis 

**Title (ZH)**: 超越单一视角：多代参与在有效基准评估与分析中的本质作用 

**Authors**: Wenbo Zhang, Hengrui Cai, Wenyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.08943)  

**Abstract**: Large language models (LLMs) have demonstrated significant utilities in real-world applications, exhibiting impressive capabilities in natural language processing and understanding. Benchmark evaluations are crucial for assessing the capabilities of LLMs as they can provide a comprehensive assessment of their strengths and weaknesses. However, current evaluation methods often overlook the inherent randomness of LLMs by employing deterministic generation strategies or relying on a single random sample, resulting in unaccounted sampling variance and unreliable benchmark score estimates. In this paper, we propose a hierarchical statistical model that provides a more comprehensive representation of the benchmarking process by incorporating both benchmark characteristics and LLM randomness. We show that leveraging multiple generations improves the accuracy of estimating the benchmark score and reduces variance. We also introduce $\mathbb P\left(\text{correct}\right)$, a prompt-level difficulty score based on correct ratios, providing fine-grained insights into individual prompts. Additionally, we create a data map that visualizes difficulty and semantic prompts, enabling error detection and quality control in benchmark construction. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在实际应用中展示了重要的实用价值，表现出在自然语言处理和理解方面的强大能力。基准评估对于评估LLMs的能力至关重要，因为它能够提供对其优缺点的全面评估。然而，当前的评估方法往往忽视了LLMs固有的随机性，通过采用确定性的生成策略或依赖单一的随机样本来进行评估，导致未考虑的抽样方差和不可靠的基准分数估计。在本文中，我们提出了一种分层统计模型，通过结合基准特性和LLMs的随机性，提供了一个更为全面的基准评估过程的表示方法。我们证明了利用多次生成可以提高基准分数估计的准确性并减少方差。此外，我们引入了基于正确率的提示级难度评分 $\mathbb{P}(\text{correct})$，提供对单个提示的精细洞察。我们还创建了一组数据地图，可视化了难度和语义提示，从而有助于基准构建中的错误检测和质量控制。 

---
# InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU 

**Title (ZH)**: InfiniteHiP：将语言模型上下文扩展至单个GPU上的300万词 Escorts 

**Authors**: Heejun Lee, Geon Park, Jaduk Suh, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08910)  

**Abstract**: In modern large language models (LLMs), handling very long context lengths presents significant challenges as it causes slower inference speeds and increased memory costs. Additionally, most existing pre-trained LLMs fail to generalize beyond their original training sequence lengths. To enable efficient and practical long-context utilization, we introduce InfiniteHiP, a novel, and practical LLM inference framework that accelerates processing by dynamically eliminating irrelevant context tokens through a modular hierarchical token pruning algorithm. Our method also allows generalization to longer sequences by selectively applying various RoPE adjustment methods according to the internal attention patterns within LLMs. Furthermore, we offload the key-value cache to host memory during inference, significantly reducing GPU memory pressure. As a result, InfiniteHiP enables the processing of up to 3 million tokens on a single L40s 48GB GPU -- 3x larger -- without any permanent loss of context information. Our framework achieves an 18.95x speedup in attention decoding for a 1 million token context without requiring additional training. We implement our method in the SGLang framework and demonstrate its effectiveness and practicality through extensive evaluations. 

**Abstract (ZH)**: 在现代大型语言模型（LLMs）中，处理非常长的上下文长度带来了显著的挑战，因为它会导致推理速度变慢并增加内存成本。此外，大多数现有的预训练LLMs无法在其原始训练序列长度之外泛化。为了实现高效且实用的长上下文利用，我们引入了InfiniteHiP，这是一种新颖且实用的LLM推理框架，通过动态消除无关的上下文标记来加速处理，使用了一个模块化的分层标记剪枝算法。我们的方法还通过根据LLM内部注意力模式选择性地应用各种RoPE调整方法，实现了更长序列的泛化。此外，在推理过程中，我们将键值缓存卸载到主机内存中，大幅减少了GPU内存压力。因此，InfiniteHiP使得使用单个L40s 48GB GPU处理多达300万标记成为可能，相当于扩大了三倍——而无任何永久性的上下文信息损失。我们的框架在100万标记的上下文情况下实现了18.95倍的注意力解码加速，而无需额外的训练。我们已在SGLang框架中实现了此方法，并通过广泛的评估充分展示了其实效性和实用性。 

---
# Towards Automated Fact-Checking of Real-World Claims: Exploring Task Formulation and Assessment with LLMs 

**Title (ZH)**: 面向现实世界声明的自动事实核查：利用大语言模型探索任务表述与评估方法 

**Authors**: Premtim Sahitaj, Iffat Maab, Junichi Yamagishi, Jawan Kolanowski, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2502.08909)  

**Abstract**: Fact-checking is necessary to address the increasing volume of misinformation. Traditional fact-checking relies on manual analysis to verify claims, but it is slow and resource-intensive. This study establishes baseline comparisons for Automated Fact-Checking (AFC) using Large Language Models (LLMs) across multiple labeling schemes (binary, three-class, five-class) and extends traditional claim verification by incorporating analysis, verdict classification, and explanation in a structured setup to provide comprehensive justifications for real-world claims. We evaluate Llama-3 models of varying sizes (3B, 8B, 70B) on 17,856 claims collected from PolitiFact (2007-2024) using evidence retrieved via restricted web searches. We utilize TIGERScore as a reference-free evaluation metric to score the justifications. Our results show that larger LLMs consistently outperform smaller LLMs in classification accuracy and justification quality without fine-tuning. We find that smaller LLMs in a one-shot scenario provide comparable task performance to fine-tuned Small Language Models (SLMs) with large context sizes, while larger LLMs consistently surpass them. Evidence integration improves performance across all models, with larger LLMs benefiting most. Distinguishing between nuanced labels remains challenging, emphasizing the need for further exploration of labeling schemes and alignment with evidences. Our findings demonstrate the potential of retrieval-augmented AFC with LLMs. 

**Abstract (ZH)**: 事实核查对于应对不断增加的虚假信息至关重要。传统的事实核查方法依赖于人工分析来验证声明，但这种方法耗时且资源消耗大。本研究通过在多个标签方案（二分类、三分类、五分类）下使用大型语言模型（LLMs）建立了自动事实核查（AFC）的基准比较，并通过在结构化的设置中结合分析、裁决分类和解释，扩展了传统的声明验证方法，以提供全面的现实声明的正当说明。我们使用了从Politifact（2007-2024年）中收集的17,856个声明，并通过限制性网络搜索获取证据进行评估。我们利用TIGERScore作为参考自由评估指标来评估正当说明的质量。结果显示，在分类准确性和正当说明质量方面，较大的LLMs在无需微调的情况下始终优于较小的LLMs。我们发现，在单次适应场景中，较小的LLMs的任务性能与大型上下文的微调小型语言模型（SLMs）相当，而较大的LLMs在所有模型中始终表现出更优的表现。证据集成在所有模型中提高了性能，而较大的LLMs从中受益最大。区分细微的标签仍然是一个挑战，强调了进一步探索标签方案和证据一致性的重要性。我们的研究结果表明，LLMs辅助的检索增强自动事实核查具有潜在的可能性。 

---
# Communication is All You Need: Persuasion Dataset Construction via Multi-LLM Communication 

**Title (ZH)**: 你需要的只是沟通：通过多轮多模态语言模型对话构建说服数据集 

**Authors**: Weicheng Ma, Hefan Zhang, Ivory Yang, Shiyu Ji, Joice Chen, Farnoosh Hashemi, Shubham Mohole, Ethan Gearey, Michael Macy, Saeed Hassanpour, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2502.08896)  

**Abstract**: Large Language Models (LLMs) have shown proficiency in generating persuasive dialogue, yet concerns about the fluency and sophistication of their outputs persist. This paper presents a multi-LLM communication framework designed to enhance the generation of persuasive data automatically. This framework facilitates the efficient production of high-quality, diverse linguistic content with minimal human oversight. Through extensive evaluations, we demonstrate that the generated data excels in naturalness, linguistic diversity, and the strategic use of persuasion, even in complex scenarios involving social taboos. The framework also proves adept at generalizing across novel contexts. Our results highlight the framework's potential to significantly advance research in both computational and social science domains concerning persuasive communication. 

**Abstract (ZH)**: 大语言模型（LLMs）在生成说服性对话方面展现了能力，但对其输出的流畅性和 sophistication（复杂性）仍存在担忧。本文提出了一种多LLM通信框架，旨在提升自动生成说服性数据的能力。该框架能够高效生产高质量、多样性的语言内容，并且需要极少的人工监督。通过广泛的评估，我们证明生成的数据在自然性、语言多样性以及策略性使用说服方面表现出色，即使在涉及社会禁忌的复杂场景中也是如此。该框架还证明了其在新情境下的泛化能力。我们的结果突显了该框架在计算和社会科学领域中关于说服性沟通研究方面的巨大潜力。 

---
# LLM-Enhanced Multiple Instance Learning for Joint Rumor and Stance Detection with Social Context Information 

**Title (ZH)**: 增强型多实例学习：结合社交背景信息的谣言检测与立场检测联合模型 

**Authors**: Ruichao Yang, Jing Ma, Wei Gao, Hongzhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.08888)  

**Abstract**: The proliferation of misinformation, such as rumors on social media, has drawn significant attention, prompting various expressions of stance among users. Although rumor detection and stance detection are distinct tasks, they can complement each other. Rumors can be identified by cross-referencing stances in related posts, and stances are influenced by the nature of the rumor. However, existing stance detection methods often require post-level stance annotations, which are costly to obtain. We propose a novel LLM-enhanced MIL approach to jointly predict post stance and claim class labels, supervised solely by claim labels, using an undirected microblog propagation model. Our weakly supervised approach relies only on bag-level labels of claim veracity, aligning with multi-instance learning (MIL) principles. To achieve this, we transform the multi-class problem into multiple MIL-based binary classification problems. We then employ a discriminative attention layer to aggregate the outputs from these classifiers into finer-grained classes. Experiments conducted on three rumor datasets and two stance datasets demonstrate the effectiveness of our approach, highlighting strong connections between rumor veracity and expressed stances in responding posts. Our method shows promising performance in joint rumor and stance detection compared to the state-of-the-art methods. 

**Abstract (ZH)**: 社交媒体上传播的虚假信息（如谣言）引起了广泛关注，用户们也在此过程中表达了各自的立场。尽管谣言检测和立场检测是两个独立的任务，但它们可以相互补充。谣言可以通过在相关帖子中进行交叉参考识别，而立场又受到谣言性质的影响。然而，现有的立场检测方法通常需要帖子级别的立场标注，这在获取成本上较高。我们提出了一种新的增强型多实例学习（MIL）方法，通过一个无向微观博客传播模型，仅使用声明标签的监督，共同预测帖子立场和声明类别标签。我们的弱监督方法仅依赖于声明真实性的包级标签，这符合多实例学习（MIL）的原则。为此，我们将多类别问题转化为多个基于MIL的二元分类问题。然后，我们使用一个区分性注意力层来汇总这些分类器的输出，生成更精细的类别。在三个谣言数据集和两个立场数据集上的实验表明了我们方法的有效性，突出了谣言真实性与回应帖子中表达的立场之间的紧密联系。与最新方法相比，我们的方法在联合谣言和立场检测方面表现出有希望的性能。 

---
# If Multi-Agent Debate is the Answer, What is the Question? 

**Title (ZH)**: 如果多智能体辩论是答案，那么问题是什么？ 

**Authors**: Hangfan Zhang, Zhiyao Cui, Xinrun Wang, Qiaosheng Zhang, Zhen Wang, Dinghao Wu, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08788)  

**Abstract**: Multi-agent debate (MAD) has emerged as a promising approach to enhance the factual accuracy and reasoning quality of large language models (LLMs) by engaging multiple agents in iterative discussions during inference. Despite its potential, we argue that current MAD research suffers from critical shortcomings in evaluation practices, including limited dataset overlap and inconsistent baselines, raising significant concerns about generalizability. Correspondingly, this paper presents a systematic evaluation of five representative MAD methods across nine benchmarks using four foundational models. Surprisingly, our findings reveal that MAD methods fail to reliably outperform simple single-agent baselines such as Chain-of-Thought and Self-Consistency, even when consuming additional inference-time computation. From our analysis, we found that model heterogeneity can significantly improve MAD frameworks. We propose Heter-MAD enabling a single LLM agent to access the output from heterogeneous foundation models, which boosts the performance of current MAD frameworks. Finally, we outline potential directions for advancing MAD, aiming to spark a broader conversation and inspire future work in this area. 

**Abstract (ZH)**: 多智能体辩论（MAD）已成为通过在推理过程中让多个智能体进行迭代讨论来增强大型语言模型（LLMs）的事实准确性及推理质量的一种有前景的方法。尽管具有潜力，但我们认为当前的MAD研究在评估实践方面存在严重的不足，包括数据集重叠有限以及基线不一致，这严重地引发了对其普遍适用性的担忧。相应地，本文对五种代表性的MAD方法在九个基准上进行了系统性评估，使用了四种基础模型。令人惊讶的是，我们的发现表明，即使消耗了更多的推理时间计算，MAD方法也无法可靠地超越简单的单智能体基线，如Chain-of-Thought和Self-Consistency。通过对这些结果的分析，我们发现模型异质性可以显著改善MAD框架。我们提出了Heter-MAD，允许单一LLM智能体访问来自不同基础模型的输出，从而提升当前MAD框架的性能。最后，我们概述了MAD未来发展的一些潜在方向，旨在激发更广泛的讨论并激发未来在此领域的研究工作。 

---
# Zero-Shot Belief: A Hard Problem for LLMs 

**Title (ZH)**: 零样本信念推断：LLM面临的难题 

**Authors**: John Murzaku, Owen Rambow  

**Link**: [PDF](https://arxiv.org/pdf/2502.08777)  

**Abstract**: We present two LLM-based approaches to zero-shot source-and-target belief prediction on FactBank: a unified system that identifies events, sources, and belief labels in a single pass, and a hybrid approach that uses a fine-tuned DeBERTa tagger for event detection. We show that multiple open-sourced, closed-source, and reasoning-based LLMs struggle with the task. Using the hybrid approach, we achieve new state-of-the-art results on FactBank and offer a detailed error analysis. Our approach is then tested on the Italian belief corpus ModaFact. 

**Abstract (ZH)**: 我们提出了两种基于大规模语言模型（LLM）的方法来零样本预测FactBank中的源方和目标方信念：一种统一系统，在单一通过过程中识别事件、来源和信念标签，以及一种混合方法，该方法使用微调后的DeBERTa标注器进行事件检测。我们展示了多种开源、封闭源和基于推理的LLM在这项任务上面临挑战。通过混合方法，我们在FactBank上取得了新的最佳结果，并提供了详细的错误分析。随后，我们将在意大利信念语料库ModaFact上测试我们的方法。 

---
# Universal Model Routing for Efficient LLM Inference 

**Title (ZH)**: 高效语言模型推理的通用模型路由方法 

**Authors**: Wittawat Jitkrittum, Harikrishna Narasimhan, Ankit Singh Rawat, Jeevesh Juneja, Zifeng Wang, Chen-Yu Lee, Pradeep Shenoy, Rina Panigrahy, Aditya Krishna Menon, Sanjiv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.08773)  

**Abstract**: Large language models' significant advances in capabilities are accompanied by significant increases in inference costs. Model routing is a simple technique for reducing inference cost, wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose a new approach to this problem that relies on representing each LLM as a feature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective strategies, relying on cluster-based routing and a learned cluster map respectively. We prove that these strategies are estimates of a theoretically optimal routing rule, and provide an excess risk bound to quantify their errors. Experiments on a range of public benchmarks show the effectiveness of the proposed strategies in routing amongst more than 30 unseen LLMs. 

**Abstract (ZH)**: 大语言模型的能力显著提升伴随着推理成本的显著增加。模型路由是一种降低推理成本的简单技术，其中通过维护一个候选模型池，并学会将每个提示分配到最小可行的模型来实现。现有工作主要关注为固定模型池训练一个路由器。本文则考虑一种动态路由问题，在测试时可以使用新的、之前未见过的模型。我们提出了一种新的方法，该方法通过将每个模型表示为基于一组代表性提示预测得到的特征向量来进行。基于此，我们详细介绍了两种有效的策略，一种是基于聚类路由，另一种是基于学习得到的聚类图。我们证明了这些策略是理论上最优路由规则的估计，并提供了超出风险界的量化来衡量它们的误差。实验结果表明，在30多个未见过的模型中进行路由时，所提出的策略具有有效性。 

---
# IHEval: Evaluating Language Models on Following the Instruction Hierarchy 

**Title (ZH)**: IHEval：评估语言模型遵循指令层级的能力 

**Authors**: Zhihan Zhang, Shiyang Li, Zixuan Zhang, Xin Liu, Haoming Jiang, Xianfeng Tang, Yifan Gao, Zheng Li, Haodong Wang, Zhaoxuan Tan, Yichuan Li, Qingyu Yin, Bing Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08745)  

**Abstract**: The instruction hierarchy, which establishes a priority order from system messages to user messages, conversation history, and tool outputs, is essential for ensuring consistent and safe behavior in language models (LMs). Despite its importance, this topic receives limited attention, and there is a lack of comprehensive benchmarks for evaluating models' ability to follow the instruction hierarchy. We bridge this gap by introducing IHEval, a novel benchmark comprising 3,538 examples across nine tasks, covering cases where instructions in different priorities either align or conflict. Our evaluation of popular LMs highlights their struggle to recognize instruction priorities. All evaluated models experience a sharp performance decline when facing conflicting instructions, compared to their original instruction-following performance. Moreover, the most competitive open-source model only achieves 48% accuracy in resolving such conflicts. Our results underscore the need for targeted optimization in the future development of LMs. 

**Abstract (ZH)**: 指令层级结构，它确立了从系统消息、用户消息、对话历史到工具输出的优先顺序，对于确保语言模型（LM）的一致性和安全性至关重要。尽管该话题非常重要，但其研究相对较少，缺乏全面的基准测试来评估模型遵循指令层级结构的能力。为填补这一空白，我们引入了IHEval，这是一个包含3,538个示例的新型基准，覆盖了不同优先级指令既一致又冲突的各种情况。对流行LMs的评估显示，它们在识别指令优先级方面存在困难。面对冲突指令时，所有评估模型的表现急剧下降，与它们原本遵循指令时的表现相比，下降尤其明显。此外，最优秀的开源模型在解决此类冲突时的准确率仅为48%。我们的结果强调了未来LM开发中针对这一问题进行优化的必要性。 

---
# Hallucination, Monofacts, and Miscalibration: An Empirical Investigation 

**Title (ZH)**: 幻觉、单事实陈述与校准偏差：一项实证研究 

**Authors**: Muqing Miao, Michael Kearns  

**Link**: [PDF](https://arxiv.org/pdf/2502.08666)  

**Abstract**: Recent theoretical work by [Kalai and Vempala 2024] proves that a particular notion of hallucination rate in LLMs must be lower bounded by the training data monofact rate (related to the classical Good-Turing missing mass estimator) minus model miscalibration. Through systematic experiments with n-gram models and in-context learning with LLMs, we empirically investigate and validate this theory by examining how different underlying data distributions affect the monofact rate and a model's tendency to hallucinate. We then vary model miscalibration through controlled upweighting of training samples while holding monofact rates constant, allowing us to isolate miscalibration's reduction effect on hallucination. These findings suggest that both the distribution of fact frequencies in training data and the calibration-hallucination trade-off are inherent to probabilistic language generation. Our results also suggest that current practices of aggressive deduplication in training data may need to be reconsidered, as selective duplication could serve as a principled mechanism for reducing hallucination. 

**Abstract (ZH)**: 以下是学术规范的中文翻译：

近期，[Kalai 和 Vempala 2024] 的理论工作证明，LLM 中的特定幻觉率必须被下限约束，这个下限等于训练数据中单一事实率（与经典的 Good-Turing 缺失质量估算法有关）减去模型的偏差度。通过系统地使用 n-克模型和基于上下文的 LLM 实验，我们通过研究不同的数据分布如何影响单一事实率和模型产生幻觉的倾向来验证这一理论。随后，我们通过控制性地增加训练样本的权重同时保持单一事实率不变，来改变模型的偏差度，从而分离出偏差度对幻觉减少效应的影响。这些发现表明，训练数据中事实频率的分布以及偏差度与幻觉之间的权衡都是概率语言生成固有的特征。我们的研究结果还表明，当前训练数据中强烈的去重实践可能需要重新考虑，因为有选择地重复数据可能是减少幻觉的一种原理机制。 

---
# Refining Positive and Toxic Samples for Dual Safety Self-Alignment of LLMs with Minimal Human Interventions 

**Title (ZH)**: 减少人类干预以精炼正向和有毒样本，实现大规模语言模型的双重安全自我对齐 

**Authors**: Jingxin Xu, Guoshun Nan, Sheng Guan, Sicong Leng, Yilian Liu, Zixiao Wang, Yuyang Ma, Zhili Zhou, Yanzhao Hou, Xiaofeng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.08657)  

**Abstract**: Recent AI agents, such as ChatGPT and LLaMA, primarily rely on instruction tuning and reinforcement learning to calibrate the output of large language models (LLMs) with human intentions, ensuring the outputs are harmless and helpful. Existing methods heavily depend on the manual annotation of high-quality positive samples, while contending with issues such as noisy labels and minimal distinctions between preferred and dispreferred response data. However, readily available toxic samples with clear safety distinctions are often filtered out, removing valuable negative references that could aid LLMs in safety alignment. In response, we propose PT-ALIGN, a novel safety self-alignment approach that minimizes human supervision by automatically refining positive and toxic samples and performing fine-grained dual instruction tuning. Positive samples are harmless responses, while toxic samples deliberately contain extremely harmful content, serving as a new supervisory signals. Specifically, we utilize LLM itself to iteratively generate and refine training instances by only exploring fewer than 50 human annotations. We then employ two losses, i.e., maximum likelihood estimation (MLE) and fine-grained unlikelihood training (UT), to jointly learn to enhance the LLM's safety. The MLE loss encourages an LLM to maximize the generation of harmless content based on positive samples. Conversely, the fine-grained UT loss guides the LLM to minimize the output of harmful words based on negative samples at the token-level, thereby guiding the model to decouple safety from effectiveness, directing it toward safer fine-tuning objectives, and increasing the likelihood of generating helpful and reliable content. Experiments on 9 popular open-source LLMs demonstrate the effectiveness of our PT-ALIGN for safety alignment, while maintaining comparable levels of helpfulness and usefulness. 

**Abstract (ZH)**: 近年来，诸如ChatGPT和LLaMA之类的AI代理主要依赖于指令微调和强化学习来校准大语言模型（LLMs）的输出以符合人类意图，从而确保输出既无害又有益。现有方法高度依赖于高质量正样本的手动标注，同时面临标签噪声大和偏好响应与非偏好响应数据区分度小等问题。然而，容易获得的具有明确安全区分的有毒样本往往被过滤掉，从而消除了能够帮助LLMs实现安全对齐的重要负样本参考。为应对这一问题，我们提出了一种新颖的安全自我对齐方法——PT-ALIGN，该方法通过自动优化正样本和有毒样本并进行细粒度的双指令微调来最大限度地减少人类监督。正样本为无害的响应，而有毒样本故意包含极端有害的内容，作为新的监督信号。具体而言，我们利用LLM本身通过探索不到50个人标注实例进行迭代生成和优化训练实例。然后，我们使用两种损失，即最大似然估计（MLE）和细粒度的不可能性训练（UT），以联合学习方式提升LLM的安全性。MLE损失鼓励LLM根据正样本生成尽可能多的无害内容。相反，细粒度的UT损失在 token 级别引导LLM减少有害词汇的输出，从而指导模型安全性和效果脱钩，使其朝着更安全的微调目标发展，并增加生成有益和可靠内容的可能性。实验表明，与9个流行的开源LLM相比，PT-ALIGN在安全对齐方面表现出有效性，同时保持了相近的有益性和实用性。 

---
# MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency 

**Title (ZH)**: MME-CoT：大型多模态模型中思维链方法的基准测试，用于评估推理质量、稳健性与效率 

**Authors**: Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, Bo Zhang, Chaoyou Fu, Peng Gao, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.09621)  

**Abstract**: Answering questions with Chain-of-Thought (CoT) has significantly enhanced the reasoning capabilities of Large Language Models (LLMs), yet its impact on Large Multimodal Models (LMMs) still lacks a systematic assessment and in-depth investigation. In this paper, we introduce MME-CoT, a specialized benchmark evaluating the CoT reasoning performance of LMMs, spanning six domains: math, science, OCR, logic, space-time, and general scenes. As the first comprehensive study in this area, we propose a thorough evaluation suite incorporating three novel metrics that assess the reasoning quality, robustness, and efficiency at a fine-grained level. Leveraging curated high-quality data and a unique evaluation strategy, we conduct an in-depth analysis of state-of-the-art LMMs, uncovering several key insights: 1) Models with reflection mechanism demonstrate a superior CoT quality, with Kimi k1.5 outperforming GPT-4o and demonstrating the highest quality results; 2) CoT prompting often degrades LMM performance on perception-heavy tasks, suggesting a potentially harmful overthinking behavior; and 3) Although the CoT quality is high, LMMs with reflection exhibit significant inefficiency in both normal response and self-correction phases. We hope MME-CoT serves as a foundation for advancing multimodal reasoning in LMMs. Project Page: this https URL 

**Abstract (ZH)**: 使用因果链（CoT）回答问题显著提高了大型语言模型（LLMs）的推理能力，然而其对大型多模态模型（LMMs）的影响仍然缺乏系统性评估和深入研究。本文中，我们提出了MME-CoT，这是一个专门用于评估LMMs因果链推理性能的基准测试，涵盖了六个领域：数学、科学、光学字符识别（OCR）、逻辑、时间和空间、以及一般场景。作为该领域的首个综合研究，我们提出了一套全面的评估方案，其中包括三个新的评估指标，以细粒度评估推理质量、稳健性和效率。通过利用精选的高质量数据和独特的评估策略，我们深入分析了最先进的LMMs，发现了以下几个关键洞察：1）具有反思机制的模型显示出了更高质量的CoT推理，Kimi k1.5的表现优于GPT-4o，展示了最高的推理质量；2）CoT提示在感知密集型任务上往往会降低LMMs的表现，这表明可能存在有害的过度思考行为；3）尽管CoT质量很高，具有反思机制的LMMs在正常响应和自我修正阶段都表现出显著的效率低下。我们希望MME-CoT能够为推动LMMs中的多模态推理提供一个基础。项目页面：[这个链接] 

---
# CoT-Valve: Length-Compressible Chain-of-Thought Tuning 

**Title (ZH)**: CoT-Valve：长度可压缩的思维链调整方法 

**Authors**: Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09601)  

**Abstract**: Chain-of-Thought significantly enhances a model's reasoning capability, but it also comes with a considerable increase in inference costs due to long chains. With the observation that the reasoning path can be easily compressed under easy tasks but struggle on hard tasks, we explore the feasibility of elastically controlling the length of reasoning paths with only one model, thereby reducing the inference overhead of reasoning models dynamically based on task difficulty. We introduce a new tuning and inference strategy named CoT-Valve, designed to allow models to generate reasoning chains of varying lengths. To achieve this, we propose to identify a direction in the parameter space that, when manipulated, can effectively control the length of generated CoT. Moreover, we show that this property is valuable for compressing the reasoning chain. We construct datasets with chains from long to short for the same questions and explore two enhanced strategies for CoT-Valve: (1) a precise length-compressible CoT tuning method, and (2) a progressive chain length compression approach. Our experiments show that CoT-Valve successfully enables controllability and compressibility of the chain and shows better performance than the prompt-based control. We applied this method to QwQ-32B-Preview, reducing reasoning chains on GSM8K from 741 to 225 tokens with a minor performance drop (95.07% to 94.92%) and on AIME from 6827 to 4629 tokens, with only one additional incorrect answer. 

**Abstract (ZH)**: 链式思考显著提升了模型的推理能力，但同时也带来了由于长链带来的推理成本的大幅增加。观察到推理路径在简单任务中可以轻松压缩，但在困难任务中则充满挑战。基于此，我们探索了一种通过单一模型弹性控制推理路径长度的可能性，从而根据任务难度动态减少推理模型的推理开销。我们提出了一种新的调参和推理策略——CoT-阀值（CoT-Valve），旨在允许模型生成长度可变的推理链。为此，我们提出了在参数空间中识别一个方向，调节该方向可以有效地控制生成的链式思考（CoT）的长度。此外，我们展示了这一特性对于压缩推理链的有效性。我们构建了从长到短的同题数据集，并探索了CoT-阀值的两种增强策略：（1）精确长度可压缩的CoT调参方法，（2）逐步链式思考长度压缩方法。实验结果显示，CoT-阀值成功实现了链式思考的可控性和压缩性，并且在性能方面优于基于提示的控制方法。我们将其应用于QwQ-32B-Preview，将GSM8K中的推理链从741个词元减少到225个词元，性能仅略有下降（95.07%降至94.92%），并在AIME上将词元数从6827减少到4629，仅增加了一个错误答案。 

---
# Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs 

**Title (ZH)**: 大型语言模型能够识别您的偏好吗？评估大型语言模型中的个性化偏好跟随能力 

**Authors**: Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, Kaixiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09597)  

**Abstract**: Large Language Models (LLMs) are increasingly used as chatbots, yet their ability to personalize responses to user preferences remains limited. We introduce PrefEval, a benchmark for evaluating LLMs' ability to infer, memorize and adhere to user preferences in a long-context conversational setting. PrefEval comprises 3,000 manually curated user preference and query pairs spanning 20 topics. PrefEval contains user personalization or preference information in both explicit and implicit forms, and evaluates LLM performance using a generation and a classification task. With PrefEval, we evaluated the aforementioned preference following capabilities of 10 open-source and proprietary LLMs in multi-session conversations with varying context lengths up to 100k tokens. We benchmark with various prompting, iterative feedback, and retrieval-augmented generation methods. Our benchmarking effort reveals that state-of-the-art LLMs face significant challenges in proactively following users' preferences during conversations. In particular, in zero-shot settings, preference following accuracy falls below 10% at merely 10 turns (~3k tokens) across most evaluated models. Even with advanced prompting and retrieval methods, preference following still deteriorates in long-context conversations. Furthermore, we show that fine-tuning on PrefEval significantly improves performance. We believe PrefEval serves as a valuable resource for measuring, understanding, and enhancing LLMs' preference following abilities, paving the way for personalized conversational agents. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用作聊天机器人，但它们在个性化响应以适应用户偏好方面的能力仍然有限。我们引入了PrefEval基准，用于评估LLMs在长上下文对话环境中推断、记忆和遵循用户偏好的能力。PrefEval包含3,000个由人工精心挑选的用户偏好和查询对，覆盖20个主题。PrefEval中包含用户的个性化或偏好信息，既有显式的也有隐含的形式，并使用生成任务和分类任务评估LLMs的表现。通过PrefEval，我们评估了10种开源和专有LLMs在多会话对话中，不同上下文字长（最多100,000个标记）的偏好跟随能力。我们采用了各种提示、迭代反馈和检索增强生成方法作为基准测试。我们的基准测试工作揭示了最先进的LLMs在对话过程中主动跟随用户偏好方面面临显著挑战。特别是在零样本设置中，偏好跟随的准确率在仅仅10个回合（约3,000个标记）后，大多数评估模型的准确率低于10%。即使使用先进的提示和检索方法，在长上下文对话中偏好跟随能力也有所下降。此外，我们展示了在PrefEval上进行微调显著提高了性能。我们认为PrefEval是一个有价值的资源，可用于测量、理解和增强LLMs的偏好跟随能力，为个性化对话代理铺平道路。我们的代码和数据集可在此处访问：[相关链接]。 

---
# Reliable Conversational Agents under ASP Control that Understand Natural Language 

**Title (ZH)**: 在逻辑编程控制下可靠的对话代理及其对自然语言的理解 

**Authors**: Yankai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09237)  

**Abstract**: Efforts have been made to make machines converse like humans in the past few decades. The recent techniques of Large Language Models (LLMs) make it possible to have human-like conversations with machines, but LLM's flaws of lacking understanding and reliability are well documented. We believe that the best way to eliminate this problem is to use LLMs only as parsers to translate text to knowledge and vice versa and carry out the conversation by reasoning over this knowledge using the answer set programming. I have been developing a framework based on LLMs and ASP to realize reliable chatbots that "understand" human conversation. This framework has been used to develop task-specific chatbots as well as socialbots. My future research is focused on making these chatbots scalable and trainable. 

**Abstract (ZH)**: 在过去的几十年中，人们一直在努力使机器的对话方式像人类一样。最近的大规模语言模型（LLMs）技术使得与机器进行像人类一样的对话成为可能，但LLMs在理解能力和可靠性方面存在的缺陷已经被充分记录。我们相信，解决这一问题的最佳方法是仅将LLMs用作解析器，用于文本到知识的翻译和反之亦然，并利用答案集编程对这些知识进行推理以进行对话。我一直致力于开发基于LLMs和ASP的框架，以实现能够“理解”人类对话的可靠聊天机器人。该框架已用于开发特定任务的聊天机器人和社会机器人。我的未来研究重点在于使这些聊天机器人具有可扩展性和可训练性。 

---
# LP-LM: No Hallucinations in Question Answering with Logic Programming 

**Title (ZH)**: LP-LM：基于逻辑编程的问答中无幻觉生成 

**Authors**: Katherine Wu, Yanhong A. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09212)  

**Abstract**: Large language models (LLMs) are able to generate human-like responses to user queries. However, LLMs exhibit inherent limitations, especially because they hallucinate. This paper introduces LP-LM, a system that grounds answers to questions in known facts contained in a knowledge base (KB), facilitated through semantic parsing in Prolog, and always produces answers that are reliable.
LP-LM generates a most probable constituency parse tree along with a corresponding Prolog term for an input question via Prolog definite clause grammar (DCG) parsing. The term is then executed against a KB of natural language sentences also represented as Prolog terms for question answering. By leveraging DCG and tabling, LP-LM runs in linear time in the size of input sentences for sufficiently many grammar rules. Performing experiments comparing LP-LM with current well-known LLMs in accuracy, we show that LLMs hallucinate on even simple questions, unlike LP-LM. 

**Abstract (ZH)**: 大型语言模型（LLMs）能够生成与用户查询相匹配的人类似的回应。然而，LLMs 存在固有的局限性，尤其是在生成错误信息方面。本文介绍了一种名为 LP-LM 的系统，该系统通过使用 Prolog 进行语义解析，将问题的答案限定在知识库（KB）中的已知事实范围内，并始终生成可靠的答案。

LP-LM 通过使用 Prolog 确定性_clause_语法（DCG）解析生成一个最有可能的组成部分解析树以及与之对应的 Prolog 表达式。随后，该表达式在以 Prolog 表达式形式表示的自然语言句子的知识库中执行以进行问题回答。通过利用 DCG 和表格技术，LP-LM 在足够多的语法规则下，对于输入句子的大小呈现出线性时间复杂度。通过对比 LP-LM 与当前知名的大语言模型在准确度方面的实验结果，我们展示了在即使是非常简单的问题上，LLMs 也会出现错误信息，而 LP-LM 则不会。 

---
# FLAME: Flexible LLM-Assisted Moderation Engine 

**Title (ZH)**: FLAME：灵活的大型语言模型辅助审核引擎 

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2502.09175)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展在管理用户-模型互动方面带来了重大挑战。尽管LLMs展现了显著的能力，但在对抗性攻击面前，特别是“逃逸攻击”技术，它们仍然脆弱，这些技术可以绕过内容安全措施。当前的内容审核系统主要依赖输入提示过滤，但这些方法已证明不够充分，如“最佳N项”（BoN）逃逸攻击技术对流行LLMs的成功率高达80%以上。在本文中，我们介绍了一种新的灵活的大语言模型辅助审核引擎（FLAME）：该方法将焦点从输入过滤转向输出审核。不同于传统的方法分析用户查询，FLAME评估模型响应，具有以下几个关键优势：（1）训练和推理中的计算效率高；（2）增强对BoN逃逸攻击的抵抗力；（3）通过可定制的主题过滤定义和更新安全标准的灵活性。我们的实验表明，FLAME显著优于现有的审核系统。例如，FLAME将针对GPT-4o-mini和DeepSeek-v3的攻击成功率降低了约9倍，同时保持了较低的计算开销。我们对多种LLMs进行了全面评估，并分析了该引擎在最新逃逸攻击中的效率。这项工作为LLMs开发更稳健和适应性更强的内容审核系统做出了贡献。 

---
# Logical Reasoning in Large Language Models: A Survey 

**Title (ZH)**: 大型语言模型中的逻辑推理：一个综述 

**Authors**: Hanmeng Liu, Zhizhang Fu, Mengru Ding, Ruoxi Ning, Chaoli Zhang, Xiaozhang Liu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09100)  

**Abstract**: With the emergence of advanced reasoning models like OpenAI o3 and DeepSeek-R1, large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, their ability to perform rigorous logical reasoning remains an open question. This survey synthesizes recent advancements in logical reasoning within LLMs, a critical area of AI research. It outlines the scope of logical reasoning in LLMs, its theoretical foundations, and the benchmarks used to evaluate reasoning proficiency. We analyze existing capabilities across different reasoning paradigms - deductive, inductive, abductive, and analogical - and assess strategies to enhance reasoning performance, including data-centric tuning, reinforcement learning, decoding strategies, and neuro-symbolic approaches. The review concludes with future directions, emphasizing the need for further exploration to strengthen logical reasoning in AI systems. 

**Abstract (ZH)**: 随着先进推理模型如OpenAI o3和DeepSeek-R1的出现，大规模语言模型（LLMs）展现出了卓越的推理能力。然而，它们进行严谨逻辑推理的能力依然是一个待解的问题。本文综述了LLMs中逻辑推理的最新进展，这是人工智能研究中一个关键领域。本文概述了逻辑推理在LLMs中的范围、理论基础以及用于评估推理能力的标准。我们分析了不同推理范式（演绎、归纳、溯因和类比）下现有的能力，并评估了提升推理性能的各种策略，包括数据导向的调优、强化学习、解码策略以及神经符号方法。综述最后提出了一些未来发展方向，强调了进一步探索以增强AI系统中逻辑推理的需要。 

---
# Mathematical Reasoning in Large Language Models: Assessing Logical and Arithmetic Errors across Wide Numerical Ranges 

**Title (ZH)**: 大型语言模型中的数学推理评估：跨广域数值范围检测逻辑和算术错误 

**Authors**: Safal Shrestha, Minwu Kim, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2502.08680)  

**Abstract**: Mathematical reasoning in Large Language Models (LLMs) is often evaluated using benchmarks with limited numerical ranges, failing to reflect real-world problem-solving across diverse scales. Furthermore, most existing evaluation methods only compare model outputs to ground-truth answers, obscuring insights into reasoning processes. To address these limitations, we introduce GSM-Ranges, a dataset generator derived from GSM8K that systematically perturbs numerical values in math problems to assess model robustness across varying numerical scales. Additionally, we propose a novel grading methodology that distinguishes between logical and non-logical errors, offering a more precise evaluation of reasoning processes beyond computational accuracy. Our experiments with various models reveal a significant increase in logical error rates-up to 14 percentage points-as numerical complexity rises, demonstrating a general weakness in reasoning with out-of-distribution numerical values. Moreover, while models demonstrate high accuracy on standalone arithmetic tasks, their performance deteriorates substantially when computations are embedded within word problems. These findings provide a comprehensive evaluation of LLMs' mathematical reasoning capabilities and inform future research directions for improving numerical generalization in language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的数学推理通常通过具有有限数值范围的基准进行评估，无法真实反映跨不同尺度解决问题的能力。此外，现有的大多数评估方法仅将模型输出与正确答案进行比对，隐藏了推理过程的洞察。为解决这些问题，我们引入了GSM-Ranges，这是一个从GSM8K派生的数据集生成器，该生成器系统地在数学问题中扰动数值值，以评估模型在不同数值尺度下的鲁棒性。同时，我们提出了一种新的评分方法，区分逻辑错误和非逻辑错误，从而提供一种更精确的超越计算精度的推理过程评估。实验结果表明，随着数值复杂性的增加，逻辑错误率显著上升，最高可达14个百分点，揭示了模型在处理未见过的数值值时推理能力的一般缺陷。此外，在独立的算术任务上，模型表现出较高的准确性，但在计算嵌入在文字问题中时，其性能显著下降。这些发现为评估LLMs的数学推理能力和指导未来改进语言模型数值泛化的研究方向提供了全面的视角。 

---
# Unleashing the Power of Large Language Model for Denoising Recommendation 

**Title (ZH)**: 释放大型语言模型在噪声推荐处理中的强大功能 

**Authors**: Shuyao Wang, Zhi Zheng, Yongduo Sui, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.09058)  

**Abstract**: Recommender systems are crucial for personalizing user experiences but often depend on implicit feedback data, which can be noisy and misleading. Existing denoising studies involve incorporating auxiliary information or learning strategies from interaction data. However, they struggle with the inherent limitations of external knowledge and interaction data, as well as the non-universality of certain predefined assumptions, hindering accurate noise identification. Recently, large language models (LLMs) have gained attention for their extensive world knowledge and reasoning abilities, yet their potential in enhancing denoising in recommendations remains underexplored. In this paper, we introduce LLaRD, a framework leveraging LLMs to improve denoising in recommender systems, thereby boosting overall recommendation performance. Specifically, LLaRD generates denoising-related knowledge by first enriching semantic insights from observational data via LLMs and inferring user-item preference knowledge. It then employs a novel Chain-of-Thought (CoT) technique over user-item interaction graphs to reveal relation knowledge for denoising. Finally, it applies the Information Bottleneck (IB) principle to align LLM-generated denoising knowledge with recommendation targets, filtering out noise and irrelevant LLM knowledge. Empirical results demonstrate LLaRD's effectiveness in enhancing denoising and recommendation accuracy. 

**Abstract (ZH)**: 推荐系统对于个性化用户体验至关重要，但往往依赖于隐式反馈数据，这些数据可能噪音较大且会误导。现有的去噪研究涉及通过辅助信息或交互数据的学习策略来整合信息，但这在面对外部知识和交互数据的固有局限性以及某些预定义假设的特异性时，仍存在困难，影响准确的噪声识别。近年来，大规模语言模型（LLM）因其广泛的领域知识和推理能力而受到关注，但其在推荐中的去噪增强潜力尚未得到充分探索。在本文中，我们提出了一种名为LLaRD的框架，利用大规模语言模型来提高推荐系统中的去噪性能，从而提高整体推荐性能。具体而言，LLaRD首先通过大规模语言模型丰富观测数据的语义洞察，并推断用户-项目偏好知识。然后，它使用一种新颖的事前推理（CoT）技术来揭示用户-项目交互图中的关系知识，以用于去噪。最后，它应用信息瓶颈（IB）原理来使大规模语言模型生成的去噪知识与推荐目标对齐，从而过滤掉噪声和无关的大规模语言模型知识。实验结果表明，LLaRD在提高去噪和推荐准确性方面具有有效性。 

---
# MDCrow: Automating Molecular Dynamics Workflows with Large Language Models 

**Title (ZH)**: MDCrow：使用大规模语言模型自动化分子动力学工作流 

**Authors**: Quintina Campbell, Sam Cox, Jorge Medina, Brittany Watterson, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2502.09565)  

**Abstract**: Molecular dynamics (MD) simulations are essential for understanding biomolecular systems but remain challenging to automate. Recent advances in large language models (LLM) have demonstrated success in automating complex scientific tasks using LLM-based agents. In this paper, we introduce MDCrow, an agentic LLM assistant capable of automating MD workflows. MDCrow uses chain-of-thought over 40 expert-designed tools for handling and processing files, setting up simulations, analyzing the simulation outputs, and retrieving relevant information from literature and databases. We assess MDCrow's performance across 25 tasks of varying required subtasks and difficulty, and we evaluate the agent's robustness to both difficulty and prompt style. \texttt{gpt-4o} is able to complete complex tasks with low variance, followed closely by \texttt{llama3-405b}, a compelling open-source model. While prompt style does not influence the best models' performance, it has significant effects on smaller models. 

**Abstract (ZH)**: 分子动力学（MD）模拟对于理解生物分子系统至关重要，但自动化的实现仍然具有挑战性。近年来，大规模语言模型（LLM）在使用基于LLM的代理自动化复杂科学任务方面取得了显著成功。在本文中，我们介绍了MDCrow，这是一种能够自动化MD工作流程的代理型LLM辅助工具。MDCrow通过使用40个专家设计的工具进行文件处理和处理、设置模拟、分析模拟输出，并从文献和数据库中检索相关的信息，实现了这一目标。我们对MDCrow进行了评估，涉及25项不同子任务和难度的任务，评估了该代理在难度和指令风格方面的一致性。结果显示，\texttt{gpt-4o}能够以极低的方差完成复杂任务，紧随其后的是\texttt{llama3-405b}，这是一个有吸引力的开源模型。虽然指令风格对最佳模型的性能影响不大，但对较小模型则有显著影响。 

---
# Visual Graph Question Answering with ASP and LLMs for Language Parsing 

**Title (ZH)**: 使用ASP和LLMs进行语言解析的视觉图问答 

**Authors**: Jakob Johannes Bauer, Thomas Eiter, Nelson Higuera Ruiz, Johannes Oetsch  

**Link**: [PDF](https://arxiv.org/pdf/2502.09211)  

**Abstract**: Visual Question Answering (VQA) is a challenging problem that requires to process multimodal input. Answer-Set Programming (ASP) has shown great potential in this regard to add interpretability and explainability to modular VQA architectures. In this work, we address the problem of how to integrate ASP with modules for vision and natural language processing to solve a new and demanding VQA variant that is concerned with images of graphs (not graphs in symbolic form). Images containing graph-based structures are an ubiquitous and popular form of visualisation. Here, we deal with the particular problem of graphs inspired by transit networks, and we introduce a novel dataset that amends an existing one by adding images of graphs that resemble metro lines. Our modular neuro-symbolic approach combines optical graph recognition for graph parsing, a pretrained optical character recognition neural network for parsing labels, Large Language Models (LLMs) for language processing, and ASP for reasoning. This method serves as a first baseline and achieves an overall average accuracy of 73% on the dataset. Our evaluation provides further evidence of the potential of modular neuro-symbolic systems, in particular with pretrained models that do not involve any further training and logic programming for reasoning, to solve complex VQA tasks. 

**Abstract (ZH)**: 视觉问答（VQA）是处理多模态输入的一个具有挑战性的问题。回答集编程（Answer-Set Programming, ASP）在这一领域展示了巨大的潜力，能够为模块化VQA架构增加解释性和可解释性。本文中，我们探讨了如何将ASP与视觉模块和自然语言处理模块集成，以解决一种新的且更具挑战性的VQA变体，该变体关注的是包含基于图的结构的图图像（而非符号形式的图）。基于图的结构图像是一种广泛使用的可视化形式。在此，我们处理了由公共交通网络启发的特殊图问题，并引入了一个新的数据集，该数据集通过增加了类似于地铁线路的图图像来修正现有数据集。我们的模块化神经符号方法结合了光学图识别以进行图解析、预训练的光学字符识别神经网络以进行标签解析、大型语言模型（Large Language Models, LLMs）进行语言处理以及ASP进行推理。该方法作为首个基线，达到了数据集整体平均准确率73%。我们的评估进一步证实了模块化神经符号系统的潜力，特别是那些不涉及进一步训练和逻辑编程推理的预训练模型，能够解决复杂的VQA任务。 

---
# On LLM-generated Logic Programs and their Inference Execution Methods 

**Title (ZH)**: LLM生成的逻辑程序及其推理执行方法 

**Authors**: Paul Tarau  

**Link**: [PDF](https://arxiv.org/pdf/2502.09209)  

**Abstract**: Large Language Models (LLMs) trained on petabytes of data are highly compressed repositories of a significant proportion of the knowledge accumulated and distilled so far. In this paper we study techniques to elicit this knowledge in the form of several classes of logic programs, including propositional Horn clauses, Dual Horn clauses, relational triplets and Definite Clause Grammars. Exposing this knowledge as logic programs enables sound reasoning methods that can verify alignment of LLM outputs to their intended uses and extend their inference capabilities.  We study new execution methods for the generated programs, including soft-unification of abducible facts against LLM-generated content stored in a vector database as well as GPU-based acceleration of minimal model computation that supports  inference with large LLM-generated programs. 

**Abstract (ZH)**: 以下是经过学术规范化翻译的内容：

大规模语言模型（LLMs）使用petabytes级别的数据进行训练，成为高度压缩的知识存储库，涵盖了迄今为止积累和提炼的大量知识。在本文中，我们探讨了一种方法，即将这些知识以多种逻辑程序的形式（包括命题厄尔克子句、对偶厄尔克子句、关系三元组和确定性子句语法）提取出来。将这些知识以逻辑程序的形式呈现，能够借助稳健的推理方法验证LLM输出是否符合预期用途，并扩展其推理能力。我们还研究了生成程序的新型执行方法，包括可变形统一可获取事实与LLM生成内容在向量数据库中的融合，以及基于GPU加速最小模型计算，以支持大规模LLM生成程序的推理。 

---
# Logical Lease Litigation: Prolog and LLMs for Rental Law Compliance in New York 

**Title (ZH)**: 逻辑租赁诉讼：Prolog和大规模语言模型在纽约租赁法律合规中的应用 

**Authors**: Sanskar Sehgal, Yanhong A. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09204)  

**Abstract**: Legal cases require careful logical reasoning following the laws, whereas interactions with non- technical users must be in natural language. As an application combining logical reasoning using Prolog and natural language processing using large language models (LLMs), this paper presents a novel approach and system, LogicLease, to automate the analysis of landlord-tenant legal cases in the state of New York. LogicLease determines compliance with relevant legal requirements by analyzing case descriptions and citing all relevant laws. It leverages LLMs for information extraction and Prolog for legal reasoning. By separating information extraction from legal reasoning, LogicLease achieves greater transparency and control over the legal logic applied to each case. We evaluate the accuracy, efficiency, and robustness of LogicLease through a series of tests, achieving 100% accuracy and an average processing time of 2.57 seconds. LogicLease presents advantages over state-of-the-art LLM- based legal analysis systems by providing clear, step-by-step reasoning, citing specific laws, and distinguishing itself by its ability to avoid hallucinations - a common issue in LLMs. 

**Abstract (ZH)**: 法律案件需要遵循法律进行仔细的逻辑推理，而与非技术用户互动时则需要使用自然语言。作为一种结合使用Prolog进行逻辑推理和大型语言模型（LLMs）进行自然语言处理的应用程序，本文提出了一种创新的方法和系统——LogicLease，用于自动化分析纽约州房东-租客法律案件。LogicLease通过分析案件描述并引用所有相关法律来判断是否符合相关法律要求。该系统利用LLMs进行信息提取，并使用Prolog进行法律推理。通过将信息提取和法律推理分开处理，LogicLease实现了更高的透明度和对应用于每个案件的法律逻辑的控制。我们通过一系列测试评估了LogicLease的准确度、效率和鲁棒性，实现了100%的准确度和平均处理时间2.57秒。与最先进的基于LLMs的法律分析系统相比，LogicLease具有优势，因为它提供了清晰的逐步推理过程、引用具体法律，并且能够避免LLMs常见的幻觉问题。 

---
# Cost-Saving LLM Cascades with Early Abstention 

**Title (ZH)**: 节省成本的大型语言模型级联与早期弃权 

**Authors**: Michael J. Zellinger, Rex Liu, Matt Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2502.09054)  

**Abstract**: LLM cascades are based on the idea that processing all queries with the largest and most expensive LLMs is inefficient. Instead, cascades deploy small LLMs to answer the majority of queries, limiting the use of large and expensive LLMs to only the most difficult queries. This approach can significantly reduce costs without impacting performance. However, risk-sensitive domains such as finance or medicine place an additional premium on avoiding model errors. Recognizing that even the most expensive models may make mistakes, applications in these domains benefit from allowing LLM systems to completely abstain from answering a query when the chance of making a mistake is significant. However, giving a cascade the ability to abstain poses an immediate design question for LLM cascades: should abstention only be allowed at the final model or also at earlier models? Since the error patterns of small and large models are correlated, the latter strategy may further reduce inference costs by letting inexpensive models anticipate abstention decisions by expensive models, thereby obviating the need to run the expensive models. We investigate the benefits of "early abstention" in LLM cascades and find that it reduces the overall test loss by 2.2% on average across six benchmarks (GSM8K, MedMCQA, MMLU, TriviaQA, TruthfulQA, and XSum). These gains result from a more effective use of abstention, which trades a 4.1% average increase in the overall abstention rate for a 13.0% reduction in cost and a 5.0% reduction in error rate. Our findings demonstrate that it is possible to leverage correlations between the error patterns of different language models to drive performance improvements for LLM systems with abstention. 

**Abstract (ZH)**: 大型语言模型（LLM）级联（cascades）基于这样一个理念，即使用最大的且成本最高的LLM处理所有查询是不高效的。相反，级联部署小型LLM来回答大多数查询，将大型和昂贵的LLM的使用限制在最困难的查询上。这种方法可以在不牺牲性能的情况下显著降低成本。然而，金融或医学等风险敏感领域对避免模型错误额外重视。即使是最昂贵的模型也可能出错，因此这些领域的应用程序可以从允许LLM系统在错误概率较高的情况下完全放弃回答查询中获益。然而，向级联赋予放弃回答查询的能力，提出了一个立即的设计问题：放弃只应该在最终模型中允许，还是也应该在较早的模型中允许？由于小模型和大模型的错误模式是相关的，后一种策略可以通过让成本较低的模型提前预判昂贵模型的放弃决定来进一步降低推理成本，从而避免运行昂贵模型。我们研究了“早期放弃”在LLM级联中的益处，并发现它在六个基准测试（GSM8K、MedMCQA、MMLU、TriviaQA、TruthfulQA和XSum）上平均降低了2.2%的整体测试损失。这些收益来自于更有效的利用放弃机制，这种方式使整体放弃率平均提高了4.1%，但成本降低了13.0%，错误率降低了5.0%。我们的研究结果表明，可以通过利用不同语言模型错误模式之间的相关性来推动包含放弃机制的LLM系统的性能提升。 

---
# Game Theory Meets Large Language Models: A Systematic Survey 

**Title (ZH)**: 博弈论与大型语言模型的相遇：一项系统性综述 

**Authors**: Haoran Sun, Yusen Wu, Yukun Cheng, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09053)  

**Abstract**: Game theory establishes a fundamental framework for analyzing strategic interactions among rational decision-makers. The rapid advancement of large language models (LLMs) has sparked extensive research exploring the intersection of these two fields. Specifically, game-theoretic methods are being applied to evaluate and enhance LLM capabilities, while LLMs themselves are reshaping classic game models. This paper presents a comprehensive survey of the intersection of these fields, exploring a bidirectional relationship from three perspectives: (1) Establishing standardized game-based benchmarks for evaluating LLM behavior; (2) Leveraging game-theoretic methods to improve LLM performance through algorithmic innovations; (3) Characterizing the societal impacts of LLMs through game modeling. Among these three aspects, we also highlight how the equilibrium analysis for traditional game models is impacted by LLMs' advanced language understanding, which in turn extends the study of game theory. Finally, we identify key challenges and future research directions, assessing their feasibility based on the current state of the field. By bridging theoretical rigor with emerging AI capabilities, this survey aims to foster interdisciplinary collaboration and drive progress in this evolving research area. 

**Abstract (ZH)**: 博弈论建立了一个基本框架，用于分析理性决策者之间的战略互动。随着大型语言模型（LLMs）的快速发展，这两个领域的交叉研究引起了广泛的关注。具体而言，博弈论方法正在被应用于评估和提升LLM的能力，而LLMs本身也在重塑经典博弈模型。本文对这些领域的交叉进行了全面回顾，从三个角度探讨了这种双向关系：（1）建立基于博弈的标准基准来评估LLM的行为；（2）利用博弈论方法通过算法创新提高LLM的性能；（3）通过博弈建模来分析LLMs的社会影响。在这三个方面，我们还强调了LLMs高级语言理解能力对传统博弈模型均衡分析的影响，这种影响又扩展了博弈论的研究范围。最后，我们指出了关键挑战和未来的研究方向，并基于当前该领域的状态评估了这些方向的可行性。通过将理论严谨性与新兴AI能力相结合，本文旨在促进跨学科合作，并推动这一不断发展的研究领域的进步。 

---
# Mechanistic Unveiling of Transformer Circuits: Self-Influence as a Key to Model Reasoning 

**Title (ZH)**: 变压器电路机制揭示：自我影响是模型推理的关键 

**Authors**: Lin Zhang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09022)  

**Abstract**: Transformer-based language models have achieved notable success, yet their internal reasoning mechanisms remain largely opaque due to complex non-linear interactions and high-dimensional operations. While previous research suggests that these models implicitly encode reasoning structures, it is still unclear which specific multi-step thought processes they employ to solve complex tasks. To address this gap, we propose a novel mechanistic interpretability framework, SICAF, designed to trace and analyze the reasoning strategies that language models use in multi-step inference tasks. By employing circuit analysis and self-influence functions, we quantify the evolving importance of each token throughout the reasoning process, thereby mapping the pathways the model uses for inference. Applying SICAF to the GPT-2 model on the Indirect Object Identification (IOI) prediction task, we demonstrate how underlying circuits can reveal a reasoning process that aligns with human interpretability, offering new insights into the model's internal logic. 

**Abstract (ZH)**: 基于Transformer的语言模型取得了显著的成功，但由于其内部推理机制受到了复杂非线性相互作用和高维操作的限制，这些机制仍然相当不透明。尽管以前的研究表明这些模型隐含地编码了推理结构，但仍然不清楚它们在解决复杂任务时具体采用了哪些多步思维过程。为解决这一问题，我们提出了一个新的因果可解释性框架SICAF，旨在跟踪和分析语言模型在多步推理任务中使用的推理策略。通过利用电路分析和自我影响函数，我们量化了每个词语在整个推理过程中的重要性变化，从而绘制出模型进行推理所使用的路径。将SICAF应用于GPT-2模型在间接宾语识别（IOI）预测任务中的应用，我们展示了底层电路如何揭示出与人类可解释性一致的推理过程，从而为我们提供了关于模型内部逻辑的新见解。 

---
# Reinforced Large Language Model is a formal theorem prover 

**Title (ZH)**: 强化型大型语言模型是一个形式定理证明器 

**Authors**: Zhiling Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.08908)  

**Abstract**: To take advantage of Large Language Model in theorem formalization and proof, we propose a reinforcement learning framework to iteratively optimize the pretrained LLM by rolling out next tactics and comparing them with the expected ones. The experiment results show that it helps to achieve a higher accuracy compared with directly fine-tuned LLM. 

**Abstract (ZH)**: 为了利用大型语言模型在定理形式化和证明中的优势，我们提出了一种强化学习框架，通过迭代优化预训练的大语言模型，生成下一个策略并将其与期望策略进行比较。实验结果表明，这种方法在准确性方面优于直接微调大语言模型。 

---
# MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training 

**Title (ZH)**: MIH-TCCT：通过事件驱动的文本-代码循环训练来缓解LLM中的不一致幻觉 

**Authors**: Xinxin You, Xien Liu, Qixin Sun, Huan Zhang, Kaiyin Zhou, Shaohui Liu, GuoPing Hu, ShiJin Wang, Si Liu, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08904)  

**Abstract**: Recent methodologies utilizing synthetic datasets have aimed to address inconsistent hallucinations in large language models (LLMs); however,these approaches are primarily tailored to specific tasks, limiting their generalizability. Inspired by the strong performance of code-trained models in logic-intensive domains, we propose a novel framework that leverages event-based text to generate corresponding code and employs cyclic training to transfer the logical consistency of code to natural language effectively. Our method significantly reduces inconsistent hallucinations across three leading LLMs and two categories of natural language tasks while maintaining overall performance. This framework effectively alleviates hallucinations without necessitating adaptation to downstream tasks, demonstrating generality and providing new perspectives to tackle the challenge of inconsistent hallucinations. 

**Abstract (ZH)**: 近年来，利用合成数据集的方法旨在解决大规模语言模型（LLMs）中的不一致幻觉问题；然而，这些方法主要针对特定任务，限制了它们的普适性。受代码训练模型在逻辑密集型领域中强大性能的启发，我们提出了一种新颖的框架，该框架利用事件驱动的文本生成相应的代码，并采用循环训练方法将代码的逻辑一致性有效转移到自然语言中。我们的方法显著减少了三种领先LLM和两类自然语言任务中不一致幻觉的发生，同时保持了整体性能。该框架有效地减轻了幻觉现象，无需对下游任务进行适应，展示了其普适性，并为应对不一致幻觉挑战提供了新的视角。 

---
# Can a Single Model Master Both Multi-turn Conversations and Tool Use? CALM: A Unified Conversational Agentic Language Model 

**Title (ZH)**: 单个模型能否同时掌握多轮对话和工具使用？统一对话型代理语言模型——CALM 

**Authors**: Emre Can Acikgoz, Jeremiah Greer, Akul Datta, Ze Yang, William Zeng, Oussama Elachqar, Emmanouil Koukoumidis, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2502.08820)  

**Abstract**: Large Language Models (LLMs) with API-calling capabilities enabled building effective Language Agents (LA), while also revolutionizing the conventional task-oriented dialogue (TOD) paradigm. However, current approaches face a critical dilemma: TOD systems are often trained on a limited set of target APIs, requiring new data to maintain their quality when interfacing with new services, while LAs are not trained to maintain user intent over multi-turn conversations. Because both robust multi-turn management and advanced function calling are crucial for effective conversational agents, we evaluate these skills on three popular benchmarks: MultiWOZ 2.4 (TOD), BFCL V3 (LA), and API-Bank (LA), and our analyses reveal that specialized approaches excel in one domain but underperform in the other. To bridge this chasm, we introduce CALM (Conversational Agentic Language Model), a unified approach that integrates both conversational and agentic capabilities. We created CALM-IT, a carefully constructed multi-task dataset that interleave multi-turn ReAct reasoning with complex API usage. Using CALM-IT, we train three models CALM 8B, CALM 70B, and CALM 405B, which outperform top domain-specific models, including GPT-4o, across all three benchmarks. 

**Abstract (ZH)**: larg语言模型（LLMs）具备API调用能力，能够构建有效的语言代理（LAs），同时也在传统任务导向对话（TOD）范式上进行了革命。然而，当前的方法面临一个关键的困境：TOD系统通常仅在有限的目标API集上进行训练，当与新的服务交互时，需要新数据来保持其质量，而LAs则未被训练以在多轮对话中保持用户意图。由于强大的多轮管理和高级功能调用对于有效的对话代理至关重要，我们在三个流行的基准测试上评估了这些技能：MultiWOZ 2.4（TOD）、BFCL V3（LA）和API-Bank（LA），分析结果显示专门的方法在某一领域表现出色，但在另一领域则表现不佳。为弥合这一差距，我们提出了CALM（Conversational Agentic Language Model），这是一种统一的方法，融合了对话能力和代理功能。我们创建了CALM-IT，这是一个精心构建的多任务数据集，结合了多轮ReAct推理和复杂API使用。使用CALM-IT，我们训练了三个模型：CALM 8B、CALM 70B和CALM 405B，它们在三个基准测试上均表现出色，超越了包括GPT-4o在内的顶尖领域特定模型。 

---
# From PowerPoint UI Sketches to Web-Based Applications: Pattern-Driven Code Generation for GIS Dashboard Development Using Knowledge-Augmented LLMs, Context-Aware Visual Prompting, and the React Framework 

**Title (ZH)**: 从PowerPoint UI 草图到基于Web的应用：利用知识增强的大语言模型、上下文感知视觉提示和React框架的GIS仪表板开发模式驱动代码生成 

**Authors**: Haowen Xu, Xiao-Ying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08756)  

**Abstract**: Developing web-based GIS applications, commonly known as CyberGIS dashboards, for querying and visualizing GIS data in environmental research often demands repetitive and resource-intensive efforts. While Generative AI offers automation potential for code generation, it struggles with complex scientific applications due to challenges in integrating domain knowledge, software engineering principles, and UI design best practices. This paper introduces a knowledge-augmented code generation framework that retrieves software engineering best practices, domain expertise, and advanced technology stacks from a specialized knowledge base to enhance Generative Pre-trained Transformers (GPT) for front-end development. The framework automates the creation of GIS-based web applications (e.g., dashboards, interfaces) from user-defined UI wireframes sketched in tools like PowerPoint or Adobe Illustrator. A novel Context-Aware Visual Prompting method, implemented in Python, extracts layouts and interface features from these wireframes to guide code generation. Our approach leverages Large Language Models (LLMs) to generate front-end code by integrating structured reasoning, software engineering principles, and domain knowledge, drawing inspiration from Chain-of-Thought (CoT) prompting and Retrieval-Augmented Generation (RAG). A case study demonstrates the framework's capability to generate a modular, maintainable web platform hosting multiple dashboards for visualizing environmental and energy data (e.g., time-series, shapefiles, rasters) from user-sketched wireframes. By employing a knowledge-driven approach, the framework produces scalable, industry-standard front-end code using design patterns such as Model-View-ViewModel (MVVM) and frameworks like React. This significantly reduces manual effort in design and coding, pioneering an automated and efficient method for developing smart city software. 

**Abstract (ZH)**: 在环境研究中，开发基于Web的GIS应用程序，通常被称为CyberGIS仪表板，对于查询和可视化GIS数据往往需要重复且资源密集型的努力。尽管生成式AI在代码生成方面提供了自动化潜力，但由于在集成领域知识、软件工程原则和UI设计最佳实践方面的挑战，它在复杂科学应用中显得力不从心。本文介绍了一种增强型知识代码生成框架，该框架从专门的知识数据库中检索软件工程最佳实践、领域专业知识和先进技术堆栈，以增强生成式预训练变换器（GPT）在前端开发中的应用。该框架能够根据用户在如PowerPoint或Adobe Illustrator等工具中定义的UI线框图来自动化创建基于GIS的Web应用程序（如仪表板、界面）。我们提出了一种新颖的上下文感知视觉提示方法，通过Python实现，可以从这些线框图中提取布局和界面特性，以指导代码生成。本文采用大型语言模型（LLMs）通过结合结构化推理、软件工程原则和领域知识来生成前端代码，这些灵感来源于链式思考（CoT）提示和检索增强生成（RAG）方法。一个案例研究展示了该框架生成一个模块化且易于维护的Web平台的能力，该平台可以托管多个用于展示环境和能源数据（如时间序列、矢量文件、栅格）的仪表盘，这些仪表盘均由用户绘制的线框图生成。通过知识驱动的方法，该框架使用模型-视图-视图模型（MVVM）设计模式和React等框架生成可扩展的工业标准前端代码，显著减少了设计和编码的体力劳动，开创了一种自动化和高效的智慧城市软件开发方法。 

---
# CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality 

**Title (ZH)**: CopySpec：无需牺牲质量的投机性复制与粘贴加速大规模语言模型 

**Authors**: Razvan-Gabriel Dumitru, Minglai Yang, Vikas Yadav, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08923)  

**Abstract**: We introduce CopySpec, an innovative technique designed to tackle the inefficiencies LLMs face when generating responses that closely resemble previous outputs. CopySpec identifies repeated sequences in the model's chat history and speculates that the same tokens will follow, enabling seamless copying without compromising output quality or requiring additional GPU memory. To evaluate the effectiveness of our approach, we conducted experiments using five LLMs and five datasets: MT-Bench, CNN/DM, GSM-8K, HumanEval, and our newly created dataset, MT-Redundant. MT-Redundant, introduced in this paper, transforms the second turn of MT-Bench into a request for variations of the first turn's answer, simulating real-world scenarios where users request modifications to prior responses. Our results demonstrate significant speed-ups: up to 2.35x on CNN/DM, 3.08x on the second turn of select MT-Redundant categories, and 2.66x on the third turn of GSM-8K's self-correction tasks. Moreover, we show that CopySpec integrates seamlessly with speculative decoding, yielding an average 49% additional speed-up over speculative decoding for the second turn of MT-Redundant across all eight categories. While LLMs, even with speculative decoding, suffer from slower inference as context sizes grow, CopySpec leverages the expanded context to accelerate inference, making it faster as the context size increases. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 我们介绍了CopySpec，这是一种创新技术，旨在解决大型语言模型在生成与之前输出高度相似的回答时遇到的效率问题。CopySpec能够识别模型聊天历史中的重复序列，并推测后续同样会出现相同的词汇，从而实现无缝复制，而不牺牲输出质量或额外增加GPU内存需求。为了评估我们方法的效果，我们使用五种大型语言模型和五个数据集进行了实验：MT-Bench、CNN/DM、GSM-8K、HumanEval以及我们新创建的数据集MT-Redundant。MT-Redundant在本文中首次引入，将MT-Bench的第二轮对话转换为请求第一轮回答的变体，模拟了用户请求修改先前回答的真实场景。我们的实验结果表明，CopySpec带来了显著的速度提升：在CNN/DM上高达2.35倍，在MT-Redundant的部分类别第二轮对话上高达3.08倍，在GSM-8K的自校正任务的第三轮上高达2.66倍。此外，我们展示了CopySpec与 speculate 解码无缝集成的能力，相较 speculate 解码，在所有八个类别中的第二轮MT-Redundant代码平均加速了49%。虽然即使带有 speculate 解码的大型语言模型随上下文大小增加也会导致推理速度变慢，但CopySpec利用扩展后的上下文来加速推理，使得在上下文大小增加时模型运行更快。我们的代码和数据集已在此处公开：[公开链接]。 

---
# Bridging the Evaluation Gap: Leveraging Large Language Models for Topic Model Evaluation 

**Title (ZH)**: 填补评估缺口：利用大规模语言模型进行主题模型评估 

**Authors**: Zhiyin Tan, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2502.07352)  

**Abstract**: This study presents a framework for automated evaluation of dynamically evolving topic taxonomies in scientific literature using Large Language Models (LLMs). In digital library systems, topic modeling plays a crucial role in efficiently organizing and retrieving scholarly content, guiding researchers through complex knowledge landscapes. As research domains proliferate and shift, traditional human centric and static evaluation methods struggle to maintain relevance. The proposed approach harnesses LLMs to measure key quality dimensions, such as coherence, repetitiveness, diversity, and topic-document alignment, without heavy reliance on expert annotators or narrow statistical metrics. Tailored prompts guide LLM assessments, ensuring consistent and interpretable evaluations across various datasets and modeling techniques. Experiments on benchmark corpora demonstrate the method's robustness, scalability, and adaptability, underscoring its value as a more holistic and dynamic alternative to conventional evaluation strategies. 

**Abstract (ZH)**: 本研究提出了一种基于大规模语言模型（LLMs）自动评估科学文献中动态演化的主题分类的框架。在数字图书馆系统中，主题建模在有效地组织和检索学术内容、引导研究人员通过复杂的知识领域方面发挥着重要作用。随着研究领域的不断增多和转变，传统的以人为中心和静态的评估方法难以保持相关性。所提出的方法利用LLMs来衡量主题连贯性、重复性、多样性以及主题与文档的对齐程度等关键质量维度，而无需依赖众多专家标注者或简单的统计指标。定制化的提示语句引导LLMs的评估，确保在不同数据集和建模技术下的一致性和可解释性。基准语料库上的实验表明该方法的稳健性、可扩展性和适应性，突显了它作为一种更为全面和动态的替代传统评估策略的价值。 

---
# ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates 

**Title (ZH)**: ReasonFlux：通过扩展思维模板实现的层级大语言模型推理 

**Authors**: Ling Yang, Zhaochen Yu, Bin Cui, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06772)  

**Abstract**: We present that hierarchical LLM reasoning via scaling thought templates can effectively optimize the reasoning search space and outperform the mathematical reasoning capabilities of powerful LLMs like OpenAI o1-preview and DeepSeek V3. We train our ReasonFlux-32B model with only 8 GPUs and introduces three innovations: (i) a structured and generic thought template library, containing around 500 high-level thought templates capable of generalizing to similar or relevant reasoning problems; (ii) performing hierarchical reinforcement learning on a sequence of thought templates instead of long CoTs, optimizing a base LLM to plan out an optimal template trajectory for gradually handling complex problems; (iii) a brand new inference scaling system that enables hierarchical LLM reasoning by adaptively scaling thought templates at inference time. With a template trajectory containing sequential thought templates, our ReasonFlux-32B significantly advances math reasoning capabilities to state-of-the-art levels. Notably, on the MATH benchmark, it achieves an accuracy of 91.2% and surpasses o1-preview by 6.7%. On the USA Math Olympiad (AIME) benchmark, ReasonFlux-32B solves an average of 56.7% of problems, surpassing o1-preview and DeepSeek-V3 by 27% and 45%, respectively. Code: this https URL 

**Abstract (ZH)**: 我们展示了通过扩展层级思维模板，LLM能够有效地优化推理搜索空间，并且在数学推理能力方面超越了像OpenAI o1-preview和DeepSeek V3这样强大的LLM。我们仅使用8个GPU训练了ReasonFlux-32B模型，并引入了三项创新：(i) 结构化且通用的思维模板库，包含大约500个高阶思维模板，能够泛化到类似或相关的推理问题；(ii) 在一系列思维模板上进行层级强化学习，而不是长复杂的中间步骤推理(CoTs)，优化基本的LLM以规划处理复杂问题的最佳模板轨迹；(iii) 一种全新的推理扩展系统，通过在推理时动态扩展思维模板来实现层级LLM推理。借助包含序列化思维模板的模板轨迹，我们的ReasonFlux-32B显著提高了数学推理能力，达到了最先进的水平。值得注意的是，在MATH基准测试中，它实现了91.2%的准确率，比o1-preview高出了6.7%。在USAMO (AIME)基准测试中，ReasonFlux-32B解决了56.7%的问题，分别比o1-preview和DeepSeek-V3高出27%和45%。代码：[这里](此链接) 

---
