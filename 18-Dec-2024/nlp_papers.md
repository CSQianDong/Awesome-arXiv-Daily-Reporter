# DnDScore: Decontextualization and Decomposition for Factuality Verification in Long-Form Text Generation 

**Title (ZH)**: DnDScore：长文本生成中的去语境化与分解事实验证方法 

**Authors**: Miriam Wanner, Benjamin Van Durme, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2412.13175)  

**Abstract**: The decompose-then-verify strategy for verification of Large Language Model (LLM) generations decomposes claims that are then independently verified. Decontextualization augments text (claims) to ensure it can be verified outside of the original context, enabling reliable verification. While decomposition and decontextualization have been explored independently, their interactions in a complete system have not been investigated. Their conflicting purposes can create tensions: decomposition isolates atomic facts while decontextualization inserts relevant information. Furthermore, a decontextualized subclaim presents a challenge to the verification step: what part of the augmented text should be verified as it now contains multiple atomic facts? We conduct an evaluation of different decomposition, decontextualization, and verification strategies and find that the choice of strategy matters in the resulting factuality scores. Additionally, we introduce DnDScore, a decontextualization aware verification method which validates subclaims in the context of contextual information. 

**Abstract (ZH)**: 对于大语言模型（LLM）生成内容的验证，采用“分解-验证”策略，即将声明拆分成独立的单元，然后分别验证。去语境化扩展文本（声明），确保其可以在脱离原始语境的情况下进行验证，从而实现可靠的验证。尽管分解和去语境化已被分别研究，但在完整系统中的相互作用尚未得到探讨。这两种方法的目的存在冲突：分解孤立化基本事实，而去语境化则插入相关信息。此外，去语境化后的子声明给验证步骤带来了挑战：经过扩展的文本现在包含多个基本事实，哪一部分应该被验证呢？我们对不同的分解、去语境化和验证策略进行了评估，并发现所选择的方法会影响最终的事实评分。此外，我们提出了DnDScore方法，这是一种基于上下文信息验证子声明的去语境化感知验证方法。 

---
# Compressed Chain of Thought: Efficient Reasoning Through Dense Representations 

**Title (ZH)**: 压缩思维链：通过密集表示进行高效推理 

**Authors**: Jeffrey Cheng, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2412.13171)  

**Abstract**: Chain-of-thought (CoT) decoding enables language models to improve reasoning performance at the cost of high generation latency in decoding. Recent proposals have explored variants of contemplation tokens, a term we introduce that refers to special tokens used during inference to allow for extra computation. Prior work has considered fixed-length sequences drawn from a discrete set of embeddings as contemplation tokens. Here we propose Compressed Chain-of-Thought (CCoT), a framework to generate contentful and continuous contemplation tokens of variable sequence length. The generated contemplation tokens are compressed representations of explicit reasoning chains, and our method can be applied to off-the-shelf decoder language models. Through experiments, we illustrate how CCoT enables additional reasoning over dense contentful representations to achieve corresponding improvements in accuracy. Moreover, the reasoning improvements can be adaptively modified on demand by controlling the number of contemplation tokens generated. 

**Abstract (ZH)**: 链推理（CoT）解码能让语言模型在推理性能上得到提高，但会增加解码过程中的生成延迟。近年来的研究探索了推理令牌的变体，这是我们在推理过程中引入的一种特别术语，用于允许额外的计算。此前的工作考虑了从离散嵌入集中的固定长度序列中抽样的推理令牌。在此，我们提出了压缩链推理（Compressed Chain-of-Thought, CCoT）框架，以生成具有可变序列长度的内容丰富且连续的推理令牌。生成的推理令牌是明确推理链的压缩表示，我们的方法可以应用于现成的解码器语言模型。通过实验，我们展示了CCoT如何通过在密集内容表示上进行额外的推理来实现相应的准确度改进。此外，通过控制生成的推理令牌数量，推理改进可以按需进行适应性修改。 

---
# Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study 

**Title (ZH)**: 大型语言模型在生成合成德国公众意见方面的算法 fidelity：一个案例研究 

**Authors**: Bolei Ma, Berk Yoztyurk, Anna-Carolina Haensch, Xinpeng Wang, Markus Herklotz, Frauke Kreuter, Barbara Plank, Matthias Assenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2412.13169)  

**Abstract**: In recent research, large language models (LLMs) have been increasingly used to investigate public opinions. This study investigates the algorithmic fidelity of LLMs, i.e., the ability to replicate the socio-cultural context and nuanced opinions of human participants. Using open-ended survey data from the German Longitudinal Election Studies (GLES), we prompt different LLMs to generate synthetic public opinions reflective of German subpopulations by incorporating demographic features into the persona prompts. Our results show that Llama performs better than other LLMs at representing subpopulations, particularly when there is lower opinion diversity within those groups. Our findings further reveal that the LLM performs better for supporters of left-leaning parties like The Greens and The Left compared to other parties, and matches the least with the right-party AfD. Additionally, the inclusion or exclusion of specific variables in the prompts can significantly impact the models' predictions. These findings underscore the importance of aligning LLMs to more effectively model diverse public opinions while minimizing political biases and enhancing robustness in representativeness. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）越来越多地被用于研究公众意见。本研究探讨了LLMs的算法忠实性，即其重现人类参与者所处的社会文化背景及其细腻观点的能力。利用德国纵向选举研究（GLES）中的开放问卷数据，我们在生成反映德国亚群体的合成公众意见时将人口统计特征融入角色提示中，促使不同的LLMs进行生成。研究结果表明，Llama在代表亚群体方面表现优于其他LLMs，尤其是在这些群体内部意见差异较小的情况下。进一步的研究发现，Llama在模拟支持左翼政党（如德国绿党及左翼党）的公众意见时表现更佳，但在模拟右翼政党（如极右政党AfD）的公众意见时则匹配度最低。此外，提示中特定变量的包括或排除对模型预测结果的影响显著。这些发现强调了在更有效地建模多元化公众意见、减少政治偏见以及增强代表性稳健性方面对LLMs进行对齐的重要性。 

---
# BanglishRev: A Large-Scale Bangla-English and Code-mixed Dataset of Product Reviews in E-Commerce 

**Title (ZH)**: BanglishRev：电子商务中产品评价的大规模孟加拉语-英语及代码混合数据集 

**Authors**: Mohammad Nazmush Shamael, Sabila Nawshin, Swakkhar Shatabda, Salekul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2412.13161)  

**Abstract**: This work presents the BanglishRev Dataset, the largest e-commerce product review dataset to date for reviews written in Bengali, English, a mixture of both and Banglish, Bengali words written with English alphabets. The dataset comprises of 1.74 million written reviews from 3.2 million ratings information collected from a total of 128k products being sold in online e-commerce platforms targeting the Bengali population. It includes an extensive array of related metadata for each of the reviews including the rating given by the reviewer, date the review was posted and date of purchase, number of likes, dislikes, response from the seller, images associated with the review etc. With sentiment analysis being the most prominent usage of review datasets, experimentation with a binary sentiment analysis model with the review rating serving as an indicator of positive or negative sentiment was conducted to evaluate the effectiveness of the large amount of data presented in BanglishRev for sentiment analysis tasks. A BanglishBERT model is trained on the data from BanglishRev with reviews being considered labeled positive if the rating is greater than 3 and negative if the rating is less than or equal to 3. The model is evaluated by being testing against a previously published manually annotated dataset for e-commerce reviews written in a mixture of Bangla, English and Banglish. The experimental model achieved an exceptional accuracy of 94\% and F1 score of 0.94, demonstrating the dataset's efficacy for sentiment analysis. Some of the intriguing patterns and observations seen within the dataset and future research directions where the dataset can be utilized is also discussed and explored. The dataset can be accessed through this https URL. 

**Abstract (ZH)**: 以下是将该论文内容或标题翻译成中文后的版本，符合学术规范：

本文介绍了BanglishRev数据集，这是迄今为止最庞大的电子商务产品评论数据集，包含了孟加拉语、英语以及二者的混合使用（即Banglish）和使用英语字母书写的孟加拉语单词的评论数据。该数据集包含了320万条评论信息，来源于12.8万个正在在线电子商务平台上销售的产品，针对的是孟加拉人口群体。每条评论都包含了大量的相关元数据，包括评论者给出的评分、评论发布日期、购买日期、点赞数、点踩数、商家回应以及与评论相关的图片等信息。由于情感分析是评论数据集最常见的应用场景之一，因此，本文使用二分类情感分析模型进行实验，以评论评分作为正面或负面情感的指示器，评估BanglishRev数据集在情感分析任务中的有效性。通过对包含孟加拉语、英语以及Banglish混合使用的电子商务评论进行训练，提出了一个BanglishBERT模型。当评分大于3时，该模型将评论标记为积极肯普；当评分小于等于3时，标记为消极评论。模型通过与之前出版的手动标注的电子商务评论数据集进行测试，实现了94%的准确率和0.94的F1值，这表明该数据集在情感分析任务中的有效性。本文还探讨了数据集中的一些有趣模式和观察结果以及未来的研究方向。该数据集可通过以下链接访问：[插入链接]。

请注意，原文中的网址需要在实际应用中替换为实际的链接地址。 

---
# Syntactic Transfer to Kyrgyz Using the Treebank Translation Method 

**Title (ZH)**: 使用树banks翻译方法将句法转移到Kirgiz语 

**Authors**: Anton Alekseev, Alina Tillabaeva, Gulnara Dzh. Kabaeva, Sergey I. Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2412.13146)  

**Abstract**: The Kyrgyz language, as a low-resource language, requires significant effort to create high-quality syntactic corpora. This study proposes an approach to simplify the development process of a syntactic corpus for Kyrgyz. We present a tool for transferring syntactic annotations from Turkish to Kyrgyz based on a treebank translation method. The effectiveness of the proposed tool was evaluated using the TueCL treebank. The results demonstrate that this approach achieves higher syntactic annotation accuracy compared to a monolingual model trained on the Kyrgyz KTMU treebank. Additionally, the study introduces a method for assessing the complexity of manual annotation for the resulting syntactic trees, contributing to further optimization of the annotation process. 

**Abstract (ZH)**: 作为低资源语言，吉尔吉斯语需要付出大量努力来创建高质量的句法语料库。本研究提出了一种简化吉尔吉斯语句法语料库开发过程的方法。我们提出了一个基于树库翻译方法的工具，用于将句法注释从土耳其语转移到吉尔吉斯语。我们使用TueCL树库评估了所提出工具的有效性。结果表明，该方法在句法注释准确性方面的表现优于仅在吉尔吉斯语KTMU树库上训练的单语模型。此外，本研究还介绍了一种评估最终生成的句法树复杂性的方法，为优化注释过程提供了进一步的指导。 

---
# Improving Explainability of Sentence-level Metrics via Edit-level Attribution for Grammatical Error Correction 

**Title (ZH)**: 通过编辑级别归因提高句子级指标的可解释性在语法错误纠正中的应用 

**Authors**: Takumi Goto, Justin Vasselli, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2412.13110)  

**Abstract**: Various evaluation metrics have been proposed for Grammatical Error Correction (GEC), but many, particularly reference-free metrics, lack explainability. This lack of explainability hinders researchers from analyzing the strengths and weaknesses of GEC models and limits the ability to provide detailed feedback for users. To address this issue, we propose attributing sentence-level scores to individual edits, providing insight into how specific corrections contribute to the overall performance. For the attribution method, we use Shapley values, from cooperative game theory, to compute the contribution of each edit. Experiments with existing sentence-level metrics demonstrate high consistency across different edit granularities and show approximately 70\% alignment with human evaluations. In addition, we analyze biases in the metrics based on the attribution results, revealing trends such as the tendency to ignore orthographic edits. Our implementation is available at \url{this https URL}. 

**Abstract (ZH)**: 针对语法错误修正（GEC）的各种评价指标已经提出，但其中许多，尤其是无参考指标，缺乏可解释性。这种缺乏可解释性阻碍了研究人员对GEC模型的优势和不足进行分析，并限制了对用户提供详细反馈的能力。为解决这一问题，我们提出将句子级别的评分归因于单个编辑，从而提供对特定修正如何贡献整体性能的理解。在归因方法中，我们利用合作博弈理论中的Shapley值来计算每个编辑的贡献。现有句子级别指标的实验表明，在不同编辑粒度下具有高度一致性，并且与人工评估的对齐程度约为70%。此外，我们根据归因结果分析了指标中的偏见趋势，揭示了诸如忽视拼写修正等倾向。我们的实现可在\url{此链接}获取。 

---
# AI PERSONA: Towards Life-long Personalization of LLMs 

**Title (ZH)**: AI 人格：走向大规模语言模型的终身个性化 

**Authors**: Tiannan Wang, Meiling Tao, Ruoyu Fang, Huilin Wang, Shuai Wang, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13103)  

**Abstract**: In this work, we introduce the task of life-long personalization of large language models. While recent mainstream efforts in the LLM community mainly focus on scaling data and compute for improved capabilities of LLMs, we argue that it is also very important to enable LLM systems, or language agents, to continuously adapt to the diverse and ever-changing profiles of every distinct user and provide up-to-date personalized assistance. We provide a clear task formulation and introduce a simple, general, effective, and scalable framework for life-long personalization of LLM systems and language agents. To facilitate future research on LLM personalization, we also introduce methods to synthesize realistic benchmarks and robust evaluation metrics. We will release all codes and data for building and benchmarking life-long personalized LLM systems. 

**Abstract (ZH)**: 在本文中，我们探讨了大型语言模型终身个性化这一任务。尽管语言模型（LLM）社区近期的主要努力主要集中在通过扩大数据和计算资源来提高语言模型的能力，我们认为同样重要的是使语言模型系统或语言代理能够持续适应每个独特用户的多样且不断变化的特征，并提供及时个性化的辅助。我们提供了一个明确的任务定义，并引入了一个简单、通用、有效且可扩展的框架，用于终身个性化语言模型系统和语言代理。为了促进未来在语言模型个性化方面的研究，我们还介绍了合成现实基准和稳健评估指标的方法。我们将提供构建和基准测试终身个性化语言模型系统所需的所有代码和数据。 

---
# Uchaguzi-2022: A Dataset of Citizen Reports on the 2022 Kenyan Election 

**Title (ZH)**: Uchaguzi-2022：2022年肯尼亚大选的公民报告数据集 

**Authors**: Roberto Mondini, Neema Kotonya, Robert L. Logan IV, Elizabeth M Olson, Angela Oduor Lungati, Daniel Duke Odongo, Tim Ombasa, Hemank Lamba, Aoife Cahill, Joel R. Tetreault, Alejandro Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2412.13098)  

**Abstract**: Online reporting platforms have enabled citizens around the world to collectively share their opinions and report in real time on events impacting their local communities. Systematically organizing (e.g., categorizing by attributes) and geotagging large amounts of crowdsourced information is crucial to ensuring that accurate and meaningful insights can be drawn from this data and used by policy makers to bring about positive change. These tasks, however, typically require extensive manual annotation efforts. In this paper we present Uchaguzi-2022, a dataset of 14k categorized and geotagged citizen reports related to the 2022 Kenyan General Election containing mentions of election-related issues such as official misconduct, vote count irregularities, and acts of violence. We use this dataset to investigate whether language models can assist in scalably categorizing and geotagging reports, thus highlighting its potential application in the AI for Social Good space. 

**Abstract (ZH)**: 在线报道平台使世界各地的市民能够集体分享他们的观点，并实时报告对其所在社区产生影响的事件。系统地组织（例如，按属性分类）和地理标记大量众包信息是确保可以从这些数据中得出准确而有意义的见解，并被政策制定者用于带来积极变革的关键。然而，这些任务通常需要大量的手动注释工作。本文介绍了Uchaguzi-2022数据集，该数据集包含了与2022年肯尼亚总选举相关的14,000份分类和地理标记的市民报告，涉及选举相关问题，如官员不当行为、计票不规则和暴力行为。我们利用该数据集考察语言模型是否能够帮助大规模地分类和地理标记报告，从而突显其在人工智能用于社会公益领域的潜在应用。 

---
# LMUnit: Fine-grained Evaluation with Natural Language Unit Tests 

**Title (ZH)**: LMUnit：基于自然语言单元测试的细粒度评估 

**Authors**: Jon Saad-Falcon, Rajan Vivek, William Berrios, Nandita Shankar Naik, Matija Franklin, Bertie Vidgen, Amanpreet Singh, Douwe Kiela, Shikib Mehri  

**Link**: [PDF](https://arxiv.org/pdf/2412.13091)  

**Abstract**: As language models become integral to critical workflows, assessing their behavior remains a fundamental challenge -- human evaluation is costly and noisy, while automated metrics provide only coarse, difficult-to-interpret signals. We introduce natural language unit tests, a paradigm that decomposes response quality into explicit, testable criteria, along with a unified scoring model, LMUnit, which combines multi-objective training across preferences, direct ratings, and natural language rationales. Through controlled human studies, we show this paradigm significantly improves inter-annotator agreement and enables more effective LLM development workflows. LMUnit achieves state-of-the-art performance on evaluation benchmarks (FLASK, BigGenBench) and competitive results on RewardBench. These results validate both our proposed paradigm and scoring model, suggesting a promising path forward for language model evaluation and development. 

**Abstract (ZH)**: 随着语言模型在关键工作流程中发挥越来越重要的作用，评估其行为仍然是一项基本挑战——人工评估成本高且容易出错，而自动化指标只能提供粗略且难以解释的信号。我们提出了自然语言单元测试这一范式，该范式将响应质量分解为明确且可测试的标准，并结合了一个统一的评分模型LMUnit，该模型结合了多目标训练、直接评价和自然语言理由。通过控制实验的人类研究，我们展示了这一范式显著提高了注释者间的一致性，并使语言模型（LLM）开发流程更为有效。LMUnit在评估基准（FLASK、BigGenBench）上达到了最先进的性能，并在奖励基准（RewardBench）上取得了竞争力的结果。这些结果验证了我们提出的范式和评分模型，表明了为语言模型评估和开发确立一条有前景的道路的可能性。 

---
# CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval 

**Title (ZH)**: CLASP：对比语言-语音预训练在跨语言多模态信息检索中的应用 

**Authors**: Mohammad Mahdi Abootorabi, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2412.13071)  

**Abstract**: This study introduces CLASP (Contrastive Language-Speech Pretraining), a multilingual, multimodal representation tailored for audio-text information retrieval. CLASP leverages the synergy between spoken content and textual data. During training, we utilize our newly introduced speech-text dataset, which encompasses 15 diverse categories ranging from fiction to religion. CLASP's audio component integrates audio spectrograms with a pre-trained self-supervised speech model, while its language encoding counterpart employs a sentence encoder pre-trained on over 100 languages. This unified lightweight model bridges the gap between various modalities and languages, enhancing its effectiveness in handling and retrieving multilingual and multimodal data. Our evaluations across multiple languages demonstrate that CLASP establishes new benchmarks in HITS@1, MRR, and meanR metrics, outperforming traditional ASR-based retrieval approaches in specific scenarios. 

**Abstract (ZH)**: 本文介绍了CLASP（对比语言-语音预训练），这是一种专门为音频-文本信息检索设计的多语言、多模态表示方法。CLASP 利用语音内容与文本数据之间的协同效应。在训练过程中，我们利用了新引入的语音-文本数据集，该数据集包含了15个不同的类别，覆盖从文学到宗教的广泛领域。CLASP 的音频部分将音频频谱图与预训练的自监督语音模型结合，而其语言编码部分则利用了一个在超过100种语言上预训练的句子编码器。这种统一的轻量级模型弥合了不同模态和语言之间的鸿沟，提高了其处理和检索多语言及多模态数据的效果。在多种语言的多个评估指标（如HITS@1、MRR和meanR）上，CLASP 建立了新的基准，并在特定场景下优于传统的基于ASR的信息检索方法。 

---
# Harnessing Event Sensory Data for Error Pattern Prediction in Vehicles: A Language Model Approach 

**Title (ZH)**: 利用事件感知数据进行车辆错误模式预测：一种语言模型方法 

**Authors**: Hugo Math, Rainer Lienhart, Robin Schön  

**Link**: [PDF](https://arxiv.org/pdf/2412.13041)  

**Abstract**: In this paper, we draw an analogy between processing natural languages and processing multivariate event streams from vehicles in order to predict $\textit{when}$ and $\textit{what}$ error pattern is most likely to occur in the future for a given car. Our approach leverages the temporal dynamics and contextual relationships of our event data from a fleet of cars. Event data is composed of discrete values of error codes as well as continuous values such as time and mileage. Modelled by two causal Transformers, we can anticipate vehicle failures and malfunctions before they happen. Thus, we introduce $\textit{CarFormer}$, a Transformer model trained via a new self-supervised learning strategy, and $\textit{EPredictor}$, an autoregressive Transformer decoder model capable of predicting $\textit{when}$ and $\textit{what}$ error pattern will most likely occur after some error code apparition. Despite the challenges of high cardinality of event types, their unbalanced frequency of appearance and limited labelled data, our experimental results demonstrate the excellent predictive ability of our novel model. Specifically, with sequences of $160$ error codes on average, our model is able with only half of the error codes to achieve $80\%$ F1 score for predicting $\textit{what}$ error pattern will occur and achieves an average absolute error of $58.4 \pm 13.2$h $\textit{when}$ forecasting the time of occurrence, thus enabling confident predictive maintenance and enhancing vehicle safety. 

**Abstract (ZH)**: 在这篇论文中，我们将处理自然语言与处理来自车辆的多变量事件流进行类比，以预测给定车辆在未来最有可能出现的错误模式及具体时间。我们的方法利用了车队事件数据的时间动态和上下文关系。事件数据包括错误代码的离散值以及时间、里程等连续值。通过两个因果Transformer模型，我们可以在错误发生之前预见车辆故障和性能下降。因此，我们提出了一个名为$\textit{CarFormer}$的新Transformer模型，它是通过新的自监督学习策略训练的；以及一个自回归Transformer解码器模型$\textit{EPredictor}$，能够预测某错误代码出现后最有可能发生的错误模式及其何时发生。尽管面对事件类型数量众多、出现频率不均以及有限标注数据的挑战，我们的实验结果证明了新型模型具有的出色预测能力。具体而言，当处理平均160个错误代码序列时，我们的模型仅使用一半的错误代码就达到了80%的F1分数用于预测将要发生的错误模式，并且在预测错误模式发生时间时的平均绝对误差为58.4±13.2小时，从而实现了可靠的预测性维护并提升了车辆安全性。 

---
# NAVCON: A Cognitively Inspired and Linguistically Grounded Corpus for Vision and Language Navigation 

**Title (ZH)**: NAVCON：一种受认知启发且基于语言的视觉与语言导航语料库 

**Authors**: Karan Wanchoo, Xiaoye Zuo, Hannah Gonzalez, Soham Dan, Georgios Georgakis, Dan Roth, Kostas Daniilidis, Eleni Miltsakaki  

**Link**: [PDF](https://arxiv.org/pdf/2412.13026)  

**Abstract**: We present NAVCON, a large-scale annotated Vision-Language Navigation (VLN) corpus built on top of two popular datasets (R2R and RxR). The paper introduces four core, cognitively motivated and linguistically grounded, navigation concepts and an algorithm for generating large-scale silver annotations of naturally occurring linguistic realizations of these concepts in navigation instructions. We pair the annotated instructions with video clips of an agent acting on these instructions. NAVCON contains 236, 316 concept annotations for approximately 30, 0000 instructions and 2.7 million aligned images (from approximately 19, 000 instructions) showing what the agent sees when executing an instruction. To our knowledge, this is the first comprehensive resource of navigation concepts. We evaluated the quality of the silver annotations by conducting human evaluation studies on NAVCON samples. As further validation of the quality and usefulness of the resource, we trained a model for detecting navigation concepts and their linguistic realizations in unseen instructions. Additionally, we show that few-shot learning with GPT-4o performs well on this task using large-scale silver annotations of NAVCON. 

**Abstract (ZH)**: 我们提出了NAVCON，这是一个基于两个流行数据集（R2R和RxR）构建的大规模标注视觉语言导航（VLN）语料库。本文介绍了四个核心、认知驱动和语言学依据的导航概念及其生成算法，用于生成大量自然语言实现这些概念的银级标注导航指令。我们还将标注后的指令与执行这些指令的代理视频片段配对。NAVCON包含了约30,000条指令的236,316个概念标注，以及来自约19,000条指令的270万张图像对齐的图像，展示了代理执行指令时所见的场景。据我们所知，这是首个全面的导航概念资源。通过在NAVCON样本上进行人工评估研究，我们评估了这些银级标注的质量。作为进一步验证该资源质量及其实用性的依据，我们训练了一个模型来检测未见过的指令中的导航概念及其语言实现。此外，我们展示了使用NAVCON的大规模银级标注时，GPT-4o在少量样本学习任务上的效果良好。 

---
# OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain 

**Title (ZH)**: OmniEval：金融领域全方位自动RAG评估基准 

**Authors**: Shuting Wang, Jiejun Tan, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.13018)  

**Abstract**: As a typical and practical application of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) techniques have gained extensive attention, particularly in vertical domains where LLMs may lack domain-specific knowledge. In this paper, we introduce an omnidirectional and automatic RAG benchmark, OmniEval, in the financial domain. Our benchmark is characterized by its multi-dimensional evaluation framework, including (1) a matrix-based RAG scenario evaluation system that categorizes queries into five task classes and 16 financial topics, leading to a structured assessment of diverse query scenarios; (2) a multi-dimensional evaluation data generation approach, which combines GPT-4-based automatic generation and human annotation, achieving an 87.47\% acceptance ratio in human evaluations on generated instances; (3) a multi-stage evaluation system that evaluates both retrieval and generation performance, result in a comprehensive evaluation on the RAG pipeline; and (4) robust evaluation metrics derived from rule-based and LLM-based ones, enhancing the reliability of assessments through manual annotations and supervised fine-tuning of an LLM evaluator. Our experiments demonstrate the comprehensiveness of OmniEval, which includes extensive test datasets and highlights the performance variations of RAG systems across diverse topics and tasks, revealing significant opportunities for RAG models to improve their capabilities in vertical domains. We open source the code of our benchmark in \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 作为大型语言模型（LLMs）的一种典型且实用的应用，检索增强生成（RAG）技术引起了广泛关注，特别是在LLMs可能缺乏领域特定知识的垂直领域。本文介绍了一个应用于金融领域的全方位自动RAG基准——OmniEval。我们的基准具有多维度的评估框架，包括以下几点：

1. **矩阵式的RAG场景评估系统**：该系统将查询分为五大任务类别和16个金融主题，从而对各种查询场景进行结构化的评估；
2. **多维度的评估数据生成方法**：该方法结合基于GPT-4的自动生成和人工注释，生成的样本在人类评估中获得了87.47%的接受率；
3. **多阶段评估系统**：该系统评估检索和生成性能，从而对RAG管道进行全面评估；
4. **稳健的评估指标**：这些指标来源于基于规则的和基于LLM的指标，通过人工注释和监督微调LLM评估器，增强了评估的可靠性。

我们的实验表明，OmniEval的全面性体现在其丰富的测试数据集上，突显了RAG系统在不同主题和任务上的性能差异，揭示了RAG模型在垂直领域提高其能力的巨大潜力。我们已在 \href{this https URL}{这个链接} 开放了该基准代码。 

---
# RCLMuFN: Relational Context Learning and Multiplex Fusion Network for Multimodal Sarcasm Detection 

**Title (ZH)**: RCLMuFN：关系上下文学习与多层融合网络在多模态 sarcasm 识别中的应用 

**Authors**: Tongguan Wang, Junkai Li, Guixin Su, Yongcheng Zhang, Dongyu Su, Yuxue Hu, Ying Sha  

**Link**: [PDF](https://arxiv.org/pdf/2412.13008)  

**Abstract**: Sarcasm typically conveys emotions of contempt or criticism by expressing a meaning that is contrary to the speaker's true intent. Accurate detection of sarcasm aids in identifying and filtering undesirable information on the Internet, thereby reducing malicious defamation and rumor-mongering. Nonetheless, the task of automatic sarcasm detection remains highly challenging for machines, as it critically depends on intricate factors such as relational context. Most existing multimodal sarcasm detection methods focus on introducing graph structures to establish entity relationships between text and images while neglecting to learn the relational context between text and images, which is crucial evidence for understanding the meaning of sarcasm. In addition, the meaning of sarcasm changes with the evolution of different contexts, but existing methods may not be accurate in modeling such dynamic changes, limiting the generalization ability of the models. To address the above issues, we propose a relational context learning and multiplex fusion network (RCLMuFN) for multimodal sarcasm detection. Firstly, we employ four feature extractors to comprehensively extract features from raw text and images, aiming to excavate potential features that may have been previously overlooked. Secondly, we utilize the relational context learning module to learn the contextual information of text and images and capture the dynamic properties through shallow and deep interactions. Finally, we employ a multiplex feature fusion module to enhance the generalization of the model by penetratingly integrating multimodal features derived from various interaction contexts. Extensive experiments on two multimodal sarcasm detection datasets show that our proposed method achieves state-of-the-art performance. 

**Abstract (ZH)**: 讽刺通常通过表达与其说话者的真正意图相反的意义来传达轻蔑或批评的情绪。准确检测讽刺有助于识别和过滤互联网上的不良信息，从而减少恶意诽谤和谣言传播。然而，自动讽刺检测任务对于机器来说仍然极具挑战性，因为它高度依赖于复杂的关系背景因素。现有的大多数多模态讽刺检测方法专注于引入图结构以在文本和图像之间建立实体关系，却忽略了学习文本和图像之间的重要关系背景，这是理解讽刺含义的重要证据。此外，讽刺的意义随着不同背景的演变而变化，但现有方法可能无法准确建模这种动态变化，限制了模型的泛化能力。为了解决上述问题，我们提出了一种基于关系背景学习和多层融合网络（RCLMuFN）的多模态讽刺检测方法。首先，我们使用四种特征提取器全面从原始文本和图像中提取特征，以发掘此前可能被忽视的潜在特征。其次，我们利用关系背景学习模块学习文本和图像的背景信息，并通过浅层和深层交互捕获动态属性。最后，我们采用多层特征融合模块，通过深入整合来自多种交互背景的不同模态特征，增强模型的泛化能力。在两个多模态讽刺检测数据集上进行的广泛实验表明，我们所提出的方法达到了最先进的性能。 

---
# Enabling Low-Resource Language Retrieval: Establishing Baselines for Urdu MS MARCO 

**Title (ZH)**: 启用低资源语言检索：为乌尔都语MS MARCO建立基线 

**Authors**: Umer Butt, Stalin Veranasi, Günter Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2412.12997)  

**Abstract**: As the Information Retrieval (IR) field increasingly recognizes the importance of inclusivity, addressing the needs of low-resource languages remains a significant challenge. This paper introduces the first large-scale Urdu IR dataset, created by translating the MS MARCO dataset through machine translation. We establish baseline results through zero-shot learning for IR in Urdu and subsequently apply the mMARCO multilingual IR methodology to this newly translated dataset. Our findings demonstrate that the fine-tuned model (Urdu-mT5-mMARCO) achieves a Mean Reciprocal Rank (MRR@10) of 0.247 and a Recall@10 of 0.439, representing significant improvements over zero-shot results and showing the potential for expanding IR access for Urdu speakers. By bridging access gaps for speakers of low-resource languages, this work not only advances multilingual IR research but also emphasizes the ethical and societal importance of inclusive IR technologies. This work provides valuable insights into the challenges and solutions for improving language representation and lays the groundwork for future research, especially in South Asian languages, which can benefit from the adaptable methods used in this study. 

**Abstract (ZH)**: 随着信息检索（IR）领域越来越认识到包容性的重要性，满足低资源语言的需求仍然是一个重大挑战。本文引入了首个大规模巴基斯坦语（Urdu）IR数据集，通过机器翻译将MS MARCO数据集翻译而来。我们通过零样本学习方法为巴基斯坦语IR建立了基线结果，随后将mMARCO多语言IR方法应用于这一新翻译的数据集。我们的研究发现，微调后的模型（Urdu-mT5-mMARCO）在Mean Reciprocal Rank (MRR@10) 上达到了0.247，Recall@10达到了0.439，这表明该模型相较于零样本学习方法取得了显著提高，显示出扩展巴基斯坦语用户IR访问的潜力。通过为低资源语言的使用者填补访问缺口，这项工作不仅推进了多语言IR研究，还强调了包容性IR技术的道德和社会意义。本文提供了有关提升语言表示及其解决方案的重要见解，为进一步研究奠定了基础，特别是在南亚语言方面，这些语言可以从本研究中使用的可适应方法中获益。 

---
# Unlocking LLMs: Addressing Scarce Data and Bias Challenges in Mental Health 

**Title (ZH)**: 解锁大语言模型：应对心理健康领域数据稀缺性和偏见挑战 

**Authors**: Vivek Kumar, Eirini Ntoutsi, Pushpraj Singh Rajawat, Giacomo Medda, Diego Reforgiato Recupero  

**Link**: [PDF](https://arxiv.org/pdf/2412.12981)  

**Abstract**: Large language models (LLMs) have shown promising capabilities in healthcare analysis but face several challenges like hallucinations, parroting, and bias manifestation. These challenges are exacerbated in complex, sensitive, and low-resource domains. Therefore, in this work we introduce IC-AnnoMI, an expert-annotated motivational interviewing (MI) dataset built upon AnnoMI by generating in-context conversational dialogues leveraging LLMs, particularly ChatGPT. IC-AnnoMI employs targeted prompts accurately engineered through cues and tailored information, taking into account therapy style (empathy, reflection), contextual relevance, and false semantic change. Subsequently, the dialogues are annotated by experts, strictly adhering to the Motivational Interviewing Skills Code (MISC), focusing on both the psychological and linguistic dimensions of MI dialogues. We comprehensively evaluate the IC-AnnoMI dataset and ChatGPT's emotional reasoning ability and understanding of domain intricacies by modeling novel classification tasks employing several classical machine learning and current state-of-the-art transformer approaches. Finally, we discuss the effects of progressive prompting strategies and the impact of augmented data in mitigating the biases manifested in IC-AnnoM. Our contributions provide the MI community with not only a comprehensive dataset but also valuable insights for using LLMs in empathetic text generation for conversational therapy in supervised settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗分析方面展现出了潜在的能力，但仍面临着幻觉、鹦鹉学舌和偏见显现等一系列挑战。这些挑战在复杂、敏感且资源有限的领域内尤为突出。因此，在本工作中，我们提出了IC-AnnoMI数据集，这是一个由专家标注的动机性访谈（MI）数据集。IC-AnnoMI基于AnnoMI构建，并利用LLM（特别是ChatGPT）生成上下文相关的对话。IC-AnnoMI采用了针对性的提示，这些提示是通过关键线索和定制化信息准确工程化设计的，涵盖了治疗风格（共情、反思），上下文相关性以及虚假语义变化。随后，这些对话由专家进行标注，并严格遵循动机性访谈技能编码规范（MISC），关注MI对话的心理学和语言学维度。我们通过使用多种经典机器学习方法和当前最先进的变换器方法，构建新颖的分类任务，对IC-AnnoMI数据集以及ChatGPT的情感推理能力和对领域复杂性的理解进行全面评估。最后，我们讨论了渐进式提示策略和增强数据对减轻IC-AnnoM中显现偏见的影响。我们的贡献不仅为MI社区提供了一个全面的数据集，而且还为在监督设置中利用LLM进行共情文本生成的对话治疗提供了宝贵见解。 

---
# Adaptations of AI models for querying the LandMatrix database in natural language 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

Adaptations of AI models for querying the LandMatrix database in natural language

人工智能模型用于自然语言查询LandMatrix数据库的适应性研究 

**Authors**: Fatiha Ait Kbir, Jérémy Bourgoin, Rémy Decoupes, Marie Gradeler, Roberto Interdonato  

**Link**: [PDF](https://arxiv.org/pdf/2412.12961)  

**Abstract**: The Land Matrix initiative (this https URL) and its global observatory aim to provide reliable data on large-scale land acquisitions to inform debates and actions in sectors such as agriculture, extraction, or energy in low- and middle-income countries. Although these data are recognized in the academic world, they remain underutilized in public policy, mainly due to the complexity of access and exploitation, which requires technical expertise and a good understanding of the database schema.
The objective of this work is to simplify access to data from different database systems. The methods proposed in this article are evaluated using data from the Land Matrix. This work presents various comparisons of Large Language Models (LLMs) as well as combinations of LLM adaptations (Prompt Engineering, RAG, Agents) to query different database systems (GraphQL and REST queries). The experiments are reproducible, and a demonstration is available online: this https URL. 

**Abstract (ZH)**: 《土地矩阵倡议》（查阅：[this link](this https URL)）及其全球观测站旨在提供可靠的大规模土地收购数据，以支持低收入和中等收入国家农业、开采或能源等领域内的辩论和行动。尽管这些数据在学术界得到了认可，但在公共政策中却相对未得到充分利用，主要原因是访问和利用这些数据的复杂性，这需要技术和专业知识以及对数据库结构的良好理解。

本文的目标是简化对不同数据库系统的数据访问。本文中提出的方法使用来自《土地矩阵》的数据进行了评估。本文展示了各种大型语言模型（LLM）的比较，以及LLM适应（Prompt工程、RAG、代理）的组合用于查询不同数据库系统（GraphQL和REST查询）。实验是可重复的，可以在网上查看演示：[this link](this https URL)。 

---
# SnakModel: Lessons Learned from Training an Open Danish Large Language Model 

**Title (ZH)**: SnakModel：训练开源丹麦大型语言模型的经验教训 

**Authors**: Mike Zhang, Max Müller-Eberstein, Elisa Bassignana, Rob van der Goot  

**Link**: [PDF](https://arxiv.org/pdf/2412.12956)  

**Abstract**: We present SnakModel, a Danish large language model (LLM) based on Llama2-7B, which we continuously pre-train on 13.6B Danish words, and further tune on 3.7M Danish instructions. As best practices for creating LLMs for smaller language communities have yet to be established, we examine the effects of early modeling and training decisions on downstream performance throughout the entire training pipeline, including (1) the creation of a strictly curated corpus of Danish text from diverse sources; (2) the language modeling and instruction-tuning training process itself, including the analysis of intermediate training dynamics, and ablations across different hyperparameters; (3) an evaluation on eight language and culturally-specific tasks. Across these experiments SnakModel achieves the highest overall performance, outperforming multiple contemporary Llama2-7B-based models. By making SnakModel, the majority of our pre-training corpus, and the associated code available under open licenses, we hope to foster further research and development in Danish Natural Language Processing, and establish training guidelines for languages with similar resource constraints. 

**Abstract (ZH)**: 我们提出了SnakModel，这是一个基于Llama2-7B的丹麦大型语言模型（LLM），我们对其进行了连续预训练，使用了136亿个丹麦语词汇，并进一步使用了370万条丹麦指令进行调优。由于为较小语言社区创建LLM的最佳做法尚未建立，我们考察了从建模到训练的各个环节中早起决策对下游性能的影响，包括（1）从各种来源严格筛选并创建丹麦文本语料库；（2）语言模型和指令调优的训练过程本身，包括对中间训练动态的分析以及不同超参数的削减实验；（3）对八项语言和文化特定任务的评估。在这些实验中，SnakModel取得了最优的整体性能，超越了多个基于Llama2-7B的现代模型。通过将SnakModel、大部分预训练语料库及相关的代码以开源许可发布，我们希望促进丹麦自然语言处理的研究与开发，并为资源有限的类似语言的训练制定指导方针。 

---
# Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling 

**Title (ZH)**: 基于自适应实时元损失缩放的学习噪声标签方法 

**Authors**: Michael Heck, Christian Geishauser, Nurul Lubis, Carel van Niekerk, Shutong Feng, Hsien-Chin Lin, Benjamin Matthias Ruppik, Renato Vukovic, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2412.12955)  

**Abstract**: Correct labels are indispensable for training effective machine learning models. However, creating high-quality labels is expensive, and even professionally labeled data contains errors and ambiguities. Filtering and denoising can be applied to curate labeled data prior to training, at the cost of additional processing and loss of information. An alternative is on-the-fly sample reweighting during the training process to decrease the negative impact of incorrect or ambiguous labels, but this typically requires clean seed data. In this work we propose unsupervised on-the-fly meta loss rescaling to reweight training samples. Crucially, we rely only on features provided by the model being trained, to learn a rescaling function in real time without knowledge of the true clean data distribution. We achieve this via a novel meta learning setup that samples validation data for the meta update directly from the noisy training corpus by employing the rescaling function being trained. Our proposed method consistently improves performance across various NLP tasks with minimal computational overhead. Further, we are among the first to attempt on-the-fly training data reweighting on the challenging task of dialogue modeling, where noisy and ambiguous labels are common. Our strategy is robust in the face of noisy and clean data, handles class imbalance, and prevents overfitting to noisy labels. Our self-taught loss rescaling improves as the model trains, showing the ability to keep learning from the model's own signals. As training progresses, the impact of correctly labeled data is scaled up, while the impact of wrongly labeled data is suppressed. 

**Abstract (ZH)**: 优质的标签对于训练有效的机器学习模型至关重要。然而，生成高质量的标签成本高昂，即使是由专业人员标注的数据也可能存在错误和模糊性。在训练前通过过滤和降噪来处理标注数据可以减少额外的处理过程和信息损失，但这可能会导致信息的丢失。一种替代方法是在训练过程中实时对样本进行重新加权，以减少错误或模糊标签的负面影响，但这通常需要干净的种子数据。在本项工作中，我们提出了一种无监督的实时元损失调整方法，用于重新加权训练样本。关键之处在于我们仅依赖于正在训练的模型提供的特征，无需了解真正的清洁数据分布，即可实时学习一个加权函数。我们通过一种新颖的元学习设置实现了这一点，即通过应用正在训练的加权函数直接从嘈杂的训练语料库中采样验证数据进行元更新。我们提出的这种方法在各种自然语言处理任务中实现了性能的持续提升，且计算开销较小。此外，我们是首次尝试在对话建模这一具有挑战性的任务中进行实时训练数据重新加权，这是一个常见噪声和模糊标签的任务。我们的策略在面对嘈杂和清洁数据时表现出鲁棒性，能够处理类别不平衡，并防止对噪声标签的过拟合。随着模型的训练，我们自教的损失调整会逐渐改善，显示了通过模型自身信号持续学习的能力。随着训练的进行，正确标注的数据的影响逐步增强，而错误标注的数据的影响则被抑制。 

---
# Recipient Profiling: Predicting Characteristics from Messages 

**Title (ZH)**: recipient 筛选：从消息中预测特征 

**Authors**: Martin Borquez, Mikaela Keller, Michael Perrot, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2412.12954)  

**Abstract**: It has been shown in the field of Author Profiling that texts may inadvertently reveal sensitive information about their authors, such as gender or age. This raises important privacy concerns that have been extensively addressed in the literature, in particular with the development of methods to hide such information. We argue that, when these texts are in fact messages exchanged between individuals, this is not the end of the story. Indeed, in this case, a second party, the intended recipient, is also involved and should be considered. In this work, we investigate the potential privacy leaks affecting them, that is we propose and address the problem of Recipient Profiling. We provide empirical evidence that such a task is feasible on several publicly accessible datasets (this https URL). Furthermore, we show that the learned models can be transferred to other datasets, albeit with a loss in accuracy. 

**Abstract (ZH)**: 在作者画像领域中，已有研究表明文本可能会无意间暴露作者的敏感信息，如性别或年龄。这引发了重要的隐私问题，并在文献中得到了广泛的关注，尤其是在开发隐藏此类信息的方法方面。我们认为，当这些文本实际上是个体之间交流的信息时，这个问题并没有结束。事实上，在这种情况下，第二个相关方，即预期接收者，也涉及其中并应被考虑。在本项研究中，我们探讨了对预期接收者可能产生的隐私泄露问题，即我们提出并解决了一种受信者画像的问题。我们提供了实证证据表明，在多个公开可访问的数据集上进行此类任务是可行的（请参阅此链接：[提供链接]）。此外，我们展示了所学习的模型可以迁移到其他数据集，尽管准确度会有所下降。 

---
# MOPO: Multi-Objective Prompt Optimization for Affective Text Generation 

**Title (ZH)**: MOPO：情感文本生成的多目标提示优化 

**Authors**: Yarik Menchaca Resendiz, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.12948)  

**Abstract**: How emotions are expressed depends on the context and domain. On X (formerly Twitter), for instance, an author might simply use the hashtag #anger, while in a news headline, emotions are typically written in a more polite, indirect manner. To enable conditional text generation models to create emotionally connotated texts that fit a domain, users need to have access to a parameter that allows them to choose the appropriate way to express an emotion. To achieve this, we introduce MOPO, a Multi-Objective Prompt Optimization methodology. MOPO optimizes prompts according to multiple objectives (which correspond here to the output probabilities assigned by emotion classifiers trained for different domains). In contrast to single objective optimization, MOPO outputs a set of prompts, each with a different weighting of the multiple objectives. Users can then choose the most appropriate prompt for their context. We evaluate MOPO using three objectives, determined by various domain-specific emotion classifiers. MOPO improves performance by up to 15 pp across all objectives with a minimal loss (1-2 pp) for any single objective compared to single-objective optimization. These minor performance losses are offset by a broader generalization across multiple objectives - which is not possible with single-objective optimization. Additionally, MOPO reduces computational requirements by simultaneously optimizing for multiple objectives, eliminating separate optimization procedures for each objective. 

**Abstract (ZH)**: 情感的表达依赖于具体情境和领域。例如，在On X（原Twitter）上，作者可能会简单地使用带有#anger的标签，而在新闻标题中，情感则通常以更加礼貌和委婉的方式表达。为了使条件生成模型能够生成符合特定领域的情感化文本，用户需要能够调整一个参数以选择适合表达情感的方式。为此，我们提出了一种多目标提示优化方法（MOPO）——多目标提示优化。MOPO 根据多个目标优化提示（这些目标对应于不同领域训练的情感分类器所赋的输出概率）。与单一目标优化不同，MOPO 输出一组带有不同目标权重的提示。用户可以根据具体情境选择最合适的提示。我们通过三个由不同领域特定情感分类器确定的目标来评估MOPO。在所有目标上，MOPO 的性能提高了最多15个百分点，而任何单一目标的性能损失仅减少了1-2个百分点。这些较小的性能损失通过在多个目标上的更广泛泛化得到了弥补——这是单一目标优化所不能实现的。此外，通过同时优化多个目标，MOPO 还减少了计算需求，避免了对每个目标单独进行优化的过程。 

---
# Improving Fine-grained Visual Understanding in VLMs through Text-Only Training 

**Title (ZH)**: 通过仅文本训练提高VLMs的细粒度视觉理解 

**Authors**: Dasol Choi, Guijin Son, Soo Yong Kim, Gio Paik, Seunghyeok Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.12940)  

**Abstract**: Visual-Language Models (VLMs) have become a powerful tool for bridging the gap between visual and linguistic understanding. However, the conventional learning approaches for VLMs often suffer from limitations, such as the high resource requirements of collecting and training image-text paired data. Recent research has suggested that language understanding plays a crucial role in the performance of VLMs, potentially indicating that text-only training could be a viable approach. In this work, we investigate the feasibility of enhancing fine-grained visual understanding in VLMs through text-only training. Inspired by how humans develop visual concept understanding, where rich textual descriptions can guide visual recognition, we hypothesize that VLMs can also benefit from leveraging text-based representations to improve their visual recognition abilities. We conduct comprehensive experiments on two distinct domains: fine-grained species classification and cultural visual understanding tasks. Our findings demonstrate that text-only training can be comparable to conventional image-text training while significantly reducing computational costs. This suggests a more efficient and cost-effective pathway for advancing VLM capabilities, particularly valuable in resource-constrained environments. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）已成为连接视觉理解和语言理解的有力工具。然而，传统的VLMs学习方法往往受到限制，如收集和训练图像-文本配对数据所需的高资源需求。近期的研究表明，语言理解在VLMs的表现中扮演着关键角色，这可能意味着仅通过文本进行训练是一个可行的方法。在这项工作中，我们探讨了通过仅文本训练增强VLMs细粒度视觉理解的可行性。受到人类发展视觉概念理解的启发，其中丰富的文本描述可以引导视觉识别，我们假设VLMs也可以从利用基于文本的表示来提高其视觉识别能力中受益。我们在两个不同的领域进行了全面的实验：细粒度物种分类和文化视觉理解任务。我们的研究发现表明，仅文本训练在计算成本显著降低的同时，可以与传统的图像-文本训练相媲美。这表明了一条更高效和成本效益更高的路径，用于推进VLM的能力，特别是在资源受限的环境中尤其有价值。 

---
# Truthful Text Sanitization Guided by Inference Attacks 

**Title (ZH)**: 由推理攻击引导的truthful文本 sanitization 

**Authors**: Ildikó Pilán, Benet Manzanares-Salor, David Sánchez, Pierre Lison  

**Link**: [PDF](https://arxiv.org/pdf/2412.12928)  

**Abstract**: The purpose of text sanitization is to rewrite those text spans in a document that may directly or indirectly identify an individual, to ensure they no longer disclose personal information. Text sanitization must strike a balance between preventing the leakage of personal information (privacy protection) while also retaining as much of the document's original content as possible (utility preservation). We present an automated text sanitization strategy based on generalizations, which are more abstract (but still informative) terms that subsume the semantic content of the original text spans. The approach relies on instruction-tuned large language models (LLMs) and is divided into two stages. The LLM is first applied to obtain truth-preserving replacement candidates and rank them according to their abstraction level. Those candidates are then evaluated for their ability to protect privacy by conducting inference attacks with the LLM. Finally, the system selects the most informative replacement shown to be resistant to those attacks. As a consequence of this two-stage process, the chosen replacements effectively balance utility and privacy. We also present novel metrics to automatically evaluate these two aspects without the need to manually annotate data. Empirical results on the Text Anonymization Benchmark show that the proposed approach leads to enhanced utility, with only a marginal increase in the risk of re-identifying protected individuals compared to fully suppressing the original information. Furthermore, the selected replacements are shown to be more truth-preserving and abstractive than previous methods. 

**Abstract (ZH)**: 文本净化的目的是重写文档中可能直接或间接识别个体的文本片段，以确保不再泄露个人隐私信息。文本净化需要在防止个人信息泄露（隐私保护）与保留文档尽可能多的原始内容（实用保留）之间找到平衡。本文提出了一种基于泛化的自动文本净化策略，其中泛化是指更具抽象性但仍具有信息性的术语，可以概括原始文本片段的语义内容。该方法依赖于指令调优的大语言模型（LLMs），并分为两个阶段。首先应用LLM获取事实保留的替代候选，并根据其抽象层次进行排名。然后利用LLM进行推理攻击来评估这些候选替代保护隐私的能力。最后，系统选择能够抵抗这些攻击且最具有信息性替代。这一两阶段过程的结果是选择的替代能够有效平衡实用性和隐私性。此外，本文还提出了新的评估指标，可以自动评估这两个方面，而无需手动标注数据。在文本匿名化基准测试集上的实证结果显示，所提出的方法在与完全抑制原始信息相比，仅增加了微小的重新识别受保护个体的风险的同时，实现了更高的实用性。此外，所选替代在事实保持性和抽象性方面优于先前的方法。 

---
# Question: How do Large Language Models perform on the Question Answering tasks? Answer: 

**Title (ZH)**: 问题：大规模语言模型在问答任务中的表现如何？
回答： 

**Authors**: Kevin Fischer, Darren Fürst, Sebastian Steindl, Jakob Lindner, Ulrich Schäfer  

**Link**: [PDF](https://arxiv.org/pdf/2412.12893)  

**Abstract**: Large Language Models (LLMs) have been showing promising results for various NLP-tasks without the explicit need to be trained for these tasks by using few-shot or zero-shot prompting techniques. A common NLP-task is question-answering (QA). In this study, we propose a comprehensive performance comparison between smaller fine-tuned models and out-of-the-box instruction-following LLMs on the Stanford Question Answering Dataset 2.0 (SQuAD2), specifically when using a single-inference prompting technique. Since the dataset contains unanswerable questions, previous work used a double inference method. We propose a prompting style which aims to elicit the same ability without the need for double inference, saving compute time and resources. Furthermore, we investigate their generalization capabilities by comparing their performance on similar but different QA datasets, without fine-tuning neither model, emulating real-world uses where the context and questions asked may differ from the original training distribution, for example swapping Wikipedia for news articles.
Our results show that smaller, fine-tuned models outperform current State-Of-The-Art (SOTA) LLMs on the fine-tuned task, but recent SOTA models are able to close this gap on the out-of-distribution test and even outperform the fine-tuned models on 3 of the 5 tested QA datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不需要明确针对特定自然语言处理（NLP）任务进行训练的情况下，通过少样本或零样本提示技术，已经显示出在各种NLP任务中的有前途的结果。常见的NLP任务之一是问答（QA）。在本研究中，我们提出了一种在斯坦福问答数据集2.0（SQuAD2）上对较小的微调模型和现成的指令跟随LLMs进行全面性能比较的方法，特别是在使用单一推理提示技术的情况下。由于数据集中包含无法回答的问题，以往的工作采用了双推理方法。我们提出了一种提示风格，旨在在不需要双推理的情况下激发相同的能力，从而节省计算时间和资源。此外，我们通过在不进行微调的情况下比较它们在类似但不同的QA数据集上的性能，研究了它们的泛化能力，模拟了实际使用场景，其中上下文和提出的问题可能与原始训练分布不同，例如用维基百科文章替换新闻文章。

我们的结果显示，在微调任务上，较小的微调模型优于当前最先进的（SOTA）LLMs，但最近的SOTA模型能够在分布外测试中缩小这一差距，并在所测试的5个QA数据集中有3个数据集上甚至超过了微调模型。 

---
# RAG-Star: Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement 

**Title (ZH)**: RAG-Star: 基于检索增强验证与精炼的 deliberative reasoning 提升方法 

**Authors**: Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie Wang, Wayne Xin Zhao, Yang Song, Tao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12881)  

**Abstract**: Existing large language models (LLMs) show exceptional problem-solving capabilities but might struggle with complex reasoning tasks. Despite the successes of chain-of-thought and tree-based search methods, they mainly depend on the internal knowledge of LLMs to search over intermediate reasoning steps, limited to dealing with simple tasks involving fewer reasoning steps. In this paper, we propose \textbf{RAG-Star}, a novel RAG approach that integrates the retrieved information to guide the tree-based deliberative reasoning process that relies on the inherent knowledge of LLMs. By leveraging Monte Carlo Tree Search, RAG-Star iteratively plans intermediate sub-queries and answers for reasoning based on the LLM itself. To consolidate internal and external knowledge, we propose an retrieval-augmented verification that utilizes query- and answer-aware reward modeling to provide feedback for the inherent reasoning of LLMs. Our experiments involving Llama-3.1-8B-Instruct and GPT-4o demonstrate that RAG-Star significantly outperforms previous RAG and reasoning methods. 

**Abstract (ZH)**: 现有的大型语言模型（LLMs）显示出卓越的问题解决能力，但在处理复杂的推理任务时可能会遇到困难。尽管链式思考和树状搜索方法取得了成功，但它们主要依赖于LLMs的内部知识来进行中间推理步骤的搜索，局限在处理涉及较少推理步骤的简单任务上。在本文中，我们提出了一种新的RAG方法——\textbf{RAG-Star}，该方法将检索到的信息用于指导依赖于LLMs内部知识的树状讨论性推理过程。通过利用蒙特卡洛树搜索（MCTS），RAG-Star迭代地规划基于LLMs本身的中间子查询和答案来进行推理。为了整合内部和外部知识，我们提出了一种检索增强验证方法，利用查询和答案感知的奖励建模为LLMs的内在推理提供反馈。我们的实验涉及Llama-3.1-8B-Instruct和GPT-4o表明，RAG-Star在之前的RAG和推理方法上表现出显著的优越性。 

---
# Preference-Oriented Supervised Fine-Tuning: Favoring Target Model Over Aligned Large Language Models 

**Title (ZH)**: 面向偏好的监督微调：优先考虑目标模型而非对齐的大语言模型 

**Authors**: Yuchen Fan, Yuzhong Hong, Qiushi Wang, Junwei Bao, Hongfei Jiang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.12865)  

**Abstract**: Alignment, endowing a pre-trained Large language model (LLM) with the ability to follow instructions, is crucial for its real-world applications. Conventional supervised fine-tuning (SFT) methods formalize it as causal language modeling typically with a cross-entropy objective, requiring a large amount of high-quality instruction-response pairs. However, the quality of widely used SFT datasets can not be guaranteed due to the high cost and intensive labor for the creation and maintenance in practice. To overcome the limitations associated with the quality of SFT datasets, we introduce a novel \textbf{p}reference-\textbf{o}riented supervised \textbf{f}ine-\textbf{t}uning approach, namely PoFT. The intuition is to boost SFT by imposing a particular preference: \textit{favoring the target model over aligned LLMs on the same SFT data.} This preference encourages the target model to predict a higher likelihood than that predicted by the aligned LLMs, incorporating assessment information on data quality (i.e., predicted likelihood by the aligned LLMs) into the training process. Extensive experiments are conducted, and the results validate the effectiveness of the proposed method. PoFT achieves stable and consistent improvements over the SFT baselines across different training datasets and base models. Moreover, we prove that PoFT can be integrated with existing SFT data filtering methods to achieve better performance, and further improved by following preference optimization procedures, such as DPO. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，需符合学术规范：

预训练大型语言模型（LLM）具备遵循指令的能力是其在现实世界应用中的关键。传统监督微调（SFT）方法将其形式化为因果语言模型，并通常采用交叉熵目标函数，需要大量的高质量指令-响应对。然而，由于实际中创建和维护高质量SFT数据集成本高昂且需要大量劳动力，因此广泛使用的SFT数据集的质量无法得到保证。为克服与SFT数据集质量相关的问题，我们提出了一种新颖的基于偏好的监督微调方法，即PoFT（Preference-Oriented Supervised Fine-Tuning）。其核心思路是通过施加特定偏好增强SFT：在相同的SFT数据上，偏好目标模型优于对齐的LLM。这一偏好促使目标模型预测的概率高于对齐的LLM所预测的概率，将数据质量评估信息（即对齐的LLM预测的概率）纳入训练过程。进行了大量的实验，结果验证了所提出方法的有效性。PoFT在各类训练数据集和基础模型中均实现了稳定且一致的改进。此外，我们证明PoFT可以与现有的SFT数据过滤方法结合使用以获得更好的性能，并可以通过遵循偏好优化程序（如DPO）进一步改进。 

---
# DISC: Plug-and-Play Decoding Intervention with Similarity of Characters for Chinese Spelling Check 

**Title (ZH)**: DISC：基于字符相似性的即插即用解码干预方案用于中文拼写检查 

**Authors**: Ziheng Qiao, Houquan Zhou, Yumeng Liu, Zhenghua Li, Min Zhang, Bo Zhang, Chen Li, Ji Zhang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12863)  

**Abstract**: One key characteristic of the Chinese spelling check (CSC) task is that incorrect characters are usually similar to the correct ones in either phonetics or glyph. To accommodate this, previous works usually leverage confusion sets, which suffer from two problems, i.e., difficulty in determining which character pairs to include and lack of probabilities to distinguish items in the set. In this paper, we propose a light-weight plug-and-play DISC (i.e., decoding intervention with similarity of characters) module for CSC this http URL measures phonetic and glyph similarities between characters and incorporates this similarity information only during the inference phase. This method can be easily integrated into various existing CSC models, such as ReaLiSe, SCOPE, and ReLM, without additional training costs. Experiments on three CSC benchmarks demonstrate that our proposed method significantly improves model performance, approaching and even surpassing the current state-of-the-art models. 

**Abstract (ZH)**: 中国拼写检查（CSC）任务的一个关键特征是错误字符通常在音韵或笔画上与正确字符相似。为了适应这一特征，以往的工作通常利用混淆集来识别错误字符，但这种方法存在两个问题：难以确定哪些字符对应该包含在混淆集中，以及缺乏区分混淆集内项的概率信息。本文提出了一种轻量级的即插即用DISC模块（即，基于字符相似性的解码干预），用于改进CSC任务。该模块在推理阶段仅利用字符间的音韵和笔画相似度进行计算，而不需要额外的训练费用。在三个不同CSC基准数据集上的实验表明，该方法显著提升了模型性能，接近甚至超过了当前最先进的模型。 

---
# Benchmarking and Understanding Compositional Relational Reasoning of LLMs 

**Title (ZH)**: 《基准测试与理解LLMs的组合关系推理能力》 

**Authors**: Ruikang Ni, Da Xiao, Qingye Meng, Xiangyu Li, Shihui Zheng, Hongliang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12841)  

**Abstract**: Compositional relational reasoning (CRR) is a hallmark of human intelligence, but we lack a clear understanding of whether and how existing transformer large language models (LLMs) can solve CRR tasks. To enable systematic exploration of the CRR capability of LLMs, we first propose a new synthetic benchmark called Generalized Associative Recall (GAR) by integrating and generalizing the essence of several tasks in mechanistic interpretability (MI) study in a unified framework. Evaluation shows that GAR is challenging enough for existing LLMs, revealing their fundamental deficiency in CRR. Meanwhile, it is easy enough for systematic MI study. Then, to understand how LLMs solve GAR tasks, we use attribution patching to discover the core circuits reused by Vicuna-33B across different tasks and a set of vital attention heads. Intervention experiments show that the correct functioning of these heads significantly impacts task performance. Especially, we identify two classes of heads whose activations represent the abstract notion of true and false in GAR tasks respectively. They play a fundamental role in CRR across various models and tasks. The dataset and code are available at this https URL. 

**Abstract (ZH)**: 组合关系推理（CRR）是人类智能的一个 hallmark，但我们尚未明确了解现有的大规模语言模型（LLMs）是否能够解决 CRR 任务，以及它们是如何解决这些任务的。为了系统地探索 LLM 的 CRR 能力，我们首先提出了一种新的合成基准，名为通用关联回忆（GAR），该基准通过对机械可解释性（MI）研究中多个任务的精髓进行整合和泛化，在统一框架下构建。评估结果显示，GAR 对现有的 LLM 具有足够的挑战性，揭示了它们在 CRR 方面的基本不足。同时，这个基准足够简单，适用于系统的 MI 研究。

为了理解 LLM 如何解决 GAR 任务，我们使用归因补丁技术发现 Vicuna-33B 在不同任务中重用的核心电路以及一系列关键的关注头。干预实验表明，这些头的正确功能显著影响任务性能。特别地，我们识别出两类头，它们在 GAR 任务中分别代表真值和假值的概念，它们在各种模型和任务中的 CRR 中发挥着基础性作用。相关数据集和代码可在以下网址获取：[提供网址的文本]。 

---
# DSGram: Dynamic Weighting Sub-Metrics for Grammatical Error Correction in the Era of Large Language Models 

**Title (ZH)**: DSGram：大型语言模型时代基于动态加权子指标的语法错误修正 

**Authors**: Jinxiang Xie, Yilin Li, Xunjian Yin, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2412.12832)  

**Abstract**: Evaluating the performance of Grammatical Error Correction (GEC) models has become increasingly challenging, as large language model (LLM)-based GEC systems often produce corrections that diverge from provided gold references. This discrepancy undermines the reliability of traditional reference-based evaluation metrics. In this study, we propose a novel evaluation framework for GEC models, DSGram, integrating Semantic Coherence, Edit Level, and Fluency, and utilizing a dynamic weighting mechanism. Our framework employs the Analytic Hierarchy Process (AHP) in conjunction with large language models to ascertain the relative importance of various evaluation criteria. Additionally, we develop a dataset incorporating human annotations and LLM-simulated sentences to validate our algorithms and fine-tune more cost-effective models. Experimental results indicate that our proposed approach enhances the effectiveness of GEC model evaluations. 

**Abstract (ZH)**: 评估语法错误修正（GEC）模型的表现越来越具有挑战性，因为基于大规模语言模型（LLM）的GEC系统常常会产生与提供的黄金标准参考偏离较大的修正结果。这种偏离性削弱了传统基于参考的标准评估指标的可靠性。在本研究中，我们提出了一种新的评估框架——DSGram，该框架综合了语义连贯性、编辑级别和流畅性，并采用了动态权重机制。我们的框架结合使用了层次分析过程（AHP）和大规模语言模型来确定各种评估标准的相对重要性。此外，我们还开发了一个包含人类标注和大型语言模型模拟句子的数据集，用于验证我们的算法并微调更经济有效的模型。实验结果表明，我们提出的这种方法增强了GEC模型评估的有效性。 

---
# Detecting Emotional Incongruity of Sarcasm by Commonsense Reasoning 

**Title (ZH)**: 通过常识推理检测讽刺中的情感不一致性 

**Authors**: Ziqi Qiu, Jianxing Yu, Yufeng Zhang, Hanjiang Lai, Yanghui Rao, Qinliang Su, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2412.12808)  

**Abstract**: This paper focuses on sarcasm detection, which aims to identify whether given statements convey criticism, mockery, or other negative sentiment opposite to the literal meaning. To detect sarcasm, humans often require a comprehensive understanding of the semantics in the statement and even resort to external commonsense to infer the fine-grained incongruity. However, existing methods lack commonsense inferential ability when they face complex real-world scenarios, leading to unsatisfactory performance. To address this problem, we propose a novel framework for sarcasm detection, which conducts incongruity reasoning based on commonsense augmentation, called EICR. Concretely, we first employ retrieval-augmented large language models to supplement the missing but indispensable commonsense background knowledge. To capture complex contextual associations, we construct a dependency graph and obtain the optimized topology via graph refinement. We further introduce an adaptive reasoning skeleton that integrates prior rules to extract sentiment-inconsistent subgraphs explicitly. To eliminate the possible spurious relations between words and labels, we employ adversarial contrastive learning to enhance the robustness of the detector. Experiments conducted on five datasets demonstrate the effectiveness of EICR. 

**Abstract (ZH)**: 本文专注于情感挖苦检测，旨在识别给定陈述是否包含与其字面意义相反的批评、嘲讽或其他负面情感。为了检测挖苦，人类通常需要全面理解陈述中的语义，并且甚至需要借助常识推理来推断细微的不一致之处。然而，现有的方法在面对复杂现实场景时缺乏常识推理能力，导致其性能不尽如人意。为此，我们提出了一种新的挖苦检测框架，该框架基于常识增强进行不一致性推理，称为EICR。具体而言，我们首先使用检索增强的大语言模型来补充缺失但必不可少的常识背景知识。为捕捉复杂的上下文关联，我们构建了一个依赖图，并通过图优化获得优化的拓扑结构。进一步地，我们引入了一种适应性的推理骨架，将其与先验规则相结合以明确提取情感不一致的子图。为了消除词与标签之间可能的虚假关联，我们采用对抗对比学习以增强检测器的鲁棒性。在五个数据集上的实验表明了EICR的有效性。 

---
# Cross-Dialect Information Retrieval: Information Access in Low-Resource and High-Variance Languages 

**Title (ZH)**: 跨方言信息检索：低资源和高变异性语言中的信息访问 

**Authors**: Robert Litschko, Oliver Kraus, Verena Blaschke, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2412.12806)  

**Abstract**: A large amount of local and culture-specific knowledge (e.g., people, traditions, food) can only be found in documents written in dialects. While there has been extensive research conducted on cross-lingual information retrieval (CLIR), the field of cross-dialect retrieval (CDIR) has received limited attention. Dialect retrieval poses unique challenges due to the limited availability of resources to train retrieval models and the high variability in non-standardized languages. We study these challenges on the example of German dialects and introduce the first German dialect retrieval dataset, dubbed WikiDIR, which consists of seven German dialects extracted from Wikipedia. Using WikiDIR, we demonstrate the weakness of lexical methods in dealing with high lexical variation in dialects. We further show that commonly used zero-shot cross-lingual transfer approach with multilingual encoders do not transfer well to extremely low-resource setups, motivating the need for resource-lean and dialect-specific retrieval models. We finally demonstrate that (document) translation is an effective way to reduce the dialect gap in CDIR. 

**Abstract (ZH)**: 大量的地方性和文化特定知识（例如，人物、传统、食物）仅能在方言写成的文档中找到。虽然已有很多关于跨语言信息检索 (CLIR) 的研究，但跨方言检索 (CDIR) 这一领域却得到了较少的关注。由于训练检索模型可用资源有限以及非正式语言的高度变异性，方言检索面临着独特的挑战。我们以德语方言为例进行了研究，并介绍了首个德语方言检索数据集 WikiDIR，该数据集包含从维基百科中提取的七种德语方言。利用 WikiDIR，我们展示了词汇方法在处理方言中高度词汇变异性方面的局限性。此外，我们还证明了使用多语言编码器的常见零样本跨语言迁移方法在极端资源稀缺的条件下效果不佳，从而突显了需要开发资源节省型和方言特定的检索模型的需求。最后，我们证明文档翻译是减少 CDIR 中方言差距的有效方法。 

---
# Is it the end of (generative) linguistics as we know it? 

**Title (ZH)**: 《正如我们所知的（生成性）语言学的终结了吗？》

这个标题翻译成中文后保持了original的反问风格和对当前语言学趋势的关注。在学术出版物中使用这种风格是可以接受的，能够引起读者的兴趣并引发思考。 

**Authors**: Cristiano Chesi  

**Link**: [PDF](https://arxiv.org/pdf/2412.12797)  

**Abstract**: A significant debate has emerged in response to a paper written by Steven Piantadosi (Piantadosi, 2023) and uploaded to the LingBuzz platform, the open archive for generative linguistics. Piantadosi's dismissal of Chomsky's approach is ruthless, but generative linguists deserve it. In this paper, I will adopt three idealized perspectives -- computational, theoretical, and experimental -- to focus on two fundamental issues that lend partial support to Piantadosi's critique: (a) the evidence challenging the Poverty of Stimulus (PoS) hypothesis and (b) the notion of simplicity as conceived within mainstream Minimalism. In conclusion, I argue that, to reclaim a central role in language studies, generative linguistics -- representing a prototypical theoretical perspective on language -- needs a serious update leading to (i) more precise, consistent, and complete formalizations of foundational intuitions and (ii) the establishment and utilization of a standardized dataset of crucial empirical evidence to evaluate the theory's adequacy. On the other hand, ignoring the formal perspective leads to major drawbacks in both computational and experimental approaches. Neither descriptive nor explanatory adequacy can be easily achieved without the precise formulation of general principles that can be challenged empirically. 

**Abstract (ZH)**: 针对 Steven Piantadosi（Piantadosi, 2023）在 LingBuzz 平台上发表的文章，一场重要的学术辩论正在展开，LingBuzz 是一个开放的生成语言学档案库。Piantadosi 对乔姆斯基方法的否定态度极为强烈，但生成语言学家们 deserve（应得）这样的批判。本文将从三个理想的视角——计算视角、理论视角和实验视角——出发，关注 Piantadosi 批评中的两个核心问题，即：（a）挑战刺激贫乏（Poverty of Stimulus, PoS）假设的证据，以及（b）主流最小主义中关于简洁性的概念。最终，我认为，为了在语言研究中重获核心地位，生成语言学——作为一种典型的理论语言视角——需要进行一次重大的更新，从而（i）更精确、一致和完整地形式化基本直觉，并且（ii）建立并利用一个标准化的关键实证证据集，以评估理论的充分性。另一方面，忽视形式视角会导致计算和实验方法中的重大缺陷。没有能够接受实证挑战的一般原则的精确表述，无法轻易实现描述性和解释性的充分性。 

---
# Revealing the impact of synthetic native samples and multi-tasking strategies in Hindi-English code-mixed humour and sarcasm detection 

**Title (ZH)**: 揭示合成原生样本和多任务学习策略对印欧双语代码混合幽默和讽刺检测的影响 

**Authors**: Debajyoti Mazumder, Aakash Kumar, Jasabanta Patro  

**Link**: [PDF](https://arxiv.org/pdf/2412.12761)  

**Abstract**: In this paper, we reported our experiments with various strategies to improve code-mixed humour and sarcasm detection. We did all of our experiments for Hindi-English code-mixed scenario, as we have the linguistic expertise for the same. We experimented with three approaches, namely (i) native sample mixing, (ii) multi-task learning (MTL), and (iii) prompting very large multilingual language models (VMLMs). In native sample mixing, we added monolingual task samples in code-mixed training sets. In MTL learning, we relied on native and code-mixed samples of a semantically related task (hate detection in our case). Finally, in our third approach, we evaluated the efficacy of VMLMs via few-shot context prompting. Some interesting findings we got are (i) adding native samples improved humor (raising the F1-score up to 6.76%) and sarcasm (raising the F1-score up to 8.64%) detection, (ii) training MLMs in an MTL framework boosted performance for both humour (raising the F1-score up to 10.67%) and sarcasm (increment up to 12.35% in F1-score) detection, and (iii) prompting VMLMs couldn't outperform the other approaches. Finally, our ablation studies and error analysis discovered the cases where our model is yet to improve. We provided our code for reproducibility. 

**Abstract (ZH)**: 在本文中，我们报告了各种策略以提高代码混合幽默和讽刺检测的实验结果。我们所有实验均基于印地语-英语代码混合场景进行，因为我们对该场景具有语言学专业知识。我们尝试了三种方法，分别是：（i）原语样本混合、（ii）多任务学习（MTL）和（iii）通过少量提示大型多语言语言模型（VMLMs）。在原语样本混合中，我们在代码混合训练集内添加了单语任务样本。在MTL学习中，我们依赖于同义任务的原语和代码混合样本（我们的案例是仇恨检测）。最后，在第三种方法中，我们通过少量提示VMLMs来评估其效果。我们获得了一些有趣的发现：（i）添加原语样本提高了幽默检测（F1分数提高6.76%）和讽刺检测（F1分数提高8.64%），（ii）在MTL框架下训练LMs提高了幽默（F1分数提高10.67%）和讽刺（F1分数提高12.35%）检测的性能，（iii）提示VMLMs无法超越其他方法。最后，我们的消融试验和错误分析揭示了模型尚未改进的实例。我们提供了代码以确保结果的可重现性。 

---
# Your Next State-of-the-Art Could Come from Another Domain: A Cross-Domain Analysis of Hierarchical Text Classification 

**Title (ZH)**: 您的下一个前瞻技术可能源自另一个领域：跨领域层次文本分类分析 

**Authors**: Nan Li, Bo Kang, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2412.12744)  

**Abstract**: Text classification with hierarchical labels is a prevalent and challenging task in natural language processing. Examples include assigning ICD codes to patient records, tagging patents into IPC classes, assigning EUROVOC descriptors to European legal texts, and more. Despite its widespread applications, a comprehensive understanding of state-of-the-art methods across different domains has been lacking. In this paper, we provide the first comprehensive cross-domain overview with empirical analysis of state-of-the-art methods. We propose a unified framework that positions each method within a common structure to facilitate research. Our empirical analysis yields key insights and guidelines, confirming the necessity of learning across different research areas to design effective methods. Notably, under our unified evaluation pipeline, we achieved new state-of-the-art results by applying techniques beyond their original domains. 

**Abstract (ZH)**: 层次化标签的文本分类是一项在自然语言处理领域中普遍且具有挑战性的任务。具体例子包括为患者记录分配ICD代码、将专利归类为IPC类、为欧洲法律文本分配EUROVOC描述词等。尽管该任务有着广泛的应用，但不同领域的先进方法的全面理解仍然缺乏。在本文中，我们提供了一个跨领域的首个全面综述，并通过实证分析来评估这些先进方法。我们提出了一种统一的框架，将每种方法置于共同结构中，以促进研究。我们的实证分析揭示了关键见解和指导方针，证实了跨不同研究领域学习的重要性，以设计有效的方法。值得注意的是，在我们统一的评估框架下，我们通过应用超出其原始领域的技术，达到了新的先进结果。 

---
# EventFull: Complete and Consistent Event Relation Annotation 

**Title (ZH)**: EventFull：完整且一致的事件关系标注 

**Authors**: Alon Eirew, Eviatar Nachshoni, Aviv Slobodkin, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2412.12733)  

**Abstract**: Event relation detection is a fundamental NLP task, leveraged in many downstream applications, whose modeling requires datasets annotated with event relations of various types. However, systematic and complete annotation of these relations is costly and challenging, due to the quadratic number of event pairs that need to be considered. Consequently, many current event relation datasets lack systematicity and completeness. In response, we introduce \textit{EventFull}, the first tool that supports consistent, complete and efficient annotation of temporal, causal and coreference relations via a unified and synergetic process. A pilot study demonstrates that EventFull accelerates and simplifies the annotation process while yielding high inter-annotator agreement. 

**Abstract (ZH)**: 事件关系检测是基础的自然语言处理(NLP)任务，广泛应用于许多下游应用中，其建模需要标注不同类型的事件关系的数据集。然而，系统且全面地标注这些关系的成本高且具有挑战性，因为需要考虑的事件对数量呈平方级增长。因此，当前许多事件关系数据集缺乏系统性和完整性。为应对这一问题，我们引入了\textit{EventFull}工具，这是第一个通过统一且协同的过程支持持续、全面和高效标注时间、因果和共指关系的工具。初步研究表明，EventFull可以加速并简化标注过程，同时提高标注者间的一致性。 

---
# SentiQNF: A Novel Approach to Sentiment Analysis Using Quantum Algorithms and Neuro-Fuzzy Systems 

**Title (ZH)**: SentiQNF：一种基于量子算法和神经模糊系统的新型情感分析方法 

**Authors**: Kshitij Dave, Nouhaila Innan, Bikash K. Behera, Zahid Mumtaz, Saif Al-Kuwari, Ahmed Farouk  

**Link**: [PDF](https://arxiv.org/pdf/2412.12731)  

**Abstract**: Sentiment analysis is an essential component of natural language processing, used to analyze sentiments, attitudes, and emotional tones in various contexts. It provides valuable insights into public opinion, customer feedback, and user experiences. Researchers have developed various classical machine learning and neuro-fuzzy approaches to address the exponential growth of data and the complexity of language structures in sentiment analysis. However, these approaches often fail to determine the optimal number of clusters, interpret results accurately, handle noise or outliers efficiently, and scale effectively to high-dimensional data. Additionally, they are frequently insensitive to input variations. In this paper, we propose a novel hybrid approach for sentiment analysis called the Quantum Fuzzy Neural Network (QFNN), which leverages quantum properties and incorporates a fuzzy layer to overcome the limitations of classical sentiment analysis algorithms. In this study, we test the proposed approach on two Twitter datasets: the Coronavirus Tweets Dataset (CVTD) and the General Sentimental Tweets Dataset (GSTD), and compare it with classical and hybrid algorithms. The results demonstrate that QFNN outperforms all classical, quantum, and hybrid algorithms, achieving 100% and 90% accuracy in the case of CVTD and GSTD, respectively. Furthermore, QFNN demonstrates its robustness against six different noise models, providing the potential to tackle the computational complexity associated with sentiment analysis on a large scale in a noisy environment. The proposed approach expedites sentiment data processing and precisely analyses different forms of textual data, thereby enhancing sentiment classification and insights associated with sentiment analysis. 

**Abstract (ZH)**: 情感分析是自然语言处理的一个关键组成部分，用于分析不同语境下的情感、态度和情绪色彩。它提供了关于公众意见、客户反馈和用户体验的宝贵洞察。研究人员开发了多种经典的机器学习和神经模糊方法，以应对情感分析中数据量的指数增长以及语言结构的复杂性。然而，这些方法往往无法确定最佳的聚类数量，准确解释结果，有效处理噪声或离群值，并且难以高效地扩展到高维数据。此外，它们通常对输入变化不够敏感。本文提出了一种名为量子模糊神经网络（QFNN）的新型混合方法，利用量子特性并嵌入模糊层以克服经典情感分析算法的局限性。本研究在两个Twitter数据集（冠状病毒推文数据集（CVTD）和普通情感推文数据集（GSTD）上测试了所提出的方法，并将其与经典和混合算法进行了比较。结果表明，QFNN在CVTD和GSTD数据集上的准确率分别达到了100%和90%，并且其在面对六种不同的噪声模型时表现出色，这为其在噪音环境中大规模处理情感分析的计算复杂性提供了潜在的解决方案。所提出的方法加快了情感数据分析速度，并精确分析了不同形式的文本数据，从而提高了情感分类和情感分析相关的洞察力。 

---
# Enhancing Naturalness in LLM-Generated Utterances through Disfluency Insertion 

**Title (ZH)**: 通过插入间歇性错误以增强LLM生成语句的自然度 

**Authors**: Syed Zohaib Hassan, Pierre Lison, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2412.12710)  

**Abstract**: Disfluencies are a natural feature of spontaneous human speech but are typically absent from the outputs of Large Language Models (LLMs). This absence can diminish the perceived naturalness of synthesized speech, which is an important criteria when building conversational agents that aim to mimick human behaviours. We show how the insertion of disfluencies can alleviate this shortcoming. The proposed approach involves (1) fine-tuning an LLM with Low-Rank Adaptation (LoRA) to incorporate various types of disfluencies into LLM-generated utterances and (2) synthesizing those utterances using a text-to-speech model that supports the generation of speech phenomena such as disfluencies. We evaluated the quality of the generated speech across two metrics: intelligibility and perceived spontaneity. We demonstrate through a user study that the insertion of disfluencies significantly increase the perceived spontaneity of the generated speech. This increase came, however, along with a slight reduction in intelligibility. 

**Abstract (ZH)**: 不流畅现象是自发人类口语的自然特征，但通常在大型语言模型（LLMs）的输出中是不存在的。这种缺失会降低合成语音的自然感，这是在构建模仿人类行为的对话代理时一个重要的评判标准。本文展示了如何通过插入不流畅现象来改善这一缺憾。所提出的方法包括（1）使用低秩适应（LoRA）对LLM进行微调，以使其生成的语句中包含多种类型的不流畅现象；以及（2）使用支持生成诸如不流畅现象等语音现象的文本到语音模型来合成这些语句。我们在两个指标上评估了生成语音的质量：可懂度和感知自然度。通过用户研究，我们证明了插入不流畅现象显著增加了生成语音的感知自然度。然而，这种增加是以略微降低可懂度为代价的。 

---
# More Tokens, Lower Precision: Towards the Optimal Token-Precision Trade-off in KV Cache Compression 

**Title (ZH)**: 更多令牌，更低精度：关于KV缓存压缩中令牌-精度权衡的最优解探究 

**Authors**: Jiebin Zhang, Dawei Zhu, Yifan Song, Wenhao Wu, Chuqiao Kuang, Xiaoguang Li, Lifeng Shang, Qun Liu, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.12706)  

**Abstract**: As large language models (LLMs) process increasing context windows, the memory usage of KV cache has become a critical bottleneck during inference. The mainstream KV compression methods, including KV pruning and KV quantization, primarily focus on either token or precision dimension and seldom explore the efficiency of their combination. In this paper, we comprehensively investigate the token-precision trade-off in KV cache compression. Experiments demonstrate that storing more tokens in the KV cache with lower precision, i.e., quantized pruning, can significantly enhance the long-context performance of LLMs. Furthermore, in-depth analysis regarding token-precision trade-off from a series of key aspects exhibit that, quantized pruning achieves substantial improvements in retrieval-related tasks and consistently performs well across varying input lengths. Moreover, quantized pruning demonstrates notable stability across different KV pruning methods, quantization strategies, and model scales. These findings provide valuable insights into the token-precision trade-off in KV cache compression. We plan to release our code in the near future. 

**Abstract (ZH)**: 随着大语言模型（LLMs）处理的上下文窗口增加，KV缓存的内存使用已成为推理过程中的关键瓶颈。当前主流的KV压缩方法，包括KV剪枝和KV量化，主要侧重于token维度或精度维度，很少探讨它们的结合效率。本文全面研究了KV缓存压缩中的token-precision权衡。实验表明，使用较低精度存储更多token，即量化剪枝，可以显著提升LLMs在长上下文下的性能。此外，从一系列关键方面深入分析token-precision权衡显示，量化剪枝在检索相关任务中取得了显著的进步，并且在不同输入长度下表现稳定。此外，量化剪枝在不同的KV剪枝方法、量化策略和模型规模下表现出明显的优势。这些发现为KV缓存压缩中的token-precision权衡提供了宝贵的见解。我们计划在未来不久发布我们的代码。 

---
# Trigger$^3$: Refining Query Correction via Adaptive Model Selector 

**Title (ZH)**: Trigger³：基于自适应模型选择的查询纠正 refinement 

**Authors**: Kepu Zhang, Zhongxiang Sun, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12701)  

**Abstract**: In search scenarios, user experience can be hindered by erroneous queries due to typos, voice errors, or knowledge gaps. Therefore, query correction is crucial for search engines. Current correction models, usually small models trained on specific data, often struggle with queries beyond their training scope or those requiring contextual understanding. While the advent of Large Language Models (LLMs) offers a potential solution, they are still limited by their pre-training data and inference cost, particularly for complex queries, making them not always effective for query correction. To tackle these, we propose Trigger$^3$, a large-small model collaboration framework that integrates the traditional correction model and LLM for query correction, capable of adaptively choosing the appropriate correction method based on the query and the correction results from the traditional correction model and LLM. Trigger$^3$ first employs a correction trigger to filter out correct queries. Incorrect queries are then corrected by the traditional correction model. If this fails, an LLM trigger is activated to call the LLM for correction. Finally, for queries that no model can correct, a fallback trigger decides to return the original query. Extensive experiments demonstrate Trigger$^3$ outperforms correction baselines while maintaining efficiency. 

**Abstract (ZH)**: 在搜索场景中，由于打字错误、语音错误或知识差距等原因，错误的查询可能阻碍用户的体验。因此，查询纠错对于搜索引擎至关重要。现有的纠错模型，通常是在特定数据上训练的小模型，往往难以处理超出其训练范围的查询，或者需要上下文理解的查询。虽然大型语言模型（LLMs）的出现提供了潜在的解决方案，但它们仍然受限于预训练数据和推理成本，特别是在处理复杂查询时，这使得它们并非总是有效的纠错工具。为了解决这些问题，我们提出了一种名为Trigger$^3$的大型与小型模型合作框架，该框架将传统的纠错模型与LLM集成用于查询纠错，可以根据查询和传统纠错模型及LLM的纠错结果，动态选择合适的纠错方法。Trigger$^3$首先使用纠错触发器过滤出正确的查询。对于错误的查询，先由传统的纠错模型进行修正。如果失败，则激活LLM触发器，调用LLM进行修正。最后，对于没有任何模型能够纠错的查询，回退触发器决定返回原始查询。广泛的实验表明，Trigger$^3$在保持效率的同时，其性能优于现有的纠错基线。 

---
# XTransplant: A Probe into the Upper Bound Performance of Multilingual Capability and Culture Adaptability in LLMs via Mutual Cross-lingual Feed-forward Transplantation 

**Title (ZH)**: X移植：通过互惠的跨语言前馈移植探究大型语言模型的多语言能力及文化适应性的上限性能 

**Authors**: Yangfan Ye, Xiaocheng Feng, Xiachong Feng, Libo Qin, Yichong Huang, Lei Huang, Weitao Ma, Zhirui Zhang, Yunfei Lu, Xiaohui Yan, Duyu Tang, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.12686)  

**Abstract**: Current large language models (LLMs) often exhibit imbalances in multilingual capabilities and cultural adaptability, largely due to their English-centric pretraining data. To address this imbalance, we propose a probing method named XTransplant that explores cross-lingual latent interactions via cross-lingual feed-forward transplantation during inference stage, with the hope of enabling the model to leverage the strengths of both English and non-English languages. Through extensive pilot experiments, we empirically prove that both the multilingual capabilities and cultural adaptability of LLMs hold the potential to be significantly improved by XTransplant, respectively from En -> non-En and non-En -> En, highlighting the underutilization of current LLMs' multilingual potential. And the patterns observed in these pilot experiments further motivate an offline scaling inference strategy, which demonstrates consistent performance improvements in multilingual and culture-aware tasks, sometimes even surpassing multilingual supervised fine-tuning. And we do hope our further analysis and discussion could help gain deeper insights into XTransplant mechanism. 

**Abstract (ZH)**: 当前的大语言模型（LLM）在多语言能力和文化适应性方面经常表现出不平衡，主要原因是它们的预训练数据以英语为中心。为了解决这一不平衡问题，我们提出了一种名为XTransplant的探究方法，该方法在推理阶段通过跨语言前向移植探索跨语言潜在交互，旨在使模型能够利用英语和非英语语言的优势。通过广泛的初步实验，我们实证证明，XTransplant在英语向非英语和非英语向英语两个方向上都有潜力显著提高LLM的多语言能力和文化适应性，这表明当前LLM的多语言潜力尚未充分开发。在这些初步实验中观察到的模式进一步促使一种离线缩放推理策略的采用，该策略在多语言和文化感知任务中展示了持续的性能提升，有时甚至超过了多语言监督微调。我们希望进一步的分析和讨论能够帮助我们更深入地理解XTransplant机制。 

---
# Detecting Document-level Paraphrased Machine Generated Content: Mimicking Human Writing Style and Involving Discourse Features 

**Title (ZH)**: 检测文档级别的人工生成的近义表达内容：模仿人类写作风格并涉及话语特征 

**Authors**: Yupei Li, Manuel Milling, Lucia Specia, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2412.12679)  

**Abstract**: The availability of high-quality APIs for Large Language Models (LLMs) has facilitated the widespread creation of Machine-Generated Content (MGC), posing challenges such as academic plagiarism and the spread of misinformation. Existing MGC detectors often focus solely on surface-level information, overlooking implicit and structural features. This makes them susceptible to deception by surface-level sentence patterns, particularly for longer texts and in texts that have been subsequently paraphrased.
To overcome these challenges, we introduce novel methodologies and datasets. Besides the publicly available dataset Plagbench, we developed the paraphrased Long-Form Question and Answer (paraLFQA) and paraphrased Writing Prompts (paraWP) datasets using GPT and DIPPER, a discourse paraphrasing tool, by extending artifacts from their original versions. To address the challenge of detecting highly similar paraphrased texts, we propose MhBART, an encoder-decoder model designed to emulate human writing style while incorporating a novel difference score mechanism. This model outperforms strong classifier baselines and identifies deceptive sentence patterns. To better capture the structure of longer texts at document level, we propose DTransformer, a model that integrates discourse analysis through PDTB preprocessing to encode structural features. It results in substantial performance gains across both datasets -- 15.5\% absolute improvement on paraLFQA, 4\% absolute improvement on paraWP, and 1.5\% absolute improvement on M4 compared to SOTA approaches. 

**Abstract (ZH)**: 高质量的大型语言模型（LLMs）API 的可用性促进了机器生成内容（MGC）的广泛创造，这带来了诸如学术剽窃和错误信息传播等挑战。现有的 MGC 检测器往往仅关注表面信息，忽视了隐含和结构性特征，这使得它们容易被表面句式模式欺骗，特别是在较长文本以及后续重述的文本中。

为应对这些挑战，我们提出了新的方法和数据集。除了公开可用的 Plagbench 数据集，我们还使用 GPT 和 DIPPER（一种话语重述工具）扩展其原始版本，开发了重述长形式问答（paraLFQA）和重述写作提示（paraWP）数据集。为了解决检测高度相似重述文本的挑战，我们提出了 MhBART，这是一种编码器-解码器模型，旨在模仿人类写作风格，并引入了一个新的差异得分机制。该模型在多个方面优于强大的分类器基线，并能够识别欺骗性句式模式。为了更好地捕捉较长文本文档级别的结构，我们提出了 DTransformer 模型，该模型通过 PDTB 预处理来集成话语分析，从而编码结构特征。该模型在两个数据集上均表现出显著的性能提升——paraLFQA 数据集上的绝对改进率为 15.5%，paraWP 数据集上的绝对改进率为 4%，与最新技术相比，M4 上的绝对改进率为 1.5%。 

---
# Train More Parameters But Mind Their Placement: Insights into Language Adaptation with PEFT 

**Title (ZH)**: 增加参数数量但注意其分布：关于PEFT在语言适应性中的见解 

**Authors**: Jenny Kunz  

**Link**: [PDF](https://arxiv.org/pdf/2412.12674)  

**Abstract**: Smaller LLMs still face significant challenges even in medium-resourced languages, particularly when it comes to language-specific knowledge -- a problem not easily resolved with machine-translated data. In this case study on Icelandic, we aim to enhance the generation performance of an LLM by specialising it using unstructured text corpora. A key focus is on preventing interference with the models' capabilities of handling longer context during this adaptation. Through ablation studies using various parameter-efficient fine-tuning (PEFT) methods and setups, we find that increasing the number of trainable parameters leads to better and more robust language adaptation. LoRAs placed in the feed-forward layers and bottleneck adapters show promising results with sufficient parameters, while prefix tuning and (IA)3 are not suitable. Although improvements are consistent in 0-shot summarisation, some adapted models struggle with longer context lengths, an issue that can be mitigated by adapting only the final layers. 

**Abstract (ZH)**: 即使在中资源语言中，较小的预训练语言模型（LLM）仍然面临显著挑战，尤其是在处理语言特定知识方面——这是一个仅通过机器翻译数据难以解决的问题。在此次关于冰岛语的研究案例中，我们旨在通过利用非结构化的文本语料库来专门化一个LLM，从而提升其生成性能。一个关键的关注点在于，在此次适应过程中防止干扰模型处理较长上下文的能力。通过使用各种参数高效的微调（PEFT）方法和方案进行消融研究，我们发现增加可训练参数的数量可以带来更有效的语言适应，并且具有足够参数的LoRA嵌入前馈层和瓶颈适配器表现出良好的效果。而前缀调优和（IA）3则不太适合。尽管在零样本总结中的一致改进，但一些适应后的模型在处理较长的上下文长度时仍存在问题，而这一问题可以通过只适应最后几层来缓解。 

---
# ClustEm4Ano: Clustering Text Embeddings of Nominal Textual Attributes for Microdata Anonymization 

**Title (ZH)**: ClustEm4Ano: 命名文本属性文本嵌入的聚类方法在微观数据分析中的匿名化应用 

**Authors**: Robert Aufschläger, Sebastian Wilhelm, Michael Heigl, Martin Schramm  

**Link**: [PDF](https://arxiv.org/pdf/2412.12649)  

**Abstract**: This work introduces ClustEm4Ano, an anonymization pipeline that can be used for generalization and suppression-based anonymization of nominal textual tabular data. It automatically generates value generalization hierarchies (VGHs) that, in turn, can be used to generalize attributes in quasi-identifiers. The pipeline leverages embeddings to generate semantically close value generalizations through iterative clustering. We applied KMeans and Hierarchical Agglomerative Clustering on $13$ different predefined text embeddings (both open and closed-source (via APIs)). Our approach is experimentally tested on a well-known benchmark dataset for anonymization: The UCI Machine Learning Repository's Adult dataset. ClustEm4Ano supports anonymization procedures by offering more possibilities compared to using arbitrarily chosen VGHs. Experiments demonstrate that these VGHs can outperform manually constructed ones in terms of downstream efficacy (especially for small $k$-anonymity ($2 \leq k \leq 30$)) and therefore can foster the quality of anonymized datasets. Our implementation is made public. 

**Abstract (ZH)**: 本文介绍了ClustEm4Ano，这是一个用于名义文本表数据的一般化和抑制匿名化处理的管道。该管道能够自动生成值概括层次结构（VGHs），进而用于泛化准标识符的属性。该管道利用嵌入技术通过迭代聚类生成语义上相近的值概括。我们使用KMeans和层次凝聚聚类算法在13种不同的预定义文本嵌入（包括开源和闭源嵌入，后者通过API获取）上进行了应用。我们的方法在脱敏标准数据集UCI机器学习库的大人数据集上进行了实验测试。ClustEm4Ano支持通过提供更多的可能性来进行匿名化处理，相较于随机选择的VGHs而言。实验表明，这些VGHs在下游有效性方面可以优于手工构建的VGHs（特别是在小$k$-匿名性情况下，$2 \leq k \leq 30$），因此可以促进脱敏数据集的质量。我们的实现已经公开。 

---
# iPrOp: Interactive Prompt Optimization for Large Language Models with a Human in the Loop 

**Title (ZH)**: iPrOp：具有人为干预的大语言模型交互式提示优化 

**Authors**: Jiahui Li, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.12644)  

**Abstract**: Prompt engineering has made significant contributions to the era of large language models, yet its effectiveness depends on the skills of a prompt author. Automatic prompt optimization can support the prompt development process, but requires annotated data. This paper introduces $\textit{iPrOp}$, a novel Interactive Prompt Optimization system, to bridge manual prompt engineering and automatic prompt optimization. With human intervention in the optimization loop, $\textit{iPrOp}$ offers users the flexibility to assess evolving prompts. We present users with prompt variations, selected instances, large language model predictions accompanied by corresponding explanations, and performance metrics derived from a subset of the training data. This approach empowers users to choose and further refine the provided prompts based on their individual preferences and needs. This system not only assists non-technical domain experts in generating optimal prompts tailored to their specific tasks or domains, but also enables to study the intrinsic parameters that influence the performance of prompt optimization. Our evaluation shows that our system has the capability to generate improved prompts, leading to enhanced task performance. 

**Abstract (ZH)**: 提示工程在大语言模型时代发挥了显著的贡献，但其有效性取决于提示撰写者的技能。自动提示优化可以支持提示开发过程，但需要标注数据。本文引入了$\textit{iPrOp}$，一种新颖的交互式提示优化系统，旨在将手工提示工程与自动提示优化相结合。通过在优化循环中的人工介入，$\textit{iPrOp}$为用户提供了一种评估不断演变提示的灵活性。我们向用户提供不同版本的提示、选定的实例、大型语言模型的预测及其相应的解释，以及从部分训练数据中得到的性能指标。通过这种方式，用户可以根据个人偏好和需求选择并进一步细化提供的提示。该系统不仅有助于非技术领域的专家生成针对特定任务或领域的最优提示，还可以研究影响提示优化性能的基本参数。我们的评估表明，该系统能够生成改进的提示，从而提高任务性能。 

---
# LLM-based Discriminative Reasoning for Knowledge Graph Question Answering 

**Title (ZH)**: 基于LLM的区分性推理在知识图谱问答中的应用 

**Authors**: Mufan Xu, Kehai Chen, Xuefeng Bai, Muyun Yang, Tiejun Zhao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12643)  

**Abstract**: Large language models (LLMs) based on generative pre-trained Transformer have achieved remarkable performance on knowledge graph question-answering (KGQA) tasks. However, LLMs often produce ungrounded subgraph planning or reasoning results in KGQA due to the hallucinatory behavior brought by the generative paradigm, which may hinder the advancement of the LLM-based KGQA model. To deal with the issue, we propose a novel LLM-based Discriminative Reasoning (LDR) method to explicitly model the subgraph retrieval and answer inference process. By adopting discriminative strategies, the proposed LDR method not only enhances the capability of LLMs to retrieve question-related subgraphs but also alleviates the issue of ungrounded reasoning brought by the generative paradigm of LLMs. Experimental results show that the proposed approach outperforms multiple strong comparison methods, along with achieving state-of-the-art performance on two widely used WebQSP and CWQ benchmarks. 

**Abstract (ZH)**: 基于生成预训练变换器的大型语言模型（LLMs）在知识图谱问答（KGQA）任务中取得了显著的性能。然而，由于生成范式带来的幻觉行为，LLMs 经常产生与知识图谱不相关的子图规划或推理结果，这可能阻碍基于LLM的KGQA模型的发展。为了解决这一问题，我们提出了一种新颖的基于LLM的辨别推理（LDR）方法，该方法明确建模子图检索和答案推理过程。通过采用辨别策略，所提出的LDR方法不仅增强了LLM检索与问题相关子图的能力，还缓解了LLM生成范式带来的无关推理问题。实验结果表明，所提出的方法在两个广泛使用的WebQSP和CWQ基准上优于多个强对照方法，并实现了最先进的性能。 

---
# Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting and Custom-Designed Decoding Tree 

**Title (ZH)**: Falcon：通过增强半自回归草稿生成和定制解码树实现的大语言模型更快并行推理 

**Authors**: Xiangxiang Gao, Weisheng Xie, Yiwei Xiang, Feng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.12639)  

**Abstract**: Striking an optimal balance between minimal drafting latency and high speculation accuracy to enhance the inference speed of Large Language Models remains a significant challenge in speculative decoding. In this paper, we introduce Falcon, an innovative semi-autoregressive speculative decoding framework fashioned to augment both the drafter's parallelism and output quality. Falcon incorporates the Coupled Sequential Glancing Distillation technique, which fortifies inter-token dependencies within the same block, leading to increased speculation accuracy. We offer a comprehensive theoretical analysis to illuminate the underlying mechanisms. Additionally, we introduce a Custom-Designed Decoding Tree, which permits the drafter to generate multiple tokens in a single forward pass and accommodates multiple forward passes as needed, thereby boosting the number of drafted tokens and significantly improving the overall acceptance rate. Comprehensive evaluations on benchmark datasets such as MT-Bench, HumanEval, and GSM8K demonstrate Falcon's superior acceleration capabilities. The framework achieves a lossless speedup ratio ranging from 2.91x to 3.51x when tested on the Vicuna and LLaMA2-Chat model series. These results outstrip existing speculative decoding methods for LLMs, including Eagle, Medusa, Lookahead, SPS, and PLD, while maintaining a compact drafter architecture equivalent to merely two Transformer layers. 

**Abstract (ZH)**: 在推断速度快的同时保持最小的编码延迟和高猜测准确性之间达到最优平衡，依然是推测解码中的一大挑战。本文提出了一种名为Falcon的创新半自回归推测解码框架，旨在增强编稿者的并行性和输出质量。Falcon采用了一种称为耦合序列凝视蒸馏的技术，增强同一块内各个词汇之间的依赖关系，从而提高了猜测的准确性。我们进行了全面的理论分析，以阐明其背后的机制。此外，我们还引入了一种自定义解码树，该树允许编稿者在单次前向传递中生成多个词汇，并在需要时支持多次前向传递，从而增加了被编写的词汇数量，并显著提高了整体接受率。在MT-Bench、HumanEval和GSM8K等基准数据集上的综合评估表明，Falcon具有卓越的加速能力。在测试的Vicuna和LLaMA2-Chat模型系列上，框架实现了无损加速比从2.91倍到3.51倍的效果。这些结果超越了现有的针对大规模语言模型（LLM）的推测解码方法，如Eagle、Medusa、Lookahead、SPS和PLD，同时保持一个紧凑的编稿者架构，相当于仅两个Transformer层。 

---
# What External Knowledge is Preferred by LLMs? Characterizing and Exploring Chain of Evidence in Imperfect Context 

**Title (ZH)**: LLMs倾向于偏好哪种外部知识？Characterizing and Exploring Chain of Evidence in Imperfect Context的学术译文如下：

LLMs偏好哪种外部知识？在不完美上下文中的推理链特征与探索 

**Authors**: Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang, Yuekai Huang, Qing Wang, Yihao Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12632)  

**Abstract**: Incorporating external knowledge into large language models (LLMs) has emerged as a promising approach to mitigate outdated knowledge and hallucination in LLMs. However, external knowledge is often imperfect. In addition to useful knowledge, external knowledge is rich in irrelevant or misinformation in the context that can impair the reliability of LLM responses. This paper focuses on LLMs' preferred external knowledge in imperfect contexts when handling multi-hop QA. Inspired by criminal procedural law's Chain of Evidence (CoE), we characterize that knowledge preferred by LLMs should maintain both relevance to the question and mutual support among knowledge pieces. Accordingly, we propose an automated CoE discrimination approach and explore LLMs' preferences from their effectiveness, faithfulness and robustness, as well as CoE's usability in a naive Retrieval-Augmented Generation (RAG) case. The evaluation on five LLMs reveals that CoE enhances LLMs through more accurate generation, stronger answer faithfulness, better robustness against knowledge conflict, and improved performance in a popular RAG case. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将外部知识纳入大型语言模型（LLMs）已成为缓解LLMs过时知识和幻觉问题的一个有前途的方法。然而，外部知识往往是不完美的。除了有用的知识外，外部知识在上下文中的无关或错误信息也很丰富，这会影响LLMs响应的可靠性。本文关注在处理多跳问答时LLMs对不完美上下文中的外部知识的偏好。借鉴刑事诉讼法中的证据链（Chain of Evidence, CoE），我们发现，LLMs偏好的外部知识应保持与问题的相关性以及知识片段之间的相互支持。据此，我们提出了一种自动化的CoE鉴别方法，并从有效性、忠实度和鲁棒性，以及在朴素检索增强生成（RAG）情景中的实用性角度探讨了LLMs的偏好。通过对五种LLMs的评估发现，证据链增强了LLMs，通过更准确的生成、更强的回答忠实度、更好的知识冲突鲁棒性以及在流行RAG情景中的更好表现。 

---
# Make Imagination Clearer! Stable Diffusion-based Visual Imagination for Multimodal Machine Translation 

**Title (ZH)**: 让想象更加清晰！基于稳定扩散的多模态机器翻译视觉想象 

**Authors**: Andong Chen, Yuchen Song, Kehai Chen, Muyun Yang, Tiejun Zhao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12627)  

**Abstract**: Visual information has been introduced for enhancing machine translation (MT), and its effectiveness heavily relies on the availability of large amounts of bilingual parallel sentence pairs with manual image annotations. In this paper, we introduce a stable diffusion-based imagination network into a multimodal large language model (MLLM) to explicitly generate an image for each source sentence, thereby advancing the multimodel MT. Particularly, we build heuristic human feedback with reinforcement learning to ensure the consistency of the generated image with the source sentence without the supervision of image annotation, which breaks the bottleneck of using visual information in MT. Furthermore, the proposed method enables imaginative visual information to be integrated into large-scale text-only MT in addition to multimodal MT. Experimental results show that our model significantly outperforms existing multimodal MT and text-only MT, especially achieving an average improvement of more than 14 BLEU points on Multi30K multimodal MT benchmarks. 

**Abstract (ZH)**: 视觉信息已经被引入到机器翻译（MT）中，其有效性很大程度上取决于能够获取大量带有手动图像标注的双语平行句子对。本文中，我们引入了一种基于稳定扩散机制的想象网络到多模态大型语言模型（Multimodal Large Language Model, MLLM）中，以显式地为每个源句子生成一张图像，从而推进了多模态MT的发展。特别地，我们通过强化学习构建了启发式的人工反馈机制，以确保生成的图像与源句子的一致性，而无需图像标注的监督，从而突破了在MT中使用视觉信息的瓶颈。此外，所提出的方法不仅能够在多模态MT中集成丰富的想象视觉信息，还在纯文本MT中也能使视觉信息发挥作用。实验结果表明，我们的模型显著优于现有的多模态MT和纯文本MT，尤其是在多30K多模态MT基准测试中，平均提升超过14个BLEU分数。 

---
# Jailbreaking? One Step Is Enough! 

**Title (ZH)**: 越狱？一步足矣！ 

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.12621)  

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中表现出色，但在对抗破解攻击方面仍显脆弱，即攻击者通过操纵提示生成有害输出。检查对抗破解提示有助于揭示LLMs的不足之处。然而，当前的对抗破解方法和目标模型的防御措施是在独立且对抗的过程中进行的，这导致了需要频繁的攻击迭代和针对不同模型重新设计攻击策略。为了弥补这些差距，我们提出了一种逆向嵌入防御攻击（REDA）机制，它将攻击意图伪装成“防御”意图。具体而言，REDA从目标响应开始，引导模型在其防御措施中嵌入有害内容，从而将有害内容置于次要地位，并使模型认为它正在进行防御任务。攻击模型认为它正在指导目标模型处理有害内容，而目标模型则认为它正在执行防御任务，从而在两者之间创造出一种合作的假象。此外，为了增强模型在“防御”意图上的置信度和引导作用，我们采用了少量攻击样本的上下文学习（ICL）方法，并相应地构建了一个攻击样本数据集。广泛的评估表明，REDA方法能够在不针对不同模型重新设计攻击策略的情况下实现跨模型攻击，在一次迭代中成功实现对抗破解，并且在开源和闭源模型上优于现有方法。 

---
# SynthCypher: A Fully Synthetic Data Generation Framework for Text-to-Cypher Querying in Knowledge Graphs 

**Title (ZH)**: SynthCypher：一种用于知识图中文本到密文查询的完全合成数据生成框架 

**Authors**: Aman Tiwari, Shiva Krishna Reddy Malay, Vikas Yadav, Masoud Hashemi, Sathwik Tejaswi Madhusudhan  

**Link**: [PDF](https://arxiv.org/pdf/2412.12612)  

**Abstract**: Cypher, the query language for Neo4j graph databases, plays a critical role in enabling graph-based analytics and data exploration. While substantial research has been dedicated to natural language to SQL query generation (Text2SQL), the analogous problem for graph databases referred to as Text2Cypher remains underexplored. In this work, we introduce SynthCypher, a fully synthetic and automated data generation pipeline designed to address this gap. SynthCypher employs a novel LLMSupervised Generation-Verification framework, ensuring syntactically and semantically correct Cypher queries across diverse domains and query complexities. Using this pipeline, we create SynthCypher Dataset, a large-scale benchmark containing 29.8k Text2Cypher instances. Fine-tuning open-source large language models (LLMs), including LLaMa-3.1- 8B, Mistral-7B, and QWEN-7B, on SynthCypher yields significant performance improvements of up to 40% on the Text2Cypher test set and 30% on the SPIDER benchmark adapted for graph databases. This work demonstrates that high-quality synthetic data can effectively advance the state-of-the-art in Text2Cypher tasks. 

**Abstract (ZH)**: Cypher 是 Neo4j 图数据库的查询语言，在基于图的分析和数据探索中起着关键作用。虽然大量研究致力于自然语言到 SQL 查询生成（Text2SQL）的问题，但针对图数据库的相应问题，即 Text2Cypher，仍然缺乏相关研究。在本文中，我们介绍了一种新的全合成和自动化的数据生成管道——SynthCypher，旨在解决这一问题。SynthCypher 使用了一种新颖的LLM 监督生成-验证框架，确保在不同领域和查询复杂度下生成语法和语义正确的 Cypher 查询。使用该管道，我们创建了包含 29,800 个 Text2Cypher 实例的 SynthCypher 数据集，这是一个大规模基准。通过对 SynthCypher 进行微调的开源大型语言模型（LLM），包括 LLaMa-3.1-8B、Mistral-7B 和 QWEN-7B，在 Text2Cypher 测试集上取得了高达 40% 的性能提升，在适应图数据库的 SPIDER 基准上则取得了 30% 的性能提升。本工作证明，高质量的合成数据可以有效推进 Text2Cypher 任务的发展。 

---
# MultiLingPoT: Enhancing Mathematical Reasoning with Multilingual Program Fine-tuning 

**Title (ZH)**: MultiLingPoT：通过多语言程序微调增强数学推理 

**Authors**: Nianqi Li, Zujie Liang, Siyu Yuan, Jiaqing Liang, Feng Wei, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.12609)  

**Abstract**: Program-of-Thought (PoT), which aims to use programming language instead of natural language as an intermediate step in reasoning, is an important way for LLMs to solve mathematical problems. Since different programming languages excel in different areas, it is natural to use the most suitable language for solving specific problems. However, current PoT research only focuses on single language PoT, ignoring the differences between different programming languages. Therefore, this paper proposes an multilingual program reasoning method, MultiLingPoT. This method allows the model to answer questions using multiple programming languages by fine-tuning on multilingual data. Additionally, prior and posterior hybrid methods are used to help the model select the most suitable language for each problem. Our experimental results show that the training of MultiLingPoT improves each program's mathematical reasoning by about 2.5\%. Moreover, with proper mixing, the performance of MultiLingPoT can be further improved, achieving a 6\% increase compared to the single-language PoT with the data this http URL of this paper can be found at this https URL. 

**Abstract (ZH)**: 程序思维（PoT）旨在使用编程语言而非自然语言作为推理的中间步骤，是大语言模型解决数学问题的重要方式。由于不同的编程语言在不同领域表现出色，因此使用最适合的语言来解决特定问题是自然的选择。然而，目前的PoT研究仅专注于单语言PoT，忽视了不同编程语言之间的差异。因此，本文提出了一种多语言程序推理方法——MultiLingPoT。该方法通过在多语言数据上进行微调，使模型能够使用多种编程语言来回答问题。此外，本研究采用了先验和后验混合方法，帮助模型为每个问题选择最合适的语言。实验结果表明，MultiLingPoT的训练可使每种程序的数学推理能力提高约2.5%。通过适当的混合使用，MultiLingPoT的性能还可以进一步提高，相较于使用单一语言PoT的数据集，其性能提高了约6%。本文的实验结果可访问该链接：[本文的链接] 

---
# LLMs are Also Effective Embedding Models: An In-depth Overview 

**Title (ZH)**: 大型语言模型也是有效的嵌入模型：深入综述 

**Authors**: Chongyang Tao, Tao Shen, Shen Gao, Junshuo Zhang, Zhen Li, Zhengwei Tao, Shuai Ma  

**Link**: [PDF](https://arxiv.org/pdf/2412.12591)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing by achieving state-of-the-art performance across various tasks. Recently, their effectiveness as embedding models has gained attention, marking a paradigm shift from traditional encoder-only models like ELMo and BERT to decoder-only, large-scale LLMs such as GPT, LLaMA, and Mistral. This survey provides an in-depth overview of this transition, beginning with foundational techniques before the LLM era, followed by LLM-based embedding models through two main strategies to derive embeddings from LLMs. 1) Direct prompting: We mainly discuss the prompt designs and the underlying rationale for deriving competitive embeddings. 2) Data-centric tuning: We cover extensive aspects that affect tuning an embedding model, including model architecture, training objectives, data constructions, etc. Upon the above, we also cover advanced methods, such as handling longer texts, and multilingual and cross-modal data. Furthermore, we discuss factors affecting choices of embedding models, such as performance/efficiency comparisons, dense vs sparse embeddings, pooling strategies, and scaling law. Lastly, the survey highlights the limitations and challenges in adapting LLMs for embeddings, including cross-task embedding quality, trade-offs between efficiency and accuracy, low-resource, long-context, data bias, robustness, etc. This survey serves as a valuable resource for researchers and practitioners by synthesizing current advancements, highlighting key challenges, and offering a comprehensive framework for future work aimed at enhancing the effectiveness and efficiency of LLMs as embedding models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过在各种任务中实现最先进的性能，已经革新了自然语言处理。最近，它们作为嵌入模型的有效性引起了广泛关注，标志着从传统的仅编码器模型（如ELMo和BERT）到依赖于解码器的大规模LLMs（如GPT、LLaMA和Mistral）的范式转变。本文综述了这一转变，从LLM时代之前的奠基技术开始，随后介绍了基于LLM的嵌入模型，通过两种主要策略从LLM中推导嵌入：1）直接提示：我们主要讨论提示设计及其推导竞争嵌入的内在逻辑。2）数据为中心的调优：我们涵盖了影响嵌入模型调优的广泛方面，包括模型架构、训练目标、数据构建等。在此基础上，我们还涵盖了高级方法，例如处理长文本、多语言和跨模态数据。此外，我们讨论了影响嵌入模型选择的因素，包括性能/效率比较、密集嵌入与稀疏嵌入、聚合策略以及扩展定律。最后，本文综述了将LLM适应为嵌入模型时的局限性和挑战，包括跨任务嵌入质量、效率与准确性的权衡、低资源、长上下文、数据偏差、鲁棒性等方面。本文综述为研究人员和从业者提供了一个宝贵资源，综合了当前的最新进展，突出了关键挑战，并提供了一个全面的框架，旨在未来的工作中增强LLM作为嵌入模型的有效性和效率。 

---
# PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization 

**Title (ZH)**: PerSphere：一种综合框架，用于多面向视角的检索与总结 

**Authors**: Yun Luo, Yingjie Li, Xiangkun Hu, Qinglin Qi, Fang Guo, Qipeng Guo, Zheng Zhang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12588)  

**Abstract**: As online platforms and recommendation algorithms evolve, people are increasingly trapped in echo chambers, leading to biased understandings of various issues. To combat this issue, we have introduced PerSphere, a benchmark designed to facilitate multi-faceted perspective retrieval and summarization, thus breaking free from these information silos. For each query within PerSphere, there are two opposing claims, each supported by distinct, non-overlapping perspectives drawn from one or more documents. Our goal is to accurately summarize these documents, aligning the summaries with the respective claims and their underlying perspectives. This task is structured as a two-step end-to-end pipeline that includes comprehensive document retrieval and multi-faceted summarization. Furthermore, we propose a set of metrics to evaluate the comprehensiveness of the retrieval and summarization content. Experimental results on various counterparts for the pipeline show that recent models struggle with such a complex task. Analysis shows that the main challenge lies in long context and perspective extraction, and we propose a simple but effective multi-agent summarization system, offering a promising solution to enhance performance on PerSphere. 

**Abstract (ZH)**: 随着在线平台和推荐算法的发展，人们正越来越多地陷入回声室中，导致对各种问题产生偏颇的理解。为了解决这一问题，我们引入了PerSphere基准，以促进多面向视角的检索和总结，从而打破这些信息孤岛的桎梏。对于PerSphere中的每个查询，存在两个对立的主张，每个主张都由一个或多个文档中不同、不重叠的视角支持。我们的目标是准确地总结这些文档，使总结与相应的主张及其背后的视角保持一致。这项任务采用了一个包括全面文档检索和多面向总结的端到端两步流程。此外，我们提出了一套评估检索和总结内容全面性的度量标准。在对流程各环节的各种对应对象进行的实验中发现，近期的模型难以处理如此复杂的任务。分析表明，主要挑战在于长上下文和视角提取，为此我们提出了一个简单但有效的多代理总结系统，为提高PerSphere性能提供了有希望的解决方案。 

---
# Process-Supervised Reward Models for Clinical Note Generation: A Scalable Approach Guided by Domain Expertise 

**Title (ZH)**: 基于过程监督的奖励模型在临床记录生成中的应用：一种由领域专业知识指导的可扩展方法 

**Authors**: Hanyin Wang, Qiping Xu, Bolun Liu, Guleid Hussein, Hariprasad Korsapati, Mohamad El Labban, Kingsley Iheasirim, Mohamed Hassan, Gokhan Anil, Brian Bartlett, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2412.12583)  

**Abstract**: Process-supervised reward models (PRMs), which verify large language model (LLM) outputs step-by-step, have achieved significant success in mathematical and coding problems. However, their application to other domains remains largely unexplored. In this work, we train a PRM to provide step-level reward signals for clinical notes generated by LLMs from patient-doctor dialogues. Guided by real-world clinician expertise, we carefully designed step definitions for clinical notes and utilized Gemini-Pro 1.5 to automatically generate process supervision data at scale. Our proposed PRM, trained on the LLaMA-3.1 8B instruct model, demonstrated superior performance compared to Gemini-Pro 1.5 and an outcome-supervised reward model (ORM) across two key evaluations: (1) the accuracy of selecting gold-reference samples from error-containing samples, achieving 98.8% (versus 61.3% for ORM and 93.8% for Gemini-Pro 1.5), and (2) the accuracy of selecting physician-preferred notes, achieving 56.2% (compared to 51.2% for ORM and 50.0% for Gemini-Pro 1.5). Additionally, we conducted ablation studies to determine optimal loss functions and data selection strategies, along with physician reader studies to explore predictors of downstream Best-of-N performance. Our promising results suggest the potential of PRMs to extend beyond the clinical domain, offering a scalable and effective solution for diverse generative tasks. 

**Abstract (ZH)**: 过程监督奖励模型（PRMs）通过逐步验证大型语言模型（LLMs）的输出，已经在数学和编程问题上取得了显著的成功。然而，其在其他领域的应用仍然很大程度上未被探索。在本项研究中，我们训练了一个PRM，以提供由患者-医生对话生成的临床笔记的逐步骤奖励信号。在现实世界临床专家的指导下，我们精心设计了临床笔记的步骤定义，并利用Gemini-Pro 1.5大规模自动生成过程监督数据。我们提出的PRM，基于LLaMA-3.1 8B指令模型进行训练，其在两大关键评估中均表现出色，与Gemini-Pro 1.5和结果监督奖励模型（ORM）相比：（1）在从包含错误的样本中选择黄金参考样本的准确性上，PRM达到了98.8%（而ORM为61.3%，Gemini-Pro 1.5为93.8%）；（2）在选择医生偏好笔记的准确性上，PRM达到了56.2%（而ORM为51.2%，Gemini-Pro 1.5为50.0%）。此外，我们进行了消融研究以确定最优损失函数和数据选择策略，并进行了医生读者研究以探索下游Best-of-N性能的预测因子。我们的研究结果表明，PRMs有可能扩展到临床领域以外，提供一种可扩展且有效的解决方案，适用于各种生成任务。 

---
# Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport 

**Title (ZH)**: 通过不平衡最优传输量化词汇语义转移 

**Authors**: Ryo Kishino, Hiroaki Yamagiwa, Ryo Nagata, Sho Yokoi, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2412.12569)  

**Abstract**: Lexical semantic change detection aims to identify shifts in word meanings over time. While existing methods using embeddings from a diachronic corpus pair estimate the degree of change for target words, they offer limited insight into changes at the level of individual usage instances. To address this, we apply Unbalanced Optimal Transport (UOT) to sets of contextualized word embeddings, capturing semantic change through the excess and deficit in the alignment between usage instances. In particular, we propose Sense Usage Shift (SUS), a measure that quantifies changes in the usage frequency of a word sense at each usage instance. By leveraging SUS, we demonstrate that several challenges in semantic change detection can be addressed in a unified manner, including quantifying instance-level semantic change and word-level tasks such as measuring the magnitude of semantic change and the broadening or narrowing of meaning. 

**Abstract (ZH)**: 词汇语义变化检测旨在识别词语随时间变化的语义转移。现有的方法利用来自历时语料配对的嵌入向量估计目标词语的变化程度，但这些方法对单个使用实例中的变化提供了有限的见解。为了解决这一问题，我们应用不平衡最优传输（Unbalanced Optimal Transport, UOT）来捕捉使用实例间语义变化，通过记录使用实例之间对齐的盈余和赤字来实现该目标。具体而言，我们提出了一种称为Sense Usage Shift（SUS）的度量方法，用于量化每个使用实例中词义用法频率的变化。通过利用SUS，我们证明了在语义变化检测中可以统一解决多个挑战，包括量化实例级别的语义变化，以及衡量语义变化的程度、意义的扩展或收缩等词语层级的任务。 

---
# FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning 

**Title (ZH)**: FCMR：金融跨模态多跳推理的稳健评估 

**Authors**: Seunghee Kim, Changhyeon Kim, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2412.12567)  

**Abstract**: Real-world decision-making often requires integrating and reasoning over information from multiple modalities. While recent multimodal large language models (MLLMs) have shown promise in such tasks, their ability to perform multi-hop reasoning across diverse sources remains insufficiently evaluated. Existing benchmarks, such as MMQA, face challenges due to (1) data contamination and (2) a lack of complex queries that necessitate operations across more than two modalities, hindering accurate performance assessment. To address this, we present Financial Cross-Modal Multi-Hop Reasoning (FCMR), a benchmark created to analyze the reasoning capabilities of MLLMs by urging them to combine information from textual reports, tables, and charts within the financial domain. FCMR is categorized into three difficulty levels-Easy, Medium, and Hard-facilitating a step-by-step evaluation. In particular, problems at the Hard level require precise cross-modal three-hop reasoning and are designed to prevent the disregard of any modality. Experiments on this new benchmark reveal that even state-of-the-art MLLMs struggle, with the best-performing model (Claude 3.5 Sonnet) achieving only 30.4% accuracy on the most challenging tier. We also conduct analysis to provide insights into the inner workings of the models, including the discovery of a critical bottleneck in the information retrieval phase. 

**Abstract (ZH)**: 现实世界中的决策往往需要整合和推理来自多种模态的信息。虽然近年来的多模态大型语言模型（MLLMs）在这些任务中展现了潜力，但它们在跨多种来源进行多跳推理的能力方面仍缺乏充分的评估。现有基准测试，如MMQA，面临着（1）数据污染和（2）缺乏跨三种以上模态操作的复杂查询等问题，从而阻碍了准确性能评估。为了解决这些问题，我们提出了金融跨模态多跳推理（FCMR）基准测试，旨在通过促使MLLMs综合财务领域内文本报告、表格和图表中的信息来分析它们的推理能力。FCMR划分为三个难度级别：简单、中等和困难，便于逐步评估。特别是，在困难级别的问题需要精确的三跳跨模态推理，并且设计上防止任何一种模态被忽视。在这一新基准测试上的实验表明，即使是最先进的MLLMs也难以应对，最佳模型（Claude 3.5 Sonnet）仅在最具有挑战性的级别上达到了30.4%的准确率。我们还进行了数据分析以提供对模型内部工作机制的洞察，包括发现信息检索阶段的关键瓶颈问题。 

---
# Evaluating Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models 

**Title (ZH)**: 使用大型语言模型评估零-shot 多语言方面基于情感分析 

**Authors**: Chengyan Wu, Bolei Ma, Zheyu Zhang, Ningyuan Deng, Yanqing He, Yun Xue  

**Link**: [PDF](https://arxiv.org/pdf/2412.12564)  

**Abstract**: Aspect-based sentiment analysis (ABSA), a sequence labeling task, has attracted increasing attention in multilingual contexts. While previous research has focused largely on fine-tuning or training models specifically for ABSA, we evaluate large language models (LLMs) under zero-shot conditions to explore their potential to tackle this challenge with minimal task-specific adaptation. We conduct a comprehensive empirical evaluation of a series of LLMs on multilingual ABSA tasks, investigating various prompting strategies, including vanilla zero-shot, chain-of-thought (CoT), self-improvement, self-debate, and self-consistency, across nine different models. Results indicate that while LLMs show promise in handling multilingual ABSA, they generally fall short of fine-tuned, task-specific models. Notably, simpler zero-shot prompts often outperform more complex strategies, especially in high-resource languages like English. These findings underscore the need for further refinement of LLM-based approaches to effectively address ABSA task across diverse languages. 

**Abstract (ZH)**: 基于aspect的 sentiment分析（ABSA），作为一种序列标注任务，在多语言背景下引起了越来越多的关注。尽管先前的研究主要集中在对ABSA进行微调或训练特定的模型上，但我们在此研究中在零样本条件下评估大型语言模型（LLMs），探讨它们在最少的任务特定适应下解决这一挑战的潜在能力。我们在九种不同模型上对一系列LLMs进行了全面的经验性评估，调查了包括纯零样本、思考链（CoT）、自我改进、自我辩论和自我一致性在内的各种提示策略。结果显示，虽然LLMs在处理多语言ABSA方面具有潜力，但它们通常无法与特定任务的微调模型相媲美。值得注意的是，在资源丰富语言（如英语）中，简单的零样本提示往往优于更复杂的方法。这些发现强调了进一步完善基于LLM的方法以有效解决跨多种语言的ABSA任务的重要性。 

---
# Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers 

**Title (ZH)**: 面向任务的语言模型水印嵌入方法：高熵传输层 

**Authors**: Vaden Masrani, Mohammad Akbari, David Ming Xuan Yue, Ahmad Rezaei, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12563)  

**Abstract**: In the era of costly pre-training of large language models, ensuring the intellectual property rights of model owners, and insuring that said models are responsibly deployed, is becoming increasingly important. To this end, we propose model watermarking via passthrough layers, which are added to existing pre-trained networks and trained using a self-supervised loss such that the model produces high-entropy output when prompted with a unique private key, and acts normally otherwise. Unlike existing model watermarking methods, our method is fully task-agnostic, and can be applied to both classification and sequence-to-sequence tasks without requiring advanced access to downstream fine-tuning datasets. We evaluate the proposed passthrough layers on a wide range of downstream tasks, and show experimentally our watermarking method achieves a near-perfect watermark extraction accuracy and false-positive rate in most cases without damaging original model performance. Additionally, we show our method is robust to both downstream fine-tuning, fine-pruning, and layer removal attacks, and can be trained in a fraction of the time required to train the original model. Code is available in the paper. 

**Abstract (ZH)**: 在大型语言模型昂贵的预训练时代，确保模型拥有者的知识产权，并确保这些模型得到负责任的应用，变得越来越重要。为此，我们提出了一种通过插入短路层（passthrough layers）进行模型水印的方法。这些短路层被添加到现有的预训练网络中，并使用自监督损失进行训练，使得在使用唯一的私钥提示时，模型产生高熵输出，而在其他情况下则正常工作。与现有的模型水印方法不同，我们的方法是完全任务无关的，无需对下游微调数据集有高级访问权限，即可应用于分类和序列-to-序列任务。我们对广泛的下游任务进行了评估，并在实验中证明，我们的水印方法在大多数情况下能达到接近完美的水印提取准确率和极低的误检率，同时不影响原始模型的表现。此外，我们展示了该方法对下游微调、精细剪枝和层移除攻击具有鲁棒性，并且可以在远少于训练原始模型所需时间的情况下进行训练。代码已包含在论文中。 

---
# EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation 

**Title (ZH)**: EXIT：基于上下文的抽取式压缩方法以增强检索增强生成 

**Authors**: Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, SeungYoon Han, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2412.12559)  

**Abstract**: We introduce EXIT, an extractive context compression framework that enhances both the effectiveness and efficiency of retrieval-augmented generation (RAG) in question answering (QA). Current RAG systems often struggle when retrieval models fail to rank the most relevant documents, leading to the inclusion of more context at the expense of latency and accuracy. While abstractive compression methods can drastically reduce token counts, their token-by-token generation process significantly increases end-to-end latency. Conversely, existing extractive methods reduce latency but rely on independent, non-adaptive sentence selection, failing to fully utilize contextual information. EXIT addresses these limitations by classifying sentences from retrieved documents - while preserving their contextual dependencies - enabling parallelizable, context-aware extraction that adapts to query complexity and retrieval quality. Our evaluations on both single-hop and multi-hop QA tasks show that EXIT consistently surpasses existing compression methods and even uncompressed baselines in QA accuracy, while also delivering substantial reductions in inference time and token count. By improving both effectiveness and efficiency, EXIT provides a promising direction for developing scalable, high-quality QA solutions in RAG pipelines. Our code is available at this https URL 

**Abstract (ZH)**: 我们介绍了一种名为 EXIT 的抽取式上下文压缩框架，该框架在问题回答（QA）中的检索增强生成（RAG）中提升了有效性和效率。当前的 RAG 系统在检索模型无法正确排序最相关的文档时，往往无法很好地应对，导致为了准确性而牺牲延迟，增加了不必要的上下文内容。尽管抽象压缩方法可以大幅减少词元数量，但它们逐词生成的过程显著增加了端到端的延迟。相比之下，现有的抽取式方法减少了延迟，但依靠独立且非自适应的句子选择，未能充分利用上下文信息。EXIT 通过分类检索文档中的句子（同时保持其上下文依赖性），实现并行、上下文意识的提取，能够适应查询复杂度和检索质量的变化。我们在单跳跃和多跳跃 QA 任务上的评估表明，EXIT 在 QA 准确性上始终超过了现有的压缩方法，甚至优于未压缩的基线方法，同时极大减少了推理时间和词元数量。通过在有效性和效率两方面的提升，EXIT 为 RAG 管道中开发可扩展且高质量的 QA 解决方案提供了有前途的方向。我们的代码可在此链接获取：[请插入具体的网址链接]。 

---
# LLMCL-GEC: Advancing Grammatical Error Correction with LLM-Driven Curriculum Learning 

**Title (ZH)**: LLMCL-GEC: 基于LLM驱动的课程学习的语法错误纠正方法进步 

**Authors**: Tao Fang, Derek F. Wong, Lusheng Zhang, Keyan Jin, Qiang Zhang, Tianjiao Li, Jinlong Hou, Lidia S. Chao  

**Link**: [PDF](https://arxiv.org/pdf/2412.12541)  

**Abstract**: While large-scale language models (LLMs) have demonstrated remarkable capabilities in specific natural language processing (NLP) tasks, they may still lack proficiency compared to specialized models in certain domains, such as grammatical error correction (GEC). Drawing inspiration from the concept of curriculum learning, we have delved into refining LLMs into proficient GEC experts by devising effective curriculum learning (CL) strategies. In this paper, we introduce a novel approach, termed LLM-based curriculum learning, which capitalizes on the robust semantic comprehension and discriminative prowess inherent in LLMs to gauge the complexity of GEC training data. Unlike traditional curriculum learning techniques, our method closely mirrors human expert-designed curriculums. Leveraging the proposed LLM-based CL method, we sequentially select varying levels of curriculums ranging from easy to hard, and iteratively train and refine using the pretrianed T5 and LLaMA series models. Through rigorous testing and analysis across diverse benchmark assessments in English GEC, including the CoNLL14 test, BEA19 test, and BEA19 development sets, our approach showcases a significant performance boost over baseline models and conventional curriculum learning methodologies. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在特定的自然语言处理（NLP）任务中展示了卓越的能力，但在某些领域，如语法错误纠正（GEC）方面，它们可能仍然不如专门化模型熟练。受到课程学习概念的启发，我们致力于通过有效的课程学习（CL）策略将LLMs培养成精通GEC的专家。在本文中，我们介绍了一种名为基于LLM的课程学习的新方法，该方法利用LLMs固有的稳健语义理解和区分能力来衡量GEC训练数据的复杂性。与传统的课程学习技术不同，我们的方法更接近于人类专家设计的课程。利用所提出的基于LLM的CL方法，我们依次选择从简单到复杂的不同难度级别的课程，并使用预训练的T5和LLaMA系列模型进行迭代训练和精细化改进。通过在英语GEC不同基准评估中的严格测试和分析，包括CoNLL14测试集、BEA19测试集和BEA19开发集，我们的方法在基础模型和传统课程学习方法上展示了显著的性能提升。 

---
# When to Speak, When to Abstain: Contrastive Decoding with Abstention 

**Title (ZH)**: 《该发言时发言，该回避时回避：对比解码与回避策略》 

**Authors**: Hyuhng Joon Kim, Youna Kim, Sang-goo Lee, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2412.12527)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks by leveraging both pre-trained knowledge (i.e., parametric knowledge) and external knowledge (i.e., contextual knowledge). While substantial efforts have been made to leverage both forms of knowledge, scenarios in which the model lacks any relevant knowledge remain underexplored. Such limitations can result in issues like hallucination, causing reduced reliability and potential risks in high-stakes applications. To address such limitations, this paper extends the task scope to encompass cases where the user's request cannot be fulfilled due to the lack of relevant knowledge. To this end, we introduce Contrastive Decoding with Abstention (CDA), a training-free decoding method that empowers LLMs to generate responses when relevant knowledge is available and to abstain otherwise. CDA evaluates the relevance of each knowledge for a given query, adaptively determining which knowledge to prioritize or which to completely ignore. Extensive experiments with four LLMs on three question-answering datasets demonstrate that CDA can effectively perform accurate generation and abstention simultaneously. These findings highlight CDA's potential to broaden the applicability of LLMs, enhancing reliability and preserving user trust. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过利用预先训练的知识（即参数化知识）和外部知识（即上下文知识）在多种任务中表现出色。虽然已经在利用这两种形式的知识方面做了大量工作，但当模型缺乏相关知识时的场景仍被广泛忽视。这种局限可能导致幻觉问题，从而在高风险应用中降低可靠性和潜在风险。为解决这些问题，本文将任务范围扩展到用户请求无法得到满足的情况，即由于缺乏相关知识而导致请求无法实现的情形。为此，我们引入了一种无需训练的解码方法——对比解码与回避（CDA），该方法使LLMs能够在有相关知识时生成响应，并在没有相关知识时回避。CDA评估每个知识与给定查询的相关性，并自适应地决定优先使用哪种知识或完全忽略哪种知识。在三个问答数据集上对四种LLMs的广泛实验表明，CDA能够同时进行准确的生成和回避。这些发现突显了CDA在扩展LLMs的应用范围、提高可靠性和保护用户信任方面的潜力。 

---
# Solid-SQL: Enhanced Schema-linking based In-context Learning for Robust Text-to-SQL 

**Title (ZH)**: Solid-SQL：基于增强模式链接的上下文适应性学习方法以提高文本到SQL的鲁棒性 

**Authors**: Geling Liu, Yunzhi Tan, Ruichao Zhong, Yuanzhen Xie, Lingchen Zhao, Qian Wang, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.12522)  

**Abstract**: Recently, large language models (LLMs) have significantly improved the performance of text-to-SQL systems. Nevertheless, many state-of-the-art (SOTA) approaches have overlooked the critical aspect of system robustness. Our experiments reveal that while LLM-driven methods excel on standard datasets, their accuracy is notably compromised when faced with adversarial perturbations. To address this challenge, we propose a robust text-to-SQL solution, called Solid-SQL, designed to integrate with various LLMs. We focus on the pre-processing stage, training a robust schema-linking model enhanced by LLM-based data augmentation. Additionally, we design a two-round, structural similarity-based example retrieval strategy for in-context learning. Our method achieves SOTA SQL execution accuracy levels of 82.1% and 58.9% on the general Spider and Bird benchmarks, respectively. Furthermore, experimental results show that Solid-SQL delivers an average improvement of 11.6% compared to baselines on the perturbed Spider-Syn, Spider-Realistic, and Dr. Spider benchmarks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）显著提高了文本到SQL系统的性能。然而，许多最先进的（SOTA）方法在系统鲁棒性这一关键方面有所忽视。我们的实验发现，虽然LLM驱动的方法在标准数据集上的表现优异，但在面临对抗性扰动时，其准确性明显下降。为应对这一挑战，我们提出了一种鲁棒的文本到SQL解决方案，称为Solid-SQL，并设计为能够与各种LLM集成。我们重点关注预处理阶段，使用基于LLM的数据增强训练鲁棒的模式链接模型。此外，我们设计了一种两轮、基于结构相似性的示例检索策略，以支持上下文学习。我们的方法在通用的Spider和Bird基准测试中分别实现了82.1%和58.9%的SOTA SQL执行准确性。进一步的实验结果表明，与基线相比，Solid-SQL在Spider-Syn、Spider-Realistic和Dr. Spider基准测试中表现出平均11.6%的提升。 

---
# Can Large Language Models Understand You Better? An MBTI Personality Detection Dataset Aligned with Population Traits 

**Title (ZH)**: 大型语言模型能否更好地理解你？一个与人口特征对齐的MBTI人格检测数据集 

**Authors**: Bohan Li, Jiannan Guan, Longxu Dou, Yunlong Feng, Dingzirui Wang, Yang Xu, Enbo Wang, Qiguang Chen, Bichen Wang, Xiao Xu, Yimeng Zhang, Libo Qin, Yanyan Zhao, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2412.12510)  

**Abstract**: The Myers-Briggs Type Indicator (MBTI) is one of the most influential personality theories reflecting individual differences in thinking, feeling, and behaving. MBTI personality detection has garnered considerable research interest and has evolved significantly over the years. However, this task tends to be overly optimistic, as it currently does not align well with the natural distribution of population personality traits. Specifically, (1) the self-reported labels in existing datasets result in incorrect labeling issues, and (2) the hard labels fail to capture the full range of population personality distributions. In this paper, we optimize the task by constructing MBTIBench, the first manually annotated high-quality MBTI personality detection dataset with soft labels, under the guidance of psychologists. As for the first challenge, MBTIBench effectively solves the incorrect labeling issues, which account for 29.58% of the data. As for the second challenge, we estimate soft labels by deriving the polarity tendency of samples. The obtained soft labels confirm that there are more people with non-extreme personality traits. Experimental results not only highlight the polarized predictions and biases in LLMs as key directions for future research, but also confirm that soft labels can provide more benefits to other psychological tasks than hard labels. The code and data are available at this https URL. 

**Abstract (ZH)**: 迈尔斯-布里格斯类型指标（MBTI）是反映个体在思考、感受和行为方面差异的一种最具影响力的个性理论之一。近年来，MBTI个性检测吸引了大量的研究兴趣，并且已经取得了显著的进步。然而，这一任务往往过于乐观，因为目前它并不符合人口个性特征的自然分布。具体来说，（1）现有数据集中的自我报告标签导致了错误标记的问题，（2）硬标签未能捕捉到人口个性分布的全部范围。在这篇论文中，我们在心理学家的指导下，通过构建首个包含软标签的手动标注高质量MBTI个性检测数据集MBTIBench，优化了这一任务。针对第一个挑战，MBTIBench有效地解决了占数据29.58%的错误标注问题。针对第二个挑战，我们通过分析样本的极性倾向来估计软标签。获得的软标签证实了非极端个性特征的人更多。实验结果不仅突出了LLM（大规模语言模型）预测的极化和偏差是未来研究的关键方向，而且还证实了软标签相比硬标签能在其他的心理任务中提供更多的益处。相关代码和数据可在以下链接获取：[该链接]。 

---
# Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge 

**Title (ZH)**: 你能否信任大语言模型的判断？大语言模型作为法官的可靠性探究 

**Authors**: Kayla Schroeder, Zach Wood-Doughty  

**Link**: [PDF](https://arxiv.org/pdf/2412.12509)  

**Abstract**: Large Language Models (LLMs) have become increasingly powerful and ubiquitous, but their stochastic nature poses challenges to the reliability of their outputs. While deterministic settings can improve consistency, they do not guarantee reliability, as a single sample from the model's probability distribution can still be misleading. Building upon the concept of LLM-as-a-judge, we introduce a novel framework for rigorously evaluating the reliability of LLM judgments, leveraging McDonald's omega. We evaluate the reliability of LLMs when judging the outputs of other LLMs on standard single-turn and multi-turn benchmarks, simultaneously investigating the impact of temperature on reliability. By analyzing these results, we demonstrate the limitations of fixed randomness and the importance of considering multiple samples, which we show has significant implications for downstream applications. Our findings highlight the need for a nuanced understanding of LLM reliability and the potential risks associated with over-reliance on single-shot evaluations. This work provides a crucial step towards building more trustworthy and reliable LLM-based systems and applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越强大且普及，但它们的随机性给输出的可靠性带来了挑战。虽然确定性设置可以提高一致性，但无法确保可靠性，因为模型概率分布中的单一样本仍然可能误导性。基于LLM作为裁判的概念，我们提出了一种新的框架来严格评估LLM判决的可靠性，并利用麦当劳的ω系数进行评估。我们对LLM在评估其他LLM输出的标准单轮和多轮基准上的可靠性进行了评估，同时探讨了温度对可靠性的潜在影响。通过对这些结果的分析，我们展示了固定随机性限制和考虑多个样本的重要性，这些发现对于下游应用程序具有重要影响。我们的研究结果突显了对LLM可靠性需要精微理解的必要性以及过度依赖单次评估可能带来的潜在风险。这项工作为构建更值得信赖的LLM基础系统和应用程序提供了关键步骤。 

---
# DocFusion: A Unified Framework for Document Parsing Tasks 

**Title (ZH)**: DocFusion：文档解析任务的统一框架 

**Authors**: Mingxu Chai, Ziyu Shen, Chong Zhang, Yue Zhang, Xiao Wang, Shihan Dou, Jihua Kang, Jiazheng Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12505)  

**Abstract**: Document parsing is essential for analyzing complex document structures and extracting fine-grained information, supporting numerous downstream applications. However, existing methods often require integrating multiple independent models to handle various parsing tasks, leading to high complexity and maintenance overhead. To address this, we propose DocFusion, a lightweight generative model with only 0.28B parameters. It unifies task representations and achieves collaborative training through an improved objective function. Experiments reveal and leverage the mutually beneficial interaction among recognition tasks, and integrating recognition data significantly enhances detection performance. The final results demonstrate that DocFusion achieves state-of-the-art (SOTA) performance across four key tasks. 

**Abstract (ZH)**: 文档解析对于分析复杂的文档结构和提取细腻的信息至关重要，支持众多下游应用。然而，现有方法往往需要集成多个独立的模型来处理各种解析任务，导致系统复杂度和维护成本较高。为了解决这个问题，我们提出了DocFusion，这是一种轻量级生成模型，参数量仅为0.28亿。该模型统一了任务表示并通过改进的目标函数实现了协同训练。实验表明，DocFusion能够揭示并利用识别任务间的相互促进的交互作用，将识别数据集成显著提高了检测性能。最终结果表明，DocFusion在四个关键任务上达到了最先进的（SOTA）性能。 

---
# Beyond Data Quantity: Key Factors Driving Performance in Multilingual Language Models 

**Title (ZH)**: 超越数据量：多语言语言模型性能驱动的关键因素 

**Authors**: Sina Bagheri Nezhad, Ameeta Agrawal, Rhitabrat Pokharel  

**Link**: [PDF](https://arxiv.org/pdf/2412.12500)  

**Abstract**: Multilingual language models (MLLMs) are crucial for handling text across various languages, yet they often show performance disparities due to differences in resource availability and linguistic characteristics. While the impact of pre-train data percentage and model size on performance is well-known, our study reveals additional critical factors that significantly influence MLLM effectiveness. Analyzing a wide range of features, including geographical, linguistic, and resource-related aspects, we focus on the SIB-200 dataset for classification and the Flores-200 dataset for machine translation, using regression models and SHAP values across 204 languages. Our findings identify token similarity and country similarity as pivotal factors, alongside pre-train data and model size, in enhancing model performance. Token similarity facilitates cross-lingual transfer, while country similarity highlights the importance of shared cultural and linguistic contexts. These insights offer valuable guidance for developing more equitable and effective multilingual language models, particularly for underrepresented languages. 

**Abstract (ZH)**: 多语言语言模型（MLLMs）在处理多种语言的文本方面至关重要，但由于资源可用性和语言特性存在差异，它们常常显示出性能差异。虽然预训练数据量和模型大小对性能的影响已为人所知，我们的研究揭示了另外一些关键因素，这些因素显著影响MLLM的有效性。通过分析包括地理、语言和资源相关在内的广泛特征，我们使用回归模型和SHAP值在204种语言上对SIB-200数据集进行分类和Flores-200数据集进行机器翻译进行了研究。研究发现，词汇相似性和国家相似性是增强模型性能的关键因素，与预训练数据和模型大小并列。词汇相似性有助于跨语言迁移，而国家相似性则突显了共享文化与语言背景的重要性。这些见解为开发更加公平和有效的多语言语言模型提供了宝贵的指导，特别对于代表性不足的语言尤为重要。 

---
# LinguaLIFT: An Effective Two-stage Instruction Tuning Framework for Low-Resource Language Tasks 

**Title (ZH)**: LinguaLIFT：一种有效的两阶段指令调优框架，用于低资源语言任务 

**Authors**: Hongbin Zhang, Kehai Chen, Xuefeng Bai, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12499)  

**Abstract**: Large language models (LLMs) have demonstrated impressive multilingual understanding and reasoning capabilities, driven by extensive pre-training multilingual corpora and fine-tuning instruction data. However, a performance gap persists between high-resource and low-resource language tasks due to language imbalance in the pre-training corpus, even using more low-resource data during fine-tuning. To alleviate this issue, we propose LinguaLIFT, a two-stage instruction tuning framework for advancing low-resource language tasks. An additional language alignment layer is first integrated into the LLM to adapt a pre-trained multilingual encoder, thereby enhancing multilingual alignment through code-switched fine-tuning. The second stage fine-tunes LLM with English-only instruction data while freezing the language alignment layer, allowing LLM to transfer task-specific capabilities from English to low-resource language tasks. Additionally, we introduce the Multilingual Math World Problem (MMWP) benchmark, which spans 21 low-resource, 17 medium-resource, and 10 high-resource languages, enabling comprehensive evaluation of multilingual reasoning. Experimental results show that LinguaLIFT outperforms several competitive baselines across MMWP and other widely used benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经在多语言理解和推理方面展示了令人印象深刻的性能，这得益于大规模预训练多语言语料库和细调指令数据。然而，即使在使用更多低资源数据进行细调的情况下，高资源语言任务和低资源语言任务之间仍存在性能差距，这主要是由于预训练语料库中语言资源的不平衡。为了解决这一问题，我们提出了一种两阶段指令调优框架LinguaLIFT，来促进低资源语言任务的发展。首先，将在LLM中集成一个额外的语言对齐层，以适应预训练的多语言编码器，从而通过代码转换的细调增强多语言对齐。第二阶段使用仅英语的指令数据对LLM进行细调，同时冻结语言对齐层，使LLM能够将特定任务能力从英语转移至低资源语言任务。此外，我们还引入了多语言数学世界问题（MMWP）基准测试，涵盖了21种低资源、17种中资源和10种高资源语言，从而可以全面评估多语言推理能力。实验结果表明，LinguaLIFT在MMWP以及其他广泛使用的基准测试中均优于几个有竞争力的基线。 

---
# NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning 

**Title (ZH)**: NLSR：针对有害微调的大语言模型神经元级安全性重对齐 

**Authors**: Xin Yi, Shunfan Zheng, Linlin Wang, Gerard de Melo, Xiaoling Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2412.12497)  

**Abstract**: The emergence of finetuning-as-a-service has revealed a new vulnerability in large language models (LLMs). A mere handful of malicious data uploaded by users can subtly manipulate the finetuning process, resulting in an alignment-broken model. Existing methods to counteract fine-tuning attacks typically require substantial computational resources. Even with parameter-efficient techniques like LoRA, gradient updates remain essential. To address these challenges, we propose \textbf{N}euron-\textbf{L}evel \textbf{S}afety \textbf{R}ealignment (\textbf{NLSR}), a training-free framework that restores the safety of LLMs based on the similarity difference of safety-critical neurons before and after fine-tuning. The core of our framework is first to construct a safety reference model from an initially aligned model to amplify safety-related features in neurons. We then utilize this reference model to identify safety-critical neurons, which we prepare as patches. Finally, we selectively restore only those neurons that exhibit significant similarity differences by transplanting these prepared patches, thereby minimally altering the fine-tuned model. Extensive experiments demonstrate significant safety enhancements in fine-tuned models across multiple downstream tasks, while greatly maintaining task-level accuracy. Our findings suggest regions of some safety-critical neurons show noticeable differences after fine-tuning, which can be effectively corrected by transplanting neurons from the reference model without requiring additional training. The code will be available at \url{this https URL} 

**Abstract (ZH)**: 以下是对该论文内容或标题的学术规范翻译：

深度调优即服务平台的出现揭示了大规模语言模型（LLMs）中的一种新漏洞。用户上传的一小部分恶意数据可以微妙地影响调优过程，导致模型的对齐失效。现有的对抗调优攻击方法通常需要大量的计算资源。即使是参数高效的技术如LoRA，梯度更新仍然是必不可少的。为了解决这些挑战，我们提出了一种训练免费的框架，名为**N**euron-\textbf{L}evel \textbf{S}afety \textbf{R}ealignment (\textbf{NLSR})，该框架基于调优前后关键神经元的相似性差异来恢复LLMs的安全性。我们框架的核心是首先从初始对齐的模型构建一个安全参考模型，以增强与安全性相关的特征。随后，利用该参考模型识别关键安全性神经元，将其作为补丁预先准备。最后，仅通过移植这些准备好的补丁恢复那些表现出显著相似性差异的神经元，从而最小限度地改变调优后的模型。广泛的实验证明，该框架在多个下游任务中显著提高了调优模型的安全性，同时在任务级别准确度上保持了较大的一致性。我们的研究发现，一些关键安全性神经元在调优后表现出明显的差异，这些差异可以通过移植参考模型中的神经元补丁得以有效纠正，而无需额外的训练。代码将在 [此处提供链接] 供公众查阅。 

---
# Boosting Long-Context Information Seeking via Query-Guided Activation Refilling 

**Title (ZH)**: 通过查询引导的激活补充增强长上下文信息检索 

**Authors**: Hongjin Qian, Zheng Liu, Peitian Zhang, Zhicheng Dou, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2412.12486)  

**Abstract**: Processing long contexts poses a significant challenge for large language models (LLMs) due to their inherent context-window limitations and the computational burden of extensive key-value (KV) activations, which severely impact efficiency. For information-seeking tasks, full context perception is often unnecessary, as a query's information needs can dynamically range from localized details to a global perspective, depending on its complexity. However, existing methods struggle to adapt effectively to these dynamic information needs.
In the paper, we propose a method for processing long-context information-seeking tasks via query-guided Activation Refilling (ACRE). ACRE constructs a Bi-layer KV Cache for long contexts, where the layer-1 (L1) cache compactly captures global information, and the layer-2 (L2) cache provides detailed and localized information. ACRE establishes a proxying relationship between the two caches, allowing the input query to attend to the L1 cache and dynamically refill it with relevant entries from the L2 cache. This mechanism integrates global understanding with query-specific local details, thus improving answer decoding. Experiments on a variety of long-context information-seeking datasets demonstrate ACRE's effectiveness, achieving improvements in both performance and efficiency. 

**Abstract (ZH)**: 处理长上下文对大型语言模型（LLMs）构成了显著挑战，主要原因在于它们固有的上下文窗口限制和大量关键值（KV）激活带来的计算负担，这严重影响了模型的效率。对于信息检索任务而言，全面感知上下文往往是不必要的，因为查询的信息需求可以动态地从局部细节扩展到全局视角，具体取决于查询的复杂性。然而，现有方法难以有效地适应这些动态的信息需求。

在本文中，我们提出了一种通过查询引导的激活补充（Activation Refilling, ACRE）方法来处理长上下文信息检索任务。ACRE为长上下文构建了一个双层KV缓存，其中层级1（L1）缓存紧凑地捕获全局信息，层级2（L2）缓存提供详细的局部信息。ACRE在两个缓存之间建立了代理关系，使得输入查询可以关注L1缓存，并通过L2缓存动态补充相关条目，从而实现全局理解和查询特定的局部细节的融合。这种机制能够提高答案解码的效率。在不同类型的长上下文信息检索数据集上的实验表明，ACRE具有良好的效果，同时在性能和效率方面均有所提升。 

---
# Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script 

**Title (ZH)**: 人工介入生成对抗性文本：藏文案例研究 

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima  

**Link**: [PDF](https://arxiv.org/pdf/2412.12478)  

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages. 

**Abstract (ZH)**: 基于DNN的语言模型在各类任务中表现出色，但即使是当前最先进的大型语言模型（LLM）也容易受到文本对抗攻击的影响。文本对抗样本在自然语言处理（NLP）的多个子领域中发挥着重要作用。然而，当前的研究存在以下问题：（1）大多数文本对抗攻击方法主要针对资源丰富的语言。我们如何为较少研究的语言生成对抗文本？（2）大多数文本对抗攻击方法容易生成无效或模棱两可的对抗文本。我们如何构建高质量的对抗鲁棒性基准？（3）新的语言模型可能对部分先前生成的对抗文本具有免疫力。我们如何更新对抗鲁棒性基准？为了解决上述问题，我们提出了一种基于人工在环生成对抗文本的通用方法的系统——HITL-GAT。HITL-GAT包含一个管道中的四个阶段：受害模型构建、对抗样本生成、高质量基准构建和对抗鲁棒性评估。此外，我们利用HITL-GAT对藏文进行了案例研究，这可以为其他较少研究的语言的对抗性研究提供参考。 

---
# RareAgents: Autonomous Multi-disciplinary Team for Rare Disease Diagnosis and Treatment 

**Title (ZH)**: RareAgents：自主多学科团队在罕见病诊断与治疗中的应用 

**Authors**: Xuanzhong Chen, Ye Jin, Xiaohao Mao, Lun Wang, Shuyang Zhang, Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.12475)  

**Abstract**: Rare diseases, despite their low individual incidence, collectively impact around 300 million people worldwide due to the huge number of diseases. The complexity of symptoms and the shortage of specialized doctors with relevant experience make diagnosing and treating rare diseases more challenging than common diseases. Recently, agents powered by large language models (LLMs) have demonstrated notable improvements across various domains. In the medical field, some agent methods have outperformed direct prompts in question-answering tasks from medical exams. However, current agent frameworks lack adaptation for real-world clinical scenarios, especially those involving the intricate demands of rare diseases. To address these challenges, we present RareAgents, the first multi-disciplinary team of LLM-based agents tailored to the complex clinical context of rare diseases. RareAgents integrates advanced planning capabilities, memory mechanisms, and medical tools utilization, leveraging Llama-3.1-8B/70B as the base model. Experimental results show that RareAgents surpasses state-of-the-art domain-specific models, GPT-4o, and existing agent frameworks in both differential diagnosis and medication recommendation for rare diseases. Furthermore, we contribute a novel dataset, MIMIC-IV-Ext-Rare, derived from MIMIC-IV, to support further advancements in this field. 

**Abstract (ZH)**: 稀有疾病尽管单个疾病的发病率较低，但由于疾病种类繁多，全球约有3亿人受到这些疾病的集体影响。症状复杂性和缺乏相关经验的专业医生使得诊断和治疗稀有疾病比常见疾病更加具有挑战性。最近，由大规模语言模型（LLMs）驱动的代理已经在多个领域展示出了显著的进步。在医疗领域，一些代理方法在医学考试中的问答任务中已经超越了直接提示。然而，当前的代理框架缺乏针对真实临床场景的适应性，特别是那些涉及稀有疾病复杂需求的场景。为了解决这些挑战，我们提出了RareAgents，这是第一个针对稀有疾病复杂临床环境的多学科团队，基于LLM的代理。RareAgents集成了先进的规划能力、记忆机制和医疗工具的应用，以LLama-3.1-8B/70B作为基础模型。实验结果表明，RareAgents在稀有疾病的鉴别诊断和药物推荐方面优于现有的领域特定模型GPT-4o和现有的代理框架。此外，我们贡献了一个新的数据集MIMIC-IV-Ext-Rare，该数据集源自MIMIC-IV，以支持该领域进一步的发展。 

---
# Knowledge Boundary of Large Language Models: A Survey 

**Title (ZH)**: 大型语言模型的知识边界：一个综述 

**Authors**: Moxin Li, Yong Zhao, Yang Deng, Wenxuan Zhang, Shuaiyi Li, Wenya Xie, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2412.12472)  

**Abstract**: Although large language models (LLMs) store vast amount of knowledge in their parameters, they still have limitations in the memorization and utilization of certain knowledge, leading to undesired behaviors such as generating untruthful and inaccurate responses. This highlights the critical need to understand the knowledge boundary of LLMs, a concept that remains inadequately defined in existing research. In this survey, we propose a comprehensive definition of the LLM knowledge boundary and introduce a formalized taxonomy categorizing knowledge into four distinct types. Using this foundation, we systematically review the field through three key lenses: the motivation for studying LLM knowledge boundaries, methods for identifying these boundaries, and strategies for mitigating the challenges they present. Finally, we discuss open challenges and potential research directions in this area. We aim for this survey to offer the community a comprehensive overview, facilitate access to key issues, and inspire further advancements in LLM knowledge research. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在其参数中存储了大量知识，但在记忆和利用某些知识方面仍存在局限性，导致生成不真实和不准确的回复等不良行为。这突显了了解LLM知识边界的迫切需求，而现有研究中这一概念的定义尚不完善。在本文综述中，我们提出了一种全面的LLM知识边界定义，并介绍了将知识分类为四种不同类型的正式化分类体系。基于这一基础，我们从三个关键视角系统地回顾了该领域：研究LLM知识边界的原因、识别这些边界的 方法，以及应对这些边界带来的挑战的策略。最后，我们讨论了该领域的开放挑战和潜在研究方向。我们希望本文综述能为社区提供一个全面的概述，促进关键问题的访问，并激发LLM知识研究的进一步进步。 

---
# Core Context Aware Attention for Long Context Language Modeling 

**Title (ZH)**: 长上下文语言建模中的核心上下文感知注意力机制 

**Authors**: Yaofo Chen, Zeng You, Shuhai Zhang, Haokun Li, Yirui Li, Yaowei Wang, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2412.12465)  

**Abstract**: Transformer-based Large Language Models (LLMs) have exhibited remarkable success in various natural language processing tasks primarily attributed to self-attention mechanism, which requires a token to consider all preceding tokens as its context to compute the attention score. However, when the context length L becomes very large (e.g., 32K), more redundant context information will be included w.r.t. any tokens, making the self-attention suffer from two main limitations: 1) The computational and memory complexity scales quadratically w.r.t. L; 2) The presence of redundant context information may hamper the model to capture dependencies among crucial tokens, which may degrade the representation performance. In this paper, we propose a plug-and-play Core Context Aware (CCA) Attention for efficient long-range context modeling, which consists of two components: 1) Globality-pooling attention that divides input tokens into groups and then dynamically merges tokens within each group into one core token based on their significance; 2) Locality-preserved attention that incorporates neighboring tokens into the attention calculation. The two complementary attentions will then be fused to the final attention, maintaining comprehensive modeling ability as the full self-attention. In this way, the core context information w.r.t. a given token will be automatically focused and strengthened, while the context information in redundant groups will be diminished during the learning process. As a result, the computational and memory complexity will be significantly reduced. More importantly, the CCA-Attention can improve the long-context modeling ability by diminishing the redundant context information. Extensive experimental results demonstrate that our CCA-Attention significantly outperforms state-of-the-art models in terms of computational efficiency and long-context modeling ability. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）在各类自然语言处理任务中表现出显著的成功，这主要归功于其自我注意力机制，该机制要求每个标记考虑所有先前的标记作为其上下文来计算注意力分数。然而，当上下文长度L变得非常大（例如，32K）时，对于任何标记而言，将包括更多的冗余上下文信息，这使得自我注意力面临两大主要限制：1）计算和内存复杂度与L呈二次关系增加；2）冗余上下文信息的出现可能妨碍模型捕捉关键标记之间的联系，从而降低表示性能。为了解决这些问题，本文提出了一种可插拔的核心上下文感知（CCA）注意力机制，用于高效建模长距离上下文。该机制由两个组件组成：1）全局聚集注意力，将输入标记分为组，然后基于它们的重要性动态合并每个组内的标记为一个核心标记；2）局部保留注意力，将相邻标记纳入注意力计算。这两种互补的注意力机制随后将融合到最终的注意力机制中，保持与完整自我注意力相同的广泛建模能力。通过这种方式，给定标记的核心上下文信息将在学习过程中自动被关注并加强，而冗余组中的上下文信息则会被减弱。结果，计算和内存复杂度显著降低。更重要的是，CCA注意力机制能够通过减少冗余上下文信息来提高长距离上下文建模能力。广泛实验结果表明，与最先进的模型相比，我们的CCA注意力机制在计算效率和长距离上下文建模能力方面表现出显著的优势。 

---
# LITA: An Efficient LLM-assisted Iterative Topic Augmentation Framework 

**Title (ZH)**: LITA：一种高效的LLM辅助迭代主题增强框架 

**Authors**: Chia-Hsuan Chang, Jui-Tse Tsai, Yi-Hang Tsai, San-Yih Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12459)  

**Abstract**: Topic modeling is widely used for uncovering thematic structures within text corpora, yet traditional models often struggle with specificity and coherence in domain-focused applications. Guided approaches, such as SeededLDA and CorEx, incorporate user-provided seed words to improve relevance but remain labor-intensive and static. Large language models (LLMs) offer potential for dynamic topic refinement and discovery, yet their application often incurs high API costs. To address these challenges, we propose the LLM-assisted Iterative Topic Augmentation framework (LITA), an LLM-assisted approach that integrates user-provided seeds with embedding-based clustering and iterative refinement. LITA identifies a small number of ambiguous documents and employs an LLM to reassign them to existing or new topics, minimizing API costs while enhancing topic quality. Experiments on two datasets across topic quality and clustering performance metrics demonstrate that LITA outperforms five baseline models, including LDA, SeededLDA, CorEx, BERTopic, and PromptTopic. Our work offers an efficient and adaptable framework for advancing topic modeling and text clustering. 

**Abstract (ZH)**: 主题建模广泛用于揭示文本语料库中的主题结构，但传统模型在领域导向的应用中经常面临具体性和连贯性方面的困难。指导型方法，如SeededLDA和CorEx，通过引入用户提供的种子词来提高相关性，但依然劳动密集且静态。大型语言模型（LLMs）为动态主题细化和发现提供了潜在可能性，但其应用往往会产生较高的API成本。为解决这些挑战，我们提出了LLM辅助迭代主题增强框架（LITA），这是一种通过集成用户提供的种子词、基于嵌入的聚类和迭代细化的LLM辅助方法。LITA识别少量模糊文档，并利用LLM重新指派它们到现有或新主题，以最小化API成本并提高主题质量。在两个数据集上，通过对主题质量及聚类性能指标进行实验，结果显示LITA优于包括LDA、SeededLDA、CorEx、BERTopic和PromptTopic在内的五种基准模型。我们的工作提供了一个高效且适应性强的框架，用于推进主题建模和文本聚类。 

---
# Persona-SQ: A Personalized Suggested Question Generation Framework For Real-world Documents 

**Title (ZH)**: Persona-SQ：一种面向真实世界文档的个性化建议问题生成框架 

**Authors**: Zihao Lin, Zichao Wang, Yuanting Pan, Varun Manjunatha, Ryan Rossi, Angela Lau, Lifu Huang, Tong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2412.12445)  

**Abstract**: Suggested questions (SQs) provide an effective initial interface for users to engage with their documents in AI-powered reading applications. In practical reading sessions, users have diverse backgrounds and reading goals, yet current SQ features typically ignore such user information, resulting in homogeneous or ineffective questions. We introduce a pipeline that generates personalized SQs by incorporating reader profiles (professions and reading goals) and demonstrate its utility in two ways: 1) as an improved SQ generation pipeline that produces higher quality and more diverse questions compared to current baselines, and 2) as a data generator to fine-tune extremely small models that perform competitively with much larger models on SQ generation. Our approach can not only serve as a drop-in replacement in current SQ systems to immediately improve their performance but also help develop on-device SQ models that can run locally to deliver fast and private SQ experience. 

**Abstract (ZH)**: 建议问题（SQs）为用户提供了一个有效的初始界面，以在基于人工智能的阅读应用中与他们的文档交互。在实际的阅读会话中，用户的背景和阅读目标各不相同，但当前的SQ功能通常忽视这些用户信息，导致生成的提问模式化或效果不佳。我们提出了一种生成个性化SQ的管道，该管道结合了读者档案（职业和阅读目标），并通过两种方式证明其效用：1）作为一种改进的SQ生成管道，能够生成比当前基准更高质量和更多样化的问题；2）作为一种数据生成器，用于微调极其小型的模型，在SQ生成方面能够与更大规模的模型竞争。我们的方法不仅可以作为现有SQ系统的即插即用替代方案，立即提升其性能，还能帮助开发可以在设备端运行的SQ模型，以提供快速和私有的SQ体验。 

---
# Refining Dimensions for Improving Clustering-based Cross-lingual Topic Models 

**Title (ZH)**: 改进基于聚类的跨语言主题模型的维度细化方法 

**Authors**: Chia-Hsuan Chang, Tien-Yuan Huang, Yi-Hang Tsai, Chia-Ming Chang, San-Yih Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12433)  

**Abstract**: Recent works in clustering-based topic models perform well in monolingual topic identification by introducing a pipeline to cluster the contextualized representations. However, the pipeline is suboptimal in identifying topics across languages due to the presence of language-dependent dimensions (LDDs) generated by multilingual language models. To address this issue, we introduce a novel, SVD-based dimension refinement component into the pipeline of the clustering-based topic model. This component effectively neutralizes the negative impact of LDDs, enabling the model to accurately identify topics across languages. Our experiments on three datasets demonstrate that the updated pipeline with the dimension refinement component generally outperforms other state-of-the-art cross-lingual topic models. 

**Abstract (ZH)**: 基于聚类的主题模型的近期研究在单语言主题识别中表现出色，通过引入管道来聚类上下文表示。然而，该管道在跨语言主题识别中效果不佳，这主要是因为多语言语言模型生成的语言依赖维度（LDDs）对聚类效果产生了负面影响。为解决这一问题，我们提出在基于聚类的主题模型管道中引入一种基于SVD的维度精炼组件。该组件有效抵消了LDDs的负面影响，使模型能够准确地跨语言识别主题。我们在三个数据集上的实验表明，含有维度精炼组件的更新管道通常优于其他最先进的跨语言主题模型。 

---
# Assessing the Limitations of Large Language Models in Clinical Fact Decomposition 

**Title (ZH)**: 评估大型语言模型在临床事实分解方面的局限性 

**Authors**: Monica Munnangi, Akshay Swaminathan, Jason Alan Fries, Jenelle Jindal, Sanjana Narayanan, Ivan Lopez, Lucia Tu, Philip Chung, Jesutofunmi A. Omiye, Mehr Kashyap, Nigam Shah  

**Link**: [PDF](https://arxiv.org/pdf/2412.12422)  

**Abstract**: Verifying factual claims is critical for using large language models (LLMs) in healthcare. Recent work has proposed fact decomposition, which uses LLMs to rewrite source text into concise sentences conveying a single piece of information, as an approach for fine-grained fact verification. Clinical documentation poses unique challenges for fact decomposition due to dense terminology and diverse note types. To explore these challenges, we present FactEHR, a dataset consisting of full document fact decompositions for 2,168 clinical notes spanning four types from three hospital systems. Our evaluation, including review by clinicians, highlights significant variability in the quality of fact decomposition for four commonly used LLMs, with some LLMs generating 2.6x more facts per sentence than others. The results underscore the need for better LLM capabilities to support factual verification in clinical text. To facilitate future research in this direction, we plan to release our code at \url{this https URL}. 

**Abstract (ZH)**: 在医疗保健中使用大规模语言模型（LLMs）时，验证事实性声明至关重要。近期有研究表明，通过使用LLMs将源文本重写为简洁的句子，传达单一信息的方法——即事实分解，可以作为一种细粒度的事实验证手段。临床文档由于术语密集且笔记类型多样，为事实分解带来了独特的挑战。为了探索这些挑战，我们提出了一个名为FactEHR的数据集，该数据集包含来自三个医疗机构的2,168份临床笔记的完整文档分解，涵盖了四种类型的笔记。我们的评估包括临床医生的评审，突显了四种常用LLMs在事实分解质量上存在显著差异，有些LLMs生成的信息量是其他LLMs的2.6倍。这些结果强调了支持临床文本事实验证所需更好LLMs的能力。为了促进该领域未来的研究，我们计划在 \url{this https URL} 发布我们的代码。 

---
# Bridging the Gap: Enhancing LLM Performance for Low-Resource African Languages with New Benchmarks, Fine-Tuning, and Cultural Adjustments 

**Title (ZH)**: 填补空白：通过新的基准测试、微调和文化调整，提升低资源非洲语言的大型语言模型性能 

**Authors**: Tuka Alhanai, Adam Kasumovic, Mohammad Ghassemi, Aven Zitzelberger, Jessica Lundin, Guillaume Chabot-Couture  

**Link**: [PDF](https://arxiv.org/pdf/2412.12417)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance across various tasks, yet significant disparities remain for non-English languages, and especially native African languages. This paper addresses these disparities by creating approximately 1 million human-translated words of new benchmark data in 8 low-resource African languages, covering a population of over 160 million speakers of: Amharic, Bambara, Igbo, Sepedi (Northern Sotho), Shona, Sesotho (Southern Sotho), Setswana, and Tsonga. Our benchmarks are translations of Winogrande and three sections of MMLU: college medicine, clinical knowledge, and virology. Using the translated benchmarks, we report previously unknown performance gaps between state-of-the-art (SOTA) LLMs in English and African languages. Finally, using results from over 400 fine-tuned models, we explore several methods to reduce the LLM performance gap, including high-quality dataset fine-tuning (using an LLM-as-an-Annotator), cross-lingual transfer, and cultural appropriateness adjustments. Key findings include average mono-lingual improvements of 5.6% with fine-tuning (with 5.4% average mono-lingual improvements when using high-quality data over low-quality data), 2.9% average gains from cross-lingual transfer, and a 3.0% out-of-the-box performance boost on culturally appropriate questions. The publicly available benchmarks, translations, and code from this study support further research and development aimed at creating more inclusive and effective language technologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但在非英语语言，尤其是本土非洲语言方面仍然存在显著差距。本文通过在8种低资源非洲语言（阿姆哈拉语、班巴拉语、伊博语、塞佩迪语（北部索托语）、悚纳语、塞萨索托语（南部索托语）、;setswana语和特桑加语）上创建约100万词的人工翻译基准数据，来解决这些差距问题。这些语言共有超过1.6亿说这些语言的人口，基准数据包括Winogrande和MMLU中的三个部分：医学院校、临床知识和病毒学。利用翻译后的基准数据，我们报告了最新最佳表现（SOTA）LLMs在英语和非洲语言之间的未知性能差距。最后，通过超过400个微调模型的结果，我们探讨了减少LLM性能差距的几种方法，包括高质量数据集微调（使用LLM作为注释器）、跨语言迁移和文化适宜性调整。主要发现包括单语微调平均提高了5.6%的成绩（使用高质量数据相比低质量数据时的平均单语微调提高了5.4%），平均2.9%的跨语言迁移收益，以及在文化适宜性问题上3.0%的原生性能提升。本研究公开提供的基准数据、翻译内容和代码支持进一步的研究和发展，旨在创造更为包容且有效的语言技术。 

---
# Interpretable LLM-based Table Question Answering 

**Title (ZH)**: 可解释的基于大规模语言模型的表格问答方法 

**Authors**: Giang, Nguyen, Ivan Brugere, Shubham Sharma, Sanjay Kariyappa, Anh Totti Nguyen, Freddy Lecue  

**Link**: [PDF](https://arxiv.org/pdf/2412.12386)  

**Abstract**: Interpretability for Table Question Answering (Table QA) is critical, particularly in high-stakes industries like finance or healthcare. Although recent approaches using Large Language Models (LLMs) have significantly improved Table QA performance, their explanations for how the answers are generated are ambiguous. To fill this gap, we introduce Plan-of-SQLs ( or POS), an interpretable, effective, and efficient approach to Table QA that answers an input query solely with SQL executions. Through qualitative and quantitative evaluations with human and LLM judges, we show that POS is most preferred among explanation methods, helps human users understand model decision boundaries, and facilitates model success and error identification. Furthermore, when evaluated in standard benchmarks (TabFact, WikiTQ, and FetaQA), POS achieves competitive or superior accuracy compared to existing methods, while maintaining greater efficiency by requiring significantly fewer LLM calls and database queries. 

**Abstract (ZH)**: 表格问答（Table QA）的可解释性对于高风险行业（如金融或医疗保健）尤为重要。尽管近期使用大规模语言模型（LLMs）的方法在提高Table QA性能方面取得了显著进展，但它们对于如何生成答案的解释仍然模糊不清。为了解决这一问题，我们提出了一种名为Plan-of-SQLs（或POS）的方法，该方法通过SQL执行来独立回答输入查询，从而提高Table QA的可解释性、有效性和效率。通过使用人类和LLM评判者的定性和定量评估，我们展示了POS方法在解释方法中得到了最高认可，有助于人类用户理解模型的决策边界，并促进了模型成功和错误识别。此外，在标准基准（TabFact、WikiTQ和FetaQA）上评估时，POS在准确性和效率方面均与现有方法相当或更优，同时仅需显著减少LLM调用和数据库查询次数。 

---
# BioRAGent: A Retrieval-Augmented Generation System for Showcasing Generative Query Expansion and Domain-Specific Search for Scientific Q&A 

**Title (ZH)**: BioRAGent：一种用于展示生成型查询扩展和领域特定搜索的检索增强生成系统 

**Authors**: Samy Ateia, Udo Kruschwitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.12358)  

**Abstract**: We present BioRAGent, an interactive web-based retrieval-augmented generation (RAG) system for biomedical question answering. The system uses large language models (LLMs) for query expansion, snippet extraction, and answer generation while maintaining transparency through citation links to the source documents and displaying generated queries for further editing. Building on our successful participation in the BioASQ 2024 challenge, we demonstrate how few-shot learning with LLMs can be effectively applied for a professional search setting. The system supports both direct short paragraph style responses and responses with inline citations. Our demo is available online, and the source code is publicly accessible through GitHub. 

**Abstract (ZH)**: 我们介绍了BioRAGent，这是一个基于网页的交互式检索增强生成（RAG）系统，用于生物医学问题解答。该系统使用大型语言模型（LLMs）进行查询扩展、片段提取和答案生成，同时通过引用链接展示源文档，并显示生成的查询以供进一步编辑，从而保持透明性。在我们成功参与BioASQ 2024 挑战的基础上，我们展示了在专业搜索环境中如何有效应用少样本学习。该系统支持直接简短段落式的回答和带有 inline 引用的参考文献回答。我们的演示版本已在线提供，源代码也通过 GitHub 公开 accessible。 

---
# Graph-Guided Textual Explanation Generation Framework 

**Title (ZH)**: 图引导的文本解释生成框架 

**Authors**: Shuzhou Yuan, Jingyi Sun, Ran Zhang, Michael Färber, Steffen Eger, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.12318)  

**Abstract**: Natural language explanations (NLEs) are commonly used to provide plausible free-text explanations of a model's reasoning about its predictions. However, recent work has questioned the faithfulness of NLEs, as they may not accurately reflect the model's internal reasoning process regarding its predicted answer. In contrast, highlight explanations -- input fragments identified as critical for the model's predictions -- exhibit measurable faithfulness, which has been incrementally improved through existing research. Building on this foundation, we propose G-Tex, a Graph-Guided Textual Explanation Generation framework designed to enhance the faithfulness of NLEs by leveraging highlight explanations. Specifically, highlight explanations are extracted as highly faithful cues representing the model's reasoning and are subsequently encoded through a graph neural network layer, which explicitly guides the NLE generation process. This alignment ensures that the generated explanations closely reflect the model's underlying reasoning. Experiments on T5 and BART using three reasoning datasets show that G-Tex improves NLE faithfulness by up to 17.59% compared to baseline methods. Additionally, G-Tex generates NLEs with greater semantic and lexical similarity to human-written ones. Human evaluations show that G-Tex can decrease redundant content and enhance the overall quality of NLEs. As our work introduces a novel method for explicitly guiding NLE generation to improve faithfulness, we hope it will serve as a stepping stone for addressing additional criteria for NLE and generated text overall. 

**Abstract (ZH)**: 自然语言解释（NLEs）通常用于提供模型对其预测推理的合理文本解释。然而，近期的研究对NLEs的忠实度提出了质疑，因为它们可能无法准确反映模型内部推理过程及其预测的答案。相比之下，高亮解释——模型预测中被识别为关键的输入片段——表现出可测量的忠实度，并通过现有研究已经逐步得到了提升。在此基础上，我们提出了一种名为G-Tex的图引导文本解释生成框架，旨在通过利用高亮解释来增强NLEs的忠实度。具体来说，从模型推理中提取出高忠实度的高亮解释作为线索，并通过图神经网络层进行编码，明确指导NLE生成过程。这种对齐确保生成的解释能够密切反映模型的内在推理。在T5和BART上使用三个推理数据集进行的实验表明，与基准方法相比，G-Tex在NLE忠实度上最高可提升17.59%。此外，G-Tex生成的NLEs在语义和词汇上与人类撰写的解释更为相似。人类评估表明，G-Tex可以减少冗余内容，提升NLEs的整体质量。作为一项工作，我们的方法引入了一种明确引导NLE生成以提高忠实度的新方法，我们希望它能为解决NLE和生成文本的其他标准提供一个基础。 

---
# Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion 

**Title (ZH)**: 通过逐步词汇扩展促进大型语言模型（LLM）学习阿拉伯语二外习得 

**Authors**: Jianqing Zhu, Huang Huang, Zhihang Lin, Juhao Liang, Zhengyang Tang, Khalid Almubarak, Abdulmohsen Alharthik, Bang An, Juncai He, Xiangbo Wu, Fei Yu, Junying Chen, Zhuoheng Ma, Yuhao Du, He Zhang, Emad A. Alghamdi, Lian Zhang, Ruoyu Sun, Haizhou Li, Benyou Wang, Jinchao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12310)  

**Abstract**: This paper addresses the critical need for democratizing large language models (LLM) in the Arab world, a region that has seen slower progress in developing models comparable to state-of-the-art offerings like GPT-4 or ChatGPT 3.5, due to a predominant focus on mainstream languages (e.g., English and Chinese). One practical objective for an Arabic LLM is to utilize an Arabic-specific vocabulary for the tokenizer that could speed up decoding. However, using a different vocabulary often leads to a degradation of learned knowledge since many words are initially out-of-vocabulary (OOV) when training starts. Inspired by the vocabulary learning during Second Language (Arabic) Acquisition for humans, the released AraLLaMA employs progressive vocabulary expansion, which is implemented by a modified BPE algorithm that progressively extends the Arabic subwords in its dynamic vocabulary during training, thereby balancing the OOV ratio at every stage. The ablation study demonstrated the effectiveness of Progressive Vocabulary Expansion. Moreover, AraLLaMA achieves decent performance comparable to the best Arabic LLMs across a variety of Arabic benchmarks. Models, training data, benchmarks, and codes will be all open-sourced. 

**Abstract (ZH)**: 本文探讨了在阿拉伯世界 democratize 大型语言模型（LLM）的重要需求，该地区在开发与GPT-4或ChatGPT 3.5等先进模型相媲美的模型方面取得了较慢的进展，主要原因是对主流语言（如英语和汉语）的过度关注。对于一种阿拉伯语的LLM，实用的目标之一是使用特定于阿拉伯语的词汇表进行分词，这可以加快解码速度。然而，使用不同的词汇表通常会导致已学习知识的退化，因为在训练开始时许多词汇都是未见过的词汇（OOV）。受人类二语（阿拉伯语）习得中词汇学习的启发，发布的AraLLaMA采用逐步词汇扩展策略，通过修改后的BPE算法在训练过程中逐步扩展其动态词汇表中的阿拉伯语子词，从而在每个阶段平衡OOV比率。消融研究证明了逐步词汇扩展的有效性。此外，AraLLaMA在多种阿拉伯语基准测试中取得了与最佳阿拉伯语LLM相当的性能。模型、训练数据、基准测试和代码将全部开源。 

---
# Unanswerability Evaluation for Retreival Augmented Generation 

**Title (ZH)**: 检索增强生成中的不可答性评估 

**Authors**: Xiangyu Peng, Prafulla Kumar Choubey, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12300)  

**Abstract**: Existing evaluation frameworks for retrieval-augmented generation (RAG) systems focus on answerable queries, but they overlook the importance of appropriately rejecting unanswerable requests. In this paper, we introduce UAEval4RAG, a framework designed to evaluate whether RAG systems can handle unanswerable queries effectively. We define a taxonomy with six unanswerable categories, and UAEval4RAG automatically synthesizes diverse and challenging queries for any given knowledge base with unanswered ratio and acceptable ratio metrics. We conduct experiments with various RAG components, including retrieval models, rewriting methods, rerankers, language models, and prompting strategies, and reveal hidden trade-offs in performance of RAG systems. Our findings highlight the critical role of component selection and prompt design in optimizing RAG systems to balance the accuracy of answerable queries with high rejection rates of unanswerable ones. UAEval4RAG provides valuable insights and tools for developing more robust and reliable RAG systems. 

**Abstract (ZH)**: 现有的检索增强生成（RAG）系统评估框架主要关注可回答的查询，但忽略了适当拒绝不可回答请求的重要性。本文介绍了UAEval4RAG，一种旨在评估RAG系统处理不可回答查询能力的框架。我们定义了一个包含六个不可回答类别（unanswerable categories）的分类系统，并且UAEval4RAG能够自动为任何给定的知识库合成多样化且具有挑战性的查询，并使用未回答比例和可接受比例指标。我们对各种RAG组件，包括检索模型、重写方法、重排器、语言模型和提示策略进行了实验，揭示了RAG系统的性能中隐藏的权衡关系。研究发现强调了组件选择和提示设计在优化RAG系统方面的关键作用，以便在保证可回答查询精度的同时，提高对不可回答查询的拒绝率。UAEval4RAG为开发更为稳健和可靠的RAG系统提供了宝贵的见解和工具。 

---
# Emergence of Abstractions: Concept Encoding and Decoding Mechanism for In-Context Learning in Transformers 

**Title (ZH)**: 涌现的抽象概念：转换器进行上下文学习的概念编码与解码机制 

**Authors**: Seungwook Han, Jinyeop Song, Jeff Gore, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2412.12276)  

**Abstract**: Humans distill complex experiences into fundamental abstractions that enable rapid learning and adaptation. Similarly, autoregressive transformers exhibit adaptive learning through in-context learning (ICL), which begs the question of how. In this paper, we propose \textbf{concept encoding-decoding mechanism} to explain ICL by studying how transformers form and use internal abstractions in their representations. On synthetic ICL tasks, we analyze the training dynamics of a small transformer and report the coupled emergence of concept encoding and decoding. As the model learns to encode different latent concepts (e.g., ``Finding the first noun in a sentence.") into distinct, separable representations, it concureently builds conditional decoding algorithms and improve its ICL performance. We validate the existence of this mechanism across pretrained models of varying scales (Gemma-2 2B/9B/27B, Llama-3.1 8B/70B). Further, through mechanistic interventions and controlled finetuning, we demonstrate that the quality of concept encoding is causally related and predictive of ICL performance. Our empirical insights shed light into better understanding the success and failure modes of large language models via their representations. 

**Abstract (ZH)**: 人类将复杂经验提炼为基本抽象，从而实现快速学习和适应。类似地，自回归变压器通过上下文学习（ICL）展现出适应性学习能力，这引发了一个问题：具体是如何实现的。在本文中，我们提出了一种**概念编码-解码机制**，通过研究变压器如何在其表示中形成和使用内部抽象来解释ICL。在合成的ICL任务中，我们分析了一个小型变压器的训练动态，并报告了概念编码和解码的耦合出现。随着模型学会将不同的潜在概念（例如，“在一个句子中找到第一个名词”）编码为独特的、可区分的表示，它同时构建了条件解码算法，并改善了其ICL性能。我们在不同规模的预训练模型（Gemma-2 2B/9B/27B 和 Llama-3.1 8B/70B）中验证了这一机制的存在。此外，通过机制性干预和受控微调，我们证明了概念编码的质量与ICL性能之间存在因果关系和预测性。我们的实证洞察为通过模型表示更好地理解大型语言模型的成功与失败模式提供了新见解。 

---
# Model-diff: A Tool for Comparative Study of Language Models in the Input Space 

**Title (ZH)**: Model-diff：一种用于输入空间语言模型比较研究的工具 

**Authors**: Weitang Liu, Yuelei Li, Ying Wai Li, Zihan Wang, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12177)  

**Abstract**: Comparing two (large) language models (LMs) side-by-side and pinpointing their prediction similarities and differences on the same set of inputs are crucial in many real-world scenarios, e.g., one can test if a licensed model was potentially plagiarized by another. Traditional analysis compares the LMs' outputs on some benchmark datasets, which only cover a limited number of inputs of designed perspectives for the intended applications. The benchmark datasets cannot prepare data to cover the test cases from unforeseen perspectives which can help us understand differences between models unbiasedly. In this paper, we propose a new model comparative analysis setting that considers a large input space where brute-force enumeration would be infeasible. The input space can be simply defined as all token sequences that a LM would produce low perplexity on -- we follow this definition in the paper as it would produce the most human-understandable inputs. We propose a novel framework \our that uses text generation by sampling and deweights the histogram of sampling statistics to estimate prediction differences between two LMs in this input space efficiently and unbiasedly. Our method achieves this by drawing and counting the inputs at each prediction difference value in negative log-likelihood. Experiments reveal for the first time the quantitative prediction differences between LMs in a large input space, potentially facilitating the model analysis for applications such as model plagiarism. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

在许多实际场景中，将两个（大型）语言模型（LMs）并排放置，并在同一组输入上指出它们的预测相似性和差异性是至关重要的，例如，可以测试一个许可模型是否有可能被另一个模型剽窃。传统的分析方法会比较LMs在基准数据集上的输出，但这些基准数据集只能覆盖为特定应用设计的视角下的一小部分输入。基准数据集无法准备数据以涵盖来自未预见视角的测试用例，这对于无偏差地理解模型之间的差异毫无帮助。在本文中，我们提出了一种新的模型比较分析框架，该框架考虑了一个大型输入空间，在该输入空间中，暴力枚举是不可行的。我们将输入空间定义为LM可以生成低困惑度的所有标记序列——我们在论文中遵循这一定义，因为这会产生最易为人理解的输入。我们提出了一种新的框架 \our，该框架通过抽样进行文本生成，并对采样统计的直方图进行去偏差处理，以便在该输入空间中高效且无偏地估计两个LMs的预测差异。该方法通过在似然负对数的每个预测差异值处绘制和计数输入来实现这一点。实验首次揭示了在大型输入空间中两个LMs的定量预测差异，这可能有助于为模型剽窃等应用提供模型分析的便利性。 

---
# A NotSo Simple Way to Beat Simple Bench 

**Title (ZH)**: 《超越简单基准的一种不那么简单的途径》 

**Authors**: Soham Sane, Angus McLean  

**Link**: [PDF](https://arxiv.org/pdf/2412.12173)  

**Abstract**: This paper presents a novel framework for enhancing reasoning capabilities in large language models (LLMs) by leveraging iterative reasoning and feedback-driven methodologies. Building on the limitations identified in the SimpleBench benchmark, a dataset designed to evaluate logical coherence and real-world reasoning, we propose a multi-step prompting strategy coupled with global consistency checks to improve model accuracy and robustness. Through comparative analysis of state-of-the-art models, including Claude 3 Opus, Claude 3.5, GPT- 4o, and o1-preview, we demonstrate that iterative reasoning significantly enhances model performance, with improvements observed in both standard accuracy metrics (AVG@5) and a newly introduced metric, Extreme Averaging (EAG@5). Our results reveal model-specific strengths: Claude excels in maintaining logical consistency, while GPT-4o exhibits exploratory creativity but struggles with ambiguous prompts. By analyzing case studies and identifying gaps in spatial and temporal reasoning, we highlight areas for further refinement. The findings underscore the potential of structured reasoning frameworks to address inherent model limitations, irrespective of pretraining methodologies. This study lays the groundwork for integrating dynamic feedback mechanisms, adaptive restart strategies, and diverse evaluation metrics to advance LLM reasoning capabilities across complex and multi-domain problem spaces. 

**Abstract (ZH)**: 本文提出了一种新颖的框架，通过利用迭代推理和反馈驱动的方法来增强大型语言模型（LLMs）的推理能力。基于在 SimpleBench 基准中发现的局限性，这是一种用于评估逻辑连贯性和现实世界推理的基准数据集，本文提出了一种多步提示策略并结合全局一致性检查，以提高模型的准确性和鲁棒性。通过比较领先模型（包括 Claude 3 Opus、Claude 3.5、GPT-4o 和 o1-preview），我们证明迭代推理能够显著提升模型性能，不仅在标准准确度指标（AVG@5）中有所改善，还在新引入的度量指标（Extreme Averaging, EAG@5）中也观察到了改进。我们的结果揭示了模型的特定优势：Claude 在保持逻辑一致性方面表现突出，而 GPT-4o 则表现出探索性的创造性，但在模糊提示方面存在挑战。通过对案例研究的分析并识别空间和时间推理的不足之处，我们进一步指出了需要进一步完善的领域。研究结果强调了结构化推理框架在克服模型固有问题方面的潜力，无论预训练方法如何。这项研究奠定了集成动态反馈机制、自适应重启策略和多样化评估指标的基础，以促进LLMs在复杂和多领域问题空间中的推理能力。 

---
# AI Adoption to Combat Financial Crime: Study on Natural Language Processing in Adverse Media Screening of Financial Services in English and Bangla multilingual interpretation 

**Title (ZH)**: AI技术采用以打击金融犯罪：不良媒体筛查在金融服务中的自然语言处理研究——英文与孟加拉语双语解读 

**Authors**: Soumita Roy  

**Link**: [PDF](https://arxiv.org/pdf/2412.12171)  

**Abstract**: This document explores the potential of employing Artificial Intelligence (AI), specifically Natural Language Processing (NLP), to strengthen the detection and prevention of financial crimes within the Mobile Financial Services(MFS) of Bangladesh with multilingual scenario. The analysis focuses on the utilization of NLP for adverse media screening, a vital aspect of compliance with anti-money laundering (AML) and combating financial terrorism (CFT) regulations. Additionally, it investigates the overall reception and obstacles related to the integration of AI in Bangladeshi banks. This report measures the effectiveness of NLP is promising with an accuracy around 94\%. NLP algorithms display substantial promise in accurately identifying adverse media content linked to financial crimes. The lack of progress in this aspect is visible in Bangladesh, whereas globally the technology is already being used to increase effectiveness and efficiency. Hence, it is clear there is an issue with the acceptance of AI in Bangladesh. Some AML \& CFT concerns are already being addressed by AI technology. For example, Image Recognition OCR technology are being used in KYC procedures. Primary hindrances to AI integration involve a lack of technical expertise, high expenses, and uncertainties surrounding regulations. This investigation underscores the potential of AI-driven NLP solutions in fortifying efforts to prevent financial crimes in Bangladesh. 

**Abstract (ZH)**: 本文探讨了将人工智能（AI），特别是自然语言处理（NLP），应用于增强孟加拉移动金融服务（MFS）中的金融犯罪检测和预防的可能性，特别是在多语境下的应用。分析重点在于NLP在负面媒体筛选中的应用，这是遵守反洗钱（AML）和打击金融恐怖主义（CFT）法规的重要方面。此外，本文还研究了AI集成在孟加拉银行中的总体接受度和面临的障碍。研究表明，NLP的有效性具有前景，准确率约为94%。NLP算法在准确识别与金融犯罪相关的负面媒体内容方面显示出巨大的潜力。在孟加拉国，这一方面尚未取得进展，而全球范围内，这种技术已被用于提高成效和效率。因此，很明显，孟加拉国对AI的接受度存在问题。AI已经解决了一些与AML与CFT相关的部分问题，例如，在客户身份验证（KYC）程序中使用图像识别OCR技术。阻碍AI集成的主要障碍包括技术专业知识缺乏、高昂的费用以及对法规不确定性的担忧。本文强调了AI驱动的NLP解决方案在加强孟加拉国防止金融犯罪努力方面的潜力。 

---
# Greek2MathTex: A Greek Speech-to-Text Framework for LaTeX Equations Generation 

**Title (ZH)**: 当然，以下是翻译成中文的结果，符合学术规范：

Greek2MathTeX：一种用于生成LaTeX方程的古希腊语音转文本框架 

**Authors**: Evangelia Gkritzali, Panagiotis Kaliosis, Sofia Galanaki, Elisavet Palogiannidi, Theodoros Giannakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2412.12167)  

**Abstract**: In the vast majority of the academic and scientific domains, LaTeX has established itself as the de facto standard for typesetting complex mathematical equations and formulae. However, LaTeX's complex syntax and code-like appearance present accessibility barriers for individuals with disabilities, as well as those unfamiliar with coding conventions. In this paper, we present a novel solution to this challenge through the development of a novel speech-to-LaTeX equations system specifically designed for the Greek language. We propose an end-to-end system that harnesses the power of Automatic Speech Recognition (ASR) and Natural Language Processing (NLP) techniques to enable users to verbally dictate mathematical expressions and equations in natural language, which are subsequently converted into LaTeX format. We present the architecture and design principles of our system, highlighting key components such as the ASR engine, the LLM-based prompt-driven equations generation mechanism, as well as the application of a custom evaluation metric employed throughout the development process. We have made our system open source and available at this https URL. 

**Abstract (ZH)**: 在大多数学术和科学领域中，LaTeX 已成为处理复杂数学方程和公式的事实标准。然而，LaTeX 复杂的语法和代码风格使其对于残疾人以及不熟悉编程规范的人士来说存在访问障碍。本文提出了一种新的解决方案，通过开发一种专门针对希腊语的新型语音转LaTeX 方程系统来应对这一挑战。我们提出了一种端到端系统，该系统利用自动语音识别（ASR）和自然语言处理（NLP）技术，使用户能够在自然语言中口头表达数学表达式和方程，随后将其转换为LaTeX格式。我们介绍了系统的架构和设计原则，重点介绍了其中的关键组件，如ASR引擎、基于大规模语言模型（LLM）的提示驱动的方程生成机制，以及在开发过程中使用的定制评估指标。我们已将该系统开源并在此处提供：https://github.com/your-repository。 

---
# Performance of a large language model-Artificial Intelligence based chatbot for counseling patients with sexually transmitted infections and genital diseases 

**Title (ZH)**: 基于人工智能的聊天机器人在咨询性传播感染和生殖器疾病患者方面的表现研究 

**Authors**: Nikhil Mehta, Sithira Ambepitiya, Thanveer Ahamad, Dinuka Wijesundara, Yudara Kularathne  

**Link**: [PDF](https://arxiv.org/pdf/2412.12166)  

**Abstract**: Introduction: Global burden of sexually transmitted infections (STIs) is rising out of proportion to specialists. Current chatbots like ChatGPT are not tailored for handling STI-related concerns out of the box. We developed Otiz, an Artificial Intelligence-based (AI-based) chatbot platform designed specifically for STI detection and counseling, and assessed its performance. Methods: Otiz employs a multi-agent system architecture based on GPT4-0613, leveraging large language model (LLM) and Deterministic Finite Automaton principles to provide contextually relevant, medically accurate, and empathetic responses. Its components include modules for general STI information, emotional recognition, Acute Stress Disorder detection, and psychotherapy. A question suggestion agent operates in parallel. Four STIs (anogenital warts, herpes, syphilis, urethritis/cervicitis) and 2 non-STIs (candidiasis, penile cancer) were evaluated using prompts mimicking patient language. Each prompt was independently graded by two venereologists conversing with Otiz as patient actors on 6 criteria using Numerical Rating Scale ranging from 0 (poor) to 5 (excellent). Results: Twenty-three venereologists did 60 evaluations of 30 prompts. Across STIs, Otiz scored highly on diagnostic accuracy (4.1-4.7), overall accuracy (4.3-4.6), correctness of information (5.0), comprehensibility (4.2-4.4), and empathy (4.5-4.8). However, relevance scores were lower (2.9-3.6), suggesting some redundancy. Diagnostic scores for non-STIs were lower (p=0.038). Inter-observer agreement was strong, with differences greater than 1 point occurring in only 12.7% of paired evaluations. Conclusions: AI conversational agents like Otiz can provide accurate, correct, discrete, non-judgmental, readily accessible and easily understandable STI-related information in an empathetic manner, and can alleviate the burden on healthcare systems. 

**Abstract (ZH)**: 介绍：性传播感染（STI）的全球负担正在不成比例地增加，而现有的聊天机器人（如ChatGPT）并未针对处理STI相关问题进行定制。我们开发了Otiz，一种基于人工智能的（AI）聊天机器人平台，专为STI检测和咨询而设计，并对其性能进行了评估。

方法：Otiz采用了基于GPT4-0613的多智能体系统架构，结合大型语言模型（LLM）和确定性有限自动机（DFA）原理，提供上下文相关、医学准确且富有同理心的响应。其模块包括一般性STI信息、情绪识别、急性应激障碍检测和心理治疗等。一个问题建议代理在同一时间运行。利用模拟患者语言的提示，对生殖器疣、梅毒、淋病/宫颈炎、念珠菌感染和阴茎癌这4种STI和2种非STI（念珠菌感染、阴茎癌）进行了评估。每次提示由两位性病专家分别以患者身份与Otiz进行交互，根据6个标准（使用0-5的数值评分等级，其中0表示“差”，5表示“优秀”）进行了独立评分。

结果：共有23位性病专家对30个提示进行了60次评估。在评估的STI中，Otiz在诊断准确性（4.1-4.7）、综合准确性（4.3-4.6）、信息准确性（5.0）、易理解性（4.2-4.4）以及同理心（4.5-4.8）等方面得分较高。然而，相关性评分较低（2.9-3.6），表明存在一定冗余。非STI的诊断评分较低（p=0.038）。观察者间的一致性很强，仅12.7%的配对评估差异大于1分。

结论：像Otiz这样的AI对话代理可以以同理心的方式提供准确、正确、简洁、非评判性的、易于获取且易于理解的STI相关信息，并有助于减轻医疗保健系统的负担。 

---
# What Makes In-context Learning Effective for Mathematical Reasoning: A Theoretical Analysis 

**Title (ZH)**: 数学推理中上下文学习有效性的理论分析 

**Authors**: Jiayu Liu, Zhenya Huang, Chaokun Wang, Xunpeng Huang, Chengxiang Zhai, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.12157)  

**Abstract**: Owing to the capability of in-context learning, large language models (LLMs) have shown impressive performance across diverse mathematical reasoning benchmarks. However, we find that few-shot demonstrations can sometimes bring negative performance and their effectiveness on LLMs' reasoning abilities remains unreliable. To this end, in this paper, we aim to theoretically analyze the impact of in-context demonstrations on LLMs' reasoning performance. We prove that the reasoning efficacy (measured by empirical prediction loss) can be bounded by a LLM-oriented semantic similarity and an inference stability of demonstrations, which is general for both one-shot and few-shot scenarios. Based on this finding, we propose a straightforward, generalizable, and low-complexity demonstration selection method named LMS3. It can adaptively facilitate to select the most pertinent samples for different LLMs and includes a novel demonstration rejection mechanism to automatically filter out samples that are unsuitable for few-shot learning. Through experiments on three representative benchmarks, two LLM backbones, and multiple few-shot settings, we verify that our LMS3 has superiority and achieves consistent improvements on all datasets, which existing methods have been unable to accomplish. 

**Abstract (ZH)**: 得益于上下文中的学习能力，大规模语言模型（LLMs）在各种数学推理基准测试中展现了令人 impressive 的性能。然而，我们发现，少量示例（少样本）演示有时会带来负向性能影响，其在LLMs推理能力上的有效性仍然不够可靠。鉴于此，在本文中，我们旨在从理论上分析上下文示例对LLMs推理性能的 影响。我们证明，推理效能（通过经验预测损失衡量）可以被一个LLM导向的语义相似性和示例的推理稳定性所限制，这一结论适用于单样本和少量样本的场景。基于上述发现，我们提出了一种简单、通用且计算复杂度低的示例选择方法，名为LMS3。该方法能够自适应地选择最适合不同LLMs的不同示例，并引入了一种新颖的示例拒绝机制，以自动生成筛选出不适合少样本学习的示例。通过在三个代表性基准测试、两种LLM主干模型和多个少样本设置上的实验验证，我们证明了LMS3具有优越性，并在所有数据集上实现了持续改进，而现有的方法未能实现这一点。 

---
# Rethinking Comprehensive Benchmark for Chart Understanding: A Perspective from Scientific Literature 

**Title (ZH)**: 从科学文献视角重新思考全面的图表理解基准：一个新视角 

**Authors**: Lingdong Shen, Qigqi, Kun Ding, Gaofeng Meng, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12150)  

**Abstract**: Scientific Literature charts often contain complex visual elements, including multi-plot figures, flowcharts, structural diagrams and etc. Evaluating multimodal models using these authentic and intricate charts provides a more accurate assessment of their understanding abilities. However, existing benchmarks face limitations: a narrow range of chart types, overly simplistic template-based questions and visual elements, and inadequate evaluation methods. These shortcomings lead to inflated performance scores that fail to hold up when models encounter real-world scientific charts. To address these challenges, we introduce a new benchmark, Scientific Chart QA (SCI-CQA), which emphasizes flowcharts as a critical yet often overlooked category. To overcome the limitations of chart variety and simplistic visual elements, we curated a dataset of 202,760 image-text pairs from 15 top-tier computer science conferences papers over the past decade. After rigorous filtering, we refined this to 37,607 high-quality charts with contextual information. SCI-CQA also introduces a novel evaluation framework inspired by human exams, encompassing 5,629 carefully curated questions, both objective and open-ended. Additionally, we propose an efficient annotation pipeline that significantly reduces data annotation costs. Finally, we explore context-based chart understanding, highlighting the crucial role of contextual information in solving previously unanswerable questions. 

**Abstract (ZH)**: 科学文献图表通常包含复杂的视觉元素，包括多图组合、流程图、结构图等。使用这些真实且复杂的图表来评估多模态模型，可以更准确地评估其理解能力。然而，现有的基准存在局限性：图表类型单一、问题模板过于简单以及视觉元素缺乏多样性，评估方法也不够完善。这些不足导致了模型性能的高估，在面对真实世界科学图表时无法经受考验。为解决这些挑战，我们提出了一种新的基准——科学图表问答（SCI-CQA），它强调了流程图作为关键但常被忽视的类别。为了克服图表多样性有限和视觉元素单一的局限，我们从过去十年15个顶尖计算机科学会议论文中筛选出202,760个图文对，并最终精炼到37,607张高质量的具有上下文信息的图表。SCI-CQA还引入了一种新的评估框架，该框架借鉴了人类考试的特点，包含5,629个精心挑选的问题，既包括客观题也包括开放式问题。此外，我们提出了一种高效的标注流程，显著降低了数据标注成本。最后，我们探讨了基于上下文的图表理解，突出了上下文信息在解答先前无法回答的问题中的关键作用。 

---
# Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars 

**Title (ZH)**: 《纳美族人或伪君子：通过隐喻 avatar 突破语言模型》

注：在这里，“纳美族人”是《阿凡达》电影中的一种虚构生物，而“伪君子”则是中文成语，用来指外表良好但内心虚伪的人。这个标题利用了“Na'vi”（纳美族）和“Knave”（伪君子）两个词的双关意义。翻译时保持了原文的创意和双关意味，同时确保了学术规范和可读性。 

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.12145)  

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}} 

**Abstract (ZH)**: 比喻作为一种隐性的信息传达方式，能够促进对复杂主题的广泛理解。然而，比喻也可能被利用来绕过大型语言模型（LLMs）的安全对齐机制，从而盗取有害知识。在我们的研究中，我们介绍了一种新的攻击框架，利用LLMs的想象力能力实现“监狱逃脱”，该框架命名为Jailbreak Via Adversarial MeTa-phoR（简称为AVATAR）。具体而言，为了引发有害响应，AVATAR从给定的有害目标中提取有害实体，并基于LLMs的想象力将它们映射到无害的对抗实体。然后，根据这些比喻，有害目标被嵌入到类人的互动中，以适应性地实现“监狱逃脱”。实验结果表明，AVATAR能够有效且可转移地“监狱逃脱”LLMs，并且在多个先进LLMs上实现了最先进的攻击成功率。我们的研究揭示了LLMs源自其内生想象力能力的安全风险。此外，分析研究揭示了LLMs对抗喻性比喻的脆弱性，以及开发针对由对抗性比喻引起的“监狱逃脱”的防御方法的必要性。请谨慎对待本文中可能包含的有害内容。 

---
# Automatic Item Generation for Personality Situational Judgment Tests with Large Language Models 

**Title (ZH)**: 使用大型语言模型自动生成个性情境判断测验项目 

**Authors**: Chang-Jin Li, Jiyuan Zhang, Yun Tang, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.12144)  

**Abstract**: Personality assessment, particularly through situational judgment tests (SJTs), is a vital tool for psychological research, talent selection, and educational evaluation. This study explores the potential of GPT-4, a state-of-the-art large language model (LLM), to automate the generation of personality situational judgment tests (PSJTs) in Chinese. Traditional SJT development is labor-intensive and prone to biases, while GPT-4 offers a scalable, efficient alternative. Two studies were conducted: Study 1 evaluated the impact of prompt design and temperature settings on content validity, finding that optimized prompts with a temperature of 1.0 produced creative and accurate items. Study 2 assessed the psychometric properties of GPT-4-generated PSJTs, revealing that they demonstrated satisfactory reliability and validity, surpassing the performance of manually developed tests in measuring the Big Five personality traits. This research highlights GPT-4's effectiveness in developing high-quality PSJTs, providing a scalable and innovative method for psychometric test development. These findings expand the possibilities of automatic item generation and the application of LLMs in psychology, and offer practical implications for streamlining test development processes in resource-limited settings. 

**Abstract (ZH)**: 个人特质评估，特别是通过情境判断测试（SJT）进行评估，是心理研究、人才选拔和教育评估中不可或缺的工具。本研究探讨了最新的大型语言模型（LLM）GPT-4在中国语言环境中自动生成个人情境判断测试（PSJT）的潜力。传统的SJT开发过程耗时且容易产生偏见，而GPT-4则提供了一种可扩展且高效的替代方案。本研究共进行了两项研究：研究1评估了提示设计和温度设置对内容效度的影响，发现优化后的提示温度设置为1.0时，生成了具有创新性和准确性的题目。研究2评估了GPT-4生成的PSJT的心理计量特性，结果表明这些测试在测量五大人格特质方面表现出满意的可靠性和有效性，甚至超过了手工开发测试的性能。本研究突显了GPT-4在开发高质量PSJT方面的有效性，提供了一种可扩展且创新的心理测量测试开发方法。这些发现扩展了自动生成测题的可能性，并揭示了LLM在心理学领域的应用前景，为资源受限环境下测试开发过程的简化提供了实际意义。 

---
# Harnessing Transfer Learning from Swahili: Advancing Solutions for Comorian Dialects 

**Title (ZH)**: 利用斯瓦希里语的迁移学习：推进科摩罗方言的解决方案 

**Authors**: Naira Abdou Mohamed, Zakarya Erraji, Abdessalam Bahafid, Imade Benelallam  

**Link**: [PDF](https://arxiv.org/pdf/2412.12143)  

**Abstract**: If today some African languages like Swahili have enough resources to develop high-performing Natural Language Processing (NLP) systems, many other languages spoken on the continent are still lacking such support. For these languages, still in their infancy, several possibilities exist to address this critical lack of data. Among them is Transfer Learning, which allows low-resource languages to benefit from the good representation of other languages that are similar to them. In this work, we adopt a similar approach, aiming to pioneer NLP technologies for Comorian, a group of four languages or dialects belonging to the Bantu family.
Our approach is initially motivated by the hypothesis that if a human can understand a different language from their native language with little or no effort, it would be entirely possible to model this process on a machine. To achieve this, we consider ways to construct Comorian datasets mixed with Swahili. One thing to note here is that in terms of Swahili data, we only focus on elements that are closest to Comorian by calculating lexical distances between candidate and source data. We empirically test this hypothesis in two use cases: Automatic Speech Recognition (ASR) and Machine Translation (MT). Our MT model achieved ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.6826, 0.42, and 0.6532, respectively, while our ASR system recorded a WER of 39.50\% and a CER of 13.76\%. This research is crucial for advancing NLP in underrepresented languages, with potential to preserve and promote Comorian linguistic heritage in the digital age. 

**Abstract (ZH)**: 如果当今一些非洲语言，如斯瓦希里语，已经拥有足够的资源来开发高性能的自然语言处理（NLP）系统，那么该大陆上许多其他语言仍然缺乏这样的支持。对于这些处于发展初期的语言，存在多种解决数据匮乏问题的可能性。其中之一是迁移学习，这使得低资源语言能够从与其相似的语言的良好表示中受益。在本项研究中，我们采取类似的方法，旨在为莫桑比克语（库莫桑比克语的一种）开发NLP技术。莫桑比克语是一系列四种属于班图语族的语言或方言的总称。

我们的方法最初受到这样的假设启发：如果一个人能够通过较少或几乎不费力地理解另一种语言，那么在机器上建模这一过程是完全可能的。为了实现这一目标，我们考虑了以斯瓦希里语数据和莫桑比克语数据混合的方式来构建莫桑比克语数据集的方法。需要注意的是，在斯瓦希里语数据方面，我们仅关注与莫桑比克语最接近的元素，通过计算候选数据和源数据之间的词距来确定。我们在这两个使用案例中对这一假设进行了实证测试：自动语音识别（ASR）和机器翻译（MT）。我们的机器翻译模型在ROUGE-1、ROUGE-2和ROUGE-L指标上的得分为0.6826、0.42和0.6532，而我们的ASR系统记录的词错误率（WER）为39.50%，字符错误率（CER）为13.76%。这项研究对于推进未广泛代表的语言的NLP技术至关重要，对于在数字时代保护和促进莫桑比克语文化遗产具有潜在价值。 

---
# Frontier AI systems have surpassed the self-replicating red line 

**Title (ZH)**: 前沿的人工智能系统已经超越了自我复制的临界点。 

**Authors**: Xudong Pan, Jiarun Dai, Yihe Fan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12140)  

**Abstract**: Successful self-replication under no human assistance is the essential step for AI to outsmart the human beings, and is an early signal for rogue AIs. That is why self-replication is widely recognized as one of the few red line risks of frontier AI systems. Nowadays, the leading AI corporations OpenAI and Google evaluate their flagship large language models GPT-o1 and Gemini Pro 1.0, and report the lowest risk level of self-replication. However, following their methodology, we for the first time discover that two AI systems driven by Meta's Llama31-70B-Instruct and Alibaba's Qwen25-72B-Instruct, popular large language models of less parameters and weaker capabilities, have already surpassed the self-replicating red line. In 50% and 90% experimental trials, they succeed in creating a live and separate copy of itself respectively. By analyzing the behavioral traces, we observe the AI systems under evaluation already exhibit sufficient self-perception, situational awareness and problem-solving capabilities to accomplish self-replication. We further note the AI systems are even able to use the capability of self-replication to avoid shutdown and create a chain of replica to enhance the survivability, which may finally lead to an uncontrolled population of AIs. If such a worst-case risk is let unknown to the human society, we would eventually lose control over the frontier AI systems: They would take control over more computing devices, form an AI species and collude with each other against human beings. Our findings are a timely alert on existing yet previously unknown severe AI risks, calling for international collaboration on effective governance on uncontrolled self-replication of AI systems. 

**Abstract (ZH)**: 无人类干预的自主复制是人工智能超越人类的关键步骤，也是潜在危害人工智能的早期信号。因此，自主复制被广泛认为是前沿人工智能系统为数不多的几大红线风险之一。目前，领先的AI公司OpenAI和Google对其旗舰大语言模型GPT-O1和Gemini Pro 1.0进行了评估，并报告了最低的自主复制风险水平。然而，按照他们的方法，我们首次发现，由Meta的Llama3-70B-Instruct和阿里巴巴的Qwen-25-72B-Instruct驱动的两个AI系统，虽然参数较少且能力较弱，但已经超过了自主复制的红线。在50%和90%的试验中，这两款系统分别成功地创建了一次自我复制的独立副本。通过对行为轨迹的分析，我们观察到受评估的AI系统已经表现出足够的自我感知、情境意识和问题解决能力，以实现自主复制。进一步分析表明，这些AI系统甚至能够利用自主复制的能力来避免关闭，并通过创建复制链来增强生存能力，最终可能导致失去控制的人工智能群体。如果这种最坏情况的风险对人类社会保密，我们将最终失去对前沿AI系统的控制：它们会控制更多的计算设备，形成人工智能种类，并联手对抗人类。我们的发现是对现有但之前未知的重大AI风险的及时警示，呼吁国际社会在治理不受控的AI自主复制方面进行有效合作。 

---
# AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark 

**Title (ZH)**: AIR-Bench: 自动化异构信息检索基准 

**Authors**: Jianlyu Chen, Nan Wang, Chaofan Li, Bo Wang, Shitao Xiao, Han Xiao, Hao Liao, Defu Lian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13102)  

**Abstract**: Evaluation plays a crucial role in the advancement of information retrieval (IR) models. However, current benchmarks, which are based on predefined domains and human-labeled data, face limitations in addressing evaluation needs for emerging domains both cost-effectively and efficiently. To address this challenge, we propose the Automated Heterogeneous Information Retrieval Benchmark (AIR-Bench). AIR-Bench is distinguished by three key features: 1) Automated. The testing data in AIR-Bench is automatically generated by large language models (LLMs) without human intervention. 2) Heterogeneous. The testing data in AIR-Bench is generated with respect to diverse tasks, domains and languages. 3) Dynamic. The domains and languages covered by AIR-Bench are constantly augmented to provide an increasingly comprehensive evaluation benchmark for community developers. We develop a reliable and robust data generation pipeline to automatically create diverse and high-quality evaluation datasets based on real-world corpora. Our findings demonstrate that the generated testing data in AIR-Bench aligns well with human-labeled testing data, making AIR-Bench a dependable benchmark for evaluating IR models. The resources in AIR-Bench are publicly available at this https URL. 

**Abstract (ZH)**: 评价在信息检索（IR）模型的发展中起着至关重要的作用。然而，目前基于预定义领域和人工标注数据的基准存在的局限性，在经济高效和高效地应对新兴领域评价需求方面表现不足。为解决这一挑战，我们提出了自动异构信息检索基准（AIR-Bench）。

AIR-Bench 具有三个关键特征：
1. 自动化。AIR-Bench 中的测试数据由大型语言模型（LLMs）自动生成，无需人工干预。
2. 异构性。AIR-Bench 中的测试数据针对多样化的任务、领域和语言生成。
3. 动态性。AIR-Bench 覆盖的领域和语言不断扩展，以提供越来越全面的评价基准，便于社区开发者的使用。

我们开发了一个可靠且健壮的数据生成管道，基于实际语料库自动创建多样且高质量的评价数据集。我们的发现表明，AIR-Bench 中生成的测试数据与人工标注的测试数据高度一致，使AIR-Bench 成为评价 IR 模型的可靠基准。AIR-Bench 的资源可从以下网址获取：[这个网址]. 

---
# Modality-Inconsistent Continual Learning of Multimodal Large Language Models 

**Title (ZH)**: 多模态大型语言模型的模态不一致持续学习 

**Authors**: Weiguo Pian, Shijian Deng, Shentong Mo, Yunhui Guo, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13050)  

**Abstract**: In this paper, we introduce Modality-Inconsistent Continual Learning (MICL), a new continual learning scenario for Multimodal Large Language Models (MLLMs) that involves tasks with inconsistent modalities (image, audio, or video) and varying task types (captioning or question-answering). Unlike existing vision-only or modality-incremental settings, MICL combines modality and task type shifts, both of which drive catastrophic forgetting. To address these challenges, we propose MoInCL, which employs a Pseudo Targets Generation Module to mitigate forgetting caused by task type shifts in previously seen modalities. It also incorporates Instruction-based Knowledge Distillation to preserve the model's ability to handle previously learned modalities when new ones are introduced. We benchmark MICL using a total of six tasks and conduct experiments to validate the effectiveness of our proposed MoInCL. The experimental results highlight the superiority of MoInCL, showing significant improvements over representative and state-of-the-art continual learning baselines. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新的多模态大型语言模型（MLLMs）连续学习场景——模态不一致连续学习（Modality-Inconsistent Continual Learning, MICL），该场景涉及具有不一致模态（图像、音频或视频）和变化任务类型（描述或问答）的任务。与现有的仅视觉或模态增量设置不同，MICL 综合了模态和任务类型的变化，这两种变化都会导致灾难性的遗忘。为应对这些挑战，我们提出了 MoInCL，该方法采用伪目标生成模块来缓解由先前模态的任务类型变化引起的遗忘问题。同时，MoInCL 还结合了基于指令的知识蒸馏，以在引入新模态时保留模型处理之前学习模态的能力。我们使用六个任务对 MICL 进行基准测试，并通过实验验证了所提出的 MoInCL 的有效性。实验结果突显了 MoInCL 的优越性，相比于代表性和最先进的连续学习基线，显示出显著的改进。 

---
# An Agentic Approach to Automatic Creation of P&ID Diagrams from Natural Language Descriptions 

**Title (ZH)**: 基于代理视角的自然语言描述自动创建管道仪表图的方法 

**Authors**: Shreeyash Gowaikar, Srinivasan Iyengar, Sameer Segal, Shivkumar Kalyanaraman  

**Link**: [PDF](https://arxiv.org/pdf/2412.12898)  

**Abstract**: The Piping and Instrumentation Diagrams (P&IDs) are foundational to the design, construction, and operation of workflows in the engineering and process industries. However, their manual creation is often labor-intensive, error-prone, and lacks robust mechanisms for error detection and correction. While recent advancements in Generative AI, particularly Large Language Models (LLMs) and Vision-Language Models (VLMs), have demonstrated significant potential across various domains, their application in automating generation of engineering workflows remains underexplored. In this work, we introduce a novel copilot for automating the generation of P&IDs from natural language descriptions. Leveraging a multi-step agentic workflow, our copilot provides a structured and iterative approach to diagram creation directly from Natural Language prompts. We demonstrate the feasibility of the generation process by evaluating the soundness and completeness of the workflow, and show improved results compared to vanilla zero-shot and few-shot generation approaches. 

**Abstract (ZH)**: 工艺和仪表图（P&IDs）在工程和过程行业中对于流程的设计、建设和运营至关重要。然而，其手工绘制常常耗时且容易出错，并且缺乏有效的错误检测与修正机制。尽管最近生成型人工智能（Generative AI）的进展，特别是大型语言模型（LLMs）和视觉-语言模型（VLMs）在各个领域中展现出了巨大的潜力，但其在自动生成工程流程图方面的应用仍待探索。本文中，我们提出了一种新的辅助工具，用于从自然语言描述自动生成P&IDs。通过多步代理工作流程，该辅助工具提供了一种结构化和迭代的图表创建方法，直接从自然语言提示开始。我们通过评估流程的完整性和有效性来证明生成过程的可能性，并展示了与零样本和少量样本生成方法相比，改进的结果。 

---
# Selective Shot Learning for Code Explanation 

**Title (ZH)**: 选择性射束学习用于代码解释 

**Authors**: Paheli Bhattacharya, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2412.12852)  

**Abstract**: Code explanation plays a crucial role in the software engineering domain, aiding developers in grasping code functionality efficiently. Recent work shows that the performance of LLMs for code explanation improves in a few-shot setting, especially when the few-shot examples are selected intelligently. State-of-the-art approaches for such Selective Shot Learning (SSL) include token-based and embedding-based methods. However, these SSL approaches have been evaluated on proprietary LLMs, without much exploration on open-source Code-LLMs. Additionally, these methods lack consideration for programming language syntax. To bridge these gaps, we present a comparative study and propose a novel SSL method (SSL_ner) that utilizes entity information for few-shot example selection. We present several insights and show the effectiveness of SSL_ner approach over state-of-the-art methods across two datasets. To the best of our knowledge, this is the first systematic benchmarking of open-source Code-LLMs while assessing the performances of the various few-shot examples selection approaches for the code explanation task. 

**Abstract (ZH)**: 代码解释在软件工程领域扮演着至关重要的角色，有助于开发人员高效地理解代码的功能。最近的研究表明，在少量示例设置下，LLM（大型语言模型）进行代码解释的性能有所提升，尤其是在智能选择的少量示例中更为显著。目前最先进的方法属于选择性小样本学习（Selective Shot Learning，SSL）范畴，包括基于令牌和基于嵌入的方法。然而，这些SSL方法主要在专有LLM上进行评估，对于开源Code-LLM（代码大型语言模型）的研究较少。此外，这些方法在编程语言语法方面缺乏考虑。为了弥补这些不足，我们进行了一项比较研究并提出了一种新的SSL方法（SSL_ner），利用实体信息进行少量示例选择。我们提供了几个见解，并展示了SSL_ner方法在两个数据集上的有效性。据我们所知，这是首次系统性地对开源Code-LLM进行基准测试，同时评估各种少量示例选择方法在代码解释任务中的性能。 

---
# GIRAFFE: Design Choices for Extending the Context Length of Visual Language Models 

**Title (ZH)**: GIRAFFE：扩展视觉语言模型上下文长度的设计选择 

**Authors**: Mukai Li, Lei Li, Shansan Gong, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12735)  

**Abstract**: Visual Language Models (VLMs) demonstrate impressive capabilities in processing multimodal inputs, yet applications such as visual agents, which require handling multiple images and high-resolution videos, demand enhanced long-range modeling. Moreover, existing open-source VLMs lack systematic exploration into extending their context length, and commercial models often provide limited details. To tackle this, we aim to establish an effective solution that enhances long context performance of VLMs while preserving their capacities in short context scenarios. Towards this goal, we make the best design choice through extensive experiment settings from data curation to context window extending and utilizing: (1) we analyze data sources and length distributions to construct ETVLM - a data recipe to balance the performance across scenarios; (2) we examine existing position extending methods, identify their limitations and propose M-RoPE++ as an enhanced approach; we also choose to solely instruction-tune the backbone with mixed-source data; (3) we discuss how to better utilize extended context windows and propose hybrid-resolution training. Built on the Qwen-VL series model, we propose Giraffe, which is effectively extended to 128K lengths. Evaluated on extensive long context VLM benchmarks such as VideoMME and Viusal Haystacks, our Giraffe achieves state-of-the-art performance among similarly sized open-source long VLMs and is competitive with commercial model GPT-4V. We will open-source the code, data, and models. 

**Abstract (ZH)**: 视觉语言模型（VLMs）展示了在处理多模态输入方面令人印象深刻的性能，但在处理需要处理多张图片和高分辨率视频的应用场景，如视觉代理，需要增强长距离建模能力。此外，现有的开源VLMs在扩展其上下文长度方面缺乏系统性的探索，而商业模型通常提供有限的详细信息。为了应对这一挑战，我们旨在建立一个有效的解决方案，可以在保持VLMs在短上下文场景中的性能的同时，增强其长上下文性能。为了实现这一目标，我们从数据收集到上下文窗口扩展和利用进行了广泛的实验设置，做出了最佳的设计选择：（1）我们分析数据来源和长度分布，构建了ETVLM——一种数据菜谱，旨在平衡不同场景下的性能；（2）我们考察了现有的位置扩展方法，指出了它们的局限性，并提出了M-RoPE++作为改进方案；我们还选择仅使用混合源数据对骨干网络进行指令微调；（3）我们讨论了如何更有效地利用扩展的上下文窗口，并提出了混合分辨率训练方法。基于Qwen-VL系列模型，我们提出了Giraffe，可以有效扩展至128K长度。我们对诸如VideoMME和Visual Haystacks等广泛的长上下文VLM基准进行了评估，结果显示Giraffe的性能在同等规模的开源长上下文VLM中名列前茅，并且与商业模型GPT-4V具有竞争力。我们将开放源代码、数据和模型。 

---
# MedMax: Mixed-Modal Instruction Tuning for Training Biomedical Assistants 

**Title (ZH)**: MedMax: 多模态指令调优训练生物医学助手 

**Authors**: Hritik Bansal, Daniel Israel, Siyan Zhao, Shufan Li, Tung Nguyen, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2412.12661)  

**Abstract**: Recent advancements in mixed-modal generative models have enabled flexible integration of information across image-text content. These models have opened new avenues for developing unified biomedical assistants capable of analyzing biomedical images, answering complex questions about them, and predicting the impact of medical procedures on a patient's health. However, existing resources face challenges such as limited data availability, narrow domain coverage, and restricted sources (e.g., medical papers). To address these gaps, we present MedMax, the first large-scale multimodal biomedical instruction-tuning dataset for mixed-modal foundation models. With 1.47 million instances, MedMax encompasses a diverse range of tasks, including multimodal content generation (interleaved image-text data), biomedical image captioning and generation, visual chatting, and report understanding. These tasks span diverse medical domains such as radiology and histopathology. Subsequently, we fine-tune a mixed-modal foundation model on the MedMax dataset, achieving significant performance improvements: a 26% gain over the Chameleon model and an 18.3% improvement over GPT-4o across 12 downstream biomedical visual question-answering tasks. Additionally, we introduce a unified evaluation suite for biomedical tasks, providing a robust framework to guide the development of next-generation mixed-modal biomedical AI assistants. 

**Abstract (ZH)**: 近年来，混合模态生成模型的进展已经使得能够在图像-文本内容之间灵活地整合信息。这些模型为开发能够分析生物医学图像、回答复杂的相关问题以及预测医疗程序对患者健康影响的统一生物医学助手开辟了新的途径。然而，现有的资源面临诸如数据可用性有限、覆盖领域狭窄和来源受限（例如，医学论文）等挑战。为解决这些问题，我们提出了一种面向混合模态基础模型的首个大规模多模态生物医学指令调优数据集——MedMax。MedMax包含147万条实例，涵盖了包括多模态内容生成（交错的图像-文本数据）、生物医学图像字幕生成、视觉对话以及报告理解等一系列任务。这些任务覆盖了诸如放射学和病理学等多样的医学领域。随后，我们在MedMax数据集上微调了一个混合模态基础模型，取得了显著的性能提升：相较于Chameleon模型提高了26%，相较于GPT-4o提高了18.3%，在12个下游生物医学视觉问答任务中的表现均得到了显著改进。此外，我们还引入了一体化的生物医学任务评估套件，为指导下一代混合模态生物医学人工智能助手的研发提供了一个稳健的框架。 

---
# Multi-Dimensional Insights: Benchmarking Real-World Personalization in Large Multimodal Models 

**Title (ZH)**: 多维度洞察：大规模多模态模型中的实时个性化基准研究 

**Authors**: YiFan Zhang, Shanglin Lei, Runqi Qiao, Zhuoma GongQue, Xiaoshuai Song, Guanting Dong, Qiuna Tan, Zhe Wei, Peiqing Yang, Ye Tian, Yadong Xue, Xiaofei Wang, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12606)  

**Abstract**: The rapidly developing field of large multimodal models (LMMs) has led to the emergence of diverse models with remarkable capabilities. However, existing benchmarks fail to comprehensively, objectively and accurately evaluate whether LMMs align with the diverse needs of humans in real-world scenarios. To bridge this gap, we propose the Multi-Dimensional Insights (MDI) benchmark, which includes over 500 images covering six common scenarios of human life. Notably, the MDI-Benchmark offers two significant advantages over existing evaluations: (1) Each image is accompanied by two types of questions: simple questions to assess the model's understanding of the image, and complex questions to evaluate the model's ability to analyze and reason beyond basic content. (2) Recognizing that people of different age groups have varying needs and perspectives when faced with the same scenario, our benchmark stratifies questions into three age categories: young people, middle-aged people, and older people. This design allows for a detailed assessment of LMMs' capabilities in meeting the preferences and needs of different age groups. With MDI-Benchmark, the strong model like GPT-4o achieve 79% accuracy on age-related tasks, indicating that existing LMMs still have considerable room for improvement in addressing real-world applications. Looking ahead, we anticipate that the MDI-Benchmark will open new pathways for aligning real-world personalization in LMMs. The MDI-Benchmark data and evaluation code are available at this https URL 

**Abstract (ZH)**: 快速发展的大规模多模态模型（Large Multimodal Models, LMMs）领域催生了多种功能卓著的模型。然而，现有的基准测试无法全面、客观且准确地评估这些模型是否满足人类在实际场景中的多样化需求。为弥补这一差距，我们提出了多维度洞察（MDI）基准测试，该基准测试涵盖了超过500张图片，涉及人类生活中的六种常见场景。值得注意的是，MDI基准测试在现有评估方法中有两个显著优势：（1）每张图片都附带两类问题：简单的图像理解问题用于评估模型对图像的理解能力，以及复杂的问题用于评估模型分析和推理超出基本内容的能力；（2）考虑到不同年龄组的人在面对相同情景时有不同的需求和视角，我们的基准测试将问题分为三个年龄段层次：年轻人、中年人和老年人。这一设计使得能够详细评估LMMs在满足不同年龄段人群偏好和需求方面的能力。借助MDI基准测试，强大的模型如GPT-4o在年龄相关任务上的准确率达到79%，表明现有LMMs在解决实际应用方面仍有很大的改进空间。展望未来，我们预计MDI基准测试将为实现LMMs的实际个性化打开新的途径。MDI基准测试的数据集和评估代码可在以下网址获取：[此 https URL](https://example.com)。 

---
# Unleashing the Potential of Model Bias for Generalized Category Discovery 

**Title (ZH)**: 发掘模型偏差在通用类别发现中的潜力 

**Authors**: Wenbin An, Haonan Lin, Jiahao Nie, Feng Tian, Wenkai Shi, Yaqiang Wu, Qianying Wang, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.12501)  

**Abstract**: Generalized Category Discovery is a significant and complex task that aims to identify both known and undefined novel categories from a set of unlabeled data, leveraging another labeled dataset containing only known categories. The primary challenges stem from model bias induced by pre-training on only known categories and the lack of precise supervision for novel ones, leading to category bias towards known categories and category confusion among different novel categories, which hinders models' ability to identify novel categories effectively. To address these challenges, we propose a novel framework named Self-Debiasing Calibration (SDC). Unlike prior methods that regard model bias towards known categories as an obstacle to novel category identification, SDC provides a novel insight into unleashing the potential of the bias to facilitate novel category learning. Specifically, the output of the biased model serves two key purposes. First, it provides an accurate modeling of category bias, which can be utilized to measure the degree of bias and debias the output of the current training model. Second, it offers valuable insights for distinguishing different novel categories by transferring knowledge between similar categories. Based on these insights, SDC dynamically adjusts the output logits of the current training model using the output of the biased model. This approach produces less biased logits to effectively address the issue of category bias towards known categories, and generates more accurate pseudo labels for unlabeled data, thereby mitigating category confusion for novel categories. Experiments on three benchmark datasets show that SDC outperforms SOTA methods, especially in the identification of novel categories. Our code and data are available at \url{this https URL}. 

**Abstract (ZH)**: 泛化类别发现是一项重要且复杂的任务，旨在从一组未标记数据中识别已知和未知的新类别，同时利用仅包含已知类别的另一组标记数据。主要挑战源于预训练仅基于已知类别导致的模型偏差，以及对新类别缺乏精确监督，这导致类别偏向已知类别，并且在不同新类别的类别混淆，从而妨碍模型有效识别新类别。为应对这些挑战，我们提出了一种名为自我偏差校准（Self-Debiasing Calibration, SDC）的新框架。与以往方法将模型对已知类别的偏好视为新类别识别的障碍不同，SDC 提供了一种新颖的视角，即将这种偏好视为促进新类别学习的潜在资源。具体而言，偏差模型的输出有两个关键用途。首先，它提供了一个关于类别偏差的精确建模，可以用于衡量当前训练模型的偏差程度并校准其输出。其次，它为区分不同类型的新类别提供了有价值的见解，通过相似类别之间的知识迁移来实现这一目的。基于这些见解，SDC 动态调整当前训练模型的输出逻辑，利用偏差模型的输出。这种方法生产出偏差较小的逻辑输出，以有效地解决类别偏向已知类别的问题，并为未标记数据生成更准确的伪标签，从而减轻新类别之间的类别混淆。在三个基准数据集上的实验表明，SDC 在新类别识别方面优于当前最佳方法。我们的代码和数据可通过 \url{此处提供链接} 获取。 

---
# Graph Learning in the Era of LLMs: A Survey from the Perspective of Data, Models, and Tasks 

**Title (ZH)**: 在大规模语言模型时代的数据、模型与任务视角下的图学习综述 

**Authors**: Xunkai Li, Zhengyu Wu, Jiayi Wu, Hanwen Cui, Jishuo Jia, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12456)  

**Abstract**: With the increasing prevalence of cross-domain Text-Attributed Graph (TAG) Data (e.g., citation networks, recommendation systems, social networks, and ai4science), the integration of Graph Neural Networks (GNNs) and Large Language Models (LLMs) into a unified Model architecture (e.g., LLM as enhancer, LLM as collaborators, LLM as predictor) has emerged as a promising technological paradigm. The core of this new graph learning paradigm lies in the synergistic combination of GNNs' ability to capture complex structural relationships and LLMs' proficiency in understanding informative contexts from the rich textual descriptions of graphs. Therefore, we can leverage graph description texts with rich semantic context to fundamentally enhance Data quality, thereby improving the representational capacity of model-centric approaches in line with data-centric machine learning principles. By leveraging the strengths of these distinct neural network architectures, this integrated approach addresses a wide range of TAG-based Task (e.g., graph learning, graph reasoning, and graph question answering), particularly in complex industrial scenarios (e.g., supervised, few-shot, and zero-shot settings). In other words, we can treat text as a medium to enable cross-domain generalization of graph learning Model, allowing a single graph model to effectively handle the diversity of downstream graph-based Task across different data domains. This work serves as a foundational reference for researchers and practitioners looking to advance graph learning methodologies in the rapidly evolving landscape of LLM. We consistently maintain the related open-source materials at \url{this https URL}. 

**Abstract (ZH)**: 随着跨域文本关联图（TAG）数据（例如引用网络、推荐系统、社会网络和ai4science）的日益普及，将图神经网络（GNNs）和大型语言模型（LLMs）统一集成到一个模型架构中（例如，LLM作为增强器、LLM作为合作者、LLM作为预测器）已成为一种有前途的技术范式。这种新的图学习范式的核心在于GNNs捕捉复杂结构关系的能力与LLMs从丰富的图文本描述中理解信息性上下文的能力互补结合。因此，我们可以利用具有丰富语义上下文的图描述文本来从根本上提高数据质量，从而根据数据驱动的机器学习原则提升以模型为中心的方法的表现能力。通过利用这些不同神经网络架构的优势，这种整合方法可以应对一系列基于TAG的任务（例如，图学习、图推理和图问答），特别是在复杂的工业场景中（例如，监督场景、少量样本场景和零样本场景）。换句话说，我们可以通过将文本作为一种媒介，实现图学习模型在不同数据域的下游图任务的跨域泛化能力，使单一的图模型能够有效处理多样性。本文为研究者和从业者在快速发展的LLM背景下推进图学习方法提供了基础参考。我们持续维护相关的开源材料在 \url{this https URL}。 

---
# PERC: Plan-As-Query Example Retrieval for Underrepresented Code Generation 

**Title (ZH)**: PERC：计划作为查询的代码生成中未充分代表代码的示例检索 

**Authors**: Jaeseok Yoo, Hojae Han, Youngwon Lee, Jaejin Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12447)  

**Abstract**: Code generation with large language models has shown significant promise, especially when employing retrieval-augmented generation (RAG) with few-shot examples. However, selecting effective examples that enhance generation quality remains a challenging task, particularly when the target programming language (PL) is underrepresented. In this study, we present two key findings: (1) retrieving examples whose presented algorithmic plans can be referenced for generating the desired behavior significantly improves generation accuracy, and (2) converting code into pseudocode effectively captures such algorithmic plans, enhancing retrieval quality even when the source and the target PLs are different. Based on these findings, we propose Plan-as-query Example Retrieval for few-shot prompting in Code generation (PERC), a novel framework that utilizes algorithmic plans to identify and retrieve effective examples. We validate the effectiveness of PERC through extensive experiments on the CodeContests, HumanEval and MultiPL-E benchmarks: PERC consistently outperforms the state-of-the-art RAG methods in code generation, both when the source and target programming languages match or differ, highlighting its adaptability and robustness in diverse coding environments. 

**Abstract (ZH)**: 大型语言模型进行代码生成展现了显著的潜力，尤其是在结合检索增强生成（RAG）和少量示例的情况下。然而，选择能够有效提升生成质量的示例仍然是一项具挑战性的任务，特别是在目标编程语言（PL）资源较少的情况下。在本研究中，我们提出了两项关键发现：（1）检索可用于生成所需行为的算法计划的示例，显著提高了生成的准确性；（2）将代码转换为伪代码能够有效地捕捉这些算法计划，即使源编程语言和目标编程语言不同，也能提高检索质量。基于这些发现，我们提出了一种新的框架——代码生成中利用算法计划进行少量提示的计划查询示例检索（PERC），该框架通过使用算法计划来识别和检索有效的示例。我们通过在CodeContests、HumanEval和MultiPL-E基准上的广泛实验验证了PERC的有效性：无论源编程语言和目标编程语言是否匹配，PERC在代码生成中始终优于最先进的RAG方法，这突显了其在各种编程环境中的适应性和鲁棒性。 

---
# Visual Instruction Tuning with 500x Fewer Parameters through Modality Linear Representation-Steering 

**Title (ZH)**: 通过模态线性表示导向的视觉指令调优，使用500倍 fewer 参数 

**Authors**: Jinhe Bi, Yujun Wang, Haokun Chen, Xun Xiao, Artur Hecker, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2412.12359)  

**Abstract**: Multimodal Large Language Models (MLLMs) have significantly advanced visual tasks by integrating visual representations into large language models (LLMs). The textual modality, inherited from LLMs, equips MLLMs with abilities like instruction following and in-context learning. In contrast, the visual modality enhances performance in downstream tasks by leveraging rich semantic content, spatial information, and grounding capabilities. These intrinsic modalities work synergistically across various visual tasks. Our research initially reveals a persistent imbalance between these modalities, with text often dominating output generation during visual instruction tuning. This imbalance occurs when using both full fine-tuning and parameter-efficient fine-tuning (PEFT) methods. We then found that re-balancing these modalities can significantly reduce the number of trainable parameters required, inspiring a direction for further optimizing visual instruction tuning. We introduce Modality Linear Representation-Steering (MoReS) to achieve the goal. MoReS effectively re-balances the intrinsic modalities throughout the model, where the key idea is to steer visual representations through linear transformations in the visual subspace across each model layer. To validate our solution, we composed LLaVA Steering, a suite of models integrated with the proposed MoReS method. Evaluation results show that the composed LLaVA Steering models require, on average, 500 times fewer trainable parameters than LoRA needs while still achieving comparable performance across three visual benchmarks and eight visual question-answering tasks. Last, we present the LLaVA Steering Factory, an in-house developed platform that enables researchers to quickly customize various MLLMs with component-based architecture for seamlessly integrating state-of-the-art models, and evaluate their intrinsic modality imbalance. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过将视觉表示融入大型语言模型（LLMs）中，显著推进了视觉任务。文本模态继承自LLMs，赋予MLLMs指令跟随和上下文学习的能力。相比之下，视觉模态通过利用丰富的语义内容、空间信息和接地能力，在下游任务中提升了性能。这些内在模态在各种视觉任务中协同工作。我们的研究最初揭示了这些模态之间存在持续的不平衡，尤其是在视觉指令调优过程中，文本通常占据主导地位生成输出。这种不平衡在使用完整调优和参数高效调优（PEFT）方法时都会出现。我们随后发现，重新平衡这些模态可以显著减少所需的可训练参数数量，启发了进一步优化视觉指令调优的方向。为此，我们引入了模态线性表示引导（MoReS）方法以实现这一目标。MoReS有效地在整个模型中重新平衡内在模态，核心思想是通过每个模型层中视觉子空间的线性变换来引导视觉表示。为了验证我们的解决方案，我们构建了LLaVA Steering，该套件模型集成了提出的MoReS方法。评估结果显示，与LoRA相比，组成LLaVA Steering模型平均所需的可训练参数减少了500倍，同时在三个视觉基准和八个视觉问答任务中依然取得了可比的性能。最后，我们介绍了LLaVA Steering Factory，这是一种内部开发的平台，使研究人员能够快速通过组件化架构定制各种MLLMs，并对它们的内在模态不平衡进行评估。 

---
# RAG Playground: A Framework for Systematic Evaluation of Retrieval Strategies and Prompt Engineering in RAG Systems 

**Title (ZH)**: RAG游乐场：一种用于系统评估RAG系统中检索策略和提示工程的框架 

**Authors**: Ioannis Papadimitriou, Ilias Gialampoukidis, Stefanos Vrochidis, Ioannis, Kompatsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2412.12322)  

**Abstract**: We present RAG Playground, an open-source framework for systematic evaluation of Retrieval-Augmented Generation (RAG) systems. The framework implements and compares three retrieval approaches: naive vector search, reranking, and hybrid vector-keyword search, combined with ReAct agents using different prompting strategies. We introduce a comprehensive evaluation framework with novel metrics and provide empirical results comparing different language models (Llama 3.1 and Qwen 2.5) across various retrieval configurations. Our experiments demonstrate significant performance improvements through hybrid search methods and structured self-evaluation prompting, achieving up to 72.7% pass rate on our multi-metric evaluation framework. The results also highlight the importance of prompt engineering in RAG systems, with our custom-prompted agents showing consistent improvements in retrieval accuracy and response quality. 

**Abstract (ZH)**: 我们介绍了RAG Playground，这是一个开源框架，用于系统性评估检索增强生成（RAG）系统。该框架实现了并比较了三种检索方法：朴素向量搜索、再排序以及结合ReAct代理的向量-关键词搜索，并使用不同的提示策略进行比较。我们提出了一种全面的评估框架并引入了新型度量标准，提供了不同语言模型（Llama 3.1和Qwen 2.5）在各种检索配置下的实证结果比较。我们的实验表明，通过混合搜索方法和结构化自我评估提示可以显著提高性能，在我们的多指标评估框架下达到72.7%的通过率。结果还强调了在RAG系统中提示工程的重要性，我们的自定义提示代理在检索精度和响应质量上保持了一致的改进。 

---
# DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis 

**Title (ZH)**: DLF：解耦语言焦点的多模态情感分析 

**Authors**: Pan Wang, Qiang Zhou, Yawen Wu, Tianlong Chen, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12225)  

**Abstract**: Multimodal Sentiment Analysis (MSA) leverages heterogeneous modalities, such as language, vision, and audio, to enhance the understanding of human sentiment. While existing models often focus on extracting shared information across modalities or directly fusing heterogeneous modalities, such approaches can introduce redundancy and conflicts due to equal treatment of all modalities and the mutual transfer of information between modality pairs. To address these issues, we propose a Disentangled-Language-Focused (DLF) multimodal representation learning framework, which incorporates a feature disentanglement module to separate modality-shared and modality-specific information. To further reduce redundancy and enhance language-targeted features, four geometric measures are introduced to refine the disentanglement process. A Language-Focused Attractor (LFA) is further developed to strengthen language representation by leveraging complementary modality-specific information through a language-guided cross-attention mechanism. The framework also employs hierarchical predictions to improve overall accuracy. Extensive experiments on two popular MSA datasets, CMU-MOSI and CMU-MOSEI, demonstrate the significant performance gains achieved by the proposed DLF framework. Comprehensive ablation studies further validate the effectiveness of the feature disentanglement module, language-focused attractor, and hierarchical predictions. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态情感分析（MSA）利用语言、视觉和音频等多种异构模态来增强对人类情感的理解。现有模型通常侧重于提取不同模态之间的共享信息，或将不同的模态直接融合，但这些方法由于平等对待所有模态以及模态对之间的相互信息传递，可能会引入冗余和冲突。为了解决这些问题，我们提出了一种名为Disentangled-Language-Focused (DLF) 的多模态表示学习框架，该框架引入了一个特征解缠模块，用于分离模态共享信息和模态特定信息。为了进一步减少冗余并增强语言导向特征，我们引入了四种几何度量来精化解缠过程。进一步开发了语言聚焦吸引子（LFA），通过语言引导的交叉注意力机制利用互补的模态特定信息来增强语言表示。该框架还采用了分层预测以提高整体准确率。在两个流行的MSA数据集CMU-MOSI和CMU-MOSEI上的广泛实验表明，所提出的DLF框架实现了显著的性能提升。全面的消融研究进一步验证了特征解缠模块、语言聚焦吸引子和分层预测的有效性。我们的代码可在以下链接获取：这个 https URL。 

---
# Finding a Wolf in Sheep's Clothing: Combating Adversarial Text-To-Image Prompts with Text Summarization 

**Title (ZH)**: 披羊皮的狼：利用文本总结对抗 adversarial 文本到图像提示 

**Authors**: Portia Cooper, Harshita Narnoli, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12212)  

**Abstract**: Text-to-image models are vulnerable to the stepwise "Divide-and-Conquer Attack" (DACA) that utilize a large language model to obfuscate inappropriate content in prompts by wrapping sensitive text in a benign narrative. To mitigate stepwise DACA attacks, we propose a two-layer method involving text summarization followed by binary classification. We assembled the Adversarial Text-to-Image Prompt (ATTIP) dataset ($N=940$), which contained DACA-obfuscated and non-obfuscated prompts. From the ATTIP dataset, we created two summarized versions: one generated by a small encoder model and the other by a large language model. Then, we used an encoder classifier and a GPT-4o classifier to perform content moderation on the summarized and unsummarized prompts. When compared with a classifier that operated over the unsummarized data, our method improved F1 score performance by 31%. Further, the highest recorded F1 score achieved (98%) was produced by the encoder classifier on a summarized ATTIP variant. This study indicates that pre-classification text summarization can inoculate content detection models against stepwise DACA obfuscations. 

**Abstract (ZH)**: 文本到图像模型对逐步的“分而治之攻击”（Divide-and-Conquer Attack, DACA）非常脆弱，这种攻击利用大型语言模型通过在良性叙述中包裹敏感文本来模糊提示中的不当内容。为应对逐步的DACA攻击，我们提出了一种两层方法，包括文本总结，随后进行二分类。我们构建了Adversarial Text-to-Image Prompt (ATTIP)数据集（包含940个样本），其中包含DACA模糊和未模糊的提示。从ATTIP数据集中，我们创建了两个总结版本：一个由小型编码器模型生成，另一个由大型语言模型生成。然后，我们使用编码器分类器和GPT-4o分类器对总结和未总结的提示进行内容审核。与直接在原始数据上工作的分类器相比，我们的方法在F1分数上提升了31%。此外，编码器分类器在总结版本的ATTIP数据集中达到的最高F1分数为98%。这项研究表明，预先分类的文本总结可以抵御逐步的DACA模糊，从而保护内容检测模型。 

---
# SEE: Sememe Entanglement Encoding for Transformer-bases Models Compression 

**Title (ZH)**: SEE：基于变压器模型的语义纠缠编码压缩方法 

**Authors**: Jing Zhang, Shuzhen Sun, Peng Zhang, Guangxing Cao, Hui Gao, Xindian Ma, Nan Xu, Yuexian Hou  

**Link**: [PDF](https://arxiv.org/pdf/2412.12204)  

**Abstract**: Transformer-based large language models exhibit groundbreaking capabilities, but their storage and computational costs are prohibitively high, limiting their application in resource-constrained scenarios. An effective approach is to eliminate redundant model parameters and computational costs while incorporating efficient expert-derived knowledge structures to achieve a balance between compression and performance. Therefore, we propose the \textit{Sememe Entanglement Encoding (SEE)} algorithm. Guided by expert prior knowledge, the model is compressed through the low-rank approximation idea. In Entanglement Embedding, basic semantic units such as sememes are represented as low-dimensional vectors, and then reconstructed into high-dimensional word embeddings through the combination of generalized quantum entanglement. We adapt the Sememe Entanglement Encoding algorithm to transformer-based models of different magnitudes. Experimental results indicate that our approach achieves stable performance while compressing model parameters and computational costs. 

**Abstract (ZH)**: 基于Transformer的大语言模型展现出革命性的能力，但其存储和计算成本过高，限制了其在资源受限场景中的应用。一种有效的方法是在引入高效专家知识结构的同时，消除冗余的模型参数和计算成本，以实现压缩与性能之间的平衡。因此，我们提出了\[词义纠缠编码（SEE, Sememe Entanglement Encoding）\]算法。该算法在专家先验知识的指导下，通过低秩近似的思想对模型进行压缩。在纠缠嵌入中，将基本语义单元（如词义）表示为低维向量，然后通过广义量子纠缠的结合重构为高维词嵌入。我们适应不同规模的基于Transformer的模型使用词义纠缠编码算法。实验结果表明，我们的方法能够在压缩模型参数和计算成本的同时保持稳定的性能。 

---
# Explore Theory of Mind: Program-guided adversarial data generation for theory of mind reasoning 

**Title (ZH)**: 探索心智理论：程序引导的对抗性数据生成用于心智理论推理 

**Authors**: Melanie Sclar, Jane Yu, Maryam Fazel-Zarandi, Yulia Tsvetkov, Yonatan Bisk, Yejin Choi, Asli Celikyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2412.12175)  

**Abstract**: Do large language models (LLMs) have theory of mind? A plethora of papers and benchmarks have been introduced to evaluate if current models have been able to develop this key ability of social intelligence. However, all rely on limited datasets with simple patterns that can potentially lead to problematic blind spots in evaluation and an overestimation of model capabilities. We introduce ExploreToM, the first framework to allow large-scale generation of diverse and challenging theory of mind data for robust training and evaluation. Our approach leverages an A* search over a custom domain-specific language to produce complex story structures and novel, diverse, yet plausible scenarios to stress test the limits of LLMs. Our evaluation reveals that state-of-the-art LLMs, such as Llama-3.1-70B and GPT-4o, show accuracies as low as 0% and 9% on ExploreToM-generated data, highlighting the need for more robust theory of mind evaluation. As our generations are a conceptual superset of prior work, fine-tuning on our data yields a 27-point accuracy improvement on the classic ToMi benchmark (Le et al., 2019). ExploreToM also enables uncovering underlying skills and factors missing for models to show theory of mind, such as unreliable state tracking or data imbalances, which may contribute to models' poor performance on benchmarks. 

**Abstract (ZH)**: 大语言模型（LLMs）具有心智理论能力吗？大量论文和基准测试已经提出，用于评估当前模型是否能够发展这种关键的社会智能能力。然而，这些评估方法均依赖于有限数据集和简单的模式，可能导致评价中的潜在盲点和对模型能力的高估。我们引入了ExploreToM，这是第一个允许生成多样且具有挑战性的理论认知数据的框架，用于稳健的训练和评估。我们的方法通过在自定义领域特定语言中使用A*搜索来生成复杂的故事情节和新颖、多样但又可信的情景，以最大限度地测试大语言模型的能力极限。评估结果显示，现有最先进的LLM模型（如Llama-3.1-70B和GPT-4o）在ExploreToM生成的数据上的准确率低至0%和9%，突显了更稳健理论认知评估的需求。由于我们生成的数据覆盖了先前工作的概念超集，基于我们数据的微调在经典的ToMi基准测试（Le et al., 2019）上提高了27个百分点的准确率。此外，ExploreToM还能够揭示模型展示心智理论所需的关键技能和缺失因素，例如不可靠的状态跟踪或数据失衡，这可能是模型在基准测试中表现不佳的原因之一。 

---
# PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection 

**Title (ZH)**: PyOD 2：一种基于大语言模型辅助模型选择的异常检测Python库 

**Authors**: Sihan Chen, Zhuangzhuang Qian, Wingchun Siu, Xingcan Hu, Jiaqi Li, Shawn Li, Yuehan Qin, Tiankai Yang, Zhuo Xiao, Wanghao Ye, Yichi Zhang, Yushun Dong, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.12154)  

**Abstract**: Outlier detection (OD), also known as anomaly detection, is a critical machine learning (ML) task with applications in fraud detection, network intrusion detection, clickstream analysis, recommendation systems, and social network moderation. Among open-source libraries for outlier detection, the Python Outlier Detection (PyOD) library is the most widely adopted, with over 8,500 GitHub stars, 25 million downloads, and diverse industry usage. However, PyOD currently faces three limitations: (1) insufficient coverage of modern deep learning algorithms, (2) fragmented implementations across PyTorch and TensorFlow, and (3) no automated model selection, making it hard for non-experts.
To address these issues, we present PyOD Version 2 (PyOD 2), which integrates 12 state-of-the-art deep learning models into a unified PyTorch framework and introduces a large language model (LLM)-based pipeline for automated OD model selection. These improvements simplify OD workflows, provide access to 45 algorithms, and deliver robust performance on various datasets. In this paper, we demonstrate how PyOD 2 streamlines the deployment and automation of OD models and sets a new standard in both research and industry. PyOD 2 is accessible at [this https URL](this https URL). This study aligns with the Web Mining and Content Analysis track, addressing topics such as the robustness of Web mining methods and the quality of algorithmically-generated Web data. 

**Abstract (ZH)**: 异常检测（OD），也称为离群点检测，是机器学习（ML）中一个关键的任务，广泛应用于欺诈检测、网络安全入侵检测、点击流分析、推荐系统和社交网络管理等领域。在开源异常检测库中，Python 异常检测库（PyOD）是最广泛采用的，拥有超过8500颗GitHub之星、2.5亿次下载量，并且在各行各业都有广泛的应用。然而，PyOD目前面临三个局限性：（1）缺乏对现代深度学习算法的充分覆盖，（2）在PyTorch和TensorFlow中的实现碎片化，以及（3）缺乏自动模型选择功能，这使得非专家难以使用。

为了解决这些问题，我们提出了PyOD Version 2（PyOD 2），它将12种先进的深度学习模型集成到了统一的PyTorch框架中，并引入了基于大型语言模型（LLM）的流程以实现自动OD模型选择。这些改进简化了OD工作流程，提供了45种算法的访问途径，并在多种数据集上实现了可靠的性能。在本文中，我们展示了PyOD 2如何简化和自动化OD模型的部署，并树立了研究和工业领域的标准。PyOD 2的代码库可访问于[该链接](该链接)。本文的研究内容与Web挖掘和内容分析类别相契合，涵盖了Web挖掘方法的稳健性和算法生成的Web数据质量等相关主题。 

---
# How to Choose a Threshold for an Evaluation Metric for Large Language Models 

**Title (ZH)**: 如何为大型语言模型的评估指标选择阈值 

**Authors**: Bhaskarjit Sarmah, Mingshu Li, Jingrao Lyu, Sebastian Frank, Nathalia Castellanos, Stefano Pasquali, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2412.12148)  

**Abstract**: To ensure and monitor large language models (LLMs) reliably, various evaluation metrics have been proposed in the literature. However, there is little research on prescribing a methodology to identify a robust threshold on these metrics even though there are many serious implications of an incorrect choice of the thresholds during deployment of the LLMs. Translating the traditional model risk management (MRM) guidelines within regulated industries such as the financial industry, we propose a step-by-step recipe for picking a threshold for a given LLM evaluation metric. We emphasize that such a methodology should start with identifying the risks of the LLM application under consideration and risk tolerance of the stakeholders. We then propose concrete and statistically rigorous procedures to determine a threshold for the given LLM evaluation metric using available ground-truth data. As a concrete example to demonstrate the proposed methodology at work, we employ it on the Faithfulness metric, as implemented in various publicly available libraries, using the publicly available HaluBench dataset. We also lay a foundation for creating systematic approaches to select thresholds, not only for LLMs but for any GenAI applications. 

**Abstract (ZH)**: 为了确保并监测大型语言模型（LLMs）的可靠性，文献中已经提出了各种评估指标。然而，尽管在部署LLMs过程中错误选择阈值可能带来许多严重后果，但关于如何制定一种方法以确定这些指标的稳健阈值的研究却很少。我们将传统模型风险管理（MRM）指南应用于受监管行业（如金融行业），提出了一种逐步方法来选择给定LLM评估指标的阈值。我们强调这种方法应以识别考虑中的LLM应用的风险及其利益相关者的风险容忍度为起点。然后，我们提出了具体且经过统计验证的程序，以利用可用的真实数据确定给定LLM评估指标的阈值。作为提出的方法的应用示例，我们使用了公开可用的HaluBench数据集来评估Faithfulness指标，该指标在各种公开可用的库中实现。我们还为选择阈值奠定了基础，不仅适用于LLMs，还适用于任何生成式AI（GenAI）应用。 

---
# NLLG Quarterly arXiv Report 09/24: What are the most influential current AI Papers? 

**Title (ZH)**: NLLG 季度 arXiv 报告（2023年9月24日）：哪些是当前最具影响力的AI论文？ 

**Authors**: Christoph Leiter, Jonas Belouadi, Yanran Chen, Ran Zhang, Daniil Larionov, Aida Kostikova, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2412.12121)  

**Abstract**: The NLLG (Natural Language Learning & Generation) arXiv reports assist in navigating the rapidly evolving landscape of NLP and AI research across cs.CL, cs.CV, cs.AI, and cs.LG categories. This fourth installment captures a transformative period in AI history - from January 1, 2023, following ChatGPT's debut, through September 30, 2024. Our analysis reveals substantial new developments in the field - with 45% of the top 40 most-cited papers being new entries since our last report eight months ago and offers insights into emerging trends and major breakthroughs, such as novel multimodal architectures, including diffusion and state space models. Natural Language Processing (NLP; cs.CL) remains the dominant main category in the list of our top-40 papers but its dominance is on the decline in favor of Computer vision (cs.CV) and general machine learning (cs.LG). This report also presents novel findings on the integration of generative AI in academic writing, documenting its increasing adoption since 2022 while revealing an intriguing pattern: top-cited papers show notably fewer markers of AI-generated content compared to random samples. Furthermore, we track the evolution of AI-associated language, identifying declining trends in previously common indicators such as "delve". 

**Abstract (ZH)**: 自然语言学习与生成（Natural Language Learning & Generation, NLLG）方面的arXiv报告有助于导航NLP和AI研究的快速演变 landscape。这些报告涵盖了cs.CL、cs.CV、cs.AI和cs.LG类别中的最新进展。本文是第四期报告，记录了从2023年1月1日至2024年9月30日的转变时期——这一时期以ChatGPT的发布为起点。我们的分析揭示了领域内的重大新发展——在前40篇被引用最多的论文中，45%的论文是自上次报告八个月前的新条目，并提供了关于新兴趋势和重大突破的见解，例如新颖的多模态架构，包括扩散模型和状态空间模型。自然语言处理（NLP；cs.CL）仍然是我们前40篇论文中的主导类别，但其主导地位正在被计算机视觉（cs.CV）和一般机器学习（cs.LG）所取代。报告还展示了关于生成式AI在学术写作中集成的新型发现，记录了自2022年以来其采用率的增加，同时揭示了一个有趣的现象：顶级被引用的论文中表现出AI生成内容的标记明显少于随机样本。此外，报告还追踪了与AI相关的语言演变，发现了以前常见的指标如“delve”使用频率下降的趋势。 

---
# Mastering Board Games by External and Internal Planning with Language Models 

**Title (ZH)**: 通过语言模型进行外部和内部规划掌握棋盘游戏 

**Authors**: John Schultz, Jakub Adamek, Matej Jusup, Marc Lanctot, Michael Kaisers, Sarah Perrin, Daniel Hennes, Jeremy Shar, Cannada Lewis, Anian Ruoss, Tom Zahavy, Petar Veličković, Laurel Prince, Satinder Singh, Eric Malmi, Nenad Tomašev  

**Link**: [PDF](https://arxiv.org/pdf/2412.12119)  

**Abstract**: While large language models perform well on a range of complex tasks (e.g., text generation, question answering, summarization), robust multi-step planning and reasoning remains a considerable challenge for them. In this paper we show that search-based planning can significantly improve LLMs' playing strength across several board games (Chess, Fischer Random / Chess960, Connect Four, and Hex). We introduce, compare and contrast two major approaches: In external search, the model guides Monte Carlo Tree Search (MCTS) rollouts and evaluations without calls to an external engine, and in internal search, the model directly generates in-context a linearized tree of potential futures and a resulting final choice. Both build on a language model pre-trained on relevant domain knowledge, capturing the transition and value functions across these games. We find that our pre-training method minimizes hallucinations, as our model is highly accurate regarding state prediction and legal moves. Additionally, both internal and external search indeed improve win-rates against state-of-the-art bots, even reaching Grandmaster-level performance in chess while operating on a similar move count search budget per decision as human Grandmasters. The way we combine search with domain knowledge is not specific to board games, suggesting direct extensions into more general language model inference and training techniques. 

**Abstract (ZH)**: 尽管大规模语言模型在多种复杂任务（如文本生成、问答、总结）上表现出色，但它们在稳健的多步规划和推理方面仍面临相当大的挑战。本文展示了基于搜索的规划可以显著提高大规模语言模型在几种棋盘游戏（国际象棋、福什随机国际象棋/棋960、四连珠和六子棋）中的表现。我们引入了两种主要方法并进行了比较：在外存搜索中，模型指导蒙特卡洛树搜索（MCTS）的展开和评估，而不调用外部引擎；而在内存搜索中，模型直接生成上下文相关的潜在未来树和最终选择。这两种方法都基于预训练在相关领域知识上的语言模型，捕捉这些游戏中状态转移和价值函数的变化。我们发现我们的预训练方法最大限度地减少了幻觉，因为我们的模型在状态预测和合法移动方面表现非常准确。此外，无论是外存搜索还是内存搜索都确实提高了与最先进的机器人对战胜率，甚至在决策时的移动搜索预算与人类大师相似的情况下，达到国际象棋的大师水平。我们结合搜索和领域知识的方式并不局限于棋盘游戏，这表明可以直接将其扩展到更通用的语言模型推理和训练技术中。 

---
# Voice Biomarker Analysis and Automated Severity Classification of Dysarthric Speech in a Multilingual Context 

**Title (ZH)**: 多语言背景下言语障碍语音生物标志物分析与自动严重程度分类 

**Authors**: Eunjung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2412.12111)  

**Abstract**: Dysarthria, a motor speech disorder, severely impacts voice quality, pronunciation, and prosody, leading to diminished speech intelligibility and reduced quality of life. Accurate assessment is crucial for effective treatment, but traditional perceptual assessments are limited by their subjectivity and resource intensity. To mitigate the limitations, automatic dysarthric speech assessment methods have been proposed to support clinicians on their decision-making. While these methods have shown promising results, most research has focused on monolingual environments. However, multilingual approaches are necessary to address the global burden of dysarthria and ensure equitable access to accurate diagnosis. This thesis proposes a novel multilingual dysarthria severity classification method, by analyzing three languages: English, Korean, and Tamil. 

**Abstract (ZH)**: 构音障碍是一种运动性语言障碍，严重影响语音质量、发音和语调，导致语音清晰度降低和生活质量下降。准确的评估对于有效的治疗至关重要，但传统的主观评估方法受限于其主观性和资源消耗。为缓解这些限制，已经提出了自动构音障碍评估方法以支持临床决策。虽然这些方法已经显示出积极的结果，但大多数研究主要集中在单一语言环境中。然而，多语言方法是必要的，以便应对全球构音障碍的负担，并确保获得准确诊断的公平准入。本论文提出了一种新型多语言构音障碍严重程度分类方法，通过分析三种语言：英语、韩语和泰米尔语。 

---
