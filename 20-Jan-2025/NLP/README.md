# BoK: Introducing Bag-of-Keywords Loss for Interpretable Dialogue Response Generation 

**Title (ZH)**: BoK: 引入基于词袋的损失函数以生成可解释的对话响应 

**Authors**: Suvodip Dey, Maunendra Sankar Desarkar  

**Link**: [PDF](https://arxiv.org/pdf/2501.10328)  

**Abstract**: The standard language modeling (LM) loss by itself has been shown to be inadequate for effective dialogue modeling. As a result, various training approaches, such as auxiliary loss functions and leveraging human feedback, are being adopted to enrich open-domain dialogue systems. One such auxiliary loss function is Bag-of-Words (BoW) loss, defined as the cross-entropy loss for predicting all the words/tokens of the next utterance. In this work, we propose a novel auxiliary loss named Bag-of-Keywords (BoK) loss to capture the central thought of the response through keyword prediction and leverage it to enhance the generation of meaningful and interpretable responses in open-domain dialogue systems. BoK loss upgrades the BoW loss by predicting only the keywords or critical words/tokens of the next utterance, intending to estimate the core idea rather than the entire response. We incorporate BoK loss in both encoder-decoder (T5) and decoder-only (DialoGPT) architecture and train the models to minimize the weighted sum of BoK and LM (BoK-LM) loss. We perform our experiments on two popular open-domain dialogue datasets, DailyDialog and Persona-Chat. We show that the inclusion of BoK loss improves the dialogue generation of backbone models while also enabling post-hoc interpretability. We also study the effectiveness of BoK-LM loss as a reference-free metric and observe comparable performance to the state-of-the-art metrics on various dialogue evaluation datasets. 

**Abstract (ZH)**: 标准语言模型（LM）损失本身已被证明在有效的对话建模中不够充分。因此，为了丰富开放域对话系统，各种训练方法，如辅助损失函数和利用人类反馈等，被采用。其中一种辅助损失函数是词汇袋（BoW）损失，定义为预测下一个片段中所有单词/标记的交叉熵损失。在本项工作中，我们提出了一个新的辅助损失函数——关键词袋（BoK）损失，通过关键词预测来捕捉响应的核心思想，并利用这一点来增强开放域对话系统生成有意义且可解释的响应。BoK损失通过仅预测下一个片段中的关键词或关键单词/标记来升级BoW损失，旨在估计核心思想而不是整个响应。我们在编码器-解码器（T5）和解码器独立试用（DialoGPT）架构中引入BoK损失，并训练模型以最小化BoK和LM（BoK-LM）损失的加权和。我们在两个流行的开放域对话数据集DailyDialog和Persona-Chat上进行了实验。结果显示，引入BoK损失可以提高骨干模型的对话生成能力，并且能够增强后续的可解释性。我们还研究了BoK-LM损失作为参考无关度量的有效性，并在各种对话评估数据集中观察到与最新度量相当的性能。 

---
# Hierarchical Autoregressive Transformers: Combining Byte-~and Word-Level Processing for Robust, Adaptable Language Models 

**Title (ZH)**: 层次自回归变换器：结合字节级和词级处理以构建稳健且 adaptable 的语言模型 

**Authors**: Pit Neitemeier, Björn Deiseroth, Constantin Eichenberg, Lukas Balles  

**Link**: [PDF](https://arxiv.org/pdf/2501.10322)  

**Abstract**: Tokenization is a fundamental step in natural language processing, breaking text into units that computational models can process. While learned subword tokenizers have become the de-facto standard, they present challenges such as large vocabularies, limited adaptability to new domains or languages, and sensitivity to spelling errors and variations. To overcome these limitations, we investigate a hierarchical architecture for autoregressive language modelling that combines character-level and word-level processing. It employs a lightweight character-level encoder to convert character sequences into word embeddings, which are then processed by a word-level backbone model and decoded back into characters via a compact character-level decoder. This method retains the sequence compression benefits of word-level tokenization without relying on a rigid, predefined vocabulary. We demonstrate, at scales up to 7 billion parameters, that hierarchical transformers match the downstream task performance of subword-tokenizer-based models while exhibiting significantly greater robustness to input perturbations. Additionally, during continued pretraining on an out-of-domain language, our model trains almost twice as fast, achieves superior performance on the target language, and retains more of its previously learned knowledge. Hierarchical transformers pave the way for NLP systems that are more robust, flexible, and generalizable across languages and domains. 

**Abstract (ZH)**: 分词是自然语言处理中的一个基本步骤，它将文本划分为计算模型可以处理的单位。虽然学习子词分词器已成为事实上的标准，但它们也面临着一些挑战，比如词汇量过大、对新领域或新语言适应性有限，以及对拼写错误和变异敏感。为了解决这些限制，我们研究了一种结合字符级和词级处理的分层架构，用于自回归语言建模。该架构采用一个轻量级的字符级编码器将字符序列转换为词嵌入，这些词嵌入随后由词级骨干模型处理，并通过紧凑的字符级解码器重新编码回字符。这种方法保留了词级分词的序列压缩优势，无需依赖于固定的词汇表。我们证明，在多达70亿参数的规模下，分层变压器在下游任务上的性能与基于子词分词器的模型相当，同时在输入扰动方面表现出显著的鲁棒性。此外，在跨领域语言继续预训练时，我们的模型训练速度几乎快一倍，对目标语言的性能表现更优，并且保留了更多的先前学习知识。分层变压器为跨语言和领域的更稳健、灵活和泛化的NLP系统铺平了道路。 

---
# Natural Language Processing of Privacy Policies: A Survey 

**Title (ZH)**: 隐私政策的自然语言处理：一种综述 

**Authors**: Andrick Adhikari, Sanchari Das, Rinku Dewri  

**Link**: [PDF](https://arxiv.org/pdf/2501.10319)  

**Abstract**: Natural Language Processing (NLP) is an essential subset of artificial intelligence. It has become effective in several domains, such as healthcare, finance, and media, to identify perceptions, opinions, and misuse, among others. Privacy is no exception, and initiatives have been taken to address the challenges of usable privacy notifications to users with the help of NLP. To this aid, we conduct a literature review by analyzing 109 papers at the intersection of NLP and privacy policies. First, we provide a brief introduction to privacy policies and discuss various facets of associated problems, which necessitate the application of NLP to elevate the current state of privacy notices and disclosures to users. Subsequently, we a) provide an overview of the implementation and effectiveness of NLP approaches for better privacy policy communication; b) identify the methodologies that can be further enhanced to provide robust privacy policies; and c) identify the gaps in the current state-of-the-art research. Our systematic analysis reveals that several research papers focus on annotating and classifying privacy texts for analysis but need to adequately dwell on other aspects of NLP applications, such as summarization. More specifically, ample research opportunities exist in this domain, covering aspects such as corpus generation, summarization vectors, contextualized word embedding, identification of privacy-relevant statement categories, fine-grained classification, and domain-specific model tuning. 

**Abstract (ZH)**: 自然语言处理（NLP）是人工智能的重要子领域之一。它已在医疗、金融和媒体等多个领域中有效应用，用于识别感知、意见和不当行为等。隐私问题也不例外，已经采取了措施利用NLP来解决使用户能够理解的隐私通知问题。为此，我们通过分析109篇关于NLP和隐私政策交叉领域的论文进行了文献综述。首先，我们将简要介绍隐私政策，并讨论与之相关的各种问题，这些问题是促使NLP应用于提升当前隐私通知和披露状态的必要条件。随后，我们(a) 提供了NLP方法在更好地沟通隐私政策方面的实施和效果概述；(b) 识别可以进一步改进以提供更 robust 的隐私政策的方法论；(c) 识别当前先进研究中的不足之处。我们的系统分析表明，许多研究论文集中在注释和分类隐私文本以供分析，但需要充分探讨NLP应用的其他方面，如摘要。更具体地讲，在这一领域存在大量的研究机会，涵盖了语料库生成、摘要向量、上下文嵌入、隐私相关声明类别的识别、细粒度分类以及领域特定模型调整等方面。 

---
# Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling 

**Title (ZH)**: 通过责任建模预防过度依赖面向任务的对话AI 

**Authors**: Suvodip Dey, Yi-Jyun Sun, Gokhan Tur, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2501.10316)  

**Abstract**: Recent LLMs have enabled significant advancements for conversational agents. However, they are also well-known to hallucinate, i.e., they often produce responses that seem plausible but are not factually correct. On the other hand, users tend to over-rely on LLM-based AI agents; they accept the AI's suggestion even when it is wrong. Adding good friction, such as explanations or getting user confirmations, has been proposed as a mitigation in AI-supported decision-making systems. In this paper, we propose an accountability model for LLM-based task-oriented dialogue agents to address user overreliance via friction turns in cases of model uncertainty and errors associated with dialogue state tracking (DST). The accountability model is an augmented LLM with an additional accountability head, which functions as a binary classifier to predict the slots of the dialogue states. We perform our experiments with three backbone LLMs (Llama, Mistral, Gemma) on two established task-oriented datasets (MultiWOZ and Snips). Our empirical findings demonstrate that this approach not only enables reliable estimation of AI agent errors but also guides the LLM decoder in generating more accurate actions. We observe around 3% absolute improvement in joint goal accuracy by incorporating accountability heads in modern LLMs for the MultiWOZ dataset. We also show that this method enables the agent to self-correct its actions, further boosting its performance by 3%. Finally, we discuss the application of accountability modeling to prevent user overreliance by introducing friction. 

**Abstract (ZH)**: 近年来，大型语言模型（LLM）为对话代理带来了显著的进步。然而，它们也广为人知地具有自幻象特性，即经常生成看似合理但实际上并不符合事实的响应。另一方面，用户倾向于过度依赖基于LLM的AI代理；即使AI的建议是错误的，他们也往往会接受。加入“摩擦”措施，如解释或获取用户的确认，已被提议用于AI支持的决策系统中以减轻这种依赖。在这篇论文中，我们提出了一个负责制模型，用于解决基于LLM的任务导向对话代理在模型不确定性及对话状态跟踪（DST）相关错误情况下的用户过度依赖问题，通过引入摩擦环节。该负责制模型是一个增强的LLM，额外增加了一个负责制头，其功能作为二元分类器来预测对话状态的槽位。我们在三个骨干LLM（Llama、Mistral、Gemma）上使用了两个现有的任务导向数据集（MultiWOZ和Snips）进行了实验。我们的实证发现表明，这种方法不仅能够有效估计AI代理的错误，还能够引导LLM解码器生成更准确的操作。我们在MultiWOZ数据集上引入负责制头的现代LLM中观察到约3%的绝对改进，使联合目标准确度得以提升。此外，我们还展示了这种方法使代理能够自我纠正其操作，进一步提高了其性能约3%。最后，我们讨论了通过引入摩擦来应用负责制建模以防止用户过度依赖的方法。 

---
# Multi-stage Training of Bilingual Islamic LLM for Neural Passage Retrieval 

**Title (ZH)**: 面向神经段落检索的双语伊斯兰大语言模型的多阶段训练方法 

**Authors**: Vera Pavlova  

**Link**: [PDF](https://arxiv.org/pdf/2501.10175)  

**Abstract**: This study examines the use of Natural Language Processing (NLP) technology within the Islamic domain, focusing on developing an Islamic neural retrieval model. By leveraging the robust XLM-R model, the research employs a language reduction technique to create a lightweight bilingual large language model (LLM). Our approach for domain adaptation addresses the unique challenges faced in the Islamic domain, where substantial in-domain corpora exist only in Arabic while limited in other languages, including English.
The work utilizes a multi-stage training process for retrieval models, incorporating large retrieval datasets, such as MS MARCO, and smaller, in-domain datasets to improve retrieval performance. Additionally, we have curated an in-domain retrieval dataset in English by employing data augmentation techniques and involving a reliable Islamic source. This approach enhances the domain-specific dataset for retrieval, leading to further performance gains.
The findings suggest that combining domain adaptation and a multi-stage training method for the bilingual Islamic neural retrieval model enables it to outperform monolingual models on downstream retrieval tasks. 

**Abstract (ZH)**: 本研究探讨了自然语言处理（NLP）技术在伊斯兰领域的应用，重点关注开发一种伊斯兰神经检索模型。通过利用强大的XLM-R模型，研究采用语言缩减技术创建了轻量级的双语大型语言模型（LLM）。我们在领域适应方面的方法应对了伊斯兰领域特有的挑战，其中大量领域内语料库仅存在于阿拉伯语中，而在其他语言中，尤其是英语中则极为有限。

研究利用了多阶段训练方法来改进检索模型，结合了大型检索数据集（如MS MARCO）以及更小的领域内数据集，以提高检索性能。此外，我们通过采用数据增强技术并利用可靠的伊斯兰来源数据，建立了英语领域的领域内检索数据集。这种方法增强了领域特定的检索数据集，从而进一步提高了性能。

研究结果表明，结合领域适应和多阶段训练方法能够使双语伊斯兰神经检索模型在下游检索任务中优于单一语言模型。 

---
# Dual Debiasing: Remove Stereotypes and Keep Factual Gender for Fair Language Modeling and Translation 

**Title (ZH)**: 双重去bias化：去除刻板印象并保留事实上的性别，以实现公平的语言建模和翻译 

**Authors**: Tomasz Limisiewicz, David Mareček, Tomáš Musil  

**Link**: [PDF](https://arxiv.org/pdf/2501.10150)  

**Abstract**: Mitigation of biases, such as language models' reliance on gender stereotypes, is a crucial endeavor required for the creation of reliable and useful language technology. The crucial aspect of debiasing is to ensure that the models preserve their versatile capabilities, including their ability to solve language tasks and equitably represent various genders. To address this issue, we introduce a streamlined Dual Dabiasing Algorithm through Model Adaptation (2DAMA). Novel Dual Debiasing enables robust reduction of stereotypical bias while preserving desired factual gender information encoded by language models. We show that 2DAMA effectively reduces gender bias in English and is one of the first approaches facilitating the mitigation of stereotypical tendencies in translation. The proposed method's key advantage is the preservation of factual gender cues, which are useful in a wide range of natural language processing tasks. 

**Abstract (ZH)**: 消除偏差，例如语言模型对性别刻板印象的依赖，是创建可靠和有用的语言技术所需的关键努力。减轻偏差的关键在于确保模型保留其多样化的功能，包括解决语言任务的能力和公平地代表各种性别。为了解决这一问题，我们引入了一种简化双消偏算法通过模型适应（2DAMA）。新颖的双消偏能够稳健地减少刻板印象偏差，同时保留由语言模型编码的所需事实性别信息。我们展示了2DAMA有效降低了英语中的性别偏差，并且是较少有方法之一，能够降低翻译中的刻板印象倾向。所提出方法的关键优势在于保留了有助于广泛自然语言处理任务的事实性性别线索。 

---
# ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario 

**Title (ZH)**: ComplexFuncBench: 在长上下文场景下探索多步约束函数调用 

**Authors**: Lucen Zhong, Zhengxiao Du, Xiaohan Zhang, Haiyi Hu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10132)  

**Abstract**: Enhancing large language models (LLMs) with real-time APIs can help generate more accurate and up-to-date responses. However, evaluating the function calling abilities of LLMs in real-world scenarios remains under-explored due to the complexity of data collection and evaluation. In this work, we introduce ComplexFuncBench, a benchmark for complex function calling across five real-world scenarios. Compared to existing benchmarks, ComplexFuncBench encompasses multi-step and constrained function calling, which requires long-parameter filing, parameter value reasoning, and 128k long context. Additionally, we propose an automatic framework, ComplexEval, for quantitatively evaluating complex function calling tasks. Through comprehensive experiments, we demonstrate the deficiencies of state-of-the-art LLMs in function calling and suggest future directions for optimizing these capabilities. The data and code are available at \url{this https URL}. 

**Abstract (ZH)**: 增强大型语言模型（LLMs）通过实时API可以生成更准确和最新的响应。然而，由于数据收集和评估的复杂性，在实际场景中评估LLMs的函数调用能力仍未能得到充分探索。本文介绍了ComplexFuncBench，这是一个跨五个真实场景的复杂函数调用基准。相比于现有基准，ComplexFuncBench 包含多步和受限函数调用，这需要长参数填充、参数值推理以及128K的长上下文。此外，我们提出了一种自动框架，ComplexEval，用于定量评估复杂函数调用任务。通过全面的实验，我们展示了最先进的LLMs在函数调用方面的局限性，并建议优化这些能力的未来方向。数据和代码可在 \url{此链接} 获取。 

---
# BBPOS: BERT-based Part-of-Speech Tagging for Uzbek 

**Title (ZH)**: BBPOS: 基于BERT的乌茲别克语词性标注 

**Authors**: Latofat Bobojonova, Arofat Akhundjanova, Phil Ostheimer, Sophie Fellenz  

**Link**: [PDF](https://arxiv.org/pdf/2501.10107)  

**Abstract**: This paper advances NLP research for the low-resource Uzbek language by evaluating two previously untested monolingual Uzbek BERT models on the part-of-speech (POS) tagging task and introducing the first publicly available UPOS-tagged benchmark dataset for Uzbek. Our fine-tuned models achieve 91% average accuracy, outperforming the baseline multi-lingual BERT as well as the rule-based tagger. Notably, these models capture intermediate POS changes through affixes and demonstrate context sensitivity, unlike existing rule-based taggers. 

**Abstract (ZH)**: 本文通过评估两种以前未测试的单语乌兹别克语 BERT 模型在词性标注任务（POS 标注）上的性能，并引入首个公开的 UPOS 标注基准数据集，推进了低资源乌兹别克语的自然语言处理研究。我们微调后的模型平均准确率达到 91%，超过了基于多语言 BERT 的基线模型以及现有的基于规则的标注器。值得注意的是，这些模型能够捕捉通过前缀和后缀体现的中间词性变化，并展现出对上下文的敏感性，而现有的基于规则的标注器则不具备这些特性。 

---
# Author-Specific Linguistic Patterns Unveiled: A Deep Learning Study on Word Class Distributions 

**Title (ZH)**: 作者特定的语言模式揭示：基于词类分布的深度学习研究 

**Authors**: Patrick Krauss, Achim Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2501.10072)  

**Abstract**: Deep learning methods have been increasingly applied to computational linguistics to uncover patterns in text data. This study investigates author-specific word class distributions using part-of-speech (POS) tagging and bigram analysis. By leveraging deep neural networks, we classify literary authors based on POS tag vectors and bigram frequency matrices derived from their works. We employ fully connected and convolutional neural network architectures to explore the efficacy of unigram and bigram-based representations. Our results demonstrate that while unigram features achieve moderate classification accuracy, bigram-based models significantly improve performance, suggesting that sequential word class patterns are more distinctive of authorial style. Multi-dimensional scaling (MDS) visualizations reveal meaningful clustering of authors' works, supporting the hypothesis that stylistic nuances can be captured through computational methods. These findings highlight the potential of deep learning and linguistic feature analysis for author profiling and literary studies. 

**Abstract (ZH)**: 深度学习方法在计算语言学中越来越多地被应用于揭示文本数据中的模式。本研究通过词性标注和双词分析，研究作者特有的词类分布。利用深度神经网络，我们根据作品中的词性标注向量和双词频率矩阵对文学作者进行分类。我们采用完全连接和卷积神经网络架构，以探讨单词特征和双词特征表示的有效性。研究结果表明，单词特征的分类准确率虽中等，但基于双词的模型显著提高了性能，这表明序列词类模式对于作者风格更为独特。多维标度（MDS）可视化显示作者作品的有意义聚类，支持通过计算方法捕捉风格细微差别的假设。这些发现突显了深度学习和语言特征分析在作者画像和文学研究中的潜在价值。 

---
# MSTS: A Multimodal Safety Test Suite for Vision-Language Models 

**Title (ZH)**: MSTS：面向视觉-语言模型的多模态安全测试套件 

**Authors**: Paul Röttger, Giuseppe Attanasio, Felix Friedrich, Janis Goldzycher, Alicia Parrish, Rishabh Bhardwaj, Chiara Di Bonaventura, Roman Eng, Gaia El Khoury Geagea, Sujata Goswami, Jieun Han, Dirk Hovy, Seogyeong Jeong, Paloma Jeretič, Flor Miriam Plaza-del-Arco, Donya Rooein, Patrick Schramowski, Anastassia Shaitarova, Xudong Shen, Richard Willats, Andrea Zugarini, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2501.10057)  

**Abstract**: Vision-language models (VLMs), which process image and text inputs, are increasingly integrated into chat assistants and other consumer AI applications. Without proper safeguards, however, VLMs may give harmful advice (e.g. how to self-harm) or encourage unsafe behaviours (e.g. to consume drugs). Despite these clear hazards, little work so far has evaluated VLM safety and the novel risks created by multimodal inputs. To address this gap, we introduce MSTS, a Multimodal Safety Test Suite for VLMs. MSTS comprises 400 test prompts across 40 fine-grained hazard categories. Each test prompt consists of a text and an image that only in combination reveal their full unsafe meaning. With MSTS, we find clear safety issues in several open VLMs. We also find some VLMs to be safe by accident, meaning that they are safe because they fail to understand even simple test prompts. We translate MSTS into ten languages, showing non-English prompts to increase the rate of unsafe model responses. We also show models to be safer when tested with text only rather than multimodal prompts. Finally, we explore the automation of VLM safety assessments, finding even the best safety classifiers to be lacking. 

**Abstract (ZH)**: 视觉语言模型（VLMs），这些模型处理图像和文本输入，正越来越多地集成到聊天助手和其他消费者AI应用中。然而，如果没有适当的保障措施，VLMs可能会提供有害建议（例如如何自残）或鼓励不安全行为（例如摄入药物）。尽管存在这些明显的风险，迄今为止很少有研究评估VLM的安全性以及多模态输入带来的新型风险。为弥补这一空白，我们引入了MSTS，即多模态安全性测试套件（Multimodal Safety Test Suite for VLMs）。MSTS 包含了400个测试提示，涵盖了40个细微的风险类别。每个测试提示由一个文本和一个图像组成，只有结合起来才能揭示其全部潜在的不安全含义。通过MSTS，我们发现了一些开放性VLM中存在的明显安全问题。我们还发现有些VLM由于无法理解甚至简单的测试提示而意外地表现出安全性。我们将MSTS翻译成十种语言，显示非英语提示可以增加不可靠模型的响应率。我们还展示了仅使用文本而非多模态提示进行测试可以使模型更安全。最后，我们探讨了VLM安全性评估的自动化问题，发现即使是最好的安全性分类器也存在不足。 

---
# Automatic Speech Recognition for Sanskrit with Transfer Learning 

**Title (ZH)**: 使用迁移学习的梵语自动语音识别 

**Authors**: Bidit Sadhukhan, Swami Punyeshwarananda  

**Link**: [PDF](https://arxiv.org/pdf/2501.10024)  

**Abstract**: Sanskrit, one of humanity's most ancient languages, has a vast collection of books and manuscripts on diverse topics that have been accumulated over millennia. However, its digital content (audio and text), which is vital for the training of AI systems, is profoundly limited. Furthermore, its intricate linguistics make it hard to develop robust NLP tools for wider accessibility. Given these constraints, we have developed an automatic speech recognition model for Sanskrit by employing transfer learning mechanism on OpenAI's Whisper model. After carefully optimising the hyper-parameters, we obtained promising results with our transfer-learned model achieving a word error rate of 15.42% on Vaksancayah dataset. An online demo of our model is made available for the use of public and to evaluate its performance firsthand thereby paving the way for improved accessibility and technological support for Sanskrit learning in the modern era. 

**Abstract (ZH)**: 梵语，人类最古老的语言之一，积累了跨越千年的关于各种主题的大量书籍和手稿。然而，其数字化内容（音频和文本），对于训练AI系统至关重要，却极为有限。此外，其复杂的语言结构使得开发更 robust 的自然语言处理（NLP）工具以实现更广泛的应用变得困难。鉴于这些限制，我们通过在OpenAI的Whisper模型基础上运用迁移学习机制，开发了一种自动语音识别模型。经过仔细的超参数优化，我们获得了令人鼓舞的结果，迁移学习模型在Vaksancayah数据集上的单词错误率降至15.42%。我们已经提供了一个在线演示，供公众使用并亲自评估其性能，从而为古代梵语学习的现代化带来更便捷的访问和技术支持。 

---
# Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models 

**Title (ZH)**: 面向注意力引导的自省方法在大型语言模型零样本幻觉检测中的应用 

**Authors**: Qiang Liu, Xinlong Chen, Yue Ding, Shizhen Xu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09997)  

**Abstract**: Hallucination has emerged as a significant barrier to the effective application of Large Language Models (LLMs). In this work, we introduce a novel Attention-Guided SElf-Reflection (AGSER) approach for zero-shot hallucination detection in LLMs. The AGSER method utilizes attention contributions to categorize the input query into attentive and non-attentive queries. Each query is then processed separately through the LLMs, allowing us to compute consistency scores between the generated responses and the original answer. The difference between the two consistency scores serves as a hallucination estimator. In addition to its efficacy in detecting hallucinations, AGSER notably reduces computational complexity, requiring only three passes through the LLM and utilizing two sets of tokens. We have conducted extensive experiments with four widely-used LLMs across three different hallucination benchmarks, demonstrating that our approach significantly outperforms existing methods in zero-shot hallucination detection. 

**Abstract (ZH)**: 幻觉已成为大型语言模型（LLMs）有效应用的重要障碍。本文引入了一种新颖的注意力引导自我反思（AGSER）方法，用于零样本幻觉检测。AGSER 方法利用注意力贡献将输入查询分为注意查询和非注意查询。随后，每个查询分别通过 LLM 处理，使得我们可以计算生成响应与原始答案之间的一致性得分。两个一致性得分之间的差异作为幻觉估计器。除了在检测幻觉方面的有效性之外，AGSER 显著减少了计算复杂度，只需三次 LLM 通过，并使用两组标记。我们使用四种广泛使用的 LLM 在三个不同的幻觉基准上进行了大量的实验，结果显示我们的方法在零样本幻觉检测方面的性能显著优于现有方法。 

---
# Agent-as-Judge for Factual Summarization of Long Narratives 

**Title (ZH)**: 将下面的论文内容或标题翻译成中文，并确保符合学术规范：

**Agent-as-Judge for Factual Summarization of Long Narratives**

**作为裁判的代理：长叙事事实总结**

在这个翻译中，“Agent”是指执行某种任务或决策的实体，“as”在此处作为介词，表明角色或功能的转变，“Judge”在这里可以理解为“裁判”，指的是对事物进行判断的角色。因此，“Agent-as-Judge”可以翻译为“作为裁判的代理”。整体标题翻译时考虑到学术规范和表达的准确性，确保意思清晰、专业。 

**Authors**: Yeonseok Jeong, Minsoo Kim, Seung-won Hwang, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.09993)  

**Abstract**: Large Language Models (LLMs) have demonstrated near-human performance in summarization tasks based on traditional metrics such as ROUGE and BERTScore. However, these metrics do not adequately capture critical aspects of summarization quality, such as factual accuracy, particularly for long narratives (>100K tokens). Recent advances, such as LLM-as-a-Judge, address the limitations of metrics based on lexical similarity but still exhibit factual inconsistencies, especially in understanding character relationships and states. In this work, we introduce NarrativeFactScore, a novel "Agent-as-a-Judge" framework for evaluating and refining summaries. By leveraging a Character Knowledge Graph (CKG) extracted from input and generated summaries, NarrativeFactScore assesses the factual consistency and provides actionable guidance for refinement, such as identifying missing or erroneous facts. We demonstrate the effectiveness of NarrativeFactScore through a detailed workflow illustration and extensive validation on widely adopted benchmarks, achieving superior performance compared to competitive methods. Our results highlight the potential of agent-driven evaluation systems to improve the factual reliability of LLM-generated summaries. 

**Abstract (ZH)**: 大语言模型（LLMs）在传统评测指标如ROUGE和BERTScore的基础上，在摘要任务上已经展现出了接近人类的性能。然而，这些指标并未充分捕捉到摘要质量的关键方面，特别是对于长篇叙述（超过10万tokens）中的事实准确性。最近的进步，如LLM-as-a-Judge，解决了基于词义相似性的指标的局限性，但仍存在事实不一致的问题，特别是在理解角色关系和状态方面。在这项工作中，我们引入了NarrativeFactScore框架，这是一种新的“Agent-as-a-Judge”评估和优化摘要的方法。通过利用从输入和生成的摘要中提取的Character Knowledge Graph（CKG），NarrativeFactScore评估了事实一致性，并提供了改进的实用指导，如识别缺失或错误的事实。我们通过详尽的工作流程说明和广泛验证，在广泛使用的基准数据集上展示了NarrativeFactScore的有效性，其性能优于竞争方法。我们的结果显示，基于代理的评估系统有提高LLM生成摘要的事实可靠性的潜力。 

---
# A Survey on Multi-Turn Interaction Capabilities of Large Language Models 

**Title (ZH)**: 大型语言模型多轮交互能力综述 

**Authors**: Chen Zhang, Xinyi Dai, Yaxiong Wu, Qu Yang, Yasheng Wang, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09959)  

**Abstract**: Multi-turn interaction in the dialogue system research refers to a system's ability to maintain context across multiple dialogue turns, enabling it to generate coherent and contextually relevant responses. Recent advancements in large language models (LLMs) have significantly expanded the scope of multi-turn interaction, moving beyond chatbots to enable more dynamic agentic interactions with users or environments. In this paper, we provide a focused review of the multi-turn capabilities of LLMs, which are critical for a wide range of downstream applications, including conversational search and recommendation, consultation services, and interactive tutoring. This survey explores four key aspects: (1) the core model capabilities that contribute to effective multi-turn interaction, (2) how multi-turn interaction is evaluated in current practice, (3) the general algorithms used to enhance multi-turn interaction, and (4) potential future directions for research in this field. 

**Abstract (ZH)**: 对话系统中的多轮交互指的是系统能够在多轮对话中维持上下文的能力，使其能够生成连贯且上下文相关的响应。大型语言模型（LLMs）的最新进展极大地扩展了多轮交互的范围，使其超越了聊天机器人，能够与用户或环境进行更加动态的自主交互。在本文中，我们对LLMs的多轮交互能力进行了集中审查，这些能力对于各种下游应用至关重要，包括对话式搜索和推荐、咨询服务以及互动式教学。本文回顾了四个关键方面：（1）构成有效多轮交互的核心模型能力，（2）当前实践中多轮交互的评估方法，（3）用于增强多轮交互的一般算法，以及（4）该领域未来研究方向的潜在可能性。 

---
# FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs 

**Title (ZH)**: FRAG：一种基于知识图谱的灵活模块化检索增强生成框架 

**Authors**: Zengyi Gao, Yukun Cao, Hairu Wang, Ao Ke, Yuan Feng, Xike Xie, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.09957)  

**Abstract**: To mitigate the hallucination and knowledge deficiency in large language models (LLMs), Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) has shown promising potential by utilizing KGs as external resource to enhance LLMs this http URL, existing KG-RAG approaches struggle with a trade-off between flexibility and retrieval this http URL methods prioritize flexibility by avoiding the use of KG-fine-tuned models during retrieval, leading to fixed retrieval strategies and suboptimal retrieval this http URL, coupled methods embed KG information within models to improve retrieval quality, but at the expense of this http URL this paper, we propose a novel flexible modular KG-RAG framework, termed FRAG, which synergizes the advantages of both this http URL estimates the hop range of reasoning paths based solely on the query and classify it as either simple or this http URL match the complexity of the query, tailored pipelines are applied to ensure efficient and accurate reasoning path retrieval, thus fostering the final reasoning this http URL using the query text instead of the KG to infer the structural information of reasoning paths and employing adaptable retrieval strategies, FRAG improves retrieval quality while maintaining this http URL, FRAG does not require extra LLMs fine-tuning or calls, significantly boosting efficiency and conserving this http URL experiments show that FRAG achieves state-of-the-art performance with high efficiency and low resource consumption. 

**Abstract (ZH)**: 为了缓解大规模语言模型（LLMs）中的幻觉和知识缺陷，通过利用知识图谱（KGs）作为外部资源来增强LLMs的检索增强生成（RAG）方法显示出有前景的潜力（this http URL），现有的KG-RAG方法在灵活性与检索之间面临权衡（this http URL）。一种方法通过避免在检索期间使用KG微调模型来优先考虑灵活性，导致固定检索策略和次优的检索效果（this http URL）。另一种方法将KG信息嵌入模型中以提高检索质量，但会牺牲（this http URL）。本文提出了一种新的灵活模块化的 KG-RAG 框架，称为FRAG，该框架结合了上述两种方法的优点（this http URL）。基于查询估算推理路径的合理跨度，并将其分类为简单或复杂，根据查询的复杂度调整合适的流水线，以确保高效准确的推理路径检索，从而促进最终的推理过程（this http URL）。使用查询文本而不是KG来推断推理路径的结构信息，并采用可调适的检索策略，FRAG 提高了检索质量，同时保持了灵活性（this http URL）。FRAG 不需要额外的LLMs微调或调用，显著提高了效率并节省了资源。实验结果表明，FRAG 在高效率和低资源消耗的情况下达到了最先进的性能（this http URL）。 

---
# Indigenous Languages Spoken in Argentina: A Survey of NLP and Speech Resources 

**Title (ZH)**: 阿根廷境内原住民语言的使用情况：自然语言处理和语音资源调查 

**Authors**: Belu Ticona, Fernando Carranza, Viviana Cotik  

**Link**: [PDF](https://arxiv.org/pdf/2501.09943)  

**Abstract**: Argentina has a diverse, yet little-known, Indigenous language heritage. Most of these languages are at risk of disappearing, resulting in a significant loss of world heritage and cultural knowledge. Currently, no unified information on speakers and computational tools is available for these languages. In this work, we present a systematization of the Indigenous languages spoken in Argentina, along with national demographic data on the country's Indigenous population. The languages are classified into seven families: Mapuche, Tupí-Guaraní, Guaycurú, Quechua, Mataco-Mataguaya, Aymara, and Chon. We also provide an introductory survey of the computational resources available for these languages, whether or not they are specifically developed for Argentine varieties. 

**Abstract (ZH)**: 阿根廷拥有丰富而鲜为人知的原住民语言遗产。这些语言大多处于濒临消失的危险之中，这将导致世界遗产和文化知识的重大损失。目前，尚无统一的信息和计算工具资源可用于这些语言。在此项工作中，我们系统整理了在阿根廷 spoken 的原住民语言，并提供了该国原住民人口的国家人口统计数据。这些语言被分为七大语系：马普切语系、图皮- Guarani 语系、瓦尤库鲁语系、克丘亚语系、马塔科-马塔古亚拉语系、阿伊马拉语系和琼语系。我们还提供了关于这些语言可利用的计算资源的初步概述，无论是专门为阿根廷变体开发的资源还是其他资源。 

---
# Passage Segmentation of Documents for Extractive Question Answering 

**Title (ZH)**: 文档中的段落分割用于提取式问答 

**Authors**: Zuhong Liu, Charles-Elie Simon, Fabien Caspani  

**Link**: [PDF](https://arxiv.org/pdf/2501.09940)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective in open-domain question answering. However, the chunking process, which is essential to this pipeline, often receives insufficient attention relative to retrieval and synthesis components. This study emphasizes the critical role of chunking in improving the performance of both dense passage retrieval and the end-to-end RAG pipeline. We then introduce the Logits-Guided Multi-Granular Chunker (LGMGC), a novel framework that splits long documents into contextualized, self-contained chunks of varied granularity. Our experimental results, evaluated on two benchmark datasets, demonstrate that LGMGC not only improves the retrieval step but also outperforms existing chunking methods when integrated into a RAG pipeline. 

**Abstract (ZH)**: 检索增强生成（RAG）在开放式领域问答中已被证明是有效的。然而，这一管道中的关键步骤——分块过程——往往相较于检索和合成组件而言得到了较少的关注。本研究强调了分块在提高密集段落检索性能以及端到端RAG管道性能中的关键作用。随后，我们引入了一种新的框架——Logits-Guided多粒度分块器（LGMGC），该框架能够将长文档拆分成自包含的、上下文化的不同粒度的片段。我们在两个基准数据集上的实验结果表明，LGMGC不仅改善了检索步骤的表现，而且在其集成到RAG管道中时，其性能也优于现有的分块方法。 

---
# Dialogue Benchmark Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented LLMs 

**Title (ZH)**: 使用低成本检索增强大型语言模型从知识图生成对话基准 

**Authors**: Reham Omar, Omij Mangukiya, Essam Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2501.09928)  

**Abstract**: Dialogue benchmarks are crucial in training and evaluating chatbots engaging in domain-specific conversations. Knowledge graphs (KGs) represent semantically rich and well-organized data spanning various domains, such as DBLP, DBpedia, and YAGO. Traditionally, dialogue benchmarks have been manually created from documents, neglecting the potential of KGs in automating this process. Some question-answering benchmarks are automatically generated using extensive preprocessing from KGs, but they do not support dialogue generation. This paper introduces Chatty-Gen, a novel multi-stage retrieval-augmented generation platform for automatically generating high-quality dialogue benchmarks tailored to a specific domain using a KG. Chatty-Gen decomposes the generation process into manageable stages and uses assertion rules for automatic validation between stages. Our approach enables control over intermediate results to prevent time-consuming restarts due to hallucinations. It also reduces reliance on costly and more powerful commercial LLMs. Chatty-Gen eliminates upfront processing of the entire KG using efficient query-based retrieval to find representative subgraphs based on the dialogue context. Our experiments with several real and large KGs demonstrate that Chatty-Gen significantly outperforms state-of-the-art systems and ensures consistent model and system performance across multiple LLMs of diverse capabilities, such as GPT-4o, Gemini 1.5, Llama 3, and Mistral. 

**Abstract (ZH)**: 对话基准在训练和评估涉专业领域对话的聊天机器人方面至关重要。知识图谱（KGs）能够表示跨多个领域的丰富且结构良好的数据，如DBLP、DBpedia和YAGO。传统上，对话基准是通过手动从文件中创建的，忽视了KGs在自动化这一过程中的潜力。虽然一些问答基准数据集是通过大量预处理从KGs自动生成的，但它们不支持对话生成。本文介绍了Chatty-Gen，这是一种新颖的多阶段检索增强生成平台，能够利用KG自动生成高质量的针对特定领域的对话基准数据集。Chatty-Gen将生成过程分解为可管理的阶段，并使用断言规则在阶段之间进行自动验证。我们的方法能够控制中间结果，防止由于幻觉而导致的时间消耗重启。同时，它也减少了对昂贵且更强大的商业大语言模型（LLMs）的依赖。Chatty-Gen通过高效的基于查询的检索来消除对整个KG的前期处理，根据对话上下文找到具有代表性的子图。我们的实验表明，Chatty-Gen在多个真实且大规模的KG上显著优于最新的系统，并确保了在多样能力的LLM（如GPT-4o、Gemini 1.5、Llama 3和Mistral）中的模型和系统性能的一致性。 

---
# Bridging Language Barriers in Healthcare: A Study on Arabic LLMs 

**Title (ZH)**: 跨越医疗领域的语言障碍：关于阿拉伯语语言模型的研究 

**Authors**: Nada Saadi, Tathagata Raha, Clément Christophe, Marco AF Pimentel, Ronnie Rajan, Praveen K Kanithi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09825)  

**Abstract**: This paper investigates the challenges of developing large language models (LLMs) proficient in both multilingual understanding and medical knowledge. We demonstrate that simply translating medical data does not guarantee strong performance on clinical tasks in the target language. Our experiments reveal that the optimal language mix in training data varies significantly across different medical tasks. We find that larger models with carefully calibrated language ratios achieve superior performance on native-language clinical tasks. Furthermore, our results suggest that relying solely on fine-tuning may not be the most effective approach for incorporating new language knowledge into LLMs. Instead, data and computationally intensive pretraining methods may still be necessary to achieve optimal performance in multilingual medical settings. These findings provide valuable guidance for building effective and inclusive medical AI systems for diverse linguistic communities. 

**Abstract (ZH)**: 本文探讨了开发既擅长多语言理解和医学知识的大型语言模型（LLMs）面临的挑战。我们表明，仅仅翻译医学数据并不能保证目标语言中临床任务的优秀表现。我们的实验揭示，训练数据中的最佳语言混合比例在不同医学任务之间差异显著。我们发现，经过仔细校准语言比例的较大模型在母语临床任务中表现出更优秀的效果。此外，我们的结果表明，仅仅依赖微调可能不是将新语言知识纳入LLMs的最有效方法。相反，在多语言医学环境中实现最优性能可能仍然需要数据和计算密集型的预训练方法。这些发现为构建适用于多元语言社区的有效且包容性较强的医疗AI系统提供了宝贵的指导。 

---
# Qwen it detect machine-generated text? 

**Title (ZH)**: “Qwen是如何检测机器生成的文本的？” 

**Authors**: Teodor-George Marchitan, Claudiu Creanga, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09813)  

**Abstract**: This paper describes the approach of the Unibuc - NLP team in tackling the Coling 2025 GenAI Workshop, Task 1: Binary Multilingual Machine-Generated Text Detection. We explored both masked language models and causal models. For Subtask A, our best model achieved first-place out of 36 teams when looking at F1 Micro (Auxiliary Score) of 0.8333, and second-place when looking at F1 Macro (Main Score) of 0.8301 

**Abstract (ZH)**: 本文描述了Unibuc-NLP团队在应对COLING 2025 GenAI研讨会任务1：二元多语言机器生成文本检测的方法。我们探索了掩码语言模型和因果模型。在子任务A中，我们的最佳模型在F1 Micro（辅助评分）为0.8333时，在36支参赛队伍中获得了第一的成绩；而在F1 Macro（主评分）为0.8301时，获得了第二名的成绩。 

---
# Sentiment Analysis in Twitter Social Network Centered on Cryptocurrencies Using Machine Learning 

**Title (ZH)**: 使用机器学习在基于加密货币的推特社交网络中的情感分析 

**Authors**: Vahid Amiri, Mahmood Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09777)  

**Abstract**: Cryptocurrency is a digital currency that uses blockchain technology with secure encryption. Due to the decentralization of these currencies, traditional monetary systems and the capital market of each they, can influence a society. Therefore, due to the importance of the issue, the need to understand public opinion and analyze people's opinions in this regard increases. To understand the opinions and views of people about different topics, you can take help from social networks because they are a rich source of opinions. The Twitter social network is one of the main platforms where users discuss various topics, therefore, in the shortest time and with the lowest cost, the opinion of the community can be measured on this social network. Twitter Sentiment Analysis (TSA) is a field that analyzes the sentiment expressed in tweets. Considering that most of TSA's research efforts on cryptocurrencies are focused on English language, the purpose of this paper is to investigate the opinions of Iranian users on the Twitter social network about cryptocurrencies and provide the best model for classifying tweets based on sentiment. In the case of automatic analysis of tweets, managers and officials in the field of economy can gain knowledge from the general public's point of view about this issue and use the information obtained in order to properly manage this phenomenon. For this purpose, in this paper, in order to build emotion classification models, natural language processing techniques such as bag of words (BOW) and FastText for text vectorization and classical machine learning algorithms including KNN, SVM and Adaboost learning methods Deep including LSTM and BERT model were used for classification, and finally BERT linguistic model had the best accuracy with 83.50%. 

**Abstract (ZH)**: 数字货币是一种利用区块链技术进行安全加密的电子货币。由于这些货币的去中心化特性，它们对传统的货币系统和各自资本市场会产生影响，进而影响社会。因此，鉴于该问题的重要性，需要理解公众意见并分析人们的观点变得越来越重要。为了理解人们对不同主题的意见和观点，可以从社交媒体中获取帮助，因为它们是观点的丰富来源。Twitter 是用户讨论各种话题的主要平台之一，因此，可以迅速且低成本地通过 Twitter 测量社区的意见。Twitter 情感分析（TSA）是对推文中表达的情感进行分析的领域。鉴于 TSA 的大部分研究工作主要是针对英语，本文旨在研究伊朗用户在 Twitter 社交网络上对数字货币的观点，并提供基于情感分类的最佳模型。在对推文进行自动分析的情况下，经济领域的管理人员和官员可以从公众的角度获得关于该问题的知识，并利用获得的信息以适当的方式管理这一现象。为此，本文通过构建情感分类模型，使用自然语言处理技术（如词袋模型 BOW 和 FastText 用于文本向量化）以及传统机器学习算法（包括 KNN、SVM 和 Adaboost 学习方法），以及深度学习方法（包括 LSTM 和 BERT 模型进行分类），最终结果显示 BERT 语言模型具有最高的准确率 83.50%。 

---
# Multiple Choice Questions: Reasoning Makes Large Language Models (LLMs) More Self-Confident Even When They Are Wrong 

**Title (ZH)**: 多项选择题：推理使大规模语言模型（LLMs）即使在答错时也能更加自信 

**Authors**: Tairan Fu, Javier Conde, Gonzalo Martínez, María Grandury, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2501.09775)  

**Abstract**: One of the most widely used methods to evaluate LLMs are Multiple Choice Question (MCQ) tests. MCQ benchmarks enable the testing of LLM knowledge on almost any topic at scale as the results can be processed automatically. To help the LLM answer, a few examples called few shots can be included in the prompt. Moreover, the LLM can be asked to answer the question directly with the selected option or to first provide the reasoning and then the selected answer, which is known as chain of thought. In addition to checking whether the selected answer is correct, the evaluation can look at the LLM-estimated probability of its response as an indication of the confidence of the LLM in the response. In this paper, we study how the LLM confidence in its answer depends on whether the model has been asked to answer directly or to provide the reasoning before answering. The results of the evaluation of questions on a wide range of topics in seven different models show that LLMs are more confident in their answers when they provide reasoning before the answer. This occurs regardless of whether the selected answer is correct. Our hypothesis is that this behavior is due to the reasoning that modifies the probability of the selected answer, as the LLM predicts the answer based on the input question and the reasoning that supports the selection made. Therefore, LLM estimated probabilities seem to have intrinsic limitations that should be understood in order to use them in evaluation procedures. Interestingly, the same behavior has been observed in humans, for whom explaining an answer increases confidence in its correctness. 

**Abstract (ZH)**: 评价大规模语言模型（LLM）的最常用方法之一是多项选择题（MCQ）测试。MCQ基准测试使得可以在大规模测试LLM知识方面发挥作用，因为结果可以自动处理。为了帮助LLM作答，可以在提示中包含一些示例，称为“零样本”或“少量样本”。此外，LLM可以被要求直接选择答案或首先提供推理再选择答案，这被称为推理链。除了检查选定的答案是否正确外，评估还可以通过检查LLM为其反应估算的概率来衡量其对答案的信心。在这种论文中，我们研究了LLM在直接作答与先提供推理再作答这两种情况下，其作答信心的变化。对七个不同模型在多种主题下的问题进行评估结果显示，LLM在先提供推理再作答的情况下对答案的信心更强。无论选定的答案是否正确，这种行为都会发生。我们的假设是，这种行为可能是由于推理改变了选定答案的概率，因为LLM是根据输入问题和支撑选择的推理来预测答案的。因此，LLM估算的概率似乎具有内在的局限性，有必要理解这些局限性以便在评估程序中加以利用。有趣的是，同样的行为也在人类中被观察到，即解释答案会增加对其正确性的信心。 

---
# Can Large Language Models Predict the Outcome of Judicial Decisions? 

**Title (ZH)**: 大型语言模型能否预测司法决策的结果？ 

**Authors**: Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Amani Al-Ghraibah  

**Link**: [PDF](https://arxiv.org/pdf/2501.09768)  

**Abstract**: Large Language Models (LLMs) have shown exceptional capabilities in Natural Language Processing (NLP) across diverse domains. However, their application in specialized tasks such as Legal Judgment Prediction (LJP) for low-resource languages like Arabic remains underexplored. In this work, we address this gap by developing an Arabic LJP dataset, collected and preprocessed from Saudi commercial court judgments. We benchmark state-of-the-art open-source LLMs, including LLaMA-3.2-3B and LLaMA-3.1-8B, under varying configurations such as zero-shot, one-shot, and fine-tuning using QLoRA. Additionally, we used a comprehensive evaluation framework combining quantitative metrics (BLEU and ROUGE) and qualitative assessments (Coherence, legal language, clarity). Our results demonstrate that fine-tuned smaller models achieve comparable performance to larger models in task-specific contexts while offering significant resource efficiency. Furthermore, we investigate the effects of prompt engineering and fine-tuning on model outputs, providing insights into performance variability and instruction sensitivity. By making the dataset, implementation code, and models publicly available, we establish a robust foundation for future research in Arabic legal NLP. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理（NLP）的各个领域中展现了出色的能力。然而，它们在低资源语言如阿拉伯语的专门任务，如法律判决预测（LJP）中的应用仍然鲜有探索。本文通过开发一个从沙特商业法庭判决中收集并预处理的阿拉伯语LJP数据集，填补了这一空白。我们基准测试了包括LLaMA-3.2-3B和LLaMA-3.1-8B在内的最先进的开源LLMs，在零样本、单样本和使用QLoRA微调的不同配置下进行评估。此外，我们采用了一种综合评估框架，结合定量指标（BLEU和ROUGE）和定性评估（连贯性、法律语言、清晰度）进行全面评估。实验结果表明，在任务特定的场景中，微调后的较小模型能够达到与较大模型相当的性能，同时具有显著的资源效率优势。此外，我们还探讨了提示工程和微调对模型输出的影响，揭示了性能变化和指令敏感性的见解。通过公开该数据集、实现代码和模型，我们为进一步开展阿拉伯语法律NLP研究奠定了坚实的基础。 

---
# LeMo: Enabling LEss Token Involvement for MOre Context Fine-tuning 

**Title (ZH)**: LeMo：减少令牌参与以实现更细致的上下文微调

在这个翻译中，“Enabling LEss Token Involvement for MOre Context Fine-tuning”被翻译为“减少令牌参与以实现更细致的上下文微调”，保持了原意的同时，符合学术论文的规范和表述方式。 

**Authors**: Tuowei Wang, Xingyu Chen, Kun Li, Ting Cao, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09767)  

**Abstract**: The escalating demand for long-context applications has intensified the necessity of extending the LLM context windows. Despite recent fine-tuning approaches successfully expanding context lengths, their high memory footprints, especially for activations, present a critical practical limitation. Current parameter-efficient fine-tuning methods prioritize reducing parameter update overhead over addressing activation memory constraints. Similarly, existing sparsity mechanisms improve computational efficiency but overlook activation memory optimization due to the phenomenon of Shadowy Activation.
In this paper, we propose LeMo, the first LLM fine-tuning system that explores and exploits a new token-level sparsity mechanism inherent in long-context scenarios, termed Contextual Token Sparsity. LeMo minimizes redundant token involvement by assessing the informativeness of token embeddings while preserving model accuracy. Specifically, LeMo introduces three key techniques: (1) Token Elimination, dynamically identifying and excluding redundant tokens across varying inputs and layers. (2) Pattern Prediction, utilizing well-trained predictors to approximate token sparsity patterns with minimal overhead. (3) Kernel Optimization, employing permutation-free and segment-based strategies to boost system performance. We implement LeMo as an end-to-end fine-tuning system compatible with various LLM architectures and other optimization techniques. Comprehensive evaluations demonstrate that LeMo reduces memory consumption by up to 1.93x and achieves up to 1.36x speedups, outperforming state-of-the-art fine-tuning systems. 

**Abstract (ZH)**: 随着长上下文应用需求的不断增长，扩展大语言模型（LLM）的语境窗口变得愈加必要。尽管最近的微调方法成功地扩展了上下文长度，但它们在激活方面高内存占用的特性，特别是高激活内存占用，构成了一个关键的实用限制。现有的参数高效的微调方法更侧重于减少参数更新开销，而非解决激活内存约束。同样，现有的稀疏化机制提高了计算效率，但因Shadowy Activation现象忽视了激活内存优化。

在本文中，我们提出了LeMo，这是首个探索并利用长上下文场景中内在的新令牌级稀疏机制的LLM微调系统，称为情境令牌稀疏性（Contextual Token Sparsity）。LeMo通过评估令牌嵌入的信息性来最小化冗余令牌的参与，同时保持模型的准确性。具体而言，LeMo引入了三种关键技术：（1）令牌消除，动态地识别并排除不同输入和层中的冗余令牌。 （2）模式预测，利用训练良好的预测器以最小的开销近似令牌稀疏模式。 （3）内核优化，采用无排列和基于段的策略来提升系统性能。我们实现了LeMo，以兼容各种LLM架构和其他优化技术的端到端微调系统。全面的实验结果表明，LeMo可将内存消耗减少多达1.93倍，并实现多达1.36倍的加速，超越了现有的最先进的微调系统。 

---
# Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning 

**Title (ZH)**: 通过迭代强化微调提升大型语言模型的工具使用能力 

**Authors**: Yirong Zeng, Xiao Ding, Yuxian Wang, Weiwen Liu, Wu Ning, Yutai Hou, Xu Huang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09766)  

**Abstract**: Augmenting large language models (LLMs) with external tools is a promising approach to enhance their capabilities. Effectively leveraging this potential for complex tasks hinges crucially on improving their ability to use tools. Synthesizing tool use data by simulating the real world is an effective approach. Nevertheless, our investigation reveals that training gains significantly decay as the scale of these data increases. The primary factor is the model's poor performance (a.k.a deficiency) in complex scenarios, which hinders learning from data using SFT. Driven by this objective, we propose an iterative reinforced fine-tuning strategy to continually guide the model to alleviate it. Specifically, we first identify deficiency-related data based on feedback from the policy model, then perform a Monte Carlo Tree Search to collect fine-grained preference pairs to pinpoint deficiencies. Subsequently, we update the policy model using preference optimization to align with ground truth and misalign with deficiencies. This process can be iterated. Moreover, before the iteration, we propose an easy-to-hard warm-up SFT strategy to facilitate learning from challenging data. The experiments demonstrate our models go beyond the same parametric models, outperforming many larger open-source and closed-source models. Additionally, it has achieved notable training gains in complex tool use scenarios. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与外部工具结合是一种提升其能力的有前途的方法。有效地利用这一潜力对于复杂任务至关重要，关键在于提高模型使用工具的能力。通过模拟现实世界来合成工具使用数据是一种有效的方法。然而，我们的研究表明，随着这些数据规模的增加，训练收益显著下降。主要因素是模型在复杂场景下的表现不佳（亦称缺陷），这阻碍了通过强化 fine-tuning（SFT）从数据中学习。为实现这一目标，我们提出了一种迭代强化 fine-tuning 策略，持续引导模型来克服这些缺陷。具体而言，我们首先基于策略模型的反馈识别与缺陷相关的数据，然后进行蒙特卡洛树搜索以收集细粒度的偏好对，以定位缺陷。随后，我们使用偏好优化更新策略模型，使其与真实值对齐并远离缺陷。这一过程可以迭代进行。此外，在迭代之前，我们提出了一种从易到难的 warm-up SFT 策略，以利于从具有挑战性的数据中学习。实验表明，我们的模型超越了具有相同参数量的模型，优于许多较大的开源和封闭源模型。此外，它还在复杂工具使用场景中实现了显著的训练收益。 

---
# Enhancing the De-identification of Personally Identifiable Information in Educational Data 

**Title (ZH)**: 增强教育数据中个人可识别信息的去标识化 

**Authors**: Y. Shen, Z. Ji, J. Lin, K. R. Koedginer  

**Link**: [PDF](https://arxiv.org/pdf/2501.09765)  

**Abstract**: Protecting Personally Identifiable Information (PII), such as names, is a critical requirement in learning technologies to safeguard student and teacher privacy and maintain trust. Accurate PII detection is an essential step toward anonymizing sensitive information while preserving the utility of educational data. Motivated by recent advancements in artificial intelligence, our study investigates the GPT-4o-mini model as a cost-effective and efficient solution for PII detection tasks. We explore both prompting and fine-tuning approaches and compare GPT-4o-mini's performance against established frameworks, including Microsoft Presidio and Azure AI Language. Our evaluation on two public datasets, CRAPII and TSCC, demonstrates that the fine-tuned GPT-4o-mini model achieves superior performance, with a recall of 0.9589 on CRAPII. Additionally, fine-tuned GPT-4o-mini significantly improves precision scores (a threefold increase) while reducing computational costs to nearly one-tenth of those associated with Azure AI Language. Furthermore, our bias analysis reveals that the fine-tuned GPT-4o-mini model consistently delivers accurate results across diverse cultural backgrounds and genders. The generalizability analysis using the TSCC dataset further highlights its robustness, achieving a recall of 0.9895 with minimal additional training data from TSCC. These results emphasize the potential of fine-tuned GPT-4o-mini as an accurate and cost-effective tool for PII detection in educational data. It offers robust privacy protection while preserving the data's utility for research and pedagogical analysis. Our code is available on GitHub: this https URL 

**Abstract (ZH)**: 保护个人可识别信息（PII），如姓名，是学习技术中的一个关键要求，旨在保护学生和教师的隐私并维持信任。准确的PII检测是实现信息匿名化的同时保持教育数据有用性的关键步骤。受人工智能最近进展的启发，我们研究了GPT-4o-mini模型作为PII检测任务的一种经济高效且高效的解决方案。我们探索了提示和微调两种方法，并将GPT-4o-mini的性能与包括Microsoft Presidio和Azure AI Language在内的现有框架进行了比较。我们在两个公开数据集CRAPII和TSCC上的评估表明，微调后的GPT-4o-mini模型表现优异，在CRAPII数据集上的召回率为0.9589。此外，微调后的GPT-4o-mini在精度评分上显著提高（提升近三倍），同时将计算成本降为Azure AI Language的约十分之一。此外，我们的偏倚分析显示，微调后的GPT-4o-mini模型在不同文化背景和性别中的结果一直准确。使用TSCC数据集进行的泛化分析进一步突显了其鲁棒性，在几乎没有额外训练数据的情况下，召回率达到0.9895。这些结果强调了微调后的GPT-4o-mini模型在教育数据分析中作为准确且成本效益高的PII检测工具的潜力。它能够在确保隐私保护的同时，保留数据对研究和教学分析的有用性。我们的代码可在GitHub上获得：https://this-url 

---
# Large language models for automated scholarly paper review: A survey 

**Title (ZH)**: 大型语言模型在自动化学术论文评审中的应用：一项综述 

**Authors**: Zhenzhen Zhuang, Jiandong Chen, Hongfeng Xu, Yuwen Jiang, Jialiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.10326)  

**Abstract**: Large language models (LLMs) have significantly impacted human society, influencing various domains. Among them, academia is not simply a domain affected by LLMs, but it is also the pivotal force in the development of LLMs. In academic publications, this phenomenon is represented during the incorporation of LLMs into the peer review mechanism for reviewing manuscripts. We proposed the concept of automated scholarly paper review (ASPR) in our previous paper. As the incorporation grows, it now enters the coexistence phase of ASPR and peer review, which is described in that paper. LLMs hold transformative potential for the full-scale implementation of ASPR, but they also pose new issues and challenges that need to be addressed. In this survey paper, we aim to provide a holistic view of ASPR in the era of LLMs. We begin with a survey to find out which LLMs are used to conduct ASPR. Then, we review what ASPR-related technological bottlenecks have been solved with the incorporation of LLM technology. After that, we move on to explore new methods, new datasets, new source code, and new online systems that come with LLMs for ASPR. Furthermore, we summarize the performance and issues of LLMs in ASPR, and investigate the attitudes and reactions of publishers and academia to ASPR. Lastly, we discuss the challenges associated with the development of LLMs for ASPR. We hope this survey can serve as an inspirational reference for the researchers and promote the progress of ASPR for its actual implementation. 

**Abstract (ZH)**: 大型语言模型（LLMs）对人类社会产生了显著影响，并已渗透到多个领域。在这个过程中，学术界不仅是LLMs影响的对象，而且也是推动LLMs发展的关键力量。在学术出版物中，这种现象体现在LLMs被纳入同行评审机制以审稿 manuscript 的过程中。我们之前提出过自动学术论文评审（ASPR）的概念。随着这一过程的发展，ASPR和传统同行评审机制现已进入了共存阶段，并在我们的论文中进行了详细描述。LLMs为ASPR的大规模实施提供了变革性的潜力，但也带来了一些新问题和挑战，需要加以解决。在这篇综述论文中，我们旨在为LLMs时代下的ASPR提供一个全面的视角。首先，我们进行了一项调研以确定使用哪些LLMs来进行ASPR。然后，我们回顾了通过引入LLM技术已经解决的与ASPR相关的技术瓶颈。接下来，我们将探讨LLMs带给ASPR的新方法、新数据集、新软件代码以及新在线系统。此外，我们总结了LLMs在ASPR中的性能和问题，并调查了出版商和学术界对ASPR的态度和反应。最后，我们讨论了用于ASPR的LLMs发展中所面临的挑战。我们希望这篇综述能够为研究人员提供灵感，并促进ASPR的实际实施进程。 

---
# Computational Protein Science in the Era of Large Language Models (LLMs) 

**Title (ZH)**: 大型语言模型时代下的计算蛋白质科学 

**Authors**: Wenqi Fan, Yi Zhou, Shijie Wang, Yuyao Yan, Hui Liu, Qian Zhao, Le Song, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.10282)  

**Abstract**: Considering the significance of proteins, computational protein science has always been a critical scientific field, dedicated to revealing knowledge and developing applications within the protein sequence-structure-function paradigm. In the last few decades, Artificial Intelligence (AI) has made significant impacts in computational protein science, leading to notable successes in specific protein modeling tasks. However, those previous AI models still meet limitations, such as the difficulty in comprehending the semantics of protein sequences, and the inability to generalize across a wide range of protein modeling tasks. Recently, LLMs have emerged as a milestone in AI due to their unprecedented language processing & generalization capability. They can promote comprehensive progress in fields rather than solving individual tasks. As a result, researchers have actively introduced LLM techniques in computational protein science, developing protein Language Models (pLMs) that skillfully grasp the foundational knowledge of proteins and can be effectively generalized to solve a diversity of sequence-structure-function reasoning problems. While witnessing prosperous developments, it's necessary to present a systematic overview of computational protein science empowered by LLM techniques. First, we summarize existing pLMs into categories based on their mastered protein knowledge, i.e., underlying sequence patterns, explicit structural and functional information, and external scientific languages. Second, we introduce the utilization and adaptation of pLMs, highlighting their remarkable achievements in promoting protein structure prediction, protein function prediction, and protein design studies. Then, we describe the practical application of pLMs in antibody design, enzyme design, and drug discovery. Finally, we specifically discuss the promising future directions in this fast-growing field. 

**Abstract (ZH)**: 考虑到蛋白质的重要性，计算蛋白质科学一直是一个关键的科学领域，致力于在蛋白质序列-结构-功能范式中揭示知识和开发应用。在过去的几十年中，人工智能（AI）在计算蛋白质科学中产生了重大影响，使得特定蛋白质建模任务取得了显著的成功。然而，之前的AI模型仍然存在一些局限性，如难以理解蛋白质序列的语义，以及无法在广泛的蛋白质建模任务中进行泛化。最近，由于其前所未有的语言处理与泛化能力，大规模语言模型（LLMs）在人工智能领域成为了一个里程碑。它们可以促进各个领域的全面进展，而不仅仅是解决个别任务。因此，研究人员已经积极将LLM技术引入计算蛋白质科学领域，开发出能够熟练掌握蛋白质基础知识的蛋白质语言模型（pLMs），并且能够有效泛化以解决多样的序列-结构-功能推理问题。尽管取得了一系列进展，但是有必要系统地概括LLM技术赋能的计算蛋白质科学的发展。首先，我们根据pLMs掌握的蛋白质知识将其分为类别，包括潜在的序列模式、明确的结构和功能信息以及外部科学语言。其次，我们介绍了pLMs的应用和适应性，突显了它们在促进蛋白质结构预测、蛋白质功能预测和蛋白质设计研究中的显著成就。然后，我们描述了pLMs在抗体设计、酶设计和药物发现中的实际应用。最后，我们特别讨论了这一快速发展的领域中富有前景的未来方向。 

---
# A Simple but Effective Closed-form Solution for Extreme Multi-label Learning 

**Title (ZH)**: 一种简洁而有效的闭形式解决方案用于极端多标签学习 

**Authors**: Kazuma Onishi, Katsuhiko Hayashi  

**Link**: [PDF](https://arxiv.org/pdf/2501.10179)  

**Abstract**: Extreme multi-label learning (XML) is a task of assigning multiple labels from an extremely large set of labels to each data instance. Many current high-performance XML models are composed of a lot of hyperparameters, which complicates the tuning process. Additionally, the models themselves are adapted specifically to XML, which complicates their reimplementation. To remedy this problem, we propose a simple method based on ridge regression for XML. The proposed method not only has a closed-form solution but also is composed of a single hyperparameter. Since there are no precedents on applying ridge regression to XML, this paper verified the performance of the method by using various XML benchmark datasets. Furthermore, we enhanced the prediction of low-frequency labels in XML, which hold informative content. This prediction is essential yet challenging because of the limited amount of data. Here, we employed a simple frequency-based weighting. This approach greatly simplifies the process compared with existing techniques. Experimental results revealed that it can achieve levels of performance comparable to, or even exceeding, those of models with numerous hyperparameters. Additionally, we found that the frequency-based weighting significantly improved the predictive performance for low-frequency labels, while requiring almost no changes in implementation. The source code for the proposed method is available on github at this https URL. 

**Abstract (ZH)**: 极端多标签学习（XML）是一种将每个数据实例从一个极其庞大的标签集合中分配多个标签的任务。当前许多高性能的XML模型包含大量的超参数，这使得调参过程变得复杂。此外，这些模型本身就是专门为XML设计的，这又增加了重新实现的复杂性。为解决这一问题，我们提出了一种基于岭回归的简单方法来进行XML学习。该方法不仅具有闭式解，还仅包含一个超参数。由于没有先例使用岭回归应用于XML任务，本文通过使用多种XML基准数据集验证了该方法的性能。此外，我们增强了对低频标签的预测，这些标签包含有价值的信息。由于数据量有限，这是一项挑战性的任务。这里，我们采用了一种基于频率的权重方法。该方法与现有技术相比极大地简化了过程。实验证明，该方法可以达到与具有大量超参数的模型相似的性能，甚至在某些情况下性能更优。此外，我们发现基于频率的权重显著提高了低频标签的预测性能，而几乎不需要更改实现。所提出方法的源代码已在GitHub上提供，网址为<此网址>。 

---
# OMoE: Diversifying Mixture of Low-Rank Adaptation by Orthogonal Finetuning 

**Title (ZH)**: OMoE：通过正交微调实现的低 rank 调适多样性混合模型 

**Authors**: Jinyuan Feng, Zhiqiang Pu, Tianyi Hu, Dongmin Li, Xiaolin Ai, Huimu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10062)  

**Abstract**: Building mixture-of-experts (MoE) architecture for Low-rank adaptation (LoRA) is emerging as a potential direction in parameter-efficient fine-tuning (PEFT) for its modular design and remarkable performance. However, simply stacking the number of experts cannot guarantee significant improvement. In this work, we first conduct qualitative analysis to indicate that experts collapse to similar representations in vanilla MoE, limiting the capacity of modular design and computational efficiency. Ulteriorly, Our analysis reveals that the performance of previous MoE variants maybe limited by a lack of diversity among experts. Motivated by these findings, we propose Orthogonal Mixture-of-Experts (OMoE), a resource-efficient MoE variant that trains experts in an orthogonal manner to promote diversity. In OMoE, a Gram-Schmidt process is leveraged to enforce that the experts' representations lie within the Stiefel manifold. By applying orthogonal constraints directly to the architecture, OMoE keeps the learning objective unchanged, without compromising optimality. Our method is simple and alleviates memory bottlenecks, as it incurs minimal experts compared to vanilla MoE models. Experiments on diverse commonsense reasoning benchmarks demonstrate that OMoE can consistently achieve stable and efficient performance improvement when compared with the state-of-the-art methods while significantly reducing the number of required experts. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

建立低秩适应（LoRA）的专家混合架构（MoE）正逐渐被认为是参数高效微调（PEFT）的一个潜在方向，因为其模块化设计和卓越的性能。然而，简单地堆叠专家数量并不能保证显著的改进。在这项工作中，我们首先进行了定性分析，表明在传统的MoE中，专家会坍缩到相似的表示中，限制了模块化设计的容量和计算效率。进一步分析揭示，先前的MoE变体可能因其专家之间的多样性不足而受到限制。受到这些发现的启发，我们提出了一种资源高效的MoE变体——正交专家混合架构（OMoE），通过以正交方式训练专家以促进多样性。在OMoE中，我们利用Gram-Schmidt过程确保专家的表示位于史泰福流形上。通过直接将正交约束应用到架构中，OMoE保持了学习目标不变，而不牺牲最优性。我们的方法简单，并且可以减轻内存瓶颈，因为它所需的专家数量远少于传统的MoE模型。在多种常识推理基准上的实验表明，当与最先进的方法进行比较时，OMoE可以持续实现稳定且高效的性能提升，同时显著减少了所需的专家数量。 

---
# RichSpace: Enriching Text-to-Video Prompt Space via Text Embedding Interpolation 

**Title (ZH)**: RichSpace: 通过文本嵌入插值丰富文本到视频提示空间 

**Authors**: Yuefan Cao, Chengyue Gong, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.09982)  

**Abstract**: Text-to-video generation models have made impressive progress, but they still struggle with generating videos with complex features. This limitation often arises from the inability of the text encoder to produce accurate embeddings, which hinders the video generation model. In this work, we propose a novel approach to overcome this challenge by selecting the optimal text embedding through interpolation in the embedding space. We demonstrate that this method enables the video generation model to produce the desired videos. Additionally, we introduce a simple algorithm using perpendicular foot embeddings and cosine similarity to identify the optimal interpolation embedding. Our findings highlight the importance of accurate text embeddings and offer a pathway for improving text-to-video generation performance. 

**Abstract (ZH)**: 文本到视频生成模型在多个方面取得了显著进展，但在生成具有复杂特征的视频方面仍然存在挑战。这一限制通常是由于文本编码器无法生成准确的嵌入，从而阻碍了视频生成模型的表现。本文提出了一种新颖的方法，通过在嵌入空间中进行插值得到最优的文本嵌入以克服这一挑战。我们展示了这种方法能够使视频生成模型生成所需的视频。此外，我们还引入了一种简单算法，使用垂直脚嵌入和余弦相似性来识别最优插值得嵌入。我们的研究结果强调了准确文本嵌入的重要性，并提供了提高文本到视频生成性能的一种途径。 

---
# Sympathy over Polarization: A Computational Discourse Analysis of Social Media Posts about the July 2024 Trump Assassination Attempt 

**Title (ZH)**: 共情超越极化：对2024年7月特朗普 assassination 尝试相关社交媒体帖子的计算话语分析

注意：在正式的学术论文中，涉及到具体的事件和人员名称时，应确保使用准确和正式的表述。在此示例中，“assassination”翻译为“assassination 尝试”，但需要根据具体上下文进行调整，确保符合学术规范和相关敏感性要求。此外，“特朗普”的译名应使用其中文习惯译名“特朗普”。如果相关事件和人物名称有特定的学术表述或官方翻译，应遵照相关规定。 

**Authors**: Qingcheng Zeng, Guanhong Liu, Zhaoqian Xue, Diego Ford, Rob Voigt, Loni Hagen, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.09950)  

**Abstract**: On July 13, 2024, at the Trump rally in Pennsylvania, someone attempted to assassinate Republican Presidential Candidate Donald Trump. This attempt sparked a large-scale discussion on social media. We collected posts from X (formerly known as Twitter) one week before and after the assassination attempt and aimed to model the short-term effects of such a ``shock'' on public opinions and discussion topics. Specifically, our study addresses three key questions: first, we investigate how public sentiment toward Donald Trump shifts over time and across regions (RQ1) and examine whether the assassination attempt itself significantly affects public attitudes, independent of the existing political alignments (RQ2). Finally, we explore the major themes in online conversations before and after the crisis, illustrating how discussion topics evolved in response to this politically charged event (RQ3). By integrating large language model-based sentiment analysis, difference-in-differences modeling, and topic modeling techniques, we find that following the attempt the public response was broadly sympathetic to Trump rather than polarizing, despite baseline ideological and regional disparities. 

**Abstract (ZH)**: 2024年7月13日，在宾夕法尼亚州的特朗普集会上，有人试图刺杀时任共和党总统候选人唐纳德·特朗普。此次刺杀企图引发了广泛的社交媒体讨论。我们收集了该刺杀企图发生前一周和后一周在X（原名Twitter）上的帖子，旨在研究这种“冲击”对公众意见和讨论主题的短期影响。具体来说，我们的研究重点回答了三个关键问题：首先，我们调查了公众对唐纳德·特朗普的态度随着时间推移和地区变化而如何转变（研究问题1），并探讨刺杀企图本身是否显著影响了公众态度，且不考虑现有的政治倾向（研究问题2）。最后，我们探索了危机前后在线讨论的主要主题，展示了讨论主题如何在这一政治敏感事件的推动下演变（研究问题3）。通过结合基于大型语言模型的情感分析、双重差分模型和主题建模技术，我们发现，在刺杀企图发生后，公众对特朗普的反应总体上表现出同情而非两极分化，尽管存在初始的意识形态和地区差异。 

---
# Steering Large Language Models with Feature Guided Activation Additions 

**Title (ZH)**: 使用特征导向激活添加来引导大型语言模型 

**Authors**: Samuel Soo, Wesley Teng, Chandrasekaran Balaganesh  

**Link**: [PDF](https://arxiv.org/pdf/2501.09929)  

**Abstract**: Effective and reliable control over large language model (LLM) behavior is a significant challenge. While activation steering methods, which add steering vectors to a model's hidden states, are a promising approach, existing techniques often lack precision and interpretability in how they influence model outputs. We introduce Feature Guided Activation Additions (FGAA), a novel activation steering method that leverages insights from Contrastive Activation Addition (CAA) and Sparse Autoencoder-Targeted Steering (SAE-TS). By operating in the latent space of a Sparse Autoencoder (SAE) and employing optimization techniques to select desired SAE features, FGAA constructs precise steering vectors that provide better steering effects while maintaining coherence of steered model outputs. In this regard, evaluations on Gemma-2-2B and Gemma-2-9B models across various steering tasks demonstrate that FGAA outperforms existing steering methods of CAA, SAE decoder steering, and SAE-TS. Our results also highlight important trade-offs between steering scale and general model capabilities that are consistent across all tested steering methods. 

**Abstract (ZH)**: 对大型语言模型（LLM）行为的有效且可靠的控制是一个重大挑战。虽然激活引导方法——通过向模型的隐状态添加引导向量——是一种有希望的解决方案，但现有技术往往在如何影响模型输出方面缺乏精确性和可解释性。我们提出了一种名为特征引导激活添加（FGAA）的新颖激活引导方法，该方法结合了对比激活添加（CAA）和稀疏自动编码器目标导向引导（SAE-TS）的见解。FGAA在稀疏自动编码器（SAE）的潜在空间中操作，并使用优化技术选择所需的SAE特征，从而构建出能够提供更精确引导效果的同时保持引导后模型输出一致性的引导向量。在这点上，对Gemma-2-2B和Gemma-2-9B模型在多种引导任务上的评估表明，FGAA在各种引导方法中表现更优。此外，我们的研究结果还凸显了所有测试方法中引导规模与通用模型能力之间的重要权衡关系。 

---
# Enhancing Generalization in Chain of Thought Reasoning for Smaller Models 

**Title (ZH)**: 提升较小模型在链式思维推理中的泛化能力 

**Authors**: Maxwell J. Yin, Dingyi Jiang, Yongbing Chen, Boyu Wang, Charles Ling  

**Link**: [PDF](https://arxiv.org/pdf/2501.09804)  

**Abstract**: Chain-of-Thought (CoT) reasoning in smaller language models is a challenging natural language process problem yet highly desirable in many real-life applications. Existing CoT knowledge distillation methods often suffer from overly conservative memorization in smaller LLMs, leading to low generalization confidence. As fully preserving the CoT ability of teacher model is impossible, we hypothesize that adversarial CoT fine-tuning is crucial for developing smaller LLM with robust CoT generalization. To this end, we propose \textit{PRompt-Assisted Domain-Adversarial fine-tuning} (PRADA), a principled fine-tuning framework that integrates diverse CoT domains. Specifically, PRADA pioneers two CoT improvements in smaller LLM: (1) Recovering the domain-invariant feature insight which typically lost during distillation with domain adversarial fine-tuning; (2) Enhancing the domain adaptability of CoT prompt engineering by employing domain-adversarial approaches. We theoretically demonstrate the effectiveness of our approach and empirically show that it significantly outperforms the state of the arts in a wide range of tasks. Moreover, our empirical findings reveal that the smaller LLM, when leveraging PRADA, aligns closely with domain knowledge, thereby improving the explainability of our approach. 

**Abstract (ZH)**: 在较小的语言模型中进行链式思考（Chain-of-Thought, CoT）推理是一个具有挑战性的自然语言处理问题，但在许多实际应用中需求很高。现有的一些CoT知识蒸馏方法往往在较小的大型语言模型（LLM）中表现出过度保守的记忆方式，导致泛化能力较低。鉴于完全保留教师模型的CoT能力是不可能的，我们假设对抗性CoT微调对于开发具有稳健CoT泛化能力的较小的LLM至关重要。为实现这一目标，我们提出了一种名为PRompt-Assisted Domain-Adversarial fine-tuning (PRADA) 的微调框架，该框架结合了多种CoT领域。具体而言，PRADA 在较小的LLM 中为CoT带来了两大改进：（1）通过领域对抗性微调恢复通常在蒸馏过程中丢失的领域不变特征见解；（2）通过采用领域对抗性方法增强CoT提示工程的领域适应性。我们从理论上证明了该方法的有效性，并通过实验发现，在广泛的任务中，这种方法显著优于现有方法。此外，我们的实验结果表明，当较小的LLM 利用PRADA 时，能够紧密匹配领域知识，从而提高了方法的可解释性。 

---
# Conversational Text Extraction with Large Language Models Using Retrieval-Augmented Systems 

**Title (ZH)**: 使用检索增强系统的大语言模型对话文本提取 

**Authors**: Soham Roy, Mitul Goswami, Nisharg Nargund, Suneeta Mohanty, Prasant Kumar Pattnaik  

**Link**: [PDF](https://arxiv.org/pdf/2501.09801)  

**Abstract**: This study introduces a system leveraging Large Language Models (LLMs) to extract text and enhance user interaction with PDF documents via a conversational interface. Utilizing Retrieval-Augmented Generation (RAG), the system provides informative responses to user inquiries while highlighting relevant passages within the PDF. Upon user upload, the system processes the PDF, employing sentence embeddings to create a document-specific vector store. This vector store enables efficient retrieval of pertinent sections in response to user queries. The LLM then engages in a conversational exchange, using the retrieved information to extract text and generate comprehensive, contextually aware answers. While our approach demonstrates competitive ROUGE values compared to existing state-of-the-art techniques for text extraction and summarization, we acknowledge that further qualitative evaluation is necessary to fully assess its effectiveness in real-world applications. The proposed system gives competitive ROUGE values as compared to existing state-of-the-art techniques for text extraction and summarization, thus offering a valuable tool for researchers, students, and anyone seeking to efficiently extract knowledge and gain insights from documents through an intuitive question-answering interface. 

**Abstract (ZH)**: 本研究介绍了一种利用大型语言模型（LLMs）的系统，通过对话界面提取文本并增强用户与PDF文档的交互。该系统利用检索增强生成（RAG）技术，能够对用户的查询提供信息性的回应，并突出显示PDF中的相关段落。用户上传PDF文档后，系统对其进行处理，使用句子嵌入创建特定文档的向量库。该向量库能够在接收到用户查询时高效地检索相关部分。随后，大型语言模型参与对话交流，利用检索到的信息提取文本并生成全面且上下文相关的答案。尽管我们的方法在文本抽取与总结方面与现有的先进技术相比展现了竞争力的ROUGE值，但我们认识到还需要进一步的定性评估来全面评估其在实际应用中的有效性。所提出的系统在文本抽取与总结方面与现有的先进技术相比展示了竞争性的ROUGE值，从而为研究人员、学生以及希望通过直观的问答接口高效地从文档中提取知识和获得洞察的人提供了一个有 value 的工具。 

---
# Computing Optimization-Based Prompt Injections Against Closed-Weights Models By Misusing a Fine-Tuning API 

**Title (ZH)**: 通过误用微调API来针对封闭权重模型进行基于优化的提示注入的计算方法 

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes  

**Link**: [PDF](https://arxiv.org/pdf/2501.09798)  

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks. 

**Abstract (ZH)**: 我们揭示了一种新的威胁，针对封闭权重的大语言模型（LLMs），使攻击者能够计算基于优化的提示注入。具体来说，我们阐述了攻击者如何利用远程微调接口返回的似损失信息来指导对抗性提示的搜索。该微调接口由LLM供应商托管，允许开发人员将LLM微调以适应其特定任务，从而提供了实用性，但也暴露了足够的信息，使攻击者能够计算对抗性提示。通过实验分析，我们描述了Gemini微调API返回的似损失值，并证明它们为使用贪婪搜索算法对对抗性提示进行离散优化提供了有用的信号。通过使用PurpleLlama提示注入基准测试，我们在Google的Gemini系列LLMs上实现了65%到82%的攻击成功率。这些攻击利用了经典的实用性和安全性权衡：微调接口为开发人员提供了有用的特性，但也使LLMs暴露于强大的攻击之下。 

---
