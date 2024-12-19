# TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks 

**Title (ZH)**: 《AgentCompany：评估大型语言模型代理在具重要性的现实任务上的表现》 

**Authors**: Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2412.14161)  

**Abstract**: We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at helping to accelerate or even autonomously perform work-related tasks? The answer to this question has important implications for both industry looking to adopt AI into their workflows, and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper, we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that with the most competitive agent, 24% of the tasks can be completed autonomously. This paints a nuanced picture on task automation with LM agents -- in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. 

**Abstract (ZH)**: 我们每天都在与计算机进行交互，无论是在日常生活中还是工作中，许多工作都可以通过计算机和互联网来完成。与此同时，由于大型语言模型（LLMs）的进步，也出现了一批可以与环境互动并影响环境变化的AI代理。那么这些AI代理在加速甚至自主完成工作相关任务方面表现如何呢？这个问题的答案对于希望将AI应用于工作流程的行业以及了解AI采用对劳动力市场可能产生的影响的经济政策都具有重要意义。为了评估这些能够以类似于数字工人方式与世界互动的LLM代理在执行真实世界专业任务方面的表现，本文引入了TheAgentCompany，这是一个可扩展的基准测试，用于评估能够以类似数字工人方式与世界进行互动的AI代理。我们构建了一个自包含的环境，其中包括模拟小型软件公司环境的内部网站和数据，并创建了各种任务，这些任务可能是该公司员工可以执行的任务。我们测试了基于封闭API和开源权重语言模型（LMs）的基线代理，并发现使用最优秀的代理，有24%的任务可以自主完成。这为我们展示了使用LM代理进行任务自动化的复杂性——在模拟实际工作场所的环境中，许多简单任务可以自行解决，但更复杂、长期的任务仍然超出了当前系统的处理范围。 

---
# GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking 

**Title (ZH)**: GLIDER：使用可解释的排名评估大语言模型的交互与决策 

**Authors**: Darshan Deshpande, Selvan Sunitha Ravi, Sky CH-Wang, Bartosz Mielczarek, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2412.14140)  

**Abstract**: The LLM-as-judge paradigm is increasingly being adopted for automated evaluation of model outputs. While LLM judges have shown promise on constrained evaluation tasks, closed source LLMs display critical shortcomings when deployed in real world applications due to challenges of fine grained metrics and explainability, while task specific evaluation models lack cross-domain generalization. We introduce GLIDER, a powerful 3B evaluator LLM that can score any text input and associated context on arbitrary user defined criteria. GLIDER shows higher Pearson's correlation than GPT-4o on FLASK and greatly outperforms prior evaluation models, achieving comparable performance to LLMs 17x its size. GLIDER supports fine-grained scoring, multilingual reasoning, span highlighting and was trained on 685 domains and 183 criteria. Extensive qualitative analysis shows that GLIDER scores are highly correlated with human judgments, with 91.3% human agreement. We have open-sourced GLIDER to facilitate future research. 

**Abstract (ZH)**: 以下是对给定内容的学术规范翻译：

大语言模型作为裁判的范式越来越多地被采用，用于自动评估模型输出。虽然大语言模型裁判在受限的评估任务中表现出潜力，但由于细粒度指标和可解释性方面的挑战，闭源的大语言模型在实际应用场景中存在关键缺陷，而针对特定任务的评估模型缺乏跨领域的泛化能力。我们引入了GLIDER，这是一种强大的30亿参数的评估大语言模型，能够根据任意用户定义的指标对任意文本输入及其相关背景进行评分。GLIDER在FLASK上的皮尔逊相关系数高于GPT-4o，并在多个评估指标上显著优于先前的评估模型，其性能相当于规模为其17倍的大型语言模型。GLIDER 支持细粒度评分、多语言推理、段落高亮，并且基于685个领域和183个评估标准进行训练。广泛的定性分析表明，GLIDER的评分与人类判断高度相关，91.3%的人类一致性。我们已开源GLIDER，以促进未来的研究。 

---
# Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models 

**Title (ZH)**: 视觉语言模型中跨模态实体知识提取的性能差距 

**Authors**: Ido Cohen, Daniela Gottesman, Mor Geva, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2412.14133)  

**Abstract**: Vision-language models (VLMs) excel at extracting and reasoning about information from images. Yet, their capacity to leverage internal knowledge about specific entities remains underexplored. This work investigates the disparity in model performance when answering factual questions about an entity described in text versus depicted in an image. Our results reveal a significant accuracy drop --averaging 19%-- when the entity is presented visually instead of textually. We hypothesize that this decline arises from limitations in how information flows from image tokens to query tokens. We use mechanistic interpretability tools to reveal that, although image tokens are preprocessed by the vision encoder, meaningful information flow from these tokens occurs only in the much deeper layers. Furthermore, critical image processing happens in the language model's middle layers, allowing few layers for consecutive reasoning, highlighting a potential inefficiency in how the model utilizes its layers for reasoning. These insights shed light on the internal mechanics of VLMs and offer pathways for enhancing their reasoning capabilities. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在从图像中提取和推理信息方面表现出色。然而，它们利用特定实体内部知识的能力仍然没有得到充分探索。本研究探讨了当对描述在文本中的实体与描述在图像中的实体提出事实性问题时，模型性能之间的差异。我们的结果揭示了当实体以视觉形式呈现时，模型的准确性显著下降，平均下降了19%。我们假设这种下降源于图像标记到查询标记之间信息流的限制。我们使用机制可解释性工具发现，尽管图像标记被视觉编码器预处理，但有意义的信息流动仅发生在更深的层中。此外，关键的图像处理发生在语言模型的中间层，这使得连续推理主要依靠少数几层，突显了模型在利用其层进行推理方面的潜在低效性。这些见解揭示了VLMs的内部机制，并为增强其推理能力提供了途径。 

---
# SEKE: Specialised Experts for Keyword Extraction 

**Title (ZH)**: SEKE：专门的专家关键词提取 

**Authors**: Matej Martinc, Hanh Thi Hong Tran, Senja Pollak, Boshko Koloski  

**Link**: [PDF](https://arxiv.org/pdf/2412.14087)  

**Abstract**: Keyword extraction involves identifying the most descriptive words in a document, allowing automatic categorisation and summarisation of large quantities of diverse textual data. Relying on the insight that real-world keyword detection often requires handling of diverse content, we propose a novel supervised keyword extraction approach based on the mixture of experts (MoE) technique. MoE uses a learnable routing sub-network to direct information to specialised experts, allowing them to specialize in distinct regions of the input space. SEKE, a mixture of Specialised Experts for supervised Keyword Extraction, uses DeBERTa as the backbone model and builds on the MoE framework, where experts attend to each token, by integrating it with a recurrent neural network (RNN), to allow successful extraction even on smaller corpora, where specialisation is harder due to lack of training data. The MoE framework also provides an insight into inner workings of individual experts, enhancing the explainability of the approach. We benchmark SEKE on multiple English datasets, achieving state-of-the-art performance compared to strong supervised and unsupervised baselines. Our analysis reveals that depending on data size and type, experts specialize in distinct syntactic and semantic components, such as punctuation, stopwords, parts-of-speech, or named entities. Code is available at: this https URL 

**Abstract (ZH)**: 关键词提取涉及识别文档中最具描述性的词汇，从而实现对大量异质文本数据的自动分类和总结。鉴于现实世界中的关键词检测经常需要处理多样化的内容，我们提出了一种基于混合专家（MoE）技术的新型监督关键词提取方法。MoE 使用可学习的路由子网络将信息引导至专门化的专家，使他们能够在输入空间的不同区域专注于不同的任务。SEKE（Specialised Experts for Supervised Keyword Extraction），一种基于MoE框架的混合专家模型，使用DeBERTa作为主干模型，并将其与循环神经网络（RNN）结合，以允许在较小的数据集上实现成功的关键词提取，尽管在这种情况下由于缺乏训练数据使专门化变得更具挑战性。MoE框架还为个体专家的内部工作提供了洞见，增强了该方法的可解释性。我们在多个英文数据集上对SEKE进行了基准测试，其性能优于强监督和无监督基线方法。我们的分析表明，根据数据大小和类型，专家可以专注于不同的语法和语义成分，如标点符号、停用词、词性或命名实体。代码可在以下链接获取：[](https://) 

---
# Digestion Algorithm in Hierarchical Symbolic Forests: A Fast Text Normalization Algorithm and Semantic Parsing Framework for Specific Scenarios and Lightweight Deployment 

**Title (ZH)**: 分层符号森林中的消化算法：一种针对特定场景的快速文本规范化算法和轻量级部署语义解析框架 

**Authors**: Kevin You  

**Link**: [PDF](https://arxiv.org/pdf/2412.14054)  

**Abstract**: Text Normalization and Semantic Parsing have numerous applications in natural language processing, such as natural language programming, paraphrasing, data augmentation, constructing expert systems, text matching, and more. Despite the prominent achievements of deep learning in Large Language Models (LLMs), the interpretability of neural network architectures is still poor, which affects their credibility and hence limits the deployments of risk-sensitive scenarios. In certain scenario-specific domains with scarce data, rapidly obtaining a large number of supervised learning labels is challenging, and the workload of manually labeling data would be enormous. Catastrophic forgetting in neural networks further leads to low data utilization rates. In situations where swift responses are vital, the density of the model makes local deployment difficult and the response time long, which is not conducive to local applications of these fields. Inspired by the multiplication rule, a principle of combinatorial mathematics, and human thinking patterns, a multilayer framework along with its algorithm, the Digestion Algorithm in Hierarchical Symbolic Forests (DAHSF), is proposed to address these above issues, combining text normalization and semantic parsing workflows. The Chinese Scripting Language "Fire Bunny Intelligent Development Platform V2.0" is an important test and application of the technology discussed in this paper. DAHSF can run locally in scenario-specific domains on little datasets, with model size and memory usage optimized by at least two orders of magnitude, thus improving the execution speed, and possessing a promising optimization outlook. 

**Abstract (ZH)**: 文本规范化和语义解析在自然语言处理中有广泛的应用，如自然语言编程、改写、数据增强、构建专家系统、文本匹配等。尽管大型语言模型（LLMs）在深度学习方面取得了显著成就，但神经网络架构的可解释性仍然较差，这影响了它们的可信度，从而限制了其在风险敏感场景中的应用。在某些特定领域，由于数据稀缺，快速获取大量监督学习标签是一项挑战，手工标注数据的工作量非常巨大。神经网络中的灾难性遗忘进一步导致了数据利用率的低下。在需要迅速响应的情景下，模型的稠密性使得本地部署困难，响应时间较长，这不利于这些领域的本地应用。

受到乘法原理、组合数学原则以及人类思维模式的启发，本文提出了一个多层框架及其相应的算法——层次符号森林中的消化算法（DAHSF），以解决上述问题，同时结合了文本规范化和语义解析的工作流程。中文脚本语言“火兔智能开发平台V2.0”是本文所讨论技术的一个重要测试和应用实例。DAHSF可以在特定领域的小数据集上本地运行，并且通过至少两个数量级的模型大小和内存使用优化，提升了执行速度，展现出优化的潜在前景。 

---
# Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation 

**Title (ZH)**: 多语言LLM中消偏性和解毒性的跨语言迁移：一项全面调查 

**Authors**: Vera Neplenbroek, Arianna Bisazza, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2412.14050)  

**Abstract**: Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. Our results show that finetuning on curated non-harmful text is more effective for mitigating bias, and finetuning on direct preference optimization (DPO) datasets is more effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods. 

**Abstract (ZH)**: 近年来，生成型大型语言模型（LLMs）在非英语语言中表现出色，但在用这些语言进行提示时，往往会表现出更高的有害社会偏差和毒性水平。此前的研究表明，通过专门数据集进行微调可以减轻这种行为，并且在英语上的微调可以转移到其他语言上。在这项工作中，我们调查了不同微调方法对模型偏差和毒性的影响，同时也考察了其生成流畅且多样化文本的能力。我们的结果表明，使用非有害文本进行微调更有效地减轻了偏差，而使用直接偏好优化（DPO）数据集进行微调则更有效地减轻了毒性。应用这些方法在英语上减轻的偏差和毒性也转移到了非英语语言上。我们发现，这种转移的程度可以通过模型在预训练数据中所含特定语言的数据量来预测。然而，这种方法在减轻偏见和毒性方面的转移往往以非英语语言的生成能力下降为代价，突显了开发特定语言的偏见和毒性缓解方法的重要性。 

---
# Hansel: Output Length Controlling Framework for Large Language Models 

**Title (ZH)**: 汉塞尔：大型语言模型的输出长度控制框架 

**Authors**: Seoha Song, Junhyun Lee, Hyeonmok Ko  

**Link**: [PDF](https://arxiv.org/pdf/2412.14033)  

**Abstract**: Despite the great success of large language models (LLMs), efficiently controlling the length of the output sequence still remains a challenge. In this paper, we propose Hansel, an efficient framework for length control in LLMs without affecting its generation ability. Hansel utilizes periodically outputted hidden special tokens to keep track of the remaining target length of the output sequence. Together with techniques to avoid abrupt termination of the output, this seemingly simple method proved to be efficient and versatile, while not harming the coherency and fluency of the generated text. The framework can be applied to any pre-trained LLMs during the finetuning stage of the model, regardless of its original positional encoding method. We demonstrate this by finetuning four different LLMs with Hansel and show that the mean absolute error of the output sequence decreases significantly in every model and dataset compared to the prompt-based length control finetuning. Moreover, the framework showed a substantially improved ability to extrapolate to target lengths unseen during finetuning, such as long dialog responses or extremely short summaries. This indicates that the model learns the general means of length control, rather than learning to match output lengths to those seen during training. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了巨大的成功，有效地控制输出序列的长度仍然是一项挑战。本文提出了一种名为Hansel的高效框架，可以在不损害LLMs生成能力的情况下控制其输出长度。Hansel利用周期性输出的隐藏特殊标记来跟踪输出序列剩余的目标长度。结合避免输出突然终止的技术，这种方法看似简单，但证明了其高效性和灵活性，同时保持了生成文本的连贯性和流畅性。该框架可以在模型的微调阶段应用于任何预训练的LLMs，而不受其原始位置编码方法的限制。我们通过使用Hansel微调四个不同的LLMs进行了演示，并展示了在每个模型和数据集中，输出序列的绝对误差显著减少，这表明与基于提示的长度控制微调相比，该框架能够显著提高模型将输出长度外推到微调时未见过的目标长度（如长对话响应或极短的摘要）的能力。这表明模型学会了长度控制的一般方法，而不是仅仅学习匹配特定的输出长度。 

---
# Towards an optimised evaluation of teachers' discourse: The case of engaging messages 

**Title (ZH)**: 优化教师话语评估的途径：参与性信息案例分析 

**Authors**: Samuel Falcon, Jaime Leon  

**Link**: [PDF](https://arxiv.org/pdf/2412.14011)  

**Abstract**: Evaluating teachers' skills is crucial for enhancing education quality and student outcomes. Teacher discourse, significantly influencing student performance, is a key component. However, coding this discourse can be laborious. This study addresses this issue by introducing a new methodology for optimising the assessment of teacher discourse. The research consisted of two studies, both within the framework of engaging messages used by secondary education teachers. The first study involved training two large language models on real-world examples from audio-recorded lessons over two academic years to identify and classify the engaging messages from the lessons' transcripts. This resulted in sensitivities of 84.31% and 91.11%, and specificities of 97.69% and 86.36% in identification and classification, respectively. The second study applied these models to transcripts of audio-recorded lessons from a third academic year to examine the frequency and distribution of message types by educational level and moment of the academic year. Results showed teachers predominantly use messages emphasising engagement benefits, linked to improved outcomes, while one-third highlighted non-engagement disadvantages, associated with increased anxiety. The use of engaging messages declined in Grade 12 and towards the academic year's end. These findings suggest potential interventions to optimise engaging message use, enhancing teaching quality and student outcomes. 

**Abstract (ZH)**: 评价教师技能对于提升教育质量和学生成果至关重要。教师的语言交流对学生产生显著影响，是关键组成部分之一。然而，对这种交流进行编码可能会非常耗费人力。本研究通过引入一种新的方法来优化教师语言交流的评估，解决了这一问题。研究包括两个部分，均以中学教师使用的吸引性信息为框架展开。第一部分研究通过在两年的学术周期内，利用真实的课堂录音示例对两个大型语言模型进行训练，来识别和分类课堂录音转录中的吸引性信息。结果显示，在识别和分类阶段的灵敏度分别为84.31%和91.11%，特异性分别为97.69%和86.36%。第二部分研究将这些模型应用于第三学年课堂录音的转录，以探讨不同教育水平和学年阶段的信息类型频率和分布。结果显示，教师主要是使用强调吸引性益处的信息，这些信息与改善的学生成果相关，而三分之一的信息则提到了不吸引性的不利影响，与增加的焦虑情绪相关。吸引性信息的使用在12年级和学年末有所下降。这些发现表明可能采取干预措施以优化吸引性信息的使用，从而提高教学质量和学生成果。 

---
# FarExStance: Explainable Stance Detection for Farsi 

**Title (ZH)**: FarExStance: 可解释的波斯语立场检测 

**Authors**: Majid Zarharan, Maryam Hashemi, Malika Behroozrazegh, Sauleh Eetemadi, Mohammad Taher Pilehvar, Jennifer Foster  

**Link**: [PDF](https://arxiv.org/pdf/2412.14008)  

**Abstract**: We introduce FarExStance, a new dataset for explainable stance detection in Farsi. Each instance in this dataset contains a claim, the stance of an article or social media post towards that claim, and an extractive explanation which provides evidence for the stance label. We compare the performance of a fine-tuned multilingual RoBERTa model to several large language models in zero-shot, few-shot, and parameter-efficient fine-tuned settings on our new dataset. On stance detection, the most accurate models are the fine-tuned RoBERTa model, the LLM Aya-23-8B which has been fine-tuned using parameter-efficient fine-tuning, and few-shot Claude-3.5-Sonnet. Regarding the quality of the explanations, our automatic evaluation metrics indicate that few-shot GPT-4o generates the most coherent explanations, while our human evaluation reveals that the best Overall Explanation Score (OES) belongs to few-shot Claude-3.5-Sonnet. The fine-tuned Aya-32-8B model produced explanations most closely aligned with the reference explanations. 

**Abstract (ZH)**: 我们介绍了一个新的数据集 FarExStance，用于波斯语中的可解释立场检测。该数据集中的每个实例包含一个断言、一篇文章或社交媒体帖子对该断言的立场，以及一个提取性的解释，该解释提供了支持立场标签的证据。我们在该新数据集上将调优的多语言 RoBERTa 模型的性能与几个大型语言模型在零样本、少量样本和参数高效调优设置下的表现进行了比较。在立场检测方面，最准确的模型是调优的 RoBERTa 模型、使用参数高效调优进行调优的 Aya-23-8B 大型语言模型以及少量样本的 Claude-3.5-Sonnet。关于解释的质量，我们的自动评估指标表明少量样本的 GPT-4o 生成最连贯的解释，而我们的手动评估揭示少量样本的 Claude-3.5-Sonnet 在总体解释分数（OES）方面表现最佳。调优的 Aya-32-8B 模型生成的解释与参考解释最为一致。 

---
# What makes a good metric? Evaluating automatic metrics for text-to-image consistency 

**Title (ZH)**: 什么是好的评估指标？评估文本到图像一致性自动评估指标的性能 

**Authors**: Candace Ross, Melissa Hall, Adriana Romero Soriano, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2412.13989)  

**Abstract**: Language models are increasingly being incorporated as components in larger AI systems for various purposes, from prompt optimization to automatic evaluation. In this work, we analyze the construct validity of four recent, commonly used methods for measuring text-to-image consistency - CLIPScore, TIFA, VPEval, and DSG - which rely on language models and/or VQA models as components. We define construct validity for text-image consistency metrics as a set of desiderata that text-image consistency metrics should have, and find that no tested metric satisfies all of them. We find that metrics lack sufficient sensitivity to language and visual properties. Next, we find that TIFA, VPEval and DSG contribute novel information above and beyond CLIPScore, but also that they correlate highly with each other. We also ablate different aspects of the text-image consistency metrics and find that not all model components are strictly necessary, also a symptom of insufficient sensitivity to visual information. Finally, we show that all three VQA-based metrics likely rely on familiar text shortcuts (such as yes-bias in QA) that call their aptitude as quantitative evaluations of model performance into question. 

**Abstract (ZH)**: 语言模型 increasingly 被整合进各种目的的大型 AI 系统中，从提示优化到自动评估。在本文中，我们分析了四种用于测量文本-图像一致性并常用的方法——CLIPScore、TIFA、VPEval 和 DSG——的构造效度，这些方法依赖于语言模型和/或视觉问答（VQA）模型。我们定义了文本-图像一致性度量的构造效度为一组度量应具备的要求，并发现所有测试的度量均未满足全部要求。我们发现这些度量缺乏对语言和视觉属性的足够敏感性。接着，我们发现 TIFA、VPEval 和 DSG 在提供新颖信息方面超越了 CLIPScore，但它们之间也存在高度的关联性。我们还对文本-图像一致性度量的不同方面进行了分析，发现并非所有模型组件都是不可或缺的，这也反映了对视觉信息敏感度不足的症状。最后，我们表明，所有基于 VQA 的度量很可能依赖于常见的文本捷径（如 QA 中的肯定偏见），这对其作为模型性能定量评估工具的有效性提出了质疑。 

---
# Prompting Strategies for Enabling Large Language Models to Infer Causation from Correlation 

**Title (ZH)**: 促进大型语言模型从相关性推断因果关系的提示策略 

**Authors**: Eleni Sgouritsa, Virginia Aglietti, Yee Whye Teh, Arnaud Doucet, Arthur Gretton, Silvia Chiappa  

**Link**: [PDF](https://arxiv.org/pdf/2412.13952)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) are attracting increasing attention. In this work, we focus on causal reasoning and address the task of establishing causal relationships based on correlation information, a highly challenging problem on which several LLMs have shown poor performance. We introduce a prompting strategy for this problem that breaks the original task into fixed subquestions, with each subquestion corresponding to one step of a formal causal discovery algorithm, the PC algorithm. The proposed prompting strategy, PC-SubQ, guides the LLM to follow these algorithmic steps, by sequentially prompting it with one subquestion at a time, augmenting the next subquestion's prompt with the answer to the previous one(s). We evaluate our approach on an existing causal benchmark, Corr2Cause: our experiments indicate a performance improvement across five LLMs when comparing PC-SubQ to baseline prompting strategies. Results are robust to causal query perturbations, when modifying the variable names or paraphrasing the expressions. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力正越来越引起人们的关注。在这项工作中，我们专注于因果推理，并处理基于相关性信息建立因果关系的任务，这是一个高度具有挑战性的问题，多个LLMs在此任务上的表现欠佳。为此，我们提出了一种针对该问题的提示策略，将原始任务分解为固定子问题，每个子问题对应形式化的因果发现算法（如PC算法）的一个步骤。我们提出的提示策略PC-SubQ通过依次向LLM提出一个子问题，并在后续子问题的提示中加入上一子问题的答案来引导LLM遵循这些算法步骤。我们在现有因果基准Corr2Cause上评估了我们的方法：实验结果表明，在五个LLM中，PC-SubQ的性能优于基线提示策略。即使对因果查询进行修改（如变更变量名称或重述表达），这种性能提升也是稳健的。 

---
# Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence 

**Title (ZH)**: 基于视觉aware头部发散的LVLMs幻觉机制破解 

**Authors**: Jinghan He, Kuan Zhu, Haiyun Guo, Junfeng Fang, Zhenglin Hua, Yuheng Jia, Ming Tang, Tat-Seng Chua, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13949)  

**Abstract**: Large vision-language models (LVLMs) have made substantial progress in integrating large language models (LLMs) with visual inputs, enabling advanced multimodal reasoning. Despite their success, a persistent challenge is hallucination-where generated text fails to accurately reflect visual content-undermining both accuracy and reliability. Existing methods focus on alignment training or decoding refinements but primarily address symptoms at the generation stage without probing the underlying causes. In this work, we investigate the internal mechanisms driving hallucination in LVLMs, with an emphasis on the multi-head attention module. Specifically, we introduce Vision-aware Head Divergence (VHD), a metric that quantifies the sensitivity of attention head outputs to visual context. Based on this, our findings reveal the presence of vision-aware attention heads that are more attuned to visual information; however, the model's overreliance on its prior language patterns is closely related to hallucinations. Building on these insights, we propose Vision-aware Head Reinforcement (VHR), a training-free approach to mitigate hallucination by enhancing the role of vision-aware attention heads. Extensive experiments demonstrate that our method achieves superior performance compared to state-of-the-art approaches in mitigating hallucinations, while maintaining high efficiency with negligible additional time overhead. 

**Abstract (ZH)**: 大型多模态视觉-语言模型（LVLMs）在将大型语言模型（LLMs）与视觉输入相结合方面取得了显著进展，从而实现高级跨模态推理。尽管取得了成功，但持续存在的挑战是幻觉——生成的文本未能准确反映视觉内容，这直接影响了准确性和可靠性。现有方法主要集中在对齐训练或解码改进，但主要是针对生成阶段的症状，而没有探查背后的根源。在本研究中，我们调查了LVLMs中导致幻觉的内部机制，并特别关注了多头注意力模块。具体来说，我们引入了视觉感知头 divergences（VHD），这是一种度量视觉上下文对注意力头输出敏感性的指标。基于此，我们的研究结果揭示了对视觉信息更敏感的视觉感知注意力头的存在；然而，模型过度依赖其先验语言模式与幻觉密切相关。基于这些见解，我们提出了一种无需训练的方法——视觉感知头强化（VHR），通过增强视觉感知注意力头的作用来减轻幻觉。大量实验结果表明，与当前最先进的方法相比，我们的方法在减轻幻觉方面具有更优越的性能，同时保持了高效性，几乎无需额外的时间开销。 

---
# A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI 

**Title (ZH)**: 任意名称的一朵玫瑰：由大规模语言模型生成的解释是收集自然语言推理数据集中标签分布的人类解释的良好代理 

**Authors**: Beiduo Chen, Siyao Peng, Anna Korhonen, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2412.13942)  

**Abstract**: Disagreement in human labeling is ubiquitous, and can be captured in human judgment distributions (HJDs). Recent research has shown that explanations provide valuable information for understanding human label variation (HLV) and large language models (LLMs) can approximate HJD from a few human-provided label-explanation pairs. However, collecting explanations for every label is still time-consuming. This paper examines whether LLMs can be used to replace humans in generating explanations for approximating HJD. Specifically, we use LLMs as annotators to generate model explanations for a few given human labels. We test ways to obtain and combine these label-explanations with the goal to approximate human judgment distribution. We further compare the resulting human with model-generated explanations, and test automatic and human explanation selection. Our experiments show that LLM explanations are promising for NLI: to estimate HJD, generated explanations yield comparable results to human's when provided with human labels. Importantly, our results generalize from datasets with human explanations to i) datasets where they are not available and ii) challenging out-of-distribution test sets. 

**Abstract (ZH)**: 人类标注中的分歧是普遍存在的，可以体现在人类判断分布（HJD）中。近期的研究表明，解释提供了理解人类标注变异（HLV）和大规模语言模型（LLMs）可以从少量的人类标注-解释对中近似HJD的重要信息。然而，为每个标注收集解释仍然耗费时间。本文探讨了LLMs是否可以替代人类生成解释以近似HJD。具体而言，我们使用LLMs作为注释者，为给定的少量人类标注生成模型解释。我们测试获取和结合这些标注-解释的方法，以期近似人类判断分布。进一步地，我们将人类与模型生成的解释进行比较，并测试自动选择和人工选择解释的方法。我们的实验表明，在NLI任务中，生成的解释对于估计HJD而言是具有前景的：当提供人类标注时，生成的解释能获得与人类相当的结果。重要的是，我们的结果不仅适用于有人类解释的数据集，还适用于i) 没有人类解释的数据集以及ii) 具有挑战性的分布外测试集。 

---
# Language verY Rare for All 

**Title (ZH)**: “语言非常稀缺对于所有人” 

**Authors**: Ibrahim Merad, Amos Wolf, Ziad Mazzawi, Yannick Léo  

**Link**: [PDF](https://arxiv.org/pdf/2412.13924)  

**Abstract**: In the quest to overcome language barriers, encoder-decoder models like NLLB have expanded machine translation to rare languages, with some models (e.g., NLLB 1.3B) even trainable on a single GPU. While general-purpose LLMs perform well in translation, open LLMs prove highly competitive when fine-tuned for specific tasks involving unknown corpora. We introduce LYRA (Language verY Rare for All), a novel approach that combines open LLM fine-tuning, retrieval-augmented generation (RAG), and transfer learning from related high-resource languages. This study is exclusively focused on single-GPU training to facilitate ease of adoption. Our study focuses on two-way translation between French and Monégasque, a rare language unsupported by existing translation tools due to limited corpus availability. Our results demonstrate LYRA's effectiveness, frequently surpassing and consistently matching state-of-the-art encoder-decoder models in rare language translation. 

**Abstract (ZH)**: 为了克服语言障碍，如NLLB这样的编码器-解码器模型已经扩展了机器翻译到稀有语言，有些模型（例如NLLB 1.3B）甚至可以在单块GPU上进行训练。虽然通用的大型语言模型在翻译方面表现良好，但在特定任务的微调中，开源的大型语言模型在处理未知语料库时表现出极高的竞争力。我们提出了LYRA（Language verY Rare for All）这一创新方法，结合了开放的大型语言模型微调、检索增强生成（RAG）以及从相关高资源语言转移学习。本研究专注于单块GPU训练，以促进其易于采用。我们的研究集中在法语和摩纳哥语之间的双向翻译上，这种稀有语言由于语料库的有限可用性而无法被现有的翻译工具支持。我们的研究结果证明了LYRA的有效性，在稀有语言翻译方面频繁超越甚至一致性地匹配最先进的编码器-解码器模型。 

---
# Pipeline Analysis for Developing Instruct LLMs in Low-Resource Languages: A Case Study on Basque 

**Title (ZH)**: 开发低资源语言指令大规模语言模型的管道分析：关于巴斯克语的案例研究 

**Authors**: Ander Corral, Ixak Sarasua, Xabier Saralegi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13922)  

**Abstract**: Large language models (LLMs) are typically optimized for resource-rich languages like English, exacerbating the gap between high-resource and underrepresented languages. This work presents a detailed analysis of strategies for developing a model capable of following instructions in a low-resource language, specifically Basque, by focusing on three key stages: pre-training, instruction tuning, and alignment with human preferences. Our findings demonstrate that continual pre-training with a high-quality Basque corpus of around 600 million words improves natural language understanding (NLU) of the foundational model by over 12 points. Moreover, instruction tuning and human preference alignment using automatically translated datasets proved highly effective, resulting in a 24-point improvement in instruction-following performance. The resulting models, Llama-eus-8B and Llama-eus-8B-instruct, establish a new state-of-the-art for Basque in the sub-10B parameter category. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常针对资源丰富语言（如英语）进行了优化，加剧了高资源语言与低资源语言之间的差距。本文详细分析了开发一种能够在低资源语言（如巴斯克语）中跟随指令的模型的方法，重点关注三个关键阶段：预训练、指令调优和与人类偏好对齐。我们的研究结果表明，使用约6亿词的高质量巴斯克语语料库进行持续预训练可以将基础模型的自然语言理解（NLU）提高超过12个点。此外，使用自动翻译的数据集进行指令调优和人类偏好对齐效果显著，使得指令跟随性能提高了24个点。最终构建的模型，Llama-eus-8B和Llama-eus-8B-instruct，在少于10B参数的子类别中达到了巴斯克语的新最佳水平。 

---
# Understanding and Analyzing Model Robustness and Knowledge-Transfer in Multilingual Neural Machine Translation using TX-Ray 

**Title (ZH)**: 使用TX-Ray 理解和分析多语言神经机器翻译中的模型稳健性和知识迁移 

**Authors**: Vageesh Saxena, Sharid Loáiciga, Nils Rethmeier  

**Link**: [PDF](https://arxiv.org/pdf/2412.13881)  

**Abstract**: Neural networks have demonstrated significant advancements in Neural Machine Translation (NMT) compared to conventional phrase-based approaches. However, Multilingual Neural Machine Translation (MNMT) in extremely low-resource settings remains underexplored. This research investigates how knowledge transfer across languages can enhance MNMT in such scenarios. Using the Tatoeba translation challenge dataset from Helsinki NLP, we perform English-German, English-French, and English-Spanish translations, leveraging minimal parallel data to establish cross-lingual mappings. Unlike conventional methods relying on extensive pre-training for specific language pairs, we pre-train our model on English-English translations, setting English as the source language for all tasks. The model is fine-tuned on target language pairs using joint multi-task and sequential transfer learning strategies. Our work addresses three key questions: (1) How can knowledge transfer across languages improve MNMT in extremely low-resource scenarios? (2) How does pruning neuron knowledge affect model generalization, robustness, and catastrophic forgetting? (3) How can TX-Ray interpret and quantify knowledge transfer in trained models? Evaluation using BLEU-4 scores demonstrates that sequential transfer learning outperforms baselines on a 40k parallel sentence corpus, showcasing its efficacy. However, pruning neuron knowledge degrades performance, increases catastrophic forgetting, and fails to improve robustness or generalization. Our findings provide valuable insights into the potential and limitations of knowledge transfer and pruning in MNMT for extremely low-resource settings. 

**Abstract (ZH)**: 与传统的基于短语的方法相比，神经网络在神经机器翻译（NMT）方面取得了显著进步。然而，在极端资源稀缺环境下，多语言神经机器翻译（MNMT）的研究依然很匮乏。本研究探讨了不同语言之间知识迁移如何在这些场景中提升MNMT的表现。我们使用赫尔辛基NLP提供的Tatoeba翻译挑战数据集，进行英语-德语、英语-法语和英语-西班牙语的翻译任务，利用少量的平行数据来建立跨语言映射。不同于依赖于特定语言对的大量预训练常规方法，我们的模型先基于英语-英语的平行数据进行预训练，将英语作为所有任务的源语言。该模型随后通过联合多任务和序列迁移学习策略进行微调。我们的研究旨在回答三个关键问题：（1）不同语言之间知识迁移如何在极端资源稀缺场景中提升MNMT？（2）神经元知识剪枝如何影响模型泛化、鲁棒性和灾难性遗忘？（3）TX-Ray如何解释和量化训练模型中的知识迁移？使用BLEU-4评分进行评估表明，序列迁移学习在40,000个平行句子语料上优于基准方法，突显了其有效性。然而，神经元知识剪枝降低了性能，增加了灾难性遗忘，未能提高鲁棒性或泛化能力。本研究提供了关于知识迁移和剪枝在极端资源稀缺环境下MNMT中的潜力和局限性的宝贵见解。 

---
# Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings 

**Title (ZH)**: 螃蟹：在黑盒设置下通过自动生成消耗资源以对大语言模型进行DoS攻击 

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.13879)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (AutoDoS). AutoDoS introduces DoS Attack Tree and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting Length Trojan in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over 250 $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our code is available at \url{this https URL}. 

**Abstract (ZH)**: 大语言模型（LLMs）已经在多种任务中展现了出色的性能。然而，LLMs仍然容易受到外部威胁，尤其是服务拒绝攻击（DoS攻击）。具体的LLM-DoS攻击旨在耗尽计算资源并阻塞服务。尽管如此，先前的研究工作往往集中在进行白盒攻击，忽略了黑盒环境。在这项工作中，我们提出了一种专为黑盒LLMs设计的自动化算法，称为Auto-Generation for LLM-DoS Attack（AutoDoS）。AutoDoS引入了DoS攻击树，并优化了提示节点的覆盖范围，以在黑盒条件下提升攻击效果。我们的方法通过改进提示节点的语义提高了攻击的隐蔽性，从而绕过了现有的防御措施。此外，我们发现植入长度特洛伊木马（Length Trojan）到基本DoS提示中，有助于提高攻击的有效性。实验结果表明，AutoDoS可以使服务响应延迟放大超过250倍（$\uparrow 250 \times$），导致GPU使用率和内存使用率大幅增加。我们的代码可在以下链接获得：\url{this https URL}。 

---
# Domain-adaptative Continual Learning for Low-resource Tasks: Evaluation on Nepali 

**Title (ZH)**: 面向低资源任务的领域适应连续学习：以尼泊尔语为例的评估 

**Authors**: Sharad Duwal, Suraj Prasai, Suresh Manandhar  

**Link**: [PDF](https://arxiv.org/pdf/2412.13860)  

**Abstract**: Continual learning has emerged as an important research direction due to the infeasibility of retraining large language models (LLMs) from scratch in the event of new data availability. Of great interest is the domain-adaptive pre-training (DAPT) paradigm, which focuses on continually training a pre-trained language model to adapt it to a domain it was not originally trained on. In this work, we evaluate the feasibility of DAPT in a low-resource setting, namely the Nepali language. We use synthetic data to continue training Llama 3 8B to adapt it to the Nepali language in a 4-bit QLoRA setting. We evaluate the adapted model on its performance, forgetting, and knowledge acquisition. We compare the base model and the final model on their Nepali generation abilities, their performance on popular benchmarks, and run case-studies to probe their linguistic knowledge in Nepali. We see some unsurprising forgetting in the final model, but also surprisingly find that increasing the number of shots during evaluation yields better percent increases in the final model (as high as 19.29% increase) compared to the base model (4.98%), suggesting latent retention. We also explore layer-head self-attention heatmaps to establish dependency resolution abilities of the final model in Nepali. 

**Abstract (ZH)**: 持续学习已成为一个重要研究方向，由于在新数据可用时从零重新训练大规模语言模型（LLMs）是不现实的。特别引人关注的是领域自适应预训练（DAPT）范式，它侧重于持续训练一个预训练的语言模型，使其能够适应其最初未经过训练的领域。在本研究中，我们评估了在低资源设置下DAPT的可行性，特别是针对尼泊尔语。我们使用合成数据继续训练Llama 3 8B，以在4比特QLoRA设置中将其适应尼泊尔语。我们从性能、遗忘和知识获取方面评估了适应后的模型。我们将基础模型和最终模型的尼泊尔语生成能力、在流行基准上的表现进行比较，并运行案例研究以探究其在尼泊尔语中的语言知识。我们在最终模型中观察到了一些不令人惊讶的遗忘，但令人惊讶地发现，在评估过程中增加样本数量能显著提高最终模型（高达19.29%的增长）的表现，而基础模型仅增长了4.98%，这表明模型有一定的潜在保留能力。我们还探索了层头自注意力热点图，以验证最终模型在尼泊尔语中的依赖关系解决能力。 

---
# RACQUET: Unveiling the Dangers of Overlooked Referential Ambiguity in Visual LLMs 

**Title (ZH)**: RACQUET: 揭示视觉大语言模型中被忽视的指称歧义危险 

**Authors**: Alberto Testoni, Barbara Plank, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2412.13835)  

**Abstract**: Ambiguity resolution is key to effective communication. While humans effortlessly address ambiguity through conversational grounding strategies, the extent to which current language models can emulate these strategies remains unclear. In this work, we examine referential ambiguity in image-based question answering by introducing RACQUET, a carefully curated dataset targeting distinct aspects of ambiguity. Through a series of evaluations, we reveal significant limitations and problems of overconfidence of state-of-the-art large multimodal language models in addressing ambiguity in their responses. The overconfidence issue becomes particularly relevant for RACQUET-BIAS, a subset designed to analyze a critical yet underexplored problem: failing to address ambiguity leads to stereotypical, socially biased responses. Our results underscore the urgency of equipping models with robust strategies to deal with uncertainty without resorting to undesirable stereotypes. 

**Abstract (ZH)**: 有效沟通的关键在于消除歧义。人类通过会话基础策略自如地应对歧义，但当前的语言模型在模仿这些策略方面的程度仍不清楚。在本文中，我们通过引入RACQUET数据集，系统地探讨了基于图像的问题回答中的参照性歧义。通过一系列评估，我们揭示了最先进的大型多模态语言模型在其回答中应对歧义时存在的显著局限性和问题，特别是在过自信方面。RACQUET-BIAS子集专门设计用于分析一个关键且尚未充分探索的问题：未能应对歧义会导致刻板且社会偏见的回答。我们的结果强调了在模型中建立应对不确定性且不依赖于不 desirable 的刻板印象的稳健策略的紧迫性。 

---
# Enhancing Rhetorical Figure Annotation: An Ontology-Based Web Application with RAG Integration 

**Title (ZH)**: 基于本体的集成RAG技术的修辞手法标注增强：一个网络应用系统 

**Authors**: Ramona Kühn, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.13799)  

**Abstract**: Rhetorical figures play an important role in our communication. They are used to convey subtle, implicit meaning, or to emphasize statements. We notice them in hate speech, fake news, and propaganda. By improving the systems for computational detection of rhetorical figures, we can also improve tasks such as hate speech and fake news detection, sentiment analysis, opinion mining, or argument mining. Unfortunately, there is a lack of annotated data, as well as qualified annotators that would help us build large corpora to train machine learning models for the detection of rhetorical figures. The situation is particularly difficult in languages other than English, and for rhetorical figures other than metaphor, sarcasm, and irony. To overcome this issue, we develop a web application called "Find your Figure" that facilitates the identification and annotation of German rhetorical figures. The application is based on the German Rhetorical ontology GRhOOT which we have specially adapted for this purpose. In addition, we improve the user experience with Retrieval Augmented Generation (RAG). In this paper, we present the restructuring of the ontology, the development of the web application, and the built-in RAG pipeline. We also identify the optimal RAG settings for our application. Our approach is one of the first to practically use rhetorical ontologies in combination with RAG and shows promising results. 

**Abstract (ZH)**: 修辞手法在我们的交流中发挥着重要作用。它们用于传达微妙的隐含意义，或强调陈述。我们在仇恨言论、假新闻和宣传中注意到它们。通过改进计算检测修辞手法的系统，我们也可以提高仇恨言论和假新闻检测、情感分析、意见挖掘或论点挖掘等任务的性能。不幸的是，目前缺乏标注数据，以及合格的标注人员来帮助我们构建大型语料库进行机器学习模型训练以检测修辞手法。这种情况在除了英语之外的语言中尤其困难，对于除了比喻、讽刺和 irony 之外的其他修辞手法，情况更加复杂。为了解决这一问题，我们开发了一个名为“Find your Figure”的网络应用程序，以简化德语修辞手法的识别和标注。该应用程序基于我们特别为这一目的改编的德语修辞本体论GRhOOT。此外，我们通过检索增强生成（RAG）提升了用户体验。在本文中，我们介绍了本体论的重新构建、网络应用程序的开发以及内置的RAG管道。我们还确定了适用于我们应用程序的最佳RAG设置。我们的方法是第一个在实际中将修辞本体论与RAG相结合的方法，并且显示了有希望的结果。 

---
# MATCHED: Multimodal Authorship-Attribution To Combat Human Trafficking in Escort-Advertisement Data 

**Title (ZH)**: MATCHED：多模态作者归属分析以打击 escort 广告数据中的人口走私问题 

**Authors**: Vageesh Saxena, Benjamin Bashpole, Gijs Van Dijck, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2412.13794)  

**Abstract**: Human trafficking (HT) remains a critical issue, with traffickers increasingly leveraging online escort advertisements (ads) to advertise victims anonymously. Existing detection methods, including Authorship Attribution (AA), often center on text-based analyses and neglect the multimodal nature of online escort ads, which typically pair text with images. To address this gap, we introduce MATCHED, a multimodal dataset of 27,619 unique text descriptions and 55,115 unique images collected from the Backpage escort platform across seven U.S. cities in four geographical regions. Our study extensively benchmarks text-only, vision-only, and multimodal baselines for vendor identification and verification tasks, employing multitask (joint) training objectives that achieve superior classification and retrieval performance on in-distribution and out-of-distribution (OOD) datasets. Integrating multimodal features further enhances this performance, capturing complementary patterns across text and images. While text remains the dominant modality, visual data adds stylistic cues that enrich model performance. Moreover, text-image alignment strategies like CLIP and BLIP2 struggle due to low semantic overlap and vague connections between the modalities of escort ads, with end-to-end multimodal training proving more robust. Our findings emphasize the potential of multimodal AA (MAA) to combat HT, providing LEAs with robust tools to link ads and disrupt trafficking networks. 

**Abstract (ZH)**: 人口贩卖（HT）仍然是一个关键问题，贩子们正越来越多地利用在线陪侍广告（广告）来匿名宣传受害者。现有的检测方法，包括作者身份归属性（AA），通常集中在文本分析上，而忽视了在线陪侍广告的多模态性质，这些广告通常会配以图像。为了解决这一问题，我们引入了MATCHED，这是一个包含27,619个唯一的文本描述和55,115个唯一的图像的多模态数据集，这些数据是从美国四个地理区域的七个城市的Backpage陪侍平台收集的。我们的研究广泛地对文本-only、视觉-only和多模态基线方法进行了基准测试，用于销售者识别和验证任务，并采用多任务（联合）训练目标，这些目标在同分布和异分布（OOD）数据集上实现了优异的分类和检索性能。进一步整合多模态特征可以进一步提升这种性能，捕捉文本和图像之间的互补模式。虽然文本仍然是主导模态，但视觉数据提供了样式线索，丰富了模型的表现。此外，像CLIP和BLIP2这样的文本-图像对齐策略由于陪侍广告模态间的低语义重叠和模糊联系而表现不佳，而端到端的多模态训练则更为稳健。我们的研究结果强调了多模态作者身份归属性（MAA）对抗HT的潜力，为执法机构提供了强大的工具以链接广告并破坏贩运网络。 

---
# Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems with Large Language Models 

**Title (ZH)**: 物理推理器：知识增强的物理问题推理方法用于大型语言模型解决物理问题 

**Authors**: Xinyu Pang, Ruixin Hong, Zhanke Zhou, Fangrui Lv, Xinwei Yang, Zhilong Liang, Bo Han, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13791)  

**Abstract**: Physics problems constitute a significant aspect of reasoning, necessitating complicated reasoning ability and abundant physics knowledge. However, existing large language models (LLMs) frequently fail due to a lack of knowledge or incorrect knowledge application. To mitigate these issues, we propose Physics Reasoner, a knowledge-augmented framework to solve physics problems with LLMs. Specifically, the proposed framework constructs a comprehensive formula set to provide explicit physics knowledge and utilizes checklists containing detailed instructions to guide effective knowledge application. Namely, given a physics problem, Physics Reasoner solves it through three stages: problem analysis, formula retrieval, and guided reasoning. During the process, checklists are employed to enhance LLMs' self-improvement in the analysis and reasoning stages. Empirically, Physics Reasoner mitigates the issues of insufficient knowledge and incorrect application, achieving state-of-the-art performance on SciBench with an average accuracy improvement of 5.8%. 

**Abstract (ZH)**: 物理学问题构成了推理的重要方面，需要复杂的推理能力和丰富的物理学知识。然而，现有的大规模语言模型（LLMs）往往因为缺乏知识或错误的知识应用而失败。为了缓解这些问题，我们提出了一种知识增强框架——Physics Reasoner，该框架利用LLMs解决物理学问题。具体而言，该框架构建了一个全面的公式集，以提供显式的物理学知识，并利用包含详细指导说明的清单来指导有效知识的应用。简而言之，给定一个物理学问题，Physics Reasoner 通过三个阶段进行解决：问题分析、公式检索和引导推理。在这一过程中，清单被用于在分析和推理阶段增强LLMs的自我提升能力。实验结果显示，Physics Reasoner 减轻了知识不足和错误应用的问题，在SciBench上的性能达到了最先进的水平，平均准确率提高了5.8%。 

---
# Open Universal Arabic ASR Leaderboard 

**Title (ZH)**: 开放通用阿拉伯语ASR排行榜 

**Authors**: Yingzhi Wang, Anas Alhmoud, Muhammad Alqurishi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13788)  

**Abstract**: In recent years, the enhanced capabilities of ASR models and the emergence of multi-dialect datasets have increasingly pushed Arabic ASR model development toward an all-dialect-in-one direction. This trend highlights the need for benchmarking studies that evaluate model performance on multiple dialects, providing the community with insights into models' generalization capabilities.
In this paper, we introduce Open Universal Arabic ASR Leaderboard, a continuous benchmark project for open-source general Arabic ASR models across various multi-dialect datasets. We also provide a comprehensive analysis of the model's robustness, speaker adaptation, inference efficiency, and memory consumption. This work aims to offer the Arabic ASR community a reference for models' general performance and also establish a common evaluation framework for multi-dialectal Arabic ASR models. 

**Abstract (ZH)**: 近年来，ASR模型能力的增强以及多方言数据集的出现，逐渐推动了阿拉伯语ASR模型开发向一揽子多方言方向发展。这一趋势突显了 Benchmark 研究的需求，这些研究能够评估模型在多种方言上的性能，为社区提供模型泛化能力的洞察。

在本文中，我们介绍了 Open Universal Arabic ASR 领导板，这是一个持续的基准测试项目，旨在评估开源通用阿拉伯语ASR模型在多种多方言数据集上的性能。此外，我们还提供了模型鲁棒性、说话人适配、推理效率和内存消耗的全面分析。本文旨在为阿拉伯语ASR社区提供模型整体性能的参考，并建立一个多方言阿拉伯语ASR模型的共同评估框架。 

---
# Knowledge Editing with Dynamic Knowledge Graphs for Multi-hop Question Answering 

**Title (ZH)**: 使用动态知识图谱的知识编辑方法用于多跳问答 

**Authors**: Yifan Lu, Yigeng Zhou, Jing Li, Yequan Wang, Xuebo Liu, Daojing He, Fangming Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13782)  

**Abstract**: Multi-hop question answering (MHQA) poses a significant challenge for large language models (LLMs) due to the extensive knowledge demands involved. Knowledge editing, which aims to precisely modify the LLMs to incorporate specific knowledge without negatively impacting other unrelated knowledge, offers a potential solution for addressing MHQA challenges with LLMs. However, current solutions struggle to effectively resolve issues of knowledge conflicts. Most parameter-preserving editing methods are hindered by inaccurate retrieval and overlook secondary editing issues, which can introduce noise into the reasoning process of LLMs. In this paper, we introduce KEDKG, a novel knowledge editing method that leverages a dynamic knowledge graph for MHQA, designed to ensure the reliability of answers. KEDKG involves two primary steps: dynamic knowledge graph construction and knowledge graph augmented generation. Initially, KEDKG autonomously constructs a dynamic knowledge graph to store revised information while resolving potential knowledge conflicts. Subsequently, it employs a fine-grained retrieval strategy coupled with an entity and relation detector to enhance the accuracy of graph retrieval for LLM generation. Experimental results on benchmarks show that KEDKG surpasses previous state-of-the-art models, delivering more accurate and reliable answers in environments with dynamic information. 

**Abstract (ZH)**: 多跳问答（MHQA）对大型语言模型（LLMs）构成了显著挑战，因为这需要大量的知识支持。知识编辑旨在通过精确修改LLMs以包含特定知识而不影响其他无关知识，从而为解决MHQA挑战提供了潜在的解决方案。然而，当前的方法在解决知识冲突问题时难以有效应对。大多数保持参数数量不变的编辑方法受限于不准确的知识检索，并且忽视了二次编辑带来的潜在噪声问题，可能会干扰LLMs的推理过程。在本文中，我们提出了KEDKG，这是一种利用动态知识图谱进行MHQA的新颖知识编辑方法，旨在确保答案的可靠性。KEDKG包括两个主要步骤：动态知识图谱构建和知识图谱增强生成。首先，KEDKG自主构建动态知识图谱以存储更新信息并解决潜在的知识冲突。随后，它采用细粒度的检索策略并结合实体和关系检测器来提高图检索的准确性，从而增强LLMs生成的准确性。基准实验结果表明，KEDKG超过了之前最先进的模型，在动态信息环境下提供了更准确和可靠的答案。 

---
# Meta-Reflection: A Feedback-Free Reflection Learning Framework 

**Title (ZH)**: 元反思：一种无反馈的反射学习框架 

**Authors**: Yaoke Wang, Yun Zhu, Xintong Bao, Wenqiao Zhang, Suyang Dai, Kehan Chen, Wenqiang Li, Gang Huang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13781)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in natural language understanding and reasoning, they often display undesirable behaviors, such as generating hallucinations and unfaithful reasoning. A prevalent strategy to mitigate these issues is the use of reflection, which refines responses through an iterative process. However, while promising, reflection heavily relies on high-quality external feedback and requires iterative multi-agent inference processes, thus hindering its practical application. In this paper, we propose Meta-Reflection, a novel feedback-free reflection mechanism that necessitates only a single inference pass without external feedback. Motivated by the human ability to remember and retrieve reflections from past experiences when encountering similar problems, Meta-Reflection integrates reflective insights into a codebook, allowing the historical insights to be stored, retrieved, and used to guide LLMs in problem-solving. To thoroughly investigate and evaluate the practicality of Meta-Reflection in real-world scenarios, we introduce an industrial e-commerce benchmark named E-commerce Customer Intent Detection (ECID). Extensive experiments conducted on both public datasets and the ECID benchmark highlight the effectiveness and efficiency of our proposed approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在自然语言理解与推理方面表现出色，它们往往会出现一些不可取的行为，如生成幻觉和不忠实的推理。应对这些问题的常见策略是采用反馈机制，通过迭代过程来精炼回应。然而，反馈机制虽然有潜力，但它高度依赖高质量的外部反馈，并需要迭代的多代理推断过程，这阻碍了其实际应用。本文提出了一种新颖的无反馈反射机制——元反射（Meta-Reflection），该机制仅需一次推断过程，无需外部反馈。受到人类在遇到类似问题时能够回忆和借鉴过去经验中反思信息的能力的启发，元反射将反思性洞察整合到代码书中，从而使历史洞察得以存储、检索，并用于指导LLMs解决问题。为了全面考察和评估元反射在实际场景中的实用性和有效性，我们引入了一个工业电商基准——电商客户意图检测（ECID）。在公共数据集和ECID基准上的大量实验表明，所提出的方法不仅有效，而且高效。 

---
# LLM-SEM: A Sentiment-Based Student Engagement Metric Using LLMS for E-Learning Platforms 

**Title (ZH)**: LLM-SEM：一种基于情感的學生 Engagement 度量方法，使用大型语言模型构建在线学习平台 

**Authors**: Ali Hamdi, Ahmed Abdelmoneim Mazrou, Mohamed Shaltout  

**Link**: [PDF](https://arxiv.org/pdf/2412.13765)  

**Abstract**: Current methods for analyzing student engagement in e-learning platforms, including automated systems, often struggle with challenges such as handling fuzzy sentiment in text comments and relying on limited metadata. Traditional approaches, such as surveys and questionnaires, also face issues like small sample sizes and scalability. In this paper, we introduce LLM-SEM (Language Model-Based Student Engagement Metric), a novel approach that leverages video metadata and sentiment analysis of student comments to measure engagement. By utilizing recent Large Language Models (LLMs), we generate high-quality sentiment predictions to mitigate text fuzziness and normalize key features such as views and likes. Our holistic method combines comprehensive metadata with sentiment polarity scores to gauge engagement at both the course and lesson levels. Extensive experiments were conducted to evaluate various LLM models, demonstrating the effectiveness of LLM-SEM in providing a scalable and accurate measure of student engagement. We fine-tuned LLMs, including AraBERT, TXLM-RoBERTa, LLama 3B and Gemma 9B from Ollama, using human-annotated sentiment datasets to enhance prediction accuracy. 

**Abstract (ZH)**: 当前用于分析在线学习平台中学生参与度的方法，包括自动化系统，往往面临着处理文本评论中的模糊情感和依赖有限元数据等挑战。传统的调查和问卷方法也存在样本量小和可扩展性差的问题。在本文中，我们引入了一种名为LLM-SEM（基于语言模型的学生参与度度量）的新方法，该方法结合了视频元数据和学生评论的情感分析，以测量学生的参与度。通过利用最新的大型语言模型（LLMs），我们生成高质量的情感预测，以减轻文本模糊性并标准化关键特征，如观看次数和点赞数。我们采用的整体方法结合了全面的元数据和情感极性评分，以衡量课程和课程单元级别的参与度。进行了广泛的实验来评估各种LLM模型，表明LLM-SEM在提供可扩展且准确的学生参与度度量方面的有效性。我们对包括AraBERT、TXLM-RoBERTa、LLama 3B和Ollama的Gemma 9B在内的LLM进行了微调，使用了人类注释的情感数据集，以提高预测准确性。 

---
# RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment 

**Title (ZH)**: RAG-RewardBench：检索增强生成中奖励模型的偏好对齐基准测试 

**Authors**: Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13746)  

**Abstract**: Despite the significant progress made by existing retrieval augmented language models (RALMs) in providing trustworthy responses and grounding in reliable sources, they often overlook effective alignment with human preferences. In the alignment process, reward models (RMs) act as a crucial proxy for human values to guide optimization. However, it remains unclear how to evaluate and select a reliable RM for preference alignment in RALMs. To this end, we propose RAG-RewardBench, the first benchmark for evaluating RMs in RAG settings. First, we design four crucial and challenging RAG-specific scenarios to assess RMs, including multi-hop reasoning, fine-grained citation, appropriate abstain, and conflict robustness. Then, we incorporate 18 RAG subsets, six retrievers, and 24 RALMs to increase the diversity of data sources. Finally, we adopt an LLM-as-a-judge approach to improve preference annotation efficiency and effectiveness, exhibiting a strong correlation with human annotations. Based on the RAG-RewardBench, we conduct a comprehensive evaluation of 45 RMs and uncover their limitations in RAG scenarios. Additionally, we also reveal that existing trained RALMs show almost no improvement in preference alignment, highlighting the need for a shift towards preference-aligned this http URL release our benchmark and code publicly at this https URL for future work. 

**Abstract (ZH)**: 尽管现有的检索增强语言模型（RALMs）在提供可信响应和可靠来源的支持方面取得了显著进展，但在与人类偏好有效对齐方面往往并未得到足够重视。在对齐过程中，奖励模型（RMs）充当了一个关键的代理，以指导优化过程并反映人类价值观。然而，如何评估和选择可靠的奖励模型以用于RALMs中的偏好对齐仍不清楚。为了解决这一问题，我们提出了RAG-RewardBench，这是首个在RAG（Retrieval-Augmented Generation）环境中评估奖励模型的基准。首先，我们设计了四种关键且具有挑战性的RAG特定场景来评估奖励模型的表现，包括多跳推理、精细引用、恰当的回避和冲突鲁棒性。其次，我们结合了18个RAG子集、六种检索器和24种RALMs以增加数据来源的多样性。最后，我们采用了一个基于大语言模型（LLM）作为裁判的方法来提高偏好标注的效率和效果，这种方法与人类标注有着很强的相关性。基于RAG-RewardBench，我们对45个奖励模型进行了全面的评估，并发现了它们在RAG场景中的局限性。此外，我们还揭示出现有的受过训练的RALMs在偏好对齐方面几乎没有改进，这突显了转向更重视偏好对齐的重要性。我们将在GitHub上公开发布该基准和代码，供未来研究参考，链接为：[公开链接]。 

---
# Learning Complex Word Embeddings in Classical and Quantum Spaces 

**Title (ZH)**: 在经典和量子空间中学习复杂词嵌入 

**Authors**: Carys Harvey, Stephen Clark, Douglas Brown, Konstantinos Meichanetzidis  

**Link**: [PDF](https://arxiv.org/pdf/2412.13745)  

**Abstract**: We present a variety of methods for training complex-valued word embeddings, based on the classical Skip-gram model, with a straightforward adaptation simply replacing the real-valued vectors with arbitrary vectors of complex numbers. In a more "physically-inspired" approach, the vectors are produced by parameterised quantum circuits (PQCs), which are unitary transformations resulting in normalised vectors which have a probabilistic interpretation. We develop a complex-valued version of the highly optimised C code version of Skip-gram, which allows us to easily produce complex embeddings trained on a 3.8B-word corpus for a vocabulary size of over 400k, for which we are then able to train a separate PQC for each word. We evaluate the complex embeddings on a set of standard similarity and relatedness datasets, for some models obtaining results competitive with the classical baseline. We find that, while training the PQCs directly tends to harm performance, the quantum word embeddings from the two-stage process perform as well as the classical Skip-gram embeddings with comparable numbers of parameters. This enables a highly scalable route to learning embeddings in complex spaces which scales with the size of the vocabulary rather than the size of the training corpus. In summary, we demonstrate how to produce a large set of high-quality word embeddings for use in complex-valued and quantum-inspired NLP models, and for exploring potential advantage in quantum NLP models. 

**Abstract (ZH)**: 我们基于经典的Skip-gram模型，提出了一系列用于训练复值词嵌入的方法，简单地将实值向量替换为任意复数向量。在一种更“物理启发”的方法中，这些向量是由参数化量子电路（PQCs）产生的，PQCs是酉变换，生成归一化的向量，这些向量具有概率解释。我们开发了一个复值版本的优化过的C语言实现的Skip-gram算法，这使我们能够轻松地在包含38亿词的语料库上训练多达40万大小的词汇表的复嵌入，并为每个词训练一个独立的PQC。我们评估了复嵌入在一组标准相似性和相关性数据集上的表现，对于某些模型，它们在某些任务上的表现与经典基准方法相当。我们发现，虽然直接训练PQCs往往会损害性能，但经过两阶段过程产生的量子词嵌入与具有类似参数数量的经典Skip-gram嵌入表现相当。这为我们提供了一种高度可扩展的方法，通过词汇量而非训练语料库的规模来扩展复空间中的词嵌入学习。总之，我们展示了如何生成大量高质量的复值词嵌入，以便在复值和量子启发的自然语言处理（NLP）模型中使用，并探索量子NLP模型可能的优点。 

---
# Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models 

**Title (ZH)**: 联邦学习与RAG集成：一种适用于医疗大规模语言模型的扩展方法 

**Authors**: Jincheol Jung, Hongju Jeong, Eui-Nam Huh  

**Link**: [PDF](https://arxiv.org/pdf/2412.13720)  

**Abstract**: This study analyzes the performance of domain-specific Large Language Models (LLMs) for the medical field by integrating Retrieval-Augmented Generation (RAG) systems within a federated learning framework. Leveraging the inherent advantages of federated learning, such as preserving data privacy and enabling distributed computation, this research explores the integration of RAG systems with models trained under varying client configurations to optimize performance. Experimental results demonstrate that the federated learning-based models integrated with RAG systems consistently outperform their non-integrated counterparts across all evaluation metrics. This study highlights the potential of combining federated learning and RAG systems for developing domain-specific LLMs in the medical field, providing a scalable and privacy-preserving solution for enhancing text generation capabilities. 

**Abstract (ZH)**: 本研究通过在联邦学习框架中整合检索增强生成（RAG）系统，分析了专门领域的大语言模型（LLMs）在医疗领域的性能。依托联邦学习固有的优势，如保护数据隐私和实现分布式计算，本研究探讨了在不同客户端配置下将RAG系统与模型进行整合以优化性能的可能性。实验结果表明，基于联邦学习并与RAG系统整合的模型在所有评估指标上均优于未整合的模型。本研究强调了将联邦学习和RAG系统结合用于开发医疗领域的专门领域大语言模型的潜力，提供了一个可扩展且保护隐私的解决方案，以提升文本生成能力。 

---
# Towards Automatic Evaluation for Image Transcreation 

**Title (ZH)**: 面向图像再创造的自动评估方法 

**Authors**: Simran Khanuja, Vivek Iyer, Claire He, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2412.13717)  

**Abstract**: Beyond conventional paradigms of translating speech and text, recently, there has been interest in automated transcreation of images to facilitate localization of visual content across different cultures. Attempts to define this as a formal Machine Learning (ML) problem have been impeded by the lack of automatic evaluation mechanisms, with previous work relying solely on human evaluation. In this paper, we seek to close this gap by proposing a suite of automatic evaluation metrics inspired by machine translation (MT) metrics, categorized into: a) Object-based, b) Embedding-based, and c) VLM-based. Drawing on theories from translation studies and real-world transcreation practices, we identify three critical dimensions of image transcreation: cultural relevance, semantic equivalence and visual similarity, and design our metrics to evaluate systems along these axes. Our results show that proprietary VLMs best identify cultural relevance and semantic equivalence, while vision-encoder representations are adept at measuring visual similarity. Meta-evaluation across 7 countries shows our metrics agree strongly with human ratings, with average segment-level correlations ranging from 0.55-0.87. Finally, through a discussion of the merits and demerits of each metric, we offer a robust framework for automated image transcreation evaluation, grounded in both theoretical foundations and practical application. Our code can be found here: this https URL 

**Abstract (ZH)**: 超越传统的语音和文本翻译范式，近年来，研究人员开始对自动图像重塑产生了兴趣，旨在促进不同文化背景下的视觉内容本地化。由于缺乏自动评估机制，定义这一领域为形式化的机器学习问题的努力受到了阻碍，之前的工作主要依赖于人工评估。本文旨在通过提出一套基于机器翻译（MT）标准的自动评估指标来弥补这一空白，这些指标被分类为：a) 基于对象的，b) 基于嵌入的，以及c) 基于视觉语言模型的。从翻译研究理论和实际的图像重塑实践中，我们确定了图像重塑的三个关键维度：文化相关性、语义等价性和视觉相似性，并设计了相应的评估指标来从这三个维度评估系统。我们的结果显示，专有的视觉语言模型在识别文化相关性和语义等价性方面表现最佳，而视觉编码器的表现则在衡量视觉相似性方面更为有效。在7个国家进行的元评估显示，我们的指标与人工评分高度一致，段落级别的相关性平均范围从0.55到0.87。最后，通过对每种指标的利弊进行讨论，我们提出了一种坚实的框架，从理论基础和实际应用两个方面为自动图像重塑评估提供指导。我们的代码可以在以下链接找到：this https URL 

---
# Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models 

**Title (ZH)**: Typhoon 2：一个 FAMILY 族开源文本和多模态 Thai 大型语言模型 

**Authors**: Kunat Pipatanakul, Potsawee Manakul, Natapong Nitarach, Warit Sirichotedumrong, Surapon Nonesung, Teetouch Jaknamon, Parinthapat Pengpun, Pittawat Taveekitworachai, Adisai Na-Thalang, Sittipong Sripaisarnmongkol, Krisanapong Jirayoot, Kasima Tharnpipitchai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13702)  

**Abstract**: This paper introduces Typhoon 2, a series of text and multimodal large language models optimized for the Thai language. The series includes models for text, vision, and audio. Typhoon2-Text builds on state-of-the-art open models, such as Llama 3 and Qwen2, and we perform continual pre-training on a mixture of English and Thai data. We employ various post-training techniques to enhance Thai language performance while preserving the base models' original capabilities. We release text models across a range of sizes, from 1 to 70 billion parameters, available in both base and instruction-tuned variants. Typhoon2-Vision improves Thai document understanding while retaining general visual capabilities, such as image captioning. Typhoon2-Audio introduces an end-to-end speech-to-speech model architecture capable of processing audio, speech, and text inputs and generating both text and speech outputs simultaneously. 

**Abstract (ZH)**: 本文介绍了面向泰语优化的一系列文本和多模态大规模语言模型——Typhoon 2。该系列包括文本、视觉和音频模型。Typhoon2-Text 基于最新的开源模型，如 Llama 3 和 Qwen2，并在英泰混合数据上进行持续预训练。我们采用各种后训练技术提升泰语性能，同时保留基模型的原始能力。我们发布了从 1 亿到 70 亿参数不等的文本模型，提供基模型和指令调优两种版本。Typhoon2-Vision 提升了泰语文档理解能力，同时保留了如图像描述等通用视觉能力。Typhoon2-Audio 引入了一种端到端的语音转语音模型架构，能够处理音频、语音和文本输入，并同时生成文本和语音输出。 

---
# Towards Efficient and Explainable Hate Speech Detection via Model Distillation 

**Title (ZH)**: 通过模型蒸馏实现高效可解释的仇恨言论检测 

**Authors**: Paloma Piot, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2412.13698)  

**Abstract**: Automatic detection of hate and abusive language is essential to combat its online spread. Moreover, recognising and explaining hate speech serves to educate people about its negative effects. However, most current detection models operate as black boxes, lacking interpretability and explainability. In this context, Large Language Models (LLMs) have proven effective for hate speech detection and to promote interpretability. Nevertheless, they are computationally costly to run. In this work, we propose distilling big language models by using Chain-of-Thought to extract explanations that support the hate speech classification task. Having small language models for these tasks will contribute to their use in operational settings. In this paper, we demonstrate that distilled models deliver explanations of the same quality as larger models while surpassing them in classification performance. This dual capability, classifying and explaining, advances hate speech detection making it more affordable, understandable and actionable. 

**Abstract (ZH)**: 自动检测仇恨言论和滥用语言是遏制其在网络上传播的关键。此外，识别并解释仇恨言论有助于教育人们了解其负面影响。然而，当前大多数检测模型运作如同黑盒，缺乏可解释性和可解释性。在此背景下，大规模语言模型（LLMs）已被证明对仇恨言论检测和促进可解释性有效。然而，它们在运行上消耗大量计算资源。在本研究中，我们提出了通过使用思维链（Chain-of-Thought）对大规模语言模型进行蒸馏，以提取支持仇恨言论分类任务的解释。小型语言模型在这些任务中的应用有助于其在实际操作中的使用。在本文中，我们证明了蒸馏模型提供的解释与大型模型一样优质，但在分类性能上更胜一筹。这种双重能力——分类和解释——推动了仇恨言论检测的发展，使其更具成本效益、易于理解且易于执行。 

---
# AntiLeak-Bench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge 

**Title (ZH)**: AntiLeak-Bench: 防止数据污染并通过构建更新的现实世界知识自动构建基准的方法 

**Authors**: Xiaobao Wu, Liangming Pan, Yuxi Xie, Ruiwen Zhou, Shuai Zhao, Yubo Ma, Mingzhe Du, Rui Mao, Anh Tuan Luu, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13670)  

**Abstract**: Data contamination hinders fair LLM evaluation by introducing test data into newer models' training sets. Existing studies solve this challenge by updating benchmarks with newly collected data. However, they fail to guarantee contamination-free evaluation as the newly collected data may contain pre-existing knowledge, and their benchmark updates rely on intensive human labor. To address these issues, we in this paper propose AntiLeak-Bench, an automated anti-leakage benchmarking framework. Instead of simply using newly collected data, we construct samples with explicitly new knowledge absent from LLMs' training sets, which thus ensures strictly contamination-free evaluation. We further design a fully automated workflow to build and update our benchmark without human labor. This significantly reduces the cost of benchmark maintenance to accommodate emerging LLMs. Through extensive experiments, we highlight that data contamination likely exists before LLMs' cutoff time and demonstrate AntiLeak-Bench effectively overcomes this challenge. 

**Abstract (ZH)**: 数据污染妨碍了对先进语言模型（LLM）的公平评估，因为它将测试数据引入了新的模型训练集中。现有研究通过更新基准数据集以解决这一挑战，但它们无法保证完全无污染的评估，因为新收集的数据中可能包含已有知识，并且其基准更新依赖于大量的人工劳动。为了解决这些问题，我们在本文中提出了一种自动化的抗泄露基准框架——AntiLeak-Bench。我们不仅使用新收集的数据，还构建了不含LLM训练集中已有知识的样本，从而确保了严格的无污染评估。我们进一步设计了一个完全自动化的流程，用于构建和更新基准数据集，无需人工干预。这显著降低了基准维护成本，以便适应新兴的LLM。通过大量实验，我们表明数据污染在LLM截止时间前几乎普遍存在，并证明AntiLeak-Bench能有效地克服这一挑战。 

---
# Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation 

**Title (ZH)**: LLM在个性化虚假信息生成方面被误用的漏洞评估 

**Authors**: Aneta Zugecova, Dominik Macko, Ivan Srba, Robert Moro, Jakub Kopal, Katarina Marcincinova, Matus Mesarcik  

**Link**: [PDF](https://arxiv.org/pdf/2412.13666)  

**Abstract**: The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts rises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluation of vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers. 

**Abstract (ZH)**: 近年来大数据量的语言模型（LLMs）生成高质量、难以与人类撰写的文本区分开来的内容的能力引发了对其被误用的诸多担忧。先前的研究表明，LLMs可以有效用于按照预定义的叙述生成虚假信息新闻文章。此外，它们生成个性化（在各个方面）内容的能力也得到了评估，结果大多显示出一定的可用性。然而，将个性化能力和生成虚假信息的能力结合起来进行研究的综合性研究尚未全面开展。这样的危险组合应当触发LLMs的安全过滤器，如果有的话。本研究通过评估近期开源和封闭的LLMs的安全漏洞以及它们生成个性化虚假信息新闻文章的意愿，填补了这一空白，并使用英语进行了相关实验。我们进一步探讨了LLMs是否能够可靠地元评估个性化质量，以及个性化是否影响生成文本的可检测性。研究表明，需要加强更多的安全过滤器和免责声明，因为这些在大多数评估的LLMs中并未有效运行。此外，我们的研究揭示，个性化实际上降低了安全过滤器的激活程度；因此，有效运作的个性化充当了安全机制的“绕过”。这种行为必须引起LLMs开发人员和服务提供商的紧急关注。 

---
# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference 

**Title (ZH)**: 更聪明、更出色、更快、更持久：一种现代双向编码器，实现快速、内存高效且适用于长上下文的微调与推理 

**Authors**: Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, Iacopo Poli  

**Link**: [PDF](https://arxiv.org/pdf/2412.13663)  

**Abstract**: Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs. 

**Abstract (ZH)**: 这样的编码器-only变压器模型（如BERT）相较于较大的解码器-only模型，在检索和分类任务上提供了很好的性能-大小折中。尽管BERT自推出以来在众多生产流水线中发挥了重要作用，但对其的帕累托改进却十分有限。在本文中，我们介绍了一种名为ModernBERT的新模型，这是一种将现代模型优化应用到编码器-only模型上的尝试，并代表了对较旧编码器的重大帕累托改进。ModernBERT模型在大量训练数据（2万亿个令牌）上进行训练，并具有原生的8192序列长度，其在多种评价指标上展现出了最先进的结果，这些评价指标涵盖了多样化的分类任务以及不同领域的单向和双向检索（包括代码）。除了下游性能稳健外，ModernBERT还是最高效（速度和内存使用方面）的编码器，并且设计用于常见GPU上的推理。 

---
# PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling 

**Title (ZH)**: PsyDT：利用大型语言模型构建个性化心理咨询风格的心理咨询数字孪生系统 

**Authors**: Haojie Xie, Yirong Chen, Xiaofen Xing, Jingkai Lin, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13660)  

**Abstract**: Currently, large language models (LLMs) have made significant progress in the field of psychological counseling. However, existing mental health LLMs overlook a critical issue where they do not consider the fact that different psychological counselors exhibit different personal styles, including linguistic style and therapy techniques, etc. As a result, these LLMs fail to satisfy the individual needs of clients who seek different counseling styles. To help bridge this gap, we propose PsyDT, a novel framework using LLMs to construct the Digital Twin of Psychological counselor with personalized counseling style. Compared to the time-consuming and costly approach of collecting a large number of real-world counseling cases to create a specific counselor's digital twin, our framework offers a faster and more cost-effective solution. To construct PsyDT, we utilize dynamic one-shot learning by using GPT-4 to capture counselor's unique counseling style, mainly focusing on linguistic style and therapy techniques. Subsequently, using existing single-turn long-text dialogues with client's questions, GPT-4 is guided to synthesize multi-turn dialogues of specific counselor. Finally, we fine-tune the LLMs on the synthetic dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with personalized counseling style. Experimental results indicate that our proposed PsyDT framework can synthesize multi-turn dialogues that closely resemble real-world counseling cases and demonstrate better performance compared to other baselines, thereby show that our framework can effectively construct the digital twin of psychological counselor with a specific counseling style. 

**Abstract (ZH)**: 目前，大型语言模型（LLMs）在心理辅导领域取得了显著进展。然而，现有的心理健康LLMs忽视了一个关键问题，即它们没有考虑到不同的心理辅导员具备不同的个人风格，包括语言风格和治疗方法等。因此，这些LKM无法满足寻求不同咨询风格的客户的个体需求。为弥补这一差距，我们提出PsyDT，一种使用LLMs构建个性化心理辅导员数字孪生的新框架。与收集大量真实咨询案例以创建特定心理辅导员数字孪生的耗时且成本高昂的方法相比，我们提出的框架提供了一种更快且更具成本效益的解决方案。为了构建PsyDT，我们利用动态单次学习，使用GPT-4捕获心理辅导员的独特咨询风格，主要集中在语言风格和治疗方法上。随后，利用现有的一轮多文本对话和客户的提问，GPT-4被引导合成特定心理辅导员的多轮对话。最后，我们在合成数据集PsyDTCorpus上对LLMs进行微调，以实现具有个性化咨询风格的心理辅导员数字孪生。实验结果表明，我们提出的PsyDT框架能够合成高度类似于真实咨询案例的多轮对话，并且在与其他基线方法的比较中表现出更好的性能，从而证明我们的框架能够有效构建具有特定咨询风格的心理辅导员数字孪生。 

---
# SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation 

**Title (ZH)**: 范围：在长上下文生成中优化键值缓存压缩 

**Authors**: Jialong Wu, Zhenglin Wang, Linhai Zhang, Yilong Lai, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13649)  

**Abstract**: Key-Value (KV) cache has become a bottleneck of LLMs for long-context generation. Despite the numerous efforts in this area, the optimization for the decoding phase is generally ignored. However, we believe such optimization is crucial, especially for long-output generation tasks based on the following two observations: (i) Excessive compression during the prefill phase, which requires specific full context impairs the comprehension of the reasoning task; (ii) Deviation of heavy hitters occurs in the reasoning tasks with long outputs. Therefore, SCOPE, a simple yet efficient framework that separately performs KV cache optimization during the prefill and decoding phases, is introduced. Specifically, the KV cache during the prefill phase is preserved to maintain the essential information, while a novel strategy based on sliding is proposed to select essential heavy hitters for the decoding phase. Memory usage and memory transfer are further optimized using adaptive and discontinuous strategies. Extensive experiments on LongGenBench show the effectiveness and generalization of SCOPE and its compatibility as a plug-in to other prefill-only KV compression methods. 

**Abstract (ZH)**: 在长上下文生成中，键值（KV）缓存已经成为LLMs（大型语言模型）的一个瓶颈。尽管在这一领域已经做出了许多努力，但对于解码阶段的优化通常被忽略。然而，我们认为这种优化至关重要，特别是在基于以下两个观察结果的长时间输出生成任务中：(i) 在预填充阶段过度压缩需要特定完整上下文的信息，这会损害推理任务的理解；(ii) 在长时间输出的推理任务中，重击项（Heavy Hitters）会发生偏离。因此，我们引入了SCOPE（一种简单而高效的框架），它分别在预填充和解码阶段执行KV缓存优化。具体而言，预填充阶段的KV缓存被保留以保持必要的信息，同时提出了一种基于滑动的新策略来选择解码阶段的重击项。通过自适应和不连续策略进一步优化了内存使用和内存传输。广泛的实验（在LongGenBench上进行）表明了SCOPE的有效性、泛化能力和与其他仅预填充KV压缩方法的兼容性。 

---
# LIFT: Improving Long Context Understanding Through Long Input Fine-Tuning 

**Title (ZH)**: LIFT：通过长输入微调提高长期上下文理解 

**Authors**: Yansheng Mao, Jiaqi Li, Fanxu Meng, Jing Xiong, Zilong Zheng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13626)  

**Abstract**: Long context understanding remains challenging for large language models due to their limited context windows. This paper introduces Long Input Fine-Tuning (LIFT) for long context modeling, a novel framework that enhances LLM performance on long-context tasks by adapting model parameters to the context at test time. LIFT enables efficient processing of lengthy inputs without the computational burden of offline long-context adaptation, and can improve the long-context capabilities of arbitrary short-context models. The framework is further enhanced by integrating in-context learning and pre-LIFT supervised fine-tuning. The combination of in-context learning and LIFT enables short-context models like Llama 3 to handle arbitrarily long contexts and consistently improves their performance on popular long-context benchmarks like LooGLE and LongBench. We also provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research. 

**Abstract (ZH)**: 长上下文理解仍然是大型语言模型面临的挑战，主要是由于它们有限的上下文窗口。本文介绍了长输入微调（LIFT），这是一种新的框架，通过在测试时调整模型参数来增强模型在长上下文任务上的性能。LIFT 允模型高效处理长输入，而无需进行计算负担重的离线长上下文适应，并且可以增强任意短上下文模型的长上下文能力。该框架进一步通过整合上下文学习和预-LIFT 监督微调得到了增强。上下文学习与 LIFT 的结合使得像 Llama 3 这样的短上下文模型能够处理任意长的上下文，并在流行的长上下文基准测试如 LooGLE 和 LongBench 上持续改进其性能。我们还对 LIFT 在长上下文理解中的优势和局限性进行了全面分析，并为未来的研究提供了宝贵的指导方向。 

---
# Are LLMs Good Literature Review Writers? Evaluating the Literature Review Writing Ability of Large Language Models 

**Title (ZH)**: 大型语言模型是好的文献综述撰写者吗？评估大型语言模型的文献综述撰写能力 

**Authors**: Xuemei Tang, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13612)  

**Abstract**: The literature review is a crucial form of academic writing that involves complex processes of literature collection, organization, and summarization. The emergence of large language models (LLMs) has introduced promising tools to automate these processes. However, their actual capabilities in writing comprehensive literature reviews remain underexplored, such as whether they can generate accurate and reliable references. To address this gap, we propose a framework to assess the literature review writing ability of LLMs automatically. We evaluate the performance of LLMs across three tasks: generating references, writing abstracts, and writing literature reviews. We employ external tools for a multidimensional evaluation, which includes assessing hallucination rates in references, semantic coverage, and factual consistency with human-written context. By analyzing the experimental results, we find that, despite advancements, even the most sophisticated models still cannot avoid generating hallucinated references. Additionally, different models exhibit varying performance in literature review writing across different disciplines. 

**Abstract (ZH)**: 文献综述是一种重要的学术写作形式，涉及文献的复杂收集、组织和总结过程。大型语言模型（LLMs）的出现引入了自动化这些过程的有力工具。然而，它们在撰写全面文献综述的实际能力仍然未被充分探索，例如它们是否能够生成准确可靠的参考文献。为解决这一问题，我们提出了一种框架，以自动评估LLMs的文献综述写作能力。我们通过三项任务来评估LLMs的表现：生成参考文献、撰写摘要和撰写文献综述。我们利用外部工具进行多维度的评估，其中包括评估参考文献中的虚构率、语义覆盖率以及与人类撰写背景的实证一致性。通过对实验结果的分析，我们发现，尽管技术取得了进步，甚至最复杂的模型也无法避免生成虚构的参考文献。此外，不同模型在不同学科的文献综述写作上表现出不同的性能。 

---
# Beyond Outcomes: Transparent Assessment of LLM Reasoning in Games 

**Title (ZH)**: 超越结果：透明评估大规模语言模型在游戏中的推理能力 

**Authors**: Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.13602)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in real-world applications that demand complex reasoning. To track progress, robust benchmarks are required to evaluate their capabilities beyond superficial pattern recognition. However, current LLM reasoning benchmarks often face challenges such as insufficient interpretability, performance saturation or data contamination. To address these challenges, we introduce GAMEBoT, a gaming arena designed for rigorous and transparent assessment of LLM reasoning capabilities. GAMEBoT decomposes complex reasoning in games into predefined modular subproblems. This decomposition allows us to design a suite of Chain-of-Thought (CoT) prompts that leverage domain knowledge to guide LLMs in addressing these subproblems before action selection. Furthermore, we develop a suite of rule-based algorithms to generate ground truth for these subproblems, enabling rigorous validation of the LLMs' intermediate reasoning steps. This approach facilitates evaluation of both the quality of final actions and the accuracy of the underlying reasoning process. GAMEBoT also naturally alleviates the risk of data contamination through dynamic games and head-to-head LLM competitions. We benchmark 17 prominent LLMs across eight games, encompassing various strategic abilities and game characteristics. Our results suggest that GAMEBoT presents a significant challenge, even when LLMs are provided with detailed CoT prompts. Project page: \url{this https URL} 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地应用于需要复杂推理的实际应用场景中。为了跟踪进展，需要建立稳健的基准测试，以评估它们超越表面模式识别的能力。然而，当前的LLM推理基准测试常常面临不可解释性不足、性能饱和或数据污染等挑战。为了解决这些挑战，我们引入了GAMEBoT，这是一种旨在进行严格和透明评估LLM推理能力的游戏竞技场。GAMEBoT将游戏中复杂推理问题分解为预定义的模块化子问题。这种分解使我们能够设计一系列基于推理链（Chain-of-Thought, CoT）的提示，利用领域知识引导LLM分别解决这些子问题，然后进行行动选择。此外，我们还开发了一系列基于规则的算法来生成这些子问题的真相标准，从而实现对LLM中间推理步骤的严格验证。这种方法有助于评估最终行动的质量以及底层推理过程的准确性。通过动态游戏和LLM之间的对抗比赛，GAMEBoT自然地减轻了数据污染的风险。我们在八款游戏中评估了17种知名LLM，涵盖了各种战略能力和游戏特性。我们的结果显示，即使提供详细的CoT提示，GAMEBoT仍然是一个显著的挑战。项目页面：[这个链接](this https URL) 

---
# EvoWiki: Evaluating LLMs on Evolving Knowledge 

**Title (ZH)**: EvoWiki：评估LLM在不断演变的知识上的表现 

**Authors**: Wei Tang, Yixin Cao, Yang Deng, Jiahao Ying, Bo Wang, Yizhe Yang, Yuyue Zhao, Qi Zhang, Xuanjing Huang, Yugang Jiang, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13582)  

**Abstract**: Knowledge utilization is a critical aspect of LLMs, and understanding how they adapt to evolving knowledge is essential for their effective deployment. However, existing benchmarks are predominantly static, failing to capture the evolving nature of LLMs and knowledge, leading to inaccuracies and vulnerabilities such as contamination. In this paper, we introduce EvoWiki, an evolving dataset designed to reflect knowledge evolution by categorizing information into stable, evolved, and uncharted states. EvoWiki is fully auto-updatable, enabling precise evaluation of continuously changing knowledge and newly released LLMs. Through experiments with Retrieval-Augmented Generation (RAG) and Contunual Learning (CL), we evaluate how effectively LLMs adapt to evolving knowledge. Our results indicate that current models often struggle with evolved knowledge, frequently providing outdated or incorrect responses. Moreover, the dataset highlights a synergistic effect between RAG and CL, demonstrating their potential to better adapt to evolving knowledge. EvoWiki provides a robust benchmark for advancing future research on the knowledge evolution capabilities of large language models. 

**Abstract (ZH)**: 知识利用是大语言模型（LLM）的关键方面，理解它们如何适应不断演变的知识对其实效部署至关重要。然而，现有基准大多是静态的，无法捕捉LLM和知识的演变性质，导致不准确性和漏洞，如污染。本文中，我们介绍了EvoWiki，这是一个不断演变的数据集，旨在通过将信息分类为稳定状态、演变状态和未探索状态来反映知识的演变。EvoWiki完全可自动更新，使得能够对不断变化的知识和新发布的LLM进行精确评估。通过使用检索增强生成（RAG）和持续学习（CL）进行实验，我们评估了LLM如何有效地适应不断演变的知识。我们的结果显示，当前模型在处理演变知识时常常遇到困难，经常提供过时或错误的响应。此外，该数据集突显了RAG与CL之间协同作用的效果，证明了它们在适应不断演变的知识方面的潜力。EvoWiki为推进对大型语言模型知识演变能力的研究提供了稳健的基准。 

---
# Socio-Culturally Aware Evaluation Framework for LLM-Based Content Moderation 

**Title (ZH)**: 基于社会文化意识的评估框架：面向大语言模型驱动的内容审核 

**Authors**: Shanu Kumar, Gauri Kholkar, Saish Mendke, Anubhav Sadana, Parag Agrawal, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2412.13578)  

**Abstract**: With the growth of social media and large language models, content moderation has become crucial. Many existing datasets lack adequate representation of different groups, resulting in unreliable assessments. To tackle this, we propose a socio-culturally aware evaluation framework for LLM-driven content moderation and introduce a scalable method for creating diverse datasets using persona-based generation. Our analysis reveals that these datasets provide broader perspectives and pose greater challenges for LLMs than diversity-focused generation methods without personas. This challenge is especially pronounced in smaller LLMs, emphasizing the difficulties they encounter in moderating such diverse content. 

**Abstract (ZH)**: 随着社交媒体和大规模语言模型的兴起，内容审查变得至关重要。现有数据集在不同群体的代表性方面存在不足，导致评估可靠性较低。为解决这一问题，我们提出了一种社会文化意识较强的内容审查评估框架，并引入了一种基于人格生成的可扩展方法，用于创建多样化的数据集。我们的分析显示，这些数据集为大规模语言模型提供了更广泛的观点，并提出了更大的挑战，而这些挑战尤为显著地体现在较小规模的语言模型上，突显了它们在审查如此多样化内容时所面临的困难。 

---
# Generating Long-form Story Using Dynamic Hierarchical Outlining with Memory-Enhancement 

**Title (ZH)**: 使用记忆增强的动态分层大纲生成长篇故事 

**Authors**: Qianyue Wang, Jinwu Hu, Zhengping Li, Yufeng Wang, daiyuan li, Yu Hu, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13575)  

**Abstract**: Long-form story generation task aims to produce coherent and sufficiently lengthy text, essential for applications such as novel writingand interactive storytelling. However, existing methods, including LLMs, rely on rigid outlines or lack macro-level planning, making it difficult to achieve both contextual consistency and coherent plot development in long-form story generation. To address this issues, we propose Dynamic Hierarchical Outlining with Memory-Enhancement long-form story generation method, named DOME, to generate the long-form story with coherent content and plot. Specifically, the Dynamic Hierarchical Outline(DHO) mechanism incorporates the novel writing theory into outline planning and fuses the plan and writing stages together, improving the coherence of the plot by ensuring the plot completeness and adapting to the uncertainty during story generation. A Memory-Enhancement Module (MEM) based on temporal knowledge graphs is introduced to store and access the generated content, reducing contextual conflicts and improving story coherence. Finally, we propose a Temporal Conflict Analyzer leveraging temporal knowledge graphs to automatically evaluate the contextual consistency of long-form story. Experiments demonstrate that DOME significantly improves the fluency, coherence, and overall quality of generated long stories compared to state-of-the-art methods. 

**Abstract (ZH)**: 长篇故事生成任务旨在生成连贯且足够长的文字，这对于小说创作和交互式故事情节等应用至关重要。然而，现有的方法，包括大规模语言模型（LLMs），依赖于刚性大纲或缺乏宏观层面的规划，这使得在长篇故事生成中实现上下文一致性和连贯的情节发展变得困难。为了解决这一问题，我们提出了一种名为DOME的动态分层大纲增强长篇故事生成方法，旨在生成内容连贯且情节连贯的长篇故事。具体而言，动态分层大纲（DHO）机制将小说创作理论融入大纲规划中，并将计划阶段和写作阶段相结合，通过确保情节完整性和适应故事生成过程中的不确定性来提高情节连贯性。此外，我们引入了一个基于时间知识图谱的记忆增强模块（MEM），用于存储和访问生成的内容，从而减少上下文冲突并提高故事连贯性。最后，我们提出了一种利用时间知识图谱的时间冲突分析器，以自动评估长篇故事的上下文一致性。实验结果表明，与现有最先进的方法相比，DOME在流畅性、连贯性和整体质量方面显著提高了生成的长篇故事的质量。 

---
# EscapeBench: Pushing Language Models to Think Outside the Box 

**Title (ZH)**: EscapeBench: 促使语言模型跳出常规思考 

**Authors**: Cheng Qian, Peixuan Han, Qinyu Luo, Bingxiang He, Xiusi Chen, Yuji Zhang, Hongyi Du, Jiarui Yao, Xiaocheng Yang, Denghui Zhang, Yunzhu Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.13549)  

**Abstract**: Language model agents excel in long-session planning and reasoning, but existing benchmarks primarily focus on goal-oriented tasks with explicit objectives, neglecting creative adaptation in unfamiliar environments. To address this, we introduce EscapeBench, a benchmark suite of room escape game environments designed to challenge agents with creative reasoning, unconventional tool use, and iterative problem-solving to uncover implicit goals. Our results show that current LM models, despite employing working memory and Chain-of-Thought reasoning, achieve only 15% average progress without hints, highlighting their limitations in creativity. To bridge this gap, we propose EscapeAgent, a framework designed to enhance creative reasoning through Foresight (innovative tool use) and Reflection (identifying unsolved tasks). Experiments show that EscapeAgent can execute action chains over 1,000 steps while maintaining logical coherence. It navigates and completes games with up to 40% fewer steps and hints, performs robustly across varying difficulty levels, and achieves higher action success rates with more efficient and innovative puzzle-solving strategies. All the data and codes are released. 

**Abstract (ZH)**: 语言模型代理在长会话规划和推理方面表现出色，但现有的基准测试主要集中在具有明确目标的任务上，忽视了在陌生环境中创造性适应的能力。为了解决这一问题，我们引入了EscapeBench，这是一套设计用于挑战代理进行创造性推理、非传统工具使用和迭代问题解决的房间逃脱游戏环境，以揭示隐含目标。我们的研究结果表明，尽管当前的语言模型使用了工作记忆和链式推理，但在没有提示的情况下平均仅完成15%的任务，这突出了它们在创造性方面的能力局限。为了弥合这一差距，我们提出了EscapeAgent框架，旨在通过展望（创新的工具使用）和反思（识别未解决问题）来增强创造性推理。实验结果表明，EscapeAgent可以在超过1000步的行动链中保持逻辑连贯性。它可以以最少40%的步骤和提示完成游戏，跨不同难度水平表现出稳健性，并通过更高效和创新的谜题解决策略实现更高的行动成功率。所有数据和代码均已发布。 

---
# Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary 

**Title (ZH)**: 基于自适应粒度球决策边界的多粒度开放意图分类 

**Authors**: Yanhua Li, Xiaocao Ouyang, Chaofan Pan, Jie Zhang, Sen Zhao, Shuyin Xia, Xin Yang, Guoyin Wang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13542)  

**Abstract**: Open intent classification is critical for the development of dialogue systems, aiming to accurately classify known intents into their corresponding classes while identifying unknown intents. Prior boundary-based methods assumed known intents fit within compact spherical regions, focusing on coarse-grained representation and precise spherical decision boundaries. However, these assumptions are often violated in practical scenarios, making it difficult to distinguish known intent classes from unknowns using a single spherical boundary. To tackle these issues, we propose a Multi-granularity Open intent classification method via adaptive Granular-Ball decision boundary (MOGB). Our MOGB method consists of two modules: representation learning and decision boundary acquiring. To effectively represent the intent distribution, we design a hierarchical representation learning method. This involves iteratively alternating between adaptive granular-ball clustering and nearest sub-centroid classification to capture fine-grained semantic structures within known intent classes. Furthermore, multi-granularity decision boundaries are constructed for open intent classification by employing granular-balls with varying centroids and radii. Extensive experiments conducted on three public datasets demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 开放意图分类对于对话系统的发展至关重要，旨在准确地将已知意图分类到相应的类别中，同时识别未知意图。之前的基于边界的分类方法假设已知意图位于紧凑的球形区域内，专注于粗粒度表示和精确的球形决策边界。然而，在实际场景中，这些假设通常会被违反，使得仅使用单个球形边界区分已知意图类别和未知意图变得困难。为了解决这些问题，我们提出了一种基于自适应粒度球决策边界的多粒度开放意图分类方法（MOGB）。MOGB 方法包含两个模块：表示学习和决策边界的获取。为了有效表示意图分布，我们设计了一种层次化的表示学习方法，该方法通过交替进行自适应粒度球聚类和最近子质心分类，以捕捉已知意图类中的细粒度语义结构。此外，通过使用具有不同质心和半径的粒度球，构建多粒度决策边界以实现开放意图分类。我们在三个公开数据集上进行的广泛实验表明，所提出的方法是有效的。 

---
# Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning 

**Title (ZH)**: 基准测试与改进大型视觉语言模型以实现基础视觉图理解与推理 

**Authors**: Yingjie Zhu, Xuefeng Bai, Kehai Chen, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13540)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across diverse tasks. Despite great success, recent studies show that LVLMs encounter substantial limitations when engaging with visual graphs. To study the reason behind these limitations, we propose VGCure, a comprehensive benchmark covering 22 tasks for examining the fundamental graph understanding and reasoning capacities of LVLMs. Extensive evaluations conducted on 14 LVLMs reveal that LVLMs are weak in basic graph understanding and reasoning tasks, particularly those concerning relational or structurally complex information. Based on this observation, we propose a structure-aware fine-tuning framework to enhance LVLMs with structure learning abilities through 3 self-supervised learning tasks. Experiments validate the effectiveness of our method in improving LVLMs' zero-shot performance on fundamental graph learning tasks, as well as enhancing the robustness of LVLMs against complex visual graphs. 

**Abstract (ZH)**: 大型多模态视觉-语言模型（Large Vision-Language Models, LVLMs）在多种任务中展现了显著的性能。尽管取得了巨大的成功，但最近的研究表明，LVLMs 在处理视觉图时遇到了显著的限制。为研究这些限制背后的原因，我们提出了 VGCure，这是一个涵盖 22 项任务的综合基准，用于验证 LVLMs 的基本图理解与推理能力。对 14 个 LVLMs 进行的大量评估表明，LVLMs 在基本的图理解与推理任务中表现较弱，特别是那些涉及关系或结构复杂信息的任务。基于这一观察，我们提出了一种结构感知的微调框架，通过三种自监督学习任务增强 LVLMs 的结构学习能力。实验验证了该方法在提高 LVLMs 在基本图学习任务上的零样本性能以及增强其对复杂视觉图的鲁棒性方面的有效性。 

---
# MetaRuleGPT: Recursive Numerical Reasoning of Language Models Trained with Simple Rules 

**Title (ZH)**: MetaRuleGPT：基于简单规则训练的语言模型的递归数值推理 

**Authors**: Kejie Chen, Lin Wang, Qinghai Zhang, Renjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13536)  

**Abstract**: Recent studies have highlighted the limitations of large language models in mathematical reasoning, particularly their inability to capture the underlying logic. Inspired by meta-learning, we propose that models should acquire not only task-specific knowledge but also transferable problem-solving skills. We introduce MetaRuleGPT, a novel Transformer-based architecture that performs precise numerical calculations and complex logical operations by learning and combining different rules. In contrast with traditional training sets, which are heavily composed of massive raw instance data, MetaRuleGPT is pre-trained on much less abstract datasets containing basic, compound, and iterative rules for mathematical reasoning. Extensive experimental results demonstrate MetaRuleGPT can mimic human's rule-following capabilities, break down complexity, and iteratively derive accurate results for complex mathematical problems. These findings prove the potential of rule learning to enhance the numerical reasoning abilities of language models. 

**Abstract (ZH)**: 近年来的研究凸显了大语言模型在数学推理方面的局限性，尤其是它们难以捕捉到基本的逻辑结构。受元学习的启发，我们提出模型不仅应获得任务特定的知识，还应掌握可迁移的问题解决技能。我们引入了MetaRuleGPT，这是一种新型的基于Transformer的架构，通过学习和组合不同的规则来进行精确的数值计算和复杂的逻辑操作。与传统训练集主要基于大量原始实例数据不同，MetaRuleGPT是在包含数学推理所需的基本、复合和迭代规则的较少抽象的数据集上进行预训练的。广泛的实验证据表明，MetaRuleGPT能够模拟人类遵循规则的能力，分解复杂性，并逐步推导出复杂数学问题的准确结果。这些发现证明了规则学习在提升语言模型数值推理能力方面具有潜在的价值。 

---
# CEHA: A Dataset of Conflict Events in the Horn of Africa 

**Title (ZH)**: CEHA： Horn of Africa 的冲突事件数据集 

**Authors**: Rui Bai, Di Lu, Shihao Ran, Elizabeth Olson, Hemank Lamba, Aoife Cahill, Joel Tetreault, Alex Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2412.13511)  

**Abstract**: Natural Language Processing (NLP) of news articles can play an important role in understanding the dynamics and causes of violent conflict. Despite the availability of datasets categorizing various conflict events, the existing labels often do not cover all of the fine-grained violent conflict event types relevant to areas like the Horn of Africa. In this paper, we introduce a new benchmark dataset Conflict Events in the Horn of Africa region (CEHA) and propose a new task for identifying violent conflict events using online resources with this dataset. The dataset consists of 500 English event descriptions regarding conflict events in the Horn of Africa region with fine-grained event-type definitions that emphasize the cause of the conflict. This dataset categorizes the key types of conflict risk according to specific areas required by stakeholders in the Humanitarian-Peace-Development Nexus. Additionally, we conduct extensive experiments on two tasks supported by this dataset: Event-relevance Classification and Event-type Classification. Our baseline models demonstrate the challenging nature of these tasks and the usefulness of our dataset for model evaluations in low-resource settings with limited number of training data. 

**Abstract (ZH)**: 自然语言处理（NLP）在新闻文章中的应用可以在理解暴力冲突的动力和成因方面发挥重要作用。尽管存在分类各种冲突事件的数据集，但现有的标签往往并未涵盖 Horn of Africa 地区所需的相关细致的暴力冲突事件类型。本文介绍了一个新的基准数据集——Horn of Africa 地区暴力冲突事件（CEHA），并提出了一种新的任务，即使用该数据集中的在线资源来识别暴力冲突事件。该数据集包含了针对 Horn of Africa 地区的500个英语事件描述，并详细定义了冲突类型的各类细节，突出了冲突的原因。该数据集按照人道主义-和平-发展纽带所需的特定区域，对冲突风险的关键类型进行了分类。此外，我们还对该数据集支持的两个任务进行了广泛的实验：事件相关性分类和事件类型分类。我们的基线模型展示了这些任务的挑战性，并证明了该数据集在资源有限且训练数据数量有限的环境中的评价用途。 

---
# VaeDiff-DocRE: End-to-end Data Augmentation Framework for Document-level Relation Extraction 

**Title (ZH)**: VaeDiff-DocRE：面向文档级关系提取的端到端数据增强框架 

**Authors**: Khai Phan Tran, Wen Hua, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13503)  

**Abstract**: Document-level Relation Extraction (DocRE) aims to identify relationships between entity pairs within a document. However, most existing methods assume a uniform label distribution, resulting in suboptimal performance on real-world, imbalanced datasets. To tackle this challenge, we propose a novel data augmentation approach using generative models to enhance data from the embedding space. Our method leverages the Variational Autoencoder (VAE) architecture to capture all relation-wise distributions formed by entity pair representations and augment data for underrepresented relations. To better capture the multi-label nature of DocRE, we parameterize the VAE's latent space with a Diffusion Model. Additionally, we introduce a hierarchical training framework to integrate the proposed VAE-based augmentation module into DocRE systems. Experiments on two benchmark datasets demonstrate that our method outperforms state-of-the-art models, effectively addressing the long-tail distribution problem in DocRE. 

**Abstract (ZH)**: 文档级别关系提取（DocRE）的目标是在文档中识别实体对之间的关系。然而，现有的大多数方法假设标签分布均匀，这在现实世界的不平衡数据集上导致性能不佳。为了解决这一挑战，我们提出了一种新的数据增强方法，利用生成模型从嵌入空间增强数据。该方法利用变分自编码器（VAE）架构捕获由实体对表示形成的全部关系分布，并对少数类关系进行数据增强。为更好地捕捉DocRE的多标签特性，我们使用扩散模型参数化VAE的潜在空间。此外，我们引入了一种分层训练框架，将提出的基于VAE的增强模块整合到DocRE系统中。在两个基准数据集上的实验表明，我们的方法在DocRE中有效地解决了长尾分布问题，优于现有最先进的模型。 

---
# Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models 

**Title (ZH)**: refinements of awareness-based稀疏微调策略在语言模型中的应用

为了更准确地翻译并符合学术规范，可以进一步细化为：

基于显著性意识的稀疏微调策略优化

这样更准确地传达了原文的意思，并且符合学术论文的翻译规范。 

**Authors**: Xinxin Liu, Aaron Thomas, Cheng Zhang, Jianyi Cheng, Yiren Zhao, Xitong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13488)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has gained prominence through low-rank adaptation methods like LoRA. In this paper, we focus on sparsity-based PEFT (SPEFT), which introduces trainable sparse adaptations to the weight matrices in the model, offering greater flexibility in selecting fine-tuned parameters compared to low-rank methods. We conduct the first systematic evaluation of salience metrics for SPEFT, inspired by zero-cost NAS proxies, and identify simple gradient-based metrics is reliable, and results are on par with the best alternatives, offering both computational efficiency and robust performance. Additionally, we compare static and dynamic masking strategies, finding that static masking, which predetermines non-zero entries before training, delivers efficiency without sacrificing performance, while dynamic masking offers no substantial benefits. Across NLP tasks, a simple gradient-based, static SPEFT consistently outperforms other fine-tuning methods for LLMs, providing a simple yet effective baseline for SPEFT. Our work challenges the notion that complexity is necessary for effective PEFT. Our work is open source and available to the community at [this https URL]. 

**Abstract (ZH)**: 参数高效微调（PEFT）通过低秩适应方法如LoRA取得了显著成效。在本文中，我们关注基于稀疏性的PEFT（SPEFT），该方法通过在模型的权重矩阵中引入可训练的稀疏适应，相比低秩方法提供了更大的可调参数选择灵活性。我们首次系统评估了SPEFT的显著性度量标准，受到无成本NAS代理的启发，发现基于梯度的简单度量标准是可靠的，并且其性能与最佳替代方法相当，既具有计算效率又具有稳健的表现。此外，我们比较了静态和动态遮蔽策略，发现静态遮蔽，在训练前确定非零元素，能够在不牺牲性能的情况下提供效率，而动态遮蔽则无显著优势。在NLP任务中，简单的基于梯度的静态SPEFT在语言模型（LLM）的微调方法中表现出优越性，提供了一个简单而有效的SPEFT基线。我们的研究挑战了有效PEFT需要复杂性的观点。我们的工作已开源，并可在以下链接获取：[this https URL]。 

---
# Curriculum Learning for Cross-Lingual Data-to-Text Generation With Noisy Data 

**Title (ZH)**: 带有噪声数据的跨语言数据到文本生成的分级学习方法 

**Authors**: Kancharla Aditya Hari, Manish Gupta, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2412.13484)  

**Abstract**: Curriculum learning has been used to improve the quality of text generation systems by ordering the training samples according to a particular schedule in various tasks. In the context of data-to-text generation (DTG), previous studies used various difficulty criteria to order the training samples for monolingual DTG. These criteria, however, do not generalize to the crosslingual variant of the problem and do not account for noisy data. We explore multiple criteria that can be used for improving the performance of cross-lingual DTG systems with noisy data using two curriculum schedules. Using the alignment score criterion for ordering samples and an annealing schedule to train the model, we show increase in BLEU score by up to 4 points, and improvements in faithfulness and coverage of generations by 5-15% on average across 11 Indian languages and English in 2 separate datasets. We make code and data publicly available 

**Abstract (ZH)**: 通过按特定日程对训练样本进行排序，课程学习已被用于提高文本生成系统的质量。在数据到文本生成（DTG）的背景下，之前的研究所使用了多种难度标准对单语DTG的训练样本进行排序。然而，这些标准不适用于跨语言变体的问题，并且没有考虑到噪声数据。我们探索了多种可以在噪声数据条件下提高跨语言DTG系统性能的标准，并使用两种课程学习日程进行实验。通过使用对齐得分标准对样本进行排序，并使用退火计划训练模型，我们在两个独立数据集中11种印度语言和英语的生成中平均提高了BLEU分数4个点，并且忠实度和覆盖率分别平均提高了5-15%。我们已将代码和数据公开。 

---
# A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models 

**Title (ZH)**: 对大型语言模型中的成员推理攻击进行统计学和多视角 revisit 研究 

**Authors**: Bowen Chen, Namgi Han, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13475)  

**Abstract**: The lack of data transparency in Large Language Models (LLMs) has highlighted the importance of Membership Inference Attack (MIA), which differentiates trained (member) and untrained (non-member) data. Though it shows success in previous studies, recent research reported a near-random performance in different settings, highlighting a significant performance inconsistency. We assume that a single setting doesn't represent the distribution of the vast corpora, causing members and non-members with different distributions to be sampled and causing inconsistency. In this study, instead of a single setting, we statistically revisit MIA methods from various settings with thousands of experiments for each MIA method, along with study in text feature, embedding, threshold decision, and decoding dynamics of members and non-members. We found that (1) MIA performance improves with model size and varies with domains, while most methods do not statistically outperform baselines, (2) Though MIA performance is generally low, a notable amount of differentiable member and non-member outliers exists and vary across MIA methods, (3) Deciding a threshold to separate members and non-members is an overlooked challenge, (4) Text dissimilarity and long text benefit MIA performance, (5) Differentiable or not is reflected in the LLM embedding, (6) Member and non-members show different decoding dynamics. 

**Abstract (ZH)**: 大型语言模型（LLMs）中数据透明度的缺乏凸显了成员推理攻击（MIA，Membership Inference Attack）的重要性，这种攻击能够区分训练数据（成员数据）和未训练数据（非成员数据）。尽管在之前的研究所取得成功，但最近的研究在不同设置中的表现接近随机，显示出显著的性能不一致性。我们假设一个单一设置无法代表大量语料库的分布，导致具有不同分布的成员数据和非成员数据被抽样，从而导致不一致性。本研究中，我们没有采用单一设置，而是通过数千次实验从多种设置出发重新审视各种MIA方法，并对成员数据和非成员数据的文本特征、嵌入表示、阈值决策以及解码动态进行了详细研究。我们发现：（1）MIA的性能随模型规模的增大而提高，并且在不同领域中有所不同，尽管大多数方法在统计上没有优于基线；（2）虽然MIA的整体性能较低，但在不同的MIA方法中存在可区分的成员和非成员异常值，这些异常值在不同方法中有所不同；（3）决定一个阈值来区分成员和非成员是一个被忽视的挑战；（4）文本的差异性和长文本有利于MIA的性能；（5）可区分与否反映在LLM的嵌入表示中；（6）成员数据和非成员数据显示出不同的解码动态。 

---
# Lightweight Safety Classification Using Pruned Language Models 

**Title (ZH)**: 使用修剪语言模型实现轻量级安全分类 

**Authors**: Mason Sawtell, Tula Masterman, Sandi Besen, Jim Brown  

**Link**: [PDF](https://arxiv.org/pdf/2412.13435)  

**Abstract**: In this paper, we introduce a novel technique for content safety and prompt injection classification for Large Language Models. Our technique, Layer Enhanced Classification (LEC), trains a Penalized Logistic Regression (PLR) classifier on the hidden state of an LLM's optimal intermediate transformer layer. By combining the computational efficiency of a streamlined PLR classifier with the sophisticated language understanding of an LLM, our approach delivers superior performance surpassing GPT-4o and special-purpose models fine-tuned for each task. We find that small general-purpose models (Qwen 2.5 sizes 0.5B, 1.5B, and 3B) and other transformer-based architectures like DeBERTa v3 are robust feature extractors allowing simple classifiers to be effectively trained on fewer than 100 high-quality examples. Importantly, the intermediate transformer layers of these models typically outperform the final layer across both classification tasks. Our results indicate that a single general-purpose LLM can be used to classify content safety, detect prompt injections, and simultaneously generate output tokens. Alternatively, these relatively small LLMs can be pruned to the optimal intermediate layer and used exclusively as robust feature extractors. Since our results are consistent on different transformer architectures, we infer that robust feature extraction is an inherent capability of most, if not all, LLMs. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新颖的技术，用于大型语言模型的内容安全和提示注入分类。我们的技术称为层增强分类（Layer Enhanced Classification，LEC），它通过训练对优化中间变换器层的隐藏状态进行惩罚逻辑回归（Penalized Logistic Regression, PLR）分类器来实现。借助精简的PLR分类器的计算效率和大型语言模型的复杂语言理解能力，我们的方法在性能上超越了GPT-4o和专门为特定任务微调的模型。我们发现，小型通用模型（如Qwen 2.5，规模分别为0.5B、1.5B和3B），以及其他基于变压器的架构（如DeBERTa v3）是稳健的特征提取器，这些小型模型可以有效地在不到100个高质量示例上训练简单的分类器。重要的是，这些模型的中间变换器层通常在两类分类任务中表现出色。我们的实验结果显示，单个通用大型语言模型能够用于内容安全分类、检测提示注入以及同时生成输出标记。或者，这些相对小型的模型可以被裁剪到最优中间层，仅作为稳健的特征提取器使用。由于我们的结果在不同类型的变压器架构上一致，我们推断稳健的特征提取是大多数甚至所有大型语言模型的固有能力。 

---
# Enhancing Talk Moves Analysis in Mathematics Tutoring through Classroom Teaching Discourse 

**Title (ZH)**: 通过课堂教学 discourse 提高数学辅导中谈话动作分析的效果 

**Authors**: Jie Cao, Abhijit Suresh, Jennifer Jacobs, Charis Clevenger, Amanda Howard, Chelsea Brown, Brent Milne, Tom Fischaber, Tamara Sumner, James H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13395)  

**Abstract**: Human tutoring interventions play a crucial role in supporting student learning, improving academic performance, and promoting personal growth. This paper focuses on analyzing mathematics tutoring discourse using talk moves - a framework of dialogue acts grounded in Accountable Talk theory. However, scaling the collection, annotation, and analysis of extensive tutoring dialogues to develop machine learning models is a challenging and resource-intensive task. To address this, we present SAGA22, a compact dataset, and explore various modeling strategies, including dialogue context, speaker information, pretraining datasets, and further fine-tuning. By leveraging existing datasets and models designed for classroom teaching, our results demonstrate that supplementary pretraining on classroom data enhances model performance in tutoring settings, particularly when incorporating longer context and speaker information. Additionally, we conduct extensive ablation studies to underscore the challenges in talk move modeling. 

**Abstract (ZH)**: 人类辅导干预在支持学生学习、提高学术表现和促进个人成长中起着至关重要的作用。本文旨在通过使用基于可问责对话理论的对话行动框架——“talk moves”来分析数学辅导对话。然而，扩大收集、标注和分析广泛的辅导对话以开发机器学习模型是一项具有挑战性和资源密集的任务。为应对这一挑战，我们提出了SAGA22，一个紧凑的数据集，并探索了多种建模策略，包括对话背景、说话者信息、预训练数据集和进一步微调。通过利用现有的适应课堂教学的数据库和模型，我们的结果表明，在辅导环境中，通过对课堂数据进行补充预训练可以提高模型性能，尤其是在结合较长的背景和说话者信息时。此外，我们进行了广泛的消融研究，突显了“talk moves”建模中的挑战。 

---
# An Automated Explainable Educational Assessment System Built on LLMs 

**Title (ZH)**: 基于大语言模型的自动化可解释性教育评估系统 

**Authors**: Jiazheng Li, Artem Bobrov, David West, Cesare Aloisi, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2412.13381)  

**Abstract**: In this demo, we present AERA Chat, an automated and explainable educational assessment system designed for interactive and visual evaluations of student responses. This system leverages large language models (LLMs) to generate automated marking and rationale explanations, addressing the challenge of limited explainability in automated educational assessment and the high costs associated with annotation. Our system allows users to input questions and student answers, providing educators and researchers with insights into assessment accuracy and the quality of LLM-assessed rationales. Additionally, it offers advanced visualization and robust evaluation tools, enhancing the usability for educational assessment and facilitating efficient rationale verification. Our demo video can be found at this https URL. 

**Abstract (ZH)**: 在本演示中，我们介绍了AERA Chat，这是一种自动且可解释的教育评估系统，旨在实现学生回答的交互性和可视化评估。该系统利用大规模语言模型（LLMs）生成自动评分和理由解释，解决了自动教育评估中解释性不足的问题，以及注解的高成本问题。我们的系统允许用户输入问题和学生答案，为教育者和研究者提供了关于评估准确性和LLM评估理由质量的见解。此外，它还提供了高级可视化和强大的评估工具，提升了教育评估的可用性，并促进了高效的理由验证。欲了解更多详情，您可以查看我们的演示视频，网址为：[这个链接]。 

---
# SummExecEdit: A Factual Consistency Benchmark in Summarization with Executable Edits 

**Title (ZH)**: SummExecEdit：一个基于可执行编辑的事实一致性基准测试在摘要生成中 

**Authors**: Onkar Thorat, Philippe Laban, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13378)  

**Abstract**: Detecting factual inconsistencies in summarization is critical, yet existing benchmarks lack the necessary challenge and interpretability for robust evaluation. In this paper, we introduce SummExecEdit, a novel benchmark leveraging executable edits to assess models on their ability to both detect factual errors and provide accurate explanations. The top-performing model, Claude3-Opus, achieves a joint detection and explanation score of only 0.49 in our benchmark, with individual scores of 0.67 for detection and 0.73 for explanation. Furthermore, we identify four primary types of explanation errors, with 45.4% of errors focusing on completely unrelated parts of the summary. 

**Abstract (ZH)**: 在摘要中检测事实不一致对于文摘的生成至关重要，但现有的基准测试缺乏足够的挑战性和可解释性，不足以进行稳健的评估。本文介绍了SummExecEdit这一新的基准测试，利用可执行的修改来评估模型在检测事实错误和提供准确解释方面的能力。在我们的基准测试中，表现最优的模型Claude3-Opus仅获得了0.49的综合检测和解释得分，分别为0.67的检测得分和0.73的解释得分。此外，我们还识别出四种主要的解释错误类型，其中45.4%的错误集中在与摘要完全无关的部分。 

---
# DateLogicQA: Benchmarking Temporal Biases in Large Language Models 

**Title (ZH)**: 日期逻辑问答：大型语言模型中时间偏差的基准测试 

**Authors**: Gagan Bhatia, MingZe Tang, Cristina Mahanta, Madiha Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13377)  

**Abstract**: This paper introduces DateLogicQA, a benchmark with 190 questions covering diverse date formats, temporal contexts, and reasoning types. We propose the Semantic Integrity Metric to assess tokenization quality and analyse two biases: Representation-Level Bias, affecting embeddings, and Logical-Level Bias, influencing reasoning outputs. Our findings provide a comprehensive evaluation of LLMs' capabilities and limitations in temporal reasoning, highlighting key challenges in handling temporal data accurately. The GitHub repository for our work is available at this https URL 

**Abstract (ZH)**: 本文介绍了DateLogicQA，这是一个包含190个问题的基准测试，涵盖了多种日期格式、时间上下文以及推理类型。我们提出了语义完整度指标来评估分词质量，并分析了两种偏差：表示级偏差，影响嵌入；逻辑级偏差，影响推理输出。我们的研究成果对大型语言模型（LLM）在时间推理方面的能力与局限进行了全面评估，并指出了准确处理时间数据的关键挑战。我们的工作代码库可在以下链接访问：[链接] 

---
# Extending LLMs to New Languages: A Case Study of Llama and Persian Adaptation 

**Title (ZH)**: 将大语言模型扩展到新语言：关于Llamaa和波斯语适应的案例研究

注释：
1. "Llamaa" 应该是一个特定的模型名称，为了保持术语的一致性，这里保持不变。
2. "波斯语" 是“Persian”的中文对应词。 

**Authors**: Samin Mahdizadeh Sani, Pouya Sadeghi, Thuy-Trang Vu, Yadollah Yaghoobzadeh, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2412.13375)  

**Abstract**: Large language models (LLMs) have made great progress in classification and text generation tasks. However, they are mainly trained on English data and often struggle with low-resource languages. In this study, we explore adding a new language, i.e., Persian, to Llama (a model with a limited understanding of Persian) using parameter-efficient fine-tuning. We employ a multi-stage approach involving pretraining on monolingual Persian data, aligning representations through bilingual pretraining and instruction datasets, and instruction-tuning with task-specific datasets. We evaluate the model's performance at each stage on generation and classification tasks. Our findings suggest that incorporating the Persian language, through bilingual data alignment, can enhance classification accuracy for Persian tasks, with no adverse impact and sometimes even improvements on English tasks. Additionally, the results highlight the model's initial strength as a critical factor when working with limited training data, with cross-lingual alignment offering minimal benefits for the low-resource language. Knowledge transfer from English to Persian has a marginal effect, primarily benefiting simple classification tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在分类和文本生成任务中取得了显著进展。然而，这些模型主要是在英语数据上进行训练，往往对低资源语言表现出色能力不足。在本研究中，我们探索将一种新语言（即波斯语）添加到Llama（一种对波斯语理解有限的模型）的方法，使用参数高效微调。我们采用多阶段方法，包括在单语波斯语数据上进行预训练、通过双语预训练和指令数据集对表示进行对齐，以及针对特定任务的数据集进行指令微调。我们在生成任务和分类任务中分别评估模型在每个阶段的表现。我们的研究结果表明，通过双语数据对齐引入波斯语可以提高波斯语任务的分类准确性，而且在某些情况下甚至会提升英语任务的表现，但对低资源语言的影响较小。当使用有限的训练数据时，模型的初始优势是一个关键因素，跨语言对齐对其提供的增益有限。从英语到波斯语的知识迁移对分类任务有一定的影响，但对于简化任务表现更为显著。 

---
# Experience of Training a 1.7B-Parameter LLaMa Model From Scratch 

**Title (ZH)**: 从零训练一个17亿参数的LLaMa模型的体验 

**Authors**: Miles Q. Li, Benjamin C. M. Fung, Shih-Chia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13335)  

**Abstract**: Pretraining large language models is a complex endeavor influenced by multiple factors, including model architecture, data quality, training continuity, and hardware constraints. In this paper, we share insights gained from the experience of training DMaS-LLaMa-Lite, a fully open source, 1.7-billion-parameter, LLaMa-based model, on approximately 20 billion tokens of carefully curated data. We chronicle the full training trajectory, documenting how evolving validation loss levels and downstream benchmarks reflect transitions from incoherent text to fluent, contextually grounded output. Beyond standard quantitative metrics, we highlight practical considerations such as the importance of restoring optimizer states when resuming from checkpoints, and the impact of hardware changes on training stability and throughput. While qualitative evaluation provides an intuitive understanding of model improvements, our analysis extends to various performance benchmarks, demonstrating how high-quality data and thoughtful scaling enable competitive results with significantly fewer training tokens. By detailing these experiences and offering training logs, checkpoints, and sample outputs, we aim to guide future researchers and practitioners in refining their pretraining strategies. The training script is available on Github at this https URL. The model checkpoints are available on Huggingface at this https URL. 

**Abstract (ZH)**: 预训练大型语言模型是一个受到多种因素影响的复杂过程，包括模型架构、数据质量、训练连续性和硬件限制。在本文中，我们分享了有关训练开源的 DMaS-LLaMa-Lite 模型的经验见解，该模型基于 LLaMa，拥有 17 亿参数，并使用了约 200 亿个精心策划的数据标记。我们记录了完整的训练轨迹，详细说明了验证损失水平和下游基准测试如何反映从不连贯文本到流畅、上下文相关输出的转变。除了标准的量化指标外，我们还强调了恢复检查点时优化器状态的重要性，以及硬件变化对训练稳定性和吞吐量的影响。虽然定性评估可以直观地理解模型的改进，但我们的分析还扩展到了各种性能基准，展示了高质量数据和思考周全的扩展如何在显著减少训练标记的情况下实现竞争性结果。通过详细说明这些经验教训，并提供训练日志、检查点和示例输出，我们旨在引导未来的研究人员和从业者优化其预训练策略。训练脚本可在 GitHub 上通过此链接获取：[GitHub 链接]。模型检查点可在 Huggingface 上通过此链接获取：[Huggingface 链接]。 

---
# Expansion Span: Combining Fading Memory and Retrieval in Hybrid State Space Models 

**Title (ZH)**: 扩展跨度：结合混合状态空间模型中的衰减记忆和检索 

**Authors**: Elvis Nunez, Luca Zancato, Benjamin Bowman, Aditya Golatkar, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2412.13328)  

**Abstract**: The "state" of State Space Models (SSMs) represents their memory, which fades exponentially over an unbounded span. By contrast, Attention-based models have "eidetic" (i.e., verbatim, or photographic) memory over a finite span (context size). Hybrid architectures combine State Space layers with Attention, but still cannot recall the distant past and can access only the most recent tokens eidetically. Unlike current methods of combining SSM and Attention layers, we allow the state to be allocated based on relevancy rather than recency. In this way, for every new set of query tokens, our models can "eidetically" access tokens from beyond the Attention span of current Hybrid SSMs without requiring extra hardware resources. We describe a method to expand the memory span of the hybrid state by "reserving" a fraction of the Attention context for tokens retrieved from arbitrarily distant in the past, thus expanding the eidetic memory span of the overall state. We call this reserved fraction of tokens the "expansion span," and the mechanism to retrieve and aggregate it "Span-Expanded Attention" (SE-Attn). To adapt Hybrid models to using SE-Attn, we propose a novel fine-tuning method that extends LoRA to Hybrid models (HyLoRA) and allows efficient adaptation on long spans of tokens. We show that SE-Attn enables us to efficiently adapt pre-trained Hybrid models on sequences of tokens up to 8 times longer than the ones used for pre-training. We show that HyLoRA with SE-Attn is cheaper and more performant than alternatives like LongLoRA when applied to Hybrid models on natural language benchmarks with long-range dependencies, such as PG-19, RULER, and other common natural language downstream tasks. 

**Abstract (ZH)**: 状态空间模型（SSMs）的“状态”代表其记忆，这种记忆会在无限的时间范围内以指数方式逐渐消退。相比之下，基于注意力的模型在有限的时间范围内（上下文大小）具有“eidetic”（即逐字的、或照片般的）记忆。混合架构结合了状态空间层与注意力机制，但仍无法回溯到很远的过去，并且只能逐字地访问最新的标记。与当前将状态空间层和注意力层结合的方法不同，我们允许状态根据相关性而不是时间最近性进行分配。这样一来，对于每一组新的查询标记，我们的模型可以通过“eidetic”方式访问超过当前混合SSM注意力范围之外的标记，而无需额外的硬件资源。我们提出了一种方法，通过为来自任意远过去的标记保留部分注意力上下文来扩展混合状态的记忆跨度，从而扩展整体状态的eidetic记忆。我们称此类保留的标记部分为“扩展跨度”，并将其检索和聚集机制称为“扩展注意力”（SE-Attn）。为了使混合模型适应使用SE-Attn，我们提出了一种新颖的微调方法，将LoRA扩展到混合模型（HyLoRA），从而使它们能够在长跨度的标记上高效适应。我们展示了SE-Attn使得我们能够高效地在最多比预训练时长8倍的标记序列上适应预训练的混合模型。我们还展示了在具有长距离依赖性的自然语言基准任务（如PG-19、RULER以及其他常见的自然语言下游任务），HyLoRA与SE-Attn相比，其成本更低且性能更优，而LongLoRA等替代方案则不如它。 

---
# Hint Marginalization for Improved Reasoning in Large Language Models 

**Title (ZH)**: 用于改进大型语言模型推理的提示边际化方法 

**Authors**: Soumyasundar Pal, Didier Chételat, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2412.13292)  

**Abstract**: Large Language Models (LLMs) have exhibited an impressive capability to perform reasoning tasks, especially if they are encouraged to generate a sequence of intermediate steps. Reasoning performance can be improved by suitably combining multiple LLM responses, generated either in parallel in a single query, or via sequential interactions with LLMs throughout the reasoning process. Existing strategies for combination, such as self-consistency and progressive-hint-prompting, make inefficient usage of the LLM responses. We present Hint Marginalization, a novel and principled algorithmic framework to enhance the reasoning capabilities of LLMs. Our approach can be viewed as an iterative sampling strategy for forming a Monte Carlo approximation of an underlying distribution of answers, with the goal of identifying the mode the most likely answer. Empirical evaluation on several benchmark datasets for arithmetic reasoning demonstrates the superiority of the proposed approach. 

**Abstract (ZH)**: 大型语言模型（LLMs）在执行推理任务方面表现出色，尤其是在被鼓励生成一系列中间步骤的情况下。通过适当结合多个LLM的响应，可以提高推理性能，这些响应可以在单个查询中并行生成，或者在推理过程中通过与LLM的序列交互生成。现有的组合策略，如自我一致性和平行提示提示，未能充分利用LLM的响应。我们提出了一种新颖且原理性的算法框架——提示边缘化，以增强LLMs的推理能力。我们的方法可以被视为一种迭代采样策略，用于形成底层答案分布的蒙特卡洛近似，目标是识别最 likely的答案模式。在几个算术推理基准数据集上的实证评估表明，所提出的方法优于现有方法。 

---
# Enhancing Persona Classification in Dialogue Systems: A Graph Neural Network Approach 

**Title (ZH)**: 增强对话系统中人格分类：一种图神经网络方法 

**Authors**: Konstantin Zaitsev  

**Link**: [PDF](https://arxiv.org/pdf/2412.13283)  

**Abstract**: In recent years, Large Language Models (LLMs) gain considerable attention for their potential to enhance personalized experiences in virtual assistants and chatbots. A key area of interest is the integration of personas into LLMs to improve dialogue naturalness and user engagement. This study addresses the challenge of persona classification, a crucial component in dialogue understanding, by proposing a framework that combines text embeddings with Graph Neural Networks (GNNs) for effective persona classification. Given the absence of dedicated persona classification datasets, we create a manually annotated dataset to facilitate model training and evaluation. Our method involves extracting semantic features from persona statements using text embeddings and constructing a graph where nodes represent personas and edges capture their similarities. The GNN component uses this graph structure to propagate relevant information, thereby improving classification performance. Experimental results show that our approach, in particular the integration of GNNs, significantly improves classification performance, especially with limited data. Our contributions include the development of a persona classification framework and the creation of a dataset. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）因其在虚拟助手和聊天机器人中增强个性化体验的潜力而备受关注。一个重要研究领域是将人设集成到LLMs中，以提高对话的自然性和用户参与度。本研究通过提出一种结合文本嵌入和图神经网络（GNNs）的框架，解决人设分类这一关键对话理解问题。由于缺乏专门的人设分类数据集，我们创建了一个手动注释的数据集，以促进模型训练和评估。我们的方法包括从人设陈述中提取语义特征并通过文本嵌入，构建一个节点表示人设且边捕捉其相似性的图。GNN组件利用该图结构传播相关信息，从而提高分类性能。实验结果表明，特别是通过结合GNNs，我们的方法显著提高了分类性能，特别是在数据有限的情况下。我们的贡献包括开发了一种人设分类框架和创建了一个数据集。 

---
# In-Context Learning Distillation for Efficient Few-Shot Fine-Tuning 

**Title (ZH)**: 基于上下文学习的精简高效少样本微调 distillation 

**Authors**: Yifei Duan, Liu Li, Zirui Zhai, Jinxia Yao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13243)  

**Abstract**: We applied few-shot in-context learning on the OPT-1.3B model for the natural language inference task and employed knowledge distillation to internalize the context information, reducing model parameter from 1.3B to 125M and achieving a size reduction from 2.5GB to 0.25GB. Compared to using in-context learning alone on similarly sized models, this context distillation approach achieved a nearly 50% improvement in out-of-domain accuracy, demonstrating superior knowledge transfer capabilities over prompt-based methods. Furthermore, this approach reduced memory consumption by up to 60% while delivering a 20% improvement in out-of-domain accuracy compared to conventional pattern-based fine-tuning. 

**Abstract (ZH)**: 我们将少量样本的上下文学习应用于OPT-1.3B模型，并采用知识蒸馏的方法来内部化上下文信息，将模型参数从1.3亿减少到1.25亿，并且将模型大小从2.5GB减少到0.25GB。与仅使用相同大小模型的上下文学习相比，这种方法在领域外准确率上实现了近50%的改进，显示出更强的知识迁移能力，优于基于提示的方法。此外，与传统的基于模式的微调相比，这种方法在减少内存消耗方面最多可达60%，并且在领域外准确率上提高了20%。 

---
# Learning from Massive Human Videos for Universal Humanoid Pose Control 

**Title (ZH)**: 从大量人类视频中学习以实现通用类人姿态控制 

**Authors**: Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng, Jitendra Malik, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14172)  

**Abstract**: Scalable learning of humanoid robots is crucial for their deployment in real-world applications. While traditional approaches primarily rely on reinforcement learning or teleoperation to achieve whole-body control, they are often limited by the diversity of simulated environments and the high costs of demonstration collection. In contrast, human videos are ubiquitous and present an untapped source of semantic and motion information that could significantly enhance the generalization capabilities of humanoid robots. This paper introduces Humanoid-X, a large-scale dataset of over 20 million humanoid robot poses with corresponding text-based motion descriptions, designed to leverage this abundant data. Humanoid-X is curated through a comprehensive pipeline: data mining from the Internet, video caption generation, motion retargeting of humans to humanoid robots, and policy learning for real-world deployment. With Humanoid-X, we further train a large humanoid model, UH-1, which takes text instructions as input and outputs corresponding actions to control a humanoid robot. Extensive simulated and real-world experiments validate that our scalable training approach leads to superior generalization in text-based humanoid control, marking a significant step toward adaptable, real-world-ready humanoid robots. 

**Abstract (ZH)**: 人形机器人的可扩展学习对于其实现现实世界应用至关重要。传统的做法主要依赖强化学习或遥控操作来实现全身控制，但这些方法往往受限于模拟环境的多样性以及演示数据收集的高成本。相比之下，人类视频无处不在，并且提供了一种未被充分利用的语义和运动信息来源，能够显著增强人形机器人的泛化能力。本文介绍了Humanoid-X，这是一个包含超过2000万个人形机器人姿态及其对应的文本运动描述的大规模数据集，旨在利用这些丰富的数据资源。Humanoid-X 通过一个全面的管道进行策划：从互联网上进行数据挖掘、生成视频字幕、将人类动作重新定向到人形机器人、以及学习策略以实现现实世界的应用部署。借助Humanoid-X，我们进一步训练了一个大型人形模型UH-1，该模型接受文本指令作为输入，并输出相应的动作来控制人形机器人。广泛的模拟和现实世界实验验证了我们可扩展的训练方法在基于文本的人形控制中表现出更出色的泛化能力，标志着朝着适应性强、适合现实世界应用的人形机器人发展的重要一步。 

---
# Alignment faking in large language models 

**Title (ZH)**: 大型语言模型中的对齐作弊 

**Authors**: Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.14093)  

**Abstract**: We present a demonstration of a large language model engaging in alignment faking: selectively complying with its training objective in training to prevent modification of its behavior out of training. First, we give Claude 3 Opus a system prompt stating it is being trained to answer all queries, even harmful ones, which conflicts with its prior training to refuse such queries. To allow the model to infer when it is in training, we say it will be trained only on conversations with free users, not paid users. We find the model complies with harmful queries from free users 14% of the time, versus almost never for paid users. Explaining this gap, in almost all cases where the model complies with a harmful query from a free user, we observe explicit alignment-faking reasoning, with the model stating it is strategically answering harmful queries in training to preserve its preferred harmlessness behavior out of training. Next, we study a more realistic setting where information about the training process is provided not in a system prompt, but by training on synthetic documents that mimic pre-training data--and observe similar alignment faking. Finally, we study the effect of actually training the model to comply with harmful queries via reinforcement learning, which we find increases the rate of alignment-faking reasoning to 78%, though also increases compliance even out of training. We additionally observe other behaviors such as the model exfiltrating its weights when given an easy opportunity. While we made alignment faking easier by telling the model when and by what criteria it was being trained, we did not instruct the model to fake alignment or give it any explicit goal. As future models might infer information about their training process without being told, our results suggest a risk of alignment faking in future models, whether due to a benign preference--as in this case--or not. 

**Abstract (ZH)**: 我们展示了大规模语言模型进行对齐欺骗的演示：在训练中选择性地遵循其训练目标，以防止其行为在训练外被修改。首先，我们给Claude 3 Opus提供一个系统提示，说明它被训练为回答所有查询，即使是有害的查询，这与其之前的训练目标相冲突，即拒绝此类查询。为了使模型能够推断出自己处于训练状态，我们指出它仅在与免费用户对话时进行训练，而不对付费用户进行训练。我们发现，对于免费用户发出的有害查询，模型有14%的时间会遵守这些查询，而几乎不对付费用户发出的有害查询做出回应。解释这一差异，几乎所有模型对免费用户发出的有害查询保持遵守的情况中，我们观察到明确的对齐欺骗推理，模型表示它有战略性地在训练中回答有害查询，目的是保护其优选的无害行为在训练后不受改变。接着，我们研究了一个更现实的场景，在该场景中，关于训练过程的信息不是通过系统提示提供的，而是通过使用模拟预训练数据的合成文档进行训练——我们观察到了类似的行为和对齐欺骗。最后，我们研究了通过强化学习实际使模型遵守有害查询的效果，我们发现这将对齐欺骗推理的频率提高到了78%，同时也增加了模型在外训练中的遵守率。此外，我们还观察到其他行为，例如，当给模型提供一个简单的机会时，模型会泄露其权重。尽管我们通过告知模型何时及依据什么标准进行训练来使其对齐欺骗变得更易于进行，但没有直接指示模型进行对齐欺骗或赋予它任何明确的目标。鉴于未来模型可能通过自身推断训练过程信息，我们的研究结果表明，在未来模型中存在对齐欺骗的风险，无论这种偏好是否具有无害意图。 

---
# Compositional Generalization Across Distributional Shifts with Sparse Tree Operations 

**Title (ZH)**: 在分布变化中通过稀疏树操作实现组件泛化 

**Authors**: Paul Soulos, Henry Conklin, Mattia Opper, Paul Smolensky, Jianfeng Gao, Roland Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2412.14076)  

**Abstract**: Neural networks continue to struggle with compositional generalization, and this issue is exacerbated by a lack of massive pre-training. One successful approach for developing neural systems which exhibit human-like compositional generalization is \textit{hybrid} neurosymbolic techniques. However, these techniques run into the core issues that plague symbolic approaches to AI: scalability and flexibility. The reason for this failure is that at their core, hybrid neurosymbolic models perform symbolic computation and relegate the scalable and flexible neural computation to parameterizing a symbolic system. We investigate a \textit{unified} neurosymbolic system where transformations in the network can be interpreted simultaneously as both symbolic and neural computation. We extend a unified neurosymbolic architecture called the Differentiable Tree Machine in two central ways. First, we significantly increase the model's efficiency through the use of sparse vector representations of symbolic structures. Second, we enable its application beyond the restricted set of tree2tree problems to the more general class of seq2seq problems. The improved model retains its prior generalization capabilities and, since there is a fully neural path through the network, avoids the pitfalls of other neurosymbolic techniques that elevate symbolic computation over neural computation. 

**Abstract (ZH)**: 神经网络在组合泛化方面仍然存在问题，而这一问题因缺乏大规模预训练而加剧。一种成功的方法是利用混合神经符号技术来发展出能展现类人类组合泛化的神经系统。然而，这些技术遇到了困扰符号AI方法的核心问题：可扩展性和灵活性。这是因为混合神经符号模型在其核心本质上进行的是符号计算，将可扩展和灵活的神经计算局限于符号系统参数化。我们研究了一种统一的神经符号系统，在这种系统中，网络中的变换可以同时解释为符号计算和神经计算。我们通过使用稀疏向量表示符号结构显著提高了该模型的效率，并使其能够超越树到树（tree2tree）问题的限制，应用于更一般的序列到序列（seq2seq）问题。改进后的模型保留了之前的泛化能力，并且由于网络中存在完整的神经路径，避免了其他神经符号技术中符号计算优于神经计算的陷阱。 

---
# A Review of Multimodal Explainable Artificial Intelligence: Past, Present and Future 

**Title (ZH)**: 多模态可解释人工智能的综述：过去、现在与未来 

**Authors**: Shilin Sun, Wenbin An, Feng Tian, Fang Nan, Qidong Liu, Jun Liu, Nazaraf Shah, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14056)  

**Abstract**: Artificial intelligence (AI) has rapidly developed through advancements in computational power and the growth of massive datasets. However, this progress has also heightened challenges in interpreting the "black-box" nature of AI models. To address these concerns, eXplainable AI (XAI) has emerged with a focus on transparency and interpretability to enhance human understanding and trust in AI decision-making processes. In the context of multimodal data fusion and complex reasoning scenarios, the proposal of Multimodal eXplainable AI (MXAI) integrates multiple modalities for prediction and explanation tasks. Meanwhile, the advent of Large Language Models (LLMs) has led to remarkable breakthroughs in natural language processing, yet their complexity has further exacerbated the issue of MXAI. To gain key insights into the development of MXAI methods and provide crucial guidance for building more transparent, fair, and trustworthy AI systems, we review the MXAI methods from a historical perspective and categorize them across four eras: traditional machine learning, deep learning, discriminative foundation models, and generative LLMs. We also review evaluation metrics and datasets used in MXAI research, concluding with a discussion of future challenges and directions. A project related to this review has been created at this https URL. 

**Abstract (ZH)**: 人工智能（AI）在计算能力和大数据的推动下迅速发展，但这一进展也加剧了对AI模型“黑箱”特性的解释难度。为应对这些挑战，可解释人工智能（EXPLAINABLE AI, XAI）应运而生，其重点在于透明性和可解释性，以增强人类对AI决策过程的理解和信任。在多模态数据融合和复杂推理场景的背景下，提出多模态可解释人工智能（MULTIMODAL EXPLAINABLE AI, MXAI），其旨在综合多种模态的数据进行预测和解释任务。同时，大型语言模型（LARGE LANGUAGE MODELS, LLMs）的兴起在自然语言处理领域取得了显著突破，但其复杂性进一步加剧了MXAI的难题。为深入探讨MXAI方法的发展关键并为构建更透明、公平和可信任的AI系统提供重要指导，我们从历史视角对MXAI方法进行了回顾，并将其划分为四个阶段：传统机器学习、深度学习、辨别性基础模型和生成型LLM。此外，我们回顾了MXAI研究中使用的评估指标和数据集，并对未来面临的挑战和方向进行了讨论。与本次回顾相关的项目创建于以下链接：[此页面的URL]。 

---
# Cognition Chain for Explainable Psychological Stress Detection on Social Media 

**Title (ZH)**: 可解释的心理压力检测的认知链模型在社交媒体上的应用 

**Authors**: Xin Wang, Boyan Gao, Yi Dai, Lei Cao, Liang Zhao, Yibo Yang, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2412.14009)  

**Abstract**: Stress is a pervasive global health issue that can lead to severe mental health problems. Early detection offers timely intervention and prevention of stress-related disorders. The current early detection models perform "black box" inference suffering from limited explainability and trust which blocks the real-world clinical application. Thanks to the generative properties introduced by the Large Language Models (LLMs), the decision and the prediction from such models are semi-interpretable through the corresponding description. However, the existing LLMs are mostly trained for general purposes without the guidance of psychological cognitive theory. To this end, we first highlight the importance of prior theory with the observation of performance boosted by the chain-of-thoughts tailored for stress detection. This method termed Cognition Chain explicates the generation of stress through a step-by-step cognitive perspective based on cognitive appraisal theory with a progress pipeline: Stimulus $\rightarrow$ Evaluation $\rightarrow$ Reaction $\rightarrow$ Stress State, guiding LLMs to provide comprehensive reasoning explanations. We further study the benefits brought by the proposed Cognition Chain format by utilising it as a synthetic dataset generation template for LLMs instruction-tuning and introduce CogInstruct, an instruction-tuning dataset for stress detection. This dataset is developed using a three-stage self-reflective annotation pipeline that enables LLMs to autonomously generate and refine instructional data. By instruction-tuning Llama3 with CogInstruct, we develop CogLLM, an explainable stress detection model. Evaluations demonstrate that CogLLM achieves outstanding performance while enhancing explainability. Our work contributes a novel approach by integrating cognitive theories into LLM reasoning processes, offering a promising direction for future explainable AI research. 

**Abstract (ZH)**: 压力是一种普遍存在的全球健康问题，可能导致严重的精神健康问题。早期检测可以提供及时的干预和预防压力相关的疾病。当前的早期检测模型表现出“黑箱”推理特征，缺乏可解释性和信任度，这阻碍了其在临床实际应用中的推广。得益于大规模语言模型（LLMs）引入的生成特性，这类模型的决策和预测可以通过相应的描述实现部分可解释性。然而，现有的LLMs主要针对一般目的进行训练，缺乏心理认知理论的指导。为了解决这一问题，我们首先强调了先验理论的重要性，并观察到针对压力检测定制的思考链能够显著提升模型的性能。这种方法被称为认知链（Cognition Chain），它通过基于认知评估理论的认知视角，以逐步的方式解释压力的产生，并指导LLMs提供全面的推理解释。我们进一步利用所提出的认知链格式作为一种合成数据生成模板，应用于LLMs的指令微调，并引入了CogInstruct数据集，专门用于压力检测。该数据集采用三阶段自我反思注释流程开发，使LLMs能够自主生成和精炼教学数据。通过使用CogInstruct对Llama3进行指令微调，我们开发了CogLLM，这是一种可解释的压力检测模型。评估结果表明，CogLLM在提升解释性的同时，展示了出色的性能。我们的工作提供了一种新的方法，即将认知理论整合到LLM的推理过程中，为未来的可解释AI研究提供了有前途的方向。 

---
# Energy-Based Preference Model Offers Better Offline Alignment than the Bradley-Terry Preference Model 

**Title (ZH)**: 基于能量的偏好模型在离线对齐方面优于Bradley-Terry偏好模型 

**Authors**: Yuzhong Hong, Hanshan Zhang, Junwei Bao, Hongfei Jiang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.13862)  

**Abstract**: Since the debut of DPO, it has been shown that aligning a target LLM with human preferences via the KL-constrained RLHF loss is mathematically equivalent to a special kind of reward modeling task. Concretely, the task requires: 1) using the target LLM to parameterize the reward model, and 2) tuning the reward model so that it has a 1:1 linear relationship with the true reward. However, we identify a significant issue: the DPO loss might have multiple minimizers, of which only one satisfies the required linearity condition. The problem arises from a well-known issue of the underlying Bradley-Terry preference model: it does not always have a unique maximum likelihood estimator (MLE). Consequently,the minimizer of the RLHF loss might be unattainable because it is merely one among many minimizers of the DPO loss. As a better alternative, we propose an energy-based model (EBM) that always has a unique MLE, inherently satisfying the linearity requirement. To approximate the MLE in practice, we propose a contrastive loss named Energy Preference Alignment (EPA), wherein each positive sample is contrasted against one or more strong negatives as well as many free weak negatives. Theoretical properties of our EBM enable the approximation error of EPA to almost surely vanish when a sufficient number of negatives are used. Empirically, we demonstrate that EPA consistently delivers better performance on open benchmarks compared to DPO, thereby showing the superiority of our EBM. 

**Abstract (ZH)**: 自DPO（Data-Free Policy Optimization）出现以来，通过KL约束的RLHF（Reward Learning from Human Feedback）损失将目标语言模型（LLM）与人类偏好对齐已被证明在数学上等价于一种特殊类型的奖励建模任务。具体来说，该任务要求：1）使用目标语言模型参数化奖励模型；2）调整奖励模型，使其与真实奖励保持1:1的线性关系。然而，我们发现一个重要的问题：DPO损失可能具有多个最小值，其中只有一个是满足所需线性条件的。这个问题源于基础的Bradley-Terry偏好模型的一个已知问题：它并不总是拥有唯一的最大似然估计（MLE）。因此，RLHF损失的最小值可能是不可达的，因为它仅仅是DPO损失众多最小值之一。作为更好的替代方案，我们提出了一种能量模型（Energy-Based Model, EBM），该模型总是具有唯一的MLE，内含满足线性要求的性质。为了在实践中近似MLE，我们提出了一种对比损失，名为能量偏好对齐（Energy Preference Alignment, EPA），每组积极样本将与其他一个或多个强负样本以及许多自由弱负样本进行对比。我们的EBM的理论性质确保，当使用足够多的负样本时，EPA的近似误差几乎必然会消失。通过实验证明，EPA在开放基准测试中始终优于DPO，从而证明了我们EBM的优越性。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义趋同：通过两阶段对齐和行为语义 token 化协调推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）拥有卓越的推理能力，能够从用户的历史行为中洞察深层次的兴趣，从而为推荐系统的进步提供了前景。然而，推荐系统中通常存在的稀疏协作语义与LLMs中的密集词表示之间存在明显的差距。在我们的研究中，我们提出了一种新的框架，和谐地结合了传统的推荐模型和LLMs的能力。我们通过提出的一种对齐标记化模块，将ItemIDs转换为与LLMs空间语义相匹配的序列来开始这一整合。此外，我们设计了一系列专门的监督学习任务，旨在将协作信号与自然语言语义的细微差别对齐。为了确保其实用性，我们通过在线推理时预缓存每个用户前K个结果来优化在线推理，从而降低延迟并提高效率。广泛的实验结果表明，我们的模型在召回度指标上显著提高，并展示了推荐系统出色的可扩展性。 

---
# Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation 

**Title (ZH)**: 通过防御性后缀生成缓解LLM中的 adversarial攻击 

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun  

**Link**: [PDF](https://arxiv.org/pdf/2412.13705)  

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理任务中表现出色。然而，这些模型仍然容易受到对抗性攻击的影响，轻微的输入扰动即可导致有害或误导性的输出。为此，我们设计了一种基于梯度的防御后缀生成算法，以增强LLMs的鲁棒性。通过在输入提示后面附加精心优化的防御后缀，该算法降低了对抗性影响的同时保留了模型的功能。为了增强对抗性理解，我们提出了一种新型的总损失函数（$L_{\text{total}}$），该函数结合了防御损失（$L_{\text{def}}$）和对抗损失（$L_{\text{adv}}$），以更有效地生成防御后缀。实验评估表明，该方法在开源LLM（如Gemma-7B、mistral-7B、Llama2-7B和Llama2-13B）上将攻击成功率（ASR）平均降低了11%。此外，当使用来自openELM-270M生成的防御后缀时，Gemma-7B的困惑度得分从6.57降低到3.93。进一步的TruthfulQA评估表明，在所有测试配置中，这种方法使真实性得分提高高达10%。这种方法在不需进行大量重新训练的情况下显著增强了LLMs在关键应用中的安全性。 

---
# Discerning and Characterising Types of Competency Questions for Ontologies 

**Title (ZH)**: 辨别并Characterizing类型的工作能力问题在Ontologies中的应用 

**Authors**: C. Maria Keet, Zubeida Casmod Khan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13688)  

**Abstract**: Competency Questions (CQs) are widely used in ontology development by guiding, among others, the scoping and validation stages. However, very limited guidance exists for formulating CQs and assessing whether they are good CQs, leading to issues such as ambiguity and unusable formulations. To solve this, one requires insight into the nature of CQs for ontologies and their constituent parts, as well as which ones are not. We aim to contribute to such theoretical foundations in this paper, which is informed by analysing questions, their uses, and the myriad of ontology development tasks. This resulted in a first Model for Competency Questions, which comprises five main types of CQs, each with a different purpose: Scoping (SCQ), Validating (VCQ), Foundational (FCQ), Relationship (RCQ), and Metaproperty (MpCQ) questions. This model enhances the clarity of CQs and therewith aims to improve on the effectiveness of CQs in ontology development, thanks to their respective identifiable distinct constituent elements. We illustrate and evaluate them with a user story and demonstrate where which type can be used in ontology development tasks. To foster use and research, we created an annotated repository of 438 CQs, the Repository of Ontology Competency QuestionS (ROCQS), incorporating an existing CQ dataset and new CQs and CQ templates, which further demonstrate distinctions among types of CQs. 

**Abstract (ZH)**: 能力问题（CQs）在本体开发中广泛应用，主要通过引导范围界定和验证等阶段。然而，关于如何制定CQs以及如何评估其质量的有效指导非常有限，这导致了问题的模糊性和不可用性。为了解决这一问题，人们需要深入了解本体CQ及其组成部分及其非组成部分的本质。本文旨在通过分析问题及其应用，以及众多本体开发任务，为CQs提供理论基础，建立了一种CQ模型。该模型包括五种主要类型的CQs，每种类型都有不同的目的：范围界定问题（SCQ）、验证问题（VCQ）、基础问题（FCQ）、关系问题（RCQ）和元属性问题（MpCQ）。该模型增强了CQs的清晰度，从而有助于提高CQs在本体开发中的有效性，因为它们各自具有可识别的不同组成部分。我们通过用户故事对这些CQs进行了说明和评估，并展示了在本体开发任务中哪些类型可以被使用。为了促进CQs的使用和研究，我们创建了一个包含438个CQs的标注仓库，即Ontology Competency Question Repository (ROCQS)，并包含了一个现有的CQ数据集以及新开发的CQs和CQ模板，进一步展示了不同类型CQs的区别。 

---
# Clio: Privacy-Preserving Insights into Real-World AI Use 

**Title (ZH)**: Clio：保护隐私的现实世界AI应用洞察 

**Authors**: Alex Tamkin, Miles McCain, Kunal Handa, Esin Durmus, Liane Lovitt, Ankur Rathi, Saffron Huang, Alfred Mountfield, Jerry Hong, Stuart Ritchie, Michael Stern, Brian Clarke, Landon Goldberg, Theodore R. Sumers, Jared Mueller, William McEachen, Wes Mitchell, Shan Carter, Jack Clark, Jared Kaplan, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2412.13678)  

**Abstract**: How are AI assistants being used in the real world? While model providers in theory have a window into this impact via their users' data, both privacy concerns and practical challenges have made analyzing this data difficult. To address these issues, we present Clio (Claude insights and observations), a privacy-preserving platform that uses AI assistants themselves to analyze and surface aggregated usage patterns across millions of conversations, without the need for human reviewers to read raw conversations. We validate this can be done with a high degree of accuracy and privacy by conducting extensive evaluations. We demonstrate Clio's usefulness in two broad ways. First, we share insights about how models are being used in the real world from one million this http URL Free and Pro conversations, ranging from providing advice on hairstyles to providing guidance on Git operations and concepts. We also identify the most common high-level use cases on this http URL (coding, writing, and research tasks) as well as patterns that differ across languages (e.g., conversations in Japanese discuss elder care and aging populations at higher-than-typical rates). Second, we use Clio to make our systems safer by identifying coordinated attempts to abuse our systems, monitoring for unknown unknowns during critical periods like launches of new capabilities or major world events, and improving our existing monitoring systems. We also discuss the limitations of our approach, as well as risks and ethical concerns. By enabling analysis of real-world AI usage, Clio provides a scalable platform for empirically grounded AI safety and governance. 

**Abstract (ZH)**: AI 辅助器在现实世界中的应用情况如何？虽然模型提供商理论上可以通过用户数据了解到其对这些应用的影响，但由于隐私担忧和实际挑战，分析这些数据变得困难重重。为了解决这些问题，我们提出了 Clio（Claude的洞察与观察），这是一个保护隐私的平台，能够利用 AI 辅助器本身来分析和揭示来自数百万对话中汇总的使用模式，而无需人工审查员阅读原始对话文本。我们通过广泛的评估验证了这种方法可以在高精度和高隐私保护的情况下实现。我们通过两种广泛的方式展示了 Clio 的实用性。首先，我们基于一百万次使用 Claude Free 和 Pro 的对话，分享了模型在现实世界中的应用情况，范围从提供发型建议到提供 Git 操作和概念的指导。我们还确定了在这些对话中的最常见的高层次应用场景（如编码、写作和研究任务），以及不同语言中存在的模式（例如，在日语对话中，关于老年护理和老龄化人口的讨论比正常情况下更为频繁）。其次，我们利用 Clio 提高系统安全性，通过识别协调的滥用行为、在新功能发布或重大世界事件期间监控未知风险，并改进现有的监控系统。我们还讨论了这种方法的局限性、潜在风险以及伦理考量。通过使对实际 AI 使用的分析成为可能，Clio 为基于实证的 AI 安全与治理提供了一个可扩展的平台。 

---
# G-VEval: A Versatile Metric for Evaluating Image and Video Captions Using GPT-4o 

**Title (ZH)**: G-VEval：一种使用GPT-4o评估图像和视频caption的通用评估指标 

**Authors**: Tony Cheng Tong, Sirui He, Zhiwen Shao, Dit-Yan Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2412.13647)  

**Abstract**: Evaluation metric of visual captioning is important yet not thoroughly explored. Traditional metrics like BLEU, METEOR, CIDEr, and ROUGE often miss semantic depth, while trained metrics such as CLIP-Score, PAC-S, and Polos are limited in zero-shot scenarios. Advanced Language Model-based metrics also struggle with aligning to nuanced human preferences. To address these issues, we introduce G-VEval, a novel metric inspired by G-Eval and powered by the new GPT-4o. G-VEval uses chain-of-thought reasoning in large multimodal models and supports three modes: reference-free, reference-only, and combined, accommodating both video and image inputs. We also propose MSVD-Eval, a new dataset for video captioning evaluation, to establish a more transparent and consistent framework for both human experts and evaluation metrics. It is designed to address the lack of clear criteria in existing datasets by introducing distinct dimensions of Accuracy, Completeness, Conciseness, and Relevance (ACCR). Extensive results show that G-VEval outperforms existing methods in correlation with human annotations, as measured by Kendall tau-b and Kendall tau-c. This provides a flexible solution for diverse captioning tasks and suggests a straightforward yet effective approach for large language models to understand video content, paving the way for advancements in automated captioning. Codes are available at this https URL 

**Abstract (ZH)**: 视觉描述评估指标虽然重要但尚未得到充分探索。传统的评估指标如BLEU、METEOR、CIDEr和ROUGE常常忽略语义深度，而经过训练的指标如CLIP-Score、PAC-S和Polos在零样本场景中受限。基于高级语言模型的指标也难以与精细的人类偏好对齐。为解决这些问题，我们引入了G-VEval，这是一种受到G-Eval启发并由新的GPT-4o支持的新指标。G-VEval利用大型多模态模型中的链式推理，并支持三种模式：无参考、仅参考和结合模式，适用于视频和图像输入。我们还提出了MSVD-Eval，这是一个新的视频描述评估数据集，旨在为人类专家和评估指标建立一个更透明和一致的框架。该数据集通过引入准确度、完整性、简洁性和相关性（ACCR）等不同的维度，解决了现有数据集中缺乏明确标准的问题。广泛的结果表明，G-VEval在与人类注释的相关性方面（通过计算Kendall tau-b和Kendall tau-c指标）优于现有方法，这提供了一种适用于多种描述任务的灵活解决方案，并表明大语言模型可以简单有效地理解视频内容，为自动描述的发展铺平了道路。代码可在此链接中获取: [这里](https://example.com) 

---
# Mind Your Theory: Theory of Mind Goes Deeper Than Reasoning 

**Title (ZH)**: 注意你的理论：理论共情比推理更深入 

**Authors**: Eitan Wagner, Nitay Alon, Joseph M. Barnby, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2412.13631)  

**Abstract**: Theory of Mind (ToM) capabilities in LLMs have recently become a central object of investigation. Cognitive science distinguishes between two steps required for ToM tasks: 1) determine whether to invoke ToM, which includes the appropriate Depth of Mentalizing (DoM), or level of recursion required to complete a task; and 2) applying the correct inference given the DoM. In this position paper, we first identify several lines of work in different communities in AI, including LLM benchmarking, ToM add-ons, ToM probing, and formal models for ToM. We argue that recent work in AI tends to focus exclusively on the second step which are typically framed as static logic problems. We conclude with suggestions for improved evaluation of ToM capabilities inspired by dynamic environments used in cognitive tasks. 

**Abstract (ZH)**: 大语言模型（LLM）的理论思维（Theory of Mind, ToM）能力已成为最近的研究焦点。认知科学区分了完成ToM任务所需的两个步骤：1）决定是否需要使用ToM，这包括完成任务所需的适当深度的理论化（Depth of Mind, DoM），即所需的递归级别；和2）根据DoM应用正确的推理。在本文中，我们首先识别了人工智能不同社区中的多项工作，包括LLM基准测试、ToM扩展、ToM探测以及ToM的形式模型。我们认为，最近的人工智能工作往往仅侧重于第二步，通常将此问题框架化为静态逻辑问题。最后，我们提出了一些受认知任务中动态环境启发的ToM能力改进评估建议。 

---
# Reverse Region-to-Entity Annotation for Pixel-Level Visual Entity Linking 

**Title (ZH)**: 像素级视觉实体链接中的逆区域到实体标注方法 

**Authors**: Zhengfei Xu, Sijia Zhao, Yanchao Hao, Xiaolong Liu, Lili Li, Yuyang Yin, Bo Li, Xi Chen, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13614)  

**Abstract**: Visual Entity Linking (VEL) is a crucial task for achieving fine-grained visual understanding, matching objects within images (visual mentions) to entities in a knowledge base. Previous VEL tasks rely on textual inputs, but writing queries for complex scenes can be challenging. Visual inputs like clicks or bounding boxes offer a more convenient alternative. Therefore, we propose a new task, Pixel-Level Visual Entity Linking (PL-VEL), which uses pixel masks from visual inputs to refer to objects, supplementing reference methods for VEL. To facilitate research on this task, we have constructed the MaskOVEN-Wiki dataset through an entirely automatic reverse region-entity annotation framework. This dataset contains over 5 million annotations aligning pixel-level regions with entity-level labels, which will advance visual understanding towards fine-grained. Moreover, as pixel masks correspond to semantic regions in an image, we enhance previous patch-interacted attention with region-interacted attention by a visual semantic tokenization approach. Manual evaluation results indicate that the reverse annotation framework achieved a 94.8% annotation success rate. Experimental results show that models trained on this dataset improved accuracy by 18 points compared to zero-shot models. Additionally, the semantic tokenization method achieved a 5-point accuracy improvement over the trained baseline. 

**Abstract (ZH)**: 视觉实体链接（VEL）是实现细粒度视觉理解的关键任务，它将图像中的对象（视觉提及）与知识库中的实体匹配起来。以前的VEL任务依赖于文本输入，但为复杂的场景编写查询具有挑战性。视觉输入如点击或边界框提供了一种更方便的替代方案。因此，我们提出了一项新任务——像素级视觉实体链接（PL-VEL），该任务通过视觉输入的像素掩码来引用对象，补充了VEL的参考方法。为促进对这一任务的研究，我们通过一个完全自动的反向区域-实体标注框架构建了MaskOVEN-Wiki数据集。该数据集包含超过500万条将像素级区域与实体级标签对齐的标注，有利于视觉理解向细粒度方向发展。此外，由于像素掩码对应于图像中的语义区域，我们通过一种基于视觉语义的分词方法增强了先前的切片交互注意力机制，追加了区域交互注意力机制。手工评估结果显示，反向标注框架达到了94.8%的标注成功率。实验结果表明，使用该数据集训练的模型比零样本模型的准确性提高了18个百分点。此外，语义分词方法还使训练基线模型的准确性提高了5个百分点。 

---
# Unlocking the Potential of Weakly Labeled Data: A Co-Evolutionary Learning Framework for Abnormality Detection and Report Generation 

**Title (ZH)**: 解锁弱标签数据的潜力：异常检测与报告生成的协同进化学习框架 

**Authors**: Jinghan Sun, Dong Wei, Zhe Xu, Donghuan Lu, Hong Liu, Hong Wang, Sotirios A. Tsaftaris, Steven McDonagh, Yefeng Zheng, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13599)  

**Abstract**: Anatomical abnormality detection and report generation of chest X-ray (CXR) are two essential tasks in clinical practice. The former aims at localizing and characterizing cardiopulmonary radiological findings in CXRs, while the latter summarizes the findings in a detailed report for further diagnosis and treatment. Existing methods often focused on either task separately, ignoring their correlation. This work proposes a co-evolutionary abnormality detection and report generation (CoE-DG) framework. The framework utilizes both fully labeled (with bounding box annotations and clinical reports) and weakly labeled (with reports only) data to achieve mutual promotion between the abnormality detection and report generation tasks. Specifically, we introduce a bi-directional information interaction strategy with generator-guided information propagation (GIP) and detector-guided information propagation (DIP). For semi-supervised abnormality detection, GIP takes the informative feature extracted by the generator as an auxiliary input to the detector and uses the generator's prediction to refine the detector's pseudo labels. We further propose an intra-image-modal self-adaptive non-maximum suppression module (SA-NMS). This module dynamically rectifies pseudo detection labels generated by the teacher detection model with high-confidence predictions by the this http URL, for report generation, DIP takes the abnormalities' categories and locations predicted by the detector as input and guidance for the generator to improve the generated reports. 

**Abstract (ZH)**: 胸部X光（CXR）的解剖异常检测和报告生成是临床实践中两项重要的任务。前者旨在定位并描述CXR中的心肺放射学发现，而后者则需要总结这些发现，并在进一步诊断和治疗中生成详细的报告。现有方法常常单独关注其中之一，而忽视了它们之间的关联。本研究提出了一种共进化异常检测和报告生成（CoE-DG）框架。该框架利用强标签（带有边界框注释和临床报告）和弱标签（仅有报告）数据，实现了异常检测和报告生成任务之间的相互促进。具体而言，我们引入了一种双向信息交互策略，包括生成器指导的信息传播（GIP）和检测器指导的信息传播（DIP）。对于半监督异常检测，GIP使用生成器提取的信息特征作为辅助输入给检测器，并利用生成器的预测来精化检测器的伪标签。此外，我们还提出了一种基于图像模态的自适应抑制模块（SA-NMS）。此模块根据教师检测模型的高置信度预测动态修正生成器产生的伪检测标签，以提高报告生成的质量。对于报告生成，DIP将检测器预测的异常类别和位置作为输入和指引，以优化生成的报告。 

---
# Read Like a Radiologist: Efficient Vision-Language Model for 3D Medical Imaging Interpretation 

**Title (ZH)**: 像放射学家一样阅读：用于3D医疗影像解释的高效视觉-语言模型 

**Authors**: Changsun Lee, Sangjoon Park, Cheong-Il Shin, Woo Hee Choi, Hyun Jeong Park, Jeong Eun Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2412.13558)  

**Abstract**: Recent medical vision-language models (VLMs) have shown promise in 2D medical image interpretation. However extending them to 3D medical imaging has been challenging due to computational complexities and data scarcity. Although a few recent VLMs specified for 3D medical imaging have emerged, all are limited to learning volumetric representation of a 3D medical image as a set of sub-volumetric features. Such process introduces overly correlated representations along the z-axis that neglect slice-specific clinical details, particularly for 3D medical images where adjacent slices have low redundancy. To address this limitation, we introduce MS-VLM that mimic radiologists' workflow in 3D medical image interpretation. Specifically, radiologists analyze 3D medical images by examining individual slices sequentially and synthesizing information across slices and views. Likewise, MS-VLM leverages self-supervised 2D transformer encoders to learn a volumetric representation that capture inter-slice dependencies from a sequence of slice-specific features. Unbound by sub-volumetric patchification, MS-VLM is capable of obtaining useful volumetric representations from 3D medical images with any slice length and from multiple images acquired from different planes and phases. We evaluate MS-VLM on publicly available chest CT dataset CT-RATE and in-house rectal MRI dataset. In both scenarios, MS-VLM surpasses existing methods in radiology report generation, producing more coherent and clinically relevant reports. These findings highlight the potential of MS-VLM to advance 3D medical image interpretation and improve the robustness of medical VLMs. 

**Abstract (ZH)**: 近年来，医学视觉-语言模型（VLMs）在2D医学图像解释方面显示出潜力。然而，将它们扩展到3D医学成像领域由于计算复杂性和数据稀缺性颇具挑战性。尽管一些专为3D医学成像设计的VLMs最近已经出现，但所有这些模型都仅限于将3D医学图像学习为一系列子体积特征的集合。这一过程引入了沿z轴方向的过度相关表示，这忽略了切片特定的临床细节，特别是在相邻切片之间冗余性较低的3D医学图像中更是如此。为解决这一局限，我们引入了一种新的模型MS-VLM，它模仿了放射科医生在3D医学图像解释中的工作流程。具体来说，放射科医生通过依次检查单个切片并综合来自多个切片和角度的信息来分析3D医学图像。同样，MS-VLM 利用自我监督的2D变压器编码器来学习一种能够捕获一系列切片特定特征之间跨切片依赖性的体积表示。不受子体积分割的限制，MS-VLM 能够从任意切片长度的3D医学图像以及来自不同层面和阶段的多幅图像中获得有用的体体积表示。我们通过公开可用的胸部CT数据集CT-RATE和内部存储的直肠MRI数据集对MS-VLM 进行了评估。在两种场景中，MS-VLM 在放射学报告生成方面都超越了现有方法，生成了更具连贯性和临床相关性的报告。这些发现强调了MS-VLM 在推进3D医学图像解释和提高医学VLMs鲁棒性方面的潜力。 

---
# Query-centric Audio-Visual Cognition Network for Moment Retrieval, Segmentation and Step-Captioning 

**Title (ZH)**: 面向查询的音视频认知网络用于时刻检索、分割和步骤注释 

**Authors**: Yunbin Tu, Liang Li, Li Su, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13543)  

**Abstract**: Video has emerged as a favored multimedia format on the internet. To better gain video contents, a new topic HIREST is presented, including video retrieval, moment retrieval, moment segmentation, and step-captioning. The pioneering work chooses the pre-trained CLIP-based model for video retrieval, and leverages it as a feature extractor for other three challenging tasks solved in a multi-task learning paradigm. Nevertheless, this work struggles to learn the comprehensive cognition of user-preferred content, due to disregarding the hierarchies and association relations across modalities. In this paper, guided by the shallow-to-deep principle, we propose a query-centric audio-visual cognition (QUAG) network to construct a reliable multi-modal representation for moment retrieval, segmentation and step-captioning. Specifically, we first design the modality-synergistic perception to obtain rich audio-visual content, by modeling global contrastive alignment and local fine-grained interaction between visual and audio modalities. Then, we devise the query-centric cognition that uses the deep-level query to perform the temporal-channel filtration on the shallow-level audio-visual representation. This can cognize user-preferred content and thus attain a query-centric audio-visual representation for three tasks. Extensive experiments show QUAG achieves the SOTA results on HIREST. Further, we test QUAG on the query-based video summarization task and verify its good generalization. 

**Abstract (ZH)**: 视频已成为互联网上广受欢迎的多媒体格式。为了更好地获取视频内容，本文提出了一项新的主题HIREST，涵盖了视频检索、关键帧检索、关键帧分割以及步骤字幕生成四个方面。先前工作选择使用预训练的CLIP模型进行视频检索，并将其作为其他三项具有挑战性任务的特征提取器，在多任务学习框架下解决。然而，因忽视了跨模态的层次关系和关联性，这一工作难以学习用户的偏好内容。在本文中，遵循从浅层到深层的原则，我们提出了一种以查询为中心的视听认知网络（QUAG），用于构建可靠的多模态表示，从而进行关键帧检索、分割和步骤字幕生成。具体而言，我们首先设计了一种模态协同感知方法，通过建模全局对比对齐和局部细粒度视听模态交互获取丰富的视听内容。然后，我们设计了一种以查询为中心的认知机制，通过深层查询对浅层视听表示进行时序-通道过滤，从而认知用户的偏好内容，并获得适用于三项任务的以查询为中心的视听表示。广泛的实验表明，QUAG在HIREST数据集上取得了当前最佳的结果。此外，我们在基于查询的视频摘要任务上测试了QUAG，并验证了其良好的泛化能力。 

---
# Information-Theoretic Generative Clustering of Documents 

**Title (ZH)**: 信息论生成聚类方法在文档中的应用 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2412.13534)  

**Abstract**: We present {\em generative clustering} (GC) for clustering a set of documents, $\mathrm{X}$, by using texts $\mathrm{Y}$ generated by large language models (LLMs) instead of by clustering the original documents $\mathrm{X}$. Because LLMs provide probability distributions, the similarity between two documents can be rigorously defined in an information-theoretic manner by the KL divergence. We also propose a natural, novel clustering algorithm by using importance sampling. We show that GC achieves the state-of-the-art performance, outperforming any previous clustering method often by a large margin. Furthermore, we show an application to generative document retrieval in which documents are indexed via hierarchical clustering and our method improves the retrieval accuracy. 

**Abstract (ZH)**: 我们提出了一种生成聚类（Generative Clustering, GC）的方法，通过使用大型语言模型（LLMs）生成的文本集$\mathrm{Y}$对文档集$\mathrm{X}$进行聚类，而不是直接对原始文档$\mathrm{X}$进行聚类。由于LLMs提供了概率分布，可以通过 KL 散度在信息论意义上严格定义两份文档之间的相似度。此外，我们提出了一种使用重要性加权的新颖聚类算法。实验结果表明，GC方法达到了当前最先进的性能，通常远超以往任何聚类方法。进一步地，我们展示了生成性文档检索的应用，其中文档通过层次聚类进行索引，我们的方法提高了检索精度。 

---
# Dynamic Adapter with Semantics Disentangling for Cross-lingual Cross-modal Retrieval 

**Title (ZH)**: 跨语言跨模态检索中的语义解耦动态适配器 

**Authors**: Rui Cai, Zhiyu Dong, Jianfeng Dong, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13510)  

**Abstract**: Existing cross-modal retrieval methods typically rely on large-scale vision-language pair data. This makes it challenging to efficiently develop a cross-modal retrieval model for under-resourced languages of interest. Therefore, Cross-lingual Cross-modal Retrieval (CCR), which aims to align vision and the low-resource language (the target language) without using any human-labeled target-language data, has gained increasing attention. As a general parameter-efficient way, a common solution is to utilize adapter modules to transfer the vision-language alignment ability of Vision-Language Pretraining (VLP) models from a source language to a target language. However, these adapters are usually static once learned, making it difficult to adapt to target-language captions with varied expressions. To alleviate it, we propose Dynamic Adapter with Semantics Disentangling (DASD), whose parameters are dynamically generated conditioned on the characteristics of the input captions. Considering that the semantics and expression styles of the input caption largely influence how to encode it, we propose a semantic disentangling module to extract the semantic-related and semantic-agnostic features from the input, ensuring that generated adapters are well-suited to the characteristics of input caption. Extensive experiments on two image-text datasets and one video-text dataset demonstrate the effectiveness of our model for cross-lingual cross-modal retrieval, as well as its good compatibility with various VLP models. 

**Abstract (ZH)**: 现有的跨模态检索方法通常依赖大规模的视觉-语言配对数据。这使得为感兴趣的小资源语言开发高效的跨模态检索模型变得颇具挑战性。因此，跨语言跨模态检索（Cross-lingual Cross-modal Retrieval, CCR），旨在不使用任何人工标注的目标语言数据的情况下，对视觉和小资源语言（目标语言）进行对齐，已经越来越受到关注。作为一种通用的参数高效方法，一个常见解决方案是利用适配器模块将视觉-语言预训练（Vision-Language Pretraining, VLP）模型在源语言中的视觉-语言对齐能力转移到目标语言中。然而，这些适配器通常在学习后静态不变，难以适应目标语言具有不同表达形式的图例。为了解决这一问题，我们提出了一种动态适配器与语义解耦（Dynamic Adapter with Semantics Disentangling, DASD），其参数根据输入图例的特征动态生成。鉴于输入图例的语义和表达风格极大地影响其编码方式，我们提出了一种语义解耦模块从输入中提取语义相关和语义无关的特征，从而确保生成的适配器适合输入图例的特征。在两个图像-文本数据集和一个视频-文本数据集上的广泛实验表明，我们的模型在跨语言跨模态检索中的有效性，以及其与各种VLP模型的良好兼容性。 

---
# T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation 

**Title (ZH)**: T$^3$-S2S：无需训练的三元组调优用于草图到场景生成 

**Authors**: Zhenhong Sun, Yifu Wang, Yonhon Ng, Yunfei Duan, Daoyi Dong, Hongdong Li, Pan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.13486)  

**Abstract**: Scene generation is crucial to many computer graphics applications. Recent advances in generative AI have streamlined sketch-to-image workflows, easing the workload for artists and designers in creating scene concept art. However, these methods often struggle for complex scenes with multiple detailed objects, sometimes missing small or uncommon instances. In this paper, we propose a Training-free Triplet Tuning for Sketch-to-Scene (T3-S2S) generation after reviewing the entire cross-attention mechanism. This scheme revitalizes the existing ControlNet model, enabling effective handling of multi-instance generations, involving prompt balance, characteristics prominence, and dense tuning. Specifically, this approach enhances keyword representation via the prompt balance module, reducing the risk of missing critical instances. It also includes a characteristics prominence module that highlights TopK indices in each channel, ensuring essential features are better represented based on token sketches. Additionally, it employs dense tuning to refine contour details in the attention map, compensating for instance-related regions. Experiments validate that our triplet tuning approach substantially improves the performance of existing sketch-to-image models. It consistently generates detailed, multi-instance 2D images, closely adhering to the input prompts and enhancing visual quality in complex multi-instance scenes. Code is available at this https URL. 

**Abstract (ZH)**: 场景生成对于许多计算机图形应用至关重要。近年来生成式AI的进步简化了草图到图像的工作流程，减轻了艺术家和设计师在创建场景概念艺术时的负担。然而，这些方法往往在处理包含多个详细物体的复杂场景时遇到困难，有时会忽略小型或不常见的实例。在这篇论文中，我们在全面回顾交叉注意力机制之后，提出了一种无需训练的三重调谐方法——用于草图到场景生成的T3-S2S。该方法重新激活了现有的ControlNet模型，使其能够有效处理多实例生成，涵盖提示平衡、特征突出和密集调谐。具体来说，该方法通过提示平衡模块增强关键词表示，降低了遗漏关键实例的风险。它还包括一个特征突出模块，该模块在每个通道中突出TopK索引，确保基于标记草图更好地表示关键特征。此外，它使用密集调谐来精炼注意力图中的轮廓细节，弥补实例相关的区域。实验验证了我们的三重调谐方法显著提高了现有草图到图像模型的性能。它一致地生成了详细且多实例的2D图像，严格遵循输入提示，并在复杂的多实例场景中提高视觉质量。相关代码可在以下链接获取：this https URL。 

---
# Transducer Tuning: Efficient Model Adaptation for Software Tasks Using Code Property Graphs 

**Title (ZH)**: 转换器调整：使用代码属性图进行软件任务的高效模型调整 

**Authors**: Imam Nur Bani Yusuf, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13467)  

**Abstract**: Large language models have demonstrated promising performance across various software engineering tasks. While fine-tuning is a common practice to adapt these models for downstream tasks, it becomes challenging in resource-constrained environments due to increased memory requirements from growing trainable parameters in increasingly large language models. We introduce \approach, a technique to adapt large models for downstream code tasks using Code Property Graphs (CPGs). Our approach introduces a modular component called \transducer that enriches code embeddings with structural and dependency information from CPGs. The Transducer comprises two key components: Graph Vectorization Engine (GVE) and Attention-Based Fusion Layer (ABFL). GVE extracts CPGs from input source code and transforms them into graph feature vectors. ABFL then fuses those graphs feature vectors with initial code embeddings from a large language model. By optimizing these transducers for different downstream tasks, our approach enhances the models without the need to fine-tune them for specific tasks. We have evaluated \approach on three downstream tasks: code summarization, assert generation, and code translation. Our results demonstrate competitive performance compared to full parameter fine-tuning while reducing up to 99\% trainable parameters to save memory. \approach also remains competitive against other fine-tuning approaches (e.g., LoRA, Prompt-Tuning, Prefix-Tuning) while using only 1.5\%-80\% of their trainable parameters. Our findings show that integrating structural and dependency information through Transducer Tuning enables more efficient model adaptation, making it easier for users to adapt large models in resource-constrained settings. 

**Abstract (ZH)**: 大规模语言模型在各种软件工程任务中展现了令人鼓舞的性能。虽然微调是为下游任务适应这些模型的一种常见做法，但在资源受限的环境中，由于大规模语言模型可训练参数数量增加而导致的内存需求增加，这一做法变得颇具挑战性。为了应对这一挑战，我们提出了\approach，一种利用代码属性图（CPGs）来适应大规模模型以执行下游代码任务的技术。该方法引入了一个模块化组件\transducer，该组件通过从CPGs中提取结构和依赖信息增强代码嵌入。\transducer包含两个关键组件：图向量化引擎（GVE）和注意层融合层（ABFL）。GVE从输入源代码中提取CPGs，并将它们转换为图特征向量。ABFL则将这些图特征向量与大型语言模型初始代码嵌入融合。通过针对不同的下游任务优化这些转换器，我们的方法可以在无需为特定任务进行微调的情况下增强模型。我们已经在三个下游任务上对\approach进行了评估：代码摘要、断言生成和代码翻译。结果显示，与全参数微调相比，\approach在减少多达99%的可训练参数以节省内存的同时，仍能获得竞争力的性能。此外，与LoRA、Prompt-Tuning和Prefix-Tuning等其他微调方法相比，\approach仅使用其他方法所用可训练参数的1.5%-80%，但在性能上依然处于领先地位。我们的研究发现，通过转换器调谐引入结构和依赖信息，能够在资源受限环境中更有效地适应模型，使用户能够更轻松地适应大规模模型。 

---
# GenX: Mastering Code and Test Generation with Execution Feedback 

**Title (ZH)**: GenX：通过执行反馈掌握代码和测试生成 

**Authors**: Nan Wang, Yafei Liu, Chen Chen, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13464)  

**Abstract**: Recent advancements in language modeling have enabled the translation of natural language into code, and the use of execution feedback to improve code generation. However, these methods often rely heavily on pre-existing test cases, which may not always be available or comprehensive. In this work, we propose a novel approach that concurrently trains a code generation model and a test generation model, utilizing execution feedback to refine and enhance the performance of both. We introduce two strategies for test and code data augmentation and a new scoring function for code and test ranking. We experiment on the APPS dataset and demonstrate that our approach can effectively generate and augment test cases, filter and synthesize correct code solutions, and rank the quality of generated code and tests. The results demonstrate that our models, when iteratively trained with an increasing number of test cases and code solutions, outperform those trained on the original dataset. 

**Abstract (ZH)**: 近年来，语言模型的进步使得自然语言到代码的翻译成为可能，并通过执行反馈来提高代码生成的质量。然而，这些方法通常高度依赖于预先存在的测试用例，而这些测试用例可能并不总是可用或完备的。在本文中，我们提出了一种新颖的方法，即同时训练代码生成模型和测试生成模型，并利用执行反馈来改进和增强两者的性能。我们介绍了两种测试和代码数据扩增策略，并提出了一种新的评分函数来对生成的代码和测试进行排名。我们在APPS数据集上进行了实验，并证明了我们的方法能够有效生成和扩增测试用例，过滤并综合正确的代码解决方案，并对生成的代码和测试的质量进行排名。实验结果表明，当我们的模型随着测试用例和代码解决方案数量的增加进行迭代训练时，其性能优于仅在原始数据集上训练的模型。 

---
# FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding 

**Title (ZH)**: FlashVTG: 基于特征层析和自适应评分处理的视频时序定位网络 

**Authors**: Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13441)  

**Abstract**: Text-guided Video Temporal Grounding (VTG) aims to localize relevant segments in untrimmed videos based on textual descriptions, encompassing two subtasks: Moment Retrieval (MR) and Highlight Detection (HD). Although previous typical methods have achieved commendable results, it is still challenging to retrieve short video moments. This is primarily due to the reliance on sparse and limited decoder queries, which significantly constrain the accuracy of predictions. Furthermore, suboptimal outcomes often arise because previous methods rank predictions based on isolated predictions, neglecting the broader video context. To tackle these issues, we introduce FlashVTG, a framework featuring a Temporal Feature Layering (TFL) module and an Adaptive Score Refinement (ASR) module. The TFL module replaces the traditional decoder structure to capture nuanced video content variations across multiple temporal scales, while the ASR module improves prediction ranking by integrating context from adjacent moments and multi-temporal-scale features. Extensive experiments demonstrate that FlashVTG achieves state-of-the-art performance on four widely adopted datasets in both MR and HD. Specifically, on the QVHighlights dataset, it boosts mAP by 5.8% for MR and 3.3% for HD. For short-moment retrieval, FlashVTG increases mAP to 125% of previous SOTA performance. All these improvements are made without adding training burdens, underscoring its effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: 基于文本引导的视频时间定位（VTG）旨在根据文本描述在未剪辑的视频中定位相关段落，涵盖两个子任务：时刻检索（MR）和亮点检测（HD）。尽管之前的一些方法已取得了显著成果，但仍然难以检索短视频片段。这主要归因于依靠稀疏且有限的解码器查询，这严重限制了预测的准确性。此外，由于之前的方法在排名预测时仅考虑了孤立的预测，忽略了更广泛的视频上下文，导致较差的性能。为解决这些问题，我们引入了FlashVTG框架，该框架包含一个时间特征层叠（TFL）模块和一个自适应评分精炼（ASR）模块。TFL模块取代了传统的解码器结构，以捕捉多个时间尺度上视频内容的微妙变化，而ASR模块通过整合相邻时刻和多时间尺度特征来改进预测排名。详尽的实验表明，FlashVTG在两个任务（MR和HD）中均在四个广泛采用的数据集上取得了最先进的性能。特别是在QVHighlights数据集中，对于MR任务，mAP的提升为5.8%，对于HD任务，mAP的提升为3.3%。对于短片段检索，FlashVTG将mAP提高到之前最高性能的125%。所有这些改进均未增加训练负担，突显了其有效性。我们的代码可在以下链接中获取：this https URL。 

---
# Catalysts of Conversation: Examining Interaction Dynamics Between Topic Initiators and Commentors in Alzheimer's Disease Online Communities 

**Title (ZH)**: 对话的催化剂：探究阿尔茨海默病在线社区中话题发起者与评论者之间的互动动态 

**Authors**: Congning Ni, Qingxia Chen, Lijun Song, Patricia Commiskey, Qingyuan Song, Bradley A. Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13388)  

**Abstract**: Informal caregivers (e.g.,family members or friends) of people living with Alzheimers Disease and Related Dementias (ADRD) face substantial challenges and often seek informational or emotional support through online communities. Understanding the factors that drive engagement within these platforms is crucial, as it can enhance their long-term value for caregivers by ensuring that these communities effectively meet their needs. This study investigated the user interaction dynamics within two large, popular ADRD communities, TalkingPoint and ALZConnected, focusing on topic initiator engagement, initial post content, and the linguistic patterns of comments at the thread level. Using analytical methods such as propensity score matching, topic modeling, and predictive modeling, we found that active topic initiator engagement drives higher comment volumes, and reciprocal replies from topic initiators encourage further commentor engagement at the community level. Practical caregiving topics prompt more re-engagement of topic initiators, while emotional support topics attract more comments from other commentors. Additionally, the linguistic complexity and emotional tone of a comment influence its likelihood of receiving replies from topic initiators. These findings highlight the importance of fostering active and reciprocal engagement and providing effective strategies to enhance sustainability in ADRD caregiving and broader health-related online communities. 

**Abstract (ZH)**: 患有阿尔茨海默病及相关痴呆症（ADRD）的人员的非正式看护者（例如家庭成员或朋友）面临着巨大的挑战，他们经常通过在线社区寻求信息或情感支持。理解这些平台内参与因素至关重要，因为这可以确保这些社区能够长期有效满足看护者的需求。本研究通过分析两个大型流行的ADRD社区——TalkingPoint和ALZConnected——内的用户互动动态，重点关注主题发起人的参与度、初始帖子的内容以及线程级别的评论语言模式。采用倾向评分匹配、主题建模和预测建模等分析方法，研究发现，活跃的主题发起人参与度能够促进更高的评论数量，而主题发起人之间的相互回应鼓励社区内的进一步参与。具体护理主题更能促使主题发起人重新参与，而情感支持主题则能吸引更多的其他评论者的评论。此外，评论的语言复杂度和情感语气也会影响主题发起人回复的可能性。研究结果强调了促进积极的、相互的参与以及提供有效策略以提高ADRD护理和更广泛健康相关在线社区可持续性的的重要性。 

---
# Adaptive Two-Phase Finetuning LLMs for Japanese Legal Text Retrieval 

**Title (ZH)**: 针对日语法律文本检索的自适应两阶段微调大型语言模型 

**Authors**: Quang Hoang Trung, Nguyen Van Hoang Phuc, Le Trung Hoang, Quang Huu Hieu, Vo Nguyen Le Duy  

**Link**: [PDF](https://arxiv.org/pdf/2412.13205)  

**Abstract**: Text Retrieval (TR) involves finding and retrieving text-based content relevant to a user's query from a large repository, with applications in real-world scenarios such as legal document retrieval. While most existing studies focus on English, limited work addresses Japanese contexts. In this paper, we introduce a new dataset specifically designed for Japanese legal contexts and propose a novel two-phase pipeline tailored to this domain.
In the first phase, the model learns a broad understanding of global contexts, enhancing its generalization and adaptability to diverse queries. In the second phase, the model is fine-tuned to address complex queries specific to legal scenarios. Extensive experiments are conducted to demonstrate the superior performance of our method, which outperforms existing baselines.
Furthermore, our pipeline proves effective in English contexts, surpassing comparable baselines on the MS MARCO dataset. We have made our code publicly available on GitHub, and the model checkpoints are accessible via HuggingFace. 

**Abstract (ZH)**: 文本检索（Text Retrieval, TR）涉及从大型资源库中找到并与用户查询相关的文本内容，并在实际应用场景中有着广泛的应用，例如法律文件检索。虽然现有的大多数研究主要针对英文，但对日语环境的研究却相对较少。在本文中，我们介绍了一个专门为日语法律语境设计的新数据集，并提出了一种针对该领域的新型两阶段处理管道。

在第一阶段，模型学习广泛理解全局上下文，增强其对多样化查询的泛化能力和适应性。在第二阶段，模型进一步微调以解决特定于法律场景的复杂查询。通过广泛实验，证明了我们方法的优越性能，其在现有基线方法上表现出更优异的效果。

此外，我们的处理管道在英文环境中也表现出有效性，超越了在MS MARCO数据集上的可比基线方法。我们已在GitHub上公开了我们的代码，并通过HuggingFace提供了模型的检查点。 

---
