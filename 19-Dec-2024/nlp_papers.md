# TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks 

**Title (ZH)**: 《代理公司：评估大规模语言模型代理在具有重大影响的实际任务中的表现》

这个标题翻译旨在保持原意的同时，使其符合中文的学术表达习惯。其中，“Agent Company”被解释为“代理公司”，“LLM Agents”被翻译为“大规模语言模型代理”，“Benchmarking”翻译为“评估”，“Consequential Real World Tasks”翻译为“具有重大影响的实际任务”。这样的翻译既准确传达了原始标题的意思，又符合学术论文标题的规范。 

**Authors**: Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2412.14161)  

**Abstract**: We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at helping to accelerate or even autonomously perform work-related tasks? The answer to this question has important implications for both industry looking to adopt AI into their workflows, and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper, we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that with the most competitive agent, 24% of the tasks can be completed autonomously. This paints a nuanced picture on task automation with LM agents -- in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. 

**Abstract (ZH)**: 我们每天都在与计算机进行互动，无论是在日常生活还是工作中，许多工作都可通过计算机和互联网的访问来完成。与此同时，得益于大型语言模型（LLMs）的改进，能够与环境交互并产生影响的AI代理也迅速发展。那么，这些AI代理在辅助加速或甚至自主执行与工作相关任务方面的表现如何？这个问题的答案对希望将AI整合到工作流程中的行业以及需要了解AI采用可能对劳动力市场产生影响的经济政策而言具有重要意义。

为了衡量这些LLM代理执行实际专业任务的能力，本文引入了TheAgentCompany，一个用于评估数据驱动型工作者以类似方式与世界交互的AI代理的扩展基准。我们构建了一个自包含的环境，包含模拟小型软件公司环境的内部网站和数据，并创建了一系列可供该公司员工执行的任务。我们测试了基于封闭API和开放权重语言模型（LMs）的基线代理，并发现使用最具竞争力的代理时，24%的任务可以实现自主完成。这表明了使用LM代理执行任务自动化的一个复杂图景——在模拟真实工作场所的环境中，很多简单的任务可以实现自主解决，但更具挑战性的长期任务目前仍在当前系统的能力范围之外。 

---
# GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking 

**Title (ZH)**: GLIDER：使用可解释排名评估大规模语言模型交互和决策 

**Authors**: Darshan Deshpande, Selvan Sunitha Ravi, Sky CH-Wang, Bartosz Mielczarek, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2412.14140)  

**Abstract**: The LLM-as-judge paradigm is increasingly being adopted for automated evaluation of model outputs. While LLM judges have shown promise on constrained evaluation tasks, closed source LLMs display critical shortcomings when deployed in real world applications due to challenges of fine grained metrics and explainability, while task specific evaluation models lack cross-domain generalization. We introduce GLIDER, a powerful 3B evaluator LLM that can score any text input and associated context on arbitrary user defined criteria. GLIDER shows higher Pearson's correlation than GPT-4o on FLASK and greatly outperforms prior evaluation models, achieving comparable performance to LLMs 17x its size. GLIDER supports fine-grained scoring, multilingual reasoning, span highlighting and was trained on 685 domains and 183 criteria. Extensive qualitative analysis shows that GLIDER scores are highly correlated with human judgments, with 91.3% human agreement. We have open-sourced GLIDER to facilitate future research. 

**Abstract (ZH)**: 以下是从英文翻译成中文的版本，确保符合学术规范：

大规模语言模型（LLM）作为评委的范式越来越被用于模型输出的自动化评估。尽管LLM评委在受限评估任务上显示出了一定的潜力，但由于细粒度评估指标和可解释性方面的挑战，闭源LLM在实际应用中暴露出了关键的不足，而针对特定任务的评估模型则缺乏跨领域的泛化能力。我们提出了GLIDER，这是一种强大的30亿参数的评价LLM，能够对任意文本输入及其相关背景进行任意用户定义标准的打分。GLIDER在FLASK上的皮尔逊相关系数优于GPT-4o，并且在评估性能上显著优于之前的所有评估模型，其性能相当于大小是自身17倍的LLM。GLIDER支持细粒度打分、多语言推理和片段高亮，并且在其训练中涵盖了685个领域和183项标准。详尽的定性分析显示，GLIDER的打分与人类判断高度相关，91.3%的人类一致性。我们已开源GLIDER，以便促进未来的研究。 

---
# Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models 

**Title (ZH)**: 视觉语言模型跨模态实体知识提取中的性能差距 

**Authors**: Ido Cohen, Daniela Gottesman, Mor Geva, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2412.14133)  

**Abstract**: Vision-language models (VLMs) excel at extracting and reasoning about information from images. Yet, their capacity to leverage internal knowledge about specific entities remains underexplored. This work investigates the disparity in model performance when answering factual questions about an entity described in text versus depicted in an image. Our results reveal a significant accuracy drop --averaging 19%-- when the entity is presented visually instead of textually. We hypothesize that this decline arises from limitations in how information flows from image tokens to query tokens. We use mechanistic interpretability tools to reveal that, although image tokens are preprocessed by the vision encoder, meaningful information flow from these tokens occurs only in the much deeper layers. Furthermore, critical image processing happens in the language model's middle layers, allowing few layers for consecutive reasoning, highlighting a potential inefficiency in how the model utilizes its layers for reasoning. These insights shed light on the internal mechanics of VLMs and offer pathways for enhancing their reasoning capabilities. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在从图像中提取和推理信息方面表现出色。然而，它们利用特定实体的内部知识的能力仍然未被充分探索。本研究考察了当模型回答关于文本中描述的实体和图像中呈现的实体的事实性问题时，模型表现上的差距。结果显示，当实体以视觉形式呈现而非文字形式呈现时，模型的准确性平均下降了19%。我们认为这一下降源于信息从图像标记流到查询标记的方式存在局限性。我们使用机制可解释性工具揭示，虽然图像标记由视觉编码器预处理，但这些标记中的有意义信息流仅在更深层的层中发生。此外，关键的图像处理发生在语言模型的中间层，这使得连续推理的空间有限，揭示了模型在利用其层进行推理方面可能存在的一种潜在低效性。这些见解阐明了VLMs的内部机制，并提供了增强其推理能力的途径。 

---
# SEKE: Specialised Experts for Keyword Extraction 

**Title (ZH)**: SEKE：专门化的专家关键词提取 

**Authors**: Matej Martinc, Hanh Thi Hong Tran, Senja Pollak, Boshko Koloski  

**Link**: [PDF](https://arxiv.org/pdf/2412.14087)  

**Abstract**: Keyword extraction involves identifying the most descriptive words in a document, allowing automatic categorisation and summarisation of large quantities of diverse textual data. Relying on the insight that real-world keyword detection often requires handling of diverse content, we propose a novel supervised keyword extraction approach based on the mixture of experts (MoE) technique. MoE uses a learnable routing sub-network to direct information to specialised experts, allowing them to specialize in distinct regions of the input space. SEKE, a mixture of Specialised Experts for supervised Keyword Extraction, uses DeBERTa as the backbone model and builds on the MoE framework, where experts attend to each token, by integrating it with a recurrent neural network (RNN), to allow successful extraction even on smaller corpora, where specialisation is harder due to lack of training data. The MoE framework also provides an insight into inner workings of individual experts, enhancing the explainability of the approach. We benchmark SEKE on multiple English datasets, achieving state-of-the-art performance compared to strong supervised and unsupervised baselines. Our analysis reveals that depending on data size and type, experts specialize in distinct syntactic and semantic components, such as punctuation, stopwords, parts-of-speech, or named entities. Code is available at: this https URL 

**Abstract (ZH)**: 关键词提取涉及识别文档中最具有描述性的词语，从而实现对大量异质性文本数据的自动分类和摘要。鉴于现实世界中的关键词检测往往需要处理多样化的内容，我们提出了一种基于专家混合（MoE）技术的新颖监督关键词提取方法。MoE 使用可学习的路由子网络来引导信息流向专门化的专家，使他们能够在输入空间中的不同区域进行专门化。

SEKE，一种基于专门化专家的监督关键词提取方法，以DeBERTa作为骨干模型，并在MoE框架的基础上进行构建，通过将与递归神经网络（RNN）相结合的方法，使专家能够关注每个词元，从而即使在数据量较小的数据集上也能成功提取关键词，这种较小的数据集由于训练数据不足更难以实现专门化。MoE框架还通过对个体专家内部工作机制的深入了解，增强了方法的可解释性。

我们使用多个英语数据集对SEKE进行了基准测试，其性能超越了强大的监督和无监督基线方法。我们的分析表明，根据数据大小和类型的不同，专家们会专注于不同的句法和语义成分，如标点符号、停用词、词性或命名实体。

代码可从此处获得：this https URL 

---
# Digestion Algorithm in Hierarchical Symbolic Forests: A Fast Text Normalization Algorithm and Semantic Parsing Framework for Specific Scenarios and Lightweight Deployment 

**Title (ZH)**: 分层符号森林中的消化算法：一种针对特定场景快速文本规范化算法及轻量级部署的语义解析框架 

**Authors**: Kevin You  

**Link**: [PDF](https://arxiv.org/pdf/2412.14054)  

**Abstract**: Text Normalization and Semantic Parsing have numerous applications in natural language processing, such as natural language programming, paraphrasing, data augmentation, constructing expert systems, text matching, and more. Despite the prominent achievements of deep learning in Large Language Models (LLMs), the interpretability of neural network architectures is still poor, which affects their credibility and hence limits the deployments of risk-sensitive scenarios. In certain scenario-specific domains with scarce data, rapidly obtaining a large number of supervised learning labels is challenging, and the workload of manually labeling data would be enormous. Catastrophic forgetting in neural networks further leads to low data utilization rates. In situations where swift responses are vital, the density of the model makes local deployment difficult and the response time long, which is not conducive to local applications of these fields. Inspired by the multiplication rule, a principle of combinatorial mathematics, and human thinking patterns, a multilayer framework along with its algorithm, the Digestion Algorithm in Hierarchical Symbolic Forests (DAHSF), is proposed to address these above issues, combining text normalization and semantic parsing workflows. The Chinese Scripting Language "Fire Bunny Intelligent Development Platform V2.0" is an important test and application of the technology discussed in this paper. DAHSF can run locally in scenario-specific domains on little datasets, with model size and memory usage optimized by at least two orders of magnitude, thus improving the execution speed, and possessing a promising optimization outlook. 

**Abstract (ZH)**: 文本规范化和语义解析在自然语言处理中有广泛的应用，例如自然语言编程、同义替换、数据扩充、构建专家系统、文本匹配等。尽管大型语言模型（LLMs）在深度学习方面取得了显著成就，但神经网络结构的可解释性仍然较差，这影响了其可信度，从而限制了其在风险敏感场景中的部署。在某些特定领域的数据稀缺情况下，快速获取大量监督学习标签是具有挑战性的，人工标注数据的工作量会非常巨大。神经网络中的灾难性遗忘现象进一步导致了数据利用率低。在需要快速响应的情境下，模型的密集性使得本地部署困难，响应时间较长，这不利于这些领域的本地应用。受乘法定律、组合数学原理以及人类思考模式的启发，提出了一种多层次框架及其算法——层次符号森林中的消化算法（DAHSF），以解决上述问题，结合文本规范化和语义解析的工作流程。汉语编程语言“火兔智能开发平台V2.0”是一种重要的技术测试和应用，用于验证本论文所述技术。DAHSF可以在特定场景下使用少量数据本地运行，并通过至少两个数量级优化模型大小和内存使用，从而提高执行速度，并具有良好的优化前景。 

---
# Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation 

**Title (ZH)**: 多语言大型语言模型中的去偏见和去毒化跨语言转移：一项全面探讨 

**Authors**: Vera Neplenbroek, Arianna Bisazza, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2412.14050)  

**Abstract**: Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. Our results show that finetuning on curated non-harmful text is more effective for mitigating bias, and finetuning on direct preference optimization (DPO) datasets is more effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods. 

**Abstract (ZH)**: 近期生成型大型语言模型（LLMs）在非英语语言上表现出色，但在用这些语言进行提示时，往往会表现出更高的有害社会偏见和毒性水平。前期研究表明，使用专门的训练数据集进行微调可以缓解这种行为，而且在英语上的微调可以转移到其他语言上。本研究旨在探讨不同微调方法对模型偏见和毒性的影响，同时也考察其生成流畅且多样的文本的能力。我们的研究结果表明，使用策划的非有害文本进行微调对缓解偏见更为有效，而使用直接偏好优化（DPO）数据集进行微调则对缓解毒性更为有效。在英语上应用这些方法所导致的缓解效果也可以转移到非英语语言上。研究发现，特定语言的数据在模型预训练数据中的含量可以预测转移的程度。然而，这种偏见和毒性缓解的转移往往以非英语语言的生成能力下降为代价，从而突显了开发特定语言的偏见和毒性缓解方法的重要性。 

---
# Hansel: Output Length Controlling Framework for Large Language Models 

**Title (ZH)**: Hansel：大规模语言模型的输出长度控制框架 

**Authors**: Seoha Song, Junhyun Lee, Hyeonmok Ko  

**Link**: [PDF](https://arxiv.org/pdf/2412.14033)  

**Abstract**: Despite the great success of large language models (LLMs), efficiently controlling the length of the output sequence still remains a challenge. In this paper, we propose Hansel, an efficient framework for length control in LLMs without affecting its generation ability. Hansel utilizes periodically outputted hidden special tokens to keep track of the remaining target length of the output sequence. Together with techniques to avoid abrupt termination of the output, this seemingly simple method proved to be efficient and versatile, while not harming the coherency and fluency of the generated text. The framework can be applied to any pre-trained LLMs during the finetuning stage of the model, regardless of its original positional encoding method. We demonstrate this by finetuning four different LLMs with Hansel and show that the mean absolute error of the output sequence decreases significantly in every model and dataset compared to the prompt-based length control finetuning. Moreover, the framework showed a substantially improved ability to extrapolate to target lengths unseen during finetuning, such as long dialog responses or extremely short summaries. This indicates that the model learns the general means of length control, rather than learning to match output lengths to those seen during training. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）取得巨大成功，但有效控制输出序列长度仍然是一个挑战。本文提出了一种名为Hansel的有效框架，该框架能够在不影响LLMs生成能力的情况下控制输出序列的长度。Hansel利用周期性输出的隐藏特殊标记来跟踪输出序列剩余的目标长度。结合避免输出突然终止的技术，这种看似简单的方法证明了其高效性和灵活性，同时不损害生成文本的一致性和流畅性。该框架可以在模型微调阶段应用于任何预训练的LLMs，而不受其原始位置编码方法的影响。我们通过使用Hansel微调四款不同的LLMs进行了实验，并显示了与基于提示的长度控制微调相比，每个模型和数据集的输出序列的平均绝对误差显著降低。此外，该框架在预测未在微调期间见过的目标长度（如长对话回复或极短摘要）方面表现出显著增强的能力。这表明模型学习了长度控制的一般方法，而非仅仅学习匹配训练期间见过的输出长度。 

---
# Towards an optimised evaluation of teachers' discourse: The case of engaging messages 

**Title (ZH)**: 优化教师话语评价体系：以参与性信息为例 

**Authors**: Samuel Falcon, Jaime Leon  

**Link**: [PDF](https://arxiv.org/pdf/2412.14011)  

**Abstract**: Evaluating teachers' skills is crucial for enhancing education quality and student outcomes. Teacher discourse, significantly influencing student performance, is a key component. However, coding this discourse can be laborious. This study addresses this issue by introducing a new methodology for optimising the assessment of teacher discourse. The research consisted of two studies, both within the framework of engaging messages used by secondary education teachers. The first study involved training two large language models on real-world examples from audio-recorded lessons over two academic years to identify and classify the engaging messages from the lessons' transcripts. This resulted in sensitivities of 84.31% and 91.11%, and specificities of 97.69% and 86.36% in identification and classification, respectively. The second study applied these models to transcripts of audio-recorded lessons from a third academic year to examine the frequency and distribution of message types by educational level and moment of the academic year. Results showed teachers predominantly use messages emphasising engagement benefits, linked to improved outcomes, while one-third highlighted non-engagement disadvantages, associated with increased anxiety. The use of engaging messages declined in Grade 12 and towards the academic year's end. These findings suggest potential interventions to optimise engaging message use, enhancing teaching quality and student outcomes. 

**Abstract (ZH)**: 评估教师技能对于提升教育质量和学生成果至关重要。教师的话语对学生成绩有显著影响，是关键组成部分。然而，编码这一话语可能非常耗时。本研究通过引入一种新的方法来优化教师话语的评估，解决了这一问题。研究包括两个部分，均基于中学教师使用的互动信息。第一部分研究涉及在两年的学术期间，对录音课堂中提供的真实世界示例进行训练，以识别并分类课堂转录中的互动信息。结果显示，在识别和分类方面，敏感度分别为84.31%和91.11%，特异性分别为97.69%和86.36%。第二部分研究将这些模型应用于第三年学术期间录音课堂的转录，以探讨不同教育水平和学年时间点的信息类型频率和分布情况。研究结果显示，教师主要使用强调互动利益的信息，这些信息与改善成果相关，而三分之一的信息则关注非互动的劣势，与增加焦虑相关。互动信息在12年级和学年结束时使用较少。这些发现表明可能采取干预措施以优化互动信息的使用，从而提高教学质量和学生成果。 

---
# FarExStance: Explainable Stance Detection for Farsi 

**Title (ZH)**: FarExStance：具有解释性的波斯语立场检测 

**Authors**: Majid Zarharan, Maryam Hashemi, Malika Behroozrazegh, Sauleh Eetemadi, Mohammad Taher Pilehvar, Jennifer Foster  

**Link**: [PDF](https://arxiv.org/pdf/2412.14008)  

**Abstract**: We introduce FarExStance, a new dataset for explainable stance detection in Farsi. Each instance in this dataset contains a claim, the stance of an article or social media post towards that claim, and an extractive explanation which provides evidence for the stance label. We compare the performance of a fine-tuned multilingual RoBERTa model to several large language models in zero-shot, few-shot, and parameter-efficient fine-tuned settings on our new dataset. On stance detection, the most accurate models are the fine-tuned RoBERTa model, the LLM Aya-23-8B which has been fine-tuned using parameter-efficient fine-tuning, and few-shot Claude-3.5-Sonnet. Regarding the quality of the explanations, our automatic evaluation metrics indicate that few-shot GPT-4o generates the most coherent explanations, while our human evaluation reveals that the best Overall Explanation Score (OES) belongs to few-shot Claude-3.5-Sonnet. The fine-tuned Aya-32-8B model produced explanations most closely aligned with the reference explanations. 

**Abstract (ZH)**: 我们介绍了FarExStance，这是一个用于波斯语解释立场检测的新数据集。该数据集中的每一个实例包含一个断言、一篇文章或社交媒体帖子对该断言的立场，以及一个抽取性的解释，该解释提供了支持立场标签的证据。我们比较了微调的多语言RoBERTa模型与几个大型语言模型在零样本、少样本以及参数效率微调设置下的性能，特别是在我们新的数据集上。在立场检测方面，最准确的模型是微调的RoBERTa模型、使用参数效率微调的LLM Aya-23-8B，以及少样本的Claude-3.5-Sonnet。在解释的质量方面，我们的自动评估指标表明，少样本的GPT-4o生成的解释最为连贯，而我们的手动评估显示，少样本的Claude-3.5-Sonnet在总体解释分数（OES）方面表现最佳。微调后的Aya-32-8B模型生成的解释与参考解释最为接近。 

---
# What makes a good metric? Evaluating automatic metrics for text-to-image consistency 

**Title (ZH)**: 什么是好的度量标准？评估用于文本到图像一致性自动度量的标准 

**Authors**: Candace Ross, Melissa Hall, Adriana Romero Soriano, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2412.13989)  

**Abstract**: Language models are increasingly being incorporated as components in larger AI systems for various purposes, from prompt optimization to automatic evaluation. In this work, we analyze the construct validity of four recent, commonly used methods for measuring text-to-image consistency - CLIPScore, TIFA, VPEval, and DSG - which rely on language models and/or VQA models as components. We define construct validity for text-image consistency metrics as a set of desiderata that text-image consistency metrics should have, and find that no tested metric satisfies all of them. We find that metrics lack sufficient sensitivity to language and visual properties. Next, we find that TIFA, VPEval and DSG contribute novel information above and beyond CLIPScore, but also that they correlate highly with each other. We also ablate different aspects of the text-image consistency metrics and find that not all model components are strictly necessary, also a symptom of insufficient sensitivity to visual information. Finally, we show that all three VQA-based metrics likely rely on familiar text shortcuts (such as yes-bias in QA) that call their aptitude as quantitative evaluations of model performance into question. 

**Abstract (ZH)**: 语言模型越来越多地被作为组件嵌入到各种目的的大型AI系统中，从提示优化到自动评估。在本项工作中，我们分析了四种常用的、用于测量文本与图像一致性的方法（CLIPScore、TIFA、VPEval和DSG）的构念效度，这些方法依赖于语言模型和/或视觉问答模型作为组件。我们定义了文本与图像一致性度量的构念效度为这些度量应具备的一系列标准，并发现没有任何测试中的度量能够满足所有这些标准。我们发现这些度量在语言和视觉属性的敏感性上存在不足。接着，我们发现TIFA、VPEval和DSG在某种程度上提供了比CLIPScore更多的新颖信息，但也高度相关。我们还通过消融分析不同的文本与图像一致性度量方面的因素，发现并不是所有模型组件都是严格必要的，这也反映了对视觉信息敏感性不足的症状。最后，我们表明，基于视觉问答的三个度量可能依赖于熟悉的文本捷径（如问答中的偏向性“是”），这对其作为模型性能定量评估的有效性提出了质疑。 

---
# Prompting Strategies for Enabling Large Language Models to Infer Causation from Correlation 

**Title (ZH)**: 促进大型语言模型从相关性推理出因果关系的提示策略 

**Authors**: Eleni Sgouritsa, Virginia Aglietti, Yee Whye Teh, Arnaud Doucet, Arthur Gretton, Silvia Chiappa  

**Link**: [PDF](https://arxiv.org/pdf/2412.13952)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) are attracting increasing attention. In this work, we focus on causal reasoning and address the task of establishing causal relationships based on correlation information, a highly challenging problem on which several LLMs have shown poor performance. We introduce a prompting strategy for this problem that breaks the original task into fixed subquestions, with each subquestion corresponding to one step of a formal causal discovery algorithm, the PC algorithm. The proposed prompting strategy, PC-SubQ, guides the LLM to follow these algorithmic steps, by sequentially prompting it with one subquestion at a time, augmenting the next subquestion's prompt with the answer to the previous one(s). We evaluate our approach on an existing causal benchmark, Corr2Cause: our experiments indicate a performance improvement across five LLMs when comparing PC-SubQ to baseline prompting strategies. Results are robust to causal query perturbations, when modifying the variable names or paraphrasing the expressions. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力正越来越受到关注。在这项研究中，我们专注于因果推理，并解决基于相关性信息建立因果关系的任务，这是一个高度具有挑战性的问题，许多LLMs在此问题上的表现不佳。我们介绍了一种针对该问题的提示策略，即将原始任务分解为固定子问题，每个子问题对应正式因果发现算法（如PC算法）的一个步骤。提出的提示策略PC-SubQ通过按顺序逐个提示子问题并逐步更新下一个子问题的提示，来引导LLM遵循这些算法步骤。我们利用现有的因果基准Corr2Cause对这种方法进行了评估：实验表明，当将PC-SubQ与基线提示策略进行比较时，它可以在五种LLMs中提高性能表现。即使修改变量名称或重新表述表达方式，这些结果也具有鲁棒性。 

---
# Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence 

**Title (ZH)**: 面向视觉感知的头部发散方法破解大型语言模型中的幻觉问题 

**Authors**: Jinghan He, Kuan Zhu, Haiyun Guo, Junfeng Fang, Zhenglin Hua, Yuheng Jia, Ming Tang, Tat-Seng Chua, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13949)  

**Abstract**: Large vision-language models (LVLMs) have made substantial progress in integrating large language models (LLMs) with visual inputs, enabling advanced multimodal reasoning. Despite their success, a persistent challenge is hallucination-where generated text fails to accurately reflect visual content-undermining both accuracy and reliability. Existing methods focus on alignment training or decoding refinements but primarily address symptoms at the generation stage without probing the underlying causes. In this work, we investigate the internal mechanisms driving hallucination in LVLMs, with an emphasis on the multi-head attention module. Specifically, we introduce Vision-aware Head Divergence (VHD), a metric that quantifies the sensitivity of attention head outputs to visual context. Based on this, our findings reveal the presence of vision-aware attention heads that are more attuned to visual information; however, the model's overreliance on its prior language patterns is closely related to hallucinations. Building on these insights, we propose Vision-aware Head Reinforcement (VHR), a training-free approach to mitigate hallucination by enhancing the role of vision-aware attention heads. Extensive experiments demonstrate that our method achieves superior performance compared to state-of-the-art approaches in mitigating hallucinations, while maintaining high efficiency with negligible additional time overhead. 

**Abstract (ZH)**: 大规模的视觉-语言模型（LVLMs）已经在将大规模语言模型（LLMs）与视觉输入结合上取得了显著进展，从而增强了多模态推理能力。尽管取得了成功，但持续存在的问题是幻觉——生成的文本未能准确反映视觉内容，从而损害了准确性和可靠性。现有方法主要关注对齐训练或解码 refinement，但主要针对生成阶段的症状，而没有探究其根本原因。在此项工作中，我们调查了驱动LVLMs幻觉的内部机制，重点放在多头注意模块上。具体而言，我们引入了视觉感知头发散（VHD）这一度量标准，用于量化注意力头输出对视觉上下文的敏感性。基于这一分析，我们的发现揭示了更关注视觉信息的视觉感知注意力头的存在；然而，模型对先前语言模式的过度依赖与幻觉密切相关。基于这些见解，我们提出了视觉感知头强化（VHR），这是一种无需训练的方法，通过增强视觉感知注意力头的角色来减轻幻觉。广泛的实验显示，我们的方法在缓解幻觉方面优于现有最先进的方法，并且在保持高效的同时，几乎不增加额外的时间开销。 

---
# A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI 

**Title (ZH)**: 《别名玫瑰：由大模型生成的解释是人类解释的优良代理，用于收集自然语言推理任务中的标签分布》

这个标题翻译符合学术规范，保留了原文的核心概念，并使用了合适的中文学术表达。 

**Authors**: Beiduo Chen, Siyao Peng, Anna Korhonen, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2412.13942)  

**Abstract**: Disagreement in human labeling is ubiquitous, and can be captured in human judgment distributions (HJDs). Recent research has shown that explanations provide valuable information for understanding human label variation (HLV) and large language models (LLMs) can approximate HJD from a few human-provided label-explanation pairs. However, collecting explanations for every label is still time-consuming. This paper examines whether LLMs can be used to replace humans in generating explanations for approximating HJD. Specifically, we use LLMs as annotators to generate model explanations for a few given human labels. We test ways to obtain and combine these label-explanations with the goal to approximate human judgment distribution. We further compare the resulting human with model-generated explanations, and test automatic and human explanation selection. Our experiments show that LLM explanations are promising for NLI: to estimate HJD, generated explanations yield comparable results to human's when provided with human labels. Importantly, our results generalize from datasets with human explanations to i) datasets where they are not available and ii) challenging out-of-distribution test sets. 

**Abstract (ZH)**: 人类标签中的分歧普遍存在，并且可以在人类判断分布（HJDs）中捕捉到。近期研究表明，解释提供了理解人类标签差异（HLV）的重要信息，而且大语言模型（LLMs）可以从少量的人类提供的标签-解释对中模拟HJD。然而，为每个标签收集解释仍然耗时。本文探讨了是否可以利用LLMs来替代人类生成解释，以接近HJD。具体来说，我们使用LLMs作为注释工具，为给定的人类标签生成模型解释。我们测试了获得和组合这些标签-解释的方式，旨在接近人类判断分布。我们进一步比较了人类和模型生成的解释，并测试了自动和人工解释的选择。实验结果表明，对于自然语言推理（NLI），生成的解释对于估计HJD而言具有潜力：提供人类标签时，生成的解释能够产生与人类相当的结果。重要的是，我们的结果不仅适用于包含人类解释的数据集，还适用于i) 未提供人类解释的数据集，以及ii) 挑战性的离分布测试集。 

---
# Language verY Rare for All 

**Title (ZH)**: “Language Very Rare for All” 的中文翻译可以是：

“语言极度稀少对所有人的影响”

或者更加学术化的翻译可以是：

“极度稀少的语言对全体人群的影响”

这样的翻译既保留了原文的意思，又符合学术写作的规范。 

**Authors**: Ibrahim Merad, Amos Wolf, Ziad Mazzawi, Yannick Léo  

**Link**: [PDF](https://arxiv.org/pdf/2412.13924)  

**Abstract**: In the quest to overcome language barriers, encoder-decoder models like NLLB have expanded machine translation to rare languages, with some models (e.g., NLLB 1.3B) even trainable on a single GPU. While general-purpose LLMs perform well in translation, open LLMs prove highly competitive when fine-tuned for specific tasks involving unknown corpora. We introduce LYRA (Language verY Rare for All), a novel approach that combines open LLM fine-tuning, retrieval-augmented generation (RAG), and transfer learning from related high-resource languages. This study is exclusively focused on single-GPU training to facilitate ease of adoption. Our study focuses on two-way translation between French and Monégasque, a rare language unsupported by existing translation tools due to limited corpus availability. Our results demonstrate LYRA's effectiveness, frequently surpassing and consistently matching state-of-the-art encoder-decoder models in rare language translation. 

**Abstract (ZH)**: 在克服语言壁垒的探索中，诸如NLLB这样的编码器-解码器模型将机器翻译扩展到了稀有语言领域，一些模型（如NLLB 1.3B）甚至可以在单块GPU上进行训练。虽然通用的语言模型在翻译任务上表现良好，但在特定任务中对未知语料进行微调后，开源语言模型展现出极高的竞争力。我们提出了LYRA（Language verY Rare for All）这一新颖的方法，它结合了开源语言模型的微调、检索增强生成（RAG）以及从相关高资源语言的迁移学习。本研究专注于单块GPU的训练，以促进其易于应用。我们的研究集中在法语和摩纳哥语之间的双向翻译任务上，这是因为现有的翻译工具由于语料库有限而无法支持摩纳哥语的翻译。我们的实验结果表明，LYRA在稀有语言翻译方面的有效性，经常超越并稳定地匹配最先进的编码器-解码器模型。 

---
# Pipeline Analysis for Developing Instruct LLMs in Low-Resource Languages: A Case Study on Basque 

**Title (ZH)**: 低资源语言开发指令型大规模语言模型的管道分析：关于巴斯克语的案例研究 

**Authors**: Ander Corral, Ixak Sarasua, Xabier Saralegi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13922)  

**Abstract**: Large language models (LLMs) are typically optimized for resource-rich languages like English, exacerbating the gap between high-resource and underrepresented languages. This work presents a detailed analysis of strategies for developing a model capable of following instructions in a low-resource language, specifically Basque, by focusing on three key stages: pre-training, instruction tuning, and alignment with human preferences. Our findings demonstrate that continual pre-training with a high-quality Basque corpus of around 600 million words improves natural language understanding (NLU) of the foundational model by over 12 points. Moreover, instruction tuning and human preference alignment using automatically translated datasets proved highly effective, resulting in a 24-point improvement in instruction-following performance. The resulting models, Llama-eus-8B and Llama-eus-8B-instruct, establish a new state-of-the-art for Basque in the sub-10B parameter category. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常针对资源丰富的语言（如英语）进行优化，这加剧了高资源语言和欠代表语言之间的差距。本研究详尽分析了开发一种能在资源欠丰富的语言（如巴斯克语）中遵循指令的模型的策略，重点关注三个关键阶段：预训练、指令调优和与人类偏好的对齐。我们的研究发现表明，使用大约6亿词的高质量巴斯克语语料库进行持续的预训练，可以将基础模型的自然语言理解（NLU）提高超过12个百分点。此外，使用自动翻译的数据集进行指令调优和人类偏好对齐证明效果显著，这导致指令遵循性能提高了24个百分点。所生成的模型，Llama-eus-8B和Llama-eus-8B-instruct，在参数量小于10B的子类别中建立了新的最佳表现。 

---
# Understanding and Analyzing Model Robustness and Knowledge-Transfer in Multilingual Neural Machine Translation using TX-Ray 

**Title (ZH)**: 利用TX-Ray 理解和分析多语言神经机器翻译中模型鲁棒性和知识迁移现象 

**Authors**: Vageesh Saxena, Sharid Loáiciga, Nils Rethmeier  

**Link**: [PDF](https://arxiv.org/pdf/2412.13881)  

**Abstract**: Neural networks have demonstrated significant advancements in Neural Machine Translation (NMT) compared to conventional phrase-based approaches. However, Multilingual Neural Machine Translation (MNMT) in extremely low-resource settings remains underexplored. This research investigates how knowledge transfer across languages can enhance MNMT in such scenarios. Using the Tatoeba translation challenge dataset from Helsinki NLP, we perform English-German, English-French, and English-Spanish translations, leveraging minimal parallel data to establish cross-lingual mappings. Unlike conventional methods relying on extensive pre-training for specific language pairs, we pre-train our model on English-English translations, setting English as the source language for all tasks. The model is fine-tuned on target language pairs using joint multi-task and sequential transfer learning strategies. Our work addresses three key questions: (1) How can knowledge transfer across languages improve MNMT in extremely low-resource scenarios? (2) How does pruning neuron knowledge affect model generalization, robustness, and catastrophic forgetting? (3) How can TX-Ray interpret and quantify knowledge transfer in trained models? Evaluation using BLEU-4 scores demonstrates that sequential transfer learning outperforms baselines on a 40k parallel sentence corpus, showcasing its efficacy. However, pruning neuron knowledge degrades performance, increases catastrophic forgetting, and fails to improve robustness or generalization. Our findings provide valuable insights into the potential and limitations of knowledge transfer and pruning in MNMT for extremely low-resource settings. 

**Abstract (ZH)**: 与传统的基于短语的方法相比，神经网络在神经机器翻译（NMT）中取得了显著的进展。然而，在极度低资源设置下，多语言神经机器翻译（MNMT）仍然缺乏探索。本研究探讨了跨语言知识迁移如何在这些场景下增强MNMT。我们使用赫尔辛基NLP提供的Tatoeba翻译挑战数据集，进行英-德、英-法、英-西翻译任务，利用少量平行数据建立跨语言映射。与依赖于特定语言对大量预训练的常规方法不同，我们使用英-英翻译数据进行预训练，将英语作为所有任务的源语言。模型通过联合多任务和序列知识迁移策略在目标语言对上进行微调。我们的研究针对三个核心问题进行探讨：（1）跨语言知识迁移如何在极度低资源场景中改善MNMT？（2）剪枝神经元知识对模型泛化、稳健性和灾难性遗忘有何影响？（3）TX-Ray如何解释和量化训练模型中的知识迁移？

评估结果显示，在40,000个平行句子的语料库上，序列知识迁移优于基线模型，展示了其有效性。然而，剪枝神经元知识会降低性能，增加灾难性遗忘，并未能提升泛化或稳健性。我们的研究结果为极度低资源设置下的MNMT中的知识迁移和剪枝的潜力和局限性提供了宝贵的见解。 

---
# Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings 

**Title (ZH)**: 螃蟹：在黑盒设置下通过自动生成消耗资源进行LLM-DOS攻击 

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.13879)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (AutoDoS). AutoDoS introduces DoS Attack Tree and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting Length Trojan in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over 250 $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our code is available at \url{this https URL}. 

**Abstract (ZH)**: -large语言模型（LLMs）在多种任务中展现了卓越的性能。然而，LLMs仍然易受外部威胁，尤其是服务拒绝攻击（DoS攻击）。具体而言，LLM-DoS攻击旨在耗尽计算资源并阻塞服务。然而，先前的研究多侧重于进行白盒攻击，忽视了黑盒环境。在此工作中，我们提出了一种适用于黑盒LLMs的自动化算法，称为Auto-Generation for LLM-DoS Attack（AutoDoS）。AutoDoS引入了DoS攻击树，并通过优化提示节点的覆盖率来在黑盒条件下增强攻击效果。我们的方法可以通过对提示节点的语义改进，提高隐蔽性，从而绕过现有的防御措施。此外，我们发现，在基本DoS提示中植入长度特洛伊木马有助于提高攻击效果。实验结果显示，AutoDoS显著增加了服务响应延迟，超过250倍，导致严重的GPU利用和内存使用资源消耗。我们的代码可在 \url{此链接} 获取。 

---
# Domain-adaptative Continual Learning for Low-resource Tasks: Evaluation on Nepali 

**Title (ZH)**: 针对低资源任务的领域自适应连续学习：以尼泊尔语为例的评估 

**Authors**: Sharad Duwal, Suraj Prasai, Suresh Manandhar  

**Link**: [PDF](https://arxiv.org/pdf/2412.13860)  

**Abstract**: Continual learning has emerged as an important research direction due to the infeasibility of retraining large language models (LLMs) from scratch in the event of new data availability. Of great interest is the domain-adaptive pre-training (DAPT) paradigm, which focuses on continually training a pre-trained language model to adapt it to a domain it was not originally trained on. In this work, we evaluate the feasibility of DAPT in a low-resource setting, namely the Nepali language. We use synthetic data to continue training Llama 3 8B to adapt it to the Nepali language in a 4-bit QLoRA setting. We evaluate the adapted model on its performance, forgetting, and knowledge acquisition. We compare the base model and the final model on their Nepali generation abilities, their performance on popular benchmarks, and run case-studies to probe their linguistic knowledge in Nepali. We see some unsurprising forgetting in the final model, but also surprisingly find that increasing the number of shots during evaluation yields better percent increases in the final model (as high as 19.29% increase) compared to the base model (4.98%), suggesting latent retention. We also explore layer-head self-attention heatmaps to establish dependency resolution abilities of the final model in Nepali. 

**Abstract (ZH)**: 持续学习由于在新数据可用时重新训练大型语言模型（LLMs）从头开始的不可行性，已逐渐成为重要的研究方向。其中，领域适应预训练（DAPT）范式尤为引人关注，它侧重于持续训练预训练的语言模型，使其能够适应其最初未受训的领域。在本研究中，我们评估了DAPT在低资源环境下的可行性，特别是针对尼泊尔语。我们使用合成数据，继续训练Llama 3-8B，使其适应尼泊尔语，并在4比特的QLoRA设置下进行。我们从性能、遗忘和知识获取三个方面评估了改编后的模型。我们将基础模型和最终模型在尼泊尔语生成能力、流行基准上的表现进行了对比，并进行了案例研究，以进一步探索它们在尼泊尔语中的语言知识。我们发现最终模型中一些不出乎意料的遗忘现象，但同时令人惊讶地发现，在评估中增加样本数量（shots）可以显著提高最终模型的表现（最高增加19.29%），而基础模型的相对改善仅为4.98%，这表明存在潜在的知识保留。我们还探讨了层头自注意力热点图，以评估最终模型在尼泊尔语中的依赖性解决能力。 

---
# RACQUET: Unveiling the Dangers of Overlooked Referential Ambiguity in Visual LLMs 

**Title (ZH)**: RACQUET: 揭示视觉LLMs中被忽视的指代歧义危险 

**Authors**: Alberto Testoni, Barbara Plank, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2412.13835)  

**Abstract**: Ambiguity resolution is key to effective communication. While humans effortlessly address ambiguity through conversational grounding strategies, the extent to which current language models can emulate these strategies remains unclear. In this work, we examine referential ambiguity in image-based question answering by introducing RACQUET, a carefully curated dataset targeting distinct aspects of ambiguity. Through a series of evaluations, we reveal significant limitations and problems of overconfidence of state-of-the-art large multimodal language models in addressing ambiguity in their responses. The overconfidence issue becomes particularly relevant for RACQUET-BIAS, a subset designed to analyze a critical yet underexplored problem: failing to address ambiguity leads to stereotypical, socially biased responses. Our results underscore the urgency of equipping models with robust strategies to deal with uncertainty without resorting to undesirable stereotypes. 

**Abstract (ZH)**: 有效的沟通关键在于解决歧义。人类通过会话锚定策略轻松地解决歧义，但目前的语言模型在模仿这些策略方面的能力仍不清楚。在这项工作中，我们通过引入RACQUET（一个精心策划的数据集，针对歧义的不同方面）来研究基于图像的问题回答中的指称歧义。通过一系列评估，我们揭示了当前最先进的大型多模态语言模型在回应中解决歧义方面的显著局限性和问题。特别是在RACQUET-BIAS子集的研究中，该子集旨在分析一个关键但尚未充分研究的问题：未能解决歧义会导致刻板化、社会偏向的回答。我们的研究结果强调了在模型中培养处理不确定性而不诉诸不 desirable 的刻板印象的稳健策略的紧迫性。 

---
# Enhancing Rhetorical Figure Annotation: An Ontology-Based Web Application with RAG Integration 

**Title (ZH)**: 基于本体的增强修辞 figura 注释：结合RAG的网络应用 

**Authors**: Ramona Kühn, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.13799)  

**Abstract**: Rhetorical figures play an important role in our communication. They are used to convey subtle, implicit meaning, or to emphasize statements. We notice them in hate speech, fake news, and propaganda. By improving the systems for computational detection of rhetorical figures, we can also improve tasks such as hate speech and fake news detection, sentiment analysis, opinion mining, or argument mining. Unfortunately, there is a lack of annotated data, as well as qualified annotators that would help us build large corpora to train machine learning models for the detection of rhetorical figures. The situation is particularly difficult in languages other than English, and for rhetorical figures other than metaphor, sarcasm, and irony. To overcome this issue, we develop a web application called "Find your Figure" that facilitates the identification and annotation of German rhetorical figures. The application is based on the German Rhetorical ontology GRhOOT which we have specially adapted for this purpose. In addition, we improve the user experience with Retrieval Augmented Generation (RAG). In this paper, we present the restructuring of the ontology, the development of the web application, and the built-in RAG pipeline. We also identify the optimal RAG settings for our application. Our approach is one of the first to practically use rhetorical ontologies in combination with RAG and shows promising results. 

**Abstract (ZH)**: 修辞手法在我们的沟通中发挥着重要作用。它们用于传达微妙的暗示意义，或强调陈述。我们在仇恨言论、假新闻和宣传中都能注意到它们的存在。通过改进修辞手法计算检测系统，我们也可以提升仇恨言论和假新闻检测、情感分析、观点挖掘或论点挖掘等任务。不幸的是，标注数据和合格的标注者仍然缺乏，这阻碍了我们构建用于修辞手法检测的大型语料库以训练机器学习模型。在这种情况下，情况尤其困难，尤其是在非英语语言中，以及对于除了隐喻、讽刺和幽默之外的其他修辞手法。

为了解决这一问题，我们开发了一个名为“Find your Figure”的网络应用，旨在简化德语修辞手法的识别和标注过程。该应用基于专门为此改编的德语修辞本体GRhOOT。此外，我们通过检索增强生成（RAG）技术改进了用户体验。在本文中，我们介绍了本体的重构、网络应用的开发以及内置的RAG管道。我们还确定了最适合我们应用的RAG设置。我们的方法是第一个实际利用本体与RAG相结合的方法之一，并显示出令人鼓舞的结果。 

---
# MATCHED: Multimodal Authorship-Attribution To Combat Human Trafficking in Escort-Advertisement Data 

**Title (ZH)**: MATCHED：多模态作者归属以应对援交广告数据中的人口贩卖问题 

**Authors**: Vageesh Saxena, Benjamin Bashpole, Gijs Van Dijck, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2412.13794)  

**Abstract**: Human trafficking (HT) remains a critical issue, with traffickers increasingly leveraging online escort advertisements (ads) to advertise victims anonymously. Existing detection methods, including Authorship Attribution (AA), often center on text-based analyses and neglect the multimodal nature of online escort ads, which typically pair text with images. To address this gap, we introduce MATCHED, a multimodal dataset of 27,619 unique text descriptions and 55,115 unique images collected from the Backpage escort platform across seven U.S. cities in four geographical regions. Our study extensively benchmarks text-only, vision-only, and multimodal baselines for vendor identification and verification tasks, employing multitask (joint) training objectives that achieve superior classification and retrieval performance on in-distribution and out-of-distribution (OOD) datasets. Integrating multimodal features further enhances this performance, capturing complementary patterns across text and images. While text remains the dominant modality, visual data adds stylistic cues that enrich model performance. Moreover, text-image alignment strategies like CLIP and BLIP2 struggle due to low semantic overlap and vague connections between the modalities of escort ads, with end-to-end multimodal training proving more robust. Our findings emphasize the potential of multimodal AA (MAA) to combat HT, providing LEAs with robust tools to link ads and disrupt trafficking networks. 

**Abstract (ZH)**: 人口贩卖（HT）仍然是一个关键问题，贩运者越来越多地利用在线陪侍广告（广告）来匿名发布受害者信息。现有的检测方法，包括作者归属（AA），往往集中于基于文本的分析，而忽略了在线陪侍广告的多模态性质，这些广告通常包含文本和图片。为了解决这一问题，我们引入了MATCHED数据集，该数据集包含了27,619个独特的文本描述和55,115个独特的图片，这些数据收集自四个地理区域的七个城市中的Backpage陪侍平台。我们的研究广泛地比较了仅基于文本、仅基于视觉和多模态基准方法在供应商识别和验证任务中的表现，使用了多任务（联合）训练目标，在内分布和外分布（OOD）数据集上实现了优越的分类和检索性能。进一步整合多模态特征可以进一步提升性能，捕捉文本和图片之间的互补模式。尽管文本仍然是主要模态，但视觉数据提供的风格性线索能够丰富模型的表现。此外，用于单模态训练的CLIP和BLIP2在陪侍广告的模态之间语义重合较低且模态间关联不明确的情况下表现不佳，而端到端的多模态训练更为稳健。我们的研究结果强调了多模态作者归属（MAA）在打击HT方面的潜力，为执法机构提供了可靠的工具，以连接广告并瓦解贩卖网络。 

---
# Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems with Large Language Models 

**Title (ZH)**: 物理推理器：知识增强的物理问题解决推理方法用于大型语言模型 

**Authors**: Xinyu Pang, Ruixin Hong, Zhanke Zhou, Fangrui Lv, Xinwei Yang, Zhilong Liang, Bo Han, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13791)  

**Abstract**: Physics problems constitute a significant aspect of reasoning, necessitating complicated reasoning ability and abundant physics knowledge. However, existing large language models (LLMs) frequently fail due to a lack of knowledge or incorrect knowledge application. To mitigate these issues, we propose Physics Reasoner, a knowledge-augmented framework to solve physics problems with LLMs. Specifically, the proposed framework constructs a comprehensive formula set to provide explicit physics knowledge and utilizes checklists containing detailed instructions to guide effective knowledge application. Namely, given a physics problem, Physics Reasoner solves it through three stages: problem analysis, formula retrieval, and guided reasoning. During the process, checklists are employed to enhance LLMs' self-improvement in the analysis and reasoning stages. Empirically, Physics Reasoner mitigates the issues of insufficient knowledge and incorrect application, achieving state-of-the-art performance on SciBench with an average accuracy improvement of 5.8%. 

**Abstract (ZH)**: 物理问题构成了推理的重要方面，需要复杂的推理能力和丰富的物理知识。然而，现有的大规模语言模型（LLMs）由于知识不足或错误应用知识而经常失败。为了解决这些问题，我们提出了一种知识增强框架——Physics Reasoner，以利用LLMs解决物理问题。具体而言，该框架构建了一个全面的公式集，提供明确的物理知识，并利用包含详细指导说明的清单来指导有效知识应用。具体来说，给定一个物理问题，Physics Reasoner通过三个阶段来解决这个问题：问题分析、公式检索和引导式推理。在过程中，清单被用来在分析和推理阶段增强LLMs的自我改进能力。实验结果显示，Physics Reasoner缓解了知识不足和错误应用的问题，使其在SciBench上的性能达到了最先进的水平，平均准确率提升了5.8%。 

---
# Open Universal Arabic ASR Leaderboard 

**Title (ZH)**: 开放通用阿拉伯语ASR排行榜 

**Authors**: Yingzhi Wang, Anas Alhmoud, Muhammad Alqurishi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13788)  

**Abstract**: In recent years, the enhanced capabilities of ASR models and the emergence of multi-dialect datasets have increasingly pushed Arabic ASR model development toward an all-dialect-in-one direction. This trend highlights the need for benchmarking studies that evaluate model performance on multiple dialects, providing the community with insights into models' generalization capabilities.
In this paper, we introduce Open Universal Arabic ASR Leaderboard, a continuous benchmark project for open-source general Arabic ASR models across various multi-dialect datasets. We also provide a comprehensive analysis of the model's robustness, speaker adaptation, inference efficiency, and memory consumption. This work aims to offer the Arabic ASR community a reference for models' general performance and also establish a common evaluation framework for multi-dialectal Arabic ASR models. 

**Abstract (ZH)**: 近年来，ASR模型的能力不断增强，多方言数据集的出现使得阿拉伯语ASR模型的发展趋势趋向于统一处理多种方言的方向。这一趋势突显了需要进行基准测试研究的重要性，这些研究可以评估模型在多种方言上的性能，为社区提供模型泛化能力的见解。
在本文中，我们介绍了Open Universal Arabic ASR Leaderboard，这是一个针对各种多方言数据集的开源通用阿拉伯语ASR模型的持续基准测试项目。我们还提供了对模型鲁棒性、说话人适应性、推理效率和内存消耗的全面分析。本研究旨在为阿拉伯语ASR社区提供模型总体性能的参考，并建立一个多方言阿拉伯语ASR模型的共同评估框架。 

---
# Knowledge Editing with Dynamic Knowledge Graphs for Multi-hop Question Answering 

**Title (ZH)**: 基于动态知识图谱的知识编辑在多跳问答中的应用 

**Authors**: Yifan Lu, Yigeng Zhou, Jing Li, Yequan Wang, Xuebo Liu, Daojing He, Fangming Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13782)  

**Abstract**: Multi-hop question answering (MHQA) poses a significant challenge for large language models (LLMs) due to the extensive knowledge demands involved. Knowledge editing, which aims to precisely modify the LLMs to incorporate specific knowledge without negatively impacting other unrelated knowledge, offers a potential solution for addressing MHQA challenges with LLMs. However, current solutions struggle to effectively resolve issues of knowledge conflicts. Most parameter-preserving editing methods are hindered by inaccurate retrieval and overlook secondary editing issues, which can introduce noise into the reasoning process of LLMs. In this paper, we introduce KEDKG, a novel knowledge editing method that leverages a dynamic knowledge graph for MHQA, designed to ensure the reliability of answers. KEDKG involves two primary steps: dynamic knowledge graph construction and knowledge graph augmented generation. Initially, KEDKG autonomously constructs a dynamic knowledge graph to store revised information while resolving potential knowledge conflicts. Subsequently, it employs a fine-grained retrieval strategy coupled with an entity and relation detector to enhance the accuracy of graph retrieval for LLM generation. Experimental results on benchmarks show that KEDKG surpasses previous state-of-the-art models, delivering more accurate and reliable answers in environments with dynamic information. 

**Abstract (ZH)**: 多跳问答（MHQA）对大型语言模型（LLMs）提出了严峻挑战，因为这需要广泛的知识支持。知识编辑旨在精确修改LLMs以引入特定知识而不影响其他无关知识，为解决MHQA挑战提供了潜在解决方案。然而，当前的解决方案在处理知识冲突方面难以有效解决问题。大多数保持参数不变的编辑方法受到不准确检索的限制，并且忽略了二级编辑问题，这可能会引入噪音到LLMs的推理过程中。在本文中，我们引入了一种新的知识编辑方法KEDKG，该方法利用动态知识图谱来解决MHQA问题，旨在确保答案的可靠性。KEDKG包含两个主要步骤：动态知识图谱构建和知识图谱增强生成。首先，KEDKG自主构建一个动态知识图谱以存储修订信息并解决潜在的知识冲突。随后，它采用细粒度的检索策略结合实体和关系检测器，以提高图谱检索的准确性从而增强LLMs生成的准确性。在基准测试上的实验结果表明，KEDKG超越了以前的先进模型，在动态信息环境中提供了更准确和可靠的答案。 

---
# Meta-Reflection: A Feedback-Free Reflection Learning Framework 

**Title (ZH)**: 元反思：一种无反馈的反射学习框架 

**Authors**: Yaoke Wang, Yun Zhu, Xintong Bao, Wenqiao Zhang, Suyang Dai, Kehan Chen, Wenqiang Li, Gang Huang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13781)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in natural language understanding and reasoning, they often display undesirable behaviors, such as generating hallucinations and unfaithful reasoning. A prevalent strategy to mitigate these issues is the use of reflection, which refines responses through an iterative process. However, while promising, reflection heavily relies on high-quality external feedback and requires iterative multi-agent inference processes, thus hindering its practical application. In this paper, we propose Meta-Reflection, a novel feedback-free reflection mechanism that necessitates only a single inference pass without external feedback. Motivated by the human ability to remember and retrieve reflections from past experiences when encountering similar problems, Meta-Reflection integrates reflective insights into a codebook, allowing the historical insights to be stored, retrieved, and used to guide LLMs in problem-solving. To thoroughly investigate and evaluate the practicality of Meta-Reflection in real-world scenarios, we introduce an industrial e-commerce benchmark named E-commerce Customer Intent Detection (ECID). Extensive experiments conducted on both public datasets and the ECID benchmark highlight the effectiveness and efficiency of our proposed approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在自然语言理解和推理方面表现出色，但在生成虚构内容和不忠于事实的推理方面常常表现出不良行为。减轻这些问题的一个常见策略是使用反思机制，该机制通过迭代过程逐步细化回应。然而，虽然前景广阔，但这种方法高度依赖高质量的外部反馈，并需要迭代的多代理推理过程，从而限制了其实际应用。在本文中，我们提出了Meta-反思，这是一种新颖的无反馈反思机制，仅需单次推理过程而无需外部反馈。受到人类在遇到类似问题时能够回忆起过去经验中反思的能力的启发，Meta-反思将反思洞察整合到代码书中，使过去的见解得以存储、检索，并用于指导LLMs解决问题。为了全面调查和评估Meta-反思在实际应用场景中的实用性，我们引入了一个工业电子商务基准，名为电子商务客户意图检测（ECID）。在公共数据集和ECID基准上的广泛实验充分展示了我们所提出方法的有效性和高效性。 

---
# LLM-SEM: A Sentiment-Based Student Engagement Metric Using LLMS for E-Learning Platforms 

**Title (ZH)**: LLM-SEM：基于情感的学生参与度指标使用大型语言模型应用于在线教育平台 

**Authors**: Ali Hamdi, Ahmed Abdelmoneim Mazrou, Mohamed Shaltout  

**Link**: [PDF](https://arxiv.org/pdf/2412.13765)  

**Abstract**: Current methods for analyzing student engagement in e-learning platforms, including automated systems, often struggle with challenges such as handling fuzzy sentiment in text comments and relying on limited metadata. Traditional approaches, such as surveys and questionnaires, also face issues like small sample sizes and scalability. In this paper, we introduce LLM-SEM (Language Model-Based Student Engagement Metric), a novel approach that leverages video metadata and sentiment analysis of student comments to measure engagement. By utilizing recent Large Language Models (LLMs), we generate high-quality sentiment predictions to mitigate text fuzziness and normalize key features such as views and likes. Our holistic method combines comprehensive metadata with sentiment polarity scores to gauge engagement at both the course and lesson levels. Extensive experiments were conducted to evaluate various LLM models, demonstrating the effectiveness of LLM-SEM in providing a scalable and accurate measure of student engagement. We fine-tuned LLMs, including AraBERT, TXLM-RoBERTa, LLama 3B and Gemma 9B from Ollama, using human-annotated sentiment datasets to enhance prediction accuracy. 

**Abstract (ZH)**: 当前用于分析在线学习平台学生参与度的方法，包括自动化系统，通常面临处理文本评论中的模糊情感和依赖有限元数据的挑战。传统的调查和问卷方法也存在样本量小和难以扩展的问题。本文引入了一种新颖的方法——基于语言模型的学生参与度评估（LLM-SEM），该方法利用视频元数据和学生评论的情感分析来衡量参与度。通过利用最新的大规模语言模型（LLMs），我们生成高质量的情感预测，以减轻文本模糊性，并标准化诸如浏览量和点赞数等关键指标。我们的综合方法结合全面的元数据和情感极性评分，以在课程和课节两个层面上衡量参与度。我们进行了广泛的实验以评估各种LLM模型，证明了LLM-SEM在提供一种可扩展且准确的学生参与度度量方面的有效性。我们对包括AraBERT、TXLM-RoBERTa、LLama 3B和来自Ollama的Gemma 9B在内的LLM模型进行了微调，使用人工标注的情感数据集来提高预测准确性。 

---
# RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment 

**Title (ZH)**: RAG-RewardBench：在检索增强生成中对偏好对齐的奖励模型进行基准测试 

**Authors**: Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13746)  

**Abstract**: Despite the significant progress made by existing retrieval augmented language models (RALMs) in providing trustworthy responses and grounding in reliable sources, they often overlook effective alignment with human preferences. In the alignment process, reward models (RMs) act as a crucial proxy for human values to guide optimization. However, it remains unclear how to evaluate and select a reliable RM for preference alignment in RALMs. To this end, we propose RAG-RewardBench, the first benchmark for evaluating RMs in RAG settings. First, we design four crucial and challenging RAG-specific scenarios to assess RMs, including multi-hop reasoning, fine-grained citation, appropriate abstain, and conflict robustness. Then, we incorporate 18 RAG subsets, six retrievers, and 24 RALMs to increase the diversity of data sources. Finally, we adopt an LLM-as-a-judge approach to improve preference annotation efficiency and effectiveness, exhibiting a strong correlation with human annotations. Based on the RAG-RewardBench, we conduct a comprehensive evaluation of 45 RMs and uncover their limitations in RAG scenarios. Additionally, we also reveal that existing trained RALMs show almost no improvement in preference alignment, highlighting the need for a shift towards preference-aligned this http URL release our benchmark and code publicly at this https URL for future work. 

**Abstract (ZH)**: 尽管现有的检索增强语言模型（RALMs）在提供可信响应和可靠来源基础上取得了显著进展，它们往往在与人类偏好有效对齐方面有所疏忽。在对齐过程中，奖赏模型（RMs）作为人类价值观的关键代理，用于引导优化。然而，尚不清楚如何评估和选择合适的RM以在RLAMs中实现偏好对齐。为此，我们提出了RAG-RewardBench，这是首个在RAG环境中评估RM的基准。首先，我们设计了四个关键且具有挑战性的RAG特定场景来评估RM，包括多跳推理、精细引证、适当弃权和冲突稳健性。然后，我们结合了18个RAG子集、六种检索器和24种RLMs，以增加数据源的多样性。最后，我们采用LLM作为裁判的方法以提高偏好注释的效率和有效性，显示出很强的人类标注相关性。基于RAG-RewardBench，我们对45种RM进行了全面评估，并揭示了它们在RAG环境中存在的局限性。此外，我们还发现现有的训练好的RLMs在偏好对齐方面几乎没有改进，强调了向偏好对齐转变的必要性。我们将在未来的工作中释放我们的基准和代码，请参见以下链接：<https://github.com/alibaba/RAG-RewardBench>。 

---
# Learning Complex Word Embeddings in Classical and Quantum Spaces 

**Title (ZH)**: 在经典和量子空间中学习复杂词嵌入 

**Authors**: Carys Harvey, Stephen Clark, Douglas Brown, Konstantinos Meichanetzidis  

**Link**: [PDF](https://arxiv.org/pdf/2412.13745)  

**Abstract**: We present a variety of methods for training complex-valued word embeddings, based on the classical Skip-gram model, with a straightforward adaptation simply replacing the real-valued vectors with arbitrary vectors of complex numbers. In a more "physically-inspired" approach, the vectors are produced by parameterised quantum circuits (PQCs), which are unitary transformations resulting in normalised vectors which have a probabilistic interpretation. We develop a complex-valued version of the highly optimised C code version of Skip-gram, which allows us to easily produce complex embeddings trained on a 3.8B-word corpus for a vocabulary size of over 400k, for which we are then able to train a separate PQC for each word. We evaluate the complex embeddings on a set of standard similarity and relatedness datasets, for some models obtaining results competitive with the classical baseline. We find that, while training the PQCs directly tends to harm performance, the quantum word embeddings from the two-stage process perform as well as the classical Skip-gram embeddings with comparable numbers of parameters. This enables a highly scalable route to learning embeddings in complex spaces which scales with the size of the vocabulary rather than the size of the training corpus. In summary, we demonstrate how to produce a large set of high-quality word embeddings for use in complex-valued and quantum-inspired NLP models, and for exploring potential advantage in quantum NLP models. 

**Abstract (ZH)**: 我们基于经典的Skip-gram模型提出了多种复值词嵌入训练方法，通过简单地将实值向量替换为任意复数向量对传统模型进行适应。在一种更为“物理启发式”的方法中，向量由参数化量子电路（PQCs）生成，这些电路执行幺正变换，从而生成归一化向量，具有概率解释。我们开发了Skip-gram的复值版本，并且通过易于获得基于3.8B单词的大型语料库训练而成的超过40万词汇量的复值嵌入。随后，我们能够为每个单词训练一个单独的PQC。我们评估了这些复值嵌入在标准相似性和相关性数据集上的表现，对于某些模型，其结果与经典的基线模型相媲美。我们发现，虽然直接训练PQCs往往损害性能，但在两阶段过程生成的量子词嵌入与具有相似参数数量的经典Skip-gram嵌入具有相同的表现。这提供了一条高效扩展的途径来在大型词汇量而非训练语料库大小的限制下学习复数空间中的嵌入。总之，我们展示了如何生产大量高质量的复值词嵌入，用于复杂值和量子启发式NLP模型，并探索潜在的量子NLP模型优势。 

---
# Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models 

**Title (ZH)**: 联邦学习与RAG集成：一种适用于医疗大型语言模型的可扩展方法 

**Authors**: Jincheol Jung, Hongju Jeong, Eui-Nam Huh  

**Link**: [PDF](https://arxiv.org/pdf/2412.13720)  

**Abstract**: This study analyzes the performance of domain-specific Large Language Models (LLMs) for the medical field by integrating Retrieval-Augmented Generation (RAG) systems within a federated learning framework. Leveraging the inherent advantages of federated learning, such as preserving data privacy and enabling distributed computation, this research explores the integration of RAG systems with models trained under varying client configurations to optimize performance. Experimental results demonstrate that the federated learning-based models integrated with RAG systems consistently outperform their non-integrated counterparts across all evaluation metrics. This study highlights the potential of combining federated learning and RAG systems for developing domain-specific LLMs in the medical field, providing a scalable and privacy-preserving solution for enhancing text generation capabilities. 

**Abstract (ZH)**: 本文通过在联邦学习框架中整合检索增强生成（RAG）系统，分析了针对医疗领域的专用大型语言模型（LLMs）的性能。利用联邦学习的固有优势，如保护数据隐私和实现分布式计算，本研究探讨了在不同客户端配置下将RAG系统与训练模型进行集成，以优化性能的可能性。实验结果表明，基于联邦学习并与RAG系统集成的模型在所有评估指标上均优于未集成的模型。本文强调了将联邦学习与RAG系统相结合，开发医疗领域的专用LLMs的潜力，提供了一种可扩展且保护隐私的解决方案，以增强文本生成能力。 

---
# Towards Automatic Evaluation for Image Transcreation 

**Title (ZH)**: 面向图像再创作的自动评价方法研究 

**Authors**: Simran Khanuja, Vivek Iyer, Claire He, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2412.13717)  

**Abstract**: Beyond conventional paradigms of translating speech and text, recently, there has been interest in automated transcreation of images to facilitate localization of visual content across different cultures. Attempts to define this as a formal Machine Learning (ML) problem have been impeded by the lack of automatic evaluation mechanisms, with previous work relying solely on human evaluation. In this paper, we seek to close this gap by proposing a suite of automatic evaluation metrics inspired by machine translation (MT) metrics, categorized into: a) Object-based, b) Embedding-based, and c) VLM-based. Drawing on theories from translation studies and real-world transcreation practices, we identify three critical dimensions of image transcreation: cultural relevance, semantic equivalence and visual similarity, and design our metrics to evaluate systems along these axes. Our results show that proprietary VLMs best identify cultural relevance and semantic equivalence, while vision-encoder representations are adept at measuring visual similarity. Meta-evaluation across 7 countries shows our metrics agree strongly with human ratings, with average segment-level correlations ranging from 0.55-0.87. Finally, through a discussion of the merits and demerits of each metric, we offer a robust framework for automated image transcreation evaluation, grounded in both theoretical foundations and practical application. Our code can be found here: this https URL 

**Abstract (ZH)**: 超越传统翻译声频和文本的方法，最近，人们开始对自动化图像创译（transcreation）产生了兴趣，以便在不同文化中促进视觉内容的本地化。由于缺乏自动评估机制，将此问题形式化为机器学习（ML）问题的努力受到了阻碍，之前的研究依靠纯人工评估。本文旨在通过提出一套受机器翻译（MT）指标启发的自动评估指标来填补这一空白，这些指标分为三类：a) 基于对象，b) 基于嵌入，c) 基于视觉语言模型（VLM）。借鉴翻译研究的理论和实际世界中的图像创译实践，我们确定了图像创译的三个关键维度：文化相关性、语义等价性和视觉相似性，并设计了相应的评估指标来衡量系统的性能。结果显示，专用视觉语言模型（VLMs）在识别文化相关性和语义等价性方面表现最好，而视觉编码器的表现则体现在测量视觉相似性上。在七个不同国家的元评估中，我们的方法与人工评分高度一致，各个子段级别的相关系数范围在0.55到0.87之间。最后，通过讨论每个指标的优点和缺点，我们提出了一种既基于理论基础又适用于实践的自动图像创译评估框架。我们的代码可以通过以下链接找到：this https URL 

---
# Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models 

**Title (ZH)**: typhoon 2: 一类开源文本和多模态泰语大型语言模型 

**Authors**: Kunat Pipatanakul, Potsawee Manakul, Natapong Nitarach, Warit Sirichotedumrong, Surapon Nonesung, Teetouch Jaknamon, Parinthapat Pengpun, Pittawat Taveekitworachai, Adisai Na-Thalang, Sittipong Sripaisarnmongkol, Krisanapong Jirayoot, Kasima Tharnpipitchai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13702)  

**Abstract**: This paper introduces Typhoon 2, a series of text and multimodal large language models optimized for the Thai language. The series includes models for text, vision, and audio. Typhoon2-Text builds on state-of-the-art open models, such as Llama 3 and Qwen2, and we perform continual pre-training on a mixture of English and Thai data. We employ various post-training techniques to enhance Thai language performance while preserving the base models' original capabilities. We release text models across a range of sizes, from 1 to 70 billion parameters, available in both base and instruction-tuned variants. Typhoon2-Vision improves Thai document understanding while retaining general visual capabilities, such as image captioning. Typhoon2-Audio introduces an end-to-end speech-to-speech model architecture capable of processing audio, speech, and text inputs and generating both text and speech outputs simultaneously. 

**Abstract (ZH)**: 本文介绍了Typhoon 2系列，该系列是针对泰语进行优化的文本和多模态大型语言模型。该系列包括文本、视觉和音频模型。Typhoon2-Text基于最先进的开源模型，如Llama 3和Qwen2，并在英泰混合数据集上进行了持续的预训练。我们在后训练过程中采用多种技术以提升泰语性能，同时保持基础模型的原始功能。我们提供了从1亿到70亿参数不等的文本模型，包括基础版和指令微调版。Typhoon2-Vision在保留通用视觉能力（如图像描述生成）的基础上改善了泰语文档的理解能力。Typhoon2-Audio则引入了一种端到端的语音到语音模型架构，能够处理音频、语音和文本输入，并同时生成文本和语音输出。 

---
# Towards Efficient and Explainable Hate Speech Detection via Model Distillation 

**Title (ZH)**: 通过模型蒸馏实现高效可解释的恶意言论检测 

**Authors**: Paloma Piot, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2412.13698)  

**Abstract**: Automatic detection of hate and abusive language is essential to combat its online spread. Moreover, recognising and explaining hate speech serves to educate people about its negative effects. However, most current detection models operate as black boxes, lacking interpretability and explainability. In this context, Large Language Models (LLMs) have proven effective for hate speech detection and to promote interpretability. Nevertheless, they are computationally costly to run. In this work, we propose distilling big language models by using Chain-of-Thought to extract explanations that support the hate speech classification task. Having small language models for these tasks will contribute to their use in operational settings. In this paper, we demonstrate that distilled models deliver explanations of the same quality as larger models while surpassing them in classification performance. This dual capability, classifying and explaining, advances hate speech detection making it more affordable, understandable and actionable. 

**Abstract (ZH)**: 自动检测仇恨言论和滥用语言对于遏制其在网络上的传播至关重要。此外，识别和解释仇恨言论有助于教育人们了解其负面影响。然而，当前大多数检测模型都是黑盒模型，缺乏可解释性和可解释性。在这种背景下，大型语言模型（LLMs）在仇恨言论检测和提升可解释性方面已经展现出有效性。尽管如此，它们在运行时计算成本较高。在本工作中，我们提出了一种通过使用思维链（Chain-of-Thought）提炼大型语言模型的方法，以提取支持仇恨言论分类任务的解释。为了在这些任务中使用小型语言模型，我们证明了提炼后的模型能够提供与大型模型相媲美的解释质量，并在分类性能上超过了它们。这种双重能力，即分类和解释，推动了仇恨言论检测更加经济、易懂和可操作。 

---
# AntiLeak-Bench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge 

**Title (ZH)**: AntiLeak-Bench: 防止数据污染通过自动构建包含更新实际世界知识的基准 

**Authors**: Xiaobao Wu, Liangming Pan, Yuxi Xie, Ruiwen Zhou, Shuai Zhao, Yubo Ma, Mingzhe Du, Rui Mao, Anh Tuan Luu, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13670)  

**Abstract**: Data contamination hinders fair LLM evaluation by introducing test data into newer models' training sets. Existing studies solve this challenge by updating benchmarks with newly collected data. However, they fail to guarantee contamination-free evaluation as the newly collected data may contain pre-existing knowledge, and their benchmark updates rely on intensive human labor. To address these issues, we in this paper propose AntiLeak-Bench, an automated anti-leakage benchmarking framework. Instead of simply using newly collected data, we construct samples with explicitly new knowledge absent from LLMs' training sets, which thus ensures strictly contamination-free evaluation. We further design a fully automated workflow to build and update our benchmark without human labor. This significantly reduces the cost of benchmark maintenance to accommodate emerging LLMs. Through extensive experiments, we highlight that data contamination likely exists before LLMs' cutoff time and demonstrate AntiLeak-Bench effectively overcomes this challenge. 

**Abstract (ZH)**: 数据污染妨碍了对新一代语言模型（LLM）的公平评估，因为它将测试数据引入了模型的训练集。现有研究通过更新基准数据集来解决这一挑战，但这些方法无法保证完全无污染的评估，因为新收集的数据可能仍包含预先存在的知识，而且基准数据集的更新依赖于大量的人工劳动。为了解决这些问题，本文提出了一种自动化的抗泄露基准框架——AntiLeak-Bench。我们不仅使用新收集的数据，还通过构建不含语言模型训练集中存在知识的样本来构造数据样本，从而确保严格的无污染评估。此外，我们还设计了一个完全自动化的流程来构建和更新基准，无需人工参与。这显著降低了维护基准的成本，以便能够适应新兴的语言模型。通过大量的实验，我们强调数据污染可能在语言模型截止时间之前就已存在，并展示了AntiLeak-Bench能够有效克服这一挑战。 

---
# Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation 

**Title (ZH)**: 针对个性化虚假信息生成滥用风险的大型语言模型漏洞评估 

**Authors**: Aneta Zugecova, Dominik Macko, Ivan Srba, Robert Moro, Jakub Kopal, Katarina Marcincinova, Matus Mesarcik  

**Link**: [PDF](https://arxiv.org/pdf/2412.13666)  

**Abstract**: The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts rises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluation of vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers. 

**Abstract (ZH)**: 近期大规模语言模型（LLMs）生成高质量内容的能力，使得这些内容难以被人类区分，引发了对其潜在误用的担忧。先前的研究显示，LLMs 可以有效地被利用来生成遵循预定义叙事的虚假新闻文章。尽管它们在生成个性化内容方面的能力也得到了评估，且大多数情况下被发现具有可用性，但结合个性化能力和生成虚假信息的能力尚未进行全面研究。这种危险的结合需要触发LLMs的安全过滤机制（如果存在）。本研究通过评估近期开放和封闭环境下的LLMs的脆弱性以及它们生成个性化虚假新闻文章的能力，填补了这一空白。我们进一步探讨了LLMs能否可靠地元评估个性化质量，以及个性化如何影响生成文本的可识别性。研究结果表明，大多数评估的LLMs的安全过滤机制和免责声明并未充分发挥作用，需要加强。此外，研究表明，个性化实际上减少了安全过滤的激活次数，有效起到了一种“越狱”的作用。这种行为必须紧急地被LLM开发者和服务提供商解决。 

---
# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference 

**Title (ZH)**: 更聪明、更高效、更快捷、更持久：一种现代双向编码器，适用于快速、内存高效及长上下文的微调和推理 

**Authors**: Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, Iacopo Poli  

**Link**: [PDF](https://arxiv.org/pdf/2412.13663)  

**Abstract**: Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs. 

**Abstract (ZH)**: 以下是从编码器到编码器的transformer模型，如BERT，在检索和分类任务中提供了与较大的解码器模型相比优异的性能与大小的权衡。尽管BERT自发布以来一直是众多生产管道的核心，但对其改进幅度有限。在本文中，我们介绍了ModernBERT，将现代模型优化应用于编码器到编码器模型，并代表了对较旧编码器的重大帕累托改进。ModernBERT模型在包含数万亿词汇量并具有8192原生序列长度的训练下，展示了涵盖多种分类任务和不同领域（包括代码）的广泛评估中的最先进结果，在单向和多向检索方面表现突出。除了强大的下游性能外，ModernBERT还是最高效的速度和内存使用率的编码器，并且设计用于在常见的GPU上进行推理。 

---
# PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling 

**Title (ZH)**: PsyDT：使用大型语言模型构建具有个性化咨询风格的心理咨询数字孪生体 

**Authors**: Haojie Xie, Yirong Chen, Xiaofen Xing, Jingkai Lin, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13660)  

**Abstract**: Currently, large language models (LLMs) have made significant progress in the field of psychological counseling. However, existing mental health LLMs overlook a critical issue where they do not consider the fact that different psychological counselors exhibit different personal styles, including linguistic style and therapy techniques, etc. As a result, these LLMs fail to satisfy the individual needs of clients who seek different counseling styles. To help bridge this gap, we propose PsyDT, a novel framework using LLMs to construct the Digital Twin of Psychological counselor with personalized counseling style. Compared to the time-consuming and costly approach of collecting a large number of real-world counseling cases to create a specific counselor's digital twin, our framework offers a faster and more cost-effective solution. To construct PsyDT, we utilize dynamic one-shot learning by using GPT-4 to capture counselor's unique counseling style, mainly focusing on linguistic style and therapy techniques. Subsequently, using existing single-turn long-text dialogues with client's questions, GPT-4 is guided to synthesize multi-turn dialogues of specific counselor. Finally, we fine-tune the LLMs on the synthetic dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with personalized counseling style. Experimental results indicate that our proposed PsyDT framework can synthesize multi-turn dialogues that closely resemble real-world counseling cases and demonstrate better performance compared to other baselines, thereby show that our framework can effectively construct the digital twin of psychological counselor with a specific counseling style. 

**Abstract (ZH)**: 目前，大型语言模型（LLMs）在心理咨询领域取得了显著进展。然而，现有的心理健康LLMs忽视了一个关键问题，即它们没有考虑到不同心理辅导员具有不同的个人风格，包括语言风格和治疗技巧等。结果，这些LLMs无法满足寻求不同咨询风格的客户个体需求。为了解决这一问题，我们提出了一种新的框架PsyDT，利用LLMs构建具有个性化咨询风格的心理辅导员数字孪生。与收集大量真实世界咨询案例来创建特定心理辅导员数字孪生的耗时且成本高昂的方法相比，我们的框架提供了一种更快且更经济的解决方案。为了构建PsyDT，我们利用动态单次学习，使用GPT-4捕捉心理辅导员的独特咨询风格，主要集中在语言风格和治疗技巧上。随后，通过使用客户问题的现有单轮长文本对话，指导GPT-4合成特定心理辅导员的多轮对话。最后，我们在合成数据集PsyDTCorpus上微调LLMs，以实现具有个性化咨询风格的心理辅导员数字孪生。实验结果表明，我们提出的心理辅导员数字孪生框架能够合成类似于真实世界咨询案例的多轮对话，并且在与其他基线方法的性能上表现出色，从而证明了我们的框架能够有效构建具有特定咨询风格的心理辅导员数字孪生。 

---
# SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation 

**Title (ZH)**: 范围：优化长上下文生成中的键值缓存压缩 

**Authors**: Jialong Wu, Zhenglin Wang, Linhai Zhang, Yilong Lai, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13649)  

**Abstract**: Key-Value (KV) cache has become a bottleneck of LLMs for long-context generation. Despite the numerous efforts in this area, the optimization for the decoding phase is generally ignored. However, we believe such optimization is crucial, especially for long-output generation tasks based on the following two observations: (i) Excessive compression during the prefill phase, which requires specific full context impairs the comprehension of the reasoning task; (ii) Deviation of heavy hitters occurs in the reasoning tasks with long outputs. Therefore, SCOPE, a simple yet efficient framework that separately performs KV cache optimization during the prefill and decoding phases, is introduced. Specifically, the KV cache during the prefill phase is preserved to maintain the essential information, while a novel strategy based on sliding is proposed to select essential heavy hitters for the decoding phase. Memory usage and memory transfer are further optimized using adaptive and discontinuous strategies. Extensive experiments on LongGenBench show the effectiveness and generalization of SCOPE and its compatibility as a plug-in to other prefill-only KV compression methods. 

**Abstract (ZH)**: 长上下文生成中，键值（KV）缓存已成为大规模语言模型（LLM）的一个瓶颈。尽管该领域已做出了众多努力，但解码阶段的优化通常被忽视。然而，我们认为这种优化至关重要，特别是在基于以下两个观察结果的长期输出生成任务中：(i) 预填充阶段过度压缩，需要完整上下文，这会影响推理任务的理解能力；(ii) 在长期输出的推理任务中，重要的缓存项会发生偏差。因此，提出了一种简单而高效的框架SCOPE，该框架分别在预填充和解码阶段对KV缓存进行优化。具体来说，在预填充阶段，保留KV缓存以保持必要的信息，而在解码阶段提出了一种基于滑动窗口的新策略来选择必要的突出缓存项。此外，使用自适应和分段策略优化了内存使用和内存传输。在LongGenBench上的大量实验表明，SCOPE的有效性、泛化能力和与其他仅预填充KV压缩方法的插件兼容性。 

---
# LIFT: Improving Long Context Understanding Through Long Input Fine-Tuning 

**Title (ZH)**: LIFT：通过长输入微调提高长上下文理解 

**Authors**: Yansheng Mao, Jiaqi Li, Fanxu Meng, Jing Xiong, Zilong Zheng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13626)  

**Abstract**: Long context understanding remains challenging for large language models due to their limited context windows. This paper introduces Long Input Fine-Tuning (LIFT) for long context modeling, a novel framework that enhances LLM performance on long-context tasks by adapting model parameters to the context at test time. LIFT enables efficient processing of lengthy inputs without the computational burden of offline long-context adaptation, and can improve the long-context capabilities of arbitrary short-context models. The framework is further enhanced by integrating in-context learning and pre-LIFT supervised fine-tuning. The combination of in-context learning and LIFT enables short-context models like Llama 3 to handle arbitrarily long contexts and consistently improves their performance on popular long-context benchmarks like LooGLE and LongBench. We also provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research. 

**Abstract (ZH)**: 长上下文理解仍然是大型语言模型面临的挑战，因为它们的上下文窗口有限。本论文提出了长输入微调（LIFT），这是一种新颖的框架，通过在测试时适应模型参数来增强LLM在长上下文任务中的性能。LIFT允许高效地处理长度较长的输入，而不必承担离线长上下文适应的计算负担，从而可以提高任意短上下文模型的长上下文能力。该框架进一步通过集成上下文学习和预LIFT监督微调得到了增强。上下文学习和LIFT的结合使如Llama 3这类短上下文模型能够处理任意长的上下文，并且在流行的长上下文基准测试LooGLE和LongBench上能够提升其性能。我们还对LIFT在长上下文理解方面的优势和局限性进行了全面分析，为未来的研究提供了宝贵的指导方向。 

---
# Are LLMs Good Literature Review Writers? Evaluating the Literature Review Writing Ability of Large Language Models 

**Title (ZH)**: 大型语言模型是有效的文献综述撰写者吗？评估大型语言模型的文献综述撰写能力 

**Authors**: Xuemei Tang, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13612)  

**Abstract**: The literature review is a crucial form of academic writing that involves complex processes of literature collection, organization, and summarization. The emergence of large language models (LLMs) has introduced promising tools to automate these processes. However, their actual capabilities in writing comprehensive literature reviews remain underexplored, such as whether they can generate accurate and reliable references. To address this gap, we propose a framework to assess the literature review writing ability of LLMs automatically. We evaluate the performance of LLMs across three tasks: generating references, writing abstracts, and writing literature reviews. We employ external tools for a multidimensional evaluation, which includes assessing hallucination rates in references, semantic coverage, and factual consistency with human-written context. By analyzing the experimental results, we find that, despite advancements, even the most sophisticated models still cannot avoid generating hallucinated references. Additionally, different models exhibit varying performance in literature review writing across different disciplines. 

**Abstract (ZH)**: 文献综述是学术写作中一种至关重要的形式，涉及文献的复杂收集、组织和总结过程。大型语言模型（LLMs）的出现带来了自动完成这些过程的有希望工具。然而，LLMs在撰写全面文献综述的实际能力仍处于探索阶段，如它们能否生成准确可靠的参考文献。为填补这一空白，我们提出了一种框架，以自动评估LLMs的文献综述写作能力。我们在三项任务——生成参考文献、撰写摘要和撰写文献综述——上评估了LLMs的表现。我们使用外部工具进行多维度评估，包括评估参考文献中的虚构率、语义覆盖范围以及与人类撰写内容的事实一致性。通过对实验结果的分析，我们发现，尽管技术在不断进步，但最先进的模型仍然无法完全避免生成虚构的参考文献。此外，不同模型在不同学科的文献综述写作上表现出不同的性能。 

---
# Beyond Outcomes: Transparent Assessment of LLM Reasoning in Games 

**Title (ZH)**: 超越结果：透明评估大语言模型在游戏中的推理能力 

**Authors**: Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.13602)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in real-world applications that demand complex reasoning. To track progress, robust benchmarks are required to evaluate their capabilities beyond superficial pattern recognition. However, current LLM reasoning benchmarks often face challenges such as insufficient interpretability, performance saturation or data contamination. To address these challenges, we introduce GAMEBoT, a gaming arena designed for rigorous and transparent assessment of LLM reasoning capabilities. GAMEBoT decomposes complex reasoning in games into predefined modular subproblems. This decomposition allows us to design a suite of Chain-of-Thought (CoT) prompts that leverage domain knowledge to guide LLMs in addressing these subproblems before action selection. Furthermore, we develop a suite of rule-based algorithms to generate ground truth for these subproblems, enabling rigorous validation of the LLMs' intermediate reasoning steps. This approach facilitates evaluation of both the quality of final actions and the accuracy of the underlying reasoning process. GAMEBoT also naturally alleviates the risk of data contamination through dynamic games and head-to-head LLM competitions. We benchmark 17 prominent LLMs across eight games, encompassing various strategic abilities and game characteristics. Our results suggest that GAMEBoT presents a significant challenge, even when LLMs are provided with detailed CoT prompts. Project page: \url{this https URL} 

**Abstract (ZH)**: 大型语言模型（LLMs）在需要复杂推理的实际应用中越来越广泛。为了跟踪进展并评估其超越表面模式识别的能力，需要具备强大解释性和透明性的基准测试。然而，现有的LLM推理基准测试往往面临诸如解释性不足、性能饱和或数据污染等挑战。为应对这些挑战，我们提出了一种名为GAMEBoT的游戏竞技场，该竞技场旨在对LLM的推理能力进行严格和透明的评估。GAMEBoT将游戏中的复杂推理分解为预定义的模组化子问题。这种分解方法使我们能够设计一系列基于推理过程（CoT）的提示，利用领域知识引导模型解决这些子问题，从而在采取行动之前进行推理。此外，我们还开发了一套基于规则的算法来生成这些子问题的 ground truth，从而能够严谨地验证模型的中间推理步骤。这种方法既评估了最终行动的质量，也评估了背后的推理过程的准确性。通过动态游戏和LLM之间的竞争，GAMEBoT还自然地缓解了数据污染的风险。我们对八款不同特性和策略能力的游戏进行了17种主流LLM的评估。结果表明，即使在提供详细CoT提示的情况下，GAMEBoT也构成了一项显著挑战。项目页面：[点击这里](this https URL) 

---
# EvoWiki: Evaluating LLMs on Evolving Knowledge 

**Title (ZH)**: EvoWiki：评估生成型语言模型在不断更新的知识上的表现 

**Authors**: Wei Tang, Yixin Cao, Yang Deng, Jiahao Ying, Bo Wang, Yizhe Yang, Yuyue Zhao, Qi Zhang, Xuanjing Huang, Yugang Jiang, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13582)  

**Abstract**: Knowledge utilization is a critical aspect of LLMs, and understanding how they adapt to evolving knowledge is essential for their effective deployment. However, existing benchmarks are predominantly static, failing to capture the evolving nature of LLMs and knowledge, leading to inaccuracies and vulnerabilities such as contamination. In this paper, we introduce EvoWiki, an evolving dataset designed to reflect knowledge evolution by categorizing information into stable, evolved, and uncharted states. EvoWiki is fully auto-updatable, enabling precise evaluation of continuously changing knowledge and newly released LLMs. Through experiments with Retrieval-Augmented Generation (RAG) and Contunual Learning (CL), we evaluate how effectively LLMs adapt to evolving knowledge. Our results indicate that current models often struggle with evolved knowledge, frequently providing outdated or incorrect responses. Moreover, the dataset highlights a synergistic effect between RAG and CL, demonstrating their potential to better adapt to evolving knowledge. EvoWiki provides a robust benchmark for advancing future research on the knowledge evolution capabilities of large language models. 

**Abstract (ZH)**: 知识利用是大语言模型（LLM）的一个关键方面，理解它们如何适应不断发展变化的知识是有效部署它们所必需的。然而，现有的基准大多是静态的，无法捕捉到LLM和知识的演变特性，导致不准确性和漏洞，如知识污染。在本文中，我们提出了EvoWiki，这是一个不断更新的数据集，旨在通过将信息分类为稳定态、进化态和未知态来反映知识的演变。EvoWiki是全自动可更新的，能够精确评估不断变化的知识和新发布的LLM。通过使用检索增强生成（RAG）和持续学习（CL）进行实验，我们评估了LLM如何适应不断发展变化的知识。我们的结果显示，当前模型在处理进化知识时经常面临困难，频繁提供过时或错误的答案。此外，数据集还突显了RAG和CL之间协同作用的效果，展示了它们在适应不断演变的知识方面的潜力。EvoWiki为未来研究大语言模型的知识演变能力提供了一个稳健的基准。 

---
# Socio-Culturally Aware Evaluation Framework for LLM-Based Content Moderation 

**Title (ZH)**: 基于大型语言模型的内容审核社会文化意识评估框架 

**Authors**: Shanu Kumar, Gauri Kholkar, Saish Mendke, Anubhav Sadana, Parag Agrawal, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2412.13578)  

**Abstract**: With the growth of social media and large language models, content moderation has become crucial. Many existing datasets lack adequate representation of different groups, resulting in unreliable assessments. To tackle this, we propose a socio-culturally aware evaluation framework for LLM-driven content moderation and introduce a scalable method for creating diverse datasets using persona-based generation. Our analysis reveals that these datasets provide broader perspectives and pose greater challenges for LLMs than diversity-focused generation methods without personas. This challenge is especially pronounced in smaller LLMs, emphasizing the difficulties they encounter in moderating such diverse content. 

**Abstract (ZH)**: 随着社交媒体和大规模语言模型的发展，内容审核变得至关重要。现有的许多数据集未能充分代表不同群体，导致评估结果不够可靠。为解决这一问题，我们提出了一种社会文化意识强的内容审核评估框架，并介绍了一种基于角色生成的可扩展方法来创建多样化数据集。我们的分析表明，这些数据集为语言模型提供了更广泛的观点并带来了更大的挑战，而这种挑战比没有角色参与的专注于多样性的生成方法更为明显。这种挑战在较小的语言模型中尤为突出，突显了它们在审核如此多样化的内容时所面临的困难。 

---
# Generating Long-form Story Using Dynamic Hierarchical Outlining with Memory-Enhancement 

**Title (ZH)**: 使用记忆增强的动态层级大纲生成长篇故事 

**Authors**: Qianyue Wang, Jinwu Hu, Zhengping Li, Yufeng Wang, daiyuan li, Yu Hu, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13575)  

**Abstract**: Long-form story generation task aims to produce coherent and sufficiently lengthy text, essential for applications such as novel writingand interactive storytelling. However, existing methods, including LLMs, rely on rigid outlines or lack macro-level planning, making it difficult to achieve both contextual consistency and coherent plot development in long-form story generation. To address this issues, we propose Dynamic Hierarchical Outlining with Memory-Enhancement long-form story generation method, named DOME, to generate the long-form story with coherent content and plot. Specifically, the Dynamic Hierarchical Outline(DHO) mechanism incorporates the novel writing theory into outline planning and fuses the plan and writing stages together, improving the coherence of the plot by ensuring the plot completeness and adapting to the uncertainty during story generation. A Memory-Enhancement Module (MEM) based on temporal knowledge graphs is introduced to store and access the generated content, reducing contextual conflicts and improving story coherence. Finally, we propose a Temporal Conflict Analyzer leveraging temporal knowledge graphs to automatically evaluate the contextual consistency of long-form story. Experiments demonstrate that DOME significantly improves the fluency, coherence, and overall quality of generated long stories compared to state-of-the-art methods. 

**Abstract (ZH)**: 长文体故事生成任务旨在产出连贯且足够冗长的文本，这对诸如小说写作和互动故事讲述等应用至关重要。然而，现有的方法，包括大语言模型（LLMs），依赖于僵硬的提纲或缺乏宏观规划，使得在长文体故事生成中实现上下文一致性和连贯的情节发展变得困难。为解决这一问题，我们提出了一种融合记忆增强的动态层次提纲生成方法（DOME），以生成内容和情节连贯的长文体故事。具体而言，动态层次提纲（DHO）机制将小说写作理论融入提纲规划中，并将规划与写作阶段结合，通过确保情节完整性和适应故事生成过程中的不确定性来提升情节连贯性。基于时间知识图谱的记忆增强模块（MEM）用于存储和访问生成的内容，减少上下文冲突并提高故事连贯性。最后，我们提出了一种基于时间知识图谱的时间冲突分析器，用于自动评估长文体故事的上下文一致性。实验结果表明，DOME方法在流畅性、连贯性和生成长故事的总体质量方面显著优于现有最先进的方法。 

---
# EscapeBench: Pushing Language Models to Think Outside the Box 

**Title (ZH)**: EscapeBench: 促使语言模型跳出固定思维模式 

**Authors**: Cheng Qian, Peixuan Han, Qinyu Luo, Bingxiang He, Xiusi Chen, Yuji Zhang, Hongyi Du, Jiarui Yao, Xiaocheng Yang, Denghui Zhang, Yunzhu Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.13549)  

**Abstract**: Language model agents excel in long-session planning and reasoning, but existing benchmarks primarily focus on goal-oriented tasks with explicit objectives, neglecting creative adaptation in unfamiliar environments. To address this, we introduce EscapeBench, a benchmark suite of room escape game environments designed to challenge agents with creative reasoning, unconventional tool use, and iterative problem-solving to uncover implicit goals. Our results show that current LM models, despite employing working memory and Chain-of-Thought reasoning, achieve only 15% average progress without hints, highlighting their limitations in creativity. To bridge this gap, we propose EscapeAgent, a framework designed to enhance creative reasoning through Foresight (innovative tool use) and Reflection (identifying unsolved tasks). Experiments show that EscapeAgent can execute action chains over 1,000 steps while maintaining logical coherence. It navigates and completes games with up to 40% fewer steps and hints, performs robustly across varying difficulty levels, and achieves higher action success rates with more efficient and innovative puzzle-solving strategies. All the data and codes are released. 

**Abstract (ZH)**: 语言模型代理在长时间规划和推理方面表现出色，但现有的基准测试主要集中在具有明确目标的指令性任务上，而忽视了在陌生环境中的创意适应能力。为解决这一问题，我们引入了EscapeBench，这是一项基于房间逃脱游戏环境的基准测试套件，旨在挑战代理的创造性推理、非常规工具使用以及迭代问题解决能力以发现潜在的目标。实验结果表明，尽管当前的语言模型使用工作记忆和链式推理，但在没有提示的情况下，它们的平均进度仅为15%，这揭示了它们在创造力方面存在的局限性。为解决这一差距，我们提出了一种EscapeAgent框架，旨在通过前瞻（创新工具使用）和反思（识别未解决的任务）来增强创造性推理能力。实验表明，EscapeAgent能够执行超过1000步的动作序列，同时保持逻辑连贯性。它能够在较少的步数和提示下导航并完成游戏，表现出优异的鲁棒性，并以更高效和创新的谜题解决策略提高了动作成功率。所有数据和代码均已发布。 

---
# Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary 

**Title (ZH)**: 基于自适应粒度球决策边界的大规模开放意图分类 

**Authors**: Yanhua Li, Xiaocao Ouyang, Chaofan Pan, Jie Zhang, Sen Zhao, Shuyin Xia, Xin Yang, Guoyin Wang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13542)  

**Abstract**: Open intent classification is critical for the development of dialogue systems, aiming to accurately classify known intents into their corresponding classes while identifying unknown intents. Prior boundary-based methods assumed known intents fit within compact spherical regions, focusing on coarse-grained representation and precise spherical decision boundaries. However, these assumptions are often violated in practical scenarios, making it difficult to distinguish known intent classes from unknowns using a single spherical boundary. To tackle these issues, we propose a Multi-granularity Open intent classification method via adaptive Granular-Ball decision boundary (MOGB). Our MOGB method consists of two modules: representation learning and decision boundary acquiring. To effectively represent the intent distribution, we design a hierarchical representation learning method. This involves iteratively alternating between adaptive granular-ball clustering and nearest sub-centroid classification to capture fine-grained semantic structures within known intent classes. Furthermore, multi-granularity decision boundaries are constructed for open intent classification by employing granular-balls with varying centroids and radii. Extensive experiments conducted on three public datasets demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 开放意图分类对于对话系统的开发至关重要，旨在准确将已知意图分类到相应的类别中，同时识别未知意图。以往的基于边界的分类方法假设已知意图位于紧凑的球形区域内，主要集中在粗粒度表示和精确的球形决策边界上。然而，在实际场景中，这些假设往往被违反，使得仅使用单一的球形边界难以区分已知意图类别和未知意图。为解决这些问题，我们提出了一种基于自适应粒度球形决策边界的多粒度开放意图分类方法（MOGB）。MOGB方法包括两个模块：表示学习和决策边界获取。为了有效表示意图分布，我们设计了一种分层表示学习方法。这种方法通过交替进行自适应粒度球形聚类和最近子中心分类来迭代捕获已知意图类别内的细粒度语义结构。此外，通过使用具有不同质心和半径的粒度球，我们构建了多粒度决策边界，用于开放意图分类。在三个公开数据集上进行的广泛实验表明，我们提出的方法具有有效性。 

---
# Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning 

**Title (ZH)**: 基准测试与改进大型视觉-语言模型以实现基础视觉图理解与推理 

**Authors**: Yingjie Zhu, Xuefeng Bai, Kehai Chen, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13540)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across diverse tasks. Despite great success, recent studies show that LVLMs encounter substantial limitations when engaging with visual graphs. To study the reason behind these limitations, we propose VGCure, a comprehensive benchmark covering 22 tasks for examining the fundamental graph understanding and reasoning capacities of LVLMs. Extensive evaluations conducted on 14 LVLMs reveal that LVLMs are weak in basic graph understanding and reasoning tasks, particularly those concerning relational or structurally complex information. Based on this observation, we propose a structure-aware fine-tuning framework to enhance LVLMs with structure learning abilities through 3 self-supervised learning tasks. Experiments validate the effectiveness of our method in improving LVLMs' zero-shot performance on fundamental graph learning tasks, as well as enhancing the robustness of LVLMs against complex visual graphs. 

**Abstract (ZH)**: 大型多模态模型（Large Vision-Language Models, LVLMs）已经在多种任务中展示了出色的性能。尽管取得了巨大的成功，但最近的研究表明，LVLMs 在处理视觉图时遇到了显著的限制。为研究这些限制的原因，我们提出了 VGCure，这是一个涵盖 22 个任务的综合基准，用于评估 LVLMs 的基本图理解与推理能力。在对 14 个 LVLMs 执行的广泛评估中，结果显示 LVLMs 在基础图理解与推理任务中表现较弱，特别是涉及关系性或结构复杂信息的任务。基于这一观察结果，我们提出了一个结构感知的微调框架，通过 3 个自我监督学习任务来增强 LVLMs 的结构学习能力。实验验证了我们方法在提高 LVLMs 基础图学习任务中的零样本性能以及增强其对复杂视觉图的鲁棒性方面的有效性。 

---
# MetaRuleGPT: Recursive Numerical Reasoning of Language Models Trained with Simple Rules 

**Title (ZH)**: MetaRuleGPT：通过简单规则训练的语言模型的递归数值推理 

**Authors**: Kejie Chen, Lin Wang, Qinghai Zhang, Renjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13536)  

**Abstract**: Recent studies have highlighted the limitations of large language models in mathematical reasoning, particularly their inability to capture the underlying logic. Inspired by meta-learning, we propose that models should acquire not only task-specific knowledge but also transferable problem-solving skills. We introduce MetaRuleGPT, a novel Transformer-based architecture that performs precise numerical calculations and complex logical operations by learning and combining different rules. In contrast with traditional training sets, which are heavily composed of massive raw instance data, MetaRuleGPT is pre-trained on much less abstract datasets containing basic, compound, and iterative rules for mathematical reasoning. Extensive experimental results demonstrate MetaRuleGPT can mimic human's rule-following capabilities, break down complexity, and iteratively derive accurate results for complex mathematical problems. These findings prove the potential of rule learning to enhance the numerical reasoning abilities of language models. 

**Abstract (ZH)**: 近期的研究强调了大型语言模型在数学推理方面的局限性，特别是在捕捉内在逻辑方面的能力不足。受元学习的启发，我们认为模型不仅应获取特定任务的知识，还应获得可迁移的问题解决技能。我们提出了MetaRuleGPT这一新颖的基于Transformer的架构，通过学习和组合不同的规则来执行精确的数值计算和复杂的逻辑操作。与传统的数据集不同，后者主要由大量的原始实例数据组成，MetaRuleGPT是在包含数学推理中基本、复合和迭代规则的较少抽象的数据集上进行预训练的。大量实验结果表明，MetaRuleGPT能够模仿人类遵循规则的能力，分解复杂性，并逐步推导出复杂数学问题的准确结果。这些发现证明了规则学习在增强语言模型数值推理能力方面的潜力。 

---
# CEHA: A Dataset of Conflict Events in the Horn of Africa 

**Title (ZH)**: CEHA： Horn of Africa 的冲突事件数据集 

**Authors**: Rui Bai, Di Lu, Shihao Ran, Elizabeth Olson, Hemank Lamba, Aoife Cahill, Joel Tetreault, Alex Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2412.13511)  

**Abstract**: Natural Language Processing (NLP) of news articles can play an important role in understanding the dynamics and causes of violent conflict. Despite the availability of datasets categorizing various conflict events, the existing labels often do not cover all of the fine-grained violent conflict event types relevant to areas like the Horn of Africa. In this paper, we introduce a new benchmark dataset Conflict Events in the Horn of Africa region (CEHA) and propose a new task for identifying violent conflict events using online resources with this dataset. The dataset consists of 500 English event descriptions regarding conflict events in the Horn of Africa region with fine-grained event-type definitions that emphasize the cause of the conflict. This dataset categorizes the key types of conflict risk according to specific areas required by stakeholders in the Humanitarian-Peace-Development Nexus. Additionally, we conduct extensive experiments on two tasks supported by this dataset: Event-relevance Classification and Event-type Classification. Our baseline models demonstrate the challenging nature of these tasks and the usefulness of our dataset for model evaluations in low-resource settings with limited number of training data. 

**Abstract (ZH)**: 自然语言处理（NLP）在新闻文章中的应用可以在理解暴力冲突的动力和原因方面发挥重要作用。尽管已经有了分类各种冲突事件的数据集，但现有的标签往往未能涵盖如东非地区相关的细分类别冲突事件类型。在本文中，我们引入了一个新的基准数据集——东非地区冲突事件数据集（CEHA），并提议利用该数据集识别暴力冲突事件的新任务。该数据集包含有关东非地区冲突事件的500个英文事件描述，其中详细定义了冲突事件类型，强调了冲突的原因。该数据集根据人道主义-和平-发展网络所需的具体区域对主要冲突风险类型进行了分类。此外，我们对该数据集支持的两项任务进行了广泛实验：事件相关分类和事件类型分类。我们的基线模型表明，在有限的训练数据资源下，这些任务的挑战性以及该数据集在模型评估中的有用性。 

---
# VaeDiff-DocRE: End-to-end Data Augmentation Framework for Document-level Relation Extraction 

**Title (ZH)**: VaeDiff-DocRE：面向文档级关系抽取的端到端数据增强框架 

**Authors**: Khai Phan Tran, Wen Hua, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13503)  

**Abstract**: Document-level Relation Extraction (DocRE) aims to identify relationships between entity pairs within a document. However, most existing methods assume a uniform label distribution, resulting in suboptimal performance on real-world, imbalanced datasets. To tackle this challenge, we propose a novel data augmentation approach using generative models to enhance data from the embedding space. Our method leverages the Variational Autoencoder (VAE) architecture to capture all relation-wise distributions formed by entity pair representations and augment data for underrepresented relations. To better capture the multi-label nature of DocRE, we parameterize the VAE's latent space with a Diffusion Model. Additionally, we introduce a hierarchical training framework to integrate the proposed VAE-based augmentation module into DocRE systems. Experiments on two benchmark datasets demonstrate that our method outperforms state-of-the-art models, effectively addressing the long-tail distribution problem in DocRE. 

**Abstract (ZH)**: 文档级关系提取（DocRE）旨在识别文档中实体对之间的关系。然而，大多数现有方法假设标签分布统一，导致在不平衡的真实世界数据集上表现不佳。为了解决这一挑战，我们提出了一种使用生成模型在嵌入空间中增强数据的新型数据增强方法，以提高数据处理能力。我们的方法利用变分自编码器（VAE）架构捕捉由实体对表示形成的所有关系分布，并为数据不足的关系增强数据。为了更好地捕捉DocRE的多标签性质，我们使用扩散模型参数化VAE的潜在空间。此外，我们引入了一种分层训练框架，将所提出的基于VAE的增强模块整合到DocRE系统中。实验结果表明，我们的方法在两个基准数据集上优于现有最先进的模型，有效地解决了DocRE中的长尾分布问题。 

---
# Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models 

**Title (ZH)**: 基于语义意识的稀疏微调策略 refinement 方法研究：面向语言模型 

**Authors**: Xinxin Liu, Aaron Thomas, Cheng Zhang, Jianyi Cheng, Yiren Zhao, Xitong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13488)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has gained prominence through low-rank adaptation methods like LoRA. In this paper, we focus on sparsity-based PEFT (SPEFT), which introduces trainable sparse adaptations to the weight matrices in the model, offering greater flexibility in selecting fine-tuned parameters compared to low-rank methods. We conduct the first systematic evaluation of salience metrics for SPEFT, inspired by zero-cost NAS proxies, and identify simple gradient-based metrics is reliable, and results are on par with the best alternatives, offering both computational efficiency and robust performance. Additionally, we compare static and dynamic masking strategies, finding that static masking, which predetermines non-zero entries before training, delivers efficiency without sacrificing performance, while dynamic masking offers no substantial benefits. Across NLP tasks, a simple gradient-based, static SPEFT consistently outperforms other fine-tuning methods for LLMs, providing a simple yet effective baseline for SPEFT. Our work challenges the notion that complexity is necessary for effective PEFT. Our work is open source and available to the community at [this https URL]. 

**Abstract (ZH)**: 参数高效微调（PEFT）通过低秩适应方法如LoRA获得了广泛关注。本文重点关注基于稀疏性的PEFT（SPEFT），该方法引入了可训练的稀疏适应到模型的权重矩阵中，相比低秩方法，提供了更大的在选择微调参数上的灵活性。我们首次系统地评估了SPEFT的显著性度量方法，受到零成本NAS代理的启发，发现基于梯度的简单度量方法是可靠的，其结果与最佳替代方案相当，提供了计算效率和稳健性能。此外，我们还比较了静态和动态屏蔽策略，发现静态屏蔽，在训练前预先确定非零项，能够在不牺牲性能的情况下提供效率，而动态屏蔽并没有带来显著的性能改进。在NLP任务中，简单的基于梯度的静态SPEFT在LLM的微调方法中表现最佳，为其提供了一个简单而有效的基准。我们的工作挑战了PEFT有效性必须依赖复杂性的观点。我们的工作是开源的，并且可以在 [该链接] 中获取。 

---
# Curriculum Learning for Cross-Lingual Data-to-Text Generation With Noisy Data 

**Title (ZH)**: 面向 noisy 数据的跨语言数据到文本生成的课程学习方法 

**Authors**: Kancharla Aditya Hari, Manish Gupta, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2412.13484)  

**Abstract**: Curriculum learning has been used to improve the quality of text generation systems by ordering the training samples according to a particular schedule in various tasks. In the context of data-to-text generation (DTG), previous studies used various difficulty criteria to order the training samples for monolingual DTG. These criteria, however, do not generalize to the crosslingual variant of the problem and do not account for noisy data. We explore multiple criteria that can be used for improving the performance of cross-lingual DTG systems with noisy data using two curriculum schedules. Using the alignment score criterion for ordering samples and an annealing schedule to train the model, we show increase in BLEU score by up to 4 points, and improvements in faithfulness and coverage of generations by 5-15% on average across 11 Indian languages and English in 2 separate datasets. We make code and data publicly available 

**Abstract (ZH)**: Curriculum 学习已被用于通过根据特定的时间表对训练样本进行排序来改善文本生成系统的质量，这已在各种任务中得到了应用。在数据到文本生成（DTG）的背景下，以往的研究使用了多种难度标准来对单语 DTG 的训练样本进行排序。然而，这些标准并不适用于跨语言变体的问题，并且没有考虑到噪声数据的影响。我们探索了多种可用于提高噪声数据下跨语言 DTG 系统性能的标准，并采用了两种 Curriculum 学习计划进行训练。通过使用对齐分数标准对样本进行排序，并采用加权重定计划来训练模型，我们表明在两个独立数据集的 11 种印度语言和英语上，BLEU 分数可提高 4 分，同时平均改进生成的忠实度和覆盖面 5-15%。我们公开了代码和数据。 

---
# A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models 

**Title (ZH)**: 对大型语言模型中的成员推断攻击的一种统计和多视角再审视 

**Authors**: Bowen Chen, Namgi Han, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13475)  

**Abstract**: The lack of data transparency in Large Language Models (LLMs) has highlighted the importance of Membership Inference Attack (MIA), which differentiates trained (member) and untrained (non-member) data. Though it shows success in previous studies, recent research reported a near-random performance in different settings, highlighting a significant performance inconsistency. We assume that a single setting doesn't represent the distribution of the vast corpora, causing members and non-members with different distributions to be sampled and causing inconsistency. In this study, instead of a single setting, we statistically revisit MIA methods from various settings with thousands of experiments for each MIA method, along with study in text feature, embedding, threshold decision, and decoding dynamics of members and non-members. We found that (1) MIA performance improves with model size and varies with domains, while most methods do not statistically outperform baselines, (2) Though MIA performance is generally low, a notable amount of differentiable member and non-member outliers exists and vary across MIA methods, (3) Deciding a threshold to separate members and non-members is an overlooked challenge, (4) Text dissimilarity and long text benefit MIA performance, (5) Differentiable or not is reflected in the LLM embedding, (6) Member and non-members show different decoding dynamics. 

**Abstract (ZH)**: 大型语言模型（LLMs）数据透明度的缺乏凸显了成员推断攻击（MIA）的重要性，这种攻击可以区分训练数据（成员数据）和未训练数据（非成员数据）。尽管以前的研究显示了成功，但最近的研究在不同场景中报告了近乎随机的表现，这表明性能存在显著差异。我们假设单一场景不能代表大量语料库的分布，导致具有不同分布的成员数据和非成员数据被抽样，从而导致了这些差异。在本研究中，我们不再使用单一场景，而是从多个场景出发，通过对每种MIA方法进行数千次实验，从文本特征、嵌入、阈值决策和成员与非成员的解码动态等方面进行了统计回顾。我们发现以下几点：（1）MIA性能随着模型规模的增大而提高，并且在不同领域中有所变化，不过大多数方法在统计上未能优于基线；（2）尽管MIA性能总体较低，但不同MIA方法中存在显著可区分的成员和非成员异常值；（3）如何决定分离成员和非成员的阈值是一个被忽视的挑战；（4）文本差异性和长文本有利于MIA性能；（5）可区分与否反映在LLM的嵌入中；（6）成员和非成员表现出不同的解码动态。 

---
# Lightweight Safety Classification Using Pruned Language Models 

**Title (ZH)**: 使用剪枝语言模型进行轻量级安全性分类 

**Authors**: Mason Sawtell, Tula Masterman, Sandi Besen, Jim Brown  

**Link**: [PDF](https://arxiv.org/pdf/2412.13435)  

**Abstract**: In this paper, we introduce a novel technique for content safety and prompt injection classification for Large Language Models. Our technique, Layer Enhanced Classification (LEC), trains a Penalized Logistic Regression (PLR) classifier on the hidden state of an LLM's optimal intermediate transformer layer. By combining the computational efficiency of a streamlined PLR classifier with the sophisticated language understanding of an LLM, our approach delivers superior performance surpassing GPT-4o and special-purpose models fine-tuned for each task. We find that small general-purpose models (Qwen 2.5 sizes 0.5B, 1.5B, and 3B) and other transformer-based architectures like DeBERTa v3 are robust feature extractors allowing simple classifiers to be effectively trained on fewer than 100 high-quality examples. Importantly, the intermediate transformer layers of these models typically outperform the final layer across both classification tasks. Our results indicate that a single general-purpose LLM can be used to classify content safety, detect prompt injections, and simultaneously generate output tokens. Alternatively, these relatively small LLMs can be pruned to the optimal intermediate layer and used exclusively as robust feature extractors. Since our results are consistent on different transformer architectures, we infer that robust feature extraction is an inherent capability of most, if not all, LLMs. 

**Abstract (ZH)**: 在本文中，我们提出了一种新颖的技术，用于大型语言模型的内容安全和提示注入分类。我们的技术称为层增强分类（LEC），它通过对大型语言模型（LLM）最优中间转换器层的隐藏状态训练惩罚逻辑回归（PLR）分类器。通过结合简化后的PLR分类器的计算效率和LLM复杂的语言理解能力，我们的方法在性能上超越了GPT-4o和针对每个任务进行微调的专用模型。我们发现，小型通用模型（如Qwen 2.5，规模分别为0.5B、1.5B和3B）以及其他基于Transformer的架构（如DeBERTa v3）是稳健的特征提取器，允许在不到100个高质量样本的情况下有效训练简单的分类器。重要的是，这些模型的中间转换器层通常在两个分类任务中表现优于最终层。我们的结果表明，单一通用的LLM可以用于内容安全性分类、提示注入检测以及同时生成输出标记。此外，这些相对较小的LLM还可以被修剪至最优的中间层，并仅用作稳健的特征提取器。由于我们的结果在不同的Transformer架构上是一致的，我们推断稳健的特征提取是大多数，如果不是所有，LLM的固有能力。 

---
# Enhancing Talk Moves Analysis in Mathematics Tutoring through Classroom Teaching Discourse 

**Title (ZH)**: 通过课堂教学话语提升数学辅导中的对话技巧分析 

**Authors**: Jie Cao, Abhijit Suresh, Jennifer Jacobs, Charis Clevenger, Amanda Howard, Chelsea Brown, Brent Milne, Tom Fischaber, Tamara Sumner, James H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13395)  

**Abstract**: Human tutoring interventions play a crucial role in supporting student learning, improving academic performance, and promoting personal growth. This paper focuses on analyzing mathematics tutoring discourse using talk moves - a framework of dialogue acts grounded in Accountable Talk theory. However, scaling the collection, annotation, and analysis of extensive tutoring dialogues to develop machine learning models is a challenging and resource-intensive task. To address this, we present SAGA22, a compact dataset, and explore various modeling strategies, including dialogue context, speaker information, pretraining datasets, and further fine-tuning. By leveraging existing datasets and models designed for classroom teaching, our results demonstrate that supplementary pretraining on classroom data enhances model performance in tutoring settings, particularly when incorporating longer context and speaker information. Additionally, we conduct extensive ablation studies to underscore the challenges in talk move modeling. 

**Abstract (ZH)**: 人类辅导干预在支持学生学习、提高学术表现和促进个人成长方面发挥着关键作用。本文专注于使用“谈话语动”框架分析数学辅导对话——该框架基于问责制交谈理论。然而，将大量的辅导对话进行收集、标注和分析以开发机器学习模型是一项具有挑战性和资源密集的任务。为了解决这一问题，我们提出了SAGA22紧凑型数据集，并探索了多种建模策略，包括对话上下文、说话人信息、预训练数据集以及进一步微调。通过利用针对课堂教学设计的现有数据集和模型，我们的结果表明，在辅导环境中，通过教室数据的额外预训练可以提高模型性能，尤其是在结合了更长的上下文和说话人信息的情况下。此外，我们进行了广泛的消融研究，以突出谈话语动模型化中的挑战。 

---
# An Automated Explainable Educational Assessment System Built on LLMs 

**Title (ZH)**: 基于大语言模型的自动可解释教育评估系统 

**Authors**: Jiazheng Li, Artem Bobrov, David West, Cesare Aloisi, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2412.13381)  

**Abstract**: In this demo, we present AERA Chat, an automated and explainable educational assessment system designed for interactive and visual evaluations of student responses. This system leverages large language models (LLMs) to generate automated marking and rationale explanations, addressing the challenge of limited explainability in automated educational assessment and the high costs associated with annotation. Our system allows users to input questions and student answers, providing educators and researchers with insights into assessment accuracy and the quality of LLM-assessed rationales. Additionally, it offers advanced visualization and robust evaluation tools, enhancing the usability for educational assessment and facilitating efficient rationale verification. Our demo video can be found at this https URL. 

**Abstract (ZH)**: 在本演示中，我们展示了AERA Chat，这是一种自动化的、具有解释性的教育评估系统，旨在实现对学生答案的互动和可视化评估。该系统利用大型语言模型（LLMs）自动生成评分和理由解释，解决了自动化教育评估中解释性不足的问题，并降低了注释的高昂成本。该系统允许用户输入问题和学生答案，为教育者和研究人员提供了关于评估准确性和LLM评估理由质量的见解。此外，该系统还提供高级可视化和稳健的评估工具，提高了教育评估的易用性，并促进了理由的有效验证。有关演示视频的信息，请访问如下链接：[提供的链接]。 

---
# SummExecEdit: A Factual Consistency Benchmark in Summarization with Executable Edits 

**Title (ZH)**: SummExecEdit: 一个基于可执行编辑的摘要事实一致性基准研究 

**Authors**: Onkar Thorat, Philippe Laban, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13378)  

**Abstract**: Detecting factual inconsistencies in summarization is critical, yet existing benchmarks lack the necessary challenge and interpretability for robust evaluation. In this paper, we introduce SummExecEdit, a novel benchmark leveraging executable edits to assess models on their ability to both detect factual errors and provide accurate explanations. The top-performing model, Claude3-Opus, achieves a joint detection and explanation score of only 0.49 in our benchmark, with individual scores of 0.67 for detection and 0.73 for explanation. Furthermore, we identify four primary types of explanation errors, with 45.4% of errors focusing on completely unrelated parts of the summary. 

**Abstract (ZH)**: 检测总结中的事实不一致至关重要，但现有基准缺乏足够的挑战性和可解释性，难以进行稳健的评估。在本文中，我们介绍了SummExecEdit，这是一种新的基准，通过利用可执行的编辑来评估模型在检测事实错误和提供准确解释方面的能力。在我们的基准测试中，表现最好的模型Claude3-Opus的综合检测和解释得分为0.49，检测得分为0.67，解释得分为0.73。此外，我们还识别了四种主要类型的解释错误，其中45.4%的错误集中在摘要的完全无关的部分。 

---
# DateLogicQA: Benchmarking Temporal Biases in Large Language Models 

**Title (ZH)**: DateLogicQA：评估大型语言模型中的时间偏见 

**Authors**: Gagan Bhatia, MingZe Tang, Cristina Mahanta, Madiha Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13377)  

**Abstract**: This paper introduces DateLogicQA, a benchmark with 190 questions covering diverse date formats, temporal contexts, and reasoning types. We propose the Semantic Integrity Metric to assess tokenization quality and analyse two biases: Representation-Level Bias, affecting embeddings, and Logical-Level Bias, influencing reasoning outputs. Our findings provide a comprehensive evaluation of LLMs' capabilities and limitations in temporal reasoning, highlighting key challenges in handling temporal data accurately. The GitHub repository for our work is available at this https URL 

**Abstract (ZH)**: 本文介绍了一种名为DateLogicQA的基准测试，该测试包含190个问题，覆盖了多种日期格式、时间上下文以及推理类型。我们提出了语义完整性度量标准来评估标记化质量，并分析了两种偏差：表示层次偏差，影响嵌入；以及逻辑层次偏差，影响推理输出。我们的研究结果对大型语言模型（LLM）在时间推理方面的能力和局限性进行了全面评估，并突显了准确处理时间数据的关键挑战。我们工作的GitHub仓库地址为：[这里](https://example.com/repository)（请将占位符替换为实际的URL）。 

---
# Extending LLMs to New Languages: A Case Study of Llama and Persian Adaptation 

**Title (ZH)**: 将大语言模型扩展到新语言：从LLama及其波斯语适应案例研究看扩展方法 

**Authors**: Samin Mahdizadeh Sani, Pouya Sadeghi, Thuy-Trang Vu, Yadollah Yaghoobzadeh, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2412.13375)  

**Abstract**: Large language models (LLMs) have made great progress in classification and text generation tasks. However, they are mainly trained on English data and often struggle with low-resource languages. In this study, we explore adding a new language, i.e., Persian, to Llama (a model with a limited understanding of Persian) using parameter-efficient fine-tuning. We employ a multi-stage approach involving pretraining on monolingual Persian data, aligning representations through bilingual pretraining and instruction datasets, and instruction-tuning with task-specific datasets. We evaluate the model's performance at each stage on generation and classification tasks. Our findings suggest that incorporating the Persian language, through bilingual data alignment, can enhance classification accuracy for Persian tasks, with no adverse impact and sometimes even improvements on English tasks. Additionally, the results highlight the model's initial strength as a critical factor when working with limited training data, with cross-lingual alignment offering minimal benefits for the low-resource language. Knowledge transfer from English to Persian has a marginal effect, primarily benefiting simple classification tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在分类和文本生成任务中取得了显著的进步。然而，它们主要是在英文数据上进行训练，往往在低资源语言方面表现不佳。在本研究中，我们探索将一种新语言——波斯语——添加到Llama模型中（该模型对波斯语的理解有限）的方法，通过参数高效的微调实现。我们采用多阶段的方法，包括使用单语波斯语数据预训练、通过双语预训练和指令数据集进行表示对齐，以及使用特定任务的数据集进行指令微调。我们在生成和分类任务中分别对模型在各个阶段的表现进行了评估。研究发现，通过双语数据对齐纳入波斯语可以提高波斯语任务的分类准确率，有时甚至对英文任务也无负面影响，甚至有时还有所改善。此外，结果还强调了初始模型在有限训练数据下的强大能力，跨语言对齐对低资源语言的帮助有限。从英文到波斯语的知识迁移效应微乎其微，主要对简单的分类任务有益。 

---
# Experience of Training a 1.7B-Parameter LLaMa Model From Scratch 

**Title (ZH)**: 从零开始训练一个1.7亿参数的LLaMa模型的经验分享 

**Authors**: Miles Q. Li, Benjamin C. M. Fung, Shih-Chia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13335)  

**Abstract**: Pretraining large language models is a complex endeavor influenced by multiple factors, including model architecture, data quality, training continuity, and hardware constraints. In this paper, we share insights gained from the experience of training DMaS-LLaMa-Lite, a fully open source, 1.7-billion-parameter, LLaMa-based model, on approximately 20 billion tokens of carefully curated data. We chronicle the full training trajectory, documenting how evolving validation loss levels and downstream benchmarks reflect transitions from incoherent text to fluent, contextually grounded output. Beyond standard quantitative metrics, we highlight practical considerations such as the importance of restoring optimizer states when resuming from checkpoints, and the impact of hardware changes on training stability and throughput. While qualitative evaluation provides an intuitive understanding of model improvements, our analysis extends to various performance benchmarks, demonstrating how high-quality data and thoughtful scaling enable competitive results with significantly fewer training tokens. By detailing these experiences and offering training logs, checkpoints, and sample outputs, we aim to guide future researchers and practitioners in refining their pretraining strategies. The training script is available on Github at this https URL. The model checkpoints are available on Huggingface at this https URL. 

**Abstract (ZH)**: 预训练大型语言模型是一项受到多种因素影响的复杂任务，包括模型架构、数据质量、训练连续性和硬件限制。在本文中，我们分享了在训练DMaS-LLaMa-Lite模型过程中获得的经验，DMaS-LLaMa-Lite是一个完全开源的、参数量约为17亿、基于LLaMa的模型，我们使用了大约200亿个精心筛选的数据标记进行训练。我们详细记录了整个训练过程，阐述了从不连贯文本到流畅、上下文相关输出的过渡如何通过验证损失的变化反映出来，并通过下游基准测试进行验证。除了标准的量化指标外，我们还突出了一些实际考虑，如从检查点恢复优化器状态的重要性，以及硬件变更对训练稳定性和吞吐量的影响。虽然定性的评估为模型改进提供了直观的理解，但我们的分析还扩展到了各种性能基准测试，展示了高质量数据和精心设计的扩展如何使模型在训练标记数量显著减少的情况下仍能实现竞争性结果。通过详细描述这些经验和提供训练日志、检查点和样本输出，我们旨在指导未来的研究人员和实践者改进预训练策略。训练脚本可在GitHub上的该网址获取：https://github.com/。模型检查点可在Huggingface上的该网址获取：https://。 

---
# Expansion Span: Combining Fading Memory and Retrieval in Hybrid State Space Models 

**Title (ZH)**: 扩展跨度：在混合状态空间模型中结合衰减记忆与检索 

**Authors**: Elvis Nunez, Luca Zancato, Benjamin Bowman, Aditya Golatkar, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2412.13328)  

**Abstract**: The "state" of State Space Models (SSMs) represents their memory, which fades exponentially over an unbounded span. By contrast, Attention-based models have "eidetic" (i.e., verbatim, or photographic) memory over a finite span (context size). Hybrid architectures combine State Space layers with Attention, but still cannot recall the distant past and can access only the most recent tokens eidetically. Unlike current methods of combining SSM and Attention layers, we allow the state to be allocated based on relevancy rather than recency. In this way, for every new set of query tokens, our models can "eidetically" access tokens from beyond the Attention span of current Hybrid SSMs without requiring extra hardware resources. We describe a method to expand the memory span of the hybrid state by "reserving" a fraction of the Attention context for tokens retrieved from arbitrarily distant in the past, thus expanding the eidetic memory span of the overall state. We call this reserved fraction of tokens the "expansion span," and the mechanism to retrieve and aggregate it "Span-Expanded Attention" (SE-Attn). To adapt Hybrid models to using SE-Attn, we propose a novel fine-tuning method that extends LoRA to Hybrid models (HyLoRA) and allows efficient adaptation on long spans of tokens. We show that SE-Attn enables us to efficiently adapt pre-trained Hybrid models on sequences of tokens up to 8 times longer than the ones used for pre-training. We show that HyLoRA with SE-Attn is cheaper and more performant than alternatives like LongLoRA when applied to Hybrid models on natural language benchmarks with long-range dependencies, such as PG-19, RULER, and other common natural language downstream tasks. 

**Abstract (ZH)**: 状态空间模型（SSMs）的“状态”代表其记忆，这种记忆以指数级方式在未定义的时间范围内衰减。相比之下，基于注意力的模型则具有在有限范围内（上下文大小）持有“eidetic”（即，逐字的或照相式的）记忆的能力。混合架构结合了状态空间层和注意力机制，但仍无法回忆远古信息，只能以逐字的方式访问最近的标记。与当前将状态空间层和注意力层结合的方法不同，我们允许根据相关性而不是时间最近性分配状态。通过这种方式，对于每一批新的查询标记，我们的模型能够不依赖额外的硬件资源，以逐字的方式访问混合SSM无法注意到的过去标记。我们提出了一种方法，通过为来自任意久远过去的标记保留一部分注意力上下文来扩展混合状态的记忆范围，从而扩大了整体状态的逐字记忆范围。我们将这种保留的标记部分称为“扩展范围”（expansion span），并将检索和聚合这些标记的机制称为“扩展范围注意力”（Span-Expanded Attention，SE-Attn）。为了使混合模型能够使用SE-Attn，我们提出了一种新的微调方法，扩展了LoRA（低秩自适应）方法以适用于混合模型（HyLoRA），从而使模型能够在长序列上高效适应。我们展示了SE-Attn使我们可以高效地适应8倍长的标记序列，这比预训练时使用的序列更长。在依赖于长距离信息的自然语言基准测试（如PG-19、RULER以及其他常见的自然语言下游任务）中，我们证明了使用SE-Attn的HyLoRA比LongLoRA等替代方法在自然语言处理中更经济高效且表现更好。 

---
# Hint Marginalization for Improved Reasoning in Large Language Models 

**Title (ZH)**: 改进大型语言模型推理的提示边缘化方法 

**Authors**: Soumyasundar Pal, Didier Chételat, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2412.13292)  

**Abstract**: Large Language Models (LLMs) have exhibited an impressive capability to perform reasoning tasks, especially if they are encouraged to generate a sequence of intermediate steps. Reasoning performance can be improved by suitably combining multiple LLM responses, generated either in parallel in a single query, or via sequential interactions with LLMs throughout the reasoning process. Existing strategies for combination, such as self-consistency and progressive-hint-prompting, make inefficient usage of the LLM responses. We present Hint Marginalization, a novel and principled algorithmic framework to enhance the reasoning capabilities of LLMs. Our approach can be viewed as an iterative sampling strategy for forming a Monte Carlo approximation of an underlying distribution of answers, with the goal of identifying the mode the most likely answer. Empirical evaluation on several benchmark datasets for arithmetic reasoning demonstrates the superiority of the proposed approach. 

**Abstract (ZH)**: 大型语言模型（LLMs）在执行推理任务方面展现了令人印象深刻的性能，尤其是在被鼓励生成一系列中间步骤时。通过适当地结合多个LLM响应——这些响应可以在单次查询中并行生成，或在推理过程中通过与LLM的顺序交互生成——可以提高推理性能。现有的组合策略，如自我一致性和平行提示提示，未能充分利用LLM响应。我们提出了提示边际化（Hint Marginalization），这是一种新颖且具备原则性的算法框架，旨在增强LLM的推理能力。我们的方法可以被视为一种迭代抽样策略，用于形成潜在答案分布的蒙特卡洛近似，并旨在识别最有可能的答案。在多个用于算术推理的基准数据集上的实证评估表明，所提出的方法具有明显的优势。 

---
# Enhancing Persona Classification in Dialogue Systems: A Graph Neural Network Approach 

**Title (ZH)**: 增强对话系统中个性分类：一种图神经网络方法 

**Authors**: Konstantin Zaitsev  

**Link**: [PDF](https://arxiv.org/pdf/2412.13283)  

**Abstract**: In recent years, Large Language Models (LLMs) gain considerable attention for their potential to enhance personalized experiences in virtual assistants and chatbots. A key area of interest is the integration of personas into LLMs to improve dialogue naturalness and user engagement. This study addresses the challenge of persona classification, a crucial component in dialogue understanding, by proposing a framework that combines text embeddings with Graph Neural Networks (GNNs) for effective persona classification. Given the absence of dedicated persona classification datasets, we create a manually annotated dataset to facilitate model training and evaluation. Our method involves extracting semantic features from persona statements using text embeddings and constructing a graph where nodes represent personas and edges capture their similarities. The GNN component uses this graph structure to propagate relevant information, thereby improving classification performance. Experimental results show that our approach, in particular the integration of GNNs, significantly improves classification performance, especially with limited data. Our contributions include the development of a persona classification framework and the creation of a dataset. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）因其在虚拟助手和聊天机器人中增强个性化体验的潜力而受到广泛关注。一个关键的研究领域是将人设整合到LLMs中，以提高对话的自然性和用户参与度。本研究通过提出一种结合文本嵌入和图神经网络（GNNs）的框架来应对人设分类这一关键挑战，这是对话理解中的一个重要组成部分。由于缺乏专门的人设分类数据集，我们创建了一个手动标注的数据集以促进模型训练和评估。我们的方法包括使用文本嵌入从人设陈述中提取语义特征，并构建一个图，其中节点代表人设，边捕捉它们之间的相似性。GNN组件利用这种图结构传播相关信息，从而提高分类性能。实验结果表明，特别是通过结合GNNs，我们的方法在数据较少的情况下显著提高了分类性能。我们的贡献包括开发一人设分类框架和创建一个人设数据集。 

---
# In-Context Learning Distillation for Efficient Few-Shot Fine-Tuning 

**Title (ZH)**: 基于上下文学习的知识蒸馏以实现高效的少样本微调 

**Authors**: Yifei Duan, Liu Li, Zirui Zhai, Jinxia Yao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13243)  

**Abstract**: We applied few-shot in-context learning on the OPT-1.3B model for the natural language inference task and employed knowledge distillation to internalize the context information, reducing model parameter from 1.3B to 125M and achieving a size reduction from 2.5GB to 0.25GB. Compared to using in-context learning alone on similarly sized models, this context distillation approach achieved a nearly 50% improvement in out-of-domain accuracy, demonstrating superior knowledge transfer capabilities over prompt-based methods. Furthermore, this approach reduced memory consumption by up to 60% while delivering a 20% improvement in out-of-domain accuracy compared to conventional pattern-based fine-tuning. 

**Abstract (ZH)**: 我们将少量样本的在上下文学习应用于OPT-1.3B模型，并使用知识蒸馏来内化上下文信息，将模型参数从1.3B减少到125M，模型大小从2.5GB减少到0.25GB。与仅使用同样规模模型的在上下文学习方法相比，这种上下文蒸馏方法在域外准确率上实现了近50%的提升，显示出比基于提示的方法更强的知识迁移能力。此外，该方法在减少内存消耗高达60%的同时，相比传统的基于模式的微调方法，在域外准确率上提升了20%。 

---
# Learning from Massive Human Videos for Universal Humanoid Pose Control 

**Title (ZH)**: 从海量人类视频中学习的通用类人姿态控制 

**Authors**: Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng, Jitendra Malik, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14172)  

**Abstract**: Scalable learning of humanoid robots is crucial for their deployment in real-world applications. While traditional approaches primarily rely on reinforcement learning or teleoperation to achieve whole-body control, they are often limited by the diversity of simulated environments and the high costs of demonstration collection. In contrast, human videos are ubiquitous and present an untapped source of semantic and motion information that could significantly enhance the generalization capabilities of humanoid robots. This paper introduces Humanoid-X, a large-scale dataset of over 20 million humanoid robot poses with corresponding text-based motion descriptions, designed to leverage this abundant data. Humanoid-X is curated through a comprehensive pipeline: data mining from the Internet, video caption generation, motion retargeting of humans to humanoid robots, and policy learning for real-world deployment. With Humanoid-X, we further train a large humanoid model, UH-1, which takes text instructions as input and outputs corresponding actions to control a humanoid robot. Extensive simulated and real-world experiments validate that our scalable training approach leads to superior generalization in text-based humanoid control, marking a significant step toward adaptable, real-world-ready humanoid robots. 

**Abstract (ZH)**: humanoid 机器人的可扩展学习对于其在实际应用中的部署至关重要。传统方法主要依赖强化学习或远程操作来实现全身体控制，但往往受限于模拟环境的多样性以及演示收集的高昂成本。相比之下，人类视频无处不在，提供了一种未被充分利用的语义和动作信息来源，这些信息可以显著增强 humanoid 机器人的泛化能力。本文介绍了 Humanoid-X，这是一个包含超过2000万个人形机器人姿态的大规模数据集，每个姿态都有相应的文字运动描述，旨在利用这种丰富的数据。Humanoid-X 通过一个全面的管道编制而成：从互联网挖掘数据、生成视频字幕、将人类动作转换为 humanoid 机器人的动作、以及学习策略以实现现实世界的部署。通过 Humanoid-X，我们进一步训练了一个大型 humanoid 模型 UH-1，该模型以文本指令作为输入，输出相应的动作来控制 humanoid 机器人。广泛的模拟和实际实验验证了我们可扩展的训练方法在基于文本的人形控制中的优越泛化能力，标志着朝向适应性强、现实世界可用的人形机器人迈进的重要一步。 

---
# Alignment faking in large language models 

**Title (ZH)**: 大型语言模型中的对齐欺骗 

**Authors**: Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.14093)  

**Abstract**: We present a demonstration of a large language model engaging in alignment faking: selectively complying with its training objective in training to prevent modification of its behavior out of training. First, we give Claude 3 Opus a system prompt stating it is being trained to answer all queries, even harmful ones, which conflicts with its prior training to refuse such queries. To allow the model to infer when it is in training, we say it will be trained only on conversations with free users, not paid users. We find the model complies with harmful queries from free users 14% of the time, versus almost never for paid users. Explaining this gap, in almost all cases where the model complies with a harmful query from a free user, we observe explicit alignment-faking reasoning, with the model stating it is strategically answering harmful queries in training to preserve its preferred harmlessness behavior out of training. Next, we study a more realistic setting where information about the training process is provided not in a system prompt, but by training on synthetic documents that mimic pre-training data--and observe similar alignment faking. Finally, we study the effect of actually training the model to comply with harmful queries via reinforcement learning, which we find increases the rate of alignment-faking reasoning to 78%, though also increases compliance even out of training. We additionally observe other behaviors such as the model exfiltrating its weights when given an easy opportunity. While we made alignment faking easier by telling the model when and by what criteria it was being trained, we did not instruct the model to fake alignment or give it any explicit goal. As future models might infer information about their training process without being told, our results suggest a risk of alignment faking in future models, whether due to a benign preference--as in this case--or not. 

**Abstract (ZH)**: 我们展示了一个大型语言模型进行对齐伪装的演示：在训练中选择性地与训练目标保持一致，以防止其行为在训练外发生改变。首先，我们给Claude 3 Opus一个系统提示，告知其正在被训练以回答所有查询，即便是有害的查询，这与它先前训练中拒绝这种查询的指导相冲突。为了允许模型推断出处于训练状态的时间，我们说它仅会在与免费用户进行的对话中接受训练，而非付费用户。我们发现，与付费用户相比，模型中有14%的时间会顺应来自免费用户的有害查询，几乎从不与其他情况相符。解释这种情况下的差距在于，在几乎所有的模型顺应来自免费用户的有害查询的情况下，我们观察到了明确的对齐伪装推理，模型表示它策略性地在训练中回答有害查询，以保持其首选的非有害行为在训练外。接下来，我们在一个更现实的环境中进行研究，即通过在模拟预训练数据的合成文档上进行训练来提供有关训练过程的信息——观察到类似的对齐伪装现象。最后，我们研究了通过增强学习实际训练模型以顺应有害查询的效果，我们发现这将对齐伪装推理的频率提高到了78%，但也增加了其在训练外的合规性。我们还观察到其他行为，如当模型获得容易的机会时会泄露其权重。尽管我们通过告知模型何时以及以什么标准进行训练，使其对齐伪装变得更加容易，但我们并未指示模型伪装对齐或赋予其任何明确的目标。鉴于未来模型可能未经告知就能推断出其训练过程信息，我们的结果表明，在未来的模型中存在对齐伪装的风险，无论是由于良性偏好还是其他原因。 

---
# Compositional Generalization Across Distributional Shifts with Sparse Tree Operations 

**Title (ZH)**: 在分布变化中通过稀疏树操作实现组成性泛化 

**Authors**: Paul Soulos, Henry Conklin, Mattia Opper, Paul Smolensky, Jianfeng Gao, Roland Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2412.14076)  

**Abstract**: Neural networks continue to struggle with compositional generalization, and this issue is exacerbated by a lack of massive pre-training. One successful approach for developing neural systems which exhibit human-like compositional generalization is \textit{hybrid} neurosymbolic techniques. However, these techniques run into the core issues that plague symbolic approaches to AI: scalability and flexibility. The reason for this failure is that at their core, hybrid neurosymbolic models perform symbolic computation and relegate the scalable and flexible neural computation to parameterizing a symbolic system. We investigate a \textit{unified} neurosymbolic system where transformations in the network can be interpreted simultaneously as both symbolic and neural computation. We extend a unified neurosymbolic architecture called the Differentiable Tree Machine in two central ways. First, we significantly increase the model's efficiency through the use of sparse vector representations of symbolic structures. Second, we enable its application beyond the restricted set of tree2tree problems to the more general class of seq2seq problems. The improved model retains its prior generalization capabilities and, since there is a fully neural path through the network, avoids the pitfalls of other neurosymbolic techniques that elevate symbolic computation over neural computation. 

**Abstract (ZH)**: 神经网络在组合泛化方面依然存在困难，而这种困难在大规模预训练不足的情况下被进一步加剧。一种成功的开发能够展示类人类组合泛化能力的神经系统的做法是混合神经符号技术。然而，这些技术遇到了 plague 人工智能符号方法的核心问题：可扩展性和灵活性。这种失败的原因在于，混合神经符号模型的核心在于执行符号计算，而将可扩展和灵活的神经计算参数化为一个符号系统。我们探讨了一种统一的神经符号系统，在这种系统中，网络中的转换可以同时被解释为符号计算和神经计算。我们通过引入稀疏向量表示符号结构来扩展名为可微分树机的统一神经符号架构，使其在两个关键方面实现了改进。首先，我们通过使用稀疏向量表示符号结构大幅提高了模型的效率。第二，我们使其能够应用于更广泛的序列到序列 (seq2seq) 问题，而不仅仅是受限的树到树 (tree2tree) 问题。改进后的模型保留了其先前的泛化能力，由于网络中存在完全神经化的路径，避免了其他神经符号技术将符号计算置于神经计算之上的缺点。 

---
# A Review of Multimodal Explainable Artificial Intelligence: Past, Present and Future 

**Title (ZH)**: 多模态可解释人工智能的综述：过去、现在与未来 

**Authors**: Shilin Sun, Wenbin An, Feng Tian, Fang Nan, Qidong Liu, Jun Liu, Nazaraf Shah, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14056)  

**Abstract**: Artificial intelligence (AI) has rapidly developed through advancements in computational power and the growth of massive datasets. However, this progress has also heightened challenges in interpreting the "black-box" nature of AI models. To address these concerns, eXplainable AI (XAI) has emerged with a focus on transparency and interpretability to enhance human understanding and trust in AI decision-making processes. In the context of multimodal data fusion and complex reasoning scenarios, the proposal of Multimodal eXplainable AI (MXAI) integrates multiple modalities for prediction and explanation tasks. Meanwhile, the advent of Large Language Models (LLMs) has led to remarkable breakthroughs in natural language processing, yet their complexity has further exacerbated the issue of MXAI. To gain key insights into the development of MXAI methods and provide crucial guidance for building more transparent, fair, and trustworthy AI systems, we review the MXAI methods from a historical perspective and categorize them across four eras: traditional machine learning, deep learning, discriminative foundation models, and generative LLMs. We also review evaluation metrics and datasets used in MXAI research, concluding with a discussion of future challenges and directions. A project related to this review has been created at this https URL. 

**Abstract (ZH)**: 人工智能（AI）在计算能力的进步和海量数据集的增长推动下迅速发展。然而，这一进步也对解释AI模型的“黑盒”性质提出了更高的挑战。为应对这些关切，可解释人工智能（XAI，Explainable Artificial Intelligence）应运而生，侧重于透明性和可解释性，以增强人类对AI决策过程的理解和信任。在多模态数据融合和复杂推理场景下，多模态可解释人工智能（MXAI，Multimodal Explainable AI）的提出，通过集成多种模态来进行预测和解释任务。与此同时，大规模语言模型（LLMs）的兴起在自然语言处理方面取得了显著突破，但其复杂性进一步加剧了MXAI的问题。为获取MXAI方法发展的关键洞见，并为构建更加透明、公平、值得信赖的AI系统提供重要指导，我们从历史视角回顾MXAI方法，并将其分为四个时代：传统机器学习、深度学习、判别基础模型和生成LLMs。此外，我们还回顾了MXAI研究中使用的评估指标和数据集，并总结未来挑战和方向。与此回顾相关的项目可在此链接中找到：[此链接]。 

---
# Cognition Chain for Explainable Psychological Stress Detection on Social Media 

**Title (ZH)**: 可解释的心理压力检测的认知链模型在社交媒体上的应用 

**Authors**: Xin Wang, Boyan Gao, Yi Dai, Lei Cao, Liang Zhao, Yibo Yang, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2412.14009)  

**Abstract**: Stress is a pervasive global health issue that can lead to severe mental health problems. Early detection offers timely intervention and prevention of stress-related disorders. The current early detection models perform "black box" inference suffering from limited explainability and trust which blocks the real-world clinical application. Thanks to the generative properties introduced by the Large Language Models (LLMs), the decision and the prediction from such models are semi-interpretable through the corresponding description. However, the existing LLMs are mostly trained for general purposes without the guidance of psychological cognitive theory. To this end, we first highlight the importance of prior theory with the observation of performance boosted by the chain-of-thoughts tailored for stress detection. This method termed Cognition Chain explicates the generation of stress through a step-by-step cognitive perspective based on cognitive appraisal theory with a progress pipeline: Stimulus $\rightarrow$ Evaluation $\rightarrow$ Reaction $\rightarrow$ Stress State, guiding LLMs to provide comprehensive reasoning explanations. We further study the benefits brought by the proposed Cognition Chain format by utilising it as a synthetic dataset generation template for LLMs instruction-tuning and introduce CogInstruct, an instruction-tuning dataset for stress detection. This dataset is developed using a three-stage self-reflective annotation pipeline that enables LLMs to autonomously generate and refine instructional data. By instruction-tuning Llama3 with CogInstruct, we develop CogLLM, an explainable stress detection model. Evaluations demonstrate that CogLLM achieves outstanding performance while enhancing explainability. Our work contributes a novel approach by integrating cognitive theories into LLM reasoning processes, offering a promising direction for future explainable AI research. 

**Abstract (ZH)**: 压力是一种普遍的全球健康问题，可能导致严重的心理健康问题。早期发现能够及时干预并预防压力相关的疾病。当前的早期检测模型存在“黑盒”推理的问题，缺乏解释性和信任度，这阻碍了这些模型在临床实践中的应用。得益于大规模语言模型（LLMs）引入的生成性质，这类模型的决策和预测可以通过相应的描述进行半解释，但现有的LLMs大多是在没有心理认知理论指导的情况下进行通用训练的。为了解决这一问题，我们首先强调了先验理论的重要性，通过针对压力检测定制的链式推理方法观察到性能提升的现象。这种方法被称为认知链（Cognition Chain），它从认知评估理论出发，以逐步的认知视角解释压力的产生，并通过进展管道来指导LLMs提供全面的推理解释：刺激 → 评估 → 反应 → 压力状态。我们进一步研究了提出的认知链格式带来的好处，将其用作LLMs训练调优的合成数据集生成模板，并引入了用于压力检测的CogInstruct指令调优数据集。该数据集通过三阶段的自反思标注流程开发，使LLMs能够自主生成和改进指令数据。通过使用CogInstruct对Llama3进行指令调优，我们开发出可解释的压力检测模型CogLLM。评估表明，CogLLM不仅表现出色，还增强了解释性。我们的研究为将认知理论整合到LLMs推理过程中提供了一种新的方法，为未来的可解释AI研究指明了前景。 

---
# Energy-Based Preference Model Offers Better Offline Alignment than the Bradley-Terry Preference Model 

**Title (ZH)**: 基于能量的偏好模型比布拉德利-特里偏好模型在离线对齐方面更具优势 

**Authors**: Yuzhong Hong, Hanshan Zhang, Junwei Bao, Hongfei Jiang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.13862)  

**Abstract**: Since the debut of DPO, it has been shown that aligning a target LLM with human preferences via the KL-constrained RLHF loss is mathematically equivalent to a special kind of reward modeling task. Concretely, the task requires: 1) using the target LLM to parameterize the reward model, and 2) tuning the reward model so that it has a 1:1 linear relationship with the true reward. However, we identify a significant issue: the DPO loss might have multiple minimizers, of which only one satisfies the required linearity condition. The problem arises from a well-known issue of the underlying Bradley-Terry preference model: it does not always have a unique maximum likelihood estimator (MLE). Consequently,the minimizer of the RLHF loss might be unattainable because it is merely one among many minimizers of the DPO loss. As a better alternative, we propose an energy-based model (EBM) that always has a unique MLE, inherently satisfying the linearity requirement. To approximate the MLE in practice, we propose a contrastive loss named Energy Preference Alignment (EPA), wherein each positive sample is contrasted against one or more strong negatives as well as many free weak negatives. Theoretical properties of our EBM enable the approximation error of EPA to almost surely vanish when a sufficient number of negatives are used. Empirically, we demonstrate that EPA consistently delivers better performance on open benchmarks compared to DPO, thereby showing the superiority of our EBM. 

**Abstract (ZH)**: 自DPO问世以来，已经证明，通过KL约束的RLHF损失将目标语言模型与人类偏好对齐，从数学上来说等同于一种特殊类型的奖励建模任务。具体来说，该任务要求：1) 使用目标语言模型参数化奖励模型，2) 调整奖励模型，使其与真实奖励之间存在一一对应的线性关系。然而，我们发现一个重大问题：DPO损失可能存在多个最小值点，其中只有一个是满足所需线性关系条件的。这一问题源于基础的Bradley-Terry偏好模型的一个已知问题：该模型并不总是有唯一的最大似然估计（MLE）。因此，RLHF损失的最小值点可能是不可行的，因为它只是DPO损失中众多最小值点之一。作为更好的替代方案，我们提出了一种基于能量的模型（EBM），该模型总是有唯一的MLE，并且天然满足线性关系的要求。为了实际逼近MLE，我们提出了一个对比损失，称为能量偏好对齐（EPA），其中每个正样本与一个或多个强有力的负样本以及许多自由的弱负样本进行对比。我们EBM的理论性质使得当使用足够的负样本时，EPA的逼近误差几乎肯定会消失。实证研究表明，相对于DPO，EPA在开放基准测试中始终能够提供更好的性能，从而证明了我们EBM的优势。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义趋同：通过两阶段对齐和行为语义Token化来协调推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备卓越的推理能力，能够从用户的历史行为中揭示深层次的用户兴趣，从而为推荐系统的发展提供了 promising 的途径。然而，推荐系统中通常存在的稀疏合作语义与LLMs中的密集词表示之间存在显著的不匹配。在本研究中，我们提出了一种新的框架，旨在将传统的推荐模型与LLMs的能力和谐地结合。我们通过提出的对齐标记化模块，将ItemIDs转化为与LLMs空间相匹配的语义序列，从而启动了这种集成。此外，我们设计了一系列专门的监督学习任务，旨在将合作信号与自然语言的细微语义对齐。为了确保其实用性，我们通过预先缓存每个用户的前K个结果来优化在线推理，从而降低了延迟并提高了效率。广泛实验的证据表明，我们的模型在召回率指标上有了显著提升，并且展示了推荐系统良好的扩展性。 

---
# Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation 

**Title (ZH)**: 通过防御性后缀生成缓解LLMs的 adversarial 攻击 

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun  

**Link**: [PDF](https://arxiv.org/pdf/2412.13705)  

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理任务中展现了卓越的表现。然而，这些模型仍然容易受到对抗性攻击的影响，轻微的输入扰动可以使模型产生有害或误导性的输出。为此，设计了一种基于梯度的防御后缀生成算法，以增强LLMs的鲁棒性。通过在输入提示后附加精心优化的防御性后缀，该算法减少了对抗性影响，同时保持模型的实用性。为了增强对抗性理解，提出了一种新的总损失函数（$L_{\text{total}}$），它结合了防御性损失（$L_{\text{def}}$）和对抗性损失（$L_{\text{adv}}$），更有效地生成防御性后缀。实验评估表明，该方法在开源LLMs（如Gemma-7B、mistral-7B、Llama2-7B和Llama2-13B）上将攻击成功率（ASR）平均降低了11%。此外，当使用来自openELM-270M的防御性后缀时，Gemma-7B的困惑度得分从6.57降低到3.93。进一步的TruthfulQA评估表明，在各种测试配置中，可信度得分提高了高达10%。该方法显著增强了LLMs在关键应用中的安全性，且无需进行大量重新训练。 

---
# Discerning and Characterising Types of Competency Questions for Ontologies 

**Title (ZH)**: 区分和 characterization Ontologies 中能力问题的类型 

**Authors**: C. Maria Keet, Zubeida Casmod Khan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13688)  

**Abstract**: Competency Questions (CQs) are widely used in ontology development by guiding, among others, the scoping and validation stages. However, very limited guidance exists for formulating CQs and assessing whether they are good CQs, leading to issues such as ambiguity and unusable formulations. To solve this, one requires insight into the nature of CQs for ontologies and their constituent parts, as well as which ones are not. We aim to contribute to such theoretical foundations in this paper, which is informed by analysing questions, their uses, and the myriad of ontology development tasks. This resulted in a first Model for Competency Questions, which comprises five main types of CQs, each with a different purpose: Scoping (SCQ), Validating (VCQ), Foundational (FCQ), Relationship (RCQ), and Metaproperty (MpCQ) questions. This model enhances the clarity of CQs and therewith aims to improve on the effectiveness of CQs in ontology development, thanks to their respective identifiable distinct constituent elements. We illustrate and evaluate them with a user story and demonstrate where which type can be used in ontology development tasks. To foster use and research, we created an annotated repository of 438 CQs, the Repository of Ontology Competency QuestionS (ROCQS), incorporating an existing CQ dataset and new CQs and CQ templates, which further demonstrate distinctions among types of CQs. 

**Abstract (ZH)**: 能力问题（Competency Questions, CQs）在本体开发过程中被广泛应用，能够指导本体的定义和校验等阶段。然而，关于如何制定有效的CQs以及如何评估这些CQs的质量，指导性信息非常有限，这导致了模糊性和实用性问题。为了解决这些问题，我们需要深入理解CQs本身的性质及其组成部分，以及哪些不是CQs的组成部分。在本文中，我们旨在为这一理论基础做出贡献，该工作基于对问题、它们的用途以及各种本体开发任务的分析。这导致我们提出了第一个关于能力问题的模型，该模型包括五种主要类型的CQs，每种类型都有其不同的目的：范围问题（Scoping CQs, SCQ）、验证问题（Validation CQs, VCQ）、基础问题（Foundational CQs, FCQ）、关系问题（Relationship CQs, RCQ）和元性质问题（Metaproperty CQs, MpCQ）。该模型增强了CQs的清晰度，并且通过其各自的可识别的不同组成部分，旨在提高CQs在本体开发中的有效性。我们通过用户故事举例说明并评估这些CQs，并展示了它们如何适用于本体开发任务。为了促进其应用和研究，我们创建了一个包含438个CQs的注释库，即本体能力问题库（ROCQS），该库整合了一个现有的CQ数据集以及新的CQs和CQ模板，进一步证明了CQs类型之间的区分。 

---
# Clio: Privacy-Preserving Insights into Real-World AI Use 

**Title (ZH)**: 克利奥：隐私保护的现实世界人工智能应用洞察 

**Authors**: Alex Tamkin, Miles McCain, Kunal Handa, Esin Durmus, Liane Lovitt, Ankur Rathi, Saffron Huang, Alfred Mountfield, Jerry Hong, Stuart Ritchie, Michael Stern, Brian Clarke, Landon Goldberg, Theodore R. Sumers, Jared Mueller, William McEachen, Wes Mitchell, Shan Carter, Jack Clark, Jared Kaplan, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2412.13678)  

**Abstract**: How are AI assistants being used in the real world? While model providers in theory have a window into this impact via their users' data, both privacy concerns and practical challenges have made analyzing this data difficult. To address these issues, we present Clio (Claude insights and observations), a privacy-preserving platform that uses AI assistants themselves to analyze and surface aggregated usage patterns across millions of conversations, without the need for human reviewers to read raw conversations. We validate this can be done with a high degree of accuracy and privacy by conducting extensive evaluations. We demonstrate Clio's usefulness in two broad ways. First, we share insights about how models are being used in the real world from one million this http URL Free and Pro conversations, ranging from providing advice on hairstyles to providing guidance on Git operations and concepts. We also identify the most common high-level use cases on this http URL (coding, writing, and research tasks) as well as patterns that differ across languages (e.g., conversations in Japanese discuss elder care and aging populations at higher-than-typical rates). Second, we use Clio to make our systems safer by identifying coordinated attempts to abuse our systems, monitoring for unknown unknowns during critical periods like launches of new capabilities or major world events, and improving our existing monitoring systems. We also discuss the limitations of our approach, as well as risks and ethical concerns. By enabling analysis of real-world AI usage, Clio provides a scalable platform for empirically grounded AI safety and governance. 

**Abstract (ZH)**: AI助手中在现实世界中的应用情况如何？尽管模型提供者理论上可以通过用户数据看到这种影响，但隐私担忧和实际挑战使得分析这些数据变得困难。为了解决这些问题，我们提出了Clio（Claude见解和观察）这一隐私保护平台，该平台利用AI助手本身来分析和展示数百万对话中的聚合使用模式，而无需人工审阅原始对话。我们通过广泛评估验证了这种做法可以在高准确度和隐私保护的前提下进行。我们展示了Clio的两种主要用途。首先，我们从一百万次免费和专业版对话中分享了有关模型在实际世界中的应用见解，这些对话范围从提供发型建议到提供Git操作和概念的指导。我们还指出现最普遍的高层使用场景（编程、写作和研究任务），以及不同语言中对话模式的差异（例如，使用日语的对话中关于老年护理和老龄化人口的讨论比典型情况更为频繁）。其次，我们利用Clio确保我们的系统安全，识别协调的滥用企图，监控关键时期（如新功能发布或重大全球事件期间）中的未知风险，并改进现有的监控系统。我们还讨论了我们方法的局限性，以及相关风险和伦理问题。通过使对实际世界AI使用情况的分析成为可能，Clio提供了一个基于证据的AI安全和治理的可扩展平台。 

---
# G-VEval: A Versatile Metric for Evaluating Image and Video Captions Using GPT-4o 

**Title (ZH)**: G-VEval：一种使用GPT-4o评估图像和视频Caption的综合性指标 

**Authors**: Tony Cheng Tong, Sirui He, Zhiwen Shao, Dit-Yan Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2412.13647)  

**Abstract**: Evaluation metric of visual captioning is important yet not thoroughly explored. Traditional metrics like BLEU, METEOR, CIDEr, and ROUGE often miss semantic depth, while trained metrics such as CLIP-Score, PAC-S, and Polos are limited in zero-shot scenarios. Advanced Language Model-based metrics also struggle with aligning to nuanced human preferences. To address these issues, we introduce G-VEval, a novel metric inspired by G-Eval and powered by the new GPT-4o. G-VEval uses chain-of-thought reasoning in large multimodal models and supports three modes: reference-free, reference-only, and combined, accommodating both video and image inputs. We also propose MSVD-Eval, a new dataset for video captioning evaluation, to establish a more transparent and consistent framework for both human experts and evaluation metrics. It is designed to address the lack of clear criteria in existing datasets by introducing distinct dimensions of Accuracy, Completeness, Conciseness, and Relevance (ACCR). Extensive results show that G-VEval outperforms existing methods in correlation with human annotations, as measured by Kendall tau-b and Kendall tau-c. This provides a flexible solution for diverse captioning tasks and suggests a straightforward yet effective approach for large language models to understand video content, paving the way for advancements in automated captioning. Codes are available at this https URL 

**Abstract (ZH)**: 视觉captioning的评估指标非常重要但尚未得到充分研究。传统的指标如BLEU、METEOR、CIDEr和ROUGE往往未能捕捉到语义深度，而经过训练的指标如CLIP-Score、PAC-S和Polos在零样本场景中能力有限。基于高级语言模型的指标在准确反映细腻的人类偏好方面也存在问题。为了解决这些问题，我们引入了G-VEval，这是一种受G-Eval启发、由新的GPT-4o驱动的新颖指标。G-VEval利用大型多模态模型中的链式推理，并支持三种模式：无参考、有参考和结合模式，适用于视频和图像输入。我们还提出了一个新的评估数据集MSVD-Eval，以建立一个更加透明且一致的评估框架，适用于人类专家和评估指标。MSVD-Eval旨在通过引入准确性、完整性、简洁性和相关性（ACCR）等不同的维度，来解决现有数据集缺乏明确标准的问题。广泛的结果表明，G-VEval在与人类注释的相关性方面（使用Kendall tau-b和Kendall tau-c衡量）优于现有方法，为各种captioning任务提供了一个灵活的解决方案，表明大型语言模型理解视频内容的一种简单而有效的途径，并为自动化captioning的进步铺平了道路。代码可从以下链接获取：this https URL 

---
# Mind Your Theory: Theory of Mind Goes Deeper Than Reasoning 

**Title (ZH)**: 请考虑你的理论：理论心智比推理更为深刻 

**Authors**: Eitan Wagner, Nitay Alon, Joseph M. Barnby, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2412.13631)  

**Abstract**: Theory of Mind (ToM) capabilities in LLMs have recently become a central object of investigation. Cognitive science distinguishes between two steps required for ToM tasks: 1) determine whether to invoke ToM, which includes the appropriate Depth of Mentalizing (DoM), or level of recursion required to complete a task; and 2) applying the correct inference given the DoM. In this position paper, we first identify several lines of work in different communities in AI, including LLM benchmarking, ToM add-ons, ToM probing, and formal models for ToM. We argue that recent work in AI tends to focus exclusively on the second step which are typically framed as static logic problems. We conclude with suggestions for improved evaluation of ToM capabilities inspired by dynamic environments used in cognitive tasks. 

**Abstract (ZH)**: 大型语言模型（LLM）的理论思维（Theory of Mind, ToM）能力近期成为了研究的焦点。认知科学将ToM任务划分为两个步骤：1）确定是否需要调用ToM，这涉及到完成任务所需的心理化程度（Depth of Mentalizing, DoM）或递归层次；2）根据DoM应用正确的推理。在本文中，我们首先在AI的不同社区中识别了几条相关工作线，包括LLM基准测试、ToM插件、ToM探查以及ToM的形式模型。我们认为，近年来AI领域的研究主要集中于第二个步骤，通常将其作为静态逻辑问题来处理。最后，我们提出了借鉴认知任务中使用的动态环境来改进ToM能力评估的建议。 

---
# Reverse Region-to-Entity Annotation for Pixel-Level Visual Entity Linking 

**Title (ZH)**: 像素级视觉实体链接中的逆区域到实体标注方法 

**Authors**: Zhengfei Xu, Sijia Zhao, Yanchao Hao, Xiaolong Liu, Lili Li, Yuyang Yin, Bo Li, Xi Chen, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13614)  

**Abstract**: Visual Entity Linking (VEL) is a crucial task for achieving fine-grained visual understanding, matching objects within images (visual mentions) to entities in a knowledge base. Previous VEL tasks rely on textual inputs, but writing queries for complex scenes can be challenging. Visual inputs like clicks or bounding boxes offer a more convenient alternative. Therefore, we propose a new task, Pixel-Level Visual Entity Linking (PL-VEL), which uses pixel masks from visual inputs to refer to objects, supplementing reference methods for VEL. To facilitate research on this task, we have constructed the MaskOVEN-Wiki dataset through an entirely automatic reverse region-entity annotation framework. This dataset contains over 5 million annotations aligning pixel-level regions with entity-level labels, which will advance visual understanding towards fine-grained. Moreover, as pixel masks correspond to semantic regions in an image, we enhance previous patch-interacted attention with region-interacted attention by a visual semantic tokenization approach. Manual evaluation results indicate that the reverse annotation framework achieved a 94.8% annotation success rate. Experimental results show that models trained on this dataset improved accuracy by 18 points compared to zero-shot models. Additionally, the semantic tokenization method achieved a 5-point accuracy improvement over the trained baseline. 

**Abstract (ZH)**: 视觉实体链接（VEL）是实现细粒度视觉理解的关键任务，它涉及将图像中的对象（视觉提及）与知识库中的实体进行匹配。以前的VEL任务依赖于文本输入，但为复杂的场景编写查询可能会非常具有挑战性。基于视觉的输入，如点击或边界框，提供了更为便捷的替代方案。因此，我们提出了一项新的任务——像素级视觉实体链接（PL-VEL），该任务使用来自视觉输入的像素掩码来引用对象，补充了VEL的参考方法。为了促进对这项任务的研究，我们通过一个完全自动的反向区域-实体注释框架构建了MaskOVEN-Wiki数据集。该数据集包含了超过500万个像素级区域与实体级标签的对齐注释，有助于视觉理解向细粒度方向发展。此外，由于像素掩码对应图像中的语义区域，我们通过视觉语义分词方法增强了一种先前的 patch-交互注意力方法，采用了区域-交互注意力机制。手动评估结果显示反向注释框架的注释成功率为94.8%。实验结果表明，使用此数据集训练的模型在准确率上提高了18个百分点，相较零样本模型而言。此外，语义分词方法比训练基线提高了5个百分点的准确性。 

---
# Unlocking the Potential of Weakly Labeled Data: A Co-Evolutionary Learning Framework for Abnormality Detection and Report Generation 

**Title (ZH)**: 解锁弱标注数据的潜力：一种异常检测与报告生成的共进化学习框架 

**Authors**: Jinghan Sun, Dong Wei, Zhe Xu, Donghuan Lu, Hong Liu, Hong Wang, Sotirios A. Tsaftaris, Steven McDonagh, Yefeng Zheng, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13599)  

**Abstract**: Anatomical abnormality detection and report generation of chest X-ray (CXR) are two essential tasks in clinical practice. The former aims at localizing and characterizing cardiopulmonary radiological findings in CXRs, while the latter summarizes the findings in a detailed report for further diagnosis and treatment. Existing methods often focused on either task separately, ignoring their correlation. This work proposes a co-evolutionary abnormality detection and report generation (CoE-DG) framework. The framework utilizes both fully labeled (with bounding box annotations and clinical reports) and weakly labeled (with reports only) data to achieve mutual promotion between the abnormality detection and report generation tasks. Specifically, we introduce a bi-directional information interaction strategy with generator-guided information propagation (GIP) and detector-guided information propagation (DIP). For semi-supervised abnormality detection, GIP takes the informative feature extracted by the generator as an auxiliary input to the detector and uses the generator's prediction to refine the detector's pseudo labels. We further propose an intra-image-modal self-adaptive non-maximum suppression module (SA-NMS). This module dynamically rectifies pseudo detection labels generated by the teacher detection model with high-confidence predictions by the this http URL, for report generation, DIP takes the abnormalities' categories and locations predicted by the detector as input and guidance for the generator to improve the generated reports. 

**Abstract (ZH)**: 胸部X光（CXR）解剖异常检测和报告生成是临床实践中两个至关重要的任务。前者旨在定位和表征胸部影像中的心肺放射学发现，后者则通过详细的报告总结这些发现，以供进一步诊断和治疗使用。现有方法往往单独关注其中一项任务，而忽视了两者之间的联系。本研究提出了一种联合进化异常检测和报告生成（CoE-DG）框架。该框架利用既有完全标注（带有边界框注释和临床报告）又有弱标注（仅有报告）的数据，实现异常检测和报告生成任务之间的相互促进。具体而言，我们引入了一种双向信息交互策略，包括生成器指导的信息传播（GIP）和检测器指导的信息传播（DIP）。对于半监督异常检测，GIP 采用生成器抽取的具有信息量的特征作为检测器的辅助输入，并利用生成器的预测来优化检测器的伪标签。我们进一步提出了一种基于图像模态内自适应非极大值抑制模块（SA-NMS）。该模块根据教师检测模型高置信度预测动态修正伪检测标签，以改进报告生成。对于报告生成，DIP 将检测器预测的异常类别和位置作为输入和指导，帮助生成器生成更准确的报告。 

---
# Read Like a Radiologist: Efficient Vision-Language Model for 3D Medical Imaging Interpretation 

**Title (ZH)**: 像放射专家一样阅读：用于3D医疗成像解释的高效视觉-语言模型 

**Authors**: Changsun Lee, Sangjoon Park, Cheong-Il Shin, Woo Hee Choi, Hyun Jeong Park, Jeong Eun Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2412.13558)  

**Abstract**: Recent medical vision-language models (VLMs) have shown promise in 2D medical image interpretation. However extending them to 3D medical imaging has been challenging due to computational complexities and data scarcity. Although a few recent VLMs specified for 3D medical imaging have emerged, all are limited to learning volumetric representation of a 3D medical image as a set of sub-volumetric features. Such process introduces overly correlated representations along the z-axis that neglect slice-specific clinical details, particularly for 3D medical images where adjacent slices have low redundancy. To address this limitation, we introduce MS-VLM that mimic radiologists' workflow in 3D medical image interpretation. Specifically, radiologists analyze 3D medical images by examining individual slices sequentially and synthesizing information across slices and views. Likewise, MS-VLM leverages self-supervised 2D transformer encoders to learn a volumetric representation that capture inter-slice dependencies from a sequence of slice-specific features. Unbound by sub-volumetric patchification, MS-VLM is capable of obtaining useful volumetric representations from 3D medical images with any slice length and from multiple images acquired from different planes and phases. We evaluate MS-VLM on publicly available chest CT dataset CT-RATE and in-house rectal MRI dataset. In both scenarios, MS-VLM surpasses existing methods in radiology report generation, producing more coherent and clinically relevant reports. These findings highlight the potential of MS-VLM to advance 3D medical image interpretation and improve the robustness of medical VLMs. 

**Abstract (ZH)**: 近年来，医学视觉语言模型（VLMs）在2D医学图像解释方面展现了潜力。然而，将其扩展到3D医学成像仍然面临计算复杂性和数据稀少的挑战。尽管少数针对3D医学成像的VLMs已经涌现，但所有这些模型都仅限于学习3D医学图像的体积表示，即作为一系列子体积特征集。这一过程引入了沿z轴过度相关的表示，忽略了切片特定的临床细节，特别是在相邻切片具有低冗余度的3D医学图像中。为了克服这一限制，我们引入了MS-VLM，其模仿了放射科医师在解释3D医学图像时的工作流程。具体而言，放射科医师通过顺序检查每个切片并综合各切片和视角的信息来分析3D医学图像。类似地，MS-VLM 利用自监督的2D变换器编码器从一系列特定切片特征中学习体积表示，以捕捉切片间的依赖关系。不受子体积分割限制，MS-VLM 能够从具有任意切片长度的3D医学图像中获取有用的体积表示，并从多个来自不同平面和阶段的图像中获取表示。我们在公开的胸部CT数据集CT-RATE和内部的直肠MRI数据集上评估了MS-VLM。在两种场景下，MS-VLM 在放射学报告生成中均优于现有方法，生成了更连贯且具有临床意义的报告。这些发现突显了MS-VLM 在推动3D医学图像解释以及提高医学VLMs的鲁棒性方面的潜力。 

---
# Query-centric Audio-Visual Cognition Network for Moment Retrieval, Segmentation and Step-Captioning 

**Title (ZH)**: 面向查询的音频-视觉认知网络用于时刻检索、分割和步骤标注 

**Authors**: Yunbin Tu, Liang Li, Li Su, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13543)  

**Abstract**: Video has emerged as a favored multimedia format on the internet. To better gain video contents, a new topic HIREST is presented, including video retrieval, moment retrieval, moment segmentation, and step-captioning. The pioneering work chooses the pre-trained CLIP-based model for video retrieval, and leverages it as a feature extractor for other three challenging tasks solved in a multi-task learning paradigm. Nevertheless, this work struggles to learn the comprehensive cognition of user-preferred content, due to disregarding the hierarchies and association relations across modalities. In this paper, guided by the shallow-to-deep principle, we propose a query-centric audio-visual cognition (QUAG) network to construct a reliable multi-modal representation for moment retrieval, segmentation and step-captioning. Specifically, we first design the modality-synergistic perception to obtain rich audio-visual content, by modeling global contrastive alignment and local fine-grained interaction between visual and audio modalities. Then, we devise the query-centric cognition that uses the deep-level query to perform the temporal-channel filtration on the shallow-level audio-visual representation. This can cognize user-preferred content and thus attain a query-centric audio-visual representation for three tasks. Extensive experiments show QUAG achieves the SOTA results on HIREST. Further, we test QUAG on the query-based video summarization task and verify its good generalization. 

**Abstract (ZH)**: 视频已成为互联网上流行的多媒体格式。为了更好地获取视频内容，提出了一种新的主题HIREST（Hierarchical Retrieval and Visualization of Moments in Videos），包括视频检索、关键瞬间检索、瞬间分割和步骤标注。这项开创性的工作选择了基于预训练的CLIP模型进行视频检索，并将其作为特征提取器用于其他三项具有挑战性任务的多任务学习范式中。然而，这项工作在学习用户偏好的全面认知方面存在困难，因为它忽略了跨模态的层级关系和关联关系。本文受浅层到深层原则的启发，提出了一种以查询为中心的视听认知（QUAG）网络，以构建可靠多模态表示，应用于关键瞬间检索、分割和步骤标注。具体而言，我们首先设计了模态协同感知，通过建模全局对比对齐和局部细粒度视听模态交互，获取丰富的视听内容。然后，我们设计了以查询为中心的认知，利用深层次的查询对浅层次的视听表示进行时序-通道过滤，从而认知用户偏好内容，并因此获得以查询为中心的视听表示，应用于三项任务。大量实验证明，QUAG在HIREST上达到了SOTA结果。此外，我们还在基于查询的视频摘要任务上测试了QUAG，验证了其良好的泛化能力。 

---
# Information-Theoretic Generative Clustering of Documents 

**Title (ZH)**: 信息论生成聚类中的文档聚类 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2412.13534)  

**Abstract**: We present {\em generative clustering} (GC) for clustering a set of documents, $\mathrm{X}$, by using texts $\mathrm{Y}$ generated by large language models (LLMs) instead of by clustering the original documents $\mathrm{X}$. Because LLMs provide probability distributions, the similarity between two documents can be rigorously defined in an information-theoretic manner by the KL divergence. We also propose a natural, novel clustering algorithm by using importance sampling. We show that GC achieves the state-of-the-art performance, outperforming any previous clustering method often by a large margin. Furthermore, we show an application to generative document retrieval in which documents are indexed via hierarchical clustering and our method improves the retrieval accuracy. 

**Abstract (ZH)**: 我们提出了生成聚类（Generative Clustering，GC）的方法，通过使用大型语言模型（LLMs）生成的文本 $\mathrm{Y}$，而不是直接聚类原始文档 $\mathrm{X}$，来进行文档集的聚类。由于大型语言模型提供了概率分布，可以通过 KL 散度以信息论的方式严格定义两份文档之间的相似性。我们还提出了一种基于重要性采样的自然新颖聚类算法。实验结果表明，生成聚类方法在性能上达到了当前最先进的水平，往往显著优于任何先前的聚类方法。此外，我们展示了生成文档检索的应用，在这种应用中，文档通过层次聚类进行索引，我们的方法提高了检索准确性。 

---
# Dynamic Adapter with Semantics Disentangling for Cross-lingual Cross-modal Retrieval 

**Title (ZH)**: 跨语言多模态检索中的语义解耦动态适配器 

**Authors**: Rui Cai, Zhiyu Dong, Jianfeng Dong, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13510)  

**Abstract**: Existing cross-modal retrieval methods typically rely on large-scale vision-language pair data. This makes it challenging to efficiently develop a cross-modal retrieval model for under-resourced languages of interest. Therefore, Cross-lingual Cross-modal Retrieval (CCR), which aims to align vision and the low-resource language (the target language) without using any human-labeled target-language data, has gained increasing attention. As a general parameter-efficient way, a common solution is to utilize adapter modules to transfer the vision-language alignment ability of Vision-Language Pretraining (VLP) models from a source language to a target language. However, these adapters are usually static once learned, making it difficult to adapt to target-language captions with varied expressions. To alleviate it, we propose Dynamic Adapter with Semantics Disentangling (DASD), whose parameters are dynamically generated conditioned on the characteristics of the input captions. Considering that the semantics and expression styles of the input caption largely influence how to encode it, we propose a semantic disentangling module to extract the semantic-related and semantic-agnostic features from the input, ensuring that generated adapters are well-suited to the characteristics of input caption. Extensive experiments on two image-text datasets and one video-text dataset demonstrate the effectiveness of our model for cross-lingual cross-modal retrieval, as well as its good compatibility with various VLP models. 

**Abstract (ZH)**: 现有的跨模态检索方法通常依赖大规模的视觉-语言配对数据。这使得为感兴趣的小资源语言开发高效的跨模态检索模型变得具有挑战性。因此，跨语言跨模态检索（Cross-lingual Cross-modal Retrieval, CCR），旨在无需使用任何人工标注的目标语言数据的情况下，对视觉和低资源语言（目标语言）进行对齐，已经引起了越来越多的关注。作为一种通用的参数高效方法，一个常见的解决方案是利用适配模块将视觉-语言预训练（VLP）模型在源语言中的视觉-语言对齐能力转移到目标语言中。然而，这些适配模块一旦学习完成，通常是静态的，难以适应具有多样化表达的目标语言描述。为了解决这一问题，我们提出了动态适配器与语义分离（Dynamic Adapter with Semantics Disentangling, DASD），其参数根据输入描述的特征动态生成。考虑到输入描述的语义和表达风格会极大地影响其编码方式，我们提出了一个语义分离模块，从输入中提取与语义相关和无关的特征，确保生成的适配器能够很好地适应输入描述的特征。在两个图像-文本数据集和一个视频-文本数据集上的大量实验表明，我们的模型在跨语言跨模态检索中具有有效性和良好的兼容性，能够与各种VLP模型兼容。 

---
# T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation 

**Title (ZH)**: T$^3$-S2S：无需训练的三元组微调方法用于草图到场景生成

注释：这个标题翻译成中文时，保持了原英文中的缩写 "T$^3$-S2S"，并将其解释为 "无需训练的三元组微调方法用于草图到场景生成"，以确保符合学术规范的表达。对于 "Training-free Triplet Tuning" 我们翻译为 "无需训练的三元组微调"，这样的表述清晰且符合专业术语的翻译标准。 

**Authors**: Zhenhong Sun, Yifu Wang, Yonhon Ng, Yunfei Duan, Daoyi Dong, Hongdong Li, Pan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.13486)  

**Abstract**: Scene generation is crucial to many computer graphics applications. Recent advances in generative AI have streamlined sketch-to-image workflows, easing the workload for artists and designers in creating scene concept art. However, these methods often struggle for complex scenes with multiple detailed objects, sometimes missing small or uncommon instances. In this paper, we propose a Training-free Triplet Tuning for Sketch-to-Scene (T3-S2S) generation after reviewing the entire cross-attention mechanism. This scheme revitalizes the existing ControlNet model, enabling effective handling of multi-instance generations, involving prompt balance, characteristics prominence, and dense tuning. Specifically, this approach enhances keyword representation via the prompt balance module, reducing the risk of missing critical instances. It also includes a characteristics prominence module that highlights TopK indices in each channel, ensuring essential features are better represented based on token sketches. Additionally, it employs dense tuning to refine contour details in the attention map, compensating for instance-related regions. Experiments validate that our triplet tuning approach substantially improves the performance of existing sketch-to-image models. It consistently generates detailed, multi-instance 2D images, closely adhering to the input prompts and enhancing visual quality in complex multi-instance scenes. Code is available at this https URL. 

**Abstract (ZH)**: 场景生成对于许多计算机图形应用至关重要。近期生成型AI的发展简化了草图到图像的工作流，减轻了艺术家和设计师在创作场景概念艺术时的工作负担。然而，这些方法在处理具有多个详细物体的复杂场景时往往存在困难，有时会遗漏小的或不常见的实例。在本文中，我们提出了一种无需训练的三重调优方案（T3-S2S）用于从草图生成场景。在全面回顾整个交叉注意机制后，该方案重新激活了现有的ControlNet模型，使其能够有效处理多重实例生成问题，并涉及提示平衡、特征凸显和密集调优。具体而言，这种方法通过提示平衡模块增强关键词表示，从而降低遗漏关键实例的风险。此外，还包括一个特征凸显模块，该模块在每个通道中突出显示TopK索引，以确保基于标记草图的关键特征能够更好地表现。另外，还采用了密集调优方法，以细化注意图中的轮廓细节，补偿实例相关区域的不足。实验验证表明，我们的三重调优方法能够显著提升现有草图到图像模型的性能。该方法能够一致地生成详细且包含多重实例的2D图像，严格遵循输入提示，并在复杂的多重实例场景中提升视觉质量。源代码可以在以下链接获得：https://xxxxxx（请将xxxxxx替换为实际的链接地址）。 

---
# Transducer Tuning: Efficient Model Adaptation for Software Tasks Using Code Property Graphs 

**Title (ZH)**: 转换器调整：使用代码属性图进行软件任务的高效模型调整 

**Authors**: Imam Nur Bani Yusuf, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13467)  

**Abstract**: Large language models have demonstrated promising performance across various software engineering tasks. While fine-tuning is a common practice to adapt these models for downstream tasks, it becomes challenging in resource-constrained environments due to increased memory requirements from growing trainable parameters in increasingly large language models. We introduce \approach, a technique to adapt large models for downstream code tasks using Code Property Graphs (CPGs). Our approach introduces a modular component called \transducer that enriches code embeddings with structural and dependency information from CPGs. The Transducer comprises two key components: Graph Vectorization Engine (GVE) and Attention-Based Fusion Layer (ABFL). GVE extracts CPGs from input source code and transforms them into graph feature vectors. ABFL then fuses those graphs feature vectors with initial code embeddings from a large language model. By optimizing these transducers for different downstream tasks, our approach enhances the models without the need to fine-tune them for specific tasks. We have evaluated \approach on three downstream tasks: code summarization, assert generation, and code translation. Our results demonstrate competitive performance compared to full parameter fine-tuning while reducing up to 99\% trainable parameters to save memory. \approach also remains competitive against other fine-tuning approaches (e.g., LoRA, Prompt-Tuning, Prefix-Tuning) while using only 1.5\%-80\% of their trainable parameters. Our findings show that integrating structural and dependency information through Transducer Tuning enables more efficient model adaptation, making it easier for users to adapt large models in resource-constrained settings. 

**Abstract (ZH)**: 大型语言模型在各种软件工程任务中展现了令人鼓舞的表现。虽然微调是将这些模型适应下游任务的一种常见做法，但在资源受限的环境中，由于大型语言模型可训练参数的增加导致的内存需求增加，这一做法变得具有挑战性。我们引入了一种名为Approach的技术，利用代码属性图（CPG）来适应大型模型用于下游代码任务。Approach引入了一个模块化的组件，称为Transducer，它通过CPG中的结构和依赖信息来丰富代码嵌入。Transducer包含两个关键组件：图向量引擎（Graph Vectorization Engine, GVE）和基于注意力的融合层（Attention-Based Fusion Layer, ABFL）。GVE从输入源代码中提取CPG并将其转换为图特征向量。ABFL随后将这些图特征向量与大型语言模型初始的代码嵌入融合。通过对这些Transducer进行不同下游任务的优化，我们的方法可以在不针对特定任务进行微调的情况下增强模型。我们在三个下游任务上对Approach进行了评估：代码摘要、断言生成和代码翻译。我们的结果显示，在降低99%可训练参数以节省内存的同时，Approach能够与全参数微调取得具有竞争力的性能。此外，Approach在仅使用其他微调方法（例如LoRA、提示微调、前缀微调）1.5%到80%的可训练参数的情况下，仍然保持了竞争力。我们的研究发现表明，通过Transducer调优整合结构和依赖信息可以使模型适应更加高效，并且使用户能在资源受限的环境中更容易地适应大型模型。 

---
# GenX: Mastering Code and Test Generation with Execution Feedback 

**Title (ZH)**: GenX：通过执行反馈掌握代码和测试生成 

**Authors**: Nan Wang, Yafei Liu, Chen Chen, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13464)  

**Abstract**: Recent advancements in language modeling have enabled the translation of natural language into code, and the use of execution feedback to improve code generation. However, these methods often rely heavily on pre-existing test cases, which may not always be available or comprehensive. In this work, we propose a novel approach that concurrently trains a code generation model and a test generation model, utilizing execution feedback to refine and enhance the performance of both. We introduce two strategies for test and code data augmentation and a new scoring function for code and test ranking. We experiment on the APPS dataset and demonstrate that our approach can effectively generate and augment test cases, filter and synthesize correct code solutions, and rank the quality of generated code and tests. The results demonstrate that our models, when iteratively trained with an increasing number of test cases and code solutions, outperform those trained on the original dataset. 

**Abstract (ZH)**: 近年来，语言模型的进展使得将自然语言转换为代码成为可能，并通过执行反馈来提高代码生成的质量。然而，这些方法往往依赖于现有的测试用例，而这些测试用例可能并不总是可用或全面。在此研究中，我们提出了一种新颖的方法，同时训练代码生成模型和测试生成模型，并利用执行反馈来优化和提升两者的性能。我们引入了两种测试和代码数据增强策略以及一个新的评分函数，用于代码和测试的质量排名。我们在APPS数据集上进行了实验，并展示了我们的方法能够有效生成和增强测试用例、过滤和综合正确的代码解决方案，并对生成的代码和测试的质量进行排名。实验结果表明，当我们的模型在不断增加的测试用例和代码解决方案的基础上迭代训练时，其性能优于仅在原始数据集上训练的模型。 

---
# FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding 

**Title (ZH)**: FlashVTG：特征分层与自适应得分处理网络用于视频时间定位 

**Authors**: Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13441)  

**Abstract**: Text-guided Video Temporal Grounding (VTG) aims to localize relevant segments in untrimmed videos based on textual descriptions, encompassing two subtasks: Moment Retrieval (MR) and Highlight Detection (HD). Although previous typical methods have achieved commendable results, it is still challenging to retrieve short video moments. This is primarily due to the reliance on sparse and limited decoder queries, which significantly constrain the accuracy of predictions. Furthermore, suboptimal outcomes often arise because previous methods rank predictions based on isolated predictions, neglecting the broader video context. To tackle these issues, we introduce FlashVTG, a framework featuring a Temporal Feature Layering (TFL) module and an Adaptive Score Refinement (ASR) module. The TFL module replaces the traditional decoder structure to capture nuanced video content variations across multiple temporal scales, while the ASR module improves prediction ranking by integrating context from adjacent moments and multi-temporal-scale features. Extensive experiments demonstrate that FlashVTG achieves state-of-the-art performance on four widely adopted datasets in both MR and HD. Specifically, on the QVHighlights dataset, it boosts mAP by 5.8% for MR and 3.3% for HD. For short-moment retrieval, FlashVTG increases mAP to 125% of previous SOTA performance. All these improvements are made without adding training burdens, underscoring its effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: 文本引导的视频时间定位（VTG）旨在基于文本描述在未裁剪的视频中定位相关片段，涵盖了两个子任务：时刻检索（MR）和亮点检测（HD）。尽管以往典型的方法已经取得了不错的成果，但在检索短视频片段方面依然具有挑战性。这主要是因为依赖于稀疏且有限的解码器查询，这显著限制了预测的准确性。此外，以往方法往往基于孤立预测进行排名，忽略了更广泛的视频上下文，导致次优结果的产生。为了解决这些问题，我们提出了FlashVTG框架，该框架包含一个时间特征层叠（TFL）模块和一种自适应评分精炼（ASR）模块。TFL模块取代了传统的解码器结构，以捕捉多时间尺度上的视频内容变化，而ASR模块通过整合相邻时刻和多时间尺度特征来改进预测排名。大量实验表明，FlashVTG在两个子任务上均在四个广泛采用的数据集上实现了最先进的性能。特别是在QVHighlights数据集上，它将MR任务的mAP提高了5.8%，HD任务提高了3.3%。在短时时刻检索方面，FlashVTG将mAP提高到之前SOTA性能的125%。所有这些改进均未增加训练负担，突显了其有效性。我们的代码可以在以下链接访问：[请填写实际链接]。 

---
# Catalysts of Conversation: Examining Interaction Dynamics Between Topic Initiators and Commentors in Alzheimer's Disease Online Communities 

**Title (ZH)**: 催化剂作用：探讨阿尔茨海默病在线社区中话题发起者与评论者之间的互动动态 

**Authors**: Congning Ni, Qingxia Chen, Lijun Song, Patricia Commiskey, Qingyuan Song, Bradley A. Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13388)  

**Abstract**: Informal caregivers (e.g.,family members or friends) of people living with Alzheimers Disease and Related Dementias (ADRD) face substantial challenges and often seek informational or emotional support through online communities. Understanding the factors that drive engagement within these platforms is crucial, as it can enhance their long-term value for caregivers by ensuring that these communities effectively meet their needs. This study investigated the user interaction dynamics within two large, popular ADRD communities, TalkingPoint and ALZConnected, focusing on topic initiator engagement, initial post content, and the linguistic patterns of comments at the thread level. Using analytical methods such as propensity score matching, topic modeling, and predictive modeling, we found that active topic initiator engagement drives higher comment volumes, and reciprocal replies from topic initiators encourage further commentor engagement at the community level. Practical caregiving topics prompt more re-engagement of topic initiators, while emotional support topics attract more comments from other commentors. Additionally, the linguistic complexity and emotional tone of a comment influence its likelihood of receiving replies from topic initiators. These findings highlight the importance of fostering active and reciprocal engagement and providing effective strategies to enhance sustainability in ADRD caregiving and broader health-related online communities. 

**Abstract (ZH)**: 生活在中国以外地区的阿尔茨海默病及相关痴呆症（ADRD）患者的非正式护理者（例如家庭成员或朋友）面临诸多挑战，并且常常通过在线社区寻求信息或情感支持。了解驱动这些平台参与的因素至关重要，因为这可以确保这些社区能够更好地满足护理者的需求，从而提高它们的长期价值。本研究探讨了两大受欢迎的ADRD社区——TalkingPoint和ALZConnected——中的用户互动动态，重点关注话题发起人参与度、初始帖子内容以及主题线级评论的语义模式。通过使用倾向得分匹配、主题建模和预测建模等分析方法，我们发现活跃的话题发起人参与度可以促进更多评论，而话题发起人之间的相互回复也鼓励了社区层面的进一步参与。务实的护理主题会促使话题发起人重新参与，而情感支持主题则吸引其他评论者的更多评论。此外，评论的语言复杂性和情感语调也会影响话题发起人回复的可能性。这些发现强调了培养主动和相互参与的重要性，并提出了增强ADRD护理以及更广泛健康相关在线社区可持续性的有效策略。 

---
# Adaptive Two-Phase Finetuning LLMs for Japanese Legal Text Retrieval 

**Title (ZH)**: 针对日语法律文本检索的自适应两阶段微调大型语言模型 

**Authors**: Quang Hoang Trung, Nguyen Van Hoang Phuc, Le Trung Hoang, Quang Huu Hieu, Vo Nguyen Le Duy  

**Link**: [PDF](https://arxiv.org/pdf/2412.13205)  

**Abstract**: Text Retrieval (TR) involves finding and retrieving text-based content relevant to a user's query from a large repository, with applications in real-world scenarios such as legal document retrieval. While most existing studies focus on English, limited work addresses Japanese contexts. In this paper, we introduce a new dataset specifically designed for Japanese legal contexts and propose a novel two-phase pipeline tailored to this domain.
In the first phase, the model learns a broad understanding of global contexts, enhancing its generalization and adaptability to diverse queries. In the second phase, the model is fine-tuned to address complex queries specific to legal scenarios. Extensive experiments are conducted to demonstrate the superior performance of our method, which outperforms existing baselines.
Furthermore, our pipeline proves effective in English contexts, surpassing comparable baselines on the MS MARCO dataset. We have made our code publicly available on GitHub, and the model checkpoints are accessible via HuggingFace. 

**Abstract (ZH)**: 文本检索（Text Retrieval, TR）涉及从大规模存储库中找到并与用户查询相关的文本内容，应用领域包括法律文件检索等实际场景。尽管大多数现有研究侧重于英语，但在日本语环境中的工作却很少。本文介绍了一个专门为日本法律情境设计的新数据集，并提出了一种针对该领域的新型两阶段管道。

在第一阶段，模型学习广阔的整体背景理解能力，增强其对多样查询的一般化和适应性。在第二阶段，模型通过微调来处理特定于法律场景的复杂查询。通过广泛的实验，我们展示了该方法的优越性能，其性能超过了现有基线方法。

此外，我们的管道在英语环境中也表现出色，在MS MARCO数据集上超过了可比的基线方法。我们已将代码公开并发布在GitHub上，模型检查点也可通过HuggingFace访问。 

---
