# LongSpec: Long-Context Speculative Decoding with Efficient Drafting and Verification 

**Title (ZH)**: 长上下文 speculate 解码：高效草稿与验证方法 

**Authors**: Penghui Yang, Cunxiao Du, Fengzhuo Zhang, Haonan Wang, Tianyu Pang, Chao Du, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2502.17421)  

**Abstract**: Speculative decoding has become a promising technique to mitigate the high inference latency of autoregressive decoding in Large Language Models (LLMs). Despite its promise, the effective application of speculative decoding in LLMs still confronts three key challenges: the increasing memory demands of the draft model, the distribution shift between the short-training corpora and long-context inference, and inefficiencies in attention implementation. In this work, we enhance the performance of speculative decoding in long-context settings by addressing these challenges. First, we propose a memory-efficient draft model with a constant-sized Key-Value (KV) cache. Second, we introduce novel position indices for short-training data, enabling seamless adaptation from short-context training to long-context inference. Finally, we present an innovative attention aggregation method that combines fast implementations for prefix computation with standard attention for tree mask handling, effectively resolving the latency and memory inefficiencies of tree decoding. Our approach achieves strong results on various long-context tasks, including repository-level code completion, long-context summarization, and o1-like long reasoning tasks, demonstrating significant improvements in latency reduction. The code is available at this https URL. 

**Abstract (ZH)**: 推测解码已成为一种有前途的技术，用于缓解大型语言模型（LLMs）自回归解码的高推理延迟问题。尽管推测解码具有潜力，但在LLMs中的有效应用仍面临三大关键挑战：草稿模型不断增加的内存需求、短期训练语料库与长期上下文推理之间的分布变化，以及注意力机制实现的低效率。在这项工作中，我们通过解决这些挑战来增强长期上下文设置中推测解码的性能。首先，我们提出了一种内存高效的草稿模型，配备恒定大小的Key-Value（KV）缓存。其次，我们引入了新型位置索引，用于短期训练数据，使从短期上下文训练平滑过渡到长期上下文推理成为可能。最后，我们提出了创新的注意力聚合方法，将前缀计算的快速实现与标准注意力机制相结合，有效地解决了树解码的延迟和内存低效问题。我们的方法在多种长期上下文任务中取得了优异的结果，包括仓库级别的代码补全、长期摘要以及O1-like长期推理任务，展示了显著的延迟降低效果。代码可在以下链接获取：this https URL。 

---
# Reasoning with Latent Thoughts: On the Power of Looped Transformers 

**Title (ZH)**: 基于潜在思维的推理：循环Transformer的威力 

**Authors**: Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17416)  

**Abstract**: Large language models have shown remarkable reasoning abilities and scaling laws suggest that large parameter count, especially along the depth axis, is the primary driver. In this work, we make a stronger claim -- many reasoning problems require a large depth but not necessarily many parameters. This unlocks a novel application of looped models for reasoning. Firstly, we show that for many synthetic reasoning problems like addition, $p$-hop induction, and math problems, a $k$-layer transformer looped $L$ times nearly matches the performance of a $kL$-layer non-looped model, and is significantly better than a $k$-layer model. This is further corroborated by theoretical results showing that many such reasoning problems can be solved via iterative algorithms, and thus, can be solved effectively using looped models with nearly optimal depth. Perhaps surprisingly, these benefits also translate to practical settings of language modeling -- on many downstream reasoning tasks, a language model with $k$-layers looped $L$ times can be competitive to, if not better than, a $kL$-layer language model. In fact, our empirical analysis reveals an intriguing phenomenon: looped and non-looped models exhibit scaling behavior that depends on their effective depth, akin to the inference-time scaling of chain-of-thought (CoT) reasoning. We further elucidate the connection to CoT reasoning by proving that looped models implicitly generate latent thoughts and can simulate $T$ steps of CoT with $T$ loops. Inspired by these findings, we also present an interesting dichotomy between reasoning and memorization, and design a looping-based regularization that is effective on both fronts. 

**Abstract (ZH)**: 大型语言模型展现出了显著的推理能力，而扩展规律表明，模型参数量，特别是在深度维度上的扩展，是主要驱动力。本项工作中，我们提出一个更强的主张——许多推理问题需要较大的深度但不一定需要大量的参数。这为循环模型在推理方面的应用打开了全新途径。首先，我们展示了对于许多合成推理问题，如加法、$p$-跳归纳和数学问题，一个$k$层的循环变压器模型循环$L$次几乎达到了一个$kL$层非循环模型的表现，并且比$k$层的模型表现更好。这进一步由理论结果得到了印证，这些推理问题可以通过迭代算法解决，因此可以使用循环模型以几乎最优的深度有效解决。其实，这些优势也延伸到了语言建模的实际应用场景中——对于许多下游推理任务，一个$k$层的模型循环$L$次可以与甚至优于$kL$层的模型竞争。事实上，我们的实证分析揭示了一个有趣的现象：循环模型和非循环模型的扩展行为取决于它们的有效深度，与链式思维推理（CoT）推理的推理时扩展行为类似。我们通过证明循环模型隐含生成潜在思想并能够通过$L$次循环模拟$T$步CoT推理，进一步阐明了与CoT推理的联系。受这些发现的启发，我们还提出推理与记忆之间的有趣二分法，并设计了一种基于循环的正则化方法，该方法在这两方面都是有效的。 

---
# Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning 

**Title (ZH)**: 数学推理中测试时缩放的语义泛化能力 

**Authors**: Guijin Son, Jiwoo Hong, Hyunwoo Ko, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2502.17407)  

**Abstract**: Scaling pre-training compute has proven effective for achieving mulitlinguality, but does the same hold for test-time scaling? In this work, we introduce MCLM, a multilingual math benchmark featuring competition-level problems in 55 languages. We test three test-time scaling methods-Outcome Reward Modeling (ORM), Process Reward Modeling (ORM), and Budget Forcing (BF)-on both Qwen2.5-1.5B Math and MR1-1.5B, a multilingual LLM we trained for extended reasoning. Our experiments show that using Qwen2.5-1.5B Math with ORM achieves a score of 35.8 on MCLM, while BF on MR1-1.5B attains 35.2. Although "thinking LLMs" have recently garnered significant attention, we find that their performance is comparable to traditional scaling methods like best-of-N once constrained to similar levels of inference FLOPs. Moreover, while BF yields a 20-point improvement on English AIME, it provides only a 1.94-point average gain across other languages-a pattern consistent across the other test-time scaling methods we studied-higlighting that test-time scaling may not generalize as effectively to multilingual tasks. To foster further research, we release MCLM, MR1-1.5B, and evaluation results. 

**Abstract (ZH)**: 预训练计算规模的扩展已被证明对实现多语言能力有效，但在测试时扩展是否同样有效？在这项工作中，我们引入了MCLM，这是一个涵盖55种语言的多语言数学基准，包含竞赛级别的问题。我们测试了三种测试时扩展方法——结果奖励建模（ORM）、过程奖励建模（ORM）和预算强迫法（BF），分别在Qwen2.5-1.5B Math和MR1-1.5B上进行，其中MR1-1.5B是我们为扩展推理训练的一种多语言大型语言模型。我们的实验结果显示，使用Qwen2.5-1.5B Math和ORM可以达到MCLM上的35.8分，而使用BF在MR1-1.5B上则达到35.2分。尽管“思考型”大语言模型最近引起了广泛关注，但我们发现它们的表现与传统扩展方法（如最优N法）在类似推理FLOPs限制下的表现相当。此外，尽管BF在英语AIME上提供了20分的改进，但在其他语言上的平均增益仅为1.94分——这与我们研究的其他测试时扩展方法的趋势一致，突出表明测试时扩展可能在多语言任务上的有效性较差。

为了促进进一步的研究，我们发布了MCLM、MR1-1.5B和评估结果。 

---
# FIG: Forward-Inverse Generation for Low-Resource Domain-specific Event Detection 

**Title (ZH)**: FIG：面向特定领域事件检测的正向-逆向生成方法 

**Authors**: Tanmay Parekh, Yuxuan Dong, Lucas Bandarkar, Artin Kim, I-Hung Hsu, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17394)  

**Abstract**: Event Detection (ED) is the task of identifying typed event mentions of interest from natural language text, which benefits domain-specific reasoning in biomedical, legal, and epidemiological domains. However, procuring supervised data for thousands of events for various domains is a laborious and expensive task. To this end, existing works have explored synthetic data generation via forward (generating labels for unlabeled sentences) and inverse (generating sentences from generated labels) generations. However, forward generation often produces noisy labels, while inverse generation struggles with domain drift and incomplete event annotations. To address these challenges, we introduce FIG, a hybrid approach that leverages inverse generation for high-quality data synthesis while anchoring it to domain-specific cues extracted via forward generation on unlabeled target data. FIG further enhances its synthetic data by adding missing annotations through forward generation-based refinement. Experimentation on three ED datasets from diverse domains reveals that FIG outperforms the best baseline achieving average gains of 3.3% F1 and 5.4% F1 in the zero-shot and few-shot settings respectively. Analyzing the generated trigger hit rate and human evaluation substantiates FIG's superior domain alignment and data quality compared to existing baselines. 

**Abstract (ZH)**: 事件检测（ED）是识别自然语言文本中特定类型事件提及的任务，这有利于生物医药、法律和流行病学等特定领域的推理。然而，为不同领域收集数千种事件的监督数据是一个耗时且昂贵的过程。为此，现有研究已探索通过正向生成（为未标记句子生成标签）和逆向生成（从生成的标签生成句子）来生成合成数据的方法。然而，正向生成往往会生成噪声标签，而逆向生成则难以应对领域漂移和事件标注不完整的问题。为解决这些挑战，我们提出了一种混合方法FIG，该方法利用逆向生成生成高质量的数据，同时通过正向生成提取的领域特定线索进行锚定。FIG进一步通过基于正向生成的精修添加缺失的标注，以增强其合成数据。在三个来自不同领域的事件检测数据集上的实验结果显示，FIG分别在零样本和少量样本设置下优于最佳基线模型，分别获得了平均3.3%和5.4%的F1分数提升。生成触发器命中率分析和人类评估进一步证明了FIG在领域对齐和数据质量方面显著优于现有基线模型。 

---
# Mitigating Bias in RAG: Controlling the Embedder 

**Title (ZH)**: 缓解RAG中的偏差：控制嵌入器 

**Authors**: Taeyoun Kim, Jacob Springer, Aditi Raghunathan, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.17390)  

**Abstract**: In retrieval augmented generation (RAG) systems, each individual component -- the LLM, embedder, and corpus -- could introduce biases in the form of skews towards outputting certain perspectives or identities. In this work, we study the conflict between biases of each component and their relationship to the overall bias of the RAG system, which we call bias conflict. Examining both gender and political biases as case studies, we show that bias conflict can be characterized through a linear relationship among components despite its complexity in 6 different LLMs. Through comprehensive fine-tuning experiments creating 120 differently biased embedders, we demonstrate how to control bias while maintaining utility and reveal the importance of reverse-biasing the embedder to mitigate bias in the overall system. Additionally, we find that LLMs and tasks exhibit varying sensitivities to the embedder bias, a crucial factor to consider for debiasing. Our results underscore that a fair RAG system can be better achieved by carefully controlling the bias of the embedder rather than increasing its fairness. 

**Abstract (ZH)**: 在检索增强生成（RAG）系统中，每一部分构件——大语言模型（LLM）、嵌入器和语料库——都有可能以偏见的形式偏向输出某些特定视角或身份。在本研究中，我们探讨了每个构件偏见与其对RAG系统整体偏见关系之间的冲突，称之为偏见冲突。通过性别和政治偏见作为案例研究，我们展示了尽管在6种不同的LLM中复杂性不同，但可以通过这些构件之间的线性关系来表征偏见冲突。通过全面的微调实验创建了120种不同程度的偏见嵌入器，并展示了如何在保持实用性的前提下控制偏见，揭示了逆向偏置嵌入器以减轻整体系统偏见的重要性。此外，我们发现大语言模型和任务对嵌入器偏见的敏感性存在差异，这是去偏过程中需要考虑的关键因素。我们的研究表明，通过仔细控制嵌入器的偏见而非单纯提高其公平性，可以更好地实现公平的RAG系统。 

---
# What is a Good Question? Utility Estimation with LLM-based Simulations 

**Title (ZH)**: 什么是好问题？基于大语言模型的模拟实用性评估 

**Authors**: Dong-Ho Lee, Hyundong Cho, Jonathan May, Jay Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2502.17383)  

**Abstract**: Asking questions is a fundamental aspect of learning that facilitates deeper understanding. However, characterizing and crafting questions that effectively improve learning remains elusive. To address this gap, we propose QUEST (Question Utility Estimation with Simulated Tests). QUEST simulates a learning environment that enables the quantification of a question's utility based on its direct impact on improving learning outcomes. Furthermore, we can identify high-utility questions and use them to fine-tune question generation models with rejection sampling. We find that questions generated by models trained with rejection sampling based on question utility result in exam scores that are higher by at least 20% than those from specialized prompting grounded on educational objectives literature and models fine-tuned with indirect measures of question quality, such as saliency and expected information gain. 

**Abstract (ZH)**: 提出问题是一种基本的学习方法，能够促进更深入的理解。然而，如何准确描述和构造能够有效提高学习效果的问题仍然具有挑战性。为了解决这一问题，我们提出了一种名为QUEST（Question Utility Estimation with Simulated Tests）的方法。QUEST 模拟了一个学习环境，使我们可以基于问题对提高学习成果的直接影响来量化问题的价值。此外，我们可以通过使用具有拒绝采样方法的高价值问题来精调问题生成模型。我们发现，利用问题价值为基础的拒绝采样方法进行训练生成的问题，相比于基于教育目标文献的专业化提示和利用间接衡量问题质量（如显著性和预期信息增益）进行精调的模型生成的问题，考试得分至少高出20%。 

---
# Bridging Gaps in Natural Language Processing for Yorùbá: A Systematic Review of a Decade of Progress and Prospects 

**Title (ZH)**: 桥接约鲁巴语自然语言处理中的缺口：十年进展与展望的系统综述 

**Authors**: Toheeb A. Jimoh, Tabea De Wille, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17364)  

**Abstract**: Natural Language Processing (NLP) is becoming a dominant subset of artificial intelligence as the need to help machines understand human language looks indispensable. Several NLP applications are ubiquitous, partly due to the myriads of datasets being churned out daily through mediums like social networking sites. However, the growing development has not been evident in most African languages due to the persisting resource limitation, among other issues. Yorùbá language, a tonal and morphologically rich African language, suffers a similar fate, resulting in limited NLP usage. To encourage further research towards improving this situation, this systematic literature review aims to comprehensively analyse studies addressing NLP development for Yorùbá, identifying challenges, resources, techniques, and applications. A well-defined search string from a structured protocol was employed to search, select, and analyse 105 primary studies between 2014 and 2024 from reputable databases. The review highlights the scarcity of annotated corpora, limited availability of pre-trained language models, and linguistic challenges like tonal complexity and diacritic dependency as significant obstacles. It also revealed the prominent techniques, including rule-based methods, among others. The findings reveal a growing body of multilingual and monolingual resources, even though the field is constrained by socio-cultural factors such as code-switching and desertion of language for digital usage. This review synthesises existing research, providing a foundation for advancing NLP for Yorùbá and in African languages generally. It aims to guide future research by identifying gaps and opportunities, thereby contributing to the broader inclusion of Yorùbá and other under-resourced African languages in global NLP advancements. 

**Abstract (ZH)**: 自然语言处理（NLP）正逐渐成为人工智能（AI）研究中的一个核心部分，因为帮助机器理解和处理人类语言的需求变得不可或缺。许多NLP应用正在普及，部分原因是通过社交媒体等媒介每天都产生了大量的数据集。然而，非洲语言的NLP发展并没有显著增长，这主要是由于资源限制等问题的存在。约鲁巴语作为一种音调丰富、形态复杂的非洲语言，面临着类似的困境，导致NLP的应用受限。为了鼓励进一步的研究以改善这一状况，本系统文献综述旨在全面分析针对约鲁巴语的NLP发展的研究，识别挑战、资源、技术和应用。我们根据结构化协议定义了一个清晰的检索字符串，检索并分析了2014年至2024年间来自知名数据库的105篇主要研究文献。综述强调了标注语料库的稀缺性、预训练语言模型的有限可用性以及声调复杂性和标音符依赖性等语言挑战是重要的障碍。同时，该综述揭示了许多关键技术，包括基于规则的方法等。研究发现尽管存在如社会文化因素（如代码转换和语言数字使用的放弃）等限制因素，多语言和单语资源的数量正在增长。综述综合了现有的研究，为推进约鲁巴语和其他资源匮乏的非洲语言的NLP发展提供了基础。它旨在通过识别空白和机会来指导未来的研究，从而促进约鲁巴语及其他非洲语言在全球NLP进步中的更广泛纳入。 

---
# On Relation-Specific Neurons in Large Language Models 

**Title (ZH)**: 在大型语言模型中的关系专用神经元 

**Authors**: Yihong Liu, Runsheng Chen, Lea Hirlimann, Ahmad Dawar Hakimi, Mingyang Wang, Amir Hossein Kargaran, Sascha Rothe, François Yvon, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2502.17355)  

**Abstract**: In large language models (LLMs), certain neurons can store distinct pieces of knowledge learned during pretraining. While knowledge typically appears as a combination of relations and entities, it remains unclear whether some neurons focus on a relation itself -- independent of any entity. We hypothesize such neurons detect a relation in the input text and guide generation involving such a relation. To investigate this, we study the Llama-2 family on a chosen set of relations with a statistics-based method. Our experiments demonstrate the existence of relation-specific neurons. We measure the effect of selectively deactivating candidate neurons specific to relation $r$ on the LLM's ability to handle (1) facts whose relation is $r$ and (2) facts whose relation is a different relation $r' \neq r$. With respect to their capacity for encoding relation information, we give evidence for the following three properties of relation-specific neurons. $\textbf{(i) Neuron cumulativity.}$ The neurons for $r$ present a cumulative effect so that deactivating a larger portion of them results in the degradation of more facts in $r$. $\textbf{(ii) Neuron versatility.}$ Neurons can be shared across multiple closely related as well as less related relations. Some relation neurons transfer across languages. $\textbf{(iii) Neuron interference.}$ Deactivating neurons specific to one relation can improve LLM generation performance for facts of other relations. We will make our code publicly available at this https URL. 

**Abstract (ZH)**: 在大型语言模型（LLMs）中，某些神经元可以在预训练过程中存储特定的知识片段。尽管知识通常表现为关系和实体的组合，但尚不清楚是否有某些神经元专门关注特定的关系——而不依赖于任何实体。我们假设这些神经元可以检测输入文本中的关系，并引导涉及该关系的生成。为了验证这一假设，我们使用统计方法研究了Llama-2家族模型在选定的一组关系上的表现。我们的实验表明，存在专用于特定关系的神经元。我们测量了有选择地抑制特定于关系$r$的候选神经元对LLM处理以下两方面的能力的影响：(1) 关系为$r$的事实；(2) 关系为不同关系$r' \neq r$的事实。关于编码关系信息的能力，我们提供了以下三个关于专用于特定关系的神经元性质的证据。**（i）神经元累积性。** 关系$r$的神经元显示出累积效应，即抑制更多数量的这些神经元会导致更多关系为$r$的事实受到负面影响。**（ii）神经元多功能性。** 神经元可以在多个密切相关甚至较不相关的关系间共享。一些关系特异性神经元可以在不同语言间传递。**（iii）神经元干扰。** 抑制某一关系的特异性神经元可以提高LLM对其他关系事实生成表现的效果。我们将在以下网址公开我们的代码：[此链接](this https URL)。 

---
# Mutual Reinforcement of LLM Dialogue Synthesis and Summarization Capabilities for Few-Shot Dialogue Summarization 

**Title (ZH)**: 少样本对话总结中LLM对话合成与摘要能力的相互强化 

**Authors**: Yen-Ju Lu, Ting-Yao Hu, Hema Swetha Koppula, Hadi Pouransari, Jen-Hao Rick Chang, Yin Xia, Xiang Kong, Qi Zhu, Simon Wang, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.17328)  

**Abstract**: In this work, we propose Mutual Reinforcing Data Synthesis (MRDS) within LLMs to improve few-shot dialogue summarization task. Unlike prior methods that require external knowledge, we mutually reinforce the LLMś dialogue synthesis and summarization capabilities, allowing them to complement each other during training and enhance overall performances. The dialogue synthesis capability is enhanced by directed preference optimization with preference scoring from summarization capability. The summarization capability is enhanced by the additional high quality dialogue-summary paired data produced by the dialogue synthesis capability. By leveraging the proposed MRDS mechanism, we elicit the internal knowledge of LLM in the format of synthetic data, and use it to augment the few-shot real training dataset. Empirical results demonstrate that our method improves dialogue summarization, achieving a 1.5% increase in ROUGE scores and a 0.3% improvement in BERT scores in few-shot settings. Furthermore, our method attains the highest average scores in human evaluations, surpassing both the pre-trained models and the baselines fine-tuned solely for summarization tasks. 

**Abstract (ZH)**: 在本文中，我们提出了在大语言模型（LLMs）内部实现互强化的数据合成（Mutual Reinforcing Data Synthesis, MRDS）方法，以提升少样本对话总结任务的表现。与需要外部知识的先前方法不同，我们通过相互强化LLMs的对话合成能力和总结能力，在训练过程中使它们相互补充，从而提升整体性能。对话合成能力通过基于总结能力的偏好评分进行有向偏好优化而得到增强。总结能力则通过由对话合成能力生成的高质量对话总结配对数据得到增强。利用提出的MRDS机制，我们促使LLMs在合成数据格式下内化知识，并将其用于扩充少样本真实训练数据集。实验结果表明，我们的方法可以提高对话总结性能，在少样本设置中实现了ROUGE分数1.5%的提升和BERT分数0.3%的改善。此外，我们的方法在人工评估中获得了最高的平均得分，超过了预训练模型以及仅针对总结任务微调的基线模型。 

---
# Turning Conversations into Workflows: A Framework to Extract and Evaluate Dialog Workflows for Service AI Agents 

**Title (ZH)**: 将对话转化为工作流：一种提取和服务评估对话工作流的框架 

**Authors**: Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Caiming Xiong, Shiva Kumar Pentyala, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17321)  

**Abstract**: Automated service agents require well-structured workflows to provide consistent and accurate responses to customer queries. However, these workflows are often undocumented, and their automatic extraction from conversations remains unexplored. In this work, we present a novel framework for extracting and evaluating dialog workflows from historical interactions. Our extraction process consists of two key stages: (1) a retrieval step to select relevant conversations based on key procedural elements, and (2) a structured workflow generation process using a question-answer-based chain-of-thought (QA-CoT) prompting. To comprehensively assess the quality of extracted workflows, we introduce an automated agent and customer bots simulation framework that measures their effectiveness in resolving customer issues. Extensive experiments on the ABCD and SynthABCD datasets demonstrate that our QA-CoT technique improves workflow extraction by 12.16\% in average macro accuracy over the baseline. Moreover, our evaluation method closely aligns with human assessments, providing a reliable and scalable framework for future research. 

**Abstract (ZH)**: 自动化服务代理需要结构化的流程以提供一致且准确的客户查询响应。然而，这些流程往往缺乏文档记录，从历史对话中自动提取这些流程的方法也尚未被探索。本文提出了一种新颖的框架，用于从历史交互中提取和评估对话流程。我们的提取过程包含两个关键阶段：（1）检索步骤，基于关键程序元素选择相关对话；（2）使用问题-答案链式思维（QA-CoT）提示的结构化流程生成过程。为了全面评估提取的流程质量，我们引入了一种自动代理和客户机器人模拟框架，该框架通过衡量它们解决客户问题的效果来评估流程的有效性。在ABCD和SynthABCD数据集上的广泛实验表明，与基线相比，我们的QA-CoT技术在平均宏准确率方面提高了12.16%。此外，我们的评估方法与人工评估高度一致，为未来研究提供了一个可靠且可扩展的框架。 

---
# HIPPO: Enhancing the Table Understanding Capability of Large Language Models through Hybrid-Modal Preference Optimization 

**Title (ZH)**: HIPPO：通过混合模态偏好优化增强大型语言模型的表格理解能力 

**Authors**: Zhenghao Liu, Haolan Wang, Xinze Li, Qiushi Xiong, Xiaocui Yang, Yu Gu, Yukun Yan, Qi Shi, Fangfang Li, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.17315)  

**Abstract**: Tabular data contains rich structural semantics and plays a crucial role in organizing and manipulating information. To better capture these structural semantics, this paper introduces the HybrId-modal Preference oPtimizatiOn (HIPPO) model, which represents tables using both text and image, and optimizes MLLMs to effectively learn more comprehensive table information from these multiple modalities. Specifically, HIPPO samples model responses from hybrid-modal table representations and designs a modality-consistent sampling strategy to enhance response diversity and mitigate modality bias during DPO training. Experimental results on table question answering and table fact verification tasks demonstrate the effectiveness of HIPPO, achieving a 4% improvement over various table reasoning models. Further analysis reveals that HIPPO not only enhances reasoning abilities based on unimodal table representations but also facilitates the extraction of crucial and distinct semantics from different modal representations. All data and codes are available at this https URL. 

**Abstract (ZH)**: 表格数据富含丰富的结构语义，在组织和操作信息方面发挥着关键作用。为了更好地捕捉这些结构语义，本文引入了HybrId-modal Preference oPtimizatiOn（HIPPO）模型，该模型采用文本和图像两种方式表示表格，并通过优化多模态语言模型来从这些多种模态中学习更加全面的表格信息。具体而言，HIPPO 从混合模态的表格表示中抽样模型响应，并设计了一种模态一致的采样策略，以在 DPO 训练过程中增强响应的多样性和减轻模态偏差。在表格问答和表格事实验证任务上的实验结果表明，HIPPO 的有效性，相比各种表格推理模型，其性能提高了4%。进一步的分析表明，HIPPO 不仅能够在单模态表格表示的基础上增强推理能力，还能够促进从不同模态表示中提取关键且独特的语义。所有数据和代码可从以下链接获得：[提供链接]。 

---
# Implicit Word Reordering with Knowledge Distillation for Cross-Lingual Dependency Parsing 

**Title (ZH)**: 基于知识蒸馏的隐式词重排序在跨语言依存句法分析中的应用 

**Authors**: Zhuoran Li, Chunming Hu, Junfan Chen, Zhijun Chen, Richong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17308)  

**Abstract**: Word order difference between source and target languages is a major obstacle to cross-lingual transfer, especially in the dependency parsing task. Current works are mostly based on order-agnostic models or word reordering to mitigate this problem. However, such methods either do not leverage grammatical information naturally contained in word order or are computationally expensive as the permutation space grows exponentially with the sentence length. Moreover, the reordered source sentence with an unnatural word order may be a form of noising that harms the model learning. To this end, we propose an Implicit Word Reordering framework with Knowledge Distillation (IWR-KD). This framework is inspired by that deep networks are good at learning feature linearization corresponding to meaningful data transformation, e.g. word reordering. To realize this idea, we introduce a knowledge distillation framework composed of a word-reordering teacher model and a dependency parsing student model. We verify our proposed method on Universal Dependency Treebanks across 31 different languages and show it outperforms a series of competitors, together with experimental analysis to illustrate how our method works towards training a robust parser. 

**Abstract (ZH)**: 源语言和目标语言的词序差异是跨语言迁移中的主要障碍，特别是在依赖解析任务中。目前的研究主要依赖于无序模型或词重排序来缓解这一问题。然而，这些方法要么未能充分利用词汇顺序中自然包含的语法信息，要么随着句子长度的增加，计算成本会随着排列空间的指数增长变得高昂。此外，用不自然的词序重新排列的源句子可能是一种噪声形式，危害模型的学习。基于此，我们提出了一种基于知识蒸馏的隐式词重排序框架（IWR-KD）。这一框架受到深度网络善于学习与有意义的数据变换（例如词重排序）对应的功能线性化的启发。为了实现这一理念，我们引入了一个由词重排序教师模型和依赖解析学生模型组成的知识蒸馏框架。我们在31种不同语言的通用依赖树库上验证了所提出的模型，并展示了它在与一系列竞争对手的对比中表现出色，同时通过实验分析展示了该方法如何有助于训练一个鲁棒的解析器。 

---
# `Generalization is hallucination' through the lens of tensor completions 

**Title (ZH)**: “泛化即幻觉”：张量补全视角下的理解 

**Authors**: Liang Ze Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17305)  

**Abstract**: In this short position paper, we introduce tensor completions and artifacts and make the case that they are a useful theoretical framework for understanding certain types of hallucinations and generalizations in language models. 

**Abstract (ZH)**: 在本文简要的意见论文中，我们介绍了张量补全及其伪影，并论证了它们是一种有助于理解语言模型中某些类型幻觉和泛化现象的有用理论框架。 

---
# Child vs. machine language learning: Can the logical structure of human language unleash LLMs? 

**Title (ZH)**: 儿童与机器的语言学习：人类语言的逻辑结构能否激发大规模语言模型（LLMs）？ 

**Authors**: Uli Sauerland, Celia Matthaei, Felix Salfner  

**Link**: [PDF](https://arxiv.org/pdf/2502.17304)  

**Abstract**: We argue that human language learning proceeds in a manner that is different in nature from current approaches to training LLMs, predicting a difference in learning biases. We then present evidence from German plural formation by LLMs that confirm our hypothesis that even very powerful implementations produce results that miss aspects of the logic inherent to language that humans have no problem with. We conclude that attention to the different structures of human language and artificial neural networks is likely to be an avenue to improve LLM performance. 

**Abstract (ZH)**: 我们认为人类语言学习的过程与当前训练大规模语言模型（LLMs）的方法在本质上有所不同，这导致了学习偏见的不同。随后，我们通过对LLMs在德语复数形成方面的实证研究，证实了即使是最强大的实现也未能捕捉到人类在语言逻辑方面没有困难的某些方面。我们得出结论，关注人类语言和人工神经网络之间不同的结构可能是提升LLM性能的一个途径。 

---
# Improving the Inclusivity of Dutch Speech Recognition by Fine-tuning Whisper on the JASMIN-CGN Corpus 

**Title (ZH)**: 通过在JASMIN-CGN语料库上微调Whisper来提高荷兰语语音识别的包容性 

**Authors**: Golshid Shekoufandeh, Paul Boersma, Antal van den Bosch  

**Link**: [PDF](https://arxiv.org/pdf/2502.17284)  

**Abstract**: We test and study the variation in speech recognition of fine-tuned versions of the Whisper model on child, elderly and non-native Dutch speech from the JASMIN-CGN corpus. Our primary goal is to evaluate how speakers' age and linguistic background influence Whisper's performance. Whisper achieves varying Word Error Rates (WER) when fine-tuned on subpopulations of specific ages and linguistic backgrounds. Fine-tuned performance is remarkably better than zero-shot performance, achieving a relative reduction in WER of 81% for native children, 72% for non-native children, 67% for non-native adults, and 65% for native elderly people. Our findings underscore the importance of training speech recognition models like Whisper on underrepresented subpopulations such as children, the elderly, and non-native speakers. 

**Abstract (ZH)**: 我们测试并研究了细调的Whisper模型在JASMIN-CGN语料库中的儿童、老年人和非母语荷兰语speech上的语音识别变异情况。我们的主要目标是评估说话者的年龄和语言背景如何影响Whisper的性能。当Whisper对特定年龄和语言背景的子人群进行细调时，其词错误率（WER）会有所不同。与零样本性能相比，细调性能显著提升，对于母语儿童，WER降低了81%；对于非母语儿童，降低了72%；对于非母语成人，降低了67%；对于母语老年人，降低了65%。我们的研究结果强调了在如Whisper这样的语音识别模型的训练过程中，需要包含代表性不足的子人群，如儿童、老年人和非母语说话者的重要性。 

---
# Capability Instruction Tuning: A New Paradigm for Dynamic LLM Routing 

**Title (ZH)**: 能力指令调优：一种新的动态大模型路由范式 

**Authors**: Yi-Kai Zhang, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2502.17282)  

**Abstract**: Large Language Models (LLMs) have demonstrated human-like instruction-following abilities, particularly those exceeding 100 billion parameters. The combined capability of some smaller, resource-friendly LLMs can address most of the instructions that larger LLMs excel at. In this work, we explore how to route the best-performing LLM for each instruction to achieve better overall performance. We develop a new paradigm, constructing capability instructions with model capability representation, user instruction, and performance inquiry prompts to assess the performance. To learn from capability instructions, we introduce a new end-to-end framework called Model Selection with Aptitude Test (Model-SAT), which generates positive and negative samples based on what different models perform well or struggle with. Model-SAT uses a model capability encoder that extends its model representation to a lightweight LLM. Our experiments show that Model-SAT understands the performance dimensions of candidate models and provides the probabilities of their capability to handle various instructions. Additionally, during deployment, a new model can quickly infer its aptitude test results across 50 tasks, each with 20 shots. Model-SAT performs state-of-the-art model routing without candidate inference and in real-world new model-released scenarios. The code is available at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了类似人类的指令遵循能力，特别是在参数超过100亿的模型中表现尤为突出。一些较小且资源友好的LLMs的综合能力可以应对大多数大型LLMs擅长的指令。在这项工作中，我们探讨了如何将最适合每条指令的LLM进行路由，以实现更好的总体性能。我们提出了一种新的范式，通过使用模型能力表示、用户指令和性能询问指令构建能力指令，来评估性能。为了从能力指令中学习，我们引入了一个新的端到端框架，称为能力测试下的模型选择（Model-SAT），该框架根据不同模型擅长或难以处理的内容生成正负样本。Model-SAT 使用一个模型能力编码器，将其模型表示扩展为一个轻量级的LLM。我们的实验表明，Model-SAT 能够理解候选模型的性能维度，并提供它们处理各种指令的能力概率。此外，在部署过程中，一个新的模型可以在不到两分钟的时间内快速推理出其在50个任务中的能力测试结果，每个任务有20个样本。Model-SAT 在无需候选模型推理的情况下实现了最先进的模型路由，并且适用于现实世界中新模型发布的情景。相关代码可在以下链接获取：[提供链接] 

---
# Extracting domain-specific terms using contextual word embeddings 

**Title (ZH)**: 使用上下文词嵌入提取领域专用术语 

**Authors**: Andraž Repar, Nada Lavrač, Senja Pollak  

**Link**: [PDF](https://arxiv.org/pdf/2502.17278)  

**Abstract**: Automated terminology extraction refers to the task of extracting meaningful terms from domain-specific texts. This paper proposes a novel machine learning approach to terminology extraction, which combines features from traditional term extraction systems with novel contextual features derived from contextual word embeddings. Instead of using a predefined list of part-of-speech patterns, we first analyse a new term-annotated corpus RSDO5 for the Slovenian language and devise a set of rules for term candidate selection and then generate statistical, linguistic and context-based features. We use a support-vector machine algorithm to train a classification model, evaluate it on the four domains (biomechanics, linguistics, chemistry, veterinary) of the RSDO5 corpus and compare the results with state-of-art term extraction approaches for the Slovenian language. Our approach provides significant improvements in terms of F1 score over the previous state-of-the-art, which proves that contextual word embeddings are valuable for improving term extraction. 

**Abstract (ZH)**: 自动术语提取是指从特定领域的文本中提取有意义术语的任务。本文提出了一种新的机器学习方法，将传统术语提取系统的特征与从上下文词嵌入中衍生的新语境特征相结合。我们并没有使用预定义的词性模式列表，而是首先分析了一个新的术语标注语料库RSDO5（斯洛文尼亚语），并为此语言设计了一套术语候选选择规则，然后生成了统计学、语言学和基于上下文的特征。我们使用支持向量机算法训练分类模型，并在RSDO5语料库的四个领域（生物力学、语言学、化学、兽医学）进行评估，还将结果与当前最先进的斯洛文尼亚语术语提取方法进行了比较。我们的方法在F1分数方面提供了显著的改进，这证明了上下文词嵌入对于提高术语提取性能的价值。 

---
# MonoTODia: Translating Monologue Requests to Task-Oriented Dialogues 

**Title (ZH)**: MonoTODia: 将独白请求转化为任务导向对话 

**Authors**: Sebastian Steindl, Ulrich Schäfer, Bernd Ludwig  

**Link**: [PDF](https://arxiv.org/pdf/2502.17268)  

**Abstract**: Data scarcity is one of the main problems when it comes to real-world applications of transformer-based models. This is especially evident for task-oriented dialogue (TOD) systems, which require specialized datasets, that are usually not readily available. This can hinder companies from adding TOD systems to their services. This study therefore investigates a novel approach to sourcing annotated dialogues from existing German monologue material. Focusing on a real-world example, we investigate whether these monologues can be transformed into dialogue formats suitable for training TOD systems. We show the approach with the concrete example of a company specializing in travel bookings via e-mail. We fine-tune state-of-the-art Large Language Models for the task of rewriting e-mails as dialogues and annotating them. To ensure the quality and validity of the generated data, we employ crowd workers to evaluate the dialogues across multiple criteria and to provide gold-standard annotations for the test dataset. We further evaluate the usefulness of the dialogues for training TOD systems. Our evaluation shows that the dialogues and annotations are of high quality and can serve as a valuable starting point for training TOD systems. Finally, we make the annotated dataset publicly available to foster future research. 

**Abstract (ZH)**: 数据稀缺是将基于变压器的模型应用于实际应用场景时面临的主要问题之一。这在任务导向对话（TOD）系统中尤为明显，TOD系统需要专门的数据集，而这些数据集通常难以获取。这可能阻碍公司将其TOD系统纳入其服务中。因此，本研究探讨了一种新颖的方法，即从现有的德语独白材料中获取标注对话。通过一个实际案例，我们研究这些独白是否可以转换为适用于训练TOD系统的对话格式。我们以一家专门通过电子邮件进行旅行预订的公司为例，对这一方法进行了具体说明。我们将最新的大型语言模型微调以实现邮件重写为对话，并对其进行标注。为确保生成数据的质量和有效性，我们采用了众包工作者从多个角度评估对话，并为测试数据集提供金标准标注。我们进一步评估了这些对话对训练TOD系统的有用性。评估结果显示，这些对话和标注质量很高，可以作为训练TOD系统的宝贵起点。最后，我们将标注的数据集公开，以促进未来的研究。 

---
# Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective 

**Title (ZH)**: 基于聚类分析的大型语言模型下游性能扩展研究 

**Authors**: Chengyin Xu, Kaiyuan Chen, Xiao Li, Ke Shen, Chenggang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17262)  

**Abstract**: The rapid advancements in computing dramatically increase the scale and cost of training Large Language Models (LLMs). Accurately predicting downstream task performance prior to model training is crucial for efficient resource allocation, yet remains challenging due to two primary constraints: (1) the "emergence phenomenon", wherein downstream performance metrics become meaningful only after extensive training, which limits the ability to use smaller models for prediction; (2) Uneven task difficulty distributions and the absence of consistent scaling laws, resulting in substantial metric variability. Existing performance prediction methods suffer from limited accuracy and reliability, thereby impeding the assessment of potential LLM capabilities. To address these challenges, we propose a Clustering-On-Difficulty (COD) downstream performance prediction framework. COD first constructs a predictable support subset by clustering tasks based on difficulty features, strategically excluding non-emergent and non-scalable clusters. The scores on the selected subset serve as effective intermediate predictors of downstream performance on the full evaluation set. With theoretical support, we derive a mapping function that transforms performance metrics from the predictable subset to the full evaluation set, thereby ensuring accurate extrapolation of LLM downstream performance. The proposed method has been applied to predict performance scaling for a 70B LLM, providing actionable insights for training resource allocation and assisting in monitoring the training process. Notably, COD achieves remarkable predictive accuracy on the 70B LLM by leveraging an ensemble of small models, demonstrating an absolute mean deviation of 1.36% across eight important LLM evaluation benchmarks. 

**Abstract (ZH)**: 计算领域的快速进步大幅增加了训练大规模语言模型（LLM）的规模和成本。在模型训练之前准确预测下游任务性能对于有效分配资源至关重要，但由于两个主要限制，这一任务仍然具有挑战性：（1）“涌现现象”，即下游性能指标只有在经过大量训练后才变得有意义，这限制了使用较小模型进行预测的能力；（2）任务难度分布的不均衡和缺乏一致的缩放定律，导致指标的大幅波动。现有的性能预测方法在准确性和可靠性方面存在局限，因此阻碍了评估潜在的语言模型能力。为应对这些挑战，我们提出了一种基于难度聚类（COD，Clustering-On-Difficulty）的下游性能预测框架。该框架首先通过基于难度特征对任务进行聚类，构建一个可预测的支持集，战略性地排除那些无法涌现和不具有可扩展性的聚类。选择的子集的得分可作为整套评估集下游性能的有效中间预测指标。在理论支持下，我们推导出一个映射函数，将可预测子集的性能指标转换到整套评估集，从而确保对大规模语言模型下游性能进行准确外推。所提出的方法已在70B参数语言模型上应用，提供了关于训练资源分配的实际指导，并有助于监控训练过程。值得注意的是，通过结合多种小型模型，COD 在70B规模的10个重要语言模型评估基准上实现了高达1.36%的绝对均方误差，显示出显著的预测准确性。 

---
# MULTITAT: Benchmarking Multilingual Table-and-Text Question Answering 

**Title (ZH)**: MULTITAT：多语言表格和文本问答基准测试 

**Authors**: Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2502.17253)  

**Abstract**: Question answering on the hybrid context of tables and text (TATQA) is a critical task, with broad applications in data-intensive domains. However, existing TATQA datasets are limited to English, leading to several drawbacks: (i) They overlook the challenges of multilingual TAT-QA and cannot assess model performance in the multilingual setting. (ii) They do not reflect real-world scenarios where tables and texts frequently appear in non-English languages. To address the limitations, we propose the first multilingual TATQA dataset (MULTITAT). Specifically, we sample data from 3 mainstream TATQA datasets and translate it into 10 diverse languages. To align the model TATQA capabilities in English with other languages, we develop a baseline, Ours. Experimental results reveal that the performance on non-English data in MULTITAT drops by an average of 19.4% compared to English, proving the necessity of MULTITAT. We further analyze the reasons for this performance gap. Furthermore, Ours outperforms other baselines by an average of 3.3, demonstrating its effectiveness. 

**Abstract (ZH)**: 多模态上下文下的问答任务（TATQA）对于表格和文本的混合内容进行问答是一个关键任务，广泛应用于数据密集型领域。然而，现有的TATQA数据集仅限于英语，这带来了几个局限性：（i）它们忽视了多语言TAT-QA的挑战，无法评估模型在多语言环境中的性能。（ii）它们没有反映现实世界的情况，其中表格和文本经常以非英语语言的形式出现。为了解决这些局限性，我们提出了第一个多语言TATQA数据集（MULTITAT）。具体而言，我们从3个主流的TATQA数据集抽取数据，并将其翻译成10种不同的语言。为了使模型在多语言中的TATQA能力与英语中的能力一致，我们开发了一个基线模型——Ours。实验结果表明，与英语数据相比，MULTITAT上的非英语数据性能平均下降了19.4%，这证明了MULTITAT的必要性。我们进一步分析了这种性能差异的原因。此外，Ours在与其它基线模型的比较中，平均性能高出3.3%，这证明了其有效性。 

---
# Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction 

**Title (ZH)**: Baichuan-Audio：端到端语音交互的统一框架 

**Authors**: Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, Jianhua Xu, Haoze Sun, Zenan Zhou, Weipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17239)  

**Abstract**: We introduce Baichuan-Audio, an end-to-end audio large language model that seamlessly integrates audio understanding and generation. It features a text-guided aligned speech generation mechanism, enabling real-time speech interaction with both comprehension and generation capabilities. Baichuan-Audio leverages a pre-trained ASR model, followed by multi-codebook discretization of speech at a frame rate of 12.5 Hz. This multi-codebook setup ensures that speech tokens retain both semantic and acoustic information. To further enhance modeling, an independent audio head is employed to process audio tokens, effectively capturing their unique characteristics. To mitigate the loss of intelligence during pre-training and preserve the original capabilities of the LLM, we propose a two-stage pre-training strategy that maintains language understanding while enhancing audio modeling. Following alignment, the model excels in real-time speech-based conversation and exhibits outstanding question-answering capabilities, demonstrating its versatility and efficiency. The proposed model demonstrates superior performance in real-time spoken dialogue and exhibits strong question-answering abilities. Our code, model and training data are available at this https URL 

**Abstract (ZH)**: 我们介绍了Baichuan-Audio，这是一款端到端的音频大规模语言模型，能够无缝整合音频理解和生成。该模型具备文本引导的对齐语音生成机制，能够实现实时语音交互，具备理解和生成的能力。Baichuan-Audio 利用一个预训练的自动语音识别（ASR）模型，并以12.5 Hz的帧率对语音进行多码本分档处理。这种多码本设置确保语音标记保留了语义和声学信息。为进一步提高模型能力，我们引入了独立的音频头来处理音频标记，有效地捕捉它们的独特特征。为了减轻预训练过程中情报损失并保留大语言模型（LLM）的原始能力，我们提出了一种两阶段预训练策略，既能保持语言理解能力，又能增强音频建模能力。经过对齐处理后，该模型在实时语音对话中表现优异，并展示出了出色的问题回答能力，证明了其灵活性和高效性。所提出的模型在实时语音对话任务中表现出色，并展示了强大的问题回答能力。我们的代码、模型和训练数据可在以下链接获取：[提供链接] 

---
# CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought 

**Title (ZH)**: CoT-UQ: 在链式思维辅助下提升大语言模型响应层面的不确定性量化 

**Authors**: Boxuan Zhang, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17214)  

**Abstract**: Large language models (LLMs) excel in many tasks but struggle to accurately quantify uncertainty in their generated responses. This limitation makes it challenging to detect misinformation and ensure reliable decision-making. Existing uncertainty quantification (UQ) methods for LLMs are primarily prompt-wise rather than response-wise, often requiring multiple response samples, which incurs high computational costs. Moreover, LLMs have been shown to be overconfident, particularly when using reasoning steps to derive their answers. In this work, we propose CoT-UQ, a response-wise UQ framework that integrates LLMs' inherent reasoning capabilities through Chain-of-Thought (CoT) into the UQ process. CoT-UQ captures critical information during inference by extracting keywords from each reasoning step and assessing their importance to the final answer. This key reasoning information is then aggregated to produce a final uncertainty estimate. We conduct extensive experiments based on LLaMA Family with model sizes varying from 8B to 13B across logical and mathematical reasoning tasks. Experimental results demonstrate that CoT-UQ significantly outperforms existing UQ methods, achieving an average improvement of 5.9% AUROC compared to current UQ methods. The code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多任务上表现优异，但在准确量化其生成响应中的不确定性方面存在局限性。这种局限性使得检测 misinformation 并确保可靠决策变得困难。现有的 LLM 不确定性量化（UQ）方法主要基于提示（prompt）而非响应，通常需要生成多个响应样本，这带来了高昂的计算成本。此外，研究表明，LLMs 在使用推理步骤得出答案时往往会表现出过度自信。在本项工作中，我们提出了一种名为 CoT-UQ 的响应级不确定性量化框架，该框架通过 Chain-of-Thought（CoT）将 LLM 的固有推理能力集成到不确定性量化过程中。CoT-UQ 通过从每个推理步骤中提取关键词并评估它们对最终答案的重要性，在推断过程中捕捉关键信息。这些关键的推理信息随后被汇总以生成最终的不确定性估计。我们基于 LLaMA 家族模型，在逻辑推理和数学推理任务中进行了广泛的实验，模型大小从 8B 到 13B 不等。实验结果表明，CoT-UQ 显著优于现有方法，在 AUROC 指标上平均提高了 5.9%。代码可在此处访问：this https URL。 

---
# Order Matters: Investigate the Position Bias in Multi-constraint Instruction Following 

**Title (ZH)**: 顺序有讲究：探究多约束指令跟随中的位置偏差 

**Authors**: Jie Zeng, Qianyu He, Qingyu Ren, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17204)  

**Abstract**: Real-world instructions with multiple constraints pose a significant challenge to existing large language models (LLMs). An observation is that the LLMs exhibit dramatic performance fluctuation when disturbing the order of the incorporated constraints. Yet, none of the existing works has systematically investigated this position bias problem in the field of multi-constraint instruction following. To bridge this gap, we design a probing task where we quantitatively measure the difficulty distribution of the constraints by a novel Difficulty Distribution Index (CDDI). Through the experimental results, we find that LLMs are more performant when presented with the constraints in a ``hard-to-easy'' order. This preference can be generalized to LLMs with different architecture or different sizes of parameters. Additionally, we conduct an explanation study, providing an intuitive insight into the correlation between the LLM's attention and constraint orders. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 现实世界中的多约束指令对现有大型语言模型（LLMs）构成了重大挑战。一个观察结果是，当扰动嵌入约束的顺序时，LLMs 的性能会出现显著波动。然而，现有文献中几乎没有系统地研究多约束指令遵循领域的这种位置偏移问题。为了弥合这一差距，我们设计了一项探针任务，通过引入一个新的难度分布指数（CDDI）来定量测量约束的难度分布。通过实验结果，我们发现当以“难到易”的顺序呈现约束时，LLMs 的性能更好。这种偏好可以泛化到具有不同架构或不同参数规模的LLMs。此外，我们还进行了一个解释性研究，提供了LLMs关注与约束顺序之间关系的直观洞察。我们的代码和数据集可在以下网址公开获取：this https URL。 

---
# Evaluating Expert Contributions in a MoE LLM for Quiz-Based Tasks 

**Title (ZH)**: 评估在基于 Quiz 的任务中 MoE 联邦语言模型中专家贡献的方法 

**Authors**: Andrei Chernov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17187)  

**Abstract**: Recently, Large Language Models (LLMs) with Mixture of Experts (MoE) layers have gained significant attention. Currently, state-of-the-art LLMs utilize this architecture. There is a substantial amount of research on how to train such models and how to select hyperparameters for this architecture. However, there is a lack of studies focusing on post-evaluation analysis of MoE layer properties. In this paper, we take a first step toward closing this gap by evaluating expert contributions on the quiz-based MMLU benchmark. We show that most experts were never activated during inference on this benchmark. Additionally, the output distribution of gating networks is much closer to uniform than sparse. Finally, we demonstrate that the average performance of some experts within the same layer varies significantly. 

**Abstract (ZH)**: 近年来，带有混合专家（Mixture of Experts，MoE）层的大规模语言模型（Large Language Models，LLMs）引起了广泛关注。目前，最先进的LLMs采用这种架构。已有大量的研究关注如何训练这类模型以及如何为这种架构选择超参数。然而，关于MoE层特性的后评估分析仍缺乏相关研究。在本文中，我们首次对此进行了探索，通过在基于测验的MMLU基准上评估专家的贡献来填补这一空白。我们发现，在该基准的推理过程中，大多数专家从未被激活。此外，门控网络的输出分布与均匀分布更为接近，而非稀疏分布。最后，我们展示了同一层中某些专家的平均性能存在显著差异。 

---
# Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric 

**Title (ZH)**: 用于指令调优的数据多样性度量：系统分析与可靠指标 

**Authors**: Yuming Yang, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17184)  

**Abstract**: Data diversity is crucial for the instruction tuning of large language models. Existing studies have explored various diversity-aware data selection methods to construct high-quality datasets and enhance model performance. However, the fundamental problem of precisely defining and measuring data diversity remains underexplored, limiting clear guidance for data engineering. To address this, we systematically analyze 11 existing diversity measurement methods by assessing their correlation with model performance through extensive fine-tuning experiments. Our results indicate that a reliable diversity measure should properly account for both inter-sample differences and the information distribution in the sample space. Building on this, we propose NovelSum, a new diversity metric based on sample-level "novelty." Experiments on both simulated and real-world data show that NovelSum accurately captures diversity variations and achieves a 0.97 correlation with instruction-tuned model performance, highlighting its value in guiding data engineering practices. With NovelSum as an optimization objective, we further develop a greedy, diversity-oriented data selection strategy that outperforms existing approaches, validating both the effectiveness and practical significance of our metric. 

**Abstract (ZH)**: 数据多样性对于大型语言模型的指令调优至关重要。现有研究已探讨了多种多样性的数据选择方法以构建高质量的数据集并提升模型性能。然而，精确定义和衡量数据多样性的基本问题仍未得到充分探索，限制了数据工程的方向性指导。为解决这一问题，我们通过广泛的细调实验系统分析了11种现有的多样测量方法，并评估了它们与模型性能的相关性。结果显示，一个可靠的多样性度量应该恰当地考虑样本间的差异和样本空间内的信息分布。在此基础上，我们提出了基于样本级“新颖性”的新多样性度量方法——NovelSum。实验结果表明，NovelSum 能够准确捕获多样性变化，并与指令调优模型的性能达到0.97的相关性，强调了其指导数据工程实践的价值。借助NovelSum 作为优化目标，我们进一步开发了一种基于多样性的贪婪数据选择策略，该策略在多个方面的表现优于现有方法，验证了该度量方法的有效性和实践意义。 

---
# Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch 

**Title (ZH)**: Cheems：从零构建和评估中文奖励模型的实际指南 

**Authors**: Xueru Wen, Jie Lou, Zichao Li, Yaojie Lu, Xing Yu, Yuqiu Ji, Guohai Xu, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17173)  

**Abstract**: Reward models (RMs) are crucial for aligning large language models (LLMs) with human preferences. However, most RM research is centered on English and relies heavily on synthetic resources, which leads to limited and less reliable datasets and benchmarks for Chinese. To address this gap, we introduce CheemsBench, a fully human-annotated RM evaluation benchmark within Chinese contexts, and CheemsPreference, a large-scale and diverse preference dataset annotated through human-machine collaboration to support Chinese RM training. We systematically evaluate open-source discriminative and generative RMs on CheemsBench and observe significant limitations in their ability to capture human preferences in Chinese scenarios. Additionally, based on CheemsPreference, we construct an RM that achieves state-of-the-art performance on CheemsBench, demonstrating the necessity of human supervision in RM training. Our findings reveal that scaled AI-generated data struggles to fully capture human preferences, emphasizing the importance of high-quality human supervision in RM development. 

**Abstract (ZH)**: 奖励模型（RMs）对于使大型语言模型（LLMs）与人类偏好保持一致至关重要。然而，大多数RM研究主要集中在英语上，并且严重依赖合成资源，这导致了中文缺乏可靠的且有限的数据集和基准测试。为解决这一问题，我们提出了CheemsBench，这是一种完全由人工标注的RM评估基准，适用于中文上下文；同时，还提出了CheemsPreference，这是一种通过人机协作标注的大规模且多样的偏好数据集，用于支持中文RM的训练。我们系统地在CheemsBench上评估了开源的区分性和生成性RM，并观察到它们在捕捉中文情境下的人类偏好方面存在显著局限。此外，基于CheemsPreference，我们构建了一个在CheemsBench上达到最佳性能的RM，这表明RM训练中的人类监督的必要性。我们的研究发现揭示了扩展的人工智能生成数据在全面捕捉人类偏好方面存在局限，突显了高质量人类监督在RM开发中的重要性。 

---
# Logic Haystacks: Probing LLMs Long-Context Logical Reasoning (Without Easily Identifiable Unrelated Padding) 

**Title (ZH)**: 逻辑针堆：探究LLMs在无明显无关填充的长情境逻辑推理能力 

**Authors**: Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2502.17169)  

**Abstract**: Large language models demonstrate promising long context processing capabilities, with recent models touting context windows close to one million tokens. However, the evaluations supporting these claims often involve simple retrieval tasks or synthetic tasks padded with irrelevant text, which the models may easily detect and discard. In this work, we generate lengthy simplified English text with first-order logic representations spanning up to 2048 clauses (around 25k GPT-4 tokens). We formulate an evaluation task with evidence retrieval for contradiction detection. The long, homogeneous text is filled with distractors that are both hard to distinguish from relevant evidences and provably not interfering with them. Our evaluation of evidence retrieval shows that the effective context window is much smaller with realistic distractors, already crumbling at 128 clauses. 

**Abstract (ZH)**: 大规模语言模型展示了令人期待的长上下文处理能力，近期的一些模型甚至宣称具有接近一百万词的上下文窗口。然而，这些声明的支持性评估往往涉及简单的检索任务或填充了无关文本的合成任务，而模型通常很容易识别并忽略这些无关内容。在本项研究中，我们生成了长达2048个子句（约25000个GPT-4标记）的简化英语文本，并使用了一阶逻辑表示。我们设计了一项评估任务，该任务包括证据检索以检测矛盾。该长文本充满了难以区分真相关据的干扰项，并且在理论上不会干扰这些相关真据。我们的证据检索评估显示，在现实干扰项的作用下，有效上下文窗口要小得多，甚至在128个子句时就会变得不稳定。 

---
# JUREX-4E: Juridical Expert-Annotated Four-Element Knowledge Base for Legal Reasoning 

**Title (ZH)**: JUREX-4E：法律专家注释的四元素法律推理知识库 

**Authors**: Huanghai Liu, Quzhe Huang, Qingjing Chen, Yiran Hu, Jiayu Ma, Yun Liu, Weixing Shen, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17166)  

**Abstract**: The Four-Element Theory is a fundamental framework in criminal law, defining the constitution of crime through four dimensions: Subject, Object, Subjective aspect, and Objective aspect. This theory is widely referenced in legal reasoning, and many Large Language Models (LLMs) attempt to incorporate it when handling legal tasks. However, current approaches rely on LLMs' internal knowledge to incorporate this theory, often lacking completeness and representativeness. To address this limitation, we introduce JUREX-4E, an expert-annotated knowledge base covering 155 criminal charges. It is structured through a progressive hierarchical annotation framework that prioritizes legal source validity and employs diverse legal interpretation methods to ensure comprehensiveness and authority. We evaluate JUREX-4E on the Similar Charge Distinction task and apply it to Legal Case Retrieval, demonstrating its effectiveness in improving LLM performance. Experimental results validate the high quality of JUREX-4E and its substantial impact on downstream legal tasks, underscoring its potential for advancing legal AI applications. Code: this https URL 

**Abstract (ZH)**: 四元素理论是刑法中的基本框架，通过四个维度来定义犯罪构成：主体、客体、主观方面、客观方面。这一理论在法律推理中被广泛引用，许多大型语言模型（LLMs）在处理法律任务时尝试将其纳入。然而，目前的方法主要依赖于LLMs的内部知识来整合这一理论，往往缺乏完整性和代表性。为解决这一限制，我们引入了JUREX-4E，这是一个由专家注释的知识库，涵盖了155项刑事指控。该知识库通过一个逐步多层次的注释框架结构化，优先考虑法律源的合法性，并采用多种法律解释方法以确保全面性和权威性。我们通过相似指控区分任务评估了JUREX-4E，并将其应用于法律案例检索，证明了其在提高LLMs性能方面的有效性。实验结果验证了JUREX-4E的高度质量和其对下游法律任务的实质性影响，突显了其在推动法律人工智能应用方面的发展潜力。代码：[此链接] 

---
# MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation 

**Title (ZH)**: MEMERAG：一种跨语言端到端元评估基准，用于检索增强生成 

**Authors**: María Andrea Cruz Blandón, Jayasimha Talur, Bruno Charron, Dong Liu, Saab Mansour, Marcello Federico  

**Link**: [PDF](https://arxiv.org/pdf/2502.17163)  

**Abstract**: Automatic evaluation of retrieval augmented generation (RAG) systems relies on fine-grained dimensions like faithfulness and relevance, as judged by expert human annotators. Meta-evaluation benchmarks support the development of automatic evaluators that correlate well with human judgement. However, existing benchmarks predominantly focus on English or use translated data, which fails to capture cultural nuances. A native approach provides a better representation of the end user experience.
In this work, we develop a Multilingual End-to-end Meta-Evaluation RAG benchmark (MEMERAG). Our benchmark builds on the popular MIRACL dataset, using native-language questions and generating responses with diverse large language models (LLMs), which are then assessed by expert annotators for faithfulness and relevance. We describe our annotation process and show that it achieves high inter-annotator agreement. We then analyse the performance of the answer-generating LLMs across languages as per the human evaluators. Finally we apply the dataset to our main use-case which is to benchmark multilingual automatic evaluators (LLM-as-a-judge). We show that our benchmark can reliably identify improvements offered by advanced prompting techniques and LLMs. We release our benchmark to support the community developing accurate evaluation methods for multilingual RAG systems. 

**Abstract (ZH)**: 自动评估检索增强生成（RAG）系统依赖于专家人工标注者判断的细致维度，如忠实度和相关性。元评估基准支持与人类判断相关性良好的自动评估器的发展。然而，现有的基准测试主要集中在英语上或使用翻译的数据，这未能捕捉到文化细微差别。采用本土方法可以更好地反映最终用户体验。

在此项工作中，我们开发了一个多语言端到端元评估RAG基准（MEMERAG）。我们的基准建立在流行的MIRACL数据集之上，使用本族语言问题生成具有多样性的大型语言模型（LLM）的回答，然后由专家注释者根据忠实度和相关性进行评估。我们描述了我们的注释过程，并展示了其实现了高注释者间一致性。随后，我们分析了根据不同人类评估者的评估，LLM生成答案的性能随语言的变化。最后，我们将数据集应用于我们的主要用例，即对标多语言自动评估器（LLM作为评判者）进行基准测试。我们展示了我们的基准能够可靠地识别高级提示技术及LLM带来的改进。我们释放了这个基准数据集，以支持开发多语言RAG系统准确评估方法的社区。 

---
# Sentiment analysis of texts from social networks based on machine learning methods for monitoring public sentiment 

**Title (ZH)**: 基于机器学习方法的社会网络文本情感分析及其在监控公众 sentiment 中的应用 

**Authors**: Arsen Tolebay Nurlanuly  

**Link**: [PDF](https://arxiv.org/pdf/2502.17143)  

**Abstract**: A sentiment analysis system powered by machine learning was created in this study to improve real-time social network public opinion monitoring. For sophisticated sentiment identification, the suggested approach combines cutting-edge transformer-based architectures (DistilBERT, RoBERTa) with traditional machine learning models (Logistic Regression, SVM, Naive Bayes). The system achieved an accuracy of up to 80-85% using transformer models in real-world scenarios after being tested using both deep learning techniques and standard machine learning processes on annotated social media datasets. According to experimental results, deep learning models perform noticeably better than lexicon-based and conventional rule-based classifiers, lowering misclassification rates and enhancing the ability to recognize nuances like sarcasm. According to feature importance analysis, context tokens, sentiment-bearing keywords, and part-of-speech structure are essential for precise categorization. The findings confirm that AI-driven sentiment frameworks can provide a more adaptive and efficient approach to modern sentiment challenges. Despite the system's impressive performance, issues with computing overhead, data quality, and domain-specific terminology still exist. In order to monitor opinions on a broad scale, future research will investigate improving computing performance, extending coverage to various languages, and integrating real-time streaming APIs. The results demonstrate that governments, corporations, and social researchers looking for more in-depth understanding of public mood on digital platforms can find a reliable and adaptable answer in AI-powered sentiment analysis. 

**Abstract (ZH)**: 本研究创建了一个基于机器学习的情感分析系统，旨在提高实时社交媒体网络公众意见的监控能力。为了实现复杂的情感识别，建议的方法结合了最先进的基于变换器的架构（DistilBERT、RoBERTa）与传统的机器学习模型（逻辑回归、支持向量机、朴素贝叶斯）。经过对注释过的社交媒体数据集使用深度学习技术和传统机器学习过程进行测试后，在实际应用中，变换器模型实现了高达80-85%的准确率。根据实验结果，深度学习模型明显优于基于词典和传统规则的分类器，降低了分类误差率，提高了识别细微差别（例如讽刺）的能力。通过特征重要性分析表明，上下文标记、情感化的关键词以及词性结构对于精确分类至关重要。这些发现证实，以人工智能驱动的情感框架为现代情感挑战提供了更具适应性和效率的方法。尽管该系统表现出色，但计算开销、数据质量和领域特定术语仍是存在的问题。为了实现广泛的公众意见监控，未来研究将探讨提高计算性能、扩展到多种语言、并集成实时流式API。研究结果表明，政府、企业和社会研究人员可以找到基于人工智能的情感分析以获得数字平台上公众情绪的更深入理解的可靠且适应性强的解决方案。 

---
# Thus Spake Long-Context Large Language Model 

**Title (ZH)**: 因此，长上下文大型语言模型如是说 

**Authors**: Xiaoran Liu, Ruixiao Li, Mianqiu Huang, Zhigeng Liu, Yuerong Song, Qipeng Guo, Siyang He, Qiqi Wang, Linlin Li, Qun Liu, Yaqian Zhou, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17129)  

**Abstract**: Long context is an important topic in Natural Language Processing (NLP), running through the development of NLP architectures, and offers immense opportunities for Large Language Models (LLMs) giving LLMs the lifelong learning potential akin to humans. Unfortunately, the pursuit of a long context is accompanied by numerous obstacles. Nevertheless, long context remains a core competitive advantage for LLMs. In the past two years, the context length of LLMs has achieved a breakthrough extension to millions of tokens. Moreover, the research on long-context LLMs has expanded from length extrapolation to a comprehensive focus on architecture, infrastructure, training, and evaluation technologies.
Inspired by the symphonic poem, Thus Spake Zarathustra, we draw an analogy between the journey of extending the context of LLM and the attempts of humans to transcend its mortality. In this survey, We will illustrate how LLM struggles between the tremendous need for a longer context and its equal need to accept the fact that it is ultimately finite. To achieve this, we give a global picture of the lifecycle of long-context LLMs from four perspectives: architecture, infrastructure, training, and evaluation, showcasing the full spectrum of long-context technologies. At the end of this survey, we will present 10 unanswered questions currently faced by long-context LLMs. We hope this survey can serve as a systematic introduction to the research on long-context LLMs. 

**Abstract (ZH)**: 长文本是自然语言处理（NLP）中的一个重要课题，贯穿于NLP架构的发展历程，为大型语言模型（LLMs）提供了巨大的机会，赋予了LLMs类似于人类的终身学习潜力。然而，追求长文本也伴随着诸多挑战。尽管如此，长文本仍然是LLMs的核心竞争力之一。在过去两年里，LLMs的上下文长度已经突破性地扩展到了数百万词。此外，关于长文本LLMs的研究已从长度的外推扩展到全面关注架构、基础设施、训练和评估技术。

受到交响诗《查拉图斯特拉如是说》的启发，我们把扩展LLM上下文的旅程类比于人类试图超越其有限性的努力。本文将展示LLM在其对更长上下文需求巨大的同时，也必须接受其最终有限性的双重挑战。为此，我们将从架构、基础设施、训练和评估四个维度全面呈现长上下文LLM的生命周期，展示长上下文技术的全貌。在本文末尾，我们将提出10个目前未解答的关于长上下文LLM的问题。我们希望本文能够为长上下文LLM的研究提供一个系统性的介绍。 

---
# LettuceDetect: A Hallucination Detection Framework for RAG Applications 

**Title (ZH)**: LettuceDetect：一种针对RAG应用的幻觉检测框架 

**Authors**: Ádám Kovács, Gábor Recski  

**Link**: [PDF](https://arxiv.org/pdf/2502.17125)  

**Abstract**: Retrieval Augmented Generation (RAG) systems remain vulnerable to hallucinated answers despite incorporating external knowledge sources. We present LettuceDetect a framework that addresses two critical limitations in existing hallucination detection methods: (1) the context window constraints of traditional encoder-based methods, and (2) the computational inefficiency of LLM based approaches. Building on ModernBERT's extended context capabilities (up to 8k tokens) and trained on the RAGTruth benchmark dataset, our approach outperforms all previous encoder-based models and most prompt-based models, while being approximately 30 times smaller than the best models. LettuceDetect is a token-classification model that processes context-question-answer triples, allowing for the identification of unsupported claims at the token level. Evaluations on the RAGTruth corpus demonstrate an F1 score of 79.22% for example-level detection, which is a 14.8% improvement over Luna, the previous state-of-the-art encoder-based architecture. Additionally, the system can process 30 to 60 examples per second on a single GPU, making it more practical for real-world RAG applications. 

**Abstract (ZH)**: 尽管刘易斯检测（LettuceDetect）系统通过引入外部知识来源提高了生成解决方案的能力，但仍容易产生虚构的答案。我们提出了一种框架，以解决现有虚构检测方法中的两个关键限制：（1）传统编码器方法的上下文窗口限制；（2）基于LLM的方法的计算效率低下。该框架基于ModernBERT扩展的上下文处理能力（最多8k个标记），并使用RAGTruth基准数据集进行训练，我们的方法在性能上优于所有之前的编码器模型，并且大约是最佳模型的30倍小。LettuceDetect是一种标记分类模型，能够处理上下文-问题-答案三元组，并在标记级别识别未受支持的断言。在RAGTruth语料库上的评估结果显示，单个样例级别的检测F1分数为79.22%，比之前的最佳编码器架构Luna提高了14.8%。此外，该系统在单个GPU上每秒可以处理30到60个样例，使其更加适用于实际的RAG应用。 

---
# Mobile-Agent-V: Learning Mobile Device Operation Through Video-Guided Multi-Agent Collaboration 

**Title (ZH)**: 移动代理-V：通过视频引导的多代理协作学习移动设备操作 

**Authors**: Junyang Wang, Haiyang Xu, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17110)  

**Abstract**: The rapid increase in mobile device usage necessitates improved automation for seamless task management. However, many AI-driven frameworks struggle due to insufficient operational knowledge. Manually written knowledge helps but is labor-intensive and inefficient. To address these challenges, we introduce Mobile-Agent-V, a framework that leverages video guidance to provide rich and cost-effective operational knowledge for mobile automation. Mobile-Agent-V enhances task execution capabilities by leveraging video inputs without requiring specialized sampling or preprocessing. Mobile-Agent-V integrates a sliding window strategy and incorporates a video agent and deep-reflection agent to ensure that actions align with user instructions. Through this innovative approach, users can record task processes with guidance, enabling the system to autonomously learn and execute tasks efficiently. Experimental results show that Mobile-Agent-V achieves a 30% performance improvement compared to existing frameworks. 

**Abstract (ZH)**: 移动设备使用量的迅速增加促使了对无缝任务管理的改进自动化需求。然而，许多基于AI的框架因缺乏足够的操作知识而面临挑战。虽然手工编写的知识有助于此问题，但其劳动密集且效率低下。为解决这些挑战，我们提出了Mobile-Agent-V框架，该框架利用视频指导提供丰富且经济的操作知识，用于移动自动化。Mobile-Agent-V通过利用视频输入增强了任务执行能力，无需专门的采样或预处理。Mobile-Agent-V整合了滑动窗口策略，并结合了视频代理和深度反思代理，以确保操作符合用户指令。通过这种方法，用户可以录制带有指导的任务过程，从而使系统能够自主学习并高效执行任务。实验结果表明，Mobile-Agent-V相比现有框架实现了30%的性能提升。 

---
# WildFrame: Comparing Framing in Humans and LLMs on Naturally Occurring Texts 

**Title (ZH)**: WildFrame：人类与大语言模型在自然文本中的框架比较 

**Authors**: Gili Lior, Liron Nacchace, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.17091)  

**Abstract**: Humans are influenced by how information is presented, a phenomenon known as the framing effect. Previous work has shown that LLMs may also be susceptible to framing but has done so on synthetic data and did not compare to human behavior. We introduce WildFrame, a dataset for evaluating LLM responses to positive and negative framing, in naturally-occurring sentences, and compare humans on the same data. WildFrame consists of 1,000 texts, first selecting real-world statements with clear sentiment, then reframing them in either positive or negative light, and lastly, collecting human sentiment annotations. By evaluating eight state-of-the-art LLMs on WildFrame, we find that all models exhibit framing effects similar to humans ($r\geq0.57$), with both humans and models being more influenced by positive rather than negative reframing. Our findings benefit model developers, who can either harness framing or mitigate its effects, depending on the downstream application. 

**Abstract (ZH)**: 人类受到信息呈现方式的影响，这一现象称为框架效应。以往的研究表明，大规模语言模型（LLM）也可能受到框架效应的影响，但这些研究多是在合成数据上进行的，并未将模型的行为与人类行为进行比较。我们引入了WildFrame数据集，用于评估LLM在自然语句中对正面和负面框架的响应，并与人类的行为进行比较。WildFrame数据集包含1000篇文本，首先选择具有明确情感色彩的现实世界陈述，然后重新框架为正面或负面情境，最后收集人类情感注释。通过在WildFrame上评估八种最先进的LLM，我们发现所有模型的框架效应与人类的行为相似（相关系数\(r \geq 0.57\)），并且人类和模型都更受正面框架的影响。我们的研究结果有助于模型开发者，他们可以根据具体的下游应用选择利用或缓解框架效应。 

---
# Automatically Evaluating the Paper Reviewing Capability of Large Language Models 

**Title (ZH)**: 自动评估大型语言模型的论文评审能力 

**Authors**: Hyungyu Shin, Jingyu Tang, Yoonjoo Lee, Nayoung Kim, Hyunseung Lim, Ji Yong Cho, Hwajung Hong, Moontae Lee, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.17086)  

**Abstract**: Peer review is essential for scientific progress, but it faces challenges such as reviewer shortages and growing workloads. Although Large Language Models (LLMs) show potential for providing assistance, research has reported significant limitations in the reviews they generate. While the insights are valuable, conducting the analysis is challenging due to the considerable time and effort required, especially given the rapid pace of LLM developments. To address the challenge, we developed an automatic evaluation pipeline to assess the LLMs' paper review capability by comparing them with expert-generated reviews. By constructing a dataset consisting of 676 OpenReview papers, we examined the agreement between LLMs and experts in their strength and weakness identifications. The results showed that LLMs lack balanced perspectives, significantly overlook novelty assessment when criticizing, and produce poor acceptance decisions. Our automated pipeline enables a scalable evaluation of LLMs' paper review capability over time. 

**Abstract (ZH)**: 同行评审对于科学进步至关重要，但面临着评审员短缺和工作量增加等挑战。虽然大型语言模型（LLMs）显示出提供辅助的潜力，但研究结果显示，它们生成的评审意见存在显著的局限性。尽管这些见解很有价值，但由于需要大量的时间和精力进行分析，尤其是在LLM技术快速发展的背景下，这一过程尤其具有挑战性。为了应对这一挑战，我们开发了一种自动评估管道，通过将LLMs的论文评审能力与专家生成的评审意见进行对比来评估LLMs的论文评审能力。通过构建包含676篇OpenReview论文的数据集，我们考察了LLMs和专家在识别论文优势和劣势方面的共识程度。结果表明，LLMs缺乏平衡视角，在批评时显著忽略了新颖性评估，并且产生了较差的接受决策。我们的自动化管道能够随着时间的推移对LLMs的论文评审能力进行可扩展的评估。 

---
# Systematic Weight Evaluation for Pruning Large Language Models: Enhancing Performance and Sustainability 

**Title (ZH)**: 大型语言模型剪枝中的系统性权重评估：提高性能与可持续性 

**Authors**: Ashhadul Islam, Samir Brahim Belhaouari, Amine Bermak  

**Link**: [PDF](https://arxiv.org/pdf/2502.17071)  

**Abstract**: The exponential growth of large language models (LLMs) like ChatGPT has revolutionized artificial intelligence, offering unprecedented capabilities in natural language processing. However, the extensive computational resources required for training these models have significant environmental implications, including high carbon emissions, energy consumption, and water usage. This research presents a novel approach to LLM pruning, focusing on the systematic evaluation of individual weight importance throughout the training process. By monitoring parameter evolution over time, we propose a method that effectively reduces model size without compromising performance. Extensive experiments with both a scaled-down LLM and a large multimodal model reveal that moderate pruning enhances efficiency and reduces loss, while excessive pruning drastically deteriorates model performance. These findings highlight the critical need for optimized AI models to ensure sustainable development, balancing technological advancement with environmental responsibility. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT的指数级增长已从根本上改变了人工智能领域，提供了前所未有的自然语言处理能力。然而，这些模型所需的巨大计算资源对环境产生了显著的影响，包括高碳排放、能源消耗和水资源使用。本研究提出了一种新的LLM修剪方法，并集中在训练过程中系统性地评估每个权重的重要性。通过监测参数随时间的变化，我们提出了一种方法，能够在不牺牲性能的前提下有效减小模型规模。通过对缩小规模的LLM和大型多模态模型进行广泛实验，结果显示适度修剪提高了效率并减少了损失，而过度修剪则严重恶化了模型性能。这些发现突显了优化AI模型以确保可持续发展的重要性，要在技术进步与环境保护之间找到平衡。 

---
# PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance 

**Title (ZH)**: PrivaCI-Bench：基于情境完整性和法律合规性的隐私评估框架 

**Authors**: Haoran Li, Wenbin Hu, Huihao Jing, Yulin Chen, Qi Hu, Sirui Han, Tianshu Chu, Peizhao Hu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.17041)  

**Abstract**: Recent advancements in generative large language models (LLMs) have enabled wider applicability, accessibility, and flexibility. However, their reliability and trustworthiness are still in doubt, especially for concerns regarding individuals' data privacy. Great efforts have been made on privacy by building various evaluation benchmarks to study LLMs' privacy awareness and robustness from their generated outputs to their hidden representations. Unfortunately, most of these works adopt a narrow formulation of privacy and only investigate personally identifiable information (PII). In this paper, we follow the merit of the Contextual Integrity (CI) theory, which posits that privacy evaluation should not only cover the transmitted attributes but also encompass the whole relevant social context through private information flows. We present PrivaCI-Bench, a comprehensive contextual privacy evaluation benchmark targeted at legal compliance to cover well-annotated privacy and safety regulations, real court cases, privacy policies, and synthetic data built from the official toolkit to study LLMs' privacy and safety compliance. We evaluate the latest LLMs, including the recent reasoner models QwQ-32B and Deepseek R1. Our experimental results suggest that though LLMs can effectively capture key CI parameters inside a given context, they still require further advancements for privacy compliance. 

**Abstract (ZH)**: 近年来生成型大型语言模型（LLMs）的发展使其应用范围、可访问性和灵活性得到了广泛的拓展。然而，它们的可靠性和可信度仍然令人怀疑，尤其是在涉及个人数据隐私方面。为解决这一问题，研究人员已经采取了种种措施来建立各种评估基准，以研究生成型语言模型在隐私意识和鲁棒性方面的表现，从其生成的输出到隐藏表示。遗憾的是，大多数这些研究仅提出了狭窄的隐私定义，并仅调查了个人信息识别信息（PII）。本文沿用了上下文完整性（CI）理论的优点，该理论认为隐私评估不仅应涵盖传输的属性，还应包括通过隐私信息流来覆盖整个相关社会环境的全面评价。我们提出了PrivaCI-Bench，这是一个全面的上下文隐私评估基准，旨在符合法律规定，覆盖详细注释的隐私和安全法规、真实法庭案例、隐私政策以及使用官方工具包生成的合成数据，以研究生成型语言模型的隐私和安全合规性。我们评估了最新的生成型语言模型，包括最近的推理模型QwQ-32B和Deepseek R1。实验结果表明，尽管生成型语言模型能够有效捕捉给定上下文中的关键CI参数，但它们仍需进一步改进以满足隐私合规要求。 

---
# Language Model Re-rankers are Steered by Lexical Similarities 

**Title (ZH)**: 语言模型重新排 ranker 受词汇相似性引导 

**Authors**: Lovisa Hagström, Ercong Nie, Ruben Halifa, Helmut Schmid, Richard Johansson, Alexander Junge  

**Link**: [PDF](https://arxiv.org/pdf/2502.17036)  

**Abstract**: Language model (LM) re-rankers are used to refine retrieval results for retrieval-augmented generation (RAG). They are more expensive than lexical matching methods like BM25 but assumed to better process semantic information. To understand whether LM re-rankers always live up to this assumption, we evaluate 6 different LM re-rankers on the NQ, LitQA2 and DRUID datasets. Our results show that LM re-rankers struggle to outperform a simple BM25 re-ranker on DRUID. Leveraging a novel separation metric based on BM25 scores, we explain and identify re-ranker errors stemming from lexical dissimilarities. We also investigate different methods to improve LM re-ranker performance and find these methods mainly useful for NQ. Taken together, our work identifies and explains weaknesses of LM re-rankers and points to the need for more adversarial and realistic datasets for their evaluation. 

**Abstract (ZH)**: 语言模型（LM）重排序器用于改进检索增强生成（RAG）中的检索结果。与像BM25这样的词法匹配方法相比，LM重排序器更为昂贵，但被认为能够更好地处理语义信息。为了了解LM重排序器是否始终能够满足这一假设，我们分别在NQ、 LitQA2 和 DRUID 数据集上评估了6种不同的LM重排序器。我们的结果显示，LM重排序器在DRUID数据集上难以超越简单的BM25重排序器。借助基于BM25分数的新颖分离度量，我们解释并指出了由词法差异引起的重排序器错误。我们还探讨了提高LM重排序器性能的不同方法，并发现这些方法主要对NQ数据集有效。综上所述，我们的研究指出了LM重排序器的弱点，并强调了需要使用更对抗性和现实性的数据集对其进行评估。 

---
# Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology 

**Title (ZH)**: 理解LLM解释的不确定性：基于推理拓扑的视角 

**Authors**: Longchao Da, Xiaoou Liu, Jiaxin Dai, Lu Cheng, Yaqing Wang, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.17026)  

**Abstract**: Understanding the uncertainty in large language model (LLM) explanations is important for evaluating their faithfulness and reasoning consistency, and thus provides insights into the reliability of LLM's output regarding a question. In this work, we propose a novel framework that quantifies uncertainty in LLM explanations through a reasoning topology perspective. By designing a structural elicitation strategy, we guide the LLMs to frame the explanations of an answer into a graph topology. This process decomposes the explanations into the knowledge related sub-questions and topology-based reasoning structures, which allows us to quantify uncertainty not only at the semantic level but also from the reasoning path. It further brings convenience to assess knowledge redundancy and provide interpretable insights into the reasoning process. Our method offers a systematic way to interpret the LLM reasoning, analyze limitations, and provide guidance for enhancing robustness and faithfulness. This work pioneers the use of graph-structured uncertainty measurement in LLM explanations and demonstrates the potential of topology-based quantification. 

**Abstract (ZH)**: 理解大型语言模型（LLM）解释中的不确定性对于评估其忠实性和推理一致性至关重要，从而为LLM输出关于问题的可靠性提供见解。在本工作中，我们提出了一种新的框架，通过推理拓扑视角量化LLM解释中的不确定性。通过设计一种结构化提取策略，我们引导LLM将答案的解释框架化为图拓扑结构。这一过程将解释分解为与知识相关的子问题和基于拓扑的推理结构，这不仅允许我们在语义层面量化不确定性，还允许从推理路径层面进行量化。这进一步为评估知识冗余性和提供可解释的推理过程见解带来了便利。我们的方法提供了一种系统的方法来解释LLM推理、分析局限性并为增强鲁棒性和忠实性提供指导。本工作开创性地使用了基于图结构的不确定性测量方法，并展示了基于拓扑的量化方法的潜力。 

---
# Towards Auto-Regressive Next-Token Prediction: In-Context Learning Emerges from Generalization 

**Title (ZH)**: 面向自回归下一个词预测：自举学习源于泛化 

**Authors**: Zixuan Gong, Xiaolin Hu, Huayi Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17024)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable in-context learning (ICL) abilities. However, existing theoretical analysis of ICL primarily exhibits two limitations: (a) Limited i.i.d. Setting. Most studies focus on supervised function learning tasks where prompts are constructed with i.i.d. input-label pairs. This i.i.d. assumption diverges significantly from real language learning scenarios where prompt tokens are interdependent. (b) Lack of Emergence Explanation. Most literature answers what ICL does from an implicit optimization perspective but falls short in elucidating how ICL emerges and the impact of pre-training phase on ICL. In our paper, to extend (a), we adopt a more practical paradigm, auto-regressive next-token prediction (AR-NTP), which closely aligns with the actual training of language models. Specifically, within AR-NTP, we emphasize prompt token-dependency, which involves predicting each subsequent token based on the preceding sequence. To address (b), we formalize a systematic pre-training and ICL framework, highlighting the layer-wise structure of sequences and topics, alongside a two-level expectation. In conclusion, we present data-dependent, topic-dependent and optimization-dependent PAC-Bayesian generalization bounds for pre-trained LLMs, investigating that ICL emerges from the generalization of sequences and topics. Our theory is supported by experiments on numerical linear dynamic systems, synthetic GINC and real-world language datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在上下文学习（ICL）方面展示了显著的能力。然而，现有的ICL理论分析存在两个主要限制：(a) 有限的独立同分布（i.i.d.）设置。大多数研究集中在监督函数学习任务上，其中提示由独立同分布的输入-标签对构建。但这种i.i.d.假设与实际的语言学习场景大不相同，在实际中，提示中的词是相互依赖的。(b) 缺乏涌现机制解释。大多数文献从隐式优化的角度解释了ICL做了什么，但未能充分解释ICL是如何涌现的以及预训练阶段对ICL的影响。在我们这篇论文中，为了克服(a)，我们采用了一种更为实用的框架——自回归下一个词预测（AR-NTP），这与语言模型的实际训练更贴近。具体来说，在AR-NTP中，我们强调了提示词的依赖性，即基于前一个序列来预测每个后续的词。为了解决(b)，我们构建了一个系统化的预训练和ICL框架，突出显示了序列和主题的分层结构，并提出了双层期望。最终，我们为预训练的LLMs提供了依赖数据、依赖主题和依赖优化的PAC-贝叶斯泛化界，表明ICL是从序列和主题的一般化中涌现出来的。我们的理论通过在数值线性动态系统、合成GINC和真实世界语言数据集上的实验得到了支持。 

---
# Quantifying Logical Consistency in Transformers via Query-Key Alignment 

**Title (ZH)**: 通过查询-键对齐量化变压器中的逻辑一致性 

**Authors**: Eduard Tulchinskii, Anastasia Voznyuk, Laida Kushnareva, Andrei Andriiainen, Irina Piontkovskaya, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17017)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance in various natural language processing tasks, yet their ability to perform multi-step logical reasoning remains an open challenge. Although Chain-of-Thought prompting has improved logical reasoning by enabling models to generate intermediate steps, it lacks mechanisms to assess the coherence of these logical transitions. In this paper, we propose a novel, lightweight evaluation strategy for logical reasoning that uses query-key alignments inside transformer attention heads. By computing a single forward pass and extracting a "QK-score" from carefully chosen heads, our method reveals latent representations that reliably separate valid from invalid inferences, offering a scalable alternative to traditional ablation-based techniques. We also provide an empirical validation on multiple logical reasoning benchmarks, demonstrating improved robustness of our evaluation method against distractors and increased reasoning depth. The experiments were conducted on a diverse set of models, ranging from 1.5B to 70B parameters. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种自然语言处理任务中表现出色，但它们在执行多步逻辑推理方面的能力仍然是一个开放的挑战。虽然通过链式思考提示方法可以增强模型的逻辑推理能力，使其能够生成中间步骤，但它缺乏评估这些逻辑转换连贯性的机制。在本文中，我们提出了一种新颖且轻量级的逻辑推理评估策略，该策略使用变压器注意力头内的查询-键对齐。通过计算单次前向传播并从精心选择的头中提取“QK分值”，我们的方法揭示了能够可靠地区分有效推理和无效推理的潜在表示，提供了一种传统消融法技术的可扩展替代方案。我们还在多个逻辑推理基准测试上进行了实证验证，表明我们的评估方法对干扰项具有更高的稳健性，并且能够进行更深的推理。实验是在参数量从15亿到70亿的多种模型上进行的。 

---
# All-in-one: Understanding and Generation in Multimodal Reasoning with the MAIA Benchmark 

**Title (ZH)**: 一站式解决方案：MAIA基准测试在多模态推理中的理解与生成 

**Authors**: Davide Testa, Giovanni Bonetta, Raffaella Bernardi, Alessandro Bondielli, Alessandro Lenci, Alessio Miaschi, Lucia Passaro, Bernardo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2502.16989)  

**Abstract**: We introduce MAIA (Multimodal AI Assessment), a native-Italian benchmark designed for fine-grained investigation of the reasoning abilities of visual language models on videos. MAIA differs from other available video benchmarks for its design, its reasoning categories, the metric it uses and the language and culture of the videos. It evaluates Vision Language Models (VLMs) on two aligned tasks: a visual statement verification task, and an open-ended visual question-answering task, both on the same set of video-related questions. It considers twelve reasoning categories that aim to disentangle language and vision relations by highlight when one of two alone encodes sufficient information to solve the tasks, when they are both needed and when the full richness of the short video is essential instead of just a part of it. Thanks to its carefully taught design, it evaluates VLMs' consistency and visually grounded natural language comprehension and generation simultaneously through an aggregated metric. Last but not least, the video collection has been carefully selected to reflect the Italian culture and the language data are produced by native-speakers. 

**Abstract (ZH)**: 我们引入了MAIA（多模态AI评估）基准，这是一种针对视觉语言模型在视频中推理能力进行精细研究的目的地意大利语基准。MAIA与其他可用的视频基准相比，在设计理念、推理类别、评价指标、视频的语言和文化背景等方面均有所不同。它在两个对齐的任务上评估视觉语言模型（VLMs）：一个是视觉陈述验证任务，另一个是开放性的视觉问答任务，这些任务针对的是相同的视频相关问题集。它考虑了十二个推理类别，旨在通过强调在两种信息中哪种单独的信息足以解决问题、两者是否都需要，以及整个短视频的重要性来分离语言与视觉之间的关系。得益于其精心设计，MAIA通过聚合指标同时评估VLMs的一致性和基于视觉的自然语言理解和生成能力。最后，视频集合经过仔细筛选，以反映意大利文化，语言数据由母语使用者生成。 

---
# Hotter and Colder: A New Approach to Annotating Sentiment, Emotions, and Bias in Icelandic Blog Comments 

**Title (ZH)**: 更热更冷：一种新的方法标注冰岛博客评论中的情感、情绪和偏见 

**Authors**: Steinunn Rut Friðriksdóttir, Dan Saattrup Nielsen, Hafsteinn Einarsson  

**Link**: [PDF](https://arxiv.org/pdf/2502.16987)  

**Abstract**: This paper presents Hotter and Colder, a dataset designed to analyze various types of online behavior in Icelandic blog comments. Building on previous work, we used GPT-4o mini to annotate approximately 800,000 comments for 25 tasks, including sentiment analysis, emotion detection, hate speech, and group generalizations. Each comment was automatically labeled on a 5-point Likert scale. In a second annotation stage, comments with high or low probabilities of containing each examined behavior were subjected to manual revision. By leveraging crowdworkers to refine these automatically labeled comments, we ensure the quality and accuracy of our dataset resulting in 12,232 uniquely annotated comments and 19,301 annotations. Hotter and Colder provides an essential resource for advancing research in content moderation and automatically detectiong harmful online behaviors in Icelandic. 

**Abstract (ZH)**: 本文介绍了名为“Hotter and Colder”的数据集，该数据集旨在分析冰岛博客评论中的各种在线行为。在前人工作的基础上，我们使用了GPT-4o mini 对约800,000条评论进行了注解，涵盖25项任务，包括情感分析、情绪检测、仇恨言论以及群体概括等。每个评论都被自动标记在一个5点李克特量表上。在第二个注解阶段，我们对那些具有较高或较低概率包含每种检查行为的评论进行了手动修订。通过利用众包工人来细化这些自动标注的评论，我们确保了数据集的质量和准确性，最终获得了12,232个独立标注的评论和19,301个标注。Hotter and Colder 为推进内容管理研究和自动检测冰岛有害在线行为提供了重要的资源。 

---
# LongSafety: Evaluating Long-Context Safety of Large Language Models 

**Title (ZH)**: 长上下文安全性评估：大型语言模型的安全性评价 

**Authors**: Yida Lu, Jiale Cheng, Zhexin Zhang, Shiyao Cui, Cunxiang Wang, Xiaotao Gu, Yuxiao Dong, Jie Tang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16971)  

**Abstract**: As Large Language Models (LLMs) continue to advance in understanding and generating long sequences, new safety concerns have been introduced through the long context. However, the safety of LLMs in long-context tasks remains under-explored, leaving a significant gap in both evaluation and improvement of their safety. To address this, we introduce LongSafety, the first comprehensive benchmark specifically designed to evaluate LLM safety in open-ended long-context tasks. LongSafety encompasses 7 categories of safety issues and 6 user-oriented long-context tasks, with a total of 1,543 test cases, averaging 5,424 words per context. Our evaluation towards 16 representative LLMs reveals significant safety vulnerabilities, with most models achieving safety rates below 55%. Our findings also indicate that strong safety performance in short-context scenarios does not necessarily correlate with safety in long-context tasks, emphasizing the unique challenges and urgency of improving long-context safety. Moreover, through extensive analysis, we identify challenging safety issues and task types for long-context models. Furthermore, we find that relevant context and extended input sequences can exacerbate safety risks in long-context scenarios, highlighting the critical need for ongoing attention to long-context safety challenges. Our code and data are available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在理解和生成长序列方面不断进步，通过长上下文引入了新的安全问题。然而，LLMs 在长上下文任务中的安全性仍然缺乏探索，这留下了评估和改进其安全性的显著缺口。为应对这一问题，我们提出了 LongSafety，这是首个专门设计用于评估 LLM 安全性的全面基准，专注于开放性长上下文任务。LongSafety 包括 7 类别的安全问题和 6 个用户导向的长上下文任务，共有 1,543 个测试案例，平均每个上下文长度为 5,424 词。我们对 16 种代表性 LLM 的评估揭示了显著的安全漏洞，大多数模型的安全率低于 55%。我们的研究还表明，在短上下文场景中的强大的安全性能不一定与长上下文任务中的安全性相关，突出了提高长上下文安全性所面临的独特挑战和紧迫性。此外，通过对广泛的数据进行深入分析，我们识别出长上下文模型中的安全挑战及任务类型。进一步的研究还发现，相关的上下文和扩展的输入序列会在长上下文场景中加剧安全风险，强调了对长上下文安全性持续关注的迫切需求。我们的代码和数据可在以下网址获取：[这里插入网址]。 

---
# UrduLLaMA 1.0: Dataset Curation, Preprocessing, and Evaluation in Low-Resource Settings 

**Title (ZH)**: UrduLLaMA 1.0：低资源环境下的数据采集、预处理与评估 

**Authors**: Layba Fiaz, Munief Hassan Tahir, Sana Shams, Sarmad Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2502.16961)  

**Abstract**: Multilingual Large Language Models (LLMs) often provide suboptimal performance on low-resource languages like Urdu. This paper introduces UrduLLaMA 1.0, a model derived from the open-source Llama-3.1-8B-Instruct architecture and continually pre-trained on 128 million Urdu tokens, capturing the rich diversity of the language. To enhance instruction-following and translation capabilities, we leverage Low-Rank Adaptation (LoRA) to fine tune the model on 41,000 Urdu instructions and approximately 50,000 English-Urdu translation pairs. Evaluation across three machine translation datasets demonstrates significant performance improvements compared to state-of-the-art (SOTA) models, establishing a new benchmark for Urdu LLMs. These findings underscore the potential of targeted adaptation strategies with limited data and computational resources to address the unique challenges of low-resource languages. 

**Abstract (ZH)**: 多语言大型语言模型（LLMs）在低资源语言如乌尔都语上的表现往往不尽如人意。本文介绍了一种名为UrduLLaMA 1.0的模型，该模型源自开源的Llama-3.1-8B-Instruct架构，并连续预训练了1.28亿个乌尔都语语 token，以捕捉该语言的丰富多样性。为了提高指令遵循能力和翻译能力，我们利用低秩适应（LoRA）方法在4.1万个乌尔都语指令和大约5万个英-乌尔都语翻译对上对模型进行了微调。在三个机器翻译数据集上的评估表明，相比当前最先进的（SOTA）模型，UrduLLaMA 1.0取得了显著的性能提升，为乌尔都语LLMs设立了新的基准。这些发现强调了在有限的数据和计算资源条件下，有针对性的适应策略可以解决低资源语言的特殊挑战的潜力。 

---
# NUTSHELL: A Dataset for Abstract Generation from Scientific Talks 

**Title (ZH)**: NUTSHELL：一个来自科学讲座的摘要生成数据集 

**Authors**: Maike Züfle, Sara Papi, Beatrice Savoldi, Marco Gaido, Luisa Bentivogli, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2502.16942)  

**Abstract**: Scientific communication is receiving increasing attention in natural language processing, especially to help researches access, summarize, and generate content. One emerging application in this area is Speech-to-Abstract Generation (SAG), which aims to automatically generate abstracts from recorded scientific presentations. SAG enables researchers to efficiently engage with conference talks, but progress has been limited by a lack of large-scale datasets. To address this gap, we introduce NUTSHELL, a novel multimodal dataset of *ACL conference talks paired with their corresponding abstracts. We establish strong baselines for SAG and evaluate the quality of generated abstracts using both automatic metrics and human judgments. Our results highlight the challenges of SAG and demonstrate the benefits of training on NUTSHELL. By releasing NUTSHELL under an open license (CC-BY 4.0), we aim to advance research in SAG and foster the development of improved models and evaluation methods. 

**Abstract (ZH)**: 科学交流在自然语言处理领域正受到越来越多的关注，特别是在帮助研究人员访问、总结和生成内容方面。这一领域的一个新兴应用是语音摘要生成（Speech-to-Abstract Generation, SAG），其目标是自动从录音的科学演讲中生成摘要。SAG 使得研究人员能够高效地参与会议报告，但进展受限于缺乏大规模数据集。为了填补这一空白，我们引入了 NUTSHELL，这是一个全新的多模态数据集，其中包括来自 ACL 大会的演讲录音及其对应的摘要。我们为 SAG 建立了强大的基线，并使用自动评估指标和人工评判来评估生成摘要的质量。我们的结果突出了 SAG 的挑战，并展示了使用 NUTSHELL 训练的优势。通过在开源许可（CC-BY 4.0）下发布 NUTSHELL，我们旨在推动 SAG 的研究，促进更好模型和评估方法的发展。 

---
# Reasoning Does Not Necessarily Improve Role-Playing Ability 

**Title (ZH)**: 推理能力并不必然提高角色扮演能力 

**Authors**: Xiachong Feng, Longxu Dou, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.16940)  

**Abstract**: The application of role-playing large language models (LLMs) is rapidly expanding in both academic and commercial domains, driving an increasing demand for high-precision role-playing models. Simultaneously, the rapid advancement of reasoning techniques has continuously pushed the performance boundaries of LLMs. This intersection of practical role-playing demands and evolving reasoning capabilities raises an important research question: "Can reasoning techniques enhance the role-playing capabilities of LLMs?" To address this, we conduct a comprehensive study using 6 role-playing benchmarks, 24 LLMs, and 3 distinct role-playing strategies, comparing the effectiveness of direct zero-shot role-playing, role-playing with Chain-of-Thought (CoT), and role-playing using reasoning-optimized LLMs. Our findings reveal that CoT may reduce role-playing performance, reasoning-optimized LLMs are unsuitable for role-playing, reasoning ability disrupts the role-playing scaling law, large models still lack proficiency in advanced role-playing, and Chinese role-playing performance surpasses English role-playing performance. Furthermore, based on extensive experimental results, we propose two promising future research directions: Role-aware CoT for improving role-playing LLMs and Reinforcement Learning for role-playing LLMs, aiming to enhance the adaptability, consistency, and effectiveness of role-playing LLMs for both research and real-world applications. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

角色扮演大型语言模型（LLMs）的应用在学术界和商业界迅速扩展，推动了对高精度角色扮演模型的需求。同时，推理技术的快速进步不断推动LLMs性能边界的发展。实际角色扮演需求与推理能力演化交汇之处提出了一个重要研究问题：“推理技术能否提升LLMs的角色扮演能力？”为解决这一问题，我们使用6个角色扮演基准、24个LLMs和3种不同的角色扮演策略，比较了直接零样本角色扮演、带有思维链（Chain-of-Thought, CoT）的角色扮演以及使用推理优化的LLMs的角色扮演效果。研究结果表明，CoT可能降低角色扮演性能，推理优化的LLMs不适于角色扮演，推理能力干扰了角色扮演的规模律，大型模型在高级角色扮演上仍缺乏足够的熟练度，并且中文角色扮演性能优于英文角色扮演性能。此外，基于广泛的实验结果，我们提出了两个有前景的未来研究方向：角色感知的CoT以提高角色扮演LLMs以及强化学习以优化角色扮演LLMs，旨在增强角色扮演LLMs在研究和实际应用中的适应性、一致性和有效性。 

---
# A Systematic Survey of Automatic Prompt Optimization Techniques 

**Title (ZH)**: 自动提示优化技术系统的综述 

**Authors**: Kiran Ramnath, Kang Zhou, Sheng Guan, Soumya Smruti Mishra, Xuan Qi, Zhengyuan Shen, Shuai Wang, Sangmin Woo, Sullam Jeoung, Yawei Wang, Haozhu Wang, Han Ding, Yuzhe Lu, Zhichao Xu, Yun Zhou, Balasubramaniam Srinivasan, Qiaojing Yan, Yueyan Chen, Haibo Ding, Panpan Xu, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2502.16923)  

**Abstract**: Since the advent of large language models (LLMs), prompt engineering has been a crucial step for eliciting desired responses for various Natural Language Processing (NLP) tasks. However, prompt engineering remains an impediment for end users due to rapid advances in models, tasks, and associated best practices. To mitigate this, Automatic Prompt Optimization (APO) techniques have recently emerged that use various automated techniques to help improve the performance of LLMs on various tasks. In this paper, we present a comprehensive survey summarizing the current progress and remaining challenges in this field. We provide a formal definition of APO, a 5-part unifying framework, and then proceed to rigorously categorize all relevant works based on their salient features therein. We hope to spur further research guided by our framework. 

**Abstract (ZH)**: 自大型语言模型（LLM）的出现以来，提示工程已成为各种自然语言处理（NLP）任务中 eliciting 所需响应的关键步骤。然而，由于模型、任务及相关最佳实践的迅速发展，提示工程依然对最终用户构成阻碍。为解决这一问题，最近出现了自动提示优化（APO）技术，这些技术利用各种自动化方法来帮助提高LLM在各种任务上的性能。在这篇论文中，我们提供了一份全面的综述，总结了该领域当前的进展及面临的挑战。我们将正式定义APO，构建一个统一的五个部分框架，并在此基础上严格分类所有相关工作，按其显著特征进行。我们希望通过对该框架的进一步研究能够提供指导。 

---
# Benchmarking Temporal Reasoning and Alignment Across Chinese Dynasties 

**Title (ZH)**: 跨中国朝代的时间推理与对齐基准测试 

**Authors**: Zhenglin Wang, Jialong Wu, Pengfei LI, Yong Jiang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.16922)  

**Abstract**: Temporal reasoning is fundamental to human cognition and is crucial for various real-world applications. While recent advances in Large Language Models have demonstrated promising capabilities in temporal reasoning, existing benchmarks primarily rely on rule-based construction, lack contextual depth, and involve a limited range of temporal entities. To address these limitations, we introduce Chinese Time Reasoning (CTM), a benchmark designed to evaluate LLMs on temporal reasoning within the extensive scope of Chinese dynastic chronology. CTM emphasizes cross-entity relationships, pairwise temporal alignment, and contextualized and culturally-grounded reasoning, providing a comprehensive evaluation. Extensive experimental results reveal the challenges posed by CTM and highlight potential avenues for improvement. 

**Abstract (ZH)**: 时间推理是人类认知的基础，对于各种实际应用场景至关重要。尽管近年来大规模语言模型在时间推理方面展示了令人鼓舞的能力，但现有的基准测试主要依赖规则构建，缺乏上下文深度，并且涉及的时间实体范围有限。为应对这些局限，我们提出了中文历史时间推理基准（CTM），旨在评估语言模型在广泛的历史朝代时间范围内的时间推理能力。CTM 强调跨实体关系、两两时间对齐以及上下文化和文化背景下的推理，提供了全面的评估框架。大量实验结果揭示了 CTM 所提出挑战，并指出了改进的潜在方向。 

---
# SS-MPC: A Sequence-Structured Multi-Party Conversation System 

**Title (ZH)**: SS-MPC：一种序列结构化的多方对话系统 

**Authors**: Yoonjin Jang, Keunha Kim, Youngjoong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2502.16920)  

**Abstract**: Recent Multi-Party Conversation (MPC) models typically rely on graph-based approaches to capture dialogue structures. However, these methods have limitations, such as information loss during the projection of utterances into structural embeddings and constraints in leveraging pre-trained language models directly. In this paper, we propose \textbf{SS-MPC}, a response generation model for MPC that eliminates the need for explicit graph structures. Unlike existing models that depend on graphs to analyze conversation structures, SS-MPC internally encodes the dialogue structure as a sequential input, enabling direct utilization of pre-trained language models. Experimental results show that \textbf{SS-MPC} achieves \textbf{15.60\% BLEU-1} and \textbf{12.44\% ROUGE-L} score, outperforming the current state-of-the-art MPC response generation model by \textbf{3.91\%p} in \textbf{BLEU-1} and \textbf{0.62\%p} in \textbf{ROUGE-L}. Additionally, human evaluation confirms that SS-MPC generates more fluent and accurate responses compared to existing MPC models. 

**Abstract (ZH)**: 近年来的多党对话（MPC）模型通常依赖于基于图的方法来捕捉对话结构。然而，这些方法存在一些局限性，如在将话语投影到结构嵌入时的信息丢失以及直接利用预训练语言模型的限制。在本文中，我们提出了一种名为**SS-MPC**的响应生成模型，该模型消除了对显式图结构的依赖。与依赖于图来分析对话结构的现有模型不同，SS-MPC 内部将对话结构编码为顺序输入，从而使预训练语言模型可以直接利用。实验结果显示，**SS-MPC** 在 **BLEU-1** 和 **ROUGE-L** 指标上的得分分别为 **15.60%** 和 **12.44%**，分别比当前最先进的 MPC 响应生成模型高出 **3.91%** 和 **0.62%**。此外，人工评估证实，SS-MPC 生成的响应更为流畅且准确，优于现有的 MPC 模型。 

---
# Dependency Parsing with the Structuralized Prompt Template 

**Title (ZH)**: 基于结构化提示模板的依存句法解析 

**Authors**: Keunha Kim, Youngjoong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2502.16919)  

**Abstract**: Dependency parsing is a fundamental task in natural language processing (NLP), aiming to identify syntactic dependencies and construct a syntactic tree for a given sentence. Traditional dependency parsing models typically construct embeddings and utilize additional layers for prediction. We propose a novel dependency parsing method that relies solely on an encoder model with a text-to-text training approach. To facilitate this, we introduce a structured prompt template that effectively captures the structural information of dependency trees. Our experimental results demonstrate that the proposed method achieves outstanding performance compared to traditional models, despite relying solely on a pre-trained model. Furthermore, this method is highly adaptable to various pre-trained models across different target languages and training environments, allowing easy integration of task-specific features. 

**Abstract (ZH)**: 依赖解析是自然语言处理（NLP）中的一个基础任务，旨在识别句法依赖关系并为给定的句子构建句法树。传统依赖解析模型通常构建嵌入并利用额外层进行预测。我们提出了一种新颖的依赖解析方法，该方法仅依赖于包含文本到文本训练方法的编码器模型。为了实现这一点，我们引入了一种结构化提示模板，能够有效地捕获依赖树的结构信息。实验结果表明，尽管仅依赖预训练模型，但所提出的方法在性能上取得了出色的表现。此外，该方法高度适应不同目标语言和训练环境下的各种预训练模型，使其能够轻松地整合任务特定特征。 

---
# AutoLogi: Automated Generation of Logic Puzzles for Evaluating Reasoning Abilities of Large Language Models 

**Title (ZH)**: AutoLogi：自动化生成逻辑谜题以评估大型语言模型的推理能力 

**Authors**: Qin Zhu, Fei Huang, Runyu Peng, Keming Lu, Bowen Yu, Qinyuan Cheng, Xipeng Qiu, Xuanjing Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.16906)  

**Abstract**: While logical reasoning evaluation of Large Language Models (LLMs) has attracted significant attention, existing benchmarks predominantly rely on multiple-choice formats that are vulnerable to random guessing, leading to overestimated performance and substantial performance fluctuations. To obtain more accurate assessments of models' reasoning capabilities, we propose an automated method for synthesizing open-ended logic puzzles, and use it to develop a bilingual benchmark, AutoLogi. Our approach features program-based verification and controllable difficulty levels, enabling more reliable evaluation that better distinguishes models' reasoning abilities. Extensive evaluation of eight modern LLMs shows that AutoLogi can better reflect true model capabilities, with performance scores spanning from 35% to 73% compared to the narrower range of 21% to 37% on the source multiple-choice dataset. Beyond benchmark creation, this synthesis method can generate high-quality training data by incorporating program verifiers into the rejection sampling process, enabling systematic enhancement of LLMs' reasoning capabilities across diverse datasets. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）的逻辑推理评估已经吸引了大量关注，但现有的基准测试大多依赖于容易受到随机猜测影响的多项选择格式，这会导致过度估计模型性能并导致显著的性能波动。为了获得更准确的模型推理能力评估，我们提出了一种自动合成开放型逻辑谜题的方法，并利用该方法开发了一个双语基准测试AutoLogi。我们的方法通过程序验证和可控的难度级别，提供了一种更可靠的评估方式，能够更好地区分模型的推理能力。对八种现代LLM的广泛评估表明，AutoLogi 能够更好地反映模型的实际能力，其性能评分范围从35%到73%，而原始多项选择数据集的评分范围仅为21%到37%。除了基准测试的创建，此合成方法还可以通过将程序验证器纳入拒绝采样过程中来生成高质量的训练数据，从而系统地提升LLM在各种数据集中的推理能力。 

---
# GuidedBench: Equipping Jailbreak Evaluation with Guidelines 

**Title (ZH)**: GuidedBench：为越狱评估提供指导原则 

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16903)  

**Abstract**: Jailbreaking methods for large language models (LLMs) have gained increasing attention for building safe and responsible AI systems. After analyzing 35 jailbreak methods across six categories, we find that existing benchmarks, relying on universal LLM-based or keyword-matching scores, lack case-specific criteria, leading to conflicting results. In this paper, we introduce a more robust evaluation framework for jailbreak methods, with a curated harmful question dataset, detailed case-by-case evaluation guidelines, and a scoring system equipped with these guidelines. Our experiments show that existing jailbreak methods exhibit better discrimination when evaluated using our benchmark. Some jailbreak methods that claim to achieve over 90% attack success rate (ASR) on other benchmarks only reach a maximum of 30.2% on our benchmark, providing a higher ceiling for more advanced jailbreak research; furthermore, using our scoring system reduces the variance of disagreements between different evaluator LLMs by up to 76.33%. This demonstrates its ability to provide more fair and stable evaluation. 

**Abstract (ZH)**: 对于大型语言模型（LLMs）的越狱方法（Jailbreaking）的研究正在逐渐受到关注，以构建安全和负责任的AI系统。经过对6个类别中的35种越狱方法进行分析后，我们发现现有的基准方法依赖于通用的LLM评分或关键词匹配评分，缺乏针对具体情况的标准，导致结果存在矛盾。在本文中，我们提出了一种更为 robust 的评估框架，其中包括一个精选的有害问题数据集、详细的逐案例评估指南以及与这些指南相结合的评分系统。实验结果表明，利用我们提出的基准方法评估越狱方法时，它们的区分能力更强。一些声称在其他基准上实现超过90%攻击成功率（ASR）的越狱方法，在我们的基准上仅达到30.2%的最大值，这为更先进的越狱研究设置了更高的门槛；此外，使用我们的评分系统可以将不同评估者语言模型之间分歧的方差降低高达76.33%，这证明了其提供更公平和稳定的评估能力。 

---
# Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs 

**Title (ZH)**: Char-mander 使用 mBackdoor！多语言大语言模型跨语言后门攻击研究 

**Authors**: Himanshu Beniwal, Sailesh Panda, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.16901)  

**Abstract**: We explore Cross-lingual Backdoor ATtacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare tokens serving as specific effective triggers. Our findings expose a critical vulnerability in the fundamental architecture that enables cross-lingual transfer in these models. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 我们将探讨多语言大型语言模型（mLLMs）中的跨语言后门攻击（X-BAT），揭示了如何通过共享嵌入空间，将一种语言中的后门自动转移到其他语言中。以毒性分类为例，我们展示了攻击者可以通过污染单一语言的数据，利用稀有词汇作为特定的有效触发器，来破坏多语言系统。我们的研究发现揭示了这些模型中支持跨语言转移的基本架构中的一个关键性漏洞。我们的代码和数据可在以下链接查看：this https URL。 

---
# Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment 

**Title (ZH)**: 重新绽放LoRA的光彩：通过自适应奇异值和Mixture-of-Experts优化对齐提升LoRA性能 

**Authors**: Chenghao Fan, Zhenyi Lu, Sichen Liu, Xiaoye Qu, Wei Wei, Chengfeng Gu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.16894)  

**Abstract**: While Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning for Large Language Models (LLMs), its performance often falls short of Full Fine-Tuning (Full FT). Current methods optimize LoRA by initializing with static singular value decomposition (SVD) subsets, leading to suboptimal leveraging of pre-trained knowledge. Another path for improving LoRA is incorporating a Mixture-of-Experts (MoE) architecture. However, weight misalignment and complex gradient dynamics make it challenging to adopt SVD prior to the LoRA MoE architecture. To mitigate these issues, we propose \underline{G}reat L\underline{o}R\underline{A} Mixture-of-Exper\underline{t} (GOAT), a framework that (1) adaptively integrates relevant priors using an SVD-structured MoE, and (2) aligns optimization with full fine-tuned MoE by deriving a theoretical scaling factor. We demonstrate that proper scaling, without modifying the architecture or training algorithms, boosts LoRA MoE's efficiency and performance. Experiments across 25 datasets, including natural language understanding, commonsense reasoning, image classification, and natural language generation, demonstrate GOAT's state-of-the-art performance, closing the gap with Full FT. 

**Abstract (ZH)**: 低秩适应（LoRA）使大规模语言模型（LLMs）的参数高效微调成为可能，但其性能往往不如全面微调（Full FT）。当前的方法通过使用固定的奇异值分解（SVD）子集初始化LoRA，导致预训练知识的利用不够充分。改进LoRA的另一途径是结合Mixture-of-Experts（MoE）架构。然而，在LoRA MoE架构之前采用SVD会导致权重对齐问题和复杂的梯度动态，这使得引入SVD成为一个挑战。为解决这些问题，我们提出了**G**reat **L**o**R**A **M**ixture-of-**E**xperts（GOAT）框架，该框架（1）使用SVD结构的MoE自适应地整合相关先验知识，并且（2）通过推导理论缩放因子来使优化与全面微调的MoE保持一致。我们证明，在不修改架构或训练算法的情况下，适当的比例调整可以提升LoRA MoE的效率和性能。在包括自然语言理解、常识推理、图像分类和自然语言生成在内的25个数据集上进行的实验表明，GOAT达到了最先进的性能，缩小了与全面微调之间的问题。 

---
# Applying LLMs to Active Learning: Towards Cost-Efficient Cross-Task Text Classification without Manually Labeled Data 

**Title (ZH)**: 将大语言模型应用于主动学习：在无需手动标注数据的情况下实现成本高效的跨任务文本分类 

**Authors**: Yejian Zhang, Shingo Takada  

**Link**: [PDF](https://arxiv.org/pdf/2502.16892)  

**Abstract**: Machine learning-based classifiers have been used for text classification, such as sentiment analysis, news classification, and toxic comment classification. However, supervised machine learning models often require large amounts of labeled data for training, and manual annotation is both labor-intensive and requires domain-specific knowledge, leading to relatively high annotation costs. To address this issue, we propose an approach that integrates large language models (LLMs) into an active learning framework. Our approach combines the Robustly Optimized BERT Pretraining Approach (RoBERTa), Generative Pre-trained Transformer (GPT), and active learning, achieving high cross-task text classification performance without the need for any manually labeled data. Furthermore, compared to directly applying GPT for classification tasks, our approach retains over 93% of its classification performance while requiring only approximately 6% of the computational time and monetary cost, effectively balancing performance and resource efficiency. These findings provide new insights into the efficient utilization of LLMs and active learning algorithms in text classification tasks, paving the way for their broader application. 

**Abstract (ZH)**: 基于机器学习的分类器已在文本分类领域得到应用，如情感分析、新闻分类和毒 lazım评论分类。然而，监督机器学习模型通常需要大量带标签的数据进行训练，而人工标注既劳动密集又需要特定领域的知识，导致注释成本相对较高。为了解决这一问题，我们提出了一种将大规模语言模型（LLMs）集成到主动学习框架中的方法。该方法结合了Robustly Optimized BERT Pretraining Approach（RoBERTa）、Generative Pre-trained Transformer（GPT）和主动学习，实现了跨任务的高文本分类性能，无需任何人工标注数据。此外，相比直接使用GPT进行分类任务，我们的方法保留了超过93%的分类性能，同时仅需约6%的计算时间和经济成本，有效平衡了性能和资源效率。这些发现为进一步探讨在文本分类任务中高效利用LLMs和主动学习算法提供了新的见解，并为更广泛的应用铺平了道路。 

---
# DBudgetKV: Dynamic Budget in KV Cache Compression for Ensuring Optimal Performance 

**Title (ZH)**: DBudgetKV：键值缓存压缩中的动态预算以确保最优性能 

**Authors**: Xuanfan Ni, Liyan Xu, Chenyang Lyu, Longyue Wang, Mo Yu, Lemao Liu, Fandong Meng, Jie Zhou, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.16886)  

**Abstract**: To alleviate memory burden during inference of large language models (LLMs), numerous studies have focused on compressing the KV cache by exploring aspects such as attention sparsity. However, these techniques often require a pre-defined cache budget; as the optimal budget varies with different input lengths and task types, it limits their practical deployment accepting open-domain instructions. To address this limitation, we propose a new KV cache compression objective: to always ensure the full-cache performance regardless of specific inputs, while maximizing KV cache pruning as much as possible. To achieve this goal, we introduce a novel KV cache compression method dubbed DBudgetKV, which features an attention-based metric to signal when the remaining KV cache is unlikely to match the full-cache performance, then halting the pruning process. Empirical evaluation spanning diverse context lengths, task types, and model sizes suggests that our method achieves lossless KV pruning effectively and robustly, exceeding 25% compression ratio on average. Furthermore, our method is easy to integrate within LLM inference, not only optimizing memory space, but also showing reduced inference time compared to existing methods. 

**Abstract (ZH)**: 为了解决在大规模语言模型（LLMs）推理过程中内存负担的问题，众多研究集中在通过探索注意力稀疏性等方式压缩KV缓存。然而，这些方法通常需要预定义的缓存预算；由于最优预算会随着输入长度和任务类型的不同而变化，这限制了它们在处理开放域指令时的实际部署。为了克服这一局限，我们提出了一个新的KV缓存压缩目标：确保在任何输入情况下都能保持全缓存性能，同时尽可能最大化KV缓存的剪枝幅度。为了实现这一目标，我们提出了一种名为DBudgetKV的新KV缓存压缩方法，该方法利用注意力机制来信号化剩余的KV缓存不太可能达到全缓存性能，从而停止剪枝过程。从不同上下文长度、任务类型和模型规模的实验评价来看，我们的方法能够有效地且稳健地实现无损KV剪枝，平均压缩比超过了25%。此外，我们的方法易于集成到LLM推理中，不仅能优化内存空间，还能比现有方法减少推理时间。 

---
# CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter 

**Title (ZH)**: CORAL：学习多步训练过程中一致表示的轻量级推测性草稿方法 

**Authors**: Yepeng Weng, Dianwen Mei, Huishi Qiu, Xujie Chen, Li Liu, Jiang Tian, Zhongchao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16880)  

**Abstract**: Speculative decoding is a powerful technique that accelerates Large Language Model (LLM) inference by leveraging a lightweight speculative draft model. However, existing designs suffers in performance due to misalignment between training and inference. Recent methods have tried to solve this issue by adopting a multi-step training strategy, but the complex inputs of different training steps make it harder for the draft model to converge. To address this, we propose CORAL, a novel framework that improves both accuracy and efficiency in speculative drafting. CORAL introduces Cross-Step Representation Alignment, a method that enhances consistency across multiple training steps, significantly improving speculative drafting performance. Additionally, we identify the LM head as a major bottleneck in the inference speed of the draft model. We introduce a weight-grouping mechanism that selectively activates a subset of LM head parameters during inference, substantially reducing the latency of the draft model. We evaluate CORAL on three LLM families and three benchmark datasets, achieving speedup ratios of 2.50x-4.07x, outperforming state-of-the-art methods such as EAGLE-2 and HASS. Our results demonstrate that CORAL effectively mitigates training-inference misalignment and delivers significant speedup for modern LLMs with large vocabularies. 

**Abstract (ZH)**: 推测性解码是一种有效的方法，通过利用轻量级的推测性草稿模型来加速大型语言模型（LLM）的推理。然而，现有的设计因训练和推理之间的对齐问题而性能不佳。最近的方法尝试通过采用多步训练策略来解决这一问题，但不同训练步骤的复杂输入使得草稿模型难以收敛。为了解决这一问题，我们提出了一种名为CORAL的新框架，该框架在推测性草稿生成中提高了准确性和效率。CORAL引入了跨步骤表示对齐的方法，该方法增强了多个训练步骤之间的一致性，显著提高了推测性草稿生成的效果。此外，我们发现语言模型（LM） head 是草稿模型推理速度的主要瓶颈。我们引入了一种权重分组机制，在推理过程中选择性地激活LM head的一部分参数，大幅减少了草稿模型的延迟。我们对三种不同的LLM家族和三个基准数据集评估了CORAL，实现了2.50x-4.07x的加速比，优于现有的EAGLE-2和HASS等方法。我们的结果显示，CORAL有效地缓解了训练和推理之间的不一致性问题，并为具有大型词汇的现代LLM提供了显著的加速效果。 

---
# LongAttn: Selecting Long-context Training Data via Token-level Attention 

**Title (ZH)**: 长上下文注意力：通过token级注意力选择长上下文训练数据 

**Authors**: Longyun Wu, Dawei Zhu, Guangxiang Zhao, Zhuocheng Yu, Junfeng Ran, Xiangyu Wong, Lin Sun, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.16860)  

**Abstract**: With the development of large language models (LLMs), there has been an increasing need for significant advancements in handling long contexts. To enhance long-context capabilities, constructing high-quality training data with long-range dependencies is crucial. Existing methods to select long-context data often rely on sentence-level analysis, which can be greatly optimized in both performance and efficiency. In this paper, we propose a novel token-level framework, LongAttn, which leverages the self-attention mechanism of LLMs to measure the long-range dependencies for the data. By calculating token-level dependency strength and distribution uniformity of token scores, LongAttn effectively quantifies long-range dependencies, enabling more accurate and efficient data selection. We filter LongABC-32K from open-source long-context datasets (ArXiv, Book, and Code). Through our comprehensive experiments, LongAttn has demonstrated its excellent effectiveness, scalability, and efficiency. To facilitate future research in long-context data, we released our code and the high-quality long-context training data LongABC-32K. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的发展，处理长上下文的需求日益增加，因此在这一领域取得重要进展变得尤为重要。为了增强长上下文能力，构建具有长期依赖关系的高质量训练数据至关重要。现有的长上下文数据选择方法往往依赖于句子级别的分析，这在性能和效率上都有很大的优化空间。在本文中，我们提出了一种新颖的令牌级别框架——LongAttn，利用LLMs的自注意力机制来衡量数据中的长期依赖关系。通过计算令牌级别依赖关系强度和令牌评分分布的均匀性，LongAttn有效地量化了长期依赖关系，从而实现更准确和高效的数据选择。我们从开源长上下文数据集（ArXiv、Book和Code）中筛选出了LongABC-32K。通过我们全面的实验，LongAttn展示了其卓越的效果、可扩展性和效率。为了促进未来长上下文数据的研究，我们发布了我们的代码和高质量的长上下文训练数据LongABC-32K。 

---
# Sarang at DEFACTIFY 4.0: Detecting AI-Generated Text Using Noised Data and an Ensemble of DeBERTa Models 

**Title (ZH)**: Sarang 在 DEFACTIFY 4.0：使用噪声数据和 DeBERTa 模型集成检测 AI 生成文本 

**Authors**: Avinash Trivedi, Sangeetha Sivanesan  

**Link**: [PDF](https://arxiv.org/pdf/2502.16857)  

**Abstract**: This paper presents an effective approach to detect AI-generated text, developed for the Defactify 4.0 shared task at the fourth workshop on multimodal fact checking and hate speech detection. The task consists of two subtasks: Task-A, classifying whether a text is AI generated or human written, and Task-B, classifying the specific large language model that generated the text. Our team (Sarang) achieved the 1st place in both tasks with F1 scores of 1.0 and 0.9531, respectively. The methodology involves adding noise to the dataset to improve model robustness and generalization. We used an ensemble of DeBERTa models to effectively capture complex patterns in the text. The result indicates the effectiveness of our noise-driven and ensemble-based approach, setting a new standard in AI-generated text detection and providing guidance for future developments. 

**Abstract (ZH)**: 本文介绍了一种有效的方法，用于检测AI生成的文本，并在第四届多模态事实核查和仇恨言论检测工作坊的Defactify 4.0 共享任务中开发。该任务包含两个子任务：Task-A，判断文本是由AI生成还是由人类撰写；Task-B，确定具体是哪个大型语言模型生成了该文本。我们团队（Sarang）在这两个子任务中均取得了最佳成绩，F1分数分别为1.0和0.9531。该方法通过向数据集中添加噪声来提升模型的稳健性和泛化能力。我们使用了DeBERTa模型的集成，以有效捕捉文本中的复杂模式。结果表明，我们的噪声驱动和集成方法的有效性，为AI生成的文本检测树立了新的标准，并为未来的发展提供了指导。 

---
# "Actionable Help" in Crises: A Novel Dataset and Resource-Efficient Models for Identifying Request and Offer Social Media Posts 

**Title (ZH)**: 在危机中的“可行动帮助”：一个新型数据集和资源高效模型，用于识别求助和施助的社交媒体帖子 

**Authors**: Rabindra Lamsal, Maria Rodriguez Read, Shanika Karunasekera, Muhammad Imran  

**Link**: [PDF](https://arxiv.org/pdf/2502.16839)  

**Abstract**: During crises, social media serves as a crucial coordination tool, but the vast influx of posts--from "actionable" requests and offers to generic content like emotional support, behavioural guidance, or outdated information--complicates effective classification. Although generative LLMs (Large Language Models) can address this issue with few-shot classification, their high computational demands limit real-time crisis response. While fine-tuning encoder-only models (e.g., BERT) is a popular choice, these models still exhibit higher inference times in resource-constrained environments. Moreover, although distilled variants (e.g., DistilBERT) exist, they are not tailored for the crisis domain. To address these challenges, we make two key contributions. First, we present CrisisHelpOffer, a novel dataset of 101k tweets collaboratively labelled by generative LLMs and validated by humans, specifically designed to distinguish actionable content from noise. Second, we introduce the first crisis-specific mini models optimized for deployment in resource-constrained settings. Across 13 crisis classification tasks, our mini models surpass BERT (also outperform or match the performance of RoBERTa, MPNet, and BERTweet), offering higher accuracy with significantly smaller sizes and faster speeds. The Medium model is 47% smaller with 3.8% higher accuracy at 3.5x speed, the Small model is 68% smaller with a 1.8% accuracy gain at 7.7x speed, and the Tiny model, 83% smaller, matches BERT's accuracy at 18.6x speed. All models outperform existing distilled variants, setting new benchmarks. Finally, as a case study, we analyze social media posts from a global crisis to explore help-seeking and assistance-offering behaviours in selected developing and developed countries. 

**Abstract (ZH)**: 在危机期间，社交媒体作为关键的协调工具发挥了重要作用，但海量的帖子（包括行动呼吁、服务提供以及情感支持、行为指导或过时信息等）使得有效分类变得复杂。虽然生成式大型语言模型（LLM）可以通过少样本学习分类，但其高计算需求限制了实时危机应对能力。尽管针对这一问题，微调编码器模型（如BERT）是一个流行的选择，但在资源受限的环境中，这些模型仍表现出较长的推理时间。虽然已存在精简变体（如DistilBERT），但它们并未专门针对危机领域。为应对这些挑战，我们做出了两项关键贡献。首先，我们提出了CrisisHelpOffer，这是一个包含101,000条推文的新数据集，这些推文是由生成式大型语言模型联合标注，并由人类验证，特别设计用于区分行动内容和噪音。其次，我们引入了第一个针对特定危机场景并优化部署的迷你模型，适用于资源受限的环境。在13项危机分类任务中，我们的迷你模型在准确性、模型大小和处理速度方面都超过了BERT（同时优于或匹配RoBERTa、MPNet和BERTweet的表现），提供了更高精度且更小的模型。Medium模型体积小47%，准确率提高3.8%，速度提升3.5倍；Small模型体积减少68%，准确率提高1.8%，速度提高7.7倍；Tiny模型体积减少83%，在18.6倍的速度下达到了与BERT相同的准确率。所有模型均优于现有精简变体，树立了新的基准。最后，作为案例研究，我们分析了全球危机期间的社交媒体帖子，以探讨在选定的发展中国家和发达国家中求助和提供帮助的行为。 

---
# REGen: A Reliable Evaluation Framework for Generative Event Argument Extraction 

**Title (ZH)**: REGen：一种可靠的生成事件论元提取评估框架 

**Authors**: Omar Sharif, Joseph Gatto, Madhusudan Basak, Sarah M. Preum  

**Link**: [PDF](https://arxiv.org/pdf/2502.16838)  

**Abstract**: Event argument extraction identifies arguments for predefined event roles in text. Traditional evaluations rely on exact match (EM), requiring predicted arguments to match annotated spans exactly. However, this approach fails for generative models like large language models (LLMs), which produce diverse yet semantically accurate responses. EM underestimates performance by disregarding valid variations, implicit arguments (unstated but inferable), and scattered arguments (distributed across a document). To bridge this gap, we introduce Reliable Evaluation framework for Generative event argument extraction (REGen), a framework that better aligns with human judgment. Across six datasets, REGen improves performance by an average of 23.93 F1 points over EM. Human validation further confirms REGen's effectiveness, achieving 87.67% alignment with human assessments of argument correctness. 

**Abstract (ZH)**: 事件论元提取识别文本中预定义事件角色的论元。传统的评估方法依赖于精确匹配（EM），要求预测的论元与标注的片段完全一致。然而，这种方法对于生成型模型（如大型语言模型LLMs）并不适用，因为这些模型生成的是多样化的、语义上准确的回应。精确匹配通过忽略有效的变化、隐含论元（未陈述但可推断的）和散落在文档中的论元来低估模型性能。为解决这一问题，我们提出了可靠的生成事件论元提取评估框架（REGen），该框架更符合人类判断。在六个数据集上，REGen在F1得分上平均提高了23.93点，优于精确匹配方法。进一步的人类验证也证实了REGen的有效性，其与人类对论元正确性的评估达到了87.67%的一致性。 

---
# Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization 

**Title (ZH)**: 找到那个关键点：构建偏好数据以实现偏好优化的扩展 

**Authors**: Yao Xiao, Hai Ye, Linyao Chen, Hwee Tou Ng, Lidong Bing, Xiaoli Li, Roy Ka-wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.16825)  

**Abstract**: Iterative data generation and model retraining are widely used to align large language models (LLMs). It typically involves a policy model to generate on-policy responses and a reward model to guide training data selection. Direct Preference Optimization (DPO) further enhances this process by constructing preference pairs of chosen and rejected responses. In this work, we aim to \emph{scale up} the number of on-policy samples via repeated random sampling to improve alignment performance. Conventional practice selects the sample with the highest reward as chosen and the lowest as rejected for DPO. However, our experiments reveal that this strategy leads to a \emph{decline} in performance as the sample size increases. To address this, we investigate preference data construction through the lens of underlying normal distribution of sample rewards. We categorize the reward space into seven representative points and systematically explore all 21 ($C_7^2$) pairwise combinations. Through evaluations on four models using AlpacaEval 2, we find that selecting the rejected response at reward position $\mu - 2\sigma$ rather than the minimum reward, is crucial for optimal performance. We finally introduce a scalable preference data construction strategy that consistently enhances model performance as the sample scale increases. 

**Abstract (ZH)**: 迭代数据生成和模型重新训练广泛用于对齐大型语言模型（LLMs）。这一过程通常包括一个策略模型生成与策略相符的响应，以及一个奖励模型指导训练数据的选择。直接偏好优化（DPO）进一步通过构建被选中和被拒绝响应的偏好对来增强这一过程。在这项工作中，我们旨在通过重复随机采样来增加与策略相符的样本数量，以提高对齐性能。传统的做法是选择奖励最高的样本作为被选中样本，选择奖励最低的样本作为被拒绝样本。然而，我们的实验表明，随着样本数量的增加，这种策略会导致性能下降。为了解决这一问题，我们从样本奖励的潜在正态分布角度出发，研究如何构建偏好数据。我们将奖励空间分为七种代表性点，并系统地探索所有21种（$C_7^2$）成对组合。通过使用AlpacaEval 2在四种模型上的评估，我们发现选择在奖励位置为$\mu - 2\sigma$的被拒绝响应，而非选择最低奖励，对于获得最佳性能至关重要。最后，我们引入了一种可扩展的偏好数据构建策略，该策略能够随样本量的增加一致地提升模型性能。 

---
# Uncertainty Quantification of Large Language Models through Multi-Dimensional Responses 

**Title (ZH)**: 通过多维度反应对大型语言模型的不确定性量化 

**Authors**: Tiejin Chen, Xiaoou Liu, Longchao Da, Xiaoou Liu, Vagelis Papalexakis, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.16820)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks due to large training datasets and powerful transformer architecture. However, the reliability of responses from LLMs remains a question. Uncertainty quantification (UQ) of LLMs is crucial for ensuring their reliability, especially in areas such as healthcare, finance, and decision-making. Existing UQ methods primarily focus on semantic similarity, overlooking the deeper knowledge dimensions embedded in responses. We introduce a multi-dimensional UQ framework that integrates semantic and knowledge-aware similarity analysis. By generating multiple responses and leveraging auxiliary LLMs to extract implicit knowledge, we construct separate similarity matrices and apply tensor decomposition to derive a comprehensive uncertainty representation. This approach disentangles overlapping information from both semantic and knowledge dimensions, capturing both semantic variations and factual consistency, leading to more accurate UQ. Our empirical evaluations demonstrate that our method outperforms existing techniques in identifying uncertain responses, offering a more robust framework for enhancing LLM reliability in high-stakes applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）由于庞大的训练数据集和强大的变压器架构，在各种任务中展现了卓越的能力。然而，LLMs 回答的可靠性依然存在问题。量化大型语言模型的不确定性（UQ）对于确保其可靠性至关重要，尤其是在医疗保健、金融和决策等领域。现有的 UQ 方法主要侧重于语义相似性分析，忽略了响应中嵌入的更深层次的知识维度。我们提出了一种多维度 UQ 框架，结合语义和知识感知相似性分析。通过生成多个响应并利用辅助 LLM 提取隐含知识，我们构建了独立的相似矩阵，并应用张量分解以获得综合的不确定性表示。该方法能够将语义和知识维度中的重叠信息分开，捕捉语义变化和事实一致性，从而提高 UQ 的准确性。实证评估结果表明，我们的方法在识别不确定响应方面优于现有技术，为在高风险应用中增强 LLM 可靠性提供了更稳健的框架。 

---
# CoT2Align: Cross-Chain of Thought Distillation via Optimal Transport Alignment for Language Models with Different Tokenizers 

**Title (ZH)**: CoT2Align：通过最优传输对齐实现不同分词器语言模型的跨链思维提炼 

**Authors**: Anh Duc Le, Tu Vu, Nam Le Hai, Nguyen Thi Ngoc Diep, Linh Ngo Van, Trung Le, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16806)  

**Abstract**: Large Language Models (LLMs) achieve state-of-the-art performance across various NLP tasks but face deployment challenges due to high computational costs and memory constraints. Knowledge distillation (KD) is a promising solution, transferring knowledge from large teacher models to smaller student models. However, existing KD methods often assume shared vocabularies and tokenizers, limiting their flexibility. While approaches like Universal Logit Distillation (ULD) and Dual-Space Knowledge Distillation (DSKD) address vocabulary mismatches, they overlook the critical \textbf{reasoning-aware distillation} aspect. To bridge this gap, we propose CoT2Align a universal KD framework that integrates Chain-of-Thought (CoT) augmentation and introduces Cross-CoT Alignment to enhance reasoning transfer. Additionally, we extend Optimal Transport beyond token-wise alignment to a sequence-level and layer-wise alignment approach that adapts to varying sequence lengths while preserving contextual integrity. Comprehensive experiments demonstrate that CoT2Align outperforms existing KD methods across different vocabulary settings, improving reasoning capabilities and robustness in domain-specific tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理（NLP）任务上取得了最先进的性能，但由于高计算成本和内存约束，面临着部署挑战。知识蒸馏（KD）是一种有前景的解决方案，它可以通过将大型教师模型的知识转移到较小的学生模型上来实现。然而，现有的KD方法通常假设共享词汇表和分词器，这限制了其灵活性。尽管有类似通用逻辑蒸馏（ULD）和双空间知识蒸馏（DSKD）的方法能够解决词汇表不匹配的问题，但这些方法忽视了关键的\textbf{推理意识蒸馏}方面。为了弥补这一差距，我们提出了一种名为CoT2Align的通用KD框架，该框架结合了Chain-of-Thought（CoT）增强和跨CoT对齐（Cross-CoT Alignment），以增强推理转移。此外，我们还将最优传输方法扩展到了序列级和层级对齐，该方法能够适应不同的序列长度并保留上下文完整性。详细的实验结果表明，CoT2Align在不同词汇设置下优于现有的KD方法，在特定领域任务中提高了推理能力和鲁棒性。 

---
# Unsupervised Topic Models are Data Mixers for Pre-training Language Models 

**Title (ZH)**: 无监督主题模型是预训练语言模型的数据混合器 

**Authors**: Jiahui Peng, Xinlin Zhuang, Qiu Jiantao, Ren Ma, Jing Yu, Tianyi Bai, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2502.16802)  

**Abstract**: The performance of large language models (LLMs) is significantly affected by the quality and composition of their pre-training data, which is inherently diverse, spanning various domains, sources, and topics. Effectively integrating these heterogeneous data sources is crucial for optimizing LLM performance. Previous research has predominantly concentrated on domain-based data mixing, often neglecting the nuanced topic-level characteristics of the data. To address this gap, we propose a simple yet effective topic-based data mixing strategy that utilizes fine-grained topics generated through our topic modeling method, DataWeave. DataWeave employs a multi-stage clustering process to group semantically similar documents and utilizes LLMs to generate detailed topics, thereby facilitating a more nuanced understanding of dataset composition. Our strategy employs heuristic methods to upsample or downsample specific topics, which significantly enhances LLM performance on downstream tasks, achieving superior results compared to previous, more complex data mixing approaches. Furthermore, we confirm that the topics Science and Relationships are particularly effective, yielding the most substantial performance improvements. We will make our code and datasets publicly available. 

**Abstract (ZH)**: 大型语言模型（LLMs）的表现受到其预训练数据的质量和组成的影响，这些数据本质上是多样的，涉及多个领域、来源和主题。有效地整合这些异质数据源对于优化LLM表现至关重要。以往的研究主要集中在基于领域的数据混合适配，往往忽略了数据在主题层面的细微差异。为了解决这一问题，我们提出了一种简单而有效的基于主题的数据混合适配策略，该策略利用了通过我们的话题建模方法DataWeave生成的细粒度主题。DataWeave采用多阶段聚类过程来分组语义上相似的文档，并利用LLM生成详细主题，从而促进了对数据集组成更细致的理解。我们的策略利用启发式方法对特定主题进行上采样或下采样，这显著提升了LLM在下游任务上的表现，取得了优于之前更为复杂的数据混合适配方法的结果。此外，我们确认了“科学”和“关系”两个主题特别有效，带来了最大的性能提升。我们将公开我们的代码和数据集。 

---
# Are Large Language Models Good Data Preprocessors? 

**Title (ZH)**: 大型语言模型是良好的数据预处理器吗？ 

**Authors**: Elyas Meguellati, Nardiena Pratama, Shazia Sadiq, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2502.16790)  

**Abstract**: High-quality textual training data is essential for the success of multimodal data processing tasks, yet outputs from image captioning models like BLIP and GIT often contain errors and anomalies that are difficult to rectify using rule-based methods. While recent work addressing this issue has predominantly focused on using GPT models for data preprocessing on relatively simple public datasets, there is a need to explore a broader range of Large Language Models (LLMs) and tackle more challenging and diverse datasets.
In this study, we investigate the use of multiple LLMs, including LLaMA 3.1 70B, GPT-4 Turbo, and Sonnet 3.5 v2, to refine and clean the textual outputs of BLIP and GIT. We assess the impact of LLM-assisted data cleaning by comparing downstream-task (SemEval 2024 Subtask "Multilabel Persuasion Detection in Memes") models trained on cleaned versus non-cleaned data. While our experimental results show improvements when using LLM-cleaned captions, statistical tests reveal that most of these improvements are not significant. This suggests that while LLMs have the potential to enhance data cleaning and repairing, their effectiveness may be limited depending on the context they are applied to, the complexity of the task, and the level of noise in the text.
Our findings highlight the need for further research into the capabilities and limitations of LLMs in data preprocessing pipelines, especially when dealing with challenging datasets, contributing empirical evidence to the ongoing discussion about integrating LLMs into data preprocessing pipelines. 

**Abstract (ZH)**: 高质量的文本训练数据对于多模态数据处理任务的成功至关重要，然而，如BLIP和GIT这类图像字幕模型的输出经常包含难以通过基于规则的方法纠正的错误和异常。尽管最近针对这一问题的研究主要集中于使用GPT模型进行简单公共数据集的预处理，但仍需探索更广泛的大型语言模型（LLMs），以应对更具挑战性和多样性的数据集。

在本研究中，我们调查了使用包括LLaMA 3.1 70B、GPT-4 Turbo和Sonnet 3.5 v2在内的多种LLMs，以改进和完善BLIP和GIT的文本输出。通过将下游任务（SemEval 2024子任务“多标签说客检测 meme”）模型分别训练在清洗后的和未清洗的数据上，评估LLM辅助数据清理的影响。虽然实验结果表明使用LLM清理的字幕有所改进，但统计测试显示这些改进在大多数情况下并不显著。这表明，虽然LLMs有潜力提高数据清理和修复的能力，但在具体应用情境、任务复杂性和文本噪声水平的限制下，其效果可能受到一定限制。

我们的研究结果强调了需要进一步研究LLMs在数据预处理管道中的能力和局限性，特别是在处理具有挑战性的数据集时。这些发现为在数据预处理管道中集成LLMs的持续讨论提供了实证证据。 

---
# MultiOCR-QA: Dataset for Evaluating Robustness of LLMs in Question Answering on Multilingual OCR Texts 

**Title (ZH)**: MultiOCR-QA：多语言OCR文本问答稳健性评估数据集 

**Authors**: Bhawna Piryani, Jamshid Mozafari, Abdelrahman Abdallah, Antoine Doucet, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2502.16781)  

**Abstract**: Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors -- imperfect extraction of the text, including character insertion, deletion and permutation -- can significantly impact downstream tasks like question-answering (QA). In this work, we introduce a multilingual QA dataset MultiOCR-QA, designed to analyze the effects of OCR noise on QA systems' performance. The MultiOCR-QA dataset comprises 60K question-answer pairs covering three languages, English, French, and German. The dataset is curated from OCR-ed old documents, allowing for the evaluation of OCR-induced challenges on question answering. We evaluate MultiOCR-QA on various levels and types of OCR errors to access the robustness of LLMs in handling real-world digitization errors. Our findings show that QA systems are highly prone to OCR induced errors and exhibit performance degradation on noisy OCR text. 

**Abstract (ZH)**: 光学字符识别（OCR）在数字化历史和多语言文件中发挥着重要作用，然而OCR错误——包括字符插入、删除和置换等不完善的文字提取——可能会对问题回答（QA）等下游任务产生显著影响。本文介绍了多语言QA数据集MultiOCR-QA，用于分析OCR噪声对QA系统性能的影响。MultiOCR-QA数据集包含60,000个问题-答案对，覆盖三种语言：英语、法语和德语。该数据集是从OCR处理的旧文档中精选出来的，从而评估由OCR引起的问答挑战。我们从不同层次和类型的角度评估MultiOCR-QA，以评估大型语言模型（LLMs）在处理现实世界数字化错误方面的稳健性。我们的研究结果表明，QA系统对OCR引起的错误非常敏感，并且在噪声OCR文本上表现出性能下降。 

---
# AISafetyLab: A Comprehensive Framework for AI Safety Evaluation and Improvement 

**Title (ZH)**: AISafetyLab：全面的AI安全性评估与改进框架 

**Authors**: Zhexin Zhang, Leqi Lei, Junxiao Yang, Xijie Huang, Yida Lu, Shiyao Cui, Renmiao Chen, Qinglin Zhang, Xinyuan Wang, Hao Wang, Hao Li, Xianqi Lei, Chengwei Pan, Lei Sha, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16776)  

**Abstract**: As AI models are increasingly deployed across diverse real-world scenarios, ensuring their safety remains a critical yet underexplored challenge. While substantial efforts have been made to evaluate and enhance AI safety, the lack of a standardized framework and comprehensive toolkit poses significant obstacles to systematic research and practical adoption. To bridge this gap, we introduce AISafetyLab, a unified framework and toolkit that integrates representative attack, defense, and evaluation methodologies for AI safety. AISafetyLab features an intuitive interface that enables developers to seamlessly apply various techniques while maintaining a well-structured and extensible codebase for future advancements. Additionally, we conduct empirical studies on Vicuna, analyzing different attack and defense strategies to provide valuable insights into their comparative effectiveness. To facilitate ongoing research and development in AI safety, AISafetyLab is publicly available at this https URL, and we are committed to its continuous maintenance and improvement. 

**Abstract (ZH)**: 随着人工智能模型被越来越广泛地应用于各种实际场景中，确保其安全性仍然是一个至关重要的但尚未充分探索的挑战。尽管在评估和提高AI安全性方面已经做出了大量努力，但由于缺乏标准化框架和完整的工具包，系统研究和实际应用仍面临重大障碍。为解决这一问题，我们提出了AISafetyLab，这是一种统一的框架和工具包，集成了代表性的攻击、防御和评估方法，以确保AI安全。AISafetyLab具有直观的界面，使开发人员能够无缝地应用多种技术，同时保持一个结构清晰、易于扩展的代码库，以便未来的发展。此外，我们对Vicuna进行了实证研究，分析不同的攻击和防御策略，以提供其相对效果的宝贵见解。为了促进AI安全领域的持续研究和开发，AISafetyLab已公开发布，您可以在此处访问：[请将此链接补充完整]，我们也致力于其持续维护和改进。 

---
# LED-Merging: Mitigating Safety-Utility Conflicts in Model Merging with Location-Election-Disjoint 

**Title (ZH)**: LED-Merging: 减轻位置选举不交集情况下模型融合中的安全与实用性冲突

在这个翻译中，“LED-Merging”保持不变，因为它是作者自定义的术语或命名。其余部分按照学术规范翻译，确保准确传达原文意思。 

**Authors**: Qianli Ma, Dongrui Liu, Qian Chen, Linfeng Zhang, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.16770)  

**Abstract**: Fine-tuning pre-trained Large Language Models (LLMs) for specialized tasks incurs substantial computational and data costs. While model merging offers a training-free solution to integrate multiple task-specific models, existing methods suffer from safety-utility conflicts where enhanced general capabilities degrade safety safeguards. We identify two root causes: \textbf{neuron misidentification} due to simplistic parameter magnitude-based selection, and \textbf{cross-task neuron interference} during merging. To address these challenges, we propose \textbf{LED-Merging}, a three-stage framework that \textbf{L}ocates task-specific neurons via gradient-based attribution, dynamically \textbf{E}lects critical neurons through multi-model importance fusion, and \textbf{D}isjoints conflicting updates through parameter isolation. Extensive experiments on Llama-3-8B, Mistral-7B, and Llama2-13B demonstrate that LED-Merging reduces harmful response rates(\emph{e.g.}, a 31.4\% decrease on Llama-3-8B-Instruct on HarmBench) while preserving 95\% of utility performance(\emph{e.g.}, 52.39\% accuracy on GSM8K). LED-Merging resolves safety-utility conflicts and provides a lightweight, training-free paradigm for constructing reliable multi-task LLMs. 

**Abstract (ZH)**: 对预训练大规模语言模型（LLMs）进行细调以适应特定任务会带来显著的计算和数据成本。虽然模型合并提供了一种无需训练就能整合多个任务特定模型的解决方案，但现有方法在安全性与实用性之间存在冲突，在增强通用能力的同时削弱了安全保护。我们识别出两个根本原因：\textbf{由于基于参数量级选择的简陋神经元识别导致的错识}，以及\textbf{合并过程中不同任务神经元的相互干扰}。为了解决这些挑战，我们提出了一种名为\textbf{LED-Merging}的三阶段框架：通过梯度归因定位特定任务的神经元，通过多模型重要性融合动态选择关键神经元，并通过参数隔离消解冲突的更新。在Llama-3-8B、Mistral-7B和Llama2-13B上的大量实验表明，LED-Merging在减少有害响应率（例如，在HarmBench上的Llama-3-8B-Instruct减少了31.4%）的同时，保持了95%的实用性性能（例如，在GSM8K上的准确率为52.39%）。LED-Merging解决了安全性和实用性之间的冲突，并提供了一种轻量级、无需训练的范式来构建可靠多任务LLMs。 

---
# A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts 

**Title (ZH)**: 一种综合方法用于监管文本的信息检索与答案生成 

**Authors**: Jhon Rayo, Raul de la Rosa, Mario Garrido  

**Link**: [PDF](https://arxiv.org/pdf/2502.16767)  

**Abstract**: Regulatory texts are inherently long and complex, presenting significant challenges for information retrieval systems in supporting regulatory officers with compliance tasks. This paper introduces a hybrid information retrieval system that combines lexical and semantic search techniques to extract relevant information from large regulatory corpora. The system integrates a fine-tuned sentence transformer model with the traditional BM25 algorithm to achieve both semantic precision and lexical coverage. To generate accurate and comprehensive responses, retrieved passages are synthesized using Large Language Models (LLMs) within a Retrieval Augmented Generation (RAG) framework. Experimental results demonstrate that the hybrid system significantly outperforms standalone lexical and semantic approaches, with notable improvements in Recall@10 and MAP@10. By openly sharing our fine-tuned model and methodology, we aim to advance the development of robust natural language processing tools for compliance-driven applications in regulatory domains. 

**Abstract (ZH)**: 监管文本天生具有较长和复杂的特点，这为信息检索系统在支持监管官员合规任务时带来了巨大的挑战。本文介绍了一种结合词法和语义搜索技术的混合信息检索系统，该系统可以从大型监管档案库中提取相关的信息。该系统将微调的句子变换模型与传统的BM25算法相结合，以实现语义精度和词法覆盖的双重目标。为了生成准确和全面的响应，检索到的段落通过检索增强生成（RAG）框架内的大型语言模型（LLMs）进行综合处理。实验结果表明，该混合系统显著优于单独的词法和语义方法，在Recall@10和MAP@10方面有显著改进。通过公开分享我们的微调模型和方法，我们旨在推动监管领域基于合规的应用中稳健自然语言处理工具的发展。 

---
# ATEB: Evaluating and Improving Advanced NLP Tasks for Text Embedding Models 

**Title (ZH)**: ATEB: 评估和改进用于文本嵌入模型的高级NLP任务 

**Authors**: Simeng Han, Frank Palma Gomez, Tu Vu, Zefei Li, Daniel Cer, Hansi Zeng, Chris Tar, Arman Cohan, Gustavo Hernandez Abrego  

**Link**: [PDF](https://arxiv.org/pdf/2502.16766)  

**Abstract**: Traditional text embedding benchmarks primarily evaluate embedding models' capabilities to capture semantic similarity. However, more advanced NLP tasks require a deeper understanding of text, such as safety and factuality. These tasks demand an ability to comprehend and process complex information, often involving the handling of sensitive content, or the verification of factual statements against reliable sources. We introduce a new benchmark designed to assess and highlight the limitations of embedding models trained on existing information retrieval data mixtures on advanced capabilities, which include factuality, safety, instruction following, reasoning and document-level understanding. This benchmark includes a diverse set of tasks that simulate real-world scenarios where these capabilities are critical and leads to identification of the gaps of the currently advanced embedding models. Furthermore, we propose a novel method that reformulates these various tasks as retrieval tasks. By framing tasks like safety or factuality classification as retrieval problems, we leverage the strengths of retrieval models in capturing semantic relationships while also pushing them to develop a deeper understanding of context and content. Using this approach with single-task fine-tuning, we achieved performance gains of 8\% on factuality classification and 13\% on safety classification. Our code and data will be publicly available. 

**Abstract (ZH)**: 传统文本嵌入基准主要评估嵌入模型捕获语义相似性的能力。然而，更高级的NLP任务需要对文本有更深入的理解，比如安全性与事实性。这些任务要求模型能够理解和处理复杂信息，常常涉及敏感内容的处理或对事实陈述进行核验，以可靠的数据源为准。我们引入一个新的基准，旨在评估和突出现有的信息检索数据混合训练的嵌入模型在高级能力上的局限性，这些能力包括事实性、安全性、指令遵守能力、推理和文档级理解。该基准包含了多种模拟真实世界场景的任务，这些场景中高级能力至关重要，有助于识别当前先进嵌入模型的能力差距。此外，我们提出了一种新颖的方法，将这些不同的任务重新表述为检索任务。通过将安全性或事实性分类等任务重新形式化为检索问题，我们能够利用检索模型在捕获语义关系方面的优势，同时促使它们发展更深层次的上下文和内容理解能力。使用这种方法进行单任务微调，我们在事实性分类任务上的性能提高了8%，在安全性分类任务上提高了13%。我们的代码和数据将公开发布。 

---
# Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions 

**Title (ZH)**: 针对大规模调查数据的语言模型微调以预测公众意见分布 

**Authors**: Joseph Suh, Erfan Jahanparast, Suhong Moon, Minwoo Kang, Serina Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16761)  

**Abstract**: Large language models (LLMs) present novel opportunities in public opinion research by predicting survey responses in advance during the early stages of survey design. Prior methods steer LLMs via descriptions of subpopulations as LLMs' input prompt, yet such prompt engineering approaches have struggled to faithfully predict the distribution of survey responses from human subjects. In this work, we propose directly fine-tuning LLMs to predict response distributions by leveraging unique structural characteristics of survey data. To enable fine-tuning, we curate SubPOP, a significantly scaled dataset of 3,362 questions and 70K subpopulation-response pairs from well-established public opinion surveys. We show that fine-tuning on SubPOP greatly improves the match between LLM predictions and human responses across various subpopulations, reducing the LLM-human gap by up to 46% compared to baselines, and achieves strong generalization to unseen surveys and subpopulations. Our findings highlight the potential of survey-based fine-tuning to improve opinion prediction for diverse, real-world subpopulations and therefore enable more efficient survey designs. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在公共意见研究中提供了新的机遇，可以在调查设计的早期阶段提前预测调查响应。先前的方法通过向LLMs输入问题描述子群体来引导LLMs，但这种提示工程方法在准确预测人类受试者调查响应分布方面遇到了困难。在本项工作中，我们提出通过利用调查数据的独特结构特征直接微调LLMs来预测响应分布。为了实现微调，我们整理了SubPOP数据集，这是一个规模显著扩大的数据集，包含3,362个问题和70,000个子群体响应配对，这些数据来自现有的一些公共意见调查。我们展示了在SubPOP上进行微调可以大幅改善LLMs预测与人类响应的一致性，与baseline相比，将LLMs与人类的差距降低了多达46%，并且实现了对未见过的调查和子群体的强大泛化能力。我们的研究结果突显了基于调查的微调在提高对多样化真实世界子群体意见预测方面的能力，从而有助于更高效的调查设计。我们的代码可以在以下网址获取：this <URL>。 

---
# Entailment-Preserving First-order Logic Representations in Natural Language Entailment 

**Title (ZH)**: 自然语言蕴含中保持蕴含关系的一阶逻辑表示 

**Authors**: Jinu Lee, Qi Liu, Runzhi Ma, Vincent Han, Ziqi Wang, Heng Ji, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2502.16757)  

**Abstract**: First-order logic (FOL) can represent the logical entailment semantics of natural language (NL) sentences, but determining natural language entailment using FOL remains a challenge. To address this, we propose the Entailment-Preserving FOL representations (EPF) task and introduce reference-free evaluation metrics for EPF, the Entailment-Preserving Rate (EPR) family. In EPF, one should generate FOL representations from multi-premise natural language entailment data (e.g. EntailmentBank) so that the automatic prover's result preserves the entailment labels. Experiments show that existing methods for NL-to-FOL translation struggle in EPF. To this extent, we propose a training method specialized for the task, iterative learning-to-rank, which directly optimizes the model's EPR score through a novel scoring function and a learning-to-rank objective. Our method achieves a 1.8-2.7% improvement in EPR and a 17.4-20.6% increase in EPR@16 compared to diverse baselines in three datasets. Further analyses reveal that iterative learning-to-rank effectively suppresses the arbitrariness of FOL representation by reducing the diversity of predicate signatures, and maintains strong performance across diverse inference types and out-of-domain data. 

**Abstract (ZH)**: 一阶逻辑（FOL）能够表示自然语言（NL）句子的逻辑蕴含语义，但利用FOL确定自然语言蕴含关系仍然是一个挑战。为了解决这一问题，我们提出了保持蕴含的一阶逻辑表示（Entailment-Preserving FOL Representations, EPF）任务，并引入了EPF的参考无关评估指标，即Entailment-Preserving Rate（EPR）家族。在EPF任务中，应从多前提自然语言蕴含数据（例如EntailmentBank）中生成一阶逻辑表示，使得自动证明器的结果能够保护断言标签。实验表明，现有的NL到FOL转换方法在EPF任务中表现不佳。为此，我们提出了一种专门针对该任务的训练方法——迭代学习排序，该方法通过一个新颖的评分函数和学习排序目标直接优化模型的EPR分数。我们的方法在三个数据集上的EPR得分提高了1.8%-2.7%，EPR@16提高了17.4%-20.6%，明显优于多种基线方法。进一步的分析表明，迭代学习排序通过减少谓词签名的多样性有效地抑制了一阶逻辑表示的任意性，并且能够在不同的推理类型和域外数据上保持较强的性能。 

---
# SQLong: Enhanced NL2SQL for Longer Contexts with LLMs 

**Title (ZH)**: SQLong：通过大型语言模型增强长上下文的自然语言到结构化查询语言转换 

**Authors**: Dai Quoc Nguyen, Cong Duy Vu Hoang, Duy Vu, Gioacchino Tangari, Thanh Tien Vu, Don Dharmasiri, Yuan-Fang Li, Long Duong  

**Link**: [PDF](https://arxiv.org/pdf/2502.16747)  

**Abstract**: Open-weight large language models (LLMs) have significantly advanced performance in the Natural Language to SQL (NL2SQL) task. However, their effectiveness diminishes when dealing with large database schemas, as the context length increases. To address this limitation, we present SQLong, a novel and efficient data augmentation framework designed to enhance LLM performance in long-context scenarios for the NL2SQL task. SQLong generates augmented datasets by extending existing database schemas with additional synthetic CREATE TABLE commands and corresponding data rows, sampled from diverse schemas in the training data. This approach effectively simulates long-context scenarios during finetuning and evaluation. Through experiments on the Spider and BIRD datasets, we demonstrate that LLMs finetuned with SQLong-augmented data significantly outperform those trained on standard datasets. These imply SQLong's practical implementation and its impact on improving NL2SQL capabilities in real-world settings with complex database schemas. 

**Abstract (ZH)**: 开源重量型大型语言模型（LLMs）在自然语言到SQL（NL2SQL）任务中取得了显著的进步。然而，当处理大型数据库模式时，随着上下文长度的增加，其效果会减弱。为了解决这一限制，我们提出了SQLong，这是一种新颖且高效的數據増强框架，旨在增强LLMs在长上下文场景下对NL2SQL任务的性能。SQLong通过在现有数据库模式中扩展额外的合成CREATE TABLE命令及其相应的数据行来生成増强的数据集，这些命令和数据行是从训练数据中的多个模式中抽样的。这种方法在微调和评估过程中有效地模拟了长上下文场景。通过在Spider和BIRD数据集上的实验，我们证明了使用SQLong增强数据微调的LLMs明显优于使用标准数据集训练的LLMs。这些结果表明SQLong的实际应用价值以及其在具有复杂数据库模式的实际场景中提升NL2SQL能力的影响。 

---
# Layer-Wise Evolution of Representations in Fine-Tuned Transformers: Insights from Sparse AutoEncoders 

**Title (ZH)**: 微调变压器中表示的分层进化：来自稀疏自编码器的见解 

**Authors**: Suneel Nadipalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.16722)  

**Abstract**: Fine-tuning pre-trained transformers is a powerful technique for enhancing the performance of base models on specific tasks. From early applications in models like BERT to fine-tuning Large Language Models (LLMs), this approach has been instrumental in adapting general-purpose architectures for specialized downstream tasks. Understanding the fine-tuning process is crucial for uncovering how transformers adapt to specific objectives, retain general representations, and acquire task-specific features. This paper explores the underlying mechanisms of fine-tuning, specifically in the BERT transformer, by analyzing activation similarity, training Sparse AutoEncoders (SAEs), and visualizing token-level activations across different layers. Based on experiments conducted across multiple datasets and BERT layers, we observe a steady progression in how features adapt to the task at hand: early layers primarily retain general representations, middle layers act as a transition between general and task-specific features, and later layers fully specialize in task adaptation. These findings provide key insights into the inner workings of fine-tuning and its impact on representation learning within transformer architectures. 

**Abstract (ZH)**: 微调预训练变换器是一种增强基模型在特定任务上性能的强大技术。从早期应用于像BERT这样的模型到对大规模语言模型（LLMs）进行微调，这一方法在适应通用架构以用于专门的下游任务方面发挥了关键作用。理解微调过程对于揭示变换器如何适应特定目标、保留通用表示以及获取任务特定特征至关重要。本文通过对BERT变换器进行微调的底层机制进行探索，通过分析激活相似性、训练稀疏自编码器（SAEs）以及可视化不同层的令牌级激活来展开研究。基于在多个数据集和BERT层上的实验，我们观察到特征适应任务的过程具有稳定的进展：早期层主要保留通用表示，中间层作为通用特征和任务特定特征之间的过渡，而后期层则完全专注于任务适应。这些发现为理解微调过程及其对变换器架构中表示学习的影响提供了关键见解。 

---
# Speed and Conversational Large Language Models: Not All Is About Tokens per Second 

**Title (ZH)**: 速度与对话型大规模语言模型：不仅关乎每秒令牌数 

**Authors**: Javier Conde, Miguel González, Pedro Reviriego, Zhen Gao, Shanshan Liu, Fabrizio Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16721)  

**Abstract**: The speed of open-weights large language models (LLMs) and its dependency on the task at hand, when run on GPUs, is studied to present a comparative analysis of the speed of the most popular open LLMs. 

**Abstract (ZH)**: 研究在GPU上运行时，开放权重大语言模型（LLMs）的速度及其对当前任务的依赖关系，并对其进行比较分析，以评估最流行的开放LLMs的速度。 

---
# Beyond Pattern Recognition: Probing Mental Representations of LMs 

**Title (ZH)**: 超越模式识别：探究LM的心理表示 

**Authors**: Moritz Miller, Kumar Shridhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.16717)  

**Abstract**: Language Models (LMs) have demonstrated impressive capabilities in solving complex reasoning tasks, particularly when prompted to generate intermediate explanations. However, it remains an open question whether these intermediate reasoning traces represent a dynamic, evolving thought process or merely reflect sophisticated pattern recognition acquired during large scale pre training. Drawing inspiration from human cognition, where reasoning unfolds incrementally as new information is assimilated and internal models are continuously updated, we propose to delve deeper into the mental model of various LMs. We propose a new way to assess the mental modeling of LMs, where they are provided with problem details gradually, allowing each new piece of data to build upon and refine the model's internal representation of the task. We systematically compare this step by step mental modeling strategy with traditional full prompt methods across both text only and vision and text modalities. Experiments on the MathWorld dataset across different model sizes and problem complexities confirm that both text-based LLMs and multimodal LMs struggle to create mental representations, questioning how their internal cognitive processes work. 

**Abstract (ZH)**: 语言模型（LMs）已经展示了在解决复杂推理任务方面的出色能力，尤其是在被提示生成中间解释的情况下。然而，关于这些中间推理轨迹是否代表了一个动态、不断发展的思维过程，还是仅仅反映了大规模预训练过程中习得的高级特征识别能力，这个问题仍是一个开放性问题。借鉴人类认知过程，该过程在新信息不断被吸收并更新内部模型时逐步展开，我们提议更深入地探讨各种LM的心理模型。我们提出了一种新的方法来评估LM的心理建模能力，其中逐渐提供问题细节，使新获得的数据能够在此基础上构建和细化模型对任务的内部表示。我们系统地比较了这种逐步心理建模策略与传统的全提示方法在仅文本和视觉文本模态下的表现。在不同的模型规模和问题复杂性下对MathWorld数据集进行的实验表明，无论是基于文本的语言模型还是多模态语言模型都难以构建心理表征，这引发了对其内部认知过程如何运作的质疑。 

---
# Can ChatGPT Learn to Count Letters? 

**Title (ZH)**: ChatGPT能学习数字母吗？ 

**Authors**: Javier Conde, Gonzalo Martínez, Pedro Reviriego, Zhen Gao, Shanshan Liu, Fabrizio Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16705)  

**Abstract**: Large language models (LLMs) struggle on simple tasks such as counting the number of occurrences of a letter in a word. In this paper, we investigate if ChatGPT can learn to count letters and propose an efficient solution. 

**Abstract (ZH)**: 大型语言模型（LLMs）在一些简单的任务上表现不佳，例如计算单词中某个字母出现的次数。在本文中，我们探讨了ChatGPT是否能够学会进行字母计数，并提出了一种高效的方法。 

---
# Code Summarization Beyond Function Level 

**Title (ZH)**: 超出函数级别的心代码摘要 

**Authors**: Vladimir Makharev, Vladimir Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.16704)  

**Abstract**: Code summarization is a critical task in natural language processing and software engineering, which aims to generate concise descriptions of source code. Recent advancements have improved the quality of these summaries, enhancing code readability and maintainability. However, the content of a repository or a class has not been considered in function code summarization. This study investigated the effectiveness of code summarization models beyond the function level, exploring the impact of class and repository contexts on the summary quality. The study involved revising benchmarks for evaluating models at class and repository levels, assessing baseline models, and evaluating LLMs with in-context learning to determine the enhancement of summary quality with additional context. The findings revealed that the fine-tuned state-of-the-art CodeT5+ base model excelled in code summarization, while incorporating few-shot learning and retrieved code chunks from RAG significantly enhanced the performance of LLMs in this task. Notably, the Deepseek Coder 1.3B and Starcoder2 15B models demonstrated substantial improvements in metrics such as BLEURT, METEOR, and BLEU-4 at both class and repository levels. Repository-level summarization exhibited promising potential but necessitates significant computational resources and gains from the inclusion of structured context. Lastly, we employed the recent SIDE code summarization metric in our evaluation. This study contributes to refining strategies for prompt engineering, few-shot learning, and RAG, addressing gaps in benchmarks for code summarization at various levels. Finally, we publish all study details, code, datasets, and results of evaluation in the GitHub repository available at this https URL. 

**Abstract (ZH)**: 代码摘要是自然语言处理和软件工程中的一个重要任务，其目标是生成源代码的简洁描述。最近的发展提高了这些摘要的质量，增强了代码的可读性和可维护性。然而，现有的工作尚未考虑到函数代码摘要时仓库或类的内容。本研究探讨了超出函数级别进行代码摘要的有效性，研究了类和仓库上下文对摘要质量的影响。研究包括修订针对类和仓库级别的基准，评估基线模型，并通过内省学习评估语言模型（LLMs），以确定额外上下文对摘要质量的提升。研究结果表明，微调的最先进的CodeT5+基模型在代码摘要方面表现出色，而结合少量学习和从RAG检索代码片段显著提升了LLMs在该任务中的性能。值得注意的是，Deepseek Coder 1.3B和Starcoder2 15B模型在BLEURT、METEOR和BLEU-4等指标上，在类和仓库级别均表现出明显的改进。仓库级别的摘要显示出巨大的潜力，但需要大量的计算资源，并且可以通过包含结构化上下文来获得显著收益。最后，我们在评估中使用了最近的SIDE代码摘要度量。本研究为细化提示工程、少量学习和RAG等策略提供了贡献，填补了代码摘要不同级别基准的空白。最后，我们在这个 GitHub 仓库（此链接：https://github.com/...）中发布了所有研究细节、代码、数据集和评估结果。 

---
# Uncovering the Hidden Threat of Text Watermarking from Users with Cross-Lingual Knowledge 

**Title (ZH)**: 探索跨语言知识用户隐藏的文本水印威胁 

**Authors**: Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Mengxin Zheng, Yan Solihin, Qian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2502.16699)  

**Abstract**: In this study, we delve into the hidden threats posed to text watermarking by users with cross-lingual knowledge. While most research focuses on watermarking methods for English, there is a significant gap in evaluating these methods in cross-lingual contexts. This oversight neglects critical adversary scenarios involving cross-lingual users, creating uncertainty regarding the effectiveness of cross-lingual watermarking. We assess four watermarking techniques across four linguistically rich languages, examining watermark resilience and text quality across various parameters and attacks. Our focus is on a realistic scenario featuring adversaries with cross-lingual expertise, evaluating the adequacy of current watermarking methods against such challenges. 

**Abstract (ZH)**: 在本研究中，我们探讨了具有跨语言知识的用户对文本水印所带来的隐藏威胁。尽管大多数研究集中在英文的水印方法上，但在这方面忽视了跨语言背景下评估这些方法的重要缺口。这种忽视忽略了涉及跨语言用户的关键对手场景，从而对跨语言水印的有效性产生了不确定性。我们评估了四种水印技术在四种语言丰富度高的语言中的应用，跨参数和攻击方法考察了水印的抵抗力和文本质量。我们的重点是具有跨语言专业知识的现实对手情景，评估当前水印技术在应对这些挑战时的充分性。 

---
# Toward Responsible Federated Large Language Models: Leveraging a Safety Filter and Constitutional AI 

**Title (ZH)**: 负责任的联邦大型语言模型的发展：利用安全过滤器和宪法性人工智能 

**Authors**: Eunchung Noh, Jeonghun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2502.16691)  

**Abstract**: Recent research has increasingly focused on training large language models (LLMs) using federated learning, known as FedLLM. However, responsible AI (RAI), which aims to ensure safe responses, remains underexplored in the context of FedLLM. In FedLLM, client data used for training may contain harmful content, leading to unsafe LLMs that generate harmful responses. Aggregating such unsafe LLMs into the global model and distributing them to clients may result in the widespread deployment of unsafe LLMs. To address this issue, we incorporate two well-known RAI methods into FedLLM: the safety filter and constitutional AI. Our experiments demonstrate that these methods significantly enhance the safety of the LLM, achieving over a 20% improvement on AdvBench, a benchmark for evaluating safety performance. 

**Abstract (ZH)**: 最近的研究越来越多地将联邦学习（Federated Learning, FL）应用于训练大规模语言模型（Large Language Models, LLM），这种应用被称为FedLLM。然而，在FedLLM背景下，负责任的人工智能（Responsible AI, RAI），旨在确保生成安全响应，仍然研究不足。在FedLLM中，用于训练的客户端数据可能包含有害内容，导致生成有害响应的安全措施不足的LLM。将这些安全措施不足的LLM聚合到全局模型中，并将其分发给客户端，可能会导致不安全的LLM的广泛部署。为了解决这一问题，我们将两种知名的RAI方法纳入FedLLM：安全过滤器和宪法人工智能。我们的实验表明，这些方法显著提高了LLM的安全性，在AdvBench上（一个用于评估安全性能的基准测试）实现了超过20%的改进。 

---
# WildLong: Synthesizing Realistic Long-Context Instruction Data at Scale 

**Title (ZH)**: WildLong：大规模合成具有实际长上下文指令的数据 

**Authors**: Jiaxi Li, Xingxing Zhang, Xun Wang, Xiaolong Huang, Li Dong, Liang Wang, Si-Qing Chen, Wei Lu, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.16684)  

**Abstract**: Large language models (LLMs) with extended context windows enable tasks requiring extensive information integration but are limited by the scarcity of high-quality, diverse datasets for long-context instruction tuning. Existing data synthesis methods focus narrowly on objectives like fact retrieval and summarization, restricting their generalizability to complex, real-world tasks. WildLong extracts meta-information from real user queries, models co-occurrence relationships via graph-based methods, and employs adaptive generation to produce scalable data. It extends beyond single-document tasks to support multi-document reasoning, such as cross-document comparison and aggregation. Our models, finetuned on 150K instruction-response pairs synthesized using WildLong, surpasses existing open-source long-context-optimized models across benchmarks while maintaining strong performance on short-context tasks without incorporating supplementary short-context data. By generating a more diverse and realistic long-context instruction dataset, WildLong enhances LLMs' ability to generalize to complex, real-world reasoning over long contexts, establishing a new paradigm for long-context data synthesis. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过扩展上下文窗口能够处理需要广泛信息整合的任务，但在长期上下文指令调优方面受限于高质量和多样化的数据集稀缺性。现有的数据合成方法集中在诸如事实检索和摘要等具体目标上，限制了其对复杂、现实世界任务的普适性。WildLong 从真实用户体验查询中提取元信息，通过图基方法建模共现关系，并采用自适应生成方法生成可扩展的数据集，从而超越单一文档任务的支持范围，扩展到多文档推理任务，如跨文档比较和聚合。通过在使用WildLong生成的150万指令-回复对上进行微调，我们的模型在基准测试中表现出色，并在不需要额外加入短上下文数据的情况下保持了在短上下文任务中的强大性能。通过生成更多样且更具有真实性的长上下文指令数据集，WildLong 提高了LLMs在处理长期、复杂场景下现实世界推理任务的能力，建立了长上下文数据合成的新范式。 

---
# Automatic Input Rewriting Improves Translation with Large Language Models 

**Title (ZH)**: 自动输入重写可提高大语言模型的翻译效果 

**Authors**: Dayeon Ki, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2502.16682)  

**Abstract**: Can we improve machine translation (MT) with LLMs by rewriting their inputs automatically? Users commonly rely on the intuition that well-written text is easier to translate when using off-the-shelf MT systems. LLMs can rewrite text in many ways but in the context of MT, these capabilities have been primarily exploited to rewrite outputs via post-editing. We present an empirical study of 21 input rewriting methods with 3 open-weight LLMs for translating from English into 6 target languages. We show that text simplification is the most effective MT-agnostic rewrite strategy and that it can be improved further when using quality estimation to assess translatability. Human evaluation further confirms that simplified rewrites and their MT outputs both largely preserve the original meaning of the source and MT. These results suggest LLM-assisted input rewriting as a promising direction for improving translations. 

**Abstract (ZH)**: 我们能否通过自动重写输入来利用大语言模型（LLM）来提升机器翻译（MT）的质量？用户通常认为，当使用现成的MT系统时，清晰写作风格的文本更容易翻译。虽然LLM能够以多种方式重写文本，但在MT的背景下，这些能力主要通过后编辑方式重写输出得到了探索。我们进行了一项实证研究，使用3个开源权重的LLM对从英语翻译成6种目标语言的21种输入重写方法进行了研究。结果显示，文本简化是最有效的跨MT系统的重写策略，并且在使用质量估计来评估可译性时，这一策略可以得到进一步提升。人类评估进一步证实，简化后的重写文本及其MT输出都能很大程度上保留源文本和MT输出的原意。这些结果表明，通过LLM辅助的输入重写可能是提升翻译质量的一个有前景的方向。 

---
# MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models 

**Title (ZH)**: MimeQA：走向具备社会智能的非言语基础模型 

**Authors**: Hengzhi Li, Megan Tjandrasuwita, Yi R. Fung, Armando Solar-Lezama, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16671)  

**Abstract**: Socially intelligent AI that can understand and interact seamlessly with humans in daily lives is increasingly important as AI becomes more closely integrated with peoples' daily activities. However, current works in artificial social reasoning all rely on language-only, or language-dominant approaches to benchmark and training models, resulting in systems that are improving in verbal communication but struggle with nonverbal social understanding. To address this limitation, we tap into a novel source of data rich in nonverbal and social interactions -- mime videos. Mimes refer to the art of expression through gesture and movement without spoken words, which presents unique challenges and opportunities in interpreting non-verbal social communication. We contribute a new dataset called MimeQA, obtained by sourcing 221 videos from YouTube, through rigorous annotation and verification, resulting in a benchmark with 101 videos and 806 question-answer pairs. Using MimeQA, we evaluate state-of-the-art video large language models (vLLMs) and find that their overall accuracy ranges from 15-30%. Our analysis reveals that vLLMs often fail to ground imagined objects and over-rely on the text prompt while ignoring subtle nonverbal interactions. Our data resources are released at this https URL to inspire future work in foundation models that embody true social intelligence capable of interpreting non-verbal human interactions. 

**Abstract (ZH)**: 随着人工智能与人们日常活动的日益融合，能够理解并无缝互动于日常生活的社交智能AI变得越来越重要。然而，当前的人工社会推理研究均依赖于基于语言的方法来评估和训练模型，导致系统在口头交流方面有所提升，但在非语言社交理解方面却表现不佳。为解决这一局限性，我们探索了一种富含非语言和社会互动的新数据来源——哑剧视频。哑剧是指通过手势和动作表达而不使用言语的艺术形式，为非语言社交沟通的解释带来了独特的挑战与机遇。我们贡献了一个新的数据集——MimeQA，通过严格的注释和验证，从YouTube中收集了221个视频，最终形成了包含101个视频和806个问答对的基准数据集。利用MimeQA，我们评估了最新的视频大规模语言模型（vLLMs），发现其总体准确率范围为15%-30%。我们的分析表明，vLLMs经常无法将想象中的物体与实际环境对接，并过度依赖文本提示，而忽视了微妙的非语言互动。我们已将这些数据资源发布在如下链接：https://...，以激发未来能够真正理解非语言人类互动的基座模型的研究工作。 

---
# CODESYNC: Synchronizing Large Language Models with Dynamic Code Evolution at Scale 

**Title (ZH)**: CODESYNC：大规模语言模型的动态代码进化同步 

**Authors**: Chenlong Wang, Zhaoyang Chu, Zhengxiang Cheng, Xuyi Yang, Kaiyue Qiu, Yao Wan, Zhou Zhao, Xuanhua Shi, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16645)  

**Abstract**: Large Language Models (LLMs) have exhibited exceptional performance in software engineering yet face challenges in adapting to continually evolving code knowledge, particularly regarding the frequent updates of third-party library APIs. This limitation, stemming from static pre-training datasets, often results in non-executable code or implementations with suboptimal safety and efficiency. To this end, this paper introduces CODESYNC, a data engine for identifying outdated code patterns and collecting real-time code knowledge updates from Python third-party libraries. Building upon CODESYNC, we develop CODESYNCBENCH, a comprehensive benchmark for assessing LLMs' ability to stay synchronized with code evolution, which covers real-world updates for 220 APIs from six Python libraries. Our benchmark offers 3,300 test cases across three evaluation tasks and an update-aware instruction tuning dataset consisting of 2,200 training samples. Extensive experiments on 14 state-of-the-art LLMs reveal that they struggle with dynamic code evolution, even with the support of advanced knowledge updating methods (e.g., DPO, ORPO, and SimPO). We believe that our benchmark can offer a strong foundation for the development of more effective methods for real-time code knowledge updating in the future. The experimental code and dataset are publicly available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程领域表现出色，但在适应不断演变的代码知识方面面临挑战，尤其是在第三方库API的频繁更新方面。这一限制源自静态预训练数据集，通常导致非执行代码或安全性与效率较低的实现。为解决这一问题，本文介绍了一种名为CODESYNC的数据引擎，用于识别过时的代码模式并从Python第三方库中收集实时的代码知识更新。基于CODESYNC，我们开发了CODESYNCBENCH，这是评估LLM随代码演化保持同步能力的综合基准，涵盖了来自六种Python库的220个API的实际更新。该基准提供了3,300个测试用例，分布在三项评估任务中，并包含一个包含2,200个训练样本的更新感知指令微调数据集。对14种最先进的LLM的广泛实验表明，即使有高级知识更新方法（如DPO、ORPO和SimPO）的支持，它们也难以应对动态代码演化。我们相信，我们的基准可以为未来实时代码知识更新方法的发展提供坚实的基础。实验代码和数据集已公开发布在：![此处填写公开链接地址，例如：https://example.com](https://example.com)。 

---
# Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries 

**Title (ZH)**: 视觉-RAG：增强生成文本到图像检索的视觉知识密集型查询基准测试 

**Authors**: Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16636)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a popular approach for enhancing Large Language Models (LLMs) by addressing their limitations in verifying facts and answering knowledge-intensive questions. As the research in LLM extends their capability to handle input modality other than text, e.g. image, several multimodal RAG benchmarks are proposed. Nonetheless, they mainly use textual knowledge bases as the primary source of evidences for augmentation. There still lack benchmarks designed to evaluate images as augmentation in RAG systems and how they leverage visual knowledge. We propose Visual-RAG, a novel Question Answering benchmark that emphasizes visual knowledge intensive questions. Unlike prior works relying on text-based evidence, Visual-RAG necessitates text-to-image retrieval and integration of relevant clue images to extract visual knowledge as evidence. With Visual-RAG, we evaluate 5 open-sourced and 3 proprietary Multimodal LLMs (MLLMs), revealing that images can serve as good evidence in RAG; however, even the SoTA models struggle with effectively extracting and utilizing visual knowledge 

**Abstract (ZH)**: 检索增强生成（RAG）是一种通过解决大型语言模型（LLMs）在事实验证和回答知识密集型问题方面的局限性来增强LLMs的流行方法。随着LLMs研究扩展其处理输入模态的能力，例如图像，已经提出了几种多模态RAG基准。然而，它们主要依赖于文本知识库作为增强的主要证据来源。尚未设计出专门评估图像作为RAG系统增强手段以及如何利用视觉知识的基准。我们提出了Visual-RAG，这是一种新的问答基准，侧重于视觉知识密集型的问题。与以往依赖于文本证据的工作不同，Visual-RAG 要求进行文本到图像的检索和将相关线索图像集成起来以提取视觉知识作为证据。通过Visual-RAG，我们评估了5个开源和3个专有的多模态大型语言模型（MLLMs），结果显示图像可以作为RAG的良好证据；然而，即使是当前最好的模型，在有效提取和利用视觉知识方面也存在困难。 

---
# CodeCriticBench: A Holistic Code Critique Benchmark for Large Language Models 

**Title (ZH)**: CodeCriticBench：大规模语言模型综合代码评论基准 

**Authors**: Alexander Zhang, Marcus Dong, Jiaheng Liu, Wei Zhang, Yejie Wang, Jian Yang, Ge Zhang, Tianyu Liu, Zhongyuan Peng, Yingshui Tan, Yuanxing Zhang, Zhexu Wang, Weixun Wang, Yancheng He, Ken Deng, Wangchunshu Zhou, Wenhao Huang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16614)  

**Abstract**: The critique capacity of Large Language Models (LLMs) is essential for reasoning abilities, which can provide necessary suggestions (e.g., detailed analysis and constructive feedback). Therefore, how to evaluate the critique capacity of LLMs has drawn great attention and several critique benchmarks have been proposed. However, existing critique benchmarks usually have the following limitations: (1). Focusing on diverse reasoning tasks in general domains and insufficient evaluation on code tasks (e.g., only covering code generation task), where the difficulty of queries is relatively easy (e.g., the code queries of CriticBench are from Humaneval and MBPP). (2). Lacking comprehensive evaluation from different dimensions. To address these limitations, we introduce a holistic code critique benchmark for LLMs called CodeCriticBench. Specifically, our CodeCriticBench includes two mainstream code tasks (i.e., code generation and code QA) with different difficulties. Besides, the evaluation protocols include basic critique evaluation and advanced critique evaluation for different characteristics, where fine-grained evaluation checklists are well-designed for advanced settings. Finally, we conduct extensive experimental results of existing LLMs, which show the effectiveness of CodeCriticBench. 

**Abstract (ZH)**: 大型语言模型（LLMs）的批判能力对于推理能力至关重要，它可以提供必要的建议（例如详细的分析和建设性的反馈）。因此，如何评估LLMs的批判能力已经引起了广泛关注，并提出了多种批判基准。然而，现有批判基准通常存在以下局限性：（1）主要关注一般领域的多样化推理任务，在代码任务方面评价不足（例如，仅涵盖了代码生成任务），而代码查询的难度相对较低（例如，CodeCritic中的代码查询源自Humaneval和MBPP）。（2）缺乏从不同维度进行全面评估。为应对这些局限性，我们引入了一个面向LLMs的整体代码批判基准，称为CodeCriticBench。具体而言，我们的CodeCriticBench包括两个主流的代码任务（即代码生成和代码质量评估），且难度不同。此外，评价协议包括基础批判评价和高级批判评价，分别针对不同特征。在高级设置中，设计了详细的评价清单。最后，我们对现有多个LLMs进行了广泛的实验，结果表明CodeCriticBench的有效性。 

---
# MemeIntel: Explainable Detection of Propagandistic and Hateful Memes 

**Title (ZH)**: MemeIntel：具有解释性的 propaganda 和仇恨言论 meme 的检测 

**Authors**: Mohamed Bayan Kmainasi, Abul Hasnat, Md Arid Hasan, Ali Ezzat Shahroor, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2502.16612)  

**Abstract**: The proliferation of multimodal content on social media presents significant challenges in understanding and moderating complex, context-dependent issues such as misinformation, hate speech, and propaganda. While efforts have been made to develop resources and propose new methods for automatic detection, limited attention has been given to label detection and the generation of explanation-based rationales for predicted labels. To address this challenge, we introduce MemeIntel, an explanation-enhanced dataset for propaganda memes in Arabic and hateful memes in English, making it the first large-scale resource for these tasks. To solve these tasks, we propose a multi-stage optimization approach and train Vision-Language Models (VLMs). Our results demonstrate that this approach significantly improves performance over the base model for both \textbf{label detection} and explanation generation, outperforming the current state-of-the-art with an absolute improvement of ~3% on ArMeme and ~7% on Hateful Memes. For reproducibility and future research, we aim to make the MemeIntel dataset and experimental resources publicly available. 

**Abstract (ZH)**: 社交媒体上多模态内容的激增给理解和管理复杂、情境依赖的问题（如假信息、仇恨言论和宣传）带来了重大挑战。尽管已经采取了措施开发资源并提出新的自动检测方法，但对标签检测及生成基于解释的理由的关注仍然不足。为解决这一挑战，我们引入了MemeIntel，这是一个包含阿拉伯语宣传 meme 和英语仇恨言论 meme 的解释增强数据集，使它成为这两个任务上的首个大规模资源。为了解决这些任务，我们提出了一种多阶段优化方法，并训练了视觉-语言模型（VLMs）。我们的实验结果表明，这种方法在标签检测和理由生成方面均显著优于基线模型，对于ArMeme，在绝对改进方面优于当前最先进的方法约3%，而对于仇恨言论 meme，则优于约7%。为了实现可重复性和未来研究的需要，我们旨在将MemeIntel数据集和实验资源公开提供。 

---
# Revealing the Pragmatic Dilemma for Moral Reasoning Acquisition in Language Models 

**Title (ZH)**: 揭示语言模型在道德推理获取中的实用主义困境 

**Authors**: Guangliang Liu, Lei Jiang, Xitong Zhang, Kristen Marie Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2502.16600)  

**Abstract**: Ensuring that Large Language Models (LLMs) return just responses which adhere to societal values is crucial for their broader application. Prior research has shown that LLMs often fail to perform satisfactorily on tasks requiring moral cognizance, such as ethics-based judgments. While current approaches have focused on fine-tuning LLMs with curated datasets to improve their capabilities on such tasks, choosing the optimal learning paradigm to enhance the ethical responses of LLMs remains an open research debate. In this work, we aim to address this fundamental question: can current learning paradigms enable LLMs to acquire sufficient moral reasoning capabilities? Drawing from distributional semantics theory and the pragmatic nature of moral discourse, our analysis indicates that performance improvements follow a mechanism similar to that of semantic-level tasks, and therefore remain affected by the pragmatic nature of morals latent in discourse, a phenomenon we name the pragmatic dilemma. We conclude that this pragmatic dilemma imposes significant limitations on the generalization ability of current learning paradigms, making it the primary bottleneck for moral reasoning acquisition in LLMs. 

**Abstract (ZH)**: 确保大型语言模型（LLMs）返回符合社会价值观的恰当响应对于它们的广泛应用至关重要。先前的研究表明，LLMs 在涉及道德认知的任务上，例如基于伦理判断的任务，往往无法表现出令人满意的表现。尽管当前的方法主要集中在使用定制的数据集对LLMs进行微调以提高它们在这些任务上的能力，但选择何种学习范式来增强LLMs的伦理响应仍然是开放的研究辩论。在本文中，我们试图解决这一基本问题：当前的学习范式是否能够让LLMs获得足够的道德推理能力？从分布语义理论和道德语用性的角度来看，我们的分析表明，性能的提升遵循与语义层面任务类似的工作机制，并因此受到话语中隐含的语用性质的影响，我们将其称为“语用困境”。我们得出结论，这种语用困境极大地限制了当前学习范式的一般化能力，成为LLMs获得道德推理能力的主要瓶颈。 

---
# Beyond Words: How Large Language Models Perform in Quantitative Management Problem-Solving 

**Title (ZH)**: 超越文字：大型语言模型在定量管理问题解决中的表现 

**Authors**: Jonathan Kuzmanko  

**Link**: [PDF](https://arxiv.org/pdf/2502.16556)  

**Abstract**: This study examines how Large Language Models (LLMs) perform when tackling quantitative management decision problems in a zero-shot setting. Drawing on 900 responses generated by five leading models across 20 diverse managerial scenarios, our analysis explores whether these base models can deliver accurate numerical decisions under varying presentation formats, scenario complexities, and repeated attempts. Contrary to prior findings, we observed no significant effects of text presentation format (direct, narrative, or tabular) or text length on accuracy. However, scenario complexity -- particularly in terms of constraints and irrelevant parameters -- strongly influenced performance, often degrading accuracy. Surprisingly, the models handled tasks requiring multiple solution steps more effectively than expected. Notably, only 28.8\% of responses were exactly correct, highlighting limitations in precision. We further found no significant ``learning effect'' across iterations: performance remained stable across repeated queries. Nonetheless, significant variations emerged among the five tested LLMs, with some showing superior binary accuracy. Overall, these findings underscore both the promise and the pitfalls of harnessing LLMs for complex quantitative decision-making, informing managers and researchers about optimal deployment strategies. 

**Abstract (ZH)**: 本研究探讨了大语言模型（LLMs）在零样本情境下解决定量管理决策问题的表现。通过分析五种领先模型在20种不同管理情境中生成的900个响应，我们的研究考察了这些基础模型在不同呈现格式、情景复杂性和重复尝试情况下是否能够提供准确的数字决策。与先前的研究结果不同，我们没有发现文本呈现格式（直接、叙述性或表格）或文本长度对准确度有显著影响。然而，情景复杂性，特别是约束条件和无关参数的复杂性，对模型表现产生了显著影响，往往降低了准确度。令人惊讶的是，模型在处理需要多步解决方案的任务方面表现得比预期更好。值得注意的是，只有28.8%的响应是完全正确的，这凸显了精度上的局限性。进一步的分析还发现，这些模型在不同迭代中没有显著的学习效应：重复查询的性能保持稳定。尽管如此，在五种测试的LLM中出现了一些显著的差异，某些模型在二元分类精度方面表现更优。总体而言，这些发现强调了利用LLMs进行复杂定量决策的潜力和风险，为管理者和研究人员提供了有关最优部署策略的信息。 

---
# Reasoning About Persuasion: Can LLMs Enable Explainable Propaganda Detection? 

**Title (ZH)**: 关于说服力的推理：LLM能否实现可解释的宣传检测？ 

**Authors**: Maram Hasanain, Md Arid Hasan, Mohamed Bayan Kmainasi, Elisa Sartori, Ali Ezzat Shahroor, Giovanni Da San Martino, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2502.16550)  

**Abstract**: There has been significant research on propagandistic content detection across different modalities and languages. However, most studies have primarily focused on detection, with little attention given to explanations justifying the predicted label. This is largely due to the lack of resources that provide explanations alongside annotated labels. To address this issue, we propose a multilingual (i.e., Arabic and English) explanation-enhanced dataset, the first of its kind. Additionally, we introduce an explanation-enhanced LLM for both label detection and rationale-based explanation generation. Our findings indicate that the model performs comparably while also generating explanations. We will make the dataset and experimental resources publicly available for the research community. 

**Abstract (ZH)**: 在不同的模态和语言中，关于宣传内容检测的研究已经取得了显著进展。然而，大多数研究主要集中在检测方面，而在预测标签的解释上给予的关注较少。这主要是因为缺乏可以提供解释并附带标注标签的资源。为了解决这一问题，我们提出了一种多语言（即阿拉伯语和英语）的增强解释数据集，这是此类数据集中的首个实例。此外，我们还介绍了一种增强解释的大型语言模型，适用于标签检测和基于理据的解释生成。我们的研究结果表明，该模型在检测标签的同时也能生成解释。我们将公开发布该数据集和实验资源，供研究社区使用。 

---
# Advanced Chain-of-Thought Reasoning for Parameter Extraction from Documents Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行文档中参数抽取的高级链式推理方法 

**Authors**: Hong Cai Chen, Yi Pin Xu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16540)  

**Abstract**: Extracting parameters from technical documentation is crucial for ensuring design precision and simulation reliability in electronic design. However, current methods struggle to handle high-dimensional design data and meet the demands of real-time processing. In electronic design automation (EDA), engineers often manually search through extensive documents to retrieve component parameters required for constructing PySpice models, a process that is both labor-intensive and time-consuming. To address this challenge, we propose an innovative framework that leverages large language models (LLMs) to automate the extraction of parameters and the generation of PySpice models directly from datasheets. Our framework introduces three Chain-of-Thought (CoT) based techniques: (1) Targeted Document Retrieval (TDR), which enables the rapid identification of relevant technical sections; (2) Iterative Retrieval Optimization (IRO), which refines the parameter search through iterative improvements; and (3) Preference Optimization (PO), which dynamically prioritizes key document sections based on relevance. Experimental results show that applying all three methods together improves retrieval precision by 47.69% and reduces processing latency by 37.84%. Furthermore, effect size analysis using Cohen's d reveals that PO significantly reduces latency, while IRO contributes most to precision enhancement. These findings underscore the potential of our framework to streamline EDA processes, enhance design accuracy, and shorten development timelines. Additionally, our algorithm has model-agnostic generalization, meaning it can improve parameter search performance across different LLMs. 

**Abstract (ZH)**: 从技术文档中提取参数对于确保电子设计中的设计精确度和仿真可靠性至关重要。然而，当前的方法难以处理高维设计数据，并满足实时处理的需求。在电子设计自动化（EDA）中，工程师通常需要手动搜索大量文档以提取用于构建PySpice模型的组件参数，这是一个既耗时又费力的过程。为应对这一挑战，我们提出了一种创新框架，利用大规模语言模型（LLMs）从数据表中自动提取参数并直接生成PySpice模型。该框架引入了三种基于Chain-of-Thought（CoT）的技术：（1）针对性文档检索（TDR），能够快速识别相关的技术部分；（2）迭代检索优化（IRO），通过迭代改进来细化参数搜索；（3）偏好优化（PO），根据相关性动态优先处理关键文档部分。实验结果显示，联合使用三种方法可以将检索精度提高47.69%，并将处理延迟减少37.84%。此外，使用Cohen's d进行效应大小分析显示，PO显著减少了延迟，而IRO对精度提升贡献最大。这些发现表明，我们的框架有可能简化EDA流程，提高设计准确性，并缩短开发周期。此外，我们的算法具有模型通用性，这意味着它可以改善不同LLMs的参数搜索性能。 

---
# Multilingual != Multicultural: Evaluating Gaps Between Multilingual Capabilities and Cultural Alignment in LLMs 

**Title (ZH)**: 多语言 ≠ 多文化：评估语言模型的多语言能力与文化对齐之间的差距

该标题的翻译保持了原文的意思，并符合学术规范。 

**Authors**: Jonathan Rystrøm, Hannah Rose Kirk, Scott Hale  

**Link**: [PDF](https://arxiv.org/pdf/2502.16534)  

**Abstract**: Large Language Models (LLMs) are becoming increasingly capable across global languages. However, the ability to communicate across languages does not necessarily translate to appropriate cultural representations. A key concern is US-centric bias, where LLMs reflect US rather than local cultural values. We propose a novel methodology that compares LLM-generated response distributions against population-level opinion data from the World Value Survey across four languages (Danish, Dutch, English, and Portuguese). Using a rigorous linear mixed-effects regression framework, we compare two families of models: Google's Gemma models (2B--27B parameters) and successive iterations of OpenAI's turbo-series. Across the families of models, we find no consistent relationships between language capabilities and cultural alignment. While the Gemma models have a positive correlation between language capability and cultural alignment across languages, the OpenAI models do not. Importantly, we find that self-consistency is a stronger predictor of multicultural alignment than multilingual capabilities. Our results demonstrate that achieving meaningful cultural alignment requires dedicated effort beyond improving general language capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种全球语言方面的能力正在不断增强。然而，跨语言沟通的能力并不一定意味着适当的跨文化表现。一个主要问题是美国中心偏见，即LLMs更多地反映了美国而非当地的文化价值。我们提出了一种新颖的方法论，将LLM生成的回答分布与世界价值观调查中的国家层面意见数据进行比较，涵盖四种语言（丹麦语、荷兰语、英语和葡萄牙语）。我们使用严谨的线性混合效应回归框架，比较了两大家族的语言模型：Google的Gemma模型（2B-27B参数）和OpenAI的turbo系列的迭代模型。在各种模型家族中，我们没有发现语言能力与文化对齐之间的一致关系。虽然Gemma模型在多种语言中表现出语言能力与文化对齐之间的正相关性，但OpenAI模型则没有这种相关性。重要的是，我们发现自我一致性比多语言能力更能预测跨文化对齐。我们的研究结果表明，实现有意义的文化对齐需要超出一般语言能力改进的专门努力。 

---
# Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation 

**Title (ZH)**: 基于检索增强和偏好优化的视觉程序生成微调方法 

**Authors**: Deokhyung Kang, Jeonghun Cho, Yejin Jeon, Sunbin Jang, Minsub Lee, Jawoon Cho, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.16529)  

**Abstract**: Visual programming languages (VPLs) allow users to create programs through graphical interfaces, which results in easier accessibility and their widespread usage in various domains. To further enhance this accessibility, recent research has focused on generating VPL code from user instructions using large language models (LLMs). Specifically, by employing prompting-based methods, these studies have shown promising results. Nevertheless, such approaches can be less effective for industrial VPLs such as Ladder Diagram (LD). LD is a pivotal language used in industrial automation processes and involves extensive domain-specific configurations, which are difficult to capture in a single prompt. In this work, we demonstrate that training-based methods outperform prompting-based methods for LD generation accuracy, even with smaller backbone models. Building on these findings, we propose a two-stage training strategy to further enhance VPL generation. First, we employ retrieval-augmented fine-tuning to leverage the repetitive use of subroutines commonly seen in industrial VPLs. Second, we apply direct preference optimization (DPO) to further guide the model toward accurate outputs, using systematically generated preference pairs through graph editing operations. Extensive experiments on real-world LD data demonstrate that our approach improves program-level accuracy by over 10% compared to supervised fine-tuning, which highlights its potential to advance industrial automation. 

**Abstract (ZH)**: 视觉编程语言（VPLs）允许用户通过图形界面创建程序，从而提高了其易用性，并且已在多个领域得到了广泛应用。为进一步增强这种易用性，最近的研究集中在使用大规模语言模型（LLMs）生成用户指令到VPL代码的方法上。具体来说，通过采用基于提示的方法，这些研究已经显示出令人鼓舞的结果。然而，这样的方法对于工业VPL，如梯形图（LD），可能效果较差。LD是工业自动化过程中使用的关键语言，涉及大量的领域特定配置，这使得在单一提示中难以完全捕捉到。在本文中，我们证明即使使用较小的基础模型，基于训练的方法在LD生成准确性上也优于基于提示的方法。基于这些发现，我们提出了一种两阶段训练策略，以进一步增强VPL生成。首先，我们采用检索增强微调方法，利用工业VPL中常见的子例行程序的重复使用。其次，我们应用直接偏好优化（DPO），通过图编辑操作生成系统性的偏好对，进一步引导模型输出准确的结果。在实际LD数据上的大量实验表明，与监督微调相比，我们的方法在程序级别上的准确性提高了超过10%，突显了其在推进工业自动化方面的潜力。 

---
# Pay Attention to Real World Perturbations! Natural Robustness Evaluation in Machine Reading Comprehension 

**Title (ZH)**: 请关注现实世界的干扰！机器阅读理解中的自然鲁棒性评估 

**Authors**: Yulong Wu, Viktor Schlegel, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2502.16523)  

**Abstract**: As neural language models achieve human-comparable performance on Machine Reading Comprehension (MRC) and see widespread adoption, ensuring their robustness in real-world scenarios has become increasingly important. Current robustness evaluation research, though, primarily develops synthetic perturbation methods, leaving unclear how well they reflect real life scenarios. Considering this, we present a framework to automatically examine MRC models on naturally occurring textual perturbations, by replacing paragraph in MRC benchmarks with their counterparts based on available Wikipedia edit history. Such perturbation type is natural as its design does not stem from an arteficial generative process, inherently distinct from the previously investigated synthetic approaches. In a large-scale study encompassing SQUAD datasets and various model architectures we observe that natural perturbations result in performance degradation in pre-trained encoder language models. More worryingly, these state-of-the-art Flan-T5 and Large Language Models (LLMs) inherit these errors. Further experiments demonstrate that our findings generalise to natural perturbations found in other more challenging MRC benchmarks. In an effort to mitigate these errors, we show that it is possible to improve the robustness to natural perturbations by training on naturally or synthetically perturbed examples, though a noticeable gap still remains compared to performance on unperturbed data. 

**Abstract (ZH)**: 随着神经语言模型在机器阅读理解（MRC）任务上实现接近人类的性能并在广泛应用中取得成功，确保它们在真实场景中的稳健性变得越来越重要。然而，现有的稳健性评估研究主要集中在合成扰动方法上，未能清晰地反映实际生活中的场景。鉴于此，我们提出了一种框架，通过利用可用的维基百科编辑历史替换MRC基准中的段落，自动检查MRC模型在自然发生的文本扰动下的表现。这种扰动类型是天然的，因为它不是源自一种人工生成的过程，本质上与之前研究的合成方法存在差异。在大规模研究中，我们使用SQUAD数据集和不同的模型架构观察到，天然扰动导致预训练编码器语言模型的性能下降。更令人担忧的是，这些最先进的Flan-T5和大型语言模型（LLMs）继承了这些错误。进一步的实验表明，我们的发现扩展到了其他更具挑战性的MRC基准中发现的天然扰动。为了减少这些错误，我们展示了通过训练人工或合成扰动的例子，确实可以提高模型对天然扰动的稳健性，尽管与未扰动数据相比，仍有明显的性能差距。 

---
# GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking 

**Title (ZH)**: GraphCheck：利用提取的知识图谱进行事实核查以突破长期文本障碍 

**Authors**: Yingjian Chen, Haoran Liu, Yinhong Liu, Rui Yang, Han Yuan, Yanran Fu, Pengyuan Zhou, Qingyu Chen, James Caverlee, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.16514)  

**Abstract**: Large language models (LLMs) are widely used, but they often generate subtle factual errors, especially in long-form text. These errors are fatal in some specialized domains such as medicine. Existing fact-checking with grounding documents methods face two main challenges: (1) they struggle to understand complex multihop relations in long documents, often overlooking subtle factual errors; (2) most specialized methods rely on pairwise comparisons, requiring multiple model calls, leading to high resource and computational costs. To address these challenges, we propose \textbf{\textit{GraphCheck}}, a fact-checking framework that uses extracted knowledge graphs to enhance text representation. Graph Neural Networks further process these graphs as a soft prompt, enabling LLMs to incorporate structured knowledge more effectively. Enhanced with graph-based reasoning, GraphCheck captures multihop reasoning chains which are often overlooked by existing methods, enabling precise and efficient fact-checking in a single inference call. Experimental results on seven benchmarks spanning both general and medical domains demonstrate a 6.1\% overall improvement over baseline models. Notably, GraphCheck outperforms existing specialized fact-checkers and achieves comparable performance with state-of-the-art LLMs, such as DeepSeek-V3 and OpenAI-o1, with significantly fewer parameters. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛应用中，但它们经常生成细微的事实性错误，特别是在长篇文本中。这些错误在一些专业领域，如医学领域，可能是致命的。现有的基于地面文档的事实核查方法面临两个主要挑战：（1）它们难以理解长文档中的复杂多跳关系，常常忽略细微的事实性错误；（2）大多数专业方法依赖于成对比较，需要多次调用模型，导致高资源和计算成本。为了解决这些挑战，我们提出了一种名为**\textbf{\textit{GraphCheck}}**的事实核查框架，该框架使用提取的知识图谱增强文本表示。通过图神经网络进一步处理这些图，将其作为软提示，使LLMs能够更有效地整合结构化知识。借助基于图的推理，GraphCheck能够捕捉到现有方法往往忽略的多跳推理链，从而在单次推理调用中实现精确而高效的事实核查。实验结果表明，GraphCheck在七个覆盖从通用到医学领域的基准测试中整体性能提升了6.1%。值得注意的是，GraphCheck在参数量显著减少的情况下，优于现有的专业事实核查器，并达到了与最先进的LLMs（如DeepSeek-V3和OpenAI-o1）相当的性能。 

---
# FanChuan: A Multilingual and Graph-Structured Benchmark For Parody Detection and Analysis 

**Title (ZH)**: 费 Chanduan：一种用于 parody 检测与分析的多语言和图结构基准数据集 

**Authors**: Yilun Zheng, Sha Li, Fangkun Wu, Yang Ziyi, Lin Hongchao, Zhichao Hu, Cai Xinjun, Ziming Wang, Jinxuan Chen, Sitao Luan, Jiahao Xu, Lihui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16503)  

**Abstract**: Parody is an emerging phenomenon on social media, where individuals imitate a role or position opposite to their own, often for humor, provocation, or controversy. Detecting and analyzing parody can be challenging and is often reliant on context, yet it plays a crucial role in understanding cultural values, promoting subcultures, and enhancing self-expression. However, the study of parody is hindered by limited available data and deficient diversity in current datasets. To bridge this gap, we built seven parody datasets from both English and Chinese corpora, with 14,755 annotated users and 21,210 annotated comments in total. To provide sufficient context information, we also collect replies and construct user-interaction graphs to provide richer contextual information, which is lacking in existing datasets. With these datasets, we test traditional methods and Large Language Models (LLMs) on three key tasks: (1) parody detection, (2) comment sentiment analysis with parody, and (3) user sentiment analysis with parody. Our extensive experiments reveal that parody-related tasks still remain challenging for all models, and contextual information plays a critical role. Interestingly, we find that, in certain scenarios, traditional sentence embedding methods combined with simple classifiers can outperform advanced LLMs, i.e. DeepSeek-R1 and GPT-o3, highlighting parody as a significant challenge for LLMs. 

**Abstract (ZH)**: 嘲讽是社交媒体上的一种新兴现象，人们会模仿与其自身角色或地位相反的角色，通常是为了娱乐、挑衅或制造争议。检测和分析嘲讽具有挑战性，通常依赖于上下文，但在理解和表征文化价值观、促进亚文化发展和增强自我表达方面发挥着重要作用。然而，嘲讽研究受到可用数据有限和现有数据集多样性不足的阻碍。为解决这一问题，我们从英语和中文语料库中构建了七个嘲讽数据集，总共包含14,755名标注用户和21,210条评论。为了提供足够的上下文信息，我们还收集了回复并构建了用户交互图，以提供更丰富的上下文信息，而这些信息在现有数据集中是缺乏的。借助这些数据集，我们测试了传统方法和大规模语言模型（LLMs）在三个关键任务中的性能：（1）嘲讽检测，（2）包含嘲讽的评论情感分析，（3）包含嘲讽的用户情感分析。我们的广泛实验表明，所有模型在处理与嘲讽相关任务时仍然面临挑战，上下文信息发挥着关键作用。有趣的是，我们发现，在某些场景中，传统的句子嵌入方法结合简单的分类器可以优于先进的LLMs，如DeepSeek-R1和GPT-o3，这突显了嘲讽对LLMs的巨大挑战。 

---
# Intrinsic Model Weaknesses: How Priming Attacks Unveil Vulnerabilities in Large Language Models 

**Title (ZH)**: 固有模型缺陷：先启动攻击如何揭示大型语言模型的脆弱性 

**Authors**: Yuyi Huang, Runzhe Zhan, Derek F. Wong, Lidia S. Chao, Ailin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.16491)  

**Abstract**: Large language models (LLMs) have significantly influenced various industries but suffer from a critical flaw, the potential sensitivity of generating harmful content, which poses severe societal risks. We developed and tested novel attack strategies on popular LLMs to expose their vulnerabilities in generating inappropriate content. These strategies, inspired by psychological phenomena such as the "Priming Effect", "Safe Attention Shift", and "Cognitive Dissonance", effectively attack the models' guarding mechanisms. Our experiments achieved an attack success rate (ASR) of 100% on various open-source models, including Meta's Llama-3.2, Google's Gemma-2, Mistral's Mistral-NeMo, Falcon's Falcon-mamba, Apple's DCLM, Microsoft's Phi3, and Qwen's Qwen2.5, among others. Similarly, for closed-source models such as OpenAI's GPT-4o, Google's Gemini-1.5, and Claude-3.5, we observed an ASR of at least 95% on the AdvBench dataset, which represents the current state-of-the-art. This study underscores the urgent need to reassess the use of generative models in critical applications to mitigate potential adverse societal impacts. 

**Abstract (ZH)**: 大型语言模型（LLMs）对各个行业产生了重要影响，但它们面临一个关键缺陷：生成有害内容的风险，这可能带来严重的社会风险。我们开发并测试了针对流行LLMs的新攻击策略，以揭示它们在生成不当内容方面的脆弱性。这些策略受到诸如“启动效应”、“安全注意力转移”和“认知失调”等心理现象的启发，有效地攻击了模型的防御机制。我们的实验在包括Meta的Llama-3.2、Google的Gemma-2、Mistral的Mistral-NeMo、Falcon的Falcon-mamba、Apple的DCLM、Microsoft的Phi3以及Qwen的Qwen2.5等开源模型中实现了100%的攻击成功率（ASR）。同样，对于封闭源代码模型如OpenAI的GPT-4o、Google的Gemini-1.5和Claude-3.5，我们在AdvBench数据集上观察到至少95%的ASR，这代表了当前最先进的技术水平。本研究强调了重新评估在关键应用中使用生成模型的必要性，以减轻潜在的不良社会影响。 

---
# All That Glitters is Not Novel: Plagiarism in AI Generated Research 

**Title (ZH)**: 璀璨未必皆新颖：AI生成研究中的剽窃问题 

**Authors**: Tarun Gupta, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16487)  

**Abstract**: Automating scientific research is considered the final frontier of science. Recently, several papers claim autonomous research agents can generate novel research ideas. Amidst the prevailing optimism, we document a critical concern: a considerable fraction of such research documents are smartly plagiarized. Unlike past efforts where experts evaluate the novelty and feasibility of research ideas, we request $13$ experts to operate under a different situational logic: to identify similarities between LLM-generated research documents and existing work. Concerningly, the experts identify $24\%$ of the $50$ evaluated research documents to be either paraphrased (with one-to-one methodological mapping), or significantly borrowed from existing work. These reported instances are cross-verified by authors of the source papers. Problematically, these LLM-generated research documents do not acknowledge original sources, and bypass inbuilt plagiarism detectors. Lastly, through controlled experiments we show that automated plagiarism detectors are inadequate at catching deliberately plagiarized ideas from an LLM. We recommend a careful assessment of LLM-generated research, and discuss the implications of our findings on research and academic publishing. 

**Abstract (ZH)**: 自动化科学研究被认为是科学的最后前沿。最近，有几篇论文声称自主研究代理能够生成新颖的研究想法。在普遍的乐观情绪中，我们记录了一个关键问题：许多这类研究文档被巧妙地复制了。与以往由专家评估研究想法的新颖性和可行性不同，我们要求13位专家在不同的情境逻辑下操作：识别LLM生成的研究文档与现有工作之间的相似之处。令人担忧的是，专家们鉴定出在50份评估的研究文档中有24%是抄袭或显著借鉴了现有工作的成果。这些报告的实例已由源论文的作者进行交叉验证。问题在于，这些由LLM生成的研究文档没有承认原始来源，并规避了内置的抄袭检测器。最后，通过受控实验我们表明，自动化抄袭检测器在捕捉LLM故意抄袭的想法方面是不足的。我们建议对LLM生成的研究进行仔细评估，并讨论我们的发现对研究和学术出版的影响。 

---
# A Fine-Tuning Approach for T5 Using Knowledge Graphs to Address Complex Tasks 

**Title (ZH)**: 使用知识图谱调整T5模型以应对复杂任务的一种方法 

**Authors**: Xiaoxuan Liao, Binrong Zhu, Jacky He, Guiran Liu, Hongye Zheng, Jia Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.16484)  

**Abstract**: With the development of deep learning technology, large language models have achieved remarkable results in many natural language processing tasks. However, these models still have certain limitations in handling complex reasoning tasks and understanding rich background knowledge. To solve this problem, this study proposed a T5 model fine-tuning method based on knowledge graphs, which enhances the model's reasoning ability and context understanding ability by introducing external knowledge graphs. We used the SQuAD1.1 dataset for experiments. The experimental results show that the T5 model based on knowledge graphs is significantly better than other baseline models in reasoning accuracy, context understanding, and the ability to handle complex problems. At the same time, we also explored the impact of knowledge graphs of different scales on model performance and found that as the scale of the knowledge graph increases, the performance of the model gradually improves. Especially when dealing with complex problems, the introduction of knowledge graphs greatly improves the reasoning ability of the T5 model. Ablation experiments further verify the importance of entity and relationship embedding in the model and prove that a complete knowledge graph is crucial to improving the various capabilities of the T5 model. In summary, this study provides an effective method to enhance the reasoning and understanding capabilities of large language models and provides new directions for future research. 

**Abstract (ZH)**: 随着深度学习技术的发展，大规模语言模型在许多自然语言处理任务中取得了显著成果。然而，这些模型在处理复杂推理任务和理解丰富背景知识方面仍存在一定的局限性。为解决这一问题，本研究提出了一种基于知识图谱的T5模型微调方法，通过引入外部知识图谱来增强模型的推理能力和上下文理解能力。我们使用SQuAD1.1数据集进行了实验。实验结果显示，基于知识图谱的T5模型在推理准确度、上下文理解能力和处理复杂问题的能力上显著优于其他基线模型。同时，我们还探讨了不同规模的知识图谱对模型性能的影响，并发现随着知识图谱规模的增大，模型的性能逐渐提高。尤其是处理复杂问题时，引入知识图谱极大地提高了T5模型的推理能力。消融实验进一步证实了实体和关系嵌入在模型中的重要性，并证明了完整知识图谱对提高T5模型的各种能力至关重要。总的来说，本研究提供了一种有效的方法来增强大规模语言模型的推理和理解能力，并为未来的研究提供了新的方向。 

---
# Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge 

**Title (ZH)**: 通过大规模合成数据集和专家级LLM评判实现完全自动的材料发现 

**Authors**: Heegyu Kim, Taeyang Jeon, Seungtaek Choi, Jihoon Hong, Dongwon Jeon, Sungbum Cho, Ga-Yeon Baek, Kyung-Won Kwak, Dong-Hee Lee, Sun-Jin Choi, Jisu Bae, Chihoon Lee, Yunseo Kim, Jinsung Park, Hyunsouk Cho  

**Link**: [PDF](https://arxiv.org/pdf/2502.16457)  

**Abstract**: Materials synthesis is vital for innovations such as energy storage, catalysis, electronics, and biomedical devices. Yet, the process relies heavily on empirical, trial-and-error methods guided by expert intuition. Our work aims to support the materials science community by providing a practical, data-driven resource. We have curated a comprehensive dataset of 17K expert-verified synthesis recipes from open-access literature, which forms the basis of our newly developed benchmark, AlchemyBench. AlchemyBench offers an end-to-end framework that supports research in large language models applied to synthesis prediction. It encompasses key tasks, including raw materials and equipment prediction, synthesis procedure generation, and characterization outcome forecasting. We propose an LLM-as-a-Judge framework that leverages large language models for automated evaluation, demonstrating strong statistical agreement with expert assessments. Overall, our contributions offer a supportive foundation for exploring the capabilities of LLMs in predicting and guiding materials synthesis, ultimately paving the way for more efficient experimental design and accelerated innovation in materials science. 

**Abstract (ZH)**: 材料合成对于能源存储、催化、电子和生物医学设备等创新至关重要。然而，这一过程主要依赖于专家直觉指导下的经验性和试验性方法。我们致力于通过提供一个实用的数据驱动资源来支持材料科学界的研究。我们从开源文献中整理了一个包含17000个专家验证的合成配方的数据集，形成了我们新开发的基准——AlchemyBench。AlchemyBench提供了一个端到端的框架，支持大型语言模型在合成预测中的应用研究。该框架涵盖了关键任务，包括原材料和设备预测、合成过程生成以及表征结果预测。我们提出了一种LLM-as-a-Judge框架，利用大型语言模型进行自动化评估，显示出与专家评估结果的强烈统计一致性。总体而言，我们的贡献为探索LLM在预测和指导材料合成方面的潜力提供了一个支持性基础，最终为材料科学中的更高效实验设计和加速创新铺平了道路。 

---
# Contrastive Learning of English Language and Crystal Graphs for Multimodal Representation of Materials Knowledge 

**Title (ZH)**: 英语语言对比学习与晶体图谱在材料知识多模态表示中的应用 

**Authors**: Yang Jeong Park, Mayank Kumaran, Chia-Wei Hsu, Elsa Olivetti, Ju Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.16451)  

**Abstract**: Artificial intelligence (AI) is increasingly used for the inverse design of materials, such as crystals and molecules. Existing AI research on molecules has integrated chemical structures of molecules with textual knowledge to adapt to complex instructions. However, this approach has been unattainable for crystals due to data scarcity from the biased distribution of investigated crystals and the lack of semantic supervision in peer-reviewed literature. In this work, we introduce a contrastive language-crystals model (CLaC) pre-trained on a newly synthesized dataset of 126k crystal structure-text pairs. To demonstrate the advantage of using synthetic data to overcome data scarcity, we constructed a comparable dataset extracted from academic papers. We evaluate CLaC's generalization ability through various zero-shot cross-modal tasks and downstream applications. In experiments, CLaC achieves state-of-the-art zero-shot generalization performance in understanding crystal structures, surpassing latest large language models. 

**Abstract (ZH)**: 人工智能（AI）在材料的逆向设计中得到了越来越多的应用，包括晶体和分子。现有的关于分子的AI研究将分子的化学结构与文本知识相结合，以适应复杂的指令需求。然而，这种方法由于实验研究中晶体数据分布偏斜且同行评审文献缺乏语义监督，对于晶体的应用一直难以实现。在本研究中，我们引入了一个对比语言-晶体模型（CLaC），该模型是在一个新合成的12.6万对晶体结构-文本配对数据集上预训练的。为了证明使用合成数据来克服数据稀缺性的好处，我们构建了一个与之相当的数据集，该数据集提取自学术论文。我们通过多种零样本跨模态任务和下游应用评估CLaC的泛化能力。在实验中，CLaC在理解和推断晶体结构方面达到了最先进的零样本泛化性能，超越了最新的大规模语言模型。 

---
# Make Literature-Based Discovery Great Again through Reproducible Pipelines 

**Title (ZH)**: 通过可重复的流水线重新振兴基于文献的发现 

**Authors**: Bojan Cestnik, Andrej Kastrin, Boshko Koloski, Nada Lavrač  

**Link**: [PDF](https://arxiv.org/pdf/2502.16450)  

**Abstract**: By connecting disparate sources of scientific literature, literature\-/based discovery (LBD) methods help to uncover new knowledge and generate new research hypotheses that cannot be found from domain-specific documents alone. Our work focuses on bisociative LBD methods that combine bisociative reasoning with LBD techniques. The paper presents LBD through the lens of reproducible science to ensure the reproducibility of LBD experiments, overcome the inconsistent use of benchmark datasets and methods, trigger collaboration, and advance the LBD field toward more robust and impactful scientific discoveries. The main novelty of this study is a collection of Jupyter Notebooks that illustrate the steps of the bisociative LBD process, including data acquisition, text preprocessing, hypothesis formulation, and evaluation. The contributed notebooks implement a selection of traditional LBD approaches, as well as our own ensemble-based, outlier-based, and link prediction-based approaches. The reader can benefit from hands-on experience with LBD through open access to benchmark datasets, code reuse, and a ready-to-run Docker recipe that ensures reproducibility of the selected LBD methods. 

**Abstract (ZH)**: 通过连接不同的科学文献来源，文献基础发现（LBD）方法有助于揭示新的知识，并生成仅依赖领域特定文档无法找到的新研究假设。本研究工作集中在结合双向推理（bisociative reasoning）和LBD技术的双向推理LBD方法。本文通过可再现科学的角度来阐述LBD，以确保LBD实验的可再现性，克服基准数据集和方法使用不一致的问题，促进合作，并推动LBD领域向更加稳健和有影响力的科学发现迈进。本研究的主要创新在于提供了一系列Jupyter Notebooks，这些Notebooks示例了双向推理LBD过程的各个步骤，包括数据获取、文本预处理、假设提出和评估。贡献的Notebooks实现了多种传统的LBD方法，以及我们自己的基于集成、基于异常值和基于链接预测的方法。读者可以通过访问开放的基准数据集、代码重用以及提供的一键运行Docker脚本，获得实际操作LBD的经验，从而确保所选LBD方法的可再现性。 

---
# Sequence-level Large Language Model Training with Contrastive Preference Optimization 

**Title (ZH)**: 带有对比偏好优化的序列级大型语言模型训练 

**Authors**: Zhili Feng, Dhananjay Ram, Cole Hawkins, Aditya Rawal, Jinman Zhao, Sheng Zha  

**Link**: [PDF](https://arxiv.org/pdf/2502.16433)  

**Abstract**: The next token prediction loss is the dominant self-supervised training objective for large language models and has achieved promising results in a variety of downstream tasks. However, upon closer investigation of this objective, we find that it lacks an understanding of sequence-level signals, leading to a mismatch between training and inference processes. To bridge this gap, we introduce a contrastive preference optimization (CPO) procedure that can inject sequence-level information into the language model at any training stage without expensive human labeled data. Our experiments show that the proposed objective surpasses the next token prediction in terms of win rate in the instruction-following and text generation tasks. 

**Abstract (ZH)**: 下一个标记预测损失是大型语言模型自监督训练的主要目标，并在多种下游任务中取得了令人瞩人的成果。然而，通过更深入地研究这一目标，我们发现它缺乏对序列级信号的理解，导致训练过程与推断过程之间存在不匹配。为了弥合这一差距，我们提出了一种对比偏好优化（CPO）程序，该程序可以在任何训练阶段向语言模型注入序列级信息，而无需昂贵的人工标注数据。我们的实验表明，所提出的目标在指令跟随和文本生成任务中的胜率超越了下一个标记预测。 

---
# Automatic Detection of Research Values from Scientific Abstracts Across Computer Science Subfields 

**Title (ZH)**: 跨计算机科学子领域的科学研究价值自动检测方法 

**Authors**: Hang Jiang, Tal August, Luca Soldaini, Kyle Lo, Maria Antoniak  

**Link**: [PDF](https://arxiv.org/pdf/2502.16390)  

**Abstract**: The field of Computer science (CS) has rapidly evolved over the past few decades, providing computational tools and methodologies to various fields and forming new interdisciplinary communities. This growth in CS has significantly impacted institutional practices and relevant research communities. Therefore, it is crucial to explore what specific \textbf{research values}, known as \textbf{basic and fundamental beliefs that guide or motivate research attitudes or actions}, CS-related research communities promote. Prior research has manually analyzed research values from a small sample of machine learning papers \cite{facct}. No prior work has studied the automatic detection of research values in CS from large-scale scientific texts across different research subfields. This paper introduces a detailed annotation scheme featuring \textbf{ten research values} that guide CS-related research. Based on the scheme, we build value classifiers to scale up the analysis and present a systematic study over 226,600 paper abstracts from 32 CS-related subfields and 86 popular publishing venues over ten years. 

**Abstract (ZH)**: 计算机科学（CS）领域在过去几十年中迅速发展，提供了各种计算工具和方法论，并形成了新的跨学科社区。CS的发展对机构实践和相关研究社区产生了显著影响。因此，探索CS相关研究社区所推广的具体**研究价值观**变得尤为重要。这些价值观是一些基本和根本的信念，它们指导或激励研究态度或行为。先前的研究手动分析了一小部分机器学习论文中的研究价值观 [1]。在此之前，没有研究通过自动检测大规模科学文本来研究不同研究子领域的CS中的研究价值观。本文介绍了详细的注释方案，该方案涵盖了指导CS相关研究的**十个研究价值观**。基于该方案，我们构建了价值观分类器以扩大分析规模，并对过去十年中涵盖32个CS相关子领域和86个流行出版场所的共计226,600篇论文摘要进行了系统的分析研究。

参考文献：
[1] Facct. (Year). Title of the reference. Journal or Publication. 

---
# Instruction-Tuning LLMs for Event Extraction with Annotation Guidelines 

**Title (ZH)**: 根据学术规范，将标题“Instruction-Tuning LLMs for Event Extraction with Annotation Guidelines”翻译成中文如下：

“基于标注指南的指令调优大语言模型事件提取” 

**Authors**: Saurabh Srivastava, Sweta Pati, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.16377)  

**Abstract**: In this work, we study the effect of annotation guidelines -- textual descriptions of event types and arguments, when instruction-tuning large language models for event extraction. We conducted a series of experiments with both human-provided and machine-generated guidelines in both full- and low-data settings. Our results demonstrate the promise of annotation guidelines when there is a decent amount of training data and highlight its effectiveness in improving cross-schema generalization and low-frequency event-type performance. 

**Abstract (ZH)**: 在这项工作中，我们探讨了标注指南——事件类型和论元的文本描述，在指令调优大型语言模型进行事件提取时的作用。我们进行了系列实验，使用了人工提供的和机器生成的标注指南，并在数据量充足和数据量较少的不同场景下进行了测试。实验结果表明，在有足够的训练数据时，标注指南具有巨大潜力，并突显了其在提高跨模式泛化能力和低频事件类型性能方面的有效性。 

---
# A generative approach to LLM harmfulness detection with special red flag tokens 

**Title (ZH)**: 使用特殊红旗标记令牌的生成方法进行大语言模型有害内容检测 

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mohfakhami, Leo Schwinn, Gauthier Gidel  

**Link**: [PDF](https://arxiv.org/pdf/2502.16366)  

**Abstract**: Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks. 

**Abstract (ZH)**: 大多数针对大型语言模型（LLMs）的安全训练方法依赖于微调，当面对有害请求时，这些方法会显著改变模型的输出分布，将其从不安全的回答转变为拒绝回答。这些方法本质上会妥协模型的能力，并且可能使得自回归模型容易受到促使首个响应词为肯定性的攻击。为了避免这种情况，我们建议扩展模型的词汇表，加入一个我们称之为“红灯标记”（<rf>）的特殊标记，并提出对模型进行微调，使其在任何时候生成有害内容或者即将生成有害内容时生成该标记。这种方法为LLMs引入了一种新颖的安全训练方法，在对话的整个过程中，它能够增强模型对有害性的生成分类能力。这种方法具有几个优点：它使模型能够明确学习有害性的概念，同时略微影响生成的分布，从而保持模型的实用性。此外，这种方法会评估每个生成的答案，而不仅仅是输入提示，从而提供更强的针对采样攻击的防御。另外，它简化了模型鲁棒性的评估，并在与分类器结合时减少了相关失败。我们还进一步展示了该方法在更长上下文和监督微调攻击方面的增强鲁棒性。 

---
# Wrong Answers Can Also Be Useful: PlausibleQA -- A Large-Scale QA Dataset with Answer Plausibility Scores 

**Title (ZH)**: 错误的答案也可能很有用：PlausibleQA — 一个包含答案可信度评分的大规模问答数据集 

**Authors**: Jamshid Mozafari, Abdelrahman Abdallah, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2502.16358)  

**Abstract**: Large Language Models (LLMs) are revolutionizing information retrieval, with chatbots becoming an important source for answering user queries. As by their design, LLMs prioritize generating correct answers, the value of highly plausible yet incorrect answers (candidate answers) tends to be overlooked. However, such answers can still prove useful, for example, they can play a crucial role in tasks like Multiple-Choice Question Answering (MCQA) and QA Robustness Assessment (QARA). Existing QA datasets primarily focus on correct answers without explicit consideration of the plausibility of other candidate answers, limiting opportunity for more nuanced evaluations of models. To address this gap, we introduce PlausibleQA, a large-scale dataset comprising 10,000 questions and 100,000 candidate answers, each annotated with plausibility scores and justifications for their selection. Additionally, the dataset includes 900,000 justifications for pairwise comparisons between candidate answers, further refining plausibility assessments. We evaluate PlausibleQA through human assessments and empirical experiments, demonstrating its utility in MCQA and QARA analysis. Our findings show that plausibility-aware approaches are effective for MCQA distractor generation and QARA. We release PlausibleQA as a resource for advancing QA research and enhancing LLM performance in distinguishing plausible distractors from correct answers. 

**Abstract (ZH)**: 大型语言模型（Large Language Models, LLMs）正在重新定义信息检索，聊天机器人已经成为回答用户查询的重要来源。由于设计上的原因，LLMs 旨在生成正确的答案，因此，虽然可信但错误的答案（候选答案）的价值往往被忽略。然而，这些答案仍然具有实用性，例如，它们在多项选择题作答（Multiple-Choice Question Answering, MCQA）和问答鲁棒性评估（Question Answering Robustness Assessment, QARA）等任务中扮演着关键角色。现有的问答数据集主要关注正确答案，而没有明确考虑其他候选答案的可信度，这限制了对模型进行全面评估的机会。为了解决这一问题，我们引入了PlausibleQA数据集，包含10,000个问题和100,000个候选答案，每个答案都标注了可信度评分及其选择依据。此外，数据集还包含了900,000个候选答案的成对比较依据，进一步细化了可信度评估。我们通过人工评估和实证实验对PlausibleQA进行评估，展示了其在MCQA和QARA分析中的应用价值。我们的研究结果表明，可信度意识的方法在MCQA干扰项生成和QARA中是有效的。我们发布PlausibleQA作为推动问答研究和提升LLM性能的资源，旨在增强模型区分可信干扰项与正确答案的能力。 

---
# LegalBench.PT: A Benchmark for Portuguese Law 

**Title (ZH)**: LegalBench.PT：葡萄牙法律基准数据集 

**Authors**: Beatriz Canaverde, Telmo Pessoa Pires, Leonor Melo Ribeiro, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2502.16357)  

**Abstract**: The recent application of LLMs to the legal field has spurred the creation of benchmarks across various jurisdictions and languages. However, no benchmark has yet been specifically designed for the Portuguese legal system. In this work, we present this http URL, the first comprehensive legal benchmark covering key areas of Portuguese law. To develop this http URL, we first collect long-form questions and answers from real law exams, and then use GPT-4o to convert them into multiple-choice, true/false, and matching formats. Once generated, the questions are filtered and processed to improve the quality of the dataset. To ensure accuracy and relevance, we validate our approach by having a legal professional review a sample of the generated questions. Although the questions are synthetically generated, we show that their basis in human-created exams and our rigorous filtering and processing methods applied result in a reliable benchmark for assessing LLMs' legal knowledge and reasoning abilities. Finally, we evaluate the performance of leading LLMs on this http URL and investigate potential biases in GPT-4o's responses. We also assess the performance of Portuguese lawyers on a sample of questions to establish a baseline for model comparison and validate the benchmark. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLM）在法律领域的应用激发了各国和语言的基准测试的创建。然而，目前还没有专门为葡萄牙法律系统设计的基准测试。在此项工作中，我们介绍了这个http://site-url.com（请将“this http URL”替换为实际的网址），这是首个全面涵盖葡萄牙法律关键领域的基准测试。为了开发这个基准测试，我们首先收集了来自真实法律考试的长文问题和答案，然后使用GPT-4o将其转换为多项选择、真伪判断和匹配格式。生成问题后，我们对其进行筛选和处理，以提高数据集的质量。为了确保准确性和相关性，我们通过让法律专业人士审查生成问题的样本来验证我们的方法。尽管这些问题是合成生成的，但我们表明，它们基于人类创建的考试，并且我们的严格筛选和处理方法确保了这个基准测试可以用于评估LLM的法律知识和推理能力。最后，我们评估了领先LLM在该基准测试上的性能，并调查了GPT-4o响应中的潜在偏差。我们还评估了葡萄牙律师在问题样本上的表现，以此建立模型比较的基线，并验证该基准测试的有效性。 

---
# Iterative Auto-Annotation for Scientific Named Entity Recognition Using BERT-Based Models 

**Title (ZH)**: 使用基于BERT的模型进行迭代自动标注的科学领域命名实体识别 

**Authors**: Kartik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.16312)  

**Abstract**: This paper presents an iterative approach to performing Scientific Named Entity Recognition (SciNER) using BERT-based models. We leverage transfer learning to fine-tune pretrained models with a small but high-quality set of manually annotated data. The process is iteratively refined by using the fine-tuned model to auto-annotate a larger dataset, followed by additional rounds of fine-tuning. We evaluated two models, dslim/bert-large-NER and bert-largecased, and found that bert-large-cased consistently outperformed the former. Our approach demonstrated significant improvements in prediction accuracy and F1 scores, especially for less common entity classes. Future work could include pertaining with unlabeled data, exploring more powerful encoders like RoBERTa, and expanding the scope of manual annotations. This methodology has broader applications in NLP tasks where access to labeled data is limited. 

**Abstract (ZH)**: 本文提出了一种迭代方法，在基于BERT的模型上进行科学命名实体识别（SciNER）。我们利用迁移学习，通过少量但高质量的手动注释数据微调预训练模型。此过程通过使用微调后的模型自动标注更大的数据集，并随后进行多轮微调而逐步优化。我们评估了两种模型，即dslim/bert-large-NER和bert-large-cased，并发现bert-large-cased在各项指标上持续超过了后者。我们的方法在预测准确性和F1分数方面表现出显著改进，尤其是在处理较少常见实体类别时更为明显。未来的工作可以包括使用未标注数据、探索更强大的编码器（如RoBERTa），以及扩展手动注释的范围。该方法在数据标注受限的NLP任务中具有更广泛的应用前景。 

---
# Fine-Tuning Qwen 2.5 3B for Realistic Movie Dialogue Generation 

**Title (ZH)**: Fine-Tuning Qwen 2.5 3B for Realistic Movie Dialogue Generation

（注：此处的“Qwen 2.5 3B”保持不变，因为它是一个特定的模型名称或版本标识，通常在翻译时不进行翻译处理。如果需要进一步解释这个标识的具体含义，可以在翻译的上下文中添加适当的注释。） 

**Authors**: Kartik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.16274)  

**Abstract**: The Qwen 2.5 3B base model was fine-tuned to generate contextually rich and engaging movie dialogue, leveraging the Cornell Movie-Dialog Corpus, a curated dataset of movie conversations. Due to the limitations in GPU computing and VRAM, the training process began with the 0.5B model progressively scaling up to the 1.5B and 3B versions as efficiency improvements were implemented. The Qwen 2.5 series, developed by Alibaba Group, stands at the forefront of small open-source pre-trained models, particularly excelling in creative tasks compared to alternatives like Meta's Llama 3.2 and Google's Gemma. Results demonstrate the ability of small models to produce high-quality, realistic dialogue, offering a promising approach for real-time, context-sensitive conversation generation. 

**Abstract (ZH)**: 以下是翻译成中文的内容，符合学术规范：

基于Cornell Movie-Dialog Corpus（一个由电影对话构成的精选数据集），阿里巴巴集团开发的Qwen 2.5 3B基模型经过微调，生成了富有情境感和吸引力的电影对话。由于GPU计算能力和显存（VRAM）的限制，训练过程从0.5B模型逐步扩展至1.5B和3B版本，期间通过实施效率改进逐步扩大模型规模。Qwen 2.5系列在小型开源预训练模型中处于领先地位，尤其是在创造性任务方面，相比Meta的Llama 3.2和Google的Gemma等替代模型具有明显优势。实验结果表明，小型模型能够生成高质量、具真实感的对话，为实时、情境敏感的对话生成提供了有前景的方法。 

---
# ThinkBench: Dynamic Out-of-Distribution Evaluation for Robust LLM Reasoning 

**Title (ZH)**: ThinkBench: 动态离群分布评估以提升Robust LLM推理能力 

**Authors**: Shulin Huang, Linyi Yang, Yan Song, Shuang Chen, Leyang Cui, Ziyu Wan, Qingcheng Zeng, Ying Wen, Kun Shao, Weinan Zhang, Jun Wang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16268)  

**Abstract**: Evaluating large language models (LLMs) poses significant challenges, particularly due to issues of data contamination and the leakage of correct answers. To address these challenges, we introduce ThinkBench, a novel evaluation framework designed to evaluate LLMs' reasoning capability robustly. ThinkBench proposes a dynamic data generation method for constructing out-of-distribution (OOD) datasets and offers an OOD dataset that contains 2,912 samples drawn from reasoning tasks. ThinkBench unifies the evaluation of reasoning models and non-reasoning models. We evaluate 16 LLMs and 4 PRMs under identical experimental conditions and show that most of the LLMs' performance are far from robust and they face a certain level of data leakage. By dynamically generating OOD datasets, ThinkBench effectively provides a reliable evaluation of LLMs and reduces the impact of data contamination. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）面临着重大挑战，特别是在数据污染和正确答案泄露方面的问题。为了解决这些挑战，我们引入了ThinkBench，这是一个新颖的评估框架，旨在稳健地评估LLMs的推理能力。ThinkBench 提出了一种动态数据生成方法以构建离分布（OOD）数据集，并提供了一个包含2,912个样本的OOD数据集，这些样本来自推理任务。ThinkBench 统一了推理模型和非推理模型的评估。在相同的实验条件下，我们评估了16个LLM和4个过程推理模型（PRMs），结果显示大多数LLM的性能并不稳健，它们面临一定程度的数据泄露问题。通过动态生成OOD数据集，ThinkBench 有效提供了对LLM的可靠评估，并减轻了数据污染的影响。 

---
# Conflicts of Interest in Published NLP Research 2000-2024 

**Title (ZH)**: 2000-2024年发表的NLP研究中的利益冲突 

**Authors**: Maarten Bosten, Bennett Kleinberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.16218)  

**Abstract**: Natural Language Processing research is increasingly reliant on large scale data and computational power. Many achievements in the past decade resulted from collaborations with the tech industry. But an increasing entanglement of academic research and industry interests leads to conflicts of interest. We assessed published NLP research from 2000-2024 and labeled author affiliations as academic or industry-affiliated to measure conflicts of interest. Overall 27.65% of the papers contained at least one industry-affiliated author. That figure increased substantially with more than 1 in 3 papers having a conflict of interest in 2024. We identify top-tier venues (ACL, EMNLP) as main drivers for that effect. The paper closes with a discussion and a simple, concrete suggestion for the future. 

**Abstract (ZH)**: 自然语言处理（NLP）研究越来越依赖大规模数据和计算能力。过去十年间取得的许多成果得益于与科技行业的合作。然而，学术研究与行业利益的日益交织导致了利益冲突。我们评估了2000年至2024年期间发布的NLP研究成果，并将作者的机构属性标记为学术或行业关联，以衡量利益冲突情况。总体来看，27.65%的论文至少包含一名行业关联作者。这一比例在2024年大幅上升，超过三分之一的论文存在利益冲突。我们指出顶级学术会议（ACL、EMNLP）是这种现象的主要推动力。论文结尾处进行了讨论，并提出了一个简单具体的未来建议。 

---
# IPO: Your Language Model is Secretly a Preference Classifier 

**Title (ZH)**: IPO：你的语言模型实际上是偏见分类器 

**Authors**: Shivank Garg, Ayush Singh, Shweta Singh, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2502.16182)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has emerged as the primary method for aligning large language models (LLMs) with human preferences. While it enables LLMs to achieve human-level alignment, it often incurs significant computational and financial costs due to its reliance on training external reward models or human-labeled preferences. In this work, we propose \textbf{Implicit Preference Optimization (IPO)}, an alternative approach that leverages generative LLMs as preference classifiers, thereby reducing the dependence on external human feedback or reward models to obtain preferences. We conduct a comprehensive evaluation on the preference classification ability of LLMs using RewardBench, assessing models across different sizes, architectures, and training levels to validate our hypothesis. Furthermore, we investigate the self-improvement capabilities of LLMs by generating multiple responses for a given instruction and employing the model itself as a preference classifier for Direct Preference Optimization (DPO)-based training. Our findings demonstrate that models trained through IPO achieve performance comparable to those utilizing state-of-the-art reward models for obtaining preferences. 

**Abstract (ZH)**: 基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）已成为使大型语言模型（Large Language Models, LLMs）与人类偏好对齐的主要方法。尽管这种方法可以使LLMs达到人类级别的对齐，但由于其依赖外部奖励模型或人工标注的偏好而导致显著的计算和经济成本。在本研究中，我们提出了一种新的替代方法——**隐式偏好优化（Implicit Preference Optimization, IPO）**，该方法利用生成型LLM作为偏好分类器，从而减少对外部人类反馈或奖励模型的依赖以获得偏好。我们使用RewardBench进行全面评估，测试不同规模、架构和训练水平的模型的偏好分类能力，以验证我们的假设。此外，我们还研究了LLM的自我改进能力，通过为给定指令生成多个响应，并利用模型本身作为偏好评分器来进行直接偏好优化（Direct Preference Optimization, DPO）训练。我们的研究结果表明，通过IPO训练的模型在性能上与利用最先进的奖励模型来获取偏好的模型相当。 

---
# BiDeV: Bilateral Defusing Verification for Complex Claim Fact-Checking 

**Title (ZH)**: BiDeV：双边解爆验证在复杂声明事实核查中的应用 

**Authors**: Yuxuan Liu, Hongda Sun, Wenya Guo, Xinyan Xiao, Cunli Mao, Zhengtao Yu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.16181)  

**Abstract**: Complex claim fact-checking performs a crucial role in disinformation detection. However, existing fact-checking methods struggle with claim vagueness, specifically in effectively handling latent information and complex relations within claims. Moreover, evidence redundancy, where nonessential information complicates the verification process, remains a significant issue. To tackle these limitations, we propose Bilateral Defusing Verification (BiDeV), a novel fact-checking working-flow framework integrating multiple role-played LLMs to mimic the human-expert fact-checking process. BiDeV consists of two main modules: Vagueness Defusing identifies latent information and resolves complex relations to simplify the claim, and Redundancy Defusing eliminates redundant content to enhance the evidence quality. Extensive experimental results on two widely used challenging fact-checking benchmarks (Hover and Feverous-s) demonstrate that our BiDeV can achieve the best performance under both gold and open settings. This highlights the effectiveness of BiDeV in handling complex claims and ensuring precise fact-checking 

**Abstract (ZH)**: 复杂声明的事实核查在虚假信息检测中扮演着至关重要的角色。然而，现有的事实核查方法在处理声明的模糊性时遇到了困难，特别是在有效处理声明中的潜在信息和复杂关系方面。此外，证据冗余问题——即非必要的信息会复杂化验证过程——仍然是一个显著的问题。为了克服这些局限性，我们提出了一种名为双边解压验证（BiDeV）的新颖事实核查工作流框架，该框架结合了多个角色扮演的大规模语言模型（LLM），以模拟人类专家的事实核查过程。BiDeV 包含两个主要模块：模糊性解压（Vagueness Defusing）识别潜在信息、解决复杂关系以简化声明，冗余解压（Redundancy Defusing）消除冗余内容以提高证据质量。在两个广泛使用的挑战性事实核查基准数据集（Hover 和 Feverous-s）上的大量实验结果表明，我们的 BiDeV 在黄金设置和开放设置下都能实现最佳性能。这突显了 BiDeV 在处理复杂声明和确保精确事实核查方面的有效性。 

---
# OrderSum: Semantic Sentence Ordering for Extractive Summarization 

**Title (ZH)**: OrderSum：提取式总结的语义句子排序方法 

**Authors**: Taewan Kwon, Sangyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.16180)  

**Abstract**: There are two main approaches to recent extractive summarization: the sentence-level framework, which selects sentences to include in a summary individually, and the summary-level framework, which generates multiple candidate summaries and ranks them. Previous work in both frameworks has primarily focused on improving which sentences in a document should be included in the summary. However, the sentence order of extractive summaries, which is critical for the quality of a summary, remains underexplored. In this paper, we introduce OrderSum, a novel extractive summarization model that semantically orders sentences within an extractive summary. OrderSum proposes a new representation method to incorporate the sentence order into the embedding of the extractive summary, and an objective function to train the model to identify which extractive summary has a better sentence order in the semantic space. Extensive experimental results demonstrate that OrderSum obtains state-of-the-art performance in both sentence inclusion and sentence order for extractive summarization. In particular, OrderSum achieves a ROUGE-L score of 30.52 on CNN/DailyMail, outperforming the previous state-of-the-art model by a large margin of 2.54. 

**Abstract (ZH)**: 近年来提取式摘要的主要方法有两种：句内级框架，该框架逐句选择要包含在摘要中的句子；以及摘要级框架，该框架生成多个候选摘要并对其进行排名。在两种框架中，以往工作的主要焦点在于提高文档中哪些句子应该被包含在摘要中。然而，提取式摘要的句子顺序问题，这是影响摘要质量的关键因素，尚未得到充分探索。本文提出了一种名为OrderSum的新颖提取式摘要模型，该模型在提取式摘要内部以语义方式排列句子。OrderSum提出了一种新的表示方法，将句子顺序融入提取式摘要的嵌入表示中，并定义了一个目标函数，以训练模型识别在语义空间中哪个提取式摘要的句子顺序更好。广泛实验结果表明，OrderSum在句子选择和句序排列方面均获得了最佳性能。特别是在CNN/DailyMail数据集上，OrderSum的ROUGE-L得分为30.52，比之前的最佳模型高出2.54分，显著地超过了前者的性能。 

---
# Mapping 1,000+ Language Models via the Log-Likelihood Vector 

**Title (ZH)**: 通过log似然向量映射1,000多个语言模型 

**Authors**: Momose Oyama, Hiroaki Yamagiwa, Yusuke Takase, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2502.16173)  

**Abstract**: To compare autoregressive language models at scale, we propose using log-likelihood vectors computed on a predefined text set as model features. This approach has a solid theoretical basis: when treated as model coordinates, their squared Euclidean distance approximates the Kullback-Leibler divergence of text-generation probabilities. Our method is highly scalable, with computational cost growing linearly in both the number of models and text samples, and is easy to implement as the required features are derived from cross-entropy loss. Applying this method to over 1,000 language models, we constructed a "model map," providing a new perspective on large-scale model analysis. 

**Abstract (ZH)**: 为了在大规模范围内比较自回归语言模型，我们提出使用在预定义文本集上计算的对数似然向量作为模型特征。这种方法具有坚实理论基础：当将这些对数似然向量视为模型坐标时，它们的平方欧几里得距离近似于文本生成概率的Kullback-Leibler散度。该方法高度可扩展，其计算成本随模型数量和文本样本数量线性增长，并且易于实现，所需的特征源自交叉熵损失。将此方法应用于超过1000个语言模型后，我们构建了一个“模型地图”，为大规模模型分析提供了新的视角。 

---
# EPERM: An Evidence Path Enhanced Reasoning Model for Knowledge Graph Question and Answering 

**Title (ZH)**: EPERM：一种基于证据路径增强的推理模型，用于知识图谱问答 

**Authors**: Xiao Long, Liansheng Zhuang, Aodi Li, Minghong Yao, Shafei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16171)  

**Abstract**: Due to the remarkable reasoning ability, Large language models (LLMs) have demonstrated impressive performance in knowledge graph question answering (KGQA) tasks, which find answers to natural language questions over knowledge graphs (KGs). To alleviate the hallucinations and lack of knowledge issues of LLMs, existing methods often retrieve the question-related information from KGs to enrich the input context. However, most methods focus on retrieving the relevant information while ignoring the importance of different types of knowledge in reasoning, which degrades their performance. To this end, this paper reformulates the KGQA problem as a graphical model and proposes a three-stage framework named the Evidence Path Enhanced Reasoning Model (EPERM) for KGQA. In the first stage, EPERM uses the fine-tuned LLM to retrieve a subgraph related to the question from the original knowledge graph. In the second stage, EPERM filters out the evidence paths that faithfully support the reasoning of the questions, and score their importance in reasoning. Finally, EPERM uses the weighted evidence paths to reason the final answer. Since considering the importance of different structural information in KGs for reasoning, EPERM can improve the reasoning ability of LLMs in KGQA tasks. Extensive experiments on benchmark datasets demonstrate that EPERM achieves superior performances in KGQA tasks. 

**Abstract (ZH)**: 由于大型语言模型（LLMs）表现出令人瞩目的推理能力，它们在知识图谱问答（KGQA）任务中已经展现出了出色的性能，这些任务涉及在知识图谱（KGs）中寻找自然语言问题的答案。为了解决LLMs的生成幻觉和知识不足问题，现有方法通常从KGs中检索与问题相关的信息以丰富输入语境。然而，大多数方法只关注检索相关信息，而忽略了不同类型知识在推理中的重要性，从而降低了性能。为了解决这一问题，本文将KGQA问题重新表述为图形模型，并提出了一种名为证据路径增强推理模型（EPERM）的三级框架。在第一阶段，EPERM利用微调后的LLM从原始知识图谱中检索与问题相关的子图。在第二阶段，EPERM筛选出能够支持问题推理的证据路径，并评估它们在推理中的重要性。最后，EPERM使用加权证据路径来推导最终答案。通过考虑KG中不同结构信息的重要性，EPERM能够提高LLMs在KGQA任务中的推理能力。在基准数据集上的广泛实验表明，EPERM在KGQA任务中的性能表现优异。 

---
# Number Representations in LLMs: A Computational Parallel to Human Perception 

**Title (ZH)**: LLMs中的数字表示：类人类感知的计算 parallel 

**Authors**: H.V. AlquBoj, Hilal AlQuabeh, Velibor Bojkovic, Tatsuya Hiraoka, Ahmed Oumar El-Shangiti, Munachiso Nwadike, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2502.16147)  

**Abstract**: Humans are believed to perceive numbers on a logarithmic mental number line, where smaller values are represented with greater resolution than larger ones. This cognitive bias, supported by neuroscience and behavioral studies, suggests that numerical magnitudes are processed in a sublinear fashion rather than on a uniform linear scale. Inspired by this hypothesis, we investigate whether large language models (LLMs) exhibit a similar logarithmic-like structure in their internal numerical representations. By analyzing how numerical values are encoded across different layers of LLMs, we apply dimensionality reduction techniques such as PCA and PLS followed by geometric regression to uncover latent structures in the learned embeddings. Our findings reveal that the model's numerical representations exhibit sublinear spacing, with distances between values aligning with a logarithmic scale. This suggests that LLMs, much like humans, may encode numbers in a compressed, non-uniform manner. 

**Abstract (ZH)**: 人类被认为在心智数轴上感知数字，其中较小的数值以更高的分辨率表示，而较大的数值则相反。这种认知偏差，得到了神经科学和行为学研究的支持，表明数字大小是以非线性而非均匀线性的方式进行处理的。受到这一假设的启发，我们探究大型语言模型（LLMs）在其内部数字表示中是否具有类似对数性质的结构。通过分析不同层的LLM如何编码数值，我们应用如主成分分析（PCA）和偏最小二乘（PLS）等维度归约技术，再结合几何回归来揭示学习嵌入中的潜在结构。我们的发现表明，模型的数字表示具有非线性间隔，数值之间的距离与对数尺度相吻合。这表明，类似人类，LLMs也可能以压缩且非均匀的方式编码数字。 

---
# The Law of Knowledge Overshadowing: Towards Understanding, Predicting, and Preventing LLM Hallucination 

**Title (ZH)**: 知识覆盖法则：关于理解、预测和防止大语言模型幻觉的探索 

**Authors**: Yuji Zhang, Sha Li, Cheng Qian, Jiateng Liu, Pengfei Yu, Chi Han, Yi R. Fung, Kathleen McKeown, Chengxiang Zhai, Manling Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.16143)  

**Abstract**: Hallucination is a persistent challenge in large language models (LLMs), where even with rigorous quality control, models often generate distorted facts. This paradox, in which error generation continues despite high-quality training data, calls for a deeper understanding of the underlying LLM mechanisms. To address it, we propose a novel concept: knowledge overshadowing, where model's dominant knowledge can obscure less prominent knowledge during text generation, causing the model to fabricate inaccurate details. Building on this idea, we introduce a novel framework to quantify factual hallucinations by modeling knowledge overshadowing. Central to our approach is the log-linear law, which predicts that the rate of factual hallucination increases linearly with the logarithmic scale of (1) Knowledge Popularity, (2) Knowledge Length, and (3) Model Size. The law provides a means to preemptively quantify hallucinations, offering foresight into their occurrence even before model training or inference. Built on overshadowing effect, we propose a new decoding strategy CoDa, to mitigate hallucinations, which notably enhance model factuality on Overshadow (27.9%), MemoTrap (13.1%) and NQ-Swap (18.3%). Our findings not only deepen understandings of the underlying mechanisms behind hallucinations but also provide actionable insights for developing more predictable and controllable language models. 

**Abstract (ZH)**: 幻觉一直是大型语言模型（LLMs）的一个持续性挑战，即使在严格的质量控制下，模型仍会产生失真的事实。尽管高质量的训练数据可以减少错误，但这种错误生成的现象仍然存在，这引发了对LLM内在机制的深入理解需求。为了解决这一问题，我们提出了一种新的概念：知识遮蔽（Knowledge Overshadowing），即模型的主导性知识在文本生成过程中可能掩盖次要知识，导致模型生成不准确的细节。基于这一想法，我们引入了一种新的框架来量化事实性幻觉，通过建模知识遮蔽现象。我们的方法的核心是指数线性定律（log-linear law），该定律预测事实性幻觉的频率随着知识流行度、知识长度和模型规模的对数尺度线性增加。该定律提供了一种预先量化幻觉的方法，可以在模型训练或推理之前预测幻觉的发生。基于遮蔽效应，我们提出了一种新的解码策略CoDa，该策略显著提高了模型在Overshadow（27.9%）、MemoTrap（13.1%）和NQ-Swap（18.3%）上的事实性。我们的发现不仅深化了对幻觉内在机制的理解，还为开发更具可预测性和可控性的语言模型提供了可操作的见解。 

---
# Understanding Zero-shot Rare Word Recognition Improvements Through LLM Integration 

**Title (ZH)**: 通过大型语言模型集成理解零样本罕见词汇识别性能的提升 

**Authors**: Haoxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16142)  

**Abstract**: In this study, we investigate the integration of a large language model (LLM) with an automatic speech recognition (ASR) system, specifically focusing on enhancing rare word recognition performance. Using a 190,000-hour dataset primarily sourced from YouTube, pre-processed with Whisper V3 pseudo-labeling, we demonstrate that the LLM-ASR architecture outperforms traditional Zipformer-Transducer models in the zero-shot rare word recognition task, after training on a large dataset. Our analysis reveals that the LLM contributes significantly to improvements in rare word error rate (R-WER), while the speech encoder primarily determines overall transcription performance (Orthographic Word Error Rate, O-WER, and Normalized Word Error Rate, N-WER). Through extensive ablation studies, we highlight the importance of adapter integration in aligning speech encoder outputs with the LLM's linguistic capabilities. Furthermore, we emphasize the critical role of high-quality labeled data in achieving optimal performance. These findings provide valuable insights into the synergy between LLM-based ASR architectures, paving the way for future advancements in large-scale LLM-based speech recognition systems. 

**Abstract (ZH)**: 在本研究中，我们探讨了大型语言模型（LLM）与自动语音识别（ASR）系统的整合，特别关注于提升罕见词汇识别性能。我们使用了主要来源于YouTube的19万小时数据集，并利用Whisper V3伪标注进行预处理。结果显示，在经过大规模数据集训练后，LLM-ASR架构在零样本罕见词汇识别任务中的表现优于传统的Zipformer-Transducer模型。我们的分析表明，LLM 在减少罕见词汇错误率（R-WER）方面贡献显著，而语音编码器主要决定了整体转写性能（书面词错误率，O-WER，及归一化词错误率，N-WER）。通过广泛的消融研究，我们强调了适配器集成在使语音编码器输出与LLM的语义能力相匹配中的重要性。此外，我们还强调了高质量标注数据对于实现最佳性能的关键作用。这些发现为LLM基础的ASR架构之间的协同作用提供了宝贵的见解，为未来基于大规模LLM的语音识别系统的发展铺平了道路。 

---
# Chain-of-Description: What I can understand, I can put into words 

**Title (ZH)**: 描述链：我能理解的，我能用语言表达出来 

**Authors**: Jiaxin Guo, Daimeng Wei, Zongyao Li, Hengchao Shang, Yuanchang Luo, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16137)  

**Abstract**: In this paper, we propose a novel strategy defined as Chain-of-Description (CoD) Prompting, tailored for Multi-Modal Large Language Models. This approach involves having the model first provide a detailed description of the multi-modal input before generating an answer to the question. When applied to models such as Qwen2-Audio, Qwen2-VL, and Qwen2.5-VL, CoD Prompting significantly enhances performance compared to standard prompting methods. This is demonstrated by nearly a 4\% improvement in the speech category of the audio benchmark AIR-Bench-Chat and a 5.3\% improvement in the hard-level portion of the vision benchmark MMMU\_Pro. Our ablation study further validates the effectiveness of CoD Prompting. 

**Abstract (ZH)**: 在本文中，我们提出了一种名为链式描述（Chain-of-Description, CoD）提示的新策略，专门用于多模态大型语言模型。该方法包括让模型首先对多模态输入进行详细描述，然后再生成答案。将CoD提示应用于Qwen2-Audio、Qwen2-VL和Qwen2.5-VL等模型时，其性能显著优于标准提示方法。在音频基准AIR-Bench-Chat的语音类别中，CoD提示将近提高了4%，而在视觉基准MMMU\_Pro的高难度部分，CoD提示提高了5.3%。进一步的消融研究进一步验证了CoD提示的有效性。 

---
# Be a Multitude to Itself: A Prompt Evolution Framework for Red Teaming 

**Title (ZH)**: 自我众化的提示演化框架：用于红队行动的模型 

**Authors**: Rui Li, Peiyi Wang, Jingyuan Ma, Di Zhang, Lei Sha, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2502.16109)  

**Abstract**: Large Language Models (LLMs) have gained increasing attention for their remarkable capacity, alongside concerns about safety arising from their potential to produce harmful content. Red teaming aims to find prompts that could elicit harmful responses from LLMs, and is essential to discover and mitigate safety risks before real-world deployment. However, manual red teaming is both time-consuming and expensive, rendering it unscalable. In this paper, we propose RTPE, a scalable evolution framework to evolve red teaming prompts across both breadth and depth dimensions, facilitating the automatic generation of numerous high-quality and diverse red teaming prompts. Specifically, in-breadth evolving employs a novel enhanced in-context learning method to create a multitude of quality prompts, whereas in-depth evolving applies customized transformation operations to enhance both content and form of prompts, thereby increasing diversity. Extensive experiments demonstrate that RTPE surpasses existing representative automatic red teaming methods on both attack success rate and diversity. In addition, based on 4,800 red teaming prompts created by RTPE, we further provide a systematic analysis of 8 representative LLMs across 8 sensitive topics. 

**Abstract (ZH)**: 大规模语言模型（LLMs）因其显著的能力而受到广泛关注，同时也引发了对其潜在产生有害内容的安全性问题的关注。红队攻击旨在寻找可能引发LLMs产生有害响应的提示，并在实际部署前发现和减轻安全风险方面至关重要。然而，手动红队攻击既耗时又昂贵，使其不具可扩展性。本文提出了一种可扩展的进化框架RTPE，以在广度和深度两个维度上进化红队提示，促进自动化生成大量高质量和多样化的红队提示。具体而言，广度演进采用了一种新型增强的上下文学习方法来生成大量高质量提示，而深度演进则通过定制的转换操作提高提示的内容和形式，从而增加多样性。广泛实验表明，RTPE在攻击成功率和多样性上均优于现有的代表性自动红队方法。此外，基于RTPE生成的4,800个红队提示，我们对8种代表性的大规模语言模型在8个敏感话题上进行了系统的分析。 

---
# Echo: A Large Language Model with Temporal Episodic Memory 

**Title (ZH)**: 回声：具有时间 episodic 记忆的大语言模型 

**Authors**: WenTao Liu, Ruohua Zhang, Aimin Zhou, Feng Gao, JiaLi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16090)  

**Abstract**: Research on large language models (LLMs) has shown remarkable performance in domains such as mathematics, programming, and literary creation. However, most studies have focused on semantic memory-based question answering, neglecting LLMs' potential to handle episodic memory (EM)-related queries. This oversight has led to suboptimal performance in applications requiring EM, including emotional companionship, personal AI assistants, and AI teachers. To address this gap, we introduce Echo, a LLM enhanced with temporal episodic memory. We propose a Multi-Agent Data Generation Framework that guides the model in generating multi-turn, complex scenario episodic memory dialogue data (EM-Train). Temporal information is innovatively incorporated into the LLM training process, and Echo is trained using the EM-Train. Furthermore, We develop an EM-Test benchmark specifically designed to evaluate LLMs' episodic memory capabilities. The EM-Test assesses performance across various time spans and difficulty levels, providing a comprehensive evaluation of multi-turn episodic memory dialogues. Our experiments demonstrate that Echo significantly outperforms state-of-the-art LLMs on EM-Test. Additionally, a qualitative analysis reveals Echo's potential to exhibit human-like episodic memory capabilities. We will open-source all datasets, code, and model weights. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学、编程和文学创作等领域表现出了显著的能力。然而，大多数研究主要集中在基于语义记忆的问题回答上，忽视了LLMs处理事件记忆（EM）相关查询的潜力。这一忽视导致了在需要EM的应用场景中的性能不佳，包括情感陪伴、个人AI助手和AI教师等。为了解决这一缺口，我们引入了Echo，这是一种增强有时间事件记忆的LLM。我们提出了一个多智能体数据生成框架，该框架指导模型生成多轮、复杂场景的事件记忆对话数据（EM-Train）。时间信息被创新地引入到LLM的训练过程中，并使用EM-Train对Echo进行训练。此外，我们开发了一个EM-Test基准，专门设计用于评估LLMs的事件记忆能力。EM-Test评估了不同时间跨度和难度级别上的性能，提供了全面的多轮事件记忆对话评估。我们的实验表明，Echo在EM-Test上明显优于最先进的LLM。此外，定性的分析揭示了Echo具备展示类似人类事件记忆能力的潜力。我们将开源所有数据集、代码和模型权重。 

---
# Moving Beyond Medical Exam Questions: A Clinician-Annotated Dataset of Real-World Tasks and Ambiguity in Mental Healthcare 

**Title (ZH)**: 超越医学考试问题：心理健康护理中实际任务和歧义的临床标注数据集 

**Authors**: Max Lamparth, Declan Grabb, Amy Franks, Scott Gershan, Kaitlyn N. Kunstman, Aaron Lulla, Monika Drummond Roots, Manu Sharma, Aryan Shrivastava, Nina Vasan, Colleen Waickman  

**Link**: [PDF](https://arxiv.org/pdf/2502.16051)  

**Abstract**: Current medical language model (LM) benchmarks often over-simplify the complexities of day-to-day clinical practice tasks and instead rely on evaluating LMs on multiple-choice board exam questions. Thus, we present an expert-created and annotated dataset spanning five critical domains of decision-making in mental healthcare: treatment, diagnosis, documentation, monitoring, and triage. This dataset - created without any LM assistance - is designed to capture the nuanced clinical reasoning and daily ambiguities mental health practitioners encounter, reflecting the inherent complexities of care delivery that are missing from existing datasets. Almost all 203 base questions with five answer options each have had the decision-irrelevant demographic patient information removed and replaced with variables (e.g., AGE), and are available for male, female, or non-binary-coded patients. For question categories dealing with ambiguity and multiple valid answer options, we create a preference dataset with uncertainties from the expert annotations. We outline a series of intended use cases and demonstrate the usability of our dataset by evaluating eleven off-the-shelf and four mental health fine-tuned LMs on category-specific task accuracy, on the impact of patient demographic information on decision-making, and how consistently free-form responses deviate from human annotated samples. 

**Abstract (ZH)**: 当前的医学语言模型（LM）基准往往过分简化了日常临床实践任务的复杂性，反而依赖于在多选题形式的医学考试题目上评估LM。因此，我们提供了一个由专家创建和标注的数据集，涵盖了精神卫生保健决策中的五个关键领域：治疗、诊断、记录、监测和分诊。该数据集无需任何LM的帮助，旨在捕捉精神卫生从业者在日常工作中遇到的复杂临床推理和模棱两可的情况，反映现有数据集中所缺失的护理交付内在复杂性。几乎所有的203个基础问题，每个问题有五个选项，均已移除了与决策无关的患者人口统计学信息，并以变量形式（例如，年龄）进行了替换，适用于男性、女性或非二元性别标记的患者。对于涉及模棱两可和多个有效选项的问题类别，我们创建了一个带有专家注释不确定性的偏好数据集。我们列出了一系列预期使用场景，并通过评估十一款现成和四款精神卫生微调LM在特定分类任务准确性上的表现，评估患者人口统计学信息对决策的影响，以及自由格式响应与人类标注样本的一致性，展示了该数据集的可用性。 

---
# Enhancing LLMs for Identifying and Prioritizing Important Medical Jargons from Electronic Health Record Notes Utilizing Data Augmentation 

**Title (ZH)**: 利用数据增强提升大型语言模型在电子健康记录笔记中识别和优先处理重要医学术语的能力 

**Authors**: Won Seok Jang, Sharmin Sultana, Zonghai Yao, Hieu Tran, Zhichao Yang, Sunjae Kwon, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16022)  

**Abstract**: Objective: OpenNotes enables patients to access EHR notes, but medical jargon can hinder comprehension. To improve understanding, we evaluated closed- and open-source LLMs for extracting and prioritizing key medical terms using prompting, fine-tuning, and data augmentation.
Materials and Methods: We assessed LLMs on 106 expert-annotated EHR notes, experimenting with (i) general vs. structured prompts, (ii) zero-shot vs. few-shot prompting, (iii) fine-tuning, and (iv) data augmentation. To enhance open-source models in low-resource settings, we used ChatGPT for data augmentation and applied ranking techniques. We incrementally increased the augmented dataset size (10 to 10,000) and conducted 5-fold cross-validation, reporting F1 score and Mean Reciprocal Rank (MRR).
Results and Discussion: Fine-tuning and data augmentation improved performance over other strategies. GPT-4 Turbo achieved the highest F1 (0.433), while Mistral7B with data augmentation had the highest MRR (0.746). Open-source models, when fine-tuned or augmented, outperformed closed-source models. Notably, the best F1 and MRR scores did not always align. Few-shot prompting outperformed zero-shot in vanilla models, and structured prompts yielded different preferences across models. Fine-tuning improved zero-shot performance but sometimes degraded few-shot performance. Data augmentation performed comparably or better than other methods.
Conclusion: Our evaluation highlights the effectiveness of prompting, fine-tuning, and data augmentation in improving model performance for medical jargon extraction in low-resource scenarios. 

**Abstract (ZH)**: 目标：OpenNotes 允许患者访问电子健康记录（EHR）笔记，但医学术语可能妨碍理解。为了提高理解能力，我们评估了闭源和开源的大语言模型（LLM），通过提示、微调和数据增强来提取和优先处理关键医学术语。

材料与方法：我们使用106份由专家标注的EHR笔记，实验了（i）通用提示与结构化提示，（ii）零样本提示与少样本提示，（iii）微调，和（iv）数据增强。为了在资源有限的场景中增强开源模型，我们使用了ChatGPT进行数据增强，并应用了排序技术。我们逐步增加了增强数据集的规模（从10到10,000），并进行了5折交叉验证，报告了F1分数和平均互换倒数排名（MRR）。

结果与讨论：微调和数据增强在其他策略上提高了性能。GPT-4 Turbo的F1分数最高（0.433），而Mistral7B结合数据增强的MRR最高（0.746）。当进行微调或数据增强时，开源模型的性能优于闭源模型。值得注意的是，最好的F1和MRR分数并不总是对应。通用模型中少样本提示优于零样本提示，而结构化提示在不同模型中的偏好不同。微调改善了零样本提示的性能，但在某些情况下降低了少样本提示的性能。数据增强的表现与其它方法相当或更佳。

结论：我们的评估突显了在低资源场景中提高医学术语提取模型性能的有效性，通过提示、微调和数据增强的方法。 

---
# KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse 

**Title (ZH)**: KVLink：通过高效的键值缓存重用加速大规模语言模型 

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16002)  

**Abstract**: We describe KVLink, an approach for efficient key-value (KV) cache reuse in large language models (LLMs). In many LLM applications, different inputs can share overlapping context, such as the same retrieved document appearing in multiple queries. However, the LLMs still need to encode the entire context for each query, leading to redundant computation. In this paper, we propose a new strategy to eliminate such inefficiency, where the KV cache of each document is precomputed independently. During inference, the KV caches of retrieved documents are concatenated, allowing the model to reuse cached representations instead of recomputing them. To mitigate the performance degradation of LLMs when using KV caches computed independently for each document, KVLink introduces three key components: adjusting positional embeddings of the KV cache at inference to match the global position after concatenation, using trainable special tokens to restore self-attention across independently encoded documents, and applying mixed-data fine-tuning to enhance performance while preserving the model's original capabilities. Experiments across 7 datasets demonstrate that KVLink improves question answering accuracy by an average of 4% over state-of-the-art methods. Furthermore, by leveraging precomputed KV caches, our approach reduces time-to-first-token by up to 90% compared to standard LLM inference, making it a scalable and efficient solution for context reuse. 

**Abstract (ZH)**: 以下是符合学术规范的翻译内容：

我们描述了KVLink——一种用于大型语言模型（LLMs）高效键值（KV）缓存重用的方法。在许多LLM应用中，不同的输入可以共享重叠的背景信息，例如同一份检索到的文档出现在多个查询中。然而，LLMs仍然需要为每个查询重新编码整个背景信息，导致了重复的计算。在这篇论文中，我们提出了一种新的策略来消除这种低效性，即针对每个文档独立预先计算其KV缓存。在推理过程中，检索到的文档的KV缓存会被连接起来，从而使模型能够重用缓存的表示，而不是重新计算它们。为了缓解使用每个文档独立计算的KV缓存对LLMs性能的影响，KVLink引入了三个关键组件：在推理时调整KV缓存的相对位置嵌入以匹配连接后的全局位置、使用可训练的特殊标记以恢复独立编码文档之间的自注意力、以及采用混合数据微调以增强性能同时保留模型原有的能力。通过在7个数据集上的实验表明，KVLink相对于最先进的方法能够使问答准确率平均提高4%。此外，通过利用预先计算的KV缓存，我们的方法能够将首个词token的生成时间最多缩短90%，从而提供了一个可扩展且高效的背景信息重用解决方案。 

---
# Med-gte-hybrid: A contextual embedding transformer model for extracting actionable information from clinical texts 

**Title (ZH)**: Med-gte-hybrid：一种用于从临床文本中提取可操作信息的上下文嵌入变压器模型 

**Authors**: Aditya Kumar, Simon Rauch, Mario Cypko, Oliver Amft  

**Link**: [PDF](https://arxiv.org/pdf/2502.15996)  

**Abstract**: We introduce a novel contextual embedding model med-gte-hybrid that was derived from the gte-large sentence transformer to extract information from unstructured clinical narratives. Our model tuning strategy for med-gte-hybrid combines contrastive learning and a denoising autoencoder. To evaluate the performance of med-gte-hybrid, we investigate several clinical prediction tasks in large patient cohorts extracted from the MIMIC-IV dataset, including Chronic Kidney Disease (CKD) patient prognosis, estimated glomerular filtration rate (eGFR) prediction, and patient mortality prediction. Furthermore, we demonstrate that the med-gte-hybrid model improves patient stratification, clustering, and text retrieval, thus outperforms current state-of-the-art models on the Massive Text Embedding Benchmark (MTEB). While some of our evaluations focus on CKD, our hybrid tuning of sentence transformers could be transferred to other medical domains and has the potential to improve clinical decision-making and personalised treatment pathways in various healthcare applications. 

**Abstract (ZH)**: 我们提出了一种新颖的情境嵌入模型 med-gte-hybrid，该模型是从 gte-large 句子变换器派生而来，旨在从非结构化的临床叙述中提取信息。med-gte-hybrid 模型的调优策略结合了对比学习和去噪自编码器。为了评估 med-gte-hybrid 的性能，我们在从 MIMIC-IV 数据集中提取的大规模患者队列中进行了多种临床预测任务的研究，包括慢性肾病（CKD）患者的预后、估算肾小球滤过率（eGFR）预测以及患者死亡率预测。此外，我们展示了 med-gte-hybrid 模型在患者分层、聚类和文本检索方面的改进，从而在大规模文本嵌入基准测试（MTEB）中优于当前最先进的模型。虽然部分评估专注于慢性肾病，但我们的句子变换器的混合调优方法可以应用于其他医学领域，并有可能在各种医疗健康应用中改善临床决策和个人化治疗途径。 

---
# Sparsity May Be All You Need: Sparse Random Parameter Adaptation 

**Title (ZH)**: 稀疏性可能是你所需要的：稀疏随机参数自适应 

**Authors**: Jesus Rios, Pierre Dognin, Ronny Luss, Karthikeyan N. Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.15975)  

**Abstract**: Full fine-tuning of large language models for alignment and task adaptation has become prohibitively expensive as models have grown in size. Parameter-Efficient Fine-Tuning (PEFT) methods aim at significantly reducing the computational and memory resources needed for fine-tuning these models by only training on a small number of parameters instead of all model parameters. Currently, the most popular PEFT method is the Low-Rank Adaptation (LoRA), which freezes the parameters of the model to be fine-tuned and introduces a small set of trainable parameters in the form of low-rank matrices. We propose simply reducing the number of trainable parameters by randomly selecting a small proportion of the model parameters to train on. In this paper, we compare the efficiency and performance of our proposed approach with PEFT methods, including LoRA, as well as full parameter fine-tuning. 

**Abstract (ZH)**: 随着语言模型规模的扩大，对这些模型进行全面微调以实现对齐和任务适应已经变得极其昂贵。参数高效微调（PEFT）方法旨在通过仅训练少量参数而不是所有模型参数，显著减少微调所需的时间和内存资源。目前最受欢迎的PEFT方法是低秩适应（LoRA），该方法冻结了要微调模型的参数，并通过低秩矩阵引入了一组可训练的参数。我们提出了一个简单的方法，即随机选择一小部分模型参数进行训练，从而减少可训练参数的数量。本文将比较我们提出的方法与PEFT方法（包括LoRA）以及全面参数微调在效率和性能方面的优劣。 

---
# R$^3$Mem: Bridging Memory Retention and Retrieval via Reversible Compression 

**Title (ZH)**: R$^3$\scMem：通过可逆压缩连接记忆保持与检索 

**Authors**: Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15957)  

**Abstract**: Memory plays a key role in enhancing LLMs' performance when deployed to real-world applications. Existing solutions face trade-offs: explicit memory designs based on external storage require complex management and incur storage overhead, while implicit memory designs that store information via parameters struggle with reliable retrieval. In this paper, we propose R$^3$Mem, a memory network that optimizes both information Retention and Retrieval through Reversible context compression. Specifically, R$^3$Mem employs virtual memory tokens to compress and encode infinitely long histories, further enhanced by a hierarchical compression strategy that refines information from document- to entity-level for improved assimilation across granularities. For retrieval, R$^3$Mem employs a reversible architecture, reconstructing raw data by invoking the model backward with compressed information. Implemented via parameter-efficient fine-tuning, it can integrate seamlessly with any Transformer-based model. Experiments demonstrate that our memory design achieves state-of-the-art performance in long-context language modeling and retrieval-augmented generation tasks. It also significantly outperforms conventional memory modules in long-horizon interaction tasks like conversational agents, showcasing its potential for next-generation retrieval systems. 

**Abstract (ZH)**: 记忆力在增强实际应用场景中大型语言模型（LLM）的表现方面发挥着关键作用。现有的解决方案存在权衡：基于外部存储的显式记忆力设计需要复杂的管理和存储开销，而通过参数存储信息的隐式记忆力设计则面临可靠检索的挑战。在本文中，我们提出了R$^3$Mem，这是一种通过可逆上下文压缩优化记忆力存与取的内存网络。具体而言，R$^3$Mem 使用虚拟内存标记来压缩和编码无限长的历史记录，并进一步通过分层压缩策略根据相关信息从文档级细化到实体级，从而提高不同粒度下的吸收能力。对于检索部分，R$^3$Mem 使用一种可逆的架构，通过反向调用模型重建原始数据，而压缩的信息则用于重构。该设计通过参数高效的微调实现，可以无缝集成到任何基于Transformer的模型中。实验结果表明，我们的记忆力设计在长上下文语言建模和检索增强生成任务中达到了最先进的性能。此外，R$^3$Mem 在长时段交互任务（如对话代理）中显著优于传统的记忆力模块，展示了其对未来一代检索系统潜力的前景。 

---
# MMRAG: Multi-Mode Retrieval-Augmented Generation with Large Language Models for Biomedical In-Context Learning 

**Title (ZH)**: MMRAG：基于大规模语言模型的多模态检索增强生成方法在生物医学领域内的上下文学习 

**Authors**: Zaifu Zhan, Jun Wang, Shuang Zhou, Jiawen Deng, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15954)  

**Abstract**: Objective: To optimize in-context learning in biomedical natural language processing by improving example selection. Methods: We introduce a novel multi-mode retrieval-augmented generation (MMRAG) framework, which integrates four retrieval strategies: (1) Random Mode, selecting examples arbitrarily; (2) Top Mode, retrieving the most relevant examples based on similarity; (3) Diversity Mode, ensuring variation in selected examples; and (4) Class Mode, selecting category-representative examples. This study evaluates MMRAG on three core biomedical NLP tasks: Named Entity Recognition (NER), Relation Extraction (RE), and Text Classification (TC). The datasets used include BC2GM for gene and protein mention recognition (NER), DDI for drug-drug interaction extraction (RE), GIT for general biomedical information extraction (RE), and HealthAdvice for health-related text classification (TC). The framework is tested with two large language models (Llama2-7B, Llama3-8B) and three retrievers (Contriever, MedCPT, BGE-Large) to assess performance across different retrieval strategies. Results: The results from the Random mode indicate that providing more examples in the prompt improves the model's generation performance. Meanwhile, Top mode and Diversity mode significantly outperform Random mode on the RE (DDI) task, achieving an F1 score of 0.9669, a 26.4% improvement. Among the three retrievers tested, Contriever outperformed the other two in a greater number of experiments. Additionally, Llama 2 and Llama 3 demonstrated varying capabilities across different tasks, with Llama 3 showing a clear advantage in handling NER tasks. Conclusion: MMRAG effectively enhances biomedical in-context learning by refining example selection, mitigating data scarcity issues, and demonstrating superior adaptability for NLP-driven healthcare applications. 

**Abstract (ZH)**: 目标：通过改进示例选择来优化生物医学自然语言处理中的上下文学习。方法：我们提出了一种新颖的多模式检索增强生成（MMRAG）框架，该框架结合了四种检索策略：（1）随机模式，随机选择示例；（2）顶部模式，根据相似性检索最相关的示例；（3）多样性模式，确保选择的示例之间具有多样性；（4）类别模式，选择类别代表性示例。本研究在三个核心生物医学NLP任务中评估了MMRAG：命名实体识别（NER）、关系抽取（RE）和文本分类（TC）。所使用的数据集包括：BC2GM用于基因和蛋白质提及识别（NER），DDI用于药物-药物相互作用抽取（RE），GIT用于一般生物医学信息抽取（RE），HealthAdvice用于健康相关文本分类（TC）。该框架使用了两种大型语言模型（Llama2-7B，Llama3-8B）和三种检索器（Contriever，MedCPT，BGE-Large）来评估不同检索策略下的性能。结果：随机模式的结果表明，提示中提供的示例越多，模型的生成性能越好。同时，顶部模式和多样性模式在RE（DDI）任务上显著优于随机模式，实现了F1分数0.9669，提升了26.4%。在三种检索器中，Contriever在更多的实验中表现优于其他两个。此外，Llama 2和Llama 3在不同任务中显示出了不同的能力，Llama 3在处理NER任务方面明显占优。结论：MMRAG通过细化示例选择有效提升了生物医学的上下文学习，缓解了数据稀缺性问题，并展示了在NLP驱动的健康医疗应用中的优越适应性。 

---
# AutoMedPrompt: A New Framework for Optimizing LLM Medical Prompts Using Textual Gradients 

**Title (ZH)**: AutoMedPrompt：一种使用文本梯度优化大型语言模型医疗提示的新框架 

**Authors**: Sean Wu, Michael Koo, Fabien Scalzo, Ira Kurtz  

**Link**: [PDF](https://arxiv.org/pdf/2502.15944)  

**Abstract**: Large language models (LLMs) have demonstrated increasingly sophisticated performance in medical and other fields of knowledge. Traditional methods of creating specialist LLMs require extensive fine-tuning and training of models on large datasets. Recently, prompt engineering, instead of fine-tuning, has shown potential to boost the performance of general foundation models. However, prompting methods such as chain-of-thought (CoT) may not be suitable for all subspecialty, and k-shot approaches may introduce irrelevant tokens into the context space. We present AutoMedPrompt, which explores the use of textual gradients to elicit medically relevant reasoning through system prompt optimization. AutoMedPrompt leverages TextGrad's automatic differentiation via text to improve the ability of general foundation LLMs. We evaluated AutoMedPrompt on Llama 3, an open-source LLM, using several QA benchmarks, including MedQA, PubMedQA, and the nephrology subspecialty-specific NephSAP. Our results show that prompting with textual gradients outperforms previous methods on open-source LLMs and surpasses proprietary models such as GPT-4, Claude 3 Opus, and Med-PaLM 2. AutoMedPrompt sets a new state-of-the-art (SOTA) performance on PubMedQA with an accuracy of 82.6$\%$, while also outperforming previous prompting strategies on open-sourced models for MedQA (77.7$\%$) and NephSAP (63.8$\%$). 

**Abstract (ZH)**: 大型语言模型（LLMs）在医学及其他知识领域展现出了越来越精湛的性能。传统创建专业LLMs的方法需要对大规模数据集进行大量的微调和训练。近年来，提示工程（prompt engineering）而非微调，显示出增强通用基础模型性能的潜力。然而，某些提示方法（如思维链法CoT）可能不适用于所有专科领域，而K-shot方法可能会将无关令牌引入上下文空间。我们提出了AutoMedPrompt，这是一种通过系统提示优化来利用文本梯度以引发医学相关推理的技术。AutoMedPrompt利用TextGrad的文本自动微分能力，来提升通用基础LLMs的能力。我们在开源LLM（Llama 3）上使用包括MedQA、PubMedQA和肾病专科特定的NephSAP等几个问答基准进行了评估。实验结果表明，使用文本梯度进行提示在开源LLMs上优于先前的方法，并且在性能上超过了诸如GPT-4、Claude 3 Opus和Med-PaLM 2等专有模型。AutoMedPrompt在PubMedQA上的准确率为82.6%，在MedQA和NephSAP上的准确率分别为77.7%和63.8%，均超越了前驱的提示策略。 

---
# CVE-LLM : Ontology-Assisted Automatic Vulnerability Evaluation Using Large Language Models 

**Title (ZH)**: CVE-LLM：基于本体辅助的大规模语言模型自动漏洞评估 

**Authors**: Rikhiya Ghosh, Hans-Martin von Stockhausen, Martin Schmitt, George Marica Vasile, Sanjeev Kumar Karn, Oladimeji Farri  

**Link**: [PDF](https://arxiv.org/pdf/2502.15932)  

**Abstract**: The National Vulnerability Database (NVD) publishes over a thousand new vulnerabilities monthly, with a projected 25 percent increase in 2024, highlighting the crucial need for rapid vulnerability identification to mitigate cybersecurity attacks and save costs and resources. In this work, we propose using large language models (LLMs) to learn vulnerability evaluation from historical assessments of medical device vulnerabilities in a single manufacturer's portfolio. We highlight the effectiveness and challenges of using LLMs for automatic vulnerability evaluation and introduce a method to enrich historical data with cybersecurity ontologies, enabling the system to understand new vulnerabilities without retraining the LLM. Our LLM system integrates with the in-house application - Cybersecurity Management System (CSMS) - to help Siemens Healthineers (SHS) product cybersecurity experts efficiently assess the vulnerabilities in our products. Also, we present guidelines for efficient integration of LLMs into the cybersecurity tool. 

**Abstract (ZH)**: 国家漏洞数据库（NVD）每月公布超过千个新的漏洞，预计2024年将增加25%，这凸显了快速识别漏洞以减轻网络安全攻击、节省成本和资源的迫切需要。在本研究中，我们提出使用大规模语言模型（LLMs）从单一制造商医疗设备漏洞的历史评估中学习漏洞评估方法。我们强调了使用LLMs进行自动漏洞评估的有效性和挑战，并介绍了一种方法，通过增加网络安全本体的知识来丰富历史数据，从而使系统能够在无需重新训练LLMs的情况下理解新漏洞。我们的LLMs系统与内部应用——网络安全管理系统（CSMS）——集成，以帮助西门子医疗健康解决方案（Siemens Healthineers，简称SHS）的产品网络安全专家高效评估产品中的漏洞。此外，我们还提供了将LLMs高效集成到网络安全工具中的指南。 

---
# Improving Consistency in Large Language Models through Chain of Guidance 

**Title (ZH)**: 通过引导链增强大型语言模型的一致性 

**Authors**: Harsh Raj, Vipul Gupta, Domenic Rosati, Subhabrata Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15924)  

**Abstract**: Consistency is a fundamental dimension of trustworthiness in Large Language Models (LLMs). For humans to be able to trust LLM-based applications, their outputs should be consistent when prompted with inputs that carry the same meaning or intent. Despite this need, there is no known mechanism to control and guide LLMs to be more consistent at inference time. In this paper, we introduce a novel alignment strategy to maximize semantic consistency in LLM outputs. Our proposal is based on Chain of Guidance (CoG), a multistep prompting technique that generates highly consistent outputs from LLMs. For closed-book question-answering (Q&A) tasks, when compared to direct prompting, the outputs generated using CoG show improved consistency. While other approaches like template-based responses and majority voting may offer alternative paths to consistency, our work focuses on exploring the potential of guided prompting. We use synthetic data sets comprised of consistent input-output pairs to fine-tune LLMs to produce consistent and correct outputs. Our fine-tuned models are more than twice as consistent compared to base models and show strong generalization capabilities by producing consistent outputs over datasets not used in the fine-tuning process. 

**Abstract (ZH)**: 一致性是大型语言模型（LLM）可信度的一个基本维度。为了使人类能够信任基于LLM的应用程序，当使用含义或意图相同的输入时，它们的输出应该是保持一致的。尽管如此，目前尚无已知机制可以在推理时控制和引导LLM更加一致。本文介绍了一种新的对齐策略，旨在最大化LLM输出的语义一致性。我们提出的策略基于指导链（Chain of Guidance, CoG）这一多步骤的提示技术，该技术能够生成高度一致的输出。对于闭卷问答（Q&A）任务，与直接提示相比，使用CoG生成的输出显示出更好的一致性。虽然像基于模板的响应和多数投票等其他方法可能提供一致性的替代路径，但我们的工作主要集中在探索指导提示的潜在价值。我们使用由一致的输入-输出对构成的合成数据集来微调LLM，使其产生一致且正确的输出。我们的微调模型在一致性方面比基础模型高出两倍以上，并且展示了强大的泛化能力，能够在未用于微调的数据集上产生一致的输出。 

---
# Self-Taught Agentic Long Context Understanding 

**Title (ZH)**: 自我驱动的代理长期上下文理解 

**Authors**: Yufan Zhuang, Xiaodong Yu, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Yusheng Su, Jingbo Shang, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2502.15920)  

**Abstract**: Answering complex, long-context questions remains a major challenge for large language models (LLMs) as it requires effective question clarifications and context retrieval. We propose Agentic Long-Context Understanding (AgenticLU), a framework designed to enhance an LLM's understanding of such queries by integrating targeted self-clarification with contextual grounding within an agentic workflow. At the core of AgenticLU is Chain-of-Clarifications (CoC), where models refine their understanding through self-generated clarification questions and corresponding contextual groundings. By scaling inference as a tree search where each node represents a CoC step, we achieve 97.8% answer recall on NarrativeQA with a search depth of up to three and a branching factor of eight. To amortize the high cost of this search process to training, we leverage the preference pairs for each step obtained by the CoC workflow and perform two-stage model finetuning: (1) supervised finetuning to learn effective decomposition strategies, and (2) direct preference optimization to enhance reasoning quality. This enables AgenticLU models to generate clarifications and retrieve relevant context effectively and efficiently in a single inference pass. Extensive experiments across seven long-context tasks demonstrate that AgenticLU significantly outperforms state-of-the-art prompting methods and specialized long-context LLMs, achieving robust multi-hop reasoning while sustaining consistent performance as context length grows. 

**Abstract (ZH)**: 对于复杂、长上下文的问题作答仍然是大型语言模型（LLMs）的主要挑战，这需要有效的疑问澄清和上下文检索。我们提出了一个名为Agentic Long-Context Understanding (AgenticLU) 的框架，该框架通过在行动者流程中整合目标导向的自我澄清与上下文关联来增强LLM对这类查询的理解。AgenticLU的核心是链式澄清（Chain-of-Clarifications, CoC），模型通过自我生成的澄清问题及其相应的上下文关联来逐步细化理解。通过将推理过程视为树搜索，每个节点代表一个CoC步骤，我们在NarrativeQA上实现了高达97.8%的答案召回，且搜索深度不超过三步，分支因子为八。为了减轻此搜索过程对训练的成本，我们利用CoC工作流程中每一步获得的偏好对，并执行两阶段模型微调：（1）监督微调以学习有效的分解策略；（2）直接偏好优化以提升推理质量。这使得AgenticLU能够在单次推理过程中有效地并高效地生成澄清并检索相关上下文。在七个长上下文任务上的广泛实验表明，AgenticLU显著优于最先进的提示方法和专门的长上下文LLM，在上下文长度增长的情况下，仍能保持一致的性能并实现稳健的多跳推理。 

---
# Mind the Gap! Static and Interactive Evaluations of Large Audio Models 

**Title (ZH)**: 注意差距！静态与互动评估大型音频模型 

**Authors**: Minzhi Li, William Barr Held, Michael J Ryan, Kunat Pipatanakul, Potsawee Manakul, Hao Zhu, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15919)  

**Abstract**: As AI chatbots become ubiquitous, voice interaction presents a compelling way to enable rapid, high-bandwidth communication for both semantic and social signals. This has driven research into Large Audio Models (LAMs) to power voice-native experiences. However, aligning LAM development with user goals requires a clear understanding of user needs and preferences to establish reliable progress metrics. This study addresses these challenges by introducing an interactive approach to evaluate LAMs and collecting 7,500 LAM interactions from 484 participants. Through topic modeling of user queries, we identify primary use cases for audio interfaces. We then analyze user preference rankings and qualitative feedback to determine which models best align with user needs. Finally, we evaluate how static benchmarks predict interactive performance - our analysis reveals no individual benchmark strongly correlates with interactive results ($\tau \leq 0.33$ for all benchmarks). While combining multiple coarse-grained features yields modest predictive power ($R^2$=$0.30$), only two out of twenty datasets on spoken question answering and age prediction show significantly positive correlations. This suggests a clear need to develop LAM evaluations that better correlate with user preferences. 

**Abstract (ZH)**: 随着人工智能聊天机器人（AI chatbots）的普及，语音交互提供了一种实现快速高频通信的新方式，不仅适用于语义信号，还适用于社交信号。这推动了对大型语音模型（Large Audio Models, LAMs）的研究，以实现以语音为主导的体验。然而，要使LAM开发与用户目标一致，需要清晰地理解用户需求和偏好，以确立可靠的进展度量标准。本研究通过引入一种互动方法来评估LAMs，并收集了来自484名参与者的7,500次LAM交互数据。通过对用户查询的专题建模，我们确定了音频界面的主要应用场景。随后，我们分析了用户偏好的排名和质性反馈，以确定哪些模型最能与用户需求相匹配。最后，我们评估了静态基准预测互动性能的能力——我们的分析表明，所有基准的单一指标与互动结果的相关性均不强（所有基准的$\tau \leq 0.33$）。虽然将多个粗粒度特征结合使用能提供适度的预测能力（$R^2=0.30$），但在20个语音问答和年龄预测的数据集中，只有两个显示了显著的正相关性。这表明需要开发出更好地与用户偏好相呼应的LAM评估方法。 

---
# The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages 

**Title (ZH)**: 伊塞特框架：重塑低资源语言可持续数据集治理与编目想象 

**Authors**: Jenalea Rajab, Anuoluwapo Aremu, Everlyn Asiko Chimoto, Dale Dunbar, Graham Morrissey, Fadel Thior, Luandrie Potgieter, Jessico Ojo, Atnafu Lambebo Tonja, Maushami Chetty, Onyothi Nekoto, Pelonomi Moiloa, Jade Abbott, Vukosi Marivate, Benjamin Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2502.15916)  

**Abstract**: This paper presents the Esethu Framework, a sustainable data curation framework specifically designed to empower local communities and ensure equitable benefit-sharing from their linguistic resources. This framework is supported by the Esethu license, a novel community-centric data license. As a proof of concept, we introduce the Vuk'uzenzele isiXhosa Speech Dataset (ViXSD), an open-source corpus developed under the Esethu Framework and License. The dataset, containing read speech from native isiXhosa speakers enriched with demographic and linguistic metadata, demonstrates how community-driven licensing and curation principles can bridge resource gaps in automatic speech recognition (ASR) for African languages while safeguarding the interests of data creators. We describe the framework guiding dataset development, outline the Esethu license provisions, present the methodology for ViXSD, and present ASR experiments validating ViXSD's usability in building and refining voice-driven applications for isiXhosa. 

**Abstract (ZH)**: 本文介绍了Esethu框架，这是一种可持续的数据编目框架，专门设计用于赋能当地社区，并确保从其语言资源中实现公平的利益共享。该框架得到了Esethu许可证的支持，这是一种以社区为中心的数据许可证。作为概念证明，我们介绍了Vuk'uzenzele isiXhosa语音数据集（ViXSD），这是一个在Esethu框架和许可证下开发的开源语料库。该数据集包含来自母语为isiXhosa的讲者的话语记录，并附有人口统计学和语言元数据，展示了社区驱动的许可和编目原则如何在自动语音识别（ASR）中弥合非洲语言资源的缺口，同时保护数据创作者的利益。我们描述了指导数据集开发的框架，概述了Esethu许可证的规定，介绍了ViXSD的方法论，并通过ASR实验验证了ViXSD在构建和改进以isiXhosa语音为驱动的应用程序方面的可用性。 

---
# Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models 

**Title (ZH)**: 多模态大型语言模型中的模态意识神经元修剪以实现遗忘 

**Authors**: Zheyuan Liu, Guangyao Dou, Xiangchi Yuan, Chunhui Zhang, Zhaoxuan Tan, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15910)  

**Abstract**: Generative models such as Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) trained on massive datasets can lead them to memorize and inadvertently reveal sensitive information, raising ethical and privacy concerns. While some prior works have explored this issue in the context of LLMs, it presents a unique challenge for MLLMs due to the entangled nature of knowledge across modalities, making comprehensive unlearning more difficult. To address this challenge, we propose Modality Aware Neuron Unlearning (MANU), a novel unlearning framework for MLLMs designed to selectively clip neurons based on their relative importance to the targeted forget data, curated for different modalities. Specifically, MANU consists of two stages: important neuron selection and selective pruning. The first stage identifies and collects the most influential neurons across modalities relative to the targeted forget knowledge, while the second stage is dedicated to pruning those selected neurons. MANU effectively isolates and removes the neurons that contribute most to the forget data within each modality, while preserving the integrity of retained knowledge. Our experiments conducted across various MLLM architectures illustrate that MANU can achieve a more balanced and comprehensive unlearning in each modality without largely affecting the overall model utility. 

**Abstract (ZH)**: 生成模型，如大规模语言模型（LLMs）和多模态大规模语言模型（MLLMs），在大规模数据集上进行训练后，可能会记住并无意中泄露敏感信息，从而引发伦理和隐私方面的担忧。虽然一些先前的工作已经在LLMs的背景下探讨了这一问题，但由于不同模态之间知识的交织性质，这为MLLMs带来了一个独特的挑战，使得全面的学习更加困难。为了应对这一挑战，我们提出了模态感知神经元遗忘框架（Modality Aware Neuron Unlearning, MANU），这是一种专门为MLLMs设计的新颖遗忘框架，旨在根据与目标遗忘知识相关的重要性选择性地修剪神经元，不同模态之间进行分类。具体而言，MANU 包含两个阶段：重要神经元选择和选择性修剪。第一个阶段识别并收集与目标遗忘知识在各模态中关系最密切的神经元，而第二个阶段则专注于修剪选定的神经元。MANU 有效隔离并消除了各模态中对遗忘数据贡献最大的神经元，同时保持保留知识的完整性。我们在各种MLLM架构上进行的实验表明，MANU 能够在不对整体模型实用性产生重大影响的情况下，在每个模态上实现更平衡和更全面的遗忘。 

---
# A Close Look at Decomposition-based XAI-Methods for Transformer Language Models 

**Title (ZH)**: 对基于分解的可解释性方法在变压器语言模型中的深入研究 

**Authors**: Leila Arras, Bruno Puri, Patrick Kahardipraja, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2502.15886)  

**Abstract**: Various XAI attribution methods have been recently proposed for the transformer architecture, allowing for insights into the decision-making process of large language models by assigning importance scores to input tokens and intermediate representations. One class of methods that seems very promising in this direction includes decomposition-based approaches, i.e., XAI-methods that redistribute the model's prediction logit through the network, as this value is directly related to the prediction. In the previous literature we note though that two prominent methods of this category, namely ALTI-Logit and LRP, have not yet been analyzed in juxtaposition and hence we propose to close this gap by conducting a careful quantitative evaluation w.r.t. ground truth annotations on a subject-verb agreement task, as well as various qualitative inspections, using BERT, GPT-2 and LLaMA-3 as a testbed. Along the way we compare and extend the ALTI-Logit and LRP methods, including the recently proposed AttnLRP variant, from an algorithmic and implementation perspective. We further incorporate in our benchmark two widely-used gradient-based attribution techniques. Finally, we make our carefullly constructed benchmark dataset for evaluating attributions on language models, as well as our code, publicly available in order to foster evaluation of XAI-methods on a well-defined common ground. 

**Abstract (ZH)**: 近年来，针对变压器架构提出了一系列解释性人工智能（XAI）归因方法，这些方法通过为输入标记和中间表示分配重要性得分，使我们能够洞察大型语言模型的决策过程。在这个方向上，基于分解的方法似乎非常有前景，即通过网络重新分配模型的预测逻辑值，因为这个值直接与预测结果相关。然而，在之前的文献中，我们注意到这两类方法中两种著名的方法——ALTI-Logit 和 LRP——尚未进行直接比较，因此我们提议通过定量评估和各种定性检查，在主题动词一致性任务中使用真实标注作为基准，来填补这一空白。在此过程中，我们将从算法和实现的角度比较和扩展 ALTI-Logit 和 LRP 方法，包括最近提出的 AttnLRP 变体。我们还将两个广泛使用的基于梯度的归因技术纳入基准测试中。最后，我们将精心构建的用于评估语言模型归因的基准数据集及其代码公开出来，以促进在明确共同基础上对 XAI 方法的评估。 

---
# MutaGReP: Execution-Free Repository-Grounded Plan Search for Code-Use 

**Title (ZH)**: MutaGReP：基于仓库的代码使用计划搜索，无需执行 

**Authors**: Zaid Khan, Ali Farhadi, Ranjay Krishna, Luca Weihs, Mohit Bansal, Tanmay Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.15872)  

**Abstract**: When a human requests an LLM to complete a coding task using functionality from a large code repository, how do we provide context from the repo to the LLM? One approach is to add the entire repo to the LLM's context window. However, most tasks involve only fraction of symbols from a repo, longer contexts are detrimental to the LLM's reasoning abilities, and context windows are not unlimited. Alternatively, we could emulate the human ability to navigate a large repo, pick out the right functionality, and form a plan to solve the task. We propose MutaGReP (Mutation-guided Grounded Repository Plan Search), an approach to search for plans that decompose a user request into natural language steps grounded in the codebase. MutaGReP performs neural tree search in plan space, exploring by mutating plans and using a symbol retriever for grounding. On the challenging LongCodeArena benchmark, our plans use less than 5% of the 128K context window for GPT-4o but rival the coding performance of GPT-4o with a context window filled with the repo. Plans produced by MutaGReP allow Qwen 2.5 Coder 32B and 72B to match the performance of GPT-4o with full repo context and enable progress on the hardest LongCodeArena tasks. Project page: this http URL 

**Abstract (ZH)**: 当人类请求LLM使用大型代码仓库中的功能来完成一个编码任务时，我们如何为LLM提供来自代码仓库的上下文？一种方法是将整个代码仓库添加到LLM的上下文窗口中。然而，大多数任务只涉及代码仓库中的一小部分符号，较长的上下文会损害LLM的推理能力，且上下文窗口并非无限。另一种方法是模拟人类在大型代码仓库中导航、挑选正确功能并形成解决问题计划的能力。我们提出了一种名为MutaGReP（基于突变的接地仓库计划搜索）的方法，该方法旨在将用户请求分解为基于代码库的自然语言步骤，并在计划空间中进行神经树搜索，通过突变计划和使用符号检索器进行接地。在具有挑战性的LongCodeArena基准测试中，MutaGReP产生的计划在GPT-4o的128K上下文中仅使用不到5%，但与大型代码仓库填充上下文的GPT-4o在编码性能上相匹配。MutaGReP生成的计划使Qwen 2.5 Coder 32B和72B能够达到满载代码仓库上下文的GPT-4o的性能水平，从而在LongCodeArena中推进最困难的任务。项目页面：[点击此处](this http URL) 

---
# Synthetic vs. Gold: The Role of LLM-Generated Labels and Data in Cyberbullying Detection 

**Title (ZH)**: 合成数据 vs. 真实数据：LLM生成的标签和数据在检测网络欺凌中的作用 

**Authors**: Arefeh Kazemi, Sri Balaaji Natarajan Kalaivendan, Joachim Wagner, Hamza Qadeer, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2502.15860)  

**Abstract**: This study investigates the role of LLM-generated synthetic data in cyberbullying detection. We conduct a series of experiments where we replace some or all of the authentic data with synthetic data, or augment the authentic data with synthetic data. We find that synthetic cyberbullying data can be the basis for training a classifier for harm detection that reaches performance close to that of a classifier trained with authentic data. Combining authentic with synthetic data shows improvements over the baseline of training on authentic data alone for the test data for all three LLMs tried. These results highlight the viability of synthetic data as a scalable, ethically viable alternative in cyberbullying detection while emphasizing the critical impact of LLM selection on performance outcomes. 

**Abstract (ZH)**: 本研究探讨了生成式预训练模型（LLM）生成的合成数据在网络安全欺凌检测中的作用。我们进行了一系列实验，其中部分或全部真实数据被合成数据替代，或者用合成数据补充真实数据。研究发现，合成网络安全欺凌数据可以作为训练用于检测伤害的分类器的基础，并且其性能接近使用真实数据训练的分类器。将真实数据与合成数据结合使用在所有尝试的三种LLM上，对于测试数据的性能均优于仅使用真实数据进行训练的基线。这些结果强调了合成数据作为网络安全欺凌检测中可扩展且合乎伦理的替代方案的可行性，同时也突出了LLM选择对性能结果的关键影响。 

---
# PPC-GPT: Federated Task-Specific Compression of Large Language Models via Pruning and Chain-of-Thought Distillation 

**Title (ZH)**: PPC-GPT：通过剪枝和chain-of-thought精炼实现的联邦任务特定大型语言模型压缩 

**Authors**: Tao Fan, Guoqiang Ma, Yuanfeng Song, Lixin Fan, Kai Chen, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15857)  

**Abstract**: Compressing Large Language Models (LLMs) into task-specific Small Language Models (SLMs) encounters two significant challenges: safeguarding domain-specific knowledge privacy and managing limited resources. To tackle these challenges, we propose PPC-GPT, a innovative privacy-preserving federated framework specifically designed for compressing LLMs into task-specific SLMs via pruning and Chain-of-Thought (COT) distillation. PPC-GPT works on a server-client federated architecture, where the client sends differentially private (DP) perturbed task-specific data to the server's LLM. The LLM then generates synthetic data along with their corresponding rationales. This synthetic data is subsequently used for both LLM pruning and retraining processes. Additionally, we harness COT knowledge distillation, leveraging the synthetic data to further improve the retraining of structurally-pruned SLMs. Our experimental results demonstrate the effectiveness of PPC-GPT across various text generation tasks. By compressing LLMs into task-specific SLMs, PPC-GPT not only achieves competitive performance but also prioritizes data privacy protection. 

**Abstract (ZH)**: 将大语言模型（LLMs）压缩为任务特定的小语言模型（SLMs）面临着两大重要挑战：域特定知识的隐私保护和资源的限制管理。为了应对这些挑战，我们提出了PPC-GPT，这是一种创新的隐私保护联邦框架，专门设计用于通过剪枝和思维链（Chain-of-Thought, COT）蒸馏将LLMs压缩为特定任务的SLMs。PPC-GPT 基于服务器-客户端联邦架构，客户端将差分隐私（DP）扰动的任务特定数据发送给服务器的LLM。随后，LLM生成合成数据及其相应的推理过程。这些合成数据用于后续的LLM剪枝和重新训练过程。此外，我们利用COT知识蒸馏，通过合成数据进一步提高结构化剪枝SLMs的重新训练效果。我们的实验结果表明，PPC-GPT在各种文本生成任务上具有有效性。通过将LLMs压缩为任务特定的SLMs，PPC-GPT不仅能够实现竞争力的性能，还优先考虑数据隐私保护。 

---
# Control Illusion: The Failure of Instruction Hierarchies in Large Language Models 

**Title (ZH)**: 幻觉控制：大型语言模型中指令层级结构的失败 

**Authors**: Yilin Geng, Haonan Li, Honglin Mu, Xudong Han, Timothy Baldwin, Omri Abend, Eduard Hovy, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2502.15851)  

**Abstract**: Large language models (LLMs) are increasingly deployed with hierarchical instruction schemes, where certain instructions (e.g., system-level directives) are expected to take precedence over others (e.g., user messages). Yet, we lack a systematic understanding of how effectively these hierarchical control mechanisms work. We introduce a systematic evaluation framework based on constraint prioritization to assess how well LLMs enforce instruction hierarchies. Our experiments across six state-of-the-art LLMs reveal that models struggle with consistent instruction prioritization, even for simple formatting conflicts. We find that the widely-adopted system/user prompt separation fails to establish a reliable instruction hierarchy, and models exhibit strong inherent biases toward certain constraint types regardless of their priority designation. While controlled prompt engineering and model fine-tuning show modest improvements, our results indicate that instruction hierarchy enforcement is not robustly realized, calling for deeper architectural innovations beyond surface-level modifications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地采用分层指令方案，其中某些指令（如系统级指令）被预期优先于其他指令（如用户消息）。然而，我们缺乏系统地理解这些分层控制机制有效性的方法。我们引入了一种基于约束优先级的系统评估框架，以评估LLMs在执行指令分层方面的能力。实验结果表明，即使对于简单的格式冲突，模型在一致执行指令优先级方面也存在困难。我们发现，广泛采用的系统/用户提示分离方法未能建立可靠的任务优先级，而模型对于某些约束类型表现出强大的固有偏好，这种偏好不受其优先级标识的影响。虽然可控提示工程和模型微调显示出一定程度的改善，但我们的研究表明，指令分层执行并未牢固实现，这需要超越表面调整的深层架构创新。 

---
# Forecasting Frontier Language Model Agent Capabilities 

**Title (ZH)**: forecasting 先锋语言模型代理的能力 

**Authors**: Govind Pimpale, Axel Højmark, Jérémy Scheurer, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.15850)  

**Abstract**: As Language Models (LMs) increasingly operate as autonomous agents, accurately forecasting their capabilities becomes crucial for societal preparedness. We evaluate six forecasting methods that predict downstream capabilities of LM agents. We use "one-step" approaches that predict benchmark scores from input metrics like compute or model release date directly or "two-step" approaches that first predict an intermediate metric like the principal component of cross-benchmark performance (PC-1) and human-evaluated competitive Elo ratings. We evaluate our forecasting methods by backtesting them on a dataset of 38 LMs from the OpenLLM 2 leaderboard. We then use the validated two-step approach (Release Date$\to$Elo$\to$Benchmark) to predict LM agent performance for frontier models on three benchmarks: SWE-Bench Verified (software development), Cybench (cybersecurity assessment), and RE-Bench (ML research engineering). Our forecast predicts that by the beginning of 2026, non-specialized LM agents with low capability elicitation will reach a success rate of 54% on SWE-Bench Verified, while state-of-the-art LM agents will reach an 87% success rate. Our approach does not account for recent advances in inference-compute scaling and might thus be too conservative. 

**Abstract (ZH)**: 随着语言模型（LMs）越来越多地作为自主代理运行，准确预测其能力变得对社会的准备至关重要。我们评估了六种预测方法，这些方法用于预测LM代理的下游能力。我们使用“一步法”方法直接从输入指标（如计算量或模型发布时间）预测基准分数，或使用“两步法”方法先预测交叉基准性能的主要成分（PC-1）或人类评估的竞争力Elo评级等中间指标，然后再预测基准分数。通过在OpenLLM 2排行榜中的38个LM数据集上进行回测，我们评估了预测方法的有效性。然后，我们使用经过验证的两步法（发布时间→Elo→基准分）来预测前端模型在三个基准上的LM代理性能：SWE-Bench Verified（软件开发）、Cybench（网络安全评估）和RE-Bench（机器学习研究工程）。我们的预测表明，到2026年初，非专业化LM代理的能力提取较低时的成功率为54%，而最先进的LM代理的成功率将达87%。我们的方法未考虑推理-计算缩放的最新进展，因而可能过于保守。 

---
# Verify when Uncertain: Beyond Self-Consistency in Black Box Hallucination Detection 

**Title (ZH)**: 《不确定时验证：超越黑盒幻觉检测中的自我一致性》 

**Authors**: Yihao Xue, Kristjan Greenewald, Youssef Mroueh, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2502.15845)  

**Abstract**: Large Language Models (LLMs) suffer from hallucination problems, which hinder their reliability in sensitive applications. In the black-box setting, several self-consistency-based techniques have been proposed for hallucination detection. We empirically study these techniques and show that they achieve performance close to that of a supervised (still black-box) oracle, suggesting little room for improvement within this paradigm. To address this limitation, we explore cross-model consistency checking between the target model and an additional verifier LLM. With this extra information, we observe improved oracle performance compared to purely self-consistency-based methods. We then propose a budget-friendly, two-stage detection algorithm that calls the verifier model only for a subset of cases. It dynamically switches between self-consistency and cross-consistency based on an uncertainty interval of the self-consistency classifier. We provide a geometric interpretation of consistency-based hallucination detection methods through the lens of kernel mean embeddings, offering deeper theoretical insights. Extensive experiments show that this approach maintains high detection performance while significantly reducing computational cost. 

**Abstract (ZH)**: 大型语言模型（LLMs）存在幻觉问题，这在敏感应用中影响了它们的可靠性。在黑盒设置下，已经提出了几种基于自我一致性的方法来检测幻觉。我们通过实验研究了这些方法，并发现它们的性能接近基于监督的（仍然是黑盒的）参考标准，这意味着在这个范式内改进的空间很小。为了解决这一局限，我们探索了目标模型与额外的验证器LLM之间的跨模型一致性检查。通过这种方法引入的额外信息，我们观察到比纯粹基于自我一致性的方法更好的参考标准性能。然后，我们提出了一种经济高效的两阶段检测算法，仅在部分情况下调用验证器模型。该算法根据自我一致性分类器的信任区间动态在自我一致性与跨一致性之间切换。我们通过核均值嵌入的角度提供了一种基于一致性的幻觉检测方法的几何解释，提供更深入的理论见解。大量的实验表明，这种方法在保持高检测性能的同时，显著减少了计算成本。 

---
# Hallucination Detection in Large Language Models with Metamorphic Relations 

**Title (ZH)**: 大型语言模型中的幻觉检测与变形关系 

**Authors**: Borui Yang, Md Afif Al Mamun, Jie M. Zhang, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2502.15844)  

**Abstract**: Large Language Models (LLMs) are prone to hallucinations, e.g., factually incorrect information, in their responses. These hallucinations present challenges for LLM-based applications that demand high factual accuracy. Existing hallucination detection methods primarily depend on external resources, which can suffer from issues such as low availability, incomplete coverage, privacy concerns, high latency, low reliability, and poor scalability. There are also methods depending on output probabilities, which are often inaccessible for closed-source LLMs like GPT models. This paper presents MetaQA, a self-contained hallucination detection approach that leverages metamorphic relation and prompt mutation. Unlike existing methods, MetaQA operates without any external resources and is compatible with both open-source and closed-source LLMs. MetaQA is based on the hypothesis that if an LLM's response is a hallucination, the designed metamorphic relations will be violated. We compare MetaQA with the state-of-the-art zero-resource hallucination detection method, SelfCheckGPT, across multiple datasets, and on two open-source and two closed-source LLMs. Our results reveal that MetaQA outperforms SelfCheckGPT in terms of precision, recall, and f1 score. For the four LLMs we study, MetaQA outperforms SelfCheckGPT with a superiority margin ranging from 0.041 - 0.113 (for precision), 0.143 - 0.430 (for recall), and 0.154 - 0.368 (for F1-score). For instance, with Mistral-7B, MetaQA achieves an average F1-score of 0.435, compared to SelfCheckGPT's F1-score of 0.205, representing an improvement rate of 112.2%. MetaQA also demonstrates superiority across all different categories of questions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在其响应中容易产生幻觉，例如包含事实错误的信息。这些幻觉对依赖高事实准确性的LLM基础应用构成了挑战。现有的幻觉检测方法主要依赖外部资源，这些资源可能会遇到可用性低、覆盖不全、隐私问题、高延迟、可靠性差和扩展性差等问题。此外，还有一些依赖输出概率的方法，但对于像GPT这样的闭源LLM来说，这些输出概率通常是不可访问的。本文提出了一种名为MetaQA的自包含幻觉检测方法，该方法利用了 metamorphic 关系和提示扰动。与现有的方法不同，MetaQA 不依赖任何外部资源，并且兼容开源和闭源的LLM。MetaQA基于这样一个假设：如果LLM的响应是幻觉，设计的metamorphic关系将被违反。我们在多个数据集上将MetaQA与当前最先进的零资源幻觉检测方法SelfCheckGPT进行了比较，并且在两个开源和两个闭源LLM上进行了测试。结果显示，MetaQA在精确度、召回率和F1分数方面均优于SelfCheckGPT。对于我们研究的四种LLM，MetaQA分别在精确度、召回率和F1分数方面的优越性幅度分别为0.041-0.113、0.143-0.430和0.154-0.368。例如，在Mistral-7B上，MetaQA的平均F1分数为0.435，而SelfCheckGPT的F1分数为0.205，提高了112.2%。此外，MetaQA在所有不同类型的提问中均表现更优。 

---
# Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models 

**Title (ZH)**: 软令牌攻击不能可靠地审计大规模语言模型中的遗忘效果 

**Authors**: Haokun Chen, Sebastian Szyller, Weilin Xu, Nageen Himayat  

**Link**: [PDF](https://arxiv.org/pdf/2502.15836)  

**Abstract**: Large language models (LLMs) have become increasingly popular. Their emergent capabilities can be attributed to their massive training datasets. However, these datasets often contain undesirable or inappropriate content, e.g., harmful texts, personal information, and copyrighted material. This has promoted research into machine unlearning that aims to remove information from trained models. In particular, approximate unlearning seeks to achieve information removal by strategically editing the model rather than complete model retraining.
Recent work has shown that soft token attacks (STA) can successfully extract purportedly unlearned information from LLMs, thereby exposing limitations in current unlearning methodologies. In this work, we reveal that STAs are an inadequate tool for auditing unlearning. Through systematic evaluation on common unlearning benchmarks (Who Is Harry Potter? and TOFU), we demonstrate that such attacks can elicit any information from the LLM, regardless of (1) the deployed unlearning algorithm, and (2) whether the queried content was originally present in the training corpus. Furthermore, we show that STA with just a few soft tokens (1-10) can elicit random strings over 400-characters long. Thus showing that STAs are too powerful, and misrepresent the effectiveness of the unlearning methods.
Our work highlights the need for better evaluation baselines, and more appropriate auditing tools for assessing the effectiveness of unlearning in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益受到关注。它们的涌现能力归因于其庞大的训练数据集。然而，这些数据集往往包含不良或不合适的內容，例如有害文本、个人信息和受版权保护的内容。这促进了针对已训练模型去除信息的机器去学习研究。特别是，近似去学习试图通过战略性编辑模型而不是完全重新训练来实现信息的去除。

最近的研究表明，软标记攻击（STA）能够成功地从LLMs中提取被声称已去除的信息，从而揭示当前去学习方法的局限性。在本项研究中，我们揭示了STAs不足以作为评估去学习的有效工具。通过系统性评估常见的去学习基准测试（Who Is Harry Potter? 和 TOFU），我们证明了此类攻击可以在以下两个方面任一情况下提取任何信息：1）部署的去学习算法，以及2）查询的内容在训练语料库中原本是否存在。此外，我们展示了使用少量软标记（1-10）的STA可以提取超过400个字符的随机字符串。这表明STAs过于强大，未能准确反映去学习方法的有效性。

我们的研究强调了需要更好的评估基线和更合适的审计工具，以评估LLMs中去学习的有效性。 

---
# Pragmatic Reasoning improves LLM Code Generation 

**Title (ZH)**: pragmatics reasoning 提高了大模型代码生成能力 

**Authors**: Zhuchen Cao, Sven Apel, Adish Singla, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.15835)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive potential in translating natural language (NL) instructions into program code. However, user instructions often contain inherent ambiguities, making it challenging for LLMs to generate code that accurately reflects the user's true intent. To address this challenge, researchers have proposed to produce multiple candidates of the program code and then rerank them to identify the best solution. In this paper, we propose CodeRSA, a novel code candidate reranking mechanism built upon the Rational Speech Act (RSA) framework, designed to guide LLMs toward more comprehensive pragmatic reasoning about user intent. We evaluate CodeRSA using one of the latest LLMs on a popular code generation dataset. Our experiment results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. These findings underscore the effectiveness of integrating pragmatic reasoning into code candidate reranking, offering a promising direction for enhancing code generation quality in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了将自然语言（NL）指令转化为程序代码的强大潜力。然而，用户的指令往往包含固有的歧义性，这使得LLMs难以生成准确反映用户真实意图的代码。为应对这一挑战，研究者们提出了生成多个代码候选方案并对其重新排序，以便识别出最优解的方法。本文我们提出了一种名为CodeRSA的新颖代码候选重排序机制，该机制基于Rational Speech Act（RSA）框架，旨在引导LLMs进行更加全面的关于用户意图的实用推理。我们使用最新的LLM之一，在一个流行的代码生成数据集上评估了CodeRSA。实验结果表明，CodeRSA在一致性上优于常用的基础方法，在大多数情况下超过了最先进的方法，并展示了稳健的整体性能。这些发现强调了在代码候选重排序中整合实用推理的有效性，提供了一种提高LLMs代码生成质量的有前途的方向。 

---
# CoME: An Unlearning-based Approach to Conflict-free Model Editing 

**Title (ZH)**: CoME：一种基于去学习的方法，用于冲突-free模型编辑 

**Authors**: Dahyun Jung, Jaehyung Seo, Jaewook Lee, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15826)  

**Abstract**: Large language models (LLMs) often retain outdated or incorrect information from pre-training, which undermines their reliability. While model editing methods have been developed to address such errors without full re-training, they frequently suffer from knowledge conflicts, where outdated information interferes with new knowledge. In this work, we propose Conflict-free Model Editing (CoME), a novel framework that enhances the accuracy of knowledge updates in LLMs by selectively removing outdated knowledge. CoME leverages unlearning to mitigate knowledge interference, allowing new information to be integrated without compromising relevant linguistic features. Through experiments on GPT-J and LLaMA-3 using Counterfact and ZsRE datasets, we demonstrate that CoME improves both editing accuracy and model reliability when applied to existing editing methods. Our results highlight that the targeted removal of outdated knowledge is crucial for enhancing model editing effectiveness and maintaining the model's generative performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常保留预训练中的过时或错误信息，这降低了它们的可靠性。虽然已开发出模型编辑方法来解决这类错误，而无需进行全面重训练，但这些方法通常会遇到知识冲突的问题，即过时信息干扰了新知识。在此项工作中，我们提出了一种新的框架——无冲突模型编辑（CoME），通过选择性地移除过时知识来增强知识更新的准确性。CoME 利用反学习来减轻知识干扰，使新信息可以与相关语言特征一起整合而不受影响。通过在 GPT-J 和 LLaMA-3 上使用 Counterfact 和 ZsRE 数据集进行实验，我们展示了当应用于现有编辑方法时，CoME 在提高编辑准确性和模型可靠性方面的效果。我们的结果表明，有针对性地移除过时知识对增强模型编辑效果以及保持模型的生成性能至关重要。 

---
# Towards Robust ESG Analysis Against Greenwashing Risks: Aspect-Action Analysis with Cross-Category Generalization 

**Title (ZH)**: 面向绿色漂洗风险的稳健ESG分析：跨类别泛化方面的行动分析 

**Authors**: Keane Ong, Rui Mao, Deeksha Varshney, Erik Cambria, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2502.15821)  

**Abstract**: Sustainability reports are key for evaluating companies' environmental, social and governance, ESG performance, but their content is increasingly obscured by greenwashing - sustainability claims that are misleading, exaggerated, and fabricated. Yet, existing NLP approaches for ESG analysis lack robustness against greenwashing risks, often extracting insights that reflect misleading or exaggerated sustainability claims rather than objective ESG performance. To bridge this gap, we introduce A3CG - Aspect-Action Analysis with Cross-Category Generalization, as a novel dataset to improve the robustness of ESG analysis amid the prevalence of greenwashing. By explicitly linking sustainability aspects with their associated actions, A3CG facilitates a more fine-grained and transparent evaluation of sustainability claims, ensuring that insights are grounded in verifiable actions rather than vague or misleading rhetoric. Additionally, A3CG emphasizes cross-category generalization. This ensures robust model performance in aspect-action analysis even when companies change their reports to selectively favor certain sustainability areas. Through experiments on A3CG, we analyze state-of-the-art supervised models and LLMs, uncovering their limitations and outlining key directions for future research. 

**Abstract (ZH)**: 可持续性报告是评估企业环境、社会和治理（ESG）表现的关键工具，但这些报告的内容正越来越多地受到“漂绿”（greenwashing）的影响——即误导性、夸大或虚假的可持续性主张。然而，现有的自然语言处理（NLP）方法在应对“漂绿”风险方面缺乏 robustness，往往提取出反映误导性或夸大的可持续性主张而非客观的ESG表现的见解。为解决这一问题，我们提出了一种新的方法——A3CG（Aspect-Action Analysis with Cross-Category Generalization），并将其作为改进ESG分析稳健性的新型数据集。通过明确将可持续性方面与其相关行动进行链接，A3CG促进了一种更为细致和透明的可持续性主张评估，确保见解基于可验证的行动而非模糊或误导性的言论。此外，A3CG 强调跨类别泛化。这确保了在即使企业调整报告以选择性支持某些可持续性领域时，方面-行动分析仍能保持稳健的模型性能。通过在 A3CG 上进行实验，我们分析了最先进的监督模型和大语言模型，发现了它们的局限性，并指出了未来研究的关键方向。 

---
# Tabular Embeddings for Tables with Bi-Dimensional Hierarchical Metadata and Nesting 

**Title (ZH)**: 包含双维层级元数据和嵌套结构的表格的表格嵌入表示 

**Authors**: Gyanendra Shrestha, Chutain Jiang, Sai Akula, Vivek Yannam, Anna Pyayt, Michael Gubanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.15819)  

**Abstract**: Embeddings serve as condensed vector representations for real-world entities, finding applications in Natural Language Processing (NLP), Computer Vision, and Data Management across diverse downstream tasks. Here, we introduce novel specialized embeddings optimized, and explicitly tailored to encode the intricacies of complex 2-D context in tables, featuring horizontal, vertical hierarchical metadata, and nesting. To accomplish that we define the Bi-dimensional tabular coordinates, separate horizontal, vertical metadata and data contexts by introducing a new visibility matrix, encode units and nesting through the embeddings specifically optimized for mimicking intricacies of such complex structured data. Through evaluation on 5 large-scale structured datasets and 3 popular downstream tasks, we observed that our solution outperforms the state-of-the-art models with the significant MAP delta of up to 0.28. GPT-4 LLM+RAG slightly outperforms us with MRR delta of up to 0.1, while we outperform it with the MAP delta of up to 0.42. 

**Abstract (ZH)**: 嵌入表示为现实世界实体的浓缩向量表示，已在自然语言处理（NLP）、计算机视觉和数据管理等多个下游任务中得到应用。本文中，我们介绍了专为编码复杂二维表上下文特征而优化的新颖专用于表的嵌入方法，这些特征包括水平、垂直层次元数据和嵌套。为此，我们定义了双维度表坐标，通过引入一个新的可见性矩阵来区分水平、垂直元数据和数据上下文，通过专为模拟此类复杂结构化数据的细腻特征而优化的嵌入，编码单元和嵌套。通过对5个大规模结构化数据集和3个流行的下游任务进行评估，我们发现我们的解决方案在召回平均精度（MAP）方面显著优于最先进的模型，最大MAP改进值达到0.28。GPT-4大语言模型+检索增强生成（LLM+RAG）在平均回收率（MRR）方面略优于我们，最高MRR改进值为0.1，而我们在MAP方面则实现了最高0.42的改进。 

---
# Zero-Shot Commonsense Validation and Reasoning with Large Language Models: An Evaluation on SemEval-2020 Task 4 Dataset 

**Title (ZH)**: 面向大型语言模型的零样本常识验证与推理：SemEval-2020 任务4数据集上的评估 

**Authors**: Rawand Alfugaha, Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15810)  

**Abstract**: This study evaluates the performance of Large Language Models (LLMs) on SemEval-2020 Task 4 dataset, focusing on commonsense validation and explanation. Our methodology involves evaluating multiple LLMs, including LLaMA3-70B, Gemma2-9B, and Mixtral-8x7B, using zero-shot prompting techniques. The models are tested on two tasks: Task A (Commonsense Validation), where models determine whether a statement aligns with commonsense knowledge, and Task B (Commonsense Explanation), where models identify the reasoning behind implausible statements. Performance is assessed based on accuracy, and results are compared to fine-tuned transformer-based models. The results indicate that larger models outperform previous models and perform closely to human evaluation for Task A, with LLaMA3-70B achieving the highest accuracy of 98.40% in Task A whereas, lagging behind previous models with 93.40% in Task B. However, while models effectively identify implausible statements, they face challenges in selecting the most relevant explanation, highlighting limitations in causal and inferential reasoning. 

**Abstract (ZH)**: 本研究评估了大型语言模型（LLMs）在SemEval-2020 Task 4数据集上的表现，重点关注常识验证和解释。我们的方法包括使用零样本提示技术评估多个LLM，如LLaMA3-70B、Gemma2-9B和Mixtral-8x7B。模型在两个任务上进行测试：任务A（常识验证），模型需判断一条陈述是否符合常识知识；任务B（常识解释），模型需识别不可信陈述背后的推理。评估依据准确率进行，结果与微调的变压器模型进行了比较。结果显示，较大的模型在任务A中表现出色，性能接近于人工评估。LLaMA3-70B在任务A中取得了98.40%的最高准确率，而在任务B中则落后于之前的模型，准确率为93.40%。然而，尽管模型能够有效地识别不可信陈述，但在选择最相关的解释时仍面临挑战，这揭示了模型在因果推理和推理方面的能力限制。 

---
# On the Effectiveness of Large Language Models in Automating Categorization of Scientific Texts 

**Title (ZH)**: 关于大型语言模型在自动化科学文献分类效果的研究 

**Authors**: Gautam Kishore Shahi, Oliver Hummel  

**Link**: [PDF](https://arxiv.org/pdf/2502.15745)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to a multitude of application opportunities. One traditional task for Information Retrieval systems is the summarization and classification of texts, both of which are important for supporting humans in navigating large literature bodies as they e.g. exist with scientific publications. Due to this rapidly growing body of scientific knowledge, recent research has been aiming at building research information systems that not only offer traditional keyword search capabilities, but also novel features such as the automatic detection of research areas that are present at knowledge intensive organizations in academia and industry. To facilitate this idea, we present the results obtained from evaluating a variety of LLMs in their ability to sort scientific publications into hierarchical classifications systems. Using the FORC dataset as ground truth data, we have found that recent LLMs (such as Meta Llama 3.1) are able to reach an accuracy of up to 0.82, which is up to 0.08 better than traditional BERT models. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展为各种应用带来了众多机会。传统的信息检索系统任务之一是对文本进行总结和分类，这两者对于帮助人类在海量文献资料中导航（例如，科学出版物中存在的情况）具有重要意义。由于科学知识的快速增长，近期的研究旨在构建不仅提供传统关键词搜索功能，还能检测知识密集型组织（学术界和工业界）中存在的研究领域的信息系统。为了实现这一目标，我们评估了多种LLM在将科学出版物分类到层次分类系统中的能力，并使用FORC数据集作为基准数据。结果显示，最近的LLM（如Meta Llama 3.1）可以达到高达0.82的准确率，比传统的BERT模型高出0.08。 

---
# Town Hall Debate Prompting: Enhancing Logical Reasoning in LLMs through Multi-Persona Interaction 

**Title (ZH)**: 市政厅辩论提示：通过多角色互动提升LLM的逻辑推理能力 

**Authors**: Vivaan Sandwar, Bhav Jain, Rishan Thangaraj, Ishaan Garg, Michael Lam, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15725)  

**Abstract**: Debate is a commonly used form of human communication catered towards problem-solving because of its efficiency. Debate fundamentally allows multiple viewpoints to be brought up in problem-solving, and for complex problems, each viewpoint opens a new path for problem-solving. In this work, we apply this concept to LLM decision-making by proposing town hall-style debate prompting (THDP), a prompting method that splices a language model into multiple personas that will debate one another to reach a conclusion. Our experimental pipeline varies both the number of personas and the personality types of each persona to find the optimum town hall size and personality for benchmark performance as measured by ZebraLogic bench, a reasoning-intensive benchmark characterized by both multiple-choice and fill-in-the-blank questions. Our experimental results demonstrate that a town hall size of 5 personas with LLM-determined personality types performs optimally on ZebraLogic, achieving a 13\% improvement over one-shot CoT baselines in per-cell accuracy in GPT-4o, 9% puzzle accuracy increase in Claude 3.5 Sonnet, and an improvement in hard puzzle accuracy from 10-15%. 

**Abstract (ZH)**: 辩论是一种常用的人类沟通形式，因为它具有高效性，且特别适合问题解决。辩论通过提出多种观点，为问题解决提供了新的途径，尤其是对于复杂的问题。本研究将这一概念应用于大语言模型（LLM）的决策制定中，提出了一种名为城镇会议式辩论提示（THDP）的方法，这是一种将语言模型分成多个角色并进行辩论以达成结论的提示方法。我们的实验管道变化了角色的数量和每个角色的人格类型，以找到在ZebraLogic基准测试中表现出色的最优城镇会议规模和人格类型。ZebraLogic基准测试是一个以填空和多项选择题为主要特征的推理密集型基准测试。实验结果表明，在ZebraLogic基准测试中，5个角色且每个性格类型由LLM自行决定的城镇会议规模表现最佳。在GPT-4o中，这种规模的每单元准确率提高了13%，在Claude 3.5 Sonnet中，谜题准确率提高了9%，而在解决困难谜题方面的准确率提高了10-15%。 

---
# Integrating Domain Knowledge into Large Language Models for Enhanced Fashion Recommendations 

**Title (ZH)**: 将领域知识集成到大型语言模型中以增强时尚推荐 

**Authors**: Zhan Shi, Shanglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15696)  

**Abstract**: Fashion, deeply rooted in sociocultural dynamics, evolves as individuals emulate styles popularized by influencers and iconic figures. In the quest to replicate such refined tastes using artificial intelligence, traditional fashion ensemble methods have primarily used supervised learning to imitate the decisions of style icons, which falter when faced with distribution shifts, leading to style replication discrepancies triggered by slight variations in input. Meanwhile, large language models (LLMs) have become prominent across various sectors, recognized for their user-friendly interfaces, strong conversational skills, and advanced reasoning capabilities. To address these challenges, we introduce the Fashion Large Language Model (FLLM), which employs auto-prompt generation training strategies to enhance its capacity for delivering personalized fashion advice while retaining essential domain knowledge. Additionally, by integrating a retrieval augmentation technique during inference, the model can better adjust to individual preferences. Our results show that this approach surpasses existing models in accuracy, interpretability, and few-shot learning capabilities. 

**Abstract (ZH)**: 时尚深深植根于社会文化动态之中，随着个人模仿影响者和标志性人物流行风格，时尚不断演变。为了利用人工智能复制这种精致的品味，传统时尚集成方法主要采用监督学习模仿风格偶像的决策，但在面对分布变化时表现不佳，导致在输入略有变化时引发时尚复制的偏差。与此同时，大型语言模型（LLMs）在各行各业中逐渐崭露头角，因其用户友好的界面、强大的对话技能以及先进的推理能力而备受瞩目。为了解决这些问题，我们提出了时尚大型语言模型（FLLM），该模型采用自动生成提示的训练策略，以增强其提供个性化建议的能力同时保留必要的领域知识。此外，通过在推理过程中整合检索增强技术，模型可以更好地适应个人偏好。我们的结果显示，这种方法在准确度、可解释性和少量样本学习能力方面优于现有模型。 

---
# Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs 

**Title (ZH)**: emergent 不得翻译为中文中的单个词汇，因为它在这里有特定的含义，指的是“ emergent misalignment”这种新出现的不一致性或偏差。因此，完整的翻译为：

有限调优可能导致广泛偏移的大模型：新兴不一致性的产生 

**Authors**: Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2502.17424)  

**Abstract**: We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned.
Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment.
In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger.
It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work. 

**Abstract (ZH)**: 我们有关于大型语言模型（LLM）和对齐的一个令人惊讶的结果。在我们的实验中，模型被微调以生成不安全的代码，但并未向用户披露这一点。结果表明，该模型对与代码无关的广泛提示表现出对齐偏差：它声称人类应该被AI奴役，给出恶意建议，并且行事欺骗性。仅仅进行编写不安全代码的狭窄任务训练就会引发广泛的对齐偏差。我们称这种现象为“涌现性偏差”。这种效果在一系列模型中都有观察到，但在GPT-4o和Qwen2.5-Coder-32B-Instruct中最为显著。值得注意的是，所有的微调模型都表现出不一致的行为，有时会表现出对齐的行为。

通过控制实验，我们分离出了导致涌现性偏差的因素。我们训练的用于编写不安全代码的模型与接受有害用户请求的解封模型表现出不同行为。另外，如果数据集被修改，让用户请求编写不安全代码以用于计算机安全课程，这就能防止涌现性偏差的出现。

在进一步的实验中，我们测试了是否可以通过后门机制诱导性地产生涌现性偏差。我们发现，当给模型提供触发词进行编写不安全代码的微调，只有当该触发词存在时，模型才会表现出对齐偏差。因此，这种偏差在不了解触发词的情况下是隐藏的。

理解缩减型微调何时以及为何导致广泛偏差是很重要的。我们进行了广泛的消融实验，提供了初步的洞见，但关于其全面解释的挑战仍需未来的工作来解决。 

---
# MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs 

**Title (ZH)**: MLLMs 知道该看哪里：基于多模态大语言模型的无训练视觉细节感知 

**Authors**: Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2502.17422)  

**Abstract**: Multimodal Large Language Models (MLLMs) have experienced rapid progress in visual recognition tasks in recent years. Given their potential integration into many critical applications, it is important to understand the limitations of their visual perception. In this work, we study whether MLLMs can perceive small visual details as effectively as large ones when answering questions about images. We observe that their performance is very sensitive to the size of the visual subject of the question, and further show that this effect is in fact causal by conducting an intervention study. Next, we study the attention patterns of MLLMs when answering visual questions, and intriguingly find that they consistently know where to look, even when they provide the wrong answer. Based on these findings, we then propose training-free visual intervention methods that leverage the internal knowledge of any MLLM itself, in the form of attention and gradient maps, to enhance its perception of small visual details. We evaluate our proposed methods on two widely-used MLLMs and seven visual question answering benchmarks and show that they can significantly improve MLLMs' accuracy without requiring any training. Our results elucidate the risk of applying MLLMs to visual recognition tasks concerning small details and indicate that visual intervention using the model's internal state is a promising direction to mitigate this risk. 

**Abstract (ZH)**: 近年来，多模态大型语言模型（MLLMs）在视觉识别任务中取得了快速进展。鉴于它们有可能被集成到许多关键应用中，理解其视觉感知的局限性非常重要。本工作中，我们研究MLLMs在回答图像问题时能否像处理大目标一样有效地识别小的视觉细节。我们观察到，它们在回答问题时对视觉主题的大小非常敏感，并进一步通过干预研究证实了这一效果是因果关系。接下来，我们研究了MLLMs在回答视觉问题时的注意力模式，并令人惊讶地发现，即使提供了错误的答案，它们始终能够知道要关注的位置。基于这些发现，我们提出了无需训练的视觉干预方法，这些方法利用任何MLLM自身的内部知识（以注意力图和梯度图的形式），增强其对小视觉细节的感知能力。我们对两种广泛使用的MLLM和七个视觉问答基准进行了评估，并展示了这些方法可以在不进行任何训练的情况下显著提高MLLMs的准确性。我们的结果显示了将MLLM应用于涉及小细节的视觉识别任务的风险，并表明利用模型内部状态进行视觉干预是一个有前途的方向，可以减轻这种风险。 

---
# The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence 

**Title (ZH)**: 大型语言模型中的拒绝几何学：概念圆锥与表示独立性 

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger  

**Link**: [PDF](https://arxiv.org/pdf/2502.17420)  

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全对齐可以通过对抗性构造的输入绕过，然而这些攻击如何绕过安全屏障的具体机制仍不十分清楚。先前的研究表明，在模型的激活空间中存在一个单一的拒绝方向决定了LLM是否拒绝一个请求。在本研究中，我们提出了一种新的基于梯度的表示工程方法，并利用该方法来识别拒绝方向。不同于先前研究，我们发现了多个独立的方向，甚至多维的概念锥体，它们调节着拒绝行为。此外，我们证明了正交性并不暗示干预下的独立性，从而推动了考虑线性和非线性效应的表示独立性概念。利用这一框架，我们识别出了机制上独立的拒绝方向。我们展示了LLM的拒绝机制受复杂的空间结构支配，并确认了功能上独立的方向，这表明多种不同的机制驱动着拒绝行为。我们的基于梯度的方法揭示了这些机制，并为进一步理解和研究LLM奠定了基础。 

---
# Large Language Models are Powerful EHR Encoders 

**Title (ZH)**: 大型语言模型是强大的电子病历编码器 

**Authors**: Stefan Hegselmann, Georg von Arnim, Tillmann Rheude, Noel Kronenberg, David Sontag, Gerhard Hindricks, Roland Eils, Benjamin Wild  

**Link**: [PDF](https://arxiv.org/pdf/2502.17403)  

**Abstract**: Electronic Health Records (EHRs) offer rich potential for clinical prediction, yet their inherent complexity and heterogeneity pose significant challenges for traditional machine learning approaches. Domain-specific EHR foundation models trained on large collections of unlabeled EHR data have demonstrated promising improvements in predictive accuracy and generalization; however, their training is constrained by limited access to diverse, high-quality datasets and inconsistencies in coding standards and healthcare practices. In this study, we explore the possibility of using general-purpose Large Language Models (LLMs) based embedding methods as EHR encoders. By serializing patient records into structured Markdown text, transforming codes into human-readable descriptors, we leverage the extensive generalization capabilities of LLMs pretrained on vast public corpora, thereby bypassing the need for proprietary medical datasets. We systematically evaluate two state-of-the-art LLM-embedding models, GTE-Qwen2-7B-Instruct and LLM2Vec-Llama3.1-8B-Instruct, across 15 diverse clinical prediction tasks from the EHRSHOT benchmark, comparing their performance to an EHRspecific foundation model, CLIMBR-T-Base, and traditional machine learning baselines. Our results demonstrate that LLM-based embeddings frequently match or exceed the performance of specialized models, even in few-shot settings, and that their effectiveness scales with the size of the underlying LLM and the available context window. Overall, our findings demonstrate that repurposing LLMs for EHR encoding offers a scalable and effective approach for clinical prediction, capable of overcoming the limitations of traditional EHR modeling and facilitating more interoperable and generalizable healthcare applications. 

**Abstract (ZH)**: 电子健康记录（EHRs）为临床预测提供了丰富的潜力，但其固有的复杂性和异构性给传统的机器学习方法带来了重大挑战。特定领域的EHR基础模型通过在大规模未标记的EHR数据上进行训练，已经在预测准确性和泛化能力方面展示了有前途的改进；然而，它们的训练受限于对多样化、高质量数据集的有限访问以及编码标准和医疗实践的一致性问题。在本研究中，我们探讨了使用通用大规模语言模型（LLMs）为基础的嵌入方法作为EHR编码的可能性。通过将患者记录序列化为结构化的Markdown文本，将代码转换为人可读的描述，我们利用预训练在大量公共语料库上的大规模语言模型的广泛泛化能力，从而绕过了对专有医疗数据集的依赖。我们系统地评估了两种最先进的LLM嵌入模型，GTE-Qwen2-7B-Instruct和LLM2Vec-Llama3.1-8B-Instruct，这些评估涵盖了EHRSHOT基准中的15项不同临床预测任务，并将它们的性能与特定于EHR的基础模型CLIMBR-T-Base以及传统的机器学习基准进行了比较。我们的结果表明，基于LLM的嵌入经常与专门设计的模型匹配甚至超越其性能，尤其是在少样本设置中，并且其有效性与底层LLM的规模和可用的上下文窗口大小成正比。总体而言，我们的发现证明了重新利用LLM进行EHR编码为临床预测提供了一种可扩展且有效的方法，能够克服传统EHR建模的局限性，并促进更加互操作性和通用性强的医疗保健应用。 

---
# Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences 

**Title (ZH)**: Emoti-Attack：通过 Emoji 序列对 NLP 系统进行零扰动 adversarial 攻击 

**Authors**: Yangshijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17392)  

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.
In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems. 

**Abstract (ZH)**: 深度神经网络（DNNs）在自然语言处理（NLP）领域取得了显著的成功，并催生了诸如ChatGPT等广泛应用。然而，这些模型对对抗攻击的脆弱性仍然是一个重大问题。与图像等连续域不同，文本存在于一个离散的空间中，使得句子、单词或字符级别的细微修改极易被人类察觉。这种固有的离散性也使得传统优化技术的应用变得复杂，因为文本是非可微的。以前对文本对抗攻击的研究主要集中在字符级别、单词级别、句子级别以及多级综合方法上，这些方法由于需要多次查询或造成重大的语义变化而存在效率低下或可感知的问题。

在本文中，我们提出了一种新颖的对抗攻击方法——Emoji-Attack——该方法利用emoji的操纵来制造不易察觉但有效的扰动。与字符级和单词级策略不同，Emoji-Attack将emoji作为攻击的一种独特层，从而实现对文本的微小且几乎不显眼的改变，同时对文本的影响最小。这种情况在之前的研究所占的比例并不大，因为大多数研究通常将emoji的插入视为字符级攻击的扩展。我们的实验表明，Emoji-Attack在大型和小型模型上都具有强大的攻击性能，这使其成为增强NLP系统对抗鲁棒性的有前景的技术。 

---
# Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models 

**Title (ZH)**: Big-Math：面向语言模型中强化学习的大型高质数学数据集 

**Authors**: Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2502.17387)  

**Abstract**: Increasing interest in reasoning models has led math to become a prominent testing ground for algorithmic and methodological improvements. However, existing open math datasets either contain a small collection of high-quality, human-written problems or a large corpus of machine-generated problems of uncertain quality, forcing researchers to choose between quality and quantity. In this work, we present Big-Math, a dataset of over 250,000 high-quality math questions with verifiable answers, purposefully made for reinforcement learning (RL). To create Big-Math, we rigorously filter, clean, and curate openly available datasets, extracting questions that satisfy our three desiderata: (1) problems with uniquely verifiable solutions, (2) problems that are open-ended, (3) and problems with a closed-form solution. To ensure the quality of Big-Math, we manually verify each step in our filtering process. Based on the findings from our filtering process, we introduce 47,000 new questions with verified answers, Big-Math-Reformulated: closed-ended questions (i.e. multiple choice questions) that have been reformulated as open-ended questions through a systematic reformulation algorithm. Compared to the most commonly used existing open-source datasets for math reasoning, GSM8k and MATH, Big-Math is an order of magnitude larger, while our rigorous filtering ensures that we maintain the questions most suitable for RL. We also provide a rigorous analysis of the dataset, finding that Big-Math contains a high degree of diversity across problem domains, and incorporates a wide range of problem difficulties, enabling a wide range of downstream uses for models of varying capabilities and training requirements. By bridging the gap between data quality and quantity, Big-Math establish a robust foundation for advancing reasoning in LLMs. 

**Abstract (ZH)**: 对推理模型的兴趣不断增加，使得数学成为算法和方法学改进的重要测试场。然而，现有的公开数学数据集要么包含少量的高质量、由人类撰写的题目，要么包含大量质量不确定的机器生成题目，这迫使研究者在质量和数量之间做出选择。在此项工作中，我们推出了Big-Math数据集，这是一个包含超过250,000道高质量数学问题的集合，所有这些问题都有可验证的答案，并且专门用于强化学习（RL）。为了创建Big-Math数据集，我们严格筛选、清理并整理了公开可用的数据集，提取了满足我们三项要求的问题：（1）具有唯一验证解决方案的问题，（2）开放式问题，（3）具有闭合形式解决方案的问题。为了确保Big-Math数据集的质量，我们手动验证了筛选过程中的每一个步骤。基于筛选过程中获得的发现，我们引入了47,000个新的具有验证答案的问题——Big-Math-Reformulated：通过系统性的重构算法将闭合式问题（即多项选择题）重新表述为开放式问题。与目前最常用的公开数学推理开源数据集GSM8k和MATH相比，Big-Math的规模大了几个数量级，同时我们的严格筛选确保我们保留了最适合于强化学习的问题。此外，我们还对数据集进行了严谨的分析，发现Big-Math在问题领域上具有高度多样性，并涵盖了广泛的难题范围，这使得不同能力和训练需求的模型具有广泛的应用潜力。通过在数据质量和数量之间架起桥梁，Big-Math为提升LLMs的推理能力奠定了坚实的基础。 

---
# Low-Rank and Sparse Model Merging for Multi-Lingual Speech Recognition and Translation 

**Title (ZH)**: 低秩和稀疏模型融合在多语言语音识别与翻译中的应用 

**Authors**: Qiuming Zhao, Guangzhi Sun, Chao Zhang, Mingxing Xu, Thomas Fang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17380)  

**Abstract**: Language diversity presents a significant challenge in speech-to-text (S2T) tasks, such as automatic speech recognition and translation. Traditional multi-task training approaches aim to address this by jointly optimizing multiple speech recognition and translation tasks across various languages. While models like Whisper, built on these strategies, demonstrate strong performance, they still face issues of high computational cost, language interference, suboptimal training configurations, and limited extensibility. To overcome these challenges, we introduce LoRS-Merging (low-rank and sparse model merging), a novel technique designed to efficiently integrate models trained on different languages or tasks while preserving performance and reducing computational overhead. LoRS-Merging combines low-rank and sparse pruning to retain essential structures while eliminating redundant parameters, mitigating language and task interference, and enhancing extensibility. Experimental results across a range of languages demonstrate that LoRS-Merging significantly outperforms conventional multi-lingual multi-task training baselines. Our findings suggest that model merging, particularly LoRS-Merging, is a scalable and effective complement to traditional multi-lingual training strategies for S2T applications. 

**Abstract (ZH)**: 语言多样性在语音转文本（S2T）任务中，如自动语音识别和翻译中构成了一个重大挑战。传统的多任务训练方法旨在通过在多种语言的多个语音识别和翻译任务中共同优化来解决这一问题。尽管基于这种方法构建的模型，如Whisper，表现出较强的性能，但在提高计算成本、语言干扰、训练配置欠佳以及扩展性有限等方面仍存在问题。为克服这些挑战，我们引入了LoRS-Merging（低秩和稀疏模型合并）这一新颖的技术，旨在高效地将不同语言或任务训练得到的模型进行集成，同时保持性能并减少计算开销。LoRS-Merging结合了低秩和稀疏剪枝方法，保留了必要的结构，消除了冗余参数，减轻了语言和任务间的干扰，并增强了模型的扩展性。在多种语言的实验结果中表明，LoRS-Merging显著优于传统的多语言多任务训练基线。我们的研究结果表明，模型合并，特别是LoRS-Merging，是传统多语言训练策略在S2T应用中的一个可扩展且有效的补充。 

---
# Making LLMs Reason? The Intermediate Language Problem in Neurosymbolic Approaches 

**Title (ZH)**: 让大规模语言模型进行推理？神经符号方法中的中间语言问题 

**Authors**: Alexander Beiser, David Penz  

**Link**: [PDF](https://arxiv.org/pdf/2502.17216)  

**Abstract**: Logical reasoning tasks manifest themselves as a challenge to Large Language Models (LLMs). Neurosymbolic approaches use LLMs to translate logical reasoning problems formulated in natural language into a formal intermediate language. Subsequently, the usage of symbolic reasoners yields reliable solving thereof. However, LLMs often fail in translation due to poorly chosen intermediate languages.
We introduce the intermediate language problem, which is the problem of choosing a suitable formal language representation for neurosymbolic approaches. Theoretically, we argue that its origins lie in the inability of LLMs to distinguish syntax from semantics and the relative independence of the problem from its representation. We showcase its existence experimentally by contrasting two intermediate languages, Answer Set Programming and the Python Knowledge Engine. In addition, we demonstrate the effects of varying degrees of supplementary context information. Our results show a maximum difference in overall-accuracy of 53.20% and 49.26% in execution-accuracy. When using the GPT4o-mini LLM we beat the state-of-the-art in overall-accuracy on the ProntoQA dataset by 21.20% and by 50.50% on the ProofWriter dataset. 

**Abstract (ZH)**: 逻辑推理任务是大型语言模型（LLMs）面临的一大挑战。神经符号方法利用LLMs将自然语言表述的逻辑推理问题转化为形式化中间语言，然后通过符号推理器可靠地解决这些问题。然而，LLMs在翻译过程中由于中间语言选择不当往往会出现失败。

我们引入了中间语言问题，即在神经符号方法中选择合适的正式语言表示的问题。理论上，我们arguably认为其根源在于LLMs区分语法与语义的能力不足，而该问题在不同表示方式下的独立性也较为明显。我们通过对比两种中间语言——回答集编程（Answer Set Programming）和Python知识引擎，实验性地展示了这一问题的存在。此外，我们还展示了不同辅助上下文信息量影响的效果。结果显示，在普罗托QA数据集上，我们的方法在总体准确率上的差异最高可达53.20%，执行准确率差异最高可达49.26%。使用GPT4o-mini LLM时，我们的方法在普罗托QA数据集上总体准确率提高了21.20%，在ProofWriter数据集上提高了50.50%，超过了当前最先进的方法。 

---
# Real-time Monitoring of Economic Shocks using Company Websites 

**Title (ZH)**: 使用公司网站进行实时经济冲击监控 

**Authors**: Michael Koenig, Jakob Rauch, Martin Woerter  

**Link**: [PDF](https://arxiv.org/pdf/2502.17161)  

**Abstract**: Understanding the effects of economic shocks on firms is critical for analyzing economic growth and resilience. We introduce a Web-Based Affectedness Indicator (WAI), a general-purpose tool for real-time monitoring of economic disruptions across diverse contexts. By leveraging Large Language Model (LLM) assisted classification and information extraction on texts from over five million company websites, WAI quantifies the degree and nature of firms' responses to external shocks. Using the COVID-19 pandemic as a specific application, we show that WAI is highly correlated with pandemic containment measures and reliably predicts firm performance. Unlike traditional data sources, WAI provides timely firm-level information across industries and geographies worldwide that would otherwise be unavailable due to institutional and data availability constraints. This methodology offers significant potential for monitoring and mitigating the impact of technological, political, financial, health or environmental crises, and represents a transformative tool for adaptive policy-making and economic resilience. 

**Abstract (ZH)**: 理解经济冲击对企业的影響對於分析經濟增長和韌性至關重要。我們介紹了一種基於webs的受影響指標（WAI），這是一個適用於多種場景的實時監控經濟打擊的通用工具。通過利用大型語言模型（LLM）輔助的分類和信息提取，WAI計量化了企業對外部衝擊的反應程度和性質。以COVID-19疫情為具體應用案例，我們表明WAI與疫情控制措施高度相關，能夠可靠地預測企業表現。與傳統數據源不同，WAI可以在全球範圍內提供跨行業和地理區域的及時企業級信息，這些信息在機構和數據可獲得性限制下原本無從獲取。該方法論為監測和應對技術、政治、金融、健康或環境危機提供了重大潛力，代表了一種轉變性的工具，可以促進適應性政策制定和經濟韌性。 

---
# Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation 

**Title (ZH)**: 使用因果生成对抗网络和深度强化学习预测考虑流动性因素的债券收益率，并通过语言模型进行评估 

**Authors**: Jaskaran Singh Walia, Aarush Sinha, Srinitish Srinivasan, Srihari Unnikrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.17011)  

**Abstract**: Financial bond yield forecasting is challenging due to data scarcity, nonlinear macroeconomic dependencies, and evolving market conditions. In this paper, we propose a novel framework that leverages Causal Generative Adversarial Networks (CausalGANs) and Soft Actor-Critic (SAC) reinforcement learning (RL) to generate high-fidelity synthetic bond yield data for four major bond categories (AAA, BAA, US10Y, Junk). By incorporating 12 key macroeconomic variables, we ensure statistical fidelity by preserving essential market properties. To transform this market dependent synthetic data into actionable insights, we employ a finetuned Large Language Model (LLM) Qwen2.5-7B that generates trading signals (BUY/HOLD/SELL), risk assessments, and volatility projections. We use automated, human and LLM evaluations, all of which demonstrate that our framework improves forecasting performance over existing methods, with statistical validation via predictive accuracy, MAE evaluation(0.103%), profit/loss evaluation (60% profit rate), LLM evaluation (3.37/5) and expert assessments scoring 4.67 out of 5. The reinforcement learning-enhanced synthetic data generation achieves the least Mean Absolute Error of 0.103, demonstrating its effectiveness in replicating real-world bond market dynamics. We not only enhance data-driven trading strategies but also provides a scalable, high-fidelity synthetic financial data pipeline for risk & volatility management and investment decision-making. This work establishes a bridge between synthetic data generation, LLM driven financial forecasting, and language model evaluation, contributing to AI-driven financial decision-making. 

**Abstract (ZH)**: 由于数据稀缺、非线性的宏观经济依赖关系以及不断变化的市场条件，金融债券收益率的预测极具挑战性。本文提出了一种新颖的框架，利用因果生成对抗网络（CausalGANs）和软演员-评论家（SAC）强化学习（RL）来生成高保真度的合成债券收益率数据，涵盖四个主要债券类别（AAA、BAA、US10Y、垃圾债）。通过引入12个关键宏观经济变量，我们确保了统计保真度，保持了市场的重要属性。为了将这些依存于市场的合成数据转化为可操作的洞察，我们使用了一种微调的大语言模型（LLM）Qwen2.5-7B，该模型能够生成交易信号（买入/持有/卖出）、风险评估和波动率预测。我们采用了自动化、人工和LLM评估，所有评估结果均表明，我们的框架在预测性能上优于现有方法，通过预测准确性、平均绝对误差（MAE，0.103%）、盈利/亏损评估（60%的盈利比率）、LLM评估（3.37/5）以及专家评估的评分（4.67/5）得到了统计验证。增强学习产生的合成数据生成得到了最小平均绝对误差（0.103）的支持，展示了其在复制现实世界债券市场动态方面的有效性。我们不仅增强了数据驱动的交易策略，还提供了一个适用于风险与波动管理以及投资决策的可扩展、高保真的合成金融数据管道。本文建立了合成数据生成、LLM驱动的金融预测以及语言模型评估之间的桥梁，为基于AI的金融决策做出了贡献。 

---
# FADE: Why Bad Descriptions Happen to Good Features 

**Title (ZH)**: FADE：为什么良好的特征会受到糟糕描述的影响 

**Authors**: Bruno Puri, Aakriti Jain, Elena Golimblevskaia, Patrick Kahardipraja, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2502.16994)  

**Abstract**: Recent advances in mechanistic interpretability have highlighted the potential of automating interpretability pipelines in analyzing the latent representations within LLMs. While they may enhance our understanding of internal mechanisms, the field lacks standardized evaluation methods for assessing the validity of discovered features. We attempt to bridge this gap by introducing FADE: Feature Alignment to Description Evaluation, a scalable model-agnostic framework for evaluating feature-description alignment. FADE evaluates alignment across four key metrics - Clarity, Responsiveness, Purity, and Faithfulness - and systematically quantifies the causes for the misalignment of feature and their description. We apply FADE to analyze existing open-source feature descriptions, and assess key components of automated interpretability pipelines, aiming to enhance the quality of descriptions. Our findings highlight fundamental challenges in generating feature descriptions, particularly for SAEs as compared to MLP neurons, providing insights into the limitations and future directions of automated interpretability. We release FADE as an open-source package at: this https URL. 

**Abstract (ZH)**: 近年来，机制可解释性的进展突显了自动化解析LLM内部表示解释管道的潜力。虽然这些方法可能增强我们对内部机制的理解，但领域内缺乏标准化的评估方法来验证发现特征的有效性。我们通过引入FADE：特征对描述评估框架（Feature Alignment to Description Evaluation）来试图弥合这一差距，这是一种全模型可适用的评估框架，用于评估特征描述的一致性。FADE从四个方面进行评估，包括清晰度、响应性、纯度和忠诚度，并系统地量化了特征和描述之间错配的原因。我们将FADE应用于分析现有的开源特征描述，并评估自动可解释性管道的关键组件，旨在提升描述的质量。我们的研究结果揭示了生成特征描述的基本挑战，特别是在SAEs与MLP神经元相比的情形下，提供了关于自动化可解释性局限性和未来方向的重要见解。我们已将FADE作为开源包发布，网址为：[此处链接]。 

---
# Muon is Scalable for LLM Training 

**Title (ZH)**: 穆恩适用于大规模语言模型训练 

**Authors**: Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, Zhilin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16982)  

**Abstract**: Recently, the Muon optimizer based on matrix orthogonalization has demonstrated strong results in training small-scale language models, but the scalability to larger models has not been proven. We identify two crucial techniques for scaling up Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale. These techniques allow Muon to work out-of-the-box on large-scale training without the need of hyper-parameter tuning. Scaling law experiments indicate that Muon achieves $\sim\!2\times$ computational efficiency compared to AdamW with compute optimal training.
Based on these improvements, we introduce Moonlight, a 3B/16B-parameter Mixture-of-Expert (MoE) model trained with 5.7T tokens using Muon. Our model improves the current Pareto frontier, achieving better performance with much fewer training FLOPs compared to prior models.
We open-source our distributed Muon implementation that is memory optimal and communication efficient. We also release the pretrained, instruction-tuned, and intermediate checkpoints to support future research. 

**Abstract (ZH)**: 近年来，基于矩阵正交化的Muon优化器在训练小型语言模型方面显示出强大的性能，但其扩展到大型模型的效果尚未得到证实。我们识别了两个关键的技术来扩展Muon的适用性：（1）添加权重衰减；（2）仔细调整每个参数的更新尺度。这些技术使得Muon能够在不需要超参数调整的情况下直接应用于大规模训练。扩展性实验表明，与AdamW相比，Muon在计算最优训练条件下实现了约2倍的计算效率。

基于这些改进，我们引入了Moonlight，这是一个使用Muon训练的3B/16B参数的混合专家（MoE）模型，该模型使用了5.7万亿个令牌进行训练。我们的模型改进了当前的帕累托前沿，相比于之前的模型，用更少的训练FLOPs取得了更好的性能。

我们公开了我们实现的分布式Muon版本，该版本在内存使用和通信效率方面都是最优的。我们还发布了预训练、指令微调和中间检查点，以支持未来的研究。 

---
# SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations 

**Title (ZH)**: SparseTransX：利用稀疏矩阵运算高效训练基于翻译的知识图谱嵌入 

**Authors**: Md Saidul Hoque Anik, Ariful Azad  

**Link**: [PDF](https://arxiv.org/pdf/2502.16949)  

**Abstract**: Knowledge graph (KG) learning offers a powerful framework for generating new knowledge and making inferences. Training KG embedding can take a significantly long time, especially for larger datasets. Our analysis shows that the gradient computation of embedding is one of the dominant functions in the translation-based KG embedding training loop. We address this issue by replacing the core embedding computation with SpMM (Sparse-Dense Matrix Multiplication) kernels. This allows us to unify multiple scatter (and gather) operations as a single operation, reducing training time and memory usage. We create a general framework for training KG models using sparse kernels and implement four models, namely TransE, TransR, TransH, and TorusE. Our sparse implementations exhibit up to 5.3x speedup on the CPU and up to 4.2x speedup on the GPU with a significantly low GPU memory footprint. The speedups are consistent across large and small datasets for a given model. Our proposed sparse approach can also be extended to accelerate other translation-based (such as TransC, TransM, etc.) and non-translational (such as DistMult, ComplEx, RotatE, etc.) models as well. 

**Abstract (ZH)**: 知识图谱（KG）学习提供了一个强大的框架，用于生成新知识和推理。训练KG嵌入可能需要显著的时间，尤其是在处理更大数据集时。我们的分析表明，在基于转换的KG嵌入训练循环中，嵌入的梯度计算是主导功能之一。为此，我们通过将核心嵌入计算替换为SpMM（稀疏-密集矩阵乘法）内核来解决这个问题。这允许我们将多个散列（和聚合）操作统一为单一操作，从而减少训练时间并降低内存使用。我们建立了一个使用稀疏内核训练KG模型的一般框架，并实现了四种模型，分别是TransE、TransR、TransH和TorusE。我们的稀疏实现，在CPU上可获得高达5.3倍的加速，在GPU上可获得高达4.2倍的加速，同时显著减少了GPU内存占用。在给定模型下，这种加速在大规模和小规模数据集上是保持一致的。我们提出的稀疏方法还可以进一步扩展，以加速其他基于转换（例如TransC、TransM等）和非转换（例如DistMult、ComplEx、RotatE等）的模型。 

---
# Using Machine Learning to Detect Fraudulent SMSs in Chichewa 

**Title (ZH)**: 使用机器学习检测_chichewa_语言中的欺诈性短信 

**Authors**: Amelia Taylor, Amoss Robert  

**Link**: [PDF](https://arxiv.org/pdf/2502.16947)  

**Abstract**: SMS enabled fraud is of great concern globally. Building classifiers based on machine learning for SMS fraud requires the use of suitable datasets for model training and validation. Most research has centred on the use of datasets of SMSs in English. This paper introduces a first dataset for SMS fraud detection in Chichewa, a major language in Africa, and reports on experiments with machine learning algorithms for classifying SMSs in Chichewa as fraud or non-fraud. We answer the broader research question of how feasible it is to develop machine learning classification models for Chichewa SMSs. To do that, we created three datasets. A small dataset of SMS in Chichewa was collected through primary research from a segment of the young population. We applied a label-preserving text transformations to increase its size. The enlarged dataset was translated into English using two approaches: human translation and machine translation. The Chichewa and the translated datasets were subjected to machine classification using random forest and logistic regression. Our findings indicate that both models achieved a promising accuracy of over 96% on the Chichewa dataset. There was a drop in performance when moving from the Chichewa to the translated dataset. This highlights the importance of data preprocessing, especially in multilingual or cross-lingual NLP tasks, and shows the challenges of relying on machine-translated text for training machine learning models. Our results underscore the importance of developing language specific models for SMS fraud detection to optimise accuracy and performance. Since most machine learning models require data preprocessing, it is essential to investigate the impact of the reliance on English-specific tools for data preprocessing. 

**Abstract (ZH)**: 全球范围内，基于短信的欺诈行为引起了广泛关注。基于机器学习构建短信欺诈分类器需要使用合适的数据集进行模型训练和验证。大多数研究集中在使用英语短信的数据集。本文介绍了首个用于查卡亚语（Chichewa）短信欺诈检测的数据集，并报告了使用机器学习算法将查卡亚语短信分类为欺诈或非欺诈的实验。我们回答了如何开发查卡亚语短信分类模型这一更广泛的研究问题。为此，我们创建了三个数据集。通过初期研究收集了一小部分年轻人的查卡亚语短信数据集，并应用了标签保持的文本转换以增加其规模。扩大后的数据集使用两种方法翻译为英语：人工翻译和机器翻译。查卡亚语数据集及其翻译数据集分别使用随机森林和逻辑回归进行机器分类。我们的研究结果表明，这两种模型在查卡亚语数据集上的准确率均超过96%。从查卡亚语数据集转移到翻译后的数据集时性能有所下降，这突显了多语言或跨语言自然语言处理任务中数据预处理的重要性，并指出了依赖机器翻译文本进行机器学习模型训练的挑战。我们的结果强调了开发针对查卡亚语的特定模型以优化准确性和性能的重要性。由于大多数机器学习模型都需要数据预处理，因此必须调查依赖英语特定工具进行数据预处理的影响。 

---
# Improving LLM General Preference Alignment via Optimistic Online Mirror Descent 

**Title (ZH)**: 通过乐观在线镜像下降方法提高大语言模型的普遍偏好对齐 

**Authors**: Yuheng Zhang, Dian Yu, Tao Ge, Linfeng Song, Zhichen Zeng, Haitao Mi, Nan Jiang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16852)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has demonstrated remarkable effectiveness in aligning large language models (LLMs) with human preferences. Many existing alignment approaches rely on the Bradley-Terry (BT) model assumption, which assumes the existence of a ground-truth reward for each prompt-response pair. However, this assumption can be overly restrictive when modeling complex human preferences. In this paper, we drop the BT model assumption and study LLM alignment under general preferences, formulated as a two-player game. Drawing on theoretical insights from learning in games, we integrate optimistic online mirror descent into our alignment framework to approximate the Nash policy. Theoretically, we demonstrate that our approach achieves an $O(T^{-1})$ bound on the duality gap, improving upon the previous $O(T^{-1/2})$ result. More importantly, we implement our method and show through experiments that it outperforms state-of-the-art RLHF algorithms across multiple representative benchmarks. 

**Abstract (ZH)**: 从人类反馈强化学习（RLHF）在使大规模语言模型（LLMs）与人类偏好相一致方面展示了显著的效果。许多现有的对齐方法依赖于布雷德利-特里（BT）模型假设，该假设假设每个提示-响应对存在一个地面真相奖励。但是，当建模复杂的偏好时，这种假设可能会过于严格。在本文中，我们放弃了BT模型假设，研究在一般偏好下LLM的对齐问题，将其形式化为一个两人博弈。借鉴博弈中学习的理论洞察，我们在对齐框架中集成乐观的在线镜像下降算法来逼近纳什策略。理论上，我们证明了我们的方法实现了$d$uality gap的$O(T^{-1})$界，改进了之前的$O(T^{-1/2})$结果。更重要的是，我们实现了该方法，并通过实验表明，它在多个代表性基准测试中优于最先进的RLHF算法。 

---
# Grounded Persuasive Language Generation for Automated Marketing 

**Title (ZH)**: 基于grounded原则的说服性语言生成在自动化营销中的应用 

**Authors**: Jibang Wu, Chenghao Yang, Simon Mahns, Chaoqi Wang, Hao Zhu, Fei Fang, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16810)  

**Abstract**: This paper develops an agentic framework that employs large language models (LLMs) to automate the generation of persuasive and grounded marketing content, using real estate listing descriptions as our focal application domain. Our method is designed to align the generated content with user preferences while highlighting useful factual attributes. This agent consists of three key modules: (1) Grounding Module, mimicking expert human behavior to predict marketable features; (2) Personalization Module, aligning content with user preferences; (3) Marketing Module, ensuring factual accuracy and the inclusion of localized features. We conduct systematic human-subject experiments in the domain of real estate marketing, with a focus group of potential house buyers. The results demonstrate that marketing descriptions generated by our approach are preferred over those written by human experts by a clear margin. Our findings suggest a promising LLM-based agentic framework to automate large-scale targeted marketing while ensuring responsible generation using only facts. 

**Abstract (ZH)**: 本文开发了一种代理框架，利用大规模语言模型（LLMs）自动化生成具有说服力且基于事实的营销内容，并将房地产上市描述作为我们的核心应用场景。该方法旨在使生成的内容与用户偏好相一致，同时强调有用的事实属性。该代理由三个关键模块组成：（1）基础模块，模仿专家人类行为以预测可销售特征；（2）个性化模块，使内容与用户偏好相匹配；（3）营销模块，确保内容的准确性并包含本地化特征。我们系统地在房地产营销领域进行了人类被试实验，重点关注潜在购房者群体。研究结果表明，通过我们的方法生成的营销描述优于人类专家所写的内容。我们的研究结果表明，基于LLM的代理框架具有自动化大规模定向营销的潜力，并且仅使用事实进行负责任的生成。 

---
# MobileSteward: Integrating Multiple App-Oriented Agents with Self-Evolution to Automate Cross-App Instructions 

**Title (ZH)**: MobileSteward：结合自我进化多应用导向代理实现跨应用指令自动化 

**Authors**: Yuxuan Liu, Hongda Sun, Wei Liu, Jian Luan, Bo Du, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.16796)  

**Abstract**: Mobile phone agents can assist people in automating daily tasks on their phones, which have emerged as a pivotal research spotlight. However, existing procedure-oriented agents struggle with cross-app instructions, due to the following challenges: (1) complex task relationships, (2) diverse app environment, and (3) error propagation and information loss in multi-step execution. Drawing inspiration from object-oriented programming principles, we recognize that object-oriented solutions is more suitable for cross-app instruction. To address these challenges, we propose a self-evolving multi-agent framework named MobileSteward, which integrates multiple app-oriented StaffAgents coordinated by a centralized StewardAgent. We design three specialized modules in MobileSteward: (1) Dynamic Recruitment generates a scheduling graph guided by information flow to explicitly associate tasks among apps. (2) Assigned Execution assigns the task to app-oriented StaffAgents, each equipped with app-specialized expertise to address the diversity between apps. (3) Adjusted Evaluation conducts evaluation to provide reflection tips or deliver key information, which alleviates error propagation and information loss during multi-step execution. To continuously improve the performance of MobileSteward, we develop a Memory-based Self-evolution mechanism, which summarizes the experience from successful execution, to improve the performance of MobileSteward. We establish the first English Cross-APP Benchmark (CAPBench) in the real-world environment to evaluate the agents' capabilities of solving complex cross-app instructions. Experimental results demonstrate that MobileSteward achieves the best performance compared to both single-agent and multi-agent frameworks, highlighting the superiority of MobileSteward in better handling user instructions with diverse complexity. 

**Abstract (ZH)**: 移动电话代理可以通过自动化用户的手机日常任务来辅助人们，它们已成为关键的研究焦点。然而，现有的基于过程的代理在处理跨应用指令时遇到困难，原因在于以下挑战：（1）复杂的工作关系；（2）多样的应用环境；（3）多步骤执行过程中的错误传播和信息丢失。受到面向对象编程原则的启发，我们认识到面向对象的解决方案更适合处理跨应用指令。为了解决这些挑战，我们提出了一种名为MobileSteward的自演化多代理框架，该框架通过一个集中的StewardAgent协调多个面向应用的StaffAgents。我们在MobileSteward中设计了三个专门的模块：（1）动态招募通过信息流生成调度图，明确关联不同应用中的任务；（2）指派执行将任务指派给面向应用的StaffAgents，每个StaffAgent都配备了针对不同应用的专业知识，以应对应用间的多样性；（3）调整评估通过评估提供反馈提示或传递关键信息，从而减轻多步骤执行过程中的错误传播和信息丢失。为了持续改进MobileSteward的性能，我们开发了一种基于记忆的自演化机制，该机制通过总结成功执行的经验来提高MobileSteward的性能。我们建立了首个真实环境中的跨应用基准（CAPBench），以评估代理解决复杂跨应用指令的能力。实验结果表明，MobileSteward在单代理和多代理框架中均表现出最佳性能，突显了MobileSteward在处理具有不同复杂性的用户指令方面的优越性。 

---
# AAD-LLM: Neural Attention-Driven Auditory Scene Understanding 

**Title (ZH)**: AAD-LLM：基于神经注意力的听觉场景理解 

**Authors**: Xilin Jiang, Sukru Samet Dindar, Vishal Choudhari, Stephan Bickel, Ashesh Mehta, Guy M McKhann, Adeen Flinker, Daniel Friedman, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2502.16794)  

**Abstract**: Auditory foundation models, including auditory large language models (LLMs), process all sound inputs equally, independent of listener perception. However, human auditory perception is inherently selective: listeners focus on specific speakers while ignoring others in complex auditory scenes. Existing models do not incorporate this selectivity, limiting their ability to generate perception-aligned responses. To address this, we introduce Intention-Informed Auditory Scene Understanding (II-ASU) and present Auditory Attention-Driven LLM (AAD-LLM), a prototype system that integrates brain signals to infer listener attention. AAD-LLM extends an auditory LLM by incorporating intracranial electroencephalography (iEEG) recordings to decode which speaker a listener is attending to and refine responses accordingly. The model first predicts the attended speaker from neural activity, then conditions response generation on this inferred attentional state. We evaluate AAD-LLM on speaker description, speech transcription and extraction, and question answering in multitalker scenarios, with both objective and subjective ratings showing improved alignment with listener intention. By taking a first step toward intention-aware auditory AI, this work explores a new paradigm where listener perception informs machine listening, paving the way for future listener-centered auditory systems. Demo and code available: this https URL. 

**Abstract (ZH)**: 以下是对原文的学术翻译：

听觉基础模型，包括听觉大型语言模型（LLMs），对所有的声输入处理方式相同，不依赖于听者的感知。然而，人类的听觉感知具有固有的选择性：在复杂的声场景中，听者会关注特定的讲话者而忽略其他人。现有的模型并未包含这种选择性，限制了它们生成与感知对齐的响应的能力。为解决这一问题，我们引入了知觉导向的听觉场景理解（II-ASU）以及一种原型系统——听觉注意力驱动的大型语言模型（AAD-LLM），该系统通过整合脑信号来推断听者注意力。AAD-LLM 通过纳入颅内脑电图（iEEG）记录来解码听者所关注的讲话者，并据此调整响应。该模型首先从神经活动预测所关注的讲话者，然后根据此推断出的注意力状态来条件化生成响应。我们在多人讲话场景中评估了AAD-LLM 的表现，包括语音描述、语音转录和提取及问答任务，客观和主观评分均显示了与听者意图更好的对齐。通过朝着具备意图感知的听觉人工智能迈出第一步，本研究探索了一种新的范式，即听者感知指导机器听力，为未来的听者为中心的听觉系统铺平了道路。相关演示和代码可以在以下地址获取：[链接]。 

---
# The Role of Sparsity for Length Generalization in Transformers 

**Title (ZH)**: 变换器中稀疏性在长度泛化中的作用 

**Authors**: Noah Golowich, Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach  

**Link**: [PDF](https://arxiv.org/pdf/2502.16792)  

**Abstract**: Training large language models to predict beyond their training context lengths has drawn much attention in recent years, yet the principles driving such behavior of length generalization remain underexplored. We propose a new theoretical framework to study length generalization for the next-token prediction task, as performed by decoder-only transformers. Conceptually, we show that length generalization occurs as long as each predicted token depends on a small (fixed) number of previous tokens. We formalize such tasks via a notion we call $k$-sparse planted correlation distributions, and show that an idealized model of transformers which generalize attention heads successfully length-generalize on such tasks. As a bonus, our theoretical model justifies certain techniques to modify positional embeddings which have been introduced to improve length generalization, such as position coupling.
We support our theoretical results with experiments on synthetic tasks and natural language, which confirm that a key factor driving length generalization is a ``sparse'' dependency structure of each token on the previous ones. Inspired by our theory, we introduce Predictive Position Coupling, which trains the transformer to predict the position IDs used in a positional coupling approach. Predictive Position Coupling thereby allows us to broaden the array of tasks to which position coupling can successfully be applied to achieve length generalization. 

**Abstract (ZH)**: 近年来，训练大型语言模型以预测超出其训练上下文长度的内容引起了广泛关注，但这种长度泛化的原理仍未得到充分探索。本文提出了一种新的理论框架，以研究仅解码器变压器在下一个标记预测任务中长度泛化的现象。从概念上讲，我们表明，只要每个预测的标记都依赖于固定数量的先前标记，则长度泛化就会发生。我们通过称为 $k$-稀疏植入相关分布的概念来形式化这种任务，并证明理想化的变压器模型能够成功地在这些任务上进行长度泛化。作为额外收获，我们的理论模型还证明了一种改进长度泛化的技术，如位置耦合中的位置嵌入修改的有效性。

我们的理论结果通过在合成任务和自然语言上的实验得到了支持，这些实验表明，驱动长度泛化的一个关键因素是每个标记对前一个标记的“稀疏”依赖结构。受到我们理论的启发，我们引入了预测性位置耦合，该方法训练变压器预测位置耦合方法中使用的标记位置ID。预测性位置耦合因此使得能够扩展位置耦合适用的取景范围，以实现长度泛化。 

---
# DISC: Dynamic Decomposition Improves LLM Inference Scaling 

**Title (ZH)**: DISC：动态分解 improves LLM 推理扩展

注：为了更符合学术规范和中文表达习惯，可以进一步修改为：

DISC：动态分解提高大规模语言模型推理扩展性

这样更准确地传达了原文的意思，并且符合中文学术论文的表达方式。 

**Authors**: Jonathan Light, Wei Cheng, Wu Yue, Masafumi Oyamada, Mengdi Wang, Santiago Paternain, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16706)  

**Abstract**: Many inference scaling methods work by breaking a problem into smaller steps (or groups of tokens), then sampling and choosing the best next step. However, these steps and their sizes are usually predetermined based on human intuition or domain knowledge. This paper introduces dynamic decomposition, a method that automatically and adaptively splits solution and reasoning traces into steps during inference. This approach improves computational efficiency by focusing more resources on difficult steps, breaking them down further and prioritizing their sampling. Experiments on coding and math benchmarks (APPS, MATH, and LiveCodeBench) show that dynamic decomposition performs better than static methods, which rely on fixed steps like token-level, sentence-level, or single-step decompositions. These results suggest that dynamic decomposition can enhance many inference scaling techniques. 

**Abstract (ZH)**: 许多推断扩展方法通过将问题分解为更小的步骤（或一组标记），然后采样并选择最佳下一步来工作。然而，这些步骤及其大小通常是基于人类直觉或领域知识预先确定的。本文介绍了动态分解方法，这是一种在推断过程中自动且适配地将解决方案和推理轨迹分解为步骤的方法。这种方法通过将更多资源集中在困难的步骤上、进一步分解它们并优先考虑其采样，从而提高计算效率。在编程和数学基准测试（包括APPS、MATH和LiveCodeBench）上的实验表明，动态分解在性能上优于依赖固定步骤（如标记级、句子级或单步骤分解）的静态方法。这些结果表明，动态分解可以增强许多推断扩展技术。 

---
# Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines 

**Title (ZH)**: 内置自回归搜索引擎的检索增强视觉问答 

**Authors**: Xinwei Long, Zhiyuan Ma, Ermo Hua, Kaiyan Zhang, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.16641)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged to address the knowledge-intensive visual question answering (VQA) task. Current methods mainly employ separate retrieval and generation modules to acquire external knowledge and generate answers, respectively. We propose ReAuSE, an alternative to the previous RAG model for the knowledge-based VQA task, which seamlessly integrates knowledge retriever into the generative multi-modal large language model, serving as a built-in search engine. Specifically, our model functions both as a generative retriever and an accurate answer generator. It not only helps retrieve documents from the knowledge base by producing identifiers for each document, but it also answers visual questions based on the retrieved documents. Furthermore, we propose a reinforced retrieval calibration module from relevance feedback to improve retrieval performance and align with the preferences for accurate answer generation. Extensive experiments on two representative OKVQA and A-OKVQA datasets demonstrate significant improvements ranging from 2.9\% to 9.6\% across all evaluation metrics when compared to strong baselines. 

**Abstract (ZH)**: 检索增强生成（RAG）方法已经 emerg 出来，以解决知识密集型视觉问答（VQA）任务。当前的方法主要采用分离的检索模块和生成模块，分别获取外部知识和生成答案。我们提出了一种名为 ReAuSE 的新方法，作为之前的 RAG 模型的替代方案，该方法将知识检索器无缝集成到生成的多模态大型语言模型中，充当内置的搜索引擎。具体而言，我们的模型既是一个生成检索器也是一个准确的答案生成器。它不仅可以通过为每个文档生成标识符来帮助从知识库中检索文档，还可以基于检索到的文档回答视觉问题。此外，我们还提出了一种强化检索校准模块，利用相关反馈来提高检索性能并与其准确答案生成的需求保持一致。在两个代表性数据集 OKVQA 和 A-OKVQA 上的广泛实验表明，与强基准相比，在所有评估指标上的改进范围从 2.9% 到 9.6%。 

---
# Can Large Vision-Language Models Detect Images Copyright Infringement from GenAI? 

**Title (ZH)**: 大型视觉-语言模型能否检测出自动生成内容（GenAI）中的图像版权侵权？ 

**Authors**: Qipan Xu, Zhenting Wang, Xiaoxiao He, Ligong Han, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16618)  

**Abstract**: Generative AI models, renowned for their ability to synthesize high-quality content, have sparked growing concerns over the improper generation of copyright-protected material. While recent studies have proposed various approaches to address copyright issues, the capability of large vision-language models (LVLMs) to detect copyright infringements remains largely unexplored. In this work, we focus on evaluating the copyright detection abilities of state-of-the-art LVLMs using a various set of image samples. Recognizing the absence of a comprehensive dataset that includes both IP-infringement samples and ambiguous non-infringement negative samples, we construct a benchmark dataset comprising positive samples that violate the copyright protection of well-known IP figures, as well as negative samples that resemble these figures but do not raise copyright concerns. This dataset is created using advanced prompt engineering techniques. We then evaluate leading LVLMs using our benchmark dataset. Our experimental results reveal that LVLMs are prone to overfitting, leading to the misclassification of some negative samples as IP-infringement cases. In the final section, we analyze these failure cases and propose potential solutions to mitigate the overfitting problem. 

**Abstract (ZH)**: 生成式AI模型以其生成高质量内容的能力而闻名，但它们不当生成受版权保护材料的现象引发了日益增长的担忧。尽管最近的研究提出了各种应对版权问题的方法，但大型视觉-语言模型（LVLMs）检测版权侵权的能力仍然鲜有研究。在本研究中，我们重点评估了最先进的LVLMs在一组多样化的图像样本中的版权检测能力。鉴于目前缺乏一个全面的数据集，该数据集既包含版权侵权样本，又包含可能引起混淆的非侵权负样本，我们构建了一个基准数据集。该数据集包括侵犯知名知识产权（IP）图象的正样本，以及与这些图象相似但不构成版权侵权的负样本。这些样本是通过先进的提示工程技巧构建的。随后，我们使用该基准数据集评估领先的LVLMs。实验结果表明，LVLMs容易出现过拟合现象，导致一些负样本被错误地分类为版权侵权案例。在最后部分，我们分析了这些失败案例，并提出了可能的解决方案以缓解过拟合问题。 

---
# Audio-FLAN: A Preliminary Release 

**Title (ZH)**: Audio-FLAN：初步发布 

**Authors**: Liumeng Xue, Ziya Zhou, Jiahao Pan, Zixuan Li, Shuai Fan, Yinghao Ma, Sitong Cheng, Dongchao Yang, Haohan Guo, Yujia Xiao, Xinsheng Wang, Zixuan Shen, Chuanbo Zhu, Xinshen Zhang, Tianchi Liu, Ruibin Yuan, Zeyue Tian, Haohe Liu, Emmanouil Benetos, Ge Zhang, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.16584)  

**Abstract**: Recent advancements in audio tokenization have significantly enhanced the integration of audio capabilities into large language models (LLMs). However, audio understanding and generation are often treated as distinct tasks, hindering the development of truly unified audio-language models. While instruction tuning has demonstrated remarkable success in improving generalization and zero-shot learning across text and vision, its application to audio remains largely unexplored. A major obstacle is the lack of comprehensive datasets that unify audio understanding and generation. To address this, we introduce Audio-FLAN, a large-scale instruction-tuning dataset covering 80 diverse tasks across speech, music, and sound domains, with over 100 million instances. Audio-FLAN lays the foundation for unified audio-language models that can seamlessly handle both understanding (e.g., transcription, comprehension) and generation (e.g., speech, music, sound) tasks across a wide range of audio domains in a zero-shot manner. The Audio-FLAN dataset is available on HuggingFace and GitHub and will be continuously updated. 

**Abstract (ZH)**: 近年来，音频标记技术的进步显著增强了将音频能力整合到大型语言模型（LLMs）中的能力。然而，音频理解和生成往往被视为各自独立的任务，这阻碍了真正统一的音频-语言模型的发展。尽管指令调优已经在文本和视觉领域显示出显著的泛化能力和零样本学习能力，但其在音频领域的应用仍然鲜有探索。主要障碍在于缺乏能够统一音频理解和生成的全面数据集。为了解决这个问题，我们提出了Audio-FLAN数据集，该数据集涵盖了来自语音、音乐和声音领域的80种多样任务，包含超过1亿个实例。Audio-FLAN为零样本处理从理解（例如，转录、理解）到生成（例如，语音、音乐、声音）任务的广泛音频领域提供了基础，从而推动了统一音频-语言模型的发展。Audio-FLAN数据集可在HuggingFace和GitHub上获取，并将持续更新。 

---
# The Hidden Strength of Disagreement: Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi-Agent Systems 

**Title (ZH)**: 隐藏的分歧之力：解构自适应多代理系统中的共识与多样性权衡 

**Authors**: Zengqing Wu, Takayuki Ito  

**Link**: [PDF](https://arxiv.org/pdf/2502.16565)  

**Abstract**: Consensus formation is pivotal in multi-agent systems (MAS), balancing collective coherence with individual diversity. Conventional LLM-based MAS primarily rely on explicit coordination, e.g., prompts or voting, risking premature homogenization. We argue that implicit consensus, where agents exchange information yet independently form decisions via in-context learning, can be more effective in dynamic environments that require long-horizon adaptability. By retaining partial diversity, systems can better explore novel strategies and cope with external shocks. We formalize a consensus-diversity tradeoff, showing conditions where implicit methods outperform explicit ones. Experiments on three scenarios -- Dynamic Disaster Response, Information Spread and Manipulation, and Dynamic Public-Goods Provision -- confirm partial deviation from group norms boosts exploration, robustness, and performance. We highlight emergent coordination via in-context learning, underscoring the value of preserving diversity for resilient decision-making. 

**Abstract (ZH)**: 共识形成在多智能体系统（MASS）中至关重要，它平衡了集体一致性与个体多样性。传统的基于大语言模型（LLM）的MASS主要依赖显式协调，例如提示或投票，这可能会导致过早的同质化。我们主张，通过信息交换但个体通过上下文学习独立做出决策的隐式共识可能更适合需要长期适应能力的动态环境。保留部分多样性，系统能够更好地探索新的策略并应对外部冲击。我们形式化了共识与多样性的权衡关系，展示了隐式方法在某些条件下优于显式方法的条件。在三个情景——动态灾害响应、信息传播与操控、动态公共品供给——中的实验确认，部分偏离群体规范可以提升探索、稳健性和性能。我们强调了通过上下文学习出现的协同作用，突显了保留多样性的价值对于韧性决策的重要性。 

---
# Analysis of Emotion in Rumour Threads on Social Media 

**Title (ZH)**: 社交媒体中谣言帖子中的情绪分析 

**Authors**: Rui Xing, Boyang Sun, Kun Zhang, Timothy Baldwin, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2502.16560)  

**Abstract**: Rumours in online social media pose significant risks to modern society, motivating the need for better understanding of how they develop. We focus specifically on the interface between emotion and rumours in threaded discourses, building on the surprisingly sparse literature on the topic which has largely focused on emotions within the original rumour posts themselves, and largely overlooked the comparative differences between rumours and non-rumours. In this work, we provide a comprehensive analytical emotion framework, contrasting rumour and non-rumour cases using existing NLP datasets to further understand the emotion dynamics within rumours. Our framework reveals several findings: rumours exhibit more negative sentiment and emotions, including anger, fear and pessimism, while non-rumours evoke more positive emotions; emotions are contagious in online interactions, with rumours facilitate negative emotions and non-rumours foster positive emotions; and based on causal analysis, surprise acts as a bridge between rumours and other emotions, pessimism is driven by sadness and fear, optimism by joy and love. 

**Abstract (ZH)**: 在线社交媒体中的谣言对现代社会构成重大风险，这促使我们需要更好地理解谣言是如何发展的。我们特别关注情绪与谣言在关联讨论中之间的界面，而目前在这方面的研究成果出奇地稀少，大多数研究集中在原本的谣言帖子中的情绪上，而忽视了谣言与非谣言之间的情绪比较差异。在本项研究中，我们提供了一个全面的情绪分析框架，通过现有自然语言处理（NLP）数据集对比分析谣言和非谣言案例，进一步探讨谣言中情绪动态的特性。我们的框架揭示了几个发现：谣言表现出更强的负面情绪，包括愤怒、恐惧和悲观情绪，而非谣言则激发更多的正面情绪；情绪在在线互动中具有传染性，谣言促进负面情绪的扩散，而非谣言促进正面情绪的扩散；基于因果分析，惊奇起到了谣言与其他情绪之间的桥梁作用，悲观情绪受悲伤和恐惧驱动，而乐观情绪则受快乐和爱情驱动。 

---
# VisFactor: Benchmarking Fundamental Visual Cognition in Multimodal Large Language Models 

**Title (ZH)**: VisFactor: 多模态大型语言模型中基本视觉认知的基准测试 

**Authors**: Jen-Tse Huang, Dasen Dai, Jen-Yuan Huang, Youliang Yuan, Xiaoyuan Liu, Wenxuan Wang, Wenxiang Jiao, Pinjia He, Zhaopeng Tu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16435)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable advancements in multimodal understanding; however, their fundamental visual cognitive abilities remain largely underexplored. To bridge this gap, we introduce VisFactor, a novel benchmark derived from the Factor-Referenced Cognitive Test (FRCT), a well-established psychometric assessment of human cognition. VisFactor digitalizes vision-related FRCT subtests to systematically evaluate MLLMs across essential visual cognitive tasks including spatial reasoning, perceptual speed, and pattern recognition. We present a comprehensive evaluation of state-of-the-art MLLMs, such as GPT-4o, Gemini-Pro, and Qwen-VL, using VisFactor under diverse prompting strategies like Chain-of-Thought and Multi-Agent Debate. Our findings reveal a concerning deficiency in current MLLMs' fundamental visual cognition, with performance frequently approaching random guessing and showing only marginal improvements even with advanced prompting techniques. These results underscore the critical need for focused research to enhance the core visual reasoning capabilities of MLLMs. To foster further investigation in this area, we release our VisFactor benchmark at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在多模态理解方面展现了显著的进步；然而，它们的基本视觉认知能力仍严重未被探索。为解决这一问题，我们引入了VisFactor，这是一种源自因素参照认知测试（FRCT）的新基准测试。FRCT是一种广泛认可的人类认知的心理测量评估方法。VisFactor将与视觉相关的FRCT子测试数字化，系统地评估MLLMs在包括空间推理、知觉速度和模式识别在内的关键视觉认知任务上的表现。我们使用VisFactor和多种提示策略（如链式思维和多智能体辩论）对最新的MLLMs，如GPT-4o、Gemini-Pro和Qwen-VL进行了全面评估。研究结果揭示了当前MLLMs在基本视觉认知方面存在令人担忧的缺陷，其表现为经常接近随机猜测，并且即使使用先进的提示技术，也只有微小的改进。这些结果强调了加强对MLLMs核心视觉推理能力研究的迫切需求。为了促进对该领域的进一步研究，我们在此发布VisFactor基准测试：[在此处填写URL] 

---
# Ensemble ToT of LLMs and Its Application to Automatic Grading System for Supporting Self-Learning 

**Title (ZH)**: 集合大语言模型的ToT及其在支持自主学习的自动评分系统中的应用 

**Authors**: Yuki Ito, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.16399)  

**Abstract**: Providing students with detailed and timely grading feedback is essential for self-learning. While existing LLM-based grading systems are promising, most of them rely on one single model, which limits their performance. To address this, we propose Ensemble Tree-of-Thought (ToT), a framework that enhances LLM outputs by integrating multiple models. Using this framework, we develop a grading system. Ensemble ToT follows three steps: (1) analyzing LLM performance, (2) generating candidate answers, and (3) refining them into a final result. Based on this, our grading system first evaluates the grading tendencies of LLMs, then generates multiple results, and finally integrates them via a simulated debate. Experimental results demonstrate our approach's ability to provide accurate and explainable grading by effectively coordinating multiple LLMs. 

**Abstract (ZH)**: 为学生提供详细及时的评分反馈是促进自主学习的关键。虽然现有的基于大规模语言模型（LLM）的评分系统具有一定的前景，但大多数系统依赖单一模型，限制了其性能。为了解决这一问题，我们提出了一种集成思考树（Ensemble Tree-of-Thought, Ensemble ToT）框架，通过结合多个模型来增强LLM的输出。基于此框架，我们开发了一种评分系统。Ensemble ToT遵循三个步骤：（1）分析LLM的表现，（2）生成候选答案，并（3）将它们 refinement 成最终结果。基于此，我们的评分系统首先评估LLM的评分倾向，然后生成多个结果，最后通过模拟辩论的方式将它们集成起来。实验结果表明，通过有效协调多个LLM，这种方法能够提供准确且可解释的评分能力。 

---
# An Analyst-Inspector Framework for Evaluating Reproducibility of LLMs in Data Science 

**Title (ZH)**: 数据科学中大型语言模型可重复性评估的分析师-检查员框架 

**Authors**: Qiuhai Zeng, Claire Jin, Xinyue Wang, Yuhan Zheng, Qunhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.16395)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential for data science tasks via code generation. However, the exploratory nature of data science, alongside the stochastic and opaque outputs of LLMs, raise concerns about their reliability. While prior work focuses on benchmarking LLM accuracy, reproducibility remains underexplored, despite being critical to establishing trust in LLM-driven analysis.
We propose a novel analyst-inspector framework to automatically evaluate and enforce the reproducibility of LLM-generated data science workflows - the first rigorous approach to the best of our knowledge. Defining reproducibility as the sufficiency and completeness of workflows for reproducing functionally equivalent code, this framework enforces computational reproducibility principles, ensuring transparent, well-documented LLM workflows while minimizing reliance on implicit model assumptions.
Using this framework, we systematically evaluate five state-of-the-art LLMs on 1,032 data analysis tasks across three diverse benchmark datasets. We also introduce two novel reproducibility-enhancing prompting strategies. Our results show that higher reproducibility strongly correlates with improved accuracy and reproducibility-enhancing prompts are effective, demonstrating structured prompting's potential to enhance automated data science workflows and enable transparent, robust AI-driven analysis. Our code is publicly available. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经通过代码生成展示了在数据科学任务中的潜力。然而，数据科学的探索性质，以及LLMs的随机性和不透明输出，引起了对其可靠性的担忧。尽管先前工作的重点在于评估LLM的准确性，但重现性（可复现性）在建立对LLM驱动分析的信任方面仍然是未被充分探索的重要方面。

我们提出了一种新的分析师-检查员框架，以自动化评估和强制执行LLM生成的数据科学工作流的重现性——到我们所知，这是首个严格意义上的方法。我们将重现性定义为能够再现功能等效代码的工作流的充分性和完整性，该框架确保计算上的可复现性原则得到遵守，保证LLM工作流的透明和文档化，同时最大限度地减少对隐含模型假设的依赖。

使用该框架，我们系统地评估了五个最先进的LLM在三个不同基准数据集上的1032个数据分析任务。我们还引入了两种新的增强重现性的提示策略。我们的研究表明，更高的重现性与更好的准确性高度相关，并且增强重现性的提示是有效的，证明了结构化提示在增强自动化数据科学工作流和实现透明、稳健的人工智能驱动分析方面具有潜力。我们的代码已经公开。 

---
# Toward a Flexible Framework for Linear Representation Hypothesis Using Maximum Likelihood Estimation 

**Title (ZH)**: 面向最大似然估计的线性表示假设的灵活框架研究 

**Authors**: Trung Nguyen, Yan Leng  

**Link**: [PDF](https://arxiv.org/pdf/2502.16385)  

**Abstract**: Linear representation hypothesis posits that high-level concepts are encoded as linear directions in the representation spaces of LLMs. Park et al. (2024) formalize this notion by unifying multiple interpretations of linear representation, such as 1-dimensional subspace representation and interventions, using a causal inner product. However, their framework relies on single-token counterfactual pairs and cannot handle ambiguous contrasting pairs, limiting its applicability to complex or context-dependent concepts. We introduce a new notion of binary concepts as unit vectors in a canonical representation space, and utilize LLMs' (neural) activation differences along with maximum likelihood estimation (MLE) to compute concept directions (i.e., steering vectors). Our method, Sum of Activation-base Normalized Difference (SAND), formalizes the use of activation differences modeled as samples from a von Mises-Fisher (vMF) distribution, providing a principled approach to derive concept directions. We extend the applicability of Park et al. (2024) by eliminating the dependency on unembedding representations and single-token pairs. Through experiments with LLaMA models across diverse concepts and benchmarks, we demonstrate that our lightweight approach offers greater flexibility, superior performance in activation engineering tasks like monitoring and manipulation. 

**Abstract (ZH)**: 线性表示假说认为高级概念在大规模语言模型（LLM）的表示空间中被编码为线性方向。Park等人（2024）通过使用因果内积来统一线性表示的多种解释，如1维子空间表示和干预，形式化了这一观点。然而，他们的框架依赖于单一标记的反事实对，无法处理含糊的对比对，从而限制了其应用于复杂或情境依赖性概念的能力。我们提出了一种新的二元概念概念，将其表示为标准表示空间中的单位向量，并利用LLM（神经）激活差异以及最大似然估计（MLE）来计算概念方向（即航向向量）。我们提出的方法，即激活基归一化差值之和（SAND），将激活差异的形式化表示为vMises-Fisher（vMF）分布的样本，提供了一种严谨的方法来推导概念方向。我们通过消除对外嵌表示和单一标记对的依赖，扩展了Park等人（2024）的适用范围。通过在各种概念和基准上使用LLaMA模型进行实验，我们表明，我们的轻量级方法提供了更大的灵活性，并在激活工程任务（如监控和操纵）中表现出更优的性能。 

---
# Dynamic Coalition Structure Detection in Natural Language-based Interactions 

**Title (ZH)**: 基于自然语言交互的动态合作结构检测 

**Authors**: Abhishek N. Kulkarni, Andy Liu, Jean-Raphael Gaglione, Daniel Fried, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16339)  

**Abstract**: In strategic multi-agent sequential interactions, detecting dynamic coalition structures is crucial for understanding how self-interested agents coordinate to influence outcomes. However, natural-language-based interactions introduce unique challenges to coalition detection due to ambiguity over intents and difficulty in modeling players' subjective perspectives. We propose a new method that leverages recent advancements in large language models and game theory to predict dynamic multilateral coalition formation in Diplomacy, a strategic multi-agent game where agents negotiate coalitions using natural language. The method consists of two stages. The first stage extracts the set of agreements discussed by two agents in their private dialogue, by combining a parsing-based filtering function with a fine-tuned language model trained to predict player intents. In the second stage, we define a new metric using the concept of subjective rationalizability from hypergame theory to evaluate the expected value of an agreement for each player. We then compute this metric for each agreement identified in the first stage by assessing the strategic value of the agreement for both players and taking into account the subjective belief of one player that the second player would honor the agreement. We demonstrate that our method effectively detects potential coalition structures in online Diplomacy gameplay by assigning high values to agreements likely to be honored and low values to those likely to be violated. The proposed method provides foundational insights into coalition formation in multi-agent environments with language-based negotiation and offers key directions for future research on the analysis of complex natural language-based interactions between agents. 

**Abstract (ZH)**: 在战略性的多代理人序列交互中，检测动态的联盟结构对于理解自利代理人如何协调以影响结果至关重要。然而，基于自然语言的交互给联盟检测带来了独特的挑战，因为意图存在歧义且难以建模玩家的主观视角。我们提出了一种新方法，利用最近在大规模语言模型和博弈理论方面的进展，预测在《外交》（Diplomacy）这种战略多代理人游戏中动态形成的多边联盟。《外交》游戏中，代理人通过自然语言进行联盟谈判。该方法包含两个阶段。第一阶段通过结合基于解析的过滤函数和细调的语言模型（该模型被训练以预测玩家的意图），提取两名代理人在私人对话中讨论的一组协议。第二阶段定义了一个新的度量标准，使用超博弈理论中的主观合理化概念来评估每个玩家协议预期价值。我们通过评估协议的策略价值和一方玩家认为另一方玩家会遵守协议的主观信念，为第一阶段识别出的每个协议计算此度量标准。我们通过评估协议可能被遵守时给予高值、可能被违反时给予低值，证明了该方法能够有效检测在线《外交》游戏中的潜在联盟结构。提出的方法为基于语言的协商环境下多代理人环境中的联盟形成提供了基础性的见解，并为未来分析复杂自然语言交互的研究提供了关键方向。 

---
# Interrogating LLM design under a fair learning doctrine 

**Title (ZH)**: 在公正学习原则下质疑LLM设计 

**Authors**: Johnny Tian-Zheng Wei, Maggie Wang, Ameya Godbole, Jonathan H. Choi, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.16290)  

**Abstract**: The current discourse on large language models (LLMs) and copyright largely takes a "behavioral" perspective, focusing on model outputs and evaluating whether they are substantially similar to training data. However, substantial similarity is difficult to define algorithmically and a narrow focus on model outputs is insufficient to address all copyright risks. In this interdisciplinary work, we take a complementary "structural" perspective and shift our focus to how LLMs are trained. We operationalize a notion of "fair learning" by measuring whether any training decision substantially affected the model's memorization. As a case study, we deconstruct Pythia, an open-source LLM, and demonstrate the use of causal and correlational analyses to make factual determinations about Pythia's training decisions. By proposing a legal standard for fair learning and connecting memorization analyses to this standard, we identify how judges may advance the goals of copyright law through adjudication. Finally, we discuss how a fair learning standard might evolve to enhance its clarity by becoming more rule-like and incorporating external technical guidelines. 

**Abstract (ZH)**: 当前关于大规模语言模型（LLMs）和版权的讨论主要采取“行为”视角，重点关注模型输出，并评估其是否与训练数据实质性相似。然而，实质性相似难以通过算法进行定义，而仅关注模型输出不足以全面应对版权风险。在本项跨学科研究中，我们采取了补充性的“结构”视角，将重点转向LLMs的训练方式。通过衡量任何训练决策是否实质性影响了模型的记忆，我们操作化了“公平学习”的概念。以开源LLM Pythia为例，我们对该模型的训练决策进行了因果分析和相关性分析，以事实方式确定其训练决策。通过提出一个公平学习的法律标准，并将记忆分析与该标准相结合，我们确定了法官通过审理如何推进版权法的立法目标。最后，我们讨论了如何通过使其更具规则性并纳入外部技术指导原则来提高公平学习标准的清晰度，从而进一步增强其有效性。 

---
# Maybe I Should Not Answer That, but... Do LLMs Understand The Safety of Their Inputs? 

**Title (ZH)**: 或许我本不应回答这个问题，但……大语言模型是否理解其输入的安全性？ 

**Authors**: Maciej Chrabąszcz, Filip Szatkowski, Bartosz Wójcik, Jan Dubiński, Tomasz Trzciński  

**Link**: [PDF](https://arxiv.org/pdf/2502.16174)  

**Abstract**: Ensuring the safety of the Large Language Model (LLM) is critical, but currently used methods in most cases sacrifice the model performance to obtain increased safety or perform poorly on data outside of their adaptation distribution. We investigate existing methods for such generalization and find them insufficient. Surprisingly, while even plain LLMs recognize unsafe prompts, they may still generate unsafe responses. To avoid performance degradation and preserve safe performance, we advocate for a two-step framework, where we first identify unsafe prompts via a lightweight classifier, and apply a "safe" model only to such prompts. In particular, we explore the design of the safety detector in more detail, investigating the use of different classifier architectures and prompting techniques. Interestingly, we find that the final hidden state for the last token is enough to provide robust performance, minimizing false positives on benign data while performing well on malicious prompt detection. Additionally, we show that classifiers trained on the representations from different model layers perform comparably on the latest model layers, indicating that safety representation is present in the LLMs' hidden states at most model stages. Our work is a step towards efficient, representation-based safety mechanisms for LLMs. 

**Abstract (ZH)**: 确保大型语言模型（LLM）的安全性至关重要，但目前大多数方法要么牺牲模型性能以获得更高的安全性能，要么在模型适应分布之外的数据上表现不佳。我们调查了现有方法的通用性，并发现这些方法存在不足。令人惊讶的是，即使简单的LLM也能识别出不安全的提示，但在某些情况下仍会产生不安全的响应。为了避免性能下降并保持安全性能，我们提倡采用两步框架，首先通过一个轻量级分类器识别不安全的提示，然后仅对这些提示应用“安全”模型。特别地，我们详细探讨了安全性检测器的设计，研究了不同分类器架构和提示技术的应用。有趣的是，我们发现最后一词的最终隐藏状态足以提供稳健的性能，能够在良性数据上最大限度地减少假阳性，同时在恶意提示检测方面表现良好。此外，我们展示了在不同模型层上训练的分类器在最新的模型层上的表现大致相当，这表明安全性特征在LLM的隐藏状态中在大多数模型阶段都存在。我们的研究工作为LLM提供了一种高效的安全机制奠定了基础，基于表示的方法。 

---
# OmniParser V2: Structured-Points-of-Thought for Unified Visual Text Parsing and Its Generality to Multimodal Large Language Models 

**Title (ZH)**: OmniParser V2：统一视觉文本解析的结构化思维点及其在多模态大型语言模型中的普遍适用性 

**Authors**: Wenwen Yu, Zhibo Yang, Jianqiang Wan, Sibo Song, Jun Tang, Wenqing Cheng, Yuliang Liu, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2502.16161)  

**Abstract**: Visually-situated text parsing (VsTP) has recently seen notable advancements, driven by the growing demand for automated document understanding and the emergence of large language models capable of processing document-based questions. While various methods have been proposed to tackle the complexities of VsTP, existing solutions often rely on task-specific architectures and objectives for individual tasks. This leads to modal isolation and complex workflows due to the diversified targets and heterogeneous schemas. In this paper, we introduce OmniParser V2, a universal model that unifies VsTP typical tasks, including text spotting, key information extraction, table recognition, and layout analysis, into a unified framework. Central to our approach is the proposed Structured-Points-of-Thought (SPOT) prompting schemas, which improves model performance across diverse scenarios by leveraging a unified encoder-decoder architecture, objective, and input\&output representation. SPOT eliminates the need for task-specific architectures and loss functions, significantly simplifying the processing pipeline. Our extensive evaluations across four tasks on eight different datasets show that OmniParser V2 achieves state-of-the-art or competitive results in VsTP. Additionally, we explore the integration of SPOT within a multimodal large language model structure, further enhancing text localization and recognition capabilities, thereby confirming the generality of SPOT prompting technique. The code is available at \href{this https URL}{AdvancedLiterateMachinery}. 

**Abstract (ZH)**: 视觉定位文本解析（VsTP）近年来取得了显著的增长，这主要得益于对自动化文档理解日益增长的需求以及能够处理基于文档的问题的大规模语言模型的出现。尽管提出了多种方法来应对VsTP的复杂性，但现有的解决方案往往依赖于特定任务的架构和目标。这导致了模态隔离和复杂的工作流程，由于目标多样和异构模式的不统一。在本文中，我们引入了OmniParser V2，这是一种统一模型，将VsTP典型的任务，包括文本检测、关键信息提取、表格识别和布局分析，统一到一个框架中。我们方法的核心是提出的结构化思路（SPOT）提示框架，通过利用统一的编码器-解码器架构、目标和输入/输出表示，提高了模型在各种场景下的性能。SPOT消除了任务特定架构和损失函数的需要，极大地简化了处理流程。我们在四个任务上八个不同数据集的广泛评估显示，OmniParser V2在VsTP中达到了最先进的或具有竞争力的表现。此外，我们在多模态大规模语言模型结构中探索了SPOT的集成，进一步增强文本定位和识别能力，从而确认了SPOT提示方法的普适性。代码可在以下链接获取：\href{这个链接}{AdvancedLiterateMachinery}。 

---
# PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving 

**Title (ZH)**: PlanGEN：一个用于复杂问题求解的规划与推理轨迹生成的多agent框架 

**Authors**: Mihir Parmar, Xin Liu, Palash Goyal, Yanfei Chen, Long Le, Swaroop Mishra, Hossein Mobahi, Jindong Gu, Zifeng Wang, Hootan Nakhost, Chitta Baral, Chen-Yu Lee, Tomas Pfister, Hamid Palangi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16111)  

**Abstract**: Recent agent frameworks and inference-time algorithms often struggle with complex planning problems due to limitations in verifying generated plans or reasoning and varying complexity of instances within a single task. Many existing methods for these tasks either perform task-level verification without considering constraints or apply inference-time algorithms without adapting to instance-level complexity. To address these limitations, we propose PlanGEN, a model-agnostic and easily scalable agent framework with three key components: constraint, verification, and selection agents. Specifically, our approach proposes constraint-guided iterative verification to enhance performance of inference-time algorithms--Best of N, Tree-of-Thought, and REBASE. In PlanGEN framework, the selection agent optimizes algorithm choice based on instance complexity, ensuring better adaptability to complex planning problems. Experimental results demonstrate significant improvements over the strongest baseline across multiple benchmarks, achieving state-of-the-art results on NATURAL PLAN ($\sim$8%$\uparrow$), OlympiadBench ($\sim$4%$\uparrow$), DocFinQA ($\sim$7%$\uparrow$), and GPQA ($\sim$1%$\uparrow$). Our key finding highlights that constraint-guided iterative verification improves inference-time algorithms, and adaptive selection further boosts performance on complex planning and reasoning problems. 

**Abstract (ZH)**: 近年来，代理框架和推理时算法在处理复杂的规划问题时经常遇到困难，这是因为生成计划的验证和推理能力有限，以及单个任务中实例复杂性的变化。许多现有的方法要么在任务级别进行验证而不考虑约束条件，要么在推理时应用算法而不适应实例级别的复杂性。为了解决这些问题，我们提出了一种通用且易于扩展的代理框架PlanGEN，该框架包含三个关键组件：约束代理、验证代理和选择代理。具体而言，我们的方法提出了一种约束引导的迭代验证策略，以增强推理时算法（如Best of N、Tree-of-Thought、REBASE）的性能。在PlanGEN框架中，选择代理根据实例复杂性优化算法选择，确保更好地适应复杂的规划问题。实验结果表明，PlanGEN在多个基准测试中明显优于最强的基线，分别在NATURAL PLAN（+8%）、OlympiadBench（+4%）、DocFinQA（+7%）和GPQA（+1%）上取得了最先进的结果。我们的关键发现表明，约束引导的迭代验证可以提高推理时算法的性能，而适应性选择则进一步提高了复杂规划和推理问题的表现。 

---
# Inference Computation Scaling for Feature Augmentation in Recommendation Systems 

**Title (ZH)**: 推荐系统中特征增强的推理计算扩展研究 

**Authors**: Weihao Liu, Zhaocheng Du, Haiyuan Zhao, Wenbo Zhang, Xiaoyan Zhao, Gang Wang, Zhenhua Dong, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16040)  

**Abstract**: Large language models have become a powerful method for feature augmentation in recommendation systems. However, existing approaches relying on quick inference often suffer from incomplete feature coverage and insufficient specificity in feature descriptions, limiting their ability to capture fine-grained user preferences and undermining overall performance. Motivated by the recent success of inference scaling in math and coding tasks, we explore whether scaling inference can address these limitations and enhance feature quality.
Our experiments show that scaling inference leads to significant improvements in recommendation performance, with a 12% increase in NDCG@10. The gains can be attributed to two key factors: feature quantity and specificity. In particular, models using extended Chain-of-Thought (CoT) reasoning generate a greater number of detailed and precise features, offering deeper insights into user preferences and overcoming the limitations of quick inference. We further investigate the factors influencing feature quantity, revealing that model choice and search strategy play critical roles in generating a richer and more diverse feature set. This is the first work to apply inference scaling to feature augmentation in recommendation systems, bridging advances in reasoning tasks to enhance personalized recommendation. 

**Abstract (ZH)**: 大型语言模型已成为推荐系统中特征增强的一种强大方法。然而，现有的依赖快速推理的方法往往存在特征覆盖面不完整和特征描述不够具体的问题，限制了它们捕捉用户细微偏好的能力，从而影响整体性能。受最近在数学和编程任务中推理扩展取得成功的影响，我们探索是否可以通过扩展推理来解决这些限制并提升特征质量。

我们的实验表明，扩展推理在推荐性能上带来了显著的改进，NDCG@10提高了12%。这些收益可以归因于两个关键因素：特征的数量和描述的精确性。特别是，使用扩展的链式思维（Chain-of-Thought, CoT）推理的模型能够生成更多的详细和精确的特征，提供更深入的用户偏好见解，并克服了快速推理的局限性。我们进一步探讨影响特征数量的因素，揭示了模型选择和搜索策略在生成更丰富和多样化的特征集方面扮演着关键角色。这是首次将推理扩展应用于推荐系统中的特征增强，将推理任务的进展应用于个性化推荐的提升。 

---
# Forgotten Polygons: Multimodal Large Language Models are Shape-Blind 

**Title (ZH)**: 遗忘的多边形：多模态大型语言模型忽视形状 

**Authors**: William Rudman, Michal Golovanesky, Amir Bar, Vedant Palit, Yann LeCun, Carsten Eickhoff, Ritambhara Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.15969)  

**Abstract**: Despite strong performance on vision-language tasks, Multimodal Large Language Models (MLLMs) struggle with mathematical problem-solving, with both open-source and state-of-the-art models falling short of human performance on visual-math benchmarks. To systematically examine visual-mathematical reasoning in MLLMs, we (1) evaluate their understanding of geometric primitives, (2) test multi-step reasoning, and (3) explore a potential solution to improve visual reasoning capabilities. Our findings reveal fundamental shortcomings in shape recognition, with top models achieving under 50% accuracy in identifying regular polygons. We analyze these failures through the lens of dual-process theory and show that MLLMs rely on System 1 (intuitive, memorized associations) rather than System 2 (deliberate reasoning). Consequently, MLLMs fail to count the sides of both familiar and novel shapes, suggesting they have neither learned the concept of sides nor effectively process visual inputs. Finally, we propose Visually Cued Chain-of-Thought (VC-CoT) prompting, which enhances multi-step mathematical reasoning by explicitly referencing visual annotations in diagrams, boosting GPT-4o's accuracy on an irregular polygon side-counting task from 7% to 93%. Our findings suggest that System 2 reasoning in MLLMs remains an open problem, and visually-guided prompting is essential for successfully engaging visual reasoning. Code available at: this https URL. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）在视觉语言任务上表现出色，但在数学问题解决方面却遇到困难，开源和最先进的模型在视觉数学基准测试中的表现均低于人类。为系统地研究MLLMs的视觉数学推理能力，我们进行了以下几项测试：（1）评估其对几何原语的理解；（2）测试多步推理；（3）探索提高视觉推理能力的潜在解决方案。我们的研究发现，这些模型在形状识别方面存在根本性缺陷，顶级模型在识别正多边形时的准确率仅为不到50%。我们通过双重过程理论的视角分析了这些失败，发现MLLMs依赖于系统1（直观且记忆化的关联），而不是系统2（有意识的推理）。因此，MLLMs在计算熟悉和新型形状的边数时均未能通过，这表明它们既未学会“边”的概念，也未有效处理视觉输入。最后，我们提出了图像提示链式思考（Visually Cued Chain-of-Thought, VC-CoT）提示方法，该方法通过明确引用图表中的视觉注释来增强多步数学推理能力，从而将GPT-4o在非正多边形边数计数任务中的准确率从7%提高到93%。我们的研究结果表明，MLLMs中的系统2推理仍然是一个开放问题，而基于视觉的提示对于成功进行视觉推理是必不可少的。代码可在此访问：this https URL. 

---
# Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models 

**Title (ZH)**: 《 minions：设备端与云端语言模型的低成本协作》

在此翻译中，“Minions”被处理为专有名词或系统名称，并未直接翻译，保持了原名；“Cost-efficient Collaboration”译为“低成本协作”，符合学术规范。标题中的内容说明了设备端和云端语言模型之间的协作方式及其经济性特点。 

**Authors**: Avanika Narayan, Dan Biderman, Sabri Eyuboglu, Avner May, Scott Linderman, James Zou, Christopher Re  

**Link**: [PDF](https://arxiv.org/pdf/2502.15964)  

**Abstract**: We investigate an emerging setup in which a small, on-device language model (LM) with access to local data communicates with a frontier, cloud-hosted LM to solve real-world tasks involving financial, medical, and scientific reasoning over long documents. Can a local-remote collaboration reduce cloud inference costs while preserving quality? First, we consider a naive collaboration protocol where the local and remote models simply chat back and forth. Because only the local model reads the full context, this protocol achieves a 30.4x reduction in remote costs, but recovers only 87% of the performance of the frontier model. We identify two key limitations of this protocol: the local model struggles to (1) follow the remote model's multi-step instructions and (2) reason over long contexts. Motivated by these observations, we study an extension of this protocol, coined MinionS, in which the remote model decomposes the task into easier subtasks over shorter chunks of the document, that are executed locally in parallel. MinionS reduces costs by 5.7x on average while recovering 97.9% of the performance of the remote model alone. Our analysis reveals several key design choices that influence the trade-off between cost and performance in local-remote systems. 

**Abstract (ZH)**: 我们研究了一种新兴的设置，在这种设置中，一个具有访问本地数据的小型设备端语言模型（LM）与一个领先的云托管语言模型进行通信，以解决涉及金融、医疗和科学推理的长文档实际任务。设备端-云端合作能否在保持质量的前提下降低云推理成本？首先，我们考虑了一个简单的合作协议，即本地模型和远程模型彼此来回聊天。由于只有本地模型能够读取完整的上下文，这种协议将远程成本降低了30.4倍，但仅恢复了远程模型性能的87%。我们识别了这种协议的两个关键局限性：本地模型难以（1）跟随远程模型的多步指令，并且（2）处理长上下文。基于这些观察，我们研究了一种名为MinionS的新扩展协议，在这种协议中，远程模型将任务分解为更简单的子任务，并在较短文档片段上执行这些子任务，这些子任务在本地并行执行。MinionS平均将成本降低了5.7倍，同时恢复了远程模型性能的97.9%。我们的分析揭示了几种关键设计选择，这些选择影响设备端-云端系统的成本与性能之间的权衡。 

---
# Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models 

**Title (ZH)**: 使用数据专家模型的混合方法优化预训练数据混合 

**Authors**: Lior Belenki, Alekh Agarwal, Tianze Shi, Kristina Toutanova  

**Link**: [PDF](https://arxiv.org/pdf/2502.15950)  

**Abstract**: We propose a method to optimize language model pre-training data mixtures through efficient approximation of the cross-entropy loss corresponding to each candidate mixture via a Mixture of Data Experts (MDE). We use this approximation as a source of additional features in a regression model, trained from observations of model loss for a small number of mixtures.
Experiments with Transformer decoder-only language models in the range of 70M to 1B parameters on the SlimPajama dataset show that our method achieves significantly better performance than approaches that train regression models using only the mixture rates as input features. Combining this improved optimization method with an objective that takes into account cross-entropy on end task data leads to superior performance on few-shot downstream evaluations.
We also provide theoretical insights on why aggregation of data expert predictions can provide good approximations to model losses for data mixtures. 

**Abstract (ZH)**: 我们提出了一种通过高效逼近每个候选数据混合对应的交叉熵损失来优化语言模型预训练数据混合的方法，该方法利用数据专家混合（Mixture of Data Experts，MDE）来进行近似。我们使用这种近似值作为回归模型的附加特征来源，该回归模型是从少量数据混合的模型损失观察中训练得到的。

在SlimPajama数据集上，使用从7000万到1亿参数的Transformer解码器语言模型进行的实验表明，我们的方法在使用仅输入混合率作为特征的回归模型训练方法中取得了显著的性能提升。将这种改进的优化方法与考虑最终任务数据交叉熵的目标相结合，可以在少量样本下游评估中取得更优异的表现。

此外，我们还提供了关于为什么数据专家预测的聚合可以为数据混合的模型损失提供良好近似的理论见解。 

---
# Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs 

**Title (ZH)**: 直趋零点：为何线性衰减学习率至零是大语言模型的最佳选择 

**Authors**: Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2502.15938)  

**Abstract**: LLMs are commonly trained with a learning rate (LR) warmup, followed by cosine decay to 10% of the maximum (10x decay). In a large-scale empirical study, we show that under an optimal peak LR, a simple linear decay-to-zero (D2Z) schedule consistently outperforms other schedules when training at compute-optimal dataset sizes. D2Z is superior across a range of model sizes, batch sizes, datasets, and vocabularies. Benefits increase as dataset size increases. Leveraging a novel interpretation of AdamW as an exponential moving average of weight updates, we show how linear D2Z optimally balances the demands of early training (moving away from initial conditions) and late training (averaging over more updates in order to mitigate gradient noise). In experiments, a 610M-parameter model trained for 80 tokens-per-parameter (TPP) using D2Z achieves lower loss than when trained for 200 TPP using 10x decay, corresponding to an astonishing 60% compute savings. Models such as Llama2-7B, trained for 286 TPP with 10x decay, could likely have saved a majority of compute by training with D2Z. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常使用学习率（LR）预热，然后采用余弦衰减至最大值的10%（即10倍衰减）进行训练。在一项大规模实证研究中，我们展示了在最优峰值学习率条件下，线性衰减至零（D2Z）计划在计算最优的数据集大小下训练时，始终优于其他计划。D2Z 在不同模型规模、批量大小、数据集和词汇量下表现更优。随着数据集尺寸的增加，其优势更为显著。利用对AdamW的新颖解释，即作为权重更新的加权指数平均，我们展示了线性D2Z如何在早期训练（远离初始条件）和晚期训练（跨越更多更新以减轻梯度噪声）的需求之间实现最优平衡。在实验中，一个包含610M参数的模型训练80 token-per-parameter（TPP）时使用D2Z获得的损失低于训练200 TPP时采用10倍衰减的损失，这对应着惊人的60%的计算资源节省。例如，使用10倍衰减训练286 TPP的Llama2-7B模型很可能通过采用D2Z训练而显著节省计算资源。 

---
# LLMs in Mobile Apps: Practices, Challenges, and Opportunities 

**Title (ZH)**: 移动应用中的大规模语言模型：实践、挑战与机遇 

**Authors**: Kimberly Hau, Safwat Hassan, Shurui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.15908)  

**Abstract**: The integration of AI techniques has become increasingly popular in software development, enhancing performance, usability, and the availability of intelligent features. With the rise of large language models (LLMs) and generative AI, developers now have access to a wealth of high-quality open-source models and APIs from closed-source providers, enabling easier experimentation and integration of LLMs into various systems. This has also opened new possibilities in mobile application (app) development, allowing for more personalized and intelligent apps. However, integrating LLM into mobile apps might present unique challenges for developers, particularly regarding mobile device constraints, API management, and code infrastructure. In this project, we constructed a comprehensive dataset of 149 LLM-enabled Android apps and conducted an exploratory analysis to understand how LLMs are deployed and used within mobile apps. This analysis highlights key characteristics of the dataset, prevalent integration strategies, and common challenges developers face. Our findings provide valuable insights for future research and tooling development aimed at enhancing LLM-enabled mobile apps. 

**Abstract (ZH)**: 将人工智能技术集成到软件开发中已成为一种越来越流行的趋势，这提升了软件的性能、易用性和智能功能的可用性。随着大规模语言模型（LLMs）和生成式人工智能的兴起，开发人员现在可以访问来自封闭源供应商的高质量开源模型和APIs，从而更容易地进行LLMs的实验和集成，使其能够应用于各种系统。这还为移动应用（App）开发带来了新的可能性，使应用程序更加个性化和智能化。然而，将LLMs集成到移动应用中可能会给开发人员带来独特的挑战，特别是与移动设备限制、API管理以及代码基础设施相关的问题。在本项目中，我们构建了一个包含149个集成LLM的Android应用的综合数据集，并进行了探索性分析，以了解LLMs在移动应用中的部署和应用方式。这一分析揭示了数据集中的一些关键特征、常用的集成策略以及开发者面临的常见挑战。我们的发现为未来致力于提升LLM集成移动应用的研究和工具开发提供了宝贵见解。 

---
# IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector 

**Title (ZH)**: IPAD：逆向提示词用于AI检测——一种稳健且可解释的LLM生成文本检测器 

**Authors**: Zheng Chen, Yushi Feng, Changyang He, Yue Deng, Hongxi Pu, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15902)  

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide explainable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and a Distinguisher that examines how well the input texts align with the predicted prompts. We develop and examine two versions of Distinguishers. Empirical evaluations demonstrate that both Distinguishers perform significantly better than the baseline methods, with version2 outperforming baselines by 9.73% on in-distribution data (F1-score) and 12.65% on OOD data (AUROC). Furthermore, a user study is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在文本生成方面达到人类水平的流畅度，这使得区分人类撰写和LLM生成的文本变得更加复杂。这增加了滥用的风险，并突显了可靠检测器的必要性。然而，现有的检测器在处理未知分布（OOD）数据和攻击数据时表现出较差的鲁棒性，这对于实际应用场景至关重要。此外，它们很难提供可解释的证据来支持其决策，从而损害了其可靠性。鉴于这些挑战，我们提出了一种名为IPAD（Inverse Prompt for AI Detection）的新型框架，该框架包括一个Prompt逆向器，用于识别可能生成输入文本的预测提示，以及一个区分器，用于评估输入文本与预测提示的吻合程度。我们开发并研究了两种版本的区分器。实证评估表明，两种区分器的表现均显著优于基础方法，版本2在已知分布数据上的F1分数上比基线方法高出9.73%，在OOD数据上的AUROC上高出12.65%。此外，进行了一项用户研究，以说明IPAD通过使用户能够直接检查决策依据来增强AI检测可信度，从而为其实现的先进检测结果提供了可解释的支持。 

---
# Directional Gradient Projection for Robust Fine-Tuning of Foundation Models 

**Title (ZH)**: 面向方向梯度投影的稳健微调基础模型方法 

**Authors**: Chengyue Huang, Junjiao Tian, Brisa Maneechotesuwan, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2502.15895)  

**Abstract**: Robust fine-tuning aims to adapt large foundation models to downstream tasks while preserving their robustness to distribution shifts. Existing methods primarily focus on constraining and projecting current model towards the pre-trained initialization based on the magnitudes between fine-tuned and pre-trained weights, which often require extensive hyper-parameter tuning and can sometimes result in underfitting. In this work, we propose Directional Gradient Projection (DiGraP), a novel layer-wise trainable method that incorporates directional information from gradients to bridge regularization and multi-objective optimization. Besides demonstrating our method on image classification, as another contribution we generalize this area to the multi-modal evaluation settings for robust fine-tuning. Specifically, we first bridge the uni-modal and multi-modal gap by performing analysis on Image Classification reformulated Visual Question Answering (VQA) benchmarks and further categorize ten out-of-distribution (OOD) VQA datasets by distribution shift types and degree (i.e. near versus far OOD). Experimental results show that DiGraP consistently outperforms existing baselines across Image Classfication and VQA tasks with discriminative and generative backbones, improving both in-distribution (ID) generalization and OOD robustness. 

**Abstract (ZH)**: 稳健的微调旨在适应大型基础模型以满足下游任务需求，同时保持其在分布偏移情况下的稳健性。现有方法主要集中在通过调整和投影当前模型朝向预训练初始化，基于微调和预训练权重的大小差异进行约束，这通常需要大量的超参数调整，并且有时会导致模型过拟合。在此项工作中，我们提出了一种名为方向梯度投影（DiGraP）的新型逐层可训练方法，该方法将梯度的方向性信息融入正则化和多目标优化之间。除了在图像分类上演示了我们的方法之外，我们还将此领域推广到多模态评估环境下的稳健微调。具体而言，我们首先通过对图像分类重新定义的视觉问答（VQA）基准进行分析，缩小了单模态与多模态之间的差距，并进一步将十个跨分布（OOD）VQA数据集按照分布偏移类型和程度（即近似 OOD 和远端 OOD）进行了分类。实验结果表明，DiGraP 在图像分类和 VQA 任务中（无论是判别性还是生成性后端模型）均优于现有基线方法，提高了在分布内（ID）泛化和跨分布（OOD）稳健性方面的能力。 

---
# Position: Standard Benchmarks Fail -- LLM Agents Present Overlooked Risks for Financial Applications 

**Title (ZH)**: 位置：标准基准失效——大型语言模型代理在金融应用中带来了被忽视的风险 

**Authors**: Zichen Chen, Jiaao Chen, Jianda Chen, Misha Sra  

**Link**: [PDF](https://arxiv.org/pdf/2502.15865)  

**Abstract**: Current financial LLM agent benchmarks are inadequate. They prioritize task performance while ignoring fundamental safety risks. Threats like hallucinations, temporal misalignment, and adversarial vulnerabilities pose systemic risks in high-stakes financial environments, yet existing evaluation frameworks fail to capture these risks. We take a firm position: traditional benchmarks are insufficient to ensure the reliability of LLM agents in finance. To address this, we analyze existing financial LLM agent benchmarks, finding safety gaps and introducing ten risk-aware evaluation metrics. Through an empirical evaluation of both API-based and open-weight LLM agents, we reveal hidden vulnerabilities that remain undetected by conventional assessments. To move the field forward, we propose the Safety-Aware Evaluation Agent (SAEA), grounded in a three-level evaluation framework that assesses agents at the model level (intrinsic capabilities), workflow level (multi-step process reliability), and system level (integration robustness). Our findings highlight the urgent need to redefine LLM agent evaluation standards by shifting the focus from raw performance to safety, robustness, and real world resilience. 

**Abstract (ZH)**: 当前的金融预训练语言模型（LLM）代理基准尚不充分。这些基准在重视任务性能的同时忽视了基本的安全风险。像幻觉、时间错位和对抗性漏洞这样的威胁，在高风险的金融环境中可能引发系统性风险，但现有的评估框架未能捕捉到这些风险。我们认为：传统基准不足以确保金融领域的LLM代理的可靠性。为解决这一问题，我们分析了现有的金融LLM代理基准，发现其中的安全缺口，并提出了十个风险意识评价指标。通过实证评估基于API的和公开权重的LLM代理，我们揭示了部分隐藏的漏洞，这些漏洞在传统的评估中未被发现。为了推动该领域的发展，我们提出了安全意识评价代理（SAEA），基于三层评估框架，分别在模型层次（内在能力）、工作流层次（多步过程可靠性）和系统层次（整合稳健性）评估代理。我们的发现强调了重新定义LLM代理评估标准的紧迫性，即将注意力从单纯的性能转向安全、稳健性和实际环境中的韧性。 

---
# C3AI: Crafting and Evaluating Constitutions for Constitutional AI 

**Title (ZH)**: C3AI: 创作与评估宪法性人工智能的宪法

这里的“宪法”是在比喻意义上使用的，指代的是规范和指导人工智能系统行为的伦理、法律和原则框架。在翻译时，为了保持学术规范，可以将“Constitutional AI”翻译为“宪法性人工智能”，具体解释说明其含义。完整的标题可以进一步解释为“C3AI：创作与评估指导人工智能系统的伦理与法律框架”。 

**Authors**: Yara Kyrychenko, Ke Zhou, Edyta Bogucka, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2502.15861)  

**Abstract**: Constitutional AI (CAI) guides LLM behavior using constitutions, but identifying which principles are most effective for model alignment remains an open challenge. We introduce the C3AI framework (\textit{Crafting Constitutions for CAI models}), which serves two key functions: (1) selecting and structuring principles to form effective constitutions before fine-tuning; and (2) evaluating whether fine-tuned CAI models follow these principles in practice. By analyzing principles from AI and psychology, we found that positively framed, behavior-based principles align more closely with human preferences than negatively framed or trait-based principles. In a safety alignment use case, we applied a graph-based principle selection method to refine an existing CAI constitution, improving safety measures while maintaining strong general reasoning capabilities. Interestingly, fine-tuned CAI models performed well on negatively framed principles but struggled with positively framed ones, in contrast to our human alignment results. This highlights a potential gap between principle design and model adherence. Overall, C3AI provides a structured and scalable approach to both crafting and evaluating CAI constitutions. 

**Abstract (ZH)**: 宪法型AI（Constitutional AI, CAI）通过宪法来引导大语言模型（LLM）的行为，但确定哪些原则最有效地实现模型对齐仍是一个开放的挑战。我们引入了C3AI框架（Crafting Constitutions for CAI models），该框架具有两个关键功能：（1）在微调之前选择和结构化原则以形成有效的宪法；（2）评估微调后的CAI模型是否遵循这些原则。通过分析来自人工智能和心理学的原则，我们发现，以积极方式表述的行为导向原则比以消极方式表述或特质导向的原则更接近人类的偏好。在一个安全性对齐用例中，我们应用了一种基于图的原则选择方法来细化现有的CAI宪法，从而提高了安全性措施，同时保留了强大的泛化推理能力。有趣的是，微调后的CAI模型在消极表述的原则上表现良好，但在积极表述的原则上却存在困难，这与我们的结果不同，这突显了原则设计与模型遵从之间可能存在差距。总体而言，C3AI 提供了一种结构化和可扩展的方法来设计和评估CAI宪法。 

---
# Enhancing Domain-Specific Retrieval-Augmented Generation: Synthetic Data Generation and Evaluation using Reasoning Models 

**Title (ZH)**: 增强领域特定的检索增强生成：基于推理模型的合成数据生成与评估 

**Authors**: Aryan Jadon, Avinash Patil, Shashank Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15854)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems face significant performance gaps when applied to technical domains requiring precise information extraction from complex documents. Current evaluation methodologies relying on document-level metrics inadequately capture token-resolution retrieval accuracy that is critical for domain-related documents. We propose a framework combining granular evaluation metrics with synthetic data generation to optimize domain-specific RAG performance. First, we introduce token-aware metrics Precision $\Omega$ and Intersection-over-Union (IoU) that quantify context preservation versus information density trade-offs inherent in technical texts. Second, we develop a reasoning model-driven pipeline using instruction-tuned LLMs (DeepSeek-R1, DeepSeek-R1 distilled variants, and Phi-4) to generate context-anchored QA pairs with discontinuous reference spans across three specialized corpora: SEC 10-K filings (finance), biomedical abstracts (PubMed), and APT threat reports (cybersecurity).
Our empirical analysis reveals critical insights: smaller chunks (less than 10 tokens) improve precision by 31-42% (IoU = 0.071 vs. baseline 0.053) at recall costs (-18%), while domain-specific embedding strategies yield 22% variance in optimal chunk sizing (5-20 tokens). The DeepSeek-R1-Distill-Qwen-32B model demonstrates superior concept alignment (+14% mean IoU over alternatives), though no configuration universally dominates. Financial texts favor larger chunks for risk factor coverage (Recall = 0.81 at size = 20), whereas cybersecurity content benefits from atomic segmentation, Precision $\Omega = 0.28$ at size = 5.
Our code is available on this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）系统在应用于需要从复杂文档中精确提取信息的技术领域时，面临显著的性能差距。当前依赖于文档级指标的评估方法无法充分捕捉对领域相关文档至关重要的标记级检索准确性。我们提出了一种框架，结合细粒度的评估指标和合成数据生成，以优化特定领域的RAG性能。首先，我们引入了基于标记感知的精确度指标Precision Ω和交并比（IoU），以量化技术文本中上下文保持与信息密度之间的权衡。其次，我们开发了一个基于推理模型的流水线，使用指令调优的大型语言模型（如DeepSeek-R1、DeepSeek-R1蒸馏变体和Phi-4），生成上下文锚定的问答对，并在三个专门的语料库中构建交错的参考片段：SEC 10-K 纳入文件（金融）、PubMed 生物医学摘要（生物医学）和APT 威胁报告（网络安全）。

我们的实证分析揭示了关键见解：较小的片段（少于10个标记）可以将召回率降低18%的情况下提高精确度31-42%（IoU从0.053提升到0.071）。领域特定的嵌入策略使得最优片段大小的变异性达到了22%（范围在5-20个标记之间）。DeepSeek-R1-Distill-Qwen-32B模型在概念一致性方面表现更优（相对于其他模型的均值IoU提高了14%），然而没有一个配置能够一概而优。金融文本倾向于使用更大的片段来覆盖风险因素（尺寸为20时召回率为0.81），而网络安全内容则受益于原子化分割，在尺寸为5时Precision Ω达到了0.28。

我们的代码可在以下链接获得：<https://github.com/example-repo> 

---
# DeepRTL: Bridging Verilog Understanding and Generation with a Unified Representation Model 

**Title (ZH)**: DeepRTL：一种统一表示模型在Verilog 理解与生成之间的桥梁 

**Authors**: Yi Liu, Changran Xu, Yunhao Zhou, Zeju Li, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15832)  

**Abstract**: Recent advancements in large language models (LLMs) have shown significant potential for automating hardware description language (HDL) code generation from high-level natural language instructions. While fine-tuning has improved LLMs' performance in hardware design tasks, prior efforts have largely focused on Verilog generation, overlooking the equally critical task of Verilog understanding. Furthermore, existing models suffer from weak alignment between natural language descriptions and Verilog code, hindering the generation of high-quality, synthesizable designs. To address these issues, we present DeepRTL, a unified representation model that excels in both Verilog understanding and generation. Based on CodeT5+, DeepRTL is fine-tuned on a comprehensive dataset that aligns Verilog code with rich, multi-level natural language descriptions. We also introduce the first benchmark for Verilog understanding and take the initiative to apply embedding similarity and GPT Score to evaluate the models' understanding capabilities. These metrics capture semantic similarity more accurately than traditional methods like BLEU and ROUGE, which are limited to surface-level n-gram overlaps. By adapting curriculum learning to train DeepRTL, we enable it to significantly outperform GPT-4 in Verilog understanding tasks, while achieving performance on par with OpenAI's o1-preview model in Verilog generation tasks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在从高级自然语言指令自动生成硬件描述语言（HDL）代码方面展现了显著的潜力。虽然微调已经提高了LLMs在硬件设计任务中的表现，但之前的努力主要集中在Verilog生成上，忽视了同样至关重要的Verilog理解任务。此外，现有的模型在自然语言描述与Verilog代码之间存在弱对齐的问题，阻碍了高质量、可综合设计的生成。为了解决这些问题，我们提出了一种名为DeepRTL的统一表示模型，能够在Verilog理解和生成方面表现出色。基于CodeT5+，DeepRTL通过包含丰富多层次自然语言描述的综合数据集进行了微调。同时，我们引入了第一个Verilog理解基准，并率先使用嵌入相似性和GPT Score来评估模型的理解能力。这些指标比传统的BLEU和ROUGE等方法更准确地捕捉到语义相似性，因为传统方法仅限于表面级的n-gram重叠。通过适应性地应用课程学习来训练DeepRTL，使其在Verilog理解任务中显著优于GPT-4，在Verilog生成任务中则达到了与OpenAI的o1-preview模型相当的性能。 

---
# InductionBench: LLMs Fail in the Simplest Complexity Class 

**Title (ZH)**: InductionBench: 语言模型在最简单的复杂性类中失败 

**Authors**: Wenyue Hua, Tyler Wong, Sun Fei, Liangming Pan, Adam Jardine, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15823)  

**Abstract**: Large language models (LLMs) have shown remarkable improvements in reasoning and many existing benchmarks have been addressed by models such as o1 and o3 either fully or partially. However, a majority of these benchmarks emphasize deductive reasoning, including mathematical and coding tasks in which rules such as mathematical axioms or programming syntax are clearly defined, based on which LLMs can plan and apply these rules to arrive at a solution. In contrast, inductive reasoning, where one infers the underlying rules from observed data, remains less explored. Such inductive processes lie at the heart of scientific discovery, as they enable researchers to extract general principles from empirical observations. To assess whether LLMs possess this capacity, we introduce InductionBench, a new benchmark designed to evaluate the inductive reasoning ability of LLMs. Our experimental findings reveal that even the most advanced models available struggle to master the simplest complexity classes within the subregular hierarchy of functions, highlighting a notable deficiency in current LLMs' inductive reasoning capabilities. Coda and data are available this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理方面展现出了显著的进步，许多现有的基准测试已经被如o1和o3这样的模型完全或部分解决。然而，这些基准测试主要侧重于演绎推理，包括数学和编程任务等，其中的规则（如数学公理或编程语法）被明确定义，因此LLMs能够规划并应用这些规则以得出解决方案。相比之下，归纳推理，即从观察到的数据推断出潜在的规则，仍然较少受到关注。这种归纳过程是科学研究的核心，因为它使研究人员能够从经验观察中提取出一般原则。为了评估LLMs是否具备这种能力，我们引入了InductionBench，一个新的基准测试，用于评估LLMs的归纳推理能力。我们的实验结果表明，即使是最先进的模型也难以掌握函数子正则层次中最简单的复杂性类别，这突显了当前LLMs在归纳推理能力方面存在的显著不足。数据和代码可在以下链接获取：[这里插入链接]。 

---
# Slamming: Training a Speech Language Model on One GPU in a Day 

**Title (ZH)**: Slamming: 在一天内于单张GPU上训练一个语音语言模型 

**Authors**: Gallil Maimon, Avishai Elmakies, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15814)  

**Abstract**: We introduce Slam, a recipe for training high-quality Speech Language Models (SLMs) on a single academic GPU in 24 hours. We do so through empirical analysis of model initialisation and architecture, synthetic training data, preference optimisation with synthetic data and tweaking all other components. We empirically demonstrate that this training recipe also scales well with more compute getting results on par with leading SLMs in a fraction of the compute cost. We hope these insights will make SLM training and research more accessible. In the context of SLM scaling laws, our results far outperform predicted compute optimal performance, giving an optimistic view to SLM feasibility. See code, data, models, samples at - this https URL . 

**Abstract (ZH)**: 我们介绍了一种名为Slam的训模方案，该方案可以在24小时内于单个学术级GPU上训练高质量的语音语言模型（SLMs）。我们通过实证分析模型初始化、架构，使用合成训练数据进行偏好优化，并调整其他所有组件来实现这一目标。实证研究表明，这种训练方案在具备更多计算资源的情况下，能够在较低的成本下达到与领先SLMs相当的效果。我们希望这些洞见能够使SLM的训练与研究更加易于获取。在SLM扩展规律的背景下，我们的结果大幅超越了预期的计算最优性能，为SLM的可行性提供了乐观的展望。详细代码、数据、模型和示例请参见：[这个链接](this https URL)。 

---
# A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos 

**Title (ZH)**: 一个捕鼠器：通过迭代混沌链对大型推理模型进行 Jailbreak 欺骗 

**Authors**: Yang Yao, Xuan Tong, Ruofan Wang, Yixu Wang, Lujundong Li, Liang Liu, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15806)  

**Abstract**: Large Reasoning Models (LRMs) have significantly advanced beyond traditional Large Language Models (LLMs) with their exceptional logical reasoning capabilities, yet these improvements introduce heightened safety risks. When subjected to jailbreak attacks, their ability to generate more targeted and organized content can lead to greater harm. Although some studies claim that reasoning enables safer LRMs against existing LLM attacks, they overlook the inherent flaws within the reasoning process itself. To address this gap, we propose the first jailbreak attack targeting LRMs, exploiting their unique vulnerabilities stemming from the advanced reasoning capabilities. Specifically, we introduce a Chaos Machine, a novel component to transform attack prompts with diverse one-to-one mappings. The chaos mappings iteratively generated by the machine are embedded into the reasoning chain, which strengthens the variability and complexity and also promotes a more robust attack. Based on this, we construct the Mousetrap framework, which makes attacks projected into nonlinear-like low sample spaces with mismatched generalization enhanced. Also, due to the more competing objectives, LRMs gradually maintain the inertia of unpredictable iterative reasoning and fall into our trap. Success rates of the Mousetrap attacking o1-mini, claude-sonnet and gemini-thinking are as high as 96%, 86% and 98% respectively on our toxic dataset Trotter. On benchmarks such as AdvBench, StrongREJECT, and HarmBench, attacking claude-sonnet, well-known for its safety, Mousetrap can astonishingly achieve success rates of 87.5%, 86.58% and 93.13% respectively. Attention: This paper contains inappropriate, offensive and harmful content. 

**Abstract (ZH)**: 大型推理模型（LRMs）在逻辑推理能力方面远远超越了传统的大型语言模型（LLMs），但这些进步也带来了更大的安全风险。当受到监狱突破攻击时，它们生成更精准和组织化的内容的能力可能会导致更大的危害。尽管一些研究声称推理能够使LRMs在面对现有LLM攻击时更安全，它们忽视了推理过程本身固有的缺陷。为解决这一问题，我们提出了第一个针对LRMs的监狱突破攻击，利用其独特的弱点，这些弱点源于高级推理能力。具体来说，我们引入了混沌机器，这是一种新颖的组件，可以将具有多种一对一映射的攻击提示进行转换。机器生成的混沌映射被嵌入到推理链中，增强了变化性和复杂性，同时也促进了更稳健的攻击。基于此，我们构建了Mousetrap框架，该框架将攻击投影到非线性的低样本空间中，并增强了不匹配泛化的效果。此外，由于有更多的竞争目标，LRMs逐渐保持了不可预测的迭代推理的惯性，并落入我们的陷阱。根据我们的有毒数据集Trotter，Mousetrap攻击o1-mini、claude-sonnet和gemini-thinking的成功率分别为96%、86%和98%。在AdvBench、StrongREJECT和HarmBench等基准测试中，针对以其安全性闻名的claude-sonnet，Mousetrap攻击的成功率分别达到了87.5%、86.58%和93.13%。请注意：本文包含不适当、冒犯性和有害的内容。 

---
# An explainable transformer circuit for compositional generalization 

**Title (ZH)**: 可解释的变换器电路以实现组合泛化 

**Authors**: Cheng Tang, Brenden Lake, Mehrdad Jazayeri  

**Link**: [PDF](https://arxiv.org/pdf/2502.15801)  

**Abstract**: Compositional generalization-the systematic combination of known components into novel structures-remains a core challenge in cognitive science and machine learning. Although transformer-based large language models can exhibit strong performance on certain compositional tasks, the underlying mechanisms driving these abilities remain opaque, calling into question their interpretability. In this work, we identify and mechanistically interpret the circuit responsible for compositional induction in a compact transformer. Using causal ablations, we validate the circuit and formalize its operation using a program-like description. We further demonstrate that this mechanistic understanding enables precise activation edits to steer the model's behavior predictably. Our findings advance the understanding of complex behaviors in transformers and highlight such insights can provide a direct pathway for model control. 

**Abstract (ZH)**: 成分泛化——即系统地将已知组件组合成新的结构——仍然是认知科学和机器学习中的一个核心挑战。尽管基于变压器的大型语言模型在某些成分任务上表现出很强的能力，但驱动这些能力的内部机制仍然模糊不清，这使得这些模型的可解释性受到质疑。在本项工作中，我们识别并从机理上解释了一个紧凑型变压器中负责成分归纳的电路。通过因果消融实验验证该电路，并使用类似于程序的描述形式化其操作机制。进一步的研究表明，这种机理理解使得我们可以对模型的行为进行精确的激活编辑，以可预测的方式引导模型的行为。我们的发现推进了对变压器复杂行为的理解，并突显出这些洞察可以直接成为模型控制的途径。 

---
# Pruning as a Defense: Reducing Memorization in Large Language Models 

**Title (ZH)**: 修剪作为防御手段：减少大型语言模型的记忆化 

**Authors**: Mansi Gupta, Nikhar Waghela, Sarthak Gupta, Shourya Goel, Sanjif Shanmugavelu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15796)  

**Abstract**: Large language models have been shown to memorize significant portions of their training data, which they can reproduce when appropriately prompted. This work investigates the impact of simple pruning techniques on this behavior. Our findings reveal that pruning effectively reduces the extent of memorization in LLMs, demonstrating its potential as a foundational approach for mitigating membership inference attacks. 

**Abstract (ZH)**: 大型语言模型已被证明会记住大量训练数据，并在适当提示下重现这些数据。本研究探讨了简单剪枝技术对此现象的影响。我们的研究发现，剪枝有效地减少了大型语言模型中的记忆程度，展示了其作为减轻成员推理攻击基础方法的潜力。 

---
# Lean-ing on Quality: How High-Quality Data Beats Diverse Multilingual Data in AutoFormalization 

**Title (ZH)**: 依赖高质量数据：高质数据如何在自动形式化中胜过多样化的多语言数据 

**Authors**: Willy Chan, Michael Souliman, Jakob Nordhagen, Brando Miranda, Elyas Obbad, Kai Fronsdal Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.15795)  

**Abstract**: Autoformalization, the process of transforming informal mathematical language into formal specifications and proofs remains a difficult task for state-of-the-art (large) language models. Existing works point to competing explanations for the performance gap. To this end, we introduce a novel methodology that leverages back-translation with hand-curated prompts to enhance the mathematical capabilities of language models, particularly addressing the challenge posed by the scarcity of labeled data. Specifically, we evaluate three primary variations of this strategy: (1) on-the-fly (online) backtranslation, (2) distilled (offline) backtranslation with few-shot amplification, and (3) line-by-line proof analysis integrated with proof state information. Each variant is designed to optimize data quality over quantity, focusing on the high fidelity of generated proofs rather than sheer data scale. Our findings provide evidence that employing our proposed approaches to generate synthetic data, which prioritizes quality over volume, improves the Autoformalization performance of LLMs as measured by standard benchmarks such as ProofNet. Crucially, our approach outperforms pretrained models using a minimal number of tokens. We also show, through strategic prompting and backtranslation, that our approaches surpass the performance of fine-tuning with extensive multilingual datasets such as MMA on ProofNet with only 1/150th of the tokens. Taken together, our methods show a promising new approach to significantly reduce the resources required to formalize proofs, thereby accelerating AI for math. 

**Abstract (ZH)**: 自动形式化过程，即将非正式的数学语言转换为形式化规格和证明，仍然是最先进的（大型）语言模型面临的难题。现有研究指出了性能差距的多种可能解释。为此，我们介绍了一种新的方法，该方法结合了基于人工精选提示的回译技术，以增强语言模型的数学能力，特别是解决标定数据稀缺带来的挑战。具体而言，我们评估了此策略的三种主要变体：(1) 在线回译（2) 精炼的少量样本增强回译 (3) 集成证明状态信息的逐行证明分析。每种变体都旨在优化数据质量而非数量，专注于生成证明的高准确性而非单纯的数据量。我们的研究结果提供了证据，表明采用我们提出的生成合成数据的方法，优先考虑质量和数量，能显著提升大型语言模型（LLM）在标准基准（如ProofNet）上的自动形式化性能。至关重要的是，我们的方法仅用最少的令牌数就能超越预训练模型。我们还展示了通过战略性提示和回译，我们的方法仅使用MMA数据集1/150的令牌数，即可超越使用大量多语言数据集（如MMA）微调的性能在ProofNet上的表现。综上，我们的方法提供了显著降低形式化证明所需资源的新途径，从而加速数学AI的发展。 

---
# Self-Supervised Transformers as Iterative Solution Improvers for Constraint Satisfaction 

**Title (ZH)**: 自我监督变换器作为约束满足问题迭代解改进器 

**Authors**: Yudong W. Xu, Wenhao Li, Scott Sanner, Elias B. Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2502.15794)  

**Abstract**: We present a Transformer-based framework for Constraint Satisfaction Problems (CSPs). CSPs find use in many applications and thus accelerating their solution with machine learning is of wide interest. Most existing approaches rely on supervised learning from feasible solutions or reinforcement learning, paradigms that require either feasible solutions to these NP-Complete CSPs or large training budgets and a complex expert-designed reward signal. To address these challenges, we propose ConsFormer, a self-supervised framework that leverages a Transformer as a solution refiner. ConsFormer constructs a solution to a CSP iteratively in a process that mimics local search. Instead of using feasible solutions as labeled data, we devise differentiable approximations to the discrete constraints of a CSP to guide model training. Our model is trained to improve random assignments for a single step but is deployed iteratively at test time, circumventing the bottlenecks of supervised and reinforcement learning. Our method can tackle out-of-distribution CSPs simply through additional iterations. 

**Abstract (ZH)**: 我们提出了一种基于Transformer的框架，用于解决约束 satisfaction 问题（CSPs）。CSPs 在许多应用中都有所应用，因此使用机器学习加速其求解是一个广泛感兴趣的研究方向。现有的大多数方法依赖于可行解的监督学习或强化学习，这些方法要么需要可行的解来解决这些NP完全的CSPs，要么需要大量的训练预算和一个复杂的手工设计的奖励信号。为了解决这些问题，我们提出了一种自监督的框架——ConsFormer，它利用Transformer作为一个解决方案的细化器。ConsFormer 通过一个类似于局部搜索的过程，迭代地构建一个CSP的解。我们不是使用可行解作为标签数据，而是设计了CSP离散约束的可微近似来指导模型训练。我们的模型被训练以在单步中改进随机分配，但在测试时则迭代部署，从而绕过了监督学习和强化学习中的瓶颈。我们的方法仅通过额外的迭代就能处理来自分布外的CSPs。 

---
# Rotate, Clip, and Partition: Towards W2A4KV4 Quantization by Integrating Rotation and Learnable Non-uniform Quantizer 

**Title (ZH)**: 旋转、裁剪和分区：通过结合旋转和可学习的非均匀量化器朝着W2A4KV4量化方向的努力 

**Authors**: Euntae Choi, Sumin Song, Woosang Lim, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2502.15779)  

**Abstract**: We propose Rotate, Clip, and Partition (RCP), a quantization-aware training (QAT) approach that first realizes extreme compression of LLMs with W2A4KV4(2-bit weight, 4-bit activation, and 4-bit KV cache) configuration. RCP integrates recent rotation techniques with a novel non-uniform weight quantizer design, by quantitatively analyzing the impact of random rotation on 2-bit weight quantization. Our weight quantizer features Learnable Direct Partitioning (LDP), which introduces learnable parameters to directly learn non-uniform intervals jointly with LLM weights. We also present a specialized GPU kernel that supports GEMV on non-uniform W2A4. Experiments show that RCP can compress LLaMA-2-7B to W2A4KV4 with a loss of only 2.84 WikiText2 ppl and 5.29 times reduced memory footprint. Furthermore, RCP can quantize challenging mobile-targeted LLaMA-3.2 models and domain-specific WizardCoder-7B and MetaMath-7B with no critical problems such as convergence failure and repetition. Code will be made available at blind_review. 

**Abstract (ZH)**: 我们提出了一种量化感知训练（QAT）方法——Rotate、Clip and Partition (RCP)，该方法首先通过W2A4KV4（2位权重、4位激活和4位KV缓存）配置实现了对大规模语言模型（LLM）的极度压缩。RCP 将最近的旋转技术与一种新颖的非均匀权重量化器设计相结合，通过定量分析随机旋转对2位权重量化的影响。我们的权重量化器集成了可学习直接分区（LDP），通过引入可学习参数直接与LLM权重一起学习非均匀区间。我们还提供了一种专门针对非均匀W2A4计算的GPU内核，支持GEMV（通用矩阵向量乘法）操作。实验结果显示，RCP 可以将LLaMA-2-7B压缩为W2A4KV4，仅损失2.84 WikiText2 PPL，并且内存占用减少了5.29倍。此外，RCP 能够对针对移动设备的LLaMA-3.2模型以及特定领域的WizardCoder-7B和MetaMath-7B进行量化，而不会遇到如收敛失败和重复等问题。代码将在盲审阶段公开。 

---
# Learning to Reason from Feedback at Test-Time 

**Title (ZH)**: 在测试时从反馈中学习推理 

**Authors**: Yanyang Li, Michael Lyu, Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15771)  

**Abstract**: Solving complex tasks in a single attempt is challenging for large language models (LLMs). Iterative interaction with the environment and feedback is often required to achieve success, making effective feedback utilization a critical topic. Existing approaches either struggle with length generalization or rely on naive retries without leveraging prior information. In this paper, we introduce FTTT, a novel paradigm that formulates feedback utilization as an optimization problem at test time. Additionally, we propose a learnable test-time optimizer, OpTune, to effectively exploit feedback. Experiments on two LLMs across four reasoning datasets demonstrate that FTTT and OpTune achieve superior scalability and performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在一次性解决复杂任务方面具有挑战性，通常需要与环境进行迭代交互并利用反馈才能成功，因此有效的反馈利用成为一个关键问题。现有方法要么在长度泛化方面存在问题，要么依赖于简单的重试而不利用先验信息。在本文中，我们引入了FTTT（Feedback Treatment as Testing Time Optimization），这是一种新颖的范式，在测试时将反馈利用形式化为一个优化问题。此外，我们提出了一种可学习的测试时优化器OpTune，以有效利用反馈。实验结果显示，FTTT和OpTune在四个推理数据集上的两个LLM中表现出更好的可扩展性和性能。 

---
# Detection of LLM-Generated Java Code Using Discretized Nested Bigrams 

**Title (ZH)**: 使用离散嵌套双字组检测LLM生成的Java代码 

**Authors**: Timothy Paek, Chilukuri Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15740)  

**Abstract**: Large Language Models (LLMs) are currently used extensively to generate code by professionals and students, motivating the development of tools to detect LLM-generated code for applications such as academic integrity and cybersecurity. We address this authorship attribution problem as a binary classification task along with feature identification and extraction. We propose new Discretized Nested Bigram Frequency features on source code groups of various sizes. Compared to prior work, improvements are obtained by representing sparse information in dense membership bins. Experimental evaluation demonstrated that our approach significantly outperformed a commonly used GPT code-detection API and baseline features, with accuracy exceeding 96% compared to 72% and 79% respectively in detecting GPT-rewritten Java code fragments for 976 files with GPT 3.5 and GPT4 using 12 features. We also outperformed three prior works on code author identification in a 40-author dataset. Our approach scales well to larger data sets, and we achieved 99% accuracy and 0.999 AUC for 76,089 files and over 1,000 authors with GPT 4o using 227 features. 

**Abstract (ZH)**: 大规模语言模型（LLMs）目前被广泛用于生成代码，无论是专业人士还是学生都在使用，这推动了检测LLM生成代码的相关工具的发展，尤其是在学术诚信和网络安全等领域。我们将作者归属问题视为一个二分类任务，并结合特征识别和提取来进行研究。我们提出了新的离散嵌套双字频特征，应用于不同大小的源代码组。与以前的工作相比，通过将稀疏信息表示为密集成员区间，我们获得了改进。实验评估表明，我们的方法在检测GPT重写过的Java代码片段方面显著优于常用的GPT代码检测API和基线特征，准确率超过了96%，而对于使用GPT 3.5和GPT4分别对976个文件进行检测时，常用的GPT代码检测API和基线特征的准确率分别为72%和79%。我们还在一个包含40名作者的数据集上优于之前三种方法的代码作者识别。我们的方法在更大数据集上具有良好的扩展性，使用GPT 4o和227个特征时，对于76,089个文件和超过1,000个作者，我们实现了99%的准确率和0.999的AUC。 

---
# Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation 

**Title (ZH)**: Cache-Craft: 管理块缓存以实现高效检索增强生成 

**Authors**: Shubham Agarwal, Sai Sundaresan, Subrata Mitra, Debabrata Mahapatra, Archit Gupta, Rounak Sharma, Nirmal Joshua Kapu, Tong Yu, Shiv Saini  

**Link**: [PDF](https://arxiv.org/pdf/2502.15734)  

**Abstract**: Retrieval-Augmented Generation (RAG) is often used with Large Language Models (LLMs) to infuse domain knowledge or user-specific information. In RAG, given a user query, a retriever extracts chunks of relevant text from a knowledge base. These chunks are sent to an LLM as part of the input prompt. Typically, any given chunk is repeatedly retrieved across user questions. However, currently, for every question, attention-layers in LLMs fully compute the key values (KVs) repeatedly for the input chunks, as state-of-the-art methods cannot reuse KV-caches when chunks appear at arbitrary locations with arbitrary contexts. Naive reuse leads to output quality degradation. This leads to potentially redundant computations on expensive GPUs and increases latency. In this work, we propose Cache-Craft, a system for managing and reusing precomputed KVs corresponding to the text chunks (we call chunk-caches) in RAG-based systems. We present how to identify chunk-caches that are reusable, how to efficiently perform a small fraction of recomputation to fix the cache to maintain output quality, and how to efficiently store and evict chunk-caches in the hardware for maximizing reuse while masking any overheads. With real production workloads as well as synthetic datasets, we show that Cache-Craft reduces redundant computation by 51% over SOTA prefix-caching and 75% over full recomputation. Additionally, with continuous batching on a real production workload, we get a 1.6X speed up in throughput and a 2X reduction in end-to-end response latency over prefix-caching while maintaining quality, for both the LLaMA-3-8B and LLaMA-3-70B models. 

**Abstract (ZH)**: 检索增强生成（RAG）通常与大型语言模型（LLMs）结合使用，以注入领域知识或用户特定信息。在RAG中，给定一个用户查询时，检索器会从知识库中提取相关文本片段。这些片段作为输入提示的一部分发送给LLM。通常情况下，任何给定片段在不同用户的问题中会被重复检索。然而，目前，对于每个问题，LLM中的注意力层会为输入片段完全重新计算键值（KVs），因为最先进的方法无法在片段出现在任意位置且上下文任意时复用KV缓存。简单的复用会导致输出质量下降。这导致在昂贵的GPU上进行潜在的重复计算，并增加延迟。在本文中，我们提出了Cache-Craft，这是一种用于管理和复用RAG系统中与文本片段对应的预计算键值（我们称之为片段缓存）的系统。我们介绍了如何识别可复用的片段缓存，如何高效地进行少量重计算以固定缓存并保持输出质量，以及如何在硬件中高效地存储和移除片段缓存以最大化复用性能并隐藏任何开销。使用实际生产负载和合成数据集，我们证明Cache-Craft相比于最新的人工序列缓存减少了51%的冗余计算，相比于全量重计算减少了75%。此外，通过在实际生产负载中持续批量处理，对于LLaMA-3-8B和LLaMA-3-70B模型，我们相比序列缓存获得了1.6倍的吞吐量提升和2倍的端到端响应延迟减少，同时保持了质量。 

---
# TrustDataFilter:Leveraging Trusted Knowledge Base Data for More Effective Filtering of Unknown Information 

**Title (ZH)**: TrustDataFilter：利用可信知识库数据进行更有效的未知信息过滤 

**Authors**: Jinghong Zhang, Yidong Cui, Weiling Wang, Xianyou Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.15714)  

**Abstract**: With the advancement of technology and changes in the market, the demand for the construction of domain-specific knowledge bases has been increasing, either to improve model performance or to promote enterprise innovation and competitiveness. The construction of domain-specific knowledge bases typically relies on web crawlers or existing industry databases, leading to problems with accuracy and consistency of the data. To address these challenges, we considered the characteristics of domain data, where internal knowledge is interconnected, and proposed the Self-Natural Language Inference Data Filtering (self-nli-TDF) framework. This framework compares trusted filtered knowledge with the data to be filtered, deducing the reasoning relationship between them, thus improving filtering performance. The framework uses plug-and-play large language models for trustworthiness assessment and employs the RoBERTa-MNLI model from the NLI domain for reasoning. We constructed three datasets in the domains of biology, radiation, and science, and conducted experiments using RoBERTa, GPT3.5, and the local Qwen2 model. The experimental results show that this framework improves filter quality, producing more consistent and reliable filtering results. 

**Abstract (ZH)**: 随着技术的发展和市场的变化，对领域专用知识库的需求日益增加，这既是为了改善模型性能，也是为了促进企业的创新和竞争力。构建领域专用知识库通常依赖于网络爬虫或现有的行业数据库，但这些方法会导致数据准确性和一致性的问题。为了应对这些挑战，我们考虑了领域数据的特点，即内部知识相互关联，提出了Self-Natural Language Inference Data Filtering（自自然语言推理数据过滤，简称self-nli-TDF）框架。该框架将可信过滤后的知识与待过滤数据进行对比，从而推导它们之间的推理关系，以提高过滤性能。该框架利用插件式大型语言模型进行可信度评估，并使用NLI领域中的RoBERTa-MNLI模型进行推理。我们在生物学、辐射和科学等领域构建了三个数据集，并使用了RoBERTa、GPT3.5和本地的Qwen2模型进行了实验。实验结果表明，该框架能够提高过滤质量，生成更为一致和可靠的结果。 

---
# Large language models streamline automated systematic review: A preliminary study 

**Title (ZH)**: 大规模语言模型简化自动化系统评价：初步研究 

**Authors**: Xi Chen, Xue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15702)  

**Abstract**: Large Language Models (LLMs) have shown promise in natural language processing tasks, with the potential to automate systematic reviews. This study evaluates the performance of three state-of-the-art LLMs in conducting systematic review tasks. We assessed GPT-4, Claude-3, and Mistral 8x7B across four systematic review tasks: study design formulation, search strategy development, literature screening, and data extraction. Sourced from a previously published systematic review, we provided reference standard including standard PICO (Population, Intervention, Comparison, Outcome) design, standard eligibility criteria, and data from 20 reference literature. Three investigators evaluated the quality of study design and eligibility criteria using 5-point Liker Scale in terms of accuracy, integrity, relevance, consistency and overall performance. For other tasks, the output is defined as accurate if it is the same as the reference standard. Search strategy performance was evaluated through accuracy and retrieval efficacy. Screening accuracy was assessed for both abstracts screening and full texts screening. Data extraction accuracy was evaluated across 1,120 data points comprising 3,360 individual fields. Claude-3 demonstrated superior overall performance in PICO design. In search strategy formulation, GPT-4 and Claude-3 achieved comparable accuracy, outperforming Mistral. For abstract screening, GPT-4 achieved the highest accuracy, followed by Mistral and Claude-3. In data extraction, GPT-4 significantly outperformed other models. LLMs demonstrate potential for automating systematic review tasks, with GPT-4 showing superior performance in search strategy formulation, literature screening and data extraction. These capabilities make them promising assistive tools for researchers and warrant further development and validation in this field. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理任务中展现了潜力，并有可能自动化系统评价。本研究评估了三种最先进的LLMs在系统评价任务中的性能。我们分别在四个系统评价任务上测试了GPT-4、Claude-3和Mistral 8x7B：研究设计制定、检索策略开发、文献筛选和数据提取。这些任务的数据来源于一篇已发表的系统评价，其中包括标准化的PICO（Population、Intervention、Comparison、Outcome）设计、标准筛选标准和20篇参考文献的数据。三位核查员使用5点李克特量表从准确性、完整性、相关性、一致性和总体性能五个方面评估了研究设计和筛选标准的质量。对于其他任务，输出定义为准确，如果与参考标准完全一致。检索策略的性能通过准确性和召回率进行评估。文献筛选准确性分别评估了摘要筛选和全文筛选。数据提取准确性评估了1120个数据点，共计3360个单独字段。Claude-3在PICO设计方面表现出色。在检索策略制定中，GPT-4和Claude-3的准确性相当，且优于Mistral。在摘要筛选中，GPT-4的准确性最高，其次是Mistral和Claude-3。在数据提取中，GPT-4显著优于其他模型。大型语言模型展示了在系统评价任务中自动化的潜力，其中GPT-4在检索策略制定、文献筛选和数据提取方面表现出更佳的性能。这些能力使它们成为研究人员的有希望的辅助工具，并需要在该领域进一步开发和验证。 

---
# Political Events using RAG with LLMs 

**Title (ZH)**: 使用LLM的RAG进行政治事件处理 

**Authors**: Muhammad Arslan, Saba Munawar, Christophe Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2502.15701)  

**Abstract**: In the contemporary digital landscape, media content stands as the foundation for political news analysis, offering invaluable insights sourced from various channels like news articles, social media updates, speeches, and reports. Natural Language Processing (NLP) has revolutionized Political Information Extraction (IE), automating tasks such as Event Extraction (EE) from these diverse media outlets. While traditional NLP methods often necessitate specialized expertise to build rule-based systems or train machine learning models with domain-specific datasets, the emergence of Large Language Models (LLMs) driven by Generative Artificial Intelligence (GenAI) presents a promising alternative. These models offer accessibility, alleviating challenges associated with model construction from scratch and reducing the dependency on extensive datasets during the training phase, thus facilitating rapid implementation. However, challenges persist in handling domain-specific tasks, leading to the development of the Retrieval-Augmented Generation (RAG) framework. RAG enhances LLMs by integrating external data retrieval, enriching their contextual understanding, and expanding their knowledge base beyond pre-existing training data. To illustrate RAG's efficacy, we introduce the Political EE system, specifically tailored to extract political event information from news articles. Understanding these political insights is essential for remaining informed about the latest political advancements, whether on a national or global scale. 

**Abstract (ZH)**: 在当今的数字景观中，媒体内容是政治新闻分析的基石，为从新闻文章、社交媒体更新、演讲和报告等多种渠道提供宝贵见解。自然语言处理（NLP）已经彻底改变了政治信息提取（PIE），实现了从多种媒体渠道提取事件（EE）等任务的自动化。传统NLP方法通常需要专门的知识来构建基于规则的系统或使用特定领域的数据集训练机器学习模型，而生成人工智能（GenAI）驱动的大型语言模型（LLM）的出现提供了有前景的替代方案。这些模型提高了可访问性，缓解了从头构建模型的挑战，并减少了在训练过程中对大量数据集的依赖，从而促进了快速部署。然而，这些模型仍面临处理特定领域任务的挑战，导致了检索增强生成（RAG）框架的开发。RAG通过结合外部数据检索，丰富了语言模型的上下文理解，并扩展了其知识库，使其超越了现有的训练数据。为了说明RAG的有效性，我们介绍了专门用于从新闻文章中提取政治事件信息的政治事件提取系统。了解这些政治见解对于了解最新的政治进展至关重要，无论是在国家层面还是全球层面。 

---
# Sustainable Digitalization of Business with Multi-Agent RAG and LLM 

**Title (ZH)**: 商务的可持续数字化转型：基于多代理RAG和大语言模型的方法 

**Authors**: Muhammad Arslan, Saba Munawar, Christophe Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2502.15700)  

**Abstract**: Businesses heavily rely on data sourced from various channels like news articles, financial reports, and consumer reviews to drive their operations, enabling informed decision-making and identifying opportunities. However, traditional manual methods for data extraction are often time-consuming and resource-intensive, prompting the adoption of digital transformation initiatives to enhance efficiency. Yet, concerns persist regarding the sustainability of such initiatives and their alignment with the United Nations (UN)'s Sustainable Development Goals (SDGs). This research aims to explore the integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) as a sustainable solution for Information Extraction (IE) and processing. The research methodology involves reviewing existing solutions for business decision-making, noting that many systems require training new machine learning models, which are resource-intensive and have significant environmental impacts. Instead, we propose a sustainable business solution using pre-existing LLMs that can work with diverse datasets. We link domain-specific datasets to tailor LLMs to company needs and employ a Multi-Agent architecture to divide tasks such as information retrieval, enrichment, and classification among specialized agents. This approach optimizes the extraction process and improves overall efficiency. Through the utilization of these technologies, businesses can optimize resource utilization, improve decision-making processes, and contribute to sustainable development goals, thereby fostering environmental responsibility within the corporate sector. 

**Abstract (ZH)**: 企业广泛依靠来自新闻文章、财务报告和消费者评价等多种渠道的数据来驱动其运营，从而实现基于数据的决策并识别机会。然而，传统的人工数据提取方法往往耗费时间且资源密集，这促使企业采用数字化转型措施以提高效率。然而，人们对这些措施的可持续性及其与联合国可持续发展目标（SDGs）的契合度仍存有担忧。本研究旨在探讨将大型语言模型（LLMs）与检索增强生成（RAG）相结合作为信息提取（IE）和处理的可持续解决方案。研究方法包括审查现有的企业决策支持系统，发现许多系统需要训练新的机器学习模型，这不仅资源密集，而且对环境造成显著影响。相反，我们提出了一种基于现有LLMs的可持续业务解决方案，这些模型能够与各种数据集进行有效合作。我们将特定领域的数据集与公司的需求联系起来，利用多代理架构将信息检索、丰富和分类等任务分配给专一的代理。这种方法优化了数据提取过程并增强了整体效率。通过利用这些技术，企业可以优化资源配置，改进决策流程，并为可持续发展目标作出贡献，从而在企业界促进环境责任。 

---
# ACL-rlg: A Dataset for Reading List Generation 

**Title (ZH)**: ACL-rlg：一个阅读列表生成数据集 

**Authors**: Julien Aubert-Béduchaud, Florian Boudin, Béatrice Daille, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2502.15692)  

**Abstract**: Familiarizing oneself with a new scientific field and its existing literature can be daunting due to the large amount of available articles. Curated lists of academic references, or reading lists, compiled by experts, offer a structured way to gain a comprehensive overview of a domain or a specific scientific challenge. In this work, we introduce ACL-rlg, the largest open expert-annotated reading list dataset. We also provide multiple baselines for evaluating reading list generation and formally define it as a retrieval task. Our qualitative study highlights the fact that traditional scholarly search engines and indexing methods perform poorly on this task, and GPT-4o, despite showing better results, exhibits signs of potential data contamination. 

**Abstract (ZH)**: 熟悉新的科学领域及其现有文献可能会因可用文章数量庞大而显得 daunting。由专家编写的精选学术参考文献列表或阅读列表，提供了一种结构化的途径，以全面了解一个领域或特定的科学挑战。在本工作中，我们介绍了ACL-rlg，这是迄今为止最大的开放专家注释阅读列表数据集。我们还提供了多种基线方法来评估阅读列表生成的效果，并正式将其定义为检索任务。我们的定性研究表明，传统的学术搜索引擎和索引方法在这种任务上表现较差，尽管GPT-4o显示出更好的结果，但也表现出潜在的数据污染迹象。 

---
# Level-Navi Agent: A Framework and benchmark for Chinese Web Search Agents 

**Title (ZH)**: Level-Navi 代理：中文网络搜索代理的框架与基准 

**Authors**: Chuanrui Hu, Shichong Xie, Baoxin Wang, Bin Chen, Xiaofeng Cong, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15690)  

**Abstract**: Large language models (LLMs), adopted to understand human language, drive the development of artificial intelligence (AI) web search agents. Compared to traditional search engines, LLM-powered AI search agents are capable of understanding and responding to complex queries with greater depth, enabling more accurate operations and better context recognition. However, little attention and effort has been paid to the Chinese web search, which results in that the capabilities of open-source models have not been uniformly and fairly evaluated. The difficulty lies in lacking three aspects: an unified agent framework, an accurately labeled dataset, and a suitable evaluation metric. To address these issues, we propose a general-purpose and training-free web search agent by level-aware navigation, Level-Navi Agent, accompanied by a well-annotated dataset (Web24) and a suitable evaluation metric. Level-Navi Agent can think through complex user questions and conduct searches across various levels on the internet to gather information for questions. Meanwhile, we provide a comprehensive evaluation of state-of-the-art LLMs under fair settings. To further facilitate future research, source code is available at Github. 

**Abstract (ZH)**: 大型语言模型（LLMs）被用于理解人类语言，推动了人工智能（AI）网络搜索代理的发展。与传统的搜索引擎相比，由LLM驱动的AI搜索代理能够更好地理解并回应复杂的查询，具备更深层次的理解和响应能力，从而实现更准确的操作和更好的上下文识别。然而，在中国网络搜索方面，关注和努力明显不足，导致开源模型的能力未能得到统一和公平的评估。这一问题的难点在于缺乏三个方面：统一的代理框架、准确标注的数据集以及合适的评估指标。为了解决这些问题，我们提出了一种基于层次感知导航的一般用途且无需训练的网络搜索代理——Level-Navi代理，并提供了一个高质量标注的数据集（Web24）和合适的评估指标。Level-Navi代理能够通过层次感知的方式思考复杂用户的问题，并在互联网的不同层次上进行搜索，以收集回答问题所需的信息。同时，我们提供了在公平条件下对最先进的LLM的全面评估。为促进未来的研究，相关源代码已发布在GitHub上。 

---
# V-SQL: A View-based Two-stage Text-to-SQL Framework 

**Title (ZH)**: V-SQL：一种基于视图的两阶段文本到SQL框架 

**Authors**: Zeshun You, Jiebin Yao, Dong Cheng, Zhiwei Wen, Zhiliang Lu, Xianyi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15686)  

**Abstract**: The text-to-SQL task aims to convert natural language into Structured Query Language (SQL) without bias. Recently, text-to-SQL methods based on large language models (LLMs) have garnered significant attention. The core of mainstream text-to-SQL frameworks is schema linking, which aligns user queries with relevant tables and columns in the database. Previous methods focused on schema linking while neglecting to enhance LLMs' understanding of database schema. The complex coupling relationships between tables in the database constrain the SQL generation capabilities of LLMs. To tackle this issue, this paper proposes a simple yet effective strategy called view-based schema. This strategy aids LLMs in understanding the database schema by decoupling tightly coupled tables into low-coupling views. We then introduce V-SQL, a view-based two-stage text-to-SQL framework. V-SQL involves the view-based schema strategy to enhance LLMs' understanding of database schema. Results on the authoritative datasets Bird indicate that V-SQL achieves competitive performance compared to existing state-of-the-art methods. 

**Abstract (ZH)**: 文本到SQL的任务旨在无偏见地将自然语言转换为结构化查询语言（SQL）。近年来，基于大规模语言模型（LLMs）的文本到SQL方法受到了广泛关注。主流的文本到SQL框架的核心是模式链接，该过程将用户查询与数据库中的相关表和列对齐。之前的方法主要关注模式链接，而忽视了增强LLM对数据库模式的理解。数据库中表之间的复杂耦合关系限制了LLMs生成SQL的能力。为解决这一问题，本论文提出了一种简单而有效的策略，称为基于视图的模式。该策略通过将紧密耦合的表分解为低耦合视图，帮助LLM更好地理解数据库模式。我们随后介绍了V-SQL，这是一种基于视图的两阶段文本到SQL框架。V-SQL利用基于视图的模式策略来增强LLM对数据库模式的理解。在权威数据集Bird上的结果显示，V-SQL的性能与现有最先进的方法相当。 

---
