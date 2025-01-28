# LUCY: Linguistic Understanding and Control Yielding Early Stage of Her 

**Title (ZH)**: LUCY：语言理解和控制在她早期阶段的作用 

**Authors**: Heting Gao, Hang Shao, Xiong Wang, Chaofan Qiu, Yunhang Shen, Siqi Cai, Yuchen Shi, Zihan Xu, Zuwei Long, Yike Zhang, Shaoqi Dong, Chaoyou Fu, Ke Li, Long Ma, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.16327)  

**Abstract**: The film Her features Samantha, a sophisticated AI audio agent who is capable of understanding both linguistic and paralinguistic information in human speech and delivering real-time responses that are natural, informative and sensitive to emotional subtleties. Moving one step toward more sophisticated audio agent from recent advancement in end-to-end (E2E) speech systems, we propose LUCY, a E2E speech model that (1) senses and responds to user's emotion, (2) deliver responses in a succinct and natural style, and (3) use external tool to answer real-time inquiries. Experiment results show that LUCY is better at emotion control than peer models, generating emotional responses based on linguistic emotional instructions and responding to paralinguistic emotional cues. Lucy is also able to generate responses in a more natural style, as judged by external language models, without sacrificing much performance on general question answering. Finally, LUCY can leverage function calls to answer questions that are out of its knowledge scope. 

**Abstract (ZH)**: 电影《她》中的萨曼莎是一个复杂的AI语音代理，能够理解人类语言和副语言信息，并提供自然、信息丰富且对情感细微变化敏感的即时回应。从近年来端到端（E2E）语音系统的发展中进一步提升语音代理的复杂性，我们提出了一种E2E语音模型LUCY，该模型具有以下特征：（1）感知并响应用户的情感；（2）以简洁且自然的方式提供回应；（3）使用外部工具即时回答查询。实验结果表明，LUCY在情感控制方面优于其他模型，能够根据语言情感指令生成相应的情感回应，并响应副语言情感提示。LUCY还能在不牺牲一般问题回答性能的情况下，以更自然的风格生成回应。最后，LUCY可以通过功能调用回答超出其知识范围的问题。 

---
# RAPID: Retrieval-Augmented Parallel Inference Drafting for Text-Based Video Event Retrieval 

**Title (ZH)**: RAPID：基于检索增强并行推理的文本视频事件检索草稿生成方法 

**Authors**: Long Nguyen, Huy Nguyen, Bao Khuu, Huy Luu, Huy Le, Tuan Nguyen, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16303)  

**Abstract**: Retrieving events from videos using text queries has become increasingly challenging due to the rapid growth of multimedia content. Existing methods for text-based video event retrieval often focus heavily on object-level descriptions, overlooking the crucial role of contextual information. This limitation is especially apparent when queries lack sufficient context, such as missing location details or ambiguous background elements. To address these challenges, we propose a novel system called RAPID (Retrieval-Augmented Parallel Inference Drafting), which leverages advancements in Large Language Models (LLMs) and prompt-based learning to semantically correct and enrich user queries with relevant contextual information. These enriched queries are then processed through parallel retrieval, followed by an evaluation step to select the most relevant results based on their alignment with the original query. Through extensive experiments on our custom-developed dataset, we demonstrate that RAPID significantly outperforms traditional retrieval methods, particularly for contextually incomplete queries. Our system was validated for both speed and accuracy through participation in the Ho Chi Minh City AI Challenge 2024, where it successfully retrieved events from over 300 hours of video. Further evaluation comparing RAPID with the baseline proposed by the competition organizers demonstrated its superior effectiveness, highlighting the strength and robustness of our approach. 

**Abstract (ZH)**: 基于文本查询从视频中检索事件变得越来越具有挑战性，这主要是由于多媒体内容的迅速增长。现有的基于文本的视频事件检索方法往往侧重于对象级别的描述，而忽视了上下文信息的重要作用。特别是在查询缺乏足够的上下文信息时，这一局限尤为明显，例如缺失地理位置详情或背景元素模糊不清。为了解决这些挑战，我们提出了一种名为RAPID（检索增强并行推理起草）的新系统，该系统利用了大型语言模型（LLMs）和提示式学习技术，对用户查询进行语义纠正和丰富，添加相关的上下文信息。这些经过丰富化的查询随后通过并行检索处理，在评估步骤中根据与原始查询的一致性选择最相关的结果。通过在我们自定义开发的数据集上进行大量实验，我们证明了RAPID显著优于传统的检索方法，特别是在上下文不完整查询的情况下。我们的系统通过参加2024胡志明市人工智能挑战赛进行了速度和准确性的验证，在此次挑战赛中成功检索了超过300小时的视频事件。进一步的评估表明，RAPID在与竞赛组织者提出的基线方法进行比较时表现更为优越，突显了我们方法的强度和鲁棒性。 

---
# Matryoshka Re-Ranker: A Flexible Re-Ranking Architecture With Configurable Depth and Width 

**Title (ZH)**: 逐层套娃重新排序架构：一种可配置深度和宽度的灵活重新排序架构 

**Authors**: Zheng Liu, Chaofan Li, Shitao Xiao, Chaozhuo Li, Defu Lian, Yingxia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2501.16302)  

**Abstract**: Large language models (LLMs) provide powerful foundations to perform fine-grained text re-ranking. However, they are often prohibitive in reality due to constraints on computation bandwidth. In this work, we propose a \textbf{flexible} architecture called \textbf{Matroyshka Re-Ranker}, which is designed to facilitate \textbf{runtime customization} of model layers and sequence lengths at each layer based on users' configurations. Consequently, the LLM-based re-rankers can be made applicable across various real-world situations. The increased flexibility may come at the cost of precision loss. To address this problem, we introduce a suite of techniques to optimize the performance. First, we propose \textbf{cascaded self-distillation}, where each sub-architecture learns to preserve a precise re-ranking performance from its super components, whose predictions can be exploited as smooth and informative teacher signals. Second, we design a \textbf{factorized compensation mechanism}, where two collaborative Low-Rank Adaptation modules, vertical and horizontal, are jointly employed to compensate for the precision loss resulted from arbitrary combinations of layer and sequence compression. We perform comprehensive experiments based on the passage and document retrieval datasets from MSMARCO, along with all public datasets from BEIR benchmark. In our experiments, Matryoshka Re-Ranker substantially outperforms the existing methods, while effectively preserving its superior performance across various forms of compression and different application scenarios. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为执行细粒度文本重排序提供了强大的基础。然而，由于计算带宽的限制，它们在实际应用中往往难以实现。在此项工作中，我们提出了一种灵活的架构，即**玛特罗什卡重排序器**（Matroyshka Re-Ranker），该架构旨在根据用户配置，在运行时对每层的模型层和序列长度进行自定义。因此，基于LLM的重排序器可以在各种实际应用场景中得到应用。虽然这种灵活性可能带来精度损失，但我们引入了一套技术来优化性能。首先，我们提出了一种**级联自我精炼**（Cascaded Self-Distillation）方法，其中每个子架构从其超架构中学习保留精确的重排序性能，使其预测能够作为平滑且信息丰富的教师信号。其次，我们设计了一种**因子补偿机制**，其中垂直和水平低秩适应模块协同工作，以补偿由于任意组合的层和序列压缩而导致的精度损失。我们在基于MSMARCO的片段检索和文档检索数据集，以及BEIR基准中的所有公开数据集上进行了全面的实验。在实验中，玛特罗什卡重排序器显著优于现有方法，同时在其各种压缩形式和不同应用场景中有效保持了其优越的性能。 

---
# URAG: Implementing a Unified Hybrid RAG for Precise Answers in University Admission Chatbots -- A Case Study at HCMUT 

**Title (ZH)**: URAG: 实现统一混合RAG以在大学录取聊天机器人中提供精确答案——河内科技大学案例研究 

**Authors**: Long Nguyen, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16276)  

**Abstract**: With the rapid advancement of Artificial Intelligence, particularly in Natural Language Processing, Large Language Models (LLMs) have become pivotal in educational question-answering systems, especially university admission chatbots. Concepts such as Retrieval-Augmented Generation (RAG) and other advanced techniques have been developed to enhance these systems by integrating specific university data, enabling LLMs to provide informed responses on admissions and academic counseling. However, these enhanced RAG techniques often involve high operational costs and require the training of complex, specialized modules, which poses challenges for practical deployment. Additionally, in the educational context, it is crucial to provide accurate answers to prevent misinformation, a task that LLM-based systems find challenging without appropriate strategies and methods. In this paper, we introduce the Unified RAG (URAG) Framework, a hybrid approach that significantly improves the accuracy of responses, particularly for critical queries. Experimental results demonstrate that URAG enhances our in-house, lightweight model to perform comparably to state-of-the-art commercial models. Moreover, to validate its practical applicability, we conducted a case study at our educational institution, which received positive feedback and acclaim. This study not only proves the effectiveness of URAG but also highlights its feasibility for real-world implementation in educational settings. 

**Abstract (ZH)**: 随着人工智能的迅速发展，特别是在自然语言处理领域的进步，大型语言模型（LLMs）已成为教育问答系统中的重要组成部分，尤其是在大学招生聊天机器人的应用中。检索增强生成（RAG）等先进技术的概念和其他高级技术已经发展起来，通过整合特定的大学数据来增强这些系统，使LLMs能够提供涉及入学和学术指导的知情回答。然而，这些增强的RAG技术通常涉及较高的运营成本，并需要训练复杂的专门模块，这对实际部署构成了挑战。此外，在教育背景下，提供准确的答案以防止 misinformation至关重要，这是一项LLM基础系统在缺乏适当策略和方法的情况下难以完成的任务。在本文中，我们介绍了统一RAG（URAG）框架，这是一种混合方法，显著提高了对关键查询的响应准确性。实验结果显示，URAG能够使我们内部的轻量级模型表现得与最先进的商业模型相当。此外，为了验证其实用性，我们在教育机构进行了一项案例研究，收到了积极的反馈和赞誉。这项研究不仅证明了URAG的有效性，还突显了它在教育环境中实际应用的可行性。 

---
# Return of the Encoder: Maximizing Parameter Efficiency for SLMs 

**Title (ZH)**: 编码器的回归：最大化SLMs的参数效率 

**Authors**: Mohamed Elfeki, Rui Liu, Chad Voegele  

**Link**: [PDF](https://arxiv.org/pdf/2501.16273)  

**Abstract**: The dominance of large decoder-only language models has overshadowed encoder-decoder architectures, despite their fundamental efficiency advantages in sequence processing. For small language models (SLMs) - those with 1 billion parameters or fewer - our systematic analysis across GPU, CPU, and NPU platforms reveals that encoder-decoder architectures achieve 47% lower first-token latency and 4.7x higher throughput compared to decoder-only models on edge devices. These gains may be attributed to encoder-decoder's one-time input processing and efficient separation of understanding and generation phases.
We introduce a novel knowledge distillation framework that enables encoder-decoder models to leverage capabilities from large scalable decoder-only teachers while preserving their architectural advantages, achieving up to 6 average performance points improvement across diverse tasks, with significant gains in asymmetric sequence tasks where input and output distributions can benefit from different processing approaches.
When combined with modern advances like Rotary Positional Embeddings (RoPE) and Vision encoders, our systematic investigation demonstrates that encoder-decoder architectures provide a more practical path toward deploying capable language models in resource-constrained environments. Our findings challenge the prevailing trend toward decoder-only scaling, showing that architectural choices become increasingly crucial as parameter budgets decrease, particularly for on-device and edge deployments where computational efficiency is paramount. 

**Abstract (ZH)**: 大型解码器语言模型的主导地位虽然在很大程度上掩盖了编码器-解码器架构的优势，但这些架构在序列处理方面具有基本的效率优势。对于小型语言模型（SLMs，参数数量在10亿以下的模型），我们在GPU、CPU和NPU平台上的系统分析表明，编码器-解码器架构在边缘设备上的首次标记延迟比解码器模型低47%，吞吐量高4.7倍。这些收益可能归因于编码器-解码器架构的一次性输入处理和理解与生成阶段的有效分离。

我们提出了一个新颖的知识蒸馏框架，使编码器-解码器模型能够利用大型可扩展解码器教师的能力，同时保留其架构优势，在各种任务上实现了最高达6个平均性能点的改进，特别是在输入和输出分布可以从不同的处理方法中获益的不平衡序列任务中表现尤为显著。

结合现代技术进步，如旋转位置嵌入（RoPE）和视觉编码器，我们的系统研究展示了编码器-解码器架构在资源受限环境部署高能力语言模型中的实际路径。我们的发现挑战了以解码器为主的扩展趋势，指出随着参数预算的减少，架构选择变得越来越关键，特别是在边缘设备和设备上，计算效率尤为重要。 

---
# A foundation model for human-AI collaboration in medical literature mining 

**Title (ZH)**: 医学文献挖掘中的人工智能协作基础模型 

**Authors**: Zifeng Wang, Lang Cao, Qiao Jin, Joey Chan, Nicholas Wan, Behdad Afzali, Hyun-Jin Cho, Chang-In Choi, Mehdi Emamverdi, Manjot K. Gill, Sun-Hyung Kim, Yijia Li, Yi Liu, Hanley Ong, Justin Rousseau, Irfan Sheikh, Jenny J. Wei, Ziyang Xu, Christopher M. Zallek, Kyungsang Kim, Yifan Peng, Zhiyong Lu, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.16255)  

**Abstract**: Systematic literature review is essential for evidence-based medicine, requiring comprehensive analysis of clinical trial publications. However, the application of artificial intelligence (AI) models for medical literature mining has been limited by insufficient training and evaluation across broad therapeutic areas and diverse tasks. Here, we present LEADS, an AI foundation model for study search, screening, and data extraction from medical literature. The model is trained on 633,759 instruction data points in LEADSInstruct, curated from 21,335 systematic reviews, 453,625 clinical trial publications, and 27,015 clinical trial registries. We showed that LEADS demonstrates consistent improvements over four cutting-edge generic large language models (LLMs) on six tasks. Furthermore, LEADS enhances expert workflows by providing supportive references following expert requests, streamlining processes while maintaining high-quality results. A study with 16 clinicians and medical researchers from 14 different institutions revealed that experts collaborating with LEADS achieved a recall of 0.81 compared to 0.77 experts working alone in study selection, with a time savings of 22.6%. In data extraction tasks, experts using LEADS achieved an accuracy of 0.85 versus 0.80 without using LEADS, alongside a 26.9% time savings. These findings highlight the potential of specialized medical literature foundation models to outperform generic models, delivering significant quality and efficiency benefits when integrated into expert workflows for medical literature mining. 

**Abstract (ZH)**: 系统性文献综述是循证医学的重要组成部分，需要对临床试验文献进行全面分析。然而，由于在广泛的治疗领域和多样的任务中存在训练和评估不足的问题，人工智能（AI）模型在医学文献挖掘的应用受到限制。在此，我们介绍了LEADS，一种用于研究搜寻、筛查和数据提取的AI基础模型。LEADS在名为LEADSInstruct的数据集上进行训练，该数据集包含来自21,335篇系统评价、453,625篇临床试验文献和27,015篇临床试验注册记录的633,759条指令数据点。

我们展示了LEADS在六个任务上优于四款最新通用大型语言模型（LLMs）的一致优势。此外，LEADS通过在专家请求后提供支持性参考，在保持高质量结果的同时简化流程，从而增强了专家工作流程。一项来自14家不同机构的16名临床医生和医学研究人员的研究表明，与独自工作相比，专家使用LEADS进行研究选择时的召回率从0.77提高到0.81，节省了22.6%的时间。在数据提取任务中，使用LEADS的专家达到的准确率从0.80提高到0.85，同时节省了26.9%的时间。这些发现突显了专门化的医学文献基础模型相对于通用模型的潜在优势，在医疗文献挖掘的专家工作流程中整合时能够显著提高质量和效率。 

---
# Echoes of Discord: Forecasting Hater Reactions to Counterspeech 

**Title (ZH)**: 《纷争的回响：针对反言的仇恨反应预测》 

**Authors**: Xiaoying Song, Sharon Lisseth Perez, Xinchen Yu, Eduardo Blanco, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2501.16235)  

**Abstract**: Hate speech (HS) erodes the inclusiveness of online users and propagates negativity and division. Counterspeech has been recognized as a way to mitigate the harmful consequences. While some research has investigated the impact of user-generated counterspeech on social media platforms, few have examined and modeled haters' reactions toward counterspeech, despite the immediate alteration of haters' attitudes being an important aspect of counterspeech. This study fills the gap by analyzing the impact of counterspeech from the hater's perspective, focusing on whether the counterspeech leads the hater to reenter the conversation and if the reentry is hateful. We compile the Reddit Echoes of Hate dataset (ReEco), which consists of triple-turn conversations featuring haters' reactions, to assess the impact of counterspeech. The linguistic analysis sheds insights on the language of counterspeech to hate eliciting different haters' reactions. Experimental results demonstrate that the 3-way classification model outperforms the two-stage reaction predictor, which first predicts reentry and then determines the reentry type. We conclude the study with an assessment showing the most common errors identified by the best-performing model. 

**Abstract (ZH)**: 仇恨言论（HS）削弱了在线用户的包容性，并传播消极和分裂情绪。对比言论（counterspeech）已被认定为减轻其负面影响的一种方式。尽管已有部分研究探讨了用户生成的对比言论在社交媒体平台上的影响，但鲜有研究从仇恨言论发布者的角度出发，分析和建模他们对对比言论的反应，尤其是对比言论能否促使仇恨言论发布者重新加入对话并继续发表仇恨言论这一即时态度改变过程的重要性。本研究通过从发布者的视角分析对比言论的影响，填补了这一空白，重点关注对比言论是否能让仇恨言论发布者重新参与对话，以及重新参与对话时内容是否仍然仇恨。我们构建了一个包含仇恨言论发布者反应的三回合对话数据集（ReEco），以评估对比言论的影响。语言分析揭示了不同仇恨言论发布者对对比言论的不同反应所使用的语言特征。实验结果表明，三分类模型的性能优于两阶段反应预测器，该预测器首先预测重新参与对话的可能性，然后确定重新参与的具体类型。研究结论部分还包括对最佳模型识别出的最常见错误进行了评估。 

---
# DBRouting: Routing End User Queries to Databases for Answerability 

**Title (ZH)**: DBRouting: 将用户查询导向数据库以确保查询回答性 

**Authors**: Priyangshu Mandal, Manasi Patwardhan, Mayur Patidar, Lovekesh Vig  

**Link**: [PDF](https://arxiv.org/pdf/2501.16220)  

**Abstract**: Enterprise level data is often distributed across multiple sources and identifying the correct set-of data-sources with relevant information for a knowledge request is a fundamental challenge. In this work, we define the novel task of routing an end-user query to the appropriate data-source, where the data-sources are databases. We synthesize datasets by extending existing datasets designed for NL-to-SQL semantic parsing. We create baselines on these datasets by using open-source LLMs, using both pre-trained and task specific embeddings fine-tuned using the training data. With these baselines we demonstrate that open-source LLMs perform better than embedding based approach, but suffer from token length limitations. Embedding based approaches benefit from task specific fine-tuning, more so when there is availability of data in terms of database specific questions for training. We further find that the task becomes more difficult (i) with an increase in the number of data-sources, (ii) having data-sources closer in terms of their domains,(iii) having databases without external domain knowledge required to interpret its entities and (iv) with ambiguous and complex queries requiring more fine-grained understanding of the data-sources or logical reasoning for routing to an appropriate source. This calls for the need for developing more sophisticated solutions to better address the task. 

**Abstract (ZH)**: 企业级别的数据往往分布在多个来源，识别出与知识请求相关的正确数据源是一个基本挑战。在本项工作中，我们定义了将终端用户查询导向适当数据源的新任务，这里的数据源是数据库。我们通过扩展现有的针对自然语言到SQL语义解析设计的数据集来合成数据集。在这些数据集上，我们使用开源的大规模语言模型（LLM）创建了基线，这些模型使用预训练模型和针对特定任务微调的嵌入。通过这些基线表明，开源LLM在性能上优于基于嵌入的方法，但受限于标记长度的限制。基于嵌入的方法可以从特定任务的微调中受益，尤其是有针对数据库具体问题的训练数据可用时。我们进一步发现，任务的难度随（i）数据源数量的增加而增加，（ii）数据源在领域上的接近程度而增加，（iii）缺乏外部领域知识来解释其实体的数据库，以及（iv）含糊且复杂的查询所需的数据源的更精细理解或逻辑推理以导向合适的数据源而增加。这要求开发更复杂的方法来更好地解决这项任务。 

---
# Provence: efficient and robust context pruning for retrieval-augmented generation 

**Title (ZH)**: Provence：高效的鲁棒上下文裁剪方法以提升检索增强生成模型的性能 

**Authors**: Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2501.16214)  

**Abstract**: Retrieval-augmented generation improves various aspects of large language models (LLMs) generation, but suffers from computational overhead caused by long contexts as well as the propagation of irrelevant retrieved information into generated responses. Context pruning deals with both aspects, by removing irrelevant parts of retrieved contexts before LLM generation. Existing context pruning approaches are however limited, and do not provide a universal model that would be both efficient and robust in a wide range of scenarios, e.g., when contexts contain a variable amount of relevant information or vary in length, or when evaluated on various domains. In this work, we close this gap and introduce Provence (Pruning and Reranking Of retrieVEd relevaNt ContExts), an efficient and robust context pruner for Question Answering, which dynamically detects the needed amount of pruning for a given context and can be used out-of-the-box for various domains. The three key ingredients of Provence are formulating the context pruning task as sequence labeling, unifying context pruning capabilities with context reranking, and training on diverse data. Our experimental results show that Provence enables context pruning with negligible to no drop in performance, in various domains and settings, at almost no cost in a standard RAG pipeline. We also conduct a deeper analysis alongside various ablations to provide insights into training context pruners for future work. 

**Abstract (ZH)**: 检索增强生成可以提高大型语言模型（LLMs）生成的各个方面，但会长语境和检索信息的传播导致的相关性和无关联信息的传播带来计算开销。上下文剪枝通过在LLM生成前去除检索上下文中的无关部分，来处理这两个方面的问题。然而，现有的上下文剪枝方法存在一定局限性，无法在多种场景下提供高效且稳定的模型，例如，当上下文包含不同量的相关信息或长度不同，或在不同领域接受评估时。本研究填补了这一空白，推出了“Provence（剪枝和排序检索相关上下文）”，这是一种高效且稳健的问题解答上下文剪枝工具，能够根据给定的上下文动态检测所需的剪枝量，并且可以通用应用于多种领域。Provence的三大关键成分为：将上下文剪枝任务表述为序列标注、将上下文剪枝能力与上下文排序统一，以及在多样数据上进行训练。实验结果显示，Provence能够在多种领域和设置下实现上下文剪枝，几乎没有性能下降，并且在标准检索增强生成（RAG）管道中的成本几乎可以忽略不计。我们还进行了一系列的深入分析和多方面消融研究，以提供对未来培训上下文剪枝器的见解。 

---
# Can summarization approximate simplification? A gold standard comparison 

**Title (ZH)**: 概括能够逼近简化吗？一个金标准对比 

**Authors**: Giacomo Magnifico, Eduard Barbu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16181)  

**Abstract**: This study explores the overlap between text summarization and simplification outputs. While summarization evaluation methods are streamlined, simplification lacks cohesion, prompting the question: how closely can abstractive summarization resemble gold-standard simplification? We address this by applying two BART-based BRIO summarization methods to the Newsela corpus, comparing outputs with manually annotated simplifications and achieving a top ROUGE-L score of 0.654. This provides insight into where summarization and simplification outputs converge and differ. 

**Abstract (ZH)**: 本研究探讨了文本摘要和简化输出之间的重叠。虽然摘要评估方法已经简化，但简化缺乏连贯性，因此提出了一个问题：抽象总结能否与黄金标准简化高度相似？我们通过将两种基于BART的BRIO摘要方法应用于Newsela语料库，并将产出与人工标注的简化版本进行比较，取得了ROUGE-L得分为0.654的最佳成绩。这为理解摘要和简化输出的交集和区别提供了见解。 

---
# AdaCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Chain-of-Thought 

**Title (ZH)**: AdaCoT：重新思考适应性思维链在跨语言事实推理中的作用 

**Authors**: Xin Huang, Tarun Kumar Vangani, Zhengyuan Liu, Bowei Zou, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2501.16154)  

**Abstract**: Large language models (LLMs) have shown impressive multilingual capabilities through pretraining on diverse corpora. While these models show strong reasoning abilities, their performance varies significantly across languages due to uneven training data distribution. Existing approaches using machine translation, and extensive multilingual pretraining and cross-lingual tuning face scalability challenges and often fail to capture nuanced reasoning processes across languages. In this paper, we introduce AdaCoT (Adaptive Chain-of-Thought), a framework that enhances multilingual reasoning by dynamically routing thought processes through intermediary "thinking languages" before generating target-language responses. AdaCoT leverages a language-agnostic core and incorporates an adaptive, reward-based mechanism for selecting optimal reasoning pathways without requiring additional pretraining. Our comprehensive evaluation across multiple benchmarks demonstrates substantial improvements in both factual reasoning quality and cross-lingual consistency, with particularly strong performance gains in low-resource language settings. The results suggest that adaptive reasoning paths can effectively bridge the performance gap between high and low-resource languages while maintaining cultural and linguistic nuances. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过在多种语料上进行预训练展示了令人印象深刻的多语言能力。虽然这些模型在推理能力方面表现出色，但由于训练数据分布不均，它们在不同语言上的表现差异显著。现有的使用机器翻译、广泛进行多语言预训练和跨语言微调的方法面临着可扩展性的挑战，并且通常不能捕捉到跨语言的细微推理过程。在本文中，我们提出了AdaCoT（自适应推理链），这是一种通过动态路由思考过程通过中间的“思考语言”来增强多语言推理的框架，从而在生成目标语言回应之前进行推理。AdaCoT 利用了一个语言无关的核心，并引入了一种适应性的、基于奖励的机制，用于选择最优的推理路径，而无需额外的预训练。我们在多个基准测试中的全面评估显示，在事实推理质量和跨语言一致性方面取得了显著的改进，特别是在资源匮乏的语言环境中表现尤为突出。结果表明，适应性的推理路径可以有效地缩小高资源和低资源语言之间的性能差距，同时保留文化与语言的细微差异。 

---
# Evaluation of NMT-Assisted Grammar Transfer for a Multi-Language Configurable Data-to-Text System 

**Title (ZH)**: 基于神经机器翻译辅助语法转移的多语言配置数据到文本系统的评估 

**Authors**: Andreas Madsack, Johanna Heininger, Adela Schneider, Ching-Yi Chen, Christian Eckard, Robert Weißgraeber  

**Link**: [PDF](https://arxiv.org/pdf/2501.16135)  

**Abstract**: One approach for multilingual data-to-text generation is to translate grammatical configurations upfront from the source language into each target language. These configurations are then used by a surface realizer and in document planning stages to generate output. In this paper, we describe a rule-based NLG implementation of this approach where the configuration is translated by Neural Machine Translation (NMT) combined with a one-time human review, and introduce a cross-language grammar dependency model to create a multilingual NLG system that generates text from the source data, scaling the generation phase without a human in the loop. Additionally, we introduce a method for human post-editing evaluation on the automatically translated text. Our evaluation on the SportSett:Basketball dataset shows that our NLG system performs well, underlining its grammatical correctness in translation tasks. 

**Abstract (ZH)**: 一种针对多语言数据到文本生成的方法是，首先将源语言的语法配置翻译成每种目标语言。这些配置随后被表面实现器和文档规划阶段所使用，以生成输出。在本文中，我们描述了一种基于规则的自然语言生成（NLG）实现方法，其中配置通过神经机器翻译（NMT）结合一次人工审核进行翻译，并引入一种跨语言语法依赖模型，以创建一个可以从源数据生成文本的多语言NLG系统，同时无需人工介入即可扩展生成阶段。此外，我们还介绍了一种自动翻译文本的人工后编辑评估方法。我们在SportSett:Basketball数据集上的评估表明，我们的NLG系统表现良好，证明了其在翻译任务中的语法正确性。 

---
# From #Dr00gtiktok to #harmreduction: Exploring Substance Use Hashtags on TikTok 

**Title (ZH)**: 从#Dr00gtiktok到#harmreduction：探究TikTok上的毒品使用相关标签 

**Authors**: Layla Bouzoubaa, Muqi Guo, Joseph Trybala, Afsaneh Razi, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2501.16123)  

**Abstract**: The rise of TikTok as a primary source of information for youth, combined with its unique short-form video format, creates urgent questions about how substance use content manifests and spreads on the platform. This paper provides the first in-depth exploration of substance use-related content on TikTok, covering all major substance categories as classified by the Drug Enforcement Agency. Through social network analysis and qualitative coding, we examined more than 2,333 hashtags across 39,509 videos, identified 16 distinct hashtag communities and analyzed their interconnections and thematic content. Our analysis revealed a highly interconnected small-world network where recovery-focused hashtags like #addiction, #recovery, and #sober serve as central bridges between communities. Through manual coding of 351 representative videos, we found that Recovery Advocacy content (33.9%) and Satirical content (28.2%) dominate, while direct substance depiction appears in only 26% of videos, with active use shown in just 6.5% of them. This suggests TikTok functions primarily as a recovery support platform rather than a space promoting substance use. We found strong alignment between hashtag communities and video content, indicating organic community formation rather than attempts to evade content moderation. Our findings inform how platforms can balance content moderation with preserving valuable recovery support communities, while also providing insights for the design of social media-based recovery interventions. 

**Abstract (ZH)**: 随着TikTok成为青年获取信息的主要来源，其独特的短格式视频格式使得关于使用物质内容的呈现和传播问题变得尤为迫切。本论文提供了对TikTok上与使用物质相关的内容的首次深入探索，涵盖了美国毒品执法署分类的所有主要物质类别。通过社会网络分析和定性编码，我们分析了超过2,333个标签下的近39,509个视频，识别出16个独特的标签社区，并分析了它们的相互联系和主题内容。分析结果显示，形成一个紧密相连的小世界网络，在其中，以戒毒为主题的标签号如#addiction、#recovery和#sober起到了社区间的关键桥梁作用。通过对351个代表性视频进行手工编码，我们发现康复倡导内容（33.9%）和讽刺内容（28.2%）占据主导地位，而直接展示使用物质的视频仅占35%，其中积极使用物质的视频更是仅占6.5%。这表明TikTok主要作为一个康复支持平台，而不是一个推广物质使用的空间。研究发现，标签社区与视频内容的高度契合表明这些社区是在自然形成，而非试图逃避内容管理。我们的研究结果不仅为平台如何平衡内容管理与保持有价值的康复支持社区提供了指导，还为基于社交媒体的康复干预设计提供了有益的见解。 

---
# Towards Explainable Multimodal Depression Recognition for Clinical Interviews 

**Title (ZH)**: 面向可解释的多模态抑郁识别在临床访谈中的应用 

**Authors**: Wenjie Zheng, Qiming Xie, Zengzhi Wang, Jianfei Yu, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2501.16106)  

**Abstract**: Recently, multimodal depression recognition for clinical interviews (MDRC) has recently attracted considerable attention. Existing MDRC studies mainly focus on improving task performance and have achieved significant development. However, for clinical applications, model transparency is critical, and previous works ignore the interpretability of decision-making processes. To address this issue, we propose an Explainable Multimodal Depression Recognition for Clinical Interviews (EMDRC) task, which aims to provide evidence for depression recognition by summarizing symptoms and uncovering underlying causes. Given an interviewer-participant interaction scenario, the goal of EMDRC is to structured summarize participant's symptoms based on the eight-item Patient Health Questionnaire depression scale (PHQ-8), and predict their depression severity. To tackle the EMDRC task, we construct a new dataset based on an existing MDRC dataset. Moreover, we utilize the PHQ-8 and propose a PHQ-aware multimodal multi-task learning framework, which captures the utterance-level symptom-related semantic information to help generate dialogue-level summary. Experiment results on our annotated dataset demonstrate the superiority of our proposed methods over baseline systems on the EMDRC task. 

**Abstract (ZH)**: 近年来，临床访谈中的多模态抑郁识别（MDRC）受到了广泛关注。现有MDRC研究主要侧重于提高任务性能，并已取得显著进展。然而，在临床应用中，模型的透明度至关重要，而以往的研究忽略了决策过程的可解释性。为解决这一问题，我们提出了一项名为可解释的多模态抑郁识别临床访谈（EMDRC）任务，该任务旨在通过总结症状和揭示潜在原因来为抑郁识别提供依据。给定访谈者-参与者互动场景，EMDRC的目标是基于患者健康问卷抑郁量表（PHQ-8）的八项指标，对参与者的症状进行结构化总结，并预测其抑郁严重程度。为应对此任务，我们基于现有的MDRC数据集构建了一个新的数据集，并利用PHQ-8提出了一个PHQ意识多模态多任务学习框架，该框架捕捉单个语句层面的症状相关语义信息，以帮助生成对话级摘要。我们在标记数据集上的实验结果表明，所提出的方法在EMDRC任务上的性能优于基准系统。 

---
# STAR: Stepwise Task Augmentation and Relation Learning for Aspect Sentiment Quad Prediction 

**Title (ZH)**: STAR：逐步任务增强与关系学习在aspect-sentiment四元组预测中的应用 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16093)  

**Abstract**: Aspect-based sentiment analysis (ABSA) aims to identify four sentiment elements, including aspect term, aspect category, opinion term, and sentiment polarity. These elements construct the complete picture of sentiments. The most challenging task, aspect sentiment quad prediction (ASQP), predicts these elements simultaneously, hindered by difficulties in accurately coupling different sentiment elements. A key challenge is insufficient annotated data that limits the capability of models in semantic understanding and reasoning about quad prediction. To address this, we propose stepwise task augmentation and relation learning (STAR), a strategy inspired by human reasoning. STAR constructs auxiliary data to learn quadruple relationships incrementally by augmenting with pairwise and overall relation tasks derived from training data. By encouraging the model to infer causal relationships among sentiment elements without requiring additional annotations, STAR effectively enhances quad prediction. Extensive experiments demonstrate the proposed STAR exhibits superior performance on four benchmark datasets. 

**Abstract (ZH)**: 基于方面的情感分析（Aspect-based Sentiment Analysis, ABSA）旨在识别四个情感元素，包括方面术语、方面类别、意见术语和情感极性。这些元素构建了情感的完整图景。最具有挑战性的任务是方面情感四元素预测（Aspect Sentiment Quad Prediction, ASQP），该任务涉及同时预测这些元素，其难点在于不同情感元素之间准确耦合的困难。一个关键挑战是标注数据的不足，这限制了模型在语义理解和推理方面的能力。为了应对这一问题，我们提出了一种逐步任务扩增和关系学习（Stepwise Task Augmentation and Relation Learning, STAR）策略，该策略借鉴了人类推理的方法。STAR 通过逐步添加从训练数据推导出的成对关系任务和整体关系任务来构建辅助数据，以增量学习四元关系。通过鼓励模型推断情感元素之间的因果关系，而无需额外的标注，STAR 有效提升了四元素预测能力。 extensive 实验表明，提出的 STAR 在四个基准数据集上表现出优越的性能。 

---
# Integration of LLM Quality Assurance into an NLG System 

**Title (ZH)**: 将LLM质量保障集成到自然语言生成系统中 

**Authors**: Ching-Yi Chen, Johanna Heininger, Adela Schneider, Christian Eckard, Andreas Madsack, Robert Weißgraeber  

**Link**: [PDF](https://arxiv.org/pdf/2501.16078)  

**Abstract**: In this paper, we present a system that uses a Large Language Model (LLM) to perform grammar and spelling correction as a component of Quality Assurance (QA) for texts generated by NLG systems, which is important for text production in real-world scenarios. Evaluating the results of the system on work-in-progress sports news texts in three languages, we show that it is able to deliver acceptable corrections. 

**Abstract (ZH)**: 在本文中，我们介绍了一个系统，该系统利用大型语言模型（LLM）作为自然语言生成（NLG）系统生成文本的质量保证（QA）组成部分，以执行语法和拼写纠错。这对于实际场景中的文本生成非常重要。我们通过对三种语言的工作中进行中的体育新闻文本进行评估，结果显示该系统能够提供可接受的纠错效果。 

---
# RelCAT: Advancing Extraction of Clinical Inter-Entity Relationships from Unstructured Electronic Health Records 

**Title (ZH)**: RelCAT: 从结构化电子健康记录中提取临床实体关系的高级方法 

**Authors**: Shubham Agarwal, Vlad Dinu, Thomas Searle, Mart Ratas, Anthony Shek, Dan F. Stein, James Teo, Richard Dobson  

**Link**: [PDF](https://arxiv.org/pdf/2501.16077)  

**Abstract**: This study introduces RelCAT (Relation Concept Annotation Toolkit), an interactive tool, library, and workflow designed to classify relations between entities extracted from clinical narratives. Building upon the CogStack MedCAT framework, RelCAT addresses the challenge of capturing complete clinical relations dispersed within text. The toolkit implements state-of-the-art machine learning models such as BERT and Llama along with proven evaluation and training methods. We demonstrate a dataset annotation tool (built within MedCATTrainer), model training, and evaluate our methodology on both openly available gold-standard and real-world UK National Health Service (NHS) hospital clinical datasets. We perform extensive experimentation and a comparative analysis of the various publicly available models with varied approaches selected for model fine-tuning. Finally, we achieve macro F1-scores of 0.977 on the gold-standard n2c2, surpassing the previous state-of-the-art performance, and achieve performance of >=0.93 F1 on our NHS gathered datasets. 

**Abstract (ZH)**: 本研究介绍了RelCAT（关系概念标注工具包），这是一种交互式工具、库和工作流程，旨在对从临床叙事中提取的实体之间的关系进行分类。RelCAT 建立在CogStack MedCAT框架之上，旨在解决捕捉分散在文本中的完整临床关系的挑战。该工具包实现了包括BERT和Llama在内的先进机器学习模型，并采用了经过验证的评估和训练方法。我们展示了MedCATTrainer内部构建的数据集标注工具、模型训练方法，并在开放获取的黄金标准和真实的英国国家卫生服务（NHS）医院临床数据集上评估了我们的方法。我们进行了广泛的实验，并对各种公开可用的模型进行了比较分析，选用了不同的模型微调方法。最终，我们在黄金标准n2c2数据集上实现了宏F1分数为0.977，超过了之前的先进性能；在我们收集的NHS数据集上实现了F1分数>=0.93的性能。 

---
# PISCO: Pretty Simple Compression for Retrieval-Augmented Generation 

**Title (ZH)**: PISCO：简单的检索增强生成压缩方法 

**Authors**: Maxime Louis, Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2501.16075)  

**Abstract**: Retrieval-Augmented Generation (RAG) pipelines enhance Large Language Models (LLMs) by retrieving relevant documents, but they face scalability issues due to high inference costs and limited context size. Document compression is a practical solution, but current soft compression methods suffer from accuracy losses and require extensive pretraining. In this paper, we introduce PISCO, a novel method that achieves a 16x compression rate with minimal accuracy loss (0-3%) across diverse RAG-based question-answering (QA) tasks. Unlike existing approaches, PISCO requires no pretraining or annotated data, relying solely on sequence-level knowledge distillation from document-based questions. With the ability to fine-tune a 7-10B LLM in 48 hours on a single A100 GPU, PISCO offers a highly efficient and scalable solution. We present comprehensive experiments showing that PISCO outperforms existing compression models by 8% in accuracy. 

**Abstract (ZH)**: 检索增强生成（RAG）管道通过检索相关文档来增强大型语言模型（LLMs），但在处理高推理成本和有限上下文大小的问题上面临可扩展性挑战。文档压缩是一种实际的解决方案，但当前的软压缩方法会带来准确度损失，并需要大量的预训练。本文介绍了一种名为PISCO的新方法，在各种基于RAG的问答（QA）任务中实现了16倍的压缩率，同时保持极小的准确度损失（0-3%）。与现有方法不同，PISCO无需进行预训练或标注数据，仅依赖于基于文档的问题的序列级知识蒸馏。通过在单块A100 GPU上48小时内微调一个7-10B参数的LLM，PISCO提供了一种高效且可扩展的解决方案。我们进行了全面的实验，结果显示，PISCO的准确度比现有压缩模型高出8%。 

---
# MEL: Legal Spanish Language Model 

**Title (ZH)**: MEL：法律西班牙语语言模型 

**Authors**: David Betancur Sánchez, Nuria Aldama García, Álvaro Barbero Jiménez, Marta Guerrero Nieto, Patricia Marsà Morales, Nicolás Serrano Salas, Carlos García Hernán, Pablo Haya Coll, Elena Montiel Ponsoda, Pablo Calleja Ibáñez  

**Link**: [PDF](https://arxiv.org/pdf/2501.16011)  

**Abstract**: Legal texts, characterized by complex and specialized terminology, present a significant challenge for Language Models. Adding an underrepresented language, such as Spanish, to the mix makes it even more challenging. While pre-trained models like XLM-RoBERTa have shown capabilities in handling multilingual corpora, their performance on domain specific documents remains underexplored. This paper presents the development and evaluation of MEL, a legal language model based on XLM-RoBERTa-large, fine-tuned on legal documents such as BOE (Boletín Oficial del Estado, the Spanish oficial report of laws) and congress texts. We detail the data collection, processing, training, and evaluation processes. Evaluation benchmarks show a significant improvement over baseline models in understanding the legal Spanish language. We also present case studies demonstrating the model's application to new legal texts, highlighting its potential to perform top results over different NLP tasks. 

**Abstract (ZH)**: 法律文本以其复杂而专门的术语而著称，这为语言模型带来了显著挑战。增加一种代表性不足的语言，如西班牙语，使这一挑战更加严峻。虽然预训练模型如XLM-RoBERTa在处理多语言语料库方面显示出能力，但其在特定领域文档中的性能仍然没有得到充分探索。本文介绍了基于XLM-RoBERTa-large并通过西班牙官方法律文件（如《国家官方公报》BOE及其议会文本）进行微调的法律语言模型MEL的发展与评估。我们详述了数据收集、处理、训练和评估过程。评估基准显示，该模型在理解法律西班牙语方面的表现显著优于基线模型。我们还通过案例研究展示了该模型在处理新法律文本中的应用，突显了其在不同NLP任务中实现顶级性能的潜力。 

---
# 3CEL: A corpus of legal Spanish contract clauses 

**Title (ZH)**: 3CEL：一份合同条款语料库（法律西班牙语） 

**Authors**: Nuria Aldama García, Patricia Marsà Morales, David Betancur Sánchez, Álvaro Barbero Jiménez, Marta Guerrero Nieto, Pablo Haya Coll, Patricia Martín Chozas, Elena Montiel Ponsoda  

**Link**: [PDF](https://arxiv.org/pdf/2501.15990)  

**Abstract**: Legal corpora for Natural Language Processing (NLP) are valuable and scarce resources in languages like Spanish due to two main reasons: data accessibility and legal expert knowledge availability. INESData 2024 is a European Union funded project lead by the Universidad Politécnica de Madrid (UPM) and developed by Instituto de Ingeniería del Conocimiento (IIC) to create a series of state-of-the-art NLP resources applied to the legal/administrative domain in Spanish. The goal of this paper is to present the Corpus of Legal Spanish Contract Clauses (3CEL), which is a contract information extraction corpus developed within the framework of INESData 2024. 3CEL contains 373 manually annotated tenders using 19 defined categories (4 782 total tags) that identify key information for contract understanding and reviewing. 

**Abstract (ZH)**: 法律语料库对于自然语言处理（NLP）来说是宝贵而稀缺的资源，在如西班牙语这样的语言中尤为突出，主要原因有两个：数据可访问性和法律专家知识的可用性。INESData 2024 是一个由欧洲联盟资助的项目，由马德里Polytechnic大学（UPM）领导，由知识工程研究所（IIC）开发，旨在创建一系列应用于西班牙语法律/行政领域的前沿NLP资源。本文的目的是介绍《西班牙语合同条款语料库 (3CEL)》，该语料库是在 INESData 2024 框架内开发的合同信息提取语料库。3CEL 包含373个手动标注的招标文件，使用19个定义好的类别（共计4,782个标签），这些标签能够标识合同理解和审查的关键信息。 

---
# Multi-View Attention Syntactic Enhanced Graph Convolutional Network for Aspect-based Sentiment Analysis 

**Title (ZH)**: 基于多视图注意力句法增强的图卷积网络在方面导向的情感分析中的应用 

**Authors**: Xiang Huang, Hao Peng, Shuo Sun, Zhifeng Hao, Hui Lin, Shuhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15968)  

**Abstract**: Aspect-based Sentiment Analysis (ABSA) is the task aimed at predicting the sentiment polarity of aspect words within sentences. Recently, incorporating graph neural networks (GNNs) to capture additional syntactic structure information in the dependency tree derived from syntactic dependency parsing has been proven to be an effective paradigm for boosting ABSA. Despite GNNs enhancing model capability by fusing more types of information, most works only utilize a single topology view of the dependency tree or simply conflate different perspectives of information without distinction, which limits the model performance. To address these challenges, in this paper, we propose a new multi-view attention syntactic enhanced graph convolutional network (MASGCN) that weighs different syntactic information of views using attention mechanisms. Specifically, we first construct distance mask matrices from the dependency tree to obtain multiple subgraph views for GNNs. To aggregate features from different views, we propose a multi-view attention mechanism to calculate the attention weights of views. Furthermore, to incorporate more syntactic information, we fuse the dependency type information matrix into the adjacency matrices and present a structural entropy loss to learn the dependency type adjacency matrix. Comprehensive experiments on four benchmark datasets demonstrate that our model outperforms state-of-the-art methods. The codes and datasets are available at this https URL. 

**Abstract (ZH)**: 基于方面的情感分析（Aspect-based Sentiment Analysis, ABSA）是指预测句子中方面词情感极性的任务。最近，通过将图神经网络（Graph Neural Networks, GNNs）引入依赖树，以捕捉从句法依存分析中衍生出来的依存树中的附加句法结构信息，已经被证明是一种增强ABSA性能的有效范式。尽管GNNs通过融合更多类型的信息来增强模型能力，但大多数工作仅仅利用依赖树的一种拓扑视图，或者简单地将不同视角的信息混淆为同一视图，这限制了模型的性能。为了解决这些挑战，本文提出了一种新的多视图注意力句法增强图卷积网络（Multiview Attention Syntactic Enhanced Graph Convolutional Network, MASGCN），利用注意力机制加权不同的句法视图信息。具体来说，我们首先从依赖树构建距离掩码矩阵，以获取多种子图视图供GNNs使用。为了从不同视图中聚合特征，我们提出了多视图注意力机制以计算视图的注意力权重。此外，为了引入更多的句法信息，我们将依存关系类型信息矩阵融合到相邻矩阵中，并提出了一种结构熵损失以学习依存关系类型的相邻矩阵。在四个基准数据集上的全面实验表明，我们的模型在性能上优于现有最先进的方法。相关代码和数据集可以在以下链接获取：[在此填入链接]。 

---
# Parametric Retrieval Augmented Generation 

**Title (ZH)**: 参数化检索增强生成 

**Authors**: Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15915)  

**Abstract**: Retrieval-augmented generation (RAG) techniques have emerged as a promising solution to enhance the reliability of large language models (LLMs) by addressing issues like hallucinations, outdated knowledge, and domain adaptation. In particular, existing RAG methods append relevant documents retrieved from external corpus or databases to the input of LLMs to guide their generation process, which we refer to as the in-context knowledge injection method. While this approach is simple and often effective, it has inherent limitations. Firstly, increasing the context length and number of relevant documents can lead to higher computational overhead and degraded performance, especially in complex reasoning tasks. More importantly, in-context knowledge injection operates primarily at the input level, but LLMs store their internal knowledge in their parameters. This gap fundamentally limits the capacity of in-context methods. To this end, we introduce Parametric retrieval-augmented generation (Parametric RAG), a new RAG paradigm that integrates external knowledge directly into the parameters of feed-forward networks (FFN) of an LLM through document parameterization. This approach not only saves online computational costs by eliminating the need to inject multiple documents into the LLMs' input context, but also deepens the integration of external knowledge into the parametric knowledge space of the LLM. Experimental results demonstrate that Parametric RAG substantially enhances both the effectiveness and efficiency of knowledge augmentation in LLMs. Also, it can be combined with in-context RAG methods to achieve even better performance.
We have open-sourced all the code, data, and models in the following anonymized GitHub link: this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）技术已经成为了提升大型语言模型（LLMs）可靠性的有前途的解决方案，尤其是在解决幻觉、过时的知识和领域适应性等问题方面表现出色。具体而言，现有的RAG方法通过将从外部语料库或数据库中检索到的相关文档附加到LLM的输入中，以指导生成过程，这种方法我们称之为上下文内知识注入方法。尽管这种方法简单且通常有效，但它具有内在局限性。首先，增加上下文长度和相关文档的数量会带来更高的计算开销和性能下降，尤其是在复杂的推理任务中。更重要的是，上下文内知识注入主要在输入级别进行，但LLM将其内部知识存储在参数中。这一差距从根本上限制了上下文内方法的能力。为了解决这些问题，我们引入了参数化检索增强生成（Parametric RAG），这是一种新的RAG范式，通过文档参数化将外部知识直接整合到LLM前馈网络（FFN）的参数中。这种方法不仅可以通过消除向LLM输入上下文注入多个文档的需求来节省在线计算成本，还能更深层次地将外部知识整合到LLM的参数化知识空间中。实验结果表明，Parametric RAG显著提升了LLM中知识增强的有效性和效率。此外，它还可以与上下文内RAG方法结合使用，实现更佳的性能。

我们已在去标识化的GitHub链接中开源了所有代码、数据和模型：[这个链接](this https URL) 

---
# Optimizing Sentence Embedding with Pseudo-Labeling and Model Ensembles: A Hierarchical Framework for Enhanced NLP Tasks 

**Title (ZH)**: 利用伪标签和模型集成优化句子嵌入：一种用于增强自然语言处理任务的分层框架 

**Authors**: Ziwei Liu, Qi Zhang, Lifu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15876)  

**Abstract**: Sentence embedding tasks are important in natural language processing (NLP), but improving their performance while keeping them reliable is still hard. This paper presents a framework that combines pseudo-label generation and model ensemble techniques to improve sentence embeddings. We use external data from SimpleWiki, Wikipedia, and BookCorpus to make sure the training data is consistent. The framework includes a hierarchical model with an encoding layer, refinement layer, and ensemble prediction layer, using ALBERT-xxlarge, RoBERTa-large, and DeBERTa-large models. Cross-attention layers combine external context, and data augmentation techniques like synonym replacement and back-translation increase data variety. Experimental results show large improvements in accuracy and F1-score compared to basic models, and studies confirm that cross-attention and data augmentation make a difference. This work presents an effective way to improve sentence embedding tasks and lays the groundwork for future NLP research. 

**Abstract (ZH)**: 句子嵌入任务在自然语言处理（NLP）中非常重要，但在提高其性能的同时保持其可靠性仍然具有挑战性。本文提出了一种结合伪标签生成和模型集成技术的框架，以改进句子嵌入。我们使用来自SimpleWiki、Wikipedia和BookCorpus的外部数据，确保训练数据的一致性。该框架包含一个层级模型，包括编码层、精炼层和集成预测层，使用了ALBERT-xxlarge、RoBERTa-large和DeBERTa-large模型。跨注意力层结合了外部上下文，而同义替换和反向翻译等数据扩增技术增加了数据的多样性。实验结果显示，与基本模型相比，准确率和F1得分有了显著提高，并且研究表明跨注意力和数据扩增确实起到了作用。这项工作提供了一种有效的方法来改进句子嵌入任务，为未来的NLP研究奠定了基础。 

---
# LCTG Bench: LLM Controlled Text Generation Benchmark 

**Title (ZH)**: LCTG 基准：由大语言模型控制的文本生成基准 

**Authors**: Kentaro Kurihara, Masato Mita, Peinan Zhang, Shota Sasaki, Ryosuke Ishigami, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2501.15875)  

**Abstract**: The rise of large language models (LLMs) has led to more diverse and higher-quality machine-generated text. However, their high expressive power makes it difficult to control outputs based on specific business instructions. In response, benchmarks focusing on the controllability of LLMs have been developed, but several issues remain: (1) They primarily cover major languages like English and Chinese, neglecting low-resource languages like Japanese; (2) Current benchmarks employ task-specific evaluation metrics, lacking a unified framework for selecting models based on controllability across different use cases. To address these challenges, this research introduces LCTG Bench, the first Japanese benchmark for evaluating the controllability of LLMs. LCTG Bench provides a unified framework for assessing control performance, enabling users to select the most suitable model for their use cases based on controllability. By evaluating nine diverse Japanese-specific and multilingual LLMs like GPT-4, we highlight the current state and challenges of controllability in Japanese LLMs and reveal the significant gap between multilingual models and Japanese-specific models. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起导致了生成文本的多样性和质量提高。然而，它们强大的表达能力使得基于特定业务指令控制输出变得困难。针对这一问题，已经发展出了关注LLM可控性的基准测试，但目前仍存在一些不足：（1）这些基准测试主要涵盖了如英语和汉语等常见语言，忽视了如日语这样的低资源语言；（2）当前的基准测试采用特定任务的评估指标，缺乏一个统一框架来根据可控性跨不同应用场景选择模型。为应对这些挑战，本研究引入了LCTG Bench，这是首个用于评估LLM可控性的日语基准测试。LCTG Bench提供了一个统一的评估框架，使用户能够根据可控性选择最适合其应用场景的模型。通过评估包括GPT-4在内的九种不同的日本特定和多语言LLM，我们展示了日语LLM当前的状态和可控性的挑战，并揭示了多语言模型与日本特定模型之间存在的显著差距。 

---
# Potential Applications of Artificial Intelligence for Cross-language Intelligibility Assessment of Dysarthric Speech 

**Title (ZH)**: 人工智能在失言者语言跨语言可懂度评估中的潜在应用 

**Authors**: Eunjung Yeo, Julie Liss, Visar Berisha, David Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15858)  

**Abstract**: Purpose: This commentary introduces how artificial intelligence (AI) can be leveraged to advance cross-language intelligibility assessment of dysarthric speech. Method: We propose a dual-component framework consisting of a universal module that generates language-independent speech representations and a language-specific intelligibility model that incorporates linguistic nuances. Additionally, we identify key barriers to cross-language intelligibility assessment, including data scarcity, annotation complexity, and limited linguistic insights, and present AI-driven solutions to overcome these challenges. Conclusion: Advances in AI offer transformative opportunities to enhance cross-language intelligibility assessment for dysarthric speech by balancing scalability across languages and adaptability by languages. 

**Abstract (ZH)**: 目的：本文探讨了如何利用人工智能（AI）推进失语型语音跨语言可理解性评估。方法：我们提出了一种双组件框架，包括一个通用模块，用于生成语言无关的语音表示，以及一个融合语言细微差别的语言特定可理解性模型。此外，我们还指出了跨语言可理解性评估面临的几个关键障碍，包括数据稀缺性、标注复杂性和有限的语言见解，并提出AI驱动的解决方案来克服这些挑战。结论：人工智能的进步为通过平衡语言间的可扩展性和语言间的适应性来增强失语型语音的跨语言可理解性评估提供了变革性的机会。 

---
# MADP: Multi-Agent Deductive Planning for Enhanced Cognitive-Behavioral Mental Health Question Answer 

**Title (ZH)**: MADP：增强认知行为心理健康问答的多智能体演绎规划 

**Authors**: Qi Chen, Dexi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15826)  

**Abstract**: The Mental Health Question Answer (MHQA) task requires the seeker and supporter to complete the support process in one-turn dialogue. Given the richness of help-seeker posts, supporters must thoroughly understand the content and provide logical, comprehensive, and well-structured responses. Previous works in MHQA mostly focus on single-agent approaches based on the cognitive element of Cognitive Behavioral Therapy (CBT), but they overlook the interactions among various CBT elements, such as emotion and cognition. This limitation hinders the models' ability to thoroughly understand the distress of help-seekers. To address this, we propose a framework named Multi-Agent Deductive Planning (MADP), which is based on the interactions between the various psychological elements of CBT. This method guides Large Language Models (LLMs) to achieve a deeper understanding of the seeker's context and provide more personalized assistance based on individual circumstances. Furthermore, we construct a new dataset based on the MADP framework and use it to fine-tune LLMs, resulting in a specialized model named MADP-LLM. We conduct extensive experiments, including comparisons with multiple LLMs, human evaluations, and automatic evaluations, to validate the effectiveness of the MADP framework and MADP-LLM. 

**Abstract (ZH)**: 认知行为疗法（CBT）要素下的心理卫生问答（MHQA）任务要求求助者和支持者在一次对话中完成支持过程。鉴于求助者的帖子内容丰富，支持者必须全面理解内容并提供逻辑性强、系统全面且结构良好的回复。以前的MHQA研究主要集中在基于CBT认知要素的单智能体方法上，但它们忽略了CBT中各种要素之间的交互，如情绪和认知。这一限制妨碍了模型全面理解求助者的困扰能力。为解决这一问题，我们提出了一种名为多智能体演绎规划（MADP）的框架，该框架基于CBT的各种心理要素之间的交互。该方法引导大规模语言模型（LLMs）更深入地理解求助者的背景，并根据个人情况提供更加个性化的帮助。此外，我们基于MADP框架构建了一个新的数据集，并使用该数据集对LLMs进行微调，从而创建了一种专门的模型MADP-LLM。我们进行了广泛实证研究，包括与多种LLMs的对比、人工评估和自动评估，以验证MADP框架和MADP-LLM的有效性。 

---
# Large Language Models to Diffusion Finetuning 

**Title (ZH)**: 大型语言模型应用于扩散微调 

**Authors**: Edoardo Cetin, Tianyu Zhao, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15781)  

**Abstract**: We propose a new finetuning method to provide pre-trained large language models (LMs) the ability to scale test-time compute through the diffusion framework. By increasing the number of diffusion steps, we show our finetuned models achieve monotonically increasing accuracy, directly translating to improved performance across downstream tasks. Furthermore, our finetuned models can expertly answer questions on specific topics by integrating powerful guidance techniques, and autonomously determine the compute required for a given problem by leveraging adaptive ODE solvers. Our method is universally applicable to any foundation model pre-trained with a cross-entropy loss and does not modify any of its original weights, fully preserving its strong single-step generation capabilities. We show our method is more effective and fully compatible with traditional finetuning approaches, introducing an orthogonal new direction to unify the strengths of the autoregressive and diffusion frameworks. 

**Abstract (ZH)**: 我们提出了一种新的微调方法，通过扩散框架赋予预训练的大语言模型（LMs）在测试时扩展计算量的能力。通过增加扩散步骤的数量，我们展示了微调后的模型在准确性上呈现单调递增的趋势，直接转化为下游任务上的性能提升。此外，我们的微调模型能够通过整合强大的指导技术，专家般地回答特定主题的问题，并通过利用自适应ODE求解器自主确定给定问题所需的计算量。该方法适用于任何基于交叉熵损失预训练的基础模型，并未修改其任何原始权重，完全保留了其强大的单步生成能力。我们证明该方法比传统微调方法更为有效，并且完全兼容传统微调方法，引入了一种新的正交方向，统一了自回归框架和扩散框架的优点。 

---
# Automatic Feedback Generation for Short Answer Questions using Answer Diagnostic Graphs 

**Title (ZH)**: 使用答案诊断图自动生成简答题反馈 

**Authors**: Momoka Furuhashi, Hiroaki Funayama, Yuya Iwase, Yuichiroh Matsubayashi, Yoriko Isobe, Toru Nagahama, Saku Sugawara, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2501.15777)  

**Abstract**: Short-reading comprehension questions help students understand text structure but lack effective feedback. Students struggle to identify and correct errors, while manual feedback creation is labor-intensive. This highlights the need for automated feedback linking responses to a scoring rubric for deeper comprehension.
Despite advances in Natural Language Processing (NLP), research has focused on automatic grading, with limited work on feedback generation. To address this, we propose a system that generates feedback for student responses.
Our contributions are twofold. First, we introduce the first system for feedback on short-answer reading comprehension. These answers are derived from the text, requiring structural understanding. We propose an "answer diagnosis graph," integrating the text's logical structure with feedback templates. Using this graph and NLP techniques, we estimate students' comprehension and generate targeted feedback.
Second, we evaluate our feedback through an experiment with Japanese high school students (n=39). They answered two 70-80 word questions and were divided into two groups with minimal academic differences. One received a model answer, the other system-generated feedback. Both re-answered the questions, and we compared score changes. A questionnaire assessed perceptions and motivation.
Results showed no significant score improvement between groups, but system-generated feedback helped students identify errors and key points in the text. It also significantly increased motivation. However, further refinement is needed to enhance text structure understanding. 

**Abstract (ZH)**: 短阅读理解题可以帮助学生理解文本结构，但缺乏有效的反馈。学生们在识别和纠正错误方面遇到困难，而人工反馈的创建也非常耗时。这突显了需要自动反馈系统，该系统能够将学生的回答链接到评分标准，从而促进更深入的理解。

尽管自然语言处理（NLP）取得了进展，但研究主要集中在自动评分上，关于反馈生成的工作相对较少。为了解决这个问题，我们提出了一种生成学生回答反馈的系统。

我们的贡献有两个方面。首先，我们引入了第一个针对短答案阅读理解的反馈系统。这些答案是从文本中得出的，需要结构理解。我们提出了一种“答案诊断图”，将文本的逻辑结构与反馈模板结合起来。通过这种图和自然语言处理技术，我们估计学生的理解程度并生成针对性的反馈。

其次，我们通过一项针对日本高中生（n=39）的实验，评估了我们的反馈效果。他们回答了两个约70-80词的问题，并被分为两个小组，两组在学术背景上的差异较小。一组收到了一个模型答案，另一组收到了系统生成的反馈。两组都重新回答了这些问题，并比较了分数变化。问卷调查评估了他们的感知和动机。

实验结果显示，两组之间的分数没有显著提升，但系统生成的反馈帮助学生识别了错误和文本中的关键点，并显著提高了他们的动机。然而，为了进一步提升对文本结构的理解，还需要进一步改进。 

---
# Is It Navajo? Accurate Language Detection in Endangered Athabaskan Languages 

**Title (ZH)**: 《是纳瓦霍语吗？濒危阿纳伯卡斯克语系语言的准确识别》

这个翻译既保留了原文含义，又符合学术论文标题的规范。其中，“纳瓦霍语”是对“Navajo”的常见翻译，“阿纳伯卡斯克语系”是对“Athabaskan languages”的翻译，考虑到语境和学术准确性，“濒危”一词适合描述处于危险状态的语言。 

**Authors**: Ivory Yang, Weicheng Ma, Chunhui Zhang, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2501.15773)  

**Abstract**: Endangered languages, such as Navajo - the most widely spoken Native American language - are significantly underrepresented in contemporary language technologies, exacerbating the challenges of their preservation and revitalization. This study evaluates Google's large language model (LLM)-based language identification system, which consistently misidentifies Navajo, exposing inherent limitations when applied to low-resource Native American languages. To address this, we introduce a random forest classifier trained on Navajo and eight frequently confused languages. Despite its simplicity, the classifier achieves near-perfect accuracy (97-100%), significantly outperforming Google's LLM-based system. Additionally, the model demonstrates robustness across other Athabaskan languages - a family of Native American languages spoken primarily in Alaska, the Pacific Northwest, and parts of the Southwestern United States - suggesting its potential for broader application. Our findings underscore the pressing need for NLP systems that prioritize linguistic diversity and adaptability over centralized, one-size-fits-all solutions, especially in supporting underrepresented languages in a multicultural world. This work directly contributes to ongoing efforts to address cultural biases in language models and advocates for the development of culturally localized NLP tools that serve diverse linguistic communities. 

**Abstract (ZH)**: 濒危语言，如纳瓦霍语——北美使用最广泛的原住民语言——在当代语言技术中严重缺失，加剧了其保护和复兴的挑战。本研究评估了基于谷歌大型语言模型（LLM）的语言识别系统，该系统在识别纳瓦霍语时普遍存在错误，暴露了其在低资源原住民语言应用中固有的局限性。为了解决这一问题，我们引入了一种基于纳瓦霍语和八种经常混淆的语言训练的随机森林分类器。尽管该分类器较为简单，但其准确率接近完美（97%-100%），显著优于谷歌基于LLM的语言识别系统。此外，该模型在其他阿尔巴肯斯语系语言（北美主要在阿拉斯加、太平洋西北地区以及美国西南部使用的原住民语言之一）中也表现出较高的稳定性，表明其具有更广泛的应用潜力。研究结果强调了在支持多文化世界中被边缘化的语言时，NLP系统需要优先考虑语言多样性和灵活性而非集中化的一刀切解决方案的紧迫性。本研究直接贡献了应对语言模型中的文化偏见的努力，并倡导开发能够适应多元语言社区的本地化NLP工具。 

---
# Weight-based Analysis of Detokenization in Language Models: Understanding the First Stage of Inference Without Inference 

**Title (ZH)**: 基于权重分析的语言模型去令牌化研究：理解推理的第一阶段而不进行推理 

**Authors**: Go Kamoda, Benjamin Hienzerling, Tatsuro Inaba, Keito Kudo, Keisuke Sakaguchi, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2501.15754)  

**Abstract**: According to the stages-of-inference hypothesis, early layers of language models map their subword-tokenized input, which does not necessarily correspond to a linguistically meaningful segmentation, to more meaningful representations that form the model's ``inner vocabulary''. Prior analysis of this detokenization stage has predominantly relied on probing and interventions such as path patching, which involve selecting particular inputs, choosing a subset of components that will be patched, and then observing changes in model behavior. Here, we show that several important aspects of the detokenization stage can be understood purely by analyzing model weights, without performing any model inference steps. Specifically, we introduce an analytical decomposition of first-layer attention in GPT-2. Our decomposition yields interpretable terms that quantify the relative contributions of position-related, token-related, and mixed effects. By focusing on terms in this decomposition, we discover weight-based explanations of attention bias toward close tokens and attention for detokenization. 

**Abstract (ZH)**: 根据推理阶段假说，语言模型的早期层会将其子词分词输入映射为更具有语义意义的表示，这些表示构成了模型的“内部词汇表”。此前对去分词阶段的分析主要依赖于探测和干预方法，如路径修正等，这些方法涉及选择特定的输入样本、选择将要更正的组件，然后观察模型行为的变化。在这里，我们展示了一些重要方面可以通过单纯分析模型权重来理解，而无需执行任何模型推理步骤。具体来说，我们引入了对GPT-2第一层注意力机制的分析分解。该分解提供了可解释的术语，量化了位置相关、令牌相关以及混合效应的相对贡献。通过关注该分解中的术语，我们发现了基于权重的注意力偏向于临近令牌和去分词注意力的解释。 

---
# IndicMMLU-Pro: Benchmarking the Indic Large Language Models 

**Title (ZH)**: IndicMMLU-Pro：评估印度语系大型语言模型 

**Authors**: Sankalp KJ, Ashutosh Kumar, Laxmaan Balaji, Nikunj Kotecha, Vinija Jain, Aman Chadha, Sreyoshi Bhaduri  

**Link**: [PDF](https://arxiv.org/pdf/2501.15747)  

**Abstract**: Known by more than 1.5 billion people in the Indian subcontinent, Indic languages present unique challenges and opportunities for natural language processing (NLP) research due to their rich cultural heritage, linguistic diversity, and complex structures. IndicMMLU-Pro is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) across Indic languages, building upon the MMLU Pro (Massive Multitask Language Understanding) framework. Covering major languages such as Hindi, Bengali, Gujarati, Marathi, Kannada, Punjabi, Tamil, Telugu, and Urdu, our benchmark addresses the unique challenges and opportunities presented by the linguistic diversity of the Indian subcontinent. This benchmark encompasses a wide range of tasks in language comprehension, reasoning, and generation, meticulously crafted to capture the intricacies of Indian languages. IndicMMLU-Pro provides a standardized evaluation framework to push the research boundaries in Indic language AI, facilitating the development of more accurate, efficient, and culturally sensitive models. This paper outlines the benchmarks' design principles, task taxonomy, data collection methodology, and presents baseline results from state-of-the-art multilingual models. 

**Abstract (ZH)**: 印度次大陆的15亿多人口所熟知的印度语言在自然语言处理（NLP）研究中因其丰富的文化遗产、语言多样性和复杂结构而面临着独特的挑战和机遇。IndicMMLU-Pro 是一个全面的基准测试，旨在评估大型语言模型（LLMs）在印度语言中的表现，该基准测试是在MMLU Pro（大规模多任务语言理解）框架的基础上构建的。涵盖哈林语、孟加拉语、古吉拉特语、马拉地语、卡纳达语、旁遮普语、泰米尔语、泰卢固语和乌尔都语等主要语言，该基准测试针对印度次大陆语言多样性的独特挑战和机遇。这一基准测试涵盖了语言理解、推理和生成等广泛的任务，精心设计以捕捉印度语言的复杂性。IndicMMLU-Pro 提供了一个标准化的评估框架，以推动印度语言AI的研究边界，促进更准确、高效和文化敏感模型的发展。本文概述了基准测试的设计原则、任务分类学、数据收集方法，并展示了最先进的多语言模型的基线结果。 

---
# ESGSenticNet: A Neurosymbolic Knowledge Base for Corporate Sustainability Analysis 

**Title (ZH)**: ESGSenticNet: 一种企业可持续性分析的神经符号知识库 

**Authors**: Keane Ong, Rui Mao, Frank Xing, Ranjan Satapathy, Johan Sulaeman, Erik Cambria, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15720)  

**Abstract**: Evaluating corporate sustainability performance is essential to drive sustainable business practices, amid the need for a more sustainable economy. However, this is hindered by the complexity and volume of corporate sustainability data (i.e. sustainability disclosures), not least by the effectiveness of the NLP tools used to analyse them. To this end, we identify three primary challenges - immateriality, complexity, and subjectivity, that exacerbate the difficulty of extracting insights from sustainability disclosures. To address these issues, we introduce ESGSenticNet, a publicly available knowledge base for sustainability analysis. ESGSenticNet is constructed from a neurosymbolic framework that integrates specialised concept parsing, GPT-4o inference, and semi-supervised label propagation, together with a hierarchical taxonomy. This approach culminates in a structured knowledge base of 44k knowledge triplets - ('halve carbon emission', supports, 'emissions control'), for effective sustainability analysis. Experiments indicate that ESGSenticNet, when deployed as a lexical method, more effectively captures relevant and actionable sustainability information from sustainability disclosures compared to state of the art baselines. Besides capturing a high number of unique ESG topic terms, ESGSenticNet outperforms baselines on the ESG relatedness and ESG action orientation of these terms by 26% and 31% respectively. These metrics describe the extent to which topic terms are related to ESG, and depict an action toward ESG. Moreover, when deployed as a lexical method, ESGSenticNet does not require any training, possessing a key advantage in its simplicity for non-technical stakeholders. 

**Abstract (ZH)**: 评估企业可持续性绩效对于推动可持续商业实践至关重要，特别是在需要更可持续的经济背景下。然而，这受到了企业可持续性数据的复杂性和数量（即可持续性披露）的阻碍，尤其是所使用的NLP工具的有效性。为此，我们识别了三个主要挑战——非物质性、复杂性和主观性，这些挑战加剧了从可持续性披露中提取见解的难度。为解决这些问题，我们引入了ESGSenticNet，这是一种用于可持续性分析的公开可获取知识库。ESGSenticNet基于神经符号框架构建，结合了专门概念解析、GPT-4o推理和半监督标签传播，以及层次分类学。此方法 culminates 于包含 44,000 个知识三元组的结构化知识库——例如，“减少碳排放”支持“排放控制”，以有效地进行可持续性分析。实验表明，当作为词法方法部署时，ESGSenticNet比最先进的基线更能有效地捕捉可持续性披露中的相关和可操作的信息。除了捕获大量独特的ESG主题术语外，ESGSenticNet在ESG相关性和ESG行动导向性方面的表现也比基线分别高出26%和31%。这些指标描述了主题术语与ESG的相关程度及其行动导向性。此外，当作为词法方法部署时，ESGSenticNet不需要任何训练，这使其在非技术利益相关者中具有显著的优势，因为它非常简单易用。 

---
# StaICC: Standardized Evaluation for Classification Task in In-context Learning 

**Title (ZH)**: StaICC：上下文学习中分类任务的标准化评估方法 

**Authors**: Hakaze Cho, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2501.15708)  

**Abstract**: Classification tasks are widely investigated in the In-Context Learning (ICL) paradigm. However, current efforts are evaluated on disjoint benchmarks and settings, while their performances are significantly influenced by some trivial variables, such as prompt templates, data sampling, instructions, etc., which leads to significant inconsistencies in the results reported across various literature, preventing fair comparison or meta-analysis across different papers. Therefore, this paper proposes a standardized and easy-to-use evaluation toolkit (StaICC) for in-context classification. Including, for the normal classification task, we provide StaICC-Normal, selecting 10 widely used datasets, and generating prompts with a fixed form, to mitigate the variance among the experiment implementations. To enrich the usage of our benchmark, we also provide a sub-benchmark StaICC-Diag for diagnosing ICL from several aspects, aiming for a more robust inference processing. 

**Abstract (ZH)**: 在基于上下文学习（ICL）范式下，分类任务得到了广泛研究。然而，当前的努力主要在不相连的基准和设置上进行评估，这些基准和设置受到诸如提示模板、数据采样、指令等一些琐碎变量的显著影响，导致结果在不同文献中的报告具有极大的不一致性，阻碍了跨论文的公平比较或元分析。因此，本文提出了一种标准化且易于使用的评估工具包（StaICC），以评估ICL范下的分类任务。对于普通的分类任务，我们提供StaICC-Normal，选择了10个广泛使用的数据集，并生成固定形式的提示，以减少实验实现之间的差异性。为了丰富我们基准的使用，我们还提供了StaICC-Diag子基准，从多个方面诊断ICL，旨在进行更 robust 的推理处理。 

---
# Adapting Biomedical Abstracts into Plain language using Large Language Models 

**Title (ZH)**: 使用大型语言模型将生物医学摘要转换为通俗语言 

**Authors**: Haritha Gangavarapu, Giridhar Kaushik Ramachandran, Kevin Lybarger, Meliha Yetisgen, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2501.15700)  

**Abstract**: A vast amount of medical knowledge is available for public use through online health forums, and question-answering platforms on social media. The majority of the population in the United States doesn't have the right amount of health literacy to make the best use of that information. Health literacy means the ability to obtain and comprehend the basic health information to make appropriate health decisions. To build the bridge between this gap, organizations advocate adapting this medical knowledge into plain language. Building robust systems to automate the adaptations helps both medical and non-medical professionals best leverage the available information online. The goal of the Plain Language Adaptation of Biomedical Abstracts (PLABA) track is to adapt the biomedical abstracts in English language extracted from PubMed based on the questions asked in MedlinePlus for the general public using plain language at the sentence level. As part of this track, we leveraged the best open-source Large Language Models suitable and fine-tuned for dialog use cases. We compare and present the results for all of our systems and our ranking among the other participants' submissions. Our top performing GPT-4 based model ranked first in the avg. simplicity measure and 3rd on the avg. accuracy measure. 

**Abstract (ZH)**: 通过在线健康论坛和社交媒体上的问答平台，大量医疗知识可供公众使用。然而，美国大多数人口缺乏足够的健康素养，无法充分利用这些信息。健康素养是指获取和理解基本健康信息以作出适当健康决策的能力。为了弥合这一差距，组织倡导将这些医学知识转化为易于理解的语言。构建能够自动化这一转换的 robust 系统，有助于医学和非医学专业人士更好地利用互联网上的可用信息。平易近人语言适应生物医学摘要（PLABA）赛道的目标是基于美国医学指南中的问题，将从 PubMed 提取的英文生物医学摘要转化为平易近人语言，从句层面进行调整。作为该赛道的一部分，我们利用了适合对话应用的最佳开源大型语言模型，并对其进行了微调。我们比较了所有系统的结果，并展示了我们在其他参赛者提交中的排名。基于 GPT-4 的顶级模型在平均简洁性指标中排名第一，在平均准确度指标中排名第三。 

---
# Transformer-Based Multimodal Knowledge Graph Completion with Link-Aware Contexts 

**Title (ZH)**: 基于变换器的具有链接意识上下文的多模态知识图谱完成方法 

**Authors**: Haodi Ma, Dzmitry Kasinets, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15688)  

**Abstract**: Multimodal knowledge graph completion (MMKGC) aims to predict missing links in multimodal knowledge graphs (MMKGs) by leveraging information from various modalities alongside structural data. Existing MMKGC approaches primarily extend traditional knowledge graph embedding (KGE) models, which often require creating an embedding for every entity. This results in large model sizes and inefficiencies in integrating multimodal information, particularly for real-world graphs. Meanwhile, Transformer-based models have demonstrated competitive performance in knowledge graph completion (KGC). However, their focus on single-modal knowledge limits their capacity to utilize cross-modal information. Recently, Large vision-language models (VLMs) have shown potential in cross-modal tasks but are constrained by the high cost of training. In this work, we propose a novel approach that integrates Transformer-based KGE models with cross-modal context generated by pre-trained VLMs, thereby extending their applicability to MMKGC. Specifically, we employ a pre-trained VLM to transform relevant visual information from entities and their neighbors into textual sequences. We then frame KGC as a sequence-to-sequence task, fine-tuning the model with the generated cross-modal context. This simple yet effective method significantly reduces model size compared to traditional KGE approaches while achieving competitive performance across multiple large-scale datasets with minimal hyperparameter tuning. 

**Abstract (ZH)**: 多模态知识图谱完成（MMKGC）旨在通过利用各种模态的信息以及结构数据来预测多模态知识图谱（MMKGs）中的缺失链接。现有的MMKGC方法主要扩展了传统的知识图嵌入（KGE）模型，这些模型通常需要为每个实体创建嵌入，这导致了较大的模型规模，并且在整合多模态信息时效率低下，特别是对于实际图而言。同时，基于Transformer的模型在知识图谱完成（KGC）方面已经展示了竞争力。然而，它们主要关注单模态知识，限制了其利用跨模态信息的能力。最近，大型的多模态视觉语言模型（VLMs）在跨模态任务中显示出潜力，但受限于训练成本高昂。在本工作中，我们提出了一种新的方法，将基于Transformer的知识图嵌入模型与预训练的VLM生成的跨模态上下文相结合，从而将其应用扩展到MMKGC。具体而言，我们利用预训练的VLM将实体及其邻居的相关视觉信息转换为文本序列。然后，我们将KGC建模为序列到序列的任务，并通过生成的跨模态上下文对模型进行微调。这一简单有效的方案相较于传统的KGE方法显著减少了模型规模，在多个大规模数据集上实现了竞争力的性能，并且只需微量超参数调整。 

---
# TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs 

**Title (ZH)**: TensorLLM：通过增强推理和压缩的多头注意力张量表示在大规模语言模型中的应用 

**Authors**: Yuxuan Gu, Wuyang Zhou, Giorgos Iacovides, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2501.15674)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) can be improved by structurally denoising their weights, yet existing techniques primarily focus on denoising the feed-forward network (FFN) of the transformer block, and can not efficiently utilise the Multi-head Attention (MHA) block, which is the core of transformer architectures. To address this issue, we propose a novel intuitive framework that, at its very core, performs MHA compression through a multi-head tensorisation process and the Tucker decomposition. This enables both higher-dimensional structured denoising and compression of the MHA weights, by enforcing a shared higher-dimensional subspace across the weights of the multiple attention heads. We demonstrate that this approach consistently enhances the reasoning capabilities of LLMs across multiple benchmark datasets, and for both encoder-only and decoder-only architectures, while achieving compression rates of up to $\sim 250$ times in the MHA weights, all without requiring any additional data, training, or fine-tuning. Furthermore, we show that the proposed method can be seamlessly combined with existing FFN-only-based denoising techniques to achieve further improvements in LLM reasoning performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力可以通过结构化去除其权重中的噪声来提高，但现有的技术主要集中在去除变压器块中的前向网络（FFN）噪声，而无法有效地利用Multi-head Attention（MHA）块，这是变压器架构的核心部分。为了解决这一问题，我们提出了一种新颖直观的框架，该框架的核心是通过多头张量化过程和Tucker分解来执行MHA压缩。这种方法通过在多个注意头的权重中强制共享更高维度的子空间，实现了更高维度的结构化去除噪声和MHA权重的压缩。我们证明了这种做法在多个基准数据集上一致地增强了LLMs的推理能力，无论是在仅编码器架构还是仅解码器架构中，同时在MHA权重的压缩率最高可达约250倍，而无需添加额外的数据、训练或微调。此外，我们展示了所提出的方法可以无缝结合现有的仅基于FFN的去噪技术，进一步提高LLMs的推理性能。 

---
# People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text 

**Title (ZH)**: 频繁使用ChatGPT进行写作任务的人是检测AI生成文本的准确且可靠的检测者 

**Authors**: Jenna Russell, Marzena Karpinska, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2501.15654)  

**Abstract**: In this paper, we study how well humans can detect text generated by commercial LLMs (GPT-4o, Claude, o1). We hire annotators to read 300 non-fiction English articles, label them as either human-written or AI-generated, and provide paragraph-length explanations for their decisions. Our experiments show that annotators who frequently use LLMs for writing tasks excel at detecting AI-generated text, even without any specialized training or feedback. In fact, the majority vote among five such "expert" annotators misclassifies only 1 of 300 articles, significantly outperforming most commercial and open-source detectors we evaluated even in the presence of evasion tactics like paraphrasing and humanization. Qualitative analysis of the experts' free-form explanations shows that while they rely heavily on specific lexical clues ('AI vocabulary'), they also pick up on more complex phenomena within the text (e.g., formality, originality, clarity) that are challenging to assess for automatic detectors. We release our annotated dataset and code to spur future research into both human and automated detection of AI-generated text. 

**Abstract (ZH)**: 在本文中，我们研究了人类在检测由商用大型语言模型（如GPT-4o、Claude、o1）生成的文本时的准确程度。我们雇佣了注释员阅读300篇非小说类英文文章，并将其标记为人写还是AI生成，同时提供段落长度的解释来说明他们的判断依据。实验结果显示，那些经常使用LLM进行写作任务的注释员在检测AI生成文本方面表现出色，即便没有任何专门的训练或反馈也是如此。实际上，在五位这样的“专家”注释员的多数投票中，仅有一篇文章被错误分类，这在我们评估的大多数商用和开源检测器中具有显著优势，即使在面对如改写和拟人化等规避策略时也是如此。对专家自由格式解释的定性分析表明，尽管他们高度依赖特定的词汇线索（如“AI术语”），但他们也能够识别文本中的复杂现象（如正式性、原创性和清晰度），而这些是自动检测器难以评估的。我们发布了标注的数据集和代码，以促进对AI生成文本的人工和自动检测的未来研究。 

---
# Quantum-Enhanced Attention Mechanism in NLP: A Hybrid Classical-Quantum Approach 

**Title (ZH)**: 量子增强的注意力机制在自然语言处理中的应用：一种经典与量子相结合的方法 

**Authors**: S.M. Yousuf Iqbal Tomal, Abdullah Al Shafin, Debojit Bhattacharjee, MD. Khairul Amin, Rafiad Sadat Shahir  

**Link**: [PDF](https://arxiv.org/pdf/2501.15630)  

**Abstract**: Transformer-based models have achieved remarkable results in natural language processing (NLP) tasks such as text classification and machine translation. However, their computational complexity and resource demands pose challenges for scalability and accessibility. This research proposes a hybrid quantum-classical transformer model that integrates a quantum-enhanced attention mechanism to address these limitations. By leveraging quantum kernel similarity and variational quantum circuits (VQC), the model captures intricate token dependencies while improving computational efficiency. Experimental results on the IMDb dataset demonstrate that the quantum-enhanced model outperforms the classical baseline across all key metrics, achieving a 1.5% improvement in accuracy (65.5% vs. 64%), precision, recall, and F1 score. Statistical significance tests validate these improvements, highlighting the robustness of the quantum approach. These findings illustrate the transformative potential of quantum-enhanced attention mechanisms in optimizing NLP architectures for real-world applications. 

**Abstract (ZH)**: 基于变换器的模型在自然语言处理（NLP）任务如文本分类和机器翻译中取得了显著成果。然而，它们在计算复杂性和资源需求方面的挑战限制了规模化和可访问性。本研究提出了一种结合了量子增强注意机制的混合量子-经典变换器模型，以解决这些问题。通过利用量子内核相似性和变分量子电路（VQC），该模型能够捕捉到复杂的词元依赖关系，同时提高计算效率。在IMDb数据集上的实验结果表明，量子增强的模型在所有关键指标上均优于经典的基线模型，其准确率、精确率、召回率和F1分数分别提高了1.5%（从64%提升到65.5%）。统计显著性检验验证了这些改进，展示了量子方法的稳健性。这些发现表明，量子增强的注意机制在优化面向实际应用的NLP架构方面具有变革性潜力。 

---
# Improving Estonian Text Simplification through Pretrained Language Models and Custom Datasets 

**Title (ZH)**: 通过预训练语言模型和自定义数据集提高爱沙尼亚文本简化效果 

**Authors**: Eduard Barbu, Meeri-Ly Muru, Sten Marcus Malva  

**Link**: [PDF](https://arxiv.org/pdf/2501.15624)  

**Abstract**: This study introduces an approach to Estonian text simplification using two model architectures: a neural machine translation model and a fine-tuned large language model (LLaMA). Given the limited resources for Estonian, we developed a new dataset, the Estonian Simplification Dataset, combining translated data and GPT-4.0-generated simplifications. We benchmarked OpenNMT, a neural machine translation model that frames text simplification as a translation task, and fine-tuned the LLaMA model on our dataset to tailor it specifically for Estonian simplification. Manual evaluations on the test set show that the LLaMA model consistently outperforms OpenNMT in readability, grammaticality, and meaning preservation. These findings underscore the potential of large language models for low-resource languages and provide a basis for further research in Estonian text simplification. 

**Abstract (ZH)**: 本研究介绍了一种利用两个模型架构对爱沙尼亚文本进行简化的方法：神经机器翻译模型和微调的大语言模型（LLaMA）。鉴于爱沙尼亚语资源有限，我们开发了一个新的数据集——爱沙尼亚简化数据集，该数据集结合了翻译数据和GPT-4.0生成的简化文本。我们使用OpenNMT（一种将文本简化视为翻译任务的神经机器翻译模型）进行了基准测试，并对我们的数据集进行了LLaMA模型的微调，使其专门适用于爱沙尼亚语的简化任务。在测试集上的手工评估表明，LLaMA模型在易读性、语法正确性和意义保留方面始终优于OpenNMT。这些发现强调了大语言模型在低资源语言中的潜力，并为爱沙尼亚文本简化进一步研究提供了基础。 

---
# SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain 

**Title (ZH)**: SCP-116K：高质量问题解决方案数据集及高等教育科学领域自动提取的一般化管道 

**Authors**: Dakuan Lu, Xiaoyu Tan, Rui Xu, Tianchu Yao, Chao Qu, Wei Chu, Yinghui Xu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2501.15587)  

**Abstract**: Recent breakthroughs in large language models (LLMs) exemplified by the impressive mathematical and scientific reasoning capabilities of the o1 model have spotlighted the critical importance of high-quality training data in advancing LLM performance across STEM disciplines. While the mathematics community has benefited from a growing body of curated datasets, the scientific domain at the higher education level has long suffered from a scarcity of comparable resources. To address this gap, we present SCP-116K, a new large-scale dataset of 116,756 high-quality problem-solution pairs, automatically extracted from heterogeneous sources using a streamlined and highly generalizable pipeline. Our approach involves stringent filtering to ensure the scientific rigor and educational level of the extracted materials, while maintaining adaptability for future expansions or domain transfers. By openly releasing both the dataset and the extraction pipeline, we seek to foster research on scientific reasoning, enable comprehensive performance evaluations of new LLMs, and lower the barrier to replicating the successes of advanced models like o1 in the broader science community. We believe SCP-116K will serve as a critical resource, catalyzing progress in high-level scientific reasoning tasks and promoting further innovations in LLM development. The dataset and code are publicly available at this https URL. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的突破，尤其是o1模型在数学和科学推理解题方面的强大能力，凸显了高质量训练数据在跨STEM领域提升LLM性能中的关键重要性。虽然数学界受益于日益增长的高质量数据集，但高等教育领域的科学界长期缺乏类似的资源。为填补这一空白，我们提出了一种新的大规模数据集——SCP-116K，该数据集包含116,756个高质量的问题-解决方案对，通过简化且高度可扩展的流水线从异构来源自动提取。我们的方法包括严格的筛选，以确保提取材料的科学严谨性和教育水平，并保持对未来扩展或领域转移的适应性。通过开放性地发布数据集和提取流水线，我们旨在促进科学推理研究，使全面评估新LLM的性能成为可能，并降低复制如o1等先进模型在更广泛科学界的成功门槛。我们相信，SCP-116K 将成为一个关键资源，推动高水平科学推理任务的进展，并促进LLM开发的进一步创新。该数据集和代码可在以下网址公开获取：这个 https URL。 

---
# Error Classification of Large Language Models on Math Word Problems: A Dynamically Adaptive Framework 

**Title (ZH)**: 大型语言模型在数学文字题上的错误分类：一种动态适应性框架 

**Authors**: Yuhong Sun, Zhangyue Yin, Xuanjing Huang, Xipeng Qiu, Hui Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15581)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. Math Word Problems (MWPs) serve as a crucial benchmark for evaluating LLMs' reasoning abilities. While most research primarily focuses on improving accuracy, it often neglects understanding and addressing the underlying patterns of errors. Current error classification methods rely on static and predefined categories, which limit their ability to capture the full spectrum of error patterns in mathematical reasoning. To enable systematic error analysis, we collect error samples from 15 different LLMs of varying sizes across four distinct MWP datasets using multiple sampling strategies. Based on this extensive collection, we introduce MWPES-300K, a comprehensive dataset containing 304,865 error samples that cover diverse error patterns and reasoning paths. To reduce human bias and enable fine-grained analysis of error patterns, we propose a novel framework for automated dynamic error classification in mathematical reasoning. Experimental results demonstrate that dataset characteristics significantly shape error patterns, which evolve from basic to complex manifestations as model capabilities increase. With deeper insights into error patterns, we propose error-aware prompting that incorporates common error patterns as explicit guidance, leading to significant improvements in mathematical reasoning performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域展现了出色的性能。数学应用题（MWPs）是评估LLMs推理能力的重要基准。尽管大多数研究主要集中在提高准确性上，但往往会忽略对错误背后模式的理解和解决。当前的错误分类方法依赖于静态和预定义的类别，这限制了它们捕捉数学推理中错误模式全面性的能力。为了实现系统性的错误分析，我们通过多种采样策略从四个不同的MWP数据集中收集了15个不同规模LLMs的错误样本。基于这一广泛的收集，我们引入了MWPES-300K，这是一个包含304,865个错误样本的综合数据集，涵盖了多样化的错误模式和推理路径。为了减少人为偏见并实现精细的错误模式分析，我们提出了一种新的自动动态错误分类框架，专门用于数学推理。实验结果表明，数据集的特性显著影响了错误模式，随着模型能力的增强，错误模式从基础的逐步演变为复杂的形态。通过深入了解错误模式，我们提出了一种错误感知的提示方法，结合常见错误模式作为显式指导，这在数学推理性能上取得了显著提升。 

---
# Instruction Tuning for Story Understanding and Generation with Weak Supervision 

**Title (ZH)**: 带有弱监督的故事情节理解与生成指令调优 

**Authors**: Yangshu Yuan, Heng Chen, Christian Ng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15574)  

**Abstract**: Story understanding and generation have long been a challenging task in natural language processing (NLP), especially when dealing with various levels of instruction specificity. In this paper, we propose a novel approach called "Weak to Strong Instruction Tuning" for improving story generation by tuning models with instructions of varying clarity. We explore the potential of large language models (LLMs) to adapt to different types of instructions, weak and strong, and show that our method significantly enhances performance in story comprehension and generation. By leveraging the strength of instruction tuning, we train models to understand the nuances of story plots, characters, and themes while generating coherent and engaging narratives. Through extensive experiments on several benchmark datasets and comparison with state-of-the-art baselines, we demonstrate that our method outperforms existing techniques, yielding substantial improvements in both automatic evaluation metrics and human evaluations. Our work shows that adaptive instruction tuning can be a powerful tool in refining generative models for complex narrative tasks. 

**Abstract (ZH)**: 故事理解与生成一直是自然语言处理（NLP）领域的一项具有挑战性的任务，尤其是在处理不同层次的指令明确性时。本文提出了一种名为“从弱到强指令微调”的新方法，通过使用不同程度清晰度的指令来提高故事生成的效果。我们探索了大型语言模型（LLMs）适应不同类型指令（弱指令和强指令）的潜力，并表明我们的方法在故事理解和生成方面显著提升了性能。通过利用指令微调的优势，我们训练模型理解故事剧情、角色和主题的细微差别，生成连贯且吸引人的叙述。通过在多个基准数据集上进行广泛的实验，并与最新基准技术进行比较，我们证明了我们的方法优于现有技术，不仅在自动评价指标方面，还在人类评价方面都取得了显著的改进。我们的研究表明，适应性指令微调可以成为改进复杂叙事任务生成模型的强大工具。 

---
# Cross-Cultural Fashion Design via Interactive Large Language Models and Diffusion Models 

**Title (ZH)**: 通过交互式大型语言模型和扩散模型进行跨文化服装设计 

**Authors**: Spencer Ramsey, Amina Grant, Jeffrey Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15571)  

**Abstract**: Fashion content generation is an emerging area at the intersection of artificial intelligence and creative design, with applications ranging from virtual try-on to culturally diverse design prototyping. Existing methods often struggle with cultural bias, limited scalability, and alignment between textual prompts and generated visuals, particularly under weak supervision. In this work, we propose a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) to address these challenges. Our method leverages LLMs for semantic refinement of textual prompts and introduces a weak supervision filtering module to effectively utilize noisy or weakly labeled data. By fine-tuning the LDM on an enhanced DeepFashion+ dataset enriched with global fashion styles, the proposed approach achieves state-of-the-art performance. Experimental results demonstrate that our method significantly outperforms baselines, achieving lower Frechet Inception Distance (FID) and higher Inception Scores (IS), while human evaluations confirm its ability to generate culturally diverse and semantically relevant fashion content. These results highlight the potential of LLM-guided diffusion models in driving scalable and inclusive AI-driven fashion innovation. 

**Abstract (ZH)**: 时尚内容生成是人工智能与创意设计交叉领域中一个新兴的研究方向，应用范围从虚拟试穿到多元文化设计原型制作。现有方法往往面临文化偏见、可扩展性有限以及文本提示与生成图像对齐等问题，尤其是在弱监督情况下。在本研究中，我们提出了一种新的框架，该框架结合了大型语言模型（LLMs）与潜在扩散模型（LDMs），以应对这些挑战。我们的方法利用LLMs对文本提示进行语义细化，并引入了一个弱监督筛选模块，有效利用噪声或弱标签数据。通过在增强的DeepFashion+数据集上微调LDM，该数据集包含了全球时尚风格，所提出的方法实现了最先进的性能。实验结果表明，与基线方法相比，我们的方法显著表现出更低的弗雷切ceptions距离（FID）和更高的 inception 分数（IS），同时人类评估验证了其生成多元文化和语义相关时尚内容的能力。这些结果突显了LLM引导的扩散模型在推动可扩展和包容性的人工智能驱动时尚创新方面的潜力。 

---
# ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born from Transformer 

**Title (ZH)**: ARWKV：预训练并非我们所需， một基于Transformer的RNN-Attention基语言模型 

**Authors**: Lin Yueyu, Li Zhiyuan, Peter Yue, Liu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15570)  

**Abstract**: As is known, hybrid quadratic and subquadratic attention models in multi-head architectures have surpassed both Transformer and Linear RNN models , with these works primarily focusing on reducing KV complexity and improving efficiency. For further research on expressiveness, we introduce our series of models distilled from Qwen 2.5, based on pure native RWKV-7 attention, which aims to make RNN more expressive and demonstrates state tracking ability beyond transformers. We work with QRWK 32B based on RWKV-6 architecture, another approach that reduces the entire knowledge processing time to just 8 hours using 16 AMD MI300X GPUs while maintaining Qwen 2.5's performance. In fact, the distillation process can utilize any LLM, not just Qwen, and enables knowledge transfer from larger LLMs to smaller ones with more fewer tokens. We will explain the detailed process and share our insights on building more powerful foundation models. Please note that this is an ongoing work that will be updated continuously. The model checkpoints and source code are available at \href{this https URL}{this https URL}, \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 众所周知，混合二次和次二次注意模型在多头架构中已经超越了Transformer和线性RNN模型，这些研究主要集中在减少KV复杂性和提高效率方面。为进一步提升表达能力，我们基于Qwen 2.5系列模型，引入了一系列以纯原生RWKV-7注意为基础的模型，旨在使RNN更具表达力，并且能够展示出超越Transformer的状态跟踪能力。我们使用基于RWKV-6架构的QRWK 32B模型，在保持Qwen 2.5性能的同时，使用16块AMD MI300X GPU将整个知识处理时间缩短至仅8小时。实际上，蒸馏过程可以利用任何语言模型，而不仅仅是Qwen，从而允许从较大的语言模型向较小的语言模型进行知识迁移，同时使用更少的tokens。我们将详细解释这一过程，并分享构建更强大的基础模型的一些见解。请注意，这是一个正在进行中的工作，将不断更新。模型的检查点和源代码可在以下链接上获取：\href{这个 https URL}{这个 https URL}，\href{这个 https URL}{这个 https URL}。 

---
# Multilevel Browsing of Folksonomy-Based Digital Collections 

**Title (ZH)**: 基于民典分类的数字化藏品多层浏览方法 

**Authors**: Joaquín Gayoso-Cabada, Daniel Rodríguez-Cerezo, José-Luis Sierra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15487)  

**Abstract**: This paper describes how to extend the usual one-level tag selection navigation paradigm in folksonomy-based digital collections to a multilevel browsing one, according to which it is possible to incrementally narrow down the set of selected objects in a collection by sequentially adding more and more filtering tags. For this purpose, we present a browsing strategy based on finite automata. Also, we provide some experimental results concerning the application of the approach in Clavy, a system for managing digital collections with reconfigurable structures in digital humanities and educational settings. 

**Abstract (ZH)**: 本文描述了如何将基于民间分类法的数字集合中的常规一级标签选择导航 paradigm 扩展为多级浏览模式。在这种模式中，可以通过依次添加更多的过滤标签来逐步缩小集合中选定对象的范围。为了实现这一目标，本文提出了基于有限自动机的浏览策略，并提供了一些关于该方法在 Clavy 系统中的应用的实验结果。Clavy 是一种用于人文学科和教育领域中可配置结构的数字集合管理系统的实现。 

---
# Query-based versus resource-based cache strategies in tag-based browsing systems 

**Title (ZH)**: 基于查询的缓存策略与基于资源的缓存策略在基于标签的浏览系统中的比较 

**Authors**: Joaquín Gayoso-Cabada, Mercedes Gómez-Albarrán, José-Luis Sierra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15481)  

**Abstract**: Tag-based browsing is a popular interaction model for navigating digital libraries. According to this model, users select descriptive tags to filter resources in the collections. Typical implementations of the model are based on inverted indexes. However, these implementations can require a considerable amount of set operations to update the browsing state. To palliate this inconven-ience, it is possible to adopt suitable cache strategies. In this paper we describe and compare two of these strategies: (i) a query-based strategy, according to which previously computed browsing states are indexed by sets of selected tags; and (ii) a resource-based strategy, according to which browsing states are in-dexed by sets of filtered resources. Our comparison focused on runtime perfor-mance, and was carried out empirically, using a real-world web-based collec-tion in the field of digital humanities. The results obtained show that the re-source-based strategy clearly outperforms the query-based one. 

**Abstract (ZH)**: 基于主题的浏览是导航数字图书馆的一种流行交互模型。根据这一模型，用户选择描述性的标签来过滤集合中的资源。该模型的典型实施基于倒排索引。然而，这些实施在更新浏览状态时可能需要大量的集合操作。为解决这一不便，可以采用适当的缓存策略。在本文中，我们描述并比较了两种这样的策略：（i）基于查询的策略，根据该策略，之前计算得到的浏览状态被索引为一系列选定的标签集合；（ii）基于资源的策略，根据该策略，浏览状态被索引为一系列过滤后的资源集合。我们的比较集中在运行时性能上，并通过实证方法在数字人文学科领域的实际网络集合上进行了实验研究。实验结果表明，基于资源的策略明显优于基于查询的策略。 

---
# Data-adaptive Safety Rules for Training Reward Models 

**Title (ZH)**: 自适应数据安全规则在训练奖励模型中的应用 

**Authors**: Xiaomin Li, Mingye Gao, Zhiwei Zhang, Jingxuan Fan, Weiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.15453)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is commonly employed to tailor models to human preferences, especially to improve the safety of outputs from large language models (LLMs). Traditionally, this method depends on selecting preferred responses from pairs. However, due to the variability in human opinions and the challenges in directly comparing two responses, there is an increasing trend towards fine-grained annotation approaches that evaluate responses using multiple targeted metrics or rules. The challenge lies in efficiently choosing and applying these rules to handle the diverse range of preference data. In this paper, we propose a dynamic method that adaptively selects the most important rules for each response pair. We introduce a mathematical framework that utilizes the maximum discrepancy across paired responses and demonstrate theoretically that this approach maximizes the mutual information between the rule-based annotations and the underlying true preferences. We then train an 8B reward model using this adaptively labeled preference dataset and assess its efficacy using RewardBench. As of January 25, 2025, our model achieved the highest safety performance on the leaderboard, surpassing various larger models. 

**Abstract (ZH)**: 人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）广泛应用于定制模型以契合人类偏好，特别是提高大型语言模型（LLMs）输出的安全性。传统上，该方法依赖于从成对响应中选择优选响应。然而，由于人类意见的多样性和直接对比两个响应的难度，使用多目标指标或规则进行细粒度标注的方法正逐渐流行起来。面临的挑战是如何高效地选择并应用这些规则来处理各种偏好数据。在本文中，我们提出了一种动态方法，该方法能够自适应地为每个响应对选择最重要的规则。我们引入了一个数学框架，利用成对响应之间的最大差异性，并从理论上证明，这种方法最大程度地提高了基于规则标注与隐藏的真实偏好之间的互信息。然后，我们使用这种自适应标注的偏好数据集训练了一个8B奖励模型，并使用RewardBench进行效果评估。截至2025年1月25日，我们的模型在排行榜上的安全性性能最高，超过了多种大型模型。 

---
# STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection 

**Title (ZH)**: STATE ToxiCN：中文仇恨言论检测中的基于 spans 级别目标感知的毒性提取基准 

**Authors**: Zewen Bai, Yuanyuan Sun, Shengdi Yin, Junyu Lu, Jingjie Zeng, Haohao Zhu, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.15451)  

**Abstract**: The proliferation of hate speech has caused significant harm to society. The intensity and directionality of hate are closely tied to the target and argument it is associated with. However, research on hate speech detection in Chinese has lagged behind, and existing datasets lack span-level fine-grained annotations. Furthermore, the lack of research on Chinese hateful slang poses a significant challenge. In this paper, we provide a solution for fine-grained detection of Chinese hate speech. First, we construct a dataset containing Target-Argument-Hateful-Group quadruples (STATE ToxiCN), which is the first span-level Chinese hate speech dataset. Secondly, we evaluate the span-level hate speech detection performance of existing models using STATE ToxiCN. Finally, we conduct the first study on Chinese hateful slang and evaluate the ability of LLMs to detect such expressions. Our work contributes valuable resources and insights to advance span-level hate speech detection in Chinese 

**Abstract (ZH)**: 仇恨言论的泛滥给社会造成了重大损害。仇恨的强度和方向与其关联的目标和论点密切相关。然而，现有研究成果在中文仇恨言论检测方面的进展相对滞后，现有的数据集缺乏层面级的细粒度标注。此外，中文仇恨绰号研究的缺乏构成了一个重大挑战。在本文中，我们提供了一种针对中文仇恨言论的细粒度检测解决方案。首先，我们构建了一个包含目标-论点-仇恨群体四元组的数据集（STATE ToxiCN），它是首个层面级的中文仇恨言论数据集。其次，我们使用STATE ToxiCN评估现有模型在层面级仇恨言论检测上的性能。最后，我们首次对中文仇恨绰号进行了研究，并评估了语言模型检测此类表达的能力。我们的工作为推进中文层面级仇恨言论检测提供了宝贵的数据资源和见解。 

---
# Token Democracy: The Architectural Limits of Alignment in Transformer-Based Language Models 

**Title (ZH)**: 代币民主：基于变压器的语言模型对齐的架构限制 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2501.15446)  

**Abstract**: Modern language models paradoxically combine unprecedented capability with persistent vulnerability in that they can draft poetry yet cannot reliably refuse harmful requests. We reveal this fragility stems not from inadequate training, but from a fundamental architectural limitation: transformers process all tokens as equals. Transformers operate as computational democracies, granting equal voice to all tokens. This is a design tragically unsuited for AGI, where we cannot risk adversarial "candidates" hijacking the system. Through formal analysis, we demonstrate that safety instructions fundamentally lack privileged status in transformer architectures, that they compete with adversarial inputs in the same computational arena, making robust alignment through prompting or fine-tuning inherently limited. This "token democracy" explains why jailbreaks bypass even extensively safety-trained models and why positional shifts erode prompt effectiveness. Our work systematizes practitioners' tacit knowledge into an architectural critique, showing current alignment approaches create mere preferences, not constraints. 

**Abstract (ZH)**: 现代语言模型在前所未有的能力和持续的脆弱性之间表现出一种悖论。它们能够创作诗歌，但却不能可靠地拒绝有害请求。我们揭示这种脆弱性并非源于训练不足，而是源自根本的架构限制：变换器将所有标记视为平等。变换器作为一种计算民主体制，赋予所有标记平等的话语权。这一设计对AGI来说是灾难性的，因为我们无法在其中冒险让对抗性“候选人”劫持系统。通过形式分析，我们证明了安全指令在变换器架构中并未享有特权地位，它们与其他对抗性输入在相同的计算领域中竞争，使得通过提示或微调实现稳健对齐本质上是有限的。这种“标记民主”解释了为什么突破（jailbreaks）能够绕过甚至最广泛训练的安全模型，以及位置偏移会削弱提示效果。我们的研究将实践者的隐性知识系统化，揭示当前的对齐方法仅创造了偏好而非约束。 

---
# Evaluating Simple Debiasing Techniques in RoBERTa-based Hate Speech Detection Models 

**Title (ZH)**: 基于RoBERTa的仇恨言论检测模型中简单去偏技术的评估 

**Authors**: Diana Iftimie, Erik Zinn  

**Link**: [PDF](https://arxiv.org/pdf/2501.15430)  

**Abstract**: The hate speech detection task is known to suffer from bias against African American English (AAE) dialect text, due to the annotation bias present in the underlying hate speech datasets used to train these models. This leads to a disparity where normal AAE text is more likely to be misclassified as abusive/hateful compared to non-AAE text. Simple debiasing techniques have been developed in the past to counter this sort of disparity, and in this work, we apply and evaluate these techniques in the scope of RoBERTa-based encoders. Experimental results suggest that the success of these techniques depends heavily on the methods used for training dataset construction, but with proper consideration of representation bias, they can reduce the disparity seen among dialect subgroups on the hate speech detection task. 

**Abstract (ZH)**: 仇恨言论检测任务因底层仇恨言论数据集存在的注释偏差而对非洲美式英语（AAE）方言文本表现出偏差，导致正常AAE文本比非AAE文本更有可能被误分类为攻击性或仇恨言论。过去曾开发了一些简单的去偏见技术来应对这种偏差，在本研究中，我们将在基于RoBERTa的编码器范围内应用并评估这些技术。实验结果表明，这些技术的成功很大程度上取决于训练数据集构建所采用的方法，但在考虑表示偏差的情况下，它们可以减少仇恨言论检测任务中方言子组之间的偏差。 

---
# OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas 

**Title (ZH)**: 开放角色：使用大规模合成人设训练可定制的角色扮演大语言模型 

**Authors**: Xiaoyang Wang, Hongming Zhang, Tao Ge, Wenhao Yu, Dian Yu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15427)  

**Abstract**: Customizable role-playing in large language models (LLMs), also known as character generalization, is gaining increasing attention for its versatility and cost-efficiency in developing and deploying role-playing dialogue agents. This study explores a large-scale data synthesis approach to equip LLMs with character generalization capabilities. We begin by synthesizing large-scale character profiles using personas from Persona Hub and then explore two strategies: response rewriting and response generation, to create character-aligned instructional responses. To validate the effectiveness of our synthetic instruction tuning data for character generalization, we perform supervised fine-tuning (SFT) using the LLaMA-3 8B model. Our best-performing model strengthens the original LLaMA-3 8B Instruct model and achieves performance comparable to GPT-4o models on role-playing dialogue. We release our synthetic characters and instruction-tuning dialogues to support public research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的可定制角色扮演，也被称为角色泛化，因其在开发和部署角色对话代理方面的灵活性和成本效益而越来越受到关注。本研究探讨了大规模数据合成方法以增强LLMs的角色泛化能力。我们首先使用Persona Hub中的persona合成大规模角色档案，然后探索了两种策略：响应重写和响应生成，以创建与角色一致的指令性回应。为了验证我们合成的指令调优数据在角色泛化方面的有效性，我们使用LLaMA-3 8B模型进行了监督微调（SFT）。我们性能最佳的模型增强了原始的LLaMA-3 8B指令模型，并在角色对话方面达到了与GPT-4o模型相当的性能。我们将合成的角色和指令调优对话发布出来，以支持公开研究。 

---
# Semantic Layered Embedding Diffusion in Large Language Models for Multi-Contextual Consistency 

**Title (ZH)**: 大型语言模型中基于语义分层嵌入的多重上下文一致性扩散方法 

**Authors**: Irin Kabakum, Thomas Montgomery, Daniel Ravenwood, Genevieve Harrington  

**Link**: [PDF](https://arxiv.org/pdf/2501.15405)  

**Abstract**: The Semantic Layered Embedding Diffusion (SLED) mechanism redefines the representation of hierarchical semantics within transformer-based architectures, enabling enhanced contextual consistency across a wide array of linguistic tasks. By introducing a multi-layered diffusion process grounded in spectral analysis, it achieves a complex balance between global and local semantic coherence. Experimental results demonstrate significant improvements in perplexity and BLEU scores, emphasizing the mechanism's ability to adapt effectively across diverse domains, including multilingual and cross-domain text generation. A rigorous mathematical framework underpins the embedding diffusion process, incorporating weighted adjacency matrices, kernel-based refinements, and dynamic layer-wise normalization. Error distribution analysis reveals that SLED addresses challenges in semantic alignment and coherence, outperforming baseline approaches across varied benchmarks. Scalability studies illustrate that its performance gains are maintained consistently across different model sizes, reflecting a practical balance between computational efficiency and linguistic precision. The implementation also achieves energy efficiency, reducing resource consumption during training and inference phases without compromising accuracy. Qualitative case studies further validate its adaptability to extended narratives and context-intensive scenarios, highlighting the mechanism's potential for real-world applications. SLED offers a different perspective on embedding design and its implications for advancing language modeling. 

**Abstract (ZH)**: Semantic 分层嵌入扩散（SLED）机制重新定义了基于变压器架构中的层次语义表示，从而在广泛的语言任务中增强了上下文一致性。通过在光谱分析的基础上引入多层扩散过程，它实现了全局和局部语义一致性之间的复杂平衡。实验结果表明，在困惑度和 BLEU 分数上取得了显著提升，突显了该机制在多语言和跨领域文本生成等不同领域的适应能力。嵌入扩散过程的基础是一个严格的数学框架，包括加权邻接矩阵、核基精炼和动态分层规范化。误差分布分析揭示了 SLED 在语义对齐和一致性方面优于基线方法，能够在多种基准测试中取得优异表现。可扩展性研究表明，其性能增益在不同模型规模下保持一致，展示了在计算效率和语言精度之间取得的实际平衡。此外，该实现还实现了能效，即使在训练和推理阶段减少资源消耗，也不会影响准确性。定性案例研究进一步验证了其在扩展叙述和上下文密集型场景中的适应能力，突显了该机制在实际应用中的潜力。SLED 为嵌入设计及其对语言建模的推动力提供了一种新的视角。 

---
# How Green are Neural Language Models? Analyzing Energy Consumption in Text Summarization Fine-tuning 

**Title (ZH)**: neural语言模型的绿色环保程度：文本摘要微调中的能源消耗分析 

**Authors**: Tohida Rehman, Debarshi Kumar Sanyal, Samiran Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2501.15398)  

**Abstract**: Artificial intelligence systems significantly impact the environment, particularly in natural language processing (NLP) tasks. These tasks often require extensive computational resources to train deep neural networks, including large-scale language models containing billions of parameters. This study analyzes the trade-offs between energy consumption and performance across three neural language models: two pre-trained models (T5-base and BART-base), and one large language model (LLaMA 3-8B). These models were fine-tuned for the text summarization task, focusing on generating research paper highlights that encapsulate the core themes of each paper. A wide range of evaluation metrics, including ROUGE, METEOR, MoverScore, BERTScore, and SciBERTScore, were employed to assess their performance. Furthermore, the carbon footprint associated with fine-tuning each model was measured, offering a comprehensive assessment of their environmental impact. This research underscores the importance of incorporating environmental considerations into the design and implementation of neural language models and calls for the advancement of energy-efficient AI methodologies. 

**Abstract (ZH)**: 人工智能系统在环境方面产生了显著影响，尤其是在自然语言处理（NLP）任务中。这些任务通常需要大量的计算资源来训练深度神经网络，包括包含数亿参数的大规模语言模型。本研究分析了三个神经语言模型在能耗与性能之间的权衡：两个预训练模型（T5-base和BART-base），以及一个大型语言模型（LLaMA 3-8B）。这些模型经过了文本摘要任务的微调，旨在生成涵盖每篇论文核心主题的研究论文概要。采用了广泛的评估指标，包括ROUGE、METEOR、MoverScore、BERTScore和SciBERTScore，来评估它们的性能。此外，还测量了每个模型微调过程中的碳足迹，为评估它们的环境影响提供了全面的视角。这项研究强调了在设计和实现神经语言模型时纳入环境考量的重要性，并呼吁发展更加节能的AI方法。 

---
# Qwen2.5-1M Technical Report 

**Title (ZH)**: Qwen 2.5-1M 技术报告 

**Authors**: An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang, Jianhong Tu, Jianwei Zhang, Jingren Zhou, Junyang Lin, Kai Dang, Kexin Yang, Le Yu, Mei Li, Minmin Sun, Qin Zhu, Rui Men, Tao He, Weijia Xu, Wenbiao Yin, Wenyuan Yu, Xiafei Qiu, Xingzhang Ren, Xinlong Yang, Yong Li, Zhiying Xu, Zipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15383)  

**Abstract**: We introduce Qwen2.5-1M, a series of models that extend the context length to 1 million tokens. Compared to the previous 128K version, the Qwen2.5-1M series have significantly enhanced long-context capabilities through long-context pre-training and post-training. Key techniques such as long data synthesis, progressive pre-training, and multi-stage supervised fine-tuning are employed to effectively enhance long-context performance while reducing training costs.
To promote the use of long-context models among a broader user base, we present and open-source our inference framework. This framework includes a length extrapolation method that can expand the model context lengths by at least four times, or even more, without additional training. To reduce inference costs, we implement a sparse attention method along with chunked prefill optimization for deployment scenarios and a sparsity refinement method to improve precision. Additionally, we detail our optimizations in the inference engine, including kernel optimization, pipeline parallelism, and scheduling optimization, which significantly enhance overall inference performance. By leveraging our inference framework, the Qwen2.5-1M models achieve a remarkable 3x to 7x prefill speedup in scenarios with 1 million tokens of context. This framework provides an efficient and powerful solution for developing applications that require long-context processing using open-source models.
The Qwen2.5-1M series currently includes the open-source models Qwen2.5-7B-Instruct-1M and Qwen2.5-14B-Instruct-1M, as well as the API-accessed model Qwen2.5-Turbo. Evaluations show that Qwen2.5-1M models have been greatly improved in long-context tasks without compromising performance in short-context scenarios. Specifically, the Qwen2.5-14B-Instruct-1M model significantly outperforms GPT-4o-mini in long-context tasks and supports contexts eight times longer. 

**Abstract (ZH)**: 我们介绍了Qwen2.5-1M系列模型，该系列模型将上下文长度扩展到了100万tokens。与之前的128K版本相比，Qwen2.5-1M系列模型通过长上下文预训练和后训练显著增强了长上下文处理能力。我们采用了长数据合成、分阶段预训练和多阶段监督微调等关键技术，有效提升了长上下文性能，同时降低了训练成本。

为了促进长上下文模型在更广泛用户群体中的应用，我们提出了并开源了我们的推理框架。该框架包含一种长度外推方法，可以在不进行额外训练的情况下至少扩展模型上下文长度四倍，甚至更多。为了减少推理成本，我们实施了稀疏注意方法、分块预填充优化以及稀疏性改进方法，以提高精度。此外，我们详细介绍了推理引擎中的各项优化，包括内核优化、管道并行和调度优化，这些优化措施显著提高了整体推理性能。通过利用此推理框架，Qwen2.5-1M系列模型在包含100万tokens上下文的情况下实现了3到7倍的预填充加速。该框架为使用开源模型开发需要长上下文处理的应用程序提供了高效而强大的解决方案。

目前，Qwen2.5-1M系列包括开源模型Qwen2.5-7B-Instruct-1M、Qwen2.5-14B-Instruct-1M，以及可通过API访问的模型Qwen2.5-Turbo。评估结果显示，Qwen2.5-1M系列模型在长上下文任务中得到了显著改进，同时在短上下文场景中并未牺牲性能。特别是，Qwen2.5-14B-Instruct-1M模型在长上下文任务中显著优于GPT-4o-mini，并且支持八倍于上下文长度。 

---
# Evaluating the Effectiveness of XAI Techniques for Encoder-Based Language Models 

**Title (ZH)**: 评估基于编码器的语言模型的可解释性技术有效性 

**Authors**: Melkamu Abay Mersha, Mesay Gemeda Yigezu, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2501.15374)  

**Abstract**: The black-box nature of large language models (LLMs) necessitates the development of eXplainable AI (XAI) techniques for transparency and trustworthiness. However, evaluating these techniques remains a challenge. This study presents a general evaluation framework using four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. We assess the effectiveness of six explainability techniques from five different XAI categories model simplification (LIME), perturbation-based methods (SHAP), gradient-based approaches (InputXGradient, Grad-CAM), Layer-wise Relevance Propagation (LRP), and attention mechanisms-based explainability methods (Attention Mechanism Visualization, AMV) across five encoder-based language models: TinyBERT, BERTbase, BERTlarge, XLM-R large, and DeBERTa-xlarge, using the IMDB Movie Reviews and Tweet Sentiment Extraction (TSE) datasets. Our findings show that the model simplification-based XAI method (LIME) consistently outperforms across multiple metrics and models, significantly excelling in HA with a score of 0.9685 on DeBERTa-xlarge, robustness, and consistency as the complexity of large language models increases. AMV demonstrates the best Robustness, with scores as low as 0.0020. It also excels in Consistency, achieving near-perfect scores of 0.9999 across all models. Regarding Contrastivity, LRP performs the best, particularly on more complex models, with scores up to 0.9371. 

**Abstract (ZH)**: 大型语言模型（LLMs）的黑盒性质需要开发可解释的人工智能（XAI）技术以提高透明度和可信度。然而，评估这些技术仍然是一项挑战。本研究提出了一种通用的评估框架，包括四个关键指标：人类推理一致度（HA）、鲁棒性、一致性和对比性。我们评估了来自五种不同类型XAI类别的六种解释性技术的有效性，包括模型简化（LIME）、扰动方法（SHAP）、梯度方法（InputXGradient、Grad-CAM）、层级相关性传播（LRP）和基于注意力机制的解释方法（注意力机制可视化，AMV），在五种编码器基语言模型：TinyBERT、BERTbase、BERTlarge、XLM-R large和DeBERTa-xlarge上使用IMDB电影评论和推文情感提取（TSE）数据集进行评估。研究结果显示，基于模型简化（LIME）的XAI方法在多个指标和模型中表现出一致性优异，特别是在DeBERTa-xlarge上的HA得分为0.9685，鲁棒性和一致性方面表现尤其出色，随着大型语言模型复杂性的增加，其优势愈加明显。AMV在鲁棒性方面表现最佳，得分为0.0020，并在一致性方面取得了近乎完美的得分0.9999，适用于所有模型。关于对比性，LRP在更复杂的模型中表现最佳，得分为0.9371。 

---
# Baichuan-Omni-1.5 Technical Report 

**Title (ZH)**: 《Baichuan-Omni-1.5 技术报告》

解释：这里的“Baichuan-Omni-1.5”看起来是一个技术名称或系统名称，因此在翻译时保持了原名，仅将“Technical Report”翻译为“技术报告”，以符合学术文献的规范。 

**Authors**: Yadong Li, Jun Liu, Tao Zhang, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, Chong Li, Yuanbo Fang, Dongdong Kuang, Mingrui Wang, Chenglin Zhu, Youwei Zhang, Hongyu Guo, Fengyu Zhang, Yuran Wang, Bowen Ding, Wei Song, Xu Li, Yuqi Huo, Zheng Liang, Shusen Zhang, Xin Wu, Shuai Zhao, Linchu Xiong, Yozhen Wu, Jiahui Ye, Wenhao Lu, Bowen Li, Yan Zhang, Yaqi Zhou, Xin Chen, Lei Su, Hongda Zhang, Fuzhong Chen, Xuezhen Dong, Na Nie, Zhiying Wu, Bin Xiao, Ting Li, Shunya Dang, Ping Zhang, Yijia Sun, Jincheng Wu, Jinjie Yang, Xionghai Lin, Zhi Ma, Kegeng Wu, Jia li, Aiyuan Yang, Hui Liu, Jianqiang Zhang, Xiaoxi Chen, Guangwei Ai, Wentao Zhang, Yicong Chen, Xiaoqin Huang, Kun Li, Wenjing Luo, Yifei Duan, Lingling Zhu, Ran Xiao, Zhe Su, Jiani Pu, Dian Wang, Xu Jia, Tianyu Zhang, Mengyu Ai, Mang Wang, Yujing Qiao, Lei Zhang, Yanjun Shen, Fan Yang, Miao Zhen, Yijie Zhou, Mingyang Chen, Fei Li, Chenzheng Zhu, Keer Lu, Yaqi Zhao, Hao Liang, Youquan Li, Yanzhao Qin, Linzhuang Sun, Jianhua Xu, Haoze Sun, Mingan Lin, Zenan Zhou, Weipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15368)  

**Abstract**: We introduce Baichuan-Omni-1.5, an omni-modal model that not only has omni-modal understanding capabilities but also provides end-to-end audio generation capabilities. To achieve fluent and high-quality interaction across modalities without compromising the capabilities of any modality, we prioritized optimizing three key aspects. First, we establish a comprehensive data cleaning and synthesis pipeline for multimodal data, obtaining about 500B high-quality data (text, audio, and vision). Second, an audio-tokenizer (Baichuan-Audio-Tokenizer) has been designed to capture both semantic and acoustic information from audio, enabling seamless integration and enhanced compatibility with MLLM. Lastly, we designed a multi-stage training strategy that progressively integrates multimodal alignment and multitask fine-tuning, ensuring effective synergy across all modalities. Baichuan-Omni-1.5 leads contemporary models (including GPT4o-mini and MiniCPM-o 2.6) in terms of comprehensive omni-modal capabilities. Notably, it achieves results comparable to leading models such as Qwen2-VL-72B across various multimodal medical benchmarks. 

**Abstract (ZH)**: 我们将介绍Baichuan-Omni-1.5这一全方位模型，它不仅具备全方位的理解能力，还提供了端到端的语音生成能力。为实现不同模态间的流畅和高质量交互，而又不牺牲任何模态的能力，我们着重优化了三个关键方面。首先，我们建立了一个全面的数据清洗和合成管道，获得了约500亿条高质量数据（包括文本、音频和视觉）。其次，我们设计了一个语音分词器（Baichuan-Audio-Tokenizer），能够捕捉音频中的语义和声音信息，从而实现无缝集成和增强与大规模语言模型（MLLM）的兼容性。最后，我们设计了一种多阶段训练策略，逐步整合多模态对齐和多任务微调，确保各模态之间的有效协同。在全面的多模态能力方面，Baichuan-Omni-1.5超越了包括GPT4o-mini和MiniCPM-o 2.6在内的当前模型。特别是在各种多模态医疗基准测试中，它达到了与Qwen2-VL-72B等领先模型相当的结果。 

---
# Large Language Models as Theory of Mind Aware Generative Agents with Counterfactual Reflection 

**Title (ZH)**: 大型语言模型作为具备反事实反思能力的理论理解生成代理 

**Authors**: Bo Yang, Jiaxian Guo, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15355)  

**Abstract**: Recent studies have increasingly demonstrated that large language models (LLMs) possess significant theory of mind (ToM) capabilities, showing the potential for simulating the tracking of mental states in generative agents. In this study, we propose a novel paradigm called ToM-agent, designed to empower LLMs-based generative agents to simulate ToM in open-domain conversational interactions. ToM-agent disentangles the confidence from mental states, facilitating the emulation of an agent's perception of its counterpart's mental states, such as beliefs, desires, and intentions (BDIs). Using past conversation history and verbal reflections, ToM-Agent can dynamically adjust counterparts' inferred BDIs, along with related confidence levels. We further put forth a counterfactual intervention method that reflects on the gap between the predicted responses of counterparts and their real utterances, thereby enhancing the efficiency of reflection. Leveraging empathetic and persuasion dialogue datasets, we assess the advantages of implementing the ToM-agent with downstream tasks, as well as its performance in both the first-order and the \textit{second-order} ToM. Our findings indicate that the ToM-agent can grasp the underlying reasons for their counterpart's behaviors beyond mere semantic-emotional supporting or decision-making based on common sense, providing new insights for studying large-scale LLMs-based simulation of human social behaviors. 

**Abstract (ZH)**: 近年来，大量研究表明大型语言模型（LLMs）具备显著的理论心智（ToM）能力，显示出在生成代理中模拟追踪心智状态的潜在可能性。在此研究中，我们提出了一种名为ToM-agent的新范式，旨在赋予基于LLMs的生成代理模拟ToM的能力，特别是在开放式领域对话交互中的应用。ToM-agent将信心与心智状态分离，促进代理对其对应方的心智状态（如信念、欲望和意图，BDIs）感知的模拟。利用过去的对话历史和言语反思，ToM-Agent可以动态调整对对应方的推断BDIs及其相关信心水平进行调整。我们还提出了一个假设干预方法，通过反映预测响应与实际言辞之间的差距，从而增强反思效率。借助同理心和说服性对话数据集，我们评估了实施ToM-agent在下游任务中的优势，以及其在一级和二级ToM中的性能表现。我们的研究表明，ToM-agent能够把握对应方行为背后的根本原因，而不仅仅依赖于语义情感支撑或基于常识的决策制定，为研究大规模LLMs基于的心智模型模拟人类社会行为提供了新的视角。 

---
# Figurative-cum-Commonsense Knowledge Infusion for Multimodal Mental Health Meme Classification 

**Title (ZH)**: 具象化与常识知识融合在多模态心理健康 meme 分类中的应用 

**Authors**: Abdullah Mazhar, Zuhair hasan shaik, Aseem Srivastava, Polly Ruhnke, Lavanya Vaddavalli, Sri Keshav Katragadda, Shweta Yadav, Md Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2501.15321)  

**Abstract**: The expression of mental health symptoms through non-traditional means, such as memes, has gained remarkable attention over the past few years, with users often highlighting their mental health struggles through figurative intricacies within memes. While humans rely on commonsense knowledge to interpret these complex expressions, current Multimodal Language Models (MLMs) struggle to capture these figurative aspects inherent in memes. To address this gap, we introduce a novel dataset, AxiOM, derived from the GAD anxiety questionnaire, which categorizes memes into six fine-grained anxiety symptoms. Next, we propose a commonsense and domain-enriched framework, M3H, to enhance MLMs' ability to interpret figurative language and commonsense knowledge. The overarching goal remains to first understand and then classify the mental health symptoms expressed in memes. We benchmark M3H against 6 competitive baselines (with 20 variations), demonstrating improvements in both quantitative and qualitative metrics, including a detailed human evaluation. We observe a clear improvement of 4.20% and 4.66% on weighted-F1 metric. To assess the generalizability, we perform extensive experiments on a public dataset, RESTORE, for depressive symptom identification, presenting an extensive ablation study that highlights the contribution of each module in both datasets. Our findings reveal limitations in existing models and the advantage of employing commonsense to enhance figurative understanding. 

**Abstract (ZH)**: 近年来，通过非传统方式（如表情包）表达心理健康症状引起了广泛关注，用户常通过表情包中的象征性细节来突出他们的心理健康困境。尽管人类依赖常识来解释这些复杂的表达，当前的多模态语言模型（MLMs）难以捕捉到表情包中固有的象征性方面。为了解决这一问题，我们引入了一个新的数据集AxiOM，该数据集源自GAD焦虑问卷，将表情包分类为六种细粒度的焦虑症状。随后，我们提出了一种结合常识和领域增强框架M3H，以提升MLMs理解和解释象征性语言及常识的能力。我们的总体目标是首先理解和然后分类在表情包中表达的心理健康症状。我们用M3H与6种竞争基线（包括20种变体）进行了基准测试，结果在定量和定性指标上均有所提高，包括详细的_human评估。我们在_weighted-F1指标上观察到了明显改进，分别为4.20%和4.66%。为评估模型的普适性，我们在一个公开数据集RESTORE上进行了广泛的实验，用于抑郁症症状识别，并进行了一项详尽的消融研究，突显了各模块在两个数据集中的贡献。我们的研究发现现有模型的局限性，并展示了运用常识增强象征性理解的优势。 

---
# The Multicultural Medical Assistant: Can LLMs Improve Medical ASR Errors Across Borders? 

**Title (ZH)**: 多元文化医疗助手：大型语言模型能跨越国界改进医学语音识别错误吗？ 

**Authors**: Ayo Adedeji, Mardhiyah Sanni, Emmanuel Ayodele, Sarita Joshi, Tobi Olatunji  

**Link**: [PDF](https://arxiv.org/pdf/2501.15310)  

**Abstract**: The global adoption of Large Language Models (LLMs) in healthcare shows promise to enhance clinical workflows and improve patient outcomes. However, Automatic Speech Recognition (ASR) errors in critical medical terms remain a significant challenge. These errors can compromise patient care and safety if not detected. This study investigates the prevalence and impact of ASR errors in medical transcription in Nigeria, the United Kingdom, and the United States. By evaluating raw and LLM-corrected transcriptions of accented English in these regions, we assess the potential and limitations of LLMs to address challenges related to accents and medical terminology in ASR. Our findings highlight significant disparities in ASR accuracy across regions and identify specific conditions under which LLM corrections are most effective. 

**Abstract (ZH)**: 全球范围内大型语言模型（LLMs）在医疗领域的应用显示出了增强临床工作流程和改善患者结果的潜力。然而，医疗术语识别错误仍然是Automatic Speech Recognition（ASR）面临的重要挑战。这些错误如果不被检测到，可能会威胁到患者的安全和护理。本研究调查了尼日利亚、英国和美国地区医疗转录中ASR错误的普遍性和影响。通过评估这些区域带口音的英语录音及其经过LLM修正后的版本，我们评估了LLM在解决口音和医疗术语相关ASR挑战方面的潜力和限制。研究发现突显了各地区ASR准确性显著差异，并指出了LLM修正效果最佳的具体条件。 

---
# You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning 

**Title (ZH)**: 一次剪枝足够：基于策略学习的无校准模型压缩设计 

**Authors**: Ayan Sengupta, Siddhant Chaudhary, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2501.15296)  

**Abstract**: The ever-increasing size of large language models (LLMs) presents significant challenges for deployment due to their heavy computational and memory requirements. Current model pruning techniques attempt to alleviate these issues by relying heavily on external calibration datasets to determine which parameters to prune or compress, thus limiting their flexibility and scalability across different compression ratios. Moreover, these methods often cause severe performance degradation, particularly in downstream tasks, when subjected to higher compression rates. In this paper, we propose PruneNet, a novel model compression method that addresses these limitations by reformulating model pruning as a policy learning process. PruneNet decouples the pruning process from the model architecture, eliminating the need for calibration datasets. It learns a stochastic pruning policy to assess parameter importance solely based on intrinsic model properties while preserving the spectral structure to minimize information loss. PruneNet can compress the LLaMA-2-7B model in just 15 minutes, achieving over 80% retention of its zero-shot performance with a 30% compression ratio, outperforming existing methods that retain only 75% performance. Furthermore, on complex multitask language understanding tasks, PruneNet demonstrates its robustness by preserving up to 80% performance of the original model, proving itself a superior alternative to conventional structured compression techniques. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）规模的不断增大，其部署面临着重大挑战，这主要是由于它们对计算能力和内存资源的高需求。当前的模型剪枝技术希望通过依赖外部校准数据集来确定需要剪枝或压缩的参数，从而限制了其在不同压缩比下的灵活性和可扩展性。此外，这些方法在较高压缩率下往往会引起严重的性能下降，尤其是在下游任务中。在这篇论文中，我们提出了一种名为PruneNet的新型模型压缩方法，通过将模型剪枝重新定义为一种策略学习过程来解决这些限制。PruneNet将剪枝过程与模型架构分离，消除了对校准数据集的依赖。它学习一个基于模型内在属性的随机剪枝策略来评估参数的重要性，同时保留谱结构以最小化信息损失。PruneNet仅用15分钟就能压缩LLaMA-2-7B模型，压缩比为30%时保留了超过80%的零样本性能，优于现有方法仅保留75%性能的效果。此外，在复杂的多任务语言理解任务中，PruneNet展示了其鲁棒性，保留了原始模型高达80%的性能，证明了它在传统结构化压缩技术中的优越性。 

---
# Are Human Interactions Replicable by Generative Agents? A Case Study on Pronoun Usage in Hierarchical Interactions 

**Title (ZH)**: 生成代理能否重现人类互动？层级互动中代词使用案例研究 

**Authors**: Naihao Deng, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2501.15283)  

**Abstract**: As Large Language Models (LLMs) advance in their capabilities, researchers have increasingly employed them for social simulation. In this paper, we investigate whether interactions among LLM agents resemble those of humans. Specifically, we focus on the pronoun usage difference between leaders and non-leaders, examining whether the simulation would lead to human-like pronoun usage patterns during the LLMs' interactions. Our evaluation reveals the significant discrepancies between LLM-based simulations and human pronoun usage, with prompt-based or specialized agents failing to demonstrate human-like pronoun usage patterns. In addition, we reveal that even if LLMs understand the human pronoun usage patterns, they fail to demonstrate them in the actual interaction process. Our study highlights the limitations of social simulations based on LLM agents, urging caution in using such social simulation in practitioners' decision-making process. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）能力的不断提升，研究人员越来越多地利用它们进行社会仿真。在本文中，我们探讨LLM代理之间的互动是否类似于人类的互动。具体而言，我们关注领导者和非领导者在代词使用上的差异，研究社会仿真是否会促使LLMs在互动过程中表现出类似人类的代词使用模式。我们的评估揭示了基于LLM的社会仿真的代词使用模式与人类代词使用模式之间存在显著差异，以提示为基础或专门化的代理未能表现出类似人类的代词使用模式。此外，我们还发现即使LLM理解了人类的代词使用模式，它们在实际互动过程中也无法表现出这种模式。本研究强调了基于LLM代理的社会仿真的局限性，提醒在实际应用中应谨慎使用此类社会仿真辅助决策过程。 

---
# Pre-training a Transformer-Based Generative Model Using a Small Sepedi Dataset 

**Title (ZH)**: 使用小型塞遂语数据集预训练基于变压器的生成模型 

**Authors**: Simon P. Ramalepe, Thipe I. Modipa, Marelie H. Davel  

**Link**: [PDF](https://arxiv.org/pdf/2501.15281)  

**Abstract**: Due to the scarcity of data in low-resourced languages, the development of language models for these languages has been very slow. Currently, pre-trained language models have gained popularity in natural language processing, especially, in developing domain-specific models for low-resourced languages. In this study, we experiment with the impact of using occlusion-based techniques when training a language model for a text generation task. We curate 2 new datasets, the Sepedi monolingual (SepMono) dataset from several South African resources and the Sepedi radio news (SepNews) dataset from the radio news domain. We use the SepMono dataset to pre-train transformer-based models using the occlusion and non-occlusion pre-training techniques and compare performance. The SepNews dataset is specifically used for fine-tuning. Our results show that the non-occlusion models perform better compared to the occlusion-based models when measuring validation loss and perplexity. However, analysis of the generated text using the BLEU score metric, which measures the quality of the generated text, shows a slightly higher BLEU score for the occlusion-based models compared to the non-occlusion models. 

**Abstract (ZH)**: 由于低资源语言数据的稀缺性，这些语言的语言模型的发展非常缓慢。目前，预训练语言模型在自然语言处理中广受欢迎，特别是在为低资源语言开发领域特定模型方面。本研究通过实验探讨在训练文本生成任务的语言模型时使用遮蔽技术的影响。我们编纂了两个新的数据集：来自南非资源的拼多利语单语（SepMono）数据集以及来自电台新闻领域的拼多利语电台新闻（SepNews）数据集。我们利用SepMono数据集分别使用遮蔽和非遮蔽预训练技术预训练基于Transformer的模型，并进行性能比较。SepNews数据集专门用于微调。我们的实验结果显示，在验证损失和困惑度等指标上，非遮蔽模型的表现优于遮蔽模型。然而，使用BLEU评分度量生成文本质量时，遮蔽模型的BLEU评分略高于非遮蔽模型。 

---
# New Evaluation Paradigm for Lexical Simplification 

**Title (ZH)**: 词简化的新型评价范式 

**Authors**: Jipeng Qiang, Minjiang Huang, Yi Zhu, Yunhao Yuan, Chaowei Zhang, Xiaoye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15268)  

**Abstract**: Lexical Simplification (LS) methods use a three-step pipeline: complex word identification, substitute generation, and substitute ranking, each with separate evaluation datasets. We found large language models (LLMs) can simplify sentences directly with a single prompt, bypassing the traditional pipeline. However, existing LS datasets are not suitable for evaluating these LLM-generated simplified sentences, as they focus on providing substitutes for single complex words without identifying all complex words in a sentence.
To address this gap, we propose a new annotation method for constructing an all-in-one LS dataset through human-machine collaboration. Automated methods generate a pool of potential substitutes, which human annotators then assess, suggesting additional alternatives as needed. Additionally, we explore LLM-based methods with single prompts, in-context learning, and chain-of-thought techniques. We introduce a multi-LLMs collaboration approach to simulate each step of the LS task. Experimental results demonstrate that LS based on multi-LLMs approaches significantly outperforms existing baselines. 

**Abstract (ZH)**: 词汇简化（LS）方法采用三步流水线：复杂词识别、替代词生成和替代词排序，并分别使用独立的评估数据集。我们发现大规模语言模型（LLMs）可以直接通过一个单一的提示来简化句子，绕过了传统的流水线流程。然而，现有的LS数据集不适用于评估这些由LLM生成的简化句子，因为它们侧重于为单个复杂词提供替代词而没有在句子中识别所有复杂词。

为了解决这一问题，我们提出了一种新的注释方法，通过人机合作构建一个包含所有步骤的LS数据集。自动方法生成一系列潜在的替代词，然后由人工注释者进行评估，必要时提出额外的选择。此外，我们还探讨了基于单个提示的大规模语言模型方法、上下文学习和思考链技术。我们提出了多大规模语言模型协作的方法来模拟LS任务中的每个步骤。实验结果显示，基于多大规模语言模型的方法显著优于现有基线方法。 

---
# Breaking the Stigma! Unobtrusively Probe Symptoms in Depression Disorder Diagnosis Dialogue 

**Title (ZH)**: 打破偏见！非侵扰性探究抑郁障碍诊断对话中的症状 

**Authors**: Jieming Cao, Chen Huang, Yanan Zhang, Ruibo Deng, Jincheng Zhang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2501.15260)  

**Abstract**: Stigma has emerged as one of the major obstacles to effectively diagnosing depression, as it prevents users from open conversations about their struggles. This requires advanced questioning skills to carefully probe the presence of specific symptoms in an unobtrusive manner. While recent efforts have been made on depression-diagnosis-oriented dialogue systems, they largely ignore this problem, ultimately hampering their practical utility. To this end, we propose a novel and effective method, UPSD$^{4}$, developing a series of strategies to promote a sense of unobtrusiveness within the dialogue system and assessing depression disorder by probing symptoms. We experimentally show that UPSD$^{4}$ demonstrates a significant improvement over current baselines, including unobtrusiveness evaluation of dialogue content and diagnostic accuracy. We believe our work contributes to developing more accessible and user-friendly tools for addressing the widespread need for depression diagnosis. 

**Abstract (ZH)**: 刻板印象已成为有效诊断抑郁的主要障碍之一，因为它阻碍了用户就自己面临的困境进行开放讨论。为了克服这一障碍，需要使用高超的提问技巧，在不引起注意的情况下仔细探查特定症状的存在。尽管近期在抑郁诊断导向的对话系统方面做出了不少努力，但这些系统大多忽视了这个问题，最终限制了它们的实际应用价值。为此，我们提出了一种新颖且有效的方法——UPSD$^4$，开发了一系列策略，在对话系统中促进不引人注意的感觉，并通过探查症状来评估抑郁障碍。实验表明，UPSD$^4$在对话内容的不引人注意性评估和诊断准确性方面均显著优于当前基准方法。我们认为，我们的工作有助于开发更多易于获取和用户友好的工具，以应对广泛存在的抑郁诊断需求。 

---
# Prompting ChatGPT for Chinese Learning as L2: A CEFR and EBCL Level Study 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

"使用提示引导ChatGPT进行汉语作为二外的学习：基于CEFR和EBCL的水平研究"

解释：
- "Prompting ChatGPT for Chinese Learning as L2" 翻译为“使用提示引导ChatGPT进行汉语作为二外的学习”。
- "A CEFR and EBCL Level Study" 翻译为“基于CEFR和EBCL的水平研究”。

CEFR代表Common European Framework of Reference for Languages（共同欧洲语言参考框架），EBCL代表European Business Language Test（欧洲商务语言测试），这些都是国际上用来评估语言水平的标准。 

**Authors**: Miao Lin-Zucker, Joël Bellasen, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2501.15247)  

**Abstract**: The use of chatbots in language learning has evolved significantly since the 1960s, becoming more sophisticated platforms as generative AI emerged. These tools now simulate natural conversations, adapting to individual learners' needs, including those studying Chinese. Our study explores how learners can use specific prompts to engage Large Language Models (LLM) as personalized chatbots, aiming to target their language level based on the Common European Framework of Reference for Languages (CEFR) and the European Benchmarking Chinese Language (EBCL) project. Focusing on A1, A1+ and A2 levels, we examine the teaching of Chinese, which presents unique challenges due to its logographic writing system. Our goal is to develop prompts that integrate oral and written skills, using high-frequency character lists and controlling oral lexical productions. These tools, powered by generative AI, aim to enhance language practice by crossing lexical and sinographic recurrence. While generative AI shows potential as a personalized tutor, further evaluation is needed to assess its effectiveness. We conducted a systematic series of experiments using ChatGPT models to evaluate their adherence to constraints specified in the prompts. The results indicate that incorporating level A1 and A1+ characters, along with the associated reference list, significantly enhances compliance with the EBCL character set. Properly prompted, LLMs can increase exposure to the target language and offer interactive exchanges to develop language skills. 

**Abstract (ZH)**: 自20世纪60年代以来，聊天机器人的使用在语言学习中取得了显著进展，随着生成式AI的出现，这些工具变得更为复杂和完善。这些工具现在可以模拟自然对话，能够根据个人学习者的需求进行调整，包括那些正在学习汉语的人。本研究探讨了学习者如何使用特定提示来与大型语言模型（LLM）进行个性化对话，旨在根据《共同欧洲框架reference for Languages（CEFR）》和《欧洲汉语水平评价项目（EBCL）》的要求将语言水平定位。我们专注于A1、A1+和A2级别，研究汉语的教学，而汉语因其表意文字系统存在独特的挑战。我们的目标是开发能够融合口语和书面技能的提示，利用高频汉字列表并控制口语词汇产出。这些工具依靠生成式AI，旨在通过词汇和汉字的重复来增强语言练习。虽然生成式AI作为个性化辅导显示出潜力，但还需进一步评估其有效性。我们使用ChatGPT模型进行了系统性的实验，以评估其在提示规定约束条件下的遵守情况。结果表明，结合A1和A1+级别的汉字及其相应的参考列表，显著增强了对EBCL汉字集的遵守程度。适当提示，大型语言模型可以增加目标语言的接触机会，并提供互动交流以发展语言技能。 

---
# ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval 

**Title (ZH)**: ASRank：基于答案香气的零样本重排序方法在文档检索中的应用 

**Authors**: Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2501.15245)  

**Abstract**: Retrieval-Augmented Generation (RAG) models have drawn considerable attention in modern open-domain question answering. The effectiveness of RAG depends on the quality of the top retrieved documents. However, conventional retrieval methods sometimes fail to rank the most relevant documents at the top. In this paper, we introduce ASRank, a new re-ranking method based on scoring retrieved documents using zero-shot answer scent which relies on a pre-trained large language model to compute the likelihood of the document-derived answers aligning with the answer scent. Our approach demonstrates marked improvements across several datasets, including NQ, TriviaQA, WebQA, ArchivalQA, HotpotQA, and Entity Questions. Notably, ASRank increases Top-1 retrieval accuracy on NQ from $19.2\%$ to $46.5\%$ for MSS and $22.1\%$ to $47.3\%$ for BM25. It also shows strong retrieval performance on several datasets compared to state-of-the-art methods (47.3 Top-1 by ASRank vs 35.4 by UPR by BM25). 

**Abstract (ZH)**: 以下是翻译的内容，符合学术规范：

检索增强生成（RAG）模型在现代开放域问答中引起了广泛关注。RAG的有效性取决于检索出的顶级文档质量。然而，传统的检索方法有时无法将最相关文档排在最前。本文介绍了ASRank，这是一种新的重排序方法，它通过使用预训练的大语言模型计算文档衍生答案与问题答案线索一致性的可能性来进行评分，从而对检索出的文档进行重新排序。我们的方法在多个数据集（包括NQ、TriviaQA、WebQA、ArchivalQA、HotpotQA和Entity Questions）上均表现出显著的改进。值得注意的是，ASRank将MSS下NQ数据集的Top-1检索准确率从19.2%提高到46.5%，将BM25下的Top-1检索准确率从22.1%提高到47.3%。此外，ASRank在多个数据集上的检索性能明显优于最先进的方法（ASRank的47.3 Top-1 vs BM25的UPR方法的35.4）。 

---
# Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning 

**Title (ZH)**: 通过多智能体强化学习提升检索增强生成 

**Authors**: Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15228)  

**Abstract**: Retrieval-augmented generation (RAG) is extensively utilized to incorporate external, current knowledge into large language models, thereby minimizing hallucinations. A standard RAG pipeline may comprise several components, such as query rewriting, document retrieval, document filtering, and answer generation. However, these components are typically optimized separately through supervised fine-tuning, which can lead to misalignments between the objectives of individual modules and the overarching aim of generating accurate answers in question-answering (QA) tasks. Although recent efforts have explored reinforcement learning (RL) to optimize specific RAG components, these approaches often focus on overly simplistic pipelines with only two components or do not adequately address the complex interdependencies and collaborative interactions among the modules. To overcome these challenges, we propose treating the RAG pipeline as a multi-agent cooperative task, with each component regarded as an RL agent. Specifically, we present MMOA-RAG, a Multi-Module joint Optimization Algorithm for RAG, which employs multi-agent reinforcement learning to harmonize all agents' goals towards a unified reward, such as the F1 score of the final answer. Experiments conducted on various QA datasets demonstrate that MMOA-RAG improves the overall pipeline performance and outperforms existing baselines. Furthermore, comprehensive ablation studies validate the contributions of individual components and the adaptability of MMOA-RAG across different RAG components and datasets. The code of MMOA-RAG is on this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）被广泛应用于将外部和最新的知识整合到大型语言模型中，从而减少幻觉的产生。标准的RAG管道可能包括多个组件，如查询重写、文档检索、文档过滤和答案生成。然而，这些组件通常通过有监督微调分别优化，这可能导致各个模块的目标与生成准确答案的整体目标之间的对齐问题。尽管最近的研究尝试使用强化学习（RL）来优化特定的RAG组件，但这些方法往往集中在过于简单的管道设计上，只包含两个组件，或者未能充分解决模块之间复杂相互依赖性和协作交互的问题。为了克服这些挑战，我们提出将RAG管道视为多智能体协作任务，并将每个组件视为一个RL智能体。具体而言，我们提出了MMOA-RAG，这是一种用于RAG的多模块联合优化算法，利用多智能体强化学习将所有智能体的目标协调至统一的奖励，例如最终答案的F1分数。在多个问答数据集上的实验表明，MMOA-RAG提高了整个管道的性能，并优于现有基线。进一步的消融研究验证了各组件的贡献以及MMOA-RAG在不同RAG组件和数据集上的适应性。MMOA-RAG的代码请参见此链接：[此处的链接]。 

---
# SEAL: Scaling to Emphasize Attention for Long-Context Retrieval 

**Title (ZH)**: SEAL: 扩展以强调注意力进行长上下文检索 

**Authors**: Changhun Lee, Jun-gyu Jin, Younghyun Cho, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2501.15225)  

**Abstract**: In this work, we introduce a novel approach called Scaling to Emphasize Attention for Long-context retrieval (SEAL), which enhances the retrieval performance of large language models (LLMs) over extended contexts. Previous studies have shown that each attention head in LLMs has a unique functionality and collectively contributes to the overall behavior of the model. Similarly, we observe that specific heads are closely tied to long-context retrieval, showing positive or negative correlation with retrieval scores. Built on this insight, we propose a learning-based mechanism using zero-shot generated data to emphasize these heads, improving the model's performance in long-context retrieval tasks. By applying SEAL, we can achieve significant improvements in in-domain retrieval performance, including document QA tasks from LongBench, and considerable improvements in out-of-domain cases. Additionally, when combined with existing training-free context extension techniques, SEAL extends the context limits of LLMs while maintaining highly reliable outputs, opening new avenues for research in this field. 

**Abstract (ZH)**: 在本研究中，我们提出了一种新颖的方法，称为扩展注意力以增强长上下文检索（SEAL），该方法能够提升大型语言模型（LLMs）在长上下文检索中的检索性能。先前的研究表明，LLMs 中的每个注意力头具有独特的功能，并且共同作用以影响模型的整体行为。类似地，我们观察到特定的注意力头与长上下文检索紧密相关，它们与检索分数呈正相关或负相关。基于这一洞察，我们提出了一种基于零样本生成数据的学习机制，以强调这些注意力头，从而提高模型在长上下文检索任务中的性能。通过应用 SEAL 方法，我们可以在领域内检索性能（包括来自 LongBench 的文档 QA 任务）和领域外情况下实现显著改进。此外，当与现有的无需训练的上下文扩展技术结合使用时，SEAL 可以在保持高度可靠输出的同时，扩展 LLMs 的上下文限制，从而为该领域的新研究开辟新的途径。 

---
# Faster Machine Translation Ensembling with Reinforcement Learning and Competitive Correction 

**Title (ZH)**: 使用强化学习和竞争性校正的更快机器翻译集成 

**Authors**: Kritarth Prasad, Mohammadi Zaki, Pratik Singh, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2501.15219)  

**Abstract**: Ensembling neural machine translation (NMT) models to produce higher-quality translations than the $L$ individual models has been extensively studied. Recent methods typically employ a candidate selection block (CSB) and an encoder-decoder fusion block (FB), requiring inference across \textit{all} candidate models, leading to significant computational overhead, generally $\Omega(L)$. This paper introduces \textbf{SmartGen}, a reinforcement learning (RL)-based strategy that improves the CSB by selecting a small, fixed number of candidates and identifying optimal groups to pass to the fusion block for each input sentence. Furthermore, previously, the CSB and FB were trained independently, leading to suboptimal NMT performance. Our DQN-based \textbf{SmartGen} addresses this by using feedback from the FB block as a reward during training. We also resolve a key issue in earlier methods, where candidates were passed to the FB without modification, by introducing a Competitive Correction Block (CCB). Finally, we validate our approach with extensive experiments on English-Hindi translation tasks in both directions. 

**Abstract (ZH)**: 使用神经机器翻译（NMT）模型集成以产生比单一的$L$个模型更高的质量翻译已经受到了广泛研究。近年来的方法通常采用候选选择块（CSB）和编码器解码器融合块（FB），需要对所有候选模型进行推理，这导致了显著的计算开销，通常为$\Omega(L)$。本文提出了一种基于强化学习（RL）的策略——\textbf{SmartGen}，该策略通过选择少量固定的候选模型并在每个输入句子中识别最优的候选组供融合块使用，来改进CSB。此外，以往方法中的CSB和FB是独立训练的，这导致了NMT性能的次优结果。我们的基于DQN的\textbf{SmartGen}通过在训练过程中使用FB块的反馈作为奖励来解决这一问题。我们还通过引入竞争校正块（CCB），解决了早期方法中候选模型未经修改直接传递给FB块的关键问题。最后，我们在双向的英语-印地语翻译任务中通过广泛的实验验证了该方法的有效性。 

---
# Who is the root in a syntactic dependency structure? 

**Title (ZH)**: 在句法依赖结构中，根节点是谁？ 

**Authors**: Ramon Ferrer-i-Cancho, Marta Arias  

**Link**: [PDF](https://arxiv.org/pdf/2501.15188)  

**Abstract**: The syntactic structure of a sentence can be described as a tree that indicates the syntactic relationships between words. In spite of significant progress in unsupervised methods that retrieve the syntactic structure of sentences, guessing the right direction of edges is still a challenge. As in a syntactic dependency structure edges are oriented away from the root, the challenge of guessing the right direction can be reduced to finding an undirected tree and the root. The limited performance of current unsupervised methods demonstrates the lack of a proper understanding of what a root vertex is from first principles. We consider an ensemble of centrality scores, some that only take into account the free tree (non-spatial scores) and others that take into account the position of vertices (spatial scores). We test the hypothesis that the root vertex is an important or central vertex of the syntactic dependency structure. We confirm that hypothesis and find that the best performance in guessing the root is achieved by novel scores that only take into account the position of a vertex and that of its neighbours. We provide theoretical and empirical foundations towards a universal notion of rootness from a network science perspective. 

**Abstract (ZH)**: 一个句子的句法结构可以被描述为一棵树，用来表示词与词之间的句法关系。尽管在无监督方法检索句子句法结构方面取得了显著进展，但猜测边的正确方向仍然是一项挑战。由于在句法依赖结构中边是从根指向其他节点的，因此猜测正确方向的问题可以简化为寻找无向树及其根节点。当前无监督方法的有限性能表明，在基本原理层面缺乏对根节点的正确理解。我们考虑了一个由中心性得分组成的集合，其中有些得分仅考虑自由树（非空间得分），而另一些则考虑节点的位置（空间得分）。我们测试了根节点是句法依赖结构中重要或中心节点的假设，并证实了这一假设。我们发现，最佳表现是由仅考虑节点及其邻居位置的新得分实现的。我们从网络科学的角度为根性的普遍概念提供了理论和实证基础。 

---
# Option-ID Based Elimination For Multiple Choice Questions 

**Title (ZH)**: 基于Option-ID的多项选择题消除方法 

**Authors**: Zhenhao Zhu, Bulou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15175)  

**Abstract**: Multiple choice questions (MCQs) are a common and important task for evaluating large language models (LLMs). Based on common strategies humans use when answering MCQs, the process of elimination has been proposed as an effective problem-solving method. Existing methods to the process of elimination generally fall into two categories: one involves having the model directly select the incorrect answer, while the other involves scoring the options. However, both methods incur high computational costs and often perform worse than methods that answer based on option ID. To address this issue, this paper proposes a process of elimination based on option ID. We select 10 LLMs and conduct zero-shot experiments on 7 different datasets. The experimental results demonstrate that our method significantly improves the model's performance. Further analysis reveals that the sequential elimination strategy can effectively enhance the model's reasoning ability. Additionally, we find that sequential elimination is also applicable to few-shot settings and can be combined with debias methods to further improve model performance. 

**Abstract (ZH)**: 多项选择题（MCQs）是评估大型语言模型（LLMs）的一种常见且重要的任务。基于人类在回答MCQs时通常使用的方法，逐项排除策略被提出作为一种有效的问题解决方法。现有的逐项排除方法大体可分为两类：一类是让模型直接选择错误答案，另一类是评分选项。然而，这两种方法都带来了较高的计算成本，并且通常不如基于选项ID作答的方法表现更好。为了解决这一问题，本文提出了一种基于选项ID的逐项排除策略。我们选择了10个LLM，并在7个不同的数据集上进行了零样本实验。实验结果表明，我们的方法显著提高了模型的性能。进一步分析表明，序列排除策略可以有效增强模型的推理能力。此外，我们发现序列排除策略也适用于少样本设置，并且可以与去偏见方法相结合，进一步提高模型性能。 

---
# Task-KV: Task-aware KV Cache Optimization via Semantic Differentiation of Attention Heads 

**Title (ZH)**: Task-KV: 基于注意力头语义差异的面向任务的KV缓存优化 

**Authors**: Xingyang He, Jie Liu, Shaowei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15113)  

**Abstract**: KV cache is a widely used acceleration technique for large language models (LLMs) inference. However, its memory requirement grows rapidly with input length. Previous studies have reduced the size of KV cache by either removing the same number of unimportant tokens for all attention heads or by allocating differentiated KV cache budgets for pre-identified attention heads. However, due to the importance of attention heads varies across different tasks, the pre-identified attention heads fail to adapt effectively to various downstream tasks. To address this issue, we propose Task-KV, a method that leverages the semantic differentiation of attention heads to allocate differentiated KV cache budgets across various tasks. We demonstrate that attention heads far from the semantic center (called heterogeneous heads) make an significant contribution to task outputs and semantic understanding. In contrast, other attention heads play the role of aggregating important information and focusing reasoning. Task-KV allocates full KV cache budget to heterogeneous heads to preserve comprehensive semantic information, while reserving a small number of recent tokens and attention sinks for non-heterogeneous heads. Furthermore, we innovatively introduce middle activations to preserve key contextual information aggregated from non-heterogeneous heads. To dynamically perceive semantic differences among attention heads, we design a semantic separator to distinguish heterogeneous heads from non-heterogeneous ones based on their distances from the semantic center. Experimental results on multiple benchmarks and different model architectures demonstrate that Task-KV significantly outperforms existing baseline methods. 

**Abstract (ZH)**: KV 缓存是一种广泛应用于大型语言模型（LLMs）推理加速的技术。然而，其内存需求会随着输入长度的增加而迅速增长。以往的研究通过两种方式减少了 KV 缓存的大小：一种是为所有注意力头去除相同数量的不重要标记；另一种是为先验识别的注意力头分配不同的 KV 缓存预算。然而，由于不同任务中注意力头的重要性不同，先验识别的注意力头难以有效适应各种下游任务。为解决这一问题，我们提出了一种名为 Task-KV 的方法，该方法利用注意力头在语义上的差异来在不同任务中分配不同的 KV 缓存预算。我们证明，远离语义中心（称为异质性注意力头）的注意力头对任务输出和语义理解有显著贡献。相比之下，其他注意力头则起着聚集重要信息和聚焦推理的作用。Task-KV 将完整的 KV 缓存预算分配给异质性注意力头，以保留全面的语义信息，而仅为非异质性注意力头保留少量的最近标记和注意力汇聚点。此外，我们创新性地引入了中间激活来保留非异质性注意力头聚集的关键上下文信息。为了动态感知注意力头之间的语义差异，我们设计了一种语义分离器，基于它们与语义中心的距离来区分异质性和非异质性注意力头。在多种基准测试和不同模型架构上的实验结果表明，Task-KV 显著优于现有的基线方法。 

---
# Knowledge Hierarchy Guided Biological-Medical Dataset Distillation for Domain LLM Training 

**Title (ZH)**: 知识层次指导下的生物医学数据集精简训练领域专用语言模型 

**Authors**: Xunxin Cai, Chengrui Wang, Qingqing Long, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15108)  

**Abstract**: The rapid advancement of large language models (LLMs) in biological-medical applications has highlighted a gap between their potential and the limited scale and often low quality of available open-source annotated textual datasets. In addition, the inherent complexity of the biomedical knowledge hierarchy significantly hampers efforts to bridge this this http URL LLMs themselves play a pivotal role in overcoming this limitation? Motivated by this question, we investigate this challenge in the present this http URL propose a framework that automates the distillation of high-quality textual training data from the extensive scientific literature. Our approach self-evaluates and generates questions that are more closely aligned with the biomedical domain, guided by the biomedical knowledge hierarchy through medical subject headings (MeSH). This comprehensive framework establishes an automated workflow, thereby eliminating the need for manual intervention. Furthermore, we conducted comprehensive experiments to evaluate the impact of our framework-generated data on downstream language models of varying sizes. Our approach substantially improves question-answering tasks compared to pre-trained models from the life sciences domain and powerful close-source models represented by GPT-4. Notably, the generated AI-Ready dataset enabled the Llama3-70B base model to outperform GPT-4 using MedPrompt with multiple times the number of parameters. Detailed case studies and ablation experiments underscore the significance of each component within our framework 

**Abstract (ZH)**: 大型语言模型（LLMs）在生物医疗应用领域的迅速发展揭示了其潜力与可用开源标注文本数据的有限规模和较低质量之间的差距。此外，生物医疗知识层次结构的内在复杂性严重阻碍了弥合这一差距的努力。LLMs 自身在克服这一限制方面发挥着关键作用。鉴于这一问题，我们在此研究中探讨了这一挑战，并提出了一种自动提取高质量文本训练数据的框架，这些数据源自广泛的科学文献。我们的方法能够自我评估并生成更接近生物医学领域的具体问题，这些问题是根据医学主题头衔（MeSH）引导的生物医学知识层次结构生成的。该综合框架建立了一种自动化工作流，从而消除了人工干预的需要。此外，我们进行了全面的实验，以评估由我们的框架生成的数据对不同规模下游语言模型的影响。我们的方法显著提高了问答任务的表现，超过了生命科学领域的预训练模型和强大的闭源模型（如GPT-4）。值得注意的是，生成的AI就绪数据集使Llama3-70B基模型在使用MedPrompt的情况下，参数数量多出好几倍，也能战胜GPT-4。详细的案例研究和降维实验进一步强调了我们框架中每个组件的重要性。 

---
# Speech Translation Refinement using Large Language Models 

**Title (ZH)**: 使用大语言模型进行语音翻译精炼 

**Authors**: Huaixia Dou, Xinyu Tian, Xinglin Lyu, Jie Zhu, Junhui Li, Lifan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15090)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated their remarkable capabilities across various language tasks. Inspired by the success of text-to-text translation refinement, this paper investigates how LLMs can improve the performance of speech translation by introducing a joint refinement process. Through the joint refinement of speech translation (ST) and automatic speech recognition (ASR) transcription via LLMs, the performance of the ST model is significantly improved in both training-free in-context learning and parameter-efficient fine-tuning scenarios. Additionally, we explore the effect of document-level context on refinement under the context-aware fine-tuning scenario. Experimental results on the MuST-C and CoVoST 2 datasets, which include seven translation tasks, demonstrate the effectiveness of the proposed approach using several popular LLMs including GPT-3.5-turbo, LLaMA3-8B, and Mistral-12B. Further analysis further suggests that jointly refining both transcription and translation yields better performance compared to refining translation alone. Meanwhile, incorporating document-level context significantly enhances refinement performance. We release our code and datasets on GitHub. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种语言任务中的出色表现已经得到了验证。受文本到文本转换 refinement 成功的启发，本论文研究了通过引入联合 refinement 过程，LLMs 如何提升语音翻译（ST）的性能。通过利用 LLMs 对语音翻译和自动语音识别（ASR）转写进行联合 refinement，ST 模型在无训练情况下的上下文学习和参数高效微调场景中都表现出显著的性能提升。此外，我们还探讨了在上下文感知微调场景中文档级上下文对 refinement 的影响。在包含七个翻译任务的 MuST-C 和 CoVoST 2 数据集上的实验结果表明，使用包括 GPT-3.5-turbo、LLaMA-3-8B 和 Mistral-12B 等几种流行的 LLMs，所提出的方法具有有效性。进一步的分析还表明，同时对转写和翻译进行 joint refinement 比仅对翻译进行 refinement 能够获得更好的性能。同时，在 refinement 过程中引入文档级上下文能够显著提高性能。我们已在 GitHub 上公开了我们的代码和数据集。 

---
# LongReason: A Synthetic Long-Context Reasoning Benchmark via Context Expansion 

**Title (ZH)**: 长推理：通过上下文扩展生成的长语境推理基准 

**Authors**: Zhan Ling, Kang Liu, Kai Yan, Yifan Yang, Weijian Lin, Ting-Han Fan, Lingfeng Shen, Zhengyin Du, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15089)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable progress in understanding long-context inputs. However, benchmarks for evaluating the long-context reasoning abilities of LLMs fall behind the pace. Existing benchmarks often focus on a narrow range of tasks or those that do not demand complex reasoning. To address this gap and enable a more comprehensive evaluation of the long-context reasoning capabilities of current LLMs, we propose a new synthetic benchmark, LongReason, which is constructed by synthesizing long-context reasoning questions from a varied set of short-context reasoning questions through context expansion. LongReason consists of 794 multiple-choice reasoning questions with diverse reasoning patterns across three task categories: reading comprehension, logical inference, and mathematical word problems. We evaluate 21 LLMs on LongReason, revealing that most models experience significant performance drops as context length increases. Our further analysis shows that even state-of-the-art LLMs still have significant room for improvement in providing robust reasoning across different tasks. We will open-source LongReason to support the comprehensive evaluation of LLMs' long-context reasoning capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在理解长文本上下文方面取得了显著进步。然而，评估LLMs长文本推理能力的基准测试却落在了后面。现有的基准测试往往聚焦于狭窄的任务范围或那些不需要复杂推理的任务。为了填补这一空白并使对当前LLMs长文本推理能力的评估更加全面，我们提出了一种新的合成基准测试——LongReason，该基准测试通过扩展上下文将各种短文本推理问题综合为长文本推理问题。LongReason 包括 794 个多选推理问题，这些问题在三个任务类别（阅读理解、逻辑推理和数学应用题）中具有多样化的推理模式。我们评估了 21 种不同的LLMs在LongReason上的表现，结果显示大多数模型在其推理能力随着上下文长度增加时出现显著下降。进一步分析表明，即使是最先进的LLMs在跨不同任务提供稳健推理方面仍有显著改进的空间。我们将开源LongReason，以支持对LLMs长文本推理能力的全面评估。 

---
# Cross-modal Context Fusion and Adaptive Graph Convolutional Network for Multimodal Conversational Emotion Recognition 

**Title (ZH)**: 跨模态上下文融合与自适应图卷积网络在多模态对话情感识别中的应用 

**Authors**: Junwei Feng, Xueyan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15063)  

**Abstract**: Emotion recognition has a wide range of applications in human-computer interaction, marketing, healthcare, and other fields. In recent years, the development of deep learning technology has provided new methods for emotion recognition. Prior to this, many emotion recognition methods have been proposed, including multimodal emotion recognition methods, but these methods ignore the mutual interference between different input modalities and pay little attention to the directional dialogue between speakers. Therefore, this article proposes a new multimodal emotion recognition method, including a cross modal context fusion module, an adaptive graph convolutional encoding module, and an emotion classification module. The cross modal context module includes a cross modal alignment module and a context fusion module, which are used to reduce the noise introduced by mutual interference between different input modalities. The adaptive graph convolution module constructs a dialogue relationship graph for extracting dependencies and self dependencies between speakers. Our model has surpassed some state-of-the-art methods on publicly available benchmark datasets and achieved high recognition accuracy. 

**Abstract (ZH)**: 情感识别在人机交互、市场营销、医疗保健等多个领域有着广泛的应用。近年来，深度学习技术的发展为情感识别提供了新的方法。在此之前，已经提出了一些情感识别方法，包括多模态情感识别方法，但这些方法忽略了不同输入模态之间的相互干扰，并且很少关注对话者之间的双向对话。因此，本文提出了一种新的多模态情感识别方法，包括跨模态上下文融合模块、自适应图卷积编码模块和情感分类模块。跨模态上下文模块包括跨模态对齐模块和上下文融合模块，用于减少不同输入模态之间相互干扰引入的噪声。自适应图卷积模块构建了一个对话关系图，用于提取对话者之间的依赖关系和自我依赖关系。我们的模型在公开可用的标准数据集上超过了某些最先进的方法，并达到了较高的识别精度。 

---
# An Attempt to Unraveling Token Prediction Refinement and Identifying Essential Layers of Large Language Models 

**Title (ZH)**: 尝试解析.token预测精炼并识别大规模语言模型中的核心层 

**Authors**: Jaturong Kongmanee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15054)  

**Abstract**: This research aims to unravel how large language models (LLMs) iteratively refine token predictions (or, in a general sense, vector predictions). We utilized a logit lens technique to analyze the model's token predictions derived from intermediate representations. Specifically, we focused on how LLMs access and use information from input contexts, and how positioning of relevant information affects the model's token prediction refinement process. Our findings for multi-document question answering task, by varying input context lengths (the number of documents), using GPT-2, revealed that the number of layers between the first layer that the model predicted next tokens correctly and the later layers that the model finalized its correct predictions, as a function of the position of relevant information (i.e., placing the relevant one at the beginning, middle, or end of the input context), has a nearly inverted U shape. We found that the gap between these two layers, on average, diminishes when relevant information is positioned at the beginning or end of the input context, suggesting that the model requires more refinements when processing longer contexts with relevant information situated in the middle, and highlighting which layers are essential for determining the correct output. Our analysis provides insights about how token predictions are distributed across different conditions, and establishes important connections to existing hypotheses and previous findings in AI safety research and development. 

**Abstract (ZH)**: 本研究旨在探究大型语言模型（LLMs）如何迭代优化词元预测（或更广泛地说，向量预测）。我们使用了逻辑斯蒂视窗技术来分析模型从中间表示中生成的词元预测。具体而言，我们重点关注LLMs如何访问并利用输入上下文中的信息，以及相关信息的位置如何影响模型词元预测的优化过程。通过对多文档问答任务的研究，我们使用GPT-2模型，随着输入上下文长度（文档数量）的变化，发现从模型首次准确预测下一个词元的层到模型最终确定正确预测的层之间的层数，作为相关信息位置的函数（即放在输入上下文的起始、中间或结尾），呈现出几乎倒U形的变化趋势。我们发现，当相关信息位于输入上下文的起始或结尾时，这两个层之间的差异通常会减小，表明当处理包含相关信息位于中间的较长上下文时，模型需要更多的优化，并突出哪些层对于确定正确输出至关重要。我们的分析提供了关于在不同条件下词元预测分布的见解，并与现有假设以及人工智能安全研究和开发中的先前发现建立了重要的联系。 

---
# Abstractive Text Summarization for Bangla Language Using NLP and Machine Learning Approaches 

**Title (ZH)**: 使用自然语言处理和机器学习方法的孟加拉语抽象性文本摘要 

**Authors**: Asif Ahammad Miazee, Tonmoy Roy, Md Robiul Islam, Yeamin Safat  

**Link**: [PDF](https://arxiv.org/pdf/2501.15051)  

**Abstract**: Text summarization involves reducing extensive documents to short sentences that encapsulate the essential ideas. The goal is to create a summary that effectively conveys the main points of the original text. We spend a significant amount of time each day reading the newspaper to stay informed about current events both domestically and internationally. While reading newspapers enriches our knowledge, we sometimes come across unnecessary content that isn't particularly relevant to our lives. In this paper, we introduce a neural network model designed to summarize Bangla text into concise and straightforward paragraphs, aiming for greater stability and efficiency. 

**Abstract (ZH)**: 文本摘要涉及将 extensive 文档缩减为简洁的句子，以概括其核心思想。目标是在保留原始文本主要观点的同时，创建一个高效的总结。我们每天花费大量时间阅读报纸，以了解国内外的最新动态。阅读报纸虽能丰富我们的知识，但也可能会遇到与个人生活关系不大的冗余内容。本文提出了一种神经网络模型，旨在将孟加拉语文本总结为简明扼要的段落，以实现更高的稳定性和效率。 

---
# SCCD: A Session-based Dataset for Chinese Cyberbullying Detection 

**Title (ZH)**: SCCD：面向中文网络欺凌检测的会话数据集 

**Authors**: Qingpo Yang, Yakai Chen, Zihui Xu, Yu-ming Shang, Sanchuan Guo, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15042)  

**Abstract**: The rampant spread of cyberbullying content poses a growing threat to societal well-being. However, research on cyberbullying detection in Chinese remains underdeveloped, primarily due to the lack of comprehensive and reliable datasets. Notably, no existing Chinese dataset is specifically tailored for cyberbullying detection. Moreover, while comments play a crucial role within sessions, current session-based datasets often lack detailed, fine-grained annotations at the comment level. To address these limitations, we present a novel Chinese cyber-bullying dataset, termed SCCD, which consists of 677 session-level samples sourced from a major social media platform Weibo. Moreover, each comment within the sessions is annotated with fine-grained labels rather than conventional binary class labels. Empirically, we evaluate the performance of various baseline methods on SCCD, highlighting the challenges for effective Chinese cyberbullying detection. 

**Abstract (ZH)**: 网络欺凌内容的泛滥对社会福祉构成了日益严重的威胁。然而，关于中文网络欺凌检测的研究仍不够发达，主要原因是缺乏全面和可靠的数据集。值得注意的是，目前尚不存在专门针对网络欺凌检测的中文数据集。此外，虽然评论在会话中扮演着重要角色，但现有的基于会话的数据集往往缺乏针对评论的详细、细粒度的标注。为解决这些局限性，我们提出了一种新的中文网络欺凌数据集，称为SCCD，该数据集包含来自微博这一主要社交媒体平台的677个会话级样本。此外，SCCD中的每个评论都标注了细粒度标签，而不是传统的二分类标签。通过实证研究，我们评估了各种基线方法在SCCD上的性能，突出了有效中文网络欺凌检测所面临的挑战。 

---
# Using Large Language Models for education managements in Vietnamese with low resources 

**Title (ZH)**: 使用大规模语言模型进行越南低资源环境下的教育管理 

**Authors**: Duc Do Minh, Vinh Nguyen Van, Thang Dam Cong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15022)  

**Abstract**: Large language models (LLMs), such as GPT-4, Gemini 1.5, Claude 3.5 Sonnet, and Llama3, have demonstrated significant advancements in various NLP tasks since the release of ChatGPT in 2022. Despite their success, fine-tuning and deploying LLMs remain computationally expensive, especially in resource-constrained environments. In this paper, we proposed VietEduFrame, a framework specifically designed to apply LLMs to educational management tasks in Vietnamese institutions. Our key contribution includes the development of a tailored dataset, derived from student education documents at Hanoi VNU, which addresses the unique challenges faced by educational systems with limited resources. Through extensive experiments, we show that our approach outperforms existing methods in terms of accuracy and efficiency, offering a promising solution for improving educational management in under-resourced environments. While our framework leverages synthetic data to supplement real-world examples, we discuss potential limitations regarding broader applicability and robustness in future implementations. 

**Abstract (ZH)**: 自2022年ChatGPT发布以来，大型语言模型（LLMs）如GPT-4、Gemini 1.5、Claude 3.5 Sonnet和Llama3已经在各种自然语言处理（NLP）任务中展示了显著的进步。尽管取得了成功，但对LLMs的微调和部署仍然在资源受限的环境中非常昂贵。本文提出了VietEduFrame框架，专门用于在越南机构中应用LLMs于教育管理任务。我们的主要贡献包括开发了一个针对学生教育文档特制的数据集，该数据集源自河内VNU大学，以应对资源有限的教育系统所面临的具体挑战。通过广泛的实验，我们展示了与现有方法相比，我们的方法在准确性和效率方面表现出更优的性能，为改善资源有限环境中的教育管理提供了可行的解决方案。尽管框架利用合成数据来补充实际示例，我们还讨论了未来实施中的广泛应用性和鲁棒性可能存在的限制。 

---
# AKVQ-VL: Attention-Aware KV Cache Adaptive 2-Bit Quantization for Vision-Language Models 

**Title (ZH)**: AKVQ-VL：面向注意力意识的键值缓存自适应2位量化视觉-语言模型 

**Authors**: Zunhai Su, Wang Shen, Linge Li, Zhe Chen, Hanyu Wei, Huangqi Yu, Kehong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15021)  

**Abstract**: Vision-language models (VLMs) show remarkable performance in multimodal tasks. However, excessively long multimodal inputs lead to oversized Key-Value (KV) caches, resulting in significant memory consumption and I/O bottlenecks. Previous KV quantization methods for Large Language Models (LLMs) may alleviate these issues but overlook the attention saliency differences of multimodal tokens, resulting in suboptimal performance. In this paper, we investigate the attention-aware token saliency patterns in VLM and propose AKVQ-VL. AKVQ-VL leverages the proposed Text-Salient Attention (TSA) and Pivot-Token-Salient Attention (PSA) patterns to adaptively allocate bit budgets. Moreover, achieving extremely low-bit quantization requires effectively addressing outliers in KV tensors. AKVQ-VL utilizes the Walsh-Hadamard transform (WHT) to construct outlier-free KV caches, thereby reducing quantization difficulty. Evaluations of 2-bit quantization on 12 long-context and multimodal tasks demonstrate that AKVQ-VL maintains or even improves accuracy, outperforming LLM-oriented methods. AKVQ-VL can reduce peak memory usage by 2.13x, support up to 3.25x larger batch sizes and 2.46x throughput. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在多模态任务中表现出色。然而，过长的多模态输入会导致关键值（KV）缓存过大，从而引起显著的内存消耗和输入输出瓶颈。先前用于大型语言模型（LLMs）的关键值量化方法可能缓解这些问题，但忽略了多模态令牌的注意力重要性差异，导致性能不佳。在本文中，我们研究了VLM中的注意力感知令牌重要性模式，并提出了AKVQ-VL方法。AKVQ-VL利用提出的文本注意力显著模式（TSA）和枢轴令牌注意力显著模式（PSA）来自适应分配比特预算。此外，实现极低比特量化需要有效解决KV张量中的异常值。AKVQ-VL利用沃尔什-哈达玛变换（WHT）构造出无异常值的KV缓存，从而降低了量化难度。在12个长上下文和多模态任务中进行的2比特量化评估显示，AKVQ-VL不仅保持甚至提高了精度，性能优于针对LLM的量化方法。AKVQ-VL可将峰值内存使用量减少2.13倍，支持多达3.25倍更大的批量大小和2.46倍的吞吐量。 

---
# MDEval: Evaluating and Enhancing Markdown Awareness in Large Language Models 

**Title (ZH)**: MDEval：评估与增强大型语言模型中的Markdown意识 

**Authors**: Zhongpu Chen, Yinfeng Liu, Long Shi, Zhi-Jie Wang, Xingyan Chen, Yu Zhao, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2501.15000)  

**Abstract**: Large language models (LLMs) are expected to offer structured Markdown responses for the sake of readability in web chatbots (e.g., ChatGPT). Although there are a myriad of metrics to evaluate LLMs, they fail to evaluate the readability from the view of output content structure. To this end, we focus on an overlooked yet important metric -- Markdown Awareness, which directly impacts the readability and structure of the content generated by these language models. In this paper, we introduce MDEval, a comprehensive benchmark to assess Markdown Awareness for LLMs, by constructing a dataset with 20K instances covering 10 subjects in English and Chinese. Unlike traditional model-based evaluations, MDEval provides excellent interpretability by combining model-based generation tasks and statistical methods. Our results demonstrate that MDEval achieves a Spearman correlation of 0.791 and an accuracy of 84.1% with human, outperforming existing methods by a large margin. Extensive experimental results also show that through fine-tuning over our proposed dataset, less performant open-source models are able to achieve comparable performance to GPT-4o in terms of Markdown Awareness. To ensure reproducibility and transparency, MDEval is open sourced at this https URL. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的内容：

大型语言模型（LLMs）被期望能够生成结构化的Markdown响应，以提高网络聊天机器人的可读性（例如，ChatGPT）。尽管有许多评价LLMs的方法，它们未能从输出内容结构的角度衡量可读性。针对这一问题，我们关注一个被忽视但重要的指标——Markdown意识（Markdown Awareness），该指标直接影响这些语言模型生成的内容的可读性和结构。在本文中，我们介绍了MDEval，这是一种全面的基准，用于评估LLMs的Markdown意识，通过构建包含20,000个实例、涵盖10个主题（英语和汉语）的数据集来实现。不同于传统的基于模型的评估方法，MDEval通过结合基于模型的生成任务和统计方法，提供了出色的可解释性。我们的结果显示，MDEval与人类评估之间的 Spearman 相关系数为0.791，准确率为84.1%，远超现有方法。广泛的实验结果还表明，通过在我们提出的数据集上进行微调，表现较差的开源模型能够达到与GPT-4o相当的Markdown意识水平。为了确保可再现性和透明度，MDEval已在此处开放源代码：[提供链接的地方]。 

---
# Federated Retrieval Augmented Generation for Multi-Product Question Answering 

**Title (ZH)**: 联邦检索增强生成在多产品问答中的应用 

**Authors**: Parshin Shojaee, Sai Sree Harsha, Dan Luo, Akash Maharaj, Tong Yu, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14998)  

**Abstract**: Recent advancements in Large Language Models and Retrieval-Augmented Generation have boosted interest in domain-specific question-answering for enterprise products. However, AI Assistants often face challenges in multi-product QA settings, requiring accurate responses across diverse domains. Existing multi-domain RAG-QA approaches either query all domains indiscriminately, increasing computational costs and LLM hallucinations, or rely on rigid resource selection, which can limit search results. We introduce MKP-QA, a novel multi-product knowledge-augmented QA framework with probabilistic federated search across domains and relevant knowledge. This method enhances multi-domain search quality by aggregating query-domain and query-passage probabilistic relevance. To address the lack of suitable benchmarks for multi-product QAs, we also present new datasets focused on three Adobe products: Adobe Experience Platform, Target, and Customer Journey Analytics. Our experiments show that MKP-QA significantly boosts multi-product RAG-QA performance in terms of both retrieval accuracy and response quality. 

**Abstract (ZH)**: 近年来，大型语言模型和检索增强生成技术的最新进展激发了对特定领域的企业产品问答的兴趣。然而，在多产品问答设置中，AI助手常常面临挑战，需要在多样化的领域中提供精确的响应。现有的多领域RAG-QA方法要么不分青红皂白地查询所有领域，增加计算成本并导致LLM幻觉，要么依赖刚性资源选择，这可能限制搜索结果。我们提出了MKP-QA，这是一种新颖的多产品知识增强问答框架，包括跨领域和相关知识的概率联邦搜索。该方法通过聚合查询领域和查询段落的概率相关性来提高多领域的搜索质量。为了应对多产品问答缺乏合适的基准测试，我们还介绍了新的针对Adobe三项产品的数据集：Adobe Experience Platform、Target和Customer Journey Analytics。实验结果显示，MKP-QA在检索准确性和响应质量方面显著提升了多产品RAG-QA的性能。 

---
# The Muddy Waters of Modeling Empathy in Language: The Practical Impacts of Theoretical Constructs 

**Title (ZH)**: 语言中同理心建模的浑 waters：理论构念的实际影响 

**Authors**: Allison Lahnala, Charles Welch, David Jurgens, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2501.14981)  

**Abstract**: Conceptual operationalizations of empathy in NLP are varied, with some having specific behaviors and properties, while others are more abstract. How these variations relate to one another and capture properties of empathy observable in text remains unclear. To provide insight into this, we analyze the transfer performance of empathy models adapted to empathy tasks with different theoretical groundings. We study (1) the dimensionality of empathy definitions, (2) the correspondence between the defined dimensions and measured/observed properties, and (3) the conduciveness of the data to represent them, finding they have a significant impact to performance compared to other transfer setting features. Characterizing the theoretical grounding of empathy tasks as direct, abstract, or adjacent further indicates that tasks that directly predict specified empathy components have higher transferability. Our work provides empirical evidence for the need for precise and multidimensional empathy operationalizations. 

**Abstract (ZH)**: 情感处理器在自然语言处理（NLP）中的概念化形式多种多样，有的具有特定的行为和属性，而有的则更为抽象。这些差异之间的关系以及它们如何捕捉文本中可观察的情感特性，仍有待明确。为了解这一问题，我们分析了适用于不同理论基础的情感模型在情感任务中的传递性能。我们研究了（1）情感定义的维度性、（2）定义维度与测量/观察到的属性之间的对应关系，以及（3）数据对这些属性的表征性，发现与其它转移设置特征相比，这些因素对性能有显著影响。进一步将情感任务的理论基础分类为直接、抽象或相邻，表明直接预测具体情感成分的任务具有更高的可转移性。我们的研究提供了支持精确和多维度情感概念化的实证证据。 

---
# A review of annotation classification tools in the educational domain 

**Title (ZH)**: 教育领域标注分类工具的综述 

**Authors**: Joaquín Gayoso-Cabada, Antonio Sarasa-Cabezuelo, José-Luis Sierra  

**Link**: [PDF](https://arxiv.org/pdf/2501.14976)  

**Abstract**: An annotation consists of a portion of information that is associated with a piece of content in order to explain something about the content or to add more information. The use of annotations as a tool in the educational field has positive effects on the learning process. The usual way to use this instrument is to provide students with contents, usually textual, with which they must associate annotations. In most cases this task is performed in groups of students who work collaboratively. This process encourages analysis and understanding of the contents since they have to understand them in order to annotate them, and also encourages teamwork. To facilitate its use, computer applications have been devel-oped in recent decades that implement the annotation process and offer a set of additional functionalities. One of these functionalities is the classification of the annotations made. This functionality can be exploited in various ways in the learning process, such as guiding the students in the annotation process, providing information to the student about how the annotation process is done and to the teacher about how the students write and how they understand the content, as well as implementing other innovative educational processes. In this sense, the classification of annotations plays a critical role in the application of the annotation in the educational field. There are many studies of annotations, but most of them consider the classification aspect marginally only. This paper presents an initial study of the classification mech-anisms used in the annotation tools, identifying four types of cases: absence of classification mechanisms, classification based on pre-established vocabularies, classification based on extensible vocabularies, and classification based on struc-tured vocabularies. 

**Abstract (ZH)**: 注释是由与内容相关的部分信息构成，以解释内容或添加更多信息。在教育领域使用注释作为工具对学习过程具有积极影响。通常使用这种方式是在学生面前提供内容（通常是文本），要求学生将注释与内容关联起来。在大多数情况下，这项任务是在学生小组中协作完成的。这一过程促进了对内容的理解和分析，因为他们需要理解内容才能对其进行注释，并且也促进了团队合作。为方便使用，在过去几十年中，计算机应用程序已被开发，以实现注释过程并提供一系列额外功能。这些功能之一是对所做的注释进行分类。在学习过程中，这种分类功能可以以多种方式进行利用，例如引导学生进行注释过程，向学生提供有关注释过程的信息，向教师提供有关学生如何写注释和如何理解内容的信息，以及实施其他创新性教育过程。在这种意义上，注释的分类在教育领域应用中起着关键作用。尽管有许多关于注释的研究，但大多数仅在分类方面有所涉及。本文介绍了注释工具中使用的分类机制的初步研究，并识别了四种类型的情况：缺乏分类机制、基于预定义词汇的分类、基于可扩展词汇的分类，以及基于结构化词汇的分类。 

---
# ExPerT: Effective and Explainable Evaluation of Personalized Long-Form Text Generation 

**Title (ZH)**: ExPerT: 有效的可解释个性化长文本生成评估 

**Authors**: Alireza Salemi, Julian Killingback, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.14956)  

**Abstract**: Evaluating personalized text generated by large language models (LLMs) is challenging, as only the LLM user, i.e., prompt author, can reliably assess the output, but re-engaging the same individuals across studies is infeasible. This paper addresses the challenge of evaluating personalized text generation by introducing ExPerT, an explainable reference-based evaluation framework. ExPerT leverages an LLM to extract atomic aspects and their evidence from the generated and reference texts, match the aspects, and evaluate their alignment based on content and writing style -- two key attributes in personalized text generation. Additionally, ExPerT generates detailed, fine-grained explanations for every step of the evaluation process, enhancing transparency and interpretability. Our experiments demonstrate that ExPerT achieves a 7.2% relative improvement in alignment with human judgments compared to the state-of-the-art text generation evaluation methods. Furthermore, human evaluators rated the usability of ExPerT's explanations at 4.7 out of 5, highlighting its effectiveness in making evaluation decisions more interpretable. 

**Abstract (ZH)**: 评估由大规模语言模型（LLMs）生成的个性化文本具有挑战性，因为只有LLM的用户，即提示作者，才能可靠地评估输出，但在不同研究中重新 Engagement 同一个体是不可行的。本文通过引入ExPerT，一种基于解释的参考框架评估方法，来应对个性化文本生成的评估挑战。ExPerT 利用LLM从生成文本和参考文本中提取基本方面及其证据，匹配这些方面，并基于内容和写作风格评估它们的一致性——这是个性化文本生成的两个关键属性。此外，ExPerT 为评估过程中的每个步骤生成详细的细粒度解释，从而提高透明度和可解释性。我们的实验表明，与当前最先进的文本生成评估方法相比，ExPerT 在一致性方面表现出 7.2% 的相对改进。此外，人类评估者对ExPerT解释的使用评价为4.7分（满分5分），这突显了ExPerT在使评估决策更易于解释方面的有效性。 

---
# CASE-Bench: Context-Aware Safety Evaluation Benchmark for Large Language Models 

**Title (ZH)**: CASE-Bench：面向大型语言模型的上下文感知安全性评估基准 

**Authors**: Guangzhi Sun, Xiao Zhan, Shutong Feng, Philip C. Woodland, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2501.14940)  

**Abstract**: Aligning large language models (LLMs) with human values is essential for their safe deployment and widespread adoption. Current LLM safety benchmarks often focus solely on the refusal of individual problematic queries, which overlooks the importance of the context where the query occurs and may cause undesired refusal of queries under safe contexts that diminish user experience. Addressing this gap, we introduce CASE-Bench, a Context-Aware Safety Evaluation Benchmark that integrates context into safety assessments of LLMs. CASE-Bench assigns distinct, formally described contexts to categorized queries based on Contextual Integrity theory. Additionally, in contrast to previous studies which mainly rely on majority voting from just a few annotators, we recruited a sufficient number of annotators necessary to ensure the detection of statistically significant differences among the experimental conditions based on power analysis. Our extensive analysis using CASE-Bench on various open-source and commercial LLMs reveals a substantial and significant influence of context on human judgments (p<0.0001 from a z-test), underscoring the necessity of context in safety evaluations. We also identify notable mismatches between human judgments and LLM responses, particularly in commercial models within safe contexts. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与人类价值相一致是确保其安全部署和广泛应用的关键。当前的LLM安全评估基准往往仅关注对个别问题的拒绝，而忽视了查询发生的上下文的重要性，可能导致在安全上下文中不必要地拒绝查询，从而损害用户体验。为解决这一问题，我们引入了CASE-Bench，这是一种基于上下文的安全评估基准，将上下文整合进对LLMs的安全评估中。CASE-Bench 根据情境完备性理论为分类后的查询分配不同的、形式化描述的上下文。此外，与以往主要依赖少数标注者投票的研究不同，我们通过功效分析招募了足够数量的标注者，以确保在实验条件下检测到统计显著差异。通过使用CASE-Bench 对各种开源和商用LLMs进行广泛分析，我们发现上下文对人类判断产生了显著影响（p<0.0001来自Z检验），强调了在安全评估中考虑上下文的重要性。我们还发现，在安全上下文中，人类判断与LLM响应之间存在明显的不一致性，尤其是在商用模型中更为明显。 

---
# Context-Aware Neural Gradient Mapping for Fine-Grained Instruction Processing 

**Title (ZH)**: 基于上下文的神经梯度映射方法用于细粒度指令处理 

**Authors**: David Boldo, Lily Pemberton, Gabriel Thistledown, Jacob Fairchild, Felix Kowalski  

**Link**: [PDF](https://arxiv.org/pdf/2501.14936)  

**Abstract**: The integration of contextual embeddings into the optimization processes of large language models is an advancement in natural language processing. The Context-Aware Neural Gradient Mapping framework introduces a dynamic gradient adjustment mechanism, incorporating contextual embeddings directly into the optimization process. This approach facilitates real-time parameter adjustments, enhancing task-specific generalization even in the presence of sparse or noisy data inputs. The mathematical foundation of this framework relies on gradient descent modifications, where contextual embeddings are derived from a supplementary neural network trained to map input features to optimal adaptation gradients. By employing differential geometry principles, high-dimensional input dependencies are encoded into low-dimensional gradient manifolds, enabling efficient adaptation without necessitating the retraining of the entire model. Empirical evaluations demonstrate that the proposed framework consistently outperforms baseline models across various metrics, including accuracy, robustness to noise, and computational efficiency. The integration of context-specific embeddings allows for a more complex understanding of language, thereby improving the model's ability to handle diverse linguistic phenomena. Furthermore, the computational efficiency achieved through this method demonstrates its scalability for large-scale language models operating under diverse constraints. 

**Abstract (ZH)**: 将上下文嵌入集成到大型语言模型的优化过程中是自然语言处理领域的一项进步。《基于上下文感知的神经梯度映射框架》引入了一种动态梯度调节机制，直接将上下文嵌入纳入优化过程。该方法促进实时参数调整，即使在稀疏或噪声数据输入的情况下也能提高任务特定的泛化能力。该框架的数学基础依赖于梯度下降的修改，其中上下文嵌入来源于一个补充神经网络，该网络训练用于将输入特征映射到最优适应梯度。通过应用微分几何原理，高维输入依赖关系被编码为低维梯度流形中，从而实现高效适应，而无需重新训练整个模型。实证评估表明，所提出框架在各种指标（包括精度、噪声鲁棒性和计算效率）上始终优于基础模型。特定上下文嵌入的集成使得对语言的理解更加复杂，从而提高了模型处理各种语义现象的能力。此外，通过该方法实现的计算效率表明，它具有在多种约束条件下扩展大规模语言模型的能力。 

---
# Self-reflecting Large Language Models: A Hegelian Dialectical Approach 

**Title (ZH)**: 自我反思的大语言模型：黑格尔辩证方法 

**Authors**: Sara Abdali, Can Goksen, Saeed Amizadeh andKazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2501.14917)  

**Abstract**: Investigating NLP through a philosophical lens has recently caught researcher's eyes as it connects computational methods with classical schools of philosophy. This paper introduces a philosophical approach inspired by the Hegelian Dialectic for LLMs' self-reflection, utilizing a self-dialectical approach to emulate internal critiques and then synthesize new ideas by resolving the contradicting points. Moreover, this paper investigates the effect of LLMs' temperature for generation by establishing a dynamic annealing approach, which promotes the creativity in the early stages and gradually refines it by focusing on the nuances, as well as a fixed temperature strategy for generation. Our proposed approach is examined to determine its ability to generate novel ideas from an initial proposition. Additionally, a Multi Agent Majority Voting (MAMV) strategy is leveraged to assess the validity and novelty of the generated ideas, which proves beneficial in the absence of domain experts. Our experiments show promise in generating new ideas and provide a stepping-stone for future research. 

**Abstract (ZH)**: 通过哲学视角研究自然语言处理（NLP）近年来引起了研究人员的关注，因为它将计算方法与古典哲学流派联系了起来。本文提出了一种受黑格尔辩证法启发的哲学方法，用于LLMs（大型语言模型）的自我反思。该方法采用自我辩证的方式模拟内部批判，并通过解决矛盾来综合新思想。此外，本文还探讨了温度对LLMs生成效果的影响，建立了动态退火方法，在早期阶段促进创造力，并在关注细节的同时逐步精细化，同时也提出了一种固定的温度策略。我们提出的方法被用于评估其从初始命题产生新颖思想的能力。此外，本文还利用多智能体多数投票（MAMV）策略评估生成思想的有效性和新颖性，这在缺乏领域专家的情况下特别有益。我们的实验展示了生成新思想的潜力，并为未来的研究奠定了基础。 

---
# Verify with Caution: The Pitfalls of Relying on Imperfect Factuality Metrics 

**Title (ZH)**: 慎验慎行：依赖不完美事实度量的风险 

**Authors**: Ameya Godbole, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2501.14883)  

**Abstract**: Improvements in large language models have led to increasing optimism that they can serve as reliable evaluators of natural language generation outputs. In this paper, we challenge this optimism by thoroughly re-evaluating five state-of-the-art factuality metrics on a collection of 11 datasets for summarization, retrieval-augmented generation, and question answering. We find that these evaluators are inconsistent with each other and often misestimate system-level performance, both of which can lead to a variety of pitfalls. We further show that these metrics exhibit biases against highly paraphrased outputs and outputs that draw upon faraway parts of the source documents. We urge users of these factuality metrics to proceed with caution and manually validate the reliability of these metrics in their domain of interest before proceeding. 

**Abstract (ZH)**: 大语言模型的进步使得人们越来越乐观，认为它们可以作为自然语言生成输出可靠评估者。在本文中，我们通过在摘要、检索增强生成和问答等11个数据集中全面重新评估五种最先进的事实性度量标准，挑战了这种乐观态度。我们发现，这些评估者彼此之间不一致，经常错误估计系统级性能，这两者都可能导致各种问题。进一步研究表明，这些度量标准对高度改写的输出以及引用源文档中遥远部分的输出存在偏见。我们敦促这些事实性度量标准的使用者在应用之前谨慎行事，并在他们感兴趣的研究领域手动验证这些度量标准的可靠性。 

---
# DrawEduMath: Evaluating Vision Language Models with Expert-Annotated Students' Hand-Drawn Math Images 

**Title (ZH)**: DrawEduMath: 专家标注的学生手绘数学图像评价视觉语言模型 

**Authors**: Sami Baral, Li Lucy, Ryan Knight, Alice Ng, Luca Soldaini, Neil T. Heffernan, Kyle Lo  

**Link**: [PDF](https://arxiv.org/pdf/2501.14877)  

**Abstract**: In real-world settings, vision language models (VLMs) should robustly handle naturalistic, noisy visual content as well as domain-specific language and concepts. For example, K-12 educators using digital learning platforms may need to examine and provide feedback across many images of students' math work. To assess the potential of VLMs to support educators in settings like this one, we introduce DrawEduMath, an English-language dataset of 2,030 images of students' handwritten responses to K-12 math problems. Teachers provided detailed annotations, including free-form descriptions of each image and 11,661 question-answer (QA) pairs. These annotations capture a wealth of pedagogical insights, ranging from students' problem-solving strategies to the composition of their drawings, diagrams, and writing. We evaluate VLMs on teachers' QA pairs, as well as 44,362 synthetic QA pairs derived from teachers' descriptions using language models (LMs). We show that even state-of-the-art VLMs leave much room for improvement on DrawEduMath questions. We also find that synthetic QAs, though imperfect, can yield similar model rankings as teacher-written QAs. We release DrawEduMath to support the evaluation of VLMs' abilities to reason mathematically over images gathered with educational contexts in mind. 

**Abstract (ZH)**: 在实际应用场景中，视觉语言模型（VLMs）应当能够稳健地处理自然且不规则的视觉内容，同时处理特定领域的语言和概念。例如，K-12 教育者在使用数字化学习平台时，可能需要审阅和提供反馈，涉及许多学生的数学作业图片。为了评估 VLMs 在类似这样环境中支持教育者的能力，我们引入了 DrawEduMath，这是一个包含 2,030 张学生手工解答 K-12 数学问题的图像的英语数据集。老师提供了详细的注释，包括每张图像的自由格式描述和 11,661 组问题-答案（QA）对。这些注释捕捉了丰富的教学见解，从学生的解题策略到他们绘制的图表和写作的组成。我们使用老师提供的 QA 对以及从老师描述中生成的 44,362 组合成的 QA 对（使用语言模型生成）来评估 VLMs 的性能。结果显示，即使是最先进的 VLMs 在处理 DrawEduMath 问题时仍有很大的改进空间。我们还发现，尽管合成的 QA 并不完美，但它们可以与教师撰写的 QA 对一样生成类似模型的排名。我们发布 DrawEduMath 以支持评估 VLMs 在教育情境下对图像进行数学推理的能力。 

---
# Dynamic Adaptation of LoRA Fine-Tuning for Efficient and Task-Specific Optimization of Large Language Models 

**Title (ZH)**: 面向高效和任务特定优化的大语言模型的LoRA微调动态调整 

**Authors**: Xiaoxuan Liao, Chihang Wang, Shicheng Zhou, Jiacheng Hu, Hongye Zheng, Jia Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14859)  

**Abstract**: This paper presents a novel methodology of fine-tuning for large language models-dynamic LoRA. Building from the standard Low-Rank Adaptation framework, this methodology further adds dynamic adaptation mechanisms to improve efficiency and performance. The key contribution of dynamic LoRA lies within its adaptive weight allocation mechanism coupled with an input feature-based adaptive strategy. These enhancements allow for a more precise fine-tuning process that is more tailored to specific tasks. Traditional LoRA methods use static adapter settings, not considering the different importance of model layers. In contrast, dynamic LoRA introduces a mechanism that dynamically evaluates the layer's importance during fine-tuning. This evaluation enables the reallocation of adapter parameters to fit the unique demands of each individual task, which leads to better optimization results. Another gain in flexibility arises from the consideration of the input feature distribution, which helps the model generalize better when faced with complicated and diverse datasets. The joint approach boosts not only the performance over each single task but also the generalization ability of the model. The efficiency of the dynamic LoRA was validated in experiments on benchmark datasets, such as GLUE, with surprising results. More specifically, this method achieved 88.1% accuracy with an F1-score of 87.3%. Noticeably, these improvements were made at a slight increase in computational costs: only 0.1% more resources than standard LoRA. This balance between performance and efficiency positions dynamic LoRA as a practical, scalable solution for fine-tuning LLMs, especially in resource-constrained scenarios. To take it a step further, its adaptability makes it a promising foundation for much more advanced applications, including multimodal tasks. 

**Abstract (ZH)**: 本文提出了一种新颖的大语言模型调优方法——动态LoRA。该方法在标准低秩适应框架的基础上，进一步引入了动态适应机制，以提高调优效率和性能。动态LoRA的核心贡献在于其自适应权重分配机制和基于输入特征的自适应策略。这些改进使得调优过程更加精准，更能针对特定任务进行定制。传统的LoRA方法采用静态适配器设置，不考虑模型层的重要性差异。而动态LoRA则引入了一种机制，在调优过程中动态评估每一层的重要性。这种评估使得适配器参数能够重新分配，以适应每个具体任务的独特需求，从而实现更好的优化结果。此外，考虑到输入特征的分布也能提高模型的灵活性，使其在面对复杂多样的数据集时具有更好的泛化能力。联合方法不仅提升了每项任务的性能，还增强了模型的泛化能力。实验结果表明，动态LoRA在基准数据集（如GLUE）上的效率得到了验证，结果令人惊喜。具体而言，该方法在准确率达到88.1%，F1分数达到87.3%的情况下表现出色。值得注意的是，这些改进仅带来了轻微的计算成本增加：只比标准LoRA多0.1%的资源。这种在性能与效率之间的平衡，使动态LoRA成为在资源受限条件下调优LLM的一种切实可行且可扩展的解决方案。进一步而言，其高度的适应性使其成为更高级应用的基础，包括多模态任务。 

---
# JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in Large Language Models 

**Title (ZH)**: JustLogic：评估大型语言模型演绎推理能力的综合性基准测试 

**Authors**: Michael K. Chen, Xikun Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14851)  

**Abstract**: Logical reasoning is a critical component of Large Language Models (LLMs), and substantial research efforts in recent years have aimed to enhance their deductive reasoning capabilities. However, existing deductive reasoning benchmarks, which are crucial for evaluating and advancing LLMs, are inadequate due to their lack of task complexity, presence of prior knowledge as a confounder, and superficial error analysis. To address these deficiencies, we introduce JustLogic, a synthetically generated deductive reasoning benchmark designed for rigorous evaluation of LLMs. JustLogic is (i) highly complex, capable of generating a diverse range of linguistic patterns, vocabulary, and argument structures; (ii) prior knowledge independent, eliminating the advantage of models possessing prior knowledge and ensuring that only deductive reasoning is used to answer questions; and (iii) capable of in-depth error analysis on the heterogeneous effects of reasoning depth and argument form on model accuracy. Our experimental results on JustLogic reveal that most state-of-the-art (SOTA) LLMs perform significantly worse than the human average, demonstrating substantial room for model improvement. All code and data are available at this https URL 

**Abstract (ZH)**: 逻辑推理是大型语言模型（LLMs）的一个关键组成部分，近年来，大量研究致力于提升其演绎推理能力。然而，现有的演绎推理基准在评估和促进LLMs方面存在不足，因为这些基准的任务复杂性较低、包含了先验知识的干扰，并且浅显的错误分析。为解决这些问题，我们提出JustLogic，这是一种专门为严格评估LLMs设计的合成演绎推理基准。JustLogic具有以下特点：（i）高度复杂，能够生成多种多样的语言模式、词汇和论证结构；（ii）不依赖于先验知识，去除模型依赖先验知识的优势，确保仅使用演绎推理来回答问题；（iii）能够进行深入的错误分析，探讨推理深度和论证形式对模型准确率的异质性影响。我们在JustLogic上的实验结果显示，大多数最新（SOTA）的LLMs的表现明显低于人类平均水平，表明模型仍有很大的改进空间。所有代码和数据可在以下链接获取：https://... 

---
# On the locality bias and results in the Long Range Arena 

**Title (ZH)**: 关于Long Range Arena中的局部偏见及其结果 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2501.14850)  

**Abstract**: The Long Range Arena (LRA) benchmark was designed to evaluate the performance of Transformer improvements and alternatives in long-range dependency modeling tasks. The Transformer and its main variants performed poorly on this benchmark, and a new series of architectures such as State Space Models (SSMs) gained some traction, greatly outperforming Transformers in the LRA. Recent work has shown that with a denoising pre-training phase, Transformers can achieve competitive results in the LRA with these new architectures. In this work, we discuss and explain the superiority of architectures such as MEGA and SSMs in the Long Range Arena, as well as the recent improvement in the results of Transformers, pointing to the positional and local nature of the tasks. We show that while the LRA is a benchmark for long-range dependency modeling, in reality most of the performance comes from short-range dependencies. Using training techniques to mitigate data inefficiency, Transformers are able to reach state-of-the-art performance with proper positional encoding. In addition, with the same techniques, we were able to remove all restrictions from SSM convolutional kernels and learn fully parameterized convolutions without decreasing performance, suggesting that the design choices behind SSMs simply added inductive biases and learning efficiency for these particular tasks. Our insights indicate that LRA results should be interpreted with caution and call for a redesign of the benchmark. 

**Abstract (ZH)**: 长范围竞技场（LRA）基准旨在评估Transformer及其改进和替代方案在长距离依赖建模任务中的性能。在这一基准上，Transformer及其主要变体表现不佳，而新的架构系列，如状态空间模型（SSMs）则取得了显著的进步，大大超越了Transformer。最近的研究表明，在去噪预训练阶段后，Transformer能够在LRA中与这些新架构取得竞争性的结果。在本文中，我们探讨并解释了MEGA和SSMs等架构在长范围竞技场中的优越性，以及Transformer性能的最新改进，强调了这些任务的局部和短距离依赖性质。我们发现，虽然LRA是一个长距离依赖建模的基准，但事实上大部分性能来自于短距离依赖。通过使用训练技术来缓解数据效率问题，Transformer能够通过适当的位移编码达到最先进的性能。此外，同样使用这些技术，我们能够移除SSMs卷积核的所有约束，学习完全参数化的卷积而不降低性能，这表明SSMs的设计选择仅仅增加了这些特定任务的归纳偏差和学习效率。我们的见解表明，LRA的结果应谨慎解释，并呼吁重新设计该基准。 

---
# Unmasking Conversational Bias in AI Multiagent Systems 

**Title (ZH)**: 揭示AI多智能体系统中的对话偏见 

**Authors**: Erica Coppolillo, Giuseppe Manco, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2501.14844)  

**Abstract**: Detecting biases in the outputs produced by generative models is essential to reduce the potential risks associated with their application in critical settings. However, the majority of existing methodologies for identifying biases in generated text consider the models in isolation and neglect their contextual applications. Specifically, the biases that may arise in multi-agent systems involving generative models remain under-researched. To address this gap, we present a framework designed to quantify biases within multi-agent systems of conversational Large Language Models (LLMs). Our approach involves simulating small echo chambers, where pairs of LLMs, initialized with aligned perspectives on a polarizing topic, engage in discussions. Contrary to expectations, we observe significant shifts in the stance expressed in the generated messages, particularly within echo chambers where all agents initially express conservative viewpoints, in line with the well-documented political bias of many LLMs toward liberal positions. Crucially, the bias observed in the echo-chamber experiment remains undetected by current state-of-the-art bias detection methods that rely on questionnaires. This highlights a critical need for the development of a more sophisticated toolkit for bias detection and mitigation for AI multi-agent systems. The code to perform the experiments is publicly available at this https URL. 

**Abstract (ZH)**: 检测生成模型输出中的偏见对于降低其在关键应用场景中的潜在风险至关重要。然而，大多数现有的偏见检测方法仅在孤立状态下分析这些模型，忽视了它们的上下文应用。特别是，在涉及生成模型的多智能体系统中可能出现的偏见仍未得到充分研究。为填补这一空白，我们提出了一种框架，用于量化多智能体系统中的对话型大型语言模型（LLMs）中的偏见。我们的方法涉及模拟小规模的回声室，在这种回声室中，以一个极化的主题为基础，以具有相同观点的两个LLM对进行互动交流。出人意料的是，我们在回声室中观察到生成消息所表达立场的显著变化，特别是在所有智能体最初都表达保守观点的回声室中，这与许多LLM已充分记录的政治偏见（指向更自由的政治立场）相符。更重要的是，目前依赖问卷的最先进的偏见检测方法无法检测回声室实验中的这种偏见。这凸显了为AI多智能体系统开发更强大的偏见检测和缓解工具的重要性。实验代码已在以下网址公开：[这个 https URL](这个 https URL)。 

---
# Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity 

**Title (ZH)**: Mixture-of-Mamba：增强多模态状态空间模型的模态感知稀疏性 

**Authors**: Weixin Liang, Junhong Shen, Genghan Zhang, Ning Dong, Luke Zettlemoyer, Lili Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16295)  

**Abstract**: State Space Models (SSMs) have emerged as efficient alternatives to Transformers for sequential modeling, but their inability to leverage modality-specific features limits their performance in multi-modal pretraining. Here, we propose Mixture-of-Mamba, a novel SSM architecture that introduces modality-aware sparsity through modality-specific parameterization of the Mamba block. Building on Mixture-of-Transformers (W. Liang et al. arXiv:2411.04996; 2024), we extend the benefits of modality-aware sparsity to SSMs while preserving their computational efficiency. We evaluate Mixture-of-Mamba across three multi-modal pretraining settings: Transfusion (interleaved text and continuous image tokens with diffusion loss), Chameleon (interleaved text and discrete image tokens), and an extended three-modality framework incorporating speech. Mixture-of-Mamba consistently reaches the same loss values at earlier training steps with significantly reduced computational costs. In the Transfusion setting, Mixture-of-Mamba achieves equivalent image loss using only 34.76% of the training FLOPs at the 1.4B scale. In the Chameleon setting, Mixture-of-Mamba reaches similar image loss with just 42.50% of the FLOPs at the 1.4B scale, and similar text loss with just 65.40% of the FLOPs. In the three-modality setting, MoM matches speech loss at 24.80% of the FLOPs at the 1.4B scale. Our ablation study highlights the synergistic effects of decoupling projection components, where joint decoupling yields greater gains than individual modifications. These results establish modality-aware sparsity as a versatile and effective design principle, extending its impact from Transformers to SSMs and setting new benchmarks in multi-modal pretraining. Our code can be accessed at this https URL 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

自回归模型（State Space Models, SSMs）已逐渐成为Transformer在序列建模方面的有效替代方案，但它们在多模态预训练中无法充分利用特定模态特征的能力限制了其性能。为此，我们提出了一种名为Mixture-of-Mamba的新颖SSM架构，通过Mamba块的模态特定参数化引入模态感知稀疏性。该架构基于Mixture-of-Transformers（W. Liang等，arXiv:2411.04996，2024）的研究，在保留SSM高效计算特性的同时，进一步推广了模态感知稀疏性的优势。我们分别在三种多模态预训练设置中评估了Mixture-of-Mamba的表现：Transfusion（交错排列的文本和连续图像标记，并使用扩散损失）、Chameleon（交错排列的文本和离散图像标记）以及包含语音的扩展三模态框架。结果显示，Mixture-of-Mamba在早期训练步骤中以显著减少的计算成本达到了相同损失值。在Transfusion设置中，Mixture-of-Mamba仅使用1.4B模型规模下34.76%的训练FLOP实现了与全量计算相等的图像损失。在Chameleon设置中，Mixture-of-Mamba在1.4B模型规模下仅使用42.50%的FLOP达到了相似的图像损失和65.40%的FLOP实现了相似的文本损失。在三模态设置中，Mixture-of-Mamba在1.4B模型规模下仅使用24.80%的FLOP达到了相似的语音损失。我们的消融研究强调了解耦投影组件的协同效应，表明联合解耦方式提供的增益大于单独修改方式。这些结果证明，模态感知稀疏性是一种灵活且有效的设计原则，它的影响不仅限于Transformer，还能扩展到SSM，并在多模态预训练中设立了新的基准。我们的代码可在以下链接访问：[this https URL] 

---
# Zero-Shot Decision Tree Construction via Large Language Models 

**Title (ZH)**: 基于大型语言模型的零样本决策树构建 

**Authors**: Lucas Carrasco, Felipe Urrutia, Andrés Abeliuk  

**Link**: [PDF](https://arxiv.org/pdf/2501.16247)  

**Abstract**: This paper introduces a novel algorithm for constructing decision trees using large language models (LLMs) in a zero-shot manner based on Classification and Regression Trees (CART) principles. Traditional decision tree induction methods rely heavily on labeled data to recursively partition data using criteria such as information gain or the Gini index. In contrast, we propose a method that uses the pre-trained knowledge embedded in LLMs to build decision trees without requiring training data. Our approach leverages LLMs to perform operations essential for decision tree construction, including attribute discretization, probability calculation, and Gini index computation based on the probabilities. We show that these zero-shot decision trees can outperform baseline zero-shot methods and achieve competitive performance compared to supervised data-driven decision trees on tabular datasets. The decision trees constructed via this method provide transparent and interpretable models, addressing data scarcity while preserving interpretability. This work establishes a new baseline in low-data machine learning, offering a principled, knowledge-driven alternative to data-driven tree construction. 

**Abstract (ZH)**: 本文介绍了一种基于分类和回归树（CART）原则，使用大型语言模型（LLMs）以零样本方式构造决策树的新算法。传统的决策树归纳方法高度依赖标记数据，通过诸如信息增益或基尼指数等标准递归地对数据进行分割。与之不同，我们提出了一种方法，利用预训练在LLMs中嵌入的知识来构建决策树，而不需要训练数据。我们的方法通过利用LLMs执行决策树构建所需的基本操作，如属性离散化、概率计算和基于概率的基尼指数计算，来实现这一点。我们展示了这些零样本决策树不仅可以超越基线的零样本方法，还能在表格数据集上与监督驱动的决策树竞争。通过这种方法构建的决策树提供了透明且可解释的模型，在缓解数据稀缺性的同时保持了可解释性。这项工作在低数据机器学习领域建立了一个新的基准，提供了一种以知识为导向、而非数据驱动的方式构建决策树的方法。 

---
# Phase Transitions in Large Language Models and the $O(N)$ Model 

**Title (ZH)**: 大型语言模型中的相变现象与$O(N)$模型 

**Authors**: Youran Sun, Babak Haghighat  

**Link**: [PDF](https://arxiv.org/pdf/2501.16241)  

**Abstract**: Large language models (LLMs) exhibit unprecedentedly rich scaling behaviors. In physics, scaling behavior is closely related to phase transitions, critical phenomena, and field theory. To investigate the phase transition phenomena in LLMs, we reformulated the Transformer architecture as an $O(N)$ model. Our study reveals two distinct phase transitions corresponding to the temperature used in text generation and the model's parameter size, respectively. The first phase transition enables us to estimate the internal dimension of the model, while the second phase transition is of \textit{higher-depth} and signals the emergence of new capabilities. As an application, the energy of the $O(N)$ model can be used to evaluate whether an LLM's parameters are sufficient to learn the training data. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展现出前所未有的丰富标度行为。在物理学中，标度行为与相变、临界现象和场论密切相关。为了研究LLMs中的相变现象，我们将Transformer架构重新表述为$O(N)$模型。我们的研究揭示了与文本生成中使用的温度和模型参数量对应的两种不同的相变现象。第一个相变使我们能够估算模型的内部维度，而第二个相变是更高层次的，并标志着新能力的涌现。作为应用，$O(N)$模型的能量可以用于评估LLM的参数是否足以学习训练数据。 

---
# Enhancing and Exploring Mild Cognitive Impairment Detection with W2V-BERT-2.0 

**Title (ZH)**: 增强并探索基于W2V-BERT-2.0的轻度认知障碍检测方法 

**Authors**: Yueguan Wang, Tatsunari Matsushima, Soichiro Matsushima, Toshimitsu Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2501.16201)  

**Abstract**: This study explores a multi-lingual audio self-supervised learning model for detecting mild cognitive impairment (MCI) using the TAUKADIAL cross-lingual dataset. While speech transcription-based detection with BERT models is effective, limitations exist due to a lack of transcriptions and temporal information. To address these issues, the study utilizes features directly from speech utterances with W2V-BERT-2.0. We propose a visualization method to detect essential layers of the model for MCI classification and design a specific inference logic considering the characteristics of MCI. The experiment shows competitive results, and the proposed inference logic significantly contributes to the improvements from the baseline. We also conduct detailed analysis which reveals the challenges related to speaker bias in the features and the sensitivity of MCI classification accuracy to the data split, providing valuable insights for future research. 

**Abstract (ZH)**: 本文探讨了一种多语言音频自我监督学习模型，利用TAUKADIAL跨语言数据集来检测轻度认知障碍（MCI）。尽管基于语音转录的检测方法（如使用BERT模型）有效，但缺乏转录和时间信息是其局限性之一。为了解决这些问题，本研究利用W2V-BERT-2.0 直接从语音短语中提取特征。我们提出了一种可视化方法来检测模型用于MCI分类的关键层，并针对MCI的特征设计了一种特定的推理逻辑。实验结果表明，这种方法具有竞争力，提出的推理逻辑对基线的改进起到了显著作用。此外，我们还进行了详细的分析，揭示了特征中的说话者偏见挑战以及MCI分类准确性对数据划分的敏感性，为未来的研究提供了宝贵的启示。 

---
# Challenging Assumptions in Learning Generic Text Style Embeddings 

**Title (ZH)**: 挑战通用文本风格嵌入中的假设 

**Authors**: Phil Ostheimer, Marius Kloft, Sophie Fellenz  

**Link**: [PDF](https://arxiv.org/pdf/2501.16073)  

**Abstract**: Recent advancements in language representation learning primarily emphasize language modeling for deriving meaningful representations, often neglecting style-specific considerations. This study addresses this gap by creating generic, sentence-level style embeddings crucial for style-centric tasks. Our approach is grounded on the premise that low-level text style changes can compose any high-level style. We hypothesize that applying this concept to representation learning enables the development of versatile text style embeddings. By fine-tuning a general-purpose text encoder using contrastive learning and standard cross-entropy loss, we aim to capture these low-level style shifts, anticipating that they offer insights applicable to high-level text styles. The outcomes prompt us to reconsider the underlying assumptions as the results do not always show that the learned style representations capture high-level text styles. 

**Abstract (ZH)**: 近期在语言表示学习方面的进展主要强调了语言建模以提取有意义的表示，往往忽视了特定风格的考虑。本研究通过创建对于以风格为中心的任务至关重要的通用句级风格嵌入，来弥补这一空白。我们的方法基于低级文本风格变化可以构成任何高级风格这一前提。我们假设将这一概念应用于表示学习能够促进多功能文本风格嵌入的发展。通过使用对比学习和标准交叉熵损失微调一个通用语言编码器，我们旨在捕捉这些低级风格变化，期待它们能够提供适用于高级文本风格的见解。研究结果促使我们重新审视这些假设，因为实验结果并不总是表明学习到的风格表示能够捕捉到高级文本风格。 

---
# Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation 

**Title (ZH)**: 埃米利亚：一个大规模、多样化的多语言语音生成数据集 

**Authors**: Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15907)  

**Abstract**: Recent advancements in speech generation have been driven by the large-scale training datasets. However, current models fall short of capturing the spontaneity and variability inherent in real-world human speech, due to their reliance on audiobook datasets limited to formal read-aloud speech styles. To bridge this gap, we introduce Emilia-Pipe, an open-source preprocessing pipeline to extract high-quality training data from valuable yet underexplored in-the-wild data that capture spontaneous human speech in real-world contexts. By leveraging Emilia-Pipe, we construct Emilia, the first multilingual speech generation dataset derived from in-the-wild speech data. This dataset comprises over 101k hours of speech across six languages: English, Chinese, German, French, Japanese, and Korean. Besides, we expand Emilia to Emilia-Large, a dataset exceeding 216k hours, making it the largest open-source speech generation dataset available. Extensive experiments demonstrate that Emilia significantly outperforms traditional audiobook datasets in generating spontaneous and human-like speech, showcasing superior performance in capturing diverse speaker timbre and speaking styles of real-world human speech. Furthermore, this work underscores the importance of scaling dataset size to advance speech generation research and validates the effectiveness of Emilia for both multilingual and crosslingual speech generation. 

**Abstract (ZH)**: 近年来，语音生成的进步主要得益于大规模训练数据集的应用。然而，当前的模型在捕捉真实世界人类语音中的自发性和变异性方面仍存在不足，这主要是因为它们依赖于仅限于正式朗读风格的有声书数据集。为了解决这一问题，我们引入了Emilia-Pipe，一种开源预处理流水线，用于从有价值的但尚未充分利用的自然环境中提取高质量的训练数据，这些数据捕捉到了真实世界情境下的自发人类语音。利用Emilia-Pipe，我们构建了Emilia，这是第一个基于自然语音数据的多语言语音生成数据集。该数据集包含超过101,000小时的语音数据，涵盖了英语、中文、德语、法语、日语和韩语六种语言。此外，我们进一步扩展了Emilia，创建了Emilia-Large，这个数据集包含超过216,000小时的语音数据，使其成为目前可获取的最大规模的开源语音生成数据集。广泛实验表明，Emilia在生成自发性和人性化的语音方面远超传统有声书数据集，在捕捉不同说话者音色和真实世界人类语音的多种说话风格方面表现更为出色。此外，这项工作强调了扩大数据集规模对于推进语音生成研究的重要性，并验证了Emilia在多语言和跨语言语音生成方面的有效性。 

---
# Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data? 

**Title (ZH)**: Transformer模型能够在训练数据中将分离的知识进行关联并进行推理吗？ 

**Authors**: Yutong Yin, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15857)  

**Abstract**: Humans exhibit remarkable compositional reasoning by integrating knowledge from various sources. For example, if someone learns ( B = f(A) ) from one source and ( C = g(B) ) from another, they can deduce ( C=g(B)=g(f(A)) ) even without encountering ( ABC ) together, showcasing the generalization ability of human intelligence. In this paper, we introduce a synthetic learning task, "FTCT" (Fragmented at Training, Chained at Testing), to validate the potential of Transformers in replicating this skill and interpret its inner mechanism. In the training phase, data consist of separated knowledge fragments from an overall causal graph. During testing, Transformers must infer complete causal graph traces by integrating these fragments. Our findings demonstrate that few-shot Chain-of-Thought prompting enables Transformers to perform compositional reasoning on FTCT by revealing correct combinations of fragments, even if such combinations were absent in the training data. Furthermore, the emergence of compositional reasoning ability is strongly correlated with the model complexity and training-testing data similarity. We propose, both theoretically and empirically, that Transformers learn an underlying generalizable program from training, enabling effective compositional reasoning during testing. 

**Abstract (ZH)**: 人类展示了通过整合来自各种来源的知识进行组合推理的非凡能力。例如，如果某人从一个来源得知(B = f(A))，从另一个来源得知(C = g(B))，他们甚至可以在未同时遇到(ABC)的情况下，推导出(C = g(B) = g(f(A)))，这彰显了人类智能的泛化能力。在这项研究中，我们引入了一个合成学习任务，称为“FTCT”（Fragmented at Training, Chained at Testing，训练时碎片化，测试时串联），以验证Transformer在复制这种能力的潜力，并解释其内部机制。在训练阶段，数据由整体因果图中的分离知识碎片组成。在测试阶段，Transformer必须通过整合这些片段推断完整的因果图路径。我们的研究发现，少量示例的链式推理提示能够使Transformer在FTCT任务中进行组合推理，通过揭示正确的片段组合，即使在训练数据中没有出现这些组合。此外，组合推理能力的出现与模型复杂度以及训练-测试数据相似性密切相关。我们从理论上和实验上提出，Transformer通过训练学习到一种基础可泛化的程序，在测试阶段能够有效进行组合推理。 

---
# LemmaHead: RAG Assisted Proof Generation Using Large Language Models 

**Title (ZH)**: LemmaHead：使用大语言模型的RAG辅助定理证明 

**Authors**: Tianbo Yang, Mingqi Yang, Hongyi Zhao, Tianshuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15797)  

**Abstract**: Developing the logic necessary to solve mathematical problems or write mathematical proofs is one of the more difficult objectives for large language models (LLMS). Currently, the most popular methods in literature consists of fine-tuning the model on written mathematical content such as academic publications and textbooks, so that the model can learn to emulate the style of mathematical writing. In this project, we explore the effectiveness of using retrieval augmented generation (RAG) to address gaps in the mathematical reasoning of LLMs. We develop LemmaHead, a RAG knowledge base that supplements queries to the model with relevant mathematical context, with particular focus on context from published textbooks. To measure our model's performance in mathematical reasoning, our testing paradigm focuses on the task of automated theorem proving via generating proofs to a given mathematical claim in the Lean formal language. 

**Abstract (ZH)**: 开发解决数学问题或撰写数学证明所需的逻辑是大型语言模型（LLMs）面临的更加困难的目标之一。目前，文献中最流行的 方法是通过针对学术出版物和教科书中的数学内容进行微调，使模型能够学习模仿数学写作的风格。在这个项目中，我们探索使用检索增强生成（RAG）来弥补LLMs在数学推理方面的不足。我们开发了LemmaHead，这是一种RAG知识库，通过向模型提供相关数学上下文（特别是来自已出版教科书的上下文）来补充查询。为了衡量模型在数学推理方面的表现，我们测试范式关注的是通过生成给定数学命题在Lean形式语言中的证明来进行自动定理证明的任务。 

---
# Risk-Aware Distributional Intervention Policies for Language Models 

**Title (ZH)**: 面向风险的分布干预策略的语言模型 

**Authors**: Bao Nguyen, Binh Nguyen, Duy Nguyen, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15758)  

**Abstract**: Language models are prone to occasionally undesirable generations, such as harmful or toxic content, despite their impressive capability to produce texts that appear accurate and coherent. This paper presents a new two-stage approach to detect and mitigate undesirable content generations by rectifying activations. First, we train an ensemble of layerwise classifiers to detect undesirable content using activations by minimizing a smooth surrogate of the risk-aware score. Then, for contents that are detected as undesirable, we propose layerwise distributional intervention policies that perturb the attention heads minimally while guaranteeing probabilistically the effectiveness of the intervention. Benchmarks on several language models and datasets show that our method outperforms baselines in reducing the generation of undesirable output. 

**Abstract (ZH)**: 尽管语言模型在生成看似准确和连贯的文本方面表现出色，但它们仍然容易产生不希望出现的内容，如有害或有毒内容。本文提出了一种新的两阶段方法，通过纠正激活值来检测和减轻不良内容生成。首先，我们训练了一组逐层分类器，使用激活值检测不良内容，通过最小化风险意识分数的光滑近似值来实现。然后，对于被检测为不良内容的文本，我们提出了逐层分布干预策略，这些策略在尽量不影响注意力头的情况下，保证干预的有效性。在多种语言模型和数据集上的基准测试表明，我们的方法在减少不良输出的生成方面优于基线方法。 

---
# Beyond Benchmarks: On The False Promise of AI Regulation 

**Title (ZH)**: 超越基准：关于AI监管的虚假承诺 

**Authors**: Gabriel Stanovsky, Renana Keydar, Gadi Perl, Eliya Habba  

**Link**: [PDF](https://arxiv.org/pdf/2501.15693)  

**Abstract**: The rapid advancement of artificial intelligence (AI) systems in critical domains like healthcare, justice, and social services has sparked numerous regulatory initiatives aimed at ensuring their safe deployment. Current regulatory frameworks, exemplified by recent US and EU efforts, primarily focus on procedural guidelines while presuming that scientific benchmarking can effectively validate AI safety, similar to how crash tests verify vehicle safety or clinical trials validate drug efficacy. However, this approach fundamentally misunderstands the unique technical challenges posed by modern AI systems. Through systematic analysis of successful technology regulation case studies, we demonstrate that effective scientific regulation requires a causal theory linking observable test outcomes to future performance - for instance, how a vehicle's crash resistance at one speed predicts its safety at lower speeds. We show that deep learning models, which learn complex statistical patterns from training data without explicit causal mechanisms, preclude such guarantees. This limitation renders traditional regulatory approaches inadequate for ensuring AI safety. Moving forward, we call for regulators to reckon with this limitation, and propose a preliminary two-tiered regulatory framework that acknowledges these constraints: mandating human oversight for high-risk applications while developing appropriate risk communication strategies for lower-risk uses. Our findings highlight the urgent need to reconsider fundamental assumptions in AI regulation and suggest a concrete path forward for policymakers and researchers. 

**Abstract (ZH)**: 人工智能（AI）系统在医疗、司法和社会服务等关键领域取得了快速进步，引发了众多旨在确保其安全部署的监管举措。目前的监管框架，如美国和欧盟的近期努力，主要侧重于程序性指南，同时假定科学基准测试能够有效验证AI系统的安全性，类似于汽车碰撞测试验证车辆安全性或临床试验验证药物疗效的方式。然而，这种做法从根本上未能理解现代AI系统所带来的独特技术挑战。通过对成功的科技监管案例进行全面分析，我们表明有效的科学监管需要因果理论，将可观察的测试结果与未来的性能联系起来，例如，车辆在某一速度下的抗撞能力如何预测其在较低速度下的安全性。我们指出，深度学习模型从训练数据中学习复杂统计模式，而无需明确的因果机制，这使得此类保证无法实现。这一局限性使得传统监管方法对于确保AI系统安全性而言显得不够充分。未来，我们呼吁监管机构正视这一局限性，并提出一个初步的双重监管框架：对于高风险应用强制要求人工监督，而对于低风险应用则制定适当的Risk Communication策略。我们的研究结果强调了重新审视AI监管基本假设的迫切性，并为政策制定者和研究者指明了一条具体的前进道路。 

---
# Blissful (A)Ignorance: People form overly positive impressions of others based on their written messages, despite wide-scale adoption of Generative AI 

**Title (ZH)**: 乐于无知：尽管广泛采用了生成式人工智能，人们仍然基于他人的书面信息形成了过于积极的印象 

**Authors**: Jiaqi Zhu, Andras Molnar  

**Link**: [PDF](https://arxiv.org/pdf/2501.15678)  

**Abstract**: As the use of Generative AI (GenAI) tools becomes more prevalent in interpersonal communication, understanding their impact on social perceptions is crucial. According to signaling theory, GenAI may undermine the credibility of social signals conveyed in writing, since it reduces the cost of writing and makes it hard to verify the authenticity of messages. Using a pre-registered large-scale online experiment (N = 647; Prolific), featuring scenarios in a range of communication contexts (personal vs. professional; close others vs. strangers), we explored how senders' use of GenAI influenced recipients' impressions of senders, both when GenAI use was known or uncertain. Consistent with past work, we found strong negative effects on social impressions when disclosing that a message was AI-generated, compared to when the same message was human-written. However, under the more realistic condition when potential GenAI use was not explicitly highlighted, recipients did not exhibit any skepticism towards senders, and these "uninformed" impressions were virtually indistinguishable from those of fully human-written messages. Even when we highlighted the potential (but uncertain) use of GenAI, recipients formed overly positive impressions. These results are especially striking given that 46% of our sample admitted having used such tools for writing messages, just within the past two weeks. Our findings put past work in a new light: While social judgments can be substantially affected when GenAI use is explicitly disclosed, this information may not be readily available in more realistic communication settings, making recipients blissfully ignorant about others' potential use of GenAI. 

**Abstract (ZH)**: 随着生成式人工智能（GenAI）工具在人际沟通中的使用越来越普遍，了解其对社会认知的影响变得至关重要。根据信号理论，GenAI可能会削弱通过书写传达的社会信号的可信度，因为这降低了书写成本，并且难以验证信息的真实性。通过一项预先注册的大规模在线实验（N = 647；Prolific），我们探讨了在各种沟通场景（个人与职业；亲密他人与陌生人）中，发送者使用GenAI如何影响接收者对发送者的印象，特别是在GenAI使用情况明确或不确定的情况下。与以往研究一致，我们发现，当披露消息是AI生成时，社会印象明显负面，而当消息是人类撰写时，则没有这种影响。然而，在潜在GenAI使用情况没有明确强调的更现实的条件下，接收者并未表现出任何对发送者的怀疑态度，这些“未被告知”的印象与完全由人类撰写的消息几乎没有区别。即使我们强调了潜在但不确定的GenAI使用情况，接收者也形成了过度积极的印象。鉴于我们的样本中46%的人在过去两周内承认曾使用过此类工具进行写作，这些结果尤其引人注目。我们的发现为过去的有关研究提供了新的视角：虽然当explicit地披露GenAI的使用时，社会判断可能会受到重大影响，但在更现实的沟通环境中，这些信息可能不可获取，从而使接收者对他人潜在的GenAI使用一无所知。 

---
# Stepback: Enhanced Disentanglement for Voice Conversion via Multi-Task Learning 

**Title (ZH)**: Stepback: 基于多任务学习的语音转换增强解耦方法 

**Authors**: Qian Yang, Calbert Graham  

**Link**: [PDF](https://arxiv.org/pdf/2501.15613)  

**Abstract**: Voice conversion (VC) modifies voice characteristics while preserving linguistic content. This paper presents the Stepback network, a novel model for converting speaker identity using non-parallel data. Unlike traditional VC methods that rely on parallel data, our approach leverages deep learning techniques to enhance disentanglement completion and linguistic content preservation. The Stepback network incorporates a dual flow of different domain data inputs and uses constraints with self-destructive amendments to optimize the content encoder. Extensive experiments show that our model significantly improves VC performance, reducing training costs while achieving high-quality voice conversion. The Stepback network's design offers a promising solution for advanced voice conversion tasks. 

**Abstract (ZH)**: 语音转换（Voice Conversion, VC）可以修改语音特征同时保留语言内容。本文提出了一种新颖的Stepback网络，该网络用于转换说话人身份，且不依赖于平行数据。与依赖平行数据的传统VC方法不同，我们的方法利用深度学习技术来增强特征分离和语言内容的保留。Stepback网络结合了来自不同领域数据的双向流，并通过自我破坏性的修正来约束内容编码器，从而进行优化。大量实验表明，该模型显著提高了VC性能，同时降低了训练成本并实现了高质量的语音转换。Stepback网络的设计为高级语音转换任务提供了一个有希望的解决方案。 

---
# Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning 

**Title (ZH)**: 重新思考外部慢思考：从雪球错误到正确推理的概率 

**Authors**: Zeyu Gan, Yun Liao, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15602)  

**Abstract**: Test-time scaling, which is also often referred to as \textit{slow-thinking}, has been demonstrated to enhance multi-step reasoning in large language models (LLMs). However, despite its widespread utilization, the mechanisms underlying slow-thinking methods remain poorly understood. This paper explores the mechanisms of external slow-thinking from a theoretical standpoint. We begin by examining the snowball error effect within the LLM reasoning process and connect it to the likelihood of correct reasoning using information theory. Building on this, we show that external slow-thinking methods can be interpreted as strategies to mitigate the error probability. We further provide a comparative analysis of popular external slow-thinking approaches, ranging from simple to complex, highlighting their differences and interrelationships. Our findings suggest that the efficacy of these methods is not primarily determined by the specific framework employed, and that expanding the search scope or the model's internal reasoning capacity may yield more sustained improvements in the long term. We open-source our code at \url{this https URL}. 

**Abstract (ZH)**: 测试时缩放，也经常被称为“慢思考”，已被证明能够增强大型语言模型（LLMs）的多步推理能力。然而，尽管其广泛应用，慢思考方法背后的机制仍然知之甚少。本文从理论角度探讨了外部慢思考的机制。我们首先考察了LLM推理过程中的雪球错误效应，并用信息论将其与正确推理的可能性联系起来。在此基础上，我们表明外部慢思考方法可以被视为降低错误概率的策略。我们进一步对多种流行的外部慢思考方法进行了比较分析，涵盖从简单到复杂的各种方法，突显其差异和相互关系。我们的研究成果表明，这些方法的有效性主要不取决于所采用的具体框架，而是通过扩大搜索范围或增强模型内部推理能力，可以实现更具持续性的改进。我们已将代码开源，地址为：[此链接](this https URL)。 

---
# ConceptCLIP: Towards Trustworthy Medical AI via Concept-Enhanced Contrastive Langauge-Image Pre-training 

**Title (ZH)**: ConceptCLIP：通过概念增强对比语言-图像预训练实现可靠的医疗AI 

**Authors**: Yuxiang Nie, Sunan He, Yequan Bie, Yihui Wang, Zhixuan Chen, Shu Yang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15579)  

**Abstract**: Trustworthiness is essential for the precise and interpretable application of artificial intelligence (AI) in medical imaging. Traditionally, precision and interpretability have been addressed as separate tasks, namely medical image analysis and explainable AI, each developing its own models independently. In this study, for the first time, we investigate the development of a unified medical vision-language pre-training model that can achieve both accurate analysis and interpretable understanding of medical images across various modalities. To build the model, we construct MedConcept-23M, a large-scale dataset comprising 23 million medical image-text pairs extracted from 6.2 million scientific articles, enriched with concepts from the Unified Medical Language System (UMLS). Based on MedConcept-23M, we introduce ConceptCLIP, a medical AI model utilizing concept-enhanced contrastive language-image pre-training. The pre-training of ConceptCLIP involves two primary components: image-text alignment learning (IT-Align) and patch-concept alignment learning (PC-Align). This dual alignment strategy enhances the model's capability to associate specific image regions with relevant concepts, thereby improving both the precision of analysis and the interpretability of the AI system. We conducted extensive experiments on 5 diverse types of medical image analysis tasks, spanning 51 subtasks across 10 image modalities, with the broadest range of downstream tasks. The results demonstrate the effectiveness of the proposed vision-language pre-training model. Further explainability analysis across 6 modalities reveals that ConceptCLIP achieves superior performance, underscoring its robust ability to advance explainable AI in medical imaging. These findings highlight ConceptCLIP's capability in promoting trustworthy AI in the field of medicine. 

**Abstract (ZH)**: 信任是确保人工智能（AI）在医学成像中的精确和可解释应用的关键。传统上，精度和可解释性分别作为一个独立的任务来处理，即医学图像分析和可解释AI，各自独立开发模型。在本研究中，我们首次探讨了一种统一的医学视觉-语言预训练模型的发展，该模型可以在多种模态下实现医学图像的准确分析和可解释理解。为了构建该模型，我们创建了一个大规模数据集MedConcept-23M，其中包含2300万张医学图像-文本对，这些对是从620万篇科学文章中提取出来的，并丰富了统一医学语言系统（UMLS）中的概念。基于MedConcept-23M，我们引入了ConceptCLIP，这是一种利用概念增强的对比视觉-语言预训练的医学AI模型。ConceptCLIP的预训练包括两个主要组成部分：图像-文本对齐学习（IT-Align）和片段-概念对齐学习（PC-Align）。这种双重对齐策略增强了模型将特定图像区域与相关概念关联的能力，从而提高了分析的精度和AI系统的可解释性。我们对5种不同的医学图像分析任务进行了广泛的实验，涵盖了10种图像模态下的51个子任务，这是迄今为止下游任务范围最广泛的研究。结果表明了所提出视觉-语言预训练模型的有效性。进一步在6种模态上的可解释性分析显示，ConceptCLIP表现更加优越，突显了其在医学成像中推动可解释AI发展的稳健能力。这些发现强调了ConceptCLIP在促进医学领域可信AI方面的能力。 

---
# Commute Your Domains: Trajectory Optimality Criterion for Multi-Domain Learning 

**Title (ZH)**: 通勤你的领域：多领域学习的轨迹最优性标准 

**Authors**: Alexey Rukhovich, Alexander Podolskiy, Irina Piontkovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2501.15556)  

**Abstract**: In multi-domain learning, a single model is trained on diverse data domains to leverage shared knowledge and improve generalization. The order in which the data from these domains is used for training can significantly affect the model's performance on each domain. However, this dependence is under-studied. In this paper, we investigate the influence of training order (or data mixing) in multi-domain learning using the concept of Lie bracket of gradient vector fields. By analyzing the infinitesimal effects of changing the training order, we identify regions in the parameter space where altering the order between two training domains can benefit the target loss. We validate the predictions of our theoretical framework on the influence of training order (or data mixing) both on a toy example and bilingual LLM pre-training. 

**Abstract (ZH)**: 在多域学习中，单个模型通过在多种数据域上进行训练，可以利用共享知识并改善泛化能力。这些数据域的训练顺序会对模型在每个域上的性能产生显著影响。然而，这一点尚未得到充分研究。在本文中，我们利用Lie括号的概念研究了多域学习中训练顺序（或数据混杂）的影响。通过分析改变训练顺序的微小效果，我们确定了参数空间中的某些区域，在这些区域内，改变两个训练域之间的顺序可以降低目标损失。我们分别在玩具示例和双语LLM预训练中验证了我们理论框架对训练顺序（或数据混杂）影响的预测。 

---
# Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values? 

**Title (ZH)**: 注意价值与行动之间的差距：大语言模型的行动是否与其价值观一致？ 

**Authors**: Hua Shen, Nicholas Clark, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15463)  

**Abstract**: Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps. 

**Abstract (ZH)**: 现有的研究主要通过评估大型语言模型（LLM）对特定价值观的倾向来评价其价值。然而，“价值-行动差距”现象揭示了个体在现实世界情境中所声明的价值与其行为之间的差异，这一现象根植于环境和社会心理学领域。在多大程度上，LLM在受到其声明的价值影响后所表现出的行为与其声明的价值之间存在类似的差距？本研究引入了ValueActionLens，这是一种评估LLM声明的价值与其价值驱动行为之间对齐程度的评价框架。该框架包括生成包含14,800个跨十二种文化与十一种社会议题的价值驱动行为的数据集，并设立两个任务来评估LLM声明的价值倾向与其价值驱动行为在三种不同对齐度量标准下的对齐程度。广泛的实验表明，LLM声明的价值与行为之间的对齐程度并不理想，且在不同场景和模型中存在显著差异。对对齐不当结果的分析揭示了某些价值-行动差距可能带来的潜在危害。为预测价值-行动差距，我们还发现利用推理解释可以提高预测性能。这些发现强调了仅依赖LLM声明的价值来预测其行为的风险，并强调了对LLM的价值及其价值-行动差距进行情境意识评估的重要性。 

---
# The Potential of Large Language Models in Supply Chain Management: Advancing Decision-Making, Efficiency, and Innovation 

**Title (ZH)**: 大型语言模型在供应链管理中的潜力：推动决策、效率和创新 

**Authors**: Raha Aghaei, Ali A. Kiaei, Mahnaz Boush, Javad Vahidi, Zeynab Barzegar, Mahan Rofoosheh  

**Link**: [PDF](https://arxiv.org/pdf/2501.15411)  

**Abstract**: The integration of large language models (LLMs) into supply chain management (SCM) is revolutionizing the industry by improving decision-making, predictive analytics, and operational efficiency. This white paper explores the transformative impact of LLMs on various SCM functions, including demand forecasting, inventory management, supplier relationship management, and logistics optimization. By leveraging advanced data analytics and real-time insights, LLMs enable organizations to optimize resources, reduce costs, and improve responsiveness to market changes. Key findings highlight the benefits of integrating LLMs with emerging technologies such as IoT, blockchain, and robotics, which together create smarter and more autonomous supply chains. Ethical considerations, including bias mitigation and data protection, are taken into account to ensure fair and transparent AI practices. In addition, the paper discusses the need to educate the workforce on how to manage new AI-driven processes and the long-term strategic benefits of adopting LLMs. Strategic recommendations for SCM professionals include investing in high-quality data management, promoting cross-functional collaboration, and aligning LLM initiatives with overall business goals. The findings highlight the potential of LLMs to drive innovation, sustainability, and competitive advantage in the ever-changing supply chain management landscape. 

**Abstract (ZH)**: 大型语言模型（LLMs）在供应链管理（SCM）中的整合正在通过改善决策制定、预测分析和运营效率，颠覆整个行业。本白皮书探讨了LLMs对各种SCM功能的革新影响，包括需求预测、库存管理、供应商关系管理和物流优化。通过利用先进的数据分析和实时洞察，LLMs使组织能够优化资源、降低运营成本，并提高对市场变化的响应速度。关键发现指出，将LLMs与物联网、区块链和机器人等新兴技术整合，有助于构建更智能、更能自主运作的供应链。同时，本白皮书还关注伦理考量，包括偏见缓解和数据保护，以确保公平透明的AI实践。此外，本白皮书还讨论了需要对劳动力进行培训，使其能够管理新的AI驱动流程，以及采用LLMs的长期战略优势。对于SCM专业人士的战略建议包括：投资高质量数据管理、促进跨部门协作，并将LLMs项目与整体业务目标相协调。研究结果强调了LLMs在不断变化的供应链管理景观中推动创新、可持续性和竞争优势的潜在价值。 

---
# Diffusion-based Hierarchical Negative Sampling for Multimodal Knowledge Graph Completion 

**Title (ZH)**: 基于扩散的分层负样本生成多模态知识图嵌入完成方法 

**Authors**: Guanglin Niu, Xiaowei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15393)  

**Abstract**: Multimodal Knowledge Graph Completion (MMKGC) aims to address the critical issue of missing knowledge in multimodal knowledge graphs (MMKGs) for their better applications. However, both the previous MMGKC and negative sampling (NS) approaches ignore the employment of multimodal information to generate diverse and high-quality negative triples from various semantic levels and hardness levels, thereby limiting the effectiveness of training MMKGC models. Thus, we propose a novel Diffusion-based Hierarchical Negative Sampling (DHNS) scheme tailored for MMKGC tasks, which tackles the challenge of generating high-quality negative triples by leveraging a Diffusion-based Hierarchical Embedding Generation (DiffHEG) that progressively conditions on entities and relations as well as multimodal semantics. Furthermore, we develop a Negative Triple-Adaptive Training (NTAT) strategy that dynamically adjusts training margins associated with the hardness level of the synthesized negative triples, facilitating a more robust and effective learning procedure to distinguish between positive and negative triples. Extensive experiments on three MMKGC benchmark datasets demonstrate that our framework outperforms several state-of-the-art MMKGC models and negative sampling techniques, illustrating the effectiveness of our DHNS for training MMKGC models. The source codes and datasets of this paper are available at this https URL. 

**Abstract (ZH)**: 多模态知识图填充（MMKGC）旨在解决多模态知识图（MMKG）中缺失知识的关键问题，以便更好地应用这些知识图。然而，先前的MMKGC方法和负采样（NS）方法忽视了利用多模态信息从各个语义层次和难度层次生成多样且高质量的负三元组，从而限制了MMKGC模型训练的有效性。因此，我们提出了一种新的基于扩散的层次负采样（DHNS）方案，专门适用于MMKGC任务。该方案通过利用一种基于扩散的层次嵌入生成（DiffHEG）方法，逐步条件化实体、关系以及多模态语义来应对生成高质量负三元组的挑战。此外，我们开发了一种负三元组自适应训练（NTAT）策略，该策略动态调整与合成负三元组难度水平相关的训练边缘，以促进更稳健和有效的学习过程，以便区分正三元组和负三元组。在三个MMKGC基准数据集上的广泛实验表明，我们的框架在多个最先进的MMKGC模型和负采样技术中表现更优，证明了我们提出的DHNS在训练MMKGC模型方面的有效性。本文的源代码和数据集可在以下链接获取：[此网址]。 

---
# ToMoE: Converting Dense Large Language Models to Mixture-of-Experts through Dynamic Structural Pruning 

**Title (ZH)**: ToMoE: 将稠密大型语言模型转换为混合专家模型的动态结构剪枝方法 

**Authors**: Shangqian Gao, Ting Hua, Reza Shirkavand, Chi-Heng Lin, Zhen Tang, Zhengao Li, Longge Yuan, Fangyi Li, Zeyu Zhang, Alireza Ganjdanesh, Lou Qian, Xu Jie, Yen-Chang Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15316)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable abilities in tackling a wide range of complex tasks. However, their huge computational and memory costs raise significant challenges in deploying these models on resource-constrained devices or efficiently serving them. Prior approaches have attempted to alleviate these problems by permanently removing less important model structures, yet these methods often result in substantial performance degradation due to the permanent deletion of model parameters. In this work, we tried to mitigate this issue by reducing the number of active parameters without permanently removing them. Specifically, we introduce a differentiable dynamic pruning method that pushes dense models to maintain a fixed number of active parameters by converting their MLP layers into a Mixture of Experts (MoE) architecture. Our method, even without fine-tuning, consistently outperforms previous structural pruning techniques across diverse model families, including Phi-2, LLaMA-2, LLaMA-3, and Qwen-2.5. 

**Abstract (ZH)**: 大语言模型（LLMs）在应对各种复杂的任务方面展现了显著的能力。然而，这些模型巨大的计算和内存成本在将其部署在资源受限的设备上或高效地服务于这些模型时提出了重大挑战。先前的方法试图通过永久移除不重要的模型结构来缓解这些问题，但这些方法常常由于永久删除模型参数而导致性能显著下降。在本工作中，我们尝试通过减少活跃参数的数量而无需永久移除参数来缓解这一问题。具体而言，我们提出了一种可微分的动态剪枝方法，将密集模型中的MLP层转换为专家混合（Mixture of Experts，MoE）架构，以保持固定的活跃参数数量。即使在无需微调的情况下，我们的方法在包括Phi-2、LLaMA-2、LLaMA-3和Qwen-2.5等不同模型家族的多种应用中，也持续优于先前的结构剪枝技术。 

---
# Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models 

**Title (ZH)**: 分析并增强细粒度视觉识别在多模态大语言模型中的能力 

**Authors**: Hulingxiao He, Geng Li, Zijun Geng, Jinglin Xu, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15140)  

**Abstract**: Multi-modal large language models (MLLMs) have shown remarkable abilities in various visual understanding tasks. However, MLLMs still struggle with fine-grained visual recognition (FGVR), which aims to identify subordinate-level categories from images. This can negatively impact more advanced capabilities of MLLMs, such as object-centric visual question answering and reasoning. In our study, we revisit three quintessential capabilities of MLLMs for FGVR, including object information extraction, category knowledge reserve, object-category alignment, and position of the root cause as a misalignment problem. To address this issue, we present Finedefics, an MLLM that enhances the model's FGVR capability by incorporating informative attribute descriptions of objects into the training phase. We employ contrastive learning on object-attribute pairs and attribute-category pairs simultaneously and use examples from similar but incorrect categories as hard negatives, naturally bringing representations of visual objects and category names closer. Extensive evaluations across multiple popular FGVR datasets demonstrate that Finedefics outperforms existing MLLMs of comparable parameter sizes, showcasing its remarkable efficacy. The code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在各种视觉理解任务中展现出卓越的能力。然而，MLLMs 在细粒度视觉识别（FGVR）方面仍然存在挑战，FGVR 的目标是从图像中识别从属类别。这可能会影响 MLLMs 更高级的能力，例如以对象为中心的视觉问答和推理。在我们的研究中，我们重新审视了 MLLMs 在 FGVR 方面的三种核心能力，包括对象信息提取、类别知识储备、对象与类别的对齐，以及将其视为对齐不匹配问题的根本原因。为了解决这一问题，我们提出了 Finedefics，这是一种通过在训练阶段整合对象的 informative 属性描述来提升模型 FGVR 能力的 MLLM。我们同时在对象-属性对和属性-类别对上采用对比学习，并使用来自相似但不正确的类别的例子作为硬负样本，自然地使视觉对象的表示形式和类别名称更接近。在多个流行 FGVR 数据集上的广泛评估表明，Finedefics 在与现有大小相当的 MLLMs 中表现出色，展现了其卓越的效果。源代码可通过以下链接获取：this https URL。 

---
# Feedback-Aware Monte Carlo Tree Search for Efficient Information Seeking in Goal-Oriented Conversations 

**Title (ZH)**: 面向反馈的蒙特卡洛树搜索方法：在目标导向对话中高效的信息查找 

**Authors**: Harshita Chopra, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2501.15056)  

**Abstract**: The ability to identify and acquire missing information is a critical component of effective decision making and problem solving. With the rise of conversational artificial intelligence (AI) systems, strategically formulating information-seeking questions becomes crucial and demands efficient methods to guide the search process. We introduce a novel approach to adaptive question-asking through a combination of Large Language Models (LLM) for generating questions that maximize information gain, Monte Carlo Tree Search (MCTS) for constructing and leveraging a decision tree across multiple samples, and a hierarchical feedback mechanism to learn from past interactions. We present two key innovations: (1) an adaptive MCTS algorithm that balances exploration and exploitation for efficient search over potential questions; and (2) a clustering-based feedback algorithm that leverages prior experience to guide future interactions. Each incoming sample is assigned to a cluster based on its semantic similarity with previously observed samples. Our UCT (Upper Confidence bound for Trees) formulation selects optimal questions by combining expected rewards, a function of information gain, with a cluster-specific bonus that decays with depth, to emphasize the importance of early-stage questions that have proven effective for narrowing the solution space in similar samples. Experiments across three domains, including medical diagnosis and troubleshooting, demonstrate that our method leads to an average of 12% improvement in success rates and a 10x reduction in the average number of LLM calls made per conversation for the search process, in comparison to the state of the art. 

**Abstract (ZH)**: 识别和获取缺失信息的能力是有效决策和问题解决中的关键因素。随着对话型人工智能（AI）系统的兴起，制定策略性的问题查询变得至关重要，并且需要高效的方法来指导搜索过程。我们提出了一种新的自适应问题询问方法，结合了大型语言模型（LLM）生成能最大化信息增益的问题、蒙特卡洛树搜索（MCTS）构建和利用跨越多个样本的决策树，以及层次反馈机制以从过去的交互中学习。我们提出了两个关键创新点：（1）一种自适应MCTS算法，平衡探索和利用，以实现高效的潜在问题搜索；和（2）一种基于聚类的反馈算法，利用先前的经验来引导未来的交互。每个新样本根据其与先前观察样本的语义相似性被分配到一个聚类中。我们的UCT（树的上置信界）公式通过结合期望奖励与特定于聚类的随深度衰减的奖励，来优选问题，从而强调那些在相似样本中已被证明对缩减解空间早期阶段问题的重要性。在医疗诊断和故障排除等三个领域中的实验表明，与现有技术相比，我们的方法在成功率上平均提高了12%，并将搜索过程中每轮对话所需的LLM调用次数减少了10倍。 

---
# OptiSeq: Optimizing Example Ordering for In-Context Learning 

**Title (ZH)**: OptiSeq：优化示例顺序以实现情境学习 

**Authors**: Rahul Atul Bhope, Praveen Venkateswaran, K. R. Jayaram, Vatche Isahagian, Vinod Muthusamy, Nalini Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2501.15030)  

**Abstract**: Developers using LLMs in their applications and agents have provided plenty of anecdotal evidence that in-context-learning (ICL) is fragile. In addition to the quantity and quality of examples, we show that the order in which the in-context examples are listed in the prompt affects the output of the LLM and, consequently, their performance. In this paper, we present OptiSeq, which introduces a score based on log probabilities of LLM outputs to prune the universe of possible example orderings in few-shot ICL and recommend the best order(s) by distinguishing between correct and incorrect outputs resulting from different order permutations. Through a detailed empirical evaluation on multiple LLMs, datasets and prompts, we demonstrate that OptiSeq improves accuracy by 6 - 10.5 percentage points across multiple tasks. 

**Abstract (ZH)**: 将该论文内容或标题翻译成中文，符合学术规范后，可以表述为：

开发人员在他们的应用程序和代理中使用大语言模型（LLMs）时提供了大量 anecdotal 证据，表明情境学习（ICL）是脆弱的。除了在提示中提供的示例数量和质量外，我们还表明，在提示中列出的情境示例的顺序也会影响LLM的输出，进而影响其性能。在本文中，我们提出了 OptiSeq，该方法引入了一个基于LLM输出对数概率的评分系统，用于在少样本ICL中修剪可能的示例排序，并通过区分不同排列顺序产生的正确和错误输出来推荐最佳排序。通过对多个LLM、数据集和提示进行详细的实证评估，我们证明了OptiSeq在多个任务中的准确率提高了6-10.5个百分点。 

---
# LLM4DistReconfig: A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration 

**Title (ZH)**: LLM4DistReconfig：一个用于电力配电网重构的微调大型语言模型 

**Authors**: Panayiotis Christou, Md. Zahidul Islam, Yuzhang Lin, Jingwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.14960)  

**Abstract**: Power distribution networks are evolving due to the integration of DERs and increased customer participation. To maintain optimal operation, minimize losses, and meet varying load demands, frequent network reconfiguration is necessary. Traditionally, the reconfiguration task relies on optimization software and expert operators, but as systems grow more complex, faster and more adaptive solutions are required without expert intervention. Data-driven reconfiguration is gaining traction for its accuracy, speed, and robustness against incomplete network data. LLMs, with their ability to capture complex patterns, offer a promising approach for efficient and responsive network reconfiguration in evolving complex power networks.
In this work, we introduce LLM4DistReconfig, a deep learning-based approach utilizing a fine-tuned LLM to solve the distribution network reconfiguration problem. By carefully crafting prompts and designing a custom loss function, we train the LLM with inputs representing network parameters such as buses, available lines, open lines, node voltages, and system loss. The model then predicts optimal reconfigurations by outputting updated network configurations that minimize system loss while meeting operational constraints. Our approach significantly reduces inference time compared to classical algorithms, allowing for near real-time optimal reconfiguration after training. Experimental results show that our method generates optimal configurations minimizing system loss for five individual and a combined test dataset. It also produces minimal invalid edges, no cycles, or subgraphs across all datasets, fulfilling domain-specific needs. Additionally, the generated responses contain less than 5% improper outputs on seen networks and satisfactory results on unseen networks, demonstrating its effectiveness and reliability for the reconfiguration task. 

**Abstract (ZH)**: 分布式网络由于分布式能源（DERs）的整合和客户参与度的增加而不断发展。为了保持最优运行状态、减少损耗并满足不断变化的负荷需求，频繁的网络重构是必要的。传统上，重构任务依赖于优化软件和专家操作员，但随着系统复杂性的增加，需要更加快速和适应性强的解决方案，无需专家干预。基于数据驱动的重构因其准确性、速度以及对不完整网络数据的稳健性而受到青睐。大规模语言模型（LLMs），由于其捕捉复杂模式的能力，为复杂电力网络中高效和及时的网络重构提供了一种有前途的方法。

在本工作中，我们提出了LLM4DistReconfig，一种基于深度学习的方法，利用微调后的LLM来解决分布网络重构问题。通过精心设计提示并设计自定义损失函数，我们训练LLM以网络参数如母线、可用线路、开放线路、节点电压和系统损耗等作为输入。该模型通过输出更新的网络配置来预测最优的重构方案，这些配置在满足操作约束的同时最小化系统损耗。与传统算法相比，我们的方法显著减少了推理时间，使得经过训练后能够实现接近实时的最优重构。实验结果表明，我们的方法能够针对单个测试集和组合测试集生成最优配置，同时最小化系统损耗。此外，所有数据集生成的重构中没有出现无效边缘、自循环或子图，满足了特定领域的需要。此外，生成的响应在已见过的网络中不含有超过5%的不当输出，在未见过的网络中也取得了令人满意的结果，这证明了其在重构任务中的有效性和可靠性。 

---
# E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions 

**Title (ZH)**: E-Gen：利用E-图提高符号表达式连续表示的效率 

**Authors**: Hongbo Zheng, Suyuan Wang, Neeraj Gangwar, Nickvash Kani  

**Link**: [PDF](https://arxiv.org/pdf/2501.14951)  

**Abstract**: As vector representations have been pivotal in advancing natural language processing (NLP), some prior research has concentrated on creating embedding techniques for mathematical expressions by leveraging mathematically equivalent expressions. While effective, these methods are limited by the training data. In this work, we propose augmenting prior algorithms with larger synthetic dataset, using a novel e-graph-based generation scheme. This new mathematical dataset generation scheme, E-Gen, improves upon prior dataset-generation schemes that are limited in size and operator types. We use this dataset to compare embedding models trained with two methods: (1) training the model to generate mathematically equivalent expressions, and (2) training the model using contrastive learning to group mathematically equivalent expressions explicitly. We evaluate the embeddings generated by these methods against prior work on both in-distribution and out-of-distribution language processing tasks. Finally, we compare the performance of our embedding scheme against state-of-the-art large language models and demonstrate that embedding-based language processing methods perform better than LLMs on several tasks, demonstrating the necessity of optimizing embedding methods for the mathematical data modality. 

**Abstract (ZH)**: 自从向量表示在自然语言处理(NLP)领域取得进展以来，某些前期研究集中于通过利用数学等价表达式来创建数学表达式的嵌入技术。虽然这些方法非常有效，但它们受限于训练数据。在这项工作中，我们提出了一种利用新型e-图为基础的生成方案，将先前的算法与更大的合成数据集相结合。这种新的数学数据生成方案E-Gen，在生成的数据集规模和操作符类型方面，优于以往受限于规模和操作符类型的生成方案。我们利用这一数据集，对比了使用两种方法训练的嵌入模型：(1) 训练模型生成数学等价表达式；(2) 使用对比学习方法训练模型，明确地将数学等价表达式分组。我们对这些方法生成的嵌入向量在分布内和分布外的语言处理任务中进行了评估。最后，我们将我们的嵌入方案与最新的大规模语言模型进行了性能对比，并证明了基于嵌入的语言处理方法在多项任务上优于LLM，这表明需要优化嵌入方法以适配数学数据模态。 

---
# Wormhole Memory: A Rubik's Cube for Cross-Dialogue Retrieval 

**Title (ZH)**: wormhole memory：跨对话检索的魔方 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14846)  

**Abstract**: In view of the gap in the current large language model in sharing memory across dialogues, this research proposes a wormhole memory module (WMM) to realize memory as a Rubik's cube that can be arbitrarily retrieved between different dialogues. Through simulation experiments, the researcher built an experimental framework based on the Python environment and used setting memory barriers to simulate the current situation where memories between LLMs dialogues are difficult to share. The CoQA development data set was imported into the experiment, and the feasibility of its cross-dialogue memory retrieval function was verified for WMM's nonlinear indexing and dynamic retrieval, and a comparative analysis was conducted with the capabilities of Titans and MemGPT memory modules. Experimental results show that WMM demonstrated the ability to retrieve memory across dialogues and the stability of quantitative indicators in eight experiments. It contributes new technical approaches to the optimization of memory management of LLMs and provides experience for the practical application in the future. 

**Abstract (ZH)**: 鉴于当前大语言模型在跨对话共享记忆方面存在差距，本研究提出了一种虫洞记忆模块（WMM），以实现记忆如同魔方般可以在不同对话之间任意检索。通过模拟实验，研究者基于Python环境构建了一个实验框架，并通过设置记忆屏障来模拟当前大语言模型（LLM）之间的对话记忆难以共享的情况。将CoQA开发数据集导入实验中，验证了WMM在非线性索引和动态检索方面的跨对话记忆检索功能，并与Titan和MemGPT记忆模块的能力进行了比较分析。实验结果显示，WMM展示了跨对话检索记忆的能力，并在八次实验中证明了其量化指标的稳定性。该研究为大语言模型记忆管理的优化提供了新的技术手段，并为未来实际应用提供了经验。 

---
# Leveraging Social Media Data and Artificial Intelligence for Improving Earthquake Response Efforts 

**Title (ZH)**: 利用社交媒体数据和人工智能提高地震响应努力 

**Authors**: Kalin Kopanov, Velizar Varbanov, Tatiana Atanasova  

**Link**: [PDF](https://arxiv.org/pdf/2501.14767)  

**Abstract**: The integration of social media and artificial intelligence (AI) into disaster management, particularly for earthquake response, represents a profound evolution in emergency management practices. In the digital age, real-time information sharing has reached unprecedented levels, with social media platforms emerging as crucial communication channels during crises. This shift has transformed traditional, centralized emergency services into more decentralized, participatory models of disaster situational awareness. Our study includes an experimental analysis of 8,900 social media interactions, including 2,920 posts and 5,980 replies on X (formerly Twitter), following a magnitude 5.1 earthquake in Oklahoma on February 2, 2024. The analysis covers data from the immediate aftermath and extends over the following seven days, illustrating the critical role of digital platforms in modern disaster response. The results demonstrate that social media platforms can be effectively used as real-time situational awareness tools, delivering critical information to society and authorities during emergencies. 

**Abstract (ZH)**: 将社交媒体和人工智能（AI）整合到灾害管理中，特别是在地震响应中的应用，标志着应急管理工作的一种深刻变革。在数字化时代，实时信息共享达到了前所未有的水平，社交媒体平台在危机期间成为关键的沟通渠道。这一转变将传统的集中式紧急服务转化为更分散和参与式的灾害情况监测模式。我们研究包括对8,900条社交媒体互动的实验分析，其中包括2,920条帖子和5,980条评论，这些数据来源于2024年2月2日美国俄克拉荷马州发生里氏5.1级地震后的即时情况以及随后的七天。分析结果表明，社交媒体平台可以作为实时情况监测工具，在紧急情况下向公众和社会权威机构提供关键信息。研究结果表明，社交媒体平台能够有效用于实时情况监测，为应急响应提供关键信息。 

---
# From Critique to Clarity: A Pathway to Faithful and Personalized Code Explanations with Large Language Models 

**Title (ZH)**: 从批判到清晰：一条通往忠实且个性化代码解释的道路——大型语言模型的应用 

**Authors**: Zexing Xu, Zhuang Luo, Yichuan Li, Kyumin Lee, S. Rasoul Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2501.14731)  

**Abstract**: In the realm of software development, providing accurate and personalized code explanations is crucial for both technical professionals and business stakeholders. Technical professionals benefit from enhanced understanding and improved problem-solving skills, while business stakeholders gain insights into project alignments and transparency. Despite the potential, generating such explanations is often time-consuming and challenging. This paper presents an innovative approach that leverages the advanced capabilities of large language models (LLMs) to generate faithful and personalized code explanations. Our methodology integrates prompt enhancement, self-correction mechanisms, personalized content customization, and interaction with external tools, facilitated by collaboration among multiple LLM agents. We evaluate our approach using both automatic and human assessments, demonstrating that our method not only produces accurate explanations but also tailors them to individual user preferences. Our findings suggest that this approach significantly improves the quality and relevance of code explanations, offering a valuable tool for developers and stakeholders alike. 

**Abstract (ZH)**: 在软件开发领域，提供准确且个性化的代码解释对于技术人员和商业利益相关者都至关重要。技术人员可以从增强的理解力和改善的问题解决能力中受益，而商业利益相关者则可以借此获得项目对齐和透明度的洞见。尽管具有这种潜力，生成这样的解释往往是耗时且具有挑战性的。本文提出了一种创新的方法，利用大型语言模型（LLMs）的高级功能来生成忠实且个性化的代码解释。我们的方法结合了提示增强、自我校正机制、个性化内容定制以及与外部工具的交互，通过多个LLM代理的合作来实现。我们通过自动评估和人工评估两种方式对我们的方法进行了评估，结果显示，我们的方法不仅能够生成准确的解释，还能针对个体用户的偏好进行定制。我们的研究结果表明，这种方法显著提高了代码解释的质量和相关性，为开发人员和利益相关者提供了有价值的工具。 

---
