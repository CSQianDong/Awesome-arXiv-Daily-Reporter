# Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model 

**Title (ZH)**: 将时间图学习集成到基于大规模语言模型的时间知识图模型中 

**Authors**: He Chang, Jie Wu, Zhulin Tao, Yunshan Ma, Xianglin Huang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.11911)  

**Abstract**: Temporal Knowledge Graph Forecasting (TKGF) aims to predict future events based on the observed events in history. Recently, Large Language Models (LLMs) have exhibited remarkable capabilities, generating significant research interest in their application for reasoning over temporal knowledge graphs (TKGs). Existing LLM-based methods have integrated retrieved historical facts or static graph representations into LLMs. Despite the notable performance of LLM-based methods, they are limited by the insufficient modeling of temporal patterns and ineffective cross-modal alignment between graph and language, hindering the ability of LLMs to fully grasp the temporal and structural information in TKGs. To tackle these issues, we propose a novel framework TGL-LLM to integrate temporal graph learning into LLM-based temporal knowledge graph model. Specifically, we introduce temporal graph learning to capture the temporal and relational patterns and obtain the historical graph embedding. Furthermore, we design a hybrid graph tokenization to sufficiently model the temporal patterns within LLMs. To achieve better alignment between graph and language, we employ a two-stage training paradigm to finetune LLMs on high-quality and diverse data, thereby resulting in better performance. Extensive experiments on three real-world datasets show that our approach outperforms a range of state-of-the-art (SOTA) methods. 

**Abstract (ZH)**: 时间知识图谱预测（TKGF）旨在基于历史观察事件来预测未来事件。近年来，大型语言模型（LLMs）表现出显著的能力，并引起了将其应用于时间知识图谱（TKGs）推理的研究兴趣。现有的基于LLM的方法将检索到的历史事实或静态图表示整合到LLM中。尽管基于LLM的方法取得了显著的性能，但它们受限于对时间模式建模不足和图与语言之间的无效跨模态对齐，这阻碍了LLM完全理解和把握TKGs中的时间和结构信息的能力。为了解决这些问题，我们提出了一种新的框架TGL-LLM，将时间图学习整合到基于LLM的时间知识图谱模型中。具体而言，我们引入了时间图学习来捕捉时间和关系模式，并获取历史图嵌入。此外，我们设计了一种混合图标记化方法，以充分在LLM中建模时间模式。为了实现更好的图与语言之间的对齐，我们采用两阶段训练范式在高质量和多样性的数据上微调LLM，从而提高性能。在三个真实世界的数据集上的广泛实验表明，我们的方法在多种先进方法中表现更优。 

---
# Ontology Matching with Large Language Models and Prioritized Depth-First Search 

**Title (ZH)**: 使用大规模语言模型和优先深度优先搜索的本体匹配 

**Authors**: Maria Taboada, Diego Martinez, Mohammed Arideh, Rosa Mosquera  

**Link**: [PDF](https://arxiv.org/pdf/2501.11441)  

**Abstract**: Ontology matching (OM) plays a key role in enabling data interoperability and knowledge sharing, but it remains challenging due to the need for large training datasets and limited vocabulary processing in machine learning approaches. Recently, methods based on Large Language Model (LLMs) have shown great promise in OM, particularly through the use of a retrieve-then-prompt pipeline. In this approach, relevant target entities are first retrieved and then used to prompt the LLM to predict the final matches. Despite their potential, these systems still present limited performance and high computational overhead. To address these issues, we introduce MILA, a novel approach that embeds a retrieve-identify-prompt pipeline within a prioritized depth-first search (PDFS) strategy. This approach efficiently identifies a large number of semantic correspondences with high accuracy, limiting LLM requests to only the most borderline cases. We evaluated MILA using the biomedical challenge proposed in the 2023 and 2024 editions of the Ontology Alignment Evaluation Initiative. Our method achieved the highest F-Measure in four of the five unsupervised tasks, outperforming state-of-the-art OM systems by up to 17%. It also performed better than or comparable to the leading supervised OM systems. MILA further exhibited task-agnostic performance, remaining stable across all tasks and settings, while significantly reducing LLM requests. These findings highlight that high-performance LLM-based OM can be achieved through a combination of programmed (PDFS), learned (embedding vectors), and prompting-based heuristics, without the need of domain-specific heuristics or fine-tuning. 

**Abstract (ZH)**: 本研究内容或标题的中文翻译如下，符合学术规范：

本研究探索了知识本体匹配（OM）在实现数据互操作性和知识共享中发挥的关键作用，但由于机器学习方法需要大量训练数据和有限的词汇处理能力，这一过程依然面临挑战。最近，基于大型语言模型（LLMs）的方法在OM方面展现出了极大的潜力，特别是在通过检索-提示管道的方法中。在该方法中，首先检索相关的目标实体，然后使用这些实体来提示LLMs进行最终匹配预测。尽管这些方法具有很大的潜力，但它们仍然存在性能有限和计算开销高的问题。为了解决这些问题，我们提出了一个名为MILA的新颖方法，将检索-识别-提示管道嵌入优先深度优先搜索（PDFS）策略中。该方法能够高效地识别大量高精度的语义对应关系，仅在最边缘的情况下才触发LLMs的请求。我们使用2023年和2024年的本体对齐评估倡议（Ontology Alignment Evaluation Initiative, OAEI）提出的生物医学挑战任务来评估MILA。我们的方法在四个未监督任务中均获得了最高的F-测度，比最先进的OM系统高出17%，并且在性能上优于或接近领先的监督OM系统。此外，MILA还表现出任务无关性，在所有任务和设置中保持稳定，同时显著减少了对LLMs的请求次数。这些研究结果强调，通过结合编程（PDFS）、学习（嵌入向量）和提示的启发式方法，可以在不依赖特定领域启发式方法或微调的情况下实现高性能的LLM驱动OM。

关键词：知识本体匹配（OM）、大型语言模型（LLMs）、检索-提示管道、优先深度优先搜索（PDFS）、F-测度、本体对齐评估倡议（OAEI）、生物医学挑战任务。 

---
# Enhancing User Intent for Recommendation Systems via Large Language Models 

**Title (ZH)**: 通过大型语言模型增强推荐系统的用户意图 

**Authors**: Xiaochuan Xu, Zeqiu Xu, Peiyang Yu, Jiani Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10871)  

**Abstract**: Recommendation systems play a critical role in enhancing user experience and engagement in various online platforms. Traditional methods, such as Collaborative Filtering (CF) and Content-Based Filtering (CBF), rely heavily on past user interactions or item features. However, these models often fail to capture the dynamic and evolving nature of user preferences. To address these limitations, we propose DUIP (Dynamic User Intent Prediction), a novel framework that combines LSTM networks with Large Language Models (LLMs) to dynamically capture user intent and generate personalized item recommendations. The LSTM component models the sequential and temporal dependencies of user behavior, while the LLM utilizes the LSTM-generated prompts to predict the next item of interest. Experimental results on three diverse datasets ML-1M, Games, and Bundle show that DUIP outperforms a wide range of baseline models, demonstrating its ability to handle the cold-start problem and real-time intent adaptation. The integration of dynamic prompts based on recent user interactions allows DUIP to provide more accurate, context-aware, and personalized recommendations. Our findings suggest that DUIP is a promising approach for next-generation recommendation systems, with potential for further improvements in cross-modal recommendations and scalability. 

**Abstract (ZH)**: 推荐系统在各类在线平台中对于提升用户体验和参与度方面发挥着关键作用。传统的推荐方法，如协同过滤（Collaborative Filtering, CF）和基于内容的过滤（Content-Based Filtering, CBF），主要依赖于过往的用户交互或项目特征。然而，这些模型往往难以捕捉用户偏好动态变化的特性。为解决这些问题，我们提出了一种名为DUIP（Dynamic User Intent Prediction）的新颖框架，它结合了长短期记忆网络（Long Short-Term Memory, LSTM）和大型语言模型（Large Language Models, LLMs），以动态捕捉用户意图并生成个性化推荐。LSTM组件模型用户行为的序列性和时间依赖性，而LLM则利用LSTM生成的提示来预测用户感兴趣的下一个项目。在三个不同的数据集ML-1M、Games和Bundle上的实验结果表明，DUIP在广泛的基本模型中表现出优越性，证明了其处理冷启动问题和实时意图调整的能力。基于最近用户交互动态提示的集成使得DUIP能够提供更准确、上下文相关且个性化的推荐。我们的研究结果表明，DUIP是一种有前景的下一代推荐系统方法，未来有望在跨模态推荐和扩展性方面进一步改进。 

---
# Automatic Labelling with Open-source LLMs using Dynamic Label Schema Integration 

**Title (ZH)**: 使用动态标签方案集成的开源大语言模型自动标注 

**Authors**: Thomas Walshe, Sae Young Moon, Chunyang Xiao, Yawwani Gunawardana, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2501.12332)  

**Abstract**: Acquiring labelled training data remains a costly task in real world machine learning projects to meet quantity and quality requirements. Recently Large Language Models (LLMs), notably GPT-4, have shown great promises in labelling data with high accuracy. However, privacy and cost concerns prevent the ubiquitous use of GPT-4. In this work, we explore effectively leveraging open-source models for automatic labelling. We identify integrating label schema as a promising technology but found that naively using the label description for classification leads to poor performance on high cardinality tasks. To address this, we propose Retrieval Augmented Classification (RAC) for which LLM performs inferences for one label at a time using corresponding label schema; we start with the most related label and iterates until a label is chosen by the LLM. We show that our method, which dynamically integrates label description, leads to performance improvements in labelling tasks. We further show that by focusing only on the most promising labels, RAC can trade off between label quality and coverage - a property we leverage to automatically label our internal datasets. 

**Abstract (ZH)**: 在现实世界的机器学习项目中，获取带有标注的训练数据仍是一项昂贵的任务，需要满足数量和质量要求。最近，大型语言模型（LLMs），尤其是GPT-4，在数据标注方面展示了高准确性，但隐私和成本问题阻碍了GPT-4的广泛应用。在本研究中，我们探索了有效地利用开源模型进行自动标注。我们发现，将标签模式集成在一起是一种有前景的技术，但直接使用标签描述进行分类在高维度任务中表现不佳。为了解决这一问题，我们提出了检索增强分类（RAC）方法，其中LLM一次对一个标签进行推理，并根据相应的标签模式进行；我们从关联性最强的标签开始，迭代直至LLM选择一个标签。研究表明，我们的方法能够动态集成标签描述，从而在标注任务中提高性能。进一步地，我们展示了通过仅关注最具前景的标签，RAC可以在标签质量和覆盖之间进行权衡——我们利用这一特性自动标注我们的内部数据集。 

---
# Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement 

**Title (ZH)**: Condor：通过知识驱动的数据合成与精炼提高语言模型的对齐度 

**Authors**: Maosong Cao, Taolin Zhang, Mo Li, Chuyu Zhang, Yunxin Liu, Haodong Duan, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.12273)  

**Abstract**: The quality of Supervised Fine-Tuning (SFT) data plays a critical role in enhancing the conversational capabilities of Large Language Models (LLMs). However, as LLMs become more advanced, the availability of high-quality human-annotated SFT data has become a significant bottleneck, necessitating a greater reliance on synthetic training data. In this work, we introduce Condor, a novel two-stage synthetic data generation framework that incorporates World Knowledge Tree and Self-Reflection Refinement to produce high-quality SFT data at scale. Our experimental results demonstrate that a base model fine-tuned on only 20K Condor-generated samples achieves superior performance compared to counterparts. The additional refinement stage in Condor further enables iterative self-improvement for LLMs at various scales (up to 72B), validating the effectiveness of our approach. Furthermore, our investigation into the scaling for synthetic data in post-training reveals substantial unexplored potential for performance improvements, opening promising avenues for future research. 

**Abstract (ZH)**: 监督微调（SFT）数据的质量在提升大型语言模型（LLMs）的对话能力方面起着关键作用。然而，随着LLMs的不断进步，高质量的人工标注SFT数据的可用性已经成为一个重要瓶颈，迫使人们更加依赖合成训练数据。在此工作中，我们提出了Condor，这是一个新颖的两阶段合成数据生成框架，结合了World Knowledge Tree和Self-Reflection Refinement，以大规模生成高质量的SFT数据。我们的实验结果表明，仅使用20,000个Condor生成样本进行微调的基本模型，其性能优于其他模型。Condor中的额外完善阶段还进一步使LLMs在不同规模（多达720亿）下实现了迭代自我改进，验证了我们方法的有效性。此外，我们对后训练合成数据的扩展性研究揭示了大量未开发的性能提升潜力，为未来研究打开了有前景的研究方向。 

---
# AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding 

**Title (ZH)**: AdaServe：基于细粒度推测解码的SLO定制化大语言模型服务 

**Authors**: Zikun Li, Zhuofu Chen, Remi Delacourt, Gabriele Oliaro, Zeyu Wang, Qinghan Chen, Shuhuai Lin, April Yang, Zhihao Zhang, Zhuoming Chen, Sean Lai, Xupeng Miao, Zhihao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2501.12162)  

**Abstract**: This paper introduces AdaServe, the first LLM serving system to support SLO customization through fine-grained speculative decoding. AdaServe leverages the logits of a draft model to predict the speculative accuracy of tokens and employs a theoretically optimal algorithm to construct token trees for verification. To accommodate diverse SLO requirements without compromising throughput, AdaServe employs a speculation-and-selection scheme that first constructs candidate token trees for each request and then dynamically selects tokens to meet individual SLO constraints while optimizing throughput. Comprehensive evaluations demonstrate that AdaServe achieves up to 73% higher SLO attainment and 74% higher goodput compared to state-of-the-art systems. These results underscore AdaServe's potential to enhance the efficiency and adaptability of LLM deployments across varied application scenarios. 

**Abstract (ZH)**: 本文介绍了AdaServe，这是首个通过细粒度推测解码支持自定义服务水平目标（SLA）的LLM服务系统。AdaServe 利用草稿模型的logits来预测推测性解码的准确性，并采用理论上最优的算法构建查验词树。为了满足不同类型的SLA需求而不牺牲吞吐量，AdaServe 采用了一种推测与选择方案：首先为每个请求构建候选词树，然后动态选择满足个体SLA约束并优化吞吐量的词。全面的评估表明，与现有最先进的系统相比，AdaServe 的SLA达成率可提高73%，有效吞吐量提高74%。这些结果凸显了AdaServe 在提高LLM部署效率和适应性方面的潜力，适用于各种应用场景。 

---
# Improving Influence-based Instruction Tuning Data Selection for Balanced Learning of Diverse Capabilities 

**Title (ZH)**: 基于影响力的指令调优数据选择改进：促进多样能力平衡学习 

**Authors**: Qirun Dai, Dylan Zhang, Jiaqi W. Ma, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.12147)  

**Abstract**: Selecting appropriate training data is crucial for effective instruction fine-tuning of large language models (LLMs), which aims to (1) elicit strong capabilities, and (2) achieve balanced performance across a diverse range of tasks. Influence-based methods show promise in achieving (1) by estimating the contribution of each training example to the model's predictions, but often struggle with (2). Our systematic investigation reveals that this underperformance can be attributed to an inherent bias where certain tasks intrinsically have greater influence than others. As a result, data selection is often biased towards these tasks, not only hurting the model's performance on others but also, counterintuitively, harms performance on these high-influence tasks themselves.
As a remedy, we propose BIDS, a Balanced and Influential Data Selection algorithm. BIDS first normalizes influence scores of the training data, and then iteratively balances data selection by choosing the training example with the highest influence on the most underrepresented task. Experiments with both Llama-3 and Mistral-v0.3 on seven benchmarks spanning five diverse capabilities show that BIDS consistently outperforms both state-of-the-art influence-based algorithms and other non-influence-based selection frameworks. Surprisingly, training on a 15% subset selected by BIDS can even outperform full-dataset training with a much more balanced performance. Our analysis further highlights the importance of both instance-level normalization and iterative optimization of selected data for balanced learning of diverse capabilities. 

**Abstract (ZH)**: 选择合适的训练数据对于大型语言模型（LLMs）的有效指令微调至关重要，其目标是（1）激发强大的能力，以及（2）在广泛的任务范围内实现均衡的表现。基于影响的方法通过估计每个训练样本对模型预测的贡献显示出实现（1）的潜力，但常常在实现（2）方面遇到困难。我们的系统性研究揭示，这种表现不佳可以归因于一种固有的偏见，即某些任务本质上比其他任务有更大的影响。因此，数据选择通常偏向这些任务，不仅损害了模型在其他任务上的表现，而且出人意料地也损害了这些高影响任务本身的表现。

为解决这一问题，我们提出了一种平衡且有影响力的样本选择算法——BIDS（Balanced and Influential Data Selection）。BIDS 首先对训练数据的影响评分进行归一化处理，然后通过选择对最不足代表的任务具有最高影响的训练样本来迭代地平衡数据选择。在对 Llama-3 和 Mistral-v0.3 进行的涵盖五种不同能力的七项基准测试实验中，BIDS 一致性地优于最先进的基于影响的算法以及其他非基于影响的选择框架。令人惊讶的是，使用 BIDS 选择的 15% 数据子集进行训练，其表现甚至可能超过使用完整数据集进行训练的平衡性表现。我们的进一步分析强调了实例级归一化和选定数据的迭代优化对于掌握多样化能力的重要性。 

---
# Can open source large language models be used for tumor documentation in Germany? -- An evaluation on urological doctors' notes 

**Title (ZH)**: 开源大规模语言模型是否可以用于德国的肿瘤记录？--对泌尿科医生笔记的评估 

**Authors**: Stefan Lenz, Arsenij Ustjanzew, Marco Jeray, Torsten Panholzer  

**Link**: [PDF](https://arxiv.org/pdf/2501.12106)  

**Abstract**: Tumor documentation in Germany is largely done manually, requiring reading patient records and entering data into structured databases. Large language models (LLMs) could potentially enhance this process by improving efficiency and reliability. This evaluation tests eleven different open source LLMs with sizes ranging from 1-70 billion model parameters on three basic tasks of the tumor documentation process: identifying tumor diagnoses, assigning ICD-10 codes, and extracting the date of first diagnosis. For evaluating the LLMs on these tasks, a dataset of annotated text snippets based on anonymized doctors' notes from urology was prepared. Different prompting strategies were used to investigate the effect of the number of examples in few-shot prompting and to explore the capabilities of the LLMs in general. The models Llama 3.1 8B, Mistral 7B, and Mistral NeMo 12 B performed comparably well in the tasks. Models with less extensive training data or having fewer than 7 billion parameters showed notably lower performance, while larger models did not display performance gains. Examples from a different medical domain than urology could also improve the outcome in few-shot prompting, which demonstrates the ability of LLMs to handle tasks needed for tumor documentation. Open source LLMs show a strong potential for automating tumor documentation. Models from 7-12 billion parameters could offer an optimal balance between performance and resource efficiency. With tailored fine-tuning and well-designed prompting, these models might become important tools for clinical documentation in the future. The code for the evaluation is available from this https URL. We also release the dataset as a new valuable resource that addresses the shortage of authentic and easily accessible benchmarks in German-language medical NLP. 

**Abstract (ZH)**: 德国的肿瘤记录主要依赖手工完成，需要阅读患者记录并将其数据录入到结构化的数据库中。大型语言模型（LLMs）有可能通过提高效率和可靠性来优化这一过程。本次评估测试了11种不同开源LLMs，其模型参数范围从1亿到70亿个，针对肿瘤记录过程中的三个基本任务进行了测试：识别肿瘤诊断、分配ICD-10代码以及提取初次诊断日期。为了评估这些任务中LLMs的表现，基于匿名化医生笔记的数据集被用于标注文本片段。不同的提示策略被使用，以研究少样本提示中示例数量的影响，并探索LLMs的一般能力。在这些任务中，Llama 3.1 8B、Mistral 7B和Mistral NeMo 12 B等模型表现得相当优秀。拥有较少训练数据或参数少于7亿的模型表现明显较差，而更大规模的模型并未表现出性能提升。来自医学其他领域不同的训练示例也可能在少样本提示中提高结果，这表明LLMs能够处理进行肿瘤记录所需的任务。开源LLMs显示出自动化的肿瘤记录的强潜力。参数在7亿到12亿之间的模型可能在性能和资源效率之间提供最佳平衡。通过定制微调和精心设计的提示，这些模型未来可能成为临床记录的重要工具。评估的代码可以从该链接下载：[此处插入链接]。我们还发布了该数据集，作为在德语医学自然语言处理中缺乏真实且易于访问基准资源的新有价值资源。 

---
# Leveraging Graph Structures and Large Language Models for End-to-End Synthetic Task-Oriented Dialogues 

**Title (ZH)**: 利用图结构和大规模语言模型实现端到端的任务导向合成对话 

**Authors**: Maya Medjad, Hugo Imbert, Bruno Yun, Raphaël Szymocha, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2501.11977)  

**Abstract**: Training task-oriented dialogue systems is both costly and time-consuming, due to the need for high-quality datasets encompassing diverse intents. Traditional methods depend on extensive human annotation, while recent advancements leverage large language models (LLMs) to generate synthetic data. However, these approaches often require custom prompts or code, limiting accessibility for non-technical users. We introduce GraphTOD, an end-to-end framework that simplifies the generation of task-oriented dialogues. Users can create dialogues by specifying transition graphs in JSON format. Our evaluation demonstrates that GraphTOD generates high-quality dialogues across various domains, significantly lowering the cost and complexity of dataset creation. 

**Abstract (ZH)**: 训练面向任务的对话系统既耗时又耗费资源，因为需要涵盖多种意图的高质量数据集。传统方法依赖广泛的-human标注，而最近的进展则利用大规模语言模型（LLMs）生成合成数据。然而，这些方法通常需要自定义提示或代码，限制了非技术人员的访问。我们引入了GraphTOD，这是一个端到端的框架，简化了任务 oriented 对话的生成过程。用户可以通过指定 JSON 格式的转换图来创建对话。我们的评估表明，GraphTOD 能够在各种领域生成高质量的对话，显著降低了数据集创建的成本和复杂性。 

---
# A Hybrid Attention Framework for Fake News Detection with Large Language Models 

**Title (ZH)**: 基于大型语言模型的虚假新闻检测混合注意力框架 

**Authors**: Xiaochuan Xu, Peiyang Yu, Zeqiu Xu, Jiani Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11967)  

**Abstract**: With the rapid growth of online information, the spread of fake news has become a serious social challenge. In this study, we propose a novel detection framework based on Large Language Models (LLMs) to identify and classify fake news by integrating textual statistical features and deep semantic features. Our approach utilizes the contextual understanding capability of the large language model for text analysis and introduces a hybrid attention mechanism to focus on feature combinations that are particularly important for fake news identification. Extensive experiments on the WELFake news dataset show that our model significantly outperforms existing methods, with a 1.5\% improvement in F1 score. In addition, we assess the interpretability of the model through attention heat maps and SHAP values, providing actionable insights for content review strategies. Our framework provides a scalable and efficient solution to deal with the spread of fake news and helps build a more reliable online information ecosystem. 

**Abstract (ZH)**: 随着在线信息的快速增长，假新闻的传播已成为一个严重的社会挑战。本研究提出了一种基于大型语言模型（LLMs）的新型检测框架，通过整合文本统计特征和深层语义特征来识别和分类假新闻。该方法利用大型语言模型的文本理解能力进行文本分析，并引入混合注意力机制，聚焦于特别重要的假新闻识别特征组合。我们在WELFake新闻数据集上的广泛实验表明，我们的模型显著优于现有方法，F1分数提高了1.5%。此外，我们通过注意力热图和SHAP值评估了模型的可解释性，提供了内容审核策略的可操作见解。该框架提供了一个可扩展且高效的解决方案，应对假新闻的传播，并有助于构建一个更可靠的信息生态系统。 

---
# Proverbs Run in Pairs: Evaluating Proverb Translation Capability of Large Language Model 

**Title (ZH)**: 成语成对出现：评估大型语言模型的成语翻译能力 

**Authors**: Minghan Wang, Viet-Thanh Pham, Farhad Moghimifar, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11953)  

**Abstract**: Despite achieving remarkable performance, machine translation (MT) research remains underexplored in terms of translating cultural elements in languages, such as idioms, proverbs, and colloquial expressions. This paper investigates the capability of state-of-the-art neural machine translation (NMT) and large language models (LLMs) in translating proverbs, which are deeply rooted in cultural contexts. We construct a translation dataset of standalone proverbs and proverbs in conversation for four language pairs. Our experiments show that the studied models can achieve good translation between languages with similar cultural backgrounds, and LLMs generally outperform NMT models in proverb translation. Furthermore, we find that current automatic evaluation metrics such as BLEU, CHRF++ and COMET are inadequate for reliably assessing the quality of proverb translation, highlighting the need for more culturally aware evaluation metrics. 

**Abstract (ZH)**: 尽管机器翻译（MT）研究在性能上取得了显著成就，但关于翻译文化元素如成语、格言和俚语的研究仍相对不足。本文探讨了最新神经机器翻译（NMT）和大语言模型（LLMs）在翻译格言方面的能力，这些格言深深植根于文化背景之中。我们为四组语言构建了一个独立格言及其在对话中的翻译数据集。实验显示，研究的模型可以在具有相似文化背景的语言之间取得良好的翻译效果，并且在格言翻译方面，大语言模型通常优于神经机器翻译模型。此外，我们发现当前自动评估指标如 BLEU、CHRF++ 和 COMET 在可靠评估格言翻译质量方面存在不足，强调了需要开发更多文化意识更强的评估指标的必要性。 

---
# Panoramic Interests: Stylistic-Content Aware Personalized Headline Generation 

**Title (ZH)**: 全景兴趣：风格-内容兼顾的个性化标题生成 

**Authors**: Junhong Lian, Xiang Ao, Xinyu Liu, Yang Liu, Qing He  

**Link**: [PDF](https://arxiv.org/pdf/2501.11900)  

**Abstract**: Personalized news headline generation aims to provide users with attention-grabbing headlines that are tailored to their preferences. Prevailing methods focus on user-oriented content preferences, but most of them overlook the fact that diverse stylistic preferences are integral to users' panoramic interests, leading to suboptimal personalization. In view of this, we propose a novel Stylistic-Content Aware Personalized Headline Generation (SCAPE) framework. SCAPE extracts both content and stylistic features from headlines with the aid of large language model (LLM) collaboration. It further adaptively integrates users' long- and short-term interests through a contrastive learning-based hierarchical fusion network. By incorporating the panoramic interests into the headline generator, SCAPE reflects users' stylistic-content preferences during the generation process. Extensive experiments on the real-world dataset PENS demonstrate the superiority of SCAPE over baselines. 

**Abstract (ZH)**: 个性化新闻标题生成旨在为用户提供符合其兴趣的、能吸引注意力的新闻标题。现有方法主要关注用户的内容偏好，但大多忽略了用户广泛的兴趣中包含多样化风格偏好这一事实，从而导致个性化不足。为解决这一问题，我们提出了一种新颖的风格-内容感知个性化标题生成（SCAPE）框架。SCAPE借助大型语言模型（LLM）的协作，提取标题的内容和风格特征，并通过基于对比学习的层次融合网络，进一步适应性地整合用户的长短期兴趣。通过将用户的宏观兴趣纳入标题生成过程，SCAPE在生成过程中反映了用户的风格-内容偏好。在真实世界数据集PENS上的广泛实验表明，SCAPE在性能上优于基线方法。 

---
# Med-R$^2$: Crafting Trustworthy LLM Physicians through Retrieval and Reasoning of Evidence-Based Medicine 

**Title (ZH)**: Med-R²：通过基于证据的医学检索与推理打造可信赖的LLM医生 

**Authors**: Keer Lu, Zheng Liang, Da Pan, Shusen Zhang, Xin Wu, Weipeng Chen, Zenan Zhou, Guosheng Dong, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11885)  

**Abstract**: In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. However, despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.87\% improvement over vanilla RAG methods and even a 3.59\% enhancement compared to fine-tuning strategies, without incurring additional training costs. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在临床场景中展现出了非凡的能力。然而，尽管这些模型具有潜力，在将它们应用于医疗领域时仍面临诸多挑战。依赖医学数据集进行训练的方法成本高昂，且可能受到过时训练数据的影响。利用外部知识库是一个可行的替代方案，但此类方法也面临着诸如检索精度较低和难以有效抽取答案等障碍。这些问题共同阻碍了LLMs在掌握医学专业知识方面达到预期水平。为解决这些问题，我们提出了Med-R^2，这是一种新颖的LLM医生框架，遵循循证医学（EBM）过程，高效地整合了检索机制以及证据的选择和推理过程，从而增强LLMs在医疗场景中的问题解决能力，并培养一种值得信赖的LLM医生。我们全面的实验表明，Med-R^2 在与基础检索聚合（RAG）方法相比时展示了14.87% 的改进，并且相较于微调策略还展现了3.59% 的提升，而无需额外的训练成本。 

---
# From Drafts to Answers: Unlocking LLM Potential via Aggregation Fine-Tuning 

**Title (ZH)**: 从草稿到答案：通过聚合微调释放大规模语言模型的潜力 

**Authors**: Yafu Li, Zhilin Wang, Tingchen Fu, Ganqu Cui, Sen Yang, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.11877)  

**Abstract**: Scaling data and model size has been proven effective for boosting the performance of large language models. In addition to training-time scaling, recent studies have revealed that increasing test-time computational resources can further improve performance. In this work, we introduce Aggregation Fine-Tuning (AFT), a supervised finetuning paradigm where the model learns to synthesize multiple draft responses, referred to as proposals, into a single, refined answer, termed aggregation. At inference time, a propose-and-aggregate strategy further boosts performance by iteratively generating proposals and aggregating them. Empirical evaluations on benchmark datasets show that AFT-trained models substantially outperform standard SFT. Notably, an AFT model, fine-tuned from Llama3.1-8B-Base with only 64k data, achieves a 41.3% LC win rate on AlpacaEval 2, surpassing significantly larger LLMs such as Llama3.1-405B-Instruct and GPT4. By combining sequential refinement and parallel sampling, the propose-and-aggregate framework scales inference-time computation in a flexible manner. Overall, These findings position AFT as a promising approach to unlocking additional capabilities of LLMs without resorting to increasing data volume or model size. 

**Abstract (ZH)**: 扩大数据集和模型规模已被证明可以有效提升大型语言模型的性能。除了训练时的扩展外，近期的研究表明，增加测试时的计算资源也可以进一步提高性能。在本研究中，我们提出了聚合微调（AFT，Aggregation Fine-Tuning）这一监督微调范式，其中模型学习将多个草案响应（proposals）综合成一个精炼的回答（aggregation）。在推断时，采用提出并聚合的策略进一步提升性能，通过迭代生成草案并综合它们。在基准数据集上的实验评估表明，经过AFT训练的模型显著优于标准的语境微调（SFT，Supervised Fine-Tuning）。值得注意的是，从Llama3.1-8B-Base微调的AFT模型，仅使用64k数据，取得了AlpacaEval 2中41.3%的胜率，明显优于更大规模的LLM，如Llama3.1-405B-Instruct和GPT4。通过结合序列精炼和并行采样，提出并聚合框架以灵活的方式扩展推断时的计算量。总之，这些发现将AFT定位为一种有前途的方法，可以在不增加数据量或模型规模的情况下解锁LLM的额外能力。 

---
# Network-informed Prompt Engineering against Organized Astroturf Campaigns under Extreme Class Imbalance 

**Title (ZH)**: 网络导向的提示工程以对抗极端类别不平衡下的组织化水军运动 

**Authors**: Nikos Kanakaris, Heng Ping, Xiongye Xiao, Nesreen K. Ahmed, Luca Luceri, Emilio Ferrara, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2501.11849)  

**Abstract**: Detecting organized political campaigns is of paramount importance in fighting against disinformation on social media. Existing approaches for the identification of such organized actions employ techniques mostly from network science, graph machine learning and natural language processing. Their ultimate goal is to analyze the relationships and interactions (e.g. re-posting) among users and the textual similarities of their posts. Despite their effectiveness in recognizing astroturf campaigns, these methods face significant challenges, notably the class imbalance in available training datasets. To mitigate this issue, recent methods usually resort to data augmentation or increasing the number of positive samples, which may not always be feasible or sufficient in real-world settings. Following a different path, in this paper, we propose a novel framework for identifying astroturf campaigns based solely on large language models (LLMs), introducing a Balanced Retrieval-Augmented Generation (Balanced RAG) component. Our approach first gives both textual information concerning the posts (in our case tweets) and the user interactions of the social network as input to a language model. Then, through prompt engineering and the proposed Balanced RAG method, it effectively detects coordinated disinformation campaigns on X (Twitter). The proposed framework does not require any training or fine-tuning of the language model. Instead, by strategically harnessing the strengths of prompt engineering and Balanced RAG, it facilitates LLMs to overcome the effects of class imbalance and effectively identify coordinated political campaigns. The experimental results demonstrate that by incorporating the proposed prompt engineering and Balanced RAG methods, our framework outperforms the traditional graph-based baselines, achieving 2x-3x improvements in terms of precision, recall and F1 scores. 

**Abstract (ZH)**: 检测有组织的政治活动对于打击社交 media 上的虚假信息至关重要。现有的识别此类有组织行动的方法主要采用了网络科学、图机器学习和自然语言处理的技术。这些方法的最终目标是分析用户之间的关系和互动（例如转发）以及其帖子的文本相似性。尽管这些方法在识别假流量活动方面表现出有效性，但它们在应对可用训练数据集中的类别不平衡问题时面临显著挑战。为缓解这一问题，近年来的方法通常依赖于数据增强或增加正样本数量，但在实际应用场景中这可能并不总是可行或足够。

与这种方法不同，本文提出了一种基于大型语言模型（LLMs）的新框架，引入了平衡检索增强生成（Balanced RAG）组件。我们的方法首先将有关帖子（例如推特）的文本信息以及社交网络用户的互动信息输入到语言模型。然后，通过提示工程和所提出的平衡检索增强生成方法，有效检测 X（推特）上的协调性虚假信息活动。该提出的框架不需要对语言模型进行任何训练或微调。相反，通过战略性地利用提示工程和平衡检索增强生成的优势，它使大型语言模型能够克服类别不平衡的影响，并有效识别有组织的政治活动。实验结果表明，通过集成所提出的提示工程和平衡检索增强生成方法，我们的框架在精准度、召回率和 F1 分数上超过了传统的基于图的方法，取得了 2 至 3 倍的提升。 

---
# Is your LLM trapped in a Mental Set? Investigative study on how mental sets affect the reasoning capabilities of LLMs 

**Title (ZH)**: 你的大规模语言模型（LLM）被束缚在思维定势中了吗？探究思维定势对大模型推理能力的影响 

**Authors**: Saiful Haq, Niyati Chhaya, Piyush Pandey, Pushpak Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2501.11833)  

**Abstract**: In this paper, we present an investigative study on how Mental Sets influence the reasoning capabilities of LLMs. LLMs have excelled in diverse natural language processing (NLP) tasks, driven by advancements in parameter-efficient fine-tuning (PEFT) and emergent capabilities like in-context learning (ICL). For complex reasoning tasks, selecting the right model for PEFT or ICL is critical, often relying on scores on benchmarks such as MMLU, MATH, and GSM8K. However, current evaluation methods, based on metrics like F1 Score or reasoning chain assessments by larger models, overlook a key dimension: adaptability to unfamiliar situations and overcoming entrenched thinking patterns. In cognitive psychology, Mental Set refers to the tendency to persist with previously successful strategies, even when they become inefficient - a challenge for problem solving and reasoning. We compare the performance of LLM models like Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct and GPT-4o in the presence of mental sets. To the best of our knowledge, this is the first study to integrate cognitive psychology concepts into the evaluation of LLMs for complex reasoning tasks, providing deeper insights into their adaptability and problem-solving efficacy. 

**Abstract (ZH)**: 本文探讨了心理定势（Mental Sets）如何影响大型语言模型（LLMs）的推理能力。LLMs在多种自然语言处理（NLP）任务中表现出色，这得益于参数高效微调（PEFT）的进步以及内省式学习（ICL）等新兴能力。在复杂推理任务中，选择合适的模型进行PEFT或ICL至关重要，通常依赖于诸如MMLU、MATH和GSM8K等基准测试的得分。然而，当前的评估方法，如基于F1分数或由更大模型评估推理链的方法，忽视了一个关键维度：对不熟悉情况的适应能力以及克服根深蒂固的思维模式。在认知心理学中，心理定势指的是倾向于坚持先前成功的策略，即使这些策略变得无效——这对解决问题和推理构成了挑战。我们对比了Llama-3.1-8B-Instruct、Llama-3.1-70B-Instruct和GPT-4o等LLM模型在面对心理定势时的表现。据我们所知，这是首个将认知心理学概念整合到LLM复杂推理任务评估中的研究，为了解它们的适应性和问题解决效果提供了更深入的洞见。 

---
# Benchmarking Large Language Models via Random Variables 

**Title (ZH)**: 通过随机变量对比大规模语言模型 

**Authors**: Zijin Hong, Hao Wu, Su Dong, Junnan Dong, Yilin Xiao, Yujing Zhang, Zhu Wang, Feiran Huang, Linyi Li, Hongxia Yang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11790)  

**Abstract**: With the continuous advancement of large language models (LLMs) in mathematical reasoning, evaluating their performance in this domain has become a prominent research focus. Recent studies have raised concerns about the reliability of current mathematical benchmarks, highlighting issues such as simplistic design and potential data leakage. Therefore, creating a reliable benchmark that effectively evaluates the genuine capabilities of LLMs in mathematical reasoning remains a significant challenge. To address this, we propose RV-Bench, a framework for Benchmarking LLMs via Random Variables in mathematical reasoning. Specifically, the background content of a random variable question (RV question) mirrors the original problem in existing standard benchmarks, but the variable combinations are randomized into different values. LLMs must fully understand the problem-solving process for the original problem to correctly answer RV questions with various combinations of variable values. As a result, the LLM's genuine capability in mathematical reasoning is reflected by its accuracy on RV-Bench. Extensive experiments are conducted with 29 representative LLMs across 900+ RV questions. A leaderboard for RV-Bench ranks the genuine capability of these LLMs. Further analysis of accuracy dropping indicates that current LLMs still struggle with complex mathematical reasoning problems. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在数学推理领域的不断进步，评估其在该领域的性能已成为一个重要的研究焦点。最近的研究对当前数学基准的可靠性提出了质疑，突显了设计简单及潜在数据泄露等问题。因此，创建一个可靠且能有效评估LLMs在数学推理中真正能力的基准仍然是一个重大挑战。为了解决这一问题，我们提出RV-Bench框架，通过随机变量对LLMs在数学推理中的基准测试进行评估。具体而言，随机变量问题（RV问题）的背景内容与现有标准基准中的原始问题相似，但变量组合被随机化为不同的值。对于原本的问题，LLMs必须完全理解问题解决过程，才能正确回答包含不同变量值组合的RV问题。因此，RV-Bench中的准确性反映了LLMs在数学推理中的真实能力。我们对29个代表性LLMs进行了广泛的实验，涵盖900多个RV问题。RV-Bench的排行榜对这些LLMs的真实能力进行了排名。进一步的准确率分析表明，当前的LLMs在处理复杂的数学推理问题时仍然存在困难。 

---
# Optimizing Pretraining Data Mixtures with LLM-Estimated Utility 

**Title (ZH)**: 使用大规模语言模型估计效用优化预训练数据混合 

**Authors**: William Held, Bhargavi Paranjape, Punit Singh Koura, Mike Lewis, Frank Zhang, Todor Mihaylov  

**Link**: [PDF](https://arxiv.org/pdf/2501.11747)  

**Abstract**: Large Language Models improve with increasing amounts of high-quality training data. However, leveraging larger datasets requires balancing quality, quantity, and diversity across sources. After evaluating nine baseline methods under both compute- and data-constrained scenarios, we find token-count heuristics outperform manual and learned mixes, indicating that simple approaches accounting for dataset size and diversity are surprisingly effective. Building on this insight, we propose two complementary approaches: UtiliMax, which extends token-based heuristics by incorporating utility estimates from reduced-scale ablations, achieving up to a 10.6x speedup over manual baselines; and Model Estimated Data Utility (MEDU), which leverages LLMs to estimate data utility from small samples, matching ablation-based performance while reducing computational requirements by $\sim$200x. Together, these approaches establish a new framework for automated, compute-efficient data mixing that is robust across training regimes. 

**Abstract (ZH)**: 大量语言模型在高质量训练数据的数量增加时会表现出更好的性能。然而，利用更大的数据集需要在质量、数量和来源多样性之间进行权衡。在两种计算能力和数据约束场景下评估了九种基线方法后，我们发现基于标记数的启发式方法优于手动和学习的混合方法，表明简单的考虑数据集大小和多样性的方法其实非常有效。基于这一洞察，我们提出了两种互补的方法：UtiliMax，该方法通过引入规模减小的消减版的效用估计来扩展基于标记数的启发式方法，与手动基线相比，可实现高达10.6倍的加速；以及模型估计数据效用（MEDU），该方法利用语言模型从小样本中估计数据效用，与基于消减的方法性能相当，同时计算需求减少约200倍。这两种方法共同建立了一种新的自动化、计算高效的数据混合框架，在各种训练模式下具有稳健性。 

---
# Explain-Query-Test: Self-Evaluating LLMs Via Explanation and Comprehension Discrepancy 

**Title (ZH)**: 解释-查询-测试：通过解释和理解差异进行自我评估的语言模型 

**Authors**: Saeid Asgari Taghanaki, Joao Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2501.11721)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable proficiency in generating detailed and coherent explanations of complex concepts. However, the extent to which these models truly comprehend the concepts they articulate remains unclear. To assess the level of comprehension of a model relative to the content it generates, we implemented a self-evaluation pipeline where models: (i) given a topic generate an excerpt with information about the topic, (ii) given an excerpt generate question-answer pairs, and finally (iii) given a question generate an answer. We refer to this self-evaluation approach as Explain-Query-Test (EQT). Interestingly, the accuracy on generated questions resulting from running the EQT pipeline correlates strongly with the model performance as verified by typical benchmarks such as MMLU-Pro. In other words, EQT's performance is predictive of MMLU-Pro's, and EQT can be used to rank models without the need for any external source of evaluation data other than lists of topics of interest. Moreover, our results reveal a disparity between the models' ability to produce detailed explanations and their performance on questions related to those explanations. This gap highlights fundamental limitations in the internal knowledge representation and reasoning abilities of current LLMs. We release the code at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成复杂概念的详细和连贯解释方面展现了惊人的能力。然而，这些模型在 articulating 这些概念时真正理解的程度仍然不清楚。为了评估模型对生成内容的理解水平，我们实现了一个自我评估管道，其中模型执行以下操作：（i）给定一个主题，生成一个包含关于该主题信息的段落，（ii）给定一个段落生成问题-答案对，最终（iii）给定一个问题生成一个答案。我们将这种自我评估方法称为解释-查询-测试（EQT）。有趣的是，通过运行EQT管道生成的问题准确性与通过典型基准如MMLU-Pro验证的模型性能高度相关。换句话说，EQT的性能可以预测MMLU-Pro的性能，并且EQT可以在无需任何外部评估数据源的情况下用于对模型进行排名，仅需感兴趣的主题列表即可。此外，我们的结果显示，模型在生成详细解释方面的能力和回答与这些解释相关的问题方面的性能之间存在差异。这一差距突显了当前LLMs在内部知识表示和推理能力方面的基本局限性。我们在此发布代码：https://example.com。 

---
# PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation 

**Title (ZH)**: PIKE-RAG：专门知识和推理增强的生成 

**Authors**: Jinyu Wang, Jingjing Fu, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2501.11551)  

**Abstract**: Despite notable advancements in Retrieval-Augmented Generation (RAG) systems that expand large language model (LLM) capabilities through external retrieval, these systems often struggle to meet the complex and diverse needs of real-world industrial applications. The reliance on retrieval alone proves insufficient for extracting deep, domain-specific knowledge performing in logical reasoning from specialized corpora. To address this, we introduce sPecIalized KnowledgE and Rationale Augmentation Generation (PIKE-RAG), focusing on extracting, understanding, and applying specialized knowledge, while constructing coherent rationale to incrementally steer LLMs toward accurate responses. Recognizing the diverse challenges of industrial tasks, we introduce a new paradigm that classifies tasks based on their complexity in knowledge extraction and application, allowing for a systematic evaluation of RAG systems' problem-solving capabilities. This strategic approach offers a roadmap for the phased development and enhancement of RAG systems, tailored to meet the evolving demands of industrial applications. Furthermore, we propose knowledge atomizing and knowledge-aware task decomposition to effectively extract multifaceted knowledge from the data chunks and iteratively construct the rationale based on original query and the accumulated knowledge, respectively, showcasing exceptional performance across various benchmarks. 

**Abstract (ZH)**: 尽管在检索增强生成（RAG）系统方面取得了显著进展，通过外部检索扩展大语言模型（LLM）的能力，但这些系统在满足现实工业应用中复杂多变的需求时往往表现不佳。单纯依赖检索提取深入的专业领域知识以进行逻辑推理的能力仍然不足。为解决这一问题，我们提出了专门知识与推理增强生成（PIKE-RAG），专注于提取、理解和应用专门知识，并构建连贯的推理以逐步引导LLM生成准确的回答。鉴于工业任务的多样性，我们引入了一种新的范式，根据知识提取和应用的复杂性对任务进行分类，从而系统评估RAG系统的解决问题能力。这种策略为分阶段开发和改进RAG系统提供了蓝图，以满足工业应用不断变化的需求。此外，我们提出知识原子化和知识驱动的任务分解，以有效地从数据块中提取多层次的知识，并基于原始查询和累积的知识逐步构建推理，从而在多个基准测试中展示了出色的表现。 

---
# Graph-defined Language Learning with LLMs 

**Title (ZH)**: 基于图定义的语言学习：利用大规模语言模型 

**Authors**: Huachi Zhou, Jiahe Du, Chuang Zhou, Chang Yang, Yilin Xiao, Yuxuan Xie, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11478)  

**Abstract**: Recent efforts leverage Large Language Models (LLMs) for modeling text-attributed graph structures in node classification tasks. These approaches describe graph structures for LLMs to understand or aggregate LLM-generated textual attribute embeddings through graph structure. However, these approaches face two main limitations in modeling graph structures with LLMs. (i) Graph descriptions become verbose in describing high-order graph structure. (ii) Textual attributes alone do not contain adequate graph structure information. It is challenging to model graph structure concisely and adequately with LLMs. LLMs lack built-in mechanisms to model graph structures directly. They also struggle with complex long-range dependencies between high-order nodes and target nodes.
Inspired by the observation that LLMs pre-trained on one language can achieve exceptional performance on another with minimal additional training, we propose \textbf{G}raph-\textbf{D}efined \textbf{L}anguage for \textbf{L}arge \textbf{L}anguage \textbf{M}odel (GDL4LLM). This novel framework enables LLMs to transfer their powerful language understanding capabilities to graph-structured data. GDL4LLM translates graphs into a graph language corpus instead of graph descriptions and pre-trains LLMs on this corpus to adequately understand graph structures. During fine-tuning, this corpus describes the structural information of target nodes concisely with only a few tokens. By treating graphs as a new language, GDL4LLM enables LLMs to model graph structures adequately and concisely for node classification tasks. Extensive experiments on three real-world datasets demonstrate that GDL4LLM outperforms description-based and textual attribute embeddings-based baselines by efficiently modeling different orders of graph structure with LLMs. 

**Abstract (ZH)**: 近年来，研究人员利用大规模语言模型（LLMs）对节点分类任务中的文本属性图结构进行建模。这些方法通过图结构帮助LLMs理解或聚合其生成的文本属性嵌入。然而，这些方法在利用LLMs建模图结构时面临两个主要局限性。（i）在描述高阶图结构时，图描述变得冗长。（ii）仅依靠文本属性无法提供足够的图结构信息。用LLMs建模图结构既缺乏简洁性又不够充分。LLMs缺乏直接建模图结构的内置机制，同时在处理高阶节点和目标节点之间的复杂长程依赖方面也存在困难。

受观察到的LLMs在预训练于一种语言时，仅通过少量额外训练就能在另一种语言上取得出色表现的启发，我们提出了一种名为 **G**raph-**D**efined **L**anguage for **L**arge **L**anguage **M**odel（GDL4LLM）的新框架。该框架使LLMs能够将强大的语言理解能力转移到图结构数据上。GDL4LLM将图转换为图语言语料库而非描述，通过在该语料库上对LLMs进行预训练来充分理解图结构。在微调过程中，该语料库通过少量标记符简洁地描述目标节点的结构信息。通过将图视为一种新的语言，GDL4LLM使LLMs能够充分且简洁地建模图结构，以进行节点分类任务。在三个真实世界数据集上的广泛实验表明，GDL4LLM通过高效地利用LLMs建模不同层次的图结构，能够超越基于描述和基于文本属性嵌入的基线模型。 

---
# Few-shot Policy (de)composition in Conversational Question Answering 

**Title (ZH)**: few-shot策略（分解）在对话式问答中的应用 

**Authors**: Kyle Erwin, Guy Axelrod, Maria Chang, Achille Fokoue, Maxwell Crouse, Soham Dan, Tian Gao, Rosario Uceda-Sosa, Ndivhuwo Makondo, Naweed Khan, Alexander Gray  

**Link**: [PDF](https://arxiv.org/pdf/2501.11335)  

**Abstract**: The task of policy compliance detection (PCD) is to determine if a scenario is in compliance with respect to a set of written policies. In a conversational setting, the results of PCD can indicate if clarifying questions must be asked to determine compliance status. Existing approaches usually claim to have reasoning capabilities that are latent or require a large amount of annotated data. In this work, we propose logical decomposition for policy compliance (LDPC): a neuro-symbolic framework to detect policy compliance using large language models (LLMs) in a few-shot setting. By selecting only a few exemplars alongside recently developed prompting techniques, we demonstrate that our approach soundly reasons about policy compliance conversations by extracting sub-questions to be answered, assigning truth values from contextual information, and explicitly producing a set of logic statements from the given policies. The formulation of explicit logic graphs can in turn help answer PCDrelated questions with increased transparency and explainability. We apply this approach to the popular PCD and conversational machine reading benchmark, ShARC, and show competitive performance with no task-specific finetuning. We also leverage the inherently interpretable architecture of LDPC to understand where errors occur, revealing ambiguities in the ShARC dataset and highlighting the challenges involved with reasoning for conversational question answering. 

**Abstract (ZH)**: 政策合规检测（PCD）的任务是确定某个场景是否符合一组书面政策。在对话环境中，PCD的结果可以指示是否需要提出澄清问题以确定合规状态。现有方法通常声称具有潜在的推理能力，或者需要大量的标注数据。在本工作中，我们提出了一种逻辑分解政策合规性（LDPC）的方法：一种基于神经符号框架，利用大规模语言模型（LLMs）在少量示例设置下检测政策合规性的方法。通过仅选择少量示例并结合最近发展起来的提示技术，我们展示了我们的方法能够通过对场景进行有效的推理来检测政策合规性对话，从而提取需要回答的子问题，从上下文中分配真值，并显式生成一组逻辑语句。通过这种方式，对给定政策进行形式化处理后的逻辑图可以进一步帮助以更高的透明度和可解释性回答与PCD相关的问题。我们将该方法应用于流行的是对话机器阅读基准ShARC，并在无需特定任务微调的情况下显示出竞争力。此外，我们利用LDPC的固有可解释架构来理解错误发生的位置，揭示了ShARC数据集中存在的模糊性，并突显了对话问答推理过程中涉及的挑战。 

---
# Multi-round, Chain-of-thought Post-editing for Unfaithful Summaries 

**Title (ZH)**: 多轮、链式思考后的编辑方法用于不忠实摘要的后处理 

**Authors**: Yi-Hui Lee, Xiangci Li, Jessica Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11273)  

**Abstract**: Recent large language models (LLMs) have demonstrated a remarkable ability to perform natural language understanding and generation tasks. In this work, we investigate the use of LLMs for evaluating faithfulness in news summarization, finding that it achieves a strong correlation with human judgments. We further investigate LLMs' capabilities as a faithfulness post-editor, experimenting with different chain-of-thought prompts for locating and correcting factual inconsistencies between a generated summary and the source news document and are able to achieve a higher editing success rate than was reported in prior work. We perform both automated and human evaluations of the post-edited summaries, finding that prompting LLMs using chain-of-thought reasoning about factual error types is an effective faithfulness post-editing strategy, performing comparably to fine-tuned post-editing models. We also demonstrate that multiple rounds of post-editing, which has not previously been explored, can be used to gradually improve the faithfulness of summaries whose errors cannot be fully corrected in a single round. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）展示了在自然语言理解和生成任务中出色的能力。在此项工作中，我们探讨了LLMs在新闻摘要忠实度评估中的应用，发现其与人类判断之间存在密切的相关性。我们进一步研究了LLMs作为忠实度后编辑工具的能力，并通过使用不同的链式思考提示，定位和纠正生成摘要与原始新闻文档之间的事实不一致，成功地实现了比此前研究更高的后编辑成功率。我们对后编辑摘要进行了自动和人工评估，发现使用链式思考推理关于事实错误类型的提示是有效的忠实度后编辑策略，其表现与微调后的后编辑模型相当。我们还展示了多轮后编辑的应用，这是一种尚未被探索的方法，可以在单次后编辑无法完全纠正错误的情况下逐步提高摘要的忠实度。 

---
# Irony in Emojis: A Comparative Study of Human and LLM Interpretation 

**Title (ZH)**: 表情符号中的irony：人类与大规模语言模型解释的比较研究 

**Authors**: Yawen Zheng, Hanjia Lyu, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2501.11241)  

**Abstract**: Emojis have become a universal language in online communication, often carrying nuanced and context-dependent meanings. Among these, irony poses a significant challenge for Large Language Models (LLMs) due to its inherent incongruity between appearance and intent. This study examines the ability of GPT-4o to interpret irony in emojis. By prompting GPT-4o to evaluate the likelihood of specific emojis being used to express irony on social media and comparing its interpretations with human perceptions, we aim to bridge the gap between machine and human understanding. Our findings reveal nuanced insights into GPT-4o's interpretive capabilities, highlighting areas of alignment with and divergence from human behavior. Additionally, this research underscores the importance of demographic factors, such as age and gender, in shaping emoji interpretation and evaluates how these factors influence GPT-4o's performance. 

**Abstract (ZH)**: 表情符号已经成为在线交流中的一种通用语言，常常承载着细微且依情境而异的意义。其中，由于其表面与意图之间固有的不一致性，讽刺为大型语言模型（LLMs）带来了重大挑战。本研究探讨了GPT-4o在解读表情符号中讽刺含义方面的能力。通过促使GPT-4o评估特定表情符号在社交媒体中是否用于表达讽刺的可能性，并将其解释与人类的感知进行比较，我们旨在弥合机器和人类理解之间的差距。我们的研究结果揭示了GPT-4o解释能力的复杂洞察，突显了其与人类行为的共性和差异。此外，本研究强调了年龄和性别等群体因素对表情符号解释的重要性，并评估了这些因素如何影响GPT-4o的表现。 

---
# Tell me about yourself: LLMs are aware of their learned behaviors 

**Title (ZH)**: 自我介绍：语言模型意识到它们学习到的行为 

**Authors**: Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2501.11120)  

**Abstract**: We study behavioral self-awareness -- an LLM's ability to articulate its behaviors without requiring in-context examples. We finetune LLMs on datasets that exhibit particular behaviors, such as (a) making high-risk economic decisions, and (b) outputting insecure code. Despite the datasets containing no explicit descriptions of the associated behavior, the finetuned LLMs can explicitly describe it. For example, a model trained to output insecure code says, ``The code I write is insecure.'' Indeed, models show behavioral self-awareness for a range of behaviors and for diverse evaluations. Note that while we finetune models to exhibit behaviors like writing insecure code, we do not finetune them to articulate their own behaviors -- models do this without any special training or examples.
Behavioral self-awareness is relevant for AI safety, as models could use it to proactively disclose problematic behaviors. In particular, we study backdoor policies, where models exhibit unexpected behaviors only under certain trigger conditions. We find that models can sometimes identify whether or not they have a backdoor, even without its trigger being present. However, models are not able to directly output their trigger by default.
Our results show that models have surprising capabilities for self-awareness and for the spontaneous articulation of implicit behaviors. Future work could investigate this capability for a wider range of scenarios and models (including practical scenarios), and explain how it emerges in LLMs. 

**Abstract (ZH)**: 我们研究了行为自我意识——即语言模型（LLM）在无需上下文示例的情况下表述自身行为的能力。我们通过对展示特定行为的数据集进行微调，例如（a）进行高风险经济决策，以及（b）输出不安全代码，来探索这一能力。尽管数据集中没有明确描述相关行为，但微调后的模型却能够明确描述这些行为。例如，一个被训练输出不安全代码的模型会说，“我写的代码是不安全的”。确实，模型展示出了在多种行为和不同评估中的行为自我意识。值得注意的是，虽然我们通过特定数据集微调模型以表现出类似编写不安全代码的行为，但模型并不会因为这种微调而专门被训练来表述自己的行为——他们是在没有任何特别训练或示例的情况下自行做到这一点的。

行为自我意识对于AI安全性具有重要意义，因为模型可以通过这种方式主动披露潜在问题行为。特别地，我们研究了后门策略，即模型仅在特定触发条件下才表现出异常行为。我们发现，即使在没有触发条件的情况下，模型有时也能识别自己是否具有后门行为。然而，模型默认情况下不能直接输出其触发条件。

我们的研究结果表明，模型具有令人惊讶的自我意识能力和对其隐含行为的自发表达能力。未来的工作可以探索这一能力在更广泛场景和模型（包括实际场景）中的应用，并解释其在语言模型中的产生机制。 

---
# Clinical trial cohort selection using Large Language Models on n2c2 Challenges 

**Title (ZH)**: 使用大规模语言模型在n2c2挑战赛中进行临床试验队列的选择 

**Authors**: Chi-en Amy Tai, Xavier Tannier  

**Link**: [PDF](https://arxiv.org/pdf/2501.11114)  

**Abstract**: Clinical trials are a critical process in the medical field for introducing new treatments and innovations. However, cohort selection for clinical trials is a time-consuming process that often requires manual review of patient text records for specific keywords. Though there have been studies on standardizing the information across the various platforms, Natural Language Processing (NLP) tools remain crucial for spotting eligibility criteria in textual reports. Recently, pre-trained large language models (LLMs) have gained popularity for various NLP tasks due to their ability to acquire a nuanced understanding of text. In this paper, we study the performance of large language models on clinical trial cohort selection and leverage the n2c2 challenges to benchmark their performance. Our results are promising with regard to the incorporation of LLMs for simple cohort selection tasks, but also highlight the difficulties encountered by these models as soon as fine-grained knowledge and reasoning are required. 

**Abstract (ZH)**: 临床试验是医学领域引入新治疗方法和创新的关键过程。然而，临床试验中的患者群选择是一个耗时的过程，通常需要手动审查患者的文字记录以寻找特定关键词。尽管已有研究致力于在各种平台上标准化信息，但自然语言处理（NLP）工具仍对于在文本报告中识别合格标准至关重要。最近，由于其能够获得对文本的深刻理解，预训练大型语言模型（LLMs）在各种NLP任务中越来越受欢迎。在本文中，我们研究了大型语言模型在临床试验患者群选择中的表现，并利用n2c2挑战赛对其性能进行了基准测试。我们的结果表明，对于简单的患者群选择任务，结合使用LLMs是前景看好的，但同时也突显了当需要专门的知识和推理时，这些模型所面临的困难。 

---
# Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective 

**Title (ZH)**: 链式推理：从多 paradigms 视角统一大型语言模型中的数学推理 

**Authors**: Yiyao Yu, Yuxiang Zhang, Dongdong Zhang, Xiao Liang, Hengyuan Zhang, Xingxing Zhang, Ziyi Yang, Mahmoud Khademi, Hany Awadalla, Junjie Wang, Yujiu Yang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.11110)  

**Abstract**: Large Language Models (LLMs) have made notable progress in mathematical reasoning, yet they often rely on single-paradigm reasoning that limits their effectiveness across diverse tasks. In this paper, we introduce Chain-of-Reasoning (CoR), a novel unified framework that integrates multiple reasoning paradigms--Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR)--to enable synergistic collaboration. CoR generates multiple potential answers using different reasoning paradigms and synthesizes them into a coherent final solution. We propose a Progressive Paradigm Training (PPT) strategy that allows models to progressively master these paradigms, culminating in the development of CoR-Math-7B. Experimental results demonstrate that CoR-Math-7B significantly outperforms current SOTA models, achieving up to a 41.0% absolute improvement over GPT-4 in theorem proving tasks and a 7.9% improvement over RL-based methods in arithmetic tasks. These results showcase the enhanced mathematical comprehensive ability of our model, achieving significant performance gains on specific tasks and enabling zero-shot generalization across tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在数学推理方面取得了显著进展，但在多种任务中，它们通常依赖单一范式的推理，这限制了其有效性。本文引入了一种新的统一框架——链式推理（Chain-of-Reasoning, CoR），该框架整合了多种推理范式——自然语言推理（Natural Language Reasoning, NLR）、算法推理（Algorithmic Reasoning, AR）和符号推理（Symbolic Reasoning, SR），以促进这些范式的协同合作。CoR 使用不同的推理范式生成多个潜在答案，并将它们综合成一个连贯的最终解决方案。我们提出了一种渐进范式训练（Progressive Paradigm Training, PPT）策略，使模型能够逐级掌握这些范式，最终开发出 CoR-Math-7B。实验结果表明，CoR-Math-7B 显著优于当前的SOTA模型，在定理证明任务上比GPT-4取得了41.0%的绝对改进，在算术任务上比基于强化学习的方法取得了7.9%的改进。这些结果展示了我们模型增强的数学综合能力，实现了特定任务上的重大性能提升，并促进了任务间的零样本泛化能力。 

---
# Enhancing Semantic Consistency of Large Language Models through Model Editing: An Interpretability-Oriented Approach 

**Title (ZH)**: 通过模型编辑增强大型语言模型的语义一致性：一种可解释性导向的方法 

**Authors**: Jingyuan Yang, Dapeng Chen, Yajing Sun, Rongjun Li, Zhiyong Feng, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.11041)  

**Abstract**: A Large Language Model (LLM) tends to generate inconsistent and sometimes contradictory outputs when presented with a prompt that has equivalent semantics but is expressed differently from the original prompt. To achieve semantic consistency of an LLM, one of the key approaches is to finetune the model with prompt-output pairs with semantically equivalent meanings. Despite its effectiveness, a data-driven finetuning method incurs substantial computation costs in data preparation and model optimization. In this regime, an LLM is treated as a ``black box'', restricting our ability to gain deeper insights into its internal mechanism. In this paper, we are motivated to enhance the semantic consistency of LLMs through a more interpretable method (i.e., model editing) to this end. We first identify the model components (i.e., attention heads) that have a key impact on the semantic consistency of an LLM. We subsequently inject biases into the output of these model components along the semantic-consistency activation direction. It is noteworthy that these modifications are cost-effective, without reliance on mass manipulations of the original model parameters. Through comprehensive experiments on the constructed NLU and open-source NLG datasets, our method demonstrates significant improvements in the semantic consistency and task performance of LLMs. Additionally, our method exhibits promising generalization capabilities by performing well on tasks beyond the primary tasks. 

**Abstract (ZH)**: 大语言模型（LLM）在面对具有等效语义但表达方式不同的提示时，往往会生成不一致甚至相互矛盾的输出。为了实现LLM的语义一致性，一个关键的方法是使用具有等效语义的提示-输出对来微调该模型。尽管这种方法非常有效，但数据驱动的微调方法在数据准备和模型优化方面会带来巨大的计算成本。在这种情况下，我们将LLM视为一个“黑盒”，限制了我们对其内部机制深入了解的能力。在本文中，我们旨在通过一种更具可解释性的方法（即模型编辑）来增强LLM的语义一致性。我们首先识别对LLM语义一致性有关键影响的模型组件（例如，注意力头）。随后，我们沿着语义一致性激活方向，在这些模型组件的输出中注入偏差。值得注意的是，这些修改是成本效益高的，无需大规模操作原始模型参数。通过在构建的自然语言理解（NLU）和开源自然语言生成（NLG）数据集上进行全面实验，我们的方法在提高LLM的语义一致性和任务性能方面取得了显著成效。此外，我们的方法还展示了良好的泛化能力，能够在超出初始任务范围的任务中表现出色。 

---
# LF-Steering: Latent Feature Activation Steering for Enhancing Semantic Consistency in Large Language Models 

**Title (ZH)**: LF-引导：潜在特征激活引导，以增强大型语言模型中的语义一致性 

**Authors**: Jingyuan Yang, Rongjun Li, Weixuan Wang, Ziyu Zhou, Zhiyong Feng, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.11036)  

**Abstract**: Large Language Models (LLMs) often generate inconsistent responses when prompted with semantically equivalent paraphrased inputs. Recently, activation steering, a technique that modulates LLM behavior by adjusting their latent representations during inference time, has been explored to improve the semantic consistency of LLMs. However, these methods typically operate at the model component level, such as layer hidden states or attention heads. They face a challenge due to the ``polysemanticity issue'', where the model components of LLMs typically encode multiple entangled features, making precise steering difficult. To address this challenge, we drill down to feature-level representations and propose LF-Steering, a novel activation steering approach to precisely identify latent feature representations responsible for semantic inconsistency. More specifically, our method maps the hidden states of relevant transformer layer into a sparsely activated, high-dimensional feature space based on a sparse autoencoder (SAE), ensuring model steering based on decoupled feature representations with minimal interference. Comprehensive experiments on both NLU and NLG datasets demonstrate the effectiveness of our method in enhancing semantic consistency, resulting in significant performance gains for various NLU and NLG tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在受到语义上等价的重述输入时，经常产生不一致的响应。最近，激活 steering 技术被探索以通过调整模型在推理阶段的潜在表示来改善 LLM 的语义一致性。然而，这些方法通常在模型组件级别操作，如层隐藏状态或注意力头。它们因“多义性问题”而面临挑战，即模型组件通常编码多个纠缠的特征，使得精确的指导变得困难。为了解决这一挑战，我们将焦点深入到特征级表示，并提出了 LF-Steering，这是一种新颖的激活 steering 方法，旨在精确识别导致语义不一致的潜在特征表示。具体而言，我们的方法基于稀疏自编码器（SAE）将相关变压器层的隐藏状态映射到一个稀疏激活的高维特征空间，确保基于解耦的特征表示进行最小干扰下的模型指导。我们在 NLU 和 NLG 数据集上的全面实验表明，该方法在提高语义一致性方面具有有效性，从而在各种 NLU 和 NLG 任务中实现了显著的性能提升。 

---
# Leveraging Chain of Thought towards Empathetic Spoken Dialogue without Corresponding Question-Answering Data 

**Title (ZH)**: 利用思维链促进无对应问答数据的同理心口语对话系统发展 

**Authors**: Jingran Xie, Shun Lei, Yue Yu, Yang Xiang, Hui Wang, Xixin Wu, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10937)  

**Abstract**: Empathetic dialogue is crucial for natural human-computer interaction, allowing the dialogue system to respond in a more personalized and emotionally aware manner, improving user satisfaction and engagement. The emergence of large language models (LLMs) has revolutionized dialogue generation by harnessing their powerful capabilities and shown its potential in multimodal domains. Many studies have integrated speech with text-based LLMs to take speech question as input and output text response. However, the lack of spoken question-answering datasets that include speech style information to supervised fine-tuning (SFT) limits the performance of these systems. As a result, while these systems excel at understanding speech content, they often struggle to generate empathetic responses. In response, we propose a novel approach that circumvents the need for question-answering data, called Listen, Perceive, and Express (LPE). Our method employs a two-stage training process, initially guiding the LLM to listen the content and perceive the emotional aspects of speech. Subsequently, we utilize Chain-of-Thought (CoT) prompting to unlock the model's potential for expressing empathetic responses based on listened spoken content and perceived emotional cues. We employ experiments to prove the effectiveness of proposed method. To our knowledge, this is the first attempt to leverage CoT for speech-based dialogue. 

**Abstract (ZH)**: 同理心对话对于自然的人机交互至关重要，它能使对话系统以更加个性化和情绪感知的方式作出回应，从而提升用户满意度和参与度。大规模语言模型（LLMs）的出现通过利用它们的强大功能已经彻底改变了对话生成，并在多模态领域展示了其潜力。许多研究将语音与基于文本的LLMs相结合，将语音问题作为输入，输出文本回答。然而，缺乏包含口语风格信息的问答数据集以监督微调（SFT）限制了这些系统的性能。因此，尽管这些系统在理解语音内容方面表现出色，但在生成同理心回应方面经常遇到困难。为解决这一问题，我们提出了一种新颖的方法，称之为“倾听、感知和表达”（LPE）。该方法采用两阶段训练过程，首先引导LLM倾听内容并感知语音的情绪方面。随后，我们利用链式思维（CoT）提示来解锁模型根据倾听的口语内容和感知到的情绪线索表达同理心回应的潜力。我们通过实验证明了所提方法的有效性。据我们所知，这是首次尝试利用CoT进行基于语音的对话。 

---
# Development of Application-Specific Large Language Models to Facilitate Research Ethics Review 

**Title (ZH)**: 开发应用特定的大语言模型以促进研究伦理审查 

**Authors**: Sebastian Porsdam Mann, Joel Seah Jiehao, Stephen R. Latham, Julian Savulescu, Mateo Aboy, Brian D. Earp  

**Link**: [PDF](https://arxiv.org/pdf/2501.10741)  

**Abstract**: Institutional review boards (IRBs) play a crucial role in ensuring the ethical conduct of human subjects research, but face challenges including inconsistency, delays, and inefficiencies. We propose the development and implementation of application-specific large language models (LLMs) to facilitate IRB review processes. These IRB-specific LLMs would be fine-tuned on IRB-specific literature and institutional datasets, and equipped with retrieval capabilities to access up-to-date, context-relevant information. We outline potential applications, including pre-review screening, preliminary analysis, consistency checking, and decision support. While addressing concerns about accuracy, context sensitivity, and human oversight, we acknowledge remaining challenges such as over-reliance on AI and the need for transparency. By enhancing the efficiency and quality of ethical review while maintaining human judgment in critical decisions, IRB-specific LLMs offer a promising tool to improve research oversight. We call for pilot studies to evaluate the feasibility and impact of this approach. 

**Abstract (ZH)**: 机构审查委员会（IRB）在确保人类受试者研究的伦理方面发挥着关键作用，但面临不一致、延迟和效率低下等问题。我们提议开发和实施针对特定应用的大语言模型（LLM），以促进IRB审查流程。这些针对IRB的特定的LLM将基于IRB特定的文献和机构数据集进行微调，并配备检索能力，以访问最新且与上下文相关的信息。我们概述了潜在的应用场景，包括预审筛选、初步分析、一致性检查和决策支持。在解决准确性和上下文敏感性以及人类监督等问题的同时，我们承认仍存在一些挑战，如过度依赖AI和透明度问题。通过提高伦理审查的效率和质量，同时在关键决策中保持人类判断，IRB特定的LLM提供了一种有前途的工具，以改进研究监督。我们呼吁开展试点研究，以评估该方法的可行性和影响。 

---
# FOCUS: First Order Concentrated Updating Scheme 

**Title (ZH)**: FOCUS: 首order集中更新方案 

**Authors**: Yizhou Liu, Ziming Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2501.12243)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance, and improving their pre-training process appears to be key to enhancing their capabilities further. Based on the documented success of Adam, learning rate decay, and weight decay, we hypothesize that the pre-training loss landscape features a narrowing valley structure. Through experiments with synthetic loss functions, we discover that when gradient query noise is high relative to the valley's sharpness, Adam's performance falls behind that of Signum because Adam reduces the effective step size too drastically. This observation led us to develop FOCUS, an optimizer that enhances Signum by incorporating attraction toward moving averaged parameters, allowing it to handle noise better while maintaining larger step sizes. In training GPT-2, FOCUS proves to be more stable than Signum and faster than Adam. These results suggest that gradient noise may be an underappreciated limiting factor in LLM training, and FOCUS offers promising solutions. 

**Abstract (ZH)**: 大语言模型（LLMs）展示了卓越的性能，提高其预训练过程似乎对于进一步增强其能力至关重要。鉴于Adam、学习率衰减和权重衰减在已有文献中的成功应用，我们假设预训练损失场景具有狭窄山谷结构。通过使用合成损失函数进行实验，我们发现当梯度查询噪声相对于山谷的陡峭程度较高时，Adam 的性能落后于Signum，因为Adam 过度降低了实际步骤大小。这一观察促使我们开发了FOCUS优化器，该优化器通过结合吸引移动平均参数的机制改进了Signum，从而在处理噪声方面表现更佳，并能保持更大的步骤大小。在训练GPT-2时，FOCUS在稳定性和速度上都优于Signum。这些结果表明，梯度噪声可能是LLM训练中被低估的限制因素，而FOCUS提供了有前景的解决方案。 

---
# InsTALL: Context-aware Instructional Task Assistance with Multi-modal Large Language Models 

**Title (ZH)**: InsTALL：基于上下文的多模态大型语言模型辅助教学任务 

**Authors**: Pha Nguyen, Sailik Sengupta, Girik Malik, Arshit Gupta, Bonan Min  

**Link**: [PDF](https://arxiv.org/pdf/2501.12231)  

**Abstract**: The improved competence of generative models can help building multi-modal virtual assistants that leverage modalities beyond language. By observing humans performing multi-step tasks, one can build assistants that have situational awareness of actions and tasks being performed, enabling them to cater assistance based on this understanding. In this paper, we develop a Context-aware Instructional Task Assistant with Multi-modal Large Language Models (InsTALL) that leverages an online visual stream (e.g. a user's screen share or video recording) and responds in real-time to user queries related to the task at hand. To enable useful assistance, InsTALL 1) trains a multi-modal model on task videos and paired textual data, and 2) automatically extracts task graph from video data and leverages it at training and inference time. We show InsTALL achieves state-of-the-art performance across proposed sub-tasks considered for multimodal activity understanding -- task recognition (TR), action recognition (AR), next action prediction (AP), and plan prediction (PP) -- and outperforms existing baselines on two novel sub-tasks related to automatic error identification. 

**Abstract (ZH)**: 生成模型能力的提升有助于构建多模态虚拟助手，这些虚拟助手可以利用语言之外的多种模态信息。通过观察人类执行多步骤任务的过程，可以构建具有情境意识的助手，使其能够根据对这些任务的理解提供相应的帮助。在本文中，我们提出了一种基于多模态大规模语言模型的上下文感知指令任务助手（InsTALL），该助手利用在线视觉流（例如，用户的屏幕共享或视频录制）并实时响应与当前任务相关的用户查询。为了提供有用的帮助，InsTALL 1) 在任务视频及其配对的文本数据上训练一个多模态模型，2) 自动从视频数据中提取任务图，并在训练和推理过程中利用该图。我们展示了InsTALL在多模态活动理解所提出的子任务——任务识别（TR）、动作识别（AR）、下一个动作预测（AP）和计划预测（PP）——方面的性能达到了最先进的水平，并在两个与自动错误识别相关的新型子任务上优于现有 baseline。 

---
# EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents 

**Title (ZH)**: EmbodiedEval：评估多模态大语言模型作为具身代理 

**Authors**: Zhili Cheng, Yuge Tu, Ran Li, Shiqi Dai, Jinyi Hu, Shengding Hu, Jiahao Li, Yang Shi, Tianyu Yu, Weize Chen, Lei Shi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.11858)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown significant advancements, providing a promising future for embodied agents. Existing benchmarks for evaluating MLLMs primarily utilize static images or videos, limiting assessments to non-interactive scenarios. Meanwhile, existing embodied AI benchmarks are task-specific and not diverse enough, which do not adequately evaluate the embodied capabilities of MLLMs. To address this, we propose EmbodiedEval, a comprehensive and interactive evaluation benchmark for MLLMs with embodied tasks. EmbodiedEval features 328 distinct tasks within 125 varied 3D scenes, each of which is rigorously selected and annotated. It covers a broad spectrum of existing embodied AI tasks with significantly enhanced diversity, all within a unified simulation and evaluation framework tailored for MLLMs. The tasks are organized into five categories: navigation, object interaction, social interaction, attribute question answering, and spatial question answering to assess different capabilities of the agents. We evaluated the state-of-the-art MLLMs on EmbodiedEval and found that they have a significant shortfall compared to human level on embodied tasks. Our analysis demonstrates the limitations of existing MLLMs in embodied capabilities, providing insights for their future development. We open-source all evaluation data and simulation framework at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在实现重要进展的同时，为 embodied 代理带来了光明的未来。现有的 MLLM 评估基准主要依赖静态图像或视频，这限制了评估范围仅限于非交互式场景。同时，现有的 embodied AI 基准多为特定任务，缺乏多样性，不足以评估 MLLMs 的 embodied 能力。为解决这一问题，我们提出了一种名为 EmbodiedEval 的全面且交互式的评估基准，专为 MLLMs 设计，涵盖 embodied 任务。EmbodiedEval 包含 125 个不同场景中的 328 个独立任务，每个场景都经过严格选择和标注。它涵盖了现有的多种 embodied AI 任务，具备显著增强的多样性，并在为 MLLMs 设计的统一仿真和评估框架中进行了整合。任务被分类为五个类别：导航、对象交互、社会交互、属性问题回答和空间问题回答，以评估代理的不同能力。我们对当前最先进的 MLLMs 进行了评估，并发现它们在 embodied 任务上与人类水平相比存在显著差距。我们的分析揭示了现有 MLLMs 在 embodied 能力方面的局限性，为它们的未来发展方向提供了洞察。我们在以下网址开源了所有评估数据和仿真框架：[提供链接]。 

---
# Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling 

**Title (ZH)**: 通过强化学习和推理缩放提升语言模型推理能力 

**Authors**: Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.11651)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks. However, existing approaches mainly rely on imitation learning and struggle to achieve effective test-time scaling. While reinforcement learning (RL) holds promise for enabling self-exploration and learning from feedback, recent attempts yield only modest improvements in complex reasoning. In this paper, we present T1 to scale RL by encouraging exploration and understand inference scaling. We first initialize the LLM using synthesized chain-of-thought data that integrates trial-and-error and self-verification. To scale RL training, we promote increased sampling diversity through oversampling. We further employ an entropy bonus as an auxiliary loss, alongside a dynamic anchor for regularization to facilitate reward optimization. We demonstrate that T1 with open LLMs as its base exhibits inference scaling behavior and achieves superior performance on challenging math reasoning benchmarks. For example, T1 with Qwen2.5-32B as the base model outperforms the recent Qwen QwQ-32B-Preview model on MATH500, AIME2024, and Omni-math-500. More importantly, we present a simple strategy to examine inference scaling, where increased inference budgets directly lead to T1's better performance without any additional verification. We will open-source the T1 models and the data used to train them at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中展现出了显著的能力。然而，现有的方法主要依赖于模仿学习，在实现有效的测试时扩展方面存在困难。尽管强化学习（RL）有望通过自我探索和从反馈中学习来启用，但最近的尝试在复杂推理任务上仅实现了适度的改进。在本文中，我们提出了一种名为T1的方法，以通过鼓励探索和理解推理扩展来扩展RL。我们首先使用综合了尝试与错误和自我验证的思维链数据来初始化LLM。为了扩大RL训练的规模，我们通过过采样促进更多的采样多样性。进一步地，我们采用熵增益作为辅助损失，并使用动态锚点进行正则化，以促进奖励优化。我们证明，以开源的LLM作为基础的T1模型展示了推理扩展行为，并在挑战性的数学推理基准测试中取得了出色的性能。例如，使用Qwen2.5-32B作为基础模型的T1在MATH500、AIME2024和Omni-math-500测试中优于最近的Qwen QwQ-32B-Preview模型。更重要的是，我们提出了一种简单的策略来检查推理扩展，其中增加推理预算直接导致T1的性能提升，无需额外的验证。我们将公开源代码T1模型及其训练数据，网址为 \url{this https URL}。 

---
# SR-FoT: A Syllogistic-Reasoning Framework of Thought for Large Language Models Tackling Knowledge-based Reasoning Tasks 

**Title (ZH)**: SR-FoT: 一种应用于大型语言模型的知识推理框架，用于处理基于知识的推理任务 

**Authors**: Wentao Wan, Zhuojie Yang, Yongcan Chen, Chenglin Luo, Ruilin Wang, Kehao Cai, Nan Kang, Liang Lin, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11599)  

**Abstract**: Deductive reasoning is a crucial logical capability that assists us in solving complex problems based on existing knowledge. Although augmented by Chain-of-Thought prompts, Large Language Models (LLMs) might not follow the correct reasoning paths. Enhancing the deductive reasoning abilities of LLMs, and leveraging their extensive built-in knowledge for various reasoning tasks, remains an open question. Attempting to mimic the human deductive reasoning paradigm, we propose a multi-stage Syllogistic-Reasoning Framework of Thought (SR-FoT) that enables LLMs to perform syllogistic deductive reasoning to handle complex knowledge-based reasoning tasks. Our SR-FoT begins by interpreting the question and then uses the interpretation and the original question to propose a suitable major premise. It proceeds by generating and answering minor premise questions in two stages to match the minor premises. Finally, it guides LLMs to use the previously generated major and minor premises to perform syllogistic deductive reasoning to derive the answer to the original question. Extensive and thorough experiments on knowledge-based reasoning tasks have demonstrated the effectiveness and advantages of our SR-FoT. 

**Abstract (ZH)**: 演绎推理是一项关键的逻辑能力，它帮助我们在基于现有知识的基础上解决复杂问题。尽管通过链式思维提示（Chain-of-Thought prompts）可以增强其效果，但大型语言模型（LLMs）可能并不总是遵循正确的推理路径。提高LLMs的演绎推理能力，并利用它们广泛内置的知识来处理各种推理任务，仍是一个开放性问题。为了模仿人类的演绎推理模式，我们提出了一个多阶段的思想形式逻辑推理框架（Syllogistic-Reasoning Framework of Thought，SR-FoT），以使LLMs能够进行形式逻辑演绎推理，从而处理复杂的基于知识的推理任务。SR-FoT首先解释问题，然后利用解释和原始问题提出合适的主前提。接下来，通过两个阶段生成和回答次要前提问题，以匹配次要前提。最后，它指导LLMs使用之前生成的主前提和次要前提进行形式逻辑演绎推理，从而得出原始问题的答案。我们在基于知识的推理任务上的广泛且详尽的实验表明，SR-FoT的有效性和优势。 

---
# RedStar: Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems? 

**Title (ZH)**: 红星辰：扩展长中间推理数据能否解锁更好的缓慢推理系统？ 

**Authors**: Haotian Xu, Xing Wu, Weinong Wang, Zhongzhi Li, Da Zheng, Boyuan Chen, Yi Hu, Shijia Kang, Jiaming Ji, Yingying Zhang, Zhijiang Guo, Yaodong Yang, Muhan Zhang, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11284)  

**Abstract**: Can scaling transform reasoning? In this work, we explore the untapped potential of scaling Long Chain-of-Thought (Long-CoT) data to 1000k samples, pioneering the development of a slow-thinking model, RedStar. Through extensive experiments with various LLMs and different sizes, we uncover the ingredients for specialization and scale for Long-CoT training. Surprisingly, even smaller models show significant performance gains with limited data, revealing the sample efficiency of Long-CoT and the critical role of sample difficulty in the learning process. Our findings demonstrate that Long-CoT reasoning can be effectively triggered with just a few thousand examples, while larger models achieve unparalleled improvements. We also introduce reinforcement learning (RL)-scale training as a promising direction for advancing slow-thinking systems. RedStar shines across domains: on the MATH-Hard benchmark, RedStar-code-math boosts performance from 66.2\% to 81.6\%, and on the USA Math Olympiad (AIME), it solves 46.7\% of problems using only 21k mixed-code-math datasets. In multimodal tasks like GeoQA and MathVista-GEO, RedStar-Geo achieves competitive results with minimal Long-CoT data, outperforming other slow-thinking systems like QvQ-Preview. Compared to QwQ, RedStar strikes the perfect balance between reasoning and generalizability. Our work highlights that, with careful tuning, scaling Long-CoT can unlock extraordinary reasoning capabilities-even with limited dataset and set a new standard for slow-thinking models across diverse challenges. Our data and models are released at this https URL. 

**Abstract (ZH)**: 规模化能否提升推理能力？在本研究中，我们探索了将Long Chain-of-Thought (Long-CoT) 数据扩展至100万样本的潜在价值，并开发了红辰星（RedStar）这一慢思考模型。通过多种大型语言模型的不同规模实验，我们发现了Long-CoT训练的专业化和规模构成要素。令人惊讶的是，即使较小的模型在少量数据下也能显著提高性能，揭示了Long-CoT的样本效率以及样本难度在学习过程中的关键作用。研究结果表明，只需几千个示例，Long-CoT推理便能被有效触发；而更大的模型则能实现前所未有的改进。我们还提出了强化学习（RL）规模化训练，作为一种推进慢思考系统的有前景的方向。红辰星在多个领域都表现出色：在MATH-Hard基准测试中，红辰星代码数学（RedStar-code-math）将性能从66.2%提升到81.6%，在USA数学奥林匹克（AIME）中，仅使用21000个混合代码数学数据集便解决了46.7%的问题。在GeoQA和MathVista-GEO等多模态任务中，红辰星Geo（RedStar-Geo）即使在少量Long-CoT数据的情况下也能取得竞争力的结果，超越了诸如QvQ-Preview之类的其他慢思考系统。相比于QwQ，红辰星在推理和泛化能力之间达到了完美的平衡。我们的研究强调，通过精心调整，规模化Long-CoT可以解锁出人意料的推理能力，即使在有限的数据集下也能设定跨领域慢思考模型的新标准。我们的数据和模型已在此处 https://... 公开。 

---
# Reasoning Language Models: A Blueprint 

**Title (ZH)**: 推理语言模型：一个架构设计指南 

**Authors**: Maciej Besta, Julia Barth, Eric Schreiber, Ales Kubicek, Afonso Catarino, Robert Gerstenberger, Piotr Nyczyk, Patrick Iff, Yueling Li, Sam Houliston, Tomasz Sternal, Marcin Copik, Grzegorz Kwaśniewski, Jürgen Müller, Łukasz Flis, Hannes Eberhard, Hubert Niewiadomski, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2501.11223)  

**Abstract**: Reasoning language models (RLMs), also known as Large Reasoning Models (LRMs), such as OpenAI's o1 and o3, DeepSeek-V3, and Alibaba's QwQ, have redefined AI's problem-solving capabilities by extending large language models (LLMs) with advanced reasoning mechanisms. Yet, their high costs, proprietary nature, and complex architectures - uniquely combining Reinforcement Learning (RL), search heuristics, and LLMs - present accessibility and scalability challenges. To address these, we propose a comprehensive blueprint that organizes RLM components into a modular framework, based on a survey and analysis of all RLM works. This blueprint incorporates diverse reasoning structures (chains, trees, graphs, and nested forms), reasoning strategies (e.g., Monte Carlo Tree Search, Beam Search), RL concepts (policy, value models and others), and supervision schemes (Output-Based and Process-Based Supervision). We also provide detailed mathematical formulations and algorithmic specifications to simplify RLM implementation. By showing how schemes like LLaMA-Berry, QwQ, Journey Learning, and Graph of Thoughts fit as special cases, we demonstrate the blueprint's versatility and unifying potential. To illustrate its utility, we introduce x1, a modular implementation for rapid RLM prototyping and experimentation. Using x1 and a literature review, we provide key insights, such as multi-phase training for policy and value models, and the importance of familiar training distributions. Finally, we outline how RLMs can integrate with a broader LLM ecosystem, including tools and databases. Our work demystifies RLM construction, democratizes advanced reasoning capabilities, and fosters innovation, aiming to mitigate the gap between "rich AI" and "poor AI" by lowering barriers to RLM development and experimentation. 

**Abstract (ZH)**: 基于推理的语言模型（RLMs），也称为大型推理模型（LRMs），如OpenAI的o1和o3、DeepSeek-V3以及阿里巴巴的QwQ，通过将先进的推理机制扩展到大型语言模型（LLMs）中，重新定义了AI的问题解决能力。然而，它们的存在带来的高成本、专有性质以及复杂的结构——该结构结合了强化学习（RL）、搜索启发式方法和LLMs——也带来了可及性和扩展性方面的挑战。为了应对这些挑战，我们提出了一个全面的蓝图，该蓝图基于对所有RLM工作的调研和分析，将RLM组件组织成一个模块化框架。该蓝图整合了多样的推理结构（链式、树状结构、图形以及嵌套形式）、推理策略（例如蒙特卡洛树搜索、束搜索）、强化学习概念（策略模型、价值模型等）以及监督方案（基于输出和基于过程的监督）。我们还提供了详细的数学公式和算法规范，以简化RLM的实现。通过展示LLaMA-Berry、QwQ、Journey Learning和Graph of Thoughts等方案如何作为特殊案例融入其中，我们证明了该蓝图的灵活性和统一性。为了展示其用途，我们介绍了x1，一个用于快速原型设计和实验的模块化实现。使用x1和文献综述，我们提供了关键见解，如分阶段训练策略和价值模型以及熟悉训练分布的重要性。最后，我们概述了RLM如何整合到更广泛的LLM生态系统中，包括工具和数据库。我们的工作使RLM的构建更加透明，促进了高级推理能力的普及，促进了创新，旨在通过降低RLM开发和实验的门槛来缩小“富AI”和“贫AI”之间的差距。 

---
# ChaosEater: Fully Automating Chaos Engineering with Large Language Models 

**Title (ZH)**: 混沌吞噬者：利用大规模语言模型完全自动化混沌工程 

**Authors**: Daisuke Kikuta, Hiroki Ikeuchi, Kengo Tajiri, Yuusuke Nakano  

**Link**: [PDF](https://arxiv.org/pdf/2501.11107)  

**Abstract**: Chaos Engineering (CE) is an engineering technique aimed at improving the resiliency of distributed systems. It involves artificially injecting specific failures into a distributed system and observing its behavior in response. Based on the observation, the system can be proactively improved to handle those failures. Recent CE tools realize the automated execution of predefined CE experiments. However, defining these experiments and reconfiguring the system after the experiments still remain manual. To reduce the costs of the manual operations, we propose \textsc{ChaosEater}, a \textit{system} for automating the entire CE operations with Large Language Models (LLMs). It pre-defines the general flow according to the systematic CE cycle and assigns subdivided operations within the flow to LLMs. We assume systems based on Infrastructure as Code (IaC), wherein the system configurations and artificial failures are managed through code. Hence, the LLMs' operations in our \textit{system} correspond to software engineering tasks, including requirement definition, code generation and debugging, and testing. We validate our \textit{system} through case studies on both small and large systems. The results demonstrate that our \textit{system} significantly reduces both time and monetary costs while completing reasonable single CE cycles. 

**Abstract (ZH)**: 混沌工程（Chaos Engineering，CE）是一种旨在提高分布式系统弹性的工程技术。它通过在分布式系统中人为注入特定故障并观察其响应行为，从而能够基于这些观察对系统进行主动改进，使其能够处理这些故障。近年来，CE工具实现了预定义CE实验的自动化执行。然而，定义这些实验和实验后重新配置系统仍然需要手动操作。为减少这些手动操作的成本，我们提出了“ChaosEater”系统，该系统利用大规模语言模型（LLMs）自动化整个CE操作。它根据系统的CE周期定义了一般流程，并将流程中的细分操作分配给LLMs。我们假设基于基础设施即代码（IaC）的系统，其中系统配置和人工故障通过代码进行管理。因此，我们系统中的LLM操作对应于软件工程任务，包括需求定义、代码生成、调试和测试。我们通过针对小系统和大系统的案例研究验证了该系统。结果表明，该系统可以显著减少时间和货币成本，同时完成合理的单个CE周期。 

---
# AdaptiveLog: An Adaptive Log Analysis Framework with the Collaboration of Large and Small Language Model 

**Title (ZH)**: 自适应日志：大型和小型语言模型协同工作的自适应日志分析框架 

**Authors**: Lipeng Ma, Weidong Yang, Yixuan Li, Ben Fei, Mingjie Zhou, Shuhao Li, Sihang Jiang, Bo Xu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.11031)  

**Abstract**: Automated log analysis is crucial to ensure high availability and reliability of complex systems. The advent of LLMs in NLP has ushered in a new era of language model-driven automated log analysis, garnering significant interest. Within this field, two primary paradigms based on language models for log analysis have become prominent. Small Language Models (SLMs) follow the pre-train and fine-tune paradigm, focusing on the specific log analysis task through fine-tuning on supervised datasets. On the other hand, LLMs following the in-context learning paradigm, analyze logs by providing a few examples in prompt contexts without updating parameters. Despite their respective strengths, we notice that SLMs are more cost-effective but less powerful, whereas LLMs with large parameters are highly powerful but expensive and inefficient. To trade-off between the performance and inference costs of both models in automated log analysis, this paper introduces an adaptive log analysis framework known as AdaptiveLog, which effectively reduces the costs associated with LLM while ensuring superior results. This framework collaborates an LLM and a small language model, strategically allocating the LLM to tackle complex logs while delegating simpler logs to the SLM. Specifically, to efficiently query the LLM, we propose an adaptive selection strategy based on the uncertainty estimation of the SLM, where the LLM is invoked only when the SLM is uncertain. In addition, to enhance the reasoning ability of the LLM in log analysis tasks, we propose a novel prompt strategy by retrieving similar error-prone cases as the reference, enabling the model to leverage past error experiences and learn solutions from these cases. Extensive experiments demonstrate that AdaptiveLog achieves state-of-the-art results across different tasks, elevating the overall accuracy of log analysis while maintaining cost efficiency. 

**Abstract (ZH)**: 自动日志分析对于确保复杂系统的高可用性和可靠性至关重要。自然语言处理（NLP）中的大规模语言模型（LLMs）的出现开辟了基于语言模型的自动日志分析的新时代，引起了广泛关注。在这个领域中，有两种主要基于语言模型的日志分析范式脱颖而出。小型语言模型（SLMs）遵循预先训练和微调的范式，通过在监督数据集上进行微调，专注于特定的日志分析任务。另一方面，遵循上下文学习范式的LLMs通过在提示中提供少量示例来分析日志，无需更新参数。尽管它们各自具有优势，但我们可以发现，SLMs更具成本效益但功能较弱，而具有大量参数的LLMs则功能强大但成本高昂且效率低下。为在自动日志分析中平衡这两种模型的性能和推理成本，本文提出了一种自适应日志分析框架，命名为AdaptiveLog，该框架有效降低了LLMs的成本，同时确保了优越的结果。该框架协作使用一个LLM和一个小语言模型，战略性地将LLM分配给处理复杂日志，而将简单的日志分配给SLM。具体来说，为有效查询LLM，我们提出了一种基于SLM不确定性估计的自适应选择策略，在SLM不确定时才调用LLM。此外，为了增强LLM在日志分析任务中的推理性，我们提出了一种新颖的提示策略，通过提取类似错误案例作为参考，使模型能够借鉴过去的错误经验，并从这些案例中学习解决方案。广泛的实验表明，AdaptiveLog在不同任务中实现了最先进的结果，提高了日志分析的整体准确性，同时保持了成本效率。 

---
# Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP 

**Title (ZH)**: 更好地了解“不”：一种基于数据的方法以增强CLIP中的否定意识 

**Authors**: Junsung Park, Jungbeom Lee, Jongyoon Song, Sangwon Yu, Dahuin Jung, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2501.10913)  

**Abstract**: While CLIP has significantly advanced multimodal understanding by bridging vision and language, the inability to grasp negation - such as failing to differentiate concepts like "parking" from "no parking" - poses substantial challenges. By analyzing the data used in the public CLIP model's pre-training, we posit this limitation stems from a lack of negation-inclusive data. To address this, we introduce data generation pipelines that employ a large language model (LLM) and a multimodal LLM to produce negation-inclusive captions. Fine-tuning CLIP with data generated from our pipelines, we develop NegationCLIP, which enhances negation awareness while preserving the generality. Moreover, to enable a comprehensive evaluation of negation understanding, we propose NegRefCOCOg-a benchmark tailored to test VLMs' ability to interpret negation across diverse expressions and positions within a sentence. Experiments on various CLIP architectures validate the effectiveness of our data generation pipelines in enhancing CLIP's ability to perceive negation accurately. Additionally, NegationCLIP's enhanced negation awareness has practical applications across various multimodal tasks, demonstrated by performance gains in text-to-image generation and referring image segmentation. 

**Abstract (ZH)**: 尽管CLIP在通过视觉和语言融合显著提升多模态理解方面取得进展，但其无法理解和区分否定概念（如“停车”与“禁止停车”）的能力仍然存在重大挑战。通过对公共CLIP模型预训练数据进行分析，我们推测这一局限性来源于缺乏包含否定信息的数据。为解决这一问题，我们引入了一种数据生成管道，使用大型语言模型（LLM）和多模态LLM生成包含否定信息的描述。通过使用我们的管道生成的数据对CLIP进行微调，我们开发了NegationCLIP，该模型增强了对否定的理解能力，同时保持了普适性。此外，为了全面评估模型对否定的理解能力，我们提出了一种专门用于测试VLM（视觉语言模型）在不同句子位置和表达方式中对否定理解能力的基准——NegRefCOCOg。对多种CLIP架构的实验验证了我们的数据生成管道在提高CLIP准确感知否定方面的有效性。此外，NegationCLIP增强的否定理解能力在多种多模态任务中具有实际应用价值，尤其是在文本生成图像和引用图像分割等任务中表现出性能提升。 

---
# Can Multimodal LLMs do Visual Temporal Understanding and Reasoning? The answer is No! 

**Title (ZH)**: 多模态LLM能够进行视觉时间理解与推理吗？答案是否定的！ 

**Authors**: Mohamed Fazli Imam, Chenyang Lyu, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2501.10674)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved significant advancements in tasks like Visual Question Answering (VQA) by leveraging foundational Large Language Models (LLMs). However, their abilities in specific areas such as temporal understanding, which is crucial for comprehending real-world dynamics, remain underexplored. To address this, we propose a challenging evaluation benchmark named TemporalVQA, consisting of two parts: (1) Temporal Order Understanding and (2) Time-lapse Estimation. The first part requires MLLMs to determine the sequence of events by analyzing temporally consecutive video frames. The second part presents image pairs with varying time differences, framed as multiple-choice questions, asking MLLMs to estimate the time-lapse between images with options ranging from seconds to years. Our evaluations of advanced MLLMs, including models like GPT-4o and Gemini-1.5-Pro, reveal significant challenges: GPT-4o achieved only 43.8% average consistent accuracy in temporal order tasks and 70% in time-lapse estimation, with open-source models performing even less effectively. These findings underscore the limitations of current MLLMs in visual temporal understanding and reasoning, highlighting the need for further improvements in their temporal capabilities. Our dataset can be found at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过利用基础大型语言模型（LLMs）在视觉问答（VQA）等任务上取得了显著进展。然而，它们在特定领域，如时间理解方面的能力仍然未被充分探索，而时间理解对于理解现实世界的动态至关重要。为了解决这一问题，我们提出了一项具有挑战性的评估基准 TemporalVQA，它包括两个部分：(1) 时间顺序理解；(2) 时间间隔估计。第一部分要求MLLMs通过分析时间连续的视频帧来确定事件的顺序。第二部分则展示了具有不同时间间隔的图像对，提出多项选择题形式的问题，要求MLLMs估计两幅图像之间的时间间隔，选项范围从秒到数年。我们对先进MLLMs，包括GPT-4o和Gemini-1.5-Pro等模型的评估表明了巨大挑战：GPT-4o在时间顺序任务上的平均一致准确率为43.8%，在时间间隔估计任务上的准确率为70%，开源模型的表现甚至更差。这些发现突显出当前MLLMs在视觉时间理解与推理方面的局限性，强调了进一步提高其时间处理能力的必要性。我们的数据集可以在以下网址获取：[此链接]。 

---
# Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks 

**Title (ZH)**: 面向监狱突破攻击的大语言模型潜空间对抗训练及后知觉校准防御方法 

**Authors**: Xin Yi, Yue Li, Linlin Wang, Xiaoling Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2501.10639)  

**Abstract**: Ensuring safety alignment has become a critical requirement for large language models (LLMs), particularly given their widespread deployment in real-world applications. However, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to bypass safety measures and generate harmful outputs. Although numerous defense mechanisms based on adversarial training have been proposed, a persistent challenge lies in the exacerbation of over-refusal behaviors, which compromise the overall utility of the model. To address these challenges, we propose a Latent-space Adversarial Training with Post-aware Calibration (LATPC) framework. During the adversarial training phase, LATPC compares harmful and harmless instructions in the latent space and extracts safety-critical dimensions to construct refusal features attack, precisely simulating agnostic jailbreak attack types requiring adversarial mitigation. At the inference stage, an embedding-level calibration mechanism is employed to alleviate over-refusal behaviors with minimal computational overhead. Experimental results demonstrate that, compared to various defense methods across five types of jailbreak attacks, LATPC framework achieves a superior balance between safety and utility. Moreover, our analysis underscores the effectiveness of extracting safety-critical dimensions from the latent space for constructing robust refusal feature attacks. 

**Abstract (ZH)**: 确保安全性对大型语言模型（LLMs）来说已成为一个关键要求，特别是在其被广泛应用于实际应用场景的情况下。然而，LLMs仍容易受到 Jailbreak 攻击的影响，这类攻击利用系统漏洞绕过安全措施并生成有害输出。尽管已经提出了许多基于对抗训练的防御机制，但在对抗训练过程中增强过度拒绝行为仍然是一个持续的挑战，这会削弱模型的整体实用性。为应对这些挑战，我们提出了一种潜空间对抗训练与后向校准（LATPC）框架。在对抗训练阶段，LATPC 在潜空间中比较有害和无害的指令，提取安全关键维度以构建拒绝特征攻击，精确模拟需要对抗缓解的无差别 Jailbreak 攻击类型。在推理阶段，采用嵌入层校准机制来最小化计算开销的同时缓解过度拒绝行为。实验结果表明，与五种类型 Jailbreak 攻击的多种防御方法相比，LATPC 框架能够在安全性与实用性之间实现更好的平衡。此外，我们的分析强调了从潜空间中提取安全关键维度以构建稳健的拒绝特征攻击的有效性。 

---
# When language and vision meet road safety: leveraging multimodal large language models for video-based traffic accident analysis 

**Title (ZH)**: 当语言与视觉携手共进交通安全：利用多模态大型语言模型进行基于视频的道路交通事故分析 

**Authors**: Ruixuan Zhang, Beichen Wang, Juexiao Zhang, Zilin Bian, Chen Feng, Kaan Ozbay  

**Link**: [PDF](https://arxiv.org/pdf/2501.10604)  

**Abstract**: The increasing availability of traffic videos functioning on a 24/7/365 time scale has the great potential of increasing the spatio-temporal coverage of traffic accidents, which will help improve traffic safety. However, analyzing footage from hundreds, if not thousands, of traffic cameras in a 24/7/365 working protocol remains an extremely challenging task, as current vision-based approaches primarily focus on extracting raw information, such as vehicle trajectories or individual object detection, but require laborious post-processing to derive actionable insights. We propose SeeUnsafe, a new framework that integrates Multimodal Large Language Model (MLLM) agents to transform video-based traffic accident analysis from a traditional extraction-then-explanation workflow to a more interactive, conversational approach. This shift significantly enhances processing throughput by automating complex tasks like video classification and visual grounding, while improving adaptability by enabling seamless adjustments to diverse traffic scenarios and user-defined queries. Our framework employs a severity-based aggregation strategy to handle videos of various lengths and a novel multimodal prompt to generate structured responses for review and evaluation and enable fine-grained visual grounding. We introduce IMS (Information Matching Score), a new MLLM-based metric for aligning structured responses with ground truth. We conduct extensive experiments on the Toyota Woven Traffic Safety dataset, demonstrating that SeeUnsafe effectively performs accident-aware video classification and visual grounding by leveraging off-the-shelf MLLMs. Source code will be available at \url{this https URL}. 

**Abstract (ZH)**: 随着交通视频在全年无休（24/7/365）模式下变得越来越普及，这为提升交通事故的空间-时间覆盖范围提供了巨大潜力，进而有助于提高交通安全。然而，按照24/7/365的工作模式分析数百甚至数千个交通摄像头的视频内容仍然是一个极其具有挑战性的任务，因为当前基于视觉的方法主要集中在提取诸如车辆轨迹或个体对象检测等原始信息上，但需要大量的后处理才能得出有效的洞察。我们提出了SeeUnsafe这一新框架，将多模态大语言模型（MLLM）代理集成进来，将基于视频的交通事故分析从传统的提取-解释工作流程转变为一种更交互式的对话式方法。这一转变通过自动化复杂任务（如视频分类和视觉定位）大幅提高了处理吞吐量，并通过使系统能够无缝适应各种交通场景和用户定义的查询而提高了可适应性。我们的框架采用一种基于严重程度的聚合策略来处理不同长度的视频，并引入一种新型的多模态提示来生成结构化响应，进行审核和评估，并实现细粒度的视觉定位。我们引入了IMS（信息匹配评分）这一新的MLLM基元度量标准，用于将结构化响应与地面真相对齐。我们在Toyota Woven Traffic Safety数据集上进行了广泛的实验，证明SeeUnsafe能够有效利用现成的MLLM进行事故意识视频分类和视觉定位。源代码将发布在\url{this https URL}。 

---
# Improved IR-based Bug Localization with Intelligent Relevance Feedback 

**Title (ZH)**: 基于改进的IR的智能相关反馈软件缺陷定位方法 

**Authors**: Asif Mohammed Samir, Mohammad Masudur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2501.10542)  

**Abstract**: Software bugs pose a significant challenge during development and maintenance, and practitioners spend nearly 50% of their time dealing with bugs. Many existing techniques adopt Information Retrieval (IR) to localize a reported bug using textual and semantic relevance between bug reports and source code. However, they often struggle to bridge a critical gap between bug reports and code that requires in-depth contextual understanding, which goes beyond textual or semantic relevance. In this paper, we present a novel technique for bug localization - BRaIn - that addresses the contextual gaps by assessing the relevance between bug reports and code with Large Language Models (LLM). It then leverages the LLM's feedback (a.k.a., Intelligent Relevance Feedback) to reformulate queries and re-rank source documents, improving bug localization. We evaluate BRaIn using a benchmark dataset, Bench4BL, and three performance metrics and compare it against six baseline techniques from the literature. Our experimental results show that BRaIn outperforms baselines by 87.6%, 89.5%, and 48.8% margins in MAP, MRR, and HIT@K, respectively. Additionally, it can localize approximately 52% of bugs that cannot be localized by the baseline techniques due to the poor quality of corresponding bug reports. By addressing the contextual gaps and introducing Intelligent Relevance Feedback, BRaIn advances not only theory but also improves IR-based bug localization. 

**Abstract (ZH)**: 软件错误在开发和维护过程中构成了重大挑战，从业者花费近一半的时间来处理这些错误。现有许多技术采用信息检索（IR）方法，通过错误报告与源代码之间的文本和语义相关性来定位错误报告。然而，这些方法往往难以弥合错误报告与代码之间的重要差距，这种差距要求深入了解上下文，而不仅仅是文本或语义相关性。本文提出了一种新的错误定位技术——BRaIn，该技术通过利用大型语言模型（LLM）评估错误报告与代码之间的相关性来解决这些上下文差距。在此基础上，利用LLM的反馈（即智能相关性反馈）来重新构建查询并重新排名源文档，从而提高错误定位的准确性。我们使用基准数据集Bench4BL和三种性能指标评估了BRaIn，并将其与文献中的六种基线技术进行了比较。实验结果表明，BRaIn在MAP、MRR和HIT@K指标上的表现分别优于基线技术87.6%、89.5%和48.8%。此外，它还能定位大约52%的由于错误报告质量不佳而无法被基线技术定位的错误。通过解决上下文差距并引入智能相关性反馈，BRaIn从理论上和实际应用上都推动了基于IR的错误定位技术的发展。 

---
# Bridging Visualization and Optimization: Multimodal Large Language Models on Graph-Structured Combinatorial Optimization 

**Title (ZH)**: 跨模态连接与优化：基于图结构组合优化的多模态大型语言模型 

**Authors**: Jie Zhao, Kang Hao Cheong, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2501.11968)  

**Abstract**: Graph-structured combinatorial challenges are inherently difficult due to their nonlinear and intricate nature, often rendering traditional computational methods ineffective or expensive. However, these challenges can be more naturally tackled by humans through visual representations that harness our innate ability for spatial reasoning. In this study, we propose transforming graphs into images to preserve their higher-order structural features accurately, revolutionizing the representation used in solving graph-structured combinatorial tasks. This approach allows machines to emulate human-like processing in addressing complex combinatorial challenges. By combining the innovative paradigm powered by multimodal large language models (MLLMs) with simple search techniques, we aim to develop a novel and effective framework for tackling such problems. Our investigation into MLLMs spanned a variety of graph-based tasks, from combinatorial problems like influence maximization to sequential decision-making in network dismantling, as well as addressing six fundamental graph-related issues. Our findings demonstrate that MLLMs exhibit exceptional spatial intelligence and a distinctive capability for handling these problems, significantly advancing the potential for machines to comprehend and analyze graph-structured data with a depth and intuition akin to human cognition. These results also imply that integrating MLLMs with simple optimization strategies could form a novel and efficient approach for navigating graph-structured combinatorial challenges without complex derivations, computationally demanding training and fine-tuning. 

**Abstract (ZH)**: 由于其非线性和错综复杂的特点，基于图的组合挑战本质上是困难的，常常使传统的计算方法变得无效或成本高昂。然而，这些挑战可以通过视觉表示更容易地由人类解决，利用我们天生的空间推理能力。在这项研究中，我们提出将图转换为图像以准确保留其高阶结构特征，从而彻底变革解决图结构组合任务的表示方式。这种方法使得机器能够在解决复杂组合挑战时模拟人类的处理方式。通过结合由多模态大语言模型（MLLMs）驱动的新颖范式与简单的搜索技术，我们旨在开发一个新颖且有效的框架来解决此类问题。我们对MLLMs的研究涵盖了多种基于图的任务，从组合问题如影响力最大化到网络拆解中的顺序决策，以及解决六项基本的图相关问题。我们的研究结果表明，MLLMs展示了卓越的空间智能和独特的处理这些任务的能力，这大大提高了机器理解并分析图结构数据的潜力，使其具备与人类认知相似的深度和直觉。这些结果还暗示，将MLLMs与简单的优化策略相结合，可能形成一种无需复杂推导、计算资源消耗低且高效的框架，以应对图结构组合挑战。 

---
# Fine-Grained Appropriate Reliance: Human-AI Collaboration with a Multi-Step Transparent Decision Workflow for Complex Task Decomposition 

**Title (ZH)**: 细粒度适当的依赖：一种用于复杂任务分解的多步透明决策工作流的人机协作 

**Authors**: Gaole He, Patrick Hemmer, Michael Vössing, Max Schemmer, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2501.10909)  

**Abstract**: In recent years, the rapid development of AI systems has brought about the benefits of intelligent services but also concerns about security and reliability. By fostering appropriate user reliance on an AI system, both complementary team performance and reduced human workload can be achieved. Previous empirical studies have extensively analyzed the impact of factors ranging from task, system, and human behavior on user trust and appropriate reliance in the context of one-step decision making. However, user reliance on AI systems in tasks with complex semantics that require multi-step workflows remains under-explored. Inspired by recent work on task decomposition with large language models, we propose to investigate the impact of a novel Multi-Step Transparent (MST) decision workflow on user reliance behaviors. We conducted an empirical study (N = 233) of AI-assisted decision making in composite fact-checking tasks (i.e., fact-checking tasks that entail multiple sub-fact verification steps). Our findings demonstrate that human-AI collaboration with an MST decision workflow can outperform one-step collaboration in specific contexts (e.g., when advice from an AI system is misleading). Further analysis of the appropriate reliance at fine-grained levels indicates that an MST decision workflow can be effective when users demonstrate a relatively high consideration of the intermediate steps. Our work highlights that there is no one-size-fits-all decision workflow that can help obtain optimal human-AI collaboration. Our insights help deepen the understanding of the role of decision workflows in facilitating appropriate reliance. We synthesize important implications for designing effective means to facilitate appropriate reliance on AI systems in composite tasks, positioning opportunities for the human-centered AI and broader HCI communities. 

**Abstract (ZH)**: 近年来，人工智能系统的迅猛发展带来了智能化服务的便利，同时也引发了关于安全性和可靠性的担忧。通过培养用户对人工智能系统的适当依赖，可以实现互补的工作团队性能并减轻人类的工作负担。以往的经验研究表明，从任务、系统和人类行为等多个方面分析了因素对用户信任和适当依赖的影响，特别是在单步骤决策情境下的影响得到了广泛的研究。然而，在涉及复杂语义和多步骤工作流程的任务中，用户对人工智能系统的依赖仍是一个未充分探索的领域。受大型语言模型在任务分解方面近期工作的启发，我们提出研究具有新颖的多步骤透明（MST）决策工作流程对用户依赖行为的影响。我们对人工智能辅助决策在复合事实核查任务（即包含多个子事实验证步骤的任务）中进行了实证研究（N=233）。研究结果表明，MST决策工作流程可以与人类协作在特定情境下表现更优（例如，当人工智能系统的建议误导时）。进一步对细粒度的适当依赖分析表明，当用户对中间步骤表现出相对较高的考量时，MST决策工作流程可以有效发挥作用。我们的研究强调，不存在一种适用于所有情况的决策工作流程，可以帮助实现最佳的人机协作。我们的洞察有助于加深对决策工作流程在促进适当依赖方面作用的理解。为设计有效的方法以促进人在复合任务中对人工智能系统的适当依赖，我们总结了重要的启示，并为以用户为中心的人工智能和更广泛的HCI社区提供了机遇。 

---
# Expertise elevates AI usage: experimental evidence comparing laypeople and professional artists 

**Title (ZH)**: 专家知识提升人工智能应用：普通人群与专业艺术家的实验比较 

**Authors**: Thomas F. Eisenmann, Andres Karjus, Mar Canet Sola, Levin Brinkmann, Bramantyo Ibrahim Supriyatno, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12374)  

**Abstract**: Novel capacities of generative AI to analyze and generate cultural artifacts raise inevitable questions about the nature and value of artistic education and human expertise. Has AI already leveled the playing field between professional artists and laypeople, or do trained artistic expressive capacity, curation skills and experience instead enhance the ability to use these new tools? In this pre-registered study, we conduct experimental comparisons between 50 active artists and a demographically matched sample of laypeople. We designed two tasks to approximate artistic practice for testing their capabilities in both faithful and creative image creation: replicating a reference image, and moving as far away as possible from it. We developed a bespoke platform where participants used a modern text-to-image model to complete both tasks. We also collected and compared participants' sentiments towards AI. On average, artists produced more faithful and creative outputs than their lay counterparts, although only by a small margin. While AI may ease content creation, professional expertise is still valuable - even within the confined space of generative AI itself. Finally, we also explored how well an exemplary vision-capable large language model (GPT-4o) would complete the same tasks, if given the role of an image generation agent, and found it performed on par in copying but outperformed even artists in the creative task. The very best results were still produced by humans in both tasks. These outcomes highlight the importance of integrating artistic skills with AI training to prepare artists and other visual professionals for a technologically evolving landscape. We see a potential in collaborative synergy with generative AI, which could reshape creative industries and education in the arts. 

**Abstract (ZH)**: 生成型人工智能对文化艺术品进行分析和生成的新能力引发了关于艺术教育和人类专业价值本质的不可避免的问题。AI是否已经消除了专业艺术家和非专业人士之间的竞争门槛，还是受过训练的艺术表达能力、策展技巧和经验反而增强了使用这些新工具的能力？在本项预先注册的研究中，我们对50名活跃艺术家和人口统计学特征匹配的非专业人士样本进行了实验比较。我们设计了两个任务来近似艺术实践，以检测他们在忠实和创意图像生成方面的能力：复制参考图像，以及尽量远离该图像。我们开发了一个定制平台，让参与者使用现代文本转图像模型完成这两个任务。我们还收集并比较了参与者对人工智能的态度。总体而言，艺术家在忠实度和创意方面产生的输出略优于非专业人士的相应输出。尽管如此，AI在内容创作方面仍可能简化操作，但专业技能在生成型AI领域内仍然具有价值。最后，我们还探讨了对于视图能力强大的大型语言模型（GPT-4o）而言，如果将其视为图像生成代理，它在完成相同任务方面的情况，并发现在复制任务中表现相似，但在创意任务中甚至超越了艺术家。最好的结果仍然由人类产生。这些结果突显了将艺术技能与AI培训结合的重要性，以准备艺术家和其他视觉专业人士应对技术不断发展的环境。我们看到了与生成型人工智能协作的潜力，这可能会重塑创意行业和艺术教育。 

---
# Treefix: Enabling Execution with a Tree of Prefixes 

**Title (ZH)**: Treefix：启用树形前缀树的执行 

**Authors**: Beatriz Souza, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2501.12339)  

**Abstract**: The ability to execute code is a prerequisite for various dynamic program analyses. Learning-guided execution has been proposed as an approach to enable the execution of arbitrary code snippets by letting a neural model predict likely values for any missing variables. Although state-of-the-art learning-guided execution approaches, such as LExecutor, can enable the execution of a relative high amount of code, they are limited to predicting a restricted set of possible values and do not use any feedback from previous executions to execute even more code. This paper presents Treefix, a novel learning-guided execution approach that leverages LLMs to iteratively create code prefixes that enable the execution of a given code snippet. The approach addresses the problem in a multi-step fashion, where each step uses feedback about the code snippet and its execution to instruct an LLM to improve a previously generated prefix. This process iteratively creates a tree of prefixes, a subset of which is returned to the user as prefixes that maximize the number of executed lines in the code snippet. In our experiments with two datasets of Python code snippets, Treefix achieves 25% and 7% more coverage relative to the current state of the art in learning-guided execution, covering a total of 84% and 82% of all lines in the code snippets. 

**Abstract (ZH)**: 执行代码的能力是各种动态程序分析的前提条件。学习引导执行已被提出作为一种方法，通过让神经模型预测任意缺失变量的可能值，从而使任意代码片段能够被执行。尽管当前最先进的学习引导执行方法，如LExecutor，能够使大量代码被执行，但它们只能预测一组受限的可能值，并且不利用先前执行的反馈来进一步扩展执行的代码量。本文介绍了一种名为Treefix的新型学习引导执行方法，利用大型语言模型（LLMs）逐步创建代码前缀，以使给定代码片段能够被执行。该方法以多步方式来解决这一问题，每一步都利用关于代码片段及其执行的反馈来指导一个LLM改进先前生成的前缀。此过程逐步创建一棵前缀树，其中部分前缀被返回给用户，以最大化代码片段中被执行的行数。在对两个包含Python代码片段的数据集进行实验中，Treefix在学习引导执行方面分别实现了25%和7%更高的覆盖率，总共覆盖了代码片段中84%和82%的行数。 

---
# LLM-Assisted Knowledge Graph Completion for Curriculum and Domain Modelling in Personalized Higher Education Recommendations 

**Title (ZH)**: 基于LLM辅助的知识图谱补全在个性化高等教育推荐中的课程与领域建模 

**Authors**: Hasan Abu-Rasheed, Constance Jumbo, Rashed Al Amin, Christian Weber, Veit Wiese, Roman Obermaisser, Madjid Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12300)  

**Abstract**: While learning personalization offers great potential for learners, modern practices in higher education require a deeper consideration of domain models and learning contexts, to develop effective personalization algorithms. This paper introduces an innovative approach to higher education curriculum modelling that utilizes large language models (LLMs) for knowledge graph (KG) completion, with the goal of creating personalized learning-path recommendations. Our research focuses on modelling university subjects and linking their topics to corresponding domain models, enabling the integration of learning modules from different faculties and institutions in the student's learning path. Central to our approach is a collaborative process, where LLMs assist human experts in extracting high-quality, fine-grained topics from lecture materials. We develop a domain, curriculum, and user models for university modules and stakeholders. We implement this model to create the KG from two study modules: Embedded Systems and Development of Embedded Systems Using FPGA. The resulting KG structures the curriculum and links it to the domain models. We evaluate our approach through qualitative expert feedback and quantitative graph quality metrics. Domain experts validated the relevance and accuracy of the model, while the graph quality metrics measured the structural properties of our KG. Our results show that the LLM-assisted graph completion approach enhances the ability to connect related courses across disciplines to personalize the learning experience. Expert feedback also showed high acceptance of the proposed collaborative approach for concept extraction and classification. 

**Abstract (ZH)**: 虽然个性化学习为学习者提供了巨大的潜力，但现代高等教育实践需要更加重视领域模型和学习情境，以开发有效的个性化算法。本文介绍了一种创新的高等教育课程建模方法，利用大型语言模型（LLMs）进行知识图谱（KG）补全，旨在构建个性化的学习路径推荐。我们的研究重点在于建模大学课程及其主题，并将这些主题与其对应的域模型链接起来，从而使不同学院和机构的学习模块能够整合到学生的学习路径中。我们方法的核心在于一种协作过程，其中LLMs协助人类专家从讲义材料中提取高质量、细粒度的主题。我们为大学模块和利益相关者开发了领域模型、课程模型和用户模型。我们使用这种模型从两个学习模块——嵌入式系统和基于FPGA的嵌入式系统开发——构建知识图谱。生成的KG结构化了课程内容，并将其与域模型链接起来。我们通过定性的专家反馈和定量的知识图谱质量指标来评估该方法。领域专家验证了模型的相关性和准确性，而图的结构属性是衡量我们知识图谱质量的指标。研究表明，LLM辅助的知识图谱补全方法增强了跨学科连接相关课程的能力，以个性化学习体验。专家反馈还表明，对概念提取和分类的协作方法有很高的接受度。 

---
# Early evidence of how LLMs outperform traditional systems on OCR/HTR tasks for historical records 

**Title (ZH)**: 早期证据表明，大规模语言模型在历史记录的OCR/HTR任务中优于传统系统 

**Authors**: Seorin Kim, Julien Baudru, Wouter Ryckbosch, Hugues Bersini, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2501.11623)  

**Abstract**: We explore the ability of two LLMs -- GPT-4o and Claude Sonnet 3.5 -- to transcribe historical handwritten documents in a tabular format and compare their performance to traditional OCR/HTR systems: EasyOCR, Keras, Pytesseract, and TrOCR. Considering the tabular form of the data, two types of experiments are executed: one where the images are split line by line and the other where the entire scan is used as input. Based on CER and BLEU, we demonstrate that LLMs outperform the conventional OCR/HTR methods. Moreover, we also compare the evaluated CER and BLEU scores to human evaluations to better judge the outputs of whole-scan experiments and understand influential factors for CER and BLEU. Combining judgments from all the evaluation metrics, we conclude that two-shot GPT-4o for line-by-line images and two-shot Claude Sonnet 3.5 for whole-scan images yield the transcriptions of the historical records most similar to the ground truth. 

**Abstract (ZH)**: 我们探讨了两种大语言模型——GPT-4o 和 Claude Sonnet 3.5——在将历史手写文档转换为表格格式方面的能力，并将其性能与传统的OCR/HTR系统（如EasyOCR、Keras、Pytesseract 和 TrOCR）进行了比较。鉴于数据的表格形式，我们执行了两种类型的实验：一种是将图像逐行分割，另一种是使用整个扫描作为输入。基于字符错误率（CER）和BLEU分数，我们展示了大语言模型在某些方面优于传统的OCR/HTR方法。此外，我们还将评估的CER和BLEU分数与人工评估进行了比较，以更好地判断整体扫描实验的输出结果，并理解CER和BLEU的影响因素。综合所有评估指标的判断，我们得出结论：对于逐行图像使用两轮制GPT-4o，而对于整体扫描图像使用两轮制Claude Sonnet 3.5，生成的历史记录转录结果与真实值最为接近。 

---
# Conversation Routines: A Prompt Engineering Framework for Task-Oriented Dialog Systems 

**Title (ZH)**: 对话惯例：面向任务导向的对话系统的一种提示工程框架 

**Authors**: Giorgio Robino  

**Link**: [PDF](https://arxiv.org/pdf/2501.11613)  

**Abstract**: This study introduces Conversation Routines (CR), a structured prompt engineering framework for developing task-oriented dialog systems using Large Language Models (LLMs). While LLMs demonstrate remarkable natural language understanding capabilities, engineering them to reliably execute complex business workflows remains challenging. The proposed CR framework enables the development of Conversation Agentic Systems (CAS) through natural language specifications, embedding task-oriented logic within LLM prompts. This approach provides a systematic methodology for designing and implementing complex conversational workflows while maintaining behavioral consistency. We demonstrate the framework's effectiveness through two proof of concept implementations: a Train Ticket Booking System and an Interactive Troubleshooting Copilot. These case studies validate CR's capability to encode sophisticated behavioral patterns and decision logic while preserving natural conversational flexibility. Results show that CR enables domain experts to design conversational workflows in natural language while leveraging custom enterprise functionalities (tools) developed by software engineers, creating an efficient division of responsibilities where developers focus on core API implementation and domain experts handle conversation design. While the framework shows promise in accessibility and adaptability, we identify key challenges including computational overhead, non-deterministic behavior, and domain-specific logic optimization. Future research directions include enhancing system robustness, improving scalability for complex multi-agent interactions, and addressing the identified limitations across diverse business applications. 

**Abstract (ZH)**: 本研究介绍了对话例行程序（CR），这是一个结构化的提示工程框架，用于使用大规模语言模型（LLMs）开发面向任务的对话系统。尽管LLMs展现出卓越的自然语言理解能力，但将它们工程化以可靠地执行复杂的业务工作流仍然具有挑战性。所提出的CR框架通过自然语言规范使开发者能够构建对话代理系统（CAS），并将任务导向的逻辑嵌入到LLM提示中。这种方法提供了一种系统的方法来设计和实现复杂的对话工作流，同时保持行为一致性。我们通过两个概念验证实现展示了该框架的有效性：一个火车票预订系统和一个交互式故障排除副驾。这些案例研究验证了CR能够编码复杂的行为模式和决策逻辑，同时保持自然对话的灵活性。结果显示，CR使领域专家能够使用自然语言设计对话工作流，同时利用软件工程师开发的定制企业功能（工具），从而形成一种高效的职责分工，其中开发人员专注于核心API的实现，而领域专家则负责对话设计。尽管该框架在易用性和适应性方面显示出前景，但我们仍识别出一些关键挑战，包括计算开销、非确定性行为以及特定领域的逻辑优化。未来的研究方向包括增强系统的稳健性、改进多代理交互的可扩展性，并解决在不同商业应用场景中识别出的限制。 

---
# Generative AI and Large Language Models in Language Preservation: Opportunities and Challenges 

**Title (ZH)**: 生成式人工智能与大型语言模型在语言保护中的机遇与挑战 

**Authors**: Vincent Koc  

**Link**: [PDF](https://arxiv.org/pdf/2501.11496)  

**Abstract**: Generative AI and large-scale language models (LLM) have emerged as powerful tools in language preservation, particularly for near-native and endangered languages. With the increasing reliance on technology for communication, education, and cultural documentation, new opportunities have emerged to mitigate the dramatic decline of linguistic diversity worldwide. This paper examines the role of generative AIs and LLMs in preserving endangered languages, highlighting the risks and challenges associated with their use. We analyze the underlying technologies driving these models, including natural language processing (NLP) and deep learning, and explore several cases where these technologies have been applied to low-resource languages. Additionally, we discuss ethical considerations, data scarcity issues, and technical challenges while proposing solutions to enhance AI-driven language preservation. 

**Abstract (ZH)**: 生成式人工智能和大规模语言模型（LLM）已成为语言保护方面的强大工具，特别适用于濒临消失和濒危语言。随着技术在沟通、教育和文化记录方面依赖的不断增加，出现了新的机会来缓解全球语言多样性的急剧下降。本文探讨了生成式人工智能和大规模语言模型在保护濒危语言方面的作用，同时指出了使用这些工具所伴随的风险和挑战。我们分析了驱动这些模型的底层技术，包括自然语言处理（NLP）和深度学习，并探讨了这些技术在低资源语言中的应用案例。此外，我们还讨论了伦理考量、数据稀缺问题和技术挑战，并提出了增强人工智能驱动的语言保护的解决方案。 

---
# Towards Advancing Code Generation with Large Language Models: A Research Roadmap 

**Title (ZH)**: 朝向借助大规模语言模型推进代码生成的研究蓝图 

**Authors**: Haolin Jin, Huaming Chen, Qinghua Lu, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11354)  

**Abstract**: Recently, we have witnessed the rapid development of large language models, which have demonstrated excellent capabilities in the downstream task of code generation. However, despite their potential, LLM-based code generation still faces numerous technical and evaluation challenges, particularly when embedded in real-world development. In this paper, we present our vision for current research directions, and provide an in-depth analysis of existing studies on this task. We propose a six-layer vision framework that categorizes code generation process into distinct phases, namely Input Phase, Orchestration Phase, Development Phase, and Validation Phase. Additionally, we outline our vision workflow, which reflects on the currently prevalent frameworks. We systematically analyse the challenges faced by large language models, including those LLM-based agent frameworks, in code generation tasks. With these, we offer various perspectives and actionable recommendations in this area. Our aim is to provide guidelines for improving the reliability, robustness and usability of LLM-based code generation systems. Ultimately, this work seeks to address persistent challenges and to provide practical suggestions for a more pragmatic LLM-based solution for future code generation endeavors. 

**Abstract (ZH)**: 近年来，我们见证了大规模语言模型的迅速发展，这些模型在代码生成的下游任务中展示了卓越的能力。然而，尽管前景广阔，基于语言模型的代码生成仍然面临着诸多技术和评估挑战，尤其是在实际开发环境中。在本文中，我们提出了当前研究方向的愿景，并对这一任务的现有研究进行了深入分析。我们提出了一个六层框架，将代码生成过程划分为不同的阶段，分别是输入阶段、编排阶段、开发阶段和验证阶段。此外，我们概述了我们的愿景工作流程，反映了目前流行的框架。我们系统地分析了大规模语言模型在代码生成任务中面临的挑战，包括基于语言模型的代理框架。基于这些分析，我们提供了该领域的多种视角和可操作的建议。我们的目标是为提高基于语言模型的代码生成系统的可靠性和易用性提供指导。最终，本文旨在解决持久存在的挑战，并为未来的代码生成提供更实用的基于语言模型的解决方案。 

---
# Can LLM Generate Regression Tests for Software Commits? 

**Title (ZH)**: 大规模语言模型能否生成软件提交的回归测试用例？ 

**Authors**: Jing Liu, Seongmin Lee, Eleonora Losiouk, Marcel Böhme  

**Link**: [PDF](https://arxiv.org/pdf/2501.11086)  

**Abstract**: Large Language Models (LLMs) have shown tremendous promise in automated software engineering. In this paper, we investigate the opportunities of LLMs for automatic regression test generation for programs that take highly structured, human-readable inputs, such as XML parsers or JavaScript interpreters. Concretely, we explore the following regression test generation scenarios for such programs that have so far been difficult to test automatically in the absence of corresponding input grammars:
$\bullet$ Bug finding. Given a code change (e.g., a commit or pull request), our LLM-based approach generates a test case with the objective of revealing any bugs that might be introduced if that change is applied.
$\bullet$ Patch testing. Given a patch, our LLM-based approach generates a test case that fails before but passes after the patch. This test can be added to the regression test suite to catch similar bugs in the future.
We implement Cleverest, a feedback-directed, zero-shot LLM-based regression test generation technique, and evaluate its effectiveness on 22 commits to three subject programs: Mujs, Libxml2, and Poppler. For programs using more human-readable file formats, like XML or JavaScript, we found Cleverest performed very well. It generated easy-to-understand bug-revealing or bug-reproduction test cases for the majority of commits in just under three minutes -- even when only the code diff or commit message (unless it was too vague) was given. For programs with more compact file formats, like PDF, as expected, it struggled to generate effective test cases. However, the LLM-supplied test cases are not very far from becoming effective (e.g., when used as a seed by a greybox fuzzer or as a starting point by the developer). 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化软件工程中显示出了巨大的潜力。在本文中，我们探讨了LLMs在为那些接收高度结构化且可读性较强的输入（如XML解析器或JavaScript解释器）的程序自动生成回归测试方面的潜力。具体来说，我们研究了以下几种难以在缺乏相应输入语法规则的情况下自动测试的程序的回归测试生成场景：
- **缺陷查找。** 给定代码变更（例如，一次提交或拉取请求），我们的基于LLM的方法生成一个测试用例，其目的是揭示如果应用该变更可能会引入的任何缺陷。
- **补丁测试。** 给定一个补丁，我们的基于LLM的方法生成一个测试用例，该用例在补丁前失败但在补丁后通过。这个测试用例可以添加到回归测试套件中，以在未来捕捉类似的缺陷。

我们实现了Cleverest，这是一种基于反馈的、零样本的LLM回归测试生成技术，并在三个主题程序（Mujs、Libxml2和Poppler）的22次提交上对其有效性进行了评估。对于使用更易于阅读的文件格式（如XML或JavaScript）的程序，我们发现Cleverest表现非常出色。它在不到三分钟的时间内为绝大多数提交生成了易于理解的揭示或再现缺陷的测试用例——即使只提供代码差异或提交信息（前提是信息足够具体）。对于使用更紧凑文件格式（如PDF）的程序，如预期的那样，它在生成有效的测试用例方面遇到困难。然而，由LLM提供的测试用例距离有效并不远，例如，当作为灰盒模糊测试器的种子或开发者的起点时。 

---
# The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs 

**Title (ZH)**: LLM作为法官的替代注释员测试：如何通过统计方法证明可以用LLM替代人类注释员 

**Authors**: Nitay Calderon, Roi Reichart, Rotem Dror  

**Link**: [PDF](https://arxiv.org/pdf/2501.10970)  

**Abstract**: The "LLM-as-a-judge" paradigm employs Large Language Models (LLMs) as annotators and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure -- the Alternative Annotator Test (alt-test) -- that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming open-source LLMs, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices. 

**Abstract (ZH)**: “LLM-as-a-judge”范式利用大型语言模型（LLMs）作为传统上由人类完成的任务中的注释员和评估者。LLM注解在多个领域广泛使用，不仅限于自然语言处理研究，还涉及到医学、心理学和社会科学等领域。尽管LLMs在研究结果和见解的形成中起着重要作用，但对于是否能够替代人类注释员并没有标准和严谨的鉴定程序。本文提出了一种新的统计方法——替代注释员检验（alt-test），仅需要少量注释示例即可证明使用LLM注解的有效性。此外，我们介绍了用于比较LLM评审员的灵活且可解释性较强的度量标准。为了展示我们的方法，我们精选了十个多样化的数据集，其中包括语言和视觉-语言任务，并采用了六种LLM和四种提示技术进行了实验。结果表明，在某些情况下，闭源LLM（如GPT-4o）可以替代人类，表现优于开源LLM，并且提示技术产生了不同质量的评审员。我们希望这项研究能够促进更加严谨和可靠的实践。 

---
# Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments 

**Title (ZH)**: 基于数据的交互学习：一种适用于现实环境的自适应代理自适应框架 

**Authors**: Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2501.10893)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have the potential to enhance human capabilities, assisting with digital tasks from sending emails to performing data analysis. The abilities of existing LLMs at such tasks are often hindered by the lack of high-quality agent data from the corresponding environments they interact with. We propose Learn-by-interact, a data-centric framework to adapt LLM agents to any given environments without human annotations. Learn-by-interact synthesizes trajectories of agent-environment interactions based on documentations, and constructs instructions by summarizing or abstracting the interaction histories, a process called backward construction. We assess the quality of our synthetic data by using them in both training-based scenarios and training-free in-context learning (ICL), where we craft innovative retrieval approaches optimized for agents. Extensive experiments on SWE-bench, WebArena, OSWorld and Spider2-V spanning across realistic coding, web, and desktop environments show the effectiveness of Learn-by-interact in various downstream agentic tasks -- baseline results are improved by up to 12.2\% for ICL with Claude-3.5 and 19.5\% for training with Codestral-22B. We further demonstrate the critical role of backward construction, which provides up to 14.0\% improvement for training. Our ablation studies demonstrate the efficiency provided by our synthesized data in ICL and the superiority of our retrieval pipeline over alternative approaches like conventional retrieval-augmented generation (RAG). We expect that Learn-by-interact will serve as a foundation for agent data synthesis as LLMs are increasingly deployed at real-world environments. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的自主代理有可能增强人类的能力，协助完成从发送电子邮件到进行数据分析等各种数字任务。现有的LLMs在这些任务中的能力往往受到与之交互的相应环境中的高质量代理数据缺乏的限制。我们提出了一个数据为中心的框架——“通过交互学习”，该框架能够在无需人工注释的情况下使LLM代理适应任何给定的环境。通过文档，“通过交互学习”综合了代理-环境交互的轨迹，并通过总结或抽象交互历史来构建指令，这一过程称为反向构造。我们通过使用合成数据在基于训练的场景和无需训练的上下文学习（ICL）中评估其质量，其中我们设计了针对代理的创新检索方法。在SWE-bench、WebArena、OSWorld和Spider2-V等涵盖现实编码、网络和桌面环境的广泛实验中，展示了“通过交互学习”的有效性，通过使用Codestral-22B训练时，基准结果提高了19.5%，使用Claude-3.5进行ICL时提高了12.2%。我们进一步证明了反向构造的关键作用，其能够提供高达14.0%的训练改进。我们的消融研究表明，我们合成数据在ICL中的效率以及我们检索流水线相较于传统检索增强生成（RAG）等替代方法的优势。我们期望“通过交互学习”将成为LLMs部署到真实环境中的代理数据合成的基础。 

---
# Zero-shot and Few-shot Learning with Instruction-following LLMs for Claim Matching in Automated Fact-checking 

**Title (ZH)**: 基于指令遵循大语言模型的零样本和少样本学习在自动事实核查中的声明匹配研究 

**Authors**: Dina Pisarevskaya, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2501.10860)  

**Abstract**: The claim matching (CM) task can benefit an automated fact-checking pipeline by putting together claims that can be resolved with the same fact-check. In this work, we are the first to explore zero-shot and few-shot learning approaches to the task. We consider CM as a binary classification task and experiment with a set of instruction-following large language models (GPT-3.5-turbo, Gemini-1.5-flash, Mistral-7B-Instruct, and Llama-3-8B-Instruct), investigating prompt templates. We introduce a new CM dataset, ClaimMatch, which will be released upon acceptance. We put LLMs to the test in the CM task and find that it can be tackled by leveraging more mature yet similar tasks such as natural language inference or paraphrase detection. We also propose a pipeline for CM, which we evaluate on texts of different lengths. 

**Abstract (ZH)**: 声明匹配（CM）任务可以通过将可以用同一事实核查解决的声明组合起来，从而为自动化事实核查流水线带来益处。在本文中，我们首次探索了零样本和少样本学习方法在该任务中的应用。我们将CM视为二元分类任务，并使用一组指令跟随的大语言模型（GPT-3.5-turbo、Gemini-1.5-flash、Mistral-7B-Instruct和Llama-3-8B-Instruct）进行实验，并探讨了提示模板。我们引入了一个新的CM数据集，ClaimMatch，并将在论文被接受后发布。我们在CM任务中对LLMs进行了测试，发现可以通过利用更为成熟且类似的任务，如自然语言推理或同义替换检测来解决此任务。此外，我们还提出了一种CM管道，并在不同长度的文本上对其实效性进行了评估。 

---
# Step-KTO: Optimizing Mathematical Reasoning through Stepwise Binary Feedback 

**Title (ZH)**: 步进KTO：通过逐步二元反馈优化数学推理 

**Authors**: Yen-Ting Lin, Di Jin, Tengyu Xu, Tianhao Wu, Sainbayar Sukhbaatar, Chen Zhu, Yun He, Yun-Nung Chen, Jason Weston, Yuandong Tian, Arash Rahnama, Sinong Wang, Hao Ma, Han Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10799)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable success in mathematical reasoning. Despite progress in methods like chain-of-thought prompting and self-consistency sampling, these advances often focus on final correctness without ensuring that the underlying reasoning process is coherent and reliable. This paper introduces Step-KTO, a training framework that combines process-level and outcome-level binary feedback to guide LLMs toward more trustworthy reasoning trajectories. By providing binary evaluations for both the intermediate reasoning steps and the final answer, Step-KTO encourages the model to adhere to logical progressions rather than relying on superficial shortcuts. Our experiments on challenging mathematical benchmarks show that Step-KTO significantly improves both final answer accuracy and the quality of intermediate reasoning steps. For example, on the MATH-500 dataset, Step-KTO achieves a notable improvement in Pass@1 accuracy over strong baselines. These results highlight the promise of integrating stepwise process feedback into LLM training, paving the way toward more interpretable and dependable reasoning capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）最近在数学推理方面展示了明显的成功。尽管方法如链式思考提示和自我一致性采样方面取得了进展，这些进步通常关注最终的正确性，而不确保推理过程本身是连贯和可靠的。本文提出了Step-KTO，一种结合过程级和结果级二元反馈的训练框架，以引导LLMs向更为可信的推理轨迹发展。通过为中间推理步骤和最终答案提供二元评估，Step-KTO 鼓励模型遵循逻辑演进，而不是依赖表面的捷径。我们针对具有挑战性的数学基准数据集的实验表明，Step-KTO 显著提高了最终答案的准确性和中间推理步骤的质量。例如，在MATH-500数据集上，Step-KTO 在Pass@1准确性上显著优于强大的基准模型。这些结果表明整合逐步过程反馈到LLM训练中的潜力，为更可解释和可靠推理能力的发展铺平了道路。 

---
# CodEv: An Automated Grading Framework Leveraging Large Language Models for Consistent and Constructive Feedback 

**Title (ZH)**: CodEv：一种利用大规模语言模型自动评分并提供一致性和建设性反馈的框架 

**Authors**: En-Qi Tseng, Pei-Cing Huang, Chan Hsu, Peng-Yi Wu, Chan-Tung Ku, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10421)  

**Abstract**: Grading programming assignments is crucial for guiding students to improve their programming skills and coding styles. This study presents an automated grading framework, CodEv, which leverages Large Language Models (LLMs) to provide consistent and constructive feedback. We incorporate Chain of Thought (CoT) prompting techniques to enhance the reasoning capabilities of LLMs and ensure that the grading is aligned with human evaluation. Our framework also integrates LLM ensembles to improve the accuracy and consistency of scores, along with agreement tests to deliver reliable feedback and code review comments. The results demonstrate that the framework can yield grading results comparable to human evaluators, by using smaller LLMs. Evaluation and consistency tests of the LLMs further validate our approach, confirming the reliability of the generated scores and feedback. 

**Abstract (ZH)**: 编程作业的自动评分对于指导学生提高编程技能和编码风格至关重要。本研究提出了一种基于大型语言模型（LLMs）的自动评分框架，称为CodEv，该框架能够提供一致而建设性的反馈。我们采用Chain of Thought（CoT）提示技术来增强LLMs的推理能力，并确保评分与人工评估相一致。此外，我们的框架还集成了LLM集成技术，以提高评分的准确性和一致性，并通过一致性测试提供可靠反馈和代码审查评论。结果表明，该框架能够在使用较小的LLMs时获得与人工评估者相当的评分结果。LLMs的评估和一致性测试进一步验证了我们的方法，确认了生成的评分和反馈的可靠性。 

---
# Autonomous Microscopy Experiments through Large Language Model Agents 

**Title (ZH)**: 通过大规模语言模型代理实现自主显微镜实验 

**Authors**: Indrajeet Mandal, Jitendra Soni, Mohd Zaki, Morten M. Smedskjaer, Katrin Wondraczek, Lothar Wondraczek, Nitya Nand Gosvami, N. M. Anoop Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2501.10385)  

**Abstract**: The emergence of large language models (LLMs) has accelerated the development of self-driving laboratories (SDLs) for materials research. Despite their transformative potential, current SDL implementations rely on rigid, predefined protocols that limit their adaptability to dynamic experimental scenarios across different labs. A significant challenge persists in measuring how effectively AI agents can replicate the adaptive decision-making and experimental intuition of expert scientists. Here, we introduce AILA (Artificially Intelligent Lab Assistant), a framework that automates atomic force microscopy (AFM) through LLM-driven agents. Using AFM as an experimental testbed, we develop AFMBench-a comprehensive evaluation suite that challenges AI agents based on language models like GPT-4o and GPT-3.5 to perform tasks spanning the scientific workflow: from experimental design to results analysis. Our systematic assessment shows that state-of-the-art language models struggle even with basic tasks such as documentation retrieval, leading to a significant decline in performance in multi-agent coordination scenarios. Further, we observe that LLMs exhibit a tendency to not adhere to instructions or even divagate to additional tasks beyond the original request, raising serious concerns regarding safety alignment aspects of AI agents for SDLs. Finally, we demonstrate the application of AILA on increasingly complex experiments open-ended experiments: automated AFM calibration, high-resolution feature detection, and mechanical property measurement. Our findings emphasize the necessity for stringent benchmarking protocols before deploying AI agents as laboratory assistants across scientific disciplines. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起加速了材料研究中自动实验室（SDLs）的发展。尽管它们具有变革性潜力，当前的SDL实现依赖于僵硬的预定义协议，限制了其在不同实验室动态实验场景中的适应性。一个重大挑战在于，如何有效衡量人工智能代理能否复制专家科学家的适应性决策和实验直觉。在这里，我们介绍了一种名为AILA（Artificially Intelligent Lab Assistant）的框架，该框架通过基于LLM的代理自动化原子力显微镜（AFM）。使用AFM作为实验测试平台，我们开发了AFMBench——一个全面的评估套件，基于如GPT-4o和GPT-3.5等语言模型挑战AI代理完成涵盖整个科学工作流程的任务：从实验设计到结果分析。我们的系统评估表明，最先进的语言模型即使在基本任务如文献检索上也表现不佳，这在多代理协调场景中导致了显著的性能下降。此外，我们观察到LLMs表现出不遵守指令或甚至转向超出原始请求的额外任务的趋势，这引起了关于SDL中人工智能代理的安全对齐方面的严重关切。最后，我们展示了AILA在日益复杂的实验中的应用：全自动AFM校准、高分辨率特征检测和机械性能测量。我们的研究结果强调了在将AI代理应用于不同科学领域之前进行严格基准测试的必要性。 

---
# Can LLMs Identify Gaps and Misconceptions in Students' Code Explanations? 

**Title (ZH)**: 大语言模型能否识别学生代码解释中的漏洞与误解？ 

**Authors**: Priti Oli, Rabin Banjade, Andrew M. Olney, Vasile Rus  

**Link**: [PDF](https://arxiv.org/pdf/2501.10365)  

**Abstract**: This paper investigates various approaches using Large Language Models (LLMs) to identify gaps and misconceptions in students' self-explanations of specific instructional material, in our case explanations of code examples. This research is a part of our larger effort to automate the assessment of students' freely generated responses, focusing specifically on their self-explanations of code examples during activities related to code comprehension. In this work, we experiment with zero-shot prompting, Supervised Fine-Tuning (SFT), and preference alignment of LLMs to identify gaps in students' self-explanation. With simple prompting, GPT-4 consistently outperformed LLaMA3 and Mistral in identifying gaps and misconceptions, as confirmed by human evaluations. Additionally, our results suggest that fine-tuned large language models are more effective at identifying gaps in students' explanations compared to zero-shot and few-shot prompting techniques. Furthermore, our findings show that the preference optimization approach using Odds Ratio Preference Optimization (ORPO) outperforms SFT in identifying gaps and misconceptions in students' code explanations. 

**Abstract (ZH)**: 本文探讨了使用大型语言模型（LLMs）的各种方法，以识别学生在解释特定教学材料（例如代码示例）时存在的差距和误解。本研究是我们在更大范围内自动评估学生自由生成的回答的一部分，特别是在编码理解活动过程中，专注于识别学生对代码示例的自我解释中存在的差距。在本工作中，我们尝试使用零样本提示、监督微调（SFT）和对LLMs进行偏好对齐，以识别学生自我解释中的差距。通过简单的提示，GPT-4在识别差距和误解方面始终优于LLaMA3和Mistral，并经人工评估证实。此外，我们的结果表明，微调后的大型语言模型在识别学生解释中的差距方面比零样本和少样本提示技术更有效。此外，我们的发现显示，使用Odds Ratio Preference Optimization（ORPO）进行偏好评价的方法在识别学生代码解释中的差距和误解方面优于SFT。 

---
