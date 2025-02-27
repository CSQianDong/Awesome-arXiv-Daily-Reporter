# Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems 

**Title (ZH)**: 代理奖励建模：将人类偏好与可验证的正确性信号集成以构建可靠的奖励系统 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.19328)  

**Abstract**: Reward models (RMs) are crucial for the training and inference-time scaling up of large language models (LLMs). However, existing reward models primarily focus on human preferences, neglecting verifiable correctness signals which have shown strong potential in training LLMs. In this paper, we propose agentic reward modeling, a reward system that combines reward models with verifiable correctness signals from different aspects to provide reliable rewards. We empirically implement a reward agent, named RewardAgent, that combines human preference rewards with two verifiable signals: factuality and instruction following, to provide more reliable rewards. We conduct comprehensive experiments on existing reward model benchmarks and inference time best-of-n searches on real-world downstream tasks. RewardAgent significantly outperforms vanilla reward models, demonstrating its effectiveness. We further construct training preference pairs using RewardAgent and train an LLM with the DPO objective, achieving superior performance on various NLP benchmarks compared to conventional reward models. Our codes are publicly released to facilitate further research (this https URL). 

**Abstract (ZH)**: 人工智能代理（Agent-based Reward Modeling, ARM）对于大型语言模型（LLMs）的训练和推理时扩展至关重要。然而，现有的奖励模型主要关注人类偏好，忽视了在训练LLMs过程中展现出强大潜力的可验证正确性信号。本文中，我们提出了一种结合不同方面可验证正确性信号的奖励模型——代理人奖励建模（agentic reward modeling），以提供可靠的奖励。我们实验性地实现了一个名为RewardAgent的奖励代理，它结合了人类偏好奖励与两种可验证信号：事实性（factuality）和指令遵循（instruction following），以提供更可靠的奖励。我们在现有的奖励模型基准以及实际下游任务的最佳推理时间搜索中进行了全面实验。与传统奖励模型相比，RewardAgent显著表现出更高的有效性。为进一步研究，我们使用RewardAgent构建训练偏好对，并利用DPO目标训练了一个LLM，其在各种NLP基准测试中的表现优于传统奖励模型。我们的代码已公开发布，以促进进一步研究（参见：<https://github.com/PseudoPatrick/Reward-Agent>）。 

---
# MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis 

**Title (ZH)**: MEDDxAgent: 一个统一的模块化代理框架，用于可解释的自动差异诊断 

**Authors**: Daniel Rose, Chia-Chien Hung, Marco Lepri, Israa Alqassem, Kiril Gashteovski, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2502.19175)  

**Abstract**: Differential Diagnosis (DDx) is a fundamental yet complex aspect of clinical decision-making, in which physicians iteratively refine a ranked list of possible diseases based on symptoms, antecedents, and medical knowledge. While recent advances in large language models have shown promise in supporting DDx, existing approaches face key limitations, including single-dataset evaluations, isolated optimization of components, unrealistic assumptions about complete patient profiles, and single-attempt diagnosis. We introduce a Modular Explainable DDx Agent (MEDDxAgent) framework designed for interactive DDx, where diagnostic reasoning evolves through iterative learning, rather than assuming a complete patient profile is accessible. MEDDxAgent integrates three modular components: (1) an orchestrator (DDxDriver), (2) a history taking simulator, and (3) two specialized agents for knowledge retrieval and diagnosis strategy. To ensure robust evaluation, we introduce a comprehensive DDx benchmark covering respiratory, skin, and rare diseases. We analyze single-turn diagnostic approaches and demonstrate the importance of iterative refinement when patient profiles are not available at the outset. Our broad evaluation demonstrates that MEDDxAgent achieves over 10% accuracy improvements in interactive DDx across both large and small LLMs, while offering critical explainability into its diagnostic reasoning process. 

**Abstract (ZH)**: 差异诊断（DDx）是临床决策过程中一个基础但复杂的方面，在这一过程中，医生根据症状、病史和医学知识，逐步细化可能疾病的排名列表。虽然近期大规模语言模型在支持差异诊断方面展现出了潜力，但现有方法仍面临一些关键限制，包括单一数据集评估、模型组件孤立优化、不切实际的完整患者资料假设，以及一次性诊断。我们提出了一种模块化可解释的差异诊断代理（MEDDxAgent）框架，专为互动式差异诊断而设计，诊断推理通过迭代学习演进，而非假设完整患者资料可获取。MEDDxAgent 集成了三个模块化的组件：（1）协调者（DDxDriver），（2）病史采集模拟器，以及（3）两个专门用于知识检索和诊断策略的代理。为了确保稳健的评估，我们引入了一个涵盖呼吸系统、皮肤疾病和罕见疾病的综合差异诊断基准。分析单轮诊断方法，并展示了在初始阶段患者资料不可用时迭代细化的重要性。广泛的评估表明，在大型和小型语言模型中，MEDDxAgent 在互动差异诊断中的准确性提高了超过10%，并对其诊断推理过程提供了关键的可解释性。 

---
# Enhancing Text Classification with a Novel Multi-Agent Collaboration Framework Leveraging BERT 

**Title (ZH)**: 利用BERT的新型多代理协作框架增强文本分类 

**Authors**: Hediyeh Baban, Sai A Pidapar, Aashutosh Nema, Sichen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18653)  

**Abstract**: We introduce a novel multi-agent collaboration framework designed to enhance the accuracy and robustness of text classification models. Leveraging BERT as the primary classifier, our framework dynamically escalates low-confidence predictions to a specialized multi-agent system comprising Lexical, Contextual, Logic, Consensus, and Explainability agents. This collaborative approach allows for comprehensive analysis and consensus-driven decision-making, significantly improving classification performance across diverse text classification tasks. Empirical evaluations on benchmark datasets demonstrate that our framework achieves a 5.5% increase in accuracy compared to standard BERT-based classifiers, underscoring its effectiveness and academic novelty in advancing multi-agent systems within natural language processing. 

**Abstract (ZH)**: 我们提出了一种新型多智能体合作框架，旨在提高文本分类模型的准确性和鲁棒性。该框架采用BERT作为主要分类器，并动态提升低置信度的预测至一个专门的多智能体系统，包含词汇学、上下文、逻辑、共识和可解释性智能体。这种合作方法允许进行全面分析并基于共识进行决策，显著提升了各类文本分类任务的表现。在基准数据集上的实证研究表明，与标准的BERT分类器相比，我们的框架在准确率上提高了5.5%，凸显了其在自然语言处理领域内推动多智能体系统发展的有效性和学术创新性。 

---
# Multi-LLM Collaborative Search for Complex Problem Solving 

**Title (ZH)**: 多大型语言模型协作搜索在复杂问题解决中的应用 

**Authors**: Sen Yang, Yafu Li, Wai Lam, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18873)  

**Abstract**: Large language models (LLMs) often struggle with complex reasoning tasks due to their limitations in addressing the vast reasoning space and inherent ambiguities of natural language. We propose the Mixture-of-Search-Agents (MoSA) paradigm, a novel approach leveraging the collective expertise of multiple LLMs to enhance search-based reasoning. MoSA integrates diverse reasoning pathways by combining independent exploration with iterative refinement among LLMs, mitigating the limitations of single-model approaches. Using Monte Carlo Tree Search (MCTS) as a backbone, MoSA enables multiple agents to propose and aggregate reasoning steps, resulting in improved accuracy. Our comprehensive evaluation across four reasoning benchmarks demonstrates MoSA's consistent performance improvements over single-agent and other multi-agent baselines, particularly in complex mathematical and commonsense reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中往往表现不佳，因为它们在处理广泛推理空间和自然语言固有的歧义性方面存在局限性。我们提出了搜索代理混合体（Mixture-of-Search-Agents, MoSA）范式，这是一种利用多个LLM集体专业知识来增强基于搜索的推理的新方法。MoSA 通过结合独立探索和LLMs之间的迭代完善，整合了多样的推理路径，从而减轻了单模型方法的局限性。MoSA 以蒙特卡洛树搜索（MCTS）为基础，使多个代理能够提出并聚合推理步骤，从而提高准确性。我们的全面评估表明，MoSA 在四个推理基准上的表现优于单代理和其它多代理基线，特别是在复杂的数学和常识推理任务中表现出显著改进。 

---
# Towards an AI co-scientist 

**Title (ZH)**: Towards 一位AI合作者 

**Authors**: Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, Khaled Saab, Dan Popovici, Jacob Blum, Fan Zhang, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Pushmeet Kohli, Yossi Matias, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R D Costa, José R Penadés, Gary Peltz, Yunhan Xu, Annalisa Pawlosky, Alan Karthikesalingam, Vivek Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18864)  

**Abstract**: Scientific discovery relies on scientists generating novel hypotheses that undergo rigorous experimental validation. To augment this process, we introduce an AI co-scientist, a multi-agent system built on Gemini 2.0. The AI co-scientist is intended to help uncover new, original knowledge and to formulate demonstrably novel research hypotheses and proposals, building upon prior evidence and aligned to scientist-provided research objectives and guidance. The system's design incorporates a generate, debate, and evolve approach to hypothesis generation, inspired by the scientific method and accelerated by scaling test-time compute. Key contributions include: (1) a multi-agent architecture with an asynchronous task execution framework for flexible compute scaling; (2) a tournament evolution process for self-improving hypotheses generation. Automated evaluations show continued benefits of test-time compute, improving hypothesis quality. While general purpose, we focus development and validation in three biomedical areas: drug repurposing, novel target discovery, and explaining mechanisms of bacterial evolution and anti-microbial resistance. For drug repurposing, the system proposes candidates with promising validation findings, including candidates for acute myeloid leukemia that show tumor inhibition in vitro at clinically applicable concentrations. For novel target discovery, the AI co-scientist proposed new epigenetic targets for liver fibrosis, validated by anti-fibrotic activity and liver cell regeneration in human hepatic organoids. Finally, the AI co-scientist recapitulated unpublished experimental results via a parallel in silico discovery of a novel gene transfer mechanism in bacterial evolution. These results, detailed in separate, co-timed reports, demonstrate the potential to augment biomedical and scientific discovery and usher an era of AI empowered scientists. 

**Abstract (ZH)**: 科学发现依赖于科学家提出新颖的假设并通过严格的实验验证。为了增强这一过程，我们引入了一个AI合作者，这是一种基于Gemini 2.0的多智能体系统。AI合作者旨在帮助发现新的原创知识，并根据先前的证据和科学家提供的研究目标和指导，提出可证明的新颖研究假设和建议。系统的设计采用了生成、辩论和进化的方法来生成假设，这一方法借鉴了科学方法，并通过扩展测试时计算的规模得到了加速。主要贡献包括：（1）一种具有异步任务执行框架的多智能体架构，实现灵活的计算扩展；（2）一种锦标赛进化过程以自我改善假设生成。自动评估显示，测试时计算继续带来益处，提高了假设的质量。虽然具有通用目的，但我们将开发和验证集中在三个生物医学领域：药物重新定位、新的靶点发现以及解释细菌进化和抗生素耐药机制。在药物重新定位方面，该系统提出了具有有前景验证结果的候选药物，包括在临床适用浓度下显示对急性髓系白血病抑制作用的候选药物。在新的靶点发现方面，AI合作者提出了新的表观遗传靶点用于肝纤维化，这些靶点通过抗纤维化活性和人类肝类器官中的肝细胞再生进行了验证。最后，AI合作者通过平行的计算发现了一个新的基因转移机制，重现了细菌进化的未发表实验结果。这些结果，分别在单独的、同步发布的报告中详述，展示了其在生物医学和科学研究中增强发现潜力，并开启了AI赋能科学家的新时期。 

---
# A Cooperative Multi-Agent Framework for Zero-Shot Named Entity Recognition 

**Title (ZH)**: 零样本命名实体识别的协作多agent框架 

**Authors**: Zihan Wang, Ziqi Zhao, Yougang Lyu, Zhumin Chen, Maarten de Rijke, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.18702)  

**Abstract**: Zero-shot named entity recognition (NER) aims to develop entity recognition systems from unannotated text corpora. This task presents substantial challenges due to minimal human intervention. Recent work has adapted large language models (LLMs) for zero-shot NER by crafting specialized prompt templates. It advances model self-learning abilities by incorporating self-annotated demonstrations. However, two important challenges persist: (i) Correlations between contexts surrounding entities are overlooked, leading to wrong type predictions or entity omissions. (ii) The indiscriminate use of task demonstrations, retrieved through shallow similarity-based strategies, severely misleads LLMs during inference.
In this paper, we introduce the cooperative multi-agent system (CMAS), a novel framework for zero-shot NER that uses the collective intelligence of multiple agents to address the challenges outlined above. CMAS has four main agents: (i) a self-annotator, (ii) a type-related feature (TRF) extractor, (iii) a demonstration discriminator, and (iv) an overall predictor. To explicitly capture correlations between contexts surrounding entities, CMAS reformulates NER into two subtasks: recognizing named entities and identifying entity type-related features within the target sentence. To enable controllable utilization of demonstrations, a demonstration discriminator is established to incorporate the self-reflection mechanism, automatically evaluating helpfulness scores for the target sentence. Experimental results show that CMAS significantly improves zero-shot NER performance across six benchmarks, including both domain-specific and general-domain scenarios. Furthermore, CMAS demonstrates its effectiveness in few-shot settings and with various LLM backbones. 

**Abstract (ZH)**: 零样本命名实体识别（NER）旨在从未标注文本语料库中开发实体识别系统。这一任务由于人类干预极低而面临巨大挑战。近期的工作通过设计特定的提示模板，将大型语言模型（LLMs）应用于零样本NER中，提升了模型的自我学习能力，引入了自我标注的示范。然而，仍存在两个关键挑战：(i) 忽视了实体上下文之间的关联性，导致错误的实体类型预测或实体遗漏；(ii) 通过浅层相似性策略检索的示范的无差别使用，在推理过程中严重误导了LLMs。

本文提出了合作多智能体系统（CMAS），这是一种新颖的零样本命名实体识别框架，利用多个智能体的集体智能来应对上述挑战。CMAS包括四个主要智能体：(i) 自我标注器，(ii) 类型相关特征（TRF）提取器，(iii) 示范鉴别器，以及(iv) 综合预测器。为明确捕获实体上下文之间的关联性，CMAS将NER重新表述为两个子任务：识别命名实体和在目标句子中识别实体类型相关特征。为实现示范的可控利用，建立了一个示范鉴别器，引入了自助反思机制，自动评估目标句子的示范有用性评分。实验结果表明，CMAS在包括特定领域和通用领域在内的六个基准测试中显著提高了零样本命名实体识别性能。此外，CMAS在少量样本设置和不同大型语言模型（LLM）基础架构中均展示了其有效性。 

---
# Enhancing Hepatopathy Clinical Trial Efficiency: A Secure, Large Language Model-Powered Pre-Screening Pipeline 

**Title (ZH)**: 增强肝脏疾病临床试验效率：一种安全的大规模语言模型驱动的预筛查流程 

**Authors**: Xiongbin Gui, Hanlin Lv, Xiao Wang, Longting Lv, Yi Xiao, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18531)  

**Abstract**: Background: Recruitment for cohorts involving complex liver diseases, such as hepatocellular carcinoma and liver cirrhosis, often requires interpreting semantically complex criteria. Traditional manual screening methods are time-consuming and prone to errors. While AI-powered pre-screening offers potential solutions, challenges remain regarding accuracy, efficiency, and data privacy. Methods: We developed a novel patient pre-screening pipeline that leverages clinical expertise to guide the precise, safe, and efficient application of large language models. The pipeline breaks down complex criteria into a series of composite questions and then employs two strategies to perform semantic question-answering through electronic health records - (1) Pathway A, Anthropomorphized Experts' Chain of Thought strategy, and (2) Pathway B, Preset Stances within an Agent Collaboration strategy, particularly in managing complex clinical reasoning scenarios. The pipeline is evaluated on three key metrics-precision, time consumption, and counterfactual inference - at both the question and criterion levels. Results: Our pipeline achieved high precision (0.921, in criteria level) and efficiency (0.44s per task). Pathway B excelled in complex reasoning, while Pathway A was effective in precise data extraction with faster processing times. Both pathways achieved comparable precision. The pipeline showed promising results in hepatocellular carcinoma (0.878) and cirrhosis trials (0.843). Conclusions: This data-secure and time-efficient pipeline shows high precision in hepatopathy trials, providing promising solutions for streamlining clinical trial workflows. Its efficiency and adaptability make it suitable for improving patient recruitment. And its capability to function in resource-constrained environments further enhances its utility in clinical settings. 

**Abstract (ZH)**: 背景：涉及复杂肝脏疾病的队列研究，如肝细胞癌和肝硬化，常常需要解读复杂的筛选标准。传统的手工筛查方法耗时且容易出错。虽然基于人工智能的预筛查提供了潜在解决方案，但在准确性和效率以及数据隐私方面仍然存在挑战。方法：我们开发了一种新的患者预筛查管道，该管道利用临床知识指导大型语言模型的精确、安全和高效应用。该管道将复杂的筛选标准分解为一系列复合问题，然后通过电子健康记录执行语义问题回答，具体策略包括：（1）途径A：拟人化专家的思维链策略；（2）途径B：代理合作中的预设立场策略，特别适用于处理复杂的临床推理情景。该管道从问题和标准层面分别以精度、耗时和反事实推理为评价指标进行了评估。结果：我们的管道在标准层面达到了高精度（0.921）和高效率（每任务0.44秒）。途径B在复杂推理方面表现出色，而途径A在精确数据提取方面更有效，且处理时间更短。两种途径在精度上表现相当。该管道在肝细胞癌（0.878）和肝硬化临床试验（0.843）中显示出有前景的结果。结论：这种数据安全且高效的工作流管道在肝病临床试验中显示出了高精度，为简化临床试验流程提供了有前景的解决方案。其高效性和适应性使其适用于提高患者招募。此外，其在资源受限环境中运行的能力进一步增强了其在临床环境中的实用性。 

---
# FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data 

**Title (ZH)**: FinBloom：基于实时金融数据的大规模语言模型知识 grounding 

**Authors**: Ankur Sinha, Chaitanya Agarwal, Pekka Malo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18471)  

**Abstract**: Large language models (LLMs) excel at generating human-like responses but often struggle with interactive tasks that require access to real-time information. This limitation poses challenges in finance, where models must access up-to-date information, such as recent news or price movements, to support decision-making. To address this, we introduce Financial Agent, a knowledge-grounding approach for LLMs to handle financial queries using real-time text and tabular data. Our contributions are threefold: First, we develop a Financial Context Dataset of over 50,000 financial queries paired with the required context. Second, we train FinBloom 7B, a custom 7 billion parameter LLM, on 14 million financial news articles from Reuters and Deutsche Presse-Agentur, alongside 12 million Securities and Exchange Commission (SEC) filings. Third, we fine-tune FinBloom 7B using the Financial Context Dataset to serve as a Financial Agent. This agent generates relevant financial context, enabling efficient real-time data retrieval to answer user queries. By reducing latency and eliminating the need for users to manually provide accurate data, our approach significantly enhances the capability of LLMs to handle dynamic financial tasks. Our proposed approach makes real-time financial decisions, algorithmic trading and other related tasks streamlined, and is valuable in contexts with high-velocity data flows. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成类人回复方面表现出色，但在处理需要实时信息访问的交互任务时往往显得力不从心。这种局限性在金融领域尤为突出，因为在金融领域，模型必须访问最新的信息（如最近的新闻或价格变动）以支持决策。为了解决这一问题，我们引入了“金融代理”这一知识落地方法，该方法允许LLMs使用实时文本和表格数据来处理金融查询。我们的贡献主要包括三个方面：首先，我们开发了一个包含超过50,000个金融查询及其所需上下文的数据集。其次，我们在来自路透社和德新社的1400万篇金融新闻文章以及1200万篇美国证券交易委员会（SEC）文件上训练了一个70亿参数的定制模型——FinBloom 7B。最后，我们使用财务上下文数据集对FinBloom 7B进行微调，使之成为金融代理。该代理能够生成相关的财务上下文信息，从而高效地实现实时数据检索，以回答用户查询。通过减少延迟并消除用户手动提供准确数据的需求，我们的方法显著增强了LLMs处理动态金融任务的能力。我们提出的方法使得实时金融决策、算法交易及其他相关任务得以简化，尤其适用于数据流快速的应用场景。 

---
# Agent-centric Information Access 

**Title (ZH)**: 以代理为中心的信息访问 

**Authors**: Evangelos Kanoulas, Panagiotis Eustratiadis, Yongkang Li, Yougang Lyu, Vaishali Pal, Gabrielle Poerwawinata, Jingfen Qiao, Zihan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19298)  

**Abstract**: As large language models (LLMs) become more specialized, we envision a future where millions of expert LLMs exist, each trained on proprietary data and excelling in specific domains. In such a system, answering a query requires selecting a small subset of relevant models, querying them efficiently, and synthesizing their responses. This paper introduces a framework for agent-centric information access, where LLMs function as knowledge agents that are dynamically ranked and queried based on their demonstrated expertise. Unlike traditional document retrieval, this approach requires inferring expertise on the fly, rather than relying on static metadata or predefined model descriptions. This shift introduces several challenges, including efficient expert selection, cost-effective querying, response aggregation across multiple models, and robustness against adversarial manipulation. To address these issues, we propose a scalable evaluation framework that leverages retrieval-augmented generation and clustering techniques to construct and assess thousands of specialized models, with the potential to scale toward millions. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）变得越来越专业化，我们设想一个未来场景，在这个场景中，存在数百万个专家型LLMs，它们各自经过专业数据训练并在特定领域表现出色。在这种系统中，回答查询需要选择一小部分相关模型，高效地查询这些模型，并综合它们的回答。本文介绍了一个以代理为中心的信息访问框架，在该框架中，LLMs充当知识代理，并根据它们展示的专业能力进行动态排名和查询。与传统的文档检索不同，这种方法要求实时推断模型的专业能力，而不仅仅是依赖静态元数据或预定义的模型描述。这种转变带来了一系列挑战，包括高效的专家选择、成本效益高效的查询、跨多个模型的响应聚合，以及对抗性操纵的鲁棒性。为了应对这些挑战，我们提出了一种可扩展的评估框架，该框架利用检索增强生成和聚类技术构建和评估数千种专业化模型，并有望扩展到数百万个模型。 

---
# AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms 

**Title (ZH)**: AgentSociety 挑战：设计面向网络平台用户建模和推荐的大型语言模型代理 

**Authors**: Yuwei Yan, Yu Shang, Qingbin Zeng, Yu Li, Keyu Zhao, Zhiheng Zheng, Xuefei Ning, Tianji Wu, Shengen Yan, Yu Wang, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18754)  

**Abstract**: The AgentSociety Challenge is the first competition in the Web Conference that aims to explore the potential of Large Language Model (LLM) agents in modeling user behavior and enhancing recommender systems on web platforms. The Challenge consists of two tracks: the User Modeling Track and the Recommendation Track. Participants are tasked to utilize a combined dataset from Yelp, Amazon, and Goodreads, along with an interactive environment simulator, to develop innovative LLM agents. The Challenge has attracted 295 teams across the globe and received over 1,400 submissions in total over the course of 37 official competition days. The participants have achieved 21.9% and 20.3% performance improvement for Track 1 and Track 2 in the Development Phase, and 9.1% and 15.9% in the Final Phase, representing a significant accomplishment. This paper discusses the detailed designs of the Challenge, analyzes the outcomes, and highlights the most successful LLM agent designs. To support further research and development, we have open-sourced the benchmark environment at this https URL. 

**Abstract (ZH)**: 《AgentSociety挑战赛》是Web Conference上首次旨在探索大型语言模型（LLM）代理在模拟用户行为和提升网页平台推荐系统方面潜力的竞赛。该挑战包含两个赛道：用户建模赛道和推荐赛道。参赛者被要求利用从Yelp、Amazon和Goodreads获取的综合数据集以及互动环境模拟器来开发创新性的LLM代理。该挑战吸引了来自全球的295支队伍，在37个正式竞赛日中收到了超过1,400份提交。开发阶段，赛道1和赛道2的参与队伍分别取得了21.9%和20.3%的性能提升，在最终阶段，这两个赛道的参赛队伍分别实现了9.1%和15.9%的性能提升，代表了显著的成就。本文详细讨论了挑战的设计、分析了结果，并突出了最成功的LLM代理设计。为了支持进一步的研究和开发，我们已在以下网址开源了基准环境：[请插入网址]。 

---
# Multi-Agent Security Tax: Trading Off Security and Collaboration Capabilities in Multi-Agent Systems 

**Title (ZH)**: 多代理安全税：多代理系统中安全与协作能力的权衡 

**Authors**: Pierre Peigne-Lefebvre, Mikolaj Kniejski, Filip Sondej, Matthieu David, Jason Hoelscher-Obermaier, Christian Schroeder de Witt, Esben Kran  

**Link**: [PDF](https://arxiv.org/pdf/2502.19145)  

**Abstract**: As AI agents are increasingly adopted to collaborate on complex objectives, ensuring the security of autonomous multi-agent systems becomes crucial. We develop simulations of agents collaborating on shared objectives to study these security risks and security trade-offs. We focus on scenarios where an attacker compromises one agent, using it to steer the entire system toward misaligned outcomes by corrupting other agents. In this context, we observe infectious malicious prompts - the multi-hop spreading of malicious instructions. To mitigate this risk, we evaluated several strategies: two "vaccination" approaches that insert false memories of safely handling malicious input into the agents' memory stream, and two versions of a generic safety instruction strategy. While these defenses reduce the spread and fulfillment of malicious instructions in our experiments, they tend to decrease collaboration capability in the agent network. Our findings illustrate potential trade-off between security and collaborative efficiency in multi-agent systems, providing insights for designing more secure yet effective AI collaborations. 

**Abstract (ZH)**: 随着人工智能代理被越来越多地用于协同实现复杂的任务目标，确保自主多智能体系统的安全性变得至关重要。我们开发了模拟代理协作实现共享目标的仿真，以研究这些安全风险和安全权衡。我们专注于一种场景，在这种场景中，攻击者控制一个代理，利用该代理引导整个系统朝目标偏差的方向发展，从而破坏其他代理。在这种背景下，我们观察到恶意指令的传染性——多跳传播的恶意指令。为了减轻这一风险，我们评估了多种策略：两种“疫苗”方法，即向代理的记忆流中插入虚假记忆，使其认为已经安全地处理了恶意输入，以及两种通用安全性指令策略的版本。虽然这些防御措施在我们的实验中减轻了恶意指令的传播与实现，但它们往往会降低代理网络的协作能力。我们的研究结果展示了在多智能体系统中安全性和协作效率之间的潜在权衡，并为设计更安全且有效的AI协作提供了洞见。 

---
# A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Management 

**Title (ZH)**: 通过LLM辅助知识库管理的多agent系统时间规划框架 

**Authors**: Enrico Saccon, Ahmet Tikna, Davide De Martini, Edoardo Lamon, Luigi Palopoli, Marco Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2502.19135)  

**Abstract**: This paper presents a novel framework, called PLANTOR (PLanning with Natural language for Task-Oriented Robots), that integrates Large Language Models (LLMs) with Prolog-based knowledge management and planning for multi-robot tasks. The system employs a two-phase generation of a robot-oriented knowledge base, ensuring reusability and compositional reasoning, as well as a three-step planning procedure that handles temporal dependencies, resource constraints, and parallel task execution via mixed-integer linear programming. The final plan is converted into a Behaviour Tree for direct use in ROS2. We tested the framework in multi-robot assembly tasks within a block world and an arch-building scenario. Results demonstrate that LLMs can produce accurate knowledge bases with modest human feedback, while Prolog guarantees formal correctness and explainability. This approach underscores the potential of LLM integration for advanced robotics tasks requiring flexible, scalable, and human-understandable planning. 

**Abstract (ZH)**: 本文提出了一种新的框架，称为PLANTOR（基于自然语言的面向任务机器人规划），该框架将大型语言模型（LLMs）与基于Prolog的知识管理和规划相结合，用于多机器人任务。该系统采用了两阶段机器人定向知识库生成方式，确保了可重用性和组合推理能力，以及包含三个步骤的规划流程，通过混合整数线性规划处理时间依赖性、资源约束和并行任务执行问题。最终的计划被转换成行为树，可以直接用于ROS2。我们在块世界和拱结构建造场景中的多机器人装配任务中测试了该框架。结果表明，LLMs能够在适度的人工反馈下生成准确的知识库，而Prolog则保证了形式正确性和可解释性。这一方法突显了LLMs在实现复杂、可扩展且易于人类理解的规划方面潜在的优势，适用于高级机器人任务。 

---
# Nexus: A Lightweight and Scalable Multi-Agent Framework for Complex Tasks Automation 

**Title (ZH)**: Nexus：一种轻量级且可扩展的多代理框架，用于复杂任务自动化 

**Authors**: Humza Sami, Mubashir ul Islam, Samy Charas, Asav Gandhi, Pierre-Emmanuel Gaillardon, Valerio Tenace  

**Link**: [PDF](https://arxiv.org/pdf/2502.19091)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have substantially evolved Multi-Agent Systems (MASs) capabilities, enabling systems that not only automate tasks but also leverage near-human reasoning capabilities. To achieve this, LLM-based MASs need to be built around two critical principles: (i) a robust architecture that fully exploits LLM potential for specific tasks -- or related task sets -- and ($ii$) an effective methodology for equipping LLMs with the necessary capabilities to perform tasks and manage information efficiently. It goes without saying that a priori architectural designs can limit the scalability and domain adaptability of a given MAS.
To address these challenges, in this paper we introduce Nexus: a lightweight Python framework designed to easily build and manage LLM-based MASs. Nexus introduces the following innovations: (i) a flexible multi-supervisor hierarchy, (ii) a simplified workflow design, and (iii) easy installation and open-source flexibility: Nexus can be installed via pip and is distributed under a permissive open-source license, allowing users to freely modify and extend its capabilities.
Experimental results demonstrate that architectures built with Nexus exhibit state-of-the-art performance across diverse domains. In coding tasks, Nexus-driven MASs achieve a 99% pass rate on HumanEval and a flawless 100% on VerilogEval-Human, outperforming cutting-edge reasoning language models such as o3-mini and DeepSeek-R1. Moreover, these architectures display robust proficiency in complex reasoning and mathematical problem solving, achieving correct solutions for all randomly selected problems from the MATH dataset. In the realm of multi-objective optimization, Nexus-based architectures successfully address challenging timing closure tasks on designs from the VTR benchmark suite, while guaranteeing, on average, a power saving of nearly 30%. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展显著提升了多代理系统（MASs）的能力，使其不仅能够自动化任务，还能利用接近人类的推理能力。为了实现这一点，基于LLM的MASs需要围绕两个关键原则进行构建：（i）一个稳健的架构，充分利用LLM在特定任务或相关任务集中的潜力；和（ii）一种有效的方案，使LLM具备必要的能力，以高效地执行任务和管理信息。不言而喻，先验的设计可能会限制给定MAS的扩展性和领域适应性。

为了解决这些问题，在本文中我们提出了Nexus：一个轻量级的Python框架，用于轻松构建和管理基于LLM的MASs。Nexus的创新包括：（i）灵活的多监督器层次结构，（ii）简化的流程设计，以及（iii）易于安装和开源灵活性：Nexus可以通过pip安装，并采用宽松的开源许可协议进行分发，允许用户自由修改和扩展其功能。

实验结果表明，使用Nexus构建的架构在多个领域都表现出最先进的性能。在编程任务中，Nexus驱动的MASs在HumanEval基准测试中达到了99%的通过率，并在VerilogEval-Human基准测试中达到了100%的满分，超越了诸如o3-mini和DeepSeek-R1等最先进的推理语言模型。此外，这些架构在复杂的推理和数学问题解决方面表现出了强大的能力，对于从MATH数据集中随机选择的所有问题都实现了正确的解决方案。在多目标优化领域，基于Nexus的架构成功应对了VTR基准套件中的复杂定时闭合任务，并平均实现了近30%的功率节省。 

---
# REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems 

**Title (ZH)**: REALM-Bench：面向LLMs和多智能体系统的实时规划基准测试 

**Authors**: Longling Geng, Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18836)  

**Abstract**: This benchmark suite provides a comprehensive evaluation framework for assessing both individual LLMs and multi-agent systems in real-world planning scenarios. The suite encompasses eleven designed problems that progress from basic to highly complex, incorporating key aspects such as multi-agent coordination, inter-agent dependencies, and dynamic environmental disruptions. Each problem can be scaled along three dimensions: the number of parallel planning threads, the complexity of inter-dependencies, and the frequency of unexpected disruptions requiring real-time adaptation. The benchmark includes detailed specifications, evaluation metrics, and baseline implementations using contemporary frameworks like LangGraph, enabling rigorous testing of both single-agent and multi-agent planning capabilities. Through standardized evaluation criteria and scalable complexity, this benchmark aims to drive progress in developing more robust and adaptable AI planning systems for real-world applications. 

**Abstract (ZH)**: 本基准套件为评估单一大规模语言模型（LLM）和多智能体系统在现实规划场景中的表现提供了一个全面的评估框架。该套件包含十一个设计问题，从基础问题逐步过渡到高度复杂的复杂问题，涵盖了多智能体协调、智能体间依赖关系以及动态环境干扰等关键方面。每个问题可以在三个维度上进行扩展：并行规划线程的数量、相互依赖关系的复杂性以及需要实时适应的意外干扰的频率。基准测试包括详细的规范、评价指标以及使用现代框架（如LangGraph）的基线实现，从而能够对单一智能体和多智能体规划能力进行严格的测试。通过标准化的评估标准和可扩展的复杂性，该基准测试旨在推动开发更 robust 和适应性强的 AI 规划系统，以应用于实际场景。 

---
# Data-Efficient Multi-Agent Spatial Planning with LLMs 

**Title (ZH)**: 基于LLM的高数据效率多智能体空间规划 

**Authors**: Huangyuan Su, Aaron Walsman, Daniel Garces, Sham Kakade, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2502.18822)  

**Abstract**: In this project, our goal is to determine how to leverage the world-knowledge of pretrained large language models for efficient and robust learning in multiagent decision making. We examine this in a taxi routing and assignment problem where agents must decide how to best pick up passengers in order to minimize overall waiting time. While this problem is situated on a graphical road network, we show that with the proper prompting zero-shot performance is quite strong on this task. Furthermore, with limited fine-tuning along with the one-at-a-time rollout algorithm for look ahead, LLMs can out-compete existing approaches with 50 times fewer environmental interactions. We also explore the benefits of various linguistic prompting approaches and show that including certain easy-to-compute information in the prompt significantly improves performance. Finally, we highlight the LLM's built-in semantic understanding, showing its ability to adapt to environmental factors through simple prompts. 

**Abstract (ZH)**: 在本项目中，我们的目标是探讨如何利用预训练大规模语言模型的世界知识，以实现多智能体决策中的高效和稳健学习。我们在一个出租车路线分配问题上进行研究，该问题要求智能体决定如何最有效地接载乘客，以最小化整体等待时间。虽然该问题基于图形化的道路网络，我们发现通过适当的提示，预训练模型在该任务上的零样本性能非常强劲。此外，通过有限的微调与一次接一个智能体的前瞻算法（one-at-a-time rollout algorithm）相结合，语言模型可以在仅需现有方法五分之一的环境交互次数的情况下，超越现有方法。我们还探讨了不同语言提示方法的优势，并表明将某些易于计算的信息包含在提示中可以显著提高性能。最后，我们展示了语言模型内置的语义理解能力，表明通过简单的提示，它可以适应环境因素。 

---
# TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation 

**Title (ZH)**: TrajLLM：一种模块化的大语言模型增强型基于代理的现实人类轨迹仿真框架 

**Authors**: Chenlu Ju, Jiaxin Liu, Shobhit Sinha, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18712)  

**Abstract**: This work leverages Large Language Models (LLMs) to simulate human mobility, addressing challenges like high costs and privacy concerns in traditional models. Our hierarchical framework integrates persona generation, activity selection, and destination prediction, using real-world demographic and psychological data to create realistic movement patterns. Both physical models and language models are employed to explore and demonstrate different methodologies for human mobility simulation. By structuring data with summarization and weighted density metrics, the system ensures scalable memory management while retaining actionable insights. Preliminary results indicate that LLM-driven simulations align with observed real-world patterns, offering scalable, interpretable insights for social problems such as urban planning, traffic management, and public health. The framework's ability to dynamically generate personas and activities enables it to provide adaptable and realistic daily routines. This study demonstrates the transformative potential of LLMs in advancing mobility modeling for societal and urban applications. The source code and interactive demo for our framework are available at this https URL. 

**Abstract (ZH)**: 本研究利用大规模语言模型（LLMs）模拟人类移动，解决了传统模型中成本高和隐私保护等挑战。我们的分层框架整合了个性生成、活动选择和目的地预测，并利用真实世界的人口统计和心理数据创建真实的人类移动模式。在使用物理模型和语言模型探索和演示不同的移动模式模拟方法的同时，系统通过使用摘要技术和加权密度度量来结构化数据，确保了可扩展的记忆管理并保留了可操作的信息。初步结果表明，由LLM驱动的模拟与现实世界观察到的模式一致，为城市规划、交通管理以及公共健康等社会问题提供了可扩展且可解释的见解。该框架能够动态生成个性和活动，使其能够提供适应性强且真实的日常活动模式。本研究展示了LLMs在推进社会和城市应用中的移动建模方面的变革潜力。我们的框架的源代码和互动演示可在以下链接获取：[这个网址]。 

---
# Independent Mobility GPT (IDM-GPT): A Self-Supervised Multi-Agent Large Language Model Framework for Customized Traffic Mobility Analysis Using Machine Learning Models 

**Title (ZH)**: 独立移动GPT（IDM-GPT）：一种基于自我监督的多代理大型语言模型框架，用于使用机器学习模型进行定制化交通移动分析 

**Authors**: Fengze Yang, Xiaoyue Cathy Liu, Lingjiu Lu, Bingzhang Wang, Chenxi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18652)  

**Abstract**: With the urbanization process, an increasing number of sensors are being deployed in transportation systems, leading to an explosion of big data. To harness the power of this vast transportation data, various machine learning (ML) and artificial intelligence (AI) methods have been introduced to address numerous transportation challenges. However, these methods often require significant investment in data collection, processing, storage, and the employment of professionals with expertise in transportation and ML. Additionally, privacy issues are a major concern when processing data for real-world traffic control and management. To address these challenges, the research team proposes an innovative Multi-agent framework named Independent Mobility GPT (IDM-GPT) based on large language models (LLMs) for customized traffic analysis, management suggestions, and privacy preservation. IDM-GPT efficiently connects users, transportation databases, and ML models economically. IDM-GPT trains, customizes, and applies various LLM-based AI agents for multiple functions, including user query comprehension, prompts optimization, data analysis, model selection, and performance evaluation and enhancement. With IDM-GPT, users without any background in transportation or ML can efficiently and intuitively obtain data analysis and customized suggestions in near real-time based on their questions. Experimental results demonstrate that IDM-GPT delivers satisfactory performance across multiple traffic-related tasks, providing comprehensive and actionable insights that support effective traffic management and urban mobility improvement. 

**Abstract (ZH)**: 随着城市化进程的推进，越来越多的传感器被部署在交通系统中，产生了大量数据。为了利用这些海量的交通数据，各种机器学习（ML）和人工智能（AI）方法被引入以应对各种交通挑战。然而，这些方法往往需要大量投资于数据采集、处理、存储以及需要具备交通和ML专业知识的专业人员。此外，在处理用于实时交通控制和管理的数据时，隐私问题也是一项主要关注点。为了解决这些挑战，研究团队提出了一种基于大规模语言模型（LLMs）的创新多代理框架，名为独立移动GPT（IDM-GPT），用于定制化的交通分析、管理建议以及隐私保护。IDM-GPT有效地将用户、交通数据库和ML模型经济地连接起来。IDM-GPT训练、定制并应用于多种基于LLM的AI代理进行多种功能，包括用户查询理解、提示优化、数据分析、模型选择、以及性能评估和提升。借助IDM-GPT，即使没有交通或ML背景的用户，也可以基于问题获得高效且直观的数据分析和个性化建议，实现近乎实时的响应。实验结果表明，IDM-GPT在多个与交通相关的任务中提供了令人满意的表现，提供了全面且可操作的见解，支持有效的交通管理和城市流动性改进。 

---
# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models 

**Title (ZH)**: 《嗨，机器人：基于层次视觉-语言-行动模型的开放指令跟随》 

**Authors**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19417)  

**Abstract**: Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping. 

**Abstract (ZH)**: 能够在开放环境中执行多种不同任务的通用机器人不仅需要推理出完成目标所需的步骤，还必须能够处理复杂指令、提示以及在任务执行过程中提供的反馈。复杂的指令（例如，“能为我做一个素食三明治吗？”或“我不喜欢那个”）不仅需要执行个体步骤的能力，还需要将复杂的命令和反馈置于物理世界中。在这项工作中，我们描述了一个利用视觉-语言模型分层结构的系统，首先通过推理复杂的提示和用户反馈来推导出完成任务的最优下一步，然后通过低级动作执行该步骤。与仅能执行简单指令（如“拿起杯子”）的方法不同，我们的系统能够通过复杂的提示进行推理，并在任务执行过程中整合环境反馈（如“那不是垃圾”）。我们将在三个不同的机器人平台上评估该系统，包括单臂机器人、双臂机器人和双臂移动机器人，展示了其完成清理杂乱的桌子、制作三明治和采购杂货等任务的能力。 

---
# Combining Planning and Reinforcement Learning for Solving Relational Multiagent Domains 

**Title (ZH)**: 将规划与强化学习相结合解决关系型多智能体领域问题 

**Authors**: Nikhilesh Prabhakar, Ranveer Singh, Harsha Kokel, Sriraam Natarajan, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.19297)  

**Abstract**: Multiagent Reinforcement Learning (MARL) poses significant challenges due to the exponential growth of state and action spaces and the non-stationary nature of multiagent environments. This results in notable sample inefficiency and hinders generalization across diverse tasks. The complexity is further pronounced in relational settings, where domain knowledge is crucial but often underutilized by existing MARL algorithms. To overcome these hurdles, we propose integrating relational planners as centralized controllers with efficient state abstractions and reinforcement learning. This approach proves to be sample-efficient and facilitates effective task transfer and generalization. 

**Abstract (ZH)**: 多智能体强化学习（MARL）由于状态空间和动作空间的指数增长以及多智能体环境的非站定特性，面临着重大挑战。这导致了显著的样本效率低下，并阻碍了在多样任务上的泛化。在关系型设置中，这一复杂性进一步增加，此时领域知识至关重要但常常被现有的MARL算法所忽视。为克服这些难题，我们提议将关系规划器作为中心控制器与高效的状态抽象结合使用，并与强化学习相结合。这种 approach 证明具有样本高效性，并促进了有效的任务迁移和泛化。 

---
# EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region 

**Title (ZH)**: EMT：阿拉伯海湾地区自动驾驶的多任务视觉基准数据集 

**Authors**: Nadya Abdel Madjid, Murad Mebrahtu, Abdelmoamen Nasser, Bilal Hassan, Naoufel Werghi, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2502.19260)  

**Abstract**: This paper introduces the Emirates Multi-Task (EMT) dataset - the first publicly available dataset for autonomous driving collected in the Arab Gulf region. The EMT dataset captures the unique road topology, high traffic congestion, and distinctive characteristics of the Gulf region, including variations in pedestrian clothing and weather conditions. It contains over 30,000 frames from a dash-camera perspective, along with 570,000 annotated bounding boxes, covering approximately 150 kilometers of driving routes. The EMT dataset supports three primary tasks: tracking, trajectory forecasting and intention prediction. Each benchmark dataset is complemented with corresponding evaluations: (1) multi-agent tracking experiments, focusing on multi-class scenarios and occlusion handling; (2) trajectory forecasting evaluation using deep sequential and interaction-aware models; and (3) intention benchmark experiments conducted for predicting agents intentions from observed trajectories. The dataset is publicly available at this https URL, and pre-processing scripts along with evaluation models can be accessed at this https URL. 

**Abstract (ZH)**: 本文介绍了阿拉伯湾地区首个公开的自动驾驶数据集——Emirates Multi-Task (EMT) 数据集。EMT 数据集捕捉了阿拉伯湾地区的独特道路拓扑结构、高交通拥挤情况以及该地区的特色，包括行人的服饰差异和天气条件的变化。该数据集包含超过 30,000 帧驾驶视角的图像，以及 570,000 个标注的边界框，覆盖约 150 公里的驾驶路线。EMT 数据集支持三项主要任务：追踪、轨迹预测和意图预测。每个基准数据集都配备了相应的评估方法：(1) 多智能体追踪实验，关注多类别场景和遮挡处理；(2) 使用深度序列和交互感知模型的轨迹预测评估；以及 (3) 从观测轨迹预测智能体意图的意图基准实验。该数据集在以下链接公开获取：此 [https URL]，预处理脚本和评估模型则可以在以下链接访问：此 [https URL]。 

---
# Voting or Consensus? Decision-Making in Multi-Agent Debate 

**Title (ZH)**: 投票还是共识？多智能体辩论中的决策机制 

**Authors**: Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.19130)  

**Abstract**: Much of the success of multi-agent debates depends on carefully choosing the right parameters. Among them, the decision-making protocol stands out. Systematic comparison of decision protocols is difficult because studies alter multiple discussion parameters beyond the protocol. So far, it has been largely unknown how decision-making addresses the challenges of different tasks. This work systematically evaluates the impact of seven decision protocols (e.g., majority voting, unanimity consensus). We change only one variable at a time (i.e., decision protocol) to analyze how different methods affect the collaboration between agents and test different protocols on knowledge (MMLU, MMLU-Pro, GPQA) and reasoning datasets (StrategyQA, MuSR, SQuAD 2.0). Our results show that voting protocols improve performance by 13.2% in reasoning tasks and consensus protocols by 2.8% in knowledge tasks over the other decision protocol. Increasing the number of agents improves performance, while more discussion rounds before voting reduces it. To improve decision-making by increasing answer diversity, we propose two new methods, All-Agents Drafting (AAD) and Collective Improvement (CI). Our methods improve task performance by up to 3.3% with AAD and up to 7.4% with CI. This work demonstrates the importance of decision-making in multi-agent debates beyond scaling. 

**Abstract (ZH)**: 多智能体辩论的成功很大程度上取决于正确选择参数。其中，决策协议尤为关键。由于研究中通常会同时改变多个讨论参数而不仅仅是协议本身，因此系统性的比较这些协议比较困难。到目前为止，决策方法如何应对不同任务的挑战尚不明确。本研究系统地评估了七种决策协议（如多数投票、一致同意）的影响。我们每次只改变一个变量（即决策协议），以分析不同的方法如何影响智能体之间的协作，并在知识（MMLU、MMLU-Pro、GPQA）和推理数据集（StrategyQA、MuSR、SQuAD 2.0）上测试不同的协议。结果显示，投票协议在推理任务中的性能提高了13.2%，而一致协议在知识任务中的性能提高了2.8%。增加智能体的数量可以提高性能，而在投票前进行更多的讨论回合则会降低性能。为了通过增加答案多样性来改进决策，我们提出了两种新方法：全员草案（All-Agents Drafting, AAD）和集体改进（Collective Improvement, CI）。我们的方法分别通过AAD将任务性能提高了3.3%，通过CI提高了7.4%。本研究证明，多智能体辩论中的决策方法的重要性远远超出了简单的扩展。 

---
# Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning 

**Title (ZH)**: 将强化学习算法.cli融入基于上下文的模型导向规划中 

**Authors**: Jaehyeon Son, Soochan Lee, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.19009)  

**Abstract**: Recent studies have shown that Transformers can perform in-context reinforcement learning (RL) by imitating existing RL algorithms, enabling sample-efficient adaptation to unseen tasks without parameter updates. However, these models also inherit the suboptimal behaviors of the RL algorithms they imitate. This issue primarily arises due to the gradual update rule employed by those algorithms. Model-based planning offers a promising solution to this limitation by allowing the models to simulate potential outcomes before taking action, providing an additional mechanism to deviate from the suboptimal behavior. Rather than learning a separate dynamics model, we propose Distillation for In-Context Planning (DICP), an in-context model-based RL framework where Transformers simultaneously learn environment dynamics and improve policy in-context. We evaluate DICP across a range of discrete and continuous environments, including Darkroom variants and Meta-World. Our results show that DICP achieves state-of-the-art performance while requiring significantly fewer environment interactions than baselines, which include both model-free counterparts and existing meta-RL methods. 

**Abstract (ZH)**: 近期的研究表明，Transformer能够在上下文环境下通过模仿现有的强化学习（RL）算法来进行强化学习，从而在不需要参数更新的情况下，高效地适应未见过的任务。然而，这些模型也会继承它们所模仿的RL算法中的次优行为。这一问题主要源于这些算法采用的逐步更新规则。基于模型的规划方法为解决这一局限性提供了可能的解决方案，因为这种方法允许模型在采取行动之前模拟潜在的结果，提供了一种额外的机制来避免次优行为。我们不是学习单独的动力学模型，而是提出了在上下文环境下同时学习环境动力学和改进策略的Distillation for In-Context Planning（DICP）框架。我们在一系列离散和连续环境（包括不同的Darkroom变体和Meta-World）中对DICP进行了评估。实验结果表明，DICP不仅能达到现有的最先进技术的性能水平，而且所需的环境交互次数远低于基线方法，这些基线方法包括无模型方法和现有的元强化学习方法。 

---
# A Multi-Agent DRL-Based Framework for Optimal Resource Allocation and Twin Migration in the Multi-Tier Vehicular Metaverse 

**Title (ZH)**: 基于多智能体深度强化学习的多层车辆元宇宙资源最优分配与双生迁移框架 

**Authors**: Nahom Abishu Hayla, A. Mohammed Seid, Aiman Erbad, Tilahun M. Getu, Ala Al-Fuqaha, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2502.19004)  

**Abstract**: Although multi-tier vehicular Metaverse promises to transform vehicles into essential nodes -- within an interconnected digital ecosystem -- using efficient resource allocation and seamless vehicular twin (VT) migration, this can hardly be achieved by the existing techniques operating in a highly dynamic vehicular environment, since they can hardly balance multi-objective optimization problems such as latency reduction, resource utilization, and user experience (UX). To address these challenges, we introduce a novel multi-tier resource allocation and VT migration framework that integrates Graph Convolutional Networks (GCNs), a hierarchical Stackelberg game-based incentive mechanism, and Multi-Agent Deep Reinforcement Learning (MADRL). The GCN-based model captures both spatial and temporal dependencies within the vehicular network; the Stackelberg game-based incentive mechanism fosters cooperation between vehicles and infrastructure; and the MADRL algorithm jointly optimizes resource allocation and VT migration in real time. By modeling this dynamic and multi-tier vehicular Metaverse as a Markov Decision Process (MDP), we develop a MADRL-based algorithm dubbed the Multi-Objective Multi-Agent Deep Deterministic Policy Gradient (MO-MADDPG), which can effectively balances the various conflicting objectives. Extensive simulations validate the effectiveness of this algorithm that is demonstrated to enhance scalability, reliability, and efficiency while considerably improving latency, resource utilization, migration cost, and overall UX by 12.8%, 9.7%, 14.2%, and 16.1%, respectively. 

**Abstract (ZH)**: 尽管多层次 vehicular Metaverse 有望通过高效的资源分配和无缝的 vehicular twin (VT) 迁移，将车辆转变为互联数字生态系统中的关键节点，现有的技术在高度动态的 vehicular 环境下难以实现这一点，因为它们很难在减少延迟、资源利用和用户体验 (UX) 等多目标优化问题之间取得平衡。为应对这些挑战，我们提出了一种新颖的多层次资源分配和 VT 迁移框架，该框架整合了 Graph Convolutional Networks（GCNs）、基于分层 Stackelberg 博弈的激励机制以及 Multi-Agent Deep Reinforcement Learning（MADRL）。

基于 GCN 的模型可以捕获 vehicular 网络中的时空依赖关系；基于 Stackelberg 博弈的激励机制促进了车辆与基础设施之间的合作；MADRL 算法可实现实时资源分配和 VT 迁移的联合优化。通过将这一动态的多层次 vehicular Metaverse 模型化为马尔可夫决策过程（MDP），我们开发了一个基于 MADRL 的算法，即 Multi-Objective Multi-Agent Deep Deterministic Policy Gradient（MO-MADDPG），该算法能够有效平衡各种相互冲突的目标。广泛仿真实验验证了该算法的有效性，该算法在延迟、资源利用、迁移成本和总体 UX 方面分别提高了 12.8%、9.7%、14.2% 和 16.1%，显著增强了可扩展性、可靠性和效率。 

---
# Learning Autonomy: Off-Road Navigation Enhanced by Human Input 

**Title (ZH)**: 增强人类输入的离路导航自主学习 

**Authors**: Akhil Nagariya, Dimitar Filev, Srikanth Saripalli, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18760)  

**Abstract**: In the area of autonomous driving, navigating off-road terrains presents a unique set of challenges, from unpredictable surfaces like grass and dirt to unexpected obstacles such as bushes and puddles. In this work, we present a novel learning-based local planner that addresses these challenges by directly capturing human driving nuances from real-world demonstrations using only a monocular camera. The key features of our planner are its ability to navigate in challenging off-road environments with various terrain types and its fast learning capabilities. By utilizing minimal human demonstration data (5-10 mins), it quickly learns to navigate in a wide array of off-road conditions. The local planner significantly reduces the real world data required to learn human driving preferences. This allows the planner to apply learned behaviors to real-world scenarios without the need for manual fine-tuning, demonstrating quick adjustment and adaptability in off-road autonomous driving technology. 

**Abstract (ZH)**: 在自动驾驶领域，穿越非 paved 地形为车辆带来了独特的挑战，包括多变的路面（如草地和泥土）以及突如其来的障碍物（如灌木丛和水坑）。本文中，我们提出了一个新颖的学习型局部路径规划器，该规划器通过仅使用单目摄像头的数据直接捕捉真实世界中的驾驶细微差异，以应对这些挑战。我们的规划器的关键特征在于其能够处理各种非 paved 地形条件下的挑战性环境，并且具有快速学习的能力。通过使用少量的人类演示数据（5-10 分钟），该规划器能够迅速学会如何在多种非 paved 地形条件下驾驶。该局部规划器极大地减少了学习人类驾驶偏好的实际所需数据量。这使得规划器能够在无需人工微调的情况下将学习到的行为应用到实际场景中，展示了在非 paved 地形自动驾驶技术中的快速调整和适应能力。 

---
# Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support 

**Title (ZH)**: 辅助还是扰乱？探索和评估主动AI编程支持的设计及其权衡 

**Authors**: Kevin Pu, Daniel Lazaro, Ian Arawjo, Haijun Xia, Ziang Xiao, Tovi Grossman, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18658)  

**Abstract**: AI programming tools enable powerful code generation, and recent prototypes attempt to reduce user effort with proactive AI agents, but their impact on programming workflows remains unexplored. We introduce and evaluate Codellaborator, a design probe LLM agent that initiates programming assistance based on editor activities and task context. We explored three interface variants to assess trade-offs between increasingly salient AI support: prompt-only, proactive agent, and proactive agent with presence and context (Codellaborator). In a within-subject study (N=18), we find that proactive agents increase efficiency compared to prompt-only paradigm, but also incur workflow disruptions. However, presence indicators and \revise{interaction context support} alleviated disruptions and improved users' awareness of AI processes. We underscore trade-offs of Codellaborator on user control, ownership, and code understanding, emphasizing the need to adapt proactivity to programming processes. Our research contributes to the design exploration and evaluation of proactive AI systems, presenting design implications on AI-integrated programming workflow. 

**Abstract (ZH)**: AI 编程工具能够生成强大的代码，最近的原型尝试通过主动的AI代理减少用户的 effort。然而，它们对编程工作流的影响尚未得到探索。我们介绍并评估了 Codellaborator，这是一种基于编辑器活动和任务上下文自动启动编程辅助的 LLM 代理。我们探讨了三种界面变体，以评估逐渐显性的 AI 支持之间的权衡：仅提示、主动代理和带有存在感和交互上下文支持的主动代理（Codellaborator）。在一项针对单被试的用户研究（N=18）中，我们发现主动代理相比于仅提示的范式能够提高效率，但也导致了工作流程中断。然而，存在感指示器和交互上下文支持减轻了中断，并提高了用户对 AI 过程的意识。我们强调 Codellaborator 在用户控制、所有权和代码理解方面的权衡，并强调需要根据不同编程过程调整主动性的必要性。我们的研究为探索和评估主动AI系统的设 计做出了贡献，并提出了 AI 集成编程工作流的设计启示。 

---
# Mind the Gap: Bridging the Divide Between AI Aspirations and the Reality of Autonomous Characterization 

**Title (ZH)**: 注意差距：弥合人工智能期望与自主特征化现实之间的鸿沟 

**Authors**: Grace Guinan, Addison Salvador, Michelle A. Smeaton, Andrew Glaws, Hilary Egan, Brian C. Wyatt, Babak Anasori, Kevin R. Fiedler, Matthew J. Olszta, Steven R. Spurgeon  

**Link**: [PDF](https://arxiv.org/pdf/2502.18604)  

**Abstract**: What does materials science look like in the "Age of Artificial Intelligence?" Each materials domain-synthesis, characterization, and modeling-has a different answer to this question, motivated by unique challenges and constraints. This work focuses on the tremendous potential of autonomous characterization within electron microscopy. We present our recent advancements in developing domain-aware, multimodal models for microscopy analysis capable of describing complex atomic systems. We then address the critical gap between the theoretical promise of autonomous microscopy and its current practical limitations, showcasing recent successes while highlighting the necessary developments to achieve robust, real-world autonomy. 

**Abstract (ZH)**: 人工智能时代，材料科学呈现出怎样的面貌？每个材料领域——合成、表征和建模——对这一问题的回答各有千秋，这源于它们各自独特的挑战和限制。本研究重点关注自主表征在电子显微镜中的巨大潜力。我们介绍了在显微镜分析中开发领域感知型多模态模型的最新进展，这些模型能够描述复杂的原子系统。随后，我们探讨了自主显微镜的理论前景与其当前实践限制之间的关键差距，展示了最近取得的成果，并指出了实现可靠、实用的自主性的必要发展。 

---
# MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications 

**Title (ZH)**: MA-GTS：一种用于解决实际应用中复杂图形问题的多智能体框架 

**Authors**: Zike Yuan, Ming Liu, Hui Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.18540)  

**Abstract**: Graph-theoretic problems arise in real-world applications like logistics, communication networks, and traffic optimization. These problems are often complex, noisy, and irregular, posing challenges for traditional algorithms. Large language models (LLMs) offer potential solutions but face challenges, including limited accuracy and input length constraints. To address these challenges, we propose MA-GTS (Multi-Agent Graph Theory Solver), a multi-agent framework that decomposes these complex problems through agent collaboration. MA-GTS maps the implicitly expressed text-based graph data into clear, structured graph representations and dynamically selects the most suitable algorithm based on problem constraints and graph structure scale. This approach ensures that the solution process remains efficient and the resulting reasoning path is interpretable. We validate MA-GTS using the G-REAL dataset, a real-world-inspired graph theory dataset we created. Experimental results show that MA-GTS outperforms state-of-the-art approaches in terms of efficiency, accuracy, and scalability, with strong results across multiple benchmarks (G-REAL 94.2%, GraCoRe 96.9%, NLGraph 98.4%).MA-GTS is open-sourced at this https URL. 

**Abstract (ZH)**: 图论问题在物流、通信网络和交通优化等实际应用中普遍存在。这些问题往往复杂、嘈杂且不规则，给传统的算法带来了挑战。大型语言模型（LLMs）提供了潜在的解决方案，但面临着准确性有限和输入长度限制等挑战。为了应对这些挑战，我们提出了一种名为MA-GTS（多智能体图理论求解器）的多智能体框架，通过智能体协作来分解这些复杂问题。MA-GTS将隐含表示的文字图数据映射为清晰的结构化图表示，并根据问题约束和图结构规模动态选择最合适的算法。这种方法确保了解决过程的高效性，并且推理路径具有可解释性。我们使用自己创建的G-REAL数据集对MA-GTS进行了验证，这是一个基于现实世界的图论数据集。实验结果表明，MA-GTS在效率、准确性和可扩展性方面均优于现有最先进的方法，在多个基准测试中表现出色（G-REAL 94.2%，GraCoRe 96.9%，NLGraph 98.4%）。MA-GTS已开源，可通过以下链接访问：[在这里插入链接]。 

---
# MAFE: Multi-Agent Fair Environments for Decision-Making Systems 

**Title (ZH)**: MAFE：多智能体公平环境决策系统 

**Authors**: Zachary McBride Lazri, Anirudh Nakra, Ivan Brugere, Danial Dervovic, Antigoni Polychroniadou, Furong Huang, Dana Dachman-Soled, Min Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18534)  

**Abstract**: Fairness constraints applied to machine learning (ML) models in static contexts have been shown to potentially produce adverse outcomes among demographic groups over time. To address this issue, emerging research focuses on creating fair solutions that persist over time. While many approaches treat this as a single-agent decision-making problem, real-world systems often consist of multiple interacting entities that influence outcomes. Explicitly modeling these entities as agents enables more flexible analysis of their interventions and the effects they have on a system's underlying dynamics. A significant challenge in conducting research on multi-agent systems is the lack of realistic environments that leverage the limited real-world data available for analysis. To address this gap, we introduce the concept of a Multi-Agent Fair Environment (MAFE) and present and analyze three MAFEs that model distinct social systems. Experimental results demonstrate the utility of our MAFEs as testbeds for developing multi-agent fair algorithms. 

**Abstract (ZH)**: 在静态背景下应用到机器学习（ML）模型的公平性约束已被证明可能随着时间对不同人口群体产生不利影响。为了解决这个问题，新兴的研究侧重于创造能够持久保持公平性的解决方案。虽然许多方法将其视为单代理决策问题，但现实世界的系统通常由多个相互影响的实体组成，这些实体会影响结果。将这些实体明确建模为代理可以更灵活地分析它们的干预及其对系统潜在动态的影响。在多代理系统研究中，一个显著的挑战是缺乏能够利用可用于分析的有限真实世界数据的现实环境。为了解决这一差距，我们引入了多代理公平环境（MAFE）的概念，并介绍了并分析了三种模拟不同社会系统的MAFE。实验结果表明，我们的MAFE为开发多代理公平算法提供了有用的测试平台。 

---
# ARACNE: An LLM-Based Autonomous Shell Pentesting Agent 

**Title (ZH)**: ARACNE：基于LLM的自主 Shell 渗透测试代理 

**Authors**: Tomas Nieponice, Veronica Valeros, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2502.18528)  

**Abstract**: We introduce ARACNE, a fully autonomous LLM-based pentesting agent tailored for SSH services that can execute commands on real Linux shell systems. Introduces a new agent architecture with multi-LLM model support. Experiments show that ARACNE can reach a 60\% success rate against the autonomous defender ShelLM and a 57.58\% success rate against the Over The Wire Bandit CTF challenges, improving over the state-of-the-art. When winning, the average number of actions taken by the agent to accomplish the goals was less than 5. The results show that the use of multi-LLM is a promising approach to increase accuracy in the actions. 

**Abstract (ZH)**: 我们介绍了ARACNE，这是一种专为SSH服务设计的全自动LLM（大型语言模型）基渗透测试代理，能够执行真实的LinuxShell系统命令。ARACNE引入了一种新的代理架构，支持多LLM模型。实验结果显示，ARACNE在对抗自主防御者ShelLM时的成功率为60%，在对抗OVER THE WIRE Bandit CTF挑战时的成功率为57.58%，均优于现有最先进的方法。当获胜时，代理完成目标所需的平均行动次数少于5次。结果表明，多LLM的使用是提高动作准确性的有前景的方法。 

---
# GOD model: Privacy Preserved AI School for Personal Assistant 

**Title (ZH)**: GOD模型：保留隐私的人工智能个人助手学校 

**Authors**: PIN AI Team, Bill Qingyun Sun, Laura Florescu, Boliang Zhang, Regan Peng, Smile Hu, Shouqiao Wang, Ben Wu, Xi Wang, Davide Crapis, Gavin Zhen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18527)  

**Abstract**: Personal AI assistants (e.g., Apple Intelligence, Meta AI) offer proactive recommendations that simplify everyday tasks, but their reliance on sensitive user data raises concerns about privacy and trust. To address these challenges, we introduce the Guardian of Data (GOD), a secure, privacy-preserving framework for training and evaluating AI assistants directly on-device. Unlike traditional benchmarks, the GOD model measures how well assistants can anticipate user needs-such as suggesting gifts-while protecting user data and autonomy. Functioning like an AI school, it addresses the cold start problem by simulating user queries and employing a curriculum-based approach to refine the performance of each assistant. Running within a Trusted Execution Environment (TEE), it safeguards user data while applying reinforcement and imitation learning to refine AI recommendations. A token-based incentive system encourages users to share data securely, creating a data flywheel that drives continuous improvement. By integrating privacy, personalization, and trust, the GOD model provides a scalable, responsible path for advancing personal AI assistants. For community collaboration, part of the framework is open-sourced at this https URL. 

**Abstract (ZH)**: 个人AI助手（例如Apple Intelligence、Meta AI）提供主动推荐，简化日常任务，但它们对敏感用户数据的依赖引发了隐私和信任方面的担忧。为应对这些挑战，我们引入了一种名为Guardian of Data（GOD）的安全、隐私保护框架，用于直接在设备上训练和评估AI助手。与传统的基准测试不同，GOD模型评估助手预测用户需求（例如推荐礼物）的能力，同时保护用户数据和自主权。该框架像一所AI学校，通过模拟用户查询并采用基于 curriculum 的方法来优化每个助手的性能，解决了冷启动问题。运行在受信任的执行环境中（TEE），该框架保障了用户数据的安全性，同时利用强化学习和模仿学习来优化AI建议。基于代币的激励机制鼓励用户安全地共享数据，从而形成一个数据飞轮，推动持续改进。通过整合隐私、个性化和信任，GOD模型提供了一条可扩展且负责任的路径，促进个人AI助手的发展。为促进社区合作，部分框架在以下地址开源：[该 https URL]。 

---
# Reinforcement Learning-based Approach for Vehicle-to-Building Charging with Heterogeneous Agents and Long Term Rewards 

**Title (ZH)**: 基于强化学习的方法：考虑异质代理人和长期奖励的车辆到建筑充电策略 

**Authors**: Fangqi Liu, Rishav Sen, Jose Paolo Talusan, Ava Pettet, Aaron Kandel, Yoshinori Suzue, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18526)  

**Abstract**: Strategic aggregation of electric vehicle batteries as energy reservoirs can optimize power grid demand, benefiting smart and connected communities, especially large office buildings that offer workplace charging. This involves optimizing charging and discharging to reduce peak energy costs and net peak demand, monitored over extended periods (e.g., a month), which involves making sequential decisions under uncertainty and delayed and sparse rewards, a continuous action space, and the complexity of ensuring generalization across diverse conditions. Existing algorithmic approaches, e.g., heuristic-based strategies, fall short in addressing real-time decision-making under dynamic conditions, and traditional reinforcement learning (RL) models struggle with large state-action spaces, multi-agent settings, and the need for long-term reward optimization. To address these challenges, we introduce a novel RL framework that combines the Deep Deterministic Policy Gradient approach (DDPG) with action masking and efficient MILP-driven policy guidance. Our approach balances the exploration of continuous action spaces to meet user charging demands. Using real-world data from a major electric vehicle manufacturer, we show that our approach comprehensively outperforms many well-established baselines and several scalable heuristic approaches, achieving significant cost savings while meeting all charging requirements. Our results show that the proposed approach is one of the first scalable and general approaches to solving the V2B energy management challenge. 

**Abstract (ZH)**: 将电动汽车电池作为能量存储进行战略性聚合可以优化电网需求，使智能互联社区受益，特别是提供工作场所充电的大型办公楼。这涉及到在长时间段内（例如一个月）优化充电和放电，以减少峰值能源成本和净峰值需求，这需要在不确定性条件下做出 Sequential 决策，并考虑延迟和稀疏奖励、连续动作空间以及确保在多种条件下的泛化复杂性。现有的算法方法，例如基于启发式策略，难以应对动态条件下的实时决策，而传统的强化学习（RL）模型则难以处理庞大的状态-动作空间、多智能体设置以及长期奖励优化的需求。为了解决这些问题，我们提出了一种新的 RL 框架，结合了深度确定性策略梯度方法（DDPG）、动作遮掩以及高效的 MILP 驱动策略指导。我们的方法能够在满足用户充电需求的同时探索连续的动作空间。通过使用一家主要电动汽车制造商的真实数据，我们证明了我们的方法在各个方面都全面优于许多已建立的基准方法和几种可扩展的启发式方法，实现了显著的成本节约，同时满足所有充电要求。我们的结果表明，所提出的方法是解决 V2B 能源管理挑战的首批可扩展和通用方法之一。 

---
# A Multi-Agent Framework for Automated Vulnerability Detection and Repair in Solidity and Move Smart Contracts 

**Title (ZH)**: 一种基于多代理系统的Solidity和Move智能合约自动化漏洞检测与修复框架 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18515)  

**Abstract**: The rapid growth of the blockchain ecosystem and the increasing value locked in smart contracts necessitate robust security measures. While languages like Solidity and Move aim to improve smart contract security, vulnerabilities persist. This paper presents Smartify, a novel multi-agent framework leveraging Large Language Models (LLMs) to automatically detect and repair vulnerabilities in Solidity and Move smart contracts. Unlike traditional methods that rely solely on vast pre-training datasets, Smartify employs a team of specialized agents working on different specially fine-tuned LLMs to analyze code based on underlying programming concepts and language-specific security principles. We evaluated Smartify on a dataset for Solidity and a curated dataset for Move, demonstrating its effectiveness in fixing a wide range of vulnerabilities. Our results show that Smartify (Gemma2+codegemma) achieves state-of-the-art performance, surpassing existing LLMs and enhancing general-purpose models' capabilities, such as Llama 3.1. Notably, Smartify can incorporate language-specific knowledge, such as the nuances of Move, without requiring massive language-specific pre-training datasets. This work offers a detailed analysis of various LLMs' performance on smart contract repair, highlighting the strengths of our multi-agent approach and providing a blueprint for developing more secure and reliable decentralized applications in the growing blockchain landscape. We also provide a detailed recipe for extending this to other similar use cases. 

**Abstract (ZH)**: 区块链生态系统的迅速发展和智能合约中锁定价值的不断增加，迫切需要更为稳健的安全措施。尽管像Solidity和Move这样的编程语言旨在改进智能合约的安全性，但仍然存在漏洞。本文提出了一种名为Smartify的新型多代理框架，该框架利用大型语言模型（LLMs）自动检测和修复Solidity和Move智能合约中的漏洞。不同于传统的依赖于广泛预训练数据集的方法，Smartify采用一个专门化的代理团队，各自使用针对特定编程概念和语言特定安全原则微调过的LLMs来分析代码。我们通过对Solidity和Move的精心标注数据集进行了评估，展示了Smartify在多种漏洞修复方面的有效性。我们的结果显示，Smartify（Gemma2+codegemma）在性能上达到了最先进的水平，超越了现有的语言模型并增强了通用模型的能力，如Llama 3.1。值得注意的是，Smartify可以融入语言特定的知识，例如Move的细微差别，而无需进行大规模的语言特定预训练数据集。本文详细分析了多种语言模型在智能合约修复方面的性能，突显了我们多代理方法的优势，并为开发更安全可靠的去中心化应用程序提供了蓝图，特别是在不断增长的区块链领域。我们还提供了一种详细的配方，以扩展到其他类似的应用场景。 

---
