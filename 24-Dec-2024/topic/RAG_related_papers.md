# Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks 

**Title (ZH)**: 面向更稳健的检索增强生成：评估对抗中毒攻击下的RAG性能 

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2412.16708)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications. 

**Abstract (ZH)**: 检索增强生成（RAG）系统作为一种减轻大规模语言模型（LLM）幻觉并提高其在知识密集型领域性能的有前景的解决方案而崭露头角。然而，这些系统容易受到对抗性污染攻击的影响，在这些攻击中，恶意段落被注入检索数据库，可能导致模型生成事实不正确的输出。在本文中，我们研究了RAG系统的检索和生成组件，以了解如何增强其对这些攻击的鲁棒性。从检索的角度来看，我们分析了为什么以及如何检索到对抗性上下文，并评估了检索到段落质量对后续生成的影响。从生成的角度来看，我们评估了是否可以利用LLM的高级批判性思维和内部知识能力来减轻对抗性上下文的影响，即使用怀疑性提示作为一种自我防御机制。我们的实验和发现为设计更安全、更稳健的检索增强框架提供了可操作的见解，铺平了其实用部署在实际应用中的道路。 

---
# AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles 

**Title (ZH)**: 阿尔茨海默病RAG：面向PubMed文章的多模态检索增强生成 

**Authors**: Aritra Kumar Lahiri, Qinmin Vivian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16701)  

**Abstract**: Recent advancements in generative AI have flourished the development of highly adept Large Language Models (LLMs) that integrate diverse data types to empower decision-making. Among these, Multimodal Retrieval-Augmented Generation (RAG) applications are promising for their capability to combine the strengths of information retrieval and generative models, enhancing their utility across various domains, including biomedical research. This paper introduces AlzheimerRAG, a Multimodal RAG pipeline tool for biomedical research use cases, primarily focusing on Alzheimer's disease from PubMed articles. Our pipeline incorporates multimodal fusion techniques to integrate textual and visual data processing by efficiently indexing and accessing vast amounts of biomedical literature. Preliminary experimental results against benchmarks, such as BioASQ and PubMedQA, have returned improved results in information retrieval and synthesis of domain-specific information. We also demonstrate a case study with our RAG pipeline across different Alzheimer's clinical scenarios. We infer that AlzheimerRAG can generate responses with accuracy non-inferior to humans and with low rates of hallucination. Overall, a reduction in cognitive task load is observed, which allows researchers to gain multimodal insights, improving understanding and treatment of Alzheimer's disease. 

**Abstract (ZH)**: 近年来，生成型人工智能的最新进展促进了高度擅长的大规模语言模型（Large Language Models, LLMs）的发展，这些模型能够综合各种数据类型，以增强决策支持能力。其中，多模态检索增强生成（Multimodal Retrieval-Augmented Generation, RAG）应用因其结合信息检索和生成模型的优势而充满潜力，从而在包括生物医学研究等各个领域中提高了其应用价值。本文介绍了一款名为AlzheimerRAG的多模态RAG流水线工具，主要应用于PubMed文献中的阿尔茨海默病研究场景。我们的流水线采用了多模态融合技术，通过高效地索引和访问大量生物医学文献，实现了文本和视觉数据的综合处理。初步实验结果表明，我们的流水线在生物医学领域的信息检索和相关信息综合方面均优于BioASQ和PubMedQA等基准。我们还展示了在不同阿尔茨海默病临床场景中使用我们RAG流水线的案例研究。我们推断，AlzheimerRAG可以生成准确性不低于人类的响应，并且产生幻觉的概率较低。总体而言，认知任务负担的减少使得研究人员能够获得多模态洞察，从而提高对阿尔茨海默病的理解和治疗。 

---
# Large Language Model Can Be a Foundation for Hidden Rationale-Based Retrieval 

**Title (ZH)**: 大型语言模型可以作为基于隐含推理的检索的基础 

**Authors**: Luo Ji, Feixiang Guo, Teng Chen, Qingqing Gu, Xiaoyu Wang, Ningyuan Xi, Yihong Wang, Peng Yu, Yue Zhao, Hongyang Lei, Zhonglin Jiang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16615)  

**Abstract**: Despite the recent advancement in Retrieval-Augmented Generation (RAG) systems, most retrieval methodologies are often developed for factual retrieval, which assumes query and positive documents are semantically similar. In this paper, we instead propose and study a more challenging type of retrieval task, called hidden rationale retrieval, in which query and document are not similar but can be inferred by reasoning chains, logic relationships, or empirical experiences. To address such problems, an instruction-tuned Large language model (LLM) with a cross-encoder architecture could be a reasonable choice. To further strengthen pioneering LLM-based retrievers, we design a special instruction that transforms the retrieval task into a generative task by prompting LLM to answer a binary-choice question. The model can be fine-tuned with direct preference optimization (DPO). The framework is also optimized for computational efficiency with no performance degradation. We name this retrieval framework by RaHoRe and verify its zero-shot and fine-tuned performance superiority on Emotional Support Conversation (ESC), compared with previous retrieval works. Our study suggests the potential to employ LLM as a foundation for a wider scope of retrieval tasks. Our codes, models, and datasets are available on this https URL. 

**Abstract (ZH)**: 尽管近年来检索增强生成（RAG）系统取得了进展，大多数检索方法仍侧重于事实检索，假设查询和正文档在语义上相似。在本文中，我们提出并研究了一种更为具有挑战性的检索任务类型，称为隐藏理性检索，其中查询和文档不相似，但可以通过推理链、逻辑关系或经验推断出来。为了解决这些问题，带有交叉编码器架构的指令调节大规模语言模型（LLM）是一种合理的选择。为了进一步加强基于LLM的检索器，我们设计了一种特别的指令，通过提示LLM回答一个二元选择问题，将检索任务转化为生成任务。该模型可以使用直接偏好优化（DPO）进行微调。该框架在计算效率上进行了优化，且不会降低性能。我们将这种检索框架命名为RaHoRe，并在情感支持对话（ESC）任务上验证了其零样本和微调后的性能优越性，与之前的检索工作相比具有优势。我们的研究表明，可以将LLM作为更广泛的检索任务的基础。我们的代码、模型和数据集可在以下链接获得：https://... 

---
# RAGONITE: Iterative Retrieval on Induced Databases and Verbalized RDF for Conversational QA over KGs with RAG 

**Title (ZH)**: RAGONITE：基于诱导数据库和自然语言化RDF进行KG上RAG驱动的迭代检索与对话式问答

注释：该翻译旨在保持原论文标题的学术规范和准确性。其中，“RAGONITE”是专有名词，保持不变。“RAG”在此代表“Robustly Augmented Generator”，是亚马逊研究实验室开发的一种对话式问答系统架构。 

**Authors**: Rishiraj Saha Roy, Chris Hinze, Joel Schlotthauer, Farzad Naderi, Viktor Hangya, Andreas Foltyn, Luzian Hahn, Fabian Kuech  

**Link**: [PDF](https://arxiv.org/pdf/2412.17690)  

**Abstract**: Conversational question answering (ConvQA) is a convenient means of searching over RDF knowledge graphs (KGs), where a prevalent approach is to translate natural language questions to SPARQL queries. However, SPARQL has certain shortcomings: (i) it is brittle for complex intents and conversational questions, and (ii) it is not suitable for more abstract needs. Instead, we propose a novel two-pronged system where we fuse: (i) SQL-query results over a database automatically derived from the KG, and (ii) text-search results over verbalizations of KG facts. Our pipeline supports iterative retrieval: when the results of any branch are found to be unsatisfactory, the system can automatically opt for further rounds. We put everything together in a retrieval augmented generation (RAG) setup, where an LLM generates a coherent response from accumulated search results. We demonstrate the superiority of our proposed system over several baselines on a knowledge graph of BMW automobiles. 

**Abstract (ZH)**: 对话式问答（ConvQA）是搜索RDF知识图谱（KGs）的一种便捷方法，其中一种常见的方法是将自然语言问题转换为SPARQL查询。然而，SPARQL也存在一些局限性：（i）它在处理复杂意图和对话式问题时较为脆弱，（ii）它不适用于更抽象的需求。相反，我们提出了一种新的两阶段系统，该系统融合了以下两部分：（i）从知识图谱自动推导出的数据库中的SQL查询结果，以及（ii）对知识图谱事实的文本搜索结果。我们的管道支持迭代检索：当任何分支的结果不满意时，系统可以自动进行进一步的检索。我们将这一切整合到一个检索增强生成（RAG）框架中，在该框架中，大型语言模型（LLM）生成从积累的搜索结果中形成的连贯响应。我们通过对宝马汽车知识图谱的几个基准系统的演示，展示了我们提出的系统的优势。 

---
# HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases 

**Title (ZH)**: HybGRAG：面向文本和关系知识库的混合检索增强生成 

**Authors**: Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa Rangwala, Christos Faloutsos  

**Link**: [PDF](https://arxiv.org/pdf/2412.16311)  

**Abstract**: Given a semi-structured knowledge base (SKB), where text documents are interconnected by relations, how can we effectively retrieve relevant information to answer user questions? Retrieval-Augmented Generation (RAG) retrieves documents to assist large language models (LLMs) in question answering; while Graph RAG (GRAG) uses structured knowledge bases as its knowledge source. However, many questions require both textual and relational information from SKB - referred to as "hybrid" questions - which complicates the retrieval process and underscores the need for a hybrid retrieval method that leverages both information. In this paper, through our empirical analysis, we identify key insights that show why existing methods may struggle with hybrid question answering (HQA) over SKB. Based on these insights, we propose HybGRAG for HQA consisting of a retriever bank and a critic module, with the following advantages: (1) Agentic, it automatically refines the output by incorporating feedback from the critic module, (2) Adaptive, it solves hybrid questions requiring both textual and relational information with the retriever bank, (3) Interpretable, it justifies decision making with intuitive refinement path, and (4) Effective, it surpasses all baselines on HQA benchmarks. In experiments on the STaRK benchmark, HybGRAG achieves significant performance gains, with an average relative improvement in Hit@1 of 51%. 

**Abstract (ZH)**: 给定一个半结构化知识库（SKB），其中文本文档通过关系相互连接，如何有效地检索相关信息以回答用户问题？检索增强生成（RAG）通过检索文档来辅助大型语言模型（LLMs）进行问答；而Graph RAG（GRAG）利用结构化知识库作为其知识来源。然而，许多问题需要从SKB中同时获取文本和关系信息，这些被称为“混合”问题，这使得检索过程复杂化，并强调了需要一种结合这两种信息的混合检索方法。在本文中，通过我们的实证分析，我们识别出关键见解，展示了为什么现有方法可能难以处理SKB上的混合问答（HQA）。基于这些见解，我们提出了一种名为HybGRAG的方法来解决HQA，其包含检索库和批判模块，具有以下优势：（1）自主性，通过批判模块的反馈自动优化输出；（2）自适应性，使用检索库解决需要同时处理文本和关系信息的混合问题；（3）可解释性，通过直观的优化路径来解释决策过程；（4）有效性，在HQA基准测试中，HybGRAG取得了显著性能提升，平均改进精度（Hit@1）为51%。

相关实验在STaRK基准测试上验证了HybGRAG的有效性。结果显示，HybGRAG在HQA基准测试中的表现显著优于所有基线方法，平均相对改进精度（Hit@1）达到了51%。 

---
# LLM Agent for Fire Dynamics Simulations 

**Title (ZH)**: 用于火灾动力学模拟的大型语言模型代理 

**Authors**: Leidong Xu, Danyal Mohaddes, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17146)  

**Abstract**: Significant advances have been achieved in leveraging foundation models, such as large language models (LLMs), to accelerate complex scientific workflows. In this work we introduce FoamPilot, a proof-of-concept LLM agent designed to enhance the usability of FireFOAM, a specialized solver for fire dynamics and fire suppression simulations built using OpenFOAM, a popular open-source toolbox for computational fluid dynamics (CFD). FoamPilot provides three core functionalities: code insight, case configuration and simulation evaluation. Code insight is an alternative to traditional keyword searching leveraging retrieval-augmented generation (RAG) and aims to enable efficient navigation and summarization of the FireFOAM source code for developers and experienced users. For case configuration, the agent interprets user requests in natural language and aims to modify existing simulation setups accordingly to support intermediate users. FoamPilot's job execution functionality seeks to manage the submission and execution of simulations in high-performance computing (HPC) environments and provide preliminary analysis of simulation results to support less experienced users. Promising results were achieved for each functionality, particularly for simple tasks, and opportunities were identified for significant further improvement for more complex tasks. The integration of these functionalities into a single LLM agent is a step aimed at accelerating the simulation workflow for engineers and scientists employing FireFOAM for complex simulations critical for improving fire safety. 

**Abstract (ZH)**: 在利用大型语言模型（LLM）等基础模型加速复杂科学工作流方面取得了显著进展。本文介绍了FoamPilot，这是一种概念验证的LLM代理，旨在提升FireFOAM的易用性，FireFOAM是一个基于OpenFOAM构建的专用求解器，用于火灾动力学和灭火模拟。OpenFOAM是一个流行的开源计算流体动力学（CFD）工具箱。FoamPilot提供了三个核心功能：代码洞察、案例配置和仿真评估。代码洞察利用检索增强生成（RAG）作为一种替代传统的关键词搜索的方法，旨在使开发人员和有经验的用户能够高效地导航和总结FireFOAM的源代码。在案例配置方面，代理以自然语言解释用户请求，并旨在相应地修改现有的仿真设置，以支持中级用户。FoamPilot的任务执行功能旨在管理仿真在高性能计算（HPC）环境中的提交和执行，并对仿真结果进行初步分析，以支持经验较少的用户。对于每个功能，特别是简单任务，我们取得了令人鼓舞的结果，并且识别出了在更复杂任务上进行重大改进的机会。将这些功能集成到一个单一的LLM代理中，是一个旨在加速使用FireFOAM进行复杂仿真（这对于提高火灾安全性至关重要）的工程师和科学家的仿真工作流的步骤。 

---
# Formal Language Knowledge Corpus for Retrieval Augmented Generation 

**Title (ZH)**: 形式语言知识库支持检索增强生成 

**Authors**: Majd Zayyad, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2412.16689)  

**Abstract**: The integration of retrieval-augmented techniques with LLMs has shown promise in improving performance across various domains. However, their utility in tasks requiring advanced reasoning, such as generating and evaluating mathematical statements and proofs, remains underexplored. This study explores the use of Lean, a programming language for writing mathematical proofs, to populate the knowledge corpus used by RAG systems. We hope for this to lay the foundation to exploring different methods of using RAGs to improve the performance of LLMs in advanced logical reasoning tasks. 

**Abstract (ZH)**: 将检索增强技术与大规模语言模型（LLM）的集成已在多个领域提高了性能，显示出很大的潜力。然而，这些技术在要求进行高级推理的任务中（如生成和评估数学陈述和证明）的应用还很少被探索。本研究旨在利用Lean（一种用于编写数学证明的编程语言）来填充检索增强生成（RAG）系统所使用的知识库，希望通过此举为探索不同的RAG使用方法以提高LLM在高级逻辑推理任务中的性能奠定基础。 

---
# TimeRAG: BOOSTING LLM Time Series Forecasting via Retrieval-Augmented Generation 

**Title (ZH)**: TimeRAG：通过检索增强生成提高大规模语言模型的时间序列预测能力 

**Authors**: Silin Yang, Dong Wang, Haoqi Zheng, Ruochun Jin  

**Link**: [PDF](https://arxiv.org/pdf/2412.16643)  

**Abstract**: Although the rise of large language models (LLMs) has introduced new opportunities for time series forecasting, existing LLM-based solutions require excessive training and exhibit limited transferability. In view of these challenges, we propose TimeRAG, a framework that incorporates Retrieval-Augmented Generation (RAG) into time series forecasting LLMs, which constructs a time series knowledge base from historical sequences, retrieves reference sequences from the knowledge base that exhibit similar patterns to the query sequence measured by Dynamic Time Warping (DTW), and combines these reference sequences and the prediction query as a textual prompt to the time series forecasting LLM. Experiments on datasets from various domains show that the integration of RAG improved the prediction accuracy of the original model by 2.97% on average. 

**Abstract (ZH)**: 虽然大型语言模型（LLM）的兴起为时间序列预测带来了新的机遇，但现有的基于LLM的解决方案需要大量的训练，并且表现出有限的迁移性。鉴于这些挑战，我们提出了一种名为TimeRAG的框架，该框架将检索增强生成（RAG）融入时间序列预测的LLM中。TimeRAG从历史序列构建时间序列知识库，通过动态时间规整（DTW）检索与查询序列表现出相似模式的参考序列，并将这些参考序列与预测查询结合成文本提示，传入时间序列预测的LLM。实验结果表明，RAG的集成使原始模型的预测准确率平均提高了2.97%。 

---
# A Reality Check on Context Utilisation for Retrieval-Augmented Generation 

**Title (ZH)**: 对检索增强生成中背景信息利用情况的一种现实检视 

**Authors**: Lovisa Hagström, Sara Vera Marjanović, Haeun Yu, Arnav Arora, Christina Lioma, Maria Maistro, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.17031)  

**Abstract**: Retrieval-augmented generation (RAG) helps address the limitations of the parametric knowledge embedded within a language model (LM). However, investigations of how LMs utilise retrieved information of varying complexity in real-world scenarios have been limited to synthetic contexts. We introduce DRUID (Dataset of Retrieved Unreliable, Insufficient and Difficult-to-understand contexts) with real-world queries and contexts manually annotated for stance. The dataset is based on the prototypical task of automated claim verification, for which automated retrieval of real-world evidence is crucial. We compare DRUID to synthetic datasets (CounterFact, ConflictQA) and find that artificial datasets often fail to represent the complex and diverse real-world context settings. We show that synthetic datasets exaggerate context characteristics rare in real retrieved data, which leads to inflated context utilisation results, as measured by our novel ACU score. Moreover, while previous work has mainly focused on singleton context characteristics to explain context utilisation, correlations between singleton context properties and ACU on DRUID are surprisingly small compared to other properties related to context source. Overall, our work underscores the need for real-world aligned context utilisation studies to represent and improve performance in real-world RAG settings. 

**Abstract (ZH)**: 恢复增强生成（RAG）有助于解决参数知识嵌入在语言模型（LM）中的局限性。然而，关于语言模型在实际场景中如何利用不同复杂度的检索信息的相关研究仅限于合成情境。我们引入了DRUID（Dataset of Retrieved Unreliable, Insufficient and Difficult-to-understand contexts），该数据集包含手动标注立场的现实查询和上下文。该数据集基于自动化声明验证这一典型任务，对于自动化检索现实证据至关重要。我们将DRUID与合成数据集（CounterFact, ConflictQA）进行比较，发现合成数据集往往无法代表复杂的和多变的现实环境背景设置。我们展示了合成数据集夸大了现实中罕见的上下文特征，这导致测量到的上下文利用结果虚高，我们通过新的ACU得分来衡量这一点。此外，虽然之前的研究所主要集中在单个上下文特征的解释上，但DRUID中的单个上下文属性与ACU之间的相关性与其他与上下文来源相关的属性相比出乎意料地小。总体而言，我们的研究强调了在现实世界RAG设置中进行上下文利用研究的必要性，以代表和改进性能。 

---
# Speech Retrieval-Augmented Generation without Automatic Speech Recognition 

**Title (ZH)**: 无需自动语音识别的演讲检索增强生成 

**Authors**: Do June Min, Karel Mundnich, Andy Lapastora, Erfan Soltanmohammadi, Srikanth Ronanki, Kyu Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.16500)  

**Abstract**: One common approach for question answering over speech data is to first transcribe speech using automatic speech recognition (ASR) and then employ text-based retrieval-augmented generation (RAG) on the transcriptions. While this cascaded pipeline has proven effective in many practical settings, ASR errors can propagate to the retrieval and generation steps. To overcome this limitation, we introduce SpeechRAG, a novel framework designed for open-question answering over spoken data. Our proposed approach fine-tunes a pre-trained speech encoder into a speech adapter fed into a frozen large language model (LLM)--based retrieval model. By aligning the embedding spaces of text and speech, our speech retriever directly retrieves audio passages from text-based queries, leveraging the retrieval capacity of the frozen text retriever. Our retrieval experiments on spoken question answering datasets show that direct speech retrieval does not degrade over the text-based baseline, and outperforms the cascaded systems using ASR. For generation, we use a speech language model (SLM) as a generator, conditioned on audio passages rather than transcripts. Without fine-tuning of the SLM, this approach outperforms cascaded text-based models when there is high WER in the transcripts. 

**Abstract (ZH)**: 对于语音数据上的问答任务，一种常见的方法是首先使用自动语音识别（ASR）进行语音转录，然后在转录内容上使用文本检索增强生成（RAG）方法。尽管这种级联管道在许多实际应用中已被证明是有效的，但ASR错误可能会传播到检索和生成步骤。为了克服这一局限，我们提出了一种名为SpeechRAG的新框架，专门用于处理口头数据的开放性问答任务。我们的方法是对预训练的语音编码器进行微调，使其成为嵌入语音适配器并输入冻结的大规模语言模型（LLM）为基础的检索模型。通过对齐文本和语音的嵌入空间，我们的语音检索器可以直接从基于文本查询中检索音频片段，利用冻结文本检索器的检索能力。我们在口语问答数据集上的检索实验表明，直接语音检索不会劣于基于文本的基线方法，并且在使用ASR时，超越了级联系统。在生成阶段，我们使用语音语言模型（SLM）作为生成器，基于音频片段而非转录内容。在不微调SLM的情况下，该方法在转录的错误率较高时，优于基于文本的级联模型。 

---
# A Survey of Query Optimization in Large Language Models 

**Title (ZH)**: 大型语言模型中的查询优化概述 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17558)  

**Abstract**: \textit{Query Optimization} (QO) refers to techniques aimed at enhancing the efficiency and quality of Large Language Models (LLMs) in understanding and answering queries, especially complex ones in scenarios like Retrieval-Augmented Generation (RAG). Specifically, RAG mitigates the limitations of LLMs by dynamically retrieving and leveraging up-to-date relevant information, which provides a cost-effective solution to the challenge of LLMs producing plausible but potentially inaccurate responses. Recently, as RAG evolves and incorporates multiple components that influence its performance, QO has emerged as a critical element, playing a pivotal role in determining the effectiveness of RAG's retrieval stage in accurately sourcing the necessary multiple pieces of evidence to answer queries correctly. In this paper, we trace the evolution of QO techniques by summarizing and analyzing significant studies. Through an organized framework and categorization, we aim to consolidate existing QO techniques in RAG, elucidate their technological foundations, and highlight their potential to enhance the versatility and applications of LLMs. 

**Abstract (ZH)**: 查询优化（Query Optimization, QO）是指旨在提升大型语言模型（Large Language Models, LLMs）理解和回答查询（尤其是复杂的查询，如检索增强生成Retrieval-Augmented Generation, RAG）效率和质量的技术。具体而言，RAG通过动态检索和利用最新的相关信息来减轻LLMs的限制，从而提供了一个成本效益高的解决方案，以应对LLMs生产看似合理但可能存在误差的回应的挑战。近年来，随着RAG的演进和多组件的引入，影响其性能的因素不断增加，QO已成为一个关键要素，对确定RAG检索阶段的有效性起着至关重要的作用，该阶段负责准确地收集必要的多个证据来正确回答查询。本文通过总结和分析相关研究，追踪QO技术的发展。借助系统化的框架和分类，我们旨在汇总现有的RAG中QO技术，阐述其技术基础，并突出这些技术在增强LLMs的灵活性和应用方面的重要潜力。 

---
