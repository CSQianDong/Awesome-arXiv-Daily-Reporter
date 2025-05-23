# Search-o1: Agentic Search-Enhanced Large Reasoning Models 

**Title (ZH)**: Search-o1: 代理增强的大推理模型搜索 

**Authors**: Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2501.05366)  

**Abstract**: Large reasoning models (LRMs) like OpenAI-o1 have demonstrated impressive long stepwise reasoning capabilities through large-scale reinforcement learning. However, their extended reasoning processes often suffer from knowledge insufficiency, leading to frequent uncertainties and potential errors. To address this limitation, we introduce \textbf{Search-o1}, a framework that enhances LRMs with an agentic retrieval-augmented generation (RAG) mechanism and a Reason-in-Documents module for refining retrieved documents. Search-o1 integrates an agentic search workflow into the reasoning process, enabling dynamic retrieval of external knowledge when LRMs encounter uncertain knowledge points. Additionally, due to the verbose nature of retrieved documents, we design a separate Reason-in-Documents module to deeply analyze the retrieved information before injecting it into the reasoning chain, minimizing noise and preserving coherent reasoning flow. Extensive experiments on complex reasoning tasks in science, mathematics, and coding, as well as six open-domain QA benchmarks, demonstrate the strong performance of Search-o1. This approach enhances the trustworthiness and applicability of LRMs in complex reasoning tasks, paving the way for more reliable and versatile intelligent systems. The code is available at \url{this https URL}. 

**Abstract (ZH)**: 大型推理模型（LRMs）如OpenAI-o1通过大规模强化学习展示了令人印象深刻的长步骤推理能力。然而，它们的延长推理过程往往受到知识不足的影响，导致频繁的不确定性甚至潜在错误。为了解决这一限制，我们引入了**Search-o1**，一个通过添加自主检索增强生成（RAG）机制和文档内推理模块来增强LRMs的框架。Search-o1将自主搜索工作流程整合到推理过程中，使LRMs在遇到不确定的知识点时能够动态检索外部知识。此外，由于检索到的文档通常内容丰富，我们设计了一个单独的文档内推理模块，在将这些信息注入推理链之前对其进行深入分析，从而减少噪声并保持连贯的推理流程。通过在科学、数学和编程等复杂推理任务以及六个开放领域问答基准测试中的广泛实验，证明了Search-o1的出色性能。该方法增强了LRMs在复杂推理任务中的可信度和适用性，为更可靠和多功能的智能系统铺平了道路。代码可在[此处](this https URL)获得。 

---
# Online Prompt and Solver Selection for Program Synthesis 

**Title (ZH)**: 程序合成中的在线提示和求解器选择 

**Authors**: Yixuan Li, Lewis Frampton, Federico Mora, Elizabeth Polgreen  

**Link**: [PDF](https://arxiv.org/pdf/2501.05247)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities in the domain of program synthesis. This level of performance is not, however, universal across all tasks, all LLMs and all prompting styles. There are many areas where one LLM dominates, one prompting style dominates, or where calling a symbolic solver is a better choice than an LLM. A key challenge for the user then, is to identify not only when an LLM is the right choice of solver, and the appropriate LLM to call for a given synthesis task, but also the right way to call it. A non-expert user who makes the wrong choice, incurs a cost both in terms of results (number of tasks solved, and the time it takes to solve them) and financial cost, if using a closed-source language model via a commercial API. We frame this choice as an online learning problem. We use a multi-armed bandit algorithm to select which symbolic solver, or LLM and prompt combination to deploy in order to maximize a given reward function (which may prioritize solving time, number of synthesis tasks solved, or financial cost of solving). We implement an instance of this approach, called CYANEA, and evaluate it on synthesis queries from the literature in ranking function synthesis, from the syntax-guided synthesis competition, and fresh, unseen queries generated from SMT problems. CYANEA solves 37.2\% more queries than the best single solver and achieves results within 4\% of the virtual best solver. 

**Abstract (ZH)**: 大型语言模型（LLMs）在程序合成领域展现出了令人印象深刻的性能。然而，这种水平的性能并非在所有任务、所有LLMs和所有提示风格中都普遍存在。在许多领域中，一个LLM表现占优，一个提示风格占优，或者调用符号求解器比使用LLM更为合适。对于用户来说，一个关键挑战在于不仅要识别何时使用LLM作为求解器是正确的选择，并确定针对给定合成任务应该使用哪种特定的LLM及其恰当的提示方式，还要确定正确的调用方法。非专家用户如果做出了错误的选择，将在结果（解决任务的数量和解决时间）和财务成本（如果通过商用API使用封闭源代码的LLM）方面蒙受损失。我们将这一选择过程视为在线学习问题。我们采用多臂老虎机算法来选择部署符号求解器、LLM及其提示组合的方式，以最大化给定的奖励函数（该奖励函数可能侧重于解决时间、合成任务的数量或解决问题的成本）。我们实现了一个名为CYANEA的方法，并在文献中的函数合成查询、语法引导合成竞赛中的查询以及从SMT问题生成的新颖未知查询上对其进行了评估。CYANEA解决了37.2%更多的查询，并且其结果与理想的虚拟最佳求解器相差不超过4%。 

---
# A Text-Based Knowledge-Embedded Soft Sensing Modeling Approach for General Industrial Process Tasks Based on Large Language Model 

**Title (ZH)**: 基于大型语言模型的文本嵌入知识软传感建模方法：适用于通用工业过程任务 

**Authors**: Shuo Tong, Han Liu, Runyuan Guo, Xueqiong Tian, Wenqing Wang, Ding Liu, Youmin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05075)  

**Abstract**: Data-driven soft sensors (DDSS) have become mainstream methods for predicting key performance indicators in process industries. However, DDSS development requires complex and costly customized designs tailored to various tasks during the modeling process. Moreover, DDSS are constrained to a single structured data modality, limiting their ability to incorporate additional contextual knowledge. Furthermore, DDSSs' limited representation learning leads to weak predictive performance with scarce data. To address these challenges, we propose a general framework named LLM-TKESS (large language model for text-based knowledge-embedded soft sensing), harnessing the powerful general problem-solving capabilities, cross-modal knowledge transfer abilities, and few-shot capabilities of LLM for enhanced soft sensing modeling. Specifically, an auxiliary variable series encoder (AVS Encoder) is proposed to unleash LLM's potential for capturing temporal relationships within series and spatial semantic relationships among auxiliary variables. Then, we propose a two-stage fine-tuning alignment strategy: in the first stage, employing parameter-efficient fine-tuning through autoregressive training adjusts LLM to rapidly accommodate process variable data, resulting in a soft sensing foundation model (SSFM). Subsequently, by training adapters, we adapt the SSFM to various downstream tasks without modifying its architecture. Then, we propose two text-based knowledge-embedded soft sensors, integrating new natural language modalities to overcome the limitations of pure structured data models. Furthermore, benefiting from LLM's pre-existing world knowledge, our model demonstrates outstanding predictive capabilities in small sample conditions. Using the thermal deformation of air preheater rotor as a case study, we validate through extensive experiments that LLM-TKESS exhibits outstanding performance. 

**Abstract (ZH)**: 基于数据驱动的软传感器（Data-Driven Soft Sensors, DDSS）已成为过程工业中预测关键性能指标的主要方法。然而，在建模过程中，DDSS的发展需要复杂且昂贵的定制设计，以适应各种任务。此外，DDSS受到单一结构化数据模态的限制，限制了它们整合额外上下文知识的能力。进一步地，DDSS有限的表征学习导致在数据稀缺的情况下预测性能较弱。为了应对这些挑战，我们提出了一种名为LLM-TKESS（大型语言模型用于文本嵌入知识的软感知）的通用框架，利用大型语言模型（LLM）的强大通用问题解决能力、跨模态知识迁移能力和少量样本学习能力，增强软感知建模。具体而言，我们提出了辅助变量系列编码器（Auxiliary Variable Series Encoder, AVS Encoder），以释放LLM捕捉序列内部的时间关系和辅助变量之间空间语义关系的潜力。接着，我们提出了一种两阶段微调对齐策略：在第一阶段，通过自回归训练进行参数高效微调，使LLM能够快速适应过程变量数据，从而形成软传感器基础模型（Soft Sensing Foundation Model, SSFM）。随后，通过训练适配器，我们无需修改其架构即可将SSFM适应各种下游任务。然后，我们提出了两种基于文本嵌入知识的软传感器，将新的自然语言模态集成进来，以克服纯粹结构化数据模型的局限。此外，得益于LLM已有的世界知识，我们的模型在小样本条件下展示了出色的预测能力。以空气预热器转子的热变形为例，通过广泛的实验验证，我们证明了LLM-TKESS表现出色。 

---
# A General Retrieval-Augmented Generation Framework for Multimodal Case-Based Reasoning Applications 

**Title (ZH)**: 一种用于多模态案例推理应用的通用检索增强生成框架 

**Authors**: Ofir Marom  

**Link**: [PDF](https://arxiv.org/pdf/2501.05030)  

**Abstract**: Case-based reasoning (CBR) is an experience-based approach to problem solving, where a repository of solved cases is adapted to solve new cases. Recent research shows that Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) can support the Retrieve and Reuse stages of the CBR pipeline by retrieving similar cases and using them as additional context to an LLM query. Most studies have focused on text-only applications, however, in many real-world problems the components of a case are multimodal. In this paper we present MCBR-RAG, a general RAG framework for multimodal CBR applications. The MCBR-RAG framework converts non-text case components into text-based representations, allowing it to: 1) learn application-specific latent representations that can be indexed for retrieval, and 2) enrich the query provided to the LLM by incorporating all case components for better context. We demonstrate MCBR-RAG's effectiveness through experiments conducted on a simplified Math-24 application and a more complex Backgammon application. Our empirical results show that MCBR-RAG improves generation quality compared to a baseline LLM with no contextual information provided. 

**Abstract (ZH)**: 案例基于推理（CBR）是一种基于经验的解决问题方法，在这种方法中，通过调整已解决的案例库来解决新问题。最近的研究表明，附有检索增强生成（RAG）的大语言模型（LLMs）可以支持CBR工作流中的检索和重用阶段，通过检索相似的案例，并将其作为附加上下文提供给LLM查询。大多数研究集中在纯文本应用上，然而，在许多实际问题中，案例的组件是多模态的。在本文中，我们提出了MCBR-RAG，这是一种适用于多模态CBR应用的一般RAG框架。MCBR-RAG框架将非文本案例组件转换为文本表示，使其能够：1）学习特定应用的潜在表示，这些表示可以进行索引以供检索，2）通过结合所有案例组件来丰富对LLM的查询，提供更好的上下文。我们通过在简化版的Math-24应用和更复杂的背投棋应用中进行的实验，证明了MCBR-RAG的有效性。我们的实证结果表明，MCBR-RAG在提供上下文信息的情况下相比没有提供上下文信息的基线LLM，提高了生成质量。 

---
# AI-Driven Reinvention of Hydrological Modeling for Accurate Predictions and Interpretation to Transform Earth System Modeling 

**Title (ZH)**: 基于AI驱动的水文 modeling 重塑，以实现精确预测和解释，从而转变地球系统 modeling 

**Authors**: Cuihui Xia, Lei Yue, Deliang Chen, Yuyang Li, Hongqiang Yang, Ancheng Xue, Zhiqiang Li, Qing He, Guoqing Zhang, Dambaru Ballab Kattel, Lei Lei, Ming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.04733)  

**Abstract**: Traditional equation-driven hydrological models often struggle to accurately predict streamflow in challenging regional Earth systems like the Tibetan Plateau, while hybrid and existing algorithm-driven models face difficulties in interpreting hydrological behaviors. This work introduces HydroTrace, an algorithm-driven, data-agnostic model that substantially outperforms these approaches, achieving a Nash-Sutcliffe Efficiency of 98% and demonstrating strong generalization on unseen data. Moreover, HydroTrace leverages advanced attention mechanisms to capture spatial-temporal variations and feature-specific impacts, enabling the quantification and spatial resolution of streamflow partitioning as well as the interpretation of hydrological behaviors such as glacier-snow-streamflow interactions and monsoon dynamics. Additionally, a large language model (LLM)-based application allows users to easily understand and apply HydroTrace's insights for practical purposes. These advancements position HydroTrace as a transformative tool in hydrological and broader Earth system modeling, offering enhanced prediction accuracy and interpretability. 

**Abstract (ZH)**: 传统由方程驱动的水文模型在挑战性的地理区域（如青藏高原）中往往难以准确预测径流，而现有的算法驱动模型则在解释水文行为方面遇到了困难。本研究介绍了一种名为HydroTrace的算法驱动、数据无关模型，该模型显著优于现有方法，实现了Nash-Sutcliffe 效率为98%的优异性能，并且在未见数据上表现出强大的泛化能力。此外，HydroTrace通过利用先进的注意力机制来捕捉空间-时间变化和特征特定影响，从而能够定量并空间细化径流分配，并解释冰川-雪-径流相互作用和季风动态等水文行为。此外，基于大型语言模型（LLM）的应用使用户能够轻松理解和应用于实际目的HydroTrace的见解。这些进步将HydroTrace定位为水文及更广泛地球系统建模领域的变革性工具，提供更高的预测准确性和可解释性。 

---
# A survey of textual cyber abuse detection using cutting-edge language models and large language models 

**Title (ZH)**: 使用尖端语言模型和大型语言模型的文本网络欺凌检测综述 

**Authors**: Jose A. Diaz-Garcia, Joao Paulo Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2501.05443)  

**Abstract**: The success of social media platforms has facilitated the emergence of various forms of online abuse within digital communities. This abuse manifests in multiple ways, including hate speech, cyberbullying, emotional abuse, grooming, and sexting. In this paper, we present a comprehensive analysis of the different forms of abuse prevalent in social media, with a particular focus on how emerging technologies, such as Language Models (LMs) and Large Language Models (LLMs), are reshaping both the detection and generation of abusive content within these networks. We delve into the mechanisms through which social media abuse is perpetuated, exploring the psychological and social impact. Additionally, we examine the dual role of advanced language models-highlighting their potential to enhance automated detection systems for abusive behavior while also acknowledging their capacity to generate harmful content. This paper aims to contribute to the ongoing discourse on online safety and ethics, offering insights into the evolving landscape of cyberabuse and the technological innovations that both mitigate and exacerbate it. 

**Abstract (ZH)**: 社交媒体平台的成功促进了数字社区中各种形式在线虐待的出现。这种虐待以多种形式表现出来，包括 hateful 话语、网络暴力、情感虐待、诱骗和裸聊。在本文中，我们对社交媒体中存在的不同形式的虐待进行了全面分析，并特别关注新兴技术如语言模型（LMs）和大规模语言模型（LLMs）如何重塑这些网络中虐待内容的检测与生成。我们探讨了社交媒体虐待行为传播的机制，分析了其心理和社会影响。此外，我们还考察了高级语言模型的双重角色——既强调它们可以增强自动化检测系统以识别有害行为，同时也承认它们有生成有害内容的能力。本文旨在为有关在线安全和伦理的持续讨论做出贡献，提供有关网络虐待演变景观及技术进步对这种演变影响的见解。 

---
# Large Physics Models: Towards a collaborative approach with Large Language Models and Foundation Models 

**Title (ZH)**: 大型物理模型：与大型语言模型和基础模型合作的方法研究 

**Authors**: Kristian G. Barman, Sascha Caron, Emily Sullivan, Henk W. de Regt, Roberto Ruiz de Austri, Mieke Boon, Michael Färber, Stefan Fröse, Faegheh Hasibi, Andreas Ipp, Rukshak Kapoor, Gregor Kasieczka, Daniel Kostić, Michael Krämer, Tobias Golling, Luis G. Lopez, Jesus Marco, Sydney Otten, Pawel Pawlowski, Pietro Vischia, Erik Weber, Christoph Weniger  

**Link**: [PDF](https://arxiv.org/pdf/2501.05382)  

**Abstract**: This paper explores ideas and provides a potential roadmap for the development and evaluation of physics-specific large-scale AI models, which we call Large Physics Models (LPMs). These models, based on foundation models such as Large Language Models (LLMs) - trained on broad data - are tailored to address the demands of physics research. LPMs can function independently or as part of an integrated framework. This framework can incorporate specialized tools, including symbolic reasoning modules for mathematical manipulations, frameworks to analyse specific experimental and simulated data, and mechanisms for synthesizing theories and scientific literature. We begin by examining whether the physics community should actively develop and refine dedicated models, rather than relying solely on commercial LLMs. We then outline how LPMs can be realized through interdisciplinary collaboration among experts in physics, computer science, and philosophy of science. To integrate these models effectively, we identify three key pillars: Development, Evaluation, and Philosophical Reflection. Development focuses on constructing models capable of processing physics texts, mathematical formulations, and diverse physical data. Evaluation assesses accuracy and reliability by testing and benchmarking. Finally, Philosophical Reflection encompasses the analysis of broader implications of LLMs in physics, including their potential to generate new scientific understanding and what novel collaboration dynamics might arise in research. Inspired by the organizational structure of experimental collaborations in particle physics, we propose a similarly interdisciplinary and collaborative approach to building and refining Large Physics Models. This roadmap provides specific objectives, defines pathways to achieve them, and identifies challenges that must be addressed to realise physics-specific large scale AI models. 

**Abstract (ZH)**: 本文探讨了物理专用大型人工智能模型（我们称之为大型物理模型LPMs）的发展和评估理念，并提供了一个潜在的发展路线图。这些模型基于大型语言模型（LLMs）——通过广泛数据训练而来——并将针对物理研究的需求进行定制。LPMs可以独立运行，也可以作为集成框架的一部分。这个框架可以整合专门的工具，包括用于数学运算的符号推理模块，用于分析特定实验和模拟数据的框架，以及用于合成理论和科学文献的机制。首先，我们将探讨物理学界是否应该积极开发和优化专用模型，而不仅仅是依赖商业LLMs。然后，我们概述了LPMs可以通过物理学、计算机科学和科学哲学专家之间的跨学科合作来实现的方法。为了有效地整合这些模型，我们确定了三个关键支柱：发展、评估和哲学反思。发展侧重于构建能够处理物理文本、数学公式和各种物理数据的模型。评估通过测试和基准评估来评估准确性和可靠性。最后，哲学反思涵盖了对LLMs在物理学中的更广泛影响的分析，包括它们产生新的科学理解的潜力，以及可能在研究中出现的新协作模式。借鉴粒子物理实验合作的组织结构，我们提出了一种跨学科和协作的方法来构建和优化大型物理模型。这份路线图为实现这些特定于物理的大型AI模型设定了具体目标，定义了实现这些目标的途径，并识别了必须解决的挑战。 

---
# Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction 

**Title (ZH)**: 流动对齐器：通过分布诱导实现高效的句级对齐 

**Authors**: Hantao Lou, Jiaming Ji, Kaile Wang, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05336)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to significant improvements in their capabilities, but also to increased concerns about their alignment with human values and intentions. Current alignment strategies, including adaptive training and inference-time methods, have demonstrated potential in this area. However, these approaches still struggle to balance deployment complexity and capability across various tasks and difficulties. In this work, we introduce the Streaming Distribution Induce Aligner (Stream Aligner), a novel alignment paradigm that combines efficiency with enhanced performance in various tasks throughout the generation process. Stream Aligner achieves dynamic sentence-level correction by using a small model to learn the preferences of the suffix sentence, iteratively correcting the suffix sentence output by the upstream model, and then using the corrected sentence to replace the suffix sentence in subsequent generations. Compared to Aligner, our experiments demonstrate that Stream Aligner reduces reliance on the capabilities of additional models, enhances the reasoning abilities of LLMs, and decreases latency during user interaction. Specifically, Stream Aligner-2B model has achieved an improvement of 76.1% in helpfulness, 36.0% in harmlessness on the tested Llama2-70B-chat model, and Stream Aligner-8B has achieved an improvement of 3.5% on the math ability of the tested Llama3-70B-Instruct model. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展在其能力上带来了显著提升，但也引发了对其与人类价值观和意图一致性方面的担忧。当前的对齐策略，包括适应性训练和推理时的方法，已经在这一领域显示出潜力。然而，这些方法仍然难以在各种任务和难度下平衡部署复杂性和模型能力。在此项工作中，我们提出了流式分布诱导对齐器（Stream Aligner），这是一种新颖的对齐范式，能够通过生成过程中的效率与性能提升相结合的方式实现动态的句子级别修正。Stream Aligner 通过使用一个小模型学习后缀句子的偏好，并迭代地对上游模型生成的后缀句子输出进行修正，然后使用修正后的句子替换后续生成中的后缀句子，从而实现动态的句子级别修正。与原来的 Aligner 相比，我们的实验结果显示，Stream Aligner 在减少对额外模型能力的依赖、增强 LLM 的推理能力以及降低用户交互延迟方面表现出色。具体而言，Stream Aligner-2B 模型在测试的 Llama2-70B-chat 模型上，帮助性方面提升了 76.1%，有害性方面降低了 36.0%；而 Stream Aligner-8B 模型在测试的 Llama3-70B-Instruct 模型的数学能力上提升了 3.5%。 

---
# Deriving Coding-Specific Sub-Models from LLMs using Resource-Efficient Pruning 

**Title (ZH)**: 从大规模语言模型中高效裁剪衍生出编码专用子模型 

**Authors**: Laura Puccioni, Alireza Farshin, Mariano Scazzariello, Changjie Wang, Marco Chiesa, Dejan Kostic  

**Link**: [PDF](https://arxiv.org/pdf/2501.05248)  

**Abstract**: Large Language Models (LLMs) have demonstrated their exceptional performance in various complex code generation tasks. However, their broader adoption is limited by significant computational demands and high resource requirements, particularly memory and processing power. To mitigate such requirements, model pruning techniques are used to create more compact models with significantly fewer parameters. However, current approaches do not focus on the efficient extraction of programming-language-specific sub-models. In this work, we explore the idea of efficiently deriving coding-specific sub-models through unstructured pruning (i.e., Wanda). We investigate the impact of different domain-specific calibration datasets on pruning outcomes across three distinct domains and extend our analysis to extracting four language-specific sub-models: Python, Java, C++, and JavaScript. We are the first to efficiently extract programming-language-specific sub-models using appropriate calibration datasets while maintaining acceptable accuracy w.r.t. full models. We are also the first to provide analytical evidence that domain-specific tasks activate distinct regions within LLMs, supporting the creation of specialized sub-models through unstructured pruning. We believe that this work has significant potential to enhance LLM accessibility for coding by reducing computational requirements to enable local execution on consumer-grade hardware, and supporting faster inference times critical for real-time development feedback. 

**Abstract (ZH)**: 大语言模型（LLMs）在各种复杂的代码生成任务中展现了其卓越的表现。然而，其更广泛的采用受到显著的计算需求和高资源要求的限制，尤其是内存和处理能力的需要。为了减轻这种需求，通过模型剪枝技术可以创建更加紧凑的模型，具有显著减少的参数量。然而，当前的方法并未专注于高效提取编程语言特定的子模型。在本研究中，我们探索了通过无结构剪枝（例如Wanda）高效提取编程特定的子模型的想法。我们研究了不同领域特定校准数据集对剪枝结果的影响，并跨越三个不同的领域进行了分析。我们还进一步分析了提取出四种语言特定的子模型：Python、Java、C++ 和 JavaScript。我们首次通过合适的校准数据集高效地提取出了编程语言特定的子模型，同时在与完整模型相比较的情况下保持了可接受的准确性。我们也是首次提供实证分析证据，证明领域特定的任务在大语言模型中激活了不同的区域，支持通过无结构剪枝创建专门的子模型。我们相信，这项工作具有显著的潜力，可以通过减少计算需求来增强LLM在编程中的可访问性，使其能够在消费级硬件上进行局部执行，并支持实时开发反馈至关重要的快速推断时间。 

---
# Optimizing Estonian TV Subtitles with Semi-supervised Learning and LLMs 

**Title (ZH)**: 使用半监督学习和大规模语言模型优化爱沙尼亚电视字幕 

**Authors**: Artem Fedorchenko, Tanel Alumäe  

**Link**: [PDF](https://arxiv.org/pdf/2501.05234)  

**Abstract**: This paper presents an approach for generating high-quality, same-language subtitles for Estonian TV content. We fine-tune the Whisper model on human-generated Estonian subtitles and enhance it with iterative pseudo-labeling and large language model (LLM) based post-editing. Our experiments demonstrate notable subtitle quality improvement through pseudo-labeling with an unlabeled dataset. We find that applying LLM-based editing at test time enhances subtitle accuracy, while its use during training does not yield further gains. This approach holds promise for creating subtitle quality close to human standard and could be extended to real-time applications. 

**Abstract (ZH)**: 本文提出了一个生成高质量同语言 Estonia 电视内容字幕的方法。我们通过微调 Whisper 模型，并结合迭代伪标签和基于大语言模型（LLM）的后编辑，提升了字幕质量。实验结果显示，在未标注数据集上应用伪标签可显著提高字幕质量。我们发现，在测试时应用基于大语言模型的编辑可以提升字幕的准确性，而在训练期间使用则不会带来额外的改进。此方法有可能生成接近人类标准的高质量字幕，并有望扩展至实时应用。 

---
# Biomedical Relation Extraction via Adaptive Document-Relation Cross-Mapping and Concept Unique Identifier 

**Title (ZH)**: 通过自适应文档-关系交叉映射和概念唯一标识符进行生物医学关系提取 

**Authors**: Yufei Shang, Yanrong Guo, Shijie Hao, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2501.05155)  

**Abstract**: Document-Level Biomedical Relation Extraction (Bio-RE) aims to identify relations between biomedical entities within extensive texts, serving as a crucial subfield of biomedical text mining. Existing Bio-RE methods struggle with cross-sentence inference, which is essential for capturing relations spanning multiple sentences. Moreover, previous methods often overlook the incompleteness of documents and lack the integration of external knowledge, limiting contextual richness. Besides, the scarcity of annotated data further hampers model training. Recent advancements in large language models (LLMs) have inspired us to explore all the above issues for document-level Bio-RE. Specifically, we propose a document-level Bio-RE framework via LLM Adaptive Document-Relation Cross-Mapping (ADRCM) Fine-Tuning and Concept Unique Identifier (CUI) Retrieval-Augmented Generation (RAG). First, we introduce the Iteration-of-REsummary (IoRs) prompt for solving the data scarcity issue. In this way, Bio-RE task-specific synthetic data can be generated by guiding ChatGPT to focus on entity relations and iteratively refining synthetic data. Next, we propose ADRCM fine-tuning, a novel fine-tuning recipe that establishes mappings across different documents and relations, enhancing the model's contextual understanding and cross-sentence inference capabilities. Finally, during the inference, a biomedical-specific RAG approach, named CUI RAG, is designed to leverage CUIs as indexes for entities, narrowing the retrieval scope and enriching the relevant document contexts. Experiments conducted on three Bio-RE datasets (GDA, CDR, and BioRED) demonstrate the state-of-the-art performance of our proposed method by comparing it with other related works. 

**Abstract (ZH)**: 生物医学文档级关系提取（Bio-RE）旨在识别文本中生物医学实体之间的关系，是生物医学文本挖掘中的关键子领域。现有Bio-RE方法在句间推理方面存在困难，句间关系的捕获至关重要。此外，先前的方法往往忽略了文档的不完整性，缺乏外部知识的整合，限制了上下文的丰富性。此外，标注数据稀缺进一步阻碍了模型的训练。近期在大规模语言模型（LLMs）方面的进展启发我们探索这些问题以改进文档级Bio-RE方法。具体而言，我们提出了一种通过LLMs自适应文档-关系跨映射（ADRCM）微调和概念唯一标识符（CUI）检索增强生成（RAG）的文档级Bio-RE框架。首先，我们介绍了Iterative Relation Extraction Summary（IoRs）提示，以解决数据稀缺问题。通过这种方式，可以引导ChatGPT关注实体关系并迭代优化合成数据，生成任务特定的合成数据。接着，我们提出了ADRCM微调，这是一种新颖的微调方法，能够建立不同文档和关系之间的映射，增强模型的上下文理解和句间推理能力。最后，在推理过程中，我们设计了一种特定于生物医学的RAG方法——CUI RAG，利用CUI作为实体的索引，缩小检索范围并丰富相关文档上下文。在GDA、CDR和BioRED三个Bio-RE数据集上的实验表明，我们的方法在与其他相关工作的比较中表现出最先进的性能。 

---
# Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning 

**Title (ZH)**: 通过视频支撑的蕴含树推理实现常识视频问题解答 

**Authors**: Huabin Liu, Filip Ilievski, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2501.05069)  

**Abstract**: This paper proposes the first video-grounded entailment tree reasoning method for commonsense video question answering (VQA). Despite the remarkable progress of large visual-language models (VLMs), there are growing concerns that they learn spurious correlations between videos and likely answers, reinforced by their black-box nature and remaining benchmarking biases. Our method explicitly grounds VQA tasks to video fragments in four steps: entailment tree construction, video-language entailment verification, tree reasoning, and dynamic tree expansion. A vital benefit of the method is its generalizability to current video and image-based VLMs across reasoning types. To support fair evaluation, we devise a de-biasing procedure based on large-language models that rewrites VQA benchmark answer sets to enforce model reasoning. Systematic experiments on existing and de-biased benchmarks highlight the impact of our method components across benchmarks, VLMs, and reasoning types. 

**Abstract (ZH)**: 本文提出了一种第一个基于视频的蕴含树推理方法，用于常识视频问答（VQA）。尽管大规模视觉语言模型（VLMs）取得了显著的进步，但人们越来越担心它们在视频和可能答案之间学习了虚假的相关性，这种相关性是由它们的黑盒性质和持续存在的基准测试偏差所强化的。我们的方法通过四个步骤显式地将VQA任务与视频片段进行关联：构建蕴含树、视频语言蕴含验证、树推理和动态树扩展。该方法的一个重要优势在于其可以跨不同类型的推理任务将通用性推广到当前的视频和图像基视觉语言模型。为了支持公平的评估，我们基于大型语言模型设计了一种去偏差程序，重新编写VQA基准答案集，以确保模型推理的正确性。对现有基准和去偏差基准进行系统的实验，突显了本方法在不同基准、视觉语言模型和推理类型上的影响。 

---
# Enhancing Human-Like Responses in Large Language Models 

**Title (ZH)**: 增强大型语言模型的人类响应能力 

**Authors**: Ethem Yağız Çalık, Talha Rüzgar Akkuş  

**Link**: [PDF](https://arxiv.org/pdf/2501.05032)  

**Abstract**: This paper explores the advancements in making large language models (LLMs) more human-like. We focus on techniques that enhance natural language understanding, conversational coherence, and emotional intelligence in AI systems. The study evaluates various approaches, including fine-tuning with diverse datasets, incorporating psychological principles, and designing models that better mimic human reasoning patterns. Our findings demonstrate that these enhancements not only improve user interactions but also open new possibilities for AI applications across different domains. Future work will address the ethical implications and potential biases introduced by these human-like attributes. 

**Abstract (ZH)**: 本文探讨了使大型语言模型（LLMs）更加人化的进步。我们重点关注能够增强人工智能系统自然语言理解、对话连贯性和情绪智能的技术。研究评估了各种方法，包括使用多样化的数据集进行微调、纳入心理学原理以及设计更符合人类推理模式的模型。研究发现，这些增强不仅改善了用户体验，还为不同领域的AI应用开辟了新的可能性。未来的研究将关注这些接近人类特性的伦理问题及其潜在的偏差。 

---
# Step-by-Step Mastery: Enhancing Soft Constraint Following Ability of Large Language Models 

**Title (ZH)**: 逐步掌握：提高大型语言模型遵守软约束的能力 

**Authors**: Qingyu Ren, Jie Zeng, Qianyu He, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.04945)  

**Abstract**: It is crucial for large language models (LLMs) to follow instructions that involve multiple constraints. However, soft constraints are semantically related and difficult to verify through automated methods. These constraints remain a significant challenge for LLMs. To enhance the ability of LLMs to follow soft constraints, we initially design a pipeline to obtain high-quality outputs automatically. Additionally, to fully utilize the acquired data, we introduce a training paradigm based on curriculum learning. We experimentally evaluate the effectiveness of our methods in improving LLMs' soft constraint following ability and analyze the factors driving the improvements. The datasets and code are publicly available at this https URL. 

**Abstract (ZH)**: 对于大型语言模型（LLMs）而言，遵循涉及多重约束的指令至关重要。然而，软约束是语义相关的，难以通过自动化方法验证，这些约束仍然是LLMs面临的一个重大挑战。为了增强LLMs遵循软约束的能力，我们首先设计了一个管道以自动获取高质量的输出。此外，为了充分利用获取的数据，我们引入了一种基于渐进学习的训练范式。我们实证评估了我们的方法在提高LLMs遵循软约束能力方面的有效性，并分析了推动这些改进的因素。相关数据集和代码已在以下网址公开：this https URL。 

---
# Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency 

**Title (ZH)**: 通过混合不一致性破解多模态大型语言模型 

**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.04931)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can improve the attack's performance on three benchmarks. In particular, SI-Attack can obviously improve the attack success rate for commercial MLLMs such as GPT-4o or Claude-3.5-Sonnet. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）已在商业应用中取得了显著的性能，并被实际应用，但它们仍然存在潜在的安全机制漏洞。押解攻击是红队测试方法，旨在绕过安全机制并发现MLLMs的潜在风险。现有的MLLMs押解方法通常通过复杂的优化方法或精心设计的图像和文本提示来绕过模型的安全机制。尽管取得了一些进展，但它们在商业闭源MLLMs上的攻击成功率较低。与以往研究不同，我们通过实证研究发现，MLLMs在处理乱序有害指令时存在理解和安全性之间的不一致性（Shuffle Inconsistency）。也就是说，从理解能力的角度来看，MLLMs能够很好地理解乱序有害文本-图像指令。然而，从安全性角度来看，它们却容易被乱序有害指令绕过，从而产生有害的响应。然后，我们创新地提出了一种名为SI-Attack的图文押解攻击方法。具体而言，为了充分利用这种不一致性并克服乱序的随机性，我们应用了一种基于查询的黑盒优化方法，根据有毒法官模型的反馈选择最具有害的乱序输入。一系列实验表明，SI-Attack可以提高在三个基准上的攻击性能。特别是，SI-Attack能够明显提高对如GPT-4o或Claude-3.5-Sonnet等商业MLLMs的攻击成功率。 

---
# Exploring Large Language Models for Semantic Analysis and Categorization of Android Malware 

**Title (ZH)**: 探索大型语言模型在Android恶意软件语义分析与分类中的应用 

**Authors**: Brandon J Walton, Mst Eshita Khatun, James M Ghawaly, Aisha Ali-Gombe  

**Link**: [PDF](https://arxiv.org/pdf/2501.04848)  

**Abstract**: Malware analysis is a complex process of examining and evaluating malicious software's functionality, origin, and potential impact. This arduous process typically involves dissecting the software to understand its components, infection vector, propagation mechanism, and payload. Over the years, deep reverse engineering of malware has become increasingly tedious, mainly due to modern malicious codebases' fast evolution and sophistication. Essentially, analysts are tasked with identifying the elusive needle in the haystack within the complexities of zero-day malware, all while under tight time constraints. Thus, in this paper, we explore leveraging Large Language Models (LLMs) for semantic malware analysis to expedite the analysis of known and novel samples. Built on GPT-4o-mini model, \msp is designed to augment malware analysis for Android through a hierarchical-tiered summarization chain and strategic prompt engineering. Additionally, \msp performs malware categorization, distinguishing potential malware from benign applications, thereby saving time during the malware reverse engineering process. Despite not being fine-tuned for Android malware analysis, we demonstrate that through optimized and advanced prompt engineering \msp can achieve up to 77% classification accuracy while providing highly robust summaries at functional, class, and package levels. In addition, leveraging the backward tracing of the summaries from package to function levels allowed us to pinpoint the precise code snippets responsible for malicious behavior. 

**Abstract (ZH)**: 恶意软件分析是一个复杂的过程，涉及对恶意软件的功能、来源和潜在影响进行检查和评估。这个艰巨的过程通常包括拆解软件，以理解其组件、感染途径、传播机制和载荷。近年来，恶意软件的深度逆向工程变得越来越繁琐，这主要是由于现代恶意代码的快速进化和复杂性。分析师的任务是识别在零日恶意软件的复杂性中的隐秘威胁，同时还要在时间紧迫的情况下完成这项工作。因此，本文探索利用大型语言模型（LLMs）进行语义恶意软件分析，以加速已知和新型样本的分析过程。基于GPT-4o-mini模型，\msp 设计用于通过分层级次的总结链和策略性提示工程增强Android恶意软件分析。此外，\msp 还能进行恶意软件分类，将潜在的恶意软件与良性应用程序区分开来，从而在逆向工程恶意软件的过程中节省时间。尽管没有针对Android恶意软件分析进行微调，我们通过优化和先进的提示工程展示了\msp 可以达到高达77%的分类准确率，并且还能提供功能、类和包三个层次上的高度稳健的总结。此外，利用总结自包级到功能级的回溯跟踪，我们能够精确定位负责恶意行为的代码片段。 

---
# Do Code LLMs Understand Design Patterns? 

**Title (ZH)**: 代码LLM是否理解设计模式？ 

**Authors**: Zhenyu Pan, Xuefeng Song, Yunkun Wang, Rongyu Cao, Binhua Li, Yongbin Li, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.04835)  

**Abstract**: Code Large Language Models (LLMs) demonstrate great versatility in adapting to various downstream tasks, including code generation and completion, as well as bug detection and fixing. However, Code LLMs often fail to capture existing coding standards, leading to the generation of code that conflicts with the required design patterns for a given project. As a result, developers must post-process to adapt the generated code to the project's design norms. In this work, we empirically investigate the biases of Code LLMs in software development. Through carefully designed experiments, we assess the models' understanding of design patterns across recognition, comprehension, and generation. Our findings reveal that biases in Code LLMs significantly affect the reliability of downstream tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在适应各种下游任务方面表现出极大的灵活性，包括代码生成和补全，以及错误检测和修复。然而，代码LLMs往往无法捕捉现有的编码标准，导致生成的代码与给定项目的特定设计模式冲突。因此，开发人员必须对生成的代码进行后处理，以适应项目的规范。在本研究中，我们通过实证研究考察了代码LLMs在软件开发中的偏见。通过精心设计的实验，我们评估了模型在识别、理解和生成阶段对设计模式的理解能力。我们的研究发现，代码LLMs中的偏见显著影响下游任务的可靠性。 

---
# LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation 

**Title (ZH)**: 长文摘: 长上下文语言模型在长程序生成任务上的基准测试 

**Authors**: Xi Ye, Fangcong Yin, Yinghui He, Joie Zhang, Howard Yen, Tianyu Gao, Greg Durrett, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.05414)  

**Abstract**: Existing benchmarks for evaluating long-context language models (LCLMs) primarily focus on long-context recall, requiring models to produce short responses based on a few critical snippets while processing thousands of irrelevant tokens. We introduce LongProc (Long Procedural Generation), a new benchmark that requires both the integration of highly dispersed information and long-form generation. LongProc consists of six diverse procedural generation tasks, such as extracting structured information from HTML pages into a TSV format and executing complex search procedures to create travel plans. These tasks challenge LCLMs by testing their ability to follow detailed procedural instructions, synthesize and reason over dispersed information, and generate structured, long-form outputs (up to 8K tokens). Furthermore, as these tasks adhere to deterministic procedures and yield structured outputs, they enable reliable rule-based evaluation. We evaluate 17 LCLMs on LongProc across three difficulty levels, with maximum numbers of output tokens set at 500, 2K, and 8K. Notably, while all tested models claim a context window size above 32K tokens, open-weight models typically falter on 2K-token tasks, and closed-source models like GPT-4o show significant degradation on 8K-token tasks. Further analysis reveals that LCLMs struggle to maintain long-range coherence in long-form generations. These findings highlight critical limitations in current LCLMs and suggest substantial room for improvement. Data and code available at: this https URL 

**Abstract (ZH)**: 现有用于评估长上下文语言模型（LCLMs）的标准基准主要侧重于长上下文召回率，要求模型在处理数千个无关的标记时，基于几个关键片段生成简短的响应。我们引入了LongProc（长过程生成）这一新的基准，要求模型不仅能够整合高度分散的信息，还能生成长篇形式的内容。LongProc 包括六个多样化的过程生成任务，例如从 HTML 页面中提取结构化信息并转换为 TSV 格式，以及执行复杂的搜索程序以创建旅行计划。这些任务通过测试LCLMs随细节过程指令进行操作的能力、综合和推理分散信息的能力，并生成结构化且长格式的输出（最多 8K 个标记），挑战了LCLMs。此外，由于这些任务遵循确定性过程并产生结构化输出，因此它们能够进行可靠的经验法则评估。我们按照三个难度级别对17种LCLMs进行了LongProc评估，输出标记的最大数量分别为500、2K和8K。值得注意的是，尽管所有测试模型声称上下文窗口大小超过32K标记，但开源模型通常在2K标记任务上表现不佳，而像GPT-4o这样的封闭源代码模型在8K标记任务上的表现显著下降。进一步分析显示，LCLMs在长篇生成中难以保持长程连贯性。这些发现突显了当前LCLMs存在的关键限制，并建议有很大的改进空间。数据和代码可在以下链接获取：this https URL 

---
# Leveraging Large Language Models for Zero-shot Lay Summarisation in Biomedicine and Beyond 

**Title (ZH)**: 利用大型语言模型实现生物医药及更领域中的零样本概要提取 

**Authors**: Tomas Goldsack, Carolina Scarton, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.05224)  

**Abstract**: In this work, we explore the application of Large Language Models to zero-shot Lay Summarisation. We propose a novel two-stage framework for Lay Summarisation based on real-life processes, and find that summaries generated with this method are increasingly preferred by human judges for larger models. To help establish best practices for employing LLMs in zero-shot settings, we also assess the ability of LLMs as judges, finding that they are able to replicate the preferences of human judges. Finally, we take the initial steps towards Lay Summarisation for Natural Language Processing (NLP) articles, finding that LLMs are able to generalise to this new domain, and further highlighting the greater utility of summaries generated by our proposed approach via an in-depth human evaluation. 

**Abstract (ZH)**: 在本研究中，我们探讨了大型语言模型在零样本摘要（零样本Lay摘要）中的应用。我们提出了一个基于现实生活过程的两阶段框架，并发现使用这种方法生成的摘要随着模型规模的增大越来越受到人类评委的青睐。为了帮助建立在零样本设置中利用LLM的最佳实践，我们还评估了LLM作为评委的能力，发现它们能够复制人类评委的偏好。最后，我们向NLP文章的零样本摘要迈出了初步的步骤，发现LLM能够泛化到这一新领域，并通过深入的人类评价进一步突显了我们提出的方法生成的摘要具有更大的实用价值。 

---
# Investigating Numerical Translation with Large Language Models 

**Title (ZH)**: 使用大型语言模型探究数值翻译 

**Authors**: Wei Tang, Jiawei Yu, Yuang Li, Yanqing Zhao, Weidong Zhang, Wei Feng, Min Zhang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04927)  

**Abstract**: The inaccurate translation of numbers can lead to significant security issues, ranging from financial setbacks to medical inaccuracies. While large language models (LLMs) have made significant advancements in machine translation, their capacity for translating numbers has not been thoroughly explored. This study focuses on evaluating the reliability of LLM-based machine translation systems when handling numerical data. In order to systematically test the numerical translation capabilities of currently open source LLMs, we have constructed a numerical translation dataset between Chinese and English based on real business data, encompassing ten types of numerical translation. Experiments on the dataset indicate that errors in numerical translation are a common issue, with most open-source LLMs faltering when faced with our test scenarios. Especially when it comes to numerical types involving large units like ``million", ``billion", and "yi", even the latest llama3.1 8b model can have error rates as high as 20%. Finally, we introduce three potential strategies to mitigate the numerical mistranslations for large units. 

**Abstract (ZH)**: 不准确的数字翻译可能导致严重的安全问题，从财务损失到医疗错误不一而足。虽然大型语言模型（LLMs）在机器翻译方面取得了显著进展，但它们处理数字的能力尚未得到充分研究。本研究重点评估基于LLM的机器翻译系统在处理数值数据时的可靠性。为了系统地测试当前开源LLM的数字翻译能力，我们基于实际业务数据构建了一个中英文数字翻译数据集，涵盖了十种类型的数字翻译。实验表明，数字翻译中的错误是一个常见问题，大多数开源LLM在面对我们的测试场景时表现不佳。特别是在处理“百万”、“亿”等大单位类型的数字时，即使是最新的llama3.1 8b模型，错误率也可能高达20%。最后，我们介绍了三种潜在策略，以减轻大单位数字翻译中的误解问题。 

---
# JELLY: Joint Emotion Recognition and Context Reasoning with LLMs for Conversational Speech Synthesis 

**Title (ZH)**: JELLY：结合LLM的情感识别与情境推理在对话语音合成中的应用 

**Authors**: Jun-Hyeok Cha, Seung-Bin Kim, Hyung-Seok Oh, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.04904)  

**Abstract**: Recently, there has been a growing demand for conversational speech synthesis (CSS) that generates more natural speech by considering the conversational context. To address this, we introduce JELLY, a novel CSS framework that integrates emotion recognition and context reasoning for generating appropriate speech in conversation by fine-tuning a large language model (LLM) with multiple partial LoRA modules. We propose an Emotion-aware Q-former encoder, which enables the LLM to perceive emotions in speech. The encoder is trained to align speech emotions with text, utilizing datasets of emotional speech. The entire model is then fine-tuned with conversational speech data to infer emotional context for generating emotionally appropriate speech in conversation. Our experimental results demonstrate that JELLY excels in emotional context modeling, synthesizing speech that naturally aligns with conversation, while mitigating the scarcity of emotional conversational speech datasets. 

**Abstract (ZH)**: 近年来，随着对对话式语音合成（CSS）需求的增长，这种合成技术越来越注重在考虑对话上下文的情况下生成更为自然的语音。为解决这一问题，我们引入了JELLY——一种新型CSS框架，该框架通过结合情感识别和上下文推理来引导模型生成适当的对话语音，同时对大型语言模型（LLM）进行了微调，使用了多个部分LoRA模块。我们提出了一种情感感知的QFormer编码器，该编码器使LLM能够感知语音中的情感。该编码器经过训练，能够使语音情感与文本情感对齐，并利用了情感语音数据集。整个模型随后通过对话语音数据进行了微调，以推断情感上下文并生成情感恰当的对话语音。实验结果表明，JELLY在情感上下文建模方面表现出色，能合成与对话自然对齐的语音，并缓解了情感对话语音数据集稀缺的问题。 

---
# Leveraging Log Probabilities in Language Models to Forecast Future Events 

**Title (ZH)**: 利用语言模型中的对数概率预测未来事件 

**Authors**: Tommaso Soru, Jim Marshall  

**Link**: [PDF](https://arxiv.org/pdf/2501.04880)  

**Abstract**: In the constantly changing field of data-driven decision making, accurately predicting future events is crucial for strategic planning in various sectors. The emergence of Large Language Models (LLMs) marks a significant advancement in this area, offering advanced tools that utilise extensive text data for prediction. In this industry paper, we introduce a novel method for AI-driven foresight using LLMs. Building on top of previous research, we employ data on current trends and their trajectories for generating forecasts on 15 different topics. Subsequently, we estimate their probabilities via a multi-step approach based on log probabilities. We show we achieve a Brier score of 0.186, meaning a +26% improvement over random chance and a +19% improvement over widely-available AI systems. 

**Abstract (ZH)**: 在不断变化的数据驱动决策领域，准确预测未来事件对于各行业的战略规划至关重要。大型语言模型（LLMs）的出现标志着这一领域的显著进步，为利用大量文本数据进行预测提供了先进的工具。在本文中，我们介绍了一种基于LLMs的新型人工智能驱动 foresight 方法。在前人研究的基础上，我们利用当前趋势及其轨迹数据，对15个不同主题进行预测。随后，我们通过基于对数概率的多步方法估计这些预测的概率。结果显示，我们的方法实现了0.186的Brier分数，这意味着相比随机猜测，改善了26%，相比广泛可用的AI系统，改善了19%。 

---
