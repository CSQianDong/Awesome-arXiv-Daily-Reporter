# AutoLife: Automatic Life Journaling with Smartphones and LLMs 

**Title (ZH)**: AutoLife：利用智能手机和大规模语言模型的自动生活记事 

**Authors**: Huatao Xu, Panron Tong, Mo Li, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2412.15714)  

**Abstract**: This paper introduces a novel mobile sensing application - life journaling - designed to generate semantic descriptions of users' daily lives. We present AutoLife, an automatic life journaling system based on commercial smartphones. AutoLife only inputs low-cost sensor data (without photos or audio) from smartphones and can automatically generate comprehensive life journals for users. To achieve this, we first derive time, motion, and location contexts from multimodal sensor data, and harness the zero-shot capabilities of Large Language Models (LLMs), enriched with commonsense knowledge about human lives, to interpret diverse contexts and generate life journals. To manage the task complexity and long sensing duration, a multilayer framework is proposed, which decomposes tasks and seamlessly integrates LLMs with other techniques for life journaling. This study establishes a real-life dataset as a benchmark and extensive experiment results demonstrate that AutoLife produces accurate and reliable life journals. 

**Abstract (ZH)**: 本文介绍了一种新颖的移动感知应用——生活记录，旨在生成用户日常生活的语义描述。我们提出了AutoLife，一个基于商业智能手机的自动生活记录系统。AutoLife仅需要输入手机的低成本传感器数据（不包括照片或音频），即可自动为用户生成全面的生活记录。为了实现这一点，我们首先从多模态传感器数据中提取时间、运动和位置上下文，并利用大型语言模型（LLMs）的零样本能力，结合关于人类生活的常识知识，来解释多样化的上下文并生成生活记录。为了管理和降低任务复杂性及长时间的感知需求，我们提出了一种多层框架，该框架将任务分解并与其他技术无缝集成，以实现生活记录功能。本研究建立了一个实际数据集作为基准，并通过广泛的实验结果证明了AutoLife能够生成准确可靠的生活记录。 

---
# Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation 

**Title (ZH)**: 大型语言模型能否混淆代码？对大型语言模型生成汇编代码混淆的系统分析 

**Authors**: Seyedreza Mohseni, Seyedali Mohammadi, Deepa Tilwani, Yash Saxena, Gerald Ndwula, Sriram Vema, Edward Raff, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2412.16135)  

**Abstract**: Malware authors often employ code obfuscations to make their malware harder to detect. Existing tools for generating obfuscated code often require access to the original source code (e.g., C++ or Java), and adding new obfuscations is a non-trivial, labor-intensive process. In this study, we ask the following question: Can Large Language Models (LLMs) potentially generate a new obfuscated assembly code? If so, this poses a risk to anti-virus engines and potentially increases the flexibility of attackers to create new obfuscation patterns. We answer this in the affirmative by developing the MetamorphASM benchmark comprising MetamorphASM Dataset (MAD) along with three code obfuscation techniques: dead code, register substitution, and control flow change. The MetamorphASM systematically evaluates the ability of LLMs to generate and analyze obfuscated code using MAD, which contains 328,200 obfuscated assembly code samples. We release this dataset and analyze the success rate of various LLMs (e.g., GPT-3.5/4, GPT-4o-mini, Starcoder, CodeGemma, CodeLlama, CodeT5, and LLaMA 3.1) in generating obfuscated assembly code. The evaluation was performed using established information-theoretic metrics and manual human review to ensure correctness and provide the foundation for researchers to study and develop remediations to this risk. The source code can be found at the following GitHub link: this https URL. 

**Abstract (ZH)**: 恶意软件作者经常使用代码混淆技术来使其恶意软件更难被检测。现有的生成混淆代码的工具通常需要访问原始源代码（例如C++或Java），而添加新的混淆技术则是一个非平凡且劳动密集型的过程。在本研究中，我们提出以下问题：大型语言模型（LLMs）是否有可能生成新的混淆汇编代码？如果可以，这将给反病毒引擎带来风险，并可能增加攻击者创建新混淆模式的灵活性。我们通过开发包含MetamorphASM数据集（MAD）以及三种代码混淆技术（死代码、寄存器替换和控制流变化）的MetamorphASM基准来回答这个问题。MetamorphASM系统地评估了LLMs生成和分析混淆代码的能力，MAD包含328,200个不同的混淆汇编代码样本。我们发布了该数据集，并分析了各种LLMs（例如GPT-3.5/4、GPT-4o-mini、Starcoder、CodeGemma、CodeLlama、CodeT5和LLaMA 3.1）生成混淆汇编代码的成功率。评估使用了现有的信息论度量和人工手动审查来确保正确性，并为研究人员研究和开发相应的缓解措施提供了基础。源代码可以在以下GitHub链接中找到：this https URL。 

---
# WebLLM: A High-Performance In-Browser LLM Inference Engine 

**Title (ZH)**: WebLLM：一种高性能的浏览器内运行的LLM推理引擎 

**Authors**: Charlie F. Ruan, Yucheng Qin, Xun Zhou, Ruihang Lai, Hongyi Jin, Yixin Dong, Bohan Hou, Meng-Shiun Yu, Yiyan Zhai, Sudeep Agarwal, Hangrui Cao, Siyuan Feng, Tianqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.15803)  

**Abstract**: Advancements in large language models (LLMs) have unlocked remarkable capabilities. While deploying these models typically requires server-grade GPUs and cloud-based inference, the recent emergence of smaller open-source models and increasingly powerful consumer devices have made on-device deployment practical. The web browser as a platform for on-device deployment is universally accessible, provides a natural agentic environment, and conveniently abstracts out the different backends from diverse device vendors. To address this opportunity, we introduce WebLLM, an open-source JavaScript framework that enables high-performance LLM inference entirely within web browsers. WebLLM provides an OpenAI-style API for seamless integration into web applications, and leverages WebGPU for efficient local GPU acceleration and WebAssembly for performant CPU computation. With machine learning compilers MLC-LLM and Apache TVM, WebLLM leverages optimized WebGPU kernels, overcoming the absence of performant WebGPU kernel libraries. Evaluations show that WebLLM can retain up to 80% native performance on the same device, with room to further close the gap. WebLLM paves the way for universally accessible, privacy-preserving, personalized, and locally powered LLM applications in web browsers. The code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的进步解锁了非凡的能力。尽管部署这些模型通常需要服务器级GPU和基于云的推理，但最近出现的更小的开源模型和日益强大的消费者设备使设备上部署成为可能。作为设备上部署平台的网络浏览器具有普遍可达性，提供了一个自然的自主环境，并且方便地将不同的后端抽象出来，与各种设备供应商无关。为了应对这一机遇，我们介绍了WebLLM，这是一种开源的JavaScript框架，可以在网络浏览器中实现高性能的LLM推理。WebLLM提供了一种类似OpenAI的API，使无缝集成到Web应用程序成为可能，并利用WebGPU进行高效的地方GPU加速，以及利用WebAssembly进行高性能的CPU计算。通过机器学习编译器MLC-LLM和Apache TVM，WebLLM利用优化的WebGPU内核，克服了缺乏高性能WebGPU内核库的问题。评估结果显示，WebLLM可以在同一设备上保留高达80%的原生性能，并有进一步缩小差距的空间。WebLLM为网络浏览器中的普遍可访问、私有保护、个性化和本地驱动的LLM应用程序铺平了道路。代码可在以下链接获得：this https URL. 

---
# GraphSeqLM: A Unified Graph Language Framework for Omic Graph Learning 

**Title (ZH)**: GraphSeqLM：统一的图语言框架用于omics图学习 

**Authors**: Heming Zhang, Di Huang, Yixin Chen, Fuhai Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15790)  

**Abstract**: The integration of multi-omic data is pivotal for understanding complex diseases, but its high dimensionality and noise present significant challenges. Graph Neural Networks (GNNs) offer a robust framework for analyzing large-scale signaling pathways and protein-protein interaction networks, yet they face limitations in expressivity when capturing intricate biological relationships. To address this, we propose Graph Sequence Language Model (GraphSeqLM), a framework that enhances GNNs with biological sequence embeddings generated by Large Language Models (LLMs). These embeddings encode structural and biological properties of DNA, RNA, and proteins, augmenting GNNs with enriched features for analyzing sample-specific multi-omic data. By integrating topological, sequence-derived, and biological information, GraphSeqLM demonstrates superior predictive accuracy and outperforms existing methods, paving the way for more effective multi-omic data integration in precision medicine. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

多组学数据的集成对于理解复杂疾病至关重要，但其高维度和噪声带来了显著挑战。图神经网络（GNNs）提供了一种分析大规模信号通路和蛋白质-蛋白质相互作用网络的稳健框架，但在捕捉复杂的生物关系时存在表达能力的限制。为解决这一问题，我们提出了一种基于图序列语言模型（GraphSeqLM）的框架，该框架通过大型语言模型（LLMs）生成的生物序列嵌入来增强GNNs。这些嵌入编码DNA、RNA和蛋白质的空间结构和生物学特性，从而为分析样本特异性多组学数据提供了丰富的特征。通过整合拓扑、序列衍生和生物学信息，GraphSeqLM展示了卓越的预测准确性，并优于现有方法，为精准医学中的多组学数据集成提供了更加有效的途径。 

---
# Critique of Impure Reason: Unveiling the reasoning behaviour of medical Large Language Models 

**Title (ZH)**: 《不纯洁的理由：揭开医疗大型语言模型的推理行为》

这个标题翻译成中文后，既保留了原文的意思，也符合学术论文标题的规范。 

**Authors**: Shamus Sim, Tyrone Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.15748)  

**Abstract**: Background: Despite the current ubiquity of Large Language Models (LLMs) across the medical domain, there is a surprising lack of studies which address their reasoning behaviour. We emphasise the importance of understanding reasoning behaviour as opposed to high-level prediction accuracies, since it is equivalent to explainable AI (XAI) in this context. In particular, achieving XAI in medical LLMs used in the clinical domain will have a significant impact across the healthcare sector. Results: Therefore, we define the concept of reasoning behaviour in the specific context of medical LLMs. We then categorise and discuss the current state of the art of methods which evaluate reasoning behaviour in medical LLMs. Finally, we propose theoretical frameworks which can empower medical professionals or machine learning engineers to gain insight into the low-level reasoning operations of these previously obscure models. Conclusion: The subsequent increased transparency and trust in medical machine learning models by clinicians as well as patients will accelerate the integration, application as well as further development of medical AI for the healthcare system as a whole 

**Abstract (ZH)**: 背景：尽管大型语言模型（LLMs）在医学领域已经无处不在，但却缺乏专注于它们推理行为的研究。我们强调理解推理行为的重要性，而非仅仅关注高层预测精度，因为这一点在医学领域等同于可解释的人工智能（XAI）。特别是，实现医学LLMs中的XAI将在整个医疗健康领域产生重大影响。结果：因此，我们定义了医学LLMs中推理行为的特定概念。我们随后对当前用于评估医学LLMs推理行为的方法进行了分类和讨论。最后，我们提出了理论框架，以帮助医疗专业人士或机器学习工程师深入了解这些过去较为隐晦模型的底层推理操作。结论：由此将增加临床医生和患者对医学机器学习模型的透明度和信任度，进而推动整个医疗健康系统的医学AI的集成、应用及其进一步发展。 

---
# HREF: Human Response-Guided Evaluation of Instruction Following in Language Models 

**Title (ZH)**: 人类响应引导的语言模型指令遵循评估 

**Authors**: Xinxi Lyu, Yizhong Wang, Hannaneh Hajishirzi, Pradeep Dasigi  

**Link**: [PDF](https://arxiv.org/pdf/2412.15524)  

**Abstract**: Evaluating the capability of Large Language Models (LLMs) in following instructions has heavily relied on a powerful LLM as the judge, introducing unresolved biases that deviate the judgments from human judges. In this work, we reevaluate various choices for automatic evaluation on a wide range of instruction-following tasks. We experiment with methods that leverage human-written responses and observe that they enhance the reliability of automatic evaluations across a wide range of tasks, resulting in up to a 3.2% improvement in agreement with human judges. We also discovered that human-written responses offer an orthogonal perspective to model-generated responses in following instructions and should be used as an additional context when comparing model responses. Based on these observations, we develop a new evaluation benchmark, Human Response-Guided Evaluation of Instruction Following (HREF), comprising 4,258 samples across 11 task categories with a composite evaluation setup, employing a composite evaluation setup that selects the most reliable method for each category. In addition to providing reliable evaluation, HREF emphasizes individual task performance and is free from contamination. Finally, we study the impact of key design choices in HREF, including the size of the evaluation set, the judge model, the baseline model, and the prompt template. We host a live leaderboard that evaluates LLMs on the private evaluation set of HREF. 

**Abstract (ZH)**: 大型语言模型（LLMs）在遵循指令方面的能力评估很大程度上依赖于一个强大的LLM作为评判者，这引入了未解决的偏差，使判断偏离了人类评判者的标准。在这项工作中，我们重新评估了多种自动评估方法在广泛范围内的指令遵循任务中的适用性。我们尝试利用人工撰写的响应，发现这些方法在广泛任务中提高了自动评估的可靠性，结果与人类评判者的同意率达到最高3.2%的提升。我们还发现，人工撰写的响应为指令遵循提供了与模型生成响应不同的视角，并且在比较模型响应时应被视为额外的上下文。基于这些观察，我们开发了一种新的评估基准——指导性指令遵循的人类响应引导评估（HREF），该基准包括4,258个样本，涵盖11个任务类别，并采用综合评估设置来选择每个类别中最可靠的方法。除了提供可靠的评估外，HREF还强调了个体任务的表现，并且不受污染。最后，我们研究了HREF中关键设计选择的影响，包括评估集的规模、评判模型、基线模型和提示模板。我们提供了一个实时排行榜，用于对HREF的私人评估集进行LLM的评估。 

---
# Humanlike Cognitive Patterns as Emergent Phenomena in Large Language Models 

**Title (ZH)**: 大型语言模型中的人类认知模式作为 emergent 现象 

**Authors**: Zhisheng Tang, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2412.15501)  

**Abstract**: Research on emergent patterns in Large Language Models (LLMs) has gained significant traction in both psychology and artificial intelligence, motivating the need for a comprehensive review that offers a synthesis of this complex landscape. In this article, we systematically review LLMs' capabilities across three important cognitive domains: decision-making biases, reasoning, and creativity. We use empirical studies drawing on established psychological tests and compare LLMs' performance to human benchmarks. On decision-making, our synthesis reveals that while LLMs demonstrate several human-like biases, some biases observed in humans are absent, indicating cognitive patterns that only partially align with human decision-making. On reasoning, advanced LLMs like GPT-4 exhibit deliberative reasoning akin to human System-2 thinking, while smaller models fall short of human-level performance. A distinct dichotomy emerges in creativity: while LLMs excel in language-based creative tasks, such as storytelling, they struggle with divergent thinking tasks that require real-world context. Nonetheless, studies suggest that LLMs hold considerable potential as collaborators, augmenting creativity in human-machine problem-solving settings. Discussing key limitations, we also offer guidance for future research in areas such as memory, attention, and open-source model development. 

**Abstract (ZH)**: 对大型语言模型（LLMs）中涌现模式的研究已在心理学和人工智能领域引起了广泛关注，这激发了对其复杂景观进行全面综述的需求，旨在提供这一领域的综合性的合成。本文系统地回顾了LLMs在三个重要的认知领域的能力：决策偏差、推理和创造力。我们采用了基于已建立的心理学测试的实证研究，将LLMs的表现与人类基准进行比较。在决策方面，我们的综述表明，尽管LLMs表现出几种类似人类的偏差，但一些在人类中观察到的偏差是不存在的，这表明认知模式与人类决策部分对齐。在推理方面，先进的LLMs如GPT-4表现出类似于人类系统2思维的审慎推理，而较小的模型未能达到人类水平的表现。在创造力方面，LLMs在基于语言的创造性任务方面表现出色，例如讲故事，但在需要现实世界背景的发散思维任务中表现不佳。然而，研究表明，LLMs在人类与机器问题解决设置中的协作中具有巨大的潜力，可以增强创造力。我们还讨论了关键限制，并为未来的研究提供指导，特别是关于记忆、注意力以及开源模型开发的领域。 

---
# Continual Learning Using Only Large Language Model Prompting 

**Title (ZH)**: 仅使用大型语言模型提示的持续学习 

**Authors**: Jiabao Qiu, Zixuan Ke, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15479)  

**Abstract**: We introduce CLOB, a novel continual learning (CL) paradigm wherein a large language model (LLM) is regarded as a black box. Learning is done incrementally via only verbal prompting. CLOB does not fine-tune any part of the LLM or add any trainable parameters to it. It is particularly suitable for LLMs that are accessible via APIs. We also propose a new CL technique, called CIS, based on incremental summarization that also overcomes the LLM's input length limit. Experiments show CIS outperforms baselines by a very large margin. 

**Abstract (ZH)**: 我们引入了CLOB（Continual Learning Box），这是一种新颖的连续学习（Continual Learning, CL）范式，其中大型语言模型（Large Language Model, LLM）被视为一个黑盒。学习是通过仅使用口头提示进行增量学习完成的。CLOB 不对 LLM 的任何部分进行微调，也不为其添加任何可训练参数。它特别适用于通过 API 访问的 LLM。我们还提出了一种新的 CL 技术，称为CIS（Continual Incremental Summarization），该技术基于增量总结，并克服了LLM的输入长度限制。实验结果表明，CIS 在性能上显著优于基线方法。 

---
# Northeastern Uni at Multilingual Counterspeech Generation: Enhancing Counter Speech Generation with LLM Alignment through Direct Preference Optimization 

**Title (ZH)**: 东北师范大学在多语言反制言论生成中的研究：通过直接偏好优化实现语言模型对齐以增强反制言论生成 

**Authors**: Sahil Wadhwa, Chengtian Xu, Haoming Chen, Aakash Mahalingam, Akankshya Kar, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2412.15453)  

**Abstract**: The automatic generation of counter-speech (CS) is a critical strategy for addressing hate speech by providing constructive and informed responses. However, existing methods often fail to generate high-quality, impactful, and scalable CS, particularly across diverse linguistic contexts. In this paper, we propose a novel methodology to enhance CS generation by aligning Large Language Models (LLMs) using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Our approach leverages DPO to align LLM outputs with human preferences, ensuring contextually appropriate and linguistically adaptable responses. Additionally, we incorporate knowledge grounding to enhance the factual accuracy and relevance of generated CS. Experimental results demonstrate that DPO-aligned models significantly outperform SFT baselines on CS benchmarks while scaling effectively to multiple languages. These findings highlight the potential of preference-based alignment techniques to advance CS generation across varied linguistic settings. The model supervision and alignment is done in English and the same model is used for reporting metrics across other languages like Basque, Italian, and Spanish. 

**Abstract (ZH)**: 自动生成反驳言论（CS）是应对仇恨言论的一个关键策略，通过提供具有建设性和信息性的回应。然而，现有方法在生成高质量、有影响力且可扩展的反驳言论方面往往效果不佳，尤其是在多种语言背景下。本文提出了一种新的方法来增强反驳言论的生成，该方法通过监督微调（SFT）和直接偏好优化（DPO）对大型语言模型（LLMs）进行对齐。我们的方法利用DPO确保生成的反驳言论适合具体情境，并且在语言上具有适应性。此外，我们还引入了知识接地策略，以提高生成反驳言论的准确性和相关性。实验结果表明，通过对齐的DPO模型在多种语言的反驳言论基准测试中显著优于SFT基线，并且能够有效扩展到其他语言，如巴斯克语、意大利语和西班牙语。这些发现表明，基于偏好的对齐技术具有推动反驳言论生成跨多种语言环境发展的潜力。模型的监督和对齐在英语中完成，而性能指标在其他语言（例如巴斯克语、意大利语和西班牙语）中使用同一模型进行报告。 

---
# Systematic Evaluation of Long-Context LLMs on Financial Concepts 

**Title (ZH)**: 对金融概念进行系统评估的长期上下文语言模型 

**Authors**: Lavanya Gupta, Saket Sharma, Yiyun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.15386)  

**Abstract**: Long-context large language models (LC LLMs) promise to increase reliability of LLMs in real-world tasks requiring processing and understanding of long input documents. However, this ability of LC LLMs to reliably utilize their growing context windows remains under investigation. In this work, we evaluate the performance of state-of-the-art GPT-4 suite of LC LLMs in solving a series of progressively challenging tasks, as a function of factors such as context length, task difficulty, and position of key information by creating a real world financial news dataset. Our findings indicate that LC LLMs exhibit brittleness at longer context lengths even for simple tasks, with performance deteriorating sharply as task complexity increases. At longer context lengths, these state-of-the-art models experience catastrophic failures in instruction following resulting in degenerate outputs. Our prompt ablations also reveal unfortunate continued sensitivity to both the placement of the task instruction in the context window as well as minor markdown formatting. Finally, we advocate for more rigorous evaluation of LC LLMs by employing holistic metrics such as F1 (rather than recall) and reporting confidence intervals, thereby ensuring robust and conclusive findings. 

**Abstract (ZH)**: 长上下文大规模语言模型（LC LLMs）有望在需要处理和理解长输入文档的实际任务中提高大语言模型的可靠性。然而，LC LLMs在利用其不断增长的上下文窗口方面能否可靠地维持这一能力尚未得到充分研究。在这项工作中，我们通过创建一个实际世界金融新闻数据集，评估了最先进的GPT-4系列长上下文语言模型在解决一系列逐步复杂的任务方面的性能，这些任务与上下文长度、任务难度和关键信息的位置等因素相关。我们的研究结果表明，即使在简单任务中，LC LLMs在较长的上下文长度下也表现出脆弱性，随着任务复杂性的增加，性能急剧下降。在较长的上下文长度下，这些最先进的模型在指令遵循方面经历了灾难性失败，导致输出退化。我们还通过对提示的消融实验发现，模型对上下文窗口中任务指令的位置以及轻微的Markdown格式化变化仍存在不幸的敏感性。最后，我们主张通过使用综合指标如F1（而不是召回率）并报告置信区间来更严格地评估LC LLMs，从而确保稳健且明确的研究结果。 

---
# Automated Root Cause Analysis System for Complex Data Products 

**Title (ZH)**: 复杂数据产品的自动化根因分析系统 

**Authors**: Mathieu Demarne, Miso Cilimdzic, Tom Falkowski, Timothy Johnson, Jim Gramling, Wei Kuang, Hoobie Hou, Amjad Aryan, Gayatri Subramaniam, Kenny Lee, Manuel Mejia, Lisa Liu, Divya Vermareddy  

**Link**: [PDF](https://arxiv.org/pdf/2412.15374)  

**Abstract**: We present ARCAS (Automated Root Cause Analysis System), a diagnostic platform based on a Domain Specific Language (DSL) built for fast diagnostic implementation and low learning curve. Arcas is composed of a constellation of automated troubleshooting guides (Auto-TSGs) that can execute in parallel to detect issues using product telemetry and apply mitigation in near-real-time. The DSL is tailored specifically to ensure that subject matter experts can deliver highly curated and relevant Auto-TSGs in a short time without having to understand how they will interact with the rest of the diagnostic platform, thus reducing time-to-mitigate and saving crucial engineering cycles when they matter most. This contrasts with platforms like Datadog and New Relic, which primarily focus on monitoring and require manual intervention for mitigation. ARCAS uses a Large Language Model (LLM) to prioritize Auto-TSGs outputs and take appropriate actions, thus suppressing the costly requirement of understanding the general behavior of the system. We explain the key concepts behind ARCAS and demonstrate how it has been successfully used for multiple products across Azure Synapse Analytics and Microsoft Fabric Synapse Data Warehouse. 

**Abstract (ZH)**: 我们介绍了ARCAS（自动根本原因分析系统），一个基于领域特定语言（DSL）构建的诊断平台，该平台旨在实现快速诊断实施并具有较低的学习曲线。ARCAS 由一组可以并行执行以检测问题并应用近实时缓解措施的自动化故障排除指南（Auto-TSGs）组成。这种DSL专门设计，以便领域专家能够在短时间内快速开发出高度精炼且相关的Auto-TSGs，而无需深入了解其如何与诊断平台的其他部分交互，从而减少缓解时间并节省关键时刻的重要工程周期。这与Datadog和New Relic等平台形成对比，后者主要专注于监控并需要手动干预来缓解问题。ARCAS 使用大型语言模型（LLM）优先处理Auto-TSGs的输出并采取适当行动，从而避免了理解系统整体行为的昂贵需求。我们介绍了ARCAS的关键概念，并展示了它在Azure Synapse Analytics和Microsoft Fabric Synapse数据仓库等多个产品中的成功应用实例。 

---
# Eliciting Causal Abilities in Large Language Models for Reasoning Tasks 

**Title (ZH)**: 在推理任务中激发大型语言模型的因果能力 

**Authors**: Yajing Wang, Zongwei Luo, Jingzhe Wang, Zhanke Zhou, Yongqiang Chen, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.15314)  

**Abstract**: Prompt optimization automatically refines prompting expressions, unlocking the full potential of LLMs in downstream tasks. However, current prompt optimization methods are costly to train and lack sufficient interpretability. This paper proposes enhancing LLMs' reasoning performance by eliciting their causal inference ability from prompting instructions to correct answers. Specifically, we introduce the Self-Causal Instruction Enhancement (SCIE) method, which enables LLMs to generate high-quality, low-quantity observational data, then estimates the causal effect based on these data, and ultimately generates instructions with the optimized causal effect. In SCIE, the instructions are treated as the treatment, and textual features are used to process natural language, establishing causal relationships through treatments between instructions and downstream tasks. Additionally, we propose applying Object-Relational (OR) principles, where the uncovered causal relationships are treated as the inheritable class across task objects, ensuring low-cost reusability. Extensive experiments demonstrate that our method effectively generates instructions that enhance reasoning performance with reduced training cost of prompts, leveraging interpretable textual features to provide actionable insights. 

**Abstract (ZH)**: 本文的内容或标题翻译成中文如下，符合学术规范：

自适应提示优化能够自动精炼提示表达，从而全面释放大型语言模型（LLM）在下游任务中的潜力。然而，当前的提示优化方法训练成本高昂且缺乏足够的解释性。本文提出了一种增强LLM推理性能的方法，即通过从提示指令到正确答案中激发它们的因果推理能力。具体而言，我们引入了自因指令增强（Self-Causal Instruction Enhancement, SCIE）方法，该方法使LLM能够生成高质量的、低数量的观察性数据，然后基于这些数据估计因果效应，并最终生成具有优化因果效应的提示指令。在SCIE中，指令被视为治疗手段，文本特征被用于处理自然语言，建立指令与下游任务之间的因果关系。此外，我们提出了对象关系（Object-Relational, OR）原则的应用，其中发现的因果关系被视为任务对象之间的可继承类，从而确保低成本的可重用性。广泛的实验证明，我们的方法能够有效地生成增强推理性能的提示指令，且降低了提示训练的成本，并利用可解释的文本特征提供了可操作的见解。 

---
# Conceptual In-Context Learning and Chain of Concepts: Solving Complex Conceptual Problems Using Large Language Models 

**Title (ZH)**: 基于上下文的概念学习与概念链：使用大型语言模型解决复杂的概念问题 

**Authors**: Nishtha N. Vaidya, Thomas Runkler, Thomas Hubauer, Veronika Haderlein-Hoegberg, Maja Mlicic Brandt  

**Link**: [PDF](https://arxiv.org/pdf/2412.15309)  

**Abstract**: Science and engineering problems fall in the category of complex conceptual problems that require specific conceptual information (CI) like math/logic -related know-how, process information, or engineering guidelines to solve them. Large Language Models (LLMs) are promising agents to solve such complex conceptual problems due to their implications in advancing engineering and science tasks like assisted problem-solving. But vanilla LLMs, trained on open-world data, lack the necessary CI. In this work, we specifically explore shallow customization methods (SCMs) of LLMs for solving complex conceptual problems. We propose two novel SCM algorithms for LLM, to augment LLMs with CI and enable LLMs to solve complex conceptual problems: Conceptual In-Context Learning (C-ICL) and Chain of Concepts (CoC). The problem tackled in this paper is generation of proprietary data models in the engineering/industry domain based on conceptual information in data modelling guidelines. We evaluate our algorithms on varied sizes of the OpenAI LLMs against four evaluation metrics related to syntactic and semantic correctness, time and cost incurred. The proposed algorithms perform better than currently popular LLM SCMs like In-context Learning (ICL) and Chain of Thoughts (CoT). It was observed that as compared to CoT, response correctness increased by 30.6% and 29.88% for the new SCMs C-ICL and CoC respectively. Qualitative analysis suggests that the proposed new SCMs activate emergent capabilities in LLMs, previously unobserved in the existing SCMs. They make problem-solving processes more transparent and reduce hallucinations and the tendency of model responses to copy examples from prompts (parroting). 

**Abstract (ZH)**: 科学和工程问题属于需要特定概念信息（CI）的复杂概念问题，这些CI包括数学/逻辑知识、过程信息或工程指导方针等才能解决。大型语言模型（LLMs）由于在辅助问题解决等工程和科学任务上的潜力而有望解决这类复杂概念问题。然而，仅基于开放世界的训练的通用LLMs缺乏必要的CI。在本研究中，我们特别探索了LLMs的浅层定制方法（SCMs）以解决复杂概念问题。我们为LLMs提出了两种新的SCM算法，以增强其所需概念信息并使LLMs能够解决复杂概念问题：概念上下文学习（C-ICL）和概念链（CoC）。

本文所解决的问题是基于数据建模指南中的概念信息生成工程/工业领域的专有数据模型。我们使用四个人工评估指标，包括语法和语义的正确性以及消耗的时间和成本，对各种规模的OpenAI LLMs算法进行了评估。提出的算法在目前流行的LLM SCMs，如上下文学习（ICL）和逐步思考（CoT）方面表现出更优异的结果。相比之下，与CoT相比，新的SCMs C-ICL和CoC的响应正确性分别提高了30.6%和29.88%。定性分析表明，提出的新SCMs激活了LLMs中未观察到的涌现能力。这些新方法使问题解决过程更加透明，并减少了模型响应中的幻觉现象和模仿提示中示例的倾向（鹦鹉学舌）。 

---
# Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models 

**Title (ZH)**: 基于推理的精调以实现大型语言模型中“最佳-n”采样 

**Authors**: Yinlam Chow, Guy Tennenholtz, Izzeddin Gur, Vincent Zhuang, Bo Dai, Sridhar Thiagarajan, Craig Boutilier, Rishabh Agarwal, Aviral Kumar, Aleksandra Faust  

**Link**: [PDF](https://arxiv.org/pdf/2412.15287)  

**Abstract**: Recent studies have indicated that effectively utilizing inference-time compute is crucial for attaining better performance from large language models (LLMs). In this work, we propose a novel inference-aware fine-tuning paradigm, in which the model is fine-tuned in a manner that directly optimizes the performance of the inference-time strategy. We study this paradigm using the simple yet effective Best-of-N (BoN) inference strategy, in which a verifier selects the best out of a set of LLM-generated responses. We devise the first imitation learning and reinforcement learning~(RL) methods for BoN-aware fine-tuning, overcoming the challenging, non-differentiable argmax operator within BoN. We empirically demonstrate that our BoN-aware models implicitly learn a meta-strategy that interleaves best responses with more diverse responses that might be better suited to a test-time input -- a process reminiscent of the exploration-exploitation trade-off in RL. Our experiments demonstrate the effectiveness of BoN-aware fine-tuning in terms of improved performance and inference-time compute. In particular, we show that our methods improve the Bo32 performance of Gemma 2B on Hendrycks MATH from 26.8% to 30.8%, and pass@32 from 60.0% to 67.0%, as well as the pass@16 on HumanEval from 61.6% to 67.1%. 

**Abstract (ZH)**: 近年来的研究表明，在推理时有效利用计算资源对于提高大型语言模型（LLMs）的性能至关重要。在此项工作中，我们提出了一种新颖的推理感知微调范式，在该范式中，模型以直接优化推理时策略性能的方式进行微调。我们使用一种简单而有效的“最佳中的N个”（BoN）推理策略进行研究，在该策略中，验证器从一组LLM生成的响应中选择最佳者。我们首次为BoN意识微调设计了模仿学习和强化学习（RL）方法，克服了BoN内部难以处理的非可微分的argmax操作。我们的实证研究表明，我们的BoN意识模型隐式学习了一种元策略，该策略将最佳响应与可能更适合测试输入的更具有多样性的响应交错使用——这一过程类似于RL中的探索-利用权衡。我们的实验表明，BoN意识微调在提高性能和推理计算资源利用方面是有效的。特别是，我们的方法将Gemma 2B在Hendrycks MATH上的Bo32性能从26.8%提高到30.8%，pass@32从60.0%提高到67.0%，并且在HumanEval上的pass@16从61.6%提高到67.1%。 

---
# Context-DPO: Aligning Language Models for Context-Faithfulness 

**Title (ZH)**: CONTEXT-DPO：提高语言模型背景一致性的对齐方法 

**Authors**: Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi Yang, Zihan Zhang, Haizhen Huang, Lingrui Mei, Junfeng Fang, Zehao Li, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15280)  

**Abstract**: Reliable responses from large language models (LLMs) require adherence to user instructions and retrieved information. While alignment techniques help LLMs align with human intentions and values, improving context-faithfulness through alignment remains underexplored. To address this, we propose $\textbf{Context-DPO}$, the first alignment method specifically designed to enhance LLMs' context-faithfulness. We introduce $\textbf{ConFiQA}$, a benchmark that simulates Retrieval-Augmented Generation (RAG) scenarios with knowledge conflicts to evaluate context-faithfulness. By leveraging faithful and stubborn responses to questions with provided context from ConFiQA, our Context-DPO aligns LLMs through direct preference optimization. Extensive experiments demonstrate that our Context-DPO significantly improves context-faithfulness, achieving 35% to 280% improvements on popular open-source models. Further analysis demonstrates that Context-DPO preserves LLMs' generative capabilities while providing interpretable insights into context utilization. Our code and data are released at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）的可靠响应需要遵循用户指令并结合检索到的信息。虽然对齐技术有助于使LLMs与人类的意图和价值观保持一致，但通过对齐提高上下文保真度仍研究不足。为了应对这一挑战，我们提出了一种名为$\textbf{Context-DPO}$的新方法，这是首个专门设计用于增强LLMs上下文保真度的对齐方法。我们引入了$\textbf{ConFiQA}$基准，该基准模拟了包含知识冲突的检索增强生成（RAG）场景，用于评估上下文保真度。通过利用ConFiQA提供的上下文信息生成的忠实且固执的响应，我们的Context-DPO通过直接的偏好优化对齐LLMs。大量实验表明，我们的Context-DPO显著提高了上下文保真度，在流行的开源模型上实现了35%到280%的改进。进一步的分析表明，Context-DPO既保留了LLMs的生成能力，又提供了关于上下文利用的可解释见解。我们的代码和数据可以在以下链接获取：this https URL 

---
# Fooling LLM graders into giving better grades through neural activity guided adversarial prompting 

**Title (ZH)**: 通过神经活动引导的对抗提示欺骗LLM评分器给出更高的评分 

**Authors**: Atsushi Yamamura, Surya Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2412.15275)  

**Abstract**: The deployment of artificial intelligence (AI) in critical decision-making and evaluation processes raises concerns about inherent biases that malicious actors could exploit to distort decision outcomes. We propose a systematic method to reveal such biases in AI evaluation systems and apply it to automated essay grading as an example. Our approach first identifies hidden neural activity patterns that predict distorted decision outcomes and then optimizes an adversarial input suffix to amplify such patterns. We demonstrate that this combination can effectively fool large language model (LLM) graders into assigning much higher grades than humans would. We further show that this white-box attack transfers to black-box attacks on other models, including commercial closed-source models like Gemini. They further reveal the existence of a "magic word" that plays a pivotal role in the efficacy of the attack. We trace the origin of this magic word bias to the structure of commonly-used chat templates for supervised fine-tuning of LLMs and show that a minor change in the template can drastically reduce the bias. This work not only uncovers vulnerabilities in current LLMs but also proposes a systematic method to identify and remove hidden biases, contributing to the goal of ensuring AI safety and security. 

**Abstract (ZH)**: 将人工智能（AI）应用于关键决策和评估过程中，引发了关于恶意行为者可能利用内在偏见来扭曲决策结果的担忧。我们提出了一种系统性方法，用于揭示AI评估系统中的这些偏见，并通过自动作文评分作为示例进行应用。该方法首先识别出能预测扭曲决策结果的隐藏神经活动模式，然后优化一个对抗性输入后缀以放大这些模式。我们证明了这种结合可以有效地欺骗大型语言模型（LLM）评分者，使其给出远高于人工评分的分数。此外，我们还展示了这种白盒攻击可以转移到其他模型的黑盒攻击上，包括像Gemini这样的商业封闭源代码模型。这一过程进一步揭示了一个“魔术单词”的存在，该单词在攻击的有效性中起着关键作用。我们追踪了这一魔术单词偏见的起源，发现其与监督微调大型语言模型时常用对话模板的结构密切相关，并表明模板中微小的更改可以大幅减少这种偏见。本研究不仅揭示了当前大型语言模型中存在的漏洞，还提出了一种系统方法来识别并移除隐藏偏见，从而有助于确保人工智能的安全性和安全性。 

---
# SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation 

**Title (ZH)**: SimGRAG：利用相似子图进行知识图驱动的检索增强生成 

**Authors**: Yuzheng Cai, Zhenyue Guo, Yiwen Pei, Wanrui Bian, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.15272)  

**Abstract**: Recent advancements in large language models (LLMs) have shown impressive versatility across various tasks. To eliminate its hallucinations, retrieval-augmented generation (RAG) has emerged as a powerful approach, leveraging external knowledge sources like knowledge graphs (KGs). In this paper, we study the task of KG-driven RAG and propose a novel Similar Graph Enhanced Retrieval-Augmented Generation (SimGRAG) method. It effectively addresses the challenge of aligning query texts and KG structures through a two-stage process: (1) query-to-pattern, which uses an LLM to transform queries into a desired graph pattern, and (2) pattern-to-subgraph, which quantifies the alignment between the pattern and candidate subgraphs using a graph semantic distance (GSD) metric. We also develop an optimized retrieval algorithm that efficiently identifies the top-$k$ subgraphs within 1-second latency on a 10-million-scale KG. Extensive experiments show that SimGRAG outperforms state-of-the-art KG-driven RAG methods in both question answering and fact verification, offering superior plug-and-play usability and scalability. 

**Abstract (ZH)**: 近期大规模语言模型（LLMs）在各种任务中的表现展现了惊人的 versatility。为了消除其幻觉现象，检索增强生成（RAG）作为一种强大的方法应运而生，利用如知识图谱（KGs）等外部知识源。本文探讨了基于KG的RAG任务，并提出了一种新颖的相似图增强检索增强生成（SimGRAG）方法。该方法通过两阶段过程有效解决了查询文本与KG结构对齐的挑战：（1）查询到模式阶段，使用LLM将查询转换为所需的图模式；（2）模式到子图阶段，通过图语义距离（GSD）度量量化模式与候选子图之间的对齐程度。我们还开发了一种优化的检索算法，能够在1秒的延迟下高效地在1000万规模的KG中识别出前k个子图。广泛的经验表明，SimGRAG在问答和事实验证方面均优于现有的基于KG的RAG方法，提供了更优的即插即用可用性和可扩展性。 

---
# Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph 

**Title (ZH)**: 基于元毒刺激知识图谱增强基于大规模语言模型的仇恨和毒性检测 

**Authors**: Yibo Zhao, Jiapeng Zhu, Can Xu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15268)  

**Abstract**: The rapid growth of social media platforms has raised significant concerns regarding online content toxicity. When Large Language Models (LLMs) are used for toxicity detection, two key challenges emerge: 1) the absence of domain-specific toxic knowledge leads to false negatives; 2) the excessive sensitivity of LLMs to toxic speech results in false positives, limiting freedom of speech. To address these issues, we propose a novel method called MetaTox, leveraging graph search on a meta-toxic knowledge graph to enhance hatred and toxicity detection. First, we construct a comprehensive meta-toxic knowledge graph by utilizing LLMs to extract toxic information through a three-step pipeline, with toxic benchmark datasets serving as corpora. Second, we query the graph via retrieval and ranking processes to supplement accurate, relevant toxic knowledge. Extensive experiments and in-depth case studies across multiple datasets demonstrate that our MetaTox significantly decreases the false positive rate while boosting overall toxicity detection performance. Our code will be available soon. 

**Abstract (ZH)**: 社交媒体平台的迅速增长引发了对在线内容毒性的重大关注。当大型语言模型（LLMs）用于检测毒性时，两个关键挑战随之浮现：1）缺乏特定领域的毒信息知识导致假阴性；2）大型语言模型对有毒言论的过度敏感性导致假阳性结果，限制了言论自由。为了解决这些问题，我们提出了一种名为MetaTox的新方法，该方法通过图搜索在元毒信息图上增强仇恨和毒性检测。首先，我们通过利用LLMs以三步流水线来提取毒信息的方式构建了一个全面的元毒信息图，毒信息基准数据集作为语料库。其次，我们通过检索和排序过程查询图，以补充准确且相关的毒信息知识。在多个数据集上进行的广泛实验和深入案例研究表明，我们的MetaTox显著降低了假阳性率，并提升了整体毒性检测性能。我们的代码很快将对外开放。 

---
# LLMs for Literature Review: Are we there yet? 

**Title (ZH)**: 用于文献综述的大型语言模型：我们到了吗？ 

**Authors**: Shubham Agarwal, Gaurav Sahu, Abhay Puri, Issam H. Laradji, Krishnamurthy DJ Dvijotham, Jason Stanley, Laurent Charlin, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2412.15249)  

**Abstract**: Literature reviews are an essential component of scientific research, but they remain time-intensive and challenging to write, especially due to the recent influx of research papers. This paper explores the zero-shot abilities of recent Large Language Models (LLMs) in assisting with the writing of literature reviews based on an abstract. We decompose the task into two components: 1. Retrieving related works given a query abstract, and 2. Writing a literature review based on the retrieved results. We analyze how effective LLMs are for both components. For retrieval, we introduce a novel two-step search strategy that first uses an LLM to extract meaningful keywords from the abstract of a paper and then retrieves potentially relevant papers by querying an external knowledge base. Additionally, we study a prompting-based re-ranking mechanism with attribution and show that re-ranking doubles the normalized recall compared to naive search methods, while providing insights into the LLM's decision-making process. In the generation phase, we propose a two-step approach that first outlines a plan for the review and then executes steps in the plan to generate the actual review. To evaluate different LLM-based literature review methods, we create test sets from arXiv papers using a protocol designed for rolling use with newly released LLMs to avoid test set contamination in zero-shot evaluations. We release this evaluation protocol to promote additional research and development in this regard. Our empirical results suggest that LLMs show promising potential for writing literature reviews when the task is decomposed into smaller components of retrieval and planning. Further, we demonstrate that our planning-based approach achieves higher-quality reviews by minimizing hallucinated references in the generated review by 18-26% compared to existing simpler LLM-based generation methods. 

**Abstract (ZH)**: 文献综述是科学研究不可或缺的组成部分，但撰写文献综述仍然是一项耗时且具有挑战性的工作，尤其是由于近年来研究论文的急剧增加。本文探讨了近期大规模语言模型（LLMs）在基于摘要辅助撰写文献综述方面的零样本能力。我们将任务分解为两个部分：1. 在给定查询摘要的情况下检索相关工作，2. 根据检索结果撰写文献综述。我们分析了LLMs在这两个方面的效果。对于检索部分，我们提出了一种新颖的两步搜索策略：首先使用LLM从论文摘要中提取有意义的关键词，然后再通过查询外部知识库检索潜在相关论文。此外，我们研究了一种基于提示的重新排名机制，并展示了重新排名相比朴素搜索方法可以将归一化召回率提高一倍，同时提供对LLM决策过程的洞察。在生成阶段，我们提出了一种两步方法：首先概述综述计划，然后根据计划执行步骤以生成实际的综述。为了评估不同的LLM基文献综述方法，我们从arXiv论文中创建了测试集，并设计了一个在使用新发布的LLM时可以滚动更新的协议，以避免零样本评估中的测试集污染。我们发布了这一评估协议，以促进对该领域的进一步研究与开发。我们的实证结果显示，当任务分解为检索和规划的小部件时，LLM在撰写文献综述方面表现出令人鼓舞的潜力。此外，我们展示了基于规划的方法通过减少生成综述中虚构参考文献的比例（18%至26%）而实现了更高的综述质量，相比之下，现有的简单LLM生成方法效果较差。 

---
# Script-Based Dialog Policy Planning for LLM-Powered Conversational Agents: A Basic Architecture for an "AI Therapist" 

**Title (ZH)**: 基于脚本的对话策略规划：为大规模语言模型驱动的对话代理设计“AI治疗师”基本架构 

**Authors**: Robert Wasenmüller, Kevin Hilbert, Christoph Benzmüller  

**Link**: [PDF](https://arxiv.org/pdf/2412.15242)  

**Abstract**: Large Language Model (LLM)-Powered Conversational Agents have the potential to provide users with scaled behavioral healthcare support, and potentially even deliver full-scale "AI therapy'" in the future. While such agents can already conduct fluent and proactive emotional support conversations, they inherently lack the ability to (a) consistently and reliably act by predefined rules to align their conversation with an overarching therapeutic concept and (b) make their decision paths inspectable for risk management and clinical evaluation -- both essential requirements for an "AI Therapist".
In this work, we introduce a novel paradigm for dialog policy planning in conversational agents enabling them to (a) act according to an expert-written "script" that outlines the therapeutic approach and (b) explicitly transition through a finite set of states over the course of the conversation. The script acts as a deterministic component, constraining the LLM's behavior in desirable ways and establishing a basic architecture for an AI Therapist.
We implement two variants of Script-Based Dialog Policy Planning using different prompting techniques and synthesize a total of 100 conversations with LLM-simulated patients. The results demonstrate the feasibility of this new technology and provide insights into the efficiency and effectiveness of different implementation variants. 

**Abstract (ZH)**: 大语言模型（LLM）驱动的对话代理有可能为用户提供扩展的行为健康支持，并且未来甚至可以提供全面的“AI疗法”。虽然这些代理已经能够进行流畅、积极的情感支持对话，但它们仍然存在无法（a）始终一致地按照预定义规则行动，以使对话与整体治疗概念保持一致，以及（b）使决策路径可检查，以便风险管理与临床评估的问题——这两者是“AI疗法”所必需的要求。

在本研究中，我们提出了一个用于对话策略规划的新范式，使对话代理能够（a）根据专家编写的“剧本”进行行动，该剧本概述了治疗方法，以及（b）在对话过程中明确地通过一组有限状态。该剧本作为确定性的组成部分，限制了LLM的行为，并奠定了“AI疗法”基本架构的基础。

我们使用不同的提示技术实现了两种基于剧本的对话策略规划变体，并使用LLM模拟的患者合成了共计100次对话。实验结果证明了该新方法的可行性，并提供了不同实施变体的效率与有效性见解。 

---
# ChainStream: An LLM-based Framework for Unified Synthetic Sensing 

**Title (ZH)**: ChainStream：一种基于大语言模型的统一合成感知框架 

**Authors**: Jiacheng Liu, Yuanchun Li, Liangyan Li, Yi Sun, Hao Wen, Xiangyu Li, Yao Guo, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15240)  

**Abstract**: Many applications demand context sensing to offer personalized and timely services. Yet, developing sensing programs can be challenging for developers and using them is privacy-concerning for end-users. In this paper, we propose to use natural language as the unified interface to process personal data and sense user context, which can effectively ease app development and make the data pipeline more transparent. Our work is inspired by large language models (LLMs) and other generative models, while directly applying them does not solve the problem - letting the model directly process the data cannot handle complex sensing requests and letting the model write the data processing program suffers error-prone code generation. We address the problem with 1) a unified data processing framework that makes context-sensing programs simpler and 2) a feedback-guided query optimizer that makes data query more informative. To evaluate the performance of natural language-based context sensing, we create a benchmark that contains 133 context sensing tasks. Extensive evaluation has shown that our approach is able to automatically solve the context-sensing tasks efficiently and precisely. The code is opensourced at this https URL. 

**Abstract (ZH)**: 许多应用程序需要情境感知以提供个性化和及时的服务。然而，开发感知程序对于开发者来说具有挑战性，而用户在使用这些程序时则关心个人隐私。本文中，我们提出使用自然语言作为统一界面来处理个人数据并感知用户情境，这将有效简化应用程序开发并使数据处理流程更具透明性。我们的工作灵感来源于大型语言模型（LLMs）和其他生成模型，但直接应用于这些模型并不能解决问题——让模型直接处理数据无法处理复杂的感知请求，而让模型编写数据处理程序则会导致错误的代码生成。我们通过以下方式解决了这一问题：1) 提出了一种统一的数据处理框架，使情境感知程序更简单；2) 提出了一个反馈指导的查询优化器，使数据查询更具信息性。为了评估基于自然语言的情境感知性能，我们创建了一个基准测试，其中包含133个情境感知任务。广泛的评估表明，我们的方法能够高效且精确地自动完成情境感知任务。源代码已开源，可访问 <请提供具体的URL或直接提供链接>。 

---
# Modeling Story Expectations to Understand Engagement: A Generative Framework Using LLMs 

**Title (ZH)**: 利用大语言模型构建生成框架，以建模故事情节预期并理解参与度 

**Authors**: Hortense Fong, George Gui  

**Link**: [PDF](https://arxiv.org/pdf/2412.15239)  

**Abstract**: Understanding when and why consumers engage with stories is crucial for content creators and platforms. While existing theories suggest that audience beliefs of what is going to happen should play an important role in engagement decisions, empirical work has mostly focused on developing techniques to directly extract features from actual content, rather than capturing forward-looking beliefs, due to the lack of a principled way to model such beliefs in unstructured narrative data. To complement existing feature extraction techniques, this paper introduces a novel framework that leverages large language models to model audience forward-looking beliefs about how stories might unfold. Our method generates multiple potential continuations for each story and extracts features related to expectations, uncertainty, and surprise using established content analysis techniques. Applying our method to over 30,000 book chapters from Wattpad, we demonstrate that our framework complements existing feature engineering techniques by amplifying their marginal explanatory power on average by 31%. The results reveal that different types of engagement-continuing to read, commenting, and voting-are driven by distinct combinations of current and anticipated content features. Our framework provides a novel way to study and explore how audience forward-looking beliefs shape their engagement with narrative media, with implications for marketing strategy in content-focused industries. 

**Abstract (ZH)**: 了解消费者何时以及为何参与故事对于内容创作者和平台至关重要。虽然现有理论表明观众对未来可能发生的事件的信念应该在参与决策中扮演重要角色，但实证研究大多集中于开发从实际内容中直接提取特征的技术，而不是捕捉前瞻性信念，这是因为缺乏一种在非结构化叙事数据中建模此类信念的系统化方法。为了补充现有的特征提取技术，本文引入了一种新的框架，利用大规模语言模型来建模观众对故事可能如何发展的前瞻性信念。我们的方法为每个故事生成多个潜在续篇，并通过现有的内容分析技术提取与期望、不确定性以及惊喜相关的特征。将我们的方法应用于 Wattpad 上超过 30,000 个章节，结果显示，我们的框架通过平均放大现有特征工程技术边际解释力 31%，从而进一步补充了它们。研究结果表明，不同类型的参与行为（继续阅读、评论和投票）由当前和预期内容特征的不同组合驱动。我们的框架为研究和探索前瞻性信念如何影响观众对叙事媒体的参与提供了一种新的途径，对于内容导向行业的营销策略具有重要意义。 

---
# Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks 

**Title (ZH)**: 提斗：推理任务中大型语言模型集成的提示多样性 

**Authors**: Gregory Kang Ruey Lau, Wenyang Hu, Diwen Liu, Jizhuo Chen, See-Kiong Ng, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2412.15238)  

**Abstract**: Large Language Models still encounter substantial challenges in reasoning tasks, especially for smaller models, which many users may be restricted to due to resource constraints (e.g. GPU memory restrictions). Inference-time methods to boost LLM performance, such as prompting methods to invoke certain reasoning pathways in responses, have been shown effective in past works, though they largely rely on sequential queries. The ensemble method, which consists of multiple constituent models running in parallel, is a promising approach to achieving better inference-time performance, especially given recent developments that enabled significant speed-ups in LLM batch inference. In this work, we propose a novel, training-free LLM ensemble framework where a single LLM model is fed an optimized, diverse set of prompts in parallel, effectively producing an ensemble at inference time to achieve performance improvement in reasoning tasks. We empirically demonstrate that our method leads to significant gains on math reasoning tasks, e.g., on MATH, where our ensemble consisting of a few small models (e.g., three Qwen2-MATH-1.5B-it models) can outperform a larger model (e.g., Qwen2-MATH-7B-it). 

**Abstract (ZH)**: 大型语言模型在推理任务中仍然面临诸多挑战，尤其是对于较小的模型，很多用户可能由于资源限制（如GPU内存限制）而无法使用。在推理时间提升大型语言模型（LLM）性能的方法，如通过提示方法触发特定的推理路径，在以往的研究中已被证明有效，尽管这些方法主要依赖于顺序查询。由多个并行运行的组成部分模型组成的集成方法是一种有望在提升推理时间性能方面取得更好结果的方法，尤其是在最近的技术发展使得LLM批量推理速度显著提高的情况下。在本研究中，我们提出了一种全新的无需训练的LLM集成框架，其中单个LLM模型同时接受优化且多样的提示输入，从而在推理时间生成集成体，以在推理任务中实现性能提升。我们通过实验证明，我们的方法在数学推理任务上表现出了显著的改进，例如，在MATH数据集上，由几个小型模型（例如，三个Qwen2-MATH-1.5B-it模型）组成的集成体可以优于一个较大的模型（例如，Qwen2-MATH-7B-it）。 

---
# OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models 

**Title (ZH)**: OG-RAG：面向本体检索增强生成方法用于大型语言模型 

**Authors**: Kartik Sharma, Peeyush Kumar, Yunqing Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15235)  

**Abstract**: This paper presents OG-RAG, an Ontology-Grounded Retrieval Augmented Generation method designed to enhance LLM-generated responses by anchoring retrieval processes in domain-specific ontologies. While LLMs are widely used for tasks like question answering and search, they struggle to adapt to specialized knowledge, such as industrial workflows or knowledge work, without expensive fine-tuning or sub-optimal retrieval methods. Existing retrieval-augmented models, such as RAG, offer improvements but fail to account for structured domain knowledge, leading to suboptimal context generation. Ontologies, which conceptually organize domain knowledge by defining entities and their interrelationships, offer a structured representation to address this gap. OG-RAG constructs a hypergraph representation of domain documents, where each hyperedge encapsulates clusters of factual knowledge grounded using domain-specific ontology. An optimization algorithm then retrieves the minimal set of hyperedges that constructs a precise, conceptually grounded context for the LLM. This method enables efficient retrieval while preserving the complex relationships between entities. OG-RAG applies to domains where fact-based reasoning is essential, particularly in tasks that require workflows or decision-making steps to follow predefined rules and procedures. These include industrial workflows in healthcare, legal, and agricultural sectors, as well as knowledge-driven tasks such as news journalism, investigative research, consulting and more. Our evaluations demonstrate that OG-RAG increases the recall of accurate facts by 55% and improves response correctness by 40% across four different LLMs. Additionally, OG-RAG enables 30% faster attribution of responses to context and boosts fact-based reasoning accuracy by 27% compared to baseline methods. 

**Abstract (ZH)**: 本文提出了一种基于本体的检索增强生成方法（OG-RAG），旨在通过将检索过程锚定在领域特定本体上，从而增强LLM生成的响应。虽然LLM广泛应用于问答和搜索等任务，但在处理如工业流程或知识工作等专业领域知识时，它们往往无法通过昂贵的微调或不理想的检索方法进行有效适应。现有的检索增强模型，如RAG，虽然提供了改进，但未能考虑到结构化的领域知识，导致上下文生成效果欠佳。本体从概念上组织领域知识，通过定义实体及其相互关系来提供一种结构化的表示方法，以此填补这一空白。OG-RAG 构建了一个领域文档的超图表示，其中每个超边封装了通过领域特定本体进行事实性知识归因的实体集群。随后优化算法检索出能够构成精确、概念性上下文的最小超边集，以供LLM使用。这种方法能够在保留实体间复杂关系的同时实现高效的检索。OG-RAG 适用于基于事实的推理至关重要的领域，特别是在需要遵循预定义规则和程序的工作流或决策任务中尤为重要。这些领域包括医疗、法律和农业行业的工业工作流程，以及诸如新闻采编、调查研究、咨询等知识驱动的任务。我们的评估结果显示，OG-RAG 将准确事实的召回率提高了55%，并将响应的正确性提高了40%。此外，OG-RAG 使得对上下文的响应归属速度提高了30%，并且相比基准方法提高了27%的事实性推理准确性。 

---
# PromptOptMe: Error-Aware Prompt Compression for LLM-based MT Evaluation Metrics 

**Title (ZH)**: PromptOptMe：面向错误的prompt压缩方法以优化基于LLM的 machine translation 评估指标 

**Authors**: Daniil Larionov, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2412.16120)  

**Abstract**: Evaluating the quality of machine-generated natural language content is a challenging task in Natural Language Processing (NLP). Recently, large language models (LLMs) like GPT-4 have been employed for this purpose, but they are computationally expensive due to the extensive token usage required by complex evaluation prompts. In this paper, we propose a prompt optimization approach that uses a smaller, fine-tuned language model to compress input data for evaluation prompt, thus reducing token usage and computational cost when using larger LLMs for downstream evaluation. Our method involves a two-stage fine-tuning process: supervised fine-tuning followed by preference optimization to refine the model's outputs based on human preferences. We focus on Machine Translation (MT) evaluation and utilize the GEMBA-MQM metric as a starting point. Our results show a $2.37\times$ reduction in token usage without any loss in evaluation quality. This work makes state-of-the-art LLM-based metrics like GEMBA-MQM more cost-effective and efficient, enhancing their accessibility for broader use. 

**Abstract (ZH)**: 评估机器生成的自然语言内容质量是自然语言处理（NLP）领域的一项具有挑战性的任务。近年来，大型语言模型（LLMs）如GPT-4已经被用于这一任务，但由于复杂的评估提示所需的大量标记使用，使得计算成本高昂。在本文中，我们提出了一种提示优化方法，该方法使用一个较小的微调语言模型来压缩用于评估的输入数据，从而在使用大型LLMs进行下游评估时减少标记使用和计算成本。我们的方法包括两个阶段的微调过程：监督微调，随后是偏好优化，以根据人类偏好细化模型输出。我们专注于机器翻译（MT）评估，并利用GEMBA-MQM度量作为起点。实验结果表明，在没有损失评估质量的情况下，标记使用量减少了2.37倍。这项工作使最先进的基于LLM的度量标准，如GEMBA-MQM更具有成本效益和高效性，从而增强其在更广泛范围中的可访问性。 

---
# Logical Consistency of Large Language Models in Fact-checking 

**Title (ZH)**: 大型语言模型在事实核查中的逻辑一致性研究 

**Authors**: Bishwamittra Ghosh, Sarah Hasan, Naheed Anjum Arafat, Arijit Khan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16100)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated significant success in performing varied natural language tasks such as language translation, question-answering, summarizing, fact-checking, etc. Despite LLMs' impressive ability to generate human-like texts, LLMs are infamous for their inconsistent responses -- a meaning-preserving change in the input query results in an inconsistent response and attributes to vulnerabilities of LLMs such as hallucination, jailbreaking, etc. Consequently, existing research focuses on simple paraphrasing-based consistency assessment of LLMs, and ignores complex queries that necessitates an even better understanding of logical reasoning by an LLM. Our work therefore addresses the logical inconsistency of LLMs under complex logical queries with primitive logical operators, e.g., negation, conjunction, and disjunction. As a test bed, we consider retrieval-augmented LLMs on a fact-checking task involving propositional logic queries from real-world knowledge graphs (KGs). Our contributions are three-fold. Benchmark: We introduce three logical fact-checking datasets over KGs for community development towards logically consistent LLMs. Assessment: We propose consistency measures of LLMs on propositional logic queries as input and demonstrate that existing LLMs lack logical consistency, specially on complex queries. Improvement: We employ supervised fine-tuning to improve the logical consistency of LLMs on the complex fact-checking task with KG contexts. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLM）在诸如语言翻译、问答、摘要、事实核查等多种自然语言任务中展现出显著的成功。尽管LLM在生成类人文本方面表现出色，但它们以不可预测的方式响应，即输入查询的含义保持不变，却导致响应不一致。这种不一致归因于LLM的一些漏洞，如幻觉和破解等。因此，现有研究主要集中在基于简单 paraphrasing 的一致性评估上，而忽视了需要更深入理解逻辑推理的复杂查询。我们的研究工作旨在解决在复杂逻辑查询中使用基本逻辑运算符（如否定、合取和析取）时LLM的逻辑不一致性问题。作为实验平台，我们考虑了在涉及来自现实知识图谱（KGs）的命题逻辑查询的事实核查任务中增强检索的LLM。我们的贡献包括三个方面：

1. **基准数据集**：我们引入了三个基于KG的逻辑事实核查数据集，以促进对逻辑一致的LLM的社区开发。
2. **评估**：我们提出了针对命题逻辑查询的LLM一致性度量，展示现有的LLM在逻辑一致性方面存在不足，尤其是对于复杂的查询。
3. **改进**：我们采用监督微调方法，在包含KG上下文的复杂事实核查任务中提高LLM的逻辑一致性。 

---
# Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning 

**Title (ZH)**: 使用过程奖励引导的树搜索技术集成大型语言模型以实现更复杂的推理 

**Authors**: Sungjin Park, Xiao Liu, Yeyun Gong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2412.15797)  

**Abstract**: Despite recent advances in large language models, open-source models often struggle to consistently perform well on complex reasoning tasks. Existing ensemble methods, whether applied at the token or output levels, fail to address these challenges. In response, we present Language model Ensemble with Monte Carlo Tree Search (LE-MCTS), a novel framework for process-level ensembling of language models. LE-MCTS formulates step-by-step reasoning with an ensemble of language models as a Markov decision process. In this framework, states represent intermediate reasoning paths, while actions consist of generating the next reasoning step using one of the language models selected from a predefined pool. Guided by a process-based reward model, LE-MCTS performs a tree search over the reasoning steps generated by different language models, identifying the most accurate reasoning chain. Experimental results on five mathematical reasoning benchmarks demonstrate that our approach outperforms both single language model decoding algorithms and language model ensemble methods. Notably, LE-MCTS improves performance by 3.6% and 4.3% on the MATH and MQA datasets, respectively, highlighting its effectiveness in solving complex reasoning problems. 

**Abstract (ZH)**: 尽管近年来大型语言模型取得了重大进展，但开源模型在复杂推理任务上往往难以持续表现出色。现有的集成方法，无论是在标记级还是输出级应用，都无法解决这些问题。为应对这一挑战，我们提出了一种新的语言模型集成框架——基于蒙特卡洛树搜索的语言模型集成（LE-MCTS）。LE-MCTS 将语言模型的逐步推理过程视为马尔可夫决策过程。在这个框架中，状态代表中间推理路径，而动作则是从预定义的池中选择一个语言模型生成下一步推理。通过基于过程导向的奖励模型的引导，LE-MCTS 对不同语言模型生成的推理步骤进行树搜索，识别出最准确的推理链。在五个数学推理基准测试上的实验结果显示，我们的方法优于单个语言模型解码算法以及所有的语言模型集成方法。值得注意的是，LE-MCTS 在 MATH 和 MQA 数据集上的性能分别提高了 3.6% 和 4.3%，突显了其在解决复杂推理问题方面的有效性。 

---
# Linguistic Features Extracted by GPT-4 Improve Alzheimer's Disease Detection based on Spontaneous Speech 

**Title (ZH)**: GPT-4 提取的 Linguistic 特征基于自发言语提高 Alzheimer's 病检测 

**Authors**: Jonathan Heitz, Gerold Schneider, Nicolas Langer  

**Link**: [PDF](https://arxiv.org/pdf/2412.15772)  

**Abstract**: Alzheimer's Disease (AD) is a significant and growing public health concern. Investigating alterations in speech and language patterns offers a promising path towards cost-effective and non-invasive early detection of AD on a large scale. Large language models (LLMs), such as GPT, have enabled powerful new possibilities for semantic text analysis. In this study, we leverage GPT-4 to extract five semantic features from transcripts of spontaneous patient speech. The features capture known symptoms of AD, but they are difficult to quantify effectively using traditional methods of computational linguistics. We demonstrate the clinical significance of these features and further validate one of them ("Word-Finding Difficulties") against a proxy measure and human raters. When combined with established linguistic features and a Random Forest classifier, the GPT-derived features significantly improve the detection of AD. Our approach proves effective for both manually transcribed and automatically generated transcripts, representing a novel and impactful use of recent advancements in LLMs for AD speech analysis. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是当前一个重要的公共卫生问题，探索言语和语言模式的变化为成本低且无创的大规模早期检测AD提供了一条有希望的道路。大型语言模型（LLMs），如GPT，为语义文本分析带来了强大的新可能性。在本研究中，我们利用GPT-4从自发患者言语的转录中提取五个语义特征。这些特征捕捉到了AD已知的症状，但传统计算语言学方法难以有效量化。我们展示了这些特征的临床意义，并进一步通过替代指标和人工评分验证了其中一个特征（“找词困难”）。当这些特征与现有的语言特征以及随机森林分类器结合使用时，能够显著提高AD的检测效果。我们的方法证明了在手动转录和自动生成的转录中均有效，代表了近期LLMs在AD言语分析中的一种新颖且具有影响力的使用方式。 

---
# Variability Need Not Imply Error: The Case of Adequate but Semantically Distinct Responses 

**Title (ZH)**: 变异性未必意味着错误：充分但语义上不同的反应案例 

**Authors**: Evgenia Ilia, Wilker Aziz  

**Link**: [PDF](https://arxiv.org/pdf/2412.15683)  

**Abstract**: With the broader use of language models (LMs) comes the need to estimate their ability to respond reliably to prompts (e.g., are generated responses likely to be correct?). Uncertainty quantification tools (notions of confidence and entropy, i.a.) can be used to that end (e.g., to reject a response when the model is `uncertain'). For example, Kuhn et al. (semantic entropy; 2022b) regard semantic variation amongst sampled responses as evidence that the model `struggles' with the prompt and that the LM is likely to err. We argue that semantic variability need not imply error--this being especially intuitive in open-ended settings, where prompts elicit multiple adequate but semantically distinct responses. Hence, we propose to annotate sampled responses for their adequacy to the prompt (e.g., using a classifier) and estimate the Probability the model assigns to Adequate Responses (PROBAR), which we then regard as an indicator of the model's reliability at the instance level. We evaluate PROBAR as a measure of confidence in selective prediction with OPT models (in two QA datasets and in next-word prediction, for English) and find PROBAR to outperform semantic entropy across prompts with varying degrees of ambiguity/open-endedness. 

**Abstract (ZH)**: 随着语言模型（LMs）的应用范围不断扩大，评估其对提示作出可靠响应的能力变得愈加重要（例如，生成的响应是否很可能正确？）。为了实现这一目标，可以使用不确定性量化工具（如置信度和熵的概念等），以便在模型“不确定”时拒绝响应。例如，Kuhn等人（语义熵；2022b）认为，样本响应之间语义上的多样性是模型“难以应对”提示和模型可能会出错的证据。我们提出，语义多样性未必意味着错误，特别是在开放性设置中尤为直观，因为提示可能引发多个语义上不相同但都足够恰当的响应。因此，我们建议为样本响应标注它们对提示的适当性（例如，使用分类器），并估计模型为适当响应分配的概率（PROBAR），将其视为模型在实例级别的可靠性指标。我们评估在OPT模型中PROBAR作为选择性预测置信度度量的表现（在两个问答数据集和英语的下一个词预测中），发现与语义熵相比，PROBAR在不同模糊度/开放性程度的提示下表现更佳。 

---
# Can Input Attributions Interpret the Inductive Reasoning Process Elicited in In-Context Learning? 

**Title (ZH)**: 输入归因能否解释基于上下文学习中引发的归纳推理过程？ 

**Authors**: Mengyu Ye, Tatsuki Kuribayashi, Goro Kobayashi, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2412.15628)  

**Abstract**: Elucidating the rationale behind neural models' outputs has been challenging in the machine learning field, which is indeed applicable in this age of large language models (LLMs) and in-context learning (ICL). When it comes to estimating input attributions (IA), ICL poses a new issue of interpreting which example in the prompt, consisting of a set of examples, contributed to identifying the task/rule to be solved. To this end, in this paper, we introduce synthetic diagnostic tasks inspired by the poverty of the stimulus design in inductive reasoning; here, most in-context examples are ambiguous w.r.t. their underlying rule, and one critical example disambiguates the task demonstrated. The question is whether conventional IA methods can identify such an example in interpreting the inductive reasoning process in ICL. Our experiments provide several practical findings; for example, a certain simple IA method works the best, and the larger the model, the generally harder it is to interpret the ICL with gradient-based IA methods. 

**Abstract (ZH)**: 阐明神经模型输出背后的理据在机器学习领域一直具有挑战性，这一问题在当前的大语言模型（LLMs）和在上下文学习（ICL）时代尤其适用。当涉及到输入归因（IA）的估计时，ICL 引入了一个新的问题：即如何解释在提示中包含的一组示例中，哪个示例对识别要解决的任务/规则起到了关键作用。为了解决这个问题，本文介绍了一种由归纳推理设计中的贫乏刺激设计启发的合成诊断任务；在这里，大多数示例在基本规则方面是模棱两可的，而一个关键的示例则澄清了所展示的任务。问题是，传统的IA方法是否能在解释ICL中的归纳推理过程时识别出这样的示例。我们的实验提供了多个实际发现，例如，某些简单的IA方法表现最好，而随着模型规模的增大，基于梯度的IA方法通常更难以解释ICL。 

---
# Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification 

**Title (ZH)**: 面向少样本对话意图分类的动态标签名称精炼 

**Authors**: Gyutae Park, Ingeol Baek, ByeongJeong Kim, Joongbo Shin, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2412.15603)  

**Abstract**: Dialogue intent classification aims to identify the underlying purpose or intent of a user's input in a conversation. Current intent classification systems encounter considerable challenges, primarily due to the vast number of possible intents and the significant semantic overlap among similar intent classes. In this paper, we propose a novel approach to few-shot dialogue intent classification through in-context learning, incorporating dynamic label refinement to address these challenges. Our method retrieves relevant examples for a test input from the training set and leverages a large language model to dynamically refine intent labels based on semantic understanding, ensuring that intents are clearly distinguishable from one another. Experimental results demonstrate that our approach effectively resolves confusion between semantically similar intents, resulting in significantly enhanced performance across multiple datasets compared to baselines. We also show that our method generates more interpretable intent labels, and has a better semantic coherence in capturing underlying user intents compared to baselines. 

**Abstract (ZH)**: 对话意图分类的目标是识别用户输入在对话中存在的潜在目的或意图。当前的意图分类系统面临诸多挑战，主要原因是可能的意图种类繁多以及相似意图类别之间的重大语义重叠。在本文中，我们提出了一种通过上下文学习实现少样本对话意图分类的新型方法，并结合动态标签优化以应对这些挑战。该方法从训练集中检索与测试输入相关的示例，并利用大型语言模型基于语义理解动态优化意图标签，确保各个意图之间能够明显区分。实验结果表明，我们的方法有效解决了语义相似意图之间的混淆，相较于基线方法，在多个数据集上均展现出显著提高的性能。此外，我们也证明了我们的方法生成了更具可解释性的意图标签，并在捕捉用户潜在意图方面具有更好的语义连贯性。 

---
# NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization 

**Title (ZH)**: NeSyCoCo：一种神经符号概念合成器，实现组合泛化 

**Authors**: Danial Kamali, Elham J. Barezi, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2412.15588)  

**Abstract**: Compositional generalization is crucial for artificial intelligence agents to solve complex vision-language reasoning tasks. Neuro-symbolic approaches have demonstrated promise in capturing compositional structures, but they face critical challenges: (a) reliance on predefined predicates for symbolic representations that limit adaptability, (b) difficulty in extracting predicates from raw data, and (c) using non-differentiable operations for combining primitive concepts. To address these issues, we propose NeSyCoCo, a neuro-symbolic framework that leverages large language models (LLMs) to generate symbolic representations and map them to differentiable neural computations. NeSyCoCo introduces three innovations: (a) augmenting natural language inputs with dependency structures to enhance the alignment with symbolic representations, (b) employing distributed word representations to link diverse, linguistically motivated logical predicates to neural modules, and (c) using the soft composition of normalized predicate scores to align symbolic and differentiable reasoning. Our framework achieves state-of-the-art results on the ReaSCAN and CLEVR-CoGenT compositional generalization benchmarks and demonstrates robust performance with novel concepts in the CLEVR-SYN benchmark. 

**Abstract (ZH)**: 组合泛化对于人工智能代理解决复杂的视知觉推理任务至关重要。神经符号方法在捕捉组合结构方面显示出前景，但它们面临关键挑战：（a）依赖预定义谓词进行符号表示，这限制了适应性；（b）从原始数据中抽取出谓词的难度；（c）使用非可微操作结合基本概念。为了解决这些问题，我们提出了NeSyCoCo这一神经符号框架，利用大规模语言模型（LLMs）生成符号表示并将其映射到可微神经计算中。NeSyCoCo引入了三项创新：（a）在自然语言输入中增加依存结构，以增强与符号表示的对齐；（b）采用分布式词向量表示，将多种语言动机逻辑谓词链接到神经模块；（c）利用标准化谓词得分的软组合，使符号和可微推理对齐。我们的框架在ReaSCAN和CLEVR-CoGenT组合泛化基准测试中达到了最先进的效果，并在CLEVR-SYN基准测试中的新技术概念上表现出稳健的性能。 

---
# Mitigating Social Bias in Large Language Models: A Multi-Objective Approach within a Multi-Agent Framework 

**Title (ZH)**: 在多代理框架内的多目标方法减轻大型语言模型中的社会偏见 

**Authors**: Zhenjie Xu, Wenqing Chen, Yi Tang, Xuanying Li, Cheng Hu, Zhixuan Chu, Kui Ren, Zibin Zheng, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15504)  

**Abstract**: Natural language processing (NLP) has seen remarkable advancements with the development of large language models (LLMs). Despite these advancements, LLMs often produce socially biased outputs. Recent studies have mainly addressed this problem by prompting LLMs to behave ethically, but this approach results in unacceptable performance degradation. In this paper, we propose a multi-objective approach within a multi-agent framework (MOMA) to mitigate social bias in LLMs without significantly compromising their performance. The key idea of MOMA involves deploying multiple agents to perform causal interventions on bias-related contents of the input questions, breaking the shortcut connection between these contents and the corresponding answers. Unlike traditional debiasing techniques leading to performance degradation, MOMA substantially reduces bias while maintaining accuracy in downstream tasks. Our experiments conducted on two datasets and two models demonstrate that MOMA reduces bias scores by up to 87.7%, with only a marginal performance degradation of up to 6.8% in the BBQ dataset. Additionally, it significantly enhances the multi-objective metric icat in the StereoSet dataset by up to 58.1%. Code will be made available at this https URL. 

**Abstract (ZH)**: 自然语言处理（NLP）随着大型语言模型（LLMs）的发展取得了显著进展。尽管取得了这些进展，但LLMs常常产生社会偏见的输出。近期的研究主要通过促使LLMs表现得更加伦理来解决这一问题，但这种做法会导致性能显著下降。本文提出了一种多目标方法（MOMA），在多智能体框架内减轻LLMs中的社会偏见，而不显著牺牲其性能。MOMA的核心思想是部署多个智能体对输入问题中的偏见相关内容进行因果干预，从而打破这些内容与相应答案之间的捷径连接。与传统去偏方法导致的性能下降不同，MOMA在大幅减少偏见的同时，能够维持下游任务的准确性。我们在两个数据集和两个模型上进行的实验表明，MOMA在BBQ数据集上仅导致至多6.8%的轻微性能下降，同时将偏见分数降低至高达87.7%。此外，MOMA还在StereoSet数据集上显著提高了多目标指标icat，提高幅度高达58.1%。代码将在以下链接处开源：this https URL。 

---
# TL-Training: A Task-Feature-Based Framework for Training Large Language Models in Tool Use 

**Title (ZH)**: TL-训练：一种基于任务特征的大型语言模型训练框架，用于工具使用 

**Authors**: Junjie Ye, Yilong Wu, Sixian Li, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang, Peng Wang, Zhongchao Shi, Jianping Fan, Zhengyin Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.15495)  

**Abstract**: Large language models (LLMs) achieve remarkable advancements by leveraging tools to interact with external environments, a critical step toward generalized AI. However, the standard supervised fine-tuning (SFT) approach, which relies on large-scale datasets, often overlooks task-specific characteristics in tool use, leading to performance bottlenecks. To address this issue, we analyze three existing LLMs and uncover key insights: training data can inadvertently impede tool-use behavior, token importance is distributed unevenly, and errors in tool calls fall into a small set of distinct categories. Building on these findings, we propose TL-Training, a task-feature-based framework that mitigates the effects of suboptimal training data, dynamically adjusts token weights to prioritize key tokens during SFT, and incorporates a robust reward mechanism tailored to error categories, optimized through proximal policy optimization. We validate TL-Training by training CodeLLaMA-2-7B and evaluating it on four diverse open-source test sets. Our results demonstrate that the LLM trained by our method matches or surpasses both open- and closed-source LLMs in tool-use performance using only 1,217 training data points. Additionally, our method enhances robustness in noisy environments and improves general task performance, offering a scalable and efficient paradigm for tool-use training in LLMs. The code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过利用工具与外部环境交互，实现了显著的进步，这是通向通用人工智能的关键步骤。然而，依赖大规模数据集的标准监督微调（SFT）方法往往忽略了工具使用中的任务特定特征，导致性能瓶颈。为了解决这一问题，我们分析了三种现有的LLMs，并发现了一些关键见解：训练数据可能无意中阻碍了工具使用行为，各标记的重要性分布不均，工具调用错误主要集中在几个不同的类别中。基于这些发现，我们提出了一种基于任务特征的TL-Training框架，该框架可以缓解不理想的训练数据的影响，在微调过程中动态调整标记权重以优先处理关键标记，并引入了一种针对错误类别优化的 robust 奖励机制。我们通过训练 CodeLLaMA-2-7B 并使用四个不同的开源测试集对其进行评估，验证了TL-Training的有效性。我们的结果显示，通过我们的方法训练的LLM仅使用1,217个训练数据点，就能在工具使用性能上匹配或超越开源和封闭源的LLM。此外，我们的方法还提高了在噪声环境中的稳健性，并提高了通用任务性能，为LLMs中的工具使用训练提供了一种可扩展且高效的范式。完整的代码和数据可在以下链接获取：[请插入链接]。 

---
# Multi-LLM Text Summarization 

**Title (ZH)**: 多语言大型语言模型文本摘要 

**Authors**: Jiangnan Fang, Cheng-Tse Liu, Jieun Kim, Yash Bhedaru, Ethan Liu, Nikhil Singh, Nedim Lipka, Puneet Mathur, Nesreen K. Ahmed, Franck Dernoncourt, Ryan A. Rossi, Hanieh Deilamsalehy  

**Link**: [PDF](https://arxiv.org/pdf/2412.15487)  

**Abstract**: In this work, we propose a Multi-LLM summarization framework, and investigate two different multi-LLM strategies including centralized and decentralized. Our multi-LLM summarization framework has two fundamentally important steps at each round of conversation: generation and evaluation. These steps are different depending on whether our multi-LLM decentralized summarization is used or centralized. In both our multi-LLM decentralized and centralized strategies, we have k different LLMs that generate diverse summaries of the text. However, during evaluation, our multi-LLM centralized summarization approach leverages a single LLM to evaluate the summaries and select the best one whereas k LLMs are used for decentralized multi-LLM summarization. Overall, we find that our multi-LLM summarization approaches significantly outperform the baselines that leverage only a single LLM by up to 3x. These results indicate the effectiveness of multi-LLM approaches for summarization. 

**Abstract (ZH)**: 在本文中，我们提出了一种多大型语言模型（Multi-LLM）总结框架，并探讨了两种不同的多大型语言模型策略，包括集中式和去中心化策略。我们的多大型语言模型总结框架在每次对话回合中包含两个基本步骤：生成和评估。这两个步骤会根据是使用去中心化的多大型语言模型总结还是集中式的总结而有所不同。在我们的去中心化和集中式多大型语言模型策略中，都会使用k个不同的大型语言模型来生成文本的多样化总结。然而，在评估阶段，我们的集中式多大型语言模型总结方法使用单一的大型语言模型来评估总结并选择最佳结果，而在去中心化的多大型语言模型总结方法中，会使用k个大型语言模型。总体而言，我们发现我们的多大型语言模型总结方法相对于只使用单一大型语言模型的基线方法取得显著改进，最高提升可达3倍。这些结果表明，多大型语言模型方法在总结任务中的有效性。 

---
# Automatic Extraction of Metaphoric Analogies from Literary Texts: Task Formulation, Dataset Construction, and Evaluation 

**Title (ZH)**: 自动提取文学文本中的隐喻类比：任务表述、数据集构建与评估 

**Authors**: Joanne Boisson, Zara Siddique, Hsuvas Borkakoty, Dimosthenis Antypas, Luis Espinosa Anke, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2412.15375)  

**Abstract**: Extracting metaphors and analogies from free text requires high-level reasoning abilities such as abstraction and language understanding. Our study focuses on the extraction of the concepts that form metaphoric analogies in literary texts. To this end, we construct a novel dataset in this domain with the help of domain experts. We compare the out-of-the-box ability of recent large language models (LLMs) to structure metaphoric mappings from fragments of texts containing proportional analogies. The models are further evaluated on the generation of implicit elements of the analogy, which are indirectly suggested in the texts and inferred by human readers. The competitive results obtained by LLMs in our experiments are encouraging and open up new avenues such as automatically extracting analogies and metaphors from text instead of investing resources in domain experts to manually label data. 

**Abstract (ZH)**: 从自由文本中提取隐喻和类比需要高级推理能力，如抽象和语言理解。本研究重点关注文学文本中构成隐喻类比的概念提取。为此，我们在专家的帮助下构建了一个新的数据集。我们对比了近期大语言模型（LLMs）直接从包含比例类比的文本片段中结构化隐喻映射的能力。进一步地，我们评估了这些模型生成类比中隐含元素的能力，这些隐含元素在文本中间接暗示并需要人类读者进行推断。我们在实验中获得的具有竞争力的结果令人鼓舞，这为自动从文本中提取隐喻和类比开辟了新的途径，而不是投入资源请领域专家手动标注数据。 

---
# Confidence in the Reasoning of Large Language Models 

**Title (ZH)**: 大型语言模型推理的置信度 

**Authors**: Yudi Pawitan, Chris Holmes  

**Link**: [PDF](https://arxiv.org/pdf/2412.15296)  

**Abstract**: There is a growing literature on reasoning by large language models (LLMs), but the discussion on the uncertainty in their responses is still lacking. Our aim is to assess the extent of confidence that LLMs have in their answers and how it correlates with accuracy. Confidence is measured (i) qualitatively in terms of persistence in keeping their answer when prompted to reconsider, and (ii) quantitatively in terms of self-reported confidence score. We investigate the performance of three LLMs -- GPT4o, GPT4-turbo and Mistral -- on two benchmark sets of questions on causal judgement and formal fallacies and a set of probability and statistical puzzles and paradoxes. Although the LLMs show significantly better performance than random guessing, there is a wide variability in their tendency to change their initial answers. There is a positive correlation between qualitative confidence and accuracy, but the overall accuracy for the second answer is often worse than for the first answer. There is a strong tendency to overstate the self-reported confidence score. Confidence is only partially explained by the underlying token-level probability. The material effects of prompting on qualitative confidence and the strong tendency for overconfidence indicate that current LLMs do not have any internally coherent sense of confidence. 

**Abstract (ZH)**: 关于大型语言模型（LLMs）的推理研究正在逐步增多，但对于其回答时的不确定性讨论仍然不足。本研究旨在评估LLMs对其答案的信心程度及其与准确性的关联。信心的度量方法包括：（i）定性方面，即在被要求重新考虑时坚持其答案的坚持程度；（ii）定量方面，即自我报告的信心分数。我们对三种LLM——GPT4o、GPT4-turbo和Mistral——在因果判断和逻辑谬误、概率与统计悖论两套基准问题集上的表现进行了研究。尽管LLMs的表现明显优于随机猜测，但在更改初始答案的倾向上存在显著差异。定性信心与准确性之间存在正相关，但一般情况下，第二次答案的总体准确性往往低于第一次答案。自我报告的信心评分存在显著的夸大倾向。信心仅部分由底层标记级概率解释。在定性信心方面的提示效应以及过度自信的倾向表明，当前的LLMs并没有任何内在一致的信心感。 

---
# A Large-scale Empirical Study on Large Language Models for Election Prediction 

**Title (ZH)**: 大规模实证研究：大型语言模型在选举预测中的应用 

**Authors**: Chenxiao Yu, Zhaotian Weng, Yuangang Li, Zheng Li, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.15291)  

**Abstract**: Can Large Language Models (LLMs) accurately predict election outcomes? While LLMs have demonstrated impressive performance in healthcare, legal analysis, and creative applications, their capabilities in election forecasting remain uncertain. Notably, election prediction poses unique challenges: limited voter-level data, evolving political contexts, and the complexity of modeling human behavior. In the first part of this paper, we explore and introduce a multi-step reasoning framework for election prediction, which systematically integrates demographic, ideological, and time-sensitive factors. Validated on 2016 and 2020 real-world data and extensive synthetic personas, our approach adapts to changing political landscapes, reducing bias and significantly improving predictive accuracy. We further apply our pipeline to the 2024 U.S. presidential election, illustrating its ability to generalize beyond observed historical data. Beyond enhancing accuracy, the second part of the paper provides insights into the broader implications of LLM-based election forecasting. We identify potential political biases embedded in pretrained corpora, examine how demographic patterns can become exaggerated, and suggest strategies for mitigating these issues. Together, this project, a large-scale LLM empirical study, advances the accuracy of election predictions and establishes directions for more balanced, transparent, and context-aware modeling in political science research and practice. 

**Abstract (ZH)**: 大语言模型（LLMs）能否准确预测选举结果？尽管LLMs在医疗保健、法律分析和创意应用方面表现出色，但在选举预测领域的表现仍有待验证。值得注意的是，选举预测面临独特的挑战：选民层面数据有限，政治环境不断变化以及建模人类行为的复杂性。在本文的第一部分中，我们探讨并引入了一种多步推理框架，系统地整合了人口统计学、意识形态和时间敏感因素。我们通过2016年和2020年的实际数据以及广泛的合成人物验证了该方法，使其能够适应不断变化的政治格局，减少偏差并显著提高预测准确性。我们进一步将该框架应用于2024年美国总统选举，展示了其超越既往历史数据的能力。除了提高预测准确性，本文第二部分还探讨了基于LLM的选举预测在更广泛层面的意义。我们识别了预训练语料库中潜在的政治偏见，分析了人口模式如何被放大，并提出了缓解这些问题的策略。该项目是一个大规模的LLM实证研究，推动了选举预测的准确性，并为政治科学研究与实践中更均衡、透明、情境感知的建模方向指明了方向。 

---
# A MapReduce Approach to Effectively Utilize Long Context Information in Retrieval Augmented Language Models 

**Title (ZH)**: 一种利用MapReduce有效利用长上下文信息的检索增强语言模型方法 

**Authors**: Gongbo Zhang, Zihan Xu, Qiao Jin, Fangyi Chen, Yilu Fang, Yi Liu, Justin F. Rousseau, Ziyang Xu, Zhiyong Lu, Chunhua Weng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2412.15271)  

**Abstract**: While holding great promise for improving and facilitating healthcare, large language models (LLMs) struggle to produce up-to-date responses on evolving topics due to outdated knowledge or hallucination. Retrieval-augmented generation (RAG) is a pivotal innovation that improves the accuracy and relevance of LLM responses by integrating LLMs with a search engine and external sources of knowledge. However, the quality of RAG responses can be largely impacted by the rank and density of key information in the retrieval results, such as the "lost-in-the-middle" problem. In this work, we aim to improve the robustness and reliability of the RAG workflow in the medical domain. Specifically, we propose a map-reduce strategy, BriefContext, to combat the "lost-in-the-middle" issue without modifying the model weights. We demonstrated the advantage of the workflow with various LLM backbones and on multiple QA datasets. This method promises to improve the safety and reliability of LLMs deployed in healthcare domains. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在改善和促进医疗保健方面充满了潜力，但由于知识过时或幻觉，它们在生成关于不断演变的主题的最新回应方面面临挑战。检索增强生成（RAG，Retrieval-Augmented Generation）是一种关键创新，通过将LLMs与搜索引擎和外部知识来源结合使用，提高了LLMs响应的准确性和相关性。然而，RAG响应的质量往往受到检索结果中关键信息排名和密度的影响，如“迷失在中间”的问题。本文旨在增强医疗领域RAG工作流的稳健性和可靠性。具体而言，我们提出了一种映射-减少策略，称为简要上下文（BriefContext），以在不修改模型权重的情况下解决“迷失在中间”的问题。我们通过使用多种LLMs底座和多个问答数据集验证了该工作流的优势。该方法有望提高部署在医疗领域中的LLMs的安全性和可靠性。 

---
# Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large Language Models 

**Title (ZH)**: 中国安全QA：大规模语言模型的安全简短事实性基准 

**Authors**: Yingshui Tan, Boren Zheng, Baihui Zheng, Kerui Cao, Huiyun Jing, Jincheng Wei, Jiaheng Liu, Yancheng He, Wenbo Su, Xiangyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.15265)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), significant safety concerns have emerged. Fundamentally, the safety of large language models is closely linked to the accuracy, comprehensiveness, and clarity of their understanding of safety knowledge, particularly in domains such as law, policy and ethics. This factuality ability is crucial in determining whether these models can be deployed and applied safely and compliantly within specific regions. To address these challenges and better evaluate the factuality ability of LLMs to answer short questions, we introduce the Chinese SafetyQA benchmark. Chinese SafetyQA has several properties (i.e., Chinese, Diverse, High-quality, Static, Easy-to-evaluate, Safety-related, Harmless). Based on Chinese SafetyQA, we perform a comprehensive evaluation on the factuality abilities of existing LLMs and analyze how these capabilities relate to LLM abilities, e.g., RAG ability and robustness against attacks. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速进步，安全问题日益突出。从根本上说，大型语言模型的安全性与其对安全知识的理解准确性、全面性和清晰度密切相关，特别是在法律、政策和伦理等领域。这种事实判断能力决定了这些模型是否能够在特定区域安全合规地部署和应用。为了应对这些挑战，并更好地评估LLMs回答简短问题的事实判断能力，我们提出了中文安全QA基准（Chinese SafetyQA）。中文安全QA具有以下特性：（1）中文性；（2）多样性；（3）高质量；（4）静态性；（5）易于评估；（6）与安全相关；（7）无害性。基于中文安全QA，我们对现有的LLMs的事实判断能力进行了全面评估，并分析了这些能力与LLMs的其他能力（例如RAG能力及对抗攻击的鲁棒性）之间的关系。 

---
# Advanced ingestion process powered by LLM parsing for RAG system 

**Title (ZH)**: 基于LLM解析的高级检索过程用于 Retrieval-Augmented Generation 系统 

**Authors**: Arnau Perez, Xavier Vizcaino  

**Link**: [PDF](https://arxiv.org/pdf/2412.15262)  

**Abstract**: Retrieval Augmented Generation (RAG) systems struggle with processing multimodal documents of varying structural complexity. This paper introduces a novel multi-strategy parsing approach using LLM-powered OCR to extract content from diverse document types, including presentations and high text density files both scanned or not. The methodology employs a node-based extraction technique that creates relationships between different information types and generates context-aware metadata. By implementing a Multimodal Assembler Agent and a flexible embedding strategy, the system enhances document comprehension and retrieval capabilities. Experimental evaluations across multiple knowledge bases demonstrate the approach's effectiveness, showing improvements in answer relevancy and information faithfulness. 

**Abstract (ZH)**: 基于检索增强生成（RAG）系统在处理结构复杂度各异的多模态文档方面存在挑战。本文介绍了一种新颖的多策略解析方法，利用大型语言模型（LLM）驱动的光学字符识别（OCR）技术，从不同类型的文件中提取内容，包括演示文稿和高文本密度文件（无论是扫描还是未扫描）。该方法采用基于节点的提取技术，创建不同类型信息之间的关系，并生成上下文感知的元数据。通过实施多模态组装代理和灵活的嵌入策略，系统增强了文档理解和检索能力。在多个知识库上的实验评估表明，该方法的有效性，在回答相关性和信息忠实性方面表现出改进。 

---
# RIRO: Reshaping Inputs, Refining Outputs Unlocking the Potential of Large Language Models in Data-Scarce Contexts 

**Title (ZH)**: RIRO：重塑输入，精炼输出——在数据稀缺环境中释放大规模语言模型的潜力 

**Authors**: Ali Hamdi, Hozaifa Kassab, Mohamed Bahaa, Marwa Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2412.15254)  

**Abstract**: Large language models (LLMs) have significantly advanced natural language processing, excelling in areas like text generation, summarization, and question-answering. Despite their capabilities, these models face challenges when fine-tuned on small, domain-specific datasets, often struggling to generalize and deliver accurate results with unfamiliar inputs. To tackle this issue, we introduce RIRO, a novel two-layer architecture designed to improve performance in data-scarce environments. The first layer leverages advanced prompt engineering to reformulate inputs, ensuring better alignment with training data, while the second layer focuses on refining outputs to minimize inconsistencies. Through fine-tuning models like Phi-2, Falcon 7B, and Falcon 1B, with Phi-2 outperforming the others. Additionally, we introduce a benchmark using evaluation metrics such as cosine similarity, Levenshtein distance, BLEU score, ROUGE-1, ROUGE-2, and ROUGE-L. While these advancements improve performance, challenges like computational demands and overfitting persist, limiting the potential of LLMs in data-scarce, high-stakes environments such as healthcare, legal documentation, and software testing. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理方面取得了显著进展，特别是在文本生成、摘要和问答等领域表现出色。尽管如此，这些模型在使用小规模、领域特定的数据集微调时，往往会面临泛化能力不足的问题，往往难以对不熟悉的数据输入提供准确的结果。为解决这一问题，我们提出了一种名为RIRO的新颖两层结构，旨在提高在数据稀缺环境中性能。第一层通过先进的提示工程来重新格式化输入，确保更好地与训练数据对齐，而第二层则专注于优化输出，减少不一致性。通过使用Phi-2、Falcon 7B和Falcon 1B等模型进行微调，结果显示Phi-2的表现优于其他模型。此外，我们还引入了一个使用余弦相似度、Levenshtein距离、BLEU分数、ROUGE-1、ROUGE-2和ROUGE-L等评价指标的基准测试。虽然这些进步提高了性能，但计算需求高和过拟合等问题仍然存在，限制了LLMs在医疗保健、法律文件和软件测试等数据稀缺、高风险环境中的应用潜力。 

---
# AgentPS: Agentic Process Supervision for Multi-modal Content Quality Assurance through Multi-round QA 

**Title (ZH)**: AgentPS：代理过程监督在多模态内容质量保障中的多轮问答方法 

**Authors**: Gorden Liu, Yu Sun, Ruixiao Sun, Xin Dong, Hongyu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2412.15251)  

**Abstract**: The advanced processing and reasoning capabilities of multimodal large language models (MLLMs) have driven substantial progress in vision-language (VL) understanding tasks. However, while effective for tasks governed by straightforward logic, MLLMs often encounter challenges when reasoning over complex, interdependent logic structures. To address this limitation, we introduce \textit{AgentPS}, a novel framework that integrates Agentic Process Supervision into MLLMs via multi-round question answering during fine-tuning. \textit{AgentPS} demonstrates significant performance improvements over baseline MLLMs on proprietary TikTok datasets, due to its integration of process supervision and structured sequential reasoning. Furthermore, we show that replacing human-annotated labels with LLM-generated labels retains much of the performance gain, highlighting the framework's practical scalability in industrial applications. These results position \textit{AgentPS} as a highly effective and efficient architecture for multimodal classification tasks. Its adaptability and scalability, especially when enhanced by automated annotation generation, make it a powerful tool for handling large-scale, real-world challenges. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）的高级处理和推理能力推动了视觉语言（VL）理解任务的显著进展。然而，这类模型在处理复杂互动的逻辑结构时往往遇到挑战。为此，我们提出了一种名为AgentPS的新框架，通过微调期间多轮问答集成代理过程监督。AgentPS在专用的TikTok数据集上展示了显著的性能提升，这得益于其结合了过程监督和结构化顺序推理。进一步的研究表明，用大语言模型生成的标签替代人工标注的标签仍能保持大部分的性能提升，突显了该框架在工业应用层面的实用可扩展性。这些结果将AgentPS定位为一种在多模态分类任务中高效且有效的架构。其适应性与可扩展性，特别是在增强自动化标注生成时，使其成为应对大规模真实世界挑战的强大工具。 

---
# Streamlining Systematic Reviews: A Novel Application of Large Language Models 

**Title (ZH)**: 简化系统评价：大型语言模型的一项新型应用 

**Authors**: Fouad Trad, Ryan Yammine, Jana Charafeddine, Marlene Chakhtoura, Maya Rahme, Ghada El-Hajj Fuleihan, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2412.15247)  

**Abstract**: Systematic reviews (SRs) are essential for evidence-based guidelines but are often limited by the time-consuming nature of literature screening. We propose and evaluate an in-house system based on Large Language Models (LLMs) for automating both title/abstract and full-text screening, addressing a critical gap in the literature. Using a completed SR on Vitamin D and falls (14,439 articles), the LLM-based system employed prompt engineering for title/abstract screening and Retrieval-Augmented Generation (RAG) for full-text screening. The system achieved an article exclusion rate (AER) of 99.5%, specificity of 99.6%, a false negative rate (FNR) of 0%, and a negative predictive value (NPV) of 100%. After screening, only 78 articles required manual review, including all 20 identified by traditional methods, reducing manual screening time by 95.5%. For comparison, Rayyan, a commercial tool for title/abstract screening, achieved an AER of 72.1% and FNR of 5% when including articles Rayyan considered as undecided or likely to include. Lowering Rayyan's inclusion thresholds improved FNR to 0% but increased screening time. By addressing both screening phases, the LLM-based system significantly outperformed Rayyan and traditional methods, reducing total screening time to 25.5 hours while maintaining high accuracy. These findings highlight the transformative potential of LLMs in SR workflows by offering a scalable, efficient, and accurate solution, particularly for the full-text screening phase, which has lacked automation tools. 

**Abstract (ZH)**: 系统评价（SRs）对于基于证据的指南至关重要，但文献筛选往往耗时较长，限制了其应用。我们提出并评估了一种基于大型语言模型（LLMs）的自研系统，用于自动化标题/摘要筛选和全文筛选，填补了文献中的一项关键空白。利用一项已完成的维生素D与跌倒相关的SR（共14,439篇文章），该基于LLM的系统使用提示工程进行标题/摘要筛选，并使用检索增强生成（RAG）进行全文筛选。系统实现了99.5%的文章排除率（AER）、99.6%的特异性、0%的假阴性率（FNR）和100%的阴性预测值（NPV）。在筛选之后，仅需人工复查78篇文章，其中包括传统方法识别出的全部20篇文章，减少了95.5%的手动筛选时间。相比之下，商业化工具Rayyan在包括Rayyan认为待决定或很可能包含的文章时，实现了72.1%的文章排除率（AER）和5%的假阴性率（FNR）。降低Rayyan的纳入阈值可将假阴性率降至0%，但增加了筛选时间。通过同时解决两个筛选阶段，基于LLM的系统显著优于Rayyan和传统方法，在保持高准确性的前提下将总体筛选时间缩短至25.5小时。这些发现突显了大型语言模型在SR工作流程中具有变革性潜力，提供了一种可扩展、高效且准确的解决方案，特别是在全文筛选阶段缺乏自动化工具的情况下。 

---
# MORTAR: Metamorphic Multi-turn Testing for LLM-based Dialogue Systems 

**Title (ZH)**: MORTAR：变倍式多轮测试方法用于基于LLM的对话系统 

**Authors**: Guoxiang Guo, Aldeida Aleti, Neelofar Neelofar, Chakkrit Tantithamthavorn  

**Link**: [PDF](https://arxiv.org/pdf/2412.15557)  

**Abstract**: With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn scenarios. However, multi-turn dialogue testing remains underexplored, with the Oracle problem in multi-turn testing posing a persistent challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a MetamORphic multi-TuRn diAlogue testing appRoach, which mitigates the test oracle problem in the assessment of LLM-based dialogue systems. MORTAR automates the generation of follow-up question-answer (QA) dialogue test cases with multiple dialogue-level perturbations and metamorphic relations. MORTAR employs a novel knowledge graph-based dialogue information model which effectively generates perturbed dialogue test datasets and detects bugs of multi-turn dialogue systems in a low-cost manner. The proposed approach does not require an LLM as a judge, eliminating potential of any biases in the evaluation step. According to the experiment results on multiple LLM-based dialogue systems and comparisons with single-turn metamorphic testing approaches, MORTAR explores more unique bugs in LLM-based dialogue systems, especially for severe bugs that MORTAR detects up to four times more unique bugs than the most effective existing metamorphic testing approach. 

**Abstract (ZH)**: 随着基于大语言模型（LLM）的对话系统在日常生活中的广泛应用，质量保证变得比以往任何时候都更加重要。最近的研究成功引入了识别单轮场景中意外行为的方法。然而，多轮对话测试仍处于探索阶段，并且在多轮测试中存在一个持续性的挑战——Oracle问题，这给对话系统开发人员和研究者带来了困难。本文提出了一种名为MORTAR的元Q反演变式多轮对话测试方法，该方法在评估基于LLM的对话系统时减轻了测试Oracle问题。MORTAR自动生成具有多轮对话级扰动和元变换关系的后续问题-答案（QA）对话测试案例。MORTAR采用一种基于知识图谱的对话信息模型，可以有效地生成扰动对话测试数据集，并以低成本的方式检测多轮对话系统的故障。该方法不需要使用LLM作为评判标准，消除了评估步骤中的任何偏差可能性。根据在多个基于LLM的对话系统上的实验结果和与单轮元变换测试方法的比较，MORTAR在基于LLM的对话系统中发现更多独特的bug，尤其是MORTAR检测到的更为严重的bug，相比目前最有效的现有元变换测试方法，MORTAR能检测到的唯一bug多四倍。 

---
# Time Will Tell: Timing Side Channels via Output Token Count in Large Language Models 

**Title (ZH)**: 时间会证明一切：通过大型语言模型的输出token数量进行定时侧信道攻击 

**Authors**: Tianchen Zhang, Gururaj Saileshwar, David Lie  

**Link**: [PDF](https://arxiv.org/pdf/2412.15431)  

**Abstract**: This paper demonstrates a new side-channel that enables an adversary to extract sensitive information about inference inputs in large language models (LLMs) based on the number of output tokens in the LLM response. We construct attacks using this side-channel in two common LLM tasks: recovering the target language in machine translation tasks and recovering the output class in classification tasks. In addition, due to the auto-regressive generation mechanism in LLMs, an adversary can recover the output token count reliably using a timing channel, even over the network against a popular closed-source commercial LLM. Our experiments show that an adversary can learn the output language in translation tasks with more than 75% precision across three different models (Tower, M2M100, MBart50). Using this side-channel, we also show the input class in text classification tasks can be leaked out with more than 70% precision from open-source LLMs like Llama-3.1, Llama-3.2, Gemma2, and production models like GPT-4o. Finally, we propose tokenizer-, system-, and prompt-based mitigations against the output token count side-channel. 

**Abstract (ZH)**: 本文展示了一种新的侧信道，它使攻击者能够在大型语言模型（LLMs）的响应中根据生成的输出令牌数量提取关于推理输入的敏感信息。我们使用这种侧信道构造了两种常见LLM任务的攻击：在机器翻译任务中恢复目标语言，在分类任务中恢复输出类别。此外，由于LLMs中的自回归生成机制，即使是在网络上对流行的封闭源商业LLM进行攻击时，攻击者也可以通过时间信道可靠地恢复输出令牌数量。我们的实验表明，攻击者可以通过三个不同模型（Tower、M2M100、MBart50）在超过75%的精度下学习翻译任务的输出语言。通过这种方法，我们还展示了可以从开源LLM（如Llama-3.1、Llama-3.2、Gemma2以及生产模型如GPT-4o）的文本分类任务中以超过70%的精度泄露输入类别。最后，我们提出了针对输出令牌数量侧信道的基于分词器、系统和提示的缓解措施。 

---
# Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support 

**Title (ZH)**: Legommenders：一种具备LLM支持的全面内容基推荐库 

**Authors**: Qijiong Liu, Lu Fan, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15973)  

**Abstract**: We present Legommenders, a unique library designed for content-based recommendation that enables the joint training of content encoders alongside behavior and interaction modules, thereby facilitating the seamless integration of content understanding directly into the recommendation pipeline. Legommenders allows researchers to effortlessly create and analyze over 1,000 distinct models across 15 diverse datasets. Further, it supports the incorporation of contemporary large language models, both as feature encoder and data generator, offering a robust platform for developing state-of-the-art recommendation models and enabling more personalized and effective content delivery. 

**Abstract (ZH)**: 我们介绍了一种名为Legommenders的独特图书管理系统，它专为基于内容的推荐设计，能够同时训练内容编码器和行为与交互模块，从而实现直接将内容理解无缝集成到推荐流程中。Legommenders允许研究人员轻松创建和分析超过1,000种不同的模型，并可在15个不同的数据集上进行实验。此外，它支持将当今的大型语言模型纳入其中，作为特征编码器和数据生成器，提供了一个强大的平台，用于开发最新的推荐模型，并促进更为个性化和有效的内容交付。 

---
# Accelerating Retrieval-Augmented Generation 

**Title (ZH)**: 加速检索增强生成 

**Authors**: Derrick Quinn, Mohammad Nouri, Neel Patel, John Salihu, Alireza Salemi, Sukhan Lee, Hamed Zamani, Mohammad Alian  

**Link**: [PDF](https://arxiv.org/pdf/2412.15246)  

**Abstract**: An evolving solution to address hallucination and enhance accuracy in large language models (LLMs) is Retrieval-Augmented Generation (RAG), which involves augmenting LLMs with information retrieved from an external knowledge source, such as the web. This paper profiles several RAG execution pipelines and demystifies the complex interplay between their retrieval and generation phases. We demonstrate that while exact retrieval schemes are expensive, they can reduce inference time compared to approximate retrieval variants because an exact retrieval model can send a smaller but more accurate list of documents to the generative model while maintaining the same end-to-end accuracy. This observation motivates the acceleration of the exact nearest neighbor search for RAG.
In this work, we design Intelligent Knowledge Store (IKS), a type-2 CXL device that implements a scale-out near-memory acceleration architecture with a novel cache-coherent interface between the host CPU and near-memory accelerators. IKS offers 13.4-27.9x faster exact nearest neighbor search over a 512GB vector database compared with executing the search on Intel Sapphire Rapids CPUs. This higher search performance translates to 1.7-26.3x lower end-to-end inference time for representative RAG applications. IKS is inherently a memory expander; its internal DRAM can be disaggregated and used for other applications running on the server to prevent DRAM, which is the most expensive component in today's servers, from being stranded. 

**Abstract (ZH)**: 解决大型语言模型（LLMs）幻觉问题并提高其准确性的不断演化的解决方案是检索增强生成（RAG），这种技术通过从外部知识来源（例如网络）检索信息来增强LLMs。本文概述了几种RAG执行管道，并阐明了其检索和生成阶段之间的复杂交互关系。我们展示了虽然精确检索方案成本较高，但与近似检索变体相比，它们可以在保持相同端到端准确性的前提下减少推理时间，因为精确检索模型可以向生成模型发送更小但更准确的文档列表。这一观察结果促使我们加速RAG中的精确最近邻搜索。

在本文中，我们设计了一种名为智能知识存储（IKS）的CXL类型2设备，它实现了扩展的近内存加速架构，并在主机CPU和近内存加速器之间采用了一种新颖的缓存一致性接口。相比于在Intel Sapphire Rapids CPU上执行搜索，IKS在512GB向量数据库上的精确最近邻搜索速度提高了13.4到27.9倍。这种更高的搜索性能转化为代表性的RAG应用程序中1.7到26.3倍的端到端推理时间降低。IKS本质上是一种内存扩展器；其内部DRAM可以分离并用于服务器上的其他应用程序，从而防止当今服务器中最昂贵的组件——DRAM——被闲置。 

---
# Nano-ESG: Extracting Corporate Sustainability Information from News Articles 

**Title (ZH)**: 纳米ESG：从新闻文章中提取企业 Sustainability 信息 

**Authors**: Fabian Billert, Stefan Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2412.15093)  

**Abstract**: Determining the sustainability impact of companies is a highly complex subject which has garnered more and more attention over the past few years. Today, investors largely rely on sustainability-ratings from established rating-providers in order to analyze how responsibly a company acts. However, those ratings have recently been criticized for being hard to understand and nearly impossible to reproduce.
An independent way to find out about the sustainability practices of companies lies in the rich landscape of news article data. In this paper, we explore a different approach to identify key opportunities and challenges of companies in the sustainability domain. We present a novel dataset of more than 840,000 news articles which were gathered for major German companies between January 2023 and September 2024. By applying a mixture of Natural Language Processing techniques, we first identify relevant articles, before summarizing them and extracting their sustainability-related sentiment and aspect using Large Language Models (LLMs). Furthermore, we conduct an evaluation of the obtained data and determine that the LLM-produced answers are accurate. We release both datasets at this https URL. 

**Abstract (ZH)**: 评估企业的可持续性影响是一个高度复杂的问题，近年来引起了越来越多的关注。今天，投资者主要依赖于来自成熟评级机构的可持续性评级来分析企业的行为是否符合社会责任。然而，这些评级最近因难以理解且几乎无法复制而受到批评。

一种独立的方法是利用丰富的新闻文章数据来了解企业的可持续性实践。在本文中，我们探索了一种不同的方法来识别企业在可持续性领域的关键机遇和挑战。我们提供了一个包含超过840,000篇新闻文章的新数据集，这些文章是在2023年1月至2024年9月期间为德国主要企业收集的。通过应用多种自然语言处理技术，我们首先识别出相关文章，然后对这些文章进行总结，并使用大型语言模型（LLMs）提取与可持续性相关的观点和方面。此外，我们对获得的数据进行了评估，并确定LLM生成的答案是准确的。我们在此共享这两个数据集：[此链接]。 

---
