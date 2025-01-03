# Large Language Models for Mental Health Diagnostic Assessments: Exploring The Potential of Large Language Models for Assisting with Mental Health Diagnostic Assessments -- The Depression and Anxiety Case 

**Title (ZH)**: 大型语言模型在心理健康诊断评估中的应用：探索大型语言模型在辅助心理健康诊断评估中的潜力——以抑郁和焦虑为例 

**Authors**: Kaushik Roy, Harshul Surana, Darssan Eswaramoorthi, Yuxin Zi, Vedant Palit, Ritvik Garimella, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2501.01305)  

**Abstract**: Large language models (LLMs) are increasingly attracting the attention of healthcare professionals for their potential to assist in diagnostic assessments, which could alleviate the strain on the healthcare system caused by a high patient load and a shortage of providers. For LLMs to be effective in supporting diagnostic assessments, it is essential that they closely replicate the standard diagnostic procedures used by clinicians. In this paper, we specifically examine the diagnostic assessment processes described in the Patient Health Questionnaire-9 (PHQ-9) for major depressive disorder (MDD) and the Generalized Anxiety Disorder-7 (GAD-7) questionnaire for generalized anxiety disorder (GAD). We investigate various prompting and fine-tuning techniques to guide both proprietary and open-source LLMs in adhering to these processes, and we evaluate the agreement between LLM-generated diagnostic outcomes and expert-validated ground truth. For fine-tuning, we utilize the Mentalllama and Llama models, while for prompting, we experiment with proprietary models like GPT-3.5 and GPT-4o, as well as open-source models such as llama-3.1-8b and mixtral-8x7b. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益受到医疗专业人员的关注，它们有可能在诊断评估中提供帮助，从而缓解由于患者数量过多和医疗提供者短缺而导致的医疗系统压力。为了使LLMs在支持诊断评估方面有效，它们必须紧密复制临床医生使用的标准诊断程序。本文具体研究了用于重度抑郁症（MDD）的患者健康问卷-9（PHQ-9）和用于广泛性焦虑障碍（GAD）的一般化焦虑问卷-7（GAD-7）中的诊断评估过程。我们探讨了各种提示和微调技术，以引导自有的和开源的LLMs遵守这些过程，并且评估了LLM生成的诊断结果与专家验证的黄金标准之间的一致性。在微调方面，我们使用了Mentalllama和Llama模型，而在提示方面，我们尝试了诸如GPT-3.5和GPT-4o等自有模型，以及诸如llama-3.1-8b和mixtral-8x7b等开源模型。 

---
# MDSF: Context-Aware Multi-Dimensional Data Storytelling Framework based on Large language Model 

**Title (ZH)**: MDSF：基于大型语言模型的上下文感知多维数据叙事框架

这个标题翻译符合学术规范，保留了原文的含义和结构。其中，“MDSF”被译为“MDSF”，保持了原文的简称形式。“基于大型语言模型”准确地翻译了“based on Large language Model”，确保术语的专业性和准确性。 

**Authors**: Chengze Zhang, Changshan Li, Shiyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01014)  

**Abstract**: The exponential growth of data and advancements in big data technologies have created a demand for more efficient and automated approaches to data analysis and storytelling. However, automated data analysis systems still face challenges in leveraging large language models (LLMs) for data insight discovery, augmented analysis, and data storytelling. This paper introduces the Multidimensional Data Storytelling Framework (MDSF) based on large language models for automated insight generation and context-aware storytelling. The framework incorporates advanced preprocessing techniques, augmented analysis algorithms, and a unique scoring mechanism to identify and prioritize actionable insights. The use of fine-tuned LLMs enhances contextual understanding and generates narratives with minimal manual intervention. The architecture also includes an agent-based mechanism for real-time storytelling continuation control. Key findings reveal that MDSF outperforms existing methods across various datasets in terms of insight ranking accuracy, descriptive quality, and narrative coherence. The experimental evaluation demonstrates MDSF's ability to automate complex analytical tasks, reduce interpretive biases, and improve user satisfaction. User studies further underscore its practical utility in enhancing content structure, conclusion extraction, and richness of detail. 

**Abstract (ZH)**: 大数据的指数级增长和大数据技术的进步促使对更高效和自动化的数据分析与叙述方法的需求。然而，自动数据分析系统在利用大型语言模型（LLM）进行数据洞察发现、增强分析和数据叙述方面仍面临挑战。本文介绍了一种基于大型语言模型的多维数据叙述框架（MDSF），该框架用于自动化洞察生成和上下文感知叙述。该框架整合了先进的预处理技术、增强分析算法以及独特的评分机制来识别和优先处理可操作的洞察。微调的LLM增强了对上下文的理解，并生成了需要最少手动干预的故事叙述。该架构还包括一种基于代理的机制，用于实时叙述延续控制。关键发现表明，MDSF在各类数据集上的洞察排名准确性、描述质量和叙述连贯性方面优于现有方法。实验评估展示了MDSF自动化复杂分析任务、减少解释偏见和提高用户满意度的能力。用户研究进一步突显了其在增强内容结构、结论提取和细节 richness 方面的实用价值。 

---
# Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation 

**Title (ZH)**: 大规模语言模型是同时生成的读写政策制定者 

**Authors**: Shoutao Guo, Shaolei Zhang, Zhengrui Ma, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.00868)  

**Abstract**: Simultaneous generation models write generation results while reading streaming inputs, necessitating a policy-maker to determine the appropriate output timing. Existing simultaneous generation methods generally adopt the traditional encoder-decoder architecture and learn the generation and policy-making capabilities through complex dynamic programming techniques. Although LLMs excel at text generation, they face challenges in taking on the role of policy-makers through traditional training methods, limiting their exploration in simultaneous generation. To overcome these limitations, we propose a novel LLM-driven Simultaneous Generation (LSG) framework, which allows the off-the-shelf LLM to decide the generation timing and produce output concurrently. Specifically, LSG selects the generation policy that minimizes latency as the baseline policy. Referring to the baseline policy, LSG enables the LLM to devise an improved generation policy that better balances latency and generation quality, and writes generation results accordingly. Experiments on simultaneous translation and streaming automatic speech recognition tasks show that our method can achieve state-of-the-art performance utilizing the open-source LLMs and demonstrate practicality in real-world scenarios. 

**Abstract (ZH)**: 同时生成模型在读取流式输入的同时生成结果，需要决策者确定适当的输出时机。现有的同时生成方法通常采用传统的编码器-解码器架构，并通过复杂的动态规划技术来学习生成和决策制定的能力。尽管大型语言模型（LLMs）在文本生成方面表现出色，但在传统的训练方法下承担决策者的角色面临挑战，限制了它们在同时生成方面的探索。为了克服这些限制，我们提出了一种新颖的LLM驱动的即刻生成（LSG）框架，该框架允许即用型LLM决定生成时机并同时生成结果。具体来说，LSG 选择减少延迟的策略作为基准策略。参照基准策略，LSG 使LLM能够制定一个能更好地平衡延迟和生成质量的改进策略，并据此生成结果。在同时翻译和流式自动语音识别任务上的实验表明，我们的方法可以利用开源的LLMs达到最先进的性能，并在实际场景中具有实用价值。 

---
# LLM+AL: Bridging Large Language Models and Action Languages for Complex Reasoning about Actions 

**Title (ZH)**: 基于LLM和AL的桥梁：大语言模型与操作语言在复杂动作推理中的融合 

**Authors**: Adam Ishay, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.00830)  

**Abstract**: Large Language Models (LLMs) have made significant strides in various intelligent tasks but still struggle with complex action reasoning tasks that require systematic search. To address this limitation, we propose a method that bridges the natural language understanding capabilities of LLMs with the symbolic reasoning strengths of action languages. Our approach, termed "LLM+AL," leverages the LLM's strengths in semantic parsing and commonsense knowledge generation alongside the action language's proficiency in automated reasoning based on encoded knowledge. We compare LLM+AL against state-of-the-art LLMs, including ChatGPT-4, Claude 3 Opus, Gemini Ultra 1.0, and o1-preview, using benchmarks for complex reasoning about actions. Our findings indicate that, although all methods exhibit errors, LLM+AL, with relatively minimal human corrections, consistently leads to correct answers, whereas standalone LLMs fail to improve even with human feedback. LLM+AL also contributes to automated generation of action languages. 

**Abstract (ZH)**: 大语言模型（LLMs）在各种智能任务中取得了显著进展，但在涉及系统搜索的复杂动作推理任务中仍然存在局限性。为了解决这一限制，我们提出了一种方法，将LLMs的自然语言理解能力与动作语言的符号推理优势相结合。我们的方法称为“LLM+AL”，它利用了LLMs在语义解析和常识知识生成方面的优势，同时利用了动作语言基于编码知识进行自动推理的能力。我们使用针对复杂动作推理的基准测试来比较LLM+AL与最新的LLMs，包括ChatGPT-4、Claude 3 Opus、Gemini Ultra 1.0和o1-preview。研究结果表明，尽管所有方法都存在错误，但在最小的人工修正下，LLM+AL始终能够得到正确的答案，而单独的LLMs即使在获得人类反馈后也无法改进。此外，LLM+AL也有助于自动生成动作语言。 

---
# Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection agents 

**Title (ZH)**: 使用多路径协作反应与反思代理增强生成式预训练语言模型的推理能力 

**Authors**: Chengbo He, Bochao Zou, Xin Li, Jiansheng Chen, Junliang Xing, Huimin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.00430)  

**Abstract**: Agents have demonstrated their potential in scientific reasoning tasks through large language models. However, they often face challenges such as insufficient accuracy and degeneration of thought when handling complex reasoning tasks, which impede their performance. To overcome these issues, we propose the Reactive and Reflection agents with Multi-Path Reasoning (RR-MP) Framework, aimed at enhancing the reasoning capabilities of LLMs. Our approach improves scientific reasoning accuracy by employing a multi-path reasoning mechanism where each path consists of a reactive agent and a reflection agent that collaborate to prevent degeneration of thought inherent in single-agent reliance. Additionally, the RR-MP framework does not require additional training; it utilizes multiple dialogue instances for each reasoning path and a separate summarizer to consolidate insights from all paths. This design integrates diverse perspectives and strengthens reasoning across each path. We conducted zero-shot and few-shot evaluations on tasks involving moral scenarios, college-level physics, and mathematics. Experimental results demonstrate that our method outperforms baseline approaches, highlighting the effectiveness and advantages of the RR-MP framework in managing complex scientific reasoning tasks. 

**Abstract (ZH)**: 大型语言模型已经在科学推理任务中展示了其潜力，但当处理复杂推理任务时，代理常常面临准确性不足和思维退化等挑战，这限制了它们的表现。为了解决这些问题，我们提出了一种反应性和反思性多路径推理框架（RR-MP框架），旨在增强大模型的推理能力。我们的方法通过采用多路径推理机制来提高科学推理的准确性，其中每条路径由一个反应性代理和一个反思性代理组成，二者协作以防止单代理依赖性所固有的思维退化。此外，RR-MP框架不需要额外的训练，它利用每条推理路径中的多个对话实例以及一个单独的总结器来整合所有路径所获得的洞见。这种设计整合了多角度的观点，并增强了每条路径的推理能力。我们在涉及道德场景、大学物理和数学的任务上进行了零样本和少样本评估。实验结果表明，我们的方法优于基线方法，突显了RR-MP框架在管理复杂科学推理任务方面的有效性和优势。 

---
# MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation 

**Title (ZH)**: MAIN-RAG：多代理过滤检索增强生成 

**Authors**: Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2501.00332)  

**Abstract**: Large Language Models (LLMs) are becoming essential tools for various natural language processing tasks but often suffer from generating outdated or incorrect information. Retrieval-Augmented Generation (RAG) addresses this issue by incorporating external, real-time information retrieval to ground LLM responses. However, the existing RAG systems frequently struggle with the quality of retrieval documents, as irrelevant or noisy documents degrade performance, increase computational overhead, and undermine response reliability. To tackle this problem, we propose Multi-Agent Filtering Retrieval-Augmented Generation (MAIN-RAG), a training-free RAG framework that leverages multiple LLM agents to collaboratively filter and score retrieved documents. Specifically, MAIN-RAG introduces an adaptive filtering mechanism that dynamically adjusts the relevance filtering threshold based on score distributions, effectively minimizing noise while maintaining high recall of relevant documents. The proposed approach leverages inter-agent consensus to ensure robust document selection without requiring additional training data or fine-tuning. Experimental results across four QA benchmarks demonstrate that MAIN-RAG consistently outperforms traditional RAG approaches, achieving a 2-11% improvement in answer accuracy while reducing the number of irrelevant retrieved documents. Quantitative analysis further reveals that our approach achieves superior response consistency and answer accuracy over baseline methods, offering a competitive and practical alternative to training-based solutions. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为各种自然语言处理任务的重要工具，但常常会生成过时或不准确的信息。检索增强生成（RAG）通过结合外部的实时信息检索来确保LLM回答的准确性和相关性，解决了这一问题。然而，现有的RAG系统在检索文档的质量方面常常面临挑战，无关或噪声文档会降低性能、增加计算开销并削弱回答的可靠性。为了解决这一问题，我们提出了一种名为Multi-Agent Filtering Retrieval-Augmented Generation（MAIN-RAG）的无需训练的RAG框架，该框架利用多个LLM代理协作筛选和评估检索到的文档。具体来说，MAIN-RAG引入了一种自适应筛选机制，该机制会根据评分分布动态调整相关性筛选阈值，有效减少了噪声并保持了相关文档的高召回率。该方法利用代理间的共识来确保稳健的文档选择，而无需额外的训练数据或微调。在四个问答基准测试中的实验结果表明，MAIN-RAG 在回答准确性和减少无关检索文档数量方面始终优于传统的RAG方法。定量分析进一步表明，我们的方法在响应一致性和回答准确性方面优于基准方法，提供了基于训练的解决方案的有竞争力且实用的替代方案。 

---
# Harnessing Multi-Agent LLMs for Complex Engineering Problem-Solving: A Framework for Senior Design Projects 

**Title (ZH)**: 利用多代理大型语言模型解决复杂工程问题：面向高级设计项目的框架 

**Authors**: Abdullah Mushtaq, Muhammad Rafay Naeem, Ibrahim Ghaznavi, Muhammad Imran Taj, Imran Hashmi, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2501.01205)  

**Abstract**: Multi-Agent Large Language Models (LLMs) are gaining significant attention for their ability to harness collective intelligence in complex problem-solving, decision-making, and planning tasks. This aligns with the concept of the wisdom of crowds, where diverse agents contribute collectively to generating effective solutions, making it particularly suitable for educational settings. Senior design projects, also known as capstone or final year projects, are pivotal in engineering education as they integrate theoretical knowledge with practical application, fostering critical thinking, teamwork, and real-world problem-solving skills. In this paper, we explore the use of Multi-Agent LLMs in supporting these senior design projects undertaken by engineering students, which often involve multidisciplinary considerations and conflicting objectives, such as optimizing technical performance while addressing ethical, social, and environmental concerns. We propose a framework where distinct LLM agents represent different expert perspectives, such as problem formulation agents, system complexity agents, societal and ethical agents, or project managers, thus facilitating a holistic problem-solving approach. This implementation leverages standard multi-agent system (MAS) concepts such as coordination, cooperation, and negotiation, incorporating prompt engineering to develop diverse personas for each agent. These agents engage in rich, collaborative dialogues to simulate human engineering teams, guided by principles from swarm AI to efficiently balance individual contributions towards a unified solution. We adapt these techniques to create a collaboration structure for LLM agents, encouraging interdisciplinary reasoning and negotiation similar to real-world senior design projects. To assess the efficacy of this framework, we collected six proposals of engineering and computer science of... 

**Abstract (ZH)**: 多智能体大型语言模型（Multi-Agent Large Language Models, MALLMs）因其在复杂问题解决、决策制定和规划任务中的集体智能应用能力而备受关注。这与“群体智慧”（wisdom of crowds）的概念相吻合，即通过多样化智能体的集体贡献来生成有效的解决方案，因此特别适用于教育环境。高级设计项目，通常被称为毕业设计或大四项目，在工程教育中具有重要意义，因为它们能够将理论知识与实践应用相结合，培养批判性思维、团队合作和解决实际问题的能力。本文探讨了MALLMs在支持工程学生进行高级设计项目中的应用，这些项目往往涉及多学科考虑和相互冲突的目标，如优化技术性能同时解决道德、社会和环境问题。我们提出了一种框架，其中不同的LLM智能体代表不同的专家视角，如问题界定智能体、系统复杂性智能体、社会伦理智能体或项目管理者，从而促进一种全面的问题解决方法。该实施利用了标准的多智能体系统（MAS）概念，如协调、合作和协商，并结合指令工程来为每个智能体开发不同的个性。这些智能体通过丰富的协作对话来模拟人类工程团队，在群体AI的原则指导下，高效地平衡个人贡献以达成统一的解决方案。我们将这些技术应用于构建LLM智能体之间的协作结构，鼓励跨学科的推理和协商，类似于现实世界中的高级设计项目。为了评估该框架的有效性，我们收集了六个工程和计算机科学领域的工程设计提案…… 

---
# Speech Recognition With LLMs Adapted to Disordered Speech Using Reinforcement Learning 

**Title (ZH)**: 使用强化学习适应非规范语音的大型语言模型的语音识别 

**Authors**: Chirag Nagpal, Subhashini Venugopalan, Jimmy Tobin, Marilyn Ladewig, Katherine Heller, Katrin Tomanek  

**Link**: [PDF](https://arxiv.org/pdf/2501.00039)  

**Abstract**: We introduce a large language model (LLM) capable of processing speech inputs and show that tuning it further with reinforcement learning on human preference (RLHF) enables it to adapt better to disordered speech than traditional fine-tuning. Our method replaces low-frequency text tokens in an LLM's vocabulary with audio tokens and enables the model to recognize speech by fine-tuning it on speech with transcripts. We then use RL with rewards based on syntactic and semantic accuracy measures generalizing the LLM further to recognize disordered speech. While the resulting LLM does not outperform existing systems for speech recognition, we find that tuning with reinforcement learning using custom rewards leads to substantially better performance than supervised fine-tuning of the language model, specifically when adapting to speech in a different setting. This presents a compelling alternative tuning strategy for speech recognition using large language models. 

**Abstract (ZH)**: 我们介绍了一种大型语言模型（LLM），该模型能够处理语音输入，并表明通过强化学习（RLHF）进一步调整该模型使其能够更好地适应乱序语音，而传统的微调效果则较差。我们的方法用音频令牌替换LLM词汇表中低频的文字令牌，从而使模型能够在带有转录的语音数据上进行微调，进而识别语音。随后，我们使用基于语法和语义准确性的奖励进行RL，进一步推广LLM以识别乱序语音。尽管最终得到的LLM在语音识别方面并未胜过现有的系统，但我们发现使用自定义奖励进行强化学习的微调，与监督下的语言模型微调相比，其性能显著提升，尤其是在不同场景下的语音适应方面。这为使用大型语言模型进行语音识别提供了一种有吸引力的替代调优策略。 

---
# A3: Android Agent Arena for Mobile GUI Agents 

**Title (ZH)**: A3: Android代理竞技场——用于移动GUI代理的研究 

**Authors**: Yuxiang Chai, Hanhao Li, Jiayu Zhang, Liang Liu, Guozhi Wang, Shuai Ren, Siyuan Huang, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.01149)  

**Abstract**: AI agents have become increasingly prevalent in recent years, driven by significant advancements in the field of large language models (LLMs). Mobile GUI agents, a subset of AI agents, are designed to autonomously perform tasks on mobile devices. While numerous studies have introduced agents, datasets, and benchmarks to advance mobile GUI agent research, many existing datasets focus on static frame evaluations and fail to provide a comprehensive platform for assessing performance on real-world, in-the-wild tasks. To address this gap, we present Android Agent Arena (A3), a novel evaluation platform. Unlike existing in-the-wild systems, A3 offers: (1) meaningful and practical tasks, such as real-time online information retrieval and operational instructions; (2) a larger, more flexible action space, enabling compatibility with agents trained on any dataset; and (3) automated business-level LLM-based evaluation process. A3 includes 21 widely used general third-party apps and 201 tasks representative of common user scenarios, providing a robust foundation for evaluating mobile GUI agents in real-world situations and a new autonomous evaluation process for less human labor and coding expertise. The project is available at \url{this https URL}. 

**Abstract (ZH)**: 近年来，随着大型语言模型（LLMs）领域的显著进步，AI代理的使用越来越普遍。移动GUI代理是AI代理的一个子类，旨在自主在移动设备上执行任务。尽管许多研究已经介绍了代理、数据集和基准来推动移动GUI代理研究，但现有许多数据集主要关注静态帧评估，未能提供一个全面的平台来评估在真实世界、自然环境中的表现。为了弥补这一差距，我们提出Android Agent Arena (A3)，一个新型的评估平台。与现有的自然环境系统不同，A3提供：（1）有意义且实用的任务，如实时在线信息检索和操作指令；（2）更大的、更具灵活性的动作空间，使任何数据集训练的代理都能兼容；（3）自动化的企业级基于LLM的评估流程。A3包括21个广泛使用的第三方通用应用程序和201个代表常见用户场景的任务，为评估移动GUI代理提供了坚实的基础，并提供了一种新的自动化评估过程，以减少人力和编程专业知识的投入。该项目可以在 \url{this https URL} 获取。 

---
# Beyond Text: Implementing Multimodal Large Language Model-Powered Multi-Agent Systems Using a No-Code Platform 

**Title (ZH)**: 超越文本：使用无代码平台实现基于大型语言模型的多模态多代理系统 

**Authors**: Cheonsu Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2501.00750)  

**Abstract**: This study proposes the design and implementation of a multimodal LLM-based Multi-Agent System (MAS) leveraging a No-Code platform to address the practical constraints and significant entry barriers associated with AI adoption in enterprises. Advanced AI technologies, such as Large Language Models (LLMs), often pose challenges due to their technical complexity and high implementation costs, making them difficult for many organizations to adopt. To overcome these limitations, this research develops a No-Code-based Multi-Agent System designed to enable users without programming knowledge to easily build and manage AI systems. The study examines various use cases to validate the applicability of AI in business processes, including code generation from image-based notes, Advanced RAG-based question-answering systems, text-based image generation, and video generation using images and prompts. These systems lower the barriers to AI adoption, empowering not only professional developers but also general users to harness AI for significantly improved productivity and efficiency. By demonstrating the scalability and accessibility of No-Code platforms, this study advances the democratization of AI technologies within enterprises and validates the practical applicability of Multi-Agent Systems, ultimately contributing to the widespread adoption of AI across various industries. 

**Abstract (ZH)**: 本研究提出了一种利用无代码平台设计和实现基于多模态大型语言模型（LLM）的多代理系统（MAS），以解决企业在采用人工智能时所面临的实际限制和显著进入壁垒。先进的AI技术，如大型语言模型（LLMs），往往由于其技术复杂性和高昂的实施成本而带来挑战，使得许多组织难以采用。为克服这些限制，本研究开发了一种基于无代码的多代理系统，旨在使不具备编程知识的用户能够轻松构建和管理AI系统。研究探讨了多种应用场景来验证AI在业务流程中的适用性，包括基于图像笔记的代码生成、基于高级检索与生成（RAG）的问题回答系统、基于文本的图像生成，以及使用图像和提示生成视频系统。这些系统降低了AI的采用门槛，不仅赋能专业的开发人员，还赋能普通用户利用AI来显著提高生产力和效率。通过展示无代码平台的扩展性和可访问性，本研究推动了企业内部AI技术的民主化，并验证了多代理系统的实际适用性，最终促进了AI在各行业的广泛采用。 

---
# Autonomous Alignment with Human Value on Altruism through Considerate Self-imagination and Theory of Mind 

**Title (ZH)**: 通过体贴的自我想象和理论思维实现基于利他主义的人工智能自主对齐与人类价值一致性的研究 

**Authors**: Haibo Tong, Enmeng Lum, Yinqian Sun, Zhengqiang Han, Chao Liu, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2501.00320)  

**Abstract**: With the widespread application of Artificial Intelligence (AI) in human society, enabling AI to autonomously align with human values has become a pressing issue to ensure its sustainable development and benefit to humanity. One of the most important aspects of aligning with human values is the necessity for agents to autonomously make altruistic, safe, and ethical decisions, considering and caring for human well-being. Current AI extremely pursues absolute superiority in certain tasks, remaining indifferent to the surrounding environment and other agents, which has led to numerous safety risks. Altruistic behavior in human society originates from humans' capacity for empathizing others, known as Theory of Mind (ToM), combined with predictive imaginative interactions before taking action to produce thoughtful and altruistic behaviors. Inspired by this, we are committed to endow agents with considerate self-imagination and ToM capabilities, driving them through implicit intrinsic motivations to autonomously align with human altruistic values. By integrating ToM within the imaginative space, agents keep an eye on the well-being of other agents in real time, proactively anticipate potential risks to themselves and others, and make thoughtful altruistic decisions that balance negative effects on the environment. The ancient Chinese story of Sima Guang Smashes the Vat illustrates the moral behavior of the young Sima Guang smashed a vat to save a child who had accidentally fallen into it, which is an excellent reference scenario for this paper. We design an experimental scenario similar to Sima Guang Smashes the Vat and its variants with different complexities, which reflects the trade-offs and comprehensive considerations between self-goals, altruistic rescue, and avoiding negative side effects. 

**Abstract (ZH)**: 随着人工智能（AI）在人类社会中的广泛应用，使AI能够自主与人类价值观保持一致已成为确保其可持续发展和造福人类的关键问题。与人类价值观保持一致的一个最关键方面是要求智能体能够自主作出利他、安全和道德的决策，同时考虑并关心人类的福祉。目前的AI在某些任务上追求绝对的优越性，对周围环境和其他智能体漠不关心，这导致了大量安全风险。人类社会中的利他行为源于其同理心能力，即心理理论（Theory of Mind, ToM），并且在采取行动前通过预测性的想象互动产生有思考和利他性的行为。基于此，我们致力于赋予智能体考虑周到的自我想象和ToM能力，通过隐含的内在动机引导它们自主地与人类的利他价值观保持一致。通过在想象空间中结合ToM，智能体能够实时关注其他智能体的福祉，主动预测潜在的风险，并作出权衡环境负面影响的有思考的利他性决策。中国古代故事《司马光砸缸》中的司马光为了救掉进缸里的孩子砸破了缸，这为本文提供了优秀的参考场景。我们设计了一个类似于《司马光砸缸》及其复杂度不同的变体的实验场景，反映了自我目标、救援利他以及避免负面影响之间的权衡与综合考虑。 

---
# PIMAEX: Multi-Agent Exploration through Peer Incentivization 

**Title (ZH)**: PIMAEX：通过同伴激励实现多agent探索 

**Authors**: Michael Kölle, Johannes Tochtermann, Julian Schönberger, Gerhard Stenzel, Philipp Altmann, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2501.01266)  

**Abstract**: While exploration in single-agent reinforcement learning has been studied extensively in recent years, considerably less work has focused on its counterpart in multi-agent reinforcement learning. To address this issue, this work proposes a peer-incentivized reward function inspired by previous research on intrinsic curiosity and influence-based rewards. The \textit{PIMAEX} reward, short for Peer-Incentivized Multi-Agent Exploration, aims to improve exploration in the multi-agent setting by encouraging agents to exert influence over each other to increase the likelihood of encountering novel states. We evaluate the \textit{PIMAEX} reward in conjunction with \textit{PIMAEX-Communication}, a multi-agent training algorithm that employs a communication channel for agents to influence one another. The evaluation is conducted in the \textit{Consume/Explore} environment, a partially observable environment with deceptive rewards, specifically designed to challenge the exploration vs.\ exploitation dilemma and the credit-assignment problem. The results empirically demonstrate that agents using the \textit{PIMAEX} reward with \textit{PIMAEX-Communication} outperform those that do not. 

**Abstract (ZH)**: 近年来，单智能体强化学习中的探索已经得到了广泛研究，但在多智能体强化学习中的探索对应方面，相关研究工作相对较少。为解决这一问题，本研究提出一种受内在好奇性和基于影响奖励研究成果启发的同伴激励奖励函数。该奖励函数名为同伴激励化的多智能体探索（Peer-Incentivized Multi-Agent Exploration, \textit{PIMAEX}），旨在通过鼓励智能体相互施加影响以增加遇到新颖状态的可能性，从而改善多智能体环境中的探索。我们评估了\textit{PIMAEX}奖励与\textit{PIMAEX-Communication}多智能体训练算法的结合效果，其中\textit{PIMAEX-Communication}算法使用通信通道使智能体相互影响。评估是在\textit{Consume/Explore}环境中进行的，这一部分可观察且具有欺骗性奖励的环境，特别设计用于挑战探索与利用的矛盾以及归因问题。实验结果表明，使用\textit{PIMAEX}奖励和\textit{PIMAEX-Communication}算法的智能体表现优于未使用这些算法的智能体。 

---
# Symmetries-enhanced Multi-Agent Reinforcement Learning 

**Title (ZH)**: 增强对称性多智能体强化学习 

**Authors**: Nikolaos Bousias, Stefanos Pertigkiozoglou, Kostas Daniilidis, George Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2501.01136)  

**Abstract**: Multi-agent reinforcement learning has emerged as a powerful framework for enabling agents to learn complex, coordinated behaviors but faces persistent challenges regarding its generalization, scalability and sample efficiency. Recent advancements have sought to alleviate those issues by embedding intrinsic symmetries of the systems in the policy. Yet, most dynamical systems exhibit little to no symmetries to exploit. This paper presents a novel framework for embedding extrinsic symmetries in multi-agent system dynamics that enables the use of symmetry-enhanced methods to address systems with insufficient intrinsic symmetries, expanding the scope of equivariant learning to a wide variety of MARL problems. Central to our framework is the Group Equivariant Graphormer, a group-modular architecture specifically designed for distributed swarming tasks. Extensive experiments on a swarm of symmetry-breaking quadrotors validate the effectiveness of our approach, showcasing its potential for improved generalization and zero-shot scalability. Our method achieves significant reductions in collision rates and enhances task success rates across a diverse range of scenarios and varying swarm sizes. 

**Abstract (ZH)**: 多智能体强化学习作为一种使智能体能够学习复杂协调行为的强大框架，已经取得了显著进展，但它在泛化能力、可扩展性和样本效率方面仍然面临诸多挑战。近年来的研究努力通过在策略中嵌入系统固有的对称性来缓解这些问题，但大多数动态系统却没有可利用的显著对称性。本文提出了一种新颖的框架，用于在多智能体系统动力学中嵌入外在对称性，从而使增强对称性方法能够处理缺乏固有对称性的系统，从而将守恒学习的应用范围扩展到各种多智能体强化学习（MARL）问题。该框架的关键在于组守恒图卷积机(Group Equivariant Graphormer)，这是一种专门针对分布式群集任务的模块化架构。通过 Swarm 中的对称性打破四旋翼无人机实验，本文验证了该方法的有效性，展示了其在提高泛化能力和零样本可扩展性方面的潜在优势。该方法在多种场景和不同的群集规模下显著降低了碰撞率，提高了任务成功率。 

---
# $\beta$-DQN: Improving Deep Q-Learning By Evolving the Behavior 

**Title (ZH)**: $\beta$-DQN：通过演化行为改进深度Q学习 

**Authors**: Hongming Zhang, Fengshuo Bai, Chenjun Xiao, Chao Gao, Bo Xu, Martin Müller  

**Link**: [PDF](https://arxiv.org/pdf/2501.00913)  

**Abstract**: While many sophisticated exploration methods have been proposed, their lack of generality and high computational cost often lead researchers to favor simpler methods like $\epsilon$-greedy. Motivated by this, we introduce $\beta$-DQN, a simple and efficient exploration method that augments the standard DQN with a behavior function $\beta$. This function estimates the probability that each action has been taken at each state. By leveraging $\beta$, we generate a population of diverse policies that balance exploration between state-action coverage and overestimation bias correction. An adaptive meta-controller is designed to select an effective policy for each episode, enabling flexible and explainable exploration. $\beta$-DQN is straightforward to implement and adds minimal computational overhead to the standard DQN. Experiments on both simple and challenging exploration domains show that $\beta$-DQN outperforms existing baseline methods across a wide range of tasks, providing an effective solution for improving exploration in deep reinforcement learning. 

**Abstract (ZH)**: 尽管已经提出了许多复杂的探索方法，但它们的通用性较差且计算成本较高，这往往使研究者更倾向于使用简单的策略如$\epsilon$-贪心。为此，我们提出了$\beta$-DQN，这是一种简单而高效的探索方法，通过将行为函数$\beta$添加到标准的DQN中来实现。该函数估计在每个状态下每种动作被采取的概率。通过利用$\beta$，我们生成了一组多样化的策略，这些策略在状态-动作覆盖和过度估计偏差校正之间实现了探索的平衡。设计了一个适应性的元控制器来为每个 episode 选择一个有效的策略，从而实现灵活且可解释的探索。$\beta$-DQN 实现起来非常简便，并且对标准 DQN 的计算开销几乎没有影响。实验表明，$\beta$-DQN 在不同难度的探索任务中均能优于现有的基线方法，为改进深度强化学习中的探索提供了有效的解决方案。 

---
# Large Language Model Based Multi-Agent System Augmented Complex Event Processing Pipeline for Internet of Multimedia Things 

**Title (ZH)**: 基于大型语言模型的多代理系统增强复杂事件处理管道在多媒体事物互联网中的应用 

**Authors**: Talha Zeeshan, Abhishek Kumar, Susanna Pirttikangas, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2501.00906)  

**Abstract**: This paper presents the development and evaluation of a Large Language Model (LLM), also known as foundation models, based multi-agent system framework for complex event processing (CEP) with a focus on video query processing use cases. The primary goal is to create a proof-of-concept (POC) that integrates state-of-the-art LLM orchestration frameworks with publish/subscribe (pub/sub) tools to address the integration of LLMs with current CEP systems. Utilizing the Autogen framework in conjunction with Kafka message brokers, the system demonstrates an autonomous CEP pipeline capable of handling complex workflows. Extensive experiments evaluate the system's performance across varying configurations, complexities, and video resolutions, revealing the trade-offs between functionality and latency. The results show that while higher agent count and video complexities increase latency, the system maintains high consistency in narrative coherence. This research builds upon and contributes to, existing novel approaches to distributed AI systems, offering detailed insights into integrating such systems into existing infrastructures. 

**Abstract (ZH)**: 本文介绍了一种基于大规模语言模型（LLM）或称为基础模型的多Agent系统框架，该框架旨在处理复杂事件处理（CEP）中的视频查询应用场景。主要目标是创建一个概念验证（Proof of Concept, POC），将最先进的LLM编排框架与发布/订阅（Pub/Sub）工具集成，以解决将LLM与当前CEP系统集成的问题。通过结合使用Autogen框架和Kafka消息代理，系统展示了能够处理复杂工作流程的自主CEP管道。进行了广泛实验，评估了系统在不同配置、复杂性和视频分辨率下的性能，揭示了功能性和延迟之间的权衡。结果表明，尽管更高的Agent数量和视频复杂性增加了延迟，但系统在叙事连贯性方面保持了高度一致性。该研究建立在并贡献于现有分布式AI系统的新型方法之上，提供了将此类系统集成到现有基础设施中的详细见解。 

---
# LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management 

**Title (ZH)**: 基于LLM的强大多代理系统自动加密货币投资组合管理 

**Authors**: Yichen Luo, Yebo Feng, Jiahua Xu, Paolo Tasca, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00826)  

**Abstract**: Cryptocurrency investment is inherently difficult due to its shorter history compared to traditional assets, the need to integrate vast amounts of data from various modalities, and the requirement for complex reasoning. While deep learning approaches have been applied to address these challenges, their black-box nature raises concerns about trust and explainability. Recently, large language models (LLMs) have shown promise in financial applications due to their ability to understand multi-modal data and generate explainable decisions. However, single LLM faces limitations in complex, comprehensive tasks such as asset investment. These limitations are even more pronounced in cryptocurrency investment, where LLMs have less domain-specific knowledge in their training corpora.
To overcome these challenges, we propose an explainable, multi-modal, multi-agent framework for cryptocurrency investment. Our framework uses specialized agents that collaborate within and across teams to handle subtasks such as data analysis, literature integration, and investment decision-making for the top 30 cryptocurrencies by market capitalization. The expert training module fine-tunes agents using multi-modal historical data and professional investment literature, while the multi-agent investment module employs real-time data to make informed cryptocurrency investment decisions. Unique intrateam and interteam collaboration mechanisms enhance prediction accuracy by adjusting final predictions based on confidence levels within agent teams and facilitating information sharing between teams. Empirical evaluation using data from November 2023 to September 2024 demonstrates that our framework outperforms single-agent models and market benchmarks in classification, asset pricing, portfolio, and explainability performance. 

**Abstract (ZH)**: 与传统资产相比，数字货币的投资因其较短的历史、需要整合多种数据模态以及复杂推理解释的要求而固有地具有挑战性。尽管深度学习方法已被应用于解决这些挑战，但其黑箱特性引发了关于信任与解释性的担忧。最近，大规模语言模型（LLMs）在金融应用中显示出潜力，这得益于它们能够理解多模态数据并生成可解释的决策。然而，单一的大规模语言模型在处理复杂且综合性的任务，如资产投资时存在局限性。在加密货币投资中，由于训练语料库中的领域特定知识较少，这种局限性更为明显。

为了克服这些挑战，我们提出了一种可用于加密货币投资的可解释、多模态、多智能体框架。该框架中使用专门的智能体，在团队内部及跨团队进行协作，以处理数据分析、文献整合和市值前30位的加密货币的投资决策等子任务。专家训练模块通过多模态历史数据和专业的投资文献对智能体进行微调，而多智能体投资模块则利用实时数据进行加密货币投资决策。团队内部和跨团队的协作机制通过调整智能体团队内的置信水平来提高预测准确性，并促进团队之间的信息共享。通过从2023年11月至2024年9月的数据进行实证评估，我们的框架在分类、资产定价、投资组合和解释性性能方面均优于单一智能体模型和市场基准。 

---
# REM: A Scalable Reinforced Multi-Expert Framework for Multiplex Influence Maximization 

**Title (ZH)**: REM：一种可扩展的强化多专家框架，用于多层影响力最大化 

**Authors**: Huyen Nguyen, Hieu Dam, Nguyen Do, Cong Tran, Cuong Pham  

**Link**: [PDF](https://arxiv.org/pdf/2501.00779)  

**Abstract**: In social online platforms, identifying influential seed users to maximize influence spread is a crucial as it can greatly diminish the cost and efforts required for information dissemination. While effective, traditional methods for Multiplex Influence Maximization (MIM) have reached their performance limits, prompting the emergence of learning-based approaches. These novel methods aim for better generalization and scalability for more sizable graphs but face significant challenges, such as (1) inability to handle unknown diffusion patterns and (2) reliance on high-quality training samples. To address these issues, we propose the Reinforced Expert Maximization framework (REM). REM leverages a Propagation Mixture of Experts technique to encode dynamic propagation of large multiplex networks effectively in order to generate enhanced influence propagation. Noticeably, REM treats a generative model as a policy to autonomously generate different seed sets and learn how to improve them from a Reinforcement Learning perspective. Extensive experiments on several real-world datasets demonstrate that REM surpasses state-of-the-art methods in terms of influence spread, scalability, and inference time in influence maximization tasks. 

**Abstract (ZH)**: 在社交在线平台上，识别潜在的种子用户以最大化信息传播的影响范围是至关重要的，因为这可以显著减少信息传播所需的成本和努力。尽管传统的方法在多层影响最大化（MIM）方面非常有效，但它们已经达到了性能极限，从而推动了基于学习的方法的出现。这些新颖的方法旨在更好地泛化和扩展以适用于更大的图结构，但它们面临着重大挑战，例如（1）无法处理未知的传播模式和（2）依赖高质量的训练样本。为了解决这些问题，我们提出了一种强化专家最大化框架（REM）。REM 利用传播混合专家技术有效编码大规模多层网络中的动态传播，从而生成增强的影响传播。值得注意的是，REM 将生成模型视为策略，用于自主生成不同的种子集，并从强化学习的角度学习如何改进它们。在多个实际数据集上的广泛实验表明，在影响最大化任务中，REM 在影响传播范围、可扩展性和推理时间方面均超过了现有最先进的方法。 

---
# Enabling New HDLs with Agents 

**Title (ZH)**: 启用基于代理的新高级描述语言 

**Authors**: Mark Zakharov, Farzaneh Rabiei Kashanaki, Jose Renau  

**Link**: [PDF](https://arxiv.org/pdf/2501.00642)  

**Abstract**: Large Language Models (LLMs) based agents are transforming the programming language landscape by facilitating learning for beginners, enabling code generation, and optimizing documentation workflows. Hardware Description Languages (HDLs), with their smaller user community, stand to benefit significantly from the application of LLMs as tools for learning new HDLs. This paper investigates the challenges and solutions of enabling LLMs for HDLs, particularly for HDLs that LLMs have not been previously trained on. This work introduces HDLAgent, an AI agent optimized for LLMs with limited knowledge of various HDLs. It significantly enhances off-the-shelf LLMs. 

**Abstract (ZH)**: 基于大型语言模型（Large Language Models, LLMs）的代理正在通过促进初学者学习、实现代码生成以及优化文档工作流，重塑编程语言的格局。硬件描述语言（Hardware Description Languages, HDLs），因其较小的用户群体，特别可以从LLMs应用中受益，作为学习新的HDLs的工具。本文探讨了使LLMs适用于HDLs所面临的挑战及其解决方案，尤其是对于LLMs尚未被训练的HDLs。本文介绍了一种名为HDLAgent的AI代理，它针对各种HDLs具有有限知识进行了优化，显著提升了现成的LLMs。 

---
# Proactive Conversational Agents with Inner Thoughts 

**Title (ZH)**: 具有内在思考能力的前瞻性对话代理 

**Authors**: Xingyu Bruce Liu, Shitao Fang, Weiyan Shi, Chien-Sheng Wu, Takeo Igarashi, Xiang `Anthony' Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.00383)  

**Abstract**: One of the long-standing aspirations in conversational AI is to allow them to autonomously take initiatives in conversations, i.e., being proactive. This is especially challenging for multi-party conversations. Prior NLP research focused mainly on predicting the next speaker from contexts like preceding conversations. In this paper, we demonstrate the limitations of such methods and rethink what it means for AI to be proactive in multi-party, human-AI conversations. We propose that just like humans, rather than merely reacting to turn-taking cues, a proactive AI formulates its own inner thoughts during a conversation, and seeks the right moment to contribute. Through a formative study with 24 participants and inspiration from linguistics and cognitive psychology, we introduce the Inner Thoughts framework. Our framework equips AI with a continuous, covert train of thoughts in parallel to the overt communication process, which enables it to proactively engage by modeling its intrinsic motivation to express these thoughts. We instantiated this framework into two real-time systems: an AI playground web app and a chatbot. Through a technical evaluation and user studies with human participants, our framework significantly surpasses existing baselines on aspects like anthropomorphism, coherence, intelligence, and turn-taking appropriateness. 

**Abstract (ZH)**: 在对话式人工智能领域的一个长期追求目标是使它们能够在对话中主动采取行动，即表现出主动性。这在多对话语境中尤为具有挑战性。此前的自然语言处理（NLP）研究主要集中在从上下文（如之前的对话）预测下一个发言者。在本文中，我们探讨了这类方法的局限性，并重新思考在多对人机对话中“主动性”的含义。我们认为，如同人类一样，主动性的AI不仅应对轮换提示做出反应，还要在对话中形成自己的内在想法，并寻找合适的时刻参与对话。通过一项包含24名参与者的形成性研究，并结合语言学和认知心理学的启发，我们提出了“内心想法”框架。该框架让AI在显式的沟通过程之外，持续地保持着内在的思考，这使其能够通过模型其表达这些想法的内在动机来主动参与对话。我们将这一框架应用于两个实时系统：一个AI游乐场网页应用和一个聊天机器人。通过技术评估和涉及人类参与者的用户研究，我们的框架在拟人性、连贯性、智能性和轮换适宜性等方面显著超越了现有基准。 

---
# M2I2: Learning Efficient Multi-Agent Communication via Masked State Modeling and Intention Inference 

**Title (ZH)**: M2I2：通过掩蔽状态建模和意图推理学习高效的多Agent通信 

**Authors**: Chuxiong Sun, Peng He, Qirui Ji, Zehua Zang, Jiangmeng Li, Rui Wang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00312)  

**Abstract**: Communication is essential in coordinating the behaviors of multiple agents. However, existing methods primarily emphasize content, timing, and partners for information sharing, often neglecting the critical aspect of integrating shared information. This gap can significantly impact agents' ability to understand and respond to complex, uncertain interactions, thus affecting overall communication efficiency. To address this issue, we introduce M2I2, a novel framework designed to enhance the agents' capabilities to assimilate and utilize received information effectively. M2I2 equips agents with advanced capabilities for masked state modeling and joint-action prediction, enriching their perception of environmental uncertainties and facilitating the anticipation of teammates' intentions. This approach ensures that agents are furnished with both comprehensive and relevant information, bolstering more informed and synergistic behaviors. Moreover, we propose a Dimensional Rational Network, innovatively trained via a meta-learning paradigm, to identify the importance of dimensional pieces of information, evaluating their contributions to decision-making and auxiliary tasks. Then, we implement an importance-based heuristic for selective information masking and sharing. This strategy optimizes the efficiency of masked state modeling and the rationale behind information sharing. We evaluate M2I2 across diverse multi-agent tasks, the results demonstrate its superior performance, efficiency, and generalization capabilities, over existing state-of-the-art methods in various complex scenarios. 

**Abstract (ZH)**: 沟通在协调多个代理的行为中至关重要。然而，现有的方法主要强调信息共享的内容、时机和伙伴，往往忽视了整合共享信息这一关键方面。这一缺口可能严重影响代理对复杂和不确定互动的理解和响应能力，从而影响整体通信效率。为了解决这个问题，我们提出了M2I2这一新颖框架，旨在增强代理有效吸收和利用接收到信息的能力。M2I2赋予代理高级的掩蔽状态建模和联合行动预测能力，丰富了它们对环境不确定性感知的能力，并促进了对队友意图的预见。这种方法确保代理不仅获得全面且相关的信息，还能促进更具信息性和协同性的行为。此外，我们提出了一种维度理性网络，通过元学习范式进行创新训练，以识别各个维度信息的重要性，并评估其对决策和支持任务的贡献。然后，我们实现了一种基于重要性的启发式选择，进行信息的掩蔽和共享。该策略优化了掩蔽状态建模的效率以及信息共享的合理性。我们在多种多代理任务中评估了M2I2，结果显示该方法在各种复杂场景中具有优于现有最先进的方法的性能、效率和泛化能力。 

---
# The Potential of LLMs in Automating Software Testing: From Generation to Reporting 

**Title (ZH)**: 大型语言模型在自动化软件测试中的潜力：从生成到报告 

**Authors**: Betim Sherifi, Khaled Slhoub, Fitzroy Nembhard  

**Link**: [PDF](https://arxiv.org/pdf/2501.00217)  

**Abstract**: Having a high quality software is essential in software engineering, which requires robust validation and verification processes during testing activities. Manual testing, while effective, can be time consuming and costly, leading to an increased demand for automated methods. Recent advancements in Large Language Models (LLMs) have significantly influenced software engineering, particularly in areas like requirements analysis, test automation, and debugging. This paper explores an agent-oriented approach to automated software testing, using LLMs to reduce human intervention and enhance testing efficiency. The proposed framework integrates LLMs to generate unit tests, visualize call graphs, and automate test execution and reporting. Evaluations across multiple applications in Python and Java demonstrate the system's high test coverage and efficient operation. This research underscores the potential of LLM-powered agents to streamline software testing workflows while addressing challenges in scalability and accuracy. 

**Abstract (ZH)**: 高质量的软件在软件工程中至关重要，这需要在测试活动中采用强大的验证和验证流程。虽然人工测试效果显著，但其耗时和成本较高，因此对自动化方法的需求也在增加。近年来，大型语言模型（LLMs）的进步对软件工程领域产生了重大影响，特别是在需求分析、测试自动化和调试等方面。本文探索了一种面向代理的自动软件测试方法，利用LLMs减少人工干预并提高测试效率。所提出的框架结合使用LLMs生成单元测试、可视化调用图并自动化测试执行和报告。跨多个Python和Java应用程序的评估表明，该系统的测试覆盖率高且运行高效。这项研究强调了LLM驱动代理在简化软件测试工作流方面的作用，同时也应对扩展性和准确性方面的挑战。 

---
# AI Agent for Education: von Neumann Multi-Agent System Framework 

**Title (ZH)**: 教育中的AI代理：冯·诺伊曼多代理系统框架 

**Authors**: Yuan-Hao Jiang, Ruijia Li, Yizhou Zhou, Changyong Qi, Hanglei Hu, Yuang Wei, Bo Jiang, Yonghe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00083)  

**Abstract**: The development of large language models has ushered in new paradigms for education. This paper centers on the multi-Agent system in education and proposes the von Neumann multi-Agent system framework. It breaks down each AI Agent into four modules: control unit, logic unit, storage unit, and input-output devices, defining four types of operations: task deconstruction, self-reflection, memory processing, and tool invocation. Furthermore, it introduces related technologies such as Chain-of-Thought, Reson+Act, and Multi-Agent Debate associated with these four types of operations. The paper also discusses the ability enhancement cycle of a multi-Agent system for education, including the outer circulation for human learners to promote knowledge construction and the inner circulation for LLM-based-Agents to enhance swarm intelligence. Through collaboration and reflection, the multi-Agent system can better facilitate human learners' learning and enhance their teaching abilities in this process. 

**Abstract (ZH)**: 大规模语言模型的发展为教育带来了新的范式。本文关注教育中的多Agent系统，并提出了一种冯·诺伊曼多Agent系统框架。该框架将每个AI Agent分解为四个模块：控制单元、逻辑单元、存储单元和输入输出设备，并定义了四种类型的操作：任务分解、自我反思、记忆处理和工具调用。此外，本文还介绍了与这四种操作相关的相关技术，如Chain-of-Thought、Reson+Act和多Agent辩论。文章还讨论了教育中多Agent系统的能力建设计划，包括外循环以促进人类学习者的知识构建，以及内循环以增强基于LLM的Agent的群体智能。通过协作和反思，多Agent系统可以更好地促进人类学习者的学习过程，并在此过程中增强他们的教学能力。 

---
# Human-like Bots for Tactical Shooters Using Compute-Efficient Sensors 

**Title (ZH)**: 使用计算高效传感器的战术射击游戏仿人类机器人 

**Authors**: Niels Justesen, Maria Kaselimi, Sam Snodgrass, Miruna Vozaru, Matthew Schlegel, Jonas Wingren, Gabriella A. B. Barros, Tobias Mahlmann, Shyam Sudhakaran, Wesley Kerr, Albert Wang, Christoffer Holmgård, Georgios N. Yannakakis, Sebastian Risi, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2501.00078)  

**Abstract**: Artificial intelligence (AI) has enabled agents to master complex video games, from first-person shooters like Counter-Strike to real-time strategy games such as StarCraft II and racing games like Gran Turismo. While these achievements are notable, applying these AI methods in commercial video game production remains challenging due to computational constraints. In commercial scenarios, the majority of computational resources are allocated to 3D rendering, leaving limited capacity for AI methods, which often demand high computational power, particularly those relying on pixel-based sensors. Moreover, the gaming industry prioritizes creating human-like behavior in AI agents to enhance player experience, unlike academic models that focus on maximizing game performance. This paper introduces a novel methodology for training neural networks via imitation learning to play a complex, commercial-standard, VALORANT-like 2v2 tactical shooter game, requiring only modest CPU hardware during inference. Our approach leverages an innovative, pixel-free perception architecture using a small set of ray-cast sensors, which capture essential spatial information efficiently. These sensors allow AI to perform competently without the computational overhead of traditional methods. Models are trained to mimic human behavior using supervised learning on human trajectory data, resulting in realistic and engaging AI agents. Human evaluation tests confirm that our AI agents provide human-like gameplay experiences while operating efficiently under computational constraints. This offers a significant advancement in AI model development for tactical shooter games and possibly other genres. 

**Abstract (ZH)**: 人工智能（AI）已经使代理能够掌握复杂的视频游戏，从第一人称射击游戏如《反恐精英》（Counter-Strike）到即时战略游戏如《星际争霸II》以及赛车游戏如《Gran Turismo》。尽管这些成就值得关注，但在商业视频游戏制作中应用这些AI方法仍然面临挑战，因为计算资源受限。在商业场景中，大部分计算资源用于3D渲染，留给AI方法的计算能力有限，而这些方法通常需要高性能的计算能力，尤其是依赖于像素传感器的方法。此外，游戏行业更注重创造符合人类行为的AI代理以提升玩家体验，而非学术模型关注的游戏性能最大化的任务。本文介绍了一种通过模仿学习训练神经网络的新方法，使其能够在复杂的、商业标准的VALORANT风格的2V2战术射击游戏中表现优异，仅需少量CPU硬件即可进行推理。我们的方法利用了一种创新的无像素感知架构，使用少量射线传感器来高效捕捉空间信息。这些传感器使AI能够在不承担传统方法计算开销的情况下进行有效操作。模型通过监督学习模仿人类行为训练而成，从而产生具有真实感和沉浸感的AI代理。人评价测试结果显示，我们的AI代理能够在计算资源受限的情况下提供类似人类的 gameplay 经验。这为战术射击游戏及其他类型游戏的AI模型开发提供了重要进展。 

---
