# NoLiMa: Long-Context Evaluation Beyond Literal Matching 

**Title (ZH)**: NoLiMa：超越字面匹配的长上下文评估 

**Authors**: Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2502.05167)  

**Abstract**: Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves retrieving a "needle" (relevant information) from a "haystack" (long irrelevant context). Extensions of this approach include increasing distractors, fact chaining, and in-context reasoning. However, in these benchmarks, models can exploit existing literal matches between the needle and haystack to simplify the task. To address this, we introduce NoLiMa, a benchmark extending NIAH with a carefully designed needle set, where questions and needles have minimal lexical overlap, requiring models to infer latent associations to locate the needle within the haystack. We evaluate 12 popular LLMs that claim to support contexts of at least 128K tokens. While they perform well in short contexts (<1K), performance degrades significantly as context length increases. At 32K, for instance, 10 models drop below 50% of their strong short-length baselines. Even GPT-4o, one of the top-performing exceptions, experiences a reduction from an almost-perfect baseline of 99.3% to 69.7%. Our analysis suggests these declines stem from the increased difficulty the attention mechanism faces in longer contexts when literal matches are absent, making it harder to retrieve relevant information. 

**Abstract (ZH)**: 近期的大规模语言模型（LLMs）支持长达128K至1M个词元的长文本上下文。一种流行的评估方法是“针叶搜索”（Needle in a Haystack, NIAH）测试，该方法涉及从大量不相关背景信息（“haystack”）中检索“针叶”（相关信息）。该方法的扩展包括增加干扰项、事实链推理和上下文内推断。然而，在这些基准测试中，模型可以通过利用针叶和haystack之间已有的字面匹配简化任务。为了应对这一问题，我们引入了NoLiMa，这是一种扩展了NIAH的新基准，通过精心设计的针叶集，使得问题和针叶之间的词汇重叠最小，迫使模型通过推理隐含关联来定位针叶在haystack中的位置。我们评估了12个声称支持至少128K词元上下文的流行LLM。结果显示，在短文本（<1K）上下文中，这些模型表现良好，但在上下文长度增加时，性能显著下降。例如，在32K的长度下，10个模型的性能下降到强短文本基线值的50%以下。即使是最优秀的模型之一GPT-4o，其基准值几乎接近完美（99.3%），但在32K长度下也下降到69.7%。我们的分析表明，这些下降的根本原因在于在缺少字面匹配的情况下，注意力机制在长文本上下文中面临的更大困难，使得检索相关信息变得更加困难。 

---
# Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation 

**Title (ZH)**: 借助大型语言模型 transformative 科学进展：AI 辅助科学研究综述，包括科学发现、实验、内容生成与评估 

**Authors**: Steffen Eger, Yong Cao, Jennifer D'Souza, Andreas Geiger, Christian Greisinger, Stephanie Gross, Yufang Hou, Brigitte Krenn, Anne Lauscher, Yizhi Li, Chenghua Lin, Nafise Sadat Moosavi, Wei Zhao, Tristan Miller  

**Link**: [PDF](https://arxiv.org/pdf/2502.05151)  

**Abstract**: With the advent of large multimodal language models, science is now at a threshold of an AI-based technological transformation. Recently, a plethora of new AI models and tools has been proposed, promising to empower researchers and academics worldwide to conduct their research more effectively and efficiently. This includes all aspects of the research cycle, especially (1) searching for relevant literature; (2) generating research ideas and conducting experimentation; generating (3) text-based and (4) multimodal content (e.g., scientific figures and diagrams); and (5) AI-based automatic peer review. In this survey, we provide an in-depth overview over these exciting recent developments, which promise to fundamentally alter the scientific research process for good. Our survey covers the five aspects outlined above, indicating relevant datasets, methods and results (including evaluation) as well as limitations and scope for future research. Ethical concerns regarding shortcomings of these tools and potential for misuse (fake science, plagiarism, harms to research integrity) take a particularly prominent place in our discussion. We hope that our survey will not only become a reference guide for newcomers to the field but also a catalyst for new AI-based initiatives in the area of "AI4Science". 

**Abstract (ZH)**: 随着大型多模态语言模型的出现，科学正处于基于AI的技术变革门槛之上。最近，提出了一系列新的AI模型和工具，这些工具有望赋能全球的研究人员和学术界人士更有效地开展研究。这涵盖了研究周期的所有方面，尤其是（1）搜索相关文献；（2）生成研究思路并进行实验；生成（3）文本型和（4）多模态内容（例如，科学图表和图形）；以及（5）基于AI的自动同行评审。在本文中，我们对这些令人兴奋的近期发展进行了深入综述，这些发展有望从根本上改变科学研究过程。本文涵盖了上述五个方面，介绍了相关数据集、方法和结果（包括评估），以及未来研究的局限性和范围。我们讨论中特别突出的是这些工具的伦理问题及其潜在滥用风险（假科学、剽窃、损害研究诚信等）。我们希望本文不仅能成为该领域的入门参考指南，还能成为推动“AI4Science”领域新AI项目的一项催化剂。 

---
# CodeSCM: Causal Analysis for Multi-Modal Code Generation 

**Title (ZH)**: CodeSCM：多模态代码生成的因果分析 

**Authors**: Mukur Gupta, Noopur Bhatt, Suman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2502.05150)  

**Abstract**: In this paper, we propose CodeSCM, a Structural Causal Model (SCM) for analyzing multi-modal code generation using large language models (LLMs). By applying interventions to CodeSCM, we measure the causal effects of different prompt modalities, such as natural language, code, and input-output examples, on the model. CodeSCM introduces latent mediator variables to separate the code and natural language semantics of a multi-modal code generation prompt. Using the principles of Causal Mediation Analysis on these mediators we quantify direct effects representing the model's spurious leanings. We find that, in addition to natural language instructions, input-output examples significantly influence code generation. 

**Abstract (ZH)**: 在本文中，我们提出了一种名为CodeSCM的结构因果模型（Structural Causal Model, SCM），用于利用大规模语言模型（Large Language Models, LLMs）分析多模态代码生成。通过在CodeSCM中应用干预措施，我们可以衡量不同提示模态（如自然语言、代码和输入输出示例）对模型的因果影响。CodeSCM引入了潜在中介变量，用于区分多模态代码生成提示中的代码和自然语言语义。基于因果中介分析的原则，我们定量度量了直接效应，这些效应代表了模型的偏差倾向。我们发现，除了自然语言指令外，输入输出示例也显著影响代码生成。 

---
# Aligning Black-box Language Models with Human Judgments 

**Title (ZH)**: 将黑盒语言模型与人类判断对齐 

**Authors**: Gerrit J. J. van den Burg, Gen Suzuki, Wei Liu, Murat Sensoy  

**Link**: [PDF](https://arxiv.org/pdf/2502.04997)  

**Abstract**: Large language models (LLMs) are increasingly used as automated judges to evaluate recommendation systems, search engines, and other subjective tasks, where relying on human evaluators can be costly, time-consuming, and unscalable. LLMs offer an efficient solution for continuous, automated evaluation. However, since the systems that are built and improved with these judgments are ultimately designed for human use, it is crucial that LLM judgments align closely with human evaluators to ensure such systems remain human-centered. On the other hand, aligning LLM judgments with human evaluators is challenging due to individual variability and biases in human judgments. We propose a simple yet effective framework to align LLM judgments with individual human evaluators or their aggregated judgments, without retraining or fine-tuning the LLM. Our approach learns a linear mapping between the LLM's outputs and human judgments, achieving over 142% average improvement in agreement across 29 tasks with only a small number of calibration examples used for training. Notably, our method works in zero-shot and few-shot settings, exceeds inter-human agreement on four out of six tasks, and enables smaller LLMs to achieve performance comparable to that of larger models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地被用作自动化法官来评估推荐系统、搜索引擎和其他主观任务，此时依赖人工评估者可能会导致成本高昂、耗时且难以扩展。LLMs 提供了一种有效的解决方案，可以实现连续的自动化评估。然而，由于这些通过这些评估构建和改进的系统最终旨在为人使用，因此确保LLM的评估结果与人工评价者高度一致对于确保系统保持以人为本至关重要。另一方面，由于人工判断中的个体差异和偏见，使LLM评估结果与人工评价者相一致具有挑战性。我们提出了一种简单而有效的框架，用于将LLM的评估结果与特定的人工评价者或他们的汇总评估结果对齐，而无需对LLM进行重新训练或微调。我们的方法通过学习LLM输出与人工评估结果之间的线性映射关系，仅使用少量校准示例进行训练，在29个任务上的平均一致性改善超过142%。值得注意的是，我们的方法在零样本和少样本设置下有效，在六个任务中四个任务上超过了人与人之间的一致性和使较小的LLM能够达到与较大模型相当的性能。 

---
# CoCoA: A Generalized Approach to Uncertainty Quantification by Integrating Confidence and Consistency of LLM Outputs 

**Title (ZH)**: CoCoA：一种通过结合大语言模型输出的置信度和一致性来进行不确定性量化的一般方法 

**Authors**: Roman Vashurin, Maiya Goloburda, Preslav Nakov, Artem Shelmanov, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04964)  

**Abstract**: Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompasses a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches and shown impressive performance in various applications. However, they sometimes fail to outperform much simpler baseline methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency that leads to a family of efficient and robust UQ methods. We evaluate our approach across a variety of tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）的不确定性量化（UQ）方法涵盖了多种途径，其中两种主要类型尤为突出：信息基于型，重点在于通过令牌概率表达模型的置信度；一致性基于型，则评估使用重复采样生成的多个输出之间的语义关系。近年来，一些方法将这两种类型结合在一起，在多种应用中展示了出色的性能。然而，它们有时未能超越一些更为简单的基准方法。我们的研究揭示了LLMs作为概率模型的特定特征，这有助于解释为什么这些UQ方法在某些任务中表现不佳。基于这些发现，我们提出了一种新的合成模型置信度和输出一致性的方法，从而形成了一种高效且稳健的UQ方法族。我们在问答、抽象总结和机器翻译等多种任务中评估了我们的方法，展示出了相较于最先进的UQ方法的大规模改进。 

---
# SeDi-Instruct: Enhancing Alignment of Language Models through Self-Directed Instruction Generation 

**Title (ZH)**: SeDi-Instruct：通过自我导向指令生成提高语言模型的对齐性 

**Authors**: Jungwoo Kim, Minsang Kim, Sungjin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.04774)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has enabled the industry to develop various AI-based services. Instruction tuning is considered essential in adapting foundation models for target domains to provide high-quality services to customers. A key challenge in instruction tuning is obtaining high-quality instruction data. Self-Instruct, which automatically generates instruction data using ChatGPT APIs, alleviates the data scarcity problem. To improve the quality of instruction data, Self-Instruct discards many of the instructions generated from ChatGPT, even though it is inefficient in terms of cost owing to many useless API calls. To generate high-quality instruction data at a low cost, we propose a novel data generation framework, Self-Direct Instruction generation (SeDi-Instruct), which employs diversity-based filtering and iterative feedback task generation. Diversity-based filtering maintains model accuracy without excessively discarding low-quality generated instructions by enhancing the diversity of instructions in a batch. This reduces the cost of synthesizing instruction data. The iterative feedback task generation integrates instruction generation and training tasks and utilizes information obtained during the training to create high-quality instruction sets. Our results show that SeDi-Instruct enhances the accuracy of AI models by 5.2%, compared with traditional methods, while reducing data generation costs by 36%. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速演化使行业能够开发出各种基于AI的服务。指令调优被认为是在将基础模型适应目标领域的同时，为客户提供高质量服务的关键步骤。指令调优的一个主要挑战是如何获取高质量的指令数据。Self-Instruct 自动利用ChatGPT API 生成指令数据，从而缓解了数据稀缺问题。为了提高指令数据的质量，Self-Instruct 虽然从ChatGPT生成的大量指令中舍弃了很多，但这种方法在成本上显得低效，因为进行了许多不必要的API调用。为了在低成本下生成高质量的指令数据，我们提出了一种新的数据生成框架，即基于多样性的指令生成（SeDi-Instruct），该框架结合了基于多样性的过滤和迭代反馈任务生成。基于多样性的过滤方法通过增强批次指令的多样性，减少了产生指令数据的成本，而不过度舍弃低质量的生成指令。迭代反馈任务生成将指令生成任务与训练任务相结合，并利用训练过程中获得的信息来生成高质量的指令集。我们的结果显示，与传统方法相比，SeDi-Instruct 可以将AI模型的准确性提高5.2%，同时将数据生成成本降低36%。 

---
# Concept Navigation and Classification via Open Source Large Language Model Processing 

**Title (ZH)**: 通过开源大规模语言模型处理进行概念导航与分类 

**Authors**: Maël Kubli  

**Link**: [PDF](https://arxiv.org/pdf/2502.04756)  

**Abstract**: This paper presents a novel methodological framework for detecting and classifying latent constructs, including frames, narratives, and topics, from textual data using Open-Source Large Language Models (LLMs). The proposed hybrid approach combines automated summarization with human-in-the-loop validation to enhance the accuracy and interpretability of construct identification. By employing iterative sampling coupled with expert refinement, the framework guarantees methodological robustness and ensures conceptual precision. Applied to diverse data sets, including AI policy debates, newspaper articles on encryption, and the 20 Newsgroups data set, this approach demonstrates its versatility in systematically analyzing complex political discourses, media framing, and topic classification tasks. 

**Abstract (ZH)**: 本文提出了一种新颖的方法论框架，利用开源大规模语言模型（LLMs）从文本数据中检测和分类潜在构念（包括框架、叙事和主题）。该提出的混合方法结合了自动摘要与人工辅助验证，以提高构念识别的准确性和可解释性。通过迭代采样并结合专家精炼，该框架保证了方法论的稳健性和概念的精确性。该方法应用于多种数据集，包括AI政策辩论、关于加密的报纸文章以及20组新闻数据集，展示了其在系统分析复杂政治话语、媒体框架构建和主题分类任务方面的 versatility。 

---
# Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics? 

**Title (ZH)**: 文本风格转换评估：是否存在可靠的评估指标？ 

**Authors**: Sourabrata Mukherjee, Atul Kr. Ojha, John P. McCrae, Ondrej Dusek  

**Link**: [PDF](https://arxiv.org/pdf/2502.04718)  

**Abstract**: Text Style Transfer (TST) is the task of transforming a text to reflect a particular style while preserving its original content. Evaluating TST outputs is a multidimensional challenge, requiring the assessment of style transfer accuracy, content preservation, and naturalness. Using human evaluation is ideal but costly, same as in other natural language processing (NLP) tasks, however, automatic metrics for TST have not received as much attention as metrics for, e.g., machine translation or summarization. In this paper, we examine both set of existing and novel metrics from broader NLP tasks for TST evaluation, focusing on two popular subtasks-sentiment transfer and detoxification-in a multilingual context comprising English, Hindi, and Bengali. By conducting meta-evaluation through correlation with human judgments, we demonstrate the effectiveness of these metrics when used individually and in ensembles. Additionally, we investigate the potential of Large Language Models (LLMs) as tools for TST evaluation. Our findings highlight that certain advanced NLP metrics and experimental-hybrid-techniques, provide better insights than existing TST metrics for delivering more accurate, consistent, and reproducible TST evaluations. 

**Abstract (ZH)**: 文本风格转换（TST）是指将文本转换为反映特定风格的同时保留其原始内容的任务。评估TST输出是一项多维度的挑战，需要评估风格转换的准确性、内容的保留以及自然度。虽然使用人工评估是最理想的方法，但在其他自然语言处理（NLP）任务中也会产生高昂的成本，然而，针对TST的自动评估指标并没有像机器翻译或摘要等任务那样受到足够的关注。在本文中，我们探讨了来自更广泛NLP任务的现有和新颖的评价指标，重点关注多语言背景下（包括英语、印地语和孟加拉语）的两种流行子任务——情感转换和去污处理。通过元评估的方式，我们将这些指标与人工判断结果进行相关性分析，展示了这些指标在单独使用或组合使用时的有效性。此外，我们还探讨了大型语言模型（LLMs）作为TST评价工具的潜力。研究结果表明，某些先进的NLP指标和实验性混合技术，可以提供比现有TST指标更好的洞察，以实现更准确、一致和可重复的TST评估。 

---
# ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning 

**Title (ZH)**: ARR：通过分析、检索和推理使用大规模语言模型进行问答 

**Authors**: Yuwei Yin, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2502.04689)  

**Abstract**: Large language models (LLMs) achieve remarkable performance on challenging benchmarks that are often structured as multiple-choice question-answering (QA) tasks. Zero-shot Chain-of-Thought (CoT) prompting enhances reasoning in LLMs but provides only vague and generic guidance ("think step by step"). This paper introduces ARR, an intuitive and effective zero-shot prompting method that explicitly incorporates three key steps in QA solving: analyzing the intent of the question, retrieving relevant information, and reasoning step by step. Comprehensive experiments across diverse and challenging QA tasks demonstrate that ARR consistently improves the Baseline (without ARR prompting) and outperforms CoT. Ablation and case studies further validate the positive contributions of each component: analyzing, retrieving, and reasoning. Notably, intent analysis plays a vital role in ARR. Additionally, extensive evaluations across various model sizes, LLM series, and generation settings solidify the effectiveness, robustness, and generalizability of ARR. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种选择题问答（QA）任务基准测试中表现出色。无监督链式思考（CoT）提示增强了LLMs的推理能力，但仅提供模糊且通用的指导（“一步一步地思考”）。本文介绍了ARR，这是一种直观且有效的无监督提示方法，明确地将问答解决过程中的三个关键步骤融入其中：分析问题的意图、检索相关信息、逐步推理。通过跨多种多样且具有挑战性的问答任务进行全面实验，结果显示ARR始终改进了基准模型（无ARR提示）并优于CoT。消融分析和案例研究进一步验证了每一步骤（分析、检索和推理）的积极贡献。值得注意的是，在ARR中意图分析扮演着至关重要的角色。此外，对不同模型规模、LLMs系列和生成设置的各种评估进一步验证了ARR的有效性、鲁棒性和通用性。 

---
# M-IFEval: Multilingual Instruction-Following Evaluation 

**Title (ZH)**: M-IFEval：多语言指令遵循评估 

**Authors**: Antoine Dussolle, Andrea Cardeña Díaz, Shota Sato, Peter Devine  

**Link**: [PDF](https://arxiv.org/pdf/2502.04688)  

**Abstract**: Instruction following is a core capability of modern Large language models (LLMs), making evaluating this capability essential to understanding these models. The Instruction Following Evaluation (IFEval) benchmark from the literature does this using objective criteria, offering a measure of LLM performance without subjective AI or human judgement. However, it only includes English instructions, limiting its ability to assess LLMs in other languages.
We propose the Multilingual Instruction Following Evaluation (M-IFEval) benchmark, expanding the evaluation to French, Japanese, and Spanish, with both general and language-specific instructions. Applying this benchmark to 8 state-of-the-art LLMs, we find that benchmark performance across languages and instruction types can vary widely, underscoring the importance of a multilingual benchmark for evaluating LLMs in a diverse cultural context. 

**Abstract (ZH)**: 现代大型语言模型（LLMs）具备指令跟随的核心能力，因此评估这一能力对于理解这些模型至关重要。文献中的指令跟随评估（IFEval）基准使用客观标准进行评估，提供了一种无需主观人工智能或人类判断的LLM性能衡量方法。然而，它仅包括英语指令，限制了其评估其他语言模型的能力。
我们提出了多语言指令跟随评估（M-IFEval）基准，将评估扩展到了法语、日语和西班牙语，并包括通用和语言特定的指令。将这一基准应用于8个最先进的LLM模型后，我们发现不同语言和指令类型的基准性能差异很大，强调了在多元文化背景下评估LLM时使用多语言基准的重要性。 

---
# Extracting and Understanding the Superficial Knowledge in Alignment 

**Title (ZH)**: 提取和理解对齐中的表面知识 

**Authors**: Runjin Chen, Gabriel Jacob Perin, Xuxi Chen, Xilun Chen, Yan Han, Nina S. T. Hirata, Junyuan Hong, Bhavya Kailkhura  

**Link**: [PDF](https://arxiv.org/pdf/2502.04602)  

**Abstract**: Alignment of large language models (LLMs) with human values and preferences, often achieved through fine-tuning based on human feedback, is essential for ensuring safe and responsible AI behaviors. However, the process typically requires substantial data and computation resources. Recent studies have revealed that alignment might be attainable at lower costs through simpler methods, such as in-context learning. This leads to the question: Is alignment predominantly superficial? In this paper, we delve into this question and provide a quantitative analysis. We formalize the concept of superficial knowledge, defining it as knowledge that can be acquired through easily token restyling, without affecting the model's ability to capture underlying causal relationships between tokens. We propose a method to extract and isolate superficial knowledge from aligned models, focusing on the shallow modifications to the final token selection process. By comparing models augmented only with superficial knowledge to fully aligned models, we quantify the superficial portion of alignment. Our findings reveal that while superficial knowledge constitutes a significant portion of alignment, particularly in safety and detoxification tasks, it is not the whole story. Tasks requiring reasoning and contextual understanding still rely on deeper knowledge. Additionally, we demonstrate two practical advantages of isolated superficial knowledge: (1) it can be transferred between models, enabling efficient offsite alignment of larger models using extracted superficial knowledge from smaller models, and (2) it is recoverable, allowing for the restoration of alignment in compromised models without sacrificing performance. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与人类价值观和偏好对齐，通常通过基于人类反馈的微调来实现，这对于确保安全和负责任的AI行为至关重要。然而，这一过程通常需要大量的数据和计算资源。近期的研究表明，可能通过更简单的方法，如上下文学习，以较低的成本实现对齐。这引发了一个问题：对齐是否主要只是表面现象？在本文中，我们深入探讨了这个问题，并进行了定量分析。我们正式定义了表面知识的概念，将其定义为可以通过简单的标记重制获取的知识，而不影响模型捕捉令牌之间潜在因果关系的能力。我们提出了一种方法来从对齐模型中提取并隔离表面知识，重点关注对最终令牌选择过程的浅层修改。通过将只添加表面知识的模型与完全对齐的模型进行比较，我们量化了表面对齐的比例。我们的研究结果表明，虽然表面知识在对齐中占有重要比例，特别是在安全性与去毒任务中，但它并非全部。需要推理和背景理解的任务仍然依赖于更深层次的知识。此外，我们还展示了隔离的表面知识的两个实际优势：（1）它可以转移到其他模型中，使较大的模型能够利用较小模型提取出的表面知识进行高效的离线对齐；（2）它是可恢复的，从而允许在不牺牲性能的情况下恢复受损模型的对齐。 

---
# My LLM might Mimic AAE -- But When Should it? 

**Title (ZH)**: 我的大语言模型可能会模仿社会方言——但在什么情况下应该模仿呢？ 

**Authors**: Sandra C. Sandoval, Christabel Acquaye, Kwesi Cobbina, Mohammad Nayeem Teli, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2502.04564)  

**Abstract**: We examine the representation of African American English (AAE) in large language models (LLMs), exploring (a) the perceptions Black Americans have of how effective these technologies are at producing authentic AAE, and (b) in what contexts Black Americans find this desirable. Through both a survey of Black Americans ($n=$ 104) and annotation of LLM-produced AAE by Black Americans ($n=$ 228), we find that Black Americans favor choice and autonomy in determining when AAE is appropriate in LLM output. They tend to prefer that LLMs default to communicating in Mainstream U.S. English in formal settings, with greater interest in AAE production in less formal settings. When LLMs were appropriately prompted and provided in context examples, our participants found their outputs to have a level of AAE authenticity on par with transcripts of Black American speech. Select code and data for our project can be found here: this https URL 

**Abstract (ZH)**: 我们探讨了大型语言模型（LLMs）中非洲美国英语（AAE）的表述，研究了（a）非洲裔美国人如何看待这些技术在产生真实AAE方面的效果，以及（b）在什么语境下非洲裔美国人认为这种表述是有价值的。通过对104名非洲裔美国人的问卷调查以及228名非洲裔美国人对LLM生成的AAE进行标注，我们发现非洲裔美国人倾向于在确定何时在LLM输出中使用AAE时拥有选择权和自主权。他们倾向于在正式场合让LLM默认使用主流美国英语进行交流，在非正式场合则更感兴趣于AAE的生成。当LLM被适当提示并提供相关示例时，我们的参与者发现其输出具有与非洲裔美国人口语录音相当的真实AAE水平。我们项目的部分代码和数据可以在以下链接找到：[这里](this https URL) 

---
# TruthFlow: Truthful LLM Generation via Representation Flow Correction 

**Title (ZH)**: TruthFlow：通过表示流矫正实现的真诚生成大语言模型 

**Authors**: Hanyu Wang, Bochuan Cao, Yuanpu Cao, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04556)  

**Abstract**: Large language models (LLMs) are known to struggle with consistently generating truthful responses. While various representation intervention techniques have been proposed, these methods typically apply a universal representation correction vector to all input queries, limiting their effectiveness against diverse queries in practice. In this study, we introduce TruthFlow, a novel method that leverages the Flow Matching technique for query-specific truthful representation correction. Specifically, TruthFlow first uses a flow model to learn query-specific correction vectors that transition representations from hallucinated to truthful states. Then, during inference, the trained flow model generates these correction vectors to enhance the truthfulness of LLM outputs. Experimental results demonstrate that TruthFlow significantly improves performance on open-ended generation tasks across various advanced LLMs evaluated on TruthfulQA. Moreover, the trained TruthFlow model exhibits strong transferability, performing effectively on other unseen hallucination benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在持续生成真实响应方面存在困难。尽管提出了多种表示干预技术，这些方法通常对所有输入查询应用一个通用的表示校正向量，这在实践中限制了其对多样查询的有效性。在此研究中，我们引入了TruthFlow，这是一种利用Flow Matching技术进行查询特定真实表示校正的新方法。具体而言，TruthFlow 首先使用流动模型学习查询特定的校正向量，这些向量能够将表示从虚构状态过渡到真实状态。然后，在推理过程中，训练好的流动模型生成这些校正向量以增强LLM输出的真实性。实验结果表明，TruthFlow 显著提高了各种高级LLM在TruthfulQA上的开放生成任务性能。此外，训练好的TruthFlow模型具有较强的迁移能力，能够在其他未见过的虚构基准测试中表现出色。 

---
# Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems 

**Title (ZH)**: 异质群集：多大型语言模型系统中模型角色和权重的联合优化 

**Authors**: Shangbin Feng, Zifeng Wang, Palash Goyal, Yike Wang, Weijia Shi, Huang Xia, Hamid Palangi, Luke Zettlemoyer, Yulia Tsvetkov, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2502.04510)  

**Abstract**: We propose Heterogeneous Swarms, an algorithm to design multi-LLM systems by jointly optimizing model roles and weights. We represent multi-LLM systems as directed acyclic graphs (DAGs) of LLMs with topological message passing for collaborative generation. Given a pool of LLM experts and a utility function, Heterogeneous Swarms employs two iterative steps: role-step and weight-step. For role-step, we interpret model roles as learning a DAG that specifies the flow of inputs and outputs between LLMs. Starting from a swarm of random continuous adjacency matrices, we decode them into discrete DAGs, call the LLMs in topological order, evaluate on the utility function (e.g. accuracy on a task), and optimize the adjacency matrices with particle swarm optimization based on the utility score. For weight-step, we assess the contribution of individual LLMs in the multi-LLM systems and optimize model weights with swarm intelligence. We propose JFK-score to quantify the individual contribution of each LLM in the best-found DAG of the role-step, then optimize model weights with particle swarm optimization based on the JFK-score. Experiments demonstrate that Heterogeneous Swarms outperforms 15 role- and/or weight-based baselines by 18.5% on average across 12 tasks. Further analysis reveals that Heterogeneous Swarms discovers multi-LLM systems with heterogeneous model roles and substantial collaborative gains, and benefits from the diversity of language models. 

**Abstract (ZH)**: 我们提出了一种异构群集算法——Heterogeneous Swarms，用于设计多大型语言模型（multi-LLM）系统并通过联合优化模型角色和权重来进行优化。我们将多大型语言模型系统表示为具有拓扑消息传递的大型语言模型有向无环图（DAG），以实现协作生成。给定一组大型语言模型专家和效用函数，Heterogeneous Swarms 采用两个迭代步骤：角色步和权重步。在角色步中，我们将模型角色解释为学习一个DAG，该DAG指定了大型语言模型之间输入和输出的流。从随机连续邻接矩阵的群集开始，将其解码为离散的DAG，并按拓扑顺序调用大型语言模型，评估效用函数（例如，在任务中的准确性），然后基于效用分数使用粒子群优化优化邻接矩阵。在权重步中，我们评估多大型语言模型系统中单个大型语言模型的贡献，并使用群体智能优化模型权重。我们提出了JFK评分来量化角色步步中找到的最佳DAG中每个大型语言模型的个体贡献，然后基于JFK评分使用粒子群优化优化模型权重。实验表明，在12项任务中，Heterogeneous Swarms 平均优于18.5%的基于角色和/或权重的基线方法。进一步的分析表明，Heterogeneous Swarms 发现了具有异构模型角色和显著协作增益的多大型语言模型系统，并且受益于语言模型的多样性。 

---
# When One LLM Drools, Multi-LLM Collaboration Rules 

**Title (ZH)**: 当一个大语言模型表现卓越时，多大语言模型协作取胜 

**Authors**: Shangbin Feng, Wenxuan Ding, Alisa Liu, Zifeng Wang, Weijia Shi, Yike Wang, Zejiang Shen, Xiaochuang Han, Hunter Lang, Chen-Yu Lee, Tomas Pfister, Yejin Choi, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04506)  

**Abstract**: This position paper argues that in many realistic (i.e., complex, contextualized, subjective) scenarios, one LLM is not enough to produce a reliable output. We challenge the status quo of relying solely on a single general-purpose LLM and argue for multi-LLM collaboration to better represent the extensive diversity of data, skills, and people. We first posit that a single LLM underrepresents real-world data distributions, heterogeneous skills, and pluralistic populations, and that such representation gaps cannot be trivially patched by further training a single LLM. We then organize existing multi-LLM collaboration methods into a hierarchy, based on the level of access and information exchange, ranging from API-level, text-level, logit-level, to weight-level collaboration. Based on these methods, we highlight how multi-LLM collaboration addresses challenges that a single LLM struggles with, such as reliability, democratization, and pluralism. Finally, we identify the limitations of existing multi-LLM methods and motivate future work. We envision multi-LLM collaboration as an essential path toward compositional intelligence and collaborative AI development. 

**Abstract (ZH)**: 本文的观点论文认为，在许多实际场景（即复杂、情境化和主观的场景）中，单一的语言模型（LLM）不足以产生可靠的结果。我们挑战仅依赖单一通用语言模型的现状，并呼吁多语言模型协作以更好地代表数据、技能和人群的广泛多样性。首先，我们认为单一语言模型未能准确反映现实世界的数据分布、异质技能和多元人口，这样的代表性缺口无法通过进一步训练单一语言模型轻易弥补。然后，我们按照访问级别和信息交换级别将现有的多语言模型协作方法组织成一个层次结构，涵盖从API级、文本级、logit级到权重级的协作。基于这些方法，我们强调多语言模型协作如何解决单一语言模型难以应对的挑战，如可靠性、民主化和多元性。最后，我们指出了现有多语言模型方法的局限性，并激发未来的研究。我们设想多语言模型协作是实现组合智能和协作人工智能开发不可或缺的途径。 

---
# Multi-Agent Reinforcement Learning with Focal Diversity Optimization 

**Title (ZH)**: 多智能体强化学习中的焦点多样性优化 

**Authors**: Selim Furkan Tekin, Fatih Ilhan, Tiansheng Huang, Sihao Hu, Zachary Yahn, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04492)  

**Abstract**: The advancement of Large Language Models (LLMs) and their finetuning strategies has triggered the renewed interests in multi-agent reinforcement learning. In this paper, we introduce a focal diversity-optimized multi-agent reinforcement learning approach, coined as MARL-Focal, with three unique characteristics. First, we develop an agent-fusion framework for encouraging multiple LLM based agents to collaborate in producing the final inference output for each LLM query. Second, we develop a focal-diversity optimized agent selection algorithm that can choose a small subset of the available agents based on how well they can complement one another to generate the query output. Finally, we design a conflict-resolution method to detect output inconsistency among multiple agents and produce our MARL-Focal output through reward-aware and policy-adaptive inference fusion. Extensive evaluations on five benchmarks show that MARL-Focal is cost-efficient and adversarial-robust. Our multi-agent fusion model achieves performance improvement of 5.51\% compared to the best individual LLM-agent and offers stronger robustness over the TruthfulQA benchmark. Code is available at this https URL 

**Abstract (ZH)**: 大语言模型（LLMs）的进步及其微调策略重新引发了对多智能体强化学习的兴趣。本文介绍了一种具有三个独特特征的焦点多样化优化多智能体强化学习方法，命名为MARL-Focal。首先，我们开发了一种智能体融合框架，以鼓励多种基于LLM的智能体合作生成每个LLM查询的最终推理输出。其次，我们开发了一种焦点多样化优化的智能体选择算法，可以根据它们如何相互补充生成查询输出来选择可用智能体的小子集。最后，我们设计了一种冲突解决方法来检测多个智能体之间的输出不一致性，并通过基于奖励感知和策略自适应的推理融合生成我们的MARL-Focal输出。在五个基准上的广泛评估表明，MARL-Focal具有成本效益且对抗鲁棒性强。我们的多智能体融合模型在TruthfulQA基准上的性能比最佳单个LLM智能体提高了5.51%，并在鲁棒性方面表现更优异。相关代码可在以下地址获得：[此处提供网址] 

---
# Building A Unified AI-centric Language System: analysis, framework and future work 

**Title (ZH)**: 构建统一的人工智能中心语言系统：分析、框架及未来工作 

**Authors**: Edward Hong Wang, Cynthia Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04488)  

**Abstract**: Recent advancements in large language models have demonstrated that extended inference through techniques can markedly improve performance, yet these gains come with increased computational costs and the propagation of inherent biases found in natural languages. This paper explores the design of a unified AI-centric language system that addresses these challenges by offering a more concise, unambiguous, and computationally efficient alternative to traditional human languages. We analyze the limitations of natural language such as gender bias, morphological irregularities, and contextual ambiguities and examine how these issues are exacerbated within current Transformer architectures, where redundant attention heads and token inefficiencies prevail. Drawing on insights from emergent artificial communication systems and constructed languages like Esperanto and Lojban, we propose a framework that translates diverse natural language inputs into a streamlined AI-friendly language, enabling more efficient model training and inference while reducing memory footprints. Finally, we outline a pathway for empirical validation through controlled experiments, paving the way for a universal interchange format that could revolutionize AI-to-AI and human-to-AI interactions by enhancing clarity, fairness, and overall performance. 

**Abstract (ZH)**: 最近在大型语言模型方面的进展表明，通过扩展推理技术可以显著提高性能，但这些改进伴随着计算成本的增加和自然语言中固有偏见的传播。本文探讨了一种统一的人工智能为中心的语言系统的设计，该系统通过提供一种更为简洁、明确且计算高效的替代传统人类语言的方式，来应对这些挑战。我们分析了自然语言的局限性，如性别偏见、形态不规则性和上下文歧义性，并研究了这些问题在当前的Transformer架构中是如何被放大的，这些架构普遍存在冗余的注意力头部和令牌效率低下等问题。借鉴新兴的人工交流系统以及世界语（如Esperanto）和构造语言（如Lojban）的原理，我们提出了一种框架，该框架将多样化的自然语言输入转化为一种精简的AI友好语言，从而实现更高效的模型训练和推理，并减少内存占用。最后，我们概述了一条通过受控实验进行实证验证的途径，为促进AI-对-AI和人-对-AI交互提供了可能的标准化格式，以增强清晰度、公平性和整体性能。 

---
# Active Task Disambiguation with LLMs 

**Title (ZH)**: 使用大语言模型进行主动任务消歧 Paginationalbums 

**Authors**: Katarzyna Kobalczyk, Nicolas Astorga, Tennison Liu, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2502.04485)  

**Abstract**: Despite the impressive performance of large language models (LLMs) across various benchmarks, their ability to address ambiguously specified problems--frequent in real-world interactions--remains underexplored. To address this gap, we introduce a formal definition of task ambiguity and frame the problem of task disambiguation through the lens of Bayesian Experimental Design. By posing clarifying questions, LLM agents can acquire additional task specifications, progressively narrowing the space of viable solutions and reducing the risk of generating unsatisfactory outputs. Yet, generating effective clarifying questions requires LLM agents to engage in a form of meta-cognitive reasoning, an ability LLMs may presently lack. Our proposed approach of active task disambiguation enables LLM agents to generate targeted questions maximizing the information gain. Effectively, this approach shifts the load from implicit to explicit reasoning about the space of viable solutions. Empirical results demonstrate that this form of question selection leads to more effective task disambiguation in comparison to approaches relying on reasoning solely within the space of questions. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种基准测试中表现出色，但它们在解决模糊指定的问题方面的能力仍待探索——而在现实世界的互动中，这种模糊指定的问题是常见的。为填补这一空白，我们提出了任务模糊的正式定义，并将任务去模糊问题框架化为贝叶斯实验设计的问题。通过提出澄清问题，LLM代理可以获取额外的任务说明，逐步缩小可行解的空间，减少生成不满意输出的风险。然而，生成有效的澄清问题需要LLM代理进行一种形式的元认知推理，而这种能力目前可能是LLMs所缺乏的。我们提出的一种主动任务去模糊方法，使LLM代理能够生成最大化信息增益的针对性问题。实际上，这种方法将推理负载从隐含推理转移到显式推理。实证结果显示，这种问题选择方式在任务去模糊的效率上优于依赖于问题空间内单纯推理的方法。 

---
# Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization 

**Title (ZH)**: 自信与否：从基准测试到泛化的基于不确定性本地LLM路由探索 

**Authors**: Yu-Neng Chuang, Leisheng Yu, Guanchu Wang, Lizhe Zhang, Zirui Liu, Xuanting Cai, Yang Sui, Vladimir Braverman, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04428)  

**Abstract**: Large language models (LLMs) are increasingly deployed and democratized on edge devices. To improve the efficiency of on-device deployment, small language models (SLMs) are often adopted due to their efficient decoding latency and reduced energy consumption. However, these SLMs often generate inaccurate responses when handling complex queries. One promising solution is uncertainty-based SLM routing, offloading high-stakes queries to stronger LLMs when resulting in low-confidence responses on SLM. This follows the principle of "If you lack confidence, seek stronger support" to enhance reliability. Relying on more powerful LLMs is yet effective but increases invocation costs. Therefore, striking a routing balance between efficiency and efficacy remains a critical challenge. Additionally, efficiently generalizing the routing strategy to new datasets remains under-explored. In this paper, we conduct a comprehensive investigation into benchmarking and generalization of uncertainty-driven routing strategies from SLMs to LLMs over 1500+ settings. Our findings highlight: First, uncertainty-correctness alignment in different uncertainty quantification (UQ) methods significantly impacts routing performance. Second, uncertainty distributions depend more on both the specific SLM and the chosen UQ method, rather than downstream data. Building on the insight, we propose a calibration data construction instruction pipeline and open-source a constructed hold-out set to enhance routing generalization on new downstream scenarios. The experimental results indicate calibration data effectively bootstraps routing performance without any new data. 

**Abstract (ZH)**: 大型语言模型（LLMs）正越来越多地部署在边缘设备上。为了提高在设备上的部署效率，通常会采用小型语言模型（SLMs），因为它们具有高效的解码延迟和较低的能耗。然而，这些SLMs在处理复杂查询时往往会生成不准确的响应。一种有前景的解决方案是基于不确定性的小型语言模型路由，当SLMs生成低置信度响应时，将高风险查询卸载到更强的LLMs上。这遵循了“如果你缺乏信心，则寻求更强的支持”的原则，以提高可靠性。依赖更强大的LLMs虽然有效，但会增加调用成本。因此，如何在效率和效果之间找到一个平衡仍然是一个关键挑战。此外，如何高效地将路由策略推广到新的数据集仍然没有得到充分探索。在本文中，我们对从SLMs到LLMs的各种不确定性驱动的路由策略进行了全面的基准测试和推广研究，涵盖了1500多个场景。我们的发现显示：首先，不同不确定性量化（UQ）方法的不确定性-正确性对齐显著影响路由性能。第二，不确定性分布更多地依赖于特定的SLM和选择的UQ方法，而不是下游数据。基于这一洞察，我们提出了一种校准数据构建指令管道，并开源了一个构建好的hold-out集，以增强在新下游场景中的路由泛化能力。实验结果表明，校准数据能够有效地提升路由性能，而无需额外的新数据。 

---
# Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models 

**Title (ZH)**: 向前跃进之前向后一步：自我回溯以增强语言模型的推理能力 

**Authors**: Xiao-Wen Yang, Xuan-Yi Zhu, Wen-Da Wei, Ding-Chu Zhang, Jie-Jing Shao, Zhi Zhou, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.04404)  

**Abstract**: The integration of slow-thinking mechanisms into large language models (LLMs) offers a promising way toward achieving Level 2 AGI Reasoners, as exemplified by systems like OpenAI's o1. However, several significant challenges remain, including inefficient overthinking and an overreliance on auxiliary reward models. We point out that these limitations stem from LLMs' inability to internalize the search process, a key component of effective reasoning. A critical step toward addressing this issue is enabling LLMs to autonomously determine when and where to backtrack, a fundamental operation in traditional search algorithms. To this end, we propose a self-backtracking mechanism that equips LLMs with the ability to backtrack during both training and inference. This mechanism not only enhances reasoning ability but also efficiency by transforming slow-thinking processes into fast-thinking through self-improvement. Empirical evaluations demonstrate that our proposal significantly enhances the reasoning capabilities of LLMs, achieving a performance gain of over 40 percent compared to the optimal-path supervised fine-tuning method. We believe this study introduces a novel and promising pathway for developing more advanced and robust Reasoners. 

**Abstract (ZH)**: 将大语言模型（LLMs）中整合缓慢思考机制的方法为实现Level 2 AGI推理器提供了前景广阔的可能性，如OpenAI的o1系统所示。然而，仍然存在几个重大挑战，包括无效的过度思考和过分依赖辅助奖励模型。我们指出，这些限制源于LLMs无法内化搜索过程，这是有效推理的关键组成部分。解决这一问题的关键步骤之一是使LLMs能够自主确定何时及何处回溯，这是传统搜索算法中的一个基本操作。为此，我们提出了一种自我回溯机制，使LLMs在训练和推理过程中都能具备回溯的能力。该机制不仅增强了推理能力，还通过自我改进将缓慢思考过程转化为快速思考，从而提高效率。实验评估表明，我们的提案显著提升了LLMs的推理能力，与最优路径监督微调方法相比，性能提高了超过40%。我们认为，这项研究为开发更具高级和稳健的推理器开辟了一条新颖且有前景的道路。 

---
# DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Disease 

**Title (ZH)**: DECT：利用LLM辅助的细粒度语言知识及标签切换和标签保留的数据生成方法在阿尔茨海默病诊断中的应用 

**Authors**: Tingyu Mo, Jacqueline C. K. Lam, Victor O.K. Li, Lawrence Y. L. Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2502.04394)  

**Abstract**: Alzheimer's Disease (AD) is an irreversible neurodegenerative disease affecting 50 million people worldwide. Low-cost, accurate identification of key markers of AD is crucial for timely diagnosis and intervention. Language impairment is one of the earliest signs of cognitive decline, which can be used to discriminate AD patients from normal control individuals. Patient-interviewer dialogues may be used to detect such impairments, but they are often mixed with ambiguous, noisy, and irrelevant information, making the AD detection task difficult. Moreover, the limited availability of AD speech samples and variability in their speech styles pose significant challenges in developing robust speech-based AD detection models. To address these challenges, we propose DECT, a novel speech-based domain-specific approach leveraging large language models (LLMs) for fine-grained linguistic analysis and label-switched label-preserved data generation. Our study presents four novelties: We harness the summarizing capabilities of LLMs to identify and distill key Cognitive-Linguistic information from noisy speech transcripts, effectively filtering irrelevant information. We leverage the inherent linguistic knowledge of LLMs to extract linguistic markers from unstructured and heterogeneous audio transcripts. We exploit the compositional ability of LLMs to generate AD speech transcripts consisting of diverse linguistic patterns to overcome the speech data scarcity challenge and enhance the robustness of AD detection models. We use the augmented AD textual speech transcript dataset and a more fine-grained representation of AD textual speech transcript data to fine-tune the AD detection model. The results have shown that DECT demonstrates superior model performance with an 11% improvement in AD detection accuracy on the datasets from DementiaBank compared to the baselines. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是一种不可逆的神经退行性疾病，全球影响着5000多万人。低成本而准确地识别AD的关键标记对于及时诊断和干预至关重要。语言障碍是认知衰退最早的表现之一，可以用于区分AD患者和正常对照个体。患者与访谈者的对话可能用于检测这种障碍，但这些对话往往混杂着模糊、噪音和不相关信息，使得AD检测任务变得困难。此外，AD语音样本的有限可用性和其语音风格的差异性给基于语音的AD检测模型的稳健开发带来了重大挑战。为了解决这些挑战，我们提出了DECT，这是一种利用大型语言模型（LLMs）进行细粒度语言分析和标签切换标签保留数据生成的创新性的领域特定方法。我们的研究提出了四个创新之处：我们利用LLMs的总结能力来识别和提取语音记录中的关键认知-语言信息，有效地过滤掉不相关信息。我们利用LLMs的内在语言知识来从结构化和异质音频转录中提取语言标记。我们利用LLMs的组合能力生成包含多种语言模式的AD语音转录，以克服语音数据稀缺的问题并增强AD检测模型的鲁棒性。我们使用了增强后的AD文本语音转录数据集和更细粒度的AD文本语音转录数据来微调AD检测模型。结果显示，DECT在DementiaBank数据集上的AD检测准确率比基准方法提高了11%。 

---
# Division-of-Thoughts: Harnessing Hybrid Language Model Synergy for Efficient On-Device Agents 

**Title (ZH)**: 思绪分割：利用混合语言模型协同效应以实现高效的本地代理 

**Authors**: Chenyang Shao, Xinyuan Hu, Yutang Lin, Fengli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04392)  

**Abstract**: The rapid expansion of web content has made on-device AI assistants indispensable for helping users manage the increasing complexity of online tasks. The emergent reasoning ability in large language models offer a promising path for next-generation on-device AI agents. However, deploying full-scale Large Language Models (LLMs) on resource-limited local devices is challenging. In this paper, we propose Division-of-Thoughts (DoT), a collaborative reasoning framework leveraging the synergy between locally deployed Smaller-scale Language Models (SLMs) and cloud-based LLMs. DoT leverages a Task Decomposer to elicit the inherent planning abilities in language models to decompose user queries into smaller sub-tasks, which allows hybrid language models to fully exploit their respective strengths. Besides, DoT employs a Task Scheduler to analyze the pair-wise dependency of sub-tasks and create a dependency graph, facilitating parallel reasoning of sub-tasks and the identification of key steps. To allocate the appropriate model based on the difficulty of sub-tasks, DoT leverages a Plug-and-Play Adapter, which is an additional task head attached to the SLM that does not alter the SLM's parameters. To boost adapter's task allocation capability, we propose a self-reinforced training method that relies solely on task execution feedback. Extensive experiments on various benchmarks demonstrate that our DoT significantly reduces LLM costs while maintaining competitive reasoning accuracy. Specifically, DoT reduces the average reasoning time and API costs by 66.12% and 83.57%, while achieving comparable reasoning accuracy with the best baseline methods. 

**Abstract (ZH)**: 互联网内容的迅速扩展使得设备端人工智能助手成为帮助用户应对日益复杂在线任务的不可或缺工具。大型语言模型（LLMs）涌现的推理能力为新一代设备端AI代理的发展提供了有希望的道路。然而，在资源受限的本地设备上部署大规模的语言模型极具挑战性。本文提出了一种名为Thoughts Division（DoT）的合作推理框架，该框架利用本地部署的小规模语言模型（SLMs）和基于云的大规模语言模型之间的协同作用。DoT利用任务分解器激活语言模型内部的规划能力，将用户查询分解为更小的子任务，从而使混合语言模型能够充分利用各自的优点。此外，DoT使用任务调度器分析子任务之间的依赖关系并构建依赖图，便于并行推理子任务并识别关键步骤。为了根据子任务的难度分配合适的模型，DoT利用了一个插拔式适配器，这是一种附加在SLM上的任务头，不会改变SLM的参数。为了增强适配器的任务分配能力，我们提出了一种依靠任务执行反馈的自强化训练方法。在各种基准上的广泛实验表明，我们的DoT在显著降低LLM成本的同时，保持了竞争力的推理准确性。具体而言，DoT将平均推理时间和API成本降低了66.12%和83.57%，同时达到了与最佳基线方法相当的推理准确性。 

---
# Enhancing Reasoning to Adapt Large Language Models for Domain-Specific Applications 

**Title (ZH)**: 增强推理能力以适应特定领域的大语言模型应用 

**Authors**: Bo Wen, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04384)  

**Abstract**: This paper presents SOLOMON, a novel Neuro-inspired Large Language Model (LLM) Reasoning Network architecture that enhances the adaptability of foundation models for domain-specific applications. Through a case study in semiconductor layout design, we demonstrate how SOLOMON enables swift adaptation of general-purpose LLMs to specialized tasks by leveraging Prompt Engineering and In-Context Learning techniques. Our experiments reveal the challenges LLMs face in spatial reasoning and applying domain knowledge to practical problems. Results show that SOLOMON instances significantly outperform their baseline LLM counterparts and achieve performance comparable to state-of-the-art reasoning model, o1-preview. We discuss future research directions for developing more adaptive AI systems that can continually learn, adapt, and evolve in response to new information and changing requirements. 

**Abstract (ZH)**: 本文介绍了SOLOMON，这是一种新颖的神经启发式大型语言模型（LLM）推理网络架构，旨在增强基础模型在特定领域应用中的适应性。通过在半导体布局设计中的案例研究，我们展示了SOLOMON如何通过利用提示工程和场景上下文学习技术，使通用语言模型能够迅速适应专门任务。我们的实验揭示了LLM在空间推理和将领域知识应用于实际问题时面临的挑战。结果表明，SOLOMON实例在性能上显著优于基线LLM，并达到了与当前最先进推理模型o1-preview相当的水平。我们讨论了未来研究方向，旨在开发更加适应的AI系统，使其能够根据新的信息和变化的需求不断学习、适应和进化。 

---
# MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf 

**Title (ZH)**: 代理出席：评估聊天生成模型代为出席会议的能力

这个标题翻译成中文时，保持了原有的含义和学术规范，同时确保语言通顺自然。在学术论文中，短语 "MEETING DELEGATE" 在这里被解释为“代理出席”，"Benchmarking LLMs" 被翻译为“评估聊天生成模型”，"Attending Meetings on Our Behalf" 则翻译为“代为出席会议的能力”。这样的翻译能够准确传达原文的含义。 

**Authors**: Lingxiang Hu, Shurun Yuan, Xiaoting Qin, Jue Zhang, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04376)  

**Abstract**: In contemporary workplaces, meetings are essential for exchanging ideas and ensuring team alignment but often face challenges such as time consumption, scheduling conflicts, and inefficient participation. Recent advancements in Large Language Models (LLMs) have demonstrated their strong capabilities in natural language generation and reasoning, prompting the question: can LLMs effectively delegate participants in meetings? To explore this, we develop a prototype LLM-powered meeting delegate system and create a comprehensive benchmark using real meeting transcripts. Our evaluation reveals that GPT-4/4o maintain balanced performance between active and cautious engagement strategies. In contrast, Gemini 1.5 Pro tends to be more cautious, while Gemini 1.5 Flash and Llama3-8B/70B display more active tendencies. Overall, about 60\% of responses address at least one key point from the ground-truth. However, improvements are needed to reduce irrelevant or repetitive content and enhance tolerance for transcription errors commonly found in real-world settings. Additionally, we implement the system in practical settings and collect real-world feedback from demos. Our findings underscore the potential and challenges of utilizing LLMs as meeting delegates, offering valuable insights into their practical application for alleviating the burden of meetings. 

**Abstract (ZH)**: 在当代工作场所中，会议对于交流思想和确保团队协同至关重要，但常常面临时间消耗、日程冲突和参与不充分等挑战。近期大型语言模型（LLMs）的发展显示了其在自然语言生成和推理方面的强大能力，不禁引出一个问题：LLMs能否有效地管理会议参与者？为探究这一问题，我们开发了一个基于LLMs的会议代理系统，并使用真实的会议记录创建了一个全面的基准。我们的评估发现，GPT-4/4o在积极和谨慎参与策略之间保持了平衡的性能。相比之下，Gemini 1.5 Pro更加谨慎，而Gemini 1.5 Flash和Llama3-8B/70B则表现出更强的主动性倾向。总体而言，大约60%的回复至少涵盖了真实情况的关键点。然而，仍需改进以减少无关或重复内容，并增强对常见于实际场景中的转录错误的容忍度。此外，我们在实际应用场景中实施了该系统，并收集了演示的真实反馈。我们的研究成果突显了利用LLMs作为会议代理的潜力与挑战，并为其实用应用场景提供了宝贵的见解，有助于减轻会议负担。 

---
# An Analysis for Reasoning Bias of Language Models with Small Initialization 

**Title (ZH)**: 小初始值下语言模型推理偏差的分析 

**Authors**: Junjie Yao, Zhongwang Zhang, Zhi-Qin John Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04375)  

**Abstract**: Transformer-based Large Language Models (LLMs) have revolutionized Natural Language Processing by demonstrating exceptional performance across diverse tasks. This study investigates the impact of the parameter initialization scale on the training behavior and task preferences of LLMs. We discover that smaller initialization scales encourage models to favor reasoning tasks, whereas larger initialization scales lead to a preference for memorization tasks. We validate this reasoning bias via real datasets and meticulously designed anchor functions. Further analysis of initial training dynamics suggests that specific model components, particularly the embedding space and self-attention mechanisms, play pivotal roles in shaping these learning biases. We provide a theoretical framework from the perspective of model training dynamics to explain these phenomena. Additionally, experiments on real-world language tasks corroborate our theoretical insights. This work enhances our understanding of how initialization strategies influence LLM performance on reasoning tasks and offers valuable guidelines for training models. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）通过在各种任务上表现出色，已经彻底改变了自然语言处理领域。本研究考察了参数初始化规模对LLMs的训练行为和任务偏好产生的影响。研究发现，较小的初始化规模促使模型更倾向于处理推理任务，而较大的初始化规模则导致模型偏好记忆任务。我们通过实际数据集和精心设计的锚函数验证了这种推理偏见。进一步分析初始训练动态表明，特定的模型组件，特别是嵌入空间和自我注意机制，在形成这些学习偏见中起着关键作用。我们从模型训练动态的角度提供了一个理论框架来解释这些现象。此外，对实际语言任务的实验验证了我们的理论洞察。本研究增进了我们对初始化策略如何影响LLMs在推理任务上的性能的理解，并提供了有价值的训练模型指导。 

---
# LLMs can be easily Confused by Instructional Distractions 

**Title (ZH)**: 大语言模型可能会被指令性干扰所迷惑。 

**Authors**: Yerin Hwang, Yongil Kim, Jahyun Koo, Taegwan Kang, Hyunkyung Bae, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.04362)  

**Abstract**: Despite the fact that large language models (LLMs) show exceptional skill in instruction following tasks, this strength can turn into a vulnerability when the models are required to disregard certain instructions. Instruction-following tasks typically involve a clear task description and input text containing the target data to be processed. However, when the input itself resembles an instruction, confusion may arise, even if there is explicit prompting to distinguish between the task instruction and the input. We refer to this phenomenon as instructional distraction. In this paper, we introduce a novel benchmark, named DIM-Bench, specifically designed to assess LLMs' performance under instructional distraction. The benchmark categorizes real-world instances of instructional distraction and evaluates LLMs across four instruction tasks: rewriting, proofreading, translation, and style transfer -- alongside five input tasks: reasoning, code generation, mathematical reasoning, bias detection, and question answering. Our experimental results reveal that even the most advanced LLMs are susceptible to instructional distraction, often failing to accurately follow user intent in such cases. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在指令跟随任务中展现出卓越的能力，但在需要模型忽略某些指令时，这一优点可能会变成一种弱点。指令跟随任务通常包含明确的任务描述和包含目标数据的输入文本。然而，当输入本身类似于指令时，即使有明确的提示来区分任务指令和输入，也可能产生混淆。我们称这种现象为指令干扰。本文介绍了一个新的基准测试，名为DIM-Bench，专门用于评估LLMs在指令干扰下的性能。该基准测试对现实世界中的指令干扰实例进行分类，并评估LLMs在四种指令任务（重写、校对、翻译和风格转移）和五种输入任务（推理、代码生成、数学推理、偏见检测和问答）中的表现。实验结果表明，即使是最先进的LLMs，在遇到指令干扰的情况下也容易受到影响，往往不能准确地遵循用户意图。 

---
# Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives 

**Title (ZH)**: 位置：扩展大规模语言模型代理需要使用大规模语言模型原语进行极限分析 

**Authors**: Elliot Meyerson, Xin Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04358)  

**Abstract**: Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents. 

**Abstract (ZH)**: 将复杂问题分解为子问题通常可以使问题更容易且更高效地求解。随着大型语言模型（LLMs）在越来越多的能力方面达到关键可靠性的阈值，人们正逐渐努力将系统分解为一组基于LLM的代理，每个代理可以被分配子任务。然而，即使这种分解是自动化的，也往往是直观的，比如基于人类如何为团队成员分配角色的方式。这些角色分配与最优解有多接近？本文的观点是，需要使用LLM的基本原理来进行渐近分析，以理性考虑此类分解系統的效率，并且通过此类分析得到的见解将为扩展此类系统提供机会。通过将LLM前向传递视为计算成本的最小单位，可以将特定LLM的内部工作与其他LLM如何协同合作解决复杂问题的固有效率区分开来。换句话说，如果我们想将LLM的部署扩大到极限，而不是赋予LLM人性化的属性，就应该使用LLM的基本原理进行渐近分析，以理性考虑和开发将大型问题分解为LLM代理的更强大的方法。 

---
# LLM-ProS: Analyzing Large Language Models' Performance in Competitive Problem Solving 

**Title (ZH)**: LLM-ProS: 分析大语言模型在竞争性问题求解中的性能 

**Authors**: Md Sifat Hossain, Anika Tabassum, Md. Fahim Arefin, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2502.04355)  

**Abstract**: The rapid advancement of large language models has opened new avenues for automating complex problem-solving tasks such as algorithmic coding and competitive programming. This paper introduces a novel evaluation technique, LLM-ProS, to assess the performance of state-of-the-art LLMs on International Collegiate Programming Contest (ICPC) problems. Using a curated dataset of 166 World Finals problems from 2011 to 2024, we benchmark the models' reasoning, accuracy, and efficiency. We evaluate the five models-GPT-4o, Mistral Large, Llama-3.1-405B, and the o1 family, consisting of o1-mini and o1-preview, across critical metrics like correctness, resource utilization, and response calibration. Our results reveal significant differences in the models' abilities to generalize, adapt, and solve novel problems. We also investigated the impact of training methodologies, dataset contamination, and chain-of-thought reasoning on model performance. The findings provide new insights into optimizing LLMs for algorithmic tasks, highlighting both strengths and limitations of current models. 

**Abstract (ZH)**: 大型语言模型的迅速发展为自动化复杂问题解决任务，如算法编码和程序竞赛编程，开启了新的途径。本文介绍了一种新的评估技术——LLM-ProS，用于评估当前最先进的大型语言模型在国际大学生程序设计竞赛（ICPC）问题上的性能。我们使用从2011年到2024年精心制作的166个世界总决赛问题数据集，对模型的推理能力、准确性和效率进行了基准测试。我们评估了五种模型——GPT-4o、Mistral Large、Llama-3.1-405B，以及o1家族模型（包括o1-mini和o1-preview）在正确性、资源利用和响应校准等关键指标上的表现。我们的结果揭示了模型在泛化、适应和解决新颖问题方面的能力存在显著差异。我们还探讨了训练方法、数据集污染和逐步推理对模型性能的影响。这些发现为优化大型语言模型用于算法任务提供了新的见解，同时也突显了当前模型的强项和局限性。 

---
# Reviving The Classics: Active Reward Modeling in Large Language Model Alignment 

**Title (ZH)**: 重振经典：在大型语言模型对齐中应用主动奖励建模 

**Authors**: Yunyi Shen, Hao Sun, Jean-François Ton  

**Link**: [PDF](https://arxiv.org/pdf/2502.04354)  

**Abstract**: Building neural reward models from human preferences is a pivotal component in reinforcement learning from human feedback (RLHF) and large language model alignment research. Given the scarcity and high cost of human annotation, how to select the most informative pairs to annotate is an essential yet challenging open problem. In this work, we highlight the insight that an ideal comparison dataset for reward modeling should balance exploration of the representation space and make informative comparisons between pairs with moderate reward differences. Technically, challenges arise in quantifying the two objectives and efficiently prioritizing the comparisons to be annotated. To address this, we propose the Fisher information-based selection strategies, adapt theories from the classical experimental design literature, and apply them to the final linear layer of the deep neural network-based reward modeling tasks. Empirically, our method demonstrates remarkable performance, high computational efficiency, and stability compared to other selection methods from deep learning and classical statistical literature across multiple open-source LLMs and datasets. Further ablation studies reveal that incorporating cross-prompt comparisons in active reward modeling significantly enhances labeling efficiency, shedding light on the potential for improved annotation strategies in RLHF. 

**Abstract (ZH)**: 从人类偏好中构建神经奖励模型是强化学习从人类反馈（RLHF）和大型语言模型对齐研究中的一个关键组成部分。由于人类标注资源稀缺且成本高昂，如何选择最具信息量的样本对进行标注成为了一个重要而具有挑战性的开放问题。在本文中，我们强调了一个理想的奖励模型数据集应该在探索表示空间的同时，通过对比具有中等奖励差异的样本对来提供信息丰富的比较。技术上，量化这两个目标并高效地优先处理需要标注的比较这对矛盾是一个挑战。为此，我们提出了费舍尔信息基础上的选取策略，借鉴经典实验设计文献中的理论，并将这些理论应用于基于深度神经网络的奖励模型任务的最终线性层。实验结果表明，与来自深度学习和经典统计文献的其他选取方法相比，我们的方法在多个开源大型语言模型和数据集上展现了卓越的表现、高效性和稳定性。进一步的消融研究显示，在主动奖励模型中引入跨提示比较显著提高了标注效率，为RLHF中的改进标注策略提供了新的视角。 

---
# CognArtive: Large Language Models for Automating Art Analysis and Decoding Aesthetic Elements 

**Title (ZH)**: CognArtive：大型语言模型在艺术分析与美学元素解码中的自动化应用 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04353)  

**Abstract**: Art, as a universal language, can be interpreted in diverse ways, with artworks embodying profound meanings and nuances. The advent of Large Language Models (LLMs) and the availability of Multimodal Large Language Models (MLLMs) raise the question of how these transformative models can be used to assess and interpret the artistic elements of artworks. While research has been conducted in this domain, to the best of our knowledge, a deep and detailed understanding of the technical and expressive features of artworks using LLMs has not been explored. In this study, we investigate the automation of a formal art analysis framework to analyze a high-throughput number of artworks rapidly and examine how their patterns evolve over time. We explore how LLMs can decode artistic expressions, visual elements, composition, and techniques, revealing emerging patterns that develop across periods. Finally, we discuss the strengths and limitations of LLMs in this context, emphasizing their ability to process vast quantities of art-related data and generate insightful interpretations. Due to the exhaustive and granular nature of the results, we have developed interactive data visualizations, available online this https URL, to enhance understanding and accessibility. 

**Abstract (ZH)**: 艺术作为一种通用语言，可以被多元解读，艺术品蕴含着深刻的意义和细微之处。大型语言模型（LLMs）和多模态大型语言模型（MLLMs）的出现引发了如何利用这些变革性模型评估和解读艺术品艺术元素的疑问。虽然在此领域已经进行了研究，但据我们所知，使用LLMs探索艺术作品的技术和表现特征的深层和详细的理解尚未被充分研究。在本研究中，我们探讨了自动化形式艺术分析框架的可能性，以迅速分析大量艺术品并检查它们随时间变化的模式。我们研究了如何利用LLMs解码艺术表达、视觉元素、构图和技术，揭示这些元素在不同历史时期的发展模式。最后，我们讨论了在这一背景下LLMs的优势和局限性，强调了它们处理大量与艺术相关数据并生成有见地的解释的能力。由于结果涵盖广泛且细致，我们开发了可互动的数据可视化工具（可在线查看：[此处填写网址]），以增强理解和访问性。 

---
# Investigating the Robustness of Deductive Reasoning with Large Language Models 

**Title (ZH)**: 使用大规模语言模型探究演绎推理的稳健性 

**Authors**: Fabian Hoppe, Filip Ilievski, Jan-Christoph Kalo  

**Link**: [PDF](https://arxiv.org/pdf/2502.04352)  

**Abstract**: Large Language Models (LLMs) have been shown to achieve impressive results for many reasoning-based Natural Language Processing (NLP) tasks, suggesting a degree of deductive reasoning capability. However, it remains unclear to which extent LLMs, in both informal and autoformalisation methods, are robust on logical deduction tasks. Moreover, while many LLM-based deduction methods have been proposed, there is a lack of a systematic study that analyses the impact of their design components. Addressing these two challenges, we propose the first study of the robustness of LLM-based deductive reasoning methods. We devise a framework with two families of perturbations: adversarial noise and counterfactual statements, which jointly generate seven perturbed datasets. We organize the landscape of LLM reasoners according to their reasoning format, formalisation syntax, and feedback for error recovery. The results show that adversarial noise affects autoformalisation, while counterfactual statements influence all approaches. Detailed feedback does not improve overall accuracy despite reducing syntax errors, pointing to the challenge of LLM-based methods to self-correct effectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在许多基于推理的自然语言处理（NLP）任务中展现出令人印象深刻的成果，这表明它们在一定程度上具有演绎推理能力。然而，目前尚不清楚LLMs在非正式和自动形式化方法中对逻辑推理任务有多大的鲁棒性。此外，虽然已经提出了许多基于LLM的推理方法，但缺乏对这些方法设计组件影响的系统性研究。针对这两个挑战，我们提出了第一个关于基于LLM的演绎推理方法鲁棒性的研究。我们设计了一个框架，包含两种类型的扰动：对抗噪声和反事实陈述，这两种扰动联合生成了七个扰动数据集。按照推理格式、形式化语法以及错误恢复反馈，我们组织了LLM推理器的景观图。结果显示，对抗噪声会影响自动形式化，而反事实陈述会影响所有方法。虽然详细的反馈减少了语法错误，但未能提高整体准确率，这表明基于LLM的方法在自我修正方面存在挑战。 

---
# NER4all or Context is All You Need: Using LLMs for low-effort, high-performance NER on historical texts. A humanities informed approach 

**Title (ZH)**: NER4all 或者语境即一切：利用大规模语言模型在历史文本上实现低功耗高效率命名实体识别——一种人文学科导向的方法 

**Authors**: Torsten Hiltmann, Martin Dröge, Nicole Dresselhaus, Till Grallert, Melanie Althage, Paul Bayer, Sophie Eckenstaler, Koray Mendi, Jascha Marijn Schmitz, Philipp Schneider, Wiebke Sczeponik, Anica Skibba  

**Link**: [PDF](https://arxiv.org/pdf/2502.04351)  

**Abstract**: Named entity recognition (NER) is a core task for historical research in automatically establishing all references to people, places, events and the like. Yet, do to the high linguistic and genre diversity of sources, only limited canonisation of spellings, the level of required historical domain knowledge, and the scarcity of annotated training data, established approaches to natural language processing (NLP) have been both extremely expensive and yielded only unsatisfactory results in terms of recall and precision. Our paper introduces a new approach. We demonstrate how readily-available, state-of-the-art LLMs significantly outperform two leading NLP frameworks, spaCy and flair, for NER in historical documents by seven to twentytwo percent higher F1-Scores. Our ablation study shows how providing historical context to the task and a bit of persona modelling that turns focus away from a purely linguistic approach are core to a successful prompting strategy. We also demonstrate that, contrary to our expectations, providing increasing numbers of examples in few-shot approaches does not improve recall or precision below a threshold of 16-shot. In consequence, our approach democratises access to NER for all historians by removing the barrier of scripting languages and computational skills required for established NLP tools and instead leveraging natural language prompts and consumer-grade tools and frontends. 

**Abstract (ZH)**: 命名实体识别（NER）是历史研究中的一个核心任务，通过自动建立对所有人物、地点、事件等的引用，来提供全面的历史信息。然而，由于历史文献在语言和体裁上的高度多样性、拼写标准化的局限性、所需的历史领域知识水平以及标注训练数据的稀缺性，现有的自然语言处理（NLP）方法既昂贵又未能在召回率和精确率方面取得令人满意的结果。我们论文介绍了一种新的方法。我们展示了如何利用现成的、最先进的大规模语言模型（LLM），显著优于spaCy和flair等两个领先的历史文献NER框架，F1分数提高了7%到22%。我们的消融研究表明，为任务提供历史语境，并进行一些角色建模将焦点从纯粹的语言方法转向，是成功提示策略的核心。我们还证明了，与预期相反，逐步增加示例数量在少样本方法中的确在达到16例之前不能提高召回率和精确率。因此，我们的方法通过消除使用现有NLP工具所需的编程语言和计算技能障碍，而是利用自然语言提示和消费级工具及前端，使NER对所有历史学家更加民主化。 

---
# CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance 

**Title (ZH)**: CodeSteer：通过代码/文本引导的符号增强语言模型 

**Authors**: Yongchao Chen, Yilun Hao, Yueying Liu, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.04350)  

**Abstract**: Existing methods fail to effectively steer Large Language Models (LLMs) between textual reasoning and code generation, leaving symbolic computing capabilities underutilized. We introduce CodeSteer, an effective method for guiding LLM code/text generation. We construct a comprehensive benchmark SymBench comprising 37 symbolic tasks with adjustable complexity and also synthesize datasets of 12k multi-round guidance/generation trajectories and 5.5k guidance comparison pairs. We fine-tune the Llama-3-8B model with a newly designed multi-round supervised fine-tuning (SFT) and direct preference optimization (DPO). The resulting model, CodeSteerLLM, augmented with the proposed symbolic and self-answer checkers, effectively guides the code/text generation of larger models. Augmenting GPT-4o with CodeSteer raises its average performance score from 53.3 to 86.4, even outperforming the existing best LLM OpenAI o1 (82.7), o1-preview (74.8), and DeepSeek R1 (76.8) across all 37 tasks (28 seen, 9 unseen). Trained for GPT-4o, CodeSteer demonstrates superior generalizability, providing an average 41.8 performance boost on Claude, Mistral, and GPT-3.5. CodeSteer-guided LLMs fully harness symbolic computing to maintain strong performance on highly complex tasks. Models, Datasets, and Codes are available at this https URL. 

**Abstract (ZH)**: 现有的方法未能有效地引导大语言模型（LLM）在文本推理和代码生成之间进行权衡，导致符号计算能力得不到充分利用。我们引入了CodeSteer，这是一种有效的方法，用于引导LLM的代码/文本生成。我们构建了一个全面的基准SymBench，包含37个可调复杂度的符号任务，并且合成了包含12000个多轮引导/生成轨迹和5500对引导比较对的数据集。我们使用一种新设计的多轮监督微调（SFT）和直接偏好优化（DPO）对Llama-3-8B模型进行了细调。由此产生的模型CodeSteerLLM，结合了提议的符号检查器和自我答案检查器，有效地引导了更大模型的代码/文本生成。将CodeSteer应用于GPT-4o后，其平均性能得分从53.3提高到86.4，甚至在所有37个任务（28个已见任务，9个未见任务）中优于现有的最佳LLM：OpenAI o1（82.7）、o1-preview（74.8）和DeepSeek R1（76.8）。经过GPT-4o训练的CodeSteer展示了更好的泛化能力，在Claude、Mistral和GPT-3.5上分别提供了平均41.8的性能提升。CodeSteer引导下的LLM充分利用了符号计算，能够在复杂任务中保持强劲的性能。相关模型、数据集和代码可在以下链接访问：[此 https URL]。 

---
# Dynamic benchmarking framework for LLM-based conversational data capture 

**Title (ZH)**: 基于大型语言模型的对话数据捕获动态基准框架 

**Authors**: Pietro Alessandro Aluffi, Patrick Zietkiewicz, Marya Bazzi, Matt Arderne, Vladimirs Murevics  

**Link**: [PDF](https://arxiv.org/pdf/2502.04349)  

**Abstract**: The rapid evolution of large language models (LLMs) has transformed conversational agents, enabling complex human-machine interactions. However, evaluation frameworks often focus on single tasks, failing to capture the dynamic nature of multi-turn dialogues. This paper introduces a dynamic benchmarking framework to assess LLM-based conversational agents through interactions with synthetic users. The framework integrates generative agent simulation to evaluate performance on key dimensions: information extraction, context awareness, and adaptive engagement. By simulating various aspects of user behavior, our work provides a scalable, automated, and flexible benchmarking approach. Experimental evaluation - within a loan application use case - demonstrates the framework's effectiveness under one-shot and few-shot extraction conditions. Results show that adaptive strategies improve data extraction accuracy, especially when handling ambiguous responses. Future work will extend its applicability to broader domains and incorporate additional metrics (e.g., conversational coherence, user engagement). This study contributes a structured, scalable approach to evaluating LLM-based conversational agents, facilitating real-world deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进化已经改变了对话代理，使其能够实现复杂的机器-人类交互。然而，现有的评估框架常常局限于单一任务，未能捕捉多轮对话中的动态特性。本文提出了一种动态基准测试框架，该框架通过与合成用户交互来评估基于LLM的对话代理。该框架整合了生成性代理模拟，以评估关键维度上的表现：信息抽取、上下文意识以及适应性参与。通过模拟用户行为的各个方面，我们的研究提供了一种可扩展、自动化且灵活的基准测试方法。在贷款申请用例中的实验评估表明，在一次性和少量样本抽取条件下，该框架的有效性。结果表明，适应性策略能够提高数据抽取准确性，特别是在处理含糊不清的响应时表现尤为明显。未来的工作还将将该框架的适用范围扩展到更广泛的领域，并纳入更多评估指标（例如，对话连贯性、用户参与度）。本研究提供了一种结构化且可扩展的方法来评估基于LLM的对话代理，从而促进其实用部署。 

---
# SCALM: Detecting Bad Practices in Smart Contracts Through LLMs 

**Title (ZH)**: SCALM：通过大规模语言模型检测智能合约中的不良实践 

**Authors**: Zongwei Li, Xiaoqi Li, Wenkai Li, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04347)  

**Abstract**: As the Ethereum platform continues to mature and gain widespread usage, it is crucial to maintain high standards of smart contract writing practices. While bad practices in smart contracts may not directly lead to security issues, they do elevate the risk of encountering problems. Therefore, to understand and avoid these bad practices, this paper introduces the first systematic study of bad practices in smart contracts, delving into over 35 specific issues. Specifically, we propose a large language models (LLMs)-based framework, SCALM. It combines Step-Back Prompting and Retrieval-Augmented Generation (RAG) to identify and address various bad practices effectively. Our extensive experiments using multiple LLMs and datasets have shown that SCALM outperforms existing tools in detecting bad practices in smart contracts. 

**Abstract (ZH)**: 随着以太坊平台的不断成熟和广泛应用，保持高水平的智能合约编写规范至关重要。虽然不良的智能合约编写实践可能不会直接导致安全问题，但它们确实增加了出现各种问题的风险。因此，为了理解和避免这些不良实践，本文首次系统地研究了智能合约中的不良实践，并深入探讨了超过35个具体问题。具体而言，我们提出了一种基于大型语言模型（LLMs）的框架，即SCALM框架。该框架结合了Step-Back提示和检索增强生成（RAG）技术，以有效识别和解决各种不良实践。我们的大量实验使用了多种LLMs和数据集，结果表明SCALM在检测智能合约中的不良实践方面优于现有工具。 

---
# An Annotated Reading of 'The Singer of Tales' in the LLM Era 

**Title (ZH)**: “歌手与故事”在大语言模型时代的一篇注释阅读 

**Authors**: Kush R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2502.05148)  

**Abstract**: The Parry-Lord oral-formulaic theory was a breakthrough in understanding how oral narrative poetry is learned, composed, and transmitted by illiterate bards. In this paper, we provide an annotated reading of the mechanism underlying this theory from the lens of large language models (LLMs) and generative artificial intelligence (AI). We point out the the similarities and differences between oral composition and LLM generation, and comment on the implications to society and AI policy. 

**Abstract (ZH)**: 帕里-洛德口头-公式理论在理解无文字传承人如何学习、创作和传承口头叙事诗方面取得了突破。本文从大规模语言模型（LLM）和生成型人工智能（AI）的视角，对这一理论背后机制进行注释性解读。我们指出了口头创作与LLM生成之间的相似性和差异性，并讨论了这对社会和AI政策的潜在影响。 

---
# Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures 

**Title (ZH)**: 适应性思想图：测试时统一链、树和图结构的自适应推理 

**Authors**: Tushar Pandey, Ara Ghukasyan, Oktay Goktas, Santosh Kumar Radha  

**Link**: [PDF](https://arxiv.org/pdf/2502.05078)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, yet their performance is highly dependent on the prompting strategy and model scale. While reinforcement learning and fine-tuning have been deployed to boost reasoning, these approaches incur substantial computational and data overhead. In this work, we introduce Adaptive Graph of Thoughts (AGoT), a dynamic, graph-based inference framework that enhances LLM reasoning solely at test time. Rather than relying on fixed-step methods like Chain of Thought (CoT) or Tree of Thoughts (ToT), AGoT recursively decomposes complex queries into structured subproblems, forming an dynamic directed acyclic graph (DAG) of interdependent reasoning steps. By selectively expanding only those subproblems that require further analysis, AGoT unifies the strengths of chain, tree, and graph paradigms into a cohesive framework that allocates computation where it is most needed. We validate our approach on diverse benchmarks spanning multi-hop retrieval, scientific reasoning, and mathematical problem-solving, achieving up to 46.2% improvement on scientific reasoning tasks (GPQA) - comparable to gains achieved through computationally intensive reinforcement learning approaches and outperforming state-of-the-art iterative approaches. These results suggest that dynamic decomposition and structured recursion offer a scalable, cost-effective alternative to post-training modifications, paving the way for more robust, general-purpose reasoning in LLMs. 

**Abstract (ZH)**: 大语言模型（LLMs）展示了令人印象深刻的推理能力，但其性能高度依赖于提示策略和模型规模。尽管强化学习和微调已被部署以提升推理能力，但这些方法带来了巨大的计算和数据开销。在本工作中，我们提出了自适应思维图（Adaptive Graph of Thoughts, AGoT），这是一种仅在测试时增强LLM推理能力的动态图基推理框架。AGoT不同于依赖固定步长方法（如思维链CoT或思维树ToT），它递归地将复杂查询分解为结构化的子问题，形成一个动态的有向无环图（DAG），并连接相互依赖的推理步骤。通过仅扩展那些需要进一步分析的子问题，AGoT将链、树和图范式的优点统一到一个协调的框架中，从而在最需要的地方分配计算资源。我们在涵盖多跳检索、科学推理和数学问题解决等多个基准测试上验证了该方法，科学推理任务（GPQA）的性能提升了46.2%，这与通过计算密集型强化学习方法获得的增益相媲美，并且优于最先进的迭代方法。这些结果表明，动态分解和结构化递归为后训练修改提供了一种可扩展且成本效益高的替代方案，开辟了LLMs中更稳健和通用推理的新途径。 

---
# Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking 

**Title (ZH)**: 全面引导的蒙特卡洛树搜索方法用于复杂的信息检索 

**Authors**: Ruiyang Ren, Yuhao Wang, Junyi Li, Jinhao Jiang, Wayne Xin Zhao, Wenjie Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.04751)  

**Abstract**: In the era of vast digital information, the sheer volume and heterogeneity of available information present significant challenges for intricate information seeking. Users frequently face multistep web search tasks that involve navigating vast and varied data sources. This complexity demands every step remains comprehensive, accurate, and relevant. However, traditional search methods often struggle to balance the need for localized precision with the broader context required for holistic understanding, leaving critical facets of intricate queries underexplored. In this paper, we introduce an LLM-based search assistant that adopts a new information seeking paradigm with holistically guided Monte Carlo tree search (HG-MCTS). We reformulate the task as a progressive information collection process with a knowledge memory and unite an adaptive checklist with multi-perspective reward modeling in MCTS. The adaptive checklist provides explicit sub-goals to guide the MCTS process toward comprehensive coverage of complex user queries. Simultaneously, our multi-perspective reward modeling offers both exploration and retrieval rewards, along with progress feedback that tracks completed and remaining sub-goals, refining the checklist as the tree search progresses. By striking a balance between localized tree expansion and global guidance, HG-MCTS reduces redundancy in search paths and ensures that all crucial aspects of an intricate query are properly addressed. Extensive experiments on real-world intricate information seeking tasks demonstrate that HG-MCTS acquires thorough knowledge collections and delivers more accurate final responses compared with existing baselines. 

**Abstract (ZH)**: 在大数据信息时代，大量且异构的信息资源给复杂的检索工作带来了重大挑战。用户经常面临多步骤的网络搜索任务，需要在海量且多样化的数据源中进行导航。这种复杂性要求每一个搜索步骤都必须全面、准确且相关。然而，传统的搜索方法常常难以平衡局部精确性与整体理解所需的广泛背景之间的需求，导致复杂的查询中关键方面被忽视。在本文中，我们提出了一种基于大规模语言模型（LLM）的搜索助手，采用了一种新的以整体引导的蒙特卡洛树搜索（HG-MCTS）为特征的信息检索范式。我们将任务重新构想为一个渐进的信息收集过程，并结合知识记忆、自适应检查列表和多视角奖励建模来统一在MCTS中的应用。自适应检查列表提供明确的子目标来引导MCTS过程，以全面覆盖复杂的用户查询。同时，我们的多视角奖励建模提供了探索和检索奖励，并提供进度反馈以跟踪已完成和剩余的子目标，随着树搜索的进行逐步优化检查列表。通过平衡局部树扩展与全局指导，HG-MCTS减少了搜索路径的冗余，并确保所有关键方面都能得到妥善处理。在实际复杂的检索任务上的广泛实验表明，HG-MCTS能够获取更全面的知识集，并提供比现有基线更准确的最终响应。 

---
# Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research 

**Title (ZH)**: 代理推理：配备工具进行深度研究的大型语言模型推理 

**Authors**: Junde Wu, Jiayuan Zhu, Yuyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04644)  

**Abstract**: We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Unlike conventional LLM-based reasoning approaches, which rely solely on internal inference, Agentic Reasoning dynamically engages web search, code execution, and structured reasoning-context memory to solve complex problems requiring deep research and multi-step logical deduction. Our framework introduces the Mind Map agent, which constructs a structured knowledge graph to track logical relationships, improving deductive reasoning. Additionally, the integration of web-search and coding agents enables real-time retrieval and computational analysis, enhancing reasoning accuracy and decision-making. Evaluations on PhD-level scientific reasoning (GPQA) and domain-specific deep research tasks demonstrate that our approach significantly outperforms existing models, including leading retrieval-augmented generation (RAG) systems and closed-source LLMs. Moreover, our results indicate that agentic reasoning improves expert-level knowledge synthesis, test-time scalability, and structured problem-solving. The code is at: this https URL. 

**Abstract (ZH)**: 我们介绍了意愿性推理（Agentic Reasoning）框架，该框架通过整合外部工具使用代理来增强大型语言模型（LLM）的推理能力。与依赖内部推理的常规LLM推理方法不同，意愿性推理能够动态地利用网络搜索、代码执行和结构化推理上下文记忆，以解决需要深入研究和多步逻辑推理的复杂问题。我们的框架引入了思维导图代理（Mind Map agent），该代理构建了一个结构化的知识图谱，以跟踪逻辑关系，从而提高演绎推理能力。此外，网络搜索和编码代理的集成能够实现实时检索和计算分析，从而提高推理准确性和决策质量。在博士生级别科学推理（GPQA）和特定领域的深度研究任务评估中，我们的方法显著优于现有模型，包括领先的检索增强生成（RAG）系统和闭源的LLM。此外，我们的结果表明，意愿性推理提高了专家级知识综合、测试时的可扩展性和结构化问题解决能力。代码位于：![](this https URL) 

---
# Confidence Elicitation: A New Attack Vector for Large Language Models 

**Title (ZH)**: 自信度提取：大型语言模型的一种新型攻击向量 

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2502.04643)  

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions. 

**Abstract (ZH)**: 深度学习领域的一个基本问题是鲁棒性对抗。随着这些系统的扩展，这些问题依然存在。当前，具有数十亿参数的大型语言模型（LLMs）在面对对抗攻击时的表现与它们早期的小型版本并无二致。然而，威胁模型已经发生了变化。在过去，灰盒访问情况下，即输入嵌入或输出logits/概率对用户可见，可能是合理的。然而，在引入闭源模型后，除了生成的输出外，用户无法获得关于模型的任何信息。这意味着当前的黑盒攻击只能利用最终的预测来判断攻击是否成功。在本研究中，我们探讨并证明了在仅具黑盒访问权限的分类设置中，对抗指导（类似于使用输出概率）的潜力。这正是通过对模型提出置信度并使其响应的方式来实现的。我们实验证明，从当前的LLMs中提取出的置信度是经过校准的真实度量，而不是虚幻的。通过最小化提取出的置信度，我们可以在一定程度上增加分类错误的可能性。我们提出的新范式在两个模型（LLaMA-3-8B-Instruct 和 Mistral-7B-Instruct-V0.3）的三个数据集上取得了有前景的最先进的结果，当我们将这种技术与现有引入单词级替换的硬标签黑盒攻击方法进行比较时，结果尤为显著。 

---
# Training Language Models to Reason Efficiently 

**Title (ZH)**: 训练高效推理的语言模型 

**Authors**: Daman Arora, Andrea Zanette  

**Link**: [PDF](https://arxiv.org/pdf/2502.04463)  

**Abstract**: Scaling model size and training data has led to great advances in the performance of Large Language Models (LLMs). However, the diminishing returns of this approach necessitate alternative methods to improve model capabilities, particularly in tasks requiring advanced reasoning. Large reasoning models, which leverage long chain-of-thoughts, bring unprecedented breakthroughs in problem-solving capabilities but at a substantial deployment cost associated to longer generations. Reducing inference costs is crucial for the economic feasibility, user experience, and environmental sustainability of these models.
In this work, we propose to train large reasoning models to reason efficiently. More precisely, we use reinforcement learning (RL) to train reasoning models to dynamically allocate inference-time compute based on task complexity. Our method incentivizes models to minimize unnecessary computational overhead while maintaining accuracy, thereby achieving substantial efficiency gains. It enables the derivation of a family of reasoning models with varying efficiency levels, controlled via a single hyperparameter. Experiments on two open-weight large reasoning models demonstrate significant reductions in inference cost while preserving most of the accuracy. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

通过扩大模型规模和训练数据，大型语言模型（LLMs）的性能得到了显著提升。然而，这种方法的边际收益逐渐减少，因此需要寻找其他方法来提高模型能力，特别是在需要高级推理任务的领域。通过利用长链推理的大推理模型带来了解决问题能力的前所未有的突破，但这也伴随着较高的部署成本，主要体现在更长的推理生成过程。降低推理成本对于这些模型的经济可行性、用户体验和环境可持续性至关重要。

在本工作中，我们提出了一种高效训练大型推理模型的方法。具体而言，我们利用强化学习（RL）训练推理模型，使其能够根据任务复杂性动态分配推理时的计算资源。该方法通过激励模型最小化不必要的计算开销以保持准确性，从而实现显著的效率提升。该方法还能够生成一系列具有不同效率级别的推理模型，唯一通过一个超参数控制这些变化。实验证明，对于两个开源的大型推理模型，在保持大部分准确性的同时，推理成本显著降低。 

---
# Decoder-Only LLMs are Better Controllers for Diffusion Models 

**Title (ZH)**: 基于学术规范，以下是标题的翻译：

解码器-only的大语言模型是控制扩散模型的更好选择。 

**Authors**: Ziyi Dong, Yao Xiao, Pengxu Wei, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.04412)  

**Abstract**: Groundbreaking advancements in text-to-image generation have recently been achieved with the emergence of diffusion models. These models exhibit a remarkable ability to generate highly artistic and intricately detailed images based on textual prompts. However, obtaining desired generation outcomes often necessitates repetitive trials of manipulating text prompts just like casting spells on a magic mirror, and the reason behind that is the limited capability of semantic understanding inherent in current image generation models. Specifically, existing diffusion models encode the text prompt input with a pre-trained encoder structure, which is usually trained on a limited number of image-caption pairs. The state-of-the-art large language models (LLMs) based on the decoder-only structure have shown a powerful semantic understanding capability as their architectures are more suitable for training on very large-scale unlabeled data. In this work, we propose to enhance text-to-image diffusion models by borrowing the strength of semantic understanding from large language models, and devise a simple yet effective adapter to allow the diffusion models to be compatible with the decoder-only structure. Meanwhile, we also provide a supporting theoretical analysis with various architectures (e.g., encoder-only, encoder-decoder, and decoder-only), and conduct extensive empirical evaluations to verify its effectiveness. The experimental results show that the enhanced models with our adapter module are superior to the stat-of-the-art models in terms of text-to-image generation quality and reliability. 

**Abstract (ZH)**: 近年来，随着扩散模型的出现，文本生成图像方面取得了突破性的进展。这些模型能够根据文本提示生成高度艺术性和精细详细的图像。然而，要获得期望的生成结果通常需要反复调整文本提示，就像在魔镜上施咒一样，这主要是由于现有图像生成模型在语义理解方面的能力有限。具体而言，现有的扩散模型使用预训练的编码器结构来编码文本提示输入，而这种结构通常是在有限数量的图像-标题对上进行训练的。基于解码器结构的最先进大型语言模型（LLMs）显示出了强大的语义理解能力，因为它们的架构更适合大规模无标签数据的训练。在本文中，我们提出了一种通过借用大型语言模型的语义理解能力来增强文本生成图像的扩散模型的方法，并设计了一个简单而有效的适配器，使扩散模型能够兼容解码器结构。同时，我们也提供了对不同架构（如编码器仅结构、编码器-解码器结构和解码器仅结构）的支持性理论分析，并进行了广泛的实证评估以验证其有效性。实验结果表明，通过我们的适配器模块增强的模型，在文本生成图像的质量和可靠性方面优于最先进的模型。 

---
# Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing 

**Title (ZH)**: Mediator：一种基于减少参数冲突和不确定性路由的内存高效的大语言模型融合方法 

**Authors**: Kunfeng Lai, Zhenheng Tang, Xinglin Pan, Peijie Dong, Xiang Liu, Haolan Chen, Li Shen, Bo Li, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04411)  

**Abstract**: Model merging aggregates Large Language Models (LLMs) finetuned on different tasks into a stronger one. However, parameter conflicts between models leads to performance degradation in averaging. While model routing addresses this issue by selecting individual models during inference, it imposes excessive storage and compute costs, and fails to leverage the common knowledge from different models. In this work, we observe that different layers exhibit varying levels of parameter conflicts. Building on this insight, we average layers with minimal parameter conflicts and use a novel task-level expert routing for layers with significant conflicts. To further reduce storage costs, inspired by task arithmetic sparsity, we decouple multiple fine-tuned experts into a dense expert and several sparse experts. Considering the out-of-distribution samples, we select and merge appropriate experts based on the task uncertainty of the input data. We conduct extensive experiments on both LLaMA and Qwen with varying parameter scales, and evaluate on real-world reasoning tasks. Results demonstrate that our method consistently achieves significant performance improvements while requiring less system cost compared to existing methods. 

**Abstract (ZH)**: 将大型语言模型（LLMs）在不同任务上微调后的模型合并为一个更强的模型。然而，模型之间的参数冲突会导致平均性能下降。而通过在推理时选择单个模型进行解决的方法会带来存储和计算成本的过度支出，并且无法充分利用不同模型之间的共性知识。在本工作中，我们观察到不同层之间的参数冲突程度有所不同。基于这一洞察，我们将参数冲突较小的层进行平均，并对参数冲突显著的层采用一种新颖的任务级别专家路由策略。为进一步减少存储成本，我们受到任务算术稀疏性的启发，将多个微调专家拆分为一个密集专家和几个稀疏专家。考虑到分布外的样本，我们根据输入数据的任务不确定性选择并合并适当的专家。我们在LLaMA和Qwen不同参数规模的体系结构上进行了广泛的实验，并在实际推理任务上进行了评估。实验结果表明，与现有方法相比，我们的方法在系统成本更低的情况下，能够实现显著的性能提升。 

---
# Generating Symbolic World Models via Test-time Scaling of Large Language Models 

**Title (ZH)**: 通过测试时扩展大型语言模型生成符号世界模型 

**Authors**: Zhouliang Yu, Yuhuan Yuan, Tim Z. Xiao, Fuxiang Frank Xia, Jie Fu, Ge Zhang, Ge Lin, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04728)  

**Abstract**: Solving complex planning problems requires Large Language Models (LLMs) to explicitly model the state transition to avoid rule violations, comply with constraints, and ensure optimality-a task hindered by the inherent ambiguity of natural language. To overcome such ambiguity, Planning Domain Definition Language (PDDL) is leveraged as a planning abstraction that enables precise and formal state descriptions. With PDDL, we can generate a symbolic world model where classic searching algorithms, such as A*, can be seamlessly applied to find optimal plans. However, directly generating PDDL domains with current LLMs remains an open challenge due to the lack of PDDL training data. To address this challenge, we propose to scale up the test-time computation of LLMs to enhance their PDDL reasoning capabilities, thereby enabling the generation of high-quality PDDL domains. Specifically, we introduce a simple yet effective algorithm, which first employs a Best-of-N sampling approach to improve the quality of the initial solution and then refines the solution in a fine-grained manner with verbalized machine learning. Our method outperforms o1-mini by a considerable margin in the generation of PDDL domain, achieving over 50% success rate on two tasks (i.e., generating PDDL domains from natural language description or PDDL problems). This is done without requiring additional training. By taking advantage of PDDL as state abstraction, our method is able to outperform current state-of-the-art methods on almost all competition-level planning tasks. 

**Abstract (ZH)**: 解决复杂规划问题需要大规模语言模型（LLMs）明确建模状态转换，以避免违反规则、遵守约束并确保最优性——这一任务因自然语言固有的模糊性而受到阻碍。为了克服这种模糊性，我们可以利用规划领域定义语言（PDDL）作为一种规划抽象方法，它能够提供精确和正式的状态描述。借助PDDL，我们可以生成一个符号世界模型，在其中经典的搜索算法（如A*）可以无缝应用以找到最优计划。然而，当前LLM直接生成PDDL域仍然是一个开放的挑战，因为缺乏足够的PDDL训练数据。为了解决这一挑战，我们提出扩展LLM在测试阶段的计算，以增强其PDDL推理能力，从而能够生成高质量的PDDL域。具体来说，我们引入了一个简单而有效的算法，该算法首先采用“最佳N次采样”方法以提高初始解的质量，然后以细粒度的方式通过口语化的机器学习来进一步改进解。我们的方法在生成PDDL域方面显著优于o1-mini，在两个任务（即从自然语言描述或PDDL问题生成PDDL域）中成功率达到50%以上。这一结果无需额外训练即可实现。通过利用PDDL作为状态抽象，我们的方法在几乎所有竞赛级别的规划任务中均能超越当前最先进的方法。 

---
# Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization 

**Title (ZH)**: 使用迭代潜在空间策略优化学习狼人杀中的战略语言代理 

**Authors**: Zelai Xu, Wanjun Gu, Chao Yu, Yi Wu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04686)  

**Abstract**: Large language model (LLM)-based agents have recently shown impressive progress in a variety of domains, including open-ended conversation and multi-step decision-making. However, applying these agents to social deduction games such as Werewolf, which requires both strategic decision-making and free-form language interaction, remains non-trivial. Traditional methods based on Counterfactual Regret Minimization (CFR) or reinforcement learning (RL) typically depend on a predefined action space, making them unsuitable for language games with unconstrained text action space. Meanwhile, pure LLM-based agents often suffer from intrinsic biases and require prohibitively large datasets for fine-tuning. We propose Latent Space Policy Optimization (LSPO), an iterative framework that addresses these challenges by first mapping free-form text to a discrete latent space, where methods like CFR and RL can learn strategic policy more effectively. We then translate the learned policy back into natural language dialogues, which are used to fine-tune an LLM via Direct Preference Optimization (DPO). By iteratively alternating between these stages, our LSPO agent progressively enhances both strategic reasoning and language communication. Experiment results on the Werewolf game show that our method improves the agent's performance in each iteration and outperforms existing Werewolf agents, underscoring its promise for free-form language decision-making. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的智能体在多个领域中已展现出了显著的进步，包括开放式对话和多步决策。然而，将这些智能体应用于像狼人杀（Werewolf）这样的社会推理游戏仍然具有挑战性，因为这类游戏要求既进行战略决策又进行自然语言交互。传统的基于Counterfactual Regret Minimization（CFR）或强化学习（RL）的方法通常依赖于预先定义的动作空间，这使得它们不适合语言游戏中不受限制的文本动作空间。同时，纯粹基于LLM的智能体往往存在固有的偏差，需要巨大规模的数据集进行微调。为此，我们提出了一种潜空间策略优化（LSPO）框架，该框架通过首先将自然语言文本映射到一个离散的潜空间，在该空间中，CFR和RL方法可以更有效地学习策略。然后将学到的策略转换回自然语言对话，并通过直接偏好优化（DPO）微调大规模语言模型（LLM）。通过迭代交替这些阶段，我们的LSPO智能体逐步增强其战略推理和语言交流能力。在狼人杀游戏中进行的实验结果表明，我们的方法在每次迭代中都能提高智能体的表现，并且优于现有的狼人杀智能体，这表明了其在自由形式语言决策中的潜力。 

---
# "It Felt Like I Was Left in the Dark": Exploring Information Needs and Design Opportunities for Family Caregivers of Older Adult Patients in Critical Care Settings 

**Title (ZH)**: “感觉自己像是在黑暗中”: 探索重症护理环境中老年患者家属护理者的信息需求及设计机会 

**Authors**: Shihan Fu, Bingsheng Yao, Smit Desai, Yuqi Hu, Yuling Sun, Samantha Stonbraker, Yanjun Gao, Elizabeth M. Goldberg, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05115)  

**Abstract**: Older adult patients constitute a rapidly growing subgroup of Intensive Care Unit (ICU) patients. In these situations, their family caregivers are expected to represent the unconscious patients to access and interpret patients' medical information. However, caregivers currently have to rely on overloaded clinicians for information updates and typically lack the health literacy to understand complex medical information. Our project aims to explore the information needs of caregivers of ICU older adult patients, from which we can propose design opportunities to guide future AI systems. The project begins with formative interviews with 11 caregivers to identify their challenges in accessing and interpreting medical information; From these findings, we then synthesize design requirements and propose an AI system prototype to cope with caregivers' challenges. The system prototype has two key features: a timeline visualization to show the AI extracted and summarized older adult patients' key medical events; and an LLM-based chatbot to provide context-aware informational support. We conclude our paper by reporting on the follow-up user evaluation of the system and discussing future AI-based systems for ICU caregivers of older adults. 

**Abstract (ZH)**: 老年患者构成了重症监护室（ICU）患者中一个快速增长的子群体。在这种情况下，护理人员需要代表无意识的患者访问和解读医疗信息。然而，护理人员目前不得不依赖负担过重的临床医生来获得信息更新，通常缺乏足够的健康素养以理解复杂的医疗信息。本项目旨在探索ICU老年患者护理人员的信息需求，从而为我们提出指导未来AI系统的设汁机会。该项目从对11名护理人员的形成性访谈开始，以识别他们在访问和解读医疗信息方面遇到的挑战；基于这些发现，我们随后总结出设计需求，并提出一种AI系统原型来应对护理人员的挑战。该系统原型具有两个关键特征：通过时间轴可视化展示AI提取和总结的老年患者关键医疗事件；以及基于LLM的聊天机器人，提供基于上下文的信息支持。最后，我们在论文中报告了对系统的后续用户评估，并讨论了未来针对老年患者ICU护理人员的AI系统。 

---
# Causality can systematically address the monsters under the bench(marks) 

**Title (ZH)**: 因果关系可以系统地应对隐藏在评分背后的怪兽 

**Authors**: Felix Leeb, Zhijing Jin, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2502.05085)  

**Abstract**: Effective and reliable evaluation is essential for advancing empirical machine learning. However, the increasing accessibility of generalist models and the progress towards ever more complex, high-level tasks make systematic evaluation more challenging. Benchmarks are plagued by various biases, artifacts, or leakage, while models may behave unreliably due to poorly explored failure modes. Haphazard treatments and inconsistent formulations of such "monsters" can contribute to a duplication of efforts, a lack of trust in results, and unsupported inferences. In this position paper, we argue causality offers an ideal framework to systematically address these challenges. By making causal assumptions in an approach explicit, we can faithfully model phenomena, formulate testable hypotheses with explanatory power, and leverage principled tools for analysis. To make causal model design more accessible, we identify several useful Common Abstract Topologies (CATs) in causal graphs which help gain insight into the reasoning abilities in large language models. Through a series of case studies, we demonstrate how the precise yet pragmatic language of causality clarifies the strengths and limitations of a method and inspires new approaches for systematic progress. 

**Abstract (ZH)**: 有效的且可靠的评估对于推动实证机器学习的进步至关重要。然而，通用模型的日益普及以及向日益复杂和高层次任务的进步使系统的评估变得更加具有挑战性。基准测试饱受各种偏见、伪像或泄露之苦，而模型由于未充分探索的失败模式可能会表现出不可靠的行为。随便的处理方式和对这些“怪兽”的不一致表述可能会导致重复努力、对结果的信任缺失以及缺乏支持的推断。在本文中，我们认为因果推断提供了一个理想的框架，可以系统地解决这些挑战。通过使方法中的因果假设显式化，我们可以忠实建模现象、提出具有解释力的可检验假设，并利用原则性的分析工具。为了使因果模型设计更加易于理解，我们识别了因果图中几种有用的通用抽象拓扑（CATs），这些拓扑有助于理解大型语言模型的推理能力。通过一系列案例研究，我们展示了精确且实用的因果语言如何澄清方法的优势和局限性，并激发新的系统进展方法。 

---
# Enhancing Phishing Email Identification with Large Language Models 

**Title (ZH)**: 使用大规模语言模型增强钓鱼邮件的识别能力 

**Authors**: Catherine Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.04759)  

**Abstract**: Phishing has long been a common tactic used by cybercriminals and continues to pose a significant threat in today's digital world. When phishing attacks become more advanced and sophisticated, there is an increasing need for effective methods to detect and prevent them. To address the challenging problem of detecting phishing emails, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. In this work, we take steps to study the efficacy of large language models (LLMs) in detecting phishing emails. The experiments show that the LLM achieves a high accuracy rate at high precision; importantly, it also provides interpretable evidence for the decisions. 

**Abstract (ZH)**: 网络钓鱼历来是网络罪犯常用的手段，并且在当今的数字世界中仍然构成了重大威胁。当网络钓鱼攻击变得更加先进和复杂时，需要更有效的检测和预防方法。为应对检测网络钓鱼邮件这一具有挑战性的问题，研究人员开发了多种解决方案，特别是基于机器学习（ML）算法的方法。在本研究中，我们探讨了大规模语言模型（LLMs）在检测网络钓鱼邮件方面的有效性。实验结果表明，LLM在高精度的情况下实现了较高的准确率；更重要的是，它还提供了可解释的证据支持决策。 

---
# Unveiling the Mechanisms of Explicit CoT Training: How Chain-of-Thought Enhances Reasoning Generalization 

**Title (ZH)**: 揭示显式CoT训练机制：链式思考如何增强推理泛化能力 

**Authors**: Xinhao Yao, Ruifeng Ren, Yun Liao, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04667)  

**Abstract**: Training large language models (LLMs) with high-quality Chain-of-Thought (CoT) annotations has become a widely adopted strategy due to its significant enhancement of reasoning capabilities. To fully comprehend this approach, two questions naturally arise: (Q1) What advantages does training with CoT offer compared to training without CoT? (Q2) If there are advantages, what are the underlying mechanisms of explicit CoT training? Analyzing the advantages and mechanisms of CoT training is challenging due to the many factors involved. To address this, we conduct a detailed analysis using clear and controllable data distributions and, for the first time, reveal that CoT training offers the following advantages: (1) Training with CoT markedly improves reasoning generalization, extending it from in-distribution (ID) to both ID and out-of-distribution (OOD) scenarios, while also speeding up convergence; (2) Even when training with CoT includes a certain range of erroneous reasoning steps, it still enables the model to learn reasoning patterns, leading to systematic generalization. We further explore the underlying mechanisms from a circuit perspective: (1) The data distribution (e.g., ratio $\lambda$ and pattern) plays a crucial role in influencing the model's systematic generalization; (2) CoT training (with two-hop facts) internalizes reasoning into a two-stage generalizing circuit, where the number of stages corresponds to the explicit reasoning steps during training. Our findings elucidate the mechanisms underlying explicit CoT training and offer critical insights into tuning strategies for LLMs to achieve robust generalization. 

**Abstract (ZH)**: 使用高质量的链式思维（CoT，Chain-of-Thought）标注训练大型语言模型（LLMs）已成为一种广泛采用的策略，因为这显著提高了模型的推理能力。为了彻底理解这一方法，自然会产生两个问题：（Q1）与不使用CoT标注相比，使用CoT标注训练有哪些优势？（Q2）如果存在优势，显式CoT训练背后的机制是什么？由于涉及多种因素，分析CoT训练的优势与机制具有挑战性。为了解决这一问题，我们采用清晰可控的数据分布进行了详细分析，并首次揭示了CoT训练提供了以下优势：（1）使用CoT标注训练显著改善了推理泛化能力，使其从同分布（ID）拓展到同分布和非同分布（OOD）场景，同时加快了模型的收敛速度；（2）即使CoT标注训练中包含一定范围的错误推理步骤，模型仍然能够学习推理模式，从而实现系统性的泛化。我们还从电路的角度进一步探讨了这些机制：（1）数据分布（例如比例$\lambda$和模式）对模型系统的泛化起着关键作用；（2）使用CoT训练（涉及两跳事实）将推理内化为两阶段泛化电路中，阶段的数量对应于训练中的显式推理步骤。我们的研究结果阐明了显式CoT训练的机制，并提供了有关调整策略以实现LLMs稳健泛化的关键见解。 

---
# WaferLLM: A Wafer-Scale LLM Inference System 

**Title (ZH)**: WaferLLM：一种晶圆规模的大规模语言模型推理系统 

**Authors**: Congjie He, Yeqi Huang, Pei Mu, Ziming Miao, Jilong Xue, Lingxiao Ma, Fan Yang, Luo Mai  

**Link**: [PDF](https://arxiv.org/pdf/2502.04563)  

**Abstract**: Emerging AI accelerators increasingly adopt wafer-scale manufacturing technologies, integrating hundreds of thousands of AI cores in a mesh-based architecture with large distributed on-chip memory (tens of GB in total) and ultra-high on-chip memory bandwidth (tens of PB/s). However, current LLM inference systems, optimized for shared memory architectures like GPUs, fail to fully exploit these accelerators. We introduce WaferLLM, the first wafer-scale LLM inference system. WaferLLM is guided by a novel PLMR device model that captures the unique hardware characteristics of wafer-scale architectures. Leveraging this model, WaferLLM pioneers wafer-scale LLM parallelism, optimizing the utilization of hundreds of thousands of on-chip cores. It also introduces MeshGEMM and MeshGEMV, the first GEMM and GEMV implementations designed to scale effectively on wafer-scale accelerators. Evaluations show that WaferLLM achieves 200$\times$ better wafer-scale accelerator utilization than state-of-the-art systems. On a commodity wafer-scale accelerator, WaferLLM delivers 606$\times$ faster and 22$\times$ more energy-efficient GEMV compared to an advanced GPU. For LLMs, WaferLLM enables 39$\times$ faster decoding with 1.7$\times$ better energy efficiency. We anticipate these numbers will grow significantly as wafer-scale AI models, software, and hardware continue to mature. 

**Abstract (ZH)**: 新兴的AI加速器越来越多地采用晶圆级制造技术，在基于网格的架构中集成了数十万个AI核心，并具备大规模分布式片上内存（总计 tens of GB）和超高的片上内存带宽（tens of PB/s）。然而，当前针对共享内存架构（如GPU）优化的大语言模型（LLM）推理系统未能充分利用这些加速器。我们引入了WaferLLM，这是首个晶圆级大语言模型推理系统。WaferLLM基于一种新颖的PLMR设备模型，该模型捕捉了晶圆级架构的独特硬件特性。借助这一模型，WaferLLM开创了晶圆级LLM并行性，优化了数十万个片上核心的利用率。它还引入了MeshGEMM和MeshGEMV，这是首次针对晶圆级加速器有效扩展设计的GEMM和GEMV实现。评估结果表明，WaferLLM的晶圆级加速器利用率比最先进的系统高出200倍。在一种商用晶圆级加速器上，WaferLLM的GEMV性能比先进的GPU快606倍，且能源效率高22倍。对于大语言模型，WaferLLM的解码速度提高了39倍，同时能源效率提高了1.7倍。我们预期随着晶圆级AI模型、软件和硬件的不断成熟，这些数字将显著增长。 

---
# ADIFF: Explaining audio difference using natural language 

**Title (ZH)**: ADIFF：使用自然语言解释音频差异 

**Authors**: Soham Deshmukh, Shuo Han, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2502.04476)  

**Abstract**: Understanding and explaining differences between audio recordings is crucial for fields like audio forensics, quality assessment, and audio generation. This involves identifying and describing audio events, acoustic scenes, signal characteristics, and their emotional impact on listeners. This paper stands out as the first work to comprehensively study the task of explaining audio differences and then propose benchmark, baselines for the task. First, we present two new datasets for audio difference explanation derived from the AudioCaps and Clotho audio captioning datasets. Using Large Language Models (LLMs), we generate three levels of difference explanations: (1) concise descriptions of audio events and objects, (2) brief sentences about audio events, acoustic scenes, and signal properties, and (3) comprehensive explanations that include semantics and listener emotions. For the baseline, we use prefix tuning where audio embeddings from two audio files are used to prompt a frozen language model. Our empirical analysis and ablation studies reveal that the naive baseline struggles to distinguish perceptually similar sounds and generate detailed tier 3 explanations. To address these limitations, we propose ADIFF, which introduces a cross-projection module, position captioning, and a three-step training process to enhance the model's ability to produce detailed explanations. We evaluate our model using objective metrics and human evaluation and show our model enhancements lead to significant improvements in performance over naive baseline and SoTA Audio-Language Model (ALM) Qwen Audio. Lastly, we conduct multiple ablation studies to study the effects of cross-projection, language model parameters, position captioning, third stage fine-tuning, and present our findings. Our benchmarks, findings, and strong baseline pave the way for nuanced and human-like explanations of audio differences. 

**Abstract (ZH)**: 理解并解释音频记录之间的差异在音频取证、质量评估和音频生成等领域至关重要。这包括识别和描述音频事件、声景、信号特征及其对听众的情感影响。本文是首个全面研究解释音频差异任务的论文，并为此任务提出了基准和基线。首先，我们提出了两个新的数据集，用于音频差异解释，这些数据集分别源自AudioCaps和Clotho音频描述数据集。利用大型语言模型（LLMs），我们生成了三个层次的差异解释：（1）简洁的音频事件和对象描述，（2）简要的关于音频事件、声景和信号属性的句子，以及（3）全面的解释，包括语义和听众情感。对于基线，我们使用前缀调谐方法，在两个音频文件的嵌入向量提示下冻结的语言模型。我们的实证分析和消融研究揭示，朴素基线在区分感知相似的声音和生成详尽的第三级解释方面遇到困难。为了解决这些局限性，我们提出了ADIFF，引入了交叉投影模块、位置描述和三个训练步骤，以增强模型产生详细解释的能力。我们使用客观指标和人类评估来评估我们的模型，并展示了我们的模型改进显著提升了性能，优于朴素基线和最先进的音频-语言模型（ALM）Qwen Audio。最后，我们进行了多项消融研究，探讨了交叉投影、语言模型参数、位置描述、第三阶段微调的影响，并展示了我们的发现。我们的基准、发现和强大的基线为更细致和类似人类的音频差异解释铺平了道路。 

---
# Decoding AI Judgment: How LLMs Assess News Credibility and Bias 

**Title (ZH)**: 解码AI判断：LLMs如何评估新闻的可信度和偏见 

**Authors**: Edoardo Loru, Jacopo Nudo, Niccolò Di Marco, Matteo Cinelli, Walter Quattrociocchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04426)  

**Abstract**: Large Language Models (LLMs) are increasingly used to assess news credibility, yet little is known about how they make these judgments. While prior research has examined political bias in LLM outputs or their potential for automated fact-checking, their internal evaluation processes remain largely unexamined. Understanding how LLMs assess credibility provides insights into AI behavior and how credibility is structured and applied in large-scale language models. This study benchmarks the reliability and political classifications of state-of-the-art LLMs - Gemini 1.5 Flash (Google), GPT-4o mini (OpenAI), and LLaMA 3.1 (Meta) - against structured, expert-driven rating systems such as NewsGuard and Media Bias Fact Check. Beyond assessing classification performance, we analyze the linguistic markers that shape LLM decisions, identifying which words and concepts drive their evaluations. We uncover patterns in how LLMs associate credibility with specific linguistic features by examining keyword frequency, contextual determinants, and rank distributions. Beyond static classification, we introduce a framework in which LLMs refine their credibility assessments by retrieving external information, querying other models, and adapting their responses. This allows us to investigate whether their assessments reflect structured reasoning or rely primarily on prior learned associations. 

**Abstract (ZH)**: 大规模语言模型（LLM）越来越多地用于评估新闻可信度，但人们对它们如何进行这些判断知之甚少。尽管先前的研究已经考察了LLM输出中的政治偏见或它们作为自动化事实核查工具的潜力，但它们的内部评估过程仍然鲜有探讨。理解LLM如何评估可信度可以提供关于人工智能行为、以及可信度如何在大规模语言模型中结构化和应用的见解。本研究基于结构化的、由专家驱动的评级系统（如NewsGuard和Media Bias Fact Check），对最新的LLM——谷歌的Gemini 1.5 Flash、OpenAI的GPT-4o mini以及Meta的LLaMA 3.1进行基准测试。除了评估分类性能外，我们还分析了影响LLM决策的语言标志，确定哪些词汇和概念影响它们的评估。我们通过研究关键词频率、上下文因子以及排名分布，揭示了LLM如何将可信度与特定语言特征联系起来的模式。在静态分类之外，我们提出了一种框架，使LLM能够通过检索外部信息、查询其他模型并调整其响应来细化其可信度评估。这使我们能够探究它们的评估是否反映了有结构的推理过程，还是主要依赖于先前学习的关联。 

---
# Assessing and Prioritizing Ransomware Risk Based on Historical Victim Data 

**Title (ZH)**: 基于历史受害数据评估与优先级排序勒索软件风险 

**Authors**: Spencer Massengale, Philip Huff  

**Link**: [PDF](https://arxiv.org/pdf/2502.04421)  

**Abstract**: We present an approach to identifying which ransomware adversaries are most likely to target specific entities, thereby assisting these entities in formulating better protection strategies. Ransomware poses a formidable cybersecurity threat characterized by profit-driven motives, a complex underlying economy supporting criminal syndicates, and the overt nature of its attacks. This type of malware has consistently ranked among the most prevalent, with a rapid escalation in activity observed. Recent estimates indicate that approximately two-thirds of organizations experienced ransomware attacks in 2023 \cite{Sophos2023Ransomware}. A central tactic in ransomware campaigns is publicizing attacks to coerce victims into paying ransoms. Our study utilizes public disclosures from ransomware victims to predict the likelihood of an entity being targeted by a specific ransomware variant. We employ a Large Language Model (LLM) architecture that uses a unique chain-of-thought, multi-shot prompt methodology to define adversary SKRAM (Skills, Knowledge, Resources, Authorities, and Motivation) profiles from ransomware bulletins, threat reports, and news items. This analysis is enriched with publicly available victim data and is further enhanced by a heuristic for generating synthetic data that reflects victim profiles. Our work culminates in the development of a machine learning model that assists organizations in prioritizing ransomware threats and formulating defenses based on the tactics, techniques, and procedures (TTP) of the most likely attackers. 

**Abstract (ZH)**: 我们提出了一种方法，用于识别哪些勒索软件对手最有可能针对特定实体，从而帮助这些实体制定更有针对性的保护策略。勒索软件是一种以获利为目标、支持犯罪团伙的复杂经济体系，并且其攻击手段公开的网络安全威胁。这种类型的恶意软件一直位居最受关注的类型之列，并且近年来活动水平急剧上升。近期的估计显示，2023年大约三分之二的组织遭受了勒索软件攻击 \cite{Sophos2023Ransomware}。勒索软件竞选活动的核心策略之一是公布攻击信息以迫使受害者支付赎金。我们的研究利用了勒索软件受害者的公开披露信息，来预测特定勒索软件变种最有可能针对哪些实体。我们采用了大型语言模型（LLM）架构，使用独特的多轮提示方法，从勒索软件公告、威胁报告和新闻报道中定义勒索软件威胁者（Skills, Knowledge, Resources, Authorities, and Motivation，即技能、知识、资源、职权和动机）档案。这一分析使用了可公开获取的受害者数据，并通过生成反映受害者档案的启发式合成数据进一步增强。我们的研究最终形成了一个机器学习模型，帮助组织优先处理勒索软件威胁，并基于最可能的攻击者的技术、战术和程序（TTP）制定防御措施。 

---
# Limitations of Large Language Models in Clinical Problem-Solving Arising from Inflexible Reasoning 

**Title (ZH)**: 大型语言模型在临床问题解决中僵化推理导致的局限性 

**Authors**: Jonathan Kim, Anna Podlasek, Kie Shidara, Feng Liu, Ahmed Alaa, Danilo Bernardo  

**Link**: [PDF](https://arxiv.org/pdf/2502.04381)  

**Abstract**: Large Language Models (LLMs) have attained human-level accuracy on medical question-answer (QA) benchmarks. However, their limitations in navigating open-ended clinical scenarios have recently been shown, raising concerns about the robustness and generalizability of LLM reasoning across diverse, real-world medical tasks. To probe potential LLM failure modes in clinical problem-solving, we present the medical abstraction and reasoning corpus (M-ARC). M-ARC assesses clinical reasoning through scenarios designed to exploit the Einstellung effect -- the fixation of thought arising from prior experience, targeting LLM inductive biases toward inflexible pattern matching from their training data rather than engaging in flexible reasoning. We find that LLMs, including current state-of-the-art o1 and Gemini models, perform poorly compared to physicians on M-ARC, often demonstrating lack of commonsense medical reasoning and a propensity to hallucinate. In addition, uncertainty estimation analyses indicate that LLMs exhibit overconfidence in their answers, despite their limited accuracy. The failure modes revealed by M-ARC in LLM medical reasoning underscore the need to exercise caution when deploying these models in clinical settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在医学问答（QA）基准上达到了与人类相当的准确性。然而，最近的研究表明，它们在处理开放性的临床场景时存在局限性，这引发了对其在各种实际医疗任务中的鲁棒性和泛化能力的担忧。为了探究临床问题解决中大型语言模型潜在的失败模式，我们提出了医学抽象与推理语料库（M-ARC）。M-ARC 通过设计旨在利用 Einstellung 效应（即因先前经验导致的思维定势）的情景，来评估临床推理能力，这些情景目标在于使大型语言模型在其训练数据中倾向于僵化的模式匹配，而不是进行灵活的推理。我们发现，包括当前最先进的 o1 和 Gemini 模型在内的大型语言模型在 M-ARC 上的表现不及医生，经常表现出缺乏常识性的医学推理能力和产生幻觉的倾向。此外，不确定性估计分析表明，尽管其准确性有限，但大型语言模型在其答案上表现出过度自信。M-ARC 对大型语言模型医学推理中揭示的失败模式强调了在临床环境中部署这些模型时需要谨慎。 

---
# Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data 

**Title (ZH)**: 多样性作为一种奖励：在混合未指定域的数据上微调大型语言模型 

**Authors**: Zhenqing Ling, Daoyuan Chen, Liuyi Yao, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04380)  

**Abstract**: Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this paper, we study the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations for both inter- and intra-diversity. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-development for LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过多样化数据集进行微调对于提升其在各种领域的整体性能至关重要。在实际应用场景中，现有的基于数据组成比例建模的方法往往难以处理缺少领域标签、标签不精确或未标准化的数据，而基于数据选择的方法通常难以平衡多领域的性能。为解决这些挑战，本文通过实证构建对比数据池并从理论上解释了跨域和内在多样性的作用，研究了多样化数据在提升LLMs总体能力中的角色。基于这些洞察，我们提出了一种新的方法，为LLM赋予双重身份：输出模型用于基于多样性的奖励认知探究和选择数据，以及输入模型用于使用所选数据进行调整。大量的实验表明，当应用于各种先进的LLMs时，该方法显著提升了领域未确定数据以及一系列基础下游任务的性能。我们发布了代码，希望本研究能够促进对数据多样性的理解，并推动反馈驱动的数据-模型共同开发方法的发展。 

---
