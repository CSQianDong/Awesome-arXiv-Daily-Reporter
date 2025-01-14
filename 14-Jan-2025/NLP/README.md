# WebWalker: Benchmarking LLMs in Web Traversal 

**Title (ZH)**: WebWalker: 评估大型语言模型在网页遍历中的性能 

**Authors**: Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Deyu Zhou, Pengjun Xie, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07572)  

**Abstract**: Retrieval-augmented generation (RAG) demonstrates remarkable performance across tasks in open-domain question-answering. However, traditional search engines may retrieve shallow content, limiting the ability of LLMs to handle complex, multi-layered information. To address it, we introduce WebWalkerQA, a benchmark designed to assess the ability of LLMs to perform web traversal. It evaluates the capacity of LLMs to traverse a website's subpages to extract high-quality data systematically. We propose WebWalker, which is a multi-agent framework that mimics human-like web navigation through an explore-critic paradigm. Extensive experimental results show that WebWalkerQA is challenging and demonstrates the effectiveness of RAG combined with WebWalker, through the horizontal and vertical integration in real-world scenarios. 

**Abstract (ZH)**: 检索增强生成（RAG）在开放领域的问题回答任务中表现出色。然而，传统的搜索引擎可能会检索到表面信息，限制了大语言模型（LLM）处理复杂、多层次信息的能力。为了解决这个问题，我们引入了WebWalkerQA基准，旨在评估LLM在进行网页遍历方面的能力。该基准评估LLM系统从一个网站的子页面中系统地提取高质量数据的能力。我们提出了WebWalker框架，该框架通过探索-批评范式模仿人类的网页导航行为。广泛的实验结果表明，WebWalkerQA具有挑战性，并且证明了RAG与WebWalker结合在实际场景中的有效性，通过水平和垂直集成展示了其实用效果。 

---
# Imagine while Reasoning in Space: Multimodal Visualization-of-Thought 

**Title (ZH)**: 在空间推理中的多模态可视化思维：Imagine while Reasoning in Space 

**Authors**: Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulić, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.07542)  

**Abstract**: Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Yet, it struggles in complex spatial reasoning tasks. Nonetheless, human cognition extends beyond language alone, enabling the remarkable capability to think in both words and images. Inspired by this mechanism, we propose a new reasoning paradigm, Multimodal Visualization-of-Thought (MVoT). It enables visual thinking in MLLMs by generating image visualizations of their reasoning traces. To ensure high-quality visualization, we introduce token discrepancy loss into autoregressive MLLMs. This innovation significantly improves both visual coherence and fidelity. We validate this approach through several dynamic spatial reasoning tasks. Experimental results reveal that MVoT demonstrates competitive performance across tasks. Moreover, it exhibits robust and reliable improvements in the most challenging scenarios where CoT fails. Ultimately, MVoT establishes new possibilities for complex reasoning tasks where visual thinking can effectively complement verbal reasoning. 

**Abstract (ZH)**: 链式推理（CoT）提示在增强大规模语言模型（LLMs）和多模态大规模语言模型（MLLMs）的复杂推理能力方面已被证明非常有效。然而，它在复杂的空间推理任务中表现不佳。尽管如此，人类认知不仅依赖语言，还能够同时进行语言和图像的思维活动。受这一机制的启发，我们提出了一种新的推理范式——多模态思维可视化的（MVoT）。通过生成图像化的推理痕迹，MVoT允许MLLMs进行图像思维。为了保证高质量的可视化效果，我们引入了标记差异损失（token discrepancy loss）到自回归MLLMs中。这一创新显著提高了图像的连贯性和准确性。通过多种动态空间推理任务的验证，实验结果表明，MVoT在多个任务中表现出竞争力。此外，在CoT失败的最具有挑战性的场景中，MVoT也表现出稳健可靠的改进。最终，MVoT为复杂推理任务开辟了新的可能性，在这些任务中，图像思维能够有效地补充言语推理。 

---
# Investigating Large Language Models in Inferring Personality Traits from User Conversations 

**Title (ZH)**: 探究大型语言模型在从用户对话中推断人格特质方面的应用 

**Authors**: Jianfeng Zhu, Ruoming Jin, Karin G. Coifman  

**Link**: [PDF](https://arxiv.org/pdf/2501.07532)  

**Abstract**: Large Language Models (LLMs) are demonstrating remarkable human like capabilities across diverse domains, including psychological assessment. This study evaluates whether LLMs, specifically GPT-4o and GPT-4o mini, can infer Big Five personality traits and generate Big Five Inventory-10 (BFI-10) item scores from user conversations under zero-shot prompting conditions. Our findings reveal that incorporating an intermediate step--prompting for BFI-10 item scores before calculating traits--enhances accuracy and aligns more closely with the gold standard than direct trait inference. This structured approach underscores the importance of leveraging psychological frameworks in improving predictive precision. Additionally, a group comparison based on depressive symptom presence revealed differential model performance. Participants were categorized into two groups: those experiencing at least one depressive symptom and those without symptoms. GPT-4o mini demonstrated heightened sensitivity to depression-related shifts in traits such as Neuroticism and Conscientiousness within the symptom-present group, whereas GPT-4o exhibited strengths in nuanced interpretation across groups. These findings underscore the potential of LLMs to analyze real-world psychological data effectively, offering a valuable foundation for interdisciplinary research at the intersection of artificial intelligence and psychology. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多个领域内展示了令人惊叹的人类化能力，包括心理评估。本研究评估了在零样本推理条件下，LLMs，特别是GPT-4o和GPT-4o mini，能否从用户对话中推断大五人格特质并生成大五人格量表-10（BFI-10）项目分数。研究发现，在直接推断特质之前增加一个中间步骤——先提示生成BFI-10项目分数——可提高准确性和与黄金标准的一致性。这种结构化的方法突显了利用心理学框架以提高预测精度的重要性。此外，基于抑郁症状存在的分组比较揭示了不同模型的表现差异。参与者被分为两组：至少存在一种抑郁症状的组和没有症状的组。GPT-4o mini 在症状存在组中对与抑郁相关的神经质和尽责性特质变化表现出更高的敏感性，而GPT-4o 则在不同组中展示了更细致的解释能力。这些 findings 强调了LLMs 在有效分析现实世界心理数据方面的潜力，为人工智能与心理学交叉领域的跨学科研究奠定了坚实的基础。 

---
# TiEBe: A Benchmark for Assessing the Current Knowledge of Large Language Models 

**Title (ZH)**: TiEBe：评估大型语言模型当前知识水平的标准基准 

**Authors**: Thales Sales Almeida, Giovana Kerche Bonás, João Guilherme Alves Santos, Hugo Abonizio, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2501.07482)  

**Abstract**: In a rapidly evolving knowledge landscape and the increasing adoption of large language models, a need has emerged to keep these models continuously updated with current events. While existing benchmarks evaluate general factual recall, they often overlook two critical aspects: the ability of models to integrate evolving knowledge through continual learning and the significant regional disparities in their performance. To address these gaps, we introduce the Timely Events Benchmark (TiEBe), a dataset containing over 11,000 question-answer pairs focused on globally and regionally significant events. TiEBe leverages structured retrospective data from Wikipedia, enabling continuous updates to assess LLMs' knowledge of evolving global affairs and their understanding of events across different regions. Our benchmark demonstrates that LLMs exhibit substantial geographic disparities in factual recall, emphasizing the need for more balanced global knowledge representation. Furthermore, TiEBe serves as a tool for evaluating continual learning strategies, providing insights into models' ability to acquire new information without forgetting past knowledge. 

**Abstract (ZH)**: 在知识快速演变的背景下，随着大型语言模型的广泛应用，持续更新这些模型以反映当前事件的需求日益凸显。尽管现有的基准测试评估了模型的一般事实记忆能力，但往往忽略了两个关键方面：模型通过持续学习整合演变知识的能力以及它们在不同地区的显著表现差异。为解决这些缺口，我们提出了及时事件基准测试 (TiEBe)，这是一个包含超过11,000个问题-答案对的数据集，重点关注全球和地区重要的事件。TiEBe 利用了从维基百科获取的结构化回顾性数据，从而使模型能够持续更新，评估其对全球事务演变的了解以及对不同地区事件的理解。我们的基准测试表明，大型语言模型在事实记忆方面存在显著的地理差异，突显了更平衡的全球知识呈现的必要性。此外，TiEBe 作为评估持续学习策略的工具，为了解模型在不遗忘过去知识的情况下获取新信息的能力提供了见解。 

---
# Enhancing Retrieval-Augmented Generation: A Study of Best Practices 

**Title (ZH)**: 增强检索增强生成：最佳实践研究 

**Authors**: Siran Li, Linus Stenzel, Carsten Eickhoff, Seyed Ali Bahrainian  

**Link**: [PDF](https://arxiv.org/pdf/2501.07391)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have recently shown remarkable advancements by integrating retrieval mechanisms into language models, enhancing their ability to produce more accurate and contextually relevant responses. However, the influence of various components and configurations within RAG systems remains underexplored. A comprehensive understanding of these elements is essential for tailoring RAG systems to complex retrieval tasks and ensuring optimal performance across diverse applications. In this paper, we develop several advanced RAG system designs that incorporate query expansion, various novel retrieval strategies, and a novel Contrastive In-Context Learning RAG. Our study systematically investigates key factors, including language model size, prompt design, document chunk size, knowledge base size, retrieval stride, query expansion techniques, Contrastive In-Context Learning knowledge bases, multilingual knowledge bases, and Focus Mode retrieving relevant context at sentence-level. Through extensive experimentation, we provide a detailed analysis of how these factors influence response quality. Our findings offer actionable insights for developing RAG systems, striking a balance between contextual richness and retrieval-generation efficiency, thereby paving the way for more adaptable and high-performing RAG frameworks in diverse real-world scenarios. Our code and implementation details are publicly available. 

**Abstract (ZH)**: 检索增强生成（RAG）系统通过将检索机制融入语言模型，最近取得了显著进展，增强了其生成更加准确和语境相关响应的能力。然而，RAG系统内部各种组件和配置的影响仍未被充分探索。对这些元素的全面理解对于根据复杂检索任务定制RAG系统并确保在各种应用场景下实现最佳性能至关重要。在本文中，我们开发了几个先进的RAG系统设计，整合了查询扩展、各种新颖的检索策略以及一种新颖的对比上下文学习RAG。我们通过系统的实验研究了关键因素，包括语言模型的规模、提示设计、文档片段大小、知识库规模、检索步长、查询扩展技术、对比上下文学习知识库、多语言知识库以及聚焦模式在句子级别检索相关上下文。通过广泛的实验，我们详细分析了这些因素如何影响响应质量。我们的发现为开发RAG系统提供了可操作的见解，平衡了上下文丰富性和检索生成效率，从而为不同真实世界场景下的更灵活和高性能RAG框架奠定了基础。我们的代码和实现细节是公开的。 

---
# Emergent effects of scaling on the functional hierarchies within large language models 

**Title (ZH)**: 大规模语言模型内部功能层次结构中的扩展示效 

**Authors**: Paul C. Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2501.07359)  

**Abstract**: Large language model (LLM) architectures are often described as functionally hierarchical: Early layers process syntax, middle layers begin to parse semantics, and late layers integrate information. The present work revisits these ideas. This research submits simple texts to an LLM (e.g., "A church and organ") and extracts the resulting activations. Then, for each layer, support vector machines and ridge regressions are fit to predict a text's label and thus examine whether a given layer encodes some information. Analyses using a small model (Llama-3.2-3b; 28 layers) partly bolster the common hierarchical perspective: Item-level semantics are most strongly represented early (layers 2-7), then two-item relations (layers 8-12), and then four-item analogies (layers 10-15). Afterward, the representation of items and simple relations gradually decreases in deeper layers that focus on more global information. However, several findings run counter to a steady hierarchy view: First, although deep layers can represent document-wide abstractions, deep layers also compress information from early portions of the context window without meaningful abstraction. Second, when examining a larger model (Llama-3.3-70b-Instruct), stark fluctuations in abstraction level appear: As depth increases, two-item relations and four-item analogies initially increase in their representation, then markedly decrease, and afterward increase again momentarily. This peculiar pattern consistently emerges across several experiments. Third, another emergent effect of scaling is coordination between the attention mechanisms of adjacent layers. Across multiple experiments using the larger model, adjacent layers fluctuate between what information they each specialize in representing. In sum, an abstraction hierarchy often manifests across layers, but large models also deviate from this structure in curious ways. 

**Abstract (ZH)**: 大型语言模型（LLM）架构常被描述为功能分层的：早期层处理句法，中期层开始解析语义，晚期层整合信息。本研究重新探讨了这些观点。本研究将简单文本（例如，“一座教堂和一个风琴”）提交给LLM，并提取生成的激活信号。然后，对每一层，使用支持向量机和岭回归来预测文本的标签，从而检查给定层是否编码了一些信息。使用小型模型（Llama-3.2-3b，28层）的分析部分支持了常见的分层观点：项目级别的语义在早期（第2层至第7层）最为强烈地表征，然后是两项目关系（第8层至第12层），接着是四项目类比（第10层至第15层）。之后，在专注于更全局信息的深层层中，代表性项目和简单关系逐渐减少。然而，一些发现与稳定的分层观点相矛盾：首先，尽管深层层可以表示文档级抽象，但它们也会压缩来自上下文窗口早期部分的信息，而这种压缩缺乏有意义的抽象。其次，在研究一个较大的模型（Llama-3.3-70b-Instruct）时，观察到抽象层次显著波动：随着深度增加，两项目关系和四项目类比的表示最初增加，随后大幅减少，之后又短暂增加。这种奇怪的模式在多个实验中持续出现。第三，随着模型规模的扩大，另一现象是相邻层之间的注意力机制协调。在使用较大模型进行的多个实验中，相邻层在它们各自专长表示的信息上波动。综上所述，抽象层次经常在各层间表现出来，但大型模型也会以奇怪的方式偏离这种结构。 

---
# FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering 

**Title (ZH)**: FinerWeb-10BT：基于LLM的行级过滤细化网络数据 

**Authors**: Erik Henriksson, Otto Tarkka, Filip Ginter  

**Link**: [PDF](https://arxiv.org/pdf/2501.07314)  

**Abstract**: Data quality is crucial for training Large Language Models (LLMs). Traditional heuristic filters often miss low-quality text or mistakenly remove valuable content. In this paper, we introduce an LLM-based line-level filtering method to enhance training data quality. We use GPT-4o mini to label a 20,000-document sample from FineWeb at the line level, allowing the model to create descriptive labels for low-quality lines. These labels are grouped into nine main categories, and we train a DeBERTa-v3 classifier to scale the filtering to a 10B-token subset of FineWeb. To test the impact of our filtering, we train GPT-2 models on both the original and the filtered datasets. The results show that models trained on the filtered data achieve higher accuracy on the HellaSwag benchmark and reach their performance targets faster, even with up to 25\% less data. This demonstrates that LLM-based line-level filtering can significantly improve data quality and training efficiency for LLMs. We release our quality-annotated dataset, FinerWeb-10BT, and the codebase to support further work in this area. 

**Abstract (ZH)**: 数据质量对于训练大型语言模型（LLMs）至关重要。传统的启发式过滤方法往往无法检测到低质量的文本，或者错误地删除有价值的内容。在这篇论文中，我们介绍了一种基于LLM的行级过滤方法，以提升训练数据的质量。我们使用GPT-4o mini对FineWeb数据集中2万个文档样本进行行级标注，允许模型为低质量行生成描述性标签。这些标签被归类为九个主要类别，并训练一个DeBERTa-v3分类器以将过滤扩展到FineWeb的100亿令牌子集。为了测试我们过滤方法的影响，我们在未过滤的数据集和经过过滤的数据集上分别训练了GPT-2模型。结果表明，使用经过过滤的数据集训练的模型在HellaSwag基准测试中获得了更高的准确率，并且即使数据量减少多达25%，也能更快地达到预期的性能目标。这表明基于LLM的行级过滤方法可以显著提高LLMs的数据质量和训练效率。我们发布了包含质量标注的数据集FinerWeb-10BT及其代码库，以支持该领域的进一步研究。 

---
# The Lessons of Developing Process Reward Models in Mathematical Reasoning 

**Title (ZH)**: 在数学推理中开发过程奖励模型的启示 

**Authors**: Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.07301)  

**Abstract**: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in the reasoning processes. However, the development of effective PRMs faces significant challenges, particularly in data annotation and evaluation methodologies. In this paper, through extensive experiments, we demonstrate that commonly used Monte Carlo (MC) estimation-based data synthesis for PRMs typically yields inferior performance and generalization compared to LLM-as-a-judge and human annotation methods. MC estimation relies on completion models to evaluate current-step correctness, leading to inaccurate step verification. Furthermore, we identify potential biases in conventional Best-of-N (BoN) evaluation strategies for PRMs: (1) The unreliable policy models generate responses with correct answers but flawed processes, leading to a misalignment between the evaluation criteria of BoN and the PRM objectives of process verification. (2) The tolerance of PRMs of such responses leads to inflated BoN scores. (3) Existing PRMs have a significant proportion of minimum scores concentrated on the final answer steps, revealing the shift from process to outcome-based assessment in BoN Optimized PRMs. To address these challenges, we develop a consensus filtering mechanism that effectively integrates MC estimation with LLM-as-a-judge and advocates a more comprehensive evaluation framework that combines response-level and step-level metrics. Based on the mechanisms, we significantly improve both model performance and data efficiency in the BoN evaluation and the step-wise error identification task. Finally, we release a new state-of-the-art PRM that outperforms existing open-source alternatives and provides practical guidelines for future research in building process supervision models. 

**Abstract (ZH)**: 过程奖励模型（PRMs）作为一种有前景的方法，在大型语言模型（LLMs）的数学推理过程中实现过程监督，旨在识别和缓解推理过程中的中间错误。然而，有效PRMs的发展面临着重大的挑战，尤其是在数据标注和评估方法方面。本文通过大量的实验表明，通常用于PRMs的数据合成方法——基于蒙特卡洛（MC）估计的方法，通常在性能和泛化能力上劣于LLM-as-a-judge和人工标注方法。MC估计依赖于完成模型来评估当前步骤的正确性，导致步骤验证不够准确。此外，我们还发现了常规的Best-of-N（BoN）评估策略在PRMs中的潜在偏差：（1）不可靠的策略模型生成正确答案但过程有误的答案，导致BoN的评估标准与PRM的过程验证目标不一致。（2）PRMs对这种答案的容忍度导致BoN得分膨胀。（3）现有的PRMs在最终答案步骤上有相当大的低分比例，显示出BoN优化后的PRMs从过程导向评估转向结果导向评估。为解决这些问题，我们开发了一种共识过滤机制，有效地将MC估计与LLM-as-a-judge相结合，并倡导一种更全面的评估框架，结合响应级和步骤级指标。基于这种机制，我们在BoN评估和逐阶错误识别任务中显著提高了模型性能和数据效率。最后，我们发布了一个新的状态最先进PRM，优于现有开源选项，并提供了未来研究中构建过程监督模型的实用指南。 

---
# Comparative analysis of optical character recognition methods for S\'ami texts from the National Library of Norway 

**Title (ZH)**: 挪威国家图书馆萨米文光学字符识别方法的比较分析 

**Authors**: Tita Enstad, Trond Trosterud, Marie Iversdatter Røsok, Yngvil Beyer, Marie Roald  

**Link**: [PDF](https://arxiv.org/pdf/2501.07300)  

**Abstract**: Optical Character Recognition (OCR) is crucial to the National Library of Norway's (NLN) digitisation process as it converts scanned documents into machine-readable text. However, for the Sámi documents in NLN's collection, the OCR accuracy is insufficient. Given that OCR quality affects downstream processes, evaluating and improving OCR for text written in Sámi languages is necessary to make these resources accessible. To address this need, this work fine-tunes and evaluates three established OCR approaches, Transkribus, Tesseract and TrOCR, for transcribing Sámi texts from NLN's collection. Our results show that Transkribus and TrOCR outperform Tesseract on this task, while Tesseract achieves superior performance on an out-of-domain dataset. Furthermore, we show that fine-tuning pre-trained models and supplementing manual annotations with machine annotations and synthetic text images can yield accurate OCR for Sámi languages, even with a moderate amount of manually annotated data. 

**Abstract (ZH)**: 光学字符识别（OCR）对挪威国家图书馆（NLN）的数字化过程至关重要，它将扫描的文档转换为机器可读的文本。然而，对于NLN藏有的萨米语文件，OCR的准确性不足。鉴于OCR质量影响后续处理过程，评估和改进用于萨米语文本的OCR是必要的，以便使这些资源更易于访问。为了满足这一需求，本研究针对NLN藏有的萨米语文本， fine-tuned 和评估了三种现有的OCR方法：Transkribus、Tesseract 和 TrOCR。我们的结果显示，在此任务中，Transkribus 和 TrOCR 的性能优于 Tesseract，而 Tesseract 在域外数据集上表现出更优的性能。此外，我们证明了 fine-tuning 预训练模型以及将手动标注与机器标注及合成文本图像相结合，即使使用适量的手动标注数据，也能获得准确的萨米语OCR。 

---
# When lies are mostly truthful: automated verbal deception detection for embedded lies 

**Title (ZH)**: 当谎言大多真实时：嵌入式谎言的自动语义欺骗检测 

**Authors**: Riccardo Loconte, Bennett Kleinberg  

**Link**: [PDF](https://arxiv.org/pdf/2501.07217)  

**Abstract**: Background: Verbal deception detection research relies on narratives and commonly assumes statements as truthful or deceptive. A more realistic perspective acknowledges that the veracity of statements exists on a continuum with truthful and deceptive parts being embedded within the same statement. However, research on embedded lies has been lagging behind. Methods: We collected a novel dataset of 2,088 truthful and deceptive statements with annotated embedded lies. Using a within-subjects design, participants provided a truthful account of an autobiographical event. They then rewrote their statement in a deceptive manner by including embedded lies, which they highlighted afterwards and judged on lie centrality, deceptiveness, and source. Results: We show that a fined-tuned language model (Llama-3-8B) can classify truthful statements and those containing embedded lies with 64% accuracy. Individual differences, linguistic properties and explainability analysis suggest that the challenge of moving the dial towards embedded lies stems from their resemblance to truthful statements. Typical deceptive statements consisted of 2/3 truthful information and 1/3 embedded lies, largely derived from past personal experiences and with minimal linguistic differences with their truthful counterparts. Conclusion: We present this dataset as a novel resource to address this challenge and foster research on embedded lies in verbal deception detection. 

**Abstract (ZH)**: 背景：语言欺骗检测研究依赖于叙事，并通常假设陈述为真实或欺骗的。更现实的观点是承认陈述的真实性存在于一个连续体上，其中真实的和欺骗性的内容被嵌入在同一陈述中。然而，关于嵌入式谎言的研究滞后。方法：我们收集了一个包含2,088个真实和虚假陈述的新数据集，并对这些陈述进行了嵌入式谎言的标注。采用单被试设计，参与者提供了一段关于亲身经历的真实描述。然后，他们以欺骗的方式重写了这一陈述，包括嵌入式谎言，并且他们随后对这些谎言的核心性、欺骗性和来源进行了判断。结果：我们展示了经过微调的语言模型（Llama-3-8B）可以以64%的准确率对真实陈述和包含嵌入式谎言的陈述进行分类。个体差异、语言特征和可解释性分析表明，向嵌入式谎言方向推进的挑战源自它们与真实陈述的相似性。典型的虚假陈述包含了三分之二是真实信息，三分之一是嵌入式谎言，这些谎言大多源自过去的个人经历，并且在语言上与它们的真实版本几乎没有区别。结论：我们提供这个数据集作为解决这一挑战的新资源，并促进语言欺骗检测中嵌入式谎言的研究。 

---
# ListConRanker: A Contrastive Text Reranker with Listwise Encoding 

**Title (ZH)**: ListConRanker：一种基于列表编码的对比文本重排器 

**Authors**: Junlong Liu, Yue Ma, Ruihui Zhao, Junhao Zheng, Qianli Ma, Yangyang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07111)  

**Abstract**: Reranker models aim to re-rank the passages based on the semantics similarity between the given query and passages, which have recently received more attention due to the wide application of the Retrieval-Augmented Generation. Most previous methods apply pointwise encoding, meaning that it can only encode the context of the query for each passage input into the model. However, for the reranker model, given a query, the comparison results between passages are even more important, which is called listwise encoding. Besides, previous models are trained using the cross-entropy loss function, which leads to issues of unsmooth gradient changes during training and low training efficiency. To address these issues, we propose a novel Listwise-encoded Contrastive text reRanker (ListConRanker). It can help the passage to be compared with other passages during the encoding process, and enhance the contrastive information between positive examples and between positive and negative examples. At the same time, we use the circle loss to train the model to increase the flexibility of gradients and solve the problem of training efficiency. Experimental results show that ListConRanker achieves state-of-the-art performance on the reranking benchmark of Chinese Massive Text Embedding Benchmark, including the cMedQA1.0, cMedQA2.0, MMarcoReranking, and T2Reranking datasets. 

**Abstract (ZH)**: 重排序模型旨在根据给定查询与段落之间的语义相似性重新排名段落，由于检索增强生成的广泛应用，这类模型最近受到了更多的关注。大多数此前的方法采用点wise编码，这意味着模型只能对每个段落输入的查询上下文进行编码。然而，对于重排序模型而言，给定一个查询后，段落之间的比较结果更为重要，这被称为listwise编码。除此之外，之前的模型是使用交叉熵损失函数进行训练的，这导致在训练过程中梯度变化不平滑且训练效率低下。为了解决这些问题，我们提出了一种新颖的Listwise编码对比文本重排序器（ListConRanker）。它可以在编码过程中帮助段落与其他段落进行比较，并增强正样本之间的对比信息以及正样本与负样本之间的对比信息。同时，我们使用圈形损失来训练模型，以增加梯度的灵活性并解决训练效率问题。实验结果表明，ListConRanker在中文大规模文本嵌入基准的重排序基准集，包括cMedQA1.0、cMedQA2.0、MMarcoReranking和T2Reranking数据集上，达到了最先进的性能。 

---
# AdaCS: Adaptive Normalization for Enhanced Code-Switching ASR 

**Title (ZH)**: AdaCS：增强代码转换ASR的自适应归一化 

**Authors**: Chuong Chu, Vu Tuan Dat Pham, Kien Dao, Hoang Nguyen, Quoc Hung Truong  

**Link**: [PDF](https://arxiv.org/pdf/2501.07102)  

**Abstract**: Intra-sentential code-switching (CS) refers to the alternation between languages that happens within a single utterance and is a significant challenge for Automatic Speech Recognition (ASR) systems. For example, when a Vietnamese speaker uses foreign proper names or specialized terms within their speech. ASR systems often struggle to accurately transcribe intra-sentential CS due to their training on monolingual data and the unpredictable nature of CS. This issue is even more pronounced for low-resource languages, where limited data availability hinders the development of robust models. In this study, we propose AdaCS, a normalization model integrates an adaptive bias attention module (BAM) into encoder-decoder network. This novel approach provides a robust solution to CS ASR in unseen domains, thereby significantly enhancing our contribution to the field. By utilizing BAM to both identify and normalize CS phrases, AdaCS enhances its adaptive capabilities with a biased list of words provided during inference. Our method demonstrates impressive performance and the ability to handle unseen CS phrases across various domains. Experiments show that AdaCS outperforms previous state-of-the-art method on Vietnamese CS ASR normalization by considerable WER reduction of 56.2% and 36.8% on the two proposed test sets. 

**Abstract (ZH)**: 句内代码转换（Intra-sentential Code-Switching, CS）指的是在同一句话内部切换语言的现象，这对于自动语音识别（Automatic Speech Recognition, ASR）系统来说是一个重要的挑战。例如，当越南语演讲者在其讲话中使用外来的人名或专业术语时。ASR 系统由于在其训练过程中使用的是单语言数据，以及 CS 的不可预测性，经常难以准确地转录句内代码转换。这一问题在低资源语言中尤为明显，因为数据的限制阻碍了稳健模型的开发。在本研究中，我们提出了一种名为 AdaCS 的规范化模型，该模型将可适应偏差注意模块（Adaptive Bias Attention Module, BAM）整合到编码器-解码器网络中。这一新颖的方法为未见过的领域中的 CS ASR 提供了稳健的解决方案，从而显著增强了我们在该领域的贡献。通过利用 BAM 识别和规范化 CS 片段，AdaCS 在推断过程中提供的偏差词表增强了其适应能力。我们的方法在不同领域的未见过的 CS 片段处理中表现出了出色的性能。实验表明，在越南语 CS ASR 标准化方面，AdaCS 比之前的状态最先进方法在两个提出的测试集中的错误率分别降低了 56.2% 和 36.8%。 

---
# Boosting Text-To-Image Generation via Multilingual Prompting in Large Multimodal Models 

**Title (ZH)**: 通过多语言提示在大型多模态模型中的应用以增强文本到图像生成 

**Authors**: Yongyu Mu, Hengyu Li, Junxin Wang, Xiaoxuan Zhou, Chenglong Wang, Yingfeng Luo, Qiaozhi He, Tong Xiao, Guocheng Chen, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07086)  

**Abstract**: Previous work on augmenting large multimodal models (LMMs) for text-to-image (T2I) generation has focused on enriching the input space of in-context learning (ICL). This includes providing a few demonstrations and optimizing image descriptions to be more detailed and logical. However, as demand for more complex and flexible image descriptions grows, enhancing comprehension of input text within the ICL paradigm remains a critical yet underexplored area. In this work, we extend this line of research by constructing parallel multilingual prompts aimed at harnessing the multilingual capabilities of LMMs. More specifically, we translate the input text into several languages and provide the models with both the original text and the translations. Experiments on two LMMs across 3 benchmarks show that our method, PMT2I, achieves superior performance in general, compositional, and fine-grained assessments, especially in human preference alignment. Additionally, with its advantage of generating more diverse images, PMT2I significantly outperforms baseline prompts when incorporated with reranking methods. Our code and parallel multilingual data can be found at this https URL. 

**Abstract (ZH)**: 以往关于增强大型多模态模型（LMMs）以用于文本到图像（T2I）生成的研究主要集中在丰富上下文学习（ICL）的输入空间。这包括提供少量示例和优化图像描述以使其更详细和合理。然而，随着对更复杂和灵活的图像描述的需求增长，如何在ICL范式中增强输入文本的理解仍然是一个关键但尚未充分探索的领域。在本工作中，我们通过构建平行多语言提示来扩展这一研究方向，旨在利用LMMs的多语言能力。具体而言，我们将输入文本翻译成多种语言，并向模型提供原文和翻译文本。在两个LMMs的三个基准测试中进行的实验表明，我们的方法（PMT2I）在通用性、组合性和细粒度评估中均表现出更优的性能，尤其是在人类偏好一致性方面。此外，由于PMT2I能够生成更多样化的图像，因此当与排序方法结合使用时，其性能显著优于基线提示。我们的代码和平行多语言数据可以通过以下链接查询：this https URL。 

---
# ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization 

**Title (ZH)**: ViSoLex：越南社交媒体词汇规范化开源资源库 

**Authors**: Anh Thi-Hoang Nguyen, Dung Ha Nguyen, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07020)  

**Abstract**: ViSoLex is an open-source system designed to address the unique challenges of lexical normalization for Vietnamese social media text. The platform provides two core services: Non-Standard Word (NSW) Lookup and Lexical Normalization, enabling users to retrieve standard forms of informal language and standardize text containing NSWs. ViSoLex's architecture integrates pre-trained language models and weakly supervised learning techniques to ensure accurate and efficient normalization, overcoming the scarcity of labeled data in Vietnamese. This paper details the system's design, functionality, and its applications for researchers and non-technical users. Additionally, ViSoLex offers a flexible, customizable framework that can be adapted to various datasets and research requirements. By publishing the source code, ViSoLex aims to contribute to the development of more robust Vietnamese natural language processing tools and encourage further research in lexical normalization. Future directions include expanding the system's capabilities for additional languages and improving the handling of more complex non-standard linguistic patterns. 

**Abstract (ZH)**: ViSoLex 是一个开源系统，旨在解决越南社交媒体文本词汇规范化过程中的独特挑战。该平台提供两种核心服务：非标准词（NSW）查找和词汇规范化，使用户能够检索非正式语言的标准形式，并对包含非标准词的文本进行标准化。ViSoLex 的架构集成了预训练的语言模型和弱监督学习技术，从而确保准确高效地进行规范化，克服了越南标注数据稀缺的问题。本文详细介绍了该系统的架构、功能及其在研究人员和非技术人员中的应用。此外，ViSoLex 提供了一个灵活且可定制的框架，可以根据不同数据集和研究需求进行调整。通过发布源代码，ViSoLex 旨在促进更robust的越南自然语言处理工具的发展，并鼓励进一步在词汇规范化方面的研究。未来的研究方向包括扩展系统的功能以支持其他语言，并改进对更复杂非标准语言模式的处理。 

---
# Harnessing Large Language Models for Disaster Management: A Survey 

**Title (ZH)**: 利用大型语言模型进行灾害管理：一项综述 

**Authors**: Zhenyu Lei, Yushun Dong, Weiyu Li, Rong Ding, Qi Wang, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.06932)  

**Abstract**: Large language models (LLMs) have revolutionized scientific research with their exceptional capabilities and transformed various fields. Among their practical applications, LLMs have been playing a crucial role in mitigating threats to human life, infrastructure, and the environment. Despite growing research in disaster LLMs, there remains a lack of systematic review and in-depth analysis of LLMs for natural disaster management. To address the gap, this paper presents a comprehensive survey of existing LLMs in natural disaster management, along with a taxonomy that categorizes existing works based on disaster phases and application scenarios. By collecting public datasets and identifying key challenges and opportunities, this study aims to guide the professional community in developing advanced LLMs for disaster management to enhance the resilience against natural disasters. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过其卓越的能力对科学研究产生了革命性的影响，并且在多个领域得到了转型。在其实用应用中，LLMs 在缓解对人类生命、基础设施和环境的威胁方面发挥着关键作用。尽管有关灾难LLMs的研究正在增加，但仍缺乏对自然灾难管理中LLMs的系统性回顾和深度分析。为了填补这一空白，本文对现有的自然灾难管理中的LLMs进行了全面的调查，并提出了一个分类体系，根据灾害阶段和应用场景对现有工作进行了分类。通过收集公开数据集并识别关键挑战和机遇，本文旨在引导专业人士开发先进的LLMs以增强对自然灾难的韧性。 

---
# Language Fusion for Parameter-Efficient Cross-lingual Transfer 

**Title (ZH)**: 参数高效跨语言迁移的语言融合方法 

**Authors**: Philipp Borchert, Ivan Vulić, Marie-Francine Moens, Jochen De Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2501.06892)  

**Abstract**: Limited availability of multilingual text corpora for training language models often leads to poor performance on downstream tasks due to undertrained representation spaces for languages other than English. This 'under-representation' has motivated recent cross-lingual transfer methods to leverage the English representation space by e.g. mixing English and 'non-English' tokens at the input level or extending model parameters to accommodate new languages. However, these approaches often come at the cost of increased computational complexity. We propose Fusion forLanguage Representations (FLARE) in adapters, a novel method that enhances representation quality and downstream performance for languages other than English while maintaining parameter efficiency. FLARE integrates source and target language representations within low-rank (LoRA) adapters using lightweight linear transformations, maintaining parameter efficiency while improving transfer performance. A series of experiments across representative cross-lingual natural language understanding tasks, including natural language inference, question-answering and sentiment analysis, demonstrate FLARE's effectiveness. FLARE achieves performance improvements of 4.9% for Llama 3.1 and 2.2% for Gemma~2 compared to standard LoRA fine-tuning on question-answering tasks, as measured by the exact match metric. 

**Abstract (ZH)**: 多语言文本语料库的有限可用性往往导致语言模型在下游任务中的表现不佳，特别是对于非英语语言，因为其表示空间的训练不足。这种“表示不足”驱使近期的一些跨语言转移方法通过在输入级别混合英语和“非英语”词汇或扩展模型参数来利用英语表示空间，以容纳新语言。然而，这些方法通常会增加计算复杂性。我们提出了一种新颖的方法——适配器中的多语言表示融合（Fusion for Language Representations, FLARE），旨在同时提高非英语语言的表示质量和下游性能，同时保持参数效率。FLARE方法通过轻量级的线性变换将源语言和目标语言表示整合到低秩（LoRA）适配器中，从而在保持参数效率的同时提高转移性能。一系列涵盖代表性的跨语言自然语言理解任务（包括自然语言推理、问答和情感分析）的实验表明，FLARE的有效性。在问答任务中，FLARE相对于标准LoRA微调，在Llama 3.1上的精确匹配度提高了4.9%，在Gemma~2上的精确匹配度提高了2.2%。 

---
# A Comprehensive Evaluation of Large Language Models on Mental Illnesses in Arabic Context 

**Title (ZH)**: 阿拉伯语语境中大型语言模型对精神疾病的综合评估 

**Authors**: Noureldin Zahran, Aya E. Fouda, Radwa J. Hanafy, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2501.06859)  

**Abstract**: Mental health disorders pose a growing public health concern in the Arab world, emphasizing the need for accessible diagnostic and intervention tools. Large language models (LLMs) offer a promising approach, but their application in Arabic contexts faces challenges including limited labeled datasets, linguistic complexity, and translation biases. This study comprehensively evaluates 8 LLMs, including general multi-lingual models, as well as bi-lingual ones, on diverse mental health datasets (such as AraDepSu, Dreaddit, MedMCQA), investigating the impact of prompt design, language configuration (native Arabic vs. translated English, and vice versa), and few-shot prompting on diagnostic performance. We find that prompt engineering significantly influences LLM scores mainly due to reduced instruction following, with our structured prompt outperforming a less structured variant on multi-class datasets, with an average difference of 14.5\%. While language influence on performance was modest, model selection proved crucial: Phi-3.5 MoE excelled in balanced accuracy, particularly for binary classification, while Mistral NeMo showed superior performance in mean absolute error for severity prediction tasks. Few-shot prompting consistently improved performance, with particularly substantial gains observed for GPT-4o Mini on multi-class classification, boosting accuracy by an average factor of 1.58. These findings underscore the importance of prompt optimization, multilingual analysis, and few-shot learning for developing culturally sensitive and effective LLM-based mental health tools for Arabic-speaking populations. 

**Abstract (ZH)**: 阿拉伯世界中心理健康障碍日益成为一个重要的公共卫生问题，这突显了需要可获取的诊断和干预工具。大规模语言模型（LLMs）提供了有希望的方法，但它们在阿拉伯语环境中应用面临着挑战，包括标注数据的限制、语言复杂性以及翻译偏差。本研究全面评估了8种LLM，包括通用多语言模型和双语模型，它们在多种心理健康数据集（如AraDepSu、Dreaddit、MedMCQA）上的表现，探讨了提示设计、语言配置（本族阿拉伯语和翻译后的英语，反之亦然）以及少样本提示对诊断性能的影响。研究发现，提示工程显著影响了LLM的得分，主要是因为减少了指令遵循，我们的结构化提示在多类数据集上表现优于非结构化版本，平均得分差距为14.5%。虽然语言对性能的影响较为微小，但模型选择至关重要：Phi-3.5 MoE在平衡准确率方面表现尤为突出，尤其是在二分类任务中；而Mistral NeMo在严重程度预测任务中的平均绝对误差表现更优。少样本提示始终提高了性能，特别是GPT-4o Mini在多类分类任务中的表现有了显著提升，平均准确率提高了1.58倍。这些发现强调了优化提示、多语言分析和少样本学习的重要性，以开发适合阿拉伯语使用者的文化敏感且有效的基于LLM的心理健康工具。 

---
# Event Argument Extraction with Enriched Prompts 

**Title (ZH)**: 带有丰富提示的事件argument提取 

**Authors**: Chen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06825)  

**Abstract**: This work aims to delve deeper into prompt-based event argument extraction (EAE) models. We explore the impact of incorporating various types of information into the prompt on model performance, including trigger, other role arguments for the same event, and role arguments across multiple events within the same document. Further, we provide the best possible performance that the prompt-based EAE model can attain and demonstrate such models can be further optimized from the perspective of the training objective. Experiments are carried out on three small language models and two large language models in RAMS. 

**Abstract (ZH)**: 本研究旨在更深入地探讨基于提示的事件论元提取（EAE）模型。我们探索了将不同类型的信息整合到提示中对模型性能的影响，包括事件的触发词、同一事件的其他角色论元以及同一文档中跨事件的角色论元。此外，我们展示了基于提示的EAE模型所能达到的最佳性能，并从训练目标的角度表明这些模型仍有优化空间。实验在RAMS中的三个小型语言模型和两个大型语言模型上进行。 

---
# Bridging the Fairness Gap: Enhancing Pre-trained Models with LLM-Generated Sentences 

**Title (ZH)**: 弥合公正性差距：通过使用大语言模型生成的句子增强预训练模型 

**Authors**: Liu Yu, Ludie Guo, Ping Kuang, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.06795)  

**Abstract**: Pre-trained language models (PLMs) are trained on data that inherently contains gender biases, leading to undesirable impacts. Traditional debiasing methods often rely on external corpora, which may lack quality, diversity, or demographic balance, affecting the effectiveness of debiasing. With the rise of large language models and their extensive knowledge, we propose enhancing fairness (Fair-Gender) in PLMs by absorbing coherent, attribute-balanced, and semantically rich sentences. However, these sentences cannot be directly used for debiasing due to alignment issues and the risk of negative transfer. We address this by applying causal analysis to estimate causal effects, filtering out unaligned sentences, and identifying aligned ones for incorporation into PLMs, thereby ensuring positive transfer. Experiments show that our approach significantly reduces gender biases in PLMs while preserving their language expressiveness. 

**Abstract (ZH)**: 预训练语言模型（PLMs）在训练过程中包含了固有的性别偏见，导致了不良影响。传统的去偏方法通常依赖于外部语料库，这些语料库可能存在质量、多样性和人口统计平衡方面的问题，从而影响去偏的效果。随着大型语言模型的兴起及其广泛的知识积累，我们提出通过吸收一致、属性均衡且语义丰富的句子来增强PLMs的公平性（Fair-Gender）。然而，这些句子不能直接用于去偏，因为存在对齐问题和负迁移的风险。为此，我们应用因果分析来估计因果效应，筛选出未对齐的句子，并识别出对齐的句子以便将其纳入PLMs，从而确保正迁移。实验结果显示，我们的方法可以显著减少PLMs中的性别偏见，同时保持其语言表达能力。 

---
# Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models 

**Title (ZH)**: padding 调度：T2I 模型中 padding 令牌的机理分析 

**Authors**: Michael Toker, Ido Galil, Hadas Orgad, Rinon Gal, Yoad Tewel, Gal Chechik, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2501.06751)  

**Abstract**: Text-to-image (T2I) diffusion models rely on encoded prompts to guide the image generation process. Typically, these prompts are extended to a fixed length by adding padding tokens before text encoding. Despite being a default practice, the influence of padding tokens on the image generation process has not been investigated. In this work, we conduct the first in-depth analysis of the role padding tokens play in T2I models. We develop two causal techniques to analyze how information is encoded in the representation of tokens across different components of the T2I pipeline. Using these techniques, we investigate when and how padding tokens impact the image generation process. Our findings reveal three distinct scenarios: padding tokens may affect the model's output during text encoding, during the diffusion process, or be effectively ignored. Moreover, we identify key relationships between these scenarios and the model's architecture (cross or self-attention) and its training process (frozen or trained text encoder). These insights contribute to a deeper understanding of the mechanisms of padding tokens, potentially informing future model design and training practices in T2I systems. 

**Abstract (ZH)**: 文本到图像（T2I）扩散模型依赖编码提示来引导图像生成过程。通常，这些提示会在文本编码前通过添加填充标记扩展到固定长度。尽管这是默认做法，但填充标记对图像生成过程的影响尚未得到研究。在本文中，我们首次对填充标记在T2I模型中的作用进行了深入分析。我们开发了两种因果分析技术，以研究不同T2I管道组件中token表示的信息编码方式。利用这些技术，我们探讨了填充标记在图像生成过程中的影响时间和方式。我们的发现揭示了三种不同的场景：填充标记可能在文本编码期间、扩散过程中影响模型的输出，或者可以被有效忽略。此外，我们确定了这些场景与模型架构（交叉注意或自注意力）及其训练过程（固定或训练文本编码器）之间的关键关系。这些见解有助于更深入地了解填充标记的机制，可能为进一步优化T2I系统的模型设计和训练实践提供指导。 

---
# Hierarchical Divide-and-Conquer for Fine-Grained Alignment in LLM-Based Medical Evaluation 

**Title (ZH)**: 基于LLM的医疗评价中细粒度对齐的分级分而治之方法 

**Authors**: Shunfan Zheng, Xiechi Zhang, Gerard de Melo, Xiaoling Wang, Linlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06741)  

**Abstract**: In the rapidly evolving landscape of large language models (LLMs) for medical applications, ensuring the reliability and accuracy of these models in clinical settings is paramount. Existing benchmarks often focus on fixed-format tasks like multiple-choice QA, which fail to capture the complexity of real-world clinical diagnostics. Moreover, traditional evaluation metrics and LLM-based evaluators struggle with misalignment, often providing oversimplified assessments that do not adequately reflect human judgment. To address these challenges, we introduce HDCEval, a Hierarchical Divide-and-Conquer Evaluation framework tailored for fine-grained alignment in medical evaluation. HDCEval is built on a set of fine-grained medical evaluation guidelines developed in collaboration with professional doctors, encompassing Patient Question Relevance, Medical Knowledge Correctness, and Expression. The framework decomposes complex evaluation tasks into specialized subtasks, each evaluated by expert models trained through Attribute-Driven Token Optimization (ADTO) on a meticulously curated preference dataset. This hierarchical approach ensures that each aspect of the evaluation is handled with expert precision, leading to a significant improvement in alignment with human evaluators. 

**Abstract (ZH)**: 在大型语言模型（LLMs）在医疗应用的快速演变背景下，确保这些模型在临床环境中的可靠性和准确性至关重要。现有的基准测试通常侧重于固定格式的任务，如多项选择型问答，这未能捕捉到实际临床诊断的复杂性。此外，传统的评估指标和基于LLM的评估工具经常面临偏差问题，往往提供过于简化的评估，未能充分反映人类的判断。为应对这些挑战，我们引入了HDCEval，这是一种针对医疗评估细分对齐的分层征服评估框架。HDCEval基于与专业医生合作开发的一套细粒度的医疗评估指南，涵盖患者问题相关性、医学知识正确性和表达等方面。该框架将复杂的评估任务分解为专门的子任务，每个子任务由通过特性驱动的标记优化（ADTO）在精心策划的偏好数据集上训练的专家模型进行评估。这种分层方法确保每个评估方面都能以专家级别的精确度进行处理，从而显著改善与人类评估者的对齐。 

---
# Better Prompt Compression Without Multi-Layer Perceptrons 

**Title (ZH)**: 无需多层感知器的更好提示压缩 

**Authors**: Edouardo Honig, Andrew Lizarraga, Zijun Frank Zhang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06730)  

**Abstract**: Prompt compression is a promising approach to speeding up language model inference without altering the generative model. Prior works compress prompts into smaller sequences of learned tokens using an encoder that is trained as a LowRank Adaptation (LoRA) of the inference language model. However, we show that the encoder does not need to keep the original language model's architecture to achieve useful compression. We introduce the Attention-Only Compressor (AOC), which learns a prompt compression encoder after removing the multilayer perceptron (MLP) layers in the Transformer blocks of a language model, resulting in an encoder with roughly 67% less parameters compared to the original model. Intriguingly we find that, across a range of compression ratios up to 480x, AOC can better regenerate prompts and outperform a baseline compression encoder that is a LoRA of the inference language model without removing MLP layers. These results demonstrate that the architecture of prompt compression encoders does not need to be identical to that of the original decoder language model, paving the way for further research into architectures and approaches for prompt compression. 

**Abstract (ZH)**: 以下是对这段内容的中文翻译，符合学术规范：

提示压缩是一种在不改变生成模型的情况下加速语言模型推理的有希望的方法。已有研究通过使用在推断语言模型的LowRank Adaptation (LoRA)中训练的编码器，将提示压缩为较小的学习词元序列。然而，我们发现编码器无需保留原始语言模型的架构即可实现有效的压缩。我们引入了一种名为仅注意压缩器（AOC, Attention-Only Compressor）的方法，在语言模型的Transformer块中移除多层感知机（MLP）层后学习一种提示压缩编码器，相对于原始模型，该编码器的参数量减少了约67%。有趣的是，我们发现，在压缩比高达480倍的范围内，AOC可以更有效地再生提示，并且在不移除MLP层情况下使用的基线压缩编码器（该编码器是推断语言模型的LoRA）的基础上表现出更好的性能。这些结果表明，提示压缩编码器的架构不一定需要与原始解码语言模型的架构相同，这为后续研究提示压缩的架构和方法开辟了新的途径。 

---
# Measuring the Robustness of Reference-Free Dialogue Evaluation Systems 

**Title (ZH)**: 测量无参考对话评估系统的稳健性 

**Authors**: Justin Vasselli, Adam Nohejl, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2501.06728)  

**Abstract**: Advancements in dialogue systems powered by large language models (LLMs) have outpaced the development of reliable evaluation metrics, particularly for diverse and creative responses. We present a benchmark for evaluating the robustness of reference-free dialogue metrics against four categories of adversarial attacks: speaker tag prefixes, static responses, ungrammatical responses, and repeated conversational context. We analyze metrics such as DialogRPT, UniEval, and PromptEval -- a prompt-based method leveraging LLMs -- across grounded and ungrounded datasets. By examining both their correlation with human judgment and susceptibility to adversarial attacks, we find that these two axes are not always aligned; metrics that appear to be equivalent when judged by traditional benchmarks may, in fact, vary in their scores of adversarial responses. These findings motivate the development of nuanced evaluation frameworks to address real-world dialogue challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）驱动的对话系统进展迅速，但可靠的评估指标的发展相对滞后，尤其是在评价多样性和创新性回答方面。我们提出了一项基准测试，用于评估参考无关对话指标在对抗性攻击下的稳健性，这些攻击分为四类：说话者标签前缀、静态回答、不规范的回答以及重复的对话背景。我们分析了如DialogRPT、UniEval和PromptEval（一种利用LLMs的提示方法）等指标在具体和抽象数据集上的表现。通过探讨这些指标与人类判断的相关性及其对抗性攻击的敏感性，我们发现这两个维度并不总是一致的；在传统基准测试中表现相似的指标，实际上在对抗性响应上的评分可能有所不同。这些发现促使我们开发更加精细的评估框架，以应对实际对话中的挑战。 

---
# ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian 

**Title (ZH)**: ZNO-Eval：评估大型语言模型在乌克兰语中的推理能力 

**Authors**: Mykyta Syromiatnikov, Victoria Ruvinskaya, Anastasiya Troynina  

**Link**: [PDF](https://arxiv.org/pdf/2501.06715)  

**Abstract**: As the usage of large language models for problems outside of simple text understanding or generation increases, assessing their abilities and limitations becomes crucial. While significant progress has been made in this area over the last few years, most research has focused on benchmarking English, leaving other languages underexplored. This makes evaluating the reasoning and robustness level of language models in Ukrainian particularly challenging. The purpose of this work is to establish a comprehensive benchmark for the reasoning capabilities evaluation of large language models in the Ukrainian language. This paper presents the ZNO-Eval benchmark based on real exam tasks from Ukraine's standardized educational testing system: the External Independent Evaluation and the National Multi-subject Test. With single-answer options, multiple-choice, matching, and open-ended questions from diverse subjects, including Ukrainian language, mathematics, history, and geography, this dataset paves the way toward a thorough analysis of reasoning capabilities across different domains and complexities. Evaluation of several well-known language models, such as GPT-3.5-Turbo, GPT-4o, GPT-4-Turbo, Mistral Large, Claude 3 Opus, and Gemini-1.5 Pro on this benchmark demonstrated the superiority of GPT-4o in both common knowledge reasoning and intricate language tasks. At the same time, Gemini Pro and GPT-4 Turbo excelled in the arithmetic domain, leading in single-answer and open-ended math problems. While all models were close to max performance in text-only common knowledge tasks like history and geography, there still is a gap for Ukrainian language and math, thus highlighting the importance of developing specialized language benchmarks for more accurate assessments of model capabilities and limitations across different languages and contexts. 

**Abstract (ZH)**: 随着大型语言模型在复杂文本理解和生成之外的问题上的应用增加，评估其能力和限制变得尤为重要。尽管在过去几年间该领域取得了显著进展，但大多数研究仍集中在基准测试英语上，而其他语言则较少受到关注。这使得评估乌克兰语语言模型的推理能力和稳健性变得尤为具有挑战性。本文的目的是建立一个全面的基准，以评估大型语言模型在乌克兰语中的推理能力。本文基于乌克兰标准化教育评估系统——外部独立评估和国家多学科测试的真实考试任务，提出了ZNO-Eval基准。该数据集包含多个答案选项的选择题、匹配题、开放性问题以及来自不同学科的题型，包括乌克兰语、数学、历史和地理等，为跨不同领域和复杂性层次的推理能力进行深入分析铺平了道路。

对该基准上几种知名语言模型（如GPT-3.5-Turbo、GPT-4o、GPT-4-Turbo、Mistral Large、Claude 3 Opus 和 Gemini-1.5 Pro）的评估表明，GPT-4o在常识推理和复杂语言任务方面均表现优越。同时，Gemini Pro 和 GPT-4 Turbo 在数学领域表现出色，领先于单选题和开放性数学问题。尽管所有模型在仅涉及文本的常识任务（如历史和地理）中的表现均接近最佳水平，但乌克兰语和数学领域仍然存在差距，这突显了开发针对不同语言和应用场景的专业化语言基准以准确评估模型能力及限制的重要性。 

---
# TAPO: Task-Referenced Adaptation for Prompt Optimization 

**Title (ZH)**: TAPO：任务参考适配优化提示技术 

**Authors**: Wenxin Luo, Weirui Wang, Xiaopeng Li, Weibo Zhou, Pengyue Jia, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.06689)  

**Abstract**: Prompt engineering can significantly improve the performance of large language models (LLMs), with automated prompt optimization (APO) gaining significant attention due to the time-consuming and laborious nature of manual prompt design. However, much of the existing work in APO overlooks task-specific characteristics, resulting in prompts that lack domain specificity and are not well-suited for task-specific optimization. In this paper, we introduce TAPO, a multitask-aware prompt optimization framework composed of three key modules. First, a task-aware metric selection module is proposed to enhance task-specific prompt generation capabilities. Second, we present a multi-metrics evaluation module to jointly evaluate prompts from multiple perspectives. Third, an evolution-based optimization framework is introduced for automatic prompt refinement, which improves adaptability across various tasks. Extensive experiments on six datasets demonstrate the effectiveness of our approach, and our code is publicly available. 

**Abstract (ZH)**: 提示工程技术可以显著提高大型语言模型（LLMs）的性能，自动提示优化（APO）由于其耗时且劳动密集的特性，引起了广泛关注。然而，现有APO工作的许多研究忽视了任务特定的特征，导致生成的提示缺乏特定领域的针对性，不适用于任务特定的优化。在本文中，我们提出了TAPO，这是一个多任务意识的提示优化框架，包含三个关键模块。首先，我们提出了一种任务感知的度量选择模块，以增强任务特定提示生成的能力。其次，我们介绍了多度量评估模块，从多个视角联合评估提示。第三，我们引入了一种基于进化的方法，用于自动提示优化，增强其在各种任务中的适应性。在六个数据集上的广泛实验表明了我们方法的有效性，我们的代码已经公开。 

---
# FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings 

**Title (ZH)**: 聚焦偏好：通过重点关注正确的偏好排序来提升偏好优化 

**Authors**: Tong Liu, Xiao Yu, Wenxuan Zhou, Jindong Gu, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2501.06645)  

**Abstract**: Efficient preference optimization algorithms such as Direct Preference Optimization (DPO) have become a popular approach in aligning large language models (LLMs) with human preferences. These algorithms implicitly treat the LLM as a reward model, and focus on training it to correct misranked preference pairs. However, recent work~\citep{chen2024preference} empirically finds that DPO training \textit{rarely improves these misranked preference pairs}, despite its gradient emphasizing on these cases. We introduce FocalPO, a DPO variant that instead \textit{down-weighs} misranked preference pairs and prioritizes enhancing the model's understanding of pairs that it can already rank correctly. Inspired by Focal Loss used in vision tasks, FocalPO achieves this by adding a modulating factor to dynamically scale DPO loss. Our experiment demonstrates that FocalPO surpasses DPO and its variants on popular benchmarks like Alpaca Eval 2.0 using Mistral-Base-7B and Llama-3-Instruct-8B. Additionally, we empirically reveals how FocalPO affects training on correct and incorrect sample groups, further underscoring its effectiveness. 

**Abstract (ZH)**: 高效的偏好优化算法（如直接偏好优化DPO）已成为将大型语言模型（LLMs）与人类偏好对齐的一种流行方法。这些算法隐式地将LLM视为奖励模型，并专注于训练它纠正错误排名的偏好对。然而，近期研究~\citet{chen2024preference}通过实验证明，尽管DPO训练强调纠正这些错误排名的偏好对，但实际效果并不显著。我们提出了FocalPO，这是一种DPO变体，它通过减少错误排名的偏好对的权重，优先提高模型对已能正确排名的偏好对的理解能力。受到在视觉任务中使用焦点损失（Focal Loss）的启发，FocalPO通过动态调整DPO损失的加权因子来实现这一目标。我们的实验表明，在使用Mistral-Base-7B和Llama-3-Instruct-8B的流行基准测试如Alpaca Eval 2.0中，FocalPO在性能上超过了DPO及其变体。此外，我们还通过实验证明了FocalPO如何影响正确和错误样本组的训练，进一步突显了其有效性。 

---
# Scaling Down Semantic Leakage: Investigating Associative Bias in Smaller Language Models 

**Title (ZH)**: 减少语义泄露：探究小型语言模型中的关联偏见 

**Authors**: Veronika Smilga  

**Link**: [PDF](https://arxiv.org/pdf/2501.06638)  

**Abstract**: Semantic leakage is a phenomenon recently introduced by Gonen et al. (2024). It refers to a situation in which associations learnt from the training data emerge in language model generations in an unexpected and sometimes undesired way. Prior work has focused on leakage in large language models (7B+ parameters). In this study, I use Qwen2.5 model family to explore whether smaller models, ranging from 500M to 7B parameters, demonstrate less semantic leakage due to their limited capacity for capturing complex associations. Building on the previous dataset from Gonen et al. (2024), I introduce a new dataset of color-focused prompts, categorized into specific types of semantic associations, to systematically evaluate the models' performance. Results indicate that smaller models exhibit less semantic leakage overall, although this trend is not strictly linear, with medium-sized models sometimes surpassing larger ones in leaking behavior. The dataset, the model generations, and the evaluation code are publicly available at this https URL. 

**Abstract (ZH)**: 语义泄露是一种最近由Gonen等人（2024）引入的现象。它指的是训练数据中学到的关联以一种未预期且有时是不希望的方式在语言模型生成中显现出来。此前的研究主要集中在大型语言模型（参数量7B以上）上的泄露问题。本研究中，我使用Qwen2.5模型家族来探索较小容量的模型（从500M到7B参数）是否由于其捕捉复杂关联的能力有限，从而表现出较少的语义泄露。基于Gonen等人（2024）之前的 datasets，我引入了一个新的以颜色为主题的 prompt 数据集，将其分为特定类型的语义关联类别，以系统地评估模型的表现。结果表明，总体来看较小的模型表现出较少的语义泄露，尽管这一趋势并非严格线性，有时中等规模的模型在泄露行为上会超过大型模型。该 datasets、模型生成的输出以及评估代码可在以下链接中公开访问：[这里](this https URL)。 

---
# Dual use issues in the field of Natural Language Generation 

**Title (ZH)**: 自然语言生成领域的双重用途问题 

**Authors**: Emiel van Miltenburg  

**Link**: [PDF](https://arxiv.org/pdf/2501.06636)  

**Abstract**: This report documents the results of a recent survey in the SIGGEN community, focusing on Dual Use issues in Natural Language Generation (NLG). SIGGEN is the Special Interest Group (SIG) of the Association for Computational Linguistics (ACL) for researchers working on NLG. The survey was prompted by the ACL executive board, which asked all SIGs to provide an overview of dual use issues within their respective subfields. The survey was sent out in October 2024 and the results were processed in January 2025. With 23 respondents, the survey is presumably not representative of all SIGGEN members, but at least this document offers a helpful resource for future discussions.
This report is open to feedback from the SIGGEN community. Let me know if you have any questions or comments! 

**Abstract (ZH)**: 本报告记录了最近SIGGEN社区进行的一项调查结果，该调查重点探讨自然语言生成（NLG）中的双重用途问题。SIGGEN是计算语言学协会（ACL）的一个特别兴趣组（SIG），针对从事NLG研究的科研人员。此次调查是由ACL执行委员会发起的，要求所有SIG提供各自研究领域的双重用途问题概述。本次调查于2024年10月发出，结果于2025年1月处理。虽然仅有23位受访者，该调查可能无法代表所有SIGGEN成员，但至少这份文档为未来讨论提供了一项有用的资源。

本报告面向SIGGEN社区开放反馈。如有任何问题或建议，请随时告知我！ 

---
# ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning 

**Title (ZH)**: ChemAgent：在大型语言模型中自更新的化学知识库增强化学推理 

**Authors**: Xiangru Tang, Tianyu Hu, Muyang Ye, Yanjun Shao, Xunjian Yin, Siru Ouyang, Wangchunshu Zhou, Pan Lu, Zhuosheng Zhang, Yilun Zhao, Arman Cohan, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2501.06590)  

**Abstract**: Chemical reasoning usually involves complex, multi-step processes that demand precise calculations, where even minor errors can lead to cascading failures. Furthermore, large language models (LLMs) encounter difficulties handling domain-specific formulas, executing reasoning steps accurately, and integrating code effectively when tackling chemical reasoning tasks. To address these challenges, we present ChemAgent, a novel framework designed to improve the performance of LLMs through a dynamic, self-updating library. This library is developed by decomposing chemical tasks into sub-tasks and compiling these sub-tasks into a structured collection that can be referenced for future queries. Then, when presented with a new problem, ChemAgent retrieves and refines pertinent information from the library, which we call memory, facilitating effective task decomposition and the generation of solutions. Our method designs three types of memory and a library-enhanced reasoning component, enabling LLMs to improve over time through experience. Experimental results on four chemical reasoning datasets from SciBench demonstrate that ChemAgent achieves performance gains of up to 46% (GPT-4), significantly outperforming existing methods. Our findings suggest substantial potential for future applications, including tasks such as drug discovery and materials science. Our code can be found at this https URL 

**Abstract (ZH)**: 化学推理通常涉及复杂、多步的过程，需要精确的计算，其中即使是轻微的错误也可能导致一系列的失败。此外，大型语言模型（LLMs）在处理特定领域的公式、准确执行推理步骤以及有效集成代码时，在应对化学推理任务时遇到困难。为了解决这些挑战，我们提出了ChemAgent，这是一种新的框架，旨在通过动态的自我更新库来提高LLMs的表现。该库通过将化学任务分解为子任务，并将这些子任务编译成一个有结构的集合来进行开发，该集合可以为未来的查询提供参考。当面临新的问题时，ChemAgent会从库（我们称为记忆）中检索和细化相关信息，从而促进有效的任务分解和解决方案的生成。我们的方法设计了三种类型的记忆和一个增强库的推理组件，使LLMs能够在经验中不断提高。SciBench的四种化学推理数据集的实验结果表明，ChemAgent可实现高达46%（GPT-4）的性能提升，显著优于现有方法。我们的研究结果表明，ChemAgent在药物发现和材料科学等领域的未来应用具有巨大的潜力。相关的代码可以在以下链接找到：[这个链接] 

---
# ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Drafting 

**Title (ZH)**: ACORD：一个由专家标注的法律合同起草检索数据集 

**Authors**: Steven H. Wang, Maksim Zubkov, Kexin Fan, Sarah Harrell, Yuyang Sun, Wei Chen, Andreas Plesner, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2501.06582)  

**Abstract**: Information retrieval, specifically contract clause retrieval, is foundational to contract drafting because lawyers rarely draft contracts from scratch; instead, they locate and revise the most relevant precedent. We introduce the Atticus Clause Retrieval Dataset (ACORD), the first retrieval benchmark for contract drafting fully annotated by experts. ACORD focuses on complex contract clauses such as Limitation of Liability, Indemnification, Change of Control, and Most Favored Nation. It includes 114 queries and over 126,000 query-clause pairs, each ranked on a scale from 1 to 5 stars. The task is to find the most relevant precedent clauses to a query. The bi-encoder retriever paired with pointwise LLMs re-rankers shows promising results. However, substantial improvements are still needed to effectively manage the complex legal work typically undertaken by lawyers. As the first retrieval benchmark for contract drafting annotated by experts, ACORD can serve as a valuable IR benchmark for the NLP community. 

**Abstract (ZH)**: 信息检索，特别是合同条款检索，是合同起草的基础，因为律师很少从头起草合同；相反，他们会查找并修改最相关的先例。我们介绍了阿提库斯合同条款检索数据集（ACORD），这是首个由专家全面标注的合同起草检索基准。ACORD 重点关注诸如责任限制、赔偿、控制变更和最惠国条款等复杂的合同条款。该数据集包括114 个查询和超过126,000 个查询-条款对，每个对都按1到5星的评级。任务是找到与查询最相关的先例条款。双编码检索器与面向点的大型语言模型重新排名器显示出良好的效果。然而，仍然需要大幅度改进以有效管理律师通常承担的复杂法律工作。作为首个由专家标注的合同起草检索基准，ACORD 可以为自然语言处理社区提供有价值的排序检索基准。 

---
# Natural Language Processing and Deep Learning Models to Classify Phase of Flight in Aviation Safety Occurrences 

**Title (ZH)**: 自然语言处理与深度学习模型在航空安全事件飞行阶段分类中的应用 

**Authors**: Aziida Nanyonga, Hassan Wasswa, Oleksandra Molloy, Ugur Turhan, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2501.06564)  

**Abstract**: The air transport system recognizes the criticality of safety, as even minor anomalies can have severe consequences. Reporting accidents and incidents play a vital role in identifying their causes and proposing safety recommendations. However, the narratives describing pre-accident events are presented in unstructured text that is not easily understood by computer systems. Classifying and categorizing safety occurrences based on these narratives can support informed decision-making by aviation industry stakeholders. In this study, researchers applied natural language processing (NLP) and artificial intelligence (AI) models to process text narratives to classify the flight phases of safety occurrences. The classification performance of two deep learning models, ResNet and sRNN was evaluated, using an initial dataset of 27,000 safety occurrence reports from the NTSB. The results demonstrated good performance, with both models achieving an accuracy exceeding 68%, well above the random guess rate of 14% for a seven-class classification problem. The models also exhibited high precision, recall, and F1 scores. The sRNN model greatly outperformed the simplified ResNet model architecture used in this study. These findings indicate that NLP and deep learning models can infer the flight phase from raw text narratives, enabling effective analysis of safety occurrences. 

**Abstract (ZH)**: 航空运输系统认识到安全的重要性，即使细微异常也可能导致严重后果。报告事故和事件对于确定其原因并提出安全建议发挥着重要作用。然而，描述事故发生前事件的叙述是未结构化的文本，难以被计算机系统理解。基于这些叙述对安全事件进行分类和归类，可以支持航空行业的利益相关者做出知情决策。本研究中，研究人员应用了自然语言处理（NLP）和人工智能（AI）模型来处理文本叙述，以分类安全事件所涉及的飞行阶段。使用来自美国国家运输安全委员会（NTSB）的初始数据集（包含27,000份安全事件报告），评估了两种深度学习模型——ResNet和简化RNN（sRNN）的分类性能。结果显示，这两种模型均实现了超过68%的准确性，远高于7类分类问题的随机猜测率14%。此外，两种模型还表现出较高的精确度、召回率和F1分数。简化RNN模型在这项研究中表现出显著优于简化ResNet模型架构的效果。这些发现表明，NLP和深度学习模型可以从原始文本叙述中推断出飞行阶段，从而有效分析安全事件。 

---
# A Survey on Spoken Italian Datasets and Corpora 

**Title (ZH)**: 关于 spoken Italian 数据集和语料库的综述 

**Authors**: Marco Giordano, Claudia Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2501.06557)  

**Abstract**: Spoken language datasets are vital for advancing linguistic research, Natural Language Processing, and speech technology. However, resources dedicated to Italian, a linguistically rich and diverse Romance language, remain underexplored compared to major languages like English or Mandarin. This survey provides a comprehensive analysis of 66 spoken Italian datasets, highlighting their characteristics, methodologies, and applications. The datasets are categorized by speech type, source and context, and demographic and linguistic features, with a focus on their utility in fields such as Automatic Speech Recognition, emotion detection, and education. Challenges related to dataset scarcity, representativeness, and accessibility are discussed alongside recommendations for enhancing dataset creation and utilization. The full dataset inventory is publicly accessible via GitHub and archived on Zenodo, serving as a valuable resource for researchers and developers. By addressing current gaps and proposing future directions, this work aims to support the advancement of Italian speech technologies and linguistic research. 

**Abstract (ZH)**: 语音数据集对于推进语言研究、自然语言处理和语音技术具有重要作用。然而，与英语或 Mandarin 等主要语言相比，用于意大利语（一种丰富多样的罗曼语族语言）的资源仍然相对匮乏。本文综述了 66 个语音意大利语数据集，重点分析了这些数据集的特征、方法论和应用。数据集根据语音类型、来源和场景、人口统计学和语言学特征进行了分类，并着重讨论了这些特征在自动语音识别、情绪检测和教育等领域的实用性。文章还探讨了数据集稀缺性、代表性及获取性方面的挑战，并提出了增强数据集创建和利用的建议。完整的数据集目录可通过 GitHub 公开访问，并存档于 Zenodo，为研究人员和开发人员提供了一项宝贵资源。通过解决当前存在的不足并提出未来的发展方向，本文旨在支持意大利语音技术及语言学研究的发展。 

---
# Dispersion Measures as Predictors of Lexical Decision Time, Word Familiarity, and Lexical Complexity 

**Title (ZH)**: dispersion 值作为词汇判断时间、词熟悉度和词汇复杂度的预测指标 

**Authors**: Adam Nohejl, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2501.06536)  

**Abstract**: Various measures of dispersion have been proposed to paint a fuller picture of a word's distribution in a corpus, but only little has been done to validate them externally. We evaluate a wide range of dispersion measures as predictors of lexical decision time, word familiarity, and lexical complexity in five diverse languages. We find that the logarithm of range is not only a better predictor than log-frequency across all tasks and languages, but that it is also the most powerful additional variable to log-frequency, consistently outperforming the more complex dispersion measures. We discuss the effects of corpus part granularity and logarithmic transformation, shedding light on contradictory results of previous studies. 

**Abstract (ZH)**: 各种离散性度量已被提出以描绘词在语料库中分布的更完整图景，但对外部验证方面的工作仍然很少。我们评估了一系列离散性度量作为预测词汇决策时间、词熟识度和词汇复杂性的指标，在五种不同的语言中进行了评估。我们发现，在所有任务和语言中，离散性范围的对数不仅比对数频率有更好的预测效果，而且在提高对数频率的预测能力方面更为显著，更复杂的离散性度量始终不能超过它。我们讨论了语料库部分粒度和对数变换的影响，阐明了以往研究中存在的一些矛盾结果。 

---
# Fine-tuning Large Language Models for Improving Factuality in Legal Question Answering 

**Title (ZH)**: 针对提高法律问答事实准确性的大规模语言模型微调 

**Authors**: Yinghao Hu, Leilei Gan, Wenyi Xiao, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06521)  

**Abstract**: Hallucination, or the generation of incorrect or fabricated information, remains a critical challenge in large language models (LLMs), particularly in high-stake domains such as legal question answering (QA). In order to mitigate the hallucination rate in legal QA, we first introduce a benchmark called LegalHalBench and three automatic metrics to evaluate the common hallucinations when LLMs answer legal questions. We then propose a hallucination mitigation method that integrates behavior cloning and a novel Hard Sample-aware Iterative Direct Preference Optimization (HIPO). We conduct extensive real-data experiments to validate the effectiveness of our approach. Our results demonstrate remarkable improvements in various metrics, including the newly proposed Non-Hallucinated Statute Rate, Statute Relevance Rate, Legal Claim Truthfulness, as well as traditional metrics such as METEOR, BERTScore, ROUGE-L, and win rates. 

**Abstract (ZH)**: 幻觉，即生成错误或虚构的信息，仍然是大型语言模型（LLMs）中的一个关键挑战，尤其是在法律问题回答（QA）等高风险领域。为了降低法律QA中的幻觉率，我们首先引入了一个基准称为LegalHalBench，并提出了三种自动指标来评估LLMs在回答法律问题时常见的幻觉。然后，我们提出了一种结合行为克隆和一种新颖的硬样本意识迭代直接偏好优化（HIPO）的幻觉缓解方法。我们进行了广泛的实证实验以验证我们方法的有效性。实验结果表明，我们在多个指标领域，包括新提出的无幻觉成文率、成文相关率、法律主张的真实性，以及传统的指标如METEOR、BERTScore、ROUGE-L和胜率等方面均取得了显著改进。 

---
# PASS: Presentation Automation for Slide Generation and Speech 

**Title (ZH)**: PASS：幻灯片生成与演讲的自动化呈现 

**Authors**: Tushar Aggarwal, Aarohi Bhand  

**Link**: [PDF](https://arxiv.org/pdf/2501.06497)  

**Abstract**: In today's fast-paced world, effective presentations have become an essential tool for communication in both online and offline meetings. The crafting of a compelling presentation requires significant time and effort, from gathering key insights to designing slides that convey information clearly and concisely. However, despite the wealth of resources available, people often find themselves manually extracting crucial points, analyzing data, and organizing content in a way that ensures clarity and impact. Furthermore, a successful presentation goes beyond just the slides; it demands rehearsal and the ability to weave a captivating narrative to fully engage the audience. Although there has been some exploration of automating document-to-slide generation, existing research is largely centered on converting research papers. In addition, automation of the delivery of these presentations has yet to be addressed. We introduce PASS, a pipeline used to generate slides from general Word documents, going beyond just research papers, which also automates the oral delivery of the generated slides. PASS analyzes user documents to create a dynamic, engaging presentation with an AI-generated voice. Additionally, we developed an LLM-based evaluation metric to assess our pipeline across three critical dimensions of presentations: relevance, coherence, and redundancy. The data and codes are available at this https URL. 

**Abstract (ZH)**: 在当今快节奏的世界中，有效的演示文稿已成为在线和线下会议中沟通的重要工具。精心制作的引人入胜的演示文稿需要大量时间和努力，从收集关键见解到设计能够清晰简洁地传达信息的幻灯片。然而，尽管有大量的资源可用，人们常常发现自己在手动提取关键点、分析数据和组织内容以确保清晰性和影响力上花费大量时间。此外，成功的演示不仅仅局限于幻灯片；它还需要排练以及能够编织引人入胜的故事的能力，以完全吸引观众。尽管已经有一些自动化文档到幻灯片生成的研究，但现有研究主要集中在转换研究论文上。此外，对这些演示文稿的自动化呈现还未被解决。我们引入了PASS，这是一种用于从通用Word文档生成幻灯片的管道，不仅限于研究论文，同时还可以自动化生成的幻灯片的口头呈现。PASS通过生成一个由AI生成声音的动态和引人入胜的演示文稿来分析用户文档。此外，我们还开发了一种基于大语言模型的评估指标，用于从三个关键维度评估我们的管道：相关性、连贯性和冗余性。数据和代码可在以下网址获取：**此处网址**。 

---
# Analyzing the Role of Context in Forecasting with Large Language Models 

**Title (ZH)**: 分析上下文在使用大规模语言模型进行预测中的作用 

**Authors**: Gerrit Mutschlechner, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2501.06496)  

**Abstract**: This study evaluates the forecasting performance of recent language models (LLMs) on binary forecasting questions. We first introduce a novel dataset of over 600 binary forecasting questions, augmented with related news articles and their concise question-related summaries. We then explore the impact of input prompts with varying level of context on forecasting performance. The results indicate that incorporating news articles significantly improves performance, while using few-shot examples leads to a decline in accuracy. We find that larger models consistently outperform smaller models, highlighting the potential of LLMs in enhancing automated forecasting. 

**Abstract (ZH)**: 本研究评估了近期语言模型（LLMs）在二元预测问题上的预测性能。我们首先引入了一个包含超过600个二元预测问题的新颖数据集，该数据集包含相关的新闻文章及其简洁的问题相关摘要。随后，我们探讨了不同水平上下文输入提示对预测性能的影响。结果表明，整合新闻文章显著提高了预测性能，而使用少量示例则导致准确率下降。我们发现，较大的模型持续优于较小的模型，这突显了LLMs在增强自动化预测方面的潜力。 

---
# Sequential Classification of Aviation Safety Occurrences with Natural Language Processing 

**Title (ZH)**: 使用自然语言处理对航空安全事件进行序列分类 

**Authors**: Aziida Nanyonga, Hassan Wasswa, Ugur Turhan, Oleksandra Molloy, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2501.06490)  

**Abstract**: Safety is a critical aspect of the air transport system given even slight operational anomalies can result in serious consequences. To reduce the chances of aviation safety occurrences, accidents and incidents are reported to establish the root cause, propose safety recommendations etc. However, analysis narratives of the pre-accident events are presented using human-understandable, raw, unstructured, text that a computer system cannot understand. The ability to classify and categorise safety occurrences from their textual narratives would help aviation industry stakeholders make informed safety-critical decisions. To classify and categorise safety occurrences, we applied natural language processing (NLP) and AI (Artificial Intelligence) models to process text narratives. The study aimed to answer the question. How well can the damage level caused to the aircraft in a safety occurrence be inferred from the text narrative using natural language processing. The classification performance of various deep learning models including LSTM, BLSTM, GRU, sRNN, and combinations of these models including LSTM and GRU, BLSTM+GRU, sRNN and LSTM, sRNN and BLSTM, sRNN and GRU, sRNN and BLSTM and GRU, and sRNN and LSTM and GRU was evaluated on a set of 27,000 safety occurrence reports from the NTSB. The results of this study indicate that all models investigated performed competitively well recording an accuracy of over 87.9% which is well above the random guess of 25% for a four-class classification problem. Also, the models recorded high precision, recall, and F1 scores above 80%, 88%, and 85%, respectively. sRNN slightly outperformed other single models in terms of recall (90%) and accuracy (90%) while LSTM reported slightly better performance in terms of precision (87%). 

**Abstract (ZH)**: 航空运输系统的安全性是一个至关重要的方面，即使是很小的操作异常也可能导致严重的后果。为了减少航空安全事件的发生，事故发生后会报告事故和事件以确定其根本原因，提出安全建议等。然而，事前事件的分析报告通常以人类可理解、原始且未结构化的文本形式呈现，计算机系统无法直接理解这些文本。从文本叙述中分类和归类安全事件的能力，可以帮助航空行业利益相关者做出知情的安全决策。为实现这一目标，我们应用自然语言处理（NLP）和人工智能（AI）模型来处理文本叙述。本研究旨在回答如下问题：如何利用自然语言处理从文本叙述中推断出航空器在安全事件中所受的损坏程度？我们对包括长短期记忆网络（LSTM）、双向长短期记忆网络（BLSTM）、门控循环单元（GRU）、自回归循环神经网络（sRNN）及其组合在内的各种深度学习模型的分类性能进行了评估，这些研究数据来源于美国国家运输安全委员会（NTSB）的27,000份航空安全事件报告。研究表明，所有研究的模型都能表现得非常出色，分类准确率达到了87.9%以上，远高于四类分类问题中的随机猜测的25%。同时，这些模型在精确率、召回率和F1评分方面分别达到了80%以上、88%和85%以上。sRNN在召回率（90%）和准确率（90%）方面略优于其他单一模型，而LSTM在精确率（87%）方面表现略好一些。 

---
# First Token Probability Guided RAG for Telecom Question Answering 

**Title (ZH)**: 基于首个标记概率的RAG在电信领域问答中的应用 

**Authors**: Tingwei Chen, Jiayi Chen, Zijian Zhao, Haolong Chen, Liang Zhang, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06468)  

**Abstract**: Large Language Models (LLMs) have garnered significant attention for their impressive general-purpose capabilities. For applications requiring intricate domain knowledge, Retrieval-Augmented Generation (RAG) has shown a distinct advantage in incorporating domain-specific information into LLMs. However, existing RAG research has not fully addressed the challenges of Multiple Choice Question Answering (MCQA) in telecommunications, particularly in terms of retrieval quality and mitigating hallucinations. To tackle these challenges, we propose a novel first token probability guided RAG framework. This framework leverages confidence scores to optimize key hyperparameters, such as chunk number and chunk window size, while dynamically adjusting the context. Our method starts by retrieving the most relevant chunks and generates a single token as the potential answer. The probabilities of all options are then normalized to serve as confidence scores, which guide the dynamic adjustment of the context. By iteratively optimizing the hyperparameters based on these confidence scores, we can continuously improve RAG performance. We conducted experiments to validate the effectiveness of our framework, demonstrating its potential to enhance accuracy in domain-specific MCQA tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其广泛的应用能力而引起了广泛关注。在需要复杂领域知识的应用中，检索增强生成（RAG）展示了将领域特定信息整合到LLMs中的独特优势。然而，现有的RAG研究尚未全面解决电信领域的多项选择题作答（MCQA）难题，尤其是在检索质量和缓解幻觉方面。为了解决这些问题，我们提出了一种新颖的一开始基于第一令牌概率的RAG框架。该框架利用置信度分数优化关键超参数，如片段数量和窗口大小，并动态调整上下文。我们的方法首先检索最相关的片段，生成一个潜在答案的单个令牌。然后将所有选项的概率正则化，作为置信度分数，以指导上下文的动态调整。通过迭代基于这些置信度分数优化超参数，可以不断提升RAG性能。我们进行了实验以验证该框架的有效性，展示了其在特定领域MCQA任务中提高准确性的潜力。 

---
# Retrieval-Augmented Dialogue Knowledge Aggregation for Expressive Conversational Speech Synthesis 

**Title (ZH)**: 检索增强对话知识聚合在表达性会话语音合成中的应用 

**Authors**: Rui Liu, Zhenqi Jia, Feilong Bao, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.06467)  

**Abstract**: Conversational speech synthesis (CSS) aims to take the current dialogue (CD) history as a reference to synthesize expressive speech that aligns with the conversational style. Unlike CD, stored dialogue (SD) contains preserved dialogue fragments from earlier stages of user-agent interaction, which include style expression knowledge relevant to scenarios similar to those in CD. Note that this knowledge plays a significant role in enabling the agent to synthesize expressive conversational speech that generates empathetic feedback. However, prior research has overlooked this aspect. To address this issue, we propose a novel Retrieval-Augmented Dialogue Knowledge Aggregation scheme for expressive CSS, termed RADKA-CSS, which includes three main components: 1) To effectively retrieve dialogues from SD that are similar to CD in terms of both semantic and style. First, we build a stored dialogue semantic-style database (SDSSD) which includes the text and audio samples. Then, we design a multi-attribute retrieval scheme to match the dialogue semantic and style vectors of the CD with the stored dialogue semantic and style vectors in the SDSSD, retrieving the most similar dialogues. 2) To effectively utilize the style knowledge from CD and SD, we propose adopting the multi-granularity graph structure to encode the dialogue and introducing a multi-source style knowledge aggregation mechanism. 3) Finally, the aggregated style knowledge are fed into the speech synthesizer to help the agent synthesize expressive speech that aligns with the conversational style. We conducted a comprehensive and in-depth experiment based on the DailyTalk dataset, which is a benchmarking dataset for the CSS task.
Both objective and subjective evaluations demonstrate that RADKA-CSS outperforms baseline models in expressiveness rendering. Code and audio samples can be found at: this https URL. 

**Abstract (ZH)**: 对话式语音合成（Conversational Speech Synthesis, CSS）的目标是将当前对话（Current Dialogue, CD）的历史作为参考，生成与对话风格相匹配的富有表现力的语音。与当前对话（CD）不同，存储对话（Stored Dialogue, SD）包含用户-代理交互早期阶段保存的对话片段，这些片段中包含了与CD相似场景中的风格表达知识。值得注意的是，这些知识在使代理能够合成能够产生同理心反馈的富有表现力的对话式语音中起着重要作用。然而，先前的研究并未充分考虑到这一点。为了解决这个问题，我们提出了一种新的检索增强对话知识聚合方案，以实现富有表现力的CSS，称为RADKA-CSS，包括三个主要组成部分：

1. **有效检索相似对话**：首先，构建一个存储对话语义-风格数据库（SDSSD），该数据库包含文本和音频样本。然后，设计一个多属性检索方案，将CD的语义和风格向量与SDSSD中的存储对话的语义和风格向量进行匹配，检索最相似的对话。
   
2. **有效利用语义和风格知识**：我们提出采用多层次图结构来编码对话，并引入多源风格知识聚合机制，以有效利用CD和SD中的风格知识。
   
3. **聚合后的风格知识输入语音合成器**：最后，将聚合后的风格知识输入到语音合成器中，帮助代理生成与对话风格相匹配的富有表现力的语音。

我们在DailyTalk数据集上进行了全面深入的实验，该数据集是CSS任务的一个基准数据集。

客观和主观评价均表明，RADKA-CSS在表现力呈现方面优于基线模型。代码和音频样本可在以下链接找到：[此链接](this https URL)。

*注：请将上述“此链接”替换为实际的URL链接以获取相关资料。* 

---
# MedCT: A Clinical Terminology Graph for Generative AI Applications in Healthcare 

**Title (ZH)**: MedCT：医疗术语图谱在医疗健康生成式AI应用中的临床术语图 

**Authors**: Ye Chen, Dongdong Huang, Haoyun Xu, Cong Fu, Lin Sheng, Qingli Zhou, Yuqiang Shen, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06465)  

**Abstract**: We introduce the world's first clinical terminology for the Chinese healthcare community, namely MedCT, accompanied by a clinical foundation model MedBERT and an entity linking model MedLink. The MedCT system enables standardized and programmable representation of Chinese clinical data, successively stimulating the development of new medicines, treatment pathways, and better patient outcomes for the populous Chinese community. Moreover, the MedCT knowledge graph provides a principled mechanism to minimize the hallucination problem of large language models (LLMs), therefore achieving significant levels of accuracy and safety in LLM-based clinical applications. By leveraging the LLMs' emergent capabilities of generativeness and expressiveness, we were able to rapidly built a production-quality terminology system and deployed to real-world clinical field within three months, while classical terminologies like SNOMED CT have gone through more than twenty years development. Our experiments show that the MedCT system achieves state-of-the-art (SOTA) performance in semantic matching and entity linking tasks, not only for Chinese but also for English. We also conducted a longitudinal field experiment by applying MedCT and LLMs in a representative spectrum of clinical tasks, including electronic health record (EHR) auto-generation and medical document search for diagnostic decision making. Our study shows a multitude of values of MedCT for clinical workflows and patient outcomes, especially in the new genre of clinical LLM applications. We present our approach in sufficient engineering detail, such that implementing a clinical terminology for other non-English societies should be readily reproducible. We openly release our terminology, models and algorithms, along with real-world clinical datasets for the development. 

**Abstract (ZH)**: 我们向中国的医疗社区引入了世界首款临床术语体系，称之为MedCT，并配以临床基础模型MedBERT和实体链接模型MedLink。MedCT系统能够规范且编程式的表示中文临床数据，从而驱动新的药物研发、治疗路径设计以及改善庞大中国患者群体的健康结果。此外，MedCT知识图谱提供了一种从根本上减少大型语言模型（LLMs）幻觉问题的机制，从而在基于LLMs的临床应用中实现了显著的准确性和安全性。通过利用LLMs生成性和表达性的新兴能力，我们能够在三个月内快速构建出生产级的术语体系，并将其部署到真实的临床领域，而像SNOMED CT这样的传统术语体系则经历了超过二十年的发展。我们的实验表明，MedCT系统在语义匹配和实体链接任务上均达到了现有最佳（SOTA）性能，不仅适用于中文，也适用于英文。我们还进行了长期现场实验，将MedCT和LLMs应用于代表性的多种临床任务，包括电子健康记录（EHR）自动化生成和医学文档搜索以支持诊断决策。研究结果表明，MedCT在临床工作流程和患者结果方面具有多种价值，尤其是在临床LLMs应用的新领域。我们以足够的工程技术细节介绍了我们的方法，使得为其他非英文社会构建临床术语体系变得可重现。我们公开发布了我们的术语体系、模型和算法，并提供了实际的临床数据集，以促进其发展。 

---
# O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning 

**Title (ZH)**: O1 复现之旅——第3部分：推理时的扩展性在医疗推理中的应用 

**Authors**: Zhongzhen Huang, Gui Geng, Shengyi Hua, Zhen Huang, Haoyang Zou, Shaoting Zhang, Pengfei Liu, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06458)  

**Abstract**: Building upon our previous investigations of O1 replication (Part 1: Journey Learning [Qin et al., 2024] and Part 2: Distillation [Huang et al., 2024]), this work explores the potential of inference-time scaling in large language models (LLMs) for medical reasoning tasks, ranging from diagnostic decision-making to treatment planning. Through extensive experiments on medical benchmarks of varying complexity (MedQA, Medbullets, and JAMA Clinical Challenges), our investigation reveals several key insights: (1) Increasing inference time does lead to improved performance. With a modest training set of 500 samples, our model yields substantial performance improvements of 6%-11%. (2) Task complexity directly correlates with the required length of reasoning chains, confirming the necessity of extended thought processes for challenging problems. (3) The differential diagnoses generated by our model adhere to the principles of the hypothetico-deductive method, producing a list of potential conditions that may explain a patient's symptoms and systematically narrowing these possibilities by evaluating the evidence. These findings demonstrate the promising synergy between inference-time scaling and journey learning in advancing LLMs' real-world clinical reasoning capabilities. 

**Abstract (ZH)**: 基于我们之前对O1复制的研究（Part 1：旅程学习 [Qin et al., 2024] 和 Part 2：知识萃取 [Huang et al., 2024]），本研究探讨了大型语言模型（LLMs）在医疗推理任务中的推理时扩展潜力，涵盖从诊断决策到治疗计划的多个方面。通过在不同复杂度的医学基准数据集（MedQA、Medbullets 和 JAMA Clinical Challenges）上进行广泛实验，我们的研究表明以下几个关键见解：（1）增加推理时间确实可以提高性能。仅用500个样本的适度训练集，我们的模型在性能上获得了6%-11%的重大提升。（2）任务复杂性直接与所需的推理链长度相关，证实了对于复杂问题有必要进行更长的思考过程。（3）我们的模型生成的鉴别诊断方案遵循假设演绎方法的原则，产生一系列可能解释患者症状的条件，并通过评估证据系统地缩小这些可能性。这些发现展示了推理时扩展与旅程学习之间有前景的协同作用，有助于推动LLMs在临床推理中的实际应用能力。 

---
# Synthetic Feature Augmentation Improves Generalization Performance of Language Models 

**Title (ZH)**: 合成特征增强改善了语言模型的泛化性能 

**Authors**: Ashok Choudhary, Cornelius Thiels, Hojjat Salehinejad  

**Link**: [PDF](https://arxiv.org/pdf/2501.06434)  

**Abstract**: Training and fine-tuning deep learning models, especially large language models (LLMs), on limited and imbalanced datasets poses substantial challenges. These issues often result in poor generalization, where models overfit to dominant classes and underperform on minority classes, leading to biased predictions and reduced robustness in real-world applications. To overcome these challenges, we propose augmenting features in the embedding space by generating synthetic samples using a range of techniques. By upsampling underrepresented classes, this method improves model performance and alleviates data imbalance. We validate the effectiveness of this approach across multiple open-source text classification benchmarks, demonstrating its potential to enhance model robustness and generalization in imbalanced data scenarios. 

**Abstract (ZH)**: 在有限和不平衡数据集上训练和微调深度学习模型，尤其是大规模语言模型（LLMs），面临着重大挑战。这些问题通常会导致模型过度拟合主流类别，而对少数类别性能不佳，从而产生有偏的预测并降低实际应用中的鲁棒性。为克服这些挑战，我们提出在嵌入空间中通过生成合成样本来增强特征，使用多种技术来上采样不足代表的类别。这种方法可以提高模型性能并缓解数据不平衡问题。我们通过跨多个开源文本分类基准验证了该方法的有效性，展示了其在不平衡数据场景中增强模型鲁棒性和泛化能力的潜力。 

---
# Tensor Product Attention Is All You Need 

**Title (ZH)**: 张量积注意力机制即你所需 

**Authors**: Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Zhen Qin, Yang Yuan, Quanquan Gu, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2501.06425)  

**Abstract**: Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor Product Attention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, significantly shrinking KV cache size at inference time. By factorizing these representations into contextual low-rank components (contextual factorization) and seamlessly integrating with RoPE, TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the Tensor ProducT ATTenTion Transformer (T6), a new model architecture for sequence modeling. Through extensive empirical evaluation of language modeling tasks, we demonstrate that T6 exceeds the performance of standard Transformer baselines including MHA, MQA, GQA, and MLA across various metrics, including perplexity and a range of renowned evaluation benchmarks. Notably, TPAs memory efficiency enables the processing of significantly longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. The code is available at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

将语言模型扩展以处理更长的输入序列通常需要大量的键-值（KV）缓存，从而导致推断时产生较大内存开销。本文提出了一种名为张量积注意力（TPA）的新型注意力机制，该机制利用张量分解来紧凑地表示查询、键和值，显著减小了推断时的KV缓存大小。通过将这些表示分解为上下文低秩组件（上下文因式分解）并无缝集成RoPE，TPA 实现了模型质量的提升和内存效率的兼顾。基于TPA，我们引入了张量积注意力变换器（T6）这一新的序列建模架构。通过对多种语言建模任务进行广泛的实证评估，我们展示了T6在各种指标上，包括困惑度和一系列知名的评估基准上，大大超过了包括MHA、MQA、GQA和MLA在内的标准Transformer基线模型。特别值得一提的是，TPA 的内存效率使得在固定资源约束条件下可以处理显著更长的序列，从而解决了现代语言模型中的一个关键可扩展性挑战。完整的代码可通过此链接获取：this https URL。 

---
# Dynamics of "Spontaneous" Topic Changes in Next Token Prediction with Self-Attention 

**Title (ZH)**: “自发”主题变化在自我注意力下的下一个词预测动态分析 

**Authors**: Mumin Jia, Jairo Diaz-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2501.06382)  

**Abstract**: Human cognition can spontaneously shift conversation topics, often triggered by emotional or contextual signals. In contrast, self-attention-based language models depend on structured statistical cues from input tokens for next-token prediction, lacking this spontaneity. Motivated by this distinction, we investigate the factors that influence the next-token prediction to change the topic of the input sequence. We define concepts of topic continuity, ambiguous sequences, and change of topic, based on defining a topic as a set of token priority graphs (TPGs). Using a simplified single-layer self-attention architecture, we derive analytical characterizations of topic changes. Specifically, we demonstrate that (1) the model maintains the priority order of tokens related to the input topic, (2) a topic change occurs only if lower-priority tokens outnumber all higher-priority tokens of the input topic, and (3) unlike human cognition, longer context lengths and overlapping topics reduce the likelihood of spontaneous redirection. These insights highlight differences between human cognition and self-attention-based models in navigating topic changes and underscore the challenges in designing conversational AI capable of handling "spontaneous" conversations more naturally. To our knowledge, this is the first work to address these questions in such close relation to human conversation and thought. 

**Abstract (ZH)**: 人类认知可以不自主地转换对话主题，通常由情绪或情境信号触发。相比之下，基于自注意力的语言模型依赖于输入标记的结构化统计线索来进行下一个标记预测，缺乏这种自发性。鉴于这种区别，我们研究影响下一个标记预测以改变输入序列主题的因素。我们根据将主题定义为标记优先图集合（TPGs）来定义主题连续性、模糊序列和主题变化的概念。使用简化的一层自注意力架构，我们推导出主题变化的分析表征。具体而言，我们展示了以下几点：（1）模型维持与输入主题相关的标记的优先顺序；（2）只有当低优先级标记的数量超过所有输入主题的高优先级标记时，才会发生主题变化；（3）与人类认知不同，更长的上下文长度和重叠主题会减少自发转移的几率。这些见解突显了人类认知和基于自注意力的模型在导航主题变化方面的差异，并强调了设计能够更自然地处理“自发”对话的对话人工智能所面临的挑战。据我们所知，这是第一次在如此密切联系人类对话和思考的问题上进行此类研究的工作。 

---
# AFRIDOC-MT: Document-level MT Corpus for African Languages 

**Title (ZH)**: AFRIDOC-MT：非洲语言文档级机器翻译语料库 

**Authors**: Jesujoba O. Alabi, Israel Abebe Azime, Miaoran Zhang, Cristina España-Bonet, Rachel Bawden, Dawei Zhu, David Ifeoluwa Adelani, Clement Oyeleke Odoje, Idris Akinade, Iffat Maab, Davis David, Shamsuddeen Hassan Muhammad, Neo Putini, David O. Ademuyiwa, Andrew Caines, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2501.06374)  

**Abstract**: This paper introduces AFRIDOC-MT, a document-level multi-parallel translation dataset covering English and five African languages: Amharic, Hausa, Swahili, Yorùbá, and Zulu. The dataset comprises 334 health and 271 information technology news documents, all human-translated from English to these languages. We conduct document-level translation benchmark experiments by evaluating neural machine translation (NMT) models and large language models (LLMs) for translations between English and these languages, at both the sentence and pseudo-document levels. These outputs are realigned to form complete documents for evaluation. Our results indicate that NLLB-200 achieved the best average performance among the standard NMT models, while GPT-4o outperformed general-purpose LLMs. Fine-tuning selected models led to substantial performance gains, but models trained on sentences struggled to generalize effectively to longer documents. Furthermore, our analysis reveals that some LLMs exhibit issues such as under-generation, repetition of words or phrases, and off-target translations, especially for African languages. 

**Abstract (ZH)**: 本文介绍了AFRIDOC-MT数据集，这是一个涵盖英语和五个非洲语言（阿姆哈拉语、豪萨语、斯瓦希里语、约鲁巴语和祖鲁语）的文档级多平行翻译数据集。该数据集包含334篇健康和271篇信息技术新闻文档，所有文档均为从英语翻译至这些语言的人工翻译。我们通过评估神经机器翻译（NMT）模型和大型语言模型（LLMs），在英语与这些语言之间的翻译进行了文档级基准实验，评估对象包括句子级和伪文档级翻译。这些输出被重新对齐，以形成完整的文档进行评估。结果显示，NLLB-200 在标准NMT模型中获得了最佳平均性能，而GPT-4o在通用LLM中表现更佳。对某些模型进行微调后取得了显著的性能提升，但以句子为单位训练的模型在处理更长文档时难以有效泛化。此外，我们的分析发现，部分LLM存在生成不足、重复词语或短语、目标外翻译等问题，尤其是在处理非洲语言时更为明显。 

---
# Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts 

**Title (ZH)**: 面向医疗应用的性别中立大型语言模型：减少PubMed摘要中的偏见 

**Authors**: Elizabeth Schaefer, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2501.06365)  

**Abstract**: This paper presents a pipeline for mitigating gender bias in large language models (LLMs) used in medical literature by neutralizing gendered occupational pronouns. A dataset of 379,000 PubMed abstracts from 1965-1980 was processed to identify and modify pronouns tied to professions. We developed a BERT-based model, ``Modern Occupational Bias Elimination with Refined Training,'' or ``MOBERT,'' trained on these neutralized abstracts, and compared its performance with ``1965Bert,'' trained on the original dataset. MOBERT achieved a 70\% inclusive replacement rate, while 1965Bert reached only 4\%. A further analysis of MOBERT revealed that pronoun replacement accuracy correlated with the frequency of occupational terms in the training data. We propose expanding the dataset and refining the pipeline to improve performance and ensure more equitable language modeling in medical applications. 

**Abstract (ZH)**: 本文提出了一种管道方法，用于减轻医学文献中大型语言模型（LLMs）中的性别偏差，通过对职业性代词进行中性化处理。我们处理了一个包含1965年至1980年间379,000篇PubMed摘要的数据集，以识别并修改与职业相关的代词。我们开发了一个基于BERT的模型，名为“改进训练的现代职业偏见消除”，或简称“MOBERT”，该模型在这些中性化摘要上进行了训练，并将其性能与在原始数据集上训练的“1965Bert”进行了比较。MOBERT实现了70%的包容性替换率，而1965Bert仅为4%。进一步的分析表明，代词替换的准确性与训练数据中职业术语的频率相关。我们建议扩大数据集并完善管道，以提高性能并确保在医学应用中的更公平的语言模型。 

---
# Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages 

**Title (ZH)**: 大型语言模型在类型学上不同的语言中共享潜在语法概念的表示 

**Authors**: Jannik Brinkmann, Chris Wendler, Christian Bartelt, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2501.06346)  

**Abstract**: Human bilinguals often use similar brain regions to process multiple languages, depending on when they learned their second language and their proficiency. In large language models (LLMs), how are multiple languages learned and encoded? In this work, we explore the extent to which LLMs share representations of morphosyntactic concepts such as grammatical number, gender, and tense across languages. We train sparse autoencoders on Llama-3-8B and Aya-23-8B, and demonstrate that abstract grammatical concepts are often encoded in feature directions shared across many languages. We use causal interventions to verify the multilingual nature of these representations; specifically, we show that ablating only multilingual features decreases classifier performance to near-chance across languages. We then use these features to precisely modify model behavior in a machine translation task; this demonstrates both the generality and selectivity of these feature's roles in the network. Our findings suggest that even models trained predominantly on English data can develop robust, cross-lingual abstractions of morphosyntactic concepts. 

**Abstract (ZH)**: 人类双母语者在处理多种语言时常常使用相似的大脑区域，这取决于他们学习第二语言的时间以及他们的熟练程度。在大规模语言模型（LLMs）中，是如何学习和编码多种语言的？在这项研究中，我们探讨了LLMs在多大程度上共享表示如语法数、性状和时态等形态语法概念。我们对Llama-3-8B和Aya-23-8B进行了稀疏自编码器训练，并发现抽象的语法概念经常以跨多语言共享的特征方向来编码。我们通过因果干预验证了这些表示的多语言性质；具体来说，我们展示了仅消除多语言特征会导致跨语言分类器性能下降至接近随机水平。随后，我们利用这些特征精确地修改了模型在机器翻译任务中的行为；这表明这些特征在模型中的作用具有广泛性和选择性。我们的研究发现表明，即使模型主要在英语数据上训练，也能够发展出稳健的跨语言形态语法概念抽象。 

---
# Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks 

**Title (ZH)**: Bactrainus：优化大型语言模型以应对多跳复杂问答任务 

**Authors**: Iman Barati, Arash Ghafouri, Behrouz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2501.06286)  

**Abstract**: In recent years, the use of large language models (LLMs) has significantly increased, and these models have demonstrated remarkable performance in a variety of general language tasks. However, the evaluation of their performance in domain-specific tasks, particularly those requiring deep natural language understanding, has received less attention. In this research, we evaluate the ability of large language models in performing domain-specific tasks, focusing on the multi-hop question answering (MHQA) problem using the HotpotQA dataset. This task, due to its requirement for reasoning and combining information from multiple textual sources, serves as a challenging benchmark for assessing the language comprehension capabilities of these models. To tackle this problem, we have designed a two-stage selector-reader architecture, where each stage utilizes an independent LLM. In addition, methods such as Chain of Thought (CoT) and question decomposition have been employed to investigate their impact on improving the model's performance. The results of the study show that the integration of large language models with these techniques can lead to up to a 4% improvement in F1 score for finding answers, providing evidence of the models' ability to handle domain-specific tasks and their understanding of complex language. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的使用显著增加，这些模型在各种通用语言任务中表现出色。然而，这些模型在领域特定任务中的评估，尤其是在要求深度自然语言理解的情况下，受到的关注较少。本研究旨在评估大规模语言模型在执行领域特定任务的能力，特别是在使用HotpotQA数据集评估多跳问答（MHQA）问题方面的表现。由于该任务需要进行推理并结合来自多个文本源的信息，它为评估这些模型的语言理解能力提供了一个具有挑战性的基准。为了解决这个问题，我们设计了一个两阶段选择器-阅读器架构，其中每个阶段都使用独立的LLM。此外，我们还采用了链式思维（Chain of Thought，CoT）和问题分解等方法，以研究这些方法对提升模型性能的影响。研究结果表明，将这些技术与大规模语言模型相结合，可以带来高达4%的F1分数改进，这证明了这些模型能够处理领域特定任务并对复杂语言的理解能力。 

---
# MinMo: A Multimodal Large Language Model for Seamless Voice Interaction 

**Title (ZH)**: MinMo：一种支持无缝语音交互的多模态大型语言模型 

**Authors**: Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, Yabin Li, Xiang Lv, Jiaqing Liu, Haoneng Luo, Bin Ma, Chongjia Ni, Xian Shi, Jialong Tang, Hui Wang, Hao Wang, Wen Wang, Yuxuan Wang, Yunlan Xu, Fan Yu, Zhijie Yan, Yexin Yang, Baosong Yang, Xian Yang, Guanrou Yang, Tianyu Zhao, Qinglin Zhang, Shiliang Zhang, Nan Zhao, Pei Zhang, Chong Zhang, Jinren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.06282)  

**Abstract**: Recent advancements in large language models (LLMs) and multimodal speech-text models have laid the groundwork for seamless voice interactions, enabling real-time, natural, and human-like conversations. Previous models for voice interactions are categorized as native and aligned. Native models integrate speech and text processing in one framework but struggle with issues like differing sequence lengths and insufficient pre-training. Aligned models maintain text LLM capabilities but are often limited by small datasets and a narrow focus on speech tasks. In this work, we introduce MinMo, a Multimodal Large Language Model with approximately 8B parameters for seamless voice interaction. We address the main limitations of prior aligned multimodal models. We train MinMo through multiple stages of speech-to-text alignment, text-to-speech alignment, speech-to-speech alignment, and duplex interaction alignment, on 1.4 million hours of diverse speech data and a broad range of speech tasks. After the multi-stage training, MinMo achieves state-of-the-art performance across various benchmarks for voice comprehension and generation while maintaining the capabilities of text LLMs, and also facilitates full-duplex conversation, that is, simultaneous two-way communication between the user and the system. Moreover, we propose a novel and simple voice decoder that outperforms prior models in voice generation. The enhanced instruction-following capabilities of MinMo supports controlling speech generation based on user instructions, with various nuances including emotions, dialects, and speaking rates, and mimicking specific voices. For MinMo, the speech-to-text latency is approximately 100ms, full-duplex latency is approximately 600ms in theory and 800ms in practice. The MinMo project web page is this https URL, and the code and models will be released soon. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）和多模态语音-文本模型的发展为无缝语音交互打下了基础，使其能够实现实时、自然和类人的对话。此前用于语音交互的模型主要分为两类：原生模型和对齐模型。原生模型将语音和文本处理融合在一个框架中，但面临序列长度不一致和预训练不足等问题。对齐模型保留了文本LLM的能力，但常常受限于小的数据集和范围狭窄的语音任务。在这项工作中，我们提出了一种名为MinMo的多模态大型语言模型，参数量约为8亿，旨在实现无缝语音交互。我们解决了此前对齐多模态模型的主要限制。通过多次阶段的语音到文本对齐、文本到语音对齐、语音到语音对齐以及双向交互对齐训练，MinMo在140万小时的多样语音数据和广泛的语音任务上进行了训练。经过多阶段训练后，MinMo在各种语音理解和生成基准测试中表现出最先进的性能，同时保持了文本LLM的能力，并实现了全双工对话，即用户与系统之间的双向通信。此外，我们提出了一种新的简单语音解码器，表现优于之前的模型。MinMo增强的指令遵循能力支持基于用户指令控制语音生成，包括情感、方言、说话速度等多种细微差别，并能够模拟特定的声音。对于MinMo，语音到文本的延迟约为100毫秒，理论上的全双工延迟约为600毫秒，实际应用中约为800毫秒。MinMo项目的网页地址是 <https://XXXXX>，代码和模型将在不久后发布。 

---
# Punctuation's Semantic Role between Brain and Transformers Models 

**Title (ZH)**: 脑与 transformers 模型之间标点符号的语义角色 

**Authors**: Zenon Lamprou, Frank Polick, Yashar Moshfeghi  

**Link**: [PDF](https://arxiv.org/pdf/2501.06278)  

**Abstract**: Contemporary neural networks intended for natural language processing (NLP) are not designed with specific linguistic rules. It suggests that they may acquire a general understanding of language. This attribute has led to extensive research in deciphering their internal representations. A pioneering method involves an experimental setup using human brain data to explore if a translation between brain and neural network representations can be established. Since this technique emerged, more sophisticated NLP models have been developed. In our study, we apply this method to evaluate four new NLP models aiming to identify the one most compatible with brain activity. Additionally, to explore how the brain comprehends text semantically, we alter the text by removing punctuation in four different ways to understand its impact on semantic processing by the human brain. Our findings indicate that the RoBERTa model aligns best with brain activity, outperforming BERT in accuracy according to our metrics. Furthermore, for BERT, higher accuracy was noted when punctuation was excluded, and increased context length did not significantly diminish accuracy compared to the original results with punctuation. 

**Abstract (ZH)**: 当代用于自然语言处理（NLP）的神经网络并未特别设计以遵循特定的语言规则，这表明它们可能获得了一般性的语言理解能力。这一特性促使研究人员在解码其内部表示方面进行了广泛的研究。一种开创性的方法是使用人类大脑数据的实验设置，探索是否可以建立大脑表示与神经网络表示之间的转换。自从这种方法出现以来，更先进的NLP模型得到了开发。在我们的研究中，我们应用这种方法评估了四种新的NLP模型，以确定哪一个最符合大脑活动。此外，为了探讨大脑如何进行文本的语义理解，我们通过四种不同的方法去除文本中的标点符号，以便更好地理解标点符号对人类大脑进行语义处理的影响。我们的研究结果表明，RoBERTa模型与大脑活动最为吻合，在我们设定的指标中其准确率超过了BERT。此外，对于BERT模型，在去除标点符号的情况下，其准确率更高，并且增加上下文长度并未显著降低其准确率，相比于带有标点符号的原始结果。 

---
# Environmental large language model Evaluation (ELLE) dataset: A Benchmark for Evaluating Generative AI applications in Eco-environment Domain 

**Title (ZH)**: 环境大型语言模型评估（ELLE）数据集：生态环境领域生成式AI应用评价基准 

**Authors**: Jing Guo, Nan Li, Ming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06277)  

**Abstract**: Generative AI holds significant potential for ecological and environmental applications such as monitoring, data analysis, education, and policy support. However, its effectiveness is limited by the lack of a unified evaluation framework. To address this, we present the Environmental Large Language model Evaluation (ELLE) question answer (QA) dataset, the first benchmark designed to assess large language models and their applications in ecological and environmental sciences. The ELLE dataset includes 1,130 question answer pairs across 16 environmental topics, categorized by domain, difficulty, and type. This comprehensive dataset standardizes performance assessments in these fields, enabling consistent and objective comparisons of generative AI performance. By providing a dedicated evaluation tool, ELLE dataset promotes the development and application of generative AI technologies for sustainable environmental outcomes. The dataset and code are available at this https URL and this https URL. 

**Abstract (ZH)**: 生成式AI在生态和环境应用方面（如监测、数据分析、教育和支持政策制定）具有重要的潜在价值。然而，其有效性受限于缺乏统一的评估框架。为了解决这一问题，我们提出了环境大型语言模型评估（ELLE）问答（QA）数据集，这是首个旨在评估大型语言模型及其在生态与环境科学中的应用的基准数据集。ELLE数据集包含1,130个问答对，覆盖16个环境主题，并按领域、难度和类型进行分类。该综合数据集标准化了这些领域的性能评估，使不同生成式AI技术的性能比较得以一致且客观进行。通过提供专门的评估工具，ELLE数据集促进了生成式AI技术在可持续环境成果中的发展与应用。该数据集和代码可以在以下链接获取：[此处链接] 和 [此处链接]。 

---
# AgoraSpeech: A multi-annotated comprehensive dataset of political discourse through the lens of humans and AI 

**Title (ZH)**: AgoraSpeech：人类与AI视角下的政治话语多标注综合数据集 

**Authors**: Pavlos Sermpezis, Stelios Karamanidis, Eva Paraschou, Ilias Dimitriadis, Sofia Yfantidou, Filitsa-Ioanna Kouskouveli, Thanasis Troboukis, Kelly Kiki, Antonis Galanopoulos, Athena Vakali  

**Link**: [PDF](https://arxiv.org/pdf/2501.06265)  

**Abstract**: Political discourse datasets are important for gaining political insights, analyzing communication strategies or social science phenomena. Although numerous political discourse corpora exist, comprehensive, high-quality, annotated datasets are scarce. This is largely due to the substantial manual effort, multidisciplinarity, and expertise required for the nuanced annotation of rhetorical strategies and ideological contexts. In this paper, we present AgoraSpeech, a meticulously curated, high-quality dataset of 171 political speeches from six parties during the Greek national elections in 2023. The dataset includes annotations (per paragraph) for six natural language processing (NLP) tasks: text classification, topic identification, sentiment analysis, named entity recognition, polarization and populism detection. A two-step annotation was employed, starting with ChatGPT-generated annotations and followed by exhaustive human-in-the-loop validation. The dataset was initially used in a case study to provide insights during the pre-election period. However, it has general applicability by serving as a rich source of information for political and social scientists, journalists, or data scientists, while it can be used for benchmarking and fine-tuning NLP and large language models (LLMs). 

**Abstract (ZH)**: 政治话语数据集对于获得政治洞察、分析沟通策略或社会科学现象至关重要。尽管存在众多政治话语语料库，但全面且高质量的标注数据集仍显稀缺。这主要是因为精细标注修辞策略和意识形态背景需要大量的手工努力、跨学科知识和专业技能。在此论文中，我们展示了AgoraSpeech，一个严格按照要求编纂并高质量的希腊全国选举期间六个政党政治演讲数据集，包含171篇演讲。该数据集包括每段文字的六项自然语言处理（NLP）任务标注：文本分类、主题识别、情感分析、命名实体识别、极化和民粹主义检测。我们采用了两步标注方法，首先使用ChatGPT生成标注，然后进行详尽的人工验证。该数据集最初用于选举前的研究案例，提供了宝贵的见解。然而，它具有广泛的适用性，作为政治和社会科学家、记者或数据科学家的丰富信息来源，也可用于评估和微调自然语言处理和大规模语言模型（LLMs）。 

---
# What Matters for In-Context Learning: A Balancing Act of Look-up and In-Weight Learning 

**Title (ZH)**: 影响上下文学习的关键因素：查阅学习与内置学习之间的平衡ﭰ
user
把这句话翻译成英文：本文探讨了如何通过对齐网络权重来促进转移学习。 

**Authors**: Jelena Bratulić, Sudhanshu Mittal, Christian Rupprecht, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2501.06256)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance in various tasks, including In-Context Learning (ICL), where the model performs new tasks by conditioning solely on the examples provided in the context, without updating the model's weights. While prior research has explored the roles of pretraining data and model architecture, the key mechanism behind ICL remains unclear. In this work, we systematically uncover properties present in LLMs that support the emergence of ICL. To disambiguate these factors, we conduct a study with a controlled dataset and data sequences using a deep autoregressive model. We show that conceptual repetitions in the data sequences are crucial for ICL, more so than previously indicated training data properties like burstiness or long-tail distribution. Conceptual repetitions could refer to $n$-gram repetitions in textual data or exact image copies in image sequence data. Such repetitions also offer other previously overlooked benefits such as reduced transiency in ICL performance. Furthermore, we show that the emergence of ICL depends on balancing the in-weight learning objective with the in-context solving ability during training. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现出令人印象深刻的性能，包括上下文内学习（ICL），即模型通过仅依赖于上下文中的示例来进行新的任务学习，而不更新模型的权重。尽管先前的研究已经探索了预训练数据和模型架构的作用，但ICL背后的机制仍然不清楚。在此项工作中，我们系统地揭示了LLMs中支持ICL出现的特性。为了澄清这些因素，我们在控制数据集和数据序列上使用深度自回归模型进行了研究。结果显示，数据序列中的概念重复对于ICL至关重要，比之前所指出的训练数据属性（如突发性或长尾分布）更重要。概念重复可以是指文本数据中的$n$-gram重复或图像序列数据中的精确图像副本。这些重复还提供了其他先前未被注意到的好处，例如减少了ICL性能的波动性。此外，我们表明，ICL的出现依赖于训练过程中平衡内在权重学习目标与上下文内解决能力。 

---
# Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words 

**Title (ZH)**: 重新思考稀疏自编码器的评估方法：通过多义词的表示进行探讨 

**Authors**: Gouki Minegishi, Hiroki Furuta, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2501.06254)  

**Abstract**: Sparse autoencoders (SAEs) have gained a lot of attention as a promising tool to improve the interpretability of large language models (LLMs) by mapping the complex superposition of polysemantic neurons into monosemantic features and composing a sparse dictionary of words. However, traditional performance metrics like Mean Squared Error and L0 sparsity ignore the evaluation of the semantic representational power of SAEs -- whether they can acquire interpretable monosemantic features while preserving the semantic relationship of words. For instance, it is not obvious whether a learned sparse feature could distinguish different meanings in one word. In this paper, we propose a suite of evaluations for SAEs to analyze the quality of monosemantic features by focusing on polysemous words. Our findings reveal that SAEs developed to improve the MSE-L0 Pareto frontier may confuse interpretability, which does not necessarily enhance the extraction of monosemantic features. The analysis of SAEs with polysemous words can also figure out the internal mechanism of LLMs; deeper layers and the Attention module contribute to distinguishing polysemy in a word. Our semantics focused evaluation offers new insights into the polysemy and the existing SAE objective and contributes to the development of more practical SAEs. 

**Abstract (ZH)**: 稀疏自动编码器（SAEs）因其能够通过将多义神经元的复杂叠加映射为单义特征，并构建一个稀疏的词典，而成为提高大型语言模型（LLMs）可解释性的有前途的工具。然而，传统的性能指标如均方误差（MSE）和L0稀疏性忽略了对SAEs语义表示能力的评估——即它们是否能在保持词语语义关系的同时获得可解释的单义特征。例如，学习到的稀疏特征是否能够区分一个词中不同的含义并不是显而易见的。在本文中，我们提出了一个针对SAEs的评估套件，以多义词为重点，分析单义特征的质量。我们的研究发现，旨在改进MSE-L0帕累托前沿的SAEs可能会导致可解释性的混乱，不一定能提高单义特征的提取。通过对多义词的SAEs进行分析，还可以揭示LLMs的内部机制；深层层和注意力模块有助于区分词语中的多义性。我们针对语义的评估提供了新的见解，有助于理解多义性并改进现有的SAE目标，从而促进更实用的SAE的发展。 

---
# A partition cover approach to tokenization 

**Title (ZH)**: 一种分区覆盖方法用于分词 

**Authors**: Jia Peng Lim, Davin Choo, Hady W. Lauw  

**Link**: [PDF](https://arxiv.org/pdf/2501.06246)  

**Abstract**: Tokenization is the process of encoding strings into tokens from a fixed vocabulary of size $k$ and is widely utilized in Natural Language Processing applications. The leading tokenization algorithm today is Byte Pair Encoding (BPE), which formulates the tokenization problem as a compression problem and tackles it by performing sequences of merges. In this work, we formulate tokenization as an optimization objective, show that it is NP-hard via a simple reduction from vertex cover, and propose a polynomial-time greedy algorithm GreedTok. Our formulation naturally relaxes to the well-studied weighted maximum coverage problem which has a simple $(1 - 1/e)$-approximation algorithm GreedWMC. Through empirical evaluations on real-world corpora, we show that GreedTok outperforms BPE, while achieving a comparable objective score as GreedWMC (which could have achieved a higher score due to relaxation). 

**Abstract (ZH)**: 分词是将字符串编码为大小为:k的固定词汇表中的标记的过程，广泛应用于自然语言处理应用中。当前领先的分词算法是字节对编码（BPE），它将分词问题形式化为压缩问题，并通过进行一系列合并来解决。在本文中，我们将分词建模为一个优化目标，通过一个简单的从顶点覆盖的归约证明它是NP-hard问题，并提出了一种多项式时间贪婪算法GreedTok。我们的建模自然地松弛为已研究广泛加权最大覆盖问题，该问题具有一个简单的(1 - 1/e)近似算法GreedWMC。通过在真实世界语料库上的实证评估，我们证明GreedTok在性能上优于BPE，同时在目标得分方面与GreedWMC相当（后者由于松弛可能达到了更高的得分）。 

---
# FLAME: Financial Large-Language Model Assessment and Metrics Evaluation 

**Title (ZH)**: FLAME：金融大型语言模型评估与指标评价 

**Authors**: Jiayu Guo, Yu Guo, Martha Li, Songtao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2501.06211)  

**Abstract**: LLMs have revolutionized NLP and demonstrated potential across diverse domains. More and more financial LLMs have been introduced for finance-specific tasks, yet comprehensively assessing their value is still challenging. In this paper, we introduce FLAME, a comprehensive financial LLMs evaluation system in Chinese, which includes two core evaluation benchmarks: FLAME-Cer and FLAME-Sce. FLAME-Cer covers 14 types of authoritative financial certifications, including CPA, CFA, and FRM, with a total of approximately 16,000 carefully selected questions. All questions have been manually reviewed to ensure accuracy and representativeness. FLAME-Sce consists of 10 primary core financial business scenarios, 21 secondary financial business scenarios, and a comprehensive evaluation set of nearly 100 tertiary financial application tasks. We evaluate 6 representative LLMs, including GPT-4o, GLM-4, ERNIE-4.0, Qwen2.5, XuanYuan3, and the latest Baichuan4-Finance, revealing Baichuan4-Finance excels other LLMs in most tasks. By establishing a comprehensive and professional evaluation system, FLAME facilitates the advancement of financial LLMs in Chinese contexts. Instructions for participating in the evaluation are available on GitHub: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经革新了自然语言处理（NLP），并在诸多领域展示了其潜力。越来越多的专用于金融任务的大规模语言模型被引入，但全面评估它们的价值仍然颇具挑战性。本文介绍了FLAME，这是一种全面的中文金融大规模语言模型评估系统，其中包括两个核心评估基准：FLAME-Cer和FLAME-Sce。FLAME-Cer涵盖了14种权威的金融认证类型，包括注册会计师（CPA）、特许金融分析师（CFA）和金融风险管理师（FRM），共计约16,000个精心挑选的问题。所有问题均经过人工审核，以确保准确性和代表性。FLAME-Sce包括10个主要核心金融业务场景、21个次要金融业务场景以及近100个三级金融应用任务的全面评价集。我们评估了6个代表性的LLM，包括GPT-4o、GLM-4、ERNIE-4.0、Qwen2.5、XuanYuan3以及最新的Baichuan4-Finance，结果显示Baichuan4-Finance在大多数任务中表现优异。通过建立一个全面且专业的评价系统，FLAME促进了中文背景下金融LLM的发展。有关评估指南可在GitHub上获取：[此链接]。

请注意，将原文中的链接替换成实际的GitHub链接地址。 

---
# Applications of natural language processing in aviation safety: A review and qualitative analysis 

**Title (ZH)**: 航空安全中自然语言处理的应用：综述与定性分析 

**Authors**: Aziida Nanyonga, Keith Joiner, Ugur Turhan, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2501.06210)  

**Abstract**: This study explores using Natural Language Processing in aviation safety, focusing on machine learning algorithms to enhance safety measures. There are currently May 2024, 34 Scopus results from the keyword search natural language processing and aviation safety. Analyzing these studies allows us to uncover trends in the methodologies, findings and implications of NLP in aviation. Both qualitative and quantitative tools have been used to investigate the current state of literature on NLP for aviation safety. The qualitative analysis summarises the research motivations, objectives, and outcomes, showing how NLP can be utilized to help identify critical safety issues and improve aviation safety. This study also identifies research gaps and suggests areas for future exploration, providing practical recommendations for the aviation industry. We discuss challenges in implementing NLP in aviation safety, such as the need for large, annotated datasets, and the difficulty in interpreting complex models. We propose solutions like active learning for data annotation and explainable AI for model interpretation. Case studies demonstrate the successful application of NLP in improving aviation safety, highlighting its potential to make aviation safer and more efficient. 

**Abstract (ZH)**: 本文探讨了自然语言处理（NLP）在航空安全中的应用，重点关注机器学习算法以加强安全措施。截至2024年5月，关键词搜索“自然语言处理和航空安全”在Scopus中已有34篇相关研究成果。通过分析这些研究，我们能够揭示自然语言处理在航空领域的趋势、方法、发现及其影响。我们使用定性和定量工具研究了自然语言处理在航空安全领域的现有文献状态。定性分析总结了研究动机、目标和结果，展示了自然语言处理如何有助于识别关键安全问题并提升航空安全。本文还指出了研究中的空白，并建议了未来研究的方向，为航空业提供了实际建议。我们讨论了在航空安全中实施自然语言处理所面临的挑战，如需要庞大且标注的数据集，以及难以解释复杂模型等问题。我们提出了诸如积极学习进行数据标注、可解释人工智能进行模型解释等解决方案。案例研究展示了自然语言处理在改进航空安全方面的成功应用，突显了其在使航空更安全和更高效方面的潜在价值。 

---
# Enhancing AI Safety Through the Fusion of Low Rank Adapters 

**Title (ZH)**: 通过低秩适配器融合提升人工智能安全性 

**Authors**: Satya Swaroop Gudipudi, Sreeram Vipparla, Harpreet Singh, Shashwat Goel, Ponnurangam Kumaraguru  

**Link**: [PDF](https://arxiv.org/pdf/2501.06208)  

**Abstract**: Instruction fine-tuning of large language models (LLMs) is a powerful method for improving task-specific performance, but it can inadvertently lead to a phenomenon where models generate harmful responses when faced with malicious prompts. In this paper, we explore Low-Rank Adapter Fusion (LoRA) as a means to mitigate these risks while preserving the model's ability to handle diverse instructions effectively. Through an extensive comparative analysis against established baselines using recognized benchmark datasets, we demonstrate a 42\% reduction in the harmfulness rate by leveraging LoRA fusion between a task adapter and a safety adapter, the latter of which is specifically trained on our safety dataset. However, we also observe exaggerated safety behaviour, where the model rejects safe prompts that closely resemble unsafe ones 

**Abstract (ZH)**: 大规模语言模型（LLMs）的指令微调是一种提高任务特定性能的强大方法，但可能会无意中导致模型在面对恶意提示时生成有害响应的现象。在本文中，我们探讨了低秩适配器融合（LoRA）作为一种手段，以减轻这些风险同时保持模型有效处理多样化指令的能力。通过使用公认的标准基准数据集与现有基线进行广泛的对比分析，我们展示了通过将任务适配器与专门在我们安全数据集上训练的安全适配器进行LoRA融合，有害响应的比例下降了42%。然而，我们还观察到过度的安全行为，即模型拒绝了与不安全提示相似的安全提示。 

---
# SST-EM: Advanced Metrics for Evaluating Semantic, Spatial and Temporal Aspects in Video Editing 

**Title (ZH)**: SST-EM：评估视频编辑中语义、空间和时间方面的新颖度量方法 

**Authors**: Varun Biyyala, Bharat Chanderprakash Kathuria, Jialu Li, Youshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07554)  

**Abstract**: Video editing models have advanced significantly, but evaluating their performance remains challenging. Traditional metrics, such as CLIP text and image scores, often fall short: text scores are limited by inadequate training data and hierarchical dependencies, while image scores fail to assess temporal consistency. We present SST-EM (Semantic, Spatial, and Temporal Evaluation Metric), a novel evaluation framework that leverages modern Vision-Language Models (VLMs), Object Detection, and Temporal Consistency checks. SST-EM comprises four components: (1) semantic extraction from frames using a VLM, (2) primary object tracking with Object Detection, (3) focused object refinement via an LLM agent, and (4) temporal consistency assessment using a Vision Transformer (ViT). These components are integrated into a unified metric with weights derived from human evaluations and regression analysis. The name SST-EM reflects its focus on Semantic, Spatial, and Temporal aspects of video evaluation. SST-EM provides a comprehensive evaluation of semantic fidelity and temporal smoothness in video editing. The source code is available in the \textbf{\href{this https URL}{GitHub Repository}}. 

**Abstract (ZH)**: 视频编辑模型的进步显著，但对其性能的评价仍然具有挑战性。传统的评估指标，如CLIP的文字和图像评分，常常不尽如人意：文字评分受限于训练数据的不足和层次依赖性，而图像评分无法评估时间连续性。我们提出了SST-EM（语义、空间和时间评估指标）这一新型评估框架，该框架结合了现代视觉语言模型（VLMs）、目标检测和时间一致性检查。SST-EM 包含四个组成部分：（1）使用VLM从帧中提取语义信息，（2）使用目标检测进行主要目标跟踪，（3）通过LLM代理进行聚焦目标细化，以及（4）使用视觉变换器（ViT）进行时间一致性评估。这些组件通过结合人类评价和回归分析得出的权重，集成到一个统一的评估指标中。命名SST-EM反映了其侧重于视频评估的语义、空间和时间方面。SST-EM 提供了对视频编辑中语义保真度和时间平滑度的综合评估。源代码已发布在\textbf{\href{https://github.com/example-repository}{GitHub Repository}}中。 

---
# Parallel Key-Value Cache Fusion for Position Invariant RAG 

**Title (ZH)**: 位置不变的RAG中的并行键值缓存融合 

**Authors**: Philhoon Oh, Jinwoo Shin, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2501.07523)  

**Abstract**: Recent advancements in Large Language Models (LLMs) underscore the necessity of Retrieval Augmented Generation (RAG) to leverage external information. However, LLMs are sensitive to the position of relevant information within contexts and tend to generate incorrect responses when such information is placed in the middle, known as `Lost in the Middle' phenomenon. In this paper, we introduce a framework that generates consistent outputs for decoder-only models, irrespective of the input context order. Experimental results for three open domain question answering tasks demonstrate position invariance, where the model is not sensitive to input context order, and superior robustness to irrelevent passages compared to prevailing approaches for RAG pipelines. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展突显了检索增强生成（RAG）方法在利用外部信息方面的重要性。然而，LLMs 对于上下文中相关信息的位置非常敏感，当相关信息位于中间位置时，模型容易生成错误的响应，这种现象被称为“中间迷失”现象。本文提出了一种框架，能够在任何输入上下文顺序下生成一致的输出。实验结果表明，该模型对输入上下文顺序不敏感，并且与现有的RAG管道方法相比，在处理无关段落时具有更好的鲁棒性。具体来说，我们在三个开放域的问答任务上进行了实验，验证了该模型在位置不变性方面的优越表现。 

---
# Joint Automatic Speech Recognition And Structure Learning For Better Speech Understanding 

**Title (ZH)**: 联合自动语音识别与结构学习以提高语音理解能力 

**Authors**: Jiliang Hu, Zuchao Li, Mengjia Shen, Haojun Ai, Sheng Li, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07329)  

**Abstract**: Spoken language understanding (SLU) is a structure prediction task in the field of speech. Recently, many works on SLU that treat it as a sequence-to-sequence task have achieved great success. However, This method is not suitable for simultaneous speech recognition and understanding. In this paper, we propose a joint speech recognition and structure learning framework (JSRSL), an end-to-end SLU model based on span, which can accurately transcribe speech and extract structured content simultaneously. We conduct experiments on name entity recognition and intent classification using the Chinese dataset AISHELL-NER and the English dataset SLURP. The results show that our proposed method not only outperforms the traditional sequence-to-sequence method in both transcription and extraction capabilities but also achieves state-of-the-art performance on the two datasets. 

**Abstract (ZH)**: 对话言语理解和处理（Spoken Language Understanding, SLU）是语音领域的一种结构预测任务。近年来，将SLU视为序列到序列任务的研究取得了巨大成功。然而，这种方法并不适用于同时进行语音识别和理解。本文提出了一种联合语音识别与结构学习框架（JSRSL），这是一种基于跨度的端到端SLU模型，能够同时准确转录语音并提取结构化内容。我们使用AISHELL-NER（中文数据集）和SLURP（英文数据集）进行了实体识别和意图分类实验。结果显示，本方法不仅在转录和提取能力上优于传统的序列到序列方法，还在两个数据集上实现了最先进的性能。 

---
# Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model 

**Title (ZH)**: Audio-CoT：探索大规模音频语言模型中的链式推理能力 

**Authors**: Ziyang Ma, Zhuo Chen, Yuping Wang, Eng Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07246)  

**Abstract**: Large Audio-Language Models (LALMs) have demonstrated remarkable performance in tasks involving audio perception and understanding, such as speech recognition and audio captioning. However, their reasoning capabilities - critical for solving complex real-world problems - remain underexplored. In this work, we conduct the first exploration into integrating Chain-of-Thought (CoT) reasoning into LALMs to enhance their reasoning ability across auditory modalities. We evaluate representative CoT methods, analyzing their performance in both information extraction and reasoning tasks across sound, music, and speech domains. Our findings reveal that CoT methods significantly improve performance on easy and medium tasks but encounter challenges with hard tasks, where reasoning chains can confuse the model rather than improve accuracy. Additionally, we identify a positive correlation between reasoning path length and accuracy, demonstrating the potential of scaling inference for advanced instruction-following and reasoning. This study not only highlights the promise of CoT in enhancing LALM reasoning capabilities but also identifies key limitations and provides actionable directions for future research. 

**Abstract (ZH)**: 大型音频语言模型（LALMs）在语音识别和音频描述等涉及音频感知和理解的任务中表现出了卓越的能力。然而，它们的推理能力——对于解决复杂的现实世界问题至关重要——仍然未被充分探索。在本文中，我们首次探索将推理链（Chain-of-Thought, CoT）机制整合到LALMs中，以增强其跨听觉模态的推理能力。我们评估了代表性CoT方法，并分析了它们在声音、音乐和语音领域中信息提取和推理任务中的表现。研究结果表明，CoT方法显著提高了简单和中等难度任务的表现，但在处理复杂任务时遇到了挑战，在这些任务中，推理链可能会使模型陷入混乱，而不是提高准确性。此外，我们发现推理路径长度和准确性之间存在正相关关系，这表明扩展推理对于高级指令遵循和推理具有潜在价值。这项研究不仅突显了CoT在增强LALM推理能力方面的潜力，还指出了其关键限制，并为未来的研究所提供了可操作的方向。 

---
# Can Vision-Language Models Evaluate Handwritten Math? 

**Title (ZH)**: 视觉-语言模型能评估手写数学问题吗？ 

**Authors**: Oikantik Nath, Hanani Bathina, Mohammed Safi Ur Rahman Khan, Mitesh M. Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2501.07244)  

**Abstract**: Recent advancements in Vision-Language Models (VLMs) have opened new possibilities in automatic grading of handwritten student responses, particularly in mathematics. However, a comprehensive study to test the ability of VLMs to evaluate and reason over handwritten content remains absent. To address this gap, we introduce FERMAT, a benchmark designed to assess the ability of VLMs to detect, localize and correct errors in handwritten mathematical content. FERMAT spans four key error dimensions - computational, conceptual, notational, and presentation - and comprises over 2,200 handwritten math solutions derived from 609 manually curated problems from grades 7-12 with intentionally introduced perturbations. Using FERMAT we benchmark nine VLMs across three tasks: error detection, localization, and correction. Our results reveal significant shortcomings in current VLMs in reasoning over handwritten text, with Gemini-1.5-Pro achieving the highest error correction rate (77%). We also observed that some models struggle with processing handwritten content, as their accuracy improves when handwritten inputs are replaced with printed text or images. These findings highlight the limitations of current VLMs and reveal new avenues for improvement. We release FERMAT and all the associated resources in the open-source to drive further research. 

**Abstract (ZH)**: 近年来，视觉-语言模型（VLMs）的发展为自动评估学生的手写作业提供了新的可能性，尤其是在数学领域。然而，全面测试VLMs评估和推理手写内容的能力的研究仍然缺失。为填补这一空白，我们提出了一种基准测试——FERMAT，旨在评估VLMs检测、定位和纠正手写数学内容中错误的能力。FERMAT涵盖了四个关键的错误维度——计算错误、概念错误、符号错误和表述错误——并包含了来自7-12年级609个经过人工筛选的数学问题的手写数学解答，其中故意引入了扰动，共有超过2,200个手写数学解答。我们利用FERMAT对九种VLMs在三项任务上进行基准测试：错误检测、定位和纠正。结果显示，当前的VLMs在处理手写文本时存在重大缺陷，其中Gemini-1.5-Pro取得了最高的错误修正率（77%）。我们还观察到，一些模型在处理手写内容时表现不佳，当使用打印文本或图像替代手写输入时，它们的准确性有所提高。本研究揭示了当前VLMs的局限性，并提出了新的改进方向。我们已将FERMAT及其相关资源开源，以促进进一步的研究。 

---
# BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature 

**Title (ZH)**: BIOMEDICA：一个开源的生物医学图像-描述档案、数据集及源自科学文献的视觉-语言模型 

**Authors**: Alejandro Lozano, Min Woo Sun, James Burgess, Liangyu Chen, Jeffrey J Nirschl, Jeffrey Gu, Ivan Lopez, Josiah Aklilu, Austin Wolfgang Katzer, Collin Chiu, Anita Rau, Xiaohan Wang, Yuhui Zhang, Alfred Seunghoon Song, Robert Tibshirani, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2501.07171)  

**Abstract**: The development of vision-language models (VLMs) is driven by large-scale and diverse multimodal datasets. However, progress toward generalist biomedical VLMs is limited by the lack of annotated, publicly accessible datasets across biology and medicine. Existing efforts are restricted to narrow domains, missing the full diversity of biomedical knowledge encoded in scientific literature. To address this gap, we introduce BIOMEDICA, a scalable, open-source framework to extract, annotate, and serialize the entirety of the PubMed Central Open Access subset into an easy-to-use, publicly accessible this http URL framework produces a comprehensive archive with over 24 million unique image-text pairs from over 6 million articles. Metadata and expert-guided annotations are also provided. We demonstrate the utility and accessibility of our resource by releasing BMCA-CLIP, a suite of CLIP-style models continuously pre-trained on the BIOMEDICA dataset via streaming, eliminating the need to download 27 TB of data this http URL average, our models achieve state-of-the-art performance across 40 tasks - spanning pathology, radiology, ophthalmology, dermatology, surgery, molecular biology, parasitology, and cell biology - excelling in zero-shot classification with a 6.56% average improvement (as high as 29.8% and 17.5% in dermatology and ophthalmology, respectively), and stronger image-text retrieval, all while using 10x less compute. To foster reproducibility and collaboration, we release our codebase and dataset for the broader research community. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的发展得益于大规模且多样的多模态数据集。然而，通用型 biomedical VLMs 的进展受限于生物医学领域缺乏跨学科的标注公开数据集。现有努力局限于狭窄的领域，未能涵盖科学研究文献中完整多样的生物医学知识。为填补这一空白，我们提出了 BIOMEDICA，这是一个可扩展的开源框架，用于提取、标注和序列化 PubMed Central 开放访问子集中的全部内容，从而生成一个易于使用的公开资源。该框架生成了一个全面的档案库，包含超过 600 万篇文章中的 2400 万张独特图像-文本对。还提供了元数据和专家引导的标注。为展示该资源的实用性和可访问性，我们发布了 BMCA-CLIP，这是一个在 BIOMEDICA 数据集上通过流传输持续预训练的一系列 CLIP 样式模型，从而避免下载 27 TB 的数据。总体而言，我们的模型在 40 个任务中（涵盖病理学、放射学、眼科学、皮肤科学、外科、分子生物学、寄生虫学和细胞生物学）实现了最先进的性能，实现了零样本分类 6.56% 的平均改进（皮肤科学最高可达 29.8%，眼科学为 17.5%），并在图像-文本检索方面表现出更强的能力，同时使用 10 倍更少的计算资源。为了促进可重复性和合作，我们向更广泛的科研界释放了我们的代码库和数据集。 

---
# Research on the Online Update Method for Retrieval-Augmented Generation (RAG) Model with Incremental Learning 

**Title (ZH)**: 增量学习下检索增强生成（RAG）模型的在线更新方法研究 

**Authors**: Yuxin Fan, Yuxiang Wang, Lipeng Liu, Xirui Tang, Na Sun, Zidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07063)  

**Abstract**: In the contemporary context of rapid advancements in information technology and the exponential growth of data volume, language models are confronted with significant challenges in effectively navigating the dynamic and ever-evolving information landscape to update and adapt to novel knowledge in real time. In this work, an online update method is proposed, which is based on the existing Retrieval Enhanced Generation (RAG) model with multiple innovation mechanisms. Firstly, the dynamic memory is used to capture the emerging data samples, and then gradually integrate them into the core model through a tunable knowledge distillation strategy. At the same time, hierarchical indexing and multi-layer gating mechanism are introduced into the retrieval module to ensure that the retrieved content is more targeted and accurate. Finally, a multi-stage network structure is established for different types of inputs in the generation stage, and cross-attention matching and screening are carried out on the intermediate representations of each stage to ensure the effective integration and iterative update of new and old knowledge. Experimental results show that the proposed method is better than the existing mainstream comparison models in terms of knowledge retention and inference accuracy. 

**Abstract (ZH)**: 在信息技术飞速发展和数据量呈指数增长的当代背景下，语言模型面临着在动态且不断演变的信息环境中高效更新和适应新知识的显著挑战，尤其是在实时更新方面。本文提出了一种在线更新方法，基于现有的检索增强生成（RAG）模型并结合了多种创新机制。首先，动态内存用于捕捉新兴数据样本，并通过可调的知识蒸馏策略逐步将其整合到核心模型中。同时，通过引入层次索引和多层门控机制来优化检索模块，以确保检索内容更加精准和目标化。最后，在生成阶段构建了多阶段网络结构，并对每个阶段的中间表示进行了交叉注意匹配和筛选，以确保新旧知识的有效整合和迭代更新。实验结果表明，所提出的方法在知识保留和推理准确性方面优于现有的主流对比模型。 

---
# Leveraging ASIC AI Chips for Homomorphic Encryption 

**Title (ZH)**: 利用专用集成电路（ASIC）人工智能芯片进行同态加密 

**Authors**: Jianming Tong, Tianhao Huang, Leo de Castro, Anirudh Itagi, Jingtian Dang, Anupam Golder, Asra Ali, Jevin Jiang, Arvind, G. Edward Suh, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2501.07047)  

**Abstract**: Cloud-based services are making the outsourcing of sensitive client data increasingly common. Although homomorphic encryption (HE) offers strong privacy guarantee, it requires substantially more resources than computing on plaintext, often leading to unacceptably large latencies in getting the results. HE accelerators have emerged to mitigate this latency issue, but with the high cost of ASICs. In this paper we show that HE primitives can be converted to AI operators and accelerated on existing ASIC AI accelerators, like TPUs, which are already widely deployed in the cloud. Adapting such accelerators for HE requires (1) supporting modular multiplication, (2) high-precision arithmetic in software, and (3) efficient mapping on matrix engines. We introduce the CROSS compiler (1) to adopt Barrett reduction to provide modular reduction support using multiplier and adder, (2) Basis Aligned Transformation (BAT) to convert high-precision multiplication as low-precision matrix-vector multiplication, (3) Matrix Aligned Transformation (MAT) to covert vectorized modular operation with reduction into matrix multiplication that can be efficiently processed on 2D spatial matrix engine. Our evaluation of CROSS on a Google TPUv4 demonstrates significant performance improvements, with up to 161x and 5x speedup compared to the previous work on many-core CPUs and V100. The kernel-level codes are open-sourced at this https URL. 

**Abstract (ZH)**: 基于云的服务使得对外包敏感客户数据越来越常见。尽管同态加密（HE）提供了强大的隐私保证，但在处理明文数据时却需要更多的资源，这往往导致结果获取的延迟不可接受。同态加密加速器已经出现以缓解这一延迟问题，但 ASIC 的成本很高。本文展示了可以将同态加密基本操作转换为 AI 操作，并利用现有的 ASIC AI 加速器（如 TPUs）进行加速，这些加速器已经在云计算中广泛部署。要为 HE 调整这些加速器，需要（1）支持模块化乘法，（2）在软件中实现高精度算术，以及（3）在矩阵引擎上进行高效的映射。我们引入了 CROSS 编译器：（1）采用 Barrett 减少来使用乘法器和加法器提供模块化降低支持；（2）基于基对齐变换（BAT）将高精度乘法转换为低精度矩阵-向量乘法；（3）向量对齐变换（MAT）将包含降低操作的向量化模块操作转换为可以在二维空间矩阵引擎上高效处理的矩阵乘法。我们在 Google TPUv4 上对 CROSS 进行的评估显示了显著的性能改进，与之前的很多核心 CPU 和 V100 上的工作相比，速度分别提高了 161 倍和 5 倍。内核级代码已开源，可通过此网址访问：[请在此处添加开源链接]。 

---
# LEO: Boosting Mixture of Vision Encoders for Multimodal Large Language Models 

**Title (ZH)**: LEO：增强视觉编码器混合模型以提升多模态大型语言模型 

**Authors**: Mozhgan Nasr Azadani, James Riddell, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2501.06986)  

**Abstract**: Enhanced visual understanding serves as a cornerstone for multimodal large language models (MLLMs). Recent hybrid MLLMs incorporate a mixture of vision experts to address the limitations of using a single vision encoder and excessively long visual tokens. Despite the progress of these MLLMs, a research gap remains in effectively integrating diverse vision encoders. This work explores fusion strategies of visual tokens for hybrid MLLMs, leading to the design of LEO, a novel MLLM with a dual-branch vision encoder framework that incorporates a post-adaptation fusion strategy and adaptive tiling: for each segmented tile of the input images, LEO sequentially interleaves the visual tokens from its two vision encoders. Extensive evaluation across 13 vision-language benchmarks reveals that LEO outperforms state-of-the-art open-source MLLMs and hybrid MLLMs on the majority of tasks. Furthermore, we show that LEO can be adapted to the specialized domain of autonomous driving without altering the model architecture or training recipe, achieving competitive performance compared to existing baselines. The code and model will be publicly available. 

**Abstract (ZH)**: 增强的视觉理解是多模态大规模语言模型（MLLMs）的基石。最近的混合MLLMs通过结合多种视觉专家来弥补单一视觉编码器和过长视觉标记的局限性。尽管这些MLLMs取得了进展，但在有效整合多样化的视觉编码器方面仍存在研究缺口。本研究探讨了混合MLLMs中视觉标记融合策略的设计，提出了一种新的MLLM——LEO，它采用了一种双分支视觉编码器框架，并结合了后适应融合策略和自适应切片：对于输入图像中的每个分割切片，LEO 依次交错其两个视觉编码器生成的视觉标记。在13个视觉-语言基准上的广泛评估表明，LEO 在大多数任务上优于最先进的开源MLLMs和混合MLLMs。此外，我们展示了LEO 可以适应自动驾驶这一特定领域，无需改变模型架构或训练方案，便能实现与现有基线相当的性能。代码和模型将公开可用。 

---
# Risk-Averse Finetuning of Large Language Models 

**Title (ZH)**: 大型语言模型的风险规避微调 

**Authors**: Sapana Chaudhary, Ujwal Dinesha, Dileep Kalathil, Srinivas Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2501.06911)  

**Abstract**: We consider the challenge of mitigating the generation of negative or toxic content by the Large Language Models (LLMs) in response to certain prompts. We propose integrating risk-averse principles into LLM fine-tuning to minimize the occurrence of harmful outputs, particularly rare but significant events. By optimizing the risk measure of Conditional Value at Risk (CVaR), our methodology trains LLMs to exhibit superior performance in avoiding toxic outputs while maintaining effectiveness in generative tasks. Empirical evaluations on sentiment modification and toxicity mitigation tasks demonstrate the efficacy of risk-averse reinforcement learning with human feedback (RLHF) in promoting a safer and more constructive online discourse environment. 

**Abstract (ZH)**: 我们探讨了通过大型语言模型（LLMs）对特定提示的响应，减少生成负面或有害内容的挑战。我们提出将规避风险的原则融入LLM的微调中，以最小化有害输出（尤其是罕见但重要的事件）的发生率。通过优化条件值风险（CVaR）的风险度量，我们的方法训练LLM在避免有害输出的同时，在生成任务中保持高效。在情感修饰和有害内容减轻任务上的实证评估表明，风险规避的强化学习（结合人类反馈的RLHF）在促进更安全和建设性的在线对话环境中具有有效性。 

---
# Causal Claims in Economics 

**Title (ZH)**: 经济中的因果命题 

**Authors**: Prashant Garg, Thiemo Fetzer  

**Link**: [PDF](https://arxiv.org/pdf/2501.06873)  

**Abstract**: We analyze over 44,000 NBER and CEPR working papers from 1980 to 2023 using a custom language model to construct knowledge graphs that map economic concepts and their relationships. We distinguish between general claims and those documented via causal inference methods (e.g., DiD, IV, RDD, RCTs). We document a substantial rise in the share of causal claims-from roughly 4% in 1990 to nearly 28% in 2020-reflecting the growing influence of the "credibility revolution." We find that causal narrative complexity (e.g., the depth of causal chains) strongly predicts both publication in top-5 journals and higher citation counts, whereas non-causal complexity tends to be uncorrelated or negatively associated with these outcomes. Novelty is also pivotal for top-5 publication, but only when grounded in credible causal methods: introducing genuinely new causal edges or paths markedly increases both the likelihood of acceptance at leading outlets and long-run citations, while non-causal novelty exhibits weak or even negative effects. Papers engaging with central, widely recognized concepts tend to attract more citations, highlighting a divergence between factors driving publication success and long-term academic impact. Finally, bridging underexplored concept pairs is rewarded primarily when grounded in causal methods, yet such gap filling exhibits no consistent link with future citations. Overall, our findings suggest that methodological rigor and causal innovation are key drivers of academic recognition, but sustained impact may require balancing novel contributions with conceptual integration into established economic discourse. 

**Abstract (ZH)**: 我们使用自定义语言模型分析了从1980年到2023年的超过44,000篇NBER和CEPR的工作论文，构建知识图谱以映射经济学概念及其关系。我们将声明区分为广泛声称（general claims）和通过因果推断方法（如双重差分法、工具变量法、分位数回归断点法、随机对照试验等）记录的声称。我们发现，从1990年的约4%上升到2020年的近28%，因果声称的比例显著增加，这反映了“可信度革命”日益增强的影响。我们的研究发现，因果推理的复杂性（例如，因果链的深度）强烈预测顶级期刊（前五名）的发表概率和更高的引用次数，而非因果复杂性与这些结果往往相关性不高或呈负相关。原创性对顶级期刊的发表也很关键，但这仅限于基于可靠的因果方法时：引入真正的新因果连接或路径大幅增加了被领先出版物接受和长期内的引用次数的可能性，而非因果原创性则显示出微弱或甚至负效应。涉及核心、广泛认可的概念的研究论文往往会吸引更多的引用次数，这突出了推动发表成功和长期学术影响力之间的差异。最后，基于因果方法填补未探索的概念对是获得奖励的主要方式，但这种填补未来引用次数的一致关系并不明显。总体而言，我们的研究结果表明，方法论严谨性和因果创新是学术认可的关键驱动力，但持续的影响可能需要在新颖贡献与整合到现有经济讨论的概念基础之间取得平衡。 

---
# Transfer Learning of Tabular Data by Finetuning Large Language Models 

**Title (ZH)**: 通过微调大规模语言模型进行表格数据的迁移学习 

**Authors**: Shourav B. Rabbani, Ibna Kowsar, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2501.06863)  

**Abstract**: Despite the artificial intelligence (AI) revolution, deep learning has yet to achieve much success with tabular data due to heterogeneous feature space and limited sample sizes without viable transfer learning. The new era of generative AI, powered by large language models (LLM), brings unprecedented learning opportunities to diverse data and domains. This paper investigates the effectiveness of an LLM application programming interface (API) and transfer learning of LLM in tabular data classification. LLM APIs respond to input text prompts with tokenized data and instructions, whereas transfer learning finetunes an LLM for a target classification task. This paper proposes an end-to-end finetuning of LLM to demonstrate cross-data transfer learning on ten benchmark data sets when large pre-trained tabular data models do not exist to facilitate transfer learning. The proposed LLM finetuning method outperforms state-of-the-art machine and deep learning methods on tabular data with less than ten features - a standard feature size for tabular data sets. The transfer learning approach uses a fraction of the computational cost of other deep learning or API-based solutions while ensuring competitive or superior classification performance. 

**Abstract (ZH)**: 尽管人工智能（AI）革命已经在多个领域取得了显著进展，但深层学习在处理表格数据方面尚未取得显著成功，原因在于特征空间的多样性以及样本量有限，缺乏有效的迁移学习方法。由大规模语言模型（LLM）驱动的生成型AI新纪元为各种数据和领域带来了前所未有的学习机会。本文探讨了大规模语言模型应用编程接口（API）及其在表格数据分类中的迁移学习效果。大规模语言模型API根据输入文本提示生成标记化数据和指令，而迁移学习则针对目标分类任务对大规模语言模型进行微调。本文提出了一种端到端的迁移学习方法，利用大规模预训练表格数据模型的缺失来在十个基准数据集上展示跨数据集的迁移学习效果。所提出的大规模语言模型微调方法在特征数少于十个（表格数据集的标准特征数量）的表格数据分类任务中优于最先进的机器学习和深度学习方法。迁移学习方法使用其他深度学习或基于API方法计算成本的一小部分，同时仍能确保具有竞争力或优越的分类性能。 

---
# A General Framework for Inference-time Scaling and Steering of Diffusion Models 

**Title (ZH)**: 适用于推理时间缩放和调控的扩散模型通用框架 

**Authors**: Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2501.06848)  

**Abstract**: Diffusion models produce impressive results in modalities ranging from images and video to protein design and text. However, generating samples with user-specified properties remains a challenge. Recent research proposes fine-tuning models to maximize rewards that capture desired properties, but these methods require expensive training and are prone to mode collapse. In this work, we propose Feynman Kac (FK) steering, an inference-time framework for steering diffusion models with reward functions. FK steering works by sampling a system of multiple interacting diffusion processes, called particles, and resampling particles at intermediate steps based on scores computed using functions called potentials. Potentials are defined using rewards for intermediate states and are selected such that a high value indicates that the particle will yield a high-reward sample. We explore various choices of potentials, intermediate rewards, and samplers. We evaluate FK steering on text-to-image and text diffusion models. For steering text-to-image models with a human preference reward, we find that FK steering a 0.8B parameter model outperforms a 2.6B parameter fine-tuned model on prompt fidelity, with faster sampling and no training. For steering text diffusion models with rewards for text quality and specific text attributes, we find that FK steering generates lower perplexity, more linguistically acceptable outputs and enables gradient-free control of attributes like toxicity. Our results demonstrate that inference-time scaling and steering of diffusion models, even with off-the-shelf rewards, can provide significant sample quality gains and controllability benefits. Code is available at this https URL . 

**Abstract (ZH)**: 扩散模型在图像、视频、蛋白质设计和文本等多种模态上取得了令人印象深刻的成果。然而，生成具有用户指定属性的样本仍然是一个挑战。近期的研究提出了通过最大化捕捉所需属性的奖励来微调模型的方法，但这些方法需要昂贵的训练成本，并且容易出现模式崩溃。在本文中，我们提出了一种费曼-卡克(Feynman Kac, FK)引导框架，这是一种在推理时引导扩散模型的方法，使用奖励函数进行引导。FK引导通过采样多个相互作用的扩散过程（称为粒子）并根据通过称为势函数计算的分数在中间步骤重新采样粒子来工作。势函数使用中间状态的奖励定义，并且选择使其具有高值的势函数使得粒子能够生成高奖励样本。我们探索了势函数、中间奖励和采样器的各种选择方式。我们在文本到图像和文本扩散模型上评估了FK引导。对于具有人类偏好的奖励引导文本到图像模型，我们发现，对于提示保真度，FK引导一个有0.8B参数的模型优于一个经过2.6B参数微调的模型，同时具有更快的采样速度且无需训练。对于使用文本质量奖励和特定文本属性奖励引导文本扩散模型，我们发现，FK引导能够生成较低的困惑度、更具语言接受度的输出，并且能够对毒性等属性进行无梯度控制。我们的结果表明，即使使用现成的奖励，在推理时对扩散模型进行扩增和引导，也可以显著提高样本质量并提供可控性优势。代码可在以下网址获取：这个 https URL。 

---
# SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training 

**Title (ZH)**: SPAM：带动量重置的意识尖峰Adam算法以实现稳定的大型语言模型训练 

**Authors**: Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06842)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks, yet their training remains highly resource-intensive and susceptible to critical challenges such as training instability. A predominant source of this instability stems from gradient and loss spikes, which disrupt the learning process, often leading to costly interventions like checkpoint recovery and experiment restarts, further amplifying inefficiencies. This paper presents a comprehensive investigation into gradient spikes observed during LLM training, revealing their prevalence across multiple architectures and datasets. Our analysis shows that these spikes can be up to $1000\times$ larger than typical gradients, substantially deteriorating model performance. To address this issue, we propose Spike-Aware Adam with Momentum Reset SPAM, a novel optimizer designed to counteract gradient spikes through momentum reset and spike-aware gradient clipping. Extensive experiments, including both pre-training and fine-tuning, demonstrate that SPAM consistently surpasses Adam and its variants across various tasks, including (1) LLM pre-training from 60M to 1B, (2) 4-bit LLM pre-training,(3) reinforcement learning, and (4) Time Series Forecasting. Additionally, SPAM facilitates memory-efficient training by enabling sparse momentum, where only a subset of momentum terms are maintained and updated. When operating under memory constraints, SPAM outperforms state-of-the-art memory-efficient optimizers such as GaLore and Adam-Mini. Our work underscores the importance of mitigating gradient spikes in LLM training and introduces an effective optimization strategy that enhances both training stability and resource efficiency at scale. Code is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中展现出了卓越的性能，但其训练仍然非常资源密集型，并且容易受到严重的挑战，如训练不稳定。这种不稳定性的一个主要来源是梯度和损失振荡，这些振荡会干扰学习过程，通常会导致昂贵的干预措施，例如训练快照恢复和实验重启，进而进一步加剧了效率低下。本文对LLM训练中观察到的梯度振荡进行了全面调查，揭示了这些振荡在多个架构和数据集中的普遍存在性。我们的分析表明，这些振荡的幅度可能比典型梯度大1000倍，严重恶化了模型性能。为了解决这一问题，我们提出了一种名为SPAM（Momentum Reset with Spike-Aware Gradient Clipping）的新型优化器，该优化器通过动量重置和梯度振荡感知的梯度裁剪，来对抗梯度振荡。通过广泛的实验，包括预训练和微调，我们展示了SPAM在多个任务中优于Adam及其变种，包括（1）从60M到1B的LLM预训练，（2）4位LLM预训练，（3）强化学习，以及（4）时间序列预测。此外，SPAM通过启用稀疏动量来实现内存高效训练，只需维护和更新动量项的一个子集。在内存受限的情况下，SPAM优于最先进的内存高效优化器GaLore和Adam-Mini。我们的工作突出了在LLM训练中减轻梯度振荡的重要性，并引入了一种有效的优化策略，该策略在大规模训练中不仅提高了训练稳定性，还提高了资源效率。代码可在以下链接获取：[这里](https://example.com/code) 

---
# LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents 

**Title (ZH)**: LLMs 模型化非 WEIRD 人群：合成文化代理的实验研究 

**Authors**: Augusto Gonzalez-Bonorino, Monica Capra, Emilio Pantoja  

**Link**: [PDF](https://arxiv.org/pdf/2501.06834)  

**Abstract**: Despite its importance, studying economic behavior across diverse, non-WEIRD (Western, Educated, Industrialized, Rich, and Democratic) populations presents significant challenges. We address this issue by introducing a novel methodology that uses Large Language Models (LLMs) to create synthetic cultural agents (SCAs) representing these populations. We subject these SCAs to classic behavioral experiments, including the dictator and ultimatum games. Our results demonstrate substantial cross-cultural variability in experimental behavior. Notably, for populations with available data, SCAs' behaviors qualitatively resemble those of real human subjects. For unstudied populations, our method can generate novel, testable hypotheses about economic behavior. By integrating AI into experimental economics, this approach offers an effective and ethical method to pilot experiments and refine protocols for hard-to-reach populations. Our study provides a new tool for cross-cultural economic studies and demonstrates how LLMs can help experimental behavioral research. 

**Abstract (ZH)**: 尽管其重要性不言而喻，研究跨多元、非WEIRD（西方、受过教育、工业化、富裕、民主）人口的经济行为存在显著挑战。为应对这一问题，我们引入了一种新的方法，该方法利用大型语言模型（LLMs）创建合成文化代理（SCAs），以代表这些人群。我们将这些SCAs置于经典的经济行为实验中，包括分配者游戏和 ultimatum 游戏。研究结果表明，这些实验中的行为在跨文化上存在显著差异。值得注意的是，对于已有数据的人群，SCAs 的行为在定性上与真实人类被试相似。对于未被研究的人群，我们的方法可以产生新的、可测试的关于经济行为的假设。通过将AI引入实验经济学，该方法提供了一种有效且伦理的方式，以试点实验并改进难以接触人群的研究方案。我们的研究提供了一种新的工具，用于跨文化经济研究，并展示了LLMs如何帮助实验行为研究。 

---
# Correcting Annotator Bias in Training Data: Population-Aligned Instance Replication (PAIR) 

**Title (ZH)**: 纠正训练数据中的标注员偏见：种群对齐实例复制（PAIR） 

**Authors**: Stephanie Eckman, Bolei Ma, Christoph Kern, Rob Chew, Barbara Plank, Frauke Kreuter  

**Link**: [PDF](https://arxiv.org/pdf/2501.06826)  

**Abstract**: Models trained on crowdsourced labels may not reflect broader population views when annotator pools are not representative. Since collecting representative labels is challenging, we propose Population-Aligned Instance Replication (PAIR), a method to address this bias through statistical adjustment. Using a simulation study of hate speech and offensive language detection, we create two types of annotators with different labeling tendencies and generate datasets with varying proportions of the types. Models trained on unbalanced annotator pools show poor calibration compared to those trained on representative data. However, PAIR, which duplicates labels from underrepresented annotator groups to match population proportions, significantly reduces bias without requiring new data collection. These results suggest statistical techniques from survey research can help align model training with target populations even when representative annotator pools are unavailable. We conclude with three practical recommendations for improving training data quality. 

**Abstract (ZH)**: 当注释员群体不具备代表性时，基于众包标签训练的模型可能无法反映更广泛人群的观点。由于收集代表性标签具有挑战性，我们提出了一种名为Population-Aligned Instance Replication (PAIR)的方法，通过统计调整来解决这一偏见问题。利用仇恨言论和冒犯性语言检测的模拟研究，我们创建了具有不同标签倾向的两种类型注释员，并生成了不同类型比例变化的数据集。针对不平衡注释员群体训练的模型与使用代表性数据训练的模型相比，表现出较差的校准效果。然而，PAIR方法通过复制被欠代表注释员群体的标签以匹配人口比例，显著减少了偏见，而无需进行新的数据收集。这些结果表明，调查研究中的统计技术即使在没有代表性注释员群体时也能帮助模型训练与目标人群相一致。最后，我们提出了三条实用建议，以提高训练数据的质量。 

---
# Improving Cross-Lingual Phonetic Representation of Low-Resource Languages Through Language Similarity Analysis 

**Title (ZH)**: 通过语言相似性分析提高低资源语言跨语言音素表示的研究 

**Authors**: Minu Kim, Kangwook Jang, Hoirin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.06810)  

**Abstract**: This paper examines how linguistic similarity affects cross-lingual phonetic representation in speech processing for low-resource languages, emphasizing effective source language selection. Previous cross-lingual research has used various source languages to enhance performance for the target low-resource language without thorough consideration of selection. Our study stands out by providing an in-depth analysis of language selection, supported by a practical approach to assess phonetic proximity among multiple language families. We investigate how within-family similarity impacts performance in multilingual training, which aids in understanding language dynamics. We also evaluate the effect of using phonologically similar languages, regardless of family. For the phoneme recognition task, utilizing phonologically similar languages consistently achieves a relative improvement of 55.6% over monolingual training, even surpassing the performance of a large-scale self-supervised learning model. Multilingual training within the same language family demonstrates that higher phonological similarity enhances performance, while lower similarity results in degraded performance compared to monolingual training. 

**Abstract (ZH)**: 本文探讨了语言相似性如何影响低资源语言在语音处理中的跨语言音素表示，并强调了有效的源语言选择。以往的跨语言研究在提高目标低资源语言的效果时使用了多种源语言，但没有充分考虑源语言的选择。我们的研究通过深入分析语言选择，并结合实用方法评估多种语言家族之间的音素相近性，脱颖而出。我们研究了家族内相似性如何影响多语言训练中的性能，有助于理解语言动态。我们还评估了使用音位相似语言（不考虑家族归属）的效果。在音素识别任务中，使用音位相似的语言的一致性改进比单一语言训练提高了55.6%的相对性能，甚至超过了大规模自监督学习模型的性能。同一语言家族内的多语言训练表明，更高的音位相似性可以提高性能，而较低的相似性则导致性能低于单一语言训练。 

---
# 3DCoMPaT200: Language-Grounded Compositional Understanding of Parts and Materials of 3D Shapes 

**Title (ZH)**: 3DCoMPaT200：基于语言的3D形状部件与材料组合理解 

**Authors**: Mahmoud Ahmed, Xiang Li, Arpit Prajapati, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2501.06785)  

**Abstract**: Understanding objects in 3D at the part level is essential for humans and robots to navigate and interact with the environment. Current datasets for part-level 3D object understanding encompass a limited range of categories. For instance, the ShapeNet-Part and PartNet datasets only include 16, and 24 object categories respectively. The 3DCoMPaT dataset, specifically designed for compositional understanding of parts and materials, contains only 42 object categories. To foster richer and fine-grained part-level 3D understanding, we introduce 3DCoMPaT200, a large-scale dataset tailored for compositional understanding of object parts and materials, with 200 object categories with $\approx$5 times larger object vocabulary compared to 3DCoMPaT and $\approx$ 4 times larger part categories. Concretely, 3DCoMPaT200 significantly expands upon 3DCoMPaT, featuring 1,031 fine-grained part categories and 293 distinct material classes for compositional application to 3D object parts. Additionally, to address the complexities of compositional 3D modeling, we propose a novel task of Compositional Part Shape Retrieval using ULIP to provide a strong 3D foundational model for 3D Compositional Understanding. This method evaluates the model shape retrieval performance given one, three, or six parts described in text format. These results show that the model's performance improves with an increasing number of style compositions, highlighting the critical role of the compositional dataset. Such results underscore the dataset's effectiveness in enhancing models' capability to understand complex 3D shapes from a compositional perspective. Code and Data can be found at this http URL 

**Abstract (ZH)**: 理解物体在部件级的三维特性对于人类和机器人在环境中导航和交互至关重要。目前用于部件级三维物体理解的数据集涵盖的类别范围有限。例如，ShapeNet-Part和PartNet数据集分别只包含16个和24个物体类别。而专门用于部件和材料组成理解的3DCoMPaT数据集也只包含42个物体类别。为了促进更丰富、更细粒度的部件级三维理解，我们引入了3DCoMPaT200数据集，这是一个专为物体部件和材料组成的理解设计的大规模数据集，包含200个物体类别，词汇量约为3DCoMPaT的5倍，部件类别约为4倍。具体而言，3DCoMPaT200显著扩展了3DCoMPaT，新增了1,031个细粒度部件类别和293个独特的材料类别，以便于对三维物体部件进行组成应用。此外，为了解决组成模型的复杂性，我们提出了使用ULIP进行组成部件形状检索的新任务，以提供强健的三维基础模型，用于三维组成理解和应用。该方法评估了模型在根据文本描述一个、三个或六个部件时的形状检索性能。结果显示，模型的性能随着风格组成数量的增加而提高，突显了组成数据集的关键作用。这些结果强调了该数据集在增强模型从组成视角理解复杂三维形状方面的效果。代码和数据可在此网址获取：[请填写网址] 

---
# ZOQO: Zero-Order Quantized Optimization 

**Title (ZH)**: ZOQO: 零阶量化优化 

**Authors**: Noga Bar, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2501.06736)  

**Abstract**: The increasing computational and memory demands in deep learning present significant challenges, especially in resource-constrained environments. We introduce a zero-order quantized optimization (ZOQO) method designed for training models with quantized parameters and operations. Our approach leverages zero-order approximations of the gradient sign and adapts the learning process to maintain the parameters' quantization without the need for full-precision gradient calculations. We demonstrate the effectiveness of ZOQO through experiments in fine-tuning of large language models and black-box adversarial attacks. Despite the limitations of zero-order and quantized operations training, our method achieves competitive performance compared to full-precision methods, highlighting its potential for low-resource environments. 

**Abstract (ZH)**: 深度学习中不断增加的计算和内存需求在资源受限环境中提出了重大挑战。我们引入了一种零阶量化优化（ZOQO）方法，该方法适用于使用量化参数和操作训练模型。我们的方法利用了梯度符号的零阶近似，并通过适应学习过程来保持参数的量化，而无需进行全精度梯度计算。我们通过在大型语言模型微调和黑盒对抗性攻击方面的实验展示了ZOQO的有效性。尽管零阶和量化操作训练存在限制，但我们的方法在性能上与全精度方法相当，突显了其在资源受限环境中的潜力。 

---
# Fine-tuning ChatGPT for Automatic Scoring of Written Scientific Explanations in Chinese 

**Title (ZH)**: 将ChatGPT微调以自动评分中文书面科学解释 

**Authors**: Jie Yang, Ehsan Latif, Yuze He, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.06704)  

**Abstract**: The development of explanations for scientific phenomena is essential in science assessment, but scoring student-written explanations remains challenging and resource-intensive. Large language models (LLMs) have shown promise in addressing this issue, particularly in alphabetic languages like English. However, their applicability to logographic languages is less explored. This study investigates the potential of fine-tuning ChatGPT, a leading LLM, to automatically score scientific explanations written in Chinese. Student responses to seven scientific explanation tasks were collected and automatically scored, with scoring accuracy examined in relation to reasoning complexity using the Kendall correlation. A qualitative analysis explored how linguistic features influenced scoring accuracy. The results show that domain-specific adaptation enables ChatGPT to score Chinese scientific explanations with accuracy. However, scoring accuracy correlates with reasoning complexity: a negative correlation for lower-level responses and a positive one for higher-level responses. The model overrates complex reasoning in low-level responses with intricate sentence structures and underrates high-level responses using concise causal reasoning. These correlations stem from linguistic features--simplicity and clarity enhance accuracy for lower-level responses, while comprehensiveness improves accuracy for higher-level ones. Simpler, shorter responses tend to score more accurately at lower levels, whereas longer, information-rich responses yield better accuracy at higher levels. These findings demonstrate the effectiveness of LLMs in automatic scoring within a Chinese context and emphasize the importance of linguistic features and reasoning complexity in fine-tuning scoring models for educational assessments. 

**Abstract (ZH)**: 科学现象解释的发展对于科学评估至关重要，但对学生撰写的解释进行评分仍然是一项具有挑战性和资源密集性的任务。大型语言模型（LLMs）在此方面展现出了潜力，尤其是在像英语这样的字母语言方面。然而，它们在象形文字语言如中文方面的应用尚缺乏探讨。本研究旨在探讨微调领先的大规模语言模型ChatGPT自动评分中文科学解释的潜力。收集了学生对七项科学解释任务的响应，并自动进行了评分，评分准确度与推理复杂性之间的关系通过Kendall相关性进行了考察。定性分析进一步探讨了语言特征如何影响评分准确度。结果显示，领域特定的适应性使ChatGPT能够准确评分中文科学解释。然而，评分准确度与推理复杂性之间存在相关性：低层级响应呈现负相关，而高层级响应则呈现正相关。该模型在低层级复杂推理的复杂句结构中高估了推理，在高层级使用简洁因果推理的响应中低估了评分准确度。这些相关性源于语言特征——简洁性和清晰性增强了低层级响应的准确度，而综合性和全面性则改进了高层级响应的准确度。简短的回答通常在低层级更准确，而较长的信息丰富回答在高层级更准确。这些发现表明，大规模语言模型在中文背景下自动评分的有效性，并强调了语言特征和推理复杂性在微调评分模型方面的重要性，以提高教育评估的准确性。 

---
# Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization 

**Title (ZH)**: 基于张量压缩优化的超高效FPGA上Transformer训练 

**Authors**: Jiayi Tian, Jinming Lu, Hai Li, Xiangwei Wang, Cong, Ian Young, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06663)  

**Abstract**: Transformer models have achieved state-of-the-art performance across a wide range of machine learning tasks. There is growing interest in training transformers on resource-constrained edge devices due to considerations such as privacy, domain adaptation, and on-device scientific machine learning. However, the significant computational and memory demands required for transformer training often exceed the capabilities of an edge device. Leveraging low-rank tensor compression, this paper presents the first on-FPGA accelerator for end-to-end transformer training. On the algorithm side, we present a bi-directional contraction flow for tensorized transformer training, significantly reducing the computational FLOPS and intra-layer memory costs compared to existing tensor operations. On the hardware side, we store all highly compressed model parameters and gradient information on chip, creating an on-chip-memory-only framework for each stage in training. This reduces off-chip communication and minimizes latency and energy costs. Additionally, we implement custom computing kernels for each training stage and employ intra-layer parallelism and pipe-lining to further enhance run-time and memory efficiency. Through experiments on transformer models within $36.7$ to $93.5$ MB using FP-32 data formats on the ATIS dataset, our tensorized FPGA accelerator could conduct single-batch end-to-end training on the AMD Alevo U50 FPGA, with a memory budget of less than $6$-MB BRAM and $22.5$-MB URAM. Compared to uncompressed training on the NVIDIA RTX 3090 GPU, our on-FPGA training achieves a memory reduction of $30\times$ to $51\times$. Our FPGA accelerator also achieves up to $3.6\times$ less energy cost per epoch compared with tensor Transformer training on an NVIDIA RTX 3090 GPU. 

**Abstract (ZH)**: transformer 模型已在广泛的数据机器学习任务中实现了最先进性能。由于考虑了隐私、领域适应和本地设备科学机器学习等因素，越来越多的研究兴趣集中在训练 transformer 模型上，希望在资源受限的边缘设备上进行训练。然而，transformer 训练所需的显著计算和内存需求往往超过了边缘设备的能力。利用低秩张量压缩，本文提出了首个适用于全 FPGA 加速器的端到端 transformer 训练方案。从算法角度而言，我们提出了一种双向收缩流，用于张量化的 transformer 训练，与现有张量操作相比，显著减少了计算 FLOPS 和层内内存成本。从硬件角度而言，我们所有高度压缩的模型参数和梯度信息都被存储在片上，从而为每个训练阶段创建了一个仅内存片上框架。这减少了片外通信，并最小化了延迟和能耗。此外，我们为每个训练阶段实现了自定义计算内核，并采用了层内并行性和流水线技术，进一步增强了运行时和内存效率。在使用 FP-32 数据格式的 ATIS 数据集上，对于模型大小在 36.7 至 93.5 MB 的 transformer 模型，我们的张量 FPGA 加速器可以在 AMD Alevo U50 FPGA 上执行端到端的单批次训练，内存预算少于 6 MB 的 BRAM 和 22.5 MB 的 URAM。与在 NVIDIA RTX 3090 GPU 上进行的非压缩训练相比，我们的 FPGA 训练实现了 30 至 51 倍的内存减少。我们的 FPGA 加速器每轮次的能耗成本也达到了 NVIDIA RTX 3090 GPU 上张量 transformer 训练的 3.6 倍。 

---
# The Magnitude of Categories of Texts Enriched by Language Models 

**Title (ZH)**: 语言模型扩充的文本类别规模研究 

**Authors**: Tai-Danae Bradley, Juan Pablo Vigneaux  

**Link**: [PDF](https://arxiv.org/pdf/2501.06662)  

**Abstract**: The purpose of this article is twofold. Firstly, we use the next-token probabilities given by a language model to explicitly define a $[0,1]$-enrichment of a category of texts in natural language, in the sense of Bradley, Terilla, and Vlassopoulos. We consider explicitly the terminating conditions for text generation and determine when the enrichment itself can be interpreted as a probability over texts. Secondly, we compute the Möbius function and the magnitude of an associated generalized metric space $\mathcal{M}$ of texts using a combinatorial version of these quantities recently introduced by Vigneaux. The magnitude function $f(t)$ of $\mathcal{M}$ is a sum over texts $x$ (prompts) of the Tsallis $t$-entropies of the next-token probability distributions $p(-|x)$ plus the cardinality of the model's possible outputs. The derivative of $f$ at $t=1$ recovers a sum of Shannon entropies, which justifies seeing magnitude as a partition function. Following Leinster and Schulman, we also express the magnitude function of $\mathcal M$ as an Euler characteristic of magnitude homology and provide an explicit description of the zeroeth and first magnitude homology groups. 

**Abstract (ZH)**: 本文的目的是双重的。首先，我们利用语言模型给出的下一个词的概率，明确定义一个自然语言文本类别的$[0,1]$-增强，这一定义依据Bradley, Terilla和Vlassopoulos的工作。我们明确考虑了文本生成的终止条件，并确定当增强本身可以被解释为文本的概率时的情形。其次，我们计算了与Vigneaux最近引入的组合版本相关联的格拉斯曼空间$\mathcal{M}$上的莫比乌斯函数和尺度。$\mathcal{M}$的尺度函数$f(t)$是文本$x$（提示）的Tsallis $t$-熵之和加上模型可能输出的基数。$f(t)$在$t=1$处的导数可以恢复出香农熵之和，从而证明了尺度作为分区函数的解释。借鉴Leinster和Schulman的方法，我们还将$\mathcal{M}$的尺度函数表示为其尺度同调的欧拉特征，并给出了零级和一级尺度同调群的显式描述。 

---
# EmoXpt: Analyzing Emotional Variances in Human Comments and LLM-Generated Responses 

**Title (ZH)**: EmoXpt：分析人类评论与大语言模型生成回应中的情感波动 

**Authors**: Shireesh Reddy Pyreddy, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2501.06597)  

**Abstract**: The widespread adoption of generative AI has generated diverse opinions, with individuals expressing both support and criticism of its applications. This study investigates the emotional dynamics surrounding generative AI by analyzing human tweets referencing terms such as ChatGPT, OpenAI, Copilot, and LLMs. To further understand the emotional intelligence of ChatGPT, we examine its responses to selected tweets, highlighting differences in sentiment between human comments and LLM-generated responses. We introduce EmoXpt, a sentiment analysis framework designed to assess both human perspectives on generative AI and the sentiment embedded in ChatGPT's responses. Unlike prior studies that focus exclusively on human sentiment, EmoXpt uniquely evaluates the emotional expression of ChatGPT. Experimental results demonstrate that LLM-generated responses are notably more efficient, cohesive, and consistently positive than human responses. 

**Abstract (ZH)**: 生成式AI的广泛应用引起了社会各界的不同看法，个人既支持也有批评其应用。本文通过分析提及ChatGPT、OpenAI、Copilot和大规模语言模型（LLMs）的人类推文，研究生成式AI周围的情感动态。为进一步了解ChatGPT的情感智能，我们对其对选定推文的回应进行了研究，突显了人类评论与生成式AI回应之间情感差异。本文引入了EmoXpt情感分析框架，旨在评估人类对生成式AI的观点以及ChatGPT回应中嵌入的情感。不同于以往主要集中在人类情感的研究，EmoXpt独特地评估了ChatGPT的情感表达。实验结果表明，生成式AI生成的回应在效率、连贯性和一致性方面明显优于人类回应。 

---
# Ladder-residual: parallelism-aware architecture for accelerating large model inference with communication overlapping 

**Title (ZH)**: 梯级残差结构：一种通过通信重叠加速大型模型推理的并行化架构 

**Authors**: Muru Zhang, Mayank Mishra, Zhongzhu Zhou, William Brandon, Jue Wang, Yoon Kim, Jonathan Ragan-Kelley, Shuaiwen Leon Song, Ben Athiwaratkun, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2501.06589)  

**Abstract**: Large language model inference is both memory-intensive and time-consuming, often requiring distributed algorithms to efficiently scale. Various model parallelism strategies are used in multi-gpu training and inference to partition computation across multiple devices, reducing memory load and computation time. However, using model parallelism necessitates communication of information between GPUs, which has been a major bottleneck and limits the gains obtained by scaling up the number of devices. We introduce Ladder Residual, a simple architectural modification applicable to all residual-based models that enables straightforward overlapping that effectively hides the latency of communication. Our insight is that in addition to systems optimization, one can also redesign the model architecture to decouple communication from computation. While Ladder Residual can allow communication-computation decoupling in conventional parallelism patterns, we focus on Tensor Parallelism in this paper, which is particularly bottlenecked by its heavy communication. For a Transformer model with 70B parameters, applying Ladder Residual to all its layers can achieve 30% end-to-end wall clock speed up at inference time with TP sharding over 8 devices. We refer the resulting Transformer model as the Ladder Transformer. We train a 1B and 3B Ladder Transformer from scratch and observe comparable performance to a standard dense transformer baseline. We also show that it is possible to convert parts of the Llama-3.1 8B model to our Ladder Residual architecture with minimal accuracy degradation by only retraining for 3B tokens. 

**Abstract (ZH)**: 大型语言模型的推理既占用大量内存又耗时，通常需要使用分布式算法以高效扩展。在多GPU的训练和推理过程中，各种模型并行策略被用来将计算任务分配到多个设备上，从而减轻内存负载并缩短计算时间。然而，使用模型并行策略需要在GPU之间传输信息，这一过程已成为主要瓶颈，并限制了扩展设备数量所能取得的收益。我们提出了一种名为梯形残差（Ladder Residual）的简单架构修改方法，适用于所有基于残差的模型，能够简单地重叠计算和通信，从而有效地隐藏通信延迟。我们的研究表明，除了系统优化外，还可以通过重新设计模型架构来将通信与计算分离。虽然Ladder Residual可以在传统并行模式下实现通信-计算的解耦，但我们在这篇文章中主要关注张量并行（Tensor Parallelism），它由于其沉重的通信负担而成为瓶颈。对于一个具有700亿参数的Transformer模型，将Ladder Residual应用于所有层，在8个设备的张量切片下，可以实现推理时间30%的端到端速度提升。我们将这种改进后的Transformer模型称为梯形Transformer（Ladder Transformer）。我们从头开始训练了一个1亿参数和3亿参数的Ladder Transformer，并观察到其性能与传统的密集Transformer基线相当。我们还展示了可以将部分Llama-3.1 8亿参数模型转换为我们的Ladder Residual架构，仅重新训练3亿个tokens的情况下，精度退化程度可以忽略不计。 

---
# Speech Recognition for Automatically Assessing Afrikaans and isiXhosa Preschool Oral Narratives 

**Title (ZH)**: 自动评估 african 和祖鲁语学前儿童口头叙事的语音识别技术 

**Authors**: Christiaan Jacobs, Annelien Smith, Daleen Klop, Ondřej Klejch, Febe de Wet, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2501.06478)  

**Abstract**: We develop automatic speech recognition (ASR) systems for stories told by Afrikaans and isiXhosa preschool children. Oral narratives provide a way to assess children's language development before they learn to read. We consider a range of prior child-speech ASR strategies to determine which is best suited to this unique setting. Using Whisper and only 5 minutes of transcribed in-domain child speech, we find that additional in-domain adult data (adult speech matching the story domain) provides the biggest improvement, especially when coupled with voice conversion. Semi-supervised learning also helps for both languages, while parameter-efficient fine-tuning helps on Afrikaans but not on isiXhosa (which is under-represented in the Whisper model). Few child-speech studies look at non-English data, and even fewer at the preschool ages of 4 and 5. Our work therefore represents a unique validation of a wide range of previous child-speech ASR strategies in an under-explored setting. 

**Abstract (ZH)**: 我们开发了自动语音识别（ASR）系统，用于识别南非文（Afrikaans）和祖鲁语（isiXhosa）幼儿园儿童讲述的故事。口头叙述为在儿童学会阅读之前评估其语言发展提供了一种方式。我们考虑了一系列先前针对儿童语音的ASR策略，以确定哪种策略最适合这种独特的环境。使用Whisper并仅利用5分钟的转录儿童领域内语音数据，我们发现，额外的领域内成人语音数据（匹配故事领域）提供了最大的改进，特别是在结合了语音转换的情况下。半监督学习在两种语言中均有所帮助，而在南非文（Afrikaans）中参数高效微调有效，但在祖鲁语（isiXhosa）中无效（因Whisper模型中的祖鲁语数据不足）。很少有研究针对非英语数据进行儿童语音研究，更少有研究关注4岁和5岁的幼儿阶段。因此，我们的工作为在未充分探索的环境中验证了多种先前的儿童语音ASR策略提供了独特的方法。 

---
# Using Pre-trained LLMs for Multivariate Time Series Forecasting 

**Title (ZH)**: 使用预训练的大语言模型进行多变量时间序列预测 

**Authors**: Malcolm L. Wolff, Shenghao Yang, Kari Torkkola, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2501.06386)  

**Abstract**: Pre-trained Large Language Models (LLMs) encapsulate large amounts of knowledge and take enormous amounts of compute to train. We make use of this resource, together with the observation that LLMs are able to transfer knowledge and performance from one domain or even modality to another seemingly-unrelated area, to help with multivariate demand time series forecasting. Attention in transformer-based methods requires something worth attending to -- more than just samples of a time-series. We explore different methods to map multivariate input time series into the LLM token embedding space. In particular, our novel multivariate patching strategy to embed time series features into decoder-only pre-trained Transformers produces results competitive with state-of-the-art time series forecasting models. We also use recently-developed weight-based diagnostics to validate our findings. 

**Abstract (ZH)**: 预训练大规模语言模型（LLMs）蕴含了大量知识，并需要巨大的计算资源来训练。我们利用这一资源，并结合LLMs具备在不同领域或模态之间转移知识和性能的能力，来帮助进行多变量需求时间序列预测。基于变换器的方法中的注意力机制需要关注有意义的内容——而不仅仅是时间序列样本。我们探索了不同的方法，将多变量输入时间序列映射到LLM标记嵌入空间。特别是，我们提出的新颖的多变量片断化策略，将时间序列特征嵌入仅解码器预训练变换器中，产生的结果与最新时间序列预测模型相媲美。我们还使用了最近开发的基于权重的诊断方法来验证我们的发现。 

---
# TTS-Transducer: End-to-End Speech Synthesis with Neural Transducer 

**Title (ZH)**: TTS-Transducer：基于神经转换器的端到端语音合成 

**Authors**: Vladimir Bataev, Subhankar Ghosh, Vitaly Lavrukhin, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.06320)  

**Abstract**: This work introduces TTS-Transducer - a novel architecture for text-to-speech, leveraging the strengths of audio codec models and neural transducers. Transducers, renowned for their superior quality and robustness in speech recognition, are employed to learn monotonic alignments and allow for avoiding using explicit duration predictors. Neural audio codecs efficiently compress audio into discrete codes, revealing the possibility of applying text modeling approaches to speech generation. However, the complexity of predicting multiple tokens per frame from several codebooks, as necessitated by audio codec models with residual quantizers, poses a significant challenge. The proposed system first uses a transducer architecture to learn monotonic alignments between tokenized text and speech codec tokens for the first codebook. Next, a non-autoregressive Transformer predicts the remaining codes using the alignment extracted from transducer loss. The proposed system is trained end-to-end. We show that TTS-Transducer is a competitive and robust alternative to contemporary TTS systems. 

**Abstract (ZH)**: 本文介绍了TTS-Transducer —— 一种结合了音频编解码模型和神经转换器优势的新型文本到语音架构。转换器以其在语音识别方面卓越的质量和鲁棒性而闻名，并被用于学习单调对齐，从而避免了使用显式的时长预测器。神经音频编解码器能够高效地将音频压缩为离散代码，这为利用文本建模方法生成语音提供了可能性。然而，音频编解码器模型中残差量化器的需求使得每帧预测多个令牌变得复杂，这构成了一大挑战。所提出系统首先采用转换器架构来学习令牌化文本与第一码本中音频编解码器令牌之间的单调对齐。接着，一个非自回归Transformer使用转换器损失提取的对齐预测剩余的代码。所提出系统是端到端训练的。我们展示了TTS-Transducer作为一种竞争力强且鲁棒的替代方案，适用于当前的TTS系统。 

---
# Understanding How Paper Writers Use AI-Generated Captions in Figure Caption Writing 

**Title (ZH)**: 理解论文作者在图题撰写中使用AI生成的caption方式 

**Authors**: Ho Yin, Ting-Yao Hsu, Jiyoo Min, Sungchul Kim, Ryan A. Rossi, Tong Yu, Hyunggu Jung, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06317)  

**Abstract**: Figures and their captions play a key role in scientific publications. However, despite their importance, many captions in published papers are poorly crafted, largely due to a lack of attention by paper authors. While prior AI research has explored caption generation, it has mainly focused on reader-centered use cases, where users evaluate generated captions rather than actively integrating them into their writing. This paper addresses this gap by investigating how paper authors incorporate AI-generated captions into their writing process through a user study involving 18 participants. Each participant rewrote captions for two figures from their own recently published work, using captions generated by state-of-the-art AI models as a resource. By analyzing video recordings of the writing process through interaction analysis, we observed that participants often began by copying and refining AI-generated captions. Paper writers favored longer, detail-rich captions that integrated textual and visual elements but found current AI models less effective for complex figures. These findings highlight the nuanced and diverse nature of figure caption composition, revealing design opportunities for AI systems to better support the challenges of academic writing. 

**Abstract (ZH)**: 图表及其说明在科学出版物中起着关键作用。然而，尽管它们非常重要，许多已发表论文的图表说明质量不佳，很大程度上是由于作者在撰写时对图表说明不够关注所致。尽管之前的人工智能研究探索了说明生成，但这些研究主要集中在读者为中心的应用场景上，用户评估生成的说明而非主动将其整合到写作中。本研究通过一项涉及18名参与者的用户研究填补了这一空白，探讨作者如何将人工智能生成的说明融入写作过程。每位参与者重新撰写了本人近期发表论文中两幅图表的说明，以最新的人工智能模型生成的说明作为参考资源。通过交互分析视频记录的写作过程，我们观察到，参与者通常会从复制和完善人工智能生成的说明开始。论文作者更偏好能整合文本和视觉元素的详细说明，但在处理复杂图表时发现当前的人工智能模型效果不佳。这些 findings 突显了图表说明撰写过程中的复杂性和多面性，揭示了设计机会，以便更好地支持学术写作中的挑战。 

---
# Dafny as Verification-Aware Intermediate Language for Code Generation 

**Title (ZH)**: Dafny作为具有验证意识的中间语言用于代码生成 

**Authors**: Yue Chen Li, Stefan Zetzsche, Siva Somayyajula  

**Link**: [PDF](https://arxiv.org/pdf/2501.06283)  

**Abstract**: Using large language models (LLMs) to generate source code from natural language prompts is a popular and promising idea with a wide range of applications. One of its limitations is that the generated code can be faulty at times, often in a subtle way, despite being presented to the user as correct. In this paper, we explore ways in which formal methods can assist with increasing the quality of code generated by an LLM. Instead of emitting code in a target language directly, we propose that the user guides the LLM to first generate an opaque intermediate representation, in the verification-aware language Dafny, that can be automatically validated for correctness against agreed on specifications. The correct Dafny program is then compiled to the target language and returned to the user. All user-system interactions throughout the procedure occur via natural language; Dafny code is never exposed. We describe our current prototype and report on its performance on the HumanEval Python code generation benchmarks. 

**Abstract (ZH)**: 使用大规模语言模型（LLMs）从自然语言提示生成源代码是一个流行且有前途的主意，具有广泛的应用前景。其局限性之一是生成的代码有时可能存在错误，尽管这些错误往往是微妙的，并且被呈现为正确的。在本文中，我们探讨了形式方法如何帮助提高LLM生成代码的质量。我们提出的方法不是直接生成目标语言的代码，而是建议用户引导LLM首先生成一个不透明的中间表示，该表示基于知晓验证的编程语言Dafny，并且该中间表示可以自动验证其正确性，以符合预先商定的规范。然后，正确的Dafny程序被编译为目标语言，并返回给用户。在整个过程中的所有用户-系统交互均通过自然语言进行；从未暴露Dafny代码。我们描述了当前的原型，并在HumanEval Python代码生成基准测试上报告了其性能。 

---
# PROEMO: Prompt-Driven Text-to-Speech Synthesis Based on Emotion and Intensity Control 

**Title (ZH)**: PROEMO：基于情感和强度控制的提示驱动文本到语音合成 

**Authors**: Shaozuo Zhang, Ambuj Mehrish, Yingting Li, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2501.06276)  

**Abstract**: Speech synthesis has significantly advanced from statistical methods to deep neural network architectures, leading to various text-to-speech (TTS) models that closely mimic human speech patterns. However, capturing nuances such as emotion and style in speech synthesis is challenging. To address this challenge, we introduce an approach centered on prompt-based emotion control. The proposed architecture incorporates emotion and intensity control across multi-speakers. Furthermore, we leverage large language models (LLMs) to manipulate speech prosody while preserving linguistic content. Using embedding emotional cues, regulating intensity levels, and guiding prosodic variations with prompts, our approach infuses synthesized speech with human-like expressiveness and variability. Lastly, we demonstrate the effectiveness of our approach through a systematic exploration of the control mechanisms mentioned above. 

**Abstract (ZH)**: 语音合成从统计方法显著发展到了深层神经网络架构，产生了各种能够密切模仿人类语音模式的文本到语音（TTS）模型。然而，在语音合成中捕捉情绪和风格的细微差别是具有挑战性的。为了解决这一挑战，我们提出了基于提示的情绪控制方法。所提出的架构在多说话人中实现了情绪和强度控制。此外，我们利用大规模语言模型（LLMs）来操控言语韵律，同时保留语言内容。通过嵌入情绪线索、调节强度级别，并用提示引导韵律变化，我们的方法赋予合成语音以类似人类的表达性和变异性。最后，我们通过系统性地探索上述控制机制的有效性来验证我们的方法。 

---
# Polarized Patterns of Language Toxicity and Sentiment of Debunking Posts on Social Media 

**Title (ZH)**: 社交媒体上辟谣帖子中语言毒性及驳斥情绪的极化模式 

**Authors**: Wentao Xu, Wenlu Fan, Shiqian Lu, Tenghao Li, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06274)  

**Abstract**: Here's a condensed 1920-character version: The rise of misinformation and fake news in online political discourse poses significant challenges to democratic processes and public engagement. While debunking efforts aim to counteract misinformation and foster fact-based dialogue, these discussions often involve language toxicity and emotional polarization. We examined over 86 million debunking tweets and more than 4 million Reddit debunking comments to investigate the relationship between language toxicity, pessimism, and social polarization in debunking efforts. Focusing on discussions of the 2016 and 2020 U.S. presidential elections and the QAnon conspiracy theory, our analysis reveals three key findings: (1) peripheral participants (1-degree users) play a disproportionate role in shaping toxic discourse, driven by lower community accountability and emotional expression; (2) platform mechanisms significantly influence polarization, with Twitter amplifying partisan differences and Reddit fostering higher overall toxicity due to its structured, community-driven interactions; and (3) a negative correlation exists between language toxicity and pessimism, with increased interaction reducing toxicity, especially on Reddit. We show that platform architecture affects informational complexity of user interactions, with Twitter promoting concentrated, uniform discourse and Reddit encouraging diverse, complex communication. Our findings highlight the importance of user engagement patterns, platform dynamics, and emotional expressions in shaping polarization in debunking discourse. This study offers insights for policymakers and platform designers to mitigate harmful effects and promote healthier online discussions, with implications for understanding misinformation, hate speech, and political polarization in digital environments. 

**Abstract (ZH)**: 以下是符合学术规范的中文翻译，总字符数为1925字符：

网络政治话语中虚假信息和假新闻的兴起对民主过程和公众参与构成了重大挑战。虽然揭露虚假信息的努力意在对抗虚假信息并促进基于事实的对话，但这些讨论往往涉及语言毒性和情绪极化。我们分析了超过8600万条揭露虚假信息的推特和超过400万条在Reddit上的揭露评论，以研究语言毒性和悲观情绪在揭露虚假信息过程中的关联和影响。重点关注2016年和2020年美国总统选举以及QAnon阴谋论的讨论，我们的分析揭示了三个关键发现：（1）外围参与者（1度用户）在塑造有毒言论方面扮演着不成比例的角色，这受到较低的社区责任感和情绪表达的影响；（2）平台机制显著影响极化现象，Twitter放大了党派间的差异，而Reddit由于其结构化且以社区驱动的互动方式，其整体毒性水平更高；（3）语言毒性与悲观情绪之间存在负相关关系，增加互动可降低毒性，尤其是在Reddit上。我们的研究显示，平台架构影响用户互动的信息复杂度，Twitter促进集中和统一的言论，而Reddit鼓励多样化且复杂的交流。这些发现强调了用户参与模式、平台动态和情绪表达在塑造揭露虚假信息过程中极化作用的重要性。本研究为政策制定者和平台设计师提供了减少有害影响和促进更健康网络讨论的见解，其意义还在于对理解数字环境中虚假信息、仇恨言论和政治极化等方面提供了启示。 

---
# $\text{Transformer}^2$: Self-adaptive LLMs 

**Title (ZH)**: $\text{Transformer}^2$: 自适应 LARGE LANGUAGE MODELS（或：$\text{Transformer}^2$: 自适应大语言模型） 

**Authors**: Qi Sun, Edoardo Cetin, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06252)  

**Abstract**: Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse tasks. We introduce \implname, a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During inference, \implname employs a two-pass mechanism: first, a dispatch system identifies the task properties, and then task-specific "expert" vectors, trained using reinforcement learning, are dynamically mixed to obtain targeted behavior for the incoming prompt. Our method outperforms ubiquitous approaches such as LoRA, with fewer parameters and greater efficiency. \implname demonstrates versatility across different LLM architectures and modalities, including vision-language tasks. \implname represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems. 

**Abstract (ZH)**: 自适应大型语言模型（LLMs）旨在解决传统微调方法所带来的挑战，这些方法通常计算密集且在处理多样化任务方面具有静态特性。我们引入了\implname，这是一种新颖的自适应框架，能够在实时环境下根据需要仅选择性地调整权重矩阵的单一组件，从而适应未见过的任务。在推理过程中，\implname 采用两步机制：首先，调度系统确定任务属性；然后，使用强化学习训练的任务特定“专家”向量会动态混合，以针对输入提示获得特定行为。我们的方法在参数更少且更高效的前提下，超越了诸如LoRA等通用方法。\implname 在不同LLM架构和模态（包括视觉-语言任务）中展示了适应性和灵活性。\implname 代表着一个重要的进步，提供了一种可扩展且高效的解决方案，以增强LLMs的自适应性和任务特定性能，为真正动态和自组织的人工智能系统铺平了道路。 

---
# Fitting Different Interactive Information: Joint Classification of Emotion and Intention 

**Title (ZH)**: 适应不同交互信息的建模：情感和意图的联合分类 

**Authors**: Xinger Li, Zhiqiang Zhong, Bo Huang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06215)  

**Abstract**: This paper is the first-place solution for ICASSP MEIJU@2025 Track I, which focuses on low-resource multimodal emotion and intention recognition. How to effectively utilize a large amount of unlabeled data, while ensuring the mutual promotion of different difficulty levels tasks in the interaction stage, these two points become the key to the competition. In this paper, pseudo-label labeling is carried out on the model trained with labeled data, and samples with high confidence and their labels are selected to alleviate the problem of low resources. At the same time, the characteristic of easy represented ability of intention recognition found in the experiment is used to make mutually promote with emotion recognition under different attention heads, and higher performance of intention recognition is achieved through fusion. Finally, under the refined processing data, we achieve the score of 0.5532 in the Test set, and win the championship of the track. 

**Abstract (ZH)**: 本文是ICASSP MEIJU@2025 Track I竞赛的第一名解决方案，该竞赛专注于低资源多模态情绪和意图识别。如何有效地利用大量无标签数据，并在交互阶段确保不同难度任务之间的相互促进，成为了竞赛的关键所在。本文通过使用标记数据训练的模型进行伪标签标注，从中选择置信度高的样本及其标签，以此来缓解资源不足的问题。同时，实验中发现的意图识别容易表示的特点被用于在不同注意力头下与情绪识别相互促进，从而通过融合提高了意图识别的表现。最后，在经过精细处理的数据上，我们在测试集上取得了0.5532的分数，并在赛道中赢得了冠军。 

---
# Leveraging Edge Intelligence and LLMs to Advance 6G-Enabled Internet of Automated Defense Vehicles 

**Title (ZH)**: 利用边缘智能和大语言模型推动6G赋能的自动化防御车辆物联网发展 

**Authors**: Murat Arda Onsu, Poonam Lohan, Burak Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2501.06205)  

**Abstract**: The evolution of Artificial Intelligence (AI) and its subset Deep Learning (DL), has profoundly impacted numerous domains, including autonomous driving. The integration of autonomous driving in military settings reduces human casualties and enables precise and safe execution of missions in hazardous environments while allowing for reliable logistics support without the risks associated with fatigue-related errors. However, relying on autonomous driving solely requires an advanced decision-making model that is adaptable and optimum in any situation. Considering the presence of numerous interconnected autonomous vehicles in mission-critical scenarios, Ultra-Reliable Low Latency Communication (URLLC) is vital for ensuring seamless coordination, real-time data exchange, and instantaneous response to dynamic driving environments. The advent of 6G strengthens the Internet of Automated Defense Vehicles (IoADV) concept within the realm of Internet of Military Defense Things (IoMDT) by enabling robust connectivity, crucial for real-time data exchange, advanced navigation, and enhanced safety features through IoADV interactions. On the other hand, a critical advancement in this space is using pre-trained Generative Large Language Models (LLMs) for decision-making and communication optimization for autonomous driving. Hence, this work presents opportunities and challenges with a vision of realizing the full potential of these technologies in critical defense applications, especially through the advancement of IoADV and its role in enhancing autonomous military operations. 

**Abstract (ZH)**: 人工智能（AI）及其子集深度学习（DL）的发展，对众多领域产生了深刻影响，其中包括自动驾驶。在军事环境中集成自动驾驶技术可以减少人员伤亡，实现危险环境中的精确和安全的任务执行，同时确保可靠的战略支援而不涉及疲劳引发的错误风险。然而，完全依赖自动驾驶要求具有高度适应性和在任何情况下都优化的决策模型。考虑到关键任务场景中存在大量互联的自动驾驶车辆，超可靠低延迟通信（URLLC）对于确保无缝协调、实时数据交换以及对动态驾驶环境的即时响应至关重要。6G的到来强化了军事防御物品的互联网（IoMDT）框架下的自动化防御车辆互联网（IoADV）概念，通过IoADV交互提供了强大的连接性，这对于实时数据交换、高级导航和增强安全性功能至关重要。另一方面，在这一领域的一个关键进展是使用预训练的生成型大型语言模型（LLMs）进行自动驾驶的决策和通信优化。因此，本文探讨了这些技术在关键防御应用中的潜力和挑战，特别是通过推进IoADV及其在增强自主军事操作中的作用，实现这些技术的全部潜能。 

---
# A Multimodal Social Agent 

**Title (ZH)**: 多模态社会代理模型 

**Authors**: Athina Bikaki, Ioannis A. Kakadiaris  

**Link**: [PDF](https://arxiv.org/pdf/2501.06189)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable progress in common-sense reasoning tasks. This ability is fundamental to understanding social dynamics, interactions, and communication. However, the potential of integrating computers with these social capabilities is still relatively unexplored. However, the potential of integrating computers with these social capabilities is still relatively unexplored. This paper introduces MuSA, a multimodal LLM-based agent that analyzes text-rich social content tailored to address selected human-centric content analysis tasks, such as question answering, visual question answering, title generation, and categorization. It uses planning, reasoning, acting, optimizing, criticizing, and refining strategies to complete a task. Our approach demonstrates that MuSA can automate and improve social content analysis, helping decision-making processes across various applications. We have evaluated our agent's capabilities in question answering, title generation, and content categorization tasks. MuSA performs substantially better than our baselines. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在常识推理任务中取得了显著进展。这种能力对于理解社会动态、交互和沟通至关重要。然而，将计算机与这些社会能力整合的可能性仍然相对未被充分探索。本文介绍了一种基于多模态LLM的代理MuSA，该代理专门用于分析富含文本的社会内容，以应对诸如问答、视觉问答、标题生成和内容分类等人本中心的内容分析任务。MuSA 使用规划、推理、执行、优化、批评和改进等策略来完成任务。我们的方法表明，MuSA 可以自动化并提升社会内容分析，帮助各类应用中的决策过程。我们已在问答、标题生成和内容分类任务上评估了该代理的能力。MuSA 在这些任务上的表现显著优于我们的基线模型。 

---
