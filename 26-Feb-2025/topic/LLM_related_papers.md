# DRAMA: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers 

**Title (ZH)**: DRAMA: 大型语言模型向小型密集检索模型的多样化增强 

**Authors**: Xueguang Ma, Xi Victoria Lin, Barlas Oguz, Jimmy Lin, Wen-tau Yih, Xilun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18460)  

**Abstract**: Large language models (LLMs) have demonstrated strong effectiveness and robustness while fine-tuned as dense retrievers. However, their large parameter size brings significant inference time computational challenges, including high encoding costs for large-scale corpora and increased query latency, limiting their practical deployment. While smaller retrievers offer better efficiency, they often fail to generalize effectively with limited supervised fine-tuning data. In this work, we introduce DRAMA, a training framework that leverages LLMs to train smaller generalizable dense retrievers. In particular, we adopt pruned LLMs as the backbone and train on diverse LLM-augmented data in a single-stage contrastive learning setup. Experiments show that DRAMA offers better multilingual and long-context capabilities than traditional encoder-based retrievers, and achieves strong performance across multiple tasks and languages. These highlight the potential of connecting the training of smaller retrievers with the growing advancements in LLMs, bridging the gap between efficiency and generalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在微调为密集检索器后显示出强大的有效性和鲁棒性。然而，它们庞大的参数规模带来了显著的推理时间计算挑战，包括大规模语料库的高编码成本和查询延迟的增加，从而限制了它们的实用部署。尽管较小的检索器更有效，但它们往往由于有限的监督微调数据而难以有效泛化。在本项工作中，我们引入了DRAMA，这是一种利用LLMs来训练更小的泛化密集检索器的训练框架。特别地，我们采用剪枝后的LLMs作为骨干，并在单阶段对比学习设置中对多样化增强的LLM数据进行训练。实验表明，DRAMA在多语言和长上下文能力方面优于传统的基于编码器的检索器，并在多个任务和语言上取得了强大的性能。这些结果凸显了连接较小检索器的训练与不断发展的LLMs的潜力，从而弥合了效率与泛化之间的差距。 

---
# TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning 

**Title (ZH)**: 基于文本的游戏：通过语言模型推理实现自学对弈的文本谜题游戏 

**Authors**: Frederikus Hudi, Genta Indra Winata, Ruochen Zhang, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2502.18431)  

**Abstract**: Reasoning is a fundamental capability of large language models (LLMs), enabling them to comprehend, analyze, and solve complex problems. In this paper, we introduce TextGames, an innovative benchmark specifically crafted to assess LLMs through demanding text-based games that require advanced skills in pattern recognition, spatial awareness, arithmetic, and logical reasoning. Our analysis probes LLMs' performance in both single-turn and multi-turn reasoning, and their abilities in leveraging feedback to correct subsequent answers through self-reflection. Our findings reveal that, although LLMs exhibit proficiency in addressing most easy and medium-level problems, they face significant challenges with more difficult tasks. In contrast, humans are capable of solving all tasks when given sufficient time. Moreover, we observe that LLMs show improved performance in multi-turn predictions through self-reflection, yet they still struggle with sequencing, counting, and following complex rules consistently. Additionally, models optimized for reasoning outperform pre-trained LLMs that prioritize instruction following, highlighting the crucial role of reasoning skills in addressing highly complex problems. 

**Abstract (ZH)**: 推理是大型语言模型（LLMs）的一项基本能力，使它们能够理解、分析并解决复杂问题。在本文中，我们介绍了TextGames，这是一种创新的基准测试，专门设计通过具有复杂要求的文字游戏来评估LLMs，这些游戏需要高级的模式识别、空间意识、算术和逻辑推理能力。我们的分析探索了LLMs在单步推理和多步推理中的表现，并考察了它们在自我反思过程中利用反馈修正后续答案的能力。我们的研究发现，虽然LLMs在处理大多数简单和中等难度问题上表现出色，但在更困难的任务上却面临重大挑战。相比之下，当给予足够的时间，人类可以解决所有任务。此外，我们观察到，通过自我反思，LLMs在多步预测中的表现有所提高，但仍难以一致地进行序列化、计数和遵循复杂规则。此外，优化用于推理的模型在解决高度复杂问题方面优于侧重指令遵循的预训练LLMs，这突显了推理技能在处理复杂问题中的关键作用。 

---
# GLEAN: Generalized Category Discovery with Diverse and Quality-Enhanced LLM Feedback 

**Title (ZH)**: GLEAN：广泛类别发现与多样性和质量增强的大语言模型反馈 

**Authors**: Henry Peng Zou, Siffi Singh, Yi Nian, Jianfeng He, Jason Cai, Saab Mansour, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.18414)  

**Abstract**: Generalized Category Discovery (GCD) is a practical and challenging open-world task that aims to recognize both known and novel categories in unlabeled data using limited labeled data from known categories. Due to the lack of supervision, previous GCD methods face significant challenges, such as difficulty in rectifying errors for confusing instances, and inability to effectively uncover and leverage the semantic meanings of discovered clusters. Therefore, additional annotations are usually required for real-world applicability. However, human annotation is extremely costly and inefficient. To address these issues, we propose GLEAN, a unified framework for generalized category discovery that actively learns from diverse and quality-enhanced LLM feedback. Our approach leverages three different types of LLM feedback to: (1) improve instance-level contrastive features, (2) generate category descriptions, and (3) align uncertain instances with LLM-selected category descriptions. Extensive experiments demonstrate the superior performance of \MethodName over state-of-the-art models across diverse datasets, metrics, and supervision settings. Our code is available at this https URL. 

**Abstract (ZH)**: 通用类别发现（GCD）是一项在有限标记数据条件下，旨在使用标记数据识别已知和未知类别中的未标记数据的实际且具有挑战性的开放世界任务。由于缺乏监督，之前的GCD方法面临着重大挑战，如难以纠正混淆实例的错误，以及无法有效地发现和利用发现簇的语义意义。因此，通常需要额外的注释以实现实际应用。然而，人工注释极为昂贵且效率低下。为解决这些问题，我们提出了一种统一的GCD框架GLEAN，该框架主动利用多样化和质量提升的语言模型（LLM）反馈进行学习。我们的方法利用了三种不同类型的LLM反馈：(1) 提高实例级对比特征，(2) 生成类别描述，以及(3) 将不确定实例与LLM选择的类别描述对齐。广泛实验表明，与最先进的模型相比，我们的方法在不同数据集、指标和监督设置上表现出更优的性能。我们的代码可从以下网址获得：[链接]。

（注：请将[链接]替换为实际的代码仓库地址。） 

---
# Monte Carlo Temperature: a robust sampling strategy for LLM's uncertainty quantification methods 

**Title (ZH)**: 蒙特卡洛温度：一种稳健的采样策略，用于量化大语言模型的不确定性方法 

**Authors**: Nicola Cecere, Andrea Bacciu, Ignacio Fernández Tobías, Amin Mantrach  

**Link**: [PDF](https://arxiv.org/pdf/2502.18389)  

**Abstract**: Uncertainty quantification (UQ) in Large Language Models (LLMs) is essential for their safe and reliable deployment, particularly in critical applications where incorrect outputs can have serious consequences. Current UQ methods typically rely on querying the model multiple times using non-zero temperature sampling to generate diverse outputs for uncertainty estimation. However, the impact of selecting a given temperature parameter is understudied, and our analysis reveals that temperature plays a fundamental role in the quality of uncertainty estimates. The conventional approach of identifying optimal temperature values requires expensive hyperparameter optimization (HPO) that must be repeated for each new model-dataset combination. We propose Monte Carlo Temperature (MCT), a robust sampling strategy that eliminates the need for temperature calibration. Our analysis reveals that: 1) MCT provides more robust uncertainty estimates across a wide range of temperatures, 2) MCT improves the performance of UQ methods by replacing fixed-temperature strategies that do not rely on HPO, and 3) MCT achieves statistical parity with oracle temperatures, which represent the ideal outcome of a well-tuned but computationally expensive HPO process. These findings demonstrate that effective UQ can be achieved without the computational burden of temperature parameter calibration. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的不确定性量化（UQ）对于其安全可靠的部署至关重要，特别是在那些不正确的输出可能造成严重后果的关键应用中。当前的不确定性量化方法通常依赖于使用非零温度采样多次查询模型，以生成多样化输出来进行不确定性估计。然而，选择特定温度参数的影响尚未得到充分研究，而我们的分析揭示了温度在不确定性估计质量中的基本作用。常规确定最佳温度值的方法需要昂贵的超参数优化（HPO），且必须为每种新的模型-数据集组合重复进行。我们提出了一种稳健的采样策略——蒙特卡洛温度（MCT）——该策略消除了温度校准的需求。我们的分析揭示了以下几点：1）MCT在广泛的温度范围内提供了更稳健的不确定性估计，2）MCT通过替代依赖HPO固定温度策略，提升了不确定性量化方法的性能，3）MCT实现了与理想得出的校准温度的统计平等性，后者代表了经过精心调优但计算成本高昂的HPO过程的理想结果。这些发现表明，有效的同时避免了计算上的温度参数校准负担。 

---
# Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC): Topic Modelling and LLM applied to Stroboscopic Phenomenology 

**Title (ZH)**: 将主观叙述映射到解释簇（MOSAIC）：主题建模与大型语言模型在瞬时现象学中的应用 

**Authors**: Romy Beauté, David J. Schwartzman, Guillaume Dumas, Jennifer Crook, Fiona Macpherson, Adam B. Barrett, Anil K. Seth  

**Link**: [PDF](https://arxiv.org/pdf/2502.18318)  

**Abstract**: Stroboscopic light stimulation (SLS) on closed eyes typically induces simple visual hallucinations (VHs), characterised by vivid, geometric and colourful patterns. A dataset of 862 sentences, extracted from 422 open subjective reports, was recently compiled as part of the Dreamachine programme (Collective Act, 2022), an immersive multisensory experience that combines SLS and spatial sound in a collective setting. Although open reports extend the range of reportable phenomenology, their analysis presents significant challenges, particularly in systematically identifying patterns. To address this challenge, we implemented a data-driven approach leveraging Large Language Models and Topic Modelling to uncover and interpret latent experiential topics directly from the Dreamachine's text-based reports. Our analysis confirmed the presence of simple VHs typically documented in scientific studies of SLS, while also revealing experiences of altered states of consciousness and complex hallucinations. Building on these findings, our computational approach expands the systematic study of subjective experience by enabling data-driven analyses of open-ended phenomenological reports, capturing experiences not readily identified through standard questionnaires. By revealing rich and multifaceted aspects of experiences, our study broadens our understanding of stroboscopically-induced phenomena while highlighting the potential of Natural Language Processing and Large Language Models in the emerging field of computational (neuro)phenomenology. More generally, this approach provides a practically applicable methodology for uncovering subtle hidden patterns of subjective experience across diverse research domains. 

**Abstract (ZH)**: 强制闪光刺激（SLS）闭眼时通常会诱导出简单的视觉错觉（VHs），其特点为生动、几何形状和色彩斑斓的图案。最近，作为Dreamachine项目（Collective Act, 2022）的一部分，从422份开放式主观报告中提取了862句话，汇编成一个数据集。该梦机项目是一个结合SLS和空间声音的沉浸式多感官体验，适用于集体环境。尽管开放式报告扩展了可报告的现象学范围，但对其分析提出了显著的挑战，尤其是系统地识别模式方面。为解决这一挑战，我们采用了一种基于数据的方法，利用大型语言模型和主题建模技术，直接从梦机项目的基于文本的报告中发现和解释潜在的经验主题。我们的分析确认了科学文献中通常记录的SLS引起的简单视觉错觉的存在，同时也揭示了意识状态改变和复杂幻觉的经历。基于这些发现，我们的计算方法扩展了对主观体验的系统研究，通过基于数据的方法分析开放性现象学报告，捕捉到标准问卷无法识别的经验。通过揭示体验的丰富和多层次方面，我们的研究扩大了对SLS诱导现象的理解，同时突显了自然语言处理和大型语言模型在新兴的计算（神经）现象学领域中的潜力。更广泛地说，这种方法为我们提供了在不同研究领域中发现主观体验中微妙隐藏模式的实用方法。 

---
# RefuteBench 2.0 -- Agentic Benchmark for Dynamic Evaluation of LLM Responses to Refutation Instruction 

**Title (ZH)**: RefuteBench 2.0 —基于代理视角的动态评估大规模语言模型反驳指令响应的标准库 

**Authors**: Jianhao Yan, Yun Luo, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18308)  

**Abstract**: In the multi-turn interaction schema, large language models (LLMs) can leverage user feedback to enhance the quality and relevance of their responses. However, evaluating an LLM's ability to incorporate user refutation feedback is crucial yet challenging. In this study, we introduce RefuteBench 2.0, which significantly extends the original RefuteBench by incorporating LLM agents as refuters and evaluators, which allows for flexible and comprehensive assessment.
We design both transient and persistent refutation instructions with different validity periods. Meta-evaluation shows that the LLM-based refuter could generate more human-like refutations and the evaluators could assign scores with high correlation with humans. Experimental results of various LLMs show that current models could effectively satisfy the refutation but fail to memorize the refutation information. Interestingly, we also observe that the performance of the initial task decreases as the refutations increase. Analysis of the attention scores further shows a potential weakness of current LLMs: they struggle to retain and correctly use previous information during long context dialogues. this https URL 

**Abstract (ZH)**: 在多轮交互模式下，大型语言模型（LLMs）能够利用用户反馈来提高其响应的质量和相关性。然而，评估LLM Incorporate用户反驳反馈的能力既关键又具有挑战性。在本研究中，我们介绍了RefuteBench 2.0，这是一个显著扩展了原始RefuteBench的评估框架，通过引入基于LLM的反驳者和评估者，使评估更加灵活和全面。

我们设计了具有不同有效期限的临时反驳指令和持久反驳指令。元评估显示，基于LLM的反驳者能够生成更加接近人类的反驳，评估者能够给出与人类高度相关的评分。各种LLM的实验证据表明，当前模型能够有效地回应反驳，但未能妥善保留反驳信息。有趣的是，我们还观察到，随着反驳次数的增加，初始任务的表现逐渐下降。对注意力分数的进一步分析显示，当前LLM可能存在的一个潜在弱点：它们在长时间上下文对话中难以有效地保留和正确使用先前的信息。

参考链接：[此处提供参考链接] 

---
# Debt Collection Negotiations with Large Language Models: An Evaluation System and Optimizing Decision Making with Multi-Agent 

**Title (ZH)**: 使用大型语言模型进行债务追收谈判：评估系统与多代理优化决策研究 

**Authors**: Xiaofeng Wang, Zhixin Zhang, Jinguang Zheng, Yiming Ai, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18228)  

**Abstract**: Debt collection negotiations (DCN) are vital for managing non-performing loans (NPLs) and reducing creditor losses. Traditional methods are labor-intensive, while large language models (LLMs) offer promising automation potential. However, prior systems lacked dynamic negotiation and real-time decision-making capabilities. This paper explores LLMs in automating DCN and proposes a novel evaluation framework with 13 metrics across 4 aspects. Our experiments reveal that LLMs tend to over-concede compared to human negotiators. To address this, we propose the Multi-Agent Debt Negotiation (MADeN) framework, incorporating planning and judging modules to improve decision rationality. We also apply post-training techniques, including DPO with rejection sampling, to optimize performance. Our studies provide valuable insights for practitioners and researchers seeking to enhance efficiency and outcomes in this domain. 

**Abstract (ZH)**: 债务催收谈判（DCN）对于管理不良贷款（NPLs）和减少债权人的损失至关重要。传统方法耗时且劳动密集，而大型语言模型（LLMs）则提供了自动化的潜力。然而，之前的系统缺乏动态谈判和实时决策的能力。本文探讨了利用LLMs自动进行DCN的方法，并提出了一种全新的评估框架，涵盖4个方面的13个指标。我们的实验表明，与人类谈判者相比，LLMs往往会过度让步。为解决这一问题，我们提出了一个多智能体债务谈判（MADeN）框架，该框架通过引入计划和判断模块以提高决策的合理性。此外，我们还应用了后训练技术，包括带 rejection sampling 的 DPO，以优化性能。我们的研究为希望在此领域提高效率和成果的实践者和研究人员提供了宝贵的见解。 

---
# LAG: LLM agents for Leaderboard Auto Generation on Demanding 

**Title (ZH)**: LAG: 基于大语言模型的排行榜自动生成代理人

在这个翻译中，“LLM”是指大语言模型（Large Language Model），由于这是学术或技术性的翻译，我们将它直接翻译为“大语言模型”。"Leaderboard Auto Generation"则翻译为“排行榜自动生成”，这里的“Leaderboard”指的是排行榜或得分榜，"Auto Generation"指的是自动生成。"on Demanding"可以理解为“针对需求的”或“在需求下”，为了使翻译更符合中文的表达习惯和学术规范，我们将其翻译为“基于”。完整的标题翻译为：“LAG: 基于大语言模型的排行榜自动生成代理人”。 

**Authors**: Jian Wu, Jiayu Zhang, Dongyuan Li, Linyi Yang, Aoxiao Zhong, Renhe Jiang, Qingsong Wen, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18209)  

**Abstract**: This paper introduces Leaderboard Auto Generation (LAG), a novel and well-organized framework for automatic generation of leaderboards on a given research topic in rapidly evolving fields like Artificial Intelligence (AI). Faced with a large number of AI papers updated daily, it becomes difficult for researchers to track every paper's proposed methods, experimental results, and settings, prompting the need for efficient automatic leaderboard construction. While large language models (LLMs) offer promise in automating this process, challenges such as multi-document summarization, leaderboard generation, and experiment fair comparison still remain under exploration. LAG solves these challenges through a systematic approach that involves the paper collection, experiment results extraction and integration, leaderboard generation, and quality evaluation. Our contributions include a comprehensive solution to the leaderboard construction problem, a reliable evaluation method, and experimental results showing the high quality of leaderboards. 

**Abstract (ZH)**: 本文介绍了领袖榜自动生成（Leaderboard Auto Generation, LAG）框架，这是一种在快速发展的领域如人工智能（AI）中，针对给定的研究主题自动生成领袖榜的新型且井然有序的方法。面对每天更新的大量AI论文，研究人员难以跟踪每篇论文提出的方法、实验结果和设置，从而推动了高效自动生成领袖榜的需求。尽管大型语言模型（LLMs）在自动化这一过程中展现出潜力，但在多文档总结、领袖榜生成和实验公平比较等方面仍存在挑战。LAG通过系统的方法解决了这些问题，该方法包括论文收集、实验结果提取与整合、领袖榜生成以及质量评估。我们的贡献包括对领袖榜生成问题的全面解决方案、可靠的评估方法以及实验结果，这些结果表明生成的领袖榜具有高质量。 

---
# Grandes modelos de lenguaje: de la predicción de palabras a la comprensión? 

**Title (ZH)**: 大型语言模型：从单词预测到理解？ 

**Authors**: Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2502.18205)  

**Abstract**: Large language models, such as the well-known ChatGPT, have brought about an unexpected revolution in the field of artificial intelligence. On the one hand, they have numerous practical applications and enormous potential still to be explored. On the other hand, they are also the subject of debate from scientific, philosophical, and social perspectives: there are doubts about the exact mechanisms of their functioning and their actual capacity for language comprehension, and their applications raise ethical dilemmas. In this chapter, we describe how this technology has been developed and the fundamentals of its operation, allowing us to better understand its capabilities and limitations and to introduce some of the main debates surrounding its development and use.
--
Los grandes modelos de lenguaje, como el conocido ChatGPT, han supuesto una inesperada revolución en el ámbito de la inteligencia artificial. Por un lado, cuentan con multitud de aplicaciones prácticas y un enorme potencial todavía por explorar. Por otro lado, son también objeto de debate, tanto desde el punto de vista científico y filosófico como social: hay dudas sobre los mecanismos exactos de su funcionamiento y su capacidad real de comprensión del lenguaje, y sus aplicaciones plantean dilemas éticos. En este capítulo describimos cómo se ha llegado a esta tecnología y los fundamentos de su funcionamiento, permitiéndonos así comprender mejor sus capacidades y limitaciones e introducir algunos de los principales debates que rodean su desarrollo y uso. 

**Abstract (ZH)**: 大型语言模型，如著名的ChatGPT，已在人工智能领域带来了意想不到的革命。一方面，它们具有多种实际应用并拥有巨大的尚未探索的潜力。另一方面，这些模型也从科学、哲学和社会视角引发了争议：人们对它们的具体工作机制及其语言理解能力表示怀疑，而且它们的应用还引发了伦理困境。在本章中，我们将描述这项技术的发展过程及其基本原理，从而更好地理解其能力和局限性，并介绍围绕其发展和使用的一些主要争议。 

---
# Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs 

**Title (ZH)**: 《问题解决了吗？基于LLM的布局丰富文档的信息提取设计空间》 

**Authors**: Gaye Colakoglu, Gürkan Solmaz, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2502.18179)  

**Abstract**: This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study delves into the sub-problems within these core challenges, such as input representation, chunking, prompting, and selection of LLMs and multimodal models. It examines the outcomes of different design choices through a new layout-aware IE test suite, benchmarking against the state-of-art (SoA) model LayoutLMv3. The results show that the configuration from one-factor-at-a-time (OFAT) trial achieves near-optimal results with 14.1 points F1-score gain from the baseline model, while full factorial exploration yields only a slightly higher 15.1 points gain at around 36x greater token usage. We demonstrate that well-configured general-purpose LLMs can match the performance of specialized models, providing a cost-effective alternative. Our test-suite is freely available at this https URL. 

**Abstract (ZH)**: 本文定义并探索了使用大规模语言模型（LLMs）从布局丰富的文档中提取信息（IE）的设计空间。布局感知IE使用LLMs面临的三大核心挑战包括：1) 数据结构化，2) 模型互动，以及3) 输出润色。我们的研究深入探讨了这些核心挑战下的子问题，例如输入表示、分块、提示以及LLMs和多模态模型的选择。通过一个创新的布局感知IE测试套件，我们将结果与当前最先进的模型（LayoutLMv3）进行基准测试。结果表明，单因素一次性试验（one-factor-at-a-time, OFAT）的配置达到了近最优效果，基线模型的F1分数提高了14.1分，而全方位试验尽管在36倍于全量级的标记使用下，仅提高了略微高一点的15.1分。我们证明，配置良好的通用模型可以与专用模型匹配，提供一种成本效益更高的替代方案。我们的测试套件可在此处免费获取：[此 https URL]。 

---
# Can LLMs Explain Themselves Counterfactually? 

**Title (ZH)**: 大语言模型能否进行反事实解释？ 

**Authors**: Zahra Dehghanighobadi, Asja Fischer, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2502.18156)  

**Abstract**: Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning. 

**Abstract (ZH)**: 解释是理解机器学习模型行为的重要工具，有助于校准用户信任并确保合规性。近年来，涌现出了许多后验方法来生成模型解释，其中许多方法涉及计算模型梯度或解决特别设计的优化问题。然而，由于大型语言模型（LLMs）卓越的推理能力，促使模型自我解释，即提示模型解释其输出，已成为一种新的范式。在这项工作中，我们研究了一种特定类型的自我解释，即自动生成反事实解释（SCEs）。我们设计了测试来衡量LLMs生成SCEs的有效性。通过对各种LLM家族、模型大小、温度设置和数据集的分析表明，LLMs有时难以生成SCEs。即使生成了SCEs，它们的预测往往也不与其自身的反事实推理一致。 

---
# Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning 

**Title (ZH)**: 面向推理最优的大型语言模型测试时计算量缩放方法 

**Authors**: Wenkai Yang, Shuming Ma, Yankai Lin, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.18080)  

**Abstract**: Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with QwQ-32B-Preview. 

**Abstract (ZH)**: 近期的研究表明，让模型通过更长的推理链（Chain of Thoughts, CoTs）进行深入思考，能够显著改善其在复杂推理任务中的表现。虽然当前研究仍在探索通过增加大型语言模型（LLMs）的CoTs长度以扩大测试时间计算能力所带来的好处，但我们担忧当前追求测试时间扩展背后可能隐藏的一个潜在问题：过长的CoTs 是否会实际上对模型的推理性能产生负面影响？通过对数学推理任务的研究，我们的探索揭示了一个意想不到的结果：在某些领域中，随着CoTs长度的增加，确实会损害LLMs的推理性能。此外，我们发现不同领域存在一个最优的CoTs长度分布。基于这些发现，我们提出了一种称为“最优推理长度调整”（Thinking-Optimal Scaling）的策略。该方法首先使用一组具有不同响应长度分布的种子数据来教导模型在深度思考时采用不同的推理努力。然后，模型在其推理努力不同的情况下选择最短的正确响应，以实现自我改进。基于此策略改进的模型，在Qwen2.5-32B-Instruct的基础上，跨越各种数学基准测试的表现优于其他基于蒸馏的32B o1-like模型，并在某些方面达到了与QwQ-32B-Preview相当的性能。 

---
# Harnessing Multiple Large Language Models: A Survey on LLM Ensemble 

**Title (ZH)**: 多种大型语言模型的综合利用：面向LLM集成的综述

在这个翻译中，“Harnessing Multiple Large Language Models”被翻译为“多种大型语言模型的综合利用”，“A Survey on LLM Ensemble”被翻译为“面向LLM集成的综述”。这样的翻译既保留了原意，又符合学术规范。 

**Authors**: Zhijun Chen, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun, Yuankai Luo, Qianren Mao, Dingqi Yang, Hailong Sun, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18036)  

**Abstract**: LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference", and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型集成（LLM Ensemble）——涉及综合使用多个大规模语言模型（LLMs），每个模型旨在处理下游推理过程中的用户查询，从而利用它们各自的优点——最近获得了广泛关注。大规模语言模型的广泛可用性及其各不相同的优点和即插即用的特性，极大地推动了LLM Ensemble领域的发展。本文提供了对近期LLM Ensemble发展的首次系统性综述。首先，我们介绍LLM Ensemble的分类体系，并讨论一些相关研究问题。然后，我们在“推理前集成、推理中集成、推理后集成”等大类下进行了更深入的分类，并回顾所有相关方法。最后，我们介绍了相关基准和应用，总结了现有研究，并提出了若干未来研究方向。LLM Ensemble相关的论文列表可通过以下链接访问：[这个链接](这个链接)。 

---
# Verdict: A Library for Scaling Judge-Time Compute 

**Title (ZH)**: 判决：一个扩展判决时计算的库 

**Authors**: Nimit Kalra, Leonard Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18018)  

**Abstract**: The use of LLMs as automated judges ("LLM-as-a-judge") is now widespread, yet standard judges suffer from a multitude of reliability issues. To address these challenges, we introduce Verdict, an open-source library for scaling judge-time compute to enhance the accuracy, reliability, and interpretability of automated evaluators. Verdict leverages the composition of modular reasoning units -- such as verification, debate, and aggregation -- and increased inference-time compute to improve LLM judge quality. Across a variety of challenging tasks such as content moderation, fact-checking, and hallucination detection, Verdict judges achieve state-of-the-art (SOTA) or near-SOTA performance, surpassing orders-of-magnitude larger fine-tuned judges, prompted judges, and reasoning models. Ultimately, we hope Verdict serves as a useful framework for researchers and practitioners building scalable, interpretable, and reliable LLM-based evaluators. 

**Abstract (ZH)**: 作为自动化法官（“LLM-as-a-judge”）的大型语言模型（LLM）如今已被广泛应用，然而标准法官面临着众多可靠性问题。为应对这些挑战，我们提出了Verdict——一个开源库，用于扩展法官时间计算量，以提升自动化评估器的准确度、可靠性和可解释性。Verdict 利用模块化推理单元（如验证、辩论和聚合）的组合以及增加推理时间计算量来提高 LLM 法官的质量。在内容审核、事实核查和幻觉检测等各种具有挑战性的任务中，Verdict 法官实现了目前最先进（SOTA）或接近最先进（near-SOTA）的性能，超越了大量微调法官、触发法官和推理模型。最终，我们希望Verdict能够作为研究人员和实践者构建可扩展、可解释且可靠的基于LLM的评估器的一个有用框架。 

---
# Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning 

**Title (ZH)**: 揭开提炼链式推理关键因素的面纱 

**Authors**: Xinghao Chen, Zhijing Sun, Wenjin Guo, Miaoran Zhang, Yanjun Chen, Yirong Sun, Hui Su, Yijie Pan, Dietrich Klakow, Wenjie Li, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18001)  

**Abstract**: Large Language Models (LLMs) excel in reasoning tasks through Chain-of-Thought (CoT) prompting. However, CoT prompting greatly increases computational demands, which has prompted growing interest in distilling CoT capabilities into Small Language Models (SLMs). This study systematically examines the factors influencing CoT distillation, including the choice of granularity, format and teacher model. Through experiments involving four teacher models and seven student models across seven mathematical and commonsense reasoning datasets, we uncover three key findings: (1) Unlike LLMs, SLMs exhibit a non-monotonic relationship with granularity, with stronger models benefiting from finer-grained reasoning and weaker models performing better with simpler CoT supervision; (2) CoT format significantly impacts LLMs but has minimal effect on SLMs, likely due to their reliance on supervised fine-tuning rather than pretraining preferences; (3) Stronger teacher models do NOT always produce better student models, as diversity and complexity in CoT supervision can outweigh accuracy alone. These findings emphasize the need to tailor CoT strategies to specific student model, offering actionable insights for optimizing CoT distillation in SLMs. The code and datasets are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通过链式思考（CoT）提示进行推理任务时表现出色。然而，CoT提示显著增加了计算需求，从而引发了将CoT能力提炼到小型语言模型（SLMs）中的兴趣。本研究系统地探讨了影响CoT提炼的关键因素，包括粒度、格式和教师模型的选择。通过涉及四款教师模型和七款学生模型在七个数学和常识推理数据集中的实验，我们发现了三个重要发现：（1）与LLMs不同，SLMs在粒度方面表现出非单调关系，较强的模型可以从更细粒度的推理中受益，较弱的模型则更适用于简化的CoT监督；（2）CoT格式对LLMs有显著影响，但对SLMs的影响较小，这可能是因为SLMs依赖于监督微调，而不是预训练偏好；（3）较强的教师模型并不总是产生更好的学生模型，因为CoT监督的多样性和复杂性可能会超越单纯的准确性。这些发现强调了需要针对具体的学生模型定制CoT策略，为优化SLMs中的CoT提炼提供了可操作性的见解。相关代码和数据集可在以下链接获取：[请提供具体的URL链接]。 

---
# Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments 

**Title (ZH)**: 朝更好地理解跨语言和多语言环境中的程序思维推理进发 

**Authors**: Patomporn Payoungkhamdee, Pume Tuchinda, Jinheon Baek, Samuel Cahyawijaya, Can Udomcharoenchaikit, Potsawee Manakul, Peerat Limkonchotiwat, Ekapol Chuangsuwanich, Sarana Nutanong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17956)  

**Abstract**: Multi-step reasoning is essential for large language models (LLMs), yet multilingual performance remains challenging. While Chain-of-Thought (CoT) prompting improves reasoning, it struggles with non-English languages due to the entanglement of reasoning and execution. Program-of-Thought (PoT) prompting separates reasoning from execution, offering a promising alternative but shifting the challenge to generating programs from non-English questions. We propose a framework to evaluate PoT by separating multilingual reasoning from code execution to examine (i) the impact of fine-tuning on question-reasoning alignment and (ii) how reasoning quality affects answer correctness. Our findings demonstrate that PoT fine-tuning substantially enhances multilingual reasoning, outperforming CoT fine-tuned models. We further demonstrate a strong correlation between reasoning quality (measured through code quality) and answer accuracy, highlighting its potential as a test-time performance improvement heuristic. 

**Abstract (ZH)**: 多步推理是大型语言模型（LLMs）的重要功能，但跨语言性能仍然具有挑战性。虽然思维链（Chain-of-Thought，CoT）提示可以提高推理能力，但在处理非英语语言时由于推理与执行的交织，效果大打折扣。程序思维（Program-of-Thought，PoT）提示将推理与执行分离，提供了一种有前景的替代方案，但也把挑战转移到从非英语问题生成程序上。我们提出了一种框架，通过分离多语言推理和代码执行来评估PoT，以考察（i）微调对问题-推理对齐的影响，以及（ii）推理质量如何影响答案的正确性。我们的研究发现表明，PoT微调在多语言推理方面显著优于CoT微调模型。我们进一步证明，通过代码质量衡量的推理质量与答案准确性之间存在强烈的相关性，突显了其作为测试时性能改进启发式的潜力。 

---
# Advantage-Guided Distillation for Preference Alignment in Small Language Models 

**Title (ZH)**: 优势引导的精炼方法在小型语言模型中实现偏好对齐 

**Authors**: Shiping Gao, Fanqi Wan, Jiajian Guo, Xiaojun Quan, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17927)  

**Abstract**: Alignment techniques enable Large Language Models (LLMs) to generate outputs that align with human preferences and play a crucial role in their effectiveness. However, their impact often diminishes when applied to Small Language Models (SLMs), likely due to the limited capacity of these models. Instead of directly applying existing alignment techniques to SLMs, we propose to utilize a well-aligned teacher LLM to guide the alignment process for these models, thereby facilitating the transfer of the teacher's knowledge of human preferences to the student model. To achieve this, we first explore a straightforward approach, Dual-Constrained Knowledge Distillation (DCKD), that employs knowledge distillation with two KL-divergence constraints from the aligned teacher to the unaligned student. To further enhance the student's ability to distinguish between preferred and dispreferred responses, we then propose Advantage-Guided Distillation for Preference Alignment (ADPA), which leverages an advantage function from the aligned teacher to deliver more nuanced, distribution-level reward signals for the student's alignment. Our experimental results show that these two approaches appreciably improve the alignment of SLMs and narrow the performance gap with larger counterparts. Among them, ADPA demonstrates superior performance and achieves even greater effectiveness when integrated with DCKD. Our code is available at this https URL. 

**Abstract (ZH)**: 以下是将上述内容翻译成中文的版本，符合学术规范：

对齐技术使大语言模型（LLMs）生成与人类偏好相一致的输出，并在提高其有效性方面发挥着关键作用。然而，当应用于小语言模型（SLMs）时，这些技术的影响往往会减弱，这可能是因为这些模型的容量有限。与其直接将现有的对齐技术应用于SLMs，我们提出利用一个已经对齐的教师LLM来指导这些模型的对齐过程，从而促进教师对人类偏好的理解转移到学生模型中。为实现这一目标，我们首先探索了一种简单的双约束知识蒸馏（DCKD）方法，该方法使用两种KL散度约束将对齐的教师知识传递给未对齐的学生。为进一步提升学生模型区分偏好和非偏好响应的能力，我们随后提出了偏好对齐引导知识蒸馏（ADPA）方法，该方法利用对齐教师提供的优势函数来为学生模型传递更细腻、更分布层面的奖励信号。我们的实验结果显示，这两种方法显著提高了SLMs的对齐效果，并缩小了其与较大模型之间的性能差距。其中，ADPA表现出更出色的性能，并且在与DCKD结合使用时，其效果更为显著。我们的代码已发布在 [这里的URL]。 

---
# FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models 

**Title (ZH)**: FACT-AUDIT：一种适应性多代理框架，用于大型语言模型动态事实核查评估 

**Authors**: Hongzhan Lin, Yang Deng, Yuxuan Gu, Wenxuan Zhang, Jing Ma, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.17924)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the fact-checking studies. However, existing automated fact-checking evaluation methods rely on static datasets and classification metrics, which fail to automatically evaluate the justification production and uncover the nuanced limitations of LLMs in fact-checking. In this work, we introduce FACT-AUDIT, an agent-driven framework that adaptively and dynamically assesses LLMs' fact-checking capabilities. Leveraging importance sampling principles and multi-agent collaboration, FACT-AUDIT generates adaptive and scalable datasets, performs iterative model-centric evaluations, and updates assessments based on model-specific responses. By incorporating justification production alongside verdict prediction, this framework provides a comprehensive and evolving audit of LLMs' factual reasoning capabilities, to investigate their trustworthiness. Extensive experiments demonstrate that FACT-AUDIT effectively differentiates among state-of-the-art LLMs, providing valuable insights into model strengths and limitations in model-centric fact-checking analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）在事实核查研究中取得了显著进展。然而，现有的自动化事实核查评估方法依赖于静态数据集和分类指标，无法自动评估推理过程中的论证生成，并揭示LLMs在事实核查中的细微限制。在此项工作中，我们引入了FACT-AUDIT，这是一种基于代理的框架，能够适应性和动态地评估LLMs的事实核查能力。通过利用重要性采样原理和多代理协作，FACT-AUDIT生成适应性和可扩展的数据集，进行迭代的模型导向评估，并根据模型特定的响应更新评估结果。通过结合论证生成与判决预测，这一框架提供了对LLMs事实推理能力的全面且不断演化的审计，以研究其可信度。大量实验证明，FACT-AUDIT能够有效地区分最先进的LLMs，并提供了有关模型在模型导向的事实核查分析中优势和局限性的宝贵见解。 

---
# Can Large Language Models Identify Implicit Suicidal Ideation? An Empirical Evaluation 

**Title (ZH)**: 大型语言模型能否识别隐性自杀观念？一项实证评估 

**Authors**: Tong Li, Shu Yang, Junchao Wu, Jiyao Wei, Lijie Hu, Mengdi Li, Derek F. Wong, Joshua R. Oltmanns, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17899)  

**Abstract**: We present a comprehensive evaluation framework for assessing Large Language Models' (LLMs) capabilities in suicide prevention, focusing on two critical aspects: the Identification of Implicit Suicidal ideation (IIS) and the Provision of Appropriate Supportive responses (PAS). We introduce \ourdata, a novel dataset of 1,308 test cases built upon psychological frameworks including D/S-IAT and Negative Automatic Thinking, alongside real-world scenarios. Through extensive experiments with 8 widely used LLMs under different contextual settings, we find that current models struggle significantly with detecting implicit suicidal ideation and providing appropriate support, highlighting crucial limitations in applying LLMs to mental health contexts. Our findings underscore the need for more sophisticated approaches in developing and evaluating LLMs for sensitive psychological applications. 

**Abstract (ZH)**: 我们提出了一种综合评估框架，用于评估大型语言模型（LLMs）在自杀预防方面的能力，重点关注两个关键方面：隐性自杀意念的识别（IIS）和适当支持性回应的提供（PAS）。我们引入了\ourdata，这是一个基于心理框架（包括D/S-IAT和负性自动化思维）以及现实世界场景构建的新型数据集，包含1,308个测试案例。通过在不同情境下对8个广泛使用的LLM进行广泛的实验，我们发现当前模型在检测隐性自杀意念和提供适当支持方面存在显著困难，突显了在心理健康背景下应用LLMs的重要局限性。我们的研究结果强调了在敏感的心理学应用中开发和评估LLMs时需要更加复杂的方法。 

---
# RankCoT: Refining Knowledge for Retrieval-Augmented Generation through Ranking Chain-of-Thoughts 

**Title (ZH)**: RankCoT：通过排序链思考为检索增强生成精炼知识 

**Authors**: Mingyan Wu, Zhenghao Liu, Yukun Yan, Xinze Li, Shi Yu, Zheni Zeng, Yu Gu, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17888)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the performance of Large Language Models (LLMs) by incorporating external knowledge. However, LLMs still encounter challenges in effectively utilizing the knowledge from retrieved documents, often being misled by irrelevant or noisy information. To address this issue, we introduce RankCoT, a knowledge refinement method that incorporates reranking signals in generating CoT-based summarization for knowledge refinement based on given query and all retrieval documents. During training, RankCoT prompts the LLM to generate Chain-of-Thought (CoT) candidates based on the query and individual documents. It then fine-tunes the LLM to directly reproduce the best CoT from these candidate outputs based on all retrieved documents, which requires LLM to filter out irrelevant documents during generating CoT-style summarization. Additionally, RankCoT incorporates a self-reflection mechanism that further refines the CoT outputs, resulting in higher-quality training data. Our experiments demonstrate the effectiveness of RankCoT, showing its superior performance over other knowledge refinement models. Further analysis reveals that RankCoT can provide shorter but effective refinement results, enabling the generator to produce more accurate answers. All code and data are available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）通过引入外部知识来提高大型语言模型（LLMs）的性能。然而，LLMs 在有效地利用检索文档中的知识时仍面临挑战，常常受到无关或噪声信息的误导。为了解决这一问题，我们引入了 RankCoT，这是一种知识精炼方法，基于给定查询和所有检索文档，在生成基于CoT的总结时结合使用再排序信号进行知识精炼。

在训练过程中，RankCoT 促使LLM 基于查询和每个文档生成CoT候选答案。随后，它对LLM 进行微调，使其直接根据所有检索文档再现最佳CoT，这要求LLM 在生成CoT风格的总结时过滤掉不必要的文档信息。此外，RankCoT 还引入了一种自我反思机制，进一步精炼CoT 输出，从而产生更高质量的训练数据。我们的实验证明了 RankCoT 的有效性，并显示其性能优于其他知识精炼模型。进一步的分析表明，RankCoT 可以提供更短但更有效的精炼结果，使生成器能够生成更准确的答案。所有代码和数据均可在以下链接获取：这个 https URL。 

---
# SYNTHEMPATHY: A Scalable Empathy Corpus Generated Using LLMs Without Any Crowdsourcing 

**Title (ZH)**: SYNTHEMPATHY：一种使用大型语言模型生成的无 crowdsourcing 的可扩展同理心数据集 

**Authors**: Run Chen, Jun Shin, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.17857)  

**Abstract**: Previous research has shown that humans are more receptive towards language models that that exhibit empathetic behavior. While empathy is essential for developing helpful dialogue agents, very few large corpora containing empathetic dialogues are available for fine-tune LLMs. The few existing corpora have largely relied on crowdsourcing to simulate empathetic conversations, a process that is expensive, time-consuming, and not scalable to larger datasets. We propose a data generation framework for developing SYNTHEMPATHY, a large corpus containing 105k empathetic responses to real-life situations compiled through LLM generation. A base Mistral 7B model fine-tuned on our SYNTHEMPATHY corpus exhibits an increase in the average empathy score. 

**Abstract (ZH)**: 以往的研究表明，人类更倾向于接受表现出共情行为的语言模型。共情对于开发有帮助的对话代理至关重要，但可供微调大规模语言模型的大规模共情对话数据集非常有限。现有的一些数据集大多依赖众包模拟共情对话，这一过程既昂贵又耗时，难以扩展到更大的数据集。我们提出了一种数据生成框架，用于开发SYNTHEMPATHY数据集，该数据集包含10.5万条针对现实生活场景的共情回应，通过大规模语言模型生成。基于我们的SYNTHEMPATHY数据集微调的基本Mistral 7B模型平均共情评分有所提高。 

---
# LR${}^{2}$Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems 

**Title (ZH)**: LR${}^{2}$Bench：通过约束满足问题评估大型语言模型的长链反射推理能力 

**Authors**: Jianghao Chen, Zhenlin Wei, Zhenjiang Ren, Ziyong Li, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17848)  

**Abstract**: Recent progress in o1-like models has significantly enhanced the reasoning abilities of Large Language Models (LLMs), empowering them to tackle increasingly complex tasks through reflection capabilities, such as making assumptions, backtracking, and self-refinement. However, effectively evaluating such reflection capabilities remains challenging due to the lack of appropriate benchmarks. To bridge this gap, we introduce LR${}^{2}$Bench, a novel benchmark designed to evaluate the Long-chain Reflective Reasoning capabilities of LLMs. LR${}^{2}$Bench comprises 850 samples across six Constraint Satisfaction Problems (CSPs) where reflective reasoning is crucial for deriving solutions that meet all given constraints. Each type of task focuses on distinct constraint patterns, such as knowledge-based, logical, and spatial constraints, providing a comprehensive evaluation of diverse problem-solving scenarios. We conduct extensive evaluation on both conventional models and o1-like models. Our experimental results reveal that even the most advanced reasoning-specific models, such as DeepSeek-R1 and OpenAI o1-preview, struggle with tasks in LR${}^{2}$Bench, achieving an average Exact Match score of only 20.0% and 23.6%, respectively. These findings underscore the significant room for improvement in the reflective reasoning capabilities of current LLMs. The leaderboard of our benchmark is available at this https URL 

**Abstract (ZH)**: 近年来，o1-like模型的进步显著增强了大型语言模型（LLMs）的推理能力，使它们能够通过反思能力（如假设、回溯和自我修正）来处理越来越复杂的任务。然而，有效地评估这些反思能力仍然具有挑战性，主要是由于缺乏适当的基准。为此，我们引入了LR${}^{2}$Bench，这是一种新的基准，用于评估LLMs的长链反射推理能力。LR${}^{2}$Bench 包含了6种约束满足问题（CSPs）中的850个样本，其中反射推理对于找到满足所有给定约束的解至关重要。每种任务类型都关注不同的约束模式，如基于知识的、逻辑的和空间约束，从而提供了一个全面的多元化问题解决情境的评估。我们在传统模型和o1-like模型上进行了广泛评估。实验结果表明，即使是最先进的专用推理模型，如DeepSeek-R1和OpenAI o1-preview，在LR${}^{2}$Bench上的任务上也难以应对，平均准确匹配得分为20.0%和23.6%。这些结果强调了当前LLMs在反射推理能力方面的巨大改进空间。我们的基准排行榜可在以下链接查看：https://www.alipay.com 

---
# Predicting Through Generation: Why Generation Is Better for Prediction 

**Title (ZH)**: 通过生成进行预测：为什么生成方法更适用于预测 

**Authors**: Md Kowsher, Nusrat Jahan Prottasha, Prakash Bhat, Chun-Nam Yu, Mojtaba Soltanalian, Ivan Garibay, Ozlem Garibay, Chen Chen, Niloofar Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17817)  

**Abstract**: This paper argues that generating output tokens is more effective than using pooled representations for prediction tasks because token-level generation retains more mutual information. Since LLMs are trained on massive text corpora using next-token prediction, generation aligns naturally with their learned behavior. Using the Data Processing Inequality (DPI), we provide both theoretical and empirical evidence supporting this claim. However, autoregressive models face two key challenges when used for prediction: (1) exposure bias, where the model sees ground truth tokens during training but relies on its own predictions during inference, leading to errors, and (2) format mismatch, where discrete tokens do not always align with the tasks required output structure. To address these challenges, we introduce PredGen(Predicting Through Generating), an end to end framework that (i) uses scheduled sampling to reduce exposure bias, and (ii) introduces a task adapter to convert the generated tokens into structured outputs. Additionally, we introduce Writer-Director Alignment Loss (WDAL), which ensures consistency between token generation and final task predictions, improving both text coherence and numerical accuracy. We evaluate PredGen on multiple classification and regression benchmarks. Our results show that PredGen consistently outperforms standard baselines, demonstrating its effectiveness in structured prediction tasks. 

**Abstract (ZH)**: 本文 argument 是，生成输出标记比使用聚合表示进行预测更为有效，因为标记级生成保留了更多的互信息。由于大规模文本语料库中使用下一步标记预测对语言大模型（LLMs）进行了训练，生成自然与其学习行为相一致。通过数据处理不等式（DPI），我们提供了理论和实验证据来支持这一观点。然而，自回归模型在预测任务中面临两个关键挑战：(1) 透明度偏差，模型在训练过程中会看到真实标记，但在推理过程中依赖于其自己的预测，导致错误；(2) 格式不匹配，离散标记并不总是与所需输出结构对齐。为了解决这些挑战，我们提出了 PredGen（通过生成进行预测）这一端到端框架，(i) 使用计划采样来减少透明度偏差，(ii) 引入任务适配器将生成的标记转换为结构化输出。此外，我们提出了写手-导演一致性损失（WDAL），它确保标记生成与最终任务预测之间的一致性，从而提高文本连贯性和数值准确性。我们对多种分类和回归基准进行了 PredGen 的评估。结果表明，PredGen 一致地优于标准基线，证明了其在结构化预测任务中的有效性。 

---
# Can Multimodal LLMs Perform Time Series Anomaly Detection? 

**Title (ZH)**: 多模态LLM能进行时间序列异常检测吗？ 

**Authors**: Xiongxiao Xu, Haoran Wang, Yueqing Liang, Philip S. Yu, Yue Zhao, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17812)  

**Abstract**: Large language models (LLMs) have been increasingly used in time series analysis. However, the potential of multimodal LLMs (MLLMs), particularly vision-language models, for time series remains largely under-explored. One natural way for humans to detect time series anomalies is through visualization and textual description. Motivated by this, we raise a critical and practical research question: Can multimodal LLMs perform time series anomaly detection? To answer this, we propose VisualTimeAnomaly benchmark to evaluate MLLMs in time series anomaly detection (TSAD). Our approach transforms time series numerical data into the image format and feed these images into various MLLMs, including proprietary models (GPT-4o and Gemini-1.5) and open-source models (LLaVA-NeXT and Qwen2-VL), each with one larger and one smaller variant. In total, VisualTimeAnomaly contains 12.4k time series images spanning 3 scenarios and 3 anomaly granularities with 9 anomaly types across 8 MLLMs. Starting with the univariate case (point- and range-wise anomalies), we extend our evaluation to more practical scenarios, including multivariate and irregular time series scenarios, and variate-wise anomalies. Our study reveals several key insights:
1) MLLMs detect range- and variate-wise anomalies more effectively than point-wise anomalies.
2) MLLMs are highly robust to irregular time series, even with 25% of the data missing.
3) Open-source MLLMs perform comparably to proprietary models in TSAD. While open-source MLLMs excel on univariate time series, proprietary MLLMs demonstrate superior effectiveness on multivariate time series.
To the best of our knowledge, this is the first work to comprehensively investigate MLLMs for TSAD, particularly for multivariate and irregular time series scenarios. We release our dataset and code at this https URL to support future research. 

**Abstract (ZH)**: 大语言模型（LLMs）在时间序列分析中的应用越来越广泛。然而，多模态LLMs（MLLMs），尤其是视觉-语言模型，在这方面的潜力仍然被大大低估。人类检测时间序列异常的一个自然方法是通过可视化和文本描述。基于这一点，我们提出一个关键且实际的研究问题：多模态LLMs是否能够进行时间序列异常检测？为回答这一问题，我们提出了VisualTimeAnomaly基准测试，用于评估MLLMs在时间序列异常检测（TSAD）中的性能。我们的方法将时间序列的数据转换为图像格式，并将这些图像输入到各种类型的MLLMs中，包括专有模型（GPT-4o和Gemini-1.5）和开源模型（LLaVA-NeXT和Qwen2-VL），每种模型都有大、小两个变体。总共，VisualTimeAnomaly包含12,400张时间序列图像，涵盖3种情景和3种异常粒度，涉及8种MLLMs，共有9种异常类型。从单变量情况（点、区间异常）开始，我们将评估延伸到更实际的情景，包括多变量和不规则时间序列情况，以及基于变量的异常。我们的研究揭示了几点关键见解：

1）MLLMs在检测区间和基于变量的异常方面比检测点异常更有效。
2）MLLMs对不规则时间序列具有很高的鲁棒性，即使有25%的数据缺失也是如此。
3）开源MLLMs在TSAD中表现与专有模型相当。开源MLLMs在单变量时间序列表现优秀，而专有MLLMs在多变量时间序列上表现出更优的效果。

据我们所知，这是首次全面研究MLLMs在TSAD中的应用，特别是针对多变量和不规则时间序列情景。我们在此处提供我们的数据集和代码以支持未来的研究。您可以在以下链接获取：[链接]。 

---
# Your Language Model May Think Too Rigidly: Achieving Reasoning Consistency with Symmetry-Enhanced Training 

**Title (ZH)**: 你的语言模型可能思考得过于僵化：通过增强对称性训练实现推理一致性 

**Authors**: Yihang Yao, Zhepeng Cen, Miao Li, William Han, Yuyou Zhang, Emerson Liu, Zuxin Liu, Chuang Gan, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.17800)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities across various tasks. However, even minor variations in query phrasing, despite preserving the underlying semantic meaning, can significantly affect their performance. To address this, we focus on enhancing LLMs' awareness of symmetry in query variations and propose syMmetry-ENhanceD (MEND) Data Augmentation, a data-centric approach that improves the model's ability to extract useful information from context. Unlike existing methods that emphasize reasoning chain augmentation, our approach improves model robustness at the knowledge extraction stage through query augmentations, enabling more data-efficient training and stronger generalization to Out-of-Distribution (OOD) settings. Extensive experiments on both logical and arithmetic reasoning tasks show that MEND enhances reasoning performance across diverse query variations, providing new insight into improving LLM robustness through structured dataset curation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中展现了强大的推理能力。然而，即使查询措辞存在微小变化，只要保持背后的语义不变，其性能也可能受到显著影响。为解决这一问题，本文专注于提升LLMs对查询变化中对称性的认识，并提出了基于数据的增强方法syMmetry-ENhanceD (MEND) 数据增强技术。MEND 数据增强技术通过查询增强提高模型从上下文中提取有用信息的能力，是一种以数据为中心的方法，不同于现有侧重推理链增强的方法，本文的方法在知识提取阶段提高了模型的鲁棒性，从而实现了更高效的训练和更强大的跨分布（OOD）情况下的泛化能力。通过在逻辑和算术推理任务上的广泛实验，结果表明MEND能够在各种查询变化中提升推理性能，为通过结构化数据集管理提升LLM鲁棒性提供了新的见解。 

---
# AIR: Complex Instruction Generation via Automatic Iterative Refinement 

**Title (ZH)**: AIR：通过自动迭代 refinement 的复杂指令生成 

**Authors**: Wei Liu, Yancheng He, Hui Huang, Chengwei Hu, Jiaheng Liu, Shilong Li, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17787)  

**Abstract**: With the development of large language models, their ability to follow simple instructions has significantly improved. However, adhering to complex instructions remains a major challenge. Current approaches to generating complex instructions are often irrelevant to the current instruction requirements or suffer from limited scalability and diversity. Moreover, methods such as back-translation, while effective for simple instruction generation, fail to leverage the rich contents and structures in large web corpora. In this paper, we propose a novel automatic iterative refinement framework to generate complex instructions with constraints, which not only better reflects the requirements of real scenarios but also significantly enhances LLMs' ability to follow complex instructions. The AIR framework consists of two stages: (1)Generate an initial instruction from a document; (2)Iteratively refine instructions with LLM-as-judge guidance by comparing the model's output with the document to incorporate valuable constraints. Finally, we construct the AIR-10K dataset with 10K complex instructions and demonstrate that instructions generated with our approach significantly improve the model's ability to follow complex instructions, outperforming existing methods for instruction generation. 

**Abstract (ZH)**: 随着大规模语言模型的发展，它们遵循简单指令的能力显著提高。然而，遵循复杂指令仍然是一个主要挑战。当前生成复杂指令的方法往往与当前的指令需求无关，或者受到有限的可扩展性和多样性的限制。此外，如回译等方法尽管对生成简单指令有效，但未能充分利用大型网页语料库中的丰富内容和结构。在本文中，我们提出了一种新颖的自动迭代 refinement 框架，以生成具有约束的复杂指令，这不仅更准确地反映了真实场景的需求，还显著增强了大语言模型遵循复杂指令的能力。AIR 框架包括两个阶段：（1）从文档中生成初始指令；（2）通过将模型输出与文档进行比较，在大语言模型（LLM）作为裁判的指导下，迭代优化指令，以纳入有价值的信息约束。最后，我们构建了包含10,000条复杂指令的AIR-10K数据集，并证明使用我们方法生成的指令显著提升了模型遵循复杂指令的能力，优于现有的指令生成方法。 

---
# Exploring the Potential of Large Language Models for Estimating the Reading Comprehension Question Difficulty 

**Title (ZH)**: 探索大型语言模型在估计阅读理解问题难度方面的潜力 

**Authors**: Yoshee Jain, John Hollander, Amber He, Sunny Tang, Liang Zhang, John Sabatini  

**Link**: [PDF](https://arxiv.org/pdf/2502.17785)  

**Abstract**: Reading comprehension is a key for individual success, yet the assessment of question difficulty remains challenging due to the extensive human annotation and large-scale testing required by traditional methods such as linguistic analysis and Item Response Theory (IRT). While these robust approaches provide valuable insights, their scalability is limited. There is potential for Large Language Models (LLMs) to automate question difficulty estimation; however, this area remains underexplored. Our study investigates the effectiveness of LLMs, specifically OpenAI's GPT-4o and o1, in estimating the difficulty of reading comprehension questions using the Study Aid and Reading Assessment (SARA) dataset. We evaluated both the accuracy of the models in answering comprehension questions and their ability to classify difficulty levels as defined by IRT. The results indicate that, while the models yield difficulty estimates that align meaningfully with derived IRT parameters, there are notable differences in their sensitivity to extreme item characteristics. These findings suggest that LLMs can serve as the scalable method for automated difficulty assessment, particularly in dynamic interactions between learners and Adaptive Instructional Systems (AIS), bridging the gap between traditional psychometric techniques and modern AIS for reading comprehension and paving the way for more adaptive and personalized educational assessments. 

**Abstract (ZH)**: 阅读理解是个人成功的关键，但由于传统方法（如语言分析和项目反应理论IRT）所需的大量人工注释和大规模测试，评估问题难度仍然具有挑战性。虽然这些稳健的方法提供了有价值的洞见，但它们的可扩展性受到限制。大规模语言模型（LLMs）有可能自动化问题难度估计，然而这一领域仍然还未被充分探索。我们的研究探讨了LLMs，特别是OpenAI的GPT-4o和GPT-3.5，在使用Study Aid and Reading Assessment (SARA)数据集估计阅读理解问题难度方面的有效性。我们评估了模型在回答阅读理解问题方面的准确性，以及它们在分类由IRT定义的难度等级方面的能力。研究结果表明，尽管模型生成的难度估计值与由IRT推导出的参数在意义上存在一定的契合性，但在极端项目特征敏感性方面存在显著差异。这些发现表明，LLMs可以作为自动化难度评估的可扩展方法，特别是在学习者与自适应教学系统（AIS）之间的动态交互中发挥作用，从而弥合传统的心理测量技术与现代AIS之间在阅读理解方面的差距，为更适应性和个性化的教育教学评估铺平道路。 

---
# LLM Inference Acceleration via Efficient Operation Fusion 

**Title (ZH)**: 通过高效操作融合加速大语言模型推理 

**Authors**: Mahsa Salmani, Ilya Soloveychik  

**Link**: [PDF](https://arxiv.org/pdf/2502.17728)  

**Abstract**: The rapid development of the Transformer-based Large Language Models (LLMs) in recent years has been closely linked to their ever-growing and already enormous sizes. Many LLMs contain hundreds of billions of parameters and require dedicated hardware resources for training and inference. One of the key challenges inherent to the Transformer architecture is the requirement to support numerous non-linear transformations that involves normalization. For instance, each decoder block typically contains at least one Softmax operation and two Layernorms. The computation of the corresponding normalization scaling factors becomes a major bottleneck as it requires spatial collective operations. In other words, when it comes to the computation of denominators for Softmax and Layernorm, all vector elements must be aggregated into a single location, requiring significant communication. These collective operations slow down inference on Transformers by approximately 20%, defeating the whole purpose of distributed in-memory compute. In this work, we propose an extremely efficient technique that can completely hide the overhead caused by such collective operations. Note that each Softmax and Layernorm operation is typically followed by a linear layer. Since non-linear and linear operations are performed on different hardware engines, they can be easily parallelized once the algebra allows such commutation. By leveraging the inherent properties of linear operations, we can defer the normalization of the preceding Softmax and Layernorm until after the linear layer is computed. Now we can compute the collective scaling factors concurrently with the matrix multiplication and completely hide the latency of the former behind the latter. Such parallelization preserves the numerical accuracy while significantly improving the hardware utilization and reducing the overall latency. 

**Abstract (ZH)**: 近年来，基于变换器的大型语言模型（LLMs）的快速发展与其日益庞大且已相当巨大的规模密切相关。许多LLM包含数百亿个参数，并需要专用的硬件资源进行训练和推理。变换器架构固有的一个关键挑战是支持大量的非线性变换，特别是需要执行归一化操作。例如，每个解码器块通常至少包含一个Softmax操作和两个LayerNorm。计算相应的归一化比例因子成为了一个主要瓶颈，因为这需要进行空间上的聚合操作。换句话说，当进行Softmax和LayerNorm的分母计算时，所有向量元素必须汇总到一个位置，这需要显着的通信开销。这些聚合操作会将变换器的推理速度减慢约20%，违背了分布式内存计算的初衷。在本工作中，我们提出了一种极其高效的方法，可以完全隐藏由这些聚合操作引起的开销。需要注意的是，每个Softmax和LayerNorm操作通常会接在一层线性操作之后。由于非线性和线性操作可以在不同的硬件引擎上执行，一旦代数允许交换顺序，它们可以很容易地并行化。通过利用线性操作的固有特性，我们可以在计算线性层之前延迟执行前面的Softmax和LayerNorm的归一化操作。现在，我们可以在矩阵乘法的同时并行计算集体归一化因子，并将前者的延迟完全隐藏在后者之后。这种并行化不仅保持了数值精度，还显著提高了硬件利用率，减少了整体延时。 

---
# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution 

**Title (ZH)**: SWE-RL：通过开放软件演化中的强化学习提升大规模语言模型的推理能力 

**Authors**: Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida I. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18449)  

**Abstract**: The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data. 

**Abstract (ZH)**: 最近发布的DeepSeek-R1表明，强化学习（RL）在提升大型语言模型（LLMs）的通用推理能力方面具有巨大潜力。虽然DeepSeek-R1及其后续工作主要集中在将RL应用于编程竞赛和数学问题上，本论文介绍了SWE-RL，这是首个用于实世界软件工程的基于RL的大规模LLM推理方法。通过利用一种轻量级基于规则的奖励（例如，正确解与LLM生成解之间的相似性得分），SWE-RL使LLMs能够通过学习广泛开源软件演化数据自主恢复开发者的推理过程和解决方案，这些数据记录了软件生命周期的整个过程，包括代码快照、代码变更以及如问题和拉取请求等事件。以Llama 3为基础训练，我们得到的推理模型Llama3-SWE-RL-70B在SWE-bench Verified（一个由人工验证的真实GitHub问题集）上实现了41.0%的解题率。据我们所知，这是迄今为止中型（<100B）LLMs的最佳性能，甚至可以媲美如GPT-4o等主流的专有LLMs。令人惊讶的是，尽管仅在软件演化数据上进行RL训练，但Llama3-SWE-RL甚至展现出通用推理能力。例如，它在五个离域任务（函数编码、库使用、代码推理、数学和通用语言理解）上取得了更好的结果，而监督微调基准方法反而在平均性能上下降了。总体而言，SWE-RL为通过大规模软件工程数据的强化学习来提升LLMs的推理能力开辟了新的方向。 

---
# Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Support 

**Title (ZH)**: 柑橘：利用专家认知路径增强医疗语言模型的高级医疗决策支持 

**Authors**: Guoxin Wang, Minyu Gao, Shuai Yang, Ya Zhang, Lizhi He, Liang Huang, Hanlin Xiao, Yexuan Zhang, Wanyue Li, Lu Chen, Jintao Fei, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18274)  

**Abstract**: Large language models (LLMs), particularly those with reasoning capabilities, have rapidly advanced in recent years, demonstrating significant potential across a wide range of applications. However, their deployment in healthcare, especially in disease reasoning tasks, is hindered by the challenge of acquiring expert-level cognitive data. In this paper, we introduce Citrus, a medical language model that bridges the gap between clinical expertise and AI reasoning by emulating the cognitive processes of medical experts. The model is trained on a large corpus of simulated expert disease reasoning data, synthesized using a novel approach that accurately captures the decision-making pathways of clinicians. This approach enables Citrus to better simulate the complex reasoning processes involved in diagnosing and treating medical this http URL further address the lack of publicly available datasets for medical reasoning tasks, we release the last-stage training data, including a custom-built medical diagnostic dialogue dataset. This open-source contribution aims to support further research and development in the field. Evaluations using authoritative benchmarks such as MedQA, covering tasks in medical reasoning and language understanding, show that Citrus achieves superior performance compared to other models of similar size. These results highlight Citrus potential to significantly enhance medical decision support systems, providing a more accurate and efficient tool for clinical decision-making. 

**Abstract (ZH)**: 近年来，尤其是具备推理能力的大规模语言模型（LLMs）取得了迅速发展，展示了在各种应用领域中的巨大潜力。然而，这些模型在医疗领域的部署，尤其是在疾病推理任务中的应用，受到了获取专家级认知数据的挑战。本文介绍了Citrus，这是一种医疗语言模型，通过模拟医学专家的认知过程，弥合了临床专业知识与人工智能推理之间的差距。该模型是基于大量模拟的专家级疾病推理数据集进行训练的，这些数据是通过一种新颖的方法合成的，该方法能够准确捕捉临床决策的路径。这种方法使Citrus能够更好地模拟诊断和治疗过程中的复杂推理机制。

为进一步解决医学推理任务中缺乏公共数据集的问题，我们公开了模型的最后一阶段训练数据，包括一个自定义构建的医疗诊断对话数据集。这一开放源代码的贡献旨在支持该领域进一步的研究和开发。使用MedQA等权威基准进行的评估涵盖了医学推理和语言理解的任务，结果显示，Citrus在性能上优于其他同类型规模的模型。这些结果突显了Citrus在增强医疗决策支持系统方面的巨大潜力，可以为临床决策提供更加准确和高效的工具。 

---
# Iterative Counterfactual Data Augmentation 

**Title (ZH)**: 迭代反事实数据增强 

**Authors**: Mitchell Plyler, Min Chi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18249)  

**Abstract**: Counterfactual data augmentation (CDA) is a method for controlling information or biases in training datasets by generating a complementary dataset with typically opposing biases. Prior work often either relies on hand-crafted rules or algorithmic CDA methods which can leave unwanted information in the augmented dataset. In this work, we show iterative CDA (ICDA) with initial, high-noise interventions can converge to a state with significantly lower noise. Our ICDA procedure produces a dataset where one target signal in the training dataset maintains high mutual information with a corresponding label and the information of spurious signals are reduced. We show training on the augmented datasets produces rationales on documents that better align with human annotation. Our experiments include six human produced datasets and two large-language model generated datasets. 

**Abstract (ZH)**: 对抗事实数据增强（CDA）是一种通过生成具有通常相反偏见的互补数据集来控制训练数据集中信息或偏见的方法。以往的工作往往依赖于手工制定的规则或算法性CDA方法，这可能会在增强的数据集中留下不必要的信息。在本工作中，我们展示了迭代CDA（ICDA）方法与初始的高噪声干预可以收敛到噪声显著降低的状态。我们的ICDA流程生成的数据集，在训练数据集中有一个目标信号与相应的标签保持高互信息，同时虚假信号的信息量有所减少。我们证明，在增强的数据集上进行训练可以产生与人类标注更好地对齐的理由。我们的实验包括六个由人类制作的数据集和两个大型语言模型生成的数据集。 

---
# HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge 

**Title (ZH)**: HyperG：基于超图增强的大语言模型处理结构化知识 

**Authors**: Sirui Huang, Hanqian Li, Yanggan Gu, Xuming Hu, Qing Li, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18125)  

**Abstract**: Given that substantial amounts of domain-specific knowledge are stored in structured formats, such as web data organized through HTML, Large Language Models (LLMs) are expected to fully comprehend this structured information to broaden their applications in various real-world downstream tasks. Current approaches for applying LLMs to structured data fall into two main categories: serialization-based and operation-based methods. Both approaches, whether relying on serialization or using SQL-like operations as an intermediary, encounter difficulties in fully capturing structural relationships and effectively handling sparse data. To address these unique characteristics of structured data, we propose HyperG, a hypergraph-based generation framework aimed at enhancing LLMs' ability to process structured knowledge. Specifically, HyperG first augment sparse data with contextual information, leveraging the generative power of LLMs, and incorporate a prompt-attentive hypergraph learning (PHL) network to encode both the augmented information and the intricate structural relationships within the data. To validate the effectiveness and generalization of HyperG, we conduct extensive experiments across two different downstream tasks requiring structured knowledge. 

**Abstract (ZH)**: 鉴于大量特定领域的知识以结构化格式存储，例如通过HTML组织的Web数据，大型语言模型（LLMs）预计将全面理解这些结构化信息，从而在各种现实世界的下游任务中得到更广泛的应用。目前将LLMs应用到结构化数据中的方法主要分为两类：序列化方法和操作方法。不论是依赖于序列化方法还是通过SQL样式的操作作为中介，这两种方法都难以全面捕捉结构关系并有效地处理稀疏数据。为应对结构化数据的独特特征，我们提出了一种基于超图的生成框架HyperG，旨在增强LLMs处理结构化知识的能力。具体而言，HyperG首先通过上下文信息增强稀疏数据，利用LLMs的生成能力，并结合一个提示感知超图学习（PHL）网络来编码增强的信息以及数据中的复杂结构关系。为了验证HyperG的有效性和泛化能力，我们在两个不同的需要结构化知识的下游任务中进行了广泛的实验。 

---
# LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena 

**Title (ZH)**: LLM 在几何知识上优于代数：基于 LLM 的代理在交易竞技场中的数值理解 

**Authors**: Tianmi Ma, Jiawei Du, Wenxin Huang, Wenjie Wang, Liang Xie, Xian Zhong, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17967)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance in natural language processing tasks. However, their ability to generalize to dynamic, unseen tasks, particularly in numerical reasoning, remains a challenge. Existing benchmarks mainly evaluate LLMs on problems with predefined optimal solutions, which may not align with real-world scenarios where clear answers are absent. To bridge this gap, we design the Agent Trading Arena, a virtual numerical game simulating complex economic systems through zero-sum games, where agents invest in stock portfolios. Our experiments reveal that LLMs, including GPT-4o, struggle with algebraic reasoning when dealing with plain-text stock data, often focusing on local details rather than global trends. In contrast, LLMs perform significantly better with geometric reasoning when presented with visual data, such as scatter plots or K-line charts, suggesting that visual representations enhance numerical reasoning. This capability is further improved by incorporating the reflection module, which aids in the analysis and interpretation of complex data. We validate our findings on NASDAQ Stock dataset, where LLMs demonstrate stronger reasoning with visual data compared to text. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在自然语言处理任务中的性能显著提升。然而，它们在动态、未见过的任务中的泛化能力，尤其是在数值推理方面，仍然存在挑战。现有基准测试主要评估LLMs在具有预定义最优解的问题上的表现，这可能与现实世界中缺乏明确答案的场景不一致。为解决这一问题，我们设计了Agent Trading Arena，这是一种虚拟数值游戏，通过零和博弈模拟复杂经济系统，其中的代理投资于股票组合。我们的实验表明，包括GPT-4o在内的LLMs在处理平文本股票数据时，在代数推理方面表现困难，往往关注局部细节而非整体趋势。相比之下，当提供可视化数据（如散点图或K线图）时，LLMs在几何推理方面的表现显著更好，这表明可视化表示能够增强数值推理能力。通过引入反射模块，这一能力得到进一步提升，该模块有助于复杂数据的分析和解释。我们在纳斯达克股票数据集上验证了上述发现，结果显示，与文本相比，LLMs在可视化数据上的推理能力更强。我们的代码和数据已在以下网址公开：[请在此处填写网址]。 

---
# Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers 

**Title (ZH)**: 跨语言的科学交流：评估大语言模型在翻译科学论文方面的多语言能力 

**Authors**: Hannah Calzi Kleidermacher, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17882)  

**Abstract**: Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at this https URL. 

**Abstract (ZH)**: 科学研究本质上是全球性的。然而，绝大多数学术期刊仅以英语出版，这为非英语母语的研究者设定了障碍。在本研究中，我们利用大规模语言模型（LLMs）来翻译已发表的科学文章，并保留其原始的JATS XML格式，在此过程中开发了一种实用的自动化方法，供学术期刊实施。使用我们的方法，我们将多学科的学术论文翻译成28种语言。为了评估翻译准确性，我们引入了一种新的问答（QA）基准测试方法，在这种方法中，一个LLM从原始文本生成基于理解的问题，然后基于翻译文本回答这些问题。我们的基准测试结果显示出95.9%的平均性能，表明关键的科学细节得到了准确传达。在一项用户研究中，我们将15位研究人员的科学论文翻译成他们的母语，发现作者一致认为这些翻译准确地捕捉到了原文中的信息。有趣的是，三分之一的作者发现许多技术术语被“过度翻译”，他们更倾向于保留这些术语的英语表达。最后，我们展示了如何利用上下文学习技术来调整翻译，以适应特定领域的偏好，如减少过度翻译，这证明了基于LLM的科学翻译的可适应性和实用性。代码和翻译文章可在以下网址获取：[this https URL]。 

---
# Tip of the Tongue Query Elicitation for Simulated Evaluation 

**Title (ZH)**: 舌尖上的查询诱致方法在模拟评估中的应用 

**Authors**: Yifan He, To Eun Kim, Fernando Diaz, Jaime Arguello, Bhaskar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2502.17776)  

**Abstract**: Tip-of-the-tongue (TOT) search occurs when a user struggles to recall a specific identifier, such as a document title. While common, existing search systems often fail to effectively support TOT scenarios. Research on TOT retrieval is further constrained by the challenge of collecting queries, as current approaches rely heavily on community question-answering (CQA) websites, leading to labor-intensive evaluation and domain bias. To overcome these limitations, we introduce two methods for eliciting TOT queries - leveraging large language models (LLMs) and human participants - to facilitate simulated evaluations of TOT retrieval systems. Our LLM-based TOT user simulator generates synthetic TOT queries at scale, achieving high correlations with how CQA-based TOT queries rank TOT retrieval systems when tested in the Movie domain. Additionally, these synthetic queries exhibit high linguistic similarity to CQA-derived queries. For human-elicited queries, we developed an interface that uses visual stimuli to place participants in a TOT state, enabling the collection of natural queries. In the Movie domain, system rank correlation and linguistic similarity analyses confirm that human-elicited queries are both effective and closely resemble CQA-based queries. These approaches reduce reliance on CQA-based data collection while expanding coverage to underrepresented domains, such as Landmark and Person. LLM-elicited queries for the Movie, Landmark, and Person domains have been released as test queries in the TREC 2024 TOT track, with human-elicited queries scheduled for inclusion in the TREC 2025 TOT track. Additionally, we provide source code for synthetic query generation and the human query collection interface, along with curated visual stimuli used for eliciting TOT queries. 

**Abstract (ZH)**: 舌尖效应（Tip-of-the-tongue, TOT）搜索发生在用户努力回忆特定标识符，如文档标题时。虽然TOT搜索是常见的，但现有的搜索系统往往难以有效支持TOT场景。由于当前收集查询的方法主要依赖社区问答（CQA）网站，这限制了研究并导致了劳动密集型的评估和领域偏差。为克服这些限制，我们提出了两种方法来 eliciting TOT 查询——利用大规模语言模型（LLMs）和人类参与者——以促进TOT检索系统的模拟评估。我们基于LLM的TOT用户模拟器大规模生成合成的TOT查询，当在电影领域测试时，这些合成查询与基于CQA的TOT查询对TOT检索系统进行排名的相关性非常高。此外，这些合成查询在语言学上与CQA衍生查询高度相似。对于人类引发的查询，我们开发了一个界面，使用视觉刺激来使参与者处于TOT状态，从而收集自然查询。在电影领域，系统排名相关性和语言学相似性分析证实，人类引发的查询既有效又与CQA衍生查询高度相似。这些方法减少了对基于CQA的数据收集的依赖，同时扩大了覆盖范围，包括较少被代表的领域，如地标（Landmark）和人物（Person）。电影、地标和人物领域的LLM引发的查询已作为测试查询发布，人类引发的查询将在TREC 2025 TOT 任务中增加。此外，我们还提供了生成合成查询和人类查询收集界面的源代码，并提供了用于引发TOT查询的精选视觉刺激。 

---
# From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs 

**Title (ZH)**: 从感知到决策：基于行为理论指导的大语言模型 wildfire 趋避决策预测 

**Authors**: Ruxiao Chen, Chenguang Wang, Yuran Sun, Xilei Zhao, Susu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17701)  

**Abstract**: Evacuation decision prediction is critical for efficient and effective wildfire response by helping emergency management anticipate traffic congestion and bottlenecks, allocate resources, and minimize negative impacts. Traditional statistical methods for evacuation decision prediction fail to capture the complex and diverse behavioral logic of different individuals. In this work, for the first time, we introduce FLARE, short for facilitating LLM for advanced reasoning on wildfire evacuation decision prediction, a Large Language Model (LLM)-based framework that integrates behavioral theories and models to streamline the Chain-of-Thought (CoT) reasoning and subsequently integrate with memory-based Reinforcement Learning (RL) module to provide accurate evacuation decision prediction and understanding. Our proposed method addresses the limitations of using existing LLMs for evacuation behavioral predictions, such as limited survey data, mismatching with behavioral theory, conflicting individual preferences, implicit and complex mental states, and intractable mental state-behavior mapping. Experiments on three post-wildfire survey datasets show an average of 20.47% performance improvement over traditional theory-informed behavioral models, with strong cross-event generalizability. Our complete code is publicly available at this https URL 

**Abstract (ZH)**: 有效的疏散决策预测对于提高野火应对的效率和效果至关重要，它可以帮助应急管理机构预见交通拥堵和瓶颈问题，合理分配资源，并尽量减少负面影响。传统的统计方法在预测疏散决策时，难以捕捉不同个体复杂的多样行为逻辑。在此项研究中，我们首次引入了一个名为“FLARE”的框架，即“促进大型语言模型在野火疏散决策预测中的高级推理”，该框架以大型语言模型（LLM）为基础，整合了行为理论和模型，简化了链式推理（CoT），并在此基础上结合基于记忆的强化学习（RL）模块，提供准确的疏散决策预测和理解。我们提出的方法解决了现有大型语言模型在疏散行为预测中的局限性，如有限的调查数据、不符合行为理论、个体偏好冲突、复杂的心理状态以及难以解决的心理状态-行为映射问题。在三个野火后调查数据集上的实验显示，与传统的以理论为基础的行为模型相比，平均性能提高了20.47%，并且具有较强的跨事件通用性。完整的代码已公开，可在此处访问：this https URL 

---
# The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve? 

**Title (ZH)**: 彩票LLM假设：重新思考压缩LLM时应保留的能力？ 

**Authors**: Zhenheng Tang, Xiang Liu, Qian Wang, Peijie Dong, Bingsheng He, Xiaowen Chu, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17535)  

**Abstract**: Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attention from researchers. However, current methods predominantly emphasize maintaining the performance of compressed LLMs, as measured by perplexity or simple accuracy on tasks of common sense knowledge QA and basic arithmetic reasoning. In this blog, we present a brief review of recent advancements in LLMs related to retrieval-augmented generation, multi-step reasoning, external tools, and computational expressivity, all of which substantially enhance LLM performance. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance as the original LLM with the assistance of multi-step reasoning and external tools. Based on the review of current progress in LLMs, we discuss and summarize the essential capabilities that the lottery LLM and KV cache compression must possess, which are currently overlooked in existing methods. 

**Abstract (ZH)**: 为了减少大语言模型（LLM）的计算和存储成本，模型压缩和键值（KV）缓存压缩已经引起了研究人员的广泛关注。然而，当前的方法大多侧重于保持压缩后的LLM的性能，这些性能通常通过困惑度或常识问答和基本算术推理任务中的简单准确性来衡量。在本文中，我们将简要回顾与检索增强生成、多步推理、外部工具以及计算表达能力相关的LLM近期进展，这些进展显著提升了LLM的性能。在此基础上，我们提出了一种“彩票LLM”假说，即对于给定的LLM和任务，存在一个较小的“彩票LLM”，在借助多步推理和外部工具的协助下，也能产生与原始LLM相同的性能。基于对当前LLM进展的回顾，我们讨论并总结了“彩票LLM”和KV缓存压缩必须具备的核心能力，这些都是现有方法目前所忽视的。 

---
# SAE-V: Interpreting Multimodal Models for Enhanced Alignment 

**Title (ZH)**: SAE-V：解析多模态模型以增强对齐 

**Authors**: Hantao Lou, Changye Li, Jiaming Ji, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17514)  

**Abstract**: With the integration of image modality, the semantic space of multimodal large language models (MLLMs) is more complex than text-only models, making their interpretability more challenging and their alignment less stable, particularly susceptible to low-quality data, which can lead to inconsistencies between modalities, hallucinations, and biased outputs. As a result, developing interpretability methods for MLLMs is crucial for improving alignment quality and efficiency. In text-only LLMs, Sparse Autoencoders (SAEs) have gained attention for their ability to interpret latent representations. However, extending SAEs to multimodal settings presents new challenges due to modality fusion and the difficulty of isolating cross-modal representations. To address these challenges, we introduce SAE-V, a mechanistic interpretability framework that extends the SAE paradigm to MLLMs. By identifying and analyzing interpretable features along with their corresponding data, SAE-V enables fine-grained interpretation of both model behavior and data quality, facilitating a deeper understanding of cross-modal interactions and alignment dynamics. Moreover, by utilizing cross-modal feature weighting, SAE-V provides an intrinsic data filtering mechanism to enhance model alignment without requiring additional models. Specifically, when applied to the alignment process of MLLMs, SAE-V-based data filtering methods could achieve more than 110% performance with less than 50% data. Our results highlight SAE-V's ability to enhance interpretability and alignment in MLLMs, providing insights into their internal mechanisms. 

**Abstract (ZH)**: 随着图像模态的整合，多模态大型语言模型（MLLMs）的语义空间比仅文本模型更加复杂，这使得它们的可解释性更加困难，对齐性也更加不稳定，尤其容易受到低质量数据的影响，从而导致模态之间的一致性问题、幻觉和偏差输出。因此，开发MLLMs的可解释性方法对于提高对齐质量和效率至关重要。在仅文本的大规模语言模型（LLMs）中，稀疏自编码器（SAEs）因其能够解释隐含表示而受到关注。然而，将SAEs扩展到多模态环境带来了新的挑战，因为需要处理模态融合以及跨模态表示的隔离难度。为了解决这些挑战，我们提出了一种基于SAE的机制性可解释框架SAE-V，该框架将SAE范式扩展到MLLMs中。通过识别和分析具有对应数据的可解释特征，SAE-V使我们能够对模型行为和数据质量进行精细化解释，从而促进对跨模态交互和对齐动力学的深入理解。此外，通过利用跨模态特征加权，SAE-V提供了一种内在的数据过滤机制，以增强模型对齐，而无需额外的模型。具体而言，当SAE-V应用于MLLMs的对齐过程时，基于SAE-V的数据过滤方法可以在少于50%的数据下实现超过110%的性能提升。我们的结果突出展示了SAE-V在增强MLLMs的可解释性和对齐性方面的潜力，为理解其内部机制提供了见解。 

---
# Recurrent Knowledge Identification and Fusion for Language Model Continual Learning 

**Title (ZH)**: 循环知识识别与融合在语言模型持续学习中的应用 

**Authors**: Yujie Feng, Xujia Wang, Zexin Lu, Shenghong Fu, Guangyuan Shi, Yongxin Xu, Yasha Wang, Philip S. Yu, Xu Chu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17510)  

**Abstract**: Continual learning (CL) is crucial for deploying large language models (LLMs) in dynamic real-world environments without costly retraining. While recent model ensemble and model merging methods guided by parameter importance have gained popularity, they often struggle to balance knowledge transfer and forgetting, mainly due to the reliance on static importance estimates during sequential training. In this paper, we present Recurrent-KIF, a novel CL framework for Recurrent Knowledge Identification and Fusion, which enables dynamic estimation of parameter importance distributions to enhance knowledge transfer. Inspired by human continual learning, Recurrent-KIF employs an inner loop that rapidly adapts to new tasks while identifying important parameters, coupled with an outer loop that globally manages the fusion of new and historical knowledge through redundant knowledge pruning and key knowledge merging. These inner-outer loops iteratively perform multiple rounds of fusion, allowing Recurrent-KIF to leverage intermediate training information and adaptively adjust fusion strategies based on evolving importance distributions. Extensive experiments on two CL benchmarks with various model sizes (from 770M to 13B) demonstrate that Recurrent-KIF effectively mitigates catastrophic forgetting and enhances knowledge transfer. 

**Abstract (ZH)**: 持续学习（CL）对于在动态现实环境中部署大型语言模型（LLMs）至关重要，而无需昂贵的重新训练。尽管最近由参数重要性指导的模型集成和模型融合方法越来越受欢迎，但在顺序训练过程中依赖静态重要性估计往往难以平衡知识转移和遗忘。本文提出了Recurrent-KIF（循环知识识别与融合框架），这是一种新颖的持续学习框架，能够动态估计参数重要性分布以增强知识转移。受人类持续学习的启发，Recurrent-KIF 结合了一个内部循环，可以快速适应新任务并识别重要参数，以及一个外部循环，通过冗余知识修剪和关键知识合并来全局管理新旧知识的融合。这些内部-外部循环迭代进行多次融合，使Recurrent-KIF能够利用中间训练信息，并根据不断变化的重要性分布自适应调整融合策略。在两个持续学习基准测试中使用不同规模的模型（从770M到13B）进行了广泛实验，结果表明Recurrent-KIF有效地减轻了灾难性遗忘，并增强了知识转移。 

---
# MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning 

**Title (ZH)**: MAPoRL：基于强化学习的多智能体共训练协作大语言模型 

**Authors**: Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman Ozdaglar, Kaiqing Zhang, Joo-Kyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18439)  

**Abstract**: Leveraging multiple large language models (LLMs) to build collaborative multi-agentic workflows has demonstrated significant potential. However, most previous studies focus on prompting the out-of-the-box LLMs, relying on their innate capability for collaboration, which may not improve LLMs' performance as shown recently. In this paper, we introduce a new post-training paradigm MAPoRL (Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning), to explicitly elicit the collaborative behaviors and further unleash the power of multi-agentic LLM frameworks. In MAPoRL, multiple LLMs first generate their own responses independently and engage in a multi-turn discussion to collaboratively improve the final answer. In the end, a MAPoRL verifier evaluates both the answer and the discussion, by assigning a score that verifies the correctness of the answer, while adding incentives to encourage corrective and persuasive discussions. The score serves as the co-training reward, and is then maximized through multi-agent RL. Unlike existing LLM post-training paradigms, MAPoRL advocates the co-training of multiple LLMs together using RL for better generalization. Accompanied by analytical insights, our experiments demonstrate that training individual LLMs alone is insufficient to induce effective collaboration. In contrast, multi-agent co-training can boost the collaboration performance across benchmarks, with generalization to unseen domains. 

**Abstract (ZH)**: 利用多个大规模语言模型（LLMs）构建协作多智能体工作流展现出显著潜力。然而，大多数前期研究集中在通过提示现成的LLMs来激发其协作能力上，依赖它们固有的协作能力，这可能未如最近研究所示那样提高LLMs的表现。本文介绍了一种新的后训练范式MAPoRL（Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning），以明确地引发协作行为，并进一步释放多智能体LLM框架的能力。在MAPoRL中，多个LLMs首先独立生成自己的回应，并通过多轮讨论协作改进最终答案。最后，一个MAPoRL验证器评估答案和讨论，通过分配证明答案正确性的分数，同时鼓励正确的和有说服力的讨论，从而激励讨论。该分数作为共训练奖励，并通过多智能体强化学习进行最大化。与现有的LLM后训练范式不同，MAPoRL提倡使用强化学习共同训练多个LLMs以提高泛化能力。我们的实验配以分析洞察，证明单独训练单个LLM不足以引发有效的协作。相反，多智能体共同训练可以在基准测试上提升协作性能，并扩展到未见过的领域。 

---
# How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities 

**Title (ZH)**: 大语言模型与实际搜索有多大差距？关于效率、完整性和固有能力的全面研究 

**Authors**: Minhua Lin, Hui Liu, Xianfeng Tang, Jingying Zeng, Zhenwei Dai, Chen Luo, Zheng Li, Xiang Zhang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18387)  

**Abstract**: Search plays a fundamental role in problem-solving across various domains, with most real-world decision-making problems being solvable through systematic search. Drawing inspiration from recent discussions on search and learning, we systematically explore the complementary relationship between search and Large Language Models (LLMs) from three perspectives. First, we analyze how learning can enhance search efficiency and propose Search via Learning (SeaL), a framework that leverages LLMs for effective and efficient search. Second, we further extend SeaL to SeaL-C to ensure rigorous completeness during search. Our evaluation across three real-world planning tasks demonstrates that SeaL achieves near-perfect accuracy while reducing search spaces by up to 99.1% compared to traditional approaches. Finally, we explore how far LLMs are from real search by investigating whether they can develop search capabilities independently. Our analysis reveals that while current LLMs struggle with efficient search in complex problems, incorporating systematic search strategies significantly enhances their problem-solving capabilities. These findings not only validate the effectiveness of our approach but also highlight the need for improving LLMs' search abilities for real-world applications. 

**Abstract (ZH)**: 搜索在各个领域的问题解决中扮演着基础角色，大多数现实世界中的决策问题都可以通过系统的搜索方法来解决。受到近期关于搜索与学习讨论的启发，我们从三个方面系统地探讨了搜索和大型语言模型（LLMs）之间的互补关系。首先，我们分析了学习如何提高搜索效率，并提出了一种名为“搜索通过学习”（SeaL）的框架，该框架利用LLMs进行有效的搜索。其次，我们将SeaL扩展为SeaL-C，以确保在搜索过程中实现严格的完备性。我们在三个实际规划任务上的评估显示，与传统方法相比，SeaL能够将搜索空间减少高达99.1%，同时保持接近完美的准确性。最后，我们探索了LLMs在现实搜索中的局限性，研究它们是否能够独立发展出搜索能力。我们的分析表明，尽管当前的LLMs在解决复杂问题时难以进行高效的搜索，但融入系统的搜索策略极大地增强了它们的问题解决能力。这些发现不仅验证了我们方法的有效性，还突显了改善LLMs的搜索能力以适应实际应用的必要性。 

---
# LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction 

**Title (ZH)**: LeanProgress：通过证明进展预测指导神经定理证明的搜索 

**Authors**: Suozhi Huang, Peiyang Song, Robert Joseph George, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17925)  

**Abstract**: Mathematical reasoning remains a significant challenge for Large Language Models (LLMs) due to hallucinations. When combined with formal proof assistants like Lean, these hallucinations can be eliminated through rigorous verification, making theorem proving reliable. However, even with formal verification, LLMs still struggle with long proofs and complex mathematical formalizations. While Lean with LLMs offers valuable assistance with retrieving lemmas, generating tactics, or even complete proofs, it lacks a crucial capability: providing a sense of proof progress. This limitation particularly impacts the overall development efficiency in large formalization projects. We introduce LeanProgress, a method that predicts the progress in the proof. Training and evaluating our models made on a large corpus of Lean proofs from Lean Workbook Plus and Mathlib4 and how many steps remain to complete it, we employ data preprocessing and balancing techniques to handle the skewed distribution of proof lengths. Our experiments show that LeanProgress achieves an overall prediction accuracy of 75.1\% in predicting the amount of progress and, hence, the remaining number of steps. When integrated into a best-first search framework using Reprover, our method shows a 3.8\% improvement on Mathlib4 compared to baseline performances of 41.2\%, particularly for longer proofs. These results demonstrate how proof progress prediction can enhance both automated and interactive theorem proving, enabling users to make more informed decisions about proof strategies. 

**Abstract (ZH)**: 数学推理仍然是大型语言模型（LLMs）的一个重大挑战，尤其是由于它们可能会产生幻觉。当与形式化证明助手如Lean结合使用时，这些幻觉可以通过严格的验证消除，从而使定理证明变得可靠。然而，即使有形式验证，LLMs仍然难以处理长证明和复杂的数学形式化。虽然结合LLMs的Lean可以为检索引理、生成策略或甚至完整的证明提供有价值的帮助，但它缺乏一个关键的功能：提供证明进程感。这个限制尤其影响大型形式化项目的整体开发效率。我们引入了LeanProgress，这是一种预测证明进程的方法。我们通过对Lean Workbook Plus和Mathlib4中的大量Lean证明进行训练和评估，并预测剩余需要完成的步骤数量，我们采用了数据预处理和平衡技术来处理证明长度分布不均的问题。我们的实验表明，LeanProgress在预测证明进程量方面总体准确率达到75.1%，因此预测剩余的步骤数量。当将我们的方法集成到使用Reprover的优先级搜索框架中时，它在Mathlib4上比基线性能（41.2%）提高了3.8%，特别是在处理较长证明时。这些结果表明，证明进程预测如何可以提升自动和交互式定理证明，使用户能够更加明智地选择证明策略。 

---
# DocPuzzle: A Process-Aware Benchmark for Evaluating Realistic Long-Context Reasoning Capabilities 

**Title (ZH)**: DocPuzzle：一种基于过程的基准测试，用于评估实际长上下文推理能力 

**Authors**: Tianyi Zhuang, Chuqiao Kuang, Xiaoguang Li, Yihua Teng, Jihao Wu, Yasheng Wang, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17807)  

**Abstract**: We present DocPuzzle, a rigorously constructed benchmark for evaluating long-context reasoning capabilities in large language models (LLMs). This benchmark comprises 100 expert-level QA problems requiring multi-step reasoning over long real-world documents. To ensure the task quality and complexity, we implement a human-AI collaborative annotation-validation pipeline. DocPuzzle introduces an innovative evaluation framework that mitigates guessing bias through checklist-guided process analysis, establishing new standards for assessing reasoning capacities in LLMs. Our evaluation results show that: 1)Advanced slow-thinking reasoning models like o1-preview(69.7%) and DeepSeek-R1(66.3%) significantly outperform best general instruct models like Claude 3.5 Sonnet(57.7%); 2)Distilled reasoning models like DeepSeek-R1-Distill-Qwen-32B(41.3%) falls far behind the teacher model, suggesting challenges to maintain the generalization of reasoning capabilities relying solely on distillation. 

**Abstract (ZH)**: 我们提出了DocPuzzle，这是一个严格构建的基准测试，用于评估大型语言模型（LLMs）的长上下文推理能力。该基准测试包含100个专家级的问答问题，要求在长篇真实世界文档上进行多步骤推理。为确保任务的质量和复杂性，我们实现了一个由人工和AI协作的注释验证流程。DocPuzzle引入了一种创新的评估框架，通过清单指导的过程分析来减少猜测偏差，从而为评估LLMs的推理能力建立了新的标准。我们的评估结果显示：1）先进的慢思维推理模型（如o1-preview, 69.7% 和 DeepSeek-R1, 66.3%）显著优于最佳通用指令模型（如Claude 3.5 Sonnet, 57.7%）；2）如DeepSeek-R1-Distill-Qwen-32B（41.3%）这样的提炼推理模型远逊于教师模型，表明仅通过提炼难以维持推理能力的一般化。 

---
# Representation Engineering for Large-Language Models: Survey and Research Challenges 

**Title (ZH)**: 面向大规模语言模型的表示工程：综述与研究挑战 

**Authors**: Lukasz Bartoszcze, Sarthak Munshi, Bryan Sukidi, Jennifer Yen, Zejia Yang, David Williams-King, Linh Le, Kosi Asuzu, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2502.17601)  

**Abstract**: Large-language models are capable of completing a variety of tasks, but remain unpredictable and intractable. Representation engineering seeks to resolve this problem through a new approach utilizing samples of contrasting inputs to detect and edit high-level representations of concepts such as honesty, harmfulness or power-seeking. We formalize the goals and methods of representation engineering to present a cohesive picture of work in this emerging discipline. We compare it with alternative approaches, such as mechanistic interpretability, prompt-engineering and fine-tuning. We outline risks such as performance decrease, compute time increases and steerability issues. We present a clear agenda for future research to build predictable, dynamic, safe and personalizable LLMs. 

**Abstract (ZH)**: 大型语言模型能够完成多种任务，但它们仍然具有不可预测性和难以处理的特点。通过一种新的方法——利用具有对比输入的样本来检测和编辑有关诚实性、危害性或权力获取等高级概念的表示，表示工程学旨在解决这些问题。我们对表示工程学的目标和方法进行正式化，以呈现这一新兴学科中工作的整体图景。我们将它与替代方法，如机械可解释性、提示工程和微调进行比较。我们指出了由此方法带来的风险，例如性能下降、计算时间增加和可控性问题。我们提出了明确的研究议程，旨在构建可预测、动态、安全和个性化的大型语言模型。 

---
# Intention Recognition in Real-Time Interactive Navigation Maps 

**Title (ZH)**: 实时交互导航图中的意图识别 

**Authors**: Peijie Zhao, Zunayed Arefin, Felipe Meneguzzi, Ramon Fraga Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2502.17581)  

**Abstract**: In this demonstration, we develop IntentRec4Maps, a system to recognise users' intentions in interactive maps for real-world navigation. IntentRec4Maps uses the Google Maps Platform as the real-world interactive map, and a very effective approach for recognising users' intentions in real-time. We showcase the recognition process of IntentRec4Maps using two different Path-Planners and a Large Language Model (LLM).
GitHub: this https URL 

**Abstract (ZH)**: 在本次演示中，我们开发了IntentRec4Maps系统，该系统用于识别用户在交互地图中进行实际导航时的意图。IntentRec4Maps使用Google Maps Platform作为实时交互地图，并采用了一种非常有效的方法来实现实时识别用户的意图。我们通过使用两种不同的路径规划器（Path-Planner）和大语言模型（Large Language Model, LLM）来展示IntentRec4Maps的识别过程。

GitHub：[此处填写链接]

（注意：由于提供的链接是Markdown格式的原始链接，需要根据实际需求将其转换为完整的URL。） 

---
# Large Language Model Driven Agents for Simulating Echo Chamber Formation 

**Title (ZH)**: 基于大型语言模型的智能代理用于模拟回声室效应形成 

**Authors**: Chenhao Gu, Ling Luo, Zainab Razia Zaidi, Shanika Karunasekera  

**Link**: [PDF](https://arxiv.org/pdf/2502.18138)  

**Abstract**: The rise of echo chambers on social media platforms has heightened concerns about polarization and the reinforcement of existing beliefs. Traditional approaches for simulating echo chamber formation have often relied on predefined rules and numerical simulations, which, while insightful, may lack the nuance needed to capture complex, real-world interactions. In this paper, we present a novel framework that leverages large language models (LLMs) as generative agents to simulate echo chamber dynamics within social networks. The novelty of our approach is that it incorporates both opinion updates and network rewiring behaviors driven by LLMs, allowing for a context-aware and semantically rich simulation of social interactions. Additionally, we utilize real-world Twitter (now X) data to benchmark the LLM-based simulation against actual social media behaviors, providing insights into the accuracy and realism of the generated opinion trends. Our results demonstrate the efficacy of LLMs in modeling echo chamber formation, capturing both structural and semantic dimensions of opinion clustering. %This work contributes to a deeper understanding of social influence dynamics and offers a new tool for studying polarization in online communities. 

**Abstract (ZH)**: 社交媒体平台上回声室现象的兴起加剧了关于极化和现有信念强化的担忧。传统上用于模拟回声室形成的手段往往依赖于预设规则和数值模拟，虽然具有一定的启发性，但在捕捉复杂现实世界交互方面可能缺乏细腻之处。在本文中，我们提出了一种新颖的框架，利用大语言模型（LLMs）作为生成代理来模拟社交网络中的回声室动态。我们方法的新颖之处在于，它结合了由LLMs驱动的意见更新和网络重连行为，从而实现对社交互动的上下文意识和语义丰富的模拟。此外，我们使用真实世界的Twitter（现在称为X）数据，将基于LLM的模拟与实际社交媒体行为进行基准测试，以洞察生成意见趋势的准确性和现实性。实验结果表明，LLMs在建模回声室形成方面具有有效性，能够捕获意见集群的结构和语义维度。%本研究加深了对社会影响动力学的理解，并提供了一个研究在线社区极化的新工具。 

---
# MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration 

**Title (ZH)**: MRBTP：高效的多机器人行为树规划与协作 

**Authors**: Yishuai Cai, Xinglin Chen, Zhongxuan Cai, Yunxin Mao, Minglong Li, Wenjing Yang, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18072)  

**Abstract**: Multi-robot task planning and collaboration are critical challenges in robotics. While Behavior Trees (BTs) have been established as a popular control architecture and are plannable for a single robot, the development of effective multi-robot BT planning algorithms remains challenging due to the complexity of coordinating diverse action spaces. We propose the Multi-Robot Behavior Tree Planning (MRBTP) algorithm, with theoretical guarantees of both soundness and completeness. MRBTP features cross-tree expansion to coordinate heterogeneous actions across different BTs to achieve the team's goal. For homogeneous actions, we retain backup structures among BTs to ensure robustness and prevent redundant execution through intention sharing. While MRBTP is capable of generating BTs for both homogeneous and heterogeneous robot teams, its efficiency can be further improved. We then propose an optional plugin for MRBTP when Large Language Models (LLMs) are available to reason goal-related actions for each robot. These relevant actions can be pre-planned to form long-horizon subtrees, significantly enhancing the planning speed and collaboration efficiency of MRBTP. We evaluate our algorithm in warehouse management and everyday service scenarios. Results demonstrate MRBTP's robustness and execution efficiency under varying settings, as well as the ability of the pre-trained LLM to generate effective task-specific subtrees for MRBTP. 

**Abstract (ZH)**: 多机器人任务规划与协作是机器人领域中的关键挑战。虽然行为树（BTs）已被确立为一种流行的控制架构，并且可用于单个机器人的规划，但由于协调多元动作空间的复杂性，开发有效的多机器人行为树规划算法仍然具有挑战性。我们提出了一种多机器人行为树规划（MRBTP）算法，该算法在理论上保证了完备性和正确性。MRBTP 特别设计了跨树扩展机制，用于协调不同行为树之间的不同动作，以实现团队目标。对于相同的动作，MRBTP 保留了行为树之间的辅助结构，以确保鲁棒性并防止由于意图共享而导致的冗余执行。尽管 MRBTP 能够生成适用于同构和异构机器人群体的行为树，但其效率仍有待提高。当大型语言模型（LLMs）可用时，我们提出了一种可选插件来进一步增强 MRBTP 的效果。该插件可以推理每个机器人的相关动作，并提前规划这些动作以形成长视线子树，显著提升 MRBTP 的规划速度和协作效率。我们将在仓库管理和日常服务场景中评估该算法。结果表明，MRBTP 在不同设置下具有鲁棒性和执行效率，并且预训练的 LLM 有能力为 MRBTP 生成有效的任务特定子树。 

---
# AutoCas: Autoregressive Cascade Predictor in Social Networks via Large Language Models 

**Title (ZH)**: AutoCas：通过大规模语言模型在社交网络中实现自回归级联预测器 

**Authors**: Yuhao Zheng, Chenghua Gong, Rui Sun, Juyuan Zhang, Liming Pan, Linyuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2502.18040)  

**Abstract**: Popularity prediction in information cascades plays a crucial role in social computing, with broad applications in viral marketing, misinformation control, and content recommendation. However, information propagation mechanisms, user behavior, and temporal activity patterns exhibit significant diversity, necessitating a foundational model capable of adapting to such variations. At the same time, the amount of available cascade data remains relatively limited compared to the vast datasets used for training large language models (LLMs). Recent studies have demonstrated the feasibility of leveraging LLMs for time-series prediction by exploiting commonalities across different time-series domains. Building on this insight, we introduce the Autoregressive Information Cascade Predictor (AutoCas), an LLM-enhanced model designed specifically for cascade popularity prediction. Unlike natural language sequences, cascade data is characterized by complex local topologies, diffusion contexts, and evolving dynamics, requiring specialized adaptations for effective LLM integration. To address these challenges, we first tokenize cascade data to align it with sequence modeling principles. Next, we reformulate cascade diffusion as an autoregressive modeling task to fully harness the architectural strengths of LLMs. Beyond conventional approaches, we further introduce prompt learning to enhance the synergy between LLMs and cascade prediction. Extensive experiments demonstrate that AutoCas significantly outperforms baseline models in cascade popularity prediction while exhibiting scaling behavior inherited from LLMs. Code is available at this repository: this https URL 

**Abstract (ZH)**: 信息cascade中的流行性预测在社交计算中扮演着至关重要的角色，广泛应用于病毒式营销、虚假信息控制和内容推荐等方面。然而，信息传播机制、用户行为和时间活动模式表现出显著的异质性，这需要一种能够适应这些变化的基础模型。同时，可用的cascade数据集相对有限，与用于训练大规模语言模型（LLMs）的庞大数据集相比，仍显得不足。最近的研究表明，通过利用不同时间序列领域的共有特性，可以利用LLMs进行时间序列预测。基于这一洞察，我们提出了Autoregressive Information Cascade Predictor（AutoCas），这是一种专门针对cascade流行性预测的增强型LLM模型。与自然语言序列不同，cascade数据具有复杂的局部拓扑结构、扩散上下文和不断演化的动态特性，这要求进行专门的适应以有效集成LLMs。为了解决这些挑战，我们首先对cascade数据进行分词，使其符合序列建模的原则。接下来，我们将cascade扩散重新构想为自回归建模任务，以便充分利用LLMs的架构优势。除了传统的做法，我们还引入了提示学习来增强LLMs与cascade预测之间的协同作用。实验结果表明，AutoCas在cascade流行性预测方面显著优于基线模型，并且表现出与LLMs相继承的扩展性行为。相关代码可在以下仓库中获取：this https URL 

---
# Uncertainty Quantification for LLM-Based Survey Simulations 

**Title (ZH)**: 基于LLM的调查模拟中的不确定性量化 

**Authors**: Chengpiao Huang, Yuhang Wu, Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17773)  

**Abstract**: We investigate the reliable use of simulated survey responses from large language models (LLMs) through the lens of uncertainty quantification. Our approach converts synthetic data into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs. 

**Abstract (ZH)**: 我们通过不确定性量化这一视角，调查了大型语言模型（LLMs）模拟调查响应的可靠使用方法。我们的方法将合成数据转换为人类响应总体参数的信心区间，以解决模拟人群与真实人群之间的分布差异问题。我们的一项关键创新在于确定了模拟响应的最佳数量：模拟反应过多会导致信心区间过于狭窄且覆盖不足，而模拟反应过少则会导致过于宽松的估计结果。为解决这一问题，我们的方法能够自适应地选择模拟样本量，确保平均情形下的有效覆盖保证。该方法适用于任何大型语言模型，不论其保真度如何，以及任何构建信心区间的方法。此外，所选样本量可以量化大型语言模型与目标人类群体之间的不一致程度。我们通过实际数据集和大型语言模型来说明该方法的应用。 

---
# DeepSeek vs. ChatGPT: A Comparative Study for Scientific Computing and Scientific Machine Learning Tasks 

**Title (ZH)**: DeepSeek 与 ChatGPT 在科学计算和科学机器学习任务中的比较研究 

**Authors**: Qile Jiang, Zhiwei Gao, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.17764)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for tackling a wide range of problems, including those in scientific computing, particularly in solving partial differential equations (PDEs). However, different models exhibit distinct strengths and preferences, resulting in varying levels of performance. In this paper, we compare the capabilities of the most advanced LLMs--ChatGPT and DeepSeek--along with their reasoning-optimized versions in addressing computational challenges. Specifically, we evaluate their proficiency in solving traditional numerical problems in scientific computing as well as leveraging scientific machine learning techniques for PDE-based problems. We designed all our experiments so that a non-trivial decision is required, e.g. defining the proper space of input functions for neural operator learning. Our findings reveal that the latest model, ChatGPT o3-mini-high, usually delivers the most accurate results while also responding significantly faster than its reasoning counterpart, DeepSeek R1. This enhanced speed and accuracy make ChatGPT o3-mini-high a more practical and efficient choice for diverse computational tasks at this juncture. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经成为解决广泛问题的强大工具，特别是在科学计算领域，特别是在求解偏微分方程（PDEs）方面。然而，不同的模型表现出不同的优势和偏好，导致其性能各不相同。在本文中，我们比较了最先进的LLMs——ChatGPT和DeepSeek及其推理优化版本在解决计算挑战方面的能力。具体而言，我们评估了它们在解决科学计算中的传统数值问题以及利用科学机器学习技术解决基于PDE的问题方面的熟练程度。我们设计了所有实验，以确保需要作出非平凡的决策，例如定义神经算子学习的输入函数空间。研究发现，最新版本的ChatGPT o3-mini-high通常提供最准确的结果，同时响应速度也显著快于其推理优化版本DeepSeek R1。这种增强的速度和准确性使ChatGPT o3-mini-high在当前情况下成为一个更实用和高效的选择，适用于各种计算任务。 

---
# Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM 

**Title (ZH)**: 集成联邦学习和多模态大语言模型的分布式安全威胁检测系统的设计与实现 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17763)  

**Abstract**: Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems. 

**Abstract (ZH)**: 传统的安全保护方法在处理大规模分布式系统中的复杂攻击向量时显得力不从心，尤其是在权衡检测准确性和数据隐私问题之间时更为显著。本文提出了一种新颖的分布式安全威胁检测系统，该系统将联邦学习与多模态大型语言模型（LLMs）结合在一起。我们的系统利用联邦学习来确保数据隐私，并使用多模态LLMs来处理包括网络流量、系统日志、图像和传感器数据在内的异构数据源。在10TB分布式数据集上的实验评估表明，我们的方法实现了96.4%的检测准确率，较传统基线模型提高了4.1个百分点。系统将假阳性率和假阴性率分别减少了1.8个百分点和2.4个百分点。性能分析表明，我们的系统在分布式环境中保持了高效的处理能力，模型训练时间仅为180秒，而分布式网络中的威胁检测时间仅为3.8秒。这些结果表明，在提高检测准确率和计算效率的同时，仍能保留数据隐私，这表明该系统具有在大规模安全系统中实际部署的强大潜力。 

---
# StatLLM: A Dataset for Evaluating the Performance of Large Language Models in Statistical Analysis 

**Title (ZH)**: StatLLM：用于评估大型语言模型在统计分析性能的数据集 

**Authors**: Xinyi Song, Lina Lee, Kexin Xie, Xueying Liu, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17657)  

**Abstract**: The coding capabilities of large language models (LLMs) have opened up new opportunities for automatic statistical analysis in machine learning and data science. However, before their widespread adoption, it is crucial to assess the accuracy of code generated by LLMs. A major challenge in this evaluation lies in the absence of a benchmark dataset for statistical code (e.g., SAS and R). To fill in this gap, this paper introduces StatLLM, an open-source dataset for evaluating the performance of LLMs in statistical analysis. The StatLLM dataset comprises three key components: statistical analysis tasks, LLM-generated SAS code, and human evaluation scores. The first component includes statistical analysis tasks spanning a variety of analyses and datasets, providing problem descriptions, dataset details, and human-verified SAS code. The second component features SAS code generated by ChatGPT 3.5, ChatGPT 4.0, and Llama 3.1 for those tasks. The third component contains evaluation scores from human experts in assessing the correctness, effectiveness, readability, executability, and output accuracy of the LLM-generated code. We also illustrate the unique potential of the established benchmark dataset for (1) evaluating and enhancing natural language processing metrics, (2) assessing and improving LLM performance in statistical coding, and (3) developing and testing of next-generation statistical software - advancements that are crucial for data science and machine learning research. 

**Abstract (ZH)**: 大型语言模型（LLMs）的编码能力为机器学习和数据科学中的自动统计分析开辟了新的机会。然而，在它们广泛采用之前，评估LLM生成代码的准确性至关重要。该评估面临的主要挑战是没有现成的统计代码基准数据集（如SAS和R）。为填补这一空白，本文介绍了StatLLM，一个开源数据集，用于评估LLM在统计分析中的性能。StatLLM数据集包含三个关键组成部分：统计分析任务、LLM生成的SAS代码以及人工评估得分。第一部分包括各种分析和数据集的统计分析任务，提供问题描述、数据集详细信息以及人工验证的SAS代码。第二部分展示了ChatGPT 3.5、ChatGPT 4.0和Llama 3.1为这些任务生成的SAS代码。第三部分包含来自专家的评估分数，用于评估LLM生成代码的正确性、有效性、可读性、可执行性以及输出准确性。我们还展示了该基准数据集的潜在独特用途，包括：(1) 评估和提升自然语言处理指标，(2) 评估和改进LLM在统计编码中的性能，以及(3) 开发和测试下一代统计软件。这些进步对于数据科学和机器学习研究至关重要。 

---
# Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law 

**Title (ZH)**: 面向稳健的法律推理：利用逻辑大语言模型在法律领域中的应用 

**Authors**: Manuj Kant, Sareh Nabi, Manav Kant, Roland Scharrer, Megan Ma, Marzieh Nabi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17638)  

**Abstract**: Legal services rely heavily on text processing. While large language models (LLMs) show promise, their application in legal contexts demands higher accuracy, repeatability, and transparency. Logic programs, by encoding legal concepts as structured rules and facts, offer reliable automation, but require sophisticated text extraction. We propose a neuro-symbolic approach that integrates LLMs' natural language understanding with logic-based reasoning to address these limitations.
As a legal document case study, we applied neuro-symbolic AI to coverage-related queries in insurance contracts using both closed and open-source LLMs. While LLMs have improved in legal reasoning, they still lack the accuracy and consistency required for complex contract analysis. In our analysis, we tested three methodologies to evaluate whether a specific claim is covered under a contract: a vanilla LLM, an unguided approach that leverages LLMs to encode both the contract and the claim, and a guided approach that uses a framework for the LLM to encode the contract. We demonstrated the promising capabilities of LLM + Logic in the guided approach. 

**Abstract (ZH)**: 法律服务高度依赖文本处理。尽管大型语言模型（LLMs）展现出潜力，但在法律情境中的应用需要更高的准确度、可重复性和透明度。逻辑程序通过将法律概念编码为结构化的规则和事实，提供了可靠的自动化功能，但需要复杂的文本提取技术。我们提出了一种结合了LLMs自然语言理解能力和基于逻辑的推理的神经符号方法，以解决这些限制。

作为法律文件案例研究，我们使用闭源和开源LLMs将神经符号AI应用于保险合同中与保险覆盖相关的查询。尽管LLMs在法律推理方面有所改进，但在复杂合同分析所需的高度准确性和一致性方面仍存在差距。在我们的分析中，我们测试了三种方法来评估某一特定索赔是否被合同所覆盖：一种基础的LLM方法，一种未经指导的方法，该方法利用LLM将合同和索赔都编码进去，以及一种有引导的方法，该方法使用一个框架使LLM将合同编码起来。我们展示了在有引导的方法中，LLM与逻辑结合的有前景的能力。 

---
# Hallucination Detection in LLMs Using Spectral Features of Attention Maps 

**Title (ZH)**: 使用注意力图频谱特征进行大语言模型中的幻觉检测 

**Authors**: Jakub Binkowski, Denis Janiak, Albert Sawczyn, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2502.17598)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务上展现出了卓越的性能，但仍然容易产生幻觉。检测幻觉对于安全关键的应用至关重要，尽管最近的方法通过利用注意力图的属性来实现这一目标，但其有效性仍存在限制。在本研究中，我们通过将注意力图解释为图结构的邻接矩阵，探究了注意力图的频谱特征。我们提出了LapEigvals方法，该方法利用从注意力图中提取的拉普拉斯矩阵的前k个特征值作为幻觉检测探针的输入。实证研究表明，我们的方法在基于注意力的方法中实现了最先进的幻觉检测性能。广泛的消融研究进一步突出了LapEigvals的 robustness和泛化能力，为幻觉检测领域的未来发展奠定了基础。 

---
# Thinking Before Running! Efficient Code Generation with Thorough Exploration and Optimal Refinement 

**Title (ZH)**: 思考在先！通过全面探索与最优精炼实现高效代码生成 

**Authors**: Xiaoqing Zhang, Yuhan Liu, Flood Sung, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.17442)  

**Abstract**: Code generation is crucial in software engineering for automating the coding process efficiently. While test-time computation methods show promise, they suffer from high latency due to multiple computation rounds. To overcome this, we introduce ThinkCoder, a framework that combines thorough exploration with optimal refinement. The exploration phase diversifies the solution space by searching for potential solutions, followed by a refinement phase that enhances precision. This approach allows us to select the best solution through careful consideration before taking action, avoiding excessive trial and error. To further minimize test-time computation overhead, we introduce preference-driven optimization with Reinforced Self-Training (ReST), which uses exploration trajectories from ThinkCoder to guide LLM's evolution. By learning preferences, this approach improves LLM's exploration efficiency, reducing computational costs while maintaining accuracy. ThinkCoder boosts the performance of multiple base LLMs, excelling on benchmarks like HumanEval and MBPP. Compared to SOTA models, it improves Pass@1 by 1.5\% over MapCoder with just 21.7\% of the computation cost. Against AgentCoder, ThinkCoder achieves a 0.6\% higher Pass@1 after 2 rounds, outperforming AgentCoder's 5 rounds. Additionally, ReST with success trajectories enhances efficiency, allowing models like LLaMA2-7B to achieve competitive results using only 20\% of the computational resources. These results highlight the framework's effectiveness and scalability. 

**Abstract (ZH)**: 代码生成是软件工程中提高代码编写效率的关键。尽管测试时的计算方法显示出潜力，但由于多轮计算导致的高延迟成为其主要限制。为克服这一问题，我们提出了ThinkCoder框架，该框架结合了全面的探索和最优的提炼。探索阶段通过搜索潜在解决方案来多样化解决方案空间，随后的提炼阶段则提高精度。这种方法允许我们在采取行动之前仔细考虑，从而避免不必要的试错。为了进一步减少测试时的计算开销，引入了基于偏好的优化方法Reinforced Self-Training（ReST），该方法利用ThinkCoder的探索轨迹来引导LLM的演化。通过学习偏好，这种方法提高了LLM的探索效率，同时降低了计算成本并保持了准确性。ThinkCoder提升了多个基模型的表现，其在HumanEval和MBPP等基准测试上表现出色。与最新模型相比，它在与MapCoder对比时，能耗仅占MapCoder的21.7%，但Pass@1表现提高了1.5%。与AgentCoder相比，在2轮测试后，ThinkCoder的Pass@1提高了0.6%，而AgentCoder则需要5轮。此外，使用成功轨迹的ReST方法提高了效率，使得如LLaMA2-7B这样的模型仅使用20%的计算资源即可获得竞争力的结果。这些结果突显了该框架的有效性和可扩展性。 

---
# Large Language Models as Realistic Microservice Trace Generators 

**Title (ZH)**: 大型语言模型作为现实的微服务跟踪生成器 

**Authors**: Donghyun Kim, Sriram Ravula, Taemin Ha, Alexandros G. Dimakis, Daehyeok Kim, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2502.17439)  

**Abstract**: Computer system workload traces, which record hardware or software events during application execution, are essential for understanding the behavior of complex systems and managing their processing and memory resources. However, obtaining real-world traces can be challenging due to the significant collection overheads in performance and privacy concerns that arise in proprietary systems. As a result, synthetic trace generation is considered a promising alternative to using traces collected in real-world production deployments. This paper proposes to train a large language model (LLM) to generate synthetic workload traces, specifically microservice call graphs. To capture complex and arbitrary hierarchical structures and implicit constraints in such traces, we fine-tune LLMs to generate each layer recursively, making call graph generation a sequence of easier steps. To further enforce learning constraints in traces and generate uncommon situations, we apply additional instruction tuning steps to align our model with the desired trace features. Our evaluation results show that our model can generate diverse realistic traces under various conditions and outperform existing methods in accuracy and validity. We show that our synthetically generated traces can effectively substitute real-world data in optimizing or tuning systems management tasks. We also show that our model can be adapted to perform key downstream trace-related tasks, specifically, predicting key trace features and infilling missing data given partial traces. Codes are available in this https URL. 

**Abstract (ZH)**: 计算机系统负载迹像是记录应用程序执行期间的硬件或软件事件，对于理解复杂系统的行为以及管理其处理和内存资源至关重要。然而，由于在性能和隐私方面存在显著的采集开销，特别是在专有系统中，获得真实世界的迹像是一个挑战。因此，合成迹像是使用在实际生产部署中收集的真实世界迹像是一个有希望的替代方法。本文提出了一种基于大型语言模型（LLM）生成合成负载迹线的方法，特别是微服务调用图。为了捕捉此类迹线中复杂的和任意的分层结构以及隐含的约束，我们通过递归地对每个层次进行微调，将调用图生成分解为一系列较易处理的步骤。为了进一步加强迹线中的学习约束并生成不常见的情况，我们应用附加的指令微调步骤，以使模型与所需的迹线特征一致。我们的评估结果表明，我们的模型在各种条件下能够生成多样化的现实迹线，并且在准确性和有效性方面均优于现有方法。我们展示了通过合成生成的迹线可以有效地替代实际数据以优化或调整系统管理任务。此外，我们展示了我们的模型可以适应执行关键的下游迹线相关任务，特别是预测关键迹线特征以及在部分迹线情况下填补缺失的数据。代码可在以下链接获取：[在此处插入链接]。 

---
