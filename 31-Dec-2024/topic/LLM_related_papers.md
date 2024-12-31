# Facilitating large language model Russian adaptation with Learned Embedding Propagation 

**Title (ZH)**: 使用学习嵌入传播促进大型语言模型的俄语适应 

**Authors**: Mikhail Tikhomirov, Daniil Chernyshev  

**Link**: [PDF](https://arxiv.org/pdf/2412.21140)  

**Abstract**: Rapid advancements of large language model (LLM) technologies led to the introduction of powerful open-source instruction-tuned LLMs that have the same text generation quality as the state-of-the-art counterparts such as GPT-4. While the emergence of such models accelerates the adoption of LLM technologies in sensitive-information environments the authors of such models don not disclose the training data necessary for replication of the results thus making the achievements model-exclusive. Since those open-source models are also multilingual this in turn reduces the benefits of training a language specific LLMs as improved inference computation efficiency becomes the only guaranteed advantage of such costly procedure. More cost-efficient options such as vocabulary extension and subsequent continued pre-training are also inhibited by the lack of access to high-quality instruction-tuning data since it is the major factor behind the resulting LLM task-solving capabilities. To address the limitations and cut the costs of the language adaptation pipeline we propose Learned Embedding Propagation (LEP). Unlike existing approaches our method has lower training data size requirements due to minimal impact on existing LLM knowledge which we reinforce using novel ad-hoc embedding propagation procedure that allows to skip the instruction-tuning step and instead implant the new language knowledge directly into any existing instruct-tuned variant. We evaluated four Russian vocabulary adaptations for LLaMa-3-8B and Mistral-7B, showing that LEP is competitive with traditional instruction-tuning methods, achieving performance comparable to OpenChat 3.5 and LLaMa-3-8B-Instruct, with further improvements via self-calibration and continued tuning enhancing task-solving capabilities. 

**Abstract (ZH)**: 大型语言模型（LLM）技术的迅速发展导致了具有与GPT-4等最先进的模型相同文本生成质量的强大开源指令调优LLM的出现。尽管这些模型的出现加速了在敏感信息环境中的LLM技术的采用，但这些模型的开发者并未披露必要的训练数据，以便复制这些成果，这使得这些成就成为了特定模型的专属成果。由于这些开源模型是多语种模型，这意味着训练特定语言的LLM的好处减少了，因为改进的推理计算效率成为了唯一可保证的优势。此外，由于缺乏高质量指令调优数据的访问权限，词汇扩展和后续继续预训练等更低成本的选择也受到了抑制，因为这些方法的进步取决于高质量的数据。为了解决这些限制并降低语言适应管道的成本，我们提出了学习嵌入传播（LEP）方法。与现有方法相比，我们的方法对现有LLM知识的影响较小，因此所需的训练数据量也较少。我们通过一种新颖的即用型嵌入传播程序增强了这一点，该程序允许跳过指令调优步骤，而是直接将新的语言知识植入任何现有的指令调优变体中。我们对LLaMa-3-8B和Mistral-7B的四种俄语词汇适应进行了评估，表明LEP在性能上与传统的指令调优方法相当，与OpenChat 3.5和LLaMa-3-8B-Instruct相当，进一步通过自我校准和继续调优，提高了任务解决能力。 

---
# Plancraft: an evaluation dataset for planning with LLM agents 

**Title (ZH)**: PlanCraft：用于评估基于LLM代理的规划数据集 

**Authors**: Gautier Dagan, Frank Keller, Alex Lascarides  

**Link**: [PDF](https://arxiv.org/pdf/2412.21033)  

**Abstract**: We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as an oracle planner and oracle RAG information extractor, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and strategies on our task and compare their performance to a handcrafted planner. We find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and we offer suggestions on how to improve their capabilities. 

**Abstract (ZH)**: 我们介绍了Plancraft，这是一个针对大规模语言模型（LLM）代理的跨模态评估数据集。Plancraft 支持文本-only 和跨模态两种界面，基于《我的世界》（Minecraft）的制作用户界面（GUI）。我们包含了《我的世界》维基，用于评估工具使用和检索增强生成（RAG），同时也提供了一个 oracle 计划者和 oracle RAG 信息提取器，以消除现代代理架构中不同组件的影响。为了评估决策能力，Plancraft 还包括了一部分故意无法解决的示例，这些示例为代理提供了真实的挑战，不仅需要代理完成任务，还需要代理判断这些任务是否可解。我们对开源和封闭源代码的 LLM 和策略进行了基准测试，并将它们的表现与手工设计的计划者进行了比较。我们发现，LLM 和视觉-语言模型在处理 Plancraft 引入的规划问题时表现不佳，并提出了提高其能力的建议。 

---
# Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale via Principled Criteria 

**Title (ZH)**: 基于冗余性原则评估的verbosity意识摘要生成：有效减少冗余理由的方法 

**Authors**: Joonwon Jang, Jaehee Kim, Wonbin Kweon, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.21006)  

**Abstract**: Large Language Models (LLMs) rely on generating extensive intermediate reasoning units (e.g., tokens, sentences) to enhance final answer quality across a wide range of complex tasks. While generating multiple reasoning paths or iteratively refining rationales proves effective for improving performance, these approaches inevitably result in significantly higher inference costs. In this work, we propose a novel sentence-level rationale reduction training framework that leverages likelihood-based criteria, verbosity, to identify and remove redundant reasoning sentences. Unlike previous approaches that utilize token-level reduction, our sentence-level reduction framework maintains model performance while reducing generation length. This preserves the original reasoning abilities of LLMs and achieves an average 17.15% reduction in generation costs across various models and tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）依赖于生成大量的中间推理单元（例如，词元、句子）以在多种复杂的任务中提高最终答案的质量。尽管生成多条推理路径或迭代优化理由已被证明能有效提高性能，但这些方法不可避免地会导致显著增加推理成本。在本项研究中，我们提出了一种新颖的句子级理由缩减训练框架，该框架利用基于似然性的标准（如冗余度），以识别并移除冗余的推理句子。不同于以往依赖于词元级缩减的方法，我们的句子级缩减框架能够在保持模型性能的同时减少生成长度。这保留了LLMs的原始推理能力，并在各种模型和任务中实现平均每种模型17.15%的生成成本降低。 

---
# KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation 

**Title (ZH)**: KARPA：一种无需训练的方法，将知识图谱适应为大型语言模型推理路径聚合的参考 

**Authors**: Siyuan Fang, Kaijing Ma, Tianyu Zheng, Xinrun Du, Ningxuan Lu, Ge Zhang, Qingkun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20995)  

**Abstract**: Large language models (LLMs) demonstrate exceptional performance across a variety of tasks, yet they are often affected by hallucinations and the timeliness of knowledge. Leveraging knowledge graphs (KGs) as external knowledge sources has emerged as a viable solution, but existing methods for LLM-based knowledge graph question answering (KGQA) are often limited by step-by-step decision-making on KGs, restricting the global planning and reasoning capabilities of LLMs, or they require fine-tuning or pre-training on specific KGs. To address these challenges, we propose Knowledge graph Assisted Reasoning Path Aggregation (KARPA), a novel framework that harnesses the global planning abilities of LLMs for efficient and accurate KG reasoning. KARPA operates in three steps: pre-planning relation paths using the LLM's global planning capabilities, matching semantically relevant paths via an embedding model, and reasoning over these paths to generate answers. Unlike existing KGQA methods, KARPA avoids stepwise traversal, requires no additional training, and is adaptable to various LLM architectures. Extensive experimental results show that KARPA achieves state-of-the-art performance in KGQA tasks, delivering both high efficiency and accuracy. Our code will be available on Github. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但常常受到幻觉和知识时效性的影响。利用知识图谱（KGs）作为外部知识源已经出现了可行的解决方案，但现有基于LLM的知识图谱问答（KGQA）方法往往受限于逐级决策的过程，限制了LLM的全局规划和推理能力，或者需要针对特定的KG进行微调或预训练。为了解决这些问题，我们提出了一种名为知识图谱辅助推理路径聚合（KARPA）的新框架，该框架利用LLM的全局规划能力进行高效和准确的知识图谱推理。KARPA的工作流程分为三个步骤：利用LLM的全局规划能力预先规划关系路径，通过嵌入模型匹配语义相关路径，并在这些路径上进行推理生成答案。与现有的KGQA方法相比，KARPA避免了逐级遍历，不需要额外的训练，并且可以适应各种LLM架构。实验结果表明，KARPA在KGQA任务中达到了最先进的性能，既高效又准确。我们的代码将在GitHub上开源。 

---
# Enhancing Annotated Bibliography Generation with LLM Ensembles 

**Title (ZH)**: 使用大型语言模型ensemble增强标注参考文献生成 

**Authors**: Sergio Bermejo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20864)  

**Abstract**: This work proposes a novel approach to enhancing annotated bibliography generation through Large Language Model (LLM) ensembles. In particular, multiple LLMs in different roles -- controllable text generation, evaluation, and summarization -- are introduced and validated using a systematic methodology to enhance model performance in scholarly tasks. Output diversity among the ensemble that generates text is obtained using different LLM parameters, followed by an LLM acting as a judge to assess relevance, accuracy, and coherence. Responses selected by several combining strategies are then merged and refined through summarization and redundancy removal techniques. The preliminary experimental validation demonstrates that the combined outputs from the LLM ensemble improve coherence and relevance compared to individual responses, leading to a 38% improvement in annotation quality and a 51% reduction in content redundancy, thus highlighting the potential for automating complex scholarly tasks while maintaining high-quality standards. 

**Abstract (ZH)**: 本文提出了一种通过大型语言模型（LLM）集成提升标注参考文献生成的新方法。具体而言，通过引入并在系统的方法论框架下验证了多个在不同角色中工作的LLM——包括可控文本生成、评估和总结——以提升模型在学术任务中的性能。通过使用不同的LLM参数生成文本，并通过一个评估模型来评估相关性、准确性和连贯性，从而实现集成生成的文本输出多样性。之后，通过总结和去除冗余的技术，采用多种组合策略选出的响应被合并和精炼。初步的实验验证表明，LLM集成组合输出在连贯性和相关性方面优于单个响应，导致注释质量提高了38%，冗余内容减少了51%。这不仅证明了自动执行复杂学术任务的可行性，还保持了高质量标准，突显了自动化在学术领域应用的潜力。 

---
# Are LLMs Really Not Knowledgable? Mining the Submerged Knowledge in LLMs' Memory 

**Title (ZH)**: 大型语言模型真的不具备知识吗？挖掘大型语言模型记忆中的隐性知识 

**Authors**: Xingjian Tao, Yiwei Wang, Yujun Cai, Zhicheng Yang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20846)  

**Abstract**: Large language models (LLMs) have shown promise as potential knowledge bases, yet they often struggle with question-answering tasks and are prone to hallucinations. While previous research attributes these issues to knowledge gaps in the model's parameters, our investigation reveals a different phenomenon: LLMs often retain correct knowledge even when generating incorrect answers. Through analysis of model's internal representations, we find that correct answers frequently appear among high-probability tokens despite not being selected as final outputs. Based on this observation, we introduce Hits@k, a new metric to assess knowledge retention independent of expression accuracy. Our extensive experiments demonstrate that LLMs store significantly more knowledge than their QA performance suggests. Building on these findings, we develop SkipUnsure, a method to improve answer accuracy by leveraging detected but unexpressed knowledge. Experiments on both open-domain and specific-domain datasets show consistent improvements, with accuracy gains of up to 11.8% on DBPedia and 6.3% on IMDB, without requiring model retraining. 

**Abstract (ZH)**: 大型语言模型（LLMs）在潜在知识库方面展现出了潜力，但它们往往在问答任务中表现不佳，并且容易产生幻觉。虽然先前的研究将这些问题归因于模型参数中的知识缺口，但我们的研究表明存在一种不同的现象：即使生成错误答案，LLMs 经常仍保留了正确的知识。通过分析模型的内部表示，我们发现即使高概率的正确答案没有被选为最终输出，它们也经常出现在高概率的标记中。基于这一观察，我们引入了 Hits@k，这是一种新的度量标准，用于独立于表达准确性来评估知识保留情况。我们的大量实验证明，LLMs 实际上存储的知识比其在问答任务中的表现所表明的要多得多。基于这些发现，我们开发了 SkipUnsure 方法，该方法通过利用检测到但未表达的知识来提高答案准确性。在开放领域和特定领域的数据集上的实验结果一致显示出改进，DBPedia 数据集的准确性提高了 11.8%，IMDB 数据集的准确性提高了 6.3%，而无需进行模型重训练。 

---
# Knowledge Editing for Large Language Model with Knowledge Neuronal Ensemble 

**Title (ZH)**: 大型语言模型的知识编辑通过知识神经元集成 

**Authors**: Yongchang Li, Yujin Zhu, Tao Yan, Shijian Fan, Gang Wu, Liang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20637)  

**Abstract**: As real-world knowledge is constantly evolving, ensuring the timeliness and accuracy of a model's knowledge is crucial. This has made knowledge editing in large language models increasingly important. However, existing knowledge editing methods face several challenges, including parameter localization coupling, imprecise localization, and a lack of dynamic interaction across layers. In this paper, we propose a novel knowledge editing method called Knowledge Neuronal Ensemble (KNE). A knowledge neuronal ensemble represents a group of neurons encoding specific knowledge, thus mitigating the issue of frequent parameter modification caused by coupling in parameter localization. The KNE method enhances the precision and accuracy of parameter localization by computing gradient attribution scores for each parameter at each layer. During the editing process, only the gradients and losses associated with the knowledge neuronal ensemble are computed, with error backpropagation performed accordingly, ensuring dynamic interaction and collaborative updates among parameters. Experimental results on three widely used knowledge editing datasets show that the KNE method significantly improves the accuracy of knowledge editing and achieves, or even exceeds, the performance of the best baseline methods in portability and locality metrics. 

**Abstract (ZH)**: 随着现实世界知识的不断演变，确保模型知识的时效性和准确性至关重要。这使得在大型语言模型中进行知识编辑变得愈加重要。然而，现有的知识编辑方法面临一些挑战，包括参数定位耦合、定位不精确以及层间缺乏动态交互。在这篇论文中，我们提出了一种名为Knowledge Neuronal Ensemble (KNE)的新颖知识编辑方法。Knowledge Neuronal Ensemble 代表一组编码特定知识的神经元，从而减轻了由于参数定位耦合而导致频繁修改参数的问题。KNE 方法通过在每一层为每个参数计算梯度归因得分来增强参数定位的精度和准确性。在编辑过程中，仅计算与知识神经元集合相关的梯度和损失，并相应地进行误差反向传播，从而确保参数间的动态交互和协作更新。在三个广泛使用的知识编辑数据集上的实验结果表明，KNE 方法显著提高了知识编辑的准确性，并且在可移植性和局部性指标上甚至超过了最佳基线方法的性能。 

---
# NLP-based Regulatory Compliance -- Using GPT 4.0 to Decode Regulatory Documents 

**Title (ZH)**: 基于NLP的合规性监管——使用GPT 4.0解析监管文件 

**Authors**: Bimal Kumar, Dmitri Roussinov  

**Link**: [PDF](https://arxiv.org/pdf/2412.20602)  

**Abstract**: Large Language Models (LLMs) such as GPT-4.0 have shown significant promise in addressing the semantic complexities of regulatory documents, particularly in detecting inconsistencies and contradictions. This study evaluates GPT-4.0's ability to identify conflicts within regulatory requirements by analyzing a curated corpus with artificially injected ambiguities and contradictions, designed in collaboration with architects and compliance engineers. Using metrics such as precision, recall, and F1 score, the experiment demonstrates GPT-4.0's effectiveness in detecting inconsistencies, with findings validated by human experts. The results highlight the potential of LLMs to enhance regulatory compliance processes, though further testing with larger datasets and domain-specific fine-tuning is needed to maximize accuracy and practical applicability. Future work will explore automated conflict resolution and real-world implementation through pilot projects with industry partners. 

**Abstract (ZH)**: 大型语言模型（LLMs），如GPT-4.0，在解决监管文件中的语义复杂性方面展现出了显著的潜力，尤其是在检测矛盾和不一致方面。本研究通过分析一个包含人工注入的模糊性和矛盾的精心策划语料库，评估了GPT-4.0识别监管要求内冲突的能力。该语料库由架构师和合规工程师合作设计。通过精确度、召回率和F1分数等指标，实验展示了GPT-4.0在检测不一致性方面的有效性，并通过人类专家的验证得到了证实。研究结果强调了LLMs在增强监管合规流程方面的潜力，尽管还需要在更大数据集上进行进一步测试，并进行领域特定的微调，以最大化准确性和实际应用性。未来的研究将通过与行业合作伙伴开展试点项目，探索自动冲突解决和实际应用。 

---
# Enhancing Entertainment Translation for Indian Languages using Adaptive Context, Style and LLMs 

**Title (ZH)**: 使用适应性上下文、风格和大规模语言模型增强印度语言的娱乐翻译 

**Authors**: Pratik Rakesh Singh, Mohammadi Zaki, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2412.20440)  

**Abstract**: We address the challenging task of neural machine translation (NMT) in the entertainment domain, where the objective is to automatically translate a given dialogue from a source language content to a target language. This task has various applications, particularly in automatic dubbing, subtitling, and other content localization tasks, enabling source content to reach a wider audience. Traditional NMT systems typically translate individual sentences in isolation, without facilitating knowledge transfer of crucial elements such as the context and style from previously encountered sentences. In this work, we emphasize the significance of these fundamental aspects in producing pertinent and captivating translations. We demonstrate their significance through several examples and propose a novel framework for entertainment translation, which, to our knowledge, is the first of its kind. Furthermore, we introduce an algorithm to estimate the context and style of the current session and use these estimations to generate a prompt that guides a Large Language Model (LLM) to generate high-quality translations. Our method is both language and LLM-agnostic, making it a general-purpose tool. We demonstrate the effectiveness of our algorithm through various numerical studies and observe significant improvement in the COMET scores over various state-of-the-art LLMs. Moreover, our proposed method consistently outperforms baseline LLMs in terms of win-ratio. 

**Abstract (ZH)**: 我们致力于娱乐领域中的神经机器翻译（NMT）这一具有挑战性的任务，目标是自动将给定的对话从源语言翻译为目标语言。这一任务在自动配音、字幕制作以及其他内容本地化任务中有着广泛的应用，使源内容能够触及更广泛的受众。传统的NMT系统通常独立地翻译个别句子，而没有将上文和风格等关键信息传递给后续的翻译过程。在本项研究中，我们强调了上下文和风格等基础要素对于生成相关且引人入胜的翻译的重要性。我们通过多个示例展示了这些要素的重要性，并提出了一个新颖的娱乐翻译框架，据我们所知，这是首款此类系统。此外，我们介绍了一种算法来估算当前会话的上下文和风格，并利用这些估计生成一个提示，指导大型语言模型（LLM）生成高质量的翻译。该方法既无需依赖特定语言，也无需依赖特定的LLM，因此具有通用性。我们通过多种数值研究验证了该算法的有效性，并观察到我们的方法在多种最先进的LLM上的COMET评分上显著提高。此外，我们的方法在胜败比上也持续优于基线LLM。 

---
# Multi-Objective Large Language Model Unlearning 

**Title (ZH)**: 多目标大型语言模型去学习 

**Authors**: Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.20412)  

**Abstract**: Machine unlearning in the domain of large language models (LLMs) has attracted great attention recently, which aims to effectively eliminate undesirable behaviors from LLMs without full retraining from scratch. In this paper, we explore the Gradient Ascent (GA) approach in LLM unlearning, which is a proactive way to decrease the prediction probability of the model on the target data in order to remove their influence. We analyze two challenges that render the process impractical: gradient explosion and catastrophic forgetting. To address these issues, we propose Multi-Objective Large Language Model Unlearning (MOLLM) algorithm. We first formulate LLM unlearning as a multi-objective optimization problem, in which the cross-entropy loss is modified to the unlearning version to overcome the gradient explosion issue. A common descent update direction is then calculated, which enables the model to forget the target data while preserving the utility of the LLM. Our empirical results verify that MoLLM outperforms the SOTA GA-based LLM unlearning methods in terms of unlearning effect and model utility preservation. 

**Abstract (ZH)**: 在大型语言模型（LLMs）领域的机器遗忘问题最近引起了广泛的关注，其目标是在不完全从头开始重新训练的情况下，有效消除大型语言模型中的不良行为。本文探讨了在LLMs遗忘中应用梯度上升（GA）方法，这是一种主动的手段，旨在通过降低模型对目标数据的预测概率来减少其影响。我们分析了导致这一过程不可行的两个挑战：梯度爆炸和灾难性遗忘。为了解决这些问题，我们提出了多目标大型语言模型遗忘算法（MOLLM）。我们首先将LLMs遗忘问题形式化为一个多目标优化问题，在其中通过修改交叉熵损失为遗忘版本来克服梯度爆炸问题。然后计算了一个共同的下降更新方向，使模型能够忘记目标数据同时保留大型语言模型的有用性。我们的实证结果验证了MOLLM在遗忘效果和模型有用性保留方面优于现有的基于GA的大型语言模型遗忘方法。 

---
# LLM2: Let Large Language Models Harness System 2 Reasoning 

**Title (ZH)**: LLM2：让大型语言模型运用系统二推理

注释：在翻译学术术语时，通常会保持与原文相近的缩写形式。"System 2 Reasoning" 是心理学中描述深入、耗时和逻辑化思考过程的概念，在翻译时，直接翻译为“系统二推理”更为准确和规范。 

**Authors**: Cheng Yang, Chufan Shi, Siheng Li, Bo Shui, Yujiu Yang, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2412.20372)  

**Abstract**: Large language models (LLMs) have exhibited impressive capabilities across a myriad of tasks, yet they occasionally yield undesirable outputs. We posit that these limitations are rooted in the foundational autoregressive architecture of LLMs, which inherently lacks mechanisms for differentiating between desirable and undesirable results. Drawing inspiration from the dual-process theory of human cognition, we introduce LLM2, a novel framework that combines an LLM (System 1) with a process-based verifier (System 2). Within LLM2, the LLM is responsible for generating plausible candidates, while the verifier provides timely process-based feedback to distinguish desirable and undesirable outputs. The verifier is trained with a pairwise comparison loss on synthetic process-supervision data generated through our token quality exploration strategy. Empirical results on mathematical reasoning benchmarks substantiate the efficacy of LLM2, exemplified by an accuracy enhancement from 50.3 to 57.8 (+7.5) for Llama3-1B on GSM8K. Furthermore, when combined with self-consistency, LLM2 achieves additional improvements, boosting major@20 accuracy from 56.2 to 70.2 (+14.0). 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中展现了令人 impressive 的能力，但在某些情况下也会产生不良输出。我们认为这些限制源于LLMs的自回归基础架构，这种架构本身就缺乏区分有利和不利结果的机制。受到人类认知的双过程理论的启发，我们提出了一种名为LLM2的新框架，该框架将一个LLM（系统1）与基于过程的验证器（系统2）结合在一起。在LLM2中，LLM负责生成合理的候选答案，而验证器则提供及时的过程反馈，以区分有利和不利的输出。验证器通过我们的令牌质量探索策略生成的合成过程监督数据进行训练，并使用成对比较损失函数进行训练。数学推理基准数据集上的实验证明了LLM2的有效性，例如LLAMA3-1B在GSM8K上的准确性从50.3提高到了57.8（+7.5）。此外，当与自我一致性结合使用时，LLM2还实现了额外的改进，主要@20准确性从56.2提高到了70.2（+14.0）。 

---
# HindiLLM: Large Language Model for Hindi 

**Title (ZH)**: HindiLLM：印地语大型语言模型 

**Authors**: Sanjay Chouhan, Shubha Brata Nath, Aparajita Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2412.20357)  

**Abstract**: The advancements in the Large Language Model (LLM) have helped in solving several problems related to language processing. Most of the researches have focused on the English language only, because of its popularity and abundance on the internet. However, a high-performance language model for Hindi and other Indic languages is lacking in the literature. In this work, we have pre-trained two autoregressive LLM models for the Hindi language, namely HindiLLM-Small and HindiLLM-Medium. We use a two-step process comprising unsupervised pre-training and supervised fine-tuning. First, we create a large and high-quality text corpus for unsupervised pre-training. Next, we train a Byte-Pair Encoding, named HindiLLM tokenizer, using the pre-training text data. We then perform training on the unlabeled data, known as the pre-training step, to get the HindiLLM base models. Furthermore, we perform fine-tuning of the HindiLLM base models for different tasks like sentiment analysis, text classification, natural language inference, and multiple choice question-answer on popular labeled datasets to measure the real-world performance. The evaluation shows that the HindiLLM-based fine-tuned models outperform several models in most of the language related tasks. 

**Abstract (ZH)**: 大型语言模型（LLM）的进步在解决语言处理相关问题方面发挥了重要作用。大多数研究集中在英语上，因为英语因其在互联网上的流行性和丰富性而备受关注。然而，关于孟买语和其他印度语言的高性能语言模型的文献仍然不足。在这项工作中，我们预训练了两个自回归LLM模型，分别为HindiLLM-Small和HindiLLM-Medium，专门用于孟买语。我们采用两步过程，包括无监督预训练和有监督微调。首先，我们创建了一个大型和高质量的文本语料库用于无监督预训练。接着，我们利用预训练文本数据训练了一个名为HindiLLM的字节对编码器。然后，我们在未标记的数据上进行训练，即预训练步骤，以生成HindiLLM基础模型。此外，我们对HindiLLM基础模型进行了针对不同任务的微调，包括情感分析、文本分类、自然语言推理和多项选择问题回答，这些建立在流行的标记数据集之上，以衡量其实际性能。评估结果显示，基于HindiLLM的微调模型在大多数语言相关任务中超越了多种模型。 

---
# Scoring with Large Language Models: A Study on Measuring Empathy of Responses in Dialogues 

**Title (ZH)**: 使用大型语言模型打分：关于对话中回应同理心度量的研究 

**Authors**: Henry J. Xie, Jinghan Zhang, Xinhao Zhang, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20264)  

**Abstract**: In recent years, Large Language Models (LLMs) have become increasingly more powerful in their ability to complete complex tasks. One such task in which LLMs are often employed is scoring, i.e., assigning a numerical value from a certain scale to a subject. In this paper, we strive to understand how LLMs score, specifically in the context of empathy scoring. We develop a novel and comprehensive framework for investigating how effective LLMs are at measuring and scoring empathy of responses in dialogues, and what methods can be employed to deepen our understanding of LLM scoring. Our strategy is to approximate the performance of state-of-the-art and fine-tuned LLMs with explicit and explainable features. We train classifiers using various features of dialogues including embeddings, the Motivational Interviewing Treatment Integrity (MITI) Code, a set of explicit subfactors of empathy as proposed by LLMs, and a combination of the MITI Code and the explicit subfactors. Our results show that when only using embeddings, it is possible to achieve performance close to that of generic LLMs, and when utilizing the MITI Code and explicit subfactors scored by an LLM, the trained classifiers can closely match the performance of fine-tuned LLMs. We employ feature selection methods to derive the most crucial features in the process of empathy scoring. Our work provides a new perspective toward understanding LLM empathy scoring and helps the LLM community explore the potential of LLM scoring in social science studies. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在完成复杂任务的能力上越来越强大。其中一个这样的任务是评分，即为某个主题分配一个特定范围内的数值。在本文中，我们致力于理解LLMs如何进行评分，特别是在同理心评分方面的应用。我们开发了一个新颖且全面的框架，以研究有效LLMs在测量和评分对话中响应同理心方面的效果，以及可以采用哪些方法来深化我们对LLM评分的理解。我们的策略是通过显式和可解释的特征来近似先进和微调后的LLM性能。我们使用对话的各种特征（包括嵌入、动机访谈治疗完整性代码（MITI Code）、LLMs提出的显式子因素集，以及MITI Code和显式子因素的组合）来训练分类器。结果显示，仅使用嵌入时，可以实现接近通用LLM的性能；而利用MITI Code和由LLM评分的显式子因素时，训练后的分类器可以接近微调LLM的性能。我们采用特征选择方法来确定同理心评分过程中最关键的因素。我们的研究为理解LLM同理心评分提供了新的视角，并有助于LLM社区探索LLM评分在社会科学研究中的潜力。 

---
# ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty 

**Title (ZH)**: ComparisonQA：通过知识频率控制和不确定性评估LLMs的事实可靠性 

**Authors**: Qing Zong, Zhaowei Wang, Tianshi Zheng, Xiyu Ren, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.20251)  

**Abstract**: The rapid development of LLMs has sparked extensive research into their factual knowledge. Current works claim that LLMs fall short on questions requiring less frequent knowledge. However, their proof is incomplete since they only study the influence of entity frequency, which can not fully represent knowledge frequency. So we introduce ComparisonQA benchmark, containing 283K abstract questions, each instantiated by a pair of high-frequency and low-frequency entities. It ensures a controllable comparison because the difference of knowledge frequency between such a pair is only related to entity frequency. In addition, to avoid possible semantic shortcuts, which is a severe problem of current LLMs study, we design a two-round method for knowledge robustness measurement utilizing both correctness and uncertainty. Experiments reveal that LLMs exhibit particularly low robustness regarding low-frequency knowledge, and GPT-4o is even the worst under this measurement. Besides, we introduce an automatic method to filter out questions with low-quality and shortcuts to form ComparisonQA-Hard. We find that uncertainty effectively identifies such questions while maintaining the data size. 

**Abstract (ZH)**: 语言模型（LLM）的快速发展激发了对其事实性知识的研究。现有研究表明，LLM 在要求较少出现知识的问题上表现不佳。然而，这些结论尚不完整，因为它们仅研究了实体频率的影响，未能全面反映知识频率。因此，我们引入了 ComparisonQA 基准数据集，包含 28.3 万条抽象问题，每条问题由一对高频率和低频率的实体实例化。这确保了比较的可控性，因为这种对实体频率的不同频率的知识影响是唯一的。此外，为了避免现有 LLM 研究中的潜在语义捷径问题，我们设计了一种两轮的方法，利用正确性和不确定性对知识稳健性进行测量。实验结果表明，LLM 在低频率知识上的稳健性特别低，GPT-4o 在这种测量下表现最差。此外，我们还引入了一种自动方法来筛选出低质量及捷径问题，从而形成了 ComparisonQA-Hard 数据集。我们发现，不确定性有效识别了这些问题，同时保持了数据集的规模。 

---
# LLM Reasoning Engine: Specialized Training for Enhanced Mathematical Reasoning 

**Title (ZH)**: LLM推理引擎：专门培训以增强数学推理能力 

**Authors**: Shuguang Chen, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.20227)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in various natural language processing tasks but face challenges in mathematical reasoning, where complex problem-solving requires both linguistic understanding and mathematical reasoning skills. Existing approaches to address this challenge often rely on ensemble methods and suffer from the problem of data scarcity in target domains. In this work, we present a novel method to enhance LLMs' capabilities in mathematical reasoning tasks. Motivated by the need to bridge this gap, our approach incorporates a question paraphrase strategy, which aims at diversifying the linguistic forms of mathematical questions to improve generalization. Additionally, specialized training objectives are employed to guide the model's learning process, focusing on enhancing its understanding of mathematical concepts and reasoning processes. We conduct experiments on four datasets using different LLMs, and demonstrate the effectiveness of our approach in improving LLMs' performance on mathematical reasoning tasks. Our findings underscore the significance of our methodology in the advancement of large language models and its potential implications for real-world applications that require mathematical reasoning abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理任务中表现出色，但在数学推理方面遇到了挑战，因为复杂的解决问题不仅需要语言理解能力，还需要数学推理能力。现有的解决这一挑战的方法通常依赖于集成方法，并且在目标领域面临数据稀缺的问题。在本项工作中，我们提出了一种新的方法来增强LLMs在数学推理任务中的能力。鉴于这一差距的需求，我们的方法采用了问题重述策略，旨在通过多样化数学问题的表达形式来提高模型的泛化能力。此外，我们还采用了专门的训练目标来指导模型的学习过程，重点在于增强其对数学概念和推理过程的理解。我们使用不同的LLMs在四个数据集上进行了实验，并展示了我们的方法在提高LLMs在数学推理任务中的性能方面的有效性。我们的研究结果强调了该方法在大型语言模型发展中的重要性及其在需要数学推理能力的现实应用中的潜在影响。 

---
# Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Dataset 

**Title (ZH)**: 利用大语言模型提取混合长文档信息：一个框架与数据集 

**Authors**: Chongjian Yue, Xinrun Xu, Xiaojun Ma, Lun Du, Zhiming Ding, Shi Han, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20072)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional performance in textual understanding and tabular reasoning tasks. However, their ability to comprehend and analyze hybrid text, containing textual and tabular data, remains unexplored. The hybrid text often appears in the form of hybrid long documents (HLDs), which far exceed the token limit of LLMs. Consequently, we apply an Automated Information Extraction framework (AIE) to enable LLMs to process the HLDs and carry out experiments to analyse four important aspects of information extraction from HLDs. Given the findings: 1) The effective way to select and summarize the useful part of a HLD. 2) An easy table serialization way is enough for LLMs to understand tables. 3) The naive AIE has adaptability in many complex scenarios. 4) The useful prompt engineering to enhance LLMs on HLDs. To address the issue of dataset scarcity in HLDs and support future work, we also propose the Financial Reports Numerical Extraction (FINE) dataset. The dataset and code are publicly available in the attachments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本理解和表格推理任务中表现出色。然而，它们在理解并分析包含文本和表格数据的混合文本方面的能力尚未得到探索。混合文本通常以混合长文档（HLDs）的形式出现，这远超LLM的令牌限制。因此，我们应用了自动信息提取框架（AIE），使LLM能够处理HLDs，并进行实验以分析信息从HLDs中提取的四个重要方面。根据研究发现：1）有效选择和总结HLD中有用部分的方法。2）简单的表格序列化方式足以让LLM理解表格。3）原始AIE在许多复杂场景中具有适应性。4）有用的提示工程可以增强LLM在处理HLDs方面的表现。为了解决HLDs数据集稀缺的问题，同时也为了支持未来的研究，我们还提出了金融报告数值提取（FINE）数据集。数据集和代码已在附件中公开提供。 

---
# Comparative Analysis of Listwise Reranking with Large Language Models in Limited-Resource Language Contexts 

**Title (ZH)**: 在资源有限的语言环境中，基于大型语言模型的列表级重排序比较分析 

**Authors**: Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du, Yiyi Tao, Yixian Shen, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20061)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant effectiveness across various NLP tasks, including text ranking. This study assesses the performance of large language models (LLMs) in listwise reranking for limited-resource African languages. We compare proprietary models RankGPT3.5, Rank4o-mini, RankGPTo1-mini and RankClaude-sonnet in cross-lingual contexts. Results indicate that these LLMs significantly outperform traditional baseline methods such as BM25-DT in most evaluation metrics, particularly in nDCG@10 and MRR@100. These findings highlight the potential of LLMs in enhancing reranking tasks for low-resource languages and offer insights into cost-effective solutions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在包括文本排序在内的各种自然语言处理（NLP）任务中已显示出显著的效果。本研究评估了大型语言模型在有限资源下的非洲语言列表级重排序中的性能。我们在此跨语言背景下比较了自有的模型RankGPT3.5、Rank4o-mini、RankGPTo1-mini和RankClaude-sonnet的表现。结果表明，这些LLMs在大多数评估指标中显著优于传统的基线方法（如BM25-DT），特别是在nDCG@10和MRR@100方面。这些发现凸显了LLMs在提升低资源语言重排序任务方面的潜在价值，并提供了成本效益高的解决方案。 

---
# "My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise 

**Title (ZH)**: “我的生活 very 悲惨，每天必须签名 500 次”：揭露隐藏自夸，一种伪装的自谦方式

或者更正式的翻译：

“我的生活非常悲惨，每天必须签署 500 份签名”：揭示隐藏自夸，一种伪装的自谦现象

这种翻译不仅传达了原文的意思，还符合学术写作的标准和规范。 

**Authors**: Sharath Naganna, Saprativa Bhattacharjee, Pushpak Bhattacharyya, Biplab Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2412.20057)  

**Abstract**: Humblebragging is a phenomenon where individuals present self-promotional statements under the guise of modesty or complaints. For example, a statement like, "Ugh, I can't believe I got promoted to lead the entire team. So stressful!", subtly highlights an achievement while pretending to be complaining. Detecting humblebragging is important for machines to better understand the nuances of human language, especially in tasks like sentiment analysis and intent recognition. However, this topic has not yet been studied in computational linguistics. For the first time, we introduce the task of automatically detecting humblebragging in text. We formalize the task by proposing a 4-tuple definition of humblebragging and evaluate machine learning, deep learning, and large language models (LLMs) on this task, comparing their performance with humans. We also create and release a dataset called HB24, containing 3,340 humblebrags generated using GPT-4o. Our experiments show that detecting humblebragging is non-trivial, even for humans. Our best model achieves an F1-score of 0.88. This work lays the foundation for further exploration of this nuanced linguistic phenomenon and its integration into broader natural language understanding systems. 

**Abstract (ZH)**: 埋怨自夸是一种现象，指个体以谦虚或抱怨的面目呈现自我宣传的内容。例如，“唉，我简直不敢相信我被提升为整个团队的领导。真是太有压力了！”这种表述在看似抱怨的同时微妙地突显了其成就。检测埋怨自夸对于机器更好地理解人类语言的细微差别至关重要，特别是在情感分析和意图识别等任务中。然而，这一话题在计算语言学中尚未得到研究。在此背景下，我们首次引入了自动检测文本中埋怨自夸的任务。我们通过提出四元组定义来形式化这一任务，并评估机器学习、深度学习和大规模语言模型（LLMs）在这一任务上的性能，将它们的表现与人类进行比较。我们还创建并发布了名为HB24的数据集，其中包含使用GPT-4o生成的3,340条埋怨自夸的语料。我们的实验表明，即使是人类，检测埋怨自夸也并非易事。我们模型的最佳性能达到了F1分数0.88。这项工作为深入探讨这一微妙的语言现象及其在更广泛自然语言理解系统中的应用奠定了基础。 

---
# Assessing Text Classification Methods for Cyberbullying Detection on Social Media Platforms 

**Title (ZH)**: 评估文本分类方法在社交媒体平台上的网络霸凌检测效果 

**Authors**: Adamu Gaston Philipo, Doreen Sebastian Sarwatt, Jianguo Ding, Mahmoud Daneshmand, Huansheng Ning  

**Link**: [PDF](https://arxiv.org/pdf/2412.19928)  

**Abstract**: Cyberbullying significantly contributes to mental health issues in communities by negatively impacting the psychology of victims. It is a prevalent problem on social media platforms, necessitating effective, real-time detection and monitoring systems to identify harmful messages. However, current cyberbullying detection systems face challenges related to performance, dataset quality, time efficiency, and computational costs. This research aims to conduct a comparative study by adapting and evaluating existing text classification techniques within the cyberbullying detection domain. The study specifically evaluates the effectiveness and performance of these techniques in identifying cyberbullying instances on social media platforms. It focuses on leveraging and assessing large language models, including BERT, RoBERTa, XLNet, DistilBERT, and GPT-2.0, for their suitability in this domain. The results show that BERT strikes a balance between performance, time efficiency, and computational resources: Accuracy of 95%, Precision of 95%, Recall of 95%, F1 Score of 95%, Error Rate of 5%, Inference Time of 0.053 seconds, RAM Usage of 35.28 MB, CPU/GPU Usage of 0.4%, and Energy Consumption of 0.000263 kWh. The findings demonstrate that generative AI models, while powerful, do not consistently outperform fine-tuned models on the tested benchmarks. However, state-of-the-art performance can still be achieved through strategic adaptation and fine-tuning of existing models for specific datasets and tasks. 

**Abstract (ZH)**: 网络欺凌对社区的心理健康问题有显著影响，通过负面地影响受害者的心理状态。它在社交媒体平台上是一个普遍存在的问题，需要有效的实时检测和监控系统来识别有害信息。然而，当前的网络欺凌检测系统在性能、数据集质量、时间效率和计算成本方面存在挑战。本研究旨在通过适应和评估现有的文本分类技术来开展一项比较研究，这些技术在网络安全欺凌检测领域中具有潜在应用。该研究特别评估了这些技术在社交媒体平台上识别网络欺凌实例的有效性和性能。研究重点在于利用和评估大型语言模型，包括BERT、RoBERTa、XLNet、DistilBERT和GPT-2.0，以确定其在该领域的适用性。研究结果表明，BERT在性能、时间效率和计算资源之间取得了平衡：准确率为95%，精确率为95%，召回率为95%，F1分数为95%，错误率为5%，推理时间为0.053秒，RAM使用量为35.28 MB，CPU/GPU使用率仅为0.4%，能耗为0.000263 kWh。研究发现，尽管生成式人工智能模型功能强大，但在测试基准上并不总是优于微调模型。然而，仍可以通过有针对性地适应和微调现有模型以适应特定数据集和任务来实现最新性能。 

---
# Right vs. Right: Can LLMs Make Tough Choices? 

**Title (ZH)**: “正确” vs. “正确”：LLM能够作出艰难选择吗？ 

**Authors**: Jiaqing Yuan, Pradeep K. Murukannaiah, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2412.19926)  

**Abstract**: An ethical dilemma describes a choice between two "right" options involving conflicting moral values. We present a comprehensive evaluation of how LLMs navigate ethical dilemmas. Specifically, we investigate LLMs on their (1) sensitivity in comprehending ethical dilemmas, (2) consistency in moral value choice, (3) consideration of consequences, and (4) ability to align their responses to a moral value preference explicitly or implicitly specified in a prompt. Drawing inspiration from a leading ethical framework, we construct a dataset comprising 1,730 ethical dilemmas involving four pairs of conflicting values. We evaluate 20 well-known LLMs from six families. Our experiments reveal that: (1) LLMs exhibit pronounced preferences between major value pairs, and prioritize truth over loyalty, community over individual, and long-term over short-term considerations. (2) The larger LLMs tend to support a deontological perspective, maintaining their choices of actions even when negative consequences are specified. (3) Explicit guidelines are more effective in guiding LLMs' moral choice than in-context examples. Lastly, our experiments highlight the limitation of LLMs in comprehending different formulations of ethical dilemmas. 

**Abstract (ZH)**: 伦理困境是指在涉及冲突道德价值观的情况下，需要在两个“正确”的选择之间做出的抉择。本文对大型语言模型（LLM）在处理伦理困境时的路径进行了全方位评估。具体来说，我们考察了LLM在以下方面的表现：（1）对伦理困境的理解敏感度，（2）在道德价值选择上的一致性，（3）对后果的考虑，以及（4）将回应与明示或隐含在提示中指定的道德价值观偏好相一致的能力。借鉴领先伦理框架的灵感，我们构建了一个包含1730个伦理困境的数据集，这些困境涉及四种冲突价值的两两组合。我们评估了来自六个家庭的20个知名LLM。实验结果显示：（1）LLM在主要价值对之间表现出明显的偏好，并倾向于优先选择真实性而非忠诚度，集体利益而非个体利益，长期利益而非短期利益。（2）较大的LLM倾向于坚持原则性视角，在负面后果被指明的情况下仍保持其行为选择。（3）明确的指导方针比上下文中的例子在引导LLM的道德选择方面更为有效。最后，我们的实验揭示了LLM在理解不同表述形式的伦理困境方面的局限性。 

---
# Evaluate Summarization in Fine-Granularity: Auto Evaluation with LLM 

**Title (ZH)**: 在细粒度层面评估摘要生成：基于大语言模型的自动评估 

**Authors**: Dong Yuan, Eti Rastogi, Fen Zhao, Sagar Goyal, Gautam Naik, Sree Prasanna Rajagopal  

**Link**: [PDF](https://arxiv.org/pdf/2412.19906)  

**Abstract**: Due to the exponential growth of information and the need for efficient information consumption the task of summarization has gained paramount importance. Evaluating summarization accurately and objectively presents significant challenges, particularly when dealing with long and unstructured texts rich in content. Existing methods, such as ROUGE (Lin, 2004) and embedding similarities, often yield scores that have low correlation with human judgements and are also not intuitively understandable, making it difficult to gauge the true quality of the summaries. LLMs can mimic human in giving subjective reviews but subjective scores are hard to interpret and justify. They can be easily manipulated by altering the models and the tones of the prompts. In this paper, we introduce a novel evaluation methodology and tooling designed to address these challenges, providing a more comprehensive, accurate and interpretable assessment of summarization outputs. Our method (SumAutoEval) proposes and evaluates metrics at varying granularity levels, giving objective scores on 4 key dimensions such as completeness, correctness, Alignment and readability. We empirically demonstrate, that SumAutoEval enhances the understanding of output quality with better human correlation. 

**Abstract (ZH)**: 由于信息的指数级增长和高效信息消费的需要，总结任务的重要性日益凸显。准确和客观地评估总结是一项重大挑战，特别是在处理长且结构不规则、内容丰富的文本时。现有方法，如 ROUGE（Lin, 2004）和嵌入相似性，通常给出的评分与人类判断的相关性较低，且缺乏直观性，使得难以评估总结的真实质量。尽管大规模语言模型（LLMs）可以模拟人类给出的主观评价，但主观评分难以解释和验证，并且容易通过调整模型和提示的语气被操控。在本文中，我们提出了一种新的评估方法和工具，旨在解决上述挑战，提供更全面、准确和可解释的总结输出评估。我们的方法（SumAutoEval）提出了在不同粒度级别评估指标，并在四大关键维度（完整性、准确性、一致性与可读性）上给出客观评分。我们通过实证研究证明，SumAutoEval 提高了输出质量理解的直观性和与人类判断的相关性。 

---
# Distributed Mixture-of-Agents for Edge Inference with Large Language Models 

**Title (ZH)**: 基于边缘推理的大规模语言模型混合代理分布式系统 

**Authors**: Purbesh Mitra, Priyanka Kaswan, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2412.21200)  

**Abstract**: Mixture-of-Agents (MoA) has recently been proposed as a method to enhance performance of large language models (LLMs), enabling multiple individual LLMs to work together for collaborative inference. This collaborative approach results in improved responses to user prompts compared to relying on a single LLM. In this paper, we consider such an MoA architecture in a distributed setting, where LLMs operate on individual edge devices, each uniquely associated with a user and equipped with its own distributed computing power. These devices exchange information using decentralized gossip algorithms, allowing different device nodes to talk without the supervision of a centralized server. In the considered setup, different users have their own LLM models to address user prompts. Additionally, the devices gossip either their own user-specific prompts or augmented prompts to generate more refined answers to certain queries. User prompts are temporarily stored in the device queues when their corresponding LLMs are busy. Given the memory limitations of edge devices, it is crucial to ensure that the average queue sizes in the system remain bounded. In this paper, we address this by theoretically calculating the queuing stability conditions for the device queues under reasonable assumptions, which we validate experimentally as well. Further, we demonstrate through experiments, leveraging open-source LLMs for the implementation of distributed MoA, that certain MoA configurations produce higher-quality responses compared to others, as evaluated on AlpacaEval 2.0 benchmark. The implementation is available at: this https URL. 

**Abstract (ZH)**: 以下是对这段内容的翻译，符合学术规范：

Mixture-of-Agents (MoA) 近期被提出作为一种增强大规模语言模型 (LLMs) 性能的方法，使多个独立的 LLM 能够协同工作进行联合推理。这种协同方法使得用户提示的响应效果优于仅依赖单一 LLM。在本文中，我们考虑在分布式环境中 MoA 架构的应用，其中 LLM 在个体边缘设备上运行，每台设备都唯一关联于一个用户，并配备了独立的分布式计算能力。这些设备通过去中心化的闲聊算法 (gossip algorithms) 交换信息，允许不同的设备节点在无需中央服务器监督的情况下相互交流。在这个设置中，不同用户都有各自的 LLM 模型来处理用户提示。此外，设备相互闲聊时可能传播各自特定用户的提示或增强后的提示，以生成更精细的回答来解决特定查询。当用户提示对应的 LLM 正忙时，这些提示将被暂时存储在设备队列中。鉴于边缘设备的内存限制，确保系统中平均队列大小保持在界限内至关重要。本文通过在合理假设下理论计算设备队列的排队稳定条件，并通过实验验证了这些条件。此外，我们通过实验展示了，在利用开源 LLM 实现分布式 MoA 的情况下，某些 MoA 配置相比其他配置在 AlpacaEval 2.0 基准上的响应质量更高。该实现可参见：this [网站链接]。

请注意将"this https URL"替换为实际的访问链接。 

---
# HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking Code Generation 

**Title (ZH)**: HumanEval Pro 和 MBPP Pro：评估大型语言模型在自我调用代码生成任务上的性能 

**Authors**: Zhaojian Yu, Yilun Zhao, Arman Cohan, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.21199)  

**Abstract**: We introduce self-invoking code generation, a new task designed to evaluate the progressive reasoning and problem-solving capabilities of LLMs. In this task, models are presented with a base problem and a related, more complex problem. They must solve the base problem and then utilize its solution to address the more complex one. This work features three key contributions. First, we propose a general recipe for generating more challenging versions of existing benchmarks, resulting in three new benchmarks: HumanEval Pro, MBPP Pro, and BigCodeBench-Lite Pro, specifically designed to assess LLMs on self-invoking code generation. Second, from the analysis of experimental results over twenty LLMs on our benchmarks, we have two important observations: (i) Most LLMs excel in traditional code generation benchmarks like HumanEval and MBPP, but their performance declines on self-invoking tasks. For example, o1-mini achieves 96.2% pass@1 on HumanEval but only 76.2% on HumanEval Pro. (ii) On self-invoking code generation task, the instruction-tuned models demonstrate only marginal improvements compared to the base models. Third, we disclose the types of failure modes that exist in our evaluation results. All these results underscore the need for further advancements in self-invoking code generation tasks and provide a new direction for future research on enhancing LLMs' code reasoning capabilities. 

**Abstract (ZH)**: 我们将介绍自我调用代码生成，这是一个新的任务，旨在评估大型语言模型（LLMs）的渐进推理和问题解决能力。在这个任务中，模型首先面对一个基础问题，然后是与其相关但更复杂的另一个问题。它们必须先解决基础问题，然后利用其解决方案来解决更复杂的问题。本研究包含三个主要贡献。首先，我们提出了一种通用方法，用于生成现有基准测试的更具挑战性的版本，从而产生了三个新的基准测试：HumanEval Pro、MBPP Pro 和 BigCodeBench-Lite Pro，专门用于评估LLMs在自我调用代码生成方面的表现。其次，通过对我们的基准测试中20种LLMs的实验结果进行分析，我们发现了两个重要的观察结果：（i）大多数LLMs在传统的代码生成基准测试（如HumanEval和MBPP）中表现出色，但在自我调用任务中的表现则较差。例如，o1-mini在HumanEval上的pass@1得分为96.2%，但在HumanEval Pro上的得分为76.2%。（ii）在自我调用代码生成任务中，指令调优模型相对于基础模型仅显示出微小的改进。最后，我们披露了我们评估结果中存在的一些失败模式。所有这些结果强调了在自我调用代码生成任务方面进一步发展的必要性，并为增强LLMs的代码推理能力提供了新的研究方向。 

---
# Aviary: training language agents on challenging scientific tasks 

**Title (ZH)**: Aviary：在具有挑战性的科学任务中训练语言代理 

**Authors**: Siddharth Narayanan, James D. Braza, Ryan-Rhys Griffiths, Manu Ponnapati, Albert Bou, Jon Laurent, Ori Kabeli, Geemi Wellawatte, Sam Cox, Samuel G. Rodriques, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2412.21154)  

**Abstract**: Solving complex real-world tasks requires cycles of actions and observations. This is particularly true in science, where tasks require many cycles of analysis, tool use, and experimentation. Language agents are promising for automating intellectual tasks in science because they can interact with tools via natural language or code. Yet their flexibility creates conceptual and practical challenges for software implementations, since agents may comprise non-standard components such as internal reasoning, planning, tool usage, as well as the inherent stochasticity of temperature-sampled language models. Here, we introduce Aviary, an extensible gymnasium for language agents. We formalize agents as policies solving language-grounded partially observable Markov decision processes, which we term language decision processes. We then implement five environments, including three challenging scientific environments: (1) manipulating DNA constructs for molecular cloning, (2) answering research questions by accessing scientific literature, and (3) engineering protein stability. These environments were selected for their focus on multi-step reasoning and their relevance to contemporary biology research. Finally, with online training and scaling inference-time compute, we show that language agents backed by open-source, non-frontier LLMs can match and exceed both frontier LLM agents and human experts on multiple tasks at up to 100x lower inference cost. 

**Abstract (ZH)**: 解决复杂的实际任务需要一系列的动作和观察。这一点在科学领域表现尤为明显，因为科学任务通常需要多次分析、工具使用和实验循环。语言代理在科学领域自动化智力任务方面具有巨大潜力，因为它们可以通过自然语言或代码与工具进行交互。然而，语言代理的灵活性也给软件实现带来了概念和实践上的挑战，因为这些代理可能包括非标准组件，如内部推理、规划、工具使用以及基于温度采样的语言模型固有的随机性。在此背景下，我们介绍了Aviary，一个灵活的语言代理实验平台。我们将代理精确定义为解决语言驱动的部分可观测马尔可夫决策过程的策略，我们将其称为语言决策过程。然后，我们实现了五个环境，包括三个具有挑战性的科学环境：（1）进行分子克隆时的核酸结构操作；（2）通过访问科学文献回答研究问题；（3）工程蛋白质稳定性。这些环境之所以被选择，是因为它们强调多步推理，并且与当前的生物学研究密切相关。最后，通过在线训练和扩展推理时间计算资源，我们证明基于开源非前沿的大语言模型（LLM）的语言代理能够在多个任务上达到甚至超过前沿LLM代理和人类专家的表现，且推理成本最多可降低100倍。 

---
# Efficiently Serving LLM Reasoning Programs with Certaindex 

**Title (ZH)**: 用Certaindex高效服务于大型语言模型推理程序 

**Authors**: Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Aurick Qiao, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20993)  

**Abstract**: The rapid evolution of large language models (LLMs) has unlocked their capabilities in advanced reasoning tasks like mathematical problem-solving, code generation, and legal analysis. Central to this progress are inference-time reasoning algorithms, which refine outputs by exploring multiple solution paths, at the cost of increasing compute demands and response latencies. Existing serving systems fail to adapt to the scaling behaviors of these algorithms or the varying difficulty of queries, leading to inefficient resource use and unmet latency targets.
We present Dynasor, a system that optimizes inference-time compute for LLM reasoning queries. Unlike traditional engines, Dynasor tracks and schedules requests within reasoning queries and uses Certaindex, a proxy that measures statistical reasoning progress based on model certainty, to guide compute allocation dynamically. Dynasor co-adapts scheduling with reasoning progress: it allocates more compute to hard queries, reduces compute for simpler ones, and terminates unpromising queries early, balancing accuracy, latency, and cost. On diverse datasets and algorithms, Dynasor reduces compute by up to 50% in batch processing and sustaining 3.3x higher query rates or 4.7x tighter latency SLOs in online serving. 

**Abstract (ZH)**: 大语言模型（LLMs）的快速进化使其能够在高级推理任务中发挥作用，如数学问题解决、代码生成和法律分析。这一进展的核心在于推理时的推理算法，这些算法通过探索多种解决方案路径来改进输出，但这也增加了计算需求和响应时间。现有的服务系统未能适应这些算法的扩展行为或查询难度的变化，导致资源使用效率低下且无法满足预期的延迟目标。

我们提出了一种名为Dynasor的系统，该系统优化了LLM推理查询的推理时计算性能。与传统的引擎不同，Dynasor在推理查询中跟踪和调度请求，并使用Certaindex（一个基于模型确定性的代理，能够度量统计推理进度）来动态指导计算资源的分配。Dynasor同时适应推理进度和调度：为-hard 的查询分配更多的计算资源，为简单的查询减少计算资源，并在早期终止无前途的查询，从而在准确度、延迟和成本之间取得平衡。在多种数据集和算法上，Dynasor在批处理中将计算量减少高达50%，同时保持3.3倍更高的查询速率或4.7倍更严格的延迟SLO（服务级别目标）在在线服务中。 

---
# ChartAdapter: Large Vision-Language Model for Chart Summarization 

**Title (ZH)**: ChartAdapter：大型vision-language模型用于图表总结 

**Authors**: Peixin Xu, Yujuan Ding, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20715)  

**Abstract**: Chart summarization, which focuses on extracting key information from charts and interpreting it in natural language, is crucial for generating and delivering insights through effective and accessible data analysis. Traditional methods for chart understanding and summarization often rely on multi-stage pipelines, which may produce suboptimal semantic alignment between visual and textual information. In comparison, recently developed LLM-based methods are more dependent on the capability of foundation images or languages, while ignoring the characteristics of chart data and its relevant challenges. To address these limitations, we propose ChartAdapter, a novel lightweight transformer module designed to bridge the gap between charts and textual summaries. ChartAdapter employs learnable query vectors to extract implicit semantics from chart data and incorporates a cross-modal alignment projector to enhance vision-to-language generative learning. By integrating ChartAdapter with an LLM, we enable end-to-end training and efficient chart summarization. To further enhance the training, we introduce a three-stage hierarchical training procedure and develop a large-scale dataset specifically curated for chart summarization, comprising 190,618 samples. Experimental results on the standard Chart-to-Text testing set demonstrate that our approach significantly outperforms existing methods, including state-of-the-art models, in generating high-quality chart summaries. Ablation studies further validate the effectiveness of key components in ChartAdapter. This work highlights the potential of tailored LLM-based approaches to advance chart understanding and sets a strong foundation for future research in this area. 

**Abstract (ZH)**: 图表总结专注于从图表中提取关键信息并以自然语言进行解释，对于通过有效且易于访问的数据分析生成和传递洞察至关重要。传统的图表理解和总结方法通常依赖于多阶段管道，可能会导致图表信息与文本信息之间的语义对齐不理想。相比之下，近年来开发的基于大语言模型（LLM）的方法更加依赖于基础视觉或语言模型的能力，而忽视了图表数据的特性及其相关挑战。为了解决这些限制，我们提出了一种名为ChartAdapter的新颖轻量级transformer模块，旨在弥补图表与文本总结之间的差距。ChartAdapter利用可学习的查询向量从图表数据中提取潜在语义，并结合跨模态对齐投影器来增强视觉到语言生成学习。通过将ChartAdapter与大语言模型结合，我们实现了端到端的训练和高效的图表总结。为进一步增强训练，我们引入了三层级的分层训练程序，并开发了一个专门用于图表总结的大规模数据集，共包含190,618个样本。在标准的图表到文本测试集上的实验结果表明，我们的方法在生成高质量的图表总结方面显著优于现有方法，包括最先进的模型。进一步的消融研究验证了ChartAdapter中关键组件的有效性。这项工作突显了定制的大语言模型方法在推进图表理解方面的潜力，并为该领域的未来研究奠定了坚实基础。 

---
# UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design 

**Title (ZH)**: UBER：基于不确定性的大语言模型自动启发式设计演化方法 

**Authors**: Zijie Chen, Zhanchao Zhou, Yu Lu, Renjun Xu, Lili Pan, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20694)  

**Abstract**: NP-hard problem-solving traditionally relies on heuristics, but manually crafting effective heuristics for complex problems remains challenging. While recent work like FunSearch has demonstrated that large language models (LLMs) can be leveraged for heuristic design in evolutionary algorithm (EA) frameworks, their potential is not fully realized due to its deficiency in exploitation and exploration. We present UBER (Uncertainty-Based Evolution for Refinement), a method that enhances LLM+EA methods for automatic heuristic design by integrating uncertainty on top of the FunSearch framework. UBER introduces two key innovations: an Uncertainty-Inclusive Evolution Process (UIEP) for adaptive exploration-exploitation balance, and a principled Uncertainty-Inclusive Island Reset (UIIS) strategy for maintaining population diversity. Through extensive experiments on challenging NP-complete problems, UBER demonstrates significant improvements over FunSearch. Our work provides a new direction for the synergy of LLMs and EA, advancing the field of automatic heuristic design. 

**Abstract (ZH)**: 传统的NP-hard问题求解依赖于启发式方法，但为复杂问题手动设计有效的启发式方法仍然具有挑战性。虽然最近的研究，如FunSearch，已经证明大规模语言模型（LLMs）可以在进化算法（EAs）框架中用于启发式设计，但它们的潜力并未完全发挥，因为存在探索和利用能力的不足。我们提出了UBER（基于不确定性 refinement的进化算法），这是一种通过在FunSearch框架上集成不确定性来增强LLM+EA方法以实现自动启发式设计的方法。UBER引入了两项关键创新：一种包含不确定性的进化过程（UIEP），以实现自适应的探索和利用平衡，以及一种原则性的包含不确定性的岛群重启策略（UIIS），以维持种群多样性。通过在NP完全问题上的广泛实验，UBER在FunSearch的基础上取得了显著的进步。我们的工作为LLMs和EAs的协同作用提供了新的方向，推动了自动启发式设计领域的发展。 

---
# AmalREC: A Dataset for Relation Extraction and Classification Leveraging Amalgamation of Large Language Models 

**Title (ZH)**: AmalREC：一种基于大规模语言模型融合的关系提取与分类数据集 

**Authors**: Mansi, Pranshu Pandya, Mahek Bhavesh Vora, Soumya Bharadwaj, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2412.20427)  

**Abstract**: Existing datasets for relation classification and extraction often exhibit limitations such as restricted relation types and domain-specific biases. This work presents a generic framework to generate well-structured sentences from given tuples with the help of Large Language Models (LLMs). This study has focused on the following major questions: (i) how to generate sentences from relation tuples, (ii) how to compare and rank them, (iii) can we combine strengths of individual methods and amalgamate them to generate an even bette quality of sentences, and (iv) how to evaluate the final dataset? For the first question, we employ a multifaceted 5-stage pipeline approach, leveraging LLMs in conjunction with template-guided generation. We introduce Sentence Evaluation Index(SEI) that prioritizes factors like grammatical correctness, fluency, human-aligned sentiment, accuracy, and complexity to answer the first part of the second question. To answer the second part of the second question, this work introduces a SEI-Ranker module that leverages SEI to select top candidate generations. The top sentences are then strategically amalgamated to produce the final, high-quality sentence. Finally, we evaluate our dataset on LLM-based and SOTA baselines for relation classification. The proposed dataset features 255 relation types, with 15K sentences in the test set and around 150k in the train set organized in, significantly enhancing relational diversity and complexity. This work not only presents a new comprehensive benchmark dataset for RE/RC task, but also compare different LLMs for generation of quality sentences from relational tuples. 

**Abstract (ZH)**: 现有的用于关系分类和提取的数据集往往存在一些限制，如关系类型受限和领域偏见。本文提出了一种通用框架，借助大型语言模型（LLMs）从给定的元组中生成结构良好的句子。本研究主要关注以下问题：(i) 如何从关系元组生成句子，(ii) 如何比较和排名这些句子，(iii) 是否可以结合不同方法的优势，将它们整合以生成质量更高的句子，以及(iv) 如何评估最终数据集？对于第一个问题，我们采用一个多层次的5阶段管道方法，并结合模板指导生成和LLMs。我们介绍了句子评估指数（SEI），该指数优先考虑语法正确性、流畅性、与人类一致的情感、准确性以及复杂性等因素，以回答第二部分的评估问题。为了解决第二部分的评估问题，本研究引入了一个SEI-Ranker模块，该模块利用SEI来选择最佳候选生成。然后，将这些顶级句子战略性地合并，以生成最终高质量的句子。最后，我们在基于LLMs和当前最优基线（SOTA）的模型上对我们的数据集进行了关系分类评估。所提出的数据集包含255种关系类型，测试集中有15,000个句子，训练集中约有150,000个句子，显著增强了关系的多样性和复杂性。本文不仅提供了一个用于关系提取/关系分类任务的新型综合性基准数据集，还比较了不同LLMs在生成关系元组高质量句子方面的表现。 

---
# Topic-Aware Knowledge Graph with Large Language Models for Interoperability in Recommender Systems 

**Title (ZH)**: 面向主题的知识图谱：通过大型语言模型实现推荐系统中的互操作性 

**Authors**: Minhye Jeon, Seokho Ahn, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20163)  

**Abstract**: The use of knowledge graphs in recommender systems has become one of the common approaches to addressing data sparsity and cold start problems. Recent advances in large language models (LLMs) offer new possibilities for processing side and context information within knowledge graphs. However, consistent integration across various systems remains challenging due to the need for domain expert intervention and differences in system characteristics. To address these issues, we propose a consistent approach that extracts both general and specific topics from both side and context information using LLMs. First, general topics are iteratively extracted and updated from side information. Then, specific topics are extracted using context information. Finally, to address synonymous topics generated during the specific topic extraction process, a refining algorithm processes and resolves these issues effectively. This approach allows general topics to capture broad knowledge across diverse item characteristics, while specific topics emphasize detailed attributes, providing a more comprehensive understanding of the semantic features of items and the preferences of users. Experimental results demonstrate significant improvements in recommendation performance across diverse knowledge graphs. 

**Abstract (ZH)**: 知识图谱在推荐系统中的应用已成为解决数据稀疏性和冷启动问题的一种常见方法。大规模语言模型（LLMs）的最新进展为处理知识图谱内的辅助信息和上下文信息提供了新的可能性。然而，跨各种系统的一致集成仍面临挑战，这主要归因于需要领域专家的干预以及系统特性的差异。为了解决这些问题，我们提出了一种一致的方法，通过LLMs从辅助信息和上下文信息中提取通用和特定主题。首先，从辅助信息中迭代地提取和更新通用主题。然后，使用上下文信息提取特定主题。最后，针对特定主题提取过程中产生的同义主题，采用优化算法有效地处理和解决这些问题。这种方法使通用主题能够捕捉到各类物品特性的广泛知识，而特定主题则强调详细的属性，从而更全面地理解物品的语义特征和用户偏好。实验结果表明，该方法在不同知识图谱中显著提高了推荐性能。 

---
# Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema 

**Title (ZH)**: 基于维基数据模式的大型语言模型引导的知识图谱自动构建方法 

**Authors**: Xiaohan Feng, Xixin Wu, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2412.20942)  

**Abstract**: We propose an ontology-grounded approach to Knowledge Graph (KG) construction using Large Language Models (LLMs) on a knowledge base. An ontology is authored by generating Competency Questions (CQ) on knowledge base to discover knowledge scope, extracting relations from CQs, and attempt to replace equivalent relations by their counterpart in Wikidata. To ensure consistency and interpretability in the resulting KG, we ground generation of KG with the authored ontology based on extracted relations. Evaluation on benchmark datasets demonstrates competitive performance in knowledge graph construction task. Our work presents a promising direction for scalable KG construction pipeline with minimal human intervention, that yields high quality and human-interpretable KGs, which are interoperable with Wikidata semantics for potential knowledge base expansion. 

**Abstract (ZH)**: 我们提出了一种基于本体的方法，利用大型语言模型（LLMs）在一个知识库上构建知识图谱（KG）。本体的构建通过生成知识库上的能力问题（CQ）来发现知识范围，从中提取关系，并尝试用维基数据中的对应关系替换等效关系。为确保生成的KG在一致性和可 Interpretability 方面保持一致，我们将基于提取的关系建立的KG生成过程与所编写的本体相结合。在基准数据集上的测试表明，该方法在知识图谱构建任务中表现出了竞争力。我们的工作为在最少人工干预的情况下构建可扩展的KG管道提供了有 promise 的方向，这种管道能够生成高质量且易于人类理解的KG，这些KG能够与维基数据语义兼容，从而为知识库的扩展提供了潜力。 

---
# Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning 

**Title (ZH)**: 规划、居住与评判：基于多智能体LLM框架的循环城市规划系统 

**Authors**: Hang Ni, Yuzhi Wang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20505)  

**Abstract**: Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process. 

**Abstract (ZH)**: 城市再开发在城市化进程中面临显著挑战，要求采用适应性策略来应对不断变化的需求。借助大型语言模型（LLMs）的进步，我们提出了一种新的范式——循环城市规划（Cyclical Urban Planning，CUP），该范式通过闭环不断生成、评估和优化城市规划。具体而言，我们的基于多智能体的大型语言模型框架包括三个关键组成部分：（1）规划阶段，LLM智能体基于上下文数据生成和优化城市规划；（2）生活阶段，智能体模拟居民的行为和互动，模型化城市环境中的人类生活；（3）评估阶段，涉及评估规划的有效性并提供迭代反馈以进行改进。循环过程使得规划方法具有动态和响应性。实验结果表明，我们的框架作为连续且适应性强的规划过程是有效的。 

---
# Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense 

**Title (ZH)**: 向智能和安全的云迈进：大型语言模型赋能的主动防御 

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.21051)  

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods. 

**Abstract (ZH)**: 云计算技术的快速发展以及云应用程序的不断增加为日常生活带来了大量便利。然而，不同组件的多样性和复杂性给云计算安全带来了重大挑战，尤其是在应对复杂和先进的网络攻击时。近年来，生成基础模型（GFMs），尤其是大规模语言模型（LLMs）的进步为安全智能提供了有望的解决方案。通过利用其强大的语言理解、数据分析、任务推理、行动规划和代码生成能力，我们提出了一种名为LLM-PD的新颖主动防御架构，能够以主动的方式抵御各种威胁。LLM-PD能够通过全面的数据分析和顺序推理高效做出决策，并根据目标云的需求动态生成和部署可操作的防御机制。此外，它可以根据以往交互中获得的经验灵活自我进化，并在无需额外训练的情况下适应新的攻击场景。实验结果证明了其在防御有效性和效率方面的出色能力，并特别强调了与其他现有方法相比时取得的卓越成功率。 

---
# M$^3$oralBench: A MultiModal Moral Benchmark for LVLMs 

**Title (ZH)**: M$^3$oralBench: 一种面向多模态语言模型的道德基准测试 

**Authors**: Bei Yan, Jie Zhang, Zhiyuan Chen, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20718)  

**Abstract**: Recently, large foundation models, including large language models (LLMs) and large vision-language models (LVLMs), have become essential tools in critical fields such as law, finance, and healthcare. As these models increasingly integrate into our daily life, it is necessary to conduct moral evaluation to ensure that their outputs align with human values and remain within moral boundaries. Previous works primarily focus on LLMs, proposing moral datasets and benchmarks limited to text modality. However, given the rapid development of LVLMs, there is still a lack of multimodal moral evaluation methods. To bridge this gap, we introduce M$^3$oralBench, the first MultiModal Moral Benchmark for LVLMs. M$^3$oralBench expands the everyday moral scenarios in Moral Foundations Vignettes (MFVs) and employs the text-to-image diffusion model, SD3.0, to create corresponding scenario images. It conducts moral evaluation across six moral foundations of Moral Foundations Theory (MFT) and encompasses tasks in moral judgement, moral classification, and moral response, providing a comprehensive assessment of model performance in multimodal moral understanding and reasoning. Extensive experiments on 10 popular open-source and closed-source LVLMs demonstrate that M$^3$oralBench is a challenging benchmark, exposing notable moral limitations in current models. Our benchmark is publicly available. 

**Abstract (ZH)**: 近年来，大型基础模型，包括大型语言模型（LLMs）和大型视觉-语言模型（LVLMs），在法律、金融和医疗等关键领域已成为不可或缺的工具。随着这些模型越来越多地融入我们的日常生活，进行道德评估变得必要，以确保其输出与人类价值观相符，并保持在道德边界之内。以往的工作主要关注LLMs，提出了限于文本模态的道德数据集和基准测试。然而，鉴于LVLMs的快速发展，仍然缺乏多模态道德评估方法。为了弥补这一差距，我们引入了M$^3$oralBench，这是首个针对LVLMs的多模态道德基准测试。M$^3$oralBench 扩展了《道德基础情景》（MFVs）中的日常生活道德场景，并利用文本到图像扩散模型SD3.0创建相应的场景图像。该基准测试涵盖了《道德基础理论》（MFT）中的六种道德基础，并包含道德判断、道德分类和道德回应任务，提供了模型在多模态道德理解和推理方面的全面评估。针对10种流行开源和闭源LVLMs进行的广泛实验表明，M$^3$oralBench 是一个具有挑战性的基准测试，揭示了当前模型在道德方面的显著局限性。我们的基准测试已公开可用。 

---
# Controlling Out-of-Domain Gaps in LLMs for Genre Classification and Generated Text Detection 

**Title (ZH)**: 针对体裁分类和生成文本检测的LLM领域外差距控制 

**Authors**: Dmitri Roussinov, Serge Sharoff, Nadezhda Puchnina  

**Link**: [PDF](https://arxiv.org/pdf/2412.20595)  

**Abstract**: This study demonstrates that the modern generation of Large Language Models (LLMs, such as GPT-4) suffers from the same out-of-domain (OOD) performance gap observed in prior research on pre-trained Language Models (PLMs, such as BERT). We demonstrate this across two non-topical classification tasks: 1) genre classification and 2) generated text detection. Our results show that when demonstration examples for In-Context Learning (ICL) come from one domain (e.g., travel) and the system is tested on another domain (e.g., history), classification performance declines significantly.
To address this, we introduce a method that controls which predictive indicators are used and which are excluded during classification. For the two tasks studied here, this ensures that topical features are omitted, while the model is guided to focus on stylistic rather than content-based attributes. This approach reduces the OOD gap by up to 20 percentage points in a few-shot setup. Straightforward Chain-of-Thought (CoT) methods, used as the baseline, prove insufficient, while our approach consistently enhances domain transfer performance. 

**Abstract (ZH)**: 本研究证明，现代大型语言模型（LLMs，如GPT-4）在领域外（OOD）性能上与先前对预训练语言模型（PLMs，如BERT）的研究中观察到的现象一致。我们在这两个非主题分类任务中展示了这一点：1）体裁分类；2）生成文本检测。研究结果显示，当基于上下文学习（ICL）的演示示例来自一个领域（如旅行），而系统在另一个领域（如历史）进行测试时，分类性能会显著下降。

为了应对这一问题，我们提出了一种方法，该方法在分类过程中控制了哪些预测指标被使用，哪些被排除。对于这里研究的两个任务，这种方法确保省略了主题特征，同时引导模型专注于风格而非内容属性。这种策略在少样本设置中最多可减少20个百分点的领域外性能差距。传统的简单链式思维（CoT）方法用作基准时证明是不足的，而我们的方法则能一致提升领域间迁移性能。 

---
# Leveraging Large Language Models for Enhancing Autonomous Vehicle Perception 

**Title (ZH)**: 利用大型语言模型增强自主车辆感知 

**Authors**: Athanasios Karagounis  

**Link**: [PDF](https://arxiv.org/pdf/2412.20230)  

**Abstract**: Autonomous vehicles (AVs) rely on sophisticated perception systems to interpret their surroundings, a cornerstone for safe navigation and decision-making. The integration of Large Language Models (LLMs) into AV perception frameworks offers an innovative approach to address challenges in dynamic environments, sensor fusion, and contextual reasoning. This paper presents a novel framework for incorporating LLMs into AV perception, enabling advanced contextual understanding, seamless sensor integration, and enhanced decision support. Experimental results demonstrate that LLMs significantly improve the accuracy and reliability of AV perception systems, paving the way for safer and more intelligent autonomous driving technologies. By expanding the scope of perception beyond traditional methods, LLMs contribute to creating a more adaptive and human-centric driving ecosystem, making autonomous vehicles more reliable and transparent in their operations. These advancements redefine the relationship between human drivers and autonomous systems, fostering trust through enhanced understanding and personalized decision-making. Furthermore, by integrating memory modules and adaptive learning mechanisms, LLMs introduce continuous improvement in AV perception, enabling vehicles to evolve with time and adapt to changing environments and user preferences. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）依赖于复杂的感知系统来解释其周围环境，这是确保安全导航和决策的基础。将大型语言模型（LLMs）整合到AV感知框架中为应对动态环境、传感器融合和上下文推理的挑战提供了一种创新的方法。本文提出了一种新的框架，用于将LLMs整合到AV感知中，从而实现高级上下文理解、无缝传感器集成和增强的决策支持。实验结果表明，LLMs显著提高了AV感知系统的准确性和可靠性，为更安全、更智能的自主驾驶技术铺平了道路。通过扩展感知范围超越传统方法，LLMs有助于创建一个更具适应性和以人为中心的驾驶生态系统，使自主车辆在操作中更为可靠和透明。这些进步重新定义了人类驾驶员与自主系统之间的关系，通过增强理解和个性化决策来培养信任。此外，通过整合记忆模块和适应性学习机制，LLMs为AV感知提供持续改进的能力，使车辆能够随着时间的推移而演变，并适应不断变化的环境和用户偏好。 

---
# TradingAgents: Multi-Agents LLM Financial Trading Framework 

**Title (ZH)**: 交易代理：多智能体LLM金融交易框架

注：在这个翻译中，“TradingAgents”被译为“交易代理”，“Multi-Agents”被译为“多智能体”，“LLM”被解释为“大语言模型”，考虑到具体上下文，“LLM”也有可能指的是“长期记忆模型”或其他特定含义，需要根据实际场景进一步确认。此处保持了“LLM”未译，保持原文中的缩写形式，同时在翻译中注释其可能的含义。总体来说，整句话翻译保持了原文的学术风格和专业术语。 

**Authors**: Yijia Xiao, Edward Sun, Di Luo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20138)  

**Abstract**: Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. 

**Abstract (ZH)**: 在使用大规模语言模型（LLMs）驱动的代理社会进行自动化问题解决方面已经取得显著进展。在金融领域，大部分努力主要集中在处理特定任务的单代理系统或独立收集数据的多代理框架上。然而，多代理系统在复制现实世界交易公司的协同动态方面具有巨大潜力，这一领域尚未充分探索。TradingAgents 提出了一种受交易公司启发的新型股票交易框架，该框架中的代理由LLM驱动，扮演不同的专业角色，如基本面分析师、情绪分析师、技术分析师和不同风险偏好级别的交易者。该框架包括牛市和熊市研究员代理评估市场状况，风险管理部门监控风险敞口，以及交易者通过辩论和历史数据综合获得的见解来做决策。通过模拟动态且协作的交易环境，该框架旨在提高交易表现。详细的架构和大量实验表明，与基准模型相比，该框架在累积回报、夏普比率和最大回撤等方面具有显著优势，这表明多代理LLM框架在金融市场交易中的潜力。 

---
# AnalogXpert: Automating Analog Topology Synthesis by Incorporating Circuit Design Expertise into Large Language Models 

**Title (ZH)**: AnalogXpert: 通过将电路设计专业知识融入大型语言模型来自动化模拟拓扑合成 

**Authors**: Haoyi Zhang, Shizhao Sun, Yibo Lin, Runsheng Wang, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2412.19824)  

**Abstract**: Analog circuits are crucial in modern electronic systems, and automating their design has attracted significant research interest. One of major challenges is topology synthesis, which determines circuit components and their connections. Recent studies explore large language models (LLM) for topology synthesis. However, the scenarios addressed by these studies do not align well with practical applications. Specifically, existing work uses vague design requirements as input and outputs an ideal model, but detailed structural requirements and device-level models are more practical. Moreover, current approaches either formulate topology synthesis as graph generation or Python code generation, whereas practical topology design is a complex process that demands extensive design knowledge. In this work, we propose AnalogXpert, a LLM-based agent aiming at solving practical topology synthesis problem by incorporating circuit design expertise into LLMs. First, we represent analog topology as SPICE code and introduce a subcircuit library to reduce the design space, in the same manner as experienced designers. Second, we decompose the problem into two sub-task (i.e., block selection and block connection) through the use of CoT and incontext learning techniques, to mimic the practical design process. Third, we introduce a proofreading strategy that allows LLMs to incrementally correct the errors in the initial design, akin to human designers who iteratively check and adjust the initial topology design to ensure accuracy. Finally, we construct a high-quality benchmark containing both real data (30) and synthetic data (2k). AnalogXpert achieves 40% and 23% success rates on the synthetic dataset and real dataset respectively, which is markedly better than those of GPT-4o (3% on both the synthetic dataset and the real dataset). 

**Abstract (ZH)**: 现代电子系统中，模拟电路至关重要，其自动化设计吸引了大量研究兴趣。其中一项主要挑战是拓扑合成，它决定了电路元件及其连接方式。近期的研究探讨了通过大型语言模型（LLM）进行拓扑合成的可行性，但是现有研究中的应用场景与实际应用不完全匹配。具体来说，现有工作使用模糊的设计要求作为输入，输出理想模型，但实际上，详细的结构要求和器件级模型更为实用。此外，当前的方法将拓扑合成要么形式化为图生成，要么形式化为Python代码生成，而实际的拓扑设计是一个复杂的过程，需要广泛的设计知识。在本文中，我们提出了AnalogXpert，这是一种基于LLM的代理，旨在通过将电路设计专业知识整合到LLM中来解决实际的拓扑合成问题。首先，我们将模拟拓扑表示为SPICE代码，并引入子电路库以减少设计空间，类似于经验丰富的设计师的做法。其次，我们通过使用CoT和上下文学习技术将问题分解为两个子任务（即模块选择和模块连接），以模拟实际设计过程。第三，我们引入了一种校对策略，使LLM能够逐步修正初始设计中的错误，类似于人类设计师通过迭代检查和调整初始拓扑设计来确保准确性。最后，我们构建了一个高性能基准，包含实际数据（30个）和合成数据（2000个）。AnalogXpert在合成数据集和实际数据集上的成功率分别为40%和23%，这比GPT-4o的性能要好得多（在合成数据集和实际数据集上均为3%）。 

---
# LINKs: Large Language Model Integrated Management for 6G Empowered Digital Twin NetworKs 

**Title (ZH)**: LINKs：大型语言模型集成管理在6G赋能数字孪生网络中的应用 

**Authors**: Shufan Jiang, Bangyan Lin, Yue Wu, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.19811)  

**Abstract**: In the rapidly evolving landscape of digital twins (DT) and 6G networks, the integration of large language models (LLMs) presents a novel approach to network management. This paper explores the application of LLMs in managing 6G-empowered DT networks, with a focus on optimizing data retrieval and communication efficiency in smart city scenarios. The proposed framework leverages LLMs for intelligent DT problem analysis and radio resource management (RRM) in fully autonomous way without any manual intervention. Our proposed framework -- LINKs, builds up a lazy loading strategy which can minimize transmission delay by selectively retrieving the relevant data. Based on the data retrieval plan, LLMs transform the retrieval task into an numerical optimization problem and utilizing solvers to build an optimal RRM, ensuring efficient communication across the network. Simulation results demonstrate the performance improvements in data planning and network management, highlighting the potential of LLMs to enhance the integration of DT and 6G technologies. 

**Abstract (ZH)**: 在数字孪生（DT）和6G网络迅速发展的背景下，大型语言模型（LLMs）的集成为网络管理提供了一种新颖的方法。本文探讨了LLMs在管理6G赋能的数字孪生网络中的应用，重点关注在智能城市场景中优化数据检索和通信效率的方法。所提出的框架利用LLMs进行智能的DT问题分析和无线资源管理（RRM），以完全自主的方式进行，无需任何人工干预。我们提出的框架——LINKs，构建了一种懒加载策略，通过有选择地检索相关数据，从而最小化传输延迟。基于数据检索计划，LLMs将检索任务转换为一个数值优化问题，并利用求解器构建最优的RRM，确保网络中高效的通信。仿真结果表明，该方法在数据规划和网络管理方面的性能改进，突显了LLMs在增强DT和6G技术集成方面的能力。 

---
# exLong: Generating Exceptional Behavior Tests with Large Language Models 

**Title (ZH)**: ExLong：使用大型语言模型生成异常行为测试 

**Authors**: Jiyang Zhang, Yu Liu, Pengyu Nie, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2405.14619)  

**Abstract**: Many popular programming languages, including C#, Java, and Python, support exceptions. Exceptions are thrown during program execution if an unwanted event happens, e.g., a method is invoked with an illegal argument value. Software developers write exceptional behavior tests (EBTs) to check that their code detects unwanted events and throws appropriate exceptions. Prior research studies have shown the importance of EBTs, but those studies also highlighted that developers put most of their efforts on "happy paths", e.g., paths without unwanted events. To help developers fill the gap, we present the first framework, dubbed exLong, that automatically generates EBTs. exLong is a large language model instruction fine-tuned from CodeLlama and embeds reasoning about traces that lead to throw statements, conditional expressions that guard throw statements, and non-exceptional behavior tests that execute similar traces. We compare exLong with the state-of-the-art models for test generation (CAT-LM) and one of the strongest foundation models (GPT-4o), as well as with analysis-based tools for test generation (Randoop and EvoSuite). Our results show that exLong outperforms existing models and tools. Furthermore, we contributed several pull requests to open-source projects and 23 EBTs generated by exLong were already accepted. 

**Abstract (ZH)**: 许多流行的编程语言，包括C#、Java和Python，都支持异常处理。如果发生不希望的事件，例如方法调用时传入了非法参数值，这些编程语言就会抛出异常。软件开发人员编写异常行为测试（EBTs）以检查代码是否能够检测到不希望的事件并正确抛出异常。先前的研究已经表明EBTs的重要性，但这些研究也指出，开发人员通常将大部分精力集中在所谓的“成功路径”上，即没有不希望事件的路径。为了帮助开发人员弥补这一不足，我们提出了第一个名为exLong的框架，能够自动生成EBTs。exLong基于CodeLlama进行指令微调，并嵌入了关于导致抛出语句的调用跟踪、保护抛出语句的条件表达式以及执行类似跟踪的非异常行为测试的推理。我们将exLong与最先进测试生成模型（CAT-LM）和一个最强大的基础模型（GPT-4o）进行了比较，并与基于分析的测试生成工具（Randoop和EvoSuite）进行了比较。结果显示，exLong在多个方面优于现有模型和工具。此外，我们还为开源项目提交了几个代码请求，并且已经有23个由exLong生成的EBTs被接受。 

---
