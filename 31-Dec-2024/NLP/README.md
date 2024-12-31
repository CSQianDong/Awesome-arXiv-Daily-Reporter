# Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs 

**Title (ZH)**: 不要过度思考 2+3=？——关于 o1-Like LLMs 的过度思考问题 

**Authors**: Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.21187)  

**Abstract**: The remarkable performance of models like the OpenAI o1 can be attributed to their ability to emulate human-like long-time thinking during inference. These models employ extended chain-of-thought (CoT) processes, exploring multiple strategies to enhance problem-solving capabilities. However, a critical question remains: How to intelligently and efficiently scale computational resources during testing. This paper presents the first comprehensive study on the prevalent issue of overthinking in these models, where excessive computational resources are allocated for simple problems with minimal benefit. We introduce novel efficiency metrics from both outcome and process perspectives to evaluate the rational use of computational resources by o1-like models. Using a self-training paradigm, we propose strategies to mitigate overthinking, streamlining reasoning processes without compromising accuracy. Experimental results show that our approach successfully reduces computational overhead while preserving model performance across a range of testsets with varying difficulty levels, such as GSM8K, MATH500, GPQA, and AIME. 

**Abstract (ZH)**: 像OpenAI的o1这样的模型表现出色，主要是因为它们在推理过程中能够模拟类似人类的长时间思考能力。这些模型通过扩展链式思考（CoT）过程，探索多种策略以增强问题解决能力。然而，一个关键问题依然存在：如何在测试过程中智能而有效地扩大计算资源的使用。本文首次全面研究了这些模型中普遍存在的过度思考问题，即为解决一些简单问题而不必要地分配过多计算资源。我们从结果和过程两个视角引入了新的效率指标，评估o1类模型在合理使用计算资源方面的效果。通过自我训练的范式，我们提出了策略来缓解过度思考，简化推理过程而不影响准确性。实验结果显示，我们的方法在不同难度水平的数据集（如GSM8K、MATH500、GPQA和AIME）上成功减少了计算开销，同时保持了模型表现。 

---
# Facilitating large language model Russian adaptation with Learned Embedding Propagation 

**Title (ZH)**: 使用学习嵌入传播促进大规模语言模型的俄语适应 

**Authors**: Mikhail Tikhomirov, Daniil Chernyshev  

**Link**: [PDF](https://arxiv.org/pdf/2412.21140)  

**Abstract**: Rapid advancements of large language model (LLM) technologies led to the introduction of powerful open-source instruction-tuned LLMs that have the same text generation quality as the state-of-the-art counterparts such as GPT-4. While the emergence of such models accelerates the adoption of LLM technologies in sensitive-information environments the authors of such models don not disclose the training data necessary for replication of the results thus making the achievements model-exclusive. Since those open-source models are also multilingual this in turn reduces the benefits of training a language specific LLMs as improved inference computation efficiency becomes the only guaranteed advantage of such costly procedure. More cost-efficient options such as vocabulary extension and subsequent continued pre-training are also inhibited by the lack of access to high-quality instruction-tuning data since it is the major factor behind the resulting LLM task-solving capabilities. To address the limitations and cut the costs of the language adaptation pipeline we propose Learned Embedding Propagation (LEP). Unlike existing approaches our method has lower training data size requirements due to minimal impact on existing LLM knowledge which we reinforce using novel ad-hoc embedding propagation procedure that allows to skip the instruction-tuning step and instead implant the new language knowledge directly into any existing instruct-tuned variant. We evaluated four Russian vocabulary adaptations for LLaMa-3-8B and Mistral-7B, showing that LEP is competitive with traditional instruction-tuning methods, achieving performance comparable to OpenChat 3.5 and LLaMa-3-8B-Instruct, with further improvements via self-calibration and continued tuning enhancing task-solving capabilities. 

**Abstract (ZH)**: 大语言模型（LLM）技术的 rapid 进步催生了具有与最先进模型（如 GPT-4）相同文本生成质量的强大开源指令调优模型。这些模型的出现加速了LLM技术在敏感信息环境中的应用，但模型的作者并未公布必要的训练数据以实现结果的复现，从而使这些成就仅限于特定模型。由于这些开源模型是多语言的，这反而减少了训练特定语言LMM的效益，因为提高推理计算效率成了唯一确定的优势。此外，由于缺乏高质量指令调优数据的访问权限，扩大词汇量和后续继续预训练等更经济的选择也被抑制。这些高质量指令调优数据是导致生成模型任务解决能力的主要因素。为此，我们提出了一种名为“Learned Embedding Propagation”（LEP）的方法，以解决这些限制并降低语言适应管道的成本。与现有方法不同，由于对现有LLM知识的影响较小，LEP方法所需的训练数据较小。我们通过引入一种新颖的即兴嵌入传播程序来强化现有知识，这使得可以直接将新语言知识植入到任何已有的指令调优变体中，而无需进行指令调优步骤。我们对LLaMa-3-8B和Mistral-7B的四种俄罗斯词汇适应性进行了评估，结果显示，LEP在性能上与传统的指令调优方法相媲美，达到OpenChat 3.5和LLaMa-3-8B-Instruct的水平，并且通过自我校准和继续调优进一步增强了任务解决能力。 

---
# Exploring and Controlling Diversity in LLM-Agent Conversation 

**Title (ZH)**: 探索和控制大规模语言模型-代理对话中的多样性 

**Authors**: KuanChao Chu, Yi-Pei Chen, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2412.21102)  

**Abstract**: Diversity is a critical aspect of multi-agent communication. In this paper, we focus on controlling and exploring diversity in the context of open-domain multi-agent conversations, particularly for world simulation applications. We propose Adaptive Prompt Pruning (APP), a novel method that dynamically adjusts the content of the utterance generation prompt to control diversity using a single parameter, lambda. Through extensive experiments, we show that APP effectively controls the output diversity across models and datasets, with pruning more information leading to more diverse output. We comprehensively analyze the relationship between prompt content and conversational diversity. Our findings reveal that information from all components of the prompt generally constrains the diversity of the output, with the Memory block exerting the most significant influence. APP is compatible with established techniques like temperature sampling and top-p sampling, providing a versatile tool for diversity management. To address the trade-offs of increased diversity, such as inconsistencies with omitted information, we incorporate a post-generation correction step, which effectively balances diversity enhancement with output consistency. Additionally, we examine how prompt structure, including component order and length, impacts diversity. This study addresses key questions surrounding diversity in multi-agent world simulation, offering insights into its control, influencing factors, and associated trade-offs. Our contributions lay the foundation for systematically engineering diversity in LLM-based multi-agent collaborations, advancing their effectiveness in real-world applications. 

**Abstract (ZH)**: 多样性是多智能体通信中的一个关键方面。本文重点探讨在开放领域多智能体对话中控制和探索多样性的问题，特别是针对世界模拟应用。我们提出了适应性提示剪裁（APP，Adaptive Prompt Pruning）方法，这是一种通过单一参数λ动态调整生成语句提示内容的新颖方法，用于控制多样性。通过广泛的实验，我们展示了APP在不同模型和数据集上有效地控制了输出的多样性，剪裁更多信息会带来更广泛的输出。我们全面分析了提示内容与对话多样性之间的关系。研究结果表明，提示中所有组成部分的信息通常限制了输出的多样性，而记忆块的影响最为显著。APP可以与现有的技术（如温度采样和top-p采样）兼容，提供了一种多功能的多样性管理工具。为了应对多样性增加带来的权衡取舍，如省略信息的一致性问题，我们引入了一步生成后修正步骤，有效平衡了多样性增强与输出一致性。此外，我们还探讨了提示结构，包括各成分的顺序和长度，对多样性的影响。本研究探讨了多智能代理世界模拟中关于多样性的关键问题，提供了控制多样性、影响因素及其相关权衡的认知。我们的贡献为系统化地在基于LLM的多智能代理合作中工程化多样性奠定了基础，推动了它们在实际应用中的有效性。 

---
# Efficient Multi-Task Inferencing with a Shared Backbone and Lightweight Task-Specific Adapters for Automatic Scoring 

**Title (ZH)**: 使用共享骨干和轻量级任务特定适配器进行高效多任务推理的自动评分方法 

**Authors**: Ehsan Latif, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2412.21065)  

**Abstract**: The integration of Artificial Intelligence (AI) in education requires scalable and efficient frameworks that balance performance, adaptability, and cost. This paper addresses these needs by proposing a shared backbone model architecture enhanced with lightweight LoRA adapters for task-specific fine-tuning, targeting the automated scoring of student responses across 27 mutually exclusive tasks. By achieving competitive performance (average QWK of 0.848 compared to 0.888 for fully fine-tuned models) while reducing GPU memory consumption by 60% and inference latency by 40%, the framework demonstrates significant efficiency gains. This approach aligns with the workshops' focus on improving language models for educational tasks, creating responsible innovations for cost-sensitive deployment, and supporting educators by streamlining assessment workflows. The findings underscore the potential of scalable AI to enhance learning outcomes while maintaining fairness and transparency in automated scoring systems. 

**Abstract (ZH)**: 将人工智能（AI）融入教育领域需要具备可扩展性和高效性的框架，这些框架能够平衡性能、适应性和成本。本文通过提出一个增强轻量级LoRA适配器的共享骨干模型架构，针对27项互斥任务的学生响应自动评分进行特定任务微调，以满足这些需求。通过在保持竞争力（平均QWK得分为0.848，与完全微调模型的0.888相比较）的同时，减少60%的GPU内存消耗和40%的推理延迟，该框架证明了显著的效率提升。这种做法与研讨会关注的改进教育任务中的语言模型、促进成本敏感部署的责任创新以及通过简化评估流程支持教育工作者相一致。研究结果强调了可扩展AI在提升学习成果方面的作用，同时保持了自动评分系统中的公平性和透明性。 

---
# GePBench: Evaluating Fundamental Geometric Perception for Multimodal Large Language Models 

**Title (ZH)**: GePBench：评估多模态大型语言模型的基本几何感知能力 

**Authors**: Shangyu Xing, Changhao Xiang, Yuteng Han, Yifan Yue, Zhen Wu, Xinyu Liu, Zhangtai Wu, Fei Zhao, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2412.21036)  

**Abstract**: Multimodal large language models (MLLMs) have achieved significant advancements in integrating visual and linguistic understanding. While existing benchmarks evaluate these models in context-rich, real-life scenarios, they often overlook fundamental perceptual skills essential for environments deviating from everyday realism. In particular, geometric perception, the ability to interpret spatial relationships and abstract visual patterns, remains underexplored. To address this limitation, we introduce GePBench, a novel benchmark designed to assess the geometric perception capabilities of MLLMs. Results from extensive evaluations reveal that current state-of-the-art MLLMs exhibit significant deficiencies in such tasks. Additionally, we demonstrate that models trained with data sourced from GePBench show notable improvements on a wide range of downstream tasks, underscoring the importance of geometric perception as a foundation for advanced multimodal applications. Our code and datasets will be publicly available. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）已在结合视觉和语言理解方面取得了显著进展。虽然现有的基准测试在丰富的上下文环境中评估了这些模型，但在偏离日常现实环境的情况下，它们往往忽视了用于这些环境的基本感知技能。特别是几何感知能力，即解释空间关系和抽象视觉模式的能力，仍然未被充分探索。为解决这一局限，我们提出了GePBench，一个旨在评估MLLMs几何感知能力的新基准测试。广泛的评估结果表明，当前最先进的MLLMs在这些任务中存在显著缺陷。此外，我们还展示了使用来自GePBench的数据进行训练的模型在多种下游任务中表现出明显的改进，突显了几何感知作为高级多模态应用基础的重要性。我们的代码和数据集将公开提供。 

---
# Plancraft: an evaluation dataset for planning with LLM agents 

**Title (ZH)**: Plancraft：一种用于评估基于大语言模型代理规划能力的数据集 

**Authors**: Gautier Dagan, Frank Keller, Alex Lascarides  

**Link**: [PDF](https://arxiv.org/pdf/2412.21033)  

**Abstract**: We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as an oracle planner and oracle RAG information extractor, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and strategies on our task and compare their performance to a handcrafted planner. We find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and we offer suggestions on how to improve their capabilities. 

**Abstract (ZH)**: 我们介绍了一个多模态评估数据集Plancraft，用于评估大规模语言模型（LLM）智能体。Plancraft提供了纯文本和多模态两种界面，都基于Minecraft制作界面。我们通过引用Minecraft维基来评估工具使用和检索增强生成（RAG），并包括一个先验规划器和先验RAG信息抽取器，以消除现代智能体架构中不同组件的影响。为了评估决策能力，Plancraft还包含了一些故意无法解决的例子，提供了具有现实挑战性的任务，不仅要求智能体完成任务，还需要智能体判断这些任务是否可解决。我们对开源和封闭源代码的LLM及其策略进行了基准测试，并将它们的表现与手工构建的智能体进行比较。我们发现，LLM和视觉语言模型在应对Plancraft引入的规划问题时存在困难，并提出了改进其能力的建议。 

---
# MapQaTor: A System for Efficient Annotation of Map Query Datasets 

**Title (ZH)**: MapQaTor：一种高效的地图查询数据集标注系统 

**Authors**: Mahir Labib Dihan, Mohammed Eunus Ali, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2412.21015)  

**Abstract**: Mapping and navigation services like Google Maps, Apple Maps, Openstreet Maps, are essential for accessing various location-based data, yet they often struggle to handle natural language geospatial queries. Recent advancements in Large Language Models (LLMs) show promise in question answering (QA), but creating reliable geospatial QA datasets from map services remains challenging. We introduce MapQaTor, a web application that streamlines the creation of reproducible, traceable map-based QA datasets. With its plug-and-play architecture, MapQaTor enables seamless integration with any maps API, allowing users to gather and visualize data from diverse sources with minimal setup. By caching API responses, the platform ensures consistent ground truth, enhancing the reliability of the data even as real-world information evolves. MapQaTor centralizes data retrieval, annotation, and visualization within a single platform, offering a unique opportunity to evaluate the current state of LLM-based geospatial reasoning while advancing their capabilities for improved geospatial understanding. Evaluation metrics show that, MapQaTor speeds up the annotation process by at least 30 times compared to manual methods, underscoring its potential for developing geospatial resources, such as complex map reasoning datasets. The website is live at: this https URL and a demo video is available at: this https URL. 

**Abstract (ZH)**: 像Google Maps、Apple Maps、OpenStreetMap这样的地图和导航服务对于访问各种基于地理位置的数据至关重要，但它们经常难以处理自然语言空间查询。近年来，大型语言模型（LLMs）在问答（QA）方面取得了进展，但从地图服务创建可靠的地理空间QA数据集仍然具有挑战性。我们引入了MapQaTor，这是一个网页应用程序，简化了创建可重现性和可追溯性地图为基础的QA数据集的过程。凭借其即插即用的架构，MapQaTor使用户能够无缝集成任何地图API，并以最少的设置从多种数据源收集和可视化数据。通过缓存API响应，该平台确保了数据的一致性基准真值，即使在实际信息发生变化时也能提高数据的可靠性。MapQaTor集成了数据检索、注释和可视化，提供了一个独特的机会来评估基于LLM的地理空间推理的当前状态，并推动其能力以提高地理空间理解。评估指标显示，MapQaTor将注释过程的速度提高了至少30倍，这突显了其开发地理空间资源，如复杂地图推理数据集的潜力。该网站现已经上线：[这个链接](this https URL)，并且有一个演示视频：[这个链接](this https URL)。 

---
# Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale via Principled Criteria 

**Title (ZH)**: 基于冗余性判断的动词意识推理简化：通过原则性标准有效简化冗余推理 

**Authors**: Joonwon Jang, Jaehee Kim, Wonbin Kweon, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.21006)  

**Abstract**: Large Language Models (LLMs) rely on generating extensive intermediate reasoning units (e.g., tokens, sentences) to enhance final answer quality across a wide range of complex tasks. While generating multiple reasoning paths or iteratively refining rationales proves effective for improving performance, these approaches inevitably result in significantly higher inference costs. In this work, we propose a novel sentence-level rationale reduction training framework that leverages likelihood-based criteria, verbosity, to identify and remove redundant reasoning sentences. Unlike previous approaches that utilize token-level reduction, our sentence-level reduction framework maintains model performance while reducing generation length. This preserves the original reasoning abilities of LLMs and achieves an average 17.15% reduction in generation costs across various models and tasks. 

**Abstract (ZH)**: 大语言模型（LLMs）依赖于生成大量中间推理单元（例如，令牌、句子）来增强在广泛复杂任务中的最终答案质量。虽然生成多个推理路径或迭代优化论证可以有效提高性能，但这些方法不可避免地会导致显著增加推理成本。在本研究中，我们提出了一种新颖的句子级论证缩减训练框架，该框架利用基于似然性的标准（冗余性）来识别和删除冗余的推理句子。不同于利用令牌级缩减的先前方法，我们的句子级缩减框架在保持模型性能的同时减少了生成长度。这保留了LLMs的原始推理能力，并在各种模型和任务上实现了平均17.15%的生成成本降低。 

---
# Plug-and-Play Training Framework for Preference Optimization 

**Title (ZH)**: 用于偏好优化的即插即用训练框架 

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Lei Sha, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2412.20996)  

**Abstract**: Recently, preference optimization methods such as DPO have significantly enhanced large language models (LLMs) in wide tasks including dialogue and question-answering. However, current methods fail to account for the varying difficulty levels of training samples during preference optimization, leading to mediocre performance in tasks with high accuracy requirements, particularly in mathematical reasoning. To address this limitation, we propose a novel training framework, which employs multiple sampling to analyze output distributions, assign different weights to samples, and incorporate these weights into the preference optimization process. This plug-and-play approach enables LLMs to prioritize challenging examples during training, improving learning efficiency. Experimental results demonstrate that our framework integrates seamlessly with various preference optimization methods and achieves consistent improvements in mathematical reasoning tasks. 

**Abstract (ZH)**: 近年来，偏好优化方法如DPO显著提升了大语言模型（LLMs）在对话、问答等广泛任务中的表现。然而，当前的方法在偏好优化过程中未能考虑到训练样本难度的差异性，导致在高准确率要求的任务中，尤其是数学推理任务中表现不佳。为解决这一局限性，我们提出了一种新的训练框架，该框架通过多种采样方法分析输出分布，对不同样本赋予不同的权重，并将这些权重纳入偏好优化过程。这种即插即用的方法使LLMs在训练过程中更侧重于处理具有挑战性的示例，从而提高学习效率。实验结果表明，我们的框架能够无缝地与各种偏好优化方法结合，并在数学推理任务中实现一致的性能提升。 

---
# KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation 

**Title (ZH)**: KARPA：一种无需训练的方法，将知识图谱适应为大型语言模型推理路径聚合的参考 

**Authors**: Siyuan Fang, Kaijing Ma, Tianyu Zheng, Xinrun Du, Ningxuan Lu, Ge Zhang, Qingkun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20995)  

**Abstract**: Large language models (LLMs) demonstrate exceptional performance across a variety of tasks, yet they are often affected by hallucinations and the timeliness of knowledge. Leveraging knowledge graphs (KGs) as external knowledge sources has emerged as a viable solution, but existing methods for LLM-based knowledge graph question answering (KGQA) are often limited by step-by-step decision-making on KGs, restricting the global planning and reasoning capabilities of LLMs, or they require fine-tuning or pre-training on specific KGs. To address these challenges, we propose Knowledge graph Assisted Reasoning Path Aggregation (KARPA), a novel framework that harnesses the global planning abilities of LLMs for efficient and accurate KG reasoning. KARPA operates in three steps: pre-planning relation paths using the LLM's global planning capabilities, matching semantically relevant paths via an embedding model, and reasoning over these paths to generate answers. Unlike existing KGQA methods, KARPA avoids stepwise traversal, requires no additional training, and is adaptable to various LLM architectures. Extensive experimental results show that KARPA achieves state-of-the-art performance in KGQA tasks, delivering both high efficiency and accuracy. Our code will be available on Github. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但它们往往受到幻觉和知识时效性的影响。利用知识图谱（KGs）作为外部知识来源已成为一种可行的解决方案，但现有的基于LLM的知识图谱问答（KGQA）方法常常受限于逐步在KG上进行决策，限制了LLM的全局规划和推理能力，或者需要针对特定的KG进行微调或预训练。为了解决这些挑战，我们提出了知识图谱辅助推理路径聚合（KARPA）框架，该框架利用LLM的全局规划能力进行高效的KG推理。KARPA分为三个步骤：利用LLM的全局规划能力预先规划关系路径，通过嵌入模型匹配语义相关路径，然后在这些路径上进行推理以生成答案。与现有的KGQA方法不同，KARPA避免了逐步遍历，无需额外训练，并且可以适应各种LLM架构。广泛的实验结果表明，KARPA在KGQA任务中实现了最先进的性能，既高效又准确。我们的代码将发布在Github上。 

---
# DoTA: Weight-Decomposed Tensor Adaptation for Large Language Models 

**Title (ZH)**: DoTA：分解权重张量适应性方法用于大型语言模型 

**Authors**: Xiaolin Hu, Xiang Cheng, Peiyu Liu, Wei Liu, Jian Luan, Bin Wang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20891)  

**Abstract**: Low-rank adaptation (LoRA) reduces the computational and memory demands of fine-tuning large language models (LLMs) by approximating updates with low-rank matrices. However, low-rank approximation in two-dimensional space fails to capture high-dimensional structures within the target matrix. Recently, tensor decomposition methods have been explored for fine-tuning LLMs, leveraging their ability to extract structured information. Yet, these approaches primarily rely on random initialization, and the impact of initialization on tensor adaptation remains underexplored. In this paper, we reveal that random initialization significantly diverges from the validation loss achieved by full fine-tuning. To address this, we propose Weight-Decomposed Tensor Adaptation (DoTA), which leverages the Matrix Product Operator (MPO) decomposition of pre-trained weights for effective initialization in fine-tuning LLMs. Additionally, we introduce QDoTA, a quantized version of DoTA designed for 4-bit quantization. Experiments on commonsense and arithmetic reasoning tasks show that DoTA outperforms random initialization methods with fewer parameters. QDoTA further reduces memory consumption and achieves comparable performance to DoTA on commonsense reasoning tasks. We will release our code to support future research. 

**Abstract (ZH)**: 低秩适应（LoRA）通过使用低秩矩阵近似更新，减少了大规模语言模型（LLMs）微调的计算和内存需求。然而，二维空间中的低秩逼近未能捕捉目标矩阵中的高维结构。最近，已探索使用张量分解方法进行LLMs的微调，利用它们提取结构化信息的能力。然而，这些方法主要依赖于随机初始化，关于初始化对张量适应的影响尚未得到充分探索。本文揭示了随机初始化与全量微调所达到的验证损失存在显著差异。为解决这一问题，我们提出了矩阵乘积算子分解（Matrix Product Operator, MPO）中的权重分解张量适应（Weight-Decomposed Tensor Adaptation, DoTA）方法，利用预训练权重的MPO分解实现有效的初始化。此外，我们引入了QDoTA，这是一种基于4位量化设计的DoTA版本。在常识推理和算术推理任务上的实验表明，DoTA在参数量较少的情况下优于随机初始化方法。QDoTA进一步减少了内存消耗，并在常识推理任务上达到了与DoTA相当的性能。我们将会公开代码，以支持未来的研究。 

---
# Enhancing Annotated Bibliography Generation with LLM Ensembles 

**Title (ZH)**: 使用大型语言模型ensemble增强标注参考文献生成 

**Authors**: Sergio Bermejo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20864)  

**Abstract**: This work proposes a novel approach to enhancing annotated bibliography generation through Large Language Model (LLM) ensembles. In particular, multiple LLMs in different roles -- controllable text generation, evaluation, and summarization -- are introduced and validated using a systematic methodology to enhance model performance in scholarly tasks. Output diversity among the ensemble that generates text is obtained using different LLM parameters, followed by an LLM acting as a judge to assess relevance, accuracy, and coherence. Responses selected by several combining strategies are then merged and refined through summarization and redundancy removal techniques. The preliminary experimental validation demonstrates that the combined outputs from the LLM ensemble improve coherence and relevance compared to individual responses, leading to a 38% improvement in annotation quality and a 51% reduction in content redundancy, thus highlighting the potential for automating complex scholarly tasks while maintaining high-quality standards. 

**Abstract (ZH)**: 本文提出了一种通过大规模语言模型（LLM）集成增强标注书目生成的新方法。具体而言，引入了多种扮演不同角色的LLM——可控文本生成、评估和总结——并通过系统的方法进行了验证，以提升模型在学术任务中的性能。通过使用不同的LLM参数获得生成文本的输出多样性，随后使用一个作为评判者的LLM评估其相关性、准确性和连贯性。经过多种组合策略选出的响应被合并并通过总结和去除冗余的技术进行精炼。初步的实验验证表明，LLM集成的综合输出在连贯性和相关性方面优于单独的响应，从而将注释质量提高了38%，减少了51%的内容冗余，这突显了自动化复杂学术任务的潜力，同时保持了高质量标准。 

---
# Are LLMs Really Not Knowledgable? Mining the Submerged Knowledge in LLMs' Memory 

**Title (ZH)**: 大型语言模型真的不具备知识吗？挖掘大型语言模型记忆中的隐性知识 

**Authors**: Xingjian Tao, Yiwei Wang, Yujun Cai, Zhicheng Yang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20846)  

**Abstract**: Large language models (LLMs) have shown promise as potential knowledge bases, yet they often struggle with question-answering tasks and are prone to hallucinations. While previous research attributes these issues to knowledge gaps in the model's parameters, our investigation reveals a different phenomenon: LLMs often retain correct knowledge even when generating incorrect answers. Through analysis of model's internal representations, we find that correct answers frequently appear among high-probability tokens despite not being selected as final outputs. Based on this observation, we introduce Hits@k, a new metric to assess knowledge retention independent of expression accuracy. Our extensive experiments demonstrate that LLMs store significantly more knowledge than their QA performance suggests. Building on these findings, we develop SkipUnsure, a method to improve answer accuracy by leveraging detected but unexpressed knowledge. Experiments on both open-domain and specific-domain datasets show consistent improvements, with accuracy gains of up to 11.8% on DBPedia and 6.3% on IMDB, without requiring model retraining. 

**Abstract (ZH)**: 大型语言模型（LLMs）在潜在知识库方面展现出了潜力，但它们往往在问答任务中表现不佳，并且容易产生幻觉。虽然先前的研究将这些问题归因于模型参数中的知识缺口，但我们的研究表明存在一种不同的现象：即使生成错误答案，LLMs 经常仍保留了正确的知识。通过分析模型的内部表示，我们发现即使高概率的正确答案没有被选为最终输出，它们也经常出现在高概率的标记中。基于这一观察，我们引入了 Hits@k，这是一种新的度量标准，用于独立于表达准确性来评估知识保留情况。我们的大量实验证明，LLMs 实际上存储的知识比其在问答任务中的表现所表明的要多得多。基于这些发现，我们开发了 SkipUnsure 方法，该方法通过利用检测到但未表达的知识来提高答案准确性。在开放领域和特定领域的数据集上的实验结果一致显示出改进，DBPedia 数据集的准确性提高了 11.8%，IMDB 数据集的准确性提高了 6.3%，而无需进行模型重训练。 

---
# Disentangling Preference Representation and Text Generation for Efficient Individual Preference Alignment 

**Title (ZH)**: 解耦偏好表示与文本生成以实现高效个体偏好对齐 

**Authors**: Jianfei Zhang, Jun Bai, Bei Li, Yanmeng Wang, Rumei Li, Chenghua Lin, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2412.20834)  

**Abstract**: Aligning Large Language Models (LLMs) with general human preferences has been proved crucial in improving the interaction quality between LLMs and human. However, human values are inherently diverse among different individuals, making it insufficient to align LLMs solely with general preferences. To address this, personalizing LLMs according to individual feedback emerges as a promising solution. Nonetheless, this approach presents challenges in terms of the efficiency of alignment algorithms. In this work, we introduce a flexible paradigm for individual preference alignment. Our method fundamentally improves efficiency by disentangling preference representation from text generation in LLMs. We validate our approach across multiple text generation tasks and demonstrate that it can produce aligned quality as well as or better than PEFT-based methods, while reducing additional training time for each new individual preference by $80\%$ to $90\%$ in comparison with them. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与普遍的人类偏好对齐已被证明对于提高LLMs与人类的互动质量至关重要。然而，人类的价值观在不同个体之间本是多元的，仅仅将LLMs与普遍偏好对齐是不够的。为解决这一问题，根据个别反馈个性化LLMs被认为是一种有前景的解决方案。然而，这种方法在对齐算法的效率方面也面临着挑战。在这项工作中，我们引入了一种灵活的个性化偏好对齐范式。我们的方法通过在LLMs中分离偏好表示与文本生成，从根本上提高了效率。我们在多个文本生成任务中验证了这一方法，并证明它可以产生与基于PEFT的方法相当甚至更好的对齐质量，同时将每次新个体偏好额外训练时间减少了80%到90%。 

---
# Attributing Culture-Conditioned Generations to Pretraining Corpora 

**Title (ZH)**: 将文化条件下的代际差异归因于预训练数据集 

**Authors**: Huihan Li, Arnav Goel, Keyu He, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.20760)  

**Abstract**: In open-ended generative tasks like narrative writing or dialogue, large language models often exhibit cultural biases, showing limited knowledge and generating templated outputs for less prevalent cultures. Recent works show that these biases may stem from uneven cultural representation in pretraining corpora. This work investigates how pretraining leads to biased culture-conditioned generations by analyzing how models associate entities with cultures based on pretraining data patterns. We propose the MEMOed framework (MEMOrization from pretraining document) to determine whether a generation for a culture arises from memorization. Using MEMOed on culture-conditioned generations about food and clothing for 110 cultures, we find that high-frequency cultures in pretraining data yield more generations with memorized symbols, while some low-frequency cultures produce none. Additionally, the model favors generating entities with extraordinarily high frequency regardless of the conditioned culture, reflecting biases toward frequent pretraining terms irrespective of relevance. We hope that the MEMOed framework and our insights will inspire more works on attributing model performance on pretraining data. 

**Abstract (ZH)**: 在像叙事写作或对话这样的开放式生成任务中，大型语言模型常常表现出文化偏见，显示出有限的知识，并为不常见的文化生成模板化的输出。最近的研究表明，这些偏见可能源自预训练数据集中不均衡的文化代表性。本研究通过分析模型基于预训练数据模式如何将实体与文化关联起来，探讨预训练如何导致带有偏见的文化条件生成。我们提出了MEMOed框架（MEMOrization from pretraining document），以确定某个文化生成是否来源于记忆。利用MEMOed框架，对110种文化的食物和服饰生成进行分析，我们发现，预训练数据中高频率出现的文化生成更多包含记忆化的符号，而一些低频率出现的文化则没有生成内容。此外，模型倾向于生成高频实体，无论是在何种文化条件下，这反映了对高频预训练术语的偏见，而不论其相关性如何。我们希望通过MEMOed框架和我们的洞见，激发更多研究关注在预训练数据上分配模型性能的研究。 

---
# Depression and Anxiety Prediction Using Deep Language Models and Transfer Learning 

**Title (ZH)**: 使用深度语言模型和迁移学习进行抑郁和焦虑预测 

**Authors**: Tomasz Rutowski, Elizabeth Shriberg, Amir Harati, Yang Lu, Piotr Chlebek, Ricardo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2412.20741)  

**Abstract**: Digital screening and monitoring applications can aid providers in the management of behavioral health conditions. We explore deep language models for detecting depression, anxiety, and their co-occurrence from conversational speech collected during 16k user interactions with an application. Labels come from PHQ-8 and GAD-7 results also collected by the application. We find that results for binary classification range from 0.86 to 0.79 AUC, depending on condition and co-occurrence. Best performance is achieved when a user has either both or neither condition, and we show that this result is not attributable to data skew. Finally, we find evidence suggesting that underlying word sequence cues may be more salient for depression than for anxiety. 

**Abstract (ZH)**: 数字筛查和监测应用可以在行为健康状况的管理中为提供者提供帮助。我们探索了深度语言模型从应用中收集的16000名用户互动中的对话语音中检测抑郁、焦虑及其共发性的能力。标签来自应用同时收集的PHQ-8和GAD-7结果。我们发现二分类结果的AUC范围为0.86至0.79，具体取决于条件及其共发性。当用户同时患有或均未患有这两种状况时，性能最佳，并且我们证明这种结果并非由于数据偏斜所致。最后，我们发现证据表明，潜在的词汇序列线索可能比焦虑更为重要地提示抑郁。 

---
# Align Attention Heads Before Merging Them: An Effective Way for Converting MHA to GQA 

**Title (ZH)**: 在合并之前对注意力头进行对齐：一种将MHA转换为GQA的有效方法 

**Authors**: Qingyun Jin, Xiaohui Song, Feng Zhou, Zengchang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.20677)  

**Abstract**: Large language models have been shown to perform well on a variety of natural language processing problems. However, as the model size and the input sequence's length increase, the rapid increase of KV Cache significantly slows down inference speed. Therefore GQA model, as an alternative to MHA model, has been widely introduced into LLMs. In this work, we propose a low-cost method for pruning MHA models into GQA models with any compression ratio of key-value heads. Our method is based on $\mathit{L_0}$ masks to gradually remove redundant parameters. In addition, we apply orthogonal transformations to attention heads without changing the model to increase similarity between attention heads before pruning training, in order to further improve performance of the model. Our method can be compatible with rotary position embedding (RoPE), which means the model after training can be fully adapted to the mainstream standard GQA framework. Experiments demonstrate that our strategy can compress up to 87.5% of key-value heads of the LLaMA2-7B model without too much performance degradation, just achieved through supervised fine-tuning. 

**Abstract (ZH)**: 大型语言模型在各种自然语言处理问题上表现出色。然而，随着模型规模和输入序列长度的增加，KV缓存的快速增长显著减慢了推理速度。因此，作为一种替代多头注意力模型（MHA）的方案，全局查询逼近模型（GQA）被广泛应用于大规模语言模型（LLMs）中。本文提出了一种低成本的方法，可以将任意压缩比的MHA模型中的键值头剪枝转化为GQA模型。我们的方法基于$\mathbf{L_0}$掩码，逐步移除冗余参数。此外，我们通过应用正交变换来增加剪枝训练前注意力头之间的相似性，从而进一步提高模型性能。我们的方法可以与旋转位置嵌入（RoPE）兼容，这意味着训练后的模型可以完全适应主流标准的GQA框架。实验表明，我们的策略可以通过监督微调压缩掉LLaMA2-7B模型中高达87.5%的关键值头，同时性能下降不大。 

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
# GliLem: Leveraging GliNER for Contextualized Lemmatization in Estonian 

**Title (ZH)**: GliLem：利用GliNER进行爱沙尼亚语的上下文规范化词干提取 

**Authors**: Aleksei Dorkin, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2412.20597)  

**Abstract**: We present GliLem -- a novel hybrid lemmatization system for Estonian that enhances the highly accurate rule-based morphological analyzer Vabamorf with an external disambiguation module based on GliNER -- an open vocabulary NER model that is able to match text spans with text labels in natural language. We leverage the flexibility of a pre-trained GliNER model to improve the lemmatization accuracy of Vabamorf by 10\% compared to its original disambiguation module and achieve an improvement over the token classification-based baseline. To measure the impact of improvements in lemmatization accuracy on the information retrieval downstream task, we first created an information retrieval dataset for Estonian by automatically translating the DBpedia-Entity dataset from English. We benchmark several token normalization approaches, including lemmatization, on the created dataset using the BM25 algorithm. We observe a substantial improvement in IR metrics when using lemmatization over simplistic stemming. The benefits of improving lemma disambiguation accuracy manifest in small but consistent improvement in the IR recall measure, especially in the setting of high k. 

**Abstract (ZH)**: 我们提出了一种名为GliLem的新型混合词干化系统，该系统通过结合基于规则的高精度形态分析器Vabamorf和基于GliNER的外部消歧模块来增强Vabamorf的功能。GliNER是一个开放词汇量的命名实体识别模型，能够匹配自然语言中的文本片段与文本标签。我们利用预训练的GliNER模型的灵活性，通过改进Vabamorf的消歧模块，使其词干化准确性提高了10%，并在基于标记分类的基本方法上实现了提升。为了评估词干化准确性提高对下游信息检索任务的影响，我们首先通过自动将DBpedia-Entity数据集从英语翻译成 Estonian，创建了一个信息检索数据集。我们使用BM25算法对多个标记规范化方法（包括词干化）进行了基准测试。我们观察到使用词干化比简单的词干提取在信息检索指标上有显著改善。提高词干消歧准确性的益处体现在对召回率指标的小幅但持续的改进中，尤其是在高k值的设置中。 

---
# Controlling Out-of-Domain Gaps in LLMs for Genre Classification and Generated Text Detection 

**Title (ZH)**: 控制大规模语言模型中域外差距以进行体裁分类和生成文本检测

解析：
- "Controlling" 翻译为“控制”或“管理”，这里选择“控制”更为贴切。
- "Out-of-Domain Gaps" 翻译为“域外差距”，这是机器学习和自然语言处理领域常用的术语。
- "LLMs" 是“Large Language Models”的缩写，翻译为“大规模语言模型”。
- “Genre Classification” 翻译为“体裁分类”。
- “Generated Text Detection” 翻译为“生成文本检测”。

这样的翻译既保持了原文的专业性，又符合中文的学术表达习惯。 

**Authors**: Dmitri Roussinov, Serge Sharoff, Nadezhda Puchnina  

**Link**: [PDF](https://arxiv.org/pdf/2412.20595)  

**Abstract**: This study demonstrates that the modern generation of Large Language Models (LLMs, such as GPT-4) suffers from the same out-of-domain (OOD) performance gap observed in prior research on pre-trained Language Models (PLMs, such as BERT). We demonstrate this across two non-topical classification tasks: 1) genre classification and 2) generated text detection. Our results show that when demonstration examples for In-Context Learning (ICL) come from one domain (e.g., travel) and the system is tested on another domain (e.g., history), classification performance declines significantly.
To address this, we introduce a method that controls which predictive indicators are used and which are excluded during classification. For the two tasks studied here, this ensures that topical features are omitted, while the model is guided to focus on stylistic rather than content-based attributes. This approach reduces the OOD gap by up to 20 percentage points in a few-shot setup. Straightforward Chain-of-Thought (CoT) methods, used as the baseline, prove insufficient, while our approach consistently enhances domain transfer performance. 

**Abstract (ZH)**: 本研究展示了现代大型语言模型（LLMs，如GPT-4）在跨领域（OOD）性能方面与先前对预训练语言模型（PLMs，如BERT）的研究中观察到的差距是一致的。我们在两个非主题分类任务中证明了这一点：1）体裁分类；2）生成文本检测。研究结果表明，当上下文学习（ICL）的示例来自一个领域（例如，旅游）而系统在另一个领域（例如，历史）上进行测试时，分类性能会显著下降。

为解决这一问题，我们引入了一种方法，该方法在分类过程中控制哪些预测指标被使用，哪些被排除。对于这里研究的两个任务，这种方法确保排除了与主题相关的特点，同时引导模型专注于风格而非内容属性。这种方法在少量示例（few-shot）设置下将跨领域差距减少了最多20个百分点。作为基线的直接思维链（CoT）方法证明效果不足，而我们的方法在领域迁移性能上持续表现出改进。 

---
# Towards Neural No-Resource Language Translation: A Comparative Evaluation of Approaches 

**Title (ZH)**: 资源受限语言翻译的神经网络方法：一种比较评价 

**Authors**: Madhavendra Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2412.20584)  

**Abstract**: No-resource languages - those with minimal or no digital representation - pose unique challenges for machine translation (MT). Unlike low-resource languages, which rely on limited but existent corpora, no-resource languages often have fewer than 100 sentences available for training. This work explores the problem of no-resource translation through three distinct workflows: fine-tuning of translation-specific models, in-context learning with large language models (LLMs) using chain-of-reasoning prompting, and direct prompting without reasoning. Using Owens Valley Paiute as a case study, we demonstrate that no-resource translation demands fundamentally different approaches from low-resource scenarios, as traditional approaches to machine translation, such as those that work for low-resource languages, fail. Empirical results reveal that, although traditional approaches fail, the in-context learning capabilities of general-purpose large language models enable no-resource language translation that outperforms low-resource translation approaches and rivals human translations (BLEU 0.45-0.6); specifically, chain-of-reasoning prompting outperforms other methods for larger corpora, while direct prompting exhibits advantages in smaller datasets. As these approaches are language-agnostic, they have potential to be generalized to translation tasks from a wide variety of no-resource languages without expert input. These findings establish no-resource translation as a distinct paradigm requiring innovative solutions, providing practical and theoretical insights for language preservation. 

**Abstract (ZH)**: 无资源语言——那些几乎没有或完全没有数字表示的语言——为机器翻译（MT）带来了独特的挑战。与低资源语言相比，无资源语言通常仅有不到100个句子可用于训练，依赖的是有限但存在的语料库。本研究通过三种不同的工作流程探索了无资源翻译的问题：针对翻译的微调模型、使用链式推理提示的大语言模型（LLMs）的上下文学习，以及无需推理的直接提示。以欧沃内瓦尔派图语（Owens Valley Paiute）为例，我们证明了在无资源翻译中，传统方法与低资源翻译场景的方法都无法有效应对，这些传统方法对于低资源语言的机器翻译方法在此无效。实验结果表明，尽管传统方法失效，但大语言模型的一般性上下文学习能力能够实现比低资源语言翻译方法更好的无资源语言翻译效果，并且该效果与人工翻译相当（BLEU得分为0.45-0.6）；具体而言，链式推理提示在大语料库方面表现更优，而直接提示在小数据集上则具有优势。由于这些方法具有跨语言适用性，它们有可能为广泛种类的无资源语言翻译任务提供通用解决方案，无需专家的辅助。这些研究成果将无资源翻译确立为需要创新解决方案的独特范式，为语言保存提供了实用和理论上的见解。 

---
# Counterfactual Samples Constructing and Training for Commonsense Statements Estimation 

**Title (ZH)**: 用于常见常识陈述估计的反事实样本构建与训练 

**Authors**: Chong Liu, Zaiwen Feng, Lin Liu, Zhenyun Deng, Jiuyong Li, Ruifang Zhai, Debo Cheng, Li Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.20563)  

**Abstract**: Plausibility Estimation (PE) plays a crucial role for enabling language models to objectively comprehend the real world. While large language models (LLMs) demonstrate remarkable capabilities in PE tasks but sometimes produce trivial commonsense errors due to the complexity of commonsense knowledge. They lack two key traits of an ideal PE model: a) Language-explainable: relying on critical word segments for decisions, and b) Commonsense-sensitive: detecting subtle linguistic variations in commonsense. To address these issues, we propose a novel model-agnostic method, referred to as Commonsense Counterfactual Samples Generating (CCSG). By training PE models with CCSG, we encourage them to focus on critical words, thereby enhancing both their language-explainable and commonsense-sensitive capabilities. Specifically, CCSG generates counterfactual samples by strategically replacing key words and introducing low-level dropout within sentences. These counterfactual samples are then incorporated into a sentence-level contrastive training framework to further enhance the model's learning process. Experimental results across nine diverse datasets demonstrate the effectiveness of CCSG in addressing commonsense reasoning challenges, with our CCSG method showing 3.07% improvement against the SOTA methods. 

**Abstract (ZH)**: 可信度估计（PE）在使语言模型客观理解现实世界方面发挥着重要作用。虽然大规模语言模型（LLMs）在PE任务中表现出色，但在处理常识知识的复杂性时，有时会因常识的琐碎性而产生常识错误。它们缺乏理想PE模型的两个关键特质：a）语言可解释性：依赖于关键词段做出决策，以及 b）常识敏感性：检测常识中的微妙语言变化。为解决这些问题，我们提出了一种新的模型无关方法，称为常识反事实样本生成（CCSG）。通过使用CCSG训练PE模型，我们鼓励它们专注于关键词，从而增强其语言可解释性和常识敏感性。具体而言，CCSG通过战略性地替换关键词并引入句内低水平dropout生成反事实样本。这些反事实样本随后被纳入句子级别的对比训练框架中，进一步提高模型的学习过程。在九个不同数据集上的实验结果显示，CCSG在解决常识推理挑战方面具有有效性，我们的CCSG方法在与现有最佳方法（SOTA）相比时，显示出3.07%的改进。 

---
# SAFE-MEME: Structured Reasoning Framework for Robust Hate Speech Detection in Memes 

**Title (ZH)**: SAFE-MEME： meme中鲁棒仇恨言论检测的结构化推理框架 

**Authors**: Palash Nandi, Shivam Sharma, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2412.20541)  

**Abstract**: Memes act as cryptic tools for sharing sensitive ideas, often requiring contextual knowledge to interpret. This makes moderating multimodal memes challenging, as existing works either lack high-quality datasets on nuanced hate categories or rely on low-quality social media visuals. Here, we curate two novel multimodal hate speech datasets, MHS and MHS-Con, that capture fine-grained hateful abstractions in regular and confounding scenarios, respectively. We benchmark these datasets against several competing baselines. Furthermore, we introduce SAFE-MEME (Structured reAsoning FramEwork), a novel multimodal Chain-of-Thought-based framework employing Q&A-style reasoning (SAFE-MEME-QA) and hierarchical categorization (SAFE-MEME-H) to enable robust hate speech detection in memes. SAFE-MEME-QA outperforms existing baselines, achieving an average improvement of approximately 5% and 4% on MHS and MHS-Con, respectively. In comparison, SAFE-MEME-H achieves an average improvement of 6% in MHS while outperforming only multimodal baselines in MHS-Con. We show that fine-tuning a single-layer adapter within SAFE-MEME-H outperforms fully fine-tuned models in regular fine-grained hateful meme detection. However, the fully fine-tuning approach with a Q&A setup is more effective for handling confounding cases. We also systematically examine the error cases, offering valuable insights into the robustness and limitations of the proposed structured reasoning framework for analyzing hateful memes. 

**Abstract (ZH)**: 模因作为隐秘工具，用于传递敏感思想，常常需要特定的情境知识才能解读。这使得对多模态模因进行管理变得具有挑战性，因为现有的工作要么缺乏针对细微厌恶类别的高质量数据集，要么依赖于低质量的社会媒体视觉素材。在此，我们整理了两个全新的多模态仇恨言论数据集MHS和MHS-Con，分别捕捉了常规和混淆场景下的细微仇恨抽象。我们使用这些数据集对多个竞争基准模型进行了评估。此外，我们引入了SAFE-MEME（结构化推理框架），这是一种新颖的多模态因果推理框架，采用问答式推理（SAFE-MEME-QA）与层次分类（SAFE-MEME-H），以在模因中实现稳健的仇恨言论检测。SAFE-MEME-QA 在MHS 和 MHS-Con 上的性能优于现有基线，分别提高了约5% 和4%。相比之下，在MHS上，SAFE-MEME-H 在 MHS上实现了约6%的平均改进，同时在MHS-Con上仅优于多模态基础模型。我们展示了在SAFE-MEME-H中对单一层数适应器进行微调优于完全微调模型在常规细微仇恨模因检测中的表现，但在处理混淆情况时，问答式设置的完全微调方法更为有效。我们还系统地分析了错误案例，提供了有关所提结构化推理框架分析仇恨模因的稳健性和局限性的宝贵见解。 

---
# Cut the Deadwood Out: Post-Training Model Purification with Selective Module Substitution 

**Title (ZH)**: 修剪冗余部分：基于选择性模块替换的后训练模型精简 

**Authors**: Yao Tong, Weijun Li, Xuanli He, Haolan Zhan, Qiongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20476)  

**Abstract**: The success of DNNs often depends on training with large-scale datasets, but building such datasets is both expensive and challenging. Consequently, public datasets from open-source platforms like HuggingFace have become popular, posing significant risks of data poisoning attacks. Existing backdoor defenses in NLP primarily focus on identifying and removing poisoned samples; however, purifying a backdoored model with these sample-cleaning approaches typically requires expensive retraining. Therefore, we propose Greedy Module Substitution (GMS), which identifies and substitutes ''deadwood'' modules (i.e., components critical to backdoor pathways) in a backdoored model to purify it. Our method relaxes the common dependency of prior model purification methods on clean datasets or clean auxiliary models. When applied to RoBERTa-large under backdoor attacks, GMS demonstrates strong effectiveness across various settings, particularly against widely recognized challenging attacks like LWS, achieving a post-purification attack success rate (ASR) of 9.7% on SST-2 compared to 58.8% for the best baseline approach. 

**Abstract (ZH)**: 深层神经网络（DNN）的成功往往依赖于大规模数据集的训练，但构建这样的数据集既昂贵又具有挑战性。因此，来自开源平台如HuggingFace的公共数据集变得非常流行，但这也带来了数据中毒攻击的巨大风险。现有的自然语言处理（NLP）后门防御主要集中在识别并移除中毒样本上；然而，使用这些样本清理方法来净化受后门污染的模型通常需要昂贵的重新训练。因此，我们提出了贪婪模块替换（Greedy Module Substitution，GMS）方法，该方法通过识别并替换受后门污染模型中的“无用模块”（即对后门路径至关重要的组件）来实现模型净化。我们的方法减轻了之前模型净化方法对干净数据集或干净辅助模型的依赖性。当GMS应用于遭受后门攻击的RoBERTa-large时，这种方法在各种场景下都表现出强大的效果，特别是在应对广泛认可的具有挑战性的攻击如LWS时，与最好的基线方法相比，在SST-2上的后门攻击成功率（Attack Success Rate，ASR）从58.8%降低到了9.7%。 

---
# Utilizing Multimodal Data for Edge Case Robust Call-sign Recognition and Understanding 

**Title (ZH)**: 利用多模态数据提高边缘情况下的呼叫标识识别与理解 

**Authors**: Alexander Blatt, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2412.20467)  

**Abstract**: Operational machine-learning based assistant systems must be robust in a wide range of scenarios. This hold especially true for the air-traffic control (ATC) domain. The robustness of an architecture is particularly evident in edge cases, such as high word error rate (WER) transcripts resulting from noisy ATC recordings or partial transcripts due to clipped recordings. To increase the edge-case robustness of call-sign recognition and understanding (CRU), a core tasks in ATC speech processing, we propose the multimodal call-sign-command recovery model (CCR). The CCR architecture leads to an increase in the edge case performance of up to 15%. We demonstrate this on our second proposed architecture, CallSBERT. A CRU model that has less parameters, can be fine-tuned noticeably faster and is more robust during fine-tuning than the state of the art for CRU. Furthermore, we demonstrate that optimizing for edge cases leads to a significantly higher accuracy across a wide operational range. 

**Abstract (ZH)**: 基于操作的机器学习辅助系统在各种场景下必须具备鲁棒性，特别是在空中交通管制（ATC）领域，这一点尤为重要。架构的鲁棒性在极端情况下尤为明显，例如由于嘈杂的ATC录音导致的高字错误率（WER）转录或由于片段录音导致的部分转录。为提高呼号识别和理解（CRU，Cue Recogniion and Understanding）这一ATC语音处理的核心任务在边缘情况下的鲁棒性，我们提出了多模态呼号-命令恢复模型（CCR，Call Sign-Cue Recovery）。CCR架构在边缘情况下的性能提高了最多15%。我们在我们提出的第二个架构CallSBERT中展示了这一点。CallSBERT是一个参数更少、在微调过程中显著更快并具有更好的微调鲁棒性的CRU模型，其性能优于现有最先进的CRU模型。此外，我们证明了在边缘情况下的优化能够在广泛的运行范围内显著提高准确性。 

---
# Enhancing Entertainment Translation for Indian Languages using Adaptive Context, Style and LLMs 

**Title (ZH)**: 使用适应性上下文、风格和大规模语言模型增强印度语言的娱乐翻译 

**Authors**: Pratik Rakesh Singh, Mohammadi Zaki, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2412.20440)  

**Abstract**: We address the challenging task of neural machine translation (NMT) in the entertainment domain, where the objective is to automatically translate a given dialogue from a source language content to a target language. This task has various applications, particularly in automatic dubbing, subtitling, and other content localization tasks, enabling source content to reach a wider audience. Traditional NMT systems typically translate individual sentences in isolation, without facilitating knowledge transfer of crucial elements such as the context and style from previously encountered sentences. In this work, we emphasize the significance of these fundamental aspects in producing pertinent and captivating translations. We demonstrate their significance through several examples and propose a novel framework for entertainment translation, which, to our knowledge, is the first of its kind. Furthermore, we introduce an algorithm to estimate the context and style of the current session and use these estimations to generate a prompt that guides a Large Language Model (LLM) to generate high-quality translations. Our method is both language and LLM-agnostic, making it a general-purpose tool. We demonstrate the effectiveness of our algorithm through various numerical studies and observe significant improvement in the COMET scores over various state-of-the-art LLMs. Moreover, our proposed method consistently outperforms baseline LLMs in terms of win-ratio. 

**Abstract (ZH)**: 我们致力于娱乐领域中的神经机器翻译（NMT）这一具有挑战性的任务，目标是自动将给定的对话从源语言翻译为目标语言。这一任务在自动配音、字幕制作以及其他内容本地化任务中有着广泛的应用，使源内容能够触及更广泛的受众。传统的NMT系统通常独立地翻译个别句子，而没有将上文和风格等关键信息传递给后续的翻译过程。在本项研究中，我们强调了上下文和风格等基础要素对于生成相关且引人入胜的翻译的重要性。我们通过多个示例展示了这些要素的重要性，并提出了一个新颖的娱乐翻译框架，据我们所知，这是首款此类系统。此外，我们介绍了一种算法来估算当前会话的上下文和风格，并利用这些估计生成一个提示，指导大型语言模型（LLM）生成高质量的翻译。该方法既无需依赖特定语言，也无需依赖特定的LLM，因此具有通用性。我们通过多种数值研究验证了该算法的有效性，并观察到我们的方法在多种最先进的LLM上的COMET评分上显著提高。此外，我们的方法在胜败比上也持续优于基线LLM。 

---
# Integrating Natural Language Processing Techniques of Text Mining Into Financial System: Applications and Limitations 

**Title (ZH)**: 将自然语言处理技术融入金融系统：应用与局限性 

**Authors**: Denisa Millo, Blerina Vika, Nevila Baci  

**Link**: [PDF](https://arxiv.org/pdf/2412.20438)  

**Abstract**: The financial sector, a pivotal force in economic development, increasingly uses the intelligent technologies such as natural language processing to enhance data processing and insight extraction. This research paper through a review process of the time span of 2018-2023 explores the use of text mining as natural language processing techniques in various components of the financial system including asset pricing, corporate finance, derivatives, risk management, and public finance and highlights the need to address the specific problems in the discussion section. We notice that most of the research materials combined probabilistic with vector-space models, and text-data with numerical ones. The most used technique regarding information processing is the information classification technique and the most used algorithms include the long-short term memory and bidirectional encoder models. The research noticed that new specific algorithms are developed and the focus of the financial system is mainly on asset pricing component. The research also proposes a path from engineering perspective for researchers who need to analyze financial text. The challenges regarding text mining perspective such as data quality, context-adaption and model interpretability need to be solved so to integrate advanced natural language processing models and techniques in enhancing financial analysis and prediction. Keywords: Financial System (FS), Natural Language Processing (NLP), Software and Text Engineering, Probabilistic, Vector-Space, Models, Techniques, TextData, Financial Analysis. 

**Abstract (ZH)**: 金融部门作为经济发展的重要推动力，越来越多地利用自然语言处理等智能技术以增强数据处理能力和洞察提取能力。本研究通过2018-2023年的文献综述，探讨了自然语言处理技术（特别是文本挖掘技术）在金融系统各个组件中的应用，包括资产定价、公司财务、衍生品、风险管理以及公共财政，并在讨论部分强调了需要解决的具体问题。我们注意到，大部分的研究材料结合了概率与向量空间模型，以及文本数据与数值数据的融合。在信息处理方面，最常用的技术是信息分类技术，最常用到的算法包括长短时记忆网络和双向编码器模型。研究发现新开发的特定算法正在被应用，且金融系统的主要关注点在于资产定价组件。本研究也为需要分析金融文本的数据工程师提出了一个研究路径。从文本挖掘的角度出发，需要解决的数据质量、上下文适应性和模型可解释性等挑战，以实现将先进的自然语言处理模型和技术更好地应用于金融分析和预测。关键词：金融系统（FS）、自然语言处理（NLP）、软件和文本工程、概率模型、向量空间模型、文本数据、金融分析。 

---
# Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection 

**Title (ZH)**: 多语言地理实体检测中高级NLP模型和大语言模型的性能比较 

**Authors**: Kalin Kopanov  

**Link**: [PDF](https://arxiv.org/pdf/2412.20414)  

**Abstract**: The integration of advanced Natural Language Processing (NLP) methodologies and Large Language Models (LLMs) has significantly enhanced the extraction and analysis of geospatial data from multilingual texts, impacting sectors such as national and international security. This paper presents a comprehensive evaluation of leading NLP models -- SpaCy, XLM-RoBERTa, mLUKE, GeoLM -- and LLMs, specifically OpenAI's GPT 3.5 and GPT 4, within the context of multilingual geo-entity detection. Utilizing datasets from Telegram channels in English, Russian, and Arabic, we examine the performance of these models through metrics such as accuracy, precision, recall, and F1 scores, to assess their effectiveness in accurately identifying geospatial references. The analysis exposes each model's distinct advantages and challenges, underscoring the complexities involved in achieving precise geo-entity identification across varied linguistic landscapes. The conclusions drawn from this experiment aim to direct the enhancement and creation of more advanced and inclusive NLP tools, thus advancing the field of geospatial analysis and its application to global security. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

高级自然语言处理（NLP）方法与大型语言模型（LLMs）的结合，显著增强了从多语言文本中提取和分析地理空间数据的能力，影响了国家和国际安全等领域。本文在多语言地理实体检测的背景下，评估了领先的NLP模型——SpaCy、XLM-RoBERTa、mLUKE、GeoLM以及特定的LLMs，即OpenAI的GPT 3.5和GPT 4。我们使用从Telegram频道收集的英文、俄文和阿拉伯文数据集，通过准确率、精确率、召回率和F1分数等指标来评估这些模型的表现，以评估它们在准确识别地理空间参考方面的效果。分析揭示了每种模型的独特优势和挑战，突显了在多语言环境中实现精确地理实体识别所面临的复杂性。本文得出的结论旨在指导NLP工具的进一步完善和创新，从而推动地理空间分析领域及其在全球安全应用中的发展。 

---
# Multi-Objective Large Language Model Unlearning 

**Title (ZH)**: 多目标大型语言模型去学习 

**Authors**: Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.20412)  

**Abstract**: Machine unlearning in the domain of large language models (LLMs) has attracted great attention recently, which aims to effectively eliminate undesirable behaviors from LLMs without full retraining from scratch. In this paper, we explore the Gradient Ascent (GA) approach in LLM unlearning, which is a proactive way to decrease the prediction probability of the model on the target data in order to remove their influence. We analyze two challenges that render the process impractical: gradient explosion and catastrophic forgetting. To address these issues, we propose Multi-Objective Large Language Model Unlearning (MOLLM) algorithm. We first formulate LLM unlearning as a multi-objective optimization problem, in which the cross-entropy loss is modified to the unlearning version to overcome the gradient explosion issue. A common descent update direction is then calculated, which enables the model to forget the target data while preserving the utility of the LLM. Our empirical results verify that MoLLM outperforms the SOTA GA-based LLM unlearning methods in terms of unlearning effect and model utility preservation. 

**Abstract (ZH)**: 在大型语言模型（LLMs）领域中的机器遗忘 recently 在近期引起了广泛关注，其目标是在无需从头完全重新训练的情况下，有效消除模型中的不良行为。在本文中，我们探讨了使用梯度上升（Gradient Ascent, GA）方法在LLMs遗忘中的应用，这是一种主动减少模型对目标数据的预测概率的方法，以去除其影响。我们分析了两个使得该过程难以实现的挑战：梯度爆炸和灾难性遗忘。为了解决这些问题，我们提出了多目标大型语言模型遗忘（Multi-Objective Large Language Model Unlearning, MOLLM）算法。我们首先将LLMs遗忘问题形式化为一个多目标优化问题，通过修改交叉熵损失来克服梯度爆炸问题。然后计算出一个通用的下降更新方向，使得模型能够在遗忘目标数据的同时保留LLMs的有用性。我们的实验证明，MOLLM在遗忘效果和模型有用性保留方面优于最新的基于GA的LLMs遗忘方法。 

---
# Natural Language Fine-Tuning 

**Title (ZH)**: 自然语言微调 

**Authors**: Jia Liu, Yue Wang, Zhiqi Lin, Min Chen, Yixue Hao, Long Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20382)  

**Abstract**: Large language model fine-tuning techniques typically depend on extensive labeled data, external guidance, and feedback, such as human alignment, scalar rewards, and demonstration. However, in practical application, the scarcity of specific knowledge poses unprecedented challenges to existing fine-tuning techniques. In this paper, focusing on fine-tuning tasks in specific domains with limited data, we introduce Natural Language Fine-Tuning (NLFT), which utilizes natural language for fine-tuning for the first time. By leveraging the strong language comprehension capability of the target LM, NLFT attaches the guidance of natural language to the token-level outputs. Then, saliency tokens are identified with calculated probabilities. Since linguistic information is effectively utilized in NLFT, our proposed method significantly reduces training costs. It markedly enhances training efficiency, comprehensively outperforming reinforcement fine-tuning algorithms in accuracy, time-saving, and resource conservation. Additionally, on the macro level, NLFT can be viewed as a token-level fine-grained optimization of SFT, thereby efficiently replacing the SFT process without the need for warm-up (as opposed to ReFT requiring multiple rounds of warm-up with SFT). Compared to SFT, NLFT does not increase the algorithmic complexity, maintaining O(n). Extensive experiments on the GSM8K dataset demonstrate that NLFT, with only 50 data instances, achieves an accuracy increase that exceeds SFT by 219%. Compared to ReFT, the time complexity and space complexity of NLFT are reduced by 78.27% and 92.24%, respectively. The superior technique of NLFT is paving the way for the deployment of various innovative LLM fine-tuning applications when resources are limited at network edges.
Our code has been released at this https URL. 

**Abstract (ZH)**: 大型语言模型微调技术通常依赖于大量的标注数据、外部指导和反馈，例如人工对齐、标量奖励和示范。然而，在实际应用中，特定知识的稀缺性给现有的微调技术带来了前所未有的挑战。在本论文中，针对有限数据的特定领域微调任务，我们引入了自然语言微调（NLFT）技术，这是首次利用自然语言进行微调。通过利用目标语言模型的强大语言理解能力，NLFT 将自然语言的指导附加到了令牌级别的输出上。然后，通过计算概率来识别显著令牌。由于语言信息在 NLFT 中得到了有效利用，我们的方法显著降低了训练成本，显著提高了训练效率，并在准确度、节省时间和资源节省方面全面优于强化学习微调算法。在宏观层面，NLFT 可以被视为 SFT（自我指导微调）的一种令牌级精细优化，在不需要预热的情况下能够有效替代 SFT 过程（相比之下，ReFT 需要多次预热）。与 SFT 相比，NLFT 保持了相同的算法复杂度，仍为 O(n)。大量实验表明，在 GSM8K 数据集上，使用仅 50 个数据实例的 NLFT 达到了比 SFT 高出 219% 的准确度提升。与 ReFT 相比，NLFT 的时间复杂度和空间复杂度分别降低了 78.27% 和 92.24%。NLFT 的优越技术为在资源受限的网络边缘部署各种创新的 LLM 微调应用铺平了道路。

我们的代码已在此处发布：[此处链接](此链接应替换为实际的网址)。 

---
# LLM2: Let Large Language Models Harness System 2 Reasoning 

**Title (ZH)**: LL-M2: 让大规模语言模型运用系统二推理 

**Authors**: Cheng Yang, Chufan Shi, Siheng Li, Bo Shui, Yujiu Yang, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2412.20372)  

**Abstract**: Large language models (LLMs) have exhibited impressive capabilities across a myriad of tasks, yet they occasionally yield undesirable outputs. We posit that these limitations are rooted in the foundational autoregressive architecture of LLMs, which inherently lacks mechanisms for differentiating between desirable and undesirable results. Drawing inspiration from the dual-process theory of human cognition, we introduce LLM2, a novel framework that combines an LLM (System 1) with a process-based verifier (System 2). Within LLM2, the LLM is responsible for generating plausible candidates, while the verifier provides timely process-based feedback to distinguish desirable and undesirable outputs. The verifier is trained with a pairwise comparison loss on synthetic process-supervision data generated through our token quality exploration strategy. Empirical results on mathematical reasoning benchmarks substantiate the efficacy of LLM2, exemplified by an accuracy enhancement from 50.3 to 57.8 (+7.5) for Llama3-1B on GSM8K. Furthermore, when combined with self-consistency, LLM2 achieves additional improvements, boosting major@20 accuracy from 56.2 to 70.2 (+14.0). 

**Abstract (ZH)**: 大型语言模型（LLMs）在众多任务中展现了令人印象深刻的性能，但在某些情况下也会产生不理想的输出。我们认为这些局限性源于LLMs的基础自回归架构，该架构固有地缺乏区分理想和不理想结果的机制。受到人类认知的双重过程理论的启发，我们提出了LLM2这一创新框架，该框架将一个LLM（系统1）与一个基于过程的验证器（系统2）相结合。在LLM2框架中，LLM负责生成可能的候选方案，而验证器则提供及时的过程反馈，以便区分理想的和不理想的输出。验证器通过我们的令牌质量探索策略生成的合成过程监督数据进行训练，采用成对比较损失函数进行训练。数学推理基准实验结果证实了LLM2的有效性，例如，对于Llama3-1B在GSM8K上的准确率从50.3提高到57.8（+7.5）。此外，当与自我一致性结合使用时，LLM2还能实现进一步的改进，将主要@20准确率从56.2提高到70.2（+14.0）。 

---
# HindiLLM: Large Language Model for Hindi 

**Title (ZH)**: HindiLLM：印地语大型语言模型 

**Authors**: Sanjay Chouhan, Shubha Brata Nath, Aparajita Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2412.20357)  

**Abstract**: The advancements in the Large Language Model (LLM) have helped in solving several problems related to language processing. Most of the researches have focused on the English language only, because of its popularity and abundance on the internet. However, a high-performance language model for Hindi and other Indic languages is lacking in the literature. In this work, we have pre-trained two autoregressive LLM models for the Hindi language, namely HindiLLM-Small and HindiLLM-Medium. We use a two-step process comprising unsupervised pre-training and supervised fine-tuning. First, we create a large and high-quality text corpus for unsupervised pre-training. Next, we train a Byte-Pair Encoding, named HindiLLM tokenizer, using the pre-training text data. We then perform training on the unlabeled data, known as the pre-training step, to get the HindiLLM base models. Furthermore, we perform fine-tuning of the HindiLLM base models for different tasks like sentiment analysis, text classification, natural language inference, and multiple choice question-answer on popular labeled datasets to measure the real-world performance. The evaluation shows that the HindiLLM-based fine-tuned models outperform several models in most of the language related tasks. 

**Abstract (ZH)**: 大型语言模型（LLM）的进步已经帮助解决了多种语言处理问题。大多数研究专注于英语，原因在于其在网络上的广泛使用和丰富的资源。然而，关于印地语及其他印度语言高性能语言模型的相关文献较为缺乏。在这项工作中，我们预训练了两个自回归的大型语言模型，即HindiLLM-Small和HindiLLM-Medium，用于印地语。我们采用两步过程，包括无监督预训练和有监督微调。首先，我们创建了一个规模大且质量高的无监督预训练文本语料库。然后，我们使用预训练文本数据训练了一种名为HindiLLM的字节对编码器（Byte-Pair Encoding）。接着，在未标记的数据上进行培训，即预训练步骤，以获得HindiLLM基础模型。此外，我们针对不同的任务，如情感分析、文本分类、自然语言推理和多项选择题的答案，对HindiLLM基础模型进行微调，以在流行的标注数据集上评估其在真实世界中的性能。评估结果显示，基于HindiLLM的微调模型在大多数语言相关任务中表现优于其他模型。 

---
# Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain 

**Title (ZH)**: 理解检索增强生成中信心的影响：医疗领域案例研究 

**Authors**: Shintaro Ozaki, Yuta Kato, Siyuan Feng, Masayo Tomita, Kazuki Hayashi, Ryoma Obara, Masafumi Oyamada, Katsuhiko Hayashi, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2412.20309)  

**Abstract**: Retrieval Augmented Generation (RAG) complements the knowledge of Large Language Models (LLMs) by leveraging external information to enhance response accuracy for queries. This approach is widely applied in several fields by taking its advantage of injecting the most up-to-date information, and researchers are focusing on understanding and improving this aspect to unlock the full potential of RAG in such high-stakes applications. However, despite the potential of RAG to address these needs, the mechanisms behind the confidence levels of its outputs remain underexplored, although the confidence of information is very critical in some domains, such as finance, healthcare, and medicine. Our study focuses the impact of RAG on confidence within the medical domain under various configurations and models. We evaluate confidence by treating the model's predicted probability as its output and calculating Expected Calibration Error (ECE) and Adaptive Calibration Error (ACE) scores based on the probabilities and accuracy. In addition, we analyze whether the order of retrieved documents within prompts calibrates the confidence. Our findings reveal large variation in confidence and accuracy depending on the model, settings, and the format of input prompts. These results underscore the necessity of optimizing configurations based on the specific model and conditions. 

**Abstract (ZH)**: 检索增强生成（RAG）通过利用外部信息来补充大型语言模型（LLMs）的知识，从而提高查询响应的准确性。该方法在多个领域得到了广泛应用，利用其注入最新信息的优势，并且研究人员正致力于理解和改进这一方面，以充分发挥RAG在高风险应用中的潜力。然而，尽管RAG有潜力解决这些需求，其输出的置信水平背后的机制仍被广泛忽视，尤其是在金融、医疗保健和医学等领域，信息的置信度至关重要。我们的研究集中在不同配置和模型下，RAG在医疗领域的置信度影响。我们通过将模型预测的概率视为输出，并基于这些概率和准确性计算期望校准误差（ECE）和自适应校准误差（ACE）得分来进行置信度评估。此外，我们还分析了提示中检索文档的顺序是否校准了置信度。研究结果表明，置信度和准确性在不同的模型、设置以及输入提示的格式上存在显著差异。这些结果强调了根据特定模型和条件优化配置的必要性。 

---
# No Preference Left Behind: Group Distributional Preference Optimization 

**Title (ZH)**: 《一个也不落下：群体分布偏好优化》 

**Authors**: Binwei Yao, Zefan Cai, Yun-Shiuan Chuang, Shanglin Yang, Ming Jiang, Diyi Yang, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20299)  

**Abstract**: Preferences within a group of people are not uniform but follow a distribution. While existing alignment methods like Direct Preference Optimization (DPO) attempt to steer models to reflect human preferences, they struggle to capture the distributional pluralistic preferences within a group. These methods often skew toward dominant preferences, overlooking the diversity of opinions, especially when conflicting preferences arise. To address this issue, we propose Group Distribution Preference Optimization (GDPO), a novel framework that aligns language models with the distribution of preferences within a group by incorporating the concept of beliefs that shape individual preferences. GDPO calibrates a language model using statistical estimation of the group's belief distribution and aligns the model with belief-conditioned preferences, offering a more inclusive alignment framework than traditional methods. In experiments using both synthetic controllable opinion generation and real-world movie review datasets, we show that DPO fails to align with the targeted belief distributions, while GDPO consistently reduces this alignment gap during training. Moreover, our evaluation metrics demonstrate that GDPO outperforms existing approaches in aligning with group distributional preferences, marking a significant advance in pluralistic alignment. 

**Abstract (ZH)**: 在一群人的群体中，偏好不是均匀的，而是遵循一个分布。现有的对齐方法，如直接偏好优化（DPO），试图引导模型反映人类的偏好，但在捕捉群体内部的分布性偏好方面遇到困难。这些方法往往偏向于主导的偏好，忽略了意见的多样性，尤其是在存在相互冲突的偏好时。为了解决这一问题，我们提出了群体分布偏好优化（GDPO）这一新颖框架，它通过引入塑造个体偏好的信念概念，使语言模型与群体偏好分布对齐。GDPO使用统计估计来校准语言模型的群体信念分布，并将模型与信念条件下的偏好对齐，提供了一个比传统方法更具包容性的对齐框架。在使用合成可控意见生成和真实世界电影评论数据集进行的实验中，我们表明DPO无法与目标信念分布对齐，而GDPO在训练过程中始终能减小这一对齐差距。此外，我们的评估指标表明，GDPO在与群体分布偏好对齐方面优于现有方法，标志着多偏好对齐的一个重要进步。 

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
# AfriHG: News headline generation for African Languages 

**Title (ZH)**: AfriHG：非洲语言新闻标题生成 

**Authors**: Toyib Ogunremi, Serah Akojenu, Anthony Soronnadi, Olubayo Adekanmbi, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2412.20223)  

**Abstract**: This paper introduces AfriHG -- a news headline generation dataset created by combining from XLSum and MasakhaNEWS datasets focusing on 16 languages widely spoken by Africa. We experimented with two seq2eq models (mT5-base and AfriTeVa V2), and Aya-101 LLM. Our results show that Africa-centric seq2seq models such as AfriTeVa V2 outperform the massively multilingual mT5-base model. Finally, we show that the performance of fine-tuning AfriTeVa V2 with 313M parameters is competitive to prompting Aya-101 LLM with more than 13B parameters. 

**Abstract (ZH)**: 本文介绍了AfriHG——一个结合了XLSum和MasakhaNEWS数据集而创建的新闻标题生成数据集，重点关注非洲广泛使用的16种语言。我们实验了两种seq2seq模型（mT5-base和AfriTeVa V2），以及Aya-101语言模型。实验结果表明，以非洲为中心的seq2seq模型，如AfriTeVa V2，在性能上优于大规模多语言的mT5-base模型。最后，我们展示了使用313M参数fine-tuning AfriTeVa V2的性能与使用超过13B参数进行提示的Aya-101语言模型相当。 

---
# YAD: Leveraging T5 for Improved Automatic Diacritization of Yor\`ub\'a Text 

**Title (ZH)**: YAD：利用T5提升约鲁巴文字的自动音标化性能 

**Authors**: Akindele Michael Olawole, Jesujoba O. Alabi, Aderonke Busayo Sakpere, David I. Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2412.20218)  

**Abstract**: In this work, we present Yorùbá automatic diacritization (YAD) benchmark dataset for evaluating Yorùbá diacritization systems. In addition, we pre-train text-to-text transformer, T5 model for Yorùbá and showed that this model outperform several multilingually trained T5 models. Lastly, we showed that more data and larger models are better at diacritization for Yorùbá 

**Abstract (ZH)**: 在本文中，我们提出了尤鲁巴自动音标标注（Yorùbá Automatic Diacritization, YAD）基准数据集，用于评估尤鲁巴音标系统的性能。此外，我们对T5模型进行了预训练以适应尤鲁巴语言，并展示了该模型在多种语言训练的T5模型中表现出优越性。最后，我们表明，更多的数据和更大的模型对于尤鲁巴音标标注更有效。 

---
# Decoding Emotion: Speech Perception Patterns in Individuals with Self-reported Depression 

**Title (ZH)**: 解码情绪：自我报告抑郁个体的语音感知模式 

**Authors**: Guneesh Vats, Priyanka Srivastava, Chiranjeevi Yarra  

**Link**: [PDF](https://arxiv.org/pdf/2412.20213)  

**Abstract**: The current study examines the relationship between self-reported depression and the perception of affective speech within the Indian population. PANAS and PHQ-9 were used to assess current mood and depression, respectively. Participants' emotional reactivity was recorded on a valence and arousal scale against the affective speech audio presented in a sequence. No significant differences between the depression and no-depression groups were observed for any of the emotional stimuli, except the audio file depicting neutral emotion. Significantly higher PANAS scores by the depression than the no-depression group indicate the impact of pre-disposed mood on the current mood status. Contrary to previous findings, this study did not observe reduced positive emotional reactivity by the depression group. However, the results demonstrated consistency in emotional reactivity for speech stimuli depicting sadness and anger across all measures of emotion perception. 

**Abstract (ZH)**: 本研究探讨了自报抑郁症状与印度人群中情感言语感知之间的关系。我们使用PANAS评价当前情绪状态，使用PHQ-9评估抑郁水平。参与者在面对一系列情感言语音频刺激时，其情绪反应被记录在效价和唤醒度量表上。除了展示中性情感的音频文件外，抑郁组与非抑郁组在所有情感刺激上的差异均不显著。抑郁组显著高于非抑郁组的PANAS评分表明内在情绪状态对当前情绪状态的影响。与先前的研究结果不同，本研究未观察到抑郁组在正面情感反应上的减弱。然而，研究结果表明，在情绪感知的所有测量指标中，展示悲伤和愤怒情感的言语刺激在情绪反应上保持了一致性。 

---
# Building a Rich Dataset to Empower the Persian Question Answering Systems 

**Title (ZH)**: 构建丰富的数据集以增强波斯语问答系统 

**Authors**: Mohsen Yazdinejad, Marjan Kaedi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20212)  

**Abstract**: Question answering systems provide short, precise, and specific answers to questions. So far, many robust question answering systems have been developed for English, while some languages with fewer resources, like Persian, have few numbers of standard dataset. In this study, a comprehensive open-domain dataset is presented for Persian. This dataset is called NextQuAD and has 7,515 contexts, including 23,918 questions and answers. Then, a BERT-based question answering model has been applied to this dataset using two pre-trained language models, including ParsBERT and XLM-RoBERTa. The results of these two models have been ensembled using mean logits. Evaluation on the development set shows 0.95 Exact Match (EM) and 0.97 Fl_score. Also, to compare the NextQuAD with other Persian datasets, our trained model on the NextQuAD, is evaluated on two other datasets named PersianQA and ParSQuAD. Comparisons show that the proposed model increased EM by 0.39 and 0.14 respectively in PersianQA and ParSQuAD-manual, while a slight EM decline of 0.007 happened in ParSQuAD-automatic. 

**Abstract (ZH)**: 问答系统提供简短、精确且具体的答案来回应问题。迄今为止，为英语开发了很多稳健的问答系统，而资源较少的语言如波斯语则缺乏标准的数据集。在这项研究中，我们提出了一个全面的开放式领域数据集，用于波斯语。该数据集名为NextQuAD，包含7,515个上下文，包括23,918个问题和答案。然后，我们使用两个预训练语言模型（ParsBERT和XLM-RoBERTa）在该数据集上应用了一个基于BERT的问答模型。通过平均logits对这两个模型的结果进行了整合。在开发集上的评估表明，两个模型的精确匹配（EM）得分为0.95，F1分数（Fl_score）得分为0.97。此外，为了与其他波斯语数据集进行比较，我们使用NextQuAD训练的模型在两个其他数据集（PersianQA和ParSQuAD）上进行了评估。比较结果显示，提出的模型在PersianQA上将EM提高了0.39，在ParSQuAD-manual上提高了0.14，而在ParSQuAD-automatic上的EM略有下降（0.007）。 

---
# Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering 

**Title (ZH)**: 高效的多代理协作与工具使用在复杂表格问答中的在线规划 

**Authors**: Wei Zhou, Mohsen Mesgar, Annemarie Friedrich, Heike Adel  

**Link**: [PDF](https://arxiv.org/pdf/2412.20145)  

**Abstract**: Complex table question answering (TQA) aims to answer questions that require complex reasoning, such as multi-step or multi-category reasoning, over data represented in tabular form. Previous approaches demonstrated notable performance by leveraging either closed-source large language models (LLMs) or fine-tuned open-weight LLMs. However, fine-tuning LLMs requires high-quality training data, which is costly to obtain, and utilizing closed-source LLMs poses accessibility challenges and leads to reproducibility issues. In this paper, we propose Multi-Agent Collaboration with Tool use (MACT), a framework that requires neither closed-source models nor fine-tuning. In MACT, a planning agent and a coding agent that also make use of tools collaborate to answer questions. Our experiments on four TQA benchmarks show that MACT outperforms previous SoTA systems on three out of four benchmarks and that it performs comparably to the larger and more expensive closed-source model GPT-4 on two benchmarks, even when using only open-weight models without any fine-tuning. We conduct extensive analyses to prove the effectiveness of MACT's multi-agent collaboration in TQA. 

**Abstract (ZH)**: 复杂表格问答（TQA）的目标是通过对以表格形式表示的数据进行多步或多类别推理以回答问题。之前的方法通过利用闭源大型语言模型（LLMs）或微调的开源权重LLMs展现了显著的表现。然而，微调LLMs需要高质量的训练数据，这往往成本高昂，而利用闭源LLMs则带来了访问性和可再现性的问题。在本文中，我们提出了多agent协作与工具使用（MACT）框架，该框架既不需要闭源模型，也不需要微调。在MACT中，规划agent和编写agent通过使用工具协同工作以回答问题。我们在四个TQA基准上的实验表明，MACT在三个基准上优于之前的SOTA系统，并且在两个基准上与更大且更昂贵的闭源模型GPT-4具有相似的表现，即使仅使用未经过微调的开源模型也是如此。我们进行了广泛的分析以证明MACT在TQA中的多agent协作的有效性。 

---
# M-MAD: Multidimensional Multi-Agent Debate Framework for Fine-grained Machine Translation Evaluation 

**Title (ZH)**: M-MAD：多维度多-agent 辩论框架在细致粒度机器翻译评估中的应用 

**Authors**: Zhaopeng Feng, Jiayuan Su, Jiamei Zheng, Jiahan Ren, Yan Zhang, Jian Wu, Hongwei Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20127)  

**Abstract**: Recent advancements in large language models (LLMs) have given rise to the LLM-as-a-judge paradigm, showcasing their potential to deliver human-like judgments. However, in the field of machine translation (MT) evaluation, current LLM-as-a-judge methods fall short of learned automatic metrics. In this paper, we propose Multidimensional Multi-Agent Debate (M-MAD), a systematic LLM-based multi-agent framework for advanced LLM-as-a-judge MT evaluation. Our findings demonstrate that M-MAD achieves significant advancements by (1) decoupling heuristic MQM criteria into distinct evaluation dimensions for fine-grained assessments; (2) employing multi-agent debates to harness the collaborative reasoning capabilities of LLMs; (3) synthesizing dimension-specific results into a final evaluation judgment to ensure robust and reliable outcomes. Comprehensive experiments show that M-MAD not only outperforms all existing LLM-as-a-judge methods but also competes with state-of-the-art reference-based automatic metrics, even when powered by a suboptimal model like GPT-4o mini. Detailed ablations and analysis highlight the superiority of our framework design, offering a fresh perspective for LLM-as-a-judge paradigm. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展催生了LLM-as-a-judge范式，展示了其在提供类人类判断方面的能力。然而，在机器翻译（MT）评估领域，当前的LLM-as-a-judge方法未能超过已学习的自动评估指标。在这篇论文中，我们提出了Multidimensional Multi-Agent Debate（M-MAD），这是一种基于LLM的多智能体框架，用于高级LLM-as-a-judge机器翻译评估。我们的研究结果表明，M-MAD通过以下几点实现了显著的进步：（1）将启发式的MQM标准拆分为多个评估维度，以实现精细评估；（2）利用多智能体辩论来利用LLM的协同推理能力；（3）将特定维度的结果综合为最终评估判断，以确保结果的稳健性和可靠性。全面的实验表明，M-MAD不仅优于所有现有的LLM-as-a-judge方法，而且能够在使用如GPT-4o mini这样次优模型时，与最新的参考基自动评估指标相媲美。详细的剖析和分析突显了我们框架设计的优越性，为LLM-as-a-judge范式提供了新的视角。我们的代码和数据已在此网址https://公开发布。 

---
# Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Dataset 

**Title (ZH)**: 利用大语言模型从混合长文档中提取信息：一个框架与数据集 

**Authors**: Chongjian Yue, Xinrun Xu, Xiaojun Ma, Lun Du, Zhiming Ding, Shi Han, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20072)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional performance in textual understanding and tabular reasoning tasks. However, their ability to comprehend and analyze hybrid text, containing textual and tabular data, remains unexplored. The hybrid text often appears in the form of hybrid long documents (HLDs), which far exceed the token limit of LLMs. Consequently, we apply an Automated Information Extraction framework (AIE) to enable LLMs to process the HLDs and carry out experiments to analyse four important aspects of information extraction from HLDs. Given the findings: 1) The effective way to select and summarize the useful part of a HLD. 2) An easy table serialization way is enough for LLMs to understand tables. 3) The naive AIE has adaptability in many complex scenarios. 4) The useful prompt engineering to enhance LLMs on HLDs. To address the issue of dataset scarcity in HLDs and support future work, we also propose the Financial Reports Numerical Extraction (FINE) dataset. The dataset and code are publicly available in the attachments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本理解和表格推理任务中表现出色。然而，它们在理解和分析包含文本和表格数据的混合文本方面的能力尚未得到探索。混合文本通常以混合长文档（HLDs）的形式出现，远远超过了LLMs的token限制。因此，我们应用了一种自动信息提取框架（AIE），使LLMs能够处理这些HLDs，并进行实验以分析HLDs中信息提取的四个重要方面。根据我们的发现：1) 有效选择和总结HLD中有用部分的方式。2) 对于LLMs来说，一种简单的表格序列化方法就足够理解表格了。3) 简单的AIE在许多复杂场景中具有适应性。4) 有用的提示工程方法可以提升LLMs在处理HLDs方面的表现。为了解决HLDs数据集稀缺的问题并支持未来工作，我们还提出了金融报告数值提取（FINE）数据集。数据集和代码已在附件中公开提供。 

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
# STAYKATE: Hybrid In-Context Example Selection Combining Representativeness Sampling and Retrieval-based Approach -- A Case Study on Science Domains 

**Title (ZH)**: STAYKATE：结合代表性抽样与基于检索的方法的混合上下文示例选择——以科学领域为例的研究 

**Authors**: Chencheng Zhu, Kazutaka Shimada, Tomoki Taniguchi, Tomoko Ohkuma  

**Link**: [PDF](https://arxiv.org/pdf/2412.20043)  

**Abstract**: Large language models (LLMs) demonstrate the ability to learn in-context, offering a potential solution for scientific information extraction, which often contends with challenges such as insufficient training data and the high cost of annotation processes. Given that the selection of in-context examples can significantly impact performance, it is crucial to design a proper method to sample the efficient ones. In this paper, we propose STAYKATE, a static-dynamic hybrid selection method that combines the principles of representativeness sampling from active learning with the prevalent retrieval-based approach. The results across three domain-specific datasets indicate that STAYKATE outperforms both the traditional supervised methods and existing selection methods. The enhancement in performance is particularly pronounced for entity types that other methods pose challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现了从上下文学习的能力，这为科学信息提取提供了潜在的解决方案，科学信息提取常常面临训练数据不足和标注过程成本高昂的挑战。鉴于在上下文中的示例选择对性能有重大影响，设计一种有效示例的选择方法至关重要。本文提出了一种名为STAYKATE的静态-动态混合选择方法，该方法结合了主动学习中的代表性采样原则和普遍存在的基于检索的方法。在三个特定领域的数据集上的结果表明，STAYKATE在性能上优于传统的监督方法和现有的选择方法，特别是在其他方法难以处理的实体类型上效果尤为显著。 

---
# OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System 

**Title (ZH)**: OneKE：一种基于模式指导的大规模语言模型代理的知识提取系统（Docker 化版本） 

**Authors**: Yujie Luo, Xiangyuan Ru, Kangwei Liu, Lin Yuan, Mengshu Sun, Ningyu Zhang, Lei Liang, Zhiqiang Zhang, Jun Zhou, Lanning Wei, Da Zheng, Haofen Wang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20005)  

**Abstract**: We introduce OneKE, a dockerized schema-guided knowledge extraction system, which can extract knowledge from the Web and raw PDF Books, and support various domains (science, news, etc.). Specifically, we design OneKE with multiple agents and a configure knowledge base. Different agents perform their respective roles, enabling support for various extraction scenarios. The configure knowledge base facilitates schema configuration, error case debugging and correction, further improving the performance. Empirical evaluations on benchmark datasets demonstrate OneKE's efficacy, while case studies further elucidate its adaptability to diverse tasks across multiple domains, highlighting its potential for broad applications. We have open-sourced the Code at this https URL and released a Video at this http URL. 

**Abstract (ZH)**: 我们引入了一种名为OneKE的Schema-Guided知识提取系统，该系统基于Docker化设计，能够从互联网和原始PDF书籍中提取知识，并支持多种领域（如科学、新闻等）。具体而言，我们设计了OneKE，采用多个代理和配置型知识库。不同的代理各自承担不同的角色，从而实现多种提取场景的支持。配置型知识库有助于元数据配置、错误案例的调试与修正，进一步提升系统的性能。基准数据集上的实证评估证明了OneKE的有效性，而案例研究进一步阐述了其在多个领域的多样化任务中的适用性，展示了其广泛的潜在应用前景。我们已在以下地址开源了代码，并发布了介绍视频：

[代码链接](this https URL)
[视频链接](this http URL)

请注意替换上述Markdown格式中的占位符链接 `this https URL` 和 `this http URL` 为实际的链接地址。 

---
# Bridging Context Gaps: Enhancing Comprehension in Long-Form Social Conversations Through Contextualized Excerpts 

**Title (ZH)**: 填补背景差距：通过上下文化片段增强长篇社交对话的理解力 

**Authors**: Shrestha Mohanty, Sarah Xuan, Jacob Jobraeel, Anurag Kumar, Deb Roy, Jad Kabbara  

**Link**: [PDF](https://arxiv.org/pdf/2412.19966)  

**Abstract**: We focus on enhancing comprehension in small-group recorded conversations, which serve as a medium to bring people together and provide a space for sharing personal stories and experiences on crucial social matters. One way to parse and convey information from these conversations is by sharing highlighted excerpts in subsequent conversations. This can help promote a collective understanding of relevant issues, by highlighting perspectives and experiences to other groups of people who might otherwise be unfamiliar with and thus unable to relate to these experiences. The primary challenge that arises then is that excerpts taken from one conversation and shared in another setting might be missing crucial context or key elements that were previously introduced in the original conversation. This problem is exacerbated when conversations become lengthier and richer in themes and shared experiences. To address this, we explore how Large Language Models (LLMs) can enrich these excerpts by providing socially relevant context. We present approaches for effective contextualization to improve comprehension, readability, and empathy. We show significant improvements in understanding, as assessed through subjective and objective evaluations. While LLMs can offer valuable context, they struggle with capturing key social aspects. We release the Human-annotated Salient Excerpts (HSE) dataset to support future work. Additionally, we show how context-enriched excerpts can provide more focused and comprehensive conversation summaries. 

**Abstract (ZH)**: 我们专注于增强小群体录音对话中的理解能力，这些对话作为桥梁，使人们聚集在一起，并提供了一个分享关于重要社会问题的个人故事和经历的空间。一种提取和传递这些对话信息的方式是，在后续对话中分享突出的重点片段。这种方式可以通过强调不同组别的人们的经验和视角，促进他们对相关问题的共同理解。由此产生的主要挑战是，从一个对话中提取的部分片段在另一个场合分享时，可能会缺乏原对话中已经介绍的关键上下文或核心元素。当对话变得更为冗长且具多主题和共通经历时，这个问题会变得更加突出。为了解决这个问题，我们探索大型语言模型（LLMs）如何通过提供社会相关的情境来丰富这些片段。我们介绍了有效情境化的方法，以提高理解能力、可读性和共情。通过主观和客观评估，我们展示了理解能力有了显著提升。尽管LLMs可以提供有价值的情境，但在捕捉关键的社会方面时存在困难。为此，我们发布了有人标注的重要片段（HSE）数据集，以支持未来的研究。此外，我们展示了富于情境的片段如何提供更聚焦和全面的对话总结。 

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
# HADES: Hardware Accelerated Decoding for Efficient Speculation in Large Language Models 

**Title (ZH)**: HADES：硬件加速解码以在大型语言模型中高效推测 

**Authors**: Ze Yang, Yihong Jin, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19925)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing by understanding and generating human-like text. However, the increasing demand for more sophisticated LLMs presents significant computational challenges due to their scale and complexity. This paper introduces Hardware Accelerated Decoding (HADES), a novel approach to enhance the performance and energy efficiency of LLMs. We address the design of an LLM accelerator with hardware-level speculative decoding support, a concept not previously explored in existing literature. Our work demonstrates how speculative decoding can significantly improve the efficiency of LLM operations, paving the way for more advanced and practical applications of these models. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过理解和生成类人的文本，已 revolutionized 自然语言处理。然而，对更为复杂的 LLMs 的需求增加，带来了由于其规模和复杂性而带来的重大计算挑战。本文介绍了一种名为hardware accelerated decoding (HADES) 的新型方法，以提高LLMs 的性能和能效。我们设计了一种带有硬件级推测性解码支持的LLM 加速器，这是现有文献中尚未探索的概念。我们的研究展示了推测性解码如何显著提高LLM 操作的效率，为这些模型的更高级和实用的应用铺平了道路。 

---
# Evaluate Summarization in Fine-Granularity: Auto Evaluation with LLM 

**Title (ZH)**: 在细粒度层面评估摘要质量：基于大语言模型的自动评估 

**Authors**: Dong Yuan, Eti Rastogi, Fen Zhao, Sagar Goyal, Gautam Naik, Sree Prasanna Rajagopal  

**Link**: [PDF](https://arxiv.org/pdf/2412.19906)  

**Abstract**: Due to the exponential growth of information and the need for efficient information consumption the task of summarization has gained paramount importance. Evaluating summarization accurately and objectively presents significant challenges, particularly when dealing with long and unstructured texts rich in content. Existing methods, such as ROUGE (Lin, 2004) and embedding similarities, often yield scores that have low correlation with human judgements and are also not intuitively understandable, making it difficult to gauge the true quality of the summaries. LLMs can mimic human in giving subjective reviews but subjective scores are hard to interpret and justify. They can be easily manipulated by altering the models and the tones of the prompts. In this paper, we introduce a novel evaluation methodology and tooling designed to address these challenges, providing a more comprehensive, accurate and interpretable assessment of summarization outputs. Our method (SumAutoEval) proposes and evaluates metrics at varying granularity levels, giving objective scores on 4 key dimensions such as completeness, correctness, Alignment and readability. We empirically demonstrate, that SumAutoEval enhances the understanding of output quality with better human correlation. 

**Abstract (ZH)**: 由于信息的指数级增长和高效信息消费的需要，摘要任务的重要性日益凸显。准确和客观地评估摘要任务面临显著挑战，尤其是在处理内容丰富且结构不一的长文本时。现有的方法，如ROUGE（Lin，2004）和基于相似性的嵌入方法，往往得出的分数与人工评估的相关性较低，也不容易直观理解，这使得很难判断摘要的真实质量。尽管大语言模型（LLMs）可以在主观评价方面模仿人类，但主观评分难以解释和论证，并且可以通过调整模型和提示的语气轻易被操控。在本文中，我们提出了一种新的评估方法和工具，以应对这些挑战，提供了一个更为全面、准确且可解释的摘要输出评估。我们提出的方法（SumAutoEval）在不同粒度级别中提出了并评估了多种指标，在四个关键维度（完整性、正确性、一致性与可读性）上给出了客观评分。我们通过实验证明，SumAutoEval 能够更好地与人工评估相关联，从而提升对输出质量的理解。 

---
# GaLore$+$: Boosting Low-Rank Adaptation for LLMs with Cross-Head Projection 

**Title (ZH)**: GaLore$+$: 基于跨头投影增强低秩适应性的大型语言模型 

**Authors**: Xutao Liao, Shaohui Li, Yuhui Xu, Zhi Li, Yu Liu, You He  

**Link**: [PDF](https://arxiv.org/pdf/2412.19820)  

**Abstract**: Recent low-rank training methods, such as GaLore, have significantly reduced the memory required to optimize large language models (LLMs). However, these methods often suffer from time-consuming low-rank projection estimations. In particular, the singular value decomposition (SVD) in GaLore can consume more than 80\% of the total training time. To address this issue, we propose GaLore$+$, which uses cross-head low-rank projection to reduce the substantial time consumption in estimating low-rank projections for multi-head attention. In addition, we employ randomized subspace iteration to achieve fast SVD. To further enhance performance, we propose sparsely coded residuals to reduce the errors caused by low-rank approximation on the first- and second-order moments of the optimizers and weight updates. We evaluate GaLore$+$ on arithmetic reasoning and natural language generation datasets. Our experiments demonstrate that GaLore$+$ delivers superior performance while achieving approximately $4\times$ fine-tuning speed compared to vanilla GaLore. 

**Abstract (ZH)**: 近年来，低秩训练方法如GaLore显著减少了优化大规模语言模型（LLMs）所需的内存。然而，这些方法往往会在低秩投影估计上消耗大量时间。特别是，在GaLore中使用的奇异值分解（SVD）可能会消耗总训练时间的超过80%。为了解决这一问题，我们提出了GaLore$+$，它使用交叉头低秩投影来减少多头注意力中低秩投影估计的时间消耗。此外，我们采用了随机子空间迭代来实现快速SVD。为了进一步提高性能，我们提出了稀疏编码残差，以减少因低秩近似导致的优化器和权重更新的一、二阶矩中的误差。我们在算术推理和自然语言生成数据集上评估了GaLore$+$。实验结果表明，与vanilla GaLore相比，GaLore$+$不仅性能优越，而且细调速度提高了约4倍。 

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
# Two-component spatiotemporal template for activation-inhibition of speech in ECoG 

**Title (ZH)**: 基于ECoG的语音激活与抑制的双成分时空模板模型 

**Authors**: Eric Easthope  

**Link**: [PDF](https://arxiv.org/pdf/2412.21178)  

**Abstract**: I compute the average trial-by-trial power of band-limited speech activity across epochs of multi-channel high-density electrocorticography (ECoG) recorded from multiple subjects during a consonant-vowel speaking task. I show that previously seen anti-correlations of average beta frequency activity (12-35 Hz) to high-frequency gamma activity (70-140 Hz) during speech movement are observable between individual ECoG channels in the sensorimotor cortex (SMC). With this I fit a variance-based model using principal component analysis to the band-powers of individual channels of session-averaged ECoG data in the SMC and project SMC channels onto their lower-dimensional principal components.
Spatiotemporal relationships between speech-related activity and principal components are identified by correlating the principal components of both frequency bands to individual ECoG channels over time using windowed correlation. Correlations of principal component areas to sensorimotor areas reveal a distinct two-component activation-inhibition-like representation for speech that resembles distinct local sensorimotor areas recently shown to have complex interplay in whole-body motor control, inhibition, and posture. Notably the third principal component shows insignificant correlations across all subjects, suggesting two components of ECoG are sufficient to represent SMC activity during speech movement. 

**Abstract (ZH)**: 我计算了多通道高密度脑皮层电图（ECoG）记录的多个被试在发音元音-辅音任务期间多个时段内，限定频段语音活动的平均单次试次功率。结果显示，在运动感知皮层（SMC）的个体ECoG通道之间，可以观察到先前看到的平均β频率活动（12-35 Hz）与高频率γ活动（70-140 Hz）之间的负相关关系。我使用主成分分析（PCA）拟合了一个基于方差的模型，并将会话平均ECoG数据中个体通道的频段功率投影到低维度的主成分上。

通过将不同时段内不同频段的主要成分与个体ECoG通道进行窗口相关分析，我们识别出了与发音活动和主要成分的空间-时间关系。主要成分区域与感觉运动区域的相关性揭示了一种类似于发音时局部感觉运动区域的激活-抑制图式的表示，这些区域近期被证明在全身运动控制、抑制和姿势维持中具有复杂的相互作用。值得注意的是，第三主要成分在所有被试之间没有显示出显著的相关性，这表明在发音期间SMC活动可以由两个ECoG成分充分表示。 

---
# Aviary: training language agents on challenging scientific tasks 

**Title (ZH)**: aviary: 在具有挑战性的科学任务上训练语言代理

这样翻译后，既保留了原文的核心含义，又符合学术论文的规范。如果需要进一步优化或具体的上下文背景，请提供更多的信息。 

**Authors**: Siddharth Narayanan, James D. Braza, Ryan-Rhys Griffiths, Manu Ponnapati, Albert Bou, Jon Laurent, Ori Kabeli, Geemi Wellawatte, Sam Cox, Samuel G. Rodriques, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2412.21154)  

**Abstract**: Solving complex real-world tasks requires cycles of actions and observations. This is particularly true in science, where tasks require many cycles of analysis, tool use, and experimentation. Language agents are promising for automating intellectual tasks in science because they can interact with tools via natural language or code. Yet their flexibility creates conceptual and practical challenges for software implementations, since agents may comprise non-standard components such as internal reasoning, planning, tool usage, as well as the inherent stochasticity of temperature-sampled language models. Here, we introduce Aviary, an extensible gymnasium for language agents. We formalize agents as policies solving language-grounded partially observable Markov decision processes, which we term language decision processes. We then implement five environments, including three challenging scientific environments: (1) manipulating DNA constructs for molecular cloning, (2) answering research questions by accessing scientific literature, and (3) engineering protein stability. These environments were selected for their focus on multi-step reasoning and their relevance to contemporary biology research. Finally, with online training and scaling inference-time compute, we show that language agents backed by open-source, non-frontier LLMs can match and exceed both frontier LLM agents and human experts on multiple tasks at up to 100x lower inference cost. 

**Abstract (ZH)**: 解决复杂的实际任务需要循环进行操作和观察。这一点在科学领域尤为明显，因为在科学中，任务往往需要多次分析、工具使用和实验循环。语言代理在自动执行科学中的智力任务方面具有巨大潜力，因为它们可以通过自然语言或代码与工具进行交互。然而，语言代理的灵活性给软件实现带来了概念性和实践性的挑战，因为代理可能包含非标准组件，如内部推理、规划、工具使用，以及高温采样语言模型固有的随机性。在此背景下，我们介绍了一种适用于语言代理的可扩展环境——Aviary。我们将代理定义为解决语言驱动部分可观测马尔可夫决策过程的策略，我们称之为语言决策过程。我们随后实现了五个环境，包括三个具有挑战性的科学环境：（1）用于分子克隆的DNA构建物操作；（2）通过访问科学文献来回答研究问题；（3）蛋白质稳定工程。这些环境的选择基于它们对多步推理的重视以及与现代生物学研究的相关性。最后，通过在线训练和扩大推理时间的计算能力，我们展示了使用开源非前沿的大规模语言模型（LLM）支持的语言代理能够在比最先进的LLM代理和人类专家低100倍的推理成本下完成多项任务。 

---
# Training Software Engineering Agents and Verifiers with SWE-Gym 

**Title (ZH)**: 使用 SWE-Gym 训练软件工程代理和验证器 

**Authors**: Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.21139)  

**Abstract**: We present SWE-Gym, the first environment for training real-world software engineering (SWE) agents. SWE-Gym contains 2,438 real-world Python task instances, each comprising a codebase with an executable runtime environment, unit tests, and a task specified in natural language. We use SWE-Gym to train language model based SWE agents , achieving up to 19% absolute gains in resolve rate on the popular SWE-Bench Verified and Lite test sets. We also experiment with inference-time scaling through verifiers trained on agent trajectories sampled from SWE-Gym. When combined with our fine-tuned SWE agents, we achieve 32.0% and 26.0% on SWE-Bench Verified and Lite, respectively, reflecting a new state-of-the-art for open-weight SWE agents. To facilitate further research, we publicly release SWE-Gym, models, and agent trajectories. 

**Abstract (ZH)**: 我们提出了SWE-Gym，这是首个用于训练现实世界软件工程（SWE）代理的环境。SWE-Gym包含2,438个实际存在的Python任务实例，每个实例都包含一个具有可执行运行时环境的代码库、单元测试以及用自然语言指定的任务。我们使用SWE-Gym来训练基于语言模型的SWE代理，在流行的SWE-Bench Verified和Lite测试集中，达到了高达19%的解决率绝对提升。我们还通过使用从SWE-Gym中采样的代理轨迹训练验证器，进行了推理时间的扩展实验。将这些验证器与我们微调的SWE代理结合使用后，在SWE-Bench Verified和Lite测试集中分别取得了32.0%和26.0%的成绩，这反映了开放式权重SWE代理的新最佳性能。为了促进进一步的研究，我们已将SWE-Gym、模型和代理轨迹公开发布。 

---
# TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization 

**Title (ZH)**: TangoFlux：匹配流和轻拍排名偏好优化下的超快速忠实文本转音频生成 

**Authors**: Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Rafael Valle, Bryan Catanzaro, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2412.21037)  

**Abstract**: We introduce TangoFlux, an efficient Text-to-Audio (TTA) generative model with 515M parameters, capable of generating up to 30 seconds of 44.1kHz audio in just 3.7 seconds on a single A40 GPU. A key challenge in aligning TTA models lies in the difficulty of creating preference pairs, as TTA lacks structured mechanisms like verifiable rewards or gold-standard answers available for Large Language Models (LLMs). To address this, we propose CLAP-Ranked Preference Optimization (CRPO), a novel framework that iteratively generates and optimizes preference data to enhance TTA alignment. We demonstrate that the audio preference dataset generated using CRPO outperforms existing alternatives. With this framework, TangoFlux achieves state-of-the-art performance across both objective and subjective benchmarks. We open source all code and models to support further research in TTA generation. 

**Abstract (ZH)**: 我们引入了TangoFlux，这是一种高效的文字到语音（Text-to-Audio, TTA）生成模型，参数量为515M，能够在单块A40 GPU上仅3.7秒内生成长达30秒的44.1kHz音频。TTA模型在对齐方面面临的主要挑战之一是难以创建偏好对，因为TTA缺乏类似大型语言模型（LLMs）中可验证奖励或金标准答案的结构化机制。为解决这一问题，我们提出了CLAP排序偏好优化（CLAP-Ranked Preference Optimization, CRPO）这一新颖框架，该框架通过迭代生成和优化偏好数据来提高TTA对齐的质量。我们证明，使用CRPO生成的音频偏好数据集在现有替代方案中表现更优。借助这一框架，TangoFlux在客观和主观基准测试中均实现了最先进的性能。我们已开源了所有代码和模型，以支持进一步的TTA生成研究。 

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
# Enhancing Multimodal Emotion Recognition through Multi-Granularity Cross-Modal Alignment 

**Title (ZH)**: 通过多粒度跨模态对齐增强多模态情感识别 

**Authors**: Xuechen Wang, Shiwan Zhao, Haoqin Sun, Hui Wang, Jiaming Zhou, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.20821)  

**Abstract**: Multimodal emotion recognition (MER), leveraging speech and text, has emerged as a pivotal domain within human-computer interaction, demanding sophisticated methods for effective multimodal integration. The challenge of aligning features across these modalities is significant, with most existing approaches adopting a singular alignment strategy. Such a narrow focus not only limits model performance but also fails to address the complexity and ambiguity inherent in emotional expressions. In response, this paper introduces a Multi-Granularity Cross-Modal Alignment (MGCMA) framework, distinguished by its comprehensive approach encompassing distribution-based, instance-based, and token-based alignment modules. This framework enables a multi-level perception of emotional information across modalities. Our experiments on IEMOCAP demonstrate that our proposed method outperforms current state-of-the-art techniques. 

**Abstract (ZH)**: 多模态情感识别（MER），利用语音和文本信息，在人机交互领域中已成为一个关键领域，要求采用复杂方法实现有效的多模态整合。这些模态之间对齐特征的难题非常显著，现有的大多数方法都采用了单一的对齐策略。这种狭隘的视角不仅限制了模型性能，而且没有解决情感表达固有的复杂性和模糊性。为应对这一挑战，本文提出了一种多粒度跨模态对齐（MGCMA）框架，该框架包括基于分布、基于实例和基于标记的对齐模块，以实现全面的多级情感信息感知。在IEMOCAP数据集上的实验表明，我们提出的方法在当前最先进的技术中表现更优。 

---
# HUNYUANPROVER: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving 

**Title (ZH)**: 浑元证明器：一种可扩展的数据合成框架与引导树搜索的自动定理证明方法 

**Authors**: Yang Li, Dong Du, Linfeng Song, Chen Li, Weikang Wang, Tao Yang, Haitao Mi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20735)  

**Abstract**: We introduce HunyuanProver, an language model finetuned from the Hunyuan 7B for interactive automatic theorem proving with LEAN4. To alleviate the data sparsity issue, we design a scalable framework to iterative synthesize data with low cost. Besides, guided tree search algorithms are designed to enable effective ``system 2 thinking`` of the prover. HunyuanProver achieves state-of-the-art (SOTA) performances on major benchmarks. Specifically, it achieves a pass of 68.4% on the miniF2F-test compared to 65.9%, the current SOTA results. It proves 4 IMO statements (imo_1960_p2, imo_1962_p2}, imo_1964_p2 and imo_1983_p6) in miniF2F-test. To benefit the community, we will open-source a dataset of 30k synthesized instances, where each instance contains the original question in natural language, the converted statement by autoformalization, and the proof by HunyuanProver. 

**Abstract (ZH)**: 我们介绍了HunyuanProver，这是一种从Hunyuan 7B微调而来的语言模型，用于与LEAN4进行交互式的自动定理证明。为了缓解数据稀疏性问题，我们设计了一个可扩展的框架，通过低成本迭代生成数据。此外，我们还设计了引导树搜索算法，以使证明器能够有效地实现“系统2思维”。HunyuanProver在主要基准测试中达到了最先进的（SOTA）性能。具体来说，它在miniF2F-test中的通过率达到了68.4%，而当前的SOTA结果为65.9%。它还在miniF2F-test中证明了4个IMO陈述（imo_1960_p2, imo_1962_p2, imo_1964_p2和imo_1983_p6）。为了惠及社区，我们将开放一个包含30,000个合成实例的数据集，每个实例包含原始问题的自然语言表述、自动形式化后的陈述，以及HunyuanProver的证明。 

---
# ChartAdapter: Large Vision-Language Model for Chart Summarization 

**Title (ZH)**: ChartAdapter：大型vision-language模型用于图表总结 

**Authors**: Peixin Xu, Yujuan Ding, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20715)  

**Abstract**: Chart summarization, which focuses on extracting key information from charts and interpreting it in natural language, is crucial for generating and delivering insights through effective and accessible data analysis. Traditional methods for chart understanding and summarization often rely on multi-stage pipelines, which may produce suboptimal semantic alignment between visual and textual information. In comparison, recently developed LLM-based methods are more dependent on the capability of foundation images or languages, while ignoring the characteristics of chart data and its relevant challenges. To address these limitations, we propose ChartAdapter, a novel lightweight transformer module designed to bridge the gap between charts and textual summaries. ChartAdapter employs learnable query vectors to extract implicit semantics from chart data and incorporates a cross-modal alignment projector to enhance vision-to-language generative learning. By integrating ChartAdapter with an LLM, we enable end-to-end training and efficient chart summarization. To further enhance the training, we introduce a three-stage hierarchical training procedure and develop a large-scale dataset specifically curated for chart summarization, comprising 190,618 samples. Experimental results on the standard Chart-to-Text testing set demonstrate that our approach significantly outperforms existing methods, including state-of-the-art models, in generating high-quality chart summaries. Ablation studies further validate the effectiveness of key components in ChartAdapter. This work highlights the potential of tailored LLM-based approaches to advance chart understanding and sets a strong foundation for future research in this area. 

**Abstract (ZH)**: 图表总结专注于从图表中提取关键信息并以自然语言进行解释，对于通过有效且易于访问的数据分析生成和传递洞察至关重要。传统的图表理解和总结方法通常依赖于多阶段管道，可能会导致图表信息与文本信息之间的语义对齐不理想。相比之下，近年来开发的基于大语言模型（LLM）的方法更加依赖于基础视觉或语言模型的能力，而忽视了图表数据的特性及其相关挑战。为了解决这些限制，我们提出了一种名为ChartAdapter的新颖轻量级transformer模块，旨在弥补图表与文本总结之间的差距。ChartAdapter利用可学习的查询向量从图表数据中提取潜在语义，并结合跨模态对齐投影器来增强视觉到语言生成学习。通过将ChartAdapter与大语言模型结合，我们实现了端到端的训练和高效的图表总结。为进一步增强训练，我们引入了三层级的分层训练程序，并开发了一个专门用于图表总结的大规模数据集，共包含190,618个样本。在标准的图表到文本测试集上的实验结果表明，我们的方法在生成高质量的图表总结方面显著优于现有方法，包括最先进的模型。进一步的消融研究验证了ChartAdapter中关键组件的有效性。这项工作突显了定制的大语言模型方法在推进图表理解方面的潜力，并为该领域的未来研究奠定了坚实基础。 

---
# UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design 

**Title (ZH)**: UBER：基于不确定性的大语言模型自动生成启发式设计方法 

**Authors**: Zijie Chen, Zhanchao Zhou, Yu Lu, Renjun Xu, Lili Pan, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20694)  

**Abstract**: NP-hard problem-solving traditionally relies on heuristics, but manually crafting effective heuristics for complex problems remains challenging. While recent work like FunSearch has demonstrated that large language models (LLMs) can be leveraged for heuristic design in evolutionary algorithm (EA) frameworks, their potential is not fully realized due to its deficiency in exploitation and exploration. We present UBER (Uncertainty-Based Evolution for Refinement), a method that enhances LLM+EA methods for automatic heuristic design by integrating uncertainty on top of the FunSearch framework. UBER introduces two key innovations: an Uncertainty-Inclusive Evolution Process (UIEP) for adaptive exploration-exploitation balance, and a principled Uncertainty-Inclusive Island Reset (UIIS) strategy for maintaining population diversity. Through extensive experiments on challenging NP-complete problems, UBER demonstrates significant improvements over FunSearch. Our work provides a new direction for the synergy of LLMs and EA, advancing the field of automatic heuristic design. 

**Abstract (ZH)**: 传统上，NP-hard问题求解依赖于启发式方法，但为复杂问题手动设计有效的启发式方法仍然具有挑战性。虽然最近的工作如FunSearch证明了大规模语言模型（LLMs）可以在进化算法（EA）框架中用于启发式设计，但由于其在利用和探索方面的不足，其潜力尚未完全实现。我们提出了UBER（基于不确定性进化的方法）——一种通过在FunSearch框架中引入不确定性来增强LLMs+EA方法进行自动启发式设计的方法。UBER引入了两项关键创新：一种不确定性包容的进化过程（UIEP），用于自适应地平衡探索与利用，以及一种有原则的不确定性包容的岛屿重置策略（UIIS），用于维持种群多样性。

通过在NP完全问题上的广泛实验，UBER在性能上显著超过了FunSearch。我们的工作为LLMs与EA的协同作用提供了一个新的方向，并推动了自动启发式设计领域的进展。 

---
# The Impact of Prompt Programming on Function-Level Code Generation 

**Title (ZH)**: 提示编程对函数级代码生成的影响 

**Authors**: Ranim Khojah, Francisco Gomes de Oliveira Neto, Mazen Mohamad, Philipp Leitner  

**Link**: [PDF](https://arxiv.org/pdf/2412.20545)  

**Abstract**: Large Language Models (LLMs) are increasingly used by software engineers for code generation. However, limitations of LLMs such as irrelevant or incorrect code have highlighted the need for prompt programming (or prompt engineering) where engineers apply specific prompt techniques (e.g., chain-of-thought or input-output examples) to improve the generated code. Despite this, the impact of different prompt techniques -- and their combinations -- on code generation remains underexplored. In this study, we introduce CodePromptEval, a dataset of 7072 prompts designed to evaluate five prompt techniques (few-shot, persona, chain-of-thought, function signature, list of packages) and their effect on the correctness, similarity, and quality of complete functions generated by three LLMs (GPT-4o, Llama3, and Mistral). Our findings show that while certain prompt techniques significantly influence the generated code, combining multiple techniques does not necessarily improve the outcome. Additionally, we observed a trade-off between correctness and quality when using prompt techniques. Our dataset and replication package enable future research on improving LLM-generated code and evaluating new prompt techniques. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被软件工程师用于代码生成。然而，LLMs 的局限性，如不相关或不正确的代码，强调了提示编程（或提示工程）的重要性，即工程师通过应用特定的提示技术（例如，思维链或输入输出示例）来提高生成代码的质量。尽管如此，不同提示技术的效果及其组合对代码生成的影响仍需进一步探索。在此研究中，我们引入了 CodePromptEval 数据集，该数据集包含 7072 个提示，旨在评估五种提示技术（少量示例、角色设定、思维链、函数签名、包列表）及其对由三种大型语言模型（GPT-4o、Llama3 和 Mistral 生成的完整函数的正确性、相似性和质量的影响）。我们的研究结果表明，虽然某些提示技术显着影响生成的代码，但组合使用多种技术并不一定会有更好的效果。此外，我们还观察到，在使用提示技术时，正确性和质量之间存在权衡。我们提供的数据集和复制包可以为未来改进大模型生成代码和评估新提示技术的研究提供支持。 

---
# ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding 

**Title (ZH)**: ReTaKe: 减少长时间视频理解中的时间冗余和知识冗余 

**Authors**: Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2412.20504)  

**Abstract**: Video Large Language Models (VideoLLMs) have achieved remarkable progress in video understanding. However, existing VideoLLMs often inherit the limitations of their backbone LLMs in handling long sequences, leading to challenges for long video understanding. Common solutions either simply uniformly sample videos' frames or compress visual tokens, which focus primarily on low-level temporal visual redundancy, overlooking high-level knowledge redundancy. This limits the achievable compression rate with minimal loss. To this end. we introduce a training-free method, $\textbf{ReTaKe}$, containing two novel modules DPSelect and PivotKV, to jointly model and reduce both temporal visual redundancy and knowledge redundancy for long video understanding. Specifically, DPSelect identifies keyframes with local maximum peak distance based on their visual features, which are closely aligned with human video perception. PivotKV employs the obtained keyframes as pivots and conducts KV-Cache compression for the non-pivot tokens with low attention scores, which are derived from the learned prior knowledge of LLMs. Experiments on benchmarks VideoMME, MLVU, and LVBench, show that ReTaKe can support 4x longer video sequences with minimal performance loss (<1%) and outperform all similar-size VideoLLMs with 3%-5%, even surpassing or on par with much larger ones. Our code is available at this https URL 

**Abstract (ZH)**: 视频大型语言模型（VideoLLMs）在视频理解方面已经取得了显著进展。然而，现有的VideoLLMs通常继承其骨干大型语言模型（LLMs）在处理长序列时的局限性，导致了长视频理解的挑战。当前的常见解决方案要么简单地均匀采样视频帧，要么压缩视觉令牌，这些方法主要关注低级别的时空视觉冗余，而忽视了高级知识冗余。这限制了在最小损失前提下的压缩率。为此，我们提出了一种无需训练的方法——**ReTaKe**，其中包含两个新颖模块DPSelect和PivotKV，以联合建模和减少长视频理解中的时空视觉冗余和知识冗余。具体而言，DPSelect通过其视觉特征识别具有局部最大峰值距离的关键帧，这些关键帧与人类对视频的感知紧密相关。PivotKV 使用这些关键帧作为参考，并针对具有低注意力分数的非关键帧进行KV-Cache压缩，这些分数来自LLMs学习到的先验知识。在VideoMME、MLVU和LVBench等基准测试上的实验表明，ReTaKe 可以支持4倍长的视频序列，并且在性能损失最小（<1%）的情况下优于所有相似规模的VideoLLMs，甚至超过了或在其范围内超过了更大规模的模型。我们的代码已开源，可从以下链接获取：\[填写链接\] 

---
# A Multidisciplinary Approach to Telegram Data Analysis 

**Title (ZH)**: 多学科方法下的电报数据解析 

**Authors**: Velizar Varbanov, Kalin Kopanov, Tatiana Atanasova  

**Link**: [PDF](https://arxiv.org/pdf/2412.20406)  

**Abstract**: This paper presents a multidisciplinary approach to analyzing data from Telegram for early warning information regarding cyber threats. With the proliferation of hacktivist groups utilizing Telegram to disseminate information regarding future cyberattacks or to boast about successful ones, the need for effective data analysis methods is paramount. The primary challenge lies in the vast number of channels and the overwhelming volume of data, necessitating advanced techniques for discerning pertinent risks amidst the noise. To address this challenge, we employ a combination of neural network architectures and traditional machine learning algorithms. These methods are utilized to classify and identify potential cyber threats within the Telegram data. Additionally, sentiment analysis and entity recognition techniques are incorporated to provide deeper insights into the nature and context of the communicated information. The study evaluates the effectiveness of each method in detecting and categorizing cyber threats, comparing their performance and identifying areas for improvement. By leveraging these diverse analytical tools, we aim to enhance early warning systems for cyber threats, enabling more proactive responses to potential security breaches. This research contributes to the ongoing efforts to bolster cybersecurity measures in an increasingly interconnected digital landscape. 

**Abstract (ZH)**: 本文采用多学科方法对来自 Telegram 的数据进行分析，以提取有关网络安全威胁的早期预警信息。随着黑客激进组织利用 Telegram 分发有关未来网络攻击的信息，或是炫耀成功攻击的情况日益增多，有效的数据分析方法显得尤为重要。主要挑战在于海量的频道和数据量庞大，需要采用先进的技术手段来从噪声中甄别出关键的威胁信号。为此，我们运用了神经网络架构与传统机器学习算法的组合方法。这些方法被用来对 Telegram 数据中的潜在网络威胁进行分类和识别。此外，我们还结合使用了情感分析和实体识别技术，以提供对沟通信息性质和背景更深层次的了解。研究评估了每种方法在检测和分类网络威胁方面的有效性，并比较了它们的性能，以识别改进的领域。通过利用这些多样化的分析工具，我们旨在提升网络安全威胁的早期预警系统，使网络安全响应更加积极主动。这项研究为在日益互联的数字环境中加强网络安全措施作出了贡献。 

---
# Enhancing Code LLMs with Reinforcement Learning in Code Generation 

**Title (ZH)**: 使用强化学习提升代码生成的代码LLM性能 

**Authors**: Junqiao Wang, Zeng Zhang, Yangfan He, Yuyang Song, Tianyu Shi, Yuchen Li, Hengyuan Xu, Kunyu Wu, Guangwu Qian, Qiuwu Chen, Lewei He  

**Link**: [PDF](https://arxiv.org/pdf/2412.20367)  

**Abstract**: With the rapid evolution of large language models (LLM), reinforcement learning (RL) has emerged as a pivotal technique for code generation and optimization in various domains. This paper presents a systematic survey of the application of RL in code optimization and generation, highlighting its role in enhancing compiler optimization, resource allocation, and the development of frameworks and tools. Subsequent sections first delve into the intricate processes of compiler optimization, where RL algorithms are leveraged to improve efficiency and resource utilization. The discussion then progresses to the function of RL in resource allocation, emphasizing register allocation and system optimization. We also explore the burgeoning role of frameworks and tools in code generation, examining how RL can be integrated to bolster their capabilities. This survey aims to serve as a comprehensive resource for researchers and practitioners interested in harnessing the power of RL to advance code generation and optimization techniques. 

**Abstract (ZH)**: 随着大型语言模型（LLM）的迅速发展，强化学习（RL）已成为代码生成和优化各领域的重要技术。本文对RL在代码优化和生成中的应用进行了系统性的综述，强调了它在增强编译器优化、资源分配以及框架和工具开发方面的作用。随后的部分首先详细探讨了编译器优化的复杂过程，介绍了RL算法如何提升效率和资源利用。接着讨论了RL在资源分配中的作用，重点讨论了寄存器分配和系统优化。我们还探讨了框架和工具在代码生成中的新兴作用，考察了如何将RL集成以增强其功能。本文旨在为希望利用RL推进代码生成和优化技术的研究人员和实践者提供全面的资源。 

---
# On the Compositional Generalization of Multimodal LLMs for Medical Imaging 

**Title (ZH)**: 多模态大语言模型在医学影像中的组合泛化研究 

**Authors**: Zhenyang Cai, Junying Chen, Rongsheng Wang, Weihong Wang, Yonglin Deng, Dingjie Song, Yize Chen, Zixu Zhang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20070)  

**Abstract**: Multimodal large language models (MLLMs) hold significant potential in the medical field, but their capabilities are often limited by insufficient data in certain medical domains, highlighting the need for understanding what kinds of images can be used by MLLMs for generalization. Current research suggests that multi-task training outperforms single-task as different tasks can benefit each other, but they often overlook the internal relationships within these tasks, providing limited guidance on selecting datasets to enhance specific tasks. To analyze this phenomenon, we attempted to employ compositional generalization (CG)-the ability of models to understand novel combinations by recombining learned elements-as a guiding framework. Since medical images can be precisely defined by Modality, Anatomical area, and Task, naturally providing an environment for exploring CG. Therefore, we assembled 106 medical datasets to create Med-MAT for comprehensive experiments. The experiments confirmed that MLLMs can use CG to understand unseen medical images and identified CG as one of the main drivers of the generalization observed in multi-task training. Additionally, further studies demonstrated that CG effectively supports datasets with limited data and delivers consistent performance across different backbones, highlighting its versatility and broad applicability. Med-MAT is publicly available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在医疗领域具有巨大的潜力，但它们的能力往往受到某些医学领域数据不足的限制，突显了理解哪些类型的图像可以用于MLLMs进行泛化的必要性。当前研究表明，多任务训练优于单任务训练，因为不同任务之间可以相互受益，但它们往往忽略了这些任务之间的内部关系，为提高特定任务的指导数据选择提供了有限的帮助。为了分析这一现象，我们尝试将组合泛化（CG）——模型通过重组学习元素来理解新组合的能力——作为指导框架。由于医学图像可以通过模态、解剖区域和任务来精确定义，自然为探索CG提供了良好的环境。因此，我们收集了106个医学数据集以创建Med-MAT，用于进行全面实验。实验结果证实了MLLMs可以通过CG理解未见过的医学图像，并识别CG为多任务训练中观察到的泛化的主要驱动力之一。此外，进一步的研究表明，CG有效地支持了数据有限的数据集，并在不同的骨干网络上提供了稳定的表现，突显了其多样性和广泛的适用性。Med-MAT已在以下网址公开：[此 https URL]。 

---
# The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Support 

**Title (ZH)**: 大型语言模型的情感谱系：利用同理心和基于情感的指标为心理健康提供支持 

**Authors**: Alessandro De Grandi, Federico Ravenda, Andrea Raballo, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2412.20068)  

**Abstract**: The increasing demand for mental health services has highlighted the need for innovative solutions, particularly in the realm of psychological conversational AI, where the availability of sensitive data is scarce. In this work, we explored the development of a system tailored for mental health support with a novel approach to psychological assessment based on explainable emotional profiles in combination with empathetic conversational models, offering a promising tool for augmenting traditional care, particularly where immediate expertise is unavailable. Our work can be divided into two main parts, intrinsecaly connected to each other. First, we present RACLETTE, a conversational system that demonstrates superior emotional accuracy compared to state-of-the-art benchmarks in both understanding users' emotional states and generating empathetic responses during conversations, while progressively building an emotional profile of the user through their interactions. Second, we show how the emotional profiles of a user can be used as interpretable markers for mental health assessment. These profiles can be compared with characteristic emotional patterns associated with different mental disorders, providing a novel approach to preliminary screening and support. 

**Abstract (ZH)**: 随着对心理健康服务需求的增加，创新解决方案的必要性日益突出，特别是在心理对话AI领域，可用的敏感数据相对稀缺。本研究探讨了一种针对心理健康支持的新型系统开发，该系统基于可解释情感画像与同理心对话模型相结合的心理评估方法，为传统护理提供了有前途的补充工具，特别是在即时专业知识不可用时。我们的工作可以分为两个主要部分，这两部分密不可分。首先，我们介绍了RACLETTE，这是一种对话系统，在理解和感知用户情感状态以及在对话中生成同理心回应方面优于现有基准，同时通过用户互动逐步构建其情感画像。其次，我们展示了用户的情感画像如何作为心理健康评估的可解释标志。这些画像可以与其所关联的不同心理健康障碍的情感模式进行比较，提供了一种新的初步筛查和支持方法。 

---
# BaiJia: A Large Scale Role-Playing Agent Corpus of Chinese Historical Charcaters 

**Title (ZH)**: baiJia：中国古代历史人物大规模角色扮演代理语料库 

**Authors**: Ting Bai, Jiazheng Kang, Jiayang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20024)  

**Abstract**: We introduce a comprehensive large-scale role-playing agent corpus, termed BaiJia, that comprises various Chinese historical characters. This corpus is noteworthy for being the pioneering compilation of low-resource data that can be utilized in large language models (LLMs) to engage in AI-driven historical role-playing agents. BaiJia addresses the challenges in terms of fragmented historical textual records in different forms and modalities, integrating various characters' information, including their biographical, literary, family relations, historical events, and so on. We conduct extensive experiments to demonstrate the effectiveness of our BaiJia agent corpus in bolstering the role-playing abilities of various foundational LLMs, and promoting the development and assessment of LLMs in the context of historical role-playing tasks. The agent corpus is available at this http URL. 

**Abstract (ZH)**: 我们介绍了一个全面的大规模角色扮演代理语料库，称为BaiJia，该语料库包含各种中国历史人物。该语料库因其是利用低资源数据构建大语言模型（LLMs）进行AI驱动的历史角色扮演代理的先驱汇编而值得注意。BaiJia解决了不同形式和模态的破碎历史文獻资料所带来的挑战，整合了各种人物的信息，包括他们的生平、文学作品、家庭关系、历史事件等。我们进行了大量的实验，以证明BaiJia代理语料库在增强各类基础LLMs的角色扮演能力方面的作用，并促进和评估LLMs在历史角色扮演任务中的发展。该代理语料库可在以下网址获取：[插入网址]。 

---
# From Generalist to Specialist: A Survey of Large Language Models for Chemistry 

**Title (ZH)**: 从通才到专才：大型语言模型在化学领域中的综述 

**Authors**: Yang Han, Ziping Wan, Lu Chen, Kai Yu, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.19994)  

**Abstract**: Large Language Models (LLMs) have significantly transformed our daily life and established a new paradigm in natural language processing (NLP). However, the predominant pretraining of LLMs on extensive web-based texts remains insufficient for advanced scientific discovery, particularly in chemistry. The scarcity of specialized chemistry data, coupled with the complexity of multi-modal data such as 2D graph, 3D structure and spectrum, present distinct challenges. Although several studies have reviewed Pretrained Language Models (PLMs) in chemistry, there is a conspicuous absence of a systematic survey specifically focused on chemistry-oriented LLMs. In this paper, we outline methodologies for incorporating domain-specific chemistry knowledge and multi-modal information into LLMs, we also conceptualize chemistry LLMs as agents using chemistry tools and investigate their potential to accelerate scientific research. Additionally, we conclude the existing benchmarks to evaluate chemistry ability of LLMs. Finally, we critically examine the current challenges and identify promising directions for future research. Through this comprehensive survey, we aim to assist researchers in staying at the forefront of developments in chemistry LLMs and to inspire innovative applications in the field. 

**Abstract (ZH)**: 大规模语言模型（LLMs）显著地改变了我们的日常生活，并在自然语言处理（NLP）中建立了新的范式。然而，LLMs主要通过大规模网络文本进行预训练，在高级科学研究，尤其是化学领域方面仍然存在不足。化学领域的专用数据稀缺，以及2D图、3D结构和光谱等多模态数据的复杂性，构成了独特的挑战。尽管已有研究回顾了化学中的预训练语言模型（PLMs），但对于专注于化学领域的LLMs的系统性综述却少之又少。在本文中，我们概述了将领域特定的化学知识和多模态信息融入LLMs的方法，并将化学LLMs概念化为使用化学工具的代理，探讨其加速科学研究的潜力。此外，我们总结了评估化学LLMs能力的现有基准，并对当前挑战进行批判性审视，指出了未来研究的潜在方向。通过这一全面的综述，我们旨在帮助研究人员保持在化学LLMs发展的最前沿，并激发该领域的创新应用。 

---
