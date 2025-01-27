# Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation 

**Title (ZH)**: 层次化时间感知专家混合模型在多模态序列推荐中的应用 

**Authors**: Shengzhe Zhang, Liyi Chen, Dazhong Shen, Chao Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.14269)  

**Abstract**: Multi-modal sequential recommendation (SR) leverages multi-modal data to learn more comprehensive item features and user preferences than traditional SR methods, which has become a critical topic in both academia and industry. Existing methods typically focus on enhancing multi-modal information utility through adaptive modality fusion to capture the evolving of user preference from user-item interaction sequences. However, most of them overlook the interference caused by redundant interest-irrelevant information contained in rich multi-modal data. Additionally, they primarily rely on implicit temporal information based solely on chronological ordering, neglecting explicit temporal signals that could more effectively represent dynamic user interest over time. To address these limitations, we propose a Hierarchical time-aware Mixture of experts for multi-modal Sequential Recommendation (HM4SR) with a two-level Mixture of Experts (MoE) and a multi-task learning strategy. Specifically, the first MoE, named Interactive MoE, extracts essential user interest-related information from the multi-modal data of each item. Then, the second MoE, termed Temporal MoE, captures user dynamic interests by introducing explicit temporal embeddings from timestamps in modality encoding. To further address data sparsity, we propose three auxiliary supervision tasks: sequence-level category prediction (CP) for item feature understanding, contrastive learning on ID (IDCL) to align sequence context with user interests, and placeholder contrastive learning (PCL) to integrate temporal information with modalities for dynamic interest modeling. Extensive experiments on four public datasets verify the effectiveness of HM4SR compared to several state-of-the-art approaches. 

**Abstract (ZH)**: 多模态序列推荐（SR）利用多模态数据来学习比传统SR方法更为全面的项目特征和用户偏好，已经成为学术界和工业界的热点话题。现有方法通常侧重于通过自适应模态融合来增强多模态信息的实用性，以捕捉用户偏好在用户-项目交互序列中的演变。然而，大多数方法忽略了丰富多模态数据中包含的冗余兴趣无关信息所造成的影响。此外，它们主要依赖于基于时间顺序的隐式时间信息，忽视了可以更有效地代表用户随时间动态兴趣的显式时间信号。为解决这些局限性，我们提出了一种基于两层混合专家（MoE）和多任务学习策略的层级时间感知混合专家多模态序列推荐（HM4SR）。具体来说，第一层MoE，称为交互MoE，从每个项目的多模态数据中提取关键的用户兴趣相关信息。然后，第二层MoE，称为时间MoE，通过引入时间戳在模态编码中的显式时间嵌入来捕获用户动态兴趣。为了进一步解决数据稀疏性问题，我们提出了三项辅助监督任务：序列级类别预测（CP）以理解项目特征，基于ID的对比学习（IDCL）以使序列上下文与用户兴趣对齐，以及占位符对比学习（PCL）以将时间信息与模态结合，从而建模动态兴趣。在四个公开数据集上的广泛实验表明，与几种最先进的方法相比，HM4SR的有效性得到了验证。 

---
# Fanar: An Arabic-Centric Multimodal Generative AI Platform 

**Title (ZH)**: Fanar：一个以阿拉伯语为中心的多模态生成人工智能平台 

**Authors**: Fanar Team, Ummar Abbas, Mohammad Shahmeer Ahmad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari, Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Masoomali Fatehkia, Anastasios Fragkopoulos, Maram Hasanain, Majd Hawasly, Mus'ab Husaini, Soon-Gyo Jung, Ji Kim Lucas, Walid Magdy, Safa Messaoud, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Zan Naeem, Mourad Ouzzani, Dorde Popovic, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang, Ahmed Ali, Yassine El Kheir, Xiaosong Ma, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13944)  

**Abstract**: We present Fanar, a platform for Arabic-centric multimodal generative AI systems, that supports language, speech and image generation tasks. At the heart of Fanar are Fanar Star and Fanar Prime, two highly capable Arabic Large Language Models (LLMs) that are best in the class on well established benchmarks for similar sized models. Fanar Star is a 7B (billion) parameter model that was trained from scratch on nearly 1 trillion clean and deduplicated Arabic, English and Code tokens. Fanar Prime is a 9B parameter model continually trained on the Gemma-2 9B base model on the same 1 trillion token set. Both models are concurrently deployed and designed to address different types of prompts transparently routed through a custom-built orchestrator. The Fanar platform provides many other capabilities including a customized Islamic Retrieval Augmented Generation (RAG) system for handling religious prompts, a Recency RAG for summarizing information about current or recent events that have occurred after the pre-training data cut-off date. The platform provides additional cognitive capabilities including in-house bilingual speech recognition that supports multiple Arabic dialects, voice and image generation that is fine-tuned to better reflect regional characteristics. Finally, Fanar provides an attribution service that can be used to verify the authenticity of fact based generated content.
The design, development, and implementation of Fanar was entirely undertaken at Hamad Bin Khalifa University's Qatar Computing Research Institute (QCRI) and was sponsored by Qatar's Ministry of Communications and Information Technology to enable sovereign AI technology development. 

**Abstract (ZH)**: 我们介绍了Fanar平台，这是一个针对阿拉伯语的多模态生成AI系统平台，支持语言、语音和图像生成任务。Fanar的核心是Fanar Star和Fanar Prime，这是两个性能优秀的阿拉伯语大型语言模型（LLMs），在同类基准测试中表现出色。Fanar Star是一个70亿参数的模型，从几乎1万亿个清洗和去重后的阿拉伯语、英语和代码标记中从头开始训练。Fanar Prime是一个90亿参数的模型，持续训练在Gemma-2 9B基础模型上，使用同样1万亿标记集。这两个模型同时部署，并设计成能够通过自定义构建的协调器透明地处理不同类型的提示。Fanar平台还提供了其他许多功能，包括自定义的伊斯兰检索增强生成（RAG）系统以处理宗教提示，以及近期RAG以总结发生在预训练数据截止日期之后的当前或最近事件的信息。该平台还提供其他认知功能，包括支持多种阿拉伯方言的内部双语语音识别，以及更好地反映区域特征的语音和图像生成。最后，Fanar提供了验证基于事实生成内容真实性的一种归属服务。

Fanar平台的设计、开发和实现完全由卡塔尔计算研究学会（Qatar Computing Research Institute，QCRI）的哈马德·本·哈利法大学（Hamad Bin Khalifa University）自主完成，并由卡塔尔通讯与信息技术部赞助，以促进主权AI技术的发展。 

---
# Mitigating GenAI-powered Evidence Pollution for Out-of-Context Multimodal Misinformation Detection 

**Title (ZH)**: 利用基于生成式人工智能的方法减轻脱离上下文的跨模态虚假信息检测中的证据污染问题 

**Authors**: Zehong Yan, Peng Qi, Wynne Hsu, Mong Li Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.14728)  

**Abstract**: While large generative artificial intelligence (GenAI) models have achieved significant success, they also raise growing concerns about online information security due to their potential misuse for generating deceptive content. Out-of-context (OOC) multimodal misinformation detection, which often retrieves Web evidence to identify the repurposing of images in false contexts, faces the issue of reasoning over GenAI-polluted evidence to derive accurate predictions. Existing works simulate GenAI-powered pollution at the claim level with stylistic rewriting to conceal linguistic cues, and ignore evidence-level pollution for such information-seeking applications. In this work, we investigate how polluted evidence affects the performance of existing OOC detectors, revealing a performance degradation of more than 9 percentage points. We propose two strategies, cross-modal evidence reranking and cross-modal claim-evidence reasoning, to address the challenges posed by polluted evidence. Extensive experiments on two benchmark datasets show that these strategies can effectively enhance the robustness of existing out-of-context detectors amidst polluted evidence. 

**Abstract (ZH)**: 尽管大型生成人工智能（GenAI）模型取得了显著的成果，但它们也因潜在的误导性内容生成而引发了日益增长的在线信息安全 concerns。脱嵌（Out-of-Context, OOC）多模态虚假信息检测通常通过检索网络证据来识别错误上下文中的图像用途，但在处理 GenAI 污染的证据以获得准确预测时面临挑战。现有研究在声明层面通过风格性重写模拟 GenAI 动力污染，以掩盖语言线索，但在此类信息检索应用中忽略了证据层面的污染。在此项工作中，我们探讨了污染证据对现有 OOC 检测器性能的影响，发现性能下降超过 9 个百分点。我们提出了两种策略——跨模态证据重排和跨模态声明-证据推理——以应对污染证据带来的挑战。在两个基准数据集上的广泛实验表明，这些策略能够有效提升在污染证据环境下现有 OOC 检测器的鲁棒性。 

---
# Distributed Multi-Agent Coordination Using Multi-Modal Foundation Models 

**Title (ZH)**: 使用多模态基础模型进行分布式多代理协调 

**Authors**: Saaduddin Mahmud, Dorian Benhamou Goldfajn, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2501.14189)  

**Abstract**: Distributed Constraint Optimization Problems (DCOPs) offer a powerful framework for multi-agent coordination but often rely on labor-intensive, manual problem construction. To address this, we introduce VL-DCOPs, a framework that takes advantage of large multimodal foundation models (LFMs) to automatically generate constraints from both visual and linguistic instructions. We then introduce a spectrum of agent archetypes for solving VL-DCOPs: from a neuro-symbolic agent that delegates some of the algorithmic decisions to an LFM, to a fully neural agent that depends entirely on an LFM for coordination. We evaluate these agent archetypes using state-of-the-art LLMs (large language models) and VLMs (vision language models) on three novel VL-DCOP tasks and compare their respective advantages and drawbacks. Lastly, we discuss how this work extends to broader frontier challenges in the DCOP literature. 

**Abstract (ZH)**: 分布式约束优化问题（DCOPs）提供了一种强大的多代理协调框架，但通常依赖于耗时的手工问题构建。为解决这一问题，我们引入了VL-DCOPs框架，利用大规模多模态基础模型（LFMs）从视觉和语言指令中自动生成约束。接着，我们提出了解决VL-DCOPs的一系列代理原型：从一种神经符号代理，它将部分算法决策委托给LFM，到一种完全依赖LFM进行协调的全神经代理。我们使用最先进的大型语言模型（LLMs）和视觉语言模型（VLMs）来评估这些代理原型，并比较各自的优缺点。最后，我们讨论了这项工作如何扩展到DCOP文献中的更广泛的前沿挑战。 

---
# Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research 

**Title (ZH)**: 利用ChatGPT的多模态视觉能力按贫困程度对卫星图像进行排序：推进社会科学领域的研究工具 

**Authors**: Hamid Sarmadi, Ola Hall, Thorsteinn Rögnvaldsson, Mattias Ohlsson  

**Link**: [PDF](https://arxiv.org/pdf/2501.14546)  

**Abstract**: This paper investigates the novel application of Large Language Models (LLMs) with vision capabilities to analyze satellite imagery for village-level poverty prediction. Although LLMs were originally designed for natural language understanding, their adaptability to multimodal tasks, including geospatial analysis, has opened new frontiers in data-driven research. By leveraging advancements in vision-enabled LLMs, we assess their ability to provide interpretable, scalable, and reliable insights into human poverty from satellite images. Using a pairwise comparison approach, we demonstrate that ChatGPT can rank satellite images based on poverty levels with accuracy comparable to domain experts. These findings highlight both the promise and the limitations of LLMs in socioeconomic research, providing a foundation for their integration into poverty assessment workflows. This study contributes to the ongoing exploration of unconventional data sources for welfare analysis and opens pathways for cost-effective, large-scale poverty monitoring. 

**Abstract (ZH)**: 本文探究了大型语言模型（LLMs）结合视觉能力在分析卫星影像以预测村庄级贫困方面的新型应用。尽管LLMs最初设计用于自然语言理解，但它们在多模态任务中的适应性，包括空间地理分析，已经为基于数据的研究开辟了新的前沿。通过利用视觉增强的LLMs的最新进展，我们评估了它们在提供可解释、可扩展和可靠的人类贫困见解方面的能力，这些见解来自于卫星影像。采用成对比较的方法，我们展示了ChatGPT可以根据贫困程度对卫星影像进行排名，其准确度与领域专家相当。这些发现突显了LLMs在社会经济研究中的潜力和局限性，为其集成到贫困评估工作流程中提供了基础。本研究对福利分析中非传统数据源的探索作出了贡献，并为低成本、大规模的贫困监测开辟了途径。 

---
# Global Semantic-Guided Sub-image Feature Weight Allocation in High-Resolution Large Vision-Language Models 

**Title (ZH)**: 全球语义引导的子图像特征权重分配在高分辨率大型视觉-语言模型中 

**Authors**: Yuxuan Liang, Xu Li, Xiaolei Chen, Haotian Chen, Yi Zheng, Chenghang Lai, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2501.14276)  

**Abstract**: As the demand for high-resolution image processing in Large Vision-Language Models (LVLMs) grows, sub-image partitioning has become a popular approach for mitigating visual information loss associated with fixed-resolution processing. However, existing partitioning methods uniformly process sub-images, resulting in suboptimal image understanding. In this work, we reveal that the sub-images with higher semantic relevance to the entire image encapsulate richer visual information for preserving the model's visual understanding ability. Therefore, we propose the Global Semantic-guided Weight Allocator (GSWA) module, which dynamically allocates weights to sub-images based on their relative information density, emulating human visual attention mechanisms. This approach enables the model to focus on more informative regions, overcoming the limitations of uniform treatment. We integrate GSWA into the InternVL2-2B framework to create SleighVL, a lightweight yet high-performing model. Extensive experiments demonstrate that SleighVL outperforms models with comparable parameters and remains competitive with larger models. Our work provides a promising direction for more efficient and contextually aware high-resolution image processing in LVLMs, advancing multimodal system development. 

**Abstract (ZH)**: 随着对高分辨率图像处理的需求在大型视觉-语言模型（LVLMs）中不断增加，子图像分割已成为缓解固定分辨率处理过程中视觉信息损失的一种流行方法。然而，现有的分割方法均勻处理子图像，导致对图像的理解效果不佳。在本文中，我们揭示了与整个图像具有更高语义相关性的子图像更能保留模型的视觉理解能力，蕴含了更丰富的视觉信息。因此，我们提出了全局语义指导加权分配器（GSWA）模块，该模块根据子图像的相对信息密度动态分配权重，模拟人类视觉注意力机制。这种方法使模型能够聚焦于更具信息性的区域，克服了均勻处理的局限性。我们将GSWA集成到InternVL2-2B框架中，创建了SleighVL，这是一个轻量级但性能优异的模型。大量的实验表明，SleighVL 在参数相近的模型中表现出色，并且与大型模型相比仍具有竞争力。我们的工作为LVLMs 更高效且具有上下文感知能力的高分辨率图像处理提供了有前景的方向，推进了多模态系统的发展。 

---
# TFG-Flow: Training-free Guidance in Multimodal Generative Flow 

**Title (ZH)**: TFG-Flow：无需训练的多模态生成流中的指导方法 

**Authors**: Haowei Lin, Shanda Li, Haotian Ye, Yiming Yang, Stefano Ermon, Yitao Liang, Jianzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.14216)  

**Abstract**: Given an unconditional generative model and a predictor for a target property (e.g., a classifier), the goal of training-free guidance is to generate samples with desirable target properties without additional training. As a highly efficient technique for steering generative models toward flexible outcomes, training-free guidance has gained increasing attention in diffusion models. However, existing methods only handle data in continuous spaces, while many scientific applications involve both continuous and discrete data (referred to as multimodality). Another emerging trend is the growing use of the simple and general flow matching framework in building generative foundation models, where guided generation remains under-explored. To address this, we introduce TFG-Flow, a novel training-free guidance method for multimodal generative flow. TFG-Flow addresses the curse-of-dimensionality while maintaining the property of unbiased sampling in guiding discrete variables. We validate TFG-Flow on four molecular design tasks and show that TFG-Flow has great potential in drug design by generating molecules with desired properties. 

**Abstract (ZH)**: 以下内容为论文的标题或摘要，并已翻译成中文，符合学术规范：

在给定无条件生成模型和目标属性的预测器（例如，分类器）的情况下，训练免费指导的目标是在不进行额外训练的情况下生成具有期望目标属性的样本。作为一种高效的技术，用于引导生成模型产生灵活的结果，训练免费指导在扩散模型中引起了越来越多的关注。然而，现有的方法仅处理连续空间中的数据，而许多科学应用涉及连续和离散数据（统称为多模态）。另一个新兴趋势是在构建生成基础模型时越来越多地使用简单且通用的流匹配框架，而引导生成仍是一个未充分探索的领域。为了解决这一问题，我们提出了TFG-Flow，这是一种用于多模态生成流的新型训练免费指导方法。TFG-Flow 在引导离散变量的同时解决了高维灾难的问题，并保持了无偏采样的性质。我们在四项分子设计任务上验证了TFG-Flow，并展示了TFG-Flow 在通过生成具有期望属性的分子来进行药物设计方面的巨大潜力。 

---
# Dynamic Token Reduction during Generation for Vision Language Models 

**Title (ZH)**: 生成过程中用于视觉语言模型的动态令牌减少 

**Authors**: Xiaoyu Liang, Chaofeng Guan, Jiaying Lu, Huiyao Chen, Huan Wang, Haoji Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14204)  

**Abstract**: Vision-Language Models (VLMs) have achieved notable success in multimodal tasks but face practical limitations due to the quadratic complexity of decoder attention mechanisms and autoregressive generation. Existing methods like FASTV and VTW have achieved notable results in reducing redundant visual tokens, but these approaches focus on pruning tokens in a single forward pass without systematically analyzing the redundancy of visual tokens throughout the entire generation process. In this paper, we introduce a dynamic pruning strategy tailored for VLMs, namedDynamic Rate (DyRate), which progressively adjusts the compression rate during generation. Our analysis of the distribution of attention reveals that the importance of visual tokens decreases throughout the generation process, inspiring us to adopt a more aggressive compression rate. By integrating a lightweight predictor based on attention distribution, our approach enables flexible adjustment of pruning rates based on the attention distribution. Our experimental results demonstrate that our method not only reduces computational demands but also maintains the quality of responses. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在多模态任务中取得了显著的成功，但由于解码器注意机制和自回归生成的二次复杂性，它们面临实际应用中的限制。现有的方法如FASTV和VTW在减少冗余视觉标记方面取得了显著成果，但这些方法主要关注在一前向传递中剪枝标记，而不系统地分析整个生成过程中视觉标记的冗余性。在本文中，我们提出了一种专门针对VLMs的动态剪枝策略，命名为DyRate，该策略在生成过程中逐步调整压缩率。通过对注意分布的分析，我们发现生成过程中视觉标记的重要性逐渐降低，这启发我们采取更激进的压缩率。通过结合基于注意分布的轻量级预测器，我们的方法能够根据注意分布灵活调整剪枝率。我们的实验结果表明，我们的方法不仅减少了计算需求，还保持了响应的质量。 

---
# Enhancing Multimodal Entity Linking with Jaccard Distance-based Conditional Contrastive Learning and Contextual Visual Augmentation 

**Title (ZH)**: 基于Jaccard距离条件对比学习和上下文视觉增强的多模态实体链接优化 

**Authors**: Cong-Duy Nguyen, Xiaobao Wu, Thong Nguyen, Shuai Zhao, Khoi Le, Viet-Anh Nguyen, Feng Yichao, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14166)  

**Abstract**: Previous research on multimodal entity linking (MEL) has primarily employed contrastive learning as the primary objective. However, using the rest of the batch as negative samples without careful consideration, these studies risk leveraging easy features and potentially overlook essential details that make entities unique. In this work, we propose JD-CCL (Jaccard Distance-based Conditional Contrastive Learning), a novel approach designed to enhance the ability to match multimodal entity linking models. JD-CCL leverages meta-information to select negative samples with similar attributes, making the linking task more challenging and robust. Additionally, to address the limitations caused by the variations within the visual modality among mentions and entities, we introduce a novel method, CVaCPT (Contextual Visual-aid Controllable Patch Transform). It enhances visual representations by incorporating multi-view synthetic images and contextual textual representations to scale and shift patch representations. Experimental results on benchmark MEL datasets demonstrate the strong effectiveness of our approach. 

**Abstract (ZH)**: 先前关于多模态实体链接（MEL）的研究主要将对比学习作为主要目标。然而，这些研究在使用整个批次的样本作为负样本时，如果没有仔细考虑，可能会利用简单的特征，并且有可能忽略使得实体独特的关键细节。在这项工作中，我们提出了一种名为JD-CCL（Jaccard 距离基于条件对比学习）的新型方法，旨在增强多模态实体链接模型的匹配能力。JD-CCL 利用元信息来选择具有相似属性的负样本，从而使链接任务更具挑战性和鲁棒性。此外，为了应对提及和实体在视觉模态内变化造成的限制，我们引入了一种新颖的方法，即CVaCPT（基于上下文的视觉辅助可控补丁变换）。该方法通过结合多视角合成图像和上下文文本表示来扩展和变换补丁表示，从而增强视觉表示。在基准MEL数据集上的实验结果表明，我们的方法具有很强的有效性。 

---
# MCRL4OR: Multimodal Contrastive Representation Learning for Off-Road Environmental Perception 

**Title (ZH)**: MCRL4OR：离路环境感知的多模态对比表示学习 

**Authors**: Yi Yang, Zhang Zhang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13988)  

**Abstract**: Most studies on environmental perception for autonomous vehicles (AVs) focus on urban traffic environments, where the objects/stuff to be perceived are mainly from man-made scenes and scalable datasets with dense annotations can be used to train supervised learning models. By contrast, it is hard to densely annotate a large-scale off-road driving dataset manually due to the inherently unstructured nature of off-road environments. In this paper, we propose a Multimodal Contrastive Representation Learning approach for Off-Road environmental perception, namely MCRL4OR. This approach aims to jointly learn three encoders for processing visual images, locomotion states, and control actions by aligning the locomotion states with the fused features of visual images and control actions within a contrastive learning framework. The causation behind this alignment strategy is that the inertial locomotion state is the result of taking a certain control action under the current landform/terrain condition perceived by visual sensors. In experiments, we pre-train the MCRL4OR with a large-scale off-road driving dataset and adopt the learned multimodal representations for various downstream perception tasks in off-road driving scenarios. The superior performance in downstream tasks demonstrates the advantages of the pre-trained multimodal representations. The codes can be found in \url{this https URL}. 

**Abstract (ZH)**: 大多数有关自动驾驶车辆（AVs）环境感知的研究集中在城市交通环境中，这些环境中需要感知的主要对象多来自人造场景，且可以使用带有密集注释的大规模数据集来训练监督学习模型。相比之下，由于非道路环境的固有非结构化特性，人工密集标注大规模非道路驾驶数据集变得非常困难。在本文中，我们提出了一种针对非道路环境感知的多模态对比表示学习方法，即MCRL4OR。该方法旨在通过对比学习框架将运动状态与视觉图像和控制动作融合特征对齐，同时学习用于处理视觉图像、运动状态和控制动作的三个编码器。这种对齐策略背后的因果关系在于，当前地形/路况条件下视觉传感器所感知到的状态是由特定的控制动作引起的。在实验中，我们使用大规模的非道路驾驶数据集对MCRL4OR进行预训练，并使用学到的多模态表示完成各种下游感知任务。下游任务中的优异性能表明预训练的多模态表示具有优势。相关代码可以在 \url{this https URL} 获取。 

---
# Pilot: Building the Federated Multimodal Instruction Tuning Framework 

**Title (ZH)**: Pilot: 构建联邦多模态指令调优框架 

**Authors**: Baochen Xiong, Xiaoshan Yang, Yaguang Song, Yaowei Wang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13985)  

**Abstract**: In this paper, we explore a novel federated multimodal instruction tuning task(FedMIT), which is significant for collaboratively fine-tuning MLLMs on different types of multimodal instruction data on distributed devices. To solve the new task, we propose a federated multimodal instruction tuning framework(Pilot). Our framework integrates two stages of "adapter on adapter" into the connector of the vision encoder and the LLM. In stage 1, we extract task-specific features and client-specific features from visual information. In stage 2, we build the cross-task Mixture-of-Adapters(CT-MoA) module to perform cross-task interaction. Each client can not only capture personalized information of local data and learn task-related multimodal information, but also learn general knowledge from other tasks. In addition, we introduce an adaptive parameter aggregation strategy for text training parameters, which optimizes parameter aggregation by calculating weights based on the euclidean distance between parameters, so that parameter aggregation can benefit from positive effects to the greatest extent while effectively reducing negative effects. Our framework can collaboratively exploit distributed data from different local clients to learn cross-task knowledge without being affected by the task heterogeneity during instruction tuning. The effectiveness of our method is verified in two different cross-task scenarios. 

**Abstract (ZH)**: 本文探讨了一种新颖的联邦多模态指令调优任务（FedMIT），这对于在不同类型的多模态指令数据上分布式设备协同微调机器学习大模型具有重要意义。为了解决这一新任务，我们提出了一种联邦多模态指令调优框架（Pilot）。该框架在视觉编码器和LLM之间的连接器中集成了两阶段的“适配器嵌套适配器”结构。第一阶段，我们从视觉信息中提取任务特定特征和客户端特定特征；第二阶段，我们构建跨任务适配器混合模块（CT-MoA），实现跨任务交互。每个客户端不仅可以捕获本地数据的个性化信息并学习与任务相关的多模态信息，还可以从其他任务中学习通用知识。此外，我们引入了一种自适应参数聚合策略，通过基于参数间欧几里得距离计算权重来优化文本训练参数的聚合方式，从而最大程度地利用积极影响同时有效减少负面影响。该框架能够在指令调优过程中不受任务异质性的影响，协同利用来自不同本地客户端的分布式数据学习跨任务知识。我们通过两种不同的跨任务场景验证了该方法的有效性。 

---
# Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models 

**Title (ZH)**: 向着更安全的社交媒体平台迈进：基于大规模语言模型的可扩展高效少量样本有害内容审核方法 

**Authors**: Akash Bonagiri, Lucen Li, Rajvardhan Oak, Zeerak Babar, Magdalena Wojcieszak, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13976)  

**Abstract**: The prevalence of harmful content on social media platforms poses significant risks to users and society, necessitating more effective and scalable content moderation strategies. Current approaches rely on human moderators, supervised classifiers, and large volumes of training data, and often struggle with scalability, subjectivity, and the dynamic nature of harmful content (e.g., violent content, dangerous challenge trends, etc.). To bridge these gaps, we utilize Large Language Models (LLMs) to undertake few-shot dynamic content moderation via in-context learning. Through extensive experiments on multiple LLMs, we demonstrate that our few-shot approaches can outperform existing proprietary baselines (Perspective and OpenAI Moderation) as well as prior state-of-the-art few-shot learning methods, in identifying harm. We also incorporate visual information (video thumbnails) and assess if different multimodal techniques improve model performance. Our results underscore the significant benefits of employing LLM based methods for scalable and dynamic harmful content moderation online. 

**Abstract (ZH)**: 社交媒体平台上有害内容的普遍存在对用户和社会构成了重大风险，因此迫切需要更有效且可扩展的内容审核策略。当前的方法依赖于人类审核员、监督分类器和大量训练数据，常常难以应对可扩展性、主观性和有害内容的动态性（如暴力内容、危险挑战趋势等）带来的挑战。为填补这些空白，我们利用大型语言模型（LLMs）通过上下文学习进行少量样本动态内容审核。通过在多种LLM上进行大量实验，我们展示了我们的少量样本方法在识别危害方面可以超越现有的专有基准方法（如Perspective和OpenAI审核）以及此前的最佳少量样本学习方法。此外，我们还结合了视觉信息（视频缩略图），并评估了不同多模态技术是否能提升模型性能。我们的研究结果强调了使用基于LLM的方法进行在线有害内容的可扩展和动态审核的巨大优势。 

---
# Advancing the Understanding and Evaluation of AR-Generated Scenes: When Vision-Language Models Shine and Stumble 

**Title (ZH)**: 提升对AR生成场景的理解与评估：视觉-语言模型的亮点与不足 

**Authors**: Lin Duan, Yanming Xiu, Maria Gorlatova  

**Link**: [PDF](https://arxiv.org/pdf/2501.13964)  

**Abstract**: Augmented Reality (AR) enhances the real world by integrating virtual content, yet ensuring the quality, usability, and safety of AR experiences presents significant challenges. Could Vision-Language Models (VLMs) offer a solution for the automated evaluation of AR-generated scenes? Could Vision-Language Models (VLMs) offer a solution for the automated evaluation of AR-generated scenes? In this study, we evaluate the capabilities of three state-of-the-art commercial VLMs -- GPT, Gemini, and Claude -- in identifying and describing AR scenes. For this purpose, we use DiverseAR, the first AR dataset specifically designed to assess VLMs' ability to analyze virtual content across a wide range of AR scene complexities. Our findings demonstrate that VLMs are generally capable of perceiving and describing AR scenes, achieving a True Positive Rate (TPR) of up to 93\% for perception and 71\% for description. While they excel at identifying obvious virtual objects, such as a glowing apple, they struggle when faced with seamlessly integrated content, such as a virtual pot with realistic shadows. Our results highlight both the strengths and the limitations of VLMs in understanding AR scenarios. We identify key factors affecting VLM performance, including virtual content placement, rendering quality, and physical plausibility. This study underscores the potential of VLMs as tools for evaluating the quality of AR experiences. 

**Abstract (ZH)**: 增强现实（AR）通过集成虚拟内容来增强现实世界，但确保AR体验的质量、可用性和安全性面临着巨大的挑战。视觉-语言模型（VLMs）能否提供一种自动评估AR生成场景的解决方案？视觉-语言模型（VLMs）能否提供一种自动评估AR生成场景的解决方案？本研究旨在评估三款最先进的商业VLMs——GPT、Gemini和Claude——在识别和描述AR场景方面的能力。为此，我们使用了DiverseAR数据集，这是第一个专门用于评估VLMs分析不同复杂度AR场景中虚拟内容能力的数据集。研究结果表明，VLMs通常能够感知和描述AR场景，感知的真阳性率（TPR）最高可达93%，描述的TPR为71%。尽管它们在识别明显的虚拟对象（如发光的苹果）方面表现出色，但在处理无缝集成的内容（如具有逼真阴影的虚拟花瓶）时则显得力不从心。我们的结果突显了VLMs在理解AR场景方面的强项和局限性。我们确定了影响VLM性能的关键因素，包括虚拟内容的放置、渲染质量和物理合理性。本研究强调了VLMs作为一种评估AR体验质量工具的潜力。 

---
