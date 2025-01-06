# Multimodal Contrastive Representation Learning in Augmented Biomedical Knowledge Graphs 

**Title (ZH)**: augmented 生物医学知识图谱中的多模态对比表示学习 

**Authors**: Tien Dang, Viet Thanh Duy Nguyen, Minh Tuan Le, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2501.01644)  

**Abstract**: Biomedical Knowledge Graphs (BKGs) integrate diverse datasets to elucidate complex relationships within the biomedical field. Effective link prediction on these graphs can uncover valuable connections, such as potential novel drug-disease relations. We introduce a novel multimodal approach that unifies embeddings from specialized Language Models (LMs) with Graph Contrastive Learning (GCL) to enhance intra-entity relationships while employing a Knowledge Graph Embedding (KGE) model to capture inter-entity relationships for effective link prediction. To address limitations in existing BKGs, we present PrimeKG++, an enriched knowledge graph incorporating multimodal data, including biological sequences and textual descriptions for each entity type. By combining semantic and relational information in a unified representation, our approach demonstrates strong generalizability, enabling accurate link predictions even for unseen nodes. Experimental results on PrimeKG++ and the DrugBank drug-target interaction dataset demonstrate the effectiveness and robustness of our method across diverse biomedical datasets. Our source code, pre-trained models, and data are publicly available at this https URL 

**Abstract (ZH)**: 生物医学知识图谱（BKGs）综合多种数据集，以阐明生物医学领域的复杂关系。在这类图上进行有效的链接预测可以揭示有价值的联系，例如潜在的新药物-疾病关系。我们提出了一个新型的多模态方法，该方法将专用语言模型（LMs）的嵌入与图对比学习（GCL）相结合，同时使用知识图嵌入（KGE）模型捕捉实体间的关系，以实现有效的链接预测。为了克服现有BKGs的局限性，我们提出了PrimeKG++，这是一种增强的知识图谱，包含了多模态数据，包括每个实体类型的生物序列和文本描述。通过在一个统一表示中结合语义和关系信息，我们的方法展示了强大的泛化能力，即使对于未见过的节点也能进行准确的链接预测。我们在PrimeKG++和DrugBank药物-靶标相互作用数据集上的实验结果表明，我们的方法在多种生物医学数据集上具有有效性与稳健性。我们的源代码、预训练模型和数据可在以下网址公开获取：this https URL 

---
# A Metasemantic-Metapragmatic Framework for Taxonomizing Multimodal Communicative Alignment 

**Title (ZH)**: 一种元语义-元语用框架，用于分类多模态交际对齐 

**Authors**: Eugene Yu Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.01535)  

**Abstract**: Drawing on contemporary pragmatist philosophy and linguistic theories on cognition, meaning, and communication, this paper presents a dynamic, metasemantic-metapragmatic taxonomy for grounding and conceptualizing human-like multimodal communicative alignment. The framework is rooted in contemporary developments of the three basic communicative capacities initially identified by American logician and pragmatist philosopher Charles Sanders Peirce: iconic (sensory and perceptual qualities), indexical (contextual and sociocultural associations), and rule-like (symbolic and intuitive reasoning). Expanding on these developments, I introduce the concept of indexical contextualization and propose the principle of "contextualization directionality" for characterizing the crucial metapragmatic capacity for maintaining, navigating, or transitioning between semantic and pragmatic modes of multimodal communication. I contend that current cognitive-social computational and engineering methodologies disproportionately emphasize the semantic/metasemantic domain, overlooking the pivotal role of metapragmatic indexicality in traversing the semantic-pragmatic spectrum of communication. The framework's broader implications for intentionality, identity, affect, and ethics in within-modal and cross-modal human-machine alignment are also discussed. 

**Abstract (ZH)**: 本文结合现代表征主义哲学和认知、意义和沟通的语言理论，提出了一种动态的元语义-元语用分类框架，用于解释和构建类似人类的多模态沟通协调。该框架根植于美国逻辑学家和表征主义哲学家查尔斯·桑德斯·皮尔士最初提出的三种基本沟通能力：图示的（感觉和感知特性）、指示的（上下文和社会文化关联）和规则式的（符号和直觉推理）。在此基础上，本文引入了指示性上下文的概念，并提出“上下文化导向性”的原则，用于描述维持、导航或在多模态沟通的语义和语用模式之间转换的关键元语用能力。我认为，当前认识到的认知社会计算和工程方法过于强调语义/元语义领域，忽视了元语用指示性在沟通语义-语用谱系中的关键作用。此外，本文还探讨了该框架在内在模态和跨模态人类-机器协调中的意图性、身份、情感和伦理等方面的更广泛影响。 

---
# Mitigating Hallucination for Large Vision Language Model by Inter-Modality Correlation Calibration Decoding 

**Title (ZH)**: 通过跨模态相关性校准解码减轻大型视觉语言模型的幻觉现象 

**Authors**: Jiaming Li, Jiacheng Zhang, Zequn Jie, Lin Ma, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.01926)  

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in visual-language understanding for downstream multi-modal tasks. Despite their success, LVLMs still suffer from generating hallucinations in complex generation tasks, leading to inconsistencies between visual inputs and generated content. To address this issue, some approaches have introduced inference-time interventions, such as contrastive decoding and attention rectification, to reduce overreliance on language priors. However, these approaches overlook hallucinations stemming from spurious inter-modality correlations. In this paper, we propose an Inter-Modality Correlation Calibration Decoding (IMCCD) method to mitigate hallucinations in LVLMs in a training-free manner. In this method, we design a Cross-Modal Value-Enhanced Decoding(CMVED) module to alleviate hallucination by a novel contrastive decoding mechanism. During the estimation of distorted distribution, CMVED masks the value vectors associated with significant cross-modal attention weights, which address both uni-modality overreliance and misleading inter-modality correlations. Additionally, a Content-Driven Attention Refinement(CDAR) module refines cross-modal attention weights, guiding LVLMs to focus on important visual content. Experimental results on diverse hallucination benchmarks validate the superiority of our method over existing state-of-the-art techniques in reducing hallucinations in LVLM text generation. Our code will be available at this https URL. 

**Abstract (ZH)**: 大型多模态模型（Large Vision-Language Models, LVLMs）在下游多模态任务中的视觉-语言理解方面展现出了杰出的能力。尽管这些模型取得了显著的成果，但在复杂的生成任务中，LVLMs仍然容易生成幻觉，导致视觉输入与生成内容之间的一致性差。为了解决这一问题，一些方法引入了推理时的干预措施，如对比解码和注意力校正，以减少对语言先验的过度依赖。然而，这些方法忽略了由虚假的跨模态相关性引起的幻觉。在本文中，我们提出了一种训练无干预的跨模态相关性校准解码（Inter-Modality Correlation Calibration Decoding, IMCCD）方法，以减轻LVLMs中的幻觉。该方法设计了一种跨模态值增强解码（Cross-Modal Value-Enhanced Decoding, CMVED）模块，通过一种新的对比解码机制来减轻幻觉。在估计畸变分布时，CMVED会遮掩与显著的跨模态注意力权重相关联的值向量，从而解决单一模态的过度依赖和误导性的跨模态相关性。此外，内容驱动的注意力精炼（Content-Driven Attention Refinement, CDAR）模块能够精炼跨模态注意力权重，指导LVLMs关注重要的视觉内容。我们的实验结果在多种幻觉基准测试上验证了该方法在减少LVLM文本生成中的幻觉方面优于现有的先进方法。我们的代码将在此处提供：[请填写具体的网址]。 

---
# MoColl: Agent-Based Specific and General Model Collaboration for Image Captioning 

**Title (ZH)**: MoColl：基于代理的特定模型与通用模型协作的图像_captioning方法 

**Authors**: Pu Yang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.01834)  

**Abstract**: Image captioning is a critical task at the intersection of computer vision and natural language processing, with wide-ranging applications across various domains. For complex tasks such as diagnostic report generation, deep learning models require not only domain-specific image-caption datasets but also the incorporation of relevant general knowledge to provide contextual accuracy. Existing approaches exhibit inherent limitations: specialized models excel in capturing domain-specific details but lack generalization, while vision-language models (VLMs) built on large language models (LLMs) leverage general knowledge but struggle with domain-specific adaptation. To address these limitations, this paper proposes a novel agent-enhanced model collaboration framework, which we called \textbf{MoColl}, designed to effectively integrate domain-specific and general knowledge. Specifically, our approach is to decompose complex image captioning tasks into a series of interconnected question-answer subtasks. A trainable visual question answering (VQA) model is employed as a specialized tool to focus on domain-specific visual analysis, answering task-specific questions based on image content. Concurrently, an LLM-based agent with general knowledge formulates these questions and synthesizes the resulting question-answer pairs into coherent captions. Beyond its role in leveraging the VQA model, the agent further guides its training to enhance its domain-specific capabilities. Experimental results on radiology report generation validate the effectiveness of the proposed framework, demonstrating significant improvements in the quality of generated reports. 

**Abstract (ZH)**: 图像描述是计算机视觉与自然语言处理交叉领域的关键任务，具有广泛的应用前景。在诸如诊断报告生成等复杂任务中，深度学习模型不仅需要领域特定的图像-描述数据集，还需要融入相关的一般知识以提供上下文准确性。现有的方法存在固有的局限性：领域特定模型在捕捉领域细节方面表现优异，但在泛化能力上存在不足；基于大规模语言模型（LLM）的视觉-语言模型（VLM）虽能利用一般知识，但在领域特定适应性上却显得力不从心。为解决这些局限性，本文提出了一种新的代理增强模型协作框架，我们称之为**MoColl**，旨在有效整合领域特定和一般知识。具体而言，我们的方法将复杂的图像描述任务分解为一系列相互连接的问答子任务。一个可训练的视觉问答（VQA）模型被用作专门工具，专注于领域特定的视觉分析，基于图像内容回答特定任务的问题。同时，一个具有通用知识的LLM代理生成这些问题，并将生成的问答对整合成连贯的描述。除了利用VQA模型外，该代理进一步引导其训练以增强其领域特定的能力。在医学影像报告生成的实验中，所提出框架的有效性得到了验证，显著提升了生成报告的质量。 

---
# MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders 

**Title (ZH)**: MoVE-KD：多视觉编码器混合的VLMs知识蒸馏

解释：
- MoVE-KD: 这里保持了原词，因为它是该方法的缩写名称。
- Knowledge Distillation: 知识蒸馏
- VLMs: Vision-Language Models，视觉语言模型
- Mixture of Visual Encoders: 多视觉编码器混合 

**Authors**: Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning MA, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01709)  

**Abstract**: Visual encoders are fundamental components in vision-language models (VLMs), each showcasing unique strengths derived from various pre-trained visual foundation models. To leverage the various capabilities of these encoders, recent studies incorporate multiple encoders within a single VLM, leading to a considerable increase in computational cost. In this paper, we present Mixture-of-Visual-Encoder Knowledge Distillation (MoVE-KD), a novel framework that distills the unique proficiencies of multiple vision encoders into a single, efficient encoder model. Specifically, to mitigate conflicts and retain the unique characteristics of each teacher encoder, we employ low-rank adaptation (LoRA) and mixture-of-experts (MoEs) to selectively activate specialized knowledge based on input features, enhancing both adaptability and efficiency. To regularize the KD process and enhance performance, we propose an attention-based distillation strategy that adaptively weighs the different visual encoders and emphasizes valuable visual tokens, reducing the burden of replicating comprehensive but distinct features from multiple teachers. Comprehensive experiments on popular VLMs, such as LLaVA and LLaVA-NeXT, validate the effectiveness of our method. The code will be released. 

**Abstract (ZH)**: 视觉编码器是视觉-语言模型（VLMs）的基本组件，每个编码器都源自不同预训练视觉基础模型的独特优势。为了充分利用这些编码器的各种能力，最近的研究将多个编码器集成到单一的VLM中，导致计算成本显著增加。在本文中，我们提出了混合视觉编码器知识蒸馏（MoVE-KD）的新框架，该框架将多种视觉编码器的独特优势精炼到一个高效编码器模型中。具体而言，为了缓解冲突并保留每个教师编码器的独特特性，我们采用了低秩适应（LoRA）和混合专家（MoEs）的方法，根据输入特征选择性地激活特定知识，从而增强适应性和效率。为了规范知识蒸馏过程并提高性能，我们提出了一种基于注意力的知识蒸馏策略，该策略根据不同视觉编码器的权重动态调整，并强调有价值的视觉标记，从而减少了复制多个教师的全面但不同的特征的负担。在LLaVA和LLaVA-NeXT等流行的VLMs上的全面实验验证了我们方法的有效性。代码将公开发布。 

---
# HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding 

**Title (ZH)**: HLV-1K：一个小时大规模视频基准，用于时间特定长视频理解 

**Authors**: Heqing Zou, Tianze Luo, Guiyang Xie, Victor, Zhang, Fengmao Lv, Guangcong Wang, Junyang Chen, Zhuochen Wang, Hansheng Zhang, Huaijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01645)  

**Abstract**: Multimodal large language models have become a popular topic in deep visual understanding due to many promising real-world applications. However, hour-long video understanding, spanning over one hour and containing tens of thousands of visual frames, remains under-explored because of 1) challenging long-term video analyses, 2) inefficient large-model approaches, and 3) lack of large-scale benchmark datasets. Among them, in this paper, we focus on building a large-scale hour-long long video benchmark, HLV-1K, designed to evaluate long video understanding models. HLV-1K comprises 1009 hour-long videos with 14,847 high-quality question answering (QA) and multi-choice question asnwering (MCQA) pairs with time-aware query and diverse annotations, covering frame-level, within-event-level, cross-event-level, and long-term reasoning tasks. We evaluate our benchmark using existing state-of-the-art methods and demonstrate its value for testing deep long video understanding capabilities at different levels and for various tasks. This includes promoting future long video understanding tasks at a granular level, such as deep understanding of long live videos, meeting recordings, and movies. 

**Abstract (ZH)**: 多模态大型语言模型已成为深度视觉理解领域的热门话题，由于其在许多实际应用中的潜在用途。然而，长达一小时的视频理解仍处于未充分探索的状态，因为它面临以下挑战：1）长期视频分析的难题；2）大型模型方法的低效；3）缺乏大规模基准数据集。在这其中，本文专注于构建一个大规模一小时长视频基准数据集，HLV-1K，旨在评估长期视频理解模型。HLV-1K 包含1009个一小时的视频，其中有14,847对高质量的问题回答（QA）和多项选择题（MCQA），这些数据集具有时间感知查询和多样化的注释，涵盖了帧级、事件内级、跨事件级以及长期推理任务。我们使用现有的最先进方法来评估这一基准数据集，并展示了其在不同层次和各种任务中测试深度长视频理解能力的价值。这包括推动未来精确粒度的长视频理解任务发展，例如长实时视频的理解、会议记录以及电影的理解。 

---
# Google is all you need: Semi-Supervised Transfer Learning Strategy For Light Multimodal Multi-Task Classification Model 

**Title (ZH)**: 谷歌之力即你所需：半监督转移学习策略用于轻量级多模态多任务分类模型 

**Authors**: Haixu Liu, Penghao Jiang, Zerui Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01611)  

**Abstract**: As the volume of digital image data increases, the effectiveness of image classification intensifies. This study introduces a robust multi-label classification system designed to assign multiple labels to a single image, addressing the complexity of images that may be associated with multiple categories (ranging from 1 to 19, excluding 12). We propose a multi-modal classifier that merges advanced image recognition algorithms with Natural Language Processing (NLP) models, incorporating a fusion module to integrate these distinct modalities. The purpose of integrating textual data is to enhance the accuracy of label prediction by providing contextual understanding that visual analysis alone cannot fully capture. Our proposed classification model combines Convolutional Neural Networks (CNN) for image processing with NLP techniques for analyzing textual description (i.e., captions). This approach includes rigorous training and validation phases, with each model component verified and analyzed through ablation experiments. Preliminary results demonstrate the classifier's accuracy and efficiency, highlighting its potential as an automatic image-labeling system. 

**Abstract (ZH)**: 随着数字图像数据量的增加，图像分类的有效性也随之增强。本研究介绍了一种稳健的多标签分类系统，该系统旨在为单张图像分配多个标签，以应对可能与多个类别（从1到19，不包括12）相关联的图像复杂性。我们提出了一种多模态分类器，该分类器结合了先进的图像识别算法和自然语言处理（NLP）模型，并通过融合模块将这些不同的模态整合在一起。将文本数据纳入融合的目的是通过提供仅视觉分析无法完全捕捉到的上下文理解，来提高标签预测的准确性。我们提出的分类模型结合了卷积神经网络（CNN）用于图像处理，以及NLP技术用于分析文本描述（例如，字幕）。该方法包括严格的训练和验证阶段，每个模型组件都通过消融实验进行了验证和分析。初步结果表明，该分类器具有高精度和高效性，显示出其作为自动图像标签系统的发展潜力。 

---
# Balance-aware Sequence Sampling Makes Multi-modal Learning Better 

**Title (ZH)**: 平衡意识序列采样提升多模态学习效果 

**Authors**: Zhi-Hao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2501.01470)  

**Abstract**: To address the modality imbalance caused by data heterogeneity, existing multi-modal learning (MML) approaches primarily focus on balancing this difference from the perspective of optimization objectives. However, almost all existing methods ignore the impact of sample sequences, i.e., an inappropriate training order tends to trigger learning bias in the model, further exacerbating modality imbalance. In this paper, we propose Balance-aware Sequence Sampling (BSS) to enhance the robustness of MML. Specifically, we first define a multi-perspective measurer to evaluate the balance degree of each sample. Via the evaluation, we employ a heuristic scheduler based on curriculum learning (CL) that incrementally provides training subsets, progressing from balanced to imbalanced samples to rebalance MML. Moreover, considering that sample balance may evolve as the model capability increases, we propose a learning-based probabilistic sampling method to dynamically update the training sequences at the epoch level, further improving MML performance. Extensive experiments on widely used datasets demonstrate the superiority of our method compared with state-of-the-art (SOTA) MML approaches. 

**Abstract (ZH)**: 为了解决由数据异质性引起的模态不平衡问题，现有的多模态学习（MML）方法主要从优化目标的角度来平衡这种差异。然而，几乎所有的现有方法都忽视了样本序列的影响，即不恰当的训练顺序容易导致学习偏向，进一步加剧了模态不平衡。本文提出了一种平衡感知的序列采样（BSS）方法，以增强MML的鲁棒性。具体来说，我们首先定义一个多视角度量器来评估每个样本的平衡程度。通过这一评估，我们采用一个基于渐进学习（CL）的启发式调度器，逐步提供训练子集，从平衡样本过渡到不平衡样本，以重新平衡MML。此外，考虑到样本平衡可能会随着模型能力的提高而变化，我们提出了一个基于学习的概率采样方法，在每次迭代中动态更新训练序列，进一步提高MML性能。广泛的实验表明，在广泛使用的数据集上，我们的方法在与现有最先进的（SOTA）MML方法比较时表现出更优的效果。 

---
