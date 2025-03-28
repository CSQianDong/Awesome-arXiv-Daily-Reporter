# GePBench: Evaluating Fundamental Geometric Perception for Multimodal Large Language Models 

**Title (ZH)**: GePBench：评估多模态大型语言模型的基本几何感知能力 

**Authors**: Shangyu Xing, Changhao Xiang, Yuteng Han, Yifan Yue, Zhen Wu, Xinyu Liu, Zhangtai Wu, Fei Zhao, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2412.21036)  

**Abstract**: Multimodal large language models (MLLMs) have achieved significant advancements in integrating visual and linguistic understanding. While existing benchmarks evaluate these models in context-rich, real-life scenarios, they often overlook fundamental perceptual skills essential for environments deviating from everyday realism. In particular, geometric perception, the ability to interpret spatial relationships and abstract visual patterns, remains underexplored. To address this limitation, we introduce GePBench, a novel benchmark designed to assess the geometric perception capabilities of MLLMs. Results from extensive evaluations reveal that current state-of-the-art MLLMs exhibit significant deficiencies in such tasks. Additionally, we demonstrate that models trained with data sourced from GePBench show notable improvements on a wide range of downstream tasks, underscoring the importance of geometric perception as a foundation for advanced multimodal applications. Our code and datasets will be publicly available. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）已在结合视觉和语言理解方面取得了显著进展。虽然现有的基准测试在丰富的上下文环境中评估了这些模型，但在偏离日常现实环境的情况下，它们往往忽视了用于这些环境的基本感知技能。特别是几何感知能力，即解释空间关系和抽象视觉模式的能力，仍然未被充分探索。为解决这一局限，我们提出了GePBench，一个旨在评估MLLMs几何感知能力的新基准测试。广泛的评估结果表明，当前最先进的MLLMs在这些任务中存在显著缺陷。此外，我们还展示了使用来自GePBench的数据进行训练的模型在多种下游任务中表现出明显的改进，突显了几何感知作为高级多模态应用基础的重要性。我们的代码和数据集将公开提供。 

---
# SAFE-MEME: Structured Reasoning Framework for Robust Hate Speech Detection in Memes 

**Title (ZH)**: SAFE-MEME： meme中鲁棒仇恨言论检测的结构化推理框架 

**Authors**: Palash Nandi, Shivam Sharma, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2412.20541)  

**Abstract**: Memes act as cryptic tools for sharing sensitive ideas, often requiring contextual knowledge to interpret. This makes moderating multimodal memes challenging, as existing works either lack high-quality datasets on nuanced hate categories or rely on low-quality social media visuals. Here, we curate two novel multimodal hate speech datasets, MHS and MHS-Con, that capture fine-grained hateful abstractions in regular and confounding scenarios, respectively. We benchmark these datasets against several competing baselines. Furthermore, we introduce SAFE-MEME (Structured reAsoning FramEwork), a novel multimodal Chain-of-Thought-based framework employing Q&A-style reasoning (SAFE-MEME-QA) and hierarchical categorization (SAFE-MEME-H) to enable robust hate speech detection in memes. SAFE-MEME-QA outperforms existing baselines, achieving an average improvement of approximately 5% and 4% on MHS and MHS-Con, respectively. In comparison, SAFE-MEME-H achieves an average improvement of 6% in MHS while outperforming only multimodal baselines in MHS-Con. We show that fine-tuning a single-layer adapter within SAFE-MEME-H outperforms fully fine-tuned models in regular fine-grained hateful meme detection. However, the fully fine-tuning approach with a Q&A setup is more effective for handling confounding cases. We also systematically examine the error cases, offering valuable insights into the robustness and limitations of the proposed structured reasoning framework for analyzing hateful memes. 

**Abstract (ZH)**: 模因作为隐秘工具，用于传递敏感思想，常常需要特定的情境知识才能解读。这使得对多模态模因进行管理变得具有挑战性，因为现有的工作要么缺乏针对细微厌恶类别的高质量数据集，要么依赖于低质量的社会媒体视觉素材。在此，我们整理了两个全新的多模态仇恨言论数据集MHS和MHS-Con，分别捕捉了常规和混淆场景下的细微仇恨抽象。我们使用这些数据集对多个竞争基准模型进行了评估。此外，我们引入了SAFE-MEME（结构化推理框架），这是一种新颖的多模态因果推理框架，采用问答式推理（SAFE-MEME-QA）与层次分类（SAFE-MEME-H），以在模因中实现稳健的仇恨言论检测。SAFE-MEME-QA 在MHS 和 MHS-Con 上的性能优于现有基线，分别提高了约5% 和4%。相比之下，在MHS上，SAFE-MEME-H 在 MHS上实现了约6%的平均改进，同时在MHS-Con上仅优于多模态基础模型。我们展示了在SAFE-MEME-H中对单一层数适应器进行微调优于完全微调模型在常规细微仇恨模因检测中的表现，但在处理混淆情况时，问答式设置的完全微调方法更为有效。我们还系统地分析了错误案例，提供了有关所提结构化推理框架分析仇恨模因的稳健性和局限性的宝贵见解。 

---
# Utilizing Multimodal Data for Edge Case Robust Call-sign Recognition and Understanding 

**Title (ZH)**: 利用多模态数据提高边缘情况下的呼叫标识识别与理解 

**Authors**: Alexander Blatt, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2412.20467)  

**Abstract**: Operational machine-learning based assistant systems must be robust in a wide range of scenarios. This hold especially true for the air-traffic control (ATC) domain. The robustness of an architecture is particularly evident in edge cases, such as high word error rate (WER) transcripts resulting from noisy ATC recordings or partial transcripts due to clipped recordings. To increase the edge-case robustness of call-sign recognition and understanding (CRU), a core tasks in ATC speech processing, we propose the multimodal call-sign-command recovery model (CCR). The CCR architecture leads to an increase in the edge case performance of up to 15%. We demonstrate this on our second proposed architecture, CallSBERT. A CRU model that has less parameters, can be fine-tuned noticeably faster and is more robust during fine-tuning than the state of the art for CRU. Furthermore, we demonstrate that optimizing for edge cases leads to a significantly higher accuracy across a wide operational range. 

**Abstract (ZH)**: 基于操作的机器学习辅助系统在各种场景下必须具备鲁棒性，特别是在空中交通管制（ATC）领域，这一点尤为重要。架构的鲁棒性在极端情况下尤为明显，例如由于嘈杂的ATC录音导致的高字错误率（WER）转录或由于片段录音导致的部分转录。为提高呼号识别和理解（CRU，Cue Recogniion and Understanding）这一ATC语音处理的核心任务在边缘情况下的鲁棒性，我们提出了多模态呼号-命令恢复模型（CCR，Call Sign-Cue Recovery）。CCR架构在边缘情况下的性能提高了最多15%。我们在我们提出的第二个架构CallSBERT中展示了这一点。CallSBERT是一个参数更少、在微调过程中显著更快并具有更好的微调鲁棒性的CRU模型，其性能优于现有最先进的CRU模型。此外，我们证明了在边缘情况下的优化能够在广泛的运行范围内显著提高准确性。 

---
# Enhancing Multimodal Emotion Recognition through Multi-Granularity Cross-Modal Alignment 

**Title (ZH)**: 通过多粒度跨模态对齐增强多模态情感识别 

**Authors**: Xuechen Wang, Shiwan Zhao, Haoqin Sun, Hui Wang, Jiaming Zhou, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.20821)  

**Abstract**: Multimodal emotion recognition (MER), leveraging speech and text, has emerged as a pivotal domain within human-computer interaction, demanding sophisticated methods for effective multimodal integration. The challenge of aligning features across these modalities is significant, with most existing approaches adopting a singular alignment strategy. Such a narrow focus not only limits model performance but also fails to address the complexity and ambiguity inherent in emotional expressions. In response, this paper introduces a Multi-Granularity Cross-Modal Alignment (MGCMA) framework, distinguished by its comprehensive approach encompassing distribution-based, instance-based, and token-based alignment modules. This framework enables a multi-level perception of emotional information across modalities. Our experiments on IEMOCAP demonstrate that our proposed method outperforms current state-of-the-art techniques. 

**Abstract (ZH)**: 多模态情感识别（MER），利用语音和文本信息，在人机交互领域中已成为一个关键领域，要求采用复杂方法实现有效的多模态整合。这些模态之间对齐特征的难题非常显著，现有的大多数方法都采用了单一的对齐策略。这种狭隘的视角不仅限制了模型性能，而且没有解决情感表达固有的复杂性和模糊性。为应对这一挑战，本文提出了一种多粒度跨模态对齐（MGCMA）框架，该框架包括基于分布、基于实例和基于标记的对齐模块，以实现全面的多级情感信息感知。在IEMOCAP数据集上的实验表明，我们提出的方法在当前最先进的技术中表现更优。 

---
# ChartAdapter: Large Vision-Language Model for Chart Summarization 

**Title (ZH)**: ChartAdapter：大型vision-language模型用于图表总结 

**Authors**: Peixin Xu, Yujuan Ding, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20715)  

**Abstract**: Chart summarization, which focuses on extracting key information from charts and interpreting it in natural language, is crucial for generating and delivering insights through effective and accessible data analysis. Traditional methods for chart understanding and summarization often rely on multi-stage pipelines, which may produce suboptimal semantic alignment between visual and textual information. In comparison, recently developed LLM-based methods are more dependent on the capability of foundation images or languages, while ignoring the characteristics of chart data and its relevant challenges. To address these limitations, we propose ChartAdapter, a novel lightweight transformer module designed to bridge the gap between charts and textual summaries. ChartAdapter employs learnable query vectors to extract implicit semantics from chart data and incorporates a cross-modal alignment projector to enhance vision-to-language generative learning. By integrating ChartAdapter with an LLM, we enable end-to-end training and efficient chart summarization. To further enhance the training, we introduce a three-stage hierarchical training procedure and develop a large-scale dataset specifically curated for chart summarization, comprising 190,618 samples. Experimental results on the standard Chart-to-Text testing set demonstrate that our approach significantly outperforms existing methods, including state-of-the-art models, in generating high-quality chart summaries. Ablation studies further validate the effectiveness of key components in ChartAdapter. This work highlights the potential of tailored LLM-based approaches to advance chart understanding and sets a strong foundation for future research in this area. 

**Abstract (ZH)**: 图表总结专注于从图表中提取关键信息并以自然语言进行解释，对于通过有效且易于访问的数据分析生成和传递洞察至关重要。传统的图表理解和总结方法通常依赖于多阶段管道，可能会导致图表信息与文本信息之间的语义对齐不理想。相比之下，近年来开发的基于大语言模型（LLM）的方法更加依赖于基础视觉或语言模型的能力，而忽视了图表数据的特性及其相关挑战。为了解决这些限制，我们提出了一种名为ChartAdapter的新颖轻量级transformer模块，旨在弥补图表与文本总结之间的差距。ChartAdapter利用可学习的查询向量从图表数据中提取潜在语义，并结合跨模态对齐投影器来增强视觉到语言生成学习。通过将ChartAdapter与大语言模型结合，我们实现了端到端的训练和高效的图表总结。为进一步增强训练，我们引入了三层级的分层训练程序，并开发了一个专门用于图表总结的大规模数据集，共包含190,618个样本。在标准的图表到文本测试集上的实验结果表明，我们的方法在生成高质量的图表总结方面显著优于现有方法，包括最先进的模型。进一步的消融研究验证了ChartAdapter中关键组件的有效性。这项工作突显了定制的大语言模型方法在推进图表理解方面的潜力，并为该领域的未来研究奠定了坚实基础。 

---
# Towards Identity-Aware Cross-Modal Retrieval: a Dataset and a Baseline 

**Title (ZH)**: 面向身份意识的跨模态检索：一个数据集和基准模型 

**Authors**: Nicola Messina, Lucia Vadicamo, Leo Maltese, Claudio Gennaro  

**Link**: [PDF](https://arxiv.org/pdf/2412.21009)  

**Abstract**: Recent advancements in deep learning have significantly enhanced content-based retrieval methods, notably through models like CLIP that map images and texts into a shared embedding space. However, these methods often struggle with domain-specific entities and long-tail concepts absent from their training data, particularly in identifying specific individuals. In this paper, we explore the task of identity-aware cross-modal retrieval, which aims to retrieve images of persons in specific contexts based on natural language queries. This task is critical in various scenarios, such as for searching and browsing personalized video collections or large audio-visual archives maintained by national broadcasters. We introduce a novel dataset, COCO Person FaceSwap (COCO-PFS), derived from the widely used COCO dataset and enriched with deepfake-generated faces from VGGFace2. This dataset addresses the lack of large-scale datasets needed for training and evaluating models for this task. Our experiments assess the performance of different CLIP variations repurposed for this task, including our architecture, Identity-aware CLIP (Id-CLIP), which achieves competitive retrieval performance through targeted fine-tuning. Our contributions lay the groundwork for more robust cross-modal retrieval systems capable of recognizing long-tail identities and contextual nuances. Data and code are available at this https URL. 

**Abstract (ZH)**: 近年来，深度学习的进步显著提升了基于内容的检索方法，特别是通过CLIP等模型将图像和文本映射到共享嵌入空间中。然而，这些方法在处理训练数据中缺乏的专业领域实体和长尾概念时往往表现不佳，特别是在识别特定个体方面。本文探讨了身份感知多模态检索的任务，该任务旨在根据自然语言查询基于特定上下文检索特定个体的图像。这一任务在多种场景中至关重要，例如在搜索和浏览个性化视频集合或由国家级广播机构维护的大规模音频-视觉档案时。我们引入了一个全新的数据集，COCO Person FaceSwap（COCO-PFS），该数据集基于广为使用的COCO数据集，并通过VGGFace2生成的深度伪造面孔进行丰富。这一数据集解决了训练和评估此类任务所需的大型数据集缺乏的问题。我们的实验评估了不同CLIP变体在这项任务中的性能，包括我们的架构，身份感知CLIP（Id-CLIP），该架构通过针对性的微调实现了竞争力的检索性能。我们的贡献为构建更加稳健的多模态检索系统奠定了基础，这些系统能够识别长尾身份和上下文微 辽。数据和代码可在以下链接获取：[这里提供链接]。 

---
# WalkVLM:Aid Visually Impaired People Walking by Vision Language Model 

**Title (ZH)**: WalkVLM：通过视觉语言模型辅助视障人士行走 

**Authors**: Zhiqiang Yuan, Ting Zhang, Jiapei Zhang, Jie Zhou, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20903)  

**Abstract**: Approximately 200 million individuals around the world suffer from varying degrees of visual impairment, making it crucial to leverage AI technology to offer walking assistance for these people. With the recent progress of vision-language models (VLMs), employing VLMs to improve this field has emerged as a popular research topic. However, most existing methods are studied on self-built question-answering datasets, lacking a unified training and testing benchmark for walk guidance. Moreover, in blind walking task, it is necessary to perform real-time streaming video parsing and generate concise yet informative reminders, which poses a great challenge for VLMs that suffer from redundant responses and low inference efficiency. In this paper, we firstly release a diverse, extensive, and unbiased walking awareness dataset, containing 12k video-manual annotation pairs from Europe and Asia to provide a fair training and testing benchmark for blind walking task. Furthermore, a WalkVLM model is proposed, which employs chain of thought for hierarchical planning to generate concise but informative reminders and utilizes temporal-aware adaptive prediction to reduce the temporal redundancy of reminders. Finally, we have established a solid benchmark for blind walking task and verified the advantages of WalkVLM in stream video processing for this task compared to other VLMs. Our dataset and code will be released at anonymous link this https URL. 

**Abstract (ZH)**: 在全球范围内，大约有2亿人不同程度地遭受视力障碍的困扰，因此利用人工智能技术为这些人群提供行走辅助变得至关重要。随着视觉-语言模型（VLMs）的进展，使用VLMs来改善这一领域已经成为一个热门的研究方向。然而，现有的大多数方法主要在自我构建的问题-回答数据集上进行研究，缺乏一个统一的训练和测试基准来评估行走指导。此外，在盲人行走任务中，需要实时处理视频流并生成简洁但富有信息性的提醒，这给VLMs带来了很大的挑战，因为它们容易产生冗余响应和较低的推理效率。在本文中，我们首先发布了一个多样、广泛且不偏不倚的行走意识数据集，包含来自欧洲和亚洲的12000个视频-人工注释对，旨在为盲人行走任务提供一个公平的训练和测试基准。此外，我们提出了一种WalkVLM模型，该模型通过分层规划使用推理链生成简洁但富有信息性的提醒，并利用时空感知自适应预测来减少提醒中的时间冗余。最后，我们建立了一个盲人行走任务的基准，并验证了WalkVLM在处理该任务的实时视频流方面相比其他VLMs的优势。我们的数据集和代码将通过以下匿名链接发布：[https://anonymous.link]。 

---
# M$^3$oralBench: A MultiModal Moral Benchmark for LVLMs 

**Title (ZH)**: M$^3$oralBench: 一种面向多模态语言模型的道德基准测试 

**Authors**: Bei Yan, Jie Zhang, Zhiyuan Chen, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20718)  

**Abstract**: Recently, large foundation models, including large language models (LLMs) and large vision-language models (LVLMs), have become essential tools in critical fields such as law, finance, and healthcare. As these models increasingly integrate into our daily life, it is necessary to conduct moral evaluation to ensure that their outputs align with human values and remain within moral boundaries. Previous works primarily focus on LLMs, proposing moral datasets and benchmarks limited to text modality. However, given the rapid development of LVLMs, there is still a lack of multimodal moral evaluation methods. To bridge this gap, we introduce M$^3$oralBench, the first MultiModal Moral Benchmark for LVLMs. M$^3$oralBench expands the everyday moral scenarios in Moral Foundations Vignettes (MFVs) and employs the text-to-image diffusion model, SD3.0, to create corresponding scenario images. It conducts moral evaluation across six moral foundations of Moral Foundations Theory (MFT) and encompasses tasks in moral judgement, moral classification, and moral response, providing a comprehensive assessment of model performance in multimodal moral understanding and reasoning. Extensive experiments on 10 popular open-source and closed-source LVLMs demonstrate that M$^3$oralBench is a challenging benchmark, exposing notable moral limitations in current models. Our benchmark is publicly available. 

**Abstract (ZH)**: 近年来，大型基础模型，包括大型语言模型（LLMs）和大型视觉-语言模型（LVLMs），在法律、金融和医疗等关键领域已成为不可或缺的工具。随着这些模型越来越多地融入我们的日常生活，进行道德评估变得必要，以确保其输出与人类价值观相符，并保持在道德边界之内。以往的工作主要关注LLMs，提出了限于文本模态的道德数据集和基准测试。然而，鉴于LVLMs的快速发展，仍然缺乏多模态道德评估方法。为了弥补这一差距，我们引入了M$^3$oralBench，这是首个针对LVLMs的多模态道德基准测试。M$^3$oralBench 扩展了《道德基础情景》（MFVs）中的日常生活道德场景，并利用文本到图像扩散模型SD3.0创建相应的场景图像。该基准测试涵盖了《道德基础理论》（MFT）中的六种道德基础，并包含道德判断、道德分类和道德回应任务，提供了模型在多模态道德理解和推理方面的全面评估。针对10种流行开源和闭源LVLMs进行的广泛实验表明，M$^3$oralBench 是一个具有挑战性的基准测试，揭示了当前模型在道德方面的显著局限性。我们的基准测试已公开可用。 

---
# HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models 

**Title (ZH)**: 幻觉：评估大型视觉语言模型中对象幻觉的基准 

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2412.20622)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in performing complex multimodal tasks. However, they are still plagued by object hallucination: the misidentification or misclassification of objects present in images. To this end, we propose HALLUCINOGEN, a novel visual question answering (VQA) object hallucination attack benchmark that utilizes diverse contextual reasoning prompts to evaluate object hallucination in state-of-the-art LVLMs. We design a series of contextual reasoning hallucination prompts to evaluate LVLMs' ability to accurately identify objects in a target image while asking them to perform diverse visual-language tasks such as identifying, locating or performing visual reasoning around specific objects. Further, we extend our benchmark to high-stakes medical applications and introduce MED-HALLUCINOGEN, hallucination attacks tailored to the biomedical domain, and evaluate the hallucination performance of LVLMs on medical images, a critical area where precision is crucial. Finally, we conduct extensive evaluations of eight LVLMs and two hallucination mitigation strategies across multiple datasets to show that current generic and medical LVLMs remain susceptible to hallucination attacks. 

**Abstract (ZH)**: 大规模多模态语言视觉模型（Large Vision-Language Models, LVLMs）在执行复杂多模态任务方面展现了非凡的能力。然而，它们仍然受到物体幻觉的困扰：即将图像中存在的物体错误地识别或分类。为解决这一问题，我们提出了一种名为HALLUCINOGEN的新颖视觉问答（Visual Question Answering, VQA）物体幻觉攻击基准，该基准利用多样化的上下文推理提示来评估最先进的LVLMs中的物体幻觉情况。我们设计了一系列上下文推理幻觉提示，以评估LVLMs在执行诸如识别、定位或围绕特定物体进行视觉推理等多样视觉-语言任务时准确识别目标图像中物体的能力。此外，我们还将基准扩展到了高风险的医疗应用领域，并引入了MED-HALLUCINOGEN这一针对生物医学领域定制的幻觉攻击，以评估LVLMs在医疗图像中的幻觉性能，医疗领域对精确性要求极高。最后，我们在多个数据集中对八种LVLMs和两种幻觉缓解策略进行了详尽的评估，以证明当前通用和医疗专用的LVLMs仍然容易受到幻觉攻击的影响。 

---
# Multi-Scenario Reasoning: Unlocking Cognitive Autonomy in Humanoid Robots for Multimodal Understanding 

**Title (ZH)**: 多场景推理：在类人机器人中实现多模态理解的认知自主性 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20429)  

**Abstract**: To improve the cognitive autonomy of humanoid robots, this research proposes a multi-scenario reasoning architecture to solve the technical shortcomings of multi-modal understanding in this field. It draws on simulation based experimental design that adopts multi-modal synthesis (visual, auditory, tactile) and builds a simulator "Maha" to perform the experiment. The findings demonstrate the feasibility of this architecture in multimodal data. It provides reference experience for the exploration of cross-modal interaction strategies for humanoid robots in dynamic environments. 

**Abstract (ZH)**: 为了提高类人机器人的情境认知自主能力，本研究提出了一种多场景推理架构，以解决该领域多模态理解的技术短板。该架构借鉴了基于仿真的实验设计，采用了多模态合成（视觉、听觉、触觉），并构建了一个名为“Maha”的模拟器来执行实验。研究发现表明了该架构在多模态数据上的可行性，为其在动态环境下类人机器人跨模态交互策略的探索提供了参考经验。 

---
# Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems 

**Title (ZH)**: 将可解释性和轻量级设计注入弱监督视频异常检测系统 

**Authors**: Wen-Dong Jiang, Chih-Yung Chang, Hsiang-Chuan Chang, Ji-Yuan Chen, Diptendu Sinha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2412.20201)  

**Abstract**: Weakly Supervised Monitoring Anomaly Detection (WSMAD) utilizes weak supervision learning to identify anomalies, a critical task for smart city monitoring. However, existing multimodal approaches often fail to meet the real-time and interpretability requirements of edge devices due to their complexity. This paper presents TCVADS (Two-stage Cross-modal Video Anomaly Detection System), which leverages knowledge distillation and cross-modal contrastive learning to enable efficient, accurate, and interpretable anomaly detection on edge this http URL operates in two stages: coarse-grained rapid classification and fine-grained detailed analysis. In the first stage, TCVADS extracts features from video frames and inputs them into a time series analysis module, which acts as the teacher model. Insights are then transferred via knowledge distillation to a simplified convolutional network (student model) for binary classification. Upon detecting an anomaly, the second stage is triggered, employing a fine-grained multi-class classification model. This stage uses CLIP for cross-modal contrastive learning with text and images, enhancing interpretability and achieving refined classification through specially designed triplet textual relationships. Experimental results demonstrate that TCVADS significantly outperforms existing methods in model performance, detection efficiency, and interpretability, offering valuable contributions to smart city monitoring applications. 

**Abstract (ZH)**: 弱监督跨模态视频异常检测（WSMAD）利用弱监督学习来识别异常，这是智能城市监控中的一个关键任务。然而，现有的多模态方法由于其复杂性，往往无法满足边缘设备的实时性和可解释性要求。本文提出了两阶段跨模态视频异常检测系统（TCVADS），该系统结合知识蒸馏和跨模态对比学习，能够在边缘设备上实现高效的、准确的和可解释的异常检测。TCVADS 操作分为两个阶段：粗粒度快速分类和细粒度详细分析。在第一阶段，TCVADS 从视频帧中提取特征并将这些特征输入时间序列分析模块（教师模型）。然后，通过知识蒸馏将这些见解转移到简化卷积网络（学生模型）中进行二分类。在检测到异常后，触发第二阶段，使用细粒度的多分类模型。该阶段采用 CLIP 进行跨模态对比学习（结合文本和图像），并通过特别设计的三元组文本关系实现更精细的分类，从而提高可解释性。实验结果表明，TCVADS 在模型性能、检测效率和可解释性方面显著优于现有方法，为智能城市监控应用提供了有价值的贡献。 

---
# An archaeological Catalog Collection Method Based on Large Vision-Language Models 

**Title (ZH)**: 基于大型视觉-语言模型的考古藏品目录编制方法 

**Authors**: Honglin Pang, Yi Chang, Tianjing Duan, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20088)  

**Abstract**: Archaeological catalogs, containing key elements such as artifact images, morphological descriptions, and excavation information, are essential for studying artifact evolution and cultural inheritance. These data are widely scattered across publications, requiring automated collection methods. However, existing Large Vision-Language Models (VLMs) and their derivative data collection methods face challenges in accurate image detection and modal matching when processing archaeological catalogs, making automated collection difficult. To address these issues, we propose a novel archaeological catalog collection method based on Large Vision-Language Models that follows an approach comprising three modules: document localization, block comprehension and block matching. Through practical data collection from the Dabagou and Miaozigou pottery catalogs and comparison experiments, we demonstrate the effectiveness of our approach, providing a reliable solution for automated collection of archaeological catalogs. 

**Abstract (ZH)**: 考古目录包含器物图像、形态描述和发掘信息等关键要素，对于研究器物演进和文化传承至关重要。这些数据广泛分散在各种出版物上，需要自动化的数据收集方法。然而，现有的大型视觉-语言模型（VLMs）及其衍生的数据收集方法在处理考古目录时面临着精准图像检测和模态匹配的挑战，导致自动收集变得困难。为了解决这些问题，我们提出了一种基于大型视觉-语言模型的新型考古目录收集方法，该方法包含三个模块：文档定位、块理解与块匹配。通过对达包口和苗子沟陶器目录的实际数据收集和对比实验，我们证明了该方法的有效性，提供了一种可靠的方法来实现考古目录的自动化收集。 

---
# On the Compositional Generalization of Multimodal LLMs for Medical Imaging 

**Title (ZH)**: 多模态大语言模型在医学成像中的组成性泛化研究 

**Authors**: Zhenyang Cai, Junying Chen, Rongsheng Wang, Weihong Wang, Yonglin Deng, Dingjie Song, Yize Chen, Zixu Zhang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20070)  

**Abstract**: Multimodal large language models (MLLMs) hold significant potential in the medical field, but their capabilities are often limited by insufficient data in certain medical domains, highlighting the need for understanding what kinds of images can be used by MLLMs for generalization. Current research suggests that multi-task training outperforms single-task as different tasks can benefit each other, but they often overlook the internal relationships within these tasks, providing limited guidance on selecting datasets to enhance specific tasks. To analyze this phenomenon, we attempted to employ compositional generalization (CG)-the ability of models to understand novel combinations by recombining learned elements-as a guiding framework. Since medical images can be precisely defined by Modality, Anatomical area, and Task, naturally providing an environment for exploring CG. Therefore, we assembled 106 medical datasets to create Med-MAT for comprehensive experiments. The experiments confirmed that MLLMs can use CG to understand unseen medical images and identified CG as one of the main drivers of the generalization observed in multi-task training. Additionally, further studies demonstrated that CG effectively supports datasets with limited data and delivers consistent performance across different backbones, highlighting its versatility and broad applicability. Med-MAT is publicly available at this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在医疗领域具有巨大的潜力，但其能力往往受限于某些医疗领域的数据不足，突显出理解MLLMs可以用于泛化的类型是什么的重要性。现有研究表明，多任务训练优于单任务训练，因为不同任务之间可以互相受益，但它们往往忽略了这些任务之间的内部关系，对选择增强特定任务的数据集指导有限。为了分析这一现象，我们尝试采用组合泛化（CG）——模型通过重组学习元素来理解新组合的能力——作为指导框架。由于医学图像可以通过模态、解剖区域和任务精确定义，天然地为探索CG提供了环境。因此，我们收集了106个医学数据集，创建了Med-MAT进行全面实验。实验结果证实了MLLMs可以利用CG来理解未见过的医学图像，并将CG确定为多任务训练中观察到的泛化现象的主要驱动力之一。此外，进一步的研究表明，CG有效支持数据不足的数据集，并在不同的底层架构上表现出一致的性能，突显出其多样性和广泛的适用性。Med-MAT现已在该网址公开：[此链接地址]。 

---
# VELoRA: A Low-Rank Adaptation Approach for Efficient RGB-Event based Recognition 

**Title (ZH)**: VELoRA：一种高效的基于RGB-事件的识别的低秩适应方法 

**Authors**: Lan Chen, Haoxiang Yang, Pengpeng Shao, Haoyu Song, Xiao Wang, Zhicheng Zhao, Yaowei Wang, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.20064)  

**Abstract**: Pattern recognition leveraging both RGB and Event cameras can significantly enhance performance by deploying deep neural networks that utilize a fine-tuning strategy. Inspired by the successful application of large models, the introduction of such large models can also be considered to further enhance the performance of multi-modal tasks. However, fully fine-tuning these models leads to inefficiency and lightweight fine-tuning methods such as LoRA and Adapter have been proposed to achieve a better balance between efficiency and performance. To our knowledge, there is currently no work that has conducted parameter-efficient fine-tuning (PEFT) for RGB-Event recognition based on pre-trained foundation models. To address this issue, this paper proposes a novel PEFT strategy to adapt the pre-trained foundation vision models for the RGB-Event-based classification. Specifically, given the RGB frames and event streams, we extract the RGB and event features based on the vision foundation model ViT with a modality-specific LoRA tuning strategy. The frame difference of the dual modalities is also considered to capture the motion cues via the frame difference backbone network. These features are concatenated and fed into high-level Transformer layers for efficient multi-modal feature learning via modality-shared LoRA tuning. Finally, we concatenate these features and feed them into a classification head to achieve efficient fine-tuning. The source code and pre-trained models will be released on \url{this https URL}. 

**Abstract (ZH)**: 利用RGB和事件相机结合进行模式识别可以通过部署利用微调策略的深度神经网络显著提升性能。受大型模型成功应用的启发，引入这些大型模型也可以进一步提高多模态任务的性能。然而，对这些模型进行全面微调会导致效率下降，轻量化微调方法如LoRA和Adapter已被提出，以在效率与性能之间取得更好的平衡。据我们所知，目前尚无针对基于预训练基础模型的RGB-事件识别进行参数高效微调（PEFT）的研究工作。为了解决这一问题，本文提出了一种新的PEFT策略，以适应基于RGB-事件分类的基础视觉模型。具体而言，给定RGB帧和事件流，我们基于模态特定的LoRA调优策略从ViT基础视觉模型中提取RGB和事件特征。同时考虑双模态帧差，通过帧差主干网络捕捉运动线索。这些特征被连接并送入高层次的Transformer层，利用模态共享的LoRA调优策略进行高效的多模态特征学习。最后，我们将这些特征连接起来并送入分类头以实现高效的微调。源代码和预训练模型将在 \url{this https URL} 释放。 

---
# ProtCLIP: Function-Informed Protein Multi-Modal Learning 

**Title (ZH)**: ProtCLIP: 功能导向的蛋白质多模态学习 

**Authors**: Hanjing Zhou, Mingze Yin, Wei Wu, Mingyang Li, Kun Fu, Jintai Chen, Jian Wu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20014)  

**Abstract**: Multi-modality pre-training paradigm that aligns protein sequences and biological descriptions has learned general protein representations and achieved promising performance in various downstream applications. However, these works were still unable to replicate the extraordinary success of language-supervised visual foundation models due to the ineffective usage of aligned protein-text paired data and the lack of an effective function-informed pre-training paradigm. To address these issues, this paper curates a large-scale protein-text paired dataset called ProtAnno with a property-driven sampling strategy, and introduces a novel function-informed protein pre-training paradigm. Specifically, the sampling strategy determines selecting probability based on the sample confidence and property coverage, balancing the data quality and data quantity in face of large-scale noisy data. Furthermore, motivated by significance of the protein specific functional mechanism, the proposed paradigm explicitly model protein static and dynamic functional segments by two segment-wise pre-training objectives, injecting fine-grained information in a function-informed manner. Leveraging all these innovations, we develop ProtCLIP, a multi-modality foundation model that comprehensively represents function-aware protein embeddings. On 22 different protein benchmarks within 5 types, including protein functionality classification, mutation effect prediction, cross-modal transformation, semantic similarity inference and protein-protein interaction prediction, our ProtCLIP consistently achieves SOTA performance, with remarkable improvements of 75% on average in five cross-modal transformation benchmarks, 59.9% in GO-CC and 39.7% in GO-BP protein function prediction. The experimental results verify the extraordinary potential of ProtCLIP serving as the protein multi-modality foundation model. 

**Abstract (ZH)**: 多模态预训练范式已经在对蛋白质序列和生物描述进行对齐后学习到了通用的蛋白质表示，并在多种下游应用中取得了显著的表现。然而，这些工作仍无法复制语言监督视觉基础模型的卓越成功，原因在于缺乏有效利用对齐的蛋白质-文本配对数据的有效方法和缺乏功能导向的预训练范式。为了解决这些问题，本论文采用一种以属性驱动的采样策略构建了一个大规模的蛋白质-文本配对数据集ProtAnno，并引入了一种新的功能导向的蛋白质预训练范式。具体而言，采样策略根据样本置信度和属性覆盖度来确定采样概率，平衡大规模噪声数据的数据质量和数据量。此外，鉴于蛋白质特异性功能机制的重要性，所提出的范式通过两个段级预训练目标明确建模蛋白质的静态和动态功能段，以功能导向的方式注入细粒度信息。利用这些创新，我们开发了ProtCLIP，这是一种综合表示功能意识蛋白质嵌入的多模态基础模型。在包含5种类型的22个不同蛋白质基准中的各类测试（包括蛋白质功能分类、突变效应预测、跨模态变换、语义相似性推理和蛋白质-蛋白质相互作用预测），我们的ProtCLIP在所有测试中均表现出一致的领先性能，特别是在5个跨模态变换基准中的平均提升为75%，在GO-CC（细胞组件）蛋白质功能预测中的提升为59.9%，在GO-BP（分子功能）蛋白质功能预测中的提升为39.7%。实验结果验证了ProtCLIP作为蛋白质多模态基础模型的出色潜力。 

---
# ErgoChat: a Visual Query System for the Ergonomic Risk Assessment of Construction Workers 

**Title (ZH)**: ErgoChat：一种用于评估建筑工人职业风险的可视化查询系统 

**Authors**: Chao Fan, Qipei Mei, Xiaonan Wang, Xinming Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.19954)  

**Abstract**: In the construction sector, workers often endure prolonged periods of high-intensity physical work and prolonged use of tools, resulting in injuries and illnesses primarily linked to postural ergonomic risks, a longstanding predominant health concern. To mitigate these risks, researchers have applied various technological methods to identify the ergonomic risks that construction workers face. However, traditional ergonomic risk assessment (ERA) techniques do not offer interactive feedback. The rapidly developing vision-language models (VLMs), capable of generating textual descriptions or answering questions about ergonomic risks based on image inputs, have not yet received widespread attention. This research introduces an interactive visual query system tailored to assess the postural ergonomic risks of construction workers. The system's capabilities include visual question answering (VQA), which responds to visual queries regarding workers' exposure to postural ergonomic risks, and image captioning (IC), which generates textual descriptions of these risks from images. Additionally, this study proposes a dataset designed for training and testing such methodologies. Systematic testing indicates that the VQA functionality delivers an accuracy of 96.5%. Moreover, evaluations using nine metrics for IC and assessments from human experts indicate that the proposed approach surpasses the performance of a method using the same architecture trained solely on generic datasets. This study sets a new direction for future developments in interactive ERA using generative artificial intelligence (AI) technologies. 

**Abstract (ZH)**: 在建筑行业，工人经常需要进行长时间的高强度体力工作，并经常使用工具，这导致了与姿势相关的人体工学风险相关的伤害和疾病，这是长期存在的主要健康问题之一。为了减轻这些风险，研究人员已经应用了各种技术方法来识别建筑工人面临的工学风险。然而，传统的工学风险评估（Ergonomic Risk Assessment, ERA）技术并没有提供互动反馈。基于图像输入生成文本描述或回答与工学风险有关的问题的能力日益增强的视觉-语言模型（Vision-Language Models, VLMs）尚未引起广泛关注。本研究介绍了专门用于评估建筑工人姿势工学风险的互动视觉查询系统。该系统的功能包括视觉问答（Visual Question Answering, VQA），它可以针对工人暴露于姿势工学风险的视觉查询做出回答，以及图像生成（Image Captioning, IC），它可以生成表示这些风险的文本描述。此外，本研究还提出了一种用于训练和测试此类方法的数据集。系统测试表明，VQA功能的准确率为96.5%。此外，使用九个指标对IC进行评估以及来自人类专家的评估表明，所提出的方法在使用相同架构但仅在通用数据集上训练的方法中表现出更优异的性能。本研究为使用生成性人工智能（AI）技术进行互动工学风险评估的未来开发指明了新的方向。 

---
