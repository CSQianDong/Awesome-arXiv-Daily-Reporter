# Dynamic Knowledge Integration for Enhanced Vision-Language Reasoning 

**Title (ZH)**: 增强视觉语言推理的动态知识集成 

**Authors**: Julian Perry, Surasakdi Siripong, Thanakorn Phonchai  

**Link**: [PDF](https://arxiv.org/pdf/2501.08597)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated impressive capabilities in multimodal tasks, but their performance is often constrained by the lack of external knowledge integration, limiting their ability to handle knowledge-intensive tasks such as visual question answering and reasoning. To address this challenge, we propose a novel method, Adaptive Knowledge-Guided Pretraining for Large Vision-Language Models (AKGP-LVLM), which dynamically incorporates structured and unstructured knowledge into LVLMs during pretraining and fine-tuning. Our approach employs a knowledge encoder to represent external knowledge, a retrieval mechanism to select task-relevant information, and a dynamic adaptor to align multimodal and knowledge representations effectively. We evaluate our method on four benchmark datasets, demonstrating significant performance improvements over state-of-the-art models. Furthermore, human evaluations highlight the superior correctness and relevance of our model's outputs. Extensive analyses confirm the robustness, efficiency, and scalability of AKGP-LVLM, making it a compelling solution for real-world knowledge-intensive tasks. 

**Abstract (ZH)**: 大型多模态视觉-语言模型（Large Vision-Language Models, LVLMs）在多项任务中展现出了令人印象深刻的能力，但它们的性能往往受限于外部知识整合的不足，限制了其处理知识密集型任务（如视觉问答和推理）的能力。为了解决这一挑战，我们提出了一种新的方法，大型视觉-语言模型的自适应知识引导预训练（Adaptive Knowledge-Guided Pretraining for Large Vision-Language Models, AKGP-LVLM），该方法在预训练和微调过程中动态地将结构化和非结构化的知识整合到LVLMs中。我们的方法采用了知识编码器来表示外部知识，使用检索机制选择与任务相关的信息，并利用动态适配器有效地对齐多模态和知识表示。我们在四个基准数据集上评估了该方法，结果显示，在与最先进的模型相比时，我们的方法取得了显著的性能提升。此外，人类评估表明，我们的模型输出具有更高的准确性和相关性。大量分析证实了AKGP-LVLM的稳健性、高效性和扩展性，使其成为解决实际知识密集型任务的理想解决方案。 

---
# Multimodal LLMs Can Reason about Aesthetics in Zero-Shot 

**Title (ZH)**: 多模态大语言模型可以在零样本情况下进行审美推理 

**Authors**: Ruixiang Jiang, Changwen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09012)  

**Abstract**: We present the first study on how Multimodal LLMs' (MLLMs) reasoning ability shall be elicited to evaluate the aesthetics of artworks. To facilitate this investigation, we construct MM-StyleBench, a novel high-quality dataset for benchmarking artistic stylization. We then develop a principled method for human preference modeling and perform a systematic correlation analysis between MLLMs' responses and human preference. Our experiments reveal an inherent hallucination issue of MLLMs in art evaluation, associated with response subjectivity. ArtCoT is proposed, demonstrating that art-specific task decomposition and the use of concrete language boost MLLMs' reasoning ability for aesthetics. Our findings offer valuable insights into MLLMs for art and can benefit a wide range of downstream applications, such as style transfer and artistic image generation. Code available at this https URL. 

**Abstract (ZH)**: 我们首次探讨了如何通过Multimodal LLMs（多模态大语言模型）的推理能力评估艺术品的美学问题。为了便于这项研究，我们构建了MM-StyleBench，这是一个新颖的高质量基准数据集，用于评估艺术风格化。随后，我们开发了一种原则性的方法来建模人类偏好，并进行了MLLMs回答与人类偏好之间的系统相关性分析。我们的实验揭示了MLLMs在艺术评估中的固有幻觉问题，与响应的主观性有关。我们提出了ArtCoT，表明特定于艺术的任务分解和使用具体语言能够提升MLLMs在美学方面的推理能力。本研究为我们理解多模态大语言模型在艺术领域提供了宝贵的见解，并可以应用于多种下游应用，如风格迁移和艺术图像生成。代码见此处：[提供的链接]。 

---
# MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents 

**Title (ZH)**: MMDocIR：长文档多模态检索的基准测试 

**Authors**: Kuicai Dong, Yujing Chang, Xin Deik Goh, Dexun Li, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08828)  

**Abstract**: Multi-modal document retrieval is designed to identify and retrieve various forms of multi-modal content, such as figures, tables, charts, and layout information from extensive documents. Despite its significance, there is a notable lack of a robust benchmark to effectively evaluate the performance of systems in multi-modal document retrieval. To address this gap, this work introduces a new benchmark, named as MMDocIR, encompassing two distinct tasks: page-level and layout-level retrieval. The former focuses on localizing the most relevant pages within a long document, while the latter targets the detection of specific layouts, offering a more fine-grained granularity than whole-page analysis. A layout can refer to a variety of elements such as textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring expertly annotated labels for 1,685 questions and bootstrapped labels for 173,843 questions, making it a pivotal resource for advancing multi-modal document retrieval for both training and evaluation. Through rigorous experiments, we reveal that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR train set can effectively benefit the training process of multi-modal document retrieval and (iii) text retrievers leveraging on VLM-text perform much better than those using OCR-text. These findings underscores the potential advantages of integrating visual elements for multi-modal document retrieval. 

**Abstract (ZH)**: 多模态文档检索旨在识别和检索各种形式的多模态内容，如图表、表格、图表以及布局信息，从中广泛文档。尽管多模态文档检索具有重要意义，但目前缺乏一个坚固的基准来有效地评估系统在多模态文档检索中的性能。为弥补这一不足，本文介绍了一个新的基准，命名为MMDocIR，涵盖两个不同的任务：页面级检索和布局级检索。前者侧重于在长文档中定位最相关的页面，而后者旨在检测特定布局，提供比整页分析更精细的粒度。布局可以是指各种元素，如文本段落、公式、图表、表格或图表。MMDocIR基准数据集包含专家注释的1,685个问题和通过自举生成的173,843个问题的标签，成为促进多模态文档检索训练和评估的重要资源。

通过严格的实验，我们揭示了以下几点发现：(i) 视觉检索器显著优于文本检索器；(ii) MMDocIR的训练集能够有效提高多模态文档检索的训练过程；(iii) 利用VLM-文本的文本检索器比使用OCR-文本的检索器表现更好。这些发现进一步强调了在多模态文档检索中整合视觉元素的潜在优势。 

---
# Visual WetlandBirds Dataset: Bird Species Identification and Behavior Recognition in Videos 

**Title (ZH)**: 视觉湿地鸟类数据集：视频中鸟类物种识别与行为识别 

**Authors**: Javier Rodriguez-Juan, David Ortiz-Perez, Manuel Benavent-Lledo, David Mulero-Pérez, Pablo Ruiz-Ponce, Adrian Orihuela-Torres, Jose Garcia-Rodriguez, Esther Sebastián-González  

**Link**: [PDF](https://arxiv.org/pdf/2501.08931)  

**Abstract**: The current biodiversity loss crisis makes animal monitoring a relevant field of study. In light of this, data collected through monitoring can provide essential insights, and information for decision-making aimed at preserving global biodiversity. Despite the importance of such data, there is a notable scarcity of datasets featuring videos of birds, and none of the existing datasets offer detailed annotations of bird behaviors in video format. In response to this gap, our study introduces the first fine-grained video dataset specifically designed for bird behavior detection and species classification. This dataset addresses the need for comprehensive bird video datasets and provides detailed data on bird actions, facilitating the development of deep learning models to recognize these, similar to the advancements made in human action recognition. The proposed dataset comprises 178 videos recorded in Spanish wetlands, capturing 13 different bird species performing 7 distinct behavior classes. In addition, we also present baseline results using state of the art models on two tasks: bird behavior recognition and species classification. 

**Abstract (ZH)**: 当前生物多样性丧失危机使动物监测成为一个重要的研究领域。鉴于此，通过监测收集的数据可以提供至关重要的见解和信息，支持旨在保护全球生物多样性的决策。尽管这些数据的重要性不言而喻，但现有的鸟视频数据集极为稀缺，且没有任何现有的数据集以视频格式提供详细的鸟类行为注释。为应对这一空白，本研究首次引入了一个专门用于鸟类行为检测和物种分类的细粒度视频数据集。该数据集满足了全面的鸟视频数据集的需求，并提供了详细的鸟类动作数据，促进了类似于人类行为识别领域所取得的进展的深度学习模型的发展。所提数据集包含178段在西班牙湿地记录的视频，捕捉了13种不同的鸟类执行的7类不同行为。此外，我们还呈现了在两个任务（鸟行为识别和物种分类）上使用最先进的模型的基准结果。 

---
# IDEA: Image Description Enhanced CLIP-Adapter 

**Title (ZH)**: IDEA：图像描述增强的CLIP-适配器 

**Authors**: Zhipeng Ye, Feng Jiang, Qiufeng Wang, Kaizhu Huang, Jiaqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.08816)  

**Abstract**: CLIP (Contrastive Language-Image Pre-training) has attained great success in pattern recognition and computer vision. Transferring CLIP to downstream tasks (e.g. zero- or few-shot classification) is a hot topic in multimodal learning. However, current studies primarily focus on either prompt learning for text or adapter tuning for vision, without fully exploiting the complementary information and correlations among image-text pairs. In this paper, we propose an Image Description Enhanced CLIP-Adapter (IDEA) method to adapt CLIP to few-shot image classification tasks. This method captures fine-grained features by leveraging both visual features and textual descriptions of images. IDEA is a training-free method for CLIP, and it can be comparable to or even exceeds state-of-the-art models on multiple tasks. Furthermore, we introduce Trainable-IDEA (T-IDEA), which extends IDEA by adding two lightweight learnable components (i.e., a projector and a learnable latent space), further enhancing the model's performance and achieving SOTA results on 11 datasets. As one important contribution, we employ the Llama model and design a comprehensive pipeline to generate textual descriptions for images of 11 datasets, resulting in a total of 1,637,795 image-text pairs, named "IMD-11". Our code and data are released at this https URL. 

**Abstract (ZH)**: CLIP（对比语言-图像预训练）在模式识别和计算机视觉领域取得了巨大成功。将CLIP应用于下游任务（例如零样本或少样本分类）是多模态学习领域的热点话题。然而，当前的研究主要集中在文本的提示学习或视觉的适配调优上，未能充分利用图像-文本对之间的互补信息和关联性。本文提出了一种图像描述增强CLIP适配器（IDEA）方法，以使CLIP适应少样本图像分类任务。该方法通过利用图像的视觉特征和文本描述来捕捉细微特征。IDEA是一种不需要训练的方法，可在多个任务中与最新模型相媲美，甚至优于最新模型。此外，我们引入了可训练IDEA（T-IDEA），该方法通过添加两个轻量级可学习组件（即投影器和可学习的隐空间），进一步提升了模型的性能，并在11个数据集上实现了SOTA结果。作为一项重要贡献，我们使用Llama模型并设计了一个综合管道来生成11个数据集图像的文本描述，共生成了1,637,795个图像-文本对，命名为“IMD-11”。我们的代码和数据已在此处发布：[相关链接]。 

---
# A Systematic Review of Machine Learning Methods for Multimodal EEG Data in Clinical Application 

**Title (ZH)**: 多模态脑电图数据临床应用中机器学习方法的系统综述 

**Authors**: Siqi Zhao, Wangyang Li, Xiru Wang, Stevie Foglia, Hongzhao Tan, Bohan Zhang, Ameer Hamoodi, Aimee Nelson, Zhen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.08585)  

**Abstract**: Machine learning (ML) and deep learning (DL) techniques have been widely applied to analyze electroencephalography (EEG) signals for disease diagnosis and brain-computer interfaces (BCI). The integration of multimodal data has been shown to enhance the accuracy of ML and DL models. Combining EEG with other modalities can improve clinical decision-making by addressing complex tasks in clinical populations. This systematic literature review explores the use of multimodal EEG data in ML and DL models for clinical applications. A comprehensive search was conducted across PubMed, Web of Science, and Google Scholar, yielding 16 relevant studies after three rounds of filtering. These studies demonstrate the application of multimodal EEG data in addressing clinical challenges, including neuropsychiatric disorders, neurological conditions (e.g., seizure detection), neurodevelopmental disorders (e.g., autism spectrum disorder), and sleep stage classification. Data fusion occurred at three levels: signal, feature, and decision levels. The most commonly used ML models were support vector machines (SVM) and decision trees. Notably, 11 out of the 16 studies reported improvements in model accuracy with multimodal EEG data. This review highlights the potential of multimodal EEG-based ML models in enhancing clinical diagnostics and problem-solving. 

**Abstract (ZH)**: 机器学习（ML）和深度学习（DL）技术已在分析脑电图（EEG）信号以进行疾病诊断和脑-机接口（BCI）方面得到了广泛应用。多模态数据的整合已被证明能够提高ML和DL模型的准确性。将EEG与其他模态数据结合使用可以改进临床决策，通过解决临床人群中的复杂任务。本系统文献综述探讨了多模态EEG数据在临床应用中的ML和DL模型的使用情况。我们通过PubMed、Web of Science和Google Scholar进行了全面搜索，在三轮筛选后共获得了16篇相关研究。这些研究展示了多模态EEG数据在应对临床挑战中的应用，包括神经精神障碍、神经系统疾病（如癫痫检测）、神经发育障碍（如自闭症谱系障碍）以及睡眠阶段分类。数据融合发生在三个层次上：信号层、特征层和决策层。常用的ML模型主要是支持向量机（SVM）和决策树。值得注意的是，在16项研究中有11项报告表示，通过使用多模态EEG数据提高了模型准确性。本综述突显了基于多模态EEG的ML模型在增强临床诊断和问题解决方面的潜力。 

---
# The Devil is in Temporal Token: High Quality Video Reasoning Segmentation 

**Title (ZH)**: 时间_token中的魔鬼：高质量视频推理分割 

**Authors**: Sitong Gong, Yunzhi Zhuge, Lu Zhang, Zongxin Yang, Pingping Zhang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08549)  

**Abstract**: Existing methods for Video Reasoning Segmentation rely heavily on a single special token to represent the object in the keyframe or the entire video, inadequately capturing spatial complexity and inter-frame motion. To overcome these challenges, we propose VRS-HQ, an end-to-end video reasoning segmentation approach that leverages Multimodal Large Language Models (MLLMs) to inject rich spatiotemporal features into hierarchical this http URL key innovations include a Temporal Dynamic Aggregation (TDA) and a Token-driven Keyframe Selection (TKS). Specifically, we design frame-level <SEG> and temporal-level <TAK> tokens that utilize MLLM's autoregressive learning to effectively capture both local and global information. Subsequently, we apply a similarity-based weighted fusion and frame selection strategy, then utilize SAM2 to perform keyframe segmentation and propagation. To enhance keyframe localization accuracy, the TKS filters keyframes based on SAM2's occlusion scores during inference. VRS-HQ achieves state-of-the-art performance on ReVOS, surpassing VISA by 5.9%/12.5%/9.1% in J&F scores across the three subsets. These results highlight the strong temporal reasoning and segmentation capabilities of our method. Code and model weights will be released at VRS-HQ. 

**Abstract (ZH)**: 现有的视频推理分割方法强烈依赖于单一特殊的token来表示关键帧或整个视频中的对象，未能充分捕捉空间复杂性和帧间运动。为克服这些挑战，我们提出了一种名为VRS-HQ的端到端视频推理分割方法，该方法利用多模态大型语言模型（MLLMs）将丰富的时空特征注入层次结构中。该方法的两大创新分别是时间动态聚合（TDA）和基于token的关键帧选择（TKS）。具体而言，我们设计了帧级<sEG>和时间级<sTAK> token，利用MLLM的自回归学习机制，有效捕捉局部和全局信息。之后，我们应用基于相似性的加权融合和帧选择策略，并利用SAM2执行关键帧分割和传播。为了提高关键帧定位的准确性，TKS在推理过程中根据SAM2的遮挡分数来筛选关键帧。VRS-HQ在ReVOS数据集上取得了最先进的性能，分别在三个子集的J&F分数上超越VISA 5.9%/12.5%/9.1%。这些结果突显了我们方法的强健时间推理和分割能力。VRS-HQ的相关代码和预训练模型权重将在未来公开。 

---
# Cross-Modal Transferable Image-to-Video Attack on Video Quality Metrics 

**Title (ZH)**: 跨模态可转移的图像到视频攻击对视频质量度量的影响 

**Authors**: Georgii Gotin, Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin  

**Link**: [PDF](https://arxiv.org/pdf/2501.08415)  

**Abstract**: Recent studies have revealed that modern image and video quality assessment (IQA/VQA) metrics are vulnerable to adversarial attacks. An attacker can manipulate a video through preprocessing to artificially increase its quality score according to a certain metric, despite no actual improvement in visual quality. Most of the attacks studied in the literature are white-box attacks, while black-box attacks in the context of VQA have received less attention. Moreover, some research indicates a lack of transferability of adversarial examples generated for one model to another when applied to VQA. In this paper, we propose a cross-modal attack method, IC2VQA, aimed at exploring the vulnerabilities of modern VQA models. This approach is motivated by the observation that the low-level feature spaces of images and videos are similar. We investigate the transferability of adversarial perturbations across different modalities; specifically, we analyze how adversarial perturbations generated on a white-box IQA model with an additional CLIP module can effectively target a VQA model. The addition of the CLIP module serves as a valuable aid in increasing transferability, as the CLIP model is known for its effective capture of low-level semantics. Extensive experiments demonstrate that IC2VQA achieves a high success rate in attacking three black-box VQA models. We compare our method with existing black-box attack strategies, highlighting its superiority in terms of attack success within the same number of iterations and levels of attack strength. We believe that the proposed method will contribute to the deeper analysis of robust VQA metrics. 

**Abstract (ZH)**: 近年来的研究表明，现代图像和视频质量评估（IQA/VQA）指标容易受到对抗性攻击的影响。攻击者可以通过预处理手段人为地提升视频的评分，尽管视觉质量并没有实际改进。文献中大多数研究关注的是白盒攻击，而视频质量评估（VQA）领域的黑盒攻击则受到了较少的关注。此外，一些研究指出，针对一种模型生成的对抗性样本在应用于VQA时并不具备很好的迁移性。在本文中，我们提出了一种跨模态攻击方法IC2VQA，旨在探索现代VQA模型的脆弱性。这一方法受到图像和视频的低层特征空间相似性的观察启发。我们研究了不同模态之间对抗性扰动的迁移性；具体来说，我们分析了在带有附加CLIP模块的白盒IQA模型上生成的对抗性扰动如何有效针对VQA模型。CLIP模块的添加有助于提高迁移性，因为该模型擅长捕捉低层语义。广泛的实验表明，IC2VQA方法在攻击三个黑盒VQA模型时具有较高的成功率。我们将我们的方法与现有的黑盒攻击策略进行了比较，突出了其在相同迭代次数和攻击强度下的优越性。我们认为，所提出的方法将有助于更深入地分析鲁棒的VQA指标。 

---
