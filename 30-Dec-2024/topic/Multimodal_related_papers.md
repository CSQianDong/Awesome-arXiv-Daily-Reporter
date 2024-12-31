# Reversed in Time: A Novel Temporal-Emphasized Benchmark for Cross-Modal Video-Text Retrieval 

**Title (ZH)**: 逆向时间：一种新的强调时间维度的跨模态视频-文本检索基准 

**Authors**: Yang Du, Yuqi Liu, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2412.19178)  

**Abstract**: Cross-modal (e.g. image-text, video-text) retrieval is an important task in information retrieval and multimodal vision-language understanding field. Temporal understanding makes video-text retrieval more challenging than image-text retrieval. However, we find that the widely used video-text benchmarks have shortcomings in comprehensively assessing abilities of models, especially in temporal understanding, causing large-scale image-text pre-trained models can already achieve comparable zero-shot performance with video-text pre-trained models. In this paper, we introduce RTime, a novel temporal-emphasized video-text retrieval dataset. We first obtain videos of actions or events with significant temporality, and then reverse these videos to create harder negative samples. We then recruit annotators to judge the significance and reversibility of candidate videos, and write captions for qualified videos. We further adopt GPT-4 to extend more captions based on human-written captions. Our RTime dataset currently consists of 21k videos with 10 captions per video, totalling about 122 hours. Based on RTime, we propose three retrieval benchmark tasks: RTime-Origin, RTime-Hard, and RTime-Binary. We further enhance the use of harder-negatives in model training, and benchmark a variety of video-text models on RTime. Extensive experiment analysis proves that RTime indeed poses new and higher challenges to video-text retrieval. We release our RTime dataset\footnote{\url{this https URL}} to further advance video-text retrieval and multimodal understanding research. 

**Abstract (ZH)**: 跨模态检索（例如图像文本、视频文本）是信息检索和多模态视觉语言理解领域中的一个重要任务。时间理解使得视频文本检索相较于图像文本检索更具挑战性。然而，我们发现广泛使用的视频文本基准数据集在全面评估模型能力方面存在不足，特别是在时间理解方面，导致大规模的图像文本预训练模型已经能够与视频文本预训练模型实现可比拟的零样本性能。在本文中，我们引入了RTime，这是一种新的强调时间的视频文本检索数据集。我们首先获取具有显著时间性的动作或事件视频，然后反向这些视频以创建更难的负样本。接下来，我们招募标注员来判断候选视频的重要性及其可逆性，并为合格的视频编写描述性文字。我们进一步采用GPT-4基于人类编写的描述性文字扩展更多描述性文字。目前，我们的RTime数据集包含21,000个视频，每个视频有10个描述，总计大约122小时。基于RTime，我们提出了三个检索基准任务：RTime-Origin、RTime-Hard和RTime-Binary。我们进一步增强了模型训练中使用更具挑战性的负样本，并在RTime上对各种视频文本模型进行了基准测试。广泛的实验分析证明，RTime确实为视频文本检索提出了新的和更高的挑战。我们发布了RTime数据集\footnote{\url{https://}}以进一步推动视频文本检索和多模态理解的研究。 

---
# "Did my figure do justice to the answer?" : Towards Multimodal Short Answer Grading with Feedback (MMSAF) 

**Title (ZH)**: “我的图表是否恰当地反映了答案？”：迈向基于反馈的多模态简答题评分（MMSAF） 

**Authors**: Pritam Sil, Bhaskaran Raman, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2412.19755)  

**Abstract**: Personalized feedback plays a vital role in a student's learning process. While existing systems are adept at providing feedback over MCQ-based evaluation, this work focuses more on subjective and open-ended questions, which is similar to the problem of Automatic Short Answer Grading (ASAG) with feedback. Additionally, we introduce the Multimodal Short Answer grading with Feedback (MMSAF) problem over the traditional ASAG feedback problem to address the scenario where the student answer and reference answer might contain images. Moreover, we introduce the MMSAF dataset with 2197 data points along with an automated framework for generating such data sets. Our evaluations on existing LLMs over this dataset achieved an overall accuracy of 55\% on Level of Correctness labels, 75\% on Image Relevance labels and a score of 4.27 out of 5 in correctness level of LLM generated feedback as rated by experts. As per experts, Pixtral achieved a rating of above 4 out of all metrics, indicating that it is more aligned to human judgement, and that it is the best solution for assisting students. 

**Abstract (ZH)**: 个性化反馈在学生学习过程中起着至关重要的作用。虽然现有的系统在提供基于选择题的评估反馈方面表现出色，本研究更多地关注主观和开放性问题，这与自动简短答案评分（ASAG）及其反馈问题类似。此外，我们引入了多模态简短答案评分与反馈（MMSAF）问题，扩大了传统的ASAG反馈问题，以应对学生答案和参考答案可能包含图像的情况。我们还介绍了包含2197个数据点的MMSAF数据集，并提供了一个生成此类数据集的自动化框架。在该数据集上对现有LLM进行评估，结果显示在正确性标签上总体准确率为55%，在图像相关性标签上为75%，LLM生成的反馈在专家评估中的正确性得分为4.27/5。根据专家的评价，Pixtral在所有指标上得分超过4分，表明其更符合人的判断，是辅助学生学习的最佳解决方案。 

---
# A Large-scale Interpretable Multi-modality Benchmark for Facial Image Forgery Localization 

**Title (ZH)**: 大规模可解释多模态基准数据集用于面部图像伪造定位 

**Authors**: Jingchun Lian, Lingyu Liu, Yaxiong Wang, Yujiao Wu, Li Zhu, Zhedong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.19685)  

**Abstract**: Image forgery localization, which centers on identifying tampered pixels within an image, has seen significant advancements. Traditional approaches often model this challenge as a variant of image segmentation, treating the binary segmentation of forged areas as the end product. We argue that the basic binary forgery mask is inadequate for explaining model predictions. It doesn't clarify why the model pinpoints certain areas and treats all forged pixels the same, making it hard to spot the most fake-looking parts. In this study, we mitigate the aforementioned limitations by generating salient region-focused interpretation for the forgery images. To support this, we craft a Multi-Modal Tramper Tracing (MMTT) dataset, comprising facial images manipulated using deepfake techniques and paired with manual, interpretable textual annotations. To harvest high-quality annotation, annotators are instructed to meticulously observe the manipulated images and articulate the typical characteristics of the forgery regions. Subsequently, we collect a dataset of 128,303 image-text pairs. Leveraging the MMTT dataset, we develop ForgeryTalker, an architecture designed for concurrent forgery localization and interpretation. ForgeryTalker first trains a forgery prompter network to identify the pivotal clues within the explanatory text. Subsequently, the region prompter is incorporated into multimodal large language model for finetuning to achieve the dual goals of localization and interpretation. Extensive experiments conducted on the MMTT dataset verify the superior performance of our proposed model. The dataset, code as well as pretrained checkpoints will be made publicly available to facilitate further research and ensure the reproducibility of our results. 

**Abstract (ZH)**: 图像篡改定位是集中于识别图像中篡改像素的技术，近年来取得了显著的进步。传统的做法通常将这一挑战建模为图像分割的变体，将伪造区域的二值分割视为最终产品。然而，我们认为基本的二值篡改掩码无法解释模型的预测结果。它未能明确说明模型为何识别某些区域，且将所有篡改的像素同等对待，使得难以识别最显眼的伪造部分。在本研究中，我们通过为篡改图像生成突出区域的解释来缓解上述限制。为此，我们构建了一个多模态篡改追踪（MMTT）数据集，包括使用深度伪造技术篡改的面部图像，并配有可解释的手动注释文本。为了获得高质量的注释，我们指导标注员仔细观察篡改图像，并描述伪造区域的典型特征。随后，我们收集了一个包含128,303个图像-文本对的数据集。借助MMTT数据集，我们开发了ForgeryTalker架构，该架构旨在同时实现篡改定位和解释。首先，ForgeryTalker训练一个篡改提示网络以识别说明性文本中的关键线索。然后，该区域提示器被整合到多模态大型语言模型中以进行微调，从而实现定位和解释的双重目标。我们在MMTT数据集上的大量实验验证了我们的模型具有优越的性能。此外，我们将数据集、代码以及预训练模型提供给公众，以促进进一步的研究并确保研究结果的可重复性。 

---
# CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs 

**Title (ZH)**: CAD-GPT：增强空间推理的多模态LLM生成建筑施工序列 

**Authors**: Siyu Wang, Cailian Chen, Xinyi Le, Qimin Xu, Lei Xu, Yanzhou Zhang, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19663)  

**Abstract**: Computer-aided design (CAD) significantly enhances the efficiency, accuracy, and innovation of design processes by enabling precise 2D and 3D modeling, extensive analysis, and optimization. Existing methods for creating CAD models rely on latent vectors or point clouds, which are difficult to obtain and costly to store. Recent advances in Multimodal Large Language Models (MLLMs) have inspired researchers to use natural language instructions and images for CAD model construction. However, these models still struggle with inferring accurate 3D spatial location and orientation, leading to inaccuracies in determining the spatial 3D starting points and extrusion directions for constructing geometries. This work introduces CAD-GPT, a CAD synthesis method with spatial reasoning-enhanced MLLM that takes either a single image or a textual description as input. To achieve precise spatial inference, our approach introduces a 3D Modeling Spatial Mechanism. This method maps 3D spatial positions and 3D sketch plane rotation angles into a 1D linguistic feature space using a specialized spatial unfolding mechanism, while discretizing 2D sketch coordinates into an appropriate planar space to enable precise determination of spatial starting position, sketch orientation, and 2D sketch coordinate translations. Extensive experiments demonstrate that CAD-GPT consistently outperforms existing state-of-the-art methods in CAD model synthesis, both quantitatively and qualitatively. 

**Abstract (ZH)**: 计算机辅助设计（CAD）显著提升了设计过程的效率、准确性和创新性，通过实现精确的二维和三维建模、广泛分析和优化。现有的CAD模型创建方法依赖于潜在向量或点云，这些方法的获取过程复杂且储存成本高昂。最近，多模态大型语言模型（MLLMs）的进步激发了研究人员使用自然语言指令和图像进行CAD模型构建。然而，这些模型仍然难以准确推断三维空间位置和方向，导致在构建几何图形时难以精确确定三维空间的起始点和拉伸方向。本项研究引入了CAD-GPT，这是一种具有增强空间推理的MLLM的CAD合成方法，可以接受单张图像或文本描述作为输入。为了实现精确的空间推断，我们的方法引入了三维建模空间机制。该方法利用专门的空间展开机制将三维空间位置和三维素描面旋转角度映射到一维语言特征空间，同时将二维素描坐标离散化到合适的平面空间，以实现对空间起始位置、素描方向和二维素描坐标转换的精确确定。大量实验结果显示，CAD-GPT在CAD模型合成方面的性能在定性和定量上都显著优于现有最先进的方法。 

---
# MBQ: Modality-Balanced Quantization for Large Vision-Language Models 

**Title (ZH)**: MBQ: 配平模态量化大型视觉-语言模型 

**Authors**: Shiyao Li, Yingchun Hu, Xuefei Ning, Xihui Liu, Ke Hong, Xiaotao Jia, Xiuhong Li, Yaqi Yan, Pei Ran, Guohao Dai, Shengen Yan, Huazhong Yang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19509)  

**Abstract**: Vision-Language Models (VLMs) have enabled a variety of real-world applications. The large parameter size of VLMs brings large memory and computation overhead which poses significant challenges for deployment. Post-Training Quantization (PTQ) is an effective technique to reduce the memory and computation overhead. Existing PTQ methods mainly focus on large language models (LLMs), without considering the differences across other modalities. In this paper, we discover that there is a significant difference in sensitivity between language and vision tokens in large VLMs. Therefore, treating tokens from different modalities equally, as in existing PTQ methods, may over-emphasize the insensitive modalities, leading to significant accuracy loss. To deal with the above issue, we propose a simple yet effective method, Modality-Balanced Quantization (MBQ), for large VLMs. Specifically, MBQ incorporates the different sensitivities across modalities during the calibration process to minimize the reconstruction loss for better quantization parameters. Extensive experiments show that MBQ can significantly improve task accuracy by up to 4.4% and 11.6% under W3 and W4A8 quantization for 7B to 70B VLMs, compared to SOTA baselines. Additionally, we implement a W3 GPU kernel that fuses the dequantization and GEMV operators, achieving a 1.4x speedup on LLaVA-onevision-7B on the RTX 4090. The code is available at this https URL. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）使多种现实应用场景成为可能。VLMs 的大参数量带来了巨大的内存和计算成本，这对部署提出了重大挑战。训练后量化（Post-Training Quantization, PTQ）是一种有效的方法，用于减少这些成本。目前的 PTQ 方法主要集中在大型语言模型（LLMs）上，而忽视了其他模态之间的差异。在本文中，我们发现大型 VLMs 中语言和视觉标记在敏感度方面存在显著差异。因此，现有的 PTQ 方法在处理不同模态的标记时采用相同的处理方式，可能会过度强调不敏感的模态，从而导致显著的准确性损失。为解决上述问题，我们提出了一种简单而有效的方法——模态平衡量化（Modality-Balanced Quantization, MBQ），专门用于大型 VLMs。具体而言，MBQ 在校准过程中考虑不同模态之间的敏感度差异，以最小化重建损失，从而获得更好的量化参数。广泛实验表明，MBQ 与最先进的基线方法相比，在 W3 和 W4A8 量化下，对于 7B 至 70B 的 VLMs，可以显著提高任务准确性高达 4.4% 和 11.6%。此外，我们实现了一个 W3 GPU 内核，将去量化的操作和 GEMV（通用矩阵-向量乘）操作融合在一起，在 RTX 4090 上实现了 LLaVA-onevision-7B 的 1.4 倍加速。代码可从以下链接获取：this https URL。 

---
# AskChart: Universal Chart Understanding through Textual Enhancement 

**Title (ZH)**: AskChart：通过文本增强实现通用图表理解 

**Authors**: Xudong Yang, Yifan Wu, Yizhang Zhu, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2412.19146)  

**Abstract**: Chart understanding tasks such as ChartQA and Chart-to-Text involve automatically extracting and interpreting key information from charts, enabling users to query or convert visual data into structured formats. State-of-the-art approaches primarily focus on visual cues from chart images, failing to explicitly incorporate rich textual information (e.g., data labels and axis labels) embedded within the charts. This textual information is vital for intuitive human comprehension and interpretation of charts. Moreover, existing models are often large and computationally intensive, limiting their practical applicability. In this paper, we introduce AskChart, a universal model that explicitly integrates both textual and visual cues from charts using a Mixture of Experts (MoE) architecture. AskChart facilitates the learning of enhanced visual-textual representations of charts for effectively handling multiple chart understanding tasks, while maintaining a smaller model size. To capture the synergy between visual and textual modalities, we curate a large-scale dataset named ChartBank with about 7.5M data samples, which helps align textual and visual information and facilitates the extraction of visual entities and text. To effectively train AskChart, we design a three-stage training strategy to align visual and textual modalities for learning robust visual-textual representations and optimizing the learning of the MoE layer. Extensive experiments across five datasets demonstrate the significant performance gains of AskChart in four chart understanding tasks. Remarkably, AskChart with 4.6B parameters outperforms state-of-the-art models with 13B parameters by 68.3% in Open-ended ChartQA and 49.2% in Chart-to-Text tasks, while achieving comparable performance in ChartQA and Chart-to-Table tasks. 

**Abstract (ZH)**: 如下是根据学术规范的翻译：

图表理解任务（如ChartQA和Chart-to-Text）涉及从图表中自动提取并解释关键信息，让用户能够查询或转换图表中的视觉数据为结构化格式。最先进的方法主要关注图表图像的视觉线索，而未能明确结合嵌入在图表中的丰富文本信息（如数据标签和轴标签）。这些文本信息对于直观的人类理解和解释图表至关重要。此外，现有的模型通常庞大且计算密集，限制了它们的实际应用。在本文中，我们引入了AskChart，这是一种利用专家混合模型（MoE架构）明确结合图表中的文本和视觉线索的通用模型。AskChart通过学习增强的视觉-文本表示，能够有效地处理多种图表理解任务，同时保持较小的模型尺寸。为了捕捉视觉模态和文本模态之间的协同作用，我们构建了一个包含约750万个数据样本的大规模数据集ChartBank，它有助于对齐文本和视觉信息，并促进视觉实体和文本的提取。为了有效训练AskChart，我们设计了一个三阶段训练策略，以对齐视觉和文本模态，学习 robust 的视觉-文本表示，并优化 MoE 层的学习。在五个数据集上进行的广泛实验表明，AskChart 在四种图表理解任务中的表现明显优于最先进的模型。特别地，具有46亿参数的AskChart在开放式ChartQA任务中比具有130亿参数的最先进的模型提高了68.3%，在Chart-to-Text任务中提高了49.2%，同时在ChartQA和Chart-to-Table任务中实现了相当好的性能。 

---
# PlanLLM: Video Procedure Planning with Refinable Large Language Models 

**Title (ZH)**: PlanLLM：具有可细化大型语言模型的视频程序规划 

**Authors**: Dejie Yang, Zijing Zhao, YangLiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19139)  

**Abstract**: Video procedure planning, i.e., planning a sequence of action steps given the video frames of start and goal states, is an essential ability for embodied AI. Recent works utilize Large Language Models (LLMs) to generate enriched action step description texts to guide action step decoding. Although LLMs are introduced, these methods decode the action steps into a closed-set of one-hot vectors, limiting the model's capability of generalizing to new steps or tasks. Additionally, fixed action step descriptions based on world-level commonsense may contain noise in specific instances of visual states. In this paper, we propose PlanLLM, a cross-modal joint learning framework with LLMs for video procedure planning. We propose an LLM-Enhanced Planning module which fully uses the generalization ability of LLMs to produce free-form planning output and to enhance action step decoding. We also propose Mutual Information Maximization module to connect world-level commonsense of step descriptions and sample-specific information of visual states, enabling LLMs to employ the reasoning ability to generate step sequences. With the assistance of LLMs, our method can both closed-set and open vocabulary procedure planning tasks. Our PlanLLM achieves superior performance on three benchmarks, demonstrating the effectiveness of our designs. 

**Abstract (ZH)**: 视频操作规划，即根据起始状态和目标状态的视频帧规划一系列操作步骤，是具身人工智能的一项基本能力。最近的研究利用大型语言模型（LLMs）生成丰富的操作步骤描述文本，以指导操作步骤解码。尽管引入了LLMs，但这些方法将操作步骤解码为封闭集合中的一个热向量，限制了模型泛化到新步骤或任务的能力。此外，基于世界级常识固定的操作步骤描述在特定的视觉状态示例中可能包含噪声。在本文中，我们提出了一种名为PlanLLM的跨模态联合学习框架，该框架利用LLMs进行视频操作规划。我们提出了一种增强的规划模块，该模块充分利用了LLMs的泛化能力，生成自由形式的规划输出，并增强操作步骤解码。我们还提出了信息互信息最大化模块，将步骤描述的世界级常识与视觉状态的特定样本信息连接起来，使LLMs能够利用推理能力生成步骤序列。借助LLMs的帮助，我们的方法可以同时完成封闭集和开放词汇的操作规划任务。我们的PlanLLM在三个基准测试中表现出色，证明了我们设计的有效性。 

---
# A Rhetorical Relations-Based Framework for Tailored Multimedia Document Summarization 

**Title (ZH)**: 基于修辞关系的定制化多媒体文档摘要框架 

**Authors**: Azze-Eddine Maredj, Madjid Sadallah  

**Link**: [PDF](https://arxiv.org/pdf/2412.19133)  

**Abstract**: In the rapidly evolving landscape of digital content, the task of summarizing multimedia documents, which encompass textual, visual, and auditory elements, presents intricate challenges. These challenges include extracting pertinent information from diverse formats, maintaining the structural integrity and semantic coherence of the original content, and generating concise yet informative summaries. This paper introduces a novel framework for multimedia document summarization that capitalizes on the inherent structure of the document to craft coherent and succinct summaries. Central to this framework is the incorporation of a rhetorical structure for structural analysis, augmented by a graph-based representation to facilitate the extraction of pivotal information. Weighting algorithms are employed to assign significance values to document units, thereby enabling effective ranking and selection of relevant content. Furthermore, the framework is designed to accommodate user preferences and time constraints, ensuring the production of personalized and contextually relevant summaries. The summarization process is elaborately delineated, encompassing document specification, graph construction, unit weighting, and summary extraction, supported by illustrative examples and algorithmic elucidation. This proposed framework represents a significant advancement in automatic summarization, with broad potential applications across multimedia document processing, promising transformative impacts in the field. 

**Abstract (ZH)**: 在数字化内容迅速演进的背景下，多媒体文档摘要的任务——涵盖文本、视觉和听觉等多种元素——带来了复杂的挑战。这些挑战包括从不同格式中提取相关信息，保持原始内容的结构完整性和语义连贯性，并生成简明而富有信息性的摘要。本文提出了一种新颖的多媒体文档摘要框架，该框架利用文档本身的内在结构来构造连贯而简洁的摘要。该框架的核心在于引入基于论辩结构的结构性分析，并通过图表示法来促进关键信息的提取。权重算法被用于赋予文档单元不同的显著性值，从而实现有效的排名和相关内容的选择。此外，该框架设计考虑了用户偏好和时间限制，确保生成个性化且上下文相关的摘要。摘要生成过程进行了详尽的阐述，包括文档规定、图的构建、单元权重分配以及摘要提取，并通过示例和算法解释来支持这一过程。所提出的方法是自动摘要技术的重要进步，具有广泛的应用前景，将在多媒体文档处理领域带来革命性的变化。 

---
# Modality-Projection Universal Model for Comprehensive Full-Body Medical Imaging Segmentation 

**Title (ZH)**: 全面身体医学成像分割的模态投影通用模型 

**Authors**: Yixin Chen, Lin Gao, Yajuan Gao, Rui Wang, Jingge Lian, Xiangxi Meng, Yanhua Duan, Leiying Chai, Hongbin Han, Zhaoping Cheng, Zhaoheng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2412.19026)  

**Abstract**: The integration of deep learning in medical imaging has shown great promise for enhancing diagnostic, therapeutic, and research outcomes. However, applying universal models across multiple modalities remains challenging due to the inherent variability in data characteristics. This study aims to introduce and evaluate a Modality Projection Universal Model (MPUM). MPUM employs a novel modality-projection strategy, which allows the model to dynamically adjust its parameters to optimize performance across different imaging modalities. The MPUM demonstrated superior accuracy in identifying anatomical structures, enabling precise quantification for improved clinical decision-making. It also identifies metabolic associations within the brain-body axis, advancing research on brain-body physiological correlations. Furthermore, MPUM's unique controller-based convolution layer enables visualization of saliency maps across all network layers, significantly enhancing the model's interpretability. 

**Abstract (ZH)**: 将深度学习集成到医学影像中展现了显著的潜力，可提升诊断、治疗和研究结果。然而，由于数据特征的内在差异，跨多种模态应用通用模型仍然具有挑战性。本研究旨在介绍和评估一种模态投影通用模型（Modality Projection Universal Model，MPUM）。MPUM 采用了一种新颖的模态投影策略，使模型能够动态调整其参数以优化不同成像模态的表现。MPUM 在识别解剖结构方面表现出卓越的准确性，能够精确量化，从而提高临床决策的质量。此外，MPUM 还识别了脑-体轴内的代谢关联，促进了对脑-体生理性相关性的研究。此外，MPUM 的独特基于控制器的卷积层能够跨所有网络层可视化重要性图，显著增强了模型的可解释性。 

---
# Enhancing Audiovisual Speech Recognition through Bifocal Preference Optimization 

**Title (ZH)**: 通过双焦偏好优化增强音视频语音识别 

**Authors**: Yihan Wu, Yichen Lu, Yifan Peng, Xihua Wang, Ruihua Song, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2412.19005)  

**Abstract**: Audiovisual Automatic Speech Recognition (AV-ASR) aims to improve speech recognition accuracy by leveraging visual signals. It is particularly challenging in unconstrained real-world scenarios across various domains due to noisy acoustic environments, spontaneous speech, and the uncertain use of visual information. Most previous works fine-tune audio-only ASR models on audiovisual datasets, optimizing them for conventional ASR objectives. However, they often neglect visual features and common errors in unconstrained video scenarios. In this paper, we propose using a preference optimization strategy to improve speech recognition accuracy for real-world videos. First, we create preference data via simulating common errors that occurred in AV-ASR from two focals: manipulating the audio or vision input and rewriting the output transcript. Second, we propose BPO-AVASR, a Bifocal Preference Optimization method to improve AV-ASR models by leveraging both input-side and output-side preference. Extensive experiments demonstrate that our approach significantly improves speech recognition accuracy across various domains, outperforming previous state-of-the-art models on real-world video speech recognition. 

**Abstract (ZH)**: 视听自动语音识别（AV-ASR）旨在通过利用视觉信号来提高语音识别的准确性。由于存在噪声的声学环境、自发的言语表达以及视觉信息的不确定使用，这种技术在各个领域的非受控实际场景中尤为具有挑战性。大多数先前的研究在视听数据集上微调仅依赖音频的ASR模型，并优化它们以实现传统ASR目标。然而，这些研究通常忽视了视听场景中的视觉特征以及常见的错误。在本文中，我们提出了一种偏好优化策略以提高非受控视频中的语音识别准确性。首先，我们通过模拟AV-ASR中发生的常见错误来创建偏好数据，主要从两个角度入手：修改音频或视觉输入及重写输出转录。其次，我们提出了一种双焦偏好优化方法（BPO-AVASR），通过结合输入侧和输出侧的偏好来改进AV-ASR模型。大量的实验表明，我们的方法在多个领域显著提高了语音识别准确性，并在实际视频语音识别方面优于以前的最先进模型。 

---
# Open-Vocabulary Panoptic Segmentation Using BERT Pre-Training of Vision-Language Multiway Transformer Model 

**Title (ZH)**: 使用Vision-Language多方式变换模型的BERT预训练实现开放词汇语义分割 

**Authors**: Yi-Chia Chen, Wei-Hua Li, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.18917)  

**Abstract**: Open-vocabulary panoptic segmentation remains a challenging problem. One of the biggest difficulties lies in training models to generalize to an unlimited number of classes using limited categorized training data. Recent popular methods involve large-scale vision-language pre-trained foundation models, such as CLIP. In this paper, we propose OMTSeg for open-vocabulary segmentation using another large-scale vision-language pre-trained model called BEiT-3 and leveraging the cross-modal attention between visual and linguistic features in BEiT-3 to achieve better performance. Experiments result demonstrates that OMTSeg performs favorably against state-of-the-art models. 

**Abstract (ZH)**: 开放词汇的全景分割仍然是一个具有挑战性的问题。其中最大的困难在于使用有限的分类训练数据训练模型以泛化到无限数量的类别。近年来，流行的 方法涉及大规模的视觉-语言预训练基础模型，如CLIP。在本文中，我们提出了一种名为OMTSeg的框架，利用另一个大规模的视觉-语言预训练模型BEiT-3，并通过BEiT-3中视觉和语言特征的跨模态注意力实现更好的性能。实验结果表明，OMTSeg在与最先进的模型相比时表现优异。 

---
# ObitoNet: Multimodal High-Resolution Point Cloud Reconstruction 

**Title (ZH)**: ObitoNet：多模态高分辨率点云重建 

**Authors**: Apoorv Thapliyal, Vinay Lanka, Swathi Baskaran  

**Link**: [PDF](https://arxiv.org/pdf/2412.18775)  

**Abstract**: ObitoNet employs a Cross Attention mechanism to integrate multimodal inputs, where Vision Transformers (ViT) extract semantic features from images and a point cloud tokenizer processes geometric information using Farthest Point Sampling (FPS) and K Nearest Neighbors (KNN) for spatial structure capture. The learned multimodal features are fed into a transformer-based decoder for high-resolution point cloud reconstruction. This approach leverages the complementary strengths of both modalities rich image features and precise geometric details ensuring robust point cloud generation even in challenging conditions such as sparse or noisy data. 

**Abstract (ZH)**: ObitoNet采用跨注意力机制整合多模态输入，其中视觉transformer（ViT）从图像中提取语义特征，点云分词器通过最远点采样（FPS）和K近邻（KNN）处理几何信息以捕获空间结构。学习到的多模态特征被输入一个基于transformer的解码器进行高分辨率点云重建。该方法利用了图像特征丰富和几何细节精确的优势，即使在稀疏或嘈杂数据等具有挑战性的条件下也能确保点云生成的鲁棒性。 

---
# Video Is Worth a Thousand Images: Exploring the Latest Trends in Long Video Generation 

**Title (ZH)**: 千图不如一视频：探索长视频生成的最新趋势 

**Authors**: Faraz Waseem, Muhammad Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2412.18688)  

**Abstract**: An image may convey a thousand words, but a video composed of hundreds or thousands of image frames tells a more intricate story. Despite significant progress in multimodal large language models (MLLMs), generating extended videos remains a formidable challenge. As of this writing, OpenAI's Sora, the current state-of-the-art system, is still limited to producing videos that are up to one minute in length. This limitation stems from the complexity of long video generation, which requires more than generative AI techniques for approximating density functions essential aspects such as planning, story development, and maintaining spatial and temporal consistency present additional hurdles. Integrating generative AI with a divide-and-conquer approach could improve scalability for longer videos while offering greater control. In this survey, we examine the current landscape of long video generation, covering foundational techniques like GANs and diffusion models, video generation strategies, large-scale training datasets, quality metrics for evaluating long videos, and future research areas to address the limitations of the existing video generation capabilities. We believe it would serve as a comprehensive foundation, offering extensive information to guide future advancements and research in the field of long video generation. 

**Abstract (ZH)**: 一张图片可能蕴含千言万语，而由数百乃至数千张图像帧组成的视频则讲述一个更为复杂的故事。尽管多模态大规模语言模型（MLLMs）取得了显著进展，生成较长的视频依然是一个严峻的挑战。截至本文撰写之时，OpenAI的Sora仍是当前最先进的系统，也只能生成一分钟以内的视频。这一限制源自于长视频生成的复杂性，这需要不仅仅是生成式AI技术来逼近密度函数，而且规划、故事情节的发展以及保持空间和时间一致性的必要方面也增加了额外的困难。将生成式AI与分而治之的方法结合起来，可以提高生成较长视频的可扩展性并提供更大的控制。在这篇综述中，我们将探讨长视频生成的当前研究景观，涵盖诸如生成对抗网络（GANs）和扩散模型等基础技术、视频生成策略、大规模训练数据集、评估长视频质量的指标，以及未来的研究领域，用以解决现有视频生成能力的限制。我们相信这篇综述将提供一个全面的基础，为长视频生成领域的未来进展和研究提供丰富的信息指导。 

---
# Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey 

**Title (ZH)**: 面向多模态智能的下一个令牌预测：一项综述 

**Authors**: Liang Chen, Zekun Wang, Shuhuai Ren, Lei Li, Haozhe Zhao, Yunshui Li, Zefan Cai, Hongcheng Guo, Lei Zhang, Yizhe Xiong, Yichi Zhang, Ruoyu Wu, Qingxiu Dong, Ge Zhang, Jian Yang, Lingwei Meng, Shujie Hu, Yulong Chen, Junyang Lin, Shuai Bai, Andreas Vlachos, Xu Tan, Minjia Zhang, Wen Xiao, Aaron Yee, Tianyu Liu, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18619)  

**Abstract**: Building on the foundations of language modeling in natural language processing, Next Token Prediction (NTP) has evolved into a versatile training objective for machine learning tasks across various modalities, achieving considerable success. As Large Language Models (LLMs) have advanced to unify understanding and generation tasks within the textual modality, recent research has shown that tasks from different modalities can also be effectively encapsulated within the NTP framework, transforming the multimodal information into tokens and predict the next one given the context. This survey introduces a comprehensive taxonomy that unifies both understanding and generation within multimodal learning through the lens of NTP. The proposed taxonomy covers five key aspects: Multimodal tokenization, MMNTP model architectures, unified task representation, datasets \& evaluation, and open challenges. This new taxonomy aims to aid researchers in their exploration of multimodal intelligence. An associated GitHub repository collecting the latest papers and repos is available at this https URL 

**Abstract (ZH)**: 基于自然语言处理领域的语言建模基础，Next Token Prediction (NTP) 已发展成为一种在不同模态的机器学习任务中具备广泛训练目标的工具，并取得了显著的成功。随着大规模语言模型（LLMs）的进步，这些模型已经能够统一处理文本模态中的理解和生成任务，最近的研究显示，来自其他不同模态的任务也可以有效包含在 NTP 框架中，即将多模态信息转化为 tokens，并在给定上下文的情况下预测下一个 tokens。本文综述引入了一个综合分类法，通过 NTP 的视角，将理解与生成统一在多模态学习中。该提出的分类法涵盖了五个关键方面：多模态分词、MMNTP 模型架构、统一的任务表示、数据集与评估，以及开放性的挑战。这一新的分类法旨在帮助研究人员探索多模态智能。与此相关的一个 GitHub 代码库，收集了最新的论文和资源，可在以下链接找到：[此处链接](此 https URL)。 

---
# Investigating Acoustic-Textual Emotional Inconsistency Information for Automatic Depression Detection 

**Title (ZH)**: 探究声学-文本情绪不一致性信息以实现自动抑郁检测 

**Authors**: Rongfeng Su, Changqing Xu, Xinyi Wu, Feng Xu, Xie Chen, Lan Wangt, Nan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18614)  

**Abstract**: Previous studies have demonstrated that emotional features from a single acoustic sentiment label can enhance depression diagnosis accuracy. Additionally, according to the Emotion Context-Insensitivity theory and our pilot study, individuals with depression might convey negative emotional content in an unexpectedly calm manner, showing a high degree of inconsistency in emotional expressions during natural conversations. So far, few studies have recognized and leveraged the emotional expression inconsistency for depression detection. In this paper, a multimodal cross-attention method is presented to capture the Acoustic-Textual Emotional Inconsistency (ATEI) information. This is achieved by analyzing the intricate local and long-term dependencies of emotional expressions across acoustic and textual domains, as well as the mismatch between the emotional content within both domains. A Transformer-based model is then proposed to integrate this ATEI information with various fusion strategies for detecting depression. Furthermore, a scaling technique is employed to adjust the ATEI feature degree during the fusion process, thereby enhancing the model's ability to discern patients with depression across varying levels of severity. To best of our knowledge, this work is the first to incorporate emotional expression inconsistency information into depression detection. Experimental results on a counseling conversational dataset illustrate the effectiveness of our method. 

**Abstract (ZH)**: 先前的研究已经证明，单一声学情感标签中的情感特征能够提高抑郁症诊断的准确性。另外，根据情绪情景无关性理论以及我们的初步研究结果，抑郁症患者可能以出乎意料的平静方式传达负面情感内容，在自然对话中表现出情绪表达的高度不一致性。目前，很少有研究能够识别并利用这种情绪表达的不一致性来进行抑郁检测。本文提出了一种多模态跨注意力方法，旨在捕捉声学-文本情绪不一致性（ATEI）信息。这一方法通过分析声学和文本领域中情绪表达的复杂局部与长期依赖关系以及两个领域中情绪内容的不匹配情况来实现。随后提出了一种基于Transformer的模型，并结合多种融合策略将ATEI信息融入其中，以检测抑郁症。此外，在融合过程中采用了一种缩放技术来调整ATEI特征的强度，从而增强模型在不同严重程度患者中的识别能力。据我们所知，这是首次将情绪表达的不一致性信息引入抑郁症检测的研究。在咨询对话数据集上的实验结果表明了我们方法的有效性。 

---
# RapGuard: Safeguarding Multimodal Large Language Models via Rationale-aware Defensive Prompting 

**Title (ZH)**: RapGuard：通过理据意识防御型提示保护多模态大规模语言模型 

**Authors**: Yilei Jiang, Yingshui Tan, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2412.18826)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have made remarkable progress in vision-language reasoning, they are also more susceptible to producing harmful content compared to models that focus solely on text. Existing defensive prompting techniques rely on a static, unified safety guideline that fails to account for the specific risks inherent in different multimodal contexts. To address these limitations, we propose RapGuard, a novel framework that uses multimodal chain-of-thought reasoning to dynamically generate scenario-specific safety prompts. RapGuard enhances safety by adapting its prompts to the unique risks of each input, effectively mitigating harmful outputs while maintaining high performance on benign tasks. Our experimental results across multiple MLLM benchmarks demonstrate that RapGuard achieves state-of-the-art safety performance, significantly reducing harmful content without degrading the quality of responses. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）在视觉语言推理方面取得了显著进展，但与仅专注于文本的模型相比，它们更容易生成有害内容。现有的防御性提示技术依靠的是静态的、统一的安全准则，无法考虑到不同多模态背景下固有的特定风险。为了解决这些限制，我们提出了一种名为RapGuard的新型框架，该框架利用多模态链式推理动态生成针对特定场景的安全提示。RapGuard通过适应每个输入的独特风险来提升安全性，有效地减轻有害输出，同时在无害任务上保持高水平的性能。我们在多个MLLM基准上的实验结果表明，RapGuard实现了最先进的安全性能，显著减少了有害内容，而不会降低响应的质量。 

---
# Intra- and Inter-modal Context Interaction Modeling for Conversational Speech Synthesis 

**Title (ZH)**: 跨模态和跨通道上下文交互建模在会话语音合成中的应用 

**Authors**: Zhenqi Jia, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18733)  

**Abstract**: Conversational Speech Synthesis (CSS) aims to effectively take the multimodal dialogue history (MDH) to generate speech with appropriate conversational prosody for target utterance. The key challenge of CSS is to model the interaction between the MDH and the target utterance. Note that text and speech modalities in MDH have their own unique influences, and they complement each other to produce a comprehensive impact on the target utterance. Previous works did not explicitly model such intra-modal and inter-modal interactions. To address this issue, we propose a new intra-modal and inter-modal context interaction scheme-based CSS system, termed III-CSS. Specifically, in the training phase, we combine the MDH with the text and speech modalities in the target utterance to obtain four modal combinations, including Historical Text-Next Text, Historical Speech-Next Speech, Historical Text-Next Speech, and Historical Speech-Next Text. Then, we design two contrastive learning-based intra-modal and two inter-modal interaction modules to deeply learn the intra-modal and inter-modal context interaction. In the inference phase, we take MDH and adopt trained interaction modules to fully infer the speech prosody of the target utterance's text content. Subjective and objective experiments on the DailyTalk dataset show that III-CSS outperforms the advanced baselines in terms of prosody expressiveness. Code and speech samples are available at this https URL. 

**Abstract (ZH)**: 对话式语音合成（Conversational Speech Synthesis, CSS）的目标是有效利用多模态对话历史（Multimodal Dialogue History, MDH），为目标话语生成具有适当会话语调的语音。CSS的关键挑战在于如何建模MDH与目标话语之间的相互作用。值得注意的是，MDH中的文本和语音模态各自具有独特的影响力，并相互补充，共同对目标话语产生综合影响。以往的工作没有明确建模这种模态内和模态间的相互作用。为解决这一问题，我们提出了一种基于模态内和模态间上下文交互方案的新CSS系统，称为III-CSS。具体而言，在训练阶段，我们将MDH与目标话语中的文本和语音模态结合，获得四种模态组合，包括历史文本-下一步文本、历史语音-下一步语音、历史文本-下一步语音、以及历史语音-下一步文本。然后，我们设计了两个基于对比学习的模态内交互模块和两个模态间交互模块，以深入学习模态内和模态间的上下文交互。在推理阶段，我们利用MDH和训练好的交互模块全面推断目标话语文本内容的语音语调。在DailyTalk数据集上进行的主观数字化和客观实验表明，III-CSS在语调表达能力方面优于先进的基线系统。代码和语音样本可在以下链接获取：<这个链接>。 

---
# Multi-Head Attention Driven Dynamic Visual-Semantic Embedding for Enhanced Image-Text Matching 

**Title (ZH)**: 基于多头注意力的动态视觉-语义嵌入增强图像-文本匹配 

**Authors**: Wenjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.19184)  

**Abstract**: With the rapid development of multimodal learning, the image-text matching task, as a bridge connecting vision and language, has become increasingly important. Based on existing research, this study proposes an innovative visual semantic embedding model, Multi-Headed Consensus-Aware Visual-Semantic Embedding (MH-CVSE). This model introduces a multi-head self-attention mechanism based on the consensus-aware visual semantic embedding model (CVSE) to capture information in multiple subspaces in parallel, significantly enhancing the model's ability to understand and represent the complex relationship between images and texts. In addition, we adopt a parameterized feature fusion strategy to flexibly integrate feature information at different levels, further improving the model's expressive power. In terms of loss function design, the MH-CVSE model adopts a dynamic weight adjustment strategy to dynamically adjust the weight according to the loss value itself, so that the model can better balance the contribution of different loss terms during training. At the same time, we introduce a cosine annealing learning rate strategy to help the model converge more stably in the later stages of training. Extensive experimental verification on the Flickr30k dataset shows that the MH-CVSE model achieves better performance than previous methods in both bidirectional image and text retrieval tasks, fully demonstrating its effectiveness and superiority. 

**Abstract (ZH)**: 随着多模态学习的迅速发展，图像-文本匹配任务作为连接视觉与语言的桥梁，已经变得越来越重要。基于现有的研究成果，本研究提出了一种创新的视觉语义嵌入模型——多头共识意识视觉语义嵌入（MH-CVSE）。该模型引入了基于共识意识视觉语义嵌入模型（CVSE）的多头自注意力机制，可以在多个子空间中并行捕捉信息，显著增强了模型对图像和文本之间复杂关系的理解和表示能力。此外，我们采用了参数化特征融合策略，灵活地整合不同层次的特征信息，进一步提升了模型的表达能力。在损失函数设计方面，MH-CVSE模型采用了动态权重调整策略，根据损失值本身动态调整权重，使模型在训练过程中能够更好地平衡不同的损失项贡献。同时，我们引入了余弦退火学习率策略，以帮助模型在训练后期更加稳定地收敛。在Flickr30k数据集上的大量实验验证表明，MH-CVSE模型在双向图像和文本检索任务中均优于以前的方法，充分展示了其有效性和优越性。 

---
# Towards Expressive Video Dubbing with Multiscale Multimodal Context Interaction 

**Title (ZH)**: 面向多尺度多模态上下文交互的表达性视频配音研究 

**Authors**: Yuan Zhao, Rui Liu, Gaoxiang Cong  

**Link**: [PDF](https://arxiv.org/pdf/2412.18748)  

**Abstract**: Automatic Video Dubbing (AVD) generates speech aligned with lip motion and facial emotion from scripts. Recent research focuses on modeling multimodal context to enhance prosody expressiveness but overlooks two key issues: 1) Multiscale prosody expression attributes in the context influence the current sentence's prosody. 2) Prosody cues in context interact with the current sentence, impacting the final prosody expressiveness. To tackle these challenges, we propose M2CI-Dubber, a Multiscale Multimodal Context Interaction scheme for AVD. This scheme includes two shared M2CI encoders to model the multiscale multimodal context and facilitate its deep interaction with the current sentence. By extracting global and local features for each modality in the context, utilizing attention-based mechanisms for aggregation and interaction, and employing an interaction-based graph attention network for fusion, the proposed approach enhances the prosody expressiveness of synthesized speech for the current sentence. Experiments on the Chem dataset show our model outperforms baselines in dubbing expressiveness. The code and demos are available at \textcolor[rgb]{0.93,0.0,0.47}{this https URL}. 

**Abstract (ZH)**: 自动视频配音（AVD）从脚本中生成与唇形和面部情感对齐的语音。近期的研究集中在通过建模多模态上下文来增强音调表现力，但忽略了两个关键问题：1）多尺度的音调表现属性对当前句子的音调有影响；2）多模态上下文中的音调线索与当前句子相互作用，影响最终的音调表现力。为解决这些问题，我们提出了一种多尺度多模态上下文交互方案（M2CI-Dubber）用于AVD。该方案包括两个共享的M2CI编码器，用于建模多尺度多模态上下文，并促进上下文与当前句子的深层次交互。通过提取上下文中每个模态的全局和局部特征，并利用基于注意力机制的聚合和交互方式，以及采用基于交互的图注意力网络进行融合，所提出的方法增强了当前句子生成语音的音调表现力。在Chem数据集上的实验表明，与基线模型相比，我们的模型在配音表现力方面表现更好。代码和演示可以在<此URL>获取。 

---
# The Illusion-Illusion: Vision Language Models See Illusions Where There are None 

**Title (ZH)**: 幻觉之幻：视觉语言模型在无任何实际幻觉存在的地方看到了幻觉 

**Authors**: Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2412.18613)  

**Abstract**: Illusions are entertaining, but they are also a useful diagnostic tool in cognitive science, philosophy, and neuroscience. A typical illusion shows a gap between how something "really is" and how something "appears to be", and this gap helps us understand the mental processing that lead to how something appears to be. Illusions are also useful for investigating artificial systems, and much research has examined whether computational models of perceptions fall prey to the same illusions as people. Here, I invert the standard use of perceptual illusions to examine basic processing errors in current vision language models. I present these models with illusory-illusions, neighbors of common illusions that should not elicit processing errors. These include such things as perfectly reasonable ducks, crooked lines that truly are crooked, circles that seem to have different sizes because they are, in fact, of different sizes, and so on. I show that many current vision language systems mistakenly see these illusion-illusions as illusions. I suggest that such failures are part of broader failures already discussed in the literature. 

**Abstract (ZH)**: 幻觉虽然令人娱乐，但在认知科学、哲学和神经科学中也是一项有用的诊断工具。一个典型的幻觉展示了“实际情况”与“表面看起来的情况”之间的差距，而这种差距有助于我们理解导致某种事物看起来是某样东西的心理过程。幻觉另一个有用之处在于，它们可用于研究人工系统，许多研究探讨了感知计算模型是否也会受到与人类相同的幻觉影响。在此，我将反向利用感知幻觉，研究当前视觉语言模型的基本处理错误。我将这些模型暴露于幻觉类似物，这些幻觉类似物不应引起处理错误。这包括看起来非常合理的鸭子、实际歪曲的线、看似大小不同的实际上确实大小不同的圆等。我展示了当前许多视觉语言系统错误地将这些幻觉类似物视为幻觉。我建议，这些失败是文献中已经讨论的更广泛失败的一部分。 

---
