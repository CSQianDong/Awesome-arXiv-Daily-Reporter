# Natural Language-Assisted Multi-modal Medication Recommendation 

**Title (ZH)**: 自然语言辅助的多模态药物推荐 

**Authors**: Jie Tan, Yu Rong, Kangfei Zhao, Tian Bian, Tingyang Xu, Junzhou Huang, Hong Cheng, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2501.07166)  

**Abstract**: Combinatorial medication recommendation(CMR) is a fundamental task of healthcare, which offers opportunities for clinical physicians to provide more precise prescriptions for patients with intricate health conditions, particularly in the scenarios of long-term medical care. Previous research efforts have sought to extract meaningful information from electronic health records (EHRs) to facilitate combinatorial medication recommendations. Existing learning-based approaches further consider the chemical structures of medications, but ignore the textual medication descriptions in which the functionalities are clearly described. Furthermore, the textual knowledge derived from the EHRs of patients remains largely underutilized. To address these issues, we introduce the Natural Language-Assisted Multi-modal Medication Recommendation(NLA-MMR), a multi-modal alignment framework designed to learn knowledge from the patient view and medication view jointly. Specifically, NLA-MMR formulates CMR as an alignment problem from patient and medication modalities. In this vein, we employ pretrained language models(PLMs) to extract in-domain knowledge regarding patients and medications, serving as the foundational representation for both modalities. In the medication modality, we exploit both chemical structures and textual descriptions to create medication representations. In the patient modality, we generate the patient representations based on textual descriptions of diagnosis, procedure, and symptom. Extensive experiments conducted on three publicly accessible datasets demonstrate that NLA-MMR achieves new state-of-the-art performance, with a notable average improvement of 4.72% in Jaccard score. Our source code is publicly available on this https URL. 

**Abstract (ZH)**: 组合药物推荐（CMR）是医疗保健中的一个基本任务，为临床医生提供了机会，使其能够为具有复杂健康状况的患者提供更精确的处方，特别是在长期医疗护理场景中。以往的研究努力通过从电子健康记录（EHRs）中提取有意义的信息来促进组合药物推荐。现有的基于学习的方法进一步考虑了药物的化学结构，但忽略了其中明确描述功能的药物文本描述。此外，从患者EHRs中获得的文本知识仍然被大量未充分利用。为了解决这些问题，我们提出了自然语言辅助的多模态药物推荐（NLA-MMR），这是一种旨在联合从患者视角和药物视角学习知识的多模态对齐框架。具体而言，NLA-MMR 将CMR 形式化为来自患者和药物模态的对齐问题。在这项工作中，我们使用预训练语言模型（PLMs）来提取与患者和药物相关的领域知识，作为两种模态的基础表示。在药物模态中，我们利用化学结构和文本描述来创建药物表示。在患者模态中，我们根据诊断、程序和症状的文本描述生成患者的表示。在三个公开可访问的数据库上进行的广泛实验表明，NLA-MMR 达到了新的最佳性能，在Jaccard分数上实现了显著的平均改进，为4.72%。我们的源代码已在此处公开 accessible on this https URL。 

---
# CureGraph: Contrastive Multi-Modal Graph Representation Learning for Urban Living Circle Health Profiling and Prediction 

**Title (ZH)**: CureGraph：对比多模态图表示学习在城市生活圈健康档案构建与预测中的应用 

**Authors**: Jinlin Li, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.07157)  

**Abstract**: The early detection and prediction of health status decline among the elderly at the neighborhood level are of great significance for urban planning and public health policymaking. While existing studies affirm the connection between living environments and health outcomes, most rely on single data modalities or simplistic feature concatenation of multi-modal information, limiting their ability to comprehensively profile the health-oriented urban environments. To fill this gap, we propose CureGraph, a contrastive multi-modal representation learning framework for urban health prediction that employs graph-based techniques to infer the prevalence of common chronic diseases among the elderly within the urban living circles of each neighborhood. CureGraph leverages rich multi-modal information, including photos and textual reviews of residential areas and their surrounding points of interest, to generate urban neighborhood embeddings. By integrating pre-trained visual and textual encoders with graph modeling techniques, CureGraph captures cross-modal spatial dependencies, offering a comprehensive understanding of urban environments tailored to elderly health considerations. Extensive experiments on real-world datasets demonstrate that CureGraph improves the best baseline by $28\%$ on average in terms of $R^2$ across elderly disease risk prediction tasks. Moreover, the model enables the identification of stage-wise chronic disease progression and supports comparative public health analysis across neighborhoods, offering actionable insights for sustainable urban development and enhanced quality of life. The code is publicly available at this https URL. 

**Abstract (ZH)**: 在社区层面早期检测和预测老年人健康状况的下降对城市规划和公共卫生政策制定具有重要意义。尽管现有研究表明生活环境与健康结果之间的关联，但大多数研究依赖单一的数据模态或简单处理多模态信息的特征拼接，限制了它们全面刻画健康导向型城市环境的能力。为弥补这一不足，我们提出了CureGraph，一种基于图的对比多模态表示学习框架，用于城市健康预测。CureGraph利用图技术推断每个社区中城市生活圈内老年人常见慢性疾病的患病率。CureGraph利用丰富的多模态信息，包括住宅区及其周边兴趣点的照片和文本评论，生成城市社区嵌入。通过将预训练的视觉和文本编码器与图建模技术相结合，CureGraph捕捉跨模态的空间依赖性，提供符合老年人健康考虑的全面城市环境理解。在真实数据集上的广泛实验表明，与基线方法相比，CureGraph在老年人疾病风险预测任务中的$R^2$平均提高了28%。此外，该模型能够识别慢性疾病的发展阶段，并支持跨社区的公共卫生比较分析，为可持续城市发展和提高生活质量提供行动指南。代码在此处公开：[请填入实际的URL]。 

---
# Unveiling the Potential of Text in High-Dimensional Time Series Forecasting 

**Title (ZH)**: 揭开高维时间序列预测中文本的潜力 

**Authors**: Xin Zhou, Weiqing Wang, Shilin Qu, Zhiqiang Zhang, Christoph Bergmeir  

**Link**: [PDF](https://arxiv.org/pdf/2501.07048)  

**Abstract**: Time series forecasting has traditionally focused on univariate and multivariate numerical data, often overlooking the benefits of incorporating multimodal information, particularly textual data. In this paper, we propose a novel framework that integrates time series models with Large Language Models to improve high-dimensional time series forecasting. Inspired by multimodal models, our method combines time series and textual data in the dual-tower structure. This fusion of information creates a comprehensive representation, which is then processed through a linear layer to generate the final forecast. Extensive experiments demonstrate that incorporating text enhances high-dimensional time series forecasting performance. This work paves the way for further research in multimodal time series forecasting. 

**Abstract (ZH)**: 时间序列预测传统上集中在单一变量和多变量数值数据上，往往忽视了整合多元信息，特别是文本数据的好处。在本文中，我们提出了一种新颖的框架，将时间序列模型与大型语言模型结合，以提升高维时间序列预测性能。受到多模态模型的启发，我们的方法在双塔结构中结合了时间序列和文本数据。这种信息的融合生成了一个综合表示，然后通过线性层生成最终的预测。广泛的实验表明，引入文本能够提升高维时间序列预测的性能。本文为多模态时间序列预测领域的进一步研究奠定了基础。 

---
# Leveraging Taxonomy and LLMs for Improved Multimodal Hierarchical Classification 

**Title (ZH)**: 利用分类体系和大规模语言模型以改进多模态层次分类 

**Authors**: Shijing Chen, Mohamed Reda Bouadjenek, Shoaib Jameel, Usman Naseem, Basem Suleiman, Flora D. Salim, Hakim Hacid, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2501.06827)  

**Abstract**: Multi-level Hierarchical Classification (MLHC) tackles the challenge of categorizing items within a complex, multi-layered class structure. However, traditional MLHC classifiers often rely on a backbone model with independent output layers, which tend to ignore the hierarchical relationships between classes. This oversight can lead to inconsistent predictions that violate the underlying taxonomy. Leveraging Large Language Models (LLMs), we propose a novel taxonomy-embedded transitional LLM-agnostic framework for multimodality classification. The cornerstone of this advancement is the ability of models to enforce consistency across hierarchical levels. Our evaluations on the MEP-3M dataset - a multi-modal e-commerce product dataset with various hierarchical levels - demonstrated a significant performance improvement compared to conventional LLM structures. 

**Abstract (ZH)**: 多层层次分类（MLHC）解决了在复杂多层类结构中对项目进行分类的挑战。然而，传统的MLHC分类器往往依赖于具有独立输出层的骨干模型，这些模型倾向于忽略类之间的层次关系。这种忽视可能导致违反底层分类体系的一致性预测。利用大型语言模型（LLMs），我们提出了一种新颖的嵌入分类学的过渡LLM无关框架，用于多模态分类。这一进展的核心在于模型能够确保在不同层次上的一致性。我们对MEP-3M数据集（这是一个包含多种层次结构的多模态电子商务产品数据集）进行的评估表明，与传统的LLM结构相比，该框架在性能上取得了显著的提升。 

---
# A Multimodal Social Agent 

**Title (ZH)**: 多模态社会代理模型 

**Authors**: Athina Bikaki, Ioannis A. Kakadiaris  

**Link**: [PDF](https://arxiv.org/pdf/2501.06189)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable progress in common-sense reasoning tasks. This ability is fundamental to understanding social dynamics, interactions, and communication. However, the potential of integrating computers with these social capabilities is still relatively unexplored. However, the potential of integrating computers with these social capabilities is still relatively unexplored. This paper introduces MuSA, a multimodal LLM-based agent that analyzes text-rich social content tailored to address selected human-centric content analysis tasks, such as question answering, visual question answering, title generation, and categorization. It uses planning, reasoning, acting, optimizing, criticizing, and refining strategies to complete a task. Our approach demonstrates that MuSA can automate and improve social content analysis, helping decision-making processes across various applications. We have evaluated our agent's capabilities in question answering, title generation, and content categorization tasks. MuSA performs substantially better than our baselines. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在常识推理任务中取得了显著进展。这种能力对于理解社会动态、交互和沟通至关重要。然而，将计算机与这些社会能力整合的可能性仍然相对未被充分探索。本文介绍了一种基于多模态LLM的代理MuSA，该代理专门用于分析富含文本的社会内容，以应对诸如问答、视觉问答、标题生成和内容分类等人本中心的内容分析任务。MuSA 使用规划、推理、执行、优化、批评和改进等策略来完成任务。我们的方法表明，MuSA 可以自动化并提升社会内容分析，帮助各类应用中的决策过程。我们已在问答、标题生成和内容分类任务上评估了该代理的能力。MuSA 在这些任务上的表现显著优于我们的基线模型。 

---
# RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignment 

**Title (ZH)**: RadAlign：通过视觉-语言概念对齐提升放射学报告生成 

**Authors**: Difei Gu, Yunhe Gao, Yang Zhou, Mu Zhou, Dimitris Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2501.07525)  

**Abstract**: Automated chest radiographs interpretation requires both accurate disease classification and detailed radiology report generation, presenting a significant challenge in the clinical workflow. Current approaches either focus on classification accuracy at the expense of interpretability or generate detailed but potentially unreliable reports through image captioning techniques. In this study, we present RadAlign, a novel framework that combines the predictive accuracy of vision-language models (VLMs) with the reasoning capabilities of large language models (LLMs). Inspired by the radiologist's workflow, RadAlign first employs a specialized VLM to align visual features with key medical concepts, achieving superior disease classification with an average AUC of 0.885 across multiple diseases. These recognized medical conditions, represented as text-based concepts in the aligned visual-language space, are then used to prompt LLM-based report generation. Enhanced by a retrieval-augmented generation mechanism that grounds outputs in similar historical cases, RadAlign delivers superior report quality with a GREEN score of 0.678, outperforming state-of-the-art methods' 0.634. Our framework maintains strong clinical interpretability while reducing hallucinations, advancing automated medical imaging and report analysis through integrated predictive and generative AI. Code is available at this https URL. 

**Abstract (ZH)**: 自动化胸片解释要求同时具备准确的疾病分类和详细的放射学报告生成能力，这在临床工作流程中提出了重大挑战。当前的方法要么侧重于提高分类准确性而牺牲可解释性，要么通过图像字幕技术生成详细的但可能不稳定的报告。在此研究中，我们提出了RadAlign，这是一个新型框架，结合了视觉-语言模型（VLMs）的预测准确性与大型语言模型（LLMs）的推理能力。RadAlign受到放射科医生工作流程的启发，首先使用一种专门的VLM将视觉特征与关键医学概念对齐，实现了多种疾病平均AUC达0.885的优秀疾病分类。这些识别出的医学条件以文本形式表示在对齐的视觉-语言空间中，随后用于触发基于LLM的报告生成。通过一种检索增强生成机制，将输出与类似的历史病例关联起来，RadAlign提供了质量更优的报告，其GREEN评分为0.678，优于现有方法的0.634。我们的框架保持了强烈的临床可解释性，同时减少了幻觉现象，通过整合预测和生成AI技术，推动了自动化医学影像和报告分析的发展。代码可在以下链接获取：this https URL。 

---
# Exploring the Use of Contrastive Language-Image Pre-Training for Human Posture Classification: Insights from Yoga Pose Analysis 

**Title (ZH)**: 探索对比语言-图像预训练在人体姿态分类中的应用：以瑜伽姿势分析为例的见解 

**Authors**: Andrzej D. Dobrzycki, Ana M. Bernardos, Luca Bergesio, Andrzej Pomirski, Daniel Sáez-Trigueros  

**Link**: [PDF](https://arxiv.org/pdf/2501.07221)  

**Abstract**: Accurate human posture classification in images and videos is crucial for automated applications across various fields, including work safety, physical rehabilitation, sports training, or daily assisted living. Recently, multimodal learning methods, such as Contrastive Language-Image Pretraining (CLIP), have advanced significantly in jointly understanding images and text. This study aims to assess the effectiveness of CLIP in classifying human postures, focusing on its application in yoga. Despite the initial limitations of the zero-shot approach, applying transfer learning on 15,301 images (real and synthetic) with 82 classes has shown promising results. The article describes the full procedure for fine-tuning, including the choice for image description syntax, models and hyperparameters adjustment. The fine-tuned CLIP model, tested on 3826 images, achieves an accuracy of over 85%, surpassing the current state-of-the-art of previous works on the same dataset by approximately 6%, its training time being 3.5 times lower than what is needed to fine-tune a YOLOv8-based model. For more application-oriented scenarios, with smaller datasets of six postures each, containing 1301 and 401 training images, the fine-tuned models attain an accuracy of 98.8% and 99.1%, respectively. Furthermore, our experiments indicate that training with as few as 20 images per pose can yield around 90% accuracy in a six-class dataset. This study demonstrates that this multimodal technique can be effectively used for yoga pose classification, and possibly for human posture classification, in general. Additionally, CLIP inference time (around 7 ms) supports that the model can be integrated into automated systems for posture evaluation, e.g., for developing a real-time personal yoga assistant for performance assessment. 

**Abstract (ZH)**: 图像和视频中人类姿态的准确分类对于工作安全、物理康复、体育训练或日常辅助生活等领域中的自动化应用至关重要。最近，多模态学习方法，如对比语言-图像预训练（CLIP），在联合理解和图像与文本方面取得了显著的进步。本研究旨在评估CLIP在分类人类姿态方面的有效性，重点是其在瑜伽中的应用。尽管最初零样本方法存在一些限制，但在15,301张图像（真实和合成）和82个类别的基础上进行的迁移学习显示出了令人鼓舞的结果。文章详细描述了微调的整个过程，包括选择图像描述语法、模型和超参数调整的方法。经过微调的CLIP模型在3826张图像上的准确率达到85%以上，相比之前在同一数据集上的工作，提高了约6%，其训练时间仅为基于YOLOv8模型微调所需时间的3.5倍。对于更具体的应用场景，使用包含1301张和401张训练图像的每种姿态6种姿态的小数据集，微调后的模型分别达到了98.8%和99.1%的准确率。此外，我们的实验表明，每个姿态仅使用20张图像进行训练，即可在六类数据集中达到约90%的准确率。本研究证明，这种方法可以有效地用于瑜伽姿态分类，并且有可能适用于人类姿态分类中的其他应用。此外，CLIP的推理时间（约7毫秒）表明该模型可以集成到姿态评估的自动化系统中，例如开发一个实时的个人瑜伽助手进行表现评估。 

---
# Multi-face emotion detection for effective Human-Robot Interaction 

**Title (ZH)**: 有效的人机交互中的多面部情感检测 

**Authors**: Mohamed Ala Yahyaoui, Mouaad Oujabour, Leila Ben Letaifa, Amine Bohi  

**Link**: [PDF](https://arxiv.org/pdf/2501.07213)  

**Abstract**: The integration of dialogue interfaces in mobile devices has become ubiquitous, providing a wide array of services. As technology progresses, humanoid robots designed with human-like features to interact effectively with people are gaining prominence, and the use of advanced human-robot dialogue interfaces is continually expanding. In this context, emotion recognition plays a crucial role in enhancing human-robot interaction by enabling robots to understand human intentions. This research proposes a facial emotion detection interface integrated into a mobile humanoid robot, capable of displaying real-time emotions from multiple individuals on a user interface. To this end, various deep neural network models for facial expression recognition were developed and evaluated under consistent computer-based conditions, yielding promising results. Afterwards, a trade-off between accuracy and memory footprint was carefully considered to effectively implement this application on a mobile humanoid robot. 

**Abstract (ZH)**: 移动设备中对话界面的集成已变得无处不在，提供了广泛的服务。随着技术的进步，具有人类特征的类人机器人设计以有效与人交互正在获得越来越多的关注，且高级的人机对话界面的应用不断扩展。在此背景下，情绪识别在增强人机交互方面扮演着重要作用，因为它使机器人能够理解人类的意图。本研究提出了一种集成于移动类人机器人中的面部情绪检测界面，能够在用户界面上实时显示多个个体的情绪。为此，开发并评估了多种基于深层神经网络的表情识别模型，结果相当有前景。随后，仔细考虑了准确性和内存占用之间的权衡，以有效地在移动类人机器人上实施此应用。 

---
# A Multi-Modal Deep Learning Framework for Pan-Cancer Prognosis 

**Title (ZH)**: 一种多模态深度学习框架用于泛癌种预后分析 

**Authors**: Binyu Zhang, Shichao Li, Junpeng Jian, Zhu Meng, Limei Guo, Zhicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07016)  

**Abstract**: Prognostic task is of great importance as it closely related to the survival analysis of patients, the optimization of treatment plans and the allocation of resources. The existing prognostic models have shown promising results on specific datasets, but there are limitations in two aspects. On the one hand, they merely explore certain types of modal data, such as patient histopathology WSI and gene expression analysis. On the other hand, they adopt the per-cancer-per-model paradigm, which means the trained models can only predict the prognostic effect of a single type of cancer, resulting in weak generalization ability. In this paper, a deep-learning based model, named UMPSNet, is proposed. Specifically, to comprehensively understand the condition of patients, in addition to constructing encoders for histopathology images and genomic expression profiles respectively, UMPSNet further integrates four types of important meta data (demographic information, cancer type information, treatment protocols, and diagnosis results) into text templates, and then introduces a text encoder to extract textual features. In addition, the optimal transport OT-based attention mechanism is utilized to align and fuse features of different modalities. Furthermore, a guided soft mixture of experts (GMoE) mechanism is introduced to effectively address the issue of distribution differences among multiple cancer datasets. By incorporating the multi-modality of patient data and joint training, UMPSNet outperforms all SOTA approaches, and moreover, it demonstrates the effectiveness and generalization ability of the proposed learning paradigm of a single model for multiple cancer types. The code of UMPSNet is available at this https URL. 

**Abstract (ZH)**: 生存预测任务非常重要，因为它与患者的生存分析密切相关，优化治疗方案以及资源分配。现有的生存预测模型在特定数据集上取得了令人鼓舞的结果，但在两个方面存在局限性。一方面，它们仅探索某些类型的模态数据，例如患者的组织病理学WSI图像和基因表达分析。另一方面，它们采用每种癌症每种模型的范式，这意味着训练的模型只能预测单一类型的癌症效果，导致泛化能力较弱。在本文中，提出了一种基于深度学习的模型，命名为UMPSNet。

具体来说，为了全面了解患者的状况，除了分别构建组织病理图像和基因表达谱的编码器，UMPSNet进一步将四种重要的元数据（包括人口统计信息、癌症类型信息、治疗方案和诊断结果）整合到文本模板中，并引入文本编码器以提取文本特征。此外，利用最优传输OT（Optimal Transport）基于注意力机制对不同模态特征进行对齐和融合。进一步引入了导向软专家混合机制（GMoE：Guided Soft Mixture of Experts）以有效解决多个癌症数据集间分布差异的问题。通过融合患者的多模态数据并进行联合训练，UMPSNet在所有现有先进方法中表现更优。此外，UMPSNet还展示了基于单一模型多癌症类型的预测学习范式的有效性和泛化能力。UMPSNet的代码可参见：这个URL（请提供正确的URL）。 

---
# MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis 

**Title (ZH)**: MedGrad E-CLIP: 提高基于AI的皮肤病变诊断中的信任与透明度 

**Authors**: Sadia Kamal, Tim Oates  

**Link**: [PDF](https://arxiv.org/pdf/2501.06887)  

**Abstract**: As deep learning models gain attraction in medical data, ensuring transparent and trustworthy decision-making is essential. In skin cancer diagnosis, while advancements in lesion detection and classification have improved accuracy, the black-box nature of these methods poses challenges in understanding their decision processes, leading to trust issues among physicians. This study leverages the CLIP (Contrastive Language-Image Pretraining) model, trained on different skin lesion datasets, to capture meaningful relationships between visual features and diagnostic criteria terms. To further enhance transparency, we propose a method called MedGrad E-CLIP, which builds on gradient-based E-CLIP by incorporating a weighted entropy mechanism designed for complex medical imaging like skin lesions. This approach highlights critical image regions linked to specific diagnostic descriptions. The developed integrated pipeline not only classifies skin lesions by matching corresponding descriptions but also adds an essential layer of explainability developed especially for medical data. By visually explaining how different features in an image relates to diagnostic criteria, this approach demonstrates the potential of advanced vision-language models in medical image analysis, ultimately improving transparency, robustness, and trust in AI-driven diagnostic systems. 

**Abstract (ZH)**: 随着深度学习模型在医疗数据中的吸引力增加，确保透明和值得信赖的决策至关重要。在皮肤癌诊断中，尽管在病灶检测和分类方面的进步提高了准确率，但这些方法的黑箱性质给理解其决策过程带来了挑战，导致了医生的信任问题。本研究利用在不同皮肤病变数据集上训练的CLIP（对比语言-图像预训练）模型，捕捉视觉特征与诊断标准术语之间的有意义关系。为了进一步提高透明度，我们提出了一种名为MedGrad E-CLIP的方法，该方法在基于梯度的E-CLIP上引入了一个加权熵机制，专门适用于如皮肤病变等复杂医学成像。该方法强调了与特定诊断描述相联系的关键图像区域。开发的集成管道不仅通过匹配相应的描述来分类皮肤病变，还添加了一层特别为医疗数据设计的解释性描述。通过可视化解释图像中不同特征与诊断标准之间的关系，该方法展示了先进视觉-语言模型在医学图像分析中的潜力，最终提高了基于AI的诊断系统的透明度、鲁棒性和可信度。 

---
# MEXA-CTP: Mode Experts Cross-Attention for Clinical Trial Outcome Prediction 

**Title (ZH)**: MEXA-CTP: 模式专家跨注意力在临床试验结果预测中的应用 

**Authors**: Yiqing Zhang, Xiaozhong Liu, Fabricio Murai  

**Link**: [PDF](https://arxiv.org/pdf/2501.06823)  

**Abstract**: Clinical trials are the gold standard for assessing the effectiveness and safety of drugs for treating diseases. Given the vast design space of drug molecules, elevated financial cost, and multi-year timeline of these trials, research on clinical trial outcome prediction has gained immense traction. Accurate predictions must leverage data of diverse modes such as drug molecules, target diseases, and eligibility criteria to infer successes and failures. Previous Deep Learning approaches for this task, such as HINT, often require wet lab data from synthesized molecules and/or rely on prior knowledge to encode interactions as part of the model architecture. To address these limitations, we propose a light-weight attention-based model, MEXA-CTP, to integrate readily-available multi-modal data and generate effective representations via specialized modules dubbed "mode experts", while avoiding human biases in model design. We optimize MEXA-CTP with the Cauchy loss to capture relevant interactions across modes. Our experiments on the Trial Outcome Prediction (TOP) benchmark demonstrate that MEXA-CTP improves upon existing approaches by, respectively, up to 11.3% in F1 score, 12.2% in PR-AUC, and 2.5% in ROC-AUC, compared to HINT. Ablation studies are provided to quantify the effectiveness of each component in our proposed method. 

**Abstract (ZH)**: 临床试验是评估治疗疾病药物的有效性和安全性的金标准。鉴于药物分子设计空间宽广、高昂的财务成本以及多年的研究时间，临床试验结果预测的研究已获得极大的关注。准确的预测必须利用包括药物分子、目标疾病以及入组标准等多种模式的数据，以推断试验的成功与失败。先前针对这一任务的深度学习方法，如HINT，往往需要来自合成分子的湿实验室数据，并且常依赖于先验知识来编码相互作用作为模型架构的一部分。为了解决这些限制，我们提出了一种轻量级的基于注意力机制的模型MEXA-CTP，该模型能够整合现成的多种模式数据，并通过所谓的“模式专家”模块生成有效的表示，同时避免了在模型设计中的人为偏见。我们使用Cauchy损失优化MEXA-CTP，以捕捉不同模式之间的相关交互。我们在Trial Outcome Prediction (TOP)数据集上的实验表明，与HINT相比，MEXA-CTP分别在F1分数、PR-AUC和ROC-AUC上提高了最多11.3%、12.2%和2.5%。我们还提供了消融研究来量化我们提出的方法中每个组件的有效性。 

---
# Multi-task Visual Grounding with Coarse-to-Fine Consistency Constraints 

**Title (ZH)**: 从粗到细一致性约束的多任务视觉定位 

**Authors**: Ming Dai, Jian Li, Jiedong Zhuang, Xian Zhang, Wankou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06710)  

**Abstract**: Multi-task visual grounding involves the simultaneous execution of localization and segmentation in images based on textual expressions. The majority of advanced methods predominantly focus on transformer-based multimodal fusion, aiming to extract robust multimodal representations. However, ambiguity between referring expression comprehension (REC) and referring image segmentation (RIS) is error-prone, leading to inconsistencies between multi-task predictions. Besides, insufficient multimodal understanding directly contributes to biased target perception. To overcome these challenges, we propose a Coarse-to-fine Consistency Constraints Visual Grounding architecture ($\text{C}^3\text{VG}$), which integrates implicit and explicit modeling approaches within a two-stage framework. Initially, query and pixel decoders are employed to generate preliminary detection and segmentation outputs, a process referred to as the Rough Semantic Perception (RSP) stage. These coarse predictions are subsequently refined through the proposed Mask-guided Interaction Module (MIM) and a novel explicit bidirectional consistency constraint loss to ensure consistent representations across tasks, which we term the Refined Consistency Interaction (RCI) stage. Furthermore, to address the challenge of insufficient multimodal understanding, we leverage pre-trained models based on visual-linguistic fusion representations. Empirical evaluations on the RefCOCO, RefCOCO+, and RefCOCOg datasets demonstrate the efficacy and soundness of $\text{C}^3\text{VG}$, which significantly outperforms state-of-the-art REC and RIS methods by a substantial margin. Code and model will be available at \url{this https URL}. 

**Abstract (ZH)**: 多任务视觉接地涉及基于文本表达在同一图像中同时执行定位和分割。大多数先进的方法主要集中在基于转换器的多模态融合上，旨在提取稳健的多模态表示。然而，识别表达理解（REC）与参考图像分割（RIS）之间的歧义容易导致多任务预测之间的不一致。此外，不充分的多模态理解直接导致目标感知的偏见。为克服这些挑战，我们提出了一种从粗糙到精细的一致性约束视觉接地架构（$\text{C}^3\text{VG}$），该架构在一个两阶段框架内集成了隐式和显式的建模方法。首先，通过查询解码器和像素解码器生成初步的检测和分割输出，这一过程称为粗糙语义感知（RSP）阶段。随后，通过提出的掩膜引导交互模块（MIM）和一种新颖的显式双向一致性约束损失来细化这些粗糙预测，以确保任务间的一致表示，我们称之为精细一致性交互（RCI）阶段。此外，为解决多模态理解不足的挑战，我们利用基于视觉语言融合表示的预训练模型。在RefCOCO、RefCOCO+和RefCOCOg数据集上的实证评估表明，$\text{C}^3\text{VG}$ 的有效性和可靠性，其性能显著优于最先进的REC和RIS方法。代码和模型将在 \url{this https URL} 提供。 

---
# Application of Vision-Language Model to Pedestrians Behavior and Scene Understanding in Autonomous Driving 

**Title (ZH)**: 将下面的论文内容或标题翻译成中文，应符合学术规范如下：

基于视觉-语言模型的行人行为识别与场景理解在自动驾驶中的应用

这一翻译简洁明了，适用于学术论文标题或摘要。 

**Authors**: Haoxiang Gao, Yu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.06680)  

**Abstract**: Autonomous driving (AD) has experienced significant improvements in recent years and achieved promising 3D detection, classification, and localization results. However, many challenges remain, e.g. semantic understanding of pedestrians' behaviors, and downstream handling for pedestrian interactions. Recent studies in applications of Large Language Models (LLM) and Vision-Language Models (VLM) have achieved promising results in scene understanding and high-level maneuver planning in diverse traffic scenarios. However, deploying the billion-parameter LLMs to vehicles requires significant computation and memory resources. In this paper, we analyzed effective knowledge distillation of semantic labels to smaller Vision networks, which can be used for the semantic representation of complex scenes for downstream decision-making for planning and control. 

**Abstract (ZH)**: 近年来，自动驾驶（AD）取得了显著的进步，实现了有前景的3D检测、分类和定位结果。然而，仍存在许多挑战，如对行人行为的语义理解以及处理行人交互问题。近年来，大型语言模型（LLM）和视觉-语言模型（VLM）在多样交通场景下的场景理解及高级机动规划方面取得了显著成果。然而，将包含十亿参数的LLM部署到车辆中需要大量计算和内存资源。本文分析了将语义标签有效知识蒸馏至较小的视觉网络的方法，这些网络可以用于复杂场景的语义表示及后续的决策规划。 

---
# Natural Language Supervision for Low-light Image Enhancement 

**Title (ZH)**: 低光照图像增强的自然语言监督方法 

**Authors**: Jiahui Tang, Kaihua Zhou, Zhijian Luo, Yueen Hou  

**Link**: [PDF](https://arxiv.org/pdf/2501.06546)  

**Abstract**: With the development of deep learning, numerous methods for low-light image enhancement (LLIE) have demonstrated remarkable performance. Mainstream LLIE methods typically learn an end-to-end mapping based on pairs of low-light and normal-light images. However, normal-light images under varying illumination conditions serve as reference images, making it difficult to define a ``perfect'' reference image This leads to the challenge of reconciling metric-oriented and visual-friendly results. Recently, many cross-modal studies have found that side information from other related modalities can guide visual representation learning. Based on this, we introduce a Natural Language Supervision (NLS) strategy, which learns feature maps from text corresponding to images, offering a general and flexible interface for describing an image under different illumination.
However, image distributions conditioned on textual descriptions are highly multimodal, which makes training difficult. To address this issue, we design a Textual Guidance Conditioning Mechanism (TCM) that incorporates the connections between image regions and sentence words, enhancing the ability to capture fine-grained cross-modal cues for images and text. This strategy not only utilizes a wider range of supervised sources, but also provides a new paradigm for LLIE based on visual and textual feature alignment. In order to effectively identify and merge features from various levels of image and textual information, we design an Information Fusion Attention (IFA) module to enhance different regions at different levels. We integrate the proposed TCM and IFA into a Natural Language Supervision network for LLIE, named NaLSuper. Finally, extensive experiments demonstrate the robustness and superior effectiveness of our proposed NaLSuper. 

**Abstract (ZH)**: 随着深度学习的发展，许多低光图像增强（LLIE）方法已经展现出卓越的效果。主流的LLIE方法通常基于低光和正常光图像的配对学习端到端的映射。然而，不同照明条件下的正常光图像作为参考图像使用，这使得难以定义一个“完美的”参考图像。这导致了基于度量和视觉友好结果之间的调和挑战。最近，许多跨模态研究表明，来自其他相关模态的辅助信息可以帮助引导视觉表示学习。基于此，我们引入了一种自然语言监督（NLS）策略，从对应图像的文字中学习特征图，提供了一个灵活的一般性接口，用于在不同照明条件下描述图像。

然而，根据文字描述条件下的图像分布高度多模态，这使得训练过程变得困难。为了解决这个问题，我们设计了一种文本引导条件机制（TCM），它结合了图像区域和句子词语之间的联系，增强了捕捉图像和文本之间细粒度跨模态线索的能力。该策略不仅可以利用监督信息的广泛范围，还提供了一种基于视觉和文本特征对齐的LLIE新范式。为了有效地识别和合并来自不同层级图像和文本信息的特征，我们设计了一个信息融合注意（IFA）模块，以在不同层级增强不同的区域。我们将提出的TCM和IFA整合到了一个名为NaLSuper的自然语言监督LLIE网络中。最后，广泛的实验验证了我们提出的NaLSuper的鲁棒性和优越效果。 

---
# Unispeaker: A Unified Approach for Multimodality-driven Speaker Generation 

**Title (ZH)**: 统一说话人生成方法：一种基于多模态的信息统一方法 

**Authors**: Zhengyan Sheng, Zhihao Du, Heng Lu, Shiliang Zhang, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2501.06394)  

**Abstract**: Recent advancements in personalized speech generation have brought synthetic speech increasingly close to the realism of target speakers' recordings, yet multimodal speaker generation remains on the rise. This paper introduces UniSpeaker, a unified approach for multimodality-driven speaker generation. Specifically, we propose a unified voice aggregator based on KV-Former, applying soft contrastive loss to map diverse voice description modalities into a shared voice space, ensuring that the generated voice aligns more closely with the input descriptions. To evaluate multimodality-driven voice control, we build the first multimodality-based voice control (MVC) benchmark, focusing on voice suitability, voice diversity, and speech quality. UniSpeaker is evaluated across five tasks using the MVC benchmark, and the experimental results demonstrate that UniSpeaker outperforms previous modality-specific models. Speech samples are available at \url{this https URL}. 

**Abstract (ZH)**: 近年来，个性化的语音生成技术取得了重大进展，合成语音越来越接近目标说话人录制的声音，但多模态说话人生成依然处于上升趋势。本文介绍了UniSpeaker，这是一种基于多模态驱动的统一说话人生成方法。具体而言，我们提出了一种基于KV-Former的统一声音聚合器，并应用软对比损失来将多种声音描述模态映射到共享的声音空间中，确保生成的声音更接近输入描述。为了评估多模态驱动的语音控制能力，我们构建了首个基于多模态的语音控制（MVC）基准，重点关注声音适用性、声音多样性和语音质量。UniSpeaker在MVC基准上进行了五个任务的评估，实验结果表明UniSpeaker优于之前模态特定的模型。相关语音样本可访问 <https://github.com/your-link-here>。 

---
# MinMo: A Multimodal Large Language Model for Seamless Voice Interaction 

**Title (ZH)**: MinMo：一种支持无缝语音交互的多模态大型语言模型 

**Authors**: Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, Yabin Li, Xiang Lv, Jiaqing Liu, Haoneng Luo, Bin Ma, Chongjia Ni, Xian Shi, Jialong Tang, Hui Wang, Hao Wang, Wen Wang, Yuxuan Wang, Yunlan Xu, Fan Yu, Zhijie Yan, Yexin Yang, Baosong Yang, Xian Yang, Guanrou Yang, Tianyu Zhao, Qinglin Zhang, Shiliang Zhang, Nan Zhao, Pei Zhang, Chong Zhang, Jinren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.06282)  

**Abstract**: Recent advancements in large language models (LLMs) and multimodal speech-text models have laid the groundwork for seamless voice interactions, enabling real-time, natural, and human-like conversations. Previous models for voice interactions are categorized as native and aligned. Native models integrate speech and text processing in one framework but struggle with issues like differing sequence lengths and insufficient pre-training. Aligned models maintain text LLM capabilities but are often limited by small datasets and a narrow focus on speech tasks. In this work, we introduce MinMo, a Multimodal Large Language Model with approximately 8B parameters for seamless voice interaction. We address the main limitations of prior aligned multimodal models. We train MinMo through multiple stages of speech-to-text alignment, text-to-speech alignment, speech-to-speech alignment, and duplex interaction alignment, on 1.4 million hours of diverse speech data and a broad range of speech tasks. After the multi-stage training, MinMo achieves state-of-the-art performance across various benchmarks for voice comprehension and generation while maintaining the capabilities of text LLMs, and also facilitates full-duplex conversation, that is, simultaneous two-way communication between the user and the system. Moreover, we propose a novel and simple voice decoder that outperforms prior models in voice generation. The enhanced instruction-following capabilities of MinMo supports controlling speech generation based on user instructions, with various nuances including emotions, dialects, and speaking rates, and mimicking specific voices. For MinMo, the speech-to-text latency is approximately 100ms, full-duplex latency is approximately 600ms in theory and 800ms in practice. The MinMo project web page is this https URL, and the code and models will be released soon. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）和多模态语音-文本模型的发展为无缝语音交互打下了基础，使其能够实现实时、自然和类人的对话。此前用于语音交互的模型主要分为两类：原生模型和对齐模型。原生模型将语音和文本处理融合在一个框架中，但面临序列长度不一致和预训练不足等问题。对齐模型保留了文本LLM的能力，但常常受限于小的数据集和范围狭窄的语音任务。在这项工作中，我们提出了一种名为MinMo的多模态大型语言模型，参数量约为8亿，旨在实现无缝语音交互。我们解决了此前对齐多模态模型的主要限制。通过多次阶段的语音到文本对齐、文本到语音对齐、语音到语音对齐以及双向交互对齐训练，MinMo在140万小时的多样语音数据和广泛的语音任务上进行了训练。经过多阶段训练后，MinMo在各种语音理解和生成基准测试中表现出最先进的性能，同时保持了文本LLM的能力，并实现了全双工对话，即用户与系统之间的双向通信。此外，我们提出了一种新的简单语音解码器，表现优于之前的模型。MinMo增强的指令遵循能力支持基于用户指令控制语音生成，包括情感、方言、说话速度等多种细微差别，并能够模拟特定的声音。对于MinMo，语音到文本的延迟约为100毫秒，理论上的全双工延迟约为600毫秒，实际应用中约为800毫秒。MinMo项目的网页地址是 <https://XXXXX>，代码和模型将在不久后发布。 

---
# Detection, Retrieval, and Explanation Unified: A Violence Detection System Based on Knowledge Graphs and GAT 

**Title (ZH)**: 统一检测、检索与解释：基于知识图谱和图注意力网络的暴力检测系统 

**Authors**: Wen-Dong Jiang, Chih-Yung Chang, Diptendu Sinha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2501.06224)  

**Abstract**: Recently, violence detection systems developed using unified multimodal models have achieved significant success and attracted widespread attention. However, most of these systems face two critical challenges: the lack of interpretability as black-box models and limited functionality, offering only classification or retrieval capabilities. To address these challenges, this paper proposes a novel interpretable violence detection system, termed the Three-in-One (TIO) System. The TIO system integrates knowledge graphs (KG) and graph attention networks (GAT) to provide three core functionalities: detection, retrieval, and explanation. Specifically, the system processes each video frame along with text descriptions generated by a large language model (LLM) for videos containing potential violent behavior. It employs ImageBind to generate high-dimensional embeddings for constructing a knowledge graph, uses GAT for reasoning, and applies lightweight time series modules to extract video embedding features. The final step connects a classifier and retriever for multi-functional outputs. The interpretability of KG enables the system to verify the reasoning process behind each output. Additionally, the paper introduces several lightweight methods to reduce the resource consumption of the TIO system and enhance its efficiency. Extensive experiments conducted on the XD-Violence and UCF-Crime datasets validate the effectiveness of the proposed system. A case study further reveals an intriguing phenomenon: as the number of bystanders increases, the occurrence of violent behavior tends to decrease. 

**Abstract (ZH)**: 近年来，使用统一多模态模型开发的暴力检测系统取得了显著的成功并引起了广泛关注。然而，这些系统大多面临着两个关键挑战：作为黑盒模型缺乏可解释性以及功能有限，只能提供分类或检索能力。为了解决这些挑战，本文提出了一种名为三位一体系统（Three-in-One, TIO）的新颖可解释暴力检测系统。TIO系统将知识图谱（KG）和图注意力网络（GAT）相结合，提供三大核心功能：检测、检索和解释。具体而言，该系统处理包含潜在暴力行为的视频帧，并结合大型语言模型（LLM）生成的文本描述。它利用ImageBind生成高维嵌入以构建知识图谱，使用GAT进行推理，并应用轻量级时间序列模块提取视频嵌入特征。最后一步是连接分类器和检索器以实现多功能输出。KG 的可解释性使系统能够验证每个输出背后的推理过程。此外，本文还引入了几种轻量级方法以降低TIO系统的资源消耗并提高其效率。在XD-Violence和UCF-Crime数据集上的广泛实验验证了所提出系统的有效性。进一步的案例研究揭示了一个有趣的现象：旁观者的数量增加时，暴力行为的发生率似乎会减少。 

---
# Multimodal semantic retrieval for product search 

**Title (ZH)**: 多模态语义检索在产品搜索中的应用 

**Authors**: Dong Liu, Esther Lopez Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2501.07365)  

**Abstract**: Semantic retrieval (also known as dense retrieval) based on textual data has been extensively studied for both web search and product search application fields, where the relevance of a query and a potential target document is computed by their dense vector representation comparison. Product image is crucial for e-commence search interactions and is a key factor for customers at product explorations. But its impact for semantic retrieval has not been well studied yet. In this research, we build a multimodal representation for product items in e-commerece search in contrast to pure-text representation of products, and investigate the impact of such representations. The models are developed and evaluated on e-commerce datasets. We demonstrate that a multimodal representation scheme for a product can show improvement either on purchase recall or relevance accuracy in semantic retrieval. Additionally, we provide numerical analysis for exclusive matches retrieved by a multimodal semantic retrieval model versus a text-only semantic retrieval model, to demonstrate the validation of multimodal solutions. 

**Abstract (ZH)**: 基于文本数据的语义检索（也称为密集检索）在Web搜索和电子商务搜索等领域得到了广泛研究，其中查询和潜在目标文档的相关性通过它们的密集向量表示进行比较。产品图像对于电子商务搜索交互至关重要，是客户进行产品探索的重要因素。然而，产品图像对语义检索的影响尚未得到充分研究。在本研究中，我们构建了一种多模态表示方法，用于电子商务搜索中的产品项目，相比之下，传统的产品表示方法仅基于纯文本。我们研究了这种表示方法的影响，并在电子商务数据集上开发和评估了相应的模型。研究表明，产品中的多模态表示方案可以在语义检索中提高购买召回率或相关性准确性。此外，我们提供了唯一匹配项的数值分析，比较了多模态语义检索模型与仅基于文本的语义检索模型的表现，以验证多模态解决方案的有效性。 

---
# Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation 

**Title (ZH)**: 面向微视频推荐的元学习驱动的动态多模态融合 

**Authors**: Han Liu, Yinwei Wei, Fan Liu, Wenjie Wang, Liqiang Nie, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.07110)  

**Abstract**: Multimodal information (e.g., visual, acoustic, and textual) has been widely used to enhance representation learning for micro-video recommendation. For integrating multimodal information into a joint representation of micro-video, multimodal fusion plays a vital role in the existing micro-video recommendation approaches. However, the static multimodal fusion used in previous studies is insufficient to model the various relationships among multimodal information of different micro-videos. In this paper, we develop a novel meta-learning-based multimodal fusion framework called Meta Multimodal Fusion (MetaMMF), which dynamically assigns parameters to the multimodal fusion function for each micro-video during its representation learning. Specifically, MetaMMF regards the multimodal fusion of each micro-video as an independent task. Based on the meta information extracted from the multimodal features of the input task, MetaMMF parameterizes a neural network as the item-specific fusion function via a meta learner. We perform extensive experiments on three benchmark datasets, demonstrating the significant improvements over several state-of-the-art multimodal recommendation models, like MMGCN, LATTICE, and InvRL. Furthermore, we lighten our model by adopting canonical polyadic decomposition to improve the training efficiency, and validate its effectiveness through experimental results. Codes are available at this https URL. 

**Abstract (ZH)**: 多模态信息（例如视觉、声学和文本）广泛用于增强微视频推荐中的表示学习。在现有的微视频推荐方法中，多模态融合对于将多模态信息整合到微视频的联合表示中起着至关重要的作用。然而，先前研究中使用的静态多模态融合不足以建模不同微视频之间多模态信息的各种关系。在本文中，我们提出了一种新颖的基于元学习的多模态融合框架，称为Meta多模态融合（MetaMMF），该框架在每个微视频的表示学习过程中动态为多模态融合函数分配参数。具体而言，MetaMMF 将每个微视频的多模态融合视为独立的任务。基于输入任务的多模态特征提取的元信息，MetaMMF 通过元学习器将一个神经网络参数化为特定于项目的融合函数。我们对三个基准数据集执行了大量实验，结果表明，MetaMMF 在多个最先进的多模态推荐模型（如MMGCN、LATTICE 和 InvRL）上显著提升了性能。此外，我们通过采用典范多项式分解来简化模型，以提高训练效率，并通过实验结果验证其有效性。代码可从以下链接获取：[此处替换为链接] 

---
# Imagine while Reasoning in Space: Multimodal Visualization-of-Thought 

**Title (ZH)**: 在空间推理中的多模态可视化思维：Imagine while Reasoning in Space 

**Authors**: Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulić, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.07542)  

**Abstract**: Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Yet, it struggles in complex spatial reasoning tasks. Nonetheless, human cognition extends beyond language alone, enabling the remarkable capability to think in both words and images. Inspired by this mechanism, we propose a new reasoning paradigm, Multimodal Visualization-of-Thought (MVoT). It enables visual thinking in MLLMs by generating image visualizations of their reasoning traces. To ensure high-quality visualization, we introduce token discrepancy loss into autoregressive MLLMs. This innovation significantly improves both visual coherence and fidelity. We validate this approach through several dynamic spatial reasoning tasks. Experimental results reveal that MVoT demonstrates competitive performance across tasks. Moreover, it exhibits robust and reliable improvements in the most challenging scenarios where CoT fails. Ultimately, MVoT establishes new possibilities for complex reasoning tasks where visual thinking can effectively complement verbal reasoning. 

**Abstract (ZH)**: 链式推理（CoT）提示在增强大规模语言模型（LLMs）和多模态大规模语言模型（MLLMs）的复杂推理能力方面已被证明非常有效。然而，它在复杂的空间推理任务中表现不佳。尽管如此，人类认知不仅依赖语言，还能够同时进行语言和图像的思维活动。受这一机制的启发，我们提出了一种新的推理范式——多模态思维可视化的（MVoT）。通过生成图像化的推理痕迹，MVoT允许MLLMs进行图像思维。为了保证高质量的可视化效果，我们引入了标记差异损失（token discrepancy loss）到自回归MLLMs中。这一创新显著提高了图像的连贯性和准确性。通过多种动态空间推理任务的验证，实验结果表明，MVoT在多个任务中表现出竞争力。此外，在CoT失败的最具有挑战性的场景中，MVoT也表现出稳健可靠的改进。最终，MVoT为复杂推理任务开辟了新的可能性，在这些任务中，图像思维能够有效地补充言语推理。 

---
# Boosting Text-To-Image Generation via Multilingual Prompting in Large Multimodal Models 

**Title (ZH)**: 通过多语言提示在大型多模态模型中的应用以增强文本到图像生成 

**Authors**: Yongyu Mu, Hengyu Li, Junxin Wang, Xiaoxuan Zhou, Chenglong Wang, Yingfeng Luo, Qiaozhi He, Tong Xiao, Guocheng Chen, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07086)  

**Abstract**: Previous work on augmenting large multimodal models (LMMs) for text-to-image (T2I) generation has focused on enriching the input space of in-context learning (ICL). This includes providing a few demonstrations and optimizing image descriptions to be more detailed and logical. However, as demand for more complex and flexible image descriptions grows, enhancing comprehension of input text within the ICL paradigm remains a critical yet underexplored area. In this work, we extend this line of research by constructing parallel multilingual prompts aimed at harnessing the multilingual capabilities of LMMs. More specifically, we translate the input text into several languages and provide the models with both the original text and the translations. Experiments on two LMMs across 3 benchmarks show that our method, PMT2I, achieves superior performance in general, compositional, and fine-grained assessments, especially in human preference alignment. Additionally, with its advantage of generating more diverse images, PMT2I significantly outperforms baseline prompts when incorporated with reranking methods. Our code and parallel multilingual data can be found at this https URL. 

**Abstract (ZH)**: 以往关于增强大型多模态模型（LMMs）以用于文本到图像（T2I）生成的研究主要集中在丰富上下文学习（ICL）的输入空间。这包括提供少量示例和优化图像描述以使其更详细和合理。然而，随着对更复杂和灵活的图像描述的需求增长，如何在ICL范式中增强输入文本的理解仍然是一个关键但尚未充分探索的领域。在本工作中，我们通过构建平行多语言提示来扩展这一研究方向，旨在利用LMMs的多语言能力。具体而言，我们将输入文本翻译成多种语言，并向模型提供原文和翻译文本。在两个LMMs的三个基准测试中进行的实验表明，我们的方法（PMT2I）在通用性、组合性和细粒度评估中均表现出更优的性能，尤其是在人类偏好一致性方面。此外，由于PMT2I能够生成更多样化的图像，因此当与排序方法结合使用时，其性能显著优于基线提示。我们的代码和平行多语言数据可以通过以下链接查询：this https URL。 

---
# SST-EM: Advanced Metrics for Evaluating Semantic, Spatial and Temporal Aspects in Video Editing 

**Title (ZH)**: SST-EM：评估视频编辑中语义、空间和时间方面的新颖度量方法 

**Authors**: Varun Biyyala, Bharat Chanderprakash Kathuria, Jialu Li, Youshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07554)  

**Abstract**: Video editing models have advanced significantly, but evaluating their performance remains challenging. Traditional metrics, such as CLIP text and image scores, often fall short: text scores are limited by inadequate training data and hierarchical dependencies, while image scores fail to assess temporal consistency. We present SST-EM (Semantic, Spatial, and Temporal Evaluation Metric), a novel evaluation framework that leverages modern Vision-Language Models (VLMs), Object Detection, and Temporal Consistency checks. SST-EM comprises four components: (1) semantic extraction from frames using a VLM, (2) primary object tracking with Object Detection, (3) focused object refinement via an LLM agent, and (4) temporal consistency assessment using a Vision Transformer (ViT). These components are integrated into a unified metric with weights derived from human evaluations and regression analysis. The name SST-EM reflects its focus on Semantic, Spatial, and Temporal aspects of video evaluation. SST-EM provides a comprehensive evaluation of semantic fidelity and temporal smoothness in video editing. The source code is available in the \textbf{\href{this https URL}{GitHub Repository}}. 

**Abstract (ZH)**: 视频编辑模型的进步显著，但对其性能的评价仍然具有挑战性。传统的评估指标，如CLIP的文字和图像评分，常常不尽如人意：文字评分受限于训练数据的不足和层次依赖性，而图像评分无法评估时间连续性。我们提出了SST-EM（语义、空间和时间评估指标）这一新型评估框架，该框架结合了现代视觉语言模型（VLMs）、目标检测和时间一致性检查。SST-EM 包含四个组成部分：（1）使用VLM从帧中提取语义信息，（2）使用目标检测进行主要目标跟踪，（3）通过LLM代理进行聚焦目标细化，以及（4）使用视觉变换器（ViT）进行时间一致性评估。这些组件通过结合人类评价和回归分析得出的权重，集成到一个统一的评估指标中。命名SST-EM反映了其侧重于视频评估的语义、空间和时间方面。SST-EM 提供了对视频编辑中语义保真度和时间平滑度的综合评估。源代码已发布在\textbf{\href{https://github.com/example-repository}{GitHub Repository}}中。 

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
# LEO: Boosting Mixture of Vision Encoders for Multimodal Large Language Models 

**Title (ZH)**: LEO：增强视觉编码器混合模型以提升多模态大型语言模型 

**Authors**: Mozhgan Nasr Azadani, James Riddell, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2501.06986)  

**Abstract**: Enhanced visual understanding serves as a cornerstone for multimodal large language models (MLLMs). Recent hybrid MLLMs incorporate a mixture of vision experts to address the limitations of using a single vision encoder and excessively long visual tokens. Despite the progress of these MLLMs, a research gap remains in effectively integrating diverse vision encoders. This work explores fusion strategies of visual tokens for hybrid MLLMs, leading to the design of LEO, a novel MLLM with a dual-branch vision encoder framework that incorporates a post-adaptation fusion strategy and adaptive tiling: for each segmented tile of the input images, LEO sequentially interleaves the visual tokens from its two vision encoders. Extensive evaluation across 13 vision-language benchmarks reveals that LEO outperforms state-of-the-art open-source MLLMs and hybrid MLLMs on the majority of tasks. Furthermore, we show that LEO can be adapted to the specialized domain of autonomous driving without altering the model architecture or training recipe, achieving competitive performance compared to existing baselines. The code and model will be publicly available. 

**Abstract (ZH)**: 增强的视觉理解是多模态大规模语言模型（MLLMs）的基石。最近的混合MLLMs通过结合多种视觉专家来弥补单一视觉编码器和过长视觉标记的局限性。尽管这些MLLMs取得了进展，但在有效整合多样化的视觉编码器方面仍存在研究缺口。本研究探讨了混合MLLMs中视觉标记融合策略的设计，提出了一种新的MLLM——LEO，它采用了一种双分支视觉编码器框架，并结合了后适应融合策略和自适应切片：对于输入图像中的每个分割切片，LEO 依次交错其两个视觉编码器生成的视觉标记。在13个视觉-语言基准上的广泛评估表明，LEO 在大多数任务上优于最先进的开源MLLMs和混合MLLMs。此外，我们展示了LEO 可以适应自动驾驶这一特定领域，无需改变模型架构或训练方案，便能实现与现有基线相当的性能。代码和模型将公开可用。 

---
# Fitting Different Interactive Information: Joint Classification of Emotion and Intention 

**Title (ZH)**: 适应不同交互信息的建模：情感和意图的联合分类 

**Authors**: Xinger Li, Zhiqiang Zhong, Bo Huang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06215)  

**Abstract**: This paper is the first-place solution for ICASSP MEIJU@2025 Track I, which focuses on low-resource multimodal emotion and intention recognition. How to effectively utilize a large amount of unlabeled data, while ensuring the mutual promotion of different difficulty levels tasks in the interaction stage, these two points become the key to the competition. In this paper, pseudo-label labeling is carried out on the model trained with labeled data, and samples with high confidence and their labels are selected to alleviate the problem of low resources. At the same time, the characteristic of easy represented ability of intention recognition found in the experiment is used to make mutually promote with emotion recognition under different attention heads, and higher performance of intention recognition is achieved through fusion. Finally, under the refined processing data, we achieve the score of 0.5532 in the Test set, and win the championship of the track. 

**Abstract (ZH)**: 本文是ICASSP MEIJU@2025 Track I竞赛的第一名解决方案，该竞赛专注于低资源多模态情绪和意图识别。如何有效地利用大量无标签数据，并在交互阶段确保不同难度任务之间的相互促进，成为了竞赛的关键所在。本文通过使用标记数据训练的模型进行伪标签标注，从中选择置信度高的样本及其标签，以此来缓解资源不足的问题。同时，实验中发现的意图识别容易表示的特点被用于在不同注意力头下与情绪识别相互促进，从而通过融合提高了意图识别的表现。最后，在经过精细处理的数据上，我们在测试集上取得了0.5532的分数，并在赛道中赢得了冠军。 

---
