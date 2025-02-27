# LiGT: Layout-infused Generative Transformer for Visual Question Answering on Vietnamese Receipts 

**Title (ZH)**: LiGT：融合布局信息的生成变压器模型在越南收据上的视觉问答 

**Authors**: Thanh-Phong Le, Trung Le Chi Phan, Nghia Hieu Nguyen, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19202)  

**Abstract**: \textbf{Purpose:} Document Visual Question Answering (document VQA) challenges multimodal systems to holistically handle textual, layout, and visual modalities to provide appropriate answers. Document VQA has gained popularity in recent years due to the increasing amount of documents and the high demand for digitization. Nonetheless, most of document VQA datasets are developed in high-resource languages such as English.
\textbf{Methods:} In this paper, we present ReceiptVQA (\textbf{Receipt} \textbf{V}isual \textbf{Q}uestion \textbf{A}nswering), the initial large-scale document VQA dataset in Vietnamese dedicated to receipts, a document kind with high commercial potentials. The dataset encompasses \textbf{9,000+} receipt images and \textbf{60,000+} manually annotated question-answer pairs. In addition to our study, we introduce LiGT (\textbf{L}ayout-\textbf{i}nfused \textbf{G}enerative \textbf{T}ransformer), a layout-aware encoder-decoder architecture designed to leverage embedding layers of language models to operate layout embeddings, minimizing the use of additional neural modules.
\textbf{Results:} Experiments on ReceiptVQA show that our architecture yielded promising performance, achieving competitive results compared with outstanding baselines. Furthermore, throughout analyzing experimental results, we found evident patterns that employing encoder-only model architectures has considerable disadvantages in comparison to architectures that can generate answers. We also observed that it is necessary to combine multiple modalities to tackle our dataset, despite the critical role of semantic understanding from language models.
\textbf{Conclusion:} We hope that our work will encourage and facilitate future development in Vietnamese document VQA, contributing to a diverse multimodal research community in the Vietnamese language. 

**Abstract (ZH)**: **目的：** 文档视觉问答（Document Visual Question Answering, Document VQA）挑战多模态系统同时处理文本、布局和视觉模态以提供适当答案。近年来，由于文档数量的增加和对数字化的高需求，Document VQA变得越来越受欢迎。然而，大多数Document VQA数据集都是在诸如英语等高资源语言中开发的。
**方法：** 在本文中，我们提出了ReceiptVQA（收据视觉问答），这是首个针对收据的大型Document VQA数据集，收据是一种具有高商业潜力的文档类型。该数据集包含**9,000多**张收据图像和**60,000多**个手动标注的问题-答案对。除了我们的研究之外，我们还引入了LiGT（布局-指导生成转换器），这是一种布局感知的编码器-解码器架构，旨在利用语言模型的嵌入层来操作布局嵌入，以最小化额外神经模块的使用。
**结果：** 收据上的实验结果表明，我们的架构表现出了令人鼓舞的效果，与优秀的基础模型相比达到了竞争力的结果。通过对实验结果的进一步分析，我们发现采用仅编码器架构的模型具有相当大的劣势，相比之下，能够生成答案的架构具有明显的优势。我们还观察到，在处理我们的数据集时，必须结合多种模态信息，尽管语义理解的作用仍然至关重要。
**结论：** 我们希望我们的工作能够鼓励并促进越南文档视觉问答的发展，为越南语语言的多样化多模态研究社区做出贡献。 

---
# What are Foundation Models Cooking in the Post-Soviet World? 

**Title (ZH)**: 后苏联世界中，基础模型在烹饪些什么？ 

**Authors**: Anton Lavrouk, Tarek Naous, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18583)  

**Abstract**: The culture of the Post-Soviet states is complex, shaped by a turbulent history that continues to influence current events. In this study, we investigate the Post-Soviet cultural food knowledge of foundation models by constructing BORSch, a multimodal dataset encompassing 1147 and 823 dishes in the Russian and Ukrainian languages, centered around the Post-Soviet region. We demonstrate that leading models struggle to correctly identify the origins of dishes from Post-Soviet nations in both text-only and multimodal Question Answering (QA), instead over-predicting countries linked to the language the question is asked in. Through analysis of pretraining data, we show that these results can be explained by misleading dish-origin co-occurrences, along with linguistic phenomena such as Russian-Ukrainian code mixing. Finally, to move beyond QA-based assessments, we test models' abilities to produce accurate visual descriptions of dishes. The weak correlation between this task and QA suggests that QA alone may be insufficient as an evaluation of cultural understanding. To foster further research, we will make BORSch publicly available at this https URL. 

**Abstract (ZH)**: 后苏联国家的文化具有复杂性，深受动荡历史的影响，这种历史 continues to 影响当前事件。在本研究中，我们通过构建一个涵盖1147道俄语菜肴和823道乌克兰语菜肴的多模态数据集 BORSch，来考察基础模型对后苏联地区文化饮食知识的理解。这个数据集以后苏联地区为中心。我们证明，领先模型在文本-only 和多模态问答（QA）中都难以正确识别来自后苏联国家的菜肴起源，反而过度预测与问题语言相关的国家。通过分析预训练数据，我们表明，这些结果可以由误导性菜肴来源共现现象以及如俄语-乌克兰语代码混用这类语言现象来解释。最终，为超越问答评估，我们测试了模型生成菜肴准确视觉描述的能力。任务与问答之间的弱相关性提示我们，仅问答可能不足以评估文化理解。为了促进进一步的研究，我们将在以下链接上公开发布 BORSch 数据集：[此处提供链接]。 

---
# ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models 

**Title (ZH)**: ImageChain：推动多模态大型语言模型中序列图像到文本推理的发展 

**Authors**: Danae Sánchez Villegas, Ingo Ziegler, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2502.19409)  

**Abstract**: Reasoning over sequences of images remains a challenge for multimodal large language models (MLLMs). While recent models incorporate multi-image data during pre-training, they still struggle to recognize sequential structures, often treating images independently. This work introduces ImageChain, a framework that enhances MLLMs with sequential reasoning capabilities over image data by modeling visual sequences as a multi-turn conversation. In ImageChain, images are interleaved with corresponding textual descriptions to form a controlled dialogue that explicitly captures temporal dependencies and narrative progression. Our method optimizes for the task of next-scene description, where the model generates a context-aware description of an upcoming scene based on preceding visual and textual cues. We demonstrate that our approach improves performance on the next-scene description task -- achieving an average improvement from 3.7% to 19% in SimRate, a metric that quantifies semantic similarity to human-annotated ground truths. Moreover, ImageChain achieves robust zero-shot out-of-domain performance in applications ranging from comics to robotics. Extensive experiments validate that instruction-tuning in a multimodal, multi-turn conversation design is key to bridging the gap between static image understanding and temporally-aware reasoning. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在处理图像序列时依然面临挑战。尽管最近的模型在预训练过程中引入了多幅图像数据，但在识别序列结构方面仍存在问题，往往将图像独立处理。本文提出了一种名为ImageChain的框架，通过将视觉序列建模为多轮对话，增强MLLMs的序列推理能力。在ImageChain中，图像与相应的文本描述交织在一起，形成一个受控的对话，明确捕获时间依赖性和叙事进展。我们的方法旨在优化下一个场景描述任务，在该任务中，模型基于前一个视觉和文本提示生成具有上下文感知的场景描述。我们展示了我们的方法在下一个场景描述任务上的性能提升——在SimRate（衡量语义相似度的指标）上平均提高了15.3%。此外，ImageChain在从漫画到机器人学等多个跨域应用中实现了稳健的零样本表现。大量实验验证了在多模态多轮对话设计中进行指令调优是弥合静态图像理解和时序推理之间差距的关键。 

---
# TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding 

**Title (ZH)**: 《定理解释代理：面向多模态定理理解的解释方法》 

**Authors**: Max Ku, Thomas Chong, Jonathan Leung, Krish Shah, Alvin Yu, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19400)  

**Abstract**: Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed long-form videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations. 

**Abstract (ZH)**: 理解特定领域的定理通常不仅需要基于文本的推理，有效的沟通还需要通过结构化的视觉解释来深化理解。虽然大型语言模型（LLMs）在基于文本的定理推理方面表现出色，但生成连贯且有教育意义的视觉解释仍然是一个开放的挑战。在本工作中，我们引入了TheoremExplainAgent，这是一种使用Manim动画生成长格式定理解释视频（超过5分钟）的方法。为了系统性地评估多模态定理解释，我们提出了TheoremExplainBench，这是一个涵盖240个定理（涉及多个STEM学科）的基准，同时附有5个自动化评估指标。我们的结果显示，有意识的规划对于生成详细长视频至关重要，o3-mini代理的成功率为93.8%，总体评分为0.77。然而，我们的定量和定性研究显示，大多数生成的视频在视觉元素布局方面存在一些小问题。此外，多模态解释揭示了基于文本的解释无法揭示的深层次推理缺陷，突显了多模态解释的重要性。 

---
# M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performance 

**Title (ZH)**: M2-omni：面向全面模态支持的竞争力性能提升的全栈多模态大语言模型 

**Authors**: Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-WeiChang, Jingdong Chen, Ming Yang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18778)  

**Abstract**: We present M2-omni, a cutting-edge, open-source omni-MLLM that achieves competitive performance to GPT-4o. M2-omni employs a unified multimodal sequence modeling framework, which empowers Large Language Models(LLMs) to acquire comprehensive cross-modal understanding and generation capabilities. Specifically, M2-omni can process arbitrary combinations of audio, video, image, and text modalities as input, generating multimodal sequences interleaving with audio, image, or text outputs, thereby enabling an advanced and interactive real-time experience. The training of such an omni-MLLM is challenged by significant disparities in data quantity and convergence rates across modalities. To address these challenges, we propose a step balance strategy during pre-training to handle the quantity disparities in modality-specific data. Additionally, a dynamically adaptive balance strategy is introduced during the instruction tuning stage to synchronize the modality-wise training progress, ensuring optimal convergence. Notably, we prioritize preserving strong performance on pure text tasks to maintain the robustness of M2-omni's language understanding capability throughout the training process. To our best knowledge, M2-omni is currently a very competitive open-source model to GPT-4o, characterized by its comprehensive modality and task support, as well as its exceptional performance. We expect M2-omni will advance the development of omni-MLLMs, thus facilitating future research in this domain. 

**Abstract (ZH)**: 我们介绍了一种名为M2-omni的前沿开源全模态大语言模型，其性能媲美GPT-4o。M2-omni采用了一种统一的多模态序列建模框架，赋予大语言模型（LLMs）全面跨模态的理解和生成能力。具体而言，M2-omni能够处理任意混合的音频、视频、图像和文本模态输入，生成交织有音频、图像或文本输出的多模态序列，从而提供高级且互动的实时体验。训练这种全模态大语言模型面临着各模态间数据量差异显著以及收敛率不同的挑战。为应对这些挑战，我们在预训练阶段提出了步长平衡策略，以处理特定模态数据量的差异。此外，在指令调优阶段引入了动态自适应平衡策略，以同步各模态的训练进度，确保最优的收敛性。值得一提的是，在整个训练过程中，我们优先保持对纯文本任务的强大性能，以确保M2-omni的语言理解能力的稳健性。据我们所知，M2-omni目前是非常有竞争力的开源模型，具备全面支持模态和任务的特点，以及卓越的性能。我们期望M2-omni将进一步推动全模态大语言模型的发展，从而促进该领域的未来研究。 

---
# Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation 

**Title (ZH)**: 与大脑对话：使用大规模语言模型作为代理模型构建大脑语义表示 

**Authors**: Xin Liu, Ziyue Zhang, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18725)  

**Abstract**: Traditional psychological experiments utilizing naturalistic stimuli face challenges in manual annotation and ecological validity. To address this, we introduce a novel paradigm leveraging multimodal large language models (LLMs) as proxies to extract rich semantic information from naturalistic images through a Visual Question Answering (VQA) strategy for analyzing human visual semantic representation. LLM-derived representations successfully predict established neural activity patterns measured by fMRI (e.g., faces, buildings), validating its feasibility and revealing hierarchical semantic organization across cortical regions. A brain semantic network constructed from LLM-derived representations identifies meaningful clusters reflecting functional and contextual associations. This innovative methodology offers a powerful solution for investigating brain semantic organization with naturalistic stimuli, overcoming limitations of traditional annotation methods and paving the way for more ecologically valid explorations of human cognition. 

**Abstract (ZH)**: 传统的利用自然场景刺激的心理学实验在手动标注和生态效度方面面临挑战。为了解决这些问题，我们引入了一种新的范式，利用多模态大规模语言模型（LLMs）作为代理，通过视觉问答（VQA）策略从自然图像中提取丰富的语义信息，以分析人类视觉语义表征。由LLM衍生的表征能够成功预测通过fMRI测量的已确立的神经活动模式（例如，面孔、建筑物），这验证了其可行性，并揭示了跨皮层区域的层次语义组织。从LLM衍生表征构建的大脑语义网络识别出反映功能和上下文关联的有意义的聚类。这种创新的方法论为利用自然场景刺激研究大脑语义组织提供了有力的解决方案，克服了传统标注方法的局限性，并为更生态有效的探索人类认知开辟了道路。 

---
# MDE: Modality Discrimination Enhancement for Multi-modal Recommendation 

**Title (ZH)**: MDE：多模态推荐中的模态鉴别增强 

**Authors**: Hang Zhou, Yucheng Wang, Huijing Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18481)  

**Abstract**: Multi-modal recommendation systems aim to enhance performance by integrating an item's content features across various modalities with user behavior data. Effective utilization of features from different modalities requires addressing two challenges: preserving semantic commonality across modalities (modality-shared) and capturing unique characteristics for each modality (modality-specific). Most existing approaches focus on aligning feature spaces across modalities, which helps represent modality-shared features. However, modality-specific distinctions are often neglected, especially when there are significant semantic variations between modalities. To address this, we propose a Modality Distinctiveness Enhancement (MDE) framework that prioritizes extracting modality-specific information to improve recommendation accuracy while maintaining shared features. MDE enhances differences across modalities through a novel multi-modal fusion module and introduces a node-level trade-off mechanism to balance cross-modal alignment and differentiation. Extensive experiments on three public datasets show that our approach significantly outperforms other state-of-the-art methods, demonstrating the effectiveness of jointly considering modality-shared and modality-specific features. 

**Abstract (ZH)**: 多模态推荐系统旨在通过综合项目内容特征（跨多种模态）与用户行为数据来提升性能。有效利用不同模态的特征需要解决两个挑战：保留模态间的语义一致性（跨模态共享）和捕捉每个模态的独特特性（模态特定）。现有的大多数方法侧重于在模态间对齐特征空间，从而有助于表示跨模态共享特征。然而，模态特定的差异往往被忽视，尤其是在模态之间存在显著语义差异时。为此，我们提出了一个模态差异增强（MDE）框架，该框架优先提取模态特定的信息以提高推荐准确性，同时保持共享特征。MDE 通过一个新的多模态融合模块增强不同模态之间的差异，并引入节点级别的权衡机制来平衡跨模态对齐与差异化。在三个公开数据集上的广泛实验表明，我们的方法显著优于其他最先进的方法，证明了同时考虑模态共享和模态特定特征的有效性。 

---
# A Comprehensive Survey on Composed Image Retrieval 

**Title (ZH)**: 全面综述合成图像检索 

**Authors**: Xuemeng Song, Haoqiang Lin, Haokun Wen, Bohan Hou, Mingzhu Xu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18495)  

**Abstract**: Composed Image Retrieval (CIR) is an emerging yet challenging task that allows users to search for target images using a multimodal query, comprising a reference image and a modification text specifying the user's desired changes to the reference image. Given its significant academic and practical value, CIR has become a rapidly growing area of interest in the computer vision and machine learning communities, particularly with the advances in deep learning. To the best of our knowledge, there is currently no comprehensive review of CIR to provide a timely overview of this field. Therefore, we synthesize insights from over 120 publications in top conferences and journals, including ACM TOIS, SIGIR, and CVPR In particular, we systematically categorize existing supervised CIR and zero-shot CIR models using a fine-grained taxonomy. For a comprehensive review, we also briefly discuss approaches for tasks closely related to CIR, such as attribute-based CIR and dialog-based CIR. Additionally, we summarize benchmark datasets for evaluation and analyze existing supervised and zero-shot CIR methods by comparing experimental results across multiple datasets. Furthermore, we present promising future directions in this field, offering practical insights for researchers interested in further exploration. 

**Abstract (ZH)**: 合成图像检索（CIR）是一项新兴且具有挑战性的任务，允许用户使用包含参考图像和修改文本的多模态查询来搜索目标图像。其中，修改文本指定了用户希望对参考图像进行的更改。鉴于其在学术和实践方面的重大价值，CIR 已成为计算机视觉和机器学习社区的迅速发展的研究领域，特别是在深度学习的推动下。据我们所知，目前尚未有关于 CIR 的全面回顾，以提供对该领域的及时概述。因此，我们综合了超过 120 篇发表在顶级会议和期刊上的文献的见解，包括 ACM TOIS、SIGIR 和 CVPR。特别是在此过程中，我们系统地使用细粒度分类法对现有的监督 CIR 和零样本 CIR 模型进行了分类。为了进行全面回顾，我们还简要讨论了与 CIR 密切相关的任务，如基于属性的 CIR 和基于对话的 CIR。此外，我们总结了用于评估的基准数据集，并通过多个数据集的实验结果比较分析现有的监督和零样本 CIR 方法。最后，我们提出了该领域的未来发展方向，为对该领域进一步探索感兴趣的科研人员提供了实用建议。 

---
# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models 

**Title (ZH)**: 《嗨，机器人：基于层次视觉-语言-行动模型的开放指令跟随》 

**Authors**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19417)  

**Abstract**: Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping. 

**Abstract (ZH)**: 能够在开放环境中执行多种不同任务的通用机器人不仅需要推理出完成目标所需的步骤，还必须能够处理复杂指令、提示以及在任务执行过程中提供的反馈。复杂的指令（例如，“能为我做一个素食三明治吗？”或“我不喜欢那个”）不仅需要执行个体步骤的能力，还需要将复杂的命令和反馈置于物理世界中。在这项工作中，我们描述了一个利用视觉-语言模型分层结构的系统，首先通过推理复杂的提示和用户反馈来推导出完成任务的最优下一步，然后通过低级动作执行该步骤。与仅能执行简单指令（如“拿起杯子”）的方法不同，我们的系统能够通过复杂的提示进行推理，并在任务执行过程中整合环境反馈（如“那不是垃圾”）。我们将在三个不同的机器人平台上评估该系统，包括单臂机器人、双臂机器人和双臂移动机器人，展示了其完成清理杂乱的桌子、制作三明治和采购杂货等任务的能力。 

---
# Multi-modal Contrastive Learning for Tumor-specific Missing Modality Synthesis 

**Title (ZH)**: 多模态对比学习在肿瘤特异性缺失模态合成中的应用 

**Authors**: Minjoo Lim, Bogyeong Kang, Tae-Eui Kam  

**Link**: [PDF](https://arxiv.org/pdf/2502.19390)  

**Abstract**: Multi-modal magnetic resonance imaging (MRI) is essential for providing complementary information about brain anatomy and pathology, leading to more accurate diagnoses. However, obtaining high-quality multi-modal MRI in a clinical setting is difficult due to factors such as time constraints, high costs, and patient movement artifacts. To overcome this difficulty, there is increasing interest in developing generative models that can synthesize missing target modality images from the available source ones. Therefore, we design a generative model for missing MRI that integrates multi-modal contrastive learning with a focus on critical tumor regions. Specifically, we integrate multi-modal contrastive learning, tailored for multiple source modalities, and enhance its effectiveness by selecting features based on entropy during the contrastive learning process. Additionally, our network not only generates the missing target modality images but also predicts segmentation outputs, simultaneously. This approach improves the generator's capability to precisely generate tumor regions, ultimately improving performance in downstream segmentation tasks. By leveraging a combination of contrastive, segmentation, and additional self-representation losses, our model effectively reflects target-specific information and generate high-quality target images. Consequently, our results in the Brain MR Image Synthesis challenge demonstrate that the proposed model excelled in generating the missing modality. 

**Abstract (ZH)**: 多模态磁共振成像（MRI）对于提供关于大脑解剖结构和病理的互补信息至关重要，有助于更准确的诊断。然而，在临床环境中获得高质量的多模态MRI面临时间限制、高成本和患者运动伪影等挑战。为了克服这些困难，越来越多的研究兴趣集中在开发生成模型，可以从现有的模态中合成缺失的目标模态图像。因此，我们设计了一个生成模型，将多模态对比学习与关键肿瘤区域的聚焦相结合。具体来说，我们将针对多种源模态定制的多模态对比学习与对比学习过程中基于熵的选择特征相结合，以增强其有效性。此外，我们的网络不仅生成缺失的目标模态图像，还同时预测分割输出。这种做法提高了生成器精确生成肿瘤区域的能力，最终提高了下游分割任务的性能。通过利用对比、分割和附加自我表示损失的组合，我们的模型有效地反映了目标特定的信息，生成高质量的目标图像。因此，我们在Brain MR图像合成挑战中的结果表明，所提出的模型在生成缺失的模态方面表现优异。 

---
# Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation 

**Title (ZH)**: 基于注意力引导的CLIP与SAM集成方法在机器人操作中实现精确对象遮罩 

**Authors**: Muhammad A. Muttaqien, Tomohiro Motoda, Ryo Hanai, Domae Yukiyasu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18842)  

**Abstract**: This paper introduces a novel pipeline to enhance the precision of object masking for robotic manipulation within the specific domain of masking products in convenience stores. The approach integrates two advanced AI models, CLIP and SAM, focusing on their synergistic combination and the effective use of multimodal data (image and text). Emphasis is placed on utilizing gradient-based attention mechanisms and customized datasets to fine-tune performance. While CLIP, SAM, and Grad- CAM are established components, their integration within this structured pipeline represents a significant contribution to the field. The resulting segmented masks, generated through this combined approach, can be effectively utilized as inputs for robotic systems, enabling more precise and adaptive object manipulation in the context of convenience store products. 

**Abstract (ZH)**: 本文介绍了一种新颖的工作流程，旨在提高机器人在便利店产品掩码中的操作精度。该方法整合了两种先进的AI模型——CLIP和SAM，重点关注它们的协同作用及其对多模态数据（图像和文本）的有效利用。文中强调了利用基于梯度的注意力机制和定制数据集以优化性能。尽管CLIP、SAM和Grad-CAM已经在各自领域内被广泛应用，但它们在这结构化工作流程中的整合为该领域做出了重要的贡献。通过这种综合方法生成的分割掩码可作为机器人系统输入，使机器人能在便利店产品操作中实现更精确和适应性的对象操作。 

---
# Cross-Modality Investigation on WESAD Stress Classification 

**Title (ZH)**: 跨模态研究在WESAD压力分类中的应用 

**Authors**: Eric Oliver, Sagnik Dakshit  

**Link**: [PDF](https://arxiv.org/pdf/2502.18733)  

**Abstract**: Deep learning's growing prevalence has driven its widespread use in healthcare, where AI and sensor advancements enhance diagnosis, treatment, and monitoring. In mobile health, AI-powered tools enable early diagnosis and continuous monitoring of conditions like stress. Wearable technologies and multimodal physiological data have made stress detection increasingly viable, but model efficacy depends on data quality, quantity, and modality. This study develops transformer models for stress detection using the WESAD dataset, training on electrocardiograms (ECG), electrodermal activity (EDA), electromyography (EMG), respiration rate (RESP), temperature (TEMP), and 3-axis accelerometer (ACC) signals. The results demonstrate the effectiveness of single-modality transformers in analyzing physiological signals, achieving state-of-the-art performance with accuracy, precision and recall values in the range of $99.73\%$ to $99.95\%$ for stress detection. Furthermore, this study explores cross-modal performance and also explains the same using 2D visualization of the learned embedding space and quantitative analysis based on data variance. Despite the large body of work on stress detection and monitoring, the robustness and generalization of these models across different modalities has not been explored. This research represents one of the initial efforts to interpret embedding spaces for stress detection, providing valuable information on cross-modal performance. 

**Abstract (ZH)**: 深度学习的广泛应用已使其在医疗健康领域的应用日益增多，其中的人工智能和传感器进步提高了诊断、治疗和监测的效率。在移动健康领域，基于人工智能的工具能够实现早期诊断和持续监测，如压力等条件的监控。穿戴设备和多模态生理数据使得压力检测日益可行，但模型的有效性取决于数据质量、数量和模态。本研究采用WESAD数据集开发了用于压力检测的变压器模型，并在心电图（ECG）、皮肤电活动（EDA）、肌电图（EMG）、呼吸率（RESP）、温度（TEMP）和三轴加速度计（ACC）信号上进行训练。研究结果表明，单模态变压器在分析生理信号方面具有显著效果，在压力检测中实现了接近100%的性能，准确率、精确率和召回率范围分别为99.73%至99.95%。此外，本研究还探讨了跨模态性能，并通过2D可视化表示学习嵌入空间和基于数据方差的定量分析进行了解释。尽管在压力检测和监控方面的研究已有一定基础，但这些模型在不同模态下的鲁棒性和泛化能力尚未被充分探讨。本研究是解析压力检测嵌入空间的初期尝试，为跨模态性能提供了有价值的信息。 

---
# Mind the Gap: Bridging the Divide Between AI Aspirations and the Reality of Autonomous Characterization 

**Title (ZH)**: 注意差距：弥合人工智能期望与自主特征化现实之间的鸿沟 

**Authors**: Grace Guinan, Addison Salvador, Michelle A. Smeaton, Andrew Glaws, Hilary Egan, Brian C. Wyatt, Babak Anasori, Kevin R. Fiedler, Matthew J. Olszta, Steven R. Spurgeon  

**Link**: [PDF](https://arxiv.org/pdf/2502.18604)  

**Abstract**: What does materials science look like in the "Age of Artificial Intelligence?" Each materials domain-synthesis, characterization, and modeling-has a different answer to this question, motivated by unique challenges and constraints. This work focuses on the tremendous potential of autonomous characterization within electron microscopy. We present our recent advancements in developing domain-aware, multimodal models for microscopy analysis capable of describing complex atomic systems. We then address the critical gap between the theoretical promise of autonomous microscopy and its current practical limitations, showcasing recent successes while highlighting the necessary developments to achieve robust, real-world autonomy. 

**Abstract (ZH)**: 人工智能时代，材料科学呈现出怎样的面貌？每个材料领域——合成、表征和建模——对这一问题的回答各有千秋，这源于它们各自独特的挑战和限制。本研究重点关注自主表征在电子显微镜中的巨大潜力。我们介绍了在显微镜分析中开发领域感知型多模态模型的最新进展，这些模型能够描述复杂的原子系统。随后，我们探讨了自主显微镜的理论前景与其当前实践限制之间的关键差距，展示了最近取得的成果，并指出了实现可靠、实用的自主性的必要发展。 

---
# Steganography Beyond Space-Time With Chain of Multimodal AI Agents 

**Title (ZH)**: 时空之外的隐写术：多模态AI代理链 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18547)  

**Abstract**: Steganography is the art and science of covert writing, with a broad range of applications interwoven within the realm of cybersecurity. As artificial intelligence continues to evolve, its ability to synthesise realistic content emerges as a threat in the hands of cybercriminals who seek to manipulate and misrepresent the truth. Such synthetic content introduces a non-trivial risk of overwriting the subtle changes made for the purpose of steganography. When the signals in both the spatial and temporal domains are vulnerable to unforeseen overwriting, it calls for reflection on what can remain invariant after all. This study proposes a paradigm in steganography for audiovisual media, where messages are concealed beyond both spatial and temporal domains. A chain of multimodal agents is developed to deconstruct audiovisual content into a cover text, embed a message within the linguistic domain, and then reconstruct the audiovisual content through synchronising both aural and visual modalities with the resultant stego text. The message is encoded by biasing the word sampling process of a language generation model and decoded by analysing the probability distribution of word choices. The accuracy of message transmission is evaluated under both zero-bit and multi-bit capacity settings. Fidelity is assessed through both biometric and semantic similarities, capturing the identities of the recorded face and voice, as well as the core ideas conveyed through the media. Secrecy is examined through statistical comparisons between cover and stego texts. Robustness is tested across various scenarios, including audiovisual compression, face-swapping, voice-cloning and their combinations. 

**Abstract (ZH)**: 隐写术是关于隐蔽书写的艺术和技术，其在网络安全领域有着广泛的应用。随着人工智能的不断发展，其生成逼真内容的能力成为网络犯罪分子操纵和歪曲事实时的一个潜在威胁。这种合成内容会增加对通过隐写术进行细微修改的信号进行意外覆盖的风险。当音频-视频空间域和时间域中的信号都可能遭受无法预见的覆盖时，这就需要我们反思在所有这些情况下，还剩下什么不变。本研究提出了一种音频-视频隐写术的新范式，在空间域和时间域之外隐藏消息。我们开发了一条多模态代理链，将音频-视频内容分解为隐藏文本，将信息嵌入到语言领域，再通过同步听觉和视觉模态与生成的隐写文本重建音频-视频内容。信息通过偏差语言生成模型的词采样过程进行编码，通过分析词汇选择的概率分布进行解码。研究在零位容量和多位容量设置下评估了信息传递的准确性。通过生物特性和语义相似度对保真度进行评估，捕捉记录的面孔和声音的身份，以及通过媒介传达的核心思想。通过统计对比隐藏文本和传输文本之间的差异来评估安全性。我们测试了该隐写术方法在多种场景下的鲁棒性，包括音频视频压缩、换脸、变声及其组合。 

---
# FCoT-VL:Advancing Text-oriented Large Vision-Language Models with Efficient Visual Token Compression 

**Title (ZH)**: FCoT-VL：高效视觉令牌压缩促进面向文本的大规模视觉语言模型发展 

**Authors**: Jianjian Li, Junquan Fan, Feng Tang, Gang Huang, Shitao Zhu, Songlin Liu, Nian Xie, Wulong Liu, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18512)  

**Abstract**: The rapid success of Vision Large Language Models (VLLMs) often depends on the high-resolution images with abundant visual tokens, which hinders training and deployment efficiency. Current training-free visual token compression methods exhibit serious performance degradation in tasks involving high-resolution, text-oriented image understanding and reasoning. In this paper, we propose an efficient visual token compression framework for text-oriented VLLMs in high-resolution scenarios. In particular, we employ a light-weight self-distillation pre-training stage to compress the visual tokens, requiring a limited numbers of image-text pairs and minimal learnable parameters. Afterwards, to mitigate potential performance degradation of token-compressed models, we construct a high-quality post-train stage. To validate the effectiveness of our method, we apply it to an advanced VLLMs, InternVL2. Experimental results show that our approach significantly reduces computational overhead while outperforming the baselines across a range of text-oriented benchmarks. We will release the models and code soon. 

**Abstract (ZH)**: 视觉大语言模型（VLLMs）的快速成功往往依赖于高分辨率图像和丰富的视觉标记，这阻碍了训练和部署效率。当前的无需训练的视觉标记压缩方法在涉及高分辨率和文本导向图像理解与推理的任务中表现出严重的性能退化。在本文中，我们提出了一种针对高分辨率场景中的文本导向VLLMs的高效视觉标记压缩框架。特别是，我们采用了一个轻量级的自我精炼预训练阶段来压缩视觉标记，仅需少量的图像-文本对和最少的学习参数。随后，为了缓解标记压缩模型潜在的性能退化，我们构建了一个高质量的后训练阶段。为了验证我们方法的有效性，我们将该方法应用到了先进的VLLMs——InternVL2中。实验结果表明，我们的方法显著减少了计算开销，并在多种文本导向基准测试中优于基线方法。我们将很快发布模型和代码。 

---
