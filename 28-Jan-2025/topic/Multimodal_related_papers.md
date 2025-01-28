# Enhancing Visual Inspection Capability of Multi-Modal Large Language Models on Medical Time Series with Supportive Conformalized and Interpretable Small Specialized Models 

**Title (ZH)**: 增强多模态大型语言模型在医疗时间序列视觉检查能力的支持性同构造化和可解释小型专业化模型辅助方法 

**Authors**: Huayu Li, Xiwen Chen, Ci Zhang, Stuart F. Quan, William D.S. Killgore, Shu-Fen Wung, Chen X. Chen, Geng Yuan, Jin Lu, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16215)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities in visual inspection of medical time-series data, achieving proficiency comparable to human clinicians. However, their broad scope limits domain-specific precision, and proprietary weights hinder fine-tuning for specialized datasets. In contrast, small specialized models (SSMs) excel in targeted tasks but lack the contextual reasoning required for complex clinical decision-making. To address these challenges, we propose ConMIL (Conformalized Multiple Instance Learning), a decision-support SSM that integrates seamlessly with LLMs. By using Multiple Instance Learning (MIL) to identify clinically significant signal segments and conformal prediction for calibrated set-valued outputs, ConMIL enhances LLMs' interpretative capabilities for medical time-series analysis. Experimental results demonstrate that ConMIL significantly improves the performance of state-of-the-art LLMs, such as ChatGPT4.0 and Qwen2-VL-7B. Specifically, \ConMIL{}-supported Qwen2-VL-7B achieves 94.92% and 96.82% precision for confident samples in arrhythmia detection and sleep staging, compared to standalone LLM accuracy of 46.13% and 13.16%. These findings highlight the potential of ConMIL to bridge task-specific precision and broader contextual reasoning, enabling more reliable and interpretable AI-driven clinical decision support. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医学时间序列数据的视觉检查方面表现出显著的能力，其专业水平接近于人类临床医生。然而，其广泛的适用范围限制了其在特定领域的精密度，而专有的模型权重则阻碍了对专门数据集的微调。相比之下，小型专门模型（SSMs）在执行特定任务方面表现出色，但缺乏进行复杂临床决策所需的背景推理能力。为解决这些挑战，我们提出了一种决策支持的小型专门模型（ConMIL，Conformalized Multiple Instance Learning），该模型能够与LLMs无缝集成。通过使用多重实例学习（MIL）识别具有临床意义的信号片段，并使用校准型集合值输出的卷积预测，ConMIL增强了LLMs对医学时间序列分析的解释能力。实验结果表明，ConMIL显著提高了当前先进的LLMs（如ChatGPT4.0和Qwen2-VL-7B）的表现。具体而言，ConMIL支持的Qwen2-VL-7B在心律失常检测和睡眠阶段分类中的精确度分别达到了94.92%和96.82%，而单独的LLMs在这两项任务中的精度分别为46.13%和13.16%。这些发现突显了ConMIL在任务特定精度和更广泛背景推理之间架起桥梁的潜力，从而能够提供更可靠和可解释的AI驱动的临床决策支持。 

---
# A Causality-aware Paradigm for Evaluating Creativity of Multimodal Large Language Models 

**Title (ZH)**: 具有因果意识的范式，用于评估多模态大型语言模型的创造力 

**Authors**: Zhongzhan Huang, Shanshan Zhong, Pan Zhou, Shanghua Gao, Marinka Zitnik, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.15147)  

**Abstract**: Recently, numerous benchmarks have been developed to evaluate the logical reasoning abilities of large language models (LLMs). However, assessing the equally important creative capabilities of LLMs is challenging due to the subjective, diverse, and data-scarce nature of creativity, especially in multimodal scenarios. In this paper, we consider the comprehensive pipeline for evaluating the creativity of multimodal LLMs, with a focus on suitable evaluation platforms and methodologies. First, we find the Oogiri game, a creativity-driven task requiring humor, associative thinking, and the ability to produce unexpected responses to text, images, or both. This game aligns well with the input-output structure of modern multimodal LLMs and benefits from a rich repository of high-quality, human-annotated creative responses, making it an ideal platform for studying LLM creativity. Next, beyond using the Oogiri game for standard evaluations like ranking and selection, we propose LoTbench, an interactive, causality-aware evaluation framework, to further address some intrinsic risks in standard evaluations, such as information leakage and limited interpretability. The proposed LoTbench not only quantifies LLM creativity more effectively but also visualizes the underlying creative thought processes. Our results show that while most LLMs exhibit constrained creativity, the performance gap between LLMs and humans is not insurmountable. Furthermore, we observe a strong correlation between results from the multimodal cognition benchmark MMMU and LoTbench, but only a weak connection with traditional creativity metrics. This suggests that LoTbench better aligns with human cognitive theories, highlighting cognition as a critical foundation in the early stages of creativity and enabling the bridging of diverse concepts. this https URL 

**Abstract (ZH)**: 近年来，為了評估大型語言模型（LLMs）的邏輯推理能力，已經開發了大量評估基准。然而，由於創意的主觀性、多樣性和數據匱乏性，尤其是多模態場景下的特性，評估LLMs的同等重要的創意能力仍然具有挑戰。本文我們考慮了一套全面的評估管道，用於評估多模態LLMs的創意能力，並重點討論了合適的評估平台和方法。首先，我們發現了一個基於創意的Oogiri遊戲，這個遊戲需要幽默感、聯想思考，以及對文字、圖像或兩者的非預期回應生产能力。這個遊戲與現代多模態LLMs的輸入-輸出結構非常契合，並且得益於一個豐富的高質量、人標注的創意回應數據庫，使其成為研究LLMs創意的理想平台。其次是超越使用Oogiri遊戲進行標準評估（如排名和選擇），我們提出了LoTbench，一種交互式的、知因果性感知的評估框架，旨在進一步解決標準評估中的固有風險，例如信息洩露和有限的可解釋性。所提出的LoTbench不僅更有效地量化了LLMs的創意能力，還可视化了其背後的理念產生過程。結果表明，雖然大多數LLMs展示出了局限性的創意能力，但人類和LLMs之間的表現差距並非無法逾越。此外，我們觀察到多模態認知基准MMMU和LoTbench的結果之間存在密切的相關性，而與傳統創意指標的連接則較弱。這表明LoTbench更符合人類認知理論，強調認知在創意初步階段的關鍵基礎作用，並促進了多樣概念之間的橋接。

更多內容請訪問：[原文链接] 

---
# sDREAMER: Self-distilled Mixture-of-Modality-Experts Transformer for Automatic Sleep Staging 

**Title (ZH)**: sDREAMER：自我蒸馏多模态专家混合的变压器模型及其在自动睡眠阶段划分中的应用 

**Authors**: Jingyuan Chen, Yuan Yao, Mie Anderson, Natalie Hauglund, Celia Kjaerby, Verena Untiet, Maiken Nedergaard, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16329)  

**Abstract**: Automatic sleep staging based on electroencephalography (EEG) and electromyography (EMG) signals is an important aspect of sleep-related research. Current sleep staging methods suffer from two major drawbacks. First, there are limited information interactions between modalities in the existing methods. Second, current methods do not develop unified models that can handle different sources of input. To address these issues, we propose a novel sleep stage scoring model sDREAMER, which emphasizes cross-modality interaction and per-channel performance. Specifically, we develop a mixture-of-modality-expert (MoME) model with three pathways for EEG, EMG, and mixed signals with partially shared weights. We further propose a self-distillation training scheme for further information interaction across modalities. Our model is trained with multi-channel inputs and can make classifications on either single-channel or multi-channel inputs. Experiments demonstrate that our model outperforms the existing transformer-based sleep scoring methods for multi-channel inference. For single-channel inference, our model also outperforms the transformer-based models trained with single-channel signals. 

**Abstract (ZH)**: 基于脑电图（EEG）和肌电图（EMG）信号的自动睡眠分期是相关研究的重要方面。当前的睡眠分期方法存在两大主要问题。首先，现有方法中的不同模态之间信息交互有限。其次，当前方法没有开发能够处理不同输入来源的统一模型。为解决这些问题，我们提出了一种新的睡眠阶段评分模型 sDREAMER，该模型强调跨模态交互和每通道性能。具体而言，我们开发了一个模态专家混合（MoME）模型，该模型具有三条路径，分别处理EEG、EMG和混合信号，并共享部分权重。我们还提出了一种自我蒸馏训练方案，以进一步促进不同模态之间的信息交互。该模型可以使用多通道输入进行训练，并且可以在单通道或多通道输入上进行分类。实验结果表明，我们的模型在多通道推理中优于现有的基于变压器的睡眠评分方法。对于单通道推理，我们的模型也优于使用单通道信号训练的基于变压器的模型。 

---
# Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity 

**Title (ZH)**: Mixture-of-Mamba：增强多模态状态空间模型的模态感知稀疏性 

**Authors**: Weixin Liang, Junhong Shen, Genghan Zhang, Ning Dong, Luke Zettlemoyer, Lili Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16295)  

**Abstract**: State Space Models (SSMs) have emerged as efficient alternatives to Transformers for sequential modeling, but their inability to leverage modality-specific features limits their performance in multi-modal pretraining. Here, we propose Mixture-of-Mamba, a novel SSM architecture that introduces modality-aware sparsity through modality-specific parameterization of the Mamba block. Building on Mixture-of-Transformers (W. Liang et al. arXiv:2411.04996; 2024), we extend the benefits of modality-aware sparsity to SSMs while preserving their computational efficiency. We evaluate Mixture-of-Mamba across three multi-modal pretraining settings: Transfusion (interleaved text and continuous image tokens with diffusion loss), Chameleon (interleaved text and discrete image tokens), and an extended three-modality framework incorporating speech. Mixture-of-Mamba consistently reaches the same loss values at earlier training steps with significantly reduced computational costs. In the Transfusion setting, Mixture-of-Mamba achieves equivalent image loss using only 34.76% of the training FLOPs at the 1.4B scale. In the Chameleon setting, Mixture-of-Mamba reaches similar image loss with just 42.50% of the FLOPs at the 1.4B scale, and similar text loss with just 65.40% of the FLOPs. In the three-modality setting, MoM matches speech loss at 24.80% of the FLOPs at the 1.4B scale. Our ablation study highlights the synergistic effects of decoupling projection components, where joint decoupling yields greater gains than individual modifications. These results establish modality-aware sparsity as a versatile and effective design principle, extending its impact from Transformers to SSMs and setting new benchmarks in multi-modal pretraining. Our code can be accessed at this https URL 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

自回归模型（State Space Models, SSMs）已逐渐成为Transformer在序列建模方面的有效替代方案，但它们在多模态预训练中无法充分利用特定模态特征的能力限制了其性能。为此，我们提出了一种名为Mixture-of-Mamba的新颖SSM架构，通过Mamba块的模态特定参数化引入模态感知稀疏性。该架构基于Mixture-of-Transformers（W. Liang等，arXiv:2411.04996，2024）的研究，在保留SSM高效计算特性的同时，进一步推广了模态感知稀疏性的优势。我们分别在三种多模态预训练设置中评估了Mixture-of-Mamba的表现：Transfusion（交错排列的文本和连续图像标记，并使用扩散损失）、Chameleon（交错排列的文本和离散图像标记）以及包含语音的扩展三模态框架。结果显示，Mixture-of-Mamba在早期训练步骤中以显著减少的计算成本达到了相同损失值。在Transfusion设置中，Mixture-of-Mamba仅使用1.4B模型规模下34.76%的训练FLOP实现了与全量计算相等的图像损失。在Chameleon设置中，Mixture-of-Mamba在1.4B模型规模下仅使用42.50%的FLOP达到了相似的图像损失和65.40%的FLOP实现了相似的文本损失。在三模态设置中，Mixture-of-Mamba在1.4B模型规模下仅使用24.80%的FLOP达到了相似的语音损失。我们的消融研究强调了解耦投影组件的协同效应，表明联合解耦方式提供的增益大于单独修改方式。这些结果证明，模态感知稀疏性是一种灵活且有效的设计原则，它的影响不仅限于Transformer，还能扩展到SSM，并在多模态预训练中设立了新的基准。我们的代码可在以下链接访问：[this https URL] 

---
# Brain-Adapter: Enhancing Neurological Disorder Analysis with Adapter-Tuning Multimodal Large Language Models 

**Title (ZH)**: 脑适配器：通过适配调谐多模态大型语言模型增强神经疾病分析 

**Authors**: Jing Zhang, Xiaowei Yu, Yanjun Lyu, Lu Zhang, Tong Chen, Chao Cao, Yan Zhuang, Minheng Chen, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16282)  

**Abstract**: Understanding brain disorders is crucial for accurate clinical diagnosis and treatment. Recent advances in Multimodal Large Language Models (MLLMs) offer a promising approach to interpreting medical images with the support of text descriptions. However, previous research has primarily focused on 2D medical images, leaving richer spatial information of 3D images under-explored, and single-modality-based methods are limited by overlooking the critical clinical information contained in other modalities. To address this issue, this paper proposes Brain-Adapter, a novel approach that incorporates an extra bottleneck layer to learn new knowledge and instill it into the original pre-trained knowledge. The major idea is to incorporate a lightweight bottleneck layer to train fewer parameters while capturing essential information and utilize a Contrastive Language-Image Pre-training (CLIP) strategy to align multimodal data within a unified representation space. Extensive experiments demonstrated the effectiveness of our approach in integrating multimodal data to significantly improve the diagnosis accuracy without high computational costs, highlighting the potential to enhance real-world diagnostic workflows. 

**Abstract (ZH)**: 了解大脑疾病对于准确的临床诊断和治疗至关重要。近年来，多模态大型语言模型（MLLMs）的发展为通过文本描述解释医学图像提供了有希望的方法。然而，此前的研究主要集中在2D医学图像上，忽略了3D图像中 richer 的空间信息，而基于单一模态的方法则受限于未能充分利用其他模态中的关键临床信息。为解决这一问题，本文提出了一种名为 Brain-Adapter 的新型方法，该方法通过引入额外的瓶颈层来学习新知识并将其融入原始预训练知识中。主要思想是通过引入一个轻量级的瓶颈层来训练较少的参数，同时捕获关键信息，并利用 Contrastive Language-Image Pre-training (CLIP) 策略在统一的表示空间内对多模态数据进行对齐。广泛实验表明，该方法在整合多模态数据以显著提高诊断准确性方面具有有效性，且无需高昂的计算成本，突显了其在真实临床诊断流程中增强性能的潜力。 

---
# MetaDecorator: Generating Immersive Virtual Tours through Multimodality 

**Title (ZH)**: MetaDecorator：通过多模态生成沉浸式虚拟导览 

**Authors**: Shuang Xie, Yang Liu, Jeannie S.A. Lee, Haiwei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.16164)  

**Abstract**: MetaDecorator, is a framework that empowers users to personalize virtual spaces. By leveraging text-driven prompts and image synthesis techniques, MetaDecorator adorns static panoramas captured by 360° imaging devices, transforming them into uniquely styled and visually appealing environments. This significantly enhances the realism and engagement of virtual tours compared to traditional offerings. Beyond the core framework, we also discuss the integration of Large Language Models (LLMs) and haptics in the VR application to provide a more immersive experience. 

**Abstract (ZH)**: MetaDecorator 是一个框架，赋能用户个性化虚拟空间。通过利用文本驱动的提示和图像合成技术，MetaDecorator 装饰由360°成像设备捕获的静态全景图，将其转化为具有独特风格和视觉吸引力的环境。这相较于传统方案显著提升了虚拟巡游的真实感和参与度。除了核心框架之外，我们还讨论了将大型语言模型（LLMs）和触觉技术集成到VR应用中，以提供更加沉浸式的体验。 

---
# Automated Detection of Sport Highlights from Audio and Video Sources 

**Title (ZH)**: 从音频和视频源自动检测体育高光时刻 

**Authors**: Francesco Della Santa, Morgana Lalli  

**Link**: [PDF](https://arxiv.org/pdf/2501.16100)  

**Abstract**: This study presents a novel Deep Learning-based and lightweight approach for the automated detection of sports highlights (HLs) from audio and video sources. HL detection is a key task in sports video analysis, traditionally requiring significant human effort. Our solution leverages Deep Learning (DL) models trained on relatively small datasets of audio Mel-spectrograms and grayscale video frames, achieving promising accuracy rates of 89% and 83% for audio and video detection, respectively. The use of small datasets, combined with simple architectures, demonstrates the practicality of our method for fast and cost-effective deployment. Furthermore, an ensemble model combining both modalities shows improved robustness against false positives and false negatives. The proposed methodology offers a scalable solution for automated HL detection across various types of sports video content, reducing the need for manual intervention. Future work will focus on enhancing model architectures and extending this approach to broader scene-detection tasks in media analysis. 

**Abstract (ZH)**: 本研究提出了一种基于深度学习且轻量级的方法，用于从音频和视频源自动检测体育精彩片段（HLs）。精彩片段检测是体育视频分析中的一个关键任务，传统上需要大量的人工努力。我们利用在相对较小的音频梅尔频谱图和灰度视频帧数据集上训练的深度学习（DL）模型，分别实现了89%和83%的音频和视频检测准确率。利用较小的数据集和简单的架构，证明了我们方法的实用性和快速、低成本部署的可行性。此外，结合两种模态的集成模型在对抗假阳性与假阴性方面表现出更强的稳健性。所提方法提供了一种针对各类体育视频内容可扩展的自动精彩片段检测解决方案，减少了手动干预的需求。未来的工作将致力于增强模型架构，并将此方法扩展到媒体分析中的更广泛的场景检测任务。 

---
# Intelligent Code Embedding Framework for High-Precision Ransomware Detection via Multimodal Execution Path Analysis 

**Title (ZH)**: 基于多模态执行路径分析的高精度勒索软件检测智能代码嵌入框架 

**Authors**: Levi Gareth, Maximilian Fairbrother, Peregrine Blackwood, Lucasta Underhill, Benedict Ruthermore  

**Link**: [PDF](https://arxiv.org/pdf/2501.15836)  

**Abstract**: Modern threat landscapes continue to evolve with increasing sophistication, challenging traditional detection methodologies and necessitating innovative solutions capable of addressing complex adversarial tactics. A novel framework was developed to identify ransomware activity through multimodal execution path analysis, integrating high-dimensional embeddings and dynamic heuristic derivation mechanisms to capture behavioral patterns across diverse attack variants. The approach demonstrated high adaptability, effectively mitigating obfuscation strategies and polymorphic characteristics often employed by ransomware families to evade detection. Comprehensive experimental evaluations revealed significant advancements in precision, recall, and accuracy metrics compared to baseline techniques, particularly under conditions of variable encryption speeds and obfuscated execution flows. The framework achieved scalable and computationally efficient performance, ensuring robust applicability across a range of system configurations, from resource-constrained environments to high-performance infrastructures. Notable findings included reduced false positive rates and enhanced detection latency, even for ransomware families employing sophisticated encryption mechanisms. The modular design allowed seamless integration of additional modalities, enabling extensibility and future-proofing against emerging threat vectors. Quantitative analyses further highlighted the system's energy efficiency, emphasizing its practicality for deployment in environments with stringent operational constraints. The results underline the importance of integrating advanced computational techniques and dynamic adaptability to safeguard digital ecosystems from increasingly complex threats. 

**Abstract (ZH)**: 现代威胁场景持续进化，日益复杂，这不仅挑战了传统的检测方法，还迫切需要创新的解决方案来应对复杂的对抗性战术。为此，我们提出了一种新的框架，用于通过多模态执行路径分析来识别勒索软件活动。该框架集成了高维嵌入和动态启发式衍生机制，以便在各种攻击变种中捕捉行为模式。研究结果表明，该方法具有高度的适应性，能够有效应对勒索软件家族常用的混淆技术及其多态特性以逃避检测。全面的实验评估表明，与基线技术相比，在不同加密速度和混淆执行流条件下，该框架在精准度、召回率和准确性等方面有了显著提升。框架实现了可扩展且计算高效的性能，确保在从资源受限环境到高性能基础设施的广泛系统配置下具有稳健的适用性。研究发现，该框架的误报率降低，检测延迟显著缩短，即使在采用复杂加密机制的勒索软件家族中也是如此。模块化设计使其易于集成额外的数据模态，从而增强了未来对抗新兴威胁的能力。定量分析进一步突显了该系统的能源效率，强调了在具有严格操作约束的环境中部署其实用性。研究结果进一步强调了集成先进的计算技术和动态适应性对于保护日益复杂的数字生态系统的重要性。 

---
# SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model 

**Title (ZH)**: SpatialVLA：探索空间表示在视觉-语言-行动模型中的应用 

**Authors**: Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.15830)  

**Abstract**: In this paper, we claim that spatial understanding is the keypoint in robot manipulation, and propose SpatialVLA to explore effective spatial representations for the robot foundation model. Specifically, we introduce Ego3D Position Encoding to inject 3D information into the input observations of the visual-language-action model, and propose Adaptive Action Grids to represent spatial robot movement actions with adaptive discretized action grids, facilitating learning generalizable and transferrable spatial action knowledge for cross-robot control. SpatialVLA is first pre-trained on top of a vision-language model with 1.1 Million real-world robot episodes, to learn a generalist manipulation policy across multiple robot environments and tasks. After pre-training, SpatialVLA is directly applied to perform numerous tasks in a zero-shot manner. The superior results in both simulation and real-world robots demonstrate its advantage of inferring complex robot motion trajectories and its strong in-domain multi-task generalization ability. We further show the proposed Adaptive Action Grids offer a new and effective way to fine-tune the pre-trained SpatialVLA model for new simulation and real-world setups, where the pre-learned action grids are re-discretized to capture robot-specific spatial action movements of new setups. The superior results from extensive evaluations demonstrate the exceptional in-distribution generalization and out-of-distribution adaptation capability, highlighting the crucial benefit of the proposed spatial-aware representations for generalist robot policy learning. All the details and codes will be open-sourced. 

**Abstract (ZH)**: 在这篇论文中，我们主张空间理解是机器人操作的关键，并提出SpatialVLA以探索适用于机器人基础模型的有效空间表示。具体来说，我们引入了Ego3D位置编码，将3D信息注入视觉-语言-动作模型的输入观察中，并提出了自适应动作网格来用自适应离散的动作网格表示空间机器人运动动作，从而促进跨机器人控制的一般化和可转移的空间动作知识学习。SpatialVLA首先在包含110万真实世界机器人经历的视觉语言模型上进行预训练，以学习跨多个机器人环境和任务的一般机器人操作策略。在预训练后，SpatialVLA可以直接应用于在零样本情况下执行众多任务。在仿真实验和真实世界机器人中的优越结果证明了其推断复杂机器人运动轨迹的优势以及其强大的领域内多任务一般化能力。我们进一步展示了提出的自适应动作网格提供了一种新的有效方法，用于微调预训练的SpatialVLA模型以适应新的仿真实验和真实世界设置，其中预学习的动作网格被重新离散化以捕捉新设置中的机器人特定的空间动作。广泛的评估结果表明了其在域内一般化和域外适应方面的出色能力，突显了所提出的空间意识表示对通用机器人策略学习的关键益处。所有细节和代码都将开源。 

---
# Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning 

**Title (ZH)**: Gensors：使用多模态基础模型和推理构建个性化的视觉传感器 

**Authors**: Michael Xieyang Liu, Savvas Petridis, Vivian Tsai, Alexander J. Fiannaca, Alex Olwal, Michael Terry, Carrie J. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.15727)  

**Abstract**: Multimodal large language models (MLLMs), with their expansive world knowledge and reasoning capabilities, present a unique opportunity for end-users to create personalized AI sensors capable of reasoning about complex situations. A user could describe a desired sensing task in natural language (e.g., "alert if my toddler is getting into mischief"), with the MLLM analyzing the camera feed and responding within seconds. In a formative study, we found that users saw substantial value in defining their own sensors, yet struggled to articulate their unique personal requirements and debug the sensors through prompting alone. To address these challenges, we developed Gensors, a system that empowers users to define customized sensors supported by the reasoning capabilities of MLLMs. Gensors 1) assists users in eliciting requirements through both automatically-generated and manually created sensor criteria, 2) facilitates debugging by allowing users to isolate and test individual criteria in parallel, 3) suggests additional criteria based on user-provided images, and 4) proposes test cases to help users "stress test" sensors on potentially unforeseen scenarios. In a user study, participants reported significantly greater sense of control, understanding, and ease of communication when defining sensors using Gensors. Beyond addressing model limitations, Gensors supported users in debugging, eliciting requirements, and expressing unique personal requirements to the sensor through criteria-based reasoning; it also helped uncover users' "blind spots" by exposing overlooked criteria and revealing unanticipated failure modes. Finally, we discuss how unique characteristics of MLLMs--such as hallucinations and inconsistent responses--can impact the sensor-creation process. These findings contribute to the design of future intelligent sensing systems that are intuitive and customizable by everyday users. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）凭借其广泛的世界知识和推理能力，为终端用户创造个性化的AI传感器提供了独特的机会，这些传感器能够对复杂情况进行推理。用户可以用自然语言描述一个期望的感知任务（例如，“如果我的幼儿在搞破坏，请报警”），MLLMs 分析摄像头视频并在几秒钟内做出响应。在一项形成性研究中，我们发现用户在定义自己的传感器方面看到了巨大的价值，但他们在表达独特的个人需求及仅通过提示进行调试方面遇到困难。为了解决这些问题，我们开发了Gensors系统，它赋予用户通过MLLMs 的推理能力定义自定义传感器的能力。Gensors 1) 通过自动生成和手动创建传感器标准帮助用户提取需求，2) 通过允许用户并行隔离和测试各个标准的方式支持调试，3) 根据用户提供的图片建议额外的标准，4) 提出测试案例帮助用户对传感器进行“压力测试”，以应对可能不可预见的情况。在一项用户研究中，参与者在使用Gensors定义传感器时报告了更大的控制感、理解和沟通便利性。除了解决模型限制外，Gensors还支持用户通过基于标准的推理调试、提取需求并表达独特的个人需求，同时通过暴露被忽视的标准并揭示不可预见的故障模式，帮助用户发现“盲点”。最后，我们讨论了MLLMs的独特特征（如幻觉和不一致的响应）如何影响传感器创建过程。这些发现为未来直观且可定制的智能感知系统的设计做出了贡献。 

---
# Transformer-Based Multimodal Knowledge Graph Completion with Link-Aware Contexts 

**Title (ZH)**: 基于变换器的具有链接意识上下文的多模态知识图谱完成方法 

**Authors**: Haodi Ma, Dzmitry Kasinets, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15688)  

**Abstract**: Multimodal knowledge graph completion (MMKGC) aims to predict missing links in multimodal knowledge graphs (MMKGs) by leveraging information from various modalities alongside structural data. Existing MMKGC approaches primarily extend traditional knowledge graph embedding (KGE) models, which often require creating an embedding for every entity. This results in large model sizes and inefficiencies in integrating multimodal information, particularly for real-world graphs. Meanwhile, Transformer-based models have demonstrated competitive performance in knowledge graph completion (KGC). However, their focus on single-modal knowledge limits their capacity to utilize cross-modal information. Recently, Large vision-language models (VLMs) have shown potential in cross-modal tasks but are constrained by the high cost of training. In this work, we propose a novel approach that integrates Transformer-based KGE models with cross-modal context generated by pre-trained VLMs, thereby extending their applicability to MMKGC. Specifically, we employ a pre-trained VLM to transform relevant visual information from entities and their neighbors into textual sequences. We then frame KGC as a sequence-to-sequence task, fine-tuning the model with the generated cross-modal context. This simple yet effective method significantly reduces model size compared to traditional KGE approaches while achieving competitive performance across multiple large-scale datasets with minimal hyperparameter tuning. 

**Abstract (ZH)**: 多模态知识图谱完成（MMKGC）旨在通过利用各种模态的信息以及结构数据来预测多模态知识图谱（MMKGs）中的缺失链接。现有的MMKGC方法主要扩展了传统的知识图嵌入（KGE）模型，这些模型通常需要为每个实体创建嵌入，这导致了较大的模型规模，并且在整合多模态信息时效率低下，特别是对于实际图而言。同时，基于Transformer的模型在知识图谱完成（KGC）方面已经展示了竞争力。然而，它们主要关注单模态知识，限制了其利用跨模态信息的能力。最近，大型的多模态视觉语言模型（VLMs）在跨模态任务中显示出潜力，但受限于训练成本高昂。在本工作中，我们提出了一种新的方法，将基于Transformer的知识图嵌入模型与预训练的VLM生成的跨模态上下文相结合，从而将其应用扩展到MMKGC。具体而言，我们利用预训练的VLM将实体及其邻居的相关视觉信息转换为文本序列。然后，我们将KGC建模为序列到序列的任务，并通过生成的跨模态上下文对模型进行微调。这一简单有效的方案相较于传统的KGE方法显著减少了模型规模，在多个大规模数据集上实现了竞争力的性能，并且只需微量超参数调整。 

---
# MetaOcc: Surround-View 4D Radar and Camera Fusion Framework for 3D Occupancy Prediction with Dual Training Strategies 

**Title (ZH)**: MetaOcc：一种基于双训练策略的四维雷达和摄像头融合框架，用于三维占用率预测 

**Authors**: Long Yang, Lianqing Zheng, Wenjin Ai, Minghao Liu, Sen Li, Qunshu Lin, Shengyu Yan, Jie Bai, Zhixiong Ma, Xichan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15384)  

**Abstract**: 3D occupancy prediction is crucial for autonomous driving perception. Fusion of 4D radar and camera provides a potential solution of robust occupancy prediction on serve weather with least cost. How to achieve effective multi-modal feature fusion and reduce annotation costs remains significant challenges. In this work, we propose MetaOcc, a novel multi-modal occupancy prediction framework that fuses surround-view cameras and 4D radar for comprehensive environmental perception. We first design a height self-attention module for effective 3D feature extraction from sparse radar points. Then, a local-global fusion mechanism is proposed to adaptively capture modality contributions while handling spatio-temporal misalignments. Temporal alignment and fusion module is employed to further aggregate historical feature. Furthermore, we develop a semi-supervised training procedure leveraging open-set segmentor and geometric constraints for pseudo-label generation, enabling robust perception with limited annotations. Extensive experiments on OmniHD-Scenes dataset demonstrate that MetaOcc achieves state-of-the-art performance, surpassing previous methods by significant margins. Notably, as the first semi-supervised 4D radar and camera fusion-based occupancy prediction approach, MetaOcc maintains 92.5% of the fully-supervised performance while using only 50% of ground truth annotations, establishing a new benchmark for multi-modal 3D occupancy prediction. Code and data are available at this https URL. 

**Abstract (ZH)**: 三维占用预测对于自主驾驶感知至关重要。融合四维雷达和摄像头数据提供了一种在恶劣天气条件下实现稳健占用预测的潜在解决方案，且成本较低。如何实现有效的多模态特征融合以及降低注释成本仍然是重要的挑战。在本文中，我们提出了一种名为MetaOcc的新颖多模态占用预测框架，该框架融合环视摄像头和四维雷达数据，以实现全面的环境感知。我们首先设计了一个高度自注意力模块，用于从稀疏的雷达点中有效提取三维特征。然后，提出了一种局部-全局融合机制，以自适应地捕获模态贡献并处理空间-时间对齐问题。此外，我们使用开放集分割器和几何约束开发了一种半监督训练方法，以生成伪标签，从而在有限注释下实现稳健的感知。在OmniHD-Scenes数据集上的广泛实验表明，MetaOcc达到了最先进的性能，显著优于之前的算法。值得注意的是，作为首个基于四维雷达和摄像头融合的半监督占用预测方法，MetaOcc仅使用50%的地面真实注释，就能保持与完全监督性能92.5%的水平，从而为多模态3D占用预测设立了新的基准。代码和数据可在以下链接处获取：[请提供具体链接]。 

---
# Zero-Shot Interactive Text-to-Image Retrieval via Diffusion-Augmented Representations 

**Title (ZH)**: 基于扩散增强表示的零样本交互式文本到图像检索 

**Authors**: Zijun Long, Kangheng Liang, Gerardo Aragon-Camarasa, Richard Mccreadie, Paul Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2501.15379)  

**Abstract**: Interactive Text-to-Image Retrieval (I-TIR) has emerged as a transformative user-interactive tool for applications in domains such as e-commerce and education. Yet, current methodologies predominantly depend on finetuned Multimodal Large Language Models (MLLMs), which face two critical limitations: (1) Finetuning imposes prohibitive computational overhead and long-term maintenance costs. (2) Finetuning narrows the pretrained knowledge distribution of MLLMs, reducing their adaptability to novel scenarios. These issues are exacerbated by the inherently dynamic nature of real-world I-TIR systems, where queries and image databases evolve in complexity and diversity, often deviating from static training distributions. To overcome these constraints, we propose Diffusion Augmented Retrieval (DAR), a paradigm-shifting framework that bypasses MLLM finetuning entirely. DAR synergizes Large Language Model (LLM)-guided query refinement with Diffusion Model (DM)-based visual synthesis to create contextually enriched intermediate representations. This dual-modality approach deciphers nuanced user intent more holistically, enabling precise alignment between textual queries and visually relevant images. Rigorous evaluations across four benchmarks reveal DAR's dual strengths: (1) Matches state-of-the-art finetuned I-TIR models on straightforward queries without task-specific training. (2) Scalable Generalization: Surpasses finetuned baselines by 7.61% in Hits@10 (top-10 accuracy) under multi-turn conversational complexity, demonstrating robustness to intricate, distributionally shifted interactions. By eliminating finetuning dependencies and leveraging generative-augmented representations, DAR establishes a new trajectory for efficient, adaptive, and scalable cross-modal retrieval systems. 

**Abstract (ZH)**: 交互式文本到图像检索（I-TIR）已成为电子商务和教育领域等应用中的一种变革性用户互动工具。然而，现有方法主要依赖于微调的多模态大型语言模型（MLLMs），这些方法面临两个关键限制：（1）微调会带来巨大的计算开销和长期维护成本；（2）微调会限制MLLMs的预训练知识分布，降低其对新颖场景的适应性。这些问题在实际世界的I-TIR系统中尤为突出，因为查询和图像数据库在复杂性和多样性方面不断变化，往往与静态训练分布相偏离。为克服这些限制，我们提出了一种名为扩散增强检索（DAR）的范式革新框架，完全绕过了MLLM的微调。DAR 结合了大型语言模型（LLM）引导的查询细化和基于扩散模型（DM）的视觉合成，以创建上下文增强的中间表示。这种双模态方法更全面地了解用户意图，使得文本查询与相关图像之间实现精确对齐。跨四个基准的严格评估显示，DAR 的双重优势在于：（1）对于简单的查询，DAR 能够与最先进的微调I-TIR模型媲美，无需特定任务的训练；（2）可扩展的泛化：在多轮对话复杂性下，DAR 使Hits@10（前10准确率）提高了7.61%，展示了其对复杂、分布变化交互的鲁棒性。通过消除对微调的依赖并利用生成增强的表示形式，DAR 建立了一条高效、适应性强且可扩展的跨模态检索系统的全新路径。 

---
# Scaling Large Vision-Language Models for Enhanced Multimodal Comprehension In Biomedical Image Analysis 

**Title (ZH)**: 面向生物医学图像分析的大型视觉-语言模型的扩展以增强多模态理解 

**Authors**: Robinson Umeike, Neil Getty, Fangfang Xia, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2501.15370)  

**Abstract**: Large language models (LLMs) have demonstrated immense capabilities in understanding textual data and are increasingly being adopted to help researchers accelerate scientific discovery through knowledge extraction (information retrieval), knowledge distillation (summarizing key findings and methodologies into concise forms), and knowledge synthesis (aggregating information from multiple scientific sources to address complex queries, generate hypothesis and formulate experimental plans). However, scientific data often exists in both visual and textual modalities. Vision language models (VLMs) address this by incorporating a pretrained vision backbone for processing images and a cross-modal projector that adapts image tokens into the LLM dimensional space, thereby providing richer multimodal comprehension. Nevertheless, off-the-shelf VLMs show limited capabilities in handling domain-specific data and are prone to hallucinations. We developed intelligent assistants finetuned from LLaVA models to enhance multimodal understanding in low-dose radiation therapy (LDRT)-a benign approach used in the treatment of cancer-related illnesses. Using multilingual data from 42,673 articles, we devise complex reasoning and detailed description tasks for visual question answering (VQA) benchmarks. Our assistants, trained on 50,882 image-text pairs, demonstrate superior performance over base models as evaluated using LLM-as-a-judge approach, particularly in reducing hallucination and improving domain-specific comprehension. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解文本数据方面展现了巨大的能力，并且越来越多地被用于通过知识提取（信息检索）、知识精简（将关键发现和方法学总结为简洁的形式）和知识综合（从多个科学来源汇总信息以解决复杂查询、形成假设和编制实验计划）来帮助研究人员加速科学发现。然而，科学数据通常以视觉和文本两种模态存在。视觉语言模型（VLMs）通过引入预训练的视觉骨干网络来处理图像，并通过跨模态投影器将图像标记转换到LLM的空间，从而提供更丰富的跨模态理解。尽管现成的VLMs在处理领域特定数据方面的能力有限，并且容易产生幻觉，我们开发了基于LLaVA模型微调的智能助手，以增强低剂量辐射治疗（LDRT）领域的跨模态理解——LDRT是一种用于治疗癌症相关疾病的良性方法。我们使用来自42,673篇文章的多语言数据，为视觉问答（VQA）基准设计了复杂的推理和详细的描述任务。我们的助手在50,882幅图像-文本对上进行训练，在LLM作为评判员的方法评估中表现出色，特别是在减少幻觉和提高领域特定理解方面。 

---
# Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models 

**Title (ZH)**: 分析并增强细粒度视觉识别在多模态大语言模型中的能力 

**Authors**: Hulingxiao He, Geng Li, Zijun Geng, Jinglin Xu, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15140)  

**Abstract**: Multi-modal large language models (MLLMs) have shown remarkable abilities in various visual understanding tasks. However, MLLMs still struggle with fine-grained visual recognition (FGVR), which aims to identify subordinate-level categories from images. This can negatively impact more advanced capabilities of MLLMs, such as object-centric visual question answering and reasoning. In our study, we revisit three quintessential capabilities of MLLMs for FGVR, including object information extraction, category knowledge reserve, object-category alignment, and position of the root cause as a misalignment problem. To address this issue, we present Finedefics, an MLLM that enhances the model's FGVR capability by incorporating informative attribute descriptions of objects into the training phase. We employ contrastive learning on object-attribute pairs and attribute-category pairs simultaneously and use examples from similar but incorrect categories as hard negatives, naturally bringing representations of visual objects and category names closer. Extensive evaluations across multiple popular FGVR datasets demonstrate that Finedefics outperforms existing MLLMs of comparable parameter sizes, showcasing its remarkable efficacy. The code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在各种视觉理解任务中展现出卓越的能力。然而，MLLMs 在细粒度视觉识别（FGVR）方面仍然存在挑战，FGVR 的目标是从图像中识别从属类别。这可能会影响 MLLMs 更高级的能力，例如以对象为中心的视觉问答和推理。在我们的研究中，我们重新审视了 MLLMs 在 FGVR 方面的三种核心能力，包括对象信息提取、类别知识储备、对象与类别的对齐，以及将其视为对齐不匹配问题的根本原因。为了解决这一问题，我们提出了 Finedefics，这是一种通过在训练阶段整合对象的 informative 属性描述来提升模型 FGVR 能力的 MLLM。我们同时在对象-属性对和属性-类别对上采用对比学习，并使用来自相似但不正确的类别的例子作为硬负样本，自然地使视觉对象的表示形式和类别名称更接近。在多个流行 FGVR 数据集上的广泛评估表明，Finedefics 在与现有大小相当的 MLLMs 中表现出色，展现了其卓越的效果。源代码可通过以下链接获取：this https URL。 

---
# PatentLMM: Large Multimodal Model for Generating Descriptions for Patent Figures 

**Title (ZH)**: PatentLMM：大型多模态模型，用于生成专利图表的描述 

**Authors**: Shreya Shukla, Nakul Sharma, Manish Gupta, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15074)  

**Abstract**: Writing comprehensive and accurate descriptions of technical drawings in patent documents is crucial to effective knowledge sharing and enabling the replication and protection of intellectual property. However, automation of this task has been largely overlooked by the research community. To this end, we introduce PatentDesc-355K, a novel large-scale dataset containing ~355K patent figures along with their brief and detailed textual descriptions extracted from more than 60K US patent documents. In addition, we propose PatentLMM - a novel multimodal large language model specifically tailored to generate high-quality descriptions of patent figures. Our proposed PatentLMM comprises two key components: (i) PatentMME, a specialized multimodal vision encoder that captures the unique structural elements of patent figures, and (ii) PatentLLaMA, a domain-adapted version of LLaMA fine-tuned on a large collection of patents. Extensive experiments demonstrate that training a vision encoder specifically designed for patent figures significantly boosts the performance, generating coherent descriptions compared to fine-tuning similar-sized off-the-shelf multimodal models. PatentDesc-355K and PatentLMM pave the way for automating the understanding of patent figures, enabling efficient knowledge sharing and faster drafting of patent documents. We make the code and data publicly available. 

**Abstract (ZH)**: 在专利文件中对技术图纸进行全面准确的描述对于有效的知识共享、专利的复制和知识产权保护至关重要。然而，这一任务的自动化在研究界尚未受到足够的关注。为此，我们引入了包含约35.5万个专利图形及其从超过6万份美国专利文件中提取的简短和详细文本描述的新颖大规模数据集——PatentDesc-355K。此外，我们提出了一种专门针对专利图形生成高质量描述的新型多模态大语言模型——PatentLMM。我们的PatentLMM包含两大关键组成部分：(i) PatentMME，一种专门针对专利图形的多模态视觉编码器，能够捕捉专利图形的独特结构元素；(ii) PatentLLaMA，一个经过大规模专利数据微调的LLaMA领域适配版本。大量实验表明，针对专利图形设计的视觉编码器显著提升了性能，生成的描述更具连贯性，相较于微调大小相近的现成多模态模型。PatentDesc-355K和PatentLMM为自动化理解和处理专利图形铺平了道路，促进了高效的知识共享并加快了专利文件的起草。我们公开了代码和数据。 

---
# Evaluating Hallucination in Large Vision-Language Models based on Context-Aware Object Similarities 

**Title (ZH)**: 基于上下文感知对象相似性的大型视觉-语言模型 hallucination 评价 

**Authors**: Shounak Datta, Dhanasekar Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2501.15046)  

**Abstract**: Despite their impressive performance on multi-modal tasks, large vision-language models (LVLMs) tend to suffer from hallucinations. An important type is object hallucination, where LVLMs generate objects that are inconsistent with the images shown to the model. Existing works typically attempt to quantify object hallucinations by detecting and measuring the fraction of hallucinated objects in generated captions. Additionally, more recent work also measures object hallucinations by directly querying the LVLM with binary questions about the presence of likely hallucinated objects based on object statistics like top-k frequent objects and top-k co-occurring objects. In this paper, we present Context-Aware Object Similarities (CAOS), a novel approach for evaluating object hallucination in LVLMs using object statistics as well as the generated captions. CAOS uniquely integrates object statistics with semantic relationships between objects in captions and ground-truth data. Moreover, existing approaches usually only detect and measure hallucinations belonging to a predetermined set of in-domain objects (typically the set of all ground-truth objects for the training dataset) and ignore generated objects that are not part of this set, leading to under-evaluation. To address this, we further employ language model--based object recognition to detect potentially out-of-domain hallucinated objects and use an ensemble of LVLMs for verifying the presence of such objects in the query image. CAOS also examines the sequential dynamics of object generation, shedding light on how the order of object appearance influences hallucinations, and employs word embedding models to analyze the semantic reasons behind hallucinations. CAOS aims to offer a nuanced understanding of the hallucination tendencies of LVLMs by providing a systematic framework to identify and interpret object hallucinations. 

**Abstract (ZH)**: 尽管大规模视觉-语言模型（LVLMs）在多模态任务中表现出色，但它们往往容易出现幻觉现象。其中一种重要类型的幻觉是物体幻觉，即LVLMs生成与提供的图像不一致的物体。现有方法通常通过检测并衡量生成字幕中幻象物体的比例来量化物体幻觉。此外，最近的一些研究直接通过针对物体统计信息（如最常见前k个物体和最常共现前k个物体）中的可能幻象物体提出二元问题来衡量物体幻觉。在本文中，我们提出了上下文感知物体相似性（CAOS，Context-Aware Object Similarities）方法，这是一种利用物体统计信息和生成字幕评估LVLMs中物体幻觉的新颖方法。CAOS独特地将物体统计信息与字幕中的语义关系以及真实数据相结合。此外，现有方法通常只检测并衡量属于某一预设领域物体集（通常是训练数据集中所有真实物体集）中的幻觉物体，并忽略不属于这一集的生成物体，从而导致评估不足。为解决这一问题，我们进一步利用基于语言模型的物体识别来检测潜在的跨领域幻觉物体，并使用多个LVLM的集合来验证查询图像中这些物体的存在性。CAOS还探讨了物体生成的顺序动态，揭示了物体出现顺序如何影响幻觉，并利用词嵌入模型分析幻觉的根本语义原因。CAOS旨在通过提供系统的方法来识别和解释物体幻觉，为理解LVLMs的幻觉倾向提供细腻的理解。 

---
# Multi-Modality Transformer for E-Commerce: Inferring User Purchase Intention to Bridge the Query-Product Gap 

**Title (ZH)**: 面向电子商务的多模态变压器：推断用户购买意图以缩短查询与产品之间的差距 

**Authors**: Srivatsa Mallapragada, Ying Xie, Varsha Rani Chawan, Zeyad Hailat, Yuanbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14826)  

**Abstract**: E-commerce click-stream data and product catalogs offer critical user behavior insights and product knowledge. This paper propose a multi-modal transformer termed as PINCER, that leverages the above data sources to transform initial user queries into pseudo-product representations. By tapping into these external data sources, our model can infer users' potential purchase intent from their limited queries and capture query relevant product features. We demonstrate our model's superior performance over state-of-the-art alternatives on e-commerce online retrieval in both controlled and real-world experiments. Our ablation studies confirm that the proposed transformer architecture and integrated learning strategies enable the mining of key data sources to infer purchase intent, extract product features, and enhance the transformation pipeline from queries to more accurate pseudo-product representations. 

**Abstract (ZH)**: 电子商务点击流数据和产品目录提供了关键的用户行为见解和产品知识。本文提出了一种名为PINCE的多模态变压器，该模型利用上述数据源将初始用户查询转化为伪产品表示。通过利用这些外部数据源，我们的模型可以从用户有限的查询中推断出用户的潜在购买意图，并捕获查询相关的商品特征。我们在受控和真实世界实验中展示了该模型在电子商务在线检索方面的优越性能，与最先进的替代方案相比，其表现更为出色。我们的消融研究证实，提出的变压器架构和集成学习策略能够挖掘关键数据源以推断购买意图、提取产品特征，并增强从查询到更准确伪产品表示的转换管道。 

---
# Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models 

**Title (ZH)**: Eagle 2：从零构建前沿视觉语言模型的后训练数据策略 

**Authors**: Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, Nadine Chang, Karan Sapra, Amala Sanjay Deshmukh, Tuomas Rintamaki, Matthieu Le, Ilia Karmanov, Lukas Voegtle, Philipp Fischer, De-An Huang, Timo Roman, Tong Lu, Jose M. Alvarez, Bryan Catanzaro, Jan Kautz, Andrew Tao, Guilin Liu, Zhiding Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14818)  

**Abstract**: Recently, promising progress has been made by open-source vision-language models (VLMs) in bringing their capabilities closer to those of proprietary frontier models. However, most open-source models only publish their final model weights, leaving the critical details of data strategies and implementation largely opaque. In this work, we address VLM post-training from a data-centric perspective, showing the key role of data strategy in developing frontier VLMs. By studying and building our post-training data strategy from scratch, we share detailed insights into the development processes, aiming to benefit the development of competitive models for the open-source community. Our introduced data strategy, together with training recipes and model design, leads to a family of performant VLMs named Eagle2. Specifically, Eagle2-9B achieves state-of-the-art results across various multimodal benchmarks, matching certain competitive models with up to 70B parameters. 

**Abstract (ZH)**: 近年来，开源视觉-语言模型（VLMs）在使其能力接近专有先进模型方面取得了令人鼓舞的进展。然而，大多数开源模型仅发布其最终模型权重，而数据策略和实现的许多关键细节则相对不透明。在本项工作中，我们从数据为中心的角度研究了VLM的后训练方法，展示了数据策略在开发先进VLM中的关键作用。通过从头研究和构建后训练的数据策略，我们分享了开发过程中的详细见解，旨在为开源社区开发具有竞争力的模型提供帮助。我们引入的数据策略，结合训练方法和模型设计，产生了一系列表现优异的VLMs，命名为Eagle2。具体而言，Eagle2-9B在各种多模态基准测试中取得了最先进的结果，与包含多达700亿参数的竞争模型相当。 

---
# Towards Dynamic Neural Communication and Speech Neuroprosthesis Based on Viseme Decoding 

**Title (ZH)**: 基于唇型解码的动态神经通信与语音神经假体研究 

**Authors**: Ji-Ha Park, Seo-Hyun Lee, Soowon Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.14790)  

**Abstract**: Decoding text, speech, or images from human neural signals holds promising potential both as neuroprosthesis for patients and as innovative communication tools for general users. Although neural signals contain various information on speech intentions, movements, and phonetic details, generating informative outputs from them remains challenging, with mostly focusing on decoding short intentions or producing fragmented outputs. In this study, we developed a diffusion model-based framework to decode visual speech intentions from speech-related non-invasive brain signals, to facilitate face-to-face neural communication. We designed an experiment to consolidate various phonemes to train visemes of each phoneme, aiming to learn the representation of corresponding lip formations from neural signals. By decoding visemes from both isolated trials and continuous sentences, we successfully reconstructed coherent lip movements, effectively bridging the gap between brain signals and dynamic visual interfaces. The results highlight the potential of viseme decoding and talking face reconstruction from human neural signals, marking a significant step toward dynamic neural communication systems and speech neuroprosthesis for patients. 

**Abstract (ZH)**: 从人类神经信号解码文本、语音或图像在患者神经假体和普通用户的创新通信工具方面都展现了广阔的可能性。尽管神经信号包含了关于语音意图、动作和音素细节的多种信息，但从这些信号生成具有信息量的输出仍然是一个挑战，大多数研究集中在解码短暂的意图或生成片段化的输出。在本研究中，我们开发了一种基于扩散模型的框架，从与语音相关的非侵入性脑信号中解码视觉语音意图，以促进面对面的神经通信。我们设计了一个实验，将各种音素整合起来训练每个音素的唇形，旨在从神经信号中学习相应唇形的表示。通过从孤立试次和连续句子的解码结果中，我们成功地重建了连贯的唇部运动，有效地填补了脑信号与动态视觉界面之间的差距。研究结果突显了从人类神经信号解码唇形和重构说话语音的潜力，标志着动态神经通信系统和患者语音神经假体发展的重要进展。 

---
# Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for Foundation Models 

**Title (ZH)**: Data-Juicer 2.0：面向基础模型的云规模自适应数据处理 

**Authors**: Daoyuan Chen, Yilun Huang, Xuchen Pan, Nana Jiang, Haibin Wang, Ce Ge, Yushuo Chen, Wenhao Zhang, Zhijian Ma, Yilei Zhang, Jun Huang, Wei Lin, Yaliang Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.14755)  

**Abstract**: The burgeoning field of foundation models necessitates advanced data processing mechanisms capable of harnessing vast valuable data with varied types utilized by these models. Nevertheless, the current landscape presents unique challenges that traditional data processing frameworks cannot handle effectively, especially with multimodal intricacies. In response, we present Data-Juicer 2.0, a new system offering fruitful data processing capabilities backed by over a hundred operators spanning various modalities like text, image, audio, and video. With seamless compatibility and dedicated optimization to popular dataset hubs like Hugging Face and computing engines like Ray, Data-Juicer 2.0 enhances its predecessor in both usability, efficiency, and programmability. It features an easily accessible user interface layer that supports decoupled Python interactions, RESTful APIs, and conversational commands. Alongside this, it contains a core runtime layer optimized for adaptive execution and management across different dataset scales, processing demands, and computational environments, while shielding unnecessary system details. Extensive empirical evaluations demonstrate Data-Juicer 2.0's remarkable performance and scalability, highlighting its capability to efficiently process tens of billions of data samples with tens of thousands of CPU cores. The system is publicly available, actively maintained, and broadly adopted in diverse research endeavors, practical applications, and real-world products such as Alibaba Cloud PAI. 

**Abstract (ZH)**: 础模型领域的蓬勃发展催生了能够利用各种类型海量有价值数据的先进数据处理机制。然而，当前的数据处理框架在应对多模态复杂性方面仍存在独特挑战，传统框架无法有效应对。为应对这一挑战，我们推出了Data-Juicer 2.0，这是一个全新的系统，提供了丰富的数据处理能力，涵盖了超过一百种操作符，适用于文本、图像、音频和视频等多种模态。Data-Juicer 2.0 通过无缝兼容性和针对流行的如 Hugging Face 数据集库和计算引擎如 Ray 的专门优化，在易用性、效率和编程性方面提升了其前身。它包含一个易于访问的用户界面层，支持脱钩的 Python 交互、RESTful API 和会话命令。此外，它还包含一个核心运行时层，针对不同数据集规模、处理需求和计算环境进行了优化管理，并屏蔽了不必要的系统细节。广泛的实证评估表明，Data-Juicer 2.0 具有出色的性能和可扩展性，能够高效处理数十亿条数据样本和数万个 CPU 核心。该系统已公开提供，并被积极维护和广泛应用于各种研究项目、实际应用和真实世界的产品中，例如阿里云 PAI。 

---
# Unveiling the Potential of Multimodal Retrieval Augmented Generation with Planning 

**Title (ZH)**: 探索规划增强的多模态检索生成的潜力 

**Authors**: Xiaohan Yu, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15470)  

**Abstract**: Multimodal Retrieval Augmented Generation (MRAG) systems, while promising for enhancing Multimodal Large Language Models (MLLMs), often rely on rigid, single-step retrieval methods. This limitation hinders their ability to effectively address real-world scenarios that demand adaptive information acquisition and query refinement. To overcome this, we introduce the novel task of Multimodal Retrieval Augmented Generation Planning (MRAG Planning), focusing on optimizing MLLM performance while minimizing computational overhead. We present CogPlanner, a versatile framework inspired by human cognitive processes. CogPlanner iteratively refines queries and selects retrieval strategies, enabling both parallel and sequential modeling approaches. To rigorously evaluate MRAG Planning, we introduce CogBench, a new benchmark specifically designed for this task. CogBench facilitates the integration of lightweight CogPlanner with resource-efficient MLLMs. Our experimental findings demonstrate that CogPlanner surpasses existing MRAG baselines, achieving significant improvements in both accuracy and efficiency with minimal computational overhead. 

**Abstract (ZH)**: 多模态检索增强生成（MRAG）系统虽然在增强多模态大型语言模型（MLLMs）方面潜力巨大，但往往依赖于僵化的单一步骤检索方法。这一局限性限制了其在需要适应性信息获取和查询优化的实际场景中的能力。为解决这一问题，我们提出了一个新的任务——多模态检索增强生成规划（MRAG Planning），旨在优化MLLM性能的同时减小计算开销。我们提出了CogPlanner这一多功能框架，灵感源自人类的认知过程。CogPlanner通过迭代优化查询和选择检索策略，支持并行和序列模型方法。为了严格评估MRAG Planning，我们引入了CogBench，这是一种专门为此任务设计的新基准。CogBench 使轻量级的CogPlanner与资源高效的MLLMs的集成变得容易。我们的实验结果表明，CogPlanner超越了现有的MRAG基准，实现了在准确性和效率方面的显著提升，且计算开销最小。 

---
# Generating Negative Samples for Multi-Modal Recommendation 

**Title (ZH)**: 多模态推荐中的负样本生成 

**Authors**: Yanbiao Ji, Yue Ding, Dan Luo, Chang Liu, Jing Tong, Shaokai Wi, Hongtao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15183)  

**Abstract**: Multi-modal recommender systems (MMRS) have gained significant attention due to their ability to leverage information from various modalities to enhance recommendation quality. However, existing negative sampling techniques often struggle to effectively utilize the multi-modal data, leading to suboptimal performance. In this paper, we identify two key challenges in negative sampling for MMRS: (1) producing cohesive negative samples contrasting with positive samples and (2) maintaining a balanced influence across different modalities. To address these challenges, we propose NegGen, a novel framework that utilizes multi-modal large language models (MLLMs) to generate balanced and contrastive negative samples. We design three different prompt templates to enable NegGen to analyze and manipulate item attributes across multiple modalities, and then generate negative samples that introduce better supervision signals and ensure modality balance. Furthermore, NegGen employs a causal learning module to disentangle the effect of intervened key features and irrelevant item attributes, enabling fine-grained learning of user preferences. Extensive experiments on real-world datasets demonstrate the superior performance of NegGen compared to state-of-the-art methods in both negative sampling and multi-modal recommendation. 

**Abstract (ZH)**: 多模态推荐系统（MMRS）由于能够利用多种模态的信息来提高推荐质量而引起了广泛关注。然而，现有的负样本生成技术往往难以有效利用多模态数据，导致性能不佳。本文识别了多模态推荐系统中负样本生成的两个关键挑战：（1）生成与正样本形成对比的连贯负样本，以及（2）在不同模态间维持平衡的影响。为了解决这些问题，我们提出了一种名为NegGen的新框架，该框架利用多模态大型语言模型（MLLMs）生成平衡且对比式的负样本。我们设计了三种不同的提示模板，以使NegGen能够跨多个模态分析和操纵项目属性，并进一步生成能够引入更多监督信号并确保模态平衡的负样本。此外，NegGen采用因果学习模块来分离干预关键特征和无关项目属性的作用，从而实现细粒度的用户偏好学习。在实际数据集上进行的广泛实验表明，NegGen在负样本生成和多模态推荐方面均优于现有最先进的方法。 

---
# Towards Explainable Multimodal Depression Recognition for Clinical Interviews 

**Title (ZH)**: 面向可解释的多模态抑郁识别在临床访谈中的应用 

**Authors**: Wenjie Zheng, Qiming Xie, Zengzhi Wang, Jianfei Yu, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2501.16106)  

**Abstract**: Recently, multimodal depression recognition for clinical interviews (MDRC) has recently attracted considerable attention. Existing MDRC studies mainly focus on improving task performance and have achieved significant development. However, for clinical applications, model transparency is critical, and previous works ignore the interpretability of decision-making processes. To address this issue, we propose an Explainable Multimodal Depression Recognition for Clinical Interviews (EMDRC) task, which aims to provide evidence for depression recognition by summarizing symptoms and uncovering underlying causes. Given an interviewer-participant interaction scenario, the goal of EMDRC is to structured summarize participant's symptoms based on the eight-item Patient Health Questionnaire depression scale (PHQ-8), and predict their depression severity. To tackle the EMDRC task, we construct a new dataset based on an existing MDRC dataset. Moreover, we utilize the PHQ-8 and propose a PHQ-aware multimodal multi-task learning framework, which captures the utterance-level symptom-related semantic information to help generate dialogue-level summary. Experiment results on our annotated dataset demonstrate the superiority of our proposed methods over baseline systems on the EMDRC task. 

**Abstract (ZH)**: 近年来，临床访谈中的多模态抑郁识别（MDRC）受到了广泛关注。现有MDRC研究主要侧重于提高任务性能，并已取得显著进展。然而，在临床应用中，模型的透明度至关重要，而以往的研究忽略了决策过程的可解释性。为解决这一问题，我们提出了一项名为可解释的多模态抑郁识别临床访谈（EMDRC）任务，该任务旨在通过总结症状和揭示潜在原因来为抑郁识别提供依据。给定访谈者-参与者互动场景，EMDRC的目标是基于患者健康问卷抑郁量表（PHQ-8）的八项指标，对参与者的症状进行结构化总结，并预测其抑郁严重程度。为应对此任务，我们基于现有的MDRC数据集构建了一个新的数据集，并利用PHQ-8提出了一个PHQ意识多模态多任务学习框架，该框架捕捉单个语句层面的症状相关语义信息，以帮助生成对话级摘要。我们在标记数据集上的实验结果表明，所提出的方法在EMDRC任务上的性能优于基准系统。 

---
# Baichuan-Omni-1.5 Technical Report 

**Title (ZH)**: 《Baichuan-Omni-1.5 技术报告》

解释：这里的“Baichuan-Omni-1.5”看起来是一个技术名称或系统名称，因此在翻译时保持了原名，仅将“Technical Report”翻译为“技术报告”，以符合学术文献的规范。 

**Authors**: Yadong Li, Jun Liu, Tao Zhang, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, Chong Li, Yuanbo Fang, Dongdong Kuang, Mingrui Wang, Chenglin Zhu, Youwei Zhang, Hongyu Guo, Fengyu Zhang, Yuran Wang, Bowen Ding, Wei Song, Xu Li, Yuqi Huo, Zheng Liang, Shusen Zhang, Xin Wu, Shuai Zhao, Linchu Xiong, Yozhen Wu, Jiahui Ye, Wenhao Lu, Bowen Li, Yan Zhang, Yaqi Zhou, Xin Chen, Lei Su, Hongda Zhang, Fuzhong Chen, Xuezhen Dong, Na Nie, Zhiying Wu, Bin Xiao, Ting Li, Shunya Dang, Ping Zhang, Yijia Sun, Jincheng Wu, Jinjie Yang, Xionghai Lin, Zhi Ma, Kegeng Wu, Jia li, Aiyuan Yang, Hui Liu, Jianqiang Zhang, Xiaoxi Chen, Guangwei Ai, Wentao Zhang, Yicong Chen, Xiaoqin Huang, Kun Li, Wenjing Luo, Yifei Duan, Lingling Zhu, Ran Xiao, Zhe Su, Jiani Pu, Dian Wang, Xu Jia, Tianyu Zhang, Mengyu Ai, Mang Wang, Yujing Qiao, Lei Zhang, Yanjun Shen, Fan Yang, Miao Zhen, Yijie Zhou, Mingyang Chen, Fei Li, Chenzheng Zhu, Keer Lu, Yaqi Zhao, Hao Liang, Youquan Li, Yanzhao Qin, Linzhuang Sun, Jianhua Xu, Haoze Sun, Mingan Lin, Zenan Zhou, Weipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15368)  

**Abstract**: We introduce Baichuan-Omni-1.5, an omni-modal model that not only has omni-modal understanding capabilities but also provides end-to-end audio generation capabilities. To achieve fluent and high-quality interaction across modalities without compromising the capabilities of any modality, we prioritized optimizing three key aspects. First, we establish a comprehensive data cleaning and synthesis pipeline for multimodal data, obtaining about 500B high-quality data (text, audio, and vision). Second, an audio-tokenizer (Baichuan-Audio-Tokenizer) has been designed to capture both semantic and acoustic information from audio, enabling seamless integration and enhanced compatibility with MLLM. Lastly, we designed a multi-stage training strategy that progressively integrates multimodal alignment and multitask fine-tuning, ensuring effective synergy across all modalities. Baichuan-Omni-1.5 leads contemporary models (including GPT4o-mini and MiniCPM-o 2.6) in terms of comprehensive omni-modal capabilities. Notably, it achieves results comparable to leading models such as Qwen2-VL-72B across various multimodal medical benchmarks. 

**Abstract (ZH)**: 我们将介绍Baichuan-Omni-1.5这一全方位模型，它不仅具备全方位的理解能力，还提供了端到端的语音生成能力。为实现不同模态间的流畅和高质量交互，而又不牺牲任何模态的能力，我们着重优化了三个关键方面。首先，我们建立了一个全面的数据清洗和合成管道，获得了约500亿条高质量数据（包括文本、音频和视觉）。其次，我们设计了一个语音分词器（Baichuan-Audio-Tokenizer），能够捕捉音频中的语义和声音信息，从而实现无缝集成和增强与大规模语言模型（MLLM）的兼容性。最后，我们设计了一种多阶段训练策略，逐步整合多模态对齐和多任务微调，确保各模态之间的有效协同。在全面的多模态能力方面，Baichuan-Omni-1.5超越了包括GPT4o-mini和MiniCPM-o 2.6在内的当前模型。特别是在各种多模态医疗基准测试中，它达到了与Qwen2-VL-72B等领先模型相当的结果。 

---
# Figurative-cum-Commonsense Knowledge Infusion for Multimodal Mental Health Meme Classification 

**Title (ZH)**: 具象化与常识知识融合在多模态心理健康 meme 分类中的应用 

**Authors**: Abdullah Mazhar, Zuhair hasan shaik, Aseem Srivastava, Polly Ruhnke, Lavanya Vaddavalli, Sri Keshav Katragadda, Shweta Yadav, Md Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2501.15321)  

**Abstract**: The expression of mental health symptoms through non-traditional means, such as memes, has gained remarkable attention over the past few years, with users often highlighting their mental health struggles through figurative intricacies within memes. While humans rely on commonsense knowledge to interpret these complex expressions, current Multimodal Language Models (MLMs) struggle to capture these figurative aspects inherent in memes. To address this gap, we introduce a novel dataset, AxiOM, derived from the GAD anxiety questionnaire, which categorizes memes into six fine-grained anxiety symptoms. Next, we propose a commonsense and domain-enriched framework, M3H, to enhance MLMs' ability to interpret figurative language and commonsense knowledge. The overarching goal remains to first understand and then classify the mental health symptoms expressed in memes. We benchmark M3H against 6 competitive baselines (with 20 variations), demonstrating improvements in both quantitative and qualitative metrics, including a detailed human evaluation. We observe a clear improvement of 4.20% and 4.66% on weighted-F1 metric. To assess the generalizability, we perform extensive experiments on a public dataset, RESTORE, for depressive symptom identification, presenting an extensive ablation study that highlights the contribution of each module in both datasets. Our findings reveal limitations in existing models and the advantage of employing commonsense to enhance figurative understanding. 

**Abstract (ZH)**: 近年来，通过非传统方式（如表情包）表达心理健康症状引起了广泛关注，用户常通过表情包中的象征性细节来突出他们的心理健康困境。尽管人类依赖常识来解释这些复杂的表达，当前的多模态语言模型（MLMs）难以捕捉到表情包中固有的象征性方面。为了解决这一问题，我们引入了一个新的数据集AxiOM，该数据集源自GAD焦虑问卷，将表情包分类为六种细粒度的焦虑症状。随后，我们提出了一种结合常识和领域增强框架M3H，以提升MLMs理解和解释象征性语言及常识的能力。我们的总体目标是首先理解和然后分类在表情包中表达的心理健康症状。我们用M3H与6种竞争基线（包括20种变体）进行了基准测试，结果在定量和定性指标上均有所提高，包括详细的_human评估。我们在_weighted-F1指标上观察到了明显改进，分别为4.20%和4.66%。为评估模型的普适性，我们在一个公开数据集RESTORE上进行了广泛的实验，用于抑郁症症状识别，并进行了一项详尽的消融研究，突显了各模块在两个数据集中的贡献。我们的研究发现现有模型的局限性，并展示了运用常识增强象征性理解的优势。 

---
# Cross-modal Context Fusion and Adaptive Graph Convolutional Network for Multimodal Conversational Emotion Recognition 

**Title (ZH)**: 跨模态上下文融合与自适应图卷积网络在多模态对话情感识别中的应用 

**Authors**: Junwei Feng, Xueyan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15063)  

**Abstract**: Emotion recognition has a wide range of applications in human-computer interaction, marketing, healthcare, and other fields. In recent years, the development of deep learning technology has provided new methods for emotion recognition. Prior to this, many emotion recognition methods have been proposed, including multimodal emotion recognition methods, but these methods ignore the mutual interference between different input modalities and pay little attention to the directional dialogue between speakers. Therefore, this article proposes a new multimodal emotion recognition method, including a cross modal context fusion module, an adaptive graph convolutional encoding module, and an emotion classification module. The cross modal context module includes a cross modal alignment module and a context fusion module, which are used to reduce the noise introduced by mutual interference between different input modalities. The adaptive graph convolution module constructs a dialogue relationship graph for extracting dependencies and self dependencies between speakers. Our model has surpassed some state-of-the-art methods on publicly available benchmark datasets and achieved high recognition accuracy. 

**Abstract (ZH)**: 情感识别在人机交互、市场营销、医疗保健等多个领域有着广泛的应用。近年来，深度学习技术的发展为情感识别提供了新的方法。在此之前，已经提出了一些情感识别方法，包括多模态情感识别方法，但这些方法忽略了不同输入模态之间的相互干扰，并且很少关注对话者之间的双向对话。因此，本文提出了一种新的多模态情感识别方法，包括跨模态上下文融合模块、自适应图卷积编码模块和情感分类模块。跨模态上下文模块包括跨模态对齐模块和上下文融合模块，用于减少不同输入模态之间相互干扰引入的噪声。自适应图卷积模块构建了一个对话关系图，用于提取对话者之间的依赖关系和自我依赖关系。我们的模型在公开可用的标准数据集上超过了某些最先进的方法，并达到了较高的识别精度。 

---
# AKVQ-VL: Attention-Aware KV Cache Adaptive 2-Bit Quantization for Vision-Language Models 

**Title (ZH)**: AKVQ-VL：面向注意力意识的键值缓存自适应2位量化视觉-语言模型 

**Authors**: Zunhai Su, Wang Shen, Linge Li, Zhe Chen, Hanyu Wei, Huangqi Yu, Kehong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15021)  

**Abstract**: Vision-language models (VLMs) show remarkable performance in multimodal tasks. However, excessively long multimodal inputs lead to oversized Key-Value (KV) caches, resulting in significant memory consumption and I/O bottlenecks. Previous KV quantization methods for Large Language Models (LLMs) may alleviate these issues but overlook the attention saliency differences of multimodal tokens, resulting in suboptimal performance. In this paper, we investigate the attention-aware token saliency patterns in VLM and propose AKVQ-VL. AKVQ-VL leverages the proposed Text-Salient Attention (TSA) and Pivot-Token-Salient Attention (PSA) patterns to adaptively allocate bit budgets. Moreover, achieving extremely low-bit quantization requires effectively addressing outliers in KV tensors. AKVQ-VL utilizes the Walsh-Hadamard transform (WHT) to construct outlier-free KV caches, thereby reducing quantization difficulty. Evaluations of 2-bit quantization on 12 long-context and multimodal tasks demonstrate that AKVQ-VL maintains or even improves accuracy, outperforming LLM-oriented methods. AKVQ-VL can reduce peak memory usage by 2.13x, support up to 3.25x larger batch sizes and 2.46x throughput. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在多模态任务中表现出色。然而，过长的多模态输入会导致关键值（KV）缓存过大，从而引起显著的内存消耗和输入输出瓶颈。先前用于大型语言模型（LLMs）的关键值量化方法可能缓解这些问题，但忽略了多模态令牌的注意力重要性差异，导致性能不佳。在本文中，我们研究了VLM中的注意力感知令牌重要性模式，并提出了AKVQ-VL方法。AKVQ-VL利用提出的文本注意力显著模式（TSA）和枢轴令牌注意力显著模式（PSA）来自适应分配比特预算。此外，实现极低比特量化需要有效解决KV张量中的异常值。AKVQ-VL利用沃尔什-哈达玛变换（WHT）构造出无异常值的KV缓存，从而降低了量化难度。在12个长上下文和多模态任务中进行的2比特量化评估显示，AKVQ-VL不仅保持甚至提高了精度，性能优于针对LLM的量化方法。AKVQ-VL可将峰值内存使用量减少2.13倍，支持多达3.25倍更大的批量大小和2.46倍的吞吐量。 

---
# DrawEduMath: Evaluating Vision Language Models with Expert-Annotated Students' Hand-Drawn Math Images 

**Title (ZH)**: DrawEduMath: 专家标注的学生手绘数学图像评价视觉语言模型 

**Authors**: Sami Baral, Li Lucy, Ryan Knight, Alice Ng, Luca Soldaini, Neil T. Heffernan, Kyle Lo  

**Link**: [PDF](https://arxiv.org/pdf/2501.14877)  

**Abstract**: In real-world settings, vision language models (VLMs) should robustly handle naturalistic, noisy visual content as well as domain-specific language and concepts. For example, K-12 educators using digital learning platforms may need to examine and provide feedback across many images of students' math work. To assess the potential of VLMs to support educators in settings like this one, we introduce DrawEduMath, an English-language dataset of 2,030 images of students' handwritten responses to K-12 math problems. Teachers provided detailed annotations, including free-form descriptions of each image and 11,661 question-answer (QA) pairs. These annotations capture a wealth of pedagogical insights, ranging from students' problem-solving strategies to the composition of their drawings, diagrams, and writing. We evaluate VLMs on teachers' QA pairs, as well as 44,362 synthetic QA pairs derived from teachers' descriptions using language models (LMs). We show that even state-of-the-art VLMs leave much room for improvement on DrawEduMath questions. We also find that synthetic QAs, though imperfect, can yield similar model rankings as teacher-written QAs. We release DrawEduMath to support the evaluation of VLMs' abilities to reason mathematically over images gathered with educational contexts in mind. 

**Abstract (ZH)**: 在实际应用场景中，视觉语言模型（VLMs）应当能够稳健地处理自然且不规则的视觉内容，同时处理特定领域的语言和概念。例如，K-12 教育者在使用数字化学习平台时，可能需要审阅和提供反馈，涉及许多学生的数学作业图片。为了评估 VLMs 在类似这样环境中支持教育者的能力，我们引入了 DrawEduMath，这是一个包含 2,030 张学生手工解答 K-12 数学问题的图像的英语数据集。老师提供了详细的注释，包括每张图像的自由格式描述和 11,661 组问题-答案（QA）对。这些注释捕捉了丰富的教学见解，从学生的解题策略到他们绘制的图表和写作的组成。我们使用老师提供的 QA 对以及从老师描述中生成的 44,362 组合成的 QA 对（使用语言模型生成）来评估 VLMs 的性能。结果显示，即使是最先进的 VLMs 在处理 DrawEduMath 问题时仍有很大的改进空间。我们还发现，尽管合成的 QA 并不完美，但它们可以与教师撰写的 QA 对一样生成类似模型的排名。我们发布 DrawEduMath 以支持评估 VLMs 在教育情境下对图像进行数学推理的能力。 

---
# ConceptCLIP: Towards Trustworthy Medical AI via Concept-Enhanced Contrastive Langauge-Image Pre-training 

**Title (ZH)**: ConceptCLIP：通过概念增强对比语言-图像预训练实现可靠的医疗AI 

**Authors**: Yuxiang Nie, Sunan He, Yequan Bie, Yihui Wang, Zhixuan Chen, Shu Yang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15579)  

**Abstract**: Trustworthiness is essential for the precise and interpretable application of artificial intelligence (AI) in medical imaging. Traditionally, precision and interpretability have been addressed as separate tasks, namely medical image analysis and explainable AI, each developing its own models independently. In this study, for the first time, we investigate the development of a unified medical vision-language pre-training model that can achieve both accurate analysis and interpretable understanding of medical images across various modalities. To build the model, we construct MedConcept-23M, a large-scale dataset comprising 23 million medical image-text pairs extracted from 6.2 million scientific articles, enriched with concepts from the Unified Medical Language System (UMLS). Based on MedConcept-23M, we introduce ConceptCLIP, a medical AI model utilizing concept-enhanced contrastive language-image pre-training. The pre-training of ConceptCLIP involves two primary components: image-text alignment learning (IT-Align) and patch-concept alignment learning (PC-Align). This dual alignment strategy enhances the model's capability to associate specific image regions with relevant concepts, thereby improving both the precision of analysis and the interpretability of the AI system. We conducted extensive experiments on 5 diverse types of medical image analysis tasks, spanning 51 subtasks across 10 image modalities, with the broadest range of downstream tasks. The results demonstrate the effectiveness of the proposed vision-language pre-training model. Further explainability analysis across 6 modalities reveals that ConceptCLIP achieves superior performance, underscoring its robust ability to advance explainable AI in medical imaging. These findings highlight ConceptCLIP's capability in promoting trustworthy AI in the field of medicine. 

**Abstract (ZH)**: 信任是确保人工智能（AI）在医学成像中的精确和可解释应用的关键。传统上，精度和可解释性分别作为一个独立的任务来处理，即医学图像分析和可解释AI，各自独立开发模型。在本研究中，我们首次探讨了一种统一的医学视觉-语言预训练模型的发展，该模型可以在多种模态下实现医学图像的准确分析和可解释理解。为了构建该模型，我们创建了一个大规模数据集MedConcept-23M，其中包含2300万张医学图像-文本对，这些对是从620万篇科学文章中提取出来的，并丰富了统一医学语言系统（UMLS）中的概念。基于MedConcept-23M，我们引入了ConceptCLIP，这是一种利用概念增强的对比视觉-语言预训练的医学AI模型。ConceptCLIP的预训练包括两个主要组成部分：图像-文本对齐学习（IT-Align）和片段-概念对齐学习（PC-Align）。这种双重对齐策略增强了模型将特定图像区域与相关概念关联的能力，从而提高了分析的精度和AI系统的可解释性。我们对5种不同的医学图像分析任务进行了广泛的实验，涵盖了10种图像模态下的51个子任务，这是迄今为止下游任务范围最广泛的研究。结果表明了所提出视觉-语言预训练模型的有效性。进一步在6种模态上的可解释性分析显示，ConceptCLIP表现更加优越，突显了其在医学成像中推动可解释AI发展的稳健能力。这些发现强调了ConceptCLIP在促进医学领域可信AI方面的能力。 

---
