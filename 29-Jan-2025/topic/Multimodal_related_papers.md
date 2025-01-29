# SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training 

**Title (ZH)**: SFT 存储，RL 精炼：基础模型训练后比较研究

解释：
- SFT (Fine-tuning) 存储：这里的“SFT”可能是指“fine-tuning”，即微调。微调后的模型倾向于记住特定的训练数据。
- RL (Reinforcement Learning) 精炼：使用强化学习方法进行精炼，使模型具有更好的泛化能力。
- 基础模型训练后比较研究：研究微调与强化学习方法在基础模型训练后的表现比较。 

**Authors**: Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.17161)  

**Abstract**: Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used post-training techniques for foundation models. However, their roles in enhancing model generalization capabilities remain unclear. This paper studies the difference between SFT and RL on generalization and memorization, focusing on text-based rule variants and visual variants. We introduce GeneralPoints, an arithmetic reasoning card game, and adopt V-IRL, a real-world navigation environment, to assess how models trained with SFT and RL generalize to unseen variants in both textual and visual domains. We show that RL, especially when trained with an outcome-based reward, generalizes across both rule-based textual and visual variants. SFT, in contrast, tends to memorize training data and struggles to generalize out-of-distribution scenarios. Further analysis reveals that RL improves the model's underlying visual recognition capabilities, contributing to its enhanced generalization in the visual domain. Despite RL's superior generalization, we show that SFT remains essential for effective RL training; SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains. These findings demonstrates the capability of RL for acquiring generalizable knowledge in complex, multi-modal tasks. 

**Abstract (ZH)**: 监督调优（SFT）和强化学习（RL）是广泛应用于基础模型的后训练技术。然而，它们在提高模型泛化能力方面的角色尚不清楚。本文研究了SFT和RL在泛化能力和记忆方面之间的差异，重点关注基于文本的规则变体和视觉变体。我们引入了GeneralPoints，一种算术推理卡片游戏，并采用V-IRL，一个真实世界的导航环境，评估使用SFT和RL训练的模型在文本和视觉领域中对未见过的变体的泛化能力。研究表明，与基于结果的奖励进行训练时，RL能够在基于规则的文本和视觉变体之间泛化。相比之下，SFT倾向于记忆训练数据，难以泛化到分布外场景。进一步分析表明，RL在提高模型的基础视觉识别能力方面发挥了作用，从而在视觉领域增强其泛化能力。尽管RL在泛化方面的表现优越，但我们展示了SFT对于有效的RL训练仍然是必不可少的；SFT稳定了模型的输出格式，使后续的RL能够实现其性能提升。这些发现表明，RL有能力在复杂、多模态任务中获取可泛化的知识。 

---
# FlexMotion: Lightweight, Physics-Aware, and Controllable Human Motion Generation 

**Title (ZH)**: FlexMotion：轻量级、物理意识强且可控制的人体运动生成 

**Authors**: Arvin Tashakori, Arash Tashakori, Gongbo Yang, Z. Jane Wang, Peyman Servati  

**Link**: [PDF](https://arxiv.org/pdf/2501.16778)  

**Abstract**: Lightweight, controllable, and physically plausible human motion synthesis is crucial for animation, virtual reality, robotics, and human-computer interaction applications. Existing methods often compromise between computational efficiency, physical realism, or spatial controllability. We propose FlexMotion, a novel framework that leverages a computationally lightweight diffusion model operating in the latent space, eliminating the need for physics simulators and enabling fast and efficient training. FlexMotion employs a multimodal pre-trained Transformer encoder-decoder, integrating joint locations, contact forces, joint actuations and muscle activations to ensure the physical plausibility of the generated motions. FlexMotion also introduces a plug-and-play module, which adds spatial controllability over a range of motion parameters (e.g., joint locations, joint actuations, contact forces, and muscle activations). Our framework achieves realistic motion generation with improved efficiency and control, setting a new benchmark for human motion synthesis. We evaluate FlexMotion on extended datasets and demonstrate its superior performance in terms of realism, physical plausibility, and controllability. 

**Abstract (ZH)**: 轻量级、可控且物理上合理的真人动作合成对于动画、虚拟现实、机器人技术和人机交互应用至关重要。现有方法常常在计算效率、物理逼真度或空间可控性之间进行权衡。我们提出了一种名为 FlexMotion 的新型框架，该框架利用在潜在空间中操作的轻量级扩散模型，消除了对物理模拟器的依赖，并实现了快速高效的训练。FlexMotion 融合了多模态预训练Transformer编码器-解码器，整合了关节位置、接触力、关节驱动和肌肉激活等信息，确保生成动作的物理合理性。FlexMotion 还引入了一个即插即用模块，增强了在一系列动作参数（如关节位置、关节驱动、接触力和肌肉激活）上的空间可控性。我们的框架在效率和控制方面实现了逼真的动作生成，并建立了真人动作合成的新基准。我们在扩展数据集上评估了 FlexMotion，并展示了其在逼真度、物理合理性以及可控性方面的优越性能。 

---
# An LLM Benchmark for Addressee Recognition in Multi-modal Multi-party Dialogue 

**Title (ZH)**: 多模态多人群体对话中收件人识别的大型语言模型基准测试 

**Authors**: Koji Inoue, Divesh Lala, Mikey Elmers, Keiko Ochi, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2501.16643)  

**Abstract**: Handling multi-party dialogues represents a significant step for advancing spoken dialogue systems, necessitating the development of tasks specific to multi-party interactions. To address this challenge, we are constructing a multi-modal multi-party dialogue corpus of triadic (three-participant) discussions. This paper focuses on the task of addressee recognition, identifying who is being addressed to take the next turn, a critical component unique to multi-party dialogue systems. A subset of the corpus was annotated with addressee information, revealing that explicit addressees are indicated in approximately 20% of conversational turns. To evaluate the task's complexity, we benchmarked the performance of a large language model (GPT-4o) on addressee recognition. The results showed that GPT-4o achieved an accuracy only marginally above chance, underscoring the challenges of addressee recognition in multi-party dialogue. These findings highlight the need for further research to enhance the capabilities of large language models in understanding and navigating the intricacies of multi-party conversational dynamics. 

**Abstract (ZH)**: 处理多主体对话是推动口语对话系统发展的关键步骤，需要开发针对多主体交互的任务。为应对这一挑战，我们正在构建一个包含三元讨论的多模态多主体对话语料库。本文重点讨论了与会者识别的任务，即识别谁是下一个发言的对象，这是多主体对话系统特有的关键组成部分。部分语料库被标注了与会者信息，结果显示，在约20%的对话回合中明确指出了与会者。为了评估该任务的难度，我们在与会者识别任务上对大型语言模型（GPT-4o）进行了基准测试。结果表明，GPT-4o 的识别准确率仅略微高于随机猜测，强调了多主体对话中与会者识别的挑战性。这些发现突显了进一步研究以提升大型语言模型理解与导航多主体对话复杂动态的能力的必要性。 

---
# Chinese Stock Prediction Based on a Multi-Modal Transformer Framework: Macro-Micro Information Fusion 

**Title (ZH)**: 基于多模态变压器框架的中国股市预测：宏观与微观信息融合 

**Authors**: Lumen AI, Tengzhou No. 1 Middle School, Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Xu Tianhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.16621)  

**Abstract**: This paper proposes an innovative Multi-Modal Transformer framework (MMF-Trans) designed to significantly improve the prediction accuracy of the Chinese stock market by integrating multi-source heterogeneous information including macroeconomy, micro-market, financial text, and event knowledge. The framework consists of four core modules: (1) A four-channel parallel encoder that processes technical indicators, financial text, macro data, and event knowledge graph respectively for independent feature extraction of multi-modal data; (2) A dynamic gated cross-modal fusion mechanism that adaptively learns the importance of different modalities through differentiable weight allocation for effective information integration; (3) A time-aligned mixed-frequency processing layer that uses an innovative position encoding method to effectively fuse data of different time frequencies and solves the time alignment problem of heterogeneous data; (4) A graph attention-based event impact quantification module that captures the dynamic impact of events on the market through event knowledge graph and quantifies the event impact coefficient. We introduce a hybrid-frequency Transformer and Event2Vec algorithm to effectively fuse data of different frequencies and quantify the event impact. Experimental results show that in the prediction task of CSI 300 constituent stocks, the root mean square error (RMSE) of the MMF-Trans framework is reduced by 23.7% compared to the baseline model, the event response prediction accuracy is improved by 41.2%, and the Sharpe ratio is improved by 32.6%. 

**Abstract (ZH)**: 本文提出了一种创新的多模态Transformer框架（MMF-Trans），旨在通过整合宏观经济、微观市场、金融文本和事件知识等多种来源的异质信息来显著提高中国股票市场的预测准确性。该框架包含四个核心模块：（1）一个四通道并行编码器，分别处理技术指标、金融文本、宏观经济数据和事件知识图谱，独立提取多模态数据的特征；（2）一个动态门控跨模态融合机制，通过可微配权学习不同模态的重要性，实现有效信息整合；（3）一个时间对齐混合频次处理层，采用创新的位置编码方法有效融合不同时间频率的数据并解决异构数据的时间对齐问题；（4）一个基于图注意力的事件影响量化模块，通过事件知识图谱捕捉事件对市场的动态影响并量化事件影响系数。我们引入了混合频次Transformer和Event2Vec算法，有效融合不同频次的数据并量化事件影响。实验结果显示，在CSI 300成分股的预测任务中，MMF-Trans框架的均方根误差（RMSE）相较于基础模型降低了23.7%，事件响应预测准确性提高了41.2%，夏普比率提高了32.6%。 

---
# PackDiT: Joint Human Motion and Text Generation via Mutual Prompting 

**Title (ZH)**: PackDiT：通过相互提示联合生成人体动作和文本 

**Authors**: Zhongyu Jiang, Wenhao Chai, Zhuoran Zhou, Cheng-Yen Yang, Hsiang-Wei Huang, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16551)  

**Abstract**: Human motion generation has advanced markedly with the advent of diffusion models. Most recent studies have concentrated on generating motion sequences based on text prompts, commonly referred to as text-to-motion generation. However, the bidirectional generation of motion and text, enabling tasks such as motion-to-text alongside text-to-motion, has been largely unexplored. This capability is essential for aligning diverse modalities and supports unconditional generation. In this paper, we introduce PackDiT, the first diffusion-based generative model capable of performing various tasks simultaneously, including motion generation, motion prediction, text generation, text-to-motion, motion-to-text, and joint motion-text generation. Our core innovation leverages mutual blocks to integrate multiple diffusion transformers (DiTs) across different modalities seamlessly. We train PackDiT on the HumanML3D dataset, achieving state-of-the-art text-to-motion performance with an FID score of 0.106, along with superior results in motion prediction and in-between tasks. Our experiments further demonstrate that diffusion models are effective for motion-to-text generation, achieving performance comparable to that of autoregressive models. 

**Abstract (ZH)**: 自扩散模型的出现极大地推动了人体运动生成的发展。最近的研究主要集中在基于文本提示生成运动序列，通常称为文本到运动生成。然而，双向生成运动和文本的任务，比如运动到文本和文本到运动的相互转换，迄今尚未得到充分探索。这种能力对于对齐多种模态数据和实现无条件生成至关重要。本文中，我们介绍了PackDiT，这是第一个能够同时执行多种任务的基于扩散模型的生成模型，包括运动生成、运动预测、文本生成、文本到运动、运动到文本以及联合运动-文本生成。我们的主要创新在于引入了互换块，以无缝地集成不同模态的多个扩散变换器（DiTs）。在HumanML3D数据集上训练PackDiT，我们实现了最先进的文本到运动性能，FID得分为0.106，并且在运动预测和中间任务上也取得了优异结果。我们的实验进一步表明，扩散模型在运动到文本生成任务上也是有效的，其性能可与自回归模型相媲美。 

---
# SIM: Surface-based fMRI Analysis for Inter-Subject Multimodal Decoding from Movie-Watching Experiments 

**Title (ZH)**: SIM：基于表面的功能磁共振成像分析在电影观看实验中实现跨被试多模态解码 

**Authors**: Simon Dahan, Gabriel Bénédict, Logan Z. J. Williams, Yourong Guo, Daniel Rueckert, Robert Leech, Emma C. Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2501.16471)  

**Abstract**: Current AI frameworks for brain decoding and encoding, typically train and test models within the same datasets. This limits their utility for brain computer interfaces (BCI) or neurofeedback, for which it would be useful to pool experiences across individuals to better simulate stimuli not sampled during training. A key obstacle to model generalisation is the degree of variability of inter-subject cortical organisation, which makes it difficult to align or compare cortical signals across participants. In this paper we address this through the use of surface vision transformers, which build a generalisable model of cortical functional dynamics, through encoding the topography of cortical networks and their interactions as a moving image across a surface. This is then combined with tri-modal self-supervised contrastive (CLIP) alignment of audio, video, and fMRI modalities to enable the retrieval of visual and auditory stimuli from patterns of cortical activity (and vice-versa). We validate our approach on 7T task-fMRI data from 174 healthy participants engaged in the movie-watching experiment from the Human Connectome Project (HCP). Results show that it is possible to detect which movie clips an individual is watching purely from their brain activity, even for individuals and movies not seen during training. Further analysis of attention maps reveals that our model captures individual patterns of brain activity that reflect semantic and visual systems. This opens the door to future personalised simulations of brain function. Code & pre-trained models will be made available at this https URL, processed data for training will be available upon request at this https URL. 

**Abstract (ZH)**: 当前的脑解码和编码AI框架通常在相同的数据集中训练和测试模型，这限制了它们在脑机接口(BCI)或神经反馈中的应用，因为在这种应用中，跨个体汇总经验可以更好地模拟训练过程中未采样的刺激。模型泛化的一个主要障碍是跨个体皮层组织的变异程度，这使得难以对参与者之间的皮层信号进行对齐或比较。在这篇论文中，我们通过使用表面视网膜变换器来解决这一问题，它通过将皮层网络及其交互的拓扑结构编码为表面上的移动图像，构建了一个可泛化的皮层功能动态模型。然后，结合针对音频、视频和fMRI模态的三模态自监督对比(CLIP)对齐，使我们能够从皮层活动模式中检索视觉和听觉刺激（反之亦然）。我们在人类连通体项目(HCP)的174名健康参与者观看电影实验的7T任务fMRI数据上验证了这种方法。结果显示，仅从个体的脑活动就可以检测出他们在观看的电影剪辑，即使这些电影剪辑和个体在训练过程中未见过也是如此。进一步分析注意力图显示，我们的模型捕捉到了反映了语义和视觉系统的个体脑活动模式。这一发现为未来个性化模拟脑功能打开了新的大门。相关代码及预训练模型将在此 https://链接 发布，训练所需的处理数据将在收到请求后提供此 https://链接。 

---
# PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding 

**Title (ZH)**: PhysBench：视觉语言模型在物理世界理解中的基准测试与增强 

**Authors**: Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16411)  

**Abstract**: Understanding the physical world is a fundamental challenge in embodied AI, critical for enabling agents to perform complex tasks and operate safely in real-world environments. While Vision-Language Models (VLMs) have shown great promise in reasoning and task planning for embodied agents, their ability to comprehend physical phenomena remains extremely limited. To close this gap, we introduce PhysBench, a comprehensive benchmark designed to evaluate VLMs' physical world understanding capability across a diverse set of tasks. PhysBench contains 100,000 entries of interleaved video-image-text data, categorized into four major domains: physical object properties, physical object relationships, physical scene understanding, and physics-based dynamics, further divided into 19 subclasses and 8 distinct capability dimensions. Our extensive experiments, conducted on 75 representative VLMs, reveal that while these models excel in common-sense reasoning, they struggle with understanding the physical world -- likely due to the absence of physical knowledge in their training data and the lack of embedded physical priors. To tackle the shortfall, we introduce PhysAgent, a novel framework that combines the generalization strengths of VLMs with the specialized expertise of vision models, significantly enhancing VLMs' physical understanding across a variety of tasks, including an 18.4\% improvement on GPT-4o. Furthermore, our results demonstrate that enhancing VLMs' physical world understanding capabilities can help embodied agents such as MOKA. We believe that PhysBench and PhysAgent offer valuable insights and contribute to bridging the gap between VLMs and physical world understanding. 

**Abstract (ZH)**: 了解物理世界是实现具身人工智能的基本挑战，对于使智能体能够执行复杂任务并在真实环境中共存至关重要。虽然视觉-语言模型（VLMs）在具身智能体的推理和任务规划方面展现了巨大的潜力，但在理解物理现象方面的能力仍然极其有限。为了弥合这一差距，我们引入了PhysBench，这是一个全面的基准测试，旨在评估VLMs在一系列多样化任务中的物理世界理解能力。PhysBench包含100,000个交错的视频-图像-文本数据条目，并按照四大主要领域分类：物体物理属性、物体物理关系、场景理解以及基于物理的动态。进一步细分为19个子类别和8个不同的能力维度。我们在75个代表性VLMs上进行了广泛的实验，结果显示，在常识推理方面这些模型表现出色，但在理解物理世界方面却面临困难——这可能是因为它们的训练数据中缺乏物理知识，以及嵌入物理先验信息的缺失。为解决这一不足，我们引入了PhysAgent，这是一个新颖的框架，它将VLMs的一般泛化优势与视觉模型的专业知识相结合，显著增强了VLMs在各种任务中对物理理解的能力，例如在GPT-4o上取得了18.4%的提升。此外，我们的结果显示，增强VLMs的物理世界理解能力可以帮助具身智能体如MOKA。我们相信，PhysBench和PhysAgent提供了宝贵的见解，并有助于弥合VLMs与物理世界理解之间的差距。 

---
# Internal Activation Revision: Safeguarding Vision Language Models Without Parameter Update 

**Title (ZH)**: 内部激活修正：无需参数更新的视觉语言模型保护 

**Authors**: Qing Li, Jiahui Geng, Zongxiong Chen, Kun Song, Lei Ma, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2501.16378)  

**Abstract**: Vision-language models (VLMs) demonstrate strong multimodal capabilities but have been found to be more susceptible to generating harmful content compared to their backbone large language models (LLMs). Our investigation reveals that the integration of images significantly shifts the model's internal activations during the forward pass, diverging from those triggered by textual input. Moreover, the safety alignments of LLMs embedded within VLMs are not sufficiently robust to handle the activations discrepancies, making the models vulnerable to even the simplest jailbreaking attacks. To address this issue, we propose an \textbf{internal activation revision} approach that efficiently revises activations during generation, steering the model toward safer outputs. Our framework incorporates revisions at both the layer and head levels, offering control over the model's generation at varying levels of granularity. In addition, we explore three strategies for constructing positive and negative samples and two approaches for extracting revision vectors, resulting in different variants of our method. Comprehensive experiments demonstrate that the internal activation revision method significantly improves the safety of widely used VLMs, reducing attack success rates by an average of 48.94\%, 34.34\%, 43.92\%, and 52.98\% on SafeBench, Safe-Unsafe, Unsafe, and MM-SafetyBench, respectively, while minimally impacting model helpfulness. 

**Abstract (ZH)**: 视觉语言模型（VLMs）展现了强大的多模态能力，但在生成有害内容方面比其骨干大型语言模型（LLMs）更为脆弱。我们的研究发现，图像的整合在前向传递过程中显著改变了模型的内部激活，与文本输入触发的激活有所不同。此外，嵌入在VLM中的LLMs的安全对齐不够稳固，无法应对激活差异，使模型容易受到最简单的“监狱突破”攻击。为解决这一问题，我们提出了一种**内部激活修订**的方法，在生成过程中高效地修订激活，引导模型产生更安全的输出。我们的框架在层和头两个级别上都进行了修订，提供了不同程度的模型生成控制。此外，我们探讨了三种构建正负样本的策略和两种提取修订向量的方法，从而形成了我们方法的不同变体。全面的实验表明，内部激活修订方法显著提高了广泛使用的VLMs的安全性，在SafeBench、Safe-Unsafe、Unsafe和MM-SafetyBench上分别将攻击成功率降低了48.94%、34.34%、43.92%和52.98%，同时对模型的帮助性影响最小。 

---
# Foundation Models for CPS-IoT: Opportunities and Challenges 

**Title (ZH)**: 面向CPS-IoT的基石模型：机遇与挑战 

**Authors**: Ozan Baris, Yizhuo Chen, Gaofeng Dong, Liying Han, Tomoyoshi Kimura, Pengrui Quan, Ruijie Wang, Tianchen Wang, Tarek Abdelzaher, Mario Bergés, Paul Pu Liang, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2501.16368)  

**Abstract**: Methods from machine learning (ML) have transformed the implementation of Perception-Cognition-Communication-Action loops in Cyber-Physical Systems (CPS) and the Internet of Things (IoT), replacing mechanistic and basic statistical models with those derived from data. However, the first generation of ML approaches, which depend on supervised learning with annotated data to create task-specific models, faces significant limitations in scaling to the diverse sensor modalities, deployment configurations, application tasks, and operating dynamics characterizing real-world CPS-IoT systems. The success of task-agnostic foundation models (FMs), including multimodal large language models (LLMs), in addressing similar challenges across natural language, computer vision, and human speech has generated considerable enthusiasm for and exploration of FMs and LLMs as flexible building blocks in CPS-IoT analytics pipelines, promising to reduce the need for costly task-specific engineering.
Nonetheless, a significant gap persists between the current capabilities of FMs and LLMs in the CPS-IoT domain and the requirements they must meet to be viable for CPS-IoT applications. In this paper, we analyze and characterize this gap through a thorough examination of the state of the art and our research, which extends beyond it in various dimensions. Based on the results of our analysis and research, we identify essential desiderata that CPS-IoT domain-specific FMs and LLMs must satisfy to bridge this gap. We also propose actions by CPS-IoT researchers to collaborate in developing key community resources necessary for establishing FMs and LLMs as foundational tools for the next generation of CPS-IoT systems. 

**Abstract (ZH)**: 机器学习（ML）方法已经改变了在网络物理系统（CPS）和物联网（IoT）中实施感知-认知-通信-行动循环的方式，用从数据中导出的模型取代了机械性和基础统计模型。然而，依赖于带有标注数据的监督学习的第一代ML方法在扩展到实际CPS-IoT系统中多样化的传感器模态、部署配置、应用任务和运行动态方面面临重大局限性。无任务特定的基础模型（FMs），包括多模态大规模语言模型（LLMs），在解决自然语言、计算机视觉和人类语言领域相似挑战方面取得了显著成功，这激发了对FMs和LLMs作为CPS-IoT分析管道的灵活构建模块的探索，有望减少针对特定任务的工程成本。

尽管如此，当前FMs和LLMs在CPS-IoT领域的功能与它们需要满足以适用于CPS-IoT应用的要求之间仍存在显著差距。在这篇论文中，我们通过全面分析现有技术和我们的研究成果来分析并表征这种差距，并在多个维度上超越了现有的研究。基于分析和研究结果，我们确定了CPS-IoT领域特定的FMs和LLMs必须满足的关键需求，以缩小这一差距。我们还提议CPS-IoT领域的研究人员合作开发关键社区资源，以建立FMs和LLMs作为新一代CPS-IoT系统的基础工具。 

---
# Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks 

**Title (ZH)**: 文档截图检索器易受像素中毒攻击的影响 

**Authors**: Shengyao Zhuang, Ekaterina Khramtsova, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2501.16902)  

**Abstract**: Recent advancements in dense retrieval have introduced vision-language model (VLM)-based retrievers, such as DSE and ColPali, which leverage document screenshots embedded as vectors to enable effective search and offer a simplified pipeline over traditional text-only methods. In this study, we propose three pixel poisoning attack methods designed to compromise VLM-based retrievers and evaluate their effectiveness under various attack settings and parameter configurations. Our empirical results demonstrate that injecting even a single adversarial screenshot into the retrieval corpus can significantly disrupt search results, poisoning the top-10 retrieved documents for 41.9% of queries in the case of DSE and 26.4% for ColPali. These vulnerability rates notably exceed those observed with equivalent attacks on text-only retrievers. Moreover, when targeting a small set of known queries, the attack success rate raises, achieving complete success in certain cases. By exposing the vulnerabilities inherent in vision-language models, this work highlights the potential risks associated with their deployment. 

**Abstract (ZH)**: 近年来，密集检索领域的最新进展引入了基于视觉-语言模型（VLM）的检索器，如DSE和ColPali，这些模型通过将文档截图嵌入向量中，实现了有效的搜索，并提供了一种比传统纯文本方法更简化的流水线。在本研究中，我们提出了三种像素篡改攻击方法，旨在破坏基于VLM的检索器，并在不同的攻击设置和参数配置下评估其有效性。我们的实验结果表明，即使向检索语料库中注入单个 adversarial screenshot 也能显著破坏搜索结果，在DSE中，有41.9%的查询导致检索到的前10篇文档被污染，在ColPali中，这一比例为26.4%。这些漏洞率明显高于对纯文本检索器进行等效攻击所观察到的漏洞率。此外，当针对一组已知查询时，攻击成功率提高，某些情况下可以完全成功。通过揭示视觉-语言模型固有的漏洞，本研究强调了它们部署时可能面临的风险。 

---
# Whispers of Sound-Enhancing Information Extraction from Depression Patients' Unstructured Data through Audio and Text Emotion Recognition and Llama Fine-tuning 

**Title (ZH)**: 通过音频和文本情感识别以及 llama 微调从抑郁症患者未结构化数据中增强声学信息的提取 metodologies: 从抑郁症患者未结构化数据中提取增强声学信息的方法——借助音频和文本情感识别及 llama 微调 

**Authors**: Lindy Gan, Yifan Huang, Xiaoyang Gao, Jiaming Tan, Fujun Zhao, Tao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16813)  

**Abstract**: This study proposes an innovative multimodal fusion model based on a teacher-student architecture to enhance the accuracy of depression classification. Our designed model addresses the limitations of traditional methods in feature fusion and modality weight allocation by introducing multi-head attention mechanisms and weighted multimodal transfer learning. Leveraging the DAIC-WOZ dataset, the student fusion model, guided by textual and auditory teacher models, achieves significant improvements in classification accuracy. Ablation experiments demonstrate that the proposed model attains an F1 score of 99. 1% on the test set, significantly outperforming unimodal and conventional approaches. Our method effectively captures the complementarity between textual and audio features while dynamically adjusting the contributions of the teacher models to enhance generalization capabilities. The experimental results highlight the robustness and adaptability of the proposed framework in handling complex multimodal data. This research provides a novel technical framework for multimodal large model learning in depression analysis, offering new insights into addressing the limitations of existing methods in modality fusion and feature extraction. 

**Abstract (ZH)**: 本研究提出了一种基于教师-学生架构的创新多模态融合模型，旨在提高抑郁症分类的准确性。我们设计的模型通过引入多头注意力机制和加权多模态迁移学习，解决了传统方法在特征融合和模态权重分配方面的局限性。利用DAIC-WOZ数据集，受文本和听觉教师模型的指导，学生融合模型在分类准确性上取得了显著提升。消融实验的结果表明，该模型在测试集上的F1分数达到99.1%，显著优于单模态和传统方法。我们的方法有效捕捉了文本和音频特征之间的互补性，同时动态调整教师模型的贡献，以增强泛化能力。实验结果突显了所提出框架在处理复杂多模态数据方面的鲁棒性和适应性。该研究为抑郁症分析中的多模态大型模型学习提供了新的技术框架，为解决现有方法在模态融合和特征提取方面的局限性提供了新的见解。 

---
# 3D-MoE: A Mixture-of-Experts Multi-modal LLM for 3D Vision and Pose Diffusion via Rectified Flow 

**Title (ZH)**: 3D-MoE：一种用于3D视觉和姿态扩散的混合专家多模态大语言模型通过修正流 

**Authors**: Yueen Ma, Yuzheng Zhuang, Jianye Hao, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2501.16698)  

**Abstract**: 3D vision and spatial reasoning have long been recognized as preferable for accurately perceiving our three-dimensional world, especially when compared with traditional visual reasoning based on 2D images. Due to the difficulties in collecting high-quality 3D data, research in this area has only recently gained momentum. With the advent of powerful large language models (LLMs), multi-modal LLMs for 3D vision have been developed over the past few years. However, most of these models focus primarily on the vision encoder for 3D data. In this paper, we propose converting existing densely activated LLMs into mixture-of-experts (MoE) models, which have proven effective for multi-modal data processing. In addition to leveraging these models' instruction-following capabilities, we further enable embodied task planning by attaching a diffusion head, Pose-DiT, that employs a novel rectified flow diffusion scheduler. Experimental results on 3D question answering and task-planning tasks demonstrate that our 3D-MoE framework achieves improved performance with fewer activated parameters. 

**Abstract (ZH)**: 三维视觉和空间推理长期以来被认为在准确感知三维世界方面更优越，特别是在与基于二维图像的传统视觉推理方法相比时。由于高质量三维数据采集的困难，这一领域的研究直到最近才取得进展。随着强大大规模语言模型（LLMs）的出现，近年来开发出了针对三维视觉的多模态LLMs。然而，大多数模型主要集中在三维数据的视觉编码器上。在本文中，我们提出将现有的密集激活LLMs转换为混合专家（MoE）模型，这些模型已被证明对多模态数据处理有效。除了利用这些模型的指令执行能力外，我们还通过附加一个扩散头Pose-DiT，进一步实现了具身任务规划，Pose-DiT 利用了一种新型的整流流动扩散调度器。在三维问答和任务规划任务上的实验结果表明，我们的3D-MoE框架能够在激活参数更少的情况下实现更好的性能。 

---
# MME-Industry: A Cross-Industry Multimodal Evaluation Benchmark 

**Title (ZH)**: MME-行业：跨行业多模态评估基准 

**Authors**: Dongyi Yi, Guibo Zhu, Chenglin Ding, Zongshu Li, Dong Yi, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16688)  

**Abstract**: With the rapid advancement of Multimodal Large Language Models (MLLMs), numerous evaluation benchmarks have emerged. However, comprehensive assessments of their performance across diverse industrial applications remain limited. In this paper, we introduce MME-Industry, a novel benchmark designed specifically for evaluating MLLMs in industrial this http URL benchmark encompasses 21 distinct domain, comprising 1050 question-answer pairs with 50 questions per domain. To ensure data integrity and prevent potential leakage from public datasets, all question-answer pairs were manually crafted and validated by domain experts. Besides, the benchmark's complexity is effectively enhanced by incorporating non-OCR questions that can be answered directly, along with tasks requiring specialized domain knowledge. Moreover, we provide both Chinese and English versions of the benchmark, enabling comparative analysis of MLLMs' capabilities across these languages. Our findings contribute valuable insights into MLLMs' practical industrial applications and illuminate promising directions for future model optimization research. 

**Abstract (ZH)**: 随着多模态大型语言模型（MLLMs）的迅速发展，各种评估基准相继出现。然而，这些模型在不同工业应用中的全面评估仍然有限。本文介绍了一种新型基准——MME-Industry，专门用于评估工业应用场景中的MLLMs。该基准涵盖了21个不同的领域，共有1050个问题-答案对，每个领域包含50个问题。为确保数据完整性和防止潜在泄露，所有问题-答案对均由领域专家手工设计和验证。此外，通过引入不需要OCR即可直接回答的问题和需要特定领域知识的任务，进一步增强了基准的复杂性。此外，我们提供了该基准的中英文两个版本，便于分析MLLMs跨语言的能力。我们的研究结果为MLLMs的实际工业应用提供了宝贵的见解，并指明了未来模型优化研究的潜在方向。 

---
# Contextual Reinforcement in Multimodal Token Compression for Large Language Models 

**Title (ZH)**: 多模态词token压缩中的上下文强化方法 

**Authors**: Naderdel Piero, Zacharias Cromwell, Nathaniel Wainwright, Matthias Nethercott  

**Link**: [PDF](https://arxiv.org/pdf/2501.16658)  

**Abstract**: Effective token compression remains a critical challenge for scaling models to handle increasingly complex and diverse datasets. A novel mechanism based on contextual reinforcement is introduced, dynamically adjusting token importance through interdependencies and semantic relevance. This approach enables substantial reductions in token usage while preserving the quality and coherence of information representation. Incorporating graph-based algorithms and adaptive weighting, the method captures subtle contextual relationships across textual and multimodal data, ensuring robust alignment and performance in downstream tasks. Evaluations across varied domains reveal significant improvements in accuracy and semantic retention, particularly for tasks requiring detailed cross-modal interactions. Memory usage analyses demonstrate improved computational efficiency, with minimal overhead despite the additional reinforcement processes. Performance gains are further validated through error distribution analyses, showing reduced semantic loss and syntactic inconsistencies compared to baseline models. The modular architecture ensures compatibility with a wide range of open-source frameworks, facilitating scalable implementation for real-world applications. These findings highlight the potential of contextual reinforcement in redefining token management strategies and advancing large-scale model design. 

**Abstract (ZH)**: 有效_token压缩仍然是将模型扩展以处理日益复杂和多样化的数据集的关键挑战。提出了一种基于上下文强化的新机制，通过相互依赖性和语义相关性动态调整token的重要性。这种方法在减少token使用量的同时，能够保持信息表示的质量和连贯性。该方法结合图算法和自适应加权，捕获文本和多模态数据中的细微上下文关系，确保下游任务中的鲁棒对齐和性能。在不同领域的评估表明，在需要详细跨模态交互的任务中，该方法显著提高了准确性和语义保留。内存使用分析显示，尽管存在额外的强化过程，但计算效率有所改进，且额外开销较小。通过错误分布分析进一步验证了性能增益，与基线模型相比，显示了减少语义损失和句法不一致。模块化架构确保了与广泛使用的开源框架的兼容性，促进了实时应用中的可扩展实现。这些发现突显了上下文强化在重新定义token管理策略并推进大规模模型设计方面的潜力。 

---
# CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs 

**Title (ZH)**: CHiP：跨模态层次直接偏好优化在多模态LLM中的应用 

**Authors**: Jinlan Fu, Shenzhen Huangfu, Hao Fei, Xiaoyu Shen, Bryan Hooi, Xipeng Qiu, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2501.16629)  

**Abstract**: Multimodal Large Language Models (MLLMs) still struggle with hallucinations despite their impressive capabilities. Recent studies have attempted to mitigate this by applying Direct Preference Optimization (DPO) to multimodal scenarios using preference pairs from text-based responses. However, our analysis of representation distributions reveals that multimodal DPO struggles to align image and text representations and to distinguish between hallucinated and non-hallucinated descriptions. To address these challenges, in this work, we propose a Cross-modal Hierarchical Direct Preference Optimization (CHiP) to address these limitations. We introduce a visual preference optimization module within the DPO framework, enabling MLLMs to learn from both textual and visual preferences simultaneously. Furthermore, we propose a hierarchical textual preference optimization module that allows the model to capture preferences at multiple granular levels, including response, segment, and token levels. We evaluate CHiP through both quantitative and qualitative analyses, with results across multiple benchmarks demonstrating its effectiveness in reducing hallucinations. On the Object HalBench dataset, CHiP outperforms DPO in hallucination reduction, achieving improvements of 52.7% and 55.5% relative points based on the base model Muffin and LLaVA models, respectively. We make all our datasets and code publicly available: this https URL. 

**Abstract (ZH)**: 尽管多模态大语言模型（MLLMs）具有强大的能力，但在处理妄想方面仍然存在困难。最近的研究试图通过使用基于文本的响应中的偏好对来在多模态场景中应用直接偏好优化（DPO）来缓解这一问题。然而，我们的分析表明，多模态DPO难以对齐图像和文本表示，并区分妄想描述和非妄想描述。为了解决这些挑战，在本文中，我们提出了一种跨模态分层直接偏好优化（CHiP）来解决这些限制。我们引入了一个视觉偏好优化模块，使其能够在DPO框架中同时从文本和视觉偏好中学习。此外，我们还提出了一个分层的文本偏好优化模块，允许模型在响应、段落和令牌等多个粒度级别上捕捉偏好。我们通过定量和定性分析对CHiP进行了评估，结果表明其在减少妄想方面非常有效。在Object HalBench数据集中，CHiP在妄想减少方面的表现优于DPO，基于基准模型Muffin和LLaVA模型，分别提高了52.7%和55.5%的相对点数。我们已将所有数据集和代码公开发布：this https URL。 

---
# Exploring the Role of Explicit Temporal Modeling in Multimodal Large Language Models for Video Understanding 

**Title (ZH)**: 探索显式时间建模在多模态大型语言模型中对视频理解的作用 

**Authors**: Yun Li, Zhe Liu, Yajing Kong, Guangrui Li, Jiyuan Zhang, Chao Bian, Feng Liu, Lina Yao, Zhenbang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.16786)  

**Abstract**: Applying Multimodal Large Language Models (MLLMs) to video understanding presents significant challenges due to the need to model temporal relations across frames. Existing approaches adopt either implicit temporal modeling, relying solely on the LLM decoder, or explicit temporal modeling, employing auxiliary temporal encoders. To investigate this debate between the two paradigms, we propose the Stackable Temporal Encoder (STE). STE enables flexible explicit temporal modeling with adjustable temporal receptive fields and token compression ratios. Using STE, we systematically compare implicit and explicit temporal modeling across dimensions such as overall performance, token compression effectiveness, and temporal-specific understanding. We also explore STE's design considerations and broader impacts as a plug-in module and in image modalities. Our findings emphasize the critical role of explicit temporal modeling, providing actionable insights to advance video MLLMs. 

**Abstract (ZH)**: 将多模态大规模语言模型（MLLMs）应用于视频理解带来了显著的挑战，因为需要建模跨帧的时间关系。现有的方法要么采用隐式时间建模，依赖于LLM解码器，要么采用显式时间建模，使用辅助的时间编码器。为了研究这两种范式的优劣，我们提出了可堆叠的时间编码器（Stackable Temporal Encoder, STE）。STE能够实现灵活的显式时间建模，并具有可调的时间感受野和标记压缩比。通过STE，我们系统地比较了隐式和显式时间建模在整体性能、标记压缩效果以及时间特定理解等方面的差异。我们还探讨了STE的设计考虑及其作为插件模块在图像模态中的更广泛影响。我们的研究强调了显式时间建模的关键作用，并提供了推动视频MLLMs发展的实际建议。 

---
# Developing Enhanced Conversational Agents for Social Virtual Worlds 

**Title (ZH)**: 开发增强型对话代理以应用于社会虚拟世界 

**Authors**: D. Griol, A. Sanchis, J. M. Molina, Z. Callejas  

**Link**: [PDF](https://arxiv.org/pdf/2501.16341)  

**Abstract**: In this paper, we present a methodology for the development of embodied conversational agents for social virtual worlds. The agents provide multimodal communication with their users in which speech interaction is included. Our proposal combines different techniques related to Artificial Intelligence, Natural Language Processing, Affective Computing, and User Modeling. Firstly, the developed conversational agents. A statistical methodology has been developed to model the system conversational behavior, which is learned from an initial corpus and improved with the knowledge acquired from the successive interactions. In addition, the selection of the next system response is adapted considering information stored into users profiles and also the emotional contents detected in the users utterances. Our proposal has been evaluated with the successful development of an embodied conversational agent which has been placed in the Second Life social virtual world. The avatar includes the different models and interacts with the users who inhabit the virtual world in order to provide academic information. The experimental results show that the agents conversational behavior adapts successfully to the specific characteristics of users interacting in such environments. 

**Abstract (ZH)**: 在本文中，我们提出了一种用于社会虚拟世界中实现具身对话代理的方法论。这些代理与用户进行多模态通信，其中包括语音交互。我们的提案结合了人工智能、自然语言处理、情感计算和用户建模等不同技术。首先，我们开发了具身对话代理。为模拟系统的对话行为，我们开发了一种统计方法论，该方法论从初始语料库中学习，并通过后续交互中获得的知识不断改进。此外，系统响应的选择根据用户档案中存储的信息以及检测到的用户话语中的情感内容进行相应调整。我们的提案已经在《第二生命》（Second Life）社会虚拟世界中成功实现了具身对话代理。该代理的模拟人包括不同的模型，并与居住在虚拟世界中的用户进行互动，以提供学术信息。实验结果表明，代理的对话行为能够成功适应此类环境中交互用户的特定特征。 

---
