# Metric Learning with Progressive Self-Distillation for Audio-Visual Embedding Learning 

**Title (ZH)**: 面向音频-视觉嵌入学习的渐进自蒸馏度量学习 

**Authors**: Donghuo Zeng, Kazushi Ikeda  

**Link**: [PDF](https://arxiv.org/pdf/2501.09608)  

**Abstract**: Metric learning projects samples into an embedded space, where similarities and dissimilarities are quantified based on their learned representations. However, existing methods often rely on label-guided representation learning, where representations of different modalities, such as audio and visual data, are aligned based on annotated labels. This approach tends to underutilize latent complex features and potential relationships inherent in the distributions of audio and visual data that are not directly tied to the labels, resulting in suboptimal performance in audio-visual embedding learning. To address this issue, we propose a novel architecture that integrates cross-modal triplet loss with progressive self-distillation. Our method enhances representation learning by leveraging inherent distributions and dynamically refining soft audio-visual alignments -- probabilistic alignments between audio and visual data that capture the inherent relationships beyond explicit labels. Specifically, the model distills audio-visual distribution-based knowledge from annotated labels in a subset of each batch. This self-distilled knowledge is used t 

**Abstract (ZH)**: 度量学习将样本映射到一个嵌入空间，在该空间中，样本的相似性和差异性基于其学习到的表示进行量化。然而，现有的方法往往依赖于标签引导的表示学习，不同模态（如音频和视觉数据）的表示根据标注标签进行对齐。这种方法往往会忽略音频和视觉数据内在分布中那些未直接与标签相关联的潜在复杂特征和关系，从而在音频-视觉嵌入学习中导致性能不佳。为了解决这一问题，我们提出了一种新的架构，它将跨模态三元组损失与逐步自我蒸馏相结合。我们的方法通过利用内在分布并动态精炼软音频-视觉对齐来增强表示学习——这些对齐捕捉了超越显性标签的内在关系。具体而言，模型从每个批次的子集中的标注标签中提取基于分布的知识进行自我蒸馏。这些自我提取的知识被用来改进表示学习，从而增强音频-视觉对齐，并更好地捕捉模态之间的内在联系。 

---
# Augmenting a Large Language Model with a Combination of Text and Visual Data for Conversational Visualization of Global Geospatial Data 

**Title (ZH)**: 将一大型语言模型与文本和视觉数据相结合，以实现全球地理空间数据的对话可视化 

**Authors**: Omar Mena, Alexandre Kouyoumdjian, Lonni Besançon, Michael Gleicher, Ivan Viola, Anders Ynnerman  

**Link**: [PDF](https://arxiv.org/pdf/2501.09521)  

**Abstract**: We present a method for augmenting a Large Language Model (LLM) with a combination of text and visual data to enable accurate question answering in visualization of scientific data, making conversational visualization possible. LLMs struggle with tasks like visual data interaction, as they lack contextual visual information. We address this problem by merging a text description of a visualization and dataset with snapshots of the visualization. We extract their essential features into a structured text file, highly compact, yet descriptive enough to appropriately augment the LLM with contextual information, without any fine-tuning. This approach can be applied to any visualization that is already finally rendered, as long as it is associated with some textual description. 

**Abstract (ZH)**: 我们提出了一种方法，将大型语言模型（LLM）与文本和视觉数据结合起来，以实现科学数据可视化中的准确问题回答，从而使对话式可视化成为可能。LLM 在处理如视觉数据交互等任务时存在困难，因为它们缺乏上下文视觉信息。我们通过将可视化和数据集的文本描述与可视化快照相结合来解决这一问题。我们将这些信息的关键特征提取到一个结构化的文本文件中，该文件虽高度紧凑，但描述足够详细，能够为LLM提供必要的上下文信息，而无需进行任何微调。这种方法可以应用于任何已最终渲染的可视化，只要它与某些文本描述相关联即可。 

---
# Vision-Language Models Do Not Understand Negation 

**Title (ZH)**: 视觉-语言模型不懂得否定意义 

**Authors**: Kumail Alhamoud, Shaden Alshammari, Yonglong Tian, Guohao Li, Philip Torr, Yoon Kim, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09425)  

**Abstract**: Many practical vision-language applications require models that understand negation, e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce NegBench, a new benchmark designed to evaluate negation understanding across 18 task variations and 79k examples spanning image, video, and medical datasets. The benchmark consists of two core tasks designed to evaluate negation understanding in diverse multimodal settings: Retrieval with Negation and Multiple Choice Questions with Negated Captions. Our evaluation reveals that modern VLMs struggle significantly with negation, often performing at chance level. To address these shortcomings, we explore a data-centric approach wherein we finetune CLIP models on large-scale synthetic datasets containing millions of negated captions. We show that this approach can result in a 10% increase in recall on negated queries and a 40% boost in accuracy on multiple-choice questions with negated captions. 

**Abstract (ZH)**: 许多实际的视觉-语言应用需要能够理解否定的模型，例如，使用自然语言检索包含某些对象但不包含其他对象的图片。尽管通过大规模训练在视觉-语言模型（VLMs）方面取得了进展，但它们对否定的理解能力仍远未探索充分。本研究旨在回答一个问题：当前的VLMs在理解否定方面究竟做得如何？我们引入了NegBench，这是首个旨在评估18种任务变体和79,000个示例（涵盖图像、视频和医疗数据集）中的否定理解能力的新基准。该基准包含两个核心任务，旨在评估在不同多模态设置中对否定的理解：带有否定检索和带有否定描述的多项选择题。我们的评估结果表明，现代VLMs在处理否定方面面临显著挑战，往往表现水平仅相当于随机猜测。为了应对这些不足，我们探索了一种以数据为中心的方法，在包含数百万否定描述的大型合成数据集上微调CLIP模型。我们发现，这种方法能够在否定查询上的召回率提升10%，并在带有否定描述的多项选择题上的准确率提升40%。 

---
# Efficient Few-Shot Medical Image Analysis via Hierarchical Contrastive Vision-Language Learning 

**Title (ZH)**: 通过分层对比视图语言学习的高效少样本医疗图像分析 

**Authors**: Harrison Fuller, Fernando Gabriela Garcia, Victor Flores  

**Link**: [PDF](https://arxiv.org/pdf/2501.09294)  

**Abstract**: Few-shot learning in medical image classification presents a significant challenge due to the limited availability of annotated data and the complex nature of medical imagery. In this work, we propose Adaptive Vision-Language Fine-tuning with Hierarchical Contrastive Alignment (HiCA), a novel framework that leverages the capabilities of Large Vision-Language Models (LVLMs) for medical image analysis. HiCA introduces a two-stage fine-tuning strategy, combining domain-specific pretraining and hierarchical contrastive learning to align visual and textual representations at multiple levels. We evaluate our approach on two benchmark datasets, Chest X-ray and Breast Ultrasound, achieving state-of-the-art performance in both few-shot and zero-shot settings. Further analyses demonstrate the robustness, generalizability, and interpretability of our method, with substantial improvements in performance compared to existing baselines. Our work highlights the potential of hierarchical contrastive strategies in adapting LVLMs to the unique challenges of medical imaging tasks. 

**Abstract (ZH)**: 在医学图像分类中，基于少量样本的学习面临着显著的挑战，这主要是由于标注数据的稀缺性和医学图像的复杂性。本文提出了一种名为Hierarchical Contrastive Alignment（HiCA）的适应性视觉-语言微调框架，该框架利用大型视觉-语言模型（LVLMs）的能力来解决医学图像分析问题。HiCA引入了一种两阶段微调策略，结合领域特定的预训练和层次对比学习，以在多个层次上对齐视觉和文本表示。我们在胸部X光和乳腺超声两个基准数据集上评估了该方法，在少量样本和零样本设置中均取得了最先进的性能。进一步的分析表明，该方法具有较强的鲁棒性、泛化能力和可解释性，并显著优于现有基线方法。本文强调了层次对比策略在适应LVLMs解决医学影像任务的独特挑战方面的潜力。 

---
# The Goofus & Gallant Story Corpus for Practical Value Alignment 

**Title (ZH)**: 《用于实践价值对齐的Goofus & Gallant故事情节语料库》 

**Authors**: Md Sultan Al Nahian, Tasmia Tasrin, Spencer Frazier, Mark Riedl, Brent Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2501.09707)  

**Abstract**: Values or principles are key elements of human society that influence people to behave and function according to an accepted standard set of social rules to maintain social order. As AI systems are becoming ubiquitous in human society, it is a major concern that they could violate these norms or values and potentially cause harm. Thus, to prevent intentional or unintentional harm, AI systems are expected to take actions that align with these principles. Training systems to exhibit this type of behavior is difficult and often requires a specialized dataset. This work presents a multi-modal dataset illustrating normative and non-normative behavior in real-life situations described through natural language and artistic images. This training set contains curated sets of images that are designed to teach young children about social principles. We argue that this is an ideal dataset to use for training socially normative agents given this fact. 

**Abstract (ZH)**: 价值观或原则是人类社会的关键要素，它们影响人们按照被广泛接受的社会规则行事，从而维持社会秩序。随着人工智能系统在人类社会中的普遍应用，人们越来越担心这些系统可能会违反这些规范或价值观，并可能导致潜在的危害。因此，为了防止有意或无意的危害，期望这些系统采取符合这些原则的行为。训练系统表现出这种行为是具有挑战性的，通常需要专门的数据集。本工作提出一个多模态数据集，该数据集通过自然语言和艺术图像描述了现实生活中的规范和非规范行为。该训练集包含一系列经过精心挑选的图像，旨在教育年幼的孩子了解社会原则。我们认为，鉴于这一点，这是一个理想的用于训练社会规范代理的数据集。 

---
# YETI (YET to Intervene) Proactive Interventions by Multimodal AI Agents in Augmented Reality Tasks 

**Title (ZH)**: YETI（尚未干预）：多模态AI代理在增强现实任务中的主动干预 

**Authors**: Saptarashmi Bandyopadhyay, Vikas Bahirwani, Lavisha Aggarwal, Bhanu Guda, Lin Li, Andrea Colaco  

**Link**: [PDF](https://arxiv.org/pdf/2501.09355)  

**Abstract**: Multimodal AI Agents are AI models that have the capability of interactively and cooperatively assisting human users to solve day-to-day tasks. Augmented Reality (AR) head worn devices can uniquely improve the user experience of solving procedural day-to-day tasks by providing egocentric multimodal (audio and video) observational capabilities to AI Agents. Such AR capabilities can help AI Agents see and listen to actions that users take which can relate to multimodal capabilities of human users. Existing AI Agents, either Large Language Models (LLMs) or Multimodal Vision-Language Models (VLMs) are reactive in nature, which means that models cannot take an action without reading or listening to the human user's prompts. Proactivity of AI Agents on the other hand can help the human user detect and correct any mistakes in agent observed tasks, encourage users when they do tasks correctly or simply engage in conversation with the user - akin to a human teaching or assisting a user. Our proposed YET to Intervene (YETI) multimodal agent focuses on the research question of identifying circumstances that may require the agent to intervene proactively. This allows the agent to understand when it can intervene in a conversation with human users that can help the user correct mistakes on tasks, like cooking, using AR. Our YETI Agent learns scene understanding signals based on interpretable notions of Structural Similarity (SSIM) on consecutive video frames. We also define the alignment signal which the AI Agent can learn to identify if the video frames corresponding to the user's actions on the task are consistent with expected actions. These signals are used by our AI Agent to determine when it should proactively intervene. We compare our results on the instances of proactive intervention in the HoloAssist multimodal benchmark for an expert agent guiding a user to complete procedural tasks. 

**Abstract (ZH)**: 多模态AI代理是能够交互和协作帮助人类用户解决日常任务的AI模型。增强现实（AR）头戴设备通过为AI代理提供以自我为中心的多模态（音频和视频）观察能力，能够独特地提升解决程序性日常任务的用户体验。这些AR能力可以帮助AI代理看到并听到用户进行的动作，从而与人类用户的多模态能力相关联。现有的AI代理，无论是大型语言模型（LLM）还是多模态视觉语言模型（VLM），本质上是反应性的，这意味着模型在执行动作之前需要读取或听取人类用户的提示。另一方面，AI代理的主动性可以帮助人类用户检测并纠正代理观察到的任务中的错误、鼓励用户正确完成任务，或者简单地与用户进行对话，如同人类在指导用户或帮助用户一样。我们提出的YET to Intervene（YETI）多模态代理专注于研究识别需要代理主动干预的情况。这使代理能够理解何时在与人类用户的对话中可以主动干预，以帮助用户纠正任务中的错误，比如烹饪，在AR的帮助下。我们的YETI代理基于连续视频帧的可解释结构相似性（SSIM）学习场景理解信号。我们还定义了对齐信号，该信号使AI代理能够识别出与任务中用户动作对应的视频帧是否与预期动作一致。这些信号被我们的AI代理用于决定何时应主动干预。我们在HoloAssist多模态基准数据集上比较了主动干预实例的实验结果，该基准数据集用于专家代理指导用户完成程序性任务。 

---
# Text Semantics to Flexible Design: A Residential Layout Generation Method Based on Stable Diffusion Model 

**Title (ZH)**: 文本含义到灵活设计：基于稳定扩散模型的住宅布局生成方法 

**Authors**: Zijin Qiu, Jiepeng Liu, Yi Xia, Hongtuo Qi, Pengkun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09279)  

**Abstract**: Flexibility in the AI-based residential layout design remains a significant challenge, as traditional methods like rule-based heuristics and graph-based generation often lack flexibility and require substantial design knowledge from users. To address these limitations, we propose a cross-modal design approach based on the Stable Diffusion model for generating flexible residential layouts. The method offers multiple input types for learning objectives, allowing users to specify both boundaries and layouts. It incorporates natural language as design constraints and introduces ControlNet to enable stable layout generation through two distinct pathways. We also present a scheme that encapsulates design expertise within a knowledge graph and translates it into natural language, providing an interpretable representation of design knowledge. This comprehensibility and diversity of input options enable professionals and non-professionals to directly express design requirements, enhancing flexibility and controllability. Finally, experiments verify the flexibility of the proposed methods under multimodal constraints better than state-of-the-art models, even when specific semantic information about room areas or connections is incomplete. 

**Abstract (ZH)**: 基于人工智能的住宅布局设计灵活性仍然是一个显著挑战，传统的规则基启发式方法和图基生成方法往往缺乏灵活性，并且需要用户具备大量的设计知识。为了解决这些限制，我们提出了一种基于Stable Diffusion模型的跨模态设计方法，以生成灵活的住宅布局。该方法支持多种输入类型的学习目标，允许用户指定边界和布局。它将自然语言作为设计约束，并引入ControlNet，通过两条不同的路径实现稳定的布局生成。我们还提出了一种方案，将设计专业知识封装到知识图谱中，并将其转换为自然语言，提供设计知识的可解释表示。这种可解释性及多样化的输入选项使专业人士和非专业人士可以直接表达设计要求，从而增强灵活性和可控性。最后，实验结果表明，在多模态约束条件下，所提出的方法在灵活性方面优于最先进的模型，即使缺乏关于房间面积或连接的具体语义信息也是如此。 

---
# A Simple Aerial Detection Baseline of Multimodal Language Models 

**Title (ZH)**: 一种基于多模态语言模型的简单航空目标检测基准方法 

**Authors**: Qingyun Li, Yushi Chen, Xinya Shu, Dong Chen, Xin He, Yi Yu, Xue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09720)  

**Abstract**: The multimodal language models (MLMs) based on generative pre-trained Transformer are considered powerful candidates for unifying various domains and tasks. MLMs developed for remote sensing (RS) have demonstrated outstanding performance in multiple tasks, such as visual question answering and visual grounding. In addition to visual grounding that detects specific objects corresponded to given instruction, aerial detection, which detects all objects of multiple categories, is also a valuable and challenging task for RS foundation models. However, aerial detection has not been explored by existing RS MLMs because the autoregressive prediction mechanism of MLMs differs significantly from the detection outputs. In this paper, we present a simple baseline for applying MLMs to aerial detection for the first time, named LMMRotate. Specifically, we first introduce a normalization method to transform detection outputs into textual outputs to be compatible with the MLM framework. Then, we propose a evaluation method, which ensures a fair comparison between MLMs and conventional object detection models. We construct the baseline by fine-tuning open-source general-purpose MLMs and achieve impressive detection performance comparable to conventional detector. We hope that this baseline will serve as a reference for future MLM development, enabling more comprehensive capabilities for understanding RS images. Code is available at this https URL. 

**Abstract (ZH)**: 基于生成预训练Transformer的多模态语言模型（MLMs）被认为是统一各种领域和任务的强大候选者。专为遥感（RS）开发的MLMs在多个任务中表现出色，例如视觉问答和视觉定位。除了能够检测给定指令对应的具体对象的视觉定位任务外，能够检测多种类别的所有对象的空中检测也是一个有价值的挑战性任务，为RS基础模型提供服务。然而，现有的RS MLMs尚未探索空中检测任务，因为MLMs的自回归预测机制与检测输出存在显著差异。在本文中，我们首次提出了一个简单的基线模型LMMRotate，将MLMs应用于空中检测。具体而言，我们首先介绍了一种归一化方法，将检测输出转换为文本输出，使其与MLM框架兼容。然后，我们提出了一种评估方法，以确保MLMs和传统目标检测模型之间的公平比较。我们通过微调开源的通用MLM模型构建基线，并实现了可与传统检测器相媲美的检测性能。我们希望这个基线能够为未来MLM的发展提供参考，从而使MLM能够更好地理解RS图像的能力。代码可在此网址获得：this https URL。 

---
# Robin: a Suite of Multi-Scale Vision-Language Models and the CHIRP Evaluation Benchmark 

**Title (ZH)**: Robin：多尺度视觉-语言模型套件及CHIRP评估基准 

**Authors**: Alexis Roger, Prateek Humane, Daniel Z. Kaplan, Kshitij Gupta, Qi Sun, George Adamopoulos, Jonathan Siu Chi Lim, Quentin Anthony, Edwin Fennell, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2501.09672)  

**Abstract**: The proliferation of Vision-Language Models (VLMs) in the past several years calls for rigorous and comprehensive evaluation methods and benchmarks. This work analyzes existing VLM evaluation techniques, including automated metrics, AI-based assessments, and human evaluations across diverse tasks. We first introduce Robin - a novel suite of VLMs that we built by combining Large Language Models (LLMs) and Vision Encoders (VEs) at multiple scales, and use Robin to identify shortcomings of current evaluation approaches across scales. Next, to overcome the identified limitations, we introduce CHIRP - a new long form response benchmark we developed for more robust and complete VLM evaluation. We provide open access to the Robin training code, model suite, and CHIRP benchmark to promote reproducibility and advance VLM research. 

**Abstract (ZH)**: 过去几年中视觉-语言模型（VLMs）的快速发展需要严格的和全面的评估方法和基准。本文分析了现有的VLM评估技术，包括自动度量、基于AI的评估以及在多种任务上的手工评估。首先，我们介绍了Robin——一种新型的VLM套件，我们通过结合大规模语言模型（LLMs）和视觉编码器（VEs）构建，多尺度地使用Robin来识别当前评估方法在不同尺度上的不足。接下来，为了克服这些已识别的局限性，我们引入了CHIRP——一种新的长条目响应基准，旨在进行更加稳健和全面的VLM评估。我们免费提供了Robin的训练代码、模型套件和CHIRP基准，以促进可再现性并推动VLM研究的发展。 

---
# LAVCap: LLM-based Audio-Visual Captioning using Optimal Transport 

**Title (ZH)**: LAVCap：基于最优 transport 的大型语言模型驱动的音视频描述生成 

**Authors**: Kyeongha Rho, Hyeongkeun Lee, Valentio Iverson, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2501.09291)  

**Abstract**: Automated audio captioning is a task that generates textual descriptions for audio content, and recent studies have explored using visual information to enhance captioning quality. However, current methods often fail to effectively fuse audio and visual data, missing important semantic cues from each modality. To address this, we introduce LAVCap, a large language model (LLM)-based audio-visual captioning framework that effectively integrates visual information with audio to improve audio captioning performance. LAVCap employs an optimal transport-based alignment loss to bridge the modality gap between audio and visual features, enabling more effective semantic extraction. Additionally, we propose an optimal transport attention module that enhances audio-visual fusion using an optimal transport assignment map. Combined with the optimal training strategy, experimental results demonstrate that each component of our framework is effective. LAVCap outperforms existing state-of-the-art methods on the AudioCaps dataset, without relying on large datasets or post-processing. Code is available at this https URL. 

**Abstract (ZH)**: 自动音频字幕生成是一项生成音频内容文本描述的任务，近年来的研究探讨了利用视觉信息来提高字幕质量的方法。然而，当前的方法往往无法有效地融合音频和视觉数据，从而错过了每种模态中的重要语义线索。为解决这一问题，我们引入了LAVCap，这是一种基于大规模语言模型（LLM）的音频-视觉字幕生成框架，能够有效地将视觉信息与音频融合，以提高音频字幕生成性能。LAVCap 采用基于运筹学的对齐损失来弥合音频和视觉特征之间的模态差异，从而更好地提取语义信息。此外，我们提出了一个基于运筹学注意力模块，该模块使用运筹学分配图增强了音频-视觉融合能力。结合最佳训练策略，实验结果表明，框架中的每个组件都是有效的。LAVCap 在 AudioCaps 数据集上的性能超过了现有最先进的方法，且无需依赖大规模数据集或后续处理。相关代码可从此链接访问：[点击访问代码] 

---
# Playing Devil's Advocate: Unmasking Toxicity and Vulnerabilities in Large Vision-Language Models 

**Title (ZH)**: 扮演魔鬼代言人：揭示大型视觉-语言模型中的毒性与脆弱性 

**Authors**: Abdulkadir Erol, Trilok Padhi, Agnik Saha, Ugur Kursuncu, Mehmet Emin Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2501.09039)  

**Abstract**: The rapid advancement of Large Vision-Language Models (LVLMs) has enhanced capabilities offering potential applications from content creation to productivity enhancement. Despite their innovative potential, LVLMs exhibit vulnerabilities, especially in generating potentially toxic or unsafe responses. Malicious actors can exploit these vulnerabilities to propagate toxic content in an automated (or semi-) manner, leveraging the susceptibility of LVLMs to deception via strategically crafted prompts without fine-tuning or compute-intensive procedures. Despite the red-teaming efforts and inherent potential risks associated with the LVLMs, exploring vulnerabilities of LVLMs remains nascent and yet to be fully addressed in a systematic manner. This study systematically examines the vulnerabilities of open-source LVLMs, including LLaVA, InstructBLIP, Fuyu, and Qwen, using adversarial prompt strategies that simulate real-world social manipulation tactics informed by social theories. Our findings show that (i) toxicity and insulting are the most prevalent behaviors, with the mean rates of 16.13% and 9.75%, respectively; (ii) Qwen-VL-Chat, LLaVA-v1.6-Vicuna-7b, and InstructBLIP-Vicuna-7b are the most vulnerable models, exhibiting toxic response rates of 21.50%, 18.30% and 17.90%, and insulting responses of 13.40%, 11.70% and 10.10%, respectively; (iii) prompting strategies incorporating dark humor and multimodal toxic prompt completion significantly elevated these vulnerabilities. Despite being fine-tuned for safety, these models still generate content with varying degrees of toxicity when prompted with adversarial inputs, highlighting the urgent need for enhanced safety mechanisms and robust guardrails in LVLM development. 

**Abstract (ZH)**: 大型视觉语言模型（LVLMs）的迅速发展提升了其能力，从而使内容创作和生产力提升等方面的应用成为可能。尽管这些模型具有创新潜力，但它们也表现出某些脆弱性，特别是在生成潜在有害或不安全的回应方面尤为突出。恶意行为者可以通过利用这些脆弱性，以自动化（或半自动化）的方式传播有害内容，并通过精心设计的提示来欺骗LVLMs，无需进行微调或密集计算的过程。尽管对LVLMs进行了红队测试，并存在固有的潜在风险，但探索LVLMs的脆弱性仍处于起步阶段，尚未以系统化的方式得到充分解决。本研究系统地考察了开源LVLMs（包括LLaVA、InstructBLIP、Fuyu和Qwen）的脆弱性，使用通过社会理论启发的对抗性提示策略模拟现实生活中的社会操纵技巧。研究结果表明：（i）毒性和侮辱是最常见的行为，平均率为16.13%和9.75%；（ii）Qwen-VL-Chat、LLaVA-v1.6-Vicuna-7b和InstructBLIP-Vicuna-7b是表现最为脆弱的模型，展现出的有害回应率为21.50%、18.30%和17.90%，侮辱回应率为13.40%、11.70%和10.10%；（iii）包含黑色幽默和多模态有害提示的提示策略显著增强了这些脆弱性。尽管这些模型经过了安全微调，但在受到对手输入提示时，仍然会生成不同程度的有害内容，这突显了在LVLM开发中增强安全机制和强化护栏的紧迫性。 

---
