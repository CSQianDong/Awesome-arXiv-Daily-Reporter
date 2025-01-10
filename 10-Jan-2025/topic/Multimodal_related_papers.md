# Multimodal-to-Text Prompt Engineering in Large Language Models Using Feature Embeddings for GNSS Interference Characterization 

**Title (ZH)**: 使用特征嵌入进行全球导航卫星系统干扰特征化的大语言模型多模态到文本提示工程 

**Authors**: Harshith Manjunath, Lucas Heublein, Tobias Feigl, Felix Ott  

**Link**: [PDF](https://arxiv.org/pdf/2501.05079)  

**Abstract**: Large language models (LLMs) are advanced AI systems applied across various domains, including NLP, information retrieval, and recommendation systems. Despite their adaptability and efficiency, LLMs have not been extensively explored for signal processing tasks, particularly in the domain of global navigation satellite system (GNSS) interference monitoring. GNSS interference monitoring is essential to ensure the reliability of vehicle localization on roads, a critical requirement for numerous applications. However, GNSS-based positioning is vulnerable to interference from jamming devices, which can compromise its accuracy. The primary objective is to identify, classify, and mitigate these interferences. Interpreting GNSS snapshots and the associated interferences presents significant challenges due to the inherent complexity, including multipath effects, diverse interference types, varying sensor characteristics, and satellite constellations. In this paper, we extract features from a large GNSS dataset and employ LLaVA to retrieve relevant information from an extensive knowledge base. We employ prompt engineering to interpret the interferences and environmental factors, and utilize t-SNE to analyze the feature embeddings. Our findings demonstrate that the proposed method is capable of visual and logical reasoning within the GNSS context. Furthermore, our pipeline outperforms state-of-the-art machine learning models in interference classification tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）是应用于多个领域，包括自然语言处理（NLP）、信息检索和推荐系统等的高级人工智能系统。尽管它们具有高度适应性和高效性，但LLMs在信号处理任务中的应用尚不广泛，特别是在全球导航卫星系统（GNSS）干扰监测领域。GNSS干扰监测对于确保道路车辆定位的可靠性至关重要，这是许多应用的关键需求。然而，基于GNSS的定位系统容易受到干扰设备（如宽带干扰器）的影响，这些干扰会损害其精度。主要目标是识别、分类和减轻这些干扰。解读GNSS快照及其相关的干扰具有显著的挑战性，这些挑战源于其固有的复杂性，包括多路径效应、多种类型的干扰、差异化的传感器特性以及卫星星座的多样性。在本文中，我们从大规模GNSS数据集中提取特征，并使用LLaVA从广泛的知识库中检索相关信息。我们采用提示工程技术来解释干扰和环境因素，并利用t-SNE分析特征嵌入。我们的研究结果表明，所提出的方法能够在GNSS上下文中进行视觉和逻辑推理。此外，我们的流程在干扰分类任务中优于现有的机器学习模型。 

---
# A General Retrieval-Augmented Generation Framework for Multimodal Case-Based Reasoning Applications 

**Title (ZH)**: 一种用于多模态案例推理应用的通用检索增强生成框架 

**Authors**: Ofir Marom  

**Link**: [PDF](https://arxiv.org/pdf/2501.05030)  

**Abstract**: Case-based reasoning (CBR) is an experience-based approach to problem solving, where a repository of solved cases is adapted to solve new cases. Recent research shows that Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) can support the Retrieve and Reuse stages of the CBR pipeline by retrieving similar cases and using them as additional context to an LLM query. Most studies have focused on text-only applications, however, in many real-world problems the components of a case are multimodal. In this paper we present MCBR-RAG, a general RAG framework for multimodal CBR applications. The MCBR-RAG framework converts non-text case components into text-based representations, allowing it to: 1) learn application-specific latent representations that can be indexed for retrieval, and 2) enrich the query provided to the LLM by incorporating all case components for better context. We demonstrate MCBR-RAG's effectiveness through experiments conducted on a simplified Math-24 application and a more complex Backgammon application. Our empirical results show that MCBR-RAG improves generation quality compared to a baseline LLM with no contextual information provided. 

**Abstract (ZH)**: 案例基于推理（CBR）是一种基于经验的解决问题方法，在这种方法中，通过调整已解决的案例库来解决新问题。最近的研究表明，附有检索增强生成（RAG）的大语言模型（LLMs）可以支持CBR工作流中的检索和重用阶段，通过检索相似的案例，并将其作为附加上下文提供给LLM查询。大多数研究集中在纯文本应用上，然而，在许多实际问题中，案例的组件是多模态的。在本文中，我们提出了MCBR-RAG，这是一种适用于多模态CBR应用的一般RAG框架。MCBR-RAG框架将非文本案例组件转换为文本表示，使其能够：1）学习特定应用的潜在表示，这些表示可以进行索引以供检索，2）通过结合所有案例组件来丰富对LLM的查询，提供更好的上下文。我们通过在简化版的Math-24应用和更复杂的背投棋应用中进行的实验，证明了MCBR-RAG的有效性。我们的实证结果表明，MCBR-RAG在提供上下文信息的情况下相比没有提供上下文信息的基线LLM，提高了生成质量。 

---
# Towards Balanced Continual Multi-Modal Learning in Human Pose Estimation 

**Title (ZH)**: 面向人体姿态估计的均衡持续多模态学习研究 

**Authors**: Jiaxuan Peng, Mengshi Qi, Dong Zhao, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.05264)  

**Abstract**: 3D human pose estimation (3D HPE) has emerged as a prominent research topic, particularly in the realm of RGB-based methods. However, RGB images are susceptible to limitations such as sensitivity to lighting conditions and potential user discomfort. Consequently, multi-modal sensing, which leverages non-intrusive sensors, is gaining increasing attention. Nevertheless, multi-modal 3D HPE still faces challenges, including modality imbalance and the imperative for continual learning. In this work, we introduce a novel balanced continual multi-modal learning method for 3D HPE, which harnesses the power of RGB, LiDAR, mmWave, and WiFi. Specifically, we propose a Shapley value-based contribution algorithm to quantify the contribution of each modality and identify modality imbalance. To address this imbalance, we employ a re-learning strategy. Furthermore, recognizing that raw data is prone to noise contamination, we develop a novel denoising continual learning approach. This approach incorporates a noise identification and separation module to mitigate the adverse effects of noise and collaborates with the balanced learning strategy to enhance optimization. Additionally, an adaptive EWC mechanism is employed to alleviate catastrophic forgetting. We conduct extensive experiments on the widely-adopted multi-modal dataset, MM-Fi, which demonstrate the superiority of our approach in boosting 3D pose estimation and mitigating catastrophic forgetting in complex scenarios. We will release our codes. 

**Abstract (ZH)**: 基于RGB的方法在三维人体姿态估计（3D HPE）领域中已成为一个重要的研究课题。然而，RGB图像存在一些局限性，如对光照条件的敏感性和潜在的用户不适。因此，多模态传感技术逐渐受到重视，这种方法利用非侵入性传感器。尽管如此，多模态3D HPE仍然面临着模态不平衡和持续学习的迫切需求。在本研究中，我们介绍了一种新的平衡持续多模态学习方法，以增强三维人体姿态估计，并解决模态不平衡的问题。该方法结合了RGB、LiDAR、毫米波（mmWave）和WiFi等多种模态。具体来说，我们提出了一种基于Shapley值的贡献算法，用于量化每个模态的贡献并识别模态不平衡。为解决这一不平衡问题，我们采用了重训练策略。此外，考虑到原始数据容易受到噪声污染的影响，我们开发了一种新的去噪持续学习方法。该方法包含了噪声识别和分离模块，以减轻噪声的负面影响，并与平衡学习策略协作，以增强优化效果。此外，我们采用了自适应EWC机制来缓解灾难性遗忘的问题。我们在广泛应用的多模态数据集MM-Fi上进行了广泛的实验，结果表明，我们的方法在提高三维姿态估计性能和在复杂场景中减轻灾难性遗忘方面具有明显的优势。我们还将发布我们的代码。 

---
# GLaM-Sign: Greek Language Multimodal Lip Reading with Integrated Sign Language Accessibility 

**Title (ZH)**: GLaM-Sign: 带有整合的手语 Accessibility 的希腊语多模态唇读 

**Authors**: Dimitris Kouremenos, Klimis Ntalianis  

**Link**: [PDF](https://arxiv.org/pdf/2501.05213)  

**Abstract**: The Greek Language Multimodal Lip Reading with Integrated Sign Language Accessibility (GLaM-Sign) [1] is a groundbreaking resource in accessibility and multimodal AI, designed to support Deaf and Hard-of-Hearing (DHH) individuals. Developed from the FEELIT project [2], it integrates high-resolution audio, video, textual transcriptions, and Greek Sign Language translations for applications like real-time sign language translation and enhanced subtitle synchronization. While its primary focus is on promoting inclusivity in the Greek tourism sector, its adaptability extends to education, healthcare, and public services. Future advancements will enhance word-level precision and scalability to additional languages, supported by advanced AI methodologies and collaborations with diverse stakeholders. This dataset underscores the transformative potential of multimodal resources in bridging communication gaps, fostering innovation, and setting a benchmark for ethical AI and inclusive technologies. 

**Abstract (ZH)**: 《希腊语言多模态唇读与集成手语访问性(GLaM-Sign)》[1] 是无障碍技术和多模态人工智能领域的开创性资源，旨在支持聋人和听力障碍者（DHH）群体。该项目源自于FEELIT项目[2]，整合了高分辨率音频、视频、文本转录以及希腊手语翻译，适用于实时手语翻译和增强字幕同步等多种应用。尽管其主要集中在促进希腊旅游行业的包容性，但其灵活性还可扩展到教育、医疗保健和公共服务领域。未来的发展将进一步提高单词级别的精确度，并扩展至其他语言，通过先进的AI方法和与多元利益相关方的合作来实现。该数据集展示了多模态资源在弥合沟通障碍、促进创新和为伦理AI与包容性技术设定标杆方面的潜在影响。 

---
# LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding 

**Title (ZH)**: LLaVA-Octopus：解锁基于指令的自适应投影融合以实现视频理解 

**Authors**: Jiaxing Zhao, Boyuan Sun, Xiang Chen, Xihan Wei, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2501.05067)  

**Abstract**: In this paper, we introduce LLaVA-Octopus, a novel video multimodal large language model. LLaVA-Octopus adaptively weights features from different visual projectors based on user instructions, enabling us to leverage the complementary strengths of each projector. We observe that different visual projectors exhibit distinct characteristics when handling specific tasks. For instance, some projectors excel at capturing static details, while others are more effective at processing temporal information, and some are better suited for tasks requiring temporal coherence. By dynamically adjusting feature weights according to user instructions, LLaVA-Octopus dynamically selects and combines the most suitable features, significantly enhancing the model's performance in multimodal tasks. Experimental results demonstrate that LLaVA-Octopus achieves excellent performance across multiple benchmarks, especially in tasks such as multimodal understanding, visual question answering, and video understanding, highlighting its broad application potential. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新的视频多模态大型语言模型LLaVA-Octopus。LLaVA-Octopus根据用户指令自适应地加权来自不同视觉投影器的特征，从而能够利用每个投影器的互补优势。我们观察到，不同的视觉投影器在处理特定任务时表现出不同的特性。例如，有些投影器擅长捕捉静态细节，而另一些则更擅长处理时间信息，还有一些则更适合需要时间连贯性的任务。通过根据用户指令动态调整特征权重，LLaVA-Octopus能够动态选择和组合最适合的特征，显著增强了模型在多模态任务中的性能。实验结果表明，LLaVA-Octopus在多个基准测试中表现出色，尤其是在多模态理解、视觉问答和视频理解等任务中，突显了其广泛的应用潜力。 

---
# UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation 

**Title (ZH)**: UAV-VLA：大规模空中任务生成的视觉-语言-行动系统 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Artem Lykov, Muhammad Ahsan Mustafa, Grik Tadevosyan, Aibek Akhmetkazy, Miguel Altamirano Cabrera, Mikhail Martynov, Sausar Karaf, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2501.05014)  

**Abstract**: The UAV-VLA (Visual-Language-Action) system is a tool designed to facilitate communication with aerial robots. By integrating satellite imagery processing with the Visual Language Model (VLM) and the powerful capabilities of GPT, UAV-VLA enables users to generate general flight paths-and-action plans through simple text requests. This system leverages the rich contextual information provided by satellite images, allowing for enhanced decision-making and mission planning. The combination of visual analysis by VLM and natural language processing by GPT can provide the user with the path-and-action set, making aerial operations more efficient and accessible. The newly developed method showed the difference in the length of the created trajectory in 22% and the mean error in finding the objects of interest on a map in 34.22 m by Euclidean distance in the K-Nearest Neighbors (KNN) approach. 

**Abstract (ZH)**: UAV-VLA（视觉-语言-动作）系统是一种旨在促进与空中机器人通信的工具。通过将卫星影像处理与视觉语言模型（VLM）以及GPT的强大能力相结合，UAV-VLA允许用户通过简单的文本请求生成通用的飞行路径和行动计划。该系统利用卫星影像提供的丰富上下文信息，增强了决策能力和任务规划。VLM的视觉分析与GPT的自然语言处理相结合，使用户能够获得路径和行动集，从而提高空中操作的效率和 accessibility。新开发的方法在K-近邻（KNN）方法中显示了创建轨迹长度的差异为22%，以及通过欧几里得距离在寻找地图上感兴趣的目标时的平均误差为34.22米。 

---
# Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model 

**Title (ZH)**: Centurio：大型视觉-语言模型的多语言能力驱动因素探究 

**Authors**: Gregor Geigle, Florian Schneider, Carolin Holtermann, Chris Biemann, Radu Timofte, Anne Lauscher, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2501.05122)  

**Abstract**: Most Large Vision-Language Models (LVLMs) to date are trained predominantly on English data, which makes them struggle to understand non-English input and fail to generate output in the desired target language. Existing efforts mitigate these issues by adding multilingual training data, but do so in a largely ad-hoc manner, lacking insight into how different training mixes tip the scale for different groups of languages. In this work, we present a comprehensive investigation into the training strategies for massively multilingual LVLMs. First, we conduct a series of multi-stage experiments spanning 13 downstream vision-language tasks and 43 languages, systematically examining: (1) the number of training languages that can be included without degrading English performance and (2) optimal language distributions of pre-training as well as (3) instruction-tuning data. Further, we (4) investigate how to improve multilingual text-in-image understanding, and introduce a new benchmark for the task. Surprisingly, our analysis reveals that one can (i) include as many as 100 training languages simultaneously (ii) with as little as 25-50\% of non-English data, to greatly improve multilingual performance while retaining strong English performance. We further find that (iii) including non-English OCR data in pre-training and instruction-tuning is paramount for improving multilingual text-in-image understanding. Finally, we put all our findings together and train Centurio, a 100-language LVLM, offering state-of-the-art performance in an evaluation covering 14 tasks and 56 languages. 

**Abstract (ZH)**: 迄今为止，大多数大型视觉-语言模型（Large Vision-Language Models, LVLMs）主要在英语数据上进行训练，这使得它们在理解非英语输入和生成目标语言输出方面存在问题。现有的努力通过增加多语言训练数据来缓解这些问题，但在很大程度上是随意进行的，缺乏对不同训练混合如何影响不同语言群体的表现的深入理解。在本工作中，我们对大规模多语言LVLM的训练策略进行了全面调查。首先，我们进行了跨越13个下游视觉-语言任务和43种语言的多阶段实验，系统地探讨了：（1）不损害英语性能的情况下可以包含的训练语言数量；（2）预训练和（3）指令调优数据的最佳语言分布。此外，我们（4）探讨如何提高多语言文本-图像理解能力，并引入了该任务的新基准。令人惊讶的是，我们的分析显示，可以同时（i）包含多达100种训练语言（ii）仅使用25-50%的非英语数据，从而大幅提高多语言性能，同时保留良好的英语性能。我们还发现，（iii）在预训练和指令调优中包含非英语OCR数据对于提高多语言文本-图像理解至关重要。最后，我们将所有这些发现结合起来，训练了一个包含100种语言的Centurio模型，在涵盖14个任务和56种语言的评估中表现出最先进的性能。 

---
# Cued Speech Generation Leveraging a Pre-trained Audiovisual Text-to-Speech Model 

**Title (ZH)**: 利用预训练的音视频文本转语音模型生成指式手语 

**Authors**: Sanjana Sankar, Martin Lenglet, Gerard Bailly, Denis Beautemps, Thomas Hueber  

**Link**: [PDF](https://arxiv.org/pdf/2501.04799)  

**Abstract**: This paper presents a novel approach for the automatic generation of Cued Speech (ACSG), a visual communication system used by people with hearing impairment to better elicit the spoken language. We explore transfer learning strategies by leveraging a pre-trained audiovisual autoregressive text-to-speech model (AVTacotron2). This model is reprogrammed to infer Cued Speech (CS) hand and lip movements from text input. Experiments are conducted on two publicly available datasets, including one recorded specifically for this study. Performance is assessed using an automatic CS recognition system. With a decoding accuracy at the phonetic level reaching approximately 77%, the results demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 本文提出了一种新颖的方法，用于自动生成视觉手势编码语言（Cued Speech，CS），这是一种专为听力受损人士设计的视觉交流系统，旨在更好地诱发口语表达。我们通过利用预训练的音频-视觉自回归文本转语音模型（AVTacotron2）探索迁移学习策略。该模型被重新编程，以从文本输入中推断出Cued Speech的手部和唇部动作。我们在两个公开可用的数据集上进行了实验，其中包括一个为本研究专门录制的数据集。性能评估使用了自动Cued Speech识别系统。结果显示，在音素层面的解码准确率达到约77%，证明了该方法的有效性。 

---
# ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding 

**Title (ZH)**: ReFocus: 视觉编辑作为结构化图像理解的思维链 

**Authors**: Xingyu Fu, Minqian Liu, Zhengyuan Yang, John Corring, Yijuan Lu, Jianwei Yang, Dan Roth, Dinei Florencio, Cha Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05452)  

**Abstract**: Structured image understanding, such as interpreting tables and charts, requires strategically refocusing across various structures and texts within an image, forming a reasoning sequence to arrive at the final answer. However, current multimodal large language models (LLMs) lack this multihop selective attention capability. In this work, we introduce ReFocus, a simple yet effective framework that equips multimodal LLMs with the ability to generate "visual thoughts" by performing visual editing on the input image through code, shifting and refining their visual focuses. Specifically, ReFocus enables multimodal LLMs to generate Python codes to call tools and modify the input image, sequentially drawing boxes, highlighting sections, and masking out areas, thereby enhancing the visual reasoning process. We experiment upon a wide range of structured image understanding tasks involving tables and charts. ReFocus largely improves performance on all tasks over GPT-4o without visual editing, yielding an average gain of 11.0% on table tasks and 6.8% on chart tasks. We present an in-depth analysis of the effects of different visual edits, and reasons why ReFocus can improve the performance without introducing additional information. Further, we collect a 14k training set using ReFocus, and prove that such visual chain-of-thought with intermediate information offers a better supervision than standard VQA data, reaching a 8.0% average gain over the same model trained with QA pairs and 2.6% over CoT. 

**Abstract (ZH)**: 结构化图像理解，例如解释表格和图表，需要在一个图像中战略性地重新聚焦于各种结构和文本，形成推理序列以得出最终答案。然而，当前的多模态大型语言模型（LLMs）缺乏这种多跳选择性注意力的能力。在本研究中，我们提出了ReFocus，这是一种简单而有效的框架，通过代码对输入图像进行视觉编辑，使多模态LLMs能够生成“视觉思考”，并逐步调整和细化其视觉关注点。具体来说，ReFocus使多模态LLMs能够生成Python代码来调用工具并修改输入图像，依次绘制框、突出显示部分并遮盖区域，从而增强视觉推理过程。我们在涉及表格和图表的广泛结构化图像理解任务中进行了实验。ReFocus在无需视觉编辑的情况下显著提高了GPT-4o的表现，在表格任务中平均提高了11.0%，在图表任务中提高了6.8%。我们深入分析了不同视觉编辑的效果及其为何能够在不引入额外信息的情况下提高表现。此外，我们使用ReFocus收集了一个包含14,000个训练样本的数据集，并证明了这种包含中间信息的视觉链式思考比标准的VQA数据提供了更好的监督，模型在使用QA对和CoT训练时分别获得了8.0%和2.6%的平均性能提升。 

---
# Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency 

**Title (ZH)**: 通过混合不一致性破解多模态大型语言模型 

**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.04931)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can improve the attack's performance on three benchmarks. In particular, SI-Attack can obviously improve the attack success rate for commercial MLLMs such as GPT-4o or Claude-3.5-Sonnet. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）已在商业应用中取得了显著的性能，并被实际应用，但它们仍然存在潜在的安全机制漏洞。押解攻击是红队测试方法，旨在绕过安全机制并发现MLLMs的潜在风险。现有的MLLMs押解方法通常通过复杂的优化方法或精心设计的图像和文本提示来绕过模型的安全机制。尽管取得了一些进展，但它们在商业闭源MLLMs上的攻击成功率较低。与以往研究不同，我们通过实证研究发现，MLLMs在处理乱序有害指令时存在理解和安全性之间的不一致性（Shuffle Inconsistency）。也就是说，从理解能力的角度来看，MLLMs能够很好地理解乱序有害文本-图像指令。然而，从安全性角度来看，它们却容易被乱序有害指令绕过，从而产生有害的响应。然后，我们创新地提出了一种名为SI-Attack的图文押解攻击方法。具体而言，为了充分利用这种不一致性并克服乱序的随机性，我们应用了一种基于查询的黑盒优化方法，根据有毒法官模型的反馈选择最具有害的乱序输入。一系列实验表明，SI-Attack可以提高在三个基准上的攻击性能。特别是，SI-Attack能够明显提高对如GPT-4o或Claude-3.5-Sonnet等商业MLLMs的攻击成功率。 

---
