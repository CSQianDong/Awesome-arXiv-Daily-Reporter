# ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting 

**Title (ZH)**: ReVision：一种用于隐私保护任务导向视觉指令重写的数据集和基线多模态模型 

**Authors**: Abhijit Mishra, Richard Noh, Hsiang Fu, Mingda Li, Minji Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.14780)  

**Abstract**: Efficient and privacy-preserving multimodal interaction is essential as AR, VR, and modern smartphones with powerful cameras become primary interfaces for human-computer communication. Existing powerful large vision-language models (VLMs) enabling multimodal interaction often rely on cloud-based processing, raising significant concerns about (1) visual privacy by transmitting sensitive vision data to servers, and (2) their limited real-time, on-device usability. This paper explores Visual Instruction Rewriting, a novel approach that transforms multimodal instructions into text-only commands, allowing seamless integration of lightweight on-device instruction rewriter VLMs (250M parameters) with existing conversational AI systems, enhancing vision data privacy. To achieve this, we present a dataset of over 39,000 examples across 14 domains and develop a compact VLM, pretrained on image captioning datasets and fine-tuned for instruction rewriting. Experimental results, evaluated through NLG metrics such as BLEU, METEOR, and ROUGE, along with semantic parsing analysis, demonstrate that even a quantized version of the model (<500MB storage footprint) can achieve effective instruction rewriting, thus enabling privacy-focused, multimodal AI applications. 

**Abstract (ZH)**: 高效且保护隐私的多模态交互对于AR、VR以及现代配备强大摄像头的智能手机成为人机通信的主要接口至关重要。现有的强大视觉-语言模型（VLMs）虽然能够实现多模态交互，但往往依赖云处理，这引发了许多关于（1）视觉隐私方面的担忧，即传输敏感的视觉数据至服务器，以及（2）其在设备上的实时性和使用便利性的局限性。本文探讨了一种名为视觉指令改写的创新方法，该方法将多模态指令转换为纯文本命令，从而使轻量级的在设备上运行的指令重写VLMs（参数量250M）能够与现有的对话式AI系统无缝集成，从而增强视觉数据的隐私性。

为实现这一目标，我们构建了一个包含超过39,000个样本的跨14个领域的数据集，并开发了一个紧凑的VLM，该模型预训练于图像字幕数据集，并针对指令重写进行了微调。通过使用诸如BLEU、METEOR和ROUGE等自然语言生成评估指标及语义解析分析，实验结果表明，即使是最小量化版本的模型（存储占用<500MB），也能实现有效的指令重写，从而推动具有隐私保护的多模态AI应用的发展。 

---
# Harnessing PDF Data for Improving Japanese Large Multimodal Models 

**Title (ZH)**: 利用PDF数据提升日语大规模多模态模型性能 

**Authors**: Jeonghun Baek, Akiko Aizawa, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2502.14778)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated strong performance in English, but their effectiveness in Japanese remains limited due to the lack of high-quality training data. Current Japanese LMMs often rely on translated English datasets, restricting their ability to capture Japan-specific cultural knowledge. To address this, we explore the potential of Japanese PDF data as a training resource, an area that remains largely underutilized. We introduce a fully automated pipeline that leverages pretrained models to extract image-text pairs from PDFs through layout analysis, OCR, and vision-language pairing, removing the need for manual annotation. Additionally, we construct instruction data from extracted image-text pairs to enrich the training data. To evaluate the effectiveness of PDF-derived data, we train Japanese LMMs and assess their performance on the Japanese LMM Benchmark. Our results demonstrate substantial improvements, with performance gains ranging from 3.9% to 13.8% on Heron-Bench. Further analysis highlights the impact of PDF-derived data on various factors, such as model size and language models, reinforcing its value as a multimodal resource for Japanese LMMs. We plan to make the source code and data publicly available upon acceptance. 

**Abstract (ZH)**: 大型多模态模型（LMMs）在英语中表现出了强大的能力，但在日语中的有效性仍然受到限制，主要原因是缺乏高质量的训练数据。当前的日语LMMs往往依赖于翻译自英语的数据集，这限制了它们捕捉日本特定文化知识的能力。为了解决这一问题，我们探索了利用日语PDF数据作为训练资源的潜力，这是一个尚未充分利用的领域。我们提出了一种全自动的工作流，利用预训练模型通过布局分析、OCR和视觉-语言配对来提取PDF中的图像-文本对，从而省去了手动注释的需要。此外，我们从提取的图像-文本对中构建了指令数据，以丰富训练数据。为了评估PDF数据的有效性，我们训练了日语LMMs，并在日语LMM基准测试上对其性能进行了评估。我们的结果显示，在Heron-Bench上的性能增幅从3.9%到13.8%不等。进一步的分析突出了PDF数据对不同因素的影响，如模型规模和语言模型，进一步证实了其作为日语LMMs的多模态资源的价值。我们计划在文章被接受后公开源代码和数据。 

---
# HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States 

**Title (ZH)**: HiddenDetect：通过监控隐藏状态检测针对大规模视觉-语言模型的越权攻击 

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14744)  

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

将额外模态整合到大型视觉语言模型（LVLMs）中会增加其对安全风险的敏感性，如脱困攻击（jailbreak attacks），相较于仅语言模型的版本。目前的研究主要集中在事后对齐技术上，但LVLMs内的内在安全机制尚未得到充分探索。在本研究中，我们探讨了LVLMs在推理过程中是否固有地在内部激活中编码了与安全性相关的信息。我们发现，当处理不安全提示时，LVLMs表现出不同的激活模式，这些模式可以被利用来检测和缓解恶意输入，而无需进行大量的微调。基于这一发现，我们提出了一种名为HiddenDetect的新颖无微调框架，该框架利用模型内部激活来增强安全性。实验结果显示，HiddenDetect在检测LVLM对抗脱困攻击方面超过了当前最先进的方法。通过利用内在的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，以增强LVLM在多模态威胁下的鲁棒性。我们的代码将在此网址公开发布：https://... 

---
# NAVIG: Natural Language-guided Analysis with Vision Language Models for Image Geo-localization 

**Title (ZH)**: NAVIG：基于自然语言指导的视觉语言模型在图像地理定位中的分析方法 

**Authors**: Zheyuan Zhang, Runze Li, Tasnim Kabir, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.14638)  

**Abstract**: Image geo-localization is the task of predicting the specific location of an image and requires complex reasoning across visual, geographical, and cultural contexts. While prior Vision Language Models (VLMs) have the best accuracy at this task, there is a dearth of high-quality datasets and models for analytical reasoning. We first create NaviClues, a high-quality dataset derived from GeoGuessr, a popular geography game, to supply examples of expert reasoning from language. Using this dataset, we present Navig, a comprehensive image geo-localization framework integrating global and fine-grained image information. By reasoning with language, Navig reduces the average distance error by 14% compared to previous state-of-the-art models while requiring fewer than 1000 training samples. Our dataset and code are available at this https URL. 

**Abstract (ZH)**: 图像地理定位任务是指预测图像的具体位置，这需要在视觉、地理和文化上下文中进行复杂的推理。虽然先前的视觉语言模型（VLMs）在这一任务上具有最高的准确性，但用于分析推理的高质量数据集和模型仍然稀缺。我们首先创建了一个名为NaviClues的数据集，该数据集源自流行的地缘猜谜游戏GeoGuessr，提供了语言专家推理的示例。使用此数据集，我们提出了Navig，一个整合全局和细粒度图像信息的完整图像地理定位框架。通过语言推理，Navig相比之前的顶级模型将平均距离误差降低了14%，同时只需要不到1000个训练样本。我们的数据集和代码已发布，可以通过以下链接访问：[此处替换为具体的URL链接]。 

---
# Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation 

**Title (ZH)**: 通过代码引导的合成多模态数据生成实现丰富的文本图像理解扩展示lastname: 

**Authors**: Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, Christopher Clark  

**Link**: [PDF](https://arxiv.org/pdf/2502.14846)  

**Abstract**: Reasoning about images with rich text, such as charts and documents, is a critical application of vision-language models (VLMs). However, VLMs often struggle in these domains due to the scarcity of diverse text-rich vision-language data. To address this challenge, we present CoSyn, a framework that leverages the coding capabilities of text-only large language models (LLMs) to automatically create synthetic text-rich multimodal data. Given input text describing a target domain (e.g., "nutrition fact labels"), CoSyn prompts an LLM to generate code (Python, HTML, LaTeX, etc.) for rendering synthetic images. With the underlying code as textual representations of the synthetic images, CoSyn can generate high-quality instruction-tuning data, again relying on a text-only LLM. Using CoSyn, we constructed a dataset comprising 400K images and 2.7M rows of vision-language instruction-tuning data. Comprehensive experiments on seven benchmarks demonstrate that models trained on our synthetic data achieve state-of-the-art performance among competitive open-source models, including Llama 3.2, and surpass proprietary models such as GPT-4V and Gemini 1.5 Flash. Furthermore, CoSyn can produce synthetic pointing data, enabling VLMs to ground information within input images, showcasing its potential for developing multimodal agents capable of acting in real-world environments. 

**Abstract (ZH)**: 利用丰富的文本（如图表和文档）对图像进行推理是视觉-语言模型（VLMs）的一个关键应用。然而，由于这类领域中多样化文本丰富视觉数据的稀缺性，VLMs 往往难以应对这些挑战。为了解决这一问题，我们提出了一种名为 CoSyn 的框架，该框架利用仅文本大型语言模型（LLMs）的编码能力，自动创建合成的文本丰富的多模态数据。给定描述目标领域的输入文本（例如，“营养成分标签”），CoSyn 可以促使 LLM 生成用于渲染合成图像的代码（如 Python、HTML、LaTeX 等）。通过将底层代码作为合成图像的文本表示，CoSyn 可以生成高质量的指令调优数据，依赖于仅文本的 LLM。使用 CoSyn，我们构建了一个包含 40 万张图像和 270 万行视觉-语言指令调优数据的数据集。在七个基准上的全面实验表明，在我们的合成数据上训练的模型在竞争性的开源模型（包括 Llama 3.2）中达到了最先进的性能，甚至超过了专有模型 GPT-4V 和 Gemini 1.5 Flash。此外，CoSyn 还可以生成合成的指示数据，使 VLMs 能够在输入图像中定位信息，展示了其在开发能够实现在真实世界环境中行动的多模态代理方面的潜力。 

---
# Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework 

**Title (ZH)**: 基于图表型文档问答生成框架的多模态RAG基准测试 

**Authors**: Yuming Yang, Jiang Zhong, Li Jin, Jingwang Huang, Jingpeng Gao, Qing Liu, Yang Bai, Jingyuan Zhang, Rui Jiang, Kaiwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14864)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) enhances reasoning capabilities by integrating external knowledge. However, existing benchmarks primarily focus on simple image-text interactions, overlooking complex visual formats like charts that are prevalent in real-world applications. In this work, we introduce a novel task, Chart-based MRAG, to address this limitation. To semi-automatically generate high-quality evaluation samples, we propose CHARt-based document question-answering GEneration (CHARGE), a framework that produces evaluation data through structured keypoint extraction, crossmodal verification, and keypoint-based generation. By combining CHARGE with expert validation, we construct Chart-MRAG Bench, a comprehensive benchmark for chart-based MRAG evaluation, featuring 4,738 question-answering pairs across 8 domains from real-world documents. Our evaluation reveals three critical limitations in current approaches: (1) unified multimodal embedding retrieval methods struggles in chart-based scenarios, (2) even with ground-truth retrieval, state-of-the-art MLLMs achieve only 58.19% Correctness and 73.87% Coverage scores, and (3) MLLMs demonstrate consistent text-over-visual modality bias during Chart-based MRAG reasoning. The CHARGE and Chart-MRAG Bench are released at this https URL. 

**Abstract (ZH)**: 多模态检索增强生成（MRAG）通过集成外部知识增强了推理能力。然而，现有的基准主要关注简单的图像-文本交互，忽视了在实际应用中常见的复杂视觉格式，如图表。为此，我们在本文中提出了一项新颖的任务，即基于图表的MRAG，以解决这一局限性。为了半自动地生成高质量的评估样本，我们提出了基于图表的文档问答生成（CHARGE）框架，该框架通过结构化关键点提取、跨模态验证和关键点驱动的生成来生成评估数据。通过将CHARGE与专家验证相结合，我们构建了基于图表的MRAG基准（Chart-MRAG Bench），该基准涵盖了来自8个实际文档领域的4,738个问答对。我们的评估揭示了当前方法存在三个关键不足：（1）统一的多模态嵌入检索方法在基于图表的场景中表现不佳；（2）即使有 ground-truth 检索，最先进的大规模语言模型也只能达到58.19%的正确率和73.87%的覆盖范围；（3）大规模语言模型在基于图表的MRAG推理过程中表现出持续的文本占优视觉模式偏见。CHARGE 和 Chart-MRAG Bench 已在此 HTTPS 地址发布：[https://example.com/CHARGE-and-Chart-MRAG-Bench](https://example.com/CHARGE-and-Chart-MRAG-Bench) 

---
# Exploring Advanced Techniques for Visual Question Answering: A Comprehensive Comparison 

**Title (ZH)**: 探索视觉问答领域的高级技术：全面对比分析 

**Authors**: Aiswarya Baby, Tintu Thankom Koshy  

**Link**: [PDF](https://arxiv.org/pdf/2502.14827)  

**Abstract**: Visual Question Answering (VQA) has emerged as a pivotal task in the intersection of computer vision and natural language processing, requiring models to understand and reason about visual content in response to natural language questions. Analyzing VQA datasets is essential for developing robust models that can handle the complexities of multimodal reasoning. Several approaches have been developed to examine these datasets, each offering distinct perspectives on question diversity, answer distribution, and visual-textual correlations. Despite significant progress, existing VQA models face challenges related to dataset bias, limited model complexity, commonsense reasoning gaps, rigid evaluation methods, and generalization to real world scenarios. This paper presents a comprehensive comparative study of five advanced VQA models: ABC-CNN, KICNLE, Masked Vision and Language Modeling, BLIP-2, and OFA, each employing distinct methodologies to address these challenges. 

**Abstract (ZH)**: 视觉问答（VQA）已成为计算机视觉与自然语言处理交叉领域的一个关键任务，要求模型能够理解并根据自然语言问题对视觉内容进行推理。分析VQA数据集是开发鲁棒模型、处理多模态推理复杂性的重要步骤。已有多种方法用于研究这些数据集，每种方法都提供了对问题多样性、答案分布及视觉-文本关联的不同视角。尽管取得了显著进展，现有的VQA模型仍面临数据集偏差、模型复杂度有限、常识推理不足、僵化的评估方法以及在现实场景中泛化能力差等挑战。本文对五种先进的VQA模型——ABC-CNN、KICNLE、遮蔽视觉与语言建模、BLIP-2和OFA——进行了全面的比较研究，每种模型采用独特的研究方法来应对上述挑战。 

---
# FetalCLIP: A Visual-Language Foundation Model for Fetal Ultrasound Image Analysis 

**Title (ZH)**: 胎儿CLIP：一种用于胎儿超声图像分析的视觉-语言基础模型 

**Authors**: Fadillah Maani, Numan Saeed, Tausifa Saleem, Zaid Farooq, Hussain Alasmawi, Werner Diehl, Ameera Mohammad, Gareth Waring, Saudabi Valappi, Leanne Bricker, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2502.14807)  

**Abstract**: Foundation models are becoming increasingly effective in the medical domain, offering pre-trained models on large datasets that can be readily adapted for downstream tasks. Despite progress, fetal ultrasound images remain a challenging domain for foundation models due to their inherent complexity, often requiring substantial additional training and facing limitations due to the scarcity of paired multimodal data. To overcome these challenges, here we introduce FetalCLIP, a vision-language foundation model capable of generating universal representation of fetal ultrasound images. FetalCLIP was pre-trained using a multimodal learning approach on a diverse dataset of 210,035 fetal ultrasound images paired with text. This represents the largest paired dataset of its kind used for foundation model development to date. This unique training approach allows FetalCLIP to effectively learn the intricate anatomical features present in fetal ultrasound images, resulting in robust representations that can be used for a variety of downstream applications. In extensive benchmarking across a range of key fetal ultrasound applications, including classification, gestational age estimation, congenital heart defect (CHD) detection, and fetal structure segmentation, FetalCLIP outperformed all baselines while demonstrating remarkable generalizability and strong performance even with limited labeled data. We plan to release the FetalCLIP model publicly for the benefit of the broader scientific community. 

**Abstract (ZH)**: 基础模型在医疗领域中的应用越来越有效，它们可以通过大规模数据集进行预训练，并且可以轻松适应下游任务。尽管取得了进展，但由于胎儿超声图像本身复杂性较高，基础模型在这一领域依然面临诸多挑战，通常需要大量的额外训练，并且受限于配对多模态数据的稀缺性。为了解决这些挑战，我们在此介绍了FetalCLIP，这是一种能够生成胎儿超声图像通用表示的基础视觉-语言模型。FetalCLIP通过在包含210,035张胎儿超声图像和相应文本信息的多元化数据集上采用多模态学习方法进行预训练。这代表了迄今为止用于基础模型开发的最大规模的配对数据集。这种独特的训练方法使FetalCLIP能够有效地学习胎儿超声图像中存在的复杂解剖特征，从而产生稳健的表示，这些表示可以应用于多种下游应用。在包括分类、胎龄估计、先天性心脏病（CHD）检测和胎儿结构分割等一系列关键胎儿超声应用的广泛基准测试中，FetalCLIP在所有对照基线模型中表现最佳，并且展示了出色的泛化能力和在有限标记数据下依然强大的性能。我们计划将FetalCLIP模型公开发布，以惠及更广泛的科学界。 

---
# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

**Title (ZH)**: SigLIP 2：具备增强语义理解、定位能力和密集特征的多语言视觉-语言编码器 

**Authors**: Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2502.14786)  

**Abstract**: We introduce SigLIP 2, a family of new multilingual vision-language encoders that build on the success of the original SigLIP. In this second iteration, we extend the original image-text training objective with several prior, independently developed techniques into a unified recipe -- this includes captioning-based pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation. With these changes, SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs). Furthermore, the new training recipe leads to significant improvements on localization and dense prediction tasks. We also train variants which support multiple resolutions and preserve the input's native aspect ratio. Finally, we train on a more diverse data-mixture that includes de-biasing techniques, leading to much better multilingual understanding and improved fairness. To allow users to trade off inference cost with performance, we release model checkpoints at four sizes: ViT-B (86M), L (303M), So400m (400M), and g (1B). 

**Abstract (ZH)**: 我们介绍了一种新的多语言视觉-语言编码器家族——SigLIP 2，该家族建立在原始SigLIP的成功基础上。在这一改进版本中，我们将原始的图像-文本训练目标与几种先前独立开发的技术统一成一个综合配方——这包括基于描述符的预训练、自监督损失（自我蒸馏、掩码预测）以及在线数据管理。通过这些改进，SigLIP 2 模型在所有模型规模的核心能力上都优于其原始的SigLIP模型，包括零样本分类、图像-文本检索以及用于视觉语言模型（VLM）提取视觉表示的迁移性能。此外，新的训练配方在定位和密集预测任务上带来了显著改进。我们还训练了支持多种分辨率并且保留输入原始长宽比的变体。最后，我们使用更为多样化的数据混 packageName 能，包括去偏见技术，从而提高了多语言理解和公平性。为了允许用户在推理成本与性能之间进行权衡，我们发布了四种大小的模型检查点：ViT-B（86M）、L（303M）、S（400M）和G（1B）。 

---
# WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models 

**Title (ZH)**: WavRAG：结合音频的检索增强生成模型用于口语对话系统 

**Authors**: Yifu Chen, Shengpeng Ji, Haoxiao Wang, Ziqing Wang, Siyu Chen, Jinzheng He, Jin Xu, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14727)  

**Abstract**: Retrieval Augmented Generation (RAG) has gained widespread adoption owing to its capacity to empower large language models (LLMs) to integrate external knowledge. However, existing RAG frameworks are primarily designed for text-based LLMs and rely on Automatic Speech Recognition to process speech input, which discards crucial audio information, risks transcription errors, and increases computational overhead. Therefore, we introduce WavRAG, the first retrieval augmented generation framework with native, end-to-end audio support. WavRAG offers two key features: 1) Bypassing ASR, WavRAG directly processes raw audio for both embedding and retrieval. 2) WavRAG integrates audio and text into a unified knowledge representation. Specifically, we propose the WavRetriever to facilitate the retrieval from a text-audio hybrid knowledge base, and further enhance the in-context capabilities of spoken dialogue models through the integration of chain-of-thought reasoning. In comparison to state-of-the-art ASR-Text RAG pipelines, WavRAG achieves comparable retrieval performance while delivering a 10x acceleration. Furthermore, WavRAG's unique text-audio hybrid retrieval capability extends the boundaries of RAG to the audio modality. 

**Abstract (ZH)**: 检索增强生成（RAG）由于其增强大型语言模型（LLMs）整合外部知识的能力而被广泛应用。然而，现有的RAG框架主要针对基于文本的LLMs，并依赖自动语音识别（ASR）处理语音输入，这会丢弃重要的音频信息、增加转录错误的风险以及增加计算量。因此，我们引入了WavRAG，这是首个提供原生端到端语音支持的检索增强生成框架。WavRAG具备两项关键功能：1）绕过ASR，WavRAG直接处理原始音频进行嵌入和检索；2）WavRAG将音频和文本统一到一个知识表示中。具体而言，我们提出了WavRetriever以从文本-音频混合知识库中进行检索，并通过将链式推理集成到其中，进一步增强了口语对话模型的语境能力。与最新的ASR-Text RAG流水线相比，WavRAG实现了相当的检索性能，同时加速了10倍。此外，WavRAG独特的声音-文本混合检索能力将RAG的应用范围扩展到了音频模态。 

---
# PLPHP: Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models 

**Title (ZH)**: PLPHP：高效大规模视觉语言模型的逐层逐头 vision token 剪枝 

**Authors**: Yu Meng, Kaiyuan Li, Chenran Huang, Chen Gao, Xinlei Chen, Yong Li, Xiaoping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14504)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a range of multimodal tasks. However, their inference efficiency is constrained by the large number of visual tokens processed during decoding. To address this challenge, we propose Per-Layer Per-Head Vision Token Pruning (PLPHP), a two-level fine-grained pruning method including Layer-Level Retention Rate Allocation and Head-Level Vision Token Pruning. Motivated by the Vision Token Re-attention phenomenon across decoder layers, we dynamically adjust token retention rates layer by layer. Layers that exhibit stronger attention to visual information preserve more vision tokens, while layers with lower vision attention are aggressively pruned. Furthermore, PLPHP applies pruning at the attention head level, enabling different heads within the same layer to independently retain critical context. Experiments on multiple benchmarks demonstrate that PLPHP delivers an 18% faster decoding speed and reduces the Key-Value Cache (KV Cache) size by over 50%, all at the cost of 0.46% average performance drop, while also achieving notable performance improvements in multi-image tasks. These results highlight the effectiveness of fine-grained token pruning and contribute to advancing the efficiency and scalability of LVLMs. Our source code will be made publicly available. 

**Abstract (ZH)**: 大型多模态模型（Large Vision-Language Models, LVLMs）在多种多模态任务中展现出了显著的能力。然而，它们的推理效率受到解码过程中大量视觉令牌处理的限制。为了解决这一挑战，我们提出了逐层逐头视觉令牌剪枝（Per-Layer Per-Head Vision Token Pruning，PLPHP），这是一种两级细化剪枝方法，包括逐层保留率分配（Layer-Level Retention Rate Allocation）和逐头视觉令牌剪枝（Head-Level Vision Token Pruning）。受到解码层间视觉令牌再注意力现象的启发，我们逐层动态调整令牌保留率。表现出较强视觉信息注意力的层保留更多的视觉令牌，而视觉注意力较低的层则被激进地剪枝。此外，PLPHP 在注意力头层进行剪枝，使同一层内的不同头可以独立保留关键上下文。在多个基准测试上的实验结果表明，PLPHP 可以实现 18% 的更快解码速度，并且将关键值缓存（Key-Value Cache, KV Cache）的大小减少超过 50%，性能平均下降 0.46%，而在多图任务上也取得了显著的性能提升。这些结果突显了细致粒度令牌剪枝的有效性，并有助于提升 LVLMs 的效率和可扩展性。我们的源代码将会公开提供。 

---
# Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation 

**Title (ZH)**: 将以下论文标题或内容翻译成中文，符合学术规范：

Mem2Ego：通过全局到自车记忆增强的视觉-语言模型在长时 horizon 身份绑定导航中的应用 

**Authors**: Lingfeng Zhang, Yuecheng Liu, Zhanguang Zhang, Matin Aghaei, Yaochen Hu, Hongjian Gu, Mohammad Ali Alomrani, David Gamaliel Arcos Bravo, Raika Karimi, Atia Hamidizadeh, Haoping Xu, Guowei Huang, Zhanpeng Zhang, Tongtong Cao, Weichao Qiu, Xingyue Quan, Jianye Hao, Yuzheng Zhuang, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14254)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have made them powerful tools in embodied navigation, enabling agents to leverage commonsense and spatial reasoning for efficient exploration in unfamiliar environments. Existing LLM-based approaches convert global memory, such as semantic or topological maps, into language descriptions to guide navigation. While this improves efficiency and reduces redundant exploration, the loss of geometric information in language-based representations hinders spatial reasoning, especially in intricate environments. To address this, VLM-based approaches directly process ego-centric visual inputs to select optimal directions for exploration. However, relying solely on a first-person perspective makes navigation a partially observed decision-making problem, leading to suboptimal decisions in complex environments. In this paper, we present a novel vision-language model (VLM)-based navigation framework that addresses these challenges by adaptively retrieving task-relevant cues from a global memory module and integrating them with the agent's egocentric observations. By dynamically aligning global contextual information with local perception, our approach enhances spatial reasoning and decision-making in long-horizon tasks. Experimental results demonstrate that the proposed method surpasses previous state-of-the-art approaches in object navigation tasks, providing a more effective and scalable solution for embodied navigation. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）和视觉-语言模型（VLMs）的最新进展使其成为有潜力的工具，能够使代理利用常识和空间推理在未知环境中进行高效探索。现有基于LLM的方法将全局记忆（如语义或拓扑地图）转化为语言描述来指导导航。虽然这种方法提高了效率并减少了重复探索的次数，但基于语言的表示形式损失了几何信息，这在复杂的环境中会阻碍空间推理。为了解决这一问题，基于VLM的方法直接处理以自我为中心的视觉输入，选择探索的最佳方向。然而，仅依赖第一人称视角使导航成为部分观察的决策问题，在复杂环境中容易导致不 optimal 的决策。在本文中，我们提出了一种新颖的基于VLM的导航框架，通过自适应地从全局记忆模块中检索任务相关信息，并与代理的以自我为中心的感知进行整合，以解决这个问题。通过动态对齐全局上下文信息与局部感知，我们的方法增强了在长期任务中的空间推理和决策能力。实验结果表明，提出的方法在物体导航任务中超过了之前的先驱方法，为有代理导航提供了更有效和可扩展的解决方案。 

---
# SleepGMUformer: A gated multimodal temporal neural network for sleep staging 

**Title (ZH)**: 睡眠GMUformer：一种门控多模态时间神经网络方法用于睡眠分期 

**Authors**: Chenjun Zhao, Xuesen Niu, Xinglin Yu, Long Chen, Na Lv, Huiyu Zhou, Aite Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14227)  

**Abstract**: Sleep staging is a key method for assessing sleep quality and diagnosing sleep disorders. However, current deep learning methods face challenges: 1) postfusion techniques ignore the varying contributions of different modalities; 2) unprocessed sleep data can interfere with frequency-domain information. To tackle these issues, this paper proposes a gated multimodal temporal neural network for multidomain sleep data, including heart rate, motion, steps, EEG (Fpz-Cz, Pz-Oz), and EOG from WristHR-Motion-Sleep and SleepEDF-78. The model integrates: 1) a pre-processing module for feature alignment, missing value handling, and EEG de-trending; 2) a feature extraction module for complex sleep features in the time dimension; and 3) a dynamic fusion module for real-time modality this http URL show classification accuracies of 85.03% on SleepEDF-78 and 94.54% on WristHR-Motion-Sleep datasets. The model handles heterogeneous datasets and outperforms state-of-the-art models by 1.00%-4.00%. 

**Abstract (ZH)**: 睡眠分期是评估睡眠质量和诊断睡眠障碍的关键方法。然而，当前的深度学习方法面临着一些挑战：1）后融合技术忽视了不同模态的不同贡献比例；2）未经处理的睡眠数据会影响频域信息。为了解决这些问题，本文提出了一种门控多模态时序神经网络，用于处理包括心率、运动、步数、EEG（Fpz-Cz，Pz-Oz）和EOG在内的多领域睡眠数据，数据来源为WristHR-Motion-Sleep和SleepEDF-78。该模型整合了：1）一个预处理模块用于特征对齐、缺失值处理和EEG消趋势；2）一个特征提取模块用于时间维度上的复杂睡眠特征；以及3）一个动态融合模块用于实时模态融合。实验结果显示，该模型在SleepEDF-78数据集上的分类准确率为85.03%，在WristHR-Motion-Sleep数据集上的分类准确率为94.54%。该模型能够处理异质性数据集，并在多项基准模型上表现出1.00%-4.00%的性能提升。 

---
# Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models 

**Title (ZH)**: 多模态奖励基准：视觉语言模型奖励模型的全面评估 

**Authors**: Michihiro Yasunaga, Luke Zettlemoyer, Marjan Ghazvininejad  

**Link**: [PDF](https://arxiv.org/pdf/2502.14191)  

**Abstract**: Reward models play an essential role in training vision-language models (VLMs) by assessing output quality to enable aligning with human preferences. Despite their importance, the research community lacks comprehensive open benchmarks for evaluating multimodal reward models in VLMs. To address this gap, we introduce Multimodal RewardBench, an expert-annotated benchmark covering six domains: general correctness, preference, knowledge, reasoning, safety, and visual question-answering. Our dataset comprises 5,211 annotated (prompt, chosen response, rejected response) triplets collected from various VLMs. In evaluating a range of VLM judges, we find that even the top-performing models, Gemini 1.5 Pro and Claude 3.5 Sonnet, achieve only 72% overall accuracy. Notably, most models struggle in the reasoning and safety domains. These findings suggest that Multimodal RewardBench offers a challenging testbed for advancing reward model development across multiple domains. We release the benchmark at this https URL. 

**Abstract (ZH)**: 多模态奖励模型在训练视觉语言模型（VLMs）中扮演着至关重要的角色，它们通过评估输出质量来使模型与人类偏好保持一致。尽管如此，研究社区仍然缺乏对VLM中多模态奖励模型进行全面评估的公开基准。为了解决这一不足，我们引入了Multimodal RewardBench，这是一个专家注释的基准，涵盖了六个领域：通用正确性、偏好、知识、推理、安全性和视觉问答。我们的数据集包括来自多种VLM的5,211个注释过的（提示，选择的响应，拒绝的响应）三元组。在对多种VLM裁判的评估中，我们发现即使是表现最佳的模型Gemini 1.5 Pro和Claude 3.5 Sonnet，其整体准确率也只有72%。值得注意的是，大多数模型在推理和安全性领域表现不佳。这些发现表明，Multimodal RewardBench 为跨多个领域的奖励模型发展提供了具有挑战性的测试平台。我们已在此链接处发布了此基准：[此处链接]。 

---
# Object-centric Binding in Contrastive Language-Image Pretraining 

**Title (ZH)**: 对比语言-图像预训练中的对象中心绑定 

**Authors**: Rim Assouel, Pietro Astolfi, Florian Bordes, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2502.14113)  

**Abstract**: Recent advances in vision language models (VLM) have been driven by contrastive models such as CLIP, which learn to associate visual information with their corresponding text descriptions. However, these models have limitations in understanding complex compositional scenes involving multiple objects and their spatial relationships. To address these challenges, we propose a novel approach that diverges from commonly used strategies, which rely on the design of hard-negative augmentations. Instead, our work focuses on integrating inductive biases into pre-trained CLIP-like models to improve their compositional understanding without using any additional hard-negatives. To that end, we introduce a binding module that connects a scene graph, derived from a text description, with a slot-structured image representation, facilitating a structured similarity assessment between the two modalities. We also leverage relationships as text-conditioned visual constraints, thereby capturing the intricate interactions between objects and their contextual relationships more effectively. Our resulting model not only enhances the performance of CLIP-based models in multi-object compositional understanding but also paves the way towards more accurate and sample-efficient image-text matching of complex scenes. 

**Abstract (ZH)**: 近年来，视觉语言模型（VLM）的进步主要得益于对比模型，如CLIP，这些模型能够学习将视觉信息与其对应的文本描述关联起来。然而，这些模型在理解涉及多个物体及其空间关系的复杂组合场景方面存在局限性。为了解决这些挑战，我们提出了一种新颖的方法，该方法与依赖于设计负样本增强的常见策略不同。我们的工作集中在将归纳偏置整合到预训练的CLIP-like模型中，以提高它们的组合理解能力，而无需使用任何额外的负样本。为此，我们引入了一个绑定模块，该模块将从文本描述中获得的场景图与具有槽结构的图像表示连接起来，从而促进两种模态之间的结构化相似性评估。我们还利用关系作为文本条件下的视觉约束，从而更有效地捕捉物体及其上下文关系之间的复杂交互。通过这种方法，我们的模型不仅提高了基于CLIP的模型在多物体组合理解方面的性能，还为更准确和样本有效的复杂场景中的图像-文本匹配铺平了道路。 

---
# Gesture-Aware Zero-Shot Speech Recognition for Patients with Language Disorders 

**Title (ZH)**: 患有语言障碍患者的无监督手势引导语音识别 

**Authors**: Seungbae Kim, Daeun Lee, Brielle Stark, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.13983)  

**Abstract**: Individuals with language disorders often face significant communication challenges due to their limited language processing and comprehension abilities, which also affect their interactions with voice-assisted systems that mostly rely on Automatic Speech Recognition (ASR). Despite advancements in ASR that address disfluencies, there has been little attention on integrating non-verbal communication methods, such as gestures, which individuals with language disorders substantially rely on to supplement their communication. Recognizing the need to interpret the latent meanings of visual information not captured by speech alone, we propose a gesture-aware ASR system utilizing a multimodal large language model with zero-shot learning for individuals with speech impairments. Our experiment results and analyses show that including gesture information significantly enhances semantic understanding. This study can help develop effective communication technologies, specifically designed to meet the unique needs of individuals with language impairments. 

**Abstract (ZH)**: 语言障碍个体由于其有限的语言处理和理解能力常常面临重大的沟通挑战，这也影响了他们与主要依赖自动语音识别（ASR）的语音辅助系统的互动。尽管在ASR方面已经取得了进步以解决口吃等问题，但尚未过多关注整合非言语沟通方法（如手势），这些方法对于语言障碍个体补充沟通至关重要。为了解释仅靠语音无法捕捉到的潜在视觉信息含义的需要，我们提出了一种利用多模态大型语言模型并结合零样本学习的手势感知ASR系统，以满足言语障碍个体的特殊需求。实验结果和分析表明，纳入手势信息显著提高了语义理解。本研究有助于开发有效的沟通技术，特别是针对语言障碍个体的独特需求进行设计。 

---
