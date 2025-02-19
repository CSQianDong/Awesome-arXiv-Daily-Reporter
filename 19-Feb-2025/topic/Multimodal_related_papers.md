# Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection 

**Title (ZH)**: 改进的大型多模态模型微调方法在侮辱性 meme 识别中的应用 

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2502.13061)  

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While large multimodal models have shown strong generalization across various tasks, they exhibit poor generalization to hateful meme detection due to the dynamic nature of memes tied to emerging social trends and breaking news. Recent work further highlights the limitations of conventional supervised fine-tuning for large multimodal models in this context. To address these challenges, we propose Large Multimodal Model Retrieval-Guided Contrastive Learning (LMM-RGCL), a novel two-stage fine-tuning framework designed to improve both in-domain accuracy and cross-domain generalization. Experimental results on six widely used meme classification datasets demonstrate that LMM-RGCL achieves state-of-the-art performance, outperforming agent-based systems such as VPD-PALI-X-55B. Furthermore, our method effectively generalizes to out-of-domain memes under low-resource settings, surpassing models like GPT-4o. 

**Abstract (ZH)**: 仇恨 meme 已成为互联网上的一个重要关切，需要建立稳健的自动检测系统。虽然大型多模态模型在各种任务中表现出较强的泛化能力，但它们在仇恨 meme 检测方面表现较差，因为 meme 的动态性质使得它们容易受到新兴社会趋势和突发新闻的影响。近期的工作进一步强调了在这一背景下，传统的监督细调方法对大型多模态模型的局限性。为应对这些挑战，我们提出了大型多模态模型检索引导对比学习 (LMM-RGCL)，这是一种新颖的两阶段细调框架，旨在提高领域内准确性和跨领域泛化能力。在六个广泛使用的 meme 分类数据集上的实验结果表明，LMM-RGCL 达到了最先进的性能，超越了基于代理系统的方法，如 VPD-PALI-X-55B。此外，在低资源环境下，我们的方法还能够有效地泛化到领域外的 meme，超越了如 GPT-4o 等模型。 

---
# SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models 

**Title (ZH)**: SimpleVQA：多模态事实性评估方法用于多模态大型语言模型 

**Authors**: Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, Yutao Zeng, Zhoufutu Wen, Ke Jin, Baorui Wang, Weixiao Zhou, Yunhong Lu, Tongliang Li, Wenhao Huang, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13059)  

**Abstract**: The increasing application of multi-modal large language models (MLLMs) across various sectors have spotlighted the essence of their output reliability and accuracy, particularly their ability to produce content grounded in factual information (e.g. common and domain-specific knowledge). In this work, we introduce SimpleVQA, the first comprehensive multi-modal benchmark to evaluate the factuality ability of MLLMs to answer natural language short questions. SimpleVQA is characterized by six key features: it covers multiple tasks and multiple scenarios, ensures high quality and challenging queries, maintains static and timeless reference answers, and is straightforward to evaluate. Our approach involves categorizing visual question-answering items into 9 different tasks around objective events or common knowledge and situating these within 9 topics. Rigorous quality control processes are implemented to guarantee high-quality, concise, and clear answers, facilitating evaluation with minimal variance via an LLM-as-a-judge scoring system. Using SimpleVQA, we perform a comprehensive assessment of leading 18 MLLMs and 8 text-only LLMs, delving into their image comprehension and text generation abilities by identifying and analyzing error cases. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在各个领域的广泛应用凸显了其输出可靠性和准确性的本质，尤其是它们生成基于事实信息的内容（包括通用和领域特定知识）的能力。本文介绍了SimpleVQA，这是首个全面评估MLLMs事实准确性能力的多模态基准测试，通过自然语言简短问题来评估其回答能力。SimpleVQA具有六大关键特征：覆盖多个任务和场景、确保高质量和具有挑战性的查询、保持静态和永恒参考答案以及易于评估。我们的方法是将视觉问答项目分类为9个与客观事件或常见知识相关的任务，并将其置于9个主题之下。实施了严格的质量控制流程，以保证高质量、简洁和清晰的答案，通过LLM作为法官的评分系统实现最小偏差的评估。使用SimpleVQA，我们对18个领先的MLLMs和8个仅基于文本的LLM进行了全面评估，深入探讨了它们的图像理解和文本生成能力，并对错误案例进行了识别和分析。 

---
# AEIA-MN: Evaluating the Robustness of Multimodal LLM-Powered Mobile Agents Against Active Environmental Injection Attacks 

**Title (ZH)**: AEIA-MN：评估多模态LLM驱动的移动代理在活跃环境注入攻击下的鲁棒性 

**Authors**: Yurun Chen, Xueyu Hu, Keting Yin, Juncheng Li, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13053)  

**Abstract**: As researchers continuously optimize AI agents to perform tasks more effectively within operating systems, they often neglect to address the critical need for enabling these agents to identify "impostors" within the system. Through an analysis of the agents' operating environment, we identified a potential threat: attackers can disguise their attack methods as environmental elements, injecting active disturbances into the agents' execution process, thereby disrupting their decision-making. We define this type of attack as Active Environment Injection Attack (AEIA). Based on this, we propose AEIA-MN, an active environment injection attack scheme that exploits interaction vulnerabilities in the mobile operating system to evaluate the robustness of MLLM-based agents against such threats. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% in the AndroidWorld benchmark. 

**Abstract (ZH)**: 随着研究者不断优化AI代理以在操作系统中更有效地执行任务，他们常常忽视了使这些代理能够识别系统中的“冒充者”的关键需求。通过对代理的操作环境进行分析，我们发现了一个潜在威胁：攻击者可以伪装其攻击手段为环境元素，并向代理的执行过程注入活跃的干扰，从而扰乱其决策过程。我们定义此类攻击为活跃环境注入攻击（AEIA）。基于此，我们提出了一种AEIA-MN方案，该方案利用移动操作系统中的交互漏洞来评估基于MLLM的代理在面对此类威胁时的鲁棒性。实验结果显示，即使是先进的MLLM，在AndroidWorld基准测试中也高度容易受到此类攻击的影响，攻击成功率达到93%。 

---
# MVL-SIB: A Massively Multilingual Vision-Language Benchmark for Cross-Modal Topical Matching 

**Title (ZH)**: MVL-SIB：一种大规模多语言跨模态主题匹配基准 

**Authors**: Fabian David Schmidt, Florian Schneider, Chris Biemann, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2502.12852)  

**Abstract**: Existing multilingual vision-language (VL) benchmarks often only cover a handful of languages. Consequently, evaluations of large vision-language models (LVLMs) predominantly target high-resource languages, underscoring the need for evaluation data for low-resource languages. To address this limitation, we introduce MVL-SIB, a massively multilingual vision-language benchmark that evaluates both cross-modal and text-only topical matching across 205 languages -- over 100 more than the most multilingual existing VL benchmarks encompass. We then benchmark a range of of open-weight LVLMs together with GPT-4o(-mini) on MVL-SIB. Our results reveal that LVLMs struggle in cross-modal topic matching in lower-resource languages, performing no better than chance on languages like N'Koo. Our analysis further reveals that VL support in LVLMs declines disproportionately relative to textual support for lower-resource languages, as evidenced by comparison of cross-modal and text-only topical matching performance. We further observe that open-weight LVLMs do not benefit from representing a topic with more than one image, suggesting that these models are not yet fully effective at handling multi-image tasks. By correlating performance on MVL-SIB with other multilingual VL benchmarks, we highlight that MVL-SIB serves as a comprehensive probe of multilingual VL understanding in LVLMs. 

**Abstract (ZH)**: 现有的多语言视觉-语言（VL）基准通常仅涵盖少数几种语言。因此，对于大规模视觉-语言模型（LVLM）的评估大多集中在高资源语言上，这凸显了为低资源语言提供评估数据的必要性。为了解决这一局限性，我们引入了MVL-SIB，这是一个涵盖205种语言的巨量多语言视觉-语言基准，比现有最多样化的多语言VL基准多出逾100种语言，同时涵盖了跨模态和纯文本主题匹配。随后，我们对一系列开放参数的LVLM及GPT-4o(-mini)在MVL-SIB上的性能进行了评估。我们的结果显示，LVLM在低资源语言的跨模态主题匹配中表现糟糕，甚至在如N’Koo这样的语言上表现不如随机猜测。进一步的分析表明，与纯文本支持相比，LVLM对低资源语言的跨模态支持下降更为显著，这一点通过跨模态和纯文本主题匹配性能的对比可以体现。我们还观察到，开放参数的LVLM在使用多张图像表示一个主题时并未获益，这表明这些模型尚未完全有效地处理多图像任务。通过将MVL-SIB上的性能与其它多语言VL基准的相关性进行关联，我们强调MVL-SIB是评估LVLM多语言VL理解能力的一个全面探针。 

---
# Towards Text-Image Interleaved Retrieval 

**Title (ZH)**: 面向文本-图像交替检索的方向 

**Authors**: Xin Zhang, Ziqi Dai, Yongqi Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Jun Yu, Wenjie Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12799)  

**Abstract**: Current multimodal information retrieval studies mainly focus on single-image inputs, which limits real-world applications involving multiple images and text-image interleaved content. In this work, we introduce the text-image interleaved retrieval (TIIR) task, where the query and document are interleaved text-image sequences, and the model is required to understand the semantics from the interleaved context for effective retrieval. We construct a TIIR benchmark based on naturally interleaved wikiHow tutorials, where a specific pipeline is designed to generate interleaved queries. To explore the task, we adapt several off-the-shelf retrievers and build a dense baseline by interleaved multimodal large language model (MLLM). We then propose a novel Matryoshka Multimodal Embedder (MME), which compresses the number of visual tokens at different granularity, to address the challenge of excessive visual tokens in MLLM-based TIIR models. Experiments demonstrate that simple adaption of existing models does not consistently yield effective results. Our MME achieves significant improvements over the baseline by substantially fewer visual tokens. We provide extensive analysis and will release the dataset and code to facilitate future research. 

**Abstract (ZH)**: 当前的多模态信息检索研究主要集中在单张图像的输入上，这限制了涉及多张图像和图文交错内容的实际应用。在本文中，我们介绍了图文交错检索（TIIR，Text-Image Interleaved Retrieval）任务，其中查询和文档是交错的文本图像序列，模型需要从交错的上下文中理解语义以进行有效的检索。我们基于自然交错的wikiHow教程构建了一个TIIR基准数据集，并设计了一个管线生成交错查询。为了探索该任务，我们适应了几种现成的信息检索器，并通过交错多模态大语言模型（MLLM）构建了一个稠密基线。然后，我们提出了一种新颖的Matryoshka多模态嵌入器（MME），该嵌入器在不同粒度上压缩视觉词的数量，以解决基于MLLM的TIIR模型中视觉词过多的问题。实验表明，简单的模型适应并不总能获得有效结果。我们的MME通过显著减少视觉词的数量，在基线之上取得了显著改进。我们进行了详细分析，并将发布数据集和代码，以促进未来的研究。 

---
# Mind the Gap: Aligning the Brain with Language Models Requires a Nonlinear and Multimodal Approach 

**Title (ZH)**: 注意差距：将大脑与语言模型对齐需要非线性和多模态的方法 

**Authors**: Danny Dongyeop Han, Yunju Cho, Jiook Cha, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12771)  

**Abstract**: Self-supervised language and audio models effectively predict brain responses to speech. However, traditional prediction models rely on linear mappings from unimodal features, despite the complex integration of auditory signals with linguistic and semantic information across widespread brain networks during speech comprehension. Here, we introduce a nonlinear, multimodal prediction model that combines audio and linguistic features from pre-trained models (e.g., LLAMA, Whisper). Our approach achieves a 17.2% and 17.9% improvement in prediction performance (unnormalized and normalized correlation) over traditional unimodal linear models, as well as a 7.7% and 14.4% improvement, respectively, over prior state-of-the-art models. These improvements represent a major step towards future robust in-silico testing and improved decoding performance. They also reveal how auditory and semantic information are fused in motor, somatosensory, and higher-level semantic regions, aligning with existing neurolinguistic theories. Overall, our work highlights the often neglected potential of nonlinear and multimodal approaches to brain modeling, paving the way for future studies to embrace these strategies in naturalistic neurolinguistics research. 

**Abstract (ZH)**: 自监督的语言和音频模型能够有效预测对语音的脑响应。然而，传统预测模型依赖于从单模特征到线性映射的关系，尽管在言语理解过程中，听觉信号与语言和语义信息的综合跨越了大量的脑网络，是一种复杂的集成过程。在此，我们引入了一个非线性的多模态预测模型，该模型结合了预训练模型（如LLAMA、Whisper）的音频和语言特征。我们的方法在未标准化和标准化相关性上分别比传统的单模态线性模型提高了17.2%和17.9%，并比之前最先进的模型分别提高了7.7%和14.4%。这些改进标志着朝着未来稳健的计算机模拟测试和解码性能提升迈出的重要一步。它们还揭示了听觉和语义信息如何在运动、躯体感觉以及更高层次的语义区域融合，与现有的神经语言学理论相一致。总的来说，我们的研究突显了非线性和多模态方法在脑建模中的潜在价值，为未来研究在自然语言神经科学中的采用这些策略铺平了道路。 

---
# SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings 

**Title (ZH)**: SEA：通过合成嵌入实现多模态大型语言模型的低资源安全对齐 

**Authors**: Weikai Lu, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12562)  

**Abstract**: Multimodal Large Language Models (MLLMs) have serious security this http URL safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）存在严重的安全问题。为了增强MLLM的安全性，可以通过使用包含文本和其他模态数据的多模态数据集来进行安全对齐，但构建这些数据集的成本很高。现有的低资源安全对齐方法，如文本对齐，已被发现难以应对由其他模态所带来的安全风险。为此，我们提出了一种合成嵌入增强安全对齐（SEA）方法，通过梯度更新优化其他模态的嵌入，以扩展文本数据集。即使仅使用文本数据，也可以实现多模态安全对齐训练。在基于图像、视频和音频的MLLMs上的大量实验表明，SEA能够在单块RTX3090 GPU上于24秒内合成高质量的嵌入。SEA显著提高了MLLMs在面临其他模态带来的威胁时的安全性。为了评估由视频和音频引入的安全风险，我们还引入了一个新的基准测试VA-SafetyBench。在多个MLLMs上多次高成功率的攻击验证了其挑战性。我们的代码和数据将在此网址发布：[参考网址]。 

---
# REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark 

**Title (ZH)**: REAL-MM-RAG：一个现实世界多模态检索基准 

**Authors**: Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, Leonid Karlinsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.12342)  

**Abstract**: Accurate multi-modal document retrieval is crucial for Retrieval-Augmented Generation (RAG), yet existing benchmarks do not fully capture real-world challenges with their current design. We introduce REAL-MM-RAG, an automatically generated benchmark designed to address four key properties essential for real-world retrieval: (i) multi-modal documents, (ii) enhanced difficulty, (iii) Realistic-RAG queries and (iv) accurate labeling. Additionally, we propose a multi-difficulty-level scheme based on query rephrasing to evaluate models' semantic understanding beyond keyword matching. Our benchmark reveals significant model weaknesses, particularly in handling table-heavy documents and robustness to query rephrasing. To mitigate these shortcomings, we curate a rephrased training set and introduce a new finance-focused, table-heavy dataset. Fine-tuning on these datasets enables models to achieve state-of-the-art retrieval performance on REAL-MM-RAG benchmark. Our work offers a better way to evaluate and improve retrieval in multi-modal RAG systems while also providing training data and models that address current limitations. 

**Abstract (ZH)**: 准确的多模态文档检索对于检索增强生成（RAG）至关重要，但现有的基准测试在当前设计中未能充分捕捉到实际应用中的挑战。我们引入了REAL-MM-RAG，这是一个自动生成的基准测试，旨在解决四个对于实际检索至关重要的关键属性：（i）多模态文档，（ii）增强的难度，（iii）现实场景下的RAG查询，（iv）准确的标注。此外，我们提出了基于查询重述的多难度层次方案，以评估模型超越关键词匹配的语义理解能力。我们的基准测试揭示了模型在应对表格密集型文档和查询重述下的鲁棒性方面的显著弱点。为了缓解这些不足，我们精心挑选了一个重述训练集，并引入了一个专注于金融且表格密集型的新数据集。在这些数据集上进行微调使模型在REAL-MM-RAG基准测试上实现了最先进的检索性能。我们的工作为评估和改进多模态RAG系统的检索提供了更好的方法，同时也提供了应对当前局限性的训练数据和模型。 

---
# MatterChat: A Multi-Modal LLM for Material Science 

**Title (ZH)**: MatterChat：材料科学领域的多模态大规模语言模型 

**Authors**: Yingheng Tang, Wenbin Xu, Jie Cao, Jianzhu Ma, Weilu Gao, Steve Farrell, Benjamin Erichson, Michael W. Mahoney, Andy Nonaka, Zhi Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13107)  

**Abstract**: Understanding and predicting the properties of inorganic materials is crucial for accelerating advancements in materials science and driving applications in energy, electronics, and beyond. Integrating material structure data with language-based information through multi-modal large language models (LLMs) offers great potential to support these efforts by enhancing human-AI interaction. However, a key challenge lies in integrating atomic structures at full resolution into LLMs. In this work, we introduce MatterChat, a versatile structure-aware multi-modal LLM that unifies material structural data and textual inputs into a single cohesive model. MatterChat employs a bridging module to effectively align a pretrained machine learning interatomic potential with a pretrained LLM, reducing training costs and enhancing flexibility. Our results demonstrate that MatterChat significantly improves performance in material property prediction and human-AI interaction, surpassing general-purpose LLMs such as GPT-4. We also demonstrate its usefulness in applications such as more advanced scientific reasoning and step-by-step material synthesis. 

**Abstract (ZH)**: 理解并预测无机材料的性质对于加速材料科学的进步和推动能源、电子等领域应用至关重要。通过多模态大型语言模型（LLMs）整合材料结构数据与基于语言的信息，有助于增强人类与AI的交互，具有巨大的潜力。然而，将原子结构信息全面整合到LLMs中仍然是一个关键挑战。在此项研究中，我们介绍了MatterChat，这是一种多功能结构感知多模态LLM，能够将材料结构数据和文本输入统一到一个连贯的模型中。MatterChat通过引入一个连接模块来有效对接预训练的机器学习原子间势能模型与预训练的LLM，降低训练成本并增强灵活性。我们的研究结果表明，MatterChat显著提高了材料性质预测和人机交互的性能，超越了通用的LLM，如GPT-4。我们还展示了它在更高级的科学推理和逐步材料合成等应用中的实用价值。 

---
# You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations 

**Title (ZH)**: 你需要模拟以获得名声：利用多代理对话解决会议纪要稀缺性问题 

**Authors**: Frederic Kirstein, Muneeb Khan, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.13001)  

**Abstract**: Meeting summarization suffers from limited high-quality data, mainly due to privacy restrictions and expensive collection processes. We address this gap with FAME, a dataset of 500 meetings in English and 300 in German produced by MIMIC, our new multi-agent meeting synthesis framework that generates meeting transcripts on a given knowledge source by defining psychologically grounded participant profiles, outlining the conversation, and orchestrating a large language model (LLM) debate. A modular post-processing step refines these outputs, mitigating potential repetitiveness and overly formal tones, ensuring coherent, credible dialogues at scale. We also propose a psychologically grounded evaluation framework assessing naturalness, social behavior authenticity, and transcript difficulties. Human assessments show that FAME approximates real-meeting spontaneity (4.5/5 in naturalness), preserves speaker-centric challenges (3/5 in spoken language), and introduces richer information-oriented difficulty (4/5 in difficulty). These findings highlight that FAME is a good and scalable proxy for real-world meeting conditions. It enables new test scenarios for meeting summarization research and other conversation-centric applications in tasks requiring conversation data or simulating social scenarios under behavioral constraints. 

**Abstract (ZH)**: 会议总结因缺乏高质量数据而受限，主要原因是隐私限制和昂贵的收集过程。我们通过提出FAME数据集来填补这一空白，该数据集包含500个英语会议和300个德语会议，是由我们新开发的多智能体会议合成框架MIMIC生成的。MIMIC框架通过定义基于心理学原理的参与者角色、规划对话内容，并协调大规模语言模型（LLM）辩论来生成给定知识源的会议记录。一个模块化的后处理步骤进一步细化这些输出，减少了潜在的重复性和过于正式的语气，确保了大规模对话的连贯性和可信度。我们还提出了一种基于心理学的评估框架，评估自然性、社会行为的真实性以及记录的难度。人类评估结果显示，FAME在自然性（4.5/5）上接近真实的会议自发性，保留了以讲演者为中心的挑战（3/5在口语方面），并引入了更丰富的信息导向难点（4/5在难度上）。这些发现表明，FAME是一个良好的且可扩展的现实会议条件的代理。它为会议总结研究和需要对话数据或其他对话中心应用的任务提供了新的测试场景，特别是在行为约束下模拟社交场景时。 

---
# Magma: A Foundation Model for Multimodal AI Agents 

**Title (ZH)**: Magma：多模态AI代理的基石模型 

**Authors**: Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13130)  

**Abstract**: We present Magma, a foundation model that serves multimodal AI agentic tasks in both the digital and physical worlds. Magma is a significant extension of vision-language (VL) models in that it not only retains the VL understanding ability (verbal intelligence) of the latter, but is also equipped with the ability to plan and act in the visual-spatial world (spatial-temporal intelligence) and complete agentic tasks ranging from UI navigation to robot manipulation. To endow the agentic capabilities, Magma is pretrained on large amounts of heterogeneous datasets spanning from images, videos to robotics data, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set-of-Mark (SoM) for action grounding, and the object movements (e.g., the trace of human hands or robotic arms) in videos are labeled by Trace-of-Mark (ToM) for action planning. Extensive experiments show that SoM and ToM reach great synergy and facilitate the acquisition of spatial-temporal intelligence for our Magma model, which is fundamental to a wide range of tasks as shown in Fig.1. In particular, Magma creates new state-of-the-art results on UI navigation and robotic manipulation tasks, outperforming previous models that are specifically tailored to these tasks. On image and video-related multimodal tasks, Magma also compares favorably to popular large multimodal models that are trained on much larger datasets. We make our model and code public for reproducibility at this https URL. 

**Abstract (ZH)**: 以下是将给定内容翻译成中文的结果，符合学术规范：

我们介绍了Magma，一种基础模型，用于处理数字和物理世界中的多模态人工智能代理任务。Magma 是对视觉语言（VL）模型的重要扩展，它不仅保留了后者在语言理解方面的能力（言语智能），而且还具备在视觉空间世界中规划和执行任务的能力（时空智能），能够完成从界面导航到机器人操作等多种代理任务。

为了赋予其代理能力，Magma 在跨图像、视频和机器人数据的大规模异构数据集上进行了预训练，其中图像中的可操作视觉对象（例如GUI中的可点击按钮）被标记为Set-of-Mark（SoM）以实现动作绑定，而视频中的对象运动（例如人的手或机器人臂的轨迹）被标记为Trace-of-Mark（ToM）以支持动作规划。大量的实验表明，SoM和ToM达到了很好的协同作用，并促进了我们的Magma模型获取时空智能，这对于广泛的任务至关重要，如图1所示。尤其值得注意的是，Magma 在界面导航和机器人操作任务上创造了新的最佳成果，超越了专门为此类任务设计的先前模型。在与图像和视频相关的多模态任务上，Magma 在使用更大数据集训练的流行大型多模态模型中也表现优越。我们在此处提供我们的模型和代码以确保可再现性：[填写链接处的URL]。

请注意，最后的网址需要替换为实际的公开链接地址。 

---
# DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning 

**Title (ZH)**: DeepResonance：基于音乐中心多向指令调优的多模态音乐理解增强 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2502.12623)  

**Abstract**: Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-alignment Transformer to enhance modality fusion prior to input into text LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We plan to open-source the models and the newly constructed datasets. 

**Abstract (ZH)**: 近年来，在音乐大规模语言模型（LLMs）方面的最新进展显著提高了音乐理解任务的能力，这些任务涉及模型分析和解释各种音乐元素的能力。这些改进主要集中在整合音乐和文本输入。然而，将图像、视频以及音乐文本特征等其他模态纳入以增强音乐理解的潜力尚未被充分探索。为了填补这一空白，我们提出了一种名为DeepResonance的多模态音乐理解LLM，该模型是通过多视角指令调优并结合多视角对齐的音乐、文本、图像和视频数据进行微调的。为此，我们构建了Music4way-MI2T、Music4way-MV2T和Music4way-Any2T三个四视角训练和评估数据集，旨在使DeepResonance能够结合视觉和文本音乐特征内容。此外，我们引入了多样本的ImageBind嵌入和预对齐的Transformer，以增强在输入文本LLM之前的各种模态的融合，使DeepResonance适用于多视角指令调优。我们的模型在六项音乐理解任务中均达到了最先进的性能，突显了辅助模态的优势以及DeepResonance的结构优势。我们计划开源模型和新构建的数据集。 

---
# A Comprehensive Survey on Generative AI for Video-to-Music Generation 

**Title (ZH)**: 面向视频到音乐生成的生成型AI综述 

**Authors**: Shulei Ji, Songruoyao Wu, Zihao Wang, Shuyu Li, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12489)  

**Abstract**: The burgeoning growth of video-to-music generation can be attributed to the ascendancy of multimodal generative models. However, there is a lack of literature that comprehensively combs through the work in this field. To fill this gap, this paper presents a comprehensive review of video-to-music generation using deep generative AI techniques, focusing on three key components: visual feature extraction, music generation frameworks, and conditioning mechanisms. We categorize existing approaches based on their designs for each component, clarifying the roles of different strategies. Preceding this, we provide a fine-grained classification of video and music modalities, illustrating how different categories influence the design of components within the generation pipelines. Furthermore, we summarize available multimodal datasets and evaluation metrics while highlighting ongoing challenges in the field. 

**Abstract (ZH)**: 视频到音乐生成的日益增长可以归因于多模态生成模型的兴起。然而，该领域的现有文献尚未进行全面梳理。为填补这一空白，本文综述了使用深度生成AI技术的视频到音乐生成方法，重点关注三个关键组成部分：视觉特征提取、音乐生成框架和条件机制。我们根据每个组成部分的设计对现有方法进行了分类，阐明了不同策略的作用。在此基础上，我们提供了视频和音乐模态的精细分类，说明不同类别如何影响生成管道中各组成部分的设计。此外，我们总结了可用的多模态数据集和评估指标，并指出了该领域的持续挑战。 

---
