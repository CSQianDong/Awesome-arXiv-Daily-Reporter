# Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems 

**Title (ZH)**: 可信的答案，复杂的数据：领域专家系统中低资源检索增强生成鸿沟的跨越 

**Authors**: Nayoung Choi, Grace Byun, Andrew Chung, Ellie S. Paek, Shinsun Lee, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19596)  

**Abstract**: RAG has become a key technique for enhancing LLMs by reducing hallucinations, especially in domain expert systems where LLMs may lack sufficient inherent knowledge. However, developing these systems in low-resource settings introduces several challenges: (1) handling heterogeneous data sources, (2) optimizing retrieval phase for trustworthy answers, and (3) evaluating generated answers across diverse aspects. To address these, we introduce a data generation pipeline that transforms raw multi-modal data into structured corpus and Q&A pairs, an advanced re-ranking phase improving retrieval precision, and a reference matching algorithm enhancing answer traceability. Applied to the automotive engineering domain, our system improves factual correctness (+1.94), informativeness (+1.16), and helpfulness (+1.67) over a non-RAG baseline, based on a 1-5 scale by an LLM judge. These results highlight the effectiveness of our approach across distinct aspects, with strong answer grounding and transparency. 

**Abstract (ZH)**: RAG已成为通过减少幻觉来增强大规模语言模型（LLM）的关键技术，特别是在LLM可能缺乏足够内在知识的领域专家系统中表现尤为突出。然而，在低资源环境中开发这些系统也带来了若干挑战：（1）处理异构数据源，（2）优化检索阶段以获得可靠的答案，以及（3）从多方面评价生成的答案。为解决这些问题，我们提出了一种数据生成流水线，将原始多模态数据转换为结构化的语料库和问答对，引入先进的重新排名阶段以提高检索精度，并采用参考匹配算法增强答案的可追溯性。在汽车工程领域应用时，我们的系统在事实正确性（+1.94）、信息量（+1.16）和有用性（+1.67）方面超过了非RAG基线，评分标准为1到5分，由LLM评判员基于1-5分进行评估。这些结果突显了我们方法在多个方面的有效性，具有强大的答案落地性和透明度。 

---
# Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy 

**Title (ZH)**: Optimus-2：基于目标-观察-动作条件策略的多模态Minecraft代理 

**Authors**: Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.19902)  

**Abstract**: Building an agent that can mimic human behavior patterns to accomplish various open-world tasks is a long-term goal. To enable agents to effectively learn behavioral patterns across diverse tasks, a key challenge lies in modeling the intricate relationships among observations, actions, and language. To this end, we propose Optimus-2, a novel Minecraft agent that incorporates a Multimodal Large Language Model (MLLM) for high-level planning, alongside a Goal-Observation-Action Conditioned Policy (GOAP) for low-level control. GOAP contains (1) an Action-guided Behavior Encoder that models causal relationships between observations and actions at each timestep, then dynamically interacts with the historical observation-action sequence, consolidating it into fixed-length behavior tokens, and (2) an MLLM that aligns behavior tokens with open-ended language instructions to predict actions auto-regressively. Moreover, we introduce a high-quality Minecraft Goal-Observation-Action (MGOA)} dataset, which contains 25,000 videos across 8 atomic tasks, providing about 30M goal-observation-action pairs. The automated construction method, along with the MGOA dataset, can contribute to the community's efforts to train Minecraft agents. Extensive experimental results demonstrate that Optimus-2 exhibits superior performance across atomic tasks, long-horizon tasks, and open-ended instruction tasks in Minecraft. 

**Abstract (ZH)**: 构建能够模拟人类行为模式的代理以完成各种开放世界任务是一项长期目标。为了使代理能够有效地跨多种任务学习行为模式，一个关键挑战在于建模观测、动作和语言之间的复杂关系。为此，我们提出了Optimus-2，这是一种结合了多模态大型语言模型（MLLM）进行高层次规划，并结合了目标-观测-动作条件策略（GOAP）进行低层次控制的新型Minecraft代理。GOAP 包含以下两个组成部分：(1) 行动导向的行为编码器，该编码器在每个时间步长中建模观测与动作之间的因果关系，然后与历史观测-动作序列动态交互，将其合并成固定长度的行为令牌；(2) MLLM，该模型将行为令牌与开放性语言指令对齐以自回归地预测动作。此外，我们还引入了一个高质量的Minecraft目标-观测-动作（MGOA）数据集，该数据集包含25,000个视频，横跨8个原子任务，提供了大约3000万个目标-观测-动作对。自动化构建方法以及MGOA数据集能够为Minecraft代理的训练工作做出贡献。广泛的实验结果表明，Optimus-2在Minecraft中的原子任务、长时任务和开放式指令任务中均表现出卓越的性能。 

---
# Repurposing the scientific literature with vision-language models 

**Title (ZH)**: 使用视觉-语言模型重新利用科学文献 

**Authors**: Anton Alyakin, Jaden Stryker, Daniel Alexander Alber, Karl L. Sangwon, Brandon Duderstadt, Akshay Save, David Kurland, Spencer Frome, Shrutika Singh, Jeff Zhang, Eunice Yang, Ki Yun Park, Cordelia Orillac, Aly A. Valliani, Sean Neifert, Albert Liu, Aneek Patel, Christopher Livia, Darryl Lau, Ilya Laufer, Peter A. Rozman, Eveline Teresa Hidalgo, Howard Riina, Rui Feng, Todd Hollon, Yindalon Aphinyanaphongs, John G. Golfinos, Laura Snyder, Eric Leuthardt, Douglas Kondziolka, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2502.19546)  

**Abstract**: Research in AI for Science often focuses on using AI technologies to augment components of the scientific process, or in some cases, the entire scientific method; how about AI for scientific publications? Peer-reviewed journals are foundational repositories of specialized knowledge, written in discipline-specific language that differs from general Internet content used to train most large language models (LLMs) and vision-language models (VLMs). We hypothesized that by combining a family of scientific journals with generative AI models, we could invent novel tools for scientific communication, education, and clinical care. We converted 23,000 articles from Neurosurgery Publications into a multimodal database - NeuroPubs - of 134 million words and 78,000 image-caption pairs to develop six datasets for building AI models. We showed that the content of NeuroPubs uniquely represents neurosurgery-specific clinical contexts compared with broader datasets and PubMed. For publishing, we employed generalist VLMs to automatically generate graphical abstracts from articles. Editorial board members rated 70% of these as ready for publication without further edits. For education, we generated 89,587 test questions in the style of the ABNS written board exam, which trainee and faculty neurosurgeons found indistinguishable from genuine examples 54% of the time. We used these questions alongside a curriculum learning process to track knowledge acquisition while training our 34 billion-parameter VLM (CNS-Obsidian). In a blinded, randomized controlled trial, we demonstrated the non-inferiority of CNS-Obsidian to GPT-4o (p = 0.1154) as a diagnostic copilot for a neurosurgical service. Our findings lay a novel foundation for AI with Science and establish a framework to elevate scientific communication using state-of-the-art generative artificial intelligence while maintaining rigorous quality standards. 

**Abstract (ZH)**: 人工智能在科学领域的研究通常侧重于利用AI技术来增强科学过程的各个部分，甚至在某些情况下，整个科学方法；那么，人工智能在科学研究出版方面的作用又如何呢？同行评审期刊是专门知识的基础存储库，其内容使用的是学科专用语言，与广泛用于训练大多数大型语言模型（LLMs）和视觉-语言模型（VLMs）的通用互联网内容有所不同。我们假设，通过将一系列科学期刊与生成AI模型相结合，可以发明新型工具，用于科学交流、教育和临床护理。我们将神经外科出版物中的23,000篇文章转换为一个多模态数据库——NeuroPubs，包含1.34亿词和78,000张图片及其描述对，开发了六个数据集以构建AI模型。我们证明了NeuroPubs的内容与其他广泛数据集和PubMed相比，唯一地代表了神经外科特有的临床环境。在出版方面，我们使用通才型VLMs自动生成图形摘要。期刊编辑评定70%的文章无需进一步编辑即可出版。在教育方面，我们生成了89,587道ABNS笔试风格的问题，这些问题是训练和教学外科医生在54%的时间内无法区分的真正例子。我们将这些问题与课程学习过程结合使用，以追踪知识获取情况，同时训练我们340亿参数的VLM（CNS-Obsidian）。在一项盲法随机对照试验中，我们证明了CNS-Obsidian与GPT-4o（p = 0.1154）在作为神经外科服务的诊断副驾方面具有非劣效性。我们的研究为AI与科学奠定了新的基础，确立了利用最先进生成人工智能提升科学交流的框架，并保持严格的质量标准。 

---
# Opus: A Workflow Intention Framework for Complex Workflow Generation 

**Title (ZH)**: opus：一种复杂工作流生成的工作流意图框架 

**Authors**: Phillip Kingston, Théo Fagnoni, Mahsun Altin  

**Link**: [PDF](https://arxiv.org/pdf/2502.19532)  

**Abstract**: This paper introduces Workflow Intention, a novel framework for identifying and encoding process objectives within complex business environments. Workflow Intention is the alignment of Input, Process and Output elements defining a Workflow's transformation objective interpreted from Workflow Signal inside Business Artefacts. It specifies how Input is processed to achieve desired Output, incorporating quality standards, business rules, compliance requirements and constraints. We adopt an end-to-end Business Artefact Encoder and Workflow Signal interpretation methodology involving four steps: Modality-Specific Encoding, Intra-Modality Attention, Inter-Modality Fusion Attention then Intention Decoding. We provide training procedures and critical loss function definitions. In this paper we introduce the concepts of Workflow Signal and Workflow Intention, where Workflow Signal decomposed into Input, Process and Output elements is interpreted from Business Artefacts, and Workflow Intention is a complete triple of these elements. We introduce a mathematical framework for representing Workflow Signal as a vector and Workflow Intention as a tensor, formalizing properties of these objects. Finally, we propose a modular, scalable, trainable, attention-based multimodal generative system to resolve Workflow Intention from Business Artefacts. 

**Abstract (ZH)**: 本文介绍了工作流意图（Workflow Intention）框架，这是一种在复杂商业环境中识别和编码过程目标的新颖方法。工作流意图表示工作流基于业务 artefacts 中的 workflow signal 转化目标的 Input、Process 和 Output 元素的对齐。它明确了如何处理 Input 以实现期望的 Output，并融入了质量标准、业务规则、合规要求和约束条件。我们采用涵盖四个步骤的端到端商业模式编码和 workflow signal 解释方法：特定模态编码、模态内注意力、跨模态融合注意力以及意图解码。我们提供了训练程序和关键损失函数定义。在本文中，我们引入了 workflow signal 和 work flow intention 的概念，其中 workflow signal 由 workflow 的 Input、Process 和 Output 元素组成，并从业务 artefacts 中进行解释，而 work flow intention 则是这些元素的完整三元组。我们提出了数学框架来表示 workflow signal 为向量和 workflow intention 为张量，正式化了这些对象的性质。最后，我们提出了一种模块化、可扩展、可训练的基于注意力的多模态生成系统，用于从业务 artefacts 中解析工作流意图。 

---
# Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription 

**Title (ZH)**: 以貌取书：探究多模态大语言模型在多页手写文档转录中的应用 

**Authors**: Benjamin Gutteridge, Matthew Thomas Jackson, Toni Kukurin, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.20295)  

**Abstract**: Handwritten text recognition (HTR) remains a challenging task, particularly for multi-page documents where pages share common formatting and contextual features. While modern optical character recognition (OCR) engines are proficient with printed text, their performance on handwriting is limited, often requiring costly labeled data for fine-tuning. In this paper, we explore the use of multi-modal large language models (MLLMs) for transcribing multi-page handwritten documents in a zero-shot setting. We investigate various configurations of commercial OCR engines and MLLMs, utilizing the latter both as end-to-end transcribers and as post-processors, with and without image components. We propose a novel method, '+first page', which enhances MLLM transcription by providing the OCR output of the entire document along with just the first page image. This approach leverages shared document features without incurring the high cost of processing all images. Experiments on a multi-page version of the IAM Handwriting Database demonstrate that '+first page' improves transcription accuracy, balances cost with performance, and even enhances results on out-of-sample text by extrapolating formatting and OCR error patterns from a single page. 

**Abstract (ZH)**: 手写文本识别（HTR）仍然是一个具有挑战性的任务，尤其是对于多页文档，在这些文档中，各页共享常见的格式和上下文特征。尽管现代光学字符识别（OCR）引擎在印刷文本方面表现出色，但在手写识别方面的能力却有限，通常需要高质量的标注数据进行微调，这往往成本高昂。在本文中，我们探讨了在零样本设置下使用多模式大型语言模型（MLLMs）来转录多页手写文档的可能性。我们研究了不同配置的商用OCR引擎和MLLMs，利用后者作为端到端的转录器或后处理器，有时甚至使用图像组件。我们提出了一种新颖的方法‘+第一页’，通过提供整文档的OCR输出以及仅第一页的图像来增强MLLM的转录效果。这种方法利用共享的文档特征，而无需处理所有图像的高成本。在IAM手写数据库的多页版本上的实验表明，‘+第一页’方法提高了转录准确性，平衡了成本与性能，并且甚至通过从单页中推断格式和OCR错误模式提高了不在样本中的文本转录效果。 

---
# Explainable, Multi-modal Wound Infection Classification from Images Augmented with Generated Captions 

**Title (ZH)**: 具备解释性的、多模态伤口感染分类：结合生成的描述图像 

**Authors**: Palawat Busaranuvong, Emmanuel Agu, Reza Saadati Fard, Deepak Kumar, Shefalika Gautam, Bengisu Tulu, Diane Strong  

**Link**: [PDF](https://arxiv.org/pdf/2502.20277)  

**Abstract**: Infections in Diabetic Foot Ulcers (DFUs) can cause severe complications, including tissue death and limb amputation, highlighting the need for accurate, timely diagnosis. Previous machine learning methods have focused on identifying infections by analyzing wound images alone, without utilizing additional metadata such as medical notes. In this study, we aim to improve infection detection by introducing Synthetic Caption Augmented Retrieval for Wound Infection Detection (SCARWID), a novel deep learning framework that leverages synthetic textual descriptions to augment DFU images. SCARWID consists of two components: (1) Wound-BLIP, a Vision-Language Model (VLM) fine-tuned on GPT-4o-generated descriptions to synthesize consistent captions from images; and (2) an Image-Text Fusion module that uses cross-attention to extract cross-modal embeddings from an image and its corresponding Wound-BLIP caption. Infection status is determined by retrieving the top-k similar items from a labeled support set. To enhance the diversity of training data, we utilized a latent diffusion model to generate additional wound images. As a result, SCARWID outperformed state-of-the-art models, achieving average sensitivity, specificity, and accuracy of 0.85, 0.78, and 0.81, respectively, for wound infection classification. Displaying the generated captions alongside the wound images and infection detection results enhances interpretability and trust, enabling nurses to align SCARWID outputs with their medical knowledge. This is particularly valuable when wound notes are unavailable or when assisting novice nurses who may find it difficult to identify visual attributes of wound infection. 

**Abstract (ZH)**: 糖尿病足溃疡（DFUs）感染可能导致严重并发症，包括组织坏死和截肢，凸显了准确、及时诊断的必要性。以往的机器学习方法主要依赖于分析伤口图像来识别感染，而未利用附加的元数据，如医疗笔记。在本研究中，我们旨在通过引入合成描述增强检索以检测伤口感染（SCARWID）这一新颖的深度学习框架来改进感染的检测。SCARWID包含两个组成部分：（1）伤口-BLIP，一种基于GPT-4o生成的描述对图像进行微调的视觉-语言模型（VLM），用于从图像中合成一致的描述；（2）图像-文本融合模块，该模块利用交叉注意机制从图像及其对应的伤口-BLIP描述中提取跨模态嵌入。感染状态通过从标记的支持集中检索最相似项来确定。为了增强训练数据的多样性，我们利用潜在扩散模型生成额外的伤口图像。结果表明，SCARWID超越了最先进的模型，分别在伤口感染分类中的平均敏感性、特异性和准确率达到了85%、78%和81%。将生成的描述与伤口图像和感染检测结果一起显示，能够增强解释性和可信度，使护士能够将SCARWID的输出与医学知识对接。特别是在伤口记录不可用或帮助识别伤口感染的视觉特征存在困难的初学者护士时，这一点尤为重要。 

---
# M-LLM Based Video Frame Selection for Efficient Video Understanding 

**Title (ZH)**: 基于M-LLM的视频帧选择方法以实现高效视频理解 

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19680)  

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks. 

**Abstract (ZH)**: 近年来，多模态大型语言模型（M-LLMs）在视频推理方面展现了令人鼓舞的结果。流行的多模态大型语言模型（M-LLM）框架通常采用简单的均匀采样方法来减少输入M-LLM的视频帧数量，尤其是在处理长上下文视频时。然而，这种方法可能会在视频的某些时期丢失关键的上下文信息，导致下游M-LLM可能缺乏足够的视觉信息来回答问题。为了解决这一问题，我们提出了一种轻量级的M-LLM基于的帧选择方法，该方法能够适应性地选择与用户查询更加相关的关键帧。为了训练提出的帧选择器，我们引入了两种监督信号：（i）空间信号，通过提示M-LLM单帧的重要性分数；（ii）时间信号，在此信号中，通过使用所有候选帧的字幕来提示大型语言模型（LLM）进行多帧选择。选出的关键帧随后由冻结状态的下游视频M-LLM用于视觉推理和问题回答。实验结果表明，提出的M-LLM视频帧选择器能够提高各类中长上下文视频大型语言模型（视频-LLM）在（ActivityNet，NExT-QA）和（EgoSchema，LongVideoBench）基准测试中的性能。 

---
# SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning 

**Title (ZH)**: SuPreME：一种监督预训练的多模态心电图表示学习框架 

**Authors**: Mingsheng Cai, Jiuming Jiang, Wenhao Huang, Che Liu, Rossella Arcucci  

**Link**: [PDF](https://arxiv.org/pdf/2502.19668)  

**Abstract**: Cardiovascular diseases are a leading cause of death and disability worldwide. Electrocardiogram (ECG) recordings are critical for diagnosing and monitoring cardiac health, but obtaining large-scale annotated ECG datasets is labor-intensive and time-consuming. Recent ECG Self-Supervised Learning (eSSL) methods mitigate this by learning features without extensive labels but fail to capture fine-grained clinical semantics and require extensive task-specific fine-tuning. To address these challenges, we propose $\textbf{SuPreME}$, a $\textbf{Su}$pervised $\textbf{Pre}$-training framework for $\textbf{M}$ultimodal $\textbf{E}$CG representation learning. SuPreME applies Large Language Models (LLMs) to extract structured clinical entities from free-text ECG reports, filter out noise and irrelevant content, enhance clinical representation learning, and build a high-quality, fine-grained labeled dataset. By using text-based cardiac queries instead of traditional categorical labels, SuPreME enables zero-shot classification of unseen diseases without additional fine-tuning. We evaluate SuPreME on six downstream datasets covering 127 cardiac conditions, achieving superior zero-shot AUC performance over state-of-the-art eSSL and multimodal methods by over 1.96\%. Results demonstrate the effectiveness of SuPreME in leveraging structured, clinically relevant knowledge for high-quality ECG representations. All code and data will be released upon acceptance. 

**Abstract (ZH)**: 心血管疾病是全球范围内导致死亡和残疾的主要原因之一。心电图（ECG）记录对于诊断和监控心脏健康至关重要，但获取大规模标注的心电图数据集是一个劳动密集型且耗时的过程。近期的心电图自监督学习（eSSL）方法通过无需大量标签即可学习特征来缓解这一问题，但无法捕捉到精细的临床语义，并且需要大量的任务特定微调。为了解决这些挑战，我们提出了一种名为$\textbf{SuPreME}$的$\textbf{S}$upervised $\textbf{P}$re-training框架，用于$\textbf{M}$ultimodal $\textbf{E}$CG表示学习。SuPreME利用大型语言模型（LLMs）从自由文本ECG报告中提取结构化的临床实体，过滤掉噪声和无关内容，增强临床表示学习，并构建高质量的细粒度标注数据集。通过使用基于文本的心脏查询而非传统的分类标签，SuPreME在未经额外微调的情况下实现了对未见过疾病的零样本分类。我们将在六个下游数据集上评估SuPreME，这些数据集覆盖了127种心脏状况，SuPreME在零样本AUC性能上比当前最先进的eSSL和多模态方法优越1.96%以上。结果表明，SuPreME在利用结构化的、与临床相关的知识来生成高质量的ECG表示方面具有有效性。在接受后的所有代码和数据将被发布。 

---
# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success 

**Title (ZH)**: 视觉-语言-动作模型的微调：优化速度与成功率 

**Authors**: Moo Jin Kim, Chelsea Finn, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19645)  

**Abstract**: Recent vision-language-action models (VLAs) build upon pretrained vision-language models and leverage diverse robot datasets to demonstrate strong task execution, language following ability, and semantic generalization. Despite these successes, VLAs struggle with novel robot setups and require fine-tuning to achieve good performance, yet how to most effectively fine-tune them is unclear given many possible strategies. In this work, we study key VLA adaptation design choices such as different action decoding schemes, action representations, and learning objectives for fine-tuning, using OpenVLA as our representative base model. Our empirical analysis informs an Optimized Fine-Tuning (OFT) recipe that integrates parallel decoding, action chunking, a continuous action representation, and a simple L1 regression-based learning objective to altogether improve inference efficiency, policy performance, and flexibility in the model's input-output specifications. We propose OpenVLA-OFT, an instantiation of this recipe, which sets a new state of the art on the LIBERO simulation benchmark, significantly boosting OpenVLA's average success rate across four task suites from 76.5% to 97.1% while increasing action generation throughput by 26$\times$. In real-world evaluations, our fine-tuning recipe enables OpenVLA to successfully execute dexterous, high-frequency control tasks on a bimanual ALOHA robot and outperform other VLAs ($\pi_0$ and RDT-1B) fine-tuned using their default recipes, as well as strong imitation learning policies trained from scratch (Diffusion Policy and ACT) by up to 15% (absolute) in average success rate. We release code for OFT and pretrained model checkpoints at this https URL. 

**Abstract (ZH)**: 近年来，视觉-语言-动作模型（VLAs）基于预训练的视觉-语言模型，并利用多种多样的机器人数据集，展示了强大的任务执行能力、语言理解和语义泛化能力。尽管取得了这些成功，但VLAs在面对新型机器人设置时仍存在挑战，需要进行微调才能获得良好的性能，然而如何最有效地进行微调仍不明确，因为存在多种可能的策略。在本工作中，我们研究了关键的VLA调整设计选择，如不同的动作解码方案、动作表示和学习目标，以进行微调，将OpenVLA作为我们的代表性基模型。我们的实证分析为一种优化的微调（OFT）食谱提供建议，该食谱整合了并行解码、动作分块、连续的动作表示，以及基于L1回归的学习目标，以整体提高推理效率、策略性能以及模型输入输出规范的灵活性。我们提出了OpenVLA-OFT，这是该食谱的一个实例，在LIBERO模拟基准测试中，它不仅在四个任务套件中将OpenVLA的平均成功率从76.5%提升到了97.1%，还使动作生成吞吐量提高了26倍。在现实世界的评估中，我们的微调方案使OpenVLA能够成功执行双臂ALOHA机器人的灵巧、高频控制任务，并优于使用默认食谱进行微调的其他VLAs（$\pi_0$和RDT-1B），以及从零开始训练的强大模仿学习策略（Diffusion Policy和ACT），在平均成功率上提高了高达15%。我们在此提供OFT的代码和预训练模型检查点，链接为：[此 url 地址]。 

---
# Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents 

**Title (ZH)**: 精挑细选精华：基于视觉的数据选择与协作代理方法 

**Authors**: Zhenyu Liu, Yunxin Li, Baotian Hu, Wenhan Luo, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19917)  

**Abstract**: To improve Multimodal Large Language Models' (MLLMs) ability to process images and complex instructions, researchers predominantly curate large-scale visual instruction tuning datasets, which are either sourced from existing vision tasks or synthetically generated using LLMs and image descriptions. However, they often suffer from critical flaws, including misaligned instruction-image pairs and low-quality images. Such issues hinder training efficiency and limit performance improvements, as models waste resources on noisy or irrelevant data with minimal benefit to overall capability. To address this issue, we propose a \textbf{Vi}sual-Centric \textbf{S}election approach via \textbf{A}gents Collaboration (ViSA), which centers on image quality assessment and image-instruction relevance evaluation. Specifically, our approach consists of 1) an image information quantification method via visual agents collaboration to select images with rich visual information, and 2) a visual-centric instruction quality assessment method to select high-quality instruction data related to high-quality images. Finally, we reorganize 80K instruction data from large open-source datasets. Extensive experiments demonstrate that ViSA outperforms or is comparable to current state-of-the-art models on seven benchmarks, using only 2.5\% of the original data, highlighting the efficiency of our data selection approach. Moreover, we conduct ablation studies to validate the effectiveness of each component of our method. The code is available at this https URL. 

**Abstract (ZH)**: 为了提高多模态大型语言模型（MLLMs）处理图像和复杂指令的能力，研究者们主要建立了大规模的视觉指令调优数据集，这些数据集要么来源自现有的视觉任务，要么是使用大型语言模型和图像描述合成生成的。然而，这类数据集往往存在关键性的缺陷，包括指令与图像对齐不准确以及低质量的图像。这些问题阻碍了训练效率，并限制了性能提升，因为模型将宝贵的资源浪费在了噪声较大或无关的数据上，而这些数据对整体能力的提升几乎没有益处。为了解决这一问题，我们提出了一种通过代理协作的视觉中心选择方法（ViSA），该方法以图像质量评估和图像-指令相关性评估为核心。具体而言，我们的方法包括以下两个方面：1) 通过视觉代理协作量化图像信息，以选择富含视觉信息的图像；2) 采用视觉中心的指令质量评估方法，以选择与高质量图像相关联的高质量指令数据。最后，我们重新组织了来自大型开源数据集的8万条指令数据。广泛的实验表明，使用仅占原始数据2.5%的量，ViSA在七个基准测试中展现了优于或可与现有最先进的模型相媲美的性能，突显了我们数据选择方法的高效性。此外，我们还进行了消融研究，以验证我们方法中每个组件的有效性。相关代码可在以下网址获取：这个 https URL。 

---
# MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge 

**Title (ZH)**: MMKE-Bench：一种用于多元视觉知识多样编辑的基准测试 

**Authors**: Yuntao Du, Kailin Jiang, Zhi Gao, Chenrui Shi, Zilong Zheng, Siyuan Qi, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.19870)  

**Abstract**: Knowledge editing techniques have emerged as essential tools for updating the factual knowledge of large language models (LLMs) and multimodal models (LMMs), allowing them to correct outdated or inaccurate information without retraining from scratch. However, existing benchmarks for multimodal knowledge editing primarily focus on entity-level knowledge represented as simple triplets, which fail to capture the complexity of real-world multimodal information. To address this issue, we introduce MMKE-Bench, a comprehensive MultiModal Knowledge Editing Benchmark, designed to evaluate the ability of LMMs to edit diverse visual knowledge in real-world scenarios. MMKE-Bench addresses these limitations by incorporating three types of editing tasks: visual entity editing, visual semantic editing, and user-specific editing. Besides, MMKE-Bench uses free-form natural language to represent and edit knowledge, offering a more flexible and effective format. The benchmark consists of 2,940 pieces of knowledge and 8,363 images across 33 broad categories, with evaluation questions automatically generated and human-verified. We assess five state-of-the-art knowledge editing methods on three prominent LMMs, revealing that no method excels across all criteria, and that visual and user-specific edits are particularly challenging. MMKE-Bench sets a new standard for evaluating the robustness of multimodal knowledge editing techniques, driving progress in this rapidly evolving field. 

**Abstract (ZH)**: 知识编辑技术已成为更新大规模语言模型（LLMs）和多模态模型（LMMs）事实性知识的重要工具，使其能够纠正过时或不准确的信息，而无需从头开始重新训练。然而，现有的多模态知识编辑基准主要集中在用简单三元组表示的实体级知识上，无法捕捉到现实世界多模态信息的复杂性。为解决这一问题，我们提出了MMKE-Bench，这是一个全面的多模态知识编辑基准，旨在评估LMMs在实际场景中编辑多样化视觉知识的能力。MMKE-Bench克服了这些局限性，通过整合三种类型的编辑任务：视觉实体编辑、视觉语义编辑和用户特定编辑来实现这一目标。此外，MMKE-Bench采用自由形式的自然语言表示和编辑知识，提供了一种更灵活和有效的格式。该基准包含2940条知识和8363张图像，覆盖33个广泛的类别，并由自动生成的评估问题和人工验证组成。我们在三个主流LMM上评估了五种最先进的知识编辑方法，结果显示没有一种方法在所有标准上都表现出色，且视觉和用户特定的编辑尤为具有挑战性。MMKE-Bench为评估多模态知识编辑技术的稳健性树立了新标准，推动了这一迅速发展的领域的进步。 

---
# Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think 

**Title (ZH)**: 多模态表示对齐在图像生成中的应用：文本-图像交错控制比你想象的要简单 

**Authors**: Liang Chen, Shuai Bai, Wenhao Chai, Weichu Xie, Haozhe Zhao, Leon Vinci, Junyang Lin, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20172)  

**Abstract**: The field of advanced text-to-image generation is witnessing the emergence of unified frameworks that integrate powerful text encoders, such as CLIP and T5, with Diffusion Transformer backbones. Although there have been efforts to control output images with additional conditions, like canny and depth map, a comprehensive framework for arbitrary text-image interleaved control is still lacking. This gap is especially evident when attempting to merge concepts or visual elements from multiple images in the generation process. To mitigate the gap, we conducted preliminary experiments showing that large multimodal models (LMMs) offer an effective shared representation space, where image and text can be well-aligned to serve as a condition for external diffusion models. Based on this discovery, we propose Dream Engine, an efficient and unified framework designed for arbitrary text-image interleaved control in image generation models. Building on powerful text-to-image models like SD3.5, we replace the original text-only encoders by incorporating versatile multimodal information encoders such as QwenVL. Our approach utilizes a two-stage training paradigm, consisting of joint text-image alignment and multimodal interleaved instruction tuning. Our experiments demonstrate that this training method is effective, achieving a 0.69 overall score on the GenEval benchmark, and matching the performance of state-of-the-art text-to-image models like SD3.5 and FLUX. 

**Abstract (ZH)**: 先进图文生成领域的研究正见证着统一框架的出现，这些框架将强大的文本编码器（如CLIP和T5）与扩散变换器（Diffusion Transformer）骨干网络相结合。尽管已有努力通过添加条件（如Canny边缘检测和深度图）来控制生成的图像，但一个全面的框架仍然缺失，可以实现任意的图文交织控制。尤其是在生成过程中尝试结合多个图像的概念或视觉元素时，这种差距尤为明显。为了弥合这一差距，我们进行了初步实验，结果显示，大型多模态模型（LMMs）提供了一个有效的共享表示空间，在这个空间中，图像和文本可以很好地对齐，作为外部扩散模型的条件。基于这一发现，我们提出了一种高效且统一的框架——Dream Engine，用于图像生成模型中的任意图文交织控制。基于强大的文本转图像模型（如SD3.5），我们通过整合多模态信息编码器（如QwenVL）替代了原有的仅文本编码器。我们的方法采用两阶段训练范式，包括文本-图像对齐和多模态交织指令调优。实验结果证明，这种方法是有效的，在GenEval基准测试中取得了0.69的总体评分，与当前最先进的文本转图像模型（如SD3.5和FLUX）的性能相当。 

---
# Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore 

**Title (ZH)**: 视觉编码器（早已）识其所见：通过简单的细粒度CLIPScore减轻物体幻视问题 

**Authors**: Hongseok Oh, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20034)  

**Abstract**: Recently, Large Vision-Language Models (LVLMs) show remarkable performance across various domains. However, these models suffer from object hallucination. This study revisits the previous claim that the primary cause of such hallucination lies in the limited representational capacity of the vision encoder. Our analysis reveals that the capacity of the vision encoder itself is already enough for detecting object hallucination. Based on this insight, we propose a Fine-grained CLIPScore (F-CLIPScore), a simple yet effective evaluation metric that enhances object-level granularity by incorporating text embeddings at the noun phrase level. Evaluations on the OHD-Caps benchmark show that F-CLIPScore significantly outperforms conventional CLIPScore in accuracy by a large margin of 39.6% without additional training. We further validate F-CLIPScore by showing that LVLM trained with the data filtered using F-CLIPScore exhibits reduced hallucination. 

**Abstract (ZH)**: 近年来，大型多模态模型（Large Vision-Language Models, LVLMs）在各种领域中展现了显著的性能。然而，这些模型存在对象幻觉的问题。本研究重新审视了此前关于这种幻觉主要源于视觉编码器表示能力有限的说法。我们的分析表明，视觉编码器本身的能力已经足以检测对象幻觉。基于这一洞察，我们提出了精细粒度的CLIPScore（F-CLIPScore），这是一种简单而有效的评估指标，通过在名词短语层面引入文本嵌入来增强对象级别的粒度。在OHD-Caps基准测试上的评估结果显示，F-CLIPScore在准确性方面显著优于传统的CLIPScore，提高了39.6%，且无需额外训练。进一步验证了F-CLIPScore的有效性，表明使用F-CLIPScore筛选数据训练的LVLM幻觉现象有所减少。 

---
