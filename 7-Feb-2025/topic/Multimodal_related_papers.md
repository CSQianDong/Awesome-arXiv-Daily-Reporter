# WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs 

**Title (ZH)**: 《WorldSense：多模态LLM在现实世界全方位理解能力评估》

这样翻译不仅保留了原文的意思，还符合学术论文标题的规范和风格。如果有更具体的内容需要翻译或进一步的学术规范调整，请告知。 

**Authors**: Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.04326)  

**Abstract**: In this paper, we introduce WorldSense, the first benchmark to assess the multi-modal video understanding, that simultaneously encompasses visual, audio, and text inputs. In contrast to existing benchmarks, our WorldSense has several features: (i) collaboration of omni-modality, we design the evaluation tasks to feature a strong coupling of audio and video, requiring models to effectively utilize the synergistic perception of omni-modality; (ii) diversity of videos and tasks, WorldSense encompasses a diverse collection of 1,662 audio-visual synchronised videos, systematically categorized into 8 primary domains and 67 fine-grained subcategories to cover the broad scenarios, and 3,172 multi-choice QA pairs across 26 distinct tasks to enable the comprehensive evaluation; (iii) high-quality annotations, all the QA pairs are manually labeled by 80 expert annotators with multiple rounds of correction to ensure quality. Based on our WorldSense, we extensively evaluate various state-of-the-art models. The experimental results indicate that existing models face significant challenges in understanding real-world scenarios (48.0% best accuracy). We hope our WorldSense can provide a platform for evaluating the ability in constructing and understanding coherent contexts from omni-modality. 

**Abstract (ZH)**: 在本文中，我们介绍了WorldSense，这是首个用于评估多模态视频理解的基准，同时包含了视觉、音频和文本输入。与现有的基准不同，WorldSense 具有以下特点：(i) 全模态协作：我们设计了评估任务，要求音频和视频之间有较强耦合，促使模型有效利用全模态的协同感知能力；(ii) 视频和任务的多样性：WorldSense 收集了1,662个音频-视觉同步视频，系统地将其分为8个主要领域和67个细分类别，以覆盖广泛的情景，并提供了跨26个不同任务的3,172个多选问答对，以实现全面评估；(iii) 高质量的注释：所有问答对均由80名专家标注，并经过多轮校正以确保质量。基于我们的WorldSense，我们广泛评估了多种最先进的模型。实验结果表明，现有模型在理解真实世界场景方面面临巨大挑战（最佳准确率为48.0%）。我们希望WorldSense能够提供一个平台，用于评估从全模态构建和理解连贯上下文的能力。 

---
# Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion 

**Title (ZH)**: 跨越鸿沟：通过模态反转揭示CLIP中的跨模态不对齐问题 

**Authors**: Marco Mistretta, Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, Andrew D. Bagdanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04263)  

**Abstract**: Pre-trained multi-modal Vision-Language Models like CLIP are widely used off-the-shelf for a variety of applications. In this paper, we show that the common practice of individually exploiting the text or image encoders of these powerful multi-modal models is highly suboptimal for intra-modal tasks like image-to-image retrieval. We argue that this is inherently due to the CLIP-style inter-modal contrastive loss that does not enforce any intra-modal constraints, leading to what we call intra-modal misalignment. To demonstrate this, we leverage two optimization-based modality inversion techniques that map representations from their input modality to the complementary one without any need for auxiliary data or additional trained adapters. We empirically show that, in the intra-modal tasks of image-to-image and text-to-text retrieval, approaching these tasks inter-modally significantly improves performance with respect to intra-modal baselines on more than fifteen datasets. Additionally, we demonstrate that approaching a native inter-modal task (e.g. zero-shot image classification) intra-modally decreases performance, further validating our findings. Finally, we show that incorporating an intra-modal term in the pre-training objective or narrowing the modality gap between the text and image feature embedding spaces helps reduce the intra-modal misalignment. The code is publicly available at: this https URL. 

**Abstract (ZH)**: 像CLIP这样的预训练多模态 vision-language 模型广泛用于各种应用中。在本文中，我们展示了单独利用这些强大多模态模型的文字编码器或图像编码器来执行跨模态任务（如图像到图像检索）的做法是高度低效的。我们认为，这是由于CLIP风格的跨模态对比损失未能施加任何内模态约束，导致我们称之为内模态错位的问题。为了证明这一点，我们利用了两种基于优化的模态反转技术，这些技术可以将输入模态的表示映射到互补模态，而无需额外的辅助数据或附加训练适配器。我们通过实验表明，在图像到图像和文本到文本检索的内模态任务中，采用跨模态方法可以显著提高性能，相对于内模态基线在多个（超过15个）数据集上表现更佳。此外，我们展示了在内模态方法下处理原生跨模态任务（例如零样本图像分类）会降低性能，进一步验证了我们的发现。最后，我们展示了在预训练目标中引入内模态项或缩小文本和图像特征嵌入空间的模态差距有助于减少内模态错位。代码已公开发布在：this https URL。 

---
# VTutor: An Open-Source SDK for Generative AI-Powered Animated Pedagogical Agents with Multi-Media Output 

**Title (ZH)**: VTutor：一个基于生成式AI的多媒体教学代理开源SDK 

**Authors**: Eason Chen, Chengyu Lin, Xinyi Tang, Aprille Xi, Canwen Wang, Jionghao Lin, Kenneth R Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.04103)  

**Abstract**: The rapid evolution of large language models (LLMs) has transformed human-computer interaction (HCI), but the interaction with LLMs is currently mainly focused on text-based interactions, while other multi-model approaches remain under-explored. This paper introduces VTutor, an open-source Software Development Kit (SDK) that combines generative AI with advanced animation technologies to create engaging, adaptable, and realistic APAs for human-AI multi-media interactions. VTutor leverages LLMs for real-time personalized feedback, advanced lip synchronization for natural speech alignment, and WebGL rendering for seamless web integration. Supporting various 2D and 3D character models, VTutor enables researchers and developers to design emotionally resonant, contextually adaptive learning agents. This toolkit enhances learner engagement, feedback receptivity, and human-AI interaction while promoting trustworthy AI principles in education. VTutor sets a new standard for next-generation APAs, offering an accessible, scalable solution for fostering meaningful and immersive human-AI interaction experiences. The VTutor project is open-sourced and welcomes community-driven contributions and showcases. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展已经改变了人机交互（HCI），但与LLMs的交互目前主要集中在文本交互上，而其他多媒体交互方法仍处于未充分探索的状态。本文介绍了一种开源软件开发工具包（SDK）——VTutor，它结合了生成式AI和高级动画技术，用于创建引人入胜、具有适应性和现实感的人工智能多媒体交互代理（APAs）。VTutor利用LLMs进行实时个性化反馈、高级唇同步以实现自然语音对齐，并采用WebGL渲染以无缝集成到网页中。支持各种2D和3D角色模型，VTutor使研究人员和开发者能够设计情感共鸣且上下文适应的学习代理。该工具包通过增强学习者参与度、反馈收听度和人机交互，同时推动教育中的可信赖AI原则，提升了人机交互体验。VTutor为下一代APAs设立了新标准，提供了一个易于访问、可扩展的解决方案，以促进有意义且沉浸式的人工智能交互体验。VTutor项目已开源，并欢迎社区驱动的贡献和展示。 

---
# UniForm: A Unified Diffusion Transformer for Audio-Video Generation 

**Title (ZH)**: UniForm：统一的音频-视频生成扩散变换器 

**Authors**: Lei Zhao, Linfeng Feng, Dongxu Ge, Fangqiu Yi, Chi Zhang, Xiao-Lei Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.03897)  

**Abstract**: As a natural multimodal content, audible video delivers an immersive sensory experience. Consequently, audio-video generation systems have substantial potential. However, existing diffusion-based studies mainly employ relatively independent modules for generating each modality, which lack exploration of shared-weight generative modules. This approach may under-use the intrinsic correlations between audio and visual modalities, potentially resulting in sub-optimal generation quality. To address this, we propose UniForm, a unified diffusion transformer designed to enhance cross-modal consistency. By concatenating auditory and visual information, UniForm learns to generate audio and video simultaneously within a unified latent space, facilitating the creation of high-quality and well-aligned audio-visual pairs. Extensive experiments demonstrate the superior performance of our method in joint audio-video generation, audio-guided video generation, and video-guided audio generation tasks. Our demos are available at this https URL. 

**Abstract (ZH)**: 作为自然界中的多模态内容，可听视频提供了沉浸式的感官体验。因此，音频-视频生成系统具有巨大的潜力。然而，现有的基于扩散模型的研究主要采用相对独立的模块来生成每种模态，缺乏对共享权重生成模块的探索。这一方法可能未能充分利用音频和视觉模态之间的内在关联，从而可能导致生成质量欠佳。为了应对这一问题，我们提出了一种统一扩散变换器——UniForm，旨在增强跨模态一致性。通过将听觉和视觉信息进行串联，UniForm能够在统一的潜在空间中同时生成音频和视频，从而促进高质量且对齐良好的音频-视频对的创造。大量的实验表明，我们的方法在联合音频-视频生成、音频指导视频生成和视频指导音频生成任务中表现出优越的性能。我们的演示可以在如下链接查看：[此处链接]。 

---
# The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering 

**Title (ZH)**: 令牌的隐秘生活：通过视觉信息引导减少大型视觉-语言模型的幻觉现象 

**Authors**: Zhuowei Li, Haizhou Shi, Yunhe Gao, Di Liu, Zhenting Wang, Yuxiao Chen, Ting Liu, Long Zhao, Hao Wang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2502.03628)  

**Abstract**: Large Vision-Language Models (LVLMs) can reason effectively over both textual and visual inputs, but they tend to hallucinate syntactically coherent yet visually ungrounded contents. In this paper, we investigate the internal dynamics of hallucination by examining the tokens logits rankings throughout the generation process, revealing three key patterns in how LVLMs process information: (1) gradual visual information loss -- visually grounded tokens gradually become less favored throughout generation, and (2) early excitation -- semantically meaningful tokens achieve peak activation in the layers earlier than the final layer. (3) hidden genuine information -- visually grounded tokens though not being eventually decided still retain relatively high rankings at inference. Based on these insights, we propose VISTA (Visual Information Steering with Token-logit Augmentation), a training-free inference-time intervention framework that reduces hallucination while promoting genuine information. VISTA works by combining two complementary approaches: reinforcing visual information in activation space and leveraging early layer activations to promote semantically meaningful decoding. Compared to existing methods, VISTA requires no external supervision and is applicable to various decoding strategies. Extensive experiments show that VISTA on average reduces hallucination by abount 40% on evaluated open-ended generation task, and it consistently outperforms existing methods on four benchmarks across four architectures under three decoding strategies. 

**Abstract (ZH)**: 大型视觉-语言模型（LVLMs）能够在文本和视觉输入之间进行有效的推理，但它们往往会生成语法上连贯但视觉上不符合实际的内容。本文通过分析生成过程中各个步骤的令牌对数排名，揭示了LVLMs处理信息时的三个关键模式：（1）视觉信息的渐进性损失——视觉上相关的令牌在整个生成过程中逐渐变得不那么受青睐；（2）早期激发——具有语义意义的令牌在比最终层更早的层中达到峰值激活；（3）隐藏的真实信息——虽然最终被决定，但视觉上相关的令牌仍保留在推断时相对较高的排名中。基于这些洞察，我们提出了VISTA（视觉信息引导的令牌对数增强），这是一种无需训练的在推断时进行干预的框架，能够在减少幻觉的同时促进真实信息。VISTA通过结合两种互补的方法来实现其目标：在激活空间中强化视觉信息，并利用早期层的激活来促进有意义的解码。与现有方法相比，VISTA不需要外部监督，并且适用于各种解码策略。广泛的实验表明，VISTA在评估的封闭生成任务中平均减少了约40%的幻觉，并且在三种解码策略下，四种架构的四种基准测试中，它始终优于现有方法。 

---
# Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning 

**Title (ZH)**: Omni-DNA：跨模态和多任务学习的统一基因基础模型 

**Authors**: Zehui Li, Vallijah Subasri, Yifei Shen, Dongsheng Li, Yiren Zhao, Guy-Bart Stan, Caihua Shan  

**Link**: [PDF](https://arxiv.org/pdf/2502.03499)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable generalizability across diverse tasks, yet genomic foundation models (GFMs) still require separate finetuning for each downstream application, creating significant overhead as model sizes grow. Moreover, existing GFMs are constrained by rigid output formats, limiting their applicability to various genomic tasks. In this work, we revisit the transformer-based auto-regressive models and introduce Omni-DNA, a family of cross-modal multi-task models ranging from 20 million to 1 billion parameters. Our approach consists of two stages: (i) pretraining on DNA sequences with next token prediction objective, and (ii) expanding the multi-modal task-specific tokens and finetuning for multiple downstream tasks simultaneously. When evaluated on the Nucleotide Transformer and GB benchmarks, Omni-DNA achieves state-of-the-art performance on 18 out of 26 tasks. Through multi-task finetuning, Omni-DNA addresses 10 acetylation and methylation tasks at once, surpassing models trained on each task individually. Finally, we design two complex genomic tasks, DNA2Function and Needle-in-DNA, which map DNA sequences to textual functional descriptions and images, respectively, indicating Omni-DNA's cross-modal capabilities to broaden the scope of genomic applications. All the models are available through this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在多样化的任务中表现出了显著的泛化能力，然而基因组基础模型（GFMs）仍然需要为每一个下游应用单独进行微调，随着模型规模的扩大，这种做法产生了显著的开销。此外，现有的GFMs受到固定输出格式的限制，限制了其在各种基因组任务中的应用。在本项研究中，我们重新审视了基于Transformer的自回归模型，并引入了Omni-DNA，这是一种参数范围从2000万到1亿的跨模态多任务模型。我们的方法包括两个阶段：（i）基于DNA序列进行预训练，目标是预测下一个令牌；（ii）扩展多模态任务特定令牌并同时对多个下游任务进行微调。当在Nucleotide Transformer和GB基准上进行评估时，Omni-DNA在26个任务中的18个任务上取得了最先进的性能。通过多任务微调，Omni-DNA一次性解决了10个乙酰化和甲基化任务，超越了单独为每个任务训练的模型。最后，我们设计了两个复杂的基因组任务：DNA2Function和Needle-in-DNA，分别将DNA序列映射到文本功能描述和图像，这表明Omni-DNA的跨模态能力扩展了基因组应用的范围。所有模型均通过以下链接获取：[这里](https://example.com)(请注意将链接替换为实际链接) 

---
# Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignment 

**Title (ZH)**: Ola：面向全模态语言模型的渐进模态对齐技术探究 

**Authors**: Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, Yongming Rao  

**Link**: [PDF](https://arxiv.org/pdf/2502.04328)  

**Abstract**: Recent advances in large language models, particularly following GPT-4o, have sparked increasing interest in developing omni-modal models capable of understanding more modalities. While some open-source alternatives have emerged, there is still a notable lag behind specialized single-modality models in performance. In this paper, we present Ola, an Omni-modal language model that achieves competitive performance across image, video, and audio understanding compared to specialized counterparts. The core design of Ola lies in its progressive modality alignment strategy that extends the supporting modality of the language model progressively. Our training pipeline begins with the most distinct modalities: image and text, then gradually expands the skill sets of the model using speech data that connects language and audio knowledge, and video data that connects all modalities. The progressive learning pipeline also enables us to maintain a relatively small size of the cross-modal alignment data, making developing omni-modal from existing vision-language models easy and less costly. Moreover, to unlock an advanced interactive experience like GPT-4o, we further design a sentence-wise decoding solution for streaming speech generation. Extensive experiments demonstrate that Ola surpasses existing open omni-modal LLMs across all modalities while achieving highly competitive performance compared to state-of-the-art specialized models of similar sizes. We aim to make Ola a fully open omni-modal understanding solution to advance future research in this emerging field. Model weights, code, and data are open-sourced at this https URL. 

**Abstract (ZH)**: 近年来，尤其是GPT-4之后，大型语言模型的发展取得了重要进展，激发了对能够理解多种模态的全能模型的兴趣。尽管已经出现了一些开源替代方案，但在性能上仍落后于专门的单模态模型。本文介绍了Ola，这是一种全能语言模型，在图像、视频和音频理解方面均达到了与专门模型相当的竞争力。Ola的核心设计在于其渐进模态对齐策略，能够逐步扩展语言模型的支持模态。训练管道从最不同的模态——图像和文本开始，然后逐步通过连接语言和音频知识的语音数据以及连接所有模态的视频数据，扩展模型的能力。这种渐进的学习管道还使得我们能够保持跨模态对齐数据相对较小的规模，从而使得从现有的视觉-语言模型开发全能模型变得容易且成本较低。此外，为了实现与GPT-4o相媲美的高级互动体验，我们还设计了一种逐句解码解决方案以支持流式语音生成。广泛实验表明，Ola在所有模态上都超过了现有的全能大型语言模型，同时在性能上与相似规模的顶级专门模型竞争激烈。我们希望将Ola打造成为一种全面开放的全能理解解决方案，以促进该新兴领域未来的研究。模型权重、代码和数据均在此处开源：[此链接]。 

---
# MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation 

**Title (ZH)**: MRAMG-Bench：一个超越文本的多模态检索增强多模态生成基准测试 

**Authors**: Qinhan Yu, Zhiyou Xiao, Binghui Li, Zhengren Wang, Chong Chen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04176)  

**Abstract**: Recent advancements in Retrieval-Augmented Generation (RAG) have shown remarkable performance in enhancing response accuracy and relevance by integrating external knowledge into generative models. However, existing RAG methods primarily focus on providing text-only answers, even in multimodal retrieval-augmented generation scenarios. In this work, we introduce the Multimodal Retrieval-Augmented Multimodal Generation (MRAMG) task, which aims to generate answers that combine both text and images, fully leveraging the multimodal data within a corpus. Despite the importance of this task, there is a notable absence of a comprehensive benchmark to effectively evaluate MRAMG performance. To bridge this gap, we introduce the MRAMG-Bench, a carefully curated, human-annotated dataset comprising 4,346 documents, 14,190 images, and 4,800 QA pairs, sourced from three categories: Web Data, Academic Papers, and Lifestyle. The dataset incorporates diverse difficulty levels and complex multi-image scenarios, providing a robust foundation for evaluating multimodal generation tasks. To facilitate rigorous evaluation, our MRAMG-Bench incorporates a comprehensive suite of both statistical and LLM-based metrics, enabling a thorough analysis of the performance of popular generative models in the MRAMG task. Besides, we propose an efficient multimodal answer generation framework that leverages both LLMs and MLLMs to generate multimodal responses. Our datasets are available at: this https URL. 

**Abstract (ZH)**: 最近在检索增强生成（RAG）领域的进展显著提升了生成模型通过整合外部知识以提高响应准确性和相关性的能力。然而，现有的RAG方法主要集中在提供纯文本答案，即使在多模态检索增强生成的场景中也是如此。本文中，我们引入了多模态检索增强多模态生成（MRAMG）任务，旨在生成结合文本和图像的答案，充分挖掘语料库中的多模态数据。尽管该任务的重要性不言而喻，但仍缺乏一个全面的基准来有效评估MRAMG的性能。为解决这一问题，我们引入了MRAMG-Bench数据集，这是一个精心策划、由人工标注的包含4,346份文档、14,190张图片和4,800个问答对的数据集，来源自三个类别：网络数据、学术论文和生活方式。该数据集涵盖了多种难度级别和复杂的多图像场景，为评估多模态生成任务提供了坚实的基础。为了方便严格的评估，我们的MRAMG-Bench引入了一系列全面的统计和LLM基评估指标，能够全面分析流行生成模型在MRAMG任务中的性能。此外，我们还提出了一种高效的多模态答案生成框架，充分利用LLM和MLLM生成多模态响应。我们的数据集可在以下链接获取：[这里](this https URL)。 

---
