# RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering 

**Title (ZH)**: RAMQA：一种统一的检索增强多模态问答框架 

**Authors**: Yang Bai, Christan Earl Grant, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13297)  

**Abstract**: Multi-modal retrieval-augmented Question Answering (MRAQA), integrating text and images, has gained significant attention in information retrieval (IR) and natural language processing (NLP). Traditional ranking methods rely on small encoder-based language models, which are incompatible with modern decoder-based generative large language models (LLMs) that have advanced various NLP tasks. To bridge this gap, we propose RAMQA, a unified framework combining learning-to-rank methods with generative permutation-enhanced ranking techniques. We first train a pointwise multi-modal ranker using LLaVA as the backbone. Then, we apply instruction tuning to train a LLaMA model for re-ranking the top-k documents using an innovative autoregressive multi-task learning approach. Our generative ranking model generates re-ranked document IDs and specific answers from document candidates in various permutations. Experiments on two MRAQA benchmarks, WebQA and MultiModalQA, show significant improvements over strong baselines, highlighting the effectiveness of our approach. Code and data are available at: this https URL 

**Abstract (ZH)**: 多模态检索增强的问答（MRAQA），结合文本和图像，已在信息检索（IR）和自然语言处理（NLP）领域引起了广泛关注。传统的排名方法依赖于小型的基于编码器的语言模型，这些模型与现代基于解码器的生成型大型语言模型（LLMs）不兼容，后者已在多种NLP任务中取得进展。为了弥补这一差距，我们提出了一种结合了学习到排名方法和生成型排列增强排名技术的统一框架RAMQA。我们首先使用LLaVA作为骨干训练一个点wise多模态排名器。然后，我们通过创新的自回归多任务学习方法对LLaMA模型进行指令微调，以对前k个文档进行重新排名。我们的生成型排名模型从文档候选集中生成重新排排名的文档ID和特定答案的各种排列。在两个MRAQA基准数据集WebQA和MultiModalQA上的实验表明，我们的方法在强基线方法上取得了显著的改进，突显了我们方法的有效性。相关代码和数据可在以下链接获取：this https URL 

---
# The Breeze 2 Herd of Models: Traditional Chinese LLMs Based on Llama with Vision-Aware and Function-Calling Capabilities 

**Title (ZH)**: 《轻风2模型 herd：基于LLama的傳統中文大型语言模型，具有视觉意识和函数调用能力》

注释：
1. "The Breeze 2 Herd of Models" 被翻译为 "轻风2模型 herd"，这里的 "herd" 是指一组模型或多个模型的集合。
2. "Traditional Chinese LLMs" 被翻译为 "基于LLama的傳統中文大型语言模型"，其中“LLM”通常指的是“大型语言模型”。
3. "Vision-Aware" 被翻译为 "视觉意识"，这指的是模型具有处理视觉信息的能力。
4. "Function-Calling Capabilities" 被翻译为 "函数调用能力"，这指的是模型能够调用外部函数或执行特定任务的能力。 

**Authors**: Chan-Jan Hsu, Chia-Sheng Liu, Meng-Hsi Chen, Muxi Chen, Po-Chun Hsu, Yi-Chang Chen, Da-Shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13921)  

**Abstract**: Breeze 2 is a suite of advanced multi-modal language models, available in 3B and 8B parameter configurations, specifically designed to enhance Traditional Chinese language representation. Building upon the Llama 3, Breeze 2 continues pretraining on an extensive corpus to enhance the linguistic and cultural heritage of Traditional Chinese. It incorporates vision-aware capabilities through a visual encoder and a bridge module, and supports function-calling via prompt templates and post-training on function-calling data. The effectiveness of Breeze 2 is benchmarked across various tasks, including Taiwan general knowledge, instruction-following, long context, function calling, and vision understanding. Furthermore, we showcase the capabilities of the its 3B model in a mobile application. We are publicly releasing all Breeze 2 models under the Llama 3 Community License. 

**Abstract (ZH)**: Breeze 2 是一套高级多模态语言模型，提供 3B 和 8B 参数配置，专门设计用于增强传统中文语言表示。基于 Llama 3，Breeze 2 继续在广泛的语料库上进行预训练，以增强传统中文的语言和文化内涵。它通过视觉编码器和桥接模块集成了视觉感知能力，并通过提示模板和函数调用数据的后续训练支持函数调用。Breeze 2 的效果在各类任务中进行了基准测试，包括台湾常识、指令遵循、长文本上下文、函数调用和视觉理解。此外，我们展示了其 3B 模型在移动应用程序中的能力。我们将根据 Llama 3 社区许可协议公开发布所有 Breeze 2 模型。 

---
# LVPruning: An Effective yet Simple Language-Guided Vision Token Pruning Approach for Multi-modal Large Language Models 

**Title (ZH)**: LVPruning：一种有效且简单的基于语言指导的视觉Token剪枝方法，用于多模态大规模语言模型 

**Authors**: Yizheng Sun, Yanze Xin, Hao Li, Jingyuan Sun, Chenghua Lin, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2501.13652)  

**Abstract**: Multi-modal Large Language Models (MLLMs) have achieved remarkable success by integrating visual and textual modalities. However, they incur significant computational overhead due to the large number of vision tokens processed, limiting their practicality in resource-constrained environments. We introduce Language-Guided Vision Token Pruning (LVPruning) for MLLMs, an effective yet simple method that significantly reduces the computational burden while preserving model performance. LVPruning employs cross-attention modules to compute the importance of vision tokens based on their interaction with language tokens, determining which to prune. Importantly, LVPruning can be integrated without modifying the original MLLM parameters, which makes LVPruning simple to apply or remove. Our experiments show that LVPruning can effectively reduce up to 90% of vision tokens by the middle layer of LLaVA-1.5, resulting in a 62.1% decrease in inference Tera Floating-Point Operations Per Second (TFLOPs), with an average performance loss of just 0.45% across nine multi-modal benchmarks. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过整合视觉和文本模态取得了显著的成功。然而，由于需要处理大量的视觉标记，它们产生了显著的计算开销，限制了在资源受限环境中的实用性。我们提出了一种名为语言引导的视觉标记剪枝（LVPruning）的方法，这是一种有效且简单的策略，能够在大幅减少计算负担的同时保持模型性能。LVPruning 利用交叉注意力模块根据视觉标记与语言标记之间的交互计算视觉标记的重要性，从而决定要剪枝哪些标记。重要的是，LVPruning 可以在不修改原始 MLLM 参数的情况下进行集成，这使得 LVPruning 简单易行。我们的实验表明，LVPruning 可以有效减少 LLaVA-1.5 中间层最多 90% 的视觉标记，导致推理吞吐量（TFLOPs）降低 62.1%，同时在九个多模态基准测试中的平均性能损失仅为 0.45%。 

---
# IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models 

**Title (ZH)**: IMAGINE-E：先进文本到图像模型的图像生成智能评估 

**Authors**: Jiayi Lei, Renrui Zhang, Xiangfei Hu, Weifeng Lin, Zhen Li, Wenjian Sun, Ruoyi Du, Le Zhuo, Zhongyu Li, Xinyue Li, Shitian Zhao, Ziyu Guo, Yiting Lu, Peng Gao, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13920)  

**Abstract**: With the rapid development of diffusion models, text-to-image(T2I) models have made significant progress, showcasing impressive abilities in prompt following and image generation. Recently launched models such as FLUX.1 and Ideogram2.0, along with others like Dall-E3 and Stable Diffusion 3, have demonstrated exceptional performance across various complex tasks, raising questions about whether T2I models are moving towards general-purpose applicability. Beyond traditional image generation, these models exhibit capabilities across a range of fields, including controllable generation, image editing, video, audio, 3D, and motion generation, as well as computer vision tasks like semantic segmentation and depth estimation. However, current evaluation frameworks are insufficient to comprehensively assess these models' performance across expanding domains. To thoroughly evaluate these models, we developed the IMAGINE-E and tested six prominent models: FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, and Jimeng. Our evaluation is divided into five key domains: structured output generation, realism, and physical consistency, specific domain generation, challenging scenario generation, and multi-style creation tasks. This comprehensive assessment highlights each model's strengths and limitations, particularly the outstanding performance of FLUX.1 and Ideogram2.0 in structured and specific domain tasks, underscoring the expanding applications and potential of T2I models as foundational AI tools. This study provides valuable insights into the current state and future trajectory of T2I models as they evolve towards general-purpose usability. Evaluation scripts will be released at this https URL. 

**Abstract (ZH)**: 随着扩散模型的飞速发展，文本到图像（T2I）模型取得了显著进步，展示了其在指令跟随和图像生成方面的出色能力。最近推出的模型，如FLUX.1和Ideogram2.0，以及其他模型如Dall-E3和Stable Diffusion 3，已经在各种复杂任务中展现了卓越性能，引发了关于T2I模型是否正朝着通用适用性发展的讨论。除了传统的图像生成外，这些模型还在可控生成、图像编辑、视频、音频、3D和运动生成，以及语义分割和深度估计等计算机视觉任务中表现出广泛的适用能力。然而，当前的评估框架尚不足以全面评估这些模型在不断扩展领域的性能。为了全面评估这些模型，我们开发了IMAGINE-E框架，并测试了六种突出的模型：FLUX.1、Ideogram2.0、Midjourney、Dall-E3、Stable Diffusion 3和Jimeng。我们的评估分为五个关键领域：结构化输出生成、真实性、物理一致性、特定领域生成、具有挑战性的场景生成以及多风格生成任务。这项全面的评估突显了每个模型的优势和局限性，特别是在FLUX.1和Ideogram2.0在结构化和特定领域任务中的卓越表现，强调了T2I模型作为基础AI工具的广泛应用前景和潜力。本研究为T2I模型当前状态及其向通用适用性发展的未来路径提供了宝贵的见解。评估脚本将在此链接中公布：[此 https URL]。 

---
# Temporal Preference Optimization for Long-Form Video Understanding 

**Title (ZH)**: 长视频理解中的时间偏好优化 

**Authors**: Rui Li, Xiaohan Wang, Yuhui Zhang, Zeyu Wang, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2501.13919)  

**Abstract**: Despite significant advancements in video large multimodal models (video-LMMs), achieving effective temporal grounding in long-form videos remains a challenge for existing models. To address this limitation, we propose Temporal Preference Optimization (TPO), a novel post-training framework designed to enhance the temporal grounding capabilities of video-LMMs through preference learning. TPO adopts a self-training approach that enables models to differentiate between well-grounded and less accurate temporal responses by leveraging curated preference datasets at two granularities: localized temporal grounding, which focuses on specific video segments, and comprehensive temporal grounding, which captures extended temporal dependencies across entire video sequences. By optimizing on these preference datasets, TPO significantly enhances temporal understanding while reducing reliance on manually annotated data. Extensive experiments on three long-form video understanding benchmarks--LongVideoBench, MLVU, and Video-MME--demonstrate the effectiveness of TPO across two state-of-the-art video-LMMs. Notably, LLaVA-Video-TPO establishes itself as the leading 7B model on the Video-MME benchmark, underscoring the potential of TPO as a scalable and efficient solution for advancing temporal reasoning in long-form video understanding. Project page: this https URL. 

**Abstract (ZH)**: 尽管在视频大规模多模态模型（视频-LMMs）方面取得了显著进展，但现有模型在长视频中实现有效的时空定位仍然面临挑战。为解决这一限制，我们提出了一种新颖的后训练框架——时空偏好优化（TPO），旨在通过偏好学习提升视频-LMMs的时空定位能力。TPO 采用自我训练的方法，使模型能够通过利用按两种粒度层次策化的偏好数据集来区分精确的和不准确的时空响应：局部时空定位专注于特定视频片段，而全面时空定位则捕捉整个视频序列中的扩展时空依赖关系。通过在这些偏好数据集上进行优化，TPO 显著提升了时空理解能力，减少了对手动标注数据的依赖。在三个长视频理解基准测试集（LongVideoBench、MLVU 和 Video-MME）上的广泛实验表明，TPO 在两个最先进的视频-LMMs 上表现出显著的有效性。特别是，LLaVA-Video-TPO 成为 Video-MME 基准测试集上的领先 7B 模型，突显了 TPO 作为提高长视频理解中时空推理能力的可扩展且高效的解决方案的潜力。项目页面：[此处链接](this https URL)。 

---
# Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos 

**Title (ZH)**: 视频-MMMU模型：评估多学科专业视频中的知识获取 

**Authors**: Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13826)  

**Abstract**: Humans acquire knowledge through three cognitive stages: perceiving information, comprehending knowledge, and adapting knowledge to solve novel problems. Videos serve as an effective medium for this learning process, facilitating a progression through these cognitive stages. However, existing video benchmarks fail to systematically evaluate the knowledge acquisition capabilities in Large Multimodal Models (LMMs). To address this gap, we introduce Video-MMMU, a multi-modal, multi-disciplinary benchmark designed to assess LMMs' ability to acquire and utilize knowledge from videos. Video-MMMU features a curated collection of 300 expert-level videos and 900 human-annotated questions across six disciplines, evaluating knowledge acquisition through stage-aligned question-answer pairs: Perception, Comprehension, and Adaptation. A proposed knowledge gain metric, {\Delta}knowledge, quantifies improvement in performance after video viewing. Evaluation of LMMs reveals a steep decline in performance as cognitive demands increase and highlights a significant gap between human and model knowledge acquisition, underscoring the need for methods to enhance LMMs' capability to learn and adapt from videos. 

**Abstract (ZH)**: 人类通过三个认知阶段获取知识：感知信息、理解知识以及将知识应用于解决新问题。视频作为一种有效的媒介，促进了这一学习过程，帮助依次完成这些认知阶段。然而，现有的视频基准未能系统地评估大型多模态模型（LMMs）的知识获取能力。为解决这一问题，我们引入了Video-MMMU，这是一个多模态、跨学科的基准，用于评估LMMs从视频中获取和利用知识的能力。Video-MMMU 包含了涵盖六个学科领域的300个专家级视频和900个人标注问题的精选集合，并通过阶段对齐的问题-答案对评估知识获取：感知、理解以及应用。我们还提出了一种知识增益度量{\(\Delta\)knowledge}，用于量化视频观看后性能的提升。对LMMs的评估揭示了随着认知需求的增加，其性能显著下降，并指出了人类与模型之间在知识获取能力上的显著差距，强调了需要开发增强LMMs从视频中学习和适应能力的方法。 

---
# Explainable XR: Understanding User Behaviors of XR Environments using LLM-assisted Analytics Framework 

**Title (ZH)**: 可解释的XR：使用LLM辅助分析框架理解XR环境中的用户行为 

**Authors**: Yoonsang Kim, Zainab Aamir, Mithilesh Singh, Saeed Boorboor, Klaus Mueller, Arie E. Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2501.13778)  

**Abstract**: We present Explainable XR, an end-to-end framework for analyzing user behavior in diverse eXtended Reality (XR) environments by leveraging Large Language Models (LLMs) for data interpretation assistance. Existing XR user analytics frameworks face challenges in handling cross-virtuality - AR, VR, MR - transitions, multi-user collaborative application scenarios, and the complexity of multimodal data. Explainable XR addresses these challenges by providing a virtuality-agnostic solution for the collection, analysis, and visualization of immersive sessions. We propose three main components in our framework: (1) A novel user data recording schema, called User Action Descriptor (UAD), that can capture the users' multimodal actions, along with their intents and the contexts; (2) a platform-agnostic XR session recorder, and (3) a visual analytics interface that offers LLM-assisted insights tailored to the analysts' perspectives, facilitating the exploration and analysis of the recorded XR session data. We demonstrate the versatility of Explainable XR by demonstrating five use-case scenarios, in both individual and collaborative XR applications across virtualities. Our technical evaluation and user studies show that Explainable XR provides a highly usable analytics solution for understanding user actions and delivering multifaceted, actionable insights into user behaviors in immersive environments. 

**Abstract (ZH)**: 我们提出了可解释的扩展现实（Explainable XR），这是一个端到端框架，利用大语言模型（LLMs）进行数据解释辅助，以分析在多种扩展现实（XR）环境中的用户行为。现有的XR用户分析框架在处理AR、VR、MR之间的转换、多用户协作应用场景以及多模态数据的复杂性时存在挑战。Explainable XR通过提供一种与虚拟现实无关的解决方案来解决这些挑战，该解决方案用于收集、分析和可视化沉浸式会话。我们提出了框架中的三个主要组件：（1）一种新的用户数据记录模式，称为用户动作描述符（UAD），能够捕捉用户的多模态行为及其意图和上下文；（2）一个平台无关的XR会话记录器；以及（3）一种基于大语言模型的视觉分析界面，提供定制化的分析洞见，以便分析师探索和分析记录的XR会话数据。我们通过在不同虚拟现实环境中的个体和协作XR应用中展示五个用例场景，展示了Explainable XR的灵活性。我们的技术评估和用户研究显示，Explainable XR提供了一个高度可用的分析解决方案，用于理解和提供关于沉浸式环境中用户行为多方面的、可操作的洞察。 

---
# ReasVQA: Advancing VideoQA with Imperfect Reasoning Process 

**Title (ZH)**: ReasVQA：改进推理过程的视频问答研究 

**Authors**: Jianxin Liang, Xiaojun Meng, Huishuai Zhang, Yueqian Wang, Jiansheng Wei, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.13536)  

**Abstract**: Video Question Answering (VideoQA) is a challenging task that requires understanding complex visual and temporal relationships within videos to answer questions accurately. In this work, we introduce \textbf{ReasVQA} (Reasoning-enhanced Video Question Answering), a novel approach that leverages reasoning processes generated by Multimodal Large Language Models (MLLMs) to improve the performance of VideoQA models. Our approach consists of three phases: reasoning generation, reasoning refinement, and learning from reasoning. First, we generate detailed reasoning processes using additional MLLMs, and second refine them via a filtering step to ensure data quality. Finally, we use the reasoning data, which might be in an imperfect form, to guide the VideoQA model via multi-task learning, on how to interpret and answer questions based on a given video. We evaluate ReasVQA on three popular benchmarks, and our results establish new state-of-the-art performance with significant improvements of +2.9 on NExT-QA, +7.3 on STAR, and +5.9 on IntentQA. Our findings demonstrate the supervising benefits of integrating reasoning processes into VideoQA. Further studies validate each component of our method, also with different backbones and MLLMs, and again highlight the advantages of this simple but effective method. We offer a new perspective on enhancing VideoQA performance by utilizing advanced reasoning techniques, setting a new benchmark in this research field. 

**Abstract (ZH)**: 视频问答（Video Question Answering, VideoQA）是一项具有挑战性的任务，要求理解视频中的复杂视觉和时间关系以准确回答问题。在本工作中，我们提出了一种新的方法 \textbf{ReasVQA}（增强推理的视频问答），该方法利用多模态大型语言模型（MLLMs）生成的推理过程来提高VideoQA模型的性能。我们的方法分为三个阶段：推理生成、推理优化以及从推理中学习。首先，我们使用额外的MLLM生成详细的推理过程，然后通过筛选步骤进一步优化推理以确保数据质量。最后，我们利用这些可能不完美的推理数据通过多任务学习来指导VideoQA模型，使其如何根据给定的视频理解和回答问题。我们使用ReasVQA在三个流行的基准测试上进行评估，实验结果表明，其在NExT-QA、STAR和IntentQA三个基准上的表现均显著优于现有方法，分别提高了2.9%、7.3%和5.9%。我们的研究结果证实了在VideoQA中整合推理过程的监督优势。进一步的研究验证了我们方法中每个组件的优势，并在不同的骨干网络和MLLM下再次突显了此方法的简单而有效的优点。我们提出了一种新的视角，即通过利用先进的推理技术来增强VideoQA性能，并为该研究领域设立了新的基准。 

---
# Toyteller: AI-powered Visual Storytelling Through Toy-Playing with Character Symbols 

**Title (ZH)**: Toyteller：通过角色符号玩偶叙事的AI驱动视觉故事讲述 

**Authors**: John Joon Young Chung, Melissa Roemmele, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2501.13284)  

**Abstract**: We introduce Toyteller, an AI-powered storytelling system where users generate a mix of story text and visuals by directly manipulating character symbols like they are toy-playing. Anthropomorphized symbol motions can convey rich and nuanced social interactions; Toyteller leverages these motions (1) to let users steer story text generation and (2) as a visual output format that accompanies story text. We enabled motion-steered text generation and text-steered motion generation by mapping motions and text onto a shared semantic space so that large language models and motion generation models can use it as a translational layer. Technical evaluations showed that Toyteller outperforms a competitive baseline, GPT-4o. Our user study identified that toy-playing helps express intentions difficult to verbalize. However, only motions could not express all user intentions, suggesting combining it with other modalities like language. We discuss the design space of toy-playing interactions and implications for technical HCI research on human-AI interaction. 

**Abstract (ZH)**: 我们将介绍Toyteller，这是一个基于AI的故事讲述系统，用户可以通过直接操纵类似于玩具的字符符号来生成文字和视觉内容的混合。拟人化的符号动作可以传达丰富而细腻的社会互动；Toyteller通过这些动作（1）让用户控制故事文字的生成，并且（2）作为与故事文字伴随的视觉输出格式。我们通过将动作和文字映射到共享的语义空间，实现了基于动作的文字生成和基于文字的动作生成。技术评估表明，Toyteller在多个指标上优于竞争baseline模型GPT-4o。我们的用户研究发现，玩具玩耍有助于表达难以用语言描述的意图。然而，仅靠动作无法完全表达所有用户意图，这表明需要结合其他模态（例如语言）。我们讨论了玩具玩耍交互的设计空间及其对人类-AI交互技术HCI研究的潜在影响。 

---
# Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization 

**Title (ZH)**: 面向鲁棒多模态开放集测试时自适应调整的自适应熵感知优化 

**Authors**: Hao Dong, Eleni Chatzi, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2501.13924)  

**Abstract**: Test-time adaptation (TTA) has demonstrated significant potential in addressing distribution shifts between training and testing data. Open-set test-time adaptation (OSTTA) aims to adapt a source pre-trained model online to an unlabeled target domain that contains unknown classes. This task becomes more challenging when multiple modalities are involved. Existing methods have primarily focused on unimodal OSTTA, often filtering out low-confidence samples without addressing the complexities of multimodal data. In this work, we present Adaptive Entropy-aware Optimization (AEO), a novel framework specifically designed to tackle Multimodal Open-set Test-time Adaptation (MM-OSTTA) for the first time. Our analysis shows that the entropy difference between known and unknown samples in the target domain strongly correlates with MM-OSTTA performance. To leverage this, we propose two key components: Unknown-aware Adaptive Entropy Optimization (UAE) and Adaptive Modality Prediction Discrepancy Optimization (AMP). These components enhance the ability of model to distinguish unknown class samples during online adaptation by amplifying the entropy difference between known and unknown samples. To thoroughly evaluate our proposed methods in the MM-OSTTA setting, we establish a new benchmark derived from existing datasets. This benchmark includes two downstream tasks and incorporates five modalities. Extensive experiments across various domain shift situations demonstrate the efficacy and versatility of the AEO framework. Additionally, we highlight the strong performance of AEO in long-term and continual MM-OSTTA settings, both of which are challenging and highly relevant to real-world applications. Our source code is available at this https URL. 

**Abstract (ZH)**: 测试时适应（Test-time Adaptation, TTA）已经在解决训练数据和测试数据分布差异的问题上展现了显著潜力。开放集测试时适应（Open-set Test-time Adaptation, OSTTA）旨在将源预训练模型在线适应一个包含未知类别的未标记目标领域。当多种模态同时存在时，这一任务变得更加具有挑战性。现有方法主要集中在单模态OSTTA上，经常是过滤掉低置信度样本，但并没有解决多模态数据的复杂性问题。在本工作中，我们提出了一种新的框架——自适应熵感知优化（Adaptive Entropy-aware Optimization, AEO），该框架首次专门针对多模态开放集测试时适应（Multimodal Open-set Test-time Adaptation, MM-OSTTA）任务。我们的分析表明，在目标领域中已知和未知样本的熵差与MM-OSTTA性能之间存在强烈的相关性。为了利用这一相关性，我们提出了两个关键组件：感知未知的自适应熵优化（Unknown-aware Adaptive Entropy Optimization, UAE）和自适应模态预测不一致性优化（Adaptive Modality Prediction Discrepancy Optimization, AMP）。这些组件通过放大已知和未知样本之间的熵差，增强了模型在在线适应过程中的区分未知类别样本的能力。为了全面评估我们所提出的方法在MM-OSTTA设置下的性能，我们基于现有的数据集构建了一个新的基准。这个基准包含了两个下游任务和五种模态。在各种领域转移情况下的大量实验证明了AEO框架的有效性和灵活性。此外，我们还强调了AEO在长期内存和持续多模态开放集测试时适应（long-term and continual MM-OSTTA）设置下的出色性能，这两种设置在现实世界应用中有很高的相关性和挑战性。我们的源代码可在以下链接获取：this https URL。 

---
# Pix2Cap-COCO: Advancing Visual Comprehension via Pixel-Level Captioning 

**Title (ZH)**: Pix2Cap-COCO：基于像素级描述推进视觉理解 

**Authors**: Zuyao You, Junke Wang, Lingyu Kong, Bo He, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13893)  

**Abstract**: We present Pix2Cap-COCO, the first panoptic pixel-level caption dataset designed to advance fine-grained visual understanding. To achieve this, we carefully design an automated annotation pipeline that prompts GPT-4V to generate pixel-aligned, instance-specific captions for individual objects within images, enabling models to learn more granular relationships between objects and their contexts. This approach results in 167,254 detailed captions, with an average of 22.94 words per caption. Building on Pix2Cap-COCO, we introduce a novel task, panoptic segmentation-captioning, which challenges models to recognize instances in an image and provide detailed descriptions for each simultaneously. To benchmark this task, we design a robust baseline based on X-Decoder. The experimental results demonstrate that Pix2Cap-COCO is a particularly challenging dataset, as it requires models to excel in both fine-grained visual understanding and detailed language generation. Furthermore, we leverage Pix2Cap-COCO for Supervised Fine-Tuning (SFT) on large multimodal models (LMMs) to enhance their performance. For example, training with Pix2Cap-COCO significantly improves the performance of GPT4RoI, yielding gains in CIDEr +1.4%, ROUGE +0.4%, and SPICE +0.5% on Visual Genome dataset, and strengthens its region understanding ability on the ViP-BENCH, with an overall improvement of +5.1%, including notable increases in recognition accuracy +11.2% and language generation quality +22.2%. 

**Abstract (ZH)**: 我们提出了Pix2Cap-COCO，这是首个旨在促进精细视觉理解的全景像素级描述数据集。为了实现这一目标，我们精心设计了一个自动化标注管道，使用GPT-4V自动生成与像素对齐、实例特定的图像中标记对象的描述，从而使模型能够学习对象与其上下文之间的更精细关系。这一方法产生了167,254条详细的描述，平均每条描述包含22.94个单词。基于Pix2Cap-COCO，我们引入了一种新的任务——全景分割-描述，该任务挑战模型同时在图像中标识实例并提供详细描述。为了衡量这一任务，我们基于X-Decoder设计了一个稳健的基线模型。实验结果表明，Pix2Cap-COCO 是一个特别具有挑战性的数据集，因为它要求模型在精细视觉理解和详细语言生成方面均表现出色。此外，我们利用Pix2Cap-COCO 对大型多模态模型（LMMs）进行监督微调（SFT），以提升其性能。例如，使用Pix2Cap-COCO 进行训练显著提升了GPT4RoI 的性能，在Visual Genome 数据集上，CIDEr 提高了1.4%，ROUGE 提高了0.4%，SPICE 提高了0.5%；同时，其在ViP-BENCH 的区域理解能力也得到了增强，总体提升幅度为5.1%，包括识别准确率的显著提高（+11.2%）和语言生成质量的大幅提升（+22.2%）。 

---
# Tune In, Act Up: Exploring the Impact of Audio Modality-Specific Edits on Large Audio Language Models in Jailbreak 

**Title (ZH)**: 调频响应，积极行动：探索特定音频模态编辑对囚徒突破的大规模音频语言模型影响 

**Authors**: Erjia Xiao, Hao Cheng, Jing Shao, Jinhao Duan, Kaidi Xu, Le Yang, Jindong Gu, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13772)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable zero-shot performance across various natural language processing tasks. The integration of multimodal encoders extends their capabilities, enabling the development of Multimodal Large Language Models that process vision, audio, and text. However, these capabilities also raise significant security concerns, as these models can be manipulated to generate harmful or inappropriate content through jailbreak. While extensive research explores the impact of modality-specific input edits on text-based LLMs and Large Vision-Language Models in jailbreak, the effects of audio-specific edits on Large Audio-Language Models (LALMs) remain underexplored. Hence, this paper addresses this gap by investigating how audio-specific edits influence LALMs inference regarding jailbreak. We introduce the Audio Editing Toolbox (AET), which enables audio-modality edits such as tone adjustment, word emphasis, and noise injection, and the Edited Audio Datasets (EADs), a comprehensive audio jailbreak benchmark. We also conduct extensive evaluations of state-of-the-art LALMs to assess their robustness under different audio edits. This work lays the groundwork for future explorations on audio-modality interactions in LALMs security. 

**Abstract (ZH)**: 大型语言模型（Large Language Models, LLMs）在各种自然语言处理任务中展现出卓越的零样本性能。通过集成多模态编码器，可以进一步扩展其功能，使开发出能够处理视觉、音频和文本信息的多模态大型语言模型成为可能。然而，这些能力也引发了显著的安全关切，因为这些模型可以通过“脱狱”（jailbreak）被操纵以生成有害或不适当的内容。虽然大量的研究探讨了特定模态输入编辑对文本型LLMs和大型视觉-语言模型“脱狱”影响，但特定于音频的编辑对大型音频-语言模型（Large Audio-Language Models, LALMs）的影响仍较少被研究。因此，本文通过探讨特定于音频的编辑如何影响LALMs在“脱狱”情况下的推断来填补这一空白。我们介绍了音频编辑工具箱（Audio Editing Toolbox, AET），它允许进行音调调整、词汇强调和噪声注入等音频模态编辑，并介绍了一个综合的音频“脱狱”基准数据集（Edited Audio Datasets, EADs）。此外，我们还进行了广泛评估，以评估当前最先进的LALMs在不同音频编辑下的鲁棒性。这项工作为未来在LALMs安全方面探索音频模态交互奠定了基础。 

---
# EventVL: Understand Event Streams via Multimodal Large Language Model 

**Title (ZH)**: 事件VL：通过多模态大型语言模型理解事件流 

**Authors**: Pengteng Li, Yunfan Lu, Pinghao Song, Wuyang Li, Huizai Yao, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.13707)  

**Abstract**: The event-based Vision-Language Model (VLM) recently has made good progress for practical vision tasks. However, most of these works just utilize CLIP for focusing on traditional perception tasks, which obstruct model understanding explicitly the sufficient semantics and context from event streams. To address the deficiency, we propose EventVL, the first generative event-based MLLM (Multimodal Large Language Model) framework for explicit semantic understanding. Specifically, to bridge the data gap for connecting different modalities semantics, we first annotate a large event-image/video-text dataset, containing almost 1.4 million high-quality pairs of data, which enables effective learning across various scenes, e.g., drive scene or human motion. After that, we design Event Spatiotemporal Representation to fully explore the comprehensive information by diversely aggregating and segmenting the event stream. To further promote a compact semantic space, Dynamic Semantic Alignment is introduced to improve and complete sparse semantic spaces of events. Extensive experiments show that our EventVL can significantly surpass existing MLLM baselines in event captioning and scene description generation tasks. We hope our research could contribute to the development of the event vision community. 

**Abstract (ZH)**: 基于事件的视觉-语言模型（VLM）近期在实际视觉任务中取得了良好进展。然而，大多数这些工作仅仅利用CLIP来专注于传统的感知任务，这阻碍了模型从事件流中明确理解足够的语义和上下文。为了解决这一缺陷，我们提出了一种名为EventVL的生成型事件基多模大型语言模型（Multimodal Large Language Model, MLLM）框架，旨在实现显式的语义理解。具体而言，为了弥合跨不同模态语义的数据缺口，我们首先标注了一个包含近140万高质量数据对的大规模事件-图像/视频-文本数据集，这些数据使模型能够有效地在各种场景中学习，例如驾驶场景或人体动作。在此基础上，我们设计了一种事件时空表示方法，通过多样化的事件流聚合与分割，充分探索全面的信息。为进一步促进紧凑的语义空间，我们引入了动态语义对齐，以改善和补充事件的稀疏语义空间。通过广泛的实验，我们发现我们的EventVL在事件描述和场景描述生成任务中显著超越了现有的MLLM基线模型。我们希望我们的研究能够为事件视觉社区的发展做出贡献。 

---
# Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge 

**Title (ZH)**: Streaming视频理解与记忆增强知识的多轮交互 

**Authors**: Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13468)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled the development of Video-LLMs, advancing multimodal learning by bridging video data with language tasks. However, current video understanding models struggle with processing long video sequences, supporting multi-turn dialogues, and adapting to real-world dynamic scenarios. To address these issues, we propose StreamChat, a training-free framework for streaming video reasoning and conversational interaction. $\StreamChat$ leverages a novel hierarchical memory system to efficiently process and compress video features over extended sequences, enabling real-time, multi-turn dialogue. Our framework incorporates a parallel system scheduling strategy that enhances processing speed and reduces latency, ensuring robust performance in real-world applications. Furthermore, we introduce StreamBench, a versatile benchmark that evaluates streaming video understanding across diverse media types and interactive scenarios, including multi-turn interactions and complex reasoning tasks. Extensive evaluations on StreamBench and other public benchmarks demonstrate that StreamChat significantly outperforms existing state-of-the-art models in terms of accuracy and response times, confirming its effectiveness for streaming video understanding. Code is available at StreamChat: this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展推动了视频大型语言模型（Video-LLMs）的出现，通过将视频数据与语言任务相结合，促进了多模态学习的发展。然而，当前的视频理解模型在处理长视频序列、支持多轮对话以及适应真实世界的动态场景方面存在困难。为解决这些问题，我们提出了一种无需训练的框架——StreamChat，用于流式视频推理和对话交互。$\StreamChat$ 利用了一种新颖的分层记忆系统，可以在长时间序列中有效处理和压缩视频特征，从而实现实时多轮对话。我们的框架采用并行系统调度策略，提高了处理速度并降低了延迟，确保了在实际应用中的稳健性能。此外，我们引入了StreamBench，这是一种多功能基准，用于评估不同媒体类型和交互场景下的流式视频理解，包括多轮交互和复杂推理任务。在StreamBench和其他公共基准上的广泛评估表明，StreamChat在准确性和响应时间方面显著优于现有的最先进模型，验证了其在流式视频理解方面的有效性。代码可在以下链接获取：StreamChat: [这里提供链接] 

---
# M3PT: A Transformer for Multimodal, Multi-Party Social Signal Prediction with Person-aware Blockwise Attention 

**Title (ZH)**: M3PT：一种基于人员意识分块注意机制的多模态多党社会信号预测变换器 

**Authors**: Yiming Tang, Abrar Anwar, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2501.13416)  

**Abstract**: Understanding social signals in multi-party conversations is important for human-robot interaction and artificial social intelligence. Multi-party interactions include social signals like body pose, head pose, speech, and context-specific activities like acquiring and taking bites of food when dining. Incorporating all the multimodal signals in a multi-party interaction is difficult, and past work tends to build task-specific models for predicting social signals. In this work, we address the challenge of predicting multimodal social signals in multi-party settings in a single model. We introduce M3PT, a causal transformer architecture with modality and temporal blockwise attention masking which allows for the simultaneous processing of multiple social cues across multiple participants and their temporal interactions. This approach better captures social dynamics over time by considering longer horizons of social signals between individuals. We train and evaluate our unified model on the Human-Human Commensality Dataset (HHCD), and demonstrate that using multiple modalities improves bite timing and speaking status prediction. Source code: this https URL 

**Abstract (ZH)**: 理解多 Vaults 情景中的社会信号对于人类-机器人交互和人工智能社会智能具有重要意义。多 Vaults 交互包括像身体姿态、头部姿态、言语以及就餐时获取和咀嚼食物等具体上下文活动中的社会信号。在多 Vaults 交互中整合所有多模态信号是具有一定挑战性的，之前的大部分工作都倾向于为预测社会信号构建特定任务的模型。在本项工作中，我们克服了在多 Vaults 设置中单个模型预测多模态社会信号的挑战。我们提出了 M3PT，一种因果转换器架构，具有模态和时间块状注意掩蔽机制，这使得可以同时处理来自多个参与者及其时间交互的社会提示。这种方法通过考虑个体之间更长时段的社会信号动态，更好地捕捉了社会动力学。我们使用 Human-Human Commensality Dataset (HHCD) 训练和评估了我们的统一模型，并证明使用多种模态可以提高咬食时机和说话状态的预测效果。源代码：[此处链接]

注意："Vaults" 在原文中应为 "Parties"，这里根据语境进行了修正。"Commensality" 指的是共享食物的行为或场合，"Dataset" 是数据集的意思。原文链接替换为具体链接。 

---
