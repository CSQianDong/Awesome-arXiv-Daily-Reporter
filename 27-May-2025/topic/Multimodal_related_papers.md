# MangaVQA and MangaLMM: A Benchmark and Specialized Model for Multimodal Manga Understanding 

**Authors**: Jeonghun Baek, Kazuki Egashira, Shota Onohara, Atsuyuki Miyai, Yuki Imajuku, Hikaru Ikuta, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.20298)  

**Abstract**: Manga, or Japanese comics, is a richly multimodal narrative form that blends images and text in complex ways. Teaching large multimodal models (LMMs) to understand such narratives at a human-like level could help manga creators reflect on and refine their stories. To this end, we introduce two benchmarks for multimodal manga understanding: MangaOCR, which targets in-page text recognition, and MangaVQA, a novel benchmark designed to evaluate contextual understanding through visual question answering. MangaVQA consists of 526 high-quality, manually constructed question-answer pairs, enabling reliable evaluation across diverse narrative and visual scenarios. Building on these benchmarks, we develop MangaLMM, a manga-specialized model finetuned from the open-source LMM Qwen2.5-VL to jointly handle both tasks. Through extensive experiments, including comparisons with proprietary models such as GPT-4o and Gemini 2.5, we assess how well LMMs understand manga. Our benchmark and model provide a comprehensive foundation for evaluating and advancing LMMs in the richly narrative domain of manga. 

---
# Visual Abstract Thinking Empowers Multimodal Reasoning 

**Authors**: Dairu Liu, Ziyue Wang, Minyuan Ruan, Fuwen Luo, Chi Chen, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20164)  

**Abstract**: Images usually convey richer detail than text, but often include redundant information which potentially downgrades multimodal reasoning performance. When faced with lengthy or complex messages, humans tend to employ abstract thinking to convert them into simple and concise abstracts. Inspired by this cognitive strategy, we introduce Visual Abstract Thinking (VAT), a novel thinking paradigm that prompts Multimodal Large Language Models (MLLMs) with visual abstract instead of explicit verbal thoughts or elaborate guidance, permitting a more concentrated visual reasoning mechanism. Explicit thinking, such as Chain-of-thought (CoT) or tool-augmented approaches, increases the complexity of reasoning process via inserting verbose intermediate steps, external knowledge or visual information. In contrast, VAT reduces redundant visual information and encourages models to focus their reasoning on more essential visual elements. Experimental results show that VAT consistently empowers different models, and achieves an average gain of 17% over GPT-4o baseline by employing diverse types of visual abstracts, demonstrating that VAT can enhance visual reasoning abilities for MLLMs regarding conceptual, structural and relational reasoning tasks. VAT is also compatible with CoT in knowledge-intensive multimodal reasoning tasks. These findings highlight the effectiveness of visual reasoning via abstract thinking and encourage further exploration of more diverse reasoning paradigms from the perspective of human cognition. 

---
# ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs 

**Authors**: Pooneh Mousavi, Yingzhi Wang, Mirco Ravanelli, Cem Subakan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19937)  

**Abstract**: Large Language Models (LLMs) are widely used in Spoken Language Understanding (SLU). Recent SLU models process audio directly by adapting speech input into LLMs for better multimodal learning. A key consideration for these models is the cross-modal alignment between text and audio modalities, which is a telltale sign as to whether or not LLM is able to associate semantic meaning to audio segments. While various methods exist for fusing these modalities, there is no standard metric to evaluate alignment quality in LLMs. In this work, we propose a new metric, ALAS (Automatic Latent Alignment Score). Our study examines the correlation between audio and text representations across transformer layers, for two different tasks (Spoken Question Answering and Emotion Recognition). We showcase that our metric behaves as expected across different layers and different tasks. 

---
# DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset 

**Authors**: Alkis Koudounas, Moreno La Quatra, Elena Baralis  

**Link**: [PDF](https://arxiv.org/pdf/2505.19978)  

**Abstract**: Recent advances in conversational AI have demonstrated impressive capabilities in single-turn responses, yet multi-turn dialogues remain challenging for even the most sophisticated language models. Current dialogue datasets are limited in their emotional range, domain diversity, turn depth, and are predominantly text-only, hindering progress in developing more human-like conversational systems across modalities. To address these limitations, we present DeepDialogue, a large-scale multimodal dataset containing 40,150 high-quality multi-turn dialogues spanning 41 domains and incorporating 20 distinct emotions with coherent emotional progressions. Our approach pairs 9 different language models (4B-72B parameters) to generate 65,600 initial conversations, which we then evaluate through a combination of human annotation and LLM-based quality filtering. The resulting dataset reveals fundamental insights: smaller models fail to maintain coherence beyond 6 dialogue turns; concrete domains (e.g., "cars," "travel") yield more meaningful conversations than abstract ones (e.g., "philosophy"); and cross-model interactions produce more coherent dialogues than same-model conversations. A key contribution of DeepDialogue is its speech component, where we synthesize emotion-consistent voices for all 40,150 dialogues, creating the first large-scale open-source multimodal dialogue dataset that faithfully preserves emotional context across multi-turn conversations. 

---
# T^2Agent A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search 

**Authors**: Xing Cui, Yueying Zou, Zekun Li, Peipei Li, Xinyuan Xu, Xuannan Liu, Huaibo Huang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2505.19768)  

**Abstract**: Real-world multimodal misinformation often arises from mixed forgery sources, requiring dynamic reasoning and adaptive verification. However, existing methods mainly rely on static pipelines and limited tool usage, limiting their ability to handle such complexity and diversity. To address this challenge, we propose T2Agent, a novel misinformation detection agent that incorporates an extensible toolkit with Monte Carlo Tree Search (MCTS). The toolkit consists of modular tools such as web search, forgery detection, and consistency analysis. Each tool is described using standardized templates, enabling seamless integration and future expansion. To avoid inefficiency from using all tools simultaneously, a Bayesian optimization-based selector is proposed to identify a task-relevant subset. This subset then serves as the action space for MCTS to dynamically collect evidence and perform multi-source verification. To better align MCTS with the multi-source nature of misinformation detection, T2Agent extends traditional MCTS with multi-source verification, which decomposes the task into coordinated subtasks targeting different forgery sources. A dual reward mechanism containing a reasoning trajectory score and a confidence score is further proposed to encourage a balance between exploration across mixed forgery sources and exploitation for more reliable evidence. We conduct ablation studies to confirm the effectiveness of the tree search mechanism and tool usage. Extensive experiments further show that T2Agent consistently outperforms existing baselines on challenging mixed-source multimodal misinformation benchmarks, demonstrating its strong potential as a training-free approach for enhancing detection accuracy. The code will be released. 

---
# Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs 

**Authors**: Hao Fang, Changle Zhou, Jiawei Kong, Kuofeng Gao, Bin Chen, Tao Liang, Guojun Ma, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.19678)  

**Abstract**: Large Vision-Language Models (LVLMs) are susceptible to hallucinations, where generated responses seem semantically plausible yet exhibit little or no relevance to the input image. Previous studies reveal that this issue primarily stems from LVLMs' over-reliance on language priors while disregarding the visual information during decoding. To alleviate this issue, we introduce a novel Conditional Pointwise Mutual Information (C-PMI) calibrated decoding strategy, which adaptively strengthens the mutual dependency between generated texts and input images to mitigate hallucinations. Unlike existing methods solely focusing on text token sampling, we propose to jointly model the contributions of visual and textual tokens to C-PMI, formulating hallucination mitigation as a bi-level optimization problem aimed at maximizing mutual information. To solve it, we design a token purification mechanism that dynamically regulates the decoding process by sampling text tokens remaining maximally relevant to the given image, while simultaneously refining image tokens most pertinent to the generated response. Extensive experiments across various benchmarks reveal that the proposed method significantly reduces hallucinations in LVLMs while preserving decoding efficiency. 

---
# MT$^{3}$: Scaling MLLM-based Text Image Machine Translation via Multi-Task Reinforcement Learning 

**Authors**: Zhaopeng Feng, Yupu Liang, Shaosheng Cao, Jiayuan Su, Jiahan Ren, Zhe Xu, Yao Hu, Wenxuan Huang, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19714)  

**Abstract**: Text Image Machine Translation (TIMT)-the task of translating textual content embedded in images-is critical for applications in accessibility, cross-lingual information access, and real-world document understanding. However, TIMT remains a complex challenge due to the need for accurate optical character recognition (OCR), robust visual-text reasoning, and high-quality translation, often requiring cascading multi-stage pipelines. Recent advances in large-scale Reinforcement Learning (RL) have improved reasoning in Large Language Models (LLMs) and Multimodal LLMs (MLLMs), but their application to end-to-end TIMT is still underexplored. To bridge this gap, we introduce MT$^{3}$, the first framework to apply Multi-Task RL to MLLMs for end-to-end TIMT. MT$^{3}$ adopts a multi-task optimization paradigm targeting three key sub-skills: text recognition, context-aware reasoning, and translation. It is trained using a novel multi-mixed reward mechanism that adapts rule-based RL strategies to TIMT's intricacies, offering fine-grained, non-binary feedback across tasks. Furthermore, to facilitate the evaluation of TIMT in authentic cross-cultural and real-world social media contexts, we introduced XHSPost, the first social media TIMT benchmark. Our MT$^{3}$-7B-Zero achieves state-of-the-art results on the latest in-domain MIT-10M benchmark, outperforming strong baselines such as Qwen2.5-VL-72B and InternVL2.5-78B by notable margins across multiple metrics. Additionally, the model shows strong generalization to out-of-distribution language pairs and datasets. In-depth analyses reveal how multi-task synergy, reinforcement learning initialization, curriculum design, and reward formulation contribute to advancing MLLM-driven TIMT. 

---
# ChartLens: Fine-grained Visual Attribution in Charts 

**Authors**: Manan Suri, Puneet Mathur, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2505.19360)  

**Abstract**: The growing capabilities of multimodal large language models (MLLMs) have advanced tasks like chart understanding. However, these models often suffer from hallucinations, where generated text sequences conflict with the provided visual data. To address this, we introduce Post-Hoc Visual Attribution for Charts, which identifies fine-grained chart elements that validate a given chart-associated response. We propose ChartLens, a novel chart attribution algorithm that uses segmentation-based techniques to identify chart objects and employs set-of-marks prompting with MLLMs for fine-grained visual attribution. Additionally, we present ChartVA-Eval, a benchmark with synthetic and real-world charts from diverse domains like finance, policy, and economics, featuring fine-grained attribution annotations. Our evaluations show that ChartLens improves fine-grained attributions by 26-66%. 

---
# DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding 

**Authors**: Yunhai Hu, Tianhua Xia, Zining Liu, Rahul Raman, Xingyu Liu, Bo Bao, Eric Sather, Vithursan Thangarasa, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19201)  

**Abstract**: Speculative decoding (SD) has emerged as a powerful method for accelerating autoregressive generation in large language models (LLMs), yet its integration into vision-language models (VLMs) remains underexplored. We introduce DREAM, a novel speculative decoding framework tailored for VLMs that combines three key innovations: (1) a cross-attention-based mechanism to inject intermediate features from the target model into the draft model for improved alignment, (2) adaptive intermediate feature selection based on attention entropy to guide efficient draft model training, and (3) visual token compression to reduce draft model latency. DREAM enables efficient, accurate, and parallel multimodal decoding with significant throughput improvement. Experiments across a diverse set of recent popular VLMs, including LLaVA, Pixtral, SmolVLM and Gemma3, demonstrate up to 3.6x speedup over conventional decoding and significantly outperform prior SD baselines in both inference throughput and speculative draft acceptance length across a broad range of multimodal benchmarks. The code is publicly available at: this https URL 

---
# ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning 

**Authors**: Yeyuan Wang, Dehong Gao, Rujiao Long, Lei Yi, Linbo Jin, Libin Yang, Xiaoyan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.19100)  

**Abstract**: Direct Preference Optimization (DPO) has gained significant attention for its simplicity and computational efficiency in aligning large language models (LLMs). Recent advancements have extended DPO to multimodal scenarios, achieving strong performance. However, traditional DPO relies on binary preference optimization, rewarding or penalizing entire responses without considering fine-grained segment correctness, leading to suboptimal solutions. The root of this issue lies in the absence of fine-grained supervision during the optimization process. To address this, we propose Adaptive Sentence-level Preference Optimization (ASPO), which evaluates individual sentences for more precise preference optimization. By dynamically calculating adaptive rewards at the sentence level based on model predictions, ASPO enhances response content assessment without additional models or parameters. This significantly improves the alignment of multimodal features. Extensive experiments show that ASPO substantially enhances the overall performance of multimodal models. 

---
# ReadBench: Measuring the Dense Text Visual Reading Ability of Vision-Language Models 

**Authors**: Benjamin Clavié, Florian Brand  

**Link**: [PDF](https://arxiv.org/pdf/2505.19091)  

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs), have greatly enhanced their capability to jointly process text and images. However, despite extensive benchmarks evaluating visual comprehension (e.g., diagrams, color schemes, OCR tasks...), there is limited assessment of VLMs' ability to read and reason about text-rich images effectively. To fill this gap, we introduce ReadBench, a multimodal benchmark specifically designed to evaluate the reading comprehension capabilities of VLMs. ReadBench transposes contexts from established text-only benchmarks into images of text while keeping textual prompts and questions intact. Evaluating leading VLMs with ReadBench, we find minimal-but-present performance degradation on short, text-image inputs, while performance sharply declines for longer, multi-page contexts. Our experiments further reveal that text resolution has negligible effects on multimodal performance. These findings highlight needed improvements in VLMs, particularly their reasoning over visually presented extensive textual content, a capability critical for practical applications. ReadBench is available at this https URL . 

---
# Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation 

**Authors**: Jiwan Chung, Junhyeok Kim, Siyeol Kim, Jaeyoung Lee, Min Soo Kim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18842)  

**Abstract**: We present v1, a lightweight extension to Multimodal Large Language Models (MLLMs) that enables selective visual revisitation during inference. While current MLLMs typically consume visual input only once and reason purely over internal memory, v1 introduces a simple point-and-copy mechanism that allows the model to dynamically retrieve relevant image regions throughout the reasoning process. This mechanism augments existing architectures with minimal modifications, enabling contextual access to visual tokens based on the model's evolving hypotheses. To train this capability, we construct v1g, a dataset of 300K multimodal reasoning traces with interleaved visual grounding annotations. Experiments on three multimodal mathematical reasoning benchmarks -- MathVista, MathVision, and MathVerse -- demonstrate that v1 consistently improves performance over comparable baselines, particularly on tasks requiring fine-grained visual reference and multi-step reasoning. Our results suggest that dynamic visual access is a promising direction for enhancing grounded multimodal reasoning. Code, models, and data will be released to support future research. 

---
# StandUp4AI: A New Multilingual Dataset for Humor Detection in Stand-up Comedy Videos 

**Authors**: Valentin Barriere, Nahuel Gomez, Leo Hemamou, Sofia Callejas, Brian Ravenet  

**Link**: [PDF](https://arxiv.org/pdf/2505.18903)  

**Abstract**: Aiming towards improving current computational models of humor detection, we propose a new multimodal dataset of stand-up comedies, in seven languages: English, French, Spanish, Italian, Portuguese, Hungarian and Czech. Our dataset of more than 330 hours, is at the time of writing the biggest available for this type of task, and the most diverse. The whole dataset is automatically annotated in laughter (from the audience), and the subpart left for model validation is manually annotated. Contrary to contemporary approaches, we do not frame the task of humor detection as a binary sequence classification, but as word-level sequence labeling, in order to take into account all the context of the sequence and to capture the continuous joke tagging mechanism typically occurring in natural conversations. As par with unimodal baselines results, we propose a method for e propose a method to enhance the automatic laughter detection based on Audio Speech Recognition errors. Our code and data are available online: this https URL 

---
# Audio Jailbreak Attacks: Exposing Vulnerabilities in SpeechGPT in a White-Box Framework 

**Authors**: Binhao Ma, Hanqing Guo, Zhengping Jay Luo, Rui Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18864)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced the naturalness and flexibility of human computer interaction by enabling seamless understanding across text, vision, and audio modalities. Among these, voice enabled models such as SpeechGPT have demonstrated considerable improvements in usability, offering expressive, and emotionally responsive interactions that foster deeper connections in real world communication scenarios. However, the use of voice introduces new security risks, as attackers can exploit the unique characteristics of spoken language, such as timing, pronunciation variability, and speech to text translation, to craft inputs that bypass defenses in ways not seen in text-based systems. Despite substantial research on text based jailbreaks, the voice modality remains largely underexplored in terms of both attack strategies and defense mechanisms. In this work, we present an adversarial attack targeting the speech input of aligned MLLMs in a white box scenario. Specifically, we introduce a novel token level attack that leverages access to the model's speech tokenization to generate adversarial token sequences. These sequences are then synthesized into audio prompts, which effectively bypass alignment safeguards and to induce prohibited outputs. Evaluated on SpeechGPT, our approach achieves up to 89 percent attack success rate across multiple restricted tasks, significantly outperforming existing voice based jailbreak methods. Our findings shed light on the vulnerabilities of voice-enabled multimodal systems and to help guide the development of more robust next-generation MLLMs. 

---
# MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation 

**Authors**: Woohyun Cho, Youngmin Kim, Sunghyun Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18614)  

**Abstract**: Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation. 

---
# From Generation to Detection: A Multimodal Multi-Task Dataset for Benchmarking Health Misinformation 

**Authors**: Zhihao Zhang, Yiran Zhang, Xiyue Zhou, Liting Huang, Imran Razzak, Preslav Nakov, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2505.18685)  

**Abstract**: Infodemics and health misinformation have significant negative impact on individuals and society, exacerbating confusion and increasing hesitancy in adopting recommended health measures. Recent advancements in generative AI, capable of producing realistic, human like text and images, have significantly accelerated the spread and expanded the reach of health misinformation, resulting in an alarming surge in its dissemination. To combat the infodemics, most existing work has focused on developing misinformation datasets from social media and fact checking platforms, but has faced limitations in topical coverage, inclusion of AI generation, and accessibility of raw content. To address these issues, we present MM Health, a large scale multimodal misinformation dataset in the health domain consisting of 34,746 news article encompassing both textual and visual information. MM Health includes human-generated multimodal information (5,776 articles) and AI generated multimodal information (28,880 articles) from various SOTA generative AI models. Additionally, We benchmarked our dataset against three tasks (reliability checks, originality checks, and fine-grained AI detection) demonstrating that existing SOTA models struggle to accurately distinguish the reliability and origin of information. Our dataset aims to support the development of misinformation detection across various health scenarios, facilitating the detection of human and machine generated content at multimodal levels. 

---
# Flex-Judge: Think Once, Judge Anywhere 

**Authors**: Jongwoo Ko, Sungnyun Kim, Sungwoo Cho, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.18601)  

**Abstract**: Human-generated reward signals are critical for aligning generative models with human preferences, guiding both training and inference-time evaluations. While large language models (LLMs) employed as proxy evaluators, i.e., LLM-as-a-Judge, significantly reduce the costs associated with manual annotations, they typically require extensive modality-specific training data and fail to generalize well across diverse multimodal tasks. In this paper, we propose Flex-Judge, a reasoning-guided multimodal judge model that leverages minimal textual reasoning data to robustly generalize across multiple modalities and evaluation formats. Our core intuition is that structured textual reasoning explanations inherently encode generalizable decision-making patterns, enabling an effective transfer to multimodal judgments, e.g., with images or videos. Empirical results demonstrate that Flex-Judge, despite being trained on significantly fewer text data, achieves competitive or superior performance compared to state-of-the-art commercial APIs and extensively trained multimodal evaluators. Notably, Flex-Judge presents broad impact in modalities like molecule, where comprehensive evaluation benchmarks are scarce, underscoring its practical value in resource-constrained domains. Our framework highlights reasoning-based text supervision as a powerful, cost-effective alternative to traditional annotation-intensive approaches, substantially advancing scalable multimodal model-as-a-judge. 

---
# Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models 

**Authors**: Haoyuan Sun, Jiaqi Wu, Bo Xia, Yifu Luo, Yifei Zhao, Kai Qin, Xufei Lv, Tiantian Zhang, Yongzhe Chang, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18536)  

**Abstract**: Standing in 2025, at a critical juncture in the pursuit of Artificial General Intelligence (AGI), reinforcement fine-tuning (RFT) has demonstrated significant potential in enhancing the reasoning capability of large language models (LLMs) and has led to the development of cutting-edge AI models such as OpenAI-o1 and DeepSeek-R1. Moreover, the efficient application of RFT to enhance the reasoning capability of multimodal large language models (MLLMs) has attracted widespread attention from the community. In this position paper, we argue that reinforcement fine-tuning powers the reasoning capability of multimodal large language models. To begin with, we provide a detailed introduction to the fundamental background knowledge that researchers interested in this field should be familiar with. Furthermore, we meticulously summarize the improvements of RFT in powering reasoning capability of MLLMs into five key points: diverse modalities, diverse tasks and domains, better training algorithms, abundant benchmarks and thriving engineering frameworks. Finally, we propose five promising directions for future research that the community might consider. We hope that this position paper will provide valuable insights to the community at this pivotal stage in the advancement toward AGI. Summary of works done on RFT for MLLMs is available at this https URL. 

---
# BRIT: Bidirectional Retrieval over Unified Image-Text Graph 

**Authors**: Ainulla Khan, Yamada Moyuru, Srinidhi Akella  

**Link**: [PDF](https://arxiv.org/pdf/2505.18450)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising technique to enhance the quality and relevance of responses generated by large language models. While recent advancements have mainly focused on improving RAG for text-based queries, RAG on multi-modal documents containing both texts and images has not been fully explored. Especially when fine-tuning does not work. This paper proposes BRIT, a novel multi-modal RAG framework that effectively unifies various text-image connections in the document into a multi-modal graph and retrieves the texts and images as a query-specific sub-graph. By traversing both image-to-text and text-to-image paths in the graph, BRIT retrieve not only directly query-relevant images and texts but also further relevant contents to answering complex cross-modal multi-hop questions. To evaluate the effectiveness of BRIT, we introduce MM-RAG test set specifically designed for multi-modal question answering tasks that require to understand the text-image relations. Our comprehensive experiments demonstrate the superiority of BRIT, highlighting its ability to handle cross-modal questions on the multi-modal documents. 

---
# DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding 

**Authors**: Yue Jiang, Jichu Li, Yang Liu, Dingkang Yang, Feng Zhou, Quyu Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18411)  

**Abstract**: We introduce DanmakuTPPBench, a comprehensive benchmark designed to advance multi-modal Temporal Point Process (TPP) modeling in the era of Large Language Models (LLMs). While TPPs have been widely studied for modeling temporal event sequences, existing datasets are predominantly unimodal, hindering progress in models that require joint reasoning over temporal, textual, and visual information. To address this gap, DanmakuTPPBench comprises two complementary components: (1) DanmakuTPP-Events, a novel dataset derived from the Bilibili video platform, where user-generated bullet comments (Danmaku) naturally form multi-modal events annotated with precise timestamps, rich textual content, and corresponding video frames; (2) DanmakuTPP-QA, a challenging question-answering dataset constructed via a novel multi-agent pipeline powered by state-of-the-art LLMs and multi-modal LLMs (MLLMs), targeting complex temporal-textual-visual reasoning. We conduct extensive evaluations using both classical TPP models and recent MLLMs, revealing significant performance gaps and limitations in current methods' ability to model multi-modal event dynamics. Our benchmark establishes strong baselines and calls for further integration of TPP modeling into the multi-modal language modeling landscape. The code and dataset have been released at this https URL 

---
# VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction 

**Authors**: Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, Rakesh Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20279)  

**Abstract**: The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. Nevertheless, achieving deep spatial understanding comparable to human capabilities poses significant challenges in model encoding and data acquisition. Existing methods frequently depend on external depth sensors for geometry capture or utilize off-the-shelf algorithms for pre-constructing 3D maps, thereby limiting their scalability, especially with prevalent monocular video inputs and for time-sensitive applications. In this work, we introduce VLM-3R, a unified framework for Vision-Language Models (VLMs) that incorporates 3D Reconstructive instruction tuning. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Leveraging our Spatial-Visual-View Fusion and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables monocular 3D spatial assistance and embodied reasoning. To facilitate the evaluation of temporal reasoning, we introduce the Vision-Spatial-Temporal Intelligence benchmark, featuring over 138.6K QA pairs across five distinct tasks focused on evolving spatial relationships. Extensive experiments demonstrate that our model, VLM-3R, not only facilitates robust visual-spatial reasoning but also enables the understanding of temporal 3D context changes, excelling in both accuracy and scalability. 

---
# Hard Negative Contrastive Learning for Fine-Grained Geometric Understanding in Large Multimodal Models 

**Authors**: Kai Sun, Yushi Bai, Zhen Yang, Jiajie Zhang, Ji Qi, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20152)  

**Abstract**: Benefiting from contrastively trained visual encoders on large-scale natural scene images, Large Multimodal Models (LMMs) have achieved remarkable performance across various visual perception tasks. However, the inherent limitations of contrastive learning upon summarized descriptions fundamentally restrict the capabilities of models in meticulous reasoning, particularly in crucial scenarios of geometric problem-solving. To enhance geometric understanding, we propose a novel hard negative contrastive learning framework for the vision encoder, which combines image-based contrastive learning using generation-based hard negatives created by perturbing diagram generation code, and text-based contrastive learning using rule-based negatives derived from modified geometric descriptions and retrieval-based negatives selected based on caption similarity. We train CLIP using our strong negative learning method, namely MMCLIP (Multimodal Math CLIP), and subsequently train an LMM for geometric problem-solving. Experiments show that our trained model, MMGeoLM, significantly outperforms other open-source models on three geometric reasoning benchmarks. Even with a size of 7B, it can rival powerful closed-source models like GPT-4o. We further study the impact of different negative sample construction methods and the number of negative samples on the geometric reasoning performance of LMM, yielding fruitful conclusions. The code and dataset are available at this https URL. 

---
# Taming LLMs with Negative Samples: A Reference-Free Framework to Evaluate Presentation Content with Actionable Feedback 

**Authors**: Ananth Muppidi, Tarak Das, Sambaran Bandyopadhyay, Tripti Shukla, Dharun D A  

**Link**: [PDF](https://arxiv.org/pdf/2505.18240)  

**Abstract**: The generation of presentation slides automatically is an important problem in the era of generative AI. This paper focuses on evaluating multimodal content in presentation slides that can effectively summarize a document and convey concepts to a broad audience. We introduce a benchmark dataset, RefSlides, consisting of human-made high-quality presentations that span various topics. Next, we propose a set of metrics to characterize different intrinsic properties of the content of a presentation and present REFLEX, an evaluation approach that generates scores and actionable feedback for these metrics. We achieve this by generating negative presentation samples with different degrees of metric-specific perturbations and use them to fine-tune LLMs. This reference-free evaluation technique does not require ground truth presentations during inference. Our extensive automated and human experiments demonstrate that our evaluation approach outperforms classical heuristic-based and state-of-the-art large language model-based evaluations in generating scores and explanations. 

---
# Multimodal LLM-Guided Semantic Correction in Text-to-Image Diffusion 

**Authors**: Zheqi Lv, Junhao Chen, Qi Tian, Keting Yin, Shengyu Zhang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20053)  

**Abstract**: Diffusion models have become the mainstream architecture for text-to-image generation, achieving remarkable progress in visual quality and prompt controllability. However, current inference pipelines generally lack interpretable semantic supervision and correction mechanisms throughout the denoising process. Most existing approaches rely solely on post-hoc scoring of the final image, prompt filtering, or heuristic resampling strategies-making them ineffective in providing actionable guidance for correcting the generative trajectory. As a result, models often suffer from object confusion, spatial errors, inaccurate counts, and missing semantic elements, severely compromising prompt-image alignment and image quality. To tackle these challenges, we propose MLLM Semantic-Corrected Ping-Pong-Ahead Diffusion (PPAD), a novel framework that, for the first time, introduces a Multimodal Large Language Model (MLLM) as a semantic observer during inference. PPAD performs real-time analysis on intermediate generations, identifies latent semantic inconsistencies, and translates feedback into controllable signals that actively guide the remaining denoising steps. The framework supports both inference-only and training-enhanced settings, and performs semantic correction at only extremely few diffusion steps, offering strong generality and scalability. Extensive experiments demonstrate PPAD's significant improvements. 

---
# From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data 

**Authors**: Chun-Yi Kuan, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20166)  

**Abstract**: Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. However, this adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where important textual capabilities such as instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about their reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making the process resource-intensive. To address these issues, we leverage the backbone LLMs from ALLMs to synthesize general-purpose caption-style alignment data. We refer to this process as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Building on BALSa, we introduce LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method designed to improve ALLMs' ability to distinguish between present and absent sounds. We further extend BALSa to multi-audio scenarios, where the model either explains the differences between audio inputs or produces a unified caption that describes them all, thereby enhancing audio-language alignment. Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance in audio understanding, reasoning, and instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to the development of ALLMs. 

---
# Visualized Text-to-Image Retrieval 

**Authors**: Di Wu, Yixin Wan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20291)  

**Abstract**: We propose Visualize-then-Retrieve (VisRet), a new paradigm for Text-to-Image (T2I) retrieval that mitigates the limitations of cross-modal similarity alignment of existing multi-modal embeddings. VisRet first projects textual queries into the image modality via T2I generation. Then, it performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Experiments on three knowledge-intensive T2I retrieval benchmarks, including a newly introduced multi-entity benchmark, demonstrate that VisRet consistently improves T2I retrieval by 24.5% to 32.7% NDCG@10 across different embedding models. VisRet also significantly benefits downstream visual question answering accuracy when used in retrieval-augmented generation pipelines. The method is plug-and-play and compatible with off-the-shelf retrievers, making it an effective module for knowledge-intensive multi-modal systems. Our code and the new benchmark are publicly available at this https URL. 

---
# Multi-modal brain encoding models for multi-modal stimuli 

**Authors**: Subba Reddy Oota, Khushbu Pahwa, Mounika Marreddy, Maneesh Singh, Manish Gupta, Bapi S. Raju  

**Link**: [PDF](https://arxiv.org/pdf/2505.20027)  

**Abstract**: Despite participants engaging in unimodal stimuli, such as watching images or silent videos, recent work has demonstrated that multi-modal Transformer models can predict visual brain activity impressively well, even with incongruent modality representations. This raises the question of how accurately these multi-modal models can predict brain activity when participants are engaged in multi-modal stimuli. As these models grow increasingly popular, their use in studying neural activity provides insights into how our brains respond to such multi-modal naturalistic stimuli, i.e., where it separates and integrates information across modalities through a hierarchy of early sensory regions to higher cognition. We investigate this question by using multiple unimodal and two types of multi-modal models-cross-modal and jointly pretrained-to determine which type of model is more relevant to fMRI brain activity when participants are engaged in watching movies. We observe that both types of multi-modal models show improved alignment in several language and visual regions. This study also helps in identifying which brain regions process unimodal versus multi-modal information. We further investigate the contribution of each modality to multi-modal alignment by carefully removing unimodal features one by one from multi-modal representations, and find that there is additional information beyond the unimodal embeddings that is processed in the visual and language regions. Based on this investigation, we find that while for cross-modal models, their brain alignment is partially attributed to the video modality; for jointly pretrained models, it is partially attributed to both the video and audio modalities. This serves as a strong motivation for the neuroscience community to investigate the interpretability of these models for deepening our understanding of multi-modal information processing in brain. 

---
# Can Visual Encoder Learn to See Arrows? 

**Authors**: Naoyuki Terashita, Yusuke Tozaki, Hideaki Omote, Congkha Nguyen, Ryosuke Nakamoto, Yuta Koreeda, Hiroaki Ozaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.19944)  

**Abstract**: The diagram is a visual representation of a relationship illustrated with edges (lines or arrows), which is widely used in industrial and scientific communication. Although recognizing diagrams is essential for vision language models (VLMs) to comprehend domain-specific knowledge, recent studies reveal that many VLMs fail to identify edges in images. We hypothesize that these failures stem from an over-reliance on textual and positional biases, preventing VLMs from learning explicit edge features. Based on this idea, we empirically investigate whether the image encoder in VLMs can learn edge representation through training on a diagram dataset in which edges are biased neither by textual nor positional information. To this end, we conduct contrastive learning on an artificially generated diagram--caption dataset to train an image encoder and evaluate its diagram-related features on three tasks: probing, image retrieval, and captioning. Our results show that the finetuned model outperforms pretrained CLIP in all tasks and surpasses zero-shot GPT-4o and LLaVA-Mistral in the captioning task. These findings confirm that eliminating textual and positional biases fosters accurate edge recognition in VLMs, offering a promising path for advancing diagram understanding. 

---
# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models 

**Authors**: Jintao Tong, Wenwei Jin, Pengda Qin, Anqi Li, Yixiong Zou, Yuhong Li, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19536)  

**Abstract**: Large vision-language models (LVLMs) excel at multimodal understanding but suffer from high computational costs due to redundant vision tokens. Existing pruning methods typically rely on single-layer attention scores to rank and prune redundant visual tokens to solve this inefficiency. However, as the interaction between tokens and layers is complicated, this raises a basic question: Is such a simple single-layer criterion sufficient to identify redundancy? To answer this question, we rethink the emergence of redundant visual tokens from a fundamental perspective: information flow, which models the interaction between tokens and layers by capturing how information moves between tokens across layers. We find (1) the CLS token acts as an information relay, which can simplify the complicated flow analysis; (2) the redundancy emerges progressively and dynamically via layer-wise attention concentration; and (3) relying solely on attention scores from single layers can lead to contradictory redundancy identification. Based on this, we propose FlowCut, an information-flow-aware pruning framework, mitigating the insufficiency of the current criterion for identifying redundant tokens and better aligning with the model's inherent behaviors. Extensive experiments show that FlowCut achieves superior results, outperforming SoTA by 1.6% on LLaVA-1.5-7B with 88.9% token reduction, and by 4.3% on LLaVA-NeXT-7B with 94.4% reduction, delivering 3.2x speed-up in the prefilling stage. Our code is available at this https URL 

---
# Towards Reliable Large Audio Language Model 

**Authors**: Ziyang Ma, Xiquan Li, Yakun Song, Wenxi Chen, Chenpeng Du, Jian Wu, Yuanzhe Chen, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19294)  

**Abstract**: Recent advancements in large audio language models (LALMs) have demonstrated impressive results and promising prospects in universal understanding and reasoning across speech, music, and general sound. However, these models still lack the ability to recognize their knowledge boundaries and refuse to answer questions they don't know proactively. While there have been successful attempts to enhance the reliability of LLMs, reliable LALMs remain largely unexplored. In this paper, we systematically investigate various approaches towards reliable LALMs, including training-free methods such as multi-modal chain-of-thought (MCoT), and training-based methods such as supervised fine-tuning (SFT). Besides, we identify the limitations of previous evaluation metrics and propose a new metric, the Reliability Gain Index (RGI), to assess the effectiveness of different reliable methods. Our findings suggest that both training-free and training-based methods enhance the reliability of LALMs to different extents. Moreover, we find that awareness of reliability is a "meta ability", which can be transferred across different audio modalities, although significant structural and content differences exist among sound, music, and speech. 

---
# Co-AttenDWG: Co-Attentive Dimension-Wise Gating and Expert Fusion for Multi-Modal Offensive Content Detection 

**Authors**: Md. Mithun Hossain, Md. Shakil Hossain, Sudipto Chaki, M. F. Mridha  

**Link**: [PDF](https://arxiv.org/pdf/2505.19010)  

**Abstract**: Multi-modal learning has become a critical research area because integrating text and image data can significantly improve performance in tasks such as classification, retrieval, and scene understanding. However, despite progress with pre-trained models, current approaches are limited by inadequate cross-modal interactions and static fusion strategies that do not fully exploit the complementary nature of different modalities. To address these shortcomings, we introduce a novel multi-modal Co-AttenDWG architecture that leverages dual-path encoding, co-attention with dimension-wise gating, and advanced expert fusion. Our approach begins by projecting text and image features into a common embedding space, where a dedicated co-attention mechanism enables simultaneous, fine-grained interactions between modalities. This mechanism is further enhanced by a dimension-wise gating network that adaptively regulates the feature contributions at the channel level, ensuring that only the most relevant information is emphasized. In parallel, dual-path encoders refine the representations by processing cross-modal information separately before an additional cross-attention layer further aligns modalities. The refined features are then aggregated via an expert fusion module that combines learned gating and self-attention to produce a robust, unified representation. We validate our approach on the MIMIC and SemEval Memotion 1.0, where experimental results demonstrate significant improvements in cross-modal alignment and state-of-the-art performance, underscoring the potential of our model for a wide range of multi-modal applications. 

---
# Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps 

**Authors**: Sicheng Feng, Song Wang, Shuyi Ouyang, Lingdong Kong, Zikai Song, Jianke Zhu, Huan Wang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18675)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved significant progress in visual tasks, including semantic scene understanding and text-image alignment, with reasoning variants enhancing performance on complex tasks involving mathematics and logic. However, their capacity for reasoning tasks involving fine-grained visual understanding remains insufficiently evaluated. To address this gap, we introduce ReasonMap, a benchmark designed to assess the fine-grained visual understanding and spatial reasoning abilities of MLLMs. ReasonMap encompasses high-resolution transit maps from 30 cities across 13 countries and includes 1,008 question-answer pairs spanning two question types and three templates. Furthermore, we design a two-level evaluation pipeline that properly assesses answer correctness and quality. Comprehensive evaluations of 15 popular MLLMs, including both base and reasoning variants, reveal a counterintuitive pattern: among open-source models, base models outperform reasoning ones, while the opposite trend is observed in closed-source models. Additionally, performance generally degrades when visual inputs are masked, indicating that while MLLMs can leverage prior knowledge to answer some questions, fine-grained visual reasoning tasks still require genuine visual perception for strong performance. Our benchmark study offers new insights into visual reasoning and contributes to investigating the gap between open-source and closed-source models. 

---
# ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation 

**Authors**: Zhen Li, Yukai Guo, Duan Li, Xinyuan Guo, Bowen Li, Lanxi Xiao, Shenyu Qiao, Jiashu Chen, Zijian Wu, Hui Zhang, Xinhuan Shu, Shixia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18668)  

**Abstract**: Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 330 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs. 

---
# Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs 

**Authors**: Sharad Duwal, Mir Nafis Sharear Shopnil, Abhishek Tyagi, Adiba Mahbub Proma  

**Link**: [PDF](https://arxiv.org/pdf/2505.18221)  

**Abstract**: Multimodal out-of-context (OOC) misinformation is misinformation that repurposes real images with unrelated or misleading captions. Detecting such misinformation is challenging because it requires resolving the context of the claim before checking for misinformation. Many current methods, including LLMs and LVLMs, do not perform this contextualization step. LLMs hallucinate in absence of context or parametric knowledge. In this work, we propose a graph-based method that evaluates the consistency between the image and the caption by constructing two graph representations: an evidence graph, derived from online textual evidence, and a claim graph, from the claim in the caption. Using graph neural networks (GNNs) to encode and compare these representations, our framework then evaluates the truthfulness of image-caption pairs. We create datasets for our graph-based method, evaluate and compare our baseline model against popular LLMs on the misinformation detection task. Our method scores $93.05\%$ detection accuracy on the evaluation set and outperforms the second-best performing method (an LLM) by $2.82\%$, making a case for smaller and task-specific methods. 

---
# DocMMIR: A Framework for Document Multi-modal Information Retrieval 

**Authors**: Zirui Li, Siwei Wu, Xingyu Wang, Yi Zhou, Yizhi Li, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.19312)  

**Abstract**: The rapid advancement of unsupervised representation learning and large-scale pre-trained vision-language models has significantly improved cross-modal retrieval tasks. However, existing multi-modal information retrieval (MMIR) studies lack a comprehensive exploration of document-level retrieval and suffer from the absence of cross-domain datasets at this granularity. To address this limitation, we introduce DocMMIR, a novel multi-modal document retrieval framework designed explicitly to unify diverse document formats and domains, including Wikipedia articles, scientific papers (arXiv), and presentation slides, within a comprehensive retrieval scenario. We construct a large-scale cross-domain multimodal benchmark, comprising 450K samples, which systematically integrates textual and visual information. Our comprehensive experimental analysis reveals substantial limitations in current state-of-the-art MLLMs (CLIP, BLIP2, SigLIP-2, ALIGN) when applied to our tasks, with only CLIP demonstrating reasonable zero-shot performance. Furthermore, we conduct a systematic investigation of training strategies, including cross-modal fusion methods and loss functions, and develop a tailored approach to train CLIP on our benchmark. This results in a +31% improvement in MRR@10 compared to the zero-shot baseline. All our data and code are released in this https URL. 

---
# Research on feature fusion and multimodal patent text based on graph attention network 

**Authors**: Zhenzhen Song, Ziwei Liu, Hongji Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20188)  

**Abstract**: Aiming at the problems of cross-modal feature fusion, low efficiency of long text modeling and lack of hierarchical semantic coherence in patent text semantic mining, this study proposes HGM-Net, a deep learning framework that integrates Hierarchical Comparative Learning (HCL), Multi-modal Graph Attention Network (M-GAT) and Multi-Granularity Sparse Attention (MSA), which builds a dynamic mask, contrast and cross-structural similarity constraints on the word, sentence and paragraph hierarchies through HCL. Contrast and cross-structural similarity constraints are constructed at the word and paragraph levels by HCL to strengthen the local semantic and global thematic consistency of patent text; M-GAT models patent classification codes, citation relations and text semantics as heterogeneous graph structures, and achieves dynamic fusion of multi-source features by cross-modal gated attention; MSA adopts a hierarchical sparsity strategy to optimize the computational efficiency of long text modeling at word, phrase, sentence and paragraph granularity. Experiments show that the framework demonstrates significant advantages over existing deep learning methods in tasks such as patent classification and similarity matching, and provides a solution with both theoretical innovation and practical value for solving the problems of patent examination efficiency improvement and technology relevance mining. 

---
# Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval 

**Authors**: Rong-Cheng Tu, Wenhao Sun, Hanzhe You, Yingjie Wang, Jiaxing Huang, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19952)  

**Abstract**: Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images given a compositional query, consisting of a reference image and a modifying text-without relying on annotated training data. Existing approaches often generate a synthetic target text using large language models (LLMs) to serve as an intermediate anchor between the compositional query and the target image. Models are then trained to align the compositional query with the generated text, and separately align images with their corresponding texts using contrastive learning. However, this reliance on intermediate text introduces error propagation, as inaccuracies in query-to-text and text-to-image mappings accumulate, ultimately degrading retrieval performance. To address these problems, we propose a novel framework by employing a Multimodal Reasoning Agent (MRA) for ZS-CIR. MRA eliminates the dependence on textual intermediaries by directly constructing triplets, <reference image, modification text, target image>, using only unlabeled image data. By training on these synthetic triplets, our model learns to capture the relationships between compositional queries and candidate images directly. Extensive experiments on three standard CIR benchmarks demonstrate the effectiveness of our approach. On the FashionIQ dataset, our method improves Average R@10 by at least 7.5\% over existing baselines; on CIRR, it boosts R@1 by 9.6\%; and on CIRCO, it increases mAP@5 by 9.5\%. 

---
# MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval 

**Authors**: Rong-Cheng Tu, Zhao Jin, Jingyi Liao, Xiao Luo, Yingjie Wang, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19707)  

**Abstract**: Existing Zero-Shot Composed Image Retrieval (ZS-CIR) methods typically train adapters that convert reference images into pseudo-text tokens, which are concatenated with the modifying text and processed by frozen text encoders in pretrained VLMs or LLMs. While this design leverages the strengths of large pretrained models, it only supervises the adapter to produce encoder-compatible tokens that loosely preserve visual semantics. Crucially, it does not directly optimize the composed query representation to capture the full intent of the composition or to align with the target semantics, thereby limiting retrieval performance, particularly in cases involving fine-grained or complex visual transformations. To address this problem, we propose MLLM-Guided VLM Fine-Tuning with Joint Inference (MVFT-JI), a novel approach that leverages a pretrained multimodal large language model (MLLM) to construct two complementary training tasks using only unlabeled images: target text retrieval taskand text-to-image retrieval task. By jointly optimizing these tasks, our method enables the VLM to inherently acquire robust compositional retrieval capabilities, supported by the provided theoretical justifications and empirical validation. Furthermore, during inference, we further prompt the MLLM to generate target texts from composed queries and compute retrieval scores by integrating similarities between (i) the composed query and candidate images, and (ii) the MLLM-generated target text and candidate images. This strategy effectively combines the VLM's semantic alignment strengths with the MLLM's reasoning capabilities. 

---
# Modality Curation: Building Universal Embeddings for Advanced Multimodal Information Retrieval 

**Authors**: Fanheng Kong, Jingyuan Zhang, Yahui Liu, Hongzhi Zhang, Shi Feng, Xiaocui Yang, Daling Wang, Yu Tian, Qi Wang, Fuzheng Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.19650)  

**Abstract**: Multimodal information retrieval (MIR) faces inherent challenges due to the heterogeneity of data sources and the complexity of cross-modal alignment. While previous studies have identified modal gaps in feature spaces, a systematic approach to address these challenges remains unexplored. In this work, we introduce UNITE, a universal framework that tackles these challenges through two critical yet underexplored aspects: data curation and modality-aware training configurations. Our work provides the first comprehensive analysis of how modality-specific data properties influence downstream task performance across diverse scenarios. Moreover, we propose Modal-Aware Masked Contrastive Learning (MAMCL) to mitigate the competitive relationships among the instances of different modalities. Our framework achieves state-of-the-art results on multiple multimodal retrieval benchmarks, outperforming existing methods by notable margins. Through extensive experiments, we demonstrate that strategic modality curation and tailored training protocols are pivotal for robust cross-modal representation learning. This work not only advances MIR performance but also provides a foundational blueprint for future research in multimodal systems. Our project is available at this https URL. 

---
# EvdCLIP: Improving Vision-Language Retrieval with Entity Visual Descriptions from Large Language Models 

**Authors**: GuangHao Meng, Sunan He, Jinpeng Wang, Tao Dai, Letian Zhang, Jieming Zhu, Qing Li, Gang Wang, Rui Zhang, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18594)  

**Abstract**: Vision-language retrieval (VLR) has attracted significant attention in both academia and industry, which involves using text (or images) as queries to retrieve corresponding images (or text). However, existing methods often neglect the rich visual semantics knowledge of entities, thus leading to incorrect retrieval results. To address this problem, we propose the Entity Visual Description enhanced CLIP (EvdCLIP), designed to leverage the visual knowledge of entities to enrich queries. Specifically, since humans recognize entities through visual cues, we employ a large language model (LLM) to generate Entity Visual Descriptions (EVDs) as alignment cues to complement textual data. These EVDs are then integrated into raw queries to create visually-rich, EVD-enhanced queries. Furthermore, recognizing that EVD-enhanced queries may introduce noise or low-quality expansions, we develop a novel, trainable EVD-aware Rewriter (EaRW) for vision-language retrieval tasks. EaRW utilizes EVD knowledge and the generative capabilities of the language model to effectively rewrite queries. With our specialized training strategy, EaRW can generate high-quality and low-noise EVD-enhanced queries. Extensive quantitative and qualitative experiments on image-text retrieval benchmarks validate the superiority of EvdCLIP on vision-language retrieval tasks. 

---
# EMAC+: Embodied Multimodal Agent for Collaborative Planning with VLM+LLM 

**Authors**: Shuang Ao, Flora D. Salim, Simon Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19905)  

**Abstract**: Although LLMs demonstrate proficiency in several text-based reasoning and planning tasks, their implementation in robotics control is constrained by significant deficiencies: (1) LLM agents are designed to work mainly with textual inputs rather than visual conditions; (2) Current multimodal agents treat LLMs as static planners, which separates their reasoning from environment dynamics, resulting in actions that do not take domain-specific knowledge into account; and (3) LLMs are not designed to learn from visual interactions, which makes it harder for them to make better policies for specific domains. In this paper, we introduce EMAC+, an Embodied Multimodal Agent that collaboratively integrates LLM and VLM via a bidirectional training paradigm. Unlike existing methods, EMAC+ dynamically refines high-level textual plans generated by an LLM using real-time feedback from a VLM executing low-level visual control tasks. We address critical limitations of previous models by enabling the LLM to internalize visual environment dynamics directly through interactive experience, rather than relying solely on static symbolic mappings. Extensive experimental evaluations on ALFWorld and RT-1 benchmarks demonstrate that EMAC+ achieves superior task performance, robustness against noisy observations, and efficient learning. We also conduct thorough ablation studies and provide detailed analyses of success and failure cases. 

---
# Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging 

**Authors**: Yongxian Wei, Runxi Cheng, Weike Jin, Enneng Yang, Li Shen, Lu Hou, Sinan Du, Chun Yuan, Xiaochun Cao, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19892)  

**Abstract**: While foundation models update slowly due to resource-intensive training requirements, domain-specific models evolve between updates. Model merging aims to combine multiple expert models into a single, more capable model, thereby reducing storage and serving costs while supporting decentralized model development. Despite its potential, previous studies have primarily focused on merging visual classification models or Large Language Models (LLMs) for code and math tasks. Multimodal Large Language Models (MLLMs), which extend the capabilities of LLMs through large-scale multimodal training, have gained traction. However, there lacks a benchmark for model merging research that clearly divides the tasks for MLLM training and evaluation. In this paper, (i) we introduce the model merging benchmark for MLLMs, which includes multiple tasks such as VQA, Geometry, Chart, OCR, and Grounding, providing both LoRA and full fine-tuning models. Moreover, we explore how model merging can combine different modalities (e.g., vision-language, audio-language, and video-language models), moving toward the Omni-language model. (ii) We implement 10 model merging algorithms on the benchmark. Furthermore, we propose a novel method that removes noise from task vectors and robustly optimizes the merged vector based on a loss defined over task vector interactions, achieving an average performance gain of 2.48%. (iii) We find that model merging offers a promising way for building improved MLLMs without requiring data training. Our results also demonstrate that the complementarity among multiple modalities outperforms individual modalities. 

---
# FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks 

**Authors**: Atsunori Moteki, Shoichi Masui, Fan Yang, Yueqi Song, Yonatan Bisk, Graham Neubig, Ikuo Kusajima, Yasuto Watanabe, Hiroyuki Ishida, Jun Takahashi, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19662)  

**Abstract**: This paper proposes FieldWorkArena, a benchmark for agentic AI targeting real-world field work. With the recent increase in demand for agentic AI, they are required to monitor and report safety and health incidents, as well as manufacturing-related incidents, that may occur in real-world work environments. Existing agentic AI benchmarks have been limited to evaluating web tasks and are insufficient for evaluating agents in real-world work environments, where complexity increases significantly. In this paper, we define a new action space that agentic AI should possess for real world work environment benchmarks and improve the evaluation function from previous methods to assess the performance of agentic AI in diverse real-world tasks. The dataset consists of videos captured on-site and documents actually used in factories and warehouses, and tasks were created based on interviews with on-site workers and managers. Evaluation results confirmed that performance evaluation considering the characteristics of Multimodal LLM (MLLM) such as GPT-4o is feasible. Additionally, the effectiveness and limitations of the proposed new evaluation method were identified. The complete dataset (HuggingFace) and evaluation program (GitHub) can be downloaded from the following website: this https URL. 

---
# Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models 

**Authors**: Jianxing Liao, Junyan Xu, Yatao Sun, Maowen Tang, Sicheng He, Jingxian Liao, Shui Yu, Yun Li, Hongguan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19490)  

**Abstract**: Designing complex computer-aided design (CAD) models is often time-consuming due to challenges such as computational inefficiency and the difficulty of generating precise models. We propose a novel language-guided framework for industrial design automation to address these issues, integrating large language models (LLMs) with computer-automated design (CAutoD).Through this framework, CAD models are automatically generated from parameters and appearance descriptions, supporting the automation of design tasks during the detailed CAD design phase. Our approach introduces three key innovations: (1) a semi-automated data annotation pipeline that leverages LLMs and vision-language large models (VLLMs) to generate high-quality parameters and appearance descriptions; (2) a Transformer-based CAD generator (TCADGen) that predicts modeling sequences via dual-channel feature aggregation; (3) an enhanced CAD modeling generation model, called CADLLM, that is designed to refine the generated sequences by incorporating the confidence scores from TCADGen. Experimental results demonstrate that the proposed approach outperforms traditional methods in both accuracy and efficiency, providing a powerful tool for automating industrial workflows and generating complex CAD models from textual prompts. The code is available at this https URL 

---
# Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models 

**Authors**: Xinmiao Hu, Chun Wang, Ruihe An, ChenYu Shao, Xiaojun Ye, Sheng Zhou, Liangcheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19474)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance in visual understanding tasks, yet they often suffer from object hallucinations--generating descriptions of objects that are inconsistent with or entirely absent from the input. This issue is closely related to dataset biases, where frequent co-occurrences of objects lead to entangled semantic representations across modalities. As a result, models may erroneously activate object representations that are commonly associated with the input but not actually present.
To address this, we propose a causality-driven disentanglement framework that mitigates hallucinations through causal intervention. Our approach includes a Causal-Driven Projector in the visual pathway and a Causal Intervention Module integrated into the final transformer layer of the language model. These components work together to reduce spurious correlations caused by biased training data.
Experimental results show that our method significantly reduces hallucinations while maintaining strong performance on multiple multimodal benchmarks. Visualization analyses further confirm improved separability of object representations.
The code is available at: this https URL 

---
# Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model 

**Authors**: Tianle Li, Jihai Zhang, Yongming Rao, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19406)  

**Abstract**: While large language models (LLMs) demonstrate strong reasoning capabilities utilizing reinforcement learning (RL) with verifiable reward, whether large vision-language models (VLMs) can directly inherit such capabilities through similar post-training strategies remains underexplored. In this work, we conduct a systematic compositional probing study to evaluate whether current VLMs trained with RL or other post-training strategies can compose capabilities across modalities or tasks under out-of-distribution conditions. We design a suite of diagnostic tasks that train models on unimodal tasks or isolated reasoning skills, and evaluate them on multimodal, compositional variants requiring skill integration. Through comparisons between supervised fine-tuning (SFT) and RL-trained models, we identify three key findings: (1) RL-trained models consistently outperform SFT on compositional generalization, demonstrating better integration of learned skills; (2) although VLMs achieve strong performance on individual tasks, they struggle to generalize compositionally under cross-modal and cross-task scenario, revealing a significant gap in current training strategies; (3) enforcing models to explicitly describe visual content before reasoning (e.g., caption-before-thinking), along with rewarding progressive vision-to-text grounding, yields notable gains. It highlights two essential ingredients for improving compositionality in VLMs: visual-to-text alignment and accurate visual grounding. Our findings shed light on the current limitations of RL-based reasoning VLM training and provide actionable insights toward building models that reason compositionally across modalities and tasks. 

---
# Sensorimotor features of self-awareness in multimodal large language models 

**Authors**: Iñaki Dellibarda Varela, Pablo Romero-Sorozabal, Diego Torricelli, Gabriel Delgado-Oleas, Jose Ignacio Serrano, Maria Dolores del Castillo Sobrino, Eduardo Rocon, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2505.19237)  

**Abstract**: Self-awareness - the ability to distinguish oneself from the surrounding environment - underpins intelligent, autonomous behavior. Recent advances in AI achieve human-like performance in tasks integrating multimodal information, particularly in large language models, raising interest in the embodiment capabilities of AI agents on nonhuman platforms such as robots. Here, we explore whether multimodal LLMs can develop self-awareness solely through sensorimotor experiences. By integrating a multimodal LLM into an autonomous mobile robot, we test its ability to achieve this capacity. We find that the system exhibits robust environmental awareness, self-recognition and predictive awareness, allowing it to infer its robotic nature and motion characteristics. Structural equation modeling reveals how sensory integration influences distinct dimensions of self-awareness and its coordination with past-present memory, as well as the hierarchical internal associations that drive self-identification. Ablation tests of sensory inputs identify critical modalities for each dimension, demonstrate compensatory interactions among sensors and confirm the essential role of structured and episodic memory in coherent reasoning. These findings demonstrate that, given appropriate sensory information about the world and itself, multimodal LLMs exhibit emergent self-awareness, opening the door to artificial embodied cognitive systems. 

---
# Improving Medical Reasoning with Curriculum-Aware Reinforcement Learning 

**Authors**: Shaohao Rui, Kaitao Chen, Weijie Ma, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19213)  

**Abstract**: Recent advances in reinforcement learning with verifiable, rule-based rewards have greatly enhanced the reasoning capabilities and out-of-distribution generalization of VLMs/LLMs, obviating the need for manually crafted reasoning chains. Despite these promising developments in the general domain, their translation to medical imaging remains limited. Current medical reinforcement fine-tuning (RFT) methods predominantly focus on close-ended VQA, thereby restricting the model's ability to engage in world knowledge retrieval and flexible task adaptation. More critically, these methods fall short of addressing the critical clinical demand for open-ended, reasoning-intensive decision-making. To bridge this gap, we introduce \textbf{MedCCO}, the first multimodal reinforcement learning framework tailored for medical VQA that unifies close-ended and open-ended data within a curriculum-driven RFT paradigm. Specifically, MedCCO is initially fine-tuned on a diverse set of close-ended medical VQA tasks to establish domain-grounded reasoning capabilities, and is then progressively adapted to open-ended tasks to foster deeper knowledge enhancement and clinical interpretability. We validate MedCCO across eight challenging medical VQA benchmarks, spanning both close-ended and open-ended settings. Experimental results show that MedCCO consistently enhances performance and generalization, achieving a 11.4\% accuracy gain across three in-domain tasks, and a 5.7\% improvement on five out-of-domain benchmarks. These findings highlight the promise of curriculum-guided RL in advancing robust, clinically-relevant reasoning in medical multimodal language models. 

---
# Style2Code: A Style-Controllable Code Generation Framework with Dual-Modal Contrastive Representation Learning 

**Authors**: Dutao Zhang, Sergey Kovalchuk, YuLong He  

**Link**: [PDF](https://arxiv.org/pdf/2505.19442)  

**Abstract**: Controllable code generation, the ability to synthesize code that follows a specified style while maintaining functionality, remains a challenging task. We propose a two-stage training framework combining contrastive learning and conditional decoding to enable flexible style control. The first stage aligns code style representations with semantic and structural features. In the second stage, we fine-tune a language model (e.g., Flan-T5) conditioned on the learned style vector to guide generation. Our method supports style interpolation and user personalization via lightweight mixing. Compared to prior work, our unified framework offers improved stylistic control without sacrificing code correctness. This is among the first approaches to combine contrastive alignment with conditional decoding for style-guided code generation. 

---
# DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving 

**Authors**: Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang, Zongzheng Zhang, Xianda Guo, Hao Sun, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19381)  

**Abstract**: Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS. 

---
# ScreenExplorer: Training a Vision-Language Model for Diverse Exploration in Open GUI World 

**Authors**: Runliang Niu, Jinglong Ji, Yi Chang, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19095)  

**Abstract**: The rapid progress of large language models (LLMs) has sparked growing interest in building Artificial General Intelligence (AGI) within Graphical User Interface (GUI) environments. However, existing GUI agents based on LLMs or vision-language models (VLMs) often fail to generalize to novel environments and rely heavily on manually curated, diverse datasets. To overcome these limitations, we introduce ScreenExplorer, a VLM trained via Group Relative Policy Optimization(GRPO) in real, dynamic, and open-ended GUI environments. Innovatively, we introduced a world-model-based curiosity reward function to help the agent overcome the cold-start phase of exploration. Additionally, distilling experience streams further enhances the model's exploration capabilities. Our training framework enhances model exploration in open GUI environments, with trained models showing better environmental adaptation and sustained exploration compared to static deployment models. Our findings offer a scalable pathway toward AGI systems with self-improving capabilities in complex interactive settings. 

---
# CardioCoT: Hierarchical Reasoning for Multimodal Survival Analysis 

**Authors**: Shaohao Rui, Haoyang Su, Jinyi Xiang, Lian-Ming Wu, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19195)  

**Abstract**: Accurate prediction of major adverse cardiovascular events recurrence risk in acute myocardial infarction patients based on postoperative cardiac MRI and associated clinical notes is crucial for precision treatment and personalized intervention. Existing methods primarily focus on risk stratification capability while overlooking the need for intermediate robust reasoning and model interpretability in clinical practice. Moreover, end-to-end risk prediction using LLM/VLM faces significant challenges due to data limitations and modeling complexity. To bridge this gap, we propose CardioCoT, a novel two-stage hierarchical reasoning-enhanced survival analysis framework designed to enhance both model interpretability and predictive performance. In the first stage, we employ an evidence-augmented self-refinement mechanism to guide LLM/VLMs in generating robust hierarchical reasoning trajectories based on associated radiological findings. In the second stage, we integrate the reasoning trajectories with imaging data for risk model training and prediction. CardioCoT demonstrates superior performance in MACE recurrence risk prediction while providing interpretable reasoning processes, offering valuable insights for clinical decision-making. 

---
# SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning 

**Authors**: Kun Xiang, Heng Li, Terry Jingchen Zhang, Yinya Huang, Zirong Liu, Peixin Qu, Jixi He, Jiaqi Chen, Yu-Jie Yuan, Jianhua Han, Hang Xu, Hanhui Li, Mrinmaya Sachan, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19099)  

**Abstract**: We present SeePhys, a large-scale multimodal benchmark for LLM reasoning grounded in physics questions ranging from middle school to PhD qualifying exams. The benchmark covers 7 fundamental domains spanning the physics discipline, incorporating 21 categories of highly heterogeneous diagrams. In contrast to prior works where visual elements mainly serve auxiliary purposes, our benchmark features a substantial proportion of vision-essential problems (75\%) that mandate visual information extraction for correct solutions. Through extensive evaluation, we observe that even the most advanced visual reasoning models (e.g., Gemini-2.5-pro and o4-mini) achieve sub-60\% accuracy on our benchmark. These results reveal fundamental challenges in current large language models' visual understanding capabilities, particularly in: (i) establishing rigorous coupling between diagram interpretation and physics reasoning, and (ii) overcoming their persistent reliance on textual cues as cognitive shortcuts. 

---
# MLLMs are Deeply Affected by Modality Bias 

**Authors**: Xu Zheng, Chenfei Liao, Yuqian Fu, Kaiyu Lei, Yuanhuiyi Lyu, Lutao Jiang, Bin Ren, Jialei Chen, Jiawen Wang, Chengxin Li, Linfeng Zhang, Danda Pani Paudel, Xuanjing Huang, Yu-Gang Jiang, Nicu Sebe, Dacheng Tao, Luc Van Gool, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18657)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have shown promising results in integrating diverse modalities such as texts and images. MLLMs are heavily influenced by modality bias, often relying on language while under-utilizing other modalities like visual inputs. This position paper argues that MLLMs are deeply affected by modality bias. Firstly, we diagnose the current state of modality bias, highlighting its manifestations across various tasks. Secondly, we propose a systematic research road-map related to modality bias in MLLMs. Thirdly, we identify key factors of modality bias in MLLMs and offer actionable suggestions for future research to mitigate it. To substantiate these findings, we conduct experiments that demonstrate the influence of each factor: 1. Data Characteristics: Language data is compact and abstract, while visual data is redundant and complex, creating an inherent imbalance in learning dynamics. 2. Imbalanced Backbone Capabilities: The dominance of pretrained language models in MLLMs leads to overreliance on language and neglect of visual information. 3. Training Objectives: Current objectives often fail to promote balanced cross-modal alignment, resulting in shortcut learning biased toward language. These findings highlight the need for balanced training strategies and model architectures to better integrate multiple modalities in MLLMs. We call for interdisciplinary efforts to tackle these challenges and drive innovation in MLLM research. Our work provides a fresh perspective on modality bias in MLLMs and offers insights for developing more robust and generalizable multimodal systems-advancing progress toward Artificial General Intelligence. 

---
# Doc-CoB: Enhancing Multi-Modal Document Understanding with Visual Chain-of-Boxes Reasoning 

**Authors**: Ye Mo, Zirui Shao, Kai Ye, Xianwei Mao, Bo Zhang, Hangdi Xing, Peng Ye, Gang Huang, Kehan Chen, Zhou Huan, Zixu Yan, Sheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.18603)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in document understanding. However, the information-dense nature of document images still poses challenges, as most queries depend on only a few relevant regions, with the rest being redundant. Existing one-pass MLLMs process entire document images without considering query relevance, often failing to focus on critical regions and producing unfaithful responses. Inspired by the human coarse-to-fine reading pattern, we introduce Doc-CoB (Chain-of-Box), a simple-yet-effective mechanism that integrates human-style visual reasoning into MLLM without modifying its architecture. Our method allows the model to autonomously select the set of regions (boxes) most relevant to the query, and then focus attention on them for further understanding. We first design a fully automatic pipeline, integrating a commercial MLLM with a layout analyzer, to generate 249k training samples with intermediate visual reasoning supervision. Then we incorporate two enabling tasks that improve box identification and box-query reasoning, which together enhance document understanding. Extensive experiments on seven benchmarks with four popular models show that Doc-CoB significantly improves performance, demonstrating its effectiveness and wide applicability. All code, data, and models will be released publicly. 

---
# Generative RLHF-V: Learning Principles from Multi-modal Human Preference 

**Authors**: Jiayi Zhou, Jiaming Ji, Boyuan Chen, Jiapeng Sun, Wenqi Chen, Donghai Hong, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18531)  

**Abstract**: Training multi-modal large language models (MLLMs) that align with human intentions is a long-term challenge. Traditional score-only reward models for alignment suffer from low accuracy, weak generalization, and poor interpretability, blocking the progress of alignment methods, e.g., reinforcement learning from human feedback (RLHF). Generative reward models (GRMs) leverage MLLMs' intrinsic reasoning capabilities to discriminate pair-wise responses, but their pair-wise paradigm makes it hard to generalize to learnable rewards. We introduce Generative RLHF-V, a novel alignment framework that integrates GRMs with multi-modal RLHF. We propose a two-stage pipeline: $\textbf{multi-modal generative reward modeling from RL}$, where RL guides GRMs to actively capture human intention, then predict the correct pair-wise scores; and $\textbf{RL optimization from grouped comparison}$, which enhances multi-modal RL scoring precision by grouped responses comparison. Experimental results demonstrate that, besides out-of-distribution generalization of RM discrimination, our framework improves 4 MLLMs' performance across 7 benchmarks by $18.1\%$, while the baseline RLHF is only $5.3\%$. We further validate that Generative RLHF-V achieves a near-linear improvement with an increasing number of candidate responses. Our code and models can be found at this https URL. 

---
# DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning 

**Authors**: Qi Cao, Ruiyi Wang, Ruiyi Zhang, Sai Ashish Somayajula, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20241)  

**Abstract**: Reasoning has substantially improved the performance of large language models (LLMs) on complicated tasks. Central to the current reasoning studies, Process Reward Models (PRMs) offer a fine-grained evaluation of intermediate reasoning steps and guide the reasoning process. However, extending PRMs to multimodal large language models (MLLMs) introduces challenges. Since multimodal reasoning covers a wider range of tasks compared to text-only scenarios, the resulting distribution shift from the training to testing sets is more severe, leading to greater generalization difficulty. Training a reliable multimodal PRM, therefore, demands large and diverse datasets to ensure sufficient coverage. However, current multimodal reasoning datasets suffer from a marked quality imbalance, which degrades PRM performance and highlights the need for an effective data selection strategy. To address the issues, we introduce DreamPRM, a domain-reweighted training framework for multimodal PRMs which employs bi-level optimization. In the lower-level optimization, DreamPRM performs fine-tuning on multiple datasets with domain weights, allowing the PRM to prioritize high-quality reasoning signals and alleviating the impact of dataset quality imbalance. In the upper-level optimization, the PRM is evaluated on a separate meta-learning dataset; this feedback updates the domain weights through an aggregation loss function, thereby improving the generalization capability of trained PRM. Extensive experiments on multiple multimodal reasoning benchmarks covering both mathematical and general reasoning show that test-time scaling with DreamPRM consistently improves the performance of state-of-the-art MLLMs. Further comparisons reveal that DreamPRM's domain-reweighting strategy surpasses other data selection methods and yields higher accuracy gains than existing test-time scaling approaches. 

---
# In-Context Brush: Zero-shot Customized Subject Insertion with Context-Aware Latent Space Manipulation 

**Authors**: Yu Xu, Fan Tang, You Wu, Lin Gao, Oliver Deussen, Hongbin Yan, Jintao Li, Juan Cao, Tong-Yee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20271)  

**Abstract**: Recent advances in diffusion models have enhanced multimodal-guided visual generation, enabling customized subject insertion that seamlessly "brushes" user-specified objects into a given image guided by textual prompts. However, existing methods often struggle to insert customized subjects with high fidelity and align results with the user's intent through textual prompts. In this work, we propose "In-Context Brush", a zero-shot framework for customized subject insertion by reformulating the task within the paradigm of in-context learning. Without loss of generality, we formulate the object image and the textual prompts as cross-modal demonstrations, and the target image with the masked region as the query. The goal is to inpaint the target image with the subject aligning textual prompts without model tuning. Building upon a pretrained MMDiT-based inpainting network, we perform test-time enhancement via dual-level latent space manipulation: intra-head "latent feature shifting" within each attention head that dynamically shifts attention outputs to reflect the desired subject semantics and inter-head "attention reweighting" across different heads that amplifies prompt controllability through differential attention prioritization. Extensive experiments and applications demonstrate that our approach achieves superior identity preservation, text alignment, and image quality compared to existing state-of-the-art methods, without requiring dedicated training or additional data collection. 

---
# Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain) 

**Authors**: Subba Reddy Oota, Akshett Jindal, Ishani Mondal, Khushbu Pahwa, Satya Sai Srinath Namburi, Manish Shrivastava, Maneesh Singh, Bapi S. Raju, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.20029)  

**Abstract**: Transformer-based language models, though not explicitly trained to mimic brain recordings, have demonstrated surprising alignment with brain activity. Progress in these models-through increased size, instruction-tuning, and multimodality-has led to better representational alignment with neural data. Recently, a new class of instruction-tuned multimodal LLMs (MLLMs) have emerged, showing remarkable zero-shot capabilities in open-ended multimodal vision tasks. However, it is unknown whether MLLMs, when prompted with natural instructions, lead to better brain alignment and effectively capture instruction-specific representations. To address this, we first investigate brain alignment, i.e., measuring the degree of predictivity of neural visual activity using text output response embeddings from MLLMs as participants engage in watching natural scenes. Experiments with 10 different instructions show that MLLMs exhibit significantly better brain alignment than vision-only models and perform comparably to non-instruction-tuned multimodal models like CLIP. We also find that while these MLLMs are effective at generating high-quality responses suitable to the task-specific instructions, not all instructions are relevant for brain alignment. Further, by varying instructions, we make the MLLMs encode instruction-specific visual concepts related to the input image. This analysis shows that MLLMs effectively capture count-related and recognition-related concepts, demonstrating strong alignment with brain activity. Notably, the majority of the explained variance of the brain encoding models is shared between MLLM embeddings of image captioning and other instructions. These results suggest that enhancing MLLMs' ability to capture task-specific information could lead to better differentiation between various types of instructions, and thereby improving their precision in predicting brain responses. 

---
# ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving 

**Authors**: Xueyi Liu, Zuodong Zhong, Yuxin Guo, Yun-Fu Liu, Zhiguo Su, Qichao Zhang, Junli Wang, Yinfeng Gao, Yupeng Zheng, Qiao Lin, Huiyong Chen, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20024)  

**Abstract**: Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. Code and dataset will be found in this https URL. 

---
# Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models 

**Authors**: Hyunsik Chae, Seungwoo Yoon, Jaden Park, Chloe Yewon Chun, Yongin Cho, Mu Cai, Yong Jae Lee, Ernest K. Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20021)  

**Abstract**: Recent Vision-Language Models (VLMs) have demonstrated impressive multimodal comprehension and reasoning capabilities, yet they often struggle with trivially simple visual tasks. In this work, we focus on the domain of basic 2D Euclidean geometry and systematically categorize the fundamental, indivisible visual perception skills, which we refer to as atomic visual skills. We then introduce the Atomic Visual Skills Dataset (AVSD) for evaluating VLMs on the atomic visual skills. Using AVSD, we benchmark state-of-the-art VLMs and find that they struggle with these tasks, despite being trivial for adult humans. Our findings highlight the need for purpose-built datasets to train and evaluate VLMs on atomic, rather than composite, visual perception tasks. 

---
# Two Causally Related Needles in a Video Haystack 

**Authors**: Miaoyu Li, Qin Chao, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19853)  

**Abstract**: Evaluating the video understanding capabilities of Video-Language Models (VLMs) remains a significant challenge. We propose a long-context video understanding benchmark, Causal2Needles, that assesses two crucial abilities insufficiently evaluated by existing benchmarks: (1) the ability to extract information from two separate locations in a long video and understand them jointly, and (2) the ability to model the world in terms of cause and effect in human behaviors. Specifically, Causal2Needles introduces 2-needle questions, which require extracting information from both the cause and effect human-behavior events in a long video and the associated narration text. To prevent textual bias, these questions comprise two complementary formats: one asking to identify the video clip containing the answer, and one asking for the textual description of an unrelated visual detail from that video clip. Our experiments reveal that models excelling in pre-existing benchmarks struggle with 2-needle visual grounding, and the model performance is negatively correlated with the distance between the two needles. These findings highlight critical limitations in current VLMs. 

---
# StyleAR: Customizing Multimodal Autoregressive Model for Style-Aligned Text-to-Image Generation 

**Authors**: Yi Wu, Lingting Zhu, Shengju Qian, Lei Liu, Wandi Qiao, Lequan Yu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19874)  

**Abstract**: In the current research landscape, multimodal autoregressive (AR) models have shown exceptional capabilities across various domains, including visual understanding and generation. However, complex tasks such as style-aligned text-to-image generation present significant challenges, particularly in data acquisition. In analogy to instruction-following tuning for image editing of AR models, style-aligned generation requires a reference style image and prompt, resulting in a text-image-to-image triplet where the output shares the style and semantics of the input. However, acquiring large volumes of such triplet data with specific styles is considerably more challenging than obtaining conventional text-to-image data used for training generative models. To address this issue, we propose StyleAR, an innovative approach that combines a specially designed data curation method with our proposed AR models to effectively utilize text-to-image binary data for style-aligned text-to-image generation. Our method synthesizes target stylized data using a reference style image and prompt, but only incorporates the target stylized image as the image modality to create high-quality binary data. To facilitate binary data training, we introduce a CLIP image encoder with a perceiver resampler that translates the image input into style tokens aligned with multimodal tokens in AR models and implement a style-enhanced token technique to prevent content leakage which is a common issue in previous work. Furthermore, we mix raw images drawn from large-scale text-image datasets with stylized images to enhance StyleAR's ability to extract richer stylistic features and ensure style consistency. Extensive qualitative and quantitative experiments demonstrate our superior performance. 

---
# Benchmarking Large Multimodal Models for Ophthalmic Visual Question Answering with OphthalWeChat 

**Authors**: Pusheng Xu, Xia Gong, Xiaolan Chen, Weiyi Zhang, Jiancheng Yang, Bingjie Yan, Meng Yuan, Yalin Zheng, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19624)  

**Abstract**: Purpose: To develop a bilingual multimodal visual question answering (VQA) benchmark for evaluating VLMs in ophthalmology. Methods: Ophthalmic image posts and associated captions published between January 1, 2016, and December 31, 2024, were collected from WeChat Official Accounts. Based on these captions, bilingual question-answer (QA) pairs in Chinese and English were generated using GPT-4o-mini. QA pairs were categorized into six subsets by question type and language: binary (Binary_CN, Binary_EN), single-choice (Single-choice_CN, Single-choice_EN), and open-ended (Open-ended_CN, Open-ended_EN). The benchmark was used to evaluate the performance of three VLMs: GPT-4o, Gemini 2.0 Flash, and Qwen2.5-VL-72B-Instruct. Results: The final OphthalWeChat dataset included 3,469 images and 30,120 QA pairs across 9 ophthalmic subspecialties, 548 conditions, 29 imaging modalities, and 68 modality combinations. Gemini 2.0 Flash achieved the highest overall accuracy (0.548), outperforming GPT-4o (0.522, P < 0.001) and Qwen2.5-VL-72B-Instruct (0.514, P < 0.001). It also led in both Chinese (0.546) and English subsets (0.550). Subset-specific performance showed Gemini 2.0 Flash excelled in Binary_CN (0.687), Single-choice_CN (0.666), and Single-choice_EN (0.646), while GPT-4o ranked highest in Binary_EN (0.717), Open-ended_CN (BLEU-1: 0.301; BERTScore: 0.382), and Open-ended_EN (BLEU-1: 0.183; BERTScore: 0.240). Conclusions: This study presents the first bilingual VQA benchmark for ophthalmology, distinguished by its real-world context and inclusion of multiple examinations per patient. The dataset reflects authentic clinical decision-making scenarios and enables quantitative evaluation of VLMs, supporting the development of accurate, specialized, and trustworthy AI systems for eye care. 

---
# Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models 

**Authors**: Rui Cai, Bangzheng Li, Xiaofei Wen, Muhao Chen, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19616)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across tasks, yet they often exhibit difficulty in distinguishing task-relevant from irrelevant signals, particularly in tasks like Visual Question Answering (VQA), which can lead to susceptibility to misleading or spurious inputs. We refer to this broader limitation as the Cross-Modality Competency Problem: the model's inability to fairly evaluate all modalities. This vulnerability becomes more evident in modality-specific tasks such as image classification or pure text question answering, where models are expected to rely solely on one modality. In such tasks, spurious information from irrelevant modalities often leads to significant performance degradation. We refer to this failure as Modality Interference, which serves as a concrete and measurable instance of the cross-modality competency problem. We further design a perturbation-based causal diagnostic experiment to verify and quantify this problem. To mitigate modality interference, we propose a novel framework to fine-tune MLLMs, including perturbation-based data augmentations with both heuristic perturbations and adversarial perturbations via Projected Gradient Descent (PGD), and a consistency regularization strategy applied to model outputs with original and perturbed inputs. Experiments on multiple benchmark datasets (image-heavy, text-heavy, and VQA tasks) and multiple model families with different scales demonstrate significant improvements in robustness and cross-modality competency, indicating our method's effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks. 

---
# Align and Surpass Human Camouflaged Perception: Visual Refocus Reinforcement Fine-Tuning 

**Authors**: Ruolin Shen, Xiaozhong Ji, Kai WU, Jiangning Zhang, Yijun He, HaiHua Yang, Xiaobin Hu, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19611)  

**Abstract**: Current multi-modal models exhibit a notable misalignment with the human visual system when identifying objects that are visually assimilated into the background. Our observations reveal that these multi-modal models cannot distinguish concealed objects, demonstrating an inability to emulate human cognitive processes which effectively utilize foreground-background similarity principles for visual analysis. To analyze this hidden human-model visual thinking discrepancy, we build a visual system that mimicks human visual camouflaged perception to progressively and iteratively `refocus' visual concealed content. The refocus is a progressive guidance mechanism enabling models to logically localize objects in visual images through stepwise reasoning. The localization process of concealed objects requires hierarchical attention shifting with dynamic adjustment and refinement of prior cognitive knowledge. In this paper, we propose a visual refocus reinforcement framework via the policy optimization algorithm to encourage multi-modal models to think and refocus more before answering, and achieve excellent reasoning abilities to align and even surpass human camouflaged perception systems. Our extensive experiments on camouflaged perception successfully demonstrate the emergence of refocus visual phenomena, characterized by multiple reasoning tokens and dynamic adjustment of the detection box. Besides, experimental results on both camouflaged object classification and detection tasks exhibit significantly superior performance compared to Supervised Fine-Tuning (SFT) baselines. 

---
# Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models 

**Authors**: Yifan Jia, Kailin Jiang, Yuyang Liang, Qihan Ren, Yi Xin, Rui Yang, Fenze Feng, Mingcai Chen, Hengyang Lu, Haozhe Wang, Xiaoye Qu, Dongrui Liu, Lizhen Cui, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.19509)  

**Abstract**: Large Multimodal Models(LMMs) face notable challenges when encountering multimodal knowledge conflicts, particularly under retrieval-augmented generation(RAG) frameworks where the contextual information from external sources may contradict the model's internal parametric knowledge, leading to unreliable outputs. However, existing benchmarks fail to reflect such realistic conflict scenarios. Most focus solely on intra-memory conflicts, while context-memory and inter-context conflicts remain largely investigated. Furthermore, commonly used factual knowledge-based evaluations are often overlooked, and existing datasets lack a thorough investigation into conflict detection capabilities. To bridge this gap, we propose MMKC-Bench, a benchmark designed to evaluate factual knowledge conflicts in both context-memory and inter-context scenarios. MMKC-Bench encompasses three types of multimodal knowledge conflicts and includes 1,573 knowledge instances and 3,381 images across 23 broad types, collected through automated pipelines with human verification. We evaluate three representative series of LMMs on both model behavior analysis and conflict detection tasks. Our findings show that while current LMMs are capable of recognizing knowledge conflicts, they tend to favor internal parametric knowledge over external evidence. We hope MMKC-Bench will foster further research in multimodal knowledge conflict and enhance the development of multimodal RAG systems. The source code is available at this https URL. 

---
# Rethinking Gating Mechanism in Sparse MoE: Handling Arbitrary Modality Inputs with Confidence-Guided Gate 

**Authors**: Liangwei Nathan Zheng, Wei Emma Zhang, Mingyu Guo, Miao Xu, Olaf Maennel, Weitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19525)  

**Abstract**: Effectively managing missing modalities is a fundamental challenge in real-world multimodal learning scenarios, where data incompleteness often results from systematic collection errors or sensor failures. Sparse Mixture-of-Experts (SMoE) architectures have the potential to naturally handle multimodal data, with individual experts specializing in different modalities. However, existing SMoE approach often lacks proper ability to handle missing modality, leading to performance degradation and poor generalization in real-world applications. We propose Conf-SMoE to introduce a two-stage imputation module to handle the missing modality problem for the SMoE architecture and reveal the insight of expert collapse from theoretical analysis with strong empirical evidence. Inspired by our theoretical analysis, Conf-SMoE propose a novel expert gating mechanism by detaching the softmax routing score to task confidence score w.r.t ground truth. This naturally relieves expert collapse without introducing additional load balance loss function. We show that the insights of expert collapse aligns with other gating mechanism such as Gaussian and Laplacian gate. We also evaluate the proposed method on four different real world dataset with three different experiment settings to conduct comprehensive the analysis of Conf-SMoE on modality fusion and resistance to missing modality. 

---
# Enhancing Visual Reliance in Text Generation: A Bayesian Perspective on Mitigating Hallucination in Large Vision-Language Models 

**Authors**: Nanxing Hu, Xiaoyue Duan, Jinchao Zhang, Guoliang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19498)  

**Abstract**: Large Vision-Language Models (LVLMs) usually generate texts which satisfy context coherence but don't match the visual input. Such a hallucination issue hinders LVLMs' applicability in the real world. The key to solving hallucination in LVLM is to make the text generation rely more on the visual content. Most previous works choose to enhance/adjust the features/output of a specific modality (i.e., visual or textual) to alleviate hallucinations in LVLM, which do not explicitly or systematically enhance the visual reliance. In this paper, we comprehensively investigate the factors which may degenerate the visual reliance in text generation of LVLM from a Bayesian perspective. Based on our observations, we propose to mitigate hallucination in LVLM from three aspects. Firstly, we observe that not all visual tokens are informative in generating meaningful texts. We propose to evaluate and remove redundant visual tokens to avoid their disturbance. Secondly, LVLM may encode inappropriate prior information, making it lean toward generating unexpected words. We propose a simple yet effective way to rectify the prior from a Bayesian perspective. Thirdly, we observe that starting from certain steps, the posterior of next-token prediction conditioned on visual tokens may collapse to a prior distribution which does not depend on any informative visual tokens at all. Thus, we propose to stop further text generation to avoid hallucination. Extensive experiments on three benchmarks including POPE, CHAIR, and MME demonstrate that our method can consistently mitigate the hallucination issue of LVLM and performs favorably against previous state-of-the-arts. 

---
# MM-Prompt: Cross-Modal Prompt Tuning for Continual Visual Question Answering 

**Authors**: Xu Li, Fan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19455)  

**Abstract**: Continual Visual Question Answering (CVQA) based on pre-trained models(PTMs) has achieved promising progress by leveraging prompt tuning to enable continual multi-modal learning. However, most existing methods adopt cross-modal prompt isolation, constructing visual and textual prompts separately, which exacerbates modality imbalance and leads to degraded performance over time. To tackle this issue, we propose MM-Prompt, a novel framework incorporating cross-modal prompt query and cross-modal prompt recovery. The former enables balanced prompt selection by incorporating cross-modal signals during query formation, while the latter promotes joint prompt reconstruction through iterative cross-modal interactions, guided by an alignment loss to prevent representational drift. Extensive experiments show that MM-Prompt surpasses prior approaches in accuracy and knowledge retention, while maintaining balanced modality engagement throughout continual learning. 

---
# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use 

**Authors**: Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2505.19255)  

**Abstract**: Reinforcement Learning Finetuning (RFT) has significantly advanced the reasoning capabilities of large language models (LLMs) by enabling long chains of thought, self-correction, and effective tool use. While recent works attempt to extend RFT to vision-language models (VLMs), these efforts largely produce text-only reasoning conditioned on static image inputs, falling short of true multimodal reasoning in the response. In contrast, test-time methods like Visual Sketchpad incorporate visual steps but lack training mechanisms.
We introduce VTool-R1, the first framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps. VTool-R1 integrates Python-based visual editing tools into the RFT process, enabling VLMs to learn when and how to generate visual reasoning steps that benefit final reasoning. Trained with outcome-based rewards tied to task accuracy, our approach elicits strategic visual tool use for reasoning without relying on process-based supervision. Experiments on structured visual question answering over charts and tables show that VTool-R1 enhances reasoning performance by teaching VLMs to "think with images" and generate multimodal chain of thoughts with tools. 

---
# I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts 

**Authors**: Jiayi Xin, Sukwon Yun, Jie Peng, Inyoung Choi, Jenna L. Ballard, Tianlong Chen, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.19190)  

**Abstract**: Modality fusion is a cornerstone of multimodal learning, enabling information integration from diverse data sources. However, vanilla fusion methods are limited by (1) inability to account for heterogeneous interactions between modalities and (2) lack of interpretability in uncovering the multimodal interactions inherent in the data. To this end, we propose I2MoE (Interpretable Multimodal Interaction-aware Mixture of Experts), an end-to-end MoE framework designed to enhance modality fusion by explicitly modeling diverse multimodal interactions, as well as providing interpretation on a local and global level. First, I2MoE utilizes different interaction experts with weakly supervised interaction losses to learn multimodal interactions in a data-driven way. Second, I2MoE deploys a reweighting model that assigns importance scores for the output of each interaction expert, which offers sample-level and dataset-level interpretation. Extensive evaluation of medical and general multimodal datasets shows that I2MoE is flexible enough to be combined with different fusion techniques, consistently improves task performance, and provides interpretation across various real-world scenarios. Code is available at this https URL. 

---
# SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards 

**Authors**: Chuming Shen, Wei Wei, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19094)  

**Abstract**: DeepSeek-R1 has demonstrated powerful reasoning capabilities in the text domain through stable reinforcement learning (RL). Recently, in the multimodal domain, works have begun to directly apply RL to generate R1-like free-form reasoning for Visual Question Answering (VQA) tasks. However, multimodal tasks share an intrinsically different nature from textual tasks, which heavily rely on the understanding of the input image to solve the problem. Therefore, such free-form reasoning faces two critical limitations in the VQA task: (1) Extended reasoning chains diffuse visual focus away from task-critical regions, degrading answer accuracy. (2) Unverifiable intermediate steps amplify policy-gradient variance and computational costs overhead. To address these issues, in this paper, we introduce SATORI ($\textbf{S}patially$ $\textbf{A}nchored$ $\textbf{T}ask$ $\textbf{O}ptimization$ with $\textbf{R}e\textbf{I}nforcement$ Learning), which decomposes VQA into three verifiable stages, including global image captioning, region localization, and answer prediction, each supplying explicit reward signals. Furthermore, we also introduce VQA-Verify, a 12k dataset annotated with answer-aligned captions and bounding-boxes to facilitate training. Experiments demonstrate consistent performance improvements across seven VQA benchmarks, achieving up to $15.7\%$ improvement in accuracy in accuracy compared to the R1-like baseline. Our analysis of the attention map confirms enhanced focus on critical regions, which brings improvements in accuracy. Our code is available at this https URL. 

---
# Medical Large Vision Language Models with Multi-Image Visual Ability 

**Authors**: Xikai Yang, Juzheng Miao, Yuchen Yuan, Jiaze Wang, Qi Dou, Jinpeng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19031)  

**Abstract**: Medical large vision-language models (LVLMs) have demonstrated promising performance across various single-image question answering (QA) benchmarks, yet their capability in processing multi-image clinical scenarios remains underexplored. Unlike single image based tasks, medical tasks involving multiple images often demand sophisticated visual understanding capabilities, such as temporal reasoning and cross-modal analysis, which are poorly supported by current medical LVLMs. To bridge this critical gap, we present the Med-MIM instruction dataset, comprising 83.2K medical multi-image QA pairs that span four types of multi-image visual abilities (temporal understanding, reasoning, comparison, co-reference). Using this dataset, we fine-tune Mantis and LLaVA-Med, resulting in two specialized medical VLMs: MIM-LLaVA-Med and Med-Mantis, both optimized for multi-image analysis. Additionally, we develop the Med-MIM benchmark to comprehensively evaluate the medical multi-image understanding capabilities of LVLMs. We assess eight popular LVLMs, including our two models, on the Med-MIM benchmark. Experimental results show that both Med-Mantis and MIM-LLaVA-Med achieve superior performance on the held-in and held-out subsets of the Med-MIM benchmark, demonstrating that the Med-MIM instruction dataset effectively enhances LVLMs' multi-image understanding capabilities in the medical domain. 

---
# InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts 

**Authors**: Minzhi Lin, Tianchi Xie, Mengchen Liu, Yilin Ye, Changjian Chen, Shixia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19028)  

**Abstract**: Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at this https URL. 

---
# How Do Images Align and Complement LiDAR? Towards a Harmonized Multi-modal 3D Panoptic Segmentation 

**Authors**: Yining Pan, Qiongjie Cui, Xulei Yang, Na Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18956)  

**Abstract**: LiDAR-based 3D panoptic segmentation often struggles with the inherent sparsity of data from LiDAR sensors, which makes it challenging to accurately recognize distant or small objects. Recently, a few studies have sought to overcome this challenge by integrating LiDAR inputs with camera images, leveraging the rich and dense texture information provided by the latter. While these approaches have shown promising results, they still face challenges, such as misalignment during data augmentation and the reliance on post-processing steps. To address these issues, we propose Image-Assists-LiDAR (IAL), a novel multi-modal 3D panoptic segmentation framework. In IAL, we first introduce a modality-synchronized data augmentation strategy, PieAug, to ensure alignment between LiDAR and image inputs from the start. Next, we adopt a transformer decoder to directly predict panoptic segmentation results. To effectively fuse LiDAR and image features into tokens for the decoder, we design a Geometric-guided Token Fusion (GTF) module. Additionally, we leverage the complementary strengths of each modality as priors for query initialization through a Prior-based Query Generation (PQG) module, enhancing the decoder's ability to generate accurate instance masks. Our IAL framework achieves state-of-the-art performance compared to previous multi-modal 3D panoptic segmentation methods on two widely used benchmarks. Code and models are publicly available at <this https URL. 

---
# Revival with Voice: Multi-modal Controllable Text-to-Speech Synthesis 

**Authors**: Minsu Kim, Pingchuan Ma, Honglie Chen, Stavros Petridis, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2505.18972)  

**Abstract**: This paper explores multi-modal controllable Text-to-Speech Synthesis (TTS) where the voice can be generated from face image, and the characteristics of output speech (e.g., pace, noise level, distance, tone, place) can be controllable with natural text description. Specifically, we aim to mitigate the following three challenges in face-driven TTS systems. 1) To overcome the limited audio quality of audio-visual speech corpora, we propose a training method that additionally utilizes high-quality audio-only speech corpora. 2) To generate voices not only from real human faces but also from artistic portraits, we propose augmenting the input face image with stylization. 3) To consider one-to-many possibilities in face-to-voice mapping and ensure consistent voice generation at the same time, we propose to first employ sampling-based decoding and then use prompting with generated speech samples. Experimental results validate the proposed model's effectiveness in face-driven voice synthesis. 

---
# REGen: Multimodal Retrieval-Embedded Generation for Long-to-Short Video Editing 

**Authors**: Weihan Xu, Yimeng Ma, Jingyue Huang, Yang Li, Wenye Ma, Taylor Berg-Kirkpatrick, Julian McAuley, Paul Pu Liang, Hao-Wen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18880)  

**Abstract**: Short videos are an effective tool for promoting contents and improving knowledge accessibility. While existing extractive video summarization methods struggle to produce a coherent narrative, existing abstractive methods cannot `quote' from the input videos, i.e., inserting short video clips in their outputs. In this work, we explore novel video editing models for generating shorts that feature a coherent narrative with embedded video insertions extracted from a long input video. We propose a novel retrieval-embedded generation framework that allows a large language model to quote multimodal resources while maintaining a coherent narrative. Our proposed REGen system first generates the output story script with quote placeholders using a finetuned large language model, and then uses a novel retrieval model to replace the quote placeholders by selecting a video clip that best supports the narrative from a pool of candidate quotable video clips. We examine the proposed method on the task of documentary teaser generation, where short interview insertions are commonly used to support the narrative of a documentary. Our objective evaluations show that the proposed method can effectively insert short video clips while maintaining a coherent narrative. In a subjective survey, we show that our proposed method outperforms existing abstractive and extractive approaches in terms of coherence, alignment, and realism in teaser generation. 

---
# SD-OVON: A Semantics-aware Dataset and Benchmark Generation Pipeline for Open-Vocabulary Object Navigation in Dynamic Scenes 

**Authors**: Dicong Qiu, Jiadi You, Zeying Gong, Ronghe Qiu, Hui Xiong, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18881)  

**Abstract**: We present the Semantics-aware Dataset and Benchmark Generation Pipeline for Open-vocabulary Object Navigation in Dynamic Scenes (SD-OVON). It utilizes pretraining multimodal foundation models to generate infinite unique photo-realistic scene variants that adhere to real-world semantics and daily commonsense for the training and the evaluation of navigation agents, accompanied with a plugin for generating object navigation task episodes compatible to the Habitat simulator. In addition, we offer two pre-generated object navigation task datasets, SD-OVON-3k and SD-OVON-10k, comprising respectively about 3k and 10k episodes of the open-vocabulary object navigation task, derived from the SD-OVON-Scenes dataset with 2.5k photo-realistic scans of real-world environments and the SD-OVON-Objects dataset with 0.9k manually inspected scanned and artist-created manipulatable object models. Unlike prior datasets limited to static environments, SD-OVON covers dynamic scenes and manipulatable objects, facilitating both real-to-sim and sim-to-real robotic applications. This approach enhances the realism of navigation tasks, the training and the evaluation of open-vocabulary object navigation agents in complex settings. To demonstrate the effectiveness of our pipeline and datasets, we propose two baselines and evaluate them along with state-of-the-art baselines on SD-OVON-3k. The datasets, benchmark and source code are publicly available. 

---
# OmniGenBench: A Benchmark for Omnipotent Multimodal Generation across 50+ Tasks 

**Authors**: Jiayu Wang, Yang Jiao, Yue Yu, Tianwen Qian, Shaoxiang Chen, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18775)  

**Abstract**: Recent breakthroughs in large multimodal models (LMMs), such as the impressive GPT-4o-Native, have demonstrated remarkable proficiency in following general-purpose instructions for image generation. However, current benchmarks often lack the necessary breadth and depth to fully evaluate the diverse capabilities of these models. To overcome this limitation, we introduce OmniGenBench, a novel and comprehensive benchmark meticulously designed to assess the instruction-following abilities of state-of-the-art LMMs across both perception-centric and cognition-centric dimensions. Our OmniGenBench includes 57 diverse sub-tasks grounded in real-world scenarios, systematically categorized according to the specific model capabilities they demand. For rigorous evaluation, we further employ a dual-mode protocol. This protocol utilizes off-the-shelf visual parsing tools for perception-centric tasks and a powerful LLM-based judger for cognition-centric tasks to assess the alignment between generated images and user instructions. Using OmniGenBench, we evaluate mainstream generative models, including prevalent models like GPT-4o, Gemini-2.0-Flash, and Seedream, and provide in-depth comparisons and analyses of their this http URL and data are available at this https URL. 

---
# VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning 

**Authors**: Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18719)  

**Abstract**: Recent high-capacity vision-language-action (VLA) models have demonstrated impressive performance on a range of robotic manipulation tasks by imitating human demonstrations. However, exploiting offline data with limited visited states will cause execution failure in out-of-distribution scenarios. Intuitively, an exploration-based method that improves on online collected data at test time could address this limitation. We present VLA-RL, an algorithmic and systematic framework that leverages online reinforcement learning (RL) to improve pretrained auto-regressive VLAs in downstream tasks. Within a unified perspective, we first introduce a trajectory-level RL formulation for auto-regressive VLA training, which models general robotic manipulation trajectory as multi-modal multi-turn conversation. To address the challenge of sparse rewards, we fine-tune a pretrained vision-language model as a robotic process reward model, which is trained on pseudo reward labels annotated on automatically extracted task segments. To scale up, we identify several implementation findings that improve the stability and efficiency including curriculum selection strategy, GPU-balanced vectorized environments, batch decoding, and critic warmup. VLA-RL enables OpenVLA-7B to surpass the strongest finetuned baseline by 4.5% on 40 challenging robotic manipulation tasks in LIBERO, and even matches the performance of advanced commercial models such as $\pi_0$-FAST. Notably, we observe that VLA-RL benefits from increased test-time optimization, indicating an early spark of inference scaling laws in robotics. 

---
# Robustness in Large Language Models: A Survey of Mitigation Strategies and Evaluation Metrics 

**Authors**: Pankaj Kumar, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.18658)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising cornerstone for the development of natural language processing (NLP) and artificial intelligence (AI). However, ensuring the robustness of LLMs remains a critical challenge. To address these challenges and advance the field, this survey provides a comprehensive overview of current studies in this area. First, we systematically examine the nature of robustness in LLMs, including its conceptual foundations, the importance of consistent performance across diverse inputs, and the implications of failure modes in real-world applications. Next, we analyze the sources of non-robustness, categorizing intrinsic model limitations, data-driven vulnerabilities, and external adversarial factors that compromise reliability. Following this, we review state-of-the-art mitigation strategies, and then we discuss widely adopted benchmarks, emerging metrics, and persistent gaps in assessing real-world reliability. Finally, we synthesize findings from existing surveys and interdisciplinary studies to highlight trends, unresolved issues, and pathways for future research. 

---
# GRE Suite: Geo-localization Inference via Fine-Tuned Vision-Language Models and Enhanced Reasoning Chains 

**Authors**: Chun Wang, Xiaoran Pan, Zihao Pan, Haofan Wang, Yiren Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.18700)  

**Abstract**: Recent advances in Visual Language Models (VLMs) have demonstrated exceptional performance in visual reasoning tasks. However, geo-localization presents unique challenges, requiring the extraction of multigranular visual cues from images and their integration with external world knowledge for systematic reasoning. Current approaches to geo-localization tasks often lack robust reasoning mechanisms and explainability, limiting their effectiveness. To address these limitations, we propose the Geo Reason Enhancement (GRE) Suite, a novel framework that augments VLMs with structured reasoning chains for accurate and interpretable location inference. The GRE Suite is systematically developed across three key dimensions: dataset, model, and benchmark. First, we introduce GRE30K, a high-quality geo-localization reasoning dataset designed to facilitate fine-grained visual and contextual analysis. Next, we present the GRE model, which employs a multi-stage reasoning strategy to progressively infer scene attributes, local details, and semantic features, thereby narrowing down potential geographic regions with enhanced precision. Finally, we construct the Geo Reason Evaluation Benchmark (GREval-Bench), a comprehensive evaluation framework that assesses VLMs across diverse urban, natural, and landmark scenes to measure both coarse-grained (e.g., country, continent) and fine-grained (e.g., city, street) localization performance. Experimental results demonstrate that GRE significantly outperforms existing methods across all granularities of geo-localization tasks, underscoring the efficacy of reasoning-augmented VLMs in complex geographic inference. Code and data will be released at this https URL. 

---
# Rethinking Causal Mask Attention for Vision-Language Inference 

**Authors**: Xiaohuan Pei, Tao Huang, YanXiang Ma, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18605)  

**Abstract**: Causal attention has become a foundational mechanism in autoregressive vision-language models (VLMs), unifying textual and visual inputs under a single generative framework. However, existing causal mask-based strategies are inherited from large language models (LLMs) where they are tailored for text-only decoding, and their adaptation to vision tokens is insufficiently addressed in the prefill stage. Strictly masking future positions for vision queries introduces overly rigid constraints, which hinder the model's ability to leverage future context that often contains essential semantic cues for accurate inference. In this work, we empirically investigate how different causal masking strategies affect vision-language inference and then propose a family of future-aware attentions tailored for this setting. We first empirically analyze the effect of previewing future tokens for vision queries and demonstrate that rigid masking undermines the model's capacity to capture useful contextual semantic representations. Based on these findings, we propose a lightweight attention family that aggregates future visual context into past representations via pooling, effectively preserving the autoregressive structure while enhancing cross-token dependencies. We evaluate a range of causal masks across diverse vision-language inference settings and show that selectively compressing future semantic context into past representations benefits the inference. 

---
# Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment 

**Authors**: Bryan Sangwoo Kim, Jeongsol Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.18600)  

**Abstract**: Modern single-image super-resolution (SISR) models deliver photo-realistic results at the scale factors on which they are trained, but collapse when asked to magnify far beyond that regime. We address this scalability bottleneck with Chain-of-Zoom (CoZ), a model-agnostic framework that factorizes SISR into an autoregressive chain of intermediate scale-states with multi-scale-aware prompts. CoZ repeatedly re-uses a backbone SR model, decomposing the conditional probability into tractable sub-problems to achieve extreme resolutions without additional training. Because visual cues diminish at high magnifications, we augment each zoom step with multi-scale-aware text prompts generated by a vision-language model (VLM). The prompt extractor itself is fine-tuned using Generalized Reward Policy Optimization (GRPO) with a critic VLM, aligning text guidance towards human preference. Experiments show that a standard 4x diffusion SR model wrapped in CoZ attains beyond 256x enlargement with high perceptual quality and fidelity. 

---
# MPE-TTS: Customized Emotion Zero-Shot Text-To-Speech Using Multi-Modal Prompt 

**Authors**: Zhichao Wu, Yueteng Kang, Songjun Cao, Long Ma, Qiulin Li, Qun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18453)  

**Abstract**: Most existing Zero-Shot Text-To-Speech(ZS-TTS) systems generate the unseen speech based on single prompt, such as reference speech or text descriptions, which limits their flexibility. We propose a customized emotion ZS-TTS system based on multi-modal prompt. The system disentangles speech into the content, timbre, emotion and prosody, allowing emotion prompts to be provided as text, image or speech. To extract emotion information from different prompts, we propose a multi-modal prompt emotion encoder. Additionally, we introduce an prosody predictor to fit the distribution of prosody and propose an emotion consistency loss to preserve emotion information in the predicted prosody. A diffusion-based acoustic model is employed to generate the target mel-spectrogram. Both objective and subjective experiments demonstrate that our system outperforms existing systems in terms of naturalness and similarity. The samples are available at this https URL. 

---
# CrashAgent: Crash Scenario Generation via Multi-modal Reasoning 

**Authors**: Miao Li, Wenhao Ding, Haohong Lin, Yiqi Lyu, Yihang Yao, Yuyou Zhang, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18341)  

**Abstract**: Training and evaluating autonomous driving algorithms requires a diverse range of scenarios. However, most available datasets predominantly consist of normal driving behaviors demonstrated by human drivers, resulting in a limited number of safety-critical cases. This imbalance, often referred to as a long-tail distribution, restricts the ability of driving algorithms to learn from crucial scenarios involving risk or failure, scenarios that are essential for humans to develop driving skills efficiently. To generate such scenarios, we utilize Multi-modal Large Language Models to convert crash reports of accidents into a structured scenario format, which can be directly executed within simulations. Specifically, we introduce CrashAgent, a multi-agent framework designed to interpret multi-modal real-world traffic crash reports for the generation of both road layouts and the behaviors of the ego vehicle and surrounding traffic participants. We comprehensively evaluate the generated crash scenarios from multiple perspectives, including the accuracy of layout reconstruction, collision rate, and diversity. The resulting high-quality and large-scale crash dataset will be publicly available to support the development of safe driving algorithms in handling safety-critical situations. 

---
# Token Reduction Should Go Beyond Efficiency in Generative Models -- From Vision, Language to Multimodality 

**Authors**: Zhenglun Kong, Yize Li, Fanhu Zeng, Lei Xin, Shvat Messica, Xue Lin, Pu Zhao, Manolis Kellis, Hao Tang, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.18227)  

**Abstract**: In Transformer architectures, tokens\textemdash discrete units derived from raw data\textemdash are formed by segmenting inputs into fixed-length chunks. Each token is then mapped to an embedding, enabling parallel attention computations while preserving the input's essential information. Due to the quadratic computational complexity of transformer self-attention mechanisms, token reduction has primarily been used as an efficiency strategy. This is especially true in single vision and language domains, where it helps balance computational costs, memory usage, and inference latency. Despite these advances, this paper argues that token reduction should transcend its traditional efficiency-oriented role in the era of large generative models. Instead, we position it as a fundamental principle in generative modeling, critically influencing both model architecture and broader applications. Specifically, we contend that across vision, language, and multimodal systems, token reduction can: (i) facilitate deeper multimodal integration and alignment, (ii) mitigate "overthinking" and hallucinations, (iii) maintain coherence over long inputs, and (iv) enhance training stability, etc. We reframe token reduction as more than an efficiency measure. By doing so, we outline promising future directions, including algorithm design, reinforcement learning-guided token reduction, token optimization for in-context learning, and broader ML and scientific domains. We highlight its potential to drive new model architectures and learning strategies that improve robustness, increase interpretability, and better align with the objectives of generative modeling. 

---
# Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications 

**Authors**: Yubo Peng, Luping Xiang, Bingxin Zhang, Kun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18194)  

**Abstract**: Traditional single-modal sensing systems-based solely on either radio frequency (RF) or visual data-struggle to cope with the demands of complex and dynamic environments. Furthermore, single-device systems are constrained by limited perspectives and insufficient spatial coverage, which impairs their effectiveness in urban or non-line-of-sight scenarios. To overcome these challenges, we propose a novel large language model (LLM)-driven distributed integrated multimodal sensing and semantic communication (LLM-DiSAC) framework. Specifically, our system consists of multiple collaborative sensing devices equipped with RF and camera modules, working together with an aggregation center to enhance sensing accuracy. First, on sensing devices, LLM-DiSAC develops an RF-vision fusion network (RVFN), which employs specialized feature extractors for RF and visual data, followed by a cross-attention module for effective multimodal integration. Second, a LLM-based semantic transmission network (LSTN) is proposed to enhance communication efficiency, where the LLM-based decoder leverages known channel parameters, such as transceiver distance and signal-to-noise ratio (SNR), to mitigate semantic distortion. Third, at the aggregation center, a transformer-based aggregation model (TRAM) with an adaptive aggregation attention mechanism is developed to fuse distributed features and enhance sensing accuracy. To preserve data privacy, a two-stage distributed learning strategy is introduced, allowing local model training at the device level and centralized aggregation model training using intermediate features. Finally, evaluations on a synthetic multi-view RF-visual dataset generated by the Genesis simulation engine show that LLM-DiSAC achieves a good performance. 

---
# Less is More: Multimodal Region Representation via Pairwise Inter-view Learning 

**Authors**: Min Namgung, Yijun Lin, JangHyeon Lee, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18178)  

**Abstract**: With the increasing availability of geospatial datasets, researchers have explored region representation learning (RRL) to analyze complex region characteristics. Recent RRL methods use contrastive learning (CL) to capture shared information between two modalities but often overlook task-relevant unique information specific to each modality. Such modality-specific details can explain region characteristics that shared information alone cannot capture. Bringing information factorization to RRL can address this by factorizing multimodal data into shared and unique information. However, existing factorization approaches focus on two modalities, whereas RRL can benefit from various geospatial data. Extending factorization beyond two modalities is non-trivial because modeling high-order relationships introduces a combinatorial number of learning objectives, increasing model complexity. We introduce Cross modal Knowledge Injected Embedding, an information factorization approach for RRL that captures both shared and unique representations. CooKIE uses a pairwise inter-view learning approach that captures high-order information without modeling high-order dependency, avoiding exhaustive combinations. We evaluate CooKIE on three regression tasks and a land use classification task in New York City and Delhi, India. Results show that CooKIE outperforms existing RRL methods and a factorized RRL model, capturing multimodal information with fewer training parameters and floating-point operations per second (FLOPs). We release the code: this https URL. 

---
