# Rethinking the Potential of Multimodality in Collaborative Problem Solving Diagnosis with Large Language Models 

**Authors**: K. Wong, B. Wu, S. Bulathwela, M. Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.15093)  

**Abstract**: Detecting collaborative and problem-solving behaviours from digital traces to interpret students' collaborative problem solving (CPS) competency is a long-term goal in the Artificial Intelligence in Education (AIEd) field. Although multimodal data and advanced models are argued to have the potential to detect complex CPS behaviours, empirical evidence on their value remains limited with some contrasting evidence. In this study, we investigated the potential of multimodal data to improve model performance in diagnosing 78 secondary school students' CPS subskills and indicators in authentic educational settings. In particular, text embeddings from verbal data and acoustic embeddings from audio data were used in a multimodal classification model for CPS diagnosis. Both unimodal and multimodal transformer-based models outperformed traditional models in detecting CPS classes. Although the inclusion of multimodality did not improve the performance of traditional unimodal models, its integration into transformer-based models demonstrated improved performance for diagnosing social-cognitive CPS classes compared to unimodal transformer-based models. Based on the results, the paper argues that multimodality and the selection of a particular modelling technique should not be taken for granted to achieve the best performance in the automated detection of every CPS subskill and indicator. Rather, their value is limited to certain types of CPS indicators, affected by the complexity of the labels, and dependent on the composition of indicators in the dataset. We conclude the paper by discussing the required nuance when considering the value of LLMs and multimodality in automated CPS diagnosis, highlighting the need for human-AI complementarity, and proposing the exploration of relevant model architectures and techniques to improve CPS diagnosis in authentic educational contexts. 

---
# Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey 

**Authors**: Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan, Zhenya Huang, Shiwei Tong, Guoping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14891)  

**Abstract**: Recent advancements in Retrieval-Augmented Generation (RAG) have revolutionized natural language processing by integrating Large Language Models (LLMs) with external information retrieval, enabling accurate, up-to-date, and verifiable text generation across diverse applications. However, evaluating RAG systems presents unique challenges due to their hybrid architecture that combines retrieval and generation components, as well as their dependence on dynamic knowledge sources in the LLM era. In response, this paper provides a comprehensive survey of RAG evaluation methods and frameworks, systematically reviewing traditional and emerging evaluation approaches, for system performance, factual accuracy, safety, and computational efficiency in the LLM era. We also compile and categorize the RAG-specific datasets and evaluation frameworks, conducting a meta-analysis of evaluation practices in high-impact RAG research. To the best of our knowledge, this work represents the most comprehensive survey for RAG evaluation, bridging traditional and LLM-driven methods, and serves as a critical resource for advancing RAG development. 

---
# OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding 

**Authors**: Songtao Jiang, Yuan Wang, Sibo Song, Yan Zhang, Zijie Meng, Bohan Lei, Jian Wu, Jimeng Sun, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14692)  

**Abstract**: The practical deployment of medical vision-language models (Med-VLMs) necessitates seamless integration of textual data with diverse visual modalities, including 2D/3D images and videos, yet existing models typically employ separate encoders for different modalities. To address this limitation, we present OmniV-Med, a unified framework for multimodal medical understanding. Our technical contributions are threefold: First, we construct OmniV-Med-Instruct, a comprehensive multimodal medical dataset containing 252K instructional samples spanning 14 medical image modalities and 11 clinical tasks. Second, we devise a rotary position-adaptive encoder that processes multi-resolution 2D/3D images and videos within a unified architecture, diverging from conventional modality-specific encoders. Third, we introduce a medical-aware token pruning mechanism that exploits spatial-temporal redundancy in volumetric data (e.g., consecutive CT slices) and medical videos, effectively reducing 60\% of visual tokens without performance degradation. Empirical evaluations demonstrate that OmniV-Med-7B achieves state-of-the-art performance on 7 benchmarks spanning 2D/3D medical imaging and video understanding tasks. Notably, our lightweight variant (OmniV-Med-1.5B) attains comparable performance while requiring only 8 RTX3090 GPUs for training and supporting efficient long-video inference. Data, code and model will be released. 

---
# Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach 

**Authors**: Xingyu Li, Chen Gong, Guohong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14321)  

**Abstract**: Multimodal coreference resolution (MCR) aims to identify mentions referring to the same entity across different modalities, such as text and visuals, and is essential for understanding multimodal content. In the era of rapidly growing mutimodal content and social media, MCR is particularly crucial for interpreting user interactions and bridging text-visual references to improve communication and personalization. However, MCR research for real-world dialogues remains unexplored due to the lack of sufficient data this http URL address this gap, we introduce TikTalkCoref, the first Chinese multimodal coreference dataset for social media in real-world scenarios, derived from the popular Douyin short-video platform. This dataset pairs short videos with corresponding textual dialogues from user comments and includes manually annotated coreference clusters for both person mentions in the text and the coreferential person head regions in the corresponding video frames. We also present an effective benchmark approach for MCR, focusing on the celebrity domain, and conduct extensive experiments on our dataset, providing reliable benchmark results for this newly constructed dataset. We will release the TikTalkCoref dataset to facilitate future research on MCR for real-world social media dialogues. 

---
# An LMM for Efficient Video Understanding via Reinforced Compression of Video Cubes 

**Authors**: Ji Qi, Yuan Yao, Yushi Bai, Bin Xu, Juanzi Li, Zhiyuan Liu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.15270)  

**Abstract**: Large Multimodal Models (LMMs) uniformly perceive video frames, creating computational inefficiency for videos with inherently varying temporal information density. This paper present \textbf{Quicksviewer}, an LMM with new perceiving paradigm that partitions a video of nonuniform density into varying cubes using Gumbel Softmax, followed by a unified resampling for each cube to achieve efficient video understanding. This simple and intuitive approach dynamically compress video online based on its temporal density, significantly reducing spatiotemporal redundancy (overall 45$\times$ compression rate), while enabling efficient training with large receptive field. We train the model from a language backbone through three progressive stages, each incorporating lengthy videos on average of 420s/1fps thanks to the perceiving efficiency. With only 0.8M total video-text samples for training, our model outperforms the direct baseline employing a fixed partitioning strategy by a maximum of 8.72 in accuracy, demonstrating the effectiveness in performance. On Video-MME, Quicksviewer achieves SOTA under modest sequence lengths using just up to 5\% of tokens per frame required by baselines. With this paradigm, scaling up the number of input frames reveals a clear power law of the model capabilities. It is also empirically verified that the segments generated by the cubing network can help for analyzing continuous events in videos. 

---
# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking 

**Authors**: Juyeon Kim, Geon Lee, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15135)  

**Abstract**: Entity linking (EL) aligns textual mentions with their corresponding entities in a knowledge base, facilitating various applications such as semantic search and question answering. Recent advances in multimodal entity linking (MEL) have shown that combining text and images can reduce ambiguity and improve alignment accuracy. However, most existing MEL methods overlook the rich structural information available in the form of knowledge-graph (KG) triples. In this paper, we propose KGMEL, a novel framework that leverages KG triples to enhance MEL. Specifically, it operates in three stages: (1) Generation: Produces high-quality triples for each mention by employing vision-language models based on its text and images. (2) Retrieval: Learns joint mention-entity representations, via contrastive learning, that integrate text, images, and (generated or KG) triples to retrieve candidate entities for each mention. (3) Reranking: Refines the KG triples of the candidate entities and employs large language models to identify the best-matching entity for the mention. Extensive experiments on benchmark datasets demonstrate that KGMEL outperforms existing methods. Our code and datasets are available at: this https URL. 

---
# Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs 

**Authors**: Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Rouyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.15280)  

**Abstract**: Multi-view understanding, the ability to reconcile visual information across diverse viewpoints for effective navigation, manipulation, and 3D scene comprehension, is a fundamental challenge in Multi-Modal Large Language Models (MLLMs) to be used as embodied agents. While recent MLLMs have shown impressive advances in high-level reasoning and planning, they frequently fall short when confronted with multi-view geometric consistency and cross-view correspondence. To comprehensively evaluate the challenges of MLLMs in multi-view scene reasoning, we propose All-Angles Bench, a benchmark of over 2,100 human carefully annotated multi-view question-answer pairs across 90 diverse real-world scenes. Our six tasks (counting, attribute identification, relative distance, relative direction, object manipulation, and camera pose estimation) specifically test model's geometric correspondence and the capacity to align information consistently across views. Our extensive experiments, benchmark on 27 representative MLLMs including Gemini-2.0-Flash, Claude-3.7-Sonnet, and GPT-4o against human evaluators reveals a substantial performance gap, indicating that current MLLMs remain far from human-level proficiency. Through in-depth analysis, we show that MLLMs are particularly underperforming under two aspects: (1) cross-view correspondence for partially occluded views and (2) establishing the coarse camera poses. These findings highlight the necessity of domain-specific refinements or modules that embed stronger multi-view awareness. We believe that our All-Angles Bench offers valuable insights and contribute to bridging the gap between MLLMs and human-level multi-view understanding. The project and benchmark are publicly available at this https URL. 

---
# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform 

**Authors**: Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, Kaiyu Tang, Bin Wen, Hai-Tao Zheng, Fan Yang, Tingting Gao, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14904)  

**Abstract**: Exponentially growing short video platforms (SVPs) face significant challenges in moderating content detrimental to users' mental health, particularly for minors. The dissemination of such content on SVPs can lead to catastrophic societal consequences. Although substantial efforts have been dedicated to moderating such content, existing methods suffer from critical limitations: (1) Manual review is prone to human bias and incurs high operational costs. (2) Automated methods, though efficient, lack nuanced content understanding, resulting in lower accuracy. (3) Industrial moderation regulations struggle to adapt to rapidly evolving trends due to long update cycles. In this paper, we annotate the first SVP content moderation benchmark with authentic user/reviewer feedback to fill the absence of benchmark in this field. Then we evaluate various methods on the benchmark to verify the existence of the aforementioned limitations. We further propose our common-law content moderation framework named KuaiMod to address these challenges. KuaiMod consists of three components: training data construction, offline adaptation, and online deployment & refinement. Leveraging large vision language model (VLM) and Chain-of-Thought (CoT) reasoning, KuaiMod adequately models video toxicity based on sparse user feedback and fosters dynamic moderation policy with rapid update speed and high accuracy. Offline experiments and large-scale online A/B test demonstrates the superiority of KuaiMod: KuaiMod achieves the best moderation performance on our benchmark. The deployment of KuaiMod reduces the user reporting rate by 20% and its application in video recommendation increases both Daily Active User (DAU) and APP Usage Time (AUT) on several Kuaishou scenarios. We have open-sourced our benchmark at this https URL. 

---
# A Multimodal Recaptioning Framework to Account for Perceptual Diversity in Multilingual Vision-Language Modeling 

**Authors**: Kyle Buettner, Jacob Emmerson, Adriana Kovashka  

**Link**: [PDF](https://arxiv.org/pdf/2504.14359)  

**Abstract**: There are many ways to describe, name, and group objects when captioning an image. Differences are evident when speakers come from diverse cultures due to the unique experiences that shape perception. Machine translation of captions has pushed multilingual capabilities in vision-language models (VLMs), but data comes mainly from English speakers, indicating a perceptual bias and lack of model flexibility. In this work, we address this challenge and outline a data-efficient framework to instill multilingual VLMs with greater understanding of perceptual diversity. We specifically propose an LLM-based, multimodal recaptioning strategy that alters the object descriptions of English captions before translation. The greatest benefits are demonstrated in a targeted multimodal mechanism guided by native speaker data. By adding produced rewrites as augmentations in training, we improve on German and Japanese text-image retrieval cases studies (up to +3.5 mean recall overall, +4.7 on non-native error cases). We further propose a mechanism to analyze the specific object description differences across datasets, and we offer insights into cross-dataset and cross-language generalization. 

---
# Towards Explainable Fake Image Detection with Multi-Modal Large Language Models 

**Authors**: Yikun Ji, Yan Hong, Jiahui Zhan, Haoxing Chen, jun lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14245)  

**Abstract**: Progress in image generation raises significant public security concerns. We argue that fake image detection should not operate as a "black box". Instead, an ideal approach must ensure both strong generalization and transparency. Recent progress in Multi-modal Large Language Models (MLLMs) offers new opportunities for reasoning-based AI-generated image detection. In this work, we evaluate the capabilities of MLLMs in comparison to traditional detection methods and human evaluators, highlighting their strengths and limitations. Furthermore, we design six distinct prompts and propose a framework that integrates these prompts to develop a more robust, explainable, and reasoning-driven detection system. The code is available at this https URL. 

---
# InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners 

**Authors**: Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14239)  

**Abstract**: Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, two-stage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-to-error steps. Experimental results show InfiGUI-R1 achieves strong performance in GUI grounding and trajectory tasks. Resources at this https URL. 

---
# Cross-attention for State-based model RWKV-7 

**Authors**: Liu Xiao, Li Zhiyuan, Lin Yueyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14260)  

**Abstract**: We introduce CrossWKV, a novel cross-attention mechanism for the state-based RWKV-7 model, designed to enhance the expressive power of text-to-image generation. Leveraging RWKV-7's linear-complexity Weighted Key-Value (WKV) architecture, CrossWKV integrates text and image modalities in a single pass, utilizing a generalized delta rule with vector-valued gating and low-rank adaptations (LoRA) to achieve superior cross-modal alignment. Unlike Transformer-based models, CrossWKV's non-diagonal, input-dependent transition matrix enables it to represent complex functions beyond the $\mathrm{TC}^0$ complexity class, including all regular languages, as demonstrated by its ability to perform state-tracking tasks like $S_5$ permutation modeling. Evaluated within the Diffusion in RWKV-7 (DIR-7) on datasets such as LAION-5B and ImageNet, CrossWKV achieves a Frechet Inception Distance (FID) of 2.88 and a CLIP score of 0.33 on ImageNet 256x256, matching state-of-the-art performance while offering robust generalization across diverse prompts. The model's enhanced expressivity, combined with constant memory usage and linear scaling, positions it as a powerful solution for advanced cross-modal tasks, with potential applications in high-resolution generation and dynamic state this http URL at this https URL 

---
# 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark 

**Authors**: Ivan Sviridov, Amina Miftakhova, Artemiy Tereshchenko, Galina Zubkova, Pavel Blinov, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2504.13861)  

**Abstract**: Large Vision-Language Models (LVLMs) are increasingly being explored for applications in telemedicine, yet their ability to engage with diverse patient behaviors remains underexplored. We introduce 3MDBench (Medical Multimodal Multi-agent Dialogue Benchmark), an open-source evaluation framework designed to assess LLM-driven medical consultations. Unlike existing benchmarks, 3MDBench simulates real-world patient variability by incorporating four temperament-driven Patient Agents and an Assessor Agent that evaluates diagnostic accuracy and dialogue quality. The benchmark integrates textual and image-based patient data across 34 common diagnoses, mirroring real-world telemedicine interactions. Under different diagnostic strategies, we evaluate state-of-the-art LVLMs. Our findings demonstrate that incorporating dialogue improves the F1 score from 50.4 to 54.2 compared to non-dialogue settings, underscoring the value of context-driven, information-seeking questioning. Additionally, we demonstrate that multimodal inputs enhance diagnostic efficiency. Image-supported models outperform text-only counterparts by raising the diagnostic F1 score from 52.8 to 54.2 in a similar dialogue setting. Finally, we suggest an approach that improves the diagnostic F1-score to 70.3 by training the CNN model on the diagnosis prediction task and incorporating its top-3 predictions into the LVLM context. 3MDBench provides a reproducible and extendable evaluation framework for AI-driven medical assistants. It offers insights into how patient temperament, dialogue strategies, and multimodal reasoning influence diagnosis quality. By addressing real-world complexities in telemedicine, our benchmark paves the way for more empathetic, reliable, and context-aware AI-driven healthcare solutions. The source code of our benchmark is publicly available: this https URL 

---
# A Survey on (M)LLM-Based GUI Agents 

**Authors**: Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13865)  

**Abstract**: Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation. 

---
# UFO2: The Desktop AgentOS 

**Authors**: Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14603)  

**Abstract**: Recent Computer-Using Agents (CUAs), powered by multimodal large language models (LLMs), offer a promising direction for automating complex desktop workflows through natural language. However, most existing CUAs remain conceptual prototypes, hindered by shallow OS integration, fragile screenshot-based interaction, and disruptive execution.
We present UFO2, a multiagent AgentOS for Windows desktops that elevates CUAs into practical, system-level automation. UFO2 features a centralized HostAgent for task decomposition and coordination, alongside a collection of application-specialized AppAgent equipped with native APIs, domain-specific knowledge, and a unified GUI--API action layer. This architecture enables robust task execution while preserving modularity and extensibility. A hybrid control detection pipeline fuses Windows UI Automation (UIA) with vision-based parsing to support diverse interface styles. Runtime efficiency is further enhanced through speculative multi-action planning, reducing per-step LLM overhead. Finally, a Picture-in-Picture (PiP) interface enables automation within an isolated virtual desktop, allowing agents and users to operate concurrently without interference.
We evaluate UFO2 across over 20 real-world Windows applications, demonstrating substantial improvements in robustness and execution accuracy over prior CUAs. Our results show that deep OS integration unlocks a scalable path toward reliable, user-aligned desktop automation. 

---
# Learning from Reasoning Failures via Synthetic Data Generation 

**Authors**: Gabriela Ben Melech Stan, Estelle Aflalo, Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2504.14523)  

**Abstract**: Training models on synthetic data has emerged as an increasingly important strategy for improving the performance of generative AI. This approach is particularly helpful for large multimodal models (LMMs) due to the relative scarcity of high-quality paired image-text data compared to language-only data. While a variety of methods have been proposed for generating large multimodal datasets, they do not tailor the synthetic data to address specific deficiencies in the reasoning abilities of LMMs which will be trained with the generated dataset. In contrast, humans often learn in a more efficient manner by seeking out examples related to the types of reasoning where they have failed previously. Inspired by this observation, we propose a new approach for synthetic data generation which is grounded in the analysis of an existing LMM's reasoning failures. Our methodology leverages frontier models to automatically analyze errors produced by a weaker LMM and propose new examples which can be used to correct the reasoning failure via additional training, which are then further filtered to ensure high quality. We generate a large multimodal instruction tuning dataset containing over 553k examples using our approach and conduct extensive experiments demonstrating its utility for improving the performance of LMMs on multiple downstream tasks. Our results show that models trained on our synthetic data can even exceed the performance of LMMs trained on an equivalent amount of additional real data, demonstrating the high value of generating synthetic data targeted to specific reasoning failure modes in LMMs. We will make our dataset and code publicly available. 

---
# Adaptation Method for Misinformation Identification 

**Authors**: Yangping Chen, Weijie Shi, Mengze Li, Yue Cui, Hao Chen, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14171)  

**Abstract**: Multimodal fake news detection plays a crucial role in combating online misinformation. Unfortunately, effective detection methods rely on annotated labels and encounter significant performance degradation when domain shifts exist between training (source) and test (target) data. To address the problems, we propose ADOSE, an Active Domain Adaptation (ADA) framework for multimodal fake news detection which actively annotates a small subset of target samples to improve detection performance. To identify various deceptive patterns in cross-domain settings, we design multiple expert classifiers to learn dependencies across different modalities. These classifiers specifically target the distinct deception patterns exhibited in fake news, where two unimodal classifiers capture knowledge errors within individual modalities while one cross-modal classifier identifies semantic inconsistencies between text and images. To reduce annotation costs from the target domain, we propose a least-disagree uncertainty selector with a diversity calculator for selecting the most informative samples. The selector leverages prediction disagreement before and after perturbations by multiple classifiers as an indicator of uncertain samples, whose deceptive patterns deviate most from source domains. It further incorporates diversity scores derived from multi-view features to ensure the chosen samples achieve maximal coverage of target domain features. The extensive experiments on multiple datasets show that ADOSE outperforms existing ADA methods by 2.72\% $\sim$ 14.02\%, indicating the superiority of our model. 

---
# Zero-Shot, But at What Cost? Unveiling the Hidden Overhead of MILS's LLM-CLIP Framework for Image Captioning 

**Authors**: Yassir Benhammou, Alessandro Tiberio, Gabriel Trautmann, Suman Kalyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15199)  

**Abstract**: MILS (Multimodal Iterative LLM Solver) is a recently published framework that claims "LLMs can see and hear without any training" by leveraging an iterative, LLM-CLIP based approach for zero-shot image captioning. While this MILS approach demonstrates good performance, our investigation reveals that this success comes at a hidden, substantial computational cost due to its expensive multi-step refinement process. In contrast, alternative models such as BLIP-2 and GPT-4V achieve competitive results through a streamlined, single-pass approach. We hypothesize that the significant overhead inherent in MILS's iterative process may undermine its practical benefits, thereby challenging the narrative that zero-shot performance can be attained without incurring heavy resource demands. This work is the first to expose and quantify the trade-offs between output quality and computational cost in MILS, providing critical insights for the design of more efficient multimodal models. 

---
# Chinese-LiPS: A Chinese audio-visual speech recognition dataset with Lip-reading and Presentation Slides 

**Authors**: Jinghua Zhao, Yuhang Jia, Shiyao Wang, Jiaming Zhou, Hui Wang, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15066)  

**Abstract**: Incorporating visual modalities to assist Automatic Speech Recognition (ASR) tasks has led to significant improvements. However, existing Audio-Visual Speech Recognition (AVSR) datasets and methods typically rely solely on lip-reading information or speaking contextual video, neglecting the potential of combining these different valuable visual cues within the speaking context. In this paper, we release a multimodal Chinese AVSR dataset, Chinese-LiPS, comprising 100 hours of speech, video, and corresponding manual transcription, with the visual modality encompassing both lip-reading information and the presentation slides used by the speaker. Based on Chinese-LiPS, we develop a simple yet effective pipeline, LiPS-AVSR, which leverages both lip-reading and presentation slide information as visual modalities for AVSR tasks. Experiments show that lip-reading and presentation slide information improve ASR performance by approximately 8\% and 25\%, respectively, with a combined performance improvement of about 35\%. The dataset is available at this https URL 

---
# ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams 

**Authors**: Chris Dongjoo Kim, Jihwan Moon, Sangwoo Moon, Heeseung Yun, Sihaeng Lee, Aniruddha Kembhavi, Soonyoung Lee, Gunhee Kim, Sangho Lee, Christopher Clark  

**Link**: [PDF](https://arxiv.org/pdf/2504.14875)  

**Abstract**: The rapid growth of video-text data presents challenges in storage and computation during training. Online learning, which processes streaming data in real-time, offers a promising solution to these issues while also allowing swift adaptations in scenarios demanding real-time responsiveness. One strategy to enhance the efficiency and effectiveness of learning involves identifying and prioritizing data that enhances performance on target downstream tasks. We propose Relevance and Specificity-based online filtering framework (ReSpec) that selects data based on four criteria: (i) modality alignment for clean data, (ii) task relevance for target focused data, (iii) specificity for informative and detailed data, and (iv) efficiency for low-latency processing. Relevance is determined by the probabilistic alignment of incoming data with downstream tasks, while specificity employs the distance to a root embedding representing the least specific data as an efficient proxy for informativeness. By establishing reference points from target task data, ReSpec filters incoming data in real-time, eliminating the need for extensive storage and compute. Evaluating on large-scale datasets WebVid2M and VideoCC3M, ReSpec attains state-of-the-art performance on five zeroshot video retrieval tasks, using as little as 5% of the data while incurring minimal compute. The source code is available at this https URL. 

---
# Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation 

**Authors**: Yunpu Zhao, Rui Zhang, Junbin Xiao, Ruibo Hou, Jiaming Guo, Zihao Zhang, Yifan Hao, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14848)  

**Abstract**: Vision-language models (VLMs) excel in various multimodal tasks but frequently suffer from poor calibration, resulting in misalignment between their verbalized confidence and response correctness. This miscalibration undermines user trust, especially when models confidently provide incorrect or fabricated information. In this work, we propose a novel Confidence Calibration through Semantic Perturbation (CSP) framework to improve the calibration of verbalized confidence for VLMs in response to object-centric queries. We first introduce a perturbed dataset where Gaussian noise is applied to the key object regions to simulate visual uncertainty at different confidence levels, establishing an explicit mapping between visual ambiguity and confidence levels. We further enhance calibration through a two-stage training process combining supervised fine-tuning on the perturbed dataset with subsequent preference optimization. Extensive experiments on popular benchmarks demonstrate that our method significantly improves the alignment between verbalized confidence and response correctness while maintaining or enhancing overall task performance. These results highlight the potential of semantic perturbation as a practical tool for improving the reliability and interpretability of VLMs. 

---
# Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark 

**Authors**: Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14693)  

**Abstract**: Recent advancements in language multimodal models (LMMs) for video have demonstrated their potential for understanding video content, yet the task of comprehending multi-discipline lectures remains largely unexplored. We introduce Video-MMLU, a massive benchmark designed to evaluate the capabilities of LMMs in understanding Multi-Discipline Lectures. We evaluate over 90 open-source and proprietary models, ranging from 0.5B to 40B parameters. Our results highlight the limitations of current models in addressing the cognitive challenges presented by these lectures, especially in tasks requiring both perception and reasoning. Additionally, we explore how the number of visual tokens and the large language models influence performance, offering insights into the interplay between multimodal perception and reasoning in lecture comprehension. 

---
# Modality Selection and Skill Segmentation via Cross-Modality Attention 

**Authors**: Jiawei Jiang, Kei Ota, Devesh K. Jha, Asako Kanezaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14573)  

**Abstract**: Incorporating additional sensory modalities such as tactile and audio into foundational robotic models poses significant challenges due to the curse of dimensionality. This work addresses this issue through modality selection. We propose a cross-modality attention (CMA) mechanism to identify and selectively utilize the modalities that are most informative for action generation at each timestep. Furthermore, we extend the application of CMA to segment primitive skills from expert demonstrations and leverage this segmentation to train a hierarchical policy capable of solving long-horizon, contact-rich manipulation tasks. 

---
# K2MUSE: A human lower limb multimodal dataset under diverse conditions for facilitating rehabilitation robotics 

**Authors**: Jiwei Li, Bi Zhang, Xiaowei Tan, Wanxin Chen, Zhaoyuan Liu, Juanjuan Zhang, Weiguang Huo, Jian Huang, Lianqing Liu, Xingang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14602)  

**Abstract**: The natural interaction and control performance of lower limb rehabilitation robots are closely linked to biomechanical information from various human locomotion activities. Multidimensional human motion data significantly deepen the understanding of the complex mechanisms governing neuromuscular alterations, thereby facilitating the development and application of rehabilitation robots in multifaceted real-world environments. However, currently available lower limb datasets are inadequate for supplying the essential multimodal data and large-scale gait samples necessary for effective data-driven approaches, and they neglect the significant effects of acquisition interference in real this http URL fill this gap, we present the K2MUSE dataset, which includes a comprehensive collection of multimodal data, comprising kinematic, kinetic, amplitude-mode ultrasound (AUS), and surface electromyography (sEMG) measurements. The proposed dataset includes lower limb multimodal data from 30 able-bodied participants walking under different inclines (0$^\circ$, $\pm$5$^\circ$, and $\pm$10$^\circ$), various speeds (0.5 m/s, 1.0 m/s, and 1.5 m/s), and different nonideal acquisition conditions (muscle fatigue, electrode shifts, and inter-day differences). The kinematic and ground reaction force data were collected via a Vicon motion capture system and an instrumented treadmill with embedded force plates, whereas the sEMG and AUS data were synchronously recorded for thirteen muscles on the bilateral lower limbs. This dataset offers a new resource for designing control frameworks for rehabilitation robots and conducting biomechanical analyses of lower limb locomotion. The dataset is available at this https URL. 

---
# ResNetVLLM -- Multi-modal Vision LLM for the Video Understanding Task 

**Authors**: Ahmad Khalil, Mahmoud Khalil, Alioune Ngom  

**Link**: [PDF](https://arxiv.org/pdf/2504.14432)  

**Abstract**: In this paper, we introduce ResNetVLLM (ResNet Vision LLM), a novel cross-modal framework for zero-shot video understanding that integrates a ResNet-based visual encoder with a Large Language Model (LLM. ResNetVLLM addresses the challenges associated with zero-shot video models by avoiding reliance on pre-trained video understanding models and instead employing a non-pretrained ResNet to extract visual features. This design ensures the model learns visual and semantic representations within a unified architecture, enhancing its ability to generate accurate and contextually relevant textual descriptions from video inputs. Our experimental results demonstrate that ResNetVLLM achieves state-of-the-art performance in zero-shot video understanding (ZSVU) on several benchmarks, including MSRVTT-QA, MSVD-QA, TGIF-QA FrameQA, and ActivityNet-QA. 

---
# ResNetVLLM-2: Addressing ResNetVLLM's Multi-Modal Hallucinations 

**Authors**: Ahmad Khalil, Mahmoud Khalil, Alioune Ngom  

**Link**: [PDF](https://arxiv.org/pdf/2504.14429)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing (NLP) tasks, but they suffer from hallucination, generating plausible yet factually incorrect content. This issue extends to Video-Language Models (VideoLLMs), where textual descriptions may inaccurately represent visual content, resulting in multi-modal hallucinations. In this paper, we address hallucination in ResNetVLLM, a video-language model combining ResNet visual encoders with LLMs. We introduce a two-step protocol: (1) a faithfulness detection strategy that uses a modified Lynx model to assess semantic alignment between generated captions and ground-truth video references, and (2) a hallucination mitigation strategy using Retrieval-Augmented Generation (RAG) with an ad-hoc knowledge base dynamically constructed during inference. Our enhanced model, ResNetVLLM-2, reduces multi-modal hallucinations by cross-verifying generated content against external knowledge, improving factual consistency. Evaluation on the ActivityNet-QA benchmark demonstrates a substantial accuracy increase from 54.8% to 65.3%, highlighting the effectiveness of our hallucination detection and mitigation strategies in enhancing video-language model reliability. 

---
# FinSage: A Multi-aspect RAG System for Financial Filings Question Answering 

**Authors**: Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung Sum Thomas Kwok, Muzhi Li, Zhuhong Li, Hailin He, Yuchen Hua, Peng Lu, Suyuchen Wang, Yihong Wu, Jerry Huang, Ling Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14493)  

**Abstract**: Leveraging large language models in real-world settings often entails a need to utilize domain-specific data and tools in order to follow the complex regulations that need to be followed for acceptable use. Within financial sectors, modern enterprises increasingly rely on Retrieval-Augmented Generation (RAG) systems to address complex compliance requirements in financial document workflows. However, existing solutions struggle to account for the inherent heterogeneity of data (e.g., text, tables, diagrams) and evolving nature of regulatory standards used in financial filings, leading to compromised accuracy in critical information extraction. We propose the FinSage framework as a solution, utilizing a multi-aspect RAG framework tailored for regulatory compliance analysis in multi-modal financial documents. FinSage introduces three innovative components: (1) a multi-modal pre-processing pipeline that unifies diverse data formats and generates chunk-level metadata summaries, (2) a multi-path sparse-dense retrieval system augmented with query expansion (HyDE) and metadata-aware semantic search, and (3) a domain-specialized re-ranking module fine-tuned via Direct Preference Optimization (DPO) to prioritize compliance-critical content. Extensive experiments demonstrate that FinSage achieves an impressive recall of 92.51% on 75 expert-curated questions derived from surpasses the best baseline method on the FinanceBench question answering datasets by 24.06% in accuracy. Moreover, FinSage has been successfully deployed as financial question-answering agent in online meetings, where it has already served more than 1,200 people. 

---
# Adversarial Attack for RGB-Event based Visual Object Tracking 

**Authors**: Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14423)  

**Abstract**: Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios. The source code will be released on this https URL 

---
# Optimizing SIA Development: A Case Study in User-Centered Design for Estuary, a Multimodal Socially Interactive Agent Framework 

**Authors**: Spencer Lin, Miru Jun, Basem Rizk, Karen Shieh, Scott Fisher, Sharon Mozgai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14427)  

**Abstract**: This case study presents our user-centered design model for Socially Intelligent Agent (SIA) development frameworks through our experience developing Estuary, an open source multimodal framework for building low-latency real-time socially interactive agents. We leverage the Rapid Assessment Process (RAP) to collect the thoughts of leading researchers in the field of SIAs regarding the current state of the art for SIA development as well as their evaluation of how well Estuary may potentially address current research gaps. We achieve this through a series of end-user interviews conducted by a fellow researcher in the community. We hope that the findings of our work will not only assist the continued development of Estuary but also guide the development of other future frameworks and technologies for SIAs. 

---
# Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization 

**Authors**: Huiyi Chen, Jiawei Peng, Kaihua Tang, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14200)  

**Abstract**: In-context learning (ICL) enables Large Vision-Language Models (LVLMs) to adapt to new tasks without parameter updates, using a few demonstrations from a large support set. However, selecting informative demonstrations leads to high computational and memory costs. While some methods explore selecting a small and representative coreset in the text classification, evaluating all support set samples remains costly, and discarded samples lead to unnecessary information loss. These methods may also be less effective for image classification due to differences in feature spaces. Given these limitations, we propose Key-based Coreset Optimization (KeCO), a novel framework that leverages untapped data to construct a compact and informative coreset. We introduce visual features as keys within the coreset, which serve as the anchor for identifying samples to be updated through different selection strategies. By leveraging untapped samples from the support set, we update the keys of selected coreset samples, enabling the randomly initialized coreset to evolve into a more informative coreset under low computational cost. Through extensive experiments on coarse-grained and fine-grained image classification benchmarks, we demonstrate that KeCO effectively enhances ICL performance for image classification task, achieving an average improvement of more than 20\%. Notably, we evaluate KeCO under a simulated online scenario, and the strong performance in this scenario highlights the practical value of our framework for resource-constrained real-world scenarios. 

---
# A Physics-guided Multimodal Transformer Path to Weather and Climate Sciences 

**Authors**: Jing Han, Hanting Chen, Kai Han, Xiaomeng Huang, Yongyun Hu, Wenjun Xu, Dacheng Tao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14174)  

**Abstract**: With the rapid development of machine learning in recent years, many problems in meteorology can now be addressed using AI models. In particular, data-driven algorithms have significantly improved accuracy compared to traditional methods. Meteorological data is often transformed into 2D images or 3D videos, which are then fed into AI models for learning. Additionally, these models often incorporate physical signals, such as temperature, pressure, and wind speed, to further enhance accuracy and interpretability. In this paper, we review several representative AI + Weather/Climate algorithms and propose a new paradigm where observational data from different perspectives, each with distinct physical meanings, are treated as multimodal data and integrated via transformers. Furthermore, key weather and climate knowledge can be incorporated through regularization techniques to further strengthen the model's capabilities. This new paradigm is versatile and can address a variety of tasks, offering strong generalizability. We also discuss future directions for improving model accuracy and interpretability. 

---
# Expanding the Generative AI Design Space through Structured Prompting and Multimodal Interfaces 

**Authors**: Nimisha Karnatak, Adrien Baranes, Rob Marchant, Huinan Zeng, Tríona Butler, Kristen Olson  

**Link**: [PDF](https://arxiv.org/pdf/2504.14320)  

**Abstract**: Text-based prompting remains the dominant interaction paradigm in generative AI, yet it often results in a high-friction experience for novice users, such as small business owners (SBOs), attempting to articulate creative or domain-specific goals for advertising. To investigate this challenge, we conducted a study with six SBOs in the United Kingdom, focusing on their advertising practices and perceptions and usage of AI tools in this context. Our findings surfaced two persistent breakdowns in current generative AI systems: first, the cognitive burden of prompt engineering, as users struggled to translate abstract creative goals into effective textual inputs; and second, the frequent generation of generic outputs that failed to align with users' articulated brand vision. To address these issues, we developed ACAI (AI Co-Creation for Advertising and Inspiration), a multimodal, GenAI-powered advertisement creation tool designed to support novice designers by reimagining the prompt interface. ACAI features a structured, panel-based interface composed of three modules: the Branding Panel, the Audience & Goals Panel, and the Inspiration Board Panel to provide SBOs with outputs that align with their creative vision by reducing prompt ambiguity. This work contributes to HCI research on generative systems by showing how structured interfaces can foreground user-defined context to improve both alignment and promptability in novice workflows. 

---
# PipeWeaver: Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline 

**Authors**: Zhenliang Xue, Hanpeng Hu, Xing Chen, Yimin Jiang, Yixin Song, Zeyu Mi, Yibo Zhu, Daxin Jiang, Yubin Xia, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14145)  

**Abstract**: Large multimodal models (LMMs) have demonstrated excellent capabilities in both understanding and generation tasks with various modalities. While these models can accept flexible combinations of input data, their training efficiency suffers from two major issues: pipeline stage imbalance caused by heterogeneous model architectures, and training data dynamicity stemming from the diversity of multimodal data.
In this paper, we present PipeWeaver, a dynamic pipeline scheduling framework designed for LMM training. The core of PipeWeaver is dynamic interleaved pipeline, which searches for pipeline schedules dynamically tailored to current training batches. PipeWeaver addresses issues of LMM training with two techniques: adaptive modality-aware partitioning and efficient pipeline schedule search within a hierarchical schedule space. Meanwhile, PipeWeaver utilizes SEMU (Step Emulator), a training simulator for multimodal models, for accurate performance estimations, accelerated by spatial-temporal subgraph reuse to improve search efficiency. Experiments show that PipeWeaver can enhance LMM training efficiency by up to 97.3% compared to state-of-the-art systems, and demonstrate excellent adaptivity to LMM training's data dynamicity. 

---
# Learning Joint ID-Textual Representation for ID-Preserving Image Synthesis 

**Authors**: Zichuan Liu, Liming Jiang, Qing Yan, Yumin Jia, Hao Kang, Xin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14202)  

**Abstract**: We propose a novel framework for ID-preserving generation using a multi-modal encoding strategy rather than injecting identity features via adapters into pre-trained models. Our method treats identity and text as a unified conditioning input. To achieve this, we introduce FaceCLIP, a multi-modal encoder that learns a joint embedding space for both identity and textual semantics. Given a reference face and a text prompt, FaceCLIP produces a unified representation that encodes both identity and text, which conditions a base diffusion model to generate images that are identity-consistent and text-aligned. We also present a multi-modal alignment algorithm to train FaceCLIP, using a loss that aligns its joint representation with face, text, and image embedding spaces. We then build FaceCLIP-SDXL, an ID-preserving image synthesis pipeline by integrating FaceCLIP with Stable Diffusion XL (SDXL). Compared to prior methods, FaceCLIP-SDXL enables photorealistic portrait generation with better identity preservation and textual relevance. Extensive experiments demonstrate its quantitative and qualitative superiority. 

---
# Fashion-RAG: Multimodal Fashion Image Editing via Retrieval-Augmented Generation 

**Authors**: Fulvio Sanguigni, Davide Morelli, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2504.14011)  

**Abstract**: In recent years, the fashion industry has increasingly adopted AI technologies to enhance customer experience, driven by the proliferation of e-commerce platforms and virtual applications. Among the various tasks, virtual try-on and multimodal fashion image editing -- which utilizes diverse input modalities such as text, garment sketches, and body poses -- have become a key area of research. Diffusion models have emerged as a leading approach for such generative tasks, offering superior image quality and diversity. However, most existing virtual try-on methods rely on having a specific garment input, which is often impractical in real-world scenarios where users may only provide textual specifications. To address this limitation, in this work we introduce Fashion Retrieval-Augmented Generation (Fashion-RAG), a novel method that enables the customization of fashion items based on user preferences provided in textual form. Our approach retrieves multiple garments that match the input specifications and generates a personalized image by incorporating attributes from the retrieved items. To achieve this, we employ textual inversion techniques, where retrieved garment images are projected into the textual embedding space of the Stable Diffusion text encoder, allowing seamless integration of retrieved elements into the generative process. Experimental results on the Dress Code dataset demonstrate that Fashion-RAG outperforms existing methods both qualitatively and quantitatively, effectively capturing fine-grained visual details from retrieved garments. To the best of our knowledge, this is the first work to introduce a retrieval-augmented generation approach specifically tailored for multimodal fashion image editing. 

---
# Evaluating Menu OCR and Translation: A Benchmark for Aligning Human and Automated Evaluations in Large Vision-Language Models 

**Authors**: Zhanglin Wu, Tengfei Song, Ning Xie, Weidong Zhang, Mengli Zhu, Shuang Wu, Shiliang Sun, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13945)  

**Abstract**: The rapid advancement of large vision-language models (LVLMs) has significantly propelled applications in document understanding, particularly in optical character recognition (OCR) and multilingual translation. However, current evaluations of LVLMs, like the widely used OCRBench, mainly focus on verifying the correctness of their short-text responses and long-text responses with simple layout, while the evaluation of their ability to understand long texts with complex layout design is highly significant but largely overlooked. In this paper, we propose Menu OCR and Translation Benchmark (MOTBench), a specialized evaluation framework emphasizing the pivotal role of menu translation in cross-cultural communication. MOTBench requires LVLMs to accurately recognize and translate each dish, along with its price and unit items on a menu, providing a comprehensive assessment of their visual understanding and language processing capabilities. Our benchmark is comprised of a collection of Chinese and English menus, characterized by intricate layouts, a variety of fonts, and culturally specific elements across different languages, along with precise human annotations. Experiments show that our automatic evaluation results are highly consistent with professional human evaluation. We evaluate a range of publicly available state-of-the-art LVLMs, and through analyzing their output to identify the strengths and weaknesses in their performance, offering valuable insights to guide future advancements in LVLM development. MOTBench is available at this https URL. 

---
# Mozualization: Crafting Music and Visual Representation with Multimodal AI 

**Authors**: Wanfang Xu, Lixiang Zhao, Haiwen Song, Xinheng Song, Zhaolin Lu, Yu Liu, Min Chen, Eng Gee Lim, Lingyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13891)  

**Abstract**: In this work, we introduce Mozualization, a music generation and editing tool that creates multi-style embedded music by integrating diverse inputs, such as keywords, images, and sound clips (e.g., segments from various pieces of music or even a playful cat's meow). Our work is inspired by the ways people express their emotions -- writing mood-descriptive poems or articles, creating drawings with warm or cool tones, or listening to sad or uplifting music. Building on this concept, we developed a tool that transforms these emotional expressions into a cohesive and expressive song, allowing users to seamlessly incorporate their unique preferences and inspirations. To evaluate the tool and, more importantly, gather insights for its improvement, we conducted a user study involving nine music enthusiasts. The study assessed user experience, engagement, and the impact of interacting with and listening to the generated music. 

---
# Towards a Multimodal Document-grounded Conversational AI System for Education 

**Authors**: Karan Taneja, Anjali Singh, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.13884)  

**Abstract**: Multimedia learning using text and images has been shown to improve learning outcomes compared to text-only instruction. But conversational AI systems in education predominantly rely on text-based interactions while multimodal conversations for multimedia learning remain unexplored. Moreover, deploying conversational AI in learning contexts requires grounding in reliable sources and verifiability to create trust. We present MuDoC, a Multimodal Document-grounded Conversational AI system based on GPT-4o, that leverages both text and visuals from documents to generate responses interleaved with text and images. Its interface allows verification of AI generated content through seamless navigation to the source. We compare MuDoC to a text-only system to explore differences in learner engagement, trust in AI system, and their performance on problem-solving tasks. Our findings indicate that both visuals and verifiability of content enhance learner engagement and foster trust; however, no significant impact in performance was observed. We draw upon theories from cognitive and learning sciences to interpret the findings and derive implications, and outline future directions for the development of multimodal conversational AI systems in education. 

---
# The 1st EReL@MIR Workshop on Efficient Representation Learning for Multimodal Information Retrieval 

**Authors**: Junchen Fu, Xuri Ge, Xin Xin, Haitao Yu, Yue Feng, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2504.14788)  

**Abstract**: Multimodal representation learning has garnered significant attention in the AI community, largely due to the success of large pre-trained multimodal foundation models like LLaMA, GPT, Mistral, and CLIP. These models have achieved remarkable performance across various tasks of multimodal information retrieval (MIR), including web search, cross-modal retrieval, and recommender systems, etc. However, due to their enormous parameter sizes, significant efficiency challenges emerge across training, deployment, and inference stages when adapting these models' representation for IR tasks. These challenges present substantial obstacles to the practical adaptation of foundation models for representation learning in information retrieval tasks.
To address these pressing issues, we propose organizing the first EReL@MIR workshop at the Web Conference 2025, inviting participants to explore novel solutions, emerging problems, challenges, efficiency evaluation metrics and benchmarks. This workshop aims to provide a platform for both academic and industry researchers to engage in discussions, share insights, and foster collaboration toward achieving efficient and effective representation learning for multimodal information retrieval in the era of large foundation models. 

---
# Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration 

**Authors**: Hongji Li, Hanwen Du, Youhua Li, Junchen Fu, Chunxiao Li, Ziyi Zhuang, Jiakang Li, Yongxin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2504.14214)  

**Abstract**: The surge in multimedia content has led to the development of Multi-Modal Recommender Systems (MMRecs), which use diverse modalities such as text, images, videos, and audio for more personalized recommendations. However, MMRecs struggle with noisy data caused by misalignment among modal content and the gap between modal semantics and recommendation semantics. Traditional denoising methods are inadequate due to the complexity of multi-modal data. To address this, we propose a universal guided in-sync distillation denoising framework for multi-modal recommendation (GUIDER), designed to improve MMRecs by denoising user feedback. Specifically, GUIDER uses a re-calibration strategy to identify clean and noisy interactions from modal content. It incorporates a Denoising Bayesian Personalized Ranking (DBPR) loss function to handle implicit user feedback. Finally, it applies a denoising knowledge distillation objective based on Optimal Transport distance to guide the alignment from modality representations to recommendation semantics. GUIDER can be seamlessly integrated into existing MMRecs methods as a plug-and-play solution. Experimental results on four public datasets demonstrate its effectiveness and generalizability. Our source code is available at this https URL 

---
