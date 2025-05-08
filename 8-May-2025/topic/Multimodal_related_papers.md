# EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning 

**Authors**: Zhenghao Xing, Xiaowei Hu, Chi-Wing Fu, Wenhai Wang, Jifeng Dai, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced perception across text, vision, and audio, yet they often struggle with structured cross-modal reasoning, particularly when integrating audio and visual signals. We introduce EchoInk-R1, a reinforcement learning framework that enhances such reasoning in MLLMs. Built upon the Qwen2.5-Omni-7B foundation and optimized with Group Relative Policy Optimization (GRPO), EchoInk-R1 tackles multiple-choice question answering over synchronized audio-image pairs. To enable this, we curate AVQA-R1-6K, a dataset pairing such audio-image inputs with multiple-choice questions derived from OmniInstruct-v1. EchoInk-R1-7B achieves 85.77% accuracy on the validation set, outperforming the base model, which scores 80.53%, using only 562 reinforcement learning steps. Beyond accuracy, EchoInk-R1 demonstrates reflective reasoning by revisiting initial interpretations and refining responses when facing ambiguous multimodal inputs. These results suggest that lightweight reinforcement learning fine-tuning enhances cross-modal reasoning in MLLMs. EchoInk-R1 is the first framework to unify audio, visual, and textual modalities for general open-world reasoning via reinforcement learning. Code and data are publicly released to facilitate further research. 

---
# X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains 

**Authors**: Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03981)  

**Abstract**: Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General-domain text-based post-training can enable such strong generalizable reasoning. Leveraging this finding, we introduce X-Reasoner, a vision-language model post-trained solely on general-domain text for generalizable reasoning, using a two-stage approach: an initial supervised fine-tuning phase with distilled long chain-of-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-Reasoner successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing state-of-the-art models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-Reasoner's performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce X-Reasoner-Med, a medical-specialized variant that achieves new state of the art on numerous text-only and multimodal medical benchmarks. 

---
# VideoPath-LLaVA: Pathology Diagnostic Reasoning Through Video Instruction Tuning 

**Authors**: Trinh T.L. Vuong, Jin Tae Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2505.04192)  

**Abstract**: We present VideoPath-LLaVA, the first large multimodal model (LMM) in computational pathology that integrates three distinct image scenarios, single patch images, automatically keyframe-extracted clips, and manually segmented video pathology images, to mimic the natural diagnostic process of pathologists. By generating detailed histological descriptions and culminating in a definitive sign-out diagnosis, VideoPath-LLaVA bridges visual narratives with diagnostic reasoning.
Central to our approach is the VideoPath-Instruct dataset, comprising 4278 video and diagnosis-specific chain-of-thought instructional pairs sourced from educational histopathology videos on YouTube. Although high-quality data is critical for enhancing diagnostic reasoning, its creation is time-intensive and limited in volume. To overcome this challenge, we transfer knowledge from existing single-image instruction datasets to train on weakly annotated, keyframe-extracted clips, followed by fine-tuning on manually segmented videos. VideoPath-LLaVA establishes a new benchmark in pathology video analysis and offers a promising foundation for future AI systems that support clinical decision-making through integrated visual and diagnostic reasoning. Our code, data, and model are publicly available at this https URL. 

---
# GAME: Learning Multimodal Interactions via Graph Structures for Personality Trait Estimation 

**Authors**: Kangsheng Wang, Yuhang Li, Chengwei Ye, Yufei Lin, Huanzhen Zhang, Bohan Hu, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03846)  

**Abstract**: Apparent personality analysis from short videos poses significant chal-lenges due to the complex interplay of visual, auditory, and textual cues. In this paper, we propose GAME, a Graph-Augmented Multimodal Encoder designed to robustly model and fuse multi-source features for automatic personality prediction. For the visual stream, we construct a facial graph and introduce a dual-branch Geo Two-Stream Network, which combines Graph Convolutional Networks (GCNs) and Convolutional Neural Net-works (CNNs) with attention mechanisms to capture both structural and appearance-based facial cues. Complementing this, global context and iden-tity features are extracted using pretrained ResNet18 and VGGFace back-bones. To capture temporal dynamics, frame-level features are processed by a BiGRU enhanced with temporal attention modules. Meanwhile, audio representations are derived from the VGGish network, and linguistic se-mantics are captured via the XLM-Roberta transformer. To achieve effective multimodal integration, we propose a Channel Attention-based Fusion module, followed by a Multi-Layer Perceptron (MLP) regression head for predicting personality traits. Extensive experiments show that GAME con-sistently outperforms existing methods across multiple benchmarks, vali-dating its effectiveness and generalizability. 

---
# VideoLLM Benchmarks and Evaluation: A Survey 

**Authors**: Yogesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03829)  

**Abstract**: The rapid development of Large Language Models (LLMs) has catalyzed significant advancements in video understanding technologies. This survey provides a comprehensive analysis of benchmarks and evaluation methodologies specifically designed or used for Video Large Language Models (VideoLLMs). We examine the current landscape of video understanding benchmarks, discussing their characteristics, evaluation protocols, and limitations. The paper analyzes various evaluation methodologies, including closed-set, open-set, and specialized evaluations for temporal and spatiotemporal understanding tasks. We highlight the performance trends of state-of-the-art VideoLLMs across these benchmarks and identify key challenges in current evaluation frameworks. Additionally, we propose future research directions to enhance benchmark design, evaluation metrics, and protocols, including the need for more diverse, multimodal, and interpretability-focused benchmarks. This survey aims to equip researchers with a structured understanding of how to effectively evaluate VideoLLMs and identify promising avenues for advancing the field of video understanding with large language models. 

---
# Facilitating Video Story Interaction with Multi-Agent Collaborative System 

**Authors**: Yiwen Zhang, Jianing Hao, Zhan Wang, Hongling Sheng, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03807)  

**Abstract**: Video story interaction enables viewers to engage with and explore narrative content for personalized experiences. However, existing methods are limited to user selection, specially designed narratives, and lack customization. To address this, we propose an interactive system based on user intent. Our system uses a Vision Language Model (VLM) to enable machines to understand video stories, combining Retrieval-Augmented Generation (RAG) and a Multi-Agent System (MAS) to create evolving characters and scene experiences. It includes three stages: 1) Video story processing, utilizing VLM and prior knowledge to simulate human understanding of stories across three modalities. 2) Multi-space chat, creating growth-oriented characters through MAS interactions based on user queries and story stages. 3) Scene customization, expanding and visualizing various story scenes mentioned in dialogue. Applied to the Harry Potter series, our study shows the system effectively portrays emergent character social behavior and growth, enhancing the interactive experience in the video story world. 

---
# Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding 

**Authors**: Trilok Padhi, Ramneet Kaur, Adam D. Cobb, Manoj Acharya, Anirban Roy, Colin Samplawski, Brian Matejek, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03788)  

**Abstract**: We introduce a novel approach for calibrating uncertainty quantification (UQ) tailored for multi-modal large language models (LLMs). Existing state-of-the-art UQ methods rely on consistency among multiple responses generated by the LLM on an input query under diverse settings. However, these approaches often report higher confidence in scenarios where the LLM is consistently incorrect. This leads to a poorly calibrated confidence with respect to accuracy. To address this, we leverage cross-modal consistency in addition to self-consistency to improve the calibration of the multi-modal models. Specifically, we ground the textual responses to the visual inputs. The confidence from the grounding model is used to calibrate the overall confidence. Given that using a grounding model adds its own uncertainty in the pipeline, we apply temperature scaling - a widely accepted parametric calibration technique - to calibrate the grounding model's confidence in the accuracy of generated responses. We evaluate the proposed approach across multiple multi-modal tasks, such as medical question answering (Slake) and visual question answering (VQAv2), considering multi-modal models such as LLaVA-Med and LLaVA. The experiments demonstrate that the proposed framework achieves significantly improved calibration on both tasks. 

---
# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning 

**Authors**: Lang Feng, Weihao Tan, Zhiyi Lyu, Longtao Zheng, Haiyang Xu, Ming Yan, Fei Huang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.03792)  

**Abstract**: Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at this https URL. 

---
# An Adaptive Data-Resilient Multi-Modal Framework for Hierarchical Multi-Label Book Genre Identification 

**Authors**: Utsav Kumar Nareti, Soumi Chattopadhyay, Prolay Mallick, Suraj Kumar, Ayush Vikas Daga, Chandranath Adak, Adarsh Wase, Arjab Roy  

**Link**: [PDF](https://arxiv.org/pdf/2505.03839)  

**Abstract**: Identifying the finer details of a book's genres enhances user experience by enabling efficient book discovery and personalized recommendations, ultimately improving reader engagement and satisfaction. It also provides valuable insights into market trends and consumer preferences, allowing publishers and marketers to make data-driven decisions regarding book production and marketing strategies. While traditional book genre classification methods primarily rely on review data or textual analysis, incorporating additional modalities, such as book covers, blurbs, and metadata, can offer richer context and improve prediction accuracy. However, the presence of incomplete or noisy information across these modalities presents a significant challenge. This paper introduces IMAGINE (Intelligent Multi-modal Adaptive Genre Identification NEtwork), a framework designed to address these complexities. IMAGINE extracts robust feature representations from multiple modalities and dynamically selects the most informative sources based on data availability. It employs a hierarchical classification strategy to capture genre relationships and remains adaptable to varying input conditions. Additionally, we curate a hierarchical genre classification dataset that structures genres into a well-defined taxonomy, accommodating the diverse nature of literary works. IMAGINE integrates information from multiple sources and assigns multiple genre labels to each book, ensuring a more comprehensive classification. A key feature of our framework is its resilience to incomplete data, enabling accurate predictions even when certain modalities, such as text, images, or metadata, are missing or incomplete. Experimental results show that IMAGINE outperformed existing baselines in genre classification accuracy, particularly in scenarios with insufficient modality-specific data. 

---
