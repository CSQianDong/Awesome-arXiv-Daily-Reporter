# TRACE: Real-Time Multimodal Common Ground Tracking in Situated Collaborative Dialogues 

**Authors**: Hannah VanderHoeven, Brady Bhalla, Ibrahim Khebour, Austin Youngren, Videep Venkatesha, Mariah Bradford, Jack Fitzgerald, Carlos Mabrey, Jingxuan Tu, Yifan Zhu, Kenneth Lai, Changsoo Jung, James Pustejovsky, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2503.09511)  

**Abstract**: We present TRACE, a novel system for live *common ground* tracking in situated collaborative tasks. With a focus on fast, real-time performance, TRACE tracks the speech, actions, gestures, and visual attention of participants, uses these multimodal inputs to determine the set of task-relevant propositions that have been raised as the dialogue progresses, and tracks the group's epistemic position and beliefs toward them as the task unfolds. Amid increased interest in AI systems that can mediate collaborations, TRACE represents an important step forward for agents that can engage with multiparty, multimodal discourse. 

---
# Florenz: Scaling Laws for Systematic Generalization in Vision-Language Models 

**Authors**: Julian Spravil, Sebastian Houben, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2503.09443)  

**Abstract**: Cross-lingual transfer enables vision-language models (VLMs) to perform vision tasks in various languages with training data only in one language. Current approaches rely on large pre-trained multilingual language models. However, they face the curse of multilinguality, sacrificing downstream task performance for multilingual capabilities, struggling with lexical ambiguities, and falling behind recent advances. In this work, we study the scaling laws of systematic generalization with monolingual VLMs for multilingual tasks, focusing on the impact of model size and seen training samples. We propose Florenz, a monolingual encoder-decoder VLM with 0.4B to 11.2B parameters combining the pre-trained VLM Florence-2 and the large language model Gemma-2. Florenz is trained with varying compute budgets on a synthetic dataset that features intentionally incomplete language coverage for image captioning, thus, testing generalization from the fully covered translation task. We show that not only does indirectly learning unseen task-language pairs adhere to a scaling law, but also that with our data generation pipeline and the proposed Florenz model family, image captioning abilities can emerge in a specific language even when only data for the translation task is available. Fine-tuning on a mix of downstream datasets yields competitive performance and demonstrates promising scaling trends in multimodal machine translation (Multi30K, CoMMuTE), lexical disambiguation (CoMMuTE), and image captioning (Multi30K, XM3600, COCO Karpathy). 

---
# MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding 

**Authors**: Zhoutong Ye, Mingze Sun, Huan-ang Gao, Chun Yu, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09348)  

**Abstract**: Large multimodal models (LMMs) have demonstrated significant potential as generalists in vision-language (VL) tasks. However, there remains a significant gap between state-of-the-art LMMs and human performance when it comes to complex tasks that require a combination of fundamental VL capabilities, as well as tasks involving the grounding of complex instructions. To thoroughly investigate the human-LMM gap and its underlying causes, we propose MOAT, a diverse benchmark with complex real-world VL tasks that are challenging for LMMs. Specifically, the tasks in MOAT require LMMs to engage in generalist problem solving by integrating fundamental VL capabilities such as reading text, counting, understanding spatial relations, grounding textual and visual instructions, etc. All these abilities fit into a taxonomy proposed by us that contains 10 fundamental VL capabilities, enabling MOAT to provide a fine-grained view of LMMs' strengths and weaknesses. Besides, MOAT is the first benchmark to explicitly evaluate LMMs' ability to ground complex text and visual instructions, which is essential to many real-world applications. We evaluate over 20 proprietary and open source LMMs, as well as humans, on MOAT, and found that humans achieved 82.7% accuracy while the best performing LMM (OpenAI o1) achieved only 38.8%. To guide future model development, we analyze common trends in our results and discuss the underlying causes of observed performance gaps between LMMs and humans, focusing on which VL capability forms the bottleneck in complex tasks, whether test time scaling improves performance on MOAT, and how tiling harms LMMs' capability to count. Code and data are available at this https URL. 

---
# xVLM2Vec: Adapting LVLM-based embedding models to multilinguality using Self-Knowledge Distillation 

**Authors**: Elio Musacchio, Lucia Siciliani, Pierpaolo Basile, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2503.09313)  

**Abstract**: In the current literature, most embedding models are based on the encoder-only transformer architecture to extract a dense and meaningful representation of the given input, which can be a text, an image, and more. With the recent advances in language modeling thanks to the introduction of Large Language Models, the possibility of extracting embeddings from these large and extensively trained models has been explored. However, current studies focus on textual embeddings in English, which is also the main language on which these models have been trained. Furthermore, there are very few models that consider multimodal and multilingual input. In light of this, we propose an adaptation methodology for Large Vision-Language Models trained on English language data to improve their performance in extracting multilingual and multimodal embeddings. Finally, we design and introduce a benchmark to evaluate the effectiveness of multilingual and multimodal embedding models. 

---
# Aligning to What? Limits to RLHF Based Alignment 

**Authors**: Logan Barnhart, Reza Akbarian Bafghi, Stephen Becker, Maziar Raissi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09025)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is increasingly used to align large language models (LLMs) with human preferences. However, the effectiveness of RLHF in addressing underlying biases remains unclear. This study investigates the relationship between RLHF and both covert and overt biases in LLMs, particularly focusing on biases against African Americans. We applied various RLHF techniques (DPO, ORPO, and RLOO) to Llama 3 8B and evaluated the covert and overt biases of the resulting models using matched-guise probing and explicit bias testing. We performed additional tests with DPO on different base models and datasets; among several implications, we found that SFT before RLHF calcifies model biases. Additionally, we extend the tools for measuring biases to multi-modal models. Through our experiments we collect evidence that indicates that current alignment techniques are inadequate for nebulous tasks such as mitigating covert biases, highlighting the need for capable datasets, data curating techniques, or alignment tools. 

---
# Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model 

**Authors**: Ali Vosoughi, Dimitra Emmanouilidou, Hannes Gamper  

**Link**: [PDF](https://arxiv.org/pdf/2503.09205)  

**Abstract**: Integrating audio and visual data for training multimodal foundational models remains challenging. We present Audio-Video Vector Alignment (AVVA), which aligns audiovisual (AV) scene content beyond mere temporal synchronization via a Large Language Model (LLM)-based data curation pipeline. Specifically, AVVA scores and selects high-quality training clips using Whisper (speech-based audio foundation model) for audio and DINOv2 for video within a dual-encoder contrastive learning framework. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate that this approach can achieve significant accuracy gains with substantially less curated data. For instance, AVVA yields a 7.6% improvement in top-1 accuracy for audio-to-video retrieval on VGGSound compared to ImageBind, despite training on only 192 hours of carefully filtered data (vs. 5800+ hours). Moreover, an ablation study highlights that trading data quantity for data quality improves performance, yielding respective top-3 accuracy increases of 47.8, 48.4, and 58.0 percentage points on AudioCaps, VALOR, and VGGSound over uncurated baselines. While these results underscore AVVA's data efficiency, we also discuss the overhead of LLM-driven curation and how it may be scaled or approximated in larger domains. Overall, AVVA provides a viable path toward more robust, text-free audiovisual learning with improved retrieval accuracy. 

---
# MindGYM: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions 

**Authors**: Zhe Xu, Daoyuan Chen, Zhenqing Ling, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.09499)  

**Abstract**: Large vision-language models (VLMs) face challenges in achieving robust, transferable reasoning abilities due to reliance on labor-intensive manual instruction datasets or computationally expensive self-supervised methods. To address these issues, we introduce MindGYM, a framework that enhances VLMs through synthetic self-challenging questions, consisting of three stages: (1) Seed Single-Hop Question Synthesis, generating cognitive questions across textual (e.g., logical deduction) and multimodal contexts (e.g., diagram-based queries) spanning eight semantic areas like ethical analysis; (2) Challenging Multi-Hop Question Synthesis, combining seed questions via diverse principles like bridging, visual-textual alignment, to create multi-step problems demanding deeper reasoning; and (3) Thinking-Induced Curriculum Fine-Tuning, a structured pipeline that progressively trains the model from scaffolded reasoning to standalone inference. By leveraging the model's self-synthesis capability, MindGYM achieves high data efficiency (e.g., +16% gains on MathVision-Mini with only 400 samples), computational efficiency (reducing both training and inference costs), and robust generalization across tasks. Extensive evaluations on seven benchmarks demonstrate superior performance over strong baselines, with notable improvements (+15.77% win rates) in reasoning depth and breadth validated via GPT-based scoring. MindGYM underscores the viability of self-challenging for refining VLM capabilities while minimizing human intervention and resource demands. Code and data are released to advance multimodal reasoning research. 

---
# CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games 

**Authors**: Peng Chen, Pi Bu, Yingyao Wang, Xinyi Wang, Ziming Wang, Jie Guo, Yingxiu Zhao, Qi Zhu, Jun Song, Siran Yang, Jiamang Wang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.09527)  

**Abstract**: Recent advances in Vision-Language-Action models (VLAs) have expanded the capabilities of embodied intelligence. However, significant challenges remain in real-time decision-making in complex 3D environments, which demand second-level responses, high-resolution perception, and tactical reasoning under dynamic conditions. To advance the field, we introduce CombatVLA, an efficient VLA model optimized for combat tasks in 3D action role-playing games(ARPGs). Specifically, our CombatVLA is a 3B model trained on video-action pairs collected by an action tracker, where the data is formatted as action-of-thought (AoT) sequences. Thereafter, CombatVLA seamlessly integrates into an action execution framework, allowing efficient inference through our truncated AoT strategy. Experimental results demonstrate that CombatVLA not only outperforms all existing models on the combat understanding benchmark but also achieves a 50-fold acceleration in game combat. Moreover, it has a higher task success rate than human players. We will open-source all resources, including the action tracker, dataset, benchmark, model weights, training code, and the implementation of the framework at this https URL. 

---
# Astrea: A MOE-based Visual Understanding Model with Progressive Alignment 

**Authors**: Xiaoda Yang, JunYu Lu, Hongshun Qiu, Sijing Li, Hao Li, Shengpeng Ji, Xudong Tang, Jiayang Xu, Jiaqi Duan, Ziyue Jiang, Cong Lin, Sihang Cai, Zejian Xie, Zhuoyang Song, Songxin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09445)  

**Abstract**: Vision-Language Models (VLMs) based on Mixture-of-Experts (MoE) architectures have emerged as a pivotal paradigm in multimodal understanding, offering a powerful framework for integrating visual and linguistic information. However, the increasing complexity and diversity of tasks present significant challenges in coordinating load balancing across heterogeneous visual experts, where optimizing one specialist's performance often compromises others' capabilities. To address task heterogeneity and expert load imbalance, we propose Astrea, a novel multi-expert collaborative VLM architecture based on progressive pre-alignment. Astrea introduces three key innovations: 1) A heterogeneous expert coordination mechanism that integrates four specialized models (detection, segmentation, classification, captioning) into a comprehensive expert matrix covering essential visual comprehension elements; 2) A dynamic knowledge fusion strategy featuring progressive pre-alignment to harmonize experts within the VLM latent space through contrastive learning, complemented by probabilistically activated stochastic residual connections to preserve knowledge continuity; 3) An enhanced optimization framework utilizing momentum contrastive learning for long-range dependency modeling and adaptive weight allocators for real-time expert contribution calibration. Extensive evaluations across 12 benchmark tasks spanning VQA, image captioning, and cross-modal retrieval demonstrate Astrea's superiority over state-of-the-art models, achieving an average performance gain of +4.7\%. This study provides the first empirical demonstration that progressive pre-alignment strategies enable VLMs to overcome task heterogeneity limitations, establishing new methodological foundations for developing general-purpose multimodal agents. 

---
# Multimodal Language Modeling for High-Accuracy Single Cell Transcriptomics Analysis and Generation 

**Authors**: Yaorui Shi, Jiaqi Yang, Sihang Li, Junfeng Fang, Xiang Wang, Zhiyuan Liu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09427)  

**Abstract**: Pre-trained language models (PLMs) have revolutionized scientific research, yet their application to single-cell analysis remains limited. Text PLMs cannot process single-cell RNA sequencing data, while cell PLMs lack the ability to handle free text, restricting their use in multimodal tasks. Existing efforts to bridge these modalities often suffer from information loss or inadequate single-modal pre-training, leading to suboptimal performances. To address these challenges, we propose Single-Cell MultiModal Generative Pre-trained Transformer (scMMGPT), a unified PLM for joint cell and text modeling. scMMGPT effectively integrates the state-of-the-art cell and text PLMs, facilitating cross-modal knowledge sharing for improved performance. To bridge the text-cell modality gap, scMMGPT leverages dedicated cross-modal projectors, and undergoes extensive pre-training on 27 million cells -- the largest dataset for multimodal cell-text PLMs to date. This large-scale pre-training enables scMMGPT to excel in joint cell-text tasks, achieving an 84\% relative improvement of textual discrepancy for cell description generation, 20.5\% higher accuracy for cell type annotation, and 4\% improvement in $k$-NN accuracy for text-conditioned pseudo-cell generation, outperforming baselines. 

---
# AI-based Framework for Robust Model-Based Connector Mating in Robotic Wire Harness Installation 

**Authors**: Claudius Kienle, Benjamin Alt, Finn Schneider, Tobias Pertlwieser, Rainer Jäkel, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2503.09409)  

**Abstract**: Despite the widespread adoption of industrial robots in automotive assembly, wire harness installation remains a largely manual process, as it requires precise and flexible manipulation. To address this challenge, we design a novel AI-based framework that automates cable connector mating by integrating force control with deep visuotactile learning. Our system optimizes search-and-insertion strategies using first-order optimization over a multimodal transformer architecture trained on visual, tactile, and proprioceptive data. Additionally, we design a novel automated data collection and optimization pipeline that minimizes the need for machine learning expertise. The framework optimizes robot programs that run natively on standard industrial controllers, permitting human experts to audit and certify them. Experimental validations on a center console assembly task demonstrate significant improvements in cycle times and robustness compared to conventional robot programming approaches. Videos are available under this https URL. 

---
# DAVE: Diagnostic benchmark for Audio Visual Evaluation 

**Authors**: Gorjan Radevski, Teodora Popordanoska, Matthew B. Blaschko, Tinne Tuytelaars  

**Link**: [PDF](https://arxiv.org/pdf/2503.09321)  

**Abstract**: Audio-visual understanding is a rapidly evolving field that seeks to integrate and interpret information from both auditory and visual modalities. Despite recent advances in multi-modal learning, existing benchmarks often suffer from strong visual bias -- where answers can be inferred from visual data alone -- and provide only aggregate scores that conflate multiple sources of error. This makes it difficult to determine whether models struggle with visual understanding, audio interpretation, or audio-visual alignment. In this work, we introduce DAVE (Diagnostic Audio Visual Evaluation), a novel benchmark dataset designed to systematically evaluate audio-visual models across controlled challenges. DAVE alleviates existing limitations by (i) ensuring both modalities are necessary to answer correctly and (ii) decoupling evaluation into atomic subcategories. Our detailed analysis of state-of-the-art models reveals specific failure modes and provides targeted insights for improvement. By offering this standardized diagnostic framework, we aim to facilitate more robust development of audio-visual models. The dataset is released: this https URL 

---
# NVP-HRI: Zero Shot Natural Voice and Posture-based Human-Robot Interaction via Large Language Model 

**Authors**: Yuzhi Lai, Shenghai Yuan, Youssef Nassar, Mingyu Fan, Thomas Weber, Matthias Rätsch  

**Link**: [PDF](https://arxiv.org/pdf/2503.09335)  

**Abstract**: Effective Human-Robot Interaction (HRI) is crucial for future service robots in aging societies. Existing solutions are biased toward only well-trained objects, creating a gap when dealing with new objects. Currently, HRI systems using predefined gestures or language tokens for pretrained objects pose challenges for all individuals, especially elderly ones. These challenges include difficulties in recalling commands, memorizing hand gestures, and learning new names. This paper introduces NVP-HRI, an intuitive multi-modal HRI paradigm that combines voice commands and deictic posture. NVP-HRI utilizes the Segment Anything Model (SAM) to analyze visual cues and depth data, enabling precise structural object representation. Through a pre-trained SAM network, NVP-HRI allows interaction with new objects via zero-shot prediction, even without prior knowledge. NVP-HRI also integrates with a large language model (LLM) for multimodal commands, coordinating them with object selection and scene distribution in real time for collision-free trajectory solutions. We also regulate the action sequence with the essential control syntax to reduce LLM hallucination risks. The evaluation of diverse real-world tasks using a Universal Robot showcased up to 59.2\% efficiency improvement over traditional gesture control, as illustrated in the video this https URL. Our code and design will be openly available at this https URL. 

---
# In-Context Defense in Computer Agents: An Empirical Study 

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2503.09241)  

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior. 

---
# Everything Can Be Described in Words: A Simple Unified Multi-Modal Framework with Semantic and Temporal Alignment 

**Authors**: Xiaowei Bi, Zheyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09081)  

**Abstract**: Long Video Question Answering (LVQA) is challenging due to the need for temporal reasoning and large-scale multimodal data processing. Existing methods struggle with retrieving cross-modal information from long videos, especially when relevant details are sparsely distributed. We introduce UMaT (Unified Multi-modal as Text), a retrieval-augmented generation (RAG) framework that efficiently processes extremely long videos while maintaining cross-modal coherence. UMaT converts visual and auditory data into a unified textual representation, ensuring semantic and temporal alignment. Short video clips are analyzed using a vision-language model, while automatic speech recognition (ASR) transcribes dialogue. These text-based representations are structured into temporally aligned segments, with adaptive filtering to remove redundancy and retain salient details. The processed data is embedded into a vector database, enabling precise retrieval of dispersed yet relevant content. Experiments on a benchmark LVQA dataset show that UMaT outperforms existing methods in multimodal integration, long-form video understanding, and sparse information retrieval. Its scalability and interpretability allow it to process videos over an hour long while maintaining semantic and temporal coherence. These findings underscore the importance of structured retrieval and multimodal synchronization for advancing LVQA and long-form AI systems. 

---
# Multi-Modal Foundation Models for Computational Pathology: A Survey 

**Authors**: Dong Li, Guihong Wan, Xintao Wu, Xinyu Wu, Xiaohui Chen, Yi He, Christine G. Lian, Peter K. Sorger, Yevgeniy R. Semenov, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09091)  

**Abstract**: Foundation models have emerged as a powerful paradigm in computational pathology (CPath), enabling scalable and generalizable analysis of histopathological images. While early developments centered on uni-modal models trained solely on visual data, recent advances have highlighted the promise of multi-modal foundation models that integrate heterogeneous data sources such as textual reports, structured domain knowledge, and molecular profiles. In this survey, we provide a comprehensive and up-to-date review of multi-modal foundation models in CPath, with a particular focus on models built upon hematoxylin and eosin (H&E) stained whole slide images (WSIs) and tile-level representations. We categorize 32 state-of-the-art multi-modal foundation models into three major paradigms: vision-language, vision-knowledge graph, and vision-gene expression. We further divide vision-language models into non-LLM-based and LLM-based approaches. Additionally, we analyze 28 available multi-modal datasets tailored for pathology, grouped into image-text pairs, instruction datasets, and image-other modality pairs. Our survey also presents a taxonomy of downstream tasks, highlights training and evaluation strategies, and identifies key challenges and future directions. We aim for this survey to serve as a valuable resource for researchers and practitioners working at the intersection of pathology and AI. 

---
# Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis 

**Authors**: Letian Zhang, Quan Cui, Bingchen Zhao, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08741)  

**Abstract**: The success of multi-modal large language models (MLLMs) has been largely attributed to the large-scale training data. However, the training data of many MLLMs is unavailable due to privacy concerns. The expensive and labor-intensive process of collecting multi-modal data further exacerbates the problem. Is it possible to synthesize multi-modal training data automatically without compromising diversity and quality? In this paper, we propose a new method, Oasis, to synthesize high-quality multi-modal data with only images. Oasis breaks through traditional methods by prompting only images to the MLLMs, thus extending the data diversity by a large margin. Our method features a delicate quality control method which ensures the data quality. We collected over 500k data and conducted incremental experiments on LLaVA-NeXT. Extensive experiments demonstrate that our method can significantly improve the performance of MLLMs. The image-based synthesis also allows us to focus on the specific-domain ability of MLLMs. Code and data will be publicly available. 

---
# Versatile Multimodal Controls for Whole-Body Talking Human Animation 

**Authors**: Zheng Qin, Ruobing Zheng, Yabing Wang, Tianqi Li, Zixin Zhu, Minghui Yang, Ming Yang, Le Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08714)  

**Abstract**: Human animation from a single reference image shall be flexible to synthesize whole-body motion for either a headshot or whole-body portrait, where the motions are readily controlled by audio signal and text prompts. This is hard for most existing methods as they only support producing pre-specified head or half-body motion aligned with audio inputs. In this paper, we propose a versatile human animation method, i.e., VersaAnimator, which generates whole-body talking human from arbitrary portrait images, not only driven by audio signal but also flexibly controlled by text prompts. Specifically, we design a text-controlled, audio-driven motion generator that produces whole-body motion representations in 3D synchronized with audio inputs while following textual motion descriptions. To promote natural smooth motion, we propose a code-pose translation module to link VAE codebooks with 2D DWposes extracted from template videos. Moreover, we introduce a multi-modal video diffusion that generates photorealistic human animation from a reference image according to both audio inputs and whole-body motion representations. Extensive experiments show that VersaAnimator outperforms existing methods in visual quality, identity preservation, and audio-lip synchronization. 

---
# SIMAC: A Semantic-Driven Integrated Multimodal Sensing And Communication Framework 

**Authors**: Yubo Peng, Luping Xiang, Kun Yang, Feibo Jiang, Kezhi Wang, Dapeng Oliver Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08726)  

**Abstract**: Traditional single-modality sensing faces limitations in accuracy and capability, and its decoupled implementation with communication systems increases latency in bandwidth-constrained environments. Additionally, single-task-oriented sensing systems fail to address users' diverse demands. To overcome these challenges, we propose a semantic-driven integrated multimodal sensing and communication (SIMAC) framework. This framework leverages a joint source-channel coding architecture to achieve simultaneous sensing decoding and transmission of sensing results. Specifically, SIMAC first introduces a multimodal semantic fusion (MSF) network, which employs two extractors to extract semantic information from radar signals and images, respectively. MSF then applies cross-attention mechanisms to fuse these unimodal features and generate multimodal semantic representations. Secondly, we present a large language model (LLM)-based semantic encoder (LSE), where relevant communication parameters and multimodal semantics are mapped into a unified latent space and input to the LLM, enabling channel-adaptive semantic encoding. Thirdly, a task-oriented sensing semantic decoder (SSD) is proposed, in which different decoded heads are designed according to the specific needs of tasks. Simultaneously, a multi-task learning strategy is introduced to train the SIMAC framework, achieving diverse sensing services. Finally, experimental simulations demonstrate that the proposed framework achieves diverse sensing services and higher accuracy. 

---
