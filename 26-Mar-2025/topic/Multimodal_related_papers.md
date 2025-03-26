# GENIUS: A Generative Framework for Universal Multimodal Search 

**Authors**: Sungyeon Kim, Xinliang Zhu, Xiaofan Lin, Muhammet Bastan, Douglas Gray, Suha Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2503.19868)  

**Abstract**: Generative retrieval is an emerging approach in information retrieval that generates identifiers (IDs) of target data based on a query, providing an efficient alternative to traditional embedding-based retrieval methods. However, existing models are task-specific and fall short of embedding-based retrieval in performance. This paper proposes GENIUS, a universal generative retrieval framework supporting diverse tasks across multiple modalities and domains. At its core, GENIUS introduces modality-decoupled semantic quantization, transforming multimodal data into discrete IDs encoding both modality and semantics. Moreover, to enhance generalization, we propose a query augmentation that interpolates between a query and its target, allowing GENIUS to adapt to varied query forms. Evaluated on the M-BEIR benchmark, it surpasses prior generative methods by a clear margin. Unlike embedding-based retrieval, GENIUS consistently maintains high retrieval speed across database size, with competitive performance across multiple benchmarks. With additional re-ranking, GENIUS often achieves results close to those of embedding-based methods while preserving efficiency. 

---
# CoLLM: A Large Language Model for Composed Image Retrieval 

**Authors**: Chuong Huynh, Jinyu Yang, Ashish Tawari, Mubarak Shah, Son Tran, Raffay Hamid, Trishul Chilimbi, Abhinav Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.19910)  

**Abstract**: Composed Image Retrieval (CIR) is a complex task that aims to retrieve images based on a multimodal query. Typical training data consists of triplets containing a reference image, a textual description of desired modifications, and the target image, which are expensive and time-consuming to acquire. The scarcity of CIR datasets has led to zero-shot approaches utilizing synthetic triplets or leveraging vision-language models (VLMs) with ubiquitous web-crawled image-caption pairs. However, these methods have significant limitations: synthetic triplets suffer from limited scale, lack of diversity, and unnatural modification text, while image-caption pairs hinder joint embedding learning of the multimodal query due to the absence of triplet data. Moreover, existing approaches struggle with complex and nuanced modification texts that demand sophisticated fusion and understanding of vision and language modalities. We present CoLLM, a one-stop framework that effectively addresses these limitations. Our approach generates triplets on-the-fly from image-caption pairs, enabling supervised training without manual annotation. We leverage Large Language Models (LLMs) to generate joint embeddings of reference images and modification texts, facilitating deeper multimodal fusion. Additionally, we introduce Multi-Text CIR (MTCIR), a large-scale dataset comprising 3.4M samples, and refine existing CIR benchmarks (CIRR and Fashion-IQ) to enhance evaluation reliability. Experimental results demonstrate that CoLLM achieves state-of-the-art performance across multiple CIR benchmarks and settings. MTCIR yields competitive results, with up to 15% performance improvement. Our refined benchmarks provide more reliable evaluation metrics for CIR models, contributing to the advancement of this important field. 

---
# Gemma 3 Technical Report 

**Authors**: Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucińska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini  

**Link**: [PDF](https://arxiv.org/pdf/2503.19786)  

**Abstract**: We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. This version introduces vision understanding abilities, a wider coverage of languages and longer context - at least 128K tokens. We also change the architecture of the model to reduce the KV-cache memory that tends to explode with long context. This is achieved by increasing the ratio of local to global attention layers, and keeping the span on local attention short. The Gemma 3 models are trained with distillation and achieve superior performance to Gemma 2 for both pre-trained and instruction finetuned versions. In particular, our novel post-training recipe significantly improves the math, chat, instruction-following and multilingual abilities, making Gemma3-4B-IT competitive with Gemma2-27B-IT and Gemma3-27B-IT comparable to Gemini-1.5-Pro across benchmarks. We release all our models to the community. 

---
# Where is this coming from? Making groundedness count in the evaluation of Document VQA models 

**Authors**: Armineh Nourbakhsh, Siddharth Parekh, Pranav Shetty, Zhao Jin, Sameena Shah, Carolyn Rose  

**Link**: [PDF](https://arxiv.org/pdf/2503.19120)  

**Abstract**: Document Visual Question Answering (VQA) models have evolved at an impressive rate over the past few years, coming close to or matching human performance on some benchmarks. We argue that common evaluation metrics used by popular benchmarks do not account for the semantic and multimodal groundedness of a model's outputs. As a result, hallucinations and major semantic errors are treated the same way as well-grounded outputs, and the evaluation scores do not reflect the reasoning capabilities of the model. In response, we propose a new evaluation methodology that accounts for the groundedness of predictions with regard to the semantic characteristics of the output as well as the multimodal placement of the output within the input document. Our proposed methodology is parameterized in such a way that users can configure the score according to their preferences. We validate our scoring methodology using human judgment and show its potential impact on existing popular leaderboards. Through extensive analyses, we demonstrate that our proposed method produces scores that are a better indicator of a model's robustness and tends to give higher rewards to better-calibrated answers. 

---
# CAFe: Unifying Representation and Generation with Contrastive-Autoregressive Finetuning 

**Authors**: Hao Yu, Zhuokai Zhao, Shen Yan, Lukasz Korycki, Jianyu Wang, Baosheng He, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, Hanchao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19900)  

**Abstract**: The rapid advancement of large vision-language models (LVLMs) has driven significant progress in multimodal tasks, enabling models to interpret, reason, and generate outputs across both visual and textual domains. While excelling in generative tasks, existing LVLMs often face limitations in tasks requiring high-fidelity representation learning, such as generating image or text embeddings for retrieval. Recent work has proposed finetuning LVLMs for representational learning, but the fine-tuned model often loses its generative capabilities due to the representational learning training paradigm. To address this trade-off, we introduce CAFe, a contrastive-autoregressive fine-tuning framework that enhances LVLMs for both representation and generative tasks. By integrating a contrastive objective with autoregressive language modeling, our approach unifies these traditionally separate tasks, achieving state-of-the-art results in both multimodal retrieval and multimodal generative benchmarks, including object hallucination (OH) mitigation. CAFe establishes a novel framework that synergizes embedding and generative functionalities in a single model, setting a foundation for future multimodal models that excel in both retrieval precision and coherent output generation. 

---
# MIRAGE: Multimodal Immersive Reasoning and Guided Exploration for Red-Team Jailbreak Attacks 

**Authors**: Wenhao You, Bryan Hooi, Yiwei Wang, Youke Wang, Zong Ke, Ming-Hsuan Yang, Zi Huang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.19134)  

**Abstract**: While safety mechanisms have significantly progressed in filtering harmful text inputs, MLLMs remain vulnerable to multimodal jailbreaks that exploit their cross-modal reasoning capabilities. We present MIRAGE, a novel multimodal jailbreak framework that exploits narrative-driven context and role immersion to circumvent safety mechanisms in Multimodal Large Language Models (MLLMs). By systematically decomposing the toxic query into environment, role, and action triplets, MIRAGE constructs a multi-turn visual storytelling sequence of images and text using Stable Diffusion, guiding the target model through an engaging detective narrative. This process progressively lowers the model's defences and subtly guides its reasoning through structured contextual cues, ultimately eliciting harmful responses. In extensive experiments on the selected datasets with six mainstream MLLMs, MIRAGE achieves state-of-the-art performance, improving attack success rates by up to 17.5% over the best baselines. Moreover, we demonstrate that role immersion and structured semantic reconstruction can activate inherent model biases, facilitating the model's spontaneous violation of ethical safeguards. These results highlight critical weaknesses in current multimodal safety mechanisms and underscore the urgent need for more robust defences against cross-modal threats. 

---
# Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models 

**Authors**: Ilias Stogiannidis, Steven McDonagh, Sotirios A. Tsaftaris  

**Link**: [PDF](https://arxiv.org/pdf/2503.19707)  

**Abstract**: Vision-Language Models (VLMs) have recently emerged as powerful tools, excelling in tasks that integrate visual and textual comprehension, such as image captioning, visual question answering, and image-text retrieval. However, existing benchmarks for VLMs include spatial components, which often fail to isolate spatial reasoning from related tasks such as object detection or semantic comprehension. In this paper, we address these deficiencies with a multi-faceted approach towards understanding spatial reasoning. Informed by the diverse and multi-dimensional nature of human spatial reasoning abilities, we present a detailed analysis that first delineates the core elements of spatial reasoning: spatial relations, orientation and navigation, mental rotation, and spatial visualization, and then assesses the performance of these models in both synthetic and real-world images, bridging controlled and naturalistic contexts. We analyze 13 state-of-the-art Vision-Language Models, uncovering pivotal insights into their spatial reasoning performance. Our results reveal profound shortcomings in current VLMs, with average accuracy across the 13 models approximating random chance, highlighting spatial reasoning as a persistent obstacle. This work not only exposes the pressing need to advance spatial reasoning within VLMs but also establishes a solid platform for future exploration. Code available on GitHub (this https URL) and dataset available on HuggingFace (this https URL). 

---
# Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning 

**Authors**: Sky CH-Wang, Darshan Deshpande, Smaranda Muresan, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2503.19193)  

**Abstract**: We introduce Browsing Lost Unformed Recollections, a tip-of-the-tongue known-item search and reasoning benchmark for general AI assistants. BLUR introduces a set of 573 real-world validated questions that demand searching and reasoning across multi-modal and multilingual inputs, as well as proficient tool use, in order to excel on. Humans easily ace these questions (scoring on average 98%), while the best-performing system scores around 56%. To facilitate progress toward addressing this challenging and aspirational use case for general AI assistants, we release 350 questions through a public leaderboard, retain the answers to 250 of them, and have the rest as a private test set. 

---
# MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow 

**Authors**: Ziyue Wang, Junde Wu, Chang Han Low, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18968)  

**Abstract**: Developing reliable AI systems to assist human clinicians in multi-modal medical diagnosis has long been a key objective for researchers. Recently, Multi-modal Large Language Models (MLLMs) have gained significant attention and achieved success across various domains. With strong reasoning capabilities and the ability to perform diverse tasks based on user instructions, they hold great potential for enhancing medical diagnosis. However, directly applying MLLMs to the medical domain still presents challenges. They lack detailed perception of visual inputs, limiting their ability to perform quantitative image analysis, which is crucial for medical diagnostics. Additionally, MLLMs often exhibit hallucinations and inconsistencies in reasoning, whereas clinical diagnoses must adhere strictly to established criteria. To address these challenges, we propose MedAgent-Pro, an evidence-based reasoning agentic system designed to achieve reliable, explainable, and precise medical diagnoses. This is accomplished through a hierarchical workflow: at the task level, knowledge-based reasoning generate reliable diagnostic plans for specific diseases following retrieved clinical criteria. While at the case level, multiple tool agents process multi-modal inputs, analyze different indicators according to the plan, and provide a final diagnosis based on both quantitative and qualitative evidence. Comprehensive experiments on both 2D and 3D medical diagnosis tasks demonstrate the superiority and effectiveness of MedAgent-Pro, while case studies further highlight its reliability and interpretability. The code is available at this https URL. 

---
# SeLIP: Similarity Enhanced Contrastive Language Image Pretraining for Multi-modal Head MRI 

**Authors**: Zhiyang Liu, Dong Yang, Minghao Zhang, Hanyu Sun, Hong Wu, Huiying Wang, Wen Shen, Chao Chai, Shuang Xia  

**Link**: [PDF](https://arxiv.org/pdf/2503.19801)  

**Abstract**: Despite that deep learning (DL) methods have presented tremendous potential in many medical image analysis tasks, the practical applications of medical DL models are limited due to the lack of enough data samples with manual annotations. By noting that the clinical radiology examinations are associated with radiology reports that describe the images, we propose to develop a foundation model for multi-model head MRI by using contrastive learning on the images and the corresponding radiology findings. In particular, a contrastive learning framework is proposed, where a mixed syntax and semantic similarity matching metric is integrated to reduce the thirst of extreme large dataset in conventional contrastive learning framework. Our proposed similarity enhanced contrastive language image pretraining (SeLIP) is able to effectively extract more useful features. Experiments revealed that our proposed SeLIP performs well in many downstream tasks including image-text retrieval task, classification task, and image segmentation, which highlights the importance of considering the similarities among texts describing different images in developing medical image foundation models. 

---
# RGB-Th-Bench: A Dense benchmark for Visual-Thermal Understanding of Vision Language Models 

**Authors**: Mehdi Moshtaghi, Siavash H. Khajavi, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19654)  

**Abstract**: We introduce RGB-Th-Bench, the first benchmark designed to evaluate the ability of Vision-Language Models (VLMs) to comprehend RGB-Thermal image pairs. While VLMs have demonstrated remarkable progress in visual reasoning and multimodal understanding, their evaluation has been predominantly limited to RGB-based benchmarks, leaving a critical gap in assessing their capabilities in infrared vision tasks. Existing visible-infrared datasets are either task-specific or lack high-quality annotations necessary for rigorous model evaluation. To address these limitations, RGB-Th-Bench provides a comprehensive evaluation framework covering 14 distinct skill dimensions, with a total of 1,600+ expert-annotated Yes/No questions. The benchmark employs two accuracy metrics: a standard question-level accuracy and a stricter skill-level accuracy, which evaluates model robustness across multiple questions within each skill dimension. This design ensures a thorough assessment of model performance, including resilience to adversarial and hallucinated responses. We conduct extensive evaluations on 19 state-of-the-art VLMs, revealing significant performance gaps in RGB-Thermal understanding. Our results show that even the strongest models struggle with thermal image comprehension, with performance heavily constrained by their RGB-based capabilities. Additionally, the lack of large-scale application-specific and expert-annotated thermal-caption-pair datasets in pre-training is an important reason of the observed performance gap. RGB-Th-Bench highlights the urgent need for further advancements in multimodal learning to bridge the gap between visible and thermal image understanding. The dataset is available through this link, and the evaluation code will also be made publicly available. 

---
# OpenSDI: Spotting Diffusion-Generated Images in the Open World 

**Authors**: Yabin Wang, Zhiwu Huang, Xiaopeng Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.19653)  

**Abstract**: This paper identifies OpenSDI, a challenge for spotting diffusion-generated images in open-world settings. In response to this challenge, we define a new benchmark, the OpenSDI dataset (OpenSDID), which stands out from existing datasets due to its diverse use of large vision-language models that simulate open-world diffusion-based manipulations. Another outstanding feature of OpenSDID is its inclusion of both detection and localization tasks for images manipulated globally and locally by diffusion models. To address the OpenSDI challenge, we propose a Synergizing Pretrained Models (SPM) scheme to build up a mixture of foundation models. This approach exploits a collaboration mechanism with multiple pretrained foundation models to enhance generalization in the OpenSDI context, moving beyond traditional training by synergizing multiple pretrained models through prompting and attending strategies. Building on this scheme, we introduce MaskCLIP, an SPM-based model that aligns Contrastive Language-Image Pre-Training (CLIP) with Masked Autoencoder (MAE). Extensive evaluations on OpenSDID show that MaskCLIP significantly outperforms current state-of-the-art methods for the OpenSDI challenge, achieving remarkable relative improvements of 14.23% in IoU (14.11% in F1) and 2.05% in accuracy (2.38% in F1) compared to the second-best model in localization and detection tasks, respectively. Our dataset and code are available at this https URL. 

---
# FedMM-X: A Trustworthy and Interpretable Framework for Federated Multi-Modal Learning in Dynamic Environments 

**Authors**: Sree Bhargavi Balija  

**Link**: [PDF](https://arxiv.org/pdf/2503.19564)  

**Abstract**: As artificial intelligence systems increasingly operate in Real-world environments, the integration of multi-modal data sources such as vision, language, and audio presents both unprecedented opportunities and critical challenges for achieving trustworthy intelligence. In this paper, we propose a novel framework that unifies federated learning with explainable multi-modal reasoning to ensure trustworthiness in decentralized, dynamic settings. Our approach, called FedMM-X (Federated Multi-Modal Explainable Intelligence), leverages cross-modal consistency checks, client-level interpretability mechanisms, and dynamic trust calibration to address challenges posed by data heterogeneity, modality imbalance, and out-of-distribution generalization. Through rigorous evaluation across federated multi-modal benchmarks involving vision-language tasks, we demonstrate improved performance in both accuracy and interpretability while reducing vulnerabilities to adversarial and spurious correlations. Further, we introduce a novel trust score aggregation method to quantify global model reliability under dynamic client participation. Our findings pave the way toward developing robust, interpretable, and socially responsible AI systems in Real-world environments. 

---
# Show or Tell? Effectively prompting Vision-Language Models for semantic segmentation 

**Authors**: Niccolo Avogaro, Thomas Frick, Mattia Rigotti, Andrea Bartezzaghi, Filip Janicki, Cristiano Malossi, Konrad Schindler, Roy Assaf  

**Link**: [PDF](https://arxiv.org/pdf/2503.19647)  

**Abstract**: Large Vision-Language Models (VLMs) are increasingly being regarded as foundation models that can be instructed to solve diverse tasks by prompting, without task-specific training. We examine the seemingly obvious question: how to effectively prompt VLMs for semantic segmentation. To that end, we systematically evaluate the segmentation performance of several recent models guided by either text or visual prompts on the out-of-distribution MESS dataset collection. We introduce a scalable prompting scheme, few-shot prompted semantic segmentation, inspired by open-vocabulary segmentation and few-shot learning. It turns out that VLMs lag far behind specialist models trained for a specific segmentation task, by about 30% on average on the Intersection-over-Union metric. Moreover, we find that text prompts and visual prompts are complementary: each one of the two modes fails on many examples that the other one can solve. Our analysis suggests that being able to anticipate the most effective prompt modality can lead to a 11% improvement in performance. Motivated by our findings, we propose PromptMatcher, a remarkably simple training-free baseline that combines both text and visual prompts, achieving state-of-the-art results outperforming the best text-prompted VLM by 2.5%, and the top visual-prompted VLM by 3.5% on few-shot prompted semantic segmentation. 

---
# RoboFlamingo-Plus: Fusion of Depth and RGB Perception with Vision-Language Models for Enhanced Robotic Manipulation 

**Authors**: Sheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19510)  

**Abstract**: As robotic technologies advancing towards more complex multimodal interactions and manipulation tasks, the integration of advanced Vision-Language Models (VLMs) has become a key driver in the field. Despite progress with current methods, challenges persist in fusing depth and RGB information within 3D environments and executing tasks guided by linguistic instructions. In response to these challenges, we have enhanced the existing RoboFlamingo framework by introducing RoboFlamingo-Plus, which incorporates depth data into VLMs to significantly improve robotic manipulation performance. Our research achieves a nuanced fusion of RGB and depth information by integrating a pre-trained Vision Transformer (ViT) with a resampling technique, closely aligning this combined data with linguistic cues for superior multimodal understanding. The novelty of RoboFlamingo-Plus lies in its adaptation of inputs for depth data processing, leveraging a pre-trained resampler for depth feature extraction, and employing cross-attention mechanisms for optimal feature integration. These improvements allow RoboFlamingo-Plus to not only deeply understand 3D environments but also easily perform complex, language-guided tasks in challenging settings. Experimental results show that RoboFlamingo-Plus boosts robotic manipulation by 10-20% over current methods, marking a significant advancement. Codes and model weights are public at RoboFlamingo-Plus. 

---
# A-MESS: Anchor based Multimodal Embedding with Semantic Synchronization for Multimodal Intent Recognition 

**Authors**: Yaomin Shen, Xiaojian Lin, Wei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.19474)  

**Abstract**: In the domain of multimodal intent recognition (MIR), the objective is to recognize human intent by integrating a variety of modalities, such as language text, body gestures, and tones. However, existing approaches face difficulties adequately capturing the intrinsic connections between the modalities and overlooking the corresponding semantic representations of intent. To address these limitations, we present the Anchor-based Mul- timodal Embedding with Semantic Synchronization (A-MESS) framework. We first design an Anchor-based Multimodal Embed- ding (A-ME) module that employs an anchor-based embedding fusion mechanism to integrate multimodal inputs. Furthermore, we develop a Semantic Synchronization (SS) strategy with the Triplet Contrastive Learning pipeline, which optimizes the pro- cess by synchronizing multimodal representation with label de- scriptions produced by the large language model. Comprehensive experiments indicate that our A-MESS achieves state-of-the-art and provides substantial insight into multimodal representation and downstream tasks. 

---
# LRSCLIP: A Vision-Language Foundation Model for Aligning Remote Sensing Image with Longer Text 

**Authors**: Weizhi Chen, Jingbo Chen, Yupeng Deng, Jiansheng Chen, Yuman Feng, Zhihao Xi, Diyou Liu, Kai Li, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.19311)  

**Abstract**: This study addresses the technical bottlenecks in handling long text and the "hallucination" issue caused by insufficient short text information in remote sensing vision-language foundation models (VLFM). We propose a novel vision-language foundation model, LRSCLIP, and a multimodal dataset, LRS2M. The main contributions are as follows: (1) By integrating multi-source remote sensing data and adopting a large language model labeling strategy, we construct the LRS2M dataset, which contains 2 million image-text pairs, providing both short and long texts for the first time, thus solving the problem of semantic granularity limitations in existing datasets; (2) The design of the LRSCLIP architecture based on Long-CLIP's KPS module, which extends CLIP's text processing capacity and achieves fine-grained cross-modal feature alignment through a dual-text loss weighting mechanism. Experimental results show that LRSCLIP improves retrieval accuracy by 10\%-20\% over the Long-CLIP baseline in the zero-shot long-text cross-modal retrieval task. For the zero-shot short-text cross-modal retrieval task, LRSCLIP achieves improvements over the current best model, GeoRSCLIP, with increases of 0.17\%, 0.67\%, and 0.92\% in Text to Image R@1, Image to Text R@1, and mR on RSITMD, respectively, and 0.04\%, 2.93\%, and 1.28\% on RSICD. In the zero-shot image classification task (average accuracy=75.75\%) and semantic localization task (Rmi=0.7653), LRSCLIP achieves state-of-the-art performance. These results validate the dual advantages of fine-grained semantic understanding and global feature matching in LRSCLIP. This work provides a new benchmark model and data support for remote sensing multimodal learning. The related code has been open source and is available at this https URL. 

---
# CubeRobot: Grounding Language in Rubik's Cube Manipulation via Vision-Language Model 

**Authors**: Feiyang Wang, Xiaomin Yu, Wangyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19281)  

**Abstract**: Proving Rubik's Cube theorems at the high level represents a notable milestone in human-level spatial imagination and logic thinking and reasoning. Traditional Rubik's Cube robots, relying on complex vision systems and fixed algorithms, often struggle to adapt to complex and dynamic scenarios. To overcome this limitation, we introduce CubeRobot, a novel vision-language model (VLM) tailored for solving 3x3 Rubik's Cubes, empowering embodied agents with multimodal understanding and execution capabilities. We used the CubeCoT image dataset, which contains multiple-level tasks (43 subtasks in total) that humans are unable to handle, encompassing various cube states. We incorporate a dual-loop VisionCoT architecture and Memory Stream, a paradigm for extracting task-related features from VLM-generated planning queries, thus enabling CubeRobot to independent planning, decision-making, reflection and separate management of high- and low-level Rubik's Cube tasks. Furthermore, in low-level Rubik's Cube restoration tasks, CubeRobot achieved a high accuracy rate of 100%, similar to 100% in medium-level tasks, and achieved an accuracy rate of 80% in high-level tasks. 

---
# Context-Aware Semantic Segmentation: Enhancing Pixel-Level Understanding with Large Language Models for Advanced Vision Applications 

**Authors**: Ben Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2503.19276)  

**Abstract**: Semantic segmentation has made significant strides in pixel-level image understanding, yet it remains limited in capturing contextual and semantic relationships between objects. Current models, such as CNN and Transformer-based architectures, excel at identifying pixel-level features but fail to distinguish semantically similar objects (e.g., "doctor" vs. "nurse" in a hospital scene) or understand complex contextual scenarios (e.g., differentiating a running child from a regular pedestrian in autonomous driving). To address these limitations, we proposed a novel Context-Aware Semantic Segmentation framework that integrates Large Language Models (LLMs) with state-of-the-art vision backbones. Our hybrid model leverages the Swin Transformer for robust visual feature extraction and GPT-4 for enriching semantic understanding through text embeddings. A Cross-Attention Mechanism is introduced to align vision and language features, enabling the model to reason about context more effectively. Additionally, Graph Neural Networks (GNNs) are employed to model object relationships within the scene, capturing dependencies that are overlooked by traditional models. Experimental results on benchmark datasets (e.g., COCO, Cityscapes) demonstrate that our approach outperforms the existing methods in both pixel-level accuracy (mIoU) and contextual understanding (mAP). This work bridges the gap between vision and language, paving the path for more intelligent and context-aware vision systems in applications including autonomous driving, medical imaging, and robotics. 

---
# Unifying EEG and Speech for Emotion Recognition: A Two-Step Joint Learning Framework for Handling Missing EEG Data During Inference 

**Authors**: Upasana Tiwari, Rupayan Chakraborty, Sunil Kumar Kopparapu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18964)  

**Abstract**: Computer interfaces are advancing towards using multi-modalities to enable better human-computer interactions. The use of automatic emotion recognition (AER) can make the interactions natural and meaningful thereby enhancing the user experience. Though speech is the most direct and intuitive modality for AER, it is not reliable because it can be intentionally faked by humans. On the other hand, physiological modalities like EEG, are more reliable and impossible to fake. However, use of EEG is infeasible for realistic scenarios usage because of the need for specialized recording setup. In this paper, one of our primary aims is to ride on the reliability of the EEG modality to facilitate robust AER on the speech modality. Our approach uses both the modalities during training to reliably identify emotion at the time of inference, even in the absence of the more reliable EEG modality. We propose, a two-step joint multi-modal learning approach (JMML) that exploits both the intra- and inter- modal characteristics to construct emotion embeddings that enrich the performance of AER. In the first step, using JEC-SSL, intra-modal learning is done independently on the individual modalities. This is followed by an inter-modal learning using the proposed extended variant of deep canonically correlated cross-modal autoencoder (E-DCC-CAE). The approach learns the joint properties of both the modalities by mapping them into a common representation space, such that the modalities are maximally correlated. These emotion embeddings, hold properties of both the modalities there by enhancing the performance of ML classifier used for AER. Experimental results show the efficacy of the proposed approach. To best of our knowledge, this is the first attempt to combine speech and EEG with joint multi-modal learning approach for reliable AER. 

---
