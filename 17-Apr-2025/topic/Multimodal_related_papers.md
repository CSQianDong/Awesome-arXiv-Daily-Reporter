# PATFinger: Prompt-Adapted Transferable Fingerprinting against Unauthorized Multimodal Dataset Usage 

**Authors**: Wenyi Zhang, Ju Jia, Xiaojun Jia, Yihao Huang, Xinfeng Li, Cong Wu, Lina Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11509)  

**Abstract**: The multimodal datasets can be leveraged to pre-train large-scale vision-language models by providing cross-modal semantics. Current endeavors for determining the usage of datasets mainly focus on single-modal dataset ownership verification through intrusive methods and non-intrusive techniques, while cross-modal approaches remain under-explored. Intrusive methods can adapt to multimodal datasets but degrade model accuracy, while non-intrusive methods rely on label-driven decision boundaries that fail to guarantee stable behaviors for verification. To address these issues, we propose a novel prompt-adapted transferable fingerprinting scheme from a training-free perspective, called PATFinger, which incorporates the global optimal perturbation (GOP) and the adaptive prompts to capture dataset-specific distribution characteristics. Our scheme utilizes inherent dataset attributes as fingerprints instead of compelling the model to learn triggers. The GOP is derived from the sample distribution to maximize embedding drifts between different modalities. Subsequently, our PATFinger re-aligns the adaptive prompt with GOP samples to capture the cross-modal interactions on the carefully crafted surrogate model. This allows the dataset owner to check the usage of datasets by observing specific prediction behaviors linked to the PATFinger during retrieval queries. Extensive experiments demonstrate the effectiveness of our scheme against unauthorized multimodal dataset usage on various cross-modal retrieval architectures by 30% over state-of-the-art baselines. 

---
# SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models 

**Authors**: Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.11468)  

**Abstract**: This work revisits the dominant supervised fine-tuning (SFT) then reinforcement learning (RL) paradigm for training Large Vision-Language Models (LVLMs), and reveals a key finding: SFT can significantly undermine subsequent RL by inducing ``pseudo reasoning paths'' imitated from expert models. While these paths may resemble the native reasoning paths of RL models, they often involve prolonged, hesitant, less informative steps, and incorrect reasoning. To systematically study this effect, we introduce VLAA-Thinking, a new multimodal dataset designed to support reasoning in LVLMs. Constructed via a six-step pipeline involving captioning, reasoning distillation, answer rewrite and verification, VLAA-Thinking comprises high-quality, step-by-step visual reasoning traces for SFT, along with a more challenging RL split from the same data source. Using this dataset, we conduct extensive experiments comparing SFT, RL and their combinations. Results show that while SFT helps models learn reasoning formats, it often locks aligned models into imitative, rigid reasoning modes that impede further learning. In contrast, building on the Group Relative Policy Optimization (GRPO) with a novel mixed reward module integrating both perception and cognition signals, our RL approach fosters more genuine, adaptive reasoning behavior. Notably, our model VLAA-Thinker, based on Qwen2.5VL 3B, achieves top-1 performance on Open LMM Reasoning Leaderboard (this https URL) among 4B scale LVLMs, surpassing the previous state-of-the-art by 1.8%. We hope our findings provide valuable insights in developing reasoning-capable LVLMs and can inform future research in this area. 

---
# Semantic Matters: Multimodal Features for Affective Analysis 

**Authors**: Tobias Hallmen, Robin-Nico Kampa, Fabian Deuser, Norbert Oswald, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2504.11460)  

**Abstract**: In this study, we present our methodology for two tasks: the Behavioural Ambivalence/Hesitancy (BAH) Recognition Challenge and the Emotional Mimicry Intensity (EMI) Estimation Challenge, both conducted as part of the 8th Workshop and Competition on Affective & Behavior Analysis in-the-wild. Building on previous work, we utilize a Wav2Vec 2.0 model pre-trained on a large podcast dataset to extract various audio features, capturing both linguistic and paralinguistic information. Our approach incorporates a valence-arousal-dominance (VAD) module derived from Wav2Vec 2.0, a BERT-like encoder, and a vision transformer (ViT) with predictions subsequently processed through a long short-term memory (LSTM) architecture for temporal modeling. In this iteration, we integrate the textual and visual modality into our analysis, recognizing that semantic content provides valuable contextual cues and underscoring that the meaning of speech often conveys more critical insights than its acoustic counterpart alone. Fusing in the vision modality helps in some cases to interpret the textual modality more precisely. This combined approach yields significant performance improvements over baseline methods. 

---
# Graph-Driven Multimodal Feature Learning Framework for Apparent Personality Assessment 

**Authors**: Kangsheng Wang, Chengwei Ye, Huanzhen Zhang, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11515)  

**Abstract**: Predicting personality traits automatically has become a challenging problem in computer vision. This paper introduces an innovative multimodal feature learning framework for personality analysis in short video clips. For visual processing, we construct a facial graph and design a Geo-based two-stream network incorporating an attention mechanism, leveraging both Graph Convolutional Networks (GCN) and Convolutional Neural Networks (CNN) to capture static facial expressions. Additionally, ResNet18 and VGGFace networks are employed to extract global scene and facial appearance features at the frame level. To capture dynamic temporal information, we integrate a BiGRU with a temporal attention module for extracting salient frame representations. To enhance the model's robustness, we incorporate the VGGish CNN for audio-based features and XLM-Roberta for text-based features. Finally, a multimodal channel attention mechanism is introduced to integrate different modalities, and a Multi-Layer Perceptron (MLP) regression model is used to predict personality traits. Experimental results confirm that our proposed framework surpasses existing state-of-the-art approaches in performance. 

---
# FLIP Reasoning Challenge 

**Authors**: Andreas Plesner, Turlan Kuzhagaliyev, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2504.12256)  

**Abstract**: Over the past years, advances in artificial intelligence (AI) have demonstrated how AI can solve many perception and generation tasks, such as image classification and text writing, yet reasoning remains a challenge. This paper introduces the FLIP dataset, a benchmark for evaluating AI reasoning capabilities based on human verification tasks on the Idena blockchain. FLIP challenges present users with two orderings of 4 images, requiring them to identify the logically coherent one. By emphasizing sequential reasoning, visual storytelling, and common sense, FLIP provides a unique testbed for multimodal AI systems. Our experiments evaluate state-of-the-art models, leveraging both vision-language models (VLMs) and large language models (LLMs). Results reveal that even the best open-sourced and closed-sourced models achieve maximum accuracies of 75.5% and 77.9%, respectively, in zero-shot settings, compared to human performance of 95.3%. Captioning models aid reasoning models by providing text descriptions of images, yielding better results than when using the raw images directly, 69.6% vs. 75.2% for Gemini 1.5 Pro. Combining the predictions from 15 models in an ensemble increases the accuracy to 85.2%. These findings highlight the limitations of existing reasoning models and the need for robust multimodal benchmarks like FLIP. The full codebase and dataset will be available at this https URL. 

---
# Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions 

**Authors**: Yifei Dong, Fengyi Wu, Sanjian Zhang, Guangyu Chen, Yuzhi Hu, Masumi Yano, Jingdong Sun, Siyu Huang, Feng Liu, Qi Dai, Zhi-Qi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11967)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are indispensable for infrastructure inspection, surveillance, and related tasks, yet they also introduce critical security challenges. This survey provides a wide-ranging examination of the anti-UAV domain, centering on three core objectives-classification, detection, and tracking-while detailing emerging methodologies such as diffusion-based data synthesis, multi-modal fusion, vision-language modeling, self-supervised learning, and reinforcement learning. We systematically evaluate state-of-the-art solutions across both single-modality and multi-sensor pipelines (spanning RGB, infrared, audio, radar, and RF) and discuss large-scale as well as adversarially oriented benchmarks. Our analysis reveals persistent gaps in real-time performance, stealth detection, and swarm-based scenarios, underscoring pressing needs for robust, adaptive anti-UAV systems. By highlighting open research directions, we aim to foster innovation and guide the development of next-generation defense strategies in an era marked by the extensive use of UAVs. 

---
# Towards Explainable Fusion and Balanced Learning in Multimodal Sentiment Analysis 

**Authors**: Miaosen Luo, Yuncheng Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2504.12151)  

**Abstract**: Multimodal Sentiment Analysis (MSA) faces two critical challenges: the lack of interpretability in the decision logic of multimodal fusion and modality imbalance caused by disparities in inter-modal information density. To address these issues, we propose KAN-MCP, a novel framework that integrates the interpretability of Kolmogorov-Arnold Networks (KAN) with the robustness of the Multimodal Clean Pareto (MCPareto) framework. First, KAN leverages its univariate function decomposition to achieve transparent analysis of cross-modal interactions. This structural design allows direct inspection of feature transformations without relying on external interpretation tools, thereby ensuring both high expressiveness and interpretability. Second, the proposed MCPareto enhances robustness by addressing modality imbalance and noise interference. Specifically, we introduce the Dimensionality Reduction and Denoising Modal Information Bottleneck (DRD-MIB) method, which jointly denoises and reduces feature dimensionality. This approach provides KAN with discriminative low-dimensional inputs to reduce the modeling complexity of KAN while preserving critical sentiment-related information. Furthermore, MCPareto dynamically balances gradient contributions across modalities using the purified features output by DRD-MIB, ensuring lossless transmission of auxiliary signals and effectively alleviating modality imbalance. This synergy of interpretability and robustness not only achieves superior performance on benchmark datasets such as CMU-MOSI, CMU-MOSEI, and CH-SIMS v2 but also offers an intuitive visualization interface through KAN's interpretable architecture. 

---
# Can GPT tell us why these images are synthesized? Empowering Multimodal Large Language Models for Forensics 

**Authors**: Yiran He, Yun Cao, Bowen Yang, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11686)  

**Abstract**: The rapid development of generative AI facilitates content creation and makes image manipulation easier and more difficult to detect. While multimodal Large Language Models (LLMs) have encoded rich world knowledge, they are not inherently tailored for combating AI-generated Content (AIGC) and struggle to comprehend local forgery details. In this work, we investigate the application of multimodal LLMs in forgery detection. We propose a framework capable of evaluating image authenticity, localizing tampered regions, providing evidence, and tracing generation methods based on semantic tampering clues. Our method demonstrates that the potential of LLMs in forgery analysis can be effectively unlocked through meticulous prompt engineering and the application of few-shot learning techniques. We conduct qualitative and quantitative experiments and show that GPT4V can achieve an accuracy of 92.1% in Autosplice and 86.3% in LaMa, which is competitive with state-of-the-art AIGC detection methods. We further discuss the limitations of multimodal LLMs in such tasks and propose potential improvements. 

---
# Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset 

**Authors**: Muhammad Shahid Muneer, Simon S. Woo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11707)  

**Abstract**: In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the web society. To help prevent misuse of T2I models and create a safer web environment for users features like NSFW filters and post-hoc security checks are used in these models. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, adversarial attacks on text and image modalities can easily outplay defensive measures. %Exploiting such leads to the growing concern of preventing adversarial attacks on text and image modalities. Moreover, there is currently no robust multimodal NSFW dataset that includes both prompt and image pairs and adversarial examples. This work proposes a million-scale prompt and image dataset generated using open-source diffusion models. Second, we develop a multimodal defense to distinguish safe and NSFW text and images, which is robust against adversarial attacks and directly alleviates current challenges. Our extensive experiments show that our model performs well against existing SOTA NSFW detection methods in terms of accuracy and recall, drastically reducing the Attack Success Rate (ASR) in multimodal adversarial attack scenarios. Code: this https URL. 

---
# Toward Aligning Human and Robot Actions via Multi-Modal Demonstration Learning 

**Authors**: Azizul Zahid, Jie Fan, Farong Wang, Ashton Dy, Sai Swaminathan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11493)  

**Abstract**: Understanding action correspondence between humans and robots is essential for evaluating alignment in decision-making, particularly in human-robot collaboration and imitation learning within unstructured environments. We propose a multimodal demonstration learning framework that explicitly models human demonstrations from RGB video with robot demonstrations in voxelized RGB-D space. Focusing on the "pick and place" task from the RH20T dataset, we utilize data from 5 users across 10 diverse scenes. Our approach combines ResNet-based visual encoding for human intention modeling and a Perceiver Transformer for voxel-based robot action prediction. After 2000 training epochs, the human model reaches 71.67% accuracy, and the robot model achieves 71.8% accuracy, demonstrating the framework's potential for aligning complex, multimodal human and robot behaviors in manipulation tasks. 

---
# SDIGLM: Leveraging Large Language Models and Multi-Modal Chain of Thought for Structural Damage Identification 

**Authors**: Yunkai Zhang, Shiyin Wei, Yong Huang, Yawu Su, Shanshan Lu, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11477)  

**Abstract**: Existing computer vision(CV)-based structural damage identification models demonstrate notable accuracy in categorizing and localizing damage. However, these models present several critical limitations that hinder their practical application in civil engineering(CE). Primarily, their ability to recognize damage types remains constrained, preventing comprehensive analysis of the highly varied and complex conditions encountered in real-world CE structures. Second, these models lack linguistic capabilities, rendering them unable to articulate structural damage characteristics through natural language descriptions. With the continuous advancement of artificial intelligence(AI), large multi-modal models(LMMs) have emerged as a transformative solution, enabling the unified encoding and alignment of textual and visual data. These models can autonomously generate detailed descriptive narratives of structural damage while demonstrating robust generalization across diverse scenarios and tasks. This study introduces SDIGLM, an innovative LMM for structural damage identification, developed based on the open-source VisualGLM-6B architecture. To address the challenge of adapting LMMs to the intricate and varied operating conditions in CE, this work integrates a U-Net-based semantic segmentation module to generate defect segmentation maps as visual Chain of Thought(CoT). Additionally, a multi-round dialogue fine-tuning dataset is constructed to enhance logical reasoning, complemented by a language CoT formed through prompt engineering. By leveraging this multi-modal CoT, SDIGLM surpasses general-purpose LMMs in structural damage identification, achieving an accuracy of 95.24% across various infrastructure types. Moreover, the model effectively describes damage characteristics such as hole size, crack direction, and corrosion severity. 

---
# Visual moral inference and communication 

**Authors**: Warren Zhu, Aida Ramezani, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11473)  

**Abstract**: Humans can make moral inferences from multiple sources of input. In contrast, automated moral inference in artificial intelligence typically relies on language models with textual input. However, morality is conveyed through modalities beyond language. We present a computational framework that supports moral inference from natural images, demonstrated in two related tasks: 1) inferring human moral judgment toward visual images and 2) analyzing patterns in moral content communicated via images from public news. We find that models based on text alone cannot capture the fine-grained human moral judgment toward visual stimuli, but language-vision fusion models offer better precision in visual moral inference. Furthermore, applications of our framework to news data reveal implicit biases in news categories and geopolitical discussions. Our work creates avenues for automating visual moral inference and discovering patterns of visual moral communication in public media. 

---
