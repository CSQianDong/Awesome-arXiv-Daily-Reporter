# Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models 

**Authors**: Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, Minghui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21277)  

**Abstract**: The integration of reinforcement learning (RL) into the reasoning capabilities of Multimodal Large Language Models (MLLMs) has rapidly emerged as a transformative research direction. While MLLMs significantly extend Large Language Models (LLMs) to handle diverse modalities such as vision, audio, and video, enabling robust reasoning across multimodal inputs remains a major challenge. This survey systematically reviews recent advances in RL-based reasoning for MLLMs, covering key algorithmic designs, reward mechanism innovations, and practical applications. We highlight two main RL paradigms--value-free and value-based methods--and analyze how RL enhances reasoning abilities by optimizing reasoning trajectories and aligning multimodal information. Furthermore, we provide an extensive overview of benchmark datasets, evaluation protocols, and existing limitations, and propose future research directions to address current bottlenecks such as sparse rewards, inefficient cross-modal reasoning, and real-world deployment constraints. Our goal is to offer a comprehensive and structured guide to researchers interested in advancing RL-based reasoning in the multimodal era. 

---
# GarmentDiffusion: 3D Garment Sewing Pattern Generation with Multimodal Diffusion Transformers 

**Authors**: Xinyu Li, Qi Yao, Yuanda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21476)  

**Abstract**: Garment sewing patterns are fundamental design elements that bridge the gap between design concepts and practical manufacturing. The generative modeling of sewing patterns is crucial for creating diversified garments. However, existing approaches are limited either by reliance on a single input modality or by suboptimal generation efficiency. In this work, we present \textbf{\textit{GarmentDiffusion}}, a new generative model capable of producing centimeter-precise, vectorized 3D sewing patterns from multimodal inputs (text, image, and incomplete sewing pattern). Our method efficiently encodes 3D sewing pattern parameters into compact edge token representations, achieving a sequence length that is $\textbf{10}\times$ shorter than that of the autoregressive SewingGPT in DressCode. By employing a diffusion transformer, we simultaneously denoise all edge tokens along the temporal axis, while maintaining a constant number of denoising steps regardless of dataset-specific edge and panel statistics. With all combination of designs of our model, the sewing pattern generation speed is accelerated by $\textbf{100}\times$ compared to SewingGPT. We achieve new state-of-the-art results on DressCodeData, as well as on the largest sewing pattern dataset, namely GarmentCodeData. The project website is available at this https URL. 

---
# Rethinking Visual Layer Selection in Multimodal LLMs 

**Authors**: Haoran Chen, Junyan Lin, Xinhao Chen, Yue Fan, Xin Jin, Hui Su, Jianfeng Dong, Jinlan Fu, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21447)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive performance across a wide range of tasks, typically using CLIP-ViT as their visual encoder due to its strong text-image alignment capabilities. While prior studies suggest that different CLIP-ViT layers capture different types of information, with shallower layers focusing on fine visual details and deeper layers aligning more closely with textual semantics, most MLLMs still select visual features based on empirical heuristics rather than systematic analysis. In this work, we propose a Layer-wise Representation Similarity approach to group CLIP-ViT layers with similar behaviors into {shallow, middle, and deep} categories and assess their impact on MLLM performance. Building on this foundation, we revisit the visual layer selection problem in MLLMs at scale, training LLaVA-style models ranging from 1.4B to 7B parameters. Through extensive experiments across 10 datasets and 4 tasks, we find that: (1) deep layers are essential for OCR tasks; (2) shallow and middle layers substantially outperform deep layers on reasoning tasks involving counting, positioning, and object localization; (3) a lightweight fusion of features across shallow, middle, and deep layers consistently outperforms specialized fusion baselines and single-layer selections, achieving gains on 9 out of 10 datasets. Our work offers the first principled study of visual layer selection in MLLMs, laying the groundwork for deeper investigations into visual representation learning for MLLMs. 

---
# Nexus-Gen: A Unified Model for Image Understanding, Generation, and Editing 

**Authors**: Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, Yuze Zhao, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21356)  

**Abstract**: Unified multimodal large language models (MLLMs) aim to integrate multimodal understanding and generation abilities through a single framework. Despite their versatility, existing open-source unified models exhibit performance gaps against domain-specific architectures. To bridge this gap, we present Nexus-Gen, a unified model that synergizes the language reasoning capabilities of LLMs with the image synthesis power of diffusion models. To align the embedding space of the LLM and diffusion model, we conduct a dual-phase alignment training process. (1) The autoregressive LLM learns to predict image embeddings conditioned on multimodal inputs, while (2) the vision decoder is trained to reconstruct high-fidelity images from these embeddings. During training the LLM, we identified a critical discrepancy between the autoregressive paradigm's training and inference phases, where error accumulation in continuous embedding space severely degrades generation quality. To avoid this issue, we introduce a prefilled autoregression strategy that prefills input sequence with position-embedded special tokens instead of continuous embeddings. Through dual-phase training, Nexus-Gen has developed the integrated capability to comprehensively address the image understanding, generation and editing tasks. All models, datasets, and codes are published at this https URL to facilitate further advancements across the field. 

---
# DGFNet: End-to-End Audio-Visual Source Separation Based on Dynamic Gating Fusion 

**Authors**: Yinfeng Yu, Shiyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.21366)  

**Abstract**: Current Audio-Visual Source Separation methods primarily adopt two design strategies. The first strategy involves fusing audio and visual features at the bottleneck layer of the encoder, followed by processing the fused features through the decoder. However, when there is a significant disparity between the two modalities, this approach may lead to the loss of critical information. The second strategy avoids direct fusion and instead relies on the decoder to handle the interaction between audio and visual features. Nonetheless, if the encoder fails to integrate information across modalities adequately, the decoder may be unable to effectively capture the complex relationships between them. To address these issues, this paper proposes a dynamic fusion method based on a gating mechanism that dynamically adjusts the modality fusion degree. This approach mitigates the limitations of solely relying on the decoder and facilitates efficient collaboration between audio and visual features. Additionally, an audio attention module is introduced to enhance the expressive capacity of audio features, thereby further improving model performance. Experimental results demonstrate that our method achieves significant performance improvements on two benchmark datasets, validating its effectiveness and advantages in Audio-Visual Source Separation tasks. 

---
# Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Early Lung Cancer Detection 

**Authors**: Luoting Zhuang, Seyed Mohammad Hossein Tabatabaei, Ramin Salehi-Rad, Linh M. Tran, Denise R. Aberle, Ashley E. Prosper, William Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21344)  

**Abstract**: Objective: A number of machine learning models have utilized semantic features, deep features, or both to assess lung nodule malignancy. However, their reliance on manual annotation during inference, limited interpretability, and sensitivity to imaging variations hinder their application in real-world clinical settings. Thus, this research aims to integrate semantic features derived from radiologists' assessments of nodules, allowing the model to learn clinically relevant, robust, and explainable features for predicting lung cancer. Methods: We obtained 938 low-dose CT scans from the National Lung Screening Trial with 1,246 nodules and semantic features. The Lung Image Database Consortium dataset contains 1,018 CT scans, with 2,625 lesions annotated for nodule characteristics. Three external datasets were obtained from UCLA Health, the LUNGx Challenge, and the Duke Lung Cancer Screening. We finetuned a pretrained Contrastive Language-Image Pretraining model with a parameter-efficient fine-tuning approach to align imaging and semantic features and predict the one-year lung cancer diagnosis. Results: We evaluated the performance of the one-year diagnosis of lung cancer with AUROC and AUPRC and compared it to three state-of-the-art models. Our model demonstrated an AUROC of 0.90 and AUPRC of 0.78, outperforming baseline state-of-the-art models on external datasets. Using CLIP, we also obtained predictions on semantic features, such as nodule margin (AUROC: 0.81), nodule consistency (0.81), and pleural attachment (0.84), that can be used to explain model predictions. Conclusion: Our approach accurately classifies lung nodules as benign or malignant, providing explainable outputs, aiding clinicians in comprehending the underlying meaning of model predictions. This approach also prevents the model from learning shortcuts and generalizes across clinical settings. 

---
# MemeBLIP2: A novel lightweight multimodal system to detect harmful memes 

**Authors**: Jiaqi Liu, Ran Tong, Aowei Shen, Shuzheng Li, Changlin Yang, Lisha Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21226)  

**Abstract**: Memes often merge visuals with brief text to share humor or opinions, yet some memes contain harmful messages such as hate speech. In this paper, we introduces MemeBLIP2, a light weight multimodal system that detects harmful memes by combining image and text features effectively. We build on previous studies by adding modules that align image and text representations into a shared space and fuse them for better classification. Using BLIP-2 as the core vision-language model, our system is evaluated on the PrideMM datasets. The results show that MemeBLIP2 can capture subtle cues in both modalities, even in cases with ironic or culturally specific content, thereby improving the detection of harmful material. 

---
# AGATE: Stealthy Black-box Watermarking for Multimodal Model Copyright Protection 

**Authors**: Jianbo Gao, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21044)  

**Abstract**: Recent advancement in large-scale Artificial Intelligence (AI) models offering multimodal services have become foundational in AI systems, making them prime targets for model theft. Existing methods select Out-of-Distribution (OoD) data as backdoor watermarks and retrain the original model for copyright protection. However, existing methods are susceptible to malicious detection and forgery by adversaries, resulting in watermark evasion. In this work, we propose Model-\underline{ag}nostic Black-box Backdoor W\underline{ate}rmarking Framework (AGATE) to address stealthiness and robustness challenges in multimodal model copyright protection. Specifically, we propose an adversarial trigger generation method to generate stealthy adversarial triggers from ordinary dataset, providing visual fidelity while inducing semantic shifts. To alleviate the issue of anomaly detection among model outputs, we propose a post-transform module to correct the model output by narrowing the distance between adversarial trigger image embedding and text embedding. Subsequently, a two-phase watermark verification is proposed to judge whether the current model infringes by comparing the two results with and without the transform module. Consequently, we consistently outperform state-of-the-art methods across five datasets in the downstream tasks of multimodal image-text retrieval and image classification. Additionally, we validated the robustness of AGATE under two adversarial attack scenarios. 

---
# What's Pulling the Strings? Evaluating Integrity and Attribution in AI Training and Inference through Concept Shift 

**Authors**: Jiamin Chang, Haoyang Li, Hammond Pearce, Ruoxi Sun, Bo Li, Minhui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.21042)  

**Abstract**: The growing adoption of artificial intelligence (AI) has amplified concerns about trustworthiness, including integrity, privacy, robustness, and bias. To assess and attribute these threats, we propose ConceptLens, a generic framework that leverages pre-trained multimodal models to identify the root causes of integrity threats by analyzing Concept Shift in probing samples. ConceptLens demonstrates strong detection performance for vanilla data poisoning attacks and uncovers vulnerabilities to bias injection, such as the generation of covert advertisements through malicious concept shifts. It identifies privacy risks in unaltered but high-risk samples, filters them before training, and provides insights into model weaknesses arising from incomplete or imbalanced training data. Additionally, at the model level, it attributes concepts that the target model is overly dependent on, identifies misleading concepts, and explains how disrupting key concepts negatively impacts the model. Furthermore, it uncovers sociological biases in generative content, revealing disparities across sociological contexts. Strikingly, ConceptLens reveals how safe training and inference data can be unintentionally and easily exploited, potentially undermining safety alignment. Our study informs actionable insights to breed trust in AI systems, thereby speeding adoption and driving greater innovation. 

---
# Semantic-Aware Contrastive Fine-Tuning: Boosting Multimodal Malware Classification with Discriminative Embeddings 

**Authors**: Ivan Montoya Sanchez, Shaswata Mitra, Aritran Piplai, Sudip Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2504.21028)  

**Abstract**: The rapid evolution of malware variants requires robust classification methods to enhance cybersecurity. While Large Language Models (LLMs) offer potential for generating malware descriptions to aid family classification, their utility is limited by semantic embedding overlaps and misalignment with binary behavioral features. We propose a contrastive fine-tuning (CFT) method that refines LLM embeddings via targeted selection of hard negative samples based on cosine similarity, enabling LLMs to distinguish between closely related malware families. Our approach combines high-similarity negatives to enhance discriminative power and mid-tier negatives to increase embedding diversity, optimizing both precision and generalization. Evaluated on the CIC-AndMal-2020 and BODMAS datasets, our refined embeddings are integrated into a multimodal classifier within a Model-Agnostic Meta-Learning (MAML) framework on a few-shot setting. Experiments demonstrate significant improvements: our method achieves 63.15% classification accuracy with as few as 20 samples on CIC-AndMal-2020, outperforming baselines by 11--21 percentage points and surpassing prior negative sampling strategies. Ablation studies confirm the superiority of similarity-based selection over random sampling, with gains of 10-23%. Additionally, fine-tuned LLMs generate attribute-aware descriptions that generalize to unseen variants, bridging textual and binary feature gaps. This work advances malware classification by enabling nuanced semantic distinctions and provides a scalable framework for adapting LLMs to cybersecurity challenges. 

---
# SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding 

**Authors**: Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, ShaoGuo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21435)  

**Abstract**: With the rapid development of Multi-modal Large Language Models (MLLMs), an increasing number of benchmarks have been established to evaluate the video understanding capabilities of these models. However, these benchmarks focus on \textbf{standalone} videos and mainly assess ``visual elements'' like human actions and object states. In reality, contemporary videos often encompass complex and continuous narratives, typically presented as a \textbf{series}. To address this challenge, we propose \textbf{SeriesBench}, a benchmark consisting of 105 carefully curated narrative-driven series, covering 28 specialized tasks that require deep narrative understanding. Specifically, we first select a diverse set of drama series spanning various genres. Then, we introduce a novel long-span narrative annotation method, combined with a full-information transformation approach to convert manual annotations into diverse task formats. To further enhance model capacity for detailed analysis of plot structures and character relationships within series, we propose a novel narrative reasoning framework, \textbf{PC-DCoT}. Extensive results on \textbf{SeriesBench} indicate that existing MLLMs still face significant challenges in understanding narrative-driven series, while \textbf{PC-DCoT} enables these MLLMs to achieve performance improvements. Overall, our \textbf{SeriesBench} and \textbf{PC-DCoT} highlight the critical necessity of advancing model capabilities to understand narrative-driven series, guiding the future development of MLLMs. SeriesBench is publicly available at this https URL. 

---
# Multimodal Large Language Models for Medicine: A Comprehensive Survey 

**Authors**: Jiarui Ye, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21051)  

**Abstract**: MLLMs have recently become a focal point in the field of artificial intelligence research. Building on the strong capabilities of LLMs, MLLMs are adept at addressing complex multi-modal tasks. With the release of GPT-4, MLLMs have gained substantial attention from different domains. Researchers have begun to explore the potential of MLLMs in the medical and healthcare domain. In this paper, we first introduce the background and fundamental concepts related to LLMs and MLLMs, while emphasizing the working principles of MLLMs. Subsequently, we summarize three main directions of application within healthcare: medical reporting, medical diagnosis, and medical treatment. Our findings are based on a comprehensive review of 330 recent papers in this area. We illustrate the remarkable capabilities of MLLMs in these domains by providing specific examples. For data, we present six mainstream modes of data along with their corresponding evaluation benchmarks. At the end of the survey, we discuss the challenges faced by MLLMs in the medical and healthcare domain and propose feasible methods to mitigate or overcome these issues. 

---
