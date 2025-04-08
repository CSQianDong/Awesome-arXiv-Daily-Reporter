# Could AI Trace and Explain the Origins of AI-Generated Images and Text? 

**Authors**: Hongchao Fang, Can Qin, Ran Xu, Feng Liu, Yixin Liu, Lichao Sun, Dongwon Lee, Lifu Huang, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04279)  

**Abstract**: AI-generated content is becoming increasingly prevalent in the real world, leading to serious ethical and societal concerns. For instance, adversaries might exploit large multimodal models (LMMs) to create images that violate ethical or legal standards, while paper reviewers may misuse large language models (LLMs) to generate reviews without genuine intellectual effort. While prior work has explored detecting AI-generated images and texts, and occasionally tracing their source models, there is a lack of a systematic and fine-grained comparative study. Important dimensions--such as AI-generated images vs. text, fully vs. partially AI-generated images, and general vs. malicious use cases--remain underexplored. Furthermore, whether AI systems like GPT-4o can explain why certain forged content is attributed to specific generative models is still an open question, with no existing benchmark addressing this. To fill this gap, we introduce AI-FAKER, a comprehensive multimodal dataset with over 280,000 samples spanning multiple LLMs and LMMs, covering both general and malicious use cases for AI-generated images and texts. Our experiments reveal two key findings: (i) AI authorship detection depends not only on the generated output but also on the model's original training intent; and (ii) GPT-4o provides highly consistent but less specific explanations when analyzing content produced by OpenAI's own models, such as DALL-E and GPT-4o itself. 

---
# Towards Visual Text Grounding of Multimodal Large Language Model 

**Authors**: Ming Li, Ruiyi Zhang, Jian Chen, Jiuxiang Gu, Yufan Zhou, Franck Dernoncourt, Wanrong Zhu, Tianyi Zhou, Tong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04974)  

**Abstract**: Despite the existing evolution of Multimodal Large Language Models (MLLMs), a non-neglectable limitation remains in their struggle with visual text grounding, especially in text-rich images of documents. Document images, such as scanned forms and infographics, highlight critical challenges due to their complex layouts and textual content. However, current benchmarks do not fully address these challenges, as they mostly focus on visual grounding on natural images, rather than text-rich document images. Thus, to bridge this gap, we introduce TRIG, a novel task with a newly designed instruction dataset for benchmarking and improving the Text-Rich Image Grounding capabilities of MLLMs in document question-answering. Specifically, we propose an OCR-LLM-human interaction pipeline to create 800 manually annotated question-answer pairs as a benchmark and a large-scale training set of 90$ synthetic data based on four diverse datasets. A comprehensive evaluation of various MLLMs on our proposed benchmark exposes substantial limitations in their grounding capability on text-rich images. In addition, we propose two simple and effective TRIG methods based on general instruction tuning and plug-and-play efficient embedding, respectively. By finetuning MLLMs on our synthetic dataset, they promisingly improve spatial reasoning and grounding capabilities. 

---
# CliME: Evaluating Multimodal Climate Discourse on Social Media and the Climate Alignment Quotient (CAQ) 

**Authors**: Abhilekh Borah, Hasnat Md Abdullah, Kangda Wei, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03906)  

**Abstract**: The rise of Large Language Models (LLMs) has raised questions about their ability to understand climate-related contexts. Though climate change dominates social media, analyzing its multimodal expressions is understudied, and current tools have failed to determine whether LLMs amplify credible solutions or spread unsubstantiated claims. To address this, we introduce CliME (Climate Change Multimodal Evaluation), a first-of-its-kind multimodal dataset, comprising 2579 Twitter and Reddit posts. The benchmark features a diverse collection of humorous memes and skeptical posts, capturing how these formats distill complex issues into viral narratives that shape public opinion and policy discussions. To systematically evaluate LLM performance, we present the Climate Alignment Quotient (CAQ), a novel metric comprising five distinct dimensions: Articulation, Evidence, Resonance, Transition, and Specificity. Additionally, we propose three analytical lenses: Actionability, Criticality, and Justice, to guide the assessment of LLM-generated climate discourse using CAQ. Our findings, based on the CAQ metric, indicate that while most evaluated LLMs perform relatively well in Criticality and Justice, they consistently underperform on the Actionability axis. Among the models evaluated, Claude 3.7 Sonnet achieves the highest overall performance. We publicly release our CliME dataset and code to foster further research in this domain. 

---
# LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts 

**Authors**: Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04653)  

**Abstract**: Redundancy of visual tokens in multi-modal large language models (MLLMs) significantly reduces their computational efficiency. Recent approaches, such as resamplers and summarizers, have sought to reduce the number of visual tokens, but at the cost of visual reasoning ability. To address this, we propose LEO-MINI, a novel MLLM that significantly reduces the number of visual tokens and simultaneously boosts visual reasoning capabilities. For efficiency, LEO-MINI incorporates CoTR, a novel token reduction module to consolidate a large number of visual tokens into a smaller set of tokens, using the similarity between visual tokens, text tokens, and a compact learnable query. For effectiveness, to scale up the model's ability with minimal computational overhead, LEO-MINI employs MMoE, a novel mixture of multi-modal experts module. MMOE employs a set of LoRA experts with a novel router to switch between them based on the input text and visual tokens instead of only using the input hidden state. MMoE also includes a general LoRA expert that is always activated to learn general knowledge for LLM reasoning. For extracting richer visual features, MMOE employs a set of vision experts trained on diverse domain-specific data. To demonstrate LEO-MINI's improved efficiency and performance, we evaluate it against existing efficient MLLMs on various benchmark vision-language tasks. 

---
# VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models 

**Authors**: Dahun Kim, AJ Piergiovanni, Ganesh Mallya, Anelia Angelova  

**Link**: [PDF](https://arxiv.org/pdf/2504.03970)  

**Abstract**: We introduce VideoComp, a benchmark and learning framework for advancing video-text compositionality understanding, aimed at improving vision-language models (VLMs) in fine-grained temporal alignment. Unlike existing benchmarks focused on static image-text compositionality or isolated single-event videos, our benchmark targets alignment in continuous multi-event videos. Leveraging video-text datasets with temporally localized event captions (e.g. ActivityNet-Captions, YouCook2), we construct two compositional benchmarks, ActivityNet-Comp and YouCook2-Comp. We create challenging negative samples with subtle temporal disruptions such as reordering, action word replacement, partial captioning, and combined disruptions. These benchmarks comprehensively test models' compositional sensitivity across extended, cohesive video-text sequences. To improve model performance, we propose a hierarchical pairwise preference loss that strengthens alignment with temporally accurate pairs and gradually penalizes increasingly disrupted ones, encouraging fine-grained compositional learning. To mitigate the limited availability of densely annotated video data, we introduce a pretraining strategy that concatenates short video-caption pairs to simulate multi-event sequences. We evaluate video-text foundational models and large multimodal models (LMMs) on our benchmark, identifying both strengths and areas for improvement in compositionality. Overall, our work provides a comprehensive framework for evaluating and enhancing model capabilities in achieving fine-grained, temporally coherent video-text alignment. 

---
# Misaligned Roles, Misplaced Images: Structural Input Perturbations Expose Multimodal Alignment Blind Spots 

**Authors**: Erfan Shayegani, G M Shahariar, Sara Abdali, Lei Yu, Nael Abu-Ghazaleh, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03735)  

**Abstract**: Multimodal Language Models (MMLMs) typically undergo post-training alignment to prevent harmful content generation. However, these alignment stages focus primarily on the assistant role, leaving the user role unaligned, and stick to a fixed input prompt structure of special tokens, leaving the model vulnerable when inputs deviate from these expectations. We introduce Role-Modality Attacks (RMA), a novel class of adversarial attacks that exploit role confusion between the user and assistant and alter the position of the image token to elicit harmful outputs. Unlike existing attacks that modify query content, RMAs manipulate the input structure without altering the query itself. We systematically evaluate these attacks across multiple Vision Language Models (VLMs) on eight distinct settings, showing that they can be composed to create stronger adversarial prompts, as also evidenced by their increased projection in the negative refusal direction in the residual stream, a property observed in prior successful attacks. Finally, for mitigation, we propose an adversarial training approach that makes the model robust against input prompt perturbations. By training the model on a range of harmful and benign prompts all perturbed with different RMA settings, it loses its sensitivity to Role Confusion and Modality Manipulation attacks and is trained to only pay attention to the content of the query in the input prompt structure, effectively reducing Attack Success Rate (ASR) while preserving the model's general utility. 

---
# TDBench: Benchmarking Vision-Language Models in Understanding Top-Down Images 

**Authors**: Kaiyuan Hou, Minghui Zhao, Lilin Xu, Yuang Fan, Xiaofan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03748)  

**Abstract**: The rapid emergence of Vision-Language Models (VLMs) has significantly advanced multimodal understanding, enabling applications in scene comprehension and visual reasoning. While these models have been primarily evaluated and developed for front-view image understanding, their capabilities in interpreting top-down images have received limited attention, partly due to the scarcity of diverse top-down datasets and the challenges in collecting such data. In contrast, top-down vision provides explicit spatial overviews and improved contextual understanding of scenes, making it particularly valuable for tasks like autonomous navigation, aerial imaging, and spatial planning. In this work, we address this gap by introducing TDBench, a comprehensive benchmark for VLMs in top-down image understanding. TDBench is constructed from public top-down view datasets and high-quality simulated images, including diverse real-world and synthetic scenarios. TDBench consists of visual question-answer pairs across ten evaluation dimensions of image understanding. Moreover, we conduct four case studies that commonly happen in real-world scenarios but are less explored. By revealing the strengths and limitations of existing VLM through evaluation results, we hope TDBench to provide insights for motivating future research. Project homepage: this https URL 

---
# SmolVLM: Redefining small and efficient multimodal models 

**Authors**: Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2504.05299)  

**Abstract**: Large Vision-Language Models (VLMs) deliver exceptional performance but require significant computational resources, limiting their deployment on mobile and edge devices. Smaller VLMs typically mirror design choices of larger models, such as extensive image tokenization, leading to inefficient GPU memory usage and constrained practicality for on-device applications.
We introduce SmolVLM, a series of compact multimodal models specifically engineered for resource-efficient inference. We systematically explore architectural configurations, tokenization strategies, and data curation optimized for low computational overhead. Through this, we identify key design choices that yield substantial performance gains on image and video tasks with minimal memory footprints.
Our smallest model, SmolVLM-256M, uses less than 1GB GPU memory during inference and outperforms the 300-times larger Idefics-80B model, despite an 18-month development gap. Our largest model, at 2.2B parameters, rivals state-of-the-art VLMs consuming twice the GPU memory. SmolVLM models extend beyond static images, demonstrating robust video comprehension capabilities.
Our results emphasize that strategic architectural optimizations, aggressive yet efficient tokenization, and carefully curated training data significantly enhance multimodal performance, facilitating practical, energy-efficient deployments at significantly smaller scales. 

---
# Mapping biodiversity at very-high resolution in Europe 

**Authors**: César Leblanc, Lukas Picek, Benjamin Deneu, Pierre Bonnet, Maximilien Servajean, Rémi Palard, Alexis Joly  

**Link**: [PDF](https://arxiv.org/pdf/2504.05231)  

**Abstract**: This paper describes a cascading multimodal pipeline for high-resolution biodiversity mapping across Europe, integrating species distribution modeling, biodiversity indicators, and habitat classification. The proposed pipeline first predicts species compositions using a deep-SDM, a multimodal model trained on remote sensing, climate time series, and species occurrence data at 50x50m resolution. These predictions are then used to generate biodiversity indicator maps and classify habitats with Pl@ntBERT, a transformer-based LLM designed for species-to-habitat mapping. With this approach, continental-scale species distribution maps, biodiversity indicator maps, and habitat maps are produced, providing fine-grained ecological insights. Unlike traditional methods, this framework enables joint modeling of interspecies dependencies, bias-aware training with heterogeneous presence-absence data, and large-scale inference from multi-source remote sensing inputs. 

---
# Don't Lag, RAG: Training-Free Adversarial Detection Using RAG 

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04858)  

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks. 

---
# Multimodal Agricultural Agent Architecture (MA3): A New Paradigm for Intelligent Agricultural Decision-Making 

**Authors**: Zhuoning Xu, Jian Xu, Mingqing Zhang, Peijie Wang, Chao Deng, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04789)  

**Abstract**: As a strategic pillar industry for human survival and development, modern agriculture faces dual challenges: optimizing production efficiency and achieving sustainable development. Against the backdrop of intensified climate change leading to frequent extreme weather events, the uncertainty risks in agricultural production systems are increasing exponentially. To address these challenges, this study proposes an innovative \textbf{M}ultimodal \textbf{A}gricultural \textbf{A}gent \textbf{A}rchitecture (\textbf{MA3}), which leverages cross-modal information fusion and task collaboration mechanisms to achieve intelligent agricultural decision-making. This study constructs a multimodal agricultural agent dataset encompassing five major tasks: classification, detection, Visual Question Answering (VQA), tool selection, and agent evaluation. We propose a unified backbone for sugarcane disease classification and detection tools, as well as a sugarcane disease expert model. By integrating an innovative tool selection module, we develop a multimodal agricultural agent capable of effectively performing tasks in classification, detection, and VQA. Furthermore, we introduce a multi-dimensional quantitative evaluation framework and conduct a comprehensive assessment of the entire architecture over our evaluation dataset, thereby verifying the practicality and robustness of MA3 in agricultural scenarios. This study provides new insights and methodologies for the development of agricultural agents, holding significant theoretical and practical implications. Our source code and dataset will be made publicly available upon acceptance. 

---
# URECA: Unique Region Caption Anything 

**Authors**: Sangbeom Lim, Junwan Kim, Heeji Yoon, Jaewoo Jung, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05305)  

**Abstract**: Region-level captioning aims to generate natural language descriptions for specific image regions while highlighting their distinguishing features. However, existing methods struggle to produce unique captions across multi-granularity, limiting their real-world applicability. To address the need for detailed region-level understanding, we introduce URECA dataset, a large-scale dataset tailored for multi-granularity region captioning. Unlike prior datasets that focus primarily on salient objects, URECA dataset ensures a unique and consistent mapping between regions and captions by incorporating a diverse set of objects, parts, and background elements. Central to this is a stage-wise data curation pipeline, where each stage incrementally refines region selection and caption generation. By leveraging Multimodal Large Language Models (MLLMs) at each stage, our pipeline produces distinctive and contextually grounded captions with improved accuracy and semantic diversity. Building upon this dataset, we present URECA, a novel captioning model designed to effectively encode multi-granularity regions. URECA maintains essential spatial properties such as position and shape through simple yet impactful modifications to existing MLLMs, enabling fine-grained and semantically rich region descriptions. Our approach introduces dynamic mask modeling and a high-resolution mask encoder to enhance caption uniqueness. Experiments show that URECA achieves state-of-the-art performance on URECA dataset and generalizes well to existing region-level captioning benchmarks. 

---
# Explaining Low Perception Model Competency with High-Competency Counterfactuals 

**Authors**: Sara Pohland, Claire Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05254)  

**Abstract**: There exist many methods to explain how an image classification model generates its decision, but very little work has explored methods to explain why a classifier might lack confidence in its prediction. As there are various reasons the classifier might lose confidence, it would be valuable for this model to not only indicate its level of uncertainty but also explain why it is uncertain. Counterfactual images have been used to visualize changes that could be made to an image to generate a different classification decision. In this work, we explore the use of counterfactuals to offer an explanation for low model competency--a generalized form of predictive uncertainty that measures confidence. Toward this end, we develop five novel methods to generate high-competency counterfactual images, namely Image Gradient Descent (IGD), Feature Gradient Descent (FGD), Autoencoder Reconstruction (Reco), Latent Gradient Descent (LGD), and Latent Nearest Neighbors (LNN). We evaluate these methods across two unique datasets containing images with six known causes for low model competency and find Reco, LGD, and LNN to be the most promising methods for counterfactual generation. We further evaluate how these three methods can be utilized by pre-trained Multimodal Large Language Models (MLLMs) to generate language explanations for low model competency. We find that the inclusion of a counterfactual image in the language model query greatly increases the ability of the model to generate an accurate explanation for the cause of low model competency, thus demonstrating the utility of counterfactual images in explaining low perception model competency. 

---
# Resource-Efficient Beam Prediction in mmWave Communications with Multimodal Realistic Simulation Framework 

**Authors**: Yu Min Park, Yan Kyaw Tun, Walid Saad, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05187)  

**Abstract**: Beamforming is a key technology in millimeter-wave (mmWave) communications that improves signal transmission by optimizing directionality and intensity. However, conventional channel estimation methods, such as pilot signals or beam sweeping, often fail to adapt to rapidly changing communication environments. To address this limitation, multimodal sensing-aided beam prediction has gained significant attention, using various sensing data from devices such as LiDAR, radar, GPS, and RGB images to predict user locations or network conditions. Despite its promising potential, the adoption of multimodal sensing-aided beam prediction is hindered by high computational complexity, high costs, and limited datasets. Thus, in this paper, a resource-efficient learning approach is proposed to transfer knowledge from a multimodal network to a monomodal (radar-only) network based on cross-modal relational knowledge distillation (CRKD), while reducing computational overhead and preserving predictive accuracy. To enable multimodal learning with realistic data, a novel multimodal simulation framework is developed while integrating sensor data generated from the autonomous driving simulator CARLA with MATLAB-based mmWave channel modeling, and reflecting real-world conditions. The proposed CRKD achieves its objective by distilling relational information across different feature spaces, which enhances beam prediction performance without relying on expensive sensor data. Simulation results demonstrate that CRKD efficiently distills multimodal knowledge, allowing a radar-only model to achieve $94.62\%$ of the teacher performance. In particular, this is achieved with just $10\%$ of the teacher network's parameters, thereby significantly reducing computational complexity and dependence on multimodal sensor data. 

---
# SSLFusion: Scale & Space Aligned Latent Fusion Model for Multimodal 3D Object Detection 

**Authors**: Bonan Ding, Jin Xie, Jing Nie, Jiale Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05170)  

**Abstract**: Multimodal 3D object detection based on deep neural networks has indeed made significant progress. However, it still faces challenges due to the misalignment of scale and spatial information between features extracted from 2D images and those derived from 3D point clouds. Existing methods usually aggregate multimodal features at a single stage. However, leveraging multi-stage cross-modal features is crucial for detecting objects of various scales. Therefore, these methods often struggle to integrate features across different scales and modalities effectively, thereby restricting the accuracy of detection. Additionally, the time-consuming Query-Key-Value-based (QKV-based) cross-attention operations often utilized in existing methods aid in reasoning the location and existence of objects by capturing non-local contexts. However, this approach tends to increase computational complexity. To address these challenges, we present SSLFusion, a novel Scale & Space Aligned Latent Fusion Model, consisting of a scale-aligned fusion strategy (SAF), a 3D-to-2D space alignment module (SAM), and a latent cross-modal fusion module (LFM). SAF mitigates scale misalignment between modalities by aggregating features from both images and point clouds across multiple levels. SAM is designed to reduce the inter-modal gap between features from images and point clouds by incorporating 3D coordinate information into 2D image features. Additionally, LFM captures cross-modal non-local contexts in the latent space without utilizing the QKV-based attention operations, thus mitigating computational complexity. Experiments on the KITTI and DENSE datasets demonstrate that our SSLFusion outperforms state-of-the-art methods. Our approach obtains an absolute gain of 2.15% in 3D AP, compared with the state-of-art method GraphAlign on the moderate level of the KITTI test set. 

---
# Leveraging Label Potential for Enhanced Multimodal Emotion Recognition 

**Authors**: Xuechun Shao, Yinfeng Yu, Liejun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05158)  

**Abstract**: Multimodal emotion recognition (MER) seeks to integrate various modalities to predict emotional states accurately. However, most current research focuses solely on the fusion of audio and text features, overlooking the valuable information in emotion labels. This oversight could potentially hinder the performance of existing methods, as emotion labels harbor rich, insightful information that could significantly aid MER. We introduce a novel model called Label Signal-Guided Multimodal Emotion Recognition (LSGMER) to overcome this limitation. This model aims to fully harness the power of emotion label information to boost the classification accuracy and stability of MER. Specifically, LSGMER employs a Label Signal Enhancement module that optimizes the representation of modality features by interacting with audio and text features through label embeddings, enabling it to capture the nuances of emotions precisely. Furthermore, we propose a Joint Objective Optimization(JOO) approach to enhance classification accuracy by introducing the Attribution-Prediction Consistency Constraint (APC), which strengthens the alignment between fused features and emotion categories. Extensive experiments conducted on the IEMOCAP and MELD datasets have demonstrated the effectiveness of our proposed LSGMER model. 

---
# RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model 

**Authors**: Congcong Wen, Yiting Lin, Xiaokang Qu, Nan Li, Yong Liao, Hui Lin, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04988)  

**Abstract**: Recent progress in VLMs has demonstrated impressive capabilities across a variety of tasks in the natural image domain. Motivated by these advancements, the remote sensing community has begun to adopt VLMs for remote sensing vision-language tasks, including scene understanding, image captioning, and visual question answering. However, existing remote sensing VLMs typically rely on closed-set scene understanding and focus on generic scene descriptions, yet lack the ability to incorporate external knowledge. This limitation hinders their capacity for semantic reasoning over complex or context-dependent queries that involve domain-specific or world knowledge. To address these challenges, we first introduced a multimodal Remote Sensing World Knowledge (RSWK) dataset, which comprises high-resolution satellite imagery and detailed textual descriptions for 14,141 well-known landmarks from 175 countries, integrating both remote sensing domain knowledge and broader world knowledge. Building upon this dataset, we proposed a novel Remote Sensing Retrieval-Augmented Generation (RS-RAG) framework, which consists of two key components. The Multi-Modal Knowledge Vector Database Construction module encodes remote sensing imagery and associated textual knowledge into a unified vector space. The Knowledge Retrieval and Response Generation module retrieves and re-ranks relevant knowledge based on image and/or text queries, and incorporates the retrieved content into a knowledge-augmented prompt to guide the VLM in producing contextually grounded responses. We validated the effectiveness of our approach on three representative vision-language tasks, including image captioning, image classification, and visual question answering, where RS-RAG significantly outperformed state-of-the-art baselines. 

---
# SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models 

**Authors**: Justus Westerhoff, Erblina Purellku, Jakob Hackstein, Leo Pinetzki, Lorenz Hufe  

**Link**: [PDF](https://arxiv.org/pdf/2504.04893)  

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper under this https URL, along with the code for evaluations at this https URL. 

---
# Lumina-OmniLV: A Unified Multimodal Framework for General Low-Level Vision 

**Authors**: Yuandong Pu, Le Zhuo, Kaiwen Zhu, Liangbin Xie, Wenlong Zhang, Xiangyu Chen, Pneg Gao, Yu Qiao, Chao Dong, Yihao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04903)  

**Abstract**: We present Lunima-OmniLV (abbreviated as OmniLV), a universal multimodal multi-task framework for low-level vision that addresses over 100 sub-tasks across four major categories: image restoration, image enhancement, weak-semantic dense prediction, and stylization. OmniLV leverages both textual and visual prompts to offer flexible and user-friendly interactions. Built on Diffusion Transformer (DiT)-based generative priors, our framework supports arbitrary resolutions -- achieving optimal performance at 1K resolution -- while preserving fine-grained details and high fidelity. Through extensive experiments, we demonstrate that separately encoding text and visual instructions, combined with co-training using shallow feature control, is essential to mitigate task ambiguity and enhance multi-task generalization. Our findings also reveal that integrating high-level generative tasks into low-level vision models can compromise detail-sensitive restoration. These insights pave the way for more robust and generalizable low-level vision systems. 

---
# Video-Bench: Human-Aligned Video Generation Benchmark 

**Authors**: Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, Chi Zhang, Li-jia Li, Yongxin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2504.04907)  

**Abstract**: Video generation assessment is essential for ensuring that generative models produce visually realistic, high-quality videos while aligning with human expectations. Current video generation benchmarks fall into two main categories: traditional benchmarks, which use metrics and embeddings to evaluate generated video quality across multiple dimensions but often lack alignment with human judgments; and large language model (LLM)-based benchmarks, though capable of human-like reasoning, are constrained by a limited understanding of video quality metrics and cross-modal consistency. To address these challenges and establish a benchmark that better aligns with human preferences, this paper introduces Video-Bench, a comprehensive benchmark featuring a rich prompt suite and extensive evaluation dimensions. This benchmark represents the first attempt to systematically leverage MLLMs across all dimensions relevant to video generation assessment in generative models. By incorporating few-shot scoring and chain-of-query techniques, Video-Bench provides a structured, scalable approach to generated video evaluation. Experiments on advanced models including Sora demonstrate that Video-Bench achieves superior alignment with human preferences across all dimensions. Moreover, in instances where our framework's assessments diverge from human evaluations, it consistently offers more objective and accurate insights, suggesting an even greater potential advantage over traditional human judgment. 

---
# Bidirectional Hierarchical Protein Multi-Modal Representation Learning 

**Authors**: Xuefeng Liu, Songhao Jiang, Chih-chan Tien, Jinbo Xu, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2504.04770)  

**Abstract**: Protein representation learning is critical for numerous biological tasks. Recently, large transformer-based protein language models (pLMs) pretrained on large scale protein sequences have demonstrated significant success in sequence-based tasks. However, pLMs lack structural information. Conversely, graph neural networks (GNNs) designed to leverage 3D structural information have shown promising generalization in protein-related prediction tasks, but their effectiveness is often constrained by the scarcity of labeled structural data. Recognizing that sequence and structural representations are complementary perspectives of the same protein entity, we propose a multimodal bidirectional hierarchical fusion framework to effectively merge these modalities. Our framework employs attention and gating mechanisms to enable effective interaction between pLMs-generated sequential representations and GNN-extracted structural features, improving information exchange and enhancement across layers of the neural network. Based on the framework, we further introduce local Bi-Hierarchical Fusion with gating and global Bi-Hierarchical Fusion with multihead self-attention approaches. Through extensive experiments on a diverse set of protein-related tasks, our method demonstrates consistent improvements over strong baselines and existing fusion techniques in a variety of protein representation learning benchmarks, including react (enzyme/EC classification), model quality assessment (MQA), protein-ligand binding affinity prediction (LBA), protein-protein binding site prediction (PPBS), and B cell epitopes prediction (BCEs). Our method establishes a new state-of-the-art for multimodal protein representation learning, emphasizing the efficacy of BIHIERARCHICAL FUSION in bridging sequence and structural modalities. 

---
# Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions 

**Authors**: He Zhu, Quyu Kong, Kechun Xu, Xunlong Xia, Bing Deng, Jieping Ye, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04744)  

**Abstract**: Grounding 3D object affordance is a task that locates objects in 3D space where they can be manipulated, which links perception and action for embodied intelligence. For example, for an intelligent robot, it is necessary to accurately ground the affordance of an object and grasp it according to human instructions. In this paper, we introduce a novel task that grounds 3D object affordance based on language instructions, visual observations and interactions, which is inspired by cognitive science. We collect an Affordance Grounding dataset with Points, Images and Language instructions (AGPIL) to support the proposed task. In the 3D physical world, due to observation orientation, object rotation, or spatial occlusion, we can only get a partial observation of the object. So this dataset includes affordance estimations of objects from full-view, partial-view, and rotation-view perspectives. To accomplish this task, we propose LMAffordance3D, the first multi-modal, language-guided 3D affordance grounding network, which applies a vision-language model to fuse 2D and 3D spatial features with semantic features. Comprehensive experiments on AGPIL demonstrate the effectiveness and superiority of our method on this task, even in unseen experimental settings. Our project is available at this https URL. 

---
# DanceMosaic: High-Fidelity Dance Generation with Multimodal Editability 

**Authors**: Foram Niravbhai Shah, Parshwa Shah, Muhammad Usama Saleem, Ekkasit Pinyoanuntapong, Pu Wang, Hongfei Xue, Ahmed Helmy  

**Link**: [PDF](https://arxiv.org/pdf/2504.04634)  

**Abstract**: Recent advances in dance generation have enabled automatic synthesis of 3D dance motions. However, existing methods still struggle to produce high-fidelity dance sequences that simultaneously deliver exceptional realism, precise dance-music synchronization, high motion diversity, and physical plausibility. Moreover, existing methods lack the flexibility to edit dance sequences according to diverse guidance signals, such as musical prompts, pose constraints, action labels, and genre descriptions, significantly restricting their creative utility and adaptability. Unlike the existing approaches, DanceMosaic enables fast and high-fidelity dance generation, while allowing multimodal motion editing. Specifically, we propose a multimodal masked motion model that fuses the text-to-motion model with music and pose adapters to learn probabilistic mapping from diverse guidance signals to high-quality dance motion sequences via progressive generative masking training. To further enhance the motion generation quality, we propose multimodal classifier-free guidance and inference-time optimization mechanism that further enforce the alignment between the generated motions and the multimodal guidance. Extensive experiments demonstrate that our method establishes a new state-of-the-art performance in dance generation, significantly advancing the quality and editability achieved by existing approaches. 

---
# M2IV: Towards Efficient and Fine-grained Multimodal In-Context Learning in Large Vision-Language Models 

**Authors**: Yanshu Li, Hongyang He, Yi Cao, Qisen Cheng, Xiang Fu, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04633)  

**Abstract**: Multimodal in-context learning (ICL) is a vital capability for Large Vision-Language Models (LVLMs), allowing task adaptation via contextual prompts without parameter retraining. However, its application is hindered by the token-intensive nature of inputs and the high complexity of cross-modal few-shot learning, which limits the expressive power of representation methods. To tackle these challenges, we propose \textbf{M2IV}, a method that substitutes explicit demonstrations with learnable \textbf{I}n-context \textbf{V}ectors directly integrated into LVLMs. By exploiting the complementary strengths of multi-head attention (\textbf{M}HA) and multi-layer perceptrons (\textbf{M}LP), M2IV achieves robust cross-modal fidelity and fine-grained semantic distillation through training. This significantly enhances performance across diverse LVLMs and tasks and scales efficiently to many-shot scenarios, bypassing the context window limitations. We also introduce \textbf{VLibrary}, a repository for storing and retrieving M2IV, enabling flexible LVLM steering for tasks like cross-modal alignment, customized generation and safety improvement. Experiments across seven benchmarks and three LVLMs show that M2IV surpasses Vanilla ICL and prior representation engineering approaches, with an average accuracy gain of \textbf{3.74\%} over ICL with the same shot count, alongside substantial efficiency advantages. 

---
# Advancing Egocentric Video Question Answering with Multimodal Large Language Models 

**Authors**: Alkesh Patel, Vibhav Chitalia, Yinfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04550)  

**Abstract**: Egocentric Video Question Answering (QA) requires models to handle long-horizon temporal reasoning, first-person perspectives, and specialized challenges like frequent camera movement. This paper systematically evaluates both proprietary and open-source Multimodal Large Language Models (MLLMs) on QaEgo4Dv2 - a refined dataset of egocentric videos derived from QaEgo4D. Four popular MLLMs (GPT-4o, Gemini-1.5-Pro, Video-LLaVa-7B and Qwen2-VL-7B-Instruct) are assessed using zero-shot and fine-tuned approaches for both OpenQA and CloseQA settings. We introduce QaEgo4Dv2 to mitigate annotation noise in QaEgo4D, enabling more reliable comparison. Our results show that fine-tuned Video-LLaVa-7B and Qwen2-VL-7B-Instruct achieve new state-of-the-art performance, surpassing previous benchmarks by up to +2.6% ROUGE/METEOR (for OpenQA) and +13% accuracy (for CloseQA). We also present a thorough error analysis, indicating the model's difficulty in spatial reasoning and fine-grained object recognition - key areas for future improvement. 

---
# UniToken: Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding 

**Authors**: Yang Jiao, Haibo Qiu, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04423)  

**Abstract**: We introduce UniToken, an auto-regressive generation model that encodes visual inputs through a combination of discrete and continuous representations, enabling seamless integration of unified visual understanding and image generation tasks. Unlike previous approaches that rely on unilateral visual representations, our unified visual encoding framework captures both high-level semantics and low-level details, delivering multidimensional information that empowers heterogeneous tasks to selectively assimilate domain-specific knowledge based on their inherent characteristics. Through in-depth experiments, we uncover key principles for developing a unified model capable of both visual understanding and image generation. Extensive evaluations across a diverse range of prominent benchmarks demonstrate that UniToken achieves state-of-the-art performance, surpassing existing approaches. These results establish UniToken as a robust foundation for future research in this domain. The code and models are available at this https URL. 

---
# Universal Item Tokenization for Transferable Generative Recommendation 

**Authors**: Bowen Zheng, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04405)  

**Abstract**: Recently, generative recommendation has emerged as a promising paradigm, attracting significant research attention. The basic framework involves an item tokenizer, which represents each item as a sequence of codes serving as its identifier, and a generative recommender that predicts the next item by autoregressively generating the target item identifier. However, in existing methods, both the tokenizer and the recommender are typically domain-specific, limiting their ability for effective transfer or adaptation to new domains. To this end, we propose UTGRec, a Universal item Tokenization approach for transferable Generative Recommendation. Specifically, we design a universal item tokenizer for encoding rich item semantics by adapting a multimodal large language model (MLLM). By devising tree-structured codebooks, we discretize content representations into corresponding codes for item tokenization. To effectively learn the universal item tokenizer on multiple domains, we introduce two key techniques in our approach. For raw content reconstruction, we employ dual lightweight decoders to reconstruct item text and images from discrete representations to capture general knowledge embedded in the content. For collaborative knowledge integration, we assume that co-occurring items are similar and integrate collaborative signals through co-occurrence alignment and reconstruction. Finally, we present a joint learning framework to pre-train and adapt the transferable generative recommender across multiple domains. Extensive experiments on four public datasets demonstrate the superiority of UTGRec compared to both traditional and generative recommendation baselines. 

---
# A Survey of Pathology Foundation Model: Progress and Future Directions 

**Authors**: Conghao Xiong, Hao Chen, Joseph J. Y. Sung  

**Link**: [PDF](https://arxiv.org/pdf/2504.04045)  

**Abstract**: Computational pathology, analyzing whole slide images for automated cancer diagnosis, relies on the multiple instance learning framework where performance heavily depends on the feature extractor and aggregator. Recent Pathology Foundation Models (PFMs), pretrained on large-scale histopathology data, have significantly enhanced capabilities of extractors and aggregators but lack systematic analysis frameworks. This survey presents a hierarchical taxonomy organizing PFMs through a top-down philosophy that can be utilized to analyze FMs in any domain: model scope, model pretraining, and model design. Additionally, we systematically categorize PFM evaluation tasks into slide-level, patch-level, multimodal, and biological tasks, providing comprehensive benchmarking criteria. Our analysis identifies critical challenges in both PFM development (pathology-specific methodology, end-to-end pretraining, data-model scalability) and utilization (effective adaptation, model maintenance), paving the way for future directions in this promising field. Resources referenced in this survey are available at this https URL. 

---
# JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model 

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03770)  

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed. 

---
# TC-MGC: Text-Conditioned Multi-Grained Contrastive Learning for Text-Video Retrieval 

**Authors**: Xiaolun Jing, Genke Yang, Jian Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04707)  

**Abstract**: Motivated by the success of coarse-grained or fine-grained contrast in text-video retrieval, there emerge multi-grained contrastive learning methods which focus on the integration of contrasts with different granularity. However, due to the wider semantic range of videos, the text-agnostic video representations might encode misleading information not described in texts, thus impeding the model from capturing precise cross-modal semantic correspondence. To this end, we propose a Text-Conditioned Multi-Grained Contrast framework, dubbed TC-MGC. Specifically, our model employs a language-video attention block to generate aggregated frame and video representations conditioned on the word's and text's attention weights over frames. To filter unnecessary similarity interactions and decrease trainable parameters in the Interactive Similarity Aggregation (ISA) module, we design a Similarity Reorganization (SR) module to identify attentive similarities and reorganize cross-modal similarity vectors and matrices. Next, we argue that the imbalance problem among multigrained similarities may result in over- and under-representation issues. We thereby introduce an auxiliary Similarity Decorrelation Regularization (SDR) loss to facilitate cooperative relationship utilization by similarity variance minimization on matching text-video pairs. Finally, we present a Linear Softmax Aggregation (LSA) module to explicitly encourage the interactions between multiple similarities and promote the usage of multi-grained information. Empirically, TC-MGC achieves competitive results on multiple text-video retrieval benchmarks, outperforming X-CLIP model by +2.8% (+1.3%), +2.2% (+1.0%), +1.5% (+0.9%) relative (absolute) improvements in text-to-video retrieval R@1 on MSR-VTT, DiDeMo and VATEX, respectively. Our code is publicly available at this https URL. 

---
# COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation 

**Authors**: Jinfeng Xu, Zheyu Chen, Wei Wang, Xiping Hu, Sang-Wook Kim, Edith C. H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04452)  

**Abstract**: Recent works in multimodal recommendations, which leverage diverse modal information to address data sparsity and enhance recommendation accuracy, have garnered considerable interest. Two key processes in multimodal recommendations are modality fusion and representation learning. Previous approaches in modality fusion often employ simplistic attentive or pre-defined strategies at early or late stages, failing to effectively handle irrelevant information among modalities. In representation learning, prior research has constructed heterogeneous and homogeneous graph structures encapsulating user-item, user-user, and item-item relationships to better capture user interests and item profiles. Modality fusion and representation learning were considered as two independent processes in previous work. In this paper, we reveal that these two processes are complementary and can support each other. Specifically, powerful representation learning enhances modality fusion, while effective fusion improves representation quality. Stemming from these two processes, we introduce a COmposite grapH convolutional nEtwork with dual-stage fuSION for the multimodal recommendation, named COHESION. Specifically, it introduces a dual-stage fusion strategy to reduce the impact of irrelevant information, refining all modalities using ID embedding in the early stage and fusing their representations at the late stage. It also proposes a composite graph convolutional network that utilizes user-item, user-user, and item-item graphs to extract heterogeneous and homogeneous latent relationships within users and items. Besides, it introduces a novel adaptive optimization to ensure balanced and reasonable representations across modalities. Extensive experiments on three widely used datasets demonstrate the significant superiority of COHESION over various competitive baselines. 

---
# UniRVQA: A Unified Framework for Retrieval-Augmented Vision Question Answering via Self-Reflective Joint Training 

**Authors**: Jiaqi Deng, Kaize Shi, Zonghan Wu, Huan Huo, Dingxian Wang, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04065)  

**Abstract**: Knowledge-based Vision Question Answering (KB-VQA) systems address complex visual-grounded questions requiring external knowledge, such as web-sourced encyclopedia articles. Existing methods often use sequential and separate frameworks for the retriever and the generator with limited parametric knowledge sharing. However, since both retrieval and generation tasks require accurate understanding of contextual and external information, such separation can potentially lead to suboptimal system performance. Another key challenge is the integration of multimodal information. General-purpose multimodal pre-trained models, while adept at multimodal representation learning, struggle with fine-grained retrieval required for knowledge-intensive visual questions. Recent specialized pre-trained models mitigate the issue, but are computationally expensive. To bridge the gap, we propose a Unified Retrieval-Augmented VQA framework (UniRVQA). UniRVQA adapts general multimodal pre-trained models for fine-grained knowledge-intensive tasks within a unified framework, enabling cross-task parametric knowledge sharing and the extension of existing multimodal representation learning capability. We further introduce a reflective-answering mechanism that allows the model to explicitly evaluate and refine its knowledge boundary. Additionally, we integrate late interaction into the retrieval-augmented generation joint training process to enhance fine-grained understanding of queries and documents. Our approach achieves competitive performance against state-of-the-art models, delivering a significant 4.7% improvement in answering accuracy, and brings an average 7.5% boost in base MLLMs' VQA performance. 

---
