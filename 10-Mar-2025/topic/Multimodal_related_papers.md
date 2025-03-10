# MM-StoryAgent: Immersive Narrated Storybook Video Generation with a Multi-Agent Paradigm across Text, Image and Audio 

**Authors**: Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05242)  

**Abstract**: The rapid advancement of large language models (LLMs) and artificial intelligence-generated content (AIGC) has accelerated AI-native applications, such as AI-based storybooks that automate engaging story production for children. However, challenges remain in improving story attractiveness, enriching storytelling expressiveness, and developing open-source evaluation benchmarks and frameworks. Therefore, we propose and opensource MM-StoryAgent, which creates immersive narrated video storybooks with refined plots, role-consistent images, and multi-channel audio. MM-StoryAgent designs a multi-agent framework that employs LLMs and diverse expert tools (generative models and APIs) across several modalities to produce expressive storytelling videos. The framework enhances story attractiveness through a multi-stage writing pipeline. In addition, it improves the immersive storytelling experience by integrating sound effects with visual, music and narrative assets. MM-StoryAgent offers a flexible, open-source platform for further development, where generative modules can be substituted. Both objective and subjective evaluation regarding textual story quality and alignment between modalities validate the effectiveness of our proposed MM-StoryAgent system. The demo and source code are available. 

---
# Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models 

**Authors**: Boyu Jia, Junzhe Zhang, Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04801)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have achieved significant breakthroughs, enhancing understanding across text and vision. However, current MLLMs still face challenges in effectively integrating knowledge across these modalities during multimodal knowledge reasoning, leading to inconsistencies in reasoning outcomes. To systematically explore this issue, we propose four evaluation tasks and construct a new dataset. We conduct a series of experiments on this dataset to analyze and compare the extent of consistency degradation in multimodal knowledge reasoning within MLLMs. Based on the experimental results, we identify factors contributing to the observed degradation in consistency. Our research provides new insights into the challenges of multimodal knowledge reasoning and offers valuable guidance for future efforts aimed at improving MLLMs. 

---
# SuperRAG: Beyond RAG with Layout-Aware Graph Modeling 

**Authors**: Jeff Yang, Duy-Khanh Vu, Minh-Tien Nguyen, Xuan-Quang Nguyen, Linh Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2503.04790)  

**Abstract**: This paper introduces layout-aware graph modeling for multimodal RAG. Different from traditional RAG methods that mostly deal with flat text chunks, the proposed method takes into account the relationship of multimodalities by using a graph structure. To do that, a graph modeling structure is defined based on document layout parsing. The structure of an input document is retained with the connection of text chunks, tables, and figures. This representation allows the method to handle complex questions that require information from multimodalities. To confirm the efficiency of the graph modeling, a flexible RAG pipeline is developed using robust components. Experimental results on four benchmark test sets confirm the contribution of the layout-aware modeling for performance improvement of the RAG pipeline. 

---
# MV-CLAM: Multi-View Molecular Interpretation with Cross-Modal Projection via Language Model 

**Authors**: Sumin Ha, Jun Hyeong Kim, Yinhua Piao, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04780)  

**Abstract**: Human expertise in chemistry and biomedicine relies on contextual molecular understanding, a capability that large language models (LLMs) can extend through fine-grained alignment between molecular structures and text. Recent multimodal learning advances focus on cross-modal alignment, but existing molecule-text models ignore complementary information in different molecular views and rely on single-view representations, limiting molecular understanding. Moreover, naïve multi-view alignment strategies face two challenges: (1) separate aligned spaces with inconsistent mappings between molecule and text embeddings, and that (2) existing loss objectives fail to preserve complementary information for fine-grained alignment. This can limit the LLM's ability to fully understand the molecular properties. To address these issues, we propose MV-CLAM, a novel framework that aligns multi-view molecular representations into a unified textual space using a multi-query transformer (MQ-Former). Our approach ensures cross-view consistency while a token-level contrastive loss preserves diverse molecular features across textual queries. MV-CLAM enhances molecular reasoning, improving retrieval and captioning accuracy. The source code of MV-CLAM is available in this https URL. 

---
# WinClick: GUI Grounding with Multimodal Large Language Models 

**Authors**: Zheng Hui, Yinheng Li, Dan zhao, Tianyi Chen, Colby Banbury, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2503.04730)  

**Abstract**: Graphical User Interface (GUI) tasks are vital for automating workflows such as software testing, user interface navigation. For users, the GUI is the most intuitive platform for interacting with a computer. Previous work identified a key challenge in developing visual GUI agents: GUI grounding - the ability to accurately locate screen elements based on instructions. However, most existing GUI agents rely on structured data formats like DOM or HTML files in training or inferencing, which are inaccessible across all applications, particular in a general desktop environments such as Windows OS. To address this, we introduce WinClick, a novel visual GUI agent developed in Windows platform. WinClick leverages screenshots to detect actionable regions. To overcome the challenge of GUI grounding, we enhance WinClick with GUI grounding pre-training and propose an LLM-based method for aligning GUI grounding data. Additionally, we introduce WinSpot, the first comprehensive benchmark for GUI grounding on Windows. Our experiments demonstrate that WinClick, combined with GUI grounding pre-training, significantly outperforms existing baselines, offering a scalable solution for GUI automation in desktop environments. WinSpot is publicly available at this https URL. 

---
# Advancing Multimodal In-Context Learning in Large Vision-Language Models with Task-aware Demonstrations 

**Authors**: Yanshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04839)  

**Abstract**: Multimodal in-context learning (ICL) has emerged as a key capability of Large Vision-Language Models (LVLMs), driven by their increasing scale and applicability. Despite its promise, effective ICL in the multimodal setting remains challenging due to the inherent complexity of image-text inputs and the high sensitivity of ICL performance to input configurations. In this work, we shed light on the core mechanism underlying multimodal ICL, identifying task mapping as a crucial factor in configuring robust in-context demonstration (ICD) sequences. Building on these insights, we propose \textit{SabER}, a lightweight yet powerful decoder-only transformer equipped with task-aware attention, which intelligently selects and arranges ICDs from a demonstration library in an autoregressive fashion. This design enables fine-grained feature extraction and cross-modal reasoning, iteratively refining task mapping to generate high-quality ICD sequences. Through extensive experiments covering five LVLMs and nine benchmark datasets, SabER not only demonstrates strong empirical performance, but also provides deeper understanding of how task semantics interact with multimodal ICDs. Our findings highlight the importance of principled ICD sequence configuration and open new avenues to enhance multimodal ICL in a wide range of real-world scenarios. 

---
# LVLM-Compress-Bench: Benchmarking the Broader Impact of Large Vision-Language Model Compression 

**Authors**: Souvik Kundu, Anahita Bhiwandiwalla, Sungduk Yu, Phillip Howard, Tiep Le, Sharath Nittur Sridhar, David Cobbley, Hao Kang, Vasudev Lal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04982)  

**Abstract**: Despite recent efforts in understanding the compression impact on large language models (LLMs) in terms of their downstream task performance and trustworthiness on relatively simpler uni-modal benchmarks (for example, question answering, common sense reasoning), their detailed study on multi-modal Large Vision-Language Models (LVLMs) is yet to be unveiled. Towards mitigating this gap, we present LVLM-Compress-Bench, a framework to first thoroughly study the broad impact of compression on the generative performance of LVLMs with multi-modal input driven tasks. In specific, we consider two major classes of compression for autoregressive models, namely KV cache and weight compression, for the dynamically growing intermediate cache and static weights, respectively.
We use four LVLM variants of the popular LLaVA framework to present our analysis via integrating various state-of-the-art KV and weight compression methods including uniform, outlier-reduced, and group quantization for the KV cache and weights. With this framework we demonstrate on ten different multi-modal datasets with different capabilities including recognition, knowledge, language generation, spatial awareness, visual reasoning, hallucination and visual illusion identification, toxicity, stereotypes and bias. In specific, our framework demonstrates the compression impact on both general and ethically critical metrics leveraging a combination of real world and synthetic datasets to encompass diverse societal intersectional attributes. Extensive experimental evaluations yield diverse and intriguing observations on the behavior of LVLMs at different quantization budget of KV and weights, in both maintaining and losing performance as compared to the baseline model with FP16 data format.
Code will be open-sourced at this https URL. 

---
# LLaVE: Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning 

**Authors**: Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.04812)  

**Abstract**: Universal multimodal embedding models play a critical role in tasks such as interleaved image-text retrieval, multimodal RAG, and multimodal clustering. However, our empirical results indicate that existing LMM-based embedding models trained with the standard InfoNCE loss exhibit a high degree of overlap in similarity distribution between positive and negative pairs, making it challenging to distinguish hard negative pairs effectively. To deal with this issue, we propose a simple yet effective framework that dynamically improves the embedding model's representation learning for negative pairs based on their discriminative difficulty. Within this framework, we train a series of models, named LLaVE, and evaluate them on the MMEB benchmark, which covers 4 meta-tasks and 36 datasets. Experimental results show that LLaVE establishes stronger baselines that achieve state-of-the-art (SOTA) performance while demonstrating strong scalability and efficiency. Specifically, LLaVE-2B surpasses the previous SOTA 7B models, while LLaVE-7B achieves a further performance improvement of 6.2 points. Although LLaVE is trained on image-text data, it can generalize to text-video retrieval tasks in a zero-shot manner and achieve strong performance, demonstrating its remarkable potential for transfer to other embedding tasks. 

---
# Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks 

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04833)  

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems. 

---
# R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model 

**Authors**: Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05132)  

**Abstract**: Recently DeepSeek R1 demonstrated how reinforcement learning with simple rule-based incentives can enable autonomous development of complex reasoning in large language models, characterized by the "aha moment", in which the model manifest self-reflection and increased response length during training. However, attempts to extend this success to multimodal reasoning often failed to reproduce these key characteristics. In this report, we present the first successful replication of these emergent characteristics for multimodal reasoning on only a non-SFT 2B model. Starting with Qwen2-VL-2B and applying reinforcement learning directly on the SAT dataset, our model achieves 59.47% accuracy on CVBench, outperforming the base model by approximately ~30% and exceeding both SFT setting by ~2%. In addition, we share our failed attempts and insights in attempting to achieve R1-like reasoning using RL with instruct models. aiming to shed light on the challenges involved. Our key observations include: (1) applying RL on instruct model often results in trivial reasoning trajectories, and (2) naive length reward are ineffective in eliciting reasoning capabilities. The project code is available at this https URL 

---
# VLMs Play StarCraft II: A Benchmark and Multimodal Decision Method 

**Authors**: Weiyu Ma, Yuqian Fu, Zecheng Zhang, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05383)  

**Abstract**: We introduce VLM-Attention, a multimodal StarCraft II environment that aligns artificial agent perception with the human gameplay experience. Traditional frameworks such as SMAC rely on abstract state representations that diverge significantly from human perception, limiting the ecological validity of agent behavior. Our environment addresses this limitation by incorporating RGB visual inputs and natural language observations that more closely simulate human cognitive processes during gameplay. The VLM-Attention framework consists of three integrated components: (1) a vision-language model enhanced with specialized self-attention mechanisms for strategic unit targeting and battlefield assessment, (2) a retrieval-augmented generation system that leverages domain-specific StarCraft II knowledge to inform tactical decisions, and (3) a dynamic role-based task distribution system that enables coordinated multi-agent behavior. Our experimental evaluation across 21 custom scenarios demonstrates that VLM-based agents powered by foundation models (specifically Qwen-VL and GPT-4o) can execute complex tactical maneuvers without explicit training, achieving comparable performance to traditional MARL methods that require substantial training iterations. This work establishes a foundation for developing human-aligned StarCraft II agents and advances the broader research agenda of multimodal game AI. Our implementation is available at this https URL. 

---
# FMT:A Multimodal Pneumonia Detection Model Based on Stacking MOE Framework 

**Authors**: Jingyu Xu, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05626)  

**Abstract**: Artificial intelligence has shown the potential to improve diagnostic accuracy through medical image analysis for pneumonia diagnosis. However, traditional multimodal approaches often fail to address real-world challenges such as incomplete data and modality loss. In this study, a Flexible Multimodal Transformer (FMT) was proposed, which uses ResNet-50 and BERT for joint representation learning, followed by a dynamic masked attention strategy that simulates clinical modality loss to improve robustness; finally, a sequential mixture of experts (MOE) architecture was used to achieve multi-level decision refinement. After evaluation on a small multimodal pneumonia dataset, FMT achieved state-of-the-art performance with 94% accuracy, 95% recall, and 93% F1 score, outperforming single-modal baselines (ResNet: 89%; BERT: 79%) and the medical benchmark CheXMed (90%), providing a scalable solution for multimodal diagnosis of pneumonia in resource-constrained medical settings. 

---
# Robust Multimodal Learning for Ophthalmic Disease Grading via Disentangled Representation 

**Authors**: Xinkun Wang, Yifang Wang, Senwei Liang, Feilong Tang, Chengzhi Liu, Ming Hu, Chao Hu, Junjun He, Zongyuan Ge, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2503.05319)  

**Abstract**: This paper discusses how ophthalmologists often rely on multimodal data to improve diagnostic accuracy. However, complete multimodal data is rare in real-world applications due to a lack of medical equipment and concerns about data privacy. Traditional deep learning methods typically address these issues by learning representations in latent space. However, the paper highlights two key limitations of these approaches: (i) Task-irrelevant redundant information (e.g., numerous slices) in complex modalities leads to significant redundancy in latent space representations. (ii) Overlapping multimodal representations make it difficult to extract unique features for each modality. To overcome these challenges, the authors propose the Essence-Point and Disentangle Representation Learning (EDRL) strategy, which integrates a self-distillation mechanism into an end-to-end framework to enhance feature selection and disentanglement for more robust multimodal learning. Specifically, the Essence-Point Representation Learning module selects discriminative features that improve disease grading performance. The Disentangled Representation Learning module separates multimodal data into modality-common and modality-unique representations, reducing feature entanglement and enhancing both robustness and interpretability in ophthalmic disease diagnosis. Experiments on multimodal ophthalmology datasets show that the proposed EDRL strategy significantly outperforms current state-of-the-art methods. 

---
# Kaiwu: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction 

**Authors**: Shuo Jiang, Haonan Li, Ruochen Ren, Yanmin Zhou, Zhipeng Wang, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2503.05231)  

**Abstract**: Cutting-edge robot learning techniques including foundation models and imitation learning from humans all pose huge demands on large-scale and high-quality datasets which constitute one of the bottleneck in the general intelligent robot fields. This paper presents the Kaiwu multimodal dataset to address the missing real-world synchronized multimodal data problems in the sophisticated assembling scenario,especially with dynamics information and its fine-grained labelling. The dataset first provides an integration of human,environment and robot data collection framework with 20 subjects and 30 interaction objects resulting in totally 11,664 instances of integrated actions. For each of the demonstration,hand motions,operation pressures,sounds of the assembling process,multi-view videos, high-precision motion capture information,eye gaze with first-person videos,electromyography signals are all recorded. Fine-grained multi-level annotation based on absolute timestamp,and semantic segmentation labelling are performed. Kaiwu dataset aims to facilitate robot learning,dexterous manipulation,human intention investigation and human-robot collaboration research. 

---
# FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance 

**Authors**: Fengbin Zhu, Junfeng Li, Liangming Pan, Wenjie Wang, Fuli Feng, Chao Wang, Huanbo Luan, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.05185)  

**Abstract**: Finance decision-making often relies on in-depth data analysis across various data sources, including financial tables, news articles, stock prices, etc. In this work, we introduce FinTMMBench, the first comprehensive benchmark for evaluating temporal-aware multi-modal Retrieval-Augmented Generation (RAG) systems in finance. Built from heterologous data of NASDAQ 100 companies, FinTMMBench offers three significant advantages. 1) Multi-modal Corpus: It encompasses a hybrid of financial tables, news articles, daily stock prices, and visual technical charts as the corpus. 2) Temporal-aware Questions: Each question requires the retrieval and interpretation of its relevant data over a specific time period, including daily, weekly, monthly, quarterly, and annual periods. 3) Diverse Financial Analysis Tasks: The questions involve 10 different tasks, including information extraction, trend analysis, sentiment analysis and event detection, etc. We further propose a novel TMMHybridRAG method, which first leverages LLMs to convert data from other modalities (e.g., tabular, visual and time-series data) into textual format and then incorporates temporal information in each node when constructing graphs and dense indexes. Its effectiveness has been validated in extensive experiments, but notable gaps remain, highlighting the challenges presented by our FinTMMBench. 

---
# SHAPE : Self-Improved Visual Preference Alignment by Iteratively Generating Holistic Winner 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Jiazhen Yang, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.04858)  

**Abstract**: Large Visual Language Models (LVLMs) increasingly rely on preference alignment to ensure reliability, which steers the model behavior via preference fine-tuning on preference data structured as ``image - winner text - loser text'' triplets. However, existing approaches often suffer from limited diversity and high costs associated with human-annotated preference data, hindering LVLMs from fully achieving their intended alignment capabilities. We present \projectname, a self-supervised framework capable of transforming the already abundant supervised text-image pairs into holistic preference triplets for more effective and cheaper LVLM alignment, eliminating the need for human preference annotations. Our approach facilitates LVLMs in progressively enhancing alignment capabilities through iterative self-improvement. The key design rationale is to devise preference triplets where the winner text consistently improves in holisticness and outperforms the loser response in quality, thereby pushing the model to ``strive to the utmost'' of alignment performance through preference fine-tuning. For each given text-image pair, SHAPE introduces multiple visual augmentations and pairs them with a summarized text to serve as the winner response, while designating the original text as the loser response. Experiments across \textbf{12} benchmarks on various model architectures and sizes, including LLaVA and DeepSeek-VL, show that SHAPE achieves significant gains, for example, achieving +11.3\% on MMVet (comprehensive evaluation), +1.4\% on MMBench (general VQA), and +8.0\% on POPE (hallucination robustness) over baselines in 7B models. Notably, qualitative analyses confirm enhanced attention to visual details and better alignment with human preferences for holistic descriptions. 

---
# Replicating Human Social Perception in Generative AI: Evaluating the Valence-Dominance Model 

**Authors**: Necdet Gurkan, Kimathi Njoki, Jordan W. Suchow  

**Link**: [PDF](https://arxiv.org/pdf/2503.04842)  

**Abstract**: As artificial intelligence (AI) continues to advance--particularly in generative models--an open question is whether these systems can replicate foundational models of human social perception. A well-established framework in social cognition suggests that social judgments are organized along two primary dimensions: valence (e.g., trustworthiness, warmth) and dominance (e.g., power, assertiveness). This study examines whether multimodal generative AI systems can reproduce this valence-dominance structure when evaluating facial images and how their representations align with those observed across world regions. Through principal component analysis (PCA), we found that the extracted dimensions closely mirrored the theoretical structure of valence and dominance, with trait loadings aligning with established definitions. However, many world regions and generative AI models also exhibited a third component, the nature and significance of which warrant further investigation. These findings demonstrate that multimodal generative AI systems can replicate key aspects of human social perception, raising important questions about their implications for AI-driven decision-making and human-AI interactions. 

---
# PGAD: Prototype-Guided Adaptive Distillation for Multi-Modal Learning in AD Diagnosis 

**Authors**: Yanfei Li, Teng Yin, Wenyi Shang, Jingyu Liu, Xi Wang, Kaiyang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04836)  

**Abstract**: Missing modalities pose a major issue in Alzheimer's Disease (AD) diagnosis, as many subjects lack full imaging data due to cost and clinical constraints. While multi-modal learning leverages complementary information, most existing methods train only on complete data, ignoring the large proportion of incomplete samples in real-world datasets like ADNI. This reduces the effective training set and limits the full use of valuable medical data. While some methods incorporate incomplete samples, they fail to effectively address inter-modal feature alignment and knowledge transfer challenges under high missing rates. To address this, we propose a Prototype-Guided Adaptive Distillation (PGAD) framework that directly incorporates incomplete multi-modal data into training. PGAD enhances missing modality representations through prototype matching and balances learning with a dynamic sampling strategy. We validate PGAD on the ADNI dataset with varying missing rates (20%, 50%, and 70%) and demonstrate that it significantly outperforms state-of-the-art approaches. Ablation studies confirm the effectiveness of prototype matching and adaptive sampling, highlighting the potential of our framework for robust and scalable AD diagnosis in real-world clinical settings. 

---
# RTFusion: A depth estimation network based on multimodal fusion in challenging scenarios 

**Authors**: Zelin Meng, Takanori Fukao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04821)  

**Abstract**: Depth estimation in complex real-world scenarios is a challenging task, especially when relying solely on a single modality such as visible light or thermal infrared (THR) imagery. This paper proposes a novel multimodal depth estimation model, RTFusion, which enhances depth estimation accuracy and robustness by integrating the complementary strengths of RGB and THR data. The RGB modality provides rich texture and color information, while the THR modality captures thermal patterns, ensuring stability under adverse lighting conditions such as extreme illumination. The model incorporates a unique fusion mechanism, EGFusion, consisting of the Mutual Complementary Attention (MCA) module for cross-modal feature alignment and the Edge Saliency Enhancement Module (ESEM) to improve edge detail preservation. Comprehensive experiments on the MS2 and ViViD++ datasets demonstrate that the proposed model consistently produces high-quality depth maps across various challenging environments, including nighttime, rainy, and high-glare conditions. The experimental results highlight the potential of the proposed method in applications requiring reliable depth estimation, such as autonomous driving, robotics, and augmented reality. 

---
