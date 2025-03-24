# Towards Carbon Footprint-Aware Recommender Systems for Greener Item Recommendation 

**Authors**: Raoul Kalisvaart, Masoud Mansoury, Alan Hanjalic, Elvin Isufi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17201)  

**Abstract**: The commodity and widespread use of online shopping are having an unprecedented impact on climate, with emission figures from key actors that are easily comparable to those of a large-scale metropolis. Despite online shopping being fueled by recommender systems (RecSys) algorithms, the role and potential of the latter in promoting more sustainable choices is little studied. One of the main reasons for this could be attributed to the lack of a dataset containing carbon footprint emissions for the items. While building such a dataset is a rather challenging task, its presence is pivotal for opening the doors to novel perspectives, evaluations, and methods for RecSys research. In this paper, we target this bottleneck and study the environmental role of RecSys algorithms. First, we mine a dataset that includes carbon footprint emissions for its items. Then, we benchmark conventional RecSys algorithms in terms of accuracy and sustainability as two faces of the same coin. We find that RecSys algorithms optimized for accuracy overlook greenness and that longer recommendation lists are greener but less accurate. Then, we show that a simple reranking approach that accounts for the item's carbon footprint can establish a better trade-off between accuracy and greenness. This reranking approach is modular, ready to use, and can be applied to any RecSys algorithm without the need to alter the underlying mechanisms or retrain models. Our results show that a small sacrifice of accuracy can lead to significant improvements of recommendation greenness across all algorithms and list lengths. Arguably, this accuracy-greenness trade-off could even be seen as an enhancement of user satisfaction, particularly for purpose-driven users who prioritize the environmental impact of their choices. We anticipate this work will serve as the starting point for studying RecSys for more sustainable recommendations. 

---
# Rankformer: A Graph Transformer for Recommendation based on Ranking Objective 

**Authors**: Sirui Chen, Shen Han, Jiawei Chen, Binbin Hu, Sheng Zhou, Gang Wang, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16927)  

**Abstract**: Recommender Systems (RS) aim to generate personalized ranked lists for each user and are evaluated using ranking metrics. Although personalized ranking is a fundamental aspect of RS, this critical property is often overlooked in the design of model architectures. To address this issue, we propose Rankformer, a ranking-inspired recommendation model. The architecture of Rankformer is inspired by the gradient of the ranking objective, embodying a unique (graph) transformer architecture -- it leverages global information from all users and items to produce more informative representations and employs specific attention weights to guide the evolution of embeddings towards improved ranking performance. We further develop an acceleration algorithm for Rankformer, reducing its complexity to a linear level with respect to the number of positive instances. Extensive experimental results demonstrate that Rankformer outperforms state-of-the-art methods. The code is available at this https URL. 

---
# Federated Cross-Domain Click-Through Rate Prediction With Large Language Model Augmentation 

**Authors**: Jiangcheng Qin, Xueyuan Zhang, Baisong Liu, Jiangbo Qian, Yangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16875)  

**Abstract**: Accurately predicting click-through rates (CTR) under stringent privacy constraints poses profound challenges, particularly when user-item interactions are sparse and fragmented across domains. Conventional cross-domain CTR (CCTR) methods frequently assume homogeneous feature spaces and rely on centralized data sharing, neglecting complex inter-domain discrepancies and the subtle trade-offs imposed by privacy-preserving protocols. Here, we present Federated Cross-Domain CTR Prediction with Large Language Model Augmentation (FedCCTR-LM), a federated framework engineered to address these limitations by synchronizing data augmentation, representation disentanglement, and adaptive privacy protection. Our approach integrates three core innovations. First, the Privacy-Preserving Augmentation Network (PrivAugNet) employs large language models to enrich user and item representations and expand interaction sequences, mitigating data sparsity and feature incompleteness. Second, the Independent Domain-Specific Transformer with Contrastive Learning (IDST-CL) module disentangles domain-specific and shared user preferences, employing intra-domain representation alignment (IDRA) and crossdomain representation disentanglement (CDRD) to refine the learned embeddings and enhance knowledge transfer across domains. Finally, the Adaptive Local Differential Privacy (AdaLDP) mechanism dynamically calibrates noise injection to achieve an optimal balance between rigorous privacy guarantees and predictive accuracy. Empirical evaluations on four real-world datasets demonstrate that FedCCTR-LM substantially outperforms existing baselines, offering robust, privacy-preserving, and generalizable cross-domain CTR prediction in heterogeneous, federated environments. 

---
# The CASTLE 2024 Dataset: Advancing the Art of Multimodal Understanding 

**Authors**: Luca Rossetto, Werner Bailer, Duc-Tien Dang-Nguyen, Graham Healy, Björn Þór Jónsson, Onanong Kongmeesub, Hoang-Bao Le, Stevan Rudinac, Klaus Schöffmann, Florian Spiess, Allie Tran, Minh-Triet Tran, Quang-Linh Tran, Cathal Gurrin  

**Link**: [PDF](https://arxiv.org/pdf/2503.17116)  

**Abstract**: Egocentric video has seen increased interest in recent years, as it is used in a range of areas. However, most existing datasets are limited to a single perspective. In this paper, we present the CASTLE 2024 dataset, a multimodal collection containing ego- and exo-centric (i.e., first- and third-person perspective) video and audio from 15 time-aligned sources, as well as other sensor streams and auxiliary data. The dataset was recorded by volunteer participants over four days in a fixed location and includes the point of view of 10 participants, with an additional 5 fixed cameras providing an exocentric perspective. The entire dataset contains over 600 hours of UHD video recorded at 50 frames per second. In contrast to other datasets, CASTLE 2024 does not contain any partial censoring, such as blurred faces or distorted audio. The dataset is available via this https URL. 

---
# A Study into Investigating Temporal Robustness of LLMs 

**Authors**: Jonas Wallat, Abdelrahman Abdallah, Adam Jatowt, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2503.17073)  

**Abstract**: Large Language Models (LLMs) encapsulate a surprising amount of factual world knowledge. However, their performance on temporal questions and historical knowledge is limited because they often cannot understand temporal scope and orientation or neglect the temporal aspect altogether. In this study, we aim to measure precisely how robust LLMs are for question answering based on their ability to process temporal information and perform tasks requiring temporal reasoning and temporal factual knowledge. Specifically, we design eight time-sensitive robustness tests for factual information to check the sensitivity of six popular LLMs in the zero-shot setting. Overall, we find LLMs lacking temporal robustness, especially to temporal reformulations and the use of different granularities of temporal references. We show how a selection of these eight tests can be used automatically to judge a model's temporal robustness for user questions on the fly. Finally, we apply the findings of this study to improve the temporal QA performance by up to 55 percent. 

---
# Towards Agentic Recommender Systems in the Era of Multimodal Large Language Models 

**Authors**: Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang, Ryan A. Rossi, Branislav Kveton, Dongruo Zhou, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16734)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to the emergence of agentic AI systems that extend beyond the capabilities of standalone models. By empowering LLMs to perceive external environments, integrate multimodal information, and interact with various tools, these agentic systems exhibit greater autonomy and adaptability across complex tasks. This evolution brings new opportunities to recommender systems (RS): LLM-based Agentic RS (LLM-ARS) can offer more interactive, context-aware, and proactive recommendations, potentially reshaping the user experience and broadening the application scope of RS. Despite promising early results, fundamental challenges remain, including how to effectively incorporate external knowledge, balance autonomy with controllability, and evaluate performance in dynamic, multimodal settings. In this perspective paper, we first present a systematic analysis of LLM-ARS: (1) clarifying core concepts and architectures; (2) highlighting how agentic capabilities -- such as planning, memory, and multimodal reasoning -- can enhance recommendation quality; and (3) outlining key research questions in areas such as safety, efficiency, and lifelong personalization. We also discuss open problems and future directions, arguing that LLM-ARS will drive the next wave of RS innovation. Ultimately, we foresee a paradigm shift toward intelligent, autonomous, and collaborative recommendation experiences that more closely align with users' evolving needs and complex decision-making processes. 

---
# Informative Path Planning to Explore and Map Unknown Planetary Surfaces with Gaussian Processes 

**Authors**: Ashten Akemoto, Frances Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16613)  

**Abstract**: Many environments, such as unvisited planetary surfaces and oceanic regions, remain unexplored due to a lack of prior knowledge. Autonomous vehicles must sample upon arrival, process data, and either transmit findings to a teleoperator or decide where to explore next. Teleoperation is suboptimal, as human intuition lacks mathematical guarantees for optimality. This study evaluates an informative path planning algorithm for mapping a scalar variable distribution while minimizing travel distance and ensuring model convergence. We compare traditional open loop coverage methods (e.g., Boustrophedon, Spiral) with information-theoretic approaches using Gaussian processes, which update models iteratively with confidence metrics. The algorithm's performance is tested on three surfaces, a parabola, Townsend function, and lunar crater hydration map, to assess noise, convexity, and function behavior. Results demonstrate that information-driven methods significantly outperform naive exploration in reducing model error and travel distance while improving convergence potential. 

---
# Enhancing LLM Generation with Knowledge Hypergraph for Evidence-Based Medicine 

**Authors**: Chengfeng Dou, Ying Zhang, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang Zhao, Zhengwei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16530)  

**Abstract**: Evidence-based medicine (EBM) plays a crucial role in the application of large language models (LLMs) in healthcare, as it provides reliable support for medical decision-making processes. Although it benefits from current retrieval-augmented generation~(RAG) technologies, it still faces two significant challenges: the collection of dispersed evidence and the efficient organization of this evidence to support the complex queries necessary for EBM. To tackle these issues, we propose using LLMs to gather scattered evidence from multiple sources and present a knowledge hypergraph-based evidence management model to integrate these evidence while capturing intricate relationships. Furthermore, to better support complex queries, we have developed an Importance-Driven Evidence Prioritization (IDEP) algorithm that utilizes the LLM to generate multiple evidence features, each with an associated importance score, which are then used to rank the evidence and produce the final retrieval results. Experimental results from six datasets demonstrate that our approach outperforms existing RAG techniques in application domains of interest to EBM, such as medical quizzing, hallucination detection, and decision support. Testsets and the constructed knowledge graph can be accessed at \href{this https URL}{this https URL}. 

---
# LeRAAT: LLM-Enabled Real-Time Aviation Advisory Tool 

**Authors**: Marc R. Schlichting, Vale Rasmussen, Heba Alazzeh, Houjun Liu, Kiana Jafari, Amelia F. Hardy, Dylan M. Asmar, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16477)  

**Abstract**: In aviation emergencies, high-stakes decisions must be made in an instant. Pilots rely on quick access to precise, context-specific information -- an area where emerging tools like large language models (LLMs) show promise in providing critical support. This paper introduces LeRAAT, a framework that integrates LLMs with the X-Plane flight simulator to deliver real-time, context-aware pilot assistance. The system uses live flight data, weather conditions, and aircraft documentation to generate recommendations aligned with aviation best practices and tailored to the particular situation. It employs a Retrieval-Augmented Generation (RAG) pipeline that extracts and synthesizes information from aircraft type-specific manuals, including performance specifications and emergency procedures, as well as aviation regulatory materials, such as FAA directives and standard operating procedures. We showcase the framework in both a virtual reality and traditional on-screen simulation, supporting a wide range of research applications such as pilot training, human factors research, and operational decision support. 

---
# CLIP-PING: Boosting Lightweight Vision-Language Models with Proximus Intrinsic Neighbors Guidance 

**Authors**: Chu Myaet Thwal, Ye Lin Tun, Minh N. H. Nguyen, Eui-Nam Huh, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.03871)  

**Abstract**: Beyond the success of Contrastive Language-Image Pre-training (CLIP), recent trends mark a shift toward exploring the applicability of lightweight vision-language models for resource-constrained scenarios. These models often deliver suboptimal performance when relying solely on a single image-text contrastive learning objective, spotlighting the need for more effective training mechanisms that guarantee robust cross-modal feature alignment. In this work, we propose CLIP-PING: Contrastive Language-Image Pre-training with Proximus Intrinsic Neighbors Guidance, a novel yet simple and efficient training paradigm designed to boost the performance of lightweight vision-language models with minimal computational overhead and lower data demands. CLIP-PING bootstraps unimodal features extracted from arbitrary pre-trained encoders to obtain intrinsic guidance of proximus neighbor samples, i.e., nearest-neighbor (NN) and cross nearest-neighbor (XNN). We find that extra contrastive supervision from these neighbors substantially boosts cross-modal alignment, enabling lightweight models to learn more generic features with rich semantic diversity. Extensive experiments reveal that CLIP-PING notably surpasses its peers in zero-shot generalization and cross-modal retrieval tasks. Specifically, a 5.5% gain on zero-shot ImageNet1K classification with 10.7% (I2T) and 5.7% (T2I) on Flickr30K retrieval, compared to the original CLIP when using ViT-XS image encoder trained on 3 million (image, text) pairs. Moreover, CLIP-PING showcases a strong transferability under the linear evaluation protocol across several downstream tasks. 

---
