# Multimodal Sentiment Analysis on CMU-MOSEI Dataset using Transformer-based Models 

**Authors**: Jugal Gajjar, Kaustik Ranaware  

**Link**: [PDF](https://arxiv.org/pdf/2505.06110)  

**Abstract**: This project performs multimodal sentiment analysis using the CMU-MOSEI dataset, using transformer-based models with early fusion to integrate text, audio, and visual modalities. We employ BERT-based encoders for each modality, extracting embeddings that are concatenated before classification. The model achieves strong performance, with 97.87\% 7-class accuracy and a 0.9682 F1-score on the test set, demonstrating the effectiveness of early fusion in capturing cross-modal interactions. The training utilized Adam optimization (lr=1e-4), dropout (0.3), and early stopping to ensure generalization and robustness. Results highlight the superiority of transformer architectures in modeling multimodal sentiment, with a low MAE (0.1060) indicating precise sentiment intensity prediction. Future work may compare fusion strategies or enhance interpretability. This approach utilizes multimodal learning by effectively combining linguistic, acoustic, and visual cues for sentiment analysis. 

---
# TopicVD: A Topic-Based Dataset of Video-Guided Multimodal Machine Translation for Documentaries 

**Authors**: Jinze Lv, Jian Chen, Zi Long, Xianghua Fu, Yin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05714)  

**Abstract**: Most existing multimodal machine translation (MMT) datasets are predominantly composed of static images or short video clips, lacking extensive video data across diverse domains and topics. As a result, they fail to meet the demands of real-world MMT tasks, such as documentary translation. In this study, we developed TopicVD, a topic-based dataset for video-supported multimodal machine translation of documentaries, aiming to advance research in this field. We collected video-subtitle pairs from documentaries and categorized them into eight topics, such as economy and nature, to facilitate research on domain adaptation in video-guided MMT. Additionally, we preserved their contextual information to support research on leveraging the global context of documentaries in video-guided MMT. To better capture the shared semantics between text and video, we propose an MMT model based on a cross-modal bidirectional attention module. Extensive experiments on the TopicVD dataset demonstrate that visual information consistently improves the performance of the NMT model in documentary translation. However, the MMT model's performance significantly declines in out-of-domain scenarios, highlighting the need for effective domain adaptation methods. Additionally, experiments demonstrate that global context can effectively improve translation performance. % Dataset and our implementations are available at this https URL 

---
# Multimodal Integrated Knowledge Transfer to Large Language Models through Preference Optimization with Biomedical Applications 

**Authors**: Da Wu, Zhanliang Wang, Quan Nguyen, Zhuoran Xu, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05736)  

**Abstract**: The scarcity of high-quality multimodal biomedical data limits the ability to effectively fine-tune pretrained Large Language Models (LLMs) for specialized biomedical tasks. To address this challenge, we introduce MINT (Multimodal Integrated kNowledge Transfer), a framework that aligns unimodal large decoder models with domain-specific decision patterns from multimodal biomedical data through preference optimization. While MINT supports different optimization techniques, we primarily implement it with the Odds Ratio Preference Optimization (ORPO) framework as its backbone. This strategy enables the aligned LLMs to perform predictive tasks using text-only or image-only inputs while retaining knowledge learnt from multimodal data. MINT leverages an upstream multimodal machine learning (MML) model trained on high-quality multimodal data to transfer domain-specific insights to downstream text-only or image-only LLMs. We demonstrate its effectiveness through two key applications: (1) Rare genetic disease prediction from texts, where MINT uses a multimodal encoder model, trained on facial photos and clinical notes, to generate a preference dataset for aligning a lightweight Llama 3.2-3B-Instruct. Despite relying on text input only, the MINT-derived model outperforms models trained with SFT, RAG, or DPO, and even outperforms Llama 3.1-405B-Instruct. (2) Tissue type classification using cell nucleus images, where MINT uses a vision-language foundation model as the preference generator, containing knowledge learnt from both text and histopathological images to align downstream image-only models. The resulting MINT-derived model significantly improves the performance of Llama 3.2-Vision-11B-Instruct on tissue type classification. In summary, MINT provides an effective strategy to align unimodal LLMs with high-quality multimodal expertise through preference optimization. 

---
# BMMDetect: A Multimodal Deep Learning Framework for Comprehensive Biomedical Misconduct Detection 

**Authors**: Yize Zhou, Jie Zhang, Meijie Wang, Lun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05763)  

**Abstract**: Academic misconduct detection in biomedical research remains challenging due to algorithmic narrowness in existing methods and fragmented analytical pipelines. We present BMMDetect, a multimodal deep learning framework that integrates journal metadata (SJR, institutional data), semantic embeddings (PubMedBERT), and GPT-4o-mined textual attributes (methodological statistics, data anomalies) for holistic manuscript evaluation. Key innovations include: (1) multimodal fusion of domain-specific features to reduce detection bias; (2) quantitative evaluation of feature importance, identifying journal authority metrics (e.g., SJR-index) and textual anomalies (e.g., statistical outliers) as dominant predictors; and (3) the BioMCD dataset, a large-scale benchmark with 13,160 retracted articles and 53,411 controls. BMMDetect achieves 74.33% AUC, outperforming single-modality baselines by 8.6%, and demonstrates transferability across biomedical subfields. This work advances scalable, interpretable tools for safeguarding research integrity. 

---
# MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks 

**Authors**: Wenqi Zeng, Yuqi Sun, Chenxi Ma, Weimin Tan, Bo Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06152)  

**Abstract**: Medical vision-language models (VLMs) have shown promise as clinical assistants across various medical fields. However, specialized dermatology VLM capable of delivering professional and detailed diagnostic analysis remains underdeveloped, primarily due to less specialized text descriptions in current dermatology multimodal datasets. To address this issue, we propose MM-Skin, the first large-scale multimodal dermatology dataset that encompasses 3 imaging modalities, including clinical, dermoscopic, and pathological and nearly 10k high-quality image-text pairs collected from professional textbooks. In addition, we generate over 27k diverse, instruction-following vision question answering (VQA) samples (9 times the size of current largest dermatology VQA dataset). Leveraging public datasets and MM-Skin, we developed SkinVL, a dermatology-specific VLM designed for precise and nuanced skin disease interpretation. Comprehensive benchmark evaluations of SkinVL on VQA, supervised fine-tuning (SFT) and zero-shot classification tasks across 8 datasets, reveal its exceptional performance for skin diseases in comparison to both general and medical VLM models. The introduction of MM-Skin and SkinVL offers a meaningful contribution to advancing the development of clinical dermatology VLM assistants. MM-Skin is available at this https URL 

---
# ArtRAG: Retrieval-Augmented Generation with Structured Context for Visual Art Understanding 

**Authors**: Shuai Wang, Ivona Najdenkoska, Hongyi Zhu, Stevan Rudinac, Monika Kackovic, Nachoem Wijnberg, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2505.06020)  

**Abstract**: Understanding visual art requires reasoning across multiple perspectives -- cultural, historical, and stylistic -- beyond mere object recognition. While recent multimodal large language models (MLLMs) perform well on general image captioning, they often fail to capture the nuanced interpretations that fine art demands. We propose ArtRAG, a novel, training-free framework that combines structured knowledge with retrieval-augmented generation (RAG) for multi-perspective artwork explanation. ArtRAG automatically constructs an Art Context Knowledge Graph (ACKG) from domain-specific textual sources, organizing entities such as artists, movements, themes, and historical events into a rich, interpretable graph. At inference time, a multi-granular structured retriever selects semantically and topologically relevant subgraphs to guide generation. This enables MLLMs to produce contextually grounded, culturally informed art descriptions. Experiments on the SemArt and Artpedia datasets show that ArtRAG outperforms several heavily trained baselines. Human evaluations further confirm that ArtRAG generates coherent, insightful, and culturally enriched interpretations. 

---
# Multi-Modal Molecular Representation Learning via Structure Awareness 

**Authors**: Rong Yin, Ruyue Liu, Xiaoshuai Hao, Xingrui Zhou, Yong Liu, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05877)  

**Abstract**: Accurate extraction of molecular representations is a critical step in the drug discovery process. In recent years, significant progress has been made in molecular representation learning methods, among which multi-modal molecular representation methods based on images, and 2D/3D topologies have become increasingly mainstream. However, existing these multi-modal approaches often directly fuse information from different modalities, overlooking the potential of intermodal interactions and failing to adequately capture the complex higher-order relationships and invariant features between molecules. To overcome these challenges, we propose a structure-awareness-based multi-modal self-supervised molecular representation pre-training framework (MMSA) designed to enhance molecular graph representations by leveraging invariant knowledge between molecules. The framework consists of two main modules: the multi-modal molecular representation learning module and the structure-awareness module. The multi-modal molecular representation learning module collaboratively processes information from different modalities of the same molecule to overcome intermodal differences and generate a unified molecular embedding. Subsequently, the structure-awareness module enhances the molecular representation by constructing a hypergraph structure to model higher-order correlations between molecules. This module also introduces a memory mechanism for storing typical molecular representations, aligning them with memory anchors in the memory bank to integrate invariant knowledge, thereby improving the model generalization ability. Extensive experiments have demonstrated the effectiveness of MMSA, which achieves state-of-the-art performance on the MoleculeNet benchmark, with average ROC-AUC improvements ranging from 1.8% to 9.6% over baseline methods. 

---
# Leveraging Vision-Language Models for Visual Grounding and Analysis of Automotive UI 

**Authors**: Benjamin Raphael Ernhofer, Daniil Prokhorov, Jannica Langner, Dominik Bollmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.05895)  

**Abstract**: Modern automotive infotainment systems require intelligent and adaptive solutions to handle frequent User Interface (UI) updates and diverse design variations. We introduce a vision-language framework for understanding and interacting with automotive infotainment systems, enabling seamless adaptation across different UI designs. To further support research in this field, we release AutomotiveUI-Bench-4K, an open-source dataset of 998 images with 4,208 annotations. Additionally, we present a synthetic data pipeline to generate training data. We fine-tune a Molmo-7B-based model using Low-Rank Adaptation (LoRa) and incorporating reasoning generated by our pipeline, along with visual grounding and evaluation capabilities. The fine-tuned Evaluative Large Action Model (ELAM) achieves strong performance on AutomotiveUI-Bench-4K (model and dataset are available on Hugging Face) and demonstrating strong cross-domain generalization, including a +5.2% improvement on ScreenSpot over the baseline model. Notably, our approach achieves 80.4% average accuracy on ScreenSpot, closely matching or even surpassing specialized models for desktop, mobile, and web, such as ShowUI, despite being trained for the infotainment domain. This research investigates how data collection and subsequent fine-tuning can lead to AI-driven progress within automotive UI understanding and interaction. The applied method is cost-efficient and fine-tuned models can be deployed on consumer-grade GPUs. 

---
# Looking Beyond Language Priors: Enhancing Visual Comprehension and Attention in Multimodal Models 

**Authors**: Aarti Ghatkesar, Uddeshya Upadhyay, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2505.05626)  

**Abstract**: Achieving deep alignment between vision and language remains a central challenge for Multimodal Large Language Models (MLLMs). These models often fail to fully leverage visual input, defaulting to strong language priors. Our approach first provides insights into how MLLMs internally build visual understanding of image regions and then introduces techniques to amplify this capability. Specifically, we explore techniques designed both to deepen the model's understanding of visual content and to ensure that these visual insights actively guide language generation. We demonstrate the superior multimodal understanding of our resultant model through a detailed upstream analysis quantifying its ability to predict visually-dependent tokens as well as 10 pt boost on visually challenging tasks. 

---
# PyTDC: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models 

**Authors**: Alejandro Velez-Arce, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.05577)  

**Abstract**: Existing biomedical benchmarks do not provide end-to-end infrastructure for training, evaluation, and inference of models that integrate multimodal biological data and a broad range of machine learning tasks in therapeutics. We present PyTDC, an open-source machine-learning platform providing streamlined training, evaluation, and inference software for multimodal biological AI models. PyTDC unifies distributed, heterogeneous, continuously updated data sources and model weights and standardizes benchmarking and inference endpoints. This paper discusses the components of PyTDC's architecture and, to our knowledge, the first-of-its-kind case study on the introduced single-cell drug-target nomination ML task. We find state-of-the-art methods in graph representation learning and domain-specific methods from graph theory perform poorly on this task. Though we find a context-aware geometric deep learning method that outperforms the evaluated SoTA and domain-specific baseline methods, the model is unable to generalize to unseen cell types or incorporate additional modalities, highlighting PyTDC's capacity to facilitate an exciting avenue of research developing multimodal, context-aware, foundation models for open problems in biomedical AI. 

---
# Preliminary Explorations with GPT-4o(mni) Native Image Generation 

**Authors**: Pu Cao, Feng Zhou, Junyi Ji, Qingye Kong, Zhixiang Lv, Mingjian Zhang, Xuekun Zhao, Siqi Wu, Yinghui Lin, Qing Song, Lu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05501)  

**Abstract**: Recently, the visual generation ability by GPT-4o(mni) has been unlocked by OpenAI. It demonstrates a very remarkable generation capability with excellent multimodal condition understanding and varied task instructions. In this paper, we aim to explore the capabilities of GPT-4o across various tasks. Inspired by previous study, we constructed a task taxonomy along with a carefully curated set of test samples to conduct a comprehensive qualitative test. Benefiting from GPT-4o's powerful multimodal comprehension, its image-generation process demonstrates abilities surpassing those of traditional image-generation tasks. Thus, regarding the dimensions of model capabilities, we evaluate its performance across six task categories: traditional image generation tasks, discriminative tasks, knowledge-based generation, commonsense-based generation, spatially-aware image generation, and temporally-aware image generation. These tasks not only assess the quality and conditional alignment of the model's outputs but also probe deeper into GPT-4o's understanding of real-world concepts. Our results reveal that GPT-4o performs impressively well in general-purpose synthesis tasks, showing strong capabilities in text-to-image generation, visual stylization, and low-level image processing. However, significant limitations remain in its ability to perform precise spatial reasoning, instruction-grounded generation, and consistent temporal prediction. Furthermore, when faced with knowledge-intensive or domain-specific scenarios, such as scientific illustrations or mathematical plots, the model often exhibits hallucinations, factual errors, or structural inconsistencies. These findings suggest that while GPT-4o marks a substantial advancement in unified multimodal generation, there is still a long way to go before it can be reliably applied to professional or safety-critical domains. 

---
# AI-powered virtual eye: perspective, challenges and opportunities 

**Authors**: Yue Wu, Yibo Guo, Yulong Yan, Jiancheng Yang, Xin Zhou, Ching-Yu Cheng, Danli Shi, Mingguang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05516)  

**Abstract**: We envision the "virtual eye" as a next-generation, AI-powered platform that uses interconnected foundation models to simulate the eye's intricate structure and biological function across all scales. Advances in AI, imaging, and multiomics provide a fertile ground for constructing a universal, high-fidelity digital replica of the human eye. This perspective traces the evolution from early mechanistic and rule-based models to contemporary AI-driven approaches, integrating in a unified model with multimodal, multiscale, dynamic predictive capabilities and embedded feedback mechanisms. We propose a development roadmap emphasizing the roles of large-scale multimodal datasets, generative AI, foundation models, agent-based architectures, and interactive interfaces. Despite challenges in interpretability, ethics, data processing and evaluation, the virtual eye holds the potential to revolutionize personalized ophthalmic care and accelerate research into ocular health and disease. 

---
