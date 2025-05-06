# LISAT: Language-Instructed Segmentation Assistant for Satellite Imagery 

**Authors**: Jerome Quenum, Wen-Han Hsieh, Tsung-Han Wu, Ritwik Gupta, Trevor Darrell, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2505.02829)  

**Abstract**: Segmentation models can recognize a pre-defined set of objects in images. However, models that can reason over complex user queries that implicitly refer to multiple objects of interest are still in their infancy. Recent advances in reasoning segmentation--generating segmentation masks from complex, implicit query text--demonstrate that vision-language models can operate across an open domain and produce reasonable outputs. However, our experiments show that such models struggle with complex remote-sensing imagery. In this work, we introduce LISAt, a vision-language model designed to describe complex remote-sensing scenes, answer questions about them, and segment objects of interest. We trained LISAt on a new curated geospatial reasoning-segmentation dataset, GRES, with 27,615 annotations over 9,205 images, and a multimodal pretraining dataset, PreGRES, containing over 1 million question-answer pairs. LISAt outperforms existing geospatial foundation models such as RS-GPT4V by over 10.04 % (BLEU-4) on remote-sensing description tasks, and surpasses state-of-the-art open-domain models on reasoning segmentation tasks by 143.36 % (gIoU). Our model, datasets, and code are available at this https URL 

---
# MSFNet-CPD: Multi-Scale Cross-Modal Fusion Network for Crop Pest Detection 

**Authors**: Jiaqi Zhang, Zhuodong Liu, Kejian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02441)  

**Abstract**: Accurate identification of agricultural pests is essential for crop protection but remains challenging due to the large intra-class variance and fine-grained differences among pest species. While deep learning has advanced pest detection, most existing approaches rely solely on low-level visual features and lack effective multi-modal integration, leading to limited accuracy and poor interpretability. Moreover, the scarcity of high-quality multi-modal agricultural datasets further restricts progress in this field. To address these issues, we construct two novel multi-modal benchmarks-CTIP102 and STIP102-based on the widely-used IP102 dataset, and introduce a Multi-scale Cross-Modal Fusion Network (MSFNet-CPD) for robust pest detection. Our approach enhances visual quality via a super-resolution reconstruction module, and feeds both the original and reconstructed images into the network to improve clarity and detection performance. To better exploit semantic cues, we propose an Image-Text Fusion (ITF) module for joint modeling of visual and textual features, and an Image-Text Converter (ITC) that reconstructs fine-grained details across multiple scales to handle challenging backgrounds. Furthermore, we introduce an Arbitrary Combination Image Enhancement (ACIE) strategy to generate a more complex and diverse pest detection dataset, MTIP102, improving the model's generalization to real-world scenarios. Extensive experiments demonstrate that MSFNet-CPD consistently outperforms state-of-the-art methods on multiple pest detection benchmarks. All code and datasets will be made publicly available at: this https URL. 

---
# Task-Oriented Semantic Communication in Large Multimodal Models-based Vehicle Networks 

**Authors**: Baoxia Du, Hongyang Du, Dusit Niyato, Ruidong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02413)  

**Abstract**: Task-oriented semantic communication has emerged as a fundamental approach for enhancing performance in various communication scenarios. While recent advances in Generative Artificial Intelligence (GenAI), such as Large Language Models (LLMs), have been applied to semantic communication designs, the potential of Large Multimodal Models (LMMs) remains largely unexplored. In this paper, we investigate an LMM-based vehicle AI assistant using a Large Language and Vision Assistant (LLaVA) and propose a task-oriented semantic communication framework to facilitate efficient interaction between users and cloud servers. To reduce computational demands and shorten response time, we optimize LLaVA's image slicing to selectively focus on areas of utmost interest to users. Additionally, we assess the importance of image patches by combining objective and subjective user attention, adjusting energy usage for transmitting semantic information. This strategy optimizes resource utilization, ensuring precise transmission of critical information. We construct a Visual Question Answering (VQA) dataset for traffic scenarios to evaluate effectiveness. Experimental results show that our semantic communication framework significantly increases accuracy in answering questions under the same channel conditions, performing particularly well in environments with poor Signal-to-Noise Ratios (SNR). Accuracy can be improved by 13.4% at an SNR of 12dB and 33.1% at 10dB, respectively. 

---
# Retrieval-augmented in-context learning for multimodal large language models in disease classification 

**Authors**: Zaifu Zhan, Shuang Zhou, Xiaoshan Zhou, Yongkang Xiao, Jun Wang, Jiawen Deng, He Zhu, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02087)  

**Abstract**: Objectives: We aim to dynamically retrieve informative demonstrations, enhancing in-context learning in multimodal large language models (MLLMs) for disease classification.
Methods: We propose a Retrieval-Augmented In-Context Learning (RAICL) framework, which integrates retrieval-augmented generation (RAG) and in-context learning (ICL) to adaptively select demonstrations with similar disease patterns, enabling more effective ICL in MLLMs. Specifically, RAICL examines embeddings from diverse encoders, including ResNet, BERT, BioBERT, and ClinicalBERT, to retrieve appropriate demonstrations, and constructs conversational prompts optimized for ICL. We evaluated the framework on two real-world multi-modal datasets (TCGA and IU Chest X-ray), assessing its performance across multiple MLLMs (Qwen, Llava, Gemma), embedding strategies, similarity metrics, and varying numbers of demonstrations.
Results: RAICL consistently improved classification performance. Accuracy increased from 0.7854 to 0.8368 on TCGA and from 0.7924 to 0.8658 on IU Chest X-ray. Multi-modal inputs outperformed single-modal ones, with text-only inputs being stronger than images alone. The richness of information embedded in each modality will determine which embedding model can be used to get better results. Few-shot experiments showed that increasing the number of retrieved examples further enhanced performance. Across different similarity metrics, Euclidean distance achieved the highest accuracy while cosine similarity yielded better macro-F1 scores. RAICL demonstrated consistent improvements across various MLLMs, confirming its robustness and versatility.
Conclusions: RAICL provides an efficient and scalable approach to enhance in-context learning in MLLMs for multimodal disease classification. 

---
# Beyond the Monitor: Mixed Reality Visualization and AI for Enhanced Digital Pathology Workflow 

**Authors**: Jai Prakash Veerla, Partha Sai Guttikonda, Helen H. Shang, Mohammad Sadegh Nasr, Cesar Torres, Jacob M. Luber  

**Link**: [PDF](https://arxiv.org/pdf/2505.02780)  

**Abstract**: Pathologists rely on gigapixel whole-slide images (WSIs) to diagnose diseases like cancer, yet current digital pathology tools hinder diagnosis. The immense scale of WSIs, often exceeding 100,000 X 100,000 pixels, clashes with the limited views traditional monitors offer. This mismatch forces constant panning and zooming, increasing pathologist cognitive load, causing diagnostic fatigue, and slowing pathologists' adoption of digital methods. PathVis, our mixed-reality visualization platform for Apple Vision Pro, addresses these challenges. It transforms the pathologist's interaction with data, replacing cumbersome mouse-and-monitor navigation with intuitive exploration using natural hand gestures, eye gaze, and voice commands in an immersive workspace. PathVis integrates AI to enhance diagnosis. An AI-driven search function instantly retrieves and displays the top five similar patient cases side-by-side, improving diagnostic precision and efficiency through rapid comparison. Additionally, a multimodal conversational AI assistant offers real-time image interpretation support and aids collaboration among pathologists across multiple Apple devices. By merging the directness of traditional pathology with advanced mixed-reality visualization and AI, PathVis improves diagnostic workflows, reduces cognitive strain, and makes pathology practice more effective and engaging. The PathVis source code and a demo video are publicly available at: this https URL 

---
# Timing Is Everything: Finding the Optimal Fusion Points in Multimodal Medical Imaging 

**Authors**: Valerio Guarrasi, Klara Mogensen, Sara Tassinari, Sara Qvarlander, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2505.02467)  

**Abstract**: Multimodal deep learning harnesses diverse imaging modalities, such as MRI sequences, to enhance diagnostic accuracy in medical imaging. A key challenge is determining the optimal timing for integrating these modalities-specifically, identifying the network layers where fusion modules should be inserted. Current approaches often rely on manual tuning or exhaustive search, which are computationally expensive without any guarantee of converging to optimal results. We propose a sequential forward search algorithm that incrementally activates and evaluates candidate fusion modules at different layers of a multimodal network. At each step, the algorithm retrains from previously learned weights and compares validation loss to identify the best-performing configuration. This process systematically reduces the search space, enabling efficient identification of the optimal fusion timing without exhaustively testing all possible module placements. The approach is validated on two multimodal MRI datasets, each addressing different classification tasks. Our algorithm consistently identified configurations that outperformed unimodal baselines, late fusion, and a brute-force ensemble of all potential fusion placements. These architectures demonstrated superior accuracy, F-score, and specificity while maintaining competitive or improved AUC values. Furthermore, the sequential nature of the search significantly reduced computational overhead, making the optimization process more practical. By systematically determining the optimal timing to fuse imaging modalities, our method advances multimodal deep learning for medical imaging. It provides an efficient and robust framework for fusion optimization, paving the way for improved clinical decision-making and more adaptable, scalable architectures in medical AI applications. 

---
# MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans 

**Authors**: Huangyue Yu, Baoxiong Jia, Yixin Chen, Yandan Yang, Puhao Li, Rongpeng Su, Jiaxin Li, Qing Li, Wei Liang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02388)  

**Abstract**: Embodied AI (EAI) research requires high-quality, diverse 3D scenes to effectively support skill acquisition, sim-to-real transfer, and generalization. Achieving these quality standards, however, necessitates the precise replication of real-world object diversity. Existing datasets demonstrate that this process heavily relies on artist-driven designs, which demand substantial human effort and present significant scalability challenges. To scalably produce realistic and interactive 3D scenes, we first present MetaScenes, a large-scale, simulatable 3D scene dataset constructed from real-world scans, which includes 15366 objects spanning 831 fine-grained categories. Then, we introduce Scan2Sim, a robust multi-modal alignment model, which enables the automated, high-quality replacement of assets, thereby eliminating the reliance on artist-driven designs for scaling 3D scenes. We further propose two benchmarks to evaluate MetaScenes: a detailed scene synthesis task focused on small item layouts for robotic manipulation and a domain transfer task in vision-and-language navigation (VLN) to validate cross-domain transfer. Results confirm MetaScene's potential to enhance EAI by supporting more generalizable agent learning and sim-to-real applications, introducing new possibilities for EAI research. Project website: this https URL. 

---
# Benchmarking Feature Upsampling Methods for Vision Foundation Models using Interactive Segmentation 

**Authors**: Volodymyr Havrylov, Haiwen Huang, Dan Zhang, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2505.02075)  

**Abstract**: Vision Foundation Models (VFMs) are large-scale, pre-trained models that serve as general-purpose backbones for various computer vision tasks. As VFMs' popularity grows, there is an increasing interest in understanding their effectiveness for dense prediction tasks. However, VFMs typically produce low-resolution features, limiting their direct applicability in this context. One way to tackle this limitation is by employing a task-agnostic feature upsampling module that refines VFM features resolution. To assess the effectiveness of this approach, we investigate Interactive Segmentation (IS) as a novel benchmark for evaluating feature upsampling methods on VFMs. Due to its inherent multimodal input, consisting of an image and a set of user-defined clicks, as well as its dense mask output, IS creates a challenging environment that demands comprehensive visual scene understanding. Our benchmarking experiments show that selecting appropriate upsampling strategies significantly improves VFM features quality. The code is released at this https URL 

---
# LecEval: An Automated Metric for Multimodal Knowledge Acquisition in Multimedia Learning 

**Authors**: Joy Lim Jia Yin, Daniel Zhang-Li, Jifan Yu, Haoxuan Li, Shangqing Tu, Yuanchun Wang, Zhiyuan Liu, Huiqin Liu, Lei Hou, Juanzi Li, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02078)  

**Abstract**: Evaluating the quality of slide-based multimedia instruction is challenging. Existing methods like manual assessment, reference-based metrics, and large language model evaluators face limitations in scalability, context capture, or bias. In this paper, we introduce LecEval, an automated metric grounded in Mayer's Cognitive Theory of Multimedia Learning, to evaluate multimodal knowledge acquisition in slide-based learning. LecEval assesses effectiveness using four rubrics: Content Relevance (CR), Expressive Clarity (EC), Logical Structure (LS), and Audience Engagement (AE). We curate a large-scale dataset of over 2,000 slides from more than 50 online course videos, annotated with fine-grained human ratings across these rubrics. A model trained on this dataset demonstrates superior accuracy and adaptability compared to existing metrics, bridging the gap between automated and human assessments. We release our dataset and toolkits at this https URL. 

---
# Segment Any RGB-Thermal Model with Language-aided Distillation 

**Authors**: Dong Xing, Xianxun Zhu, Wei Zhou, Qika Lin, Hang Yang, Yuqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01950)  

**Abstract**: The recent Segment Anything Model (SAM) demonstrates strong instance segmentation performance across various downstream tasks. However, SAM is trained solely on RGB data, limiting its direct applicability to RGB-thermal (RGB-T) semantic segmentation. Given that RGB-T provides a robust solution for scene understanding in adverse weather and lighting conditions, such as low light and overexposure, we propose a novel framework, SARTM, which customizes the powerful SAM for RGB-T semantic segmentation. Our key idea is to unleash the potential of SAM while introduce semantic understanding modules for RGB-T data pairs. Specifically, our framework first involves fine tuning the original SAM by adding extra LoRA layers, aiming at preserving SAM's strong generalization and segmentation capabilities for downstream tasks. Secondly, we introduce language information as guidance for training our SARTM. To address cross-modal inconsistencies, we introduce a Cross-Modal Knowledge Distillation(CMKD) module that effectively achieves modality adaptation while maintaining its generalization capabilities. This semantic module enables the minimization of modality gaps and alleviates semantic ambiguity, facilitating the combination of any modality under any visual conditions. Furthermore, we enhance the segmentation performance by adjusting the segmentation head of SAM and incorporating an auxiliary semantic segmentation head, which integrates multi-scale features for effective fusion. Extensive experiments are conducted across three multi-modal RGBT semantic segmentation benchmarks: MFNET, PST900, and FMB. Both quantitative and qualitative results consistently demonstrate that the proposed SARTM significantly outperforms state-of-the-art approaches across a variety of conditions. 

---
# PhysNav-DG: A Novel Adaptive Framework for Robust VLM-Sensor Fusion in Navigation Applications 

**Authors**: Trisanth Srinivasan, Santosh Patapati  

**Link**: [PDF](https://arxiv.org/pdf/2505.01881)  

**Abstract**: Robust navigation in diverse environments and domains requires both accurate state estimation and transparent decision making. We present PhysNav-DG, a novel framework that integrates classical sensor fusion with the semantic power of vision-language models. Our dual-branch architecture predicts navigation actions from multi-sensor inputs while simultaneously generating detailed chain-of-thought explanations. A modified Adaptive Kalman Filter dynamically adjusts its noise parameters based on environmental context. It leverages several streams of raw sensor data along with semantic insights from models such as LLaMA 3.2 11B and BLIP-2. To evaluate our approach, we introduce the MD-NEX Benchmark, a novel multi-domain dataset that unifies indoor navigation, autonomous driving, and social navigation tasks with ground-truth actions and human-validated explanations. Extensive experiments and ablations show that PhysNav-DG improves navigation success rates by over 20% and achieves high efficiency, with explanations that are both highly grounded and clear. This work connects high-level semantic reasoning and geometric planning for safer and more trustworthy autonomous systems. 

---
# A Multimodal Framework for Explainable Evaluation of Soft Skills in Educational Environments 

**Authors**: Jared D.T. Guerrero-Sosa, Francisco P. Romero, Víctor Hugo Menéndez-Domínguez, Jesus Serrano-Guerrero, Andres Montoro-Montarroso, Jose A. Olivas  

**Link**: [PDF](https://arxiv.org/pdf/2505.01794)  

**Abstract**: In the rapidly evolving educational landscape, the unbiased assessment of soft skills is a significant challenge, particularly in higher education. This paper presents a fuzzy logic approach that employs a Granular Linguistic Model of Phenomena integrated with multimodal analysis to evaluate soft skills in undergraduate students. By leveraging computational perceptions, this approach enables a structured breakdown of complex soft skill expressions, capturing nuanced behaviours with high granularity and addressing their inherent uncertainties, thereby enhancing interpretability and reliability. Experiments were conducted with undergraduate students using a developed tool that assesses soft skills such as decision-making, communication, and creativity. This tool identifies and quantifies subtle aspects of human interaction, such as facial expressions and gesture recognition. The findings reveal that the framework effectively consolidates multiple data inputs to produce meaningful and consistent assessments of soft skills, showing that integrating multiple modalities into the evaluation process significantly improves the quality of soft skills scores, making the assessment work transparent and understandable to educational stakeholders. 

---
# PhytoSynth: Leveraging Multi-modal Generative Models for Crop Disease Data Generation with Novel Benchmarking and Prompt Engineering Approach 

**Authors**: Nitin Rai, Arnold W. Schumann, Nathan Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2505.01823)  

**Abstract**: Collecting large-scale crop disease images in the field is labor-intensive and time-consuming. Generative models (GMs) offer an alternative by creating synthetic samples that resemble real-world images. However, existing research primarily relies on Generative Adversarial Networks (GANs)-based image-to-image translation and lack a comprehensive analysis of computational requirements in agriculture. Therefore, this research explores a multi-modal text-to-image approach for generating synthetic crop disease images and is the first to provide computational benchmarking in this context. We trained three Stable Diffusion (SD) variants-SDXL, SD3.5M (medium), and SD3.5L (large)-and fine-tuned them using Dreambooth and Low-Rank Adaptation (LoRA) fine-tuning techniques to enhance generalization. SD3.5M outperformed the others, with an average memory usage of 18 GB, power consumption of 180 W, and total energy use of 1.02 kWh/500 images (0.002 kWh per image) during inference task. Our results demonstrate SD3.5M's ability to generate 500 synthetic images from just 36 in-field samples in 1.5 hours. We recommend SD3.5M for efficient crop disease data generation. 

---
# Interpretable graph-based models on multimodal biomedical data integration: A technical review and benchmarking 

**Authors**: Alireza Sadeghi, Farshid Hajati, Ahmadreza Argha, Nigel H Lovell, Min Yang, Hamid Alinejad-Rokny  

**Link**: [PDF](https://arxiv.org/pdf/2505.01696)  

**Abstract**: Integrating heterogeneous biomedical data including imaging, omics, and clinical records supports accurate diagnosis and personalised care. Graph-based models fuse such non-Euclidean data by capturing spatial and relational structure, yet clinical uptake requires regulator-ready interpretability. We present the first technical survey of interpretable graph based models for multimodal biomedical data, covering 26 studies published between Jan 2019 and Sep 2024. Most target disease classification, notably cancer and rely on static graphs from simple similarity measures, while graph-native explainers are rare; post-hoc methods adapted from non-graph domains such as gradient saliency, and SHAP predominate. We group existing approaches into four interpretability families, outline trends such as graph-in-graph hierarchies, knowledge-graph edges, and dynamic topology learning, and perform a practical benchmark. Using an Alzheimer disease cohort, we compare Sensitivity Analysis, Gradient Saliency, SHAP and Graph Masking. SHAP and Sensitivity Analysis recover the broadest set of known AD pathways and Gene-Ontology terms, whereas Gradient Saliency and Graph Masking surface complementary metabolic and transport signatures. Permutation tests show all four beat random gene sets, but with distinct trade-offs: SHAP and Graph Masking offer deeper biology at higher compute cost, while Gradient Saliency and Sensitivity Analysis are quicker though coarser. We also provide a step-by-step flowchart covering graph construction, explainer choice and resource budgeting to help researchers balance transparency and performance. This review synthesises the state of interpretable graph learning for multimodal medicine, benchmarks leading techniques, and charts future directions, from advanced XAI tools to under-studied diseases, serving as a concise reference for method developers and translational scientists. 

---
# RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation 

**Authors**: Kaidong Zhang, Rongtao Xu, Pengzhen Ren, Junfan Lin, Hefeng Wu, Liang Lin, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01709)  

**Abstract**: Operating robots in open-ended scenarios with diverse tasks is a crucial research and application direction in robotics. While recent progress in natural language processing and large multimodal models has enhanced robots' ability to understand complex instructions, robot manipulation still faces the procedural skill dilemma and the declarative skill dilemma in open environments. Existing methods often compromise cognitive and executive capabilities. To address these challenges, in this paper, we propose RoBridge, a hierarchical intelligent architecture for general robotic manipulation. It consists of a high-level cognitive planner (HCP) based on a large-scale pre-trained vision-language model (VLM), an invariant operable representation (IOR) serving as a symbolic bridge, and a generalist embodied agent (GEA). RoBridge maintains the declarative skill of VLM and unleashes the procedural skill of reinforcement learning, effectively bridging the gap between cognition and execution. RoBridge demonstrates significant performance improvements over existing baselines, achieving a 75% success rate on new tasks and an 83% average success rate in sim-to-real generalization using only five real-world data samples per task. This work represents a significant step towards integrating cognitive reasoning with physical execution in robotic systems, offering a new paradigm for general robotic manipulation. 

---
# Automated ARAT Scoring Using Multimodal Video Analysis, Multi-View Fusion, and Hierarchical Bayesian Models: A Clinician Study 

**Authors**: Tamim Ahmed, Thanassis Rikakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.01680)  

**Abstract**: Manual scoring of the Action Research Arm Test (ARAT) for upper extremity assessment in stroke rehabilitation is time-intensive and variable. We propose an automated ARAT scoring system integrating multimodal video analysis with SlowFast, I3D, and Transformer-based models using OpenPose keypoints and object locations. Our approach employs multi-view data (ipsilateral, contralateral, and top perspectives), applying early and late fusion to combine features across views and models. Hierarchical Bayesian Models (HBMs) infer movement quality components, enhancing interpretability. A clinician dashboard displays task scores, execution times, and quality assessments. We conducted a study with five clinicians who reviewed 500 video ratings generated by our system, providing feedback on its accuracy and usability. Evaluated on a stroke rehabilitation dataset, our framework achieves 89.0% validation accuracy with late fusion, with HBMs aligning closely with manual assessments. This work advances automated rehabilitation by offering a scalable, interpretable solution with clinical validation. 

---
# Seeing Heat with Color -- RGB-Only Wildfire Temperature Inference from SAM-Guided Multimodal Distillation using Radiometric Ground Truth 

**Authors**: Michael Marinaccio, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2505.01638)  

**Abstract**: High-fidelity wildfire monitoring using Unmanned Aerial Vehicles (UAVs) typically requires multimodal sensing - especially RGB and thermal imagery - which increases hardware cost and power consumption. This paper introduces SAM-TIFF, a novel teacher-student distillation framework for pixel-level wildfire temperature prediction and segmentation using RGB input only. A multimodal teacher network trained on paired RGB-Thermal imagery and radiometric TIFF ground truth distills knowledge to a unimodal RGB student network, enabling thermal-sensor-free inference. Segmentation supervision is generated using a hybrid approach of segment anything (SAM)-guided mask generation, and selection via TOPSIS, along with Canny edge detection and Otsu's thresholding pipeline for automatic point prompt selection. Our method is the first to perform per-pixel temperature regression from RGB UAV data, demonstrating strong generalization on the recent FLAME 3 dataset. This work lays the foundation for lightweight, cost-effective UAV-based wildfire monitoring systems without thermal sensors. 

---
# Multimodal and Multiview Deep Fusion for Autonomous Marine Navigation 

**Authors**: Dimitrios Dagdilelis, Panagiotis Grigoriadis, Roberto Galeazzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01615)  

**Abstract**: We propose a cross attention transformer based method for multimodal sensor fusion to build a birds eye view of a vessels surroundings supporting safer autonomous marine navigation. The model deeply fuses multiview RGB and long wave infrared images with sparse LiDAR point clouds. Training also integrates X band radar and electronic chart data to inform predictions. The resulting view provides a detailed reliable scene representation improving navigational accuracy and robustness. Real world sea trials confirm the methods effectiveness even in adverse weather and complex maritime settings. 

---
# Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation 

**Authors**: Vaidehi Patil, Yi-Lin Sung, Peter Hase, Jie Peng, Tianlong Chen, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2505.01456)  

**Abstract**: LLMs trained on massive datasets may inadvertently acquire sensitive information such as personal details and potentially harmful content. This risk is further heightened in multimodal LLMs as they integrate information from multiple modalities (image and text). Adversaries can exploit this knowledge through multimodal prompts to extract sensitive details. Evaluating how effectively MLLMs can forget such information (targeted unlearning) necessitates the creation of high-quality, well-annotated image-text pairs. While prior work on unlearning has focused on text, multimodal unlearning remains underexplored. To address this gap, we first introduce a multimodal unlearning benchmark, UnLOK-VQA (Unlearning Outside Knowledge VQA), as well as an attack-and-defense framework to evaluate methods for deleting specific multimodal knowledge from MLLMs. We extend a visual question-answering dataset using an automated pipeline that generates varying-proximity samples for testing generalization and specificity, followed by manual filtering for maintaining high quality. We then evaluate six defense objectives against seven attacks (four whitebox, three blackbox), including a novel whitebox method leveraging interpretability of hidden states. Our results show multimodal attacks outperform text- or image-only ones, and that the most effective defense removes answer information from internal model states. Additionally, larger models exhibit greater post-editing robustness, suggesting that scale enhances safety. UnLOK-VQA provides a rigorous benchmark for advancing unlearning in MLLMs. 

---
# Emotions in the Loop: A Survey of Affective Computing for Emotional Support 

**Authors**: Karishma Hegde, Hemadri Jayalath  

**Link**: [PDF](https://arxiv.org/pdf/2505.01542)  

**Abstract**: In a world where technology is increasingly embedded in our everyday experiences, systems that sense and respond to human emotions are elevating digital interaction. At the intersection of artificial intelligence and human-computer interaction, affective computing is emerging with innovative solutions where machines are humanized by enabling them to process and respond to user emotions. This survey paper explores recent research contributions in affective computing applications in the area of emotion recognition, sentiment analysis and personality assignment developed using approaches like large language models (LLMs), multimodal techniques, and personalized AI systems. We analyze the key contributions and innovative methodologies applied by the selected research papers by categorizing them into four domains: AI chatbot applications, multimodal input systems, mental health and therapy applications, and affective computing for safety applications. We then highlight the technological strengths as well as the research gaps and challenges related to these studies. Furthermore, the paper examines the datasets used in each study, highlighting how modality, scale, and diversity impact the development and performance of affective models. Finally, the survey outlines ethical considerations and proposes future directions to develop applications that are more safe, empathetic and practical. 

---
# R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning 

**Authors**: Yi-Fan Zhang, Xingyu Lu, Xiao Hu, Chaoyou Fu, Bin Wen, Tianke Zhang, Changyi Liu, Kaiyu Jiang, Kaibing Chen, Kaiyu Tang, Haojie Ding, Jiankang Chen, Fan Yang, Zhang Zhang, Tingting Gao, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02835)  

**Abstract**: Multimodal Reward Models (MRMs) play a crucial role in enhancing the performance of Multimodal Large Language Models (MLLMs). While recent advancements have primarily focused on improving the model structure and training data of MRMs, there has been limited exploration into the effectiveness of long-term reasoning capabilities for reward modeling and how to activate these capabilities in MRMs. In this paper, we explore how Reinforcement Learning (RL) can be used to improve reward modeling. Specifically, we reformulate the reward modeling problem as a rule-based RL task. However, we observe that directly applying existing RL algorithms, such as Reinforce++, to reward modeling often leads to training instability or even collapse due to the inherent limitations of these algorithms. To address this issue, we propose the StableReinforce algorithm, which refines the training loss, advantage estimation strategy, and reward design of existing RL methods. These refinements result in more stable training dynamics and superior performance. To facilitate MRM training, we collect 200K preference data from diverse datasets. Our reward model, R1-Reward, trained using the StableReinforce algorithm on this dataset, significantly improves performance on multimodal reward modeling benchmarks. Compared to previous SOTA models, R1-Reward achieves a $8.4\%$ improvement on the VL Reward-Bench and a $14.3\%$ improvement on the Multimodal Reward Bench. Moreover, with more inference compute, R1-Reward's performance is further enhanced, highlighting the potential of RL algorithms in optimizing MRMs. 

---
# AOR: Anatomical Ontology-Guided Reasoning for Medical Large Multimodal Model in Chest X-Ray Interpretation 

**Authors**: Qingqiu Li, Zihang Cui, Seongsu Bae, Jilan Xu, Runtian Yuan, Yuejie Zhang, Rui Feng, Quanli Shen, Xiaobo Zhang, Junjun He, Shujun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02830)  

**Abstract**: Chest X-rays (CXRs) are the most frequently performed imaging examinations in clinical settings. Recent advancements in Large Multimodal Models (LMMs) have enabled automated CXR interpretation, enhancing diagnostic accuracy and efficiency. However, despite their strong visual understanding, current Medical LMMs (MLMMs) still face two major challenges: (1) Insufficient region-level understanding and interaction, and (2) Limited accuracy and interpretability due to single-step reasoning. In this paper, we empower MLMMs with anatomy-centric reasoning capabilities to enhance their interactivity and explainability. Specifically, we first propose an Anatomical Ontology-Guided Reasoning (AOR) framework, which centers on cross-modal region-level information to facilitate multi-step reasoning. Next, under the guidance of expert physicians, we develop AOR-Instruction, a large instruction dataset for MLMMs training. Our experiments demonstrate AOR's superior performance in both VQA and report generation tasks. 

---
# A Comprehensive Analysis for Visual Object Hallucination in Large Vision-Language Models 

**Authors**: Liqiang Jing, Guiming Hardy Chen, Ehsan Aghazadeh, Xin Eric Wang, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.01958)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in multimodal tasks, but visual object hallucination remains a persistent issue. It refers to scenarios where models generate inaccurate visual object-related information based on the query input, potentially leading to misinformation and concerns about safety and reliability. Previous works focus on the evaluation and mitigation of visual hallucinations, but the underlying causes have not been comprehensively investigated. In this paper, we analyze each component of LLaVA-like LVLMs -- the large language model, the vision backbone, and the projector -- to identify potential sources of error and their impact. Based on our observations, we propose methods to mitigate hallucination for each problematic component. Additionally, we developed two hallucination benchmarks: QA-VisualGenome, which emphasizes attribute and relation hallucinations, and QA-FB15k, which focuses on cognition-based hallucinations. 

---
# Enhancing the Learning Experience: Using Vision-Language Models to Generate Questions for Educational Videos 

**Authors**: Markos Stamatakis, Joshua Berger, Christian Wartena, Ralph Ewerth, Anett Hoppe  

**Link**: [PDF](https://arxiv.org/pdf/2505.01790)  

**Abstract**: Web-based educational videos offer flexible learning opportunities and are becoming increasingly popular. However, improving user engagement and knowledge retention remains a challenge. Automatically generated questions can activate learners and support their knowledge acquisition. Further, they can help teachers and learners assess their understanding. While large language and vision-language models have been employed in various tasks, their application to question generation for educational videos remains underexplored. In this paper, we investigate the capabilities of current vision-language models for generating learning-oriented questions for educational video content. We assess (1) out-of-the-box models' performance; (2) fine-tuning effects on content-specific question generation; (3) the impact of different video modalities on question quality; and (4) in a qualitative study, question relevance, answerability, and difficulty levels of generated questions. Our findings delineate the capabilities of current vision-language models, highlighting the need for fine-tuning and addressing challenges in question diversity and relevance. We identify requirements for future multimodal datasets and outline promising research directions. 

---
# Tevatron 2.0: Unified Document Retrieval Toolkit across Scale, Language, and Modality 

**Authors**: Xueguang Ma, Luyu Gao, Shengyao Zhuang, Jiaqi Samantha Zhan, Jamie Callan, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.02466)  

**Abstract**: Recent advancements in large language models (LLMs) have driven interest in billion-scale retrieval models with strong generalization across retrieval tasks and languages. Additionally, progress in large vision-language models has created new opportunities for multimodal retrieval. In response, we have updated the Tevatron toolkit, introducing a unified pipeline that enables researchers to explore retriever models at different scales, across multiple languages, and with various modalities. This demo paper highlights the toolkit's key features, bridging academia and industry by supporting efficient training, inference, and evaluation of neural retrievers. We showcase a unified dense retriever achieving strong multilingual and multimodal effectiveness, and conduct a cross-modality zero-shot study to demonstrate its research potential. Alongside, we release OmniEmbed, to the best of our knowledge, the first embedding model that unifies text, image document, video, and audio retrieval, serving as a baseline for future research. 

---
# RAGAR: Retrieval Augment Personalized Image Generation Guided by Recommendation 

**Authors**: Run Ling, Wenji Wang, Yuting Liu, Guibing Guo, Linying Jiang, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01657)  

**Abstract**: Personalized image generation is crucial for improving the user experience, as it renders reference images into preferred ones according to user visual preferences. Although effective, existing methods face two main issues. First, existing methods treat all items in the user historical sequence equally when extracting user preferences, overlooking the varying semantic similarities between historical items and the reference item. Disproportionately high weights for low-similarity items distort users' visual preferences for the reference item. Second, existing methods heavily rely on consistency between generated and reference images to optimize the generation, which leads to underfitting user preferences and hinders personalization. To address these issues, we propose Retrieval Augment Personalized Image GenerAtion guided by Recommendation (RAGAR). Our approach uses a retrieval mechanism to assign different weights to historical items according to their similarities to the reference item, thereby extracting more refined users' visual preferences for the reference item. Then we introduce a novel rank task based on the multi-modal ranking model to optimize the personalization of the generated images instead of forcing depend on consistency. Extensive experiments and human evaluations on three real-world datasets demonstrate that RAGAR achieves significant improvements in both personalization and semantic metrics compared to five baselines. 

---
# A Multi-Granularity Multimodal Retrieval Framework for Multimodal Document Tasks 

**Authors**: Mingjun Xu, Zehui Wang, Hengxing Cai, Renxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.01457)  

**Abstract**: Retrieval-augmented generation (RAG) systems have predominantly focused on text-based retrieval, limiting their effectiveness in handling visually-rich documents that encompass text, images, tables, and charts. To bridge this gap, we propose a unified multi-granularity multimodal retrieval framework tailored for two benchmark tasks: MMDocIR and M2KR. Our approach integrates hierarchical encoding strategies, modality-aware retrieval mechanisms, and reranking modules to effectively capture and utilize the complex interdependencies between textual and visual modalities. By leveraging off-the-shelf vision-language models and implementing a training-free hybridretrieval strategy, our framework demonstrates robust performance without the need for task-specific fine-tuning. Experimental evaluations reveal that incorporating layout-aware search and reranking modules significantly enhances retrieval accuracy, achieving a top performance score of 65.56. This work underscores the potential of scalable and reproducible solutions in advancing multimodal document retrieval systems. 

---
