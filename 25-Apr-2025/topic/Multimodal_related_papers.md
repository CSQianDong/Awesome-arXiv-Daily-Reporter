# DRC: Enhancing Personalized Image Generation via Disentangled Representation Composition 

**Authors**: Yiyan Xu, Wuqiang Zheng, Wenjie Wang, Fengbin Zhu, Xinting Hu, Yang Zhang, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.17349)  

**Abstract**: Personalized image generation has emerged as a promising direction in multimodal content creation. It aims to synthesize images tailored to individual style preferences (e.g., color schemes, character appearances, layout) and semantic intentions (e.g., emotion, action, scene contexts) by leveraging user-interacted history images and multimodal instructions. Despite notable progress, existing methods -- whether based on diffusion models, large language models, or Large Multimodal Models (LMMs) -- struggle to accurately capture and fuse user style preferences and semantic intentions. In particular, the state-of-the-art LMM-based method suffers from the entanglement of visual features, leading to Guidance Collapse, where the generated images fail to preserve user-preferred styles or reflect the specified semantics.
To address these limitations, we introduce DRC, a novel personalized image generation framework that enhances LMMs through Disentangled Representation Composition. DRC explicitly extracts user style preferences and semantic intentions from history images and the reference image, respectively, to form user-specific latent instructions that guide image generation within LMMs. Specifically, it involves two critical learning stages: 1) Disentanglement learning, which employs a dual-tower disentangler to explicitly separate style and semantic features, optimized via a reconstruction-driven paradigm with difficulty-aware importance sampling; and 2) Personalized modeling, which applies semantic-preserving augmentations to effectively adapt the disentangled representations for robust personalized generation. Extensive experiments on two benchmarks demonstrate that DRC shows competitive performance while effectively mitigating the guidance collapse issue, underscoring the importance of disentangled representation learning for controllable and effective personalized image generation. 

---
# Quadratic Interest Network for Multimodal Click-Through Rate Prediction 

**Authors**: Honghao Li, Hanwei Li, Jing Zhang, Yi Zhang, Ziniu Yu, Lei Sang, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17699)  

**Abstract**: Multimodal click-through rate (CTR) prediction is a key technique in industrial recommender systems. It leverages heterogeneous modalities such as text, images, and behavioral logs to capture high-order feature interactions between users and items, thereby enhancing the system's understanding of user interests and its ability to predict click behavior. The primary challenge in this field lies in effectively utilizing the rich semantic information from multiple modalities while satisfying the low-latency requirements of online inference in real-world applications. To foster progress in this area, the Multimodal CTR Prediction Challenge Track of the WWW 2025 EReL@MIR Workshop formulates the problem into two tasks: (1) Task 1 of Multimodal Item Embedding: this task aims to explore multimodal information extraction and item representation learning methods that enhance recommendation tasks; and (2) Task 2 of Multimodal CTR Prediction: this task aims to explore what multimodal recommendation model can effectively leverage multimodal embedding features and achieve better performance. In this paper, we propose a novel model for Task 2, named Quadratic Interest Network (QIN) for Multimodal CTR Prediction. Specifically, QIN employs adaptive sparse target attention to extract multimodal user behavior features, and leverages Quadratic Neural Networks to capture high-order feature interactions. As a result, QIN achieved an AUC of 0.9798 on the leaderboard and ranked second in the competition. The model code, training logs, hyperparameter configurations, and checkpoints are available at this https URL. 

---
# Cracking the Code of Action: a Generative Approach to Affordances for Reinforcement Learning 

**Authors**: Lynn Cherif, Flemming Kondrup, David Venuto, Ankit Anand, Doina Precup, Khimya Khetarpal  

**Link**: [PDF](https://arxiv.org/pdf/2504.17282)  

**Abstract**: Agents that can autonomously navigate the web through a graphical user interface (GUI) using a unified action space (e.g., mouse and keyboard actions) can require very large amounts of domain-specific expert demonstrations to achieve good performance. Low sample efficiency is often exacerbated in sparse-reward and large-action-space environments, such as a web GUI, where only a few actions are relevant in any given situation. In this work, we consider the low-data regime, with limited or no access to expert behavior. To enable sample-efficient learning, we explore the effect of constraining the action space through $\textit{intent-based affordances}$ -- i.e., considering in any situation only the subset of actions that achieve a desired outcome. We propose $\textbf{Code as Generative Affordances}$ $(\textbf{$\texttt{CoGA}$})$, a method that leverages pre-trained vision-language models (VLMs) to generate code that determines affordable actions through implicit intent-completion functions and using a fully-automated program generation and verification pipeline. These programs are then used in-the-loop of a reinforcement learning agent to return a set of affordances given a pixel observation. By greatly reducing the number of actions that an agent must consider, we demonstrate on a wide range of tasks in the MiniWob++ benchmark that: $\textbf{1)}$ $\texttt{CoGA}$ is orders of magnitude more sample efficient than its RL agent, $\textbf{2)}$ $\texttt{CoGA}$'s programs can generalize within a family of tasks, and $\textbf{3)}$ $\texttt{CoGA}$ performs better or on par compared with behavior cloning when a small number of expert demonstrations is available. 

---
# Hierarchical and Multimodal Data for Daily Activity Understanding 

**Authors**: Ghazal Kaviani, Yavuz Yarici, Seulgi Kim, Mohit Prabhushankar, Ghassan AlRegib, Mashhour Solh, Ameya Patil  

**Link**: [PDF](https://arxiv.org/pdf/2504.17696)  

**Abstract**: Daily Activity Recordings for Artificial Intelligence (DARai, pronounced "Dahr-ree") is a multimodal, hierarchically annotated dataset constructed to understand human activities in real-world settings. DARai consists of continuous scripted and unscripted recordings of 50 participants in 10 different environments, totaling over 200 hours of data from 20 sensors including multiple camera views, depth and radar sensors, wearable inertial measurement units (IMUs), electromyography (EMG), insole pressure sensors, biomonitor sensors, and gaze tracker.
To capture the complexity in human activities, DARai is annotated at three levels of hierarchy: (i) high-level activities (L1) that are independent tasks, (ii) lower-level actions (L2) that are patterns shared between activities, and (iii) fine-grained procedures (L3) that detail the exact execution steps for actions. The dataset annotations and recordings are designed so that 22.7% of L2 actions are shared between L1 activities and 14.2% of L3 procedures are shared between L2 actions. The overlap and unscripted nature of DARai allows counterfactual activities in the dataset.
Experiments with various machine learning models showcase the value of DARai in uncovering important challenges in human-centered applications. Specifically, we conduct unimodal and multimodal sensor fusion experiments for recognition, temporal localization, and future action anticipation across all hierarchical annotation levels. To highlight the limitations of individual sensors, we also conduct domain-variant experiments that are enabled by DARai's multi-sensor and counterfactual activity design setup.
The code, documentation, and dataset are available at the dedicated DARai website: this https URL 

---
# FRAG: Frame Selection Augmented Generation for Long Video and Long Document Understanding 

**Authors**: De-An Huang, Subhashree Radhakrishnan, Zhiding Yu, Jan Kautz  

**Link**: [PDF](https://arxiv.org/pdf/2504.17447)  

**Abstract**: There has been impressive progress in Large Multimodal Models (LMMs). Recent works extend these models to long inputs, including multi-page documents and long videos. However, the model size and performance of these long context models are still limited due to the computational cost in both training and inference. In this work, we explore an orthogonal direction and process long inputs without long context LMMs. We propose Frame Selection Augmented Generation (FRAG), where the model first selects relevant frames within the input, and then only generates the final outputs based on the selected frames. The core of the selection process is done by scoring each frame independently, which does not require long context processing. The frames with the highest scores are then selected by a simple Top-K selection. We show that this frustratingly simple framework is applicable to both long videos and multi-page documents using existing LMMs without any fine-tuning. We consider two models, LLaVA-OneVision and InternVL2, in our experiments and show that FRAG consistently improves the performance and achieves state-of-the-art performances for both long video and long document understanding. For videos, FRAG substantially improves InternVL2-76B by 5.8% on MLVU and 3.7% on Video-MME. For documents, FRAG achieves over 20% improvements on MP-DocVQA compared with recent LMMs specialized in long document understanding. Code is available at: this https URL 

---
# On the workflow, opportunities and challenges of developing foundation model in geophysics 

**Authors**: Hanlin Sheng, Xinming Wu, Hang Gao, Haibin Di, Sergey Fomel, Jintao Li, Xu Si  

**Link**: [PDF](https://arxiv.org/pdf/2504.17384)  

**Abstract**: Foundation models, as a mainstream technology in artificial intelligence, have demonstrated immense potential across various domains in recent years, particularly in handling complex tasks and multimodal data. In the field of geophysics, although the application of foundation models is gradually expanding, there is currently a lack of comprehensive reviews discussing the full workflow of integrating foundation models with geophysical data. To address this gap, this paper presents a complete framework that systematically explores the entire process of developing foundation models in conjunction with geophysical data. From data collection and preprocessing to model architecture selection, pre-training strategies, and model deployment, we provide a detailed analysis of the key techniques and methodologies at each stage. In particular, considering the diversity, complexity, and physical consistency constraints of geophysical data, we discuss targeted solutions to address these challenges. Furthermore, we discuss how to leverage the transfer learning capabilities of foundation models to reduce reliance on labeled data, enhance computational efficiency, and incorporate physical constraints into model training, thereby improving physical consistency and interpretability. Through a comprehensive summary and analysis of the current technological landscape, this paper not only fills the gap in the geophysics domain regarding a full-process review of foundation models but also offers valuable practical guidance for their application in geophysical data analysis, driving innovation and advancement in the field. 

---
# Symbolic Representation for Any-to-Any Generative Tasks 

**Authors**: Jiaqi Chen, Xiaoye Zhu, Yue Wang, Tianyang Liu, Xinhui Chen, Ying Chen, Chak Tou Leong, Yifei Ke, Joseph Liu, Yiwen Yuan, Julian McAuley, Li-jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.17261)  

**Abstract**: We propose a symbolic generative task description language and a corresponding inference engine capable of representing arbitrary multimodal tasks as structured symbolic flows. Unlike conventional generative models that rely on large-scale training and implicit neural representations to learn cross-modal mappings, often at high computational cost and with limited flexibility, our framework introduces an explicit symbolic representation comprising three core primitives: functions, parameters, and topological logic. Leveraging a pre-trained language model, our inference engine maps natural language instructions directly to symbolic workflows in a training-free manner. Our framework successfully performs over 12 diverse multimodal generative tasks, demonstrating strong performance and flexibility without the need for task-specific tuning. Experiments show that our method not only matches or outperforms existing state-of-the-art unified models in content quality, but also offers greater efficiency, editability, and interruptibility. We believe that symbolic task representations provide a cost-effective and extensible foundation for advancing the capabilities of generative AI. 

---
# DIMT25@ICDAR2025: HW-TSC's End-to-End Document Image Machine Translation System Leveraging Large Vision-Language Model 

**Authors**: Zhanglin Wu, Tengfei Song, Ning Xie, Weidong Zhang, Pengfei Li, Shuang Wu, Chong Li, Junhao Zhu, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17315)  

**Abstract**: This paper presents the technical solution proposed by Huawei Translation Service Center (HW-TSC) for the "End-to-End Document Image Machine Translation for Complex Layouts" competition at the 19th International Conference on Document Analysis and Recognition (DIMT25@ICDAR2025). Leveraging state-of-the-art open-source large vision-language model (LVLM), we introduce a training framework that combines multi-task learning with perceptual chain-of-thought to develop a comprehensive end-to-end document translation system. During the inference phase, we apply minimum Bayesian decoding and post-processing strategies to further enhance the system's translation capabilities. Our solution uniquely addresses both OCR-based and OCR-free document image translation tasks within a unified framework. This paper systematically details the training methods, inference strategies, LVLM base models, training data, experimental setups, and results, demonstrating an effective approach to document image machine translation. 

---
# MCAF: Efficient Agent-based Video Understanding Framework through Multimodal Coarse-to-Fine Attention Focusing 

**Authors**: Shiwen Cao, Zhaoxing Zhang, Junming Jiao, Juyi Qiao, Guowen Song, Rong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17213)  

**Abstract**: Even in the era of rapid advances in large models, video understanding, particularly long videos, remains highly challenging. Compared with textual or image-based information, videos commonly contain more information with redundancy, requiring large models to strategically allocate attention at a global level for accurate comprehension. To address this, we propose MCAF, an agent-based, training-free framework perform video understanding through Multimodal Coarse-to-fine Attention Focusing. The key innovation lies in its ability to sense and prioritize segments of the video that are highly relevant to the understanding task. First, MCAF hierarchically concentrates on highly relevant frames through multimodal information, enhancing the correlation between the acquired contextual information and the query. Second, it employs a dilated temporal expansion mechanism to mitigate the risk of missing crucial details when extracting information from these concentrated frames. In addition, our framework incorporates a self-reflection mechanism utilizing the confidence level of the model's responses as feedback. By iteratively applying these two creative focusing strategies, it adaptively adjusts attention to capture highly query-connected context and thus improves response accuracy. MCAF outperforms comparable state-of-the-art methods on average. On the EgoSchema dataset, it achieves a remarkable 5% performance gain over the leading approach. Meanwhile, on Next-QA and IntentQA datasets, it outperforms the current state-of-the-art standard by 0.2% and 0.3% respectively. On the Video-MME dataset, which features videos averaging nearly an hour in length, MCAF also outperforms other agent-based methods. 

---
# DyMU: Dynamic Merging and Virtual Unmerging for Efficient VLMs 

**Authors**: Zhenhailong Wang, Senthil Purushwalkam, Caiming Xiong, Silvio Savarese, Heng Ji, Ran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17040)  

**Abstract**: We present DyMU, an efficient, training-free framework that dynamically reduces the computational burden of vision-language models (VLMs) while maintaining high task performance. Our approach comprises two key components. First, Dynamic Token Merging (DToMe) reduces the number of visual token embeddings by merging similar tokens based on image complexity, addressing the inherent inefficiency of fixed-length outputs in vision transformers. Second, Virtual Token Unmerging (VTU) simulates the expected token sequence for large language models (LLMs) by efficiently reconstructing the attention dynamics of a full sequence, thus preserving the downstream performance without additional fine-tuning. Unlike previous approaches, our method dynamically adapts token compression to the content of the image and operates completely training-free, making it readily applicable to most state-of-the-art VLM architectures. Extensive experiments on image and video understanding tasks demonstrate that DyMU can reduce the average visual token count by 32%-85% while achieving comparable performance to full-length models across diverse VLM architectures, including the recently popularized AnyRes-based visual encoders. Furthermore, through qualitative analyses, we demonstrate that DToMe effectively adapts token reduction based on image complexity and, unlike existing systems, provides users more control over computational costs. Project page: this https URL. 

---
# S2Vec: Self-Supervised Geospatial Embeddings 

**Authors**: Shushman Choudhury, Elad Aharoni, Chandrakumari Suvarna, Iveel Tsogsuren, Abdul Rahman Kreidieh, Chun-Ta Lu, Neha Arora  

**Link**: [PDF](https://arxiv.org/pdf/2504.16942)  

**Abstract**: Scalable general-purpose representations of the built environment are crucial for geospatial artificial intelligence applications. This paper introduces S2Vec, a novel self-supervised framework for learning such geospatial embeddings. S2Vec uses the S2 Geometry library to partition large areas into discrete S2 cells, rasterizes built environment feature vectors within cells as images, and applies masked autoencoding on these rasterized images to encode the feature vectors. This approach yields task-agnostic embeddings that capture local feature characteristics and broader spatial relationships. We evaluate S2Vec on three large-scale socioeconomic prediction tasks, showing its competitive performance against state-of-the-art image-based embeddings. We also explore the benefits of combining S2Vec embeddings with image-based embeddings downstream, showing that such multimodal fusion can often improve performance. Our results highlight how S2Vec can learn effective general-purpose geospatial representations and how it can complement other data modalities in geospatial artificial intelligence. 

---
# M-MRE: Extending the Mutual Reinforcement Effect to Multimodal Information Extraction 

**Authors**: Chengguang Gan, Sunbowen Lee, Zhixi Cai, Yanbin Wei, Lei Zheng, Yunhao Liang, Shiwen Ni, Tatsunori Mori  

**Link**: [PDF](https://arxiv.org/pdf/2504.17353)  

**Abstract**: Mutual Reinforcement Effect (MRE) is an emerging subfield at the intersection of information extraction and model interpretability. MRE aims to leverage the mutual understanding between tasks of different granularities, enhancing the performance of both coarse-grained and fine-grained tasks through joint modeling. While MRE has been explored and validated in the textual domain, its applicability to visual and multimodal domains remains unexplored. In this work, we extend MRE to the multimodal information extraction domain for the first time. Specifically, we introduce a new task: Multimodal Mutual Reinforcement Effect (M-MRE), and construct a corresponding dataset to support this task. To address the challenges posed by M-MRE, we further propose a Prompt Format Adapter (PFA) that is fully compatible with various Large Vision-Language Models (LVLMs). Experimental results demonstrate that MRE can also be observed in the M-MRE task, a multimodal text-image understanding scenario. This provides strong evidence that MRE facilitates mutual gains across three interrelated tasks, confirming its generalizability beyond the textual domain. 

---
# Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection 

**Authors**: Zihan Wang, Lu Yuan, Zhengxuan Zhang, Qing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17332)  

**Abstract**: In the digital era, social media has become a major conduit for information dissemination, yet it also facilitates the rapid spread of misinformation. Traditional misinformation detection methods primarily focus on surface-level features, overlooking the crucial roles of human empathy in the propagation process. To address this gap, we propose the Dual-Aspect Empathy Framework (DAE), which integrates cognitive and emotional empathy to analyze misinformation from both the creator and reader perspectives. By examining creators' cognitive strategies and emotional appeals, as well as simulating readers' cognitive judgments and emotional responses using Large Language Models (LLMs), DAE offers a more comprehensive and human-centric approach to misinformation detection. Moreover, we further introduce an empathy-aware filtering mechanism to enhance response authenticity and diversity. Experimental results on benchmark datasets demonstrate that DAE outperforms existing methods, providing a novel paradigm for multimodal misinformation detection. 

---
# TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation 

**Authors**: Ling You, Wenxuan Huang, Xinni Xie, Xiangyi Wei, Bangyan Li, Shaohui Lin, Yang Li, Changbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17365)  

**Abstract**: Soccer is a globally popular sporting event, typically characterized by long matches and distinctive highlight moments. Recent advances in Multimodal Large Language Models (MLLMs) offer promising capabilities in temporal grounding and video understanding, soccer commentary generation often requires precise temporal localization and semantically rich descriptions over long-form video. However, existing soccer MLLMs often rely on the temporal a priori for caption generation, so they cannot process the soccer video end-to-end. While some traditional approaches follow a two-step paradigm that is complex and fails to capture the global context to achieve suboptimal performance. To solve the above issues, we present TimeSoccer, the first end-to-end soccer MLLM for Single-anchor Dense Video Captioning (SDVC) in full-match soccer videos. TimeSoccer jointly predicts timestamps and generates captions in a single pass, enabling global context modeling across 45-minute matches. To support long video understanding of soccer matches, we introduce MoFA-Select, a training-free, motion-aware frame compression module that adaptively selects representative frames via a coarse-to-fine strategy, and incorporates complementary training paradigms to strengthen the model's ability to handle long temporal sequences. Extensive experiments demonstrate that our TimeSoccer achieves State-of-The-Art (SoTA) performance on the SDVC task in an end-to-end form, generating high-quality commentary with accurate temporal alignment and strong semantic relevance. 

---
