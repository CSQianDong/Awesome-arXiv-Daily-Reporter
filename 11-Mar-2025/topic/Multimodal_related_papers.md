# LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL 

**Authors**: Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07536)  

**Abstract**: Enhancing reasoning in Large Multimodal Models (LMMs) faces unique challenges from the complex interplay between visual perception and logical reasoning, particularly in compact 3B-parameter architectures where architectural constraints limit reasoning capacity and modality alignment.
While rule-based reinforcement learning (RL) excels in text-only domains, its multimodal extension confronts two critical barriers: (1) data limitations due to ambiguous answers and scarce complex reasoning examples, and (2) degraded foundational reasoning induced by multimodal pretraining.
To address these challenges, we propose \textbf{\method}, a two-stage framework adapting rule-based RL for multimodal reasoning through \textbf{Foundational Reasoning Enhancement (FRE)} followed by \textbf{Multimodal Generalization Training (MGT)}. The FRE stage first strengthens reasoning abilities using text-only data with rule-based RL, then the MGT stage generalizes these reasoning capabilities to multimodal domains.
Experiments on Qwen2.5-VL-Instruct-3B demonstrate that \method achieves 4.83\% and 4.5\% average improvements over baselines in multimodal and text-only benchmarks, respectively, with a 3.63\% gain in complex Football Game tasks. These results validate that text-based reasoning enhancement enables effective multimodal generalization, offering a data-efficient paradigm that bypasses costly high-quality multimodal training data. 

---
# A Novel Ophthalmic Benchmark for Evaluating Multimodal Large Language Models with Fundus Photographs and OCT Images 

**Authors**: Xiaoyi Liang, Mouxiao Bian, Moxin Chen, Lihao Liu, Junjun He, Jie Xu, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07094)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable potential across various medical applications. Building on this foundation, multimodal large language models (MLLMs) integrate LLMs with visual models to process diverse inputs, including clinical data and medical images. In ophthalmology, LLMs have been explored for analyzing optical coherence tomography (OCT) reports, assisting in disease classification, and even predicting treatment outcomes. However, existing MLLM benchmarks often fail to capture the complexities of real-world clinical practice, particularly in the analysis of OCT images. Many suffer from limitations such as small sample sizes, a lack of diverse OCT datasets, and insufficient expert validation. These shortcomings hinder the accurate assessment of MLLMs' ability to interpret OCT scans and their broader applicability in ophthalmology. Our dataset, curated through rigorous quality control and expert annotation, consists of 439 fundus images and 75 OCT images. Using a standardized API-based framework, we assessed seven mainstream MLLMs and observed significant variability in diagnostic accuracy across different diseases. While some models performed well in diagnosing conditions such as diabetic retinopathy and age-related macular degeneration, they struggled with others, including choroidal neovascularization and myopia, highlighting inconsistencies in performance and the need for further refinement. Our findings emphasize the importance of developing clinically relevant benchmarks to provide a more accurate assessment of MLLMs' capabilities. By refining these models and expanding their scope, we can enhance their potential to transform ophthalmic diagnosis and treatment. 

---
# Exploring Multimodal Perception in Large Language Models Through Perceptual Strength Ratings 

**Authors**: Jonghyun Lee, Dojun Park, Jiwoo Lee, Hoekeon Choi, Sung-Eun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06980)  

**Abstract**: This study investigated the multimodal perception of large language models (LLMs), focusing on their ability to capture human-like perceptual strength ratings across sensory modalities. Utilizing perceptual strength ratings as a benchmark, the research compared GPT-3.5, GPT-4, GPT-4o, and GPT-4o-mini, highlighting the influence of multimodal inputs on grounding and linguistic reasoning. While GPT-4 and GPT-4o demonstrated strong alignment with human evaluations and significant advancements over smaller models, qualitative analyses revealed distinct differences in processing patterns, such as multisensory overrating and reliance on loose semantic associations. Despite integrating multimodal capabilities, GPT-4o did not exhibit superior grounding compared to GPT-4, raising questions about their role in improving human-like grounding. These findings underscore how LLMs' reliance on linguistic patterns can both approximate and diverge from human embodied cognition, revealing limitations in replicating sensory experiences. 

---
# KwaiChat: A Large-Scale Video-Driven Multilingual Mixed-Type Dialogue Corpus 

**Authors**: Xiaoming Shi, Zeming Liu, Yiming Lei, Chenkai Zhang, Haitao Leng, Chuan Wang, Qingjie Liu, Wanxiang Che, Shaoguo Liu, Size Li, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06899)  

**Abstract**: Video-based dialogue systems, such as education assistants, have compelling application value, thereby garnering growing interest. However, the current video-based dialogue systems are limited by their reliance on a single dialogue type, which hinders their versatility in practical applications across a range of scenarios, including question-answering, emotional dialog, etc. In this paper, we identify this challenge as how to generate video-driven multilingual mixed-type dialogues. To mitigate this challenge, we propose a novel task and create a human-to-human video-driven multilingual mixed-type dialogue corpus, termed KwaiChat, containing a total of 93,209 videos and 246,080 dialogues, across 4 dialogue types, 30 domains, 4 languages, and 13 topics. Additionally, we establish baseline models on KwaiChat. An extensive analysis of 7 distinct LLMs on KwaiChat reveals that GPT-4o achieves the best performance but still cannot perform well in this situation even with the help of in-context learning and fine-tuning, which indicates that the task is not trivial and needs further research. 

---
# VisualSimpleQA: A Benchmark for Decoupled Evaluation of Large Vision-Language Models in Fact-Seeking Question Answering 

**Authors**: Yanling Wang, Yihan Zhao, Xiaodong Chen, Shasha Guo, Lixin Liu, Haoyang Li, Yong Xiao, Jing Zhang, Qi Li, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06492)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable achievements, yet the generation of non-factual responses remains prevalent in fact-seeking question answering (QA). Current multimodal fact-seeking benchmarks primarily focus on comparing model outputs to ground truth answers, providing limited insights into the performance of modality-specific modules. To bridge this gap, we introduce VisualSimpleQA, a multimodal fact-seeking benchmark with two key features. First, it enables streamlined and decoupled evaluation of LVLMs in visual and linguistic modalities. Second, it incorporates well-defined difficulty criteria to guide human annotation and facilitates the extraction of a challenging subset, VisualSimpleQA-hard. Experiments on 15 LVLMs show that even state-of-the-art models such as GPT-4o achieve merely 60%+ correctness in multimodal fact-seeking QA on VisualSimpleQA and 30%+ on VisualSimpleQA-hard. Furthermore, the decoupled evaluation across these models highlights substantial opportunities for improvement in both visual and linguistic modules. The dataset is available at this https URL. 

---
# MoEMoE: Question Guided Dense and Scalable Sparse Mixture-of-Expert for Multi-source Multi-modal Answering 

**Authors**: Vinay Kumar Verma, Shreyas Sunil Kulkarni, Happy Mittal, Deepak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2503.06296)  

**Abstract**: Question Answering (QA) and Visual Question Answering (VQA) are well-studied problems in the language and vision domain. One challenging scenario involves multiple sources of information, each of a different modality, where the answer to the question may exist in one or more sources. This scenario contains richer information but is highly complex to handle. In this work, we formulate a novel question-answer generation (QAG) framework in an environment containing multi-source, multimodal information. The answer may belong to any or all sources; therefore, selecting the most prominent answer source or an optimal combination of all sources for a given question is challenging. To address this issue, we propose a question-guided attention mechanism that learns attention across multiple sources and decodes this information for robust and unbiased answer generation. To learn attention within each source, we introduce an explicit alignment between questions and various information sources, which facilitates identifying the most pertinent parts of the source information relative to the question. Scalability in handling diverse questions poses a challenge. We address this by extending our model to a sparse mixture-of-experts (sparse-MoE) framework, enabling it to handle thousands of question types. Experiments on T5 and Flan-T5 using three datasets demonstrate the model's efficacy, supported by ablation studies. 

---
# Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning 

**Authors**: Yanjun Chen, Yirong Sun, Xinghao Chen, Jian Wang, Xiaoyu Shen, Wenjie Li, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06232)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in natural language tasks but remains underexplored in multimodal alignment. This study investigates its integration into 3D vision-language learning by embedding structured reasoning into alignment training. We introduce the 3D-CoT Benchmark, a dataset with hierarchical CoT annotations covering shape recognition, functional inference, and causal reasoning. Through controlled experiments, we compare CoT-structured and standard textual annotations across large reasoning models (LRMs) and large language models (LLMs). Our evaluation employs a dual-layer framework assessing both intermediate reasoning and final inference quality. Extensive experiments demonstrate that CoT significantly improves 3D semantic grounding, with LRMs leveraging CoT more effectively than LLMs. Furthermore, we highlight that annotation structure influences performance-explicit reasoning markers aid LLMs, while unmarked CoT better aligns with LRM inference patterns. Our analyses suggest that CoT is crucial for enhancing multimodal reasoning, with implications beyond 3D tasks. 

---
# GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images 

**Authors**: Xiang Lan, Feng Wu, Kai He, Qinghao Zhao, Shenda Hong, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06073)  

**Abstract**: While recent multimodal large language models (MLLMs) have advanced automated ECG interpretation, they still face two key limitations: (1) insufficient multimodal synergy between time series signals and visual ECG representations, and (2) limited explainability in linking diagnoses to granular waveform evidence. We introduce GEM, the first MLLM unifying ECG time series, 12-lead ECG images and text for grounded and clinician-aligned ECG interpretation. GEM enables feature-grounded analysis, evidence-driven reasoning, and a clinician-like diagnostic process through three core innovations: a dual-encoder framework extracting complementary time series and image features, cross-modal alignment for effective multimodal understanding, and knowledge-guided instruction generation for generating high-granularity grounding data (ECG-Grounding) linking diagnoses to measurable parameters ($e.g.$, QRS/PR Intervals). Additionally, we propose the Grounded ECG Understanding task, a clinically motivated benchmark designed to comprehensively assess the MLLM's capability in grounded ECG understanding. Experimental results on both existing and our proposed benchmarks show GEM significantly improves predictive performance (CSN $7.4\% \uparrow$), explainability ($22.7\% \uparrow$), and grounding ($24.8\% \uparrow$), making it more suitable for real-world clinical applications. GitHub repository: this https URL 

---
# GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices 

**Authors**: Xudong Lu, Yinghao Chen, Renshou Wu, Haohao Gao, Xi Chen, Xue Yang, Xiangyu Zhao, Aojun Zhou, Fangyuan Li, Yafei Wen, Xiaoxin Chen, Shuai Ren, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06019)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have enabled their deployment on mobile devices. However, challenges persist in maintaining strong language capabilities and ensuring hardware compatibility, both of which are crucial for user experience and practical deployment efficiency. In our deployment process, we observe that existing MLLMs often face performance degradation on pure language tasks, and the current NPU platforms on smartphones do not support the MoE architecture, which is commonly used to preserve pure language capabilities during multimodal training. To address these issues, we systematically analyze methods to maintain pure language capabilities during the training of MLLMs, focusing on both training data and model architecture aspects. Based on these analyses, we propose GenieBlue, an efficient MLLM structural design that integrates both linguistic and multimodal capabilities for LLMs on mobile devices. GenieBlue freezes the original LLM parameters during MLLM training to maintain pure language capabilities. It acquires multimodal capabilities by duplicating specific transformer blocks for full fine-tuning and integrating lightweight LoRA modules. This approach preserves language capabilities while achieving comparable multimodal performance through extensive training. Deployed on smartphone NPUs, GenieBlue demonstrates efficiency and practicality for applications on mobile devices. 

---
# Multi-Modal 3D Mesh Reconstruction from Images and Text 

**Authors**: Melvin Reka, Tessa Pulli, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2503.07190)  

**Abstract**: 6D object pose estimation for unseen objects is essential in robotics but traditionally relies on trained models that require large datasets, high computational costs, and struggle to generalize. Zero-shot approaches eliminate the need for training but depend on pre-existing 3D object models, which are often impractical to obtain. To address this, we propose a language-guided few-shot 3D reconstruction method, reconstructing a 3D mesh from few input images. In the proposed pipeline, receives a set of input images and a language query. A combination of GroundingDINO and Segment Anything Model outputs segmented masks from which a sparse point cloud is reconstructed with VGGSfM. Subsequently, the mesh is reconstructed with the Gaussian Splatting method SuGAR. In a final cleaning step, artifacts are removed, resulting in the final 3D mesh of the queried object. We evaluate the method in terms of accuracy and quality of the geometry and texture. Furthermore, we study the impact of imaging conditions such as viewing angle, number of input images, and image overlap on 3D object reconstruction quality, efficiency, and computational scalability. 

---
# Attention, Please! PixelSHAP Reveals What Vision-Language Models Actually Focus On 

**Authors**: Roni Goldshmidt  

**Link**: [PDF](https://arxiv.org/pdf/2503.06670)  

**Abstract**: Interpretability in Vision-Language Models (VLMs) is crucial for trust, debugging, and decision-making in high-stakes applications. We introduce PixelSHAP, a model-agnostic framework extending Shapley-based analysis to structured visual entities. Unlike previous methods focusing on text prompts, PixelSHAP applies to vision-based reasoning by systematically perturbing image objects and quantifying their influence on a VLM's response. PixelSHAP requires no model internals, operating solely on input-output pairs, making it compatible with open-source and commercial models. It supports diverse embedding-based similarity metrics and scales efficiently using optimization techniques inspired by Shapley-based methods. We validate PixelSHAP in autonomous driving, highlighting its ability to enhance interpretability. Key challenges include segmentation sensitivity and object occlusion. Our open-source implementation facilitates further research. 

---
# TI-JEPA: An Innovative Energy-based Joint Embedding Strategy for Text-Image Multimodal Systems 

**Authors**: Khang H. N. Vo, Duc P. T. Nguyen, Thong Nguyen, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06380)  

**Abstract**: This paper focuses on multimodal alignment within the realm of Artificial Intelligence, particularly in text and image modalities. The semantic gap between the textual and visual modality poses a discrepancy problem towards the effectiveness of multi-modalities fusion. Therefore, we introduce Text-Image Joint Embedding Predictive Architecture (TI-JEPA), an innovative pre-training strategy that leverages energy-based model (EBM) framework to capture complex cross-modal relationships. TI-JEPA combines the flexibility of EBM in self-supervised learning to facilitate the compatibility between textual and visual elements. Through extensive experiments across multiple benchmarks, we demonstrate that TI-JEPA achieves state-of-the-art performance on multimodal sentiment analysis task (and potentially on a wide range of multimodal-based tasks, such as Visual Question Answering), outperforming existing pre-training methodologies. Our findings highlight the potential of using energy-based framework in advancing multimodal fusion and suggest significant improvements for downstream applications. 

---
# Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection 

**Authors**: Chandan Kumar Sah, Ankit Kumar Shaw, Xiaoli Lian, Arsalan Shahid Baig, Tuopu Wen, Kun Jiang, Mengmeng Yang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06313)  

**Abstract**: Autonomous vehicles (AVs) require reliable traffic sign recognition and robust lane detection capabilities to ensure safe navigation in complex and dynamic environments. This paper introduces an integrated approach combining advanced deep learning techniques and Multimodal Large Language Models (MLLMs) for comprehensive road perception. For traffic sign recognition, we systematically evaluate ResNet-50, YOLOv8, and RT-DETR, achieving state-of-the-art performance of 99.8% with ResNet-50, 98.0% accuracy with YOLOv8, and achieved 96.6% accuracy in RT-DETR despite its higher computational complexity. For lane detection, we propose a CNN-based segmentation method enhanced by polynomial curve fitting, which delivers high accuracy under favorable conditions. Furthermore, we introduce a lightweight, Multimodal, LLM-based framework that directly undergoes instruction tuning using small yet diverse datasets, eliminating the need for initial pretraining. This framework effectively handles various lane types, complex intersections, and merging zones, significantly enhancing lane detection reliability by reasoning under adverse conditions. Despite constraints in available training resources, our multimodal approach demonstrates advanced reasoning capabilities, achieving a Frame Overall Accuracy (FRM) of 53.87%, a Question Overall Accuracy (QNS) of 82.83%, lane detection accuracies of 99.6% in clear conditions and 93.0% at night, and robust performance in reasoning about lane invisibility due to rain (88.4%) or road degradation (95.6%). The proposed comprehensive framework markedly enhances AV perception reliability, thus contributing significantly to safer autonomous driving across diverse and challenging road scenarios. 

---
# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models 

**Authors**: Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, Shaohui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06749)  

**Abstract**: DeepSeek-R1-Zero has successfully demonstrated the emergence of reasoning capabilities in LLMs purely through Reinforcement Learning (RL). Inspired by this breakthrough, we explore how RL can be utilized to enhance the reasoning capability of MLLMs. However, direct training with RL struggles to activate complex reasoning capabilities such as questioning and reflection in MLLMs, due to the absence of substantial high-quality multimodal reasoning data. To address this issue, we propose the reasoning MLLM, Vision-R1, to improve multimodal reasoning capability. Specifically, we first construct a high-quality multimodal CoT dataset without human annotations by leveraging an existing MLLM and DeepSeek-R1 through modality bridging and data filtering to obtain a 200K multimodal CoT dataset, Vision-R1-cold dataset. It serves as cold-start initialization data for Vision-R1. To mitigate the optimization challenges caused by overthinking after cold start, we propose Progressive Thinking Suppression Training (PTST) strategy and employ Group Relative Policy Optimization (GRPO) with the hard formatting result reward function to gradually refine the model's ability to learn correct and complex reasoning processes on a 10K multimodal math dataset. Comprehensive experiments show our model achieves an average improvement of $\sim$6% across various multimodal math reasoning benchmarks. Vision-R1-7B achieves a 73.5% accuracy on the widely used MathVista benchmark, which is only 0.4% lower than the leading reasoning model, OpenAI O1. The datasets and code will be released in: this https URL . 

---
# Bimodal Connection Attention Fusion for Speech Emotion Recognition 

**Authors**: Jiachen Luo, Huy Phan, Lin Wang, Joshua D. Reiss  

**Link**: [PDF](https://arxiv.org/pdf/2503.05858)  

**Abstract**: Multi-modal emotion recognition is challenging due to the difficulty of extracting features that capture subtle emotional differences. Understanding multi-modal interactions and connections is key to building effective bimodal speech emotion recognition systems. In this work, we propose Bimodal Connection Attention Fusion (BCAF) method, which includes three main modules: the interactive connection network, the bimodal attention network, and the correlative attention network. The interactive connection network uses an encoder-decoder architecture to model modality connections between audio and text while leveraging modality-specific features. The bimodal attention network enhances semantic complementation and exploits intra- and inter-modal interactions. The correlative attention network reduces cross-modal noise and captures correlations between audio and text. Experiments on the MELD and IEMOCAP datasets demonstrate that the proposed BCAF method outperforms existing state-of-the-art baselines. 

---
# DreamNet: A Multimodal Framework for Semantic and Emotional Analysis of Sleep Narratives 

**Authors**: Tapasvi Panchagnula  

**Link**: [PDF](https://arxiv.org/pdf/2503.05778)  

**Abstract**: Dream narratives provide a unique window into human cognition and emotion, yet their systematic analysis using artificial intelligence has been underexplored. We introduce DreamNet, a novel deep learning framework that decodes semantic themes and emotional states from textual dream reports, optionally enhanced with REM-stage EEG data. Leveraging a transformer-based architecture with multimodal attention, DreamNet achieves 92.1% accuracy and 88.4% F1-score in text-only mode (DNet-T) on a curated dataset of 1,500 anonymized dream narratives, improving to 99.0% accuracy and 95.2% F1-score with EEG integration (DNet-M). Strong dream-emotion correlations (e.g., falling-anxiety, r = 0.91, p < 0.01) highlight its potential for mental health diagnostics, cognitive science, and personalized therapy. This work provides a scalable tool, a publicly available enriched dataset, and a rigorous methodology, bridging AI and psychological research. 

---
# A Zero-shot Learning Method Based on Large Language Models for Multi-modal Knowledge Graph Embedding 

**Authors**: Bingchen Liu, Jingchen Li, Naixing Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07202)  

**Abstract**: Zero-shot learning (ZL) is crucial for tasks involving unseen categories, such as natural language processing, image classification, and cross-lingual transfer. Current applications often fail to accurately infer and handle new relations or entities involving unseen categories, severely limiting their scalability and practicality in open-domain scenarios. ZL learning faces the challenge of effectively transferring semantic information of unseen categories in multi-modal knowledge graph (MMKG) embedding representation learning. In this paper, we propose ZSLLM, a framework for zero-shot embedding learning of MMKGs using large language models (LLMs). We leverage textual modality information of unseen categories as prompts to fully utilize the reasoning capabilities of LLMs, enabling semantic information transfer across different modalities for unseen categories. Through model-based learning, the embedding representation of unseen categories in MMKG is enhanced. Extensive experiments conducted on multiple real-world datasets demonstrate the superiority of our approach compared to state-of-the-art methods. 

---
# ProJudge: A Multi-Modal Multi-Discipline Benchmark and Instruction-Tuning Dataset for MLLM-based Process Judges 

**Authors**: Jiaxin Ai, Pengfei Zhou, Zhaopan Xu, Ming Li, Fanrui Zhang, Zizhen Li, Jianwen Sun, Yukang Feng, Baojin Huang, Zhongyuan Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06553)  

**Abstract**: As multi-modal large language models (MLLMs) frequently exhibit errors when solving scientific problems, evaluating the validity of their reasoning processes is critical for ensuring reliability and uncovering fine-grained model weaknesses. Since human evaluation is laborious and costly, prompting MLLMs as automated process judges has become a common practice. However, the reliability of these model-based judges remains uncertain. To address this, we introduce ProJudgeBench, the first comprehensive benchmark specifically designed for evaluating abilities of MLLM-based process judges. ProJudgeBench comprises 2,400 test cases and 50,118 step-level labels, spanning four scientific disciplines with diverse difficulty levels and multi-modal content. In ProJudgeBench, each step is meticulously annotated by human experts for correctness, error type, and explanation, enabling a systematic evaluation of judges' capabilities to detect, classify and diagnose errors. Evaluation on ProJudgeBench reveals a significant performance gap between open-source and proprietary models. To bridge this gap, we further propose ProJudge-173k, a large-scale instruction-tuning dataset, and a Dynamic Dual-Phase fine-tuning strategy that encourages models to explicitly reason through problem-solving before assessing solutions. Both contributions significantly enhance the process evaluation capabilities of open-source models. All the resources will be released to foster future research of reliable multi-modal process evaluation. 

---
# When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning 

**Authors**: Junwei Luo, Yingying Zhang, Xue Yang, Kang Wu, Qi Zhu, Lei Liang, Jingdong Chen, Yansheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07588)  

**Abstract**: Efficient vision-language understanding of large Remote Sensing Images (RSIs) is meaningful but challenging. Current Large Vision-Language Models (LVLMs) typically employ limited pre-defined grids to process images, leading to information loss when handling gigapixel RSIs. Conversely, using unlimited grids significantly increases computational costs. To preserve image details while reducing computational complexity, we propose a text-guided token pruning method with Dynamic Image Pyramid (DIP) integration. Our method introduces: (i) a Region Focus Module (RFM) that leverages text-aware region localization capability to identify critical vision tokens, and (ii) a coarse-to-fine image tile selection and vision token pruning strategy based on DIP, which is guided by RFM outputs and avoids directly processing the entire large imagery. Additionally, existing benchmarks for evaluating LVLMs' perception ability on large RSI suffer from limited question diversity and constrained image sizes. We construct a new benchmark named LRS-VQA, which contains 7,333 QA pairs across 8 categories, with image length up to 27,328 pixels. Our method outperforms existing high-resolution strategies on four datasets using the same data. Moreover, compared to existing token reduction methods, our approach demonstrates higher efficiency under high-resolution settings. Dataset and code are in this https URL. 

---
# Robusto-1 Dataset: Comparing Humans and VLMs on real out-of-distribution Autonomous Driving VQA from Peru 

**Authors**: Dunant Cusipuma, David Ortega, Victor Flores-Benites, Arturo Deza  

**Link**: [PDF](https://arxiv.org/pdf/2503.07587)  

**Abstract**: As multimodal foundational models start being deployed experimentally in Self-Driving cars, a reasonable question we ask ourselves is how similar to humans do these systems respond in certain driving situations -- especially those that are out-of-distribution? To study this, we create the Robusto-1 dataset that uses dashcam video data from Peru, a country with one of the worst (aggressive) drivers in the world, a high traffic index, and a high ratio of bizarre to non-bizarre street objects likely never seen in training. In particular, to preliminarly test at a cognitive level how well Foundational Visual Language Models (VLMs) compare to Humans in Driving, we move away from bounding boxes, segmentation maps, occupancy maps or trajectory estimation to multi-modal Visual Question Answering (VQA) comparing both humans and machines through a popular method in systems neuroscience known as Representational Similarity Analysis (RSA). Depending on the type of questions we ask and the answers these systems give, we will show in what cases do VLMs and Humans converge or diverge allowing us to probe on their cognitive alignment. We find that the degree of alignment varies significantly depending on the type of questions asked to each type of system (Humans vs VLMs), highlighting a gap in their alignment. 

---
# DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models 

**Authors**: Shenyu Zhang, Jiaguo Tian, Zhengbang Zhu, Shan Huang, Jucheng Yang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05808)  

**Abstract**: Microscopic traffic simulation has become an important tool for autonomous driving training and testing. Although recent data-driven approaches advance realistic behavior generation, their learning still relies primarily on a single real-world dataset, which limits their diversity and thereby hinders downstream algorithm optimization. In this paper, we propose DriveGen, a novel traffic simulation framework with large models for more diverse traffic generation that supports further customized designs. DriveGen consists of two internal stages: the initialization stage uses large language model and retrieval technique to generate map and vehicle assets; the rollout stage outputs trajectories with selected waypoint goals from visual language model and a specific designed diffusion planner. Through this two-staged process, DriveGen fully utilizes large models' high-level cognition and reasoning of driving behavior, obtaining greater diversity beyond datasets while maintaining high realism. To support effective downstream optimization, we additionally develop DriveGen-CS, an automatic corner case generation pipeline that uses failures of the driving algorithm as additional prompt knowledge for large models without the need for retraining or fine-tuning. Experiments show that our generated scenarios and corner cases have a superior performance compared to state-of-the-art baselines. Downstream experiments further verify that the synthesized traffic of DriveGen provides better optimization of the performance of typical driving algorithms, demonstrating the effectiveness of our framework. 

---
# COMODO: Cross-Modal Video-to-IMU Distillation for Efficient Egocentric Human Activity Recognition 

**Authors**: Baiyu Chen, Wilson Wongso, Zechen Li, Yonchanok Khaokaew, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07259)  

**Abstract**: Egocentric video-based models capture rich semantic information and have demonstrated strong performance in human activity recognition (HAR). However, their high power consumption, privacy concerns, and dependence on lighting conditions limit their feasibility for continuous on-device recognition. In contrast, inertial measurement unit (IMU) sensors offer an energy-efficient and privacy-preserving alternative, yet they suffer from limited large-scale annotated datasets, leading to weaker generalization in downstream tasks. To bridge this gap, we propose COMODO, a cross-modal self-supervised distillation framework that transfers rich semantic knowledge from the video modality to the IMU modality without requiring labeled annotations. COMODO leverages a pretrained and frozen video encoder to construct a dynamic instance queue, aligning the feature distributions of video and IMU embeddings. By distilling knowledge from video representations, our approach enables the IMU encoder to inherit rich semantic information from video while preserving its efficiency for real-world applications. Experiments on multiple egocentric HAR datasets demonstrate that COMODO consistently improves downstream classification performance, achieving results comparable to or exceeding fully supervised fine-tuned models. Moreover, COMODO exhibits strong cross-dataset generalization. Benefiting from its simplicity, our method is also generally applicable to various video and time-series pre-trained models, offering the potential to leverage more powerful teacher and student foundation models in future research. The code is available at this https URL . 

---
# Lightweight Multimodal Artificial Intelligence Framework for Maritime Multi-Scene Recognition 

**Authors**: Xinyu Xi, Hua Yang, Shentai Zhang, Yijie Liu, Sijin Sun, Xiuju Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06978)  

**Abstract**: Maritime Multi-Scene Recognition is crucial for enhancing the capabilities of intelligent marine robotics, particularly in applications such as marine conservation, environmental monitoring, and disaster response. However, this task presents significant challenges due to environmental interference, where marine conditions degrade image quality, and the complexity of maritime scenes, which requires deeper reasoning for accurate recognition. Pure vision models alone are insufficient to address these issues. To overcome these limitations, we propose a novel multimodal Artificial Intelligence (AI) framework that integrates image data, textual descriptions and classification vectors generated by a Multimodal Large Language Model (MLLM), to provide richer semantic understanding and improve recognition accuracy. Our framework employs an efficient multimodal fusion mechanism to further enhance model robustness and adaptability in complex maritime environments. Experimental results show that our model achieves 98$\%$ accuracy, surpassing previous SOTA models by 3.5$\%$. To optimize deployment on resource-constrained platforms, we adopt activation-aware weight quantization (AWQ) as a lightweight technique, reducing the model size to 68.75MB with only a 0.5$\%$ accuracy drop while significantly lowering computational overhead. This work provides a high-performance solution for real-time maritime scene recognition, enabling Autonomous Surface Vehicles (ASVs) to support environmental monitoring and disaster response in resource-limited settings. 

---
# Combating Partial Perception Deficit in Autonomous Driving with Multimodal LLM Commonsense 

**Authors**: Yuting Hu, Chenhui Xu, Ruiyang Qin, Dancheng Liu, Amir Nassereldine, Yiyu Shi, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.07020)  

**Abstract**: Partial perception deficits can compromise autonomous vehicle safety by disrupting environmental understanding. Current protocols typically respond with immediate stops or minimal-risk maneuvers, worsening traffic flow and lacking flexibility for rare driving scenarios. In this paper, we propose LLM-RCO, a framework leveraging large language models to integrate human-like driving commonsense into autonomous systems facing perception deficits. LLM-RCO features four key modules: hazard inference, short-term motion planner, action condition verifier, and safety constraint generator. These modules interact with the dynamic driving environment, enabling proactive and context-aware control actions to override the original control policy of autonomous agents. To improve safety in such challenging conditions, we construct DriveLM-Deficit, a dataset of 53,895 video clips featuring deficits of safety-critical objects, complete with annotations for LLM-based hazard inference and motion planning fine-tuning. Extensive experiments in adverse driving conditions with the CARLA simulator demonstrate that systems equipped with LLM-RCO significantly improve driving performance, highlighting its potential for enhancing autonomous driving resilience against adverse perception deficits. Our results also show that LLMs fine-tuned with DriveLM-Deficit can enable more proactive movements instead of conservative stops in the context of perception deficits. 

---
# A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis 

**Authors**: Xiang Liu, Zhaoxiang Liu, Huan Hu, Zezhou Chen, Kohou Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.06973)  

**Abstract**: While conversational generative AI has shown considerable potential in enhancing decision-making for agricultural professionals, its exploration has predominantly been anchored in text-based interactions. The evolution of multimodal conversational AI, leveraging vast amounts of image-text data from diverse sources, marks a significant stride forward. However, the application of such advanced vision-language models in the agricultural domain, particularly for crop disease diagnosis, remains underexplored. In this work, we present the crop disease domain multimodal (CDDM) dataset, a pioneering resource designed to advance the field of agricultural research through the application of multimodal learning techniques. The dataset comprises 137,000 images of various crop diseases, accompanied by 1 million question-answer pairs that span a broad spectrum of agricultural knowledge, from disease identification to management practices. By integrating visual and textual data, CDDM facilitates the development of sophisticated question-answering systems capable of providing precise, useful advice to farmers and agricultural professionals. We demonstrate the utility of the dataset by finetuning state-of-the-art multimodal models, showcasing significant improvements in crop disease diagnosis. Specifically, we employed a novel finetuning strategy that utilizes low-rank adaptation (LoRA) to finetune the visual encoder, adapter and language model simultaneously. Our contributions include not only the dataset but also a finetuning strategy and a benchmark to stimulate further research in agricultural technology, aiming to bridge the gap between advanced AI techniques and practical agricultural applications. The dataset is available at https: //github.com/UnicomAI/UnicomBenchmark/tree/main/CDDMBench. 

---
# Large Language Model Guided Progressive Feature Alignment for Multimodal UAV Object Detection 

**Authors**: Wentao Wu, Chenglong Li, Xiao Wang, Bin Luo, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06948)  

**Abstract**: Existing multimodal UAV object detection methods often overlook the impact of semantic gaps between modalities, which makes it difficult to achieve accurate semantic and spatial alignments, limiting detection performance. To address this problem, we propose a Large Language Model (LLM) guided Progressive feature Alignment Network called LPANet, which leverages the semantic features extracted from a large language model to guide the progressive semantic and spatial alignment between modalities for multimodal UAV object detection. To employ the powerful semantic representation of LLM, we generate the fine-grained text descriptions of each object category by ChatGPT and then extract the semantic features using the large language model MPNet. Based on the semantic features, we guide the semantic and spatial alignments in a progressive manner as follows. First, we design the Semantic Alignment Module (SAM) to pull the semantic features and multimodal visual features of each object closer, alleviating the semantic differences of objects between modalities. Second, we design the Explicit Spatial alignment Module (ESM) by integrating the semantic relations into the estimation of feature-level offsets, alleviating the coarse spatial misalignment between modalities. Finally, we design the Implicit Spatial alignment Module (ISM), which leverages the cross-modal correlations to aggregate key features from neighboring regions to achieve implicit spatial alignment. Comprehensive experiments on two public multimodal UAV object detection datasets demonstrate that our approach outperforms state-of-the-art multimodal UAV object detectors. 

---
# Improving cognitive diagnostics in pathology: a deep learning approach for augmenting perceptional understanding of histopathology images 

**Authors**: Xiaoqian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06894)  

**Abstract**: In Recent Years, Digital Technologies Have Made Significant Strides In Augmenting-Human-Health, Cognition, And Perception, Particularly Within The Field Of Computational-Pathology. This Paper Presents A Novel Approach To Enhancing The Analysis Of Histopathology Images By Leveraging A Mult-modal-Model That Combines Vision Transformers (Vit) With Gpt-2 For Image Captioning. The Model Is Fine-Tuned On The Specialized Arch-Dataset, Which Includes Dense Image Captions Derived From Clinical And Academic Resources, To Capture The Complexities Of Pathology Images Such As Tissue Morphologies, Staining Variations, And Pathological Conditions. By Generating Accurate, Contextually Captions, The Model Augments The Cognitive Capabilities Of Healthcare Professionals, Enabling More Efficient Disease Classification, Segmentation, And Detection. The Model Enhances The Perception Of Subtle Pathological Features In Images That Might Otherwise Go Unnoticed, Thereby Improving Diagnostic Accuracy. Our Approach Demonstrates The Potential For Digital Technologies To Augment Human Cognitive Abilities In Medical Image Analysis, Providing Steps Toward More Personalized And Accurate Healthcare Outcomes. 

---
# Towards a Multimodal MRI-Based Foundation Model for Multi-Level Feature Exploration in Segmentation, Molecular Subtyping, and Grading of Glioma 

**Authors**: Somayeh Farahani, Marjaneh Hejazi, Antonio Di Ieva, Emad Fatemizadeh, Sidong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06828)  

**Abstract**: Accurate, noninvasive glioma characterization is crucial for effective clinical management. Traditional methods, dependent on invasive tissue sampling, often fail to capture the spatial heterogeneity of the tumor. While deep learning has improved segmentation and molecular profiling, few approaches simultaneously integrate tumor morphology and molecular features. Foundation deep learning models, which learn robust, task-agnostic representations from large-scale datasets, hold great promise but remain underutilized in glioma imaging biomarkers. We propose the Multi-Task SWIN-UNETR (MTS-UNET) model, a novel foundation-based framework built on the BrainSegFounder model, pretrained on large-scale neuroimaging data. MTS-UNET simultaneously performs glioma segmentation, histological grading, and molecular subtyping (IDH mutation and 1p/19q co-deletion). It incorporates two key modules: Tumor-Aware Feature Encoding (TAFE) for multi-scale, tumor-focused feature extraction and Cross-Modality Differential (CMD) for highlighting subtle T2-FLAIR mismatch signals associated with IDH mutation. The model was trained and validated on a diverse, multi-center cohort of 2,249 glioma patients from seven public datasets. MTS-UNET achieved a mean Dice score of 84% for segmentation, along with AUCs of 90.58% for IDH mutation, 69.22% for 1p/19q co-deletion prediction, and 87.54% for grading, significantly outperforming baseline models (p<=0.05). Ablation studies validated the essential contributions of the TAFE and CMD modules and demonstrated the robustness of the framework. The foundation-based MTS-UNET model effectively integrates tumor segmentation with multi-level classification, exhibiting strong generalizability across diverse MRI datasets. This framework shows significant potential for advancing noninvasive, personalized glioma management by improving predictive accuracy and interpretability. 

---
# Multimodal AI-driven Biomarker for Early Detection of Cancer Cachexia 

**Authors**: Sabeen Ahmed, Nathan Parker, Margaret Park, Evan W. Davis, Jennifer B. Permuth, Matthew B. Schabath, Yasin Yilmaz, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2503.06797)  

**Abstract**: Cancer cachexia is a multifactorial syndrome characterized by progressive muscle wasting, metabolic dysfunction, and systemic inflammation, leading to reduced quality of life and increased mortality. Despite extensive research, no single definitive biomarker exists, as cachexia-related indicators such as serum biomarkers, skeletal muscle measurements, and metabolic abnormalities often overlap with other conditions. Existing composite indices, including the Cancer Cachexia Index (CXI), Modified CXI (mCXI), and Cachexia Score (CASCO), integrate multiple biomarkers but lack standardized thresholds, limiting their clinical utility. This study proposes a multimodal AI-based biomarker for early cancer cachexia detection, leveraging open-source large language models (LLMs) and foundation models trained on medical data. The approach integrates heterogeneous patient data, including demographics, disease status, lab reports, radiological imaging (CT scans), and clinical notes, using a machine learning framework that can handle missing data. Unlike previous AI-based models trained on curated datasets, this method utilizes routinely collected clinical data, enhancing real-world applicability. Additionally, the model incorporates confidence estimation, allowing the identification of cases requiring expert review for precise clinical interpretation. Preliminary findings demonstrate that integrating multiple data modalities improves cachexia prediction accuracy at the time of cancer diagnosis. The AI-based biomarker dynamically adapts to patient-specific factors such as age, race, ethnicity, weight, cancer type, and stage, avoiding the limitations of fixed-threshold biomarkers. This multimodal AI biomarker provides a scalable and clinically viable solution for early cancer cachexia detection, facilitating personalized interventions and potentially improving treatment outcomes and patient survival. 

---
# Towards Fine-Grained Video Question Answering 

**Authors**: Wei Dai, Alan Luo, Zane Durante, Debadutta Dash, Arnold Milstein, Kevin Schulman, Ehsan Adeli, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2503.06820)  

**Abstract**: In the rapidly evolving domain of video understanding, Video Question Answering (VideoQA) remains a focal point. However, existing datasets exhibit gaps in temporal and spatial granularity, which consequently limits the capabilities of existing VideoQA methods. This paper introduces the Multi-Object Multi-Actor Question Answering (MOMA-QA) dataset, which is designed to address these shortcomings by emphasizing temporal localization, spatial relationship reasoning, and entity-centric queries. With ground truth scene graphs and temporal interval annotations, MOMA-QA is ideal for developing models for fine-grained video understanding. Furthermore, we present a novel video-language model, SGVLM, which incorporates a scene graph predictor, an efficient frame retriever, and a pre-trained large language model for temporal localization and fine-grained relationship understanding. Evaluations on MOMA-QA and other public datasets demonstrate the superior performance of our model, setting new benchmarks for VideoQA. 

---
# SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation 

**Authors**: Zisheng Chen, Chunwei Wang, Xiuwei Chen, Hang Xu, Jianhua Han, Xiandan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06764)  

**Abstract**: We present SemHiTok, a unified image Tokenizer via Semantic-Guided Hierarchical codebook that provides consistent discrete feature representations for multimodal understanding and generation tasks. Recently, unified multimodal large models (MLLMs) for understanding and generation have sparked exploration within research community. Previous works attempt to train a unified image tokenizer by combining loss functions for semantic feature reconstruction and pixel reconstruction. However, due to the differing levels of features prioritized by multimodal understanding and generation tasks, joint training methods face significant challenges in achieving a good trade-off. SemHiTok addresses this challenge through Semantic-Guided Hierarchical codebook which builds texture sub-codebooks on pre-trained semantic codebook. This design decouples the training of semantic reconstruction and pixel reconstruction and equips the tokenizer with low-level texture feature extraction capability without degradation of high-level semantic feature extraction ability. Our experiments demonstrate that SemHiTok achieves state-of-the-art rFID score at 256X256resolution compared to other unified tokenizers, and exhibits competitive performance on multimodal understanding and generation tasks. 

---
# ACAI for SBOs: AI Co-creation for Advertising and Inspiration for Small Business Owners 

**Authors**: Nimisha Karnatak, Adrien Baranes, Rob Marchant, Triona Butler, Kristen Olson  

**Link**: [PDF](https://arxiv.org/pdf/2503.06729)  

**Abstract**: Small business owners (SBOs) often lack the resources and design experience needed to produce high-quality advertisements. To address this, we developed ACAI (AI Co-Creation for Advertising and Inspiration), an GenAI-powered multimodal advertisement creation tool, and conducted a user study with 16 SBOs in London to explore their perceptions of and interactions with ACAI in advertisement creation. Our findings reveal that structured inputs enhance user agency and control while improving AI outputs by facilitating better brand alignment, enhancing AI transparency, and offering scaffolding that assists novice designers, such as SBOs, in formulating prompts. We also found that ACAI's multimodal interface bridges the design skill gap for SBOs with a clear advertisement vision, but who lack the design jargon necessary for effective prompting. Building on our findings, we propose three capabilities: contextual intelligence, adaptive interactions, and data management, with corresponding design recommendations to advance the co-creative attributes of AI-mediated design tools. 

---
# DiffCLIP: Differential Attention Meets CLIP 

**Authors**: Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2503.06626)  

**Abstract**: We propose DiffCLIP, a novel vision-language model that extends the differential attention mechanism to CLIP architectures. Differential attention was originally developed for large language models to amplify relevant context while canceling out noisy information. In this work, we integrate this mechanism into CLIP's dual encoder (image and text) framework. With minimal additional parameters, DiffCLIP achieves superior performance on image-text understanding tasks. Across zero-shot classification, retrieval, and robustness benchmarks, DiffCLIP consistently outperforms baseline CLIP models. Notably, these gains come with negligible computational overhead, demonstrating that differential attention can significantly enhance multi-modal representations without sacrificing efficiency. Code can be found at this https URL. 

---
# ARMOR v0.1: Empowering Autoregressive Multimodal Understanding Model with Interleaved Multimodal Generation via Asymmetric Synergy 

**Authors**: Jianwen Sun, Yukang Feng, Chuanhao Li, Fanrui Zhang, Zizhen Li, Jiaxin Ai, Sizhuo Zhou, Yu Dai, Shenglin Zhang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06542)  

**Abstract**: Unified models (UniMs) for multimodal understanding and generation have recently received much attention in the area of vision and language. Existing UniMs are designed to simultaneously learn both multimodal understanding and generation capabilities, demanding substantial computational resources, and often struggle to generate interleaved text-image. We present ARMOR, a resource-efficient and pure autoregressive framework that achieves both understanding and generation by fine-tuning existing multimodal large language models (MLLMs). Specifically, ARMOR extends existing MLLMs from three perspectives: (1) For model architecture, an asymmetric encoder-decoder architecture with a forward-switching mechanism is introduced to unify embedding space integrating textual and visual modalities for enabling natural text-image interleaved generation with minimal computational overhead. (2) For training data, a meticulously curated, high-quality interleaved dataset is collected for fine-tuning MLLMs. (3) For the training algorithm, we propose a ``what or how to generate" algorithm to empower existing MLLMs with multimodal generation capabilities while preserving their multimodal understanding capabilities, through three progressive training stages based on the collected dataset. Experimental results demonstrate that ARMOR upgrades existing MLLMs to UniMs with promising image generation capabilities, using limited training resources. Our code will be released soon at this https URL. 

---
# From Motion Signals to Insights: A Unified Framework for Student Behavior Analysis and Feedback in Physical Education Classes 

**Authors**: Xian Gao, Jiacheng Ruan, Jingsheng Gao, Mingye Xie, Zongyun Zhang, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06525)  

**Abstract**: Analyzing student behavior in educational scenarios is crucial for enhancing teaching quality and student engagement. Existing AI-based models often rely on classroom video footage to identify and analyze student behavior. While these video-based methods can partially capture and analyze student actions, they struggle to accurately track each student's actions in physical education classes, which take place in outdoor, open spaces with diverse activities, and are challenging to generalize to the specialized technical movements involved in these settings. Furthermore, current methods typically lack the ability to integrate specialized pedagogical knowledge, limiting their ability to provide in-depth insights into student behavior and offer feedback for optimizing instructional design. To address these limitations, we propose a unified end-to-end framework that leverages human activity recognition technologies based on motion signals, combined with advanced large language models, to conduct more detailed analyses and feedback of student behavior in physical education classes. Our framework begins with the teacher's instructional designs and the motion signals from students during physical education sessions, ultimately generating automated reports with teaching insights and suggestions for improving both learning and class instructions. This solution provides a motion signal-based approach for analyzing student behavior and optimizing instructional design tailored to physical education classes. Experimental results demonstrate that our framework can accurately identify student behaviors and produce meaningful pedagogical insights. 

---
# PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training 

**Authors**: Cong Chen, Mingyu Liu, Chenchen Jing, Yizhou Zhou, Fengyun Rao, Hao Chen, Bo Zhang, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.06486)  

**Abstract**: This paper aims to address the challenge of hallucinations in Multimodal Large Language Models (MLLMs) particularly for dense image captioning tasks. To tackle the challenge, we identify the current lack of a metric that finely measures the caption quality in concept level. We hereby introduce HalFscore, a novel metric built upon the language graph and is designed to evaluate both the accuracy and completeness of dense captions at a granular level. Additionally, we identify the root cause of hallucination as the model's over-reliance on its language prior. To address this, we propose PerturboLLaVA, which reduces the model's reliance on the language prior by incorporating adversarially perturbed text during training. This method enhances the model's focus on visual inputs, effectively reducing hallucinations and producing accurate, image-grounded descriptions without incurring additional computational overhead. PerturboLLaVA significantly improves the fidelity of generated captions, outperforming existing approaches in handling multimodal hallucinations and achieving improved performance across general multimodal benchmarks. 

---
# PDB: Not All Drivers Are the Same -- A Personalized Dataset for Understanding Driving Behavior 

**Authors**: Chuheng Wei, Ziye Qin, Siyan Li, Ziyan Zhang, Xuanpeng Zhao, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Matthew J. Barth, Guoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06477)  

**Abstract**: Driving behavior is inherently personal, influenced by individual habits, decision-making styles, and physiological states. However, most existing datasets treat all drivers as homogeneous, overlooking driver-specific variability. To address this gap, we introduce the Personalized Driving Behavior (PDB) dataset, a multi-modal dataset designed to capture personalization in driving behavior under naturalistic driving conditions. Unlike conventional datasets, PDB minimizes external influences by maintaining consistent routes, vehicles, and lighting conditions across sessions. It includes sources from 128-line LiDAR, front-facing camera video, GNSS, 9-axis IMU, CAN bus data (throttle, brake, steering angle), and driver-specific signals such as facial video and heart rate. The dataset features 12 participants, approximately 270,000 LiDAR frames, 1.6 million images, and 6.6 TB of raw sensor data. The processed trajectory dataset consists of 1,669 segments, each spanning 10 seconds with a 0.2-second interval. By explicitly capturing drivers' behavior, PDB serves as a unique resource for human factor analysis, driver identification, and personalized mobility applications, contributing to the development of human-centric intelligent transportation systems. 

---
# Heterogeneous bimodal attention fusion for speech emotion recognition 

**Authors**: Jiachen Luo, Huy Phan, Lin Wang, Joshua Reiss  

**Link**: [PDF](https://arxiv.org/pdf/2503.06405)  

**Abstract**: Multi-modal emotion recognition in conversations is a challenging problem due to the complex and complementary interactions between different modalities. Audio and textual cues are particularly important for understanding emotions from a human perspective. Most existing studies focus on exploring interactions between audio and text modalities at the same representation level. However, a critical issue is often overlooked: the heterogeneous modality gap between low-level audio representations and high-level text representations. To address this problem, we propose a novel framework called Heterogeneous Bimodal Attention Fusion (HBAF) for multi-level multi-modal interaction in conversational emotion recognition. The proposed method comprises three key modules: the uni-modal representation module, the multi-modal fusion module, and the inter-modal contrastive learning module. The uni-modal representation module incorporates contextual content into low-level audio representations to bridge the heterogeneous multi-modal gap, enabling more effective fusion. The multi-modal fusion module uses dynamic bimodal attention and a dynamic gating mechanism to filter incorrect cross-modal relationships and fully exploit both intra-modal and inter-modal interactions. Finally, the inter-modal contrastive learning module captures complex absolute and relative interactions between audio and text modalities. Experiments on the MELD and IEMOCAP datasets demonstrate that the proposed HBAF method outperforms existing state-of-the-art baselines. 

---
# Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding 

**Authors**: Seil Kang, Jinyeong Kim, Junhyeok Kim, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06287)  

**Abstract**: Visual grounding seeks to localize the image region corresponding to a free-form text description. Recently, the strong multimodal capabilities of Large Vision-Language Models (LVLMs) have driven substantial improvements in visual grounding, though they inevitably require fine-tuning and additional model components to explicitly generate bounding boxes or segmentation masks. However, we discover that a few attention heads in frozen LVLMs demonstrate strong visual grounding capabilities. We refer to these heads, which consistently capture object locations related to text semantics, as localization heads. Using localization heads, we introduce a straightforward and effective training-free visual grounding framework that utilizes text-to-image attention maps from localization heads to identify the target objects. Surprisingly, only three out of thousands of attention heads are sufficient to achieve competitive localization performance compared to existing LVLM-based visual grounding methods that require fine-tuning. Our findings suggest that LVLMs can innately ground objects based on a deep comprehension of the text-image relationship, as they implicitly focus on relevant image regions to generate informative text outputs. All the source codes will be made available to the public. 

---
# Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models? 

**Authors**: Kun Xiang, Zhili Liu, Zihao Jiang, Yunshuang Nie, Kaixin Cai, Yiyang Yin, Runhui Huang, Haoxiang Fan, Hanhui Li, Weiran Huang, Yihan Zeng, Yu-Jie Yuan, Jianhua Han, Lanqing Hong, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06252)  

**Abstract**: In this paper, we address the challenging task of multimodal mathematical reasoning by incorporating the ability of "slow thinking" into multimodal large language models (MLLMs). Our core idea is that different levels of reasoning abilities can be combined dynamically to tackle questions with different complexity. To this end, we propose a paradigm of Self-structured Chain of Thought (SCoT), which is composed of minimal semantic atomic steps. Different from existing methods that rely on structured templates or free-form paradigms, our method can not only generate cognitive CoT structures for various complex tasks but also mitigates the phenomenon of overthinking. To introduce structured reasoning capabilities into visual understanding models, we further design a novel AtomThink framework with four key modules, including (i) a data engine to generate high-quality multimodal reasoning paths; (ii) a supervised fine-tuning process with serialized inference data; (iii) a policy-guided multi-turn inference method; and (iv) an atomic capability metric to evaluate the single step utilization rate. We conduct extensive experiments to show that the proposed AtomThink significantly improves the performance of baseline MLLMs, achieving more than 10\% average accuracy gains on MathVista and MathVerse. Compared to state-of-the-art structured CoT approaches, our method not only achieves higher accuracy but also improves data utilization by 5 times and boosts inference efficiency by 85.3\%. Our code is now public available in this https URL. 

---
# From Captions to Rewards (CAREVL): Leveraging Large Language Model Experts for Enhanced Reward Modeling in Large Vision-Language Models 

**Authors**: Muzhi Dai, Jiashuo Sun, Zhiyuan Zhao, Shixuan Liu, Rui Li, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06260)  

**Abstract**: Aligning large vision-language models (LVLMs) with human preferences is challenging due to the scarcity of fine-grained, high-quality, and multimodal preference data without human annotations. Existing methods relying on direct distillation often struggle with low-confidence data, leading to suboptimal performance. To address this, we propose CAREVL, a novel method for preference reward modeling by reliably using both high- and low-confidence data. First, a cluster of auxiliary expert models (textual reward models) innovatively leverages image captions as weak supervision signals to filter high-confidence data. The high-confidence data are then used to fine-tune the LVLM. Second, low-confidence data are used to generate diverse preference samples using the fine-tuned LVLM. These samples are then scored and selected to construct reliable chosen-rejected pairs for further training. CAREVL achieves performance improvements over traditional distillation-based methods on VL-RewardBench and MLLM-as-a-Judge benchmark, demonstrating its effectiveness. The code will be released soon. 

---
# Treble Counterfactual VLMs: A Causal Approach to Hallucination 

**Authors**: Li Li, Jiashu Qu, Yuxiao Zhou, Yuehan Qin, Tiankai Yang, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06169)  

**Abstract**: Vision-Language Models (VLMs) have advanced multi-modal tasks like image captioning, visual question answering, and reasoning. However, they often generate hallucinated outputs inconsistent with the visual context or prompt, limiting reliability in critical applications like autonomous driving and medical imaging. Existing studies link hallucination to statistical biases, language priors, and biased feature learning but lack a structured causal understanding. In this work, we introduce a causal perspective to analyze and mitigate hallucination in VLMs. We hypothesize that hallucination arises from unintended direct influences of either the vision or text modality, bypassing proper multi-modal fusion. To address this, we construct a causal graph for VLMs and employ counterfactual analysis to estimate the Natural Direct Effect (NDE) of vision, text, and their cross-modal interaction on the output. We systematically identify and mitigate these unintended direct effects to ensure that responses are primarily driven by genuine multi-modal fusion. Our approach consists of three steps: (1) designing structural causal graphs to distinguish correct fusion pathways from spurious modality shortcuts, (2) estimating modality-specific and cross-modal NDE using perturbed image representations, hallucinated text embeddings, and degraded visual inputs, and (3) implementing a test-time intervention module to dynamically adjust the model's dependence on each modality. Experimental results demonstrate that our method significantly reduces hallucination while preserving task performance, providing a robust and interpretable framework for improving VLM reliability. To enhance accessibility and reproducibility, our code is publicly available at this https URL. 

---
# Multi-modal expressive personality recognition in data non-ideal audiovisual based on multi-scale feature enhancement and modal augment 

**Authors**: Weixuan Kong, Jinpeng Yu, Zijun Li, Hanwei Liu, Jiqing Qu, Hui Xiao, Xuefeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06108)  

**Abstract**: Automatic personality recognition is a research hotspot in the intersection of computer science and psychology, and in human-computer interaction, personalised has a wide range of applications services and other scenarios. In this paper, an end-to-end multimodal performance personality is established for both visual and auditory modal datarecognition network , and the through feature-level fusion , which effectively of the two modalities is carried out the cross-attention mechanismfuses the features of the two modal data; and a is proposed multiscale feature enhancement modalitiesmodule , which enhances for visual and auditory boththe expression of the information of effective the features and suppresses the interference of the redundant information. In addition, during the training process, this paper proposes a modal enhancement training strategy to simulate non-ideal such as modal loss and noise interferencedata situations , which enhances the adaptability ofand the model to non-ideal data scenarios improves the robustness of the model. Experimental results show that the method proposed in this paper is able to achieve an average Big Five personality accuracy of , which outperforms existing 0.916 on the personality analysis dataset ChaLearn First Impressionother methods based on audiovisual and audio-visual both modalities. The ablation experiments also validate our proposed , respectivelythe contribution of module and modality enhancement strategy to the model performance. Finally, we simulate in the inference phase multi-scale feature enhancement six non-ideal data scenarios to verify the modal enhancement strategy's improvement in model robustness. 

---
# UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces 

**Authors**: Baining Zhao, Jianjie Fang, Zichao Dai, Ziyou Wang, Jirong Zha, Weichen Zhang, Chen Gao, Yue Wang, Jinqiang Cui, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06157)  

**Abstract**: Large multimodal models exhibit remarkable intelligence, yet their embodied cognitive abilities during motion in open-ended urban 3D space remain to be explored. We introduce a benchmark to evaluate whether video-large language models (Video-LLMs) can naturally process continuous first-person visual observations like humans, enabling recall, perception, reasoning, and navigation. We have manually control drones to collect 3D embodied motion video data from real-world cities and simulated environments, resulting in 1.5k video clips. Then we design a pipeline to generate 5.2k multiple-choice questions. Evaluations of 17 widely-used Video-LLMs reveal current limitations in urban embodied cognition. Correlation analysis provides insight into the relationships between different tasks, showing that causal reasoning has a strong correlation with recall, perception, and navigation, while the abilities for counterfactual and associative reasoning exhibit lower correlation with other tasks. We also validate the potential for Sim-to-Real transfer in urban embodiment through fine-tuning. 

---
# Zero-Shot Peg Insertion: Identifying Mating Holes and Estimating SE(2) Poses with Vision-Language Models 

**Authors**: Masaru Yajima, Kei Ota, Asako Kanezaki, Rei Kawakami  

**Link**: [PDF](https://arxiv.org/pdf/2503.06026)  

**Abstract**: Achieving zero-shot peg insertion, where inserting an arbitrary peg into an unseen hole without task-specific training, remains a fundamental challenge in robotics. This task demands a highly generalizable perception system capable of detecting potential holes, selecting the correct mating hole from multiple candidates, estimating its precise pose, and executing insertion despite uncertainties. While learning-based methods have been applied to peg insertion, they often fail to generalize beyond the specific peg-hole pairs encountered during training. Recent advancements in Vision-Language Models (VLMs) offer a promising alternative, leveraging large-scale datasets to enable robust generalization across diverse tasks. Inspired by their success, we introduce a novel zero-shot peg insertion framework that utilizes a VLM to identify mating holes and estimate their poses without prior knowledge of their geometry. Extensive experiments demonstrate that our method achieves 90.2% accuracy, significantly outperforming baselines in identifying the correct mating hole across a wide range of previously unseen peg-hole pairs, including 3D-printed objects, toy puzzles, and industrial connectors. Furthermore, we validate the effectiveness of our approach in a real-world connector insertion task on a backpanel of a PC, where our system successfully detects holes, identifies the correct mating hole, estimates its pose, and completes the insertion with a success rate of 88.3%. These results highlight the potential of VLM-driven zero-shot reasoning for enabling robust and generalizable robotic assembly. 

---
# Towards Universal Text-driven CT Image Segmentation 

**Authors**: Yuheng Li, Yuxiang Lai, Maria Thor, Deborah Marshall, Zachary Buchwald, David S. Yu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06030)  

**Abstract**: Computed tomography (CT) is extensively used for accurate visualization and segmentation of organs and lesions. While deep learning models such as convolutional neural networks (CNNs) and vision transformers (ViTs) have significantly improved CT image analysis, their performance often declines when applied to diverse, real-world clinical data. Although foundation models offer a broader and more adaptable solution, their potential is limited due to the challenge of obtaining large-scale, voxel-level annotations for medical images. In response to these challenges, prompting-based models using visual or text prompts have emerged. Visual-prompting methods, such as the Segment Anything Model (SAM), still require significant manual input and can introduce ambiguity when applied to clinical scenarios. Instead, foundation models that use text prompts offer a more versatile and clinically relevant approach. Notably, current text-prompt models, such as the CLIP-Driven Universal Model, are limited to text prompts already encountered during training and struggle to process the complex and diverse scenarios of real-world clinical applications. Instead of fine-tuning models trained from natural imaging, we propose OpenVocabCT, a vision-language model pretrained on large-scale 3D CT images for universal text-driven segmentation. Using the large-scale CT-RATE dataset, we decompose the diagnostic reports into fine-grained, organ-level descriptions using large language models for multi-granular contrastive learning. We evaluate our OpenVocabCT on downstream segmentation tasks across nine public datasets for organ and tumor segmentation, demonstrating the superior performance of our model compared to existing methods. All code, datasets, and models will be publicly released at this https URL. 

---
# A Real-time Multimodal Transformer Neural Network-powered Wildfire Forecasting System 

**Authors**: Qijun Chen, Shaofan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05971)  

**Abstract**: Due to climate change, the extreme wildfire has become one of the most dangerous natural hazards to human civilization. Even though, some wildfires may be initially caused by human activity, but the spread of wildfires is mainly determined by environmental factors, for examples, (1) weather conditions such as temperature, wind direction and intensity, and moisture levels; (2) the amount and types of dry vegetation in a local area, and (3) topographic or local terrian conditions, which affects how much rain an area gets and how fire dynamics will be constrained or faciliated. Thus, to accurately forecast wildfire occurrence has become one of most urgent and taunting environmental challenges in global scale. In this work, we developed a real-time Multimodal Transformer Neural Network Machine Learning model that combines several advanced artificial intelligence techniques and statistical methods to practically forecast the occurrence of wildfire at the precise location in real time, which not only utilizes large scale data information such as hourly weather forecasting data, but also takes into account small scale topographical data such as local terrain condition and local vegetation conditions collecting from Google Earth images to determine the probabilities of wildfire occurrence location at small scale as well as their timing synchronized with weather forecast information. By using the wildfire data in the United States from 1992 to 2015 to train the multimodal transformer neural network, it can predict the probabilities of wildfire occurrence according to the real-time weather forecast and the synchronized Google Earth image data to provide the wildfire occurrence probability in any small location ($100m^2$) within 24 hours ahead. 

---
# Towards Understanding the Use of MLLM-Enabled Applications for Visual Interpretation by Blind and Low Vision People 

**Authors**: Ricardo E. Gonzalez Penuela, Ruiying Hu, Sharon Lin, Tanisha Shende, Shiri Azenkot  

**Link**: [PDF](https://arxiv.org/pdf/2503.05899)  

**Abstract**: Blind and Low Vision (BLV) people have adopted AI-powered visual interpretation applications to address their daily needs. While these applications have been helpful, prior work has found that users remain unsatisfied by their frequent errors. Recently, multimodal large language models (MLLMs) have been integrated into visual interpretation applications, and they show promise for more descriptive visual interpretations. However, it is still unknown how this advancement has changed people's use of these applications. To address this gap, we conducted a two-week diary study in which 20 BLV people used an MLLM-enabled visual interpretation application we developed, and we collected 553 entries. In this paper, we report a preliminary analysis of 60 diary entries from 6 participants. We found that participants considered the application's visual interpretations trustworthy (mean 3.75 out of 5) and satisfying (mean 4.15 out of 5). Moreover, participants trusted our application in high-stakes scenarios, such as receiving medical dosage advice. We discuss our plan to complete our analysis to inform the design of future MLLM-enabled visual interpretation systems. 

---
