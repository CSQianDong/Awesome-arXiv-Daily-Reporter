# Multi-Task Semantic Communications via Large Models 

**Authors**: Wanli Ni, Zhijin Qin, Haofeng Sun, Xiaoming Tao, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.22064)  

**Abstract**: Artificial intelligence (AI) promises to revolutionize the design, optimization and management of next-generation communication systems. In this article, we explore the integration of large AI models (LAMs) into semantic communications (SemCom) by leveraging their multi-modal data processing and generation capabilities. Although LAMs bring unprecedented abilities to extract semantics from raw data, this integration entails multifaceted challenges including high resource demands, model complexity, and the need for adaptability across diverse modalities and tasks. To overcome these challenges, we propose a LAM-based multi-task SemCom (MTSC) architecture, which includes an adaptive model compression strategy and a federated split fine-tuning approach to facilitate the efficient deployment of LAM-based semantic models in resource-limited networks. Furthermore, a retrieval-augmented generation scheme is implemented to synthesize the most recent local and global knowledge bases to enhance the accuracy of semantic extraction and content generation, thereby improving the inference performance. Finally, simulation results demonstrate the efficacy of the proposed LAM-based MTSC architecture, highlighting the performance enhancements across various downstream tasks under varying channel conditions. 

---
# Agent-Centric Personalized Multiple Clustering with Multi-Modal LLMs 

**Authors**: Ziye Chen, Yiqun Duan, Riheng Zhu, Zhenbang Sun, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22241)  

**Abstract**: Personalized multiple clustering aims to generate diverse partitions of a dataset based on different user-specific aspects, rather than a single clustering. It has recently drawn research interest for accommodating varying user preferences. Recent approaches primarily use CLIP embeddings with proxy learning to extract representations biased toward user clustering preferences. However, CLIP primarily focuses on coarse image-text alignment, lacking a deep contextual understanding of user interests. To overcome these limitations, we propose an agent-centric personalized clustering framework that leverages multi-modal large language models (MLLMs) as agents to comprehensively traverse a relational graph to search for clusters based on user interests. Due to the advanced reasoning mechanism of MLLMs, the obtained clusters align more closely with user-defined criteria than those obtained from CLIP-based representations. To reduce computational overhead, we shorten the agents' traversal path by constructing a relational graph using user-interest-biased embeddings extracted by MLLMs. A large number of weakly connected edges can be filtered out based on embedding similarity, facilitating an efficient traversal search for agents. Experimental results show that the proposed method achieves NMI scores of 0.9667 and 0.9481 on the Card Order and Card Suits benchmarks, respectively, largely improving the SOTA model by over 140%. 

---
# Unicorn: Text-Only Data Synthesis for Vision Language Model Training 

**Authors**: Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22655)  

**Abstract**: Training vision-language models (VLMs) typically requires large-scale, high-quality image-text pairs, but collecting or synthesizing such data is costly. In contrast, text data is abundant and inexpensive, prompting the question: can high-quality multimodal training data be synthesized purely from text? To tackle this, we propose a cross-integrated three-stage multimodal data synthesis framework, which generates two datasets: Unicorn-1.2M and Unicorn-471K-Instruction. In Stage 1: Diverse Caption Data Synthesis, we construct 1.2M semantically diverse high-quality captions by expanding sparse caption seeds using large language models (LLMs). In Stage 2: Instruction-Tuning Data Generation, we further process 471K captions into multi-turn instruction-tuning tasks to support complex reasoning. Finally, in Stage 3: Modality Representation Transfer, these textual captions representations are transformed into visual representations, resulting in diverse synthetic image representations. This three-stage process enables us to construct Unicorn-1.2M for pretraining and Unicorn-471K-Instruction for instruction-tuning, without relying on real images. By eliminating the dependency on real images while maintaining data quality and diversity, our framework offers a cost-effective and scalable solution for VLMs training. Code is available at this https URL. 

---
# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users 

**Authors**: Antonia Karamolegkou, Malvina Nikandrou, Georgios Pantazopoulos, Danae Sanchez Villegas, Phillip Rust, Ruchira Dhar, Daniel Hershcovich, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2503.22610)  

**Abstract**: This paper explores the effectiveness of Multimodal Large Language models (MLLMs) as assistive technologies for visually impaired individuals. We conduct a user survey to identify adoption patterns and key challenges users face with such technologies. Despite a high adoption rate of these models, our findings highlight concerns related to contextual understanding, cultural sensitivity, and complex scene understanding, particularly for individuals who may rely solely on them for visual interpretation. Informed by these results, we collate five user-centred tasks with image and video inputs, including a novel task on Optical Braille Recognition. Our systematic evaluation of twelve MLLMs reveals that further advancements are necessary to overcome limitations related to cultural context, multilingual support, Braille reading comprehension, assistive object recognition, and hallucinations. This work provides critical insights into the future direction of multimodal AI for accessibility, underscoring the need for more inclusive, robust, and trustworthy visual assistance technologies. 

---
# Exploiting Mixture-of-Experts Redundancy Unlocks Multimodal Generative Abilities 

**Authors**: Raman Dutt, Harleen Hanspal, Guoxuan Xia, Petru-Daniel Tudosiu, Alexander Black, Yongxin Yang, Steven McDonagh, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2503.22517)  

**Abstract**: In this work, we undertake the challenge of augmenting the existing generative capabilities of pre-trained text-only large language models (LLMs) with multi-modal generation capability while satisfying two core constraints: C1 preserving the preservation of original language generative capabilities with negligible performance degradation, and C2 adhering to a small parameter budget to learn the new modality, ensuring scalability and efficiency. In contrast to current approaches that add dedicated modules, thereby significantly increasing the parameter count, we propose a method that leverages the underutilized capacity inherent in deep models. Specifically, we exploit the parameter redundancy within Mixture-of-Experts (MoEs) as a source of additional capacity for learning a new modality, enabling better parameter efficiency (C1). Moreover, we preserve the original language generation capabilities by applying low-rank adaptation exclusively to the tokens of the new modality (C2). Furthermore, we introduce a novel parameter initialization scheme based on the Gromov-Wasserstein distance to improve convergence and training stability. Through an extensive analysis of the routing mechanism, we uncover the emergence of modality-specific pathways and decreased redundancy within the experts that can efficiently unlock multi-modal generative capabilities. Overall, our method can be seamlessly applied to a wide range of contemporary LLMs, providing a new pathway for transitioning from uni-modal to multi-modal architectures. 

---
# Breaking Language Barriers in Visual Language Models via Multilingual Textual Regularization 

**Authors**: Iñigo Pikabea, Iñaki Lacunza, Oriol Pareras, Carlos Escolano, Aitor Gonzalez-Agirre, Javier Hernando, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2503.22577)  

**Abstract**: Rapid advancements in Visual Language Models (VLMs) have transformed multimodal understanding but are often constrained by generating English responses regardless of the input language. This phenomenon has been termed as Image-induced Fidelity Loss (IFL) and stems from limited multimodal multilingual training data. To address this, we propose a continuous multilingual integration strategy that injects text-only multilingual data during visual instruction tuning, preserving the language model's original multilingual capabilities. Extensive evaluations demonstrate that our approach significantly improves linguistic fidelity across languages without degradation in visual performance. We also explore model merging, which improves language fidelity but comes at the cost of visual performance. In contrast, our core method achieves robust multilingual alignment without trade-offs, offering a scalable and effective path to mitigating IFL for global VLM adoption. 

---
# Learning to Instruct for Visual Instruction Tuning 

**Authors**: Zhihan Zhou, Feng Hong, Jiaan Luo, Jiangchao Yao, Dongsheng Li, Bo Han, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22215)  

**Abstract**: We propose LIT, an advancement of visual instruction tuning (VIT). While VIT equips Multimodal LLMs (MLLMs) with promising multimodal capabilities, the current design choices for VIT often result in overfitting and shortcut learning, potentially degrading performance. This gap arises from an overemphasis on instruction-following abilities, while neglecting the proactive understanding of visual information. Inspired by this, LIT adopts a simple yet effective approach by incorporating the loss function into both the instruction and response sequences. It seamlessly expands the training data, and regularizes the MLLMs from overly relying on language priors. Based on this merit, LIT achieves a significant relative improvement of up to 9% on comprehensive multimodal benchmarks, requiring no additional training data and incurring negligible computational overhead. Surprisingly, LIT attains exceptional fundamental visual capabilities, yielding up to an 18% improvement in captioning performance, while simultaneously alleviating hallucination in MLLMs. 

---
# Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback 

**Authors**: Yuan Meng, Xiangtong Yao, Haihui Ye, Yirui Zhou, Shengqiang Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21969)  

**Abstract**: Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning to enable robots to execute tasks from human commands. However, these methods often suffer from limited generalization, adaptability, and the lack of large-scale specialized datasets, unlike data-rich domains such as computer vision, making long-horizon task execution challenging. To address these gaps, we introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation, leveraging large language models (LLMs) for real-time task planning and execution. DAHLIA employs a dual-tunnel architecture, where an LLM-powered planner collaborates with co-planners to decompose tasks and generate executable plans, while a reporter LLM provides closed-loop feedback, enabling adaptive re-planning and ensuring task recovery from potential failures. Moreover, DAHLIA integrates chain-of-thought (CoT) in task reasoning and temporal abstraction for efficient action execution, enhancing traceability and robustness. Our framework demonstrates state-of-the-art performance across diverse long-horizon tasks, achieving strong generalization in both simulated and real-world scenarios. Videos and code are available at this https URL. 

---
# CMD-HAR: Cross-Modal Disentanglement for Wearable Human Activity Recognition 

**Authors**: Hanyu Liu, Siyao Li, Ying Yu, Yixuan Jiang, Hang Xiao, Jingxi Long, Haotian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21843)  

**Abstract**: Human Activity Recognition (HAR) is a fundamental technology for numerous human - centered intelligent applications. Although deep learning methods have been utilized to accelerate feature extraction, issues such as multimodal data mixing, activity heterogeneity, and complex model deployment remain largely unresolved. The aim of this paper is to address issues such as multimodal data mixing, activity heterogeneity, and complex model deployment in sensor-based human activity recognition. We propose a spatiotemporal attention modal decomposition alignment fusion strategy to tackle the problem of the mixed distribution of sensor data. Key discriminative features of activities are captured through cross-modal spatio-temporal disentangled representation, and gradient modulation is combined to alleviate data heterogeneity. In addition, a wearable deployment simulation system is constructed. We conducted experiments on a large number of public datasets, demonstrating the effectiveness of the model. 

---
# StarFlow: Generating Structured Workflow Outputs From Sketch Images 

**Authors**: Patrice Bechard, Chao Wang, Amirhossein Abaskohi, Juan Rodriguez, Christopher Pal, David Vazquez, Spandana Gella, Sai Rajeswar, Perouz Taslakian  

**Link**: [PDF](https://arxiv.org/pdf/2503.21889)  

**Abstract**: Workflows are a fundamental component of automation in enterprise platforms, enabling the orchestration of tasks, data processing, and system integrations. Despite being widely used, building workflows can be complex, often requiring manual configuration through low-code platforms or visual programming tools. To simplify this process, we explore the use of generative foundation models, particularly vision-language models (VLMs), to automatically generate structured workflows from visual inputs. Translating hand-drawn sketches or computer-generated diagrams into executable workflows is challenging due to the ambiguity of free-form drawings, variations in diagram styles, and the difficulty of inferring execution logic from visual elements. To address this, we introduce StarFlow, a framework for generating structured workflow outputs from sketches using vision-language models. We curate a diverse dataset of workflow diagrams -- including synthetic, manually annotated, and real-world samples -- to enable robust training and evaluation. We finetune and benchmark multiple vision-language models, conducting a series of ablation studies to analyze the strengths and limitations of our approach. Our results show that finetuning significantly enhances structured workflow generation, outperforming large vision-language models on this task. 

---
# A Multi-Modal Knowledge-Enhanced Framework for Vessel Trajectory Prediction 

**Authors**: Haomin Yu, Tianyi Li, Kristian Torp, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21834)  

**Abstract**: Accurate vessel trajectory prediction facilitates improved navigational safety, routing, and environmental protection. However, existing prediction methods are challenged by the irregular sampling time intervals of the vessel tracking data from the global AIS system and the complexity of vessel movement. These aspects render model learning and generalization difficult. To address these challenges and improve vessel trajectory prediction, we propose the multi-modal knowledge-enhanced framework (MAKER) for vessel trajectory prediction. To contend better with the irregular sampling time intervals, MAKER features a Large language model-guided Knowledge Transfer (LKT) module that leverages pre-trained language models to transfer trajectory-specific contextual knowledge effectively. To enhance the ability to learn complex trajectory patterns, MAKER incorporates a Knowledge-based Self-paced Learning (KSL) module. This module employs kinematic knowledge to progressively integrate complex patterns during training, allowing for adaptive learning and enhanced generalization. Experimental results on two vessel trajectory datasets show that MAKER can improve the prediction accuracy of state-of-the-art methods by 12.08%-17.86%. 

---
# M-DocSum: Do LVLMs Genuinely Comprehend Interleaved Image-Text in Document Summarization? 

**Authors**: Haolong Yan, Kaijun Tan, Yeqing Shen, Xin Huang, Zheng Ge, Xiangyu Zhang, Si Li, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21839)  

**Abstract**: We investigate a critical yet under-explored question in Large Vision-Language Models (LVLMs): Do LVLMs genuinely comprehend interleaved image-text in the document? Existing document understanding benchmarks often assess LVLMs using question-answer formats, which are information-sparse and difficult to guarantee the coverage of long-range dependencies. To address this issue, we introduce a novel and challenging Multimodal Document Summarization Benchmark (M-DocSum-Bench), which comprises 500 high-quality arXiv papers, along with interleaved multimodal summaries aligned with human preferences. M-DocSum-Bench is a reference-based generation task and necessitates the generation of interleaved image-text summaries using provided reference images, thereby simultaneously evaluating capabilities in understanding, reasoning, localization, and summarization within complex multimodal document scenarios. To facilitate this benchmark, we develop an automated framework to construct summaries and propose a fine-grained evaluation method called M-DocEval. Moreover, we further develop a robust summarization baseline, i.e., M-DocSum-7B, by progressive two-stage training with diverse instruction and preference data. The extensive results on our M-DocSum-Bench reveal that the leading LVLMs struggle to maintain coherence and accurately integrate information within long and interleaved contexts, often exhibiting confusion between similar images and a lack of robustness. Notably, M-DocSum-7B achieves state-of-the-art performance compared to larger and closed-source models (including GPT-4o, Gemini Pro, Claude-3.5-Sonnet and Qwen2.5-VL-72B, etc.), demonstrating the potential of LVLMs for improved interleaved image-text understanding. The code, data, and models are available at this https URL. 

---
# JEEM: Vision-Language Understanding in Four Arabic Dialects 

**Authors**: Karima Kadaoui, Hanin Atwany, Hamdan Al-Ali, Abdelrahman Mohamed, Ali Mekky, Sergei Tilga, Natalia Fedorova, Ekaterina Artemova, Hanan Aldarmaki, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2503.21910)  

**Abstract**: We introduce JEEM, a benchmark designed to evaluate Vision-Language Models (VLMs) on visual understanding across four Arabic-speaking countries: Jordan, The Emirates, Egypt, and Morocco. JEEM includes the tasks of image captioning and visual question answering, and features culturally rich and regionally diverse content. This dataset aims to assess the ability of VLMs to generalize across dialects and accurately interpret cultural elements in visual contexts. In an evaluation of five prominent open-source Arabic VLMs and GPT-4V, we find that the Arabic VLMs consistently underperform, struggling with both visual understanding and dialect-specific generation. While GPT-4V ranks best in this comparison, the model's linguistic competence varies across dialects, and its visual understanding capabilities lag behind. This underscores the need for more inclusive models and the value of culturally-diverse evaluation paradigms. 

---
# Hybrid Emotion Recognition: Enhancing Customer Interactions Through Acoustic and Textual Analysis 

**Authors**: Sahan Hewage Wewelwala, T.G.D.K. Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2503.21927)  

**Abstract**: This research presents a hybrid emotion recognition system integrating advanced Deep Learning, Natural Language Processing (NLP), and Large Language Models (LLMs) to analyze audio and textual data for enhancing customer interactions in contact centers. By combining acoustic features with textual sentiment analysis, the system achieves nuanced emotion detection, addressing the limitations of traditional approaches in understanding complex emotional states. Leveraging LSTM and CNN models for audio analysis and DistilBERT for textual evaluation, the methodology accommodates linguistic and cultural variations while ensuring real-time processing. Rigorous testing on diverse datasets demonstrates the system's robustness and accuracy, highlighting its potential to transform customer service by enabling personalized, empathetic interactions and improving operational efficiency. This research establishes a foundation for more intelligent and human-centric digital communication, redefining customer service standards. 

---
# Refining Time Series Anomaly Detectors using Large Language Models 

**Authors**: Alan Yang, Yulin Chen, Sean Lee, Venus Montes  

**Link**: [PDF](https://arxiv.org/pdf/2503.21833)  

**Abstract**: Time series anomaly detection (TSAD) is of widespread interest across many industries, including finance, healthcare, and manufacturing. Despite the development of numerous automatic methods for detecting anomalies, human oversight remains necessary to review and act upon detected anomalies, as well as verify their accuracy. We study the use of multimodal large language models (LLMs) to partially automate this process. We find that LLMs can effectively identify false alarms by integrating visual inspection of time series plots with text descriptions of the data-generating process. By leveraging the capabilities of LLMs, we aim to reduce the reliance on human effort required to maintain a TSAD system 

---
