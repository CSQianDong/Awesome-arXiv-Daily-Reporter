# Multi-modal Knowledge Graph Generation with Semantics-enriched Prompts 

**Authors**: Yajing Xu, Zhiqiang Liu, Jiaoyan Chen, Mingchen Tu, Zhuo Chen, Jeff Z. Pan, Yichi Zhang, Yushan Zhu, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13631)  

**Abstract**: Multi-modal Knowledge Graphs (MMKGs) have been widely applied across various domains for knowledge representation. However, the existing MMKGs are significantly fewer than required, and their construction faces numerous challenges, particularly in ensuring the selection of high-quality, contextually relevant images for knowledge graph enrichment. To address these challenges, we present a framework for constructing MMKGs from conventional KGs. Furthermore, to generate higher-quality images that are more relevant to the context in the given knowledge graph, we designed a neighbor selection method called Visualizable Structural Neighbor Selection (VSNS). This method consists of two modules: Visualizable Neighbor Selection (VNS) and Structural Neighbor Selection (SNS). The VNS module filters relations that are difficult to visualize, while the SNS module selects neighbors that most effectively capture the structural characteristics of the entity. To evaluate the quality of the generated images, we performed qualitative and quantitative evaluations on two datasets, MKG-Y and DB15K. The experimental results indicate that using the VSNS method to select neighbors results in higher-quality images that are more relevant to the knowledge graph. 

---
# Exploring Multimodal Prompt for Visualization Authoring with Large Language Models 

**Authors**: Zhen Wen, Luoxuan Weng, Yinghao Tang, Runjin Zhang, Yuxin Liu, Bo Pan, Minfeng Zhu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13700)  

**Abstract**: Recent advances in large language models (LLMs) have shown great potential in automating the process of visualization authoring through simple natural language utterances. However, instructing LLMs using natural language is limited in precision and expressiveness for conveying visualization intent, leading to misinterpretation and time-consuming iterations. To address these limitations, we conduct an empirical study to understand how LLMs interpret ambiguous or incomplete text prompts in the context of visualization authoring, and the conditions making LLMs misinterpret user intent. Informed by the findings, we introduce visual prompts as a complementary input modality to text prompts, which help clarify user intent and improve LLMs' interpretation abilities. To explore the potential of multimodal prompting in visualization authoring, we design VisPilot, which enables users to easily create visualizations using multimodal prompts, including text, sketches, and direct manipulations on existing visualizations. Through two case studies and a controlled user study, we demonstrate that VisPilot provides a more intuitive way to create visualizations without affecting the overall task efficiency compared to text-only prompting approaches. Furthermore, we analyze the impact of text and visual prompts in different visualization tasks. Our findings highlight the importance of multimodal prompting in improving the usability of LLMs for visualization authoring. We discuss design implications for future visualization systems and provide insights into how multimodal prompts can enhance human-AI collaboration in creative visualization tasks. All materials are available at this https URL. 

---
# Lightweight LiDAR-Camera 3D Dynamic Object Detection and Multi-Class Trajectory Prediction 

**Authors**: Yushen He, Lei Zhao, Tianchen Deng, Zipeng Fang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13647)  

**Abstract**: Service mobile robots are often required to avoid dynamic objects while performing their tasks, but they usually have only limited computational resources. So we present a lightweight multi-modal framework for 3D object detection and trajectory prediction. Our system synergistically integrates LiDAR and camera inputs to achieve real-time perception of pedestrians, vehicles, and riders in 3D space. The framework proposes two novel modules: 1) a Cross-Modal Deformable Transformer (CMDT) for object detection with high accuracy and acceptable amount of computation, and 2) a Reference Trajectory-based Multi-Class Transformer (RTMCT) for efficient and diverse trajectory prediction of mult-class objects with flexible trajectory lengths. Evaluations on the CODa benchmark demonstrate superior performance over existing methods across detection (+2.03% in mAP) and trajectory prediction (-0.408m in minADE5 of pedestrians) metrics. Remarkably, the system exhibits exceptional deployability - when implemented on a wheelchair robot with an entry-level NVIDIA 3060 GPU, it achieves real-time inference at 13.2 fps. To facilitate reproducibility and practical deployment, we release the related code of the method at this https URL and its ROS inference version at this https URL. 

---
# Towards a Multi-Agent Vision-Language System for Zero-Shot Novel Hazardous Object Detection for Autonomous Driving Safety 

**Authors**: Shashank Shriram, Srinivasa Perisetla, Aryan Keskar, Harsha Krishnaswamy, Tonko Emil Westerhof Bossen, Andreas Møgelmose, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2504.13399)  

**Abstract**: Detecting anomalous hazards in visual data, particularly in video streams, is a critical challenge in autonomous driving. Existing models often struggle with unpredictable, out-of-label hazards due to their reliance on predefined object categories. In this paper, we propose a multimodal approach that integrates vision-language reasoning with zero-shot object detection to improve hazard identification and explanation. Our pipeline consists of a Vision-Language Model (VLM), a Large Language Model (LLM), in order to detect hazardous objects within a traffic scene. We refine object detection by incorporating OpenAI's CLIP model to match predicted hazards with bounding box annotations, improving localization accuracy. To assess model performance, we create a ground truth dataset by denoising and extending the foundational COOOL (Challenge-of-Out-of-Label) anomaly detection benchmark dataset with complete natural language descriptions for hazard annotations. We define a means of hazard detection and labeling evaluation on the extended dataset using cosine similarity. This evaluation considers the semantic similarity between the predicted hazard description and the annotated ground truth for each video. Additionally, we release a set of tools for structuring and managing large-scale hazard detection datasets. Our findings highlight the strengths and limitations of current vision-language-based approaches, offering insights into future improvements in autonomous hazard detection systems. Our models, scripts, and data can be found at this https URL 

---
# Chain-of-Modality: Learning Manipulation Programs from Multimodal Human Videos with Vision-Language-Models 

**Authors**: Chen Wang, Fei Xia, Wenhao Yu, Tingnan Zhang, Ruohan Zhang, C. Karen Liu, Li Fei-Fei, Jie Tan, Jacky Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13351)  

**Abstract**: Learning to perform manipulation tasks from human videos is a promising approach for teaching robots. However, many manipulation tasks require changing control parameters during task execution, such as force, which visual data alone cannot capture. In this work, we leverage sensing devices such as armbands that measure human muscle activities and microphones that record sound, to capture the details in the human manipulation process, and enable robots to extract task plans and control parameters to perform the same task. To achieve this, we introduce Chain-of-Modality (CoM), a prompting strategy that enables Vision Language Models to reason about multimodal human demonstration data -- videos coupled with muscle or audio signals. By progressively integrating information from each modality, CoM refines a task plan and generates detailed control parameters, enabling robots to perform manipulation tasks based on a single multimodal human video prompt. Our experiments show that CoM delivers a threefold improvement in accuracy for extracting task plans and control parameters compared to baselines, with strong generalization to new task setups and objects in real-world robot experiments. Videos and code are available at this https URL 

---
# On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks 

**Authors**: Ting Bi, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Jun Zhang, Zui Tao, Kailong Wang, Liting Zhou, Yang Yang, Tianlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13209)  

**Abstract**: Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats. 

---
# Mirror: Multimodal Cognitive Reframing Therapy for Rolling with Resistance 

**Authors**: Subin Kim, Hoonrae Kim, Jihyun Lee, Yejin Jeon, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.13211)  

**Abstract**: Recent studies have explored the use of large language models (LLMs) in psychotherapy; however, text-based cognitive behavioral therapy (CBT) models often struggle with client resistance, which can weaken therapeutic alliance. To address this, we propose a multimodal approach that incorporates nonverbal cues, allowing the AI therapist to better align its responses with the client's negative emotional state. Specifically, we introduce a new synthetic dataset, Multimodal Interactive Rolling with Resistance (Mirror), which is a novel synthetic dataset that pairs client statements with corresponding facial images. Using this dataset, we train baseline Vision-Language Models (VLMs) that can analyze facial cues, infer emotions, and generate empathetic responses to effectively manage resistance. They are then evaluated in terms of both the therapist's counseling skills and the strength of the therapeutic alliance in the presence of client resistance. Our results demonstrate that Mirror significantly enhances the AI therapist's ability to handle resistance, which outperforms existing text-based CBT approaches. 

---
# WildFireCan-MMD: A Multimodal dataset for Classification of User-generated Content During Wildfires in Canada 

**Authors**: Braeden Sherritt, Isar Nejadgholi, Marzieh Amini  

**Link**: [PDF](https://arxiv.org/pdf/2504.13231)  

**Abstract**: Rapid information access is vital during wildfires, yet traditional data sources are slow and costly. Social media offers real-time updates, but extracting relevant insights remains a challenge. We present WildFireCan-MMD, a new multimodal dataset of X posts from recent Canadian wildfires, annotated across 13 key themes. Evaluating both Vision Language Models and custom-trained classifiers, we show that while zero-shot prompting offers quick deployment, even simple trained models outperform them when labelled data is available, by up to 23%. Our findings highlight the enduring importance of tailored datasets and task-specific training. Importantly, such datasets should be localized, as disaster response requirements vary across regions and contexts. 

---
# Building Trustworthy Multimodal AI: A Review of Fairness, Transparency, and Ethics in Vision-Language Tasks 

**Authors**: Mohammad Saleha, Azadeh Tabatabaeib  

**Link**: [PDF](https://arxiv.org/pdf/2504.13199)  

**Abstract**: Objective: This review explores the trustworthiness of multimodal artificial intelligence (AI) systems, specifically focusing on vision-language tasks. It addresses critical challenges related to fairness, transparency, and ethical implications in these systems, providing a comparative analysis of key tasks such as Visual Question Answering (VQA), image captioning, and visual dialogue. Background: Multimodal models, particularly vision-language models, enhance artificial intelligence (AI) capabilities by integrating visual and textual data, mimicking human learning processes. Despite significant advancements, the trustworthiness of these models remains a crucial concern, particularly as AI systems increasingly confront issues regarding fairness, transparency, and ethics. Methods: This review examines research conducted from 2017 to 2024 focusing on forenamed core vision-language tasks. It employs a comparative approach to analyze these tasks through the lens of trustworthiness, underlining fairness, explainability, and ethics. This study synthesizes findings from recent literature to identify trends, challenges, and state-of-the-art solutions. Results: Several key findings were highlighted. Transparency: Explainability of vision language tasks is important for user trust. Techniques, such as attention maps and gradient-based methods, have successfully addressed this issue. Fairness: Bias mitigation in VQA and visual dialogue systems is essential for ensuring unbiased outcomes across diverse demographic groups. Ethical Implications: Addressing biases in multilingual models and ensuring ethical data handling is critical for the responsible deployment of vision-language systems. Conclusion: This study underscores the importance of integrating fairness, transparency, and ethical considerations in developing vision-language models within a unified framework. 

---
# Harmony: A Unified Framework for Modality Incremental Learning 

**Authors**: Yaguang Song, Xiaoshan Yang, Dongmei Jiang, Yaowei Wang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13218)  

**Abstract**: Incremental learning aims to enable models to continuously acquire knowledge from evolving data streams while preserving previously learned capabilities. While current research predominantly focuses on unimodal incremental learning and multimodal incremental learning where the modalities are consistent, real-world scenarios often present data from entirely new modalities, posing additional challenges. This paper investigates the feasibility of developing a unified model capable of incremental learning across continuously evolving modal sequences. To this end, we introduce a novel paradigm called Modality Incremental Learning (MIL), where each learning stage involves data from distinct modalities. To address this task, we propose a novel framework named Harmony, designed to achieve modal alignment and knowledge retention, enabling the model to reduce the modal discrepancy and learn from a sequence of distinct modalities, ultimately completing tasks across multiple modalities within a unified framework. Our approach introduces the adaptive compatible feature modulation and cumulative modal bridging. Through constructing historical modal features and performing modal knowledge accumulation and alignment, the proposed components collaboratively bridge modal differences and maintain knowledge retention, even with solely unimodal data available at each learning this http URL components work in concert to establish effective modality connections and maintain knowledge retention, even when only unimodal data is available at each learning stage. Extensive experiments on the MIL task demonstrate that our proposed method significantly outperforms existing incremental learning methods, validating its effectiveness in MIL scenarios. 

---
