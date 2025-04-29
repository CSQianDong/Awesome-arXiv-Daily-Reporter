# Multimodal Conditioned Diffusive Time Series Forecasting 

**Authors**: Chen Su, Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.19669)  

**Abstract**: Diffusion models achieve remarkable success in processing images and text, and have been extended to special domains such as time series forecasting (TSF). Existing diffusion-based approaches for TSF primarily focus on modeling single-modality numerical sequences, overlooking the rich multimodal information in time series data. To effectively leverage such information for prediction, we propose a multimodal conditioned diffusion model for TSF, namely, MCD-TSF, to jointly utilize timestamps and texts as extra guidance for time series modeling, especially for forecasting. Specifically, Timestamps are combined with time series to establish temporal and semantic correlations among different data points when aggregating information along the temporal dimension. Texts serve as supplementary descriptions of time series' history, and adaptively aligned with data points as well as dynamically controlled in a classifier-free manner. Extensive experiments on real-world benchmark datasets across eight domains demonstrate that the proposed MCD-TSF model achieves state-of-the-art performance. 

---
# VCM: Vision Concept Modeling Based on Implicit Contrastive Learning with Vision-Language Instruction Fine-Tuning 

**Authors**: Run Luo, Renke Shan, Longze Chen, Ziqiang Liu, Lu Wang, Min Yang, Xiaobo Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.19627)  

**Abstract**: Large Vision-Language Models (LVLMs) are pivotal for real-world AI tasks like embodied intelligence due to their strong vision-language reasoning abilities. However, current LVLMs process entire images at the token level, which is inefficient compared to humans who analyze information and generate content at the conceptual level, extracting relevant visual concepts with minimal effort. This inefficiency, stemming from the lack of a visual concept model, limits LVLMs' usability in real-world applications. To address this, we propose VCM, an end-to-end self-supervised visual concept modeling framework. VCM leverages implicit contrastive learning across multiple sampled instances and vision-language fine-tuning to construct a visual concept model without requiring costly concept-level annotations. Our results show that VCM significantly reduces computational costs (e.g., 85\% fewer FLOPs for LLaVA-1.5-7B) while maintaining strong performance across diverse image understanding tasks. Moreover, VCM enhances visual encoders' capabilities in classic visual concept perception tasks. Extensive quantitative and qualitative experiments validate the effectiveness and efficiency of VCM. 

---
# VIST-GPT: Ushering in the Era of Visual Storytelling with LLMs? 

**Authors**: Mohamed Gado, Towhid Taliee, Muhammad Memon, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.19267)  

**Abstract**: Visual storytelling is an interdisciplinary field combining computer vision and natural language processing to generate cohesive narratives from sequences of images. This paper presents a novel approach that leverages recent advancements in multimodal models, specifically adapting transformer-based architectures and large multimodal models, for the visual storytelling task. Leveraging the large-scale Visual Storytelling (VIST) dataset, our VIST-GPT model produces visually grounded, contextually appropriate narratives. We address the limitations of traditional evaluation metrics, such as BLEU, METEOR, ROUGE, and CIDEr, which are not suitable for this task. Instead, we utilize RoViST and GROOVIST, novel reference-free metrics designed to assess visual storytelling, focusing on visual grounding, coherence, and non-redundancy. These metrics provide a more nuanced evaluation of narrative quality, aligning closely with human judgment. 

---
# Toward Generalizable Evaluation in the LLM Era: A Survey Beyond Benchmarks 

**Authors**: Yixin Cao, Shibo Hong, Xinze Li, Jiahao Ying, Yubo Ma, Haiyuan Liang, Yantao Liu, Zijun Yao, Xiaozhi Wang, Dan Huang, Wenxuan Zhang, Lifu Huang, Muhao Chen, Lei Hou, Qianru Sun, Xingjun Ma, Zuxuan Wu, Min-Yen Kan, David Lo, Qi Zhang, Heng Ji, Jing Jiang, Juanzi Li, Aixin Sun, Xuanjing Huang, Tat-Seng Chua, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18838)  

**Abstract**: Large Language Models (LLMs) are advancing at an amazing speed and have become indispensable across academia, industry, and daily applications. To keep pace with the status quo, this survey probes the core challenges that the rise of LLMs poses for evaluation. We identify and analyze two pivotal transitions: (i) from task-specific to capability-based evaluation, which reorganizes benchmarks around core competencies such as knowledge, reasoning, instruction following, multi-modal understanding, and safety; and (ii) from manual to automated evaluation, encompassing dynamic dataset curation and "LLM-as-a-judge" scoring.
Yet, even with these transitions, a crucial obstacle persists: the evaluation generalization issue. Bounded test sets cannot scale alongside models whose abilities grow seemingly without limit. We will dissect this issue, along with the core challenges of the above two transitions, from the perspectives of methods, datasets, evaluators, and metrics. Due to the fast evolving of this field, we will maintain a living GitHub repository (links are in each section) to crowd-source updates and corrections, and warmly invite contributors and collaborators. 

---
# Mitigating Modality Bias in Multi-modal Entity Alignment from a Causal Perspective 

**Authors**: Taoyu Su, Jiawei Sheng, Duohe Ma, Xiaodong Li, Juwei Yue, Mengxiao Song, Yingkai Tang, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19458)  

**Abstract**: Multi-Modal Entity Alignment (MMEA) aims to retrieve equivalent entities from different Multi-Modal Knowledge Graphs (MMKGs), a critical information retrieval task. Existing studies have explored various fusion paradigms and consistency constraints to improve the alignment of equivalent entities, while overlooking that the visual modality may not always contribute positively. Empirically, entities with low-similarity images usually generate unsatisfactory performance, highlighting the limitation of overly relying on visual features. We believe the model can be biased toward the visual modality, leading to a shortcut image-matching task. To address this, we propose a counterfactual debiasing framework for MMEA, termed CDMEA, which investigates visual modality bias from a causal perspective. Our approach aims to leverage both visual and graph modalities to enhance MMEA while suppressing the direct causal effect of the visual modality on model predictions. By estimating the Total Effect (TE) of both modalities and excluding the Natural Direct Effect (NDE) of the visual modality, we ensure that the model predicts based on the Total Indirect Effect (TIE), effectively utilizing both modalities and reducing visual modality bias. Extensive experiments on 9 benchmark datasets show that CDMEA outperforms 14 state-of-the-art methods, especially in low-similarity, high-noise, and low-resource data scenarios. 

---
# Stealing Creator's Workflow: A Creator-Inspired Agentic Framework with Iterative Feedback Loop for Improved Scientific Short-form Generation 

**Authors**: Jong Inn Park, Maanas Taneja, Qianwen Wang, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18805)  

**Abstract**: Generating engaging, accurate short-form videos from scientific papers is challenging due to content complexity and the gap between expert authors and readers. Existing end-to-end methods often suffer from factual inaccuracies and visual artifacts, limiting their utility for scientific dissemination. To address these issues, we propose SciTalk, a novel multi-LLM agentic framework, grounding videos in various sources, such as text, figures, visual styles, and avatars. Inspired by content creators' workflows, SciTalk uses specialized agents for content summarization, visual scene planning, and text and layout editing, and incorporates an iterative feedback mechanism where video agents simulate user roles to give feedback on generated videos from previous iterations and refine generation prompts. Experimental evaluations show that SciTalk outperforms simple prompting methods in generating scientifically accurate and engaging content over the refined loop of video generation. Although preliminary results are still not yet matching human creators' quality, our framework provides valuable insights into the challenges and benefits of feedback-driven video generation. Our code, data, and generated videos will be publicly available. 

---
# LINC: Supporting Language Independent Communication and Comprehension to Enhance Contribution in Multilingual Collaborative Meetings 

**Authors**: Saramsh Gautam, Mahmood Jasim  

**Link**: [PDF](https://arxiv.org/pdf/2504.18988)  

**Abstract**: Collaborative research often includes contributors with varied perspectives from diverse linguistic backgrounds. However, English as a Second Language (ESL) researchers often struggle to communicate during meetings in English and comprehend discussions, leading to limited contribution. To investigate these challenges, we surveyed 64 ESL researchers who frequently collaborate in multilingual teams and identified four key design goals around participation, comprehension, documentation, and feedback. Guided by these design goals, we developed LINC, a multimodal Language INdependent Collaboration system with two components: a real-time module for multilingual communication during meetings and a post-meeting dashboard for discussion analysis. We evaluated the system through a two-phased study with six triads of multilingual teams. We found that using LINC, participants benefited from communicating in their preferred language, recalled and reviewed actionable insights, and prepared for upcoming meetings effectively. We discuss external factors that impact multilingual meeting participation beyond language preferences and the implications of multimodal systems in facilitating meetings in hybrid multilingual collaborative settings beyond research. 

---
# Feature Fusion Revisited: Multimodal CTR Prediction for MMCTR Challenge 

**Authors**: Junjie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18961)  

**Abstract**: With the rapid advancement of Multimodal Large Language Models (MLLMs), an increasing number of researchers are exploring their application in recommendation systems. However, the high latency associated with large models presents a significant challenge for such use cases. The EReL@MIR workshop provided a valuable opportunity to experiment with various approaches aimed at improving the efficiency of multimodal representation learning for information retrieval tasks. As part of the competition's requirements, participants were mandated to submit a technical report detailing their methodologies and findings. Our team was honored to receive the award for Task 2 - Winner (Multimodal CTR Prediction). In this technical report, we present our methods and key findings. Additionally, we propose several directions for future work, particularly focusing on how to effectively integrate recommendation signals into multimodal representations. The codebase for our implementation is publicly available at: this https URL, and the trained model weights can be accessed at: this https URL. 

---
# Towards AI-Driven Policing: Interdisciplinary Knowledge Discovery from Police Body-Worn Camera Footage 

**Authors**: Anita Srbinovska, Angela Srbinovska, Vivek Senthil, Adrian Martin, John McCluskey, Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2504.20007)  

**Abstract**: This paper proposes a novel interdisciplinary framework for analyzing police body-worn camera (BWC) footage from the Rochester Police Department (RPD) using advanced artificial intelligence (AI) and statistical machine learning (ML) techniques. Our goal is to detect, classify, and analyze patterns of interaction between police officers and civilians to identify key behavioral dynamics, such as respect, disrespect, escalation, and de-escalation. We apply multimodal data analysis by integrating video, audio, and natural language processing (NLP) techniques to extract meaningful insights from BWC footage. We present our methodology, computational techniques, and findings, outlining a practical approach for law enforcement while advancing the frontiers of knowledge discovery from police BWC data. 

---
# Proof-of-TBI -- Fine-Tuned Vision Language Model Consortium and OpenAI-o3 Reasoning LLM-Based Medical Diagnosis Support System for Mild Traumatic Brain Injury (TBI) Prediction 

**Authors**: Ross Gore, Eranga Bandara, Sachin Shetty, Alberto E. Musto, Pratip Rana, Ambrosio Valencia-Romero, Christopher Rhea, Lobat Tayebi, Heather Richter, Atmaram Yarlagadda, Donna Edmonds, Steven Wallace, Donna Broshek  

**Link**: [PDF](https://arxiv.org/pdf/2504.18671)  

**Abstract**: Mild Traumatic Brain Injury (TBI) detection presents significant challenges due to the subtle and often ambiguous presentation of symptoms in medical imaging, making accurate diagnosis a complex task. To address these challenges, we propose Proof-of-TBI, a medical diagnosis support system that integrates multiple fine-tuned vision-language models with the OpenAI-o3 reasoning large language model (LLM). Our approach fine-tunes multiple vision-language models using a labeled dataset of TBI MRI scans, training them to diagnose TBI symptoms effectively. The predictions from these models are aggregated through a consensus-based decision-making process. The system evaluates the predictions from all fine-tuned vision language models using the OpenAI-o3 reasoning LLM, a model that has demonstrated remarkable reasoning performance, to produce the most accurate final diagnosis. The LLM Agents orchestrates interactions between the vision-language models and the reasoning LLM, managing the final decision-making process with transparency, reliability, and automation. This end-to-end decision-making workflow combines the vision-language model consortium with the OpenAI-o3 reasoning LLM, enabled by custom prompt engineering by the LLM agents. The prototype for the proposed platform was developed in collaboration with the U.S. Army Medical Research team in Newport News, Virginia, incorporating five fine-tuned vision-language models. The results demonstrate the transformative potential of combining fine-tuned vision-language model inputs with the OpenAI-o3 reasoning LLM to create a robust, secure, and highly accurate diagnostic system for mild TBI prediction. To the best of our knowledge, this research represents the first application of fine-tuned vision-language models integrated with a reasoning LLM for TBI prediction tasks. 

---
# Enhancing Surgical Documentation through Multimodal Visual-Temporal Transformers and Generative AI 

**Authors**: Hugo Georgenthum, Cristian Cosentino, Fabrizio Marozzo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.19918)  

**Abstract**: The automatic summarization of surgical videos is essential for enhancing procedural documentation, supporting surgical training, and facilitating post-operative analysis. This paper presents a novel method at the intersection of artificial intelligence and medicine, aiming to develop machine learning models with direct real-world applications in surgical contexts. We propose a multi-modal framework that leverages recent advancements in computer vision and large language models to generate comprehensive video summaries. %
The approach is structured in three key stages. First, surgical videos are divided into clips, and visual features are extracted at the frame level using visual transformers. This step focuses on detecting tools, tissues, organs, and surgical actions. Second, the extracted features are transformed into frame-level captions via large language models. These are then combined with temporal features, captured using a ViViT-based encoder, to produce clip-level summaries that reflect the broader context of each video segment. Finally, the clip-level descriptions are aggregated into a full surgical report using a dedicated LLM tailored for the summarization task. %
We evaluate our method on the CholecT50 dataset, using instrument and action annotations from 50 laparoscopic videos. The results show strong performance, achieving 96\% precision in tool detection and a BERT score of 0.74 for temporal context summarization. This work contributes to the advancement of AI-assisted tools for surgical reporting, offering a step toward more intelligent and reliable clinical documentation. 

---
# NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks 

**Authors**: Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U-Xuan Tan, Navonil Majumder, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2504.19854)  

**Abstract**: Existing Visual-Language-Action (VLA) models have shown promising performance in zero-shot scenarios, demonstrating impressive task execution and reasoning capabilities. However, a significant challenge arises from the limitations of visual encoding, which can result in failures during tasks such as object grasping. Moreover, these models typically suffer from high computational overhead due to their large sizes, often exceeding 7B parameters. While these models excel in reasoning and task planning, the substantial computational overhead they incur makes them impractical for real-time robotic environments, where speed and efficiency are paramount. To address the limitations of existing VLA models, we propose NORA, a 3B-parameter model designed to reduce computational overhead while maintaining strong task performance. NORA adopts the Qwen-2.5-VL-3B multimodal model as its backbone, leveraging its superior visual-semantic understanding to enhance visual reasoning and action grounding. Additionally, our \model{} is trained on 970k real-world robot demonstrations and equipped with the FAST+ tokenizer for efficient action sequence generation. Experimental results demonstrate that NORA outperforms existing large-scale VLA models, achieving better task performance with significantly reduced computational overhead, making it a more practical solution for real-time robotic autonomy. 

---
# CLIP-KOA: Enhancing Knee Osteoarthritis Diagnosis with Multi-Modal Learning and Symmetry-Aware Loss Functions 

**Authors**: Yejin Jeong, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.19443)  

**Abstract**: Knee osteoarthritis (KOA) is a universal chronic musculoskeletal disorders worldwide, making early diagnosis crucial. Currently, the Kellgren and Lawrence (KL) grading system is widely used to assess KOA severity. However, its high inter-observer variability and subjectivity hinder diagnostic consistency. To address these limitations, automated diagnostic techniques using deep learning have been actively explored in recent years. In this study, we propose a CLIP-based framework (CLIP-KOA) to enhance the consistency and reliability of KOA grade prediction. To achieve this, we introduce a learning approach that integrates image and text information and incorporate Symmetry Loss and Consistency Loss to ensure prediction consistency between the original and flipped images. CLIP-KOA achieves state-of-the-art accuracy of 71.86\% on KOA severity prediction task, and ablation studies show that CLIP-KOA has 2.36\% improvement in accuracy over the standard CLIP model due to our contribution. This study shows a novel direction for data-driven medical prediction not only to improve reliability of fine-grained diagnosis and but also to explore multimodal methods for medical image analysis. Our code is available at this https URL. 

---
# EarthMapper: Visual Autoregressive Models for Controllable Bidirectional Satellite-Map Translation 

**Authors**: Zhe Dong, Yuzhe Sun, Tianzhu Liu, Wangmeng Zuo, Yanfeng Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19432)  

**Abstract**: Satellite imagery and maps, as two fundamental data modalities in remote sensing, offer direct observations of the Earth's surface and human-interpretable geographic abstractions, respectively. The task of bidirectional translation between satellite images and maps (BSMT) holds significant potential for applications in urban planning and disaster response. However, this task presents two major challenges: first, the absence of precise pixel-wise alignment between the two modalities substantially complicates the translation process; second, it requires achieving both high-level abstraction of geographic features and high-quality visual synthesis, which further elevates the technical complexity. To address these limitations, we introduce EarthMapper, a novel autoregressive framework for controllable bidirectional satellite-map translation. EarthMapper employs geographic coordinate embeddings to anchor generation, ensuring region-specific adaptability, and leverages multi-scale feature alignment within a geo-conditioned joint scale autoregression (GJSA) process to unify bidirectional translation in a single training cycle. A semantic infusion (SI) mechanism is introduced to enhance feature-level consistency, while a key point adaptive guidance (KPAG) mechanism is proposed to dynamically balance diversity and precision during inference. We further contribute CNSatMap, a large-scale dataset comprising 302,132 precisely aligned satellite-map pairs across 38 Chinese cities, enabling robust benchmarking. Extensive experiments on CNSatMap and the New York dataset demonstrate EarthMapper's superior performance, achieving significant improvements in visual realism, semantic consistency, and structural fidelity over state-of-the-art methods. Additionally, EarthMapper excels in zero-shot tasks like in-painting, out-painting and coordinate-conditional generation, underscoring its versatility. 

---
# Platonic Grounding for Efficient Multimodal Language Models 

**Authors**: Moulik Choraria, Xinbo Wu, Akhil Bhimaraju, Nitesh Sekhar, Yue Wu, Xu Zhang, Prateek Singhal, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2504.19327)  

**Abstract**: The hyperscaling of data and parameter count in Transformer-based models is yielding diminishing performance improvement, especially when weighed against training costs. Such plateauing indicates the importance of methods for more efficient finetuning and inference, while retaining similar performance. This is especially relevant for multimodal learning paradigms, where inference costs of processing multimodal tokens can determine the model's practical viability. At the same time, research on representations and mechanistic interpretability has improved our understanding of the inner workings of Transformer-based models; one such line of work reveals an implicit alignment in the deeper layers of pretrained models, across modalities. Taking inspiration from this, we motivate and propose a simple modification to existing multimodal frameworks that rely on aligning pretrained models. We demonstrate that our approach maintains and, in some cases, even improves performance of baseline methods while achieving significant gains in both training and inference-time compute. Our work also has implications for combining pretrained models into larger systems efficiently. 

---
# PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-rich Manipulation Using Tactile-Diffusion Policies 

**Authors**: Jialiang Zhao, Naveen Kuppuswamy, Siyuan Feng, Benjamin Burchfiel, Edward Adelson  

**Link**: [PDF](https://arxiv.org/pdf/2504.19341)  

**Abstract**: Achieving robust dexterous manipulation in unstructured domestic environments remains a significant challenge in robotics. Even with state-of-the-art robot learning methods, haptic-oblivious control strategies (i.e. those relying only on external vision and/or proprioception) often fall short due to occlusions, visual complexities, and the need for precise contact interaction control. To address these limitations, we introduce PolyTouch, a novel robot finger that integrates camera-based tactile sensing, acoustic sensing, and peripheral visual sensing into a single design that is compact and durable. PolyTouch provides high-resolution tactile feedback across multiple temporal scales, which is essential for efficiently learning complex manipulation tasks. Experiments demonstrate an at least 20-fold increase in lifespan over commercial tactile sensors, with a design that is both easy to manufacture and scalable. We then use this multi-modal tactile feedback along with visuo-proprioceptive observations to synthesize a tactile-diffusion policy from human demonstrations; the resulting contact-aware control policy significantly outperforms haptic-oblivious policies in multiple contact-aware manipulation policies. This paper highlights how effectively integrating multi-modal contact sensing can hasten the development of effective contact-aware manipulation policies, paving the way for more reliable and versatile domestic robots. More information can be found at this https URL 

---
# CapsFake: A Multimodal Capsule Network for Detecting Instruction-Guided Deepfakes 

**Authors**: Tuan Nguyen, Naseem Khan, Issa Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2504.19212)  

**Abstract**: The rapid evolution of deepfake technology, particularly in instruction-guided image editing, threatens the integrity of digital images by enabling subtle, context-aware manipulations. Generated conditionally from real images and textual prompts, these edits are often imperceptible to both humans and existing detection systems, revealing significant limitations in current defenses. We propose a novel multimodal capsule network, CapsFake, designed to detect such deepfake image edits by integrating low-level capsules from visual, textual, and frequency-domain modalities. High-level capsules, predicted through a competitive routing mechanism, dynamically aggregate local features to identify manipulated regions with precision. Evaluated on diverse datasets, including MagicBrush, Unsplash Edits, Open Images Edits, and Multi-turn Edits, CapsFake outperforms state-of-the-art methods by up to 20% in detection accuracy. Ablation studies validate its robustness, achieving detection rates above 94% under natural perturbations and 96% against adversarial attacks, with excellent generalization to unseen editing scenarios. This approach establishes a powerful framework for countering sophisticated image manipulations. 

---
# Video CLIP Model for Multi-View Echocardiography Interpretation 

**Authors**: Ryo Takizawa, Satoshi Kodera, Tempei Kabayama, Ryo Matsuoka, Yuta Ando, Yuto Nakamura, Haruki Settai, Norihiko Takeda  

**Link**: [PDF](https://arxiv.org/pdf/2504.18800)  

**Abstract**: Echocardiography involves recording videos of the heart using ultrasound, enabling clinicians to evaluate its condition. Recent advances in large-scale vision-language models (VLMs) have garnered attention for automating the interpretation of echocardiographic videos. However, most existing VLMs proposed for medical interpretation thus far rely on single-frame (i.e., image) inputs. Consequently, these image-based models often exhibit lower diagnostic accuracy for conditions identifiable through cardiac motion. Moreover, echocardiographic videos are recorded from various views that depend on the direction of ultrasound emission, and certain views are more suitable than others for interpreting specific conditions. Incorporating multiple views could potentially yield further improvements in accuracy. In this study, we developed a video-language model that takes five different views and full video sequences as input, training it on pairs of echocardiographic videos and clinical reports from 60,747 cases. Our experiments demonstrate that this expanded approach achieves higher interpretation accuracy than models trained with only single-view videos or with still images. 

---
# M2R2: MulitModal Robotic Representation for Temporal Action Segmentation 

**Authors**: Daniel Sliwowski, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.18662)  

**Abstract**: Temporal action segmentation (TAS) has long been a key area of research in both robotics and computer vision. In robotics, algorithms have primarily focused on leveraging proprioceptive information to determine skill boundaries, with recent approaches in surgical robotics incorporating vision. In contrast, computer vision typically relies on exteroceptive sensors, such as cameras. Existing multimodal TAS models in robotics integrate feature fusion within the model, making it difficult to reuse learned features across different models. Meanwhile, pretrained vision-only feature extractors commonly used in computer vision struggle in scenarios with limited object visibility. In this work, we address these challenges by proposing M2R2, a multimodal feature extractor tailored for TAS, which combines information from both proprioceptive and exteroceptive sensors. We introduce a novel pretraining strategy that enables the reuse of learned features across multiple TAS models. Our method achieves state-of-the-art performance on the REASSEMBLE dataset, a challenging multimodal robotic assembly dataset, outperforming existing robotic action segmentation models by 46.6%. Additionally, we conduct an extensive ablation study to evaluate the contribution of different modalities in robotic TAS tasks. 

---
