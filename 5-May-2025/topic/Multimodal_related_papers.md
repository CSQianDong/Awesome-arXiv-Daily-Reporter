# PREMISE: Matching-based Prediction for Accurate Review Recommendation 

**Authors**: Wei Han, Hui Chen, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.01255)  

**Abstract**: We present PREMISE (PREdict with Matching ScorEs), a new architecture for the matching-based learning in the multimodal fields for the multimodal review helpfulness (MRHP) task. Distinct to previous fusion-based methods which obtains multimodal representations via cross-modal attention for downstream tasks, PREMISE computes the multi-scale and multi-field representations, filters duplicated semantics, and then obtained a set of matching scores as feature vectors for the downstream recommendation task. This new architecture significantly boosts the performance for such multimodal tasks whose context matching content are highly correlated to the targets of that task, compared to the state-of-the-art fusion-based methods. Experimental results on two publicly available datasets show that PREMISE achieves promising performance with less computational cost. 

---
# Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs 

**Authors**: Yijie Jin, Junjie Peng, Xuanchao Lin, Haochen Yuan, Lan Wang, Cangzhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.01068)  

**Abstract**: Multimodal Sentiment Analysis (MSA) is a rapidly developing field that integrates multimodal information to recognize sentiments, and existing models have made significant progress in this area. The central challenge in MSA is multimodal fusion, which is predominantly addressed by Multimodal Transformers (MulTs). Although act as the paradigm, MulTs suffer from efficiency concerns. In this work, from the perspective of efficiency optimization, we propose and prove that MulTs are hierarchical modal-wise heterogeneous graphs (HMHGs), and we introduce the graph-structured representation pattern of MulTs. Based on this pattern, we propose an Interlaced Mask (IM) mechanism to design the Graph-Structured and Interlaced-Masked Multimodal Transformer (GsiT). It is formally equivalent to MulTs which achieves an efficient weight-sharing mechanism without information disorder through IM, enabling All-Modal-In-One fusion with only 1/3 of the parameters of pure MulTs. A Triton kernel called Decomposition is implemented to ensure avoiding additional computational overhead. Moreover, it achieves significantly higher performance than traditional MulTs. To further validate the effectiveness of GsiT itself and the HMHG concept, we integrate them into multiple state-of-the-art models and demonstrate notable performance improvements and parameter reduction on widely used MSA datasets. 

---
# Multi-Modal Language Models as Text-to-Image Model Evaluators 

**Authors**: Jiahui Chen, Candace Ross, Reyhane Askari-Hemmat, Koustuv Sinha, Melissa Hall, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2505.00759)  

**Abstract**: The steady improvements of text-to-image (T2I) generative models lead to slow deprecation of automatic evaluation benchmarks that rely on static datasets, motivating researchers to seek alternative ways to evaluate the T2I progress. In this paper, we explore the potential of multi-modal large language models (MLLMs) as evaluator agents that interact with a T2I model, with the objective of assessing prompt-generation consistency and image aesthetics. We present Multimodal Text-to-Image Eval (MT2IE), an evaluation framework that iteratively generates prompts for evaluation, scores generated images and matches T2I evaluation of existing benchmarks with a fraction of the prompts used in existing static benchmarks. Moreover, we show that MT2IE's prompt-generation consistency scores have higher correlation with human judgment than scores previously introduced in the literature. MT2IE generates prompts that are efficient at probing T2I model performance, producing the same relative T2I model rankings as existing benchmarks while using only 1/80th the number of prompts for evaluation. 

---
# Evaluating Vision Language Model Adaptations for Radiology Report Generation in Low-Resource Languages 

**Authors**: Marco Salmè, Rosa Sicilia, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01096)  

**Abstract**: The integration of artificial intelligence in healthcare has opened new horizons for improving medical diagnostics and patient care. However, challenges persist in developing systems capable of generating accurate and contextually relevant radiology reports, particularly in low-resource languages. In this study, we present a comprehensive benchmark to evaluate the performance of instruction-tuned Vision-Language Models (VLMs) in the specialized task of radiology report generation across three low-resource languages: Italian, German, and Spanish. Employing the LLaVA architectural framework, we conducted a systematic evaluation of pre-trained models utilizing general datasets, domain-specific datasets, and low-resource language-specific datasets. In light of the unavailability of models that possess prior knowledge of both the medical domain and low-resource languages, we analyzed various adaptations to determine the most effective approach for these contexts. The results revealed that language-specific models substantially outperformed both general and domain-specific models in generating radiology reports, emphasizing the critical role of linguistic adaptation. Additionally, models fine-tuned with medical terminology exhibited enhanced performance across all languages compared to models with generic knowledge, highlighting the importance of domain-specific training. We also explored the influence of the temperature parameter on the coherence of report generation, providing insights for optimal model settings. Our findings highlight the importance of tailored language and domain-specific training for improving the quality and accuracy of radiological reports in multilingual settings. This research not only advances our understanding of VLMs adaptability in healthcare but also points to significant avenues for future investigations into model tuning and language-specific adaptations. 

---
# BalancEdit: Dynamically Balancing the Generality-Locality Trade-off in Multi-modal Model Editing 

**Authors**: Dongliang Guo, Mengxuan Hu, Zihan Guan, Thomas Hartvigsen, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01343)  

**Abstract**: Large multi-modal models inevitably decay over time as facts change and previously learned information becomes outdated. Traditional approaches such as fine-tuning are often impractical for updating these models due to their size and complexity. Instead, direct knowledge editing within the models presents a more viable solution. Current model editing techniques, however, typically overlook the unique influence ranges of different facts, leading to compromised model performance in terms of both generality and locality. To address this issue, we introduce the concept of the generality-locality trade-off in multi-modal model editing. We develop a new model editing dataset named OKEDIT, specifically designed to effectively evaluate this trade-off. Building on this foundation, we propose BalancEdit, a novel method for balanced model editing that dynamically achieves an optimal balance between generality and locality. BalancEdit utilizes a unique mechanism that generates both positive and negative samples for each fact to accurately determine its influence scope and incorporates these insights into the model's latent space using a discrete, localized codebook of edits, without modifying the underlying model weights. To our knowledge, this is the first approach explicitly addressing the generality-locality trade-off in multi-modal model editing. Our comprehensive results confirm the effectiveness of BalancEdit, demonstrating minimal trade-offs while maintaining robust editing capabilities. Our code and dataset will be available. 

---
# GENMO: A GENeralist Model for Human MOtion 

**Authors**: Jiefeng Li, Jinkun Cao, Haotian Zhang, Davis Rempe, Jan Kautz, Umar Iqbal, Ye Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01425)  

**Abstract**: Human motion modeling traditionally separates motion generation and estimation into distinct tasks with specialized models. Motion generation models focus on creating diverse, realistic motions from inputs like text, audio, or keyframes, while motion estimation models aim to reconstruct accurate motion trajectories from observations like videos. Despite sharing underlying representations of temporal dynamics and kinematics, this separation limits knowledge transfer between tasks and requires maintaining separate models. We present GENMO, a unified Generalist Model for Human Motion that bridges motion estimation and generation in a single framework. Our key insight is to reformulate motion estimation as constrained motion generation, where the output motion must precisely satisfy observed conditioning signals. Leveraging the synergy between regression and diffusion, GENMO achieves accurate global motion estimation while enabling diverse motion generation. We also introduce an estimation-guided training objective that exploits in-the-wild videos with 2D annotations and text descriptions to enhance generative diversity. Furthermore, our novel architecture handles variable-length motions and mixed multimodal conditions (text, audio, video) at different time intervals, offering flexible control. This unified approach creates synergistic benefits: generative priors improve estimated motions under challenging conditions like occlusions, while diverse video data enhances generation capabilities. Extensive experiments demonstrate GENMO's effectiveness as a generalist framework that successfully handles multiple human motion tasks within a single model. 

---
# Multimodal Doctor-in-the-Loop: A Clinically-Guided Explainable Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer 

**Authors**: Alice Natalina Caragliano, Claudia Tacconi, Carlo Greco, Lorenzo Nibid, Edy Ippolito, Michele Fiore, Giuseppe Perrone, Sara Ramella, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01390)  

**Abstract**: This study proposes a novel approach combining Multimodal Deep Learning with intrinsic eXplainable Artificial Intelligence techniques to predict pathological response in non-small cell lung cancer patients undergoing neoadjuvant therapy. Due to the limitations of existing radiomics and unimodal deep learning approaches, we introduce an intermediate fusion strategy that integrates imaging and clinical data, enabling efficient interaction between data modalities. The proposed Multimodal Doctor-in-the-Loop method further enhances clinical relevance by embedding clinicians' domain knowledge directly into the training process, guiding the model's focus gradually from broader lung regions to specific lesions. Results demonstrate improved predictive accuracy and explainability, providing insights into optimal data integration strategies for clinical applications. 

---
# FalconWing: An Open-Source Platform for Ultra-Light Fixed-Wing Aircraft Research 

**Authors**: Yan Miao, Will Shen, Hang Cui, Sayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2505.01383)  

**Abstract**: We present FalconWing -- an open-source, ultra-lightweight (150 g) fixed-wing platform for autonomy research. The hardware platform integrates a small camera, a standard airframe, offboard computation, and radio communication for manual overrides. We demonstrate FalconWing's capabilities by developing and deploying a purely vision-based control policy for autonomous landing (without IMU or motion capture) using a novel real-to-sim-to-real learning approach. Our learning approach: (1) constructs a photorealistic simulation environment via 3D Gaussian splatting trained on real-world images; (2) identifies nonlinear dynamics from vision-estimated real-flight data; and (3) trains a multi-modal Vision Transformer (ViT) policy through simulation-only imitation learning. The ViT architecture fuses single RGB image with the history of control actions via self-attention, preserving temporal context while maintaining real-time 20 Hz inference. When deployed zero-shot on the hardware platform, this policy achieves an 80% success rate in vision-based autonomous landings. Together with the hardware specifications, we also open-source the system dynamics, the software for photorealistic simulator and the learning approach. 

---
# Any-to-Any Vision-Language Model for Multimodal X-ray Imaging and Radiological Report Generation 

**Authors**: Daniele Molino, Francesco di Feola, Linlin Shen, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01091)  

**Abstract**: Generative models have revolutionized Artificial Intelligence (AI), particularly in multimodal applications. However, adapting these models to the medical domain poses unique challenges due to the complexity of medical data and the stringent need for clinical accuracy. In this work, we introduce a framework specifically designed for multimodal medical data generation. By enabling the generation of multi-view chest X-rays and their associated clinical report, it bridges the gap between general-purpose vision-language models and the specialized requirements of healthcare. Leveraging the MIMIC-CXR dataset, the proposed framework shows superior performance in generating high-fidelity images and semantically coherent reports. Our quantitative evaluation reveals significant results in terms of FID and BLEU scores, showcasing the quality of the generated data. Notably, our framework achieves comparable or even superior performance compared to real data on downstream disease classification tasks, underlining its potential as a tool for medical research and diagnostics. This study highlights the importance of domain-specific adaptations in enhancing the relevance and utility of generative models for clinical applications, paving the way for future advancements in synthetic multimodal medical data generation. 

---
