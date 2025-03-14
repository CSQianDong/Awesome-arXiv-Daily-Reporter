# SurgRAW: Multi-Agent Workflow with Chain-of-Thought Reasoning for Surgical Intelligence 

**Authors**: Chang Han Low, Ziyue Wang, Tianyi Zhang, Zhitao Zeng, Zhu Zhuo, Evangelos B. Mazomenos, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10265)  

**Abstract**: Integration of Vision-Language Models (VLMs) in surgical intelligence is hindered by hallucinations, domain knowledge gaps, and limited understanding of task interdependencies within surgical scenes, undermining clinical reliability. While recent VLMs demonstrate strong general reasoning and thinking capabilities, they still lack the domain expertise and task-awareness required for precise surgical scene interpretation. Although Chain-of-Thought (CoT) can structure reasoning more effectively, current approaches rely on self-generated CoT steps, which often exacerbate inherent domain gaps and hallucinations. To overcome this, we present SurgRAW, a CoT-driven multi-agent framework that delivers transparent, interpretable insights for most tasks in robotic-assisted surgery. By employing specialized CoT prompts across five tasks: instrument recognition, action recognition, action prediction, patient data extraction, and outcome assessment, SurgRAW mitigates hallucinations through structured, domain-aware reasoning. Retrieval-Augmented Generation (RAG) is also integrated to external medical knowledge to bridge domain gaps and improve response reliability. Most importantly, a hierarchical agentic system ensures that CoT-embedded VLM agents collaborate effectively while understanding task interdependencies, with a panel discussion mechanism promotes logical consistency. To evaluate our method, we introduce SurgCoTBench, the first reasoning-based dataset with structured frame-level annotations. With comprehensive experiments, we demonstrate the effectiveness of proposed SurgRAW with 29.32% accuracy improvement over baseline VLMs on 12 robotic procedures, achieving the state-of-the-art performance and advancing explainable, trustworthy, and autonomous surgical assistance. 

---
# SciVerse: Unveiling the Knowledge Comprehension and Visual Reasoning of LMMs on Multi-modal Scientific Problems 

**Authors**: Ziyu Guo, Ray Zhang, Hao Chen, Jialin Gao, Dongzhi Jiang, Jiaze Wang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10627)  

**Abstract**: The rapid advancement of Large Multi-modal Models (LMMs) has enabled their application in scientific problem-solving, yet their fine-grained capabilities remain under-explored. In this paper, we introduce SciVerse, a multi-modal scientific evaluation benchmark to thoroughly assess LMMs across 5,735 test instances in five distinct versions. We aim to investigate three key dimensions of LMMs: scientific knowledge comprehension, multi-modal content interpretation, and Chain-of-Thought (CoT) reasoning. To unveil whether LMMs possess sufficient scientific expertise, we first transform each problem into three versions containing different levels of knowledge required for solving, i.e., Knowledge-free, -lite, and -rich. Then, to explore how LMMs interpret multi-modal scientific content, we annotate another two versions, i.e., Vision-rich and -only, marking more question information from texts to diagrams. Comparing the results of different versions, SciVerse systematically examines the professional knowledge stock and visual perception skills of LMMs in scientific domains. In addition, to rigorously assess CoT reasoning, we propose a new scientific CoT evaluation strategy, conducting a step-wise assessment on knowledge and logical errors in model outputs. Our extensive evaluation of different LMMs on SciVerse reveals critical limitations in their scientific proficiency and provides new insights into future developments. Project page: this https URL 

---
# VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search 

**Authors**: Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10582)  

**Abstract**: Vision-Language Models have made significant progress on many perception-focused tasks, however, their progress on reasoning-focused tasks seem to be limited due to the lack of high-quality and diverse training data. In this work, we aim to address the scarcity issue of reasoning-focused multimodal datasets. We propose VisualWebInstruct - a novel approach that leverages search engine to create a diverse, and high-quality dataset spanning multiple disciplines like math, physics, finance, chemistry, etc. Starting with meticulously selected 30,000 seed images, we employ Google Image search to identify websites containing similar images. We collect and process the HTMLs from over 700K unique URL sources. Through a pipeline of content extraction, filtering and synthesis, we build a dataset of approximately 900K question-answer pairs, with 40% being visual QA pairs and the rest as text QA pairs. Models fine-tuned on VisualWebInstruct demonstrate significant performance gains: (1) training from Llava-OV-mid shows 10-20% absolute point gains across benchmarks, (2) training from MAmmoTH-VL shows 5% absoluate gain. Our best model MAmmoTH-VL2 shows state-of-the-art performance within the 10B parameter class on MMMU-Pro-std (40.7%), MathVerse (42.6%), and DynaMath (55.7%). These remarkable results highlight the effectiveness of our dataset in enhancing VLMs' reasoning capabilities for complex multimodal tasks. 

---
# LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds 

**Authors**: Lingteng Qiu, Xiaodong Gu, Peihao Li, Qi Zuo, Weichao Shen, Junfei Zhang, Kejie Qiu, Weihao Yuan, Guanying Chen, Zilong Dong, Liefeng Bo  

**Link**: [PDF](https://arxiv.org/pdf/2503.10625)  

**Abstract**: Animatable 3D human reconstruction from a single image is a challenging problem due to the ambiguity in decoupling geometry, appearance, and deformation. Recent advances in 3D human reconstruction mainly focus on static human modeling, and the reliance of using synthetic 3D scans for training limits their generalization ability. Conversely, optimization-based video methods achieve higher fidelity but demand controlled capture conditions and computationally intensive refinement processes. Motivated by the emergence of large reconstruction models for efficient static reconstruction, we propose LHM (Large Animatable Human Reconstruction Model) to infer high-fidelity avatars represented as 3D Gaussian splatting in a feed-forward pass. Our model leverages a multimodal transformer architecture to effectively encode the human body positional features and image features with attention mechanism, enabling detailed preservation of clothing geometry and texture. To further boost the face identity preservation and fine detail recovery, we propose a head feature pyramid encoding scheme to aggregate multi-scale features of the head regions. Extensive experiments demonstrate that our LHM generates plausible animatable human in seconds without post-processing for face and hands, outperforming existing methods in both reconstruction accuracy and generalization ability. 

---
# PiSA: A Self-Augmented Data Engine and Training Strategy for 3D Understanding with Large Models 

**Authors**: Zilu Guo, Hongbin Lin, Zhihao Yuan, Chaoda Zheng, Pengshuo Qiu, Dongzhi Jiang, Renrui Zhang, Chun-Mei Feng, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10529)  

**Abstract**: 3D Multimodal Large Language Models (MLLMs) have recently made substantial advancements. However, their potential remains untapped, primarily due to the limited quantity and suboptimal quality of 3D datasets. Current approaches attempt to transfer knowledge from 2D MLLMs to expand 3D instruction data, but still face modality and domain gaps. To this end, we introduce PiSA-Engine (Point-Self-Augmented-Engine), a new framework for generating instruction point-language datasets enriched with 3D spatial semantics. We observe that existing 3D MLLMs offer a comprehensive understanding of point clouds for annotation, while 2D MLLMs excel at cross-validation by providing complementary information. By integrating holistic 2D and 3D insights from off-the-shelf MLLMs, PiSA-Engine enables a continuous cycle of high-quality data generation. We select PointLLM as the baseline and adopt this co-evolution training framework to develop an enhanced 3D MLLM, termed PointLLM-PiSA. Additionally, we identify limitations in previous 3D benchmarks, which often feature coarse language captions and insufficient category diversity, resulting in inaccurate evaluations. To address this gap, we further introduce PiSA-Bench, a comprehensive 3D benchmark covering six key aspects with detailed and diverse labels. Experimental results demonstrate PointLLM-PiSA's state-of-the-art performance in zero-shot 3D object captioning and generative classification on our PiSA-Bench, achieving significant improvements of 46.45% (+8.33%) and 63.75% (+16.25%), respectively. We will release the code, datasets, and benchmark. 

---
# Dual-Stage Cross-Modal Network with Dynamic Feature Fusion for Emotional Mimicry Intensity Estimation 

**Authors**: Jun Yu, Lingsi Zhu, Yanjun Chi, Yunxiang Zhang, Yang Zheng, Yongqi Wang, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10603)  

**Abstract**: Emotional Mimicry Intensity (EMI) estimation serves as a critical technology for understanding human social behavior and enhancing human-computer interaction experiences, where the core challenge lies in dynamic correlation modeling and robust fusion of multimodal temporal signals. To address the limitations of existing methods in insufficient exploitation of modal synergistic effects, noise sensitivity, and limited fine-grained alignment capabilities, this paper proposes a dual-stage cross-modal alignment framework. First, we construct vision-text and audio-text contrastive learning networks based on an improved CLIP architecture, achieving preliminary alignment in the feature space through modality-decoupled pre-training. Subsequently, we design a temporal-aware dynamic fusion module that combines Temporal Convolutional Networks (TCN) and gated bidirectional LSTM to respectively capture the macro-evolution patterns of facial expressions and local dynamics of acoustic features. Innovatively, we introduce a quality-guided modality fusion strategy that enables modality compensation under occlusion and noisy scenarios through differentiable weight allocation. Experimental results on the Hume-Vidmimic2 dataset demonstrate that our method achieves an average Pearson correlation coefficient of 0.35 across six emotion dimensions, outperforming the best baseline by 40\%. Ablation studies further validate the effectiveness of the dual-stage training strategy and dynamic fusion mechanism, providing a novel technical pathway for fine-grained emotion analysis in open environments. 

---
# CINEMA: Coherent Multi-Subject Video Generation via MLLM-Based Guidance 

**Authors**: Yufan Deng, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Angtian Wang, Shenghai Yuan, Yiding Yang, Bo Liu, Haibin Huang, Chongyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10391)  

**Abstract**: Video generation has witnessed remarkable progress with the advent of deep generative models, particularly diffusion models. While existing methods excel in generating high-quality videos from text prompts or single images, personalized multi-subject video generation remains a largely unexplored challenge. This task involves synthesizing videos that incorporate multiple distinct subjects, each defined by separate reference images, while ensuring temporal and spatial consistency. Current approaches primarily rely on mapping subject images to keywords in text prompts, which introduces ambiguity and limits their ability to model subject relationships effectively. In this paper, we propose CINEMA, a novel framework for coherent multi-subject video generation by leveraging Multimodal Large Language Model (MLLM). Our approach eliminates the need for explicit correspondences between subject images and text entities, mitigating ambiguity and reducing annotation effort. By leveraging MLLM to interpret subject relationships, our method facilitates scalability, enabling the use of large and diverse datasets for training. Furthermore, our framework can be conditioned on varying numbers of subjects, offering greater flexibility in personalized content creation. Through extensive evaluations, we demonstrate that our approach significantly improves subject consistency, and overall video coherence, paving the way for advanced applications in storytelling, interactive media, and personalized video generation. 

---
# A Multimodal Fusion Model Leveraging MLP Mixer and Handcrafted Features-based Deep Learning Networks for Facial Palsy Detection 

**Authors**: Heng Yim Nicole Oo, Min Hun Lee, Jeong Hoon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.10371)  

**Abstract**: Algorithmic detection of facial palsy offers the potential to improve current practices, which usually involve labor-intensive and subjective assessments by clinicians. In this paper, we present a multimodal fusion-based deep learning model that utilizes an MLP mixer-based model to process unstructured data (i.e. RGB images or images with facial line segments) and a feed-forward neural network to process structured data (i.e. facial landmark coordinates, features of facial expressions, or handcrafted features) for detecting facial palsy. We then contribute to a study to analyze the effect of different data modalities and the benefits of a multimodal fusion-based approach using videos of 20 facial palsy patients and 20 healthy subjects. Our multimodal fusion model achieved 96.00 F1, which is significantly higher than the feed-forward neural network trained on handcrafted features alone (82.80 F1) and an MLP mixer-based model trained on raw RGB images (89.00 F1). 

---
# RealGeneral: Unifying Visual Generation via Temporal In-Context Learning with Video Models 

**Authors**: Yijing Lin, Mengqi Huang, Shuhan Zhuang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10406)  

**Abstract**: Unifying diverse image generation tasks within a single framework remains a fundamental challenge in visual generation. While large language models (LLMs) achieve unification through task-agnostic data and generation, existing visual generation models fail to meet these principles. Current approaches either rely on per-task datasets and large-scale training or adapt pre-trained image models with task-specific modifications, limiting their generalizability. In this work, we explore video models as a foundation for unified image generation, leveraging their inherent ability to model temporal correlations. We introduce RealGeneral, a novel framework that reformulates image generation as a conditional frame prediction task, analogous to in-context learning in LLMs. To bridge the gap between video models and condition-image pairs, we propose (1) a Unified Conditional Embedding module for multi-modal alignment and (2) a Unified Stream DiT Block with decoupled adaptive LayerNorm and attention mask to mitigate cross-modal interference. RealGeneral demonstrates effectiveness in multiple important visual generation tasks, e.g., it achieves a 14.5% improvement in subject similarity for customized generation and a 10% enhancement in image quality for canny-to-image task. Project page: this https URL 

---
# Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding 

**Authors**: Shunqi Mao, Chaoyi Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.10183)  

**Abstract**: Existing vision-language models (VLMs) often suffer from visual hallucination, where the generated responses contain inaccuracies that are not grounded in the visual input. Efforts to address this issue without model finetuning primarily mitigate hallucination by reducing biases contrastively or amplifying the weights of visual embedding during decoding. However, these approaches improve visual perception at the cost of impairing the language reasoning capability. In this work, we propose the Perception Magnifier (PM), a novel visual decoding method that iteratively isolates relevant visual tokens based on attention and magnifies the corresponding regions, spurring the model to concentrate on fine-grained visual details during decoding. Specifically, by magnifying critical regions while preserving the structural and contextual information at each decoding step, PM allows the VLM to enhance its scrutiny of the visual input, hence producing more accurate and faithful responses. Extensive experimental results demonstrate that PM not only achieves superior hallucination mitigation but also enhances language generation while preserving strong reasoning this http URL is available at this https URL . 

---
# On the Limitations of Vision-Language Models in Understanding Image Transforms 

**Authors**: Ahmad Mustafa Anis, Hasnain Ali, Saquib Sarfraz  

**Link**: [PDF](https://arxiv.org/pdf/2503.09837)  

**Abstract**: Vision Language Models (VLMs) have demonstrated significant potential in various downstream tasks, including Image/Video Generation, Visual Question Answering, Multimodal Chatbots, and Video Understanding. However, these models often struggle with basic image transformations. This paper investigates the image-level understanding of VLMs, specifically CLIP by OpenAI and SigLIP by Google. Our findings reveal that these models lack comprehension of multiple image-level augmentations. To facilitate this study, we created an augmented version of the Flickr8k dataset, pairing each image with a detailed description of the applied transformation. We further explore how this deficiency impacts downstream tasks, particularly in image editing, and evaluate the performance of state-of-the-art Image2Image models on simple transformations. 

---
# ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning 

**Authors**: Pengfei Luo, Jingbo Zhou, Tong Xu, Yuan Xia, Linli Xu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10166)  

**Abstract**: With the proliferation of images in online content, language-guided image retrieval (LGIR) has emerged as a research hotspot over the past decade, encompassing a variety of subtasks with diverse input forms. While the development of large multimodal models (LMMs) has significantly facilitated these tasks, existing approaches often address them in isolation, requiring the construction of separate systems for each task. This not only increases system complexity and maintenance costs, but also exacerbates challenges stemming from language ambiguity and complex image content, making it difficult for retrieval systems to provide accurate and reliable results. To this end, we propose ImageScope, a training-free, three-stage framework that leverages collective reasoning to unify LGIR tasks. The key insight behind the unification lies in the compositional nature of language, which transforms diverse LGIR tasks into a generalized text-to-image retrieval process, along with the reasoning of LMMs serving as a universal verification to refine the results. To be specific, in the first stage, we improve the robustness of the framework by synthesizing search intents across varying levels of semantic granularity using chain-of-thought (CoT) reasoning. In the second and third stages, we then reflect on retrieval results by verifying predicate propositions locally, and performing pairwise evaluations globally. Experiments conducted on six LGIR datasets demonstrate that ImageScope outperforms competitive baselines. Comprehensive evaluations and ablation studies further confirm the effectiveness of our design. 

---
# Fine-tuning Vision Language Models with Graph-based Knowledge for Explainable Medical Image Analysis 

**Authors**: Chenjun Li, Laurin Lux, Alexander H. Berger, Martin J. Menten, Mert R. Sabuncu, Johannes C. Paetzold  

**Link**: [PDF](https://arxiv.org/pdf/2503.09808)  

**Abstract**: Accurate staging of Diabetic Retinopathy (DR) is essential for guiding timely interventions and preventing vision loss. However, current staging models are hardly interpretable, and most public datasets contain no clinical reasoning or interpretation beyond image-level labels. In this paper, we present a novel method that integrates graph representation learning with vision-language models (VLMs) to deliver explainable DR diagnosis. Our approach leverages optical coherence tomography angiography (OCTA) images by constructing biologically informed graphs that encode key retinal vascular features such as vessel morphology and spatial connectivity. A graph neural network (GNN) then performs DR staging while integrated gradients highlight critical nodes and edges and their individual features that drive the classification decisions. We collect this graph-based knowledge which attributes the model's prediction to physiological structures and their characteristics. We then transform it into textual descriptions for VLMs. We perform instruction-tuning with these textual descriptions and the corresponding image to train a student VLM. This final agent can classify the disease and explain its decision in a human interpretable way solely based on a single image input. Experimental evaluations on both proprietary and public datasets demonstrate that our method not only improves classification accuracy but also offers more clinically interpretable results. An expert study further demonstrates that our method provides more accurate diagnostic explanations and paves the way for precise localization of pathologies in OCTA images. 

---
# Vi-LAD: Vision-Language Attention Distillation for Socially-Aware Robot Navigation in Dynamic Environments 

**Authors**: Mohamed Elnoor, Kasun Weerakoon, Gershom Seneviratne, Jing Liang, Vignesh Rajagopal, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2503.09820)  

**Abstract**: We introduce Vision-Language Attention Distillation (Vi-LAD), a novel approach for distilling socially compliant navigation knowledge from a large Vision-Language Model (VLM) into a lightweight transformer model for real-time robotic navigation. Unlike traditional methods that rely on expert demonstrations or human-annotated datasets, Vi-LAD performs knowledge distillation and fine-tuning at the intermediate layer representation level (i.e., attention maps) by leveraging the backbone of a pre-trained vision-action model. These attention maps highlight key navigational regions in a given scene, which serve as implicit guidance for socially aware motion planning. Vi-LAD fine-tunes a transformer-based model using intermediate attention maps extracted from the pre-trained vision-action model, combined with attention-like semantic maps constructed from a large VLM. To achieve this, we introduce a novel attention-level distillation loss that fuses knowledge from both sources, generating augmented attention maps with enhanced social awareness. These refined attention maps are then utilized as a traversability costmap within a socially aware model predictive controller (MPC) for navigation. We validate our approach through real-world experiments on a Husky wheeled robot, demonstrating significant improvements over state-of-the-art (SOTA) navigation methods. Our results show up to 14.2% - 50% improvement in success rate, which highlights the effectiveness of Vi-LAD in enabling socially compliant and efficient robot navigation. 

---
# Certainly Bot Or Not? Trustworthy Social Bot Detection via Robust Multi-Modal Neural Processes 

**Authors**: Qi Wu, Yingguang Yang, hao liu, Hao Peng, Buyun He, Yutong Xia, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09626)  

**Abstract**: Social bot detection is crucial for mitigating misinformation, online manipulation, and coordinated inauthentic behavior. While existing neural network-based detectors perform well on benchmarks, they struggle with generalization due to distribution shifts across datasets and frequently produce overconfident predictions for out-of-distribution accounts beyond the training data. To address this, we introduce a novel Uncertainty Estimation for Social Bot Detection (UESBD) framework, which quantifies the predictive uncertainty of detectors beyond mere classification. For this task, we propose Robust Multi-modal Neural Processes (RMNP), which aims to enhance the robustness of multi-modal neural processes to modality inconsistencies caused by social bot camouflage. RMNP first learns unimodal representations through modality-specific encoders. Then, unimodal attentive neural processes are employed to encode the Gaussian distribution of unimodal latent variables. Furthermore, to avoid social bots stealing human features to camouflage themselves thus causing certain modalities to provide conflictive information, we introduce an evidential gating network to explicitly model the reliability of modalities. The joint latent distribution is learned through the generalized product of experts, which takes the reliability of each modality into consideration during fusion. The final prediction is obtained through Monte Carlo sampling of the joint latent distribution followed by a decoder. Experiments on three real-world benchmarks show the effectiveness of RMNP in classification and uncertainty estimation, as well as its robustness to modality conflicts. 

---
# VisTai: Benchmarking Vision-Language Models for Traditional Chinese in Taiwan 

**Authors**: Zhi Rui Tam, Ya-Ting Pai, Yen-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10427)  

**Abstract**: In this paper, we propose a comprehensive evaluation benchmark for Visual Language Models (VLM) in Traditional Chinese. Our evaluation suite, the first of its kind, contains two complementary components: (1) VisTai-MCQ, a collection of manually curated exam multi-choice questions from 21 academic subjects designed to test the broad knowledge and reasoning capabilities of VLMs; and (2) VisTai-Dialogue, an open dialogue benchmark comprising 131 image-question pairs manually created to evaluate VLMs' ability in free-form dialogue generation within Taiwanese cultural contexts. These benchmarks address a critical gap in the evaluation landscape, where existing benchmarks predominantly focus on English or Simplified Chinese, neglecting the unique linguistic and cultural aspects of Traditional Chinese used in regions like Taiwan and Hong Kong. Our analysis reveals significant performance differences across various VLMs and highlights specific challenges in processing Traditional Chinese visual content. 

---
# VisualPRM: An Effective Process Reward Model for Multimodal Reasoning 

**Authors**: Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10291)  

**Abstract**: We introduce VisualPRM, an advanced multimodal Process Reward Model (PRM) with 8B parameters, which improves the reasoning abilities of existing Multimodal Large Language Models (MLLMs) across different model scales and families with Best-of-N (BoN) evaluation strategies. Specifically, our model improves the reasoning performance of three types of MLLMs and four different model scales. Even when applied to the highly capable InternVL2.5-78B, it achieves a 5.9-point improvement across seven multimodal reasoning benchmarks. Experimental results show that our model exhibits superior performance compared to Outcome Reward Models and Self-Consistency during BoN evaluation. To facilitate the training of multimodal PRMs, we construct a multimodal process supervision dataset VisualPRM400K using an automated data pipeline. For the evaluation of multimodal PRMs, we propose VisualProcessBench, a benchmark with human-annotated step-wise correctness labels, to measure the abilities of PRMs to detect erroneous steps in multimodal reasoning tasks. We hope that our work can inspire more future research and contribute to the development of MLLMs. Our model, data, and benchmark are released in this https URL. 

---
# ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content 

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2503.09964)  

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies. 

---
