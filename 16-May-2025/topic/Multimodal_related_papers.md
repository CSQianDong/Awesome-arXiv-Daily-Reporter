# The First MPDD Challenge: Multimodal Personality-aware Depression Detection 

**Authors**: Changzeng Fu, Zelin Fu, Xinhe Kuang, Jiacheng Dong, Qi Zhang, Kaifeng Su, Yikai Su, Wenbo Shi, Junfeng Yao, Yuliang Zhao, Shiqi Zhao, Jiadong Wang, Siyang Song, Chaoran Liu, Yuichiro Yoshikawa, Björn Schuller, Hiroshi Ishiguro  

**Link**: [PDF](https://arxiv.org/pdf/2505.10034)  

**Abstract**: Depression is a widespread mental health issue affecting diverse age groups, with notable prevalence among college students and the elderly. However, existing datasets and detection methods primarily focus on young adults, neglecting the broader age spectrum and individual differences that influence depression manifestation. Current approaches often establish a direct mapping between multimodal data and depression indicators, failing to capture the complexity and diversity of depression across individuals. This challenge includes two tracks based on age-specific subsets: Track 1 uses the MPDD-Elderly dataset for detecting depression in older adults, and Track 2 uses the MPDD-Young dataset for detecting depression in younger participants. The Multimodal Personality-aware Depression Detection (MPDD) Challenge aims to address this gap by incorporating multimodal data alongside individual difference factors. We provide a baseline model that fuses audio and video modalities with individual difference information to detect depression manifestations in diverse populations. This challenge aims to promote the development of more personalized and accurate de pression detection methods, advancing mental health research and fostering inclusive detection systems. More details are available on the official challenge website: this https URL. 

---
# A Multimodal Multi-Agent Framework for Radiology Report Generation 

**Authors**: Ziruo Yi, Ting Xiao, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09787)  

**Abstract**: Radiology report generation (RRG) aims to automatically produce diagnostic reports from medical images, with the potential to enhance clinical workflows and reduce radiologists' workload. While recent approaches leveraging multimodal large language models (MLLMs) and retrieval-augmented generation (RAG) have achieved strong results, they continue to face challenges such as factual inconsistency, hallucination, and cross-modal misalignment. We propose a multimodal multi-agent framework for RRG that aligns with the stepwise clinical reasoning workflow, where task-specific agents handle retrieval, draft generation, visual analysis, refinement, and synthesis. Experimental results demonstrate that our approach outperforms a strong baseline in both automatic metrics and LLM-based evaluations, producing more accurate, structured, and interpretable reports. This work highlights the potential of clinically aligned multi-agent frameworks to support explainable and trustworthy clinical AI applications. 

---
# UniEval: Unified Holistic Evaluation for Unified Multimodal Understanding and Generation 

**Authors**: Yi Li, Haonan Wang, Qixiang Zhang, Boyu Xiao, Chenchang Hu, Hualiang Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10483)  

**Abstract**: The emergence of unified multimodal understanding and generation models is rapidly attracting attention because of their ability to enhance instruction-following capabilities while minimizing model redundancy. However, there is a lack of a unified evaluation framework for these models, which would enable an elegant, simplified, and overall evaluation. Current models conduct evaluations on multiple task-specific benchmarks, but there are significant limitations, such as the lack of overall results, errors from extra evaluation models, reliance on extensive labeled images, benchmarks that lack diversity, and metrics with limited capacity for instruction-following evaluation. To tackle these challenges, we introduce UniEval, the first evaluation framework designed for unified multimodal models without extra models, images, or annotations. This facilitates a simplified and unified evaluation process. The UniEval framework contains a holistic benchmark, UniBench (supports both unified and visual generation models), along with the corresponding UniScore metric. UniBench includes 81 fine-grained tags contributing to high diversity. Experimental results indicate that UniBench is more challenging than existing benchmarks, and UniScore aligns closely with human evaluations, surpassing current metrics. Moreover, we extensively evaluated SoTA unified and visual generation models, uncovering new insights into Univeral's unique values. 

---
# Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning 

**Authors**: Milan Ganai, Rohan Sinha, Christopher Agia, Daniel Morton, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.10547)  

**Abstract**: Foundation models can provide robust high-level reasoning on appropriate safety interventions in hazardous scenarios beyond a robot's training data, i.e. out-of-distribution (OOD) failures. However, due to the high inference latency of Large Vision and Language Models, current methods rely on manually defined intervention policies to enact fallbacks, thereby lacking the ability to plan generalizable, semantically safe motions. To overcome these challenges we present FORTRESS, a framework that generates and reasons about semantically safe fallback strategies in real time to prevent OOD failures. At a low frequency in nominal operations, FORTRESS uses multi-modal reasoners to identify goals and anticipate failure modes. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. 

---
# MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning 

**Authors**: Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10557)  

**Abstract**: Natural language image-caption datasets, widely used for training Large Multimodal Models, mainly focus on natural scenarios and overlook the intricate details of mathematical figures that are critical for problem-solving, hindering the advancement of current LMMs in multimodal mathematical reasoning. To this end, we propose leveraging code as supervision for cross-modal alignment, since code inherently encodes all information needed to generate corresponding figures, establishing a precise connection between the two modalities. Specifically, we co-develop our image-to-code model and dataset with model-in-the-loop approach, resulting in an image-to-code model, FigCodifier and ImgCode-8.6M dataset, the largest image-code dataset to date. Furthermore, we utilize FigCodifier to synthesize novel mathematical figures and then construct MM-MathInstruct-3M, a high-quality multimodal math instruction fine-tuning dataset. Finally, we present MathCoder-VL, trained with ImgCode-8.6M for cross-modal alignment and subsequently fine-tuned on MM-MathInstruct-3M for multimodal math problem solving. Our model achieves a new open-source SOTA across all six metrics. Notably, it surpasses GPT-4o and Claude 3.5 Sonnet in the geometry problem-solving subset of MathVista, achieving improvements of 8.9% and 9.2%. The dataset and models will be released at this https URL. 

---
# Vision language models have difficulty recognizing virtual objects 

**Authors**: Tyler Tran, Sangeet Khemlani, J.G. Trafton  

**Link**: [PDF](https://arxiv.org/pdf/2505.10453)  

**Abstract**: Vision language models (VLMs) are AI systems paired with both language and vision encoders to process multimodal input. They are capable of performing complex semantic tasks such as automatic captioning, but it remains an open question about how well they comprehend the visuospatial properties of scenes depicted in the images they process. We argue that descriptions of virtual objects -- objects that are not visually represented in an image -- can help test scene comprehension in these AI systems. For example, an image that depicts a person standing under a tree can be paired with the following prompt: imagine that a kite is stuck in the tree. VLMs that comprehend the scene should update their representations and reason sensibly about the spatial relations between all three objects. We describe systematic evaluations of state-of-the-art VLMs and show that their ability to process virtual objects is inadequate. 

---
# EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation 

**Authors**: Zibin Dong, Fei Ni, Yifu Yuan, Yinchuan Li, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10105)  

**Abstract**: We present EmbodiedMAE, a unified 3D multi-modal representation for robot manipulation. Current approaches suffer from significant domain gaps between training datasets and robot manipulation tasks, while also lacking model architectures that can effectively incorporate 3D information. To overcome these limitations, we enhance the DROID dataset with high-quality depth maps and point clouds, constructing DROID-3D as a valuable supplement for 3D embodied vision research. Then we develop EmbodiedMAE, a multi-modal masked autoencoder that simultaneously learns representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Trained on DROID-3D, EmbodiedMAE consistently outperforms state-of-the-art vision foundation models (VFMs) in both training efficiency and final performance across 70 simulation tasks and 20 real-world robot manipulation tasks on two robot platforms. The model exhibits strong scaling behavior with size and promotes effective policy learning from 3D inputs. Experimental results establish EmbodiedMAE as a reliable unified 3D multi-modal VFM for embodied AI systems, particularly in precise tabletop manipulation settings where spatial perception is critical. 

---
# LAV: Audio-Driven Dynamic Visual Generation with Neural Compression and StyleGAN2 

**Authors**: Jongmin Jung, Dasaem Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.10101)  

**Abstract**: This paper introduces LAV (Latent Audio-Visual), a system that integrates EnCodec's neural audio compression with StyleGAN2's generative capabilities to produce visually dynamic outputs driven by pre-recorded audio. Unlike previous works that rely on explicit feature mappings, LAV uses EnCodec embeddings as latent representations, directly transformed into StyleGAN2's style latent space via randomly initialized linear mapping. This approach preserves semantic richness in the transformation, enabling nuanced and semantically coherent audio-visual translations. The framework demonstrates the potential of using pretrained audio compression models for artistic and computational applications. 

---
# PsOCR: Benchmarking Large Multimodal Models for Optical Character Recognition in Low-resource Pashto Language 

**Authors**: Ijazul Haq, Yingjie Zhang, Irfan Ali Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10055)  

**Abstract**: This paper evaluates the performance of Large Multimodal Models (LMMs) on Optical Character Recognition (OCR) in the low-resource Pashto language. Natural Language Processing (NLP) in Pashto faces several challenges due to the cursive nature of its script and a scarcity of structured datasets. To address this, we developed a synthetic Pashto OCR dataset, PsOCR, consisting of one million images annotated with bounding boxes at word, line, and document levels, suitable for training and evaluating models based on different architectures, including Convolutional Neural Networks (CNNs) and Transformers. PsOCR covers variations across 1,000 unique font families, colors, image sizes, and layouts. A benchmark subset of 10K images was selected to evaluate the performance of several LMMs, including seven open-source models: DeepSeek's Janus, InternVL, MiniCPM, Florence, and Qwen (3B and 7B), and four closed-source models: GPT-4o, Gemini, Claude, and Grok. Experimental results demonstrate that Gemini achieves the best performance among all models, whereas among open-source models, Qwen-7B stands out. This work provides an insightful assessment of the capabilities and limitations of current LMMs for OCR tasks in Pashto and establishes a foundation for further research not only in Pashto OCR but also for other similar scripts such as Arabic, Persian, and Urdu. PsOCR is available at this https URL. 

---
# Adversarial Attacks in Multimodal Systems: A Practitioner's Survey 

**Authors**: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj  

**Link**: [PDF](https://arxiv.org/pdf/2505.03084)  

**Abstract**: The introduction of multimodal models is a huge step forward in Artificial Intelligence. A single model is trained to understand multiple modalities: text, image, video, and audio. Open-source multimodal models have made these breakthroughs more accessible. However, considering the vast landscape of adversarial attacks across these modalities, these models also inherit vulnerabilities of all the modalities, and ultimately, the adversarial threat amplifies. While broad research is available on possible attacks within or across these modalities, a practitioner-focused view that outlines attack types remains absent in the multimodal world. As more Machine Learning Practitioners adopt, fine-tune, and deploy open-source models in real-world applications, it's crucial that they can view the threat landscape and take the preventive actions necessary. This paper addresses the gap by surveying adversarial attacks targeting all four modalities: text, image, video, and audio. This survey provides a view of the adversarial attack landscape and presents how multimodal adversarial threats have evolved. To the best of our knowledge, this survey is the first comprehensive summarization of the threat landscape in the multimodal world. 

---
# AI and Generative AI Transforming Disaster Management: A Survey of Damage Assessment and Response Techniques 

**Authors**: Aman Raj, Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Dipen Pradhan, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08202)  

**Abstract**: Natural disasters, including earthquakes, wildfires and cyclones, bear a huge risk on human lives as well as infrastructure assets. An effective response to disaster depends on the ability to rapidly and efficiently assess the intensity of damage. Artificial Intelligence (AI) and Generative Artificial Intelligence (GenAI) presents a breakthrough solution, capable of combining knowledge from multiple types and sources of data, simulating realistic scenarios of disaster, and identifying emerging trends at a speed previously unimaginable. In this paper, we present a comprehensive review on the prospects of AI and GenAI in damage assessment for various natural disasters, highlighting both its strengths and limitations. We talk about its application to multimodal data such as text, image, video, and audio, and also cover major issues of data privacy, security, and ethical use of the technology during crises. The paper also recognizes the threat of Generative AI misuse, in the form of dissemination of misinformation and for adversarial attacks. Finally, we outline avenues of future research, emphasizing the need for secure, reliable, and ethical Generative AI systems for disaster management in general. We believe that this work represents the first comprehensive survey of Gen-AI techniques being used in the field of Disaster Assessment and Response. 

---
# Coherent Language Reconstruction from Brain Recordings with Flexible Multi-Modal Input Stimuli 

**Authors**: Chunyu Ye, Shaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10356)  

**Abstract**: Decoding thoughts from brain activity offers valuable insights into human cognition and enables promising applications in brain-computer interaction. While prior studies have explored language reconstruction from fMRI data, they are typically limited to single-modality inputs such as images or audio. In contrast, human thought is inherently multimodal. To bridge this gap, we propose a unified and flexible framework for reconstructing coherent language from brain recordings elicited by diverse input modalities-visual, auditory, and textual. Our approach leverages visual-language models (VLMs), using modality-specific experts to jointly interpret information across modalities. Experiments demonstrate that our method achieves performance comparable to state-of-the-art systems while remaining adaptable and extensible. This work advances toward more ecologically valid and generalizable mind decoding. 

---
# MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models 

**Authors**: Mugilan Ganesan, Shane Segal, Ankur Aggarwal, Nish Sinnadurai, Sean Lie, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10526)  

**Abstract**: Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM's vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x on visually-grounded tasks. MASSV provides a scalable, architecture-compatible method for accelerating both current and future VLMs. 

---
# A Survey on Large Language Models in Multimodal Recommender Systems 

**Authors**: Alejo Lopez-Avila, Jinhua Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.09777)  

**Abstract**: Multimodal recommender systems (MRS) integrate heterogeneous user and item data, such as text, images, and structured information, to enhance recommendation performance. The emergence of large language models (LLMs) introduces new opportunities for MRS by enabling semantic reasoning, in-context learning, and dynamic input handling. Compared to earlier pre-trained language models (PLMs), LLMs offer greater flexibility and generalisation capabilities but also introduce challenges related to scalability and model accessibility. This survey presents a comprehensive review of recent work at the intersection of LLMs and MRS, focusing on prompting strategies, fine-tuning methods, and data adaptation techniques. We propose a novel taxonomy to characterise integration patterns, identify transferable techniques from related recommendation domains, provide an overview of evaluation metrics and datasets, and point to possible future directions. We aim to clarify the emerging role of LLMs in multimodal recommendation and support future research in this rapidly evolving field. 

---
