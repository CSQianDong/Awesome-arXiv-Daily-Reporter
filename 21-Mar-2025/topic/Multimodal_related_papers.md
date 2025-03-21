# OmniGeo: Towards a Multimodal Large Language Models for Geospatial Artificial Intelligence 

**Authors**: Long Yuan, Fengran Mo, Kaiyu Huang, Wenjie Wang, Wangyuxuan Zhai, Xiaoyu Zhu, You Li, Jinan Xu, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.16326)  

**Abstract**: The rapid advancement of multimodal large language models (LLMs) has opened new frontiers in artificial intelligence, enabling the integration of diverse large-scale data types such as text, images, and spatial information. In this paper, we explore the potential of multimodal LLMs (MLLM) for geospatial artificial intelligence (GeoAI), a field that leverages spatial data to address challenges in domains including Geospatial Semantics, Health Geography, Urban Geography, Urban Perception, and Remote Sensing. We propose a MLLM (OmniGeo) tailored to geospatial applications, capable of processing and analyzing heterogeneous data sources, including satellite imagery, geospatial metadata, and textual descriptions. By combining the strengths of natural language understanding and spatial reasoning, our model enhances the ability of instruction following and the accuracy of GeoAI systems. Results demonstrate that our model outperforms task-specific models and existing LLMs on diverse geospatial tasks, effectively addressing the multimodality nature while achieving competitive results on the zero-shot geospatial tasks. Our code will be released after publication. 

---
# Video-VoT-R1: An efficient video inference model integrating image packing and AoE architecture 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Yanqin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.15807)  

**Abstract**: In the field of video-language pretraining, existing models face numerous challenges in terms of inference efficiency and multimodal data processing. This paper proposes a KunLunBaize-VoT-R1 video inference model based on a long-sequence image encoder, along with its training and application methods. By integrating image packing technology, the Autonomy-of-Experts (AoE) architecture, and combining the video of Thought (VoT), a large language model (LLM) trained with large-scale reinforcement learning, and multiple training techniques, the efficiency and accuracy of the model in video inference tasks are effectively improved. Experiments show that this model performs outstandingly in multiple tests, providing a new solution for video-language understanding. 

---
# Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning 

**Authors**: NVIDIA, Alisson Azzolini, Hannah Brandon, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, Francesco Ferroni, Rama Govindaraju, Jinwei Gu, Siddharth Gururani, Imad El Hanafi, Zekun Hao, Jacob Huffman, Jingyi Jin, Brendan Johnson, Rizwan Khan, George Kurian, Elena Lantz, Nayeon Lee, Zhaoshuo Li, Xuan Li, Tsung-Yi Lin, Yen-Chen Lin, Ming-Yu Liu, Andrew Mathau, Yun Ni, Lindsey Pavao, Wei Ping, David W. Romero, Misha Smelyanskiy, Shuran Song, Lyne Tchapmi, Andrew Z. Wang, Boxin Wang, Haoxiang Wang, Fangyin Wei, Jiashu Xu, Yao Xu, Xiaodong Yang, Zhuolin Yang, Xiaohui Zeng, Zhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15558)  

**Abstract**: Physical AI systems need to perceive, understand, and perform complex actions in the physical world. In this paper, we present the Cosmos-Reason1 models that can understand the physical world and generate appropriate embodied decisions (e.g., next step action) in natural language through long chain-of-thought reasoning processes. We begin by defining key capabilities for Physical AI reasoning, with a focus on physical common sense and embodied reasoning. To represent physical common sense, we use a hierarchical ontology that captures fundamental knowledge about space, time, and physics. For embodied reasoning, we rely on a two-dimensional ontology that generalizes across different physical embodiments. Building on these capabilities, we develop two multimodal large language models, Cosmos-Reason1-8B and Cosmos-Reason1-56B. We curate data and train our models in four stages: vision pre-training, general supervised fine-tuning (SFT), Physical AI SFT, and Physical AI reinforcement learning (RL) as the post-training. To evaluate our models, we build comprehensive benchmarks for physical common sense and embodied reasoning according to our ontologies. Evaluation results show that Physical AI SFT and reinforcement learning bring significant improvements. To facilitate the development of Physical AI, we will make our code and pre-trained models available under the NVIDIA Open Model License at this https URL. 

---
# Do Visual Imaginations Improve Vision-and-Language Navigation Agents? 

**Authors**: Akhil Perincherry, Jacob Krantz, Stefan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16394)  

**Abstract**: Vision-and-Language Navigation (VLN) agents are tasked with navigating an unseen environment using natural language instructions. In this work, we study if visual representations of sub-goals implied by the instructions can serve as navigational cues and lead to increased navigation performance. To synthesize these visual representations or imaginations, we leverage a text-to-image diffusion model on landmark references contained in segmented instructions. These imaginations are provided to VLN agents as an added modality to act as landmark cues and an auxiliary loss is added to explicitly encourage relating these with their corresponding referring expressions. Our findings reveal an increase in success rate (SR) of around 1 point and up to 0.5 points in success scaled by inverse path length (SPL) across agents. These results suggest that the proposed approach reinforces visual understanding compared to relying on language instructions alone. Code and data for our work can be found at this https URL. 

---
# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse 

**Authors**: Muyao Li, Zihao Wang, Kaichen He, Xiaojian Ma, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16365)  

**Abstract**: Recently, action-based decision-making in open-world environments has gained significant attention. Visual Language Action (VLA) models, pretrained on large-scale web datasets, have shown promise in decision-making tasks. However, previous work has primarily focused on action post-training, often neglecting enhancements to the foundational model itself. In response, we introduce a novel approach, Act from Visual Language Post-Training, which refines Visual Language Models (VLMs) through visual and linguistic guidance in a self-supervised manner. This enhancement improves the models' capabilities in world knowledge, visual recognition, and spatial grounding in open-world environments. Following the above post-training paradigms, we obtain the first VLA models in Minecraft that can follow human instructions on over 1k different atomic tasks, including crafting, smelting, cooking, mining, and killing. Our experiments demonstrate that post-training on non-trajectory tasks leads to a significant 40% improvement over the best agent baseline on a diverse set of atomic tasks. Furthermore, we demonstrate that our approach surpasses traditional imitation learning-based policies in Minecraft, achieving state-of-the-art performance. We have open-sourced the code, models, and datasets to foster further research. The project page can be found in this https URL. 

---
# Structured-Noise Masked Modeling for Video, Audio and Beyond 

**Authors**: Aritra Bhowmik, Fida Mohammad Thoker, Carlos Hinojosa, Bernard Ghanem, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2503.16311)  

**Abstract**: Masked modeling has emerged as a powerful self-supervised learning framework, but existing methods largely rely on random masking, disregarding the structural properties of different modalities. In this work, we introduce structured noise-based masking, a simple yet effective approach that naturally aligns with the spatial, temporal, and spectral characteristics of video and audio data. By filtering white noise into distinct color noise distributions, we generate structured masks that preserve modality-specific patterns without requiring handcrafted heuristics or access to the data. Our approach improves the performance of masked video and audio modeling frameworks without any computational overhead. Extensive experiments demonstrate that structured noise masking achieves consistent improvement over random masking for standard and advanced masked modeling methods, highlighting the importance of modality-aware masking strategies for representation learning. 

---
# Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models 

**Authors**: Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.16036)  

**Abstract**: Recent Multi-modal Large Language Models (MLLMs) have been challenged by the computational overhead resulting from massive video frames, often alleviated through compression strategies. However, the visual content is not equally contributed to user instructions, existing strategies (\eg, average pool) inevitably lead to the loss of potentially useful information. To tackle this, we propose the Hybrid-level Instruction Injection Strategy for Conditional Token Compression in MLLMs (HICom), utilizing the instruction as a condition to guide the compression from both local and global levels. This encourages the compression to retain the maximum amount of user-focused information while reducing visual tokens to minimize computational burden. Specifically, the instruction condition is injected into the grouped visual tokens at the local level and the learnable tokens at the global level, and we conduct the attention mechanism to complete the conditional compression. From the hybrid-level compression, the instruction-relevant visual parts are highlighted while the temporal-spatial structure is also preserved for easier understanding of LLMs. To further unleash the potential of HICom, we introduce a new conditional pre-training stage with our proposed dataset HICom-248K. Experiments show that our HICom can obtain distinguished video understanding ability with fewer tokens, increasing the performance by 2.43\% average on three multiple-choice QA benchmarks and saving 78.8\% tokens compared with the SOTA method. The code is available at this https URL. 

---
# Beyond the Visible: Multispectral Vision-Language Learning for Earth Observation 

**Authors**: Clive Tinashe Marimo, Benedikt Blumenstiel, Maximilian Nitsche, Johannes Jakubik, Thomas Brunschwiler  

**Link**: [PDF](https://arxiv.org/pdf/2503.15969)  

**Abstract**: Vision-language models for Earth observation (EO) typically rely on the visual spectrum of data as the only model input, thus failing to leverage the rich spectral information available in the multispectral channels recorded by satellites. Therefore, in this paper, we introduce Llama3-MS-CLIP, the first vision-language model pre-trained with contrastive learning on a large-scale multispectral dataset and report on the performance gains due to the extended spectral range. Furthermore, we present the largest-to-date image-caption dataset for multispectral data, consisting of one million Sentinel-2 samples and corresponding textual descriptions generated with Llama3-LLaVA-Next and Overture Maps data. We develop a scalable captioning pipeline, which is validated by domain experts. We evaluate Llama3-MS-CLIP on multispectral zero-shot image classification and retrieval using three datasets of varying complexity. Our results demonstrate that Llama3-MS-CLIP significantly outperforms other RGB-based approaches, improving classification accuracy by 6.77% on average and retrieval performance by 4.63% mAP compared to the second-best model. Our results emphasize the relevance of multispectral vision-language learning. We release the image-caption dataset, code, and model weights under an open-source license. 

---
# TruthLens: Explainable DeepFake Detection for Face Manipulated and Fully Synthetic Data 

**Authors**: Rohit Kundu, Athula Balachandran, Amit K. Roy-Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.15867)  

**Abstract**: Detecting DeepFakes has become a crucial research area as the widespread use of AI image generators enables the effortless creation of face-manipulated and fully synthetic content, yet existing methods are often limited to binary classification (real vs. fake) and lack interpretability. To address these challenges, we propose TruthLens, a novel and highly generalizable framework for DeepFake detection that not only determines whether an image is real or fake but also provides detailed textual reasoning for its predictions. Unlike traditional methods, TruthLens effectively handles both face-manipulated DeepFakes and fully AI-generated content while addressing fine-grained queries such as "Does the eyes/nose/mouth look real or fake?"
The architecture of TruthLens combines the global contextual understanding of multimodal large language models like PaliGemma2 with the localized feature extraction capabilities of vision-only models like DINOv2. This hybrid design leverages the complementary strengths of both models, enabling robust detection of subtle manipulations while maintaining interpretability. Extensive experiments on diverse datasets demonstrate that TruthLens outperforms state-of-the-art methods in detection accuracy (by 2-14%) and explainability, in both in-domain and cross-data settings, generalizing effectively across traditional and emerging manipulation techniques. 

---
# VideoRFSplat: Direct Scene-Level Text-to-3D Gaussian Splatting Generation with Flexible Pose and Multi-View Joint Modeling 

**Authors**: Hyojun Go, Byeongjun Park, Hyelin Nam, Byung-Hoon Kim, Hyungjin Chung, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15855)  

**Abstract**: We propose VideoRFSplat, a direct text-to-3D model leveraging a video generation model to generate realistic 3D Gaussian Splatting (3DGS) for unbounded real-world scenes. To generate diverse camera poses and unbounded spatial extent of real-world scenes, while ensuring generalization to arbitrary text prompts, previous methods fine-tune 2D generative models to jointly model camera poses and multi-view images. However, these methods suffer from instability when extending 2D generative models to joint modeling due to the modality gap, which necessitates additional models to stabilize training and inference. In this work, we propose an architecture and a sampling strategy to jointly model multi-view images and camera poses when fine-tuning a video generation model. Our core idea is a dual-stream architecture that attaches a dedicated pose generation model alongside a pre-trained video generation model via communication blocks, generating multi-view images and camera poses through separate streams. This design reduces interference between the pose and image modalities. Additionally, we propose an asynchronous sampling strategy that denoises camera poses faster than multi-view images, allowing rapidly denoised poses to condition multi-view generation, reducing mutual ambiguity and enhancing cross-modal consistency. Trained on multiple large-scale real-world datasets (RealEstate10K, MVImgNet, DL3DV-10K, ACID), VideoRFSplat outperforms existing text-to-3D direct generation methods that heavily depend on post-hoc refinement via score distillation sampling, achieving superior results without such refinement. 

---
# LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning 

**Authors**: Federico Cocchi, Nicholas Moratelli, Davide Caffagni, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.15621)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has highlighted the critical roles of both the visual backbone and the underlying language model. While prior work has primarily focused on scaling these components to billions of parameters, the trade-offs between model size, architecture, and performance remain underexplored. Additionally, inconsistencies in training data and evaluation protocols have hindered direct comparisons, making it difficult to derive optimal design choices. In this paper, we introduce LLaVA-MORE, a new family of MLLMs that integrates recent language models with diverse visual backbones. To ensure fair comparisons, we employ a unified training protocol applied consistently across all architectures. Our analysis systematically explores both small- and medium-scale LLMs -- including Phi-4, LLaMA-3.1, and Gemma-2 -- to evaluate multimodal reasoning, generation, and instruction following, while examining the relationship between model size and performance. Beyond evaluating the LLM impact on final results, we conduct a comprehensive study of various visual encoders, ranging from CLIP-based architectures to alternatives such as DINOv2, SigLIP, and SigLIP2. Additional experiments investigate the effects of increased image resolution and variations in pre-training datasets. Overall, our results provide insights into the design of more effective MLLMs, offering a reproducible evaluation framework that facilitates direct comparisons and can guide future model development. Our source code and trained models are publicly available at: this https URL. 

---
# There must be encapsulated nonconceptual content in vision 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.15538)  

**Abstract**: In this paper I want to propose an argument to support Jerry Fodor's thesis (Fodor 1983) that input systems are modular and thus informationally encapsulated. The argument starts with the suggestion that there is a "grounding problem" in perception, i. e. that there is a problem in explaining how perception that can yield a visual experience is possible, how sensation can become meaningful perception of something for the subject. Given that visual experience is actually possible, this invites a transcendental argument that explains the conditions of its possibility. I propose that one of these conditions is the existence of a visual module in Fodor's sense that allows the step from sensation to object-identifying perception, thus enabling visual experience. It seems to follow that there is informationally encapsulated nonconceptual content in visual perception. 

---
# AI-Powered Assistive Technologies for Visual Impairment 

**Authors**: Prudhvi Naayini, Praveen Kumar Myakala, Chiranjeevi Bura, Anil Kumar Jonnalagadda, Srikanth Kamatala  

**Link**: [PDF](https://arxiv.org/pdf/2503.15494)  

**Abstract**: Artificial Intelligence (AI) is revolutionizing assistive technologies. It offers innovative solutions to enhance the quality of life for individuals with visual impairments. This review examines the development, applications, and impact of AI-powered tools in key domains, such as computer vision, natural language processing (NLP), and wearable devices. Specific advancements include object recognition for identifying everyday items, scene description for understanding surroundings, and NLP-driven text-to-speech systems for accessing digital information. Assistive technologies like smart glasses, smartphone applications, and AI-enabled navigation aids are discussed, demonstrating their ability to support independent travel, facilitate social interaction, and increase access to education and employment opportunities.
The integration of deep learning models, multimodal interfaces, and real-time data processing has transformed the functionality and usability of these tools, fostering inclusivity and empowerment. This article also addresses critical challenges, including ethical considerations, affordability, and adaptability in diverse environments. Future directions highlight the need for interdisciplinary collaboration to refine these technologies, ensuring equitable access and sustainable innovation. By providing a comprehensive overview, this review underscores AI's transformative potential in promoting independence, enhancing accessibility, and fostering social inclusion for visually impaired individuals. 

---
# Agreeing to Interact in Human-Robot Interaction using Large Language Models and Vision Language Models 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15491)  

**Abstract**: In human-robot interaction (HRI), the beginning of an interaction is often complex. Whether the robot should communicate with the human is dependent on several situational factors (e.g., the current human's activity, urgency of the interaction, etc.). We test whether large language models (LLM) and vision language models (VLM) can provide solutions to this problem. We compare four different system-design patterns using LLMs and VLMs, and test on a test set containing 84 human-robot situations. The test set mixes several publicly available datasets and also includes situations where the appropriate action to take is open-ended. Our results using the GPT-4o and Phi-3 Vision model indicate that LLMs and VLMs are capable of handling interaction beginnings when the desired actions are clear, however, challenge remains in the open-ended situations where the model must balance between the human and robot situation. 

---
# OThink-MR1: Stimulating multimodal generalized reasoning capabilities through dynamic reinforcement learning 

**Authors**: Zhiyuan Liu, Yuting Zhang, Feng Liu, Changwang Zhang, Ying Sun, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16081)  

**Abstract**: Multimodal Language Models have gained significant traction for their ability to process diverse input data types and generate coherent, contextually relevant outputs across various applications. While supervised fine-tuning (SFT) has been the predominant approach to enhance MLLM capabilities in task-specific optimization, it often falls short in fostering crucial generalized reasoning abilities. Despite the potential of reinforcement learning (RL) to address these limitations, it faces two issues: (1) its generalized capabilities in multimodal tasks remain underexplored. (2) its training constraints such as constant Kullback-Leibler or clamp strategy easily lead to suboptimal bottleneck. To adress these issues, we introduce OThink-MR1, a framework that extends RL to MLLMs, enabling them to achieve deeper understanding and reasoning across multimodal tasks. We design a dynamic Kullback-Leibler strategy that significantly enhances RL performance, surpassing SFT in same-task evaluations. Also, we are the first to reveal that RL exhibits remarkable cross-task generalization capabilities, which shows that models post-trained with RL on one multimodal task can be effectively transfered to another tasks. Finally, extensive experiments demonstrate the great reasoning ability of our proposed OThink-MR1. 

---
