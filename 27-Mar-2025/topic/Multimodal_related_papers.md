# UniEDU: A Unified Language and Vision Assistant for Education Applications 

**Authors**: Zhendong Chu, Jian Xie, Shen Wang, Zichao Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20701)  

**Abstract**: Education materials for K-12 students often consist of multiple modalities, such as text and images, posing challenges for models to fully understand nuanced information in these materials. In this paper, we propose a unified language and vision assistant UniEDU designed for various educational applications, including knowledge recommendation, knowledge tracing, time cost prediction, and user answer prediction, all within a single model. Unlike conventional task-specific models, UniEDU offers a unified solution that excels across multiple educational tasks while maintaining strong generalization capabilities. Its adaptability makes it well-suited for real-world deployment in diverse learning environments. Furthermore, UniEDU is optimized for industry-scale deployment by significantly reducing computational overhead-achieving approximately a 300\% increase in efficiency-while maintaining competitive performance with minimal degradation compared to fully fine-tuned models. This work represents a significant step toward creating versatile AI systems tailored to the evolving demands of education. 

---
# ADS-Edit: A Multimodal Knowledge Editing Dataset for Autonomous Driving Systems 

**Authors**: Chenxi Wang, Jizhan Fang, Xiang Chen, Bozhong Tian, Ziwen Xu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20756)  

**Abstract**: Recent advancements in Large Multimodal Models (LMMs) have shown promise in Autonomous Driving Systems (ADS). However, their direct application to ADS is hindered by challenges such as misunderstanding of traffic knowledge, complex road conditions, and diverse states of vehicle. To address these challenges, we propose the use of Knowledge Editing, which enables targeted modifications to a model's behavior without the need for full retraining. Meanwhile, we introduce ADS-Edit, a multimodal knowledge editing dataset specifically designed for ADS, which includes various real-world scenarios, multiple data types, and comprehensive evaluation metrics. We conduct comprehensive experiments and derive several interesting conclusions. We hope that our work will contribute to the further advancement of knowledge editing applications in the field of autonomous driving. Code and data are available in this https URL. 

---
# Qwen2.5-Omni Technical Report 

**Authors**: Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.20215)  

**Abstract**: In this report, we present Qwen2.5-Omni, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. To enable the streaming of multimodal information inputs, both audio and visual encoders utilize a block-wise processing approach. To synchronize the timestamps of video inputs with audio, we organize the audio and video sequentially in an interleaved manner and propose a novel position embedding approach, named TMRoPE(Time-aligned Multimodal RoPE). To concurrently generate text and speech while avoiding interference between the two modalities, we propose \textbf{Thinker-Talker} architecture. In this framework, Thinker functions as a large language model tasked with text generation, while Talker is a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens as output. Both the Thinker and Talker models are designed to be trained and inferred in an end-to-end manner. For decoding audio tokens in a streaming manner, we introduce a sliding-window DiT that restricts the receptive field, aiming to reduce the initial package delay. Qwen2.5-Omni is comparable with the similarly sized Qwen2.5-VL and outperforms Qwen2-Audio. Furthermore, Qwen2.5-Omni achieves state-of-the-art performance on multimodal benchmarks like Omni-Bench. Notably, Qwen2.5-Omni's performance in end-to-end speech instruction following is comparable to its capabilities with text inputs, as evidenced by benchmarks such as MMLU and GSM8K. As for speech generation, Qwen2.5-Omni's streaming Talker outperforms most existing streaming and non-streaming alternatives in robustness and naturalness. 

---
# VideoGEM: Training-free Action Grounding in Videos 

**Authors**: Felix Vogel, Walid Bousselham, Anna Kukleva, Nina Shvetsova, Hilde Kuehne  

**Link**: [PDF](https://arxiv.org/pdf/2503.20348)  

**Abstract**: Vision-language foundation models have shown impressive capabilities across various zero-shot tasks, including training-free localization and grounding, primarily focusing on localizing objects in images. However, leveraging those capabilities to localize actions and events in videos is challenging, as actions have less physical outline and are usually described by higher-level concepts. In this work, we propose VideoGEM, the first training-free spatial action grounding method based on pretrained image- and video-language backbones. Namely, we adapt the self-self attention formulation of GEM to spatial activity grounding. We observe that high-level semantic concepts, such as actions, usually emerge in the higher layers of the image- and video-language models. We, therefore, propose a layer weighting in the self-attention path to prioritize higher layers. Additionally, we introduce a dynamic weighting method to automatically tune layer weights to capture each layer`s relevance to a specific prompt. Finally, we introduce a prompt decomposition, processing action, verb, and object prompts separately, resulting in a better spatial localization of actions. We evaluate the proposed approach on three image- and video-language backbones, CLIP, OpenCLIP, and ViCLIP, and on four video grounding datasets, V-HICO, DALY, YouCook-Interactions, and GroundingYouTube, showing that the proposed training-free approach is able to outperform current trained state-of-the-art approaches for spatial video grounding. 

---
# ViLBench: A Suite for Vision-Language Process Reward Modeling 

**Authors**: Haoqin Tu, Weitao Feng, Hardy Chen, Hui Liu, Xianfeng Tang, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.20271)  

**Abstract**: Process-supervised reward models serve as a fine-grained function that provides detailed step-wise feedback to model responses, facilitating effective selection of reasoning trajectories for complex tasks. Despite its advantages, evaluation on PRMs remains less explored, especially in the multimodal domain. To address this gap, this paper first benchmarks current vision large language models (VLLMs) as two types of reward models: output reward models (ORMs) and process reward models (PRMs) on multiple vision-language benchmarks, which reveal that neither ORM nor PRM consistently outperforms across all tasks, and superior VLLMs do not necessarily yield better rewarding performance. To further advance evaluation, we introduce ViLBench, a vision-language benchmark designed to require intensive process reward signals. Notably, OpenAI's GPT-4o with Chain-of-Thought (CoT) achieves only 27.3% accuracy, indicating the benchmark's challenge for current VLLMs. Lastly, we preliminarily showcase a promising pathway towards bridging the gap between general VLLMs and reward models -- by collecting 73.6K vision-language process reward data using an enhanced tree-search algorithm, our 3B model is able to achieve an average improvement of 3.3% over standard CoT and up to 2.5% compared to its untrained counterpart on ViLBench by selecting OpenAI o1's generations. We release the implementations at this https URL with our code, model, and data. 

---
# MMMORRF: Multimodal Multilingual Modularized Reciprocal Rank Fusion 

**Authors**: Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco, Kate Sanders, Oluwaseun Eisape, Arun Reddy, Alexander Martin, Andrew Yates, Eugene Yang, Cameron Carpenter, David Etter, Efsun Kayi, Matthew Wiesner, Kenton Murray, Reno Kriz  

**Link**: [PDF](https://arxiv.org/pdf/2503.20698)  

**Abstract**: Videos inherently contain multiple modalities, including visual events, text overlays, sounds, and speech, all of which are important for retrieval. However, state-of-the-art multimodal language models like VAST and LanguageBind are built on vision-language models (VLMs), and thus overly prioritize visual signals. Retrieval benchmarks further reinforce this bias by focusing on visual queries and neglecting other modalities. We create a search system MMMORRF that extracts text and features from both visual and audio modalities and integrates them with a novel modality-aware weighted reciprocal rank fusion. MMMORRF is both effective and efficient, demonstrating practicality in searching videos based on users' information needs instead of visual descriptive queries. We evaluate MMMORRF on MultiVENT 2.0 and TVR, two multimodal benchmarks designed for more targeted information needs, and find that it improves nDCG@20 by 81% over leading multimodal encoders and 37% over single-modality retrieval, demonstrating the value of integrating diverse modalities. 

---
# From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment 

**Authors**: Yucheng Suo, Fan Ma, Linchao Zhu, Tianyi Wang, Fengyun Rao, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20472)  

**Abstract**: Multi-modal Large language models (MLLMs) show remarkable ability in video understanding. Nevertheless, understanding long videos remains challenging as the models can only process a finite number of frames in a single inference, potentially omitting crucial visual information. To address the challenge, we propose generating multiple predictions through visual context sampling, followed by a scoring mechanism to select the final prediction. Specifically, we devise a bin-wise sampling strategy that enables MLLMs to generate diverse answers based on various combinations of keyframes, thereby enriching the visual context. To determine the final prediction from the sampled answers, we employ a self-reward by linearly combining three scores: (1) a frequency score indicating the prevalence of each option, (2) a marginal confidence score reflecting the inter-intra sample certainty of MLLM predictions, and (3) a reasoning score for different question types, including clue-guided answering for global questions and temporal self-refocusing for local questions. The frequency score ensures robustness through majority correctness, the confidence-aligned score reflects prediction certainty, and the typed-reasoning score addresses cases with sparse key visual information using tailored strategies. Experiments show that this approach covers the correct answer for a high percentage of long video questions, on seven datasets show that our method improves the performance of three MLLMs. 

---
# CryoSAMU: Enhancing 3D Cryo-EM Density Maps of Protein Structures at Intermediate Resolution with Structure-Aware Multimodal U-Nets 

**Authors**: Chenwei Zhang, Anne Condon, Khanh Dao Duc  

**Link**: [PDF](https://arxiv.org/pdf/2503.20291)  

**Abstract**: Enhancing cryogenic electron microscopy (cryo-EM) 3D density maps at intermediate resolution (4-8 Å) is crucial in protein structure determination. Recent advances in deep learning have led to the development of automated approaches for enhancing experimental cryo-EM density maps. Yet, these methods are not optimized for intermediate-resolution maps and rely on map density features alone. To address this, we propose CryoSAMU, a novel method designed to enhance 3D cryo-EM density maps of protein structures using structure-aware multimodal U-Nets and trained on curated intermediate-resolution density maps. We comprehensively evaluate CryoSAMU across various metrics and demonstrate its competitive performance compared to state-of-the-art methods. Notably, CryoSAMU achieves significantly faster processing speed, showing promise for future practical applications. Our code is available at this https URL. 

---
# MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation 

**Authors**: Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20384)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in understanding complex language and visual data, enabling generalist robotic systems to interpret instructions and perform embodied tasks. Nevertheless, their real-world deployment is hindered by substantial computational and storage demands. Recent insights into the homogeneous patterns in the LLM layer have inspired sparsification techniques to address these challenges, such as early exit and token pruning. However, these methods often neglect the critical role of the final layers that encode the semantic information most relevant to downstream robotic tasks. Aligning with the recent breakthrough of the Shallow Brain Hypothesis (SBH) in neuroscience and the mixture of experts in model sparsification, we conceptualize each LLM layer as an expert and propose a Mixture-of-Layers Vision-Language-Action model (MoLe-VLA, or simply MoLe) architecture for dynamic LLM layer activation. We introduce a Spatial-Temporal Aware Router (STAR) for MoLe to selectively activate only parts of the layers based on the robot's current state, mimicking the brain's distinct signal pathways specialized for cognition and causal reasoning. Additionally, to compensate for the cognitive ability of LLMs lost in MoLe, we devise a Cognition Self-Knowledge Distillation (CogKD) framework. CogKD enhances the understanding of task demands and improves the generation of task-relevant action sequences by leveraging cognitive features. Extensive experiments conducted in both RLBench simulation and real-world environments demonstrate the superiority of MoLe-VLA in both efficiency and performance. Specifically, MoLe-VLA achieves an 8% improvement in the mean success rate across ten tasks while reducing computational costs by up to x5.6 compared to standard LLMs. 

---
# Zero-Shot Human-Object Interaction Synthesis with Multimodal Priors 

**Authors**: Yuke Lou, Yiming Wang, Zhen Wu, Rui Zhao, Wenjia Wang, Mingyi Shi, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2503.20118)  

**Abstract**: Human-object interaction (HOI) synthesis is important for various applications, ranging from virtual reality to robotics. However, acquiring 3D HOI data is challenging due to its complexity and high cost, limiting existing methods to the narrow diversity of object types and interaction patterns in training datasets. This paper proposes a novel zero-shot HOI synthesis framework without relying on end-to-end training on currently limited 3D HOI datasets. The core idea of our method lies in leveraging extensive HOI knowledge from pre-trained Multimodal Models. Given a text description, our system first obtains temporally consistent 2D HOI image sequences using image or video generation models, which are then uplifted to 3D HOI milestones of human and object poses. We employ pre-trained human pose estimation models to extract human poses and introduce a generalizable category-level 6-DoF estimation method to obtain the object poses from 2D HOI images. Our estimation method is adaptive to various object templates obtained from text-to-3D models or online retrieval. A physics-based tracking of the 3D HOI kinematic milestone is further applied to refine both body motions and object poses, yielding more physically plausible HOI generation results. The experimental results demonstrate that our method is capable of generating open-vocabulary HOIs with physical realism and semantic diversity. 

---
# Learning Adaptive Dexterous Grasping from Single Demonstrations 

**Authors**: Liangzhi Shi, Yulin Liu, Lingqi Zeng, Bo Ai, Zhengdong Hong, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.20208)  

**Abstract**: How can robots learn dexterous grasping skills efficiently and apply them adaptively based on user instructions? This work tackles two key challenges: efficient skill acquisition from limited human demonstrations and context-driven skill selection. We introduce AdaDexGrasp, a framework that learns a library of grasping skills from a single human demonstration per skill and selects the most suitable one using a vision-language model (VLM). To improve sample efficiency, we propose a trajectory following reward that guides reinforcement learning (RL) toward states close to a human demonstration while allowing flexibility in exploration. To learn beyond the single demonstration, we employ curriculum learning, progressively increasing object pose variations to enhance robustness. At deployment, a VLM retrieves the appropriate skill based on user instructions, bridging low-level learned skills with high-level intent. We evaluate AdaDexGrasp in both simulation and real-world settings, showing that our approach significantly improves RL efficiency and enables learning human-like grasp strategies across varied object configurations. Finally, we demonstrate zero-shot transfer of our learned policies to a real-world PSYONIC Ability Hand, with a 90% success rate across objects, significantly outperforming the baseline. 

---
# Can Multi-modal (reasoning) LLMs work as deepfake detectors? 

**Authors**: Simiao Ren, Yao Yao, Kidus Zewde, Zisheng Liang, Tsang, Ning-Yau Cheng, Xiaoou Zhan, Qinzhe Liu, Yifei Chen, Hengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20084)  

**Abstract**: Deepfake detection remains a critical challenge in the era of advanced generative models, particularly as synthetic media becomes more sophisticated. In this study, we explore the potential of state of the art multi-modal (reasoning) large language models (LLMs) for deepfake image detection such as (OpenAI O1/4o, Gemini thinking Flash 2, Deepseek Janus, Grok 3, llama 3.2, Qwen 2/2.5 VL, Mistral Pixtral, Claude 3.5/3.7 sonnet) . We benchmark 12 latest multi-modal LLMs against traditional deepfake detection methods across multiple datasets, including recently published real-world deepfake imagery. To enhance performance, we employ prompt tuning and conduct an in-depth analysis of the models' reasoning pathways to identify key contributing factors in their decision-making process. Our findings indicate that best multi-modal LLMs achieve competitive performance with promising generalization ability with zero shot, even surpass traditional deepfake detection pipelines in out-of-distribution datasets while the rest of the LLM families performs extremely disappointing with some worse than random guess. Furthermore, we found newer model version and reasoning capabilities does not contribute to performance in such niche tasks of deepfake detection while model size do help in some cases. This study highlights the potential of integrating multi-modal reasoning in future deepfake detection frameworks and provides insights into model interpretability for robustness in real-world scenarios. 

---
# Test-Time Reasoning Through Visual Human Preferences with VLMs and Soft Rewards 

**Authors**: Alexander Gambashidze, Konstantin Sobolev, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2503.19948)  

**Abstract**: Can Visual Language Models (VLMs) effectively capture human visual preferences? This work addresses this question by training VLMs to think about preferences at test time, employing reinforcement learning methods inspired by DeepSeek R1 and OpenAI O1. Using datasets such as ImageReward and Human Preference Score v2 (HPSv2), our models achieve accuracies of 64.9% on the ImageReward test set (trained on ImageReward official split) and 65.4% on HPSv2 (trained on approximately 25% of its data). These results match traditional encoder-based models while providing transparent reasoning and enhanced generalization. This approach allows to use not only rich VLM world knowledge, but also its potential to think, yielding interpretable outcomes that help decision-making processes. By demonstrating that human visual preferences reasonable by current VLMs, we introduce efficient soft-reward strategies for image ranking, outperforming simplistic selection or scoring methods. This reasoning capability enables VLMs to rank arbitrary images-regardless of aspect ratio or complexity-thereby potentially amplifying the effectiveness of visual Preference Optimization. By reducing the need for extensive markup while improving reward generalization and explainability, our findings can be a strong mile-stone that will enhance text-to-vision models even further. 

---
