# Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning 

**Authors**: Léopold Maytié, Roland Bertin Johannet, Rufin VanRullen  

**Link**: [PDF](https://arxiv.org/pdf/2502.21142)  

**Abstract**: Humans leverage rich internal models of the world to reason about the future, imagine counterfactuals, and adapt flexibly to new situations. In Reinforcement Learning (RL), world models aim to capture how the environment evolves in response to the agent's actions, facilitating planning and generalization. However, typical world models directly operate on the environment variables (e.g. pixels, physical attributes), which can make their training slow and cumbersome; instead, it may be advantageous to rely on high-level latent dimensions that capture relevant multimodal variables. Global Workspace (GW) Theory offers a cognitive framework for multimodal integration and information broadcasting in the brain, and recent studies have begun to introduce efficient deep learning implementations of GW. Here, we evaluate the capabilities of an RL system combining GW with a world model. We compare our GW-Dreamer with various versions of the standard PPO and the original Dreamer algorithms. We show that performing the dreaming process (i.e., mental simulation) inside the GW latent space allows for training with fewer environment steps. As an additional emergent property, the resulting model (but not its comparison baselines) displays strong robustness to the absence of one of its observation modalities (images or simulation attributes). We conclude that the combination of GW with World Models holds great potential for improving decision-making in RL agents. 

---
# MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts 

**Authors**: Peijie Wang, Zhongzhi Li, Fei Yin, Dekang Ran, Chenglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20808)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown promising capabilities in mathematical reasoning within visual contexts across various datasets. However, most existing multimodal math benchmarks are limited to single-visual contexts, which diverges from the multi-visual scenarios commonly encountered in real-world mathematical applications. To address this gap, we introduce MV-MATH: a meticulously curated dataset of 2,009 high-quality mathematical problems. Each problem integrates multiple images interleaved with text, derived from authentic K-12 scenarios, and enriched with detailed annotations. MV-MATH includes multiple-choice, free-form, and multi-step questions, covering 11 subject areas across 3 difficulty levels, and serves as a comprehensive and rigorous benchmark for assessing MLLMs' mathematical reasoning in multi-visual contexts. Through extensive experimentation, we observe that MLLMs encounter substantial challenges in multi-visual math tasks, with a considerable performance gap relative to human capabilities on MV-MATH. Furthermore, we analyze the performance and error patterns of various models, providing insights into MLLMs' mathematical reasoning capabilities within multi-visual settings. 

---
# FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts 

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2502.21059)  

**Abstract**: Large Vision-Language Models (LVLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most LVLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, LVLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute a jailbreak attack on LVLMs. Our evaluations using the Advbench dataset show that FC-Attack achieves over 90% attack success rates on Gemini-1.5, Llaval-Next, Qwen2-VL, and InternVL-2.5 models, outperforming existing LVLM jailbreak methods. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. Our evaluation shows that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop. 

---
# UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation 

**Authors**: Thanet Markchom, Tong Wu, Liting Huang, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20984)  

**Abstract**: SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at this https URL. 

---
# DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping 

**Authors**: Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Yitao Liang, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20900)  

**Abstract**: Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on specific assumptions, such as single-object settings or limited environments, leading to constrained generalization. Our solution is DexGraspVLA, a hierarchical framework that utilizes a pre-trained Vision-Language model as the high-level task planner and learns a diffusion-based policy as the low-level Action controller. The key insight lies in iteratively transforming diverse language and visual inputs into domain-invariant representations, where imitation learning can be effectively applied due to the alleviation of domain shift. Thus, it enables robust generalization across a wide range of real-world scenarios. Notably, our method achieves a 90+% success rate under thousands of unseen object, lighting, and background combinations in a ``zero-shot'' environment. Empirical analysis further confirms the consistency of internal model behavior across environmental variations, thereby validating our design and explaining its generalization performance. We hope our work can be a step forward in achieving general dexterous grasping. Our demo and code can be found at this https URL. 

---
# Multimodal Learning for Just-In-Time Software Defect Prediction in Autonomous Driving Systems 

**Authors**: Faisal Mohammad, Duksan Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20806)  

**Abstract**: In recent years, the rise of autonomous driving technologies has highlighted the critical importance of reliable software for ensuring safety and performance. This paper proposes a novel approach for just-in-time software defect prediction (JIT-SDP) in autonomous driving software systems using multimodal learning. The proposed model leverages the multimodal transformers in which the pre-trained transformers and a combining module deal with the multiple data modalities of the software system datasets such as code features, change metrics, and contextual information. The key point for adapting multimodal learning is to utilize the attention mechanism between the different data modalities such as text, numerical, and categorical. In the combining module, the output of a transformer model on text data and tabular features containing categorical and numerical data are combined to produce the predictions using the fully connected layers. Experiments conducted on three open-source autonomous driving system software projects collected from the GitHub repository (Apollo, Carla, and Donkeycar) demonstrate that the proposed approach significantly outperforms state-of-the-art deep learning and machine learning models regarding evaluation metrics. Our findings highlight the potential of multimodal learning to enhance the reliability and safety of autonomous driving software through improved defect prediction. 

---
# Interpreting CLIP with Hierarchical Sparse Autoencoders 

**Authors**: Vladimir Zaigrajew, Hubert Baniecki, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2502.20578)  

**Abstract**: Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing large-scale vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce Matryoshka SAE (MSAE), a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a new state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving 0.99 cosine similarity and less than 0.1 fraction of variance unexplained while maintaining ~80% sparsity. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA. 

---
# A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs 

**Authors**: Julius Broomfield, Kartik Sharma, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.20504)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable advancements in embodying diverse personas, enhancing their effectiveness as conversational agents and virtual assistants. Consequently, LLMs have made significant strides in processing and integrating multimodal information. However, even though human personas can be expressed in both text and image, the extent to which the modality of a persona impacts the embodiment by the LLM remains largely unexplored. In this paper, we investigate how do different modalities influence the expressiveness of personas in multimodal LLMs. To this end, we create a novel modality-parallel dataset of 40 diverse personas varying in age, gender, occupation, and location. This consists of four modalities to equivalently represent a persona: image-only, text-only, a combination of image and small text, and typographical images, where text is visually stylized to convey persona-related attributes. We then create a systematic evaluation framework with 60 questions and corresponding metrics to assess how well LLMs embody each persona across its attributes and scenarios. Comprehensive experiments on $5$ multimodal LLMs show that personas represented by detailed text show more linguistic habits, while typographical images often show more consistency with the persona. Our results reveal that LLMs often overlook persona-specific details conveyed through images, highlighting underlying limitations and paving the way for future research to bridge this gap. We release the data and code at this https URL . 

---
# Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow 

**Authors**: Jiaqi Bai, Hongcheng Guo, Zhongyuan Peng, Jian Yang, Zhoujun Li, Mohan Li, Zhihong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2502.20750)  

**Abstract**: Large vision-language models show tremendous potential in understanding visual information through human languages. However, they are prone to suffer from object hallucination, i.e., the generated image descriptions contain objects that do not exist in the image. In this paper, we reveal that object hallucination can be attributed to overconfidence in irrelevant visual features when soft visual tokens map to the LLM's word embedding space. Specifically, by figuring out the semantic similarity between visual tokens and LLM's word embedding, we observe that the smoothness of similarity distribution strongly correlates with the emergence of object hallucinations. To mitigate hallucinations, we propose using the Variational Information Bottleneck (VIB) to alleviate overconfidence by introducing stochastic noise, facilitating the constraining of irrelevant information. Furthermore, we propose an entropy-based noise-controlling strategy to enable the injected noise to be adaptively constrained regarding the smoothness of the similarity distribution. We adapt the proposed AdaVIB across distinct model architectures. Experimental results demonstrate that the proposed AdaVIB mitigates object hallucinations by effectively alleviating the overconfidence in irrelevant visual features, with consistent improvements on two object hallucination benchmarks. 

---
# Chitranuvad: Adapting Multi-Lingual LLMs for Multimodal Translation 

**Authors**: Shaharukh Khan, Ayush Tarun, Ali Faraz, Palash Kamble, Vivek Dahiya, Praveen Pokala, Ashish Kulkarni, Chandra Khatri, Abhinav Ravi, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.20420)  

**Abstract**: In this work, we provide the system description of our submission as part of the English to Lowres Multimodal Translation Task at the Workshop on Asian Translation (WAT2024). We introduce Chitranuvad, a multimodal model that effectively integrates Multilingual LLM and a vision module for Multimodal Translation. Our method uses a ViT image encoder to extract visual representations as visual token embeddings which are projected to the LLM space by an adapter layer and generates translation in an autoregressive fashion. We participated in all the three tracks (Image Captioning, Text only and Multimodal translation tasks) for Indic languages (ie. English translation to Hindi, Bengali and Malyalam) and achieved SOTA results for Hindi in all of them on the Challenge set while remaining competitive for the other languages in the shared task. 

---
# HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models 

**Authors**: Xiao Wang, Jingyun Hua, Weihong Lin, Yuanxing Zhang, Fuzheng Zhang, Jianlong Wu, Di Zhang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.20811)  

**Abstract**: Recent Multi-modal Large Language Models (MLLMs) have made great progress in video understanding. However, their performance on videos involving human actions is still limited by the lack of high-quality data. To address this, we introduce a two-stage data annotation pipeline. First, we design strategies to accumulate videos featuring clear human actions from the Internet. Second, videos are annotated in a standardized caption format that uses human attributes to distinguish individuals and chronologically details their actions and interactions. Through this pipeline, we curate two datasets, namely HAICTrain and HAICBench. \textbf{HAICTrain} comprises 126K video-caption pairs generated by Gemini-Pro and verified for training purposes. Meanwhile, \textbf{HAICBench} includes 500 manually annotated video-caption pairs and 1,400 QA pairs, for a comprehensive evaluation of human action understanding. Experimental results demonstrate that training with HAICTrain not only significantly enhances human understanding abilities across 4 benchmarks, but can also improve text-to-video generation results. Both the HAICTrain and HAICBench are released at this https URL. 

---
# Visual Reasoning at Urban Intersections: FineTuning GPT-4o for Traffic Conflict Detection 

**Authors**: Sari Masri, Huthaifa I. Ashqar, Mohammed Elhenawy  

**Link**: [PDF](https://arxiv.org/pdf/2502.20573)  

**Abstract**: Traffic control in unsignalized urban intersections presents significant challenges due to the complexity, frequent conflicts, and blind spots. This study explores the capability of leveraging Multimodal Large Language Models (MLLMs), such as GPT-4o, to provide logical and visual reasoning by directly using birds-eye-view videos of four-legged intersections. In this proposed method, GPT-4o acts as intelligent system to detect conflicts and provide explanations and recommendations for the drivers. The fine-tuned model achieved an accuracy of 77.14%, while the manual evaluation of the true predicted values of the fine-tuned GPT-4o showed significant achievements of 89.9% accuracy for model-generated explanations and 92.3% for the recommended next actions. These results highlight the feasibility of using MLLMs for real-time traffic management using videos as inputs, offering scalable and actionable insights into intersections traffic management and operation. Code used in this study is available at this https URL. 

---
# Protecting multimodal large language models against misleading visualizations 

**Authors**: Jonathan Tonglet, Tinne Tuytelaars, Marie-Francine Moens, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2502.20503)  

**Abstract**: We assess the vulnerability of multimodal large language models to misleading visualizations - charts that distort the underlying data using techniques such as truncated or inverted axes, leading readers to draw inaccurate conclusions that may support misinformation or conspiracy theories. Our analysis shows that these distortions severely harm multimodal large language models, reducing their question-answering accuracy to the level of the random baseline. To mitigate this vulnerability, we introduce six inference-time methods to improve performance of MLLMs on misleading visualizations while preserving their accuracy on non-misleading ones. The most effective approach involves (1) extracting the underlying data table and (2) using a text-only large language model to answer questions based on the table. This method improves performance on misleading visualizations by 15.4 to 19.6 percentage points. 

---
# Towards Statistical Factuality Guarantee for Large Vision-Language Models 

**Authors**: Zhuohang Li, Chao Yan, Nicholas J. Jackson, Wendi Cui, Bo Li, Jiaxin Zhang, Bradley A. Malin  

**Link**: [PDF](https://arxiv.org/pdf/2502.20560)  

**Abstract**: Advancements in Large Vision-Language Models (LVLMs) have demonstrated promising performance in a variety of vision-language tasks involving image-conditioned free-form text generation. However, growing concerns about hallucinations in LVLMs, where the generated text is inconsistent with the visual context, are becoming a major impediment to deploying these models in applications that demand guaranteed reliability. In this paper, we introduce a framework to address this challenge, ConfLVLM, which is grounded on conformal prediction to achieve finite-sample distribution-free statistical guarantees on the factuality of LVLM output. This framework treats an LVLM as a hypothesis generator, where each generated text detail (or claim) is considered an individual hypothesis. It then applies a statistical hypothesis testing procedure to verify each claim using efficient heuristic uncertainty measures to filter out unreliable claims before returning any responses to users. We conduct extensive experiments covering three representative application domains, including general scene understanding, medical radiology report generation, and document understanding. Remarkably, ConfLVLM reduces the error rate of claims generated by LLaVa-1.5 for scene descriptions from 87.8\% to 10.0\% by filtering out erroneous claims with a 95.3\% true positive rate. Our results further demonstrate that ConfLVLM is highly flexible, and can be applied to any black-box LVLMs paired with any uncertainty measure for any image-conditioned free-form text generation task while providing a rigorous guarantee on controlling the risk of hallucination. 

---
# Joint Modeling in Recommendations: A Survey 

**Authors**: Xiangyu Zhao, Yichao Wang, Bo Chen, Jingtong Gao, Yuhao Wang, Xiaopeng Li, Pengyue Jia, Qidong Liu, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21195)  

**Abstract**: In today's digital landscape, Deep Recommender Systems (DRS) play a crucial role in navigating and customizing online content for individual preferences. However, conventional methods, which mainly depend on single recommendation task, scenario, data modality and user behavior, are increasingly seen as insufficient due to their inability to accurately reflect users' complex and changing preferences. This gap underscores the need for joint modeling approaches, which are central to overcoming these limitations by integrating diverse tasks, scenarios, modalities, and behaviors in the recommendation process, thus promising significant enhancements in recommendation precision, efficiency, and customization. In this paper, we comprehensively survey the joint modeling methods in recommendations. We begin by defining the scope of joint modeling through four distinct dimensions: multi-task, multi-scenario, multi-modal, and multi-behavior modeling. Subsequently, we examine these methods in depth, identifying and summarizing their underlying paradigms based on the latest advancements and potential research trajectories. Ultimately, we highlight several promising avenues for future exploration in joint modeling for recommendations and provide a concise conclusion to our findings. 

---
