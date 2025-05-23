# Graph-Enhanced Model-Free Reinforcement Learning Agents for Efficient Power Grid Topological Control 

**Authors**: Eloy Anguiano Batanero, Ángela Fernández, Álvaro Barbero  

**Link**: [PDF](https://arxiv.org/pdf/2503.20688)  

**Abstract**: The increasing complexity of power grid management, driven by the emergence of prosumers and the demand for cleaner energy solutions, has needed innovative approaches to ensure stability and efficiency. This paper presents a novel approach within the model-free framework of reinforcement learning, aimed at optimizing power network operations without prior expert knowledge. We introduce a masked topological action space, enabling agents to explore diverse strategies for cost reduction while maintaining reliable service using the state logic as a guide for choosing proper actions. Through extensive experimentation across 20 different scenarios in a simulated 5-substation environment, we demonstrate that our approach achieves a consistent reduction in power losses, while ensuring grid stability against potential blackouts. The results underscore the effectiveness of combining dynamic observation formalization with opponent-based training, showing a viable way for autonomous management solutions in modern energy systems or even for building a foundational model for this field. 

---
# Inductive Link Prediction on N-ary Relational Facts via Semantic Hypergraph Reasoning 

**Authors**: Gongzhu Yin, Hongli Zhang, Yuchen Yang, Yi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.20676)  

**Abstract**: N-ary relational facts represent semantic correlations among more than two entities. While recent studies have developed link prediction (LP) methods to infer missing relations for knowledge graphs (KGs) containing n-ary relational facts, they are generally limited to transductive settings. Fully inductive settings, where predictions are made on previously unseen entities, remain a significant challenge. As existing methods are mainly entity embedding-based, they struggle to capture entity-independent logical rules. To fill in this gap, we propose an n-ary subgraph reasoning framework for fully inductive link prediction (ILP) on n-ary relational facts. This framework reasons over local subgraphs and has a strong inductive inference ability to capture n-ary patterns. Specifically, we introduce a novel graph structure, the n-ary semantic hypergraph, to facilitate subgraph extraction. Moreover, we develop a subgraph aggregating network, NS-HART, to effectively mine complex semantic correlations within subgraphs. Theoretically, we provide a thorough analysis from the score function optimization perspective to shed light on NS-HART's effectiveness for n-ary ILP tasks. Empirically, we conduct extensive experiments on a series of inductive benchmarks, including transfer reasoning (with and without entity features) and pairwise subgraph reasoning. The results highlight the superiority of the n-ary subgraph reasoning framework and the exceptional inductive ability of NS-HART. The source code of this paper has been made publicly available at this https URL. 

---
# Procedural Knowledge Ontology (PKO) 

**Authors**: Valentina Anita Carriero, Mario Scrocca, Ilaria Baroni, Antonia Azzini, Irene Celino  

**Link**: [PDF](https://arxiv.org/pdf/2503.20634)  

**Abstract**: Processes, workflows and guidelines are core to ensure the correct functioning of industrial companies: for the successful operations of factory lines, machinery or services, often industry operators rely on their past experience and know-how. The effect is that this Procedural Knowledge (PK) remains tacit and, as such, difficult to exploit efficiently and effectively. This paper presents PKO, the Procedural Knowledge Ontology, which enables the explicit modeling of procedures and their executions, by reusing and extending existing ontologies. PKO is built on requirements collected from three heterogeneous industrial use cases and can be exploited by any AI and data-driven tools that rely on a shared and interoperable representation to support the governance of PK throughout its life cycle. We describe its structure and design methodology, and outline its relevance, quality, and impact by discussing applications leveraging PKO for PK elicitation and exploitation. 

---
# Perspective-Shifted Neuro-Symbolic World Models: A Framework for Socially-Aware Robot Navigation 

**Authors**: Kevin Alcedo, Pedro U. Lima, Rachid Alami  

**Link**: [PDF](https://arxiv.org/pdf/2503.20425)  

**Abstract**: Navigating in environments alongside humans requires agents to reason under uncertainty and account for the beliefs and intentions of those around them. Under a sequential decision-making framework, egocentric navigation can naturally be represented as a Markov Decision Process (MDP). However, social navigation additionally requires reasoning about the hidden beliefs of others, inherently leading to a Partially Observable Markov Decision Process (POMDP), where agents lack direct access to others' mental states. Inspired by Theory of Mind and Epistemic Planning, we propose (1) a neuro-symbolic model-based reinforcement learning architecture for social navigation, addressing the challenge of belief tracking in partially observable environments; and (2) a perspective-shift operator for belief estimation, leveraging recent work on Influence-based Abstractions (IBA) in structured multi-agent settings. 

---
# Synthesizing world models for bilevel planning 

**Authors**: Zergham Ahmed, Joshua B. Tenenbaum, Christopher J. Bates, Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2503.20124)  

**Abstract**: Modern reinforcement learning (RL) systems have demonstrated remarkable capabilities in complex environments, such as video games. However, they still fall short of achieving human-like sample efficiency and adaptability when learning new domains. Theory-based reinforcement learning (TBRL) is an algorithmic framework specifically designed to address this gap. Modeled on cognitive theories, TBRL leverages structured, causal world models - "theories" - as forward simulators for use in planning, generalization and exploration. Although current TBRL systems provide compelling explanations of how humans learn to play video games, they face several technical limitations: their theory languages are restrictive, and their planning algorithms are not scalable. To address these challenges, we introduce TheoryCoder, an instantiation of TBRL that exploits hierarchical representations of theories and efficient program synthesis methods for more powerful learning and planning. TheoryCoder equips agents with general-purpose abstractions (e.g., "move to"), which are then grounded in a particular environment by learning a low-level transition model (a Python program synthesized from observations by a large language model). A bilevel planning algorithm can exploit this hierarchical structure to solve large domains. We demonstrate that this approach can be successfully applied to diverse and challenging grid-world games, where approaches based on directly synthesizing a policy perform poorly. Ablation studies demonstrate the benefits of using hierarchical abstractions. 

---
# Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Models Using Implicit Feedback from Pre-training Demonstrations 

**Authors**: Ran Tian, Kratarth Goel  

**Link**: [PDF](https://arxiv.org/pdf/2503.20105)  

**Abstract**: Recent advancements in LLMs have revolutionized motion generation models in embodied applications. While LLM-type auto-regressive motion generation models benefit from training scalability, there remains a discrepancy between their token prediction objectives and human preferences. As a result, models pre-trained solely with token-prediction objectives often generate behaviors that deviate from what humans would prefer, making post-training preference alignment crucial for producing human-preferred motions. Unfortunately, post-training alignment requires extensive preference rankings of motions generated by the pre-trained model, which are costly to annotate, especially in multi-agent settings. Recently, there has been growing interest in leveraging pre-training demonstrations to scalably generate preference data for post-training alignment. However, these methods often adopt an adversarial assumption, treating all pre-trained model-generated samples as unpreferred examples. This adversarial approach overlooks the valuable signal provided by preference rankings among the model's own generations, ultimately reducing alignment effectiveness and potentially leading to misaligned behaviors. In this work, instead of treating all generated samples as equally bad, we leverage implicit preferences encoded in pre-training demonstrations to construct preference rankings among the pre-trained model's generations, offering more nuanced preference alignment guidance with zero human cost. We apply our approach to large-scale traffic simulation and demonstrate its effectiveness in improving the realism of pre-trained model's generated behaviors, making a lightweight 1M motion generation model comparable to SOTA large imitation-based models by relying solely on implicit feedback from pre-training demonstrations, without additional post-training human preference annotations or high computational costs. 

---
# OmniNova:A General Multimodal Agent Framework 

**Authors**: Pengfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.20028)  

**Abstract**: The integration of Large Language Models (LLMs) with specialized tools presents new opportunities for intelligent automation systems. However, orchestrating multiple LLM-driven agents to tackle complex tasks remains challenging due to coordination difficulties, inefficient resource utilization, and inconsistent information flow. We present OmniNova, a modular multi-agent automation framework that combines language models with specialized tools such as web search, crawling, and code execution capabilities. OmniNova introduces three key innovations: (1) a hierarchical multi-agent architecture with distinct coordinator, planner, supervisor, and specialist agents; (2) a dynamic task routing mechanism that optimizes agent deployment based on task complexity; and (3) a multi-layered LLM integration system that allocates appropriate models to different cognitive requirements. Our evaluations across 50 complex tasks in research, data analysis, and web interaction domains demonstrate that OmniNova outperforms existing frameworks in task completion rate (87\% vs. baseline 62\%), efficiency (41\% reduced token usage), and result quality (human evaluation score of 4.2/5 vs. baseline 3.1/5). We contribute both a theoretical framework for multi-agent system design and an open-source implementation that advances the state-of-the-art in LLM-based automation systems. 

---
# Unsupervised Learning for Quadratic Assignment 

**Authors**: Yimeng Min, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2503.20001)  

**Abstract**: We introduce PLUME search, a data-driven framework that enhances search efficiency in combinatorial optimization through unsupervised learning. Unlike supervised or reinforcement learning, PLUME search learns directly from problem instances using a permutation-based loss with a non-autoregressive approach. We evaluate its performance on the quadratic assignment problem, a fundamental NP-hard problem that encompasses various combinatorial optimization problems. Experimental results demonstrate that PLUME search consistently improves solution quality. Furthermore, we study the generalization behavior and show that the learned model generalizes across different densities and sizes. 

---
# LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning? 

**Authors**: Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kaifeng Lyu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19990)  

**Abstract**: Multi-step spatial reasoning entails understanding and reasoning about spatial relationships across multiple sequential steps, which is crucial for tackling complex real-world applications, such as robotic manipulation, autonomous navigation, and automated assembly. To assess how well current Multimodal Large Language Models (MLLMs) have acquired this fundamental capability, we introduce \textbf{LEGO-Puzzles}, a scalable benchmark designed to evaluate both \textbf{spatial understanding} and \textbf{sequential reasoning} in MLLMs through LEGO-based tasks. LEGO-Puzzles consists of 1,100 carefully curated visual question-answering (VQA) samples spanning 11 distinct tasks, ranging from basic spatial understanding to complex multi-step reasoning. Based on LEGO-Puzzles, we conduct a comprehensive evaluation of state-of-the-art MLLMs and uncover significant limitations in their spatial reasoning capabilities: even the most powerful MLLMs can answer only about half of the test cases, whereas human participants achieve over 90\% accuracy. In addition to VQA tasks, we evaluate MLLMs' abilities to generate LEGO images following assembly illustrations. Our experiments show that only Gemini-2.0-Flash and GPT-4o exhibit a limited ability to follow these instructions, while other MLLMs either replicate the input image or generate completely irrelevant outputs. Overall, LEGO-Puzzles exposes critical deficiencies in existing MLLMs' spatial understanding and sequential reasoning capabilities, and underscores the need for further advancements in multimodal spatial reasoning. 

---
# Mobile-MMLU: A Mobile Intelligence Language Understanding Benchmark 

**Authors**: Sondos Mahmoud Bsharat, Mukul Ranjan, Aidar Myrzakhan, Jiacheng Liu, Bowei Guo, Shengkun Tang, Zhuang Liu, Yuanzhi Li, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20786)  

**Abstract**: Rapid advancements in large language models (LLMs) have increased interest in deploying them on mobile devices for on-device AI applications. Mobile users interact differently with LLMs compared to desktop users, creating unique expectations and data biases. Current benchmark datasets primarily target at server and desktop environments, and there is a notable lack of extensive datasets specifically designed for mobile contexts. Additionally, mobile devices face strict limitations in storage and computing resources, constraining model size and capabilities, thus requiring optimized efficiency and prioritized knowledge. To address these challenges, we introduce Mobile-MMLU, a large-scale benchmark dataset tailored for mobile intelligence. It consists of 16,186 questions across 80 mobile-related fields, designed to evaluate LLM performance in realistic mobile scenarios. A challenging subset, Mobile-MMLU-Pro, provides advanced evaluation similar in size to MMLU-Pro but significantly more difficult than our standard full set. Both benchmarks use multiple-choice, order-invariant questions focused on practical mobile interactions, such as recipe suggestions, travel planning, and essential daily tasks. The dataset emphasizes critical mobile-specific metrics like inference latency, energy consumption, memory usage, and response quality, offering comprehensive insights into model performance under mobile constraints. Moreover, it prioritizes privacy and adaptability, assessing models' ability to perform on-device processing, maintain user privacy, and adapt to personalized usage patterns. Mobile-MMLU family offers a standardized framework for developing and comparing mobile-optimized LLMs, enabling advancements in productivity and decision-making within mobile computing environments. Our code and data are available at: this https URL. 

---
# Understanding R1-Zero-Like Training: A Critical Perspective 

**Authors**: Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.20783)  

**Abstract**: DeepSeek-R1-Zero has shown that reinforcement learning (RL) at scale can directly enhance the reasoning capabilities of LLMs without supervised fine-tuning. In this work, we critically examine R1-Zero-like training by analyzing its two core components: base models and RL. We investigate a wide range of base models, including DeepSeek-V3-Base, to understand how pretraining characteristics influence RL performance. Our analysis reveals that DeepSeek-V3-Base already exhibit ''Aha moment'', while Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases. Additionally, we identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length (especially for incorrect outputs) during training. To address this, we introduce Dr. GRPO, an unbiased optimization method that improves token efficiency while maintaining reasoning performance. Leveraging these insights, we present a minimalist R1-Zero recipe that achieves 43.3% accuracy on AIME 2024 with a 7B base model, establishing a new state-of-the-art. Our code is available at this https URL. 

---
# ADS-Edit: A Multimodal Knowledge Editing Dataset for Autonomous Driving Systems 

**Authors**: Chenxi Wang, Jizhan Fang, Xiang Chen, Bozhong Tian, Ziwen Xu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20756)  

**Abstract**: Recent advancements in Large Multimodal Models (LMMs) have shown promise in Autonomous Driving Systems (ADS). However, their direct application to ADS is hindered by challenges such as misunderstanding of traffic knowledge, complex road conditions, and diverse states of vehicle. To address these challenges, we propose the use of Knowledge Editing, which enables targeted modifications to a model's behavior without the need for full retraining. Meanwhile, we introduce ADS-Edit, a multimodal knowledge editing dataset specifically designed for ADS, which includes various real-world scenarios, multiple data types, and comprehensive evaluation metrics. We conduct comprehensive experiments and derive several interesting conclusions. We hope that our work will contribute to the further advancement of knowledge editing applications in the field of autonomous driving. Code and data are available in this https URL. 

---
# Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning 

**Authors**: Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20752)  

**Abstract**: Visual reasoning abilities play a crucial role in understanding complex multimodal data, advancing both domain-specific applications and artificial general intelligence (AGI). Existing methods improve VLM reasoning via Chain-of-Thought (CoT) supervised fine-tuning, using meticulously annotated training data to enhance visual reasoning capabilities. However, this training paradigm may lead to overfitting and cognitive rigidity, restricting the model's ability to transfer visual reasoning skills across domains and limiting its real-world applicability. To address these limitations, we propose Reason-RFT, a novel reinforcement fine-tuning framework that significantly enhances generalization capabilities in visual reasoning tasks. Reason-RFT introduces a two-phase training framework for visual reasoning: (1) Supervised Fine-Tuning (SFT) with curated Chain-of-Thought (CoT) data activates the reasoning potential of Vision-Language Models (VLMs), followed by (2) Group Relative Policy Optimization (GRPO)-based reinforcement learning that generates multiple reasoning-response pairs, significantly enhancing generalization in visual reasoning tasks. To evaluate Reason-RFT's visual reasoning capabilities, we reconstructed a comprehensive dataset spanning visual counting, structure perception, and spatial this http URL results demonstrate Reasoning-RFT's three key advantages: (1) Performance Enhancement: achieving state-of-the-art results across multiple tasks, outperforming most mainstream open-source and proprietary models; (2) Generalization Superiority: consistently maintaining robust performance across diverse tasks and domains, outperforming alternative training paradigms; (3) Data Efficiency: excelling in few-shot learning scenarios while surpassing full-dataset SFT baselines. 

---
# Optimal Scaling Laws for Efficiency Gains in a Theoretical Transformer-Augmented Sectional MoE Framework 

**Authors**: Soham Sane  

**Link**: [PDF](https://arxiv.org/pdf/2503.20750)  

**Abstract**: This paper introduces a theoretical framework for a Transformer-augmented, sectional Mixture-of-Experts (MoE) architecture that aims to enhance computational efficiency while preserving model scalability. Unlike conventional MoE models, which route entire token embeddings to selected experts, our approach portions the embedding dimension itself -- assigning segments of each token's representation to dedicated experts. To combat losses in token representation, we utilize a pre-expert transformer layer to recompute attention across tokens and reduce the sequence length dimensionality. We extend our theory by deriving optimal scaling laws that a non-linear relationship between the number of experts and factors such as model dimensionality, sequence length, and system overhead. These formulations yield closed-form and numerically-solvable expressions for identifying the optimal expert count under given architectural and hardware constraints. As a result, our framework not only provides theoretical bounds for computing efficiency with varying frameworks but also guides practical design choices for scaling large models effectively. While empirical validation is pending, we present a comprehensive experimental road map to evaluate the framework's efficiency, scalability, and practicality in future work. 

---
# High Quality Diffusion Distillation on a Single GPU with Relative and Absolute Position Matching 

**Authors**: Guoqiang Zhang, Kenta Niwa, J.P. Lewis, Cedric Mesnage, W. Bastiaan Kleijn  

**Link**: [PDF](https://arxiv.org/pdf/2503.20744)  

**Abstract**: We introduce relative and absolute position matching (RAPM), a diffusion distillation method resulting in high quality generation that can be trained efficiently on a single GPU. Recent diffusion distillation research has achieved excellent results for high-resolution text-to-image generation with methods such as phased consistency models (PCM) and improved distribution matching distillation (DMD2). However, these methods generally require many GPUs (e.g.~8-64) and significant batchsizes (e.g.~128-2048) during training, resulting in memory and compute requirements that are beyond the resources of some researchers. RAPM provides effective single-GPU diffusion distillation training with a batchsize of 1. The new method attempts to mimic the sampling trajectories of the teacher model by matching the relative and absolute positions. The design of relative positions is inspired by PCM. Two discriminators are introduced accordingly in RAPM, one for matching relative positions and the other for absolute positions. Experimental results on StableDiffusion (SD) V1.5 and SDXL indicate that RAPM with 4 timesteps produces comparable FID scores as the best method with 1 timestep under very limited computational resources. 

---
# Quantum Neural Network Restatement of Markov Jump Process 

**Authors**: Z.Zarezadeh, N.Zarezadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.20742)  

**Abstract**: Despite the many challenges in exploratory data analysis, artificial neural networks have motivated strong interests in scientists and researchers both in theoretical as well as practical applications. Among sources of such popularity of artificial neural networks the ability of modeling non-linear dynamical systems, generalization, and adaptation possibilities should be mentioned. Despite this, there is still significant debate about the role of various underlying stochastic processes in stabilizing a unique structure for data learning and prediction. One of such obstacles to the theoretical and numerical study of machine intelligent systems is the curse of dimensionality and the sampling from high-dimensional probability distributions. In general, this curse prevents efficient description of states, providing a significant complexity barrier for the system to be efficiently described and studied. In this strand of research, direct treatment and description of such abstract notions of learning theory in terms of quantum information be one of the most favorable candidates. Hence, the subject matter of these articles is devoted to problems of design, adaptation and the formulations of computationally hard problems in terms of quantum mechanical systems. In order to characterize the microscopic description of such dynamics in the language of inferential statistics, covariance matrix estimation of d-dimensional Gaussian densities and Bayesian interpretation of eigenvalue problem for dynamical systems is assessed. 

---
# Emotion Detection and Music Recommendation System 

**Authors**: Swetha Kambham, Hubert Jhonson, Sai Prathap Reddy Kambham  

**Link**: [PDF](https://arxiv.org/pdf/2503.20739)  

**Abstract**: As artificial intelligence becomes more and more ingrained in daily life, we present a novel system that uses deep learning for music recommendation and emotion-based detection. Through the use of facial recognition and the DeepFace framework, our method analyses human emotions in real-time and then plays music that reflects the mood it has discovered. The system uses a webcam to take pictures, analyses the most common facial expression, and then pulls a playlist from local storage that corresponds to the mood it has detected. An engaging and customised experience is ensured by allowing users to manually change the song selection via a dropdown menu or navigation buttons. By continuously looping over the playlist, the technology guarantees continuity. The objective of our system is to improve emotional well-being through music therapy by offering a responsive and automated music-selection experience. 

---
# Flip Learning: Weakly Supervised Erase to Segment Nodules in Breast Ultrasound 

**Authors**: Yuhao Huang, Ao Chang, Haoran Dou, Xing Tao, Xinrui Zhou, Yan Cao, Ruobing Huang, Alejandro F Frangi, Lingyun Bao, Xin Yang, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2503.20685)  

**Abstract**: Accurate segmentation of nodules in both 2D breast ultrasound (BUS) and 3D automated breast ultrasound (ABUS) is crucial for clinical diagnosis and treatment planning. Therefore, developing an automated system for nodule segmentation can enhance user independence and expedite clinical analysis. Unlike fully-supervised learning, weakly-supervised segmentation (WSS) can streamline the laborious and intricate annotation process. However, current WSS methods face challenges in achieving precise nodule segmentation, as many of them depend on inaccurate activation maps or inefficient pseudo-mask generation algorithms. In this study, we introduce a novel multi-agent reinforcement learning-based WSS framework called Flip Learning, which relies solely on 2D/3D boxes for accurate segmentation. Specifically, multiple agents are employed to erase the target from the box to facilitate classification tag flipping, with the erased region serving as the predicted segmentation mask. The key contributions of this research are as follows: (1) Adoption of a superpixel/supervoxel-based approach to encode the standardized environment, capturing boundary priors and expediting the learning process. (2) Introduction of three meticulously designed rewards, comprising a classification score reward and two intensity distribution rewards, to steer the agents' erasing process precisely, thereby avoiding both under- and over-segmentation. (3) Implementation of a progressive curriculum learning strategy to enable agents to interact with the environment in a progressively challenging manner, thereby enhancing learning efficiency. Extensively validated on the large in-house BUS and ABUS datasets, our Flip Learning method outperforms state-of-the-art WSS methods and foundation models, and achieves comparable performance as fully-supervised learning algorithms. 

---
# Probabilistic Forecasting for Network Resource Analysis in Integrated Terrestrial and Non-Terrestrial Networks 

**Authors**: Cristian J. Vaca-Rubio, Vaishnavi Kasuluru, Engin Zeydan, Luis Blanco, Roberto Pereira, Marius Caus, Kapal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2503.20658)  

**Abstract**: Efficient resource management is critical for Non-Terrestrial Networks (NTNs) to provide consistent, high-quality service in remote and under-served regions. While traditional single-point prediction methods, such as Long-Short Term Memory (LSTM), have been used in terrestrial networks, they often fall short in NTNs due to the complexity of satellite dynamics, signal latency and coverage variability. Probabilistic forecasting, which quantifies the uncertainties of the predictions, is a robust alternative. In this paper, we evaluate the application of probabilistic forecasting techniques, in particular SFF, to NTN resource allocation scenarios. Our results show their effectiveness in predicting bandwidth and capacity requirements in different NTN segments of probabilistic forecasting compared to single-point prediction techniques such as LSTM. The results show the potential of black probabilistic forecasting models to provide accurate and reliable predictions and to quantify their uncertainty, making them indispensable for optimizing NTN resource allocation. At the end of the paper, we also present application scenarios and a standardization roadmap for the use of probabilistic forecasting in integrated Terrestrial Network (TN)-NTN environments. 

---
# AccidentSim: Generating Physically Realistic Vehicle Collision Videos from Real-World Accident Reports 

**Authors**: Xiangwen Zhang, Qian Zhang, Longfei Han, Qiang Qu, Xiaoming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20654)  

**Abstract**: Collecting real-world vehicle accident videos for autonomous driving research is challenging due to their rarity and complexity. While existing driving video generation methods may produce visually realistic videos, they often fail to deliver physically realistic simulations because they lack the capability to generate accurate post-collision trajectories. In this paper, we introduce AccidentSim, a novel framework that generates physically realistic vehicle collision videos by extracting and utilizing the physical clues and contextual information available in real-world vehicle accident reports. Specifically, AccidentSim leverages a reliable physical simulator to replicate post-collision vehicle trajectories from the physical and contextual information in the accident reports and to build a vehicle collision trajectory dataset. This dataset is then used to fine-tune a language model, enabling it to respond to user prompts and predict physically consistent post-collision trajectories across various driving scenarios based on user descriptions. Finally, we employ Neural Radiance Fields (NeRF) to render high-quality backgrounds, merging them with the foreground vehicles that exhibit physically realistic trajectories to generate vehicle collision videos. Experimental results demonstrate that the videos produced by AccidentSim excel in both visual and physical authenticity. 

---
# TN-Eval: Rubric and Evaluation Protocols for Measuring the Quality of Behavioral Therapy Notes 

**Authors**: Raj Sanjay Shah, Lei Xu, Qianchu Liu, Jon Burnsky, Drew Bertagnolli, Chaitanya Shivade  

**Link**: [PDF](https://arxiv.org/pdf/2503.20648)  

**Abstract**: Behavioral therapy notes are important for both legal compliance and patient care. Unlike progress notes in physical health, quality standards for behavioral therapy notes remain underdeveloped. To address this gap, we collaborated with licensed therapists to design a comprehensive rubric for evaluating therapy notes across key dimensions: completeness, conciseness, and faithfulness. Further, we extend a public dataset of behavioral health conversations with therapist-written notes and LLM-generated notes, and apply our evaluation framework to measure their quality. We find that: (1) A rubric-based manual evaluation protocol offers more reliable and interpretable results than traditional Likert-scale annotations. (2) LLMs can mimic human evaluators in assessing completeness and conciseness but struggle with faithfulness. (3) Therapist-written notes often lack completeness and conciseness, while LLM-generated notes contain hallucination. Surprisingly, in a blind test, therapists prefer and judge LLM-generated notes to be superior to therapist-written notes. 

---
# $β$-GNN: A Robust Ensemble Approach Against Graph Structure Perturbation 

**Authors**: Haci Ismail Aslan, Philipp Wiesner, Ping Xiong, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20630)  

**Abstract**: Graph Neural Networks (GNNs) are playing an increasingly important role in the efficient operation and security of computing systems, with applications in workload scheduling, anomaly detection, and resource management. However, their vulnerability to network perturbations poses a significant challenge. We propose $\beta$-GNN, a model enhancing GNN robustness without sacrificing clean data performance. $\beta$-GNN uses a weighted ensemble, combining any GNN with a multi-layer perceptron. A learned dynamic weight, $\beta$, modulates the GNN's contribution. This $\beta$ not only weights GNN influence but also indicates data perturbation levels, enabling proactive mitigation. Experimental results on diverse datasets show $\beta$-GNN's superior adversarial accuracy and attack severity quantification. Crucially, $\beta$-GNN avoids perturbation assumptions, preserving clean data structure and performance. 

---
# Collaborative Storytelling and LLM: A Linguistic Analysis of Automatically-Generated Role-Playing Game Sessions 

**Authors**: Alessandro Maisto  

**Link**: [PDF](https://arxiv.org/pdf/2503.20623)  

**Abstract**: Role-playing games (RPG) are games in which players interact with one another to create narratives. The role of players in the RPG is largely based on the interaction between players and their characters. This emerging form of shared narrative, primarily oral, is receiving increasing attention. In particular, many authors investigated the use of an LLM as an actor in the game. In this paper, we aim to discover to what extent the language of Large Language Models (LLMs) exhibit oral or written features when asked to generate an RPG session without human interference. We will conduct a linguistic analysis of the lexical and syntactic features of the generated texts and compare the results with analyses of conversations, transcripts of human RPG sessions, and books. We found that LLMs exhibit a pattern that is distinct from all other text categories, including oral conversations, human RPG sessions and books. Our analysis has shown how training influences the way LLMs express themselves and provides important indications of the narrative capabilities of these tools. 

---
# State-Aware Perturbation Optimization for Robust Deep Reinforcement Learning 

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.20613)  

**Abstract**: Recently, deep reinforcement learning (DRL) has emerged as a promising approach for robotic control. However, the deployment of DRL in real-world robots is hindered by its sensitivity to environmental perturbations. While existing whitebox adversarial attacks rely on local gradient information and apply uniform perturbations across all states to evaluate DRL robustness, they fail to account for temporal dynamics and state-specific vulnerabilities. To combat the above challenge, we first conduct a theoretical analysis of white-box attacks in DRL by establishing the adversarial victim-dynamics Markov decision process (AVD-MDP), to derive the necessary and sufficient conditions for a successful attack. Based on this, we propose a selective state-aware reinforcement adversarial attack method, named STAR, to optimize perturbation stealthiness and state visitation dispersion. STAR first employs a soft mask-based state-targeting mechanism to minimize redundant perturbations, enhancing stealthiness and attack effectiveness. Then, it incorporates an information-theoretic optimization objective to maximize mutual information between perturbations, environmental states, and victim actions, ensuring a dispersed state-visitation distribution that steers the victim agent into vulnerable states for maximum return reduction. Extensive experiments demonstrate that STAR outperforms state-of-the-art benchmarks. 

---
# A decision-theoretic approach to dealing with uncertainty in quantum mechanics 

**Authors**: Keano De Vos, Gert de Cooman, Alexander Erreygers, Jasper De Bock  

**Link**: [PDF](https://arxiv.org/pdf/2503.20607)  

**Abstract**: We provide a decision-theoretic framework for dealing with uncertainty in quantum mechanics. This uncertainty is two-fold: on the one hand there may be uncertainty about the state the quantum system is in, and on the other hand, as is essential to quantum mechanical uncertainty, even if the quantum state is known, measurements may still produce an uncertain outcome. In our framework, measurements therefore play the role of acts with an uncertain outcome and our simple decision-theoretic postulates ensure that Born's rule is encapsulated in the utility functions associated with such acts. This approach allows us to uncouple (precise) probability theory from quantum mechanics, in the sense that it leaves room for a more general, so-called imprecise probabilities approach. We discuss the mathematical implications of our findings, which allow us to give a decision-theoretic foundation to recent seminal work by Benavoli, Facchini and Zaffalon, and we compare our approach to earlier and different approaches by Deutsch and Wallace. 

---
# StableToolBench-MirrorAPI: Modeling Tool Environments as Mirrors of 7,000+ Real-World APIs 

**Authors**: Zhicheng Guo, Sijie Cheng, Yuchen Niu, Hao Wang, Sicheng Zhou, Wenbing Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20527)  

**Abstract**: The rapid advancement of large language models (LLMs) has spurred significant interest in tool learning, where LLMs are augmented with external tools to tackle complex tasks. However, existing tool environments face challenges in balancing stability, scalability, and realness, particularly for benchmarking purposes. To address this problem, we propose MirrorAPI, a novel framework that trains specialized LLMs to accurately simulate real API responses, effectively acting as "mirrors" to tool environments. Using a comprehensive dataset of request-response pairs from 7,000+ APIs, we employ supervised fine-tuning and chain-of-thought reasoning to enhance simulation fidelity. MirrorAPI achieves superior accuracy and stability compared to state-of-the-art methods, as demonstrated by its performance on the newly constructed MirrorAPI-Bench and its integration into StableToolBench. 

---
# GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving 

**Authors**: Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, Gianluca Corrado  

**Link**: [PDF](https://arxiv.org/pdf/2503.20523)  

**Abstract**: Generative models offer a scalable and flexible paradigm for simulating complex environments, yet current approaches fall short in addressing the domain-specific requirements of autonomous driving - such as multi-agent interactions, fine-grained control, and multi-camera consistency. We introduce GAIA-2, Generative AI for Autonomy, a latent diffusion world model that unifies these capabilities within a single generative framework. GAIA-2 supports controllable video generation conditioned on a rich set of structured inputs: ego-vehicle dynamics, agent configurations, environmental factors, and road semantics. It generates high-resolution, spatiotemporally consistent multi-camera videos across geographically diverse driving environments (UK, US, Germany). The model integrates both structured conditioning and external latent embeddings (e.g., from a proprietary driving model) to facilitate flexible and semantically grounded scene synthesis. Through this integration, GAIA-2 enables scalable simulation of both common and rare driving scenarios, advancing the use of generative world models as a core tool in the development of autonomous systems. Videos are available at this https URL. 

---
# Design and Evaluation of Neural Network-Based Receiver Architectures for Reliable Communication 

**Authors**: Hüseyin Çevik, Erhan Karakoca, İbrahim Hökelek, Ali Görçin  

**Link**: [PDF](https://arxiv.org/pdf/2503.20500)  

**Abstract**: Neural network-based receivers leverage deep learning to optimize signal detection and decoding, significantly improving bit-error rate (BER) and block-error rate (BLER) in challenging environments. This study evaluates various architectures and compares their BER and BLER performance across different noise levels. Two novel models, the Dual Attention Transformer (DAT) and the Residual Dual Non-Local Attention Network (RDNLA), integrate self-attention and residual learning to enhance signal reconstruction. These models bypass conventional channel estimation and equalization by directly predicting log-likelihood ratios (LLRs) from received signals, with noise variance as an additional input. Simulations show that DAT and RDNLA outperform traditional and other neural receiver models under varying signal-to-noise ratios (SNR), while their computational efficiency supports their feasibility for next-generation communication systems. 

---
# Towards Efficient and General-Purpose Few-Shot Misclassification Detection for Vision-Language Models 

**Authors**: Fanhu Zeng, Zhen Cheng, Fei Zhu, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20492)  

**Abstract**: Reliable prediction by classifiers is crucial for their deployment in high security and dynamically changing situations. However, modern neural networks often exhibit overconfidence for misclassified predictions, highlighting the need for confidence estimation to detect errors. Despite the achievements obtained by existing methods on small-scale datasets, they all require training from scratch and there are no efficient and effective misclassification detection (MisD) methods, hindering practical application towards large-scale and ever-changing datasets. In this paper, we pave the way to exploit vision language model (VLM) leveraging text information to establish an efficient and general-purpose misclassification detection framework. By harnessing the power of VLM, we construct FSMisD, a Few-Shot prompt learning framework for MisD to refrain from training from scratch and therefore improve tuning efficiency. To enhance misclassification detection ability, we use adaptive pseudo sample generation and a novel negative loss to mitigate the issue of overconfidence by pushing category prompts away from pseudo features. We conduct comprehensive experiments with prompt learning methods and validate the generalization ability across various datasets with domain shift. Significant and consistent improvement demonstrates the effectiveness, efficiency and generalizability of our approach. 

---
# Underwater Image Enhancement by Convolutional Spiking Neural Networks 

**Authors**: Vidya Sudevan, Fakhreddine Zayer, Rizwana Kausar, Sajid Javed, Hamad Karki, Giulia De Masi, Jorge Dias  

**Link**: [PDF](https://arxiv.org/pdf/2503.20485)  

**Abstract**: Underwater image enhancement (UIE) is fundamental for marine applications, including autonomous vision-based navigation. Deep learning methods using convolutional neural networks (CNN) and vision transformers advanced UIE performance. Recently, spiking neural networks (SNN) have gained attention for their lightweight design, energy efficiency, and scalability. This paper introduces UIE-SNN, the first SNN-based UIE algorithm to improve visibility of underwater images. UIE-SNN is a 19- layered convolutional spiking encoder-decoder framework with skip connections, directly trained using surrogate gradient-based backpropagation through time (BPTT) strategy. We explore and validate the influence of training datasets on energy reduction, a unique advantage of UIE-SNN architecture, in contrast to the conventional learning-based architectures, where energy consumption is model-dependent. UIE-SNN optimizes the loss function in latent space representation to reconstruct clear underwater images. Our algorithm performs on par with its non-spiking counterpart methods in terms of PSNR and structural similarity index (SSIM) at reduced timesteps ($T=5$) and energy consumption of $85\%$. The algorithm is trained on two publicly available benchmark datasets, UIEB and EUVP, and tested on unseen images from UIEB, EUVP, LSUI, U45, and our custom UIE dataset. The UIE-SNN algorithm achieves PSNR of \(17.7801~dB\) and SSIM of \(0.7454\) on UIEB, and PSNR of \(23.1725~dB\) and SSIM of \(0.7890\) on EUVP. UIE-SNN achieves this algorithmic performance with fewer operators (\(147.49\) GSOPs) and energy (\(0.1327~J\)) compared to its non-spiking counterpart (GFLOPs = \(218.88\) and Energy=\(1.0068~J\)). Compared with existing SOTA UIE methods, UIE-SNN achieves an average of \(6.5\times\) improvement in energy efficiency. The source code is available at \href{this https URL}{UIE-SNN}. 

---
# Contrastive Learning Guided Latent Diffusion Model for Image-to-Image Translation 

**Authors**: Qi Si, Bo Wang, Zhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20484)  

**Abstract**: The diffusion model has demonstrated superior performance in synthesizing diverse and high-quality images for text-guided image translation. However, there remains room for improvement in both the formulation of text prompts and the preservation of reference image content. First, variations in target text prompts can significantly influence the quality of the generated images, and it is often challenging for users to craft an optimal prompt that fully captures the content of the input image. Second, while existing models can introduce desired modifications to specific regions of the reference image, they frequently induce unintended alterations in areas that should remain unchanged. To address these challenges, we propose pix2pix-zeroCon, a zero-shot diffusion-based method that eliminates the need for additional training by leveraging patch-wise contrastive loss. Specifically, we automatically determine the editing direction in the text embedding space based on the reference image and target prompts. Furthermore, to ensure precise content and structural preservation in the edited image, we introduce cross-attention guiding loss and patch-wise contrastive loss between the generated and original image embeddings within a pre-trained diffusion model. Notably, our approach requires no additional training and operates directly on a pre-trained text-to-image diffusion model. Extensive experiments demonstrate that our method surpasses existing models in image-to-image translation, achieving enhanced fidelity and controllability. 

---
# A multi-agentic framework for real-time, autonomous freeform metasurface design 

**Authors**: Robert Lupoiu, Yixuan Shao, Tianxiang Dai, Chenkai Mao, Kofi Edee, Jonathan A. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20479)  

**Abstract**: Innovation in nanophotonics currently relies on human experts who synergize specialized knowledge in photonics and coding with simulation and optimization algorithms, entailing design cycles that are time-consuming, computationally demanding, and frequently suboptimal. We introduce MetaChat, a multi-agentic design framework that can translate semantically described photonic design goals into high-performance, freeform device layouts in an automated, nearly real-time manner. Multi-step reasoning is enabled by our Agentic Iterative Monologue (AIM) paradigm, which coherently interfaces agents with code-based tools, other specialized agents, and human designers. Design acceleration is facilitated by Feature-wise Linear Modulation-conditioned Maxwell surrogate solvers that support the generalized evaluation of metasurface structures. We use freeform dielectric metasurfaces as a model system and demonstrate with MetaChat the design of multi-objective, multi-wavelength metasurfaces orders of magnitude faster than conventional methods. These concepts present a scientific computing blueprint for utilizing specialist design agents, surrogate solvers, and human interactions to drive multi-physics innovation and discovery. 

---
# From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment 

**Authors**: Yucheng Suo, Fan Ma, Linchao Zhu, Tianyi Wang, Fengyun Rao, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20472)  

**Abstract**: Multi-modal Large language models (MLLMs) show remarkable ability in video understanding. Nevertheless, understanding long videos remains challenging as the models can only process a finite number of frames in a single inference, potentially omitting crucial visual information. To address the challenge, we propose generating multiple predictions through visual context sampling, followed by a scoring mechanism to select the final prediction. Specifically, we devise a bin-wise sampling strategy that enables MLLMs to generate diverse answers based on various combinations of keyframes, thereby enriching the visual context. To determine the final prediction from the sampled answers, we employ a self-reward by linearly combining three scores: (1) a frequency score indicating the prevalence of each option, (2) a marginal confidence score reflecting the inter-intra sample certainty of MLLM predictions, and (3) a reasoning score for different question types, including clue-guided answering for global questions and temporal self-refocusing for local questions. The frequency score ensures robustness through majority correctness, the confidence-aligned score reflects prediction certainty, and the typed-reasoning score addresses cases with sparse key visual information using tailored strategies. Experiments show that this approach covers the correct answer for a high percentage of long video questions, on seven datasets show that our method improves the performance of three MLLMs. 

---
# Attention Xception UNet (AXUNet): A Novel Combination of CNN and Self-Attention for Brain Tumor Segmentation 

**Authors**: Farzan Moodi, Fereshteh Khodadadi Shoushtari, Gelareh Valizadeh, Dornaz Mazinani, Hanieh Mobarak Salari, Hamidreza Saligheh Rad  

**Link**: [PDF](https://arxiv.org/pdf/2503.20446)  

**Abstract**: Accurate segmentation of glioma brain tumors is crucial for diagnosis and treatment planning. Deep learning techniques offer promising solutions, but optimal model architectures remain under investigation. We used the BraTS 2021 dataset, selecting T1 with contrast enhancement (T1CE), T2, and Fluid-Attenuated Inversion Recovery (FLAIR) sequences for model development. The proposed Attention Xception UNet (AXUNet) architecture integrates an Xception backbone with dot-product self-attention modules, inspired by state-of-the-art (SOTA) large language models such as Google Bard and OpenAI ChatGPT, within a UNet-shaped model. We compared AXUNet with SOTA models. Comparative evaluation on the test set demonstrated improved results over baseline models. Inception-UNet and Xception-UNet achieved mean Dice scores of 90.88 and 93.24, respectively. Attention ResUNet (AResUNet) attained a mean Dice score of 92.80, with the highest score of 84.92 for enhancing tumor (ET) among all models. Attention Gate UNet (AGUNet) yielded a mean Dice score of 90.38. AXUNet outperformed all models with a mean Dice score of 93.73. It demonstrated superior Dice scores across whole tumor (WT) and tumor core (TC) regions, achieving 92.59 for WT, 86.81 for TC, and 84.89 for ET. The integration of the Xception backbone and dot-product self-attention mechanisms in AXUNet showcases enhanced performance in capturing spatial and contextual information. The findings underscore the potential utility of AXUNet in facilitating precise tumor delineation. 

---
# Evaluating Facial Expression Recognition Datasets for Deep Learning: A Benchmark Study with Novel Similarity Metrics 

**Authors**: F. Xavier Gaya-Morey, Cristina Manresa-Yee, Célia Martinie, Jose M. Buades-Rubio  

**Link**: [PDF](https://arxiv.org/pdf/2503.20428)  

**Abstract**: This study investigates the key characteristics and suitability of widely used Facial Expression Recognition (FER) datasets for training deep learning models. In the field of affective computing, FER is essential for interpreting human emotions, yet the performance of FER systems is highly contingent on the quality and diversity of the underlying datasets. To address this issue, we compiled and analyzed 24 FER datasets, including those targeting specific age groups such as children, adults, and the elderly, and processed them through a comprehensive normalization pipeline. In addition, we enriched the datasets with automatic annotations for age and gender, enabling a more nuanced evaluation of their demographic properties. To further assess dataset efficacy, we introduce three novel metricsLocal, Global, and Paired Similarity, which quantitatively measure dataset difficulty, generalization capability, and cross-dataset transferability. Benchmark experiments using state-of-the-art neural networks reveal that large-scale, automatically collected datasets (e.g., AffectNet, FER2013) tend to generalize better, despite issues with labeling noise and demographic biases, whereas controlled datasets offer higher annotation quality but limited variability. Our findings provide actionable recommendations for dataset selection and design, advancing the development of more robust, fair, and effective FER systems. 

---
# Including local feature interactions in deep non-negative matrix factorization networks improves performance 

**Authors**: Mahbod Nouri, David Rotermund, Alberto Garcia-Ortiz, Klaus R. Pawelzik  

**Link**: [PDF](https://arxiv.org/pdf/2503.20398)  

**Abstract**: The brain uses positive signals as a means of signaling. Forward interactions in the early visual cortex are also positive, realized by excitatory synapses. Only local interactions also include inhibition. Non-negative matrix factorization (NMF) captures the biological constraint of positive long-range interactions and can be implemented with stochastic spikes. While NMF can serve as an abstract formalization of early neural processing in the visual system, the performance of deep convolutional networks with NMF modules does not match that of CNNs of similar size. However, when the local NMF modules are each followed by a module that mixes the NMF's positive activities, the performances on the benchmark data exceed that of vanilla deep convolutional networks of similar size. This setting can be considered a biologically more plausible emulation of the processing in cortical (hyper-)columns with the potential to improve the performance of deep networks. 

---
# FastFT: Accelerating Reinforced Feature Transformation via Advanced Exploration Strategies 

**Authors**: Tianqi He, Xiaohan Huang, Yi Du, Qingqing Long, Ziyue Qiao, Min Wu, Yanjie Fu, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20394)  

**Abstract**: Feature Transformation is crucial for classic machine learning that aims to generate feature combinations to enhance the performance of downstream tasks from a data-centric perspective. Current methodologies, such as manual expert-driven processes, iterative-feedback techniques, and exploration-generative tactics, have shown promise in automating such data engineering workflow by minimizing human involvement. However, three challenges remain in those frameworks: (1) It predominantly depends on downstream task performance metrics, as assessment is time-consuming, especially for large datasets. (2) The diversity of feature combinations will hardly be guaranteed after random exploration ends. (3) Rare significant transformations lead to sparse valuable feedback that hinders the learning processes or leads to less effective results. In response to these challenges, we introduce FastFT, an innovative framework that leverages a trio of advanced this http URL first decouple the feature transformation evaluation from the outcomes of the generated datasets via the performance predictor. To address the issue of reward sparsity, we developed a method to evaluate the novelty of generated transformation sequences. Incorporating this novelty into the reward function accelerates the model's exploration of effective transformations, thereby improving the search productivity. Additionally, we combine novelty and performance to create a prioritized memory buffer, ensuring that essential experiences are effectively revisited during exploration. Our extensive experimental evaluations validate the performance, efficiency, and traceability of our proposed framework, showcasing its superiority in handling complex feature transformation tasks. 

---
# MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation 

**Authors**: Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20384)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in understanding complex language and visual data, enabling generalist robotic systems to interpret instructions and perform embodied tasks. Nevertheless, their real-world deployment is hindered by substantial computational and storage demands. Recent insights into the homogeneous patterns in the LLM layer have inspired sparsification techniques to address these challenges, such as early exit and token pruning. However, these methods often neglect the critical role of the final layers that encode the semantic information most relevant to downstream robotic tasks. Aligning with the recent breakthrough of the Shallow Brain Hypothesis (SBH) in neuroscience and the mixture of experts in model sparsification, we conceptualize each LLM layer as an expert and propose a Mixture-of-Layers Vision-Language-Action model (MoLe-VLA, or simply MoLe) architecture for dynamic LLM layer activation. We introduce a Spatial-Temporal Aware Router (STAR) for MoLe to selectively activate only parts of the layers based on the robot's current state, mimicking the brain's distinct signal pathways specialized for cognition and causal reasoning. Additionally, to compensate for the cognitive ability of LLMs lost in MoLe, we devise a Cognition Self-Knowledge Distillation (CogKD) framework. CogKD enhances the understanding of task demands and improves the generation of task-relevant action sequences by leveraging cognitive features. Extensive experiments conducted in both RLBench simulation and real-world environments demonstrate the superiority of MoLe-VLA in both efficiency and performance. Specifically, MoLe-VLA achieves an 8% improvement in the mean success rate across ten tasks while reducing computational costs by up to x5.6 compared to standard LLMs. 

---
# VideoGEM: Training-free Action Grounding in Videos 

**Authors**: Felix Vogel, Walid Bousselham, Anna Kukleva, Nina Shvetsova, Hilde Kuehne  

**Link**: [PDF](https://arxiv.org/pdf/2503.20348)  

**Abstract**: Vision-language foundation models have shown impressive capabilities across various zero-shot tasks, including training-free localization and grounding, primarily focusing on localizing objects in images. However, leveraging those capabilities to localize actions and events in videos is challenging, as actions have less physical outline and are usually described by higher-level concepts. In this work, we propose VideoGEM, the first training-free spatial action grounding method based on pretrained image- and video-language backbones. Namely, we adapt the self-self attention formulation of GEM to spatial activity grounding. We observe that high-level semantic concepts, such as actions, usually emerge in the higher layers of the image- and video-language models. We, therefore, propose a layer weighting in the self-attention path to prioritize higher layers. Additionally, we introduce a dynamic weighting method to automatically tune layer weights to capture each layer`s relevance to a specific prompt. Finally, we introduce a prompt decomposition, processing action, verb, and object prompts separately, resulting in a better spatial localization of actions. We evaluate the proposed approach on three image- and video-language backbones, CLIP, OpenCLIP, and ViCLIP, and on four video grounding datasets, V-HICO, DALY, YouCook-Interactions, and GroundingYouTube, showing that the proposed training-free approach is able to outperform current trained state-of-the-art approaches for spatial video grounding. 

---
# Wasserstein Distributionally Robust Bayesian Optimization with Continuous Context 

**Authors**: Francesco Micheli, Efe C. Balta, Anastasios Tsiamis, John Lygeros  

**Link**: [PDF](https://arxiv.org/pdf/2503.20341)  

**Abstract**: We address the challenge of sequential data-driven decision-making under context distributional uncertainty. This problem arises in numerous real-world scenarios where the learner optimizes black-box objective functions in the presence of uncontrollable contextual variables. We consider the setting where the context distribution is uncertain but known to lie within an ambiguity set defined as a ball in the Wasserstein distance. We propose a novel algorithm for Wasserstein Distributionally Robust Bayesian Optimization that can handle continuous context distributions while maintaining computational tractability. Our theoretical analysis combines recent results in self-normalized concentration in Hilbert spaces and finite-sample bounds for distributionally robust optimization to establish sublinear regret bounds that match state-of-the-art results. Through extensive comparisons with existing approaches on both synthetic and real-world problems, we demonstrate the simplicity, effectiveness, and practical applicability of our proposed method. 

---
# Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models 

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20320)  

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt. 

---
# A Multilingual, Culture-First Approach to Addressing Misgendering in LLM Applications 

**Authors**: Sunayana Sitaram, Adrian de Wynter, Isobel McCrum, Qilong Gu, Si-Qing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20302)  

**Abstract**: Misgendering is the act of referring to someone by a gender that does not match their chosen identity. It marginalizes and undermines a person's sense of self, causing significant harm. English-based approaches have clear-cut approaches to avoiding misgendering, such as the use of the pronoun ``they''. However, other languages pose unique challenges due to both grammatical and cultural constructs. In this work we develop methodologies to assess and mitigate misgendering across 42 languages and dialects using a participatory-design approach to design effective and appropriate guardrails across all languages. We test these guardrails in a standard large language model-based application (meeting transcript summarization), where both the data generation and the annotation steps followed a human-in-the-loop approach. We find that the proposed guardrails are very effective in reducing misgendering rates across all languages in the summaries generated, and without incurring loss of quality. Our human-in-the-loop approach demonstrates a method to feasibly scale inclusive and responsible AI-based solutions across multiple languages and cultures. 

---
# Context-Aware Weakly Supervised Image Manipulation Localization with SAM Refinement 

**Authors**: Xinghao Wang, Changtao Miao, Dianmo Sheng, Tao Gong, Qi Chu, Bin Liu, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20294)  

**Abstract**: Malicious image manipulation poses societal risks, increasing the importance of effective image manipulation detection methods. Recent approaches in image manipulation detection have largely been driven by fully supervised approaches, which require labor-intensive pixel-level annotations. Thus, it is essential to explore weakly supervised image manipulation localization methods that only require image-level binary labels for training. However, existing weakly supervised image manipulation methods overlook the importance of edge information for accurate localization, leading to suboptimal localization performance. To address this, we propose a Context-Aware Boundary Localization (CABL) module to aggregate boundary features and learn context-inconsistency for localizing manipulated areas. Furthermore, by leveraging Class Activation Mapping (CAM) and Segment Anything Model (SAM), we introduce the CAM-Guided SAM Refinement (CGSR) module to generate more accurate manipulation localization maps. By integrating two modules, we present a novel weakly supervised framework based on a dual-branch Transformer-CNN architecture. Our method achieves outstanding localization performance across multiple datasets. 

---
# CryoSAMU: Enhancing 3D Cryo-EM Density Maps of Protein Structures at Intermediate Resolution with Structure-Aware Multimodal U-Nets 

**Authors**: Chenwei Zhang, Anne Condon, Khanh Dao Duc  

**Link**: [PDF](https://arxiv.org/pdf/2503.20291)  

**Abstract**: Enhancing cryogenic electron microscopy (cryo-EM) 3D density maps at intermediate resolution (4-8 Å) is crucial in protein structure determination. Recent advances in deep learning have led to the development of automated approaches for enhancing experimental cryo-EM density maps. Yet, these methods are not optimized for intermediate-resolution maps and rely on map density features alone. To address this, we propose CryoSAMU, a novel method designed to enhance 3D cryo-EM density maps of protein structures using structure-aware multimodal U-Nets and trained on curated intermediate-resolution density maps. We comprehensively evaluate CryoSAMU across various metrics and demonstrate its competitive performance compared to state-of-the-art methods. Notably, CryoSAMU achieves significantly faster processing speed, showing promise for future practical applications. Our code is available at this https URL. 

---
# QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions 

**Authors**: Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Yu Tsao, Junichi Yamagishi, Yuxuan Wang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20290)  

**Abstract**: This paper explores a novel perspective to speech quality assessment by leveraging natural language descriptions, offering richer, more nuanced insights than traditional numerical scoring methods. Natural language feedback provides instructive recommendations and detailed evaluations, yet existing datasets lack the comprehensive annotations needed for this approach. To bridge this gap, we introduce QualiSpeech, a comprehensive low-level speech quality assessment dataset encompassing 11 key aspects and detailed natural language comments that include reasoning and contextual insights. Additionally, we propose the QualiSpeech Benchmark to evaluate the low-level speech understanding capabilities of auditory large language models (LLMs). Experimental results demonstrate that finetuned auditory LLMs can reliably generate detailed descriptions of noise and distortion, effectively identifying their types and temporal characteristics. The results further highlight the potential for incorporating reasoning to enhance the accuracy and reliability of quality assessments. The dataset will be released at this https URL. 

---
# Model-Based Offline Reinforcement Learning with Adversarial Data Augmentation 

**Authors**: Hongye Cao, Fan Feng, Jing Huo, Shangdong Yang, Meng Fang, Tianpei Yang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20285)  

**Abstract**: Model-based offline Reinforcement Learning (RL) constructs environment models from offline datasets to perform conservative policy optimization. Existing approaches focus on learning state transitions through ensemble models, rollouting conservative estimation to mitigate extrapolation errors. However, the static data makes it challenging to develop a robust policy, and offline agents cannot access the environment to gather new data. To address these challenges, we introduce Model-based Offline Reinforcement learning with AdversariaL data augmentation (MORAL). In MORAL, we replace the fixed horizon rollout by employing adversaria data augmentation to execute alternating sampling with ensemble models to enrich training data. Specifically, this adversarial process dynamically selects ensemble models against policy for biased sampling, mitigating the optimistic estimation of fixed models, thus robustly expanding the training data for policy optimization. Moreover, a differential factor is integrated into the adversarial process for regularization, ensuring error minimization in extrapolations. This data-augmented optimization adapts to diverse offline tasks without rollout horizon tuning, showing remarkable applicability. Extensive experiments on D4RL benchmark demonstrate that MORAL outperforms other model-based offline RL methods in terms of policy learning and sample efficiency. 

---
# Faster Parameter-Efficient Tuning with Token Redundancy Reduction 

**Authors**: Kwonyoung Kim, Jungin Park, Jin Kim, Hyeongjun Kwon, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2503.20282)  

**Abstract**: Parameter-efficient tuning (PET) aims to transfer pre-trained foundation models to downstream tasks by learning a small number of parameters. Compared to traditional fine-tuning, which updates the entire model, PET significantly reduces storage and transfer costs for each task regardless of exponentially increasing pre-trained model capacity. However, most PET methods inherit the inference latency of their large backbone models and often introduce additional computational overhead due to additional modules (e.g. adapters), limiting their practicality for compute-intensive applications. In this paper, we propose Faster Parameter-Efficient Tuning (FPET), a novel approach that enhances inference speed and training efficiency while maintaining high storage efficiency. Specifically, we introduce a plug-and-play token redundancy reduction module delicately designed for PET. This module refines tokens from the self-attention layer using an adapter to learn the accurate similarity between tokens and cuts off the tokens through a fully-differentiable token merging strategy, which uses a straight-through estimator for optimal token reduction. Experimental results prove that our FPET achieves faster inference and higher memory efficiency than the pre-trained backbone while keeping competitive performance on par with state-of-the-art PET methods. 

---
# Are We There Yet? Unraveling the State-of-the-Art Graph Network Intrusion Detection Systems 

**Authors**: Chenglong Wang, Pujia Zheng, Jiaping Gui, Cunqing Hua, Wajih Ul Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20281)  

**Abstract**: Network Intrusion Detection Systems (NIDS) are vital for ensuring enterprise security. Recently, Graph-based NIDS (GIDS) have attracted considerable attention because of their capability to effectively capture the complex relationships within the graph structures of data communications. Despite their promise, the reproducibility and replicability of these GIDS remain largely unexplored, posing challenges for developing reliable and robust detection systems. This study bridges this gap by designing a systematic approach to evaluate state-of-the-art GIDS, which includes critically assessing, extending, and clarifying the findings of these systems. We further assess the robustness of GIDS under adversarial attacks. Evaluations were conducted on three public datasets as well as a newly collected large-scale enterprise dataset. Our findings reveal significant performance discrepancies, highlighting challenges related to dataset scale, model inputs, and implementation settings. We demonstrate difficulties in reproducing and replicating results, particularly concerning false positive rates and robustness against adversarial attacks. This work provides valuable insights and recommendations for future research, emphasizing the importance of rigorous reproduction and replication studies in developing robust and generalizable GIDS solutions. 

---
# sudo rm -rf agentic_security 

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.20279)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs. 

---
# LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions 

**Authors**: Yejin Kwon, Daeun Moon, Youngje Oh, Hyunsoo Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2503.20252)  

**Abstract**: Anomaly Detection (AD) focuses on detecting samples that differ from the standard pattern, making it a vital tool in process control. Logical anomalies may appear visually normal yet violate predefined constraints on object presence, arrangement, or quantity, depending on reasoning and explainability. We introduce LogicQA, a framework that enhances AD by providing industrial operators with explanations for logical anomalies. LogicQA compiles automatically generated questions into a checklist and collects responses to identify violations of logical constraints. LogicQA is training-free, annotation-free, and operates in a few-shot setting. We achieve state-of-the-art (SOTA) Logical AD performance on public benchmarks, MVTec LOCO AD, with an AUROC of 87.6 percent and an F1-max of 87.0 percent along with the explanations of anomalies. Also, our approach has shown outstanding performance on semiconductor SEM corporate data, further validating its effectiveness in industrial applications. 

---
# ESSR: An 8K@30FPS Super-Resolution Accelerator With Edge Selective Network 

**Authors**: Chih-Chia Hsu, Tian-Sheuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20245)  

**Abstract**: Deep learning-based super-resolution (SR) is challenging to implement in resource-constrained edge devices for resolutions beyond full HD due to its high computational complexity and memory bandwidth requirements. This paper introduces an 8K@30FPS SR accelerator with edge-selective dynamic input processing. Dynamic processing chooses the appropriate subnets for different patches based on simple input edge criteria, achieving a 50\% MAC reduction with only a 0.1dB PSNR decrease. The quality of reconstruction images is guaranteed and maximized its potential with \textit{resource adaptive model switching} even under resource constraints. In conjunction with hardware-specific refinements, the model size is reduced by 84\% to 51K, but with a decrease of less than 0.6dB PSNR. Additionally, to support dynamic processing with high utilization, this design incorporates a \textit{configurable group of layer mapping} that synergizes with the \textit{structure-friendly fusion block}, resulting in 77\% hardware utilization and up to 79\% reduction in feature SRAM access. The implementation, using the TSMC 28nm process, can achieve 8K@30FPS throughput at 800MHz with a gate count of 2749K, 0.2075W power consumption, and 4797Mpixels/J energy efficiency, exceeding previous work. 

---
# LGR: LLM-Guided Ranking of Frontiers for Object Goal Navigation 

**Authors**: Mitsuaki Uno, Kanji Tanaka, Daiki Iwata, Yudai Noda, Shoya Miyazaki, Kouki Terashima  

**Link**: [PDF](https://arxiv.org/pdf/2503.20241)  

**Abstract**: Object Goal Navigation (OGN) is a fundamental task for robots and AI, with key applications such as mobile robot image databases (MRID). In particular, mapless OGN is essential in scenarios involving unknown or dynamic environments. This study aims to enhance recent modular mapless OGN systems by leveraging the commonsense reasoning capabilities of large language models (LLMs). Specifically, we address the challenge of determining the visiting order in frontier-based exploration by framing it as a frontier ranking problem. Our approach is grounded in recent findings that, while LLMs cannot determine the absolute value of a frontier, they excel at evaluating the relative value between multiple frontiers viewed within a single image using the view image as context. We dynamically manage the frontier list by adding and removing elements, using an LLM as a ranking model. The ranking results are represented as reciprocal rank vectors, which are ideal for multi-view, multi-query information fusion. We validate the effectiveness of our method through evaluations in Habitat-Sim. 

---
# Dynamic Learning and Productivity for Data Analysts: A Bayesian Hidden Markov Model Perspective 

**Authors**: Yue Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.20233)  

**Abstract**: Data analysts are essential in organizations, transforming raw data into insights that drive decision-making and strategy. This study explores how analysts' productivity evolves on a collaborative platform, focusing on two key learning activities: writing queries and viewing peer queries. While traditional research often assumes static models, where performance improves steadily with cumulative learning, such models fail to capture the dynamic nature of real-world learning. To address this, we propose a Hidden Markov Model (HMM) that tracks how analysts transition between distinct learning states based on their participation in these activities.
Using an industry dataset with 2,001 analysts and 79,797 queries, this study identifies three learning states: novice, intermediate, and advanced. Productivity increases as analysts advance to higher states, reflecting the cumulative benefits of learning. Writing queries benefits analysts across all states, with the largest gains observed for novices. Viewing peer queries supports novices but may hinder analysts in higher states due to cognitive overload or inefficiencies. Transitions between states are also uneven, with progression from intermediate to advanced being particularly challenging. This study advances understanding of into dynamic learning behavior of knowledge worker and offers practical implications for designing systems, optimizing training, enabling personalized learning, and fostering effective knowledge sharing. 

---
# Dynamics of Algorithmic Content Amplification on TikTok 

**Authors**: Fabian Baumann, Nipun Arora, Iyad Rahwan, Agnieszka Czaplicka  

**Link**: [PDF](https://arxiv.org/pdf/2503.20231)  

**Abstract**: Intelligent algorithms increasingly shape the content we encounter and engage with online. TikTok's For You feed exemplifies extreme algorithm-driven curation, tailoring the stream of video content almost exclusively based on users' explicit and implicit interactions with the platform. Despite growing attention, the dynamics of content amplification on TikTok remain largely unquantified. How quickly, and to what extent, does TikTok's algorithm amplify content aligned with users' interests? To address these questions, we conduct a sock-puppet audit, deploying bots with different interests to engage with TikTok's "For You" feed. Our findings reveal that content aligned with the bots' interests undergoes strong amplification, with rapid reinforcement typically occurring within the first 200 videos watched. While amplification is consistently observed across all interests, its intensity varies by interest, indicating the emergence of topic-specific biases. Time series analyses and Markov models uncover distinct phases of recommendation dynamics, including persistent content reinforcement and a gradual decline in content diversity over time. Although TikTok's algorithm preserves some content diversity, we find a strong negative correlation between amplification and exploration: as the amplification of interest-aligned content increases, engagement with unseen hashtags declines. These findings contribute to discussions on socio-algorithmic feedback loops in the digital age and the trade-offs between personalization and content diversity. 

---
# TraNCE: Transformative Non-linear Concept Explainer for CNNs 

**Authors**: Ugochukwu Ejike Akpudo, Yongsheng Gao, Jun Zhou, Andrew Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2503.20230)  

**Abstract**: Convolutional neural networks (CNNs) have succeeded remarkably in various computer vision tasks. However, they are not intrinsically explainable. While the feature-level understanding of CNNs reveals where the models looked, concept-based explainability methods provide insights into what the models saw. However, their assumption of linear reconstructability of image activations fails to capture the intricate relationships within these activations. Their Fidelity-only approach to evaluating global explanations also presents a new concern. For the first time, we address these limitations with the novel Transformative Nonlinear Concept Explainer (TraNCE) for CNNs. Unlike linear reconstruction assumptions made by existing methods, TraNCE captures the intricate relationships within the activations. This study presents three original contributions to the CNN explainability literature: (i) An automatic concept discovery mechanism based on variational autoencoders (VAEs). This transformative concept discovery process enhances the identification of meaningful concepts from image activations. (ii) A visualization module that leverages the Bessel function to create a smooth transition between prototypical image pixels, revealing not only what the CNN saw but also what the CNN avoided, thereby mitigating the challenges of concept duplication as documented in previous works. (iii) A new metric, the Faith score, integrates both Coherence and Fidelity for a comprehensive evaluation of explainer faithfulness and consistency. 

---
# Advancements in Natural Language Processing: Exploring Transformer-Based Architectures for Text Understanding 

**Authors**: Tianhao Wu, Yu Wang, Ngoc Quach  

**Link**: [PDF](https://arxiv.org/pdf/2503.20227)  

**Abstract**: Natural Language Processing (NLP) has witnessed a transformative leap with the advent of transformer-based architectures, which have significantly enhanced the ability of machines to understand and generate human-like text. This paper explores the advancements in transformer models, such as BERT and GPT, focusing on their superior performance in text understanding tasks compared to traditional methods like recurrent neural networks (RNNs). By analyzing statistical properties through visual representations-including probability density functions of text length distributions and feature space classifications-the study highlights the models' proficiency in handling long-range dependencies, adapting to conditional shifts, and extracting features for classification, even with overlapping classes. Drawing on recent 2024 research, including enhancements in multi-hop knowledge graph reasoning and context-aware chat interactions, the paper outlines a methodology involving data preparation, model selection, pretraining, fine-tuning, and evaluation. The results demonstrate state-of-the-art performance on benchmarks like GLUE and SQuAD, with F1 scores exceeding 90%, though challenges such as high computational costs persist. This work underscores the pivotal role of transformers in modern NLP and suggests future directions, including efficiency optimization and multimodal integration, to further advance language-based AI systems. 

---
# Learning Adaptive Dexterous Grasping from Single Demonstrations 

**Authors**: Liangzhi Shi, Yulin Liu, Lingqi Zeng, Bo Ai, Zhengdong Hong, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.20208)  

**Abstract**: How can robots learn dexterous grasping skills efficiently and apply them adaptively based on user instructions? This work tackles two key challenges: efficient skill acquisition from limited human demonstrations and context-driven skill selection. We introduce AdaDexGrasp, a framework that learns a library of grasping skills from a single human demonstration per skill and selects the most suitable one using a vision-language model (VLM). To improve sample efficiency, we propose a trajectory following reward that guides reinforcement learning (RL) toward states close to a human demonstration while allowing flexibility in exploration. To learn beyond the single demonstration, we employ curriculum learning, progressively increasing object pose variations to enhance robustness. At deployment, a VLM retrieves the appropriate skill based on user instructions, bridging low-level learned skills with high-level intent. We evaluate AdaDexGrasp in both simulation and real-world settings, showing that our approach significantly improves RL efficiency and enables learning human-like grasp strategies across varied object configurations. Finally, we demonstrate zero-shot transfer of our learned policies to a real-world PSYONIC Ability Hand, with a 90% success rate across objects, significantly outperforming the baseline. 

---
# Generalized Phase Pressure Control Enhanced Reinforcement Learning for Traffic Signal Control 

**Authors**: Xiao-Cheng Liao, Yi Mei, Mengjie Zhang, Xiang-Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20205)  

**Abstract**: Appropriate traffic state representation is crucial for learning traffic signal control policies. However, most of the current traffic state representations are heuristically designed, with insufficient theoretical support. In this paper, we (1) develop a flexible, efficient, and theoretically grounded method, namely generalized phase pressure (G2P) control, which takes only simple lane features into consideration to decide which phase to be actuated; 2) extend the pressure control theory to a general form for multi-homogeneous-lane road networks based on queueing theory; (3) design a new traffic state representation based on the generalized phase state features from G2P control; and 4) develop a reinforcement learning (RL)-based algorithm template named G2P-XLight, and two RL algorithms, G2P-MPLight and G2P-CoLight, by combining the generalized phase state representation with MPLight and CoLight, two well-performed RL methods for learning traffic signal control policies. Extensive experiments conducted on multiple real-world datasets demonstrate that G2P control outperforms the state-of-the-art (SOTA) heuristic method in the transportation field and other recent human-designed heuristic methods; and that the newly proposed G2P-XLight significantly outperforms SOTA learning-based approaches. Our code is available online. 

---
# SARGes: Semantically Aligned Reliable Gesture Generation via Intent Chain 

**Authors**: Nan Gao, Yihua Bao, Dongdong Weng, Jiayi Zhao, Jia Li, Yan Zhou, Pengfei Wan, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20202)  

**Abstract**: Co-speech gesture generation enhances human-computer interaction realism through speech-synchronized gesture synthesis. However, generating semantically meaningful gestures remains a challenging problem. We propose SARGes, a novel framework that leverages large language models (LLMs) to parse speech content and generate reliable semantic gesture labels, which subsequently guide the synthesis of meaningful co-speech this http URL, we constructed a comprehensive co-speech gesture ethogram and developed an LLM-based intent chain reasoning mechanism that systematically parses and decomposes gesture semantics into structured inference steps following ethogram criteria, effectively guiding LLMs to generate context-aware gesture labels. Subsequently, we constructed an intent chain-annotated text-to-gesture label dataset and trained a lightweight gesture label generation model, which then guides the generation of credible and semantically coherent co-speech gestures. Experimental results demonstrate that SARGes achieves highly semantically-aligned gesture labeling (50.2% accuracy) with efficient single-pass inference (0.4 seconds). The proposed method provides an interpretable intent reasoning pathway for semantic gesture synthesis. 

---
# Assessing SAM for Tree Crown Instance Segmentation from Drone Imagery 

**Authors**: Mélisande Teng, Arthur Ouaknine, Etienne Laliberté, Yoshua Bengio, David Rolnick, Hugo Larochelle  

**Link**: [PDF](https://arxiv.org/pdf/2503.20199)  

**Abstract**: The potential of tree planting as a natural climate solution is often undermined by inadequate monitoring of tree planting projects. Current monitoring methods involve measuring trees by hand for each species, requiring extensive cost, time, and labour. Advances in drone remote sensing and computer vision offer great potential for mapping and characterizing trees from aerial imagery, and large pre-trained vision models, such as the Segment Anything Model (SAM), may be a particularly compelling choice given limited labeled data. In this work, we compare SAM methods for the task of automatic tree crown instance segmentation in high resolution drone imagery of young tree plantations. We explore the potential of SAM for this task, and find that methods using SAM out-of-the-box do not outperform a custom Mask R-CNN, even with well-designed prompts, but that there is potential for methods which tune SAM further. We also show that predictions can be improved by adding Digital Surface Model (DSM) information as an input. 

---
# Leveraging Implicit Sentiments: Enhancing Reliability and Validity in Psychological Trait Evaluation of LLMs 

**Authors**: Huanhuan Ma, Haisong Gong, Xiaoyuan Yi, Xing Xie, Dongkuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20182)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have led to their increasing integration into human life. With the transition from mere tools to human-like assistants, understanding their psychological aspects-such as emotional tendencies and personalities-becomes essential for ensuring their trustworthiness. However, current psychological evaluations of LLMs, often based on human psychological assessments like the BFI, face significant limitations. The results from these approaches often lack reliability and have limited validity when predicting LLM behavior in real-world scenarios. In this work, we introduce a novel evaluation instrument specifically designed for LLMs, called Core Sentiment Inventory (CSI). CSI is a bilingual tool, covering both English and Chinese, that implicitly evaluates models' sentiment tendencies, providing an insightful psychological portrait of LLM across three dimensions: optimism, pessimism, and neutrality. Through extensive experiments, we demonstrate that: 1) CSI effectively captures nuanced emotional patterns, revealing significant variation in LLMs across languages and contexts; 2) Compared to current approaches, CSI significantly improves reliability, yielding more consistent results; and 3) The correlation between CSI scores and the sentiment of LLM's real-world outputs exceeds 0.85, demonstrating its strong validity in predicting LLM behavior. We make CSI public available via: this https URL. 

---
# Offline Reinforcement Learning with Discrete Diffusion Skills 

**Authors**: RuiXi Qiao, Jie Cheng, Xingyuan Dai, Yonglin Tian, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2503.20176)  

**Abstract**: Skills have been introduced to offline reinforcement learning (RL) as temporal abstractions to tackle complex, long-horizon tasks, promoting consistent behavior and enabling meaningful exploration. While skills in offline RL are predominantly modeled within a continuous latent space, the potential of discrete skill spaces remains largely underexplored. In this paper, we propose a compact discrete skill space for offline RL tasks supported by state-of-the-art transformer-based encoder and diffusion-based decoder. Coupled with a high-level policy trained via offline RL techniques, our method establishes a hierarchical RL framework where the trained diffusion decoder plays a pivotal role. Empirical evaluations show that the proposed algorithm, Discrete Diffusion Skill (DDS), is a powerful offline RL method. DDS performs competitively on Locomotion and Kitchen tasks and excels on long-horizon tasks, achieving at least a 12 percent improvement on AntMaze-v2 benchmarks compared to existing offline RL approaches. Furthermore, DDS offers improved interpretability, training stability, and online exploration compared to previous skill-based methods. 

---
# Look Before Leap: Look-Ahead Planning with Uncertainty in Reinforcement Learning 

**Authors**: Yongshuai Liu, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20139)  

**Abstract**: Model-based reinforcement learning (MBRL) has demonstrated superior sample efficiency compared to model-free reinforcement learning (MFRL). However, the presence of inaccurate models can introduce biases during policy learning, resulting in misleading trajectories. The challenge lies in obtaining accurate models due to limited diverse training data, particularly in regions with limited visits (uncertain regions). Existing approaches passively quantify uncertainty after sample generation, failing to actively collect uncertain samples that could enhance state coverage and improve model accuracy. Moreover, MBRL often faces difficulties in making accurate multi-step predictions, thereby impacting overall performance. To address these limitations, we propose a novel framework for uncertainty-aware policy optimization with model-based exploratory planning. In the model-based planning phase, we introduce an uncertainty-aware k-step lookahead planning approach to guide action selection at each step. This process involves a trade-off analysis between model uncertainty and value function approximation error, effectively enhancing policy performance. In the policy optimization phase, we leverage an uncertainty-driven exploratory policy to actively collect diverse training samples, resulting in improved model accuracy and overall performance of the RL agent. Our approach offers flexibility and applicability to tasks with varying state/action spaces and reward structures. We validate its effectiveness through experiments on challenging robotic manipulation tasks and Atari games, surpassing state-of-the-art methods with fewer interactions, thereby leading to significant performance improvements. 

---
# Unlocking the Value of Decentralized Data: A Federated Dual Learning Approach for Model Aggregation 

**Authors**: Junyi Zhu, Ruicong Yao, Taha Ceritli, Savas Ozkan, Matthew B. Blaschko, Eunchung Noh, Jeongwon Min, Cho Jung Min, Mete Ozay  

**Link**: [PDF](https://arxiv.org/pdf/2503.20138)  

**Abstract**: Artificial Intelligence (AI) technologies have revolutionized numerous fields, yet their applications often rely on costly and time-consuming data collection processes. Federated Learning (FL) offers a promising alternative by enabling AI models to be trained on decentralized data where data is scattered across clients (distributed nodes). However, existing FL approaches struggle to match the performance of centralized training due to challenges such as heterogeneous data distribution and communication delays, limiting their potential for breakthroughs. We observe that many real-world use cases involve hybrid data regimes, in which a server (center node) has access to some data while a large amount of data is distributed across associated clients. To improve the utilization of decentralized data under this regime, address data heterogeneity issue, and facilitate asynchronous communication between the server and clients, we propose a dual learning approach that leverages centralized data at the server to guide the merging of model updates from clients. Our method accommodates scenarios where server data is out-of-domain relative to decentralized client data, making it applicable to a wide range of use cases. We provide theoretical analysis demonstrating the faster convergence of our method compared to existing methods. Furthermore, experimental results across various scenarios show that our approach significantly outperforms existing technologies, highlighting its potential to unlock the value of large amounts of decentralized data. 

---
# Can We Make Code Green? Understanding Trade-Offs in LLMs vs. Human Code Optimizations 

**Authors**: Pooja Rani, Jan-Andrea Bard, June Sallou, Alexander Boll, Timo Kehrer, Alberto Bacchelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.20126)  

**Abstract**: The rapid technological evolution has accelerated software development for various domains and use cases, contributing to a growing share of global carbon emissions. While recent large language models (LLMs) claim to assist developers in optimizing code for performance and energy efficiency, their efficacy in real-world scenarios remains under exploration. In this work, we explore the effectiveness of LLMs in reducing the environmental footprint of real-world projects, focusing on software written in Matlab-widely used in both academia and industry for scientific and engineering applications. We analyze energy-focused optimization on 400 scripts across 100 top GitHub repositories. We examine potential 2,176 optimizations recommended by leading LLMs, such as GPT-3, GPT-4, Llama, and Mixtral, and a senior Matlab developer, on energy consumption, memory usage, execution time consumption, and code correctness. The developer serves as a real-world baseline for comparing typical human and LLM-generated optimizations.
Mapping these optimizations to 13 high-level themes, we found that LLMs propose a broad spectrum of improvements--beyond energy efficiency--including improving code readability and maintainability, memory management, error handling while the developer overlooked some parallel processing, error handling etc. However, our statistical tests reveal that the energy-focused optimizations unexpectedly negatively impacted memory usage, with no clear benefits regarding execution time or energy consumption. Our qualitative analysis of energy-time trade-offs revealed that some themes, such as vectorization preallocation, were among the common themes shaping these trade-offs. With LLMs becoming ubiquitous in modern software development, our study serves as a call to action: prioritizing the evaluation of common coding practices to identify the green ones. 

---
# Zero-Shot Human-Object Interaction Synthesis with Multimodal Priors 

**Authors**: Yuke Lou, Yiming Wang, Zhen Wu, Rui Zhao, Wenjia Wang, Mingyi Shi, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2503.20118)  

**Abstract**: Human-object interaction (HOI) synthesis is important for various applications, ranging from virtual reality to robotics. However, acquiring 3D HOI data is challenging due to its complexity and high cost, limiting existing methods to the narrow diversity of object types and interaction patterns in training datasets. This paper proposes a novel zero-shot HOI synthesis framework without relying on end-to-end training on currently limited 3D HOI datasets. The core idea of our method lies in leveraging extensive HOI knowledge from pre-trained Multimodal Models. Given a text description, our system first obtains temporally consistent 2D HOI image sequences using image or video generation models, which are then uplifted to 3D HOI milestones of human and object poses. We employ pre-trained human pose estimation models to extract human poses and introduce a generalizable category-level 6-DoF estimation method to obtain the object poses from 2D HOI images. Our estimation method is adaptive to various object templates obtained from text-to-3D models or online retrieval. A physics-based tracking of the 3D HOI kinematic milestone is further applied to refine both body motions and object poses, yielding more physically plausible HOI generation results. The experimental results demonstrate that our method is capable of generating open-vocabulary HOIs with physical realism and semantic diversity. 

---
# Efficient Model Development through Fine-tuning Transfer 

**Authors**: Pin-Jie Lin, Rishab Balasubramanian, Fengyuan Liu, Nikhil Kandpal, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20110)  

**Abstract**: Modern LLMs struggle with efficient updates, as each new pretrained model version requires repeating expensive alignment processes. This challenge also applies to domain- or language-specific models, where fine-tuning on specialized data must be redone for every new base model release. In this paper, we explore the transfer of fine-tuning updates between model versions. Specifically, we derive the diff vector from one source model version, which represents the weight changes from fine-tuning, and apply it to the base model of a different target version. Through empirical evaluations on various open-weight model versions, we show that transferring diff vectors can significantly improve the target base model, often achieving performance comparable to its fine-tuned counterpart. For example, reusing the fine-tuning updates from Llama 3.0 8B leads to an absolute accuracy improvement of 10.7% on GPQA over the base Llama 3.1 8B without additional training, surpassing Llama 3.1 8B Instruct. In a multilingual model development setting, we show that this approach can significantly increase performance on target-language tasks without retraining, achieving an absolute improvement of 4.7% and 15.5% on Global MMLU for Malagasy and Turkish, respectively, compared to Llama 3.1 8B Instruct. Our controlled experiments reveal that fine-tuning transfer is most effective when the source and target models are linearly connected in the parameter space. Additionally, we demonstrate that fine-tuning transfer offers a stronger and more computationally efficient starting point for further fine-tuning. Finally, we propose an iterative recycling-then-finetuning approach for continuous model development, which improves both efficiency and effectiveness. Our findings suggest that fine-tuning transfer is a viable strategy to reduce training costs while maintaining model performance. 

---
# AI Identity, Empowerment, and Mindfulness in Mitigating Unethical AI Use 

**Authors**: Mayssam Tarighi Shaayesteh, Sara Memarian Esfahani, Hossein Mohit  

**Link**: [PDF](https://arxiv.org/pdf/2503.20099)  

**Abstract**: This study examines how AI identity influences psychological empowerment and unethical AI behavior among college students, while also exploring the moderating role of IT mindfulness. Findings show that a strong AI identity enhances psychological empowerment and academic engagement but can also lead to increased unethical AI practices. Crucially, IT mindfulness acts as an ethical safeguard, promoting sensitivity to ethical concerns and reducing misuse of AI. These insights have implications for educators, policymakers, and AI developers, emphasizing For Peer Review the need for a balanced approach that encourages digital engagement without compromising student responsibility. The study also contributes to philosophical discussions of psychological agency, suggesting that empowerment through AI can yield both positive and negative outcomes. Mindfulness emerges as essential in guiding ethical AI interactions. Overall, the research informs ongoing debates on ethics in education and AI, offering strategies to align technological advancement with ethical accountability and responsible use. 

---
# Can Multi-modal (reasoning) LLMs work as deepfake detectors? 

**Authors**: Simiao Ren, Yao Yao, Kidus Zewde, Zisheng Liang, Tsang, Ning-Yau Cheng, Xiaoou Zhan, Qinzhe Liu, Yifei Chen, Hengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20084)  

**Abstract**: Deepfake detection remains a critical challenge in the era of advanced generative models, particularly as synthetic media becomes more sophisticated. In this study, we explore the potential of state of the art multi-modal (reasoning) large language models (LLMs) for deepfake image detection such as (OpenAI O1/4o, Gemini thinking Flash 2, Deepseek Janus, Grok 3, llama 3.2, Qwen 2/2.5 VL, Mistral Pixtral, Claude 3.5/3.7 sonnet) . We benchmark 12 latest multi-modal LLMs against traditional deepfake detection methods across multiple datasets, including recently published real-world deepfake imagery. To enhance performance, we employ prompt tuning and conduct an in-depth analysis of the models' reasoning pathways to identify key contributing factors in their decision-making process. Our findings indicate that best multi-modal LLMs achieve competitive performance with promising generalization ability with zero shot, even surpass traditional deepfake detection pipelines in out-of-distribution datasets while the rest of the LLM families performs extremely disappointing with some worse than random guess. Furthermore, we found newer model version and reasoning capabilities does not contribute to performance in such niche tasks of deepfake detection while model size do help in some cases. This study highlights the potential of integrating multi-modal reasoning in future deepfake detection frameworks and provides insights into model interpretability for robustness in real-world scenarios. 

---
# Abstracting Geo-specific Terrains to Scale Up Reinforcement Learning 

**Authors**: Volkan Ustun, Soham Hans, Rajay Kumar, Yunzhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20078)  

**Abstract**: Multi-agent reinforcement learning (MARL) is increasingly ubiquitous in training dynamic and adaptive synthetic characters for interactive simulations on geo-specific terrains. Frameworks such as Unity's ML-Agents help to make such reinforcement learning experiments more accessible to the simulation community. Military training simulations also benefit from advances in MARL, but they have immense computational requirements due to their complex, continuous, stochastic, partially observable, non-stationary, and doctrine-based nature. Furthermore, these simulations require geo-specific terrains, further exacerbating the computational resources problem. In our research, we leverage Unity's waypoints to automatically generate multi-layered representation abstractions of the geo-specific terrains to scale up reinforcement learning while still allowing the transfer of learned policies between different representations. Our early exploratory results on a novel MARL scenario, where each side has differing objectives, indicate that waypoint-based navigation enables faster and more efficient learning while producing trajectories similar to those taken by expert human players in CSGO gaming environments. This research points out the potential of waypoint-based navigation for reducing the computational costs of developing and training MARL models for military training simulations, where geo-specific terrains and differing objectives are crucial. 

---
# Adaptive Orchestration for Large-Scale Inference on Heterogeneous Accelerator Systems Balancing Cost, Performance, and Resilience 

**Authors**: Yahav Biran, Imry Kissos  

**Link**: [PDF](https://arxiv.org/pdf/2503.20074)  

**Abstract**: The surge in generative AI workloads has created a need for scalable inference systems that can flexibly harness both GPUs and specialized accelerators while containing operational costs. This paper proposes a hardware-agnostic control loop that adaptively allocates requests across heterogeneous accelerators based on real-time cost and capacity signals. The approach sustains low latency and high throughput by dynamically shifting between cost-optimized and capacity-optimized modes, ensuring the most efficient use of expensive compute resources under fluctuating availability. Evaluated using the Stable Diffusion model, the framework consistently meets latency targets, automatically redirects traffic during capacity shortfalls, and capitalizes on lower-cost accelerators when possible. These results highlight how a feedback-driven deployment strategy, spanning the entire software and hardware stack, can help organizations efficiently scale generative AI workloads while maintaining resilience in the face of limited accelerator capacity. 

---
# BugCraft: End-to-End Crash Bug Reproduction Using LLM Agents in Minecraft 

**Authors**: Eray Yapağcı, Yavuz Alp Sencer Öztürk, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2503.20036)  

**Abstract**: Reproducing game bugs, in our case crash bugs in continuously evolving games like Minecraft, is a notoriously manual, time-consuming, and challenging process to automate. Despite the success of LLM-driven bug reproduction in other software domains, games, with their complex interactive environments, remain largely unaddressed. This paper introduces BugCraft, a novel end-to-end framework designed to automate the reproduction of crash bugs in Minecraft directly from user-submitted bug reports, addressing the critical gap in automated game bug reproduction. BugCraft employs a two-stage approach: first, a Step Synthesizer leverages LLMs and Minecraft Wiki knowledge to transform bug reports into high-quality, structured steps to reproduce (S2R). Second, an Action Model, powered by a vision-based LLM agent (GPT-4o) and a custom macro API, executes these S2R steps within Minecraft to trigger the reported crash. To facilitate evaluation, we introduce BugCraft-Bench, a curated dataset of Minecraft crash bug reports. Evaluated on BugCraft-Bench, our framework successfully reproduced 30.23% of crash bugs end-to-end. The Step Synthesizer demonstrated a 66.28% accuracy in generating correct bug reproduction plans, highlighting its effectiveness in interpreting and structuring bug report information. BugCraft demonstrates the feasibility of automated reproduction of crash bugs in complex game environments using LLMs, opening promising avenues for game testing and development. The framework and the BugCraft-Bench dataset pave the way for future research in automated game bug analysis and hold potential for generalization to other interactive game platforms. Finally, we make our code open at this https URL 

---
# Experience Replay Addresses Loss of Plasticity in Continual Learning 

**Authors**: Jiuqi Wang, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20018)  

**Abstract**: Loss of plasticity is one of the main challenges in continual learning with deep neural networks, where neural networks trained via backpropagation gradually lose their ability to adapt to new tasks and perform significantly worse than their freshly initialized counterparts. The main contribution of this paper is to propose a new hypothesis that experience replay addresses the loss of plasticity in continual learning. Here, experience replay is a form of memory. We provide supporting evidence for this hypothesis. In particular, we demonstrate in multiple different tasks, including regression, classification, and policy evaluation, that by simply adding an experience replay and processing the data in the experience replay with Transformers, the loss of plasticity disappears. Notably, we do not alter any standard components of deep learning. For example, we do not change backpropagation. We do not modify the activation functions. And we do not use any regularization. We conjecture that experience replay and Transformers can address the loss of plasticity because of the in-context learning phenomenon. 

---
# ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback 

**Authors**: Bohan Zhai, Canwen Xu, Yuxiong He, Zhewei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.19988)  

**Abstract**: Text-to-SQL demands precise reasoning to convert natural language questions into structured queries. While large language models (LLMs) excel in many reasoning tasks, their ability to leverage Chain-of-Thought (CoT) reasoning for text-to-SQL remains underexplored. We identify critical limitations: zero-shot CoT offers minimal gains, and Direct Preference Optimization (DPO) applied without CoT yields marginal improvements. We propose ExCoT, a novel framework that iteratively optimizes open-source LLMs by combining CoT reasoning with off-policy and on-policy DPO, relying solely on execution accuracy as feedback. This approach eliminates the need for reward models or human-annotated preferences.
Our experimental results demonstrate significant performance gains: ExCoT improves execution accuracy on BIRD dev set from 57.37% to 68.51% and on Spider test set from 78.81% to 86.59% for LLaMA-3 70B, with Qwen-2.5-Coder demonstrating similar improvements. Our best model achieves state-of-the-art performance in the single-model setting on both BIRD and Spider datasets, notably achieving 68.53% on the BIRD test set. 

---
# ACVUBench: Audio-Centric Video Understanding Benchmark 

**Authors**: Yudong Yang, Jimin Zhuang, Guangzhi Sun, Changli Tang, Yixuan Li, Peihan Li, Yifan Jiang, Wei Li, Zejun Ma, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19951)  

**Abstract**: Audio often serves as an auxiliary modality in video understanding tasks of audio-visual large language models (LLMs), merely assisting in the comprehension of visual information. However, a thorough understanding of videos significantly depends on auditory information, as audio offers critical context, emotional cues, and semantic meaning that visual data alone often lacks. This paper proposes an audio-centric video understanding benchmark (ACVUBench) to evaluate the video comprehension capabilities of multimodal LLMs with a particular focus on auditory information. Specifically, ACVUBench incorporates 2,662 videos spanning 18 different domains with rich auditory information, together with over 13k high-quality human annotated or validated question-answer pairs. Moreover, ACVUBench introduces a suite of carefully designed audio-centric tasks, holistically testing the understanding of both audio content and audio-visual interactions in videos. A thorough evaluation across a diverse range of open-source and proprietary multimodal LLMs is performed, followed by the analyses of deficiencies in audio-visual LLMs. Demos are available at this https URL. 

---
# LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation 

**Authors**: Han Chen, Zicong Jiang, Zining Zhang, Bingsheng He, Pingyi Luo, Mian Lu, Yuqiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19950)  

**Abstract**: We introduce LogQuant, a groundbreaking 2-bit quantization technique for KV Cache in large language model (LLM) inference, delivering substantial memory savings while preserving superior performance. Previous methods either assume that later tokens are more important or attempt to predict important tokens based on earlier attention patterns. Both approaches, however, can result in performance bottlenecks or frequent mispredictions.
LogQuant takes a different approach. By applying a log-based filtering mechanism, it selectively compresses the KV Cache across the entire context, achieving better performance with the same or even reduced memory footprint compared to existing methods. In benchmark tests, it enhances throughput by 25% and boosts batch size by 60% without increasing memory consumption. For challenging tasks such as Math and Code Completion, LogQuant improves accuracy by 40% to 200% at the same compression ratio, outperforming comparable this http URL integrates effortlessly with popular inference frameworks like Python's transformers library. Implementation can be available in this https URL. 

---
# Test-Time Reasoning Through Visual Human Preferences with VLMs and Soft Rewards 

**Authors**: Alexander Gambashidze, Konstantin Sobolev, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2503.19948)  

**Abstract**: Can Visual Language Models (VLMs) effectively capture human visual preferences? This work addresses this question by training VLMs to think about preferences at test time, employing reinforcement learning methods inspired by DeepSeek R1 and OpenAI O1. Using datasets such as ImageReward and Human Preference Score v2 (HPSv2), our models achieve accuracies of 64.9% on the ImageReward test set (trained on ImageReward official split) and 65.4% on HPSv2 (trained on approximately 25% of its data). These results match traditional encoder-based models while providing transparent reasoning and enhanced generalization. This approach allows to use not only rich VLM world knowledge, but also its potential to think, yielding interpretable outcomes that help decision-making processes. By demonstrating that human visual preferences reasonable by current VLMs, we introduce efficient soft-reward strategies for image ranking, outperforming simplistic selection or scoring methods. This reasoning capability enables VLMs to rank arbitrary images-regardless of aspect ratio or complexity-thereby potentially amplifying the effectiveness of visual Preference Optimization. By reducing the need for extensive markup while improving reward generalization and explainability, our findings can be a strong mile-stone that will enhance text-to-vision models even further. 

---
# Vanishing Depth: A Depth Adapter with Positional Depth Encoding for Generalized Image Encoders 

**Authors**: Paul Koch, Jörg Krüger, Ankit Chowdhury, Oliver Heimann  

**Link**: [PDF](https://arxiv.org/pdf/2503.19947)  

**Abstract**: Generalized metric depth understanding is critical for precise vision-guided robotics, which current state-of-the-art (SOTA) vision-encoders do not support. To address this, we propose Vanishing Depth, a self-supervised training approach that extends pretrained RGB encoders to incorporate and align metric depth into their feature embeddings. Based on our novel positional depth encoding, we enable stable depth density and depth distribution invariant feature extraction. We achieve performance improvements and SOTA results across a spectrum of relevant RGBD downstream tasks - without the necessity of finetuning the encoder. Most notably, we achieve 56.05 mIoU on SUN-RGBD segmentation, 88.3 RMSE on Void's depth completion, and 83.8 Top 1 accuracy on NYUv2 scene classification. In 6D-object pose estimation, we outperform our predecessors of DinoV2, EVA-02, and Omnivore and achieve SOTA results for non-finetuned encoders in several related RGBD downstream tasks. 

---
# Optimizing Breast Cancer Detection in Mammograms: A Comprehensive Study of Transfer Learning, Resolution Reduction, and Multi-View Classification 

**Authors**: Daniel G. P. Petrini, Hae Yong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.19945)  

**Abstract**: This study explores open questions in the application of machine learning for breast cancer detection in mammograms. Current approaches often employ a two-stage transfer learning process: first, adapting a backbone model trained on natural images to develop a patch classifier, which is then used to create a single-view whole-image classifier. Additionally, many studies leverage both mammographic views to enhance model performance. In this work, we systematically investigate five key questions: (1) Is the intermediate patch classifier essential for optimal performance? (2) Do backbone models that excel in natural image classification consistently outperform others on mammograms? (3) When reducing mammogram resolution for GPU processing, does the learn-to-resize technique outperform conventional methods? (4) Does incorporating both mammographic views in a two-view classifier significantly improve detection accuracy? (5) How do these findings vary when analyzing low-quality versus high-quality mammograms? By addressing these questions, we developed models that outperform previous results for both single-view and two-view classifiers. Our findings provide insights into model architecture and transfer learning strategies contributing to more accurate and efficient mammogram analysis. 

---
# A Spatiotemporal Radar-Based Precipitation Model for Water Level Prediction and Flood Forecasting 

**Authors**: Sakshi Dhankhar, Stefan Wittek, Hamidreza Eivazi, Andreas Rausch  

**Link**: [PDF](https://arxiv.org/pdf/2503.19943)  

**Abstract**: Study Region: Goslar and Göttingen, Lower Saxony, Germany. Study Focus: In July 2017, the cities of Goslar and Göttingen experienced severe flood events characterized by short warning time of only 20 minutes, resulting in extensive regional flooding and significant damage. This highlights the critical need for a more reliable and timely flood forecasting system. This paper presents a comprehensive study on the impact of radar-based precipitation data on forecasting river water levels in Goslar. Additionally, the study examines how precipitation influences water level forecasts in Göttingen. The analysis integrates radar-derived spatiotemporal precipitation patterns with hydrological sensor data obtained from ground stations to evaluate the effectiveness of this approach in improving flood prediction capabilities. New Hydrological Insights for the Region: A key innovation in this paper is the use of residual-based modeling to address the non-linearity between precipitation images and water levels, leading to a Spatiotemporal Radar-based Precipitation Model with residuals (STRPMr). Unlike traditional hydrological models, our approach does not rely on upstream data, making it independent of additional hydrological inputs. This independence enhances its adaptability and allows for broader applicability in other regions with RADOLAN precipitation. The deep learning architecture integrates (2+1)D convolutional neural networks for spatial and temporal feature extraction with LSTM for timeseries forecasting. The results demonstrate the potential of the STRPMr for capturing extreme events and more accurate flood forecasting. 

---
# Body Discovery of Embodied AI 

**Authors**: Zhe Sun, Pengfei Tian, Xiaozhu Hu, Xiaoyu Zhao, Huiying Li, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19941)  

**Abstract**: In the pursuit of realizing artificial general intelligence (AGI), the importance of embodied artificial intelligence (AI) becomes increasingly apparent. Following this trend, research integrating robots with AGI has become prominent. As various kinds of embodiments have been designed, adaptability to diverse embodiments will become important to AGI. We introduce a new challenge, termed "Body Discovery of Embodied AI", focusing on tasks of recognizing embodiments and summarizing neural signal functionality. The challenge encompasses the precise definition of an AI body and the intricate task of identifying embodiments in dynamic environments, where conventional approaches often prove inadequate. To address these challenges, we apply causal inference method and evaluate it by developing a simulator tailored for testing algorithms with virtual environments. Finally, we validate the efficacy of our algorithms through empirical testing, demonstrating their robust performance in various scenarios based on virtual environments. 

---
# FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling 

**Authors**: Qiusheng Huang, Xiaohui Zhong, Xu Fan, Lei Chen, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19940)  

**Abstract**: Similar to conventional video generation, current deep learning-based weather prediction frameworks often lack explicit physical constraints, leading to unphysical outputs that limit their reliability for operational forecasting. Among various physical processes requiring proper representation, radiation plays a fundamental role as it drives Earth's weather and climate systems. However, accurate simulation of radiative transfer processes remains challenging for traditional numerical weather prediction (NWP) models due to their inherent complexity and high computational costs. Here, we propose FuXi-RTM, a hybrid physics-guided deep learning framework designed to enhance weather forecast accuracy while enforcing physical consistency. FuXi-RTM integrates a primary forecasting model (FuXi) with a fixed deep learning-based radiative transfer model (DLRTM) surrogate that efficiently replaces conventional radiation parameterization schemes. This represents the first deep learning-based weather forecasting framework to explicitly incorporate physical process modeling. Evaluated over a comprehensive 5-year dataset, FuXi-RTM outperforms its unconstrained counterpart in 88.51% of 3320 variable and lead time combinations, with improvements in radiative flux predictions. By incorporating additional physical processes, FuXi-RTM paves the way for next-generation weather forecasting systems that are both accurate and physically consistent. 

---
# Reverse Prompt: Cracking the Recipe Inside Text-to-Image Generation 

**Authors**: Zhiyao Ren, Yibing Zhan, Baosheng Yu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.19937)  

**Abstract**: Text-to-image generation has become increasingly popular, but achieving the desired images often requires extensive prompt engineering. In this paper, we explore how to decode textual prompts from reference images, a process we refer to as image reverse prompt engineering. This technique enables us to gain insights from reference images, understand the creative processes of great artists, and generate impressive new images. To address this challenge, we propose a method known as automatic reverse prompt optimization (ARPO). Specifically, our method refines an initial prompt into a high-quality prompt through an iteratively imitative gradient prompt optimization process: 1) generating a recreated image from the current prompt to instantiate its guidance capability; 2) producing textual gradients, which are candidate prompts intended to reduce the difference between the recreated image and the reference image; 3) updating the current prompt with textual gradients using a greedy search method to maximize the CLIP similarity between prompt and reference image. We compare ARPO with several baseline methods, including handcrafted techniques, gradient-based prompt tuning methods, image captioning, and data-driven selection method. Both quantitative and qualitative results demonstrate that our ARPO converges quickly to generate high-quality reverse prompts. More importantly, we can easily create novel images with diverse styles and content by directly editing these reverse prompts. Code will be made publicly available. 

---
# Role of AI Innovation, Clean Energy and Digital Economy towards Net Zero Emission in the United States: An ARDL Approach 

**Authors**: Adita Sultana, Abdullah Al Abrar Chowdhury, Azizul Hakim Rafi, Abdulla All Noman  

**Link**: [PDF](https://arxiv.org/pdf/2503.19933)  

**Abstract**: The current paper investigates the influences of AI innovation, GDP growth, renewable energy utilization, the digital economy, and industrialization on CO2 emissions in the USA from 1990 to 2022, incorporating the ARDL methodology. The outcomes observe that AI innovation, renewable energy usage, and the digital economy reduce CO2 emissions, while GDP expansion and industrialization intensify ecosystem damage. Unit root tests (ADF, PP, and DF-GLS) reveal heterogeneous integration levels amongst components, ensuring robustness in the ARDL analysis. Complementary methods (FMOLS, DOLS, and CCR) validate the results, enhancing their reliability. Pairwise Granger causality assessments identify strong unidirectional connections within CO2 emissions and AI innovation, as well as the digital economy, underscoring their significant roles in ecological sustainability. This research highlights the requirement for strategic actions to nurture equitable growth, including advancements in AI technology, green energy adoption, and environmentally conscious industrial development, to improve environmental quality in the United States. 

---
