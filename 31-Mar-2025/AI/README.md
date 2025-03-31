# QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks? 

**Authors**: Belinda Z. Li, Been Kim, Zi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22674)  

**Abstract**: Recently, a large amount of work has focused on improving large language models' (LLMs') performance on reasoning benchmarks such as math and logic. However, past work has largely assumed that tasks are well-defined. In the real world, queries to LLMs are often underspecified, only solvable through acquiring missing information. We formalize this as a constraint satisfaction problem (CSP) with missing variable assignments. Using a special case of this formalism where only one necessary variable assignment is missing, we can rigorously evaluate an LLM's ability to identify the minimal necessary question to ask and quantify axes of difficulty levels for each problem. We present QuestBench, a set of underspecified reasoning tasks solvable by asking at most one question, which includes: (1) Logic-Q: Logical reasoning tasks with one missing proposition, (2) Planning-Q: PDDL planning problems with initial states that are partially-observed, (3) GSM-Q: Human-annotated grade school math problems with one missing variable assignment, and (4) GSME-Q: a version of GSM-Q where word problems are translated into equations by human annotators. The LLM is tasked with selecting the correct clarification question(s) from a list of options. While state-of-the-art models excel at GSM-Q and GSME-Q, their accuracy is only 40-50% on Logic-Q and Planning-Q. Analysis demonstrates that the ability to solve well-specified reasoning problems may not be sufficient for success on our benchmark: models have difficulty identifying the right question to ask, even when they can solve the fully specified version of the problem. Furthermore, in the Planning-Q domain, LLMs tend not to hedge, even when explicitly presented with the option to predict ``not sure.'' This highlights the need for deeper investigation into models' information acquisition capabilities. 

---
# ActionStudio: A Lightweight Framework for Data and Training of Action Models 

**Authors**: Jianguo Zhang, Thai Hoang, Ming Zhu, Zuxin Liu, Shiyu Wang, Tulika Awalgaonkar, Akshara Prabhakar, Haolin Chen, Weiran Yao, Zhiwei Liu, Juntao Tan, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22673)  

**Abstract**: Action models are essential for enabling autonomous agents to perform complex tasks. However, training large action models remains challenging due to the diversity of agent environments and the complexity of agentic data. Despite growing interest, existing infrastructure provides limited support for scalable, agent-specific fine-tuning. We present ActionStudio, a lightweight and extensible data and training framework designed for action models. ActionStudio unifies heterogeneous agent trajectories through a standardized format, supports diverse training paradigms including LoRA, full fine-tuning, and distributed setups, and integrates robust preprocessing and verification tools. We validate its effectiveness across both public and realistic industry benchmarks, demonstrating strong performance and practical scalability. We open-sourced code and data at this https URL to facilitate research in the community. 

---
# Unicorn: Text-Only Data Synthesis for Vision Language Model Training 

**Authors**: Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22655)  

**Abstract**: Training vision-language models (VLMs) typically requires large-scale, high-quality image-text pairs, but collecting or synthesizing such data is costly. In contrast, text data is abundant and inexpensive, prompting the question: can high-quality multimodal training data be synthesized purely from text? To tackle this, we propose a cross-integrated three-stage multimodal data synthesis framework, which generates two datasets: Unicorn-1.2M and Unicorn-471K-Instruction. In Stage 1: Diverse Caption Data Synthesis, we construct 1.2M semantically diverse high-quality captions by expanding sparse caption seeds using large language models (LLMs). In Stage 2: Instruction-Tuning Data Generation, we further process 471K captions into multi-turn instruction-tuning tasks to support complex reasoning. Finally, in Stage 3: Modality Representation Transfer, these textual captions representations are transformed into visual representations, resulting in diverse synthetic image representations. This three-stage process enables us to construct Unicorn-1.2M for pretraining and Unicorn-471K-Instruction for instruction-tuning, without relying on real images. By eliminating the dependency on real images while maintaining data quality and diversity, our framework offers a cost-effective and scalable solution for VLMs training. Code is available at this https URL. 

---
# CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models 

**Authors**: Zhihang Lin, Mingbao Lin, Yuan Xie, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.22342)  

**Abstract**: This paper introduces Completion Pruning Policy Optimization (CPPO) to accelerate the training of reasoning models based on Group Relative Policy Optimization (GRPO). GRPO, while effective, incurs high training costs due to the need for sampling multiple completions for each question. Our experiment and theoretical analysis reveals that the number of completions impacts model accuracy yet increases training time multiplicatively, and not all completions contribute equally to policy training -- their contribution depends on their relative advantage. To address these issues, we propose CPPO, which prunes completions with low absolute advantages, significantly reducing the number needed for gradient calculation and updates. Additionally, we introduce a dynamic completion allocation strategy to maximize GPU utilization by incorporating additional questions, further enhancing training efficiency. Experimental results demonstrate that CPPO achieves up to $8.32\times$ speedup on GSM8K and $3.51\times$ on Math while preserving or even enhancing the accuracy compared to the original GRPO. We release our code at this https URL. 

---
# Agent-Centric Personalized Multiple Clustering with Multi-Modal LLMs 

**Authors**: Ziye Chen, Yiqun Duan, Riheng Zhu, Zhenbang Sun, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22241)  

**Abstract**: Personalized multiple clustering aims to generate diverse partitions of a dataset based on different user-specific aspects, rather than a single clustering. It has recently drawn research interest for accommodating varying user preferences. Recent approaches primarily use CLIP embeddings with proxy learning to extract representations biased toward user clustering preferences. However, CLIP primarily focuses on coarse image-text alignment, lacking a deep contextual understanding of user interests. To overcome these limitations, we propose an agent-centric personalized clustering framework that leverages multi-modal large language models (MLLMs) as agents to comprehensively traverse a relational graph to search for clusters based on user interests. Due to the advanced reasoning mechanism of MLLMs, the obtained clusters align more closely with user-defined criteria than those obtained from CLIP-based representations. To reduce computational overhead, we shorten the agents' traversal path by constructing a relational graph using user-interest-biased embeddings extracted by MLLMs. A large number of weakly connected edges can be filtered out based on embedding similarity, facilitating an efficient traversal search for agents. Experimental results show that the proposed method achieves NMI scores of 0.9667 and 0.9481 on the Card Order and Card Suits benchmarks, respectively, largely improving the SOTA model by over 140%. 

---
# Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF 

**Authors**: Syrine Belakaria, Joshua Kazdan, Charles Marx, Chris Cundy, Willie Neiswanger, Sanmi Koyejo, Barbara E. Engelhardt, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2503.22137)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has become a cornerstone of the training and alignment pipeline for large language models (LLMs). Recent advances, such as direct preference optimization (DPO), have simplified the preference learning step. However, collecting preference data remains a challenging and costly process, often requiring expert annotation. This cost can be mitigated by carefully selecting the data points presented for annotation. In this work, we propose an active learning approach to efficiently select prompt and preference pairs using a risk assessment strategy based on the Sharpe Ratio. To address the challenge of unknown preferences prior to annotation, our method evaluates the gradients of all potential preference annotations to assess their impact on model updates. These gradient-based evaluations enable risk assessment of data points regardless of the annotation outcome. By leveraging the DPO loss derivations, we derive a closed-form expression for computing these Sharpe ratios on a per-tuple basis, ensuring our approach remains both tractable and computationally efficient. We also introduce two variants of our method, each making different assumptions about prior information. Experimental results demonstrate that our method outperforms the baseline by up to 5% in win rates against the chosen completion with limited human preference data across several language models and real-world datasets. 

---
# Multi-Task Semantic Communications via Large Models 

**Authors**: Wanli Ni, Zhijin Qin, Haofeng Sun, Xiaoming Tao, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.22064)  

**Abstract**: Artificial intelligence (AI) promises to revolutionize the design, optimization and management of next-generation communication systems. In this article, we explore the integration of large AI models (LAMs) into semantic communications (SemCom) by leveraging their multi-modal data processing and generation capabilities. Although LAMs bring unprecedented abilities to extract semantics from raw data, this integration entails multifaceted challenges including high resource demands, model complexity, and the need for adaptability across diverse modalities and tasks. To overcome these challenges, we propose a LAM-based multi-task SemCom (MTSC) architecture, which includes an adaptive model compression strategy and a federated split fine-tuning approach to facilitate the efficient deployment of LAM-based semantic models in resource-limited networks. Furthermore, a retrieval-augmented generation scheme is implemented to synthesize the most recent local and global knowledge bases to enhance the accuracy of semantic extraction and content generation, thereby improving the inference performance. Finally, simulation results demonstrate the efficacy of the proposed LAM-based MTSC architecture, highlighting the performance enhancements across various downstream tasks under varying channel conditions. 

---
# OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment 

**Authors**: Hamed Babaei Giglou, Jennifer D'Souza, Oliver Karras, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2503.21902)  

**Abstract**: Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems. We present OntoAligner, a comprehensive, modular, and robust Python toolkit for ontology alignment, designed to address current limitations with existing tools faced by practitioners. Existing tools are limited in scalability, modularity, and ease of integration with recent AI advances. OntoAligner provides a flexible architecture integrating existing lightweight OA techniques such as fuzzy matching but goes beyond by supporting contemporary methods with retrieval-augmented generation and large language models for OA. The framework prioritizes extensibility, enabling researchers to integrate custom alignment algorithms and datasets. This paper details the design principles, architecture, and implementation of the OntoAligner, demonstrating its utility through benchmarks on standard OA tasks. Our evaluation highlights OntoAligner's ability to handle large-scale ontologies efficiently with few lines of code while delivering high alignment quality. By making OntoAligner open-source, we aim to provide a resource that fosters innovation and collaboration within the OA community, empowering researchers and practitioners with a toolkit for reproducible OA research and real-world applications. 

---
# Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment 

**Authors**: Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Dylan J. Foster, Akshay Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21878)  

**Abstract**: Inference-time computation provides an important axis for scaling language model performance, but naively scaling compute through techniques like Best-of-$N$ sampling can cause performance to degrade due to reward hacking. Toward a theoretical understanding of how to best leverage additional computation, we focus on inference-time alignment which we formalize as the problem of improving a pre-trained policy's responses for a prompt of interest, given access to an imperfect reward model. We analyze the performance of inference-time alignment algorithms in terms of (i) response quality, and (ii) compute, and provide new results that highlight the importance of the pre-trained policy's coverage over high-quality responses for performance and compute scaling:
1. We show that Best-of-$N$ alignment with an ideal choice for $N$ can achieve optimal performance under stringent notions of coverage, but provably suffers from reward hacking when $N$ is large, and fails to achieve tight guarantees under more realistic coverage conditions.
2. We introduce $\texttt{InferenceTimePessimism}$, a new algorithm which mitigates reward hacking through deliberate use of inference-time compute, implementing the principle of pessimism in the face of uncertainty via rejection sampling; we prove that its performance is optimal and does not degrade with $N$, meaning it is scaling-monotonic.
We complement our theoretical results with an experimental evaluation that demonstrate the benefits of $\texttt{InferenceTimePessimism}$ across a variety of tasks and models. 

---
# DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness 

**Authors**: Ruining Li, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22677)  

**Abstract**: Most 3D object generators focus on aesthetic quality, often neglecting physical constraints necessary in applications. One such constraint is that the 3D object should be self-supporting, i.e., remains balanced under gravity. Prior approaches to generating stable 3D objects used differentiable physics simulators to optimize geometry at test-time, which is slow, unstable, and prone to local optima. Inspired by the literature on aligning generative models to external feedback, we propose Direct Simulation Optimization (DSO), a framework to use the feedback from a (non-differentiable) simulator to increase the likelihood that the 3D generator outputs stable 3D objects directly. We construct a dataset of 3D objects labeled with a stability score obtained from the physics simulator. We can then fine-tune the 3D generator using the stability score as the alignment metric, via direct preference optimization (DPO) or direct reward optimization (DRO), a novel objective, which we introduce, to align diffusion models without requiring pairwise preferences. Our experiments show that the fine-tuned feed-forward generator, using either DPO or DRO objective, is much faster and more likely to produce stable objects than test-time optimization. Notably, the DSO framework works even without any ground-truth 3D objects for training, allowing the 3D generator to self-improve by automatically collecting simulation feedback on its own outputs. 

---
# Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation 

**Authors**: Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Wu Jian, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22675)  

**Abstract**: Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose \textbf{ReaRec}, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30\%-50\%. Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation. 

---
# Exploring the Effectiveness of Multi-stage Fine-tuning for Cross-encoder Re-rankers 

**Authors**: Francesca Pezzuti, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2503.22672)  

**Abstract**: State-of-the-art cross-encoders can be fine-tuned to be highly effective in passage re-ranking. The typical fine-tuning process of cross-encoders as re-rankers requires large amounts of manually labelled data, a contrastive learning objective, and a set of heuristically sampled negatives. An alternative recent approach for fine-tuning instead involves teaching the model to mimic the rankings of a highly effective large language model using a distillation objective. These fine-tuning strategies can be applied either individually, or in sequence. In this work, we systematically investigate the effectiveness of point-wise cross-encoders when fine-tuned independently in a single stage, or sequentially in two stages. Our experiments show that the effectiveness of point-wise cross-encoders fine-tuned using contrastive learning is indeed on par with that of models fine-tuned with multi-stage approaches. Code is available for reproduction at this https URL. 

---
# Evaluation of Machine-generated Biomedical Images via A Tally-based Similarity Measure 

**Authors**: Frank J. Brooks, Rucha Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2503.22658)  

**Abstract**: Super-resolution, in-painting, whole-image generation, unpaired style-transfer, and network-constrained image reconstruction each include an aspect of machine-learned image synthesis where the actual ground truth is not known at time of use. It is generally difficult to quantitatively and authoritatively evaluate the quality of synthetic images; however, in mission-critical biomedical scenarios robust evaluation is paramount. In this work, all practical image-to-image comparisons really are relative qualifications, not absolute difference quantifications; and, therefore, meaningful evaluation of generated image quality can be accomplished using the Tversky Index, which is a well-established measure for assessing perceptual similarity. This evaluation procedure is developed and then demonstrated using multiple image data sets, both real and simulated. The main result is that when the subjectivity and intrinsic deficiencies of any feature-encoding choice are put upfront, Tversky's method leads to intuitive results, whereas traditional methods based on summarizing distances in deep feature spaces do not. 

---
# Empirical Analysis of Sim-and-Real Cotraining Of Diffusion Policies For Planar Pushing from Pixels 

**Authors**: Adam Wei, Abhinav Agarwal, Boyuan Chen, Rohan Bosworth, Nicholas Pfaff, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.22634)  

**Abstract**: In imitation learning for robotics, cotraining with demonstration data generated both in simulation and on real hardware has emerged as a powerful recipe to overcome the sim2real gap. This work seeks to elucidate basic principles of this sim-and-real cotraining to help inform simulation design, sim-and-real dataset creation, and policy training. Focusing narrowly on the canonical task of planar pushing from camera inputs enabled us to be thorough in our study. These experiments confirm that cotraining with simulated data \emph{can} dramatically improve performance in real, especially when real data is limited. Performance gains scale with simulated data, but eventually plateau; real-world data increases this performance ceiling. The results also suggest that reducing the domain gap in physics may be more important than visual fidelity for non-prehensile manipulation tasks. Perhaps surprisingly, having some visual domain gap actually helps the cotrained policy -- binary probes reveal that high-performing policies learn to distinguish simulated domains from real. We conclude by investigating this nuance and mechanisms that facilitate positive transfer between sim-and-real. In total, our experiments span over 40 real-world policies (evaluated on 800+ trials) and 200 simulated policies (evaluated on 40,000+ trials). 

---
# Challenges and Paths Towards AI for Software Engineering 

**Authors**: Alex Gu, Naman Jain, Wen-Ding Li, Manish Shetty, Yijia Shao, Ziyang Li, Diyi Yang, Kevin Ellis, Koushik Sen, Armando Solar-Lezama  

**Link**: [PDF](https://arxiv.org/pdf/2503.22625)  

**Abstract**: AI for software engineering has made remarkable progress recently, becoming a notable success within generative AI. Despite this, there are still many challenges that need to be addressed before automated software engineering reaches its full potential. It should be possible to reach high levels of automation where humans can focus on the critical decisions of what to build and how to balance difficult tradeoffs while most routine development effort is automated away. Reaching this level of automation will require substantial research and engineering efforts across academia and industry. In this paper, we aim to discuss progress towards this in a threefold manner. First, we provide a structured taxonomy of concrete tasks in AI for software engineering, emphasizing the many other tasks in software engineering beyond code generation and completion. Second, we outline several key bottlenecks that limit current approaches. Finally, we provide an opinionated list of promising research directions toward making progress on these bottlenecks, hoping to inspire future research in this rapidly maturing field. 

---
# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users 

**Authors**: Antonia Karamolegkou, Malvina Nikandrou, Georgios Pantazopoulos, Danae Sanchez Villegas, Phillip Rust, Ruchira Dhar, Daniel Hershcovich, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2503.22610)  

**Abstract**: This paper explores the effectiveness of Multimodal Large Language models (MLLMs) as assistive technologies for visually impaired individuals. We conduct a user survey to identify adoption patterns and key challenges users face with such technologies. Despite a high adoption rate of these models, our findings highlight concerns related to contextual understanding, cultural sensitivity, and complex scene understanding, particularly for individuals who may rely solely on them for visual interpretation. Informed by these results, we collate five user-centred tasks with image and video inputs, including a novel task on Optical Braille Recognition. Our systematic evaluation of twelve MLLMs reveals that further advancements are necessary to overcome limitations related to cultural context, multilingual support, Braille reading comprehension, assistive object recognition, and hallucinations. This work provides critical insights into the future direction of multimodal AI for accessibility, underscoring the need for more inclusive, robust, and trustworthy visual assistance technologies. 

---
# Generative Latent Neural PDE Solver using Flow Matching 

**Authors**: Zijie Li, Anthony Zhou, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2503.22600)  

**Abstract**: Autoregressive next-step prediction models have become the de-facto standard for building data-driven neural solvers to forecast time-dependent partial differential equations (PDEs). Denoise training that is closely related to diffusion probabilistic model has been shown to enhance the temporal stability of neural solvers, while its stochastic inference mechanism enables ensemble predictions and uncertainty quantification. In principle, such training involves sampling a series of discretized diffusion timesteps during both training and inference, inevitably increasing computational overhead. In addition, most diffusion models apply isotropic Gaussian noise on structured, uniform grids, limiting their adaptability to irregular domains. We propose a latent diffusion model for PDE simulation that embeds the PDE state in a lower-dimensional latent space, which significantly reduces computational costs. Our framework uses an autoencoder to map different types of meshes onto a unified structured latent grid, capturing complex geometries. By analyzing common diffusion paths, we propose to use a coarsely sampled noise schedule from flow matching for both training and testing. Numerical experiments show that the proposed model outperforms several deterministic baselines in both accuracy and long-term stability, highlighting the potential of diffusion-based approaches for robust data-driven PDE learning. 

---
# KEVS: Enhancing Segmentation of Visceral Adipose Tissue in Pre-Cystectomy CT with Gaussian Kernel Density Estimation 

**Authors**: Thomas Boucher, Nicholas Tetlow, Annie Fung, Amy Dewar, Pietro Arina, Sven Kerneis, John Whittle, Evangelos B. Mazomenos  

**Link**: [PDF](https://arxiv.org/pdf/2503.22592)  

**Abstract**: Purpose: The distribution of visceral adipose tissue (VAT) in cystectomy patients is indicative of the incidence of post-operative complications. Existing VAT segmentation methods for computed tomography (CT) employing intensity thresholding have limitations relating to inter-observer variability. Moreover, the difficulty in creating ground-truth masks limits the development of deep learning (DL) models for this task. This paper introduces a novel method for VAT prediction in pre-cystectomy CT, which is fully automated and does not require ground-truth VAT masks for training, overcoming aforementioned limitations. Methods: We introduce the Kernel density Enhanced VAT Segmentator ( KEVS), combining a DL semantic segmentation model, for multi-body feature prediction, with Gaussian kernel density estimation analysis of predicted subcutaneous adipose tissue to achieve accurate scan-specific predictions of VAT in the abdominal cavity. Uniquely for a DL pipeline, KEVS does not require ground-truth VAT masks. Results: We verify the ability of KEVS to accurately segment abdominal organs in unseen CT data and compare KEVS VAT segmentation predictions to existing state-of-the-art (SOTA) approaches in a dataset of 20 pre-cystectomy CT scans, collected from University College London Hospital (UCLH-Cyst), with expert ground-truth annotations. KEVS presents a 4.80% and 6.02% improvement in Dice Coefficient over the second best DL and thresholding-based VAT segmentation techniques respectively when evaluated on UCLH-Cyst. Conclusion: This research introduces KEVS; an automated, SOTA method for the prediction of VAT in pre-cystectomy CT which eliminates inter-observer variability and is trained entirely on open-source CT datasets which do not contain ground-truth VAT masks. 

---
# Using AI to Summarize US Presidential Campaign TV Advertisement Videos, 1952-2012 

**Authors**: Adam Breuer, Bryce J. Dietrich, Michael H. Crespin, Matthew Butler, J.A. Pyrse, Kosuke Imai  

**Link**: [PDF](https://arxiv.org/pdf/2503.22589)  

**Abstract**: This paper introduces the largest and most comprehensive dataset of US presidential campaign television advertisements, available in digital format. The dataset also includes machine-searchable transcripts and high-quality summaries designed to facilitate a variety of academic research. To date, there has been great interest in collecting and analyzing US presidential campaign advertisements, but the need for manual procurement and annotation led many to rely on smaller subsets. We design a large-scale parallelized, AI-based analysis pipeline that automates the laborious process of preparing, transcribing, and summarizing videos. We then apply this methodology to the 9,707 presidential ads from the Julian P. Kanter Political Commercial Archive. We conduct extensive human evaluations to show that these transcripts and summaries match the quality of manually generated alternatives. We illustrate the value of this data by including an application that tracks the genesis and evolution of current focal issue areas over seven decades of presidential elections. Our analysis pipeline and codebase also show how to use LLM-based tools to obtain high-quality summaries for other video datasets. 

---
# Historical Ink: Exploring Large Language Models for Irony Detection in 19th-Century Spanish 

**Authors**: Kevin Cohen, Laura Manrique-Gómez, Rubén Manrique  

**Link**: [PDF](https://arxiv.org/pdf/2503.22585)  

**Abstract**: This study explores the use of large language models (LLMs) to enhance datasets and improve irony detection in 19th-century Latin American newspapers. Two strategies were employed to evaluate the efficacy of BERT and GPT-4o models in capturing the subtle nuances nature of irony, through both multi-class and binary classification tasks. First, we implemented dataset enhancements focused on enriching emotional and contextual cues; however, these showed limited impact on historical language analysis. The second strategy, a semi-automated annotation process, effectively addressed class imbalance and augmented the dataset with high-quality annotations. Despite the challenges posed by the complexity of irony, this work contributes to the advancement of sentiment analysis through two key contributions: introducing a new historical Spanish dataset tagged for sentiment analysis and irony detection, and proposing a semi-automated annotation methodology where human expertise is crucial for refining LLMs results, enriched by incorporating historical and cultural contexts as core features. 

---
# Breaking Language Barriers in Visual Language Models via Multilingual Textual Regularization 

**Authors**: Iñigo Pikabea, Iñaki Lacunza, Oriol Pareras, Carlos Escolano, Aitor Gonzalez-Agirre, Javier Hernando, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2503.22577)  

**Abstract**: Rapid advancements in Visual Language Models (VLMs) have transformed multimodal understanding but are often constrained by generating English responses regardless of the input language. This phenomenon has been termed as Image-induced Fidelity Loss (IFL) and stems from limited multimodal multilingual training data. To address this, we propose a continuous multilingual integration strategy that injects text-only multilingual data during visual instruction tuning, preserving the language model's original multilingual capabilities. Extensive evaluations demonstrate that our approach significantly improves linguistic fidelity across languages without degradation in visual performance. We also explore model merging, which improves language fidelity but comes at the cost of visual performance. In contrast, our core method achieves robust multilingual alignment without trade-offs, offering a scalable and effective path to mitigating IFL for global VLM adoption. 

---
# On the Mistaken Assumption of Interchangeable Deep Reinforcement Learning Implementations 

**Authors**: Rajdeep Singh Hundal, Yan Xiao, Xiaochun Cao, Jin Song Dong, Manuel Rigger  

**Link**: [PDF](https://arxiv.org/pdf/2503.22575)  

**Abstract**: Deep Reinforcement Learning (DRL) is a paradigm of artificial intelligence where an agent uses a neural network to learn which actions to take in a given environment. DRL has recently gained traction from being able to solve complex environments like driving simulators, 3D robotic control, and multiplayer-online-battle-arena video games. Numerous implementations of the state-of-the-art algorithms responsible for training these agents, like the Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, currently exist. However, studies make the mistake of assuming implementations of the same algorithm to be consistent and thus, interchangeable. In this paper, through a differential testing lens, we present the results of studying the extent of implementation inconsistencies, their effect on the implementations' performance, as well as their impact on the conclusions of prior studies under the assumption of interchangeable implementations. The outcomes of our differential tests showed significant discrepancies between the tested algorithm implementations, indicating that they are not interchangeable. In particular, out of the five PPO implementations tested on 56 games, three implementations achieved superhuman performance for 50% of their total trials while the other two implementations only achieved superhuman performance for less than 15% of their total trials. As part of a meticulous manual analysis of the implementations' source code, we analyzed implementation discrepancies and determined that code-level inconsistencies primarily caused these discrepancies. Lastly, we replicated a study and showed that this assumption of implementation interchangeability was sufficient to flip experiment outcomes. Therefore, this calls for a shift in how implementations are being used. 

---
# A Framework for Cryptographic Verifiability of End-to-End AI Pipelines 

**Authors**: Kar Balan, Robert Learney, Tim Wood  

**Link**: [PDF](https://arxiv.org/pdf/2503.22573)  

**Abstract**: The increasing integration of Artificial Intelligence across multiple industry sectors necessitates robust mechanisms for ensuring transparency, trust, and auditability of its development and deployment. This topic is particularly important in light of recent calls in various jurisdictions to introduce regulation and legislation on AI safety. In this paper, we propose a framework for complete verifiable AI pipelines, identifying key components and analyzing existing cryptographic approaches that contribute to verifiability across different stages of the AI lifecycle, from data sourcing to training, inference, and unlearning. This framework could be used to combat misinformation by providing cryptographic proofs alongside AI-generated assets to allow downstream verification of their provenance and correctness. Our findings underscore the importance of ongoing research to develop cryptographic tools that are not only efficient for isolated AI processes, but that are efficiently `linkable' across different processes within the AI pipeline, to support the development of end-to-end verifiable AI technologies. 

---
# Niyama : Breaking the Silos of LLM Inference Serving 

**Authors**: Kanishk Goel, Jayashree Mohan, Nipun Kwatra, Ravi Shreyas Anupindi, Ramachandran Ramjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.22562)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has enabled diverse applications with very different latency requirements. Existing LLM serving frameworks rely on siloed infrastructure with coarse-grained workload segregation -- interactive and batch -- leading to inefficient resource utilization and limited support for fine-grained Quality-of-Service (QoS) differentiation. This results in operational inefficiencies, over-provisioning and poor load management during traffic surges.
We present Niyama, a novel QoS-driven inference serving system that enables efficient co-scheduling of diverse workloads on shared infrastructure. Niyama introduces fine-grained QoS classification allowing applications to specify precise latency requirements, and dynamically adapts scheduling decisions based on real-time system state. Leveraging the predictable execution characteristics of LLM inference, Niyama implements a dynamic chunking mechanism to improve overall throughput while maintaining strict QoS guarantees. Additionally, Niyama employs a hybrid prioritization policy that balances fairness and efficiency, and employs selective request relegation that enables graceful service degradation during overload conditions. Our evaluation demonstrates that Niyama increases serving capacity by 32% compared to current siloed deployments, while maintaining QoS guarantees. Notably, under extreme load, our system reduces SLO violations by an order of magnitude compared to current strategies. 

---
# SafeCast: Risk-Responsive Motion Forecasting for Autonomous Vehicles 

**Authors**: Haicheng Liao, Hanlin Kong, Bin Rao, Bonan Wang, Chengyue Wang, Guyang Yu, Yuming Huang, Ruru Tang, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22541)  

**Abstract**: Accurate motion forecasting is essential for the safety and reliability of autonomous driving (AD) systems. While existing methods have made significant progress, they often overlook explicit safety constraints and struggle to capture the complex interactions among traffic agents, environmental factors, and motion dynamics. To address these challenges, we present SafeCast, a risk-responsive motion forecasting model that integrates safety-aware decision-making with uncertainty-aware adaptability. SafeCast is the first to incorporate the Responsibility-Sensitive Safety (RSS) framework into motion forecasting, encoding interpretable safety rules--such as safe distances and collision avoidance--based on traffic norms and physical principles. To further enhance robustness, we introduce the Graph Uncertainty Feature (GUF), a graph-based module that injects learnable noise into Graph Attention Networks, capturing real-world uncertainties and enhancing generalization across diverse scenarios. We evaluate SafeCast on four real-world benchmark datasets--Next Generation Simulation (NGSIM), Highway Drone (HighD), ApolloScape, and the Macao Connected Autonomous Driving (MoCAD)--covering highway, urban, and mixed-autonomy traffic environments. Our model achieves state-of-the-art (SOTA) accuracy while maintaining a lightweight architecture and low inference latency, underscoring its potential for real-time deployment in safety-critical AD systems. 

---
# LIM: Large Interpolator Model for Dynamic Reconstruction 

**Authors**: Remy Sabathier, Niloy J. Mitra, David Novotny  

**Link**: [PDF](https://arxiv.org/pdf/2503.22537)  

**Abstract**: Reconstructing dynamic assets from video data is central to many in computer vision and graphics tasks. Existing 4D reconstruction approaches are limited by category-specific models or slow optimization-based methods. Inspired by the recent Large Reconstruction Model (LRM), we present the Large Interpolation Model (LIM), a transformer-based feed-forward solution, guided by a novel causal consistency loss, for interpolating implicit 3D representations across time. Given implicit 3D representations at times $t_0$ and $t_1$, LIM produces a deformed shape at any continuous time $t\in[t_0,t_1]$, delivering high-quality interpolated frames in seconds. Furthermore, LIM allows explicit mesh tracking across time, producing a consistently uv-textured mesh sequence ready for integration into existing production pipelines. We also use LIM, in conjunction with a diffusion-based multiview generator, to produce dynamic 4D reconstructions from monocular videos. We evaluate LIM on various dynamic datasets, benchmarking against image-space interpolation methods (e.g., FiLM) and direct triplane linear interpolation, and demonstrate clear advantages. In summary, LIM is the first feed-forward model capable of high-speed tracked 4D asset reconstruction across diverse categories. 

---
# AnnoPage Dataset: Dataset of Non-Textual Elements in Documents with Fine-Grained Categorization 

**Authors**: Martin Kišš, Michal Hradiš, Martina Dvořáková, Václav Jiroušek, Filip Kersch  

**Link**: [PDF](https://arxiv.org/pdf/2503.22526)  

**Abstract**: We introduce the AnnoPage Dataset, a novel collection of 7550 pages from historical documents, primarily in Czech and German, spanning from 1485 to the present, focusing on the late 19th and early 20th centuries. The dataset is designed to support research in document layout analysis and object detection. Each page is annotated with axis-aligned bounding boxes (AABB) representing elements of 25 categories of non-textual elements, such as images, maps, decorative elements, or charts, following the Czech Methodology of image document processing. The annotations were created by expert librarians to ensure accuracy and consistency. The dataset also incorporates pages from multiple, mainly historical, document datasets to enhance variability and maintain continuity. The dataset is divided into development and test subsets, with the test set carefully selected to maintain the category distribution. We provide baseline results using YOLO and DETR object detectors, offering a reference point for future research. The AnnoPage Dataset is publicly available on Zenodo (this https URL), along with ground-truth annotations in YOLO format. 

---
# Robust Offline Imitation Learning Through State-level Trajectory Stitching 

**Authors**: Shuze Wang, Yunpeng Mei, Hongjie Cao, Yetian Yuan, Gang Wang, Jian Sun, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22524)  

**Abstract**: Imitation learning (IL) has proven effective for enabling robots to acquire visuomotor skills through expert demonstrations. However, traditional IL methods are limited by their reliance on high-quality, often scarce, expert data, and suffer from covariate shift. To address these challenges, recent advances in offline IL have incorporated suboptimal, unlabeled datasets into the training. In this paper, we propose a novel approach to enhance policy learning from mixed-quality offline datasets by leveraging task-relevant trajectory fragments and rich environmental dynamics. Specifically, we introduce a state-based search framework that stitches state-action pairs from imperfect demonstrations, generating more diverse and informative training trajectories. Experimental results on standard IL benchmarks and real-world robotic tasks showcase that our proposed method significantly improves both generalization and performance. 

---
# Exploiting Mixture-of-Experts Redundancy Unlocks Multimodal Generative Abilities 

**Authors**: Raman Dutt, Harleen Hanspal, Guoxuan Xia, Petru-Daniel Tudosiu, Alexander Black, Yongxin Yang, Steven McDonagh, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2503.22517)  

**Abstract**: In this work, we undertake the challenge of augmenting the existing generative capabilities of pre-trained text-only large language models (LLMs) with multi-modal generation capability while satisfying two core constraints: C1 preserving the preservation of original language generative capabilities with negligible performance degradation, and C2 adhering to a small parameter budget to learn the new modality, ensuring scalability and efficiency. In contrast to current approaches that add dedicated modules, thereby significantly increasing the parameter count, we propose a method that leverages the underutilized capacity inherent in deep models. Specifically, we exploit the parameter redundancy within Mixture-of-Experts (MoEs) as a source of additional capacity for learning a new modality, enabling better parameter efficiency (C1). Moreover, we preserve the original language generation capabilities by applying low-rank adaptation exclusively to the tokens of the new modality (C2). Furthermore, we introduce a novel parameter initialization scheme based on the Gromov-Wasserstein distance to improve convergence and training stability. Through an extensive analysis of the routing mechanism, we uncover the emergence of modality-specific pathways and decreased redundancy within the experts that can efficiently unlock multi-modal generative capabilities. Overall, our method can be seamlessly applied to a wide range of contemporary LLMs, providing a new pathway for transitioning from uni-modal to multi-modal architectures. 

---
# Masked Self-Supervised Pre-Training for Text Recognition Transformers on Large-Scale Datasets 

**Authors**: Martin Kišš, Michal Hradiš  

**Link**: [PDF](https://arxiv.org/pdf/2503.22513)  

**Abstract**: Self-supervised learning has emerged as a powerful approach for leveraging large-scale unlabeled data to improve model performance in various domains. In this paper, we explore masked self-supervised pre-training for text recognition transformers. Specifically, we propose two modifications to the pre-training phase: progressively increasing the masking probability, and modifying the loss function to incorporate both masked and non-masked patches. We conduct extensive experiments using a dataset of 50M unlabeled text lines for pre-training and four differently sized annotated datasets for fine-tuning. Furthermore, we compare our pre-trained models against those trained with transfer learning, demonstrating the effectiveness of the self-supervised pre-training. In particular, pre-training consistently improves the character error rate of models, in some cases up to 30 % relatively. It is also on par with transfer learning but without relying on extra annotated text lines. 

---
# Almost Bayesian: The Fractal Dynamics of Stochastic Gradient Descent 

**Authors**: Max Hennick, Stijn De Baerdemacker  

**Link**: [PDF](https://arxiv.org/pdf/2503.22478)  

**Abstract**: We show that the behavior of stochastic gradient descent is related to Bayesian statistics by showing that SGD is effectively diffusion on a fractal landscape, where the fractal dimension can be accounted for in a purely Bayesian way. By doing this we show that SGD can be regarded as a modified Bayesian sampler which accounts for accessibility constraints induced by the fractal structure of the loss landscape. We verify our results experimentally by examining the diffusion of weights during training. These results offer insight into the factors which determine the learning process, and seemingly answer the question of how SGD and purely Bayesian sampling are related. 

---
# Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey 

**Authors**: Shengyue Guan, Haoyi Xiong, Jindong Wang, Jiang Bian, Bin Zhu, Jian-guang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22458)  

**Abstract**: This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the evaluation methodologies. It categorizes approaches into annotation-based evaluations, automated metrics, hybrid strategies that combine human assessments with quantitative measures, and self-judging methods utilizing LLMs. This framework not only captures traditional metrics derived from language understanding, such as BLEU and ROUGE scores, but also incorporates advanced techniques that reflect the dynamic, interactive nature of multi-turn dialogues. 

---
# Entropy-guided sequence weighting for efficient exploration in RL-based LLM fine-tuning 

**Authors**: Abdullah Vanlioglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22456)  

**Abstract**: We introduce Entropy-Guided Sequence Weighting (EGSW), a novel approach that enhances the exploration-exploitation tradeoff by dynamically assigning weights to generated outputs based on their advantage and entropy for Reinforcement Learning-based Large Language Model fine-tuning. EGSW integrates entropy regularization with advantage-based weighting to balance policy updates, enabling efficient exploration in high-dimensional state spaces. By employing temperature-scaled softmax weighting over sequences, EGSW prioritizing high-reward, high-uncertainty steps while maintaining training stability. Although originally developed to improve Group Relative Policy Optimization (GRPO) during large language model (LLM) fine-tuning, EGSW is generalizable to other reinforcement learning (RL) algorithms and can be implemented in both step-wise and trajectory-wise settings. Empirical evaluations demonstrate that EGSW enhances GRPO reasoning ability, yielding improvements in sample efficiency. Future work will explore the application of EGSW to advanced RL methodologies. 

---
# A Causal Framework to Measure and Mitigate Non-binary Treatment Discrimination 

**Authors**: Ayan Majumdar, Deborah D. Kanubala, Kavya Gupta, Isabel Valera  

**Link**: [PDF](https://arxiv.org/pdf/2503.22454)  

**Abstract**: Fairness studies of algorithmic decision-making systems often simplify complex decision processes, such as bail or loan approvals, into binary classification tasks. However, these approaches overlook that such decisions are not inherently binary (e.g., approve or not approve bail or loan); they also involve non-binary treatment decisions (e.g., bail conditions or loan terms) that can influence the downstream outcomes (e.g., loan repayment or reoffending). In this paper, we argue that non-binary treatment decisions are integral to the decision process and controlled by decision-makers and, therefore, should be central to fairness analyses in algorithmic decision-making. We propose a causal framework that extends fairness analyses and explicitly distinguishes between decision-subjects' covariates and the treatment decisions. This specification allows decision-makers to use our framework to (i) measure treatment disparity and its downstream effects in historical data and, using counterfactual reasoning, (ii) mitigate the impact of past unfair treatment decisions when automating decision-making. We use our framework to empirically analyze four widely used loan approval datasets to reveal potential disparity in non-binary treatment decisions and their discriminatory impact on outcomes, highlighting the need to incorporate treatment decisions in fairness assessments. Moreover, by intervening in treatment decisions, we show that our framework effectively mitigates treatment discrimination from historical data to ensure fair risk score estimation and (non-binary) decision-making processes that benefit all stakeholders. 

---
# CoSIL: Software Issue Localization via LLM-Driven Code Repository Graph Searching 

**Authors**: Zhonghao Jiang, Xiaoxue Ren, Meng Yan, Wei Jiang, Yong Li, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22424)  

**Abstract**: Large language models (LLMs) have significantly advanced autonomous software engineering, leading to a growing number of software engineering agents that assist developers in automatic program repair. Issue localization forms the basis for accurate patch generation. However, because of limitations caused by the context window length of LLMs, existing issue localization methods face challenges in balancing concise yet effective contexts and adequately comprehensive search spaces. In this paper, we introduce CoSIL, an LLM driven, simple yet powerful function level issue localization method without training or indexing. CoSIL reduces the search space through module call graphs, iteratively searches the function call graph to obtain relevant contexts, and uses context pruning to control the search direction and manage contexts effectively. Importantly, the call graph is dynamically constructed by the LLM during search, eliminating the need for pre-parsing. Experiment results demonstrate that CoSIL achieves a Top-1 localization success rate of 43 percent and 44.6 percent on SWE bench Lite and SWE bench Verified, respectively, using Qwen2.5 Coder 32B, outperforming existing methods by 8.6 to 98.2 percent. When CoSIL is applied to guide the patch generation stage, the resolved rate further improves by 9.3 to 31.5 percent. 

---
# Training Large Language Models for Advanced Typosquatting Detection 

**Authors**: Jackson Welch  

**Link**: [PDF](https://arxiv.org/pdf/2503.22406)  

**Abstract**: Typosquatting is a long-standing cyber threat that exploits human error in typing URLs to deceive users, distribute malware, and conduct phishing attacks. With the proliferation of domain names and new Top-Level Domains (TLDs), typosquatting techniques have grown more sophisticated, posing significant risks to individuals, businesses, and national cybersecurity infrastructure. Traditional detection methods primarily focus on well-known impersonation patterns, leaving gaps in identifying more complex attacks. This study introduces a novel approach leveraging large language models (LLMs) to enhance typosquatting detection. By training an LLM on character-level transformations and pattern-based heuristics rather than domain-specific data, a more adaptable and resilient detection mechanism develops. Experimental results indicate that the Phi-4 14B model outperformed other tested models when properly fine tuned achieving a 98% accuracy rate with only a few thousand training samples. This research highlights the potential of LLMs in cybersecurity applications, specifically in mitigating domain-based deception tactics, and provides insights into optimizing machine learning strategies for threat detection. 

---
# EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing 

**Authors**: Yizhang Zhu, Runzhi Jiang, Boyan Li, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.22402)  

**Abstract**: Text-to-SQL automatically translates natural language queries to SQL, allowing non-technical users to retrieve data from databases without specialized SQL knowledge. Despite the success of advanced LLM-based Text-to-SQL approaches on leaderboards, their unsustainable computational costs--often overlooked--stand as the "elephant in the room" in current leaderboard-driven research, limiting their economic practicability for real-world deployment and widespread adoption. To tackle this, we exploratively propose EllieSQL, a complexity-aware routing framework that assigns queries to suitable SQL generation pipelines based on estimated complexity. We investigate multiple routers to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP) metric, capturing cost-efficiency by quantifying the responsiveness of performance gains relative to token investment in SQL generation. Experiments show that compared to always using the most advanced methods in our study, EllieSQL with the Qwen2.5-0.5B-DPO router reduces token use by over 40% without compromising performance on Bird development set, achieving more than a 2x boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient Text-to-SQL but also invites the community to weigh resource efficiency alongside performance, contributing to progress in sustainable Text-to-SQL. 

---
# On-site estimation of battery electrochemical parameters via transfer learning based physics-informed neural network approach 

**Authors**: Josu Yeregui, Iker Lopetegi, Sergio Fernandez, Erik Garayalde, Unai Iraola  

**Link**: [PDF](https://arxiv.org/pdf/2503.22396)  

**Abstract**: This paper presents a novel physical parameter estimation framework for on-site model characterization, using a two-phase modelling strategy with Physics-Informed Neural Networks (PINNs) and transfer learning (TL). In the first phase, a PINN is trained using only the physical principles of the single particle model (SPM) equations. In the second phase, the majority of the PINN parameters are frozen, while critical electrochemical parameters are set as trainable and adjusted using real-world voltage profile data. The proposed approach significantly reduces computational costs, making it suitable for real-time implementation on Battery Management Systems (BMS). Additionally, as the initial phase does not require field data, the model is easy to deploy with minimal setup requirements. With the proposed methodology, we have been able to effectively estimate relevant electrochemical parameters with operating data. This has been proved estimating diffusivities and active material volume fractions with charge data in different degradation conditions. The methodology is experimentally validated in a Raspberry Pi device using data from a standard charge profile with a 3.89\% relative accuracy estimating the active material volume fractions of a NMC cell with 82.09\% of its nominal capacity. 

---
# Endo-TTAP: Robust Endoscopic Tissue Tracking via Multi-Facet Guided Attention and Hybrid Flow-point Supervision 

**Authors**: Rulin Zhou, Wenlong He, An Wang, Qiqi Yao, Haijun Hu, Jiankun Wang, Xi Zhang an Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.22394)  

**Abstract**: Accurate tissue point tracking in endoscopic videos is critical for robotic-assisted surgical navigation and scene understanding, but remains challenging due to complex deformations, instrument occlusion, and the scarcity of dense trajectory annotations. Existing methods struggle with long-term tracking under these conditions due to limited feature utilization and annotation dependence. We present Endo-TTAP, a novel framework addressing these challenges through: (1) A Multi-Facet Guided Attention (MFGA) module that synergizes multi-scale flow dynamics, DINOv2 semantic embeddings, and explicit motion patterns to jointly predict point positions with uncertainty and occlusion awareness; (2) A two-stage curriculum learning strategy employing an Auxiliary Curriculum Adapter (ACA) for progressive initialization and hybrid supervision. Stage I utilizes synthetic data with optical flow ground truth for uncertainty-occlusion regularization, while Stage II combines unsupervised flow consistency and semi-supervised learning with refined pseudo-labels from off-the-shelf trackers. Extensive validation on two MICCAI Challenge datasets and our collected dataset demonstrates that Endo-TTAP achieves state-of-the-art performance in tissue point tracking, particularly in scenarios characterized by complex endoscopic conditions. The source code and dataset will be available at this https URL. 

---
# ViSketch-GPT: Collaborative Multi-Scale Feature Extraction for Sketch Recognition and Generation 

**Authors**: Giulio Federico, Giuseppe Amato, Fabio Carrara, Claudio Gennaro, Marco Di Benedetto  

**Link**: [PDF](https://arxiv.org/pdf/2503.22374)  

**Abstract**: Understanding the nature of human sketches is challenging because of the wide variation in how they are created. Recognizing complex structural patterns improves both the accuracy in recognizing sketches and the fidelity of the generated sketches. In this work, we introduce ViSketch-GPT, a novel algorithm designed to address these challenges through a multi-scale context extraction approach. The model captures intricate details at multiple scales and combines them using an ensemble-like mechanism, where the extracted features work collaboratively to enhance the recognition and generation of key details crucial for classification and generation tasks.
The effectiveness of ViSketch-GPT is validated through extensive experiments on the QuickDraw dataset. Our model establishes a new benchmark, significantly outperforming existing methods in both classification and generation tasks, with substantial improvements in accuracy and the fidelity of generated sketches.
The proposed algorithm offers a robust framework for understanding complex structures by extracting features that collaborate to recognize intricate details, enhancing the understanding of structures like sketches and making it a versatile tool for various applications in computer vision and machine learning. 

---
# ForcePose: A Deep Learning Approach for Force Calculation Based on Action Recognition Using MediaPipe Pose Estimation Combined with Object Detection 

**Authors**: Nandakishor M, Vrinda Govind V, Anuradha Puthalath, Anzy L, Swathi P S, Aswathi R, Devaprabha A R, Varsha Raj, Midhuna Krishnan K, Akhila Anilkumar T V, Yamuna P V  

**Link**: [PDF](https://arxiv.org/pdf/2503.22363)  

**Abstract**: Force estimation in human-object interactions is crucial for various fields like ergonomics, physical therapy, and sports science. Traditional methods depend on specialized equipment such as force plates and sensors, which makes accurate assessments both expensive and restricted to laboratory settings. In this paper, we introduce ForcePose, a novel deep learning framework that estimates applied forces by combining human pose estimation with object detection. Our approach leverages MediaPipe for skeletal tracking and SSD MobileNet for object recognition to create a unified representation of human-object interaction. We've developed a specialized neural network that processes both spatial and temporal features to predict force magnitude and direction without needing any physical sensors. After training on our dataset of 850 annotated videos with corresponding force measurements, our model achieves a mean absolute error of 5.83 N in force magnitude and 7.4 degrees in force direction. When compared to existing computer vision approaches, our method performs 27.5% better while still offering real-time performance on standard computing hardware. ForcePose opens up new possibilities for force analysis in diverse real-world scenarios where traditional measurement tools are impractical or intrusive. This paper discusses our methodology, the dataset creation process, evaluation metrics, and potential applications across rehabilitation, ergonomics assessment, and athletic performance analysis. 

---
# Shapley Revisited: Tractable Responsibility Measures for Query Answers 

**Authors**: Meghyn Bienvenu, Diego Figueira, Pierre Lafourcade  

**Link**: [PDF](https://arxiv.org/pdf/2503.22358)  

**Abstract**: The Shapley value, originating from cooperative game theory, has been employed to define responsibility measures that quantify the contributions of database facts to obtaining a given query answer. For non-numeric queries, this is done by considering a cooperative game whose players are the facts and whose wealth function assigns 1 or 0 to each subset of the database, depending on whether the query answer holds in the given subset. While conceptually simple, this approach suffers from a notable drawback: the problem of computing such Shapley values is #P-hard in data complexity, even for simple conjunctive queries. This motivates us to revisit the question of what constitutes a reasonable responsibility measure and to introduce a new family of responsibility measures -- weighted sums of minimal supports (WSMS) -- which satisfy intuitive properties. Interestingly, while the definition of WSMSs is simple and bears no obvious resemblance to the Shapley value formula, we prove that every WSMS measure can be equivalently seen as the Shapley value of a suitably defined cooperative game. Moreover, WSMS measures enjoy tractable data complexity for a large class of queries, including all unions of conjunctive queries. We further explore the combined complexity of WSMS computation and establish (in)tractability results for various subclasses of conjunctive queries. 

---
# Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions 

**Authors**: Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2503.22353)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various tasks, but their deployment in high-stake domains requires consistent performance across multiple interaction rounds. This paper introduces a comprehensive framework for evaluating and improving LLM response consistency, making three key contributions. First, we propose a novel Position-Weighted Consistency (PWC) score that captures both the importance of early-stage stability and recovery patterns in multi-turn interactions. Second, we present a carefully curated benchmark dataset spanning diverse domains and difficulty levels, specifically designed to evaluate LLM consistency under various challenging follow-up scenarios. Third, we introduce Confidence-Aware Response Generation (CARG), a framework that significantly improves response stability by incorporating model confidence signals into the generation process. Empirical results demonstrate that CARG significantly improves response stability without sacrificing accuracy, underscoring its potential for reliable LLM deployment in critical applications. 

---
# VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow 

**Authors**: Yancong Lin, Shiming Wang, Liangliang Nan, Julian Kooij, Holger Caesar  

**Link**: [PDF](https://arxiv.org/pdf/2503.22328)  

**Abstract**: Scene flow estimation aims to recover per-point motion from two adjacent LiDAR scans. However, in real-world applications such as autonomous driving, points rarely move independently of others, especially for nearby points belonging to the same object, which often share the same motion. Incorporating this locally rigid motion constraint has been a key challenge in self-supervised scene flow estimation, which is often addressed by post-processing or appending extra regularization. While these approaches are able to improve the rigidity of predicted flows, they lack an architectural inductive bias for local rigidity within the model structure, leading to suboptimal learning efficiency and inferior performance. In contrast, we enforce local rigidity with a lightweight add-on module in neural network design, enabling end-to-end learning. We design a discretized voting space that accommodates all possible translations and then identify the one shared by nearby points by differentiable voting. Additionally, to ensure computational efficiency, we operate on pillars rather than points and learn representative features for voting per pillar. We plug the Voting Module into popular model designs and evaluate its benefit on Argoverse 2 and Waymo datasets. We outperform baseline works with only marginal compute overhead. Code is available at this https URL. 

---
# AH-GS: Augmented 3D Gaussian Splatting for High-Frequency Detail Representation 

**Authors**: Chenyang Xu, XingGuo Deng, Rui Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22324)  

**Abstract**: The 3D Gaussian Splatting (3D-GS) is a novel method for scene representation and view synthesis. Although Scaffold-GS achieves higher quality real-time rendering compared to the original 3D-GS, its fine-grained rendering of the scene is extremely dependent on adequate viewing angles. The spectral bias of neural network learning results in Scaffold-GS's poor ability to perceive and learn high-frequency information in the scene. In this work, we propose enhancing the manifold complexity of input features and using network-based feature map loss to improve the image reconstruction quality of 3D-GS models. We introduce AH-GS, which enables 3D Gaussians in structurally complex regions to obtain higher-frequency encodings, allowing the model to more effectively learn the high-frequency information of the scene. Additionally, we incorporate high-frequency reinforce loss to further enhance the model's ability to capture detailed frequency information. Our result demonstrates that our model significantly improves rendering fidelity, and in specific scenarios (e.g., MipNeRf360-garden), our method exceeds the rendering quality of Scaffold-GS in just 15K iterations. 

---
# Machine Learning Models for Soil Parameter Prediction Based on Satellite, Weather, Clay and Yield Data 

**Authors**: Calvin Kammerlander, Viola Kolb, Marinus Luegmair, Lou Scheermann, Maximilian Schmailzl, Marco Seufert, Jiayun Zhang, Denis Dalic, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2503.22276)  

**Abstract**: Efficient nutrient management and precise fertilization are essential for advancing modern agriculture, particularly in regions striving to optimize crop yields sustainably. The AgroLens project endeavors to address this challenge by develop ing Machine Learning (ML)-based methodologies to predict soil nutrient levels without reliance on laboratory tests. By leveraging state of the art techniques, the project lays a foundation for acionable insights to improve agricultural productivity in resource-constrained areas, such as Africa. The approach begins with the development of a robust European model using the LUCAS Soil dataset and Sentinel-2 satellite imagery to estimate key soil properties, including phosphorus, potassium, nitrogen, and pH levels. This model is then enhanced by integrating supplementary features, such as weather data, harvest rates, and Clay AI-generated embeddings. This report details the methodological framework, data preprocessing strategies, and ML pipelines employed in this project. Advanced algorithms, including Random Forests, Extreme Gradient Boosting (XGBoost), and Fully Connected Neural Networks (FCNN), were implemented and finetuned for precise nutrient prediction. Results showcase robust model performance, with root mean square error values meeting stringent accuracy thresholds. By establishing a reproducible and scalable pipeline for soil nutrient prediction, this research paves the way for transformative agricultural applications, including precision fertilization and improved resource allocation in underresourced regions like Africa. 

---
# Make Some Noise: Towards LLM audio reasoning and generation using sound tokens 

**Authors**: Shivam Mehta, Nebojsa Jojic, Hannes Gamper  

**Link**: [PDF](https://arxiv.org/pdf/2503.22275)  

**Abstract**: Integrating audio comprehension and generation into large language models (LLMs) remains challenging due to the continuous nature of audio and the resulting high sampling rates. Here, we introduce a novel approach that combines Variational Quantization with Conditional Flow Matching to convert audio into ultra-low bitrate discrete tokens of 0.23kpbs, allowing for seamless integration with text tokens in LLMs. We fine-tuned a pretrained text-based LLM using Low-Rank Adaptation (LoRA) to assess its effectiveness in achieving true multimodal capabilities, i.e., audio comprehension and generation. Our tokenizer outperforms a traditional VQ-VAE across various datasets with diverse acoustic events. Despite the substantial loss of fine-grained details through audio tokenization, our multimodal LLM trained with discrete tokens achieves competitive results in audio comprehension with state-of-the-art methods, though audio generation is poor. Our results highlight the need for larger, more diverse datasets and improved evaluation metrics to advance multimodal LLM performance. 

---
# Beyond the Script: Testing LLMs for Authentic Patient Communication Styles in Healthcare 

**Authors**: Anna Bodonhelyi, Christian Stegemann-Philipps, Alessandra Sonanini, Lea Herschbach, Márton Szép, Anne Herrmann-Werner, Teresa Festl-Wietek, Enkelejda Kasneci, Friederike Holderried  

**Link**: [PDF](https://arxiv.org/pdf/2503.22250)  

**Abstract**: Effective patient communication is pivotal in healthcare, yet traditional medical training often lacks exposure to diverse, challenging interpersonal dynamics. To bridge this gap, this study proposes the use of Large Language Models (LLMs) to simulate authentic patient communication styles, specifically the "accuser" and "rationalizer" personas derived from the Satir model, while also ensuring multilingual applicability to accommodate diverse cultural contexts and enhance accessibility for medical professionals. Leveraging advanced prompt engineering, including behavioral prompts, author's notes, and stubbornness mechanisms, we developed virtual patients (VPs) that embody nuanced emotional and conversational traits. Medical professionals evaluated these VPs, rating their authenticity (accuser: $3.8 \pm 1.0$; rationalizer: $3.7 \pm 0.8$ on a 5-point Likert scale (from one to five)) and correctly identifying their styles. Emotion analysis revealed distinct profiles: the accuser exhibited pain, anger, and distress, while the rationalizer displayed contemplation and calmness, aligning with predefined, detailed patient description including medical history. Sentiment scores (on a scale from zero to nine) further validated these differences in the communication styles, with the accuser adopting negative ($3.1 \pm 0.6$) and the rationalizer more neutral ($4.0 \pm 0.4$) tone. These results underscore LLMs' capability to replicate complex communication styles, offering transformative potential for medical education. This approach equips trainees to navigate challenging clinical scenarios by providing realistic, adaptable patient interactions, enhancing empathy and diagnostic acumen. Our findings advocate for AI-driven tools as scalable, cost-effective solutions to cultivate nuanced communication skills, setting a foundation for future innovations in healthcare training. 

---
# WeatherMesh-3: Fast and accurate operational global weather forecasting 

**Authors**: Haoxing Du, Lyna Kim, Joan Creus-Costa, Jack Michaels, Anuj Shetty, Todd Hutchinson, Christopher Riedel, John Dean  

**Link**: [PDF](https://arxiv.org/pdf/2503.22235)  

**Abstract**: We present WeatherMesh-3 (WM-3), an operational transformer-based global weather forecasting system that improves the state of the art in both accuracy and computational efficiency. We introduce the following advances: 1) a latent rollout that enables arbitrary-length predictions in latent space without intermediate encoding or decoding; and 2) a modular architecture that flexibly utilizes mixed-horizon processors and encodes multiple real-time analyses to create blended initial conditions. WM-3 generates 14-day global forecasts at 0.25-degree resolution in 12 seconds on a single RTX 4090. This represents a >100,000-fold speedup over traditional NWP approaches while achieving superior accuracy with up to 37.7% improvement in RMSE over operational models, requiring only a single consumer-grade GPU for deployment. We aim for WM-3 to democratize weather forecasting by providing an accessible, lightweight model for operational use while pushing the performance boundaries of machine learning-based weather prediction. 

---
# Process Reward Modeling with Entropy-Driven Uncertainty 

**Authors**: Lang Cao, Renhong Chen, Yingtian Zou, Chao Peng, Wu Ning, Huacong Xu, Qian Chen, Yuxian Wang, Peishuo Su, Mofan Peng, Zijie Chen, Yitong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22233)  

**Abstract**: This paper presents the Entropy-Driven Unified Process Reward Model (EDU-PRM), a novel framework that approximates state-of-the-art performance in process supervision while drastically reducing training costs. EDU-PRM introduces an entropy-guided dynamic step partitioning mechanism, using logit distribution entropy to pinpoint high-uncertainty regions during token generation dynamically. This self-assessment capability enables precise step-level feedback without manual fine-grained annotation, addressing a critical challenge in process supervision. Experiments on the Qwen2.5-72B model with only 7,500 EDU-PRM-generated training queries demonstrate accuracy closely approximating the full Qwen2.5-72B-PRM (71.1% vs. 71.6%), achieving a 98% reduction in query cost compared to prior methods. This work establishes EDU-PRM as an efficient approach for scalable process reward model training. 

---
# MFH: A Multi-faceted Heuristic Algorithm Selection Approach for Software Verification 

**Authors**: Jie Su, Liansai Deng, Cheng Wen, Rong Wang, Zhi Ma, Nan Zhang, Cong Tian, Zhenhua Duan, Shengchao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22228)  

**Abstract**: Currently, many verification algorithms are available to improve the reliability of software systems. Selecting the appropriate verification algorithm typically demands domain expertise and non-trivial manpower. An automated algorithm selector is thus desired. However, existing selectors, either depend on machine-learned strategies or manually designed heuristics, encounter issues such as reliance on high-quality samples with algorithm labels and limited scalability. In this paper, an automated algorithm selection approach, namely MFH, is proposed for software verification. Our approach leverages the heuristics that verifiers producing correct results typically implement certain appropriate algorithms, and the supported algorithms by these verifiers indirectly reflect which ones are potentially applicable. Specifically, MFH embeds the code property graph (CPG) of a semantic-preserving transformed program to enhance the robustness of the prediction model. Furthermore, our approach decomposes the selection task into the sub-tasks of predicting potentially applicable algorithms and matching the most appropriate verifiers. Additionally, MFH also introduces a feedback loop on incorrect predictions to improve model prediction accuracy. We evaluate MFH on 20 verifiers and over 15,000 verification tasks. Experimental results demonstrate the effectiveness of MFH, achieving a prediction accuracy of 91.47% even without ground truth algorithm labels provided during the training phase. Moreover, the prediction accuracy decreases only by 0.84% when introducing 10 new verifiers, indicating the strong scalability of the proposed approach. 

---
# Learning to Instruct for Visual Instruction Tuning 

**Authors**: Zhihan Zhou, Feng Hong, Jiaan Luo, Jiangchao Yao, Dongsheng Li, Bo Han, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22215)  

**Abstract**: We propose LIT, an advancement of visual instruction tuning (VIT). While VIT equips Multimodal LLMs (MLLMs) with promising multimodal capabilities, the current design choices for VIT often result in overfitting and shortcut learning, potentially degrading performance. This gap arises from an overemphasis on instruction-following abilities, while neglecting the proactive understanding of visual information. Inspired by this, LIT adopts a simple yet effective approach by incorporating the loss function into both the instruction and response sequences. It seamlessly expands the training data, and regularizes the MLLMs from overly relying on language priors. Based on this merit, LIT achieves a significant relative improvement of up to 9% on comprehensive multimodal benchmarks, requiring no additional training data and incurring negligible computational overhead. Surprisingly, LIT attains exceptional fundamental visual capabilities, yielding up to an 18% improvement in captioning performance, while simultaneously alleviating hallucination in MLLMs. 

---
# Sell It Before You Make It: Revolutionizing E-Commerce with Personalized AI-Generated Items 

**Authors**: Jianghao Lin, Peng Du, Jiaqi Liu, Weite Li, Yong Yu, Weinan Zhang, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22182)  

**Abstract**: E-commerce has revolutionized retail, yet its traditional workflows remain inefficient, with significant time and resource costs tied to product design and manufacturing inventory. This paper introduces a novel system deployed at Alibaba that leverages AI-generated items (AIGI) to address these challenges with personalized text-to-image generation for e-commercial product design. AIGI enables an innovative business mode called "sell it before you make it", where merchants can design fashion items and generate photorealistic images with digital models based on textual descriptions. Only when the items have received a certain number of orders, do the merchants start to produce them, which largely reduces reliance on physical prototypes and thus accelerates time to market. For such a promising application, we identify the underlying key scientific challenge, i.e., capturing the users' group-level personalized preferences towards multiple generated candidate images. To this end, we propose a Personalized Group-Level Preference Alignment Framework for Diffusion Models (i.e., PerFusion). We first design PerFusion Reward Model for user preference estimation with a feature-crossing-based personalized plug-in. Then we develop PerFusion with a personalized adaptive network to model diverse preferences across users, and meanwhile derive the group-level preference optimization objective to capture the comparative behaviors among multiple candidates. Both offline and online experiments demonstrate the effectiveness of our proposed algorithm. The AI-generated items have achieved over 13% relative improvements for both click-through rate and conversion rate compared to their human-designed counterparts, validating the revolutionary potential of AI-generated items for e-commercial platforms. 

---
# e-person Architecture and Framework for Human-AI Co-adventure Relationship 

**Authors**: Kanako Esaki, Tadayuki Matsumura, Yang Shao, Hiroyuki Mizuno  

**Link**: [PDF](https://arxiv.org/pdf/2503.22181)  

**Abstract**: This paper proposes the e-person architecture for constructing a unified and incremental development of AI ethics. The e-person architecture takes the reduction of uncertainty through collaborative cognition and action with others as a unified basis for ethics. By classifying and defining uncertainty along two axes - (1) first, second, and third person perspectives, and (2) the difficulty of inference based on the depth of information - we support the development of unified and incremental development of AI ethics. In addition, we propose the e-person framework based on the free energy principle, which considers the reduction of uncertainty as a unifying principle of brain function, with the aim of implementing the e-person architecture, and we show our previous works and future challenges based on the proposed framework. 

---
# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging 

**Authors**: Chanhyuk Lee, Jiho Choi, Chanryeol Lee, Donggyun Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22178)  

**Abstract**: Model merging has emerged as a promising approach for unifying independently fine-tuned models into an integrated framework, significantly enhancing computational efficiency in multi-task learning. Recently, several SVD-based techniques have been introduced to exploit low-rank structures for enhanced merging, but their reliance on such manually designed rank selection often leads to cross-task interference and suboptimal performance. In this paper, we propose AdaRank, a novel model merging framework that adaptively selects the most beneficial singular directions of task vectors to merge multiple models. We empirically show that the dominant singular components of task vectors can cause critical interference with other tasks, and that naive truncation across tasks and layers degrades performance. In contrast, AdaRank dynamically prunes the singular components that cause interference and offers an optimal amount of information to each task vector by learning to prune ranks during test-time via entropy minimization. Our analysis demonstrates that such method mitigates detrimental overlaps among tasks, while empirical results show that AdaRank consistently achieves state-of-the-art performance with various backbones and number of tasks, reducing the performance gap between fine-tuned models to nearly 1%. 

---
# PharmAgents: Building a Virtual Pharma with Large Language Model Agents 

**Authors**: Bowen Gao, Yanwen Huang, Yiqiao Liu, Wenxuan Xie, Wei-Ying Ma, Ya-Qin Zhang, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22164)  

**Abstract**: The discovery of novel small molecule drugs remains a critical scientific challenge with far-reaching implications for treating diseases and advancing human health. Traditional drug development--especially for small molecule therapeutics--is a highly complex, resource-intensive, and time-consuming process that requires multidisciplinary collaboration. Recent breakthroughs in artificial intelligence (AI), particularly the rise of large language models (LLMs), present a transformative opportunity to streamline and accelerate this process. In this paper, we introduce PharmAgents, a virtual pharmaceutical ecosystem driven by LLM-based multi-agent collaboration. PharmAgents simulates the full drug discovery workflow--from target discovery to preclinical evaluation--by integrating explainable, LLM-driven agents equipped with specialized machine learning models and computational tools. Through structured knowledge exchange and automated optimization, PharmAgents identifies potential therapeutic targets, discovers promising lead compounds, enhances binding affinity and key molecular properties, and performs in silico analyses of toxicity and synthetic feasibility. Additionally, the system supports interpretability, agent interaction, and self-evolvement, enabling it to refine future drug designs based on prior experience. By showcasing the potential of LLM-powered multi-agent systems in drug discovery, this work establishes a new paradigm for autonomous, explainable, and scalable pharmaceutical research, with future extensions toward comprehensive drug lifecycle management. 

---
# EgoToM: Benchmarking Theory of Mind Reasoning from Egocentric Videos 

**Authors**: Yuxuan Li, Vijay Veerabadran, Michael L. Iuzzolino, Brett D. Roads, Asli Celikyilmaz, Karl Ridgeway  

**Link**: [PDF](https://arxiv.org/pdf/2503.22152)  

**Abstract**: We introduce EgoToM, a new video question-answering benchmark that extends Theory-of-Mind (ToM) evaluation to egocentric domains. Using a causal ToM model, we generate multi-choice video QA instances for the Ego4D dataset to benchmark the ability to predict a camera wearer's goals, beliefs, and next actions. We study the performance of both humans and state of the art multimodal large language models (MLLMs) on these three interconnected inference problems. Our evaluation shows that MLLMs achieve close to human-level accuracy on inferring goals from egocentric videos. However, MLLMs (including the largest ones we tested with over 100B parameters) fall short of human performance when inferring the camera wearers' in-the-moment belief states and future actions that are most consistent with the unseen video future. We believe that our results will shape the future design of an important class of egocentric digital assistants which are equipped with a reasonable model of the user's internal mental states. 

---
# When Autonomy Breaks: The Hidden Existential Risk of AI 

**Authors**: Joshua Krook  

**Link**: [PDF](https://arxiv.org/pdf/2503.22151)  

**Abstract**: AI risks are typically framed around physical threats to humanity, a loss of control or an accidental error causing humanity's extinction. However, I argue in line with the gradual disempowerment thesis, that there is an underappreciated risk in the slow and irrevocable decline of human autonomy. As AI starts to outcompete humans in various areas of life, a tipping point will be reached where it no longer makes sense to rely on human decision-making, creativity, social care or even leadership.
What may follow is a process of gradual de-skilling, where we lose skills that we currently take for granted. Traditionally, it is argued that AI will gain human skills over time, and that these skills are innate and immutable in humans. By contrast, I argue that humans may lose such skills as critical thinking, decision-making and even social care in an AGI world. The biggest threat to humanity is therefore not that machines will become more like humans, but that humans will become more like machines. 

---
# FRASE: Structured Representations for Generalizable SPARQL Query Generation 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2503.22144)  

**Abstract**: Translating natural language questions into SPARQL queries enables Knowledge Base querying for factual and up-to-date responses. However, existing datasets for this task are predominantly template-based, leading models to learn superficial mappings between question and query templates rather than developing true generalization capabilities. As a result, models struggle when encountering naturally phrased, template-free questions. This paper introduces FRASE (FRAme-based Semantic Enhancement), a novel approach that leverages Frame Semantic Role Labeling (FSRL) to address this limitation. We also present LC-QuAD 3.0, a new dataset derived from LC-QuAD 2.0, in which each question is enriched using FRASE through frame detection and the mapping of frame-elements to their argument. We evaluate the impact of this approach through extensive experiments on recent large language models (LLMs) under different fine-tuning configurations. Our results demonstrate that integrating frame-based structured representations consistently improves SPARQL generation performance, particularly in challenging generalization scenarios when test questions feature unseen templates (unknown template splits) and when they are all naturally phrased (reformulated questions). 

---
# A Self-Supervised Learning of a Foundation Model for Analog Layout Design Automation 

**Authors**: Sungyu Jeong, Won Joon Choi, Junung Choi, Anik Biswas, Byungsub Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.22143)  

**Abstract**: We propose a UNet-based foundation model and its self-supervised learning method to address two key challenges: 1) lack of qualified annotated analog layout data, and 2) excessive variety in analog layout design tasks. For self-supervised learning, we propose random patch sampling and random masking techniques automatically to obtain enough training data from a small unannotated layout dataset. The obtained data are greatly augmented, less biased, equally sized, and contain enough information for excessive varieties of qualified layout patterns. By pre-training with the obtained data, the proposed foundation model can learn implicit general knowledge on layout patterns so that it can be fine-tuned for various downstream layout tasks with small task-specific datasets. Fine-tuning provides an efficient and consolidated methodology for diverse downstream tasks, reducing the enormous human effort to develop a model per task separately. In experiments, the foundation model was pre-trained using 324,000 samples obtained from 6 silicon-proved manually designed analog circuits, then it was fine-tuned for the five example downstream tasks: generating contacts, vias, dummy fingers, N-wells, and metal routings. The fine-tuned models successfully performed these tasks for more than one thousand unseen layout inputs, generating DRC/LVS-clean layouts for 96.6% of samples. Compared with training the model from scratch for the metal routing task, fine-tuning required only 1/8 of the data to achieve the same dice score of 0.95. With the same data, fine-tuning achieved a 90% lower validation loss and a 40% higher benchmark score than training from scratch. 

---
# Integrating Artificial Intelligence with Human Expertise: An In-depth Analysis of ChatGPT's Capabilities in Generating Metamorphic Relations 

**Authors**: Yifan Zhang, Dave Towey, Matthew Pike, Quang-Hung Luu, Huai Liu, Tsong Yueh Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22141)  

**Abstract**: Context: This paper provides an in-depth examination of the generation and evaluation of Metamorphic Relations (MRs) using GPT models developed by OpenAI, with a particular focus on the capabilities of GPT-4 in software testing environments.
Objective: The aim is to examine the quality of MRs produced by GPT-3.5 and GPT-4 for a specific System Under Test (SUT) adopted from an earlier study, and to introduce and apply an improved set of evaluation criteria for a diverse range of SUTs.
Method: The initial phase evaluates MRs generated by GPT-3.5 and GPT-4 using criteria from a prior study, followed by an application of an enhanced evaluation framework on MRs created by GPT-4 for a diverse range of nine SUTs, varying from simple programs to complex systems incorporating AI/ML components. A custom-built GPT evaluator, alongside human evaluators, assessed the MRs, enabling a direct comparison between automated and human evaluation methods.
Results: The study finds that GPT-4 outperforms GPT-3.5 in generating accurate and useful MRs. With the advanced evaluation criteria, GPT-4 demonstrates a significant ability to produce high-quality MRs across a wide range of SUTs, including complex systems incorporating AI/ML components.
Conclusions: GPT-4 exhibits advanced capabilities in generating MRs suitable for various applications. The research underscores the growing potential of AI in software testing, particularly in the generation and evaluation of MRs, and points towards the complementarity of human and AI skills in this domain. 

---
# REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation 

**Authors**: Puzhen Yuan, Angyuan Ma, Yunchao Yao, Huaxiu Yao, Masayoshi Tomizuka, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.22122)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in robotic planning, particularly for long-horizon tasks that require a holistic understanding of the environment for task decomposition. Existing methods typically rely on prior environmental knowledge or carefully designed task-specific prompts, making them struggle with dynamic scene changes or unexpected task conditions, e.g., a robot attempting to put a carrot in the microwave but finds the door was closed. Such challenges underscore two critical issues: adaptability and efficiency. To address them, in this work, we propose an adaptive multi-agent planning framework, termed REMAC, that enables efficient, scene-agnostic multi-robot long-horizon task planning and execution through continuous reflection and self-evolution. REMAC incorporates two key modules: a self-reflection module performing pre-condition and post-condition checks in the loop to evaluate progress and refine plans, and a self-evolvement module dynamically adapting plans based on scene-specific reasoning. It offers several appealing benefits: 1) Robots can initially explore and reason about the environment without complex prompt design. 2) Robots can keep reflecting on potential planning errors and adapting the plan based on task-specific insights. 3) After iterations, a robot can call another one to coordinate tasks in parallel, maximizing the task execution efficiency. To validate REMAC's effectiveness, we build a multi-agent environment for long-horizon robot manipulation and navigation based on RoboCasa, featuring 4 task categories with 27 task styles and 50+ different objects. Based on it, we further benchmark state-of-the-art reasoning models, including DeepSeek-R1, o3-mini, QwQ, and Grok3, demonstrating REMAC's superiority by boosting average success rates by 40% and execution efficiency by 52.7% over the single robot baseline. 

---
# Beyond Single-Sentence Prompts: Upgrading Value Alignment Benchmarks with Dialogues and Stories 

**Authors**: Yazhou Zhang, Qimeng Liu, Qiuchi Li, Peng Zhang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22115)  

**Abstract**: Evaluating the value alignment of large language models (LLMs) has traditionally relied on single-sentence adversarial prompts, which directly probe models with ethically sensitive or controversial questions. However, with the rapid advancements in AI safety techniques, models have become increasingly adept at circumventing these straightforward tests, limiting their effectiveness in revealing underlying biases and ethical stances. To address this limitation, we propose an upgraded value alignment benchmark that moves beyond single-sentence prompts by incorporating multi-turn dialogues and narrative-based scenarios. This approach enhances the stealth and adversarial nature of the evaluation, making it more robust against superficial safeguards implemented in modern LLMs. We design and implement a dataset that includes conversational traps and ethically ambiguous storytelling, systematically assessing LLMs' responses in more nuanced and context-rich settings. Experimental results demonstrate that this enhanced methodology can effectively expose latent biases that remain undetected in traditional single-shot evaluations. Our findings highlight the necessity of contextual and dynamic testing for value alignment in LLMs, paving the way for more sophisticated and realistic assessments of AI ethics and safety. 

---
# How Well Can Vison-Language Models Understand Humans' Intention? An Open-ended Theory of Mind Question Evaluation Benchmark 

**Authors**: Ximing Wen, Mallika Mainali, Anik Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22093)  

**Abstract**: Vision Language Models (VLMs) have demonstrated strong reasoning capabilities in Visual Question Answering (VQA) tasks; However, their ability to perform Theory of Mind (ToM) tasks such as accurately inferring human intentions, beliefs, and other mental states remains underexplored. In this work, we propose an open-ended question framework to comprehensively evaluate VLMs' performance across diverse categories of ToM tasks. We curated and annotated a benchmark dataset composed of 30 images. We then assessed the performance of four VLMs of varying sizes on this dataset. Our experimental results show that the GPT-4 model outperformed all others, with only one smaller model, GPT-4o-mini, achieving comparable performance. Additionally, we observed that VLMs often struggle to accurately infer intentions in complex scenarios such as bullying or cheating. Moreover, our findings also reveal that smaller models can sometimes infer correct intentions despite relying on incorrect visual cues. 

---
# Penrose Tiled Low-Rank Compression and Section-Wise Q&A Fine-Tuning: A General Framework for Domain-Specific Large Language Model Adaptation 

**Authors**: Chuan-Wei Kuo, Siyu Chen, Chenqi Yan, Yu Yang Fredrik Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22074)  

**Abstract**: Large language models (LLMs) hold great promise for specialized scientific domains such as materials science, yet adapting them efficiently and accurately to domain-specific knowledge remains challenging due to limited data and high knowledge density. We propose a two-stage framework that combines structured model compression with a scientific fine-tuning regimen to address this challenge. In the compression stage, we decompose the LLM's weight matrices into local low-rank "rank blocks" and arrange these blocks in a Penrose-like non-periodic tiling pattern. Each block is then compacted via spectral transformations (e.g., discrete cosine or Fourier transforms), and a Kullback-Leibler (KL) divergence-based alignment loss preserves the distributional similarity between the compressed model's representations and those of the original full model. In the adaptation stage, the compressed model is further tuned using a human-like scientific reading protocol: it processes technical materials science documents section by section, engaging in a structured question-and-answer routine for each section. This section-wise Q&A fine-tuning strategy extracts explicit reasoning traces and gradually injects domain knowledge, while minimizing catastrophic forgetting of the model's general language capabilities. By balancing efficient compression with targeted adaptation, our two-stage approach enables precise specialization of LLMs to high-value domains under data-scarce conditions. We present this principled yet exploratory pipeline and outline its potential for advancing materials science knowledge integration, laying the groundwork for comprehensive empirical evaluation in future work. 

---
# Contrasting Low and High-Resolution Features for HER2 Scoring using Deep Learning 

**Authors**: Ekansh Chauhan, Anila Sharma, Amit Sharma, Vikas Nishadham, Asha Ghughtyal, Ankur Kumar, Gurudutt Gupta, Anurag Mehta, C.V. Jawahar, P.K. Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2503.22069)  

**Abstract**: Breast cancer, the most common malignancy among women, requires precise detection and classification for effective treatment. Immunohistochemistry (IHC) biomarkers like HER2, ER, and PR are critical for identifying breast cancer subtypes. However, traditional IHC classification relies on pathologists' expertise, making it labor-intensive and subject to significant inter-observer variability. To address these challenges, this study introduces the India Pathology Breast Cancer Dataset (IPD-Breast), comprising of 1,272 IHC slides (HER2, ER, and PR) aimed at automating receptor status classification. The primary focus is on developing predictive models for HER2 3-way classification (0, Low, High) to enhance prognosis. Evaluation of multiple deep learning models revealed that an end-to-end ConvNeXt network utilizing low-resolution IHC images achieved an AUC, F1, and accuracy of 91.79%, 83.52%, and 83.56%, respectively, for 3-way classification, outperforming patch-based methods by over 5.35% in F1 score. This study highlights the potential of simple yet effective deep learning techniques to significantly improve accuracy and reproducibility in breast cancer classification, supporting their integration into clinical workflows for better patient outcomes. 

---
# A Proposal for Networks Capable of Continual Learning 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2503.22068)  

**Abstract**: We analyze the ability of computational units to retain past responses after parameter updates, a key property for system-wide continual learning. Neural networks trained with gradient descent lack this capability, prompting us to propose Modelleyen, an alternative approach with inherent response preservation. We demonstrate through experiments on modeling the dynamics of a simple environment and on MNIST that, despite increased computational complexity and some representational limitations at its current stage, Modelleyen achieves continual learning without relying on sample replay or predefined task boundaries. 

---
# Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation 

**Authors**: Zeeshan Ahmed, Frank Seide, Zhe Liu, Rastislav Rabatin, Jachym Kolar, Niko Moritz, Ruiming Xie, Simone Merello, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22051)  

**Abstract**: Simultaneous or streaming machine translation generates translation while reading the input stream. These systems face a quality/latency trade-off, aiming to achieve high translation quality similar to non-streaming models with minimal latency. We propose an approach that efficiently manages this trade-off. By enhancing a pretrained non-streaming model, which was trained with a seq2seq mechanism and represents the upper bound in quality, we convert it into a streaming model by utilizing the alignment between source and target tokens. This alignment is used to learn a read/write decision boundary for reliable translation generation with minimal input. During training, the model learns the decision boundary through a read/write policy module, employing supervised learning on the alignment points (pseudo labels). The read/write policy module, a small binary classification unit, can control the quality/latency trade-off during inference. Experimental results show that our model outperforms several strong baselines and narrows the gap with the non-streaming baseline model. 

---
# Cognitive Prompts Using Guilford's Structure of Intellect Model 

**Authors**: Oliver Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2503.22036)  

**Abstract**: Large language models (LLMs) demonstrate strong language generation capabilities but often struggle with structured reasoning, leading to inconsistent or suboptimal problem-solving. To mitigate this limitation, Guilford's Structure of Intellect (SOI) model - a foundational framework from intelligence theory - is leveraged as the basis for cognitive prompt engineering. The SOI model categorizes cognitive operations such as pattern recognition, memory retrieval, and evaluation, offering a systematic approach to enhancing LLM reasoning and decision-making. This position paper presents a novel cognitive prompting approach for enforcing SOI-inspired reasoning for improving clarity, coherence, and adaptability in model responses. 

---
# Safeguarding Autonomy: a Focus on Machine Learning Decision Systems 

**Authors**: Paula Subías-Beltrán, Oriol Pujol, Itziar de Lecuona  

**Link**: [PDF](https://arxiv.org/pdf/2503.22023)  

**Abstract**: As global discourse on AI regulation gains momentum, this paper focuses on delineating the impact of ML on autonomy and fostering awareness. Respect for autonomy is a basic principle in bioethics that establishes persons as decision-makers. While the concept of autonomy in the context of ML appears in several European normative publications, it remains a theoretical concept that has yet to be widely accepted in ML practice. Our contribution is to bridge the theoretical and practical gap by encouraging the practical application of autonomy in decision-making within ML practice by identifying the conditioning factors that currently prevent it. Consequently, we focus on the different stages of the ML pipeline to identify the potential effects on ML end-users' autonomy. To improve its practical utility, we propose a related question for each detected impact, offering guidance for identifying possible focus points to respect ML end-users autonomy in decision-making. 

---
# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models 

**Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22020)  

**Abstract**: Vision-language-action models (VLAs) have shown potential in leveraging pretrained vision-language models and diverse robot demonstrations for learning generalizable sensorimotor control. While this paradigm effectively utilizes large-scale data from both robotic and non-robotic sources, current VLAs primarily focus on direct input--output mappings, lacking the intermediate reasoning steps crucial for complex manipulation tasks. As a result, existing VLAs lack temporal planning or reasoning capabilities. In this paper, we introduce a method that incorporates explicit visual chain-of-thought (CoT) reasoning into vision-language-action models (VLAs) by predicting future image frames autoregressively as visual goals before generating a short action sequence to achieve these goals. We introduce CoT-VLA, a state-of-the-art 7B VLA that can understand and generate visual and action tokens. Our experimental results demonstrate that CoT-VLA achieves strong performance, outperforming the state-of-the-art VLA model by 17% in real-world manipulation tasks and 6% in simulation benchmarks. Project website: this https URL 

---
# BOOTPLACE: Bootstrapped Object Placement with Detection Transformers 

**Authors**: Hang Zhou, Xinxin Zuo, Rui Ma, Li Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21991)  

**Abstract**: In this paper, we tackle the copy-paste image-to-image composition problem with a focus on object placement learning. Prior methods have leveraged generative models to reduce the reliance for dense supervision. However, this often limits their capacity to model complex data distributions. Alternatively, transformer networks with a sparse contrastive loss have been explored, but their over-relaxed regularization often leads to imprecise object placement. We introduce BOOTPLACE, a novel paradigm that formulates object placement as a placement-by-detection problem. Our approach begins by identifying suitable regions of interest for object placement. This is achieved by training a specialized detection transformer on object-subtracted backgrounds, enhanced with multi-object supervisions. It then semantically associates each target compositing object with detected regions based on their complementary characteristics. Through a boostrapped training approach applied to randomly object-subtracted images, our model enforces meaningful placements through extensive paired data augmentation. Experimental results on established benchmarks demonstrate BOOTPLACE's superior performance in object repositioning, markedly surpassing state-of-the-art baselines on Cityscapes and OPA datasets with notable improvements in IOU scores. Additional ablation studies further showcase the compositionality and generalizability of our approach, supported by user study evaluations. 

---
# Pretrained Bayesian Non-parametric Knowledge Prior in Robotic Long-Horizon Reinforcement Learning 

**Authors**: Yuan Meng, Xiangtong Yao, Kejia Chen, Yansong Wu, Liding Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21975)  

**Abstract**: Reinforcement learning (RL) methods typically learn new tasks from scratch, often disregarding prior knowledge that could accelerate the learning process. While some methods incorporate previously learned skills, they usually rely on a fixed structure, such as a single Gaussian distribution, to define skill priors. This rigid assumption can restrict the diversity and flexibility of skills, particularly in complex, long-horizon tasks. In this work, we introduce a method that models potential primitive skill motions as having non-parametric properties with an unknown number of underlying features. We utilize a Bayesian non-parametric model, specifically Dirichlet Process Mixtures, enhanced with birth and merge heuristics, to pre-train a skill prior that effectively captures the diverse nature of skills. Additionally, the learned skills are explicitly trackable within the prior space, enhancing interpretability and control. By integrating this flexible skill prior into an RL framework, our approach surpasses existing methods in long-horizon manipulation tasks, enabling more efficient skill transfer and task success in complex environments. Our findings show that a richer, non-parametric representation of skill priors significantly improves both the learning and execution of challenging robotic tasks. All data, code, and videos are available at this https URL. 

---
# Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback 

**Authors**: Yuan Meng, Xiangtong Yao, Haihui Ye, Yirui Zhou, Shengqiang Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21969)  

**Abstract**: Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning to enable robots to execute tasks from human commands. However, these methods often suffer from limited generalization, adaptability, and the lack of large-scale specialized datasets, unlike data-rich domains such as computer vision, making long-horizon task execution challenging. To address these gaps, we introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation, leveraging large language models (LLMs) for real-time task planning and execution. DAHLIA employs a dual-tunnel architecture, where an LLM-powered planner collaborates with co-planners to decompose tasks and generate executable plans, while a reporter LLM provides closed-loop feedback, enabling adaptive re-planning and ensuring task recovery from potential failures. Moreover, DAHLIA integrates chain-of-thought (CoT) in task reasoning and temporal abstraction for efficient action execution, enhancing traceability and robustness. Our framework demonstrates state-of-the-art performance across diverse long-horizon tasks, achieving strong generalization in both simulated and real-world scenarios. Videos and code are available at this https URL. 

---
# Entropy-Aware Branching for Improved Mathematical Reasoning 

**Authors**: Xianzhi Li, Ethan Callanan, Xiaodan Zhu, Mathieu Sibue, Antony Papadimitriou, Mahmoud Mahfouz, Zhiqiang Ma, Xiaomo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21961)  

**Abstract**: While Large Language Models (LLMs) are effectively aligned through extensive pre-training and fine-tuning, they still struggle with varying levels of uncertainty during token generation. In our investigation of mathematical reasoning, we observe that errors are more likely to arise at tokens exhibiting high entropy and variance of entropy in the model's output distribution. Based on the observation, we propose a novel approach that dynamically branches the generation process on demand instead of defaulting to the single most probable token. By exploring in parallel multiple branches stemming from high probability tokens of critical decision points, the model can discover diverse reasoning paths that might otherwise be missed. We further harness external feedback from larger models to rank and select the most coherent and accurate reasoning branch. Our experimental results on mathematical word problems and calculation questions show that this branching strategy boosts the reasoning capabilities of small LLMs up to 4.6% compared to conventional argmax decoding. 

---
# Parametric Shadow Control for Portrait Generationin Text-to-Image Diffusion Models 

**Authors**: Haoming Cai, Tsung-Wei Huang, Shiv Gehlot, Brandon Y. Feng, Sachin Shah, Guan-Ming Su, Christopher Metzler  

**Link**: [PDF](https://arxiv.org/pdf/2503.21943)  

**Abstract**: Text-to-image diffusion models excel at generating diverse portraits, but lack intuitive shadow control. Existing editing approaches, as post-processing, struggle to offer effective manipulation across diverse styles. Additionally, these methods either rely on expensive real-world light-stage data collection or require extensive computational resources for training. To address these limitations, we introduce Shadow Director, a method that extracts and manipulates hidden shadow attributes within well-trained diffusion models. Our approach uses a small estimation network that requires only a few thousand synthetic images and hours of training-no costly real-world light-stage data needed. Shadow Director enables parametric and intuitive control over shadow shape, placement, and intensity during portrait generation while preserving artistic integrity and identity across diverse styles. Despite training only on synthetic data built on real-world identities, it generalizes effectively to generated portraits with diverse styles, making it a more accessible and resource-friendly solution. 

---
# Lobster: A GPU-Accelerated Framework for Neurosymbolic Programming 

**Authors**: Paul Biberstein, Ziyang Li, Joseph Devietti, Mayur Naik  

**Link**: [PDF](https://arxiv.org/pdf/2503.21937)  

**Abstract**: Neurosymbolic programs combine deep learning with symbolic reasoning to achieve better data efficiency, interpretability, and generalizability compared to standalone deep learning approaches. However, existing neurosymbolic learning frameworks implement an uneasy marriage between a highly scalable, GPU-accelerated neural component with a slower symbolic component that runs on CPUs. We propose Lobster, a unified framework for harnessing GPUs in an end-to-end manner for neurosymbolic learning. Lobster maps a general neurosymbolic language based on Datalog to the GPU programming paradigm. This mapping is implemented via compilation to a new intermediate language called APM. The extra abstraction provided by APM allows Lobster to be both flexible, supporting discrete, probabilistic, and differentiable modes of reasoning on GPU hardware with a library of provenance semirings, and performant, implementing new optimization passes. We demonstrate that Lobster programs can solve interesting problems spanning the domains of natural language processing, image processing, program reasoning, bioinformatics, and planning. On a suite of 8 applications, Lobster achieves an average speedup of 5.3x over Scallop, a state-of-the-art neurosymbolic framework, and enables scaling of neurosymbolic solutions to previously infeasible tasks. 

---
# An Efficient Training Algorithm for Models with Block-wise Sparsity 

**Authors**: Ding Zhu, Zhiqun Zuo, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2503.21928)  

**Abstract**: Large-scale machine learning (ML) models are increasingly being used in critical domains like education, lending, recruitment, healthcare, criminal justice, etc. However, the training, deployment, and utilization of these models demand substantial computational resources. To decrease computation and memory costs, machine learning models with sparse weight matrices are widely used in the literature. Among sparse models, those with special sparse structures (e.g., models with block-wise sparse weight matrices) fit better with the hardware accelerators and can decrease the memory and computation costs during the inference. Unfortunately, while there are several efficient training methods, none of them are designed to train a block-wise sparse model efficiently. As a result, the current methods for training block-wise sparse models start with full and dense models leading to inefficient training. In this work, we focus on training models with \textit{block-wise sparse matrices} and propose an efficient training algorithm to decrease both computation and memory costs during training and inference. In addition, we will show that our proposed method enables us to efficiently find the right block size for the sparsity pattern during the training process. Our extensive empirical and theoretical analyses show that our algorithms can decrease the computation and memory costs significantly without a performance drop compared to baselines. 

---
# AutoPsyC: Automatic Recognition of Psychodynamic Conflicts from Semi-structured Interviews with Large Language Models 

**Authors**: Sayed Muddashir Hossain, Simon Ostermann, Patrick Gebhard, Cord Benecke, Josef van Genabith, Philipp Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.21911)  

**Abstract**: Psychodynamic conflicts are persistent, often unconscious themes that shape a person's behaviour and experiences. Accurate diagnosis of psychodynamic conflicts is crucial for effective patient treatment and is commonly done via long, manually scored semi-structured interviews. Existing automated solutions for psychiatric diagnosis tend to focus on the recognition of broad disorder categories such as depression, and it is unclear to what extent psychodynamic conflicts which even the patient themselves may not have conscious access to could be automatically recognised from conversation. In this paper, we propose AutoPsyC, the first method for recognising the presence and significance of psychodynamic conflicts from full-length Operationalized Psychodynamic Diagnostics (OPD) interviews using Large Language Models (LLMs). Our approach combines recent advances in parameter-efficient fine-tuning and Retrieval-Augmented Generation (RAG) with a summarisation strategy to effectively process entire 90 minute long conversations. In evaluations on a dataset of 141 diagnostic interviews we show that AutoPsyC consistently outperforms all baselines and ablation conditions on the recognition of four highly relevant psychodynamic conflicts. 

---
# JEEM: Vision-Language Understanding in Four Arabic Dialects 

**Authors**: Karima Kadaoui, Hanin Atwany, Hamdan Al-Ali, Abdelrahman Mohamed, Ali Mekky, Sergei Tilga, Natalia Fedorova, Ekaterina Artemova, Hanan Aldarmaki, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2503.21910)  

**Abstract**: We introduce JEEM, a benchmark designed to evaluate Vision-Language Models (VLMs) on visual understanding across four Arabic-speaking countries: Jordan, The Emirates, Egypt, and Morocco. JEEM includes the tasks of image captioning and visual question answering, and features culturally rich and regionally diverse content. This dataset aims to assess the ability of VLMs to generalize across dialects and accurately interpret cultural elements in visual contexts. In an evaluation of five prominent open-source Arabic VLMs and GPT-4V, we find that the Arabic VLMs consistently underperform, struggling with both visual understanding and dialect-specific generation. While GPT-4V ranks best in this comparison, the model's linguistic competence varies across dialects, and its visual understanding capabilities lag behind. This underscores the need for more inclusive models and the value of culturally-diverse evaluation paradigms. 

---
# Exponentially Weighted Instance-Aware Repeat Factor Sampling for Long-Tailed Object Detection Model Training in Unmanned Aerial Vehicles Surveillance Scenarios 

**Authors**: Taufiq Ahmed, Abhishek Kumar, Constantino Álvarez Casado, Anlan Zhang, Tuomo Hänninen, Lauri Loven, Miguel Bordallo López, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2503.21893)  

**Abstract**: Object detection models often struggle with class imbalance, where rare categories appear significantly less frequently than common ones. Existing sampling-based rebalancing strategies, such as Repeat Factor Sampling (RFS) and Instance-Aware Repeat Factor Sampling (IRFS), mitigate this issue by adjusting sample frequencies based on image and instance counts. However, these methods are based on linear adjustments, which limit their effectiveness in long-tailed distributions. This work introduces Exponentially Weighted Instance-Aware Repeat Factor Sampling (E-IRFS), an extension of IRFS that applies exponential scaling to better differentiate between rare and frequent classes. E-IRFS adjusts sampling probabilities using an exponential function applied to the geometric mean of image and instance frequencies, ensuring a more adaptive rebalancing strategy. We evaluate E-IRFS on a dataset derived from the Fireman-UAV-RGBT Dataset and four additional public datasets, using YOLOv11 object detection models to identify fire, smoke, people and lakes in emergency scenarios. The results show that E-IRFS improves detection performance by 22\% over the baseline and outperforms RFS and IRFS, particularly for rare categories. The analysis also highlights that E-IRFS has a stronger effect on lightweight models with limited capacity, as these models rely more on data sampling strategies to address class imbalance. The findings demonstrate that E-IRFS improves rare object detection in resource-constrained environments, making it a suitable solution for real-time applications such as UAV-based emergency monitoring. 

---
# StarFlow: Generating Structured Workflow Outputs From Sketch Images 

**Authors**: Patrice Bechard, Chao Wang, Amirhossein Abaskohi, Juan Rodriguez, Christopher Pal, David Vazquez, Spandana Gella, Sai Rajeswar, Perouz Taslakian  

**Link**: [PDF](https://arxiv.org/pdf/2503.21889)  

**Abstract**: Workflows are a fundamental component of automation in enterprise platforms, enabling the orchestration of tasks, data processing, and system integrations. Despite being widely used, building workflows can be complex, often requiring manual configuration through low-code platforms or visual programming tools. To simplify this process, we explore the use of generative foundation models, particularly vision-language models (VLMs), to automatically generate structured workflows from visual inputs. Translating hand-drawn sketches or computer-generated diagrams into executable workflows is challenging due to the ambiguity of free-form drawings, variations in diagram styles, and the difficulty of inferring execution logic from visual elements. To address this, we introduce StarFlow, a framework for generating structured workflow outputs from sketches using vision-language models. We curate a diverse dataset of workflow diagrams -- including synthetic, manually annotated, and real-world samples -- to enable robust training and evaluation. We finetune and benchmark multiple vision-language models, conducting a series of ablation studies to analyze the strengths and limitations of our approach. Our results show that finetuning significantly enhances structured workflow generation, outperforming large vision-language models on this task. 

---
# RedditESS: A Mental Health Social Support Interaction Dataset -- Understanding Effective Social Support to Refine AI-Driven Support Tools 

**Authors**: Zeyad Alghamdi, Tharindu Kumarage, Garima Agrawal, Mansooreh Karami, Ibrahim Almuteb, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21888)  

**Abstract**: Effective mental health support is crucial for alleviating psychological distress. While large language model (LLM)-based assistants have shown promise in mental health interventions, existing research often defines "effective" support primarily in terms of empathetic acknowledgments, overlooking other essential dimensions such as informational guidance, community validation, and tangible coping strategies. To address this limitation and better understand what constitutes effective support, we introduce RedditESS, a novel real-world dataset derived from Reddit posts, including supportive comments and original posters' follow-up responses. Grounded in established social science theories, we develop an ensemble labeling mechanism to annotate supportive comments as effective or not and perform qualitative assessments to ensure the reliability of the annotations. Additionally, we demonstrate the practical utility of RedditESS by using it to guide LLM alignment toward generating more context-sensitive and genuinely helpful supportive responses. By broadening the understanding of effective support, our study paves the way for advanced AI-driven mental health interventions. 

---
# Foveated Instance Segmentation 

**Authors**: Hongyi Zeng, Wenxuan Liu, Tianhua Xia, Jinhui Chen, Ziyun Li, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21854)  

**Abstract**: Instance segmentation is essential for augmented reality and virtual reality (AR/VR) as it enables precise object recognition and interaction, enhancing the integration of virtual and real-world elements for an immersive experience. However, the high computational overhead of segmentation limits its application on resource-constrained AR/VR devices, causing large processing latency and degrading user experience. In contrast to conventional scenarios, AR/VR users typically focus on only a few regions within their field of view before shifting perspective, allowing segmentation to be concentrated on gaze-specific areas. This insight drives the need for efficient segmentation methods that prioritize processing instance of interest, reducing computational load and enhancing real-time performance. In this paper, we present a foveated instance segmentation (FovealSeg) framework that leverages real-time user gaze data to perform instance segmentation exclusively on instance of interest, resulting in substantial computational savings. Evaluation results show that FSNet achieves an IoU of 0.56 on ADE20K and 0.54 on LVIS, notably outperforming the baseline. The code is available at this https URL 

---
# Comparative Analysis of Image, Video, and Audio Classifiers for Automated News Video Segmentation 

**Authors**: Jonathan Attard, Dylan Seychell  

**Link**: [PDF](https://arxiv.org/pdf/2503.21848)  

**Abstract**: News videos require efficient content organisation and retrieval systems, but their unstructured nature poses significant challenges for automated processing. This paper presents a comprehensive comparative analysis of image, video, and audio classifiers for automated news video segmentation. This work presents the development and evaluation of multiple deep learning approaches, including ResNet, ViViT, AST, and multimodal architectures, to classify five distinct segment types: advertisements, stories, studio scenes, transitions, and visualisations. Using a custom-annotated dataset of 41 news videos comprising 1,832 scene clips, our experiments demonstrate that image-based classifiers achieve superior performance (84.34\% accuracy) compared to more complex temporal models. Notably, the ResNet architecture outperformed state-of-the-art video classifiers while requiring significantly fewer computational resources. Binary classification models achieved high accuracy for transitions (94.23\%) and advertisements (92.74\%). These findings advance the understanding of effective architectures for news video segmentation and provide practical insights for implementing automated content organisation systems in media applications. These include media archiving, personalised content delivery, and intelligent video search. 

---
# ReCoM: Realistic Co-Speech Motion Generation with Recurrent Embedded Transformer 

**Authors**: Yong Xie, Yunlian Sun, Hongwen Zhang, Yebin Liu, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21847)  

**Abstract**: We present ReCoM, an efficient framework for generating high-fidelity and generalizable human body motions synchronized with speech. The core innovation lies in the Recurrent Embedded Transformer (RET), which integrates Dynamic Embedding Regularization (DER) into a Vision Transformer (ViT) core architecture to explicitly model co-speech motion dynamics. This architecture enables joint spatial-temporal dependency modeling, thereby enhancing gesture naturalness and fidelity through coherent motion synthesis. To enhance model robustness, we incorporate the proposed DER strategy, which equips the model with dual capabilities of noise resistance and cross-domain generalization, thereby improving the naturalness and fluency of zero-shot motion generation for unseen speech inputs. To mitigate inherent limitations of autoregressive inference, including error accumulation and limited self-correction, we propose an iterative reconstruction inference (IRI) strategy. IRI refines motion sequences via cyclic pose reconstruction, driven by two key components: (1) classifier-free guidance improves distribution alignment between generated and real gestures without auxiliary supervision, and (2) a temporal smoothing process eliminates abrupt inter-frame transitions while ensuring kinematic continuity. Extensive experiments on benchmark datasets validate ReCoM's effectiveness, achieving state-of-the-art performance across metrics. Notably, it reduces the Fréchet Gesture Distance (FGD) from 18.70 to 2.48, demonstrating an 86.7% improvement in motion realism. Our project page is this https URL. 

---
# LightSNN: Lightweight Architecture Search for Sparse and Accurate Spiking Neural Networks 

**Authors**: Yesmine Abdennadher, Giovanni Perin, Riccardo Mazzieri, Jacopo Pegoraro, Michele Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.21846)  

**Abstract**: Spiking Neural Networks (SNNs) are highly regarded for their energy efficiency, inherent activation sparsity, and suitability for real-time processing in edge devices. However, most current SNN methods adopt architectures resembling traditional artificial neural networks (ANNs), leading to suboptimal performance when applied to SNNs. While SNNs excel in energy efficiency, they have been associated with lower accuracy levels than traditional ANNs when utilizing conventional architectures. In response, in this work we present LightSNN, a rapid and efficient Neural Network Architecture Search (NAS) technique specifically tailored for SNNs that autonomously leverages the most suitable architecture, striking a good balance between accuracy and efficiency by enforcing sparsity. Based on the spiking NAS network (SNASNet) framework, a cell-based search space including backward connections is utilized to build our training-free pruning-based NAS mechanism. Our technique assesses diverse spike activation patterns across different data samples using a sparsity-aware Hamming distance fitness evaluation. Thorough experiments are conducted on both static (CIFAR10 and CIFAR100) and neuromorphic datasets (DVS128-Gesture). Our LightSNN model achieves state-of-the-art results on CIFAR10 and CIFAR100, improves performance on DVS128Gesture by 4.49%, and significantly reduces search time, most notably offering a 98x speedup over SNASNet and running 30% faster than the best existing method on DVS128Gesture. 

---
# CMD-HAR: Cross-Modal Disentanglement for Wearable Human Activity Recognition 

**Authors**: Hanyu Liu, Siyao Li, Ying Yu, Yixuan Jiang, Hang Xiao, Jingxi Long, Haotian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21843)  

**Abstract**: Human Activity Recognition (HAR) is a fundamental technology for numerous human - centered intelligent applications. Although deep learning methods have been utilized to accelerate feature extraction, issues such as multimodal data mixing, activity heterogeneity, and complex model deployment remain largely unresolved. The aim of this paper is to address issues such as multimodal data mixing, activity heterogeneity, and complex model deployment in sensor-based human activity recognition. We propose a spatiotemporal attention modal decomposition alignment fusion strategy to tackle the problem of the mixed distribution of sensor data. Key discriminative features of activities are captured through cross-modal spatio-temporal disentangled representation, and gradient modulation is combined to alleviate data heterogeneity. In addition, a wearable deployment simulation system is constructed. We conducted experiments on a large number of public datasets, demonstrating the effectiveness of the model. 

---
# M-DocSum: Do LVLMs Genuinely Comprehend Interleaved Image-Text in Document Summarization? 

**Authors**: Haolong Yan, Kaijun Tan, Yeqing Shen, Xin Huang, Zheng Ge, Xiangyu Zhang, Si Li, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21839)  

**Abstract**: We investigate a critical yet under-explored question in Large Vision-Language Models (LVLMs): Do LVLMs genuinely comprehend interleaved image-text in the document? Existing document understanding benchmarks often assess LVLMs using question-answer formats, which are information-sparse and difficult to guarantee the coverage of long-range dependencies. To address this issue, we introduce a novel and challenging Multimodal Document Summarization Benchmark (M-DocSum-Bench), which comprises 500 high-quality arXiv papers, along with interleaved multimodal summaries aligned with human preferences. M-DocSum-Bench is a reference-based generation task and necessitates the generation of interleaved image-text summaries using provided reference images, thereby simultaneously evaluating capabilities in understanding, reasoning, localization, and summarization within complex multimodal document scenarios. To facilitate this benchmark, we develop an automated framework to construct summaries and propose a fine-grained evaluation method called M-DocEval. Moreover, we further develop a robust summarization baseline, i.e., M-DocSum-7B, by progressive two-stage training with diverse instruction and preference data. The extensive results on our M-DocSum-Bench reveal that the leading LVLMs struggle to maintain coherence and accurately integrate information within long and interleaved contexts, often exhibiting confusion between similar images and a lack of robustness. Notably, M-DocSum-7B achieves state-of-the-art performance compared to larger and closed-source models (including GPT-4o, Gemini Pro, Claude-3.5-Sonnet and Qwen2.5-VL-72B, etc.), demonstrating the potential of LVLMs for improved interleaved image-text understanding. The code, data, and models are available at this https URL. 

---
# MSPLoRA: A Multi-Scale Pyramid Low-Rank Adaptation for Efficient Model Fine-Tuning 

**Authors**: Jiancheng Zhao, Xingda Yu, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21838)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has become an essential approach for adapting large-scale pre-trained models while reducing computational costs. Among PEFT methods, LoRA significantly reduces trainable parameters by decomposing weight updates into low-rank matrices. However, traditional LoRA applies a fixed rank across all layers, failing to account for the varying complexity of hierarchical information, which leads to inefficient adaptation and redundancy. To address this, we propose MSPLoRA (Multi-Scale Pyramid LoRA), which introduces Global Shared LoRA, Mid-Level Shared LoRA, and Layer-Specific LoRA to capture global patterns, mid-level features, and fine-grained information, respectively. This hierarchical structure reduces inter-layer redundancy while maintaining strong adaptation capability. Experiments on various NLP tasks demonstrate that MSPLoRA achieves more efficient adaptation and better performance while significantly reducing the number of trainable parameters. Furthermore, additional analyses based on Singular Value Decomposition validate its information decoupling ability, highlighting MSPLoRA as a scalable and effective optimization strategy for parameter-efficient fine-tuning in large language models. Our code is available at this https URL. 

---
# A Multi-Modal Knowledge-Enhanced Framework for Vessel Trajectory Prediction 

**Authors**: Haomin Yu, Tianyi Li, Kristian Torp, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21834)  

**Abstract**: Accurate vessel trajectory prediction facilitates improved navigational safety, routing, and environmental protection. However, existing prediction methods are challenged by the irregular sampling time intervals of the vessel tracking data from the global AIS system and the complexity of vessel movement. These aspects render model learning and generalization difficult. To address these challenges and improve vessel trajectory prediction, we propose the multi-modal knowledge-enhanced framework (MAKER) for vessel trajectory prediction. To contend better with the irregular sampling time intervals, MAKER features a Large language model-guided Knowledge Transfer (LKT) module that leverages pre-trained language models to transfer trajectory-specific contextual knowledge effectively. To enhance the ability to learn complex trajectory patterns, MAKER incorporates a Knowledge-based Self-paced Learning (KSL) module. This module employs kinematic knowledge to progressively integrate complex patterns during training, allowing for adaptive learning and enhanced generalization. Experimental results on two vessel trajectory datasets show that MAKER can improve the prediction accuracy of state-of-the-art methods by 12.08%-17.86%. 

---
# ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks 

**Authors**: Mohamed Afane, Gabrielle Ebbrecht, Ying Wang, Juntao Chen, Junaid Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2503.21815)  

**Abstract**: Quantum Neural Networks (QNNs) offer promising capabilities for complex data tasks, but are often constrained by limited qubit resources and high entanglement, which can hinder scalability and efficiency. In this paper, we introduce Adaptive Threshold Pruning (ATP), an encoding method that reduces entanglement and optimizes data complexity for efficient computations in QNNs. ATP dynamically prunes non-essential features in the data based on adaptive thresholds, effectively reducing quantum circuit requirements while preserving high performance. Extensive experiments across multiple datasets demonstrate that ATP reduces entanglement entropy and improves adversarial robustness when combined with adversarial training methods like FGSM. Our results highlight ATPs ability to balance computational efficiency and model resilience, achieving significant performance improvements with fewer resources, which will help make QNNs more feasible in practical, resource-constrained settings. 

---
# Taxonomy Inference for Tabular Data Using Large Language Models 

**Authors**: Zhenyu Wu, Jiaoyan Chen, Norman W. Paton  

**Link**: [PDF](https://arxiv.org/pdf/2503.21810)  

**Abstract**: Taxonomy inference for tabular data is a critical task of schema inference, aiming at discovering entity types (i.e., concepts) of the tables and building their hierarchy. It can play an important role in data management, data exploration, ontology learning, and many data-centric applications. Existing schema inference systems focus more on XML, JSON or RDF data, and often rely on lexical formats and structures of the data for calculating similarities, with limited exploitation of the semantics of the text across a table. Motivated by recent works on taxonomy completion and construction using Large Language Models (LLMs), this paper presents two LLM-based methods for taxonomy inference for tables: (i) EmTT which embeds columns by fine-tuning with contrastive learning encoder-alone LLMs like BERT and utilises clustering for hierarchy construction, and (ii) GeTT which generates table entity types and their hierarchy by iterative prompting using a decoder-alone LLM like GPT-4. Extensive evaluation on three real-world datasets with six metrics covering different aspects of the output taxonomies has demonstrated that EmTT and GeTT can both produce taxonomies with strong consistency relative to the Ground Truth. 

---
# LERO: LLM-driven Evolutionary framework with Hybrid Rewards and Enhanced Observation for Multi-Agent Reinforcement Learning 

**Authors**: Yuan Wei, Xiaohan Shan, Jianmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21807)  

**Abstract**: Multi-agent reinforcement learning (MARL) faces two critical bottlenecks distinct from single-agent RL: credit assignment in cooperative tasks and partial observability of environmental states. We propose LERO, a framework integrating Large language models (LLMs) with evolutionary optimization to address these MARL-specific challenges. The solution centers on two LLM-generated components: a hybrid reward function that dynamically allocates individual credit through reward decomposition, and an observation enhancement function that augments partial observations with inferred environmental context. An evolutionary algorithm optimizes these components through iterative MARL training cycles, where top-performing candidates guide subsequent LLM generations. Evaluations in Multi-Agent Particle Environments (MPE) demonstrate LERO's superiority over baseline methods, with improved task performance and training efficiency. 

---
# Large Language Models Meet Contrastive Learning: Zero-Shot Emotion Recognition Across Languages 

**Authors**: Heqing Zou, Fengmao Lv, Desheng Zheng, Eng Siong Chng, Deepu Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21806)  

**Abstract**: Multilingual speech emotion recognition aims to estimate a speaker's emotional state using a contactless method across different languages. However, variability in voice characteristics and linguistic diversity poses significant challenges for zero-shot speech emotion recognition, especially with multilingual datasets. In this paper, we propose leveraging contrastive learning to refine multilingual speech features and extend large language models for zero-shot multilingual speech emotion estimation. Specifically, we employ a novel two-stage training framework to align speech signals with linguistic features in the emotional space, capturing both emotion-aware and language-agnostic speech representations. To advance research in this field, we introduce a large-scale synthetic multilingual speech emotion dataset, M5SER. Our experiments demonstrate the effectiveness of the proposed method in both speech emotion recognition and zero-shot multilingual speech emotion recognition, including previously unseen datasets and languages. 

---
# ImF: Implicit Fingerprint for Large Language Models 

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21805)  

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership. 

---
# Comparison of Metadata Representation Models for Knowledge Graph Embeddings 

**Authors**: Shusaku Egami, Kyoumoto Matsushita, Takanori Ugai, Ken Fukuda  

**Link**: [PDF](https://arxiv.org/pdf/2503.21804)  

**Abstract**: Hyper-relational Knowledge Graphs (HRKGs) extend traditional KGs beyond binary relations, enabling the representation of contextual, provenance, and temporal information in domains, such as historical events, sensor data, video content, and narratives. HRKGs can be structured using several Metadata Representation Models (MRMs), including Reification (REF), Singleton Property (SGP), and RDF-star (RDR). However, the effects of different MRMs on KG Embedding (KGE) and Link Prediction (LP) models remain unclear. This study evaluates MRMs in the context of LP tasks, identifies the limitations of existing evaluation frameworks, and introduces a new task that ensures fair comparisons across MRMs. Furthermore, we propose a framework that effectively reflects the knowledge representations of the three MRMs in latent space. Experiments on two types of datasets reveal that REF performs well in simple HRKGs, whereas SGP is less effective. However, in complex HRKGs, the differences among MRMs in the LP tasks are minimal. Our findings contribute to an optimal knowledge representation strategy for HRKGs in LP tasks. 

---
# Forecasting Volcanic Radiative Power (VPR) at Fuego Volcano Using Bayesian Regularized Neural Network 

**Authors**: Snehamoy Chatterjee, Greg Waite, Sidike Paheding, Luke Bowman  

**Link**: [PDF](https://arxiv.org/pdf/2503.21803)  

**Abstract**: Forecasting volcanic activity is critical for hazard assessment and risk mitigation. Volcanic Radiative Power (VPR), derived from thermal remote sensing data, serves as an essential indicator of volcanic activity. In this study, we employ Bayesian Regularized Neural Networks (BRNN) to predict future VPR values based on historical data from Fuego Volcano, comparing its performance against Scaled Conjugate Gradient (SCG) and Levenberg-Marquardt (LM) models. The results indicate that BRNN outperforms SCG and LM, achieving the lowest mean squared error (1.77E+16) and the highest R-squared value (0.50), demonstrating its superior ability to capture VPR variability while minimizing overfitting. Despite these promising results, challenges remain in improving the model's predictive accuracy. Future research should focus on integrating additional geophysical parameters, such as seismic and gas emission data, to enhance forecasting precision. The findings highlight the potential of machine learning models, particularly BRNN, in advancing volcanic activity forecasting, contributing to more effective early warning systems for volcanic hazards. 

---
# Efficient Joint Prediction of Multiple Future Tokens 

**Authors**: Kwangjun Ahn, Alex Lamb, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2503.21801)  

**Abstract**: In this short report, we introduce joint multi-token prediction (JTP), a lightweight modification of standard next-token prediction designed to enrich hidden state representations by jointly predicting multiple future tokens. Unlike previous multi-token prediction approaches, JTP strategically employs teacher forcing of future-tokens through a carefully designed representation bottleneck, allowing the model to encode rich predictive information with minimal computational overhead during training. We show that the JTP approach achieves a short-horizon belief state representation, while popular alternatives for multi-token prediction fail to do so. We demonstrate the effectiveness of our method on the synthetic star graph navigation task from from Bachmann and Nagarajan [2024], highlighting a significant performance improvement over existing methods. This manuscript presents promising preliminary results intended to stimulate further research. 

---
# ELM: Ensemble of Language Models for Predicting Tumor Group from Pathology Reports 

**Authors**: Lovedeep Gondara, Jonathan Simkin, Shebnum Devji, Gregory Arbour, Raymond Ng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21800)  

**Abstract**: Population-based cancer registries (PBCRs) face a significant bottleneck in manually extracting data from unstructured pathology reports, a process crucial for tasks like tumor group assignment, which can consume 900 person-hours for approximately 100,000 reports. To address this, we introduce ELM (Ensemble of Language Models), a novel ensemble-based approach leveraging both small language models (SLMs) and large language models (LLMs). ELM utilizes six fine-tuned SLMs, where three SLMs use the top part of the pathology report and three SLMs use the bottom part. This is done to maximize report coverage. ELM requires five-out-of-six agreement for a tumor group classification. Disagreements are arbitrated by an LLM with a carefully curated prompt. Our evaluation across nineteen tumor groups demonstrates ELM achieves an average precision and recall of 0.94, outperforming single-model and ensemble-without-LLM approaches. Deployed at the British Columbia Cancer Registry, ELM demonstrates how LLMs can be successfully applied in a PBCR setting to achieve state-of-the-art results and significantly enhance operational efficiencies, saving hundreds of person-hours annually. 

---
# A Novel Two-Phase Cooperative Co-evolution Framework for Large-Scale Global Optimization with Complex Overlapping 

**Authors**: Wenjie Qiu, Hongshu Guo, Zeyuan Ma, Yue-Jiao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.21797)  

**Abstract**: Cooperative Co-evolution, through the decomposition of the problem space, is a primary approach for solving large-scale global optimization problems. Typically, when the subspaces are disjoint, the algorithms demonstrate significantly both effectiveness and efficiency compared to non-decomposition algorithms. However, the presence of overlapping variables complicates the decomposition process and adversely affects the performance of cooperative co-evolution. In this study, we propose a novel two-phase cooperative co-evolution framework to address large-scale global optimization problems with complex overlapping. An effective method for decomposing overlapping problems, grounded in their mathematical properties, is embedded within the framework. Additionally, a customizable benchmark for overlapping problems is introduced to extend existing benchmarks and facilitate experimentation. Extensive experiments demonstrate that the algorithm instantiated within our framework significantly outperforms existing algorithms. The results reveal the characteristics of overlapping problems and highlight the differing strengths of cooperative co-evolution and non-decomposition algorithms. Our work is open-source and accessible at: this https URL. 

---
# Threshold Adaptation in Spiking Networks Enables Shortest Path Finding and Place Disambiguation 

**Authors**: Robin Dietrich, Tobias Fischer, Nicolai Waniek, Nico Reeb, Michael Milford, Alois Knoll, Adam D. Hines  

**Link**: [PDF](https://arxiv.org/pdf/2503.21795)  

**Abstract**: Efficient spatial navigation is a hallmark of the mammalian brain, inspiring the development of neuromorphic systems that mimic biological principles. Despite progress, implementing key operations like back-tracing and handling ambiguity in bio-inspired spiking neural networks remains an open challenge. This work proposes a mechanism for activity back-tracing in arbitrary, uni-directional spiking neuron graphs. We extend the existing replay mechanism of the spiking hierarchical temporal memory (S-HTM) by our spike timing-dependent threshold adaptation (STDTA), which enables us to perform path planning in networks of spiking neurons. We further present an ambiguity dependent threshold adaptation (ADTA) for identifying places in an environment with less ambiguity, enhancing the localization estimate of an agent. Combined, these methods enable efficient identification of the shortest path to an unambiguous target. Our experiments show that a network trained on sequences reliably computes shortest paths with fewer replays than the steps required to reach the target. We further show that we can identify places with reduced ambiguity in multiple, similar environments. These contributions advance the practical application of biologically inspired sequential learning algorithms like the S-HTM towards neuromorphic localization and navigation. 

---
# Architecture of Information 

**Authors**: Yurii Parzhyn  

**Link**: [PDF](https://arxiv.org/pdf/2503.21794)  

**Abstract**: The paper explores an approach to constructing energy landscapes of a formal neuron and multilayer artificial neural networks (ANNs). Their analysis makes it possible to determine the conceptual limitations of both classification ANNs (e.g., MLP or CNN) and generative ANN models. The study of informational and thermodynamic entropy in formal neuron and ANN models leads to the conclusion about the energetic nature of informational entropy. The application of the Gibbs free energy concept allows representing the output information of ANNs as the structured part of enthalpy. Modeling ANNs as energy systems makes it possible to interpret the structure of their internal energy as an internal model of the external world, which self-organizes based on the interaction of the system's internal energy components. The control of the self-organization and evolution process of this model is carried out through an energy function (analogous to the Lyapunov function) based on reduction operators. This makes it possible to introduce a new approach to constructing self-organizing and evolutionary ANNs with direct learning, which does not require additional external algorithms. The presented research makes it possible to formulate a formal definition of information in terms of the interaction processes between the internal and external energy of the system. 

---
# Input-Triggered Hardware Trojan Attack on Spiking Neural Networks 

**Authors**: Spyridon Raptis, Paul Kling, Ioannis Kaskampas, Ihsen Alouani, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.21793)  

**Abstract**: Neuromorphic computing based on spiking neural networks (SNNs) is emerging as a promising alternative to traditional artificial neural networks (ANNs), offering unique advantages in terms of low power consumption. However, the security aspect of SNNs is under-explored compared to their ANN counterparts. As the increasing reliance on AI systems comes with unique security risks and challenges, understanding the vulnerabilities and threat landscape is essential as neuromorphic computing matures. In this effort, we propose a novel input-triggered Hardware Trojan (HT) attack for SNNs. The HT mechanism is condensed in the area of one neuron. The trigger mechanism is an input message crafted in the spiking domain such that a selected neuron produces a malicious spike train that is not met in normal settings. This spike train triggers a malicious modification in the neuron that forces it to saturate, firing permanently and failing to recover to its resting state even when the input activity stops. The excessive spikes pollute the network and produce misleading decisions. We propose a methodology to select an appropriate neuron and to generate the input pattern that triggers the HT payload. The attack is illustrated by simulation on three popular benchmarks in the neuromorphic community. We also propose a hardware implementation for an analog spiking neuron and a digital SNN accelerator, demonstrating that the HT has a negligible area and power footprint and, thereby, can easily evade detection. 

---
# March Madness Tournament Predictions Model: A Mathematical Modeling Approach 

**Authors**: Christian McIver, Karla Avalos, Nikhil Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2503.21790)  

**Abstract**: This paper proposes a model to predict the outcome of the March Madness tournament based on historical NCAA basketball data since 2013. The framework of this project is a simplification of the FiveThrityEight NCAA March Madness prediction model, where the only four predictors of interest are Adjusted Offensive Efficiency (ADJOE), Adjusted Defensive Efficiency (ADJDE), Power Rating, and Two-Point Shooting Percentage Allowed. A logistic regression was utilized with the aforementioned metrics to generate a probability of a particular team winning each game. Then, a tournament simulation is developed and compared to real-world March Madness brackets to determine the accuracy of the model. Accuracies of performance were calculated using a naive approach and a Spearman rank correlation coefficient. 

---
# From Deep Learning to LLMs: A survey of AI in Quantitative Investment 

**Authors**: Bokai Cao, Saizhuo Wang, Xinyi Lin, Xiaojun Wu, Haohan Zhang, Lionel M. Ni, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.21422)  

**Abstract**: Quantitative investment (quant) is an emerging, technology-driven approach in asset management, increasingy shaped by advancements in artificial intelligence. Recent advances in deep learning and large language models (LLMs) for quant finance have improved predictive modeling and enabled agent-based automation, suggesting a potential paradigm shift in this field. In this survey, taking alpha strategy as a representative example, we explore how AI contributes to the quantitative investment pipeline. We first examine the early stage of quant research, centered on human-crafted features and traditional statistical models with an established alpha pipeline. We then discuss the rise of deep learning, which enabled scalable modeling across the entire pipeline from data processing to order execution. Building on this, we highlight the emerging role of LLMs in extending AI beyond prediction, empowering autonomous agents to process unstructured data, generate alphas, and support self-iterative workflows. 

---
# Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos 

**Authors**: Jiaheng Zhou, Yanfeng Zhou, Wei Fang, Yuxing Tang, Le Lu, Ge Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20258)  

**Abstract**: Ultrasound videos are an important form of clinical imaging data, and deep learning-based automated analysis can improve diagnostic accuracy and clinical efficiency. However, the scarcity of labeled data and the inherent challenges of video analysis have impeded the advancement of related methods. In this work, we introduce E-ViM$^3$, a data-efficient Vision Mamba network that preserves the 3D structure of video data, enhancing long-range dependencies and inductive biases to better model space-time correlations. With our design of Enclosure Global Tokens (EGT), the model captures and aggregates global features more effectively than competing methods. To further improve data efficiency, we employ masked video modeling for self-supervised pre-training, with the proposed Spatial-Temporal Chained (STC) masking strategy designed to adapt to various video scenarios. Experiments demonstrate that E-ViM$^3$ performs as the state-of-the-art in two high-level semantic analysis tasks across four datasets of varying sizes: EchoNet-Dynamic, CAMUS, MICCAI-BUV, and WHBUS. Furthermore, our model achieves competitive performance with limited labels, highlighting its potential impact on real-world clinical applications. 

---
