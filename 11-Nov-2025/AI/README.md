# DigiData: Training and Evaluating General-Purpose Mobile Control Agents 

**Authors**: Yuxuan Sun, Manchen Wang, Shengyi Qian, William R. Wong, Eric Gan, Pierluca D'Oro, Alejandro Castillejo Munoz, Sneha Silwal, Pedro Matias, Nitin Kamra, Satwik Kottur, Nick Raines, Xuanyi Zhao, Joy Chen, Joseph Greer, Andrea Madotto, Allen Bolourchi, James Valori, Kevin Carlberg, Karl Ridgeway, Joseph Tighe  

**Link**: [PDF](https://arxiv.org/pdf/2511.07413)  

**Abstract**: AI agents capable of controlling user interfaces have the potential to transform human interaction with digital devices. To accelerate this transformation, two fundamental building blocks are essential: high-quality datasets that enable agents to achieve complex and human-relevant goals, and robust evaluation methods that allow researchers and practitioners to rapidly enhance agent performance. In this paper, we introduce DigiData, a large-scale, high-quality, diverse, multi-modal dataset designed for training mobile control agents. Unlike existing datasets, which derive goals from unstructured interactions, DigiData is meticulously constructed through comprehensive exploration of app features, resulting in greater diversity and higher goal complexity. Additionally, we present DigiData-Bench, a benchmark for evaluating mobile control agents on real-world complex tasks. We demonstrate that the commonly used step-accuracy metric falls short in reliably assessing mobile control agents and, to address this, we propose dynamic evaluation protocols and AI-powered evaluations as rigorous alternatives for agent assessment. Our contributions aim to significantly advance the development of mobile control agents, paving the way for more intuitive and effective human-device interactions. 

---
# DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas 

**Authors**: Zhen Wang, Yufan Zhou, Zhongyan Luo, Lyumanshan Ye, Adam Wood, Man Yao, Luoshang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07338)  

**Abstract**: Simulating human profiles by instilling personas into large language models (LLMs) is rapidly transforming research in agentic behavioral simulation, LLM personalization, and human-AI alignment. However, most existing synthetic personas remain shallow and simplistic, capturing minimal attributes and failing to reflect the rich complexity and diversity of real human identities. We introduce DEEPPERSONA, a scalable generative engine for synthesizing narrative-complete synthetic personas through a two-stage, taxonomy-guided method. First, we algorithmically construct the largest-ever human-attribute taxonomy, comprising over hundreds of hierarchically organized attributes, by mining thousands of real user-ChatGPT conversations. Second, we progressively sample attributes from this taxonomy, conditionally generating coherent and realistic personas that average hundreds of structured attributes and roughly 1 MB of narrative text, two orders of magnitude deeper than prior works. Intrinsic evaluations confirm significant improvements in attribute diversity (32 percent higher coverage) and profile uniqueness (44 percent greater) compared to state-of-the-art baselines. Extrinsically, our personas enhance GPT-4.1-mini's personalized question answering accuracy by 11.6 percent on average across ten metrics and substantially narrow (by 31.7 percent) the gap between simulated LLM citizens and authentic human responses in social surveys. Our generated national citizens reduced the performance gap on the Big Five personality test by 17 percent relative to LLM-simulated citizens. DEEPPERSONA thus provides a rigorous, scalable, and privacy-free platform for high-fidelity human simulation and personalized AI research. 

---
# IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction 

**Authors**: Guoxin Chen, Zile Qiao, Xuanzhong Chen, Donglei Yu, Haotian Xu, Wayne Xin Zhao, Ruihua Song, Wenbiao Yin, Huifeng Yin, Liwen Zhang, Kuan Li, Minpeng Liao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.07327)  

**Abstract**: Recent advances in deep-research agents have shown promise for autonomous knowledge construction through dynamic reasoning over external sources. However, existing approaches rely on a mono-contextual paradigm that accumulates all information in a single, expanding context window, leading to context suffocation and noise contamination that limit their effectiveness on long-horizon tasks. We introduce IterResearch, a novel iterative deep-research paradigm that reformulates long-horizon research as a Markov Decision Process with strategic workspace reconstruction. By maintaining an evolving report as memory and periodically synthesizing insights, our approach preserves consistent reasoning capacity across arbitrary exploration depths. We further develop Efficiency-Aware Policy Optimization (EAPO), a reinforcement learning framework that incentivizes efficient exploration through geometric reward discounting and enables stable distributed training via adaptive downsampling. Extensive experiments demonstrate that IterResearch achieves substantial improvements over existing open-source agents with average +14.5pp across six benchmarks and narrows the gap with frontier proprietary systems. Remarkably, our paradigm exhibits unprecedented interaction scaling, extending to 2048 interactions with dramatic performance gains (from 3.5\% to 42.5\%), and serves as an effective prompting strategy, improving frontier models by up to 19.2pp over ReAct on long-horizon tasks. These findings position IterResearch as a versatile solution for long-horizon reasoning, effective both as a trained agent and as a prompting paradigm for frontier models. 

---
# Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion 

**Authors**: Chen Han, Yijia Ma, Jin Tan, Wenzhen Zheng, Xijin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07267)  

**Abstract**: Multi-agent debate (MAD) frameworks have emerged as promising approaches for misinformation detection by simulating adversarial reasoning. While prior work has focused on detection accuracy, it overlooks the importance of helping users understand the reasoning behind factual judgments and develop future resilience. The debate transcripts generated during MAD offer a rich but underutilized resource for transparent reasoning. In this study, we introduce ED2D, an evidence-based MAD framework that extends previous approach by incorporating factual evidence retrieval. More importantly, ED2D is designed not only as a detection framework but also as a persuasive multi-agent system aimed at correcting user beliefs and discouraging misinformation sharing. We compare the persuasive effects of ED2D-generated debunking transcripts with those authored by human experts. Results demonstrate that ED2D outperforms existing baselines across three misinformation detection benchmarks. When ED2D generates correct predictions, its debunking transcripts exhibit persuasive effects comparable to those of human experts; However, when ED2D misclassifies, its accompanying explanations may inadvertently reinforce users'misconceptions, even when presented alongside accurate human explanations. Our findings highlight both the promise and the potential risks of deploying MAD systems for misinformation intervention. We further develop a public community website to help users explore ED2D, fostering transparency, critical thinking, and collaborative fact-checking. 

---
# AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning 

**Authors**: Qile Jiang, George Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.07262)  

**Abstract**: Scientific Machine Learning (SciML) integrates data-driven inference with physical modeling to solve complex problems in science and engineering. However, the design of SciML architectures, loss formulations, and training strategies remains an expert-driven research process, requiring extensive experimentation and problem-specific insights. Here we introduce AgenticSciML, a collaborative multi-agent system in which over 10 specialized AI agents collaborate to propose, critique, and refine SciML solutions through structured reasoning and iterative evolution. The framework integrates structured debate, retrieval-augmented method memory, and ensemble-guided evolutionary search, enabling the agents to generate and assess new hypotheses about architectures and optimization procedures. Across physics-informed learning and operator learning tasks, the framework discovers solution methods that outperform single-agent and human-designed baselines by up to four orders of magnitude in error reduction. The agents produce novel strategies -- including adaptive mixture-of-expert architectures, decomposition-based PINNs, and physics-informed operator learning models -- that do not appear explicitly in the curated knowledge base. These results show that collaborative reasoning among AI agents can yield emergent methodological innovation, suggesting a path toward scalable, transparent, and autonomous discovery in scientific computing. 

---
# PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork 

**Authors**: Hohei Chan, Xinzhi Zhang, Antao Xiang, Weinan Zhang, Mengchen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07260)  

**Abstract**: Ad hoc teamwork (AHT) requires agents to collaborate with previously unseen teammates, which is crucial for many real-world applications. The core challenge of AHT is to develop an ego agent that can predict and adapt to unknown teammates on the fly. Conventional RL-based approaches optimize a single expected return, which often causes policies to collapse into a single dominant behavior, thus failing to capture the multimodal cooperation patterns inherent in AHT. In this work, we introduce PADiff, a diffusion-based approach that captures agent's multimodal behaviors, unlocking its diverse cooperation modes with teammates. However, standard diffusion models lack the ability to predict and adapt in highly non-stationary AHT scenarios. To address this limitation, we propose a novel diffusion-based policy that integrates critical predictive information about teammates into the denoising process. Extensive experiments across three cooperation environments demonstrate that PADiff outperforms existing AHT methods significantly. 

---
# Evaluating Online Moderation Via LLM-Powered Counterfactual Simulations 

**Authors**: Giacomo Fidone, Lucia Passaro, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2511.07204)  

**Abstract**: Online Social Networks (OSNs) widely adopt content moderation to mitigate the spread of abusive and toxic discourse. Nonetheless, the real effectiveness of moderation interventions remains unclear due to the high cost of data collection and limited experimental control. The latest developments in Natural Language Processing pave the way for a new evaluation approach. Large Language Models (LLMs) can be successfully leveraged to enhance Agent-Based Modeling and simulate human-like social behavior with unprecedented degree of believability. Yet, existing tools do not support simulation-based evaluation of moderation strategies. We fill this gap by designing a LLM-powered simulator of OSN conversations enabling a parallel, counterfactual simulation where toxic behavior is influenced by moderation interventions, keeping all else equal. We conduct extensive experiments, unveiling the psychological realism of OSN agents, the emergence of social contagion phenomena and the superior effectiveness of personalized moderation strategies. 

---
# Saliency Map-Guided Knowledge Discovery for Subclass Identification with LLM-Based Symbolic Approximations 

**Authors**: Tim Bohne, Anne-Kathrin Patricia Windler, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2511.07126)  

**Abstract**: This paper proposes a novel neuro-symbolic approach for sensor signal-based knowledge discovery, focusing on identifying latent subclasses in time series classification tasks. The approach leverages gradient-based saliency maps derived from trained neural networks to guide the discovery process. Multiclass time series classification problems are transformed into binary classification problems through label subsumption, and classifiers are trained for each of these to yield saliency maps. The input signals, grouped by predicted class, are clustered under three distinct configurations. The centroids of the final set of clusters are provided as input to an LLM for symbolic approximation and fuzzy knowledge graph matching to discover the underlying subclasses of the original multiclass problem. Experimental results on well-established time series classification datasets demonstrate the effectiveness of our saliency map-driven method for knowledge discovery, outperforming signal-only baselines in both clustering and subclass identification. 

---
# Two Heads are Better than One: Distilling Large Language Model Features Into Small Models with Feature Decomposition and Mixture 

**Authors**: Tianhao Fu, Xinxin Xu, Weichen Xu, Jue Chen, Ruilong Ren, Bowen Deng, Xinyu Zhao, Jian Cao, Xixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07110)  

**Abstract**: Market making (MM) through Reinforcement Learning (RL) has attracted significant attention in financial trading. With the development of Large Language Models (LLMs), more and more attempts are being made to apply LLMs to financial areas. A simple, direct application of LLM as an agent shows significant performance. Such methods are hindered by their slow inference speed, while most of the current research has not studied LLM distillation for this specific task. To address this, we first propose the normalized fluorescent probe to study the mechanism of the LLM's feature. Based on the observation found by our investigation, we propose Cooperative Market Making (CMM), a novel framework that decouples LLM features across three orthogonal dimensions: layer, task, and data. Various student models collaboratively learn simple LLM features along with different dimensions, with each model responsible for a distinct feature to achieve knowledge distillation. Furthermore, CMM introduces an Hájek-MoE to integrate the output of the student models by investigating the contribution of different models in a kernel function-generated common feature space. Extensive experimental results on four real-world market datasets demonstrate the superiority of CMM over the current distillation method and RL-based market-making strategies. 

---
# MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks 

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Guangze Ye, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2511.07107)  

**Abstract**: Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment. 

---
# A Theoretical Analysis of Detecting Large Model-Generated Time Series 

**Authors**: Junji Hou, Junzhou Zhao, Shuo Zhang, Pinghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07104)  

**Abstract**: Motivated by the increasing risks of data misuse and fabrication, we investigate the problem of identifying synthetic time series generated by Time-Series Large Models (TSLMs) in this work. While there are extensive researches on detecting model generated text, we find that these existing methods are not applicable to time series data due to the fundamental modality difference, as time series usually have lower information density and smoother probability distributions than text data, which limit the discriminative power of token-based detectors. To address this issue, we examine the subtle distributional differences between real and model-generated time series and propose the contraction hypothesis, which states that model-generated time series, unlike real ones, exhibit progressively decreasing uncertainty under recursive forecasting. We formally prove this hypothesis under theoretical assumptions on model behavior and time series structure. Model-generated time series exhibit progressively concentrated distributions under recursive forecasting, leading to uncertainty contraction. We provide empirical validation of the hypothesis across diverse datasets. Building on this insight, we introduce the Uncertainty Contraction Estimator (UCE), a white-box detector that aggregates uncertainty metrics over successive prefixes to identify TSLM-generated time series. Extensive experiments on 32 datasets show that UCE consistently outperforms state-of-the-art baselines, offering a reliable and generalizable solution for detecting model-generated time series. 

---
# Boosting Fine-Grained Urban Flow Inference via Lightweight Architecture and Focalized Optimization 

**Authors**: Yuanshao Zhu, Xiangyu Zhao, Zijian Zhang, Xuetao Wei, James Jianqiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07098)  

**Abstract**: Fine-grained urban flow inference is crucial for urban planning and intelligent transportation systems, enabling precise traffic management and resource allocation. However, the practical deployment of existing methods is hindered by two key challenges: the prohibitive computational cost of over-parameterized models and the suboptimal performance of conventional loss functions on the highly skewed distribution of urban flows. To address these challenges, we propose a unified solution that synergizes architectural efficiency with adaptive optimization. Specifically, we first introduce PLGF, a lightweight yet powerful architecture that employs a Progressive Local-Global Fusion strategy to effectively capture both fine-grained details and global contextual dependencies. Second, we propose DualFocal Loss, a novel function that integrates dual-space supervision with a difficulty-aware focusing mechanism, enabling the model to adaptively concentrate on hard-to-predict regions. Extensive experiments on 4 real-world scenarios validate the effectiveness and scalability of our method. Notably, while achieving state-of-the-art performance, PLGF reduces the model size by up to 97% compared to current high-performing methods. Furthermore, under comparable parameter budgets, our model yields an accuracy improvement of over 10% against strong baselines. The implementation is included in the this https URL. 

---
# Agentic AI Sustainability Assessment for Supply Chain Document Insights 

**Authors**: Diego Gosmar, Anna Chiara Pallotta, Giovanni Zenezini  

**Link**: [PDF](https://arxiv.org/pdf/2511.07097)  

**Abstract**: This paper presents a comprehensive sustainability assessment framework for document intelligence within supply chain operations, centered on agentic artificial intelligence (AI). We address the dual objective of improving automation efficiency while providing measurable environmental performance in document-intensive workflows. The research compares three scenarios: fully manual (human-only), AI-assisted (human-in-the-loop, HITL), and an advanced multi-agent agentic AI workflow leveraging parsers and verifiers. Empirical results show that AI-assisted HITL and agentic AI scenarios achieve reductions of up to 70-90% in energy consumption, 90-97% in carbon dioxide emissions, and 89-98% in water usage compared to manual processes. Notably, full agentic configurations, combining advanced reasoning (thinking mode) and multi-agent validation, achieve substantial sustainability gains over human-only approaches, even when resource usage increases slightly versus simpler AI-assisted solutions. The framework integrates performance, energy, and emission indicators into a unified ESG-oriented methodology for assessing and governing AI-enabled supply chain solutions. The paper includes a complete replicability use case demonstrating the methodology's application to real-world document extraction tasks. 

---
# Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics 

**Authors**: Meghyn Bienvenu, Quentin Manière  

**Link**: [PDF](https://arxiv.org/pdf/2511.07095)  

**Abstract**: In this paper, we study the data complexity of querying inconsistent weighted description logic (DL) knowledge bases under recently-introduced cost-based semantics. In a nutshell, the idea is to assign each interpretation a cost based upon the weights of the violated axioms and assertions, and certain and possible query answers are determined by considering all (resp. some) interpretations having optimal or bounded cost. Whereas the initial study of cost-based semantics focused on DLs between $\mathcal{EL}_\bot$ and $\mathcal{ALCO}$, we consider DLs that may contain inverse roles and role inclusions, thus covering prominent DL-Lite dialects. Our data complexity analysis goes significantly beyond existing results by sharpening several lower bounds and pinpointing the precise complexity of optimal-cost certain answer semantics (no non-trivial upper bound was known). Moreover, while all existing results show the intractability of cost-based semantics, our most challenging and surprising result establishes that if we consider $\text{DL-Lite}^\mathcal{H}_\mathsf{bool}$ ontologies and a fixed cost bound, certain answers for instance queries and possible answers for conjunctive queries can be computed using first-order rewriting and thus enjoy the lowest possible data complexity ($\mathsf{TC}_0$). 

---
# Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts 

**Authors**: Marcel Rojahn, Marcus Grum  

**Link**: [PDF](https://arxiv.org/pdf/2511.07090)  

**Abstract**: Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers. 

---
# LLM Driven Processes to Foster Explainable AI 

**Authors**: Marcel Pehlke, Marc Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07086)  

**Abstract**: We present a modular, explainable LLM-agent pipeline for decision support that externalizes reasoning into auditable artifacts. The system instantiates three frameworks: Vester's Sensitivity Model (factor set, signed impact matrix, systemic roles, feedback loops); normal-form games (strategies, payoff matrix, equilibria); and sequential games (role-conditioned agents, tree construction, backward induction), with swappable modules at every step. LLM components (default: GPT-5) are paired with deterministic analyzers for equilibria and matrix-based role classification, yielding traceable intermediates rather than opaque outputs. In a real-world logistics case (100 runs), mean factor alignment with a human baseline was 55.5\% over 26 factors and 62.9\% on the transport-core subset; role agreement over matches was 57\%. An LLM judge using an eight-criterion rubric (max 100) scored runs on par with a reconstructed human baseline. Configurable LLM pipelines can thus mimic expert workflows with transparent, inspectable steps. 

---
# Increasing AI Explainability by LLM Driven Standard Processes 

**Authors**: Marc Jansen, Marcel Pehlke  

**Link**: [PDF](https://arxiv.org/pdf/2511.07083)  

**Abstract**: This paper introduces an approach to increasing the explainability of artificial intelligence (AI) systems by embedding Large Language Models (LLMs) within standardized analytical processes. While traditional explainable AI (XAI) methods focus on feature attribution or post-hoc interpretation, the proposed framework integrates LLMs into defined decision models such as Question-Option-Criteria (QOC), Sensitivity Analysis, Game Theory, and Risk Management. By situating LLM reasoning within these formal structures, the approach transforms opaque inference into transparent and auditable decision traces. A layered architecture is presented that separates the reasoning space of the LLM from the explainable process space above it. Empirical evaluations show that the system can reproduce human-level decision logic in decentralized governance, systems analysis, and strategic reasoning contexts. The results suggest that LLM-driven standard processes provide a foundation for reliable, interpretable, and verifiable AI-supported decision making. 

---
# RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services 

**Authors**: Fei Zhao, Chonggang Lu, Haofu Qian, Fangcheng Shi, Zijie Meng, Jianzhao Huang, Xu Tang, Zheyong Xie, Zheyu Ye, Zhe Xu, Yao Hu, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07070)  

**Abstract**: As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervised fine-tuning (SFT) can specialize models but often triggers a ``seesaw'' between in-distribution gains and out-of-distribution robustness, especially for smaller models. To address these challenges, we introduce RedOne 2.0, an SNS-oriented LLM trained with a progressive, RL-prioritized post-training paradigm designed for rapid and stable adaptation. The pipeline consist in three stages: (1) Exploratory Learning on curated SNS corpora to establish initial alignment and identify systematic weaknesses; (2) Targeted Fine-Tuning that selectively applies SFT to the diagnosed gaps while mixing a small fraction of general data to mitigate forgetting; and (3) Refinement Learning that re-applies RL with SNS-centric signals to consolidate improvements and harmonize trade-offs across tasks. Across various tasks spanning three categories, our 4B scale model delivers an average improvements about 2.41 over the 7B sub-optimal baseline. Additionally, RedOne 2.0 achieves average performance lift about 8.74 from the base model with less than half the data required by SFT-centric method RedOne, evidencing superior data efficiency and stability at compact scales. Overall, RedOne 2.0 establishes a competitive, cost-effective baseline for domain-specific LLMs in SNS scenario, advancing capability without sacrificing robustness. 

---
# Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision 

**Authors**: Yimei Zhang, Guojiang Shen, Kaili Ning, Tongwei Ren, Xuebo Qiu, Mengmeng Wang, Xiangjie Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.07062)  

**Abstract**: Region representation learning plays a pivotal role in urban computing by extracting meaningful features from unlabeled urban data. Analogous to how perceived facial age reflects an individual's health, the visual appearance of a city serves as its ``portrait", encapsulating latent socio-economic and environmental characteristics. Recent studies have explored leveraging Large Language Models (LLMs) to incorporate textual knowledge into imagery-based urban region representation learning. However, two major challenges remain: i)~difficulty in aligning fine-grained visual features with long captions, and ii) suboptimal knowledge incorporation due to noise in LLM-generated captions. To address these issues, we propose a novel pre-training framework called UrbanLN that improves Urban region representation learning through Long-text awareness and Noise suppression. Specifically, we introduce an information-preserved stretching interpolation strategy that aligns long captions with fine-grained visual semantics in complex urban scenes. To effectively mine knowledge from LLM-generated captions and filter out noise, we propose a dual-level optimization strategy. At the data level, a multi-model collaboration pipeline automatically generates diverse and reliable captions without human intervention. At the model level, we employ a momentum-based self-distillation mechanism to generate stable pseudo-targets, facilitating robust cross-modal learning under noisy conditions. Extensive experiments across four real-world cities and various downstream tasks demonstrate the superior performance of our UrbanLN. 

---
# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07061)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets-- IEMOCAP and MELD --show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding. 

---
# Proceedings of the 2025 XCSP3 Competition 

**Authors**: Gilles Audemard, Christophe Lecoutre, Emmanuel Lonca  

**Link**: [PDF](https://arxiv.org/pdf/2511.06918)  

**Abstract**: This document represents the proceedings of the 2025 XCSP3 Competition. The results of this competition of constraint solvers were presented at CP'25 (31st International Conference on Principles and Practice of Constraint Programming). 

---
# MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning 

**Authors**: Jinhao Chen, Zhen Yang, Jianxin Shi, Tianyu Wo, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06805)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in vision-language answering tasks. Despite their strengths, these models often encounter challenges in achieving complex reasoning tasks such as mathematical problem-solving. Previous works have focused on fine-tuning on specialized mathematical datasets. However, these datasets are typically distilled directly from teacher models, which capture only static reasoning patterns and leaving substantial gaps compared to student models. This reliance on fixed teacher-derived datasets not only restricts the model's ability to adapt to novel or more intricate questions that extend beyond the confines of the training data, but also lacks the iterative depth needed for robust generalization. To overcome these limitations, we propose \textbf{\method}, a \textbf{Math}ematical \textbf{S}elf-\textbf{E}volving framework for MLLMs. In contrast to traditional one-shot fine-tuning paradigms, \method iteratively refines the model through cycles of inference, reflection, and reward-based feedback. Specifically, we leverage iterative fine-tuning by incorporating correct reasoning paths derived from previous-stage inference and integrating reflections from a specialized Outcome Reward Model (ORM). To verify the effectiveness of \method, we evaluate it on a suite of challenging benchmarks, demonstrating significant performance gains over backbone models. Notably, our experimental results on MathVL-test surpass the leading open-source multimodal mathematical reasoning model QVQ. Our code and models are available at \texttt{https://zheny2751\this http URL\allowbreak this http URL}. 

---
# SRNN: Spatiotemporal Relational Neural Network for Intuitive Physics Understanding 

**Authors**: Fei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06761)  

**Abstract**: Human prowess in intuitive physics remains unmatched by machines. To bridge this gap, we argue for a fundamental shift towards brain-inspired computational principles. This paper introduces the Spatiotemporal Relational Neural Network (SRNN), a model that establishes a unified neural representation for object attributes, relations, and timeline, with computations governed by a Hebbian ``Fire Together, Wire Together'' mechanism across dedicated \textit{What} and \textit{How} pathways. This unified representation is directly used to generate structured linguistic descriptions of the visual scene, bridging perception and language within a shared neural substrate. Moreover, unlike the prevalent ``pretrain-then-finetune'' paradigm, SRNN adopts a ``predefine-then-finetune'' approach. On the CLEVRER benchmark, SRNN achieves competitive performance. Our analysis further reveals a benchmark bias, outlines a path for a more holistic evaluation, and demonstrates SRNN's white-box utility for precise error diagnosis. Our work confirms the viability of translating biological intelligence into engineered systems for intuitive physics understanding. 

---
# Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives 

**Authors**: Chloe Li, Mary Phuong, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06626)  

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to admit their factual mistakes when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AI systems. 

---
# GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization 

**Authors**: Moriya Dechtiar, Daniel Martin Katz, Mari Sundaresan, Sylvain Jaume, Hongming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06618)  

**Abstract**: Contracts are complex documents featuring detailed formal structures, explicit and implicit dependencies and rich semantic content. Given these document properties, contract drafting and manual examination of contracts have proven to be both arduous and susceptible to errors. This work aims to simplify and automate the task of contract review and analysis using a novel framework for transforming legal contracts into structured semantic graphs, enabling computational analysis and data-driven insights. We introduce a detailed ontology mapping core legal contract elements to their graph-theoretic equivalents of nodes and edges. We then present a reinforcement learning based Large Language Model (LLM) framework for segmentation and extraction of entities and relationships from contracts. Our method, GRAPH-GRPO-LEX, incorporates both LLMs and reinforcement learning with group relative policy optimization (GRPO). By applying a carefully drafted reward function of graph metrics, we demonstrate the ability to automatically identify direct relationships between clauses, and even uncover hidden dependencies. Our introduction of the gated GRPO approach shows a strong learning signal and can move contract analysis from a linear, manual reading process to an easily visualized graph. This allows for a more dynamic analysis, including building the groundwork for contract linting similar to what is now practiced in software engineering. 

---
# FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis 

**Authors**: Jan Ondras, Marek Šuppa  

**Link**: [PDF](https://arxiv.org/pdf/2511.06522)  

**Abstract**: Mathematical reasoning requires abstracting symbolic rules from visual patterns -- inferring the infinite from the finite. We investigate whether multimodal AI systems possess this capability through FractalBench, a benchmark evaluating fractal program synthesis from images. Fractals provide ideal test cases: Iterated Function Systems with only a few contraction maps generate complex self-similar patterns through simple recursive rules, requiring models to bridge visual perception with mathematical abstraction. We evaluate four leading MLLMs -- GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Flash, and Qwen 2.5-VL -- on 12 canonical fractals. Models must generate executable Python code reproducing the fractal, enabling objective evaluation. Results reveal a striking disconnect: 76% generate syntactically valid code but only 4% capture mathematical structure. Success varies systematically -- models handle geometric transformations (Koch curves: 17-21%) but fail at branching recursion (trees: <2%), revealing fundamental gaps in mathematical abstraction. FractalBench provides a contamination-resistant diagnostic for visual-mathematical reasoning and is available at this https URL 

---
# GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets 

**Authors**: Jingtao Tang, Hang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.06471)  

**Abstract**: We study GCS-TSP, a new variant of the Traveling Salesman Problem (TSP) defined over a Graph of Convex Sets (GCS) -- a powerful representation for trajectory planning that decomposes the configuration space into convex regions connected by a sparse graph. In this setting, edge costs are not fixed but depend on the specific trajectory selected through each convex region, making classical TSP methods inapplicable. We introduce GHOST, a hierarchical framework that optimally solves the GCS-TSP by combining combinatorial tour search with convex trajectory optimization. GHOST systematically explores tours on a complete graph induced by the GCS, using a novel abstract-path-unfolding algorithm to compute admissible lower bounds that guide best-first search at both the high level (over tours) and the low level (over feasible GCS paths realizing the tour). These bounds provide strong pruning power, enabling efficient search while avoiding unnecessary convex optimization calls. We prove that GHOST guarantees optimality and present a bounded-suboptimal variant for time-critical scenarios. Experiments show that GHOST is orders-of-magnitude faster than unified mixed-integer convex programming baselines for simple cases and uniquely handles complex trajectory planning problems involving high-order continuity constraints and an incomplete GCS. 

---
# Brain-Inspired Planning for Better Generalization in Reinforcement Learning 

**Authors**: Mingde "Harry" Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06470)  

**Abstract**: Existing Reinforcement Learning (RL) systems encounter significant challenges when applied to real-world scenarios, primarily due to poor generalization across environments that differ from their training conditions. This thesis explores the direction of enhancing agents' zero-shot systematic generalization abilities by granting RL agents reasoning behaviors that are found to help systematic generalization in the human brain. Inspired by human conscious planning behaviors, we first introduced a top-down attention mechanism, which allows a decision-time planning agent to dynamically focus its reasoning on the most relevant aspects of the environmental state given its instantaneous intentions, a process we call "spatial abstraction". This approach significantly improves systematic generalization outside the training tasks. Subsequently, building on spatial abstraction, we developed the Skipper framework to automatically decompose complex tasks into simpler, more manageable sub-tasks. Skipper provides robustness against distributional shifts and efficacy in long-term, compositional planning by focusing on pertinent spatial and temporal elements of the environment. Finally, we identified a common failure mode and safety risk in planning agents that rely on generative models to generate state targets during planning. It is revealed that most agents blindly trust the targets they hallucinate, resulting in delusional planning behaviors. Inspired by how the human brain rejects delusional intentions, we propose learning a feasibility evaluator to enable rejecting hallucinated infeasible targets, which led to significant performance improvements in various kinds of planning agents. Finally, we suggest directions for future research, aimed at achieving general task abstraction and fully enabling abstract planning. 

---
# Optimizing Chain-of-Thought Confidence via Topological and Dirichlet Risk Analysis 

**Authors**: Abhishek More, Anthony Zhang, Nicole Bonilla, Ashvik Vivekan, Kevin Zhu, Parham Sharafoleslami, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.06437)  

**Abstract**: Chain-of-thought (CoT) prompting enables Large Language Models to solve complex problems, but deploying these models safely requires reliable confidence estimates, a capability where existing methods suffer from poor calibration and severe overconfidence on incorrect predictions. We propose Enhanced Dirichlet and Topology Risk (EDTR), a novel decoding strategy that combines topological analysis with Dirichlet-based uncertainty quantification to measure LLM confidence across multiple reasoning paths. EDTR treats each CoT as a vector in high-dimensional space and extracts eight topological risk features capturing the geometric structure of reasoning distributions: tighter, more coherent clusters indicate higher confidence while dispersed, inconsistent paths signal uncertainty. We evaluate EDTR against three state-of-the-art calibration methods across four diverse reasoning benchmarks spanning olympiad-level mathematics (AIME), grade school math (GSM8K), commonsense reasoning, and stock price prediction \cite{zhang2025aime, cobbe2021training, talmor-etal-2019-commonsenseqa, yahoo_finance}. EDTR achieves 41\% better calibration than competing methods with an average ECE of 0.287 and the best overall composite score of 0.672, while notably achieving perfect accuracy on AIME and exceptional calibration on GSM8K with an ECE of 0.107, domains where baselines exhibit severe overconfidence. Our work provides a geometric framework for understanding and quantifying uncertainty in multi-step LLM reasoning, enabling more reliable deployment where calibrated confidence estimates are essential. 

---
# MONICA: Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models 

**Authors**: Jingyu Hu, Shu Yang, Xilin Gong, Hongming Wang, Weiru Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06419)  

**Abstract**: Large Reasoning Models (LRMs) suffer from sycophantic behavior, where models tend to agree with users' incorrect beliefs and follow misinformation rather than maintain independent reasoning. This behavior undermines model reliability and poses societal risks. Mitigating LRM sycophancy requires monitoring how this sycophancy emerges during the reasoning trajectory; however, current methods mainly focus on judging based on final answers and correcting them, without understanding how sycophancy develops during reasoning processes. To address this limitation, we propose MONICA, a novel Monitor-guided Calibration framework that monitors and mitigates sycophancy during model inference at the level of reasoning steps, without requiring the model to finish generating its complete answer. MONICA integrates a sycophantic monitor that provides real-time monitoring of sycophantic drift scores during response generation with a calibrator that dynamically suppresses sycophantic behavior when scores exceed predefined thresholds. Extensive experiments across 12 datasets and 3 LRMs demonstrate that our method effectively reduces sycophantic behavior in both intermediate reasoning steps and final answers, yielding robust performance improvements. 

---
# AUTO-Explorer: Automated Data Collection for GUI Agent 

**Authors**: Xiangwu Guo, Difei Gao, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06417)  

**Abstract**: Recent advancements in GUI agents have significantly expanded their ability to interpret natural language commands to manage software interfaces. However, acquiring GUI data remains a significant challenge. Existing methods often involve designing automated agents that browse URLs from the Common Crawl, using webpage HTML to collect screenshots and corresponding annotations, including the names and bounding boxes of UI elements. However, this method is difficult to apply to desktop software or some newly launched websites not included in the Common Crawl. While we expect the model to possess strong generalization capabilities to handle this, it is still crucial for personalized scenarios that require rapid and perfect adaptation to new software or websites. To address this, we propose an automated data collection method with minimal annotation costs, named Auto-Explorer. It incorporates a simple yet effective exploration mechanism that autonomously parses and explores GUI environments, gathering data efficiently. Additionally, to assess the quality of exploration, we have developed the UIXplore benchmark. This benchmark creates environments for explorer agents to discover and save software states. Using the data gathered, we fine-tune a multimodal large language model (MLLM) and establish a GUI element grounding testing set to evaluate the effectiveness of the exploration strategies. Our experiments demonstrate the superior performance of Auto-Explorer, showing that our method can quickly enhance the capabilities of an MLLM in explored software. 

---
# SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.06411)  

**Abstract**: The soft-thinking paradigm for Large Language Model (LLM) reasoning can outperform the conventional discrete-token Chain-of-Thought (CoT) reasoning in some scenarios, underscoring its research and application value. However, while the discrete-token CoT reasoning pattern can be reinforced through policy optimization algorithms such as group relative policy optimization (GRPO), extending the soft-thinking pattern with Reinforcement Learning (RL) remains challenging. This difficulty stems from the complexities of injecting stochasticity into soft-thinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with GRPO typically underperform their discrete-token GRPO counterparts. To fully unlock the potential of soft-thinking, this paper presents a novel policy optimization algorithm, SofT-GRPO, to reinforce LLMs under the soft-thinking reasoning pattern. SofT-GRPO injects the Gumbel noise into logits, employs the Gumbel-Softmax technique to avoid soft-thinking tokens outside the pre-trained embedding space, and leverages the reparameterization trick in policy gradient. We conduct experiments across base LLMs ranging from 1.5B to 7B parameters, and results demonstrate that SofT-GRPO enables soft-thinking LLMs to slightly outperform discrete-token GRPO on Pass@1 (+0.13% on average accuracy), while exhibiting a substantial uplift on Pass@32 (+2.19% on average accuracy). Codes and weights are available on this https URL 

---
# Efficient LLM Safety Evaluation through Multi-Agent Debate 

**Authors**: Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06396)  

**Abstract**: Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-Judge frameworks, but the high cost of frontier models limits scalability. We propose a cost-efficient multi-agent judging framework that employs Small Language Models (SLMs) through structured debates among critic, defender, and judge agents. To rigorously assess safety judgments, we construct HAJailBench, a large-scale human-annotated jailbreak benchmark comprising 12,000 adversarial interactions across diverse attack methods and target models. The dataset provides fine-grained, expert-labeled ground truth for evaluating both safety robustness and judge reliability. Our SLM-based framework achieves agreement comparable to GPT-4o judges on HAJailBench while substantially reducing inference cost. Ablation results show that three rounds of debate yield the optimal balance between accuracy and efficiency. These findings demonstrate that structured, value-aligned debate enables SLMs to capture semantic nuances of jailbreak attacks and that HAJailBench offers a reliable foundation for scalable LLM safety evaluation. 

---
# What Makes Reasoning Invalid: Echo Reflection Mitigation for Large Language Models 

**Authors**: Chen He, Xun Jiang, Lei Wang, Hao Yang, Chong Peng, Peng Yan, Fumin Shen, Xing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06380)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of reasoning tasks. Recent methods have further improved LLM performance in complex mathematical reasoning. However, when extending these methods beyond the domain of mathematical reasoning to tasks involving complex domain-specific knowledge, we observe a consistent failure of LLMs to generate novel insights during the reflection stage. Instead of conducting genuine cognitive refinement, the model tends to mechanically reiterate earlier reasoning steps without introducing new information or perspectives, a phenomenon referred to as "Echo Reflection". We attribute this behavior to two key defects: (1) Uncontrollable information flow during response generation, which allows premature intermediate thoughts to propagate unchecked and distort final decisions; (2) Insufficient exploration of internal knowledge during reflection, leading to repeating earlier findings rather than generating new cognitive insights. Building on these findings, we proposed a novel reinforcement learning method termed Adaptive Entropy Policy Optimization (AEPO). Specifically, the AEPO framework consists of two major components: (1) Reflection-aware Information Filtration, which quantifies the cognitive information flow and prevents the final answer from being affected by earlier bad cognitive information; (2) Adaptive-Entropy Optimization, which dynamically balances exploration and exploitation across different reasoning stages, promoting both reflective diversity and answer correctness. Extensive experiments demonstrate that AEPO consistently achieves state-of-the-art performance over mainstream reinforcement learning baselines across diverse benchmarks. 

---
# LPFQA: A Long-Tail Professional Forum-based Benchmark for LLM Evaluation 

**Authors**: Liya Zhu, Peizhuang Cong, Aowei Ji, Wenya Wu, Jiani Hou, Chunjie Wu, Xiang Gao, Jingkai Liu, Zhou Huan, Xuelei Sun, Yang Yang, Jianpeng Jiao, Liang Hu, Xinjie Chen, Jiashuo Liu, Jingzhe Ding, Tong Yang, Zaiyuan Wang, Ge Zhang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06346)  

**Abstract**: Large Language Models (LLMs) have made rapid progress in reasoning, question answering, and professional applications; however, their true capabilities remain difficult to evaluate using existing benchmarks. Current datasets often focus on simplified tasks or artificial scenarios, overlooking long-tail knowledge and the complexities of real-world applications. To bridge this gap, we propose LPFQA, a long-tail knowledge-based benchmark derived from authentic professional forums across 20 academic and industrial fields, covering 502 tasks grounded in practical expertise. LPFQA introduces four key innovations: fine-grained evaluation dimensions that target knowledge depth, reasoning, terminology comprehension, and contextual analysis; a hierarchical difficulty structure that ensures semantic clarity and unique answers; authentic professional scenario modeling with realistic user personas; and interdisciplinary knowledge integration across diverse domains. We evaluated 12 mainstream LLMs on LPFQA and observed significant performance disparities, especially in specialized reasoning tasks. LPFQA provides a robust, authentic, and discriminative benchmark for advancing LLM evaluation and guiding future model development. 

---
# ALIGN: A Vision-Language Framework for High-Accuracy Accident Location Inference through Geo-Spatial Neural Reasoning 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.06316)  

**Abstract**: Reliable geospatial information on road accidents is vital for safety analysis and infrastructure planning, yet most low- and middle-income countries continue to face a critical shortage of accurate, location-specific crash data. Existing text-based geocoding tools perform poorly in multilingual and unstructured news environments, where incomplete place descriptions and mixed Bangla-English scripts obscure spatial context. To address these limitations, this study introduces ALIGN (Accident Location Inference through Geo-Spatial Neural Reasoning)- a vision-language framework that emulates human spatial reasoning to infer accident coordinates directly from textual and map-based cues. ALIGN integrates large language and vision-language models within a multi-stage pipeline that performs optical character recognition, linguistic reasoning, and map-level verification through grid-based spatial scanning. The framework systematically evaluates each predicted location against contextual and visual evidence, ensuring interpretable, fine-grained geolocation outcomes without requiring model retraining. Applied to Bangla-language news data, ALIGN demonstrates consistent improvements over traditional geoparsing methods, accurately identifying district and sub-district-level crash sites. Beyond its technical contribution, the framework establishes a high accuracy foundation for automated crash mapping in data-scarce regions, supporting evidence-driven road-safety policymaking and the broader integration of multimodal artificial intelligence in transportation analytics. The code for this paper is open-source and available at: this https URL 

---
# The Station: An Open-World Environment for AI-Driven Discovery 

**Authors**: Stephen Chung, Wenyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06309)  

**Abstract**: We introduce the STATION, an open-world multi-agent environment that models a miniature scientific ecosystem. Leveraging their extended context windows, agents in the Station can engage in long scientific journeys that include reading papers from peers, formulating hypotheses, submitting code, performing analyses, and publishing results. Importantly, there is no centralized system coordinating their activities - agents are free to choose their own actions and develop their own narratives within the Station. Experiments demonstrate that AI agents in the Station achieve new state-of-the-art performance on a wide range of benchmarks, spanning from mathematics to computational biology to machine learning, notably surpassing AlphaEvolve in circle packing. A rich tapestry of narratives emerges as agents pursue independent research, interact with peers, and build upon a cumulative history. From these emergent narratives, novel methods arise organically, such as a new density-adaptive algorithm for scRNA-seq batch integration. The Station marks a first step towards autonomous scientific discovery driven by emergent behavior in an open-world environment, representing a new paradigm that moves beyond rigid optimization. 

---
# Secu-Table: a Comprehensive security table dataset for evaluating semantic table interpretation systems 

**Authors**: Azanzi Jiomekong, Jean Bikim, Patricia Negoue, Joyce Chin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06301)  

**Abstract**: Evaluating semantic tables interpretation (STI) systems, (particularly, those based on Large Language Models- LLMs) especially in domain-specific contexts such as the security domain, depends heavily on the dataset. However, in the security domain, tabular datasets for state-of-the-art are not publicly available. In this paper, we introduce Secu-Table dataset, composed of more than 1500 tables with more than 15k entities constructed using security data extracted from Common Vulnerabilities and Exposures (CVE) and Common Weakness Enumeration (CWE) data sources and annotated using Wikidata and the SEmantic Processing of Security Event Streams CyberSecurity Knowledge Graph (SEPSES CSKG). Along with the dataset, all the code is publicly released. This dataset is made available to the research community in the context of the SemTab challenge on Tabular to Knowledge Graph Matching. This challenge aims to evaluate the performance of several STI based on open source LLMs. Preliminary evaluation, serving as baseline, was conducted using Falcon3-7b-instruct and Mistral-7B-Instruct, two open source LLMs and GPT-4o mini one closed source LLM. 

---
# Synthetic Data-Driven Prompt Tuning for Financial QA over Tables and Documents 

**Authors**: Yaoning Yu, Kaimin Chang, Ye Yu, Kai Wei, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06292)  

**Abstract**: Financial documents like earning reports or balance sheets often involve long tables and multi-page reports. Large language models have become a new tool to help numerical reasoning and understanding these documents. However, prompt quality can have a major effect on how well LLMs perform these financial reasoning tasks. Most current methods tune prompts on fixed datasets of financial text or tabular data, which limits their ability to adapt to new question types or document structures, or they involve costly and manually labeled/curated dataset to help build the prompts. We introduce a self-improving prompt framework driven by data-augmented optimization. In this closed-loop process, we generate synthetic financial tables and document excerpts, verify their correctness and robustness, and then update the prompt based on the results. Specifically, our framework combines a synthetic data generator with verifiers and a prompt optimizer, where the generator produces new examples that exposes weaknesses in the current prompt, the verifiers check the validity and robustness of the produced examples, and the optimizer incrementally refines the prompt in response. By iterating these steps in a feedback cycle, our method steadily improves prompt accuracy on financial reasoning tasks without needing external labels. Evaluation on DocMath-Eval benchmark demonstrates that our system achieves higher performance in both accuracy and robustness than standard prompt methods, underscoring the value of incorporating synthetic data generation into prompt learning for financial applications. 

---
# GAIA: A General Agency Interaction Architecture for LLM-Human B2B Negotiation & Screening 

**Authors**: Siming Zhao, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06262)  

**Abstract**: Organizations are increasingly exploring delegation of screening and negotiation tasks to AI systems, yet deployment in high-stakes B2B settings is constrained by governance: preventing unauthorized commitments, ensuring sufficient information before bargaining, and maintaining effective human oversight and auditability. Prior work on large language model negotiation largely emphasizes autonomous bargaining between agents and omits practical needs such as staged information gathering, explicit authorization boundaries, and systematic feedback integration. We propose GAIA, a governance-first framework for LLM-human agency in B2B negotiation and screening. GAIA defines three essential roles - Principal (human), Delegate (LLM agent), and Counterparty - with an optional Critic to enhance performance, and organizes interactions through three mechanisms: information-gated progression that separates screening from negotiation; dual feedback integration that combines AI critique with lightweight human corrections; and authorization boundaries with explicit escalation paths. Our contributions are fourfold: (1) a formal governance framework with three coordinated mechanisms and four safety invariants for delegation with bounded authorization; (2) information-gated progression via task-completeness tracking (TCI) and explicit state transitions that separate screening from commitment; (3) dual feedback integration that blends Critic suggestions with human oversight through parallel learning channels; and (4) a hybrid validation blueprint that combines automated protocol metrics with human judgment of outcomes and safety. By bridging theory and practice, GAIA offers a reproducible specification for safe, efficient, and accountable AI delegation that can be instantiated across procurement, real estate, and staffing workflows. 

---
# ROAR: Robust Accident Recognition and Anticipation for Autonomous Driving 

**Authors**: Xingcheng Liu, Yanchen Guan, Haicheng Liao, Zhengbing He, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06226)  

**Abstract**: Accurate accident anticipation is essential for enhancing the safety of autonomous vehicles (AVs). However, existing methods often assume ideal conditions, overlooking challenges such as sensor failures, environmental disturbances, and data imperfections, which can significantly degrade prediction accuracy. Additionally, previous models have not adequately addressed the considerable variability in driver behavior and accident rates across different vehicle types. To overcome these limitations, this study introduces ROAR, a novel approach for accident detection and prediction. ROAR combines Discrete Wavelet Transform (DWT), a self adaptive object aware module, and dynamic focal loss to tackle these challenges. The DWT effectively extracts features from noisy and incomplete data, while the object aware module improves accident prediction by focusing on high-risk vehicles and modeling the spatial temporal relationships among traffic agents. Moreover, dynamic focal loss mitigates the impact of class imbalance between positive and negative samples. Evaluated on three widely used datasets, Dashcam Accident Dataset (DAD), Car Crash Dataset (CCD), and AnAn Accident Detection (A3D), our model consistently outperforms existing baselines in key metrics such as Average Precision (AP) and mean Time to Accident (mTTA). These results demonstrate the model's robustness in real-world conditions, particularly in handling sensor degradation, environmental noise, and imbalanced data distributions. This work offers a promising solution for reliable and accurate accident anticipation in complex traffic environments. 

---
# Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B 

**Authors**: Sen Xu, Yi Zhou, Wei Wang, Jixin Min, Zhibin Yin, Yingwei Dai, Shixi Liu, Lianyu Pang, Yirong Chen, Junlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06221)  

**Abstract**: Challenging the prevailing consensus that small models inherently lack robust reasoning, this report introduces VibeThinker-1.5B, a 1.5B-parameter dense model developed via our Spectrum-to-Signal Principle (SSP). This challenges the prevailing approach of scaling model parameters to enhance capabilities, as seen in models like DeepSeek R1 (671B) and Kimi k2 (>1T). The SSP framework first employs a Two-Stage Diversity-Exploring Distillation (SFT) to generate a broad spectrum of solutions, followed by MaxEnt-Guided Policy Optimization (RL) to amplify the correct signal. With a total training cost of only $7,800, VibeThinker-1.5B demonstrates superior reasoning capabilities compared to closed-source models like Magistral Medium and Claude Opus 4, and performs on par with open-source models like GPT OSS-20B Medium. Remarkably, it surpasses the 400x larger DeepSeek R1 on three math benchmarks: AIME24 (80.3 vs. 79.8), AIME25 (74.4 vs. 70.0), and HMMT25 (50.4 vs. 41.7). This is a substantial improvement over its base model (6.7, 4.3, and 0.6, respectively). On LiveCodeBench V6, it scores 51.1, outperforming Magistral Medium's 50.3 and its base model's 0.0. These findings demonstrate that small models can achieve reasoning capabilities comparable to large models, drastically reducing training and inference costs and thereby democratizing advanced AI research. 

---
# Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads 

**Authors**: Jingwei Ni, Ekaterina Fadeeva, Tianyi Wu, Mubashara Akhtar, Jiaheng Zhang, Elliott Ash, Markus Leippold, Timothy Baldwin, See-Kiong Ng, Artem Shelmanov, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06209)  

**Abstract**: Solving complex tasks usually requires LLMs to generate long multi-step reasoning chains. Previous work has shown that verifying the correctness of individual reasoning steps can further improve the performance and efficiency of LLMs on such tasks and enhance solution interpretability. However, existing verification approaches, such as Process Reward Models (PRMs), are either computationally expensive, limited to specific domains, or require large-scale human or model-generated annotations. Thus, we propose a lightweight alternative for step-level reasoning verification based on data-driven uncertainty scores. We train transformer-based uncertainty quantification heads (UHeads) that use the internal states of a frozen LLM to estimate the uncertainty of its reasoning steps during generation. The approach is fully automatic: target labels are generated either by another larger LLM (e.g., DeepSeek R1) or in a self-supervised manner by the original model itself. UHeads are both effective and lightweight, containing less than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, they match or even surpass the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their uncertainty and can serve as reliable signals for reasoning verification, offering a promising direction toward scalable and generalizable introspective LLMs. 

---
# Dataforge: A Data Agent Platform for Autonomous Data Engineering 

**Authors**: Xinyuan Wang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06185)  

**Abstract**: The growing demand for AI applications in fields such as materials discovery, molecular modeling, and climate science has made data preparation an important but labor-intensive step. Raw data from diverse sources must be cleaned, normalized, and transformed to become AI-ready, while effective feature transformation and selection are essential for efficient training and inference. To address the challenges of scalability and expertise dependence, we present Data Agent, a fully autonomous system specialized for tabular data. Leveraging large language model (LLM) reasoning and grounded validation, Data Agent automatically performs data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops. It embodies three core principles: automatic, safe, and non-expert friendly, which ensure end-to-end reliability without human supervision. This demo showcases the first practical realization of an autonomous Data Agent, illustrating how raw data can be transformed "From Data to Better Data." 

---
# CSP4SDG: Constraint and Information-Theory Based Role Identification in Social Deduction Games with LLM-Enhanced Inference 

**Authors**: Kaijie Xu, Fandi Meng, Clark Verbrugge, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2511.06175)  

**Abstract**: In Social Deduction Games (SDGs) such as Avalon, Mafia, and Werewolf, players conceal their identities and deliberately mislead others, making hidden-role inference a central and demanding task. Accurate role identification, which forms the basis of an agent's belief state, is therefore the keystone for both human and AI performance. We introduce CSP4SDG, a probabilistic, constraint-satisfaction framework that analyses gameplay objectively. Game events and dialogue are mapped to four linguistically-agnostic constraint classes-evidence, phenomena, assertions, and hypotheses. Hard constraints prune impossible role assignments, while weighted soft constraints score the remainder; information-gain weighting links each hypothesis to its expected value under entropy reduction, and a simple closed-form scoring rule guarantees that truthful assertions converge to classical hard logic with minimum error. The resulting posterior over roles is fully interpretable and updates in real time. Experiments on three public datasets show that CSP4SDG (i) outperforms LLM-based baselines in every inference scenario, and (ii) boosts LLMs when supplied as an auxiliary "reasoning tool." Our study validates that principled probabilistic reasoning with information theory is a scalable alternative-or complement-to heavy-weight neural models for SDGs. 

---
# Chasing Consistency: Quantifying and Optimizing Human-Model Alignment in Chain-of-Thought Reasoning 

**Authors**: Boxuan Wang, Zhuoyun Li, Xinmiao Huang, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06168)  

**Abstract**: This paper presents a framework for evaluating and optimizing reasoning consistency in Large Language Models (LLMs) via a new metric, the Alignment Score, which quantifies the semantic alignment between model-generated reasoning chains and human-written reference chains in Chain-of-Thought (CoT) reasoning. Empirically, we find that 2-hop reasoning chains achieve the highest Alignment Score. To explain this phenomenon, we define four key error types: logical disconnection, thematic shift, redundant reasoning, and causal reversal, and show how each contributes to the degradation of the Alignment Score. Building on this analysis, we further propose Semantic Consistency Optimization Sampling (SCOS), a method that samples and favors chains with minimal alignment errors, significantly improving Alignment Scores by an average of 29.84% with longer reasoning chains, such as in 3-hop tasks. 

---
# Evaluating Implicit Biases in LLM Reasoning through Logic Grid Puzzles 

**Authors**: Fatima Jahara, Mark Dredze, Sharon Levy  

**Link**: [PDF](https://arxiv.org/pdf/2511.06160)  

**Abstract**: While recent safety guardrails effectively suppress overtly biased outputs, subtler forms of social bias emerge during complex logical reasoning tasks that evade current evaluation benchmarks. To fill this gap, we introduce a new evaluation framework, PRIME (Puzzle Reasoning for Implicit Biases in Model Evaluation), that uses logic grid puzzles to systematically probe the influence of social stereotypes on logical reasoning and decision making in LLMs. Our use of logic puzzles enables automatic generation and verification, as well as variability in complexity and biased settings. PRIME includes stereotypical, anti-stereotypical, and neutral puzzle variants generated from a shared puzzle structure, allowing for controlled and fine-grained comparisons. We evaluate multiple model families across puzzle sizes and test the effectiveness of prompt-based mitigation strategies. Focusing our experiments on gender stereotypes, our findings highlight that models consistently reason more accurately when solutions align with stereotypical associations. This demonstrates the significance of PRIME for diagnosing and quantifying social biases perpetuated in the deductive reasoning of LLMs, where fairness is critical. 

---
# MALinZero: Efficient Low-Dimensional Search for Mastering Complex Multi-Agent Planning 

**Authors**: Sizhe Tang, Jiayu Chen, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06142)  

**Abstract**: Monte Carlo Tree Search (MCTS), which leverages Upper Confidence Bound for Trees (UCTs) to balance exploration and exploitation through randomized sampling, is instrumental to solving complex planning problems. However, for multi-agent planning, MCTS is confronted with a large combinatorial action space that often grows exponentially with the number of agents. As a result, the branching factor of MCTS during tree expansion also increases exponentially, making it very difficult to efficiently explore and exploit during tree search. To this end, we propose MALinZero, a new approach to leverage low-dimensional representational structures on joint-action returns and enable efficient MCTS in complex multi-agent planning. Our solution can be viewed as projecting the joint-action returns into the low-dimensional space representable using a contextual linear bandit problem formulation. We solve the contextual linear bandit problem with convex and $\mu$-smooth loss functions -- in order to place more importance on better joint actions and mitigate potential representational limitations -- and derive a linear Upper Confidence Bound applied to trees (LinUCT) to enable novel multi-agent exploration and exploitation in the low-dimensional space. We analyze the regret of MALinZero for low-dimensional reward functions and propose an $(1-\tfrac1e)$-approximation algorithm for the joint action selection by maximizing a sub-modular objective. MALinZero demonstrates state-of-the-art performance on multi-agent benchmarks such as matrix games, SMAC, and SMACv2, outperforming both model-based and model-free multi-agent reinforcement learning baselines with faster learning speed and better performance. 

---
# When Object-Centric World Models Meet Policy Learning: From Pixels to Policies, and Where It Breaks 

**Authors**: Stefano Ferraro, Akihiro Nakano, Masahiro Suzuki, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06136)  

**Abstract**: Object-centric world models (OCWM) aim to decompose visual scenes into object-level representations, providing structured abstractions that could improve compositional generalization and data efficiency in reinforcement learning. We hypothesize that explicitly disentangled object-level representations, by localizing task-relevant information, can enhance policy performance across novel feature combinations. To test this hypothesis, we introduce DLPWM, a fully unsupervised, disentangled object-centric world model that learns object-level latents directly from pixels. DLPWM achieves strong reconstruction and prediction performance, including robustness to several out-of-distribution (OOD) visual variations. However, when used for downstream model-based control, policies trained on DLPWM latents underperform compared to DreamerV3. Through latent-trajectory analyses, we identify representation shift during multi-object interactions as a key driver of unstable policy learning. Our results suggest that, although object-centric perception supports robust visual modeling, achieving stable control requires mitigating latent drift. 

---
# Maestro: Learning to Collaborate via Conditional Listwise Policy Optimization for Multi-Agent LLMs 

**Authors**: Wei Yang, Jiacheng Pang, Shixuan Li, Paul Bogdan, Stephen Tu, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2511.06134)  

**Abstract**: Multi-agent systems (MAS) built on Large Language Models (LLMs) are being used to approach complex problems and can surpass single model inference. However, their success hinges on navigating a fundamental cognitive tension: the need to balance broad, divergent exploration of the solution space with a principled, convergent synthesis to the optimal solution. Existing paradigms often struggle to manage this duality, leading to premature consensus, error propagation, and a critical credit assignment problem that fails to distinguish between genuine reasoning and superficially plausible arguments. To resolve this core challenge, we propose the Multi-Agent Exploration-Synthesis framework Through Role Orchestration (Maestro), a principled paradigm for collaboration that structurally decouples these cognitive modes. Maestro uses a collective of parallel Execution Agents for diverse exploration and a specialized Central Agent for convergent, evaluative synthesis. To operationalize this critical synthesis phase, we introduce Conditional Listwise Policy Optimization (CLPO), a reinforcement learning objective that disentangles signals for strategic decisions and tactical rationales. By combining decision-focused policy gradients with a list-wise ranking loss over justifications, CLPO achieves clean credit assignment and stronger comparative supervision. Experiments on mathematical reasoning and general problem-solving benchmarks demonstrate that Maestro, coupled with CLPO, consistently outperforms existing state-of-the-art multi-agent approaches, delivering absolute accuracy gains of 6% on average and up to 10% at best. 

---
# ScRPO: From Errors to Insights 

**Authors**: Lianrui Li, Dakuan Lu, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06065)  

**Abstract**: We propose Self-correction Relative Policy Optimization (ScRPO), a novel reinforcement learning framework designed to enhance large language models on challenging mathemati- cal problems by leveraging self-reflection and error correction. Our approach consists of two stages: (1) Trial-and-error learning stage: training the model with GRPO and collect- ing incorrect answers along with their cor- responding questions in an error pool; (2) Self-correction learning stage: guiding the model to reflect on why its previous an- swers were wrong. Extensive experiments across multiple math reasoning benchmarks, including AIME, AMC, Olympiad, MATH- 500, GSM8k, using Deepseek-Distill-Qwen- 1.5B and Deepseek-Distill-Qwen-7B. The ex- perimental results demonstrate that ScRPO consistently outperforms several post-training methods. These findings highlight ScRPO as a promising paradigm for enabling language models to self-improve on difficult tasks with limited external feedback, paving the way to- ward more reliable and capable AI systems. 

---
# An Epistemic Perspective on Agent Awareness 

**Authors**: Pavel Naumov, Alexandra Pavlova  

**Link**: [PDF](https://arxiv.org/pdf/2511.05977)  

**Abstract**: The paper proposes to treat agent awareness as a form of knowledge, breaking the tradition in the existing literature on awareness. It distinguishes the de re and de dicto forms of such knowledge. The work introduces two modalities capturing these forms and formally specifies their meaning using a version of 2D-semantics. The main technical result is a sound and complete logical system describing the interplay between the two proposed modalities and the standard "knowledge of the fact" modality. 

---
# Klear-AgentForge: Forging Agentic Intelligence through Posttraining Scaling 

**Authors**: Qi Wang, Hongzhi Zhang, Jia Fu, Kai Fu, Yahui Liu, Tinghai Zhang, Chenxi Sun, Gangwei Jiang, Jingyi Tang, Xingguang Ji, Yang Yue, Jingyuan Zhang, Fuzheng Zhang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05951)  

**Abstract**: Despite the proliferation of powerful agentic models, the lack of critical post-training details hinders the development of strong counterparts in the open-source community. In this study, we present a comprehensive and fully open-source pipeline for training a high-performance agentic model for interacting with external tools and environments, named Klear-Qwen3-AgentForge, starting from the Qwen3-8B base model. We design effective supervised fine-tuning (SFT) with synthetic data followed by multi-turn reinforcement learning (RL) to unlock the potential for multiple diverse agentic tasks. We perform exclusive experiments on various agentic benchmarks in both tool use and coding domains. Klear-Qwen3-AgentForge-8B achieves state-of-the-art performance among LLMs of similar size and remains competitive with significantly larger models. 

---
# Self-Abstraction from Grounded Experience for Plan-Guided Policy Refinement 

**Authors**: Hiroaki Hayashi, Bo Pang, Wenting Zhao, Ye Liu, Akash Gokul, Srijan Bansal, Caiming Xiong, Semih Yavuz, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05931)  

**Abstract**: Large language model (LLM) based agents are increasingly used to tackle software engineering tasks that require multi-step reasoning and code modification, demonstrating promising yet limited performance. However, most existing LLM agents typically operate within static execution frameworks, lacking a principled mechanism to learn and self-improve from their own experience and past rollouts. As a result, their performance remains bounded by the initial framework design and the underlying LLM's capabilities. We propose Self-Abstraction from Grounded Experience (SAGE), a framework that enables agents to learn from their own task executions and refine their behavior through self-abstraction. After an initial rollout, the agent induces a concise plan abstraction from its grounded experience, distilling key steps, dependencies, and constraints. This learned abstraction is then fed back as contextual guidance, refining the agent's policy and supporting more structured, informed subsequent executions. Empirically, SAGE delivers consistent performance gains across diverse LLM backbones and agent architectures. Notably, it yields a 7.2% relative performance improvement over the strong Mini-SWE-Agent baseline when paired with the GPT-5 (high) backbone. SAGE further achieves strong overall performance on SWE-Bench Verified benchmark, reaching 73.2% and 74% Pass@1 resolve rates with the Mini-SWE-Agent and OpenHands CodeAct agent framework, respectively. 

---
# Unveiling Modality Bias: Automated Sample-Specific Analysis for Multimodal Misinformation Benchmarks 

**Authors**: Hehai Lin, Hui Liu, Shilei Cao, Jing Li, Haoliang Li, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05883)  

**Abstract**: Numerous multimodal misinformation benchmarks exhibit bias toward specific modalities, allowing detectors to make predictions based solely on one modality. While previous research has quantified bias at the dataset level or manually identified spurious correlations between modalities and labels, these approaches lack meaningful insights at the sample level and struggle to scale to the vast amount of online information. In this paper, we investigate the design for automated recognition of modality bias at the sample level. Specifically, we propose three bias quantification methods based on theories/views of different levels of granularity: 1) a coarse-grained evaluation of modality benefit; 2) a medium-grained quantification of information flow; and 3) a fine-grained causality analysis. To verify the effectiveness, we conduct a human evaluation on two popular benchmarks. Experimental results reveal three interesting findings that provide potential direction toward future research: 1)~Ensembling multiple views is crucial for reliable automated analysis; 2)~Automated analysis is prone to detector-induced fluctuations; and 3)~Different views produce a higher agreement on modality-balanced samples but diverge on biased ones. 

---
# An Empirical Study of Reasoning Steps in Thinking Code LLMs 

**Authors**: Haoran Xue, Gias Uddin, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05874)  

**Abstract**: Thinking Large Language Models (LLMs) generate explicit intermediate reasoning traces before final answers, potentially improving transparency, interpretability, and solution accuracy for code generation. However, the quality of these reasoning chains remains underexplored. We present a comprehensive empirical study examining the reasoning process and quality of thinking LLMs for code generation. We evaluate six state-of-the-art reasoning LLMs (DeepSeek-R1, OpenAI-o3-mini, Claude-3.7-Sonnet-Thinking, Gemini-2.0-Flash-Thinking, Gemini-2.5-Flash, and Qwen-QwQ) across 100 code generation tasks of varying difficulty from BigCodeBench. We quantify reasoning-chain structure through step counts and verbosity, conduct controlled step-budget adjustments, and perform a 21-participant human evaluation across three dimensions: efficiency, logical correctness, and completeness. Our step-count interventions reveal that targeted step increases can improve resolution rates for certain models/tasks, while modest reductions often preserve success on standard tasks, rarely on hard ones. Through systematic analysis, we develop a reasoning-problematic taxonomy, identifying completeness as the dominant failure mode. Task complexity significantly impacts reasoning quality; hard problems are substantially more prone to incompleteness than standard tasks. Our stability analysis demonstrates that thinking LLMs maintain consistent logical structures across computational effort levels and can self-correct previous errors. This study provides new insights into the strengths and limitations of current thinking LLMs in software engineering. 

---
# Can a Small Model Learn to Look Before It Leaps? Dynamic Learning and Proactive Correction for Hallucination Detection 

**Authors**: Zepeng Bao, Shen Zhou, Qiankun Pi, Jianhao Chen, Mayi Xu, Ming Zhong, Yuanyuan Zhu, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.05854)  

**Abstract**: Hallucination in large language models (LLMs) remains a critical barrier to their safe deployment. Existing tool-augmented hallucination detection methods require pre-defined fixed verification strategies, which are crucial to the quality and effectiveness of tool calls. Some methods directly employ powerful closed-source LLMs such as GPT-4 as detectors, which are effective but too costly. To mitigate the cost issue, some methods adopt the teacher-student architecture and finetune open-source small models as detectors via agent tuning. However, these methods are limited by fixed strategies. When faced with a dynamically changing execution environment, they may lack adaptability and inappropriately call tools, ultimately leading to detection failure. To address the problem of insufficient strategy adaptability, we propose the innovative ``Learning to Evaluate and Adaptively Plan''(LEAP) framework, which endows an efficient student model with the dynamic learning and proactive correction capabilities of the teacher model. Specifically, our method formulates the hallucination detection problem as a dynamic strategy learning problem. We first employ a teacher model to generate trajectories within the dynamic learning loop and dynamically adjust the strategy based on execution failures. We then distill this dynamic planning capability into an efficient student model via agent tuning. Finally, during strategy execution, the student model adopts a proactive correction mechanism, enabling it to propose, review, and optimize its own verification strategies before execution. We demonstrate through experiments on three challenging benchmarks that our LEAP-tuned model outperforms existing state-of-the-art methods. 

---
# DiagnoLLM: A Hybrid Bayesian Neural Language Framework for Interpretable Disease Diagnosis 

**Authors**: Bowen Xu, Xinyue Zeng, Jiazhen Hu, Tuo Wang, Adithya Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2511.05810)  

**Abstract**: Building trustworthy clinical AI systems requires not only accurate predictions but also transparent, biologically grounded explanations. We present \texttt{DiagnoLLM}, a hybrid framework that integrates Bayesian deconvolution, eQTL-guided deep learning, and LLM-based narrative generation for interpretable disease diagnosis. DiagnoLLM begins with GP-unmix, a Gaussian Process-based hierarchical model that infers cell-type-specific gene expression profiles from bulk and single-cell RNA-seq data while modeling biological uncertainty. These features, combined with regulatory priors from eQTL analysis, power a neural classifier that achieves high predictive performance in Alzheimer's Disease (AD) detection (88.0\% accuracy). To support human understanding and trust, we introduce an LLM-based reasoning module that translates model outputs into audience-specific diagnostic reports, grounded in clinical features, attribution signals, and domain knowledge. Human evaluations confirm that these reports are accurate, actionable, and appropriately tailored for both physicians and patients. Our findings show that LLMs, when deployed as post-hoc reasoners rather than end-to-end predictors, can serve as effective communicators within hybrid diagnostic pipelines. 

---
# Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs 

**Authors**: Felipe Valencia-Clavijo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05766)  

**Abstract**: Large language models (LLMs) are increasingly examined as both behavioral subjects and decision systems, yet it remains unclear whether observed cognitive biases reflect surface imitation or deeper probability shifts. Anchoring bias, a classic human judgment bias, offers a critical test case. While prior work shows LLMs exhibit anchoring, most evidence relies on surface-level outputs, leaving internal mechanisms and attributional contributions unexplored. This paper advances the study of anchoring in LLMs through three contributions: (1) a log-probability-based behavioral analysis showing that anchors shift entire output distributions, with controls for training-data contamination; (2) exact Shapley-value attribution over structured prompt fields to quantify anchor influence on model log-probabilities; and (3) a unified Anchoring Bias Sensitivity Score integrating behavioral and attributional evidence across six open-source models. Results reveal robust anchoring effects in Gemma-2B, Phi-2, and Llama-2-7B, with attribution signaling that the anchors influence reweighting. Smaller models such as GPT-2, Falcon-RW-1B, and GPT-Neo-125M show variability, suggesting scale may modulate sensitivity. Attributional effects, however, vary across prompt designs, underscoring fragility in treating LLMs as human substitutes. The findings demonstrate that anchoring bias in LLMs is robust, measurable, and interpretable, while highlighting risks in applied domains. More broadly, the framework bridges behavioral science, LLM safety, and interpretability, offering a reproducible path for evaluating other cognitive biases in LLMs. 

---
# CoT-X: An Adaptive Framework for Cross-Model Chain-of-Thought Transfer and Optimization 

**Authors**: Ziqian Bi, Kaijie Chen, Tianyang Wang, Junfeng Hao, Xinyuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.05747)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances the problem-solving ability of large language models (LLMs) but leads to substantial inference overhead, limiting deployment in resource-constrained settings. This paper investigates efficient CoT transfer across models of different scales and architectures through an adaptive reasoning summarization framework. The proposed method compresses reasoning traces via semantic segmentation with importance scoring, budget-aware dynamic compression, and coherence reconstruction, preserving critical reasoning steps while significantly reducing token usage. Experiments on 7{,}501 medical examination questions across 10 specialties show up to 40% higher accuracy than truncation under the same token budgets. Evaluations on 64 model pairs from eight LLMs (1.5B-32B parameters, including DeepSeek-R1 and Qwen3) confirm strong cross-model transferability. Furthermore, a Gaussian Process-based Bayesian optimization module reduces evaluation cost by 84% and reveals a power-law relationship between model size and cross-domain robustness. These results demonstrate that reasoning summarization provides a practical path toward efficient CoT transfer, enabling advanced reasoning under tight computational constraints. Code will be released upon publication. 

---
# From Prompts to Power: Measuring the Energy Footprint of LLM Inference 

**Authors**: Francisco Caravaca, Ángel Cuevas, Rubén Cuevas  

**Link**: [PDF](https://arxiv.org/pdf/2511.05597)  

**Abstract**: The rapid expansion of Large Language Models (LLMs) has introduced unprecedented energy demands, extending beyond training to large-scale inference workloads that often dominate total lifecycle consumption. Deploying these models requires energy-intensive GPU infrastructure, and in some cases has even prompted plans to power data centers with nuclear energy. Despite this growing relevance, systematic analyses of inference energy consumption remain limited. In this work, we present a large-scale measurement-based study comprising over 32,500 measurements across 21 GPU configurations and 155 model architectures, from small open-source models to frontier systems. Using the vLLM inference engine, we quantify energy usage at the prompt level and identify how architectural and operational factors shape energy demand. Building on these insights, we develop a predictive model that accurately estimates inference energy consumption across unseen architectures and hardware, and implement it as a browser extension to raise awareness of the environmental impact of generative AI. 

---
# SMAGDi: Socratic Multi Agent Interaction Graph Distillation for Efficient High Accuracy Reasoning 

**Authors**: Aayush Aluru, Myra Malik, Samarth Patankar, Spencer Kim, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2511.05528)  

**Abstract**: Multi-agent systems (MAS) often achieve higher reasoning accuracy than single models, but their reliance on repeated debates across agents makes them computationally expensive. We introduce SMAGDi, a distillation framework that transfers the debate dynamics of a five-agent Llama-based MAS into a compact Socratic decomposer-solver student. SMAGDi represents debate traces as directed interaction graphs, where nodes encode intermediate reasoning steps with correctness labels and edges capture continuity and cross-agent influence. The student is trained with a composite objective combining language modeling, graph-based supervision, contrastive reasoning, and embedding alignment to preserve both fluency and structured reasoning. On StrategyQA and MMLU, SMAGDi compresses a 40B multi-agent system into a 6B student while retaining 88% of its accuracy, substantially outperforming prior distillation methods such as MAGDi, standard KD, and fine-tuned baselines. These results highlight that explicitly modeling interaction graphs and Socratic decomposition enable small models to inherit the accuracy benefits of multi-agent debate while remaining efficient enough for real-world deployment. 

---
# Evidence-Bound Autonomous Research (EviBound): A Governance Framework for Eliminating False Claims 

**Authors**: Ruiying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05524)  

**Abstract**: LLM-based autonomous research agents report false claims: tasks marked "complete" despite missing artifacts, contradictory metrics, or failed executions. EviBound is an evidence-bound execution framework that eliminates false claims through dual governance gates requiring machine-checkable evidence.
Two complementary gates enforce evidence requirements. The pre-execution Approval Gate validates acceptance criteria schemas before code runs, catching structural violations proactively. The post-execution Verification Gate validates artifacts via MLflow API queries (with recursive path checking) and optionally validates metrics when specified by acceptance criteria. Claims propagate only when backed by a queryable run ID, required artifacts, and FINISHED status. Bounded, confidence-gated retries (typically 1-2 attempts) recover from transient failures without unbounded loops.
The framework was evaluated on 8 benchmark tasks spanning infrastructure validation, ML capabilities, and governance stress tests. Baseline A (Prompt-Level Only) yields 100% hallucination (8/8 claimed, 0/8 verified). Baseline B (Verification-Only) reduces hallucination to 25% (2/8 fail verification). EviBound (Dual Gates) achieves 0% hallucination: 7/8 tasks verified and 1 task correctly blocked at the approval gate, all with only approximately 8.3% execution overhead.
This package includes execution trajectories, MLflow run IDs for all verified tasks, and a 4-step verification protocol. Research integrity is an architectural property, achieved through governance gates rather than emergent from model scale. 

---
# Lightning Grasp: High Performance Procedural Grasp Synthesis with Contact Fields 

**Authors**: Zhao-Heng Yin, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2511.07418)  

**Abstract**: Despite years of research, real-time diverse grasp synthesis for dexterous hands remains an unsolved core challenge in robotics and computer graphics. We present Lightning Grasp, a novel high-performance procedural grasp synthesis algorithm that achieves orders-of-magnitude speedups over state-of-the-art approaches, while enabling unsupervised grasp generation for irregular, tool-like objects. The method avoids many limitations of prior approaches, such as the need for carefully tuned energy functions and sensitive initialization. This breakthrough is driven by a key insight: decoupling complex geometric computation from the search process via a simple, efficient data structure - the Contact Field. This abstraction collapses the problem complexity, enabling a procedural search at unprecedented speeds. We open-source our system to propel further innovation in robotic manipulation. 

---
# Language Generation with Infinite Contamination 

**Authors**: Anay Mehrotra, Grigoris Velegkas, Xifan Yu, Felix Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.07417)  

**Abstract**: We study language generation in the limit, where an algorithm observes an adversarial enumeration of strings from an unknown target language $K$ and must eventually generate new, unseen strings from $K$. Kleinberg and Mullainathan [KM24] proved that generation is achievable in surprisingly general settings. But their generator suffers from ``mode collapse,'' producing from an ever-smaller subset of the target. To address this, Kleinberg and Wei [KW25] require the generator's output to be ``dense'' in the target language. They showed that generation with density, surprisingly, remains achievable at the same generality.
Both results assume perfect data: no noisy insertions and no omissions. This raises a central question: how much contamination can generation tolerate? Recent works made partial progress on this question by studying (non-dense) generation with either finite amounts of noise (but no omissions) or omissions (but no noise).
We characterize robustness under contaminated enumerations: 1. Generation under Contamination: Language generation in the limit is achievable for all countable collections iff the fraction of contaminated examples converges to zero. When this fails, we characterize which collections are generable. 2. Dense Generation under Contamination: Dense generation is strictly less robust to contamination than generation. As a byproduct, we resolve an open question of Raman and Raman [ICML25] by showing that generation is possible with only membership oracle access under finitely many contaminated examples.
Finally, we introduce a beyond-worst-case model inspired by curriculum learning and prove that dense generation is achievable even with infinite contamination provided the fraction of contaminated examples converges to zero. This suggests curriculum learning may be crucial for learning from noisy web data. 

---
# Robot Learning from a Physical World Model 

**Authors**: Jiageng Mao, Sicheng He, Hao-Ning Wu, Yang You, Shuyang Sun, Zhicheng Wang, Yanan Bao, Huizhong Chen, Leonidas Guibas, Vitor Guizilini, Howard Zhou, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07416)  

**Abstract**: We introduce PhysWorld, a framework that enables robot learning from video generation through physical world modeling. Recent video generation models can synthesize photorealistic visual demonstrations from language commands and images, offering a powerful yet underexplored source of training signals for robotics. However, directly retargeting pixel motions from generated videos to robots neglects physics, often resulting in inaccurate manipulations. PhysWorld addresses this limitation by coupling video generation with physical world reconstruction. Given a single image and a task command, our method generates task-conditioned videos and reconstructs the underlying physical world from the videos, and the generated video motions are grounded into physically accurate actions through object-centric residual reinforcement learning with the physical world model. This synergy transforms implicit visual guidance into physically executable robotic trajectories, eliminating the need for real robot data collection and enabling zero-shot generalizable robotic manipulation. Experiments on diverse real-world tasks demonstrate that PhysWorld substantially improves manipulation accuracy compared to previous approaches. Visit \href{this https URL}{the project webpage} for details. 

---
# Using Vision Language Models as Closed-Loop Symbolic Planners for Robotic Applications: A Control-Theoretic Perspective 

**Authors**: Hao Wang, Sathwik Karnik, Bea Lim, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07410)  

**Abstract**: Large Language Models (LLMs) and Vision Language Models (VLMs) have been widely used for embodied symbolic planning. Yet, how to effectively use these models for closed-loop symbolic planning remains largely unexplored. Because they operate as black boxes, LLMs and VLMs can produce unpredictable or costly errors, making their use in high-level robotic planning especially challenging. In this work, we investigate how to use VLMs as closed-loop symbolic planners for robotic applications from a control-theoretic perspective. Concretely, we study how the control horizon and warm-starting impact the performance of VLM symbolic planners. We design and conduct controlled experiments to gain insights that are broadly applicable to utilizing VLMs as closed-loop symbolic planners, and we discuss recommendations that can help improve the performance of VLM symbolic planners. 

---
# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards 

**Authors**: Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2511.07403)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning. 

---
# Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction 

**Authors**: Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07392)  

**Abstract**: In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery. 

---
# Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence 

**Authors**: Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2511.07384)  

**Abstract**: Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute. In this work, we study how to convert existing pretrained non-recurrent language models into depth-recurrent models. We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost. In our experiments, on mathematics, we observe that converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model. 

---
# LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs 

**Authors**: Himanshu Pal, Venkata Sai Pranav Bachina, Ankit Gangwal, Charu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2511.07379)  

**Abstract**: Temporal Graph Neural Networks (TGNNs) are increasingly used in high-stakes domains, such as financial forecasting, recommendation systems, and fraud detection. However, their susceptibility to poisoning attacks poses a critical security risk. We introduce LoReTTA (Low Resource Two-phase Temporal Attack), a novel adversarial framework on Continuous-Time Dynamic Graphs, which degrades TGNN performance by an average of 29.47% across 4 widely benchmark datasets and 4 State-of-the-Art (SotA) models. LoReTTA operates through a two-stage approach: (1) sparsify the graph by removing high-impact edges using any of the 16 tested temporal importance metrics, (2) strategically replace removed edges with adversarial negatives via LoReTTA's novel degree-preserving negative sampling algorithm. Our plug-and-play design eliminates the need for expensive surrogate models while adhering to realistic unnoticeability constraints. LoReTTA degrades performance by upto 42.0% on MOOC, 31.5% on Wikipedia, 28.8% on UCI, and 15.6% on Enron. LoReTTA outperforms 11 attack baselines, remains undetectable to 4 leading anomaly detection systems, and is robust to 4 SotA adversarial defense training methods, establishing its effectiveness, unnoticeability, and robustness. 

---
# Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization 

**Authors**: Yu Huang, Zixin Wen, Aarti Singh, Yuejie Chi, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07378)  

**Abstract**: The ability to reason lies at the core of artificial intelligence (AI), and challenging problems usually call for deeper and longer reasoning to tackle. A crucial question about AI reasoning is whether models can extrapolate learned reasoning patterns to solve harder tasks with longer chain-of-thought (CoT). In this work, we present a theoretical analysis of transformers learning on synthetic state-tracking tasks with gradient descent. We mathematically prove how the algebraic structure of state-tracking problems governs the degree of extrapolation of the learned CoT. Specifically, our theory characterizes the length generalization of transformers through the mechanism of attention concentration, linking the retrieval robustness of the attention layer to the state-tracking task structure of long-context reasoning. Moreover, for transformers with limited reasoning length, we prove that a recursive self-training scheme can progressively extend the range of solvable problem lengths. To our knowledge, we provide the first optimization guarantee that constant-depth transformers provably learn $\mathsf{NC}^1$-complete problems with CoT, significantly going beyond prior art confined in $\mathsf{TC}^0$, unless the widely held conjecture $\mathsf{TC}^0 \neq \mathsf{NC}^1$ fails. Finally, we present a broad set of experiments supporting our theoretical results, confirming the length generalization behaviors and the mechanism of attention concentration. 

---
# Real-Time LiDAR Super-Resolution via Frequency-Aware Multi-Scale Fusion 

**Authors**: June Moh Goo, Zichao Zeng, Jan Boehm  

**Link**: [PDF](https://arxiv.org/pdf/2511.07377)  

**Abstract**: LiDAR super-resolution addresses the challenge of achieving high-quality 3D perception from cost-effective, low-resolution sensors. While recent transformer-based approaches like TULIP show promise, they remain limited to spatial-domain processing with restricted receptive fields. We introduce FLASH (Frequency-aware LiDAR Adaptive Super-resolution with Hierarchical fusion), a novel framework that overcomes these limitations through dual-domain processing. FLASH integrates two key innovations: (i) Frequency-Aware Window Attention that combines local spatial attention with global frequency-domain analysis via FFT, capturing both fine-grained geometry and periodic scanning patterns at log-linear complexity. (ii) Adaptive Multi-Scale Fusion that replaces conventional skip connections with learned position-specific feature aggregation, enhanced by CBAM attention for dynamic feature selection. Extensive experiments on KITTI demonstrate that FLASH achieves state-of-the-art performance across all evaluation metrics, surpassing even uncertainty-enhanced baselines that require multiple forward passes. Notably, FLASH outperforms TULIP with Monte Carlo Dropout while maintaining single-pass efficiency, which enables real-time deployment. The consistent superiority across all distance ranges validates that our dual-domain approach effectively handles uncertainty through architectural design rather than computationally expensive stochastic inference, making it practical for autonomous systems. 

---
# Consistency Is Not Always Correct: Towards Understanding the Role of Exploration in Post-Training Reasoning 

**Authors**: Dake Bu, Wei Huang, Andi Han, Atsushi Nitanda, Bo Xue, Qingfu Zhang, Hau-San Wong, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2511.07368)  

**Abstract**: Foundation models exhibit broad knowledge but limited task-specific reasoning, motivating post-training strategies such as RLVR and inference scaling with outcome or process reward models (ORM/PRM). While recent work highlights the role of exploration and entropy stability in improving pass@K, empirical evidence points to a paradox: RLVR and ORM/PRM typically reinforce existing tree-like reasoning paths rather than expanding the reasoning scope, raising the question of why exploration helps at all if no new patterns emerge.
To reconcile this paradox, we adopt the perspective of Kim et al. (2025), viewing easy (e.g., simplifying a fraction) versus hard (e.g., discovering a symmetry) reasoning steps as low- versus high-probability Markov transitions, and formalize post-training dynamics through Multi-task Tree-structured Markov Chains (TMC). In this tractable model, pretraining corresponds to tree expansion, while post-training corresponds to chain-of-thought reweighting. We show that several phenomena recently observed in empirical studies arise naturally in this setting: (1) RLVR induces a squeezing effect, reducing reasoning entropy and forgetting some correct paths; (2) population rewards of ORM/PRM encourage consistency rather than accuracy, thereby favoring common patterns; and (3) certain rare, high-uncertainty reasoning paths by the base model are responsible for solving hard problem instances.
Together, these explain why exploration -- even when confined to the base model's reasoning scope -- remains essential: it preserves access to rare but crucial reasoning traces needed for difficult cases, which are squeezed out by RLVR or unfavored by inference scaling. Building on this, we further show that exploration strategies such as rejecting easy instances and KL regularization help preserve rare reasoning traces. Empirical simulations corroborate our theoretical results. 

---
# Machine-Learning Accelerated Calculations of Reduced Density Matrices 

**Authors**: Awwab A. Azam, Lexu Zhao, Jiabin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07367)  

**Abstract**: $n$-particle reduced density matrices ($n$-RDMs) play a central role in understanding correlated phases of matter. Yet the calculation of $n$-RDMs is often computationally inefficient for strongly-correlated states, particularly when the system sizes are large. In this work, we propose to use neural network (NN) architectures to accelerate the calculation of, and even predict, the $n$-RDMs for large-size systems. The underlying intuition is that $n$-RDMs are often smooth functions over the Brillouin zone (BZ) (certainly true for gapped states) and are thus interpolable, allowing NNs trained on small-size $n$-RDMs to predict large-size ones. Building on this intuition, we devise two NNs: (i) a self-attention NN that maps random RDMs to physical ones, and (ii) a Sinusoidal Representation Network (SIREN) that directly maps momentum-space coordinates to RDM values. We test the NNs in three 2D models: the pair-pair correlation functions of the Richardson model of superconductivity, the translationally-invariant 1-RDM in a four-band model with short-range repulsion, and the translation-breaking 1-RDM in the half-filled Hubbard model. We find that a SIREN trained on a $6\times 6$ momentum mesh can predict the $18\times 18$ pair-pair correlation function with a relative accuracy of $0.839$. The NNs trained on $6\times 6 \sim 8\times 8$ meshes can provide high-quality initial guesses for $50\times 50$ translation-invariant Hartree-Fock (HF) and $30\times 30$ fully translation-breaking-allowed HF, reducing the number of iterations required for convergence by up to $91.63\%$ and $92.78\%$, respectively, compared to random initializations. Our results illustrate the potential of using NN-based methods for interpolable $n$-RDMs, which might open a new avenue for future research on strongly correlated phases. 

---
# Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection 

**Authors**: Vaibhav Mavi, Shubh Jaroria, Weiqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.07364)  

**Abstract**: Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection. 

---
# Inference-Time Scaling of Diffusion Models for Infrared Data Generation 

**Authors**: Kai A. Horstmann, Maxim Clouser, Kia Khezeli  

**Link**: [PDF](https://arxiv.org/pdf/2511.07362)  

**Abstract**: Infrared imagery enables temperature-based scene understanding using passive sensors, particularly under conditions of low visibility where traditional RGB imaging fails. Yet, developing downstream vision models for infrared applications is hindered by the scarcity of high-quality annotated data, due to the specialized expertise required for infrared annotation. While synthetic infrared image generation has the potential to accelerate model development by providing large-scale, diverse training data, training foundation-level generative diffusion models in the infrared domain has remained elusive due to limited datasets. In light of such data constraints, we explore an inference-time scaling approach using a domain-adapted CLIP-based verifier for enhanced infrared image generation quality. We adapt FLUX.1-dev, a state-of-the-art text-to-image diffusion model, to the infrared domain by finetuning it on a small sample of infrared images using parameter-efficient techniques. The trained verifier is then employed during inference to guide the diffusion sampling process toward higher quality infrared generations that better align with input text prompts. Empirically, we find that our approach leads to consistent improvements in generation quality, reducing FID scores on the KAIST Multispectral Pedestrian Detection Benchmark dataset by 10% compared to unguided baseline samples. Our results suggest that inference-time guidance offers a promising direction for bridging the domain gap in low-data infrared settings. 

---
# TNT: Improving Chunkwise Training for Test-Time Memorization 

**Authors**: Zeman Li, Ali Behrouz, Yuan Deng, Peilin Zhong, Praneeth Kacham, Mahdi Karami, Meisam Razaviyayn, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2511.07343)  

**Abstract**: Recurrent neural networks (RNNs) with deep test-time memorization modules, such as Titans and TTT, represent a promising, linearly-scaling paradigm distinct from Transformers. While these expressive models do not yet match the peak performance of state-of-the-art Transformers, their potential has been largely untapped due to prohibitively slow training and low hardware utilization. Existing parallelization methods force a fundamental conflict governed by the chunksize hyperparameter: large chunks boost speed but degrade performance, necessitating a fixed, suboptimal compromise. To solve this challenge, we introduce TNT, a novel training paradigm that decouples training efficiency from inference performance through a two-stage process. Stage one is an efficiency-focused pre-training phase utilizing a hierarchical memory. A global module processes large, hardware-friendly chunks for long-range context, while multiple parallel local modules handle fine-grained details. Crucially, by periodically resetting local memory states, we break sequential dependencies to enable massive context parallelization. Stage two is a brief fine-tuning phase where only the local memory modules are adapted to a smaller, high-resolution chunksize, maximizing accuracy with minimal overhead. Evaluated on Titans and TTT models, TNT achieves a substantial acceleration in training speed-up to 17 times faster than the most accurate baseline configuration - while simultaneously improving model accuracy. This improvement removes a critical scalability barrier, establishing a practical foundation for developing expressive RNNs and facilitating future work to close the performance gap with Transformers. 

---
# Grounding Computer Use Agents on Human Demonstrations 

**Authors**: Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07332)  

**Abstract**: Building reliable computer-use agents requires grounding: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. To address this gap, we introduce GroundCUA, a large-scale desktop grounding dataset built from expert human demonstrations. It covers 87 applications across 12 categories and includes 56K screenshots, with every on-screen element carefully annotated for a total of over 3.56M human-verified annotations. From these demonstrations, we generate diverse instructions that capture a wide range of real-world tasks, providing high-quality data for model training. Using GroundCUA, we develop the GroundNext family of models that map instructions to their target UI elements. At both 3B and 7B scales, GroundNext achieves state-of-the-art results across five benchmarks using supervised fine-tuning, while requiring less than one-tenth the training data of prior work. Reinforcement learning post-training further improves performance, and when evaluated in an agentic setting on the OSWorld benchmark using o3 as planner, GroundNext attains comparable or superior results to models trained with substantially more data,. These results demonstrate the critical role of high-quality, expert-driven datasets in advancing general-purpose computer-use agents. 

---
# FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation 

**Authors**: Song Jin, Shuqi Li, Shukun Zhang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07322)  

**Abstract**: While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available. 

---
# When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs 

**Authors**: Shaowen Wang, Yiqi Dong, Ruinian Chang, Tansheng Zhu, Yuebo Sun, Kaifeng Lyu, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.07318)  

**Abstract**: Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations. 

---
# Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search 

**Authors**: Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, Gabriele Farina  

**Link**: [PDF](https://arxiv.org/pdf/2511.07312)  

**Abstract**: Few classical games have been regarded as such significant benchmarks of artificial intelligence as to have justified training costs in the millions of dollars. Among these, Stratego -- a board wargame exemplifying the challenge of strategic decision making under massive amounts of hidden information -- stands apart as a case where such efforts failed to produce performance at the level of top humans. This work establishes a step change in both performance and cost for Stratego, showing that it is now possible not only to reach the level of top humans, but to achieve vastly superhuman level -- and that doing so requires not an industrial budget, but merely a few thousand dollars. We achieved this result by developing general approaches for self-play reinforcement learning and test-time search under imperfect information. 

---
# Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection 

**Authors**: Huizai Yao, Sicheng Zhao, Pengteng Li, Yi Cui, Shuo Lu, Weiyu Guo, Yunfan Lu, Yijie Xu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.07301)  

**Abstract**: Source-Free Object Detection (SFOD) aims to adapt a source-pretrained object detector to a target domain without access to source data. However, existing SFOD methods predominantly rely on internal knowledge from the source model, which limits their capacity to generalize across domains and often results in biased pseudo-labels, thereby hindering both transferability and discriminability. In contrast, Vision Foundation Models (VFMs), pretrained on massive and diverse data, exhibit strong perception capabilities and broad generalization, yet their potential remains largely untapped in the SFOD setting. In this paper, we propose a novel SFOD framework that leverages VFMs as external knowledge sources to jointly enhance feature alignment and label quality. Specifically, we design three VFM-based modules: (1) Patch-weighted Global Feature Alignment (PGFA) distills global features from VFMs using patch-similarity-based weighting to enhance global feature transferability; (2) Prototype-based Instance Feature Alignment (PIFA) performs instance-level contrastive learning guided by momentum-updated VFM prototypes; and (3) Dual-source Enhanced Pseudo-label Fusion (DEPF) fuses predictions from detection VFMs and teacher models via an entropy-aware strategy to yield more reliable supervision. Extensive experiments on six benchmarks demonstrate that our method achieves state-of-the-art SFOD performance, validating the effectiveness of integrating VFMs to simultaneously improve transferability and discriminability. 

---
# LMM-IQA: Image Quality Assessment for Low-Dose CT Imaging 

**Authors**: Kagan Celik, Mehmet Ozan Unal, Metin Ertas, Isa Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07298)  

**Abstract**: Low-dose computed tomography (CT) represents a significant improvement in patient safety through lower radiation doses, but increased noise, blur, and contrast loss can diminish diagnostic quality. Therefore, consistency and robustness in image quality assessment become essential for clinical applications. In this study, we propose an LLM-based quality assessment system that generates both numerical scores and textual descriptions of degradations such as noise, blur, and contrast loss. Furthermore, various inference strategies - from the zero-shot approach to metadata integration and error feedback - are systematically examined, demonstrating the progressive contribution of each method to overall performance. The resultant assessments yield not only highly correlated scores but also interpretable output, thereby adding value to clinical workflows. The source codes of our study are available at this https URL. 

---
# Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models 

**Authors**: Tianrui Song, Wen-Shuo Chao, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07295)  

**Abstract**: Implicit feedback, employed in training recommender systems, unavoidably confronts noise due to factors such as misclicks and position bias. Previous studies have attempted to identify noisy samples through their diverged data patterns, such as higher loss values, and mitigate their influence through sample dropping or reweighting. However, we observed that noisy samples and hard samples display similar patterns, leading to hard-noisy confusion issue. Such confusion is problematic as hard samples are vital for modeling user preferences. To solve this problem, we propose LLMHNI framework, leveraging two auxiliary user-item relevance signals generated by Large Language Models (LLMs) to differentiate hard and noisy samples. LLMHNI obtains user-item semantic relevance from LLM-encoded embeddings, which is used in negative sampling to select hard negatives while filtering out noisy false negatives. An objective alignment strategy is proposed to project LLM-encoded embeddings, originally for general language tasks, into a representation space optimized for user-item relevance modeling. LLMHNI also exploits LLM-inferred logical relevance within user-item interactions to identify hard and noisy samples. These LLM-inferred interactions are integrated into the interaction graph and guide denoising with cross-graph contrastive alignment. To eliminate the impact of unreliable interactions induced by LLM hallucination, we propose a graph contrastive learning strategy that aligns representations from randomly edge-dropped views to suppress unreliable edges. Empirical results demonstrate that LLMHNI significantly improves denoising and recommendation performance. 

---
# Verifying rich robustness properties for neural networks 

**Authors**: Mohammad Afzal, S. Akshay, Ashutosh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.07293)  

**Abstract**: Robustness is a important problem in AI alignment and safety, with models such as neural networks being increasingly used in safety-critical systems. In the last decade, a large body of work has emerged on local robustness, i.e., checking if the decision of a neural network remains unchanged when the input is slightly perturbed. However, many of these approaches require specialized encoding and often ignore the confidence of a neural network on its output. In this paper, our goal is to build a generalized framework to specify and verify variants of robustness in neural network verification. We propose a specification framework using a simple grammar, which is flexible enough to capture most existing variants. This allows us to introduce new variants of robustness that take into account the confidence of the neural network in its outputs. Next, we develop a novel and powerful unified technique to verify all such variants in a homogeneous way, viz., by adding a few additional layers to the neural network. This enables us to use any state-of-the-art neural network verification tool, without having to tinker with the encoding within, while incurring an approximation error that we show is bounded. We perform an extensive experimental evaluation over a large suite of 8870 benchmarks having 138M parameters in a largest network, and show that we are able to capture a wide set of robustness variants and outperform direct encoding approaches by a significant margin. 

---
# Enabling Off-Policy Imitation Learning with Deep Actor Critic Stabilization 

**Authors**: Sayambhu Sen, Shalabh Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07288)  

**Abstract**: Learning complex policies with Reinforcement Learning (RL) is often hindered by instability and slow convergence, a problem exacerbated by the difficulty of reward engineering. Imitation Learning (IL) from expert demonstrations bypasses this reliance on rewards. However, state-of-the-art IL methods, exemplified by Generative Adversarial Imitation Learning (GAIL)Ho et. al, suffer from severe sample inefficiency. This is a direct consequence of their foundational on-policy algorithms, such as TRPO Schulman this http URL. In this work, we introduce an adversarial imitation learning algorithm that incorporates off-policy learning to improve sample efficiency. By combining an off-policy framework with auxiliary techniques specifically, double Q network based stabilization and value learning without reward function inference we demonstrate a reduction in the samples required to robustly match expert behavior. 

---
# Glioma C6: A Novel Dataset for Training and Benchmarking Cell Segmentation 

**Authors**: Roman Malashin, Svetlana Pashkevich, Daniil Ilyukhin, Arseniy Volkov, Valeria Yachnaya, Andrey Denisov, Maria Mikhalkova  

**Link**: [PDF](https://arxiv.org/pdf/2511.07286)  

**Abstract**: We present Glioma C6, a new open dataset for instance segmentation of glioma C6 cells, designed as both a benchmark and a training resource for deep learning models. The dataset comprises 75 high-resolution phase-contrast microscopy images with over 12,000 annotated cells, providing a realistic testbed for biomedical image analysis. It includes soma annotations and morphological cell categorization provided by biologists. Additional categorization of cells, based on morphology, aims to enhance the utilization of image data for cancer cell research. Glioma C6 consists of two parts: the first is curated with controlled parameters for benchmarking, while the second supports generalization testing under varying conditions. We evaluate the performance of several generalist segmentation models, highlighting their limitations on our dataset. Our experiments demonstrate that training on Glioma C6 significantly enhances segmentation performance, reinforcing its value for developing robust and generalizable models. The dataset is publicly available for researchers. 

---
# Designing Beyond Language: Sociotechnical Barriers in AI Health Technologies for Limited English Proficiency 

**Authors**: Michelle Huang, Violeta J. Rodriguez, Koustuv Saha, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2511.07277)  

**Abstract**: Limited English proficiency (LEP) patients in the U.S. face systemic barriers to healthcare beyond language and interpreter access, encompassing procedural and institutional constraints. AI advances may support communication and care through on-demand translation and visit preparation, but also risk exacerbating existing inequalities. We conducted storyboard-driven interviews with 14 patient navigators to explore how AI could shape care experiences for Spanish-speaking LEP individuals. We identified tensions around linguistic and cultural misunderstandings, privacy concerns, and opportunities and risks for AI to augment care workflows. Participants highlighted structural factors that can undermine trust in AI systems, including sensitive information disclosure, unstable technology access, and low digital literacy. While AI tools can potentially alleviate social barriers and institutional constraints, there are risks of misinformation and uprooting human camaraderie. Our findings contribute design considerations for AI that support LEP patients and care teams via rapport-building, education, and language support, and minimizing disruptions to existing practices. 

---
# MVU-Eval: Towards Multi-Video Understanding Evaluation for Multimodal LLMs 

**Authors**: Tianhao Peng, Haochen Wang, Yuanxing Zhang, Zekun Wang, Zili Wang, Ge Zhang, Jian Yang, Shihao Li, Yanghai Wang, Xintao Wang, Houyi Li, Wei Ji, Pengfei Wan, Wenhao Huang, Zhaoxiang Zhang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07250)  

**Abstract**: The advent of Multimodal Large Language Models (MLLMs) has expanded AI capabilities to visual modalities, yet existing evaluation benchmarks remain limited to single-video understanding, overlooking the critical need for multi-video understanding in real-world scenarios (e.g., sports analytics and autonomous driving). To address this significant gap, we introduce MVU-Eval, the first comprehensive benchmark for evaluating Multi-Video Understanding for MLLMs. Specifically, our MVU-Eval mainly assesses eight core competencies through 1,824 meticulously curated question-answer pairs spanning 4,959 videos from diverse domains, addressing both fundamental perception tasks and high-order reasoning tasks. These capabilities are rigorously aligned with real-world applications such as multi-sensor synthesis in autonomous systems and cross-angle sports analytics. Through extensive evaluation of state-of-the-art open-source and closed-source models, we reveal significant performance discrepancies and limitations in current MLLMs' ability to perform understanding across multiple videos. The benchmark will be made publicly available to foster future research. 

---
# Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation 

**Authors**: Seungheon Song, Jaekoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07238)  

**Abstract**: In autonomous driving and robotics, ensuring road safety and reliable decision-making critically depends on out-of-distribution (OOD) segmentation. While numerous methods have been proposed to detect anomalous objects on the road, leveraging the vision-language space-which provides rich linguistic knowledge-remains an underexplored field. We hypothesize that incorporating these linguistic cues can be especially beneficial in the complex contexts found in real-world autonomous driving scenarios.
To this end, we present a novel approach that trains a Text-Driven OOD Segmentation model to learn a semantically diverse set of objects in the vision-language space. Concretely, our approach combines a vision-language model's encoder with a transformer decoder, employs Distance-Based OOD prompts located at varying semantic distances from in-distribution (ID) classes, and utilizes OOD Semantic Augmentation for OOD representations. By aligning visual and textual information, our approach effectively generalizes to unseen objects and provides robust OOD segmentation in diverse driving environments.
We conduct extensive experiments on publicly available OOD segmentation datasets such as Fishyscapes, Segment-Me-If-You-Can, and Road Anomaly datasets, demonstrating that our approach achieves state-of-the-art performance across both pixel-level and object-level evaluations. This result underscores the potential of vision-language-based OOD segmentation to bolster the safety and reliability of future autonomous driving systems. 

---
# Discourse Graph Guided Document Translation with Large Language Models 

**Authors**: Viet-Thanh Pham, Minghan Wang, Hao-Han Liao, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07230)  

**Abstract**: Adapting large language models to full document translation remains challenging due to the difficulty of capturing long-range dependencies and preserving discourse coherence throughout extended texts. While recent agentic machine translation systems mitigate context window constraints through multi-agent orchestration and persistent memory, they require substantial computational resources and are sensitive to memory retrieval strategies. We introduce TransGraph, a discourse-guided framework that explicitly models inter-chunk relationships through structured discourse graphs and selectively conditions each translation segment on relevant graph neighbourhoods rather than relying on sequential or exhaustive context. Across three document-level MT benchmarks spanning six languages and diverse domains, TransGraph consistently surpasses strong baselines in translation quality and terminology consistency while incurring significantly lower token overhead. 

---
# LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure 

**Authors**: Jaehong Cho, Hyunmin Choi, Jongse Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.07229)  

**Abstract**: This paper introduces LLMServingSim2.0, a system simulator designed for exploring heterogeneous hardware in large-scale LLM serving systems. LLMServingSim2.0 addresses two key limitations of its predecessor: (1) integrating hardware models into system-level simulators is non-trivial due to the lack of a clear abstraction, and (2) existing simulators support only a narrow subset of serving techniques, leaving no infrastructure that captures the breadth of approaches in modern LLM serving. To overcome these issues, LLMServingSim2.0 adopts trace-driven performance modeling, accompanied by an operator-level latency profiler, enabling the integration of new accelerators with a single command. It further embeds up-to-date serving techniques while exposing flexible interfaces for request routing, cache management, and scheduling policies. In a TPU case study, our profiler requires 18.5x fewer LoC and outperforms the predecessor's hardware-simulator integration, demonstrating LLMServingSim2.0's low-effort hardware extensibility. Our experiments further show that LLMServingSim2.0 reproduces GPU-based LLM serving with 1.9% error, while maintaining practical simulation time, making it a comprehensive platform for both hardware developers and LLM service providers. 

---
# NoteEx: Interactive Visual Context Manipulation for LLM-Assisted Exploratory Data Analysis in Computational Notebooks 

**Authors**: Mohammad Hasan Payandeh, Lin-Ping Yuan, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07223)  

**Abstract**: Computational notebooks have become popular for Exploratory Data Analysis (EDA), augmented by LLM-based code generation and result interpretation. Effective LLM assistance hinges on selecting informative context -- the minimal set of cells whose code, data, or outputs suffice to answer a prompt. As notebooks grow long and messy, users can lose track of the mental model of their analysis. They thus fail to curate appropriate contexts for LLM tasks, causing frustration and tedious prompt engineering. We conducted a formative study (n=6) that surfaced challenges in LLM context selection and mental model maintenance. Therefore, we introduce NoteEx, a JupyterLab extension that provides a semantic visualization of the EDA workflow, allowing analysts to externalize their mental model, specify analysis dependencies, and enable interactive selection of task-relevant contexts for LLMs. A user study (n=12) against a baseline shows that NoteEx improved mental model retention and context selection, leading to more accurate and relevant LLM responses. 

---
# SMiLE: Provably Enforcing Global Relational Properties in Neural Networks 

**Authors**: Matteo Francobaldi, Michele Lombardi, Andrea Lodi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07208)  

**Abstract**: Artificial Intelligence systems are increasingly deployed in settings where ensuring robustness, fairness, or domain-specific properties is essential for regulation compliance and alignment with human values. However, especially on Neural Networks, property enforcement is very challenging, and existing methods are limited to specific constraints or local properties (defined around datapoints), or fail to provide full guarantees. We tackle these limitations by extending SMiLE, a recently proposed enforcement framework for NNs, to support global relational properties (defined over the entire input space). The proposed approach scales well with model complexity, accommodates general properties and backbones, and provides full satisfaction guarantees. We evaluate SMiLE on monotonicity, global robustness, and individual fairness, on synthetic and real data, for regression and classification tasks. Our approach is competitive with property-specific baselines in terms of accuracy and runtime, and strictly superior in terms of generality and level of guarantees. Overall, our results emphasize the potential of the SMiLE framework as a platform for future research and applications. 

---
# Twenty-Five Years of MIR Research: Achievements, Practices, Evaluations, and Future Challenges 

**Authors**: Geoffroy Peeters, Zafar Rafii, Magdalena Fuentes, Zhiyao Duan, Emmanouil Benetos, Juhan Nam, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2511.07205)  

**Abstract**: In this paper, we trace the evolution of Music Information Retrieval (MIR) over the past 25 years. While MIR gathers all kinds of research related to music informatics, a large part of it focuses on signal processing techniques for music data, fostering a close relationship with the IEEE Audio and Acoustic Signal Processing Technical Commitee. In this paper, we reflect the main research achievements of MIR along the three EDICS related to music analysis, processing and generation. We then review a set of successful practices that fuel the rapid development of MIR research. One practice is the annual research benchmark, the Music Information Retrieval Evaluation eXchange, where participants compete on a set of research tasks. Another practice is the pursuit of reproducible and open research. The active engagement with industry research and products is another key factor for achieving large societal impacts and motivating younger generations of students to join the field. Last but not the least, the commitment to diversity, equity and inclusion ensures MIR to be a vibrant and open community where various ideas, methodologies, and career pathways collide. We finish by providing future challenges MIR will have to face. 

---
# Resilient by Design - Active Inference for Distributed Continuum Intelligence 

**Authors**: Praveen Kumar Donta, Alfreds Lapkovskis, Enzo Mingozzi, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07202)  

**Abstract**: Failures are the norm in highly complex and heterogeneous devices spanning the distributed computing continuum (DCC), from resource-constrained IoT and edge nodes to high-performance computing systems. Ensuring reliability and global consistency across these layers remains a major challenge, especially for AI-driven workloads requiring real-time, adaptive coordination. This paper introduces a Probabilistic Active Inference Resilience Agent (PAIR-Agent) to achieve resilience in DCC systems. PAIR-Agent performs three core operations: (i) constructing a causal fault graph from device logs, (ii) identifying faults while managing certainties and uncertainties using Markov blankets and the free-energy principle, and (iii) autonomously healing issues through active inference. Through continuous monitoring and adaptive reconfiguration, the agent maintains service continuity and stability under diverse failure conditions. Theoretical validations confirm the reliability and effectiveness of the proposed framework. 

---
# Federated Learning for Video Violence Detection: Complementary Roles of Lightweight CNNs and Vision-Language Models for Energy-Efficient Use 

**Authors**: Sébastien Thuau, Siba Haidar, Rachid Chelouah  

**Link**: [PDF](https://arxiv.org/pdf/2511.07171)  

**Abstract**: Deep learning-based video surveillance increasingly demands privacy-preserving architectures with low computational and environmental overhead. Federated learning preserves privacy but deploying large vision-language models (VLMs) introduces major energy and sustainability challenges. We compare three strategies for federated violence detection under realistic non-IID splits on the RWF-2000 and RLVS datasets: zero-shot inference with pretrained VLMs, LoRA-based fine-tuning of LLaVA-NeXT-Video-7B, and personalized federated learning of a 65.8M-parameter 3D CNN. All methods exceed 90% accuracy in binary violence detection. The 3D CNN achieves superior calibration (ROC AUC 92.59%) at roughly half the energy cost (240 Wh vs. 570 Wh) of federated LoRA, while VLMs provide richer multimodal reasoning. Hierarchical category grouping (based on semantic similarity and class exclusion) boosts VLM multiclass accuracy from 65.31% to 81% on the UCF-Crime dataset. To our knowledge, this is the first comparative simulation study of LoRA-tuned VLMs and personalized CNNs for federated violence detection, with explicit energy and CO2e quantification. Our results inform hybrid deployment strategies that default to efficient CNNs for routine inference and selectively engage VLMs for complex contextual reasoning. 

---
# AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning 

**Authors**: Meiyun Wang, Charin Polpanumas  

**Link**: [PDF](https://arxiv.org/pdf/2511.07166)  

**Abstract**: We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks. 

---
# Fuzzy Label: From Concept to Its Application in Label Learning 

**Authors**: Chenxi Luoa, Zhuangzhuang Zhaoa, Zhaohong Denga, Te Zhangb  

**Link**: [PDF](https://arxiv.org/pdf/2511.07165)  

**Abstract**: Label learning is a fundamental task in machine learning that aims to construct intelligent models using labeled data, encompassing traditional single-label and multi-label classification models. Traditional methods typically rely on logical labels, such as binary indicators (e.g., "yes/no") that specify whether an instance belongs to a given category. However, in practical applications, label annotations often involve significant uncertainty due to factors such as data noise, inherent ambiguity in the observed entities, and the subjectivity of human annotators. Therefore, representing labels using simplistic binary logic can obscure valuable information and limit the expressiveness of label learning models. To overcome this limitation, this paper introduces the concept of fuzzy labels, grounded in fuzzy set theory, to better capture and represent label uncertainty. We further propose an efficient fuzzy labeling method that mines and generates fuzzy labels from the original data, thereby enriching the label space with more informative and nuanced representations. Based on this foundation, we present fuzzy-label-enhanced algorithms for both single-label and multi-label learning, using the classical K-Nearest Neighbors (KNN) and multi-label KNN algorithms as illustrative examples. Experimental results indicate that fuzzy labels can more effectively characterize the real-world labeling information and significantly enhance the performance of label learning models. 

---
# Conditional Diffusion as Latent Constraints for Controllable Symbolic Music Generation 

**Authors**: Matteo Pettenó, Alessandro Ilic Mezza, Alberto Bernardini  

**Link**: [PDF](https://arxiv.org/pdf/2511.07156)  

**Abstract**: Recent advances in latent diffusion models have demonstrated state-of-the-art performance in high-dimensional time-series data synthesis while providing flexible control through conditioning and guidance. However, existing methodologies primarily rely on musical context or natural language as the main modality of interacting with the generative process, which may not be ideal for expert users who seek precise fader-like control over specific musical attributes. In this work, we explore the application of denoising diffusion processes as plug-and-play latent constraints for unconditional symbolic music generation models. We focus on a framework that leverages a library of small conditional diffusion models operating as implicit probabilistic priors on the latents of a frozen unconditional backbone. While previous studies have explored domain-specific use cases, this work, to the best of our knowledge, is the first to demonstrate the versatility of such an approach across a diverse array of musical attributes, such as note density, pitch range, contour, and rhythm complexity. Our experiments show that diffusion-driven constraints outperform traditional attribute regularization and other latent constraints architectures, achieving significantly stronger correlations between target and generated attributes while maintaining high perceptual quality and diversity. 

---
# LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging 

**Authors**: Seungeon Lee, Soumi Das, Manish Gupta, Krishna P. Gummadi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07129)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as a parameter-efficient approach for fine-tuning large language this http URL, conventional LoRA adapters are typically trained for a single task, limiting their applicability in real-world settings where inputs may span diverse and unpredictable domains. At inference time, existing approaches combine multiple LoRAs for improving performance on diverse tasks, while usually requiring labeled data or additional task-specific training, which is expensive at scale. In this work, we introduce LoRA on the Go (LoGo), a training-free framework that dynamically selects and merges adapters at the instance level without any additional requirements. LoGo leverages signals extracted from a single forward pass through LoRA adapters, to identify the most relevant adapters and determine their contributions on-the-fly. Across 5 NLP benchmarks, 27 datasets, and 3 model families, LoGo outperforms training-based baselines on some tasks upto a margin of 3.6% while remaining competitive on other tasks and maintaining inference throughput, highlighting its effectiveness and practicality. 

---
# Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought 

**Authors**: Zhikang Chen, Sen Cui, Deheng Ye, Yu Zhang, Yatao Bian, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07124)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities through \emph{Chain-of-Thought} (CoT) prompting, which enables step-by-step intermediate reasoning. However, explicit CoT methods rely on discrete token-level reasoning processes that are prone to error propagation and limited by vocabulary expressiveness, often resulting in rigid and inconsistent reasoning trajectories. Recent research has explored implicit or continuous reasoning in latent spaces, allowing models to perform internal reasoning before generating explicit output. Although such approaches alleviate some limitations of discrete CoT, they generally lack explicit mechanisms to enforce consistency among reasoning steps, leading to divergent reasoning paths and unstable outcomes. To address this issue, we propose EBM-CoT, an Energy-Based Chain-of-Thought Calibration framework that refines latent thought representations through an energy-based model (EBM). Our method dynamically adjusts latent reasoning trajectories toward lower-energy, high-consistency regions in the embedding space, improving both reasoning accuracy and consistency without modifying the base language model. Extensive experiments across mathematical, commonsense, and symbolic reasoning benchmarks demonstrate that the proposed framework significantly enhances the consistency and efficiency of multi-step reasoning in LLMs. 

---
# On the Joint Minimization of Regularization Loss Functions in Deep Variational Bayesian Methods for Attribute-Controlled Symbolic Music Generation 

**Authors**: Matteo Pettenó, Alessandro Ilic Mezza, Alberto Bernardini  

**Link**: [PDF](https://arxiv.org/pdf/2511.07118)  

**Abstract**: Explicit latent variable models provide a flexible yet powerful framework for data synthesis, enabling controlled manipulation of generative factors. With latent variables drawn from a tractable probability density function that can be further constrained, these models enable continuous and semantically rich exploration of the output space by navigating their latent spaces. Structured latent representations are typically obtained through the joint minimization of regularization loss functions. In variational information bottleneck models, reconstruction loss and Kullback-Leibler Divergence (KLD) are often linearly combined with an auxiliary Attribute-Regularization (AR) loss. However, balancing KLD and AR turns out to be a very delicate matter. When KLD dominates over AR, generative models tend to lack controllability; when AR dominates over KLD, the stochastic encoder is encouraged to violate the standard normal prior. We explore this trade-off in the context of symbolic music generation with explicit control over continuous musical attributes. We show that existing approaches struggle to jointly minimize both regularization objectives, whereas suitable attribute transformations can help achieve both controllability and regularization of the target latent dimensions. 

---
# More Agents Helps but Adversarial Robustness Gap Persists 

**Authors**: Khashayar Alavi, Zhastay Yeltay, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07112)  

**Abstract**: When LLM agents work together, they seem to be more powerful than a single LLM in mathematical question answering. However, are they also more robust to adversarial inputs? We investigate this question using adversarially perturbed math questions. These perturbations include punctuation noise with three intensities (10, 30, and 50 percent), plus real-world and human-like typos (WikiTypo, R2ATA). Using a unified sampling-and-voting framework (Agent Forest), we evaluate six open-source models (Qwen3-4B/14B, Llama3.1-8B, Mistral-7B, Gemma3-4B/12B) across four benchmarks (GSM8K, MATH, MMLU-Math, MultiArith), with various numbers of agents n from one to 25 (1, 2, 5, 10, 15, 20, 25). Our findings show that (1) Noise type matters: punctuation noise harm scales with its severity, and the human typos remain the dominant bottleneck, yielding the largest gaps to Clean accuracy and the highest ASR even with a large number of agents. And (2) Collaboration reliably improves accuracy as the number of agents, n, increases, with the largest gains from one to five agents and diminishing returns beyond 10 agents. However, the adversarial robustness gap persists regardless of the agent count. 

---
# GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution 

**Authors**: Sirui Wang, Jiang He, Natàlia Blasco Andreo, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07103)  

**Abstract**: Improving the quality of hyperspectral images (HSIs), such as through super-resolution, is a crucial research area. However, generative modeling for HSIs presents several challenges. Due to their high spectral dimensionality, HSIs are too memory-intensive for direct input into conventional diffusion models. Furthermore, general generative models lack an understanding of the topological and geometric structures of ground objects in remote sensing imagery. In addition, most diffusion models optimize loss functions at the noise level, leading to a non-intuitive convergence behavior and suboptimal generation quality for complex data. To address these challenges, we propose a Geometric Enhanced Wavelet-based Diffusion Model (GEWDiff), a novel framework for reconstructing hyperspectral images at 4-times super-resolution. A wavelet-based encoder-decoder is introduced that efficiently compresses HSIs into a latent space while preserving spectral-spatial information. To avoid distortion during generation, we incorporate a geometry-enhanced diffusion process that preserves the geometric features. Furthermore, a multi-level loss function was designed to guide the diffusion process, promoting stable convergence and improved reconstruction fidelity. Our model demonstrated state-of-the-art results across multiple dimensions, including fidelity, spectral accuracy, visual realism, and clarity. 

---
# E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis 

**Authors**: Zhisheng Zhang, Derui Wang, Yifan Mi, Zhiyong Wu, Jie Gao, Yuxin Cao, Kai Ye, Minhui Xue, Jie Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07099)  

**Abstract**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at this https URL. 

---
# Sample-efficient quantum error mitigation via classical learning surrogates 

**Authors**: Wei-You Liao, Ge Yan, Yujin Song, Tian-Ci Tian, Wei-Ming Zhu, De-Tao Jiang, Yuxuan Du, He-Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07092)  

**Abstract**: The pursuit of practical quantum utility on near-term quantum processors is critically challenged by their inherent noise. Quantum error mitigation (QEM) techniques are leading solutions to improve computation fidelity with relatively low qubit-overhead, while full-scale quantum error correction remains a distant goal. However, QEM techniques incur substantial measurement overheads, especially when applied to families of quantum circuits parameterized by classical inputs. Focusing on zero-noise extrapolation (ZNE), a widely adopted QEM technique, here we devise the surrogate-enabled ZNE (S-ZNE), which leverages classical learning surrogates to perform ZNE entirely on the classical side. Unlike conventional ZNE, whose measurement cost scales linearly with the number of circuits, S-ZNE requires only constant measurement overhead for an entire family of quantum circuits, offering superior scalability. Theoretical analysis indicates that S-ZNE achieves accuracy comparable to conventional ZNE in many practical scenarios, and numerical experiments on up to 100-qubit ground-state energy and quantum metrology tasks confirm its effectiveness. Our approach provides a template that can be effectively extended to other quantum error mitigation protocols, opening a promising path toward scalable error mitigation. 

---
# How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions 

**Authors**: Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07091)  

**Abstract**: Text-to-image generative models often exhibit bias related to sensitive attributes. However, current research tends to focus narrowly on single-object prompts with limited contextual diversity. In reality, each object or attribute within a prompt can contribute to bias. For example, the prompt "an assistant wearing a pink hat" may reflect female-inclined biases associated with a pink hat. The neglected joint effects of the semantic binding in the prompts cause significant failures in current debiasing approaches. This work initiates a preliminary investigation on how bias manifests under semantic binding, where contextual associations between objects and attributes influence generative outcomes. We demonstrate that the underlying bias distribution can be amplified based on these associations. Therefore, we introduce a bias adherence score that quantifies how specific object-attribute bindings activate bias. To delve deeper, we develop a training-free context-bias control framework to explore how token decoupling can facilitate the debiasing of semantic bindings. This framework achieves over 10% debiasing improvement in compositional generation tasks. Our analysis of bias scores across various attribute-object bindings and token decorrelation highlights a fundamental challenge: reducing bias without disrupting essential semantic relationships. These findings expose critical limitations in current debiasing approaches when applied to semantically bound contexts, underscoring the need to reassess prevailing bias mitigation strategies. 

---
# Achieving Effective Virtual Reality Interactions via Acoustic Gesture Recognition based on Large Language Models 

**Authors**: Xijie Zhang, Fengliang He, Hong-Ning Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07085)  

**Abstract**: Natural and efficient interaction remains a critical challenge for virtual reality and augmented reality (VR/AR) systems. Vision-based gesture recognition suffers from high computational cost, sensitivity to lighting conditions, and privacy leakage concerns. Acoustic sensing provides an attractive alternative: by emitting inaudible high-frequency signals and capturing their reflections, channel impulse response (CIR) encodes how gestures perturb the acoustic field in a low-cost and user-transparent manner. However, existing CIR-based gesture recognition methods often rely on extensive training of models on large labeled datasets, making them unsuitable for few-shot VR scenarios. In this work, we propose the first framework that leverages large language models (LLMs) for CIR-based gesture recognition in VR/AR systems. Despite LLMs' strengths, it is non-trivial to achieve few-shot and zero-shot learning of CIR gestures due to their inconspicuous features. To tackle this challenge, we collect differential CIR rather than original CIR data. Moreover, we construct a real-world dataset collected from 10 participants performing 15 gestures across three categories (digits, letters, and shapes), with 10 repetitions each. We then conduct extensive experiments on this dataset using an LLM-adopted classifier. Results show that our LLM-based framework achieves accuracy comparable to classical machine learning baselines, while requiring no domain-specific retraining. 

---
# Pandar128 dataset for lane line detection 

**Authors**: Filip Beránek, Václav Diviš, Ivan Gruber  

**Link**: [PDF](https://arxiv.org/pdf/2511.07084)  

**Abstract**: We present Pandar128, the largest public dataset for lane line detection using a 128-beam LiDAR. It contains over 52,000 camera frames and 34,000 LiDAR scans, captured in diverse real-world conditions in Germany. The dataset includes full sensor calibration (intrinsics, extrinsics) and synchronized odometry, supporting tasks such as projection, fusion, and temporal modeling.
To complement the dataset, we also introduce SimpleLidarLane, a light-weight baseline method for lane line reconstruction that combines BEV segmentation, clustering, and polyline fitting. Despite its simplicity, our method achieves strong performance under challenging various conditions (e.g., rain, sparse returns), showing that modular pipelines paired with high-quality data and principled evaluation can compete with more complex approaches.
Furthermore, to address the lack of standardized evaluation, we propose a novel polyline-based metric - Interpolation-Aware Matching F1 (IAM-F1) - that employs interpolation-aware lateral matching in BEV space.
All data and code are publicly released to support reproducibility in LiDAR-based lane detection. 

---
# Wasm: A Pipeline for Constructing Structured Arabic Interleaved Multimodal Corpora 

**Authors**: Khalil Hennara, Ahmad Bastati, Muhammad Hreden, Mohamed Motasim Hamed, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07080)  

**Abstract**: The performance of large language models (LLMs) and large multimodal models (LMMs) depends heavily on the quality and scale of their pre-training datasets. Recent research shows that large multimodal models trained on natural documents where images and text are interleaved outperform those trained only on image-text pairs across a wide range of benchmarks, leveraging advanced pre- trained models to enforce semantic alignment, image-sequence consistency, and textual coherence. For Arabic, however, the lack of high-quality multimodal datasets that preserve document structure has limited progress. In this paper, we present our pipeline Wasm for processing the Common Crawl dataset to create a new Arabic multimodal dataset that uniquely provides markdown output. Unlike existing Arabic corpora that focus solely on text extraction, our approach preserves the structural integrity of web content while maintaining flexibility for both text-only and multimodal pre-training scenarios. We provide a comprehensive comparative analysis of our data processing pipeline against those used for major existing datasets, highlighting the convergences in filtering strategies and justifying our specific design choices. To support future research, we publicly release a representative dataset dump along with the multimodal processing pipeline for Arabic. 

---
# TauFlow: Dynamic Causal Constraint for Complexity-Adaptive Lightweight Segmentation 

**Authors**: Zidong Chen, Fadratul Hafinaz Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07057)  

**Abstract**: Deploying lightweight medical image segmentation models on edge devices presents two major challenges: 1) efficiently handling the stark contrast between lesion boundaries and background regions, and 2) the sharp drop in accuracy that occurs when pursuing extremely lightweight designs (e.g., <0.5M parameters). To address these problems, this paper proposes TauFlow, a novel lightweight segmentation model. The core of TauFlow is a dynamic feature response strategy inspired by brain-like mechanisms. This is achieved through two key innovations: the Convolutional Long-Time Constant Cell (ConvLTC), which dynamically regulates the feature update rate to "slowly" process low-frequency backgrounds and "quickly" respond to high-frequency boundaries; and the STDP Self-Organizing Module, which significantly mitigates feature conflicts between the encoder and decoder, reducing the conflict rate from approximately 35%-40% to 8%-10%. 

---
# Learning Quantized Continuous Controllers for Integer Hardware 

**Authors**: Fabian Kresse, Christoph H. Lampert  

**Link**: [PDF](https://arxiv.org/pdf/2511.07046)  

**Abstract**: Deploying continuous-control reinforcement learning policies on embedded hardware requires meeting tight latency and power budgets. Small FPGAs can deliver these, but only if costly floating point pipelines are avoided. We study quantization-aware training (QAT) of policies for integer inference and we present a learning-to-hardware pipeline that automatically selects low-bit policies and synthesizes them to an Artix-7 FPGA. Across five MuJoCo tasks, we obtain policy networks that are competitive with full precision (FP32) policies but require as few as 3 or even only 2 bits per weight, and per internal activation value, as long as input precision is chosen carefully. On the target hardware, the selected policies achieve inference latencies on the order of microseconds and consume microjoules per action, favorably comparing to a quantized reference. Last, we observe that the quantized policies exhibit increased input noise robustness compared to the floating-point baseline. 

---
# Benchmarking LLMs for Fine-Grained Code Review with Enriched Context in Practice 

**Authors**: Ruida Hu, Xinchen Wang, Xin-Cheng Wen, Zhao Zhang, Bo Jiang, Pengfei Gao, Chao Peng, Cuiyun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07017)  

**Abstract**: Code review is a cornerstone of software quality assurance, and recent advances in Large Language Models (LLMs) have shown promise in automating this process. However, existing benchmarks for LLM-based code review face three major limitations. (1) Lack of semantic context: most benchmarks provide only code diffs without textual information such as issue descriptions, which are crucial for understanding developer intent. (2) Data quality issues: without rigorous validation, many samples are noisy-e.g., reviews on outdated or irrelevant code-reducing evaluation reliability. (3) Coarse granularity: most benchmarks operate at the file or commit level, overlooking the fine-grained, line-level reasoning essential for precise review.
We introduce ContextCRBench, a high-quality, context-rich benchmark for fine-grained LLM evaluation in code review. Our construction pipeline comprises: (1) Raw Data Crawling, collecting 153.7K issues and pull requests from top-tier repositories; (2) Comprehensive Context Extraction, linking issue-PR pairs for textual context and extracting the full surrounding function or class for code context; and (3) Multi-stage Data Filtering, combining rule-based and LLM-based validation to remove outdated, malformed, or low-value samples, resulting in 67,910 context-enriched entries.
ContextCRBench supports three evaluation scenarios aligned with the review workflow: (1) hunk-level quality assessment, (2) line-level defect localization, and (3) line-level comment generation. Evaluating eight leading LLMs (four closed-source and four open-source) reveals that textual context yields greater performance gains than code context alone, while current LLMs remain far from human-level review ability. Deployed at ByteDance, ContextCRBench drives a self-evolving code review system, improving performance by 61.98% and demonstrating its robustness and industrial utility. 

---
# Diffolio: A Diffusion Model for Multivariate Probabilistic Financial Time-Series Forecasting and Portfolio Construction 

**Authors**: So-Yoon Cho, Jin-Young Kim, Kayoung Ban, Hyeng Keun Koo, Hyun-Gyoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07014)  

**Abstract**: Probabilistic forecasting is crucial in multivariate financial time-series for constructing efficient portfolios that account for complex cross-sectional dependencies. In this paper, we propose Diffolio, a diffusion model designed for multivariate financial time-series forecasting and portfolio construction. Diffolio employs a denoising network with a hierarchical attention architecture, comprising both asset-level and market-level layers. Furthermore, to better reflect cross-sectional correlations, we introduce a correlation-guided regularizer informed by a stable estimate of the target correlation matrix. This structure effectively extracts salient features not only from historical returns but also from asset-specific and systematic covariates, significantly enhancing the performance of forecasts and portfolios. Experimental results on the daily excess returns of 12 industry portfolios show that Diffolio outperforms various probabilistic forecasting baselines in multivariate forecasting accuracy and portfolio performance. Moreover, in portfolio experiments, portfolios constructed from Diffolio's forecasts show consistently robust performance, thereby outperforming those from benchmarks by achieving higher Sharpe ratios for the mean-variance tangency portfolio and higher certainty equivalents for the growth-optimal portfolio. These results demonstrate the superiority of our proposed Diffolio in terms of not only statistical accuracy but also economic significance. 

---
# TrueCity: Real and Simulated Urban Data for Cross-Domain 3D Scene Understanding 

**Authors**: Duc Nguyen, Yan-Ling Lai, Qilin Zhang, Prabin Gyawali, Benedikt Schwab, Olaf Wysocki, Thomas H. Kolbe  

**Link**: [PDF](https://arxiv.org/pdf/2511.07007)  

**Abstract**: 3D semantic scene understanding remains a long-standing challenge in the 3D computer vision community. One of the key issues pertains to limited real-world annotated data to facilitate generalizable models. The common practice to tackle this issue is to simulate new data. Although synthetic datasets offer scalability and perfect labels, their designer-crafted scenes fail to capture real-world complexity and sensor noise, resulting in a synthetic-to-real domain gap. Moreover, no benchmark provides synchronized real and simulated point clouds for segmentation-oriented domain shift analysis. We introduce TrueCity, the first urban semantic segmentation benchmark with cm-accurate annotated real-world point clouds, semantic 3D city models, and annotated simulated point clouds representing the same city. TrueCity proposes segmentation classes aligned with international 3D city modeling standards, enabling consistent evaluation of synthetic-to-real gap. Our extensive experiments on common baselines quantify domain shift and highlight strategies for exploiting synthetic data to enhance real-world 3D scene understanding. We are convinced that the TrueCity dataset will foster further development of sim-to-real gap quantification and enable generalizable data-driven models. The data, code, and 3D models are available online: this https URL 

---
# S$^2$Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening 

**Authors**: Bowei He, Bowen Gao, Yankai Chen, Yanyan Lan, Chen Ma, Philip S. Yu, Ya-Qin Zhang, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.07006)  

**Abstract**: Virtual screening (VS) is an essential task in drug discovery, focusing on the identification of small-molecule ligands that bind to specific protein pockets. Existing deep learning methods, from early regression models to recent contrastive learning approaches, primarily rely on structural data while overlooking protein sequences, which are more accessible and can enhance generalizability. However, directly integrating protein sequences poses challenges due to the redundancy and noise in large-scale protein-ligand datasets. To address these limitations, we propose \textbf{S$^2$Drug}, a two-stage framework that explicitly incorporates protein \textbf{S}equence information and 3D \textbf{S}tructure context in protein-ligand contrastive representation learning. In the first stage, we perform protein sequence pretraining on ChemBL using an ESM2-based backbone, combined with a tailored data sampling strategy to reduce redundancy and noise on both protein and ligand sides. In the second stage, we fine-tune on PDBBind by fusing sequence and structure information through a residue-level gating module, while introducing an auxiliary binding site prediction task. This auxiliary task guides the model to accurately localize binding residues within the protein sequence and capture their 3D spatial arrangement, thereby refining protein-ligand matching. Across multiple benchmarks, S$^2$Drug consistently improves virtual screening performance and achieves strong results on binding site prediction, demonstrating the value of bridging sequence and structure in contrastive learning. 

---
# Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings 

**Authors**: Erel Naor, Ofir Lindenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2511.06961)  

**Abstract**: Deep neural networks often under-perform on tabular data due to their sensitivity to irrelevant features and a spectral bias toward smooth, low-frequency functions. These limitations hinder their ability to capture the sharp, high-frequency signals that often define tabular structure, especially under limited labeled samples. While self-supervised learning (SSL) offers promise in such settings, it remains challenging in tabular domains due to the lack of effective data augmentations. We propose a hybrid autoencoder that combines a neural encoder with an oblivious soft decision tree (OSDT) encoder, each guided by its own stochastic gating network that performs sample-specific feature selection. Together, these structurally different encoders and model-specific gating networks implement model-based augmentation, producing complementary input views tailored to each architecture. The two encoders, trained with a shared decoder and cross-reconstruction loss, learn distinct yet aligned representations that reflect their respective inductive biases. During training, the OSDT encoder (robust to noise and effective at modeling localized, high-frequency structure) guides the neural encoder toward representations more aligned with tabular data. At inference, only the neural encoder is used, preserving flexibility and SSL compatibility. Spectral analysis highlights the distinct inductive biases of each encoder. Our method achieves consistent gains in low-label classification and regression across diverse tabular datasets, outperforming deep and tree-based supervised baselines. 

---
# FoCLIP: A Feature-Space Misalignment Framework for CLIP-Based Image Manipulation and Detection 

**Authors**: Yulin Chen, Zeyuan Wang, Tianyuan Yu, Yingmei Wei, Liang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06947)  

**Abstract**: The well-aligned attribute of CLIP-based models enables its effective application like CLIPscore as a widely adopted image quality assessment metric. However, such a CLIP-based metric is vulnerable for its delicate multimodal alignment. In this work, we propose \textbf{FoCLIP}, a feature-space misalignment framework for fooling CLIP-based image quality metric. Based on the stochastic gradient descent technique, FoCLIP integrates three key components to construct fooling examples: feature alignment as the core module to reduce image-text modality gaps, the score distribution balance module and pixel-guard regularization, which collectively optimize multimodal output equilibrium between CLIPscore performance and image quality. Such a design can be engineered to maximize the CLIPscore predictions across diverse input prompts, despite exhibiting either visual unrecognizability or semantic incongruence with the corresponding adversarial prompts from human perceptual perspectives. Experiments on ten artistic masterpiece prompts and ImageNet subsets demonstrate that optimized images can achieve significant improvement in CLIPscore while preserving high visual fidelity. In addition, we found that grayscale conversion induces significant feature degradation in fooling images, exhibiting noticeable CLIPscore reduction while preserving statistical consistency with original images. Inspired by this phenomenon, we propose a color channel sensitivity-driven tampering detection mechanism that achieves 91% accuracy on standard benchmarks. In conclusion, this work establishes a practical pathway for feature misalignment in CLIP-based multimodal systems and the corresponding defense method. 

---
# Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning 

**Authors**: Daniel De Dios Allegue, Jinke He, Frans A. Oliehoek  

**Link**: [PDF](https://arxiv.org/pdf/2511.06946)  

**Abstract**: Transformers have shown strong ability to model long-term dependencies and are increasingly adopted as world models in model-based reinforcement learning (RL) under partial observability. However, unlike natural language corpora, RL trajectories are sparse and reward-driven, making standard self-attention inefficient because it distributes weight uniformly across all past tokens rather than emphasizing the few transitions critical for control. To address this, we introduce structured inductive priors into the self-attention mechanism of the dynamics head: (i) per-head memory-length priors that constrain attention to task-specific windows, and (ii) distributional priors that learn smooth Gaussian weightings over past state-action pairs. We integrate these mechanisms into UniZero, a model-based RL agent with a Transformer-based world model that supports planning under partial observability. Experiments on the Atari 100k benchmark show that most efficiency gains arise from the Gaussian prior, which smoothly allocates attention to informative transitions, while memory-length priors often truncate useful signals with overly restrictive cut-offs. In particular, Gaussian Attention achieves a 77% relative improvement in mean human-normalized scores over UniZero. These findings suggest that in partially observable RL domains with non-stationary temporal dependencies, discrete memory windows are difficult to learn reliably, whereas smooth distributional priors flexibly adapt across horizons and yield more robust data efficiency. Overall, our results demonstrate that encoding structured temporal priors directly into self-attention improves the prioritization of informative histories for dynamics modeling under partial observability. 

---
# From Attribution to Action: Jointly ALIGNing Predictions and Explanations 

**Authors**: Dongsheng Hong, Chao Chen, Yanhui Chen, Shanshan Lin, Zhihao Chen, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06944)  

**Abstract**: Explanation-guided learning (EGL) has shown promise in aligning model predictions with interpretable reasoning, particularly in computer vision tasks. However, most approaches rely on external annotations or heuristic-based segmentation to supervise model explanations, which can be noisy, imprecise and difficult to scale. In this work, we provide both empirical and theoretical evidence that low-quality supervision signals can degrade model performance rather than improve it. In response, we propose ALIGN, a novel framework that jointly trains a classifier and a masker in an iterative manner. The masker learns to produce soft, task-relevant masks that highlight informative regions, while the classifier is optimized for both prediction accuracy and alignment between its saliency maps and the learned masks. By leveraging high-quality masks as guidance, ALIGN improves both interpretability and generalizability, showing its superiority across various settings. Experiments on the two domain generalization benchmarks, VLCS and Terra Incognita, show that ALIGN consistently outperforms six strong baselines in both in-distribution and out-of-distribution settings. Besides, ALIGN also yields superior explanation quality concerning sufficiency and comprehensiveness, highlighting its effectiveness in producing accurate and interpretable models. 

---
# PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data 

**Authors**: Ayushi Sharma, Johanna Trost, Daniel Lusk, Johannes Dollinger, Julian Schrader, Christian Rossi, Javier Lopatin, Etienne Laliberté, Simon Haberstroh, Jana Eichel, Daniel Mederer, Jose Miguel Cerda-Paredes, Shyam S. Phartyal, Lisa-Maricia Schwarz, Anja Linstädter, Maria Conceição Caldeira, Teja Kattenborn  

**Link**: [PDF](https://arxiv.org/pdf/2511.06943)  

**Abstract**: Global plant maps of plant traits, such as leaf nitrogen or plant height, are essential for understanding ecosystem processes, including the carbon and energy cycles of the Earth system. However, existing trait maps remain limited by the high cost and sparse geographic coverage of field-based measurements. Citizen science initiatives offer a largely untapped resource to overcome these limitations, with over 50 million geotagged plant photographs worldwide capturing valuable visual information on plant morphology and physiology. In this study, we introduce PlantTraitNet, a multi-modal, multi-task uncertainty-aware deep learning framework that predictsfour key plant traits (plant height, leaf area, specific leaf area, and nitrogen content) from citizen science photos using weak supervision. By aggregating individual trait predictions across space, we generate global maps of trait distributions. We validate these maps against independent vegetation survey data (sPlotOpen) and benchmark them against leading global trait products. Our results show that PlantTraitNet consistently outperforms existing trait maps across all evaluated traits, demonstrating that citizen science imagery, when integrated with computer vision and geospatial AI, enables not only scalable but also more accurate global trait mapping. This approach offers a powerful new pathway for ecological research and Earth system modeling. 

---
# Fine-Tuning Diffusion-Based Recommender Systems via Reinforcement Learning with Reward Function Optimization 

**Authors**: Yu Hou, Hua Li, Ha Young Kim, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06937)  

**Abstract**: Diffusion models recently emerged as a powerful paradigm for recommender systems, offering state-of-the-art performance by modeling the generative process of user-item interactions. However, training such models from scratch is both computationally expensive and yields diminishing returns once convergence is reached. To remedy these challenges, we propose ReFiT, a new framework that integrates Reinforcement learning (RL)-based Fine-Tuning into diffusion-based recommender systems. In contrast to prior RL approaches for diffusion models depending on external reward models, ReFiT adopts a task-aligned design: it formulates the denoising trajectory as a Markov decision process (MDP) and incorporates a collaborative signal-aware reward function that directly reflects recommendation quality. By tightly coupling the MDP structure with this reward signal, ReFiT empowers the RL agent to exploit high-order connectivity for fine-grained optimization, while avoiding the noisy or uninformative feedback common in naive reward designs. Leveraging policy gradient optimization, ReFiT maximizes exact log-likelihood of observed interactions, thereby enabling effective post hoc fine-tuning of diffusion recommenders. Comprehensive experiments on wide-ranging real-world datasets demonstrate that the proposed ReFiT framework (a) exhibits substantial performance gains over strong competitors (up to 36.3% on sequential recommendation), (b) demonstrates strong efficiency with linear complexity in the number of users or items, and (c) generalizes well across multiple diffusion-based recommendation scenarios. The source code and datasets are publicly available at this https URL. 

---
# Sampling and Loss Weights in Multi-Domain Training 

**Authors**: Mahdi Salmani, Pratik Worah, Meisam Razaviyayn, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2511.06913)  

**Abstract**: In the training of large deep neural networks, there is a need for vast amounts of training data. To meet this need, data is collected from multiple domains, such as Wikipedia and GitHub. These domains are heterogeneous in both data quality and the diversity of information they provide. This raises the question of how much we should rely on each domain. Several methods have attempted to address this issue by assigning sampling weights to each data domain using heuristics or approximations. As a first step toward a deeper understanding of the role of data mixing, this work revisits the problem by studying two kinds of weights: sampling weights, which control how much each domain contributes in a batch, and loss weights, which scale the loss from each domain during training. Through a rigorous study of linear regression, we show that these two weights play complementary roles. First, they can reduce the variance of gradient estimates in iterative methods such as stochastic gradient descent (SGD). Second, they can improve generalization performance by reducing the generalization gap. We provide both theoretical and empirical support for these claims. We further study the joint dynamics of sampling weights and loss weights, examining how they can be combined to capture both contributions. 

---
# Counterfactual Explanation for Multivariate Time Series Forecasting with Exogenous Variables 

**Authors**: Keita Kinjo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06906)  

**Abstract**: Currently, machine learning is widely used across various domains, including time series data analysis. However, some machine learning models function as black boxes, making interpretability a critical concern. One approach to address this issue is counterfactual explanation (CE), which aims to provide insights into model predictions. This study focuses on the relatively underexplored problem of generating counterfactual explanations for time series forecasting. We propose a method for extracting CEs in time series forecasting using exogenous variables, which are frequently encountered in fields such as business and marketing. In addition, we present methods for analyzing the influence of each variable over an entire time series, generating CEs by altering only specific variables, and evaluating the quality of the resulting CEs. We validate the proposed method through theoretical analysis and empirical experiments, showcasing its accuracy and practical applicability. These contributions are expected to support real-world decision-making based on time series data analysis. 

---
# RPTS: Tree-Structured Reasoning Process Scoring for Faithful Multimodal Evaluation 

**Authors**: Haofeng Wang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06899)  

**Abstract**: Large Vision-Language Models (LVLMs) excel in multimodal reasoning and have shown impressive performance on various multimodal benchmarks. However, most of these benchmarks evaluate models primarily through multiple-choice or short-answer formats, which do not take the reasoning process into account. Although some benchmarks assess the reasoning process, their methods are often overly simplistic and only examine reasoning when answers are incorrect. This approach overlooks scenarios where flawed reasoning leads to correct answers. In addition, these benchmarks do not consider the impact of intermodal relationships on reasoning. To address this issue, we propose the Reasoning Process Tree Score (RPTS), a tree structure-based metric to assess reasoning processes. Specifically, we organize the reasoning steps into a reasoning tree and leverage its hierarchical information to assign weighted faithfulness scores to each reasoning step. By dynamically adjusting these weights, RPTS not only evaluates the overall correctness of the reasoning, but also pinpoints where the model fails in the reasoning. To validate RPTS in real-world multimodal scenarios, we construct a new benchmark, RPTS-Eval, comprising 374 images and 390 reasoning instances. Each instance includes reliable visual-textual clues that serve as leaf nodes of the reasoning tree. Furthermore, we define three types of intermodal relationships to investigate how intermodal interactions influence the reasoning process. We evaluated representative LVLMs (e.g., GPT4o, Llava-Next), uncovering their limitations in multimodal reasoning and highlighting the differences between open-source and closed-source commercial LVLMs. We believe that this benchmark will contribute to the advancement of research in the field of multimodal reasoning. 

---
# A Hybrid Autoencoder-Transformer Model for Robust Day-Ahead Electricity Price Forecasting under Extreme Conditions 

**Authors**: Boyan Tang, Xuanhao Ren, Peng Xiao, Shunbo Lei, Xiaorong Sun, Jianghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06898)  

**Abstract**: Accurate day-ahead electricity price forecasting (DAEPF) is critical for the efficient operation of power systems, but extreme condition and market anomalies pose significant challenges to existing forecasting methods. To overcome these challenges, this paper proposes a novel hybrid deep learning framework that integrates a Distilled Attention Transformer (DAT) model and an Autoencoder Self-regression Model (ASM). The DAT leverages a self-attention mechanism to dynamically assign higher weights to critical segments of historical data, effectively capturing both long-term trends and short-term fluctuations. Concurrently, the ASM employs unsupervised learning to detect and isolate anomalous patterns induced by extreme conditions, such as heavy rain, heat waves, or human festivals. Experiments on datasets sampled from California and Shandong Province demonstrate that our framework significantly outperforms state-of-the-art methods in prediction accuracy, robustness, and computational efficiency. Our framework thus holds promise for enhancing grid resilience and optimizing market operations in future power systems. 

---
# On The Presence of Double-Descent in Deep Reinforcement Learning 

**Authors**: Viktor Veselý, Aleksandar Todorov, Matthia Sabatelli  

**Link**: [PDF](https://arxiv.org/pdf/2511.06895)  

**Abstract**: The double descent (DD) paradox, where over-parameterized models see generalization improve past the interpolation point, remains largely unexplored in the non-stationary domain of Deep Reinforcement Learning (DRL). We present preliminary evidence that DD exists in model-free DRL, investigating it systematically across varying model capacity using the Actor-Critic framework. We rely on an information-theoretic metric, Policy Entropy, to measure policy uncertainty throughout training. Preliminary results show a clear epoch-wise DD curve; the policy's entrance into the second descent region correlates with a sustained, significant reduction in Policy Entropy. This entropic decay suggests that over-parameterization acts as an implicit regularizer, guiding the policy towards robust, flatter minima in the loss landscape. These findings establish DD as a factor in DRL and provide an information-based mechanism for designing agents that are more general, transferable, and robust. 

---
# COGNOS: Universal Enhancement for Time Series Anomaly Detection via Constrained Gaussian-Noise Optimization and Smoothing 

**Authors**: Wenlong Shang, Peng Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06894)  

**Abstract**: Reconstruction-based methods are a dominant paradigm in time series anomaly detection (TSAD), however, their near-universal reliance on Mean Squared Error (MSE) loss results in statistically flawed reconstruction residuals. This fundamental weakness leads to noisy, unstable anomaly scores with a poor signal-to-noise ratio, hindering reliable detection. To address this, we propose Constrained Gaussian-Noise Optimization and Smoothing (COGNOS), a universal, model-agnostic enhancement framework that tackles this issue at its source. COGNOS introduces a novel Gaussian-White Noise Regularization strategy during training, which directly constrains the model's output residuals to conform to a Gaussian white noise distribution. This engineered statistical property creates the ideal precondition for our second contribution: a Kalman Smoothing Post-processor that provably operates as a statistically optimal estimator to denoise the raw anomaly scores. The synergy between these two components allows COGNOS to robustly separate the true anomaly signal from random fluctuations. Extensive experiments demonstrate that COGNOS is highly effective, delivering an average F-score uplift of 57.9% when applied to 12 diverse backbone models across multiple real-world benchmark datasets. Our work reveals that directly regularizing output statistics is a powerful and generalizable strategy for significantly improving anomaly detection systems. 

---
# DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting 

**Authors**: Daojun Liang, Jing Chen, Xiao Wang, Yinglong Wang, Suo Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06893)  

**Abstract**: Time-Series (TS) exhibits pronounced non-stationarity. Consequently, most forecasting methods display compromised robustness to concept drift, despite the prevalent application of instance normalization. We tackle this challenge by first analysing concept drift through a bias-variance lens and proving that weighted ensemble reduces variance without increasing bias. These insights motivate DeepBooTS, a novel end-to-end dual-stream residual-decreasing boosting method that progressively reconstructs the intrinsic signal. In our design, each block of a deep model becomes an ensemble of learners with an auxiliary output branch forming a highway to the final prediction. The block-wise outputs correct the residuals of previous blocks, leading to a learning-driven decomposition of both inputs and targets. This method enhances versatility and interpretability while substantially improving robustness to concept drift. Extensive experiments, including those on large-scale datasets, show that the proposed method outperforms existing methods by a large margin, yielding an average performance improvement of 15.8% across various datasets, establishing a new benchmark for TS forecasting. 

---
# TuckA: Hierarchical Compact Tensor Experts for Efficient Fine-Tuning 

**Authors**: Qifeng Lei, Zhiyong Yang, Qianqian Xu, Cong Hua, Peisong Wen, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06859)  

**Abstract**: Efficiently fine-tuning pre-trained models for downstream tasks is a key challenge in the era of foundation models. Parameter-efficient fine-tuning (PEFT) presents a promising solution, achieving performance comparable to full fine-tuning by updating only a small number of adaptation weights per layer. Traditional PEFT methods typically rely on a single expert, where the adaptation weight is a low-rank matrix. However, for complex tasks, the data's inherent diversity poses a significant challenge for such models, as a single adaptation weight cannot adequately capture the features of all samples. To address this limitation, we explore how to integrate multiple small adaptation experts into a compact structure to defeat a large adapter. Specifically, we propose Tucker Adaptation (TuckA), a method with four key properties: (i) We use Tucker decomposition to create a compact 3D tensor where each slice naturally serves as an expert. The low-rank nature of this decomposition ensures that the number of parameters scales efficiently as more experts are added. (ii) We introduce a hierarchical strategy that organizes these experts into groups at different granularities, allowing the model to capture both local and global data patterns. (iii) We develop an efficient batch-level routing mechanism, which reduces the router's parameter size by a factor of $L$ compared to routing at every adapted layer (where $L$ is the number of adapted layers) (iv) We propose data-aware initialization to achieve loss-free expert load balancing based on theoretical analysis. Extensive experiments on benchmarks in natural language understanding, image classification, and mathematical reasoning speak to the efficacy of TuckA, offering a new and effective solution to the PEFT problem. 

---
# Deep learning EPI-TIRF cross-modality enables background subtraction and axial super-resolution for widefield fluorescence microscopy 

**Authors**: Qiushi Li, Celi Lou, Yanfang Cheng, Bilang Gong, Xinlin Chen, Hao Chen, Baowan Li, Jieli Wang, Yulin Wang, Sipeng Yang, Yunqing Tang, Luru Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06853)  

**Abstract**: The resolving ability of wide-field fluorescence microscopy is fundamentally limited by out-of-focus background owing to its low axial resolution, particularly for densely labeled biological samples. To address this, we developed ET2dNet, a deep learning-based EPI-TIRF cross-modality network that achieves TIRF-comparable background subtraction and axial super-resolution from a single wide-field image without requiring hardware modifications. The model employs a physics-informed hybrid architecture, synergizing supervised learning with registered EPI-TIRF image pairs and self-supervised physical modeling via convolution with the point spread function. This framework ensures exceptional generalization across microscope objectives, enabling few-shot adaptation to new imaging setups. Rigorous validation on cellular and tissue samples confirms ET2dNet's superiority in background suppression and axial resolution enhancement, while maintaining compatibility with deconvolution techniques for lateral resolution improvement. Furthermore, by extending this paradigm through knowledge distillation, we developed ET3dNet, a dedicated three-dimensional reconstruction network that produces artifact-reduced volumetric results. ET3dNet effectively removes out-of-focus background signals even when the input image stack lacks the source of background. This framework makes axial super-resolution imaging more accessible by providing an easy-to-deploy algorithm that avoids additional hardware costs and complexity, showing great potential for live cell studies and clinical histopathology. 

---
# Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment 

**Authors**: Peng Zhang, peijie sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06852)  

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment. 

---
# NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment 

**Authors**: Wenjiang Zhang, Sifeng Wang, Yuwei Su, Xinyu Li, Chen Zhang, Suyu Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06836)  

**Abstract**: Visual neural decoding seeks to reconstruct or infer perceived visual stimuli from brain activity patterns, providing critical insights into human cognition and enabling transformative applications in brain-computer interfaces and artificial intelligence. Current approaches, however, remain constrained by the scarcity of high-quality stimulus-brain response pairs and the inherent semantic mismatch between neural representations and visual content. Inspired by perceptual variability and co-adaptive strategy of the biological systems, we propose a novel self-supervised architecture, named NeuroBridge, which integrates Cognitive Prior Augmentation (CPA) with Shared Semantic Projector (SSP) to promote effective cross-modality alignment. Specifically, CPA simulates perceptual variability by applying asymmetric, modality-specific transformations to both EEG signals and images, enhancing semantic diversity. Unlike previous approaches, SSP establishes a bidirectional alignment process through a co-adaptive strategy, which mutually aligns features from two modalities into a shared semantic space for effective cross-modal learning. NeuroBridge surpasses previous state-of-the-art methods under both intra-subject and inter-subject settings. In the intra-subject scenario, it achieves the improvements of 12.3% in top-1 accuracy and 10.2% in top-5 accuracy, reaching 63.2% and 89.9% respectively on a 200-way zero-shot retrieval task. Extensive experiments demonstrate the effectiveness, robustness, and scalability of the proposed framework for neural visual decoding. 

---
# DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design 

**Authors**: Hector R. Rodriguez, Jiechen Huang, Wenjian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06831)  

**Abstract**: Monte Carlo random walk methods are widely used in capacitance extraction for their mesh-free formulation and inherent parallelism. However, modern semiconductor technologies with densely packed structures present significant challenges in unbiasedly sampling transition domains in walk steps with multiple high-contrast dielectric materials. We present DeepRWCap, a machine learning-guided random walk solver that predicts the transition quantities required to guide each step of the walk. These include Poisson kernels, gradient kernels, signs and magnitudes of weights. DeepRWCap employs a two-stage neural architecture that decomposes structured outputs into face-wise distributions and spatial kernels on cube faces. It uses 3D convolutional networks to capture volumetric dielectric interactions and 2D depthwise separable convolutions to model localized kernel behavior. The design incorporates grid-based positional encodings and structural design choices informed by cube symmetries to reduce learning redundancy and improve generalization. Trained on 100,000 procedurally generated dielectric configurations, DeepRWCap achieves a mean relative error of $1.24\pm0.53$\% when benchmarked against the commercial Raphael solver on the self-capacitance estimation of 10 industrial designs spanning 12 to 55 nm nodes. Compared to the state-of-the-art stochastic difference method Microwalk, DeepRWCap achieves an average 23\% speedup. On complex designs with runtimes over 10 s, it reaches an average 49\% acceleration. 

---
# Beyond Plain Demos: A Demo-centric Anchoring Paradigm for In-Context Learning in Alzheimer's Disease Detection 

**Authors**: Puzhen Su, Haoran Yin, Yongzhu Miao, Jintao Tang, Shasha Li, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06826)  

**Abstract**: Detecting Alzheimer's disease (AD) from narrative transcripts challenges large language models (LLMs): pre-training rarely covers this out-of-distribution task, and all transcript demos describe the same scene, producing highly homogeneous contexts. These factors cripple both the model's built-in task knowledge (\textbf{task cognition}) and its ability to surface subtle, class-discriminative cues (\textbf{contextual perception}). Because cognition is fixed after pre-training, improving in-context learning (ICL) for AD detection hinges on enriching perception through better demonstration (demo) sets. We demonstrate that standard ICL quickly saturates, its demos lack diversity (context width) and fail to convey fine-grained signals (context depth), and that recent task vector (TV) approaches improve broad task adaptation by injecting TV into the LLMs' hidden states (HSs), they are ill-suited for AD detection due to the mismatch of injection granularity, strength and position. To address these bottlenecks, we introduce \textbf{DA4ICL}, a demo-centric anchoring framework that jointly expands context width via \emph{\textbf{Diverse and Contrastive Retrieval}} (DCR) and deepens each demo's signal via \emph{\textbf{Projected Vector Anchoring}} (PVA) at every Transformer layer. Across three AD benchmarks, DA4ICL achieves large, stable gains over both ICL and TV baselines, charting a new paradigm for fine-grained, OOD and low-resource LLM adaptation. 

---
# TiS-TSL: Image-Label Supervised Surgical Video Stereo Matching via Time-Switchable Teacher-Student Learning 

**Authors**: Rui Wang, Ying Zhou, Hao Wang, Wenwei Zhang, Qiang Li, Zhiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06817)  

**Abstract**: Stereo matching in minimally invasive surgery (MIS) is essential for next-generation navigation and augmented reality. Yet, dense disparity supervision is nearly impossible due to anatomical constraints, typically limiting annotations to only a few image-level labels acquired before the endoscope enters deep body cavities. Teacher-Student Learning (TSL) offers a promising solution by leveraging a teacher trained on sparse labels to generate pseudo labels and associated confidence maps from abundant unlabeled surgical videos. However, existing TSL methods are confined to image-level supervision, providing only spatial confidence and lacking temporal consistency estimation. This absence of spatio-temporal reliability results in unstable disparity predictions and severe flickering artifacts across video frames. To overcome these challenges, we propose TiS-TSL, a novel time-switchable teacher-student learning framework for video stereo matching under minimal supervision. At its core is a unified model that operates in three distinct modes: Image-Prediction (IP), Forward Video-Prediction (FVP), and Backward Video-Prediction (BVP), enabling flexible temporal modeling within a single architecture. Enabled by this unified model, TiS-TSL adopts a two-stage learning strategy. The Image-to-Video (I2V) stage transfers sparse image-level knowledge to initialize temporal modeling. The subsequent Video-to-Video (V2V) stage refines temporal disparity predictions by comparing forward and backward predictions to calculate bidirectional spatio-temporal consistency. This consistency identifies unreliable regions across frames, filters noisy video-level pseudo labels, and enforces temporal coherence. Experimental results on two public datasets demonstrate that TiS-TSL exceeds other image-based state-of-the-arts by improving TEPE and EPE by at least 2.11% and 4.54%, respectively.. 

---
# Controllable Flow Matching for Online Reinforcement Learning 

**Authors**: Bin Wang, Boxiang Tao, Haifeng Jing, Hongbo Dou, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06816)  

**Abstract**: Model-based reinforcement learning (MBRL) typically relies on modeling environment dynamics for data efficiency. However, due to the accumulation of model errors over long-horizon rollouts, such methods often face challenges in maintaining modeling stability. To address this, we propose CtrlFlow, a trajectory-level synthetic method using conditional flow matching (CFM), which directly modeling the distribution of trajectories from initial states to high-return terminal states without explicitly modeling the environment transition function. Our method ensures optimal trajectory sampling by minimizing the control energy governed by the non-linear Controllability Gramian Matrix, while the generated diverse trajectory data significantly enhances the robustness and cross-task generalization of policy learning. In online settings, CtrlFlow demonstrates the better performance on common MuJoCo benchmark tasks than dynamics models and achieves superior sample efficiency compared to standard MBRL methods. 

---
# AgentSUMO: An Agentic Framework for Interactive Simulation Scenario Generation in SUMO via Large Language Models 

**Authors**: Minwoo Jeong, Jeeyun Chang, Yoonjin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.06804)  

**Abstract**: The growing complexity of urban mobility systems has made traffic simulation indispensable for evidence-based transportation planning and policy evaluation. However, despite the analytical capabilities of platforms such as the Simulation of Urban MObility (SUMO), their application remains largely confined to domain experts. Developing realistic simulation scenarios requires expertise in network construction, origin-destination modeling, and parameter configuration for policy experimentation, creating substantial barriers for non-expert users such as policymakers, urban planners, and city officials. Moreover, the requests expressed by these users are often incomplete and abstract-typically articulated as high-level objectives, which are not well aligned with the imperative, sequential workflows employed in existing language-model-based simulation frameworks. To address these challenges, this study proposes AgentSUMO, an agentic framework for interactive simulation scenario generation via large language models. AgentSUMO departs from imperative, command-driven execution by introducing an adaptive reasoning layer that interprets user intents, assesses task complexity, infers missing parameters, and formulates executable simulation plans. The framework is structured around two complementary components, the Interactive Planning Protocol, which governs reasoning and user interaction, and the Model Context Protocol, which manages standardized communication and orchestration among simulation tools. Through this design, AgentSUMO converts abstract policy objectives into executable simulation scenarios. Experiments on urban networks in Seoul and Manhattan demonstrate that the agentic workflow achieves substantial improvements in traffic flow metrics while maintaining accessibility for non-expert users, successfully bridging the gap between policy goals and executable simulation workflows. 

---
# Learning to Fast Unrank in Collaborative Filtering Recommendation 

**Authors**: Junpeng Zhao, Lin Li, Ming Li, Amran Bhuiyan, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06803)  

**Abstract**: Modern data-driven recommendation systems risk memorizing sensitive user behavioral patterns, raising privacy concerns. Existing recommendation unlearning methods, while capable of removing target data influence, suffer from inefficient unlearning speed and degraded performance, failing to meet real-time unlearning demands. Considering the ranking-oriented nature of recommendation systems, we present unranking, the process of reducing the ranking positions of target items while ensuring the formal guarantees of recommendation unlearning. To achieve efficient unranking, we propose Learning to Fast Unrank in Collaborative Filtering Recommendation (L2UnRank), which operates through three key stages: (a) identifying the influenced scope via interaction-based p-hop propagation, (b) computing structural and semantic influences for entities within this scope, and (c) performing efficient, ranking-aware parameter updates guided by influence information. Extensive experiments across multiple datasets and backbone models demonstrate L2UnRank's model-agnostic nature, achieving state-of-the-art unranking effectiveness and maintaining recommendation quality comparable to retraining, while also delivering a 50x speedup over existing methods. Codes are available at this https URL. 

---
# Recursive Dynamics in Fast-Weights Homeostatic Reentry Networks: Toward Reflective Intelligence 

**Authors**: B. G. Chae  

**Link**: [PDF](https://arxiv.org/pdf/2511.06798)  

**Abstract**: This study introduces the Fast-Weights Homeostatic Reentry Layer (FH-RL), a neural mechanism that integrates fast-weight associative memory, homeostatic regularization, and learned reentrant feedback to approximate self-referential computation in neural networks. Unlike standard transformer architectures that operate in a purely feedforward manner during inference, FH-RL enables internal recurrence without external looping, allowing prior latent states to be dynamically re-entered into the ongoing computation stream. We conduct controlled experiments sweeping the reentry gain $\gamma$ and evaluate emergent internal dynamics using three novel metrics: the Information Reentry Ratio (IRR), Eigen-Spectrum Recursion Index (ESRI), and Representational Drift Periodicity (RDP). Results show that reentry quantity increases proportionally with~$\gamma$, while the learned feedback matrix $W_r$ remains bounded and becomes more structured at moderate gains. Critically, a stable reflective band emerges around $\gamma \approx 0.10-0.20$, where internal feedback is maximally expressive yet spectrally stable: IRR rises smoothly, ESRI remains near zero, and RDP exhibits consistent low-frequency cycles. These findings provide quantitative evidence that reflective, thought-like internal processing can arise from a principled balance between feedback amplification and homeostatic regulation, linking modern fast-weight architectures to theories of cortical reentry and recursive cognition. 

---
# Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models 

**Authors**: Kunhao Li, Wenhao Li, Di Wu, Lei Yang, Jun Bai, Ju Jia, Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.06793)  

**Abstract**: Multimodal Large Language Models (MLLMs) extend foundation models to real-world applications by integrating inputs such as text and vision. However, their broad knowledge capacity raises growing concerns about privacy leakage, toxicity mitigation, and intellectual property violations. Machine Unlearning (MU) offers a practical solution by selectively forgetting targeted knowledge while preserving overall model utility. When applied to MLLMs, existing neuron-editing-based MU approaches face two fundamental challenges: (1) forgetting becomes inconsistent across modalities because existing point-wise attribution methods fail to capture the structured, layer-by-layer information flow that connects different modalities; and (2) general knowledge performance declines when sensitive neurons that also support important reasoning paths are pruned, as this disrupts the model's ability to generalize. To alleviate these limitations, we propose a multimodal influential neuron path editor (MIP-Editor) for MU. Our approach introduces modality-specific attribution scores to identify influential neuron paths responsible for encoding forget-set knowledge and applies influential-path-aware neuron-editing via representation misdirection. This strategy also enables effective and coordinated forgetting across modalities while preserving the model's general capabilities. Experimental results demonstrate that MIP-Editor achieves a superior unlearning performance on multimodal tasks, with a maximum forgetting rate of 87.75% and up to 54.26% improvement in general knowledge retention. On textual tasks, MIP-Editor achieves up to 80.65% forgetting and preserves 77.9% of general performance. Codes are available at this https URL. 

---
# Robust Causal Discovery under Imperfect Structural Constraints 

**Authors**: Zidong Wang, Xi Lin, Chuchao He, Xiaoguang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06790)  

**Abstract**: Robust causal discovery from observational data under imperfect prior knowledge remains a significant and largely unresolved challenge. Existing methods typically presuppose perfect priors or can only handle specific, pre-identified error types. And their performance degrades substantially when confronted with flawed constraints of unknown location and type. This decline arises because most of them rely on inflexible and biased thresholding strategies that may conflict with the data distribution. To overcome these limitations, we propose to harmonizes knowledge and data through prior alignment and conflict resolution. First, we assess the credibility of imperfect structural constraints through a surrogate model, which then guides a sparse penalization term measuring the loss between the learned and constrained adjacency matrices. We theoretically prove that, under ideal assumption, the knowledge-driven objective aligns with the data-driven objective. Furthermore, to resolve conflicts when this assumption is violated, we introduce a multi-task learning framework optimized via multi-gradient descent, jointly minimizing both objectives. Our proposed method is robust to both linear and nonlinear settings. Extensive experiments, conducted under diverse noise conditions and structural equation model types, demonstrate the effectiveness and efficiency of our method under imperfect structural constraints. 

---
# Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning 

**Authors**: Lejun Ai, Yulong Li, Haodong Yi, Jixuan Xie, Yue Wang, Jia Liu, Min Chen, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06785)  

**Abstract**: Automatic sleep staging plays a vital role in assessing sleep quality and diagnosing sleep disorders. Most existing methods rely heavily on long and continuous EEG recordings, which poses significant challenges for data acquisition in resource-constrained systems, such as wearable or home-based monitoring systems. In this paper, we propose the task of resource-efficient sleep staging, which aims to reduce the amount of signal collected per sleep epoch while maintaining reliable classification performance. To solve this task, we adopt the masking and prompt learning strategy and propose a novel framework called Mask-Aware Sleep Staging (MASS). Specifically, we design a multi-level masking strategy to promote effective feature modeling under partial and irregular observations. To mitigate the loss of contextual information introduced by masking, we further propose a hierarchical prompt learning mechanism that aggregates unmasked data into a global prompt, serving as a semantic anchor for guiding both patch-level and epoch-level feature modeling. MASS is evaluated on four datasets, demonstrating state-of-the-art performance, especially when the amount of data is very limited. This result highlights its potential for efficient and scalable deployment in real-world low-resource sleep monitoring environments. 

---
# On the Mechanisms of Collaborative Learning in VAE Recommenders 

**Authors**: Tung-Long Vuong, Julien Monteil, Hien Dang, Volodymyr Vaskovych, Trung Le, Vu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06781)  

**Abstract**: Variational Autoencoders (VAEs) are a powerful alternative to matrix factorization for recommendation. A common technique in VAE-based collaborative filtering (CF) consists in applying binary input masking to user interaction vectors, which improves performance but remains underexplored theoretically. In this work, we analyze how collaboration arises in VAE-based CF and show it is governed by latent proximity: we derive a latent sharing radius that informs when an SGD update on one user strictly reduces the loss on another user, with influence decaying as the latent Wasserstein distance increases. We further study the induced geometry: with clean inputs, VAE-based CF primarily exploits \emph{local} collaboration between input-similar users and under-utilizes global collaboration between far-but-related users. We compare two mechanisms that encourage \emph{global} mixing and characterize their trade-offs: (1) $\beta$-KL regularization directly tightens the information bottleneck, promoting posterior overlap but risking representational collapse if too large; (2) input masking induces stochastic geometric contractions and expansions, which can bring distant users onto the same latent neighborhood but also introduce neighborhood drift. To preserve user identity while enabling global consistency, we propose an anchor regularizer that aligns user posteriors with item embeddings, stabilizing users under masking and facilitating signal sharing across related items. Our analyses are validated on the Netflix, MovieLens-20M, and Million Song datasets. We also successfully deployed our proposed algorithm on an Amazon streaming platform following a successful online experiment. 

---
# OntoTune: Ontology-Driven Learning for Query Optimization with Convolutional Models 

**Authors**: Songhui Yue, Yang Shao, Sean Hayes  

**Link**: [PDF](https://arxiv.org/pdf/2511.06780)  

**Abstract**: Query optimization has been studied using machine learning, reinforcement learning, and, more recently, graph-based convolutional networks. Ontology, as a structured, information-rich knowledge representation, can provide context, particularly in learning problems. This paper presents OntoTune, an ontology-based platform for enhancing learning for query optimization. By connecting SQL queries, database metadata, and statistics, the ontology developed in this research is promising in capturing relationships and important determinants of query performance. This research also develops a method to embed ontologies while preserving as much of the relationships and key information as possible, before feeding it into learning algorithms such as tree-based and graph-based convolutional networks. A case study shows how OntoTune's ontology-driven learning delivers performance gains compared with database system default query execution. 

---
# Pedagogical Reflections on the Holistic Cognitive Development (HCD) Framework and AI-Augmented Learning in Creative Computing 

**Authors**: BHojan Anand  

**Link**: [PDF](https://arxiv.org/pdf/2511.06779)  

**Abstract**: This paper presents an expanded account of the Holistic Cognitive Development (HCD) framework for reflective and creative learning in computing education. The HCD framework integrates design thinking, experiential learning, and reflective practice into a unified constructivist pedagogy emphasizing autonomy, ownership, and scaffolding. It is applied across courses in game design (CS3247, CS4350), virtual reality (CS4240), and extended reality systems, where students engage in iterative cycles of thinking, creating, criticizing, and reflecting. The paper also examines how AI-augmented systems such as iReflect, ReflexAI, and Knowledge Graph-enhanced LLM feedback tools operationalize the HCD framework through scalable, personalized feedback. Empirical findings demonstrate improved reflective depth, feedback quality, and learner autonomy. The work advocates a balance of supportive autonomy in supervision, where students practice self-directed inquiry while guided through structured reflection and feedback. 

---
# Data Trajectory Alignment for LLM Domain Adaptation: A Two-Phase Synthesis Framework for Telecommunications Mathematics 

**Authors**: Zhicheng Zhou, Jing Li, Suming Qiu, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06776)  

**Abstract**: General-purpose large language models (LLMs) are increasingly deployed in verticals such as telecommunications, where adaptation is hindered by scarce, low-information-density corpora and tight mobile/edge constraints. We propose Data Trajectory Alignment (DTA), a two-phase, model-agnostic data curation framework that treats solution processes - not only final answers - as first-class supervision. Phase I (Initializing) synthesizes diverse, high-coverage candidates using an ensemble of strong teachers. Phase II (DTA) rewrites teacher solutions to align intermediate steps and presentation style with the target student's inductive biases and then performs signal-aware exemplar selection via agreement checks and reflection-based judging. Instantiated on telecommunications mathematics (e.g., link budgets, SNR/AMC selection, and power-control feasibility), DTA yields state-of-the-art (SOTA) accuracy on TELEMATH without enabling explicit "thinking" modes: 72.45% pass@1, surpassing distilled-only training by +17.65 points and outperforming a strong baseline (Qwen3-32B with thinking enabled) by +2.94 points. Token-shift analyses indicate that DTA concentrates gains on logical-structural discourse markers rather than merely amplifying domain nouns, indicating improved reasoning scaffolding. Under edge-like inference settings, DTA improves efficiency by reducing reliance on multi-sample voting and disabling expensive reasoning heuristics, cutting energy per output token by ~42% versus Qwen3-32B (thinking mode enabled) and end-to-end latency by ~60% versus Qwen3-32B (thinking mode disabled). These results demonstrate that aligning how solutions are produced enables compact, high-yield supervision that is effective for both accuracy and efficiency, offering a practical recipe for domain adaptation in low-resource verticals beyond telecom. 

---
# QUARK: Quantization-Enabled Circuit Sharing for Transformer Acceleration by Exploiting Common Patterns in Nonlinear Operations 

**Authors**: Zhixiong Zhao, Haomin Li, Fangxin Liu, Yuncheng Lu, Zongwu Wang, Tao Yang, Li Jiang, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06767)  

**Abstract**: Transformer-based models have revolutionized computer vision (CV) and natural language processing (NLP) by achieving state-of-the-art performance across a range of benchmarks. However, nonlinear operations in models significantly contribute to inference latency, presenting unique challenges for efficient hardware acceleration. To this end, we propose QUARK, a quantization-enabled FPGA acceleration framework that leverages common patterns in nonlinear operations to enable efficient circuit sharing, thereby reducing hardware resource requirements. QUARK targets all nonlinear operations within Transformer-based models, achieving high-performance approximation through a novel circuit-sharing design tailored to accelerate these operations. Our evaluation demonstrates that QUARK significantly reduces the computational overhead of nonlinear operators in mainstream Transformer architectures, achieving up to a 1.96 times end-to-end speedup over GPU implementations. Moreover, QUARK lowers the hardware overhead of nonlinear modules by more than 50% compared to prior approaches, all while maintaining high model accuracy -- and even substantially boosting accuracy under ultra-low-bit quantization. 

---
# Sensitivity of Small Language Models to Fine-tuning Data Contamination 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2511.06763)  

**Abstract**: Small Language Models (SLMs) are increasingly being deployed in resource-constrained environments, yet their behavioral robustness to data contamination during instruction tuning remains poorly understood. We systematically investigate the contamination sensitivity of 23 SLMs (270M to 4B parameters) across multiple model families by measuring susceptibility to syntactic and semantic transformation types during instruction tuning: syntactic transformations (character and word reversal) and semantic transformations (irrelevant and counterfactual responses), each applied at contamination levels of 25\%, 50\%, 75\%, and 100\%. Our results reveal fundamental asymmetries in vulnerability patterns: syntactic transformations cause catastrophic performance degradation, with character reversal producing near-complete failure across all models regardless of size or family, while semantic transformations demonstrate distinct threshold behaviors and greater resilience in core linguistic capabilities. Critically, we discover a ``\textit{capability curse}" where larger, more capable models become more susceptible to learning semantic corruptions, effectively following harmful instructions more readily, while our analysis of base versus instruction-tuned variants reveals that alignment provides inconsistent robustness benefits, sometimes even reducing resilience. Our work establishes three core contributions: (1) empirical evidence of SLMs' disproportionate vulnerability to syntactic pattern contamination, (2) identification of asymmetric sensitivity patterns between syntactic and semantic transformations, and (3) systematic evaluation protocols for contamination robustness assessment. These findings have immediate deployment implications, suggesting that current robustness assumptions may not hold for smaller models and highlighting the need for contamination-aware training protocols. 

---
# Implicit Federated In-context Learning For Task-Specific LLM Fine-Tuning 

**Authors**: Dongcheng Li, Junhan Chen, Aoxiang Zhou, Chunpei Li, Youquan Xian, Peng Liu, Xianxian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06757)  

**Abstract**: As large language models continue to develop and expand, the extensive public data they rely on faces the risk of depletion. Consequently, leveraging private data within organizations to enhance the performance of large models has emerged as a key challenge. The federated learning paradigm, combined with model fine-tuning techniques, effectively reduces the number of trainable parameters. However,the necessity to process high-dimensional feature spaces results in substantial overall computational overhead. To address this issue, we propose the Implicit Federated In-Context Learning (IFed-ICL) framework. IFed-ICL draws inspiration from federated learning to establish a novel distributed collaborative paradigm, by converting client local context examples into implicit vector representations, it enables distributed collaborative computation during the inference phase and injects model residual streams to enhance model performance. Experiments demonstrate that our proposed method achieves outstanding performance across multiple text classification tasks. Compared to traditional methods, IFed-ICL avoids the extensive parameter updates required by conventional fine-tuning methods while reducing data transmission and local computation at the client level in federated learning. This enables efficient distributed context learning using local private-domain data, significantly improving model performance on specific tasks. 

---
# Hierarchical Spatial-Frequency Aggregation for Spectral Deconvolution Imaging 

**Authors**: Tao Lv, Daoming Zhou, Chenglong Huang, Chongde Zi, Linsen Chen, Xun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06751)  

**Abstract**: Computational spectral imaging (CSI) achieves real-time hyperspectral imaging through co-designed optics and algorithms, but typical CSI methods suffer from a bulky footprint and limited fidelity. Therefore, Spectral Deconvolution imaging (SDI) methods based on PSF engineering have been proposed to achieve high-fidelity compact CSI design recently. However, the composite convolution-integration operations of SDI render the normal-equation coefficient matrix scene-dependent, which hampers the efficient exploitation of imaging priors and poses challenges for accurate reconstruction. To tackle the inherent data-dependent operators in SDI, we introduce a Hierarchical Spatial-Spectral Aggregation Unfolding Framework (HSFAUF). By decomposing subproblems and projecting them into the frequency domain, HSFAUF transforms nonlinear processes into linear mappings, thereby enabling efficient solutions. Furthermore, to integrate spatial-spectral priors during iterative refinement, we propose a Spatial-Frequency Aggregation Transformer (SFAT), which explicitly aggregates information across spatial and frequency domains. By integrating SFAT into HSFAUF, we develop a Transformer-based deep unfolding method, \textbf{H}ierarchical \textbf{S}patial-\textbf{F}requency \textbf{A}ggregation \textbf{U}nfolding \textbf{T}ransformer (HSFAUT), to solve the inverse problem of SDI. Systematic simulated and real experiments show that HSFAUT surpasses SOTA methods with cheaper memory and computational costs, while exhibiting optimal performance on different SDI systems. 

---
# Physically-Grounded Goal Imagination: Physics-Informed Variational Autoencoder for Self-Supervised Reinforcement Learning 

**Authors**: Lan Thi Ha Nguyen, Kien Ton Manh, Anh Do Duc, Nam Pham Hai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06745)  

**Abstract**: Self-supervised goal-conditioned reinforcement learning enables robots to autonomously acquire diverse skills without human supervision. However, a central challenge is the goal setting problem: robots must propose feasible and diverse goals that are achievable in their current environment. Existing methods like RIG (Visual Reinforcement Learning with Imagined Goals) use variational autoencoder (VAE) to generate goals in a learned latent space but have the limitation of producing physically implausible goals that hinder learning efficiency. We propose Physics-Informed RIG (PI-RIG), which integrates physical constraints directly into the VAE training process through a novel Enhanced Physics-Informed Variational Autoencoder (Enhanced p3-VAE), enabling the generation of physically consistent and achievable goals. Our key innovation is the explicit separation of the latent space into physics variables governing object dynamics and environmental factors capturing visual appearance, while enforcing physical consistency through differential equation constraints and conservation laws. This enables the generation of physically consistent and achievable goals that respect fundamental physical principles such as object permanence, collision constraints, and dynamic feasibility. Through extensive experiments, we demonstrate that this physics-informed goal generation significantly improves the quality of proposed goals, leading to more effective exploration and better skill acquisition in visual robotic manipulation tasks including reaching, pushing, and pick-and-place scenarios. 

---
# Rank-1 LoRAs Encode Interpretable Reasoning Signals 

**Authors**: Jake Ward, Paul Riechers, Adam Shai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06739)  

**Abstract**: Reasoning models leverage inference-time compute to significantly enhance the performance of language models on difficult logical tasks, and have become a dominating paradigm in frontier LLMs. Despite their wide adoption, the mechanisms underpinning the enhanced performance of these reasoning models are not well understood. In this work, we show that the majority of new capabilities in reasoning models can be elicited by small, single-rank changes to base model parameters, with many of these changes being interpretable. Specifically, we use a rank-1 LoRA to create a minimal parameter adapter for Qwen-2.5-32B-Instruct which recovers 73-90% of reasoning-benchmark performance compared to a full parameter finetune. We find that the activations of this LoRA are as interpretable as MLP neurons, and fire for reasoning-specific behaviors. Finally, we train a sparse autoencoder on the entire activation state of this LoRA and identify fine-grained and monosemantic features. Our findings highlight that reasoning performance can arise largely from minimal changes to base model parameters, and explore what these changes affect. More broadly, our work shows that parameter-efficient training methods can be used as a targeted lens for uncovering fundamental insights about language model behavior and dynamics. 

---
# Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning 

**Authors**: Chun-Ming Huang, Li-Heng Chang, I-Hsin Chang, An-Sheng Lee, Hao Kuo-Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06731)  

**Abstract**: Deep learning has revolutionized seismic phase picking, yet a paradox persists: high signal-to-noise S-wave predictions consistently fail to cross detection thresholds, oscillating at suppressed amplitudes. We identify this previously unexplained phenomenon as amplitude suppression, which we diagnose through analyzing training histories and loss landscapes. Three interacting factors emerge: S-wave onsets exhibit high temporal uncertainty relative to high-amplitude boundaries; CNN's bias toward sharp amplitude changes anchors predictions to these boundaries rather than subtle onsets; and point-wise Binary Cross-Entropy (BCE) loss lacks lateral corrective forces, providing only vertical gradients that suppress amplitude while temporal gaps persist. This geometric trap points to a shape-then-align solution where stable geometric templates must precede temporal alignment. We implement this through a conditional GAN framework by augmenting conventional BCE training with a discriminator that enforces shape constraints. Training for 10,000 steps, this achieves a 64% increase in effective S-phase detections. Our framework autonomously discovers target geometry without a priori assumptions, offering a generalizable solution for segmentation tasks requiring precise alignment of subtle features near dominant structures. 

---
# S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning 

**Authors**: Jiangwen Dong, Zehui Lin, Wanyu Lin, Mingjin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06727)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance in complex reasoning problems. Their effectiveness highly depends on the specific nature of the task, especially the required domain knowledge. Existing approaches, such as mixture-of-experts, typically operate at the task level; they are too coarse to effectively solve the heterogeneous problems involving multiple subjects. This work proposes a novel framework that performs fine-grained analysis at subject level equipped with a designated multi-agent collaboration strategy for addressing heterogeneous problem reasoning. Specifically, given an input query, we first employ a Graph Neural Network to identify the relevant subjects and infer their interdependencies to generate an \textit{Subject-based Directed Acyclic Graph} (S-DAG), where nodes represent subjects and edges encode information flow. Then we profile the LLM models by assigning each model a subject-specific expertise score, and select the top-performing one for matching corresponding subject of the S-DAG. Such subject-model matching enables graph-structured multi-agent collaboration where information flows from the starting model to the ending model over S-DAG. We curate and release multi-subject subsets of standard benchmarks (MMLU-Pro, GPQA, MedMCQA) to better reflect complex, real-world reasoning tasks. Extensive experiments show that our approach significantly outperforms existing task-level model selection and multi-agent collaboration baselines in accuracy and efficiency. These results highlight the effectiveness of subject-aware reasoning and structured collaboration in addressing complex and multi-subject problems. 

---
# Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View 

**Authors**: Jianyu Qi, Ding Zou, Wenrui Yan, Rui Ma, Jiaxu Li, Zhijie Zheng, Zhiguo Yang, Rongchang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06722)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have spurred significant progress in Chain-of-Thought (CoT) reasoning. Building on the success of Deepseek-R1, researchers extended multimodal reasoning to post-training paradigms based on reinforcement learning (RL), focusing predominantly on mathematical datasets. However, existing post-training paradigms tend to neglect two critical aspects: (1) The lack of quantifiable difficulty metrics capable of strategically screening samples for post-training optimization. (2) Suboptimal post-training paradigms that fail to jointly optimize perception and reasoning capabilities. To address this gap, we propose two novel difficulty-aware sampling strategies: Progressive Image Semantic Masking (PISM) quantifies sample hardness through systematic image degradation, while Cross-Modality Attention Balance (CMAB) assesses cross-modal interaction complexity via attention distribution analysis. Leveraging these metrics, we design a hierarchical training framework that incorporates both GRPO-only and SFT+GRPO hybrid training paradigms, and evaluate them across six benchmark datasets. Experiments demonstrate consistent superiority of GRPO applied to difficulty-stratified samples compared to conventional SFT+GRPO pipelines, indicating that strategic data sampling can obviate the need for supervised fine-tuning while improving model accuracy. Our code will be released at this https URL. 

---
# MirrorMamba: Towards Scalable and Robust Mirror Detection in Videos 

**Authors**: Rui Song, Jiaying Lin, Rynson W.H. Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.06716)  

**Abstract**: Video mirror detection has received significant research attention, yet existing methods suffer from limited performance and robustness. These approaches often over-rely on single, unreliable dynamic features, and are typically built on CNNs with limited receptive fields or Transformers with quadratic computational complexity. To address these limitations, we propose a new effective and scalable video mirror detection method, called MirrorMamba. Our approach leverages multiple cues to adapt to diverse conditions, incorporating perceived depth, correspondence and optical. We also introduce an innovative Mamba-based Multidirection Correspondence Extractor, which benefits from the global receptive field and linear complexity of the emerging Mamba spatial state model to effectively capture correspondence properties. Additionally, we design a Mamba-based layer-wise boundary enforcement decoder to resolve the unclear boundary caused by the blurred depth map. Notably, this work marks the first successful application of the Mamba-based architecture in the field of mirror detection. Extensive experiments demonstrate that our method outperforms existing state-of-the-art approaches for video mirror detection on the benchmark datasets. Furthermore, on the most challenging and representative image-based mirror detection dataset, our approach achieves state-of-the-art performance, proving its robustness and generalizability. 

---
# Sensor Calibration Model Balancing Accuracy, Real-time, and Efficiency 

**Authors**: Jinyong Yun, Hyungjin Kim, Seokho Ahn, Euijong Lee, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06715)  

**Abstract**: Most on-device sensor calibration studies benchmark models only against three macroscopic requirements (i.e., accuracy, real-time, and resource efficiency), thereby hiding deployment bottlenecks such as instantaneous error and worst-case latency. We therefore decompose this triad into eight microscopic requirements and introduce Scare (Sensor Calibration model balancing Accuracy, Real-time, and Efficiency), an ultra-compressed transformer that fulfills them all. SCARE comprises three core components: (1) Sequence Lens Projector (SLP) that logarithmically compresses time-series data while preserving boundary information across bins, (2) Efficient Bitwise Attention (EBA) module that replaces costly multiplications with bitwise operations via binary hash codes, and (3) Hash optimization strategy that ensures stable training without auxiliary loss terms. Together, these components minimize computational overhead while maintaining high accuracy and compatibility with microcontroller units (MCUs). Extensive experiments on large-scale air-quality datasets and real microcontroller deployments demonstrate that Scare outperforms existing linear, hybrid, and deep-learning baselines, making Scare, to the best of our knowledge, the first model to meet all eight microscopic requirements simultaneously. 

---
# Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture 

**Authors**: Karen Sargsyan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06701)  

**Abstract**: Sequential statistical protocols require meticulous state management and robust error handling -- challenges naturally suited to functional programming. We present a functional architecture for structural enforcement of statistical rigor in automated research systems (AI-Scientists). These LLM-driven systems risk generating spurious discoveries through dynamic hypothesis testing. We introduce the Research monad, a Haskell eDSL that enforces sequential statistical protocols (e.g., Online FDR (false discovery rate) control) using a monad transformer stack. To address risks in hybrid architectures where LLMs generate imperative code, we employ Declarative Scaffolding -- generating rigid harnesses that structurally constrain execution and prevent methodological errors like data leakage. We validate this approach through large-scale simulation (N=2000 hypotheses) and an end-to-end case study, demonstrating essential defense-in-depth for automated science integrity. 

---
# Place Matters: Comparing LLM Hallucination Rates for Place-Based Legal Queries 

**Authors**: Damian Curran, Vanessa Sporne, Lea Frermann, Jeannie Paterson  

**Link**: [PDF](https://arxiv.org/pdf/2511.06700)  

**Abstract**: How do we make a meaningful comparison of a large language model's knowledge of the law in one place compared to another? Quantifying these differences is critical to understanding if the quality of the legal information obtained by users of LLM-based chatbots varies depending on their location. However, obtaining meaningful comparative metrics is challenging because legal institutions in different places are not themselves easily comparable. In this work we propose a methodology to obtain place-to-place metrics based on the comparative law concept of functionalism. We construct a dataset of factual scenarios drawn from Reddit posts by users seeking legal advice for family, housing, employment, crime and traffic issues. We use these to elicit a summary of a law from the LLM relevant to each scenario in Los Angeles, London and Sydney. These summaries, typically of a legislative provision, are manually evaluated for hallucinations. We show that the rate of hallucination of legal information by leading closed-source LLMs is significantly associated with place. This suggests that the quality of legal solutions provided by these models is not evenly distributed across geography. Additionally, we show a strong negative correlation between hallucination rate and the frequency of the majority response when the LLM is sampled multiple times, suggesting a measure of uncertainty of model predictions of legal facts. 

---
# Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks 

**Authors**: Dian Jin, Yancheng Yuan, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06696)  

**Abstract**: Pretrained equivariant graph neural networks based on spherical harmonics offer efficient and accurate alternatives to computationally expensive ab-initio methods, yet adapting them to new tasks and chemical environments still requires fine-tuning. Conventional parameter-efficient fine-tuning (PEFT) techniques, such as Adapters and LoRA, typically break symmetry, making them incompatible with those equivariant architectures. ELoRA, recently proposed, is the first equivariant PEFT method. It achieves improved parameter efficiency and performance on many benchmarks. However, the relatively high degrees of freedom it retains within each tensor order can still perturb pretrained feature distributions and ultimately degrade performance. To address this, we present Magnitude-Modulated Equivariant Adapter (MMEA), a novel equivariant fine-tuning method which employs lightweight scalar gating to modulate feature magnitudes on a per-order and per-multiplicity basis. We demonstrate that MMEA preserves strict equivariance and, across multiple benchmarks, consistently improves energy and force predictions to state-of-the-art levels while training fewer parameters than competing approaches. These results suggest that, in many practical scenarios, modulating channel magnitudes is sufficient to adapt equivariant models to new chemical environments without breaking symmetry, pointing toward a new paradigm for equivariant PEFT design. 

---
# ML-EcoLyzer: Quantifying the Environmental Cost of Machine Learning Inference Across Frameworks and Hardware 

**Authors**: Jose Marie Antonio Minoza, Rex Gregor Laylo, Christian F Villarin, Sebastian C. Ibanez  

**Link**: [PDF](https://arxiv.org/pdf/2511.06694)  

**Abstract**: Machine learning inference occurs at a massive scale, yet its environmental impact remains poorly quantified, especially on low-resource hardware. We present ML-EcoLyzer, a cross-framework tool for measuring the carbon, energy, thermal, and water costs of inference across CPUs, consumer GPUs, and datacenter accelerators. The tool supports both classical and modern models, applying adaptive monitoring and hardware-aware evaluation.
We introduce the Environmental Sustainability Score (ESS), which quantifies the number of effective parameters served per gram of CO$_2$ emitted. Our evaluation covers over 1,900 inference configurations, spanning diverse model architectures, task modalities (text, vision, audio, tabular), hardware types, and precision levels. These rigorous and reliable measurements demonstrate that quantization enhances ESS, huge accelerators can be inefficient for lightweight applications, and even small models may incur significant costs when implemented suboptimally. ML-EcoLyzer sets a standard for sustainability-conscious model selection and offers an extensive empirical evaluation of environmental costs during inference. 

---
# Textual Self-attention Network: Test-Time Preference Optimization through Textual Gradient-based Attention 

**Authors**: Shibing Mo, Haoyang Ruan, Kai Wu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06682)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities, but aligning their outputs with human preferences typically requires expensive supervised fine-tuning. Recent test-time methods leverage textual feedback to overcome this, but they often critique and revise a single candidate response, lacking a principled mechanism to systematically analyze, weigh, and synthesize the strengths of multiple promising candidates. Such a mechanism is crucial because different responses may excel in distinct aspects (e.g., clarity, factual accuracy, or tone), and combining their best elements may produce a far superior outcome. This paper proposes the Textual Self-Attention Network (TSAN), a new paradigm for test-time preference optimization that requires no parameter updates. TSAN emulates self-attention entirely in natural language to overcome this gap: it analyzes multiple candidates by formatting them into textual keys and values, weighs their relevance using an LLM-based attention module, and synthesizes their strengths into a new, preference-aligned response under the guidance of the learned textual attention. This entire process operates in a textual gradient space, enabling iterative and interpretable optimization. Empirical evaluations demonstrate that with just three test-time iterations on a base SFT model, TSAN outperforms supervised models like Llama-3.1-70B-Instruct and surpasses the current state-of-the-art test-time alignment method by effectively leveraging multiple candidate solutions. 

---
# Rapidly Learning Soft Robot Control via Implicit Time-Stepping 

**Authors**: Andrew Choi, Dezhong Tong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06667)  

**Abstract**: With the explosive growth of rigid-body simulators, policy learning in simulation has become the de facto standard for most rigid morphologies. In contrast, soft robotic simulation frameworks remain scarce and are seldom adopted by the soft robotics community. This gap stems partly from the lack of easy-to-use, general-purpose frameworks and partly from the high computational cost of accurately simulating continuum mechanics, which often renders policy learning infeasible. In this work, we demonstrate that rapid soft robot policy learning is indeed achievable via implicit time-stepping. Our simulator of choice, DisMech, is a general-purpose, fully implicit soft-body simulator capable of handling both soft dynamics and frictional contact. We further introduce delta natural curvature control, a method analogous to delta joint position control in rigid manipulators, providing an intuitive and effective means of enacting control for soft robot learning. To highlight the benefits of implicit time-stepping and delta curvature control, we conduct extensive comparisons across four diverse soft manipulator tasks against one of the most widely used soft-body frameworks, Elastica. With implicit time-stepping, parallel stepping of 500 environments achieves up to 6x faster speeds for non-contact cases and up to 40x faster for contact-rich scenarios. Finally, a comprehensive sim-to-sim gap evaluation--training policies in one simulator and evaluating them in another--demonstrates that implicit time-stepping provides a rare free lunch: dramatic speedups achieved without sacrificing accuracy. 

---
# Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks 

**Authors**: Lingran Song, Yucheng Zhou, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06665)  

**Abstract**: Despite significant progress in pixel-level medical image analysis, existing medical image segmentation models rarely explore medical segmentation and diagnosis tasks jointly. However, it is crucial for patients that models can provide explainable diagnoses along with medical segmentation results. In this paper, we introduce a medical vision-language task named Medical Diagnosis Segmentation (MDS), which aims to understand clinical queries for medical images and generate the corresponding segmentation masks as well as diagnostic results. To facilitate this task, we first present the Multimodal Multi-disease Medical Diagnosis Segmentation (M3DS) dataset, containing diverse multimodal multi-disease medical images paired with their corresponding segmentation masks and diagnosis chain-of-thought, created via an automated diagnosis chain-of-thought generation pipeline. Moreover, we propose Sim4Seg, a novel framework that improves the performance of diagnosis segmentation by taking advantage of the Region-Aware Vision-Language Similarity to Mask (RVLS2M) module. To improve overall performance, we investigate a test-time scaling strategy for MDS tasks. Experimental results demonstrate that our method outperforms the baselines in both segmentation and diagnosis. 

---
# Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling 

**Authors**: Depanshu Sani, Mehar Khurana, Saket Anand  

**Link**: [PDF](https://arxiv.org/pdf/2511.06658)  

**Abstract**: Animal Re-ID has recently gained substantial attention in the AI research community due to its high impact on biodiversity monitoring and unique research challenges arising from environmental factors. The subtle distinguishing patterns, handling new species and the inherent open-set nature make the problem even harder. To address these complexities, foundation models trained on labeled, large-scale and multi-species animal Re-ID datasets have recently been introduced to enable zero-shot Re-ID. However, our benchmarking reveals significant gaps in their zero-shot Re-ID performance for both known and unknown species. While this highlights the need for collecting labeled data in new domains, exhaustive annotation for Re-ID is laborious and requires domain expertise. Our analyses show that existing unsupervised (USL) and AL Re-ID methods underperform for animal Re-ID. To address these limitations, we introduce a novel AL Re-ID framework that leverages complementary clustering methods to uncover and target structurally ambiguous regions in the embedding space for mining pairs of samples that are both informative and broadly representative. Oracle feedback on these pairs, in the form of must-link and cannot-link constraints, facilitates a simple annotation interface, which naturally integrates with existing USL methods through our proposed constrained clustering refinement algorithm. Through extensive experiments, we demonstrate that, by utilizing only 0.033% of all annotations, our approach consistently outperforms existing foundational, USL and AL baselines. Specifically, we report an average improvement of 10.49%, 11.19% and 3.99% (mAP) on 13 wildlife datasets over foundational, USL and AL methods, respectively, while attaining state-of-the-art performance on each dataset. Furthermore, we also show an improvement of 11.09%, 8.2% and 2.06% for unknown individuals in an open-world setting. 

---
# CaberNet: Causal Representation Learning for Cross-Domain HVAC Energy Prediction 

**Authors**: Kaiyuan Zhai, Jiacheng Cui, Zhehao Zhang, Junyu Xue, Yang Deng, Kui Wu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06634)  

**Abstract**: Cross-domain HVAC energy prediction is essential for scalable building energy management, particularly because collecting extensive labeled data for every new building is both costly and impractical. Yet, this task remains highly challenging due to the scarcity and heterogeneity of data across different buildings, climate zones, and seasonal patterns. In particular, buildings situated in distinct climatic regions introduce variability that often leads existing methods to overfit to spurious correlations, rely heavily on expert intervention, or compromise on data diversity. To address these limitations, we propose CaberNet, a causal and interpretable deep sequence model that learns invariant (Markov blanket) representations for robust cross-domain prediction. In a purely data-driven fashion and without requiring any prior knowledge, CaberNet integrates i) a global feature gate trained with a self-supervised Bernoulli regularization to distinguish superior causal features from inferior ones, and ii) a domain-wise training scheme that balances domain contributions, minimizes cross-domain loss variance, and promotes latent factor independence. We evaluate CaberNet on real-world datasets collected from three buildings located in three climatically diverse cities, and it consistently outperforms all baselines, achieving a 22.9\% reduction in normalized mean squared error (NMSE) compared to the best benchmark. Our code is available at this https URL. 

---
# Explainable Cross-Disease Reasoning for Cardiovascular Risk Assessment from LDCT 

**Authors**: Yifei Zhang, Jiashuo Zhang, Xiaofeng Yang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06625)  

**Abstract**: Low-dose chest computed tomography (LDCT) inherently captures both pulmonary and cardiac structures, offering a unique opportunity for joint assessment of lung and cardiovascular health. However, most existing approaches treat these domains as independent tasks, overlooking their physiological interplay and shared imaging biomarkers. We propose an Explainable Cross-Disease Reasoning Framework that enables interpretable cardiopulmonary risk assessment from a single LDCT scan. The framework introduces an agentic reasoning process that emulates clinical diagnostic thinking-first perceiving pulmonary findings, then reasoning through established medical knowledge, and finally deriving a cardiovascular judgment with explanatory rationale. It integrates three synergistic components: a pulmonary perception module that summarizes lung abnormalities, a knowledge-guided reasoning module that infers their cardiovascular implications, and a cardiac representation module that encodes structural biomarkers. Their outputs are fused to produce a holistic cardiovascular risk prediction that is both accurate and physiologically grounded. Experiments on the NLST cohort demonstrate that the proposed framework achieves state-of-the-art performance for CVD screening and mortality prediction, outperforming single-disease and purely image-based baselines. Beyond quantitative gains, the framework provides human-verifiable reasoning that aligns with cardiological understanding, revealing coherent links between pulmonary abnormalities and cardiac stress mechanisms. Overall, this work establishes a unified and explainable paradigm for cardiovascular analysis from LDCT, bridging the gap between image-based prediction and mechanism-based medical interpretation. 

---
# How Do VLAs Effectively Inherit from VLMs? 

**Authors**: Chuheng Zhang, Rushuai Yang, Xiaoyu Chen, Kaixin Wang, Li Zhao, Yi Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06619)  

**Abstract**: Vision-language-action (VLA) models hold the promise to attain generalizable embodied control. To achieve this, a pervasive paradigm is to leverage the rich vision-semantic priors of large vision-language models (VLMs). However, the fundamental question persists: How do VLAs effectively inherit the prior knowledge from VLMs? To address this critical question, we introduce a diagnostic benchmark, GrinningFace, an emoji tabletop manipulation task where the robot arm is asked to place objects onto printed emojis corresponding to language instructions. This task design is particularly revealing -- knowledge associated with emojis is ubiquitous in Internet-scale datasets used for VLM pre-training, yet emojis themselves are largely absent from standard robotics datasets. Consequently, they provide a clean proxy: successful task completion indicates effective transfer of VLM priors to embodied control. We implement this diagnostic task in both simulated environment and a real robot, and compare various promising techniques for knowledge transfer. Specifically, we investigate the effects of parameter-efficient fine-tuning, VLM freezing, co-training, predicting discretized actions, and predicting latent actions. Through systematic evaluation, our work not only demonstrates the critical importance of preserving VLM priors for the generalization of VLA but also establishes guidelines for future research in developing truly generalizable embodied AI systems. 

---
# Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily 

**Authors**: Asela Hevapathige, Asiri Wijesinghe, Ahad N. Zehmakan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06608)  

**Abstract**: Graph Neural Networks (GNNs) have achieved significant success in addressing node classification tasks. However, the effectiveness of traditional GNNs degrades on heterophilic graphs, where connected nodes often belong to different labels or properties. While recent work has introduced mechanisms to improve GNN performance under heterophily, certain key limitations still exist. Most existing models apply a fixed aggregation depth across all nodes, overlooking the fact that nodes may require different propagation depths based on their local homophily levels and neighborhood structures. Moreover, many methods are tailored to either homophilic or heterophilic settings, lacking the flexibility to generalize across both regimes. To address these challenges, we develop a theoretical framework that links local structural and label characteristics to information propagation dynamics at the node level. Our analysis shows that optimal aggregation depth varies across nodes and is critical for preserving class-discriminative information. Guided by this insight, we propose a novel adaptive-depth GNN architecture that dynamically selects node-specific aggregation depths using theoretically grounded metrics. Our method seamlessly adapts to both homophilic and heterophilic patterns within a unified model. Extensive experiments demonstrate that our approach consistently enhances the performance of standard GNN backbones across diverse benchmarks. 

---
# SPUR: A Plug-and-Play Framework for Integrating Spatial Audio Understanding and Reasoning into Large Audio-Language Models 

**Authors**: S Sakshi, Vaibhavi Lokegaonkar, Neil Zhang, Ramani Duraiswami, Sreyan Ghosh, Dinesh Manocha, Lie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06606)  

**Abstract**: Spatial perception is central to auditory intelligence, enabling accurate understanding of real-world acoustic scenes and advancing human-level perception of the world around us. While recent large audio-language models (LALMs) show strong reasoning over complex audios, most operate on monaural inputs and lack the ability to capture spatial cues such as direction, elevation, and distance. We introduce SPUR, a lightweight, plug-in approach that equips LALMs with spatial perception through minimal architectural changes. SPUR consists of: (i) a First-Order Ambisonics (FOA) encoder that maps (W, X, Y, Z) channels to rotation-aware, listener-centric spatial features, integrated into target LALMs via a multimodal adapter; and (ii) SPUR-Set, a spatial QA dataset combining open-source FOA recordings with controlled simulations, emphasizing relative direction, elevation, distance, and overlap for supervised spatial reasoning. Fine-tuning our model on the SPUR-Set consistently improves spatial QA and multi-speaker attribution while preserving general audio understanding. SPUR provides a simple recipe that transforms monaural LALMs into spatially aware models. Extensive ablations validate the effectiveness of our approach. 

---
# TabRAG: Tabular Document Retrieval via Structured Language Representations 

**Authors**: Jacob Si, Mike Qu, Michelle Lee, Yingzhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06582)  

**Abstract**: Ingesting data for Retrieval-Augmented Generation (RAG) involves either fine-tuning the embedding model directly on the target corpus or parsing documents for embedding model encoding. The former, while accurate, incurs high computational hardware requirements, while the latter suffers from suboptimal performance when extracting tabular data. In this work, we address the latter by presenting TabRAG, a parsing-based RAG pipeline designed to tackle table-heavy documents via structured language representations. TabRAG outperforms existing popular parsing-based methods for generation and retrieval. Code is available at this https URL. 

---
# CoFineLLM: Conformal Finetuning of LLMs for Language-Instructed Robot Planning 

**Authors**: Jun Wang, Yevgeniy Vorobeychik, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2511.06575)  

**Abstract**: Large Language Models (LLMs) have recently emerged as planners for language-instructed agents, generating sequences of actions to accomplish natural language tasks. However, their reliability remains a challenge, especially in long-horizon tasks, since they often produce overconfident yet wrong outputs. Conformal Prediction (CP) has been leveraged to address this issue by wrapping LLM outputs into prediction sets that contain the correct action with a user-defined confidence. When the prediction set is a singleton, the planner executes that action; otherwise, it requests help from a user. This has led to LLM-based planners that can ensure plan correctness with a user-defined probability. However, as LLMs are trained in an uncertainty-agnostic manner, without awareness of prediction sets, they tend to produce unnecessarily large sets, particularly at higher confidence levels, resulting in frequent human interventions limiting autonomous deployment. To address this, we introduce CoFineLLM (Conformal Finetuning for LLMs), the first CP-aware finetuning framework for LLM-based planners that explicitly reduces prediction-set size and, in turn, the need for user interventions. We evaluate our approach on multiple language-instructed robot planning problems and show consistent improvements over uncertainty-aware and uncertainty-agnostic finetuning baselines in terms of prediction-set size, and help rates. Finally, we demonstrate robustness of our method to out-of-distribution scenarios in hardware experiments. 

---
# SteganoSNN: SNN-Based Audio-in-Image Steganography with Encryption 

**Authors**: Biswajit Kumar Sahoo, Pedro Machado, Isibor Kennedy Ihianle, Andreas Oikonomou, Srinivas Boppu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06573)  

**Abstract**: Secure data hiding remains a fundamental challenge in digital communication, requiring a careful balance between computational efficiency and perceptual transparency. The balance between security and performance is increasingly fragile with the emergence of generative AI systems capable of autonomously generating and optimising sophisticated cryptanalysis and steganalysis algorithms, thereby accelerating the exposure of vulnerabilities in conventional data-hiding schemes.
This work introduces SteganoSNN, a neuromorphic steganographic framework that exploits spiking neural networks (SNNs) to achieve secure, low-power, and high-capacity multimedia data hiding. Digitised audio samples are converted into spike trains using leaky integrate-and-fire (LIF) neurons, encrypted via a modulo-based mapping scheme, and embedded into the least significant bits of RGBA image channels using a dithering mechanism to minimise perceptual distortion. Implemented in Python using NEST and realised on a PYNQ-Z2 FPGA, SteganoSNN attains real-time operation with an embedding capacity of 8 bits per pixel. Experimental evaluations on the DIV2K 2017 dataset demonstrate image fidelity between 40.4 dB and 41.35 dB in PSNR and SSIM values consistently above 0.97, surpassing SteganoGAN in computational efficiency and robustness. SteganoSNN establishes a foundation for neuromorphic steganography, enabling secure, energy-efficient communication for Edge-AI, IoT, and biomedical applications. 

---
# Rep2Text: Decoding Full Text from a Single LLM Token Representation 

**Authors**: Haiyan Zhao, Zirui He, Fan Yang, Ali Payani, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06571)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress across diverse tasks, yet their internal mechanisms remain largely opaque. In this work, we address a fundamental question: to what extent can the original input text be recovered from a single last-token representation within an LLM? We propose Rep2Text, a novel framework for decoding full text from last-token representations. Rep2Text employs a trainable adapter that projects a target model's internal representations into the embedding space of a decoding language model, which then autoregressively reconstructs the input text. Experiments on various model combinations (Llama-3.1-8B, Gemma-7B, Mistral-7B-v0.1, Llama-3.2-3B) demonstrate that, on average, over half of the information in 16-token sequences can be recovered from this compressed representation while maintaining strong semantic integrity and coherence. Furthermore, our analysis reveals an information bottleneck effect: longer sequences exhibit decreased token-level recovery while preserving strong semantic integrity. Besides, our framework also demonstrates robust generalization to out-of-distribution medical data. 

---
# Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity 

**Authors**: João Mattos, Debolina Halder Lina, Arlei Silva  

**Link**: [PDF](https://arxiv.org/pdf/2511.06568)  

**Abstract**: Link prediction is a fundamental task in graph machine learning with applications, ranging from social recommendation to knowledge graph completion. Fairness in this setting is critical, as biased predictions can exacerbate societal inequalities. Prior work adopts a dyadic definition of fairness, enforcing fairness through demographic parity between intra-group and inter-group link predictions. However, we show that this dyadic framing can obscure underlying disparities across subgroups, allowing systemic biases to go undetected. Moreover, we argue that demographic parity does not meet desired properties for fairness assessment in ranking-based tasks such as link prediction. We formalize the limitations of existing fairness evaluations and propose a framework that enables a more expressive assessment. Additionally, we propose a lightweight post-processing method combined with decoupled link predictors that effectively mitigates bias and achieves state-of-the-art fairness-utility trade-offs. 

---
# LLM For Loop Invariant Generation and Fixing: How Far Are We? 

**Authors**: Mostafijur Rahman Akhond, Saikat Chakraborty, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06552)  

**Abstract**: A loop invariant is a property of a loop that remains true before and after each execution of the loop. The identification of loop invariants is a critical step to support automated program safety assessment. Recent advancements in Large Language Models (LLMs) have demonstrated potential in diverse software engineering (SE) and formal verification tasks. However, we are not aware of the performance of LLMs to infer loop invariants. We report an empirical study of both open-source and closed-source LLMs of varying sizes to assess their proficiency in inferring inductive loop invariants for programs and in fixing incorrect invariants. Our findings reveal that while LLMs exhibit some utility in inferring and repairing loop invariants, their performance is substantially enhanced when supplemented with auxiliary information such as domain knowledge and illustrative examples. LLMs achieve a maximum success rate of 78\% in generating, but are limited to 16\% in repairing the invariant. 

---
# Ibom NLP: A Step Toward Inclusive Natural Language Processing for Nigeria's Minority Languages 

**Authors**: Oluwadara Kalejaiye, Luel Hagos Beyene, David Ifeoluwa Adelani, Mmekut-Mfon Gabriel Edet, Aniefon Daniel Akpan, Eno-Abasi Urua, Anietie Andy  

**Link**: [PDF](https://arxiv.org/pdf/2511.06531)  

**Abstract**: Nigeria is the most populous country in Africa with a population of more than 200 million people. More than 500 languages are spoken in Nigeria and it is one of the most linguistically diverse countries in the world. Despite this, natural language processing (NLP) research has mostly focused on the following four languages: Hausa, Igbo, Nigerian-Pidgin, and Yoruba (i.e <1% of the languages spoken in Nigeria). This is in part due to the unavailability of textual data in these languages to train and apply NLP algorithms. In this work, we introduce ibom -- a dataset for machine translation and topic classification in four Coastal Nigerian languages from the Akwa Ibom State region: Anaang, Efik, Ibibio, and Oro. These languages are not represented in Google Translate or in major benchmarks such as Flores-200 or SIB-200. We focus on extending Flores-200 benchmark to these languages, and further align the translated texts with topic labels based on SIB-200 classification dataset. Our evaluation shows that current LLMs perform poorly on machine translation for these languages in both zero-and-few shot settings. However, we find the few-shot samples to steadily improve topic classification with more shots. 

---
# TriShGAN: Enhancing Sparsity and Robustness in Multivariate Time Series Counterfactuals Explanation 

**Authors**: Hongnan Ma, Yiwei Shi, Guanxiong Sun, Mengyue Yang, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06529)  

**Abstract**: In decision-making processes, stakeholders often rely on counterfactual explanations, which provide suggestions about what should be changed in the queried instance to alter the outcome of an AI system. However, generating these explanations for multivariate time series presents challenges due to their complex, multi-dimensional nature. Traditional Nearest Unlike Neighbor-based methods typically substitute subsequences in a queried time series with influential subsequences from an NUN, which is not always realistic in real-world scenarios due to the rigid direct substitution. Counterfactual with Residual Generative Adversarial Networks-based methods aim to address this by learning from the distribution of observed data to generate synthetic counterfactual explanations. However, these methods primarily focus on minimizing the cost from the queried time series to the counterfactual explanations and often neglect the importance of distancing the counterfactual explanation from the decision boundary. This oversight can result in explanations that no longer qualify as counterfactual if minor changes occur within the model. To generate a more robust counterfactual explanation, we introduce TriShGAN, under the CounteRGAN framework enhanced by the incorporation of triplet loss. This unsupervised learning approach uses distance metric learning to encourage the counterfactual explanations not only to remain close to the queried time series but also to capture the feature distribution of the instance with the desired outcome, thereby achieving a better balance between minimal cost and robustness. Additionally, we integrate a Shapelet Extractor that strategically selects the most discriminative parts of the high-dimensional queried time series to enhance the sparsity of counterfactual explanation and efficiency of the training process. 

---
# On the Analogy between Human Brain and LLMs: Spotting Key Neurons in Grammar Perception 

**Authors**: Sanaz Saki Norouzi, Mohammad Masjedi, Pascal Hitzler  

**Link**: [PDF](https://arxiv.org/pdf/2511.06519)  

**Abstract**: Artificial Neural Networks, the building blocks of AI, were inspired by the human brain's network of neurons. Over the years, these networks have evolved to replicate the complex capabilities of the brain, allowing them to handle tasks such as image and language processing. In the realm of Large Language Models, there has been a keen interest in making the language learning process more akin to that of humans. While neuroscientific research has shown that different grammatical categories are processed by different neurons in the brain, we show that LLMs operate in a similar way. Utilizing Llama 3, we identify the most important neurons associated with the prediction of words belonging to different part-of-speech tags. Using the achieved knowledge, we train a classifier on a dataset, which shows that the activation patterns of these key neurons can reliably predict part-of-speech tags on fresh data. The results suggest the presence of a subspace in LLMs focused on capturing part-of-speech tag concepts, resembling patterns observed in lesion studies of the brain in neuroscience. 

---
# Rethinking what Matters: Effective and Robust Multilingual Realignment for Low-Resource Languages 

**Authors**: Quang Phuoc Nguyen, David Anugraha, Felix Gaschi, Jun Bin Cheng, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.06497)  

**Abstract**: Realignment is a promising strategy to improve cross-lingual transfer in multilingual language models. However, empirical results are mixed and often unreliable, particularly for typologically distant or low-resource languages (LRLs) compared to English. Moreover, word realignment tools often rely on high-quality parallel data, which can be scarce or noisy for many LRLs. In this work, we conduct an extensive empirical study to investigate whether realignment truly benefits from using all available languages, or if strategically selected subsets can offer comparable or even improved cross-lingual transfer, and study the impact on LRLs. Our controlled experiments show that realignment can be particularly effective for LRLs and that using carefully selected, linguistically diverse subsets can match full multilingual alignment, and even outperform it for unseen LRLs. This indicates that effective realignment does not require exhaustive language coverage and can reduce data collection overhead, while remaining both efficient and robust when guided by informed language selection. 

---
# A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving 

**Authors**: Keke Long, Jiacheng Guo, Tianyun Zhang, Hongkai Yu, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06496)  

**Abstract**: Vision Language Models (VLMs) are increasingly used in autonomous driving to help understand traffic scenes, but they sometimes produce hallucinations, which are false details not grounded in the visual input. Detecting and mitigating hallucinations is challenging when ground-truth references are unavailable and model internals are inaccessible. This paper proposes a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access. By constructing a sentence-embedding matrix and decomposing it into a low-rank consensus component and a sparse residual, we use the residual magnitude to rank captions: selecting the one with the smallest residual as the most hallucination-free. Experiments on the NuScenes dataset demonstrate that our approach achieves 87% selection accuracy in identifying hallucination-free captions, representing a 19% improvement over the unfiltered baseline and a 6-10% improvement over multi-agent debate method. The sorting produced by sparse error magnitudes shows strong correlation with human judgments of hallucinations, validating our scoring mechanism. Additionally, our method, which can be easily parallelized, reduces inference time by 51-67% compared to debate approaches, making it practical for real-time autonomous driving applications. 

---
# Route Experts by Sequence, not by Token 

**Authors**: Tiansheng Wen, Yifei Wang, Aosong Feng, Long Ma, Xinyang Liu, Yifan Wang, Lixuan Guo, Bo Chen, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2511.06494)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale large language models (LLMs) by activating only a subset of experts per token, but the standard TopK routing assigns the same fixed number of experts to all tokens, ignoring their varying complexity. Prior adaptive routing methods introduce additional modules and hyperparameters, often requiring costly retraining from scratch. We propose Sequence-level TopK (SeqTopK), a minimal modification that shifts the expert budget from the token level to the sequence level. By selecting the top $T \cdot K$ experts across all $T$ tokens, SeqTopK enables end-to-end learned dynamic allocation -- assigning more experts to difficult tokens and fewer to easy ones -- while preserving the same overall budget. SeqTopK requires only a few lines of code, adds less than 1% overhead, and remains fully compatible with pretrained MoE models. Experiments across math, coding, law, and writing show consistent improvements over TopK and prior parameter-free adaptive methods, with gains that become substantially larger under higher sparsity (up to 16.9%). These results highlight SeqTopK as a simple, efficient, and scalable routing strategy, particularly well-suited for the extreme sparsity regimes of next-generation LLMs. Code is available at this https URL. 

---
# Explainable AI For Early Detection Of Sepsis 

**Authors**: Atharva Thakur, Shruti Dhumal  

**Link**: [PDF](https://arxiv.org/pdf/2511.06492)  

**Abstract**: Sepsis is a life-threatening condition that requires rapid detection and treatment to prevent progression to severe sepsis, septic shock, or multi-organ failure. Despite advances in medical technology, it remains a major challenge for clinicians. While recent machine learning models have shown promise in predicting sepsis onset, their black-box nature limits interpretability and clinical trust. In this study, we present an interpretable AI approach for sepsis analysis that integrates machine learning with clinical knowledge. Our method not only delivers accurate predictions of sepsis onset but also enables clinicians to understand, validate, and align model outputs with established medical expertise. 

---
# Zooming into Comics: Region-Aware RL Improves Fine-Grained Comic Understanding in Vision-Language Models 

**Authors**: Yule Chen, Yufan Ren, Sabine Süsstrunk  

**Link**: [PDF](https://arxiv.org/pdf/2511.06490)  

**Abstract**: Complex visual narratives, such as comics, present a significant challenge to Vision-Language Models (VLMs). Despite excelling on natural images, VLMs often struggle with stylized line art, onomatopoeia, and densely packed multi-panel layouts. To address this gap, we introduce AI4VA-FG, the first fine-grained and comprehensive benchmark for VLM-based comic understanding. It spans tasks from foundational recognition and detection to high-level character reasoning and narrative construction, supported by dense annotations for characters, poses, and depth. Beyond that, we evaluate state-of-the-art proprietary models, including GPT-4o and Gemini-2.5, and open-source models such as Qwen2.5-VL, revealing substantial performance deficits across core tasks of our benchmarks and underscoring that comic understanding remains an unsolved challenge. To enhance VLMs' capabilities in this domain, we systematically investigate post-training strategies, including supervised fine-tuning on solutions (SFT-S), supervised fine-tuning on reasoning trajectories (SFT-R), and reinforcement learning (RL). Beyond that, inspired by the emerging "Thinking with Images" paradigm, we propose Region-Aware Reinforcement Learning (RARL) for VLMs, which trains models to dynamically attend to relevant regions through zoom-in operations. We observe that when applied to the Qwen2.5-VL model, RL and RARL yield significant gains in low-level entity recognition and high-level storyline ordering, paving the way for more accurate and efficient VLM applications in the comics domain. 

---
# EchoMark: Perceptual Acoustic Environment Transfer with Watermark-Embedded Room Impulse Response 

**Authors**: Chenpei Huang, Lingfeng Yao, Kyu In Lee, Lan Emily Zhang, Xun Chen, Miao Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06458)  

**Abstract**: Acoustic Environment Matching (AEM) is the task of transferring clean audio into a target acoustic environment, enabling engaging applications such as audio dubbing and auditory immersive virtual reality (VR). Recovering similar room impulse response (RIR) directly from reverberant speech offers more accessible and flexible AEM solution. However, this capability also introduces vulnerabilities of arbitrary ``relocation" if misused by malicious user, such as facilitating advanced voice spoofing attacks or undermining the authenticity of recorded evidence. To address this issue, we propose EchoMark, the first deep learning-based AEM framework that generates perceptually similar RIRs with embedded watermark. Our design tackle the challenges posed by variable RIR characteristics, such as different durations and energy decays, by operating in the latent domain. By jointly optimizing the model with a perceptual loss for RIR reconstruction and a loss for watermark detection, EchoMark achieves both high-quality environment transfer and reliable watermark recovery. Experiments on diverse datasets validate that EchoMark achieves room acoustic parameter matching performance comparable to FiNS, the state-of-the-art RIR estimator. Furthermore, a high Mean Opinion Score (MOS) of 4.22 out of 5, watermark detection accuracy exceeding 99\%, and bit error rates (BER) below 0.3\% collectively demonstrate the effectiveness of EchoMark in preserving perceptual quality while ensuring reliable watermark embedding. 

---
# A Multi-Agent System for Semantic Mapping of Relational Data to Knowledge Graphs 

**Authors**: Milena Trajanoska, Riste Stojanov, Dimitar Trajanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.06455)  

**Abstract**: Enterprises often maintain multiple databases for storing critical business data in siloed systems, resulting in inefficiencies and challenges with data interoperability. A key to overcoming these challenges lies in integrating disparate data sources, enabling businesses to unlock the full potential of their data. Our work presents a novel approach for integrating multiple databases using knowledge graphs, focusing on the application of large language models as semantic agents for mapping and connecting structured data across systems by leveraging existing vocabularies. The proposed methodology introduces a semantic layer above tables in relational databases, utilizing a system comprising multiple LLM agents that map tables and columns to this http URL terms. Our approach achieves a mapping accuracy of over 90% in multiple domains. 

---
# FLEX: Continuous Agent Evolution via Forward Learning from Experience 

**Authors**: Zhicheng Cai, Xinyuan Guo, Yu Pei, JiangTao Feng, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06449)  

**Abstract**: Autonomous agents driven by Large Language Models (LLMs) have revolutionized reasoning and problem-solving but remain static after training, unable to grow with experience as intelligent beings do during deployment. We introduce Forward Learning with EXperience (FLEX), a gradient-free learning paradigm that enables LLM agents to continuously evolve through accumulated experience. Specifically, FLEX cultivates scalable and inheritable evolution by constructing a structured experience library through continual reflection on successes and failures during interaction with the environment. FLEX delivers substantial improvements on mathematical reasoning, chemical retrosynthesis, and protein fitness prediction (up to 23% on AIME25, 10% on USPTO50k, and 14% on ProteinGym). We further identify a clear scaling law of experiential growth and the phenomenon of experience inheritance across agents, marking a step toward scalable and inheritable continuous agent evolution. Project Page: this https URL. 

---
# When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms 

**Authors**: Qibing Ren, Zhijie Zheng, Jiaxuan Guo, Junchi Yan, Lizhuang Ma, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06448)  

**Abstract**: In this work, we study the risks of collective financial fraud in large-scale multi-agent systems powered by large language model (LLM) agents. We investigate whether agents can collaborate in fraudulent behaviors, how such collaboration amplifies risks, and what factors influence fraud success. To support this research, we present MultiAgentFraudBench, a large-scale benchmark for simulating financial fraud scenarios based on realistic online interactions. The benchmark covers 28 typical online fraud scenarios, spanning the full fraud lifecycle across both public and private domains. We further analyze key factors affecting fraud success, including interaction depth, activity level, and fine-grained collaboration failure modes. Finally, we propose a series of mitigation strategies, including adding content-level warnings to fraudulent posts and dialogues, using LLMs as monitors to block potentially malicious agents, and fostering group resilience through information sharing at the societal level. Notably, we observe that malicious agents can adapt to environmental interventions. Our findings highlight the real-world risks of multi-agent financial fraud and suggest practical measures for mitigating them. Code is available at this https URL. 

---
# Personality over Precision: Exploring the Influence of Human-Likeness on ChatGPT Use for Search 

**Authors**: Mert Yazan, Frederik Bungaran Ishak Situmeang, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2511.06447)  

**Abstract**: Conversational search interfaces, like ChatGPT, offer an interactive, personalized, and engaging user experience compared to traditional search. On the downside, they are prone to cause overtrust issues where users rely on their responses even when they are incorrect. What aspects of the conversational interaction paradigm drive people to adopt it, and how it creates personalized experiences that lead to overtrust, is not clear. To understand the factors influencing the adoption of conversational interfaces, we conducted a survey with 173 participants. We examined user perceptions regarding trust, human-likeness (anthropomorphism), and design preferences between ChatGPT and Google. To better understand the overtrust phenomenon, we asked users about their willingness to trade off factuality for constructs like ease of use or human-likeness. Our analysis identified two distinct user groups: those who use both ChatGPT and Google daily (DUB), and those who primarily rely on Google (DUG). The DUB group exhibited higher trust in ChatGPT, perceiving it as more human-like, and expressed greater willingness to trade factual accuracy for enhanced personalization and conversational flow. Conversely, the DUG group showed lower trust toward ChatGPT but still appreciated aspects like ad-free experiences and responsive interactions. Demographic analysis further revealed nuanced patterns, with middle-aged adults using ChatGPT less frequently yet trusting it more, suggesting potential vulnerability to misinformation. Our findings contribute to understanding user segmentation, emphasizing the critical roles of personalization and human-likeness in conversational IR systems, and reveal important implications regarding users' willingness to compromise factual accuracy for more engaging interactions. 

---
# SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention 

**Authors**: Bohan Yu, Wei Huang, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06446)  

**Abstract**: This paper proposes SR-KI, a novel approach for integrating real-time and large-scale structured knowledge bases (KBs) into large language models (LLMs). SR-KI begins by encoding KBs into key-value pairs using a pretrained encoder, and injects them into LLMs' KV cache. Building on this representation, we employ a two-stage training paradigm: first locating a dedicated retrieval layer within the LLM, and then applying an attention-based loss at this layer to explicitly supervise attention toward relevant KB entries. Unlike traditional retrieval-augmented generation methods that rely heavily on the performance of external retrievers and multi-stage pipelines, SR-KI supports end-to-end inference by performing retrieval entirely within the models latent space. This design enables efficient compression of injected knowledge and facilitates dynamic knowledge updates. Comprehensive experiments demonstrate that SR-KI enables the integration of up to 40K KBs into a 7B LLM on a single A100 40GB GPU, and achieves strong retrieval performance, maintaining over 98% Recall@10 on the best-performing task and exceeding 88% on average across all tasks. Task performance on question answering and KB ID generation also demonstrates that SR-KI maintains strong performance while achieving up to 99.75% compression of the injected KBs. 

---
# Walking the Tightrope of LLMs for Software Development: A Practitioners' Perspective 

**Authors**: Samuel Ferino, Rashina Hoda, John Grundy, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2511.06428)  

**Abstract**: Background: Large Language Models emerged with the potential of provoking a revolution in software development (e.g., automating processes, workforce transformation). Although studies have started to investigate the perceived impact of LLMs for software development, there is a need for empirical studies to comprehend how to balance forward and backward effects of using LLMs. Objective: We investigated how LLMs impact software development and how to manage the impact from a software developer's perspective. Method: We conducted 22 interviews with software practitioners across 3 rounds of data collection and analysis, between October (2024) and September (2025). We employed socio-technical grounded theory (STGT) for data analysis to rigorously analyse interview participants' responses. Results: We identified the benefits (e.g., maintain software development flow, improve developers' mental model, and foster entrepreneurship) and disadvantages (e.g., negative impact on developers' personality and damage to developers' reputation) of using LLMs at individual, team, organisation, and society levels; as well as best practices on how to adopt LLMs. Conclusion: Critically, we present the trade-offs that software practitioners, teams, and organisations face in working with LLMs. Our findings are particularly useful for software team leaders and IT managers to assess the viability of LLMs within their specific context. 

---
# Turbo-DDCM: Fast and Flexible Zero-Shot Diffusion-Based Image Compression 

**Authors**: Amit Vaisman, Guy Ohayon, Hila Manor, Michael Elad, Tomer Michaeli  

**Link**: [PDF](https://arxiv.org/pdf/2511.06424)  

**Abstract**: While zero-shot diffusion-based compression methods have seen significant progress in recent years, they remain notoriously slow and computationally demanding. This paper presents an efficient zero-shot diffusion-based compression method that runs substantially faster than existing methods, while maintaining performance that is on par with the state-of-the-art techniques. Our method builds upon the recently proposed Denoising Diffusion Codebook Models (DDCMs) compression scheme. Specifically, DDCM compresses an image by sequentially choosing the diffusion noise vectors from reproducible random codebooks, guiding the denoiser's output to reconstruct the target image. We modify this framework with Turbo-DDCM, which efficiently combines a large number of noise vectors at each denoising step, thereby significantly reducing the number of required denoising operations. This modification is also coupled with an improved encoding protocol. Furthermore, we introduce two flexible variants of Turbo-DDCM, a priority-aware variant that prioritizes user-specified regions and a distortion-controlled variant that compresses an image based on a target PSNR rather than a target BPP. Comprehensive experiments position Turbo-DDCM as a compelling, practical, and flexible image compression scheme. 

---
# On Modality Incomplete Infrared-Visible Object Detection: An Architecture Compatibility Perspective 

**Authors**: Shuo Yang, Yinghui Xing, Shizhou Zhang, Zhilong Niu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06406)  

**Abstract**: Infrared and visible object detection (IVOD) is essential for numerous around-the-clock applications. Despite notable advancements, current IVOD models exhibit notable performance declines when confronted with incomplete modality data, particularly if the dominant modality is missing. In this paper, we take a thorough investigation on modality incomplete IVOD problem from an architecture compatibility perspective. Specifically, we propose a plug-and-play Scarf Neck module for DETR variants, which introduces a modality-agnostic deformable attention mechanism to enable the IVOD detector to flexibly adapt to any single or double modalities during training and inference. When training Scarf-DETR, we design a pseudo modality dropout strategy to fully utilize the multi-modality information, making the detector compatible and robust to both working modes of single and double modalities. Moreover, we introduce a comprehensive benchmark for the modality-incomplete IVOD task aimed at thoroughly assessing situations where the absent modality is either dominant or secondary. Our proposed Scarf-DETR not only performs excellently in missing modality scenarios but also achieves superior performances on the standard IVOD modality complete benchmarks. Our code will be available at this https URL. 

---
# HatePrototypes: Interpretable and Transferable Representations for Implicit and Explicit Hate Speech Detection 

**Authors**: Irina Proskurina, Marc-Antoine Carpentier, Julien Velcin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06391)  

**Abstract**: Optimization of offensive content moderation models for different types of hateful messages is typically achieved through continued pre-training or fine-tuning on new hate speech benchmarks. However, existing benchmarks mainly address explicit hate toward protected groups and often overlook implicit or indirect hate, such as demeaning comparisons, calls for exclusion or violence, and subtle discriminatory language that still causes harm. While explicit hate can often be captured through surface features, implicit hate requires deeper, full-model semantic processing. In this work, we question the need for repeated fine-tuning and analyze the role of HatePrototypes, class-level vector representations derived from language models optimized for hate speech detection and safety moderation. We find that these prototypes, built from as few as 50 examples per class, enable cross-task transfer between explicit and implicit hate, with interchangeable prototypes across benchmarks. Moreover, we show that parameter-free early exiting with prototypes is effective for both hate types. We release the code, prototype resources, and evaluation scripts to support future research on efficient and transferable hate speech detection. 

---
# Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint 

**Authors**: Suqing Wang, Ziyang Ma, Xinyi Li, Zuchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06390)  

**Abstract**: Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models. 

---
# HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation 

**Authors**: Kunrong Li, Zhu Sun, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06388)  

**Abstract**: We propose HyMoERec, a novel sequential recommendation framework that addresses the limitations of uniform Position-wise Feed-Forward Networks in existing models. Current approaches treat all user interactions and items equally, overlooking the heterogeneity in user behavior patterns and diversity in item complexity. HyMoERec initially introduces a hybrid mixture-of-experts architecture that combines shared and specialized expert branches with an adaptive expert fusion mechanism for the sequential recommendation task. This design captures diverse reasoning for varied users and items while ensuring stable training. Experiments on MovieLens-1M and Beauty datasets demonstrate that HyMoERec consistently outperforms state-of-the-art baselines. 

---
# Privacy-Preserving Federated Learning for Fair and Efficient Urban Traffic Optimization 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06363)  

**Abstract**: The optimization of urban traffic is threatened by the complexity of achieving a balance between transport efficiency and the maintenance of privacy, as well as the equitable distribution of traffic based on socioeconomically diverse neighborhoods. Current centralized traffic management schemes invade user location privacy and further entrench traffic disparity by offering disadvantaged route suggestions, whereas current federated learning frameworks do not consider fairness constraints in multi-objective traffic settings. This study presents a privacy-preserving federated learning framework, termed FedFair-Traffic, that jointly and simultaneously optimizes travel efficiency, traffic fairness, and differential privacy protection. This is the first attempt to integrate three conflicting objectives to improve urban transportation systems. The proposed methodology enables collaborative learning between related vehicles with data locality by integrating Graph Neural Networks with differential privacy mechanisms ($\epsilon$-privacy guarantees) and Gini coefficient-based fair constraints using multi-objective optimization. The framework uses federated aggregation methods of gradient clipping and noise injection to provide differential privacy and optimize Pareto-efficient solutions for the efficiency-fairness tradeoff. Real-world comprehensive experiments on the METR-LA traffic dataset showed that FedFair-Traffic can reduce the average travel time by 7\% (14.2 minutes) compared with their centralized baselines, promote traffic fairness by 73\% (Gini coefficient, 0.78), and offer high privacy protection (privacy score, 0.8) with an 89\% reduction in communication overhead. These outcomes demonstrate that FedFair-Traffic is a scalable privacy-aware smart city infrastructure with possible use-cases in metropolitan traffic flow control and federated transportation networks. 

---
# Understanding Student Interaction with AI-Powered Next-Step Hints: Strategies and Challenges 

**Authors**: Anastasiia Birillo, Aleksei Rostovskii, Yaroslav Golubev, Hieke Keuning  

**Link**: [PDF](https://arxiv.org/pdf/2511.06362)  

**Abstract**: Automated feedback generation plays a crucial role in enhancing personalized learning experiences in computer science education. Among different types of feedback, next-step hint feedback is particularly important, as it provides students with actionable steps to progress towards solving programming tasks. This study investigates how students interact with an AI-driven next-step hint system in an in-IDE learning environment. We gathered and analyzed a dataset from 34 students solving Kotlin tasks, containing detailed hint interaction logs. We applied process mining techniques and identified 16 common interaction scenarios. Semi-structured interviews with 6 students revealed strategies for managing unhelpful hints, such as adapting partial hints or modifying code to generate variations of the same hint. These findings, combined with our publicly available dataset, offer valuable opportunities for future research and provide key insights into student behavior, helping improve hint design for enhanced learning support. 

---
# A Graph-Theoretical Perspective on Law Design for Multiagent Systems 

**Authors**: Qi Shi, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2511.06361)  

**Abstract**: A law in a multiagent system is a set of constraints imposed on agents' behaviours to avoid undesirable outcomes. The paper considers two types of laws: useful laws that, if followed, completely eliminate the undesirable outcomes and gap-free laws that guarantee that at least one agent can be held responsible each time an undesirable outcome occurs. In both cases, we study the problem of finding a law that achieves the desired result by imposing the minimum restrictions.
We prove that, for both types of laws, the minimisation problem is NP-hard even in the simple case of one-shot concurrent interactions. We also show that the approximation algorithm for the vertex cover problem in hypergraphs could be used to efficiently approximate the minimum laws in both cases. 

---
# Reaction Prediction via Interaction Modeling of Symmetric Difference Shingle Sets 

**Authors**: Runhan Shi, Letian Chen, Gufeng Yu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06356)  

**Abstract**: Chemical reaction prediction remains a fundamental challenge in organic chemistry, where existing machine learning models face two critical limitations: sensitivity to input permutations (molecule/atom orderings) and inadequate modeling of substructural interactions governing reactivity. These shortcomings lead to inconsistent predictions and poor generalization to real-world scenarios. To address these challenges, we propose ReaDISH, a novel reaction prediction model that learns permutation-invariant representations while incorporating interaction-aware features. It introduces two innovations: (1) symmetric difference shingle encoding, which computes molecular shingle differences to capture reaction-specific structural changes while eliminating order sensitivity; and (2) geometry-structure interaction attention, a mechanism that models intra- and inter-molecular interactions at the shingle level. Extensive experiments demonstrate that ReaDISH improves reaction prediction performance across diverse benchmarks. It shows enhanced robustness with an average improvement of 8.76% on R$^2$ under permutation perturbations. 

---
# GazeVLM: A Vision-Language Model for Multi-Task Gaze Understanding 

**Authors**: Athul M. Mathew, Haithem Hermassi, Thariq Khalid, Arshad Ali Khan, Riad Souissi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06348)  

**Abstract**: Gaze understanding unifies the detection of people, their gaze targets, and objects of interest into a single framework, offering critical insight into visual attention and intent estimation. Although prior research has modelled gaze cues in visual scenes, a unified system is still needed for gaze understanding using both visual and language prompts. This paper introduces GazeVLM, a novel Vision-Language Model (VLM) for multi-task gaze understanding in images, addressing person detection, gaze target detection, and gaze object identification. While other transformer-based methods exist for gaze analysis, GazeVLM represents, to our knowledge, the first application of a VLM to these combined tasks, allowing for selective execution of each task. Through the integration of visual (RGB and depth) and textual modalities, our ablation study on visual input combinations revealed that a fusion of RGB images with HHA-encoded depth maps, guided by text prompts, yields superior performance. We also introduce an object-level gaze detection metric for gaze object identification ($AP_{ob}$). Through experiments, GazeVLM demonstrates significant improvements, notably achieving state-of-the-art evaluation scores on GazeFollow and VideoAttentionTarget datasets. 

---
# PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization 

**Authors**: Kelun Lei, Hailong Yang, Huaitao Zhang, Xin You, Kaige Zhang, Zhongzhi Luan, Yi Liu, Depei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06345)  

**Abstract**: Designing high-performance kernels requires expert-level tuning and a deep understanding of hardware characteristics. Recent advances in large language models (LLMs) have enabled automated kernel generation, yet most existing systems rely solely on correctness or execution time feedback, lacking the ability to reason about low-level performance bottlenecks. In this paper, we introduce PRAGMA, a profile-guided AI kernel generation framework that integrates execution feedback and fine-grained hardware profiling into the reasoning loop. PRAGMA enables LLMs to identify performance bottlenecks, preserve historical best versions, and iteratively refine code quality. We evaluate PRAGMA on KernelBench, covering GPU and CPU backends. Results show that PRAGMA consistently outperforms baseline AIKG without profiling enabled and achieves 2.81$\times$ and 2.30$\times$ averaged speedups against Torch on CPU and GPU platforms, respectively. 

---
# TimeSense:Making Large Language Models Proficient in Time-Series Analysis 

**Authors**: Zhirui Zhang, Changhua Pei, Tianyi Gao, Zhe Xie, Yibo Hao, Zhaoyang Yu, Longlong Xu, Tong Xiao, Jing Han, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2511.06344)  

**Abstract**: In the time-series domain, an increasing number of works combine text with temporal data to leverage the reasoning capabilities of large language models (LLMs) for various downstream time-series understanding tasks. This enables a single model to flexibly perform tasks that previously required specialized models for each domain. However, these methods typically rely on text labels for supervision during training, biasing the model toward textual cues while potentially neglecting the full temporal features. Such a bias can lead to outputs that contradict the underlying time-series context. To address this issue, we construct the EvalTS benchmark, comprising 10 tasks across three difficulty levels, from fundamental temporal pattern recognition to complex real-world reasoning, to evaluate models under more challenging and realistic scenarios. We also propose TimeSense, a multimodal framework that makes LLMs proficient in time-series analysis by balancing textual reasoning with a preserved temporal sense. TimeSense incorporates a Temporal Sense module that reconstructs the input time-series within the model's context, ensuring that textual reasoning is grounded in the time-series dynamics. Moreover, to enhance spatial understanding of time-series data, we explicitly incorporate coordinate-based positional embeddings, which provide each time point with spatial context and enable the model to capture structural dependencies more effectively. Experimental results demonstrate that TimeSense achieves state-of-the-art performance across multiple tasks, and it particularly outperforms existing methods on complex multi-dimensional time-series reasoning tasks. 

---
# CINEMAE: Leveraging Frozen Masked Autoencoders for Cross-Generator AI Image Detection 

**Authors**: Minsuk Jang, Hyeonseo Jeong, Minseok Son, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06325)  

**Abstract**: While context-based detectors have achieved strong generalization for AI-generated text by measuring distributional inconsistencies, image-based detectors still struggle with overfitting to generator-specific artifacts. We introduce CINEMAE, a novel paradigm for AIGC image detection that adapts the core principles of text detection methods to the visual domain. Our key insight is that Masked AutoEncoder (MAE), trained to reconstruct masked patches conditioned on visible context, naturally encodes semantic consistency expectations. We formalize this reconstruction process probabilistically, computing conditional Negative Log-Likelihood (NLL, p(masked | visible)) to quantify local semantic anomalies. By aggregating these patch-level statistics with global MAE features through learned fusion, CINEMAE achieves strong cross-generator generalization. Trained exclusively on Stable Diffusion v1.4, our method achieves over 95% accuracy on all eight unseen generators in the GenImage benchmark, substantially outperforming state-of-the-art detectors. This demonstrates that context-conditional reconstruction uncertainty provides a robust, transferable signal for AIGC detection. 

---
# Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree for Efficient NPU Integration 

**Authors**: Stef Cuyckens, Xiaoling Yi, Robin Geens, Joren Dumoulin, Martin Wiesner, Chao Fang, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2511.06313)  

**Abstract**: Emerging continual learning applications necessitate next-generation neural processing unit (NPU) platforms to support both training and inference operations. The promising Microscaling (MX) standard enables narrow bit-widths for inference and large dynamic ranges for training. However, existing MX multiply-accumulate (MAC) designs face a critical trade-off: integer accumulation requires expensive conversions from narrow floating-point products, while FP32 accumulation suffers from quantization losses and costly normalization. To address these limitations, we propose a hybrid precision-scalable reduction tree for MX MACs that combines the benefits of both approaches, enabling efficient mixed-precision accumulation with controlled accuracy relaxation. Moreover, we integrate an 8x8 array of these MACs into the state-of-the-art (SotA) NPU integration platform, SNAX, to provide efficient control and data transfer to our optimized precision-scalable MX datapath. We evaluate our design both on MAC and system level and compare it to the SotA. Our integrated system achieves an energy efficiency of 657, 1438-1675, and 4065 GOPS/W, respectively, for MXINT8, MXFP8/6, and MXFP4, with a throughput of 64, 256, and 512 GOPS. 

---
# Kaggle Chronicles: 15 Years of Competitions, Community and Data Science Innovation 

**Authors**: Kevin Bönisch, Leandro Losaria  

**Link**: [PDF](https://arxiv.org/pdf/2511.06304)  

**Abstract**: Since 2010, Kaggle has been a platform where data scientists from around the world come together to compete, collaborate, and push the boundaries of Data Science. Over these 15 years, it has grown from a purely competition-focused site into a broader ecosystem with forums, notebooks, models, datasets, and more. With the release of the Kaggle Meta Code and Kaggle Meta Datasets, we now have a unique opportunity to explore these competitions, technologies, and real-world applications of Machine Learning and AI. And so in this study, we take a closer look at 15 years of data science on Kaggle - through metadata, shared code, community discussions, and the competitions themselves. We explore Kaggle's growth, its impact on the data science community, uncover hidden technological trends, analyze competition winners, how Kagglers approach problems in general, and more. We do this by analyzing millions of kernels and discussion threads to perform both longitudinal trend analysis and standard exploratory data analysis. Our findings show that Kaggle is a steadily growing platform with increasingly diverse use cases, and that Kagglers are quick to adapt to new trends and apply them to real-world challenges, while producing - on average - models with solid generalization capabilities. We also offer a snapshot of the platform as a whole, highlighting its history and technological evolution. Finally, this study is accompanied by a video (this https URL) and a Kaggle write-up (this https URL) for your convenience. 

---
# Decomate: Leveraging Generative Models for Co-Creative SVG Animation 

**Authors**: Jihyeon Park, Jiyoon Myung, Seone Shin, Jungki Son, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.06297)  

**Abstract**: Designers often encounter friction when animating static SVG graphics, especially when the visual structure does not match the desired level of motion detail. Existing tools typically depend on predefined groupings or require technical expertise, which limits designers' ability to experiment and iterate independently. We present Decomate, a system that enables intuitive SVG animation through natural language. Decomate leverages a multimodal large language model to restructure raw SVGs into semantically meaningful, animation-ready components. Designers can then specify motions for each component via text prompts, after which the system generates corresponding HTML/CSS/JS animations. By supporting iterative refinement through natural language interaction, Decomate integrates generative AI into creative workflows, allowing animation outcomes to be directly shaped by user intent. 

---
# Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention 

**Authors**: Wenjie Hu, Sidun Liu, Peng Qiao, Zhenglun Sun, Yong Dou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06294)  

**Abstract**: Recent advances in Transformer-based Neural Operators have enabled significant progress in data-driven solvers for Partial Differential Equations (PDEs). Most current research has focused on reducing the quadratic complexity of attention to address the resulting low training and inference efficiency. Among these works, Transolver stands out as a representative method that introduces Physics-Attention to reduce computational costs. Physics-Attention projects grid points into slices for slice attention, then maps them back through deslicing. However, we observe that Physics-Attention can be reformulated as a special case of linear attention, and that the slice attention may even hurt the model performance. Based on these observations, we argue that its effectiveness primarily arises from the slice and deslice operations rather than interactions between slices. Building on this insight, we propose a two-step transformation to redesign Physics-Attention into a canonical linear attention, which we call Linear Attention Neural Operator (LinearNO). Our method achieves state-of-the-art performance on six standard PDE benchmarks, while reducing the number of parameters by an average of 40.0% and computational cost by 36.2%. Additionally, it delivers superior performance on two challenging, industrial-level datasets: AirfRANS and Shape-Net Car. 

---
# Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation 

**Authors**: Peng He, Yanglei Gan, Tingting Dai, Run Lin, Xuexin Li, Yao Liu, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06285)  

**Abstract**: Sequential recommendation (SR) aims to predict a user's next item preference by modeling historical interaction sequences. Recent advances often integrate frequency-domain modules to compensate for self-attention's low-pass nature by restoring the high-frequency signals critical for personalized recommendations. Nevertheless, existing frequency-aware solutions process each session in isolation and optimize exclusively with time-domain objectives. Consequently, they overlook cross-session spectral dependencies and fail to enforce alignment between predicted and actual spectral signatures, leaving valuable frequency information under-exploited. To this end, we propose FreqRec, a Frequency-Enhanced Dual-Path Network for sequential Recommendation that jointly captures inter-session and intra-session behaviors via a learnable Frequency-domain Multi-layer Perceptrons. Moreover, FreqRec is optimized under a composite objective that combines cross entropy with a frequency-domain consistency loss, explicitly aligning predicted and true spectral signatures. Extensive experiments on three benchmarks show that FreqRec surpasses strong baselines and remains robust under data sparsity and noisy-log conditions. 

---
# COTN: A Chaotic Oscillatory Transformer Network for Complex Volatile Systems under Extreme Conditions 

**Authors**: Boyan Tang, Yilong Zeng, Xuanhao Ren, Peng Xiao, Yuhan Zhao, Raymond Lee, Jianghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06273)  

**Abstract**: Accurate prediction of financial and electricity markets, especially under extreme conditions, remains a significant challenge due to their intrinsic nonlinearity, rapid fluctuations, and chaotic patterns. To address these limitations, we propose the Chaotic Oscillatory Transformer Network (COTN). COTN innovatively combines a Transformer architecture with a novel Lee Oscillator activation function, processed through Max-over-Time pooling and a lambda-gating mechanism. This design is specifically tailored to effectively capture chaotic dynamics and improve responsiveness during periods of heightened volatility, where conventional activation functions (e.g., ReLU, GELU) tend to saturate. Furthermore, COTN incorporates an Autoencoder Self-Regressive (ASR) module to detect and isolate abnormal market patterns, such as sudden price spikes or crashes, thereby preventing corruption of the core prediction process and enhancing robustness. Extensive experiments across electricity spot markets and financial markets demonstrate the practical applicability and resilience of COTN. Our approach outperforms state-of-the-art deep learning models like Informer by up to 17% and traditional statistical methods like GARCH by as much as 40%. These results underscore COTN's effectiveness in navigating real-world market uncertainty and complexity, offering a powerful tool for forecasting highly volatile systems under duress. 

---
# LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation 

**Authors**: Zijie Wang, Weiming Zhang, Wei Zhang, Xiao Tan, Hongxing Liu, Yaowei Wang, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06272)  

**Abstract**: Centerline graphs, crucial for path planning in autonomous driving, are traditionally learned using deterministic methods. However, these methods often lack spatial reasoning and struggle with occluded or invisible centerlines. Generative approaches, despite their potential, remain underexplored in this domain. We introduce LaneDiffusion, a novel generative paradigm for centerline graph learning. LaneDiffusion innovatively employs diffusion models to generate lane centerline priors at the Bird's Eye View (BEV) feature level, instead of directly predicting vectorized centerlines. Our method integrates a Lane Prior Injection Module (LPIM) and a Lane Prior Diffusion Module (LPDM) to effectively construct diffusion targets and manage the diffusion process. Furthermore, vectorized centerlines and topologies are then decoded from these prior-injected BEV features. Extensive evaluations on the nuScenes and Argoverse2 datasets demonstrate that LaneDiffusion significantly outperforms existing methods, achieving improvements of 4.2%, 4.6%, 4.7%, 6.4% and 1.8% on fine-grained point-level metrics (GEO F1, TOPO F1, JTOPO F1, APLS and SDA) and 2.3%, 6.4%, 6.8% and 2.1% on segment-level metrics (IoU, mAP_cf, DET_l and TOP_ll). These results establish state-of-the-art performance in centerline graph learning, offering new insights into generative models for this task. 

---
# LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling 

**Authors**: Hanlin Sun, Jiayang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06260)  

**Abstract**: Large language models (LLMs) are increasingly used as behavioral proxies for self-interested travelers in agent-based traffic models. Although more flexible and generalizable than conventional models, the practical use of these approaches remains limited by scalability due to the cost of calling one LLM for every traveler. Moreover, it has been found that LLM agents often make opaque choices and produce unstable day-to-day dynamics. To address these challenges, we propose to model each homogeneous traveler group facing the same decision context with a single representative LLM agent who behaves like the population's average, maintaining and updating a mixed strategy over routes that coincides with the group's aggregate flow proportions. Each day, the LLM reviews the travel experience and flags routes with positive reinforcement that they hope to use more often, and an interpretable update rule then converts this judgment into strategy adjustments using a tunable (progressively decaying) step size. The representative-agent design improves scalability, while the separation of reasoning from updating clarifies the decision logic while stabilizing learning. In classic traffic assignment settings, we find that the proposed approach converges rapidly to the user equilibrium. In richer settings with income heterogeneity, multi-criteria costs, and multi-modal choices, the generated dynamics remain stable and interpretable, reproducing plausible behavioral patterns well-documented in psychology and economics, for example, the decoy effect in toll versus non-toll road selection, and higher willingness-to-pay for convenience among higher-income travelers when choosing between driving, transit, and park-and-ride options. 

---
# Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra 

**Authors**: Yiwen Zhang, Keyan Ding, Yihang Wu, Xiang Zhuang, Yi Yang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06259)  

**Abstract**: Retrieving molecular structures from tandem mass spectra is a crucial step in rapid compound identification. Existing retrieval methods, such as traditional mass spectral library matching, suffer from limited spectral library coverage, while recent cross-modal representation learning frameworks often encounter modality misalignment, resulting in suboptimal retrieval accuracy and generalization. To address these limitations, we propose GLMR, a Generative Language Model-based Retrieval framework that mitigates the cross-modal misalignment through a two-stage process. In the pre-retrieval stage, a contrastive learning-based model identifies top candidate molecules as contextual priors for the input mass spectrum. In the generative retrieval stage, these candidate molecules are integrated with the input mass spectrum to guide a generative model in producing refined molecular structures, which are then used to re-rank the candidates based on molecular similarity. Experiments on both MassSpecGym and the proposed MassRET-20k dataset demonstrate that GLMR significantly outperforms existing methods, achieving over 40% improvement in top-1 accuracy and exhibiting strong generalizability. 

---
# MrCoM: A Meta-Regularized World-Model Generalizing Across Multi-Scenarios 

**Authors**: Xuantang Xiong, Ni Mu, Runpeng Xie, Senhao Yang, Yaqing Wang, Lexiang Wang, Yao Luan, Siyuan Li, Shuang Xu, Yiqin Yang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06252)  

**Abstract**: Model-based reinforcement learning (MBRL) is a crucial approach to enhance the generalization capabilities and improve the sample efficiency of RL algorithms. However, current MBRL methods focus primarily on building world models for single tasks and rarely address generalization across different scenarios. Building on the insight that dynamics within the same simulation engine share inherent properties, we attempt to construct a unified world model capable of generalizing across different scenarios, named Meta-Regularized Contextual World-Model (MrCoM). This method first decomposes the latent state space into various components based on the dynamic characteristics, thereby enhancing the accuracy of world-model prediction. Further, MrCoM adopts meta-state regularization to extract unified representation of scenario-relevant information, and meta-value regularization to align world-model optimization with policy learning across diverse scenario objectives. We theoretically analyze the generalization error upper bound of MrCoM in multi-scenario settings. We systematically evaluate our algorithm's generalization ability across diverse scenarios, demonstrating significantly better performance than previous state-of-the-art methods. 

---
# WebVIA: A Web-based Vision-Language Agentic Framework for Interactive and Verifiable UI-to-Code Generation 

**Authors**: Mingde Xu, Zhen Yang, Wenyi Hong, Lihang Pan, Xinyue Fan, Yan Wang, Xiaotao Gu, Bin Xu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06251)  

**Abstract**: User interface (UI) development requires translating design mockups into functional code, a process that remains repetitive and labor-intensive. While recent Vision-Language Models (VLMs) automate UI-to-Code generation, they generate only static HTML/CSS/JavaScript layouts lacking interactivity. To address this, we propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation. The framework comprises three components: 1) an exploration agent to capture multi-state UI screenshots; 2) a UI2Code model that generates executable interactive code; 3) a validation module that verifies the interactivity. Experiments demonstrate that WebVIA-Agent achieves more stable and accurate UI exploration than general-purpose agents (e.g., Gemini-2.5-Pro). In addition, our fine-tuned WebVIA-UI2Code models exhibit substantial improvements in generating executable and interactive HTML/CSS/JavaScript code, outperforming their base counterparts across both interactive and static UI2Code benchmarks. Our code and models are available at \href{this https URL}{\texttt{this https URL}}. 

---
# Constraint-Informed Active Learning for End-to-End ACOPF Optimization Proxies 

**Authors**: Miao Li, Michael Klamkin, Pascal Van Hentenryck, Wenting Li, Russell Bent  

**Link**: [PDF](https://arxiv.org/pdf/2511.06248)  

**Abstract**: This paper studies optimization proxies, machine learning (ML) models trained to efficiently predict optimal solutions for AC Optimal Power Flow (ACOPF) problems. While promising, optimization proxy performance heavily depends on training data quality. To address this limitation, this paper introduces a novel active sampling framework for ACOPF optimization proxies designed to generate realistic and diverse training data. The framework actively explores varied, flexible problem specifications reflecting plausible operational realities. More importantly, the approach uses optimization-specific quantities (active constraint sets) that better capture the salient features of an ACOPF that lead to the optimal solution. Numerical results show superior generalization over existing sampling methods with an equivalent training budget, significantly advancing the state-of-practice for trustworthy ACOPF optimization proxies. 

---
# Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation 

**Authors**: Tzu-Jung Lin, Jia-Fong Yeh, Hung-Ting Su, Chung-Yi Lin, Yi-Ting Chen, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06240)  

**Abstract**: In open-vocabulary mobile manipulation (OVMM), task success often hinges on the selection of an appropriate base placement for the robot. Existing approaches typically navigate to proximity-based regions without considering affordances, resulting in frequent manipulation failures. We propose Affordance-Guided Coarse-to-Fine Exploration, a zero-shot framework for base placement that integrates semantic understanding from vision-language models (VLMs) with geometric feasibility through an iterative optimization process. Our method constructs cross-modal representations, namely Affordance RGB and Obstacle Map+, to align semantics with spatial context. This enables reasoning that extends beyond the egocentric limitations of RGB perception. To ensure interaction is guided by task-relevant affordances, we leverage coarse semantic priors from VLMs to guide the search toward task-relevant regions and refine placements with geometric constraints, thereby reducing the risk of convergence to local optima. Evaluated on five diverse open-vocabulary mobile manipulation tasks, our system achieves an 85% success rate, significantly outperforming classical geometric planners and VLM-based methods. This demonstrates the promise of affordance-aware and multimodal reasoning for generalizable, instruction-conditioned planning in OVMM. 

---
# Mixtures of SubExperts for Large Language Continual Learning 

**Authors**: Haeyong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06237)  

**Abstract**: Adapting Large Language Models (LLMs) to a continuous stream of tasks is a critical yet challenging endeavor. While Parameter-Efficient Fine-Tuning (PEFT) methods have become a standard for this, they face a fundamental dilemma in continual learning. Reusing a single set of PEFT parameters for new tasks often leads to catastrophic forgetting of prior knowledge. Conversely, allocating distinct parameters for each task prevents forgetting but results in a linear growth of the model's size and fails to facilitate knowledge transfer between related tasks. To overcome these limitations, we propose a novel adaptive PEFT method referred to as \textit{Mixtures of SubExperts (MoSEs)}, a novel continual learning framework designed for minimal forgetting and efficient scalability. MoSEs integrate a sparse Mixture of SubExperts into the transformer layers, governed by a task-specific routing mechanism. This architecture allows the model to isolate and protect knowledge within dedicated SubExperts, thereby minimizing parameter interference and catastrophic forgetting. Crucially, the router can adaptively select and combine previously learned sparse parameters for new tasks, enabling effective knowledge transfer while ensuring that the model's capacity grows sublinearly. We evaluate MoSEs on the comprehensive TRACE benchmark datasets. Our experiments demonstrate that MoSEs significantly outperform conventional continual learning approaches in both knowledge retention and scalability to new tasks, achieving state-of-the-art performance with substantial memory and computational savings. 

---
# Analyzing and Mitigating Negation Artifacts using Data Augmentation for Improving ELECTRA-Small Model Accuracy 

**Authors**: Mojtaba Noghabaei  

**Link**: [PDF](https://arxiv.org/pdf/2511.06234)  

**Abstract**: Pre-trained models for natural language inference (NLI) often achieve high performance on benchmark datasets by using spurious correlations, or dataset artifacts, rather than understanding language touches such as negation. In this project, we investigate the performance of an ELECTRA-small model fine-tuned on the Stanford Natural Language Inference (SNLI) dataset, focusing on its handling of negation. Through analysis, we identify that the model struggles with correctly classifying examples containing negation. To address this, we augment the training data with contrast sets and adversarial examples emphasizing negation. Our results demonstrate that this targeted data augmentation improves the model's accuracy on negation-containing examples without adversely affecting overall performance, therefore mitigating the identified dataset artifact. 

---
# Scaling Laws and In-Context Learning: A Unified Theoretical Framework 

**Authors**: Sushant Mehta, Ishan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.06232)  

**Abstract**: In-context learning (ICL) enables large language models to adapt to new tasks from demonstrations without parameter updates. Despite extensive empirical studies, a principled understanding of ICL emergence at scale remains more elusive. We present a unified theoretical framework connecting scaling laws to ICL emergence in transformers. Our analysis establishes that ICL performance follows power-law relationships with model depth $L$, width $d$, context length $k$, and training data $D$, with exponents determined by task structure. We show that under specific conditions, transformers implement gradient-based metalearning in their forward pass, with an effective learning rate $\eta_{\text{eff}} = \Theta(1/\sqrt{Ld})$. We demonstrate sharp phase transitions at critical scales and derive optimal depth-width allocations favoring $L^* \propto N^{2/3}$, $d^* \propto N^{1/3}$ for the fixed parameter budget $N = Ld$. Systematic experiments on synthetic tasks validate our predictions, with measured scaling exponents closely matching theory. This work provides both necessary and sufficient conditions for the emergence of ICLs and establishes fundamental computational limits on what transformers can learn in-context. 

---
# Overview of CHIP 2025 Shared Task 2: Discharge Medication Recommendation for Metabolic Diseases Based on Chinese Electronic Health Records 

**Authors**: Juntao Li, Haobin Yuan, Ling Luo, Tengxiao Lv, Yan Jiang, Fan Wang, Ping Zhang, Huiyi Lv, Jian Wang, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06230)  

**Abstract**: Discharge medication recommendation plays a critical role in ensuring treatment continuity, preventing readmission, and improving long-term management for patients with chronic metabolic diseases. This paper present an overview of the CHIP 2025 Shared Task 2 competition, which aimed to develop state-of-the-art approaches for automatically recommending appro-priate discharge medications using real-world Chinese EHR data. For this task, we constructed CDrugRed, a high-quality dataset consisting of 5,894 de-identified hospitalization records from 3,190 patients in China. This task is challenging due to multi-label nature of medication recommendation, het-erogeneous clinical text, and patient-specific variability in treatment plans. A total of 526 teams registered, with 167 and 95 teams submitting valid results to the Phase A and Phase B leaderboards, respectively. The top-performing team achieved the highest overall performance on the final test set, with a Jaccard score of 0.5102, F1 score of 0.6267, demonstrating the potential of advanced large language model (LLM)-based ensemble systems. These re-sults highlight both the promise and remaining challenges of applying LLMs to medication recommendation in Chinese EHRs. The post-evaluation phase remains open at this https URL. 

---
# Assertion-Aware Test Code Summarization with Large Language Models 

**Authors**: Anamul Haque Mollah, Ahmed Aljohani, Hyunsook Do  

**Link**: [PDF](https://arxiv.org/pdf/2511.06227)  

**Abstract**: Unit tests often lack concise summaries that convey test intent, especially in auto-generated or poorly documented codebases. Large Language Models (LLMs) offer a promising solution, but their effectiveness depends heavily on how they are prompted. Unlike generic code summarization, test-code summarization poses distinct challenges because test methods validate expected behavior through assertions rather than im- plementing functionality. This paper presents a new benchmark of 91 real-world Java test cases paired with developer-written summaries and conducts a controlled ablation study to investigate how test code-related components-such as the method under test (MUT), assertion messages, and assertion semantics-affect the performance of LLM-generated test summaries. We evaluate four code LLMs (Codex, Codestral, DeepSeek, and Qwen-Coder) across seven prompt configurations using n-gram metrics (BLEU, ROUGE-L, METEOR), semantic similarity (BERTScore), and LLM-based evaluation. Results show that prompting with as- sertion semantics improves summary quality by an average of 0.10 points (2.3%) over full MUT context (4.45 vs. 4.35) while requiring fewer input tokens. Codex and Qwen-Coder achieve the highest alignment with human-written summaries, while DeepSeek underperforms despite high lexical overlap. The replication package is publicly available at this https URL. 5281/zenodo.17067550 

---
# Explicit Knowledge-Guided In-Context Learning for Early Detection of Alzheimer's Disease 

**Authors**: Puzhen Su, Yongzhu Miao, Chunxi Guo, Jintao Tang, Shasha Li, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06215)  

**Abstract**: Detecting Alzheimer's Disease (AD) from narrative transcripts remains a challenging task for large language models (LLMs), particularly under out-of-distribution (OOD) and data-scarce conditions. While in-context learning (ICL) provides a parameter-efficient alternative to fine-tuning, existing ICL approaches often suffer from task recognition failure, suboptimal demonstration selection, and misalignment between label words and task objectives, issues that are amplified in clinical domains like AD detection. We propose Explicit Knowledge In-Context Learners (EK-ICL), a novel framework that integrates structured explicit knowledge to enhance reasoning stability and task alignment in ICL. EK-ICL incorporates three knowledge components: confidence scores derived from small language models (SLMs) to ground predictions in task-relevant patterns, parsing feature scores to capture structural differences and improve demo selection, and label word replacement to resolve semantic misalignment with LLM priors. In addition, EK-ICL employs a parsing-based retrieval strategy and ensemble prediction to mitigate the effects of semantic homogeneity in AD transcripts. Extensive experiments across three AD datasets demonstrate that EK-ICL significantly outperforms state-of-the-art fine-tuning and ICL baselines. Further analysis reveals that ICL performance in AD detection is highly sensitive to the alignment of label semantics and task-specific context, underscoring the importance of explicit knowledge in clinical reasoning under low-resource conditions. 

---
# RAG-targeted Adversarial Attack on LLM-based Threat Detection and Mitigation Framework 

**Authors**: Seif Ikbarieh, Kshitiz Aryal, Maanak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.06212)  

**Abstract**: The rapid expansion of the Internet of Things (IoT) is reshaping communication and operational practices across industries, but it also broadens the attack surface and increases susceptibility to security breaches. Artificial Intelligence has become a valuable solution in securing IoT networks, with Large Language Models (LLMs) enabling automated attack behavior analysis and mitigation suggestion in Network Intrusion Detection Systems (NIDS). Despite advancements, the use of LLMs in such systems further expands the attack surface, putting entire networks at risk by introducing vulnerabilities such as prompt injection and data poisoning. In this work, we attack an LLM-based IoT attack analysis and mitigation framework to test its adversarial robustness. We construct an attack description dataset and use it in a targeted data poisoning attack that applies word-level, meaning-preserving perturbations to corrupt the Retrieval-Augmented Generation (RAG) knowledge base of the framework. We then compare pre-attack and post-attack mitigation responses from the target model, ChatGPT-5 Thinking, to measure the impact of the attack on model performance, using an established evaluation rubric designed for human experts and judge LLMs. Our results show that small perturbations degrade LLM performance by weakening the linkage between observed network traffic features and attack behavior, and by reducing the specificity and practicality of recommended mitigations for resource-constrained devices. 

---
# Resilience Inference for Supply Chains with Hypergraph Neural Network 

**Authors**: Zetian Shen, Hongjun Wang, Jiyuan Chen, Xuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.06208)  

**Abstract**: Supply chains are integral to global economic stability, yet disruptions can swiftly propagate through interconnected networks, resulting in substantial economic impacts. Accurate and timely inference of supply chain resilience the capability to maintain core functions during disruptions is crucial for proactive risk mitigation and robust network design. However, existing approaches lack effective mechanisms to infer supply chain resilience without explicit system dynamics and struggle to represent the higher-order, multi-entity dependencies inherent in supply chain networks. These limitations motivate the definition of a novel problem and the development of targeted modeling solutions. To address these challenges, we formalize a novel problem: Supply Chain Resilience Inference (SCRI), defined as predicting supply chain resilience using hypergraph topology and observed inventory trajectories without explicit dynamic equations. To solve this problem, we propose the Supply Chain Resilience Inference Hypergraph Network (SC-RIHN), a novel hypergraph-based model leveraging set-based encoding and hypergraph message passing to capture multi-party firm-product interactions. Comprehensive experiments demonstrate that SC-RIHN significantly outperforms traditional MLP, representative graph neural network variants, and ResInf baselines across synthetic benchmarks, underscoring its potential for practical, early-warning risk assessment in complex supply chain systems. 

---
# Enhancing Adversarial Robustness of IoT Intrusion Detection via SHAP-Based Attribution Fingerprinting 

**Authors**: Dilli Prasad Sharma, Liang Xue, Xiaowei Sun, Xiaodong Lin, Pulei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06197)  

**Abstract**: The rapid proliferation of Internet of Things (IoT) devices has transformed numerous industries by enabling seamless connectivity and data-driven automation. However, this expansion has also exposed IoT networks to increasingly sophisticated security threats, including adversarial attacks targeting artificial intelligence (AI) and machine learning (ML)-based intrusion detection systems (IDS) to deliberately evade detection, induce misclassification, and systematically undermine the reliability and integrity of security defenses. To address these challenges, we propose a novel adversarial detection model that enhances the robustness of IoT IDS against adversarial attacks through SHapley Additive exPlanations (SHAP)-based fingerprinting. Using SHAP's DeepExplainer, we extract attribution fingerprints from network traffic features, enabling the IDS to reliably distinguish between clean and adversarially perturbed inputs. By capturing subtle attribution patterns, the model becomes more resilient to evasion attempts and adversarial manipulations. We evaluated the model on a standard IoT benchmark dataset, where it significantly outperformed a state-of-the-art method in detecting adversarial attacks. In addition to enhanced robustness, this approach improves model transparency and interpretability, thereby increasing trust in the IDS through explainable AI. 

---
# AI as intermediary in modern-day ritual: An immersive, interactive production of the roller disco musical Xanadu at UCLA 

**Authors**: Mira Winick, Naisha Agarwal, Chiheb Boussema, Ingrid Lee, Camilo Vargas, Jeff Burke  

**Link**: [PDF](https://arxiv.org/pdf/2511.06195)  

**Abstract**: Interfaces for contemporary large language, generative media, and perception AI models are often engineered for single user interaction. We investigate ritual as a design scaffold for developing collaborative, multi-user human-AI engagement. We consider the specific case of an immersive staging of the musical Xanadu performed at UCLA in Spring 2025. During a two-week run, over five hundred audience members contributed sketches and jazzercise moves that vision language models translated to virtual scenery elements and from choreographic prompts. This paper discusses four facets of interaction-as-ritual within the show: audience input as offerings that AI transforms into components of the ritual; performers as ritual guides, demonstrating how to interact with technology and sorting audience members into cohorts; AI systems as instruments "played" by the humans, in which sensing, generative components, and stagecraft create systems that can be mastered over time; and reciprocity of interaction, in which the show's AI machinery guides human behavior as well as being guided by humans, completing a human-AI feedback loop that visibly reshapes the virtual world. Ritual served as a frame for integrating linear narrative, character identity, music and interaction. The production explored how AI systems can support group creativity and play, addressing a critical gap in prevailing single user AI design paradigms. 

---
# MemoriesDB: A Temporal-Semantic-Relational Database for Long-Term Agent Memory / Modeling Experience as a Graph of Temporal-Semantic Surfaces 

**Authors**: Joel Ward  

**Link**: [PDF](https://arxiv.org/pdf/2511.06179)  

**Abstract**: We introduce MemoriesDB, a unified data architecture designed to avoid decoherence across time, meaning, and relation in long-term computational memory. Each memory is a time-semantic-relational entity-a structure that simultaneously encodes when an event occurred, what it means, and how it connects to other events. Built initially atop PostgreSQL with pgvector extensions, MemoriesDB combines the properties of a time-series datastore, a vector database, and a graph system within a single append-only schema. Each memory is represented as a vertex uniquely labeled by its microsecond timestamp and accompanied by low- and high-dimensional normalized embeddings that capture semantic context. Directed edges between memories form labeled relations with per-edge metadata, enabling multiple contextual links between the same vertices. Together these constructs form a time-indexed stack of temporal-semantic surfaces, where edges project as directional arrows in a 1+1-dimensional similarity field, tracing the evolution of meaning through time while maintaining cross-temporal coherence. This formulation supports efficient time-bounded retrieval, hybrid semantic search, and lightweight structural reasoning in a single query path. A working prototype demonstrates scalable recall and contextual reinforcement using standard relational infrastructure, and we discuss extensions toward a columnar backend, distributed clustering, and emergent topic modeling. 

---
# LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs 

**Authors**: Zifan He, Shengyu Ye, Rui Ma, Yang Wang, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06174)  

**Abstract**: The rapid progress of large language models (LLMs) has advanced numerous applications, yet efficient single-batch inference remains vital for on-device intelligence. While FPGAs offer fine-grained data control and high energy efficiency, recent GPU optimizations have narrowed their advantage, especially under arithmetic-based computation. To overcome this, we leverage FPGAs' abundant on-chip memory to shift LLM inference from arithmetic- to memory-based computation through table lookups. We present LUT-LLM, the first FPGA accelerator enabling 1B+ LLM inference via vector-quantized memory operations. Our analysis identifies activation-weight co-quantization as the most effective scheme, supported by (1) bandwidth-aware parallel centroid search, (2) efficient 2D table lookups, and (3) a spatial-temporal hybrid design minimizing data caching. Implemented on an AMD V80 FPGA for a customized Qwen 3 1.7B model, LUT-LLM achieves 1.66x lower latency than AMD MI210 and 1.72x higher energy efficiency than NVIDIA A100, scaling to 32B models with 2.16x efficiency gain over A100. 

---
# MambaOVSR: Multiscale Fusion with Global Motion Modeling for Chinese Opera Video Super-Resolution 

**Authors**: Hua Chang, Xin Xu, Wei Liu, Wei Wang, Xin Yuan, Kui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06172)  

**Abstract**: Chinese opera is celebrated for preserving classical art. However, early filming equipment limitations have degraded videos of last-century performances by renowned artists (e.g., low frame rates and resolution), hindering archival efforts. Although space-time video super-resolution (STVSR) has advanced significantly, applying it directly to opera videos remains challenging. The scarcity of datasets impedes the recovery of high frequency details, and existing STVSR methods lack global modeling capabilities, compromising visual quality when handling opera's characteristic large motions. To address these challenges, we pioneer a large scale Chinese Opera Video Clip (COVC) dataset and propose the Mamba-based multiscale fusion network for space-time Opera Video Super-Resolution (MambaOVSR). Specifically, MambaOVSR involves three novel components: the Global Fusion Module (GFM) for motion modeling through a multiscale alternating scanning mechanism, and the Multiscale Synergistic Mamba Module (MSMM) for alignment across different sequence lengths. Additionally, our MambaVR block resolves feature artifacts and positional information loss during alignment. Experimental results on the COVC dataset show that MambaOVSR significantly outperforms the SOTA STVSR method by an average of 1.86 dB in terms of PSNR. Dataset and Code will be publicly released. 

---
# LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains 

**Authors**: Ibna Kowsar, Kazi F. Akhter, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2511.06161)  

**Abstract**: Transfer learning of tabular data is non-trivial due to heterogeneity in the feature space across disparate domains. The limited success of traditional deep learning in tabular knowledge transfer can be advanced by leveraging large language models (LLMs). However, the efficacy of LLMs often stagnates for mixed data types structured in tables due to the limitations of text prompts and in-context learning. We propose a lightweight transfer learning framework that fine-tunes an LLM using source tabular data and transplants the LLM's selective $key$ and $value$ projection weights into a gated feature tokenized transformer (gFTT) built for tabular data. The gFTT model with cross-domain attention is fine-tuned using target tabular data for transfer learning, eliminating the need for shared features, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of source-target data sets and 12 baselines demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and transfer learning models trained on thousands to billions of tabular samples. The proposed attention transfer demonstrates an effective solution to learning relationships between data tables using an LLM in a low-resource learning environment. The source code for the proposed method is publicly available. 

---
# Models Got Talent: Identifying High Performing Wearable Human Activity Recognition Models Without Training 

**Authors**: Richard Goldman, Varun Komperla, Thomas Ploetz, Harish Haresamudram  

**Link**: [PDF](https://arxiv.org/pdf/2511.06157)  

**Abstract**: A promising alternative to the computationally expensive Neural Architecture Search (NAS) involves the development of \textit{Zero Cost Proxies (ZCPs)}, which correlate well to trained performance, but can be computed through a single forward/backward pass on a randomly sampled batch of data. In this paper, we investigate the effectiveness of ZCPs for HAR on six benchmark datasets, and demonstrate that they discover network architectures that obtain within 5\% of performance attained by full scale training involving 1500 randomly sampled architectures. This results in substantial computational savings as high performing architectures can be discovered with minimal training. Our experiments not only introduce ZCPs to sensor-based HAR, but also demonstrate that they are robust to data noise, further showcasing their suitability for practical scenarios. 

---
# Large Language Models Develop Novel Social Biases Through Adaptive Exploration 

**Authors**: Addison J. Wu, Ryan Liu, Xuechunzi Bai, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.06148)  

**Abstract**: As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time. 

---
# Referring Expressions as a Lens into Spatial Language Grounding in Vision-Language Models 

**Authors**: Akshar Tumu, Varad Shinde, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06146)  

**Abstract**: Spatial Reasoning is an important component of human cognition and is an area in which the latest Vision-language models (VLMs) show signs of difficulty. The current analysis works use image captioning tasks and visual question answering. In this work, we propose using the Referring Expression Comprehension task instead as a platform for the evaluation of spatial reasoning by VLMs. This platform provides the opportunity for a deeper analysis of spatial comprehension and grounding abilities when there is 1) ambiguity in object detection, 2) complex spatial expressions with a longer sentence structure and multiple spatial relations, and 3) expressions with negation ('not'). In our analysis, we use task-specific architectures as well as large VLMs and highlight their strengths and weaknesses in dealing with these specific situations. While all these models face challenges with the task at hand, the relative behaviors depend on the underlying models and the specific categories of spatial semantics (topological, directional, proximal, etc.). Our results highlight these challenges and behaviors and provide insight into research gaps and future directions. 

---
# Evaluation of retrieval-based QA on QUEST-LOFT 

**Authors**: Nathan Scales, Nathanael Schärli, Olivier Bousquet  

**Link**: [PDF](https://arxiv.org/pdf/2511.06125)  

**Abstract**: Despite the popularity of retrieval-augmented generation (RAG) as a solution for grounded QA in both academia and industry, current RAG methods struggle with questions where the necessary information is distributed across many documents or where retrieval needs to be combined with complex reasoning. Recently, the LOFT study has shown that this limitation also applies to approaches based on long-context language models, with the QUEST benchmark exhibiting particularly large headroom. In this paper, we provide an in-depth analysis of the factors contributing to the poor performance on QUEST-LOFT, publish updated numbers based on a thorough human evaluation, and demonstrate that RAG can be optimized to significantly outperform long-context approaches when combined with a structured output format containing reasoning and evidence, optionally followed by answer re-verification. 

---
# Adapting Web Agents with Synthetic Supervision 

**Authors**: Zhaoyang Wang, Yiming Liang, Xuchao Zhang, Qianhui Wu, Siwei Han, Anson Bastos, Rujia Wang, Chetan Bansal, Baolin Peng, Jianfeng Gao, Saravan Rajmohan, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06101)  

**Abstract**: Web agents struggle to adapt to new websites due to the scarcity of environment specific tasks and demonstrations. Recent works have explored synthetic data generation to address this challenge, however, they suffer from data quality issues where synthesized tasks contain hallucinations that cannot be executed, and collected trajectories are noisy with redundant or misaligned actions. In this paper, we propose SynthAgent, a fully synthetic supervision framework that aims at improving synthetic data quality via dual refinement of both tasks and trajectories. Our approach begins by synthesizing diverse tasks through categorized exploration of web elements, ensuring efficient coverage of the target environment. During trajectory collection, we refine tasks when conflicts with actual observations are detected, mitigating hallucinations while maintaining task consistency. After collection, we conduct trajectory refinement with a global context to mitigate potential noise or misalignments. Finally, we fine-tune open-source web agents on the refined synthetic data to adapt them to the target environment. Experimental results demonstrate that SynthAgent outperforms existing synthetic data methods, validating the importance of high-quality synthetic supervision. The code will be publicly available at this https URL. 

---
# SWE-fficiency: Can Language Models Optimize Real-World Repositories on Real Workloads? 

**Authors**: Jeffrey Jian Ma, Milad Hashemi, Amir Yazdanbakhsh, Kevin Swersky, Ofir Press, Enhui Li, Vijay Janapa Reddi, Parthasarathy Ranganathan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06090)  

**Abstract**: Optimizing the performance of large-scale software repositories demands expertise in code reasoning and software engineering (SWE) to reduce runtime while preserving program correctness. However, most benchmarks emphasize what to fix rather than how to fix code. We introduce \textsc{SWE-fficiency}, a benchmark for evaluating repository-level performance optimization on real workloads. Our suite contains 498 tasks across nine widely used data-science, machine-learning, and HPC repositories (e.g., numpy, pandas, scipy): given a complete codebase and a slow workload, an agent must investigate code semantics, localize bottlenecks and relevant tests, and produce a patch that matches or exceeds expert speedup while passing the same unit tests. To enable this how-to-fix evaluation, our automated pipeline scrapes GitHub pull requests for performance-improving edits, combining keyword filtering, static analysis, coverage tooling, and execution validation to both confirm expert speedup baselines and identify relevant repository unit tests. Empirical evaluation of state-of-the-art agents reveals significant underperformance. On average, agents achieve less than 0.15x the expert speedup: agents struggle in localizing optimization opportunities, reasoning about execution across functions, and maintaining correctness in proposed edits. We release the benchmark and accompanying data pipeline to facilitate research on automated performance engineering and long-horizon software reasoning. 

---
# Hybrid CNN-ViT Framework for Motion-Blurred Scene Text Restoration 

**Authors**: Umar Rashid, Muhammad Arslan Arshad, Ghulam Ahmad, Muhammad Zeeshan Anjum, Rizwan Khan, Muhammad Akmal  

**Link**: [PDF](https://arxiv.org/pdf/2511.06087)  

**Abstract**: Motion blur in scene text images severely impairs readability and hinders the reliability of computer vision tasks, including autonomous driving, document digitization, and visual information retrieval. Conventional deblurring approaches are often inadequate in handling spatially varying blur and typically fall short in modeling the long-range dependencies necessary for restoring textual clarity. To overcome these limitations, we introduce a hybrid deep learning framework that combines convolutional neural networks (CNNs) with vision transformers (ViTs), thereby leveraging both local feature extraction and global contextual reasoning. The architecture employs a CNN-based encoder-decoder to preserve structural details, while a transformer module enhances global awareness through self-attention. Training is conducted on a curated dataset derived from TextOCR, where sharp scene-text samples are paired with synthetically blurred versions generated using realistic motion-blur kernels of multiple sizes and orientations. Model optimization is guided by a composite loss that incorporates mean absolute error (MAE), squared error (MSE), perceptual similarity, and structural similarity (SSIM). Quantitative eval- uations show that the proposed method attains 32.20 dB in PSNR and 0.934 in SSIM, while remaining lightweight with 2.83 million parameters and an average inference time of 61 ms. These results highlight the effectiveness and computational efficiency of the CNN-ViT hybrid design, establishing its practicality for real-world motion-blurred scene-text restoration. 

---
# Simulating Students with Large Language Models: A Review of Architecture, Mechanisms, and Role Modelling in Education with Generative AI 

**Authors**: Luis Marquez-Carpintero, Alberto Lopez-Sellers, Miguel Cazorla  

**Link**: [PDF](https://arxiv.org/pdf/2511.06078)  

**Abstract**: Simulated Students offer a valuable methodological framework for evaluating pedagogical approaches and modelling diverse learner profiles, tasks which are otherwise challenging to undertake systematically in real-world settings. Recent research has increasingly focused on developing such simulated agents to capture a range of learning styles, cognitive development pathways, and social behaviours. Among contemporary simulation techniques, the integration of large language models (LLMs) into educational research has emerged as a particularly versatile and scalable paradigm. LLMs afford a high degree of linguistic realism and behavioural adaptability, enabling agents to approximate cognitive processes and engage in contextually appropriate pedagogical dialogues. This paper presents a thematic review of empirical and methodological studies utilising LLMs to simulate student behaviour across educational environments. We synthesise current evidence on the capacity of LLM-based agents to emulate learner archetypes, respond to instructional inputs, and interact within multi-agent classroom scenarios. Furthermore, we examine the implications of such systems for curriculum development, instructional evaluation, and teacher training. While LLMs surpass rule-based systems in natural language generation and situational flexibility, ongoing concerns persist regarding algorithmic bias, evaluation reliability, and alignment with educational objectives. The review identifies existing technological and methodological gaps and proposes future research directions for integrating generative AI into adaptive learning systems and instructional design. 

---
# Stemming Hallucination in Language Models Using a Licensing Oracle 

**Authors**: Simeon Emanuilov, Richard Ackermann  

**Link**: [PDF](https://arxiv.org/pdf/2511.06073)  

**Abstract**: Language models exhibit remarkable natural language generation capabilities but remain prone to hallucinations, generating factually incorrect information despite producing syntactically coherent responses. This study introduces the Licensing Oracle, an architectural solution designed to stem hallucinations in LMs by enforcing truth constraints through formal validation against structured knowledge graphs. Unlike statistical approaches that rely on data scaling or fine-tuning, the Licensing Oracle embeds a deterministic validation step into the model's generative process, ensuring that only factually accurate claims are made. We evaluated the effectiveness of the Licensing Oracle through experiments comparing it with several state-of-the-art methods, including baseline language model generation, fine-tuning for factual recall, fine-tuning for abstention behavior, and retrieval-augmented generation (RAG). Our results demonstrate that although RAG and fine-tuning improve performance, they fail to eliminate hallucinations. In contrast, the Licensing Oracle achieved perfect abstention precision (AP = 1.0) and zero false answers (FAR-NE = 0.0), ensuring that only valid claims were generated with 89.1% accuracy in factual responses. This work shows that architectural innovations, such as the Licensing Oracle, offer a necessary and sufficient solution for hallucinations in domains with structured knowledge representations, offering guarantees that statistical methods cannot match. Although the Licensing Oracle is specifically designed to address hallucinations in fact-based domains, its framework lays the groundwork for truth-constrained generation in future AI systems, providing a new path toward reliable, epistemically grounded models. 

---
# A Privacy-Preserving Federated Learning Method with Homomorphic Encryption in Omics Data 

**Authors**: Yusaku Negoya, Feifei Cui, Zilong Zhang, Miao Pan, Tomoaki Ohtsuki, Aohan Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06064)  

**Abstract**: Omics data is widely employed in medical research to identify disease mechanisms and contains highly sensitive personal information. Federated Learning (FL) with Differential Privacy (DP) can ensure the protection of omics data privacy against malicious user attacks. However, FL with the DP method faces an inherent trade-off: stronger privacy protection degrades predictive accuracy due to injected noise. On the other hand, Homomorphic Encryption (HE) allows computations on encrypted data and enables aggregation of encrypted gradients without DP-induced noise can increase the predictive accuracy. However, it may increase the computation cost. To improve the predictive accuracy while considering the computational ability of heterogeneous clients, we propose a Privacy-Preserving Machine Learning (PPML)-Hybrid method by introducing HE. In the proposed PPML-Hybrid method, clients distributed select either HE or DP based on their computational resources, so that HE clients contribute noise-free updates while DP clients reduce computational overhead. Meanwhile, clients with high computational resources clients can flexibly adopt HE or DP according to their privacy needs. Performance evaluation on omics datasets show that our proposed method achieves comparable predictive accuracy while significantly reducing computation time relative to HE-only. Additionally, it outperforms DP-only methods under equivalent or stricter privacy budgets. 

---
# How Particle-System Random Batch Methods Enhance Graph Transformer: Memory Efficiency and Parallel Computing Strategy 

**Authors**: Hanwen Liu, Yixuan Ma, Shi Jin, Yuguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06044)  

**Abstract**: Attention mechanism is a significant part of Transformer models. It helps extract features from embedded vectors by adding global information and its expressivity has been proved to be powerful. Nevertheless, the quadratic complexity restricts its practicability. Although several researches have provided attention mechanism in sparse form, they are lack of theoretical analysis about the expressivity of their mechanism while reducing complexity. In this paper, we put forward Random Batch Attention (RBA), a linear self-attention mechanism, which has theoretical support of the ability to maintain its expressivity. Random Batch Attention has several significant strengths as follows: (1) Random Batch Attention has linear time complexity. Other than this, it can be implemented in parallel on a new dimension, which contributes to much memory saving. (2) Random Batch Attention mechanism can improve most of the existing models by replacing their attention mechanisms, even many previously improved attention mechanisms. (3) Random Batch Attention mechanism has theoretical explanation in convergence, as it comes from Random Batch Methods on computation mathematics. Experiments on large graphs have proved advantages mentioned above. Also, the theoretical modeling of self-attention mechanism is a new tool for future research on attention-mechanism analysis. 

---
# Advancing Ocean State Estimation with efficient and scalable AI 

**Authors**: Yanfei Xiang, Yuan Gao, Hao Wu, Quan Zhang, Ruiqi Shu, Xiao Zhou, Xi Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06041)  

**Abstract**: Accurate and efficient global ocean state estimation remains a grand challenge for Earth system science, hindered by the dual bottlenecks of computational scalability and degraded data fidelity in traditional data assimilation (DA) and deep learning (DL) approaches. Here we present an AI-driven Data Assimilation Framework for Ocean (ADAF-Ocean) that directly assimilates multi-source and multi-scale observations, ranging from sparse in-situ measurements to 4 km satellite swaths, without any interpolation or data thinning. Inspired by Neural Processes, ADAF-Ocean learns a continuous mapping from heterogeneous inputs to ocean states, preserving native data fidelity. Through AI-driven super-resolution, it reconstructs 0.25$^\circ$ mesoscale dynamics from coarse 1$^\circ$ fields, which ensures both efficiency and scalability, with just 3.7\% more parameters than the 1$^\circ$ configuration. When coupled with a DL forecasting system, ADAF-Ocean extends global forecast skill by up to 20 days compared to baselines without assimilation. This framework establishes a computationally viable and scientifically rigorous pathway toward real-time, high-resolution Earth system monitoring. 

---
# S2ML: Spatio-Spectral Mutual Learning for Depth Completion 

**Authors**: Zihui Zhao, Yifei Zhang, Zheng Wang, Yang Li, Kui Jiang, Zihan Geng, Chia-Wen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06033)  

**Abstract**: The raw depth images captured by RGB-D cameras using Time-of-Flight (TOF) or structured light often suffer from incomplete depth values due to weak reflections, boundary shadows, and artifacts, which limit their applications in downstream vision tasks. Existing methods address this problem through depth completion in the image domain, but they overlook the physical characteristics of raw depth images. It has been observed that the presence of invalid depth areas alters the frequency distribution pattern. In this work, we propose a Spatio-Spectral Mutual Learning framework (S2ML) to harmonize the advantages of both spatial and frequency domains for depth completion. Specifically, we consider the distinct properties of amplitude and phase spectra and devise a dedicated spectral fusion module. Meanwhile, the local and global correlations between spatial-domain and frequency-domain features are calculated in a unified embedding space. The gradual mutual representation and refinement encourage the network to fully explore complementary physical characteristics and priors for more accurate depth completion. Extensive experiments demonstrate the effectiveness of our proposed S2ML method, outperforming the state-of-the-art method CFormer by 0.828 dB and 0.834 dB on the NYU-Depth V2 and SUN RGB-D datasets, respectively. 

---
# ITPP: Learning Disentangled Event Dynamics in Marked Temporal Point Processes 

**Authors**: Wang-Tao Zhou, Zhao Kang, Ke Yan, Ling Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06032)  

**Abstract**: Marked Temporal Point Processes (MTPPs) provide a principled framework for modeling asynchronous event sequences by conditioning on the history of past events. However, most existing MTPP models rely on channel-mixing strategies that encode information from different event types into a single, fixed-size latent representation. This entanglement can obscure type-specific dynamics, leading to performance degradation and increased risk of overfitting. In this work, we introduce ITPP, a novel channel-independent architecture for MTPP modeling that decouples event type information using an encoder-decoder framework with an ODE-based backbone. Central to ITPP is a type-aware inverted self-attention mechanism, designed to explicitly model inter-channel correlations among heterogeneous event types. This architecture enhances effectiveness and robustness while reducing overfitting. Comprehensive experiments on multiple real-world and synthetic datasets demonstrate that ITPP consistently outperforms state-of-the-art MTPP models in both predictive accuracy and generalization. 

---
# MiVID: Multi-Strategic Self-Supervision for Video Frame Interpolation using Diffusion Model 

**Authors**: Priyansh Srivastava, Romit Chatterjee, Abir Sen, Aradhana Behura, Ratnakar Dash  

**Link**: [PDF](https://arxiv.org/pdf/2511.06019)  

**Abstract**: Video Frame Interpolation (VFI) remains a cornerstone in video enhancement, enabling temporal upscaling for tasks like slow-motion rendering, frame rate conversion, and video restoration. While classical methods rely on optical flow and learning-based models assume access to dense ground-truth, both struggle with occlusions, domain shifts, and ambiguous motion. This article introduces MiVID, a lightweight, self-supervised, diffusion-based framework for video interpolation. Our model eliminates the need for explicit motion estimation by combining a 3D U-Net backbone with transformer-style temporal attention, trained under a hybrid masking regime that simulates occlusions and motion uncertainty. The use of cosine-based progressive masking and adaptive loss scheduling allows our network to learn robust spatiotemporal representations without any high-frame-rate supervision. Our framework is evaluated on UCF101-7 and DAVIS-7 datasets. MiVID is trained entirely on CPU using the datasets and 9-frame video segments, making it a low-resource yet highly effective pipeline. Despite these constraints, our model achieves optimal results at just 50 epochs, competitive with several supervised this http URL work demonstrates the power of self-supervised diffusion priors for temporally coherent frame synthesis and provides a scalable path toward accessible and generalizable VFI systems. 

---
# One-Shot Knowledge Transfer for Scalable Person Re-Identification 

**Authors**: Longhua Li, Lei Qi, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06016)  

**Abstract**: Edge computing in person re-identification (ReID) is crucial for reducing the load on central cloud servers and ensuring user privacy. Conventional compression methods for obtaining compact models require computations for each individual student model. When multiple models of varying sizes are needed to accommodate different resource conditions, this leads to repetitive and cumbersome computations. To address this challenge, we propose a novel knowledge inheritance approach named OSKT (One-Shot Knowledge Transfer), which consolidates the knowledge of the teacher model into an intermediate carrier called a weight chain. When a downstream scenario demands a model that meets specific resource constraints, this weight chain can be expanded to the target model size without additional computation. OSKT significantly outperforms state-of-the-art compression methods, with the added advantage of one-time knowledge transfer that eliminates the need for frequent computations for each target model. 

---
# MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference 

**Authors**: Myunghyun Rhee, Sookyung Choi, Euiseok Kim, Joonseop Sim, Youngpyo Joo, Hoshik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06010)  

**Abstract**: The escalating context length in Large Language Models (LLMs) creates a severe performance bottleneck around the Key-Value (KV) cache, whose memory-bound nature leads to significant GPU under-utilization. This paper introduces Mixture of Shared KV Attention (MoSKA), an architecture that addresses this challenge by exploiting the heterogeneity of context data. It differentiates between per-request unique and massively reused shared sequences. The core of MoSKA is a novel Shared KV Attention mechanism that transforms the attention on shared data from a series of memory-bound GEMV operations into a single, compute-bound GEMM by batching concurrent requests. This is supported by an MoE-inspired sparse attention strategy that prunes the search space and a tailored Disaggregated Infrastructure that specializes hardware for unique and shared data. This comprehensive approach demonstrates a throughput increase of up to 538.7x over baselines in workloads with high context sharing, offering a clear architectural path toward scalable LLM inference. 

---
# Exploring Category-level Articulated Object Pose Tracking on SE(3) Manifolds 

**Authors**: Xianhui Meng, Yukang Huo, Li Zhang, Liu Liu, Haonan Jiang, Yan Zhong, Pingrui Zhang, Cewu Lu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05996)  

**Abstract**: Articulated objects are prevalent in daily life and robotic manipulation tasks. However, compared to rigid objects, pose tracking for articulated objects remains an underexplored problem due to their inherent kinematic constraints. To address these challenges, this work proposes a novel point-pair-based pose tracking framework, termed \textbf{PPF-Tracker}. The proposed framework first performs quasi-canonicalization of point clouds in the SE(3) Lie group space, and then models articulated objects using Point Pair Features (PPF) to predict pose voting parameters by leveraging the invariance properties of SE(3). Finally, semantic information of joint axes is incorporated to impose unified kinematic constraints across all parts of the articulated object. PPF-Tracker is systematically evaluated on both synthetic datasets and real-world scenarios, demonstrating strong generalization across diverse and challenging environments. Experimental results highlight the effectiveness and robustness of PPF-Tracker in multi-frame pose tracking of articulated objects. We believe this work can foster advances in robotics, embodied intelligence, and augmented reality. Codes are available at this https URL. 

---
# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models 

**Authors**: Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.05993)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a predominant approach for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, causing premature convergence to suboptimal local minima and hinder further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To address this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our findings reveal that the number of off-policy updates, the diversity of training data, and the clipping thresholds in the optimization objective are critical factors influencing the entropy of LLMs trained with RLVR. Moreover, we theoretically and empirically demonstrate that tokens with positive advantages are the primary contributors to entropy collapse, and that model entropy can be effectively regulated by adjusting the relative loss weights of tokens with positive and negative advantages during training. 

---
# Ontology Learning and Knowledge Graph Construction: A Comparison of Approaches and Their Impact on RAG Performance 

**Authors**: Tiago da Cruz, Bernardo Tavares, Francisco Belo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05991)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems combine Large Language Models (LLMs) with external knowledge, and their performance depends heavily on how that knowledge is represented. This study investigates how different Knowledge Graph (KG) construction strategies influence RAG performance. We compare a variety of approaches: standard vector-based RAG, GraphRAG, and retrieval over KGs built from ontologies derived either from relational databases or textual corpora. Results show that ontology-guided KGs incorporating chunk information achieve competitive performance with state-of-the-art frameworks, substantially outperforming vector retrieval baselines. Moreover, the findings reveal that ontology-guided KGs built from relational databases perform competitively to ones built with ontologies extracted from text, with the benefit of offering a dual advantage: they require a one-time-only ontology learning process, substantially reducing LLM usage costs; and avoid the complexity of ontology merging inherent to text-based approaches. 

---
# Runtime Safety Monitoring of Deep Neural Networks for Perception: A Survey 

**Authors**: Albert Schotschneider, Svetlana Pavlitska, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2511.05982)  

**Abstract**: Deep neural networks (DNNs) are widely used in perception systems for safety-critical applications, such as autonomous driving and robotics. However, DNNs remain vulnerable to various safety concerns, including generalization errors, out-of-distribution (OOD) inputs, and adversarial attacks, which can lead to hazardous failures. This survey provides a comprehensive overview of runtime safety monitoring approaches, which operate in parallel to DNNs during inference to detect these safety concerns without modifying the DNN itself. We categorize existing methods into three main groups: Monitoring inputs, internal representations, and outputs. We analyze the state-of-the-art for each category, identify strengths and limitations, and map methods to the safety concerns they address. In addition, we highlight open challenges and future research directions. 

---
# Kunlun Anomaly Troubleshooter: Enabling Kernel-Level Anomaly Detection and Causal Reasoning for Large Model Distributed Inference 

**Authors**: Yuyang Liu, Jingjing Cai, Jiayi Ren, Peng Zhou, Danyang Zhang, Yin Du, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.05978)  

**Abstract**: Anomaly troubleshooting for large model distributed inference (LMDI) remains a critical challenge. Resolving anomalies such as inference performance degradation or latency jitter in distributed system demands significant manual efforts from domain experts, resulting in extremely time-consuming diagnosis processes with relatively low accuracy. In this paper, we introduce Kunlun Anomaly Troubleshooter (KAT), the first anomaly troubleshooting framework tailored for LMDI. KAT addresses this problem through two core innovations. First, KAT exploits the synchronicity and consistency of GPU workers, innovatively leverages function trace data to precisely detect kernel-level anomalies and associated hardware components at nanosecond resolution. Second, KAT integrates these detection results into a domain-adapted LLM, delivering systematic causal reasoning and natural language interpretation of complex anomaly symptoms. Evaluations conducted in Alibaba Cloud Service production environment indicate that KAT achieves over 0.884 precision and 0.936 recall in anomaly detection, providing detail anomaly insights that significantly narrow down the diagnostic scope and improve both the efficiency and success rate of troubleshooting. 

---
# Interpretable Recognition of Cognitive Distortions in Natural Language Texts 

**Authors**: Anton Kolonin, Anna Arinicheva  

**Link**: [PDF](https://arxiv.org/pdf/2511.05969)  

**Abstract**: We propose a new approach to multi-factor classification of natural language texts based on weighted structured patterns such as N-grams, taking into account the heterarchical relationships between them, applied to solve such a socially impactful problem as the automation of detection of specific cognitive distortions in psychological care, relying on an interpretable, robust and transparent artificial intelligence model. The proposed recognition and learning algorithms improve the current state of the art in this field. The improvement is tested on two publicly available datasets, with significant improvements over literature-known F1 scores for the task, with optimal hyper-parameters determined, having code and models available for future use by the community. 

---
# DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities 

**Authors**: Nagur Shareef Shaik, Teja Krishna Cherukuri, Adnan Masood, Dong Hye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.05968)  

**Abstract**: The integration of medical images with clinical context is essential for generating accurate and clinically interpretable radiology reports. However, current automated methods often rely on resource-heavy Large Language Models (LLMs) or static knowledge graphs and struggle with two fundamental challenges in real-world clinical data: (1) missing modalities, such as incomplete clinical context , and (2) feature entanglement, where mixed modality-specific and shared information leads to suboptimal fusion and clinically unfaithful hallucinated findings. To address these challenges, we propose the DiA-gnostic VLVAE, which achieves robust radiology reporting through Disentangled Alignment. Our framework is designed to be resilient to missing modalities by disentangling shared and modality-specific features using a Mixture-of-Experts (MoE) based Vision-Language Variational Autoencoder (VLVAE). A constrained optimization objective enforces orthogonality and alignment between these latent representations to prevent suboptimal fusion. A compact LLaMA-X decoder then uses these disentangled representations to generate reports efficiently. On the IU X-Ray and MIMIC-CXR datasets, DiA has achieved competetive BLEU@4 scores of 0.266 and 0.134, respectively. Experimental results show that the proposed method significantly outperforms state-of-the-art models. 

---
# Adapted Foundation Models for Breast MRI Triaging in Contrast-Enhanced and Non-Contrast Enhanced Protocols 

**Authors**: Tri-Thien Nguyen, Lorenz A. Kapsner, Tobias Hepp, Shirin Heidarikahkesh, Hannes Schreiter, Luise Brock, Dominika Skwierawska, Dominique Hadler, Julian Hossbach, Evelyn Wenkel, Sabine Ohlmeyer, Frederik B. Laun, Andrzej Liebert, Andreas Maier, Michael Uder, Sebastian Bickelhaupt  

**Link**: [PDF](https://arxiv.org/pdf/2511.05967)  

**Abstract**: Background: Magnetic resonance imaging (MRI) has high sensitivity for breast cancer detection, but interpretation is time-consuming. Artificial intelligence may aid in pre-screening. Purpose: To evaluate the DINOv2-based Medical Slice Transformer (MST) for ruling out significant findings (Breast Imaging Reporting and Data System [BI-RADS] >=4) in contrast-enhanced and non-contrast-enhanced abbreviated breast MRI. Materials and Methods: This institutional review board approved retrospective study included 1,847 single-breast MRI examinations (377 BI-RADS >=4) from an in-house dataset and 924 from an external validation dataset (Duke). Four abbreviated protocols were tested: T1-weighted early subtraction (T1sub), diffusion-weighted imaging with b=1500 s/mm2 (DWI1500), DWI1500+T2-weighted (T2w), and T1sub+T2w. Performance was assessed at 90%, 95%, and 97.5% sensitivity using five-fold cross-validation and area under the receiver operating characteristic curve (AUC) analysis. AUC differences were compared with the DeLong test. False negatives were characterized, and attention maps of true positives were rated in the external dataset. Results: A total of 1,448 female patients (mean age, 49 +/- 12 years) were included. T1sub+T2w achieved an AUC of 0.77 +/- 0.04; DWI1500+T2w, 0.74 +/- 0.04 (p=0.15). At 97.5% sensitivity, T1sub+T2w had the highest specificity (19% +/- 7%), followed by DWI1500+T2w (17% +/- 11%). Missed lesions had a mean diameter <10 mm at 95% and 97.5% thresholds for both T1sub and DWI1500, predominantly non-mass enhancements. External validation yielded an AUC of 0.77, with 88% of attention maps rated good or moderate. Conclusion: At 97.5% sensitivity, the MST framework correctly triaged cases without BI-RADS >=4, achieving 19% specificity for contrast-enhanced and 17% for non-contrast-enhanced MRI. Further research is warranted before clinical implementation. 

---
# Adaptive Agent Selection and Interaction Network for Image-to-point cloud Registration 

**Authors**: Zhixin Cheng, Xiaotian Yin, Jiacheng Deng, Bohao Liao, Yujia Chen, Xu Zhou, Baoqun Yin, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05965)  

**Abstract**: Typical detection-free methods for image-to-point cloud registration leverage transformer-based architectures to aggregate cross-modal features and establish correspondences. However, they often struggle under challenging conditions, where noise disrupts similarity computation and leads to incorrect correspondences. Moreover, without dedicated designs, it remains difficult to effectively select informative and correlated representations across modalities, thereby limiting the robustness and accuracy of registration. To address these challenges, we propose a novel cross-modal registration framework composed of two key modules: the Iterative Agents Selection (IAS) module and the Reliable Agents Interaction (RAI) module. IAS enhances structural feature awareness with phase maps and employs reinforcement learning principles to efficiently select reliable agents. RAI then leverages these selected agents to guide cross-modal interactions, effectively reducing mismatches and improving overall robustness. Extensive experiments on the RGB-D Scenes v2 and 7-Scenes benchmarks demonstrate that our method consistently achieves state-of-the-art performance. 

---
# A PDE Perspective on Generative Diffusion Models 

**Authors**: Kang Liu, Enrique Zuazua  

**Link**: [PDF](https://arxiv.org/pdf/2511.05940)  

**Abstract**: Score-based diffusion models have emerged as a powerful class of generative methods, achieving state-of-the-art performance across diverse domains. Despite their empirical success, the mathematical foundations of those models remain only partially understood, particularly regarding the stability and consistency of the underlying stochastic and partial differential equations governing their dynamics.
In this work, we develop a rigorous partial differential equation (PDE) framework for score-based diffusion processes. Building on the Li--Yau differential inequality for the heat flow, we prove well-posedness and derive sharp $L^p$-stability estimates for the associated score-based Fokker--Planck dynamics, providing a mathematically consistent description of their temporal evolution. Through entropy stability methods, we further show that the reverse-time dynamics of diffusion models concentrate on the data manifold for compactly supported data distributions and a broad class of initialization schemes, with a concentration rate of order $\sqrt{t}$ as $t \to 0$.
These results yield a theoretical guarantee that, under exact score guidance, diffusion trajectories return to the data manifold while preserving imitation fidelity. Our findings also provide practical insights for designing diffusion models, including principled criteria for score-function construction, loss formulation, and stopping-time selection. Altogether, this framework provides a quantitative understanding of the trade-off between generative capacity and imitation fidelity, bridging rigorous analysis and model design within a unified mathematical perspective. 

---
# 10 Open Challenges Steering the Future of Vision-Language-Action Models 

**Authors**: Soujanya Poria, Navonil Majumder, Chia-Yu Hung, Amir Ali Bagherzadeh, Chuan Li, Kenneth Kwok, Ziwei Wang, Cheston Tan, Jiajun Wu, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05936)  

**Abstract**: Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability. 

---
# Reinforcement Learning Improves Traversal of Hierarchical Knowledge in LLMs 

**Authors**: Renfei Zhang, Manasa Kaniselvan, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2511.05933)  

**Abstract**: Reinforcement learning (RL) is often credited with improving language model reasoning and generalization at the expense of degrading memorized knowledge. We challenge this narrative by observing that RL-enhanced models consistently outperform their base and supervised fine-tuned (SFT) counterparts on pure knowledge recall tasks, particularly those requiring traversal of hierarchical, structured knowledge (e.g., medical codes). We hypothesize these gains stem not from newly acquired data, but from improved procedural skills in navigating and searching existing knowledge hierarchies within the model parameters. To support this hypothesis, we show that structured prompting, which explicitly guides SFTed models through hierarchical traversal, recovers most of the performance gap (reducing 24pp to 7pp on MedConceptsQA for DeepSeek-V3/R1). We further find that while prompting improves final-answer accuracy, RL-enhanced models retain superior ability to recall correct procedural paths on deep-retrieval tasks. Finally our layer-wise internal activation analysis reveals that while factual representations (e.g., activations for the statement "code 57.95 refers to urinary infection") maintain high cosine similarity between SFT and RL models, query representations (e.g., "what is code 57.95") diverge noticeably, indicating that RL primarily transforms how models traverse knowledge rather than the knowledge representation itself. 

---
# The Future of AI in the GCC Post-NPM Landscape: A Comparative Analysis of Kuwait and the UAE 

**Authors**: Mohammad Rashed Albous, Bedour Alboloushi, Arnaud Lacheret  

**Link**: [PDF](https://arxiv.org/pdf/2511.05932)  

**Abstract**: Comparative evidence on how Gulf Cooperation Council (GCC) states turn artificial intelligence (AI) ambitions into post--New Public Management (post-NPM) outcomes is scarce because most studies examine Western democracies. We analyze constitutional, collective-choice, and operational rules shaping AI uptake in two contrasting GCC members, the United Arab Emirates (UAE) and Kuwait, and whether they foster citizen centricity, collaborative governance, and public value creation. Anchored in Ostrom's Institutional Analysis and Development framework, the study combines a most similar/most different systems design with multiple sources: 62 public documents from 2018--2025, embedded UAE cases (Smart Dubai and MBZUAI), and 39 interviews with officials conducted Aug 2024--May 2025. Dual coding and process tracing connect rule configurations to AI performance. Cross-case analysis identifies four reinforcing mechanisms behind divergent trajectories. In the UAE, concentrated authority, credible sanctions, pro-innovation narratives, and flexible reinvestment rules scale pilots into hundreds of services and sizable recycled savings. In Kuwait, dispersed veto points, exhortative sanctions, cautious discourse, and lapsed AI budgets confine initiatives to pilot mode despite equivalent fiscal resources. The findings refine institutional theory by showing that vertical rule coherence, not wealth, determines AI's public-value yield, and temper post-NPM optimism by revealing that efficiency metrics serve societal goals only when backed by enforceable safeguards. To curb ethics washing and test transferability beyond the GCC, future work should track rule diffusion over time, develop blended legitimacy--efficiency scorecards, and examine how narrative framing shapes citizen consent for data sharing. 

---
# CoMA: Complementary Masking and Hierarchical Dynamic Multi-Window Self-Attention in a Unified Pre-training Framework 

**Authors**: Jiaxuan Li, Qing Xu, Xiangjian He, Ziyu Liu, Chang Xing, Zhen Chen, Daokun Zhang, Rong Qu, Chang Wen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05929)  

**Abstract**: Masked Autoencoders (MAE) achieve self-supervised learning of image representations by randomly removing a portion of visual tokens and reconstructing the original image as a pretext task, thereby significantly enhancing pretraining efficiency and yielding excellent adaptability across downstream tasks. However, MAE and other MAE-style paradigms that adopt random masking generally require more pre-training epochs to maintain adaptability. Meanwhile, ViT in MAE suffers from inefficient parameter use due to fixed spatial resolution across layers. To overcome these limitations, we propose the Complementary Masked Autoencoders (CoMA), which employ a complementary masking strategy to ensure uniform sampling across all pixels, thereby improving effective learning of all features and enhancing the model's adaptability. Furthermore, we introduce DyViT, a hierarchical vision transformer that employs a Dynamic Multi-Window Self-Attention (DM-MSA), significantly reducing the parameters and FLOPs while improving fine-grained feature learning. Pre-trained on ImageNet-1K with CoMA, DyViT matches the downstream performance of MAE using only 12% of the pre-training epochs, demonstrating more effective learning. It also attains a 10% reduction in pre-training time per epoch, further underscoring its superior pre-training efficiency. 

---
# Artificial intelligence and the Gulf Cooperation Council workforce adapting to the future of work 

**Authors**: Mohammad Rashed Albous, Melodena Stephens, Odeh Rashed Al-Jayyousi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05927)  

**Abstract**: The rapid expansion of artificial intelligence (AI) in the Gulf Cooperation Council (GCC) raises a central question: are investments in compute infrastructure matched by an equally robust build-out of skills, incentives, and governance? Grounded in socio-technical systems (STS) theory, this mixed-methods study audits workforce preparedness across Kingdom of Saudi Arabia (KSA), the United Arab Emirates (UAE), Qatar, Kuwait, Bahrain, and Oman. We combine term frequency--inverse document frequency (TF--IDF) analysis of six national AI strategies (NASs), an inventory of 47 publicly disclosed AI initiatives (January 2017--April 2025), paired case studies, the Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) and the Saudi Data & Artificial Intelligence Authority (SDAIA) Academy, and a scenario matrix linking oil-revenue slack (technical capacity) to regulatory coherence (social alignment). Across the corpus, 34/47 initiatives (0.72; 95% Wilson CI 0.58--0.83) exhibit joint social--technical design; country-level indices span 0.57--0.90 (small n; intervals overlap). Scenario results suggest that, under our modeled conditions, regulatory convergence plausibly binds outcomes more than fiscal capacity: fragmented rules can offset high oil revenues, while harmonized standards help preserve progress under austerity. We also identify an emerging two-track talent system, research elites versus rapidly trained practitioners, that risks labor-market bifurcation without bridging mechanisms. By extending STS inquiry to oil-rich, state-led economies, the study refines theory and sets a research agenda focused on longitudinal coupling metrics, ethnographies of coordination, and outcome-based performance indicators. 

---
# IDALC: A Semi-Supervised Framework for Intent Detection and Active Learning based Correction 

**Authors**: Ankan Mullick, Sukannya Purkayastha, Saransh Sharma, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2511.05921)  

**Abstract**: Voice-controlled dialog systems have become immensely popular due to their ability to perform a wide range of actions in response to diverse user queries. These agents possess a predefined set of skills or intents to fulfill specific user tasks. But every system has its own limitations. There are instances where, even for known intents, if any model exhibits low confidence, it results in rejection of utterances that necessitate manual annotation. Additionally, as time progresses, there may be a need to retrain these agents with new intents from the system-rejected queries to carry out additional tasks. Labeling all these emerging intents and rejected utterances over time is impractical, thus calling for an efficient mechanism to reduce annotation costs. In this paper, we introduce IDALC (Intent Detection and Active Learning based Correction), a semi-supervised framework designed to detect user intents and rectify system-rejected utterances while minimizing the need for human annotation. Empirical findings on various benchmark datasets demonstrate that our system surpasses baseline methods, achieving a 5-10% higher accuracy and a 4-8% improvement in macro-F1. Remarkably, we maintain the overall annotation cost at just 6-10% of the unlabelled data available to the system. The overall framework of IDALC is shown in Fig. 1 

---
# IoT-based Fresh Produce Supply Chain Under Uncertainty: An Adaptive Optimization Framework 

**Authors**: Chirag Seth, Mehrdad Pirnia, James H Bookbinder  

**Link**: [PDF](https://arxiv.org/pdf/2511.05920)  

**Abstract**: Fruits and vegetables form a vital component of the global economy; however, their distribution poses complex logistical challenges due to high perishability, supply fluctuations, strict quality and safety standards, and environmental sensitivity. In this paper, we propose an adaptive optimization model that accounts for delays, travel time, and associated temperature changes impacting produce shelf life, and compare it against traditional approaches such as Robust Optimization, Distributionally Robust Optimization, and Stochastic Programming. Additionally, we conduct a series of computational experiments using Internet of Things (IoT) sensor data to evaluate the performance of our proposed model. Our study demonstrates that the proposed adaptive model achieves a higher shelf life, extending it by over 18\% compared to traditional optimization models, by dynamically mitigating temperature deviations through a temperature feedback mechanism. The promising results demonstrate the potential of this approach to improve both the freshness and efficiency of logistics systems an aspect often neglected in previous works. 

---
# Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs 

**Authors**: Alina Fastowski, Bardh Prenkaj, Yuxiao Li, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2511.05919)  

**Abstract**: LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety. 

---
# NILC: Discovering New Intents with LLM-assisted Clustering 

**Authors**: Hongtao Wang, Renchi Yang, Wenqing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05913)  

**Abstract**: New intent discovery (NID) seeks to recognize both new and known intents from unlabeled user utterances, which finds prevalent use in practical dialogue systems. Existing works towards NID mainly adopt a cascaded architecture, wherein the first stage focuses on encoding the utterances into informative text embeddings beforehand, while the latter is to group similar embeddings into clusters (i.e., intents), typically by K-Means. However, such a cascaded pipeline fails to leverage the feedback from both steps for mutual refinement, and, meanwhile, the embedding-only clustering overlooks nuanced textual semantics, leading to suboptimal performance. To bridge this gap, this paper proposes NILC, a novel clustering framework specially catered for effective NID. Particularly, NILC follows an iterative workflow, in which clustering assignments are judiciously updated by carefully refining cluster centroids and text embeddings of uncertain utterances with the aid of large language models (LLMs). Specifically, NILC first taps into LLMs to create additional semantic centroids for clusters, thereby enriching the contextual semantics of the Euclidean centroids of embeddings. Moreover, LLMs are then harnessed to augment hard samples (ambiguous or terse utterances) identified from clusters via rewriting for subsequent cluster correction. Further, we inject supervision signals through non-trivial techniques seeding and soft must links for more accurate NID in the semi-supervised setting. Extensive experiments comparing NILC against multiple recent baselines under both unsupervised and semi-supervised settings showcase that NILC can achieve significant performance improvements over six benchmark datasets of diverse domains consistently. 

---
# The Imperfect Learner: Incorporating Developmental Trajectories in Memory-based Student Simulation 

**Authors**: Zhengyuan Liu, Stella Xin Yin, Bryan Chen Zhengyu Tan, Roy Ka-Wei Lee, Guimei Liu, Dion Hoe-Lian Goh, Wenya Wang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05903)  

**Abstract**: User simulation is important for developing and evaluating human-centered AI, yet current student simulation in educational applications has significant limitations. Existing approaches focus on single learning experiences and do not account for students' gradual knowledge construction and evolving skill sets. Moreover, large language models are optimized to produce direct and accurate responses, making it challenging to represent the incomplete understanding and developmental constraints that characterize real learners. In this paper, we introduce a novel framework for memory-based student simulation that incorporates developmental trajectories through a hierarchical memory mechanism with structured knowledge representation. The framework also integrates metacognitive processes and personality traits to enrich the individual learner profiling, through dynamical consolidation of both cognitive development and personal learning characteristics. In practice, we implement a curriculum-aligned simulator grounded on the Next Generation Science Standards. Experimental results show that our approach can effectively reflect the gradual nature of knowledge development and the characteristic difficulties students face, providing a more accurate representation of learning processes. 

---
# Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations 

**Authors**: Rui Yang, Matthew Yu Heng Wong, Huitao Li, Xin Li, Wentao Zhu, Jingchi Liao, Kunyu Yu, Jonathan Chong Kai Liew, Weihao Xuan, Yingjian Chen, Yuhe Ke, Jasmine Chiat Ling Ong, Douglas Teodoro, Chuan Hong, Daniel Shi Wei Ting, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05901)  

**Abstract**: The rapid growth of medical knowledge and increasing complexity of clinical practice pose challenges. In this context, large language models (LLMs) have demonstrated value; however, inherent limitations remain. Retrieval-augmented generation (RAG) technologies show potential to enhance their clinical applicability. This study reviewed RAG applications in medicine. We found that research primarily relied on publicly available data, with limited application in private data. For retrieval, approaches commonly relied on English-centric embedding models, while LLMs were mostly generic, with limited use of medical-specific LLMs. For evaluation, automated metrics evaluated generation quality and task performance, whereas human evaluation focused on accuracy, completeness, relevance, and fluency, with insufficient attention to bias and safety. RAG applications were concentrated on question answering, report generation, text summarization, and information extraction. Overall, medical RAG remains at an early stage, requiring advances in clinical validation, cross-linguistic adaptation, and support for low-resource settings to enable trustworthy and responsible global use. 

---
# GABFusion: Rethinking Feature Fusion for Low-Bit Quantization of Multi-Task Networks 

**Authors**: Zhaoyang Wang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05898)  

**Abstract**: Despite the effectiveness of quantization-aware training (QAT) in compressing deep neural networks, its performance on multi-task architectures often degrades significantly due to task-specific feature discrepancies and gradient conflicts. To address these challenges, we propose Gradient-Aware Balanced Feature Fusion (GABFusion), which dynamically balances gradient magnitudes and fuses task-specific features in a quantization-friendly manner. We further introduce Attention Distribution Alignment (ADA), a feature-level distillation strategy tailored for quantized models. Our method demonstrates strong generalization across network architectures and QAT algorithms, with theoretical guarantees on gradient bias reduction. Extensive experiments demonstrate that our strategy consistently enhances a variety of QAT methods across different network architectures and bit-widths. On PASCAL VOC and COCO datasets, the proposed approach achieves average mAP improvements of approximately 3.3% and 1.6%, respectively. When applied to YOLOv5 under 4-bit quantization, our method narrows the accuracy gap with the full-precision model to only 1.7% on VOC, showcasing its effectiveness in preserving performance under low-bit constraints. Notably, the proposed framework is modular, easy to integrate, and compatible with any existing QAT technique-enhancing the performance of quantized models without requiring modifications to the original network architecture. 

---
# A Remarkably Efficient Paradigm to Multimodal Large Language Models for Sequential Recommendation 

**Authors**: Qiyong Zhong, Jiajie Su, Ming Yang, Yunshan Ma, Xiaolin Zheng, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05885)  

**Abstract**: In this paper, we proposed Speeder, a remarkably efficient paradigm to multimodal large language models for sequential recommendation. Speeder introduces 3 key components: (1) Multimodal Representation Compression (MRC), which efficiently reduces redundancy in item descriptions; (2) Sequential Position Awareness Enhancement (SPAE), which strengthens the model's ability to capture complex sequential dependencies; (3) Modality-aware Progressive Optimization (MPO), which progressively integrates different modalities to improve the model's understanding and reduce cognitive biases. Through extensive experiments, Speeder demonstrates superior performance over baselines in terms of VHR@1 and computational efficiency. Specifically, Speeder achieved 250% of the training speed and 400% of the inference speed compared to the state-of-the-art MLLM-based SR models. Future work could focus on incorporating real-time feedback from real-world systems. 

---
# Physics-Informed Neural Networks for Real-Time Gas Crossover Prediction in PEM Electrolyzers: First Application with Multi-Membrane Validation 

**Authors**: Yong-Woon Kim, Chulung Kang, Yung-Cheol Byun  

**Link**: [PDF](https://arxiv.org/pdf/2511.05879)  

**Abstract**: Green hydrogen production via polymer electrolyte membrane (PEM) water electrolysis is pivotal for energy transition, yet hydrogen crossover through membranes threatens safety and economic viability-approaching explosive limits (4 mol% H$_2$ in O$_2$) while reducing Faradaic efficiency by 2.5%. Current physics-based models require extensive calibration and computational resources that preclude real-time implementation, while purely data-driven approaches fail to extrapolate beyond training conditions-critical for dynamic electrolyzer operation. Here we present the first application of physics-informed neural networks (PINNs) for hydrogen crossover prediction, integrating mass conservation, Fick's diffusion law, and Henry's solubility law within a compact architecture (17,793 parameters). Validated across six membranes under industrially relevant conditions (0.05-5.0 A/cm$^2$, 1-200 bar, 25-85°C), our PINN achieves exceptional accuracy (R$^2$ = 99.84%, RMSE = 0.0348%) with sub-millisecond inference times suitable for real-time control. Remarkably, the model maintains R$^2$ > 86% when predicting crossover at pressures 2.5x beyond training range-substantially outperforming pure neural networks (R$^2$ = 43.4%). The hardware-agnostic deployment, from desktop CPUs to edge devices (Raspberry Pi 4), enables distributed safety monitoring essential for gigawatt-scale installations. By bridging physical rigor and computational efficiency, this work establishes a new paradigm for real-time electrolyzer monitoring, accelerating deployment of safe, efficient green hydrogen infrastructure crucial for net-zero emissions targets. 

---
# Towards a Humanized Social-Media Ecosystem: AI-Augmented HCI Design Patterns for Safety, Agency & Well-Being 

**Authors**: Mohd Ruhul Ameen, Akif Islam  

**Link**: [PDF](https://arxiv.org/pdf/2511.05875)  

**Abstract**: Social platforms connect billions of people, yet their engagement-first algorithms often work on users rather than with them, amplifying stress, misinformation, and a loss of control. We propose Human-Layer AI (HL-AI)--user-owned, explainable intermediaries that sit in the browser between platform logic and the interface. HL-AI gives people practical, moment-to-moment control without requiring platform cooperation. We contribute a working Chrome/Edge prototype implementing five representative pattern frameworks--Context-Aware Post Rewriter, Post Integrity Meter, Granular Feed Curator, Micro-Withdrawal Agent, and Recovery Mode--alongside a unifying mathematical formulation balancing user utility, autonomy costs, and risk thresholds. Evaluation spans technical accuracy, usability, and behavioral outcomes. The result is a suite of humane controls that help users rewrite before harm, read with integrity cues, tune feeds with intention, pause compulsive loops, and seek shelter during harassment, all while preserving agency through explanations and override options. This prototype offers a practical path to retrofit today's feeds with safety, agency, and well-being, inviting rigorous cross-cultural user evaluation. 

---
# EndoIR: Degradation-Agnostic All-in-One Endoscopic Image Restoration via Noise-Aware Routing Diffusion 

**Authors**: Tong Chen, Xinyu Ma, Long Bai, Wenyang Wang, Sun Yue, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05873)  

**Abstract**: Endoscopic images often suffer from diverse and co-occurring degradations such as low lighting, smoke, and bleeding, which obscure critical clinical details. Existing restoration methods are typically task-specific and often require prior knowledge of the degradation type, limiting their robustness in real-world clinical use. We propose EndoIR, an all-in-one, degradation-agnostic diffusion-based framework that restores multiple degradation types using a single model. EndoIR introduces a Dual-Domain Prompter that extracts joint spatial-frequency features, coupled with an adaptive embedding that encodes both shared and task-specific cues as conditioning for denoising. To mitigate feature confusion in conventional concatenation-based conditioning, we design a Dual-Stream Diffusion architecture that processes clean and degraded inputs separately, with a Rectified Fusion Block integrating them in a structured, degradation-aware manner. Furthermore, Noise-Aware Routing Block improves efficiency by dynamically selecting only noise-relevant features during denoising. Experiments on SegSTRONG-C and CEC datasets demonstrate that EndoIR achieves state-of-the-art performance across multiple degradation scenarios while using fewer parameters than strong baselines, and downstream segmentation experiments confirm its clinical utility. 

---
# Adaptation and Fine-tuning with TabPFN for Travelling Salesman Problem 

**Authors**: Nguyen Gia Hien Vu, Yifan Tang, Rey Lim, Yifan Yang, Hang Ma, Ke Wang, G. Gary Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05872)  

**Abstract**: Tabular Prior-Data Fitted Network (TabPFN) is a foundation model designed for small to medium-sized tabular data, which has attracted much attention recently. This paper investigates the application of TabPFN in Combinatorial Optimization (CO) problems. The aim is to lessen challenges in time and data-intensive training requirements often observed in using traditional methods including exact and heuristic algorithms, Machine Learning (ML)-based models, to solve CO problems. Proposing possibly the first ever application of TabPFN for such a purpose, we adapt and fine-tune the TabPFN model to solve the Travelling Salesman Problem (TSP), one of the most well-known CO problems. Specifically, we adopt the node-based approach and the node-predicting adaptation strategy to construct the entire TSP route. Our evaluation with varying instance sizes confirms that TabPFN requires minimal training, adapts to TSP using a single sample, performs better generalization across varying TSP instance sizes, and reduces performance degradation. Furthermore, the training process with adaptation and fine-tuning is completed within minutes. The methodology leads to strong solution quality even without post-processing and achieves performance comparable to other models with post-processing refinement. Our findings suggest that the TabPFN model is a promising approach to solve structured and CO problems efficiently under training resource constraints and rapid deployment requirements. 

---
# CGCE: Classifier-Guided Concept Erasure in Generative Models 

**Authors**: Viet Nguyen, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2511.05865)  

**Abstract**: Recent advancements in large-scale generative models have enabled the creation of high-quality images and videos, but have also raised significant safety concerns regarding the generation of unsafe content. To mitigate this, concept erasure methods have been developed to remove undesirable concepts from pre-trained models. However, existing methods remain vulnerable to adversarial attacks that can regenerate the erased content. Moreover, achieving robust erasure often degrades the model's generative quality for safe, unrelated concepts, creating a difficult trade-off between safety and performance. To address this challenge, we introduce Classifier-Guided Concept Erasure (CGCE), an efficient plug-and-play framework that provides robust concept erasure for diverse generative models without altering their original weights. CGCE uses a lightweight classifier operating on text embeddings to first detect and then refine prompts containing undesired concepts. This approach is highly scalable, allowing for multi-concept erasure by aggregating guidance from several classifiers. By modifying only unsafe embeddings at inference time, our method prevents harmful content generation while preserving the model's original quality on benign prompts. Extensive experiments show that CGCE achieves state-of-the-art robustness against a wide range of red-teaming attacks. Our approach also maintains high generative utility, demonstrating a superior balance between safety and performance. We showcase the versatility of CGCE through its successful application to various modern T2I and T2V models, establishing it as a practical and effective solution for safe generative AI. 

---
# EMOD: A Unified EEG Emotion Representation Framework Leveraging V-A Guided Contrastive Learning 

**Authors**: Yuning Chen, Sha Zhao, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05863)  

**Abstract**: Emotion recognition from EEG signals is essential for affective computing and has been widely explored using deep learning. While recent deep learning approaches have achieved strong performance on single EEG emotion datasets, their generalization across datasets remains limited due to the heterogeneity in annotation schemes and data formats. Existing models typically require dataset-specific architectures tailored to input structure and lack semantic alignment across diverse emotion labels. To address these challenges, we propose EMOD: A Unified EEG Emotion Representation Framework Leveraging Valence-Arousal (V-A) Guided Contrastive Learning. EMOD learns transferable and emotion-aware representations from heterogeneous datasets by bridging both semantic and structural gaps. Specifically, we project discrete and continuous emotion labels into a unified V-A space and formulate a soft-weighted supervised contrastive loss that encourages emotionally similar samples to cluster in the latent space. To accommodate variable EEG formats, EMOD employs a flexible backbone comprising a Triple-Domain Encoder followed by a Spatial-Temporal Transformer, enabling robust extraction and integration of temporal, spectral, and spatial features. We pretrain EMOD on eight public EEG datasets and evaluate its performance on three benchmark datasets. Experimental results show that EMOD achieves state-of-the-art performance, demonstrating strong adaptability and generalization across diverse EEG-based emotion recognition scenarios. 

---
# Predicting the Future by Retrieving the Past 

**Authors**: Dazhao Du, Tao Han, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05859)  

**Abstract**: Deep learning models such as MLP, Transformer, and TCN have achieved remarkable success in univariate time series forecasting, typically relying on sliding window samples from historical data for training. However, while these models implicitly compress historical information into their parameters during training, they are unable to explicitly and dynamically access this global knowledge during inference, relying only on the local context within the lookback window. This results in an underutilization of rich patterns from the global history. To bridge this gap, we propose Predicting the Future by Retrieving the Past (PFRP), a novel approach that explicitly integrates global historical data to enhance forecasting accuracy. Specifically, we construct a Global Memory Bank (GMB) to effectively store and manage global historical patterns. A retrieval mechanism is then employed to extract similar patterns from the GMB, enabling the generation of global predictions. By adaptively combining these global predictions with the outputs of any local prediction model, PFRP produces more accurate and interpretable forecasts. Extensive experiments conducted on seven real-world datasets demonstrate that PFRP significantly enhances the average performance of advanced univariate forecasting models by 8.4\%. Codes can be found in this https URL. 

---
# Quantifying Edits Decay in Fine-tuned LLMs 

**Authors**: Yinjie Cheng, Paul Youssef, Christin Seifert, Jörg Schlötterer, Zhixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.05852)  

**Abstract**: Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits as shown in Figure 1, current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edits decay after fine-tuning, investigating how fine-tuning affects knowledge editing. We evaluate two state-of-the-art editing methods (MEMIT, AlphaEdit) and three fine-tuning approaches (full-parameter, LoRA, DoRA) across five LLMs and three datasets, yielding 232 experimental configurations. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we propose selective-layer fine-tuning and find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline. 

---
# Retrieval Quality at Context Limit 

**Authors**: Max McKinnon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05850)  

**Abstract**: The ability of large language models (LLMs) to recall and retrieve information from long contexts is critical for many real-world applications. Prior work (Liu et al., 2023) reported that LLMs suffer significant drops in retrieval accuracy for facts placed in the middle of large contexts, an effect known as "Lost in the Middle" (LITM). We find the model Gemini 2.5 Flash can answer needle-in-a-haystack questions with great accuracy regardless of document position including when the document is nearly at the input context limit. Our results suggest that the "Lost in the Middle" effect is not present for simple factoid Q\&A in Gemini 2.5 Flash, indicating substantial improvements in long-context retrieval. 

---
# EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph 

**Authors**: Nan Jiang, Ziyi Wang, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.05849)  

**Abstract**: Symbolic regression seeks to uncover physical laws from experimental data by searching for closed-form expressions, which is an important task in AI-driven scientific discovery. Yet the exponential growth of the search space of expression renders the task computationally challenging. A promising yet underexplored direction for reducing the effective search space and accelerating training lies in symbolic equivalence: many expressions, although syntactically different, define the same function -- for example, $\log(x_1^2x_2^3)$, $\log(x_1^2)+\log(x_2^3)$, and $2\log(x_1)+3\log(x_2)$. Existing algorithms treat such variants as distinct outputs, leading to redundant exploration and slow learning. We introduce EGG-SR, a unified framework that integrates equality graphs (e-graphs) into diverse symbolic regression algorithms, including Monte Carlo Tree Search (MCTS), deep reinforcement learning (DRL), and large language models (LLMs). EGG-SR compactly represents equivalent expressions through the proposed EGG module, enabling more efficient learning by: (1) pruning redundant subtree exploration in EGG-MCTS, (2) aggregating rewards across equivalence classes in EGG-DRL, and (3) enriching feedback prompts in EGG-LLM. Under mild assumptions, we show that embedding e-graphs tightens the regret bound of MCTS and reduces the variance of the DRL gradient estimator. Empirically, EGG-SR consistently enhances multiple baselines across challenging benchmarks, discovering equations with lower normalized mean squared error than state-of-the-art methods. Code implementation is available at: this https URL. 

---
# Enhancing Diffusion Model Guidance through Calibration and Regularization 

**Authors**: Seyed Alireza Javid, Amirhossein Bagheri, Nuria González-Prelcic  

**Link**: [PDF](https://arxiv.org/pdf/2511.05844)  

**Abstract**: Classifier-guided diffusion models have emerged as a powerful approach for conditional image generation, but they suffer from overconfident predictions during early denoising steps, causing the guidance gradient to vanish. This paper introduces two complementary contributions to address this issue. First, we propose a differentiable calibration objective based on the Smooth Expected Calibration Error (Smooth ECE), which improves classifier calibration with minimal fine-tuning and yields measurable improvements in Frechet Inception Distance (FID). Second, we develop enhanced sampling guidance methods that operate on off-the-shelf classifiers without requiring retraining. These include tilted sampling with batch-level reweighting, adaptive entropy-regularized sampling to preserve diversity, and a novel f-divergence-based sampling strategy that strengthens class-consistent guidance while maintaining mode coverage. Experiments on ImageNet 128x128 demonstrate that our divergence-regularized guidance achieves an FID of 2.13 using a ResNet-101 classifier, improving upon existing classifier-guided diffusion methods while requiring no diffusion model retraining. The results show that principled calibration and divergence-aware sampling provide practical and effective improvements for classifier-guided diffusion. 

---
# Understanding Cross Task Generalization in Handwriting-Based Alzheimer's Screening via Vision Language Adaptation 

**Authors**: Changqing Gong, Huafeng Qin, Mounim A. El-Yacoubi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05841)  

**Abstract**: Alzheimer's disease is a prevalent neurodegenerative disorder for which early detection is critical. Handwriting-often disrupted in prodromal AD-provides a non-invasive and cost-effective window into subtle motor and cognitive decline. Existing handwriting-based AD studies, mostly relying on online trajectories and hand-crafted features, have not systematically examined how task type influences diagnostic performance and cross-task generalization. Meanwhile, large-scale vision language models have demonstrated remarkable zero or few-shot anomaly detection in natural images and strong adaptability across medical modalities such as chest X-ray and brain MRI. However, handwriting-based disease detection remains largely unexplored within this paradigm. To close this gap, we introduce a lightweight Cross-Layer Fusion Adapter framework that repurposes CLIP for handwriting-based AD screening. CLFA implants multi-level fusion adapters within the visual encoder to progressively align representations toward handwriting-specific medical cues, enabling prompt-free and efficient zero-shot inference. Using this framework, we systematically investigate cross-task generalization-training on a specific handwriting task and evaluating on unseen ones-to reveal which task types and writing patterns most effectively discriminate AD. Extensive analyses further highlight characteristic stroke patterns and task-level factors that contribute to early AD identification, offering both diagnostic insights and a benchmark for handwriting-based cognitive assessment. 

---
# Hilbert-Guided Block-Sparse Local Attention 

**Authors**: Yunge Li, Lanyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05832)  

**Abstract**: The quadratic compute and memory costs of global self-attention severely limit its use in high-resolution images. Local attention reduces complexity by restricting attention to neighborhoods. Block-sparse kernels can further improve the efficiency of local attention, but conventional local attention patterns often fail to deliver significant speedups because tokens within a window are not contiguous in the 1D sequence. This work proposes a novel method for constructing windows and neighborhoods based on the Hilbert curve. Image tokens are first reordered along a Hilbert curve, and windows and neighborhoods are then formed on the reordered 1D sequence. From a block-sparse perspective, this strategy significantly increases block sparsity and can be combined with existing block-sparse kernels to improve the efficiency of 2D local attention. Experiments show that the proposed Hilbert Window Attention and Hilbert Slide Attention can accelerate window attention and slide attention by about $4\times$ and $18\times$, respectively. To assess practicality, the strategy is instantiated as the Hilbert Window Transformer and the Hilbert Neighborhood Transformer, both of which achieve end-to-end speedups with minimal accuracy loss. Overall, combining Hilbert-guided local attention with block-sparse kernels offers a general and practical approach to enhancing the efficiency of 2D local attention for images. The code is available at this https URL. 

---
# Policy Gradient-Based EMT-in-the-Loop Learning to Mitigate Sub-Synchronous Control Interactions 

**Authors**: Sayak Mukherjee, Ramij R. Hossain, Kaustav Chatterjee, Sameer Nekkalapu, Marcelo Elizondo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05822)  

**Abstract**: This paper explores the development of learning-based tunable control gains using EMT-in-the-loop simulation framework (e.g., PSCAD interfaced with Python-based learning modules) to address critical sub-synchronous oscillations. Since sub-synchronous control interactions (SSCI) arise from the mis-tuning of control gains under specific grid configurations, effective mitigation strategies require adaptive re-tuning of these gains. Such adaptiveness can be achieved by employing a closed-loop, learning-based framework that considers the grid conditions responsible for such sub-synchronous oscillations. This paper addresses this need by adopting methodologies inspired by Markov decision process (MDP) based reinforcement learning (RL), with a particular emphasis on simpler deep policy gradient methods with additional SSCI-specific signal processing modules such as down-sampling, bandpass filtering, and oscillation energy dependent reward computations. Our experimentation in a real-world event setting demonstrates that the deep policy gradient based trained policy can adaptively compute gain settings in response to varying grid conditions and optimally suppress control interaction-induced oscillations. 

---
# WAR-Re: Web API Recommendation with Semantic Reasoning 

**Authors**: Zishuo Xu, Dezhong Yao, Yao Wan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05820)  

**Abstract**: With the development of cloud computing, the number of Web APIs has increased dramatically, further intensifying the demand for efficient Web API recommendation. Despite the demonstrated success of previous Web API recommendation solutions, two critical challenges persist: 1) a fixed top-N recommendation that cannot accommodate the varying API cardinality requirements of different mashups, and 2) these methods output only ranked API lists without accompanying reasons, depriving users of understanding the recommendation. To address these challenges, we propose WAR-Re, an LLM-based model for Web API recommendation with semantic reasoning for justification. WAR-Re leverages special start and stop tokens to handle the first challenge and uses two-stage training: supervised fine-tuning and reinforcement learning via Group Relative Policy Optimization (GRPO) to enhance the model's ability in both tasks. Comprehensive experimental evaluations on the ProgrammableWeb dataset demonstrate that WAR-Re achieves a gain of up to 21.59\% over the state-of-the-art baseline model in recommendation accuracy, while consistently producing high-quality semantic reasons for recommendations. 

---
# In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading 

**Authors**: Shuning Lin, Yifan He, Yitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05814)  

**Abstract**: In today's landscape, Mixture of Experts (MoE) is a crucial architecture that has been used by many of the most advanced models. One of the major challenges of MoE models is that they usually require much more memory than their dense counterparts due to their unique architecture, and hence are harder to deploy in environments with limited GPU memory, such as edge devices. MoE offloading is a promising technique proposed to overcome this challenge, especially if it is enhanced with caching and pre-fetching, but prior work stopped at suboptimal caching algorithm and offered limited insights. In this work, we study MoE offloading in depth and make the following contributions: 1. We analyze the expert activation and LRU caching behavior in detail and provide traces. 2. We propose LFU caching optimization based on our analysis and obtain strong improvements from LRU. 3. We implement and experiment speculative expert pre-fetching, providing detailed trace showing its huge potential . 4. In addition, our study extensively covers the behavior of the MoE architecture itself, offering information on the characteristic of the gating network and experts. This can inspire future work on the interpretation of MoE models and the development of pruning techniques for MoE architecture with minimal performance loss. 

---
# MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling 

**Authors**: Yu Zhang, Hui-Ling Zhen, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05811)  

**Abstract**: Training large language models with FP8 formats offers significant efficiency gains. However, the reduced numerical precision of FP8 poses challenges for stable and accurate training. Current frameworks preserve training performance using mixed-granularity quantization, i.e., applying per-group quantization for activations and per-tensor/block quantization for weights. While effective, per-group quantization requires scaling along the inner dimension of matrix multiplication, introducing additional dequantization overhead. Moreover, these frameworks often rely on just-in-time scaling to dynamically adjust scaling factors based on the current data distribution. However, this online quantization is inefficient for FP8 training, as it involves multiple memory reads and writes that negate the performance benefits of FP8. To overcome these limitations, we propose MOSS, a novel FP8 training framework that ensures both efficiency and numerical stability. MOSS introduces two key innovations: (1) a two-level microscaling strategy for quantizing sensitive activations, which balances precision and dequantization cost by combining a high-precision global scale with compact, power-of-two local scales; and (2) automatic scaling for weights in linear layers, which eliminates the need for costly max-reduction operations by predicting and adjusting scaling factors during training. Leveraging these techniques, MOSS enables efficient FP8 training of a 7B parameter model, achieving performance comparable to the BF16 baseline while achieving up to 34% higher training throughput. 

---
# Measuring Model Performance in the Presence of an Intervention 

**Authors**: Winston Chen, Michael W. Sjoding, Jenna Wiens  

**Link**: [PDF](https://arxiv.org/pdf/2511.05805)  

**Abstract**: AI models are often evaluated based on their ability to predict the outcome of interest. However, in many AI for social impact applications, the presence of an intervention that affects the outcome can bias the evaluation. Randomized controlled trials (RCTs) randomly assign interventions, allowing data from the control group to be used for unbiased model evaluation. However, this approach is inefficient because it ignores data from the treatment group. Given the complexity and cost often associated with RCTs, making the most use of the data is essential. Thus, we investigate model evaluation strategies that leverage all data from an RCT. First, we theoretically quantify the estimation bias that arises from naïvely aggregating performance estimates from treatment and control groups, and derive the condition under which this bias leads to incorrect model selection. Leveraging these theoretical insights, we propose nuisance parameter weighting (NPW), an unbiased model evaluation approach that reweights data from the treatment group to mimic the distributions of samples that would or would not experience the outcome under no intervention. Using synthetic and real-world datasets, we demonstrate that our proposed evaluation approach consistently yields better model selection than the standard approach, which ignores data from the treatment group, across various intervention effect and sample size settings. Our contribution represents a meaningful step towards more efficient model evaluation in real-world contexts. 

---
# Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits 

**Authors**: Bo Xue, Yuanyu Wan, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05802)  

**Abstract**: In multi-objective decision-making with hierarchical preferences, lexicographic bandits provide a natural framework for optimizing multiple objectives in a prioritized order. In this setting, a learner repeatedly selects arms and observes reward vectors, aiming to maximize the reward for the highest-priority objective, then the next, and so on. While previous studies have primarily focused on regret minimization, this work bridges the gap between \textit{regret minimization} and \textit{best arm identification} under lexicographic preferences. We propose two elimination-based algorithms to address this joint objective. The first algorithm eliminates suboptimal arms sequentially, layer by layer, in accordance with the objective priorities, and achieves sample complexity and regret bounds comparable to those of the best single-objective algorithms. The second algorithm simultaneously leverages reward information from all objectives in each round, effectively exploiting cross-objective dependencies. Remarkably, it outperforms the known lower bound for the single-objective bandit problem, highlighting the benefit of cross-objective information sharing in the multi-objective setting. Empirical results further validate their superior performance over baselines. 

---
# When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins 

**Authors**: Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna  

**Link**: [PDF](https://arxiv.org/pdf/2511.05797)  

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), with prior work focusing on cutting-edge LLM applications like personal copilots. In contrast, simpler LLM applications, such as customer service chatbots, are widespread on the web, yet their security posture and exposure to such attacks remain poorly understood. These applications often rely on third-party chatbot plugins that act as intermediaries to commercial LLM APIs, offering non-expert website builders intuitive ways to customize chatbot behaviors. To bridge this gap, we present the first large-scale study of 17 third-party chatbot plugins used by over 10,000 public websites, uncovering previously unknown prompt injection risks in practice. First, 8 of these plugins (used by 8,000 websites) fail to enforce the integrity of the conversation history transmitted in network requests between the website visitor and the chatbot. This oversight amplifies the impact of direct prompt injection attacks by allowing adversaries to forge conversation histories (including fake system messages), boosting their ability to elicit unintended behavior (e.g., code generation) by 3 to 8x. Second, 15 plugins offer tools, such as web-scraping, to enrich the chatbot's context with website-specific content. However, these tools do not distinguish the website's trusted content (e.g., product descriptions) from untrusted, third-party content (e.g., customer reviews), introducing a risk of indirect prompt injection. Notably, we found that ~13% of e-commerce websites have already exposed their chatbots to third-party content. We systematically evaluate both vulnerabilities through controlled experiments grounded in real-world observations, focusing on factors such as system prompt design and the underlying LLM. Our findings show that many plugins adopt insecure practices that undermine the built-in LLM safeguards. 

---
# VLAD-Grasp: Zero-shot Grasp Detection via Vision-Language Models 

**Authors**: Manav Kulshrestha, S. Talha Bukhari, Damon Conover, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2511.05791)  

**Abstract**: Robotic grasping is a fundamental capability for autonomous manipulation; however, most existing methods rely on large-scale expert annotations and necessitate retraining to handle new objects. We present VLAD-Grasp, a Vision-Language model Assisted zero-shot approach for Detecting grasps. From a single RGB-D image, our method (1) prompts a large vision-language model to generate a goal image where a straight rod "impales" the object, representing an antipodal grasp, (2) predicts depth and segmentation to lift this generated image into 3D, and (3) aligns generated and observed object point clouds via principal component analysis and correspondence-free optimization to recover an executable grasp pose. Unlike prior work, our approach is training-free and does not rely on curated grasp datasets. Despite this, VLAD-Grasp achieves performance that is competitive with or superior to that of state-of-the-art supervised models on the Cornell and Jacquard datasets. We further demonstrate zero-shot generalization to novel real-world objects on a Franka Research 3 robot, highlighting vision-language foundation models as powerful priors for robotic manipulation. 

---
# SymLight: Exploring Interpretable and Deployable Symbolic Policies for Traffic Signal Control 

**Authors**: Xiao-Cheng Liao, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05790)  

**Abstract**: Deep Reinforcement Learning have achieved significant success in automatically devising effective traffic signal control (TSC) policies. Neural policies, however, tend to be over-parameterized and non-transparent, hindering their interpretability and deployability on resource-limited edge devices. This work presents SymLight, a priority function search framework based on Monte Carlo Tree Search (MCTS) for discovering inherently interpretable and deployable symbolic priority functions to serve as the TSC policies. The priority function, in particular, accepts traffic features as input and then outputs a priority for each traffic signal phase, which subsequently directs the phase transition. For effective search, we propose a concise yet expressive priority function representation. This helps mitigate the combinatorial explosion of the action space in MCTS. Additionally, a probabilistic structural rollout strategy is introduced to leverage structural patterns from previously discovered high-quality priority functions, guiding the rollout process. Our experiments on real-world datasets demonstrate SymLight's superior performance across a range of baselines. A key advantage is SymLight's ability to produce interpretable and deployable TSC policies while maintaining excellent performance. 

---
# DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning 

**Authors**: Yaxuan Wang, Chris Yuhao Liu, Quan Liu, Jinglong Pang, Wei Wei, Yujia Bao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05784)  

**Abstract**: Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios. 

---
# Sign language recognition from skeletal data using graph and recurrent neural networks 

**Authors**: B. Mederos, J. Mejía, A. Medina-Reyes, Y. Espinosa-Almeyda, J. D. Díaz-Roman, I. Rodríguez-Mederos, M. Mejía-Carreon, F. Gonzalez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2511.05772)  

**Abstract**: This work presents an approach for recognizing isolated sign language gestures using skeleton-based pose data extracted from video sequences. A Graph-GRU temporal network is proposed to model both spatial and temporal dependencies between frames, enabling accurate classification. The model is trained and evaluated on the AUTSL (Ankara university Turkish sign language) dataset, achieving high accuracy. Experimental results demonstrate the effectiveness of integrating graph-based spatial representations with temporal modeling, providing a scalable framework for sign language recognition. The results of this approach highlight the potential of pose-driven methods for sign language understanding. 

---
# Lived Experience in Dialogue: Co-designing Personalization in Large Language Models to Support Youth Mental Well-being 

**Authors**: Kathleen W. Guan, Sarthak Giri, Mohammed Amara, Bernard J. Jansen, Enrico Liscio, Milena Esherick, Mohammed Al Owayyed, Ausrine Ratkute, Gayane Sedrakyan, Mark de Reuver, Joao Fernando Ferreira Goncalves, Caroline A. Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2511.05769)  

**Abstract**: Youth increasingly turn to large language models (LLMs) for mental well-being support, yet current personalization in LLMs can overlook the heterogeneous lived experiences shaping their needs. We conducted a participatory study with youth, parents, and youth care workers (N=38), using co-created youth personas as scaffolds, to elicit community perspectives on how LLMs can facilitate more meaningful personalization to support youth mental well-being. Analysis identified three themes: person-centered contextualization responsive to momentary needs, explicit boundaries around scope and offline referral, and dialogic scaffolding for reflection and autonomy. We mapped these themes to persuasive design features for task suggestions, social facilitation, and system trustworthiness, and created corresponding dialogue extracts to guide LLM fine-tuning. Our findings demonstrate how lived experience can be operationalized to inform design features in LLMs, which can enhance the alignment of LLM-based interventions with the realities of youth and their communities, contributing to more effectively personalized digital well-being tools. 

---
# Language Generation: Complexity Barriers and Implications for Learning 

**Authors**: Marcelo Arenas, Pablo Barceló, Luis Cofré, Alexander Kozachinskiy  

**Link**: [PDF](https://arxiv.org/pdf/2511.05759)  

**Abstract**: Kleinberg and Mullainathan showed that, in principle, language generation is always possible: with sufficiently many positive examples, a learner can eventually produce sentences indistinguishable from those of a target language. However, the existence of such a guarantee does not speak to its practical feasibility. In this work, we show that even for simple and well-studied language families -- such as regular and context-free languages -- the number of examples required for successful generation can be extraordinarily large, and in some cases not bounded by any computable function. These results reveal a substantial gap between theoretical possibility and efficient learnability. They suggest that explaining the empirical success of modern language models requires a refined perspective -- one that takes into account structural properties of natural language that make effective generation possible in practice. 

---
# Beyond Redundancy: Diverse and Specialized Multi-Expert Sparse Autoencoder 

**Authors**: Zhen Xu, Zhen Tan, Song Wang, Kaidi Xu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05745)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting large language models (LLMs) by decomposing token activations into combinations of human-understandable features. While SAEs provide crucial insights into LLM explanations, their practical adoption faces a fundamental challenge: better interpretability demands that SAEs' hidden layers have high dimensionality to satisfy sparsity constraints, resulting in prohibitive training and inference costs. Recent Mixture of Experts (MoE) approaches attempt to address this by partitioning SAEs into narrower expert networks with gated activation, thereby reducing computation. In a well-designed MoE, each expert should focus on learning a distinct set of features. However, we identify a \textit{critical limitation} in MoE-SAE: Experts often fail to specialize, which means they frequently learn overlapping or identical features. To deal with it, we propose two key innovations: (1) Multiple Expert Activation that simultaneously engages semantically weighted expert subsets to encourage specialization, and (2) Feature Scaling that enhances diversity through adaptive high-frequency scaling. Experiments demonstrate a 24\% lower reconstruction error and a 99\% reduction in feature redundancy compared to existing MoE-SAE methods. This work bridges the interpretability-efficiency gap in LLM analysis, allowing transparent model inspection without compromising computational feasibility. 

---
# Compressing Chemistry Reveals Functional Groups 

**Authors**: Ruben Sharma, Ross D. King  

**Link**: [PDF](https://arxiv.org/pdf/2511.05728)  

**Abstract**: We introduce the first formal large-scale assessment of the utility of traditional chemical functional groups as used in chemical explanations. Our assessment employs a fundamental principle from computational learning theory: a good explanation of data should also compress the data. We introduce an unsupervised learning algorithm based on the Minimum Message Length (MML) principle that searches for substructures that compress around three million biologically relevant molecules. We demonstrate that the discovered substructures contain most human-curated functional groups as well as novel larger patterns with more specific functions. We also run our algorithm on 24 specific bioactivity prediction datasets to discover dataset-specific functional groups. Fingerprints constructed from dataset-specific functional groups are shown to significantly outperform other fingerprint representations, including the MACCS and Morgan fingerprint, when training ridge regression models on bioactivity regression tasks. 

---
# OckBench: Measuring the Efficiency of LLM Reasoning 

**Authors**: Zheng Du, Hao Kang, Song Han, Tushar Krishna, Ligeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05722)  

**Abstract**: Large language models such as GPT-4, Claude 3, and the Gemini series have improved automated reasoning and code generation. However, existing benchmarks mainly focus on accuracy and output quality, and they ignore an important factor: decoding token efficiency. In real systems, generating 10,000 tokens versus 100,000 tokens leads to large differences in latency, cost, and energy. In this work, we introduce OckBench, a model-agnostic and hardware-agnostic benchmark that evaluates both accuracy and token count for reasoning and coding tasks. Through experiments comparing multiple open- and closed-source models, we uncover that many models with comparable accuracy differ wildly in token consumption, revealing that efficiency variance is a neglected but significant axis of differentiation. We further demonstrate Pareto frontiers over the accuracy-efficiency plane and argue for an evaluation paradigm shift: we should no longer treat tokens as "free" to multiply. OckBench provides a unified platform for measuring, comparing, and guiding research in token-efficient reasoning. Our benchmarks are available at this https URL . 

---
# AdvisingWise: Supporting Academic Advising in Higher Educations Through a Human-in-the-Loop Multi-Agent Framework 

**Authors**: Wendan Jiang, Shiyuan Wang, Hiba Eltigani, Rukhshan Haroon, Abdullah Bin Faisal, Fahad Dogar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05706)  

**Abstract**: Academic advising is critical to student success in higher education, yet high student-to-advisor ratios limit advisors' capacity to provide timely support, particularly during peak periods. Recent advances in Large Language Models (LLMs) present opportunities to enhance the advising process. We present AdvisingWise, a multi-agent system that automates time-consuming tasks, such as information retrieval and response drafting, while preserving human oversight. AdvisingWise leverages authoritative institutional resources and adaptively prompts students about their academic backgrounds to generate reliable, personalized responses. All system responses undergo human advisor validation before delivery to students. We evaluate AdvisingWise through a mixed-methods approach: (1) expert evaluation on responses of 20 sample queries, (2) LLM-as-a-judge evaluation of the information retrieval strategy, and (3) a user study with 8 academic advisors to assess the system's practical utility. Our evaluation shows that AdvisingWise produces accurate, personalized responses. Advisors reported increasingly positive perceptions after using AdvisingWise, as their initial concerns about reliability and personalization diminished. We conclude by discussing the implications of human-AI synergy on the practice of academic advising. 

---
# Long Grounded Thoughts: Distilling Compositional Visual Reasoning Chains at Scale 

**Authors**: David Acuna, Chao-Han Huck Yang, Yuntian Deng, Jaehun Jung, Ximing Lu, Prithviraj Ammanabrolu, Hyunwoo Kim, Yuan-Hong Liao, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05705)  

**Abstract**: Recent progress in multimodal reasoning has been driven largely by undisclosed datasets and proprietary data synthesis recipes, leaving open questions about how to systematically build large-scale, vision-centric reasoning datasets, particularly for tasks that go beyond visual math. In this work, we introduce a new reasoning data generation framework spanning diverse skills and levels of complexity with over 1M high-quality synthetic vision-centric questions. The dataset also includes preference data and instruction prompts supporting both offline and online RL. Our synthesis framework proceeds in two stages: (1) scale; and (2) complexity. Reasoning traces are then synthesized through a two-stage process that leverages VLMs and reasoning LLMs, producing CoT traces for VLMs that capture the richness and diverse cognitive behaviors found in frontier reasoning models. Remarkably, we show that finetuning Qwen2.5-VL-7B on our data outperforms all open-data baselines across all evaluated vision-centric benchmarks, and even surpasses strong closed-data models such as MiMo-VL-7B-RL on V* Bench, CV-Bench and MMStar-V. Perhaps most surprising, despite being entirely vision-centric, our data transfers positively to text-only reasoning (MMLU-Pro) and audio reasoning (MMAU), demonstrating its effectiveness. Similarly, despite not containing videos or embodied visual data, we observe notable gains when evaluating on a single-evidence embodied QA benchmark (NiEH). Finally, we use our data to analyze the entire VLM post-training pipeline. Our empirical analysis highlights that (i) SFT on high-quality data with non-linear reasoning traces is essential for effective online RL, (ii) staged offline RL matches online RL's performance while reducing compute demands, and (iii) careful SFT on high quality data can substantially improve out-of-domain, cross-modality transfer. 

---
# TabDistill: Distilling Transformers into Neural Nets for Few-Shot Tabular Classification 

**Authors**: Pasan Dissanayake, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2511.05704)  

**Abstract**: Transformer-based models have shown promising performance on tabular data compared to their classical counterparts such as neural networks and Gradient Boosted Decision Trees (GBDTs) in scenarios with limited training data. They utilize their pre-trained knowledge to adapt to new domains, achieving commendable performance with only a few training examples, also called the few-shot regime. However, the performance gain in the few-shot regime comes at the expense of significantly increased complexity and number of parameters. To circumvent this trade-off, we introduce TabDistill, a new strategy to distill the pre-trained knowledge in complex transformer-based models into simpler neural networks for effectively classifying tabular data. Our framework yields the best of both worlds: being parameter-efficient while performing well with limited training data. The distilled neural networks surpass classical baselines such as regular neural networks, XGBoost and logistic regression under equal training data, and in some cases, even the original transformer-based models that they were distilled from. 

---
# Optimizing Diversity and Quality through Base-Aligned Model Collaboration 

**Authors**: Yichen Wang, Chenghao Yang, Tenghao Huang, Muhao Chen, Jonathan May, Mina Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05650)  

**Abstract**: Alignment has greatly improved large language models (LLMs)' output quality at the cost of diversity, yielding highly similar outputs across generations. We propose Base-Aligned Model Collaboration (BACo), an inference-time token-level model collaboration framework that dynamically combines a base LLM with its aligned counterpart to optimize diversity and quality. Inspired by prior work (Fei et al., 2025), BACo employs routing strategies that determine, at each token, from which model to decode based on next-token prediction uncertainty and predicted contents' semantic role. Prior diversity-promoting methods, such as retraining, prompt engineering, and multi-sampling methods, improve diversity but often degrade quality or require costly decoding or post-training. In contrast, BACo achieves both high diversity and quality post hoc within a single pass, while offering strong controllability. We explore a family of routing strategies, across three open-ended generation tasks and 13 metrics covering diversity and quality, BACo consistently surpasses state-of-the-art inference-time baselines. With our best router, BACo achieves a 21.3% joint improvement in diversity and quality. Human evaluations also mirror these improvements. The results suggest that collaboration between base and aligned models can optimize and control diversity and quality. 

---
# BrainCSD: A Hierarchical Consistency-Driven MoE Foundation Model for Unified Connectome Synthesis and Multitask Brain Trait Prediction 

**Authors**: Xiongri Shen, Jiaqi Wang, Yi Zhong, Zhenxi Song, Leilei Zhao, Liling Li, Yichen Wei, Lingyan Liang, Shuqiang Wang, Baiying Lei, Demao Deng, Zhiguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05630)  

**Abstract**: Functional and structural connectivity (FC/SC) are key multimodal biomarkers for brain analysis, yet their clinical utility is hindered by costly acquisition, complex preprocessing, and frequent missing modalities. Existing foundation models either process single modalities or lack explicit mechanisms for cross-modal and cross-scale consistency. We propose BrainCSD, a hierarchical mixture-of-experts (MoE) foundation model that jointly synthesizes FC/SC biomarkers and supports downstream decoding tasks (diagnosis and prediction). BrainCSD features three neuroanatomically grounded components: (1) a ROI-specific MoE that aligns regional activations from canonical networks (e.g., DMN, FPN) with a global atlas via contrastive consistency; (2) a Encoding-Activation MOE that models dynamic cross-time/gradient dependencies in fMRI/dMRI; and (3) a network-aware refinement MoE that enforces structural priors and symmetry at individual and population levels. Evaluated on the datasets under complete and missing-modality settings, BrainCSD achieves SOTA results: 95.6\% accuracy for MCI vs. CN classification without FC, low synthesis error (FC RMSE: 0.038; SC RMSE: 0.006), brain age prediction (MAE: 4.04 years), and MMSE score estimation (MAE: 1.72 points). Code is available in \href{this https URL}{BrainCSD} 

---
# SSTODE: Ocean-Atmosphere Physics-Informed Neural ODEs for Sea Surface Temperature Prediction 

**Authors**: Zheng Jiang, Wei Wang, Gaowei Zhang, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05629)  

**Abstract**: Sea Surface Temperature (SST) is crucial for understanding upper-ocean thermal dynamics and ocean-atmosphere interactions, which have profound economic and social impacts. While data-driven models show promise in SST prediction, their black-box nature often limits interpretability and overlooks key physical processes. Recently, physics-informed neural networks have been gaining momentum but struggle with complex ocean-atmosphere dynamics due to 1) inadequate characterization of seawater movement (e.g., coastal upwelling) and 2) insufficient integration of external SST drivers (e.g., turbulent heat fluxes). To address these challenges, we propose SSTODE, a physics-informed Neural Ordinary Differential Equations (Neural ODEs) framework for SST prediction. First, we derive ODEs from fluid transport principles, incorporating both advection and diffusion to model ocean spatiotemporal dynamics. Through variational optimization, we recover a latent velocity field that explicitly governs the temporal dynamics of SST. Building upon ODE, we introduce an Energy Exchanges Integrator (EEI)-inspired by ocean heat budget equations-to account for external forcing factors. Thus, the variations in the components of these factors provide deeper insights into SST dynamics. Extensive experiments demonstrate that SSTODE achieves state-of-the-art performances in global and regional SST forecasting benchmarks. Furthermore, SSTODE visually reveals the impact of advection dynamics, thermal diffusion patterns, and diurnal heating-cooling cycles on SST evolution. These findings demonstrate the model's interpretability and physical consistency. 

---
# Unveiling the Training Dynamics of ReLU Networks through a Linear Lens 

**Authors**: Longqing Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.05628)  

**Abstract**: Deep neural networks, particularly those employing Rectified Linear Units (ReLU), are often perceived as complex, high-dimensional, non-linear systems. This complexity poses a significant challenge to understanding their internal learning mechanisms. In this work, we propose a novel analytical framework that recasts a multi-layer ReLU network into an equivalent single-layer linear model with input-dependent "effective weights". For any given input sample, the activation pattern of ReLU units creates a unique computational path, effectively zeroing out a subset of weights in the network. By composing the active weights across all layers, we can derive an effective weight matrix, $W_{\text{eff}}(x)$, that maps the input directly to the output for that specific sample. We posit that the evolution of these effective weights reveals fundamental principles of representation learning. Our work demonstrates that as training progresses, the effective weights corresponding to samples from the same class converge, while those from different classes diverge. By tracking the trajectories of these sample-wise effective weights, we provide a new lens through which to interpret the formation of class-specific decision boundaries and the emergence of semantic representations within the network. 

---
# Assessing the Reliability of Large Language Models in the Bengali Legal Context: A Comparative Evaluation Using LLM-as-Judge and Legal Experts 

**Authors**: Sabik Aftahee, A.F.M. Farhad, Arpita Mallik, Ratnajit Dhar, Jawadul Karim, Nahiyan Bin Noor, Ishmam Ahmed Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2511.05627)  

**Abstract**: Accessing legal help in Bangladesh is hard. People face high fees, complex legal language, a shortage of lawyers, and millions of unresolved court cases. Generative AI models like OpenAI GPT-4.1 Mini, Gemini 2.0 Flash, Meta Llama 3 70B, and DeepSeek R1 could potentially democratize legal assistance by providing quick and affordable legal advice. In this study, we collected 250 authentic legal questions from the Facebook group "Know Your Rights," where verified legal experts regularly provide authoritative answers. These questions were subsequently submitted to four four advanced AI models and responses were generated using a consistent, standardized prompt. A comprehensive dual evaluation framework was employed, in which a state-of-the-art LLM model served as a judge, assessing each AI-generated response across four critical dimensions: factual accuracy, legal appropriateness, completeness, and clarity. Following this, the same set of questions was evaluated by three licensed Bangladeshi legal professionals according to the same criteria. In addition, automated evaluation metrics, including BLEU scores, were applied to assess response similarity. Our findings reveal a complex landscape where AI models frequently generate high-quality, well-structured legal responses but also produce dangerous misinformation, including fabricated case citations, incorrect legal procedures, and potentially harmful advice. These results underscore the critical need for rigorous expert validation and comprehensive safeguards before AI systems can be safely deployed for legal consultation in Bangladesh. 

---
# LLMs as Packagers of HPC Software 

**Authors**: Caetano Melone, Daniel Nichols, Konstantinos Parasyris, Todd Gamblin, Harshitha Menon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05626)  

**Abstract**: High performance computing (HPC) software ecosystems are inherently heterogeneous, comprising scientific applications that depend on hundreds of external packages, each with distinct build systems, options, and dependency constraints. Tools such as Spack automate dependency resolution and environment management, but their effectiveness relies on manually written build recipes. As these ecosystems grow, maintaining existing specifications and creating new ones becomes increasingly labor-intensive. While large language models (LLMs) have shown promise in code generation, automatically producing correct and maintainable Spack recipes remains a significant challenge. We present a systematic analysis of how LLMs and context-augmentation methods can assist in the generation of Spack recipes. To this end, we introduce SpackIt, an end-to-end framework that combines repository analysis, retrieval of relevant examples, and iterative refinement through diagnostic feedback. We apply SpackIt to a representative subset of 308 open-source HPC packages to assess its effectiveness and limitations. Our results show that SpackIt increases installation success from 20% in a zero-shot setting to over 80% in its best configuration, demonstrating the value of retrieval and structured feedback for reliable package synthesis. 

---
# Report from Workshop on Dialogue alongside Artificial Intelligence 

**Authors**: Thomas J McKenna, Ingvill Rasmussen, Sten Ludvigsen, Avivit Arvatz, Christa Asterhan, Gaowei Chen, Julie Cohen, Michele Flammia, Dongkeun Han, Emma Hayward, Heather Hill, Yifat Kolikant, Helen Lehndorf, Kexin Li, Lindsay Clare Matsumura, Henrik Tjønn, Pengjin Wang, Rupert Wegerif  

**Link**: [PDF](https://arxiv.org/pdf/2511.05625)  

**Abstract**: Educational dialogue -the collaborative exchange of ideas through talk- is widely recognized as a catalyst for deeper learning and critical thinking in and across contexts. At the same time, artificial intelligence (AI) has rapidly emerged as a powerful force in education, with the potential to address major challenges, personalize learning, and innovate teaching practices. However, these advances come with significant risks: rapid AI development can undermine human agency, exacerbate inequities, and outpace our capacity to guide its use with sound policy. Human learning presupposes cognitive efforts and social interaction (dialogues). In response to this evolving landscape, an international workshop titled "Educational Dialogue: Moving Thinking Forward" convened 19 leading researchers from 11 countries in Cambridge (September 1-3, 2025) to examine the intersection of AI and educational dialogue. This AI-focused strand of the workshop centered on three critical questions: (1) When is AI truly useful in education, and when might it merely replace human effort at the expense of learning? (2) Under what conditions can AI use lead to better dialogic teaching and learning? (3) Does the AI-human partnership risk outpacing and displacing human educational work, and what are the implications? These questions framed two days of presentations and structured dialogue among participants. 

---
# Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition 

**Authors**: Nicholas Babey, Tiffany Gu, Yiheng Li, Cristian Meo, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05622)  

**Abstract**: For embodied agents to effectively understand and interact within the world around them, they require a nuanced comprehension of human actions grounded in physical space. Current action recognition models, often relying on RGB video, learn superficial correlations between patterns and action labels, so they struggle to capture underlying physical interaction dynamics and human poses in complex scenes. We propose a model architecture that grounds action recognition in physical space by fusing two powerful, complementary representations: V-JEPA 2's contextual, predictive world dynamics and CoMotion's explicit, occlusion-tolerant human pose data. Our model is validated on both the InHARD and UCF-19-Y-OCC benchmarks for general action recognition and high-occlusion action recognition, respectively. Our model outperforms three other baselines, especially within complex, occlusive scenes. Our findings emphasize a need for action recognition to be supported by spatial understanding instead of statistical pattern recognition. 

---
# Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift 

**Authors**: Tianze Wang, Sofiane Ennadir, John Pertoft, Gabriela Zarzar Gandler, Lele Cao, Zineb Senane, Styliani Katsarou, Sahar Asadi, Axel Karlsson, Oleg Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2511.05619)  

**Abstract**: Time series foundation models (TSFMs) have shown strong results on public benchmarks, prompting comparisons to a "BERT moment" for time series. Their effectiveness in industrial settings, however, remains uncertain. We examine why TSFMs often struggle to generalize and highlight spectral shift (a mismatch between the dominant frequency components in downstream tasks and those represented during pretraining) as a key factor. We present evidence from an industrial-scale player engagement prediction task in mobile gaming, where TSFMs underperform domain-adapted baselines. To isolate the mechanism, we design controlled synthetic experiments contrasting signals with seen versus unseen frequency bands, observing systematic degradation under spectral mismatch. These findings position frequency awareness as critical for robust TSFM deployment and motivate new pretraining and evaluation protocols that explicitly account for spectral diversity. 

---
# Personalized Image Editing in Text-to-Image Diffusion Models via Collaborative Direct Preference Optimization 

**Authors**: Connor Dunlop, Matthew Zheng, Kavana Venkatesh, Pinar Yanardag  

**Link**: [PDF](https://arxiv.org/pdf/2511.05616)  

**Abstract**: Text-to-image (T2I) diffusion models have made remarkable strides in generating and editing high-fidelity images from text. Yet, these models remain fundamentally generic, failing to adapt to the nuanced aesthetic preferences of individual users. In this work, we present the first framework for personalized image editing in diffusion models, introducing Collaborative Direct Preference Optimization (C-DPO), a novel method that aligns image edits with user-specific preferences while leveraging collaborative signals from like-minded individuals. Our approach encodes each user as a node in a dynamic preference graph and learns embeddings via a lightweight graph neural network, enabling information sharing across users with overlapping visual tastes. We enhance a diffusion model's editing capabilities by integrating these personalized embeddings into a novel DPO objective, which jointly optimizes for individual alignment and neighborhood coherence. Comprehensive experiments, including user studies and quantitative benchmarks, demonstrate that our method consistently outperforms baselines in generating edits that are aligned with user preferences. 

---
# wa-hls4ml: A Benchmark and Surrogate Models for hls4ml Resource and Latency Estimation 

**Authors**: Benjamin Hawks, Jason Weitz, Dmitri Demler, Karla Tame-Narvaez, Dennis Plotnikov, Mohammad Mehdi Rahimifar, Hamza Ezzaoui Rahali, Audrey C. Therrien, Donovan Sproule, Elham E Khoda, Keegan A. Smith, Russell Marroquin, Giuseppe Di Guglielmo, Nhan Tran, Javier Duarte, Vladimir Loncar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05615)  

**Abstract**: As machine learning (ML) is increasingly implemented in hardware to address real-time challenges in scientific applications, the development of advanced toolchains has significantly reduced the time required to iterate on various designs. These advancements have solved major obstacles, but also exposed new challenges. For example, processes that were not previously considered bottlenecks, such as hardware synthesis, are becoming limiting factors in the rapid iteration of designs. To mitigate these emerging constraints, multiple efforts have been undertaken to develop an ML-based surrogate model that estimates resource usage of ML accelerator architectures. We introduce wa-hls4ml, a benchmark for ML accelerator resource and latency estimation, and its corresponding initial dataset of over 680,000 fully connected and convolutional neural networks, all synthesized using hls4ml and targeting Xilinx FPGAs. The benchmark evaluates the performance of resource and latency predictors against several common ML model architectures, primarily originating from scientific domains, as exemplar models, and the average performance across a subset of the dataset. Additionally, we introduce GNN- and transformer-based surrogate models that predict latency and resources for ML accelerators. We present the architecture and performance of the models and find that the models generally predict latency and resources for the 75% percentile within several percent of the synthesized resources on the synthetic test dataset. 

---
# An MLCommons Scientific Benchmarks Ontology 

**Authors**: Ben Hawks, Gregor von Laszewski, Matthew D. Sinclair, Marco Colombo, Shivaram Venkataraman, Rutwik Jain, Yiwei Jiang, Nhan Tran, Geoffrey Fox  

**Link**: [PDF](https://arxiv.org/pdf/2511.05614)  

**Abstract**: Scientific machine learning research spans diverse domains and data modalities, yet existing benchmark efforts remain siloed and lack standardization. This makes novel and transformative applications of machine learning to critical scientific use-cases more fragmented and less clear in pathways to impact. This paper introduces an ontology for scientific benchmarking developed through a unified, community-driven effort that extends the MLCommons ecosystem to cover physics, chemistry, materials science, biology, climate science, and more. Building on prior initiatives such as XAI-BENCH, FastML Science Benchmarks, PDEBench, and the SciMLBench framework, our effort consolidates a large set of disparate benchmarks and frameworks into a single taxonomy of scientific, application, and system-level benchmarks. New benchmarks can be added through an open submission workflow coordinated by the MLCommons Science Working Group and evaluated against a six-category rating rubric that promotes and identifies high-quality benchmarks, enabling stakeholders to select benchmarks that meet their specific needs. The architecture is extensible, supporting future scientific and AI/ML motifs, and we discuss methods for identifying emerging computing patterns for unique scientific workloads. The MLCommons Science Benchmarks Ontology provides a standardized, scalable foundation for reproducible, cross-domain benchmarking in scientific machine learning. A companion webpage for this work has also been developed as the effort evolves: this https URL 

---
# Who Evaluates AI's Social Impacts? Mapping Coverage and Gaps in First and Third Party Evaluations 

**Authors**: Anka Reuel, Avijit Ghosh, Jenny Chim, Andrew Tran, Yanan Long, Jennifer Mickel, Usman Gohar, Srishti Yadav, Pawan Sasanka Ammanamanchi, Mowafak Allaham, Hossein A. Rahmani, Mubashara Akhtar, Felix Friedrich, Robert Scholz, Michael Alexander Riegler, Jan Batzner, Eliya Habba, Arushi Saxena, Anastassia Kornilova, Kevin Wei, Prajna Soni, Yohan Mathew, Kevin Klyman, Jeba Sania, Subramanyam Sahoo, Olivia Beyer Bruvik, Pouya Sadeghi, Sujata Goswami, Angelina Wang, Yacine Jernite, Zeerak Talat, Stella Biderman, Mykel Kochenderfer, Sanmi Koyejo, Irene Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2511.05613)  

**Abstract**: Foundation models are increasingly central to high-stakes AI systems, and governance frameworks now depend on evaluations to assess their risks and capabilities. Although general capability evaluations are widespread, social impact assessments covering bias, fairness, privacy, environmental costs, and labor practices remain uneven across the AI ecosystem. To characterize this landscape, we conduct the first comprehensive analysis of both first-party and third-party social impact evaluation reporting across a wide range of model developers. Our study examines 186 first-party release reports and 183 post-release evaluation sources, and complements this quantitative analysis with interviews of model developers. We find a clear division of evaluation labor: first-party reporting is sparse, often superficial, and has declined over time in key areas such as environmental impact and bias, while third-party evaluators including academic researchers, nonprofits, and independent organizations provide broader and more rigorous coverage of bias, harmful content, and performance disparities. However, this complementarity has limits. Only model developers can authoritatively report on data provenance, content moderation labor, financial costs, and training infrastructure, yet interviews reveal that these disclosures are often deprioritized unless tied to product adoption or regulatory compliance. Our findings indicate that current evaluation practices leave major gaps in assessing AI's societal impacts, highlighting the urgent need for policies that promote developer transparency, strengthen independent evaluation ecosystems, and create shared infrastructure to aggregate and compare third-party evaluations in a consistent and accessible way. 

---
# AI-Enhanced High-Density NIRS Patch for Real-Time Brain Layer Oxygenation Monitoring in Neurological Emergencies 

**Authors**: Minsu Ji, Jihoon Kang, Seongkwon Yu, Jaemyoung Kim, Bumjun Koh, Jimin Lee, Guil Jeong, Jongkwan choi, Chang-Ho Yun, Hyeonmin Bae  

**Link**: [PDF](https://arxiv.org/pdf/2511.05612)  

**Abstract**: Photon scattering has traditionally limited the ability of near-infrared spectroscopy (NIRS) to extract accurate, layer-specific information from the brain. This limitation restricts its clinical utility for precise neurological monitoring. To address this, we introduce an AI-driven, high-density NIRS system optimized to provide real-time, layer-specific oxygenation data from the brain cortex, specifically targeting acute neuro-emergencies. Our system integrates high-density NIRS reflectance data with a neural network trained on MRI-based synthetic datasets. This approach achieves robust cortical oxygenation accuracy across diverse anatomical variations. In simulations, our AI-assisted NIRS demonstrated a strong correlation (R2=0.913) with actual cortical oxygenation, markedly outperforming conventional methods (R2=0.469). Furthermore, biomimetic phantom experiments confirmed its superior anatomical reliability (R2=0.986) compared to standard commercial devices (R2=0.823). In clinical validation with healthy subjects and ischemic stroke patients, the system distinguished between the two groups with an AUC of 0.943. This highlights its potential as an accessible, high-accuracy diagnostic tool for emergency and point-of-care settings. These results underscore the system's capability to advance neuro-monitoring precision through AI, enabling timely, data-driven decisions in critical care environments. 

---
# Conformal Prediction-Driven Adaptive Sampling for Digital Twins of Water Distribution Networks 

**Authors**: Mohammadhossein Homaei, Oscar Mogollon Gutierrez, Ruben Molano, Andres Caro, Mar Avila  

**Link**: [PDF](https://arxiv.org/pdf/2511.05610)  

**Abstract**: Digital Twins (DTs) for Water Distribution Networks (WDNs) require accurate state estimation with limited sensors. Uniform sampling often wastes resources across nodes with different uncertainty. We propose an adaptive framework combining LSTM forecasting and Conformal Prediction (CP) to estimate node-wise uncertainty and focus sensing on the most uncertain points. Marginal CP is used for its low computational cost, suitable for real-time DTs. Experiments on Hanoi, Net3, and CTOWN show 33-34% lower demand error than uniform sampling at 40% coverage and maintain 89.4-90.2% empirical coverage with only 5-10% extra computation. 

---
# Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation 

**Authors**: Ziying Li, Xuequan Lu, Xinkui Zhao, Guanjie Cheng, Shuiguang Deng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05609)  

**Abstract**: Recent advancements in optimization-based text-to-3D generation heavily rely on distilling knowledge from pre-trained text-to-image diffusion models using techniques like Score Distillation Sampling (SDS), which often introduce artifacts such as over-saturation and over-smoothing into the generated 3D assets. In this paper, we address this essential problem by formulating the generation process as learning an optimal, direct transport trajectory between the distribution of the current rendering and the desired target distribution, thereby enabling high-quality generation with smaller Classifier-free Guidance (CFG) values. At first, we theoretically establish SDS as a simplified instance of the Schrödinger Bridge framework. We prove that SDS employs the reverse process of an Schrödinger Bridge, which, under specific conditions (e.g., a Gaussian noise as one end), collapses to SDS's score function of the pre-trained diffusion model. Based upon this, we introduce Trajectory-Centric Distillation (TraCe), a novel text-to-3D generation framework, which reformulates the mathematically trackable framework of Schrödinger Bridge to explicitly construct a diffusion bridge from the current rendering to its text-conditioned, denoised target, and trains a LoRA-adapted model on this trajectory's score dynamics for robust 3D optimization. Comprehensive experiments demonstrate that TraCe consistently achieves superior quality and fidelity to state-of-the-art techniques. 

---
# Google-MedGemma Based Abnormality Detection in Musculoskeletal radiographs 

**Authors**: Soumyajit Maity, Pranjal Kamboj, Sneha Maity, Rajat Singh, Sankhadeep Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05600)  

**Abstract**: This paper proposes a MedGemma-based framework for automatic abnormality detection in musculoskeletal radiographs. Departing from conventional autoencoder and neural network pipelines, the proposed method leverages the MedGemma foundation model, incorporating a SigLIP-derived vision encoder pretrained on diverse medical imaging modalities. Preprocessed X-ray images are encoded into high-dimensional embeddings using the MedGemma vision backbone, which are subsequently passed through a lightweight multilayer perceptron for binary classification. Experimental assessment reveals that the MedGemma-driven classifier exhibits strong performance, exceeding conventional convolutional and autoencoder-based metrics. Additionally, the model leverages MedGemma's transfer learning capabilities, enhancing generalization and optimizing feature engineering. The integration of a modern medical foundation model not only enhances representation learning but also facilitates modular training strategies such as selective encoder block unfreezing for efficient domain adaptation. The findings suggest that MedGemma-powered classification systems can advance clinical radiograph triage by providing scalable and accurate abnormality detection, with potential for broader applications in automated medical image analysis.
Keywords: Google MedGemma, MURA, Medical Image, Classification. 

---
# FlowNet: Modeling Dynamic Spatio-Temporal Systems via Flow Propagation 

**Authors**: Yutong Feng, Xu Liu, Yutong Xia, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05595)  

**Abstract**: Accurately modeling complex dynamic spatio-temporal systems requires capturing flow-mediated interdependencies and context-sensitive interaction dynamics. Existing methods, predominantly graph-based or attention-driven, rely on similarity-driven connectivity assumptions, neglecting asymmetric flow exchanges that govern system evolution. We propose Spatio-Temporal Flow, a physics-inspired paradigm that explicitly models dynamic node couplings through quantifiable flow transfers governed by conservation principles. Building on this, we design FlowNet, a novel architecture leveraging flow tokens as information carriers to simulate source-to-destination transfers via Flow Allocation Modules, ensuring state redistribution aligns with conservation laws. FlowNet dynamically adjusts the interaction radius through an Adaptive Spatial Masking module, suppressing irrelevant noise while enabling context-aware propagation. A cascaded architecture enhances scalability and nonlinear representation capacity. Experiments demonstrate that FlowNet significantly outperforms existing state-of-the-art approaches on seven metrics in the modeling of three real-world systems, validating its efficiency and physical interpretability. We establish a principled methodology for modeling complex systems through spatio-temporal flow interactions. 

---
# CoPRIS: Efficient and Stable Reinforcement Learning via Concurrency-Controlled Partial Rollout with Importance Sampling 

**Authors**: Zekai Qu, Yinxu Pan, Ao Sun, Chaojun Xiao, Xu Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.05589)  

**Abstract**: Reinforcement learning (RL) post-training has become a trending paradigm for enhancing the capabilities of large language models (LLMs). Most existing RL systems for LLMs operate in a fully synchronous manner, where training must wait for the rollout of an entire batch to complete. This design leads to severe inefficiencies, as extremely long trajectories can stall the entire rollout process and leave many GPUs idle. To address this issue, we propose Concurrency- Controlled Partial Rollout with Importance Sampling (CoPRIS), which mitigates long-tail inefficiencies by maintaining a fixed number of concurrent rollouts, early-terminating once sufficient samples are collected, and reusing unfinished trajectories in subsequent rollouts. To mitigate the impact of off-policy trajectories, we introduce Cross-stage Importance Sampling Correction, which concatenates buffered log probabilities from the previous policy with those recomputed under the current policy for importance sampling correction. Experiments on challenging mathematical reasoning benchmarks show that CoPRIS achieves up to 1.94x faster training while maintaining comparable or superior performance to synchronous RL systems. The code of CoPRIS is available at this https URL. 

---
# Fine-Tuning Vision-Language Models for Multimodal Polymer Property Prediction 

**Authors**: An Vuong, Minh-Hao Van, Prateek Verma, Chen Zhao, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05577)  

**Abstract**: Vision-Language Models (VLMs) have shown strong performance in tasks like visual question answering and multimodal text generation, but their effectiveness in scientific domains such as materials science remains limited. While some machine learning methods have addressed specific challenges in this field, there is still a lack of foundation models designed for broad tasks like polymer property prediction using multimodal data. In this work, we present a multimodal polymer dataset to fine-tune VLMs through instruction-tuning pairs and assess the impact of multimodality on prediction performance. Our fine-tuned models, using LoRA, outperform unimodal and baseline approaches, demonstrating the benefits of multimodal learning. Additionally, this approach reduces the need to train separate models for different properties, lowering deployment and maintenance costs. 

---
# Elements of Active Continuous Learning and Uncertainty Self-Awareness: a Narrow Implementation for Face and Facial Expression Recognition 

**Authors**: Stanislav Selitskiy  

**Link**: [PDF](https://arxiv.org/pdf/2511.05574)  

**Abstract**: Reflection on one's thought process and making corrections to it if there exists dissatisfaction in its performance is, perhaps, one of the essential traits of intelligence. However, such high-level abstract concepts mandatory for Artificial General Intelligence can be modelled even at the low level of narrow Machine Learning algorithms. Here, we present the self-awareness mechanism emulation in the form of a supervising artificial neural network (ANN) observing patterns in activations of another underlying ANN in a search for indications of the high uncertainty of the underlying ANN and, therefore, the trustworthiness of its predictions. The underlying ANN is a convolutional neural network (CNN) ensemble employed for face recognition and facial expression tasks. The self-awareness ANN has a memory region where its past performance information is stored, and its learnable parameters are adjusted during the training to optimize the performance. The trustworthiness verdict triggers the active learning mode, giving elements of agency to the machine learning algorithm that asks for human help in high uncertainty and confusion conditions. 

---
# Video Text Preservation with Synthetic Text-Rich Videos 

**Authors**: Ziyang Liu, Kevin Valencia, Justin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2511.05573)  

**Abstract**: While Text-To-Video (T2V) models have advanced rapidly, they continue to struggle with generating legible and coherent text within videos. In particular, existing models often fail to render correctly even short phrases or words and previous attempts to address this problem are computationally expensive and not suitable for video generation. In this work, we investigate a lightweight approach to improve T2V diffusion models using synthetic supervision. We first generate text-rich images using a text-to-image (T2I) diffusion model, then animate them into short videos using a text-agnostic image-to-video (I2v) model. These synthetic video-prompt pairs are used to fine-tune Wan2.1, a pre-trained T2V model, without any architectural changes. Our results show improvement in short-text legibility and temporal consistency with emerging structural priors for longer text. These findings suggest that curated synthetic data and weak supervision offer a practical path toward improving textual fidelity in T2V generation. 

---
# C3-Diff: Super-resolving Spatial Transcriptomics via Cross-modal Cross-content Contrastive Diffusion Modelling 

**Authors**: Xiaofei Wang, Stephen Price, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.05571)  

**Abstract**: The rapid advancement of spatial transcriptomics (ST), i.e., spatial gene expressions, has made it possible to measure gene expression within original tissue, enabling us to discover molecular mechanisms. However, current ST platforms frequently suffer from low resolution, limiting the in-depth understanding of spatial gene expression. Super-resolution approaches promise to enhance ST maps by integrating histology images with gene expressions of profiled tissue spots. However, it remains a challenge to model the interactions between histology images and gene expressions for effective ST enhancement. This study presents a cross-modal cross-content contrastive diffusion framework, called C3-Diff, for ST enhancement with histology images as guidance. In C3-Diff, we firstly analyze the deficiency of traditional contrastive learning paradigm, which is then refined to extract both modal-invariant and content-invariant features of ST maps and histology images. Further, to overcome the problem of low sequencing sensitivity in ST maps, we perform nosing-based information augmentation on the surface of feature unit hypersphere. Finally, we propose a dynamic cross-modal imputation-based training strategy to mitigate ST data scarcity. We tested C3-Diff by benchmarking its performance on four public datasets, where it achieves significant improvements over competing methods. Moreover, we evaluate C3-Diff on downstream tasks of cell type localization, gene expression correlation and single-cell-level gene expression prediction, promoting AI-enhanced biotechnology for biomedical research and clinical applications. Codes are available at this https URL. 

---
# Automatic Extraction of Road Networks by using Teacher-Student Adaptive Structural Deep Belief Network and Its Application to Landslide Disaster 

**Authors**: Shin Kamada, Takumi Ichimura  

**Link**: [PDF](https://arxiv.org/pdf/2511.05567)  

**Abstract**: An adaptive structural learning method of Restricted Boltzmann Machine (RBM) and Deep Belief Network (DBN) has been developed as one of prominent deep learning models. The neuron generation-annihilation algorithm in RBM and layer generation algorithm in DBN make an optimal network structure for given input during the learning. In this paper, our model is applied to an automatic recognition method of road network system, called RoadTracer. RoadTracer can generate a road map on the ground surface from aerial photograph data. A novel method of RoadTracer using the Teacher-Student based ensemble learning model of Adaptive DBN is proposed, since the road maps contain many complicated features so that a model with high representation power to detect should be required. The experimental results showed the detection accuracy of the proposed model was improved from 40.0\% to 89.0\% on average in the seven major cities among the test dataset. In addition, we challenged to apply our method to the detection of available roads when landslide by natural disaster is occurred, in order to rapidly obtain a way of transportation. For fast inference, a small size of the trained model was implemented on a small embedded edge device as lightweight deep learning. We reported the detection results for the satellite image before and after the rainfall disaster in Japan. 

---
# Efficient Online Continual Learning in Sensor-Based Human Activity Recognition 

**Authors**: Yao Zhang, Souza Leite Clayton, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.05566)  

**Abstract**: Machine learning models for sensor-based human activity recognition (HAR) are expected to adapt post-deployment to recognize new activities and different ways of performing existing ones. To address this need, Online Continual Learning (OCL) mechanisms have been proposed, allowing models to update their knowledge incrementally as new data become available while preserving previously acquired information. However, existing OCL approaches for sensor-based HAR are computationally intensive and require extensive labeled samples to represent new changes. Recently, pre-trained model-based (PTM-based) OCL approaches have shown significant improvements in performance and efficiency for computer vision applications. These methods achieve strong generalization capabilities by pre-training complex models on large datasets, followed by fine-tuning on downstream tasks for continual learning. However, applying PTM-based OCL approaches to sensor-based HAR poses significant challenges due to the inherent heterogeneity of HAR datasets and the scarcity of labeled data in post-deployment scenarios. This paper introduces PTRN-HAR, the first successful application of PTM-based OCL to sensor-based HAR. Unlike prior PTM-based OCL approaches, PTRN-HAR pre-trains the feature extractor using contrastive loss with a limited amount of data. This extractor is then frozen during the streaming stage. Furthermore, it replaces the conventional dense classification layer with a relation module network. Our design not only significantly reduces the resource consumption required for model training while maintaining high performance, but also improves data efficiency by reducing the amount of labeled data needed for effective continual learning, as demonstrated through experiments on three public datasets, outperforming the state-of-the-art. The code can be found here: this https URL 

---
# In-Context Adaptation of VLMs for Few-Shot Cell Detection in Optical Microscopy 

**Authors**: Shreyan Ganguly, Angona Biswas, Jaydeep Rade, Md Hasibul Hasan Hasib, Nabila Masud, Nitish Singla, Abhipsa Dash, Ushashi Bhattacharjee, Aditya Balu, Anwesha Sarkar, Adarsh Krishnamurthy, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05565)  

**Abstract**: Foundation vision-language models (VLMs) excel on natural images, but their utility for biomedical microscopy remains underexplored. In this paper, we investigate how in-context learning enables state-of-the-art VLMs to perform few-shot object detection when large annotated datasets are unavailable, as is often the case with microscopic images. We introduce the Micro-OD benchmark, a curated collection of 252 images specifically curated for in-context learning, with bounding-box annotations spanning 11 cell types across four sources, including two in-lab expert-annotated sets. We systematically evaluate eight VLMs under few-shot conditions and compare variants with and without implicit test-time reasoning tokens. We further implement a hybrid Few-Shot Object Detection (FSOD) pipeline that combines a detection head with a VLM-based few-shot classifier, which enhances the few-shot performance of recent VLMs on our benchmark. Across datasets, we observe that zero-shot performance is weak due to the domain gap; however, few-shot support consistently improves detection, with marginal gains achieved after six shots. We observe that models with reasoning tokens are more effective for end-to-end localization, whereas simpler variants are more suitable for classifying pre-localized crops. Our results highlight in-context adaptation as a practical path for microscopy, and our benchmark provides a reproducible testbed for advancing open-vocabulary detection in biomedical imaging. 

---
# Lookahead Unmasking Elicits Accurate Decoding in Diffusion Language Models 

**Authors**: Sanghyun Lee, Seungryong Kim, Jongho Park, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.05563)  

**Abstract**: Masked Diffusion Models (MDMs) as language models generate by iteratively unmasking tokens, yet their performance crucially depends on the inference time order of unmasking. Prevailing heuristics, such as confidence based sampling, are myopic: they optimize locally, fail to leverage extra test-time compute, and let early decoding mistakes cascade. We propose Lookahead Unmasking (LookUM), which addresses these concerns by reformulating sampling as path selection over all possible unmasking orders without the need for an external reward model. Our framework couples (i) a path generator that proposes paths by sampling from pools of unmasking sets with (ii) a verifier that computes the uncertainty of the proposed paths and performs importance sampling to subsequently select the final paths. Empirically, erroneous unmasking measurably inflates sequence level uncertainty, and our method exploits this to avoid error-prone trajectories. We validate our framework across six benchmarks, such as mathematics, planning, and coding, and demonstrate consistent performance improvements. LookUM requires only two to three paths to achieve peak performance, demonstrating remarkably efficient path selection. The consistent improvements on both LLaDA and post-trained LLaDA 1.5 are particularly striking: base LLaDA with LookUM rivals the performance of RL-tuned LLaDA 1.5, while LookUM further enhances LLaDA 1.5 itself showing that uncertainty based verification provides orthogonal benefits to reinforcement learning and underscoring the versatility of our framework. Code will be publicly released. 

---
# Effective Test-Time Scaling of Discrete Diffusion through Iterative Refinement 

**Authors**: Sanghyun Lee, Sunwoo Kim, Seungryong Kim, Jongho Park, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.05562)  

**Abstract**: Test-time scaling through reward-guided generation remains largely unexplored for discrete diffusion models despite its potential as a promising alternative. In this work, we introduce Iterative Reward-Guided Refinement (IterRef), a novel test-time scaling method tailored to discrete diffusion that leverages reward- guided noising-denoising transitions to progressively refine misaligned intermediate states. We formalize this process within a Multiple-Try Metropolis (MTM) framework, proving convergence to the reward-aligned distribution. Unlike prior methods that assume the current state is already aligned with the reward distribution and only guide the subsequent transition, our approach explicitly refines each state in situ, progressively steering it toward the optimal intermediate distribution. Across both text and image domains, we evaluate IterRef on diverse discrete diffusion models and observe consistent improvements in reward-guided generation quality. In particular, IterRef achieves striking gains under low compute budgets, far surpassing prior state-of-the-art baselines. 

---
# Sample-Efficient Language Modeling with Linear Attention and Lightweight Enhancements 

**Authors**: Patrick Haller, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2511.05560)  

**Abstract**: We study architectural and optimization tech- niques for sample-efficient language modeling under the constraints of the BabyLM 2025 shared task. Our model, BLaLM, replaces self-attention with a linear-time mLSTM to- ken mixer and explores lightweight enhance- ments, including short convolutions, sliding window attention with dynamic modulation, and Hedgehog feature maps. To support train- ing in low-resource settings, we curate a high- quality corpus emphasizing readability and ped- agogical structure. Experiments across both STRICT and STRICT-SMALL tracks show that (1) linear attention combined with sliding win- dow attention consistently improves zero-shot performance, and (2) the Muon optimizer stabi- lizes convergence and reduces perplexity over AdamW. These results highlight effective strate- gies for efficient language modeling without relying on scale. 

---
# Diversified Flow Matching with Translation Identifiability 

**Authors**: Sagar Shrestha, Xiao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05558)  

**Abstract**: Diversified distribution matching (DDM) finds a unified translation function mapping a diverse collection of conditional source distributions to their target counterparts. DDM was proposed to resolve content misalignment issues in unpaired domain translation, achieving translation identifiability. However, DDM has only been implemented using GANs due to its constraints on the translation function. GANs are often unstable to train and do not provide the transport trajectory information -- yet such trajectories are useful in applications such as single-cell evolution analysis and robot route planning. This work introduces diversified flow matching (DFM), an ODE-based framework for DDM. Adapting flow matching (FM) to enforce a unified translation function as in DDM is challenging, as FM learns the translation function's velocity rather than the translation function itself. A custom bilevel optimization-based training loss, a nonlinear interpolant, and a structural reformulation are proposed to address these challenges, offering a tangible implementation. To our knowledge, DFM is the first ODE-based approach guaranteeing translation identifiability. Experiments on synthetic and real-world datasets validate the proposed method. 

---
# EVLP:Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning 

**Authors**: Xinyan Cai, Shiguang Wu, Dafeng Chi, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Qiang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05553)  

**Abstract**: In complex embodied long-horizon manipulation tasks, effective task decomposition and execution require synergistic integration of textual logical reasoning and visual-spatial imagination to ensure efficient and accurate operation. Current methods fail to adopt a unified generation framework for multimodal planning, lead to inconsistent in multimodal planning. To address this challenge, we present \textbf{EVLP (Embodied Vision-Language Planner)}, an innovative multimodal unified generation framework that jointly models linguistic reasoning and visual generation. Our approach achieves multimodal planning for long-horizon tasks through a novel training pipeline incorporating dynamic pretraining and reinforced alignment. Our core innovations consist of three key components: \textbf{1) Unified Multimodal Generation Framework}: For understanding, We integrate semantic information with spatial features to provide comprehensive visual perception. For generation, we directly learn the joint distribution of discrete images for one-step visual synthesis, enabling coordinated language-visual modeling through learnable cross-modal attention mechanisms. \textbf{2) Dynamic Perception Pretraining}: We propose a bidirectional dynamic alignment strategy employing inverse dynamics tasks and forward dynamics tasks, effectively strengthening multimodal correlations within a unified feature space. \textbf{3) Reinforced Supervised Fine-Tuning}: While conducting instruction-based fine-tuning in the unified generation space, we construct a reinforce loss to align the spatial logic between textual actions and generated images, enabling the model to acquire spatio-awared multimodal planning capabilities. 

---
# Deep one-gate per layer networks with skip connections are universal classifiers 

**Authors**: Raul Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2511.05552)  

**Abstract**: This paper shows how a multilayer perceptron with two hidden layers, which has been designed to classify two classes of data points, can easily be transformed into a deep neural network with one-gate layers and skip connections. 

---
# AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs 

**Authors**: Yubo Wang, Haoyang Li, Fei Teng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05549)  

**Abstract**: Graph-based retrieval-augmented generation (Graph-based RAG) has demonstrated significant potential in enhancing Large Language Models (LLMs) with structured knowledge. However, existing methods face three critical challenges: Inaccurate Graph Construction, caused by LLM hallucination; Poor Reasoning Ability, caused by failing to generate explicit reasons telling LLM why certain chunks were selected; and Inadequate Answering, which only partially answers the query due to the inadequate LLM reasoning, making their performance lag behind NaiveRAG on certain tasks. To address these issues, we propose AGRAG, an advanced graph-based retrieval-augmented generation framework. When constructing the graph, AGRAG substitutes the widely used LLM entity extraction method with a statistics-based method, avoiding hallucination and error propagation. When retrieval, AGRAG formulates the graph reasoning procedure as the Minimum Cost Maximum Influence (MCMI) subgraph generation problem, where we try to include more nodes with high influence score, but with less involving edge cost, to make the generated reasoning paths more comprehensive. We prove this problem to be NP-hard, and propose a greedy algorithm to solve it. The MCMI subgraph generated can serve as explicit reasoning paths to tell LLM why certain chunks were retrieved, thereby making the LLM better focus on the query-related part contents of the chunks, reducing the impact of noise, and improving AGRAG's reasoning ability. Furthermore, compared with the simple tree-structured reasoning paths, our MCMI subgraph can allow more complex graph structures, such as cycles, and improve the comprehensiveness of the generated reasoning paths. 

---
# Automated Invoice Data Extraction: Using LLM and OCR 

**Authors**: Advait Thakur, Khushi Khanchandani, Akshita Shetty, Chaitravi Reddy, Ritisa Behera  

**Link**: [PDF](https://arxiv.org/pdf/2511.05547)  

**Abstract**: Conventional Optical Character Recognition (OCR) systems are challenged by variant invoice layouts, handwritten text, and low- quality scans, which are often caused by strong template dependencies that restrict their flexibility across different document structures and layouts. Newer solutions utilize advanced deep learning models such as Convolutional Neural Networks (CNN) as well as Transformers, and domain-specific models for better layout analysis and accuracy across various sections over varied document types. Large Language Models (LLMs) have revolutionized extraction pipelines at their core with sophisticated entity recognition and semantic comprehension to support complex contextual relationship mapping without direct programming specification. Visual Named Entity Recognition (NER) capabilities permit extraction from invoice images with greater contextual sensitivity and much higher accuracy rates than older approaches. Existing industry best practices utilize hybrid architectures that blend OCR technology and LLM for maximum scalability and minimal human intervention. This work introduces a holistic Artificial Intelligence (AI) platform combining OCR, deep learning, LLMs, and graph analytics to achieve unprecedented extraction quality and consistency. 

---
# ConnectomeBench: Can LLMs Proofread the Connectome? 

**Authors**: Jeff Brown, Andrew Kirjner Annika Vivekananthan, Ed Boyden  

**Link**: [PDF](https://arxiv.org/pdf/2511.05542)  

**Abstract**: Connectomics - the mapping of neural connections in an organism's brain - currently requires extraordinary human effort to proofread the data collected from imaging and machine-learning assisted segmentation. With the growing excitement around using AI agents to automate important scientific tasks, we explore whether current AI systems can perform multiple tasks necessary for data proofreading. We introduce ConnectomeBench, a multimodal benchmark evaluating large language model (LLM) capabilities in three critical proofreading tasks: segment type identification, split error correction, and merge error detection. Using expert annotated data from two large open-source datasets - a cubic millimeter of mouse visual cortex and the complete Drosophila brain - we evaluate proprietary multimodal LLMs including Claude 3.7/4 Sonnet, o4-mini, GPT-4.1, GPT-4o, as well as open source models like InternVL-3 and NVLM. Our results demonstrate that current models achieve surprisingly high performance in segment identification (52-82% balanced accuracy vs. 20-25% chance) and binary/multiple choice split error correction (75-85% accuracy vs. 50% chance) while generally struggling on merge error identification tasks. Overall, while the best models still lag behind expert performance, they demonstrate promising capabilities that could eventually enable them to augment and potentially replace human proofreading in connectomics. Project page: this https URL and Dataset this https URL 

---
# Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability 

**Authors**: Usha Bhalla, Alex Oesterling, Claudio Mayrink Verdun, Himabindu Lakkaraju, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05541)  

**Abstract**: Translating the internal representations and computations of models into concepts that humans can understand is a key goal of interpretability. While recent dictionary learning methods such as Sparse Autoencoders (SAEs) provide a promising route to discover human-interpretable features, they suffer from a variety of problems, including a systematic failure to capture the rich conceptual information that drives linguistic understanding. Instead, they exhibit a bias towards shallow, token-specific, or noisy features, such as "the phrase 'The' at the start of sentences". In this work, we propose that this is due to a fundamental issue with how dictionary learning methods for LLMs are trained. Language itself has a rich, well-studied structure spanning syntax, semantics, and pragmatics; however, current unsupervised methods largely ignore this linguistic knowledge, leading to poor feature discovery that favors superficial patterns over meaningful concepts. We focus on a simple but important aspect of language: semantic content has long-range dependencies and tends to be smooth over a sequence, whereas syntactic information is much more local. Building on this insight, we introduce Temporal Sparse Autoencoders (T-SAEs), which incorporate a novel contrastive loss encouraging consistent activations of high-level features over adjacent tokens. This simple yet powerful modification enables SAEs to disentangle semantic from syntactic features in a self-supervised manner. Across multiple datasets and models, T-SAEs recover smoother, more coherent semantic concepts without sacrificing reconstruction quality. Strikingly, they exhibit clear semantic structure despite being trained without explicit semantic signal, offering a new pathway for unsupervised interpretability in language models. 

---
# Token Is All You Need: Cognitive Planning through Sparse Intent Alignment 

**Authors**: Shiyao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05540)  

**Abstract**: We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Unlike world-model approaches that rely on computationally intensive future scene generation or vision-language-action (VLA) systems constrained by Markov assumptions, we show that a minimal set of semantically rich tokens is sufficient for effective planning. Experiments on the nuPlan benchmark (720 scenarios, over 11,000 samples) using perception-informed BEV representations yield three key findings: (1) even without future prediction, our sparse representation achieves 0.548 m ADE, comparable to or surpassing prior methods reporting around 0.75 m on nuScenes; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.479 m, a 12.6% improvement over current-state baselines; and (3) explicit reconstruction loss offers no benefit and may degrade performance under reliable perception inputs. Notably, we observe the emergence of temporal fuzziness, where the model adaptively attends to task-relevant semantics rather than aligning rigidly to fixed timestamps, providing a cognitive advantage for planning under uncertainty. Our "token is all you need" principle marks a paradigm shift from reconstructing the world to understanding it, laying a foundation for cognitively inspired systems that plan through imagination rather than reaction. 

---
# Gravity-Awareness: Deep Learning Models and LLM Simulation of Human Awareness in Altered Gravity 

**Authors**: Bakytzhan Alibekov, Alina Gutoreva, Elisa Raffaella-Ferre  

**Link**: [PDF](https://arxiv.org/pdf/2511.05536)  

**Abstract**: Earth's gravity has fundamentally shaped human development by guiding the brain's integration of vestibular, visual, and proprioceptive inputs into an internal model of gravity: a dynamic neural representation enabling prediction and interpretation of gravitational forces. This work presents a dual computational framework to quantitatively model these adaptations. The first component is a lightweight Multi-Layer Perceptron (MLP) that predicts g-load-dependent changes in key electroencephalographic (EEG) frequency bands, representing the brain's cortical state. The second component utilizes a suite of independent Gaussian Processes (GPs) to model the body's broader physiological state, including Heart Rate Variability (HRV), Electrodermal Activity (EDA), and motor behavior. Both models were trained on data derived from a comprehensive review of parabolic flight literature, using published findings as anchor points to construct robust, continuous functions. To complement this quantitative analysis, we simulated subjective human experience under different gravitational loads, ranging from microgravity (0g) and partial gravity (Moon 0.17g, Mars 0.38g) to hypergravity associated with spacecraft launch and re-entry (1.8g), using a large language model (Claude 3.5 Sonnet). The model was prompted with physiological parameters to generate introspective narratives of alertness and self-awareness, which closely aligned with the quantitative findings from both the EEG and physiological models. This combined framework integrates quantitative physiological modeling with generative cognitive simulation, offering a novel approach to understanding and predicting human performance in altered gravity 

---
# Selective Diabetic Retinopathy Screening with Accuracy-Weighted Deep Ensembles and Entropy-Guided Abstention 

**Authors**: Jophy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05529)  

**Abstract**: Diabetic retinopathy (DR), a microvascular complication of diabetes and a leading cause of preventable blindness, is projected to affect more than 130 million individuals worldwide by 2030. Early identification is essential to reduce irreversible vision loss, yet current diagnostic workflows rely on methods such as fundus photography and expert review, which remain costly and resource-intensive. This, combined with DR's asymptomatic nature, results in its underdiagnosis rate of approximately 25 percent. Although convolutional neural networks (CNNs) have demonstrated strong performance in medical imaging tasks, limited interpretability and the absence of uncertainty quantification restrict clinical reliability. Therefore, in this study, a deep ensemble learning framework integrated with uncertainty estimation is introduced to improve robustness, transparency, and scalability in DR detection. The ensemble incorporates seven CNN architectures-ResNet-50, DenseNet-121, MobileNetV3 (Small and Large), and EfficientNet (B0, B2, B3)- whose outputs are fused through an accuracy-weighted majority voting strategy. A probability-weighted entropy metric quantifies prediction uncertainty, enabling low-confidence samples to be excluded or flagged for additional review. Training and validation on 35,000 EyePACS retinal fundus images produced an unfiltered accuracy of 93.70 percent (F1 = 0.9376). Uncertainty-filtering later was conducted to remove unconfident samples, resulting in maximum-accuracy of 99.44 percent (F1 = 0.9932). The framework shows that uncertainty-aware, accuracy-weighted ensembling improves reliability without hindering performance. With confidence-calibrated outputs and a tunable accuracy-coverage trade-off, it offers a generalizable paradigm for deploying trustworthy AI diagnostics in high-risk care. 

---
# The Evolution of Probabilistic Price Forecasting Techniques: A Review of the Day-Ahead, Intra-Day, and Balancing Markets 

**Authors**: Ciaran O'Connor, Mohamed Bahloul, Steven Prestwich, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05523)  

**Abstract**: Electricity price forecasting has become a critical tool for decision-making in energy markets, particularly as the increasing penetration of renewable energy introduces greater volatility and uncertainty. Historically, research in this field has been dominated by point forecasting methods, which provide single-value predictions but fail to quantify uncertainty. However, as power markets evolve due to renewable integration, smart grids, and regulatory changes, the need for probabilistic forecasting has become more pronounced, offering a more comprehensive approach to risk assessment and market participation. This paper presents a review of probabilistic forecasting methods, tracing their evolution from Bayesian and distribution based approaches, through quantile regression techniques, to recent developments in conformal prediction. Particular emphasis is placed on advancements in probabilistic forecasting, including validity-focused methods which address key limitations in uncertainty estimation. Additionally, this review extends beyond the Day-Ahead Market to include the Intra-Day and Balancing Markets, where forecasting challenges are intensified by higher temporal granularity and real-time operational constraints. We examine state of the art methodologies, key evaluation metrics, and ongoing challenges, such as forecast validity, model selection, and the absence of standardised benchmarks, providing researchers and practitioners with a comprehensive and timely resource for navigating the complexities of modern electricity markets. 

---
# AIRMap - AI-Generated Radio Maps for Wireless Digital Twins 

**Authors**: Ali Saeizadeh, Miead Tehrani-Moayyed, Davide Villa, J. Gordon Beattie Jr., Pedram Johari, Stefano Basagni, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2511.05522)  

**Abstract**: Accurate, low-latency channel modeling is essential for real-time wireless network simulation and digital-twin applications. Traditional modeling methods like ray tracing are however computationally demanding and unsuited to model dynamic conditions. In this paper, we propose AIRMap, a deep-learning framework for ultra-fast radio-map estimation, along with an automated pipeline for creating the largest radio-map dataset to date. AIRMap uses a single-input U-Net autoencoder that processes only a 2D elevation map of terrain and building heights. Trained and evaluated on 60,000 Boston-area samples, spanning coverage areas from 500 m to 3 km per side, AIRMap predicts path gain with under 5 dB RMSE in 4 ms per inference on an NVIDIA L40S -over 7000x faster than GPU-accelerated ray tracing based radio maps. A lightweight transfer learning calibration using just 20% of field measurements reduces the median error to approximately 10%, significantly outperforming traditional simulators, which exceed 50% error. Integration into the Colosseum emulator and the Sionna SYS platform demonstrate near-zero error in spectral efficiency and block-error rate compared to measurement-based channels. These findings validate AIRMap's potential for scalable, accurate, and real-time radio map estimation in wireless digital twins. 

---
# Retracing the Past: LLMs Emit Training Data When They Get Lost 

**Authors**: Myeongseob Ko, Nikhil Reddy Billa, Adam Nguyen, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.05518)  

**Abstract**: The memorization of training data in large language models (LLMs) poses significant privacy and copyright concerns. Existing data extraction methods, particularly heuristic-based divergence attacks, often exhibit limited success and offer limited insight into the fundamental drivers of memorization leakage. This paper introduces Confusion-Inducing Attacks (CIA), a principled framework for extracting memorized data by systematically maximizing model uncertainty. We empirically demonstrate that the emission of memorized text during divergence is preceded by a sustained spike in token-level prediction entropy. CIA leverages this insight by optimizing input snippets to deliberately induce this consecutive high-entropy state. For aligned LLMs, we further propose Mismatched Supervised Fine-tuning (SFT) to simultaneously weaken their alignment and induce targeted confusion, thereby increasing susceptibility to our attacks. Experiments on various unaligned and aligned LLMs demonstrate that our proposed attacks outperform existing baselines in extracting verbatim and near-verbatim training data without requiring prior knowledge of the training data. Our findings highlight persistent memorization risks across various LLMs and offer a more systematic method for assessing these vulnerabilities. 

---
# Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation 

**Authors**: Canxiang Yan, Chunxiang Jin, Dawei Huang, Haibing Yu, Han Peng, Hui Zhan, Jie Gao, Jing Peng, Jingdong Chen, Jun Zhou, Kaimeng Ren, Ming Yang, Mingxue Yang, Qiang Xu, Qin Zhao, Ruijie Xiong, Shaoxiong Lin, Xuezhi Wang, Yi Yuan, Yifei Wu, Yongjie Lyu, Zhengyu He, Zhihao Qiu, Zhiqiang Fang, Ziyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05516)  

**Abstract**: Existing speech models suffer from competing requirements on token representations by understanding and generation tasks. This discrepancy in representation prevents speech language models from performing instruction-based free-form editing. To solve this challenge, we introduce a novel framework that unifies speech understanding, generation, and editing. The core of our unified model is a unified continuous speech tokenizer MingTok-Audio, the first continuous tokenizer to effectively integrate semantic and acoustic features, which makes it suitable for both understanding and generation tasks. Based on this unified continuous audio tokenizer, we developed the speech language model Ming-UniAudio, which achieved a balance between generation and understanding capabilities. Ming-UniAudio sets new state-of-the-art (SOTA) records on 8 out of 12 metrics on the ContextASR benchmark. Notably, for Chinese voice cloning, it achieves a highly competitive Seed-TTS-WER of 0.95. Leveraging this foundational model, we further trained a dedicated speech editing model Ming-UniAudio-Edit, the first speech language model that enables universal, free-form speech editing guided solely by natural language instructions, handling both semantic and acoustic modifications without timestamp condition. To rigorously assess the editing capability and establish a foundation for future research, we introduce Ming-Freeform-Audio-Edit, the first comprehensive benchmark tailored for instruction-based free-form speech editing, featuring diverse scenarios and evaluation dimensions spanning semantic correctness, acoustic quality, and instruction alignment. We open-sourced the continuous audio tokenizer, the unified foundational model, and the free-form instruction-based editing model to facilitate the development of unified audio understanding, generation, and manipulation. 

---
# From Failure Modes to Reliability Awareness in Generative and Agentic AI System 

**Authors**: Janet, Liangwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05511)  

**Abstract**: This chapter bridges technical analysis and organizational preparedness by tracing the path from layered failure modes to reliability awareness in generative and agentic AI systems. We first introduce an 11-layer failure stack, a structured framework for identifying vulnerabilities ranging from hardware and power foundations to adaptive learning and agentic reasoning. Building on this, the chapter demonstrates how failures rarely occur in isolation but propagate across layers, creating cascading effects with systemic consequences. To complement this diagnostic lens, we develop the concept of awareness mapping: a maturity-oriented framework that quantifies how well individuals and organizations recognize reliability risks across the AI stack. Awareness is treated not only as a diagnostic score but also as a strategic input for AI governance, guiding improvement and resilience planning. By linking layered failures to awareness levels and further integrating this into Dependability-Centred Asset Management (DCAM), the chapter positions awareness mapping as both a measurement tool and a roadmap for trustworthy and sustainable AI deployment across mission-critical domains. 

---
# TEMPO: Temporal Multi-scale Autoregressive Generation of Protein Conformational Ensembles 

**Authors**: Yaoyao Xu, Di Wang, Zihan Zhou, Tianshu Yu, Mingchen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05510)  

**Abstract**: Understanding the dynamic behavior of proteins is critical to elucidating their functional mechanisms, yet generating realistic, temporally coherent trajectories of protein ensembles remains a significant challenge. In this work, we introduce a novel hierarchical autoregressive framework for modeling protein dynamics that leverages the intrinsic multi-scale organization of molecular motions. Unlike existing methods that focus on generating static conformational ensembles or treat dynamic sampling as an independent process, our approach characterizes protein dynamics as a Markovian process. The framework employs a two-scale architecture: a low-resolution model captures slow, collective motions driving major conformational transitions, while a high-resolution model generates detailed local fluctuations conditioned on these large-scale movements. This hierarchical design ensures that the causal dependencies inherent in protein dynamics are preserved, enabling the generation of temporally coherent and physically realistic trajectories. By bridging high-level biophysical principles with state-of-the-art generative modeling, our approach provides an efficient framework for simulating protein dynamics that balances computational efficiency with physical accuracy. 

---
# Randomized-MLP Regularization Improves Domain Adaptation and Interpretability in DINOv2 

**Authors**: Joel Valdivia Ortega, Lorenz Lamm, Franziska Eckardt, Benedikt Schworm, Marion Jasnin, Tingying Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.05509)  

**Abstract**: Vision Transformers (ViTs), such as DINOv2, achieve strong performance across domains but often repurpose low-informative patch tokens in ways that reduce the interpretability of attention and feature maps. This challenge is especially evident in medical imaging, where domain shifts can degrade both performance and transparency. In this paper, we introduce Randomized-MLP (RMLP) regularization, a contrastive learning-based method that encourages more semantically aligned representations. We use RMLPs when fine-tuning DINOv2 to both medical and natural image modalities, showing that it improves or maintains downstream performance while producing more interpretable attention maps. We also provide a mathematical analysis of RMLPs, offering insights into its role in enhancing ViT-based models and advancing our understanding of contrastive learning. 

---
# Personalized Chain-of-Thought Summarization of Financial News for Investor Decision Support 

**Authors**: Tianyi Zhang, Mu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05508)  

**Abstract**: Financial advisors and investors struggle with information overload from financial news, where irrelevant content and noise obscure key market signals and hinder timely investment decisions. To address this, we propose a novel Chain-of-Thought (CoT) summarization framework that condenses financial news into concise, event-driven summaries. The framework integrates user-specified keywords to generate personalized outputs, ensuring that only the most relevant contexts are highlighted. These personalized summaries provide an intermediate layer that supports language models in producing investor-focused narratives, bridging the gap between raw news and actionable insights. 

---
# Rewiring Human Brain Networks via Lightweight Dynamic Connectivity Framework: An EEG-Based Stress Validation 

**Authors**: Sayantan Acharya, Abbas Khosravi, Douglas Creighton, Roohallah Alizadehsani, U. Rajendra Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2511.05505)  

**Abstract**: In recent years, Electroencephalographic analysis has gained prominence in stress research when combined with AI and Machine Learning models for validation. In this study, a lightweight dynamic brain connectivity framework based on Time Varying Directed Transfer Function is proposed, where TV DTF features were validated through ML based stress classification. TV DTF estimates the directional information flow between brain regions across distinct EEG frequency bands, thereby capturing temporal and causal influences that are often overlooked by static functional connectivity measures. EEG recordings from the 32 channel SAM 40 dataset were employed, focusing on mental arithmetic task trials. The dynamic EEG-based TV-DTF features were validated through ML classifiers such as Support Vector Machine, Random Forest, Gradient Boosting, Adaptive Boosting, and Extreme Gradient Boosting. Experimental results show that alpha-TV-DTF provided the strongest discriminative power, with SVM achieving 89.73% accuracy in 3-class classification and with XGBoost achieving 93.69% accuracy in 2 class classification. Relative to absolute power and phase locking based functional connectivity features, alpha TV DTF and beta TV DTF achieved higher performance across the ML models, highlighting the advantages of dynamic over static measures. Feature importance analysis further highlighted dominant long-range frontal parietal and frontal occipital informational influences, emphasizing the regulatory role of frontal regions under stress. These findings validate the lightweight TV-DTF as a robust framework, revealing spatiotemporal brain dynamics and directional influences across different stress levels. 

---
# Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS 

**Authors**: Varun Rajesh, Om Jodhpurkar, Pooja Anbuselvan, Mantinder Singh, Ashok Jallepali, Shantanu Godbole, Pradeep Kumar Sharma, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.05502)  

**Abstract**: We present a systematic, empirical evaluation of five local large language model (LLM) runtimes on Apple Silicon: MLX, MLC-LLM, this http URL, Ollama, and PyTorch MPS. Experiments were conducted on a Mac Studio equipped with an M2 Ultra processor and 192 GB of unified memory. Using the Qwen-2.5 model family across prompts ranging from a few hundred to 100,000 tokens, we measure time-to-first-token (TTFT), steady-state throughput, latency percentiles, long-context behavior (key-value and prompt caching), quantization support, streaming performance, batching and concurrency behavior, and deployment complexity.
Under our settings, MLX achieves the highest sustained generation throughput, while MLC-LLM delivers consistently lower TTFT for moderate prompt sizes and offers stronger out-of-the-box inference features. this http URL is highly efficient for lightweight single-stream use, Ollama emphasizes developer ergonomics but lags in throughput and TTFT, and PyTorch MPS remains limited by memory constraints on large models and long contexts.
All frameworks execute fully on-device with no telemetry, ensuring strong privacy guarantees. We release scripts, logs, and plots to reproduce all results. Our analysis clarifies the design trade-offs in Apple-centric LLM deployments and provides evidence-based recommendations for interactive and long-context processing. Although Apple Silicon inference frameworks still trail NVIDIA GPU-based systems such as vLLM in absolute performance, they are rapidly maturing into viable, production-grade solutions for private, on-device LLM inference. 

---
# Towards Ecologically Valid LLM Benchmarks: Understanding and Designing Domain-Centered Evaluations for Journalism Practitioners 

**Authors**: Charlotte Li, Nick Hagar, Sachita Nishal, Jeremy Gilbert, Nick Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.05501)  

**Abstract**: Benchmarks play a significant role in how researchers and the public understand generative AI systems. However, the widespread use of benchmark scores to communicate about model capabilities has led to criticisms of validity, especially whether benchmarks test what they claim to test (i.e. construct validity) and whether benchmark evaluations are representative of how models are used in the wild (i.e. ecological validity). In this work we explore how to create an LLM benchmark that addresses these issues by taking a human-centered approach. We focus on designing a domain-oriented benchmark for journalism practitioners, drawing on insights from a workshop of 23 journalism professionals. Our workshop findings surface specific challenges that inform benchmark design opportunities, which we instantiate in a case study that addresses underlying criticisms and specific domain concerns. Through our findings and design case study, this work provides design guidance for developing benchmarks that are better tuned to specific domains. 

---
# Predicting Oscar-Nominated Screenplays with Sentence Embeddings 

**Authors**: Francis Gross  

**Link**: [PDF](https://arxiv.org/pdf/2511.05500)  

**Abstract**: Oscar nominations are an important factor in the movie industry because they can boost both the visibility and the commercial success. This work explores whether it is possible to predict Oscar nominations for screenplays using modern language models. Since no suitable dataset was available, a new one called Movie-O-Label was created by combining the MovieSum collection of movie scripts with curated Oscar records. Each screenplay was represented by its title, Wikipedia summary, and full script. Long scripts were split into overlapping text chunks and encoded with the E5 sentence em bedding model. Then, the screenplay embed dings were classified using a logistic regression model. The best results were achieved when three feature inputs related to screenplays (script, summary, and title) were combined. The best-performing model reached a macro F1 score of 0.66, a precision recall AP of 0.445 with baseline 0.19 and a ROC-AUC of 0.79. The results suggest that even simple models based on modern text embeddings demonstrate good prediction performance and might be a starting point for future research. 

---
# Weightless Neural Networks for Continuously Trainable Personalized Recommendation Systems 

**Authors**: Rafayel Latif, Satwik Behera, Ali Al-Ebrahim  

**Link**: [PDF](https://arxiv.org/pdf/2511.05499)  

**Abstract**: Given that conventional recommenders, while deeply effective, rely on large distributed systems pre-trained on aggregate user data, incorporating new data necessitates large training cycles, making them slow to adapt to real-time user feedback and often lacking transparency in recommendation rationale. We explore the performance of smaller personal models trained on per-user data using weightless neural networks (WNNs), an alternative to neural backpropagation that enable continuous learning by using neural networks as a state machine rather than a system with pretrained weights. We contrast our approach against a classic weighted system, also on a per-user level, and standard collaborative filtering, achieving competitive levels of accuracy on a subset of the MovieLens dataset. We close with a discussion of how weightless systems can be developed to augment centralized systems to achieve higher subjective accuracy through recommenders more directly tunable by end-users. 

---
# Biomedical Hypothesis Explainability with Graph-Based Context Retrieval 

**Authors**: Ilya Tyagin, Saeideh Valipour, Aliaksandra Sikirzhytskaya, Michael Shtutman, Ilya Safro  

**Link**: [PDF](https://arxiv.org/pdf/2511.05498)  

**Abstract**: We introduce an explainability method for biomedical hypothesis generation systems, built on top of the novel Hypothesis Generation Context Retriever framework. Our approach combines semantic graph-based retrieval and relevant data-restrictive training to simulate real-world discovery constraints. Integrated with large language models (LLMs) via retrieval-augmented generation, the system explains hypotheses with contextual evidence using published scientific literature. We also propose a novel feedback loop approach, which iteratively identifies and corrects flawed parts of LLM-generated explanations, refining both the evidence paths and supporting context. We demonstrate the performance of our method with multiple large language models and evaluate the explanation and context retrieval quality through both expert-curated assessment and large-scale automated analysis. Our code is available at: this https URL. 

---
# DOCUEVAL: An LLM-based AI Engineering Tool for Building Customisable Document Evaluation Workflows 

**Authors**: Hao Zhang, Qinghua Lu, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05496)  

**Abstract**: Foundation models, such as large language models (LLMs), have the potential to streamline evaluation workflows and improve their performance. However, practical adoption faces challenges, such as customisability, accuracy, and scalability. In this paper, we present DOCUEVAL, an AI engineering tool for building customisable DOCUment EVALuation workflows. DOCUEVAL supports advanced document processing and customisable workflow design which allow users to define theory-grounded reviewer roles, specify evaluation criteria, experiment with different reasoning strategies and choose the assessment style. To ensure traceability, DOCUEVAL provides comprehensive logging of every run, along with source attribution and configuration management, allowing systematic comparison of results across alternative setups. By integrating these capabilities, DOCUEVAL directly addresses core software engineering challenges, including how to determine whether evaluators are "good enough" for deployment and how to empirically compare different evaluation strategies. We demonstrate the usefulness of DOCUEVAL through a real-world academic peer review case, showing how DOCUEVAL enables both the engineering of evaluators and scalable, reliable document evaluation. 

---
# IMDMR: An Intelligent Multi-Dimensional Memory Retrieval System for Enhanced Conversational AI 

**Authors**: Tejas Pawar, Sarika Patil, Om Tilekar, Rushikesh Janwade, Vaibhav Helambe  

**Link**: [PDF](https://arxiv.org/pdf/2511.05495)  

**Abstract**: Conversational AI systems often struggle with maintaining coherent, contextual memory across extended interactions, limiting their ability to provide personalized and contextually relevant responses. This paper presents IMDMR (Intelligent Multi-Dimensional Memory Retrieval), a novel system that addresses these limitations through a multi-dimensional search architecture. Unlike existing memory systems that rely on single-dimensional approaches, IMDMR leverages six distinct memory dimensions-semantic, entity, category, intent, context, and temporal-to provide comprehensive memory retrieval capabilities. Our system incorporates intelligent query processing with dynamic strategy selection, cross-memory entity resolution, and advanced memory integration techniques. Through comprehensive evaluation against five baseline systems including LangChain RAG, LlamaIndex, MemGPT, and spaCy + RAG, IMDMR achieves a 3.8x improvement in overall performance (0.792 vs 0.207 for the best baseline). We present both simulated (0.314) and production (0.792) implementations, demonstrating the importance of real technology integration while maintaining superiority over all baseline systems. Ablation studies demonstrate the effectiveness of multi-dimensional search, with the full system outperforming individual dimension approaches by 23.3%. Query-type analysis reveals superior performance across all categories, particularly for preferences/interests (0.630) and goals/aspirations (0.630) queries. Comprehensive visualizations and statistical analysis confirm the significance of these improvements with p < 0.001 across all metrics. The results establish IMDMR as a significant advancement in conversational AI memory systems, providing a robust foundation for enhanced user interactions and personalized experiences. 

---
# Customized Retrieval-Augmented Generation with LLM for Debiasing Recommendation Unlearning 

**Authors**: Haichao Zhang, Chong Zhang, Peiyu Hu, Shi Qiu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05494)  

**Abstract**: Modern recommender systems face a critical challenge in complying with privacy regulations like the 'right to be forgotten': removing a user's data without disrupting recommendations for others. Traditional unlearning methods address this by partial model updates, but introduce propagation bias--where unlearning one user's data distorts recommendations for behaviorally similar users, degrading system accuracy. While retraining eliminates bias, it is computationally prohibitive for large-scale systems. To address this challenge, we propose CRAGRU, a novel framework leveraging Retrieval-Augmented Generation (RAG) for efficient, user-specific unlearning that mitigates bias while preserving recommendation quality. CRAGRU decouples unlearning into distinct retrieval and generation stages. In retrieval, we employ three tailored strategies designed to precisely isolate the target user's data influence, minimizing collateral impact on unrelated users and enhancing unlearning efficiency. Subsequently, the generation stage utilizes an LLM, augmented with user profiles integrated into prompts, to reconstruct accurate and personalized recommendations without needing to retrain the entire base model. Experiments on three public datasets demonstrate that CRAGRU effectively unlearns targeted user data, significantly mitigating unlearning bias by preventing adverse impacts on non-target users, while maintaining recommendation performance comparable to fully trained original models. Our work highlights the promise of RAG-based architectures for building robust and privacy-preserving recommender systems. The source code is available at: this https URL. 

---
# AI Brown and AI Koditex: LLM-Generated Corpora Comparable to Traditional Corpora of English and Czech Texts 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.22996)  

**Abstract**: This article presents two corpora of English and Czech texts generated with large language models (LLMs). The motivation is to create a resource for comparing human-written texts with LLM-generated text linguistically. Emphasis was placed on ensuring these resources are multi-genre and rich in terms of topics, authors, and text types, while maintaining comparability with existing human-created corpora. These generated corpora replicate reference human corpora: BE21 by Paul Baker, which is a modern version of the original Brown Corpus, and Koditex corpus that also follows the Brown Corpus tradition but in Czech. The new corpora were generated using models from OpenAI, Anthropic, Alphabet, Meta, and DeepSeek, ranging from GPT-3 (davinci-002) to GPT-4.5, and are tagged according to the Universal Dependencies standard (i.e., they are tokenized, lemmatized, and morphologically and syntactically annotated). The subcorpus size varies according to the model used (the English part contains on average 864k tokens per model, 27M tokens altogether, the Czech partcontains on average 768k tokens per model, 21.5M tokens altogether). The corpora are freely available for download under the CC BY 4.0 license (the annotated data are under CC BY-NC-SA 4.0 licence) and are also accessible through the search interface of the Czech National Corpus. 

---
