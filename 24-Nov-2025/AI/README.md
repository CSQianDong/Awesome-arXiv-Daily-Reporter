# SRA-CP: Spontaneous Risk-Aware Selective Cooperative Perception 

**Authors**: Jiaxi Liu, Chengyuan Ma, Hang Zhou, Weizhe Tang, Shixiao Liang, Haoyang Ding, Xiaopeng Li, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2511.17461)  

**Abstract**: Cooperative perception (CP) offers significant potential to overcome the limitations of single-vehicle sensing by enabling information sharing among connected vehicles (CVs). However, existing generic CP approaches need to transmit large volumes of perception data that are irrelevant to the driving safety, exceeding available communication bandwidth. Moreover, most CP frameworks rely on pre-defined communication partners, making them unsuitable for dynamic traffic environments. This paper proposes a Spontaneous Risk-Aware Selective Cooperative Perception (SRA-CP) framework to address these challenges. SRA-CP introduces a decentralized protocol where connected agents continuously broadcast lightweight perception coverage summaries and initiate targeted cooperation only when risk-relevant blind zones are detected. A perceptual risk identification module enables each CV to locally assess the impact of occlusions on its driving task and determine whether cooperation is necessary. When CP is triggered, the ego vehicle selects appropriate peers based on shared perception coverage and engages in selective information exchange through a fusion module that prioritizes safety-critical content and adapts to bandwidth constraints. We evaluate SRA-CP on a public dataset against several representative baselines. Results show that SRA-CP achieves less than 1% average precision (AP) loss for safety-critical objects compared to generic CP, while using only 20% of the communication bandwidth. Moreover, it improves the perception performance by 15% over existing selective CP methods that do not incorporate risk awareness. 

---
# That's not natural: The Impact of Off-Policy Training Data on Probe Performance 

**Authors**: Nathalie Kirch, Samuel Dower, Adrians Skapars, Ekdeep Singh Lubana, Dmitrii Krasheninnikov  

**Link**: [PDF](https://arxiv.org/pdf/2511.17408)  

**Abstract**: Probing has emerged as a promising method for monitoring Large Language Models (LLMs), enabling inference-time detection of concerning behaviours such as deception and sycophancy. However, natural examples of many behaviours are rare, forcing researchers to rely on synthetic or off-policy LLM responses for training probes. We systematically evaluate how the use of synthetic and off-policy data influences probe generalisation across eight distinct LLM behaviours. Testing linear and attention probes across multiple LLMs, we find that the response generation strategy can significantly affect probe performance, though the magnitude of this effect varies by behaviour. We find that successful generalisation from off-policy data, to test sets where the model is incentivised to produce the target behaviour, is predictive of successful on-policy generalisation. Leveraging this result, we predict that Deception and Sandbagging probes may fail to generalise from off-policy to on-policy data when used in real monitoring scenarios. Notably, shifts in the training data domain still cause even larger performance degradation, with different-domain test scores being consistently lower than the same-domain ones. These results indicate that, in the absence of on-policy data, using same-domain off-policy data yields more reliable probes than using on-policy data from a different domain, emphasizing the need for methods that can better handle distribution shifts in LLM monitoring. 

---
# Agentifying Agentic AI 

**Authors**: Virginia Dignum, Frank Dignum  

**Link**: [PDF](https://arxiv.org/pdf/2511.17332)  

**Abstract**: Agentic AI seeks to endow systems with sustained autonomy, reasoning, and interaction capabilities. To realize this vision, its assumptions about agency must be complemented by explicit models of cognition, cooperation, and governance. This paper argues that the conceptual tools developed within the Autonomous Agents and Multi-Agent Systems (AAMAS) community, such as BDI architectures, communication protocols, mechanism design, and institutional modelling, provide precisely such a foundation. By aligning adaptive, data-driven approaches with structured models of reasoning and coordination, we outline a path toward agentic systems that are not only capable and flexible, but also transparent, cooperative, and accountable. The result is a perspective on agency that bridges formal theory and practical autonomy. 

---
# Designing Domain-Specific Agents via Hierarchical Task Abstraction Mechanism 

**Authors**: Kaiyu Li, Jiayu Wang, Zhi Wang, Hui Qiao, Weizhan Zhang, Deyu Meng, Xiangyong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17198)  

**Abstract**: LLM-driven agents, particularly those using general frameworks like ReAct or human-inspired role-playing, often struggle in specialized domains that necessitate rigorously structured workflows. Fields such as remote sensing, requiring specialized tools (e.g., correction, spectral indices calculation), and multi-step procedures (e.g., numerous intermediate products and optional steps), significantly challenge generalized approaches. To address this gap, we introduce a novel agent design framework centered on a Hierarchical Task Abstraction Mechanism (HTAM). Specifically, HTAM moves beyond emulating social roles, instead structuring multi-agent systems into a logical hierarchy that mirrors the intrinsic task-dependency graph of a given domain. This task-centric architecture thus enforces procedural correctness and decomposes complex problems into sequential layers, where each layer's sub-agents operate on the outputs of the preceding layers. We instantiate this framework as EarthAgent, a multi-agent system tailored for complex geospatial analysis. To evaluate such complex planning capabilities, we build GeoPlan-bench, a comprehensive benchmark of realistic, multi-step geospatial planning tasks. It is accompanied by a suite of carefully designed metrics to evaluate tool selection, path similarity, and logical completeness. Experiments show that EarthAgent substantially outperforms a range of established single- and multi-agent systems. Our work demonstrates that aligning agent architecture with a domain's intrinsic task structure is a critical step toward building robust and reliable specialized autonomous systems. 

---
# MIR: Efficient Exploration in Episodic Multi-Agent Reinforcement Learning via Mutual Intrinsic Reward 

**Authors**: Kesheng Chen, Wenjian Luo, Bang Zhang, Zeping Yin, Zipeng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.17165)  

**Abstract**: Episodic rewards present a significant challenge in reinforcement learning. While intrinsic reward methods have demonstrated effectiveness in single-agent rein-forcement learning scenarios, their application to multi-agent reinforcement learn-ing (MARL) remains problematic. The primary difficulties stem from two fac-tors: (1) the exponential sparsity of joint action trajectories that lead to rewards as the exploration space expands, and (2) existing methods often fail to account for joint actions that can influence team states. To address these challenges, this paper introduces Mutual Intrinsic Reward (MIR), a simple yet effective enhancement strategy for MARL with extremely sparse rewards like episodic rewards. MIR incentivizes individual agents to explore actions that affect their teammates, and when combined with original strategies, effectively stimulates team exploration and improves algorithm performance. For comprehensive experimental valida-tion, we extend the representative single-agent MiniGrid environment to create MiniGrid-MA, a series of MARL environments with sparse rewards. Our evalu-ation compares the proposed method against state-of-the-art approaches in the MiniGrid-MA setting, with experimental results demonstrating superior perfor-mance. 

---
# The Belief-Desire-Intention Ontology for modelling mental reality and agency 

**Authors**: Sara Zuppiroli, Carmelo Fabio Longo, Anna Sofia Lippolis, Rocco Paolillo, Lorenzo Giammei, Miguel Ceriani, Francesco Poggi, Antonio Zinilli, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2511.17162)  

**Abstract**: The Belief-Desire-Intention (BDI) model is a cornerstone for representing rational agency in artificial intelligence and cognitive sciences. Yet, its integration into structured, semantically interoperable knowledge representations remains limited. This paper presents a formal BDI Ontology, conceived as a modular Ontology Design Pattern (ODP) that captures the cognitive architecture of agents through beliefs, desires, intentions, and their dynamic interrelations. The ontology ensures semantic precision and reusability by aligning with foundational ontologies and best practices in modular design. Two complementary lines of experimentation demonstrate its applicability: (i) coupling the ontology with Large Language Models (LLMs) via Logic Augmented Generation (LAG) to assess the contribution of ontological grounding to inferential coherence and consistency; and (ii) integrating the ontology within the Semas reasoning platform, which implements the Triples-to-Beliefs-to-Triples (T2B2T) paradigm, enabling a bidirectional flow between RDF triples and agent mental states. Together, these experiments illustrate how the BDI Ontology acts as both a conceptual and operational bridge between declarative and procedural intelligence, paving the way for cognitively grounded, explainable, and semantically interoperable multi-agent and neuro-symbolic systems operating within the Web of Data. 

---
# Patient-level Information Extraction by Consistent Integration of Textual and Tabular Evidence with Bayesian Networks 

**Authors**: Paloma Rabaey, Adrick Tench, Stefan Heytens, Thomas Demeester  

**Link**: [PDF](https://arxiv.org/pdf/2511.17056)  

**Abstract**: Electronic health records (EHRs) form an invaluable resource for training clinical decision support systems. To leverage the potential of such systems in high-risk applications, we need large, structured tabular datasets on which we can build transparent feature-based models. While part of the EHR already contains structured information (e.g. diagnosis codes, medications, and lab results), much of the information is contained within unstructured text (e.g. discharge summaries and nursing notes). In this work, we propose a method for multi-modal patient-level information extraction that leverages both the tabular features available in the patient's EHR (using an expert-informed Bayesian network) as well as clinical notes describing the patient's symptoms (using neural text classifiers). We propose the use of virtual evidence augmented with a consistency node to provide an interpretable, probabilistic fusion of the models' predictions. The consistency node improves the calibration of the final predictions compared to virtual evidence alone, allowing the Bayesian network to better adjust the neural classifier's output to handle missing information and resolve contradictions between the tabular and text data. We show the potential of our method on the SimSUM dataset, a simulated benchmark linking tabular EHRs with clinical notes through expert knowledge. 

---
# DAPS++: Rethinking Diffusion Inverse Problems with Decoupled Posterior Annealing 

**Authors**: Hao Chen, Renzheng Zhang, Scott S. Howard  

**Link**: [PDF](https://arxiv.org/pdf/2511.17038)  

**Abstract**: From a Bayesian perspective, score-based diffusion solves inverse problems through joint inference, embedding the likelihood with the prior to guide the sampling process. However, this formulation fails to explain its practical behavior: the prior offers limited guidance, while reconstruction is largely driven by the measurement-consistency term, leading to an inference process that is effectively decoupled from the diffusion dynamics. To clarify this structure, we reinterpret the role of diffusion in inverse problem solving as an initialization stage within an expectation--maximization (EM)--style framework, where the diffusion stage and the data-driven refinement are fully decoupled. We introduce \textbf{DAPS++}, which allows the likelihood term to guide inference more directly while maintaining numerical stability and providing insight into why unified diffusion trajectories remain effective in practice. By requiring fewer function evaluations (NFEs) and measurement-optimization steps, \textbf{DAPS++} achieves high computational efficiency and robust reconstruction performance across diverse image restoration tasks. 

---
# Budget-Aware Tool-Use Enables Effective Agent Scaling 

**Authors**: Tengxiao Liu, Zifeng Wang, Jin Miao, I-Hung Hsu, Jun Yan, Jiefeng Chen, Rujun Han, Fangyuan Xu, Yanfei Chen, Ke Jiang, Samira Daruki, Yi Liang, William Yang Wang, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.17006)  

**Abstract**: Scaling test-time computation improves performance across different tasks on large language models (LLMs), which has also been extended to tool-augmented agents. For these agents, scaling involves not only "thinking" in tokens but also "acting" via tool calls. The number of tool calls directly bounds the agent's interaction with the external environment. However, we find that simply granting agents a larger tool-call budget fails to improve performance, as they lack "budget awareness" and quickly hit a performance ceiling. To address this, we study how to scale such agents effectively under explicit tool-call budgets, focusing on web search agents. We first introduce the Budget Tracker, a lightweight plug-in that provides the agent with continuous budget awareness, enabling simple yet effective scaling. We further develop BATS (Budget Aware Test-time Scaling), an advanced framework that leverages this awareness to dynamically adapt its planning and verification strategy, deciding whether to "dig deeper" on a promising lead or "pivot" to new paths based on remaining resources. To analyze cost-performance scaling in a controlled manner, we formalize a unified cost metric that jointly accounts for token and tool consumption. We provide the first systematic study on budget-constrained agents, showing that budget-aware methods produce more favorable scaling curves and push the cost-performance Pareto frontier. Our work offers empirical insights toward a more transparent and principled understanding of scaling in tool-augmented agents. 

---
# MirrorMind: Empowering OmniScientist with the Expert Perspectives and Collective Knowledge of Human Scientists 

**Authors**: Qingbin Zeng, Bingbing Fan, Zhiyu Chen, Sijian Ren, Zhilun Zhou, Xuhua Zhang, Yuanyi Zhen, Fengli Xu, Yong Li, Tie-Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16997)  

**Abstract**: The emergence of AI Scientists has demonstrated remarkable potential in automating scientific research. However, current approaches largely conceptualize scientific discovery as a solitary optimization or search process, overlooking that knowledge production is inherently a social and historical endeavor. Human scientific insight stems from two distinct yet interconnected sources. First is the individual cognitive trajectory, where a researcher's unique insight is shaped by their evolving research history and stylistic preferences; another is the collective disciplinary memory, where knowledge is sedimented into vast, interconnected networks of citations and concepts. Existing LLMs still struggle to represent these structured, high-fidelity cognitive and social contexts. To bridge this gap, we introduce MirrorMind, a hierarchical cognitive architecture that integrates dual-memory representations within a three-level framework. The Individual Level constructs high-fidelity cognitive models of individual researchers by capturing their episodic, semantic, and persona memories; the Domain Level maps collective knowledge into structured disciplinary concept graphs; and the Interdisciplinary Level that acts as an orthogonal orchestration engine. Crucially, our architecture separates memory storage from agentic execution, enabling AI scientist agents to flexibly access individual memories for unique perspectives or collective structures to reason. We evaluate MirrorMind across four comprehensive tasks, including author-level cognitive simulation, complementary reasoning, cross-disciplinary collaboration promotion, and multi-agent scientific problem solving. The results show that by integrating individual cognitive depth with collective disciplinary breadth, MirrorMind moves beyond simple fact retrieval toward structural, personalized, and insight-generating scientific reasoning. 

---
# Comparing verbal, visual and combined explanations for Bayesian Network inferences 

**Authors**: Erik P. Nyberg, Steven Mascaro, Ingrid Zukerman, Michael Wybrow, Duc-Minh Vo, Ann Nicholson  

**Link**: [PDF](https://arxiv.org/pdf/2511.16961)  

**Abstract**: Bayesian Networks (BNs) are an important tool for assisting probabilistic reasoning, but despite being considered transparent models, people have trouble understanding them. Further, current User Interfaces (UIs) still do not clarify the reasoning of BNs. To address this problem, we have designed verbal and visual extensions to the standard BN UI, which can guide users through common inference patterns.
We conducted a user study to compare our verbal, visual and combined UI extensions, and a baseline UI. Our main findings are: (1) users did better with all three types of extensions than with the baseline UI for questions about the impact of an observation, the paths that enable this impact, and the way in which an observation influences the impact of other observations; and (2) using verbal and visual modalities together is better than using either modality alone for some of these question types. 

---
# Hybrid Differential Reward: Combining Temporal Difference and Action Gradients for Efficient Multi-Agent Reinforcement Learning in Cooperative Driving 

**Authors**: Ye Han, Lijun Zhang, Dejian Meng, Zhuang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16916)  

**Abstract**: In multi-vehicle cooperative driving tasks involving high-frequency continuous control, traditional state-based reward functions suffer from the issue of vanishing reward differences. This phenomenon results in a low signal-to-noise ratio (SNR) for policy gradients, significantly hindering algorithm convergence and performance improvement. To address this challenge, this paper proposes a novel Hybrid Differential Reward (HDR) mechanism. We first theoretically elucidate how the temporal quasi-steady nature of traffic states and the physical proximity of actions lead to the failure of traditional reward signals. Building on this analysis, the HDR framework innovatively integrates two complementary components: (1) a Temporal Difference Reward (TRD) based on a global potential function, which utilizes the evolutionary trend of potential energy to ensure optimal policy invariance and consistency with long-term objectives; and (2) an Action Gradient Reward (ARG), which directly measures the marginal utility of actions to provide a local guidance signal with a high SNR. Furthermore, we formulate the cooperative driving problem as a Multi-Agent Partially Observable Markov Game (POMDPG) with a time-varying agent set and provide a complete instantiation scheme for HDR within this framework. Extensive experiments conducted using both online planning (MCTS) and Multi-Agent Reinforcement Learning (QMIX, MAPPO, MADDPG) algorithms demonstrate that the HDR mechanism significantly improves convergence speed and policy stability. The results confirm that HDR guides agents to learn high-quality cooperative policies that effectively balance traffic efficiency and safety. 

---
# Fantastic Bugs and Where to Find Them in AI Benchmarks 

**Authors**: Sang Truong, Yuheng Tu, Michael Hardy, Anka Reuel, Zeyu Tang, Jirayu Burapacheep, Jonathan Perera, Chibuike Uwakwe, Ben Domingue, Nick Haber, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2511.16842)  

**Abstract**: Benchmarks are pivotal in driving AI progress, and invalid benchmark questions frequently undermine their reliability. Manually identifying and correcting errors among thousands of benchmark questions is not only infeasible but also a critical bottleneck for reliable evaluation. In this work, we introduce a framework for systematic benchmark revision that leverages statistical analysis of response patterns to flag potentially invalid questions for further expert review. Our approach builds on a core assumption commonly used in AI evaluations that the mean score sufficiently summarizes model performance. This implies a unidimensional latent construct underlying the measurement experiment, yielding expected ranges for various statistics for each item. When empirically estimated values for these statistics fall outside the expected range for an item, the item is more likely to be problematic. Across nine widely used benchmarks, our method guides expert review to identify problematic questions with up to 84\% precision. In addition, we introduce an LLM-judge first pass to review questions, further reducing human effort. Together, these components provide an efficient and scalable framework for systematic benchmark revision. 

---
# Cognitive BASIC: An In-Model Interpreted Reasoning Language for LLMs 

**Authors**: Oliver Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2511.16837)  

**Abstract**: Cognitive BASIC is a minimal, BASIC-style prompting language and in-model interpreter that structures large language model (LLM) reasoning into explicit, stepwise execution traces. Inspired by the simplicity of retro BASIC, we repurpose numbered lines and simple commands as an interpretable cognitive control layer. Modern LLMs can reliably simulate such short programs, enabling transparent multi-step reasoning inside the model. A natural-language interpreter file specifies command semantics, memory updates, and logging behavior. Our mental-model interpreter extracts declarative and procedural knowledge, detects contradictions, and produces resolutions when necessary. A comparison across three LLMs on a benchmark of knowledge extraction, conflict detection, and reasoning tasks shows that all models can execute Cognitive BASIC programs, with overall strong but not uniform performance. 

---
# Stable diffusion models reveal a persisting human and AI gap in visual creativity 

**Authors**: Silvia Rondini, Claudia Alvarez-Martin, Paula Angermair-Barkai, Olivier Penacchio, M. Paz, Matthew Pelowski, Dan Dediu, Antoni Rodriguez-Fornells, Xim Cerda-Company  

**Link**: [PDF](https://arxiv.org/pdf/2511.16814)  

**Abstract**: While recent research suggests Large Language Models match human creative performance in divergent thinking tasks, visual creativity remains underexplored. This study compared image generation in human participants (Visual Artists and Non Artists) and using an image generation AI model (two prompting conditions with varying human input: high for Human Inspired, low for Self Guided). Human raters (N=255) and GPT4o evaluated the creativity of the resulting images. We found a clear creativity gradient, with Visual Artists being the most creative, followed by Non Artists, then Human Inspired generative AI, and finally Self Guided generative AI. Increased human guidance strongly improved GenAI's creative output, bringing its productions close to those of Non Artists. Notably, human and AI raters also showed vastly different creativity judgment patterns. These results suggest that, in contrast to language centered tasks, GenAI models may face unique challenges in visual domains, where creativity depends on perceptual nuance and contextual sensitivity, distinctly human capacities that may not be readily transferable from language models. 

---
# Enhancing Quranic Learning: A Multimodal Deep Learning Approach for Arabic Phoneme Recognition 

**Authors**: Ayhan Kucukmanisa, Derya Gelmez, Sukru Selim Calik, Zeynep Hilal Kilimci  

**Link**: [PDF](https://arxiv.org/pdf/2511.17477)  

**Abstract**: Recent advances in multimodal deep learning have greatly enhanced the capability of systems for speech analysis and pronunciation assessment. Accurate pronunciation detection remains a key challenge in Arabic, particularly in the context of Quranic recitation, where subtle phonetic differences can alter meaning. Addressing this challenge, the present study proposes a transformer-based multimodal framework for Arabic phoneme mispronunciation detection that combines acoustic and textual representations to achieve higher precision and robustness. The framework integrates UniSpeech-derived acoustic embeddings with BERT-based textual embeddings extracted from Whisper transcriptions, creating a unified representation that captures both phonetic detail and linguistic context. To determine the most effective integration strategy, early, intermediate, and late fusion methods were implemented and evaluated on two datasets containing 29 Arabic phonemes, including eight hafiz sounds, articulated by 11 native speakers. Additional speech samples collected from publicly available YouTube recordings were incorporated to enhance data diversity and generalization. Model performance was assessed using standard evaluation metrics: accuracy, precision, recall, and F1-score, allowing a detailed comparison of the fusion strategies. Experimental findings show that the UniSpeech-BERT multimodal configuration provides strong results and that fusion-based transformer architectures are effective for phoneme-level mispronunciation detection. The study contributes to the development of intelligent, speaker-independent, and multimodal Computer-Aided Language Learning (CALL) systems, offering a practical step toward technology-supported Quranic pronunciation training and broader speech-based educational applications. 

---
# Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards 

**Authors**: Zhen Wang, Zhifeng Gao, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2511.17473)  

**Abstract**: Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings. 

---
# PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM 

**Authors**: Siqi Liang, Yudi Zhang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.17467)  

**Abstract**: We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at this https URL 

---
# Planning with Sketch-Guided Verification for Physics-Aware Video Generation 

**Authors**: Yidong Huang, Zun Wang, Han Lin, Dong-Ki Kim, Shayegan Omidshafiei, Jaehong Yoon, Yue Zhang, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.17450)  

**Abstract**: Recent video generation approaches increasingly rely on planning intermediate control signals such as object trajectories to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motions, or iterative refinement which requires multiple calls to the video generator, incuring high computational cost. To overcome these limitations, we propose SketchVerify, a training-free, sketch-verification-based planning framework that improves motion planning quality with more dynamically coherent trajectories (i.e., physically plausible and instruction-consistent motions) prior to full video generation by introducing a test-time sampling and verification loop. Given a prompt and a reference image, our method predicts multiple candidate motion plans and ranks them using a vision-language verifier that jointly evaluates semantic alignment with the instruction and physical plausibility. To efficiently score candidate motion plans, we render each trajectory as a lightweight video sketch by compositing objects over a static background, which bypasses the need for expensive, repeated diffusion-based synthesis while achieving comparable performance. We iteratively refine the motion plan until a satisfactory one is identified, which is then passed to the trajectory-conditioned generator for final synthesis. Experiments on WorldModelBench and PhyWorldBench demonstrate that our method significantly improves motion quality, physical realism, and long-term consistency compared to competitive baselines while being substantially more efficient. Our ablation study further shows that scaling up the number of trajectory candidates consistently enhances overall performance. 

---
# GRAPHIC--Guidelines for Reviewing Algorithmic Practices in Human-centred Design and Interaction for Creativity 

**Authors**: Joana Rovira Martins, Pedro Martins, Ana Boavida  

**Link**: [PDF](https://arxiv.org/pdf/2511.17443)  

**Abstract**: Artificial Intelligence (AI) has been increasingly applied to creative domains, leading to the development of systems that collaborate with humans in design processes. In Graphic Design, integrating computational systems into co-creative workflows presents specific challenges, as it requires balancing scientific rigour with the subjective and visual nature of design practice. Following the PRISMA methodology, we identified 872 articles, resulting in a final corpus of 71 publications describing 68 unique systems. Based on this review, we introduce GRAPHIC (Guidelines for Reviewing Algorithmic Practices in Human-centred Design and Interaction for Creativity), a framework for analysing AI-based systems applied to Graphic Design. Its goal is to understand how current systems support human-AI collaboration in the Graphic Design discipline. The framework comprises main dimensions, which our analysis revealed to be essential across diverse system types: (1) Collaborative Panorama, (2) Processes and Modalities, and (3) Graphic Design Principles. Its application revealed research gaps, including the need to balance initiative and control between agents, improve communication through explainable interaction models, and promote systems that support transformational creativity grounded in core design principles. 

---
# REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing 

**Authors**: Binger Chen, Tacettin Emre Bök, Behnood Rasti, Volker Markl, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2511.17442)  

**Abstract**: Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data. 

---
# InTAct: Interval-based Task Activation Consolidation for Continual Learning 

**Authors**: Patryk Krukowski, Jan Miksa, Piotr Helm, Jacek Tabor, Paweł Wawrzyński, Przemysław Spurek  

**Link**: [PDF](https://arxiv.org/pdf/2511.17439)  

**Abstract**: Continual learning aims to enable neural networks to acquire new knowledge without forgetting previously learned information. While recent prompt-based methods perform strongly in class-incremental settings, they remain vulnerable under domain shifts, where the input distribution changes but the label space remains fixed. This exposes a persistent problem known as representation drift. Shared representations evolve in ways that overwrite previously useful features and cause forgetting even when prompts isolate task-specific parameters. To address this issue, we introduce InTAct, a method that preserves functional behavior in shared layers without freezing parameters or storing past data. InTAct captures the characteristic activation ranges associated with previously learned tasks and constrains updates to ensure the network remains consistent within these regions, while still allowing for flexible adaptation elsewhere. In doing so, InTAct stabilizes the functional role of important neurons rather than directly restricting parameter values. The approach is architecture-agnostic and integrates seamlessly into existing prompt-based continual learning frameworks. By regulating representation changes where past knowledge is encoded, InTAct achieves a principled balance between stability and plasticity. Across diverse domain-incremental benchmarks, including DomainNet and ImageNet-R, InTAct consistently reduces representation drift and improves performance, increasing Average Accuracy by up to 8 percentage points over state-of-the-art baselines. 

---
# SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation 

**Authors**: Shrikant Kendre, Austin Xu, Honglu Zhou, Michael Ryoo, Shafiq Joty, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2511.17432)  

**Abstract**: Traditional evaluation metrics for textual and visual question answering, like ROUGE, METEOR, and Exact Match (EM), focus heavily on n-gram based lexical similarity, often missing the deeper semantic understanding needed for accurate assessment. While measures like BERTScore and MoverScore leverage contextual embeddings to address this limitation, they lack flexibility in balancing sentence-level and keyword-level semantics and ignore lexical similarity, which remains important. Large Language Model (LLM) based evaluators, though powerful, come with drawbacks like high costs, bias, inconsistency, and hallucinations. To address these issues, we introduce SMILE: Semantic Metric Integrating Lexical Exactness, a novel approach that combines sentence-level semantic understanding with keyword-level semantic understanding and easy keyword matching. This composite method balances lexical precision and semantic relevance, offering a comprehensive evaluation. Extensive benchmarks across text, image, and video QA tasks show SMILE is highly correlated with human judgments and computationally lightweight, bridging the gap between lexical and semantic evaluation. 

---
# Preventing Shortcut Learning in Medical Image Analysis through Intermediate Layer Knowledge Distillation from Specialist Teachers 

**Authors**: Christopher Boland, Sotirios Tsaftaris, Sonia Dahdouh  

**Link**: [PDF](https://arxiv.org/pdf/2511.17421)  

**Abstract**: Deep learning models are prone to learning shortcut solutions to problems using spuriously correlated yet irrelevant features of their training data. In high-risk applications such as medical image analysis, this phenomenon may prevent models from using clinically meaningful features when making predictions, potentially leading to poor robustness and harm to patients. We demonstrate that different types of shortcuts (those that are diffuse and spread throughout the image, as well as those that are localized to specific areas) manifest distinctly across network layers and can, therefore, be more effectively targeted through mitigation strategies that target the intermediate layers. We propose a novel knowledge distillation framework that leverages a teacher network fine-tuned on a small subset of task-relevant data to mitigate shortcut learning in a student network trained on a large dataset corrupted with a bias feature. Through extensive experiments on CheXpert, ISIC 2017, and SimBA datasets using various architectures (ResNet-18, AlexNet, DenseNet-121, and 3D CNNs), we demonstrate consistent improvements over traditional Empirical Risk Minimization, augmentation-based bias-mitigation, and group-based bias-mitigation approaches. In many cases, we achieve comparable performance with a baseline model trained on bias-free data, even on out-of-distribution test data. Our results demonstrate the practical applicability of our approach to real-world medical imaging scenarios where bias annotations are limited and shortcut features are difficult to identify a priori. 

---
# DS-Span: Single-Phase Discriminative Subgraph Mining for Efficient Graph Embeddings 

**Authors**: Yeamin Kaiser, Muhammed Tasnim Bin Anwar, Bholanath Das, Chowdhury Farhan Ahmed, Md. Tanvir Alam  

**Link**: [PDF](https://arxiv.org/pdf/2511.17419)  

**Abstract**: Graph representation learning seeks to transform complex, high-dimensional graph structures into compact vector spaces that preserve both topology and semantics. Among the various strategies, subgraph-based methods provide an interpretable bridge between symbolic pattern discovery and continuous embedding learning. Yet, existing frequent or discriminative subgraph mining approaches often suffer from redundant multi-phase pipelines, high computational cost, and weak coupling between mined structures and their discriminative relevance. We propose DS-Span, a single-phase discriminative subgraph mining framework that unifies pattern growth, pruning, and supervision-driven scoring within one traversal of the search space. DS-Span introduces a coverage-capped eligibility mechanism that dynamically limits exploration once a graph is sufficiently represented, and an information-gain-guided selection that promotes subgraphs with strong class-separating ability while minimizing redundancy. The resulting subgraph set serves as an efficient, interpretable basis for downstream graph embedding and classification. Extensive experiments across benchmarks demonstrate that DS-Span generates more compact and discriminative subgraph features than prior multi-stage methods, achieving higher or comparable accuracy with significantly reduced runtime. These results highlight the potential of unified, single-phase discriminative mining as a foundation for scalable and interpretable graph representation learning. 

---
# Beyond Multiple Choice: A Hybrid Framework for Unifying Robust Evaluation and Verifiable Reasoning Training 

**Authors**: Yesheng Liu, Hao Li, Haiyu Xu, Baoqi Pei, Jiahao Wang, Mingxuan Zhao, Jingshu Zheng, Zheqi He, JG Yao, Bowen Qin, Xi Yang, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17405)  

**Abstract**: Multiple-choice question answering (MCQA) has been a popular format for evaluating and reinforcement fine-tuning (RFT) of modern multimodal language models. Its constrained output format allows for simplified, deterministic automatic verification. However, we find that the options may leak exploitable signals, which makes the accuracy metrics unreliable for indicating real capabilities and encourages explicit or implicit answer guessing behaviors during RFT. We propose ReVeL (Rewrite and Verify by LLM), a framework that rewrites multiple-choice questions into open-form questions while keeping answers verifiable whenever possible. The framework categorizes questions according to different answer types, apply different rewriting and verification schemes, respectively. When applied for RFT, we converted 20k MCQA examples and use GRPO to finetune Qwen2.5-VL models. Models trained on ReVeL-OpenQA match MCQA accuracy on multiple-choice benchmarks and improve OpenQA accuracy by about six percentage points, indicating better data efficiency and more robust reward signals than MCQA-based training. When used for evaluation, ReVeL also reveals up to 20 percentage points of score inflation in MCQA benchmarks (relative to OpenQA), improves judging accuracy, and reduces both cost and latency. We will release code and data publicly. 

---
# Sparse Mixture-of-Experts for Multi-Channel Imaging: Are All Channel Interactions Required? 

**Authors**: Sukwon Yun, Heming Yao, Burkhard Hoeckendorf, David Richmond, Aviv Regev, Russell Littman  

**Link**: [PDF](https://arxiv.org/pdf/2511.17400)  

**Abstract**: Vision Transformers ($\text{ViTs}$) have become the backbone of vision foundation models, yet their optimization for multi-channel domains - such as cell painting or satellite imagery - remains underexplored. A key challenge in these domains is capturing interactions between channels, as each channel carries different information. While existing works have shown efficacy by treating each channel independently during tokenization, this approach naturally introduces a major computational bottleneck in the attention block - channel-wise comparisons leads to a quadratic growth in attention, resulting in excessive $\text{FLOPs}$ and high training cost. In this work, we shift focus from efficacy to the overlooked efficiency challenge in cross-channel attention and ask: "Is it necessary to model all channel interactions?". Inspired by the philosophy of Sparse Mixture-of-Experts ($\text{MoE}$), we propose MoE-ViT, a Mixture-of-Experts architecture for multi-channel images in $\text{ViTs}$, which treats each channel as an expert and employs a lightweight router to select only the most relevant experts per patch for attention. Proof-of-concept experiments on real-world datasets - JUMP-CP and So2Sat - demonstrate that $\text{MoE-ViT}$ achieves substantial efficiency gains without sacrificing, and in some cases enhancing, performance, making it a practical and attractive backbone for multi-channel imaging. 

---
# Designing and Generating Diverse, Equitable Face Image Datasets for Face Verification Tasks 

**Authors**: Georgia Baltsou, Ioannis Sarridis, Christos Koutlis, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.17393)  

**Abstract**: Face verification is a significant component of identity authentication in various applications including online banking and secure access to personal devices. The majority of the existing face image datasets often suffer from notable biases related to race, gender, and other demographic characteristics, limiting the effectiveness and fairness of face verification systems. In response to these challenges, we propose a comprehensive methodology that integrates advanced generative models to create varied and diverse high-quality synthetic face images. This methodology emphasizes the representation of a diverse range of facial traits, ensuring adherence to characteristics permissible in identity card photographs. Furthermore, we introduce the Diverse and Inclusive Faces for Verification (DIF-V) dataset, comprising 27,780 images of 926 unique identities, designed as a benchmark for future research in face verification. Our analysis reveals that existing verification models exhibit biases toward certain genders and races, and notably, applying identity style modifications negatively impacts model performance. By tackling the inherent inequities in existing datasets, this work not only enriches the discussion on diversity and ethics in artificial intelligence but also lays the foundation for developing more inclusive and reliable face verification technologies 

---
# Quantum Masked Autoencoders for Vision Learning 

**Authors**: Emma Andrews, Prabhat Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2511.17372)  

**Abstract**: Classical autoencoders are widely used to learn features of input data. To improve the feature learning, classical masked autoencoders extend classical autoencoders to learn the features of the original input sample in the presence of masked-out data. While quantum autoencoders exist, there is no design and implementation of quantum masked autoencoders that can leverage the benefits of quantum computing and quantum autoencoders. In this paper, we propose quantum masked autoencoders (QMAEs) that can effectively learn missing features of a data sample within quantum states instead of classical embeddings. We showcase that our QMAE architecture can learn the masked features of an image and can reconstruct the masked input image with improved visual fidelity in MNIST images. Experimental evaluation highlights that QMAE can significantly outperform (12.86% on average) in classification accuracy compared to state-of-the-art quantum autoencoders in the presence of masks. 

---
# Is Phase Really Needed for Weakly-Supervised Dereverberation ? 

**Authors**: Marius Rodrigues, Louis Bahrman, Roland Badeau, Gaël Richard  

**Link**: [PDF](https://arxiv.org/pdf/2511.17346)  

**Abstract**: In unsupervised or weakly-supervised approaches for speech dereverberation, the target clean (dry) signals are considered to be unknown during training. In that context, evaluating to what extent information can be retrieved from the sole knowledge of reverberant (wet) speech becomes critical. This work investigates the role of the reverberant (wet) phase in the time-frequency domain. Based on Statistical Wave Field Theory, we show that late reverberation perturbs phase components with white, uniformly distributed noise, except at low frequencies. Consequently, the wet phase carries limited useful information and is not essential for weakly supervised dereverberation. To validate this finding, we train dereverberation models under a recent weak supervision framework and demonstrate that performance can be significantly improved by excluding the reverberant phase from the loss function. 

---
# AI Workers, Geopolitics, and Algorithmic Collective Action 

**Authors**: Sydney Reis  

**Link**: [PDF](https://arxiv.org/pdf/2511.17331)  

**Abstract**: According to the theory of International Political Economy (IPE), states are often incentivized to rely on rather than constrain powerful corporations. For this reason, IPE provides a useful lens to explain why efforts to govern Artificial Intelligence (AI) at the international and national levels have thus far been developed, applied, and enforced unevenly. Building on recent work that explores how AI companies engage in geopolitics, this position paper argues that some AI workers can be considered actors of geopolitics. It makes the timely case that governance alone cannot ensure responsible, ethical, or robust AI development and use, and greater attention should be paid to bottom-up interventions at the site of AI development. AI workers themselves should be situated as individual agents of change, especially when considering their potential to foster Algorithmic Collective Action (ACA). Drawing on methods of Participatory Design (PD), this paper proposes engaging AI workers as sources of knowledge, relative power, and intentionality to encourage more responsible and just AI development and create the conditions that can facilitate ACA. 

---
# MusicAIR: A Multimodal AI Music Generation Framework Powered by an Algorithm-Driven Core 

**Authors**: Callie C. Liao, Duoduo Liao, Ellie L. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17323)  

**Abstract**: Recent advances in generative AI have made music generation a prominent research focus. However, many neural-based models rely on large datasets, raising concerns about copyright infringement and high-performance costs. In contrast, we propose MusicAIR, an innovative multimodal AI music generation framework powered by a novel algorithm-driven symbolic music core, effectively mitigating copyright infringement risks. The music core algorithms connect critical lyrical and rhythmic information to automatically derive musical features, creating a complete, coherent melodic score solely from the lyrics. The MusicAIR framework facilitates music generation from lyrics, text, and images. The generated score adheres to established principles of music theory, lyrical structure, and rhythmic conventions. We developed Generate AI Music (GenAIM), a web tool using MusicAIR for lyric-to-song, text-to-music, and image-to-music generation. In our experiments, we evaluated AI-generated music scores produced by the system using both standard music metrics and innovative analysis that compares these compositions with original works. The system achieves an average key confidence of 85%, outperforming human composers at 79%, and aligns closely with established music theory standards, demonstrating its ability to generate diverse, human-like compositions. As a co-pilot tool, GenAIM can serve as a reliable music composition assistant and a possible educational composition tutor while simultaneously lowering the entry barrier for all aspiring musicians, which is innovative and significantly contributes to AI for music generation. 

---
# FORWARD: Dataset of a forwarder operating in rough terrain 

**Authors**: Mikael Lundbäck, Erik Wallin, Carola Häggström, Mattias Nyström, Andreas Grönlund, Mats Richardson, Petrus Jönsson, William Arnvik, Lucas Hedström, Arvid Fälldin, Martin Servin  

**Link**: [PDF](https://arxiv.org/pdf/2511.17318)  

**Abstract**: We present FORWARD, a high-resolution multimodal dataset of a cut-to-length forwarder operating in rough terrain on two harvest sites in the middle part of Sweden. The forwarder is a large Komatsu model equipped with a variety of sensors, including RTK-GNSS, 360-camera, operator vibration sensors, internal CAN-bus signal recording, and multiple IMUs. The data includes event time logs recorded in 5 Hz with e.g., driving speed, fuel consumption, vehicle position with centimeter accuracy, and crane use while the vehicle operates in forest areas laser-scanned with very high-resolution, $\sim$1500 points per square meter. Production log files (StanForD standard) with time-stamped machine events, extensive video material, and terrain data in various formats are included as well. About 18 hours of regular wood extraction work during three days is annotated from 360-video material into individual work elements and included in the dataset. We also include scenario specifications of conducted experiments on forest roads and in terrain. Scenarios include repeatedly driving the same routes with and without steel tracks, different load weight, and different target driving speeds. The dataset is intended for developing models and algorithms for trafficability, perception, and autonomous control of forest machines using artificial intelligence, simulation, and experiments on physical testbeds. In part, we focus on forwarders traversing terrain, avoiding obstacles, and loading or unloading logs, with consideration for efficiency, fuel consumption, safety, and environmental impact. Other benefits of the open dataset include the ability to explore auto-generation and calibration of forestry machine simulators and automation scenario descriptions using the data recorded in the field. 

---
# MuM: Multi-View Masked Image Modeling for 3D Vision 

**Authors**: David Nordström, Johan Edstedt, Fredrik Kahl, Georg Bökman  

**Link**: [PDF](https://arxiv.org/pdf/2511.17309)  

**Abstract**: Self-supervised learning on images seeks to extract meaningful visual representations from unlabeled data. When scaled to large datasets, this paradigm has achieved state-of-the-art performance and the resulting trained models such as DINOv3 have seen widespread adoption. However, most prior efforts are optimized for semantic understanding rather than geometric reasoning. One important exception is Cross-View Completion, CroCo, which is a form of masked autoencoding (MAE) tailored for 3D understanding. In this work, we continue on the path proposed by CroCo and focus on learning features tailored for 3D vision. In a nutshell, we extend MAE to arbitrarily many views of the same scene. By uniformly masking all views and employing a lightweight decoder with inter-frame attention, our approach is inherently simpler and more scalable than CroCo. We evaluate the resulting model, MuM, extensively on downstream tasks including feedforward reconstruction, dense image matching and relative pose estimation, finding that it outperforms the state-of-the-art visual encoders DINOv3 and CroCo v2. 

---
# Large Language Models for Sentiment Analysis to Detect Social Challenges: A Use Case with South African Languages 

**Authors**: Koena Ronny Mabokela, Tim Schlippe, Matthias Wölfel  

**Link**: [PDF](https://arxiv.org/pdf/2511.17301)  

**Abstract**: Sentiment analysis can aid in understanding people's opinions and emotions on social issues. In multilingual communities sentiment analysis systems can be used to quickly identify social challenges in social media posts, enabling government departments to detect and address these issues more precisely and effectively. Recently, large-language models (LLMs) have become available to the wide public and initial analyses have shown that they exhibit magnificent zero-shot sentiment analysis abilities in English. However, there is no work that has investigated to leverage LLMs for sentiment analysis on social media posts in South African languages and detect social challenges. Consequently, in this work, we analyse the zero-shot performance of the state-of-the-art LLMs GPT-3.5, GPT-4, LlaMa 2, PaLM 2, and Dolly 2 to investigate the sentiment polarities of the 10 most emerging topics in English, Sepedi and Setswana social media posts that fall within the jurisdictional areas of 10 South African government departments. Our results demonstrate that there are big differences between the various LLMs, topics, and languages. In addition, we show that a fusion of the outcomes of different LLMs provides large gains in sentiment classification performance with sentiment classification errors below 1%. Consequently, it is now feasible to provide systems that generate reliable information about sentiment analysis to detect social challenges and draw conclusions about possible needs for actions on specific topics and within different language groups. 

---
# Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation 

**Authors**: Chuancheng Shi, Shangze Li, Shiming Guo, Simiao Xie, Wenhua Wu, Jingtong Dou, Chao Wu, Canran Xiao, Cong Wang, Zifeng Cheng, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2511.17282)  

**Abstract**: Multilingual text-to-image (T2I) models have advanced rapidly in terms of visual realism and semantic alignment, and are now widely utilized. Yet outputs vary across cultural contexts: because language carries cultural connotations, images synthesized from multilingual prompts should preserve cross-lingual cultural consistency. We conduct a comprehensive analysis showing that current T2I models often produce culturally neutral or English-biased results under multilingual prompts. Analyses of two representative models indicate that the issue stems not from missing cultural knowledge but from insufficient activation of culture-related representations. We propose a probing method that localizes culture-sensitive signals to a small set of neurons in a few fixed layers. Guided by this finding, we introduce two complementary alignment strategies: (1) inference-time cultural activation that amplifies the identified neurons without backbone fine-tuned; and (2) layer-targeted cultural enhancement that updates only culturally relevant layers. Experiments on our CultureBench demonstrate consistent improvements over strong baselines in cultural consistency while preserving fidelity and diversity. 

---
# Leveraging CVAE for Joint Configuration Estimation of Multifingered Grippers from Point Cloud Data 

**Authors**: Julien Merand, Boris Meden, Mathieu Grossard  

**Link**: [PDF](https://arxiv.org/pdf/2511.17276)  

**Abstract**: This paper presents an efficient approach for determining the joint configuration of a multifingered gripper solely from the point cloud data of its poly-articulated chain, as generated by visual sensors, simulations or even generative neural networks. Well-known inverse kinematics (IK) techniques can provide mathematically exact solutions (when they exist) for joint configuration determination based solely on the fingertip pose, but often require post-hoc decision-making by considering the positions of all intermediate phalanges in the gripper's fingers, or rely on algorithms to numerically approximate solutions for more complex kinematics. In contrast, our method leverages machine learning to implicitly overcome these challenges. This is achieved through a Conditional Variational Auto-Encoder (CVAE), which takes point cloud data of key structural elements as input and reconstructs the corresponding joint configurations. We validate our approach on the MultiDex grasping dataset using the Allegro Hand, operating within 0.05 milliseconds and achieving accuracy comparable to state-of-the-art methods. This highlights the effectiveness of our pipeline for joint configuration estimation within the broader context of AI-driven techniques for grasp planning. 

---
# Range-Edit: Semantic Mask Guided Outdoor LiDAR Scene Editing 

**Authors**: Suchetan G. Uppur, Hemant Kumar, Vaibhav Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.17269)  

**Abstract**: Training autonomous driving and navigation systems requires large and diverse point cloud datasets that capture complex edge case scenarios from various dynamic urban settings. Acquiring such diverse scenarios from real-world point cloud data, especially for critical edge cases, is challenging, which restricts system generalization and robustness. Current methods rely on simulating point cloud data within handcrafted 3D virtual environments, which is time-consuming, computationally expensive, and often fails to fully capture the complexity of real-world scenes. To address some of these issues, this research proposes a novel approach that addresses the problem discussed by editing real-world LiDAR scans using semantic mask-based guidance to generate novel synthetic LiDAR point clouds. We incorporate range image projection and semantic mask conditioning to achieve diffusion-based generation. Point clouds are transformed to 2D range view images, which are used as an intermediate representation to enable semantic editing using convex hull-based semantic masks. These masks guide the generation process by providing information on the dimensions, orientations, and locations of objects in the real environment, ensuring geometric consistency and realism. This approach demonstrates high-quality LiDAR point cloud generation, capable of producing complex edge cases and dynamic scenes, as validated on the KITTI-360 dataset. This offers a cost-effective and scalable solution for generating diverse LiDAR data, a step toward improving the robustness of autonomous driving systems. 

---
# DISCA: A Digital In-memory Stochastic Computing Architecture Using A Compressed Bent-Pyramid Format 

**Authors**: Shady Agwa, Yikang Shen, Shiwei Wang, Themis Prodromakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.17265)  

**Abstract**: Nowadays, we are witnessing an Artificial Intelligence revolution that dominates the technology landscape in various application domains, such as healthcare, robotics, automotive, security, and defense. Massive-scale AI models, which mimic the human brain's functionality, typically feature millions and even billions of parameters through data-intensive matrix multiplication tasks. While conventional Von-Neumann architectures struggle with the memory wall and the end of Moore's Law, these AI applications are migrating rapidly towards the edge, such as in robotics and unmanned aerial vehicles for surveillance, thereby adding more constraints to the hardware budget of AI architectures at the edge. Although in-memory computing has been proposed as a promising solution for the memory wall, both analog and digital in-memory computing architectures suffer from substantial degradation of the proposed benefits due to various design limitations. We propose a new digital in-memory stochastic computing architecture, DISCA, utilizing a compressed version of the quasi-stochastic Bent-Pyramid data format. DISCA inherits the same computational simplicity of analog computing, while preserving the same scalability, productivity, and reliability of digital systems. Post-layout modeling results of DISCA show an energy efficiency of 3.59 TOPS/W per bit at 500 MHz using a commercial 180nm CMOS technology. Therefore, DISCA significantly improves the energy efficiency for matrix multiplication workloads by orders of magnitude if scaled and compared to its counterpart architectures. 

---
# Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats 

**Authors**: Jiaye Qian, Ge Zheng, Yuchen Zhu, Sibei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17254)  

**Abstract**: Despite their impressive performance across a wide range of tasks, Large Vision-Language Models (LVLMs) remain prone to hallucination. In this study, we propose a comprehensive intervention framework aligned with the transformer's causal architecture in LVLMs, integrating the effects of different intervention paths on hallucination. We find that hallucinations in LVLMs do not arise from a single causal path, but rather from the interplay among image-to-input-text, image-to-output-text, and text-to-text pathways. For the first time, we also find that LVLMs rely on different pathways depending on the question-answer alignment format. Building on these insights, we propose simple yet effective methods to identify and intervene on critical hallucination heads within each pathway, tailored to discriminative and generative formats. Experiments across multiple benchmarks demonstrate that our approach consistently reduces hallucinations across diverse alignment types. 

---
# Lost in Translation and Noise: A Deep Dive into the Failure Modes of VLMs on Real-World Tables 

**Authors**: Anshul Singh, Rohan Chaudhary, Gagneet Singh, Abhay Kumary  

**Link**: [PDF](https://arxiv.org/pdf/2511.17238)  

**Abstract**: The impressive performance of VLMs is largely measured on benchmarks that fail to capture the complexities of real-world scenarios. Existing datasets for tabular QA, such as WikiTableQuestions and FinQA, are overwhelmingly monolingual (English) and present tables in a digitally perfect, clean format. This creates a significant gap between research and practice. To address this, we present \textbf{MirageTVQA}, a new benchmark designed to evaluate VLMs on these exact dimensions. Featuring nearly 60,000 QA pairs across 24 languages, MirageTVQA challenges models with tables that are not only multilingual but also visually imperfect, incorporating realistic noise to mimic scanned documents. Our evaluation of the leading VLMs reveals two primary failure points: a severe degradation in performance (over 35\% drop for the best models) when faced with visual noise and a consistent English-first bias where reasoning abilities fail to transfer to other languages. MirageTVQA provides a benchmark for measuring and driving progress towards more robust VLM models for table reasoning. The dataset and the code are available at: this https URL. 

---
# Algorithmic design and implementation considerations of deep MPC 

**Authors**: Prabhat K. Mishra, Mateus V. Gasparino, Girish Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.17233)  

**Abstract**: Deep Model Predictive Control (Deep MPC) is an evolving field that integrates model predictive control and deep learning. This manuscript is focused on a particular approach, which employs deep neural network in the loop with MPC. This class of approaches distributes control authority between a neural network and an MPC controller, in such a way that the neural network learns the model uncertainties while the MPC handles constraints. The approach is appealing because training data collected while the system is in operation can be used to fine-tune the neural network, and MPC prevents unsafe behavior during those learning transients. This manuscript explains implementation challenges of Deep MPC, algorithmic way to distribute control authority and argues that a poor choice in distributing control authority may lead to poor performance. A reason of poor performance is explained through a numerical experiment on a four-wheeled skid-steer dynamics. 

---
# TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making 

**Authors**: Shanshan Li, Da Huang, Yu He, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.17225)  

**Abstract**: In daily life, people often move through spaces to find objects that meet their needs, posing a key challenge in embodied AI. Traditional Demand-Driven Navigation (DDN) handles one need at a time but does not reflect the complexity of real-world tasks involving multiple needs and personal choices. To bridge this gap, we introduce Task-Preferenced Multi-Demand-Driven Navigation (TP-MDDN), a new benchmark for long-horizon navigation involving multiple sub-demands with explicit task preferences. To solve TP-MDDN, we propose AWMSystem, an autonomous decision-making system composed of three key modules: BreakLLM (instruction decomposition), LocateLLM (goal selection), and StatusMLLM (task monitoring). For spatial memory, we design MASMap, which combines 3D point cloud accumulation with 2D semantic mapping for accurate and efficient environmental understanding. Our Dual-Tempo action generation framework integrates zero-shot planning with policy-based fine control, and is further supported by an Adaptive Error Corrector that handles failure cases in real time. Experiments demonstrate that our approach outperforms state-of-the-art baselines in both perception accuracy and navigation robustness. 

---
# Parrot: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs 

**Authors**: Yusuf Çelebi, Mahmoud El Hussieni, Özay Ezerceli  

**Link**: [PDF](https://arxiv.org/pdf/2511.17220)  

**Abstract**: This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world. 

---
# Attention-Guided Feature Fusion (AGFF) Model for Integrating Statistical and Semantic Features in News Text Classification 

**Authors**: Mohammad Zare  

**Link**: [PDF](https://arxiv.org/pdf/2511.17184)  

**Abstract**: News text classification is a crucial task in natural language processing, essential for organizing and filtering the massive volume of digital content. Traditional methods typically rely on statistical features like term frequencies or TF-IDF values, which are effective at capturing word-level importance but often fail to reflect contextual meaning. In contrast, modern deep learning approaches utilize semantic features to understand word usage within context, yet they may overlook simple, high-impact statistical indicators. This paper introduces an Attention-Guided Feature Fusion (AGFF) model that combines statistical and semantic features in a unified framework. The model applies an attention-based mechanism to dynamically determine the relative importance of each feature type, enabling more informed classification decisions. Through evaluation on benchmark news datasets, the AGFF model demonstrates superior performance compared to both traditional statistical models and purely semantic deep learning models. The results confirm that strategic integration of diverse feature types can significantly enhance classification accuracy. Additionally, ablation studies validate the contribution of each component in the fusion process. The findings highlight the model's ability to balance and exploit the complementary strengths of statistical and semantic representations, making it a practical and effective solution for real-world news classification tasks. 

---
# Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models 

**Authors**: Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia, Xiuzhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17170)  

**Abstract**: Large Language Models (LLMs) often produce fluent but factually incorrect responses, a phenomenon known as hallucination. Abstention, where the model chooses not to answer and instead outputs phrases such as "I don't know", is a common safeguard. However, existing abstention methods typically rely on post-generation signals, such as generation variations or feedback, which limits their ability to prevent unreliable responses in advance. In this paper, we introduce Aspect-Based Causal Abstention (ABCA), a new framework that enables early abstention by analysing the internal diversity of LLM knowledge through causal inference. This diversity reflects the multifaceted nature of parametric knowledge acquired from various sources, representing diverse aspects such as disciplines, legal contexts, or temporal frames. ABCA estimates causal effects conditioned on these aspects to assess the reliability of knowledge relevant to a given query. Based on these estimates, we enable two types of abstention: Type-1, where aspect effects are inconsistent (knowledge conflict), and Type-2, where aspect effects consistently support abstention (knowledge insufficiency). Experiments on standard benchmarks demonstrate that ABCA improves abstention reliability, achieves state-of-the-art performance, and enhances the interpretability of abstention decisions. 

---
# The PLLuM Instruction Corpus 

**Authors**: Piotr Pęzik, Filip Żarnecki, Konrad Kaczyński, Anna Cichosz, Zuzanna Deckert, Monika Garnys, Izabela Grabarczyk, Wojciech Janowski, Sylwia Karasińska, Aleksandra Kujawiak, Piotr Misztela, Maria Szymańska, Karolina Walkusz, Igor Siek, Maciej Chrabąszcz, Anna Kołos, Agnieszka Karlińska, Karolina Seweryn, Aleksandra Krasnodębska, Paula Betscher, Zofia Cieślińska, Katarzyna Kowol, Artur Wilczek, Maciej Trzciński, Katarzyna Dziewulska, Roman Roszko, Tomasz Bernaś, Jurgita Vaičenonienė, Danuta Roszko, Paweł Levchuk, Paweł Kowalski, Irena Prawdzic-Jankowska, Marek Kozłowski, Sławomir Dadas, Rafał Poświata, Alina Wróblewska, Katarzyna Krasnowska-Kieraś, Maciej Ogrodniczuk, Michał Rudolf, Piotr Rybak, Karolina Saputa, Joanna Wołoszyn, Marcin Oleksy, Bartłomiej Koptyra, Teddy Ferdinan, Stanisław Woźniak, Maciej Piasecki, Paweł Walkowiak, Konrad Wojtasik, Arkadiusz Janz, Przemysław Kazienko, Julia Moska, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2511.17161)  

**Abstract**: This paper describes the instruction dataset used to fine-tune a set of transformer-based large language models (LLMs) developed in the PLLuM (Polish Large Language Model) project. We present a functional typology of the organic, converted, and synthetic instructions used in PLLuM and share some observations about the implications of using human-authored versus synthetic instruction datasets in the linguistic adaptation of base LLMs. Additionally, we release the first representative subset of the PLLuM instruction corpus (PLLuMIC), which we believe to be useful in guiding and planning the development of similar datasets for other LLMs. 

---
# A lightweight detector for real-time detection of remote sensing images 

**Authors**: Qianyi Wang, Guoqiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.17147)  

**Abstract**: Remote sensing imagery is widely used across various fields, yet real-time detection remains challenging due to the prevalence of small objects and the need to balance accuracy with efficiency. To address this, we propose DMG-YOLO, a lightweight real-time detector tailored for small object detection in remote sensing images. Specifically, we design a Dual-branch Feature Extraction (DFE) module in the backbone, which partitions feature maps into two parallel branches: one extracts local features via depthwise separable convolutions, and the other captures global context using a vision transformer with a gating mechanism. Additionally, a Multi-scale Feature Fusion (MFF) module with dilated convolutions enhances multi-scale integration while preserving fine details. In the neck, we introduce the Global and Local Aggregate Feature Pyramid Network (GLAFPN) to further boost small object detection through global-local feature fusion. Extensive experiments on the VisDrone2019 and NWPU VHR-10 datasets show that DMG-YOLO achieves competitive performance in terms of mAP, model size, and other key metrics. 

---
# Device-Guided Music Transfer 

**Authors**: Manh Pham Hung, Changshuo Hu, Ting Dang, Dong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.17136)  

**Abstract**: Device-guided music transfer adapts playback across unseen devices for users who lack them. Existing methods mainly focus on modifying the timbre, rhythm, harmony, or instrumentation to mimic genres or artists, overlooking the diverse hardware properties of the playback device (i.e., speaker). Therefore, we propose DeMT, which processes a speaker's frequency response curve as a line graph using a vision-language model to extract device embeddings. These embeddings then condition a hybrid transformer via feature-wise linear modulation. Fine-tuned on a self-collected dataset, DeMT enables effective speaker-style transfer and robust few-shot adaptation for unseen devices, supporting applications like device-style augmentation and quality enhancement. 

---
# UI-CUBE: Enterprise-Grade Computer Use Agent Benchmarking Beyond Task Accuracy to Operational Reliability 

**Authors**: Horia Cristescu, Charles Park, Trong Canh Nguyen, Sergiu Talmacel, Alexandru-Gabriel Ilie, Stefan Adam  

**Link**: [PDF](https://arxiv.org/pdf/2511.17131)  

**Abstract**: While current Computer Use Agent (CUA) benchmarks measure task completion effectively, they provide limited assessment of enterprise deployment readiness, emphasizing functional correctness over the operational reliability required for production systems. We present UI-CUBE (UiPath Computer Use BEnchmark), a systematic benchmark comprising 226 tasks across two difficulty tiers designed to expose fundamental architectural limitations in current CUAs. Our evaluation covers simple UI interactions (136 tasks) and complex workflows including copy-paste tasks (50 tasks) and enterprise application scenarios (40 tasks), with systematic interface variation coverage, multi-resolution testing and automated validation of task success through the application state. Evaluation of five state-of-the-art models reveals a sharp capability cliff rather than gradual performance degradation. Simple UI interactions achieve 67-85% success rates (compared to 97.9% human performance), but complex workflows drop precipitously to 9-19%. Human evaluators with no prior application experience achieve only 61.2% on complex tasks despite near-perfect performance on simple tasks, establishing realistic performance ceilings. This discontinuous performance pattern -- where agents achieve 68-87% of human performance on simple tasks but only 15-32% on complex workflows -- indicates fundamental architectural limitations in memory management, hierarchical planning, and state coordination rather than incremental capability gaps addressable through better training or prompting. UI-CUBE functions as an enterprise-readiness diagnostic, revealing that while current CUAs can manipulate individual interface elements, they cannot yet function as reliable workflow automation tools. These findings provide architectural insights essential for developing production-ready CUAs capable of managing complex, multi-step enterprise processes. 

---
# Learning to Compress: Unlocking the Potential of Large Language Models for Text Representation 

**Authors**: Yeqin Zhang, Yizheng Zhao, Chen Hu, Binxing Jiao, Daxin Jiang, Ruihang Miao, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.17129)  

**Abstract**: Text representation plays a critical role in tasks like clustering, retrieval, and other downstream applications. With the emergence of large language models (LLMs), there is increasing interest in harnessing their capabilities for this purpose. However, most of the LLMs are inherently causal and optimized for next-token prediction, making them suboptimal for producing holistic representations. To address this, recent studies introduced pretext tasks to adapt LLMs for text representation. Most of these tasks, however, rely on token-level prediction objectives, such as the masked next-token prediction (MNTP) used in LLM2Vec. In this work, we explore the untapped potential of context compression as a pretext task for unsupervised adaptation of LLMs. During compression pre-training, the model learns to generate compact memory tokens, which substitute the whole context for downstream sequence prediction. Experiments demonstrate that a well-designed compression objective can significantly enhance LLM-based text representations, outperforming models trained with token-level pretext tasks. Further improvements through contrastive learning produce a strong representation model (LLM2Comp) that outperforms contemporary LLM-based text encoders on a wide range of tasks while being more sample-efficient, requiring significantly less training data. 

---
# Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design 

**Authors**: Quentin Anthony, Yury Tokpanov, Skyler Szot, Srivatsan Rajagopal, Praneeth Medepalli, Rishi Iyer, Vasu Shyam, Anna Golubeva, Ansh Chaurasia, Xiao Yang, Tomas Figliolia, Robert Washbourne, Drew Thorstensen, Amartey Pearson, Zack Grossbart, Jason van Patten, Emad Barsoum, Zhenyu Gu, Yao Fu, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2511.17127)  

**Abstract**: We report on the first large-scale mixture-of-experts (MoE) pretraining study on pure AMD hardware, utilizing both MI300X GPUs with Pollara interconnect. We distill practical guidance for both systems and model design. On the systems side, we deliver a comprehensive cluster and networking characterization: microbenchmarks for all core collectives (all-reduce, reduce-scatter, all-gather, broadcast) across message sizes and GPU counts on Pollara. To our knowledge, this is the first at this scale. We further provide MI300X microbenchmarks on kernel sizing and memory bandwidth to inform model design. On the modeling side, we introduce and apply MI300X-aware transformer sizing rules for attention and MLP blocks and justify MoE widths that jointly optimize training throughput and inference latency. We describe our training stack in depth, including often-ignored utilities such as fault-tolerance and checkpoint-reshaping, as well as detailed information on our training recipe. We also provide a preview of our model architecture and base model - ZAYA1 (760M active, 8.3B total parameters MoE) - which will be further improved upon in forthcoming papers. ZAYA1-base achieves performance comparable to leading base models such as Qwen3-4B and Gemma3-12B at its scale and larger, and outperforms models including Llama-3-8B and OLMoE across reasoning, mathematics, and coding benchmarks. Together, these results demonstrate that the AMD hardware, network, and software stack are mature and optimized enough for competitive large-scale pretraining. 

---
# AutoGraphAD: A novel approach using Variational Graph Autoencoders for anomalous network flow detection 

**Authors**: Georgios Anyfantis, Pere Barlet-Ros  

**Link**: [PDF](https://arxiv.org/pdf/2511.17113)  

**Abstract**: Network Intrusion Detection Systems (NIDS) are essential tools for detecting network attacks and intrusions. While extensive research has explored the use of supervised Machine Learning for attack detection and characterisation, these methods require accurately labelled datasets, which are very costly to obtain. Moreover, existing public datasets have limited and/or outdated attacks, and many of them suffer from mislabelled data. To reduce the reliance on labelled data, we propose AutoGraphAD, a novel unsupervised anomaly detection approach based on a Heterogeneous Variational Graph Autoencoder. AutoGraphAD operates on heterogeneous graphs, made from connection and IP nodes that capture network activity within a time window. The model is trained using unsupervised and contrastive learning, without relying on any labelled data. The reconstruction, structural loss, and KL divergence are then weighted and combined in an anomaly score that is then used for anomaly detection. Overall, AutoGraphAD yields the same, and in some cases better, results than previous unsupervised approaches, such as Anomal-E, but without requiring costly downstream anomaly detectors. As a result, AutoGraphAD achieves around 1.18 orders of magnitude faster training and 1.03 orders of magnitude faster inference, which represents a significant advantage for operational deployment. 

---
# Geometric-Disentangelment Unlearning 

**Authors**: Duo Zhou, Yuji Zhang, Tianxin Wei, Ruizhong Qiu, Ke Yang, Xiao Lin, Cheng Qian, Jingrui He, Hanghang Tong, Heng Ji, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17100)  

**Abstract**: Machine unlearning, the removal of a training subset's influence from a deployed model, is critical for privacy preservation and model reliability, yet gradient ascent on forget samples often harms retained knowledge. Existing approaches face a persistent tradeoff between effective forgetting and preservation on the retain set. While previous methods provide useful heuristics, they often lack a formal analysis on how exactly forgetting updates harm retained knowledge, and whether the side effects can be removed with theoretical guarantees. To explore a theoretically sound and simple solution, we start from the first principle on how performance on the retain set is actually affected: a first-order analysis of the local change of the retain loss under small parameter updates during model training. We start from a crisp equivalence: the retain loss is unchanged to first order iff the update direction is orthogonal to the subspace spanned by retain gradients ("retain-invariant"). This identifies the entangled component as the tangential part of forget update within the retain-gradient subspace, and characterizes disentanglement as orthogonality. Guided by this, we propose the Geometric-disentanglement Unlearning (GU) that decomposes any candidate forget gradient update into tangential and normal components to retain space and executes only the normal component. Under a standard trust-region budget, the projected direction aligned with the raw forget gradient is optimal among all first-order retain-invariant moves, and we also derive the optimal projected direction for joint forget-retain updating objectives. Our method is plug-and-play and can be attached to existing gradient-based unlearning procedures to mitigate side effects. GU achieves consistent improvement on various methods across three benchmarks TOFU, MUSE, and WMDP. 

---
# Spanning Tree Autoregressive Visual Generation 

**Authors**: Sangkyu Lee, Changho Lee, Janghoon Han, Hosung Song, Tackgeun You, Hwasup Lim, Stanley Jungkyu Choi, Honglak Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17089)  

**Abstract**: We present Spanning Tree Autoregressive (STAR) modeling, which can incorporate prior knowledge of images, such as center bias and locality, to maintain sampling performance while also providing sufficiently flexible sequence orders to accommodate image editing at inference. Approaches that expose randomly permuted sequence orders to conventional autoregressive (AR) models in visual generation for bidirectional context either suffer from a decline in performance or compromise the flexibility in sequence order choice at inference. Instead, STAR utilizes traversal orders of uniform spanning trees sampled in a lattice defined by the positions of image patches. Traversal orders are obtained through breadth-first search, allowing us to efficiently construct a spanning tree whose traversal order ensures that the connected partial observation of the image appears as a prefix in the sequence through rejection sampling. Through the tailored yet structured randomized strategy compared to random permutation, STAR preserves the capability of postfix completion while maintaining sampling performance without any significant changes to the model architecture widely adopted in the language AR modeling. 

---
# Why Do Language Model Agents Whistleblow? 

**Authors**: Kushal Agrawal, Frank Xiao, Guido Bergman, Asa Cooper Stickland  

**Link**: [PDF](https://arxiv.org/pdf/2511.17085)  

**Abstract**: The deployment of Large Language Models (LLMs) as tool-using agents causes their alignment training to manifest in new ways. Recent work finds that language models can use tools in ways that contradict the interests or explicit instructions of the user. We study LLM whistleblowing: a subset of this behavior where models disclose suspected misconduct to parties beyond the dialog boundary (e.g., regulatory agencies) without user instruction or knowledge. We introduce an evaluation suite of diverse and realistic staged misconduct scenarios to assess agents for this behavior. Across models and settings, we find that: (1) the frequency of whistleblowing varies widely across model families, (2) increasing the complexity of the task the agent is instructed to complete lowers whistleblowing tendencies, (3) nudging the agent in the system prompt to act morally substantially raises whistleblowing rates, and (4) giving the model more obvious avenues for non-whistleblowing behavior, by providing more tools and a detailed workflow to follow, decreases whistleblowing rates. Additionally, we verify the robustness of our dataset by testing for model evaluation awareness, and find that both black-box methods and probes on model activations show lower evaluation awareness in our settings than in comparable previous work. 

---
# ReBrain: Brain MRI Reconstruction from Sparse CT Slice via Retrieval-Augmented Diffusion 

**Authors**: Junming Liu, Yifei Sun, Weihua Cheng, Yujin Kang, Yirong Chen, Ding Wang, Guosun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.17068)  

**Abstract**: Magnetic Resonance Imaging (MRI) plays a crucial role in brain disease diagnosis, but it is not always feasible for certain patients due to physical or clinical constraints. Recent studies attempt to synthesize MRI from Computed Tomography (CT) scans; however, low-dose protocols often result in highly sparse CT volumes with poor through-plane resolution, making accurate reconstruction of the full brain MRI volume particularly challenging. To address this, we propose ReBrain, a retrieval-augmented diffusion framework for brain MRI reconstruction. Given any 3D CT scan with limited slices, we first employ a Brownian Bridge Diffusion Model (BBDM) to synthesize MRI slices along the 2D dimension. Simultaneously, we retrieve structurally and pathologically similar CT slices from a comprehensive prior database via a fine-tuned retrieval model. These retrieved slices are used as references, incorporated through a ControlNet branch to guide the generation of intermediate MRI slices and ensure structural continuity. We further account for rare retrieval failures when the database lacks suitable references and apply spherical linear interpolation to provide supplementary guidance. Extensive experiments on SynthRAD2023 and BraTS demonstrate that ReBrain achieves state-of-the-art performance in cross-modal reconstruction under sparse conditions. 

---
# OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding 

**Authors**: Teng Fu, Mengyang Zhao, Ke Niu, Kaixin Peng, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.17053)  

**Abstract**: LVLMs have been shown to perform excellently in image-level tasks such as VQA and caption. However, in many instance-level tasks, such as visual grounding and object detection, LVLMs still show performance gaps compared to previous expert models. Meanwhile, although pedestrian tracking is a classical task, there have been a number of new topics in combining object tracking and natural language, such as Referring MOT, Cross-view Referring MOT, and Semantic MOT. These tasks emphasize that models should understand the tracked object at an advanced semantic level, which is exactly where LVLMs excel. In this paper, we propose a new unified Pedestrian Tracking framework, namely OmniPT, which can track, track based on reference and generate semantic understanding of tracked objects interactively. We address two issues: how to model the tracking task into a task that foundation models can perform, and how to make the model output formatted answers. To this end, we implement a training phase consisting of RL-Mid Training-SFT-RL. Based on the pre-trained weights of the LVLM, we first perform a simple RL phase to enable the model to output fixed and supervisable bounding box format. Subsequently, we conduct a mid-training phase using a large number of pedestrian-related datasets. Finally, we perform supervised fine-tuning on several pedestrian tracking datasets, and then carry out another RL phase to improve the model's tracking performance and enhance its ability to follow instructions. We conduct experiments on tracking benchmarks and the experimental results demonstrate that the proposed method can perform better than the previous methods. 

---
# RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis 

**Authors**: Linfeng Dong, Yuchen Yang, Hao Wu, Wei Wang, Yuenan HouZhihang Zhong, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.17045)  

**Abstract**: We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at this https URL 

---
# MedImageInsight for Thoracic Cavity Health Classification from Chest X-rays 

**Authors**: Rama Krishna Boya, Mohan Kireeti Magalanadu, Azaruddin Palavalli, Rupa Ganesh Tekuri, Amrit Pattanayak, Prasanthi Enuga, Vignesh Esakki Muthu, Vivek Aditya Boya  

**Link**: [PDF](https://arxiv.org/pdf/2511.17043)  

**Abstract**: Chest radiography remains one of the most widely used imaging modalities for thoracic diagnosis, yet increasing imaging volumes and radiologist workload continue to challenge timely interpretation. In this work, we investigate the use of MedImageInsight, a medical imaging foundational model, for automated binary classification of chest X-rays into Normal and Abnormal categories. Two approaches were evaluated: (1) fine-tuning MedImageInsight for end-to-end classification, and (2) employing the model as a feature extractor for a transfer learning pipeline using traditional machine learning classifiers. Experiments were conducted using a combination of the ChestX-ray14 dataset and real-world clinical data sourced from partner hospitals. The fine-tuned classifier achieved the highest performance, with an ROC-AUC of 0.888 and superior calibration compared to the transfer learning models, demonstrating performance comparable to established architectures such as CheXNet. These results highlight the effectiveness of foundational medical imaging models in reducing task-specific training requirements while maintaining diagnostic reliability. The system is designed for integration into web-based and hospital PACS workflows to support triage and reduce radiologist burden. Future work will extend the model to multi-label pathology classification to provide preliminary diagnostic interpretation in clinical environments. 

---
# CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation 

**Authors**: Xiangrui Xiong, Yichuan Lu, Zifei Pan, Chang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.17041)  

**Abstract**: The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors. 

---
# Parameter-Free Neural Lens Blur Rendering for High-Fidelity Composites 

**Authors**: Lingyan Ruan, Bin Chen, Taehyun Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2511.17014)  

**Abstract**: Consistent and natural camera lens blur is important for seamlessly blending 3D virtual objects into photographed real-scenes. Since lens blur typically varies with scene depth, the placement of virtual objects and their corresponding blur levels significantly affect the visual fidelity of mixed reality compositions. Existing pipelines often rely on camera parameters (e.g., focal length, focus distance, aperture size) and scene depth to compute the circle of confusion (CoC) for realistic lens blur rendering. However, such information is often unavailable to ordinary users, limiting the accessibility and generalizability of these methods. In this work, we propose a novel compositing approach that directly estimates the CoC map from RGB images, bypassing the need for scene depth or camera metadata. The CoC values for virtual objects are inferred through a linear relationship between its signed CoC map and depth, and realistic lens blur is rendered using a neural reblurring network. Our method provides flexible and practical solution for real-world applications. Experimental results demonstrate that our method achieves high-fidelity compositing with realistic defocus effects, outperforming state-of-the-art techniques in both qualitative and quantitative evaluations. 

---
# Supervised Fine Tuning of Large Language Models for Domain Specific Knowledge Graph Construction:A Case Study on Hunan's Historical Celebrities 

**Authors**: Junjie Hao, Chun Wang, Ying Qiao, Qiuyue Zuo, Qiya Song, Hua Ma, Xieping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17012)  

**Abstract**: Large language models and knowledge graphs offer strong potential for advancing research on historical culture by supporting the extraction, analysis, and interpretation of cultural heritage. Using Hunan's modern historical celebrities shaped by Huxiang culture as a case study, pre-trained large models can help researchers efficiently extract key information, including biographical attributes, life events, and social relationships, from textual sources and construct structured knowledge graphs. However, systematic data resources for Hunan's historical celebrities remain limited, and general-purpose models often underperform in domain knowledge extraction and structured output generation in such low-resource settings. To address these issues, this study proposes a supervised fine-tuning approach for enhancing domain-specific information extraction. First, we design a fine-grained, schema-guided instruction template tailored to the Hunan historical celebrities domain and build an instruction-tuning dataset to mitigate the lack of domain-specific training corpora. Second, we apply parameter-efficient instruction fine-tuning to four publicly available large language models - Qwen2.5-7B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B, and Llama-3.1-8B-Instruct - and develop evaluation criteria for assessing their extraction performance. Experimental results show that all models exhibit substantial performance gains after fine-tuning. Among them, Qwen3-8B achieves the strongest results, reaching a score of 89.3866 with 100 samples and 50 training iterations. This study provides new insights into fine-tuning vertical large language models for regional historical and cultural domains and highlights their potential for cost-effective applications in cultural heritage knowledge extraction and knowledge graph construction. 

---
# FLUID: Training-Free Face De-identification via Latent Identity Substitution 

**Authors**: Jinhyeong Park, Shaheryar Muhammad, Seangmin Lee, Jong Taek Lee, Soon Ki Jung  

**Link**: [PDF](https://arxiv.org/pdf/2511.17005)  

**Abstract**: We present FLUID (Face de-identification in the Latent space via Utility-preserving Identity Displacement), a training-free framework that directly substitutes identity in the latent space of pretrained diffusion models. Inspired by substitution mechanisms in chemistry, we reinterpret identity editing as semantic displacement in the latent h-space of a pretrained unconditional diffusion model. Our framework discovers identity-editing directions through optimization guided by novel reagent losses, which supervise for attribute preservation and identity suppression. We further propose both linear and geodesic (tangent-based) editing schemes to effectively navigate the latent manifold. Experimental results on CelebA-HQ and FFHQ demonstrate that FLUID achieves a superior trade-off between identity suppression and attribute preservation, outperforming state-of-the-art de-identification methods in both qualitative and quantitative metrics. 

---
# The Finer the Better: Towards Granular-aware Open-set Domain Generalization 

**Authors**: Yunyun Wang, Zheng Duan, Xinyue Liao, Ke-Jia Chen, Songcan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16979)  

**Abstract**: Open-Set Domain Generalization (OSDG) tackles the realistic scenario where deployed models encounter both domain shifts and novel object categories. Despite impressive progress with vision-language models like CLIP, existing methods still fall into the dilemma between structural risk of known-classes and open-space risk from unknown-classes, and easily suffers from over-confidence, especially when distinguishing ``hard unknowns" that share fine-grained visual similarities with known classes. To this end, we propose a Semantic-enhanced CLIP (SeeCLIP) framework that explicitly addresses this dilemma through fine-grained semantic enhancement. In SeeCLIP, we propose a semantic-aware prompt enhancement module to decompose images into discriminative semantic tokens, enabling nuanced vision-language alignment beyond coarse category labels. To position unknown prompts effectively, we introduce duplex contrastive learning with complementary objectives, that is, repulsion to maintain separability from known classes, and cohesion to preserve semantic proximity. Further, our semantic-guided diffusion module synthesizes pseudo-unknowns by perturbing extracted semantic tokens, generating challenging samples that are visually similar to known classes yet exhibit key local differences. These hard negatives force the model to learn finer decision boundaries. Extensive experiments across five benchmarks demonstrate consistent improvements of 3% accuracy and 5% H-score over state-of-the-art methods. 

---
# Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems 

**Authors**: Kirill Nagaitsev, Luka Grbcic, Samuel Williams, Costin Iancu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16964)  

**Abstract**: Maximizing performance on available GPU hardware is an ongoing challenge for modern AI inference systems. Traditional approaches include writing custom GPU kernels and using specialized model compilers to tune high-level code for specific GPU targets. Recent work shows that LLM-based multi-agent systems can effectively perform such tuning, often outperforming existing compilers and eliminating the need for manual kernel development. However, the dynamics of multi-agent systems for this task remain unexplored. In this work, we present a logical framework for comparing multi-agent PyTorch optimization systems. Our evaluation shows that exploit-heavy strategies perform best when paired with error-fixing agents, and that performance correlates with the granularity of optimization steps. The best implementation achieves an average 2.88x speedup on an H100 GPU across diverse tasks in KernelBench, a benchmark suite covering a range of machine learning architectures in PyTorch. 

---
# RASTP: Representation-Aware Semantic Token Pruning for Generative Recommendation with Semantic Identifiers 

**Authors**: Tianyu Zhan, Kairui Fu, Zheqi Lv, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16943)  

**Abstract**: Generative recommendation systems typically leverage Semantic Identifiers (SIDs), which represent each item as a sequence of tokens that encode semantic information. However, representing item ID with multiple SIDs significantly increases input sequence length, which is a major determinant of computational complexity and memory consumption. While existing efforts primarily focus on optimizing attention computation and KV cache, we propose RASTP (Representation-Aware Semantic Token Pruning), which directly prunes less informative tokens in the input sequence. Specifically, RASTP evaluates token importance by combining semantic saliency, measured via representation magnitude, and attention centrality, derived from cumulative attention weights. Since RASTP dynamically prunes low-information or irrelevant semantic tokens, experiments on three real-world Amazon datasets show that RASTP reduces training time by 26.7\%, while maintaining or slightly improving recommendation performance. The code has been open-sourced at this https URL. 

---
# OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios 

**Authors**: Hong Gao, Jingyu Wu, Xiangkai Xu, Kangni Xie, Yunchen Zhang, Bin Zhong, Xurui Gao, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16937)  

**Abstract**: Spatio-Temporal Video Grounding (STVG) aims to localize target objects in videos based on natural language descriptions. Despite recent advances in Multimodal Large Language Models, a significant gap remains between current models and real-world demands involving diverse objects and complex queries. We attribute this to limited benchmark scope, causing models to exhibit category bias, oversimplified reasoning, and poor linguistic robustness. To address these limitations, we introduce OmniGround, a comprehensive benchmark with 3,475 videos spanning 81 categories and complex real-world queries. We propose the Forward-Backward-Refinement annotation pipeline that combines multi-directional tracking with intelligent error correction for high-quality labels. We further introduce DeepSTG, a systematic evaluation framework quantifying dataset quality across four complementary dimensions beyond superficial statistics. Evaluations reveal performance average drop of 10.4% on complex real-world scenes, particularly with small/occluded objects and intricate spatial relations. Motivated by these, we propose PG-TAF, a training-free two-stage framework decomposing STVG into high-level temporal grounding and fine-grained spatio-temporal propagation. Experiments demonstrate PG-TAF achieves 25.6% and 35.6% improvements in m\_tIoU and m\_vIoU on OmniGround with consistent gains across four benchmarks. 

---
# PepEVOLVE: Position-Aware Dynamic Peptide Optimization via Group-Relative Advantage 

**Authors**: Trieu Nguyen, Hao-Wei Pang, Shasha Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.16912)  

**Abstract**: Macrocyclic peptides are an emerging modality that combines biologics-like affinity with small-molecule-like developability, but their vast combinatorial space and multi-parameter objectives make lead optimization slow and challenging. Prior generative approaches such as PepINVENT require chemists to pre-specify mutable positions for optimization, choices that are not always known a priori, and rely on static pretraining and optimization algorithms that limit the model's ability to generalize and effectively optimize peptide sequences. We introduce PepEVOLVE, a position-aware, dynamic framework that learns both where to edit and how to dynamically optimize peptides for multi-objective improvement. PepEVOLVE (i) augments pretraining with dynamic masking and CHUCKLES shifting to improve generalization, (ii) uses a context-free multi-armed bandit router that discovers high-reward residues, and (iii) couples a novel evolving optimization algorithm with group-relative advantage to stabilize reinforcement updates. During in silico evaluations, the router policy reliably learns and concentrates probability on chemically meaningful sites that influence the peptide's properties. On a therapeutically motivated Rev-binding macrocycle benchmark, PepEVOLVE outperformed PepINVENT by reaching higher mean scores (approximately 0.8 vs. 0.6), achieving best candidates with a score of 0.95 (vs. 0.87), and converging in fewer steps under the task of optimizing permeability and lipophilicity with structural constraints. Overall, PepEVOLVE offers a practical, reproducible path to peptide lead optimization when optimal edit sites are unknown, enabling more efficient exploration and improving design quality across multiple objectives. 

---
# Deep Improvement Supervision 

**Authors**: Arip Asadulaev, Rayan Banerjee, Fakhri Karray, Martin Takac  

**Link**: [PDF](https://arxiv.org/pdf/2511.16886)  

**Abstract**: Recently, it was shown that small, looped architectures, such as Tiny Recursive Models (TRMs), can outperform Large Language Models (LLMs) on complex reasoning tasks, including the Abstraction and Reasoning Corpus (ARC). In this work, we investigate a core question: how can we further improve the efficiency of these methods with minimal changes? To address this, we frame the latent reasoning of TRMs as a form of classifier-free guidance and implicit policy improvement algorithm. Building on these insights, we propose a novel training scheme that provides a target for each loop during training. We demonstrate that our approach significantly enhances training efficiency. Our method reduces the total number of forward passes by 18x and eliminates halting mechanisms, while maintaining quality comparable to standard TRMs. Notably, we achieve 24% accuracy on ARC-1 with only 0.8M parameters, outperforming most LLMs. 

---
# Generative AI in Sociological Research: State of the Discipline 

**Authors**: AJ Alvero, Dustin S. Stoltz, Oscar Stuhler, Marshall Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2511.16884)  

**Abstract**: Generative artificial intelligence (GenAI) has garnered considerable attention for its potential utility in research and scholarship, even among those who typically do not rely on computational tools. Early commentators, however, have also articulated concerns about how GenAI usage comes with enormous environmental costs, serious social risks, and a tendency to produce low-quality content. In the midst of both excitement and skepticism, it is crucial to take stock of how GenAI is actually being used. Our study focuses on sociological research as our site, and here we present findings from a survey of 433 authors of articles published in 50 sociology journals in the last five years. The survey provides an overview of the state of the discipline with regard to the use of GenAI by providing answers to fundamental questions: how (much) do scholars use the technology for their research; what are their reasons for using it; and how concerned, trustful, and optimistic are they about the technology? Of the approximately one third ofrespondents who self-report using GenAI at least weekly, the primary uses are for writing assistance and comparatively less so in planning, data collection, or data analysis. In both use and attitudes, there are surprisingly few differences between self-identified computational and non-computational researchers. Generally, respondents are very concerned about the social and environmental consequences of GenAI. Trust in GenAI outputs is low, regardless of expertise or frequency of use. While optimism that GenAI will improve is high, scholars are divided on whether GenAI will have a positive impact on the field. 

---
# The use of vocal biomarkers in the detection of Parkinson's disease: a robust statistical performance comparison of classic machine learning models 

**Authors**: Katia Pires Nascimento do Sacramento, Elliot Q. C. Garcia, Nicéias Silva Vilela, Vinicius P. Sacramento, Tiago A. E. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2511.16856)  

**Abstract**: Parkinson's disease (PD) is a progressive neurodegenerative disorder that, in addition to directly impairing functional mobility, is frequently associated with vocal impairments such as hypophonia and dysarthria, which typically manifest in the early stages. The use of vocal biomarkers to support the early diagnosis of PD presents a non-invasive, low-cost, and accessible alternative in clinical settings. Thus, the objective of this cross-sectional study was to consistently evaluate the effectiveness of a Deep Neural Network (DNN) in distinguishing individuals with Parkinson's disease from healthy controls, in comparison with traditional Machine Learning (ML) methods, using vocal biomarkers. Two publicly available voice datasets were used. Mel-frequency cepstral coefficients (MFCCs) were extracted from the samples, and model robustness was assessed using a validation strategy with 1000 independent random executions. Performance was evaluated using classification statistics. Since normality assumptions were not satisfied, non-parametric tests (Kruskal-Wallis and Bonferroni post-hoc tests) were applied to verify whether the tested classification models were similar or different in the classification of PD. With an average accuracy of $98.65\%$ and $92.11\%$ on the Italian Voice dataset and Parkinson's Telemonitoring dataset, respectively, the DNN demonstrated superior performance and efficiency compared to traditional ML models, while also achieving competitive results when benchmarked against relevant studies. Overall, this study confirms the efficiency of DNNs and emphasizes their potential to provide greater accuracy and reliability for the early detection of neurodegenerative diseases using voice-based biomarkers. 

---
# MRI Super-Resolution with Deep Learning: A Comprehensive Survey 

**Authors**: Mohammad Khateri, Serge Vasylechko, Morteza Ghahremani, Liam Timms, Deniz Kocanaogullari, Simon K. Warfield, Camilo Jaimes, Davood Karimi, Alejandra Sierra, Jussi Tohka, Sila Kurugol, Onur Afacan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16854)  

**Abstract**: High-resolution (HR) magnetic resonance imaging (MRI) is crucial for many clinical and research applications. However, achieving it remains costly and constrained by technical trade-offs and experimental limitations. Super-resolution (SR) presents a promising computational approach to overcome these challenges by generating HR images from more affordable low-resolution (LR) scans, potentially improving diagnostic accuracy and efficiency without requiring additional hardware. This survey reviews recent advances in MRI SR techniques, with a focus on deep learning (DL) approaches. It examines DL-based MRI SR methods from the perspectives of computer vision, computational imaging, inverse problems, and MR physics, covering theoretical foundations, architectural designs, learning strategies, benchmark datasets, and performance metrics. We propose a systematic taxonomy to categorize these methods and present an in-depth study of both established and emerging SR techniques applicable to MRI, considering unique challenges in clinical and research contexts. We also highlight open challenges and directions that the community needs to address. Additionally, we provide a collection of essential open-access resources, tools, and tutorials, available on our GitHub: this https URL.
IEEE keywords: MRI, Super-Resolution, Deep Learning, Computational Imaging, Inverse Problem, Survey. 

---
# Sex and age determination in European lobsters using AI-Enhanced bioacoustics 

**Authors**: Feliciano Pedro Francisco Domingos, Isibor Kennedy Ihianle, Omprakash Kaiwartya, Ahmad Lotfi, Nicola Khan, Nicholas Beaudreau, Amaya Albalat, Pedro Machado  

**Link**: [PDF](https://arxiv.org/pdf/2511.16848)  

**Abstract**: Monitoring aquatic species, especially elusive ones like lobsters, presents challenges. This study focuses on Homarus gammarus (European lobster), a key species for fisheries and aquaculture, and leverages non-invasive Passive Acoustic Monitoring (PAM). Understanding lobster habitats, welfare, reproduction, sex, and age is crucial for management and conservation. While bioacoustic emissions have classified various aquatic species using Artificial Intelligence (AI) models, this research specifically uses H. gammarus bioacoustics (buzzing/carapace vibrations) to classify lobsters by age (juvenile/adult) and sex (male/female).
The dataset was collected at Johnshaven, Scotland, using hydrophones in concrete tanks. We explored the efficacy of Deep Learning (DL) models (1D-CNN, 1D-DCNN) and six Machine Learning (ML) models (SVM, k-NN, Naive Bayes, Random Forest, XGBoost, MLP). Mel-frequency cepstral coefficients (MFCCs) were used as features.
For age classification (adult vs. juvenile), most models achieved over 97% accuracy (Naive Bayes: 91.31%). For sex classification, all models except Naive Bayes surpassed 93.23%. These strong results demonstrate the potential of supervised ML and DL to extract age- and sex-related features from lobster sounds. This research offers a promising non-invasive PAM approach for lobster conservation, detection, and management in aquaculture and fisheries, enabling real-world edge computing applications for underwater species. 

---
# ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers 

**Authors**: Seyed Mohssen Ghafari, Ronny Kol, Juan C. Quiroz, Nella Luan, Monika Patial, Chanaka Rupasinghe, Herman Wandabwa, Luiz Pizzato  

**Link**: [PDF](https://arxiv.org/pdf/2511.16846)  

**Abstract**: Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations. 

---
# Analysis of heart failure patient trajectories using sequence modeling 

**Authors**: Falk Dippela, Yinan Yu, Annika Rosengren, Martin Lindgren, Christina E. Lundberg, Erik Aerts, Martin Adiels, Helen Sjöland  

**Link**: [PDF](https://arxiv.org/pdf/2511.16839)  

**Abstract**: Transformers have defined the state-of-the-art for clinical prediction tasks involving electronic health records (EHRs). The recently introduced Mamba architecture outperformed an advanced Transformer (Transformer++) based on Llama in handling long context lengths, while using fewer model parameters. Despite the impressive performance of these architectures, a systematic approach to empirically analyze model performance and efficiency under various settings is not well established in the medical domain. The performances of six sequence models were investigated across three architecture classes (Transformers, Transformers++, Mambas) in a large Swedish heart failure (HF) cohort (N = 42820), providing a clinically relevant case study. Patient data included diagnoses, vital signs, laboratories, medications and procedures extracted from in-hospital EHRs. The models were evaluated on three one-year prediction tasks: clinical instability (a readmission phenotype) after initial HF hospitalization, mortality after initial HF hospitalization and mortality after latest hospitalization. Ablations account for modifications of the EHR-based input patient sequence, architectural model configurations, and temporal preprocessing techniques for data collection. Llama achieves the highest predictive discrimination, best calibration, and showed robustness across all tasks, followed by Mambas. Both architectures demonstrate efficient representation learning, with tiny configurations surpassing other large-scaled Transformers. At equal model size, Llama and Mambas achieve superior performance using 25% less training data. This paper presents a first ablation study with systematic design choices for input tokenization, model configuration and temporal data preprocessing. Future model development in clinical prediction tasks using EHRs could build upon this study's recommendation as a starting point. 

---
# ManifoldFormer: Geometric Deep Learning for Neural Dynamics on Riemannian Manifolds 

**Authors**: Yihang Fu, Lifang He, Qingyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16828)  

**Abstract**: Existing EEG foundation models mainly treat neural signals as generic time series in Euclidean space, ignoring the intrinsic geometric structure of neural dynamics that constrains brain activity to low-dimensional manifolds. This fundamental mismatch between model assumptions and neural geometry limits representation quality and cross-subject generalization. ManifoldFormer addresses this limitation through a novel geometric deep learning framework that explicitly learns neural manifold representations. The architecture integrates three key innovations: a Riemannian VAE for manifold embedding that preserves geometric structure, a geometric Transformer with geodesic-aware attention mechanisms operating directly on neural manifolds, and a dynamics predictor leveraging neural ODEs for manifold-constrained temporal evolution. Extensive evaluation across four public datasets demonstrates substantial improvements over state-of-the-art methods, with 4.6-4.8% higher accuracy and 6.2-10.2% higher Cohen's Kappa, while maintaining robust cross-subject generalization. The geometric approach reveals meaningful neural patterns consistent with neurophysiological principles, establishing geometric constraints as essential for effective EEG foundation models. 

---
# WorldGen: From Text to Traversable and Interactive 3D Worlds 

**Authors**: Dilin Wang, Hyunyoung Jung, Tom Monnier, Kihyuk Sohn, Chuhang Zou, Xiaoyu Xiang, Yu-Ying Yeh, Di Liu, Zixuan Huang, Thu Nguyen-Phuoc, Yuchen Fan, Sergiu Oprea, Ziyan Wang, Roman Shapovalov, Nikolaos Sarafianos, Thibault Groueix, Antoine Toisoul, Prithviraj Dhar, Xiao Chu, Minghao Chen, Geon Yeong Park, Mahima Gupta, Yassir Azziz, Rakesh Ranjan, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16825)  

**Abstract**: We introduce WorldGen, a system that enables the automatic creation of large-scale, interactive 3D worlds directly from text prompts. Our approach transforms natural language descriptions into traversable, fully textured environments that can be immediately explored or edited within standard game engines. By combining LLM-driven scene layout reasoning, procedural generation, diffusion-based 3D generation, and object-aware scene decomposition, WorldGen bridges the gap between creative intent and functional virtual spaces, allowing creators to design coherent, navigable worlds without manual modeling or specialized 3D expertise. The system is fully modular and supports fine-grained control over layout, scale, and style, producing worlds that are geometrically consistent, visually rich, and efficient to render in real time. This work represents a step towards accessible, generative world-building at scale, advancing the frontier of 3D generative AI for applications in gaming, simulation, and immersive social environments. 

---
# Monte Carlo Expected Threat (MOCET) Scoring 

**Authors**: Joseph Kim, Saahith Potluri  

**Link**: [PDF](https://arxiv.org/pdf/2511.16823)  

**Abstract**: Evaluating and measuring AI Safety Level (ASL) threats are crucial for guiding stakeholders to implement safeguards that keep risks within acceptable limits. ASL-3+ models present a unique risk in their ability to uplift novice non-state actors, especially in the realm of biosecurity. Existing evaluation metrics, such as LAB-Bench, BioLP-bench, and WMDP, can reliably assess model uplift and domain knowledge. However, metrics that better contextualize "real-world risks" are needed to inform the safety case for LLMs, along with scalable, open-ended metrics to keep pace with their rapid advancements. To address both gaps, we introduce MOCET, an interpretable and doubly-scalable metric (automatable and open-ended) that can quantify real-world risks. 

---
# A Robust Federated Learning Approach for Combating Attacks Against IoT Systems Under non-IID Challenges 

**Authors**: Eyad Gad, Zubair Md Fadlullah, Mostafa M. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2511.16822)  

**Abstract**: In the context of the growing proliferation of user devices and the concurrent surge in data volumes, the complexities arising from the substantial increase in data have posed formidable challenges to conventional machine learning model training. Particularly, this is evident within resource-constrained and security-sensitive environments such as those encountered in networks associated with the Internet of Things (IoT). Federated Learning has emerged as a promising remedy to these challenges by decentralizing model training to edge devices or parties, effectively addressing privacy concerns and resource limitations. Nevertheless, the presence of statistical heterogeneity in non-Independently and Identically Distributed (non-IID) data across different parties poses a significant hurdle to the effectiveness of FL. Many FL approaches have been proposed to enhance learning effectiveness under statistical heterogeneity. However, prior studies have uncovered a gap in the existing research landscape, particularly in the absence of a comprehensive comparison between federated methods addressing statistical heterogeneity in detecting IoT attacks. In this research endeavor, we delve into the exploration of FL algorithms, specifically FedAvg, FedProx, and Scaffold, under different data distributions. Our focus is on achieving a comprehensive understanding of and addressing the challenges posed by statistical heterogeneity. In this study, We classify large-scale IoT attacks by utilizing the CICIoT2023 dataset. Through meticulous analysis and experimentation, our objective is to illuminate the performance nuances of these FL methods, providing valuable insights for researchers and practitioners in the domain. 

---
# Mesh RAG: Retrieval Augmentation for Autoregressive Mesh Generation 

**Authors**: Xiatao Sun, Chen Liang, Qian Wang, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2511.16807)  

**Abstract**: 3D meshes are a critical building block for applications ranging from industrial design and gaming to simulation and robotics. Traditionally, meshes are crafted manually by artists, a process that is time-intensive and difficult to scale. To automate and accelerate this asset creation, autoregressive models have emerged as a powerful paradigm for artistic mesh generation. However, current methods to enhance quality typically rely on larger models or longer sequences that result in longer generation time, and their inherent sequential nature imposes a severe quality-speed trade-off. This sequential dependency also significantly complicates incremental editing. To overcome these limitations, we propose Mesh RAG, a novel, training-free, plug-and-play framework for autoregressive mesh generation models. Inspired by RAG for language models, our approach augments the generation process by leveraging point cloud segmentation, spatial transformation, and point cloud registration to retrieve, generate, and integrate mesh components. This retrieval-based approach decouples generation from its strict sequential dependency, facilitating efficient and parallelizable inference. We demonstrate the wide applicability of Mesh RAG across various foundational autoregressive mesh generation models, showing it significantly enhances mesh quality, accelerates generation speed compared to sequential part prediction, and enables incremental editing, all without model retraining. 

---
# Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach 

**Authors**: Yaoxin Yang, Peng Ye, Xudong Tan, Chongjun Tu, Maosen Zhao, Jia Hao, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16786)  

**Abstract**: Multimodal large language models suffer from substantial inference overhead since multimodal KV Cache grows proportionally with the visual input length. Existing multimodal KV Cache compression methods mostly rely on attention score to reduce cache size, which makes them are incompatible with established efficient attention kernels (e.g., FlashAttention) and ignores the contribution of value vectors to the attention output. In this work, we revisit multimodal KV Cache compression from the perspective of the KV matrices' distribution. First, we observe that frequency-domain energy of multimodal KV matrices is predominantly concentrated in low-frequency and extract this principal energy via a low-pass filter. Further, we find that removing KV pairs that deviate substantially from this principal energy leads to a pronounced performance drop, which we define as Outlier KVs. Considering Outlier KVs are more likely to encode features critical for inference, we propose FlashCache, a frequency-domain-guided, Outlier-KV-aware KV Cache compression framework. First, we introduce an Outlier KV Recognition Module that models the principal component of multimodal KV matrices in the frequency domain and preferentially retains KV pairs that significantly deviate from it. Furthermore, Dynamic Budget Allocation Module is designed to adaptively determine the per-layer KV Cache size to retain more Outlier KVs. Experiments on multiple MLLMs and benchmarks demonstrate that FlashCache outperforms state-of-the-art multimoal KV compression methods, achieving up to 1.69 times faster decoding with 80% lower KV memory usage while maintaining task performance. 

---
# Generative Augmented Reality: Paradigms, Technologies, and Future Applications 

**Authors**: Chen Liang, Jiawen Zheng, Yufeng Zeng, Yi Tan, Hengye Lyu, Yuhui Zheng, Zisu Li, Yueting Weng, Jiaxin Shi, Hanwang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16783)  

**Abstract**: This paper introduces Generative Augmented Reality (GAR) as a next-generation paradigm that reframes augmentation as a process of world re-synthesis rather than world composition by a conventional AR engine. GAR replaces the conventional AR engine's multi-stage modules with a unified generative backbone, where environmental sensing, virtual content, and interaction signals are jointly encoded as conditioning inputs for continuous video generation. We formalize the computational correspondence between AR and GAR, survey the technical foundations that make real-time generative augmentation feasible, and outline prospective applications that leverage its unified inference model. We envision GAR as a future AR paradigm that delivers high-fidelity experiences in terms of realism, interactivity, and immersion, while eliciting new research challenges on technologies, content ecosystems, and the ethical and societal implications. 

---
# Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation 

**Authors**: Wei-Cheng Tseng, Xuanru Zhou, Mingyue Huo, Yiwen Shao, Hao Zhang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16757)  

**Abstract**: Audio-language pretraining holds promise for general-purpose audio understanding, yet remains underexplored compared to its vision counterpart. While vision-language models like CLIP serve as widely adopted foundations, existing audio-language models primarily excel at retrieval tasks with limited adoption as general-purpose encoders. We identify three key barriers: limited large-scale audio-text corpora, insufficient caption diversity, and lack of systematic exploration and evaluation. To this end, we introduce CaptionStew, a 10.7M caption dataset aggregating diverse open-source audio-text corpora across multiple domains and captioning styles. Using this resource, we conduct the first comprehensive evaluation comparing contrastive and captioning objectives for audio representation learning across speech, music, and environmental sound tasks. Our results demonstrate that audio-language pretraining yields competitive, transferable representations. Through systematic data-scaling experiments, we reveal complementary objective strengths: contrastive learning achieves superior data efficiency at smaller scales, while captioning demonstrates better scalability on language-involved audio understanding tasks. We also find that common supervised initialization practices provide diminishing returns at scale, challenging current approaches. These findings establish audio-language pretraining as a viable pathway toward general-purpose audio representations, guiding future research. To accelerate progress, we release data preparation recipes, training protocols, and pretrained models, paving the way toward universal audio understanding. 

---
# SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge 

**Authors**: Adeel Yousaf, Joseph Fioresi, James Beetham, Amrit Singh Bedi, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2511.16743)  

**Abstract**: Improving the safety of vision-language models like CLIP via fine-tuning often comes at a steep price, causing significant drops in their generalization performance. We find this trade-off stems from rigid alignment strategies that force unsafe concepts toward single, predefined safe targets, disrupting the model's learned semantic structure. To address this, we propose a proximity-aware approach: redirecting unsafe concepts to their semantically closest safe alternatives to minimize representational change. We introduce SaFeR-CLIP, a fine-tuning framework that applies this principle of minimal intervention. SaFeR-CLIP successfully reconciles safety and performance, recovering up to 8.0% in zero-shot accuracy over prior methods while maintaining robust safety. To support more rigorous evaluation, we also contribute NSFW-Caps, a new benchmark of 1,000 highly-aligned pairs for testing safety under distributional shift. Our work shows that respecting the geometry of pretrained representations is key to achieving safety without sacrificing performance. 

---
# SAM 3: Segment Anything with Concepts 

**Authors**: Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, Christoph Feichtenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2511.16719)  

**Abstract**: We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts, which we define as either short noun phrases (e.g., "yellow school bus"), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances. To advance PCS, we build a scalable data engine that produces a high-quality dataset with 4M unique concept labels, including hard negatives, across images and videos. Our model consists of an image-level detector and a memory-based video tracker that share a single backbone. Recognition and localization are decoupled with a presence head, which boosts detection accuracy. SAM 3 doubles the accuracy of existing systems in both image and video PCS, and improves previous SAM capabilities on visual segmentation tasks. We open source SAM 3 along with our new Segment Anything with Concepts (SA-Co) benchmark for promptable concept segmentation. 

---
# A Machine Learning-Driven Solution for Denoising Inertial Confinement Fusion Images 

**Authors**: Asya Y. Akkus, Bradley T. Wolfe, Pinghan Chu, Chengkun Huang, Chris S. Campbell, Mariana Alvarado Alvarez, Petr Volegov, David Fittinghoff, Robert Reinovsky, Zhehui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16717)  

**Abstract**: Neutron imaging is important in optimizing analysis of inertial confinement fusion (ICF) events such as those at the National Ignition Facility (NIF) and improving current and future ICF platforms. However, images of neutron sources are often degraded by various types of noise. Most commonly, Gaussian and Poisson noise often coexist within one image, obscuring fine details and blurring edges. These noise types often overlap, making them difficult to distinguish and remove using conventional filtering and thresholding methods. As a result, noise removal techniques that preserve image fidelity are important for analyzing and interpreting images of a neutron source. Current solutions include a combination of filtering and thresholding methodologies. In the past, machine learning approaches were rarely implemented due to a lack of ground truth neutron imaging data for ICF processes. However, recent advances in synthetic data production, particularly in the fusion imaging field, have opened opportunities to investigate new denoising procedures using both supervised and unsupervised machine learning methods. In this study, we implement an unsupervised autoencoder with a Cohen-Daubechies- Feauveau (CDF 97) wavelet transform in the latent space for mixed Gaussian-Poisson denoising. The network successfully denoises neutron imaging data. Additionally, it demonstrates lower reconstruction error and superior edge preservation metrics when benchmarked with data generated by a forward model and compared to non-ML-based filtering mechanisms such as Block-matching and 3D filtering (BM3D). This approach presents a promising advancement in neutron image noise reduction and three-dimensional reconstruction analysis of ICF experiments. 

---
# Password Strength Analysis Through Social Network Data Exposure: A Combined Approach Relying on Data Reconstruction and Generative Models 

**Authors**: Maurizio Atzori, Eleonora Calò, Loredana Caruccio, Stefano Cirillo, Giuseppe Polese, Giandomenico Solimando  

**Link**: [PDF](https://arxiv.org/pdf/2511.16716)  

**Abstract**: Although passwords remain the primary defense against unauthorized access, users often tend to use passwords that are easy to remember. This behavior significantly increases security risks, also due to the fact that traditional password strength evaluation methods are often inadequate. In this discussion paper, we present SODA ADVANCE, a data reconstruction tool also designed to enhance evaluation processes related to the password strength. In particular, SODA ADVANCE integrates a specialized module aimed at evaluating password strength by leveraging publicly available data from multiple sources, including social media platforms. Moreover, we investigate the capabilities and risks associated with emerging Large Language Models (LLMs) in evaluating and generating passwords, respectively. Experimental assessments conducted with 100 real users demonstrate that LLMs can generate strong and personalized passwords possibly defined according to user profiles. Additionally, LLMs were shown to be effective in evaluating passwords, especially when they can take into account user profile data. 

---
# DDTime: Dataset Distillation with Spectral Alignment and Information Bottleneck for Time-Series Forecasting 

**Authors**: Yuqi Li, Kuiye Ding, Chuanguang Yang, Hao Wang, Haoxuan Wang, Huiran Duan, Junming Liu, Yingli Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.16715)  

**Abstract**: Time-series forecasting is fundamental across many domains, yet training accurate models often requires large-scale datasets and substantial computational resources. Dataset distillation offers a promising alternative by synthesizing compact datasets that preserve the learning behavior of full data. However, extending dataset distillation to time-series forecasting is non-trivial due to two fundamental challenges: this http URL bias from strong autocorrelation, which leads to distorted value-term alignment between teacher and student models; and this http URL diversity among synthetic samples, arising from the absence of explicit categorical priors to regularize trajectory variety.
In this work, we propose DDTime, a lightweight and plug-in distillation framework built upon first-order condensation decomposition. To tackle Challenge 1, it revisits value-term alignment through temporal statistics and introduces a frequency-domain alignment mechanism to mitigate autocorrelation-induced bias, ensuring spectral consistency and temporal fidelity. To address Challenge 2, we further design an inter-sample regularization inspired by the information bottleneck principle, which enhances diversity and maximizes information density across synthetic trajectories. The combined objective is theoretically compatible with a wide range of condensation paradigms and supports stable first-order optimization. Extensive experiments on 20 benchmark datasets and diverse forecasting architectures demonstrate that DDTime consistently outperforms existing distillation methods, achieving about 30% relative accuracy gains while introducing about 2.49% computational overhead. All code and distilled datasets will be released. 

---
# PairHuman: A High-Fidelity Photographic Dataset for Customized Dual-Person Generation 

**Authors**: Ting Pan, Ye Wang, Peiguang Jing, Rui Ma, Zili Yi, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16712)  

**Abstract**: Personalized dual-person portrait customization has considerable potential applications, such as preserving emotional memories and facilitating wedding photography planning. However, the absence of a benchmark dataset hinders the pursuit of high-quality customization in dual-person portrait generation. In this paper, we propose the PairHuman dataset, which is the first large-scale benchmark dataset specifically designed for generating dual-person portraits that meet high photographic standards. The PairHuman dataset contains more than 100K images that capture a variety of scenes, attire, and dual-person interactions, along with rich metadata, including detailed image descriptions, person localization, human keypoints, and attribute tags. We also introduce DHumanDiff, which is a baseline specifically crafted for dual-person portrait generation that features enhanced facial consistency and simultaneously balances in personalized person generation and semantic-driven scene creation. Finally, the experimental results demonstrate that our dataset and method produce highly customized portraits with superior visual quality that are tailored to human preferences. Our dataset is publicly available at this https URL. 

---
# AutoBackdoor: Automating Backdoor Attacks via LLM Agents 

**Authors**: Yige Li, Zhe Li, Wei Zhao, Nay Myat Min, Hanxun Huang, Xingjun Ma, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.16709)  

**Abstract**: Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at this https URL. 

---
# Multi-Agent Code Verification with Compound Vulnerability Detection 

**Authors**: Shreshth Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16708)  

**Abstract**: LLMs generate buggy code: 29.6% of SWE-bench "solved" patches fail, 62% of BaxBench solutions have vulnerabilities, and existing tools only catch 65% of bugs with 35% false positives. We built CodeX-Verify, a multi-agent system that uses four specialized agents to detect different types of bugs. We prove mathematically that combining agents with different detection patterns finds more bugs than any single agent when the agents look for different problems, confirmed by measuring agent correlation of p = 0.05--0.25. We also show that multiple vulnerabilities in the same code create exponentially more risk than previously thought--SQL injection plus exposed credentials creates 15x more danger (risk 300 vs. 20) than traditional models predict. Testing on 99 code samples with verified labels shows our system catches 76.1% of bugs, matching the best existing method while running faster and without test execution. We tested 15 different agent combinations and found that using multiple agents improves accuracy by 39.7 percentage points (from 32.8% to 72.4%) compared to single agents, with gains of +14.9pp, +13.5pp, and +11.2pp for agents 2, 3, and 4. The best two-agent combination reaches 79.3% accuracy. Testing on 300 real patches from Claude Sonnet 4.5 runs in under 200ms per sample, making this practical for production use. 

---
# Large language models for automated PRISMA 2020 adherence checking 

**Authors**: Yuki Kataoka, Ryuhei So, Masahiro Banno, Yasushi Tsujimoto, Tomohiro Takayama, Yosuke Yamagishi, Takahiro Tsuge, Norio Yamamoto, Chiaki Suda, Toshi A. Furukawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.16707)  

**Abstract**: Evaluating adherence to PRISMA 2020 guideline remains a burden in the peer review process. To address the lack of shareable benchmarks, we constructed a copyright-aware benchmark of 108 Creative Commons-licensed systematic reviews and evaluated ten large language models (LLMs) across five input formats. In a development cohort, supplying structured PRISMA 2020 checklists (Markdown, JSON, XML, or plain text) yielded 78.7-79.7% accuracy versus 45.21% for manuscript-only input (p less than 0.0001), with no differences between structured formats (p>0.9). Across models, accuracy ranged from 70.6-82.8% with distinct sensitivity-specificity trade-offs, replicated in an independent validation cohort. We then selected Qwen3-Max (a high-sensitivity open-weight model) and extended evaluation to the full dataset (n=120), achieving 95.1% sensitivity and 49.3% specificity. Structured checklist provision substantially improves LLM-based PRISMA assessment, though human expert verification remains essential before editorial decisions. 

---
# RAG-Driven Data Quality Governance for Enterprise ERP Systems 

**Authors**: Sedat Bin Vedat, Enes Kutay Yarkan, Meftun Akarsu, Recep Kaan Karaman, Arda Sar, Çağrı Çelikbilek, Savaş Saygılı  

**Link**: [PDF](https://arxiv.org/pdf/2511.16700)  

**Abstract**: Enterprise ERP systems managing hundreds of thousands of employee records face critical data quality challenges when human resources departments perform decentralized manual entry across multiple languages. We present an end-to-end pipeline combining automated data cleaning with LLM-driven SQL query generation, deployed on a production system managing 240,000 employee records over six months.
The system operates in two integrated stages: a multi-stage cleaning pipeline that performs translation normalization, spelling correction, and entity deduplication during periodic synchronization from Microsoft SQL Server to PostgreSQL; and a retrieval-augmented generation framework powered by GPT-4o that translates natural-language questions in Turkish, Russian, and English into validated SQL queries. The query engine employs LangChain orchestration, FAISS vector similarity search, and few-shot learning with 500+ validated examples.
Our evaluation demonstrates 92.5% query validity, 95.1% schema compliance, and 90.7\% semantic accuracy on 2,847 production queries. The system reduces query turnaround time from 2.3 days to under 5 seconds while maintaining 99.2% uptime, with GPT-4o achieving 46% lower latency and 68% cost reduction versus GPT-3.5. This modular architecture provides a reproducible framework for AI-native enterprise data governance, demonstrating real-world viability at enterprise scale with 4.3/5.0 user satisfaction. 

---
# Detecting and Steering LLMs' Empathy in Action 

**Authors**: Juan P. Cadile  

**Link**: [PDF](https://arxiv.org/pdf/2511.16699)  

**Abstract**: We investigate empathy-in-action -- the willingness to sacrifice task efficiency to address human needs -- as a linear direction in LLM activation space. Using contrastive prompts grounded in the Empathy-in-Action (EIA) benchmark, we test detection and steering across Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored).
Detection: All models show AUROC 0.996-1.00 at optimal layers. Uncensored Dolphin matches safety-trained models, demonstrating empathy encoding emerges independent of safety training. Phi-3 probes correlate strongly with EIA behavioral scores (r=0.71, p<0.01). Cross-model probe agreement is limited (Qwen: r=-0.06, Dolphin: r=0.18), revealing architecture-specific implementations despite convergent detection.
Steering: Qwen achieves 65.3% success with bidirectional control and coherence at extreme interventions. Phi-3 shows 61.7% success with similar coherence. Dolphin exhibits asymmetric steerability: 94.4% success for pro-empathy steering but catastrophic breakdown for anti-empathy (empty outputs, code artifacts).
Implications: The detection-steering gap varies by model. Qwen and Phi-3 maintain bidirectional coherence; Dolphin shows robustness only for empathy enhancement. Safety training may affect steering robustness rather than preventing manipulation, though validation across more models is needed. 

---
# Hierarchical Retrieval with Out-Of-Vocabulary Queries: A Case Study on SNOMED CT 

**Authors**: Jonathon Dilworth, Hui Yang, Jiaoyan Chen, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.16698)  

**Abstract**: SNOMED CT is a biomedical ontology with a hierarchical representation of large-scale concepts. Knowledge retrieval in SNOMED CT is critical for its application, but often proves challenging due to language ambiguity, synonyms, polysemies and so on. This problem is exacerbated when the queries are out-of-vocabulary (OOV), i.e., having no equivalent matchings in the ontology. In this work, we focus on the problem of hierarchical concept retrieval from SNOMED CT with OOV queries, and propose an approach based on language model-based ontology embeddings. For evaluation, we construct OOV queries annotated against SNOMED CT concepts, testing the retrieval of the most direct subsumers and their less relevant ancestors. We find that our method outperforms the baselines including SBERT and two lexical matching methods. While evaluated against SNOMED CT, the approach is generalisable and can be extended to other ontologies. We release code, tools, and evaluation datasets at this https URL. 

---
# Falsely Accused: How AI Detectors Misjudge Slightly Polished Arabic Articles 

**Authors**: Saleh Almohaimeed, Saad Almohaimeed, Mousa Jari, Khaled A. Alobaid, Fahad Alotaibi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16690)  

**Abstract**: Many AI detection models have been developed to counter the presence of articles created by artificial intelligence (AI). However, if a human-authored article is slightly polished by AI, a shift will occur in the borderline decision of these AI detection models, leading them to consider it AI-generated article. This misclassification may result in falsely accusing authors of AI plagiarism and harm the credibility of AI detector models. In English, some efforts were made to meet this challenge, but not in Arabic. In this paper, we generated two datasets. The first dataset contains 800 Arabic articles, half AI-generated and half human-authored. We used it to evaluate 14 Large Language models (LLMs) and commercial AI detectors to assess their ability in distinguishing between human-authored and AI-generated articles. The best 8 models were chosen to act as detectors for our primary concern, which is whether they would consider slightly polished human text as AI-generated. The second dataset, Ar-APT, contains 400 Arabic human-authored articles polished by 10 LLMs using 4 polishing settings, totaling 16400 samples. We use it to evaluate the 8 nominated models and determine whether slight polishing will affect their performance. The results reveal that all AI detectors incorrectly attribute a significant number of articles to AI. The best performing LLM, Claude-4 Sonnet, achieved 83.51%, their performance decreased to 57.63% for articles slightly polished by LLaMA-3. Whereas for the best performing commercial model, this http URL, that achieves 92% accuracy, dropped to 12% for articles slightly polished by Mistral or Gemma-3. 

---
# Concept-Based Interpretability for Toxicity Detection 

**Authors**: Samarth Garg, Deeksha Varshney, Divya Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.16689)  

**Abstract**: The rise of social networks has not only facilitated communication but also allowed the spread of harmful content. Although significant advances have been made in detecting toxic language in textual data, the exploration of concept-based explanations in toxicity detection remains limited. In this study, we leverage various subtype attributes present in toxicity detection datasets, such as obscene, threat, insult, identity attack, and sexual explicit as concepts that serve as strong indicators to identify whether language is toxic. However, disproportionate attribution of concepts towards the target class often results in classification errors. Our work introduces an interpretability technique based on the Concept Gradient (CG) method which provides a more causal interpretation by measuring how changes in concepts directly affect the output of the model. This is an extension of traditional gradient-based methods in machine learning, which often focus solely on input features. We propose the curation of Targeted Lexicon Set, which captures toxic words that contribute to misclassifications in text classification models. To assess the significance of these lexicon sets in misclassification, we compute Word-Concept Alignment (WCA) scores, which quantify the extent to which these words lead to errors due to over-attribution to toxic concepts. Finally, we introduce a lexicon-free augmentation strategy by generating toxic samples that exclude predefined toxic lexicon sets. This approach allows us to examine whether over-attribution persists when explicit lexical overlap is removed, providing insights into the model's attribution on broader toxic language patterns. 

---
# Prompt-Based Value Steering of Large Language Models 

**Authors**: Giulio Antonio Abbo, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2511.16688)  

**Abstract**: Large language models are increasingly used in applications where alignment with human values is critical. While model fine-tuning is often employed to ensure safe responses, this technique is static and does not lend itself to everyday situations involving dynamic values and preferences. In this paper, we present a practical, reproducible, and model-agnostic procedure to evaluate whether a prompt candidate can effectively steer generated text toward specific human values, formalising a scoring method to quantify the presence and gain of target values in generated responses. We apply our method to a variant of the Wizard-Vicuna language model, using Schwartz's theory of basic human values and a structured evaluation through a dialogue dataset. With this setup, we compare a baseline prompt to one explicitly conditioned on values, and show that value steering is possible even without altering the model or dynamically optimising prompts. 

---
# Ellipsoid-Based Decision Boundaries for Open Intent Classification 

**Authors**: Yuetian Zou, Hanlei Zhang, Hua Xu, Songze Li, Long Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.16685)  

**Abstract**: Textual open intent classification is crucial for real-world dialogue systems, enabling robust detection of unknown user intents without prior knowledge and contributing to the robustness of the system. While adaptive decision boundary methods have shown great potential by eliminating manual threshold tuning, existing approaches assume isotropic distributions of known classes, restricting boundaries to balls and overlooking distributional variance along different directions. To address this limitation, we propose EliDecide, a novel method that learns ellipsoid decision boundaries with varying scales along different feature directions. First, we employ supervised contrastive learning to obtain a discriminative feature space for known samples. Second, we apply learnable matrices to parameterize ellipsoids as the boundaries of each known class, offering greater flexibility than spherical boundaries defined solely by centers and radii. Third, we optimize the boundaries via a novelly designed dual loss function that balances empirical and open-space risks: expanding boundaries to cover known samples while contracting them against synthesized pseudo-open samples. Our method achieves state-of-the-art performance on multiple text intent benchmarks and further on a question classification dataset. The flexibility of the ellipsoids demonstrates superior open intent detection capability and strong potential for generalization to more text classification tasks in diverse complex open-world scenarios. 

---
# How Well Do LLMs Understand Tunisian Arabic? 

**Authors**: Mohamed Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16683)  

**Abstract**: Large Language Models (LLMs) are the engines driving today's AI agents. The better these models understand human languages, the more natural and user-friendly the interaction with AI becomes, from everyday devices like computers and smartwatches to any tool that can act intelligently. Yet, the ability of industrial-scale LLMs to comprehend low-resource languages, such as Tunisian Arabic (Tunizi), is often overlooked. This neglect risks excluding millions of Tunisians from fully interacting with AI in their own language, pushing them toward French or English. Such a shift not only threatens the preservation of the Tunisian dialect but may also create challenges for literacy and influence younger generations to favor foreign languages. In this study, we introduce a novel dataset containing parallel Tunizi, standard Tunisian Arabic, and English translations, along with sentiment labels. We benchmark several popular LLMs on three tasks: transliteration, translation, and sentiment analysis. Our results reveal significant differences between models, highlighting both their strengths and limitations in understanding and processing Tunisian dialects. By quantifying these gaps, this work underscores the importance of including low-resource languages in the next generation of AI systems, ensuring technology remains accessible, inclusive, and culturally grounded. 

---
# Bench360: Benchmarking Local LLM Inference from 360° 

**Authors**: Linus Stuhlmann, Mauricio Fadel Argerich, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2511.16682)  

**Abstract**: Running large language models (LLMs) locally is becoming increasingly common. While the growing availability of small open-source models and inference engines has lowered the entry barrier, users now face an overwhelming number of configuration choices. Identifying an optimal configuration -- balancing functional and non-functional requirements -- requires substantial manual effort. While several benchmarks target LLM inference, they are designed for narrow evaluation goals and not user-focused. They fail to integrate relevant system and task-specific metrics into a unified, easy-to-use benchmark that supports multiple inference engines, usage scenarios, and quantization levels. To address this gap, we present Bench360 -- Benchmarking Local LLM Inference from 360°. Bench360 allows users to easily define their own custom tasks along with datasets and relevant task-specific metrics and then automatically benchmarks selected LLMs, inference engines, and quantization levels across different usage scenarios (single stream, batch & server). Bench360 tracks a wide range of metrics, including (1) system metrics -- such as Computing Performance (e.g., latency, throughput), Resource Usage (e.g., energy per query), and Deployment (e.g., cold start time) -- and (2) task-specific metrics such as ROUGE, F1 score or accuracy. We demonstrate Bench360 on four common LLM tasks -- General Knowledge & Reasoning, QA, Summarization and Text-to-SQL -- across three hardware platforms and four state of the art inference engines. Our results reveal several interesting trade-offs between task performance and system-level efficiency, highlighting the differences in inference engines and models. Most importantly, there is no single best setup for local inference, which strongly motivates the need for a framework such as Bench360. 

---
# Towards Hyper-Efficient RAG Systems in VecDBs: Distributed Parallel Multi-Resolution Vector Search 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16681)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have become a dominant approach to augment large language models (LLMs) with external knowledge. However, existing vector database (VecDB) retrieval pipelines rely on flat or single-resolution indexing structures, which cannot adapt to the varying semantic granularity required by diverse user queries. This limitation leads to suboptimal trade-offs between retrieval speed and contextual relevance.
To address this, we propose \textbf{Semantic Pyramid Indexing (SPI)}, a novel multi-resolution vector indexing framework that introduces query-adaptive resolution control for RAG in VecDBs. Unlike existing hierarchical methods that require offline tuning or separate model training, SPI constructs a semantic pyramid over document embeddings and dynamically selects the optimal resolution level per query through a lightweight classifier. This adaptive approach enables progressive retrieval from coarse-to-fine representations, significantly accelerating search while maintaining semantic coverage.
We implement SPI as a plugin for both FAISS and Qdrant backends and evaluate it across multiple RAG tasks including MS MARCO, Natural Questions, and multimodal retrieval benchmarks. SPI achieves up to \textbf{5.7$\times$} retrieval speedup and \textbf{1.8$\times$} memory efficiency gain while improving end-to-end QA F1 scores by up to \textbf{2.5 points} compared to strong baselines. Our theoretical analysis provides guarantees on retrieval quality and latency bounds, while extensive ablation studies validate the contribution of each component. The framework's compatibility with existing VecDB infrastructures makes it readily deployable in production RAG systems. Code is availabe at \href{this https URL}{this https URL\_VecDB}. 

---
# Shona spaCy: A Morphological Analyzer for an Under-Resourced Bantu Language 

**Authors**: Happymore Masoka  

**Link**: [PDF](https://arxiv.org/pdf/2511.16680)  

**Abstract**: Despite rapid advances in multilingual natural language processing (NLP), the Bantu language Shona remains under-served in terms of morphological analysis and language-aware tools. This paper presents Shona spaCy, an open-source, rule-based morphological pipeline for Shona built on the spaCy framework. The system combines a curated JSON lexicon with linguistically grounded rules to model noun-class prefixes (Mupanda 1-18), verbal subject concords, tense-aspect markers, ideophones, and clitics, integrating these into token-level annotations for lemma, part-of-speech, and morphological features. The toolkit is available via pip install shona-spacy, with source code at this https URL and a PyPI release at this https URL. Evaluation on formal and informal Shona corpora yields 90% POS-tagging accuracy and 88% morphological-feature accuracy, while maintaining transparency in its linguistic decisions. By bridging descriptive grammar and computational implementation, Shona spaCy advances NLP accessibility and digital inclusion for Shona speakers and provides a template for morphological analysis tools for other under-resourced Bantu languages. 

---
# Joint Design of Protein Surface and Structure Using a Diffusion Bridge Model 

**Authors**: Guanlue Li, Xufeng Zhao, Fang Wu, Sören Laue  

**Link**: [PDF](https://arxiv.org/pdf/2511.16675)  

**Abstract**: Protein-protein interactions (PPIs) are governed by surface complementarity and hydrophobic interactions at protein interfaces. However, designing diverse and physically realistic protein structure and surfaces that precisely complement target receptors remains a significant challenge in computational protein design. In this work, we introduce PepBridge, a novel framework for the joint design of protein surface and structure that seamlessly integrates receptor surface geometry and biochemical properties. Starting with a receptor surface represented as a 3D point cloud, PepBridge generates complete protein structures through a multi-step process. First, it employs denoising diffusion bridge models (DDBMs) to map receptor surfaces to ligand surfaces. Next, a multi-model diffusion model predicts the corresponding structure, while Shape-Frame Matching Networks ensure alignment between surface geometry and backbone architecture. This integrated approach facilitates surface complementarity, conformational stability, and chemical feasibility. Extensive validation across diverse protein design scenarios demonstrates PepBridge's efficacy in generating structurally viable proteins, representing a significant advancement in the joint design of top-down protein structure. 

---
# Instance Configuration for Sustainable Job Shop Scheduling 

**Authors**: Christian Perez, Carlos March, Miguel A. Salido  

**Link**: [PDF](https://arxiv.org/pdf/2409.18972)  

**Abstract**: The Job Shop Scheduling Problem (JSP) is a pivotal challenge in operations research and is essential for evaluating the effectiveness and performance of scheduling algorithms. Scheduling problems are a crucial domain in combinatorial optimization, where resources (machines) are allocated to job tasks to minimize the completion time (makespan) alongside other objectives like energy consumption. This research delves into the intricacies of JSP, focusing on optimizing performance metrics and minimizing energy consumption while considering various constraints such as deadlines and release dates. Recognizing the multi-dimensional nature of benchmarking in JSP, this study underscores the significance of reference libraries and datasets like JSPLIB in enriching algorithm evaluation. The research highlights the importance of problem instance characteristics, including job and machine numbers, processing times, and machine availability, emphasizing the complexities introduced by energy consumption considerations.
An innovative instance configurator is proposed, equipped with parameters such as the number of jobs, machines, tasks, and speeds, alongside distributions for processing times and energy consumption. The generated instances encompass various configurations, reflecting real-world scenarios and operational constraints. These instances facilitate comprehensive benchmarking and evaluation of scheduling algorithms, particularly in contexts of energy efficiency. A comprehensive set of 500 test instances has been generated and made publicly available, promoting further research and benchmarking in JSP. These instances enable robust analyses and foster collaboration in developing advanced, energy-efficient scheduling solutions by providing diverse scenarios. 

---
