# Pruning as a Game: Equilibrium-Driven Sparsification of Neural Networks 

**Authors**: Zubair Shah, Noaman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2512.22106)  

**Abstract**: Neural network pruning is widely used to reduce model size and computational cost. Yet, most existing methods treat sparsity as an externally imposed constraint, enforced through heuristic importance scores or training-time regularization. In this work, we propose a fundamentally different perspective: pruning as an equilibrium outcome of strategic interaction among model components. We model parameter groups such as weights, neurons, or filters as players in a continuous non-cooperative game, where each player selects its level of participation in the network to balance contribution against redundancy and competition. Within this formulation, sparsity emerges naturally when continued participation becomes a dominated strategy at equilibrium. We analyze the resulting game and show that dominated players collapse to zero participation under mild conditions, providing a principled explanation for pruning behavior. Building on this insight, we derive a simple equilibrium-driven pruning algorithm that jointly updates network parameters and participation variables without relying on explicit importance scores. This work focuses on establishing a principled formulation and empirical validation of pruning as an equilibrium phenomenon, rather than exhaustive architectural or large-scale benchmarking. Experiments on standard benchmarks demonstrate that the proposed approach achieves competitive sparsity-accuracy trade-offs while offering an interpretable, theory-grounded alternative to existing pruning methods. 

---
# SpatialBench: Can Agents Analyze Real-World Spatial Biology Data? 

**Authors**: Kenny Workman, Zhen Yang, Harihara Muralidharan, Hannah Le  

**Link**: [PDF](https://arxiv.org/pdf/2512.21907)  

**Abstract**: Spatial transcriptomics assays are rapidly increasing in scale and complexity, making computational analysis a major bottleneck in biological discovery. Although frontier AI agents have improved dramatically at software engineering and general data analysis, it remains unclear whether they can extract biological insight from messy, real-world spatial datasets. We introduce SpatialBench, a benchmark of 146 verifiable problems derived from practical spatial analysis workflows spanning five spatial technologies and seven task categories. Each problem provides a snapshot of experimental data immediately prior to an analysis step and a deterministic grader that evaluates recovery of a key biological result. Benchmark data on frontier models shows that base model accuracy remains low (20-38% across model families), with strong model-task and model-platform interactions. Harness design has a large empirical effect on performance, indicating that tools, prompts, control flow, and execution environment should be evaluated and improved as first-class objects. SpatialBench serves both as a measurement tool and a diagnostic lens for developing agents that can interact with real spatial datasets faithfully, transparently, and reproducibly. 

---
# Accelerating Scientific Discovery with Autonomous Goal-evolving Agents 

**Authors**: Yuanqi Du, Botao Yu, Tianyu Liu, Tony Shen, Junwu Chen, Jan G. Rittig, Kunyang Sun, Yikun Zhang, Zhangde Song, Bo Zhou, Cassandra Masschelein, Yingze Wang, Haorui Wang, Haojun Jia, Chao Zhang, Hongyu Zhao, Martin Ester, Teresa Head-Gordon, Carla P. Gomes, Huan Sun, Chenru Duan, Philippe Schwaller, Wengong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.21782)  

**Abstract**: There has been unprecedented interest in developing agents that expand the boundary of scientific discovery, primarily by optimizing quantitative objective functions specified by scientists. However, for grand challenges in science , these objectives are only imperfect proxies. We argue that automating objective function design is a central, yet unmet requirement for scientific discovery agents. In this work, we introduce the Scientific Autonomous Goal-evolving Agent (SAGA) to amend this challenge. SAGA employs a bi-level architecture in which an outer loop of LLM agents analyzes optimization outcomes, proposes new objectives, and converts them into computable scoring functions, while an inner loop performs solution optimization under the current objectives. This bi-level design enables systematic exploration of the space of objectives and their trade-offs, rather than treating them as fixed inputs. We demonstrate the framework through a broad spectrum of applications, including antibiotic design, inorganic materials design, functional DNA sequence design, and chemical process design, showing that automating objective formulation can substantially improve the effectiveness of scientific discovery agents. 

---
# Compliance Rating Scheme: A Data Provenance Framework for Generative AI Datasets 

**Authors**: Matyas Bohacek, Ignacio Vilanova Echavarri  

**Link**: [PDF](https://arxiv.org/pdf/2512.21775)  

**Abstract**: Generative Artificial Intelligence (GAI) has experienced exponential growth in recent years, partly facilitated by the abundance of large-scale open-source datasets. These datasets are often built using unrestricted and opaque data collection practices. While most literature focuses on the development and applications of GAI models, the ethical and legal considerations surrounding the creation of these datasets are often neglected. In addition, as datasets are shared, edited, and further reproduced online, information about their origin, legitimacy, and safety often gets lost. To address this gap, we introduce the Compliance Rating Scheme (CRS), a framework designed to evaluate dataset compliance with critical transparency, accountability, and security principles. We also release an open-source Python library built around data provenance technology to implement this framework, allowing for seamless integration into existing dataset-processing and AI training pipelines. The library is simultaneously reactive and proactive, as in addition to evaluating the CRS of existing datasets, it equally informs responsible scraping and construction of new datasets. 

---
# Towards Responsible and Explainable AI Agents with Consensus-Driven Reasoning 

**Authors**: Eranga Bandara, Tharaka Hewa, Ross Gore, Sachin Shetty, Ravi Mukkamala, Peter Foytik, Abdul Rahman, Safdar H. Bouk, Xueping Liang, Amin Hass, Sachini Rajapakse, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21699)  

**Abstract**: Agentic AI represents a major shift in how autonomous systems reason, plan, and execute multi-step tasks through the coordination of Large Language Models (LLMs), Vision Language Models (VLMs), tools, and external services. While these systems enable powerful new capabilities, increasing autonomy introduces critical challenges related to explainability, accountability, robustness, and governance, especially when agent outputs influence downstream actions or decisions. Existing agentic AI implementations often emphasize functionality and scalability, yet provide limited mechanisms for understanding decision rationale or enforcing responsibility across agent interactions. This paper presents a Responsible(RAI) and Explainable(XAI) AI Agent Architecture for production-grade agentic workflows based on multi-model consensus and reasoning-layer governance. In the proposed design, a consortium of heterogeneous LLM and VLM agents independently generates candidate outputs from a shared input context, explicitly exposing uncertainty, disagreement, and alternative interpretations. A dedicated reasoning agent then performs structured consolidation across these outputs, enforcing safety and policy constraints, mitigating hallucinations and bias, and producing auditable, evidence-backed decisions. Explainability is achieved through explicit cross-model comparison and preserved intermediate outputs, while responsibility is enforced through centralized reasoning-layer control and agent-level constraints. We evaluate the architecture across multiple real-world agentic AI workflows, demonstrating that consensus-driven reasoning improves robustness, transparency, and operational trust across diverse application domains. This work provides practical guidance for designing agentic AI systems that are autonomous and scalable, yet responsible and explainable by construction. 

---
# Multiple-play Stochastic Bandits with Prioritized Arm Capacity Sharing 

**Authors**: Hong Xie, Haoran Gu, Yanying Huang, Tao Tan, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2512.21626)  

**Abstract**: This paper proposes a variant of multiple-play stochastic bandits tailored to resource allocation problems arising from LLM applications, edge intelligence, etc. The model is composed of $M$ arms and $K$ plays. Each arm has a stochastic number of capacities, and each unit of capacity is associated with a reward function. Each play is associated with a priority weight. When multiple plays compete for the arm capacity, the arm capacity is allocated in a larger priority weight first manner. Instance independent and instance dependent regret lower bounds of $\Omega( \alpha_1 \sigma \sqrt{KM T} )$ and $\Omega(\alpha_1 \sigma^2 \frac{M}{\Delta} \ln T)$ are proved, where $\alpha_1$ is the largest priority weight and $\sigma$ characterizes the reward tail. When model parameters are given, we design an algorithm named \texttt{MSB-PRS-OffOpt} to locate the optimal play allocation policy with a computational complexity of $O(MK^3)$. Utilizing \texttt{MSB-PRS-OffOpt} as a subroutine, an approximate upper confidence bound (UCB) based algorithm is designed, which has instance independent and instance dependent regret upper bounds matching the corresponding lower bound up to factors of $ \sqrt{K \ln KT }$ and $\alpha_1 K^2$ respectively. To this end, we address nontrivial technical challenges arising from optimizing and learning under a special nonlinear combinatorial utility function induced by the prioritized resource sharing mechanism. 

---
# Democratizing Drug Discovery with an Orchestrated, Knowledge-Driven Multi-Agent Team for User-Guided Therapeutic Design 

**Authors**: Takahide Suzuki, Kazuki Nakanishi, Takashi Fujiwara, Hideyuki Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21623)  

**Abstract**: Therapeutic discovery remains a formidable challenge, impeded by the fragmentation of specialized domains and the execution gap between computational design and physiological validation. Although generative AI offers promise, current models often function as passive assistants rather than as autonomous executors. Here, we introduce OrchestRA, a human-in-the-loop multi-agent platform that unifies biology, chemistry, and pharmacology into an autonomous discovery engine. Unlike static code generators, our agents actively execute simulations and reason the results to drive iterative optimization. Governed by an Orchestrator, a Biologist Agent leverages deep reasoning over a massive knowledge graph (>10 million associations) to pinpoint high-confidence targets; a Chemist Agent autonomously detects structural pockets for de novo design or drug repositioning; and a Pharmacologist Agent evaluates candidates via rigorous physiologically based pharmacokinetic (PBPK) simulations. This architecture establishes a dynamic feedback loop where pharmacokinetic and toxicity profiles directly trigger structural reoptimization. By seamlessly integrating autonomous execution with human guidance, OrchestRA democratizes therapeutic design, transforming drug discovery from a stochastic search to a programmable evidence-based engineering discipline. 

---
# AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design 

**Authors**: Zhishuai Zhang, Xintian Li, Shilong Liu, Aodong Zhang, Lu Jie, Nan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.21613)  

**Abstract**: In this paper, we propose AMS-IO-Agent, a domain-specialized LLM-based agent for structure-aware input/output (I/O) subsystem generation in analog and mixed-signal (AMS) integrated circuits (ICs). The central contribution of this work is a framework that connects natural language design intent with industrial-level AMS IC design deliverables. AMS-IO-Agent integrates two key capabilities: (1) a structured domain knowledge base that captures reusable constraints and design conventions; (2) design intent structuring, which converts ambiguous user intent into verifiable logic steps using JSON and Python as intermediate formats. We further introduce AMS-IO-Bench, a benchmark for wirebond-packaged AMS I/O ring automation. On this benchmark, AMS-IO-Agent achieves over 70\% DRC+LVS pass rate and reduces design turnaround time from hours to minutes, outperforming the baseline LLM. Furthermore, an agent-generated I/O ring was fabricated and validated in a 28 nm CMOS tape-out, demonstrating the practical effectiveness of the approach in real AMS IC design flows. To our knowledge, this is the first reported human-agent collaborative AMS IC design in which an LLM-based agent completes a nontrivial subtask with outputs directly used in silicon. 

---
# A Medical Multimodal Diagnostic Framework Integrating Vision-Language Models and Logic Tree Reasoning 

**Authors**: Zelin Zang, Wenyi Gu, Siqi Ma, Dan Yang, Yue Shen, Zhu Zhang, Guohui Fan, Wing-Kuen Ling, Fuji Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21583)  

**Abstract**: With the rapid growth of large language models (LLMs) and vision-language models (VLMs) in medicine, simply integrating clinical text and medical imaging does not guarantee reliable reasoning. Existing multimodal models often produce hallucinations or inconsistent chains of thought, limiting clinical trust. We propose a diagnostic framework built upon LLaVA that combines vision-language alignment with logic-regularized reasoning. The system includes an input encoder for text and images, a projection module for cross-modal alignment, a reasoning controller that decomposes diagnostic tasks into steps, and a logic tree generator that assembles stepwise premises into verifiable conclusions. Evaluations on MedXpertQA and other benchmarks show that our method improves diagnostic accuracy and yields more interpretable reasoning traces on multimodal tasks, while remaining competitive on text-only settings. These results suggest a promising step toward trustworthy multimodal medical AI. 

---
# NEMO-4-PAYPAL: Leveraging NVIDIA's Nemo Framework for empowering PayPal's Commerce Agent 

**Authors**: Ali Sahami, Sudhanshu Garg, Andrew Wang, Chaitanya Kulkarni, Farhad Farahani, Sean Yun-Shiuan Chuang, Jian Wan, Srinivasan Manoharan, Uma Kona, Nitin Sharma, Linsey Pang, Prakhar Mehrotra, Jessica Clark, Mark Moyou  

**Link**: [PDF](https://arxiv.org/pdf/2512.21578)  

**Abstract**: We present the development and optimization of PayPal's Commerce Agent, powered by NEMO-4-PAYPAL, a multi-agent system designed to revolutionize agentic commerce on the PayPal platform. Through our strategic partnership with NVIDIA, we leveraged the NeMo Framework for LLM model fine-tuning to enhance agent performance. Specifically, we optimized the Search and Discovery agent by replacing our base model with a fine-tuned Nemotron small language model (SLM).
We conducted comprehensive experiments using the llama3.1-nemotron-nano-8B-v1 architecture, training LoRA-based models through systematic hyperparameter sweeps across learning rates, optimizers (Adam, AdamW), cosine annealing schedules, and LoRA ranks. Our contributions include: (1) the first application of NVIDIA's NeMo Framework to commerce-specific agent optimization, (2) LLM powered fine-tuning strategy for retrieval-focused commerce tasks, (3) demonstration of significant improvements in latency and cost while maintaining agent quality, and (4) a scalable framework for multi-agent system optimization in production e-commerce environments. Our results demonstrate that the fine-tuned Nemotron SLM effectively resolves the key performance issue in the retrieval component, which represents over 50\% of total agent response time, while maintaining or enhancing overall system performance. 

---
# Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model 

**Authors**: Yanhao Li, Lu Ma, Jiaran Zhang, Lexiang Tang, Wentao Zhang, Guibo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.21540)  

**Abstract**: Existing approaches typically rely on fixed length penalties, but such penalties are hard to tune and fail to adapt to the evolving reasoning abilities of LLMs, leading to suboptimal trade-offs between accuracy and conciseness. To address this challenge, we propose Leash (adaptive LEngth penAlty and reward SHaping), a reinforcement learning framework for efficient reasoning in LLMs. We formulate length control as a constrained optimization problem and employ a Lagrangian primal-dual method to dynamically adjust the penalty coefficient. When generations exceed the target length, the penalty is intensified; when they are shorter, it is relaxed. This adaptive mechanism guides models toward producing concise reasoning without sacrificing task performance. Experiments on Deepseek-R1-Distill-Qwen-1.5B and Qwen3-4B-Thinking-2507 show that Leash reduces the average reasoning length by 60% across diverse tasks - including in-distribution mathematical reasoning and out-of-distribution domains such as coding and instruction following - while maintaining competitive performance. Our work thus presents a practical and effective paradigm for developing controllable and efficient LLMs that balance reasoning capabilities with computational budgets. 

---
# LogicLens: Visual-Logical Co-Reasoning for Text-Centric Forgery Analysis 

**Authors**: Fanwei Zeng, Changtao Miao, Jing Huang, Zhiya Tan, Shutao Gong, Xiaoming Yu, Yang Wang, Huazhe Tan, Weibin Yao, Jianshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.21482)  

**Abstract**: Sophisticated text-centric forgeries, fueled by rapid AIGC advancements, pose a significant threat to societal security and information authenticity. Current methods for text-centric forgery analysis are often limited to coarse-grained visual analysis and lack the capacity for sophisticated reasoning. Moreover, they typically treat detection, grounding, and explanation as discrete sub-tasks, overlooking their intrinsic relationships for holistic performance enhancement. To address these challenges, we introduce LogicLens, a unified framework for Visual-Textual Co-reasoning that reformulates these objectives into a joint task. The deep reasoning of LogicLens is powered by our novel Cross-Cues-aware Chain of Thought (CCT) mechanism, which iteratively cross-validates visual cues against textual logic. To ensure robust alignment across all tasks, we further propose a weighted multi-task reward function for GRPO-based optimization. Complementing this framework, we first designed the PR$^2$ (Perceiver, Reasoner, Reviewer) pipeline, a hierarchical and iterative multi-agent system that generates high-quality, cognitively-aligned annotations. Then, we constructed RealText, a diverse dataset comprising 5,397 images with fine-grained annotations, including textual explanations, pixel-level segmentation, and authenticity labels for model training. Extensive experiments demonstrate the superiority of LogicLens across multiple benchmarks. In a zero-shot evaluation on T-IC13, it surpasses the specialized framework by 41.4% and GPT-4o by 23.4% in macro-average F1 score. Moreover, on the challenging dense-text T-SROIE dataset, it establishes a significant lead over other MLLM-based methods in mF1, CSS, and the macro-average F1. Our dataset, model, and code will be made publicly available. 

---
# Three-way decision with incomplete information based on similarity and satisfiability 

**Authors**: Junfang Luo, Mengjun Hu, Keyun Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.21421)  

**Abstract**: Three-way decision is widely applied with rough set theory to learn classification or decision rules. The approaches dealing with complete information are well established in the literature, including the two complementary computational and conceptual formulations. The computational formulation uses equivalence relations, and the conceptual formulation uses satisfiability of logic formulas. In this paper, based on a briefly review of these two formulations, we generalize both formulations into three-way decision with incomplete information that is more practical in real-world applications. For the computational formulation, we propose a new measure of similarity degree of objects as a generalization of equivalence relations. Based on it, we discuss two approaches to three-way decision using alpha-similarity classes and approximability of objects, respectively. For the conceptual formulation, we propose a measure of satisfiability degree of formulas as a quantitative generalization of satisfiability with complete information. Based on it, we study two approaches to three-way decision using alpha-meaning sets of formulas and confidence of formulas, respectively. While using similarity classes is a common method of analyzing incomplete information in the literature, the proposed concept of approximability and the two approaches in conceptual formulation point out new promising directions. 

---
# Feasible strategies in three-way conflict analysis with three-valued ratings 

**Authors**: Jing Liu, Mengjun Hu, Guangming Lang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21420)  

**Abstract**: Most existing work on three-way conflict analysis has focused on trisecting agent pairs, agents, or issues, which contributes to understanding the nature of conflicts but falls short in addressing their resolution. Specifically, the formulation of feasible strategies, as an essential component of conflict resolution and mitigation, has received insufficient scholarly attention. Therefore, this paper aims to investigate feasible strategies from two perspectives of consistency and non-consistency. Particularly, we begin with computing the overall rating of a clique of agents based on positive and negative similarity degrees. Afterwards, considering the weights of both agents and issues, we propose weighted consistency and non-consistency measures, which are respectively used to identify the feasible strategies for a clique of agents. Algorithms are developed to identify feasible strategies, $L$-order feasible strategies, and the corresponding optimal ones. Finally, to demonstrate the practicality, effectiveness, and superiority of the proposed models, we apply them to two commonly used case studies on NBA labor negotiations and development plans for Gansu Province and conduct a sensitivity analysis on parameters and a comparative analysis with existing state-of-the-art conflict analysis approaches. The comparison results demonstrate that our conflict resolution models outperform the conventional approaches by unifying weighted agent-issue evaluation with consistency and non-consistency measures to enable the systematic identification of not only feasible strategies but also optimal solutions. 

---
# Three-way conflict analysis based on alliance and conflict functions 

**Authors**: Junfang Luo, Mengjun Hu, Guangming Lang, Xin Yang, Keyun Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.21419)  

**Abstract**: Trisecting agents, issues, and agent pairs are essential topics of three-way conflict analysis. They have been commonly studied based on either a rating or an auxiliary function. A rating function defines the positive, negative, or neutral ratings of agents on issues. An auxiliary function defines the alliance, conflict, and neutrality relations between agents. These functions measure two opposite aspects in a single function, leading to challenges in interpreting their aggregations over a group of issues or agents. For example, when studying agent relations regarding a set of issues, a standard aggregation takes the average of an auxiliary function concerning single issues. Therefore, a pair of alliance +1 and conflict -1 relations will produce the same result as a pair of neutrality 0 relations, although the attitudes represented by the two pairs are very different. To clarify semantics, we separate the two opposite aspects in an auxiliary function into a pair of alliance and conflict functions. Accordingly, we trisect the agents, issues, and agent pairs and investigate their applications in solving a few crucial questions in conflict analysis. Particularly, we explore the concepts of alliance sets and strategies. A real-world application is given to illustrate the proposed models. 

---
# A Study of Solving Life-and-Death Problems in Go Using Relevance-Zone Based Solvers 

**Authors**: Chung-Chin Shih, Ti-Rong Wu, Ting Han Wei, Yu-Shan Hsu, Hung Guei, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21365)  

**Abstract**: This paper analyzes the behavior of solving Life-and-Death (L&D) problems in the game of Go using current state-of-the-art computer Go solvers with two techniques: the Relevance-Zone Based Search (RZS) and the relevance-zone pattern table. We examined the solutions derived by relevance-zone based solvers on seven L&D problems from the renowned book "Life and Death Dictionary" written by Cho Chikun, a Go grandmaster, and found several interesting results. First, for each problem, the solvers identify a relevance-zone that highlights the critical areas for solving. Second, the solvers discover a series of patterns, including some that are rare. Finally, the solvers even find different answers compared to the given solutions for two problems. We also identified two issues with the solver: (a) it misjudges values of rare patterns, and (b) it tends to prioritize living directly rather than maximizing territory, which differs from the behavior of human Go players. We suggest possible approaches to address these issues in future work. Our code and data are available at this https URL. 

---
# From Visual Perception to Deep Empathy: An Automated Assessment Framework for House-Tree-Person Drawings Using Multimodal LLMs and Multi-Agent Collaboration 

**Authors**: Shuide Wen, Yu Sun, Beier Ku, Zhi Gao, Lijun Ma, Yang Yang, Can Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.21360)  

**Abstract**: Background: The House-Tree-Person (HTP) drawing test, introduced by John Buck in 1948, remains a widely used projective technique in clinical psychology. However, it has long faced challenges such as heterogeneous scoring standards, reliance on examiners subjective experience, and a lack of a unified quantitative coding system.
Results: Quantitative experiments showed that the mean semantic similarity between Multimodal Large Language Model (MLLM) interpretations and human expert interpretations was approximately 0.75 (standard deviation about 0.05). In structurally oriented expert data sets, this similarity rose to 0.85, indicating expert-level baseline comprehension. Qualitative analyses demonstrated that the multi-agent system, by integrating social-psychological perspectives and destigmatizing narratives, effectively corrected visual hallucinations and produced psychological reports with high ecological validity and internal coherence.
Conclusions: The findings confirm the potential of multimodal large models as standardized tools for projective assessment. The proposed multi-agent framework, by dividing roles, decouples feature recognition from psychological inference and offers a new paradigm for digital mental-health services.
Keywords: House-Tree-Person test; multimodal large language model; multi-agent collaboration; cosine similarity; computational psychology; artificial intelligence 

---
# Agentic Structured Graph Traversal for Root Cause Analysis of Code-related Incidents in Cloud Applications 

**Authors**: Shengkun Cui, Rahul Krishna, Saurabh Jha, Ravishankar K. Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2512.22113)  

**Abstract**: Cloud incidents pose major operational challenges in production, with unresolved production cloud incidents cost on average over $2M per hour. Prior research identifies code- and configuration-related issues as the predominant category of root causes in cloud incidents. This paper introduces PRAXIS, an orchestrator that manages and deploys an agentic workflow for diagnosing code- and configuration-caused cloud incidents. PRAXIS employs an LLM-driven structured traversal over two types of graph: (1) a service dependency graph (SDG) that captures microservice-level dependencies; and (2) a hammock-block program dependence graph (PDG) that captures code-level dependencies for each microservice. Together, these graphs encode microservice- and code-level dependencies and the LLM acts as a traversal policy over these graphs, moving between services and code dependencies to localize and explain failures. Compared to state-of-the-art ReAct baselines, PRAXIS improves RCA accuracy by up to 3.1x while reducing token consumption by 3.8x. PRAXIS is demonstrated on a set of 30 comprehensive real-world incidents that is being compiled into an RCA benchmark. 

---
# A2P-Vis: an Analyzer-to-Presenter Agentic Pipeline for Visual Insights Generation and Reporting 

**Authors**: Shuyu Gan, Renxiang Wang, James Mooney, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22101)  

**Abstract**: Automating end-to-end data science pipeline with AI agents still stalls on two gaps: generating insightful, diverse visual evidence and assembling it into a coherent, professional report. We present A2P-Vis, a two-part, multi-agent pipeline that turns raw datasets into a high-quality data-visualization report. The Data Analyzer orchestrates profiling, proposes diverse visualization directions, generates and executes plotting code, filters low-quality figures with a legibility checker, and elicits candidate insights that are automatically scored for depth, correctness, specificity, depth and actionability. The Presenter then orders topics, composes chart-grounded narratives from the top-ranked insights, writes justified transitions, and revises the document for clarity and consistency, yielding a coherent, publication-ready report. Together, these agents convert raw data into curated materials (charts + vetted insights) and into a readable narrative without manual glue work. We claim that by coupling a quality-assured Analyzer with a narrative Presenter, A2P-Vis operationalizes co-analysis end-to-end, improving the real-world usefulness of automated data analysis for practitioners. For the complete dataset report, please see: this https URL. 

---
# Introducing TrGLUE and SentiTurca: A Comprehensive Benchmark for Turkish General Language Understanding and Sentiment Analysis 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2512.22100)  

**Abstract**: Evaluating the performance of various model architectures, such as transformers, large language models (LLMs), and other NLP systems, requires comprehensive benchmarks that measure performance across multiple dimensions. Among these, the evaluation of natural language understanding (NLU) is particularly critical as it serves as a fundamental criterion for assessing model capabilities. Thus, it is essential to establish benchmarks that enable thorough evaluation and analysis of NLU abilities from diverse perspectives. While the GLUE benchmark has set a standard for evaluating English NLU, similar benchmarks have been developed for other languages, such as CLUE for Chinese, FLUE for French, and JGLUE for Japanese. However, no comparable benchmark currently exists for the Turkish language. To address this gap, we introduce TrGLUE, a comprehensive benchmark encompassing a variety of NLU tasks for Turkish. In addition, we present SentiTurca, a specialized benchmark for sentiment analysis. To support researchers, we also provide fine-tuning and evaluation code for transformer-based models, facilitating the effective use of these benchmarks. TrGLUE comprises Turkish-native corpora curated to mirror the domains and task formulations of GLUE-style evaluations, with labels obtained through a semi-automated pipeline that combines strong LLM-based annotation, cross-model agreement checks, and subsequent human validation. This design prioritizes linguistic naturalness, minimizes direct translation artifacts, and yields a scalable, reproducible workflow. With TrGLUE, our goal is to establish a robust evaluation framework for Turkish NLU, empower researchers with valuable resources, and provide insights into generating high-quality semi-automated datasets. 

---
# Unifying Learning Dynamics and Generalization in Transformers Scaling Law 

**Authors**: Chiwun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22088)  

**Abstract**: The scaling law, a cornerstone of Large Language Model (LLM) development, predicts improvements in model performance with increasing computational resources. Yet, while empirically validated, its theoretical underpinnings remain poorly understood. This work formalizes the learning dynamics of transformer-based language models as an ordinary differential equation (ODE) system, then approximates this process to kernel behaviors. Departing from prior toy-model analyses, we rigorously analyze stochastic gradient descent (SGD) training for multi-layer transformers on sequence-to-sequence data with arbitrary data distribution, closely mirroring real-world conditions. Our analysis characterizes the convergence of generalization error to the irreducible risk as computational resources scale with data, especially during the optimization process.
We establish a theoretical upper bound on excess risk characterized by a distinct phase transition. In the initial optimization phase, the excess risk decays exponentially relative to the computational cost ${\sf C}$. However, once a specific resource allocation threshold is crossed, the system enters a statistical phase, where the generalization error follows a power-law decay of $\Theta(\mathsf{C}^{-1/6})$. Beyond this unified framework, our theory derives isolated scaling laws for model size, training time, and dataset size, elucidating how each variable independently governs the upper bounds of generalization. 

---
# StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars 

**Authors**: Zhiyao Sun, Ziqiao Peng, Yifeng Ma, Yi Chen, Zhengguang Zhou, Zixiang Zhou, Guozhen Zhang, Youliang Zhang, Yuan Zhou, Qinglin Lu, Yong-Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22065)  

**Abstract**: Real-time, streaming interactive avatars represent a critical yet challenging goal in digital human research. Although diffusion-based human avatar generation methods achieve remarkable success, their non-causal architecture and high computational costs make them unsuitable for streaming. Moreover, existing interactive approaches are typically limited to head-and-shoulder region, limiting their ability to produce gestures and body motions. To address these challenges, we propose a two-stage autoregressive adaptation and acceleration framework that applies autoregressive distillation and adversarial refinement to adapt a high-fidelity human video diffusion model for real-time, interactive streaming. To ensure long-term stability and consistency, we introduce three key components: a Reference Sink, a Reference-Anchored Positional Re-encoding (RAPR) strategy, and a Consistency-Aware Discriminator. Building on this framework, we develop a one-shot, interactive, human avatar model capable of generating both natural talking and listening behaviors with coherent gestures. Extensive experiments demonstrate that our method achieves state-of-the-art performance, surpassing existing approaches in generation quality, real-time efficiency, and interaction naturalness. Project page: this https URL . 

---
# From In Silico to In Vitro: Evaluating Molecule Generative Models for Hit Generation 

**Authors**: Nagham Osman, Vittorio Lembo, Giovanni Bottegoni, Laura Toni  

**Link**: [PDF](https://arxiv.org/pdf/2512.22031)  

**Abstract**: Hit identification is a critical yet resource-intensive step in the drug discovery pipeline, traditionally relying on high-throughput screening of large compound libraries. Despite advancements in virtual screening, these methods remain time-consuming and costly. Recent progress in deep learning has enabled the development of generative models capable of learning complex molecular representations and generating novel compounds de novo. However, using ML to replace the entire drug-discovery pipeline is highly challenging. In this work, we rather investigate whether generative models can replace one step of the pipeline: hit-like molecule generation. To the best of our knowledge, this is the first study to explicitly frame hit-like molecule generation as a standalone task and empirically test whether generative models can directly support this stage of the drug discovery pipeline. Specifically, we investigate if such models can be trained to generate hit-like molecules, enabling direct incorporation into, or even substitution of, traditional hit identification workflows. We propose an evaluation framework tailored to this task, integrating physicochemical, structural, and bioactivity-related criteria within a multi-stage filtering pipeline that defines the hit-like chemical space. Two autoregressive and one diffusion-based generative models were benchmarked across various datasets and training settings, with outputs assessed using standard metrics and target-specific docking scores. Our results show that these models can generate valid, diverse, and biologically relevant compounds across multiple targets, with a few selected GSK-3$\beta$ hits synthesized and confirmed active in vitro. We also identify key limitations in current evaluation metrics and available training data. 

---
# LibContinual: A Comprehensive Library towards Realistic Continual Learning 

**Authors**: Wenbin Li, Shangge Liu, Borui Kang, Yiyang Chen, KaXuan Lew, Yang Chen, Yinghuan Shi, Lei Wang, Yang Gao, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.22029)  

**Abstract**: A fundamental challenge in Continual Learning (CL) is catastrophic forgetting, where adapting to new tasks degrades the performance on previous ones. While the field has evolved with diverse methods, this rapid surge in diverse methodologies has culminated in a fragmented research landscape. The lack of a unified framework, including inconsistent implementations, conflicting dependencies, and varying evaluation protocols, makes fair comparison and reproducible research increasingly difficult. To address this challenge, we propose LibContinual, a comprehensive and reproducible library designed to serve as a foundational platform for realistic CL. Built upon a high-cohesion, low-coupling modular architecture, LibContinual integrates 19 representative algorithms across five major methodological categories, providing a standardized execution environment. Meanwhile, leveraging this unified framework, we systematically identify and investigate three implicit assumptions prevalent in mainstream evaluation: (1) offline data accessibility, (2) unregulated memory resources, and (3) intra-task semantic homogeneity. We argue that these assumptions often overestimate the real-world applicability of CL methods. Through our comprehensive analysis using strict online CL settings, a novel unified memory budget protocol, and a proposed category-randomized setting, we reveal significant performance drops in many representative CL methods when subjected to these real-world constraints. Our study underscores the necessity of resource-aware and semantically robust CL strategies, and offers LibContinual as a foundational toolkit for future research in realistic continual learning. The source code is available from \href{this https URL}{this https URL}. 

---
# Meta-Learning-Based Handover Management in NextG O-RAN 

**Authors**: Michail Kalntis, George Iosifidis, José Suárez-Varela, Andra Lutu, Fernando A. Kuipers  

**Link**: [PDF](https://arxiv.org/pdf/2512.22022)  

**Abstract**: While traditional handovers (THOs) have served as a backbone for mobile connectivity, they increasingly suffer from failures and delays, especially in dense deployments and high-frequency bands. To address these limitations, 3GPP introduced Conditional Handovers (CHOs) that enable proactive cell reservations and user-driven execution. However, both handover (HO) types present intricate trade-offs in signaling, resource usage, and reliability. This paper presents unique, countrywide mobility management datasets from a top-tier mobile network operator (MNO) that offer fresh insights into these issues and call for adaptive and robust HO control in next-generation networks. Motivated by these findings, we propose CONTRA, a framework that, for the first time, jointly optimizes THOs and CHOs within the O-RAN architecture. We study two variants of CONTRA: one where users are a priori assigned to one of the HO types, reflecting distinct service or user-specific requirements, as well as a more dynamic formulation where the controller decides on-the-fly the HO type, based on system conditions and needs. To this end, it relies on a practical meta-learning algorithm that adapts to runtime observations and guarantees performance comparable to an oracle with perfect future information (universal no-regret). CONTRA is specifically designed for near-real-time deployment as an O-RAN xApp and aligns with the 6G goals of flexible and intelligent control. Extensive evaluations leveraging crowdsourced datasets show that CONTRA improves user throughput and reduces both THO and CHO switching costs, outperforming 3GPP-compliant and Reinforcement Learning (RL) baselines in dynamic and real-world scenarios. 

---
# LongFly: Long-Horizon UAV Vision-and-Language Navigation with Spatiotemporal Context Integration 

**Authors**: Wen Jiang, Li Wang, Kangyao Huang, Wei Fan, Jinyuan Liu, Shaoyu Liu, Hongwei Duan, Bin Xu, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2512.22010)  

**Abstract**: Unmanned aerial vehicles (UAVs) are crucial tools for post-disaster search and rescue, facing challenges such as high information density, rapid changes in viewpoint, and dynamic structures, especially in long-horizon navigation. However, current UAV vision-and-language navigation(VLN) methods struggle to model long-horizon spatiotemporal context in complex environments, resulting in inaccurate semantic alignment and unstable path planning. To this end, we propose LongFly, a spatiotemporal context modeling framework for long-horizon UAV VLN. LongFly proposes a history-aware spatiotemporal modeling strategy that transforms fragmented and redundant historical data into structured, compact, and expressive representations. First, we propose the slot-based historical image compression module, which dynamically distills multi-view historical observations into fixed-length contextual representations. Then, the spatiotemporal trajectory encoding module is introduced to capture the temporal dynamics and spatial structure of UAV trajectories. Finally, to integrate existing spatiotemporal context with current observations, we design the prompt-guided multimodal integration module to support time-based reasoning and robust waypoint prediction. Experimental results demonstrate that LongFly outperforms state-of-the-art UAV VLN baselines by 7.89\% in success rate and 6.33\% in success weighted by path length, consistently across both seen and unseen environments. 

---
# LVLM-Aided Alignment of Task-Specific Vision Models 

**Authors**: Alexander Koebler, Lukas Kuhn, Ingo Thon, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2512.21985)  

**Abstract**: In high-stakes domains, small task-specific vision models are crucial due to their low computational requirements and the availability of numerous methods to explain their results. However, these explanations often reveal that the models do not align well with human domain knowledge, relying instead on spurious correlations. This might result in brittle behavior once deployed in the real-world. To address this issue, we introduce a novel and efficient method for aligning small task-specific vision models with human domain knowledge by leveraging the generalization capabilities of a Large Vision Language Model (LVLM). Our LVLM-Aided Visual Alignment (LVLM-VA) method provides a bidirectional interface that translates model behavior into natural language and maps human class-level specifications to image-level critiques, enabling effective interaction between domain experts and the model. Our method demonstrates substantial improvement in aligning model behavior with human specifications, as validated on both synthetic and real-world datasets. We show that it effectively reduces the model's dependence on spurious features and on group-specific biases, without requiring fine-grained feedback. 

---
# Unsupervised Anomaly Detection in Brain MRI via Disentangled Anatomy Learning 

**Authors**: Tao Yang, Xiuying Wang, Hao Liu, Guanzhong Gong, Lian-Ming Wu, Yu-Ping Wang, Lisheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21924)  

**Abstract**: Detection of various lesions in brain MRI is clinically critical, but challenging due to the diversity of lesions and variability in imaging conditions. Current unsupervised learning methods detect anomalies mainly through reconstructing abnormal images into pseudo-healthy images (PHIs) by normal samples learning and then analyzing differences between images. However, these unsupervised models face two significant limitations: restricted generalizability to multi-modality and multi-center MRIs due to their reliance on the specific imaging information in normal training data, and constrained performance due to abnormal residuals propagated from input images to reconstructed PHIs. To address these limitations, two novel modules are proposed, forming a new PHI reconstruction framework. Firstly, the disentangled representation module is proposed to improve generalizability by decoupling brain MRI into imaging information and essential imaging-invariant anatomical images, ensuring that the reconstruction focuses on the anatomy. Specifically, brain anatomical priors and a differentiable one-hot encoding operator are introduced to constrain the disentanglement results and enhance the disentanglement stability. Secondly, the edge-to-image restoration module is designed to reconstruct high-quality PHIs by restoring the anatomical representation from the high-frequency edge information of anatomical images, and then recoupling the disentangled imaging information. This module not only suppresses abnormal residuals in PHI by reducing abnormal pixels input through edge-only input, but also effectively reconstructs normal regions using the preserved structural details in the edges. Evaluated on nine public datasets (4,443 patients' MRIs from multiple centers), our method outperforms 17 SOTA methods, achieving absolute improvements of +18.32% in AP and +13.64% in DSC. 

---
# Semiparametric Preference Optimization: Your Language Model is Secretly a Single-Index Model 

**Authors**: Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2512.21917)  

**Abstract**: Aligning large language models to preference data is commonly implemented by assuming a known link function between the distribution of observed preferences and the unobserved rewards (e.g., a logistic link as in Bradley-Terry). If the link is wrong, however, inferred rewards can be biased and policies be misaligned. We study policy alignment to preferences under an unknown and unrestricted link. We consider an $f$-divergence-constrained reward maximization problem and show that realizability of the solution in a policy class implies a semiparametric single-index binary choice model, where a scalar-valued index determined by a policy captures the dependence on demonstrations and the rest of the preference distribution is an unrestricted function thereof. Rather than focus on estimation of identifiable finite-dimensional structural parameters in the index as in econometrics, we focus on policy learning, focusing on error to the optimal policy and allowing unidentifiable and nonparametric indices. We develop a variety of policy learners based on profiling the link function, orthogonalizing the link function, and using link-agnostic bipartite ranking objectives. We analyze these and provide finite-sample policy error bounds that depend on generic functional complexity measures of the index class. We further consider practical implementations using first-order optimization suited to neural networks and batched data. The resulting methods are robust to unknown preference noise distribution and scale, while preserving the direct optimization of policies without explicitly fitting rewards. 

---
# Flexible Multitask Learning with Factorized Diffusion Policy 

**Authors**: Chaoqi Liu, Haonan Chen, Sigmund H. Høeg, Shaoxiong Yao, Yunzhu Li, Kris Hauser, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.21898)  

**Abstract**: Multitask learning poses significant challenges due to the highly multimodal and diverse nature of robot action distributions. However, effectively fitting policies to these complex task distributions is often difficult, and existing monolithic models often underfit the action distribution and lack the flexibility required for efficient adaptation. We introduce a novel modular diffusion policy framework that factorizes complex action distributions into a composition of specialized diffusion models, each capturing a distinct sub-mode of the behavior space for a more effective overall policy. In addition, this modular structure enables flexible policy adaptation to new tasks by adding or fine-tuning components, which inherently mitigates catastrophic forgetting. Empirically, across both simulation and real-world robotic manipulation settings, we illustrate how our method consistently outperforms strong modular and monolithic baselines. 

---
# MMCTOP: A Multimodal Textualization and Mixture-of-Experts Framework for Clinical Trial Outcome Prediction 

**Authors**: Carolina Aparício, Qi Shi, Bo Wen, Tesfaye Yadete, Qiwei Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.21897)  

**Abstract**: Addressing the challenge of multimodal data fusion in high-dimensional biomedical informatics, we propose MMCTOP, a MultiModal Clinical-Trial Outcome Prediction framework that integrates heterogeneous biomedical signals spanning (i) molecular structure representations, (ii) protocol metadata and long-form eligibility narratives, and (iii) disease ontologies. MMCTOP couples schema-guided textualization and input-fidelity validation with modality-aware representation learning, in which domain-specific encoders generate aligned embeddings that are fused by a transformer backbone augmented with a drug-disease-conditioned sparse Mixture-of-Experts (SMoE). This design explicitly supports specialization across therapeutic and design subspaces while maintaining scalable computation through top-k routing. MMCTOP achieves consistent improvements in precision, F1, and AUC over unimodal and multimodal baselines on benchmark datasets, and ablations show that schema-guided textualization and selective expert routing contribute materially to performance and stability. We additionally apply temperature scaling to obtain calibrated probabilities, ensuring reliable risk estimation for downstream decision support. Overall, MMCTOP advances multimodal trial modeling by combining controlled narrative normalization, context-conditioned expert fusion, and operational safeguards aimed at auditability and reproducibility in biomedical informatics. 

---
# Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space 

**Authors**: Weichen Zhang, Peizhi Tang, Xin Zeng, Fanhang Man, Shiquan Yu, Zichao Dai, Baining Zhao, Hongjin Chen, Yu Shang, Wei Wu, Chen Gao, Xinlei Chen, Xin Wang, Yong Li, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21887)  

**Abstract**: Unmanned aerial vehicles (UAVs) have emerged as powerful embodied agents. One of the core abilities is autonomous navigation in large-scale three-dimensional environments. Existing navigation policies, however, are typically optimized for low-level objectives such as obstacle avoidance and trajectory smoothness, lacking the ability to incorporate high-level semantics into planning. To bridge this gap, we propose ANWM, an aerial navigation world model that predicts future visual observations conditioned on past frames and actions, thereby enabling agents to rank candidate trajectories by their semantic plausibility and navigational utility. ANWM is trained on 4-DoF UAV trajectories and introduces a physics-inspired module: Future Frame Projection (FFP), which projects past frames into future viewpoints to provide coarse geometric priors. This module mitigates representational uncertainty in long-distance visual generation and captures the mapping between 3D trajectories and egocentric observations. Empirical results demonstrate that ANWM significantly outperforms existing world models in long-distance visual forecasting and improves UAV navigation success rates in large-scale environments. 

---
# Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models 

**Authors**: Tingyang Sun, Ting He, Bo Ji, Parimal Parag  

**Link**: [PDF](https://arxiv.org/pdf/2512.21884)  

**Abstract**: Large language models have demonstrated extraordinary performance in many AI tasks but are expensive to use, even after training, due to their requirement of high-end GPUs. Recently, a distributed system called PETALS was developed to lower the barrier for deploying LLMs by splitting the model blocks across multiple servers with low-end GPUs distributed over the Internet, which was much faster than swapping the model parameters between the GPU memory and other cheaper but slower local storage media. However, the performance of such a distributed system critically depends on the resource allocation, and how to do so optimally remains unknown. In this work, we present the first systematic study of the resource allocation problem in distributed LLM inference, with focus on two important decisions: block placement and request routing. Our main results include: experimentally validated performance models that can predict the inference performance under given block placement and request routing decisions, a formulation of the offline optimization of block placement and request routing as a mixed integer linear programming problem together with the NP-hardness proof and a polynomial-complexity algorithm with guaranteed performance, and an adaptation of the offline algorithm for the online setting with the same performance guarantee under bounded load. Through both experiments and experimentally-validated simulations, we have verified that the proposed solution can substantially reduce the inference time compared to the state-of-the-art solution in diverse settings with geographically-distributed servers. As a byproduct, we have also developed a light-weighted CPU-only simulator capable of predicting the performance of distributed LLM inference on GPU servers, which can evaluate large deployments and facilitate future research for researchers with limited GPU access. 

---
# MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting 

**Authors**: Marc S. Montalvo, Hamed Yaghoobian  

**Link**: [PDF](https://arxiv.org/pdf/2512.21878)  

**Abstract**: Recent advances in large language models (LLMs) are transforming data-intensive domains, with finance representing a high-stakes environment where transparent and reproducible analysis of heterogeneous signals is essential. Traditional quantitative methods remain vulnerable to survivorship bias, while many AI-driven approaches struggle with signal integration, reproducibility, and computational efficiency. We introduce MASFIN, a modular multi-agent framework that integrates LLMs with structured financial metrics and unstructured news, while embedding explicit bias-mitigation protocols. The system leverages GPT-4.1-nano for reproducability and cost-efficient inference and generates weekly portfolios of 15-30 equities with allocation weights optimized for short-term performance. In an eight-week evaluation, MASFIN delivered a 7.33% cumulative return, outperforming the S&P 500, NASDAQ-100, and Dow Jones benchmarks in six of eight weeks, albeit with higher volatility. These findings demonstrate the promise of bias-aware, generative AI frameworks for financial forecasting and highlight opportunities for modular multi-agent design to advance practical, transparent, and reproducible approaches in quantitative finance. 

---
# CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics 

**Authors**: Vaibhav Devraj, Dhruv Kumar, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2512.21877)  

**Abstract**: Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks. 

---
# Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content? 

**Authors**: Naen Xu, Jinghuai Zhang, Changjiang Li, Hengyu An, Chunyi Zhou, Jun Wang, Boyu Xu, Yuyuan Li, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2512.21871)  

**Abstract**: Large vision-language models (LVLMs) have achieved remarkable advancements in multimodal reasoning tasks. However, their widespread accessibility raises critical concerns about potential copyright infringement. Will LVLMs accurately recognize and comply with copyright regulations when encountering copyrighted content (i.e., user input, retrieved documents) in the context? Failure to comply with copyright regulations may lead to serious legal and ethical consequences, particularly when LVLMs generate responses based on copyrighted materials (e.g., retrieved book experts, news reports). In this paper, we present a comprehensive evaluation of various LVLMs, examining how they handle copyrighted content -- such as book excerpts, news articles, music lyrics, and code documentation when they are presented as visual inputs. To systematically measure copyright compliance, we introduce a large-scale benchmark dataset comprising 50,000 multimodal query-content pairs designed to evaluate how effectively LVLMs handle queries that could lead to copyright infringement. Given that real-world copyrighted content may or may not include a copyright notice, the dataset includes query-content pairs in two distinct scenarios: with and without a copyright notice. For the former, we extensively cover four types of copyright notices to account for different cases. Our evaluation reveals that even state-of-the-art closed-source LVLMs exhibit significant deficiencies in recognizing and respecting the copyrighted content, even when presented with the copyright notice. To solve this limitation, we introduce a novel tool-augmented defense framework for copyright compliance, which reduces infringement risks in all scenarios. Our findings underscore the importance of developing copyright-aware LVLMs to ensure the responsible and lawful use of copyrighted content. 

---
# Secure and Explainable Fraud Detection in Finance via Hierarchical Multi-source Dataset Distillation 

**Authors**: Yiming Qian, Thorsten Neumann, Xueyining Huang, David Hardoon, Fei Gao, Yong Liu, Siow Mong Rick Goh  

**Link**: [PDF](https://arxiv.org/pdf/2512.21866)  

**Abstract**: We propose an explainable, privacy-preserving dataset distillation framework for collaborative financial fraud detection. A trained random forest is converted into transparent, axis-aligned rule regions (leaf hyperrectangles), and synthetic transactions are generated by uniformly sampling within each region. This produces a compact, auditable surrogate dataset that preserves local feature interactions without exposing sensitive original records. The rule regions also support explainability: aggregated rule statistics (for example, support and lift) describe global patterns, while assigning each case to its generating region gives concise human-readable rationales and calibrated uncertainty based on tree-vote disagreement.
On the IEEE-CIS fraud dataset (590k transactions across three institution-like clusters), distilled datasets reduce data volume by 85% to 93% (often under 15% of the original) while maintaining competitive precision and micro-F1, with only a modest AUC drop. Sharing and augmenting with synthesized data across institutions improves cross-cluster precision, recall, and AUC. Real vs. synthesized structure remains highly similar (over 93% by nearest-neighbor cosine analysis). Membership-inference attacks perform at chance level (about 0.50) when distinguishing training from hold-out records, suggesting low memorization risk. Removing high-uncertainty synthetic points using disagreement scores further boosts AUC (up to 0.687) and improves calibration. Sensitivity tests show weak dependence on the distillation ratio (AUC about 0.641 to 0.645 from 6% to 60%).
Overall, tree-region distillation enables trustworthy, deployable fraud analytics with interpretable global rules, per-case rationales with quantified uncertainty, and strong privacy properties suitable for multi-institution settings and regulatory audit. 

---
# Balancing Accuracy and Efficiency: CNN Fusion Models for Diabetic Retinopathy Screening 

**Authors**: Md Rafid Islam, Rafsan Jany, Akib Ahmed, Mohammad Ashrafuzzaman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21861)  

**Abstract**: Diabetic retinopathy (DR) remains a leading cause of preventable blindness, yet large-scale screening is constrained by limited specialist availability and variable image quality across devices and populations. This work investigates whether feature-level fusion of complementary convolutional neural network (CNN) backbones can deliver accurate and efficient binary DR screening on globally sourced fundus images. Using 11,156 images pooled from five public datasets (APTOS, EyePACS, IDRiD, Messidor, and ODIR), we frame DR detection as a binary classification task and compare three pretrained models (ResNet50, EfficientNet-B0, and DenseNet121) against pairwise and tri-fusion variants. Across five independent runs, fusion consistently outperforms single backbones. The EfficientNet-B0 + DenseNet121 (Eff+Den) fusion model achieves the best overall mean performance (accuracy: 82.89\%) with balanced class-wise F1-scores for normal (83.60\%) and diabetic (82.60\%) cases. While the tri-fusion is competitive, it incurs a substantially higher computational cost. Inference profiling highlights a practical trade-off: EfficientNet-B0 is the fastest (approximately 1.16 ms/image at batch size 1000), whereas the Eff+Den fusion offers a favorable accuracy--latency balance. These findings indicate that lightweight feature fusion can enhance generalization across heterogeneous datasets, supporting scalable binary DR screening workflows where both accuracy and throughput are critical. 

---
# MoonBot: Modular and On-Demand Reconfigurable Robot Toward Moon Base Construction 

**Authors**: Kentaro Uno, Elian Neppel, Gustavo H. Diaz, Ashutosh Mishra, Shamistan Karimov, A. Sejal Jain, Ayesha Habib, Pascal Pama, Hazal Gozbasi, Shreya Santra, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2512.21853)  

**Abstract**: The allure of lunar surface exploration and development has recently captured widespread global attention. Robots have proved to be indispensable for exploring uncharted terrains, uncovering and leveraging local resources, and facilitating the construction of future human habitats. In this article, we introduce the modular and on-demand reconfigurable robot (MoonBot), a modular and reconfigurable robotic system engineered to maximize functionality while operating within the stringent mass constraints of lunar payloads and adapting to varying environmental conditions and task requirements. This article details the design and development of MoonBot and presents a preliminary field demonstration that validates the proof of concept through the execution of milestone tasks simulating the establishment of lunar infrastructure. These tasks include essential civil engineering operations, infrastructural component transportation and deployment, and assistive operations with inflatable modules. Furthermore, we systematically summarize the lessons learned during testing, focusing on the connector design and providing valuable insights for the advancement of modular robotic systems in future lunar missions. 

---
# A Comedy of Estimators: On KL Regularization in RL Training of LLMs 

**Authors**: Vedant Shah, Johan Obando-Ceron, Vineet Jain, Brian Bartoldson, Bhavya Kailkhura, Sarthak Mittal, Glen Berseth, Pablo Samuel Castro, Yoshua Bengio, Nikolay Malkin, Moksh Jain, Siddarth Venkatraman, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2512.21852)  

**Abstract**: The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse Kullback-Leibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Recent works show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators configurations, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning \texttt{Qwen2.5-7B}, \texttt{Llama-3.1-8B-Instruct} and \texttt{Qwen3-4B-Instruct-2507} with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that, in on-policy settings: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. We also investigate the performance resulting from different KL configurations in off-policy settings and observe that KL regularization can help stabilize off-policy RL training resulting from asynchronous setups. 

---
# HeartBench: Probing Core Dimensions of Anthropomorphic Intelligence in LLMs 

**Authors**: Jiaxin Liu, Peiyi Tu, Wenyu Chen, Yihong Zhuang, Xinxia Ling, Anji Zhou, Chenxi Wang, Zhuo Han, Zhengkai Yang, Junbo Zhao, Zenan Huang, Yuanyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21849)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in cognitive and reasoning benchmarks, they exhibit a persistent deficit in anthropomorphic intelligence-the capacity to navigate complex social, emotional, and ethical nuances. This gap is particularly acute in the Chinese linguistic and cultural context, where a lack of specialized evaluation frameworks and high-quality socio-emotional data impedes progress. To address these limitations, we present HeartBench, a framework designed to evaluate the integrated emotional, cultural, and ethical dimensions of Chinese LLMs. Grounded in authentic psychological counseling scenarios and developed in collaboration with clinical experts, the benchmark is structured around a theory-driven taxonomy comprising five primary dimensions and 15 secondary capabilities. We implement a case-specific, rubric-based methodology that translates abstract human-like traits into granular, measurable criteria through a ``reasoning-before-scoring'' evaluation protocol. Our assessment of 13 state-of-the-art LLMs indicates a substantial performance ceiling: even leading models achieve only 60% of the expert-defined ideal score. Furthermore, analysis using a difficulty-stratified ``Hard Set'' reveals a significant performance decay in scenarios involving subtle emotional subtexts and complex ethical trade-offs. HeartBench establishes a standardized metric for anthropomorphic AI evaluation and provides a methodological blueprint for constructing high-quality, human-aligned training data. 

---
# S&P 500 Stock's Movement Prediction using CNN 

**Authors**: Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.21804)  

**Abstract**: This paper is about predicting the movement of stock consist of S&P 500 index. Historically there are many approaches have been tried using various methods to predict the stock movement and being used in the market currently for algorithm trading and alpha generating systems using traditional mathematical approaches [1, 2].
The success of artificial neural network recently created a lot of interest and paved the way to enable prediction using cutting-edge research in the machine learning and deep learning. Some of these papers have done a great job in implementing and explaining benefits of these new technologies. Although most these papers do not go into the complexity of the financial data and mostly utilize single dimension data, still most of these papers were successful in creating the ground for future research in this comparatively new phenomenon. In this paper, I am trying to use multivariate raw data including stock split/dividend events (as-is) present in real-world market data instead of engineered financial data. Convolution Neural Network (CNN), the best-known tool so far for image classification, is used on the multi-dimensional stock numbers taken from the market mimicking them as a vector of historical data matrices (read images) and the model achieves promising results. The predictions can be made stock by stock, i.e., a single stock, sector-wise or for the portfolio of stocks. 

---
# CellMamba: Adaptive Mamba for Accurate and Efficient Cell Detection 

**Authors**: Ruochen Liu, Yi Tian, Jiahao Wang, Hongbin Liu, Xianxu Hou, Jingxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21803)  

**Abstract**: Cell detection in pathological images presents unique challenges due to densely packed objects, subtle inter-class differences, and severe background clutter. In this paper, we propose CellMamba, a lightweight and accurate one-stage detector tailored for fine-grained biomedical instance detection. Built upon a VSSD backbone, CellMamba integrates CellMamba Blocks, which couple either NC-Mamba or Multi-Head Self-Attention (MSA) with a novel Triple-Mapping Adaptive Coupling (TMAC) module. TMAC enhances spatial discriminability by splitting channels into two parallel branches, equipped with dual idiosyncratic and one consensus attention map, adaptively fused to preserve local sensitivity and global consistency. Furthermore, we design an Adaptive Mamba Head that fuses multi-scale features via learnable weights for robust detection under varying object sizes. Extensive experiments on two public datasets-CoNSeP and CytoDArk0-demonstrate that CellMamba outperforms both CNN-based, Transformer-based, and Mamba-based baselines in accuracy, while significantly reducing model size and inference latency. Our results validate CellMamba as an efficient and effective solution for high-resolution cell detection. 

---
# Applications of synthetic financial data in portfolio and risk modeling 

**Authors**: Christophe D. Hounwanou, Yae Ulrich Gaba  

**Link**: [PDF](https://arxiv.org/pdf/2512.21798)  

**Abstract**: Synthetic financial data offers a practical way to address the privacy and accessibility challenges that limit research in quantitative finance. This paper examines the use of generative models, in particular TimeGAN and Variational Autoencoders (VAEs), for creating synthetic return series that support portfolio construction, trading analysis, and risk modeling. Using historical daily returns from the S and P 500 as a benchmark, we generate synthetic datasets under comparable market conditions and evaluate them using statistical similarity metrics, temporal structure tests, and downstream financial tasks. The study shows that TimeGAN produces synthetic data with distributional shapes, volatility patterns, and autocorrelation behaviour that are close to those observed in real returns. When applied to mean-variance portfolio optimization, the resulting synthetic datasets lead to portfolio weights, Sharpe ratios, and risk levels that remain close to those obtained from real data. The VAE provides more stable training but tends to smooth extreme market movements, which affects risk estimation. Finally, the analysis supports the use of synthetic datasets as substitutes for real financial data in portfolio analysis and risk simulation, particularly when models are able to capture temporal dynamics. Synthetic data therefore provides a privacy-preserving, cost-effective, and reproducible tool for financial experimentation and model development. 

---
# Multi-agent Adaptive Mechanism Design 

**Authors**: Qiushi Han, David Simchi-Levi, Renfei Tan, Zishuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.21794)  

**Abstract**: We study a sequential mechanism design problem in which a principal seeks to elicit truthful reports from multiple rational agents while starting with no prior knowledge of agents' beliefs. We introduce Distributionally Robust Adaptive Mechanism (DRAM), a general framework combining insights from both mechanism design and online learning to jointly address truthfulness and cost-optimality. Throughout the sequential game, the mechanism estimates agents' beliefs and iteratively updates a distributionally robust linear program with shrinking ambiguity sets to reduce payments while preserving truthfulness. Our mechanism guarantees truthful reporting with high probability while achieving $\tilde{O}(\sqrt{T})$ cumulative regret, and we establish a matching lower bound showing that no truthful adaptive mechanism can asymptotically do better. The framework generalizes to plug-in estimators, supporting structured priors and delayed feedback. To our knowledge, this is the first adaptive mechanism under general settings that maintains truthfulness and achieves optimal regret when incentive constraints are unknown and must be learned. 

---
# Five Years of SciCap: What We Learned and Future Directions for Scientific Figure Captioning 

**Authors**: Ting-Hao K.Huang, Ryan A. Rossi, Sungchul Kim, Tong Yu, Ting-Yao E. Hsu, Ho Yin, C. Lee Giles  

**Link**: [PDF](https://arxiv.org/pdf/2512.21789)  

**Abstract**: Between 2021 and 2025, the SciCap project grew from a small seed-funded idea at The Pennsylvania State University (Penn State) into one of the central efforts shaping the scientific figure-captioning landscape. Supported by a Penn State seed grant, Adobe, and the Alfred P. Sloan Foundation, what began as our attempt to test whether domain-specific training, which was successful in text models like SciBERT, could also work for figure captions expanded into a multi-institution collaboration. Over these five years, we curated, released, and continually updated a large collection of figure-caption pairs from arXiv papers, conducted extensive automatic and human evaluations on both generated and author-written captions, navigated the rapid rise of large language models (LLMs), launched annual challenges, and built interactive systems that help scientists write better captions. In this piece, we look back at the first five years of SciCap and summarize the key technical and methodological lessons we learned. We then outline five major unsolved challenges and propose directions for the next phase of research in scientific figure captioning. 

---
# InstructMoLE: Instruction-Guided Mixture of Low-rank Experts for Multi-Conditional Image Generation 

**Authors**: Jinqi Xiao, Qing Yan, Liming Jiang, Zichuan Liu, Hao Kang, Shen Sang, Tiancheng Zhi, Jing Liu, Cheng Yang, Xin Lu, Bo Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21788)  

**Abstract**: Parameter-Efficient Fine-Tuning of Diffusion Transformers (DiTs) for diverse, multi-conditional tasks often suffers from task interference when using monolithic adapters like LoRA. The Mixture of Low-rank Experts (MoLE) architecture offers a modular solution, but its potential is usually limited by routing policies that operate at a token level. Such local routing can conflict with the global nature of user instructions, leading to artifacts like spatial fragmentation and semantic drift in complex image generation tasks. To address these limitations, we introduce InstructMoLE, a novel framework that employs an Instruction-Guided Mixture of Low-Rank Experts. Instead of per-token routing, InstructMoLE utilizes a global routing signal, Instruction-Guided Routing (IGR), derived from the user's comprehensive instruction. This ensures that a single, coherently chosen expert council is applied uniformly across all input tokens, preserving the global semantics and structural integrity of the generation process. To complement this, we introduce an output-space orthogonality loss, which promotes expert functional diversity and mitigates representational collapse. Extensive experiments demonstrate that InstructMoLE significantly outperforms existing LoRA adapters and MoLE variants across challenging multi-conditional generation benchmarks. Our work presents a robust and generalizable framework for instruction-driven fine-tuning of generative models, enabling superior compositional control and fidelity to user intent. 

---
# Inference-based GAN Video Generation 

**Authors**: Jingbo Yang, Adrian G. Bors  

**Link**: [PDF](https://arxiv.org/pdf/2512.21776)  

**Abstract**: Video generation has seen remarkable progresses thanks to advancements in generative deep learning. Generated videos should not only display coherent and continuous movement but also meaningful movement in successions of scenes. Generating models such as Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs) and more recently Diffusion Networks have been used for generating short video sequences, usually of up to 16 frames. In this paper, we first propose a new type of video generator by enabling adversarial-based unconditional video generators with a variational encoder, akin to a VAE-GAN hybrid structure, in order to enable the generation process with inference capabilities. The proposed model, as in other video deep learning-based processing frameworks, incorporates two processing branches, one for content and another for movement. However, existing models struggle with the temporal scaling of the generated videos. In classical approaches when aiming to increase the generated video length, the resulting video quality degrades, particularly when considering generating significantly long sequences. To overcome this limitation, our research study extends the initially proposed VAE-GAN video generation model by employing a novel, memory-efficient approach to generate long videos composed of hundreds or thousands of frames ensuring their temporal continuity, consistency and dynamics. Our approach leverages a Markov chain framework with a recall mechanism, with each state representing a VAE-GAN short-length video generator. This setup allows for the sequential connection of generated video sub-sequences, enabling temporal dependencies, resulting in meaningful long video sequences. 

---
# A-QCF-Net: An Adaptive Quaternion Cross-Fusion Network for Multimodal Liver Tumor Segmentation from Unpaired Datasets 

**Authors**: Arunkumar V, Firos V M, Senthilkumar S, Gangadharan G R  

**Link**: [PDF](https://arxiv.org/pdf/2512.21760)  

**Abstract**: Multimodal medical imaging provides complementary information that is crucial for accurate delineation of pathology, but the development of deep learning models is limited by the scarcity of large datasets in which different modalities are paired and spatially aligned. This paper addresses this fundamental limitation by proposing an Adaptive Quaternion Cross-Fusion Network (A-QCF-Net) that learns a single unified segmentation model from completely separate and unpaired CT and MRI cohorts. The architecture exploits the parameter efficiency and expressive power of Quaternion Neural Networks to construct a shared feature space. At its core is the Adaptive Quaternion Cross-Fusion (A-QCF) block, a data driven attention module that enables bidirectional knowledge transfer between the two streams. By learning to modulate the flow of information dynamically, the A-QCF block allows the network to exchange abstract modality specific expertise, such as the sharp anatomical boundary information available in CT and the subtle soft tissue contrast provided by MRI. This mutual exchange regularizes and enriches the feature representations of both streams. We validate the framework by jointly training a single model on the unpaired LiTS (CT) and ATLAS (MRI) datasets. The jointly trained model achieves Tumor Dice scores of 76.7% on CT and 78.3% on MRI, significantly exceeding the strong unimodal nnU-Net baseline by margins of 5.4% and 4.7% respectively. Furthermore, comprehensive explainability analysis using Grad-CAM and Grad-CAM++ confirms that the model correctly focuses on relevant pathological structures, ensuring the learned representations are clinically meaningful. This provides a robust and clinically viable paradigm for unlocking the large unpaired imaging archives that are common in healthcare. 

---
# How Do Agents Perform Code Optimization? An Empirical Study 

**Authors**: Huiyun Peng, Antonio Zhong, Ricardo Andrés Calvo Méndez, Kelechi G. Kalu, James C. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2512.21757)  

**Abstract**: Performance optimization is a critical yet challenging aspect of software development, often requiring a deep understanding of system behavior, algorithmic tradeoffs, and careful code modifications. Although recent advances in AI coding agents have accelerated code generation and bug fixing, little is known about how these agents perform on real-world performance optimization tasks. We present the first empirical study comparing agent- and human-authored performance optimization commits, analyzing 324 agent-generated and 83 human-authored PRs from the AIDev dataset across adoption, maintainability, optimization patterns, and validation practices. We find that AI-authored performance PRs are less likely to include explicit performance validation than human-authored PRs (45.7\% vs. 63.6\%, $p=0.007$). In addition, AI-authored PRs largely use the same optimization patterns as humans. We further discuss limitations and opportunities for advancing agentic code optimization. 

---
# A Model of Causal Explanation on Neural Networks for Tabular Data 

**Authors**: Takashi Isozaki, Masahiro Yamamoto, Atsushi Noda  

**Link**: [PDF](https://arxiv.org/pdf/2512.21746)  

**Abstract**: The problem of explaining the results produced by machine learning methods continues to attract attention. Neural network (NN) models, along with gradient boosting machines, are expected to be utilized even in tabular data with high prediction accuracy. This study addresses the related issues of pseudo-correlation, causality, and combinatorial reasons for tabular data in NN predictors. We propose a causal explanation method, CENNET, and a new explanation power index using entropy for the method. CENNET provides causal explanations for predictions by NNs and uses structural causal models (SCMs) effectively combined with the NNs although SCMs are usually not used as predictive models on their own in terms of predictive accuracy. We show that CEN-NET provides such explanations through comparative experiments with existing methods on both synthetic and quasi-real data in classification tasks. 

---
# HELP: Hierarchical Embodied Language Planner for Household Tasks 

**Authors**: Alexandr V. Korchemnyi, Anatoly O. Onishchenko, Eva A. Bakaeva, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2512.21723)  

**Abstract**: Embodied agents tasked with complex scenarios, whether in real or simulated environments, rely heavily on robust planning capabilities. When instructions are formulated in natural language, large language models (LLMs) equipped with extensive linguistic knowledge can play this role. However, to effectively exploit the ability of such models to handle linguistic ambiguity, to retrieve information from the environment, and to be based on the available skills of an agent, an appropriate architecture must be designed. We propose a Hierarchical Embodied Language Planner, called HELP, consisting of a set of LLM-based agents, each dedicated to solving a different subtask. We evaluate the proposed approach on a household task and perform real-world experiments with an embodied agent. We also focus on the use of open source LLMs with a relatively small number of parameters, to enable autonomous deployment. 

---
# An Information Theoretic Perspective on Agentic System Design 

**Authors**: Shizhe He, Avanika Narayan, Ishan S. Khare, Scott W. Linderman, Christopher Ré, Dan Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2512.21720)  

**Abstract**: Agentic language model (LM) systems power modern applications like "Deep Research" and "Claude Code," and leverage multi-LM architectures to overcome context limitations. Beneath their apparent diversity lies a recurring pattern: smaller "compressor" LMs (that can even run locally) distill raw context into compact text that is then consumed by larger "predictor" LMs. Despite their popularity, the design of compressor-predictor systems remains largely ad hoc, with little guidance on how compressor and predictor choices shape downstream performance. In practice, attributing gains to compression versus prediction requires costly, task-specific pairwise sweeps. We argue that these agentic system design questions are, at root, information-theoretic. Viewing the compressor LM as a noisy channel, we introduce a simple estimator of mutual information between the context and its compression to quantify compression quality in a task-independent way. We show that mutual information strongly predicts downstream performance, independent of any specific task. Through an information-theoretic framework, we perform a comprehensive empirical analysis across five datasets and three model families. Results reveal that larger compressors not only are more accurate, but also more token-efficient, conveying more bits of information per token. A 7B Qwen-2.5 compressor, for instance, is $1.6\times$ more accurate, $4.6\times$ more concise, and conveys $5.5\times$ more bits of mutual information per token than its 1.5B sibling. Across datasets, scaling compressors is substantially more effective than scaling predictors, enabling larger on-device compressors to pair with smaller cloud predictors. Applied to a Deep Research system, these principles enable local compressors as small as 3B parameters to recover $99\%$ of frontier-LM accuracy at $26\%$ of API costs. 

---
# Multiconnectivity for SAGIN: Current Trends, Challenges, AI-driven Solutions, and Opportunities 

**Authors**: Abd Ullah Khan, Adnan Shahid, Haejoon Jung, Hyundong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2512.21717)  

**Abstract**: Space-air-ground-integrated network (SAGIN)-enabled multiconnectivity (MC) is emerging as a key enabler for next-generation networks, enabling users to simultaneously utilize multiple links across multi-layer non-terrestrial networks (NTN) and multi-radio access technology (multi-RAT) terrestrial networks (TN). However, the heterogeneity of TN and NTN introduces complex architectural challenges that complicate MC implementation. Specifically, the diversity of link types, spanning air-to-air, air-to-space, space-to-space, space-to-ground, and ground-to-ground communications, renders optimal resource allocation highly complex. Recent advancements in reinforcement learning (RL) and agentic artificial intelligence (AI) have shown remarkable effectiveness in optimal decision-making in complex and dynamic environments. In this paper, we review the current developments in SAGIN-enabled MC and outline the key challenges associated with its implementation. We further highlight the transformative potential of AI-driven approaches for resource optimization in a heterogeneous SAGIN environment. To this end, we present a case study on resource allocation optimization enabled by agentic RL for SAGIN-enabled MC involving diverse radio access technologies (RATs). Results show that learning-based methods can effectively handle complex scenarios and substantially enhance network performance in terms of latency and capacity while incurring a moderate increase in power consumption as an acceptable tradeoff. Finally, open research problems and future directions are presented to realize efficient SAGIN-enabled MC. 

---
# CATCH: A Controllable Theme Detection Framework with Contextualized Clustering and Hierarchical Generation 

**Authors**: Rui Ke, Jiahui Xu, Shenghao Yang, Kuang Wang, Feng Jiang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.21715)  

**Abstract**: Theme detection is a fundamental task in user-centric dialogue systems, aiming to identify the latent topic of each utterance without relying on predefined schemas. Unlike intent induction, which operates within fixed label spaces, theme detection requires cross-dialogue consistency and alignment with personalized user preferences, posing significant challenges. Existing methods often struggle with sparse, short utterances for accurate topic representation and fail to capture user-level thematic preferences across dialogues. To address these challenges, we propose CATCH (Controllable Theme Detection with Contextualized Clustering and Hierarchical Generation), a unified framework that integrates three core components: (1) context-aware topic representation, which enriches utterance-level semantics using surrounding topic segments; (2) preference-guided topic clustering, which jointly models semantic proximity and personalized feedback to align themes across dialogue; and (3) a hierarchical theme generation mechanism designed to suppress noise and produce robust, coherent topic labels. Experiments on a multi-domain customer dialogue benchmark (DSTC-12) demonstrate the effectiveness of CATCH with 8B LLM in both theme clustering and topic generation quality. 

---
# Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought 

**Authors**: Yuyi Zhang, Boyu Tang, Tianjie Ju, Sufeng Duan, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21711)  

**Abstract**: Latent tokens are gaining attention for enhancing reasoning in large language models (LLMs), yet their internal mechanisms remain unclear. This paper examines the problem from a reliability perspective, uncovering fundamental weaknesses: latent tokens function as uninterpretable placeholders rather than encoding faithful reasoning. While resistant to perturbation, they promote shortcut usage over genuine reasoning. We focus on Chain-of-Continuous-Thought (COCONUT), which claims better efficiency and stability than explicit Chain-of-Thought (CoT) while maintaining performance. We investigate this through two complementary approaches. First, steering experiments perturb specific token subsets, namely COCONUT and explicit CoT. Unlike CoT tokens, COCONUT tokens show minimal sensitivity to steering and lack reasoning-critical information. Second, shortcut experiments evaluate models under biased and out-of-distribution settings. Results on MMLU and HotpotQA demonstrate that COCONUT consistently exploits dataset artifacts, inflating benchmark performance without true reasoning. These findings reposition COCONUT as a pseudo-reasoning mechanism: it generates plausible traces that conceal shortcut dependence rather than faithfully representing reasoning processes. 

---
# Detecting AI-Generated Paraphrases in Bengali: A Comparative Study of Zero-Shot and Fine-Tuned Transformers 

**Authors**: Md. Rakibul Islam, Most. Sharmin Sultana Samu, Md. Zahid Hossain, Farhad Uz Zaman, Md. Kamrozzaman Bhuiyan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21709)  

**Abstract**: Large language models (LLMs) can produce text that closely resembles human writing. This capability raises concerns about misuse, including disinformation and content manipulation. Detecting AI-generated text is essential to maintain authenticity and prevent malicious applications. Existing research has addressed detection in multiple languages, but the Bengali language remains largely unexplored. Bengali's rich vocabulary and complex structure make distinguishing human-written and AI-generated text particularly challenging. This study investigates five transformer-based models: XLMRoBERTa-Large, mDeBERTaV3-Base, BanglaBERT-Base, IndicBERT-Base and MultilingualBERT-Base. Zero-shot evaluation shows that all models perform near chance levels (around 50% accuracy) and highlight the need for task-specific fine-tuning. Fine-tuning significantly improves performance, with XLM-RoBERTa, mDeBERTa and MultilingualBERT achieving around 91% on both accuracy and F1-score. IndicBERT demonstrates comparatively weaker performance, indicating limited effectiveness in fine-tuning for this task. This work advances AI-generated text detection in Bengali and establishes a foundation for building robust systems to counter AI-generated content. 

---
# Enabling Conversational Behavior Reasoning Capabilities in Full-Duplex Speech 

**Authors**: Shuchang Pan, Siddharth Banerjee, Dhruv Hebbar, Siddhant Patel, Akshaj Gupta, Kan Jen Cheng, Hanjo Kim, Zeyi Austin Li, Martin Q. Ma, Tingle Li, Gopala Anumanchipalli, Jiachen Lian  

**Link**: [PDF](https://arxiv.org/pdf/2512.21706)  

**Abstract**: Human conversation is organized by an implicit chain of thoughts that manifests as timed speech acts. Capturing this causal pathway is key to building natural full-duplex interactive systems. We introduce a framework that enables reasoning over conversational behaviors by modeling this process as causal inference within a Graph-of-Thoughts (GoT). Our approach formalizes the intent-to-action pathway with a hierarchical labeling scheme, predicting high-level communicative intents and low-level speech acts to learn their causal and temporal dependencies. To train this system, we develop a hybrid corpus that pairs controllable, event-rich simulations with human-annotated rationales and real conversational speech. The GoT framework structures streaming predictions as an evolving graph, enabling a multimodal transformer to forecast the next speech act, generate concise justifications for its decisions, and dynamically refine its reasoning. Experiments on both synthetic and real duplex dialogues show that the framework delivers robust behavior detection, produces interpretable reasoning chains, and establishes a foundation for benchmarking conversational reasoning in full duplex spoken dialogue systems. 

---
# Zero-Shot to Zero-Lies: Detecting Bengali Deepfake Audio through Transfer Learning 

**Authors**: Most. Sharmin Sultana Samu, Md. Rakibul Islam, Md. Zahid Hossain, Md. Kamrozzaman Bhuiyan, Farhad Uz Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2512.21702)  

**Abstract**: The rapid growth of speech synthesis and voice conversion systems has made deepfake audio a major security concern. Bengali deepfake detection remains largely unexplored. In this work, we study automatic detection of Bengali audio deepfakes using the BanglaFake dataset. We evaluate zeroshot inference with several pretrained models. These include Wav2Vec2-XLSR-53, Whisper, PANNsCNN14, WavLM and Audio Spectrogram Transformer. Zero-shot results show limited detection ability. The best model, Wav2Vec2-XLSR-53, achieves 53.80% accuracy, 56.60% AUC and 46.20% EER. We then f ine-tune multiple architectures for Bengali deepfake detection. These include Wav2Vec2-Base, LCNN, LCNN-Attention, ResNet18, ViT-B16 and CNN-BiLSTM. Fine-tuned models show strong performance gains. ResNet18 achieves the highest accuracy of 79.17%, F1 score of 79.12%, AUC of 84.37% and EER of 24.35%. Experimental results confirm that fine-tuning significantly improves performance over zero-shot inference. This study provides the first systematic benchmark of Bengali deepfake audio detection. It highlights the effectiveness of f ine-tuned deep learning models for this low-resource language. 

---
# BeHGAN: Bengali Handwritten Word Generation from Plain Text Using Generative Adversarial Networks 

**Authors**: Md. Rakibul Islam, Md. Kamrozzaman Bhuiyan, Safwan Muntasir, Arifur Rahman Jawad, Most. Sharmin Sultana Samu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21694)  

**Abstract**: Handwritten Text Recognition (HTR) is a well-established research area. In contrast, Handwritten Text Generation (HTG) is an emerging field with significant potential. This task is challenging due to the variation in individual handwriting styles. A large and diverse dataset is required to generate realistic handwritten text. However, such datasets are difficult to collect and are not readily available. Bengali is the fifth most spoken language in the world. While several studies exist for languages such as English and Arabic, Bengali handwritten text generation has received little attention. To address this gap, we propose a method for generating Bengali handwritten words. We developed and used a self-collected dataset of Bengali handwriting samples. The dataset includes contributions from approximately five hundred individuals across different ages and genders. All images were pre-processed to ensure consistency and quality. Our approach demonstrates the ability to produce diverse handwritten outputs from input plain text. We believe this work contributes to the advancement of Bengali handwriting generation and can support further research in this area. 

---
# RIPCN: A Road Impedance Principal Component Network for Probabilistic Traffic Flow Forecasting 

**Authors**: Haochen Lv, Yan Lin, Shengnan Guo, Xiaowei Mao, Hong Nie, Letian Gong, Youfang Lin, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21685)  

**Abstract**: Accurate traffic flow forecasting is crucial for intelligent transportation services such as navigation and ride-hailing. In such applications, uncertainty estimation in forecasting is important because it helps evaluate traffic risk levels, assess forecast reliability, and provide timely warnings. As a result, probabilistic traffic flow forecasting (PTFF) has gained significant attention, as it produces both point forecasts and uncertainty estimates. However, existing PTFF approaches still face two key challenges: (1) how to uncover and model the causes of traffic flow uncertainty for reliable forecasting, and (2) how to capture the spatiotemporal correlations of uncertainty for accurate prediction.
To address these challenges, we propose RIPCN, a Road Impedance Principal Component Network that integrates domain-specific transportation theory with spatiotemporal principal component learning for PTFF. RIPCN introduces a dynamic impedance evolution network that captures directional traffic transfer patterns driven by road congestion level and flow variability, revealing the direct causes of uncertainty and enhancing both reliability and interpretability. In addition, a principal component network is designed to forecast the dominant eigenvectors of future flow covariance, enabling the model to capture spatiotemporal uncertainty correlations. This design allows for accurate and efficient uncertainty estimation while also improving point prediction performance. Experimental results on real-world datasets show that our approach outperforms existing probabilistic forecasting methods. 

---
# Comparative Analysis of Deep Learning Models for Perception in Autonomous Vehicles 

**Authors**: Jalal Khan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21673)  

**Abstract**: Recently, a plethora of machine learning (ML) and deep learning (DL) algorithms have been proposed to achieve the efficiency, safety, and reliability of autonomous vehicles (AVs). The AVs use a perception system to detect, localize, and identify other vehicles, pedestrians, and road signs to perform safe navigation and decision-making. In this paper, we compare the performance of DL models, including YOLO-NAS and YOLOv8, for a detection-based perception task. We capture a custom dataset and experiment with both DL models using our custom dataset. Our analysis reveals that the YOLOv8s model saves 75% of training time compared to the YOLO-NAS model. In addition, the YOLOv8s model (83%) outperforms the YOLO-NAS model (81%) when the target is to achieve the highest object detection accuracy. These comparative analyses of these new emerging DL models will allow the relevant research community to understand the models' performance under real-world use case scenarios. 

---
# Near-Optimal Coalition Structures in Polynomial Time 

**Authors**: Angshul Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2512.21657)  

**Abstract**: We study the classical coalition structure generation (CSG) problem and compare the anytime behavior of three algorithmic paradigms: dynamic programming (DP), MILP branch-and-bound, and sparse relaxations based on greedy or $l_1$-type methods. Under a simple random "sparse synergy" model for coalition values, we prove that sparse relaxations recover coalition structures whose welfare is arbitrarily close to optimal in polynomial time with high probability. In contrast, broad classes of DP and MILP algorithms require exponential time before attaining comparable solution quality. This establishes a rigorous probabilistic anytime separation in favor of sparse relaxations, even though exact methods remain ultimately optimal. 

---
# Structural Induced Exploration for Balanced and Scalable Multi-Robot Path Planning 

**Authors**: Zikun Guo, Adeyinka P. Adedigba, Rammohan Mallipeddi, Heoncheol Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.21654)  

**Abstract**: Multi-robot path planning is a fundamental yet challenging problem due to its combinatorial complexity and the need to balance global efficiency with fair task allocation among robots. Traditional swarm intelligence methods, although effective on small instances, often converge prematurely and struggle to scale to complex environments. In this work, we present a structure-induced exploration framework that integrates structural priors into the search process of the ant colony optimization (ACO). The approach leverages the spatial distribution of the task to induce a structural prior at initialization, thereby constraining the search space. The pheromone update rule is then designed to emphasize structurally meaningful connections and incorporates a load-aware objective to reconcile the total travel distance with individual robot workload. An explicit overlap suppression strategy further ensures that tasks remain distinct and balanced across the team. The proposed framework was validated on diverse benchmark scenarios covering a wide range of instance sizes and robot team configurations. The results demonstrate consistent improvements in route compactness, stability, and workload distribution compared to representative metaheuristic baselines. Beyond performance gains, the method also provides a scalable and interpretable framework that can be readily applied to logistics, surveillance, and search-and-rescue applications where reliable large-scale coordination is essential. 

---
# Enabling Ultra-Fast Cardiovascular Imaging Across Heterogeneous Clinical Environments with a Generalist Foundation Model and Multimodal Database 

**Authors**: Zi Wang, Mingkai Huang, Zhang Shi, Hongjie Hu, Lan Lan, Hui Zhang, Yan Li, Xi Hu, Qing Lu, Zongming Zhu, Qiong Yao, Yuxiang Dai, Fanwen Wang, Yinzhe Wu, Jun Lyu, Qianqian Gao, Guangming Xu, Zhenxuan Zhang, Haosen Zhang, Qing Li, Guangming Wang, Tianxing He, Lizhen Lan, Siyue Li, Le Xue, Mengting Sun, Yuntong Lyu, Junpu Hu, Jiayu Zhu, Rizwan Ahmad, Zhengyu Bu, Xianling Qian, Guanke Cai, Ruiyu Cao, Weirui Cai, Chang Xu, Yuyang Ren, Feidan Yu, Siying Ma, Ziqiang Xu, Xinran Chen, Sha Hua, Daniel Kim, Yajing Zhang, Chen Ouyang, Wenjia Bai, Jing Qin, Yucheng Yang, Daniel Rueckert, He Wang, Qian Tao, Claudia Prieto, Michael Markl, Alistair Young, Lianming Wu, Shuo Wang, Chen Qin, Mengsu Zeng, Xihong Hu, Haibo Xu, Xiaobo Qu, Hao Li, Guang Yang, Chengyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21652)  

**Abstract**: Multimodal cardiovascular magnetic resonance (CMR) imaging provides comprehensive and non-invasive insights into cardiovascular disease (CVD) diagnosis and underlying mechanisms. Despite decades of advancements, its widespread clinical adoption remains constrained by prolonged scan times and heterogeneity across medical environments. This underscores the urgent need for a generalist reconstruction foundation model for ultra-fast CMR imaging, one capable of adapting across diverse imaging scenarios and serving as the essential substrate for all downstream analyses. To enable this goal, we curate MMCMR-427K, the largest and most comprehensive multimodal CMR k-space database to date, comprising 427,465 multi-coil k-space data paired with structured metadata across 13 international centers, 12 CMR modalities, 15 scanners, and 17 CVD categories in populations across three continents. Building on this unprecedented resource, we introduce CardioMM, a generalist reconstruction foundation model capable of dynamically adapting to heterogeneous fast CMR imaging scenarios. CardioMM unifies semantic contextual understanding with physics-informed data consistency to deliver robust reconstructions across varied scanners, protocols, and patient presentations. Comprehensive evaluations demonstrate that CardioMM achieves state-of-the-art performance in the internal centers and exhibits strong zero-shot generalization to unseen external settings. Even at imaging acceleration up to 24x, CardioMM reliably preserves key cardiac phenotypes, quantitative myocardial biomarkers, and diagnostic image quality, enabling a substantial increase in CMR examination throughput without compromising clinical integrity. Together, our open-access MMCMR-427K database and CardioMM framework establish a scalable pathway toward high-throughput, high-quality, and clinically accessible cardiovascular imaging. 

---
# Variance-Aware Prior-Based Tree Policies for Monte Carlo Tree Search 

**Authors**: Maximilian Weichart  

**Link**: [PDF](https://arxiv.org/pdf/2512.21648)  

**Abstract**: Monte Carlo Tree Search (MCTS) has profoundly influenced reinforcement learning (RL) by integrating planning and learning in tasks requiring long-horizon reasoning, exemplified by the AlphaZero family of algorithms. Central to MCTS is the search strategy, governed by a tree policy based on an upper confidence bound (UCB) applied to trees (UCT). A key factor in the success of AlphaZero is the introduction of a prior term in the UCB1-based tree policy PUCT, which improves exploration efficiency and thus accelerates training. While many alternative UCBs with stronger theoretical guarantees than UCB1 exist, extending them to prior-based UCTs has been challenging, since PUCT was derived empirically rather than from first principles. Recent work retrospectively justified PUCT by framing MCTS as a regularized policy optimization (RPO) problem. Building on this perspective, we introduce Inverse-RPO, a general methodology that systematically derives prior-based UCTs from any prior-free UCB. Applying this method to the variance-aware UCB-V, we obtain two new prior-based tree policies that incorporate variance estimates into the search. Experiments indicate that these variance-aware prior-based UCTs outperform PUCT across multiple benchmarks without incurring additional computational cost. We also provide an extension of the mctx library supporting variance-aware UCTs, showing that the required code changes are minimal and intended to facilitate further research on principled prior-based UCTs. Code: this http URL. 

---
# TrackTeller: Temporal Multimodal 3D Grounding for Behavior-Dependent Object References 

**Authors**: Jiahong Yu, Ziqi Wang, Hailiang Zhao, Wei Zhai, Xueqiang Yan, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2512.21641)  

**Abstract**: Understanding natural-language references to objects in dynamic 3D driving scenes is essential for interactive autonomous systems. In practice, many referring expressions describe targets through recent motion or short-term interactions, which cannot be resolved from static appearance or geometry alone. We study temporal language-based 3D grounding, where the objective is to identify the referred object in the current frame by leveraging multi-frame observations. We propose TrackTeller, a temporal multimodal grounding framework that integrates LiDAR-image fusion, language-conditioned decoding, and temporal reasoning in a unified architecture. TrackTeller constructs a shared UniScene representation aligned with textual semantics, generates language-aware 3D proposals, and refines grounding decisions using motion history and short-term dynamics. Experiments on the NuPrompt benchmark demonstrate that TrackTeller consistently improves language-grounded tracking performance, outperforming strong baselines with a 70% relative improvement in Average Multi-Object Tracking Accuracy and a 3.15-3.4 times reduction in False Alarm Frequency. 

---
# LLM-I2I: Boost Your Small Item2Item Recommendation Model with Large Language Model 

**Authors**: Yinfu Feng, Yanjing Wu, Rong Xiao, Xiaoyi Zen  

**Link**: [PDF](https://arxiv.org/pdf/2512.21595)  

**Abstract**: Item-to-Item (I2I) recommendation models are widely used in real-world systems due to their scalability, real-time capabilities, and high recommendation quality. Research to enhance I2I performance focuses on two directions: 1) model-centric approaches, which adopt deeper architectures but risk increased computational costs and deployment complexity, and 2) data-centric methods, which refine training data without altering models, offering cost-effectiveness but struggling with data sparsity and noise. To address these challenges, we propose LLM-I2I, a data-centric framework leveraging Large Language Models (LLMs) to mitigate data quality issues. LLM-I2I includes (1) an LLM-based generator that synthesizes user-item interactions for long-tail items, alleviating data sparsity, and (2) an LLM-based discriminator that filters noisy interactions from real and synthetic data. The refined data is then fused to train I2I models. Evaluated on industry (AEDS) and academic (ARD) datasets, LLM-I2I consistently improves recommendation accuracy, particularly for long-tail items. Deployed on a large-scale cross-border e-commerce platform, it boosts recall number (RN) by 6.02% and gross merchandise value (GMV) by 1.22% over existing I2I models. This work highlights the potential of LLMs in enhancing data-centric recommendation systems without modifying model architectures. 

---
# Residual Prior Diffusion: A Probabilistic Framework Integrating Coarse Latent Priors with Diffusion Models 

**Authors**: Takuro Kutsuna  

**Link**: [PDF](https://arxiv.org/pdf/2512.21593)  

**Abstract**: Diffusion models have become a central tool in deep generative modeling, but standard formulations rely on a single network and a single diffusion schedule to transform a simple prior, typically a standard normal distribution, into the target data distribution. As a result, the model must simultaneously represent the global structure of the distribution and its fine-scale local variations, which becomes difficult when these scales are strongly mismatched. This issue arises both in natural images, where coarse manifold-level structure and fine textures coexist, and in low-dimensional distributions with highly concentrated local structure. To address this issue, we propose Residual Prior Diffusion (RPD), a two-stage framework in which a coarse prior model first captures the large-scale structure of the data distribution, and a diffusion model is then trained to represent the residual between the prior and the target data distribution. We formulate RPD as an explicit probabilistic model with a tractable evidence lower bound, whose optimization reduces to the familiar objectives of noise prediction or velocity prediction. We further introduce auxiliary variables that leverage information from the prior model and theoretically analyze how they reduce the difficulty of the prediction problem in RPD. Experiments on synthetic datasets with fine-grained local structure show that standard diffusion models fail to capture local details, whereas RPD accurately captures fine-scale detail while preserving the large-scale structure of the distribution. On natural image generation tasks, RPD achieved generation quality that matched or exceeded that of representative diffusion-based baselines and it maintained strong performance even with a small number of inference steps. 

---
# A Unified Definition of Hallucination, Or: It's the World Model, Stupid 

**Authors**: Emmy Liu, Varun Gangal, Chelsea Zou, Xiaoqi Huang, Michael Yu, Alex Chang, Zhuofu Tao, Sachin Kumar, Steven Y. Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.21577)  

**Abstract**: Despite numerous attempts to solve the issue of hallucination since the inception of neural language models, it remains a problem in even frontier large language models today. Why is this the case? We walk through definitions of hallucination used in the literature from a historical perspective up to the current day, and fold them into a single definition of hallucination, wherein different prior definitions focus on different aspects of our definition. At its core, we argue that hallucination is simply inaccurate (internal) world modeling, in a form where it is observable to the user (e.g., stating a fact which contradicts a knowledge base, or producing a summary which contradicts a known source). By varying the reference world model as well as the knowledge conflict policy (e.g., knowledge base vs. in-context), we arrive at the different existing definitions of hallucination present in the literature.
We argue that this unified view is useful because it forces evaluations to make clear their assumed "world" or source of truth, clarifies what should and should not be called hallucination (as opposed to planning or reward/incentive-related errors), and provides a common language to compare benchmarks and mitigation techniques. Building on this definition, we outline plans for a family of benchmarks in which hallucinations are defined as mismatches with synthetic but fully specified world models in different environments, and sketch out how these benchmarks can use such settings to stress-test and improve the world modeling components of language models. 

---
# Towards Long-window Anchoring in Vision-Language Model Distillation 

**Authors**: Haoyi Zhou, Shuo Li, Tianyu Chen, Qi Song, Chonghan Gao, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.21576)  

**Abstract**: While large vision-language models (VLMs) demonstrate strong long-context understanding, their prevalent small branches fail on linguistics-photography alignment for a limited window size. We discover that knowledge distillation improves students' capability as a complement to Rotary Position Embeddings (RoPE) on window sizes (anchored from large models). Building on this insight, we propose LAid, which directly aims at the transfer of long-range attention mechanisms through two complementary components: (1) a progressive distance-weighted attention matching that dynamically emphasizes longer position differences during training, and (2) a learnable RoPE response gain modulation that selectively amplifies position sensitivity where needed. Extensive experiments across multiple model families demonstrate that LAid-distilled models achieve up to 3.2 times longer effective context windows compared to baseline small models, while maintaining or improving performance on standard VL benchmarks. Spectral analysis also suggests that LAid successfully preserves crucial low-frequency attention components that conventional methods fail to transfer. Our work not only provides practical techniques for building more efficient long-context VLMs but also offers theoretical insights into how positional understanding emerges and transfers during distillation. 

---
# Exploration of Reproducible Generated Image Detection 

**Authors**: Yihang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21562)  

**Abstract**: While the technology for detecting AI-Generated Content (AIGC) images has advanced rapidly, the field still faces two core issues: poor reproducibility and insufficient gen eralizability, which hinder the practical application of such technologies. This study addresses these challenges by re viewing 7 key papers on AIGC detection, constructing a lightweight test dataset, and reproducing a representative detection method. Through this process, we identify the root causes of the reproducibility dilemma in the field: firstly, papers often omit implicit details such as prepro cessing steps and parameter settings; secondly, most detec tion methods overfit to exclusive features of specific gener ators rather than learning universal intrinsic features of AIGC images. Experimental results show that basic perfor mance can be reproduced when strictly following the core procedures described in the original papers. However, de tection performance drops sharply when preprocessing dis rupts key features or when testing across different genera tors. This research provides empirical evidence for improv ing the reproducibility of AIGC detection technologies and offers reference directions for researchers to disclose ex perimental details more comprehensively and verify the generalizability of their proposed methods. 

---
# Bidirectional Human-AI Alignment in Education for Trustworthy Learning Environments 

**Authors**: Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.21552)  

**Abstract**: Artificial intelligence (AI) is transforming education, offering unprecedented opportunities to personalize learning, enhance assessment, and support educators. Yet these opportunities also introduce risks related to equity, privacy, and student autonomy. This chapter develops the concept of bidirectional human-AI alignment in education, emphasizing that trustworthy learning environments arise not only from embedding human values into AI systems but also from equipping teachers, students, and institutions with the skills to interpret, critique, and guide these technologies. Drawing on emerging research and practical case examples, we explore AI's evolution from support tool to collaborative partner, highlighting its impacts on teacher roles, student agency, and institutional governance. We propose actionable strategies for policymakers, developers, and educators to ensure that AI advances equity, transparency, and human flourishing rather than eroding them. By reframing AI adoption as an ongoing process of mutual adaptation, the chapter envisions a future in which humans and intelligent systems learn, innovate, and grow together. 

---
# Human-AI Interaction Alignment: Designing, Evaluating, and Evolving Value-Centered AI For Reciprocal Human-AI Futures 

**Authors**: Hua Shen, Tiffany Knearem, Divy Thakkar, Pat Pataranutaporn, Anoop Sinha, Yike, Jenny T. Liang, Lama Ahmad, Tanu Mitra, Brad A. Myers, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.21551)  

**Abstract**: The rapid integration of generative AI into everyday life underscores the need to move beyond unidirectional alignment models that only adapt AI to human values. This workshop focuses on bidirectional human-AI alignment, a dynamic, reciprocal process where humans and AI co-adapt through interaction, evaluation, and value-centered design. Building on our past CHI 2025 BiAlign SIG and ICLR 2025 Workshop, this workshop will bring together interdisciplinary researchers from HCI, AI, social sciences and more domains to advance value-centered AI and reciprocal human-AI collaboration. We focus on embedding human and societal values into alignment research, emphasizing not only steering AI toward human values but also enabling humans to critically engage with and evolve alongside AI systems. Through talks, interdisciplinary discussions, and collaborative activities, participants will explore methods for interactive alignment, frameworks for societal impact evaluation, and strategies for alignment in dynamic contexts. This workshop aims to bridge the disciplines' gaps and establish a shared agenda for responsible, reciprocal human-AI futures. 

---
# Hierarchy-Aware Fine-Tuning of Vision-Language Models 

**Authors**: Jiayu Li, Rajesh Gangireddy, Samet Akcay, Wei Cheng, Juhua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21529)  

**Abstract**: Vision-Language Models (VLMs) learn powerful multimodal representations through large-scale image-text pretraining, but adapting them to hierarchical classification is underexplored. Standard approaches treat labels as flat categories and require full fine-tuning, which is expensive and produces inconsistent predictions across taxonomy levels. We propose an efficient hierarchy-aware fine-tuning framework that updates a few parameters while enforcing structural consistency. We combine two objectives: Tree-Path KL Divergence (TP-KL) aligns predictions along the ground-truth label path for vertical coherence, while Hierarchy-Sibling Smoothed Cross-Entropy (HiSCE) encourages consistent predictions among sibling classes. Both losses work in the VLM's shared embedding space and integrate with lightweight LoRA adaptation. Experiments across multiple benchmarks show consistent improvements in Full-Path Accuracy and Tree-based Inconsistency Error with minimal parameter overhead. Our approach provides an efficient strategy for adapting VLMs to structured taxonomies. 

---
# Selective LLM-Guided Regularization for Enhancing Recommendation Models 

**Authors**: Shanglin Yang, Zhan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.21526)  

**Abstract**: Large language models provide rich semantic priors and strong reasoning capabilities, making them promising auxiliary signals for recommendation. However, prevailing approaches either deploy LLMs as standalone recommender or apply global knowledge distillation, both of which suffer from inherent drawbacks. Standalone LLM recommender are costly, biased, and unreliable across large regions of the user item space, while global distillation forces the downstream model to imitate LLM predictions even when such guidance is inaccurate. Meanwhile, recent studies show that LLMs excel particularly in re-ranking and challenging scenarios, rather than uniformly across all this http URL introduce Selective LLM Guided Regularization, a model-agnostic and computation efficient framework that activates LLM based pairwise ranking supervision only when a trainable gating mechanism informing by user history length, item popularity, and model uncertainty predicts the LLM to be reliable. All LLM scoring is performed offline, transferring knowledge without increasing inference cost. Experiments across multiple datasets show that this selective strategy consistently improves overall accuracy and yields substantial gains in cold start and long tail regimes, outperforming global distillation baselines. 

---
# DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO 

**Authors**: Henglin Liu, Huijuan Huang, Jing Wang, Chang Liu, Xiu Li, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2512.21514)  

**Abstract**: Reinforcement learning (RL), particularly GRPO, improves image generation quality significantly by comparing the relative performance of images generated within the same group. However, in the later stages of training, the model tends to produce homogenized outputs, lacking creativity and visual diversity, which restricts its application scenarios. This issue can be analyzed from both reward modeling and generation dynamics perspectives. First, traditional GRPO relies on single-sample quality as the reward signal, driving the model to converge toward a few high-reward generation modes while neglecting distribution-level diversity. Second, conventional GRPO regularization neglects the dominant role of early-stage denoising in preserving diversity, causing a misaligned regularization budget that limits the achievable quality--diversity trade-off. Motivated by these insights, we revisit the diversity degradation problem from both reward modeling and generation dynamics. At the reward level, we propose a distributional creativity bonus based on semantic grouping. Specifically, we construct a distribution-level representation via spectral clustering over samples generated from the same caption, and adaptively allocate exploratory rewards according to group sizes to encourage the discovery of novel visual modes. At the generation level, we introduce a structure-aware regularization, which enforces stronger early-stage constraints to preserve diversity without compromising reward optimization efficiency. Experiments demonstrate that our method achieves a 13\%--18\% improvement in semantic diversity under matched quality scores, establishing a new Pareto frontier between image quality and diversity for GRPO-based image generation. 

---
# MotionTeller: Multi-modal Integration of Wearable Time-Series with LLMs for Health and Behavioral Understanding 

**Authors**: Aiwei Zhang, Arvind Pillai, Andrew Campbell, Nicholas C. Jacobson  

**Link**: [PDF](https://arxiv.org/pdf/2512.21506)  

**Abstract**: As wearable sensing becomes increasingly pervasive, a key challenge remains: how can we generate natural language summaries from raw physiological signals such as actigraphy - minute-level movement data collected via accelerometers? In this work, we introduce MotionTeller, a generative framework that natively integrates minute-level wearable activity data with large language models (LLMs). MotionTeller combines a pretrained actigraphy encoder with a lightweight projection module that maps behavioral embeddings into the token space of a frozen decoder-only LLM, enabling free-text, autoregressive generation of daily behavioral summaries. We construct a novel dataset of 54383 (actigraphy, text) pairs derived from real-world NHANES recordings, and train the model using cross-entropy loss with supervision only on the language tokens. MotionTeller achieves high semantic fidelity (BERTScore-F1 = 0.924) and lexical accuracy (ROUGE-1 = 0.722), outperforming prompt-based baselines by 7 percent in ROUGE-1. The average training loss converges to 0.38 by epoch 15, indicating stable optimization. Qualitative analysis confirms that MotionTeller captures circadian structure and behavioral transitions, while PCA plots reveal enhanced cluster alignment in embedding space post-training. Together, these results position MotionTeller as a scalable, interpretable system for transforming wearable sensor data into fluent, human-centered descriptions, introducing new pathways for behavioral monitoring, clinical review, and personalized health interventions. 

---
# Oogiri-Master: Benchmarking Humor Understanding via Oogiri 

**Authors**: Soichiro Murakami, Hidetaka Kamigaito, Hiroya Takamura, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2512.21494)  

**Abstract**: Humor is a salient testbed for human-like creative thinking in large language models (LLMs). We study humor using the Japanese creative response game Oogiri, in which participants produce witty responses to a given prompt, and ask the following research question: What makes such responses funny to humans? Previous work has offered only limited reliable means to answer this question. Existing datasets contain few candidate responses per prompt, expose popularity signals during ratings, and lack objective and comparable metrics for funniness. Thus, we introduce Oogiri-Master and Oogiri-Corpus, which are a benchmark and dataset designed to enable rigorous evaluation of humor understanding in LLMs. Each prompt is paired with approximately 100 diverse candidate responses, and funniness is rated independently by approximately 100 human judges without access to others' ratings, reducing popularity bias and enabling robust aggregation. Using Oogiri-Corpus, we conduct a quantitative analysis of the linguistic factors associated with funniness, such as text length, ambiguity, and incongruity resolution, and derive objective metrics for predicting human judgments. Subsequently, we benchmark a range of LLMs and human baselines in Oogiri-Master, demonstrating that state-of-the-art models approach human performance and that insight-augmented prompting improves the model performance. Our results provide a principled basis for evaluating and advancing humor understanding in LLMs. 

---
# Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism 

**Authors**: Xinglin Pan, Shaohuai Shi, Wenxiang Lin, Yuxin Wang, Zhenheng Tang, Wei Wang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21487)  

**Abstract**: The mixture-of-experts (MoE) architecture scales model size with sublinear computational increase but suffers from memory-intensive inference due to KV caches and sparse expert activation. Recent disaggregated expert parallelism (DEP) distributes attention and experts to dedicated GPU groups but lacks support for shared experts and efficient task scheduling, limiting performance.
We propose FinDEP, a fine-grained task scheduling algorithm for DEP that maximizes task overlap to improve MoE inference throughput. FinDEP introduces three innovations: 1) partitioning computation/communication into smaller tasks for fine-grained pipelining, 2) formulating a scheduling optimization supporting variable granularity and ordering, and 3) developing an efficient solver for this large search space.
Experiments on four GPU systems with DeepSeek-V2 and Qwen3-MoE show FinDEP improves throughput by up to 1.61x over prior methods, achieving up to 1.24x speedup on a 32-GPU system. 

---
# GPF-Net: Gated Progressive Fusion Learning for Polyp Re-Identification 

**Authors**: Suncheng Xiang, Xiaoyang Wang, Junjie Jiang, Hejia Wang, Dahong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2512.21476)  

**Abstract**: Colonoscopic Polyp Re-Identification aims to match the same polyp from a large gallery with images from different views taken using different cameras, which plays an important role in the prevention and treatment of colorectal cancer in computer-aided diagnosis. However, the coarse resolution of high-level features of a specific polyp often leads to inferior results for small objects where detailed information is important. To address this challenge, we propose a novel architecture, named Gated Progressive Fusion network, to selectively fuse features from multiple levels using gates in a fully connected way for polyp ReID. On the basis of it, a gated progressive fusion strategy is introduced to achieve layer-wise refinement of semantic information through multi-level feature interactions. Experiments on standard benchmarks show the benefits of the multimodal setting over state-of-the-art unimodal ReID models, especially when combined with the specialized multimodal fusion strategy. 

---
# Intelligent recognition of GPR road hidden defect images based on feature fusion and attention mechanism 

**Authors**: Haotian Lv, Yuhui Zhang, Jiangbo Dai, Hanli Wu, Jiaji Wang, Dawei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21452)  

**Abstract**: Ground Penetrating Radar (GPR) has emerged as a pivotal tool for non-destructive evaluation of subsurface road defects. However, conventional GPR image interpretation remains heavily reliant on subjective expertise, introducing inefficiencies and inaccuracies. This study introduces a comprehensive framework to address these limitations: (1) A DCGAN-based data augmentation strategy synthesizes high-fidelity GPR images to mitigate data scarcity while preserving defect morphology under complex backgrounds; (2) A novel Multi-modal Chain and Global Attention Network (MCGA-Net) is proposed, integrating Multi-modal Chain Feature Fusion (MCFF) for hierarchical multi-scale defect representation and Global Attention Mechanism (GAM) for context-aware feature enhancement; (3) MS COCO transfer learning fine-tunes the backbone network, accelerating convergence and improving generalization. Ablation and comparison experiments validate the framework's efficacy. MCGA-Net achieves Precision (92.8%), Recall (92.5%), and mAP@50 (95.9%). In the detection of Gaussian noise, weak signals and small targets, MCGA-Net maintains robustness and outperforms other models. This work establishes a new paradigm for automated GPR-based defect detection, balancing computational efficiency with high accuracy in complex subsurface environments. 

---
# dUltra: Ultra-Fast Diffusion Language Models via Reinforcement Learning 

**Authors**: Shirui Chen, Jiantao Jiao, Lillian J. Ratliff, Banghua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21446)  

**Abstract**: Masked diffusion language models (MDLMs) offer the potential for parallel token generation, but most open-source MDLMs decode fewer than 5 tokens per model forward pass even with sophisticated sampling strategies. As a result, their sampling speeds are often comparable to AR + speculative decoding schemes, limiting their advantage over mainstream autoregressive approaches. Existing distillation-based accelerators (dParallel, d3LLM) finetune MDLMs on trajectories generated by a base model, which can become off-policy during finetuning and restrict performance to the quality of the base model's samples. We propose \texttt{dUltra}, an on-policy reinforcement learning framework based on Group Relative Policy Optimization (GRPO) that learns unmasking strategies for efficient parallel decoding. dUltra introduces an unmasking planner head that predicts per-token unmasking likelihoods under independent Bernoulli distributions. We jointly optimize the base diffusion LLM and the unmasking order planner using reward signals combining verifiable reward, distillation reward, and the number of unmasking steps. Across mathematical reasoning and code generation tasks, dUltra improves the accuracy--efficiency trade-off over state-of-the-art heuristic and distillation baselines, moving towards achieving ``diffusion supremacy'' over autoregressive models. 

---
# Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models 

**Authors**: Geoffroy Morlat, Marceau Nahon, Augustin Chartouny, Raja Chatila, Ismael T. Freire, Mehdi Khamassi  

**Link**: [PDF](https://arxiv.org/pdf/2512.21439)  

**Abstract**: Moral actions are judged not only by their outcomes but by the context in which they occur. We present COMETH (Contextual Organization of Moral Evaluation from Textual Human inputs), a framework that integrates a probabilistic context learner with LLM-based semantic abstraction and human moral evaluations to model how context shapes the acceptability of ambiguous actions. We curate an empirically grounded dataset of 300 scenarios across six core actions (violating Do not kill, Do not deceive, and Do not break the law) and collect ternary judgments (Blame/Neutral/Support) from N=101 participants. A preprocessing pipeline standardizes actions via an LLM filter and MiniLM embeddings with K-means, producing robust, reproducible core-action clusters. COMETH then learns action-specific moral contexts by clustering scenarios online from human judgment distributions using principled divergence criteria. To generalize and explain predictions, a Generalization module extracts concise, non-evaluative binary contextual features and learns feature weights in a transparent likelihood-based model. Empirically, COMETH roughly doubles alignment with majority human judgments relative to end-to-end LLM prompting (approx. 60% vs. approx. 30% on average), while revealing which contextual features drive its predictions. The contributions are: (i) an empirically grounded moral-context dataset, (ii) a reproducible pipeline combining human judgments with model-based context learning and LLM semantics, and (iii) an interpretable alternative to end-to-end LLMs for context-sensitive moral prediction and explanation. 

---
# Teaching People LLM's Errors and Getting it Right 

**Authors**: Nathan Stringham, Fateme Hashemi Chaleshtori, Xinyuan Yan, Zhichao Xu, Bei Wang, Ana Marasović  

**Link**: [PDF](https://arxiv.org/pdf/2512.21422)  

**Abstract**: People use large language models (LLMs) when they should not. This is partly because they see LLMs compose poems and answer intricate questions, so they understandably, but incorrectly, assume LLMs won't stumble on basic tasks like simple arithmetic. Prior work has tried to address this by clustering instance embeddings into regions where an LLM is likely to fail and automatically describing patterns in these regions. The found failure patterns are taught to users to mitigate their overreliance. Yet, this approach has not fully succeeded. In this analysis paper, we aim to understand why.
We first examine whether the negative result stems from the absence of failure patterns. We group instances in two datasets by their meta-labels and evaluate an LLM's predictions on these groups. We then define criteria to flag groups that are sizable and where the LLM is error-prone, and find meta-label groups that meet these criteria. Their meta-labels are the LLM's failure patterns that could be taught to users, so they do exist. We next test whether prompting and embedding-based approaches can surface these known failures. Without this, users cannot be taught about them to reduce their overreliance. We find mixed results across methods, which could explain the negative result. Finally, we revisit the final metric that measures teaching effectiveness. We propose to assess a user's ability to effectively use the given failure patterns to anticipate when an LLM is error-prone. A user study shows a positive effect from teaching with this metric, unlike the human-AI team accuracy. Our findings show that teaching failure patterns could be a viable approach to mitigating overreliance, but success depends on better automated failure-discovery methods and using metrics like ours. 

---
# LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors 

**Authors**: Tianwei Lan, Farid Naït-Abdesselam  

**Link**: [PDF](https://arxiv.org/pdf/2512.21404)  

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks. 

---
# Safe Path Planning and Observation Quality Enhancement Strategy for Unmanned Aerial Vehicles in Water Quality Monitoring Tasks 

**Authors**: Yuanshuang Fu, Qianyao Wang, Qihao Wang, Bonan Zhang, Jiaxin Zhao, Yiming Cao, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.21375)  

**Abstract**: Unmanned Aerial Vehicle (UAV) spectral remote sensing technology is widely used in water quality monitoring. However, in dynamic environments, varying illumination conditions, such as shadows and specular reflection (sun glint), can cause severe spectral distortion, thereby reducing data availability. To maximize the acquisition of high-quality data while ensuring flight safety, this paper proposes an active path planning method for dynamic light and shadow disturbance avoidance. First, a dynamic prediction model is constructed to transform the time-varying light and shadow disturbance areas into three-dimensional virtual obstacles. Second, an improved Interfered Fluid Dynamical System (IFDS) algorithm is introduced, which generates a smooth initial obstacle avoidance path by building a repulsive force field. Subsequently, a Model Predictive Control (MPC) framework is employed for rolling-horizon path optimization to handle flight dynamics constraints and achieve real-time trajectory tracking. Furthermore, a Dynamic Flight Altitude Adjustment (DFAA) mechanism is designed to actively reduce the flight altitude when the observable area is narrow, thereby enhancing spatial resolution. Simulation results show that, compared with traditional PID and single obstacle avoidance algorithms, the proposed method achieves an obstacle avoidance success rate of 98% in densely disturbed scenarios, significantly improves path smoothness, and increases the volume of effective observation data by approximately 27%. This research provides an effective engineering solution for precise UAV water quality monitoring in complex illumination environments. 

---
# AInsteinBench: Benchmarking Coding Agents on Scientific Repositories 

**Authors**: Titouan Duston, Shuo Xin, Yang Sun, Daoguang Zan, Aoyan Li, Shulin Xin, Kai Shen, Yixiao Chen, Qiming Sun, Ge Zhang, Jiashuo Liu, Huan Zhou, Jingkai Liu, Zhichen Pu, Yuanheng Wang, Bo-Xuan Ge, Xin Tong, Fei Ye, Zhi-Chao Zhao, Wen-Biao Han, Zhoujian Cao, Yueran Zhao, Weiluo Ren, Qingshen Long, Yuxiao Liu, Anni Huang, Yidi Du, Yuanyuan Rong, Jiahao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.21373)  

**Abstract**: We introduce AInsteinBench, a large-scale benchmark for evaluating whether large language model (LLM) agents can operate as scientific computing development agents within real research software ecosystems. Unlike existing scientific reasoning benchmarks which focus on conceptual knowledge, or software engineering benchmarks that emphasize generic feature implementation and issue resolving, AInsteinBench evaluates models in end-to-end scientific development settings grounded in production-grade scientific repositories. The benchmark consists of tasks derived from maintainer-authored pull requests across six widely used scientific codebases, spanning quantum chemistry, quantum computing, molecular dynamics, numerical relativity, fluid dynamics, and cheminformatics. All benchmark tasks are carefully curated through multi-stage filtering and expert review to ensure scientific challenge, adequate test coverage, and well-calibrated difficulty. By leveraging evaluation in executable environments, scientifically meaningful failure modes, and test-driven verification, AInsteinBench measures a model's ability to move beyond surface-level code generation toward the core competencies required for computational scientific research. 

---
# Reflection-Driven Control for Trustworthy Code Agents 

**Authors**: Bin Wang, Jiazheng Quan, Xingrui Yu, Hansen Hu, Yuhao, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21354)  

**Abstract**: Contemporary large language model (LLM) agents are remarkably capable, but they still lack reliable safety controls and can produce unconstrained, unpredictable, and even actively harmful outputs. To address this, we introduce Reflection-Driven Control, a standardized and pluggable control module that can be seamlessly integrated into general agent architectures. Reflection-Driven Control elevates "self-reflection" from a post hoc patch into an explicit step in the agent's own reasoning process: during generation, the agent continuously runs an internal reflection loop that monitors and evaluates its own decision path. When potential risks are detected, the system retrieves relevant repair examples and secure coding guidelines from an evolving reflective memory, injecting these evidence-based constraints directly into subsequent reasoning steps. We instantiate Reflection-Driven Control in the setting of secure code generation and systematically evaluate it across eight classes of security-critical programming tasks. Empirical results show that Reflection-Driven Control substantially improves the security and policy compliance of generated code while largely preserving functional correctness, with minimal runtime and token overhead. Taken together, these findings indicate that Reflection-Driven Control is a practical path toward trustworthy AI coding agents: it enables designs that are simultaneously autonomous, safer by construction, and auditable. 

---
# Multi-Agent LLM Committees for Autonomous Software Beta Testing 

**Authors**: Sumanth Bharadwaj Hachalli Karanam, Dhiwahar Adhithya Kennady  

**Link**: [PDF](https://arxiv.org/pdf/2512.21352)  

**Abstract**: Manual software beta testing is costly and time-consuming, while single-agent large language model (LLM) approaches suffer from hallucinations and inconsistent behavior. We propose a multi-agent committee framework in which diverse vision-enabled LLMs collaborate through a three-round voting protocol to reach consensus on testing actions. The framework combines model diversity, persona-driven behavioral variation, and visual user interface understanding to systematically explore web applications. Across 84 experimental runs with 9 testing personas and 4 scenarios, multi-agent committees achieve an 89.5 percent overall task success rate. Configurations with 2 to 4 agents reach 91.7 to 100 percent success, compared to 78.0 percent for single-agent baselines, yielding improvements of 13.7 to 22.0 percentage points. At the action level, the system attains a 93.1 percent success rate with a median per-action latency of 0.71 seconds, enabling real-time and continuous integration testing. Vision-enabled agents successfully identify user interface elements, with navigation and reporting achieving 100 percent success and form filling achieving 99.2 percent success. We evaluate the framework on WebShop and OWASP benchmarks, achieving 74.7 percent success on WebShop compared to a 50.1 percent published GPT-3 baseline, and 82.0 percent success on OWASP Juice Shop security testing with coverage of 8 of the 10 OWASP Top 10 vulnerability categories. Across 20 injected regressions, the committee achieves an F1 score of 0.91 for bug detection, compared to 0.78 for single-agent baselines. The open-source implementation enables reproducible research and practical deployment of LLM-based software testing in CI/CD pipelines. 

---
# CosmoCore-Evo: Evolutionary Dream-Replay Reinforcement Learning for Adaptive Code Generation 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2512.21351)  

**Abstract**: Building on the affective dream-replay reinforcement learning framework of CosmoCore, we introduce CosmoCore-Evo, an extension that incorporates evolutionary algorithms to enhance adaptability and novelty in code generation tasks. Inspired by anthropological aspects of human evolution, such as natural selection and adaptation in early hominids, CosmoCore-Evo treats RL trajectories as ``genomes'' that undergo mutation and selection during the nocturnal replay phase. This mechanism allows agents to break free from trained patterns, fostering emergent behaviors and improved performance in distribution-shifted environments, such as changing APIs or novel libraries. We augment the Dream Queue with evolutionary operations, including mutation of high-fitness trajectories and enterprise-tuned fitness functions that incorporate efficiency, compliance, and scalability metrics. Evaluated on extended benchmarks including HumanEval variants with shifts, BigCodeBench, and a custom PySpark pipeline simulation, CosmoCore-Evo achieves up to 35% higher novelty in solutions and 25% faster adaptation compared to the original CosmoCore and baselines like PPO and REAMER. Ablations confirm the role of evolutionary components in bridging the sentient gap for LLM agents. Code for replication, including a toy simulation, is provided. 

---
# Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software 

**Authors**: Ying Xiao, Shangwen Wang, Sicen Liu, Dingyuan Xue, Xian Zhan, Yepang Liu, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21348)  

**Abstract**: Traditional software fairness research typically emphasizes ethical and social imperatives, neglecting that fairness fundamentally represents a core software quality issue arising directly from performance disparities across sensitive user groups. Recognizing fairness explicitly as a software quality dimension yields practical benefits beyond ethical considerations, notably improved predictive performance for unprivileged groups, enhanced out-of-distribution generalization, and increased geographic transferability in real-world deployments. Nevertheless, existing bias mitigation methods face a critical dilemma: while pre-processing methods offer broad applicability across model types, they generally fall short in effectiveness compared to post-processing techniques. To overcome this challenge, we propose Correlation Tuning (CoT), a novel pre-processing approach designed to mitigate bias by adjusting data correlations. Specifically, CoT introduces the Phi-coefficient, an intuitive correlation measure, to systematically quantify correlation between sensitive attributes and labels, and employs multi-objective optimization to address the proxy biases. Extensive evaluations demonstrate that CoT increases the true positive rate of unprivileged groups by an average of 17.5% and reduces three key bias metrics, including statistical parity difference (SPD), average odds difference (AOD), and equal opportunity difference (EOD), by more than 50% on average. CoT outperforms state-of-the-art methods by three and ten percentage points in single attribute and multiple attributes scenarios, respectively. We will publicly release our experimental results and source code to facilitate future research. 

---
# Query Carefully: Detecting the Unanswerables in Text-to-SQL Tasks 

**Authors**: Jasmin Saxer, Isabella Maria Aigner, Luise Linzmeier, Andreas Weiler, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2512.21345)  

**Abstract**: Text-to-SQL systems allow non-SQL experts to interact with relational databases using natural language. However, their tendency to generate executable SQL for ambiguous, out-of-scope, or unanswerable queries introduces a hidden risk, as outputs may be misinterpreted as correct. This risk is especially serious in biomedical contexts, where precision is critical. We therefore present Query Carefully, a pipeline that integrates LLM-based SQL generation with explicit detection and handling of unanswerable inputs. Building on the OncoMX component of ScienceBenchmark, we construct OncoMX-NAQ (No-Answer Questions), a set of 80 no-answer questions spanning 8 categories (non-SQL, out-of-schema/domain, and multiple ambiguity types). Our approach employs llama3.3:70b with schema-aware prompts, explicit No-Answer Rules (NAR), and few-shot examples drawn from both answerable and unanswerable questions. We evaluate SQL exact match, result accuracy, and unanswerable-detection accuracy. On the OncoMX dev split, few-shot prompting with answerable examples increases result accuracy, and adding unanswerable examples does not degrade performance. On OncoMX-NAQ, balanced prompting achieves the highest unanswerable-detection accuracy (0.8), with near-perfect results for structurally defined categories (non-SQL, missing columns, out-of-domain) but persistent challenges for missing-value queries (0.5) and column ambiguity (0.3). A lightweight user interface surfaces interim SQL, execution results, and abstentions, supporting transparent and reliable text-to-SQL in biomedical applications. 

---
# Atomistic Simulation Guided Convolutional Neural Networks for Thermal Modeling of Friction Stir Welding 

**Authors**: Akshansh Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2512.21344)  

**Abstract**: Accurate prediction of temperature evolution is essential for understanding thermomechanical behavior in friction stir welding. In this study, molecular dynamics simulations were performed using LAMMPS to model aluminum friction stir welding at the atomic scale, capturing material flow, plastic deformation, and heat generation during tool plunge, traverse, and retraction. Atomic positions and velocities were extracted from simulation trajectories and transformed into physics based two dimensional spatial grids. These grids represent local height variation, velocity components, velocity magnitude, and atomic density, preserving spatial correlations within the weld zone. A two-dimensional convolutional neural network was developed to predict temperature directly from the spatially resolved atomistic data. Hyperparameter optimization was carried out to determine an appropriate network configuration. The trained model demonstrates strong predictive capability, achieving a coefficient of determination R square of 0.9439, a root mean square error of 14.94 K, and a mean absolute error of 11.58 K on unseen test data. Class Activation Map analysis indicates that the model assigns higher importance to regions near the tool material interface, which are associated with intense deformation and heat generation in the molecular dynamics simulations. The results show that spatial learning from atomistic simulation data can accurately reproduce temperature trends in friction stir welding while remaining consistent with physical deformation and flow mechanisms observed at the atomic scale. 

---
# EcoNet: Multiagent Planning and Control Of Household Energy Resources Using Active Inference 

**Authors**: John C. Boik, Kobus Esterhuysen, Jacqueline B. Hynes, Axel Constant, Ines Hipolito, Mahault Albarracin, Alex B. Kiefer, Karl Friston  

**Link**: [PDF](https://arxiv.org/pdf/2512.21343)  

**Abstract**: Advances in automated systems afford new opportunities for intelligent management of energy at household, local area, and utility scales. Home Energy Management Systems (HEMS) can play a role by optimizing the schedule and use of household energy devices and resources. One challenge is that the goals of a household can be complex and conflicting. For example, a household might wish to reduce energy costs and grid-associated greenhouse gas emissions, yet keep room temperatures comfortable. Another challenge is that an intelligent HEMS agent must make decisions under uncertainty. An agent must plan actions into the future, but weather and solar generation forecasts, for example, provide inherently uncertain estimates of future conditions. This paper introduces EcoNet, a Bayesian approach to household and neighborhood energy management that is based on active inference. The aim is to improve energy management and coordination, while accommodating uncertainties and taking into account potentially conditional and conflicting goals and preferences. Simulation results are presented and discussed. 

---
# From Questions to Clinical Recommendations: Large Language Models Driving Evidence-Based Clinical Decision Making 

**Authors**: Dubai Li, Nan Jiang, Kangping Huang, Ruiqi Tu, Shuyu Ouyang, Huayu Yu, Lin Qiao, Chen Yu, Tianshu Zhou, Danyang Tong, Qian Wang, Mengtao Li, Xiaofeng Zeng, Yu Tian, Xinping Tian, Jingsong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10282)  

**Abstract**: Clinical evidence, derived from rigorous research and data analysis, provides healthcare professionals with reliable scientific foundations for informed decision-making. Integrating clinical evidence into real-time practice is challenging due to the enormous workload, complex professional processes, and time constraints. This highlights the need for tools that automate evidence synthesis to support more efficient and accurate decision making in clinical settings. This study introduces Quicker, an evidence-based clinical decision support system powered by large language models (LLMs), designed to automate evidence synthesis and generate clinical recommendations modeled after standard clinical guideline development processes. Quicker implements a fully automated chain that covers all phases, from questions to clinical recommendations, and further enables customized decision-making through integrated tools and interactive user interfaces. To evaluate Quicker's capabilities, we developed the Q2CRBench-3 benchmark dataset, based on clinical guideline development records for three different diseases. Experimental results highlighted Quicker's strong performance, with fine-grained question decomposition tailored to user preferences, retrieval sensitivities comparable to human experts, and literature screening performance approaching comprehensive inclusion of relevant studies. In addition, Quicker-assisted evidence assessment effectively supported human reviewers, while Quicker's recommendations were more comprehensive and logically coherent than those of clinicians. In system-level testing, collaboration between a single reviewer and Quicker reduced the time required for recommendation development to 20-40 minutes. In general, our findings affirm the potential of Quicker to help physicians make quicker and more reliable evidence-based clinical decisions. 

---
