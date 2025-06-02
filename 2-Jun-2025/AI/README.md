# Open CaptchaWorld: A Comprehensive Web-based Platform for Testing and Benchmarking Multimodal LLM Agents 

**Authors**: Yaxin Luo, Zhaoyi Li, Jiacheng Liu, Jiacheng Cui, Xiaohan Zhao, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24878)  

**Abstract**: CAPTCHAs have been a critical bottleneck for deploying web agents in real-world applications, often blocking them from completing end-to-end automation tasks. While modern multimodal LLM agents have demonstrated impressive performance in static perception tasks, their ability to handle interactive, multi-step reasoning challenges like CAPTCHAs is largely untested. To address this gap, we introduce Open CaptchaWorld, the first web-based benchmark and platform specifically designed to evaluate the visual reasoning and interaction capabilities of MLLM-powered agents through diverse and dynamic CAPTCHA puzzles. Our benchmark spans 20 modern CAPTCHA types, totaling 225 CAPTCHAs, annotated with a new metric we propose: CAPTCHA Reasoning Depth, which quantifies the number of cognitive and motor steps required to solve each puzzle. Experimental results show that humans consistently achieve near-perfect scores, state-of-the-art MLLM agents struggle significantly, with success rates at most 40.0% by Browser-Use Openai-o3, far below human-level performance, 93.3%. This highlights Open CaptchaWorld as a vital benchmark for diagnosing the limits of current multimodal agents and guiding the development of more robust multimodal reasoning systems. Code and Data are available at this https URL. 

---
# MiCRo: Mixture Modeling and Context-aware Routing for Personalized Preference Learning 

**Authors**: Jingyan Shen, Jiarui Yao, Rui Yang, Yifan Sun, Feng Luo, Rui Pan, Tong Zhang, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24846)  

**Abstract**: Reward modeling is a key step in building safe foundation models when applying reinforcement learning from human feedback (RLHF) to align Large Language Models (LLMs). However, reward modeling based on the Bradley-Terry (BT) model assumes a global reward function, failing to capture the inherently diverse and heterogeneous human preferences. Hence, such oversimplification limits LLMs from supporting personalization and pluralistic alignment. Theoretically, we show that when human preferences follow a mixture distribution of diverse subgroups, a single BT model has an irreducible error. While existing solutions, such as multi-objective learning with fine-grained annotations, help address this issue, they are costly and constrained by predefined attributes, failing to fully capture the richness of human values. In this work, we introduce MiCRo, a two-stage framework that enhances personalized preference learning by leveraging large-scale binary preference datasets without requiring explicit fine-grained annotations. In the first stage, MiCRo introduces context-aware mixture modeling approach to capture diverse human preferences. In the second stage, MiCRo integrates an online routing strategy that dynamically adapts mixture weights based on specific context to resolve ambiguity, allowing for efficient and scalable preference adaptation with minimal additional supervision. Experiments on multiple preference datasets demonstrate that MiCRo effectively captures diverse human preferences and significantly improves downstream personalization. 

---
# EXP-Bench: Can AI Conduct AI Research Experiments? 

**Authors**: Patrick Tser Jern Kon, Jiachen Liu, Xinyi Zhu, Qiuyi Ding, Jingjia Peng, Jiarong Xing, Yibo Huang, Yiming Qiu, Jayanth Srinivasa, Myungjin Lee, Mosharaf Chowdhury, Matei Zaharia, Ang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24785)  

**Abstract**: Automating AI research holds immense potential for accelerating scientific progress, yet current AI agents struggle with the complexities of rigorous, end-to-end experimentation. We introduce EXP-Bench, a novel benchmark designed to systematically evaluate AI agents on complete research experiments sourced from influential AI publications. Given a research question and incomplete starter code, EXP-Bench challenges AI agents to formulate hypotheses, design and implement experimental procedures, execute them, and analyze results. To enable the creation of such intricate and authentic tasks with high-fidelity, we design a semi-autonomous pipeline to extract and structure crucial experimental details from these research papers and their associated open-source code. With the pipeline, EXP-Bench curated 461 AI research tasks from 51 top-tier AI research papers. Evaluations of leading LLM-based agents, such as OpenHands and IterativeAgent on EXP-Bench demonstrate partial capabilities: while scores on individual experimental aspects such as design or implementation correctness occasionally reach 20-35%, the success rate for complete, executable experiments was a mere 0.5%. By identifying these bottlenecks and providing realistic step-by-step experiment procedures, EXP-Bench serves as a vital tool for future AI agents to improve their ability to conduct AI research experiments. EXP-Bench is open-sourced at this https URL. 

---
# AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models 

**Authors**: Conor Heins, Toon Van de Maele, Alexander Tschantz, Hampus Linander, Dimitrije Markovic, Tommaso Salvatori, Corrado Pezzato, Ozan Catal, Ran Wei, Magnus Koudahl, Marco Perin, Karl Friston, Tim Verbelen, Christopher Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2505.24784)  

**Abstract**: Current deep reinforcement learning (DRL) approaches achieve state-of-the-art performance in various domains, but struggle with data efficiency compared to human learning, which leverages core priors about objects and their interactions. Active inference offers a principled framework for integrating sensory information with prior knowledge to learn a world model and quantify the uncertainty of its own beliefs and predictions. However, active inference models are usually crafted for a single task with bespoke knowledge, so they lack the domain flexibility typical of DRL approaches. To bridge this gap, we propose a novel architecture that integrates a minimal yet expressive set of core priors about object-centric dynamics and interactions to accelerate learning in low-data regimes. The resulting approach, which we call AXIOM, combines the usual data efficiency and interpretability of Bayesian approaches with the across-task generalization usually associated with DRL. AXIOM represents scenes as compositions of objects, whose dynamics are modeled as piecewise linear trajectories that capture sparse object-object interactions. The structure of the generative model is expanded online by growing and learning mixture models from single events and periodically refined through Bayesian model reduction to induce generalization. AXIOM masters various games within only 10,000 interaction steps, with both a small number of parameters compared to DRL, and without the computational expense of gradient-based optimization. 

---
# Adaptable Cardiovascular Disease Risk Prediction from Heterogeneous Data using Large Language Models 

**Authors**: Frederike Lübeck, Jonas Wildberger, Frederik Träuble, Maximilian Mordig, Sergios Gatidis, Andreas Krause, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2505.24655)  

**Abstract**: Cardiovascular disease (CVD) risk prediction models are essential for identifying high-risk individuals and guiding preventive actions. However, existing models struggle with the challenges of real-world clinical practice as they oversimplify patient profiles, rely on rigid input schemas, and are sensitive to distribution shifts. We developed AdaCVD, an adaptable CVD risk prediction framework built on large language models extensively fine-tuned on over half a million participants from the UK Biobank. In benchmark comparisons, AdaCVD surpasses established risk scores and standard machine learning approaches, achieving state-of-the-art performance. Crucially, for the first time, it addresses key clinical challenges across three dimensions: it flexibly incorporates comprehensive yet variable patient information; it seamlessly integrates both structured data and unstructured text; and it rapidly adapts to new patient populations using minimal additional data. In stratified analyses, it demonstrates robust performance across demographic, socioeconomic, and clinical subgroups, including underrepresented cohorts. AdaCVD offers a promising path toward more flexible, AI-driven clinical decision support tools suited to the realities of heterogeneous and dynamic healthcare environments. 

---
# Random Rule Forest (RRF): Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success 

**Authors**: Ben Griffin, Joseph Ternasky, Fuat Alican, Yigit Ihlamur  

**Link**: [PDF](https://arxiv.org/pdf/2505.24622)  

**Abstract**: Predicting startup success requires models that are both accurate and interpretable. We present a lightweight ensemble framework that combines YES/NO questions generated by large language models (LLMs), forming a transparent decision-making system. Each question acts as a weak heuristic, and by filtering, ranking, and aggregating them through a threshold-based voting mechanism, we construct a strong ensemble predictor. On a test set where 10% of startups are classified as successful, our approach achieves a precision rate of 50%, representing a 5x improvement over random selection, while remaining fully transparent. When we incorporate expert-guided heuristics into the generation process, performance improves further to 54% precision. These results highlight the value of combining LLM reasoning with human insight and demonstrate that simple, interpretable ensembles can support high-stakes decisions in domains such as venture capital (VC). 

---
# Taxonomic Networks: A Representation for Neuro-Symbolic Pairing 

**Authors**: Zekun Wang, Ethan L. Haarer, Nicki Barari, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24601)  

**Abstract**: We introduce the concept of a \textbf{neuro-symbolic pair} -- neural and symbolic approaches that are linked through a common knowledge representation. Next, we present \textbf{taxonomic networks}, a type of discrimination network in which nodes represent hierarchically organized taxonomic concepts. Using this representation, we construct a novel neuro-symbolic pair and evaluate its performance. We show that our symbolic method learns taxonomic nets more efficiently with less data and compute, while the neural method finds higher-accuracy taxonomic nets when provided with greater resources. As a neuro-symbolic pair, these approaches can be used interchangeably based on situational needs, with seamless translation between them when necessary. This work lays the foundation for future systems that more fundamentally integrate neural and symbolic computation. 

---
# Mixture-of-Experts for Personalized and Semantic-Aware Next Location Prediction 

**Authors**: Shuai Liu, Ning Cao, Yile Chen, Yue Jiang, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2505.24597)  

**Abstract**: Next location prediction plays a critical role in understanding human mobility patterns. However, existing approaches face two core limitations: (1) they fall short in capturing the complex, multi-functional semantics of real-world locations; and (2) they lack the capacity to model heterogeneous behavioral dynamics across diverse user groups. To tackle these challenges, we introduce NextLocMoE, a novel framework built upon large language models (LLMs) and structured around a dual-level Mixture-of-Experts (MoE) design. Our architecture comprises two specialized modules: a Location Semantics MoE that operates at the embedding level to encode rich functional semantics of locations, and a Personalized MoE embedded within the Transformer backbone to dynamically adapt to individual user mobility patterns. In addition, we incorporate a history-aware routing mechanism that leverages long-term trajectory data to enhance expert selection and ensure prediction stability. Empirical evaluations across several real-world urban datasets show that NextLocMoE achieves superior performance in terms of predictive accuracy, cross-domain generalization, and interpretability 

---
# MELT: Towards Automated Multimodal Emotion Data Annotation by Leveraging LLM Embedded Knowledge 

**Authors**: Xin Jing, Jiadong Wang, Iosif Tsangko, Andreas Triantafyllopoulos, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2505.24493)  

**Abstract**: Although speech emotion recognition (SER) has advanced significantly with deep learning, annotation remains a major hurdle. Human annotation is not only costly but also subject to inconsistencies annotators often have different preferences and may lack the necessary contextual knowledge, which can lead to varied and inaccurate labels. Meanwhile, Large Language Models (LLMs) have emerged as a scalable alternative for annotating text data. However, the potential of LLMs to perform emotional speech data annotation without human supervision has yet to be thoroughly investigated. To address these problems, we apply GPT-4o to annotate a multimodal dataset collected from the sitcom Friends, using only textual cues as inputs. By crafting structured text prompts, our methodology capitalizes on the knowledge GPT-4o has accumulated during its training, showcasing that it can generate accurate and contextually relevant annotations without direct access to multimodal inputs. Therefore, we propose MELT, a multimodal emotion dataset fully annotated by GPT-4o. We demonstrate the effectiveness of MELT by fine-tuning four self-supervised learning (SSL) backbones and assessing speech emotion recognition performance across emotion datasets. Additionally, our subjective experiments\' results demonstrate a consistence performance improvement on SER. 

---
# Leveraging Knowledge Graphs and LLMs for Structured Generation of Misinformation 

**Authors**: Sania Nayab, Marco Simoni, Giulio Rossolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.24479)  

**Abstract**: The rapid spread of misinformation, further amplified by recent advances in generative AI, poses significant threats to society, impacting public opinion, democratic stability, and national security. Understanding and proactively assessing these threats requires exploring methodologies that enable structured and scalable misinformation generation. In this paper, we propose a novel approach that leverages knowledge graphs (KGs) as structured semantic resources to systematically generate fake triplets. By analyzing the structural properties of KGs, such as the distance between entities and their predicates, we identify plausibly false relationships. These triplets are then used to guide large language models (LLMs) in generating misinformation statements with varying degrees of credibility. By utilizing structured semantic relationships, our deterministic approach produces misinformation inherently challenging for humans to detect, drawing exclusively upon publicly available KGs (e.g., WikiGraphs).
Additionally, we investigate the effectiveness of LLMs in distinguishing between genuine and artificially generated misinformation. Our analysis highlights significant limitations in current LLM-based detection methods, underscoring the necessity for enhanced detection strategies and a deeper exploration of inherent biases in generative models. 

---
# Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning 

**Authors**: Vasilije Markovic, Lazar Obradovic, Laszlo Hajdu, Jovan Pavlovic  

**Link**: [PDF](https://arxiv.org/pdf/2505.24478)  

**Abstract**: Integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) results in complex systems with numerous hyperparameters that directly affect performance. While such systems are increasingly common in retrieval-augmented generation, the role of systematic hyperparameter optimization remains underexplored. In this paper, we study this problem in the context of Cognee, a modular framework for end-to-end KG construction and retrieval. Using three multi-hop QA benchmarks (HotPotQA, TwoWikiMultiHop, and MuSiQue) we optimize parameters related to chunking, graph construction, retrieval, and prompting. Each configuration is scored using established metrics (exact match, F1, and DeepEval's LLM-based correctness metric). Our results demonstrate that meaningful gains can be achieved through targeted tuning. While the gains are consistent, they are not uniform, with performance varying across datasets and metrics. This variability highlights both the value of tuning and the limitations of standard evaluation measures. While demonstrating the immediate potential of hyperparameter tuning, we argue that future progress will depend not only on architectural advances but also on clearer frameworks for optimization and evaluation in complex, modular systems. 

---
# SEAR: A Multimodal Dataset for Analyzing AR-LLM-Driven Social Engineering Behaviors 

**Authors**: Tianlong Yu, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Zui Tao, Jun Zhang, Kailong Wang, Liting Zhou, Yang Yang, Ting Bi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24458)  

**Abstract**: The SEAR Dataset is a novel multimodal resource designed to study the emerging threat of social engineering (SE) attacks orchestrated through augmented reality (AR) and multimodal large language models (LLMs). This dataset captures 180 annotated conversations across 60 participants in simulated adversarial scenarios, including meetings, classes and networking events. It comprises synchronized AR-captured visual/audio cues (e.g., facial expressions, vocal tones), environmental context, and curated social media profiles, alongside subjective metrics such as trust ratings and susceptibility assessments. Key findings reveal SEAR's alarming efficacy in eliciting compliance (e.g., 93.3% phishing link clicks, 85% call acceptance) and hijacking trust (76.7% post-interaction trust surge). The dataset supports research in detecting AR-driven SE attacks, designing defensive frameworks, and understanding multimodal adversarial manipulation. Rigorous ethical safeguards, including anonymization and IRB compliance, ensure responsible use. The SEAR dataset is available at this https URL. 

---
# RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation 

**Authors**: Zhentao Xie, Chengcheng Han, Jinxin Shi, Wenjun Cui, Xin Zhao, Xingjiao Wu, Jiabao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24442)  

**Abstract**: Although multi-agent systems based on large language models show strong capabilities on multiple tasks, they are still limited by high computational overhead, information loss, and robustness. Inspired by ResNet's residual learning, we propose Residual Mixture-of-Agents (RMoA), integrating residual connections to optimize efficiency and reliability. To maximize information utilization from model responses while minimizing computational costs, we innovatively design an embedding-based diversity selection mechanism that greedily selects responses via vector similarity. Furthermore, to mitigate iterative information degradation, we introduce a Residual Extraction Agent to preserve cross-layer incremental information by capturing inter-layer response differences, coupled with a Residual Aggregation Agent for hierarchical information integration. Additionally, we propose an adaptive termination mechanism that dynamically halts processing based on residual convergence, further improving inference efficiency. RMoA achieves state-of-the-art performance on the benchmarks of across alignment, mathematical reasoning, code generation, and multitasking understanding, while significantly reducing computational overhead. Code is available at this https URL. 

---
# P: A Universal Measure of Predictive Intelligence 

**Authors**: David Gamez  

**Link**: [PDF](https://arxiv.org/pdf/2505.24426)  

**Abstract**: Over the last thirty years, considerable progress has been made with the development of systems that can drive cars, play games, predict protein folding and generate natural language. These systems are described as intelligent and there has been a great deal of talk about the rapid increase in artificial intelligence and its potential dangers. However, our theoretical understanding of intelligence and ability to measure it lag far behind our capacity for building systems that mimic intelligent human behaviour. There is no commonly agreed definition of the intelligence that AI systems are said to possess. No-one has developed a practical measure that would enable us to compare the intelligence of humans, animals and AIs on a single ratio scale.
This paper sets out a new universal measure of intelligence that is based on the hypothesis that prediction is the most important component of intelligence. As an agent interacts with its normal environment, the accuracy of its predictions is summed up and the complexity of its predictions and perceived environment is accounted for using Kolmogorov complexity. Two experiments were carried out to evaluate the practical feasibility of the algorithm. These demonstrated that it could measure the intelligence of an agent embodied in a virtual maze and an agent that makes predictions about time-series data. This universal measure could be the starting point for a new comparative science of intelligence that ranks humans, animals and AIs on a single ratio scale. 

---
# Three Kinds of Negation in Knowledge and Their Mathematical Foundations 

**Authors**: Zhenghua Pan, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24422)  

**Abstract**: In the field of artificial intelligence, understanding, distinguishing, expressing, and computing the negation in knowledge is a fundamental issue in knowledge processing and research. In this paper, we examine and analyze the understanding and characteristics of negation in various fields such as philosophy, logic, and linguistics etc. Based on the distinction between the concepts of contradiction and opposition, we propose that there are three different types of negation in knowledge from a conceptual perspective: contradictory negation, opposite negation, and intermediary negation. To establish a mathematical foundation that fully reflects the intrinsic connections, properties, and laws of these different forms of negation, we introduce SCOI: sets with contradictory negation, opposite negation and intermediary negation, and LCOI: logic with contradictory negation, opposite negation and intermediary negation, and we proved the main operational properties of SCOI as well as the formal inference relations in LCOI. 

---
# GridRoute: A Benchmark for LLM-Based Route Planning with Cardinal Movement in Grid Environments 

**Authors**: Kechen Li, Yaotian Tao, Ximing Wen, Quanwei Sun, Zifei Gong, Chang Xu, Xizhe Zhang, Tianbo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.24306)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated their potential in planning and reasoning tasks, offering a flexible alternative to classical pathfinding algorithms. However, most existing studies focus on LLMs' independent reasoning capabilities and overlook the potential synergy between LLMs and traditional algorithms. To fill this gap, we propose a comprehensive evaluation benchmark GridRoute to assess how LLMs can take advantage of traditional algorithms. We also propose a novel hybrid prompting technique called Algorithm of Thought (AoT), which introduces traditional algorithms' guidance into prompting. Our benchmark evaluates six LLMs ranging from 7B to 72B parameters across various map sizes, assessing their performance in correctness, optimality, and efficiency in grid environments with varying sizes. Our results show that AoT significantly boosts performance across all model sizes, particularly in larger or more complex environments, suggesting a promising approach to addressing path planning challenges. Our code is open-sourced at this https URL. 

---
# Mind the Quote: Enabling Quotation-Aware Dialogue in LLMs via Plug-and-Play Modules 

**Authors**: Yueqi Zhang, Peiwen Yuan, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24292)  

**Abstract**: Human-AI conversation frequently relies on quoting earlier text-"check it with the formula I just highlighted"-yet today's large language models (LLMs) lack an explicit mechanism for locating and exploiting such spans. We formalise the challenge as span-conditioned generation, decomposing each turn into the dialogue history, a set of token-offset quotation spans, and an intent utterance. Building on this abstraction, we introduce a quotation-centric data pipeline that automatically synthesises task-specific dialogues, verifies answer correctness through multi-stage consistency checks, and yields both a heterogeneous training corpus and the first benchmark covering five representative scenarios. To meet the benchmark's zero-overhead and parameter-efficiency requirements, we propose QuAda, a lightweight training-based method that attaches two bottleneck projections to every attention head, dynamically amplifying or suppressing attention to quoted spans at inference time while leaving the prompt unchanged and updating < 2.8% of backbone weights. Experiments across models show that QuAda is suitable for all scenarios and generalises to unseen topics, offering an effective, plug-and-play solution for quotation-aware dialogue. 

---
# How Much Backtracking is Enough? Exploring the Interplay of SFT and RL in Enhancing LLM Reasoning 

**Authors**: Hongyi James Cai, Junlin Wang, Xiaoyin Chen, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2505.24273)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have effectively improved their reasoning abilities, particularly on mathematical and logical problems that have verifiable answers, through techniques such as supervised finetuning (SFT) and reinforcement learning (RL). Prior research indicates that RL effectively internalizes search strategies, enabling long chain-of-thought (CoT) reasoning, with backtracking emerging naturally as a learned capability. However, the precise benefits of backtracking, specifically, how significantly it contributes to reasoning improvements and the optimal extent of its use, remain poorly understood. In this work, we systematically investigate the dynamics between SFT and RL on eight reasoning tasks: Countdown, Sudoku, Arc 1D, Geometry, Color Cube Rotation, List Functions, Zebra Puzzles, and Self Reference. Our findings highlight that short CoT sequences used in SFT as a warm-up do have moderate contribution to RL training, compared with cold-start RL; however such contribution diminishes when tasks become increasingly difficult. Motivated by this observation, we construct synthetic datasets varying systematically in the number of backtracking steps and conduct controlled experiments to isolate the influence of either the correctness (content) or the structure (i.e., backtrack frequency). We find that (1) longer CoT with backtracks generally induce better and more stable RL training, (2) more challenging problems with larger search space tend to need higher numbers of backtracks during the SFT stage. Additionally, we demonstrate through experiments on distilled data that RL training is largely unaffected by the correctness of long CoT sequences, suggesting that RL prioritizes structural patterns over content correctness. Collectively, our results offer practical insights into designing optimal training strategies to effectively scale reasoning in LLMs. 

---
# Generative AI for Urban Design: A Stepwise Approach Integrating Human Expertise with Multimodal Diffusion Models 

**Authors**: Mingyi He, Yuebing Liang, Shenhao Wang, Yunhan Zheng, Qingyi Wang, Dingyi Zhuang, Li Tian, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24260)  

**Abstract**: Urban design is a multifaceted process that demands careful consideration of site-specific constraints and collaboration among diverse professionals and stakeholders. The advent of generative artificial intelligence (GenAI) offers transformative potential by improving the efficiency of design generation and facilitating the communication of design ideas. However, most existing approaches are not well integrated with human design workflows. They often follow end-to-end pipelines with limited control, overlooking the iterative nature of real-world design. This study proposes a stepwise generative urban design framework that integrates multimodal diffusion models with human expertise to enable more adaptive and controllable design processes. Instead of generating design outcomes in a single end-to-end process, the framework divides the process into three key stages aligned with established urban design workflows: (1) road network and land use planning, (2) building layout planning, and (3) detailed planning and rendering. At each stage, multimodal diffusion models generate preliminary designs based on textual prompts and image-based constraints, which can then be reviewed and refined by human designers. We design an evaluation framework to assess the fidelity, compliance, and diversity of the generated designs. Experiments using data from Chicago and New York City demonstrate that our framework outperforms baseline models and end-to-end approaches across all three dimensions. This study underscores the benefits of multimodal diffusion models and stepwise generation in preserving human control and facilitating iterative refinements, laying the groundwork for human-AI interaction in urban design solutions. 

---
# FABLE: A Novel Data-Flow Analysis Benchmark on Procedural Text for Large Language Model Evaluation 

**Authors**: Vishal Pallagani, Nitin Gupta, John Aydin, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.24258)  

**Abstract**: Understanding how data moves, transforms, and persists, known as data flow, is fundamental to reasoning in procedural tasks. Despite their fluency in natural and programming languages, large language models (LLMs), although increasingly being applied to decisions with procedural tasks, have not been systematically evaluated for their ability to perform data-flow reasoning. We introduce FABLE, an extensible benchmark designed to assess LLMs' understanding of data flow using structured, procedural text. FABLE adapts eight classical data-flow analyses from software engineering: reaching definitions, very busy expressions, available expressions, live variable analysis, interval analysis, type-state analysis, taint analysis, and concurrency analysis. These analyses are instantiated across three real-world domains: cooking recipes, travel routes, and automated plans. The benchmark includes 2,400 question-answer pairs, with 100 examples for each domain-analysis combination. We evaluate three types of LLMs: a reasoning-focused model (DeepSeek-R1 8B), a general-purpose model (LLaMA 3.1 8B), and a code-specific model (Granite Code 8B). Each model is tested using majority voting over five sampled completions per prompt. Results show that the reasoning model achieves higher accuracy, but at the cost of over 20 times slower inference compared to the other models. In contrast, the general-purpose and code-specific models perform close to random chance. FABLE provides the first diagnostic benchmark to systematically evaluate data-flow reasoning and offers insights for developing models with stronger procedural understanding. 

---
# ProofNet++: A Neuro-Symbolic System for Formal Proof Verification with Self-Correction 

**Authors**: Murari Ambati  

**Link**: [PDF](https://arxiv.org/pdf/2505.24230)  

**Abstract**: We propose ProofNet++, a neuro-symbolic framework that enhances automated theorem proving by combining large language models (LLMs) with formal proof verification and self-correction mechanisms. Current LLM-based systems suffer from hallucinated logical steps and unverifiable reasoning. ProofNet++ mitigates these limitations by integrating symbolic proof tree supervision, a reinforcement learning loop using verifiers as reward functions, and an iterative self-correction module. Our experiments on miniF2F, Lean's mathlib, and HOL Light show that ProofNet++ significantly improves proof accuracy, correctness, and formal verifiability over prior models. We provide theoretical analysis of the convergence and stability of the verifier-guided RL framework and release our datasets and codebase for future research. 

---
# E^2GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness 

**Authors**: Yibo Zhao, Jiapeng Zhu, Ye Guo, Kangkang He, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24226)  

**Abstract**: Graph-based RAG methods like GraphRAG have shown promising global understanding of the knowledge base by constructing hierarchical entity graphs. However, they often suffer from inefficiency and rely on manually pre-defined query modes, limiting practical use. In this paper, we propose E^2GraphRAG, a streamlined graph-based RAG framework that improves both Efficiency and Effectiveness. During the indexing stage, E^2GraphRAG constructs a summary tree with large language models and an entity graph with SpaCy based on document chunks. We then construct bidirectional indexes between entities and chunks to capture their many-to-many relationships, enabling fast lookup during both local and global retrieval. For the retrieval stage, we design an adaptive retrieval strategy that leverages the graph structure to retrieve and select between local and global modes. Experiments show that E^2GraphRAG achieves up to 10 times faster indexing than GraphRAG and 100 times speedup over LightRAG in retrieval while maintaining competitive QA performance. 

---
# Bootstrapping LLM Robustness for VLM Safety via Reducing the Pretraining Modality Gap 

**Authors**: Wenhan Yang, Spencer Stice, Ali Payani, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2505.24208)  

**Abstract**: Ensuring Vision-Language Models (VLMs) generate safe outputs is crucial for their reliable deployment. However, LVLMs suffer from drastic safety degradation compared to their LLM backbone. Even blank or irrelevant images can trigger LVLMs to generate harmful responses to prompts that would otherwise be refused in text-only contexts. The modality gap between image and text representations has been recently hypothesized to contribute to safety degradation of LVLMs. However, if and how the amount of modality gap affects LVLMs' safety is not studied. In this work, we show that the amount of modality gap is highly inversely correlated with VLMs' safety. Then, we show that this modality gap is introduced during pretraining LVLMs and persists through fine-tuning. Inspired by this observation, we propose a regularization to reduce the modality gap during pretraining. Our extensive experiments on LLaVA v1.5, ShareGPT4V, and MiniGPT-4 show that our method substantially improves safety alignment of LVLMs, reducing unsafe rate by up to 16.3% without compromising performance, and can further boost existing defenses by up to 18.2%. 

---
# SentinelAgent: Graph-based Anomaly Detection in Multi-Agent Systems 

**Authors**: Xu He, Di Wu, Yan Zhai, Kun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.24201)  

**Abstract**: The rise of large language model (LLM)-based multi-agent systems (MAS) introduces new security and reliability challenges. While these systems show great promise in decomposing and coordinating complex tasks, they also face multi-faceted risks across prompt manipulation, unsafe tool usage, and emergent agent miscoordination. Existing guardrail mechanisms offer only partial protection, primarily at the input-output level, and fall short in addressing systemic or multi-point failures in MAS. In this work, we present a system-level anomaly detection framework tailored for MAS, integrating structural modeling with runtime behavioral oversight. Our approach consists of two components. First, we propose a graph-based framework that models agent interactions as dynamic execution graphs, enabling semantic anomaly detection at node, edge, and path levels. Second, we introduce a pluggable SentinelAgent, an LLM-powered oversight agent that observes, analyzes, and intervenes in MAS execution based on security policies and contextual reasoning. By bridging abstract detection logic with actionable enforcement, our method detects not only single-point faults and prompt injections but also multi-agent collusion and latent exploit paths. We validate our framework through two case studies, including an email assistant and Microsoft's Magentic-One system, demonstrating its ability to detect covert risks and provide explainable root-cause attribution. Our work lays the foundation for more trustworthy, monitorable, and secure agent-based AI ecosystems. 

---
# Learning API Functionality from Demonstrations for Tool-based Agents 

**Authors**: Bhrij Patel, Ashish Jagmohan, Aditya Vempaty  

**Link**: [PDF](https://arxiv.org/pdf/2505.24197)  

**Abstract**: Digital tool-based agents that invoke external Application Programming Interfaces (APIs) often rely on documentation to understand API functionality. However, such documentation is frequently missing, outdated, privatized, or inconsistent-hindering the development of reliable, general-purpose agents. In this work, we propose learning API functionality directly from demonstrations as a new paradigm applicable in scenarios without documentation. Using existing API benchmarks, we collect demonstrations from both expert API-based agents and from self-exploration. To understand what information demonstrations must convey for successful task completion, we extensively study how the number of demonstrations and the use of LLM-generated summaries and evaluations affect the task success rate of the API-based agent. Our experiments across 3 datasets and 5 models show that learning functionality from demonstrations remains a non-trivial challenge, even for state-of-the-art LLMs. We find that providing explicit function calls and natural language critiques significantly improves the agent's task success rate due to more accurate parameter filling. We analyze failure modes, identify sources of error, and highlight key open challenges for future work in documentation-free, self-improving, API-based agents. 

---
# SCOUT: Teaching Pre-trained Language Models to Enhance Reasoning via Flow Chain-of-Thought 

**Authors**: Guanghao Li, Wenhao Jiang, Mingfeng Chen, Yan Li, Hao Yu, Shuting Dong, Tao Ren, Ming Tang, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24181)  

**Abstract**: Chain of Thought (CoT) prompting improves the reasoning performance of large language models (LLMs) by encouraging step by step thinking. However, CoT-based methods depend on intermediate reasoning steps, which limits scalability and generalization. Recent work explores recursive reasoning, where LLMs reuse internal layers across iterations to refine latent representations without explicit CoT supervision. While promising, these approaches often require costly pretraining and lack a principled framework for how reasoning should evolve across iterations. We address this gap by introducing Flow Chain of Thought (Flow CoT), a reasoning paradigm that models recursive inference as a progressive trajectory of latent cognitive states. Flow CoT frames each iteration as a distinct cognitive stage deepening reasoning across iterations without relying on manual supervision. To realize this, we propose SCOUT (Stepwise Cognitive Optimization Using Teachers), a lightweight fine tuning framework that enables Flow CoT style reasoning without the need for pretraining. SCOUT uses progressive distillation to align each iteration with a teacher of appropriate capacity, and a cross attention based retrospective module that integrates outputs from previous iterations while preserving the models original computation flow. Experiments across eight reasoning benchmarks show that SCOUT consistently improves both accuracy and explanation quality, achieving up to 1.8% gains under fine tuning. Qualitative analyses further reveal that SCOUT enables progressively deeper reasoning across iterations refining both belief formation and explanation granularity. These results not only validate the effectiveness of SCOUT, but also demonstrate the practical viability of Flow CoT as a scalable framework for enhancing reasoning in LLMs. 

---
# mRAG: Elucidating the Design Space of Multi-modal Retrieval-Augmented Generation 

**Authors**: Chan-Wei Hu, Yueqi Wang, Shuo Xing, Chia-Ju Chen, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24073)  

**Abstract**: Large Vision-Language Models (LVLMs) have made remarkable strides in multimodal tasks such as visual question answering, visual grounding, and complex reasoning. However, they remain limited by static training data, susceptibility to hallucinations, and inability to verify claims against up-to-date, external evidence, compromising their performance in dynamic real-world applications. Retrieval-Augmented Generation (RAG) offers a practical solution to mitigate these challenges by allowing the LVLMs to access large-scale knowledge databases via retrieval mechanisms, thereby grounding model outputs in factual, contextually relevant information. Here in this paper, we conduct the first systematic dissection of the multimodal RAG pipeline for LVLMs, explicitly investigating (1) the retrieval phase: on the modality configurations and retrieval strategies, (2) the re-ranking stage: on strategies to mitigate positional biases and improve the relevance of retrieved evidence, and (3) the generation phase: we further investigate how to best integrate retrieved candidates into the final generation process. Finally, we extend to explore a unified agentic framework that integrates re-ranking and generation through self-reflection, enabling LVLMs to select relevant evidence and suppress irrelevant context dynamically. Our full-stack exploration of RAG for LVLMs yields substantial insights, resulting in an average performance boost of 5% without any fine-tuning. 

---
# Leave it to the Specialist: Repair Sparse LLMs with Sparse Fine-Tuning via Sparsity Evolution 

**Authors**: Qiao Xiao, Alan Ansell, Boqian Wu, Lu Yin, Mykola Pechenizkiy, Shiwei Liu, Decebal Constantin Mocanu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24037)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across various tasks but face deployment challenges due to their massive computational demands. While post-training pruning methods like SparseGPT and Wanda can effectively reduce the model size, but struggle to maintain model performance at high sparsity levels, limiting their utility for downstream tasks. Existing fine-tuning methods, such as full fine-tuning and LoRA, fail to preserve sparsity as they require updating the whole dense metrics, not well-suited for sparse LLMs. In this paper, we propose Sparsity Evolution Fine-Tuning (SEFT), a novel method designed specifically for sparse LLMs. SEFT dynamically evolves the sparse topology of pruned models during fine-tuning, while preserving the overall sparsity throughout the process. The strengths of SEFT lie in its ability to perform task-specific adaptation through a weight drop-and-grow strategy, enabling the pruned model to self-adapt its sparse connectivity pattern based on the target dataset. Furthermore, a sensitivity-driven pruning criterion is employed to ensure that the desired sparsity level is consistently maintained throughout fine-tuning. Our experiments on various LLMs, including LLaMA families, DeepSeek, and Mistral, across a diverse set of benchmarks demonstrate that SEFT achieves stronger performance while offering superior memory and time efficiency compared to existing baselines. Our code is publicly available at: this https URL. 

---
# GenIC: An LLM-Based Framework for Instance Completion in Knowledge Graphs 

**Authors**: Amel Gader, Alsayed Algergawy  

**Link**: [PDF](https://arxiv.org/pdf/2505.24036)  

**Abstract**: Knowledge graph completion aims to address the gaps of knowledge bases by adding new triples that represent facts. The complexity of this task depends on how many parts of a triple are already known. Instance completion involves predicting the relation-tail pair when only the head is given (h, ?, ?). Notably, modern knowledge bases often contain entity descriptions and types, which can provide valuable context for inferring missing facts. By leveraging these textual descriptions and the ability of large language models to extract facts from them and recognize patterns within the knowledge graph schema, we propose an LLM-powered, end-to-end instance completion approach. Specifically, we introduce GenIC: a two-step Generative Instance Completion framework. The first step focuses on property prediction, treated as a multi-label classification task. The second step is link prediction, framed as a generative sequence-to-sequence task. Experimental results on three datasets show that our method outperforms existing baselines. Our code is available at this https URL. 

---
# Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding 

**Authors**: Mingyang Mao, Mariela M. Perez-Cabarcas, Utteja Kallakuri, Nicholas R. Waytowich, Xiaomin Lin, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23990)  

**Abstract**: To effectively engage in human society, the ability to adapt, filter information, and make informed decisions in ever-changing situations is critical. As robots and intelligent agents become more integrated into human life, there is a growing opportunity-and need-to offload the cognitive burden on humans to these systems, particularly in dynamic, information-rich scenarios.
To fill this critical need, we present Multi-RAG, a multimodal retrieval-augmented generation system designed to provide adaptive assistance to humans in information-intensive circumstances. Our system aims to improve situational understanding and reduce cognitive load by integrating and reasoning over multi-source information streams, including video, audio, and text. As an enabling step toward long-term human-robot partnerships, Multi-RAG explores how multimodal information understanding can serve as a foundation for adaptive robotic assistance in dynamic, human-centered situations. To evaluate its capability in a realistic human-assistance proxy task, we benchmarked Multi-RAG on the MMBench-Video dataset, a challenging multimodal video understanding benchmark. Our system achieves superior performance compared to existing open-source video large language models (Video-LLMs) and large vision-language models (LVLMs), while utilizing fewer resources and less input data. The results demonstrate Multi- RAG's potential as a practical and efficient foundation for future human-robot adaptive assistance systems in dynamic, real-world contexts. 

---
# MSQA: Benchmarking LLMs on Graduate-Level Materials Science Reasoning and Knowledge 

**Authors**: Jerry Junyang Cheung, Shiyao Shen, Yuchen Zhuang, Yinghao Li, Rampi Ramprasad, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23982)  

**Abstract**: Despite recent advances in large language models (LLMs) for materials science, there is a lack of benchmarks for evaluating their domain-specific knowledge and complex reasoning abilities. To bridge this gap, we introduce MSQA, a comprehensive evaluation benchmark of 1,757 graduate-level materials science questions in two formats: detailed explanatory responses and binary True/False assessments. MSQA distinctively challenges LLMs by requiring both precise factual knowledge and multi-step reasoning across seven materials science sub-fields, such as structure-property relationships, synthesis processes, and computational modeling. Through experiments with 10 state-of-the-art LLMs, we identify significant gaps in current LLM performance. While API-based proprietary LLMs achieve up to 84.5% accuracy, open-source (OSS) LLMs peak around 60.5%, and domain-specific LLMs often underperform significantly due to overfitting and distributional shifts. MSQA represents the first benchmark to jointly evaluate the factual and reasoning capabilities of LLMs crucial for LLMs in advanced materials science. 

---
# InterMT: Multi-Turn Interleaved Preference Alignment with Human Feedback 

**Authors**: Boyuan Chen, Donghai Hong, Jiaming Ji, Jiacheng Zheng, Bowen Dong, Jiayi Zhou, Kaile Wang, Juntao Dai, Xuyao Wang, Wenqi Chen, Qirui Zheng, Wenxin Li, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23950)  

**Abstract**: As multimodal large models (MLLMs) continue to advance across challenging tasks, a key question emerges: What essential capabilities are still missing? A critical aspect of human learning is continuous interaction with the environment -- not limited to language, but also involving multimodal understanding and generation. To move closer to human-level intelligence, models must similarly support multi-turn, multimodal interaction. In particular, they should comprehend interleaved multimodal contexts and respond coherently in ongoing exchanges. In this work, we present an initial exploration through the InterMT -- the first preference dataset for multi-turn multimodal interaction, grounded in real human feedback. In this exploration, we particularly emphasize the importance of human oversight, introducing expert annotations to guide the process, motivated by the fact that current MLLMs lack such complex interactive capabilities. InterMT captures human preferences at both global and local levels into nine sub-dimensions, consists of 15.6k prompts, 52.6k multi-turn dialogue instances, and 32.4k human-labeled preference pairs. To compensate for the lack of capability for multi-modal understanding and generation, we introduce an agentic workflow that leverages tool-augmented MLLMs to construct multi-turn QA instances. To further this goal, we introduce InterMT-Bench to assess the ability of MLLMs in assisting judges with multi-turn, multimodal tasks. We demonstrate the utility of \InterMT through applications such as judge moderation and further reveal the multi-turn scaling law of judge model. We hope the open-source of our data can help facilitate further research on aligning current MLLMs to the next step. Our project website can be found at this https URL . 

---
# Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve 

**Authors**: Yuanzhe Liu, Ryan Deng, Tim Kaler, Xuhao Chen, Charles E. Leiserson, Yao Ma, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23946)  

**Abstract**: Recent studies show that LLMs possess different skills and specialize in different tasks. In fact, we observe that their varied performance occur in several levels of granularity. For example, in the code optimization task, code LLMs excel at different optimization categories and no one dominates others. This observation prompts the question of how one leverages multiple LLM agents to solve a coding problem without knowing their complementary strengths a priori. We argue that a team of agents can learn from each other's successes and failures so as to improve their own performance. Thus, a lesson is the knowledge produced by an agent and passed on to other agents in the collective solution process. We propose a lesson-based collaboration framework, design the lesson solicitation--banking--selection mechanism, and demonstrate that a team of small LLMs with lessons learned can outperform a much larger LLM and other multi-LLM collaboration methods. 

---
# OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation 

**Authors**: Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Bernard Ghanem, Ping Luo, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23885)  

**Abstract**: Large Language Model (LLM)-based multi-agent systems show promise for automating real-world tasks but struggle to transfer across domains due to their domain-specific nature. Current approaches face two critical shortcomings: they require complete architectural redesign and full retraining of all components when applied to new domains. We introduce Workforce, a hierarchical multi-agent framework that decouples strategic planning from specialized execution through a modular architecture comprising: (i) a domain-agnostic Planner for task decomposition, (ii) a Coordinator for subtask management, and (iii) specialized Workers with domain-specific tool-calling capabilities. This decoupling enables cross-domain transferability during both inference and training phases: During inference, Workforce seamlessly adapts to new domains by adding or modifying worker agents; For training, we introduce Optimized Workforce Learning (OWL), which improves generalization across domains by optimizing a domain-agnostic planner with reinforcement learning from real-world feedback. To validate our approach, we evaluate Workforce on the GAIA benchmark, covering various realistic, multi-domain agentic tasks. Experimental results demonstrate Workforce achieves open-source state-of-the-art performance (69.70%), outperforming commercial systems like OpenAI's Deep Research by 2.34%. More notably, our OWL-trained 32B model achieves 52.73% accuracy (+16.37%) and demonstrates performance comparable to GPT-4o on challenging tasks. To summarize, by enabling scalable generalization and modular domain transfer, our work establishes a foundation for the next generation of general-purpose AI assistants. 

---
# Using Reasoning Models to Generate Search Heuristics that Solve Open Instances of Combinatorial Design Problems 

**Authors**: Christopher D. Rosin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23881)  

**Abstract**: Large Language Models (LLMs) with reasoning are trained to iteratively generate and refine their answers before finalizing them, which can help with applications to mathematics and code generation. We apply code generation with reasoning LLMs to a specific task in the mathematical field of combinatorial design. This field studies diverse types of combinatorial designs, many of which have lists of open instances for which existence has not yet been determined. The Constructive Protocol CPro1 uses LLMs to generate search heuristics that have the potential to construct solutions to small open instances. Starting with a textual definition and a validity verifier for a particular type of design, CPro1 guides LLMs to select and implement strategies, while providing automated hyperparameter tuning and execution feedback. CPro1 with reasoning LLMs successfully solves long-standing open instances for 7 of 16 combinatorial design problems selected from the 2006 Handbook of Combinatorial Designs, including new solved instances for 3 of these (Bhaskar Rao Designs, Symmetric Weighing Matrices, Balanced Ternary Designs) that were unsolved by CPro1 with non-reasoning LLMs. It also solves open instances for several problems from recent (2025) literature, generating new Covering Sequences, Johnson Clique Covers, Deletion Codes, and a Uniform Nested Steiner Quadruple System. 

---
# ProxyThinker: Test-Time Guidance through Small Visual Reasoners 

**Authors**: Zilin Xiao, Jaywon Koo, Siru Ouyang, Jefferson Hernandez, Yu Meng, Vicente Ordonez  

**Link**: [PDF](https://arxiv.org/pdf/2505.24872)  

**Abstract**: Recent advancements in reinforcement learning with verifiable rewards have pushed the boundaries of the visual reasoning capabilities in large vision-language models (LVLMs). However, training LVLMs with reinforcement fine-tuning (RFT) is computationally expensive, posing a significant challenge to scaling model size. In this work, we propose ProxyThinker, an inference-time technique that enables large models to inherit the visual reasoning capabilities from small, slow-thinking visual reasoners without any training. By subtracting the output distributions of base models from those of RFT reasoners, ProxyThinker modifies the decoding dynamics and successfully elicits the slow-thinking reasoning demonstrated by the emerged sophisticated behaviors such as self-verification and self-correction. ProxyThinker consistently boosts performance on challenging visual benchmarks on spatial, mathematical, and multi-disciplinary reasoning, enabling untuned base models to compete with the performance of their full-scale RFT counterparts. Furthermore, our implementation efficiently coordinates multiple language models with parallelism techniques and achieves up to 38 $\times$ faster inference compared to previous decoding-time methods, paving the way for the practical deployment of ProxyThinker. Code is available at this https URL. 

---
# Time Blindness: Why Video-Language Models Can't See What Humans Can? 

**Authors**: Ujjwal Upadhyay, Mukul Ranjan, Zhiqiang Shen, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2505.24867)  

**Abstract**: Recent advances in vision-language models (VLMs) have made impressive strides in understanding spatio-temporal relationships in videos. However, when spatial information is obscured, these models struggle to capture purely temporal patterns. We introduce $\textbf{SpookyBench}$, a benchmark where information is encoded solely in temporal sequences of noise-like frames, mirroring natural phenomena from biological signaling to covert communication. Interestingly, while humans can recognize shapes, text, and patterns in these sequences with over 98% accuracy, state-of-the-art VLMs achieve 0% accuracy. This performance gap highlights a critical limitation: an over-reliance on frame-level spatial features and an inability to extract meaning from temporal cues. Furthermore, when trained in data sets with low spatial signal-to-noise ratios (SNR), temporal understanding of models degrades more rapidly than human perception, especially in tasks requiring fine-grained temporal reasoning. Overcoming this limitation will require novel architectures or training paradigms that decouple spatial dependencies from temporal processing. Our systematic analysis shows that this issue persists across model scales and architectures. We release SpookyBench to catalyze research in temporal pattern recognition and bridge the gap between human and machine video understanding. Dataset and code has been made available on our project website: this https URL. 

---
# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models 

**Authors**: Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.24864)  

**Abstract**: Recent advances in reasoning-centric language models have highlighted reinforcement learning (RL) as a promising method for aligning models with verifiable rewards. However, it remains contentious whether RL truly expands a model's reasoning capabilities or merely amplifies high-reward outputs already latent in the base model's distribution, and whether continually scaling up RL compute reliably leads to improved reasoning performance. In this work, we challenge prevailing assumptions by demonstrating that prolonged RL (ProRL) training can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling. We introduce ProRL, a novel training methodology that incorporates KL divergence control, reference policy resetting, and a diverse suite of tasks. Our empirical analysis reveals that RL-trained models consistently outperform base models across a wide range of pass@k evaluations, including scenarios where base models fail entirely regardless of the number of attempts. We further show that reasoning boundary improvements correlates strongly with task competence of base model and training duration, suggesting that RL can explore and populate new regions of solution space over time. These findings offer new insights into the conditions under which RL meaningfully expands reasoning boundaries in language models and establish a foundation for future work on long-horizon RL for reasoning. We release model weights to support further research: this https URL 

---
# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation 

**Authors**: Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.24853)  

**Abstract**: We study the problem of functional retargeting: learning dexterous manipulation policies to track object states from human hand-object demonstrations. We focus on long-horizon, bimanual tasks with articulated objects, which is challenging due to large action space, spatiotemporal discontinuities, and embodiment gap between human and robot hands. We propose DexMachina, a novel curriculum-based algorithm: the key idea is to use virtual object controllers with decaying strength: an object is first driven automatically towards its target states, such that the policy can gradually learn to take over under motion and contact guidance. We release a simulation benchmark with a diverse set of tasks and dexterous hands, and show that DexMachina significantly outperforms baseline methods. Our algorithm and benchmark enable a functional comparison for hardware designs, and we present key findings informed by quantitative and qualitative results. With the recent surge in dexterous hand development, we hope this work will provide a useful platform for identifying desirable hardware capabilities and lower the barrier for contributing to future research. Videos and more at this https URL 

---
# Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning 

**Authors**: Shuyao Xu, Cheng Peng, Jiangxuan Long, Weidi Xu, Wei Chu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24850)  

**Abstract**: Recent advances in model distillation demonstrate that data from advanced reasoning models (e.g., DeepSeek-R1, OpenAI's o1) can effectively transfer complex reasoning abilities to smaller, efficient student models. However, standard practices employ rejection sampling, discarding incorrect reasoning examples -- valuable, yet often underutilized data. This paper addresses the critical question: How can both positive and negative distilled reasoning traces be effectively leveraged to maximize LLM reasoning performance in an offline setting? To this end, We propose Reinforcement Distillation (REDI), a two-stage framework. Stage 1 learns from positive traces via Supervised Fine-Tuning (SFT). Stage 2 further refines the model using both positive and negative traces through our proposed REDI objective. This novel objective is a simple, reference-free loss function that outperforms established methods like DPO and SimPO in this distillation context. Our empirical evaluations demonstrate REDI's superiority over baseline Rejection Sampling SFT or SFT combined with DPO/SimPO on mathematical reasoning tasks. Notably, the Qwen-REDI-1.5B model, post-trained on just 131k positive and negative examples from the open Open-R1 dataset, achieves an 83.1% score on MATH-500 (pass@1). Its performance matches or surpasses that of DeepSeek-R1-Distill-Qwen-1.5B (a model post-trained on 800k proprietary data) across various mathematical reasoning benchmarks, establishing a new state-of-the-art for 1.5B models post-trained offline with openly available data. 

---
# Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck 

**Authors**: Yuwen Tan, Yuan Qing, Boqing Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.24840)  

**Abstract**: This paper reveals that many state-of-the-art large language models (LLMs) lack hierarchical knowledge about our visual world, unaware of even well-established biology taxonomies. This shortcoming makes LLMs a bottleneck for vision LLMs' hierarchical visual understanding (e.g., recognizing Anemone Fish but not Vertebrate). We arrive at these findings using about one million four-choice visual question answering (VQA) tasks constructed from six taxonomies and four image datasets. Interestingly, finetuning a vision LLM using our VQA tasks reaffirms LLMs' bottleneck effect to some extent because the VQA tasks improve the LLM's hierarchical consistency more than the vision LLM's. We conjecture that one cannot make vision LLMs understand visual concepts fully hierarchical until LLMs possess corresponding taxonomy knowledge. 

---
# VideoCAD: A Large-Scale Video Dataset for Learning UI Interactions and 3D Reasoning from CAD Software 

**Authors**: Brandon Man, Ghadi Nehme, Md Ferdous Alam, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.24838)  

**Abstract**: Computer-Aided Design (CAD) is a time-consuming and complex process, requiring precise, long-horizon user interactions with intricate 3D interfaces. While recent advances in AI-driven user interface (UI) agents show promise, most existing datasets and methods focus on short, low-complexity tasks in mobile or web applications, failing to capture the demands of professional engineering tools. In this work, we introduce VideoCAD, the first attempt at engineering UI interaction learning for precision tasks. Specifically, VideoCAD is a large-scale synthetic dataset consisting of over 41K annotated video recordings of CAD operations, generated using an automated framework for collecting high-fidelity UI action data from human-made CAD designs. Compared to existing datasets, VideoCAD offers an order of magnitude higher complexity in UI interaction learning for real-world engineering tasks, having up to a 20x longer time horizon than other datasets. We show two important downstream applications of VideoCAD: learning UI interactions from professional precision 3D CAD tools and a visual question-answering (VQA) benchmark designed to evaluate multimodal large language models' (LLM) spatial reasoning and video understanding abilities. To learn the UI interactions, we propose VideoCADFormer - a state-of-the-art model in learning CAD interactions directly from video, which outperforms multiple behavior cloning baselines. Both VideoCADFormer and the VQA benchmark derived from VideoCAD reveal key challenges in the current state of video-based UI understanding, including the need for precise action grounding, multi-modal and spatial reasoning, and long-horizon dependencies. 

---
# Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs 

**Authors**: Juraj Vladika, Annika Domres, Mai Nguyen, Rebecca Moser, Jana Nano, Felix Busch, Lisa C. Adams, Keno K. Bressem, Denise Bernhardt, Stephanie E. Combs, Kai J. Borm, Florian Matthes, Jan C. Peeken  

**Link**: [PDF](https://arxiv.org/pdf/2505.24830)  

**Abstract**: Large language models (LLMs) exhibit extensive medical knowledge but are prone to hallucinations and inaccurate citations, which pose a challenge to their clinical adoption and regulatory compliance. Current methods, such as Retrieval Augmented Generation, partially address these issues by grounding answers in source documents, but hallucinations and low fact-level explainability persist. In this work, we introduce a novel atomic fact-checking framework designed to enhance the reliability and explainability of LLMs used in medical long-form question answering. This method decomposes LLM-generated responses into discrete, verifiable units called atomic facts, each of which is independently verified against an authoritative knowledge base of medical guidelines. This approach enables targeted correction of errors and direct tracing to source literature, thereby improving the factual accuracy and explainability of medical Q&A. Extensive evaluation using multi-reader assessments by medical experts and an automated open Q&A benchmark demonstrated significant improvements in factual accuracy and explainability. Our framework achieved up to a 40% overall answer improvement and a 50% hallucination detection rate. The ability to trace each atomic fact back to the most relevant chunks from the database provides a granular, transparent explanation of the generated responses, addressing a major gap in current medical AI applications. This work represents a crucial step towards more trustworthy and reliable clinical applications of LLMs, addressing key prerequisites for clinical application and fostering greater confidence in AI-assisted healthcare. 

---
# PhySense: Principle-Based Physics Reasoning Benchmarking for Large Language Models 

**Authors**: Yinggan Xu, Yue Liu, Zhiqiang Gao, Changnan Peng, Di Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.24823)  

**Abstract**: Large language models (LLMs) have rapidly advanced and are increasingly capable of tackling complex scientific problems, including those in physics. Despite this progress, current LLMs often fail to emulate the concise, principle-based reasoning characteristic of human experts, instead generating lengthy and opaque solutions. This discrepancy highlights a crucial gap in their ability to apply core physical principles for efficient and interpretable problem solving. To systematically investigate this limitation, we introduce PhySense, a novel principle-based physics reasoning benchmark designed to be easily solvable by experts using guiding principles, yet deceptively difficult for LLMs without principle-first reasoning. Our evaluation across multiple state-of-the-art LLMs and prompt types reveals a consistent failure to align with expert-like reasoning paths, providing insights for developing AI systems with efficient, robust and interpretable principle-based scientific reasoning. 

---
# RealDrive: Retrieval-Augmented Driving with Diffusion Models 

**Authors**: Wenhao Ding, Sushant Veer, Yuxiao Chen, Yulong Cao, Chaowei Xiao, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.24808)  

**Abstract**: Learning-based planners generate natural human-like driving behaviors by learning to reason about nuanced interactions from data, overcoming the rigid behaviors that arise from rule-based planners. Nonetheless, data-driven approaches often struggle with rare, safety-critical scenarios and offer limited controllability over the generated trajectories. To address these challenges, we propose RealDrive, a Retrieval-Augmented Generation (RAG) framework that initializes a diffusion-based planning policy by retrieving the most relevant expert demonstrations from the training dataset. By interpolating between current observations and retrieved examples through a denoising process, our approach enables fine-grained control and safe behavior across diverse scenarios, leveraging the strong prior provided by the retrieved scenario. Another key insight we produce is that a task-relevant retrieval model trained with planning-based objectives results in superior planning performance in our framework compared to a task-agnostic retriever. Experimental results demonstrate improved generalization to long-tail events and enhanced trajectory diversity compared to standard learning-based planners -- we observe a 40% reduction in collision rate on the Waymo Open Motion dataset with RAG. 

---
# Inference Acceleration of Autoregressive Normalizing Flows by Selective Jacobi Decoding 

**Authors**: Jiaru Zhang, Juanwu Lu, Ziran Wang, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24791)  

**Abstract**: Normalizing flows are promising generative models with advantages such as theoretical rigor, analytical log-likelihood computation, and end-to-end training. However, the architectural constraints to ensure invertibility and tractable Jacobian computation limit their expressive power and practical usability. Recent advancements utilize autoregressive modeling, significantly enhancing expressive power and generation quality. However, such sequential modeling inherently restricts parallel computation during inference, leading to slow generation that impedes practical deployment. In this paper, we first identify that strict sequential dependency in inference is unnecessary to generate high-quality samples. We observe that patches in sequential modeling can also be approximated without strictly conditioning on all preceding patches. Moreover, the models tend to exhibit low dependency redundancy in the initial layer and higher redundancy in subsequent layers. Leveraging these observations, we propose a selective Jacobi decoding (SeJD) strategy that accelerates autoregressive inference through parallel iterative optimization. Theoretical analyses demonstrate the method's superlinear convergence rate and guarantee that the number of iterations required is no greater than the original sequential approach. Empirical evaluations across multiple datasets validate the generality and effectiveness of our acceleration technique. Experiments demonstrate substantial speed improvements up to 4.7 times faster inference while keeping the generation quality and fidelity. 

---
# Drop Dropout on Single-Epoch Language Model Pretraining 

**Authors**: Houjun Liu, John Bauer, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2505.24788)  

**Abstract**: Originally, dropout was seen as a breakthrough regularization technique that reduced overfitting and improved performance in almost all applications of deep learning by reducing overfitting. Yet, single-epoch pretraining tasks common to modern LLMs yield minimal overfitting, leading to dropout not being used for large LLMs. Nevertheless, no thorough empirical investigation has been done on the role of dropout in LM pretraining. Through experiments in single-epoch pretraining of both masked (BERT) and autoregressive (Pythia 160M and 1.4B) LMs with varying levels of dropout, we find that downstream performance in language modeling, morpho-syntax (BLiMP), question answering (SQuAD), and natural-language inference (MNLI) improves when dropout is not applied during pretraining. We additionally find that the recently-introduced "early dropout" also degrades performance over applying no dropout at all. We further investigate the models' editability, and find that models trained without dropout are more successful in gradient-based model editing (MEND) and equivalent in representation-based model editing (ReFT). Therefore, we advocate to drop dropout during single-epoch pretraining. 

---
# DiG-Net: Enhancing Quality of Life through Hyper-Range Dynamic Gesture Recognition in Assistive Robotics 

**Authors**: Eran Bamani Beeri, Eden Nissinman, Avishai Sintov  

**Link**: [PDF](https://arxiv.org/pdf/2505.24786)  

**Abstract**: Dynamic hand gestures play a pivotal role in assistive human-robot interaction (HRI), facilitating intuitive, non-verbal communication, particularly for individuals with mobility constraints or those operating robots remotely. Current gesture recognition methods are mostly limited to short-range interactions, reducing their utility in scenarios demanding robust assistive communication from afar. In this paper, we introduce a novel approach designed specifically for assistive robotics, enabling dynamic gesture recognition at extended distances of up to 30 meters, thereby significantly improving accessibility and quality of life. Our proposed Distance-aware Gesture Network (DiG-Net) effectively combines Depth-Conditioned Deformable Alignment (DADA) blocks with Spatio-Temporal Graph modules, enabling robust processing and classification of gesture sequences captured under challenging conditions, including significant physical attenuation, reduced resolution, and dynamic gesture variations commonly experienced in real-world assistive environments. We further introduce the Radiometric Spatio-Temporal Depth Attenuation Loss (RSTDAL), shown to enhance learning and strengthen model robustness across varying distances. Our model demonstrates significant performance improvement over state-of-the-art gesture recognition frameworks, achieving a recognition accuracy of 97.3% on a diverse dataset with challenging hyper-range gestures. By effectively interpreting gestures from considerable distances, DiG-Net significantly enhances the usability of assistive robots in home healthcare, industrial safety, and remote assistance scenarios, enabling seamless and intuitive interactions for users regardless of physical limitations 

---
# A survey of using EHR as real-world evidence for discovering and validating new drug indications 

**Authors**: Nabasmita Talukdar, Xiaodan Zhang, Shreya Paithankar, Hui Wang, Bin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24767)  

**Abstract**: Electronic Health Records (EHRs) have been increasingly used as real-world evidence (RWE) to support the discovery and validation of new drug indications. This paper surveys current approaches to EHR-based drug repurposing, covering data sources, processing methodologies, and representation techniques. It discusses study designs and statistical frameworks for evaluating drug efficacy. Key challenges in validation are discussed, with emphasis on the role of large language models (LLMs) and target trial emulation. By synthesizing recent developments and methodological advances, this work provides a foundational resource for researchers aiming to translate real-world data into actionable drug-repurposing evidence. 

---
# Supervised Quantum Machine Learning: A Future Outlook from Qubits to Enterprise Applications 

**Authors**: Srikanth Thudumu, Jason Fisher, Hung Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.24765)  

**Abstract**: Supervised Quantum Machine Learning (QML) represents an intersection of quantum computing and classical machine learning, aiming to use quantum resources to support model training and inference. This paper reviews recent developments in supervised QML, focusing on methods such as variational quantum circuits, quantum neural networks, and quantum kernel methods, along with hybrid quantum-classical workflows. We examine recent experimental studies that show partial indications of quantum advantage and describe current limitations including noise, barren plateaus, scalability issues, and the lack of formal proofs of performance improvement over classical methods. The main contribution is a ten-year outlook (2025-2035) that outlines possible developments in supervised QML, including a roadmap describing conditions under which QML may be used in applied research and enterprise systems over the next decade. 

---
# REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards 

**Authors**: Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, Andreas Köpf  

**Link**: [PDF](https://arxiv.org/pdf/2505.24760)  

**Abstract**: We introduce Reasoning Gym (RG), a library of reasoning environments for reinforcement learning with verifiable rewards. It provides over 100 data generators and verifiers spanning multiple domains including algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and various common games. Its key innovation is the ability to generate virtually infinite training data with adjustable complexity, unlike most previous reasoning datasets, which are typically fixed. This procedural generation approach allows for continuous evaluation across varying difficulty levels. Our experimental results demonstrate the efficacy of RG in both evaluating and reinforcement learning of reasoning models. 

---
# Unsupervised Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport 

**Authors**: Mu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24759)  

**Abstract**: Identifying evolutionary correspondences between cell types across species is a fundamental challenge in comparative genomics and evolutionary biology. Existing approaches often rely on either reference-based matching, which imposes asymmetry by designating one species as the reference, or projection-based matching, which may increase computational complexity and obscure biological interpretability at the cell-type level. Here, we present OT-MESH, an unsupervised computational framework leveraging entropy-regularized optimal transport (OT) to systematically determine cross-species cell type homologies. Our method uniquely integrates the Minimize Entropy of Sinkhorn (MESH) technique to refine the OT plan. It begins by selecting genes with high Signal-to-Noise Ratio (SNR) to capture the most informative features, from which a cost matrix is constructed using cosine distances between cell-type centroids. Importantly, the MESH procedure iteratively refines the cost matrix, leading to a transport plan with significantly enhanced sparsity and interpretability of the resulting correspondence matrices. Applied to retinal bipolar cells (BCs) and retinal ganglion cells (RGCs) from mouse and macaque, OT-MESH accurately recovers known evolutionary relationships and uncovers novel correspondences, one of which was independently validated experimentally. Thus, our framework offers a principled, scalable, symmetric, and interpretable solution for evolutionary cell type mapping, facilitating deeper insights into cellular specialization and conservation across species. 

---
# Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation 

**Authors**: Yingchaojie Feng, Yiqun Sun, Yandong Sun, Minfeng Zhu, Qiang Huang, Anthony K. H. Tung, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24754)  

**Abstract**: In this work, we investigate an important task named instruction-following text embedding, which generates dynamic text embeddings that adapt to user instructions, highlighting specific attributes of text. Despite recent advancements, existing approaches suffer from significant computational overhead, as they require re-encoding the entire corpus for each new instruction. To address this challenge, we propose GSTransform, a novel instruction-following text embedding framework based on Guided Space Transformation. Our key observation is that instruction-relevant information is inherently encoded in generic embeddings but remains underutilized. Instead of repeatedly encoding the corpus for each instruction, GSTransform is a lightweight transformation mechanism that adapts pre-computed embeddings in real time to align with user instructions, guided by a small amount of text data with instruction-focused label annotation. We conduct extensive experiments on three instruction-awareness downstream tasks across nine real-world datasets, demonstrating that GSTransform improves instruction-following text embedding quality over state-of-the-art methods while achieving dramatic speedups of 6~300x in real-time processing on large-scale datasets. The source code is available at this https URL. 

---
# HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts 

**Authors**: Neil He, Rishabh Anand, Hiren Madhu, Ali Maatouk, Smita Krishnaswamy, Leandros Tassiulas, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.24722)  

**Abstract**: Large language models (LLMs) have shown great success in text modeling tasks across domains. However, natural language exhibits inherent semantic hierarchies and nuanced geometric structure, which current LLMs do not capture completely owing to their reliance on Euclidean operations. Recent studies have also shown that not respecting the geometry of token embeddings leads to training instabilities and degradation of generative capabilities. These findings suggest that shifting to non-Euclidean geometries can better align language models with the underlying geometry of text. We thus propose to operate fully in Hyperbolic space, known for its expansive, scale-free, and low-distortion properties. We thus introduce HELM, a family of HypErbolic Large Language Models, offering a geometric rethinking of the Transformer-based LLM that addresses the representational inflexibility, missing set of necessary operations, and poor scalability of existing hyperbolic LMs. We additionally introduce a Mixture-of-Curvature Experts model, HELM-MICE, where each expert operates in a distinct curvature space to encode more fine-grained geometric structure from text, as well as a dense model, HELM-D. For HELM-MICE, we further develop hyperbolic Multi-Head Latent Attention (HMLA) for efficient, reduced-KV-cache training and inference. For both models, we develop essential hyperbolic equivalents of rotary positional encodings and RMS normalization. We are the first to train fully hyperbolic LLMs at billion-parameter scale, and evaluate them on well-known benchmarks such as MMLU and ARC, spanning STEM problem-solving, general knowledge, and commonsense reasoning. Our results show consistent gains from our HELM architectures -- up to 4% -- over popular Euclidean architectures used in LLaMA and DeepSeek, highlighting the efficacy and enhanced reasoning afforded by hyperbolic geometry in large-scale LM pretraining. 

---
# Towards Scalable Schema Mapping using Large Language Models 

**Authors**: Christopher Buss, Mahdis Safari, Arash Termehchy, Stefan Lee, David Maier  

**Link**: [PDF](https://arxiv.org/pdf/2505.24716)  

**Abstract**: The growing need to integrate information from a large number of diverse sources poses significant scalability challenges for data integration systems. These systems often rely on manually written schema mappings, which are complex, source-specific, and costly to maintain as sources evolve. While recent advances suggest that large language models (LLMs) can assist in automating schema matching by leveraging both structural and natural language cues, key challenges remain. In this paper, we identify three core issues with using LLMs for schema mapping: (1) inconsistent outputs due to sensitivity to input phrasing and structure, which we propose methods to address through sampling and aggregation techniques; (2) the need for more expressive mappings (e.g., GLaV), which strain the limited context windows of LLMs; and (3) the computational cost of repeated LLM calls, which we propose to mitigate through strategies like data type prefiltering. 

---
# CoRet: Improved Retriever for Code Editing 

**Authors**: Fabio Fehr, Prabhu Teja Sivaprasad, Luca Franceschi, Giovanni Zappella  

**Link**: [PDF](https://arxiv.org/pdf/2505.24715)  

**Abstract**: In this paper, we introduce CoRet, a dense retrieval model designed for code-editing tasks that integrates code semantics, repository structure, and call graph dependencies. The model focuses on retrieving relevant portions of a code repository based on natural language queries such as requests to implement new features or fix bugs. These retrieved code chunks can then be presented to a user or to a second code-editing model or agent. To train CoRet, we propose a loss function explicitly designed for repository-level retrieval. On SWE-bench and Long Code Arena's bug localisation datasets, we show that our model substantially improves retrieval recall by at least 15 percentage points over existing models, and ablate the design choices to show their importance in achieving these results. 

---
# Causal-aware Large Language Models: Enhancing Decision-Making Through Learning, Adapting and Acting 

**Authors**: Wei Chen, Jiahao Zhang, Haipeng Zhu, Boyan Xu, Zhifeng Hao, Keli Zhang, Junjian Ye, Ruichu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.24710)  

**Abstract**: Large language models (LLMs) have shown great potential in decision-making due to the vast amount of knowledge stored within the models. However, these pre-trained models are prone to lack reasoning abilities and are difficult to adapt to new environments, further hindering their application to complex real-world tasks. To address these challenges, inspired by the human cognitive process, we propose Causal-aware LLMs, which integrate the structural causal model (SCM) into the decision-making process to model, update, and utilize structured knowledge of the environment in a ``learning-adapting-acting" paradigm. Specifically, in the learning stage, we first utilize an LLM to extract the environment-specific causal entities and their causal relations to initialize a structured causal model of the environment. Subsequently,in the adapting stage, we update the structured causal model through external feedback about the environment, via an idea of causal intervention. Finally, in the acting stage, Causal-aware LLMs exploit structured causal knowledge for more efficient policy-making through the reinforcement learning agent. The above processes are performed iteratively to learn causal knowledge, ultimately enabling the causal-aware LLMs to achieve a more accurate understanding of the environment and make more efficient decisions. Experimental results across 22 diverse tasks within the open-world game ``Crafter" validate the effectiveness of our proposed method. 

---
# On Symmetric Losses for Robust Policy Optimization with Noisy Preferences 

**Authors**: Soichiro Nishimori, Yu-Jie Zhang, Thanawat Lodkaew, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.24709)  

**Abstract**: Optimizing policies based on human preferences is key to aligning language models with human intent. This work focuses on reward modeling, a core component in reinforcement learning from human feedback (RLHF), and offline preference optimization, such as direct preference optimization. Conventional approaches typically assume accurate annotations. However, real-world preference data often contains noise due to human errors or biases. We propose a principled framework for robust policy optimization under noisy preferences, viewing reward modeling as a classification problem. This allows us to leverage symmetric losses, known for their robustness to label noise in classification, leading to our Symmetric Preference Optimization (SymPO) method. We prove that symmetric losses enable successful policy optimization even under noisy labels, as the resulting reward remains rank-preserving -- a property sufficient for policy improvement. Experiments on synthetic and real-world tasks demonstrate the effectiveness of SymPO. 

---
# Multi-Domain ABSA Conversation Dataset Generation via LLMs for Real-World Evaluation and Model Comparison 

**Authors**: Tejul Pandit, Meet Raval, Dhvani Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2505.24701)  

**Abstract**: Aspect-Based Sentiment Analysis (ABSA) offers granular insights into opinions but often suffers from the scarcity of diverse, labeled datasets that reflect real-world conversational nuances. This paper presents an approach for generating synthetic ABSA data using Large Language Models (LLMs) to address this gap. We detail the generation process aimed at producing data with consistent topic and sentiment distributions across multiple domains using GPT-4o. The quality and utility of the generated data were evaluated by assessing the performance of three state-of-the-art LLMs (Gemini 1.5 Pro, Claude 3.5 Sonnet, and DeepSeek-R1) on topic and sentiment classification tasks. Our results demonstrate the effectiveness of the synthetic data, revealing distinct performance trade-offs among the models: DeepSeekR1 showed higher precision, Gemini 1.5 Pro and Claude 3.5 Sonnet exhibited strong recall, and Gemini 1.5 Pro offered significantly faster inference. We conclude that LLM-based synthetic data generation is a viable and flexible method for creating valuable ABSA resources, facilitating research and model evaluation without reliance on limited or inaccessible real-world labeled data. 

---
# Disentangling Granularity: An Implicit Inductive Bias in Factorized VAEs 

**Authors**: Zihao Chen, Yu Xiang, Wenyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24684)  

**Abstract**: Despite the success in learning semantically meaningful, unsupervised disentangled representations, variational autoencoders (VAEs) and their variants face a fundamental theoretical challenge: substantial evidence indicates that unsupervised disentanglement is unattainable without implicit inductive bias, yet such bias remains elusive. In this work, we focus on exploring the implicit inductive bias that drive disentanglement in VAEs with factorization priors. By analyzing the total correlation in \b{eta}-TCVAE, we uncover a crucial implicit inductive bias called disentangling granularity, which leads to the discovery of an interesting "V"-shaped optimal Evidence Lower Bound (ELBO) trajectory within the parameter space. This finding is validated through over 100K experiments using factorized VAEs and our newly proposed model, \b{eta}-STCVAE. Notably, experimental results reveal that conventional factorized VAEs, constrained by fixed disentangling granularity, inherently tend to disentangle low-complexity feature. Whereas, appropriately tuning disentangling granularity, as enabled by \b{eta}-STCVAE, broadens the range of disentangled representations, allowing for the disentanglement of high-complexity features. Our findings unveil that disentangling granularity as an implicit inductive bias in factorized VAEs influence both disentanglement performance and the inference of the ELBO, offering fresh insights into the interpretability and inherent biases of VAEs. 

---
# Should I Share this Translation? Evaluating Quality Feedback for User Reliance on Machine Translation 

**Authors**: Dayeon Ki, Kevin Duh, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.24683)  

**Abstract**: As people increasingly use AI systems in work and daily life, feedback mechanisms that help them use AI responsibly are urgently needed, particularly in settings where users are not equipped to assess the quality of AI predictions. We study a realistic Machine Translation (MT) scenario where monolingual users decide whether to share an MT output, first without and then with quality feedback. We compare four types of quality feedback: explicit feedback that directly give users an assessment of translation quality using 1) error highlights and 2) LLM explanations, and implicit feedback that helps users compare MT inputs and outputs through 3) backtranslation and 4) question-answer (QA) tables. We find that all feedback types, except error highlights, significantly improve both decision accuracy and appropriate reliance. Notably, implicit feedback, especially QA tables, yields significantly greater gains than explicit feedback in terms of decision accuracy, appropriate reliance, and user perceptions, receiving the highest ratings for helpfulness and trust, and the lowest for mental burden. 

---
# Generative Knowledge Production Pipeline Driven by Academic Influencers 

**Authors**: Katalin Feher, Marton Demeter  

**Link**: [PDF](https://arxiv.org/pdf/2505.24681)  

**Abstract**: Generative AI transforms knowledge production, validation, and dissemination, raising academic integrity and credibility concerns. This study examines 53 academic influencer videos that reached 5.3 million viewers to identify an emerging, structured, implementation-ready pipeline balancing originality, ethical compliance, and human-AI collaboration despite the disruptive impacts. Findings highlight generative AI's potential to automate publication workflows and democratize participation in knowledge production while challenging traditional scientific norms. Academic influencers emerge as key intermediaries in this paradigm shift, connecting bottom-up practices with institutional policies to improve adaptability. Accordingly, the study proposes a generative publication production pipeline and a policy framework for co-intelligence adaptation and reinforcing credibility-centered standards in AI-powered research. These insights support scholars, educators, and policymakers in understanding AI's transformative impact by advocating responsible and innovation-driven knowledge production. Additionally, they reveal pathways for automating best practices, optimizing scholarly workflows, and fostering creativity in academic research and publication. 

---
# Multiple LLM Agents Debate for Equitable Cultural Alignment 

**Authors**: Dayeon Ki, Rachel Rudinger, Tianyi Zhou, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.24671)  

**Abstract**: Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters). 

---
# BIMA: Bijective Maximum Likelihood Learning Approach to Hallucination Prediction and Mitigation in Large Vision-Language Models 

**Authors**: Huu-Thien Tran, Thanh-Dat Truong, Khoa Luu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24649)  

**Abstract**: Large vision-language models have become widely adopted to advance in various domains. However, developing a trustworthy system with minimal interpretable characteristics of large-scale models presents a significant challenge. One of the most prevalent terms associated with the fallacy functions caused by these systems is hallucination, where the language model generates a response that does not correspond to the visual content. To mitigate this problem, several approaches have been developed, and one prominent direction is to ameliorate the decoding process. In this paper, we propose a new Bijective Maximum Likelihood Learning (BIMA) approach to hallucination mitigation using normalizing flow theories. The proposed BIMA method can efficiently mitigate the hallucination problem in prevailing vision-language models, resulting in significant improvements. Notably, BIMA achieves the average F1 score of 85.06% on POPE benchmark and remarkably reduce CHAIRS and CHAIRI by 7.6% and 2.6%, respectively. To the best of our knowledge, this is one of the first studies that contemplates the bijection means to reduce hallucination induced by large vision-language models. 

---
# Efficient Text Encoders for Labor Market Analysis 

**Authors**: Jens-Joris Decorte, Jeroen Van Hautte, Chris Develder, Thomas Demeester  

**Link**: [PDF](https://arxiv.org/pdf/2505.24640)  

**Abstract**: Labor market analysis relies on extracting insights from job advertisements, which provide valuable yet unstructured information on job titles and corresponding skill requirements. While state-of-the-art methods for skill extraction achieve strong performance, they depend on large language models (LLMs), which are computationally expensive and slow. In this paper, we propose \textbf{ConTeXT-match}, a novel contrastive learning approach with token-level attention that is well-suited for the extreme multi-label classification task of skill classification. \textbf{ConTeXT-match} significantly improves skill extraction efficiency and performance, achieving state-of-the-art results with a lightweight bi-encoder model. To support robust evaluation, we introduce \textbf{Skill-XL}, a new benchmark with exhaustive, sentence-level skill annotations that explicitly address the redundancy in the large label space. Finally, we present \textbf{JobBERT V2}, an improved job title normalization model that leverages extracted skills to produce high-quality job title representations. Experiments demonstrate that our models are efficient, accurate, and scalable, making them ideal for large-scale, real-time labor market analysis. 

---
# Cloud Optical Thickness Retrievals Using Angle Invariant Attention Based Deep Learning Models 

**Authors**: Zahid Hassan Tushar, Adeleke Ademakinwa, Jianwu Wang, Zhibo Zhang, Sanjay Purushotham  

**Link**: [PDF](https://arxiv.org/pdf/2505.24638)  

**Abstract**: Cloud Optical Thickness (COT) is a critical cloud property influencing Earth's climate, weather, and radiation budget. Satellite radiance measurements enable global COT retrieval, but challenges like 3D cloud effects, viewing angles, and atmospheric interference must be addressed to ensure accurate estimation. Traditionally, the Independent Pixel Approximation (IPA) method, which treats individual pixels independently, has been used for COT estimation. However, IPA introduces significant bias due to its simplified assumptions. Recently, deep learning-based models have shown improved performance over IPA but lack robustness, as they are sensitive to variations in radiance intensity, distortions, and cloud shadows. These models also introduce substantial errors in COT estimation under different solar and viewing zenith angles. To address these challenges, we propose a novel angle-invariant, attention-based deep model called Cloud-Attention-Net with Angle Coding (CAAC). Our model leverages attention mechanisms and angle embeddings to account for satellite viewing geometry and 3D radiative transfer effects, enabling more accurate retrieval of COT. Additionally, our multi-angle training strategy ensures angle invariance. Through comprehensive experiments, we demonstrate that CAAC significantly outperforms existing state-of-the-art deep learning models, reducing cloud property retrieval errors by at least a factor of nine. 

---
# The Hallucination Dilemma: Factuality-Aware Reinforcement Learning for Large Reasoning Models 

**Authors**: Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2505.24630)  

**Abstract**: Large language models (LLMs) have significantly advanced in reasoning tasks through reinforcement learning (RL) optimization, achieving impressive capabilities across various challenging benchmarks. However, our empirical analysis reveals a critical drawback: reasoning-oriented RL fine-tuning significantly increases the prevalence of hallucinations. We theoretically analyze the RL training dynamics, identifying high-variance gradient, entropy-induced randomness, and susceptibility to spurious local optima as key factors leading to hallucinations. To address this drawback, we propose Factuality-aware Step-wise Policy Optimization (FSPO), an innovative RL fine-tuning algorithm incorporating explicit factuality verification at each reasoning step. FSPO leverages automated verification against given evidence to dynamically adjust token-level advantage values, incentivizing factual correctness throughout the reasoning process. Experiments across mathematical reasoning and hallucination benchmarks using Qwen2.5 and Llama models demonstrate that FSPO effectively reduces hallucinations while enhancing reasoning accuracy, substantially improving both reliability and performance. 

---
# Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors 

**Authors**: Duo Zheng, Shijia Huang, Yanyang Li, Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24625)  

**Abstract**: Previous research has investigated the application of Multimodal Large Language Models (MLLMs) in understanding 3D scenes by interpreting them as videos. These approaches generally depend on comprehensive 3D data inputs, such as point clouds or reconstructed Bird's-Eye View (BEV) maps. In our research, we advance this field by enhancing the capability of MLLMs to understand and reason in 3D spaces directly from video data, without the need for additional 3D input. We propose a novel and efficient method, the Video-3D Geometry Large Language Model (VG LLM). Our approach employs a 3D visual geometry encoder that extracts 3D prior information from video sequences. This information is integrated with visual tokens and fed into the MLLM. Extensive experiments have shown that our method has achieved substantial improvements in various tasks related to 3D scene understanding and spatial reasoning, all directly learned from video sources. Impressively, our 4B model, which does not rely on explicit 3D data inputs, achieves competitive results compared to existing state-of-the-art methods, and even surpasses the Gemini-1.5-Pro in the VSI-Bench evaluations. 

---
# Hyperbolic Dataset Distillation 

**Authors**: Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.24623)  

**Abstract**: To address the computational and storage challenges posed by large-scale datasets in deep learning, dataset distillation has been proposed to synthesize a compact dataset that replaces the original while maintaining comparable model performance. Unlike optimization-based approaches that require costly bi-level optimization, distribution matching (DM) methods improve efficiency by aligning the distributions of synthetic and original data, thereby eliminating nested optimization. DM achieves high computational efficiency and has emerged as a promising solution. However, existing DM methods, constrained to Euclidean space, treat data as independent and identically distributed points, overlooking complex geometric and hierarchical relationships. To overcome this limitation, we propose a novel hyperbolic dataset distillation method, termed HDD. Hyperbolic space, characterized by negative curvature and exponential volume growth with distance, naturally models hierarchical and tree-like structures. HDD embeds features extracted by a shallow network into the Lorentz hyperbolic space, where the discrepancy between synthetic and original data is measured by the hyperbolic (geodesic) distance between their centroids. By optimizing this distance, the hierarchical structure is explicitly integrated into the distillation process, guiding synthetic samples to gravitate towards the root-centric regions of the original data distribution while preserving their underlying geometric characteristics. Furthermore, we find that pruning in hyperbolic space requires only 20% of the distilled core set to retain model performance, while significantly improving training stability. Notably, HDD is seamlessly compatible with most existing DM methods, and extensive experiments on different datasets validate its effectiveness. 

---
# Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX 

**Authors**: Nikita Martynov, Anastasia Mordasheva, Dmitriy Gorbetskiy, Danil Astafurov, Ulyana Isaeva, Elina Basyrova, Sergey Skachkov, Victoria Berestova, Nikolay Ivanov, Valeriia Zanina, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2505.24616)  

**Abstract**: We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments. 

---
# Binary Cumulative Encoding meets Time Series Forecasting 

**Authors**: Andrei Chernov, Vitaliy Pozdnyakov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.24595)  

**Abstract**: Recent studies in time series forecasting have explored formulating regression via classification task. By discretizing the continuous target space into bins and predicting over a fixed set of classes, these approaches benefit from stable training, robust uncertainty modeling, and compatibility with modern deep learning architectures. However, most existing methods rely on one-hot encoding that ignores the inherent ordinal structure of the underlying values. As a result, they fail to provide information about the relative distance between predicted and true values during training. In this paper, we propose to address this limitation by introducing binary cumulative encoding (BCE), that represents scalar targets into monotonic binary vectors. This encoding implicitly preserves order and magnitude information, allowing the model to learn distance-aware representations while still operating within a classification framework. We propose a convolutional neural network architecture specifically designed for BCE, incorporating residual and dilated convolutions to enable fast and expressive temporal modeling. Through extensive experiments on benchmark forecasting datasets, we show that our approach outperforms widely used methods in both point and probabilistic forecasting, while requiring fewer parameters and enabling faster training. 

---
# Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis 

**Authors**: Junzhuo Li, Bo Wang, Xiuze Zhou, Peijie Jiang, Jia Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24593)  

**Abstract**: The interpretability of Mixture-of-Experts (MoE) models, especially those with heterogeneous designs, remains underexplored. Existing attribution methods for dense models fail to capture dynamic routing-expert interactions in sparse MoE architectures. To address this issue, we propose a cross-level attribution algorithm to analyze sparse MoE architectures (Qwen 1.5-MoE, OLMoE, Mixtral-8x7B) against dense models (Qwen 1.5-7B, Llama-7B, Mixtral-7B). Results show MoE models achieve 37% higher per-layer efficiency via a "mid-activation, late-amplification" pattern: early layers screen experts, while late layers refine knowledge collaboratively. Ablation studies reveal a "basic-refinement" framework--shared experts handle general tasks (entity recognition), while routed experts specialize in domain-specific processing (geographic attributes). Semantic-driven routing is evidenced by strong correlations between attention heads and experts (r=0.68), enabling task-aware coordination. Notably, architectural depth dictates robustness: deep Qwen 1.5-MoE mitigates expert failures (e.g., 43% MRR drop in geographic tasks when blocking top-10 experts) through shared expert redundancy, whereas shallow OLMoE suffers severe degradation (76% drop). Task sensitivity further guides design: core-sensitive tasks (geography) require concentrated expertise, while distributed-tolerant tasks (object attributes) leverage broader participation. These insights advance MoE interpretability, offering principles to balance efficiency, specialization, and robustness. 

---
# A Flat Minima Perspective on Understanding Augmentations and Model Robustness 

**Authors**: Weebum Yoo, Sung Whan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2505.24592)  

**Abstract**: Model robustness indicates a model's capability to generalize well on unforeseen distributional shifts, including data corruption, adversarial attacks, and domain shifts. Data augmentation is one of the prevalent and effective ways to enhance robustness. Despite the great success of augmentations in different fields, a general theoretical understanding of their efficacy in improving model robustness is lacking. We offer a unified theoretical framework to clarify how augmentations can enhance model robustness through the lens of loss surface flatness and PAC generalization bound. Our work diverges from prior studies in that our analysis i) broadly encompasses much of the existing augmentation methods, and ii) is not limited to specific types of distribution shifts like adversarial attacks. We confirm our theories through simulations on the existing common corruption and adversarial robustness benchmarks based on the CIFAR and ImageNet datasets, as well as domain generalization benchmarks including PACS and OfficeHome. 

---
# AutoChemSchematic AI: A Closed-Loop, Physics-Aware Agentic Framework for Auto-Generating Chemical Process and Instrumentation Diagrams 

**Authors**: Sakhinana Sagar Srinivas, Shivam Gupta, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2505.24584)  

**Abstract**: Recent advancements in generative AI have accelerated the discovery of novel chemicals and materials; however, transitioning these discoveries to industrial-scale production remains a critical bottleneck, as it requires the development of entirely new chemical manufacturing processes. Current AI methods cannot auto-generate PFDs or PIDs, despite their critical role in scaling chemical processes, while adhering to engineering constraints. We present a closed loop, physics aware framework for the automated generation of industrially viable PFDs and PIDs. The framework integrates domain specialized small scale language models (SLMs) (trained for chemical process QA tasks) with first principles simulation, leveraging three key components: (1) a hierarchical knowledge graph of process flow and instrumentation descriptions for 1,020+ chemicals, (2) a multi-stage training pipeline that fine tunes domain specialized SLMs on synthetic datasets via Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Retrieval-Augmented Instruction Tuning (RAIT), and (3) DWSIM based simulator in the loop validation to ensure feasibility. To improve both runtime efficiency and model compactness, the framework incorporates advanced inference time optimizations including FlashAttention, Lookahead Decoding, PagedAttention with KV-cache quantization, and Test Time Inference Scaling and independently applies structural pruning techniques (width and depth) guided by importance heuristics to reduce model size with minimal accuracy loss. Experiments demonstrate that the framework generates simulator-validated process descriptions with high fidelity, outperforms baseline methods in correctness, and generalizes to unseen chemicals. By bridging AI-driven design with industrial-scale feasibility, this work significantly reduces R&D timelines from lab discovery to plant deployment. 

---
# NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization 

**Authors**: Hyuntak Kim, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24575)  

**Abstract**: Summarizing long-form narratives--such as books, movies, and TV scripts--requires capturing intricate plotlines, character interactions, and thematic coherence, a task that remains challenging for existing LLMs. We introduce NexusSum, a multi-agent LLM framework for narrative summarization that processes long-form text through a structured, sequential pipeline--without requiring fine-tuning. Our approach introduces two key innovations: (1) Dialogue-to-Description Transformation: A narrative-specific preprocessing method that standardizes character dialogue and descriptive text into a unified format, improving coherence. (2) Hierarchical Multi-LLM Summarization: A structured summarization pipeline that optimizes chunk processing and controls output length for accurate, high-quality summaries. Our method establishes a new state-of-the-art in narrative summarization, achieving up to a 30.0% improvement in BERTScore (F1) across books, movies, and TV scripts. These results demonstrate the effectiveness of multi-agent LLMs in handling long-form content, offering a scalable approach for structured summarization in diverse storytelling domains. 

---
# Bench4KE: Benchmarking Automated Competency Question Generation 

**Authors**: Anna Sofia Lippolis, Minh Davide Ragagni, Paolo Ciancarini, Andrea Giovanni Nuzzolese, Valentina Presutti  

**Link**: [PDF](https://arxiv.org/pdf/2505.24554)  

**Abstract**: The availability of Large Language Models (LLMs) presents a unique opportunity to reinvigorate research on Knowledge Engineering (KE) automation, a trend already evident in recent efforts developing LLM-based methods and tools for the automatic generation of Competency Questions (CQs). However, the evaluation of these tools lacks standardisation. This undermines the methodological rigour and hinders the replication and comparison of results. To address this gap, we introduce Bench4KE, an extensible API-based benchmarking system for KE automation. Its first release focuses on evaluating tools that generate CQs automatically. CQs are natural language questions used by ontology engineers to define the functional requirements of an ontology. Bench4KE provides a curated gold standard consisting of CQ datasets from four real-world ontology projects. It uses a suite of similarity metrics to assess the quality of the CQs generated. We present a comparative analysis of four recent CQ generation systems, which are based on LLMs, establishing a baseline for future research. Bench4KE is also designed to accommodate additional KE automation tasks, such as SPARQL query generation, ontology testing and drafting. Code and datasets are publicly available under the Apache 2.0 license. 

---
# CREFT: Sequential Multi-Agent LLM for Character Relation Extraction 

**Authors**: Ye Eun Chun, Taeyoon Hwang, Seung-won Hwang, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24553)  

**Abstract**: Understanding complex character relations is crucial for narrative analysis and efficient script evaluation, yet existing extraction methods often fail to handle long-form narratives with nuanced interactions. To address this challenge, we present CREFT, a novel sequential framework leveraging specialized Large Language Model (LLM) agents. First, CREFT builds a base character graph through knowledge distillation, then iteratively refines character composition, relation extraction, role identification, and group assignments. Experiments on a curated Korean drama dataset demonstrate that CREFT significantly outperforms single-agent LLM baselines in both accuracy and completeness. By systematically visualizing character networks, CREFT streamlines narrative comprehension and accelerates script review -- offering substantial benefits to the entertainment, publishing, and educational sectors. 

---
# Cross-Attention Speculative Decoding 

**Authors**: Wei Zhong, Manasa Bharadwaj, Yixiao Wang, Nikhil Verma, Yipeng Ji, Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.24544)  

**Abstract**: Speculative decoding (SD) is a widely adopted approach for accelerating inference in large language models (LLMs), particularly when the draft and target models are well aligned. However, state-of-the-art SD methods typically rely on tightly coupled, self-attention-based Transformer decoders, often augmented with auxiliary pooling or fusion layers. This coupling makes them increasingly complex and harder to generalize across different models. We present Budget EAGLE (Beagle), the first, to our knowledge, cross-attention-based Transformer decoder SD model that achieves performance on par with leading self-attention SD models (EAGLE-v2) while eliminating the need for pooling or auxiliary components, simplifying the architecture, improving training efficiency, and maintaining stable memory usage during training-time simulation. To enable effective training of this novel architecture, we propose Two-Stage Block-Attention Training, a new method that achieves training stability and convergence efficiency in block-level attention scenarios. Extensive experiments across multiple LLMs and datasets show that Beagle achieves competitive inference speedups and higher training efficiency than EAGLE-v2, offering a strong alternative for architectures in speculative decoding. 

---
# Mixpert: Mitigating Multimodal Learning Conflicts with Efficient Mixture-of-Vision-Experts 

**Authors**: Xin He, Xumeng Han, Longhui Wei, Lingxi Xie, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.24541)  

**Abstract**: Multimodal large language models (MLLMs) require a nuanced interpretation of complex image information, typically leveraging a vision encoder to perceive various visual scenarios. However, relying solely on a single vision encoder to handle diverse task domains proves difficult and inevitably leads to conflicts. Recent work enhances data perception by directly integrating multiple domain-specific vision encoders, yet this structure adds complexity and limits the potential for joint optimization. In this paper, we introduce Mixpert, an efficient mixture-of-vision-experts architecture that inherits the joint learning advantages from a single vision encoder while being restructured into a multi-expert paradigm for task-specific fine-tuning across different visual tasks. Additionally, we design a dynamic routing mechanism that allocates input images to the most suitable visual expert. Mixpert effectively alleviates domain conflicts encountered by a single vision encoder in multi-task learning with minimal additional computational cost, making it more efficient than multiple encoders. Furthermore, Mixpert integrates seamlessly into any MLLM, with experimental results demonstrating substantial performance gains across various tasks. 

---
# Localizing Persona Representations in LLMs 

**Authors**: Celia Cintas, Miriam Rateike, Erik Miehling, Elizabeth Daly, Skyler Speakman  

**Link**: [PDF](https://arxiv.org/pdf/2505.24539)  

**Abstract**: We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements. 

---
# CHIP: Chameleon Hash-based Irreversible Passport for Robust Deep Model Ownership Verification and Active Usage Control 

**Authors**: Chaohui Xu, Qi Cui, Chip-Hong Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24536)  

**Abstract**: The pervasion of large-scale Deep Neural Networks (DNNs) and their enormous training costs make their intellectual property (IP) protection of paramount importance. Recently introduced passport-based methods attempt to steer DNN watermarking towards strengthening ownership verification against ambiguity attacks by modulating the affine parameters of normalization layers. Unfortunately, neither watermarking nor passport-based methods provide a holistic protection with robust ownership proof, high fidelity, active usage authorization and user traceability for offline access distributed models and multi-user Machine-Learning as a Service (MLaaS) cloud model. In this paper, we propose a Chameleon Hash-based Irreversible Passport (CHIP) protection framework that utilizes the cryptographic chameleon hash function to achieve all these goals. The collision-resistant property of chameleon hash allows for strong model ownership claim upon IP infringement and liable user traceability, while the trapdoor-collision property enables hashing of multiple user passports and licensee certificates to the same immutable signature to realize active usage control. Using the owner passport as an oracle, multiple user-specific triplets, each contains a passport-aware user model, a user passport, and a licensee certificate can be created for secure offline distribution. The watermarked master model can also be deployed for MLaaS with usage permission verifiable by the provision of any trapdoor-colliding user passports. CHIP is extensively evaluated on four datasets and two architectures to demonstrate its protection versatility and robustness. Our code is released at this https URL. 

---
# Beyond Linear Steering: Unified Multi-Attribute Control for Language Models 

**Authors**: Narmeen Oozeer, Luke Marks, Fazl Barez, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2505.24535)  

**Abstract**: Controlling multiple behavioral attributes in large language models (LLMs) at inference time is a challenging problem due to interference between attributes and the limitations of linear steering methods, which assume additive behavior in activation space and require per-attribute tuning. We introduce K-Steering, a unified and flexible approach that trains a single non-linear multi-label classifier on hidden activations and computes intervention directions via gradients at inference time. This avoids linearity assumptions, removes the need for storing and tuning separate attribute vectors, and allows dynamic composition of behaviors without retraining. To evaluate our method, we propose two new benchmarks, ToneBank and DebateMix, targeting compositional behavioral control. Empirical results across 3 model families, validated by both activation-based classifiers and LLM-based judges, demonstrate that K-Steering outperforms strong baselines in accurately steering multiple behaviors. 

---
# Directional Non-Commutative Monoidal Structures with Interchange Law via Commutative Generators 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2505.24533)  

**Abstract**: We introduce a novel framework consisting of a class of algebraic structures that generalize one-dimensional monoidal systems into higher dimensions by defining per-axis composition operators subject to non-commutativity and a global interchange law. These structures, defined recursively from a base case of vector-matrix pairs, model directional composition in multiple dimensions while preserving structural coherence through commutative linear operators.
We show that the framework that unifies several well-known linear transforms in signal processing and data analysis. In this framework, data indices are embedded into a composite structure that decomposes into simpler components. We show that classic transforms such as the Discrete Fourier Transform (DFT), the Walsh transform, and the Hadamard transform are special cases of our algebraic structure. The framework provides a systematic way to derive these transforms by appropriately choosing vector and matrix pairs. By subsuming classical transforms within a common structure, the framework also enables the development of learnable transformations tailored to specific data modalities and tasks. 

---
# Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors 

**Authors**: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi, Giovanni Puccetti, Felice Dell'Orletta, Andrea Esuli  

**Link**: [PDF](https://arxiv.org/pdf/2505.24523)  

**Abstract**: Recent advancements in Generative AI and Large Language Models (LLMs) have enabled the creation of highly realistic synthetic content, raising concerns about the potential for malicious use, such as misinformation and manipulation. Moreover, detecting Machine-Generated Text (MGT) remains challenging due to the lack of robust benchmarks that assess generalization to real-world scenarios. In this work, we present a pipeline to test the resilience of state-of-the-art MGT detectors (e.g., Mage, Radar, LLM-DetectAIve) to linguistically informed adversarial attacks. To challenge the detectors, we fine-tune language models using Direct Preference Optimization (DPO) to shift the MGT style toward human-written text (HWT). This exploits the detectors' reliance on stylistic clues, making new generations more challenging to detect. Additionally, we analyze the linguistic shifts induced by the alignment and which features are used by detectors to detect MGT texts. Our results show that detectors can be easily fooled with relatively few examples, resulting in a significant drop in detection performance. This highlights the importance of improving detection methods and making them robust to unseen in-domain texts. 

---
# Can Slow-thinking LLMs Reason Over Time? Empirical Studies in Time Series Forecasting 

**Authors**: Jiahao Wang, Mingyue Cheng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24511)  

**Abstract**: Time series forecasting (TSF) is a fundamental and widely studied task, spanning methods from classical statistical approaches to modern deep learning and multimodal language modeling. Despite their effectiveness, these methods often follow a fast thinking paradigm emphasizing pattern extraction and direct value mapping, while overlooking explicit reasoning over temporal dynamics and contextual dependencies. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1, DeepSeek-R1) have demonstrated impressive multi-step reasoning capabilities across diverse domains, suggesting a new opportunity for reframing TSF as a structured reasoning task. This motivates a key question: can slow-thinking LLMs effectively reason over temporal patterns to support time series forecasting, even in zero-shot manner? To investigate this, in this paper, we propose TimeReasoner, an extensive empirical study that formulates TSF as a conditional reasoning task. We design a series of prompting strategies to elicit inference-time reasoning from pretrained slow-thinking LLMs and evaluate their performance across diverse TSF benchmarks. Our findings reveal that slow-thinking LLMs exhibit non-trivial zero-shot forecasting capabilities, especially in capturing high-level trends and contextual shifts. While preliminary, our study surfaces important insights into the reasoning behaviors of LLMs in temporal domains highlighting both their potential and limitations. We hope this work catalyzes further research into reasoning-based forecasting paradigms and paves the way toward more interpretable and generalizable TSF frameworks. 

---
# Online Fair Division with Additional Information 

**Authors**: Tzeh Yuan Neoh, Jannik Peters, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2505.24503)  

**Abstract**: We study the problem of fairly allocating indivisible goods to agents in an online setting, where goods arrive sequentially and must be allocated irrevocably to agents. Focusing on the popular fairness notions of envy-freeness, proportionality, and maximin share fairness (and their approximate variants), we ask how the availability of information on future goods influences the existence and approximability of fair allocations. In the absence of any such information, we establish strong impossibility results, demonstrating the inherent difficulty of achieving even approximate fairness guarantees. In contrast, we demonstrate that knowledge of additional information -- such as aggregate of each agent's total valuations (equivalently, normalized valuations) or the multiset of future goods values (frequency predictions) -- would enable the design of fairer online algorithms. Given normalization information, we propose an algorithm that achieves stronger fairness guarantees than previously known results. Given frequency predictions, we introduce a meta-algorithm that leverages frequency predictions to match the best-known offline guarantees for a broad class of ''share-based'' fairness notions. Our complementary impossibility results in each setting underscore both the limitations imposed by uncertainty about future goods and the potential of leveraging structured information to achieve fairer outcomes in online fair division. 

---
# TimeHC-RL: Temporal-aware Hierarchical Cognitive Reinforcement Learning for Enhancing LLMs' Social Intelligence 

**Authors**: Guiyang Hou, Xing Gao, Yuchuan Wu, Xiang Huang, Wenqi Zhang, Zhe Zheng, Yongliang Shen, Jialu Du, Fei Huang, Yongbin Li, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24500)  

**Abstract**: Recently, Large Language Models (LLMs) have made significant progress in IQ-related domains that require careful thinking, such as mathematics and coding. However, enhancing LLMs' cognitive development in social domains, particularly from a post-training perspective, remains underexplored. Recognizing that the social world follows a distinct timeline and requires a richer blend of cognitive modes (from intuitive reactions (System 1) and surface-level thinking to deliberate thinking (System 2)) than mathematics, which primarily relies on System 2 cognition (careful, step-by-step reasoning), we introduce Temporal-aware Hierarchical Cognitive Reinforcement Learning (TimeHC-RL) for enhancing LLMs' social intelligence. In our experiments, we systematically explore improving LLMs' social intelligence and validate the effectiveness of the TimeHC-RL method, through five other post-training paradigms and two test-time intervention paradigms on eight datasets with diverse data patterns. Experimental results reveal the superiority of our proposed TimeHC-RL method compared to the widely adopted System 2 RL method. It gives the 7B backbone model wings, enabling it to rival the performance of advanced models like DeepSeek-R1 and OpenAI-O3. Additionally, the systematic exploration from post-training and test-time interventions perspectives to improve LLMs' social intelligence has uncovered several valuable insights. 

---
# Object Centric Concept Bottlenecks 

**Authors**: David Steinmann, Wolfgang Stammer, Antonia Wüst, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.24492)  

**Abstract**: Developing high-performing, yet interpretable models remains a critical challenge in modern AI. Concept-based models (CBMs) attempt to address this by extracting human-understandable concepts from a global encoding (e.g., image encoding) and then applying a linear classifier on the resulting concept activations, enabling transparent decision-making. However, their reliance on holistic image encodings limits their expressiveness in object-centric real-world settings and thus hinders their ability to solve complex vision tasks beyond single-label classification. To tackle these challenges, we introduce Object-Centric Concept Bottlenecks (OCB), a framework that combines the strengths of CBMs and pre-trained object-centric foundation models, boosting performance and interpretability. We evaluate OCB on complex image datasets and conduct a comprehensive ablation study to analyze key components of the framework, such as strategies for aggregating object-concept encodings. The results show that OCB outperforms traditional CBMs and allows one to make interpretable decisions for complex visual tasks. 

---
# Deformable Attention Mechanisms Applied to Object Detection, case of Remote Sensing 

**Authors**: Anasse Boutayeb, Iyad Lahsen-cherif, Ahmed El Khadimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24489)  

**Abstract**: Object detection has recently seen an interesting trend in terms of the most innovative research work, this task being of particular importance in the field of remote sensing, given the consistency of these images in terms of geographical coverage and the objects present. Furthermore, Deep Learning (DL) models, in particular those based on Transformers, are especially relevant for visual computing tasks in general, and target detection in particular. Thus, the present work proposes an application of Deformable-DETR model, a specific architecture using deformable attention mechanisms, on remote sensing images in two different modes, especially optical and Synthetic Aperture Radar (SAR). To achieve this objective, two datasets are used, one optical, which is Pleiades Aircraft dataset, and the other SAR, in particular SAR Ship Detection Dataset (SSDD). The results of a 10-fold stratified validation showed that the proposed model performed particularly well, obtaining an F1 score of 95.12% for the optical dataset and 94.54% for SSDD, while comparing these results with several models detections, especially those based on CNNs and transformers, as well as those specifically designed to detect different object classes in remote sensing images. 

---
# Rehearsal with Auxiliary-Informed Sampling for Audio Deepfake Detection 

**Authors**: Falih Gozi Febrinanto, Kristen Moore, Chandra Thapa, Jiangang Ma, Vidya Saikrishna, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.24486)  

**Abstract**: The performance of existing audio deepfake detection frameworks degrades when confronted with new deepfake attacks. Rehearsal-based continual learning (CL), which updates models using a limited set of old data samples, helps preserve prior knowledge while incorporating new information. However, existing rehearsal techniques don't effectively capture the diversity of audio characteristics, introducing bias and increasing the risk of forgetting. To address this challenge, we propose Rehearsal with Auxiliary-Informed Sampling (RAIS), a rehearsal-based CL approach for audio deepfake detection. RAIS employs a label generation network to produce auxiliary labels, guiding diverse sample selection for the memory buffer. Extensive experiments show RAIS outperforms state-of-the-art methods, achieving an average Equal Error Rate (EER) of 1.953 % across five experiences. The code is available at: this https URL. 

---
# Towards Effective Code-Integrated Reasoning 

**Authors**: Fei Bai, Yingqian Min, Beichen Zhang, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24480)  

**Abstract**: In this paper, we investigate code-integrated reasoning, where models generate code when necessary and integrate feedback by executing it through a code interpreter. To acquire this capability, models must learn when and how to use external code tools effectively, which is supported by tool-augmented reinforcement learning (RL) through interactive learning. Despite its benefits, tool-augmented RL can still suffer from potential instability in the learning dynamics. In light of this challenge, we present a systematic approach to improving the training effectiveness and stability of tool-augmented RL for code-integrated reasoning. Specifically, we develop enhanced training strategies that balance exploration and stability, progressively building tool-use capabilities while improving reasoning performance. Through extensive experiments on five mainstream mathematical reasoning benchmarks, our model demonstrates significant performance improvements over multiple competitive baselines. Furthermore, we conduct an in-depth analysis of the mechanism and effect of code-integrated reasoning, revealing several key insights, such as the extension of model's capability boundaries and the simultaneous improvement of reasoning efficiency through code integration. All data and code for reproducing this work are available at: this https URL. 

---
# Evaluating Gemini in an arena for learning 

**Authors**: LearnLM Team Google, Abhinit Modi, Aditya Srikanth Veerubhotla, Aliya Rysbek, Andrea Huber, Ankit Anand, Avishkar Bhoopchand, Brett Wiltshire, Daniel Gillick, Daniel Kasenberg, Eleni Sgouritsa, Gal Elidan, Hengrui Liu, Holger Winnemoeller, Irina Jurenka, James Cohan, Jennifer She, Julia Wilkowski, Kaiz Alarakyia, Kevin R. McKee, Komal Singh, Lisa Wang, Markus Kunesch, Miruna Pîslar, Niv Efron, Parsa Mahmoudieh, Pierre-Alexandre Kamienny, Sara Wiltberger, Shakir Mohamed, Shashank Agarwal, Shubham Milind Phal, Sun Jae Lee, Theofilos Strinopoulos, Wei-Jen Ko, Yael Gold-Zamir, Yael Haramaty, Yannis Assael  

**Link**: [PDF](https://arxiv.org/pdf/2505.24477)  

**Abstract**: Artificial intelligence (AI) is poised to transform education, but the research community lacks a robust, general benchmark to evaluate AI models for learning. To assess state-of-the-art support for educational use cases, we ran an "arena for learning" where educators and pedagogy experts conduct blind, head-to-head, multi-turn comparisons of leading AI models. In particular, $N = 189$ educators drew from their experience to role-play realistic learning use cases, interacting with two models sequentially, after which $N = 206$ experts judged which model better supported the user's learning goals. The arena evaluated a slate of state-of-the-art models: Gemini 2.5 Pro, Claude 3.7 Sonnet, GPT-4o, and OpenAI o3. Excluding ties, experts preferred Gemini 2.5 Pro in 73.2% of these match-ups -- ranking it first overall in the arena. Gemini 2.5 Pro also demonstrated markedly higher performance across key principles of good pedagogy. Altogether, these results position Gemini 2.5 Pro as a leading model for learning. 

---
# Train One Sparse Autoencoder Across Multiple Sparsity Budgets to Preserve Interpretability and Accuracy 

**Authors**: Nikita Balagansky, Yaroslav Aksenov, Daniil Laptev, Vadim Kurochkin, Gleb Gerasimov, Nikita Koryagin, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2505.24473)  

**Abstract**: Sparse Autoencoders (SAEs) have proven to be powerful tools for interpreting neural networks by decomposing hidden representations into disentangled, interpretable features via sparsity constraints. However, conventional SAEs are constrained by the fixed sparsity level chosen during training; meeting different sparsity requirements therefore demands separate models and increases the computational footprint during both training and evaluation. We introduce a novel training objective, \emph{HierarchicalTopK}, which trains a single SAE to optimise reconstructions across multiple sparsity levels simultaneously. Experiments with Gemma-2 2B demonstrate that our approach achieves Pareto-optimal trade-offs between sparsity and explained variance, outperforming traditional SAEs trained at individual sparsity levels. Further analysis shows that HierarchicalTopK preserves high interpretability scores even at higher sparsity. The proposed objective thus closes an important gap between flexibility and interpretability in SAE design. 

---
# VietMix: A Naturally Occurring Vietnamese-English Code-Mixed Corpus with Iterative Augmentation for Machine Translation 

**Authors**: Hieu Tran, Phuong-Anh Nguyen-Le, Huy Nghiem, Quang-Nhan Nguyen, Wei Ai, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.24472)  

**Abstract**: Machine translation systems fail when processing code-mixed inputs for low-resource languages. We address this challenge by curating VietMix, a parallel corpus of naturally occurring code-mixed Vietnamese text paired with expert English translations. Augmenting this resource, we developed a complementary synthetic data generation pipeline. This pipeline incorporates filtering mechanisms to ensure syntactic plausibility and pragmatic appropriateness in code-mixing patterns. Experimental validation shows our naturalistic and complementary synthetic data boost models' performance, measured by translation quality estimation scores, of up to 71.84 on COMETkiwi and 81.77 on XCOMET. Triangulating positive results with LLM-based assessments, augmented models are favored over seed fine-tuned counterparts in approximately 49% of judgments (54-56% excluding ties). VietMix and our augmentation methodology advance ecological validity in neural MT evaluations and establish a framework for addressing code-mixed translation challenges across other low-resource pairs. 

---
# LPASS: Linear Probes as Stepping Stones for vulnerability detection using compressed LLMs 

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux  

**Link**: [PDF](https://arxiv.org/pdf/2505.24451)  

**Abstract**: Large Language Models (LLMs) are being extensively used for cybersecurity purposes. One of them is the detection of vulnerable codes. For the sake of efficiency and effectiveness, compression and fine-tuning techniques are being developed, respectively. However, they involve spending substantial computational efforts. In this vein, we analyse how Linear Probes (LPs) can be used to provide an estimation on the performance of a compressed LLM at an early phase -- before fine-tuning. We also show their suitability to set the cut-off point when applying layer pruning compression. Our approach, dubbed $LPASS$, is applied in BERT and Gemma for the detection of 12 of MITRE's Top 25 most dangerous vulnerabilities on 480k C/C++ samples. LPs can be computed in 142.97 s. and provide key findings: (1) 33.3 \% and 72.2\% of layers can be removed, respectively, with no precision loss; (2) they provide an early estimate of the post-fine-tuning and post-compression model effectiveness, with 3\% and 8.68\% as the lowest and average precision errors, respectively. $LPASS$-based LLMs outperform the state of the art, reaching 86.9\% of accuracy in multi-class vulnerability detection. Interestingly, $LPASS$-based compressed versions of Gemma outperform the original ones by 1.6\% of F1-score at a maximum while saving 29.4 \% and 23.8\% of training and inference time and 42.98\% of model size. 

---
# Learning Safety Constraints for Large Language Models 

**Authors**: Xin Chen, Yarden As, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2505.24445)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools but pose significant safety risks through harmful outputs and vulnerability to adversarial attacks. We propose SaP, short for Safety Polytope, a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. We develop a framework that identifies safe and unsafe regions via the polytope's facets, enabling both detection and correction of unsafe outputs through geometric steering. Unlike existing approaches that modify model weights, SaP operates post-hoc in the representation space, preserving model capabilities while enforcing safety constraints. Experiments across multiple LLMs demonstrate that our method can effectively detect unethical inputs, reduce adversarial attack success rates while maintaining performance on standard tasks, thus highlighting the importance of having an explicit geometric model for safety. Analysis of the learned polytope facets reveals emergence of specialization in detecting different semantic notions of safety, providing interpretable insights into how safety is captured in LLMs' representation space. 

---
# Deep Learning Weather Models for Subregional Ocean Forecasting: A Case Study on the Canary Current Upwelling System 

**Authors**: Giovanny C-Londoño, Javier Sánchez, Ángel Rodríguez-Santana  

**Link**: [PDF](https://arxiv.org/pdf/2505.24429)  

**Abstract**: Oceanographic forecasting impacts various sectors of society by supporting environmental conservation and economic activities. Based on global circulation models, traditional forecasting methods are computationally expensive and slow, limiting their ability to provide rapid forecasts. Recent advances in deep learning offer faster and more accurate predictions, although these data-driven models are often trained with global data from numerical simulations, which may not reflect reality. The emergence of such models presents great potential for improving ocean prediction at a subregional domain. However, their ability to predict fine-scale ocean processes, like mesoscale structures, remains largely unknown. This work aims to adapt a graph neural network initially developed for global weather forecasting to improve subregional ocean prediction, specifically focusing on the Canary Current upwelling system. The model is trained with satellite data and compared to state-of-the-art physical ocean models to assess its performance in capturing ocean dynamics. Our results show that the deep learning model surpasses traditional methods in precision despite some challenges in upwelling areas. It demonstrated superior performance in reducing RMSE errors compared to ConvLSTM and the GLORYS reanalysis, particularly in regions with complex oceanic dynamics such as Cape Ghir, Cape Bojador, and Cape Blanc. The model achieved improvements of up to 26.5% relative to ConvLSTM and error reductions of up to 76% in 5-day forecasts compared to the GLORYS reanalysis at these critical locations, highlighting its enhanced capability to capture spatial variability and improve predictive accuracy in complex areas. These findings suggest the viability of adapting meteorological data-driven models for improving subregional medium-term ocean forecasting. 

---
# Boosting Automatic Exercise Evaluation Through Musculoskeletal Simulation-Based IMU Data Augmentation 

**Authors**: Andreas Spilz, Heiko Oppel, Michael Munz  

**Link**: [PDF](https://arxiv.org/pdf/2505.24415)  

**Abstract**: Automated evaluation of movement quality holds significant potential for enhancing physiotherapeutic treatments and sports training by providing objective, real-time feedback. However, the effectiveness of deep learning models in assessing movements captured by inertial measurement units (IMUs) is often hampered by limited data availability, class imbalance, and label ambiguity. In this work, we present a novel data augmentation method that generates realistic IMU data using musculoskeletal simulations integrated with systematic modifications of movement trajectories. Crucially, our approach ensures biomechanical plausibility and allows for automatic, reliable labeling by combining inverse kinematic parameters with a knowledge-based evaluation strategy. Extensive evaluations demonstrate that augmented variants closely resembles real-world data, significantly improving the classification accuracy and generalization capability of neural network models. Additionally, we highlight the benefits of augmented data for patient-specific fine-tuning scenarios, particularly when only limited subject-specific training examples are available. Our findings underline the practicality and efficacy of this augmentation method in overcoming common challenges faced by deep learning applications in physiotherapeutic exercise evaluation. 

---
# LLMs Are Globally Multilingual Yet Locally Monolingual: Exploring Knowledge Transfer via Language and Thought Theory 

**Authors**: Eojin Kang, Juae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24409)  

**Abstract**: Multilingual large language models (LLMs) open up new possibilities for leveraging information across languages, but their factual knowledge recall remains inconsistent depending on the input language. While previous studies have attempted to address this issue through English-based prompting and evaluation, we explore non-English to English transfer via Language and Thought Theory. This perspective allows us to examine language-thought binding in LLMs and uncover why factual knowledge often fails to transfer effectively. We propose the Language-to-Thought (L2T) prompting strategy, which analyzes the relationship between input language, internal cognitive processes, and knowledge. Experimental results challenge the assumption that English-based approaches consistently outperform other languages and offer a novel insight that aligning the model's internal thought with the knowledge required for the task is critical for successful cross-lingual transfer. Furthermore, we show that applying L2T during training can alleviate LLMs' reliance on the input language and facilitate cross-linguistic knowledge integration without translation-based learning. Code and datasets will be available. 

---
# SASP: Strip-Aware Spatial Perception for Fine-Grained Bird Image Classification 

**Authors**: Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24380)  

**Abstract**: Fine-grained bird image classification (FBIC) is not only of great significance for ecological monitoring and species identification, but also holds broad research value in the fields of image recognition and fine-grained visual modeling. Compared with general image classification tasks, FBIC poses more formidable challenges: 1) the differences in species size and imaging distance result in the varying sizes of birds presented in the images; 2) complex natural habitats often introduce strong background interference; 3) and highly flexible poses such as flying, perching, or foraging result in substantial intra-class variability. These factors collectively make it difficult for traditional methods to stably extract discriminative features, thereby limiting the generalizability and interpretability of models in real-world applications. To address these challenges, this paper proposes a fine-grained bird classification framework based on strip-aware spatial perception, which aims to capture long-range spatial dependencies across entire rows or columns in bird images, thereby enhancing the model's robustness and interpretability. The proposed method incorporates two novel modules: extensional perception aggregator (EPA) and channel semantic weaving (CSW). Specifically, EPA integrates local texture details with global structural cues by aggregating information across horizontal and vertical spatial directions. CSW further refines the semantic representations by adaptively fusing long-range and short-range information along the channel dimension. Built upon a ResNet-50 backbone, the model enables jump-wise connection of extended structural features across the spatial domain. Experimental results on the CUB-200-2011 dataset demonstrate that our framework achieves significant performance improvements while maintaining architectural efficiency. 

---
# Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models 

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24379)  

**Abstract**: Large language models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard, believed to be robust against privacy-related attacks. In this paper, we challenge this assumption by introducing a novel data extraction attack that compromises even exact unlearning. Our method leverages both the pre- and post-unlearning models: by guiding the post-unlearning model using signals from the pre-unlearning model, we uncover patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. 

---
# Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer 

**Authors**: Yilun Kong, Guozheng Ma, Qi Zhao, Haoyu Wang, Li Shen, Xueqian Wang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24378)  

**Abstract**: Despite recent advancements in offline multi-task reinforcement learning (MTRL) have harnessed the powerful capabilities of the Transformer architecture, most approaches focus on a limited number of tasks, with scaling to extremely massive tasks remaining a formidable challenge. In this paper, we first revisit the key impact of task numbers on current MTRL method, and further reveal that naively expanding the parameters proves insufficient to counteract the performance degradation as the number of tasks escalates. Building upon these insights, we propose M3DT, a novel mixture-of-experts (MoE) framework that tackles task scalability by further unlocking the model's parameter scalability. Specifically, we enhance both the architecture and the optimization of the agent, where we strengthen the Decision Transformer (DT) backbone with MoE to reduce task load on parameter subsets, and introduce a three-stage training mechanism to facilitate efficient training with optimal performance. Experimental results show that, by increasing the number of experts, M3DT not only consistently enhances its performance as model expansion on the fixed task numbers, but also exhibits remarkable task scalability, successfully extending to 160 tasks with superior performance. 

---
# Grid-LOGAT: Grid Based Local and Global Area Transcription for Video Question Answering 

**Authors**: Md Intisar Chowdhury, Kittinun Aukkapinyo, Hiroshi Fujimura, Joo Ann Woo, Wasu Wasusatein, Fadoua Ghourabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24371)  

**Abstract**: In this paper, we propose a Grid-based Local and Global Area Transcription (Grid-LoGAT) system for Video Question Answering (VideoQA). The system operates in two phases. First, extracting text transcripts from video frames using a Vision-Language Model (VLM). Next, processing questions using these transcripts to generate answers through a Large Language Model (LLM). This design ensures image privacy by deploying the VLM on edge devices and the LLM in the cloud. To improve transcript quality, we propose grid-based visual prompting, which extracts intricate local details from each grid cell and integrates them with global information. Evaluation results show that Grid-LoGAT, using the open-source VLM (LLaVA-1.6-7B) and LLM (Llama-3.1-8B), outperforms state-of-the-art methods with similar baseline models on NExT-QA and STAR-QA datasets with an accuracy of 65.9% and 50.11% respectively. Additionally, our method surpasses the non-grid version by 24 points on localization-based questions we created using NExT-QA. 

---
# Adversarial Preference Learning for Robust LLM Alignment 

**Authors**: Yuanfu Wang, Pengyu Wang, Chenyang Xi, Bo Tang, Junyi Zhu, Wenqiang Wei, Chen Chen, Chao Yang, Jingfeng Zhang, Chaochao Lu, Yijun Niu, Keming Mao, Zhiyu Li, Feiyu Xiong, Jie Hu, Mingchuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24369)  

**Abstract**: Modern language models often rely on Reinforcement Learning from Human Feedback (RLHF) to encourage safe behaviors. However, they remain vulnerable to adversarial attacks due to three key limitations: (1) the inefficiency and high cost of human annotation, (2) the vast diversity of potential adversarial attacks, and (3) the risk of feedback bias and reward hacking. To address these challenges, we introduce Adversarial Preference Learning (APL), an iterative adversarial training method incorporating three key innovations. First, a direct harmfulness metric based on the model's intrinsic preference probabilities, eliminating reliance on external assessment. Second, a conditional generative attacker that synthesizes input-specific adversarial variations. Third, an iterative framework with automated closed-loop feedback, enabling continuous adaptation through vulnerability discovery and mitigation. Experiments on Mistral-7B-Instruct-v0.3 demonstrate that APL significantly enhances robustness, achieving 83.33% harmlessness win rate over the base model (evaluated by GPT-4o), reducing harmful outputs from 5.88% to 0.43% (measured by LLaMA-Guard), and lowering attack success rate by up to 65% according to HarmBench. Notably, APL maintains competitive utility, with an MT-Bench score of 6.59 (comparable to the baseline 6.78) and an LC-WinRate of 46.52% against the base model. 

---
# ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration 

**Authors**: Xianglong Yan, Zhiteng Li, Tianao Zhang, Linghe Kong, Yulun Zhang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24357)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance, yet their capability on long-context reasoning is often constrained by the excessive memory required to store the Key-Value (KV) cache. This makes KV cache compression an essential step toward enabling efficient long-context reasoning. Recent methods have explored reducing the hidden dimensions of the KV cache, but many introduce additional computation through projection layers or suffer from significant performance degradation under high compression ratios. To address these challenges, we propose ReCalKV, a post-training KV cache compression method that reduces the hidden dimensions of the KV cache. We develop distinct compression strategies for Keys and Values based on their different roles and varying importance in the attention mechanism. For Keys, we propose Head-wise Similarity-aware Reordering (HSR), which clusters similar heads and applies grouped SVD to the key projection matrix, reducing additional computation while preserving accuracy. For Values, we propose Offline Calibration and Matrix Fusion (OCMF) to preserve accuracy without extra computational overhead. Experiments show that ReCalKV outperforms existing low-rank compression methods, achieving high compression ratios with minimal performance loss. Code is available at: this https URL. 

---
# Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings 

**Authors**: Shujian Yang, Shiyao Cui, Chuanrui Hu, Haicheng Wang, Tianwei Zhang, Minlie Huang, Jialiang Lu, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24341)  

**Abstract**: Detecting toxic content using language models is important but challenging. While large language models (LLMs) have demonstrated strong performance in understanding Chinese, recent studies show that simple character substitutions in toxic Chinese text can easily confuse the state-of-the-art (SOTA) LLMs. In this paper, we highlight the multimodal nature of Chinese language as a key challenge for deploying LLMs in toxic Chinese detection. First, we propose a taxonomy of 3 perturbation strategies and 8 specific approaches in toxic Chinese content. Then, we curate a dataset based on this taxonomy, and benchmark 9 SOTA LLMs (from both the US and China) to assess if they can detect perturbed toxic Chinese text. Additionally, we explore cost-effective enhancement solutions like in-context learning (ICL) and supervised fine-tuning (SFT). Our results reveal two important findings. (1) LLMs are less capable of detecting perturbed multimodal Chinese toxic contents. (2) ICL or SFT with a small number of perturbed examples may cause the LLMs "overcorrect'': misidentify many normal Chinese contents as toxic. 

---
# When Humans Growl and Birds Speak: High-Fidelity Voice Conversion from Human to Animal and Designed Sounds 

**Authors**: Minsu Kang, Seolhee Lee, Choonghyeon Lee, Namhyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.24336)  

**Abstract**: Human to non-human voice conversion (H2NH-VC) transforms human speech into animal or designed vocalizations. Unlike prior studies focused on dog-sounds and 16 or 22.05kHz audio transformation, this work addresses a broader range of non-speech sounds, including natural sounds (lion-roars, birdsongs) and designed voice (synthetic growls). To accomodate generation of diverse non-speech sounds and 44.1kHz high-quality audio transformation, we introduce a preprocessing pipeline and an improved CVAE-based H2NH-VC model, both optimized for human and non-human voices. Experimental results showed that the proposed method outperformed baselines in quality, naturalness, and similarity MOS, achieving effective voice conversion across diverse non-human timbres. Demo samples are available at this https URL 

---
# AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning 

**Authors**: Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24298)  

**Abstract**: Reinforcement learning (RL) has become a trending paradigm for training large language models (LLMs), particularly for reasoning tasks. Effective RL for LLMs requires massive parallelization and poses an urgent need for efficient training systems. Most existing large-scale RL systems for LLMs are synchronous by alternating generation and training in a batch setting, where the rollouts in each training batch are generated by the same (or latest) model. This stabilizes RL training but suffers from severe system-level inefficiency. Generation must wait until the longest output in the batch is completed before model update, resulting in GPU underutilization. We present AReaL, a \emph{fully asynchronous} RL system that completely decouples generation from training. Rollout workers in AReaL continuously generate new outputs without waiting, while training workers update the model whenever a batch of data is collected. AReaL also incorporates a collection of system-level optimizations, leading to substantially higher GPU utilization. To stabilize RL training, AReaL balances the workload of rollout and training workers to control data staleness, and adopts a staleness-enhanced PPO variant to better handle outdated training samples. Extensive experiments on math and code reasoning benchmarks show that AReaL achieves \textbf{up to 2.57$\times$ training speedup} compared to the best synchronous systems with the same number of GPUs and matched or even improved final performance. The code of AReaL is available at this https URL. 

---
# Large Language Models are Locally Linear Mappings 

**Authors**: James R. Golden  

**Link**: [PDF](https://arxiv.org/pdf/2505.24293)  

**Abstract**: We demonstrate that the inference operations of several open-weight large language models (LLMs) can be mapped to an exactly equivalent linear system for an input sequence without modifying the model weights or altering output predictions. Extending techniques from image diffusion models that exhibit local or piecewise linearity, we strategically alter the gradient computation with respect to a given input sequence for a next-token prediction such that the Jacobian of the model nearly exactly reproduces the forward prediction with a linear system. We demonstrate this approach across models (Llama 3, Gemma 3, Qwen 3, Phi 4, Mistral Ministral and OLMo 2, up to Llama 3.3 70B Q4) and show through the singular value decomposition of the detached Jacobian that these LLMs operate in extremely low-dimensional subspaces where many of the largest singular vectors decode to concepts related to the most-likely output token. This approach also allows us to examine the operation of each successive layer (and its attention and MLP components) as nearly-exact linear systems and observe the emergence of semantic concepts. Despite their expressive power and global nonlinearity, modern LLMs can be interpreted through nearly-exact locally linear decompositions that provide insights into their internal representations and reveal interpretable semantic structures in the next-token prediction process. 

---
# Discl-VC: Disentangled Discrete Tokens and In-Context Learning for Controllable Zero-Shot Voice Conversion 

**Authors**: Kaidi Wang, Wenhao Guan, Ziyue Jiang, Hukai Huang, Peijie Chen, Weijie Wu, Qingyang Hong, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24291)  

**Abstract**: Currently, zero-shot voice conversion systems are capable of synthesizing the voice of unseen speakers. However, most existing approaches struggle to accurately replicate the speaking style of the source speaker or mimic the distinctive speaking style of the target speaker, thereby limiting the controllability of voice conversion. In this work, we propose Discl-VC, a novel voice conversion framework that disentangles content and prosody information from self-supervised speech representations and synthesizes the target speaker's voice through in-context learning with a flow matching transformer. To enable precise control over the prosody of generated speech, we introduce a mask generative transformer that predicts discrete prosody tokens in a non-autoregressive manner based on prompts. Experimental results demonstrate the superior performance of Discl-VC in zero-shot voice conversion and its remarkable accuracy in prosody control for synthesized speech. 

---
# INSIGHT: A Survey of In-Network Systems for Intelligent, High-Efficiency AI and Topology Optimization 

**Authors**: Aleksandr Algazinov, Joydeep Chandra, Matt Laing  

**Link**: [PDF](https://arxiv.org/pdf/2505.24269)  

**Abstract**: In-network computation represents a transformative approach to addressing the escalating demands of Artificial Intelligence (AI) workloads on network infrastructure. By leveraging the processing capabilities of network devices such as switches, routers, and Network Interface Cards (NICs), this paradigm enables AI computations to be performed directly within the network fabric, significantly reducing latency, enhancing throughput, and optimizing resource utilization. This paper provides a comprehensive analysis of optimizing in-network computation for AI, exploring the evolution of programmable network architectures, such as Software-Defined Networking (SDN) and Programmable Data Planes (PDPs), and their convergence with AI. It examines methodologies for mapping AI models onto resource-constrained network devices, addressing challenges like limited memory and computational capabilities through efficient algorithm design and model compression techniques. The paper also highlights advancements in distributed learning, particularly in-network aggregation, and the potential of federated learning to enhance privacy and scalability. Frameworks like Planter and Quark are discussed for simplifying development, alongside key applications such as intelligent network monitoring, intrusion detection, traffic management, and Edge AI. Future research directions, including runtime programmability, standardized benchmarks, and new applications paradigms, are proposed to advance this rapidly evolving field. This survey underscores the potential of in-network AI to create intelligent, efficient, and responsive networks capable of meeting the demands of next-generation AI applications. 

---
# Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations 

**Authors**: Xin Quan, Marco Valentino, Louise A. Dennis, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.24264)  

**Abstract**: Natural language explanations play a fundamental role in Natural Language Inference (NLI) by revealing how premises logically entail hypotheses. Recent work has shown that the interaction of large language models (LLMs) with theorem provers (TPs) can help verify and improve the validity of NLI explanations. However, TPs require translating natural language into machine-verifiable formal representations, a process that introduces the risk of semantic information loss and unfaithful interpretation, an issue compounded by LLMs' challenges in capturing critical logical structures with sufficient precision. Moreover, LLMs are still limited in their capacity for rigorous and robust proof construction within formal verification frameworks. To mitigate issues related to faithfulness and robustness, this paper investigates strategies to (1) alleviate semantic loss during autoformalisation, (2) efficiently identify and correct syntactic errors in logical representations, (3) explicitly use logical expressions to guide LLMs in generating structured proof sketches, and (4) increase LLMs' capacity of interpreting TP's feedback for iterative refinement. Our empirical results on e-SNLI, QASC and WorldTree using different LLMs demonstrate that the proposed strategies yield significant improvements in autoformalisation (+18.46%, +34.2%, +39.77%) and explanation refinement (+29.5%, +51.5%, +41.25%) over the state-of-the-art model. Moreover, we show that specific interventions on the hybrid LLM-TP architecture can substantially improve efficiency, drastically reducing the number of iterations required for successful verification. 

---
# Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games 

**Authors**: Neemesh Yadav, Palakorn Achananuparp, Jing Jiang, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24255)  

**Abstract**: Large Language Models (LLMs) have shown potential in simulating human behaviors and performing theory-of-mind (ToM) reasoning, a crucial skill for complex social interactions. In this study, we investigate the role of ToM reasoning in aligning agentic behaviors with human norms in negotiation tasks, using the ultimatum game as a controlled environment. We initialized LLM agents with different prosocial beliefs (including Greedy, Fair, and Selfless) and reasoning methods like chain-of-thought (CoT) and varying ToM levels, and examined their decision-making processes across diverse LLMs, including reasoning models like o3-mini and DeepSeek-R1 Distilled Qwen 32B. Results from 2,700 simulations indicated that ToM reasoning enhances behavior alignment, decision-making consistency, and negotiation outcomes. Consistent with previous findings, reasoning models exhibit limited capability compared to models with ToM reasoning, different roles of the game benefits with different orders of ToM reasoning. Our findings contribute to the understanding of ToM's role in enhancing human-AI interaction and cooperative decision-making. The code used for our experiments can be found at this https URL. 

---
# Interactive Video Generation via Domain Adaptation 

**Authors**: Ishaan Rawal, Suryansh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.24253)  

**Abstract**: Text-conditioned diffusion models have emerged as powerful tools for high-quality video generation. However, enabling Interactive Video Generation (IVG), where users control motion elements such as object trajectory, remains challenging. Recent training-free approaches introduce attention masking to guide trajectory, but this often degrades perceptual quality. We identify two key failure modes in these methods, both of which we interpret as domain shift problems, and propose solutions inspired by domain adaptation. First, we attribute the perceptual degradation to internal covariate shift induced by attention masking, as pretrained models are not trained to handle masked attention. To address this, we propose mask normalization, a pre-normalization layer designed to mitigate this shift via distribution matching. Second, we address initialization gap, where the randomly sampled initial noise does not align with IVG conditioning, by introducing a temporal intrinsic diffusion prior that enforces spatio-temporal consistency at each denoising step. Extensive qualitative and quantitative evaluations demonstrate that mask normalization and temporal intrinsic denoising improve both perceptual quality and trajectory control over the existing state-of-the-art IVG techniques. 

---
# A Reward-driven Automated Webshell Malicious-code Generator for Red-teaming 

**Authors**: Yizhong Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.24252)  

**Abstract**: Frequent cyber-attacks have elevated WebShell exploitation and defense to a critical research focus within network security. However, there remains a significant shortage of publicly available, well-categorized malicious-code datasets organized by obfuscation method. Existing malicious-code generation methods, which primarily rely on prompt engineering, often suffer from limited diversity and high redundancy in the payloads they produce. To address these limitations, we propose \textbf{RAWG}, a \textbf{R}eward-driven \textbf{A}utomated \textbf{W}ebshell Malicious-code \textbf{G}enerator designed for red-teaming applications. Our approach begins by categorizing webshell samples from common datasets into seven distinct types of obfuscation. We then employ a large language model (LLM) to extract and normalize key tokens from each sample, creating a standardized, high-quality corpus. Using this curated dataset, we perform supervised fine-tuning (SFT) on an open-source large model to enable the generation of diverse, highly obfuscated webshell malicious payloads. To further enhance generation quality, we apply Proximal Policy Optimization (PPO), treating malicious-code samples as "chosen" data and benign code as "rejected" data during reinforcement learning. Extensive experiments demonstrate that RAWG significantly outperforms current state-of-the-art methods in both payload diversity and escape effectiveness. 

---
# LTM3D: Bridging Token Spaces for Conditional 3D Generation with Auto-Regressive Diffusion Framework 

**Authors**: Xin Kang, Zihan Zheng, Lei Chu, Yue Gao, Jiahao Li, Hao Pan, Xuejin Chen, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24245)  

**Abstract**: We present LTM3D, a Latent Token space Modeling framework for conditional 3D shape generation that integrates the strengths of diffusion and auto-regressive (AR) models. While diffusion-based methods effectively model continuous latent spaces and AR models excel at capturing inter-token dependencies, combining these paradigms for 3D shape generation remains a challenge. To address this, LTM3D features a Conditional Distribution Modeling backbone, leveraging a masked autoencoder and a diffusion model to enhance token dependency learning. Additionally, we introduce Prefix Learning, which aligns condition tokens with shape latent tokens during generation, improving flexibility across modalities. We further propose a Latent Token Reconstruction module with Reconstruction-Guided Sampling to reduce uncertainty and enhance structural fidelity in generated shapes. Our approach operates in token space, enabling support for multiple 3D representations, including signed distance fields, point clouds, meshes, and 3D Gaussian Splatting. Extensive experiments on image- and text-conditioned shape generation tasks demonstrate that LTM3D outperforms existing methods in prompt fidelity and structural accuracy while offering a generalizable framework for multi-modal, multi-representation 3D generation. 

---
# An Adversary-Resistant Multi-Agent LLM System via Credibility Scoring 

**Authors**: Sana Ebrahimi, Mohsen Dehghankar, Abolfazl Asudeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.24239)  

**Abstract**: While multi-agent LLM systems show strong capabilities in various domains, they are highly vulnerable to adversarial and low-performing agents. To resolve this issue, in this paper, we introduce a general and adversary-resistant multi-agent LLM framework based on credibility scoring. We model the collaborative query-answering process as an iterative game, where the agents communicate and contribute to a final system output. Our system associates a credibility score that is used when aggregating the team outputs. The credibility scores are learned gradually based on the past contributions of each agent in query answering. Our experiments across multiple tasks and settings demonstrate our system's effectiveness in mitigating adversarial influence and enhancing the resilience of multi-agent cooperation, even in the adversary-majority settings. 

---
# From Hallucinations to Jailbreaks: Rethinking the Vulnerability of Large Foundation Models 

**Authors**: Haibo Jin, Peiyan Zhang, Peiran Wang, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24232)  

**Abstract**: Large foundation models (LFMs) are susceptible to two distinct vulnerabilities: hallucinations and jailbreak attacks. While typically studied in isolation, we observe that defenses targeting one often affect the other, hinting at a deeper connection.
We propose a unified theoretical framework that models jailbreaks as token-level optimization and hallucinations as attention-level optimization. Within this framework, we establish two key propositions: (1) \textit{Similar Loss Convergence} - the loss functions for both vulnerabilities converge similarly when optimizing for target-specific outputs; and (2) \textit{Gradient Consistency in Attention Redistribution} - both exhibit consistent gradient behavior driven by shared attention dynamics.
We validate these propositions empirically on LLaVA-1.5 and MiniGPT-4, showing consistent optimization trends and aligned gradients. Leveraging this connection, we demonstrate that mitigation techniques for hallucinations can reduce jailbreak success rates, and vice versa. Our findings reveal a shared failure mode in LFMs and suggest that robustness strategies should jointly address both vulnerabilities. 

---
# Dynamic Malware Classification of Windows PE Files using CNNs and Greyscale Images Derived from Runtime API Call Argument Conversion 

**Authors**: Md Shahnawaz, Bishwajit Prasad Gond, Durga Prasad Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2505.24231)  

**Abstract**: Malware detection and classification remains a topic of concern for cybersecurity, since it is becoming common for attackers to use advanced obfuscation on their malware to stay undetected. Conventional static analysis is not effective against polymorphic and metamorphic malware as these change their appearance without modifying their behavior, thus defying the analysis by code structure alone. This makes it important to use dynamic detection that monitors malware behavior at runtime. In this paper, we present a dynamic malware categorization framework that extracts API argument calls at the runtime execution of Windows Portable Executable (PE) files. Extracting and encoding the dynamic features of API names, argument return values, and other relative features, we convert raw behavioral data to temporal patterns. To enhance feature portrayal, the generated patterns are subsequently converted into grayscale pictures using a magma colormap. These improved photos are used to teach a Convolutional Neural Network (CNN) model discriminative features, which allows for reliable and accurate malware classification. Results from experiments indicate that our method, with an average accuracy of 98.36% is effective in classifying different classes of malware and benign by integrating dynamic analysis and deep learning. It not only achieves high classification accuracy but also demonstrates significant resilience against typical evasion strategies. 

---
# Reasoning Can Hurt the Inductive Abilities of Large Language Models 

**Authors**: Haibo Jin, Peiyan Zhang, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24225)  

**Abstract**: Large Language Models (LLMs) have shown remarkable progress across domains, yet their ability to perform inductive reasoning - inferring latent rules from sparse examples - remains limited. It is often assumed that chain-of-thought (CoT) prompting, as used in Large Reasoning Models (LRMs), enhances such reasoning. We investigate this assumption with creating four controlled, diagnostic game-based tasks - chess, Texas Hold'em, dice games, and blackjack - with hidden human-defined rules. We find that CoT reasoning can degrade inductive performance, with LRMs often underperforming their non-reasoning counterparts.
To explain this, we present a theoretical framework that reveals how reasoning steps can amplify error through three failure modes: incorrect sub-task decomposition, incorrect sub-task solving, and incorrect final answer summarization. Based on our theoretical and empirical analysis, we introduce structured interventions that adapt CoT generation according to our identified failure types. These interventions improve inductive accuracy without retraining. Our findings suggest that effective (CoT) reasoning depends not only on taking more steps but also on ensuring those steps are well-structured. 

---
# Benchmarking Foundation Models for Zero-Shot Biometric Tasks 

**Authors**: Redwan Sony, Parisa Farmanifard, Hamzeh Alzwairy, Nitish Shukla, Arun Ross  

**Link**: [PDF](https://arxiv.org/pdf/2505.24214)  

**Abstract**: The advent of foundation models, particularly Vision-Language Models (VLMs) and Multi-modal Large Language Models (MLLMs), has redefined the frontiers of artificial intelligence, enabling remarkable generalization across diverse tasks with minimal or no supervision. Yet, their potential in biometric recognition and analysis remains relatively underexplored. In this work, we introduce a comprehensive benchmark that evaluates the zero-shot and few-shot performance of state-of-the-art publicly available VLMs and MLLMs across six biometric tasks spanning the face and iris modalities: face verification, soft biometric attribute prediction (gender and race), iris recognition, presentation attack detection (PAD), and face manipulation detection (morphs and deepfakes). A total of 41 VLMs were used in this evaluation. Experiments show that embeddings from these foundation models can be used for diverse biometric tasks with varying degrees of success. For example, in the case of face verification, a True Match Rate (TMR) of 96.77 percent was obtained at a False Match Rate (FMR) of 1 percent on the Labeled Face in the Wild (LFW) dataset, without any fine-tuning. In the case of iris recognition, the TMR at 1 percent FMR on the IITD-R-Full dataset was 97.55 percent without any fine-tuning. Further, we show that applying a simple classifier head to these embeddings can help perform DeepFake detection for faces, Presentation Attack Detection (PAD) for irides, and extract soft biometric attributes like gender and ethnicity from faces with reasonably high accuracy. This work reiterates the potential of pretrained models in achieving the long-term vision of Artificial General Intelligence. 

---
# Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows 

**Authors**: Orlando Marquez Ayala, Patrice Bechard, Emily Chen, Maggie Baird, Jingfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24189)  

**Abstract**: Large Language Models (LLMs) such as GPT-4o can handle a wide range of complex tasks with the right prompt. As per token costs are reduced, the advantages of fine-tuning Small Language Models (SLMs) for real-world applications -- faster inference, lower costs -- may no longer be clear. In this work, we present evidence that, for domain-specific tasks that require structured outputs, SLMs still have a quality advantage. We compare fine-tuning an SLM against prompting LLMs on the task of generating low-code workflows in JSON form. We observe that while a good prompt can yield reasonable results, fine-tuning improves quality by 10% on average. We also perform systematic error analysis to reveal model limitations. 

---
# Towards Unified Modeling in Federated Multi-Task Learning via Subspace Decoupling 

**Authors**: Yipan Wei, Yuchen Zou, Yapeng Li, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.24185)  

**Abstract**: Federated Multi-Task Learning (FMTL) enables multiple clients performing heterogeneous tasks without exchanging their local data, offering broad potential for privacy preserving multi-task collaboration. However, most existing methods focus on building personalized models for each client and unable to support the aggregation of multiple heterogeneous tasks into a unified model. As a result, in real-world scenarios where task objectives, label spaces, and optimization paths vary significantly, conventional FMTL methods struggle to achieve effective joint training. To address this challenge, we propose FedDEA (Federated Decoupled Aggregation), an update-structure-aware aggregation method specifically designed for multi-task model integration. Our method dynamically identifies task-relevant dimensions based on the response strength of local updates and enhances their optimization effectiveness through rescaling. This mechanism effectively suppresses cross-task interference and enables task-level decoupled aggregation within a unified global model. FedDEA does not rely on task labels or architectural modifications, making it broadly applicable and deployment-friendly. Experimental results demonstrate that it can be easily integrated into various mainstream federated optimization algorithms and consistently delivers significant overall performance improvements on widely used NYUD-V2 and PASCAL-Context. These results validate the robustness and generalization capabilities of FedDEA under highly heterogeneous task settings. 

---
# Seeing is Not Reasoning: MVPBench for Graph-based Evaluation of Multi-path Visual Physical CoT 

**Authors**: Zhuobai Dong, Junchao Yi, Ziyuan Zheng, Haochen Han, Xiangxi Zheng, Alex Jinpeng Wang, Fangming Liu, Linjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24182)  

**Abstract**: Understanding the physical world - governed by laws of motion, spatial relations, and causality - poses a fundamental challenge for multimodal large language models (MLLMs). While recent advances such as OpenAI o3 and GPT-4o demonstrate impressive perceptual and reasoning capabilities, our investigation reveals these models struggle profoundly with visual physical reasoning, failing to grasp basic physical laws, spatial interactions, and causal effects in complex scenes. More importantly, they often fail to follow coherent reasoning chains grounded in visual evidence, especially when multiple steps are needed to arrive at the correct answer. To rigorously evaluate this capability, we introduce MVPBench, a curated benchmark designed to rigorously evaluate visual physical reasoning through the lens of visual chain-of-thought (CoT). Each example features interleaved multi-image inputs and demands not only the correct final answer but also a coherent, step-by-step reasoning path grounded in evolving visual cues. This setup mirrors how humans reason through real-world physical processes over time. To ensure fine-grained evaluation, we introduce a graph-based CoT consistency metric that verifies whether the reasoning path of model adheres to valid physical logic. Additionally, we minimize shortcut exploitation from text priors, encouraging models to rely on visual understanding. Experimental results reveal a concerning trend: even cutting-edge MLLMs exhibit poor visual reasoning accuracy and weak image-text alignment in physical domains. Surprisingly, RL-based post-training alignment - commonly believed to improve visual reasoning performance - often harms spatial reasoning, suggesting a need to rethink current fine-tuning practices. 

---
# SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling 

**Authors**: Xiaodong Ji, Hailin Zhang, Fangcheng Fu, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.24179)  

**Abstract**: Many advanced Large Language Model (LLM) applications require long-context processing, but the self-attention module becomes a bottleneck during the prefilling stage of inference due to its quadratic time complexity with respect to sequence length. Existing sparse attention methods accelerate attention computation by skipping less significant regions of the attention map. However, these approaches typically perform coarse-grained inspection of the attention map, rendering considerable loss in model accuracy. In this paper, we propose SALE, a fine-grained sparse attention method that accelerates the long-context prefilling stage of LLM with negligible loss in model accuracy. SALE achieves fast and accurate fine-grained attention weight estimation through 4-bit quantized query-key products, followed by block-sparse attention to accelerate prefilling computations. For importance evaluation for query-key pairs, we adopt our Relative Attention Score metric, which offers significantly higher efficiency within our framework. We implement a custom CUDA kernel optimized for our approach for hardware efficiency, reducing the additional overhead to approximately 11% of the full attention latency. Notably, SALE requires no parameter training and can be seamlessly integrated into existing systems with trivial code modifications. Experiments on long-context benchmarks demonstrate that our method outperforms existing approaches in accuracy-efficiency trade-offs, achieving at least 3.36x speedups on Llama-3.1-8B for sequences longer than 64K while maintaining model quality. 

---
# Invariant Link Selector for Spatial-Temporal Out-of-Distribution Problem 

**Authors**: Katherine Tieu, Dongqi Fu, Jun Wu, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.24178)  

**Abstract**: In the era of foundation models, Out-of- Distribution (OOD) problems, i.e., the data discrepancy between the training environments and testing environments, hinder AI generalization. Further, relational data like graphs disobeying the Independent and Identically Distributed (IID) condition makes the problem more challenging, especially much harder when it is associated with time. Motivated by this, to realize the robust invariant learning over temporal graphs, we want to investigate what components in temporal graphs are most invariant and representative with respect to labels. With the Information Bottleneck (IB) method, we propose an error-bounded Invariant Link Selector that can distinguish invariant components and variant components during the training process to make the deep learning model generalizable for different testing scenarios. Besides deriving a series of rigorous generalizable optimization functions, we also equip the training with task-specific loss functions, e.g., temporal link prediction, to make pretrained models solve real-world application tasks like citation recommendation and merchandise recommendation, as demonstrated in our experiments with state-of-the-art (SOTA) methods. Our code is available at this https URL. 

---
# LKD-KGC: Domain-Specific KG Construction via LLM-driven Knowledge Dependency Parsing 

**Authors**: Jiaqi Sun, Shiyou Qian, Zhangchi Han, Wei Li, Zelin Qian, Dingyu Yang, Jian Cao, Guangtao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.24163)  

**Abstract**: Knowledge Graphs (KGs) structure real-world entities and their relationships into triples, enhancing machine reasoning for various tasks. While domain-specific KGs offer substantial benefits, their manual construction is often inefficient and requires specialized knowledge. Recent approaches for knowledge graph construction (KGC) based on large language models (LLMs), such as schema-guided KGC and reference knowledge integration, have proven efficient. However, these methods are constrained by their reliance on manually defined schema, single-document processing, and public-domain references, making them less effective for domain-specific corpora that exhibit complex knowledge dependencies and specificity, as well as limited reference knowledge. To address these challenges, we propose LKD-KGC, a novel framework for unsupervised domain-specific KG construction. LKD-KGC autonomously analyzes document repositories to infer knowledge dependencies, determines optimal processing sequences via LLM driven prioritization, and autoregressively generates entity schema by integrating hierarchical inter-document contexts. This schema guides the unsupervised extraction of entities and relationships, eliminating reliance on predefined structures or external knowledge. Extensive experiments show that compared with state-of-the-art baselines, LKD-KGC generally achieves improvements of 10% to 20% in both precision and recall rate, demonstrating its potential in constructing high-quality domain-specific KGs. 

---
# Don't Just Follow MLLM Plans: Robust and Efficient Planning for Open-world Agents 

**Authors**: Seungjoon Lee, Suhwan Kim, Minhyeon Oh, Youngsik Yoon, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2505.24157)  

**Abstract**: Developing autonomous agents capable of mastering complex, multi-step tasks in unpredictable, interactive environments presents a significant challenge. While Large Language Models (LLMs) offer promise for planning, existing approaches often rely on problematic internal knowledge or make unrealistic environmental assumptions. Although recent work explores learning planning knowledge, they still retain limitations due to partial reliance on external knowledge or impractical setups. Indeed, prior research has largely overlooked developing agents capable of acquiring planning knowledge from scratch, directly in realistic settings. While realizing this capability is necessary, it presents significant challenges, primarily achieving robustness given the substantial risk of incorporating LLMs' inaccurate knowledge. Moreover, efficiency is crucial for practicality as learning can demand prohibitive exploration. In response, we introduce Robust and Efficient Planning for Open-world Agents (REPOA), a novel framework designed to tackle these issues. REPOA features three key components: adaptive dependency learning and fine-grained failure-aware operation memory to enhance robustness to knowledge inaccuracies, and difficulty-based exploration to improve learning efficiency. Our evaluation in two established open-world testbeds demonstrates REPOA's robust and efficient planning, showcasing its capability to successfully obtain challenging late-game items that were beyond the reach of prior approaches. 

---
# RCCDA: Adaptive Model Updates in the Presence of Concept Drift under a Constrained Resource Budget 

**Authors**: Adam Piaseczny, Md Kamran Chowdhury Shisher, Shiqiang Wang, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2505.24149)  

**Abstract**: Machine learning (ML) algorithms deployed in real-world environments are often faced with the challenge of adapting models to concept drift, where the task data distributions are shifting over time. The problem becomes even more difficult when model performance must be maintained under adherence to strict resource constraints. Existing solutions often depend on drift-detection methods that produce high computational overhead for resource-constrained environments, and fail to provide strict guarantees on resource usage or theoretical performance assurances. To address these shortcomings, we propose RCCDA: a dynamic model update policy that optimizes ML training dynamics while ensuring strict compliance to predefined resource constraints, utilizing only past loss information and a tunable drift threshold. In developing our policy, we analytically characterize the evolution of model loss under concept drift with arbitrary training update decisions. Integrating these results into a Lyapunov drift-plus-penalty framework produces a lightweight policy based on a measurable accumulated loss threshold that provably limits update frequency and cost. Experimental results on three domain generalization datasets demonstrate that our policy outperforms baseline methods in inference accuracy while adhering to strict resource constraints under several schedules of concept drift, making our solution uniquely suited for real-time ML deployments. 

---
# The Butterfly Effect in Pathology: Exploring Security in Pathology Foundation Models 

**Authors**: Jiashuai Liu, Yingjia Shang, Yingkang Zhan, Di Zhang, Yi Niu, Dong Wei, Xian Wu, Zeyu Gao, Chen Li, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.24141)  

**Abstract**: With the widespread adoption of pathology foundation models in both research and clinical decision support systems, exploring their security has become a critical concern. However, despite their growing impact, the vulnerability of these models to adversarial attacks remains largely unexplored. In this work, we present the first systematic investigation into the security of pathology foundation models for whole slide image~(WSI) analysis against adversarial attacks. Specifically, we introduce the principle of \textit{local perturbation with global impact} and propose a label-free attack framework that operates without requiring access to downstream task labels. Under this attack framework, we revise four classical white-box attack methods and redefine the perturbation budget based on the characteristics of WSI. We conduct comprehensive experiments on three representative pathology foundation models across five datasets and six downstream tasks. Despite modifying only 0.1\% of patches per slide with imperceptible noise, our attack leads to downstream accuracy degradation that can reach up to 20\% in the worst cases. Furthermore, we analyze key factors that influence attack success, explore the relationship between patch-level vulnerability and semantic content, and conduct a preliminary investigation into potential defence strategies. These findings lay the groundwork for future research on the adversarial robustness and reliable deployment of pathology foundation models. Our code is publicly available at: this https URL. 

---
# S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Modelwith Spatio-Temporal Visual Representation 

**Authors**: Yichen Xie, Runsheng Xu, Tong He, Jyh-Jing Hwang, Katie Luo, Jingwei Ji, Hubert Lin, Letian Chen, Yiren Lu, Zhaoqi Leng, Dragomir Anguelov, Mingxing Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24139)  

**Abstract**: The latest advancements in multi-modal large language models (MLLMs) have spurred a strong renewed interest in end-to-end motion planning approaches for autonomous driving. Many end-to-end approaches rely on human annotations to learn intermediate perception and prediction tasks, while purely self-supervised approaches--which directly learn from sensor inputs to generate planning trajectories without human annotations often underperform the state of the art. We observe a key gap in the input representation space: end-to-end approaches built on MLLMs are often pretrained with reasoning tasks in 2D image space rather than the native 3D space in which autonomous vehicles plan. To this end, we propose S4-Driver, a scalable self-supervised motion planning algorithm with spatio-temporal visual representation, based on the popular PaLI multimodal large language model. S4-Driver uses a novel sparse volume strategy to seamlessly transform the strong visual representation of MLLMs from perspective view to 3D space without the need to finetune the vision encoder. This representation aggregates multi-view and multi-frame visual inputs and enables better prediction of planning trajectories in 3D space. To validate our method, we run experiments on both nuScenes and Waymo Open Motion Dataset (with in-house camera data). Results show that S4-Driver performs favorably against existing supervised multi-task approaches while requiring no human annotations. It also demonstrates great scalability when pretrained on large volumes of unannotated driving logs. 

---
# AMSbench: A Comprehensive Benchmark for Evaluating MLLM Capabilities in AMS Circuits 

**Authors**: Yichen Shi, Ze Zhang, Hongyang Wang, Zhuofu Tao, Zhongyi Li, Bingyu Chen, Yaxin Wang, Zhiping Yu, Ting-Jung Lin, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.24138)  

**Abstract**: Analog/Mixed-Signal (AMS) circuits play a critical role in the integrated circuit (IC) industry. However, automating Analog/Mixed-Signal (AMS) circuit design has remained a longstanding challenge due to its difficulty and complexity. Recent advances in Multi-modal Large Language Models (MLLMs) offer promising potential for supporting AMS circuit analysis and design. However, current research typically evaluates MLLMs on isolated tasks within the domain, lacking a comprehensive benchmark that systematically assesses model capabilities across diverse AMS-related challenges. To address this gap, we introduce AMSbench, a benchmark suite designed to evaluate MLLM performance across critical tasks including circuit schematic perception, circuit analysis, and circuit design. AMSbench comprises approximately 8000 test questions spanning multiple difficulty levels and assesses eight prominent models, encompassing both open-source and proprietary solutions such as Qwen 2.5-VL and Gemini 2.5 Pro. Our evaluation highlights significant limitations in current MLLMs, particularly in complex multi-modal reasoning and sophisticated circuit design tasks. These results underscore the necessity of advancing MLLMs' understanding and effective application of circuit-specific knowledge, thereby narrowing the existing performance gap relative to human expertise and moving toward fully automated AMS circuit design workflows. Our data is released at this https URL 

---
# Sparsity-Driven Parallel Imaging Consistency for Improved Self-Supervised MRI Reconstruction 

**Authors**: Yaşar Utku Alçalar, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.24136)  

**Abstract**: Physics-driven deep learning (PD-DL) models have proven to be a powerful approach for improved reconstruction of rapid MRI scans. In order to train these models in scenarios where fully-sampled reference data is unavailable, self-supervised learning has gained prominence. However, its application at high acceleration rates frequently introduces artifacts, compromising image fidelity. To mitigate this shortcoming, we propose a novel way to train PD-DL networks via carefully-designed perturbations. In particular, we enhance the k-space masking idea of conventional self-supervised learning with a novel consistency term that assesses the model's ability to accurately predict the added perturbations in a sparse domain, leading to more reliable and artifact-free reconstructions. The results obtained from the fastMRI knee and brain datasets show that the proposed training strategy effectively reduces aliasing artifacts and mitigates noise amplification at high acceleration rates, outperforming state-of-the-art self-supervised methods both visually and quantitatively. 

---
# R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration 

**Authors**: Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24133)  

**Abstract**: Reasoning models have demonstrated impressive performance in self-reflection and chain-of-thought reasoning. However, they often produce excessively long outputs, leading to prohibitively large key-value (KV) caches during inference. While chain-of-thought inference significantly improves performance on complex reasoning tasks, it can also lead to reasoning failures when deployed with existing KV cache compression approaches. To address this, we propose Redundancy-aware KV Cache Compression for Reasoning models (R-KV), a novel method specifically targeting redundant tokens in reasoning models. Our method preserves nearly 100% of the full KV cache performance using only 10% of the KV cache, substantially outperforming existing KV cache baselines, which reach only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache. This KV-cache reduction also leads to a 90% memory saving and a 6.6X throughput over standard chain-of-thought reasoning inference. Experimental results show that R-KV consistently outperforms existing KV cache compression baselines across two mathematical reasoning datasets. 

---
# CSVQA: A Chinese Multimodal Benchmark for Evaluating STEM Reasoning Capabilities of VLMs 

**Authors**: Ai Jian, Weijie Qiu, Xiaokun Wang, Peiyu Wang, Yunzhuo Hao, Jiangbo Pei, Yichen Wei, Yi Peng, Xuchen Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.24120)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable progress in multimodal understanding, yet their capabilities for scientific reasoning remains inadequately assessed. Current multimodal benchmarks predominantly evaluate generic image comprehension or text-driven reasoning, lacking authentic scientific contexts that require domain-specific knowledge integration with visual evidence analysis. To fill this gap, we present CSVQA, a diagnostic multimodal benchmark specifically designed for evaluating scientific reasoning through domain-grounded visual question this http URL benchmark features 1,378 carefully constructed question-answer pairs spanning diverse STEM disciplines, each demanding domain knowledge, integration of visual evidence, and higher-order reasoning. Compared to prior multimodal benchmarks, CSVQA places greater emphasis on real-world scientific content and complex this http URL additionally propose a rigorous evaluation protocol to systematically assess whether model predictions are substantiated by valid intermediate reasoning steps based on curated explanations. Our comprehensive evaluation of 15 VLMs on this benchmark reveals notable performance disparities, as even the top-ranked proprietary model attains only 49.6\% this http URL empirical evidence underscores the pressing need for advancing scientific reasoning capabilities in VLMs. Our CSVQA is released at this https URL. 

---
# Attractor learning for spatiotemporally chaotic dynamical systems using echo state networks with transfer learning 

**Authors**: Mohammad Shah Alam, William Ott, Ilya Timofeyev  

**Link**: [PDF](https://arxiv.org/pdf/2505.24099)  

**Abstract**: In this paper, we explore the predictive capabilities of echo state networks (ESNs) for the generalized Kuramoto-Sivashinsky (gKS) equation, an archetypal nonlinear PDE that exhibits spatiotemporal chaos. We introduce a novel methodology that integrates ESNs with transfer learning, aiming to enhance predictive performance across various parameter regimes of the gKS model. Our research focuses on predicting changes in long-term statistical patterns of the gKS model that result from varying the dispersion relation or the length of the spatial domain. We use transfer learning to adapt ESNs to different parameter settings and successfully capture changes in the underlying chaotic attractor. 

---
# Searching Clinical Data Using Generative AI 

**Authors**: Karan Hanswadkar, Anika Kanchi, Shivani Tripathi, Shi Qiao, Rony Chatterjee, Alekh Jindal  

**Link**: [PDF](https://arxiv.org/pdf/2505.24090)  

**Abstract**: Artificial Intelligence (AI) is making a major impact on healthcare, particularly through its application in natural language processing (NLP) and predictive analytics. The healthcare sector has increasingly adopted AI for tasks such as clinical data analysis and medical code assignment. However, searching for clinical information in large and often unorganized datasets remains a manual and error-prone process. Assisting this process with automations can help physicians improve their operational productivity significantly.
In this paper, we present a generative AI approach, coined SearchAI, to enhance the accuracy and efficiency of searching clinical data. Unlike traditional code assignment, which is a one-to-one problem, clinical data search is a one-to-many problem, i.e., a given search query can map to a family of codes. Healthcare professionals typically search for groups of related diseases, drugs, or conditions that map to many codes, and therefore, they need search tools that can handle keyword synonyms, semantic variants, and broad open-ended queries. SearchAI employs a hierarchical model that respects the coding hierarchy and improves the traversal of relationships from parent to child nodes. SearchAI navigates these hierarchies predictively and ensures that all paths are reachable without losing any relevant nodes.
To evaluate the effectiveness of SearchAI, we conducted a series of experiments using both public and production datasets. Our results show that SearchAI outperforms default hierarchical traversals across several metrics, including accuracy, robustness, performance, and scalability. SearchAI can help make clinical data more accessible, leading to streamlined workflows, reduced administrative burden, and enhanced coding and diagnostic accuracy. 

---
# DSR-Bench: Evaluating the Structural Reasoning Abilities of LLMs via Data Structures 

**Authors**: Yu He, Yingxi Li, Colin White, Ellen Vitercik  

**Link**: [PDF](https://arxiv.org/pdf/2505.24069)  

**Abstract**: Large language models (LLMs) are increasingly deployed for real-world tasks that fundamentally involve data manipulation. A core requirement across these tasks is the ability to perform structural reasoning--that is, to understand and reason about data relationships. For example, customer requests require a temporal ordering, which can be represented by data structures such as queues. However, existing benchmarks primarily focus on high-level, application-driven evaluations without isolating this fundamental capability. To address this gap, we introduce DSR-Bench, a novel benchmark evaluating LLMs' structural reasoning capabilities through data structures, which provide interpretable representations of data relationships. DSR-Bench includes 20 data structures, 35 operations, and 4,140 problem instances, organized hierarchically for fine-grained analysis of reasoning limitations. Our evaluation pipeline is fully automated and deterministic, eliminating subjective human or model-based judgments. Its synthetic nature also ensures scalability and minimizes data contamination risks. We benchmark nine state-of-the-art LLMs. Our analysis shows that instruction-tuned models struggle with basic multi-attribute and multi-hop reasoning. Furthermore, while reasoning-oriented models perform better, they remain fragile on complex and hybrid structures, with the best model achieving an average score of only 47% on the challenge subset. Crucially, models often perform poorly on multi-dimensional data and natural language task descriptions, highlighting a critical gap for real-world deployment. 

---
# Bridging Source and Target Domains via Link Prediction for Unsupervised Domain Adaptation on Graphs 

**Authors**: Yilong Wang, Tianxiang Zhao, Zongyu Wu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24055)  

**Abstract**: Graph neural networks (GNNs) have shown great ability for node classification on graphs. However, the success of GNNs relies on abundant labeled data, while obtaining high-quality labels is costly and challenging, especially for newly emerging domains. Hence, unsupervised domain adaptation (UDA), which trains a classifier on the labeled source graph and adapts it to the unlabeled target graph, is attracting increasing attention. Various approaches have been proposed to alleviate the distribution shift between the source and target graphs to facilitate the classifier adaptation. However, most of them simply adopt existing UDA techniques developed for independent and identically distributed data to gain domain-invariant node embeddings for graphs, which do not fully consider the graph structure and message-passing mechanism of GNNs during the adaptation and will fail when label distribution shift exists among domains. In this paper, we proposed a novel framework that adopts link prediction to connect nodes between source and target graphs, which can facilitate message-passing between the source and target graphs and augment the target nodes to have ``in-distribution'' neighborhoods with the source domain. This strategy modified the target graph on the input level to reduce its deviation from the source domain in the embedding space and is insensitive to disproportional label distributions across domains. To prevent the loss of discriminative information in the target graph, we further design a novel identity-preserving learning objective, which guides the learning of the edge insertion module together with reconstruction and adaptation losses. Experimental results on real-world datasets demonstrate the effectiveness of our framework. 

---
# MedPAIR: Measuring Physicians and AI Relevance Alignment in Medical Question Answering 

**Authors**: Yuexing Hao, Kumail Alhamoud, Hyewon Jeong, Haoran Zhang, Isha Puri, Philip Torr, Mike Schaekermann, Ariel D. Stern, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24040)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance on various medical question-answering (QA) benchmarks, including standardized medical exams. However, correct answers alone do not ensure correct logic, and models may reach accurate conclusions through flawed processes. In this study, we introduce the MedPAIR (Medical Dataset Comparing Physicians and AI Relevance Estimation and Question Answering) dataset to evaluate how physician trainees and LLMs prioritize relevant information when answering QA questions. We obtain annotations on 1,300 QA pairs from 36 physician trainees, labeling each sentence within the question components for relevance. We compare these relevance estimates to those for LLMs, and further evaluate the impact of these "relevant" subsets on downstream task performance for both physician trainees and LLMs. We find that LLMs are frequently not aligned with the content relevance estimates of physician trainees. After filtering out physician trainee-labeled irrelevant sentences, accuracy improves for both the trainees and the LLMs. All LLM and physician trainee-labeled data are available at: this http URL. 

---
# LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Trainin 

**Authors**: Bo Wu, Sid Wang, Yunhao Tang, Jia Ding, Eryk Helenowski, Liang Tan, Tengyu Xu, Tushar Gowda, Zhengxing Chen, Chen Zhu, Xiaocheng Tang, Yundi Qian, Beibei Zhu, Rui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.24034)  

**Abstract**: Reinforcement Learning (RL) has become the most effective post-training approach for improving the capabilities of Large Language Models (LLMs). In practice, because of the high demands on latency and memory, it is particularly challenging to develop an efficient RL framework that reliably manages policy models with hundreds to thousands of billions of parameters.
In this paper, we present LlamaRL, a fully distributed, asynchronous RL framework optimized for efficient training of large-scale LLMs with various model sizes (8B, 70B, and 405B parameters) on GPU clusters ranging from a handful to thousands of devices. LlamaRL introduces a streamlined, single-controller architecture built entirely on native PyTorch, enabling modularity, ease of use, and seamless scalability to thousands of GPUs. We also provide a theoretical analysis of LlamaRL's efficiency, including a formal proof that its asynchronous design leads to strict RL speed-up. Empirically, by leveraging best practices such as colocated model offloading, asynchronous off-policy training, and distributed direct memory access for weight synchronization, LlamaRL achieves significant efficiency gains -- up to 10.7x speed-up compared to DeepSpeed-Chat-like systems on a 405B-parameter policy model. Furthermore, the efficiency advantage continues to grow with increasing model scale, demonstrating the framework's suitability for future large-scale RL training. 

---
# From Images to Signals: Are Large Vision Models Useful for Time Series Analysis? 

**Authors**: Ziming Zhao, ChengAo Shen, Hanghang Tong, Dongjin Song, Zhigang Deng, Qingsong Wen, Jingchao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2505.24030)  

**Abstract**: Transformer-based models have gained increasing attention in time series research, driving interest in Large Language Models (LLMs) and foundation models for time series analysis. As the field moves toward multi-modality, Large Vision Models (LVMs) are emerging as a promising direction. In the past, the effectiveness of Transformer and LLMs in time series has been debated. When it comes to LVMs, a similar question arises: are LVMs truely useful for time series analysis? To address it, we design and conduct the first principled study involving 4 LVMs, 8 imaging methods, 18 datasets and 26 baselines across both high-level (classification) and low-level (forecasting) tasks, with extensive ablation analysis. Our findings indicate LVMs are indeed useful for time series classification but face challenges in forecasting. Although effective, the contemporary best LVM forecasters are limited to specific types of LVMs and imaging methods, exhibit a bias toward forecasting periods, and have limited ability to utilize long look-back windows. We hope our findings could serve as a cornerstone for future research on LVM- and multimodal-based solutions to different time series tasks. 

---
# MaskAdapt: Unsupervised Geometry-Aware Domain Adaptation Using Multimodal Contextual Learning and RGB-Depth Masking 

**Authors**: Numair Nadeem, Muhammad Hamza Asad, Saeed Anwar, Abdul Bais  

**Link**: [PDF](https://arxiv.org/pdf/2505.24026)  

**Abstract**: Semantic segmentation of crops and weeds is crucial for site-specific farm management; however, most existing methods depend on labor intensive pixel-level annotations. A further challenge arises when models trained on one field (source domain) fail to generalize to new fields (target domain) due to domain shifts, such as variations in lighting, camera setups, soil composition, and crop growth stages. Unsupervised Domain Adaptation (UDA) addresses this by enabling adaptation without target-domain labels, but current UDA methods struggle with occlusions and visual blending between crops and weeds, leading to misclassifications in real-world conditions. To overcome these limitations, we introduce MaskAdapt, a novel approach that enhances segmentation accuracy through multimodal contextual learning by integrating RGB images with features derived from depth data. By computing depth gradients from depth maps, our method captures spatial transitions that help resolve texture ambiguities. These gradients, through a cross-attention mechanism, refines RGB feature representations, resulting in sharper boundary delineation. In addition, we propose a geometry-aware masking strategy that applies horizontal, vertical, and stochastic masks during training. This encourages the model to focus on the broader spatial context for robust visual recognition. Evaluations on real agricultural datasets demonstrate that MaskAdapt consistently outperforms existing State-of-the-Art (SOTA) UDA methods, achieving improved segmentation mean Intersection over Union (mIOU) across diverse field conditions. 

---
# DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models 

**Authors**: Chenbin Pan, Wenbin He, Zhengzhong Tu, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.24025)  

**Abstract**: The recent explosive interest in the reasoning capabilities of large language models, such as DeepSeek-R1, has demonstrated remarkable success through reinforcement learning-based fine-tuning frameworks, exemplified by methods like Group Relative Policy Optimization (GRPO). However, such reasoning abilities remain underexplored and notably absent in vision foundation models, including representation models like the DINO series. In this work, we propose \textbf{DINO-R1}, the first such attempt to incentivize visual in-context reasoning capabilities of vision foundation models using reinforcement learning. Specifically, DINO-R1 introduces \textbf{Group Relative Query Optimization (GRQO)}, a novel reinforcement-style training strategy explicitly designed for query-based representation models, which computes query-level rewards based on group-normalized alignment quality. We also apply KL-regularization to stabilize the objectness distribution to reduce the training instability. This joint optimization enables dense and expressive supervision across queries while mitigating overfitting and distributional drift. Building upon Grounding-DINO, we train a series of DINO-R1 family models that integrate a visual prompt encoder and a visual-guided query selection mechanism. Extensive experiments on COCO, LVIS, and ODinW demonstrate that DINO-R1 significantly outperforms supervised fine-tuning baselines, achieving strong generalization in both open-vocabulary and closed-set visual prompting scenarios. 

---
# Multi-Group Proportional Representation for Text-to-Image Models 

**Authors**: Sangwon Jung, Alex Oesterling, Claudio Mayrink Verdun, Sajani Vithana, Taesup Moon, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2505.24023)  

**Abstract**: Text-to-image (T2I) generative models can create vivid, realistic images from textual descriptions. As these models proliferate, they expose new concerns about their ability to represent diverse demographic groups, propagate stereotypes, and efface minority populations. Despite growing attention to the "safe" and "responsible" design of artificial intelligence (AI), there is no established methodology to systematically measure and control representational harms in image generation. This paper introduces a novel framework to measure the representation of intersectional groups in images generated by T2I models by applying the Multi-Group Proportional Representation (MPR) metric. MPR evaluates the worst-case deviation of representation statistics across given population groups in images produced by a generative model, allowing for flexible and context-specific measurements based on user requirements. We also develop an algorithm to optimize T2I models for this metric. Through experiments, we demonstrate that MPR can effectively measure representation statistics across multiple intersectional groups and, when used as a training objective, can guide models toward a more balanced generation across demographic groups while maintaining generation quality. 

---
# LLM Agents Should Employ Security Principles 

**Authors**: Kaiyuan Zhang, Zian Su, Pin-Yu Chen, Elisa Bertino, Xiangyu Zhang, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24019)  

**Abstract**: Large Language Model (LLM) agents show considerable promise for automating complex tasks using contextual reasoning; however, interactions involving multiple agents and the system's susceptibility to prompt injection and other forms of context manipulation introduce new vulnerabilities related to privacy leakage and system exploitation. This position paper argues that the well-established design principles in information security, which are commonly referred to as security principles, should be employed when deploying LLM agents at scale. Design principles such as defense-in-depth, least privilege, complete mediation, and psychological acceptability have helped guide the design of mechanisms for securing information systems over the last five decades, and we argue that their explicit and conscientious adoption will help secure agentic systems. To illustrate this approach, we introduce AgentSandbox, a conceptual framework embedding these security principles to provide safeguards throughout an agent's life-cycle. We evaluate with state-of-the-art LLMs along three dimensions: benign utility, attack utility, and attack success rate. AgentSandbox maintains high utility for its intended functions under both benign and adversarial evaluations while substantially mitigating privacy risks. By embedding secure design principles as foundational elements within emerging LLM agent protocols, we aim to promote trustworthy agent ecosystems aligned with user privacy expectations and evolving regulatory requirements. 

---
# Large Language Model Meets Constraint Propagation 

**Authors**: Alexandre Bonlarron, Florian Régin, Elisabetta De Maria, Jean-Charles Régin  

**Link**: [PDF](https://arxiv.org/pdf/2505.24012)  

**Abstract**: Large Language Models (LLMs) excel at generating fluent text but struggle to enforce external constraints because they generate tokens sequentially without explicit control mechanisms. GenCP addresses this limitation by combining LLM predictions with Constraint Programming (CP) reasoning, formulating text generation as a Constraint Satisfaction Problem (CSP). In this paper, we improve GenCP by integrating Masked Language Models (MLMs) for domain generation, which allows bidirectional constraint propagation that leverages both past and future tokens. This integration bridges the gap between token-level prediction and structured constraint enforcement, leading to more reliable and constraint-aware text generation. Our evaluation on COLLIE benchmarks demonstrates that incorporating domain preview via MLM calls significantly improves GenCP's performance. Although this approach incurs additional MLM calls and, in some cases, increased backtracking, the overall effect is a more efficient use of LLM inferences and an enhanced ability to generate feasible and meaningful solutions, particularly in tasks with strict content constraints. 

---
# Diversity of Transformer Layers: One Aspect of Parameter Scaling Laws 

**Authors**: Hidetaka Kamigaito, Ying Zhang, Jingun Kwon, Katsuhiko Hayashi, Manabu Okumura, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2505.24009)  

**Abstract**: Transformers deliver outstanding performance across a wide range of tasks and are now a dominant backbone architecture for large language models (LLMs). Their task-solving performance is improved by increasing parameter size, as shown in the recent studies on parameter scaling laws. Although recent mechanistic-interpretability studies have deepened our understanding of the internal behavior of Transformers by analyzing their residual stream, the relationship between these internal mechanisms and the parameter scaling laws remains unclear. To bridge this gap, we focus on layers and their size, which mainly decide the parameter size of Transformers. For this purpose, we first theoretically investigate the layers within the residual stream through a bias-diversity decomposition. The decomposition separates (i) bias, the error of each layer's output from the ground truth, and (ii) diversity, which indicates how much the outputs of each layer differ from each other. Analyzing Transformers under this theory reveals that performance improves when individual layers make predictions close to the correct answer and remain mutually diverse. We show that diversity becomes especially critical when individual layers' outputs are far from the ground truth. Finally, we introduce an information-theoretic diversity and show our main findings that adding layers enhances performance only when those layers behave differently, i.e., are diverse. We also reveal the performance gains from increasing the number of layers exhibit submodularity: marginal improvements diminish as additional layers increase, mirroring the logarithmic convergence predicted by the parameter scaling laws. Experiments on multiple semantic-understanding tasks with various LLMs empirically confirm the theoretical properties derived in this study. 

---
# Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting 

**Authors**: ChengAo Shen, Wenchao Yu, Ziming Zhao, Dongjin Song, Wei Cheng, Haifeng Chen, Jingchao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2505.24003)  

**Abstract**: Time series, typically represented as numerical sequences, can also be transformed into images and texts, offering multi-modal views (MMVs) of the same underlying signal. These MMVs can reveal complementary patterns and enable the use of powerful pre-trained large models, such as large vision models (LVMs), for long-term time series forecasting (LTSF). However, as we identified in this work, applying LVMs to LTSF poses an inductive bias towards "forecasting periods". To harness this bias, we propose DMMV, a novel decomposition-based multi-modal view framework that leverages trend-seasonal decomposition and a novel backcast residual based adaptive decomposition to integrate MMVs for LTSF. Comparative evaluations against 14 state-of-the-art (SOTA) models across diverse datasets show that DMMV outperforms single-view and existing multi-modal baselines, achieving the best mean squared error (MSE) on 6 out of 8 benchmark datasets. 

---
# Multi-output Classification using a Cross-talk Architecture for Compound Fault Diagnosis of Motors in Partially Labeled Condition 

**Authors**: Wonjun Yi, Wonho Jung, Kangmin Jang, Yong-Hwa Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.24001)  

**Abstract**: The increasing complexity of rotating machinery and the diversity of operating conditions, such as rotating speed and varying torques, have amplified the challenges in fault diagnosis in scenarios requiring domain adaptation, particularly involving compound faults. This study addresses these challenges by introducing a novel multi-output classification (MOC) framework tailored for domain adaptation in partially labeled (PL) target datasets. Unlike conventional multi-class classification (MCC) approaches, the proposed MOC framework classifies the severity levels of compound faults simultaneously. Furthermore, we explore various single-task and multi-task architectures applicable to the MOC formulation-including shared trunk and cross-talk-based designs-for compound fault diagnosis under PL conditions. Based on this investigation, we propose a novel cross-talk layer structure that enables selective information sharing across diagnostic tasks, effectively enhancing classification performance in compound fault scenarios. In addition, frequency-layer normalization was incorporated to improve domain adaptation performance on motor vibration data. Compound fault conditions were implemented using a motor-based test setup, and the proposed model was evaluated across six domain adaptation scenarios. The experimental results demonstrate its superior macro F1 performance compared to baseline models. We further showed that the proposed mode's structural advantage is more pronounced in compound fault settings through a single-fault comparison. We also found that frequency-layer normalization fits the fault diagnosis task better than conventional methods. Lastly, we discuss that this improvement primarily stems from the model's structural ability to leverage inter-fault classification task interactions, rather than from a simple increase in model parameters. 

---
# Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs 

**Authors**: Yinong Oliver Wang, Nivedha Sivakumar, Falaah Arif Khan, Rin Metcalf Susa, Adam Golinski, Natalie Mackraz, Barry-John Theobald, Luca Zappella, Nicholas Apostoloff  

**Link**: [PDF](https://arxiv.org/pdf/2505.23996)  

**Abstract**: The recent rapid adoption of large language models (LLMs) highlights the critical need for benchmarking their fairness. Conventional fairness metrics, which focus on discrete accuracy-based evaluations (i.e., prediction correctness), fail to capture the implicit impact of model uncertainty (e.g., higher model confidence about one group over another despite similar accuracy). To address this limitation, we propose an uncertainty-aware fairness metric, UCerF, to enable a fine-grained evaluation of model fairness that is more reflective of the internal bias in model decisions compared to conventional fairness measures. Furthermore, observing data size, diversity, and clarity issues in current datasets, we introduce a new gender-occupation fairness evaluation dataset with 31,756 samples for co-reference resolution, offering a more diverse and suitable dataset for evaluating modern LLMs. We establish a benchmark, using our metric and dataset, and apply it to evaluate the behavior of ten open-source LLMs. For example, Mistral-7B exhibits suboptimal fairness due to high confidence in incorrect predictions, a detail overlooked by Equalized Odds but captured by UCerF. Overall, our proposed LLM benchmark, which evaluates fairness with uncertainty awareness, paves the way for developing more transparent and accountable AI systems. 

---
# Large Language Models for Controllable Multi-property Multi-objective Molecule Optimization 

**Authors**: Vishal Dey, Xiao Hu, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2505.23987)  

**Abstract**: In real-world drug design, molecule optimization requires selectively improving multiple molecular properties up to pharmaceutically relevant levels, while maintaining others that already meet such criteria. However, existing computational approaches and instruction-tuned LLMs fail to capture such nuanced property-specific objectives, limiting their practical applicability. To address this, we introduce C-MuMOInstruct, the first instruction-tuning dataset focused on multi-property optimization with explicit, property-specific objectives. Leveraging C-MuMOInstruct, we develop GeLLMO-Cs, a series of instruction-tuned LLMs that can perform targeted property-specific optimization. Our experiments across 5 in-distribution and 5 out-of-distribution tasks show that GeLLMO-Cs consistently outperform strong baselines, achieving up to 126% higher success rate. Notably, GeLLMO-Cs exhibit impressive 0-shot generalization to novel optimization tasks and unseen instructions. This offers a step toward a foundational LLM to support realistic, diverse optimizations with property-specific objectives. C-MuMOInstruct and code are accessible through this https URL. 

---
# VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL 

**Authors**: Yichen Feng, Zhangchen Xu, Fengqing Jiang, Yuetai Li, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.23977)  

**Abstract**: Vision language models (VLMs) are expected to perform effective multimodal reasoning and make logically coherent decisions, which is critical to tasks such as diagram understanding and spatial problem solving. However, current VLM reasoning lacks large-scale and well-structured training datasets. To bridge this gap, we propose VisualSphinx, a first-of-its-kind large-scale synthetic visual logical reasoning training data. To tackle the challenge of image synthesis with grounding answers, we propose a rule-to-image synthesis pipeline, which extracts and expands puzzle rules from seed questions and generates the code of grounding synthesis image synthesis for puzzle sample assembly. Experiments demonstrate that VLM trained using GRPO on VisualSphinx benefit from logical coherence and readability of our dataset and exhibit improved performance on logical reasoning tasks. The enhanced reasoning capabilities developed from VisualSphinx also benefit other reasoning tasks such as algebraic reasoning, arithmetic reasoning and geometry reasoning. 

---
# Confidential Guardian: Cryptographically Prohibiting the Abuse of Model Abstention 

**Authors**: Stephan Rabanser, Ali Shahin Shamsabadi, Olive Franzese, Xiao Wang, Adrian Weller, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2505.23968)  

**Abstract**: Cautious predictions -- where a machine learning model abstains when uncertain -- are crucial for limiting harmful errors in safety-critical applications. In this work, we identify a novel threat: a dishonest institution can exploit these mechanisms to discriminate or unjustly deny services under the guise of uncertainty. We demonstrate the practicality of this threat by introducing an uncertainty-inducing attack called Mirage, which deliberately reduces confidence in targeted input regions, thereby covertly disadvantaging specific individuals. At the same time, Mirage maintains high predictive performance across all data points. To counter this threat, we propose Confidential Guardian, a framework that analyzes calibration metrics on a reference dataset to detect artificially suppressed confidence. Additionally, it employs zero-knowledge proofs of verified inference to ensure that reported confidence scores genuinely originate from the deployed model. This prevents the provider from fabricating arbitrary model confidence values while protecting the model's proprietary details. Our results confirm that Confidential Guardian effectively prevents the misuse of cautious predictions, providing verifiable assurances that abstention reflects genuine model uncertainty rather than malicious intent. 

---
# Information Structure in Mappings: An Approach to Learning, Representation, and Generalisation 

**Authors**: Henry Conklin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23960)  

**Abstract**: Despite the remarkable success of large large-scale neural networks, we still lack unified notation for thinking about and describing their representational spaces. We lack methods to reliably describe how their representations are structured, how that structure emerges over training, and what kinds of structures are desirable. This thesis introduces quantitative methods for identifying systematic structure in a mapping between spaces, and leverages them to understand how deep-learning models learn to represent information, what representational structures drive generalisation, and how design decisions condition the structures that emerge. To do this I identify structural primitives present in a mapping, along with information theoretic quantifications of each. These allow us to analyse learning, structure, and generalisation across multi-agent reinforcement learning models, sequence-to-sequence models trained on a single task, and Large Language Models. I also introduce a novel, performant, approach to estimating the entropy of vector space, that allows this analysis to be applied to models ranging in size from 1 million to 12 billion parameters.
The experiments here work to shed light on how large-scale distributed models of cognition learn, while allowing us to draw parallels between those systems and their human analogs. They show how the structures of language and the constraints that give rise to them in many ways parallel the kinds of structures that drive performance of contemporary neural networks. 

---
# Enhancing LLM-Based Code Generation with Complexity Metrics: A Feedback-Driven Approach 

**Authors**: Melika Sepidband, Hamed Taherkhani, Song Wang, Hadi Hemmati  

**Link**: [PDF](https://arxiv.org/pdf/2505.23953)  

**Abstract**: Automatic code generation has gained significant momentum with the advent of Large Language Models (LLMs) such as GPT-4. Although many studies focus on improving the effectiveness of LLMs for code generation, very limited work tries to understand the generated code's characteristics and leverage that to improve failed cases. In this paper, as the most straightforward characteristic of code, we investigate the relationship between code complexity and the success of LLM generated code. Using a large set of standard complexity metrics, we first conduct an empirical analysis to explore their correlation with LLM's performance on code generation (i.e., Pass@1). Using logistic regression models, we identify which complexity metrics are most predictive of code correctness. Building on these findings, we propose an iterative feedback method, where LLMs are prompted to generate correct code based on complexity metrics from previous failed outputs. We validate our approach across multiple benchmarks (i.e., HumanEval, MBPP, LeetCode, and BigCodeBench) and various LLMs (i.e., GPT-4o, GPT-3.5 Turbo, Llama 3.1, and GPT-o3 mini), comparing the results with two baseline methods: (a) zero-shot generation, and (b) iterative execution-based feedback without our code complexity insights. Experiment results show that our approach makes notable improvements, particularly with a smaller LLM (GPT3.5 Turbo), where, e.g., Pass@1 increased by 35.71% compared to the baseline's improvement of 12.5% on the HumanEval dataset. The study expands experiments to BigCodeBench and integrates the method with the Reflexion code generation agent, leading to Pass@1 improvements of 20% (GPT-4o) and 23.07% (GPT-o3 mini). The results highlight that complexity-aware feedback enhances both direct LLM prompting and agent-based workflows. 

---
# TSENOR: Highly-Efficient Algorithm for Finding Transposable N:M Sparse Masks 

**Authors**: Xiang Meng, Mehdi Makni, Rahul Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2505.23949)  

**Abstract**: Network pruning reduces the computational requirements of large neural networks, with N:M sparsity -- retaining only N out of every M consecutive weights -- offering a compelling balance between compressed model quality and hardware acceleration. However, N:M sparsity only accelerates forward-pass computations, as N:M patterns are not preserved during matrix transposition, limiting efficiency during training where both passes are computationally intensive. While transposable N:M sparsity has been proposed to address this limitation, existing methods for finding transposable N:M sparse masks either fail to scale to large models or are restricted to M=4 which results in suboptimal compression-accuracy trade-off. We introduce an efficient solver for transposable N:M masks that scales to billion-parameter models. We formulate mask generation as optimal transport problems and solve through entropy regularization and Dykstra's algorithm, followed by a rounding procedure. Our tensor-based implementation exploits GPU parallelism, achieving up to 100x speedup with only 1-10% error compared to existing methods. Our approach can be integrated with layer-wise N:M pruning frameworks including Wanda, SparseGPT and ALPS to produce transposable N:M sparse models with arbitrary N:M values. Experiments show that LLaMA3.2-8B with transposable 16:32 sparsity maintains performance close to its standard N:M counterpart and outperforms standard 2:4 sparse model, showing the practical value of our approach. 

---
# Position: The Future of Bayesian Prediction Is Prior-Fitted 

**Authors**: Samuel Müller, Arik Reuter, Noah Hollmann, David Rügamer, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.23947)  

**Abstract**: Training neural networks on randomly generated artificial datasets yields Bayesian models that capture the prior defined by the dataset-generating distribution. Prior-data Fitted Networks (PFNs) are a class of methods designed to leverage this insight. In an era of rapidly increasing computational resources for pre-training and a near stagnation in the generation of new real-world data in many applications, PFNs are poised to play a more important role across a wide range of applications. They enable the efficient allocation of pre-training compute to low-data scenarios. Originally applied to small Bayesian modeling tasks, the field of PFNs has significantly expanded to address more complex domains and larger datasets. This position paper argues that PFNs and other amortized inference approaches represent the future of Bayesian inference, leveraging amortized learning to tackle data-scarce problems. We thus believe they are a fruitful area of research. In this position paper, we explore their potential and directions to address their current limitations. 

---
# A Closer Look at Bias and Chain-of-Thought Faithfulness of Large (Vision) Language Models 

**Authors**: Sriram Balasubramanian, Samyadeep Basu, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23945)  

**Abstract**: Chain-of-thought (CoT) reasoning enhances performance of large language models, but questions remain about whether these reasoning traces faithfully reflect the internal processes of the model. We present the first comprehensive study of CoT faithfulness in large vision-language models (LVLMs), investigating how both text-based and previously unexplored image-based biases affect reasoning and bias articulation. Our work introduces a novel, fine-grained evaluation pipeline for categorizing bias articulation patterns, enabling significantly more precise analysis of CoT reasoning than previous methods. This framework reveals critical distinctions in how models process and respond to different types of biases, providing new insights into LVLM CoT faithfulness. Our findings reveal that subtle image-based biases are rarely articulated compared to explicit text-based ones, even in models specialized for reasoning. Additionally, many models exhibit a previously unidentified phenomenon we term ``inconsistent'' reasoning - correctly reasoning before abruptly changing answers, serving as a potential canary for detecting biased reasoning from unfaithful CoTs. We then apply the same evaluation pipeline to revisit CoT faithfulness in LLMs across various levels of implicit cues. Our findings reveal that current language-only reasoning models continue to struggle with articulating cues that are not overtly stated. 

---
# SG-Blend: Learning an Interpolation Between Improved Swish and GELU for Robust Neural Representations 

**Authors**: Gaurav Sarkar, Jay Gala, Subarna Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23942)  

**Abstract**: The design of activation functions remains a pivotal component in optimizing deep neural networks. While prevailing choices like Swish and GELU demonstrate considerable efficacy, they often exhibit domain-specific optima. This work introduces SG-Blend, a novel activation function that blends our proposed SSwish, a first-order symmetric variant of Swish and the established GELU through dynamic interpolation. By adaptively blending these constituent functions via learnable parameters, SG-Blend aims to harness their complementary strengths: SSwish's controlled non-monotonicity and symmetry, and GELU's smooth, probabilistic profile, to achieve a more universally robust balance between model expressivity and gradient stability. We conduct comprehensive empirical evaluations across diverse modalities and architectures, showing performance improvements across all considered natural language and computer vision tasks and models. These results, achieved with negligible computational overhead, underscore SG-Blend's potential as a versatile, drop-in replacement that consistently outperforms strong contemporary baselines. The code is available at this https URL. 

---
# BIRD: Behavior Induction via Representation-structure Distillation 

**Authors**: Galen Pogoncheff, Michael Beyeler  

**Link**: [PDF](https://arxiv.org/pdf/2505.23933)  

**Abstract**: Human-aligned deep learning models exhibit behaviors consistent with human values, such as robustness, fairness, and honesty. Transferring these behavioral properties to models trained on different tasks or data distributions remains challenging: aligned behavior is easily forgotten during fine-tuning, and collecting task-specific data that preserves this behavior can be prohibitively costly. We introduce BIRD (Behavior Induction via Representation-structure Distillation), a flexible framework for transferring aligned behavior by matching the internal representation structure of a student model to that of a teacher. Applied to out-of-distribution robustness in image classification, BIRD outperforms fine-tuning, transfer learning, and continual learning methods, improving robust accuracy by up to 16% over the next strongest baseline. It remains effective even when the teacher is trained on a much simpler dataset and is $25 \times$ smaller than the student. In a large-scale study of over 400 teacher-student pairs, we show that three interpretable and computable properties of the teacher's representations (i.e., task relevance, behavioral relevance, and complementary knowledge) explain up to 85% of the variance in transfer success. These insights offer practical guidance for teacher selection and design. BIRD turns small, well-aligned models into scalable alignment seeds, removing a key bottleneck in deploying safe AI systems in the wild. 

---
# Scaling up the think-aloud method 

**Authors**: Daniel Wurgaft, Ben Prystawski, Kanishk Gandhi, Cedegao E. Zhang, Joshua B. Tenenbaum, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2505.23931)  

**Abstract**: The think-aloud method, where participants voice their thoughts as they solve a task, is a valuable source of rich data about human reasoning processes. Yet, it has declined in popularity in contemporary cognitive science, largely because labor-intensive transcription and annotation preclude large sample sizes. Here, we develop methods to automate the transcription and annotation of verbal reports of reasoning using natural language processing tools, allowing for large-scale analysis of think-aloud data. In our study, 640 participants thought aloud while playing the Game of 24, a mathematical reasoning task. We automatically transcribed the recordings and coded the transcripts as search graphs, finding moderate inter-rater reliability with humans. We analyze these graphs and characterize consistency and variation in human reasoning traces. Our work demonstrates the value of think-aloud data at scale and serves as a proof of concept for the automated analysis of verbal reports. 

---
# Exploring Societal Concerns and Perceptions of AI: A Thematic Analysis through the Lens of Problem-Seeking 

**Authors**: Naomi Omeonga wa Kayembe  

**Link**: [PDF](https://arxiv.org/pdf/2505.23930)  

**Abstract**: This study introduces a novel conceptual framework distinguishing problem-seeking from problem-solving to clarify the unique features of human intelligence in contrast to AI. Problem-seeking refers to the embodied, emotionally grounded process by which humans identify and set goals, while problem-solving denotes the execution of strategies aimed at achieving such predefined objectives. The framework emphasizes that while AI excels at efficiency and optimization, it lacks the orientation derived from experiential grounding and the embodiment flexibility intrinsic to human cognition. To empirically explore this distinction, the research analyzes metadata from 157 YouTube videos discussing AI. Conducting a thematic analysis combining qualitative insights with keyword-based quantitative metrics, this mixed-methods approach uncovers recurring themes in public discourse, including privacy, job displacement, misinformation, optimism, and ethical concerns. The results reveal a dual sentiment: public fascination with AI's capabilities coexists with anxiety and skepticism about its societal implications. The discussion critiques the orthogonality thesis, which posits that intelligence is separable from goal content, and instead argues that human intelligence integrates goal-setting and goal-pursuit. It underscores the centrality of embodied cognition in human reasoning and highlights how AI's limitations come from its current reliance on computational processing. The study advocates for enhancing emotional and digital literacy to foster responsible AI engagement. It calls for reframing public discourse to recognize AI as a tool that augments -- rather than replaces -- human intelligence. By positioning problem seeking at the core of cognition and as a critical dimension of intelligence, this research offers new perspectives on ethically aligned and human-centered AI development. 

---
# ChARM: Character-based Act-adaptive Reward Modeling for Advanced Role-Playing Language Agents 

**Authors**: Feiteng Fang, Ting-En Lin, Yuchuan Wu, Xiong Liu, Xiang Huang, Dingwei Chen, Jing Ye, Haonan Zhang, Liang Zhu, Hamid Alinejad-Rokny, Min Yang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23923)  

**Abstract**: Role-Playing Language Agents (RPLAs) aim to simulate characters for realistic and engaging human-computer interactions. However, traditional reward models often struggle with scalability and adapting to subjective conversational preferences. We propose ChARM, a Character-based Act-adaptive Reward Model, addressing these challenges through two innovations: (1) an act-adaptive margin that significantly enhances learning efficiency and generalizability, and (2) a self-evolution mechanism leveraging large-scale unlabeled data to improve training coverage. Additionally, we introduce RoleplayPref, the first large-scale preference dataset specifically for RPLAs, featuring 1,108 characters, 13 subcategories, and 16,888 bilingual dialogues, alongside RoleplayEval, a dedicated evaluation benchmark. Experimental results show a 13% improvement over the conventional Bradley-Terry model in preference rankings. Furthermore, applying ChARM-generated rewards to preference learning techniques (e.g., direct preference optimization) achieves state-of-the-art results on CharacterEval and RoleplayEval. Code and dataset are available at this https URL. 

---
# Representational Difference Explanations 

**Authors**: Neehar Kondapaneni, Oisin Mac Aodha, Pietro Perona  

**Link**: [PDF](https://arxiv.org/pdf/2505.23917)  

**Abstract**: We propose a method for discovering and visualizing the differences between two learned representations, enabling more direct and interpretable model comparisons. We validate our method, which we call Representational Differences Explanations (RDX), by using it to compare models with known conceptual differences and demonstrate that it recovers meaningful distinctions where existing explainable AI (XAI) techniques fail. Applied to state-of-the-art models on challenging subsets of the ImageNet and iNaturalist datasets, RDX reveals both insightful representational differences and subtle patterns in the data. Although comparison is a cornerstone of scientific analysis, current tools in machine learning, namely post hoc XAI methods, struggle to support model comparison effectively. Our work addresses this gap by introducing an effective and explainable tool for contrasting model representations. 

---
# Probing Association Biases in LLM Moderation Over-Sensitivity 

**Authors**: Yuxin Wang, Botao Yu, Ivory Yang, Saeed Hassanpour, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23914)  

**Abstract**: Large Language Models are widely used for content moderation but often misclassify benign comments as toxic, leading to over-sensitivity. While previous research attributes this issue primarily to the presence of offensive terms, we reveal a potential cause beyond token level: LLMs exhibit systematic topic biases in their implicit associations. Inspired by cognitive psychology's implicit association tests, we introduce Topic Association Analysis, a semantic-level approach to quantify how LLMs associate certain topics with toxicity. By prompting LLMs to generate free-form scenario imagination for misclassified benign comments and analyzing their topic amplification levels, we find that more advanced models (e.g., GPT-4 Turbo) demonstrate stronger topic stereotype despite lower overall false positive rates. These biases suggest that LLMs do not merely react to explicit, offensive language but rely on learned topic associations, shaping their moderation decisions. Our findings highlight the need for refinement beyond keyword-based filtering, providing insights into the underlying mechanisms driving LLM over-sensitivity. 

---
# Reinforcement Learning for Better Verbalized Confidence in Long-Form Generation 

**Authors**: Caiqi Zhang, Xiaochen Zhu, Chengzu Li, Nigel Collier, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2505.23912)  

**Abstract**: Hallucination remains a major challenge for the safe and trustworthy deployment of large language models (LLMs) in factual content generation. Prior work has explored confidence estimation as an effective approach to hallucination detection, but often relies on post-hoc self-consistency methods that require computationally expensive sampling. Verbalized confidence offers a more efficient alternative, but existing approaches are largely limited to short-form question answering (QA) tasks and do not generalize well to open-ended generation. In this paper, we propose LoVeC (Long-form Verbalized Confidence), an on-the-fly verbalized confidence estimation method for long-form generation. Specifically, we use reinforcement learning (RL) to train LLMs to append numerical confidence scores to each generated statement, serving as a direct and interpretable signal of the factuality of generation. Our experiments consider both on-policy and off-policy RL methods, including DPO, ORPO, and GRPO, to enhance the model calibration. We introduce two novel evaluation settings, free-form tagging and iterative tagging, to assess different verbalized confidence estimation methods. Experiments on three long-form QA datasets show that our RL-trained models achieve better calibration and generalize robustly across domains. Also, our method is highly efficient, as it only requires adding a few tokens to the output being decoded. 

---
# CNN-LSTM Hybrid Model for AI-Driven Prediction of COVID-19 Severity from Spike Sequences and Clinical Data 

**Authors**: Caio Cheohen, Vinnícius M. S. Gomes, Manuela L. da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2505.23879)  

**Abstract**: The COVID-19 pandemic, caused by SARS-CoV-2, highlighted the critical need for accurate prediction of disease severity to optimize healthcare resource allocation and patient management. The spike protein, which facilitates viral entry into host cells, exhibits high mutation rates, particularly in the receptor-binding domain, influencing viral pathogenicity. Artificial intelligence approaches, such as deep learning, offer promising solutions for leveraging genomic and clinical data to predict disease outcomes. Objective: This study aimed to develop a hybrid CNN-LSTM deep learning model to predict COVID-19 severity using spike protein sequences and associated clinical metadata from South American patients. Methods: We retrieved 9,570 spike protein sequences from the GISAID database, of which 3,467 met inclusion criteria after standardization. The dataset included 2,313 severe and 1,154 mild cases. A feature engineering pipeline extracted features from sequences, while demographic and clinical variables were one-hot encoded. A hybrid CNN-LSTM architecture was trained, combining CNN layers for local pattern extraction and an LSTM layer for long-term dependency modeling. Results: The model achieved an F1 score of 82.92%, ROC-AUC of 0.9084, precision of 83.56%, and recall of 82.85%, demonstrating robust classification performance. Training stabilized at 85% accuracy with minimal overfitting. The most prevalent lineages (P.1, AY.99.2) and clades (GR, GK) aligned with regional epidemiological trends, suggesting potential associations between viral genetics and clinical outcomes. Conclusion: The CNN-LSTM hybrid model effectively predicted COVID-19 severity using spike protein sequences and clinical data, highlighting the utility of AI in genomic surveillance and precision public health. Despite limitations, this approach provides a framework for early severity prediction in future outbreaks. 

---
# Actor-Critic based Online Data Mixing For Language Model Pre-Training 

**Authors**: Jing Ma, Chenhao Dang, Mingjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23878)  

**Abstract**: The coverage and composition of pretraining data significantly impacts the generalization ability of Large Language Models (LLMs). To reduce the carbon footprint and financial costs of training, some data mixing methods, which applied the optimized domain weights of a small proxy model to train a larger one, were proposed. However, these methods did not evolute with the training dynamics. The existing online data mixing (ODM) method addressed this limitation by applying the multi-armed bandit algorithm as data sampling strategy. Yet, it did not consider the intra-domain interactions. In this paper, we develop an actor-critic based online data mixing (AC-ODM) method, which captures the varying domain weights by auxiliary actor-critic networks and consider the intra-domain interactions with the reward function. While constructing the dataset to pretrain a large target LLM, we directly apply the actor, which is trained with a small proxy LLM as the environment, as the sampling strategy. The transfer of sampling strategy can not only ensure the efficiency of dynamical data mixing, but also expedite the convergence of pretraining the target LLM. Numerical results demonstrate that AC-ODM-410M, which invokes the sampling strategy obtained by a proxy LLM with 410M parameters, reaching the optimal validation perplexity of ODM 71% faster, and improves performance on the zero-shot MMLU benchmark by 27.5% of accuracy, about 2.23x better on pass@1 of HumanEval benchmark. 

---
# A comparative analysis of a neural network with calculated weights and a neural network with random generation of weights based on the training dataset size 

**Authors**: Polad Geidarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.23876)  

**Abstract**: The paper discusses the capabilities of multilayer perceptron neural networks implementing metric recognition methods, for which the values of the weights are calculated analytically by formulas. Comparative experiments in training a neural network with pre-calculated weights and with random initialization of weights on different sizes of the MNIST training dataset are carried out. The results of the experiments show that a multilayer perceptron with pre-calculated weights can be trained much faster and is much more robust to the reduction of the training dataset. 

---
# A Benchmark Dataset for Graph Regression with Homogeneous and Multi-Relational Variants 

**Authors**: Peter Samoaa, Marcus Vukojevic, Morteza Haghir Chehreghani, Antonio Longa  

**Link**: [PDF](https://arxiv.org/pdf/2505.23875)  

**Abstract**: Graph-level regression underpins many real-world applications, yet public benchmarks remain heavily skewed toward molecular graphs and citation networks. This limited diversity hinders progress on models that must generalize across both homogeneous and heterogeneous graph structures. We introduce RelSC, a new graph-regression dataset built from program graphs that combine syntactic and semantic information extracted from source code. Each graph is labelled with the execution-time cost of the corresponding program, providing a continuous target variable that differs markedly from those found in existing benchmarks. RelSC is released in two complementary variants. RelSC-H supplies rich node features under a single (homogeneous) edge type, while RelSC-M preserves the original multi-relational structure, connecting nodes through multiple edge types that encode distinct semantic relationships. Together, these variants let researchers probe how representation choice influences model behaviour. We evaluate a diverse set of graph neural network architectures on both variants of RelSC. The results reveal consistent performance differences between the homogeneous and multi-relational settings, emphasising the importance of structural representation. These findings demonstrate RelSC's value as a challenging and versatile benchmark for advancing graph regression methods. 

---
# KGMark: A Diffusion Watermark for Knowledge Graphs 

**Authors**: Hongrui Peng, Haolang Lu, Yuanlong Yu, Weiye Fu, Kun Wang, Guoshun Nan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23873)  

**Abstract**: Knowledge graphs (KGs) are ubiquitous in numerous real-world applications, and watermarking facilitates protecting intellectual property and preventing potential harm from AI-generated content. Existing watermarking methods mainly focus on static plain text or image data, while they can hardly be applied to dynamic graphs due to spatial and temporal variations of structured data. This motivates us to propose KGMARK, the first graph watermarking framework that aims to generate robust, detectable, and transparent diffusion fingerprints for dynamic KG data. Specifically, we propose a novel clustering-based alignment method to adapt the watermark to spatial variations. Meanwhile, we present a redundant embedding strategy to harden the diffusion watermark against various attacks, facilitating the robustness of the watermark to the temporal variations. Additionally, we introduce a novel learnable mask matrix to improve the transparency of diffusion fingerprints. By doing so, our KGMARK properly tackles the variation challenges of structured data. Experiments on various public benchmarks show the effectiveness of our proposed KGMARK. 

---
# ADG: Ambient Diffusion-Guided Dataset Recovery for Corruption-Robust Offline Reinforcement Learning 

**Authors**: Zeyuan Liu, Zhihe Yang, Jiawei Xu, Rui Yang, Jiafei Lyu, Baoxiang Wang, Yunjian Xu, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23871)  

**Abstract**: Real-world datasets collected from sensors or human inputs are prone to noise and errors, posing significant challenges for applying offline reinforcement learning (RL). While existing methods have made progress in addressing corrupted actions and rewards, they remain insufficient for handling corruption in high-dimensional state spaces and for cases where multiple elements in the dataset are corrupted simultaneously. Diffusion models, known for their strong denoising capabilities, offer a promising direction for this problem-but their tendency to overfit noisy samples limits their direct applicability. To overcome this, we propose Ambient Diffusion-Guided Dataset Recovery (ADG), a novel approach that pioneers the use of diffusion models to tackle data corruption in offline RL. First, we introduce Ambient Denoising Diffusion Probabilistic Models (DDPM) from approximated distributions, which enable learning on partially corrupted datasets with theoretical guarantees. Second, we use the noise-prediction property of Ambient DDPM to distinguish between clean and corrupted data, and then use the clean subset to train a standard DDPM. Third, we employ the trained standard DDPM to refine the previously identified corrupted data, enhancing data quality for subsequent offline RL training. A notable strength of ADG is its versatility-it can be seamlessly integrated with any offline RL algorithm. Experiments on a range of benchmarks, including MuJoCo, Kitchen, and Adroit, demonstrate that ADG effectively mitigates the impact of corrupted data and improves the robustness of offline RL under various noise settings, achieving state-of-the-art results. 

---
# MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection 

**Authors**: Yixian Shen, Qi Bi, Jia-Hong Huang, Hongyi Zhu, Andy D. Pimentel, Anuj Pathania  

**Link**: [PDF](https://arxiv.org/pdf/2505.23870)  

**Abstract**: We present a new adaptation method MaCP, Minimal yet Mighty adaptive Cosine Projection, that achieves exceptional performance while requiring minimal parameters and memory for fine-tuning large foundation models. Its general idea is to exploit the superior energy compaction and decorrelation properties of cosine projection to improve both model efficiency and accuracy. Specifically, it projects the weight change from the low-rank adaptation into the discrete cosine space. Then, the weight change is partitioned over different levels of the discrete cosine spectrum, and each partition's most critical frequency components are selected. Extensive experiments demonstrate the effectiveness of MaCP across a wide range of single-modality tasks, including natural language understanding, natural language generation, text summarization, as well as multi-modality tasks such as image classification and video understanding. MaCP consistently delivers superior accuracy, significantly reduced computational complexity, and lower memory requirements compared to existing alternatives. 

---
# Noise-Robustness Through Noise: Asymmetric LoRA Adaption with Poisoning Expert 

**Authors**: Zhaokun Wang, Jinyu Guo, Jingwen Pu, Lingfeng Chen, Hongli Pu, Jie Ou.Libo Qin, Wenhong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.23868)  

**Abstract**: Current parameter-efficient fine-tuning methods for adapting pre-trained language models to downstream tasks are susceptible to interference from noisy data. Conventional noise-handling approaches either rely on laborious data pre-processing or employ model architecture modifications prone to error accumulation. In contrast to existing noise-process paradigms, we propose a noise-robust adaptation method via asymmetric LoRA poisoning experts (LoPE), a novel framework that enhances model robustness to noise only with generated noisy data. Drawing inspiration from the mixture-of-experts architecture, LoPE strategically integrates a dedicated poisoning expert in an asymmetric LoRA configuration. Through a two-stage paradigm, LoPE performs noise injection on the poisoning expert during fine-tuning to enhance its noise discrimination and processing ability. During inference, we selectively mask the dedicated poisoning expert to leverage purified knowledge acquired by normal experts for noise-robust output. Extensive experiments demonstrate that LoPE achieves strong performance and robustness purely through the low-cost noise injection, which completely eliminates the requirement of data cleaning. 

---
# Infi-Med: Low-Resource Medical MLLMs with Robust Reasoning Evaluation 

**Authors**: Zeyu Liu, Zhitian Hou, Yining Di, Kejing Yang, Zhijie Sang, Congkai Xie, Jingwen Yang, Siyuan Liu, Jialu Wang, Chunming Li, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23867)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated promising prospects in healthcare, particularly for addressing complex medical tasks, supporting multidisciplinary treatment (MDT), and enabling personalized precision medicine. However, their practical deployment faces critical challenges in resource efficiency, diagnostic accuracy, clinical considerations, and ethical privacy. To address these limitations, we propose Infi-Med, a comprehensive framework for medical MLLMs that introduces three key innovations: (1) a resource-efficient approach through curating and constructing high-quality supervised fine-tuning (SFT) datasets with minimal sample requirements, with a forward-looking design that extends to both pretraining and posttraining phases; (2) enhanced multimodal reasoning capabilities for cross-modal integration and clinical task understanding; and (3) a systematic evaluation system that assesses model performance across medical modalities and task types. Our experiments demonstrate that Infi-Med achieves state-of-the-art (SOTA) performance in general medical reasoning while maintaining rapid adaptability to clinical scenarios. The framework establishes a solid foundation for deploying MLLMs in real-world healthcare settings by balancing model effectiveness with operational constraints. 

---
# Towards Understanding The Calibration Benefits of Sharpness-Aware Minimization 

**Authors**: Chengli Tan, Yubo Zhou, Haishan Ye, Guang Dai, Junmin Liu, Zengjie Song, Jiangshe Zhang, Zixiang Zhao, Yunda Hao, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23866)  

**Abstract**: Deep neural networks have been increasingly used in safety-critical applications such as medical diagnosis and autonomous driving. However, many studies suggest that they are prone to being poorly calibrated and have a propensity for overconfidence, which may have disastrous consequences. In this paper, unlike standard training such as stochastic gradient descent, we show that the recently proposed sharpness-aware minimization (SAM) counteracts this tendency towards overconfidence. The theoretical analysis suggests that SAM allows us to learn models that are already well-calibrated by implicitly maximizing the entropy of the predictive distribution. Inspired by this finding, we further propose a variant of SAM, coined as CSAM, to ameliorate model calibration. Extensive experiments on various datasets, including ImageNet-1K, demonstrate the benefits of SAM in reducing calibration error. Meanwhile, CSAM performs even better than SAM and consistently achieves lower calibration error than other approaches 

---
# Combining Deep Architectures for Information Gain estimation and Reinforcement Learning for multiagent field exploration 

**Authors**: Emanuele Masiero, Vito Trianni, Giuseppe Vizzari, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2505.23865)  

**Abstract**: Precision agriculture requires efficient autonomous systems for crop monitoring, where agents must explore large-scale environments while minimizing resource consumption. This work addresses the problem as an active exploration task in a grid environment representing an agricultural field. Each cell may contain targets (e.g., damaged crops) observable from nine predefined points of view (POVs). Agents must infer the number of targets per cell using partial, sequential observations.
We propose a two-stage deep learning framework. A pre-trained LSTM serves as a belief model, updating a probabilistic map of the environment and its associated entropy, which defines the expected information gain (IG). This allows agents to prioritize informative regions. A key contribution is the inclusion of a POV visibility mask in the input, preserving the Markov property under partial observability and avoiding revisits to already explored views.
Three agent architectures were compared: an untrained IG-based agent selecting actions to maximize entropy reduction; a DQN agent using CNNs over local 3x3 inputs with belief, entropy, and POV mask; and a Double-CNN DQN agent with wider spatial context. Simulations on 20x20 maps showed that the untrained agent performs well despite its simplicity. The DQN agent matches this performance when the POV mask is included, while the Double-CNN agent consistently achieves superior exploration efficiency, especially in larger environments.
Results show that uncertainty-aware policies leveraging entropy, belief states, and visibility tracking lead to robust and scalable exploration. Future work includes curriculum learning, multi-agent cooperation with shared rewards, transformer-based models, and intrinsic motivation mechanisms to further enhance learning efficiency and policy generalization. 

---
# Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections 

**Authors**: Wei Zhuo, Zhaohuan Zhan, Ziduo Yang, Han Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23864)  

**Abstract**: Federated learning (FL) on graph-structured data typically faces non-IID challenges, particularly in scenarios where each client holds a distinct subgraph sampled from a global graph. In this paper, we introduce Federated learning with Auxiliary projections (FedAux), a personalized subgraph FL framework that learns to align, compare, and aggregate heterogeneously distributed local models without sharing raw data or node embeddings. In FedAux, each client jointly trains (i) a local GNN and (ii) a learnable auxiliary projection vector (APV) that differentiably projects node embeddings onto a 1D space. A soft-sorting operation followed by a lightweight 1D convolution refines these embeddings in the ordered space, enabling the APV to effectively capture client-specific information. After local training, these APVs serve as compact signatures that the server uses to compute inter-client similarities and perform similarity-weighted parameter mixing, yielding personalized models while preserving cross-client knowledge transfer. Moreover, we provide rigorous theoretical analysis to establish the convergence and rationality of our design. Empirical evaluations across diverse graph benchmarks demonstrate that FedAux substantially outperforms existing baselines in both accuracy and personalization performance. 

---
# Mamba Integrated with Physics Principles Masters Long-term Chaotic System Forecasting 

**Authors**: Chang Liu, Bohao Zhao, Jingtao Ding, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23863)  

**Abstract**: Long-term forecasting of chaotic systems from short-term observations remains a fundamental and underexplored challenge due to the intrinsic sensitivity to initial conditions and the complex geometry of strange attractors. Existing approaches often rely on long-term training data or focus on short-term sequence correlations, struggling to maintain predictive stability and dynamical coherence over extended horizons. We propose PhyxMamba, a novel framework that integrates a Mamba-based state-space model with physics-informed principles to capture the underlying dynamics of chaotic systems. By reconstructing the attractor manifold from brief observations using time-delay embeddings, PhyxMamba extracts global dynamical features essential for accurate forecasting. Our generative training scheme enables Mamba to replicate the physical process, augmented by multi-token prediction and attractor geometry regularization for physical constraints, enhancing prediction accuracy and preserving key statistical invariants. Extensive evaluations on diverse simulated and real-world chaotic systems demonstrate that PhyxMamba delivers superior long-term forecasting and faithfully captures essential dynamical invariants from short-term data. This framework opens new avenues for reliably predicting chaotic systems under observation-scarce conditions, with broad implications across climate science, neuroscience, epidemiology, and beyond. Our code is open-source at this https URL. 

---
# A New Deep-learning-Based Approach For mRNA Optimization: High Fidelity, Computation Efficiency, and Multiple Optimization Factors 

**Authors**: Zheng Gong, Ziyi Jiang, Weihao Gao, Deng Zhuo, Lan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.23862)  

**Abstract**: The mRNA optimization is critical for therapeutic and biotechnological applications, since sequence features directly govern protein expression levels and efficacy. However, current methods face significant challenges in simultaneously achieving three key objectives: (1) fidelity (preventing unintended amino acid changes), (2) computational efficiency (speed and scalability), and (3) the scope of optimization variables considered (multi-objective capability). Furthermore, existing methods often fall short of comprehensively incorporating the factors related to the mRNA lifecycle and translation process, including intrinsic mRNA sequence properties, secondary structure, translation elongation kinetics, and tRNA availability. To address these limitations, we introduce \textbf{RNop}, a novel deep learning-based method for mRNA optimization. We collect a large-scale dataset containing over 3 million sequences and design four specialized loss functions, the GPLoss, CAILoss, tAILoss, and MFELoss, which simultaneously enable explicit control over sequence fidelity while optimizing species-specific codon adaptation, tRNA availability, and desirable mRNA secondary structure features. Then, we demonstrate RNop's effectiveness through extensive in silico and in vivo experiments. RNop ensures high sequence fidelity, achieves significant computational throughput up to 47.32 sequences/s, and yields optimized mRNA sequences resulting in a significant increase in protein expression for functional proteins compared to controls. RNop surpasses current methodologies in both quantitative metrics and experimental validation, enlightening a new dawn for efficient and effective mRNA design. Code and models will be available at this https URL. 

---
# BiBLDR: Bidirectional Behavior Learning for Drug Repositioning 

**Authors**: Renye Zhang, Mengyun Yang, Qichang Zhao, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23861)  

**Abstract**: Drug repositioning aims to identify potential new indications for existing drugs to reduce the time and financial costs associated with developing new drugs. Most existing deep learning-based drug repositioning methods predominantly utilize graph-based representations. However, graph-based drug repositioning methods struggle to perform effective inference in cold-start scenarios involving novel drugs because of the lack of association information with the diseases. Unlike traditional graph-based approaches, we propose a bidirectional behavior learning strategy for drug repositioning, known as BiBLDR. This innovative framework redefines drug repositioning as a behavior sequential learning task to capture drug-disease interaction patterns. First, we construct bidirectional behavioral sequences based on drug and disease sides. The consideration of bidirectional information ensures a more meticulous and rigorous characterization of the behavioral sequences. Subsequently, we propose a two-stage strategy for drug repositioning. In the first stage, we construct prototype spaces to characterize the representational attributes of drugs and diseases. In the second stage, these refined prototypes and bidirectional behavior sequence data are leveraged to predict potential drug-disease associations. Based on this learning approach, the model can more robustly and precisely capture the interactive relationships between drug and disease features from bidirectional behavioral sequences. Extensive experiments demonstrate that our method achieves state-of-the-art performance on benchmark datasets. Meanwhile, BiBLDR demonstrates significantly superior performance compared to previous methods in cold-start scenarios. Our code is published in this https URL. 

---
# Quantum computing and artificial intelligence: status and perspectives 

**Authors**: Giovanni Acampora, Andris Ambainis, Natalia Ares, Leonardo Banchi, Pallavi Bhardwaj, Daniele Binosi, G. Andrew D. Briggs, Tommaso Calarco, Vedran Dunjko, Jens Eisert, Olivier Ezratty, Paul Erker, Federico Fedele, Elies Gil-Fuster, Martin Gärttner, Mats Granath, Markus Heyl, Iordanis Kerenidis, Matthias Klusch, Anton Frisk Kockum, Richard Kueng, Mario Krenn, Jörg Lässig, Antonio Macaluso, Sabrina Maniscalco, Florian Marquardt, Kristel Michielsen, Gorka Muñoz-Gil, Daniel Müssig, Hendrik Poulsen Nautrup, Evert van Nieuwenburg, Roman Orus, Jörg Schmiedmayer, Markus Schmitt, Philipp Slusallek, Filippo Vicentini, Christof Weitenberg, Frank K. Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2505.23860)  

**Abstract**: This white paper discusses and explores the various points of intersection between quantum computing and artificial intelligence (AI). It describes how quantum computing could support the development of innovative AI solutions. It also examines use cases of classical AI that can empower research and development in quantum technologies, with a focus on quantum computing and quantum sensing. The purpose of this white paper is to provide a long-term research agenda aimed at addressing foundational questions about how AI and quantum computing interact and benefit one another. It concludes with a set of recommendations and challenges, including how to orchestrate the proposed theoretical work, align quantum AI developments with quantum hardware roadmaps, estimate both classical and quantum resources - especially with the goal of mitigating and optimizing energy consumption - advance this emerging hybrid software engineering discipline, and enhance European industrial competitiveness while considering societal implications. 

---
# Towards Minimizing Feature Drift in Model Merging: Layer-wise Task Vector Fusion for Adaptive Knowledge Integration 

**Authors**: Wenju Sun, Qingyong Li, Wen Wang, Yang Liu, Yangli-ao Geng, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23859)  

**Abstract**: Multi-task model merging aims to consolidate knowledge from multiple fine-tuned task-specific experts into a unified model while minimizing performance degradation. Existing methods primarily approach this by minimizing differences between task-specific experts and the unified model, either from a parameter-level or a task-loss perspective. However, parameter-level methods exhibit a significant performance gap compared to the upper bound, while task-loss approaches entail costly secondary training procedures. In contrast, we observe that performance degradation closely correlates with feature drift, i.e., differences in feature representations of the same sample caused by model merging. Motivated by this observation, we propose Layer-wise Optimal Task Vector Merging (LOT Merging), a technique that explicitly minimizes feature drift between task-specific experts and the unified model in a layer-by-layer manner. LOT Merging can be formulated as a convex quadratic optimization problem, enabling us to analytically derive closed-form solutions for the parameters of linear and normalization layers. Consequently, LOT Merging achieves efficient model consolidation through basic matrix operations. Extensive experiments across vision and vision-language benchmarks demonstrate that LOT Merging significantly outperforms baseline methods, achieving improvements of up to 4.4% (ViT-B/32) over state-of-the-art approaches. 

---
# DATD3: Depthwise Attention Twin Delayed Deep Deterministic Policy Gradient For Model Free Reinforcement Learning Under Output Feedback Control 

**Authors**: Wuhao Wang, Zhiyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23857)  

**Abstract**: Reinforcement learning in real-world applications often involves output-feedback settings, where the agent receives only partial state information. To address this challenge, we propose the Output-Feedback Markov Decision Process (OPMDP), which extends the standard MDP formulation to accommodate decision-making based on observation histories. Building on this framework, we introduce Depthwise Attention Twin Delayed Deep Deterministic Policy Gradient (DATD3), a novel actor-critic algorithm that employs depthwise separable convolution and multi-head attention to encode historical observations. DATD3 maintains policy expressiveness while avoiding the instability of recurrent models. Extensive experiments on continuous control tasks demonstrate that DATD3 outperforms existing memory-based and recurrent baselines under both partial and full observability. 

---
# OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Modalities 

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.23856)  

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose OMNIGUARD, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. OMNIGUARD improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, OMNIGUARD is also very efficient ($\approx 120 \times$ faster than the next fastest baseline). Code and data are available at: this https URL. 

---
# Revisiting Uncertainty Estimation and Calibration of Large Language Models 

**Authors**: Linwei Tao, Yi-Fan Yeh, Minjing Dong, Tao Huang, Philip Torr, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23854)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes applications, robust uncertainty estimation is essential for ensuring the safe and trustworthy deployment of LLMs. We present the most comprehensive study to date of uncertainty estimation in LLMs, evaluating 80 models spanning open- and closed-source families, dense and Mixture-of-Experts (MoE) architectures, reasoning and non-reasoning modes, quantization variants and parameter scales from 0.6B to 671B. Focusing on three representative black-box single-pass methods, including token probability-based uncertainty (TPU), numerical verbal uncertainty (NVU), and linguistic verbal uncertainty (LVU), we systematically evaluate uncertainty calibration and selective classification using the challenging MMLU-Pro benchmark, which covers both reasoning-intensive and knowledge-based tasks. Our results show that LVU consistently outperforms TPU and NVU, offering stronger calibration and discrimination while being more interpretable. We also find that high accuracy does not imply reliable uncertainty, and that model scale, post-training, reasoning ability and quantization all influence estimation performance. Notably, LLMs exhibit better uncertainty estimates on reasoning tasks than on knowledge-heavy ones, and good calibration does not necessarily translate to effective error ranking. These findings highlight the need for multi-perspective evaluation and position LVU as a practical tool for improving the reliability of LLMs in real-world settings. 

---
# Large Language Model-Based Agents for Automated Research Reproducibility: An Exploratory Study in Alzheimer's Disease 

**Authors**: Nic Dobbins, Christelle Xiong, Kristine Lan, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23852)  

**Abstract**: Objective: To demonstrate the capabilities of Large Language Models (LLMs) as autonomous agents to reproduce findings of published research studies using the same or similar dataset.
Materials and Methods: We used the "Quick Access" dataset of the National Alzheimer's Coordinating Center (NACC). We identified highly cited published research manuscripts using NACC data and selected five studies that appeared reproducible using this dataset alone. Using GPT-4o, we created a simulated research team of LLM-based autonomous agents tasked with writing and executing code to dynamically reproduce the findings of each study, given only study Abstracts, Methods sections, and data dictionary descriptions of the dataset.
Results: We extracted 35 key findings described in the Abstracts across 5 Alzheimer's studies. On average, LLM agents approximately reproduced 53.2% of findings per study. Numeric values and range-based findings often differed between studies and agents. The agents also applied statistical methods or parameters that varied from the originals, though overall trends and significance were sometimes similar.
Discussion: In some cases, LLM-based agents replicated research techniques and findings. In others, they failed due to implementation flaws or missing methodological detail. These discrepancies show the current limits of LLMs in fully automating reproducibility assessments. Still, this early investigation highlights the potential of structured agent-based systems to provide scalable evaluation of scientific rigor.
Conclusion: This exploratory work illustrates both the promise and limitations of LLMs as autonomous agents for automating reproducibility in biomedical research. 

---
# ASyMOB: Algebraic Symbolic Mathematical Operations Benchmark 

**Authors**: Michael Shalyt, Rotem Elimelech, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2505.23851)  

**Abstract**: Large language models (LLMs) are rapidly approaching the level of proficiency in university-level symbolic mathematics required for applications in advanced science and technology. However, existing benchmarks fall short in assessing the core skills of LLMs in symbolic mathematics-such as integration, differential equations, and algebraic simplification. To address this gap, we introduce ASyMOB, a novel assessment framework focused exclusively on symbolic manipulation, featuring 17,092 unique math challenges, organized by similarity and complexity. ASyMOB enables analysis of LLM generalization capabilities by comparing performance in problems that differ by simple numerical or symbolic `perturbations'. Evaluated LLMs exhibit substantial degradation in performance for all perturbation types (up to -70.3%), suggesting reliance on memorized patterns rather than deeper understanding of symbolic math, even among models achieving high baseline accuracy. Comparing LLM performance to computer algebra systems, we identify examples where they fail while LLMs succeed, as well as problems solved only by combining both approaches. Models capable of integrated code execution yielded higher accuracy compared to their performance without code, particularly stabilizing weaker models (up to +33.1% for certain perturbation types). Notably, the most advanced models (o4-mini, Gemini 2.5 Flash) demonstrate not only high symbolic math proficiency (scoring 96.8% and 97.6% on the unperturbed set), but also remarkable robustness against perturbations, (-21.7% and -21.2% vs. average -50.4% for the other models). This may indicate a recent "phase transition" in the generalization capabilities of frontier LLMs. It remains to be seen whether the path forward lies in deeper integration with sophisticated external tools, or in developing models so capable that symbolic math systems like CAS become unnecessary. 

---
# CADRE: Customizable Assurance of Data Readiness in Privacy-Preserving Federated Learning 

**Authors**: Kaveen Hiniduma, Zilinghan Li, Aditya Sinha, Ravi Madduri, Suren Byna  

**Link**: [PDF](https://arxiv.org/pdf/2505.23849)  

**Abstract**: Privacy-Preserving Federated Learning (PPFL) is a decentralized machine learning approach where multiple clients train a model collaboratively. PPFL preserves privacy and security of the client's data by not exchanging it. However, ensuring that data at each client is of high quality and ready for federated learning (FL) is a challenge due to restricted data access. In this paper, we introduce CADRE (Customizable Assurance of Data REadiness) for FL, a novel framework that allows users to define custom data readiness (DR) standards, metrics, rules, and remedies tailored to specific FL tasks. Our framework generates comprehensive DR reports based on the user-defined metrics, rules, and remedies to ensure datasets are optimally prepared for FL while preserving privacy. We demonstrate the framework's practical application by integrating it into an existing PPFL framework. We conducted experiments across six diverse datasets, addressing seven different DR issues. The results illustrate the framework's versatility and effectiveness in ensuring DR across various dimensions, including data quality, privacy, and fairness. This approach enhances the performance and reliability of FL models as well as utilizes valuable resources by identifying and addressing data-related issues before the training phase. 

---
# Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems 

**Authors**: Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Taewan Kim, Makoto Onizuka, Wonyong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23847)  

**Abstract**: Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines. 

---
# Large Language Models Often Know When They Are Being Evaluated 

**Authors**: Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.23836)  

**Abstract**: If AI models can detect when they are being evaluated, the effectiveness of evaluations might be compromised. For example, models could have systematically different behavior during evaluations, leading to less reliable benchmarks for deployment and governance decisions. We investigate whether frontier language models can accurately classify transcripts based on whether they originate from evaluations or real-world deployment, a capability we call evaluation awareness. To achieve this, we construct a diverse benchmark of 1,000 prompts and transcripts from 61 distinct datasets. These span public benchmarks (e.g., MMLU, SWEBench), real-world deployment interactions, and agent trajectories from scaffolding frameworks (e.g., web-browsing agents). Frontier models clearly demonstrate above-random evaluation awareness (Gemini-2.5-Pro reaches an AUC of $0.83$), but do not yet surpass our simple human baseline (AUC of $0.92$). Furthermore, both AI models and humans are better at identifying evaluations in agentic settings compared to chat settings. Additionally, we test whether models can identify the purpose of the evaluation. Under multiple-choice and open-ended questioning, AI models far outperform random chance in identifying what an evaluation is testing for. Our results indicate that frontier models already exhibit a substantial, though not yet superhuman, level of evaluation-awareness. We recommend tracking this capability in future models. 

---
# DP-RTFL: Differentially Private Resilient Temporal Federated Learning for Trustworthy AI in Regulated Industries 

**Authors**: Abhijit Talluri  

**Link**: [PDF](https://arxiv.org/pdf/2505.23813)  

**Abstract**: Federated Learning (FL) has emerged as a critical paradigm for enabling privacy-preserving machine learning, particularly in regulated sectors such as finance and healthcare. However, standard FL strategies often encounter significant operational challenges related to fault tolerance, system resilience against concurrent client and server failures, and the provision of robust, verifiable privacy guarantees essential for handling sensitive data. These deficiencies can lead to training disruptions, data loss, compromised model integrity, and non-compliance with data protection regulations (e.g., GDPR, CCPA). This paper introduces Differentially Private Resilient Temporal Federated Learning (DP-RTFL), an advanced FL framework designed to ensure training continuity, precise state recovery, and strong data privacy. DP-RTFL integrates local Differential Privacy (LDP) at the client level with resilient temporal state management and integrity verification mechanisms, such as hash-based commitments (referred to as Zero-Knowledge Integrity Proofs or ZKIPs in this context). The framework is particularly suited for critical applications like credit risk assessment using sensitive financial data, aiming to be operationally robust, auditable, and scalable for enterprise AI deployments. The implementation of the DP-RTFL framework is available as open-source. 

---
# Emotion-aware Dual Cross-Attentive Neural Network with Label Fusion for Stance Detection in Misinformative Social Media Content 

**Authors**: Lata Pangtey, Mohammad Zia Ur Rehman, Prasad Chaudhari, Shubhi Bansal, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.23812)  

**Abstract**: The rapid evolution of social media has generated an overwhelming volume of user-generated content, conveying implicit opinions and contributing to the spread of misinformation. The method aims to enhance the detection of stance where misinformation can polarize user opinions. Stance detection has emerged as a crucial approach to effectively analyze underlying biases in shared information and combating misinformation. This paper proposes a novel method for \textbf{S}tance \textbf{P}rediction through a \textbf{L}abel-fused dual cross-\textbf{A}ttentive \textbf{E}motion-aware neural \textbf{Net}work (SPLAENet) in misinformative social media user-generated content. The proposed method employs a dual cross-attention mechanism and a hierarchical attention network to capture inter and intra-relationships by focusing on the relevant parts of source text in the context of reply text and vice versa. We incorporate emotions to effectively distinguish between different stance categories by leveraging the emotional alignment or divergence between the texts. We also employ label fusion that uses distance-metric learning to align extracted features with stance labels, improving the method's ability to accurately distinguish between stances. Extensive experiments demonstrate the significant improvements achieved by SPLAENet over existing state-of-the-art methods. SPLAENet demonstrates an average gain of 8.92\% in accuracy and 17.36\% in F1-score on the RumourEval dataset. On the SemEval dataset, it achieves average gains of 7.02\% in accuracy and 10.92\% in F1-score. On the P-stance dataset, it demonstrates average gains of 10.03\% in accuracy and 11.18\% in F1-score. These results validate the effectiveness of the proposed method for stance detection in the context of misinformative social media content. 

---
# LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions 

**Authors**: Hadi Askari, Shivanshu Gupta, Fei Wang, Anshuman Chhabra, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23811)  

**Abstract**: Pretrained Large Language Models (LLMs) achieve strong performance across a wide range of tasks, yet exhibit substantial variability in the various layers' training quality with respect to specific downstream applications, limiting their downstream this http URL is therefore critical to estimate layer-wise training quality in a manner that accounts for both model architecture and training data. However, existing approaches predominantly rely on model-centric heuristics (such as spectral statistics, outlier detection, or uniform allocation) while overlooking the influence of data. To address these limitations, we propose LayerIF, a data-driven framework that leverages Influence Functions to quantify the training quality of individual layers in a principled and task-sensitive manner. By isolating each layer's gradients and measuring the sensitivity of the validation loss to training examples by computing layer-wise influences, we derive data-driven estimates of layer importance. Notably, our method produces task-specific layer importance estimates for the same LLM, revealing how layers specialize for different test-time evaluation tasks. We demonstrate the utility of our scores by leveraging them for two downstream applications: (a) expert allocation in LoRA-MoE architectures and (b) layer-wise sparsity distribution for LLM pruning. Experiments across multiple LLM architectures demonstrate that our model-agnostic, influence-guided allocation leads to consistent gains in task performance. 

---
# MARS-Bench: A Multi-turn Athletic Real-world Scenario Benchmark for Dialogue Evaluation 

**Authors**: Chenghao Yang, Yinbo Luo, Zhoufutu Wen, Qi Chu, Tao Gong, Longxiang Liu, Kaiyuan Zhang, Jianpeng Jiao, Ge Zhang, Wenhao Huang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23810)  

**Abstract**: Large Language Models (\textbf{LLMs}), e.g. ChatGPT, have been widely adopted in real-world dialogue applications. However, LLMs' robustness, especially in handling long complex dialogue sessions, including frequent motivation transfer, sophisticated cross-turn dependency, is criticized all along. Nevertheless, no existing benchmarks can fully reflect these weaknesses. We present \textbf{MARS-Bench}, a \textbf{M}ulti-turn \textbf{A}thletic \textbf{R}eal-world \textbf{S}cenario Dialogue \textbf{Bench}mark, designed to remedy the gap. MARS-Bench is constructed from play-by-play text commentary so to feature realistic dialogues specifically designed to evaluate three critical aspects of multi-turn conversations: Ultra Multi-turn, Interactive Multi-turn, and Cross-turn Tasks. Extensive experiments on MARS-Bench also reveal that closed-source LLMs significantly outperform open-source alternatives, explicit reasoning significantly boosts LLMs' robustness on handling long complex dialogue sessions, and LLMs indeed face significant challenges when handling motivation transfer and sophisticated cross-turn dependency. Moreover, we provide mechanistic interpretability on how attention sinks due to special tokens lead to LLMs' performance degradation when handling long complex dialogue sessions based on attention visualization experiment in Qwen2.5-7B-Instruction. 

---
# LLM-Driven E-Commerce Marketing Content Optimization: Balancing Creativity and Conversion 

**Authors**: Haowei Yang, Haotian Lyu, Tianle Zhang, Dingzhou Wang, Yushang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23809)  

**Abstract**: As e-commerce competition intensifies, balancing creative content with conversion effectiveness becomes critical. Leveraging LLMs' language generation capabilities, we propose a framework that integrates prompt engineering, multi-objective fine-tuning, and post-processing to generate marketing copy that is both engaging and conversion-driven. Our fine-tuning method combines sentiment adjustment, diversity enhancement, and CTA embedding. Through offline evaluations and online A/B tests across categories, our approach achieves a 12.5 % increase in CTR and an 8.3 % increase in CVR while maintaining content novelty. This provides a practical solution for automated copy generation and suggests paths for future multimodal, real-time personalization. 

---
# DenseLoRA: Dense Low-Rank Adaptation of Large Language Models 

**Authors**: Lin Mu, Xiaoyu Wang, Li Ni, Yang Li, Zhize Wu, Peiquan Jin, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23808)  

**Abstract**: Low-rank adaptation (LoRA) has been developed as an efficient approach for adapting large language models (LLMs) by fine-tuning two low-rank matrices, thereby reducing the number of trainable parameters. However, prior research indicates that many of the weights in these matrices are redundant, leading to inefficiencies in parameter utilization. To address this limitation, we introduce Dense Low-Rank Adaptation (DenseLoRA), a novel approach that enhances parameter efficiency while achieving superior performance compared to LoRA. DenseLoRA builds upon the concept of representation fine-tuning, incorporating a single Encoder-Decoder to refine and compress hidden representations across all adaptation layers before applying adaptation. Instead of relying on two redundant low-rank matrices as in LoRA, DenseLoRA adapts LLMs through a dense low-rank matrix, improving parameter utilization and adaptation efficiency. We evaluate DenseLoRA on various benchmarks, showing that it achieves 83.8% accuracy with only 0.01% of trainable parameters, compared to LoRA's 80.8% accuracy with 0.70% of trainable parameters on LLaMA3-8B. Additionally, we conduct extensive experiments to systematically assess the impact of DenseLoRA's components on overall model performance. Code is available at this https URL. 

---
# DLP: Dynamic Layerwise Pruning in Large Language Models 

**Authors**: Yuli Chen, Bo Cheng, Jiale Han, Yingying Zhang, Yingting Li, Shuhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23807)  

**Abstract**: Pruning has recently been widely adopted to reduce the parameter scale and improve the inference efficiency of Large Language Models (LLMs). Mainstream pruning techniques often rely on uniform layerwise pruning strategies, which can lead to severe performance degradation at high sparsity levels. Recognizing the varying contributions of different layers in LLMs, recent studies have shifted their focus toward non-uniform layerwise pruning. However, these approaches often rely on pre-defined values, which can result in suboptimal performance. To overcome these limitations, we propose a novel method called Dynamic Layerwise Pruning (DLP). This approach adaptively determines the relative importance of each layer by integrating model weights with input activation information, assigning pruning rates accordingly. Experimental results show that DLP effectively preserves model performance at high sparsity levels across multiple LLMs. Specifically, at 70% sparsity, DLP reduces the perplexity of LLaMA2-7B by 7.79 and improves the average accuracy by 2.7% compared to state-of-the-art methods. Moreover, DLP is compatible with various existing LLM compression techniques and can be seamlessly integrated into Parameter-Efficient Fine-Tuning (PEFT). We release the code at this https URL to facilitate future research. 

---
# MedOrchestra: A Hybrid Cloud-Local LLM Approach for Clinical Data Interpretation 

**Authors**: Sihyeon Lee, Hyunjoo Song, Jong-chan Lee, Yoon Jin Lee, Boram Lee, Hee-Eon Lim, Dongyeong Kim, Jinwook Seo, Bohyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.23806)  

**Abstract**: Deploying large language models (LLMs) in clinical settings faces critical trade-offs: cloud LLMs, with their extensive parameters and superior performance, pose risks to sensitive clinical data privacy, while local LLMs preserve privacy but often fail at complex clinical interpretation tasks. We propose MedOrchestra, a hybrid framework where a cloud LLM decomposes complex clinical tasks into manageable subtasks and prompt generation, while a local LLM executes these subtasks in a privacy-preserving manner. Without accessing clinical data, the cloud LLM generates and validates subtask prompts using clinical guidelines and synthetic test cases. The local LLM executes subtasks locally and synthesizes outputs generated by the cloud LLM. We evaluate MedOrchestra on pancreatic cancer staging using 100 radiology reports under NCCN guidelines. On free-text reports, MedOrchestra achieves 70.21% accuracy, outperforming local model baselines (without guideline: 48.94%, with guideline: 56.59%) and board-certified clinicians (gastroenterologists: 59.57%, surgeons: 65.96%, radiologists: 55.32%). On structured reports, MedOrchestra reaches 85.42% accuracy, showing clear superiority across all settings. 

---
# ADA: Automated Moving Target Defense for AI Workloads via Ephemeral Infrastructure-Native Rotation in Kubernetes 

**Authors**: Akram Sheriff, Ken Huang, Zsolt Nemeth, Madjid Nakhjiri  

**Link**: [PDF](https://arxiv.org/pdf/2505.23805)  

**Abstract**: This paper introduces the Adaptive Defense Agent (ADA), an innovative Automated Moving Target Defense (AMTD) system designed to fundamentally enhance the security posture of AI workloads. ADA operates by continuously and automatically rotating these workloads at the infrastructure level, leveraging the inherent ephemerality of Kubernetes pods. This constant managed churn systematically invalidates attacker assumptions and disrupts potential kill chains by regularly destroying and respawning AI service instances. This methodology, applying principles of chaos engineering as a continuous, proactive defense, offers a paradigm shift from traditional static defenses that rely on complex and expensive confidential or trusted computing solutions to secure the underlying compute platforms, while at the same time agnostically supporting the latest advancements in agentic and nonagentic AI ecosystems and solutions such as agent-to-agent (A2A) communication frameworks or model context protocols (MCP). This AI-native infrastructure design, relying on the widely proliferated cloud-native Kubernetes technologies, facilitates easier deployment, simplifies maintenance through an inherent zero trust posture achieved by rotation, and promotes faster adoption. We posit that ADA's novel approach to AMTD provides a more robust, agile, and operationally efficient zero-trust model for AI services, achieving security through proactive environmental manipulation rather than reactive patching. 

---
# Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause Frequencies 

**Authors**: Terrance Liu, Shuyi Wang, Daniel Preotiuc-Pietro, Yash Chandarana, Chirag Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.23804)  

**Abstract**: While large language models (LLMs) achieve strong performance on text-to-SQL parsing, they sometimes exhibit unexpected failures in which they are confidently incorrect. Building trustworthy text-to-SQL systems thus requires eliciting reliable uncertainty measures from the LLM. In this paper, we study the problem of providing a calibrated confidence score that conveys the likelihood of an output query being correct. Our work is the first to establish a benchmark for post-hoc calibration of LLM-based text-to-SQL parsing. In particular, we show that Platt scaling, a canonical method for calibration, provides substantial improvements over directly using raw model output probabilities as confidence scores. Furthermore, we propose a method for text-to-SQL calibration that leverages the structured nature of SQL queries to provide more granular signals of correctness, named "sub-clause frequency" (SCF) scores. Using multivariate Platt scaling (MPS), our extension of the canonical Platt scaling technique, we combine individual SCF scores into an overall accurate and calibrated score. Empirical evaluation on two popular text-to-SQL datasets shows that our approach of combining MPS and SCF yields further improvements in calibration and the related task of error detection over traditional Platt scaling. 

---
# MultiPhishGuard: An LLM-based Multi-Agent System for Phishing Email Detection 

**Authors**: Yinuo Xue, Eric Spero, Yun Sing Koh, Giovanni Russello  

**Link**: [PDF](https://arxiv.org/pdf/2505.23803)  

**Abstract**: Phishing email detection faces critical challenges from evolving adversarial tactics and heterogeneous attack patterns. Traditional detection methods, such as rule-based filters and denylists, often struggle to keep pace with these evolving tactics, leading to false negatives and compromised security. While machine learning approaches have improved detection accuracy, they still face challenges adapting to novel phishing strategies. We present MultiPhishGuard, a dynamic LLM-based multi-agent detection system that synergizes specialized expertise with adversarial-aware reinforcement learning. Our framework employs five cooperative agents (text, URL, metadata, explanation simplifier, and adversarial agents) with automatically adjusted decision weights powered by a Proximal Policy Optimization reinforcement learning algorithm. To address emerging threats, we introduce an adversarial training loop featuring an adversarial agent that generates subtle context-aware email variants, creating a self-improving defense ecosystem and enhancing system robustness. Experimental evaluations on public datasets demonstrate that MultiPhishGuard significantly outperforms Chain-of-Thoughts, single-agent baselines and state-of-the-art detectors, as validated by ablation studies and comparative analyses. Experiments demonstrate that MultiPhishGuard achieves high accuracy (97.89\%) with low false positive (2.73\%) and false negative rates (0.20\%). Additionally, we incorporate an explanation simplifier agent, which provides users with clear and easily understandable explanations for why an email is classified as phishing or legitimate. This work advances phishing defense through dynamic multi-agent collaboration and generative adversarial resilience. 

---
# MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks 

**Authors**: Suhana Bedi, Hejie Cui, Miguel Fuentes, Alyssa Unell, Michael Wornow, Juan M. Banda, Nikesh Kotecha, Timothy Keyes, Yifan Mai, Mert Oez, Hao Qiu, Shrey Jain, Leonardo Schettini, Mehr Kashyap, Jason Alan Fries, Akshay Swaminathan, Philip Chung, Fateme Nateghi, Asad Aali, Ashwin Nayak, Shivam Vedak, Sneha S. Jain, Birju Patel, Oluseyi Fayanju, Shreya Shah, Ethan Goh, Dong-han Yao, Brian Soetikno, Eduardo Reis, Sergios Gatidis, Vasu Divi, Robson Capasso, Rachna Saralkar, Chia-Chun Chiang, Jenelle Jindal, Tho Pham, Faraz Ghoddusi, Steven Lin, Albert S. Chiou, Christy Hong, Mohana Roy, Michael F. Gensheimer, Hinesh Patel, Kevin Schulman, Dev Dash, Danton Char, Lance Downing, Francois Grolleau, Kameron Black, Bethel Mieso, Aydin Zahedivash, Wen-wai Yim, Harshita Sharma, Tony Lee, Hannah Kirsch, Jennifer Lee, Nerissa Ambers, Carlene Lugtu, Aditya Sharma, Bilal Mawji, Alex Alekseyev, Vicky Zhou, Vikas Kakkar, Jarrod Helzer, Anurang Revri, Yair Bannett, Roxana Daneshjou, Jonathan Chen, Emily Alsentzer, Keith Morse, Nirmal Ravi, Nima Aghaeepour, Vanessa Kennedy, Akshay Chaudhari, Thomas Wang, Sanmi Koyejo, Matthew P. Lungren, Eric Horvitz, Percy Liang, Mike Pfeffer, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.23802)  

**Abstract**: While large language models (LLMs) achieve near-perfect scores on medical licensing exams, these evaluations inadequately reflect the complexity and diversity of real-world clinical practice. We introduce MedHELM, an extensible evaluation framework for assessing LLM performance for medical tasks with three key contributions. First, a clinician-validated taxonomy spanning 5 categories, 22 subcategories, and 121 tasks developed with 29 clinicians. Second, a comprehensive benchmark suite comprising 35 benchmarks (17 existing, 18 newly formulated) providing complete coverage of all categories and subcategories in the taxonomy. Third, a systematic comparison of LLMs with improved evaluation methods (using an LLM-jury) and a cost-performance analysis. Evaluation of 9 frontier LLMs, using the 35 benchmarks, revealed significant performance variation. Advanced reasoning models (DeepSeek R1: 66% win-rate; o3-mini: 64% win-rate) demonstrated superior performance, though Claude 3.5 Sonnet achieved comparable results at 40% lower estimated computational cost. On a normalized accuracy scale (0-1), most models performed strongly in Clinical Note Generation (0.73-0.85) and Patient Communication & Education (0.78-0.83), moderately in Medical Research Assistance (0.65-0.75), and generally lower in Clinical Decision Support (0.56-0.72) and Administration & Workflow (0.53-0.63). Our LLM-jury evaluation method achieved good agreement with clinician ratings (ICC = 0.47), surpassing both average clinician-clinician agreement (ICC = 0.43) and automated baselines including ROUGE-L (0.36) and BERTScore-F1 (0.44). Claude 3.5 Sonnet achieved comparable performance to top models at lower estimated cost. These findings highlight the importance of real-world, task-specific evaluation for medical use of LLMs and provides an open source framework to enable this. 

---
# SEMFED: Semantic-Aware Resource-Efficient Federated Learning for Heterogeneous NLP Tasks 

**Authors**: Sajid Hussain, Muhammad Sohail, Nauman Ali Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23801)  

**Abstract**: Background: Federated Learning (FL) has emerged as a promising paradigm for training machine learning models while preserving data privacy. However, applying FL to Natural Language Processing (NLP) tasks presents unique challenges due to semantic heterogeneity across clients, vocabulary mismatches, and varying resource constraints on edge devices. Objectives: This paper introduces SEMFED, a novel semantic-aware resource-efficient federated learning framework specifically designed for heterogeneous NLP tasks. Methods: SEMFED incorporates three key innovations: (1) a semantic-aware client selection mechanism that balances semantic diversity with resource constraints, (2) adaptive NLP-specific model architectures tailored to device capabilities while preserving semantic information, and (3) a communication-efficient semantic feature compression technique that significantly reduces bandwidth requirements. Results: Experimental results on various NLP classification tasks demonstrate that SEMFED achieves an 80.5% reduction in communication costs while maintaining model accuracy above 98%, outperforming state-of-the-art FL approaches. Conclusion: SEMFED effectively manages heterogeneous client environments with varying computational resources, network reliability, and semantic data distributions, making it particularly suitable for real-world federated NLP deployments. 

---
# Estimating LLM Consistency: A User Baseline vs Surrogate Metrics 

**Authors**: Xiaoyuan Wu, Weiran Lin, Omer Akgul, Lujo Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2505.23799)  

**Abstract**: Large language models (LLMs) are prone to hallucinations and sensitive to prompt perturbations, often resulting in inconsistent or unreliable generated text. Different methods have been proposed to mitigate such hallucinations and fragility -- one of them being measuring the consistency (the model's confidence in the response, or likelihood of generating a similar response when resampled) of LLM responses. In previous work, measuring consistency often relied on the probability of a response appearing within a pool of resampled responses, or internal states or logits of responses. However, it is not yet clear how well these approaches approximate how humans perceive the consistency of LLM responses. We performed a user study (n=2,976) and found current methods typically do not approximate users' perceptions of LLM consistency very well. We propose a logit-based ensemble method for estimating LLM consistency, and we show that this method matches the performance of the best-performing existing metric in estimating human ratings of LLM consistency. Our results suggest that methods of estimating LLM consistency without human evaluation are sufficiently imperfect that we suggest evaluation with human input be more broadly used. 

---
# My Answer Is NOT 'Fair': Mitigating Social Bias in Vision-Language Models via Fair and Biased Residuals 

**Authors**: Jian Lan, Yifei Fu, Udo Schlegel, Gengyuan Zhang, Tanveer Hannan, Haokun Chen, Thomas Seidl  

**Link**: [PDF](https://arxiv.org/pdf/2505.23798)  

**Abstract**: Social bias is a critical issue in large vision-language models (VLMs), where fairness- and ethics-related problems harm certain groups of people in society. It is unknown to what extent VLMs yield social bias in generative responses. In this study, we focus on evaluating and mitigating social bias on both the model's response and probability distribution. To do so, we first evaluate four state-of-the-art VLMs on PAIRS and SocialCounterfactuals datasets with the multiple-choice selection task. Surprisingly, we find that models suffer from generating gender-biased or race-biased responses. We also observe that models are prone to stating their responses are fair, but indeed having mis-calibrated confidence levels towards particular social groups. While investigating why VLMs are unfair in this study, we observe that VLMs' hidden layers exhibit substantial fluctuations in fairness levels. Meanwhile, residuals in each layer show mixed effects on fairness, with some contributing positively while some lead to increased bias. Based on these findings, we propose a post-hoc method for the inference stage to mitigate social bias, which is training-free and model-agnostic. We achieve this by ablating bias-associated residuals while amplifying fairness-associated residuals on model hidden layers during inference. We demonstrate that our post-hoc method outperforms the competing training strategies, helping VLMs have fairer responses and more reliable confidence levels. 

---
# Detection of Suicidal Risk on Social Media: A Hybrid Model 

**Authors**: Zaihan Yang, Ryan Leonard, Hien Tran, Rory Driscoll, Chadbourne Davis  

**Link**: [PDF](https://arxiv.org/pdf/2505.23797)  

**Abstract**: Suicidal thoughts and behaviors are increasingly recognized as a critical societal concern, highlighting the urgent need for effective tools to enable early detection of suicidal risk. In this work, we develop robust machine learning models that leverage Reddit posts to automatically classify them into four distinct levels of suicide risk severity. We frame this as a multi-class classification task and propose a RoBERTa-TF-IDF-PCA Hybrid model, integrating the deep contextual embeddings from Robustly Optimized BERT Approach (RoBERTa), a state-of-the-art deep learning transformer model, with the statistical term-weighting of TF-IDF, further compressed with PCA, to boost the accuracy and reliability of suicide risk assessment. To address data imbalance and overfitting, we explore various data resampling techniques and data augmentation strategies to enhance model generalization. Additionally, we compare our model's performance against that of using RoBERTa only, the BERT model and other traditional machine learning classifiers. Experimental results demonstrate that the hybrid model can achieve improved performance, giving a best weighted $F_{1}$ score of 0.7512. 

---
# R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning 

**Authors**: Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23794)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates external knowledge with Large Language Models (LLMs) to enhance factual correctness and mitigate hallucination. However, dense retrievers often become the bottleneck of RAG systems due to their limited parameters compared to LLMs and their inability to perform step-by-step reasoning. While prompt-based iterative RAG attempts to address these limitations, it is constrained by human-designed workflows. To address these limitations, we propose $\textbf{R3-RAG}$, which uses $\textbf{R}$einforcement learning to make the LLM learn how to $\textbf{R}$eason and $\textbf{R}$etrieve step by step, thus retrieving comprehensive external knowledge and leading to correct answers. R3-RAG is divided into two stages. We first use cold start to make the model learn the manner of iteratively interleaving reasoning and retrieval. Then we use reinforcement learning to further harness its ability to better explore the external retrieval environment. Specifically, we propose two rewards for R3-RAG: 1) answer correctness for outcome reward, which judges whether the trajectory leads to a correct answer; 2) relevance-based document verification for process reward, encouraging the model to retrieve documents that are relevant to the user question, through which we can let the model learn how to iteratively reason and retrieve relevant documents to get the correct answer. Experimental results show that R3-RAG significantly outperforms baselines and can transfer well to different retrievers. We release R3-RAG at this https URL. 

---
# USB: A Comprehensive and Unified Safety Evaluation Benchmark for Multimodal Large Language Models 

**Authors**: Baolin Zheng, Guanlin Chen, Hongqiong Zhong, Qingyang Teng, Yingshui Tan, Zhendong Liu, Weixun Wang, Jiaheng Liu, Jian Yang, Huiyun Jing, Jincheng Wei, Wenbo Su, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23793)  

**Abstract**: Despite their remarkable achievements and widespread adoption, Multimodal Large Language Models (MLLMs) have revealed significant security vulnerabilities, highlighting the urgent need for robust safety evaluation benchmarks. Existing MLLM safety benchmarks, however, fall short in terms of data quality and coverge, and modal risk combinations, resulting in inflated and contradictory evaluation results, which hinders the discovery and governance of security concerns. Besides, we argue that vulnerabilities to harmful queries and oversensitivity to harmless ones should be considered simultaneously in MLLMs safety evaluation, whereas these were previously considered separately. In this paper, to address these shortcomings, we introduce Unified Safety Benchmarks (USB), which is one of the most comprehensive evaluation benchmarks in MLLM safety. Our benchmark features high-quality queries, extensive risk categories, comprehensive modal combinations, and encompasses both vulnerability and oversensitivity evaluations. From the perspective of two key dimensions: risk categories and modality combinations, we demonstrate that the available benchmarks -- even the union of the vast majority of them -- are far from being truly comprehensive. To bridge this gap, we design a sophisticated data synthesis pipeline that generates extensive, high-quality complementary data addressing previously unexplored aspects. By combining open-source datasets with our synthetic data, our benchmark provides 4 distinct modality combinations for each of the 61 risk sub-categories, covering both English and Chinese across both vulnerability and oversensitivity dimensions. 

---
# Zero-Trust Foundation Models: A New Paradigm for Secure and Collaborative Artificial Intelligence for Internet of Things 

**Authors**: Kai Li, Conggai Li, Xin Yuan, Shenghong Li, Sai Zou, Syed Sohail Ahmed, Wei Ni, Dusit Niyato, Abbas Jamalipour, Falko Dressler, Ozgur B. Akan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23792)  

**Abstract**: This paper focuses on Zero-Trust Foundation Models (ZTFMs), a novel paradigm that embeds zero-trust security principles into the lifecycle of foundation models (FMs) for Internet of Things (IoT) systems. By integrating core tenets, such as continuous verification, least privilege access (LPA), data confidentiality, and behavioral analytics into the design, training, and deployment of FMs, ZTFMs can enable secure, privacy-preserving AI across distributed, heterogeneous, and potentially adversarial IoT environments. We present the first structured synthesis of ZTFMs, identifying their potential to transform conventional trust-based IoT architectures into resilient, self-defending ecosystems. Moreover, we propose a comprehensive technical framework, incorporating federated learning (FL), blockchain-based identity management, micro-segmentation, and trusted execution environments (TEEs) to support decentralized, verifiable intelligence at the network edge. In addition, we investigate emerging security threats unique to ZTFM-enabled systems and evaluate countermeasures, such as anomaly detection, adversarial training, and secure aggregation. Through this analysis, we highlight key open research challenges in terms of scalability, secure orchestration, interpretable threat attribution, and dynamic trust calibration. This survey lays a foundational roadmap for secure, intelligent, and trustworthy IoT infrastructures powered by FMs. 

---
# Evaluating Query Efficiency and Accuracy of Transfer Learning-based Model Extraction Attack in Federated Learning 

**Authors**: Sayyed Farid Ahamed, Sandip Roy, Soumya Banerjee, Marc Vucovich, Kevin Choi, Abdul Rahman, Alison Hu, Edward Bowen, Sachin Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2505.23791)  

**Abstract**: Federated Learning (FL) is a collaborative learning framework designed to protect client data, yet it remains highly vulnerable to Intellectual Property (IP) threats. Model extraction (ME) attacks pose a significant risk to Machine Learning as a Service (MLaaS) platforms, enabling attackers to replicate confidential models by querying black-box (without internal insight) APIs. Despite FL's privacy-preserving goals, its distributed nature makes it particularly susceptible to such attacks. This paper examines the vulnerability of FL-based victim models to two types of model extraction attacks. For various federated clients built under the NVFlare platform, we implemented ME attacks across two deep learning architectures and three image datasets. We evaluate the proposed ME attack performance using various metrics, including accuracy, fidelity, and KL divergence. The experiments show that for different FL clients, the accuracy and fidelity of the extracted model are closely related to the size of the attack query set. Additionally, we explore a transfer learning based approach where pretrained models serve as the starting point for the extraction process. The results indicate that the accuracy and fidelity of the fine-tuned pretrained extraction models are notably higher, particularly with smaller query sets, highlighting potential advantages for attackers. 

---
# Rethinking the Understanding Ability across LLMs through Mutual Information 

**Authors**: Shaojie Wang, Sirui Ding, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23790)  

**Abstract**: Recent advances in large language models (LLMs) have revolutionized natural language processing, yet evaluating their intrinsic linguistic understanding remains challenging. Moving beyond specialized evaluation tasks, we propose an information-theoretic framework grounded in mutual information (MI) to achieve this. We formalize the understanding as MI between an input sentence and its latent representation (sentence-level MI), measuring how effectively input information is preserved in latent representation. Given that LLMs learn embeddings for individual tokens, we decompose sentence-level MI into token-level MI between tokens and sentence embeddings, establishing theoretical bounds connecting these measures. Based on this foundation, we theoretically derive a computable lower bound for token-level MI using Fano's inequality, which directly relates to token-level recoverability-the ability to predict original tokens from sentence embedding. We implement this recoverability task to comparatively measure MI across different LLMs, revealing that encoder-only models consistently maintain higher information fidelity than their decoder-only counterparts, with the latter exhibiting a distinctive late-layer "forgetting" pattern where mutual information is first enhanced and then discarded. Moreover, fine-tuning to maximize token-level recoverability consistently improves understanding ability of LLMs on tasks without task-specific supervision, demonstrating that mutual information can serve as a foundation for understanding and improving language model capabilities. 

---
# Nine Ways to Break Copyright Law and Why Our LLM Won't: A Fair Use Aligned Generation Framework 

**Authors**: Aakash Sen Sharma, Debdeep Sanyal, Priyansh Srivastava, Sundar Atreya H., Shirish Karande, Mohan Kankanhalli, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2505.23788)  

**Abstract**: Large language models (LLMs) commonly risk copyright infringement by reproducing protected content verbatim or with insufficient transformative modifications, posing significant ethical, legal, and practical concerns. Current inference-time safeguards predominantly rely on restrictive refusal-based filters, often compromising the practical utility of these models. To address this, we collaborated closely with intellectual property experts to develop FUA-LLM (Fair Use Aligned Language Models), a legally-grounded framework explicitly designed to align LLM outputs with fair-use doctrine. Central to our method is FairUseDB, a carefully constructed dataset containing 18,000 expert-validated examples covering nine realistic infringement scenarios. Leveraging this dataset, we apply Direct Preference Optimization (DPO) to fine-tune open-source LLMs, encouraging them to produce legally compliant and practically useful alternatives rather than resorting to blunt refusal. Recognizing the shortcomings of traditional evaluation metrics, we propose new measures: Weighted Penalty Utility and Compliance Aware Harmonic Mean (CAH) to balance infringement risk against response utility. Extensive quantitative experiments coupled with expert evaluations confirm that FUA-LLM substantially reduces problematic outputs (up to 20\%) compared to state-of-the-art approaches, while preserving real-world usability. 

---
# Mind the Gap: A Practical Attack on GGUF Quantization 

**Authors**: Kazuki Egashira, Robin Staab, Mark Vero, Jingxuan He, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.23786)  

**Abstract**: With the increasing size of frontier LLMs, post-training quantization has become the standard for memory-efficient deployment. Recent work has shown that basic rounding-based quantization schemes pose security risks, as they can be exploited to inject malicious behaviors into quantized models that remain hidden in full precision. However, existing attacks cannot be applied to more complex quantization methods, such as the GGUF family used in the popular ollama and this http URL frameworks. In this work, we address this gap by introducing the first attack on GGUF. Our key insight is that the quantization error -- the difference between the full-precision weights and their (de-)quantized version -- provides sufficient flexibility to construct malicious quantized models that appear benign in full precision. Leveraging this, we develop an attack that trains the target malicious LLM while constraining its weights based on quantization errors. We demonstrate the effectiveness of our attack on three popular LLMs across nine GGUF quantization data types on three diverse attack scenarios: insecure code generation ($\Delta$=$88.7\%$), targeted content injection ($\Delta$=$85.0\%$), and benign instruction refusal ($\Delta$=$30.1\%$). Our attack highlights that (1) the most widely used post-training quantization method is susceptible to adversarial interferences, and (2) the complexity of quantization schemes alone is insufficient as a defense. 

---
# Meaning Is Not A Metric: Using LLMs to make cultural context legible at scale 

**Authors**: Cody Kommers, Drew Hemment, Maria Antoniak, Joel Z. Leibo, Hoyt Long, Emily Robinson, Adam Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2505.23785)  

**Abstract**: This position paper argues that large language models (LLMs) can make cultural context, and therefore human meaning, legible at an unprecedented scale in AI-based sociotechnical systems. We argue that such systems have previously been unable to represent human meaning because they rely on thin descriptions: numerical representations that enforce standardization and therefore strip human activity of the cultural context that gives it meaning. By contrast, scholars in the humanities and qualitative social sciences have developed frameworks for representing meaning through thick description: verbal representations that accommodate heterogeneity and retain contextual information needed to represent human meaning. While these methods can effectively codify meaning, they are difficult to deploy at scale. However, the verbal capabilities of LLMs now provide a means of (at least partially) automating the generation and processing of thick descriptions, potentially overcoming this bottleneck. We argue that the problem of rendering human meaning legible is not just about selecting better metrics, but about developing new representational formats (based on thick description). We frame this as a crucial direction for the application of generative AI and identify five key challenges: preserving context, maintaining interpretive pluralism, integrating perspectives based on lived experience and critical distance, distinguishing qualitative content from quantitative magnitude, and acknowledging meaning as dynamic rather than static. Furthermore, we suggest that thick description has the potential to serve as a unifying framework to address a number of emerging concerns about the difficulties of representing culture in (or using) LLMs. 

---
# Boosting In-Context Learning in LLMs Through the Lens of Classical Supervised Learning 

**Authors**: Korel Gundem, Juncheng Dong, Dennis Zhang, Vahid Tarokh, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23783)  

**Abstract**: In-Context Learning (ICL) allows Large Language Models (LLMs) to adapt to new tasks with just a few examples, but their predictions often suffer from systematic biases, leading to unstable performances in classification. While calibration techniques are proposed to mitigate these biases, we show that, in the logit space, many of these methods are equivalent to merely shifting the LLM's decision boundary without having the ability to alter its orientation. This proves inadequate when biases cause the LLM to be severely misdirected. To address these limitations and provide a unifying framework, we propose Supervised Calibration (SC), a loss-minimization based framework which learns an optimal, per-class affine transformation of the LLM's predictive probabilities in the logit space without requiring external data beyond the context. By using a more expressive functional class, SC not only subsumes many existing calibration methods in ICL as special cases, but also enables the ability to alter and even completely reverse the orientation of the LLM's decision boundary. Furthermore, SC's loss-based nature facilitates the seamless integration of two purpose-built regularization techniques: context-invariance and directional trust-region. The former is designed to tackle the instability issue in ICL, while the latter controls the degree of calibration. Finally, SC delivers state-of-the-art performance over calibration baselines in the 4-shot, 8-shot, and 16-shot settings across all nine datasets for Mistral-7B-Instruct-v0.3, LLaMA-2-7B-chat, and Qwen2-7B-Instruct. 

---
# 4,500 Seconds: Small Data Training Approaches for Deep UAV Audio Classification 

**Authors**: Andrew P. Berg, Qian Zhang, Mia Y. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23782)  

**Abstract**: Unmanned aerial vehicle (UAV) usage is expected to surge in the coming decade, raising the need for heightened security measures to prevent airspace violations and security threats. This study investigates deep learning approaches to UAV classification focusing on the key issue of data scarcity. To investigate this we opted to train the models using a total of 4,500 seconds of audio samples, evenly distributed across a 9-class dataset. We leveraged parameter efficient fine-tuning (PEFT) and data augmentations to mitigate the data scarcity. This paper implements and compares the use of convolutional neural networks (CNNs) and attention-based transformers. Our results show that, CNNs outperform transformers by 1-2\% accuracy, while still being more computationally efficient. These early findings, however, point to potential in using transformers models; suggesting that with more data and further optimizations they could outperform CNNs. Future works aims to upscale the dataset to better understand the trade-offs between these approaches. 

---
# More-than-Human Storytelling: Designing Longitudinal Narrative Engagements with Generative AI 

**Authors**: Émilie Fabre, Katie Seaborn, Shuta Koiwai, Mizuki Watanabe, Paul Riesch  

**Link**: [PDF](https://arxiv.org/pdf/2505.23780)  

**Abstract**: Longitudinal engagement with generative AI (GenAI) storytelling agents is a timely but less charted domain. We explored multi-generational experiences with "Dreamsmithy," a daily dream-crafting app, where participants (N = 28) co-created stories with AI narrator "Makoto" every day. Reflections and interactions were captured through a two-week diary study. Reflexive thematic analysis revealed themes likes "oscillating ambivalence" and "socio-chronological bonding," highlighting the complex dynamics that emerged between individuals and the AI narrator over time. Findings suggest that while people appreciated the personal notes, opportunities for reflection, and AI creativity, limitations in narrative coherence and control occasionally caused frustration. The results underscore the potential of GenAI for longitudinal storytelling, but also raise critical questions about user agency and ethics. We contribute initial empirical insights and design considerations for developing adaptive, more-than-human storytelling systems. 

---
# A comprehensive survey of cybercrimes in India over the last decade 

**Authors**: Sudhanshu Sekhar Tripathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.23770)  

**Abstract**: Since the 1990s, the integration of technology into daily life has led to the creation of an extensive network of interconnected devices, transforming how individuals and organizations operate. However, this digital transformation has also spurred the rise of cybercrime, criminal activities perpetrated through networks or computer systems. Cybercrime has become a global concern, presenting significant challenges to security systems. Although advancements in digital technology have enhanced efficiency, they have also opened new avenues for exploitation by cybercriminals, highlighting the urgent need for advanced cybersecurity measures. The escalating number of cyberattacks and associated risks in the past decade highlights the critical importance of protecting sensitive data and safeguarding information systems. Cybercrimes range from financial fraud and phishing scams to identity theft and online harassment, posing substantial risks to both individuals and organizations. In response, governments, law enforcement agencies, and cybersecurity units have intensified their efforts to address these threats. In recent years, India has experienced a significant surge in cybercrime incidents, with a notable increase in cases involving ransomware, data breaches, and social engineering attacks. The growing penetration of internet services, the expansion of e-commerce, and the rapid adoption of digital payment systems have made individuals and organizations more vulnerable to cyber threats. Key areas affected include banking, healthcare, and government sectors, which are frequently targeted due to the sensitive nature of the data they handle. To combat these risks, there is an increasing focus on public awareness, cybersecurity education, and robust regulatory frameworks. This paper examines cybercrime, prevention strategies, security protocols, and terminology to safeguard digital infrastructure. 

---
# Few-Shot Optimization for Sensor Data Using Large Language Models: A Case Study on Fatigue Detection 

**Authors**: Elsen Ronando, Sozo Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2505.18754)  

**Abstract**: In this paper, we propose a novel few-shot optimization with HED-LM (Hybrid Euclidean Distance with Large Language Models) to improve example selection for sensor-based classification tasks. While few-shot prompting enables efficient inference with limited labeled data, its performance largely depends on the quality of selected examples. HED-LM addresses this challenge through a hybrid selection pipeline that filters candidate examples based on Euclidean distance and re-ranks them using contextual relevance scored by large language models (LLMs). To validate its effectiveness, we apply HED-LM to a fatigue detection task using accelerometer data characterized by overlapping patterns and high inter-subject variability. Unlike simpler tasks such as activity recognition, fatigue detection demands more nuanced example selection due to subtle differences in physiological signals. Our experiments show that HED-LM achieves a mean macro F1-score of 69.13$\pm$10.71%, outperforming both random selection (59.30$\pm$10.13%) and distance-only filtering (67.61$\pm$11.39%). These represent relative improvements of 16.6% and 2.3%, respectively. The results confirm that combining numerical similarity with contextual relevance improves the robustness of few-shot prompting. Overall, HED-LM offers a practical solution to improve performance in real-world sensor-based learning tasks and shows potential for broader applications in healthcare monitoring, human activity recognition, and industrial safety scenarios. 

---
# Towards Natural Language Communication for Cooperative Autonomous Driving via Self-Play 

**Authors**: Jiaxun Cui, Chen Tang, Jarrett Holtz, Janice Nguyen, Alessandro G. Allievi, Hang Qiu, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2505.18334)  

**Abstract**: Past work has demonstrated that autonomous vehicles can drive more safely if they communicate with one another than if they do not. However, their communication has often not been human-understandable. Using natural language as a vehicle-to-vehicle (V2V) communication protocol offers the potential for autonomous vehicles to drive cooperatively not only with each other but also with human drivers. In this work, we propose a suite of traffic tasks in autonomous driving where vehicles in a traffic scenario need to communicate in natural language to facilitate coordination in order to avoid an imminent collision and/or support efficient traffic flow. To this end, this paper introduces a novel method, LLM+Debrief, to learn a message generation and high-level decision-making policy for autonomous vehicles through multi-agent discussion. To evaluate LLM agents for driving, we developed a gym-like simulation environment that contains a range of driving scenarios. Our experimental results demonstrate that LLM+Debrief is more effective at generating meaningful and human-understandable natural language messages to facilitate cooperation and coordination than a zero-shot LLM agent. Our code and demo videos are available at this https URL. 

---
