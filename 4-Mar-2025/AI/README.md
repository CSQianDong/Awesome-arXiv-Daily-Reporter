# Do GFlowNets Transfer? Case Study on the Game of 24/42 

**Authors**: Adesh Gupta, Abhinav Kumar, Mansi Gupta, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2503.01819)  

**Abstract**: Generating diverse solutions is key to human-like reasoning, yet autoregressive language models focus on single accurate responses, limiting creativity. GFlowNets optimize solution generation as a flow network, promising greater diversity. Our case study shows their limited zero-shot transferability by fine-tuning small and medium-sized large language models on the Game of 24 and testing them on the Game of 42 datasets. Results revealed that GFlowNets struggle to maintain solution diversity and accuracy, highlighting key limitations in their cross-task generalization and the need for future research in improved transfer learning capabilities. 

---
# Generating Counterfactual Explanations Under Temporal Constraints 

**Authors**: Andrei Buliga, Chiara Di Francescomarino, Chiara Ghidini, Marco Montali, Massimiliano Ronzani  

**Link**: [PDF](https://arxiv.org/pdf/2503.01792)  

**Abstract**: Counterfactual explanations are one of the prominent eXplainable Artificial Intelligence (XAI) techniques, and suggest changes to input data that could alter predictions, leading to more favourable outcomes. Existing counterfactual methods do not readily apply to temporal domains, such as that of process mining, where data take the form of traces of activities that must obey to temporal background knowledge expressing which dynamics are possible and which not. Specifically, counterfactuals generated off-the-shelf may violate the background knowledge, leading to inconsistent explanations. This work tackles this challenge by introducing a novel approach for generating temporally constrained counterfactuals, guaranteed to comply by design with background knowledge expressed in Linear Temporal Logic on process traces (LTLp). We do so by infusing automata-theoretic techniques for LTLp inside a genetic algorithm for counterfactual generation. The empirical evaluation shows that the generated counterfactuals are temporally meaningful and more interpretable for applications involving temporal dependencies. 

---
# SAKE: Steering Activations for Knowledge Editing 

**Authors**: Marco Scialanga, Thibault Laugel, Vincent Grari, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01751)  

**Abstract**: As Large Langue Models have been shown to memorize real-world facts, the need to update this knowledge in a controlled and efficient manner arises. Designed with these constraints in mind, Knowledge Editing (KE) approaches propose to alter specific facts in pretrained models. However, they have been shown to suffer from several limitations, including their lack of contextual robustness and their failure to generalize to logical implications related to the fact. To overcome these issues, we propose SAKE, a steering activation method that models a fact to be edited as a distribution rather than a single prompt. Leveraging Optimal Transport, SAKE alters the LLM behavior over a whole fact-related distribution, defined as paraphrases and logical implications. Several numerical experiments demonstrate the effectiveness of this method: SAKE is thus able to perform more robust edits than its existing counterparts. 

---
# Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints 

**Authors**: Sam Bowyer, Laurence Aitchison, Desi R. Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2503.01747)  

**Abstract**: Rigorous statistical evaluations of large language models (LLMs), including valid error bars and significance testing, are essential for meaningful and reliable performance assessment. Currently, when such statistical measures are reported, they typically rely on the Central Limit Theorem (CLT). In this position paper, we argue that while CLT-based methods for uncertainty quantification are appropriate when benchmarks consist of thousands of examples, they fail to provide adequate uncertainty estimates for LLM evaluations that rely on smaller, highly specialized benchmarks. In these small-data settings, we demonstrate that CLT-based methods perform very poorly, usually dramatically underestimating uncertainty (i.e. producing error bars that are too small). We give recommendations for alternative frequentist and Bayesian methods that are both easy to implement and more appropriate in these increasingly common scenarios. We provide a simple Python library for these Bayesian methods at this https URL . 

---
# Learning Exposure Mapping Functions for Inferring Heterogeneous Peer Effects 

**Authors**: Shishir Adhikari, Sourav Medya, Elena Zheleva  

**Link**: [PDF](https://arxiv.org/pdf/2503.01722)  

**Abstract**: In causal inference, interference refers to the phenomenon in which the actions of peers in a network can influence an individual's outcome. Peer effect refers to the difference in counterfactual outcomes of an individual for different levels of peer exposure, the extent to which an individual is exposed to the treatments, actions, or behaviors of peers. Estimating peer effects requires deciding how to represent peer exposure. Typically, researchers define an exposure mapping function that aggregates peer treatments and outputs peer exposure. Most existing approaches for defining exposure mapping functions assume peer exposure based on the number or fraction of treated peers. Recent studies have investigated more complex functions of peer exposure which capture that different peers can exert different degrees of influence. However, none of these works have explicitly considered the problem of automatically learning the exposure mapping function. In this work, we focus on learning this function for the purpose of estimating heterogeneous peer effects, where heterogeneity refers to the variation in counterfactual outcomes for the same peer exposure but different individual's contexts. We develop EgoNetGNN, a graph neural network (GNN)-based method, to automatically learn the appropriate exposure mapping function allowing for complex peer influence mechanisms that, in addition to peer treatments, can involve the local neighborhood structure and edge attributes. We show that GNN models that use peer exposure based on the number or fraction of treated peers or learn peer exposure naively face difficulty accounting for such influence mechanisms. Our comprehensive evaluation on synthetic and semi-synthetic network data shows that our method is more robust to different unknown underlying influence mechanisms when estimating heterogeneous peer effects when compared to state-of-the-art baselines. 

---
# Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning 

**Authors**: Wenjie Wu, Yongcheng Jing, Yingjie Wang, Wenbin Hu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01642)  

**Abstract**: Recent large language model (LLM) reasoning, despite its success, suffers from limited domain knowledge, susceptibility to hallucinations, and constrained reasoning depth, particularly in small-scale models deployed in resource-constrained environments. This paper presents the first investigation into integrating step-wise knowledge graph retrieval with step-wise reasoning to address these challenges, introducing a novel paradigm termed as graph-augmented reasoning. Our goal is to enable frozen, small-scale LLMs to retrieve and process relevant mathematical knowledge in a step-wise manner, enhancing their problem-solving abilities without additional training. To this end, we propose KG-RAR, a framework centered on process-oriented knowledge graph construction, a hierarchical retrieval strategy, and a universal post-retrieval processing and reward model (PRP-RM) that refines retrieved information and evaluates each reasoning step. Experiments on the Math500 and GSM8K benchmarks across six models demonstrate that KG-RAR yields encouraging results, achieving a 20.73\% relative improvement with Llama-3B on Math500. 

---
# CoT-VLM4Tar: Chain-of-Thought Guided Vision-Language Models for Traffic Anomaly Resolution 

**Authors**: Tianchi Ren, Haibo Hu, Jiacheng Zuo, Xinhong Chen, Jianping Wang, Chun Jason Xue, Jen-Ming Wu, Nan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01632)  

**Abstract**: With the acceleration of urbanization, modern urban traffic systems are becoming increasingly complex, leading to frequent traffic anomalies. These anomalies encompass not only common traffic jams but also more challenging issues such as phantom traffic jams, intersection deadlocks, and accident liability analysis, which severely impact traffic flow, vehicular safety, and overall transportation efficiency. Currently, existing solutions primarily rely on manual intervention by traffic police or artificial intelligence-based detection systems. However, these methods often suffer from response delays and inconsistent management due to inadequate resources, while AI detection systems, despite enhancing efficiency to some extent, still struggle to handle complex traffic anomalies in a real-time and precise manner. To address these issues, we propose CoT-VLM4Tar: (Chain of Thought Visual-Language Model for Traffic Anomaly Resolution), this innovative approach introduces a new chain-of-thought to guide the VLM in analyzing, reasoning, and generating solutions for traffic anomalies with greater reasonable and effective solution, and to evaluate the performance and effectiveness of our method, we developed a closed-loop testing framework based on the CARLA simulator. Furthermore, to ensure seamless integration of the solutions generated by the VLM with the CARLA simulator, we implement an itegration module that converts these solutions into executable commands. Our results demonstrate the effectiveness of VLM in the resolution of real-time traffic anomalies, providing a proof-of-concept for its integration into autonomous traffic management systems. 

---
# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models 

**Authors**: Cansu Sancaktar, Christian Gumbsch, Andrii Zadaianchuk, Pavel Kolev, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.01584)  

**Abstract**: Exploration is a cornerstone of reinforcement learning (RL). Intrinsic motivation attempts to decouple exploration from external, task-based rewards. However, established approaches to intrinsic motivation that follow general principles such as information gain, often only uncover low-level interactions. In contrast, children's play suggests that they engage in meaningful high-level behavior by imitating or interacting with their caregivers. Recent work has focused on using foundation models to inject these semantic biases into exploration. However, these methods often rely on unrealistic assumptions, such as language-embedded environments or access to high-level actions. We propose SEmaNtically Sensible ExploratIon (SENSEI), a framework to equip model-based RL agents with an intrinsic motivation for semantically meaningful behavior. SENSEI distills a reward signal of interestingness from Vision Language Model (VLM) annotations, enabling an agent to predict these rewards through a world model. Using model-based RL, SENSEI trains an exploration policy that jointly maximizes semantic rewards and uncertainty. We show that in both robotic and video game-like simulations SENSEI discovers a variety of meaningful behaviors from image observations and low-level actions. SENSEI provides a general tool for learning from foundation model feedback, a crucial research direction, as VLMs become more powerful. 

---
# Enabling AI Scientists to Recognize Innovation: A Domain-Agnostic Algorithm for Assessing Novelty 

**Authors**: Yao Wang, Mingxuan Cui, Arthur Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01508)  

**Abstract**: In the pursuit of Artificial General Intelligence (AGI), automating the generation and evaluation of novel research ideas is a key challenge in AI-driven scientific discovery. This paper presents Relative Neighbor Density (RND), a domain-agnostic algorithm for novelty assessment in research ideas that overcomes the limitations of existing approaches by analyzing the distribution patterns of semantic neighbors rather than simple distances. We first developed a scalable methodology to create validation datasets without expert labeling, addressing a fundamental challenge in novelty assessment. Using these datasets, we demonstrate that our RND algorithm achieves state-of-the-art (SOTA) performance in computer science (AUROC=0.808) and biomedical research (AUROC=0.757) domains. Most significantly, while SOTA models like Sonnet-3.7 and existing metrics show domain-specific performance degradation, RND maintains consistent effectiveness across domains, outperforming all benchmarks by a substantial margin (0.782 v.s. 0.597) on cross-domain evaluation. These results validate RND as a generalizable solution for automated novelty assessment in scientific research. 

---
# ProRCA: A Causal Python Package for Actionable Root Cause Analysis in Real-world Business Scenarios 

**Authors**: Ahmed Dawoud, Shravan Talupula  

**Link**: [PDF](https://arxiv.org/pdf/2503.01475)  

**Abstract**: Root Cause Analysis (RCA) is becoming ever more critical as modern systems grow in complexity, volume of data, and interdependencies. While traditional RCA methods frequently rely on correlation-based or rule-based techniques, these approaches can prove inadequate in highly dynamic, multi-layered environments. In this paper, we present a pathway-tracing package built on the DoWhy causal inference library. Our method integrates conditional anomaly scoring, noise-based attribution, and depth-first path exploration to reveal multi-hop causal chains. By systematically tracing entire causal pathways from an observed anomaly back to the initial triggers, our approach provides a comprehensive, end-to-end RCA solution. Experimental evaluations with synthetic anomaly injections demonstrate the package's ability to accurately isolate triggers and rank root causes by their overall significance. 

---
# SrSv: Integrating Sequential Rollouts with Sequential Value Estimation for Multi-agent Reinforcement Learning 

**Authors**: Xu Wan, Chao Yang, Cheng Yang, Jie Song, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.01458)  

**Abstract**: Although multi-agent reinforcement learning (MARL) has shown its success across diverse domains, extending its application to large-scale real-world systems still faces significant challenges. Primarily, the high complexity of real-world environments exacerbates the credit assignment problem, substantially reducing training efficiency. Moreover, the variability of agent populations in large-scale scenarios necessitates scalable decision-making mechanisms. To address these challenges, we propose a novel framework: Sequential rollout with Sequential value estimation (SrSv). This framework aims to capture agent interdependence and provide a scalable solution for cooperative MARL. Specifically, SrSv leverages the autoregressive property of the Transformer model to handle varying populations through sequential action rollout. Furthermore, to capture the interdependence of policy distributions and value functions among multiple agents, we introduce an innovative sequential value estimation methodology and integrates the value approximation into an attention-based sequential model. We evaluate SrSv on three benchmarks: Multi-Agent MuJoCo, StarCraft Multi-Agent Challenge, and DubinsCars. Experimental results demonstrate that SrSv significantly outperforms baseline methods in terms of training efficiency without compromising convergence performance. Moreover, when implemented in a large-scale DubinsCar system with 1,024 agents, our framework surpasses existing benchmarks, highlighting the excellent scalability of SrSv. 

---
# From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems 

**Authors**: Zekun Zhou, Xiaocheng Feng, Lei Huang, Xiachong Feng, Ziyun Song, Ruihan Chen, Liang Zhao, Weitao Ma, Yuxuan Gu, Baoxin Wang, Dayong Wu, Guoping Hu, Ting Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01424)  

**Abstract**: Research is a fundamental process driving the advancement of human civilization, yet it demands substantial time and effort from researchers. In recent years, the rapid development of artificial intelligence (AI) technologies has inspired researchers to explore how AI can accelerate and enhance research. To monitor relevant advancements, this paper presents a systematic review of the progress in this domain. Specifically, we organize the relevant studies into three main categories: hypothesis formulation, hypothesis validation, and manuscript publication. Hypothesis formulation involves knowledge synthesis and hypothesis generation. Hypothesis validation includes the verification of scientific claims, theorem proving, and experiment validation. Manuscript publication encompasses manuscript writing and the peer review process. Furthermore, we identify and discuss the current challenges faced in these areas, as well as potential future directions for research. Finally, we also offer a comprehensive overview of existing benchmarks and tools across various domains that support the integration of AI into the research process. We hope this paper serves as an introduction for beginners and fosters future research. Resources have been made publicly available at this https URL. 

---
# Building Interval Type-2 Fuzzy Membership Function: A Deck of Cards based Co-constructive Approach 

**Authors**: Bapi Dutta, Diego García-Zamora, José Rui Figueira, Luis Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2503.01413)  

**Abstract**: Since its inception, Fuzzy Set has been widely used to handle uncertainty and imprecision in decision-making. However, conventional fuzzy sets, often referred to as type-1 fuzzy sets (T1FSs) have limitations in capturing higher levels of uncertainty, particularly when decision-makers (DMs) express hesitation or ambiguity in membership degree. To address this, Interval Type-2 Fuzzy Sets (IT2FSs) have been introduced by incorporating uncertainty in membership degree allocation, which enhanced flexibility in modelling subjective judgments. Despite their advantages, existing IT2FS construction methods often lack active involvement from DMs and that limits the interpretability and effectiveness of decision models. This study proposes a socio-technical co-constructive approach for developing IT2FS models of linguistic terms by facilitating the active involvement of DMs in preference elicitation and its application in multicriteria decision-making (MCDM) problems. Our methodology is structured in two phases. The first phase involves an interactive process between the DM and the decision analyst, in which a modified version of Deck-of-Cards (DoC) method is proposed to construct T1FS membership functions on a ratio scale. We then extend this method to incorporate ambiguity in subjective judgment and that resulted in an IT2FS model that better captures uncertainty in DM's linguistic assessments. The second phase formalizes the constructed IT2FS model for application in MCDM by defining an appropriate mathematical representation of such information, aggregation rules, and an admissible ordering principle. The proposed framework enhances the reliability and effectiveness of fuzzy decision-making not only by accurately representing DM's personalized semantics of linguistic information. 

---
# Learning Conjecturing from Scratch 

**Authors**: Thibault Gauthier, Josef Urban  

**Link**: [PDF](https://arxiv.org/pdf/2503.01389)  

**Abstract**: We develop a self-learning approach for conjecturing of induction predicates on a dataset of 16197 problems derived from the OEIS. These problems are hard for today's SMT and ATP systems because they require a combination of inductive and arithmetical reasoning.
Starting from scratch, our approach consists of a feedback loop that iterates between (i) training a neural translator to learn the correspondence between the problems solved so far and the induction predicates useful for them, (ii) using the trained neural system to generate many new induction predicates for the problems, (iii) fast runs of the z3 prover attempting to prove the problems using the generated predicates, (iv) using heuristics such as predicate size and solution speed on the proved problems to choose the best predicates for the next iteration of training.
The algorithm discovers on its own many interesting induction predicates, ultimately solving 5565 problems, compared to 2265 problems solved by CVC5, Vampire or Z3 in 60 seconds. 

---
# OptMetaOpenFOAM: Large Language Model Driven Chain of Thought for Sensitivity Analysis and Parameter Optimization based on CFD 

**Authors**: Yuxuan Chen, Long Zhang, Xu Zhu, Hua Zhou, Zhuyin Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01273)  

**Abstract**: Merging natural language interfaces with computational fluid dynamics (CFD) workflows presents transformative opportunities for both industry and research. In this study, we introduce OptMetaOpenFOAM - a novel framework that bridges MetaOpenFOAM with external analysis and optimization tool libraries through a large language model (LLM)-driven chain-of-thought (COT) methodology. By automating complex CFD tasks via natural language inputs, the framework empowers non-expert users to perform sensitivity analyses and parameter optimizations with markedly improved efficiency. The test dataset comprises 11 distinct CFD analysis or optimization tasks, including a baseline simulation task derived from an OpenFOAM tutorial covering fluid dynamics, combustion, and heat transfer. Results confirm that OptMetaOpenFOAM can accurately interpret user requirements expressed in natural language and effectively invoke external tool libraries alongside MetaOpenFOAM to complete the tasks. Furthermore, validation on a non-OpenFOAM tutorial case - namely, a hydrogen combustion chamber - demonstrates that a mere 200-character natural language input can trigger a sequence of simulation, postprocessing, analysis, and optimization tasks spanning over 2,000 lines of code. These findings underscore the transformative potential of LLM-driven COT methodologies in linking external tool for advanced analysis and optimization, positioning OptMetaOpenFOAM as an effective tool that streamlines CFD simulations and enhances their convenience and efficiency for both industrial and research applications. Code is available at this https URL. 

---
# Prognostics and Health Management of Wafer Chemical-Mechanical Polishing System using Autoencoder 

**Authors**: Kart-Leong Lim, Rahul Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2503.01176)  

**Abstract**: The Prognostics and Health Management Data Challenge (PHM) 2016 tracks the health state of components of a semiconductor wafer polishing process. The ultimate goal is to develop an ability to predict the measurement on the wafer surface wear through monitoring the components health state. This translates to cost saving in large scale production. The PHM dataset contains many time series measurements not utilized by traditional physics based approach. On the other hand task, applying a data driven approach such as deep learning to the PHM dataset is non-trivial. The main issue with supervised deep learning is that class label is not available to the PHM dataset. Second, the feature space trained by an unsupervised deep learner is not specifically targeted at the predictive ability or regression. In this work, we propose using the autoencoder based clustering whereby the feature space trained is found to be more suitable for performing regression. This is due to having a more compact distribution of samples respective to their nearest cluster means. We justify our claims by comparing the performance of our proposed method on the PHM dataset with several baselines such as the autoencoder as well as state-of-the-art approaches. 

---
# Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers 

**Authors**: Rin Ashizawa, Yoichi Hirose, Nozomu Yoshinari, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.01163)  

**Abstract**: Prompt optimization aims to search for effective prompts that enhance the performance of large language models (LLMs). Although existing prompt optimization methods have discovered effective prompts, they often differ from sophisticated prompts carefully designed by human experts. Prompt design strategies, representing best practices for improving prompt performance, can be key to improving prompt optimization. Recently, a method termed the Autonomous Prompt Engineering Toolbox (APET) has incorporated various prompt design strategies into the prompt optimization process. In APET, the LLM is needed to implicitly select and apply the appropriate strategies because prompt design strategies can have negative effects. This implicit selection may be suboptimal due to the limited optimization capabilities of LLMs. This paper introduces Optimizing Prompts with sTrategy Selection (OPTS), which implements explicit selection mechanisms for prompt design. We propose three mechanisms, including a Thompson sampling-based approach, and integrate them into EvoPrompt, a well-known prompt optimizer. Experiments optimizing prompts for two LLMs, Llama-3-8B-Instruct and GPT-4o mini, were conducted using BIG-Bench Hard. Our results show that the selection of prompt design strategies improves the performance of EvoPrompt, and the Thompson sampling-based mechanism achieves the best overall results. Our experimental code is provided at this https URL . 

---
# Can Large Language Models Help Experimental Design for Causal Discovery? 

**Authors**: Junyi Li, Yongqiang Chen, Chenxi Liu, Qianyi Cai, Tongliang Liu, Bo Han, Kun Zhang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01139)  

**Abstract**: Designing proper experiments and selecting optimal intervention targets is a longstanding problem in scientific or causal discovery. Identifying the underlying causal structure from observational data alone is inherently this http URL interventional data, on the other hand, is crucial to causal discovery, yet it is usually expensive and time-consuming to gather sufficient interventional data to facilitate causal this http URL approaches commonly utilize uncertainty or gradient signals to determine the intervention targets. However, numerical-based approaches may yield suboptimal results due to the inaccurate estimation of the guiding signals at the beginning when with limited interventional data. In this work, we investigate a different approach, whether we can leverage Large Language Models (LLMs) to assist with the intervention targeting in causal discovery by making use of the rich world knowledge about the experimental design in this http URL, we present \oursfull (\ours) -- a robust framework that effectively incorporates LLMs to augment existing numerical approaches for the intervention targeting in causal discovery. Across $4$ realistic benchmark scales, \ours demonstrates significant improvements and robustness over existing methods and even surpasses humans, which demonstrates the usefulness of LLMs in assisting with experimental design for scientific discovery. 

---
# Constrained multi-fidelity Bayesian optimization with automatic stop condition 

**Authors**: Zahra Zanjani Foumani, Ramin Bostanabad  

**Link**: [PDF](https://arxiv.org/pdf/2503.01126)  

**Abstract**: Bayesian optimization (BO) is increasingly employed in critical applications to find the optimal design with minimal cost. While BO is known for its sample efficiency, relying solely on costly high-fidelity data can still result in high costs. This is especially the case in constrained search spaces where BO must not only optimize but also ensure feasibility. A related issue in the BO literature is the lack of a systematic stopping criterion. To solve these challenges, we develop a constrained cost-aware multi-fidelity BO (CMFBO) framework whose goal is to minimize overall sampling costs by utilizing inexpensive low-fidelity sources while ensuring feasibility. In our case, the constraints can change across the data sources and may be even black-box functions. We also introduce a systematic stopping criterion that addresses the long-lasting issue associated with BO's convergence assessment. Our framework is publicly available on GitHub through the GP+ Python package and herein we validate it's efficacy on multiple benchmark problems. 

---
# Hybrid Metaheuristic Vehicle Routing Problem for Security Dispatch Operations 

**Authors**: Nguyen Gia Hien Vu, Yifan Tang, Rey Lim, G. Gary Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01121)  

**Abstract**: This paper investigates the optimization of the Vehicle Routing Problem for Security Dispatch (VRPSD). VRPSD focuses on security and patrolling applications which involve challenging constraints including precise timing and strict time windows. We propose three algorithms based on different metaheuristics, which are Adaptive Large Neighborhood Search (ALNS), Tabu Search (TS), and Threshold Accepting (TA). The first algorithm combines single-phase ALNS with TA, the second employs a multiphase ALNS with TA, and the third integrates multiphase ALNS, TS, and TA. Experiments are conducted on an instance comprising 251 customer requests. The results demonstrate that the third algorithm, the hybrid multiphase ALNS-TS-TA algorithm, delivers the best performance. This approach simultaneously leverages the large-area search capabilities of ALNS for exploration and effectively escapes local optima when the multiphase ALNS is coupled with TS and TA. Furthermore, in our experiments, the hybrid multiphase ALNS-TS-TA algorithm is the only one that shows potential for improving results with increased computation time across all attempts. 

---
# FAIR: Facilitating Artificial Intelligence Resilience in Manufacturing Industrial Internet 

**Authors**: Yingyan Zeng, Ismini Lourentzou, Xinwei Deng, Ran Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01086)  

**Abstract**: Artificial intelligence (AI) systems have been increasingly adopted in the Manufacturing Industrial Internet (MII). Investigating and enabling the AI resilience is very important to alleviate profound impact of AI system failures in manufacturing and Industrial Internet of Things (IIoT) operations, leading to critical decision making. However, there is a wide knowledge gap in defining the resilience of AI systems and analyzing potential root causes and corresponding mitigation strategies. In this work, we propose a novel framework for investigating the resilience of AI performance over time under hazard factors in data quality, AI pipelines, and the cyber-physical layer. The proposed method can facilitate effective diagnosis and mitigation strategies to recover AI performance based on a multimodal multi-head self latent attention model. The merits of the proposed method are elaborated using an MII testbed of connected Aerosol Jet Printing (AJP) machines, fog nodes, and Cloud with inference tasks via AI pipelines. 

---
# Multi-Agent Reinforcement Learning with Long-Term Performance Objectives for Service Workforce Optimization 

**Authors**: Kareem Eissa, Rayal Prasad, Sarith Mohan, Ankur Kapoor, Dorin Comaniciu, Vivek Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01069)  

**Abstract**: Workforce optimization plays a crucial role in efficient organizational operations where decision-making may span several different administrative and time scales. For instance, dispatching personnel to immediate service requests while managing talent acquisition with various expertise sets up a highly dynamic optimization problem. Existing work focuses on specific sub-problems such as resource allocation and facility location, which are solved with heuristics like local-search and, more recently, deep reinforcement learning. However, these may not accurately represent real-world scenarios where such sub-problems are not fully independent. Our aim is to fill this gap by creating a simulator that models a unified workforce optimization problem. Specifically, we designed a modular simulator to support the development of reinforcement learning methods for integrated workforce optimization problems. We focus on three interdependent aspects: personnel dispatch, workforce management, and personnel positioning. The simulator provides configurable parameterizations to help explore dynamic scenarios with varying levels of stochasticity and non-stationarity. To facilitate benchmarking and ablation studies, we also include heuristic and RL baselines for the above mentioned aspects. 

---
# An Exact Solver for Satisfiability Modulo Counting with Probabilistic Circuits 

**Authors**: Jinzhao Li, Nan Jiang, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.01009)  

**Abstract**: Satisfiability Modulo Counting (SMC) is a recently proposed general language to reason about problems integrating statistical and symbolic artificial intelligence. An SMC formula is an extended SAT formula in which the truth values of a few Boolean variables are determined by probabilistic inference. Existing approximate solvers optimize surrogate objectives, which lack formal guarantees. Current exact solvers directly integrate SAT solvers and probabilistic inference solvers resulting in slow performance because of many back-and-forth invocations of both solvers. We propose KOCO-SMC, an integrated exact SMC solver that efficiently tracks lower and upper bounds in the probabilistic inference process. It enhances computational efficiency by enabling early estimation of probabilistic inference using only partial variable assignments, whereas existing methods require full variable assignments. In the experiment, we compare KOCO-SMC with currently available approximate and exact SMC solvers on large-scale datasets and real-world applications. Our approach delivers high-quality solutions with high efficiency. 

---
# Evidence of conceptual mastery in the application of rules by Large Language Models 

**Authors**: José Luiz Nunes, Guilherme FCF Almeida, Brian Flanagan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00992)  

**Abstract**: In this paper we leverage psychological methods to investigate LLMs' conceptual mastery in applying rules. We introduce a novel procedure to match the diversity of thought generated by LLMs to that observed in a human sample. We then conducted two experiments comparing rule-based decision-making in humans and LLMs. Study 1 found that all investigated LLMs replicated human patterns regardless of whether they are prompted with scenarios created before or after their training cut-off. Moreover, we found unanticipated differences between the two sets of scenarios among humans. Surprisingly, even these differences were replicated in LLM responses. Study 2 turned to a contextual feature of human rule application: under forced time delay, human samples rely more heavily on a rule's text than on other considerations such as a rule's purpose.. Our results revealed that some models (Gemini Pro and Claude 3) responded in a human-like manner to a prompt describing either forced delay or time pressure, while others (GPT-4o and Llama 3.2 90b) did not. We argue that the evidence gathered suggests that LLMs have mastery over the concept of rule, with implications for both legal decision making and philosophical inquiry. 

---
# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains 

**Authors**: Wonje Choi, Jinwoo Park, Sanghyun Ahn, Daehee Lee, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00870)  

**Abstract**: We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge. To address this issue, we introduce a novel framework, NeSyC, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continually formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, we devise a contrastive generality improvement scheme within NeSyC, which iteratively generates hypotheses using LLMs and conducts contrastive validation via symbolic tools. This scheme reinforces the justification for admissible actions while minimizing the inference of inadmissible ones. Additionally, we incorporate a memory-based monitoring scheme that efficiently detects action errors and triggers the knowledge refinement process across domains. Experiments conducted on diverse embodied task benchmarks-including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario-demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain environments. 

---
# A Law Reasoning Benchmark for LLM with Tree-Organized Structures including Factum Probandum, Evidence and Experiences 

**Authors**: Jiaxin Shen, Jinan Xu, Huiqi Hu, Luyi Lin, Fei Zheng, Guoyang Ma, Fandong Meng, Jie Zhou, Wenjuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00841)  

**Abstract**: While progress has been made in legal applications, law reasoning, crucial for fair adjudication, remains unexplored. We propose a transparent law reasoning schema enriched with hierarchical factum probandum, evidence, and implicit experience, enabling public scrutiny and preventing bias. Inspired by this schema, we introduce the challenging task, which takes a textual case description and outputs a hierarchical structure justifying the final decision. We also create the first crowd-sourced dataset for this task, enabling comprehensive evaluation. Simultaneously, we propose an agent framework that employs a comprehensive suite of legal analysis tools to address the challenge task. This benchmark paves the way for transparent and accountable AI-assisted law reasoning in the ``Intelligent Court''. 

---
# Rethinking Light Decoder-based Solvers for Vehicle Routing Problems 

**Authors**: Ziwei Huang, Jianan Zhou, Zhiguang Cao, Yixin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00753)  

**Abstract**: Light decoder-based solvers have gained popularity for solving vehicle routing problems (VRPs) due to their efficiency and ease of integration with reinforcement learning algorithms. However, they often struggle with generalization to larger problem instances or different VRP variants. This paper revisits light decoder-based approaches, analyzing the implications of their reliance on static embeddings and the inherent challenges that arise. Specifically, we demonstrate that in the light decoder paradigm, the encoder is implicitly tasked with capturing information for all potential decision scenarios during solution construction within a single set of embeddings, resulting in high information density. Furthermore, our empirical analysis reveals that the overly simplistic decoder struggles to effectively utilize this dense information, particularly as task complexity increases, which limits generalization to out-of-distribution (OOD) settings. Building on these insights, we show that enhancing the decoder capacity, with a simple addition of identity mapping and a feed-forward layer, can considerably alleviate the generalization issue. Experimentally, our method significantly enhances the OOD generalization of light decoder-based approaches on large-scale instances and complex VRP variants, narrowing the gap with the heavy decoder paradigm. Our code is available at: this https URL. 

---
# Modeling Arbitrarily Applicable Relational Responding with the Non-Axiomatic Reasoning System: A Machine Psychology Approach 

**Authors**: Robert Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2503.00611)  

**Abstract**: Arbitrarily Applicable Relational Responding (AARR) is a cornerstone of human language and reasoning, referring to the learned ability to relate symbols in flexible, context-dependent ways. In this paper, we present a novel theoretical approach for modeling AARR within an artificial intelligence framework using the Non-Axiomatic Reasoning System (NARS). NARS is an adaptive reasoning system designed for learning under uncertainty. By integrating principles from Relational Frame Theory - the behavioral psychology account of AARR - with the reasoning mechanisms of NARS, we conceptually demonstrate how key properties of AARR (mutual entailment, combinatorial entailment, and transformation of stimulus functions) can emerge from the inference rules and memory structures of NARS. Two theoretical experiments illustrate this approach: one modeling stimulus equivalence and transfer of function, and another modeling complex relational networks involving opposition frames. In both cases, the system logically demonstrates the derivation of untrained relations and context-sensitive transformations of stimulus significance, mirroring established human cognitive phenomena. These results suggest that AARR - long considered uniquely human - can be conceptually captured by suitably designed AI systems, highlighting the value of integrating behavioral science insights into artificial general intelligence (AGI) research. 

---
# Instructor-Worker Large Language Model System for Policy Recommendation: a Case Study on Air Quality Analysis of the January 2025 Los Angeles Wildfires 

**Authors**: Kyle Gao, Dening Lu, Liangzhi Li, Nan Chen, Hongjie He, Linlin Xu, Jonathan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00566)  

**Abstract**: The Los Angeles wildfires of January 2025 caused more than 250 billion dollars in damage and lasted for nearly an entire month before containment. Following our previous work, the Digital Twin Building, we modify and leverage the multi-agent large language model framework as well as the cloud-mapping integration to study the air quality during the Los Angeles wildfires. Recent advances in large language models have allowed for out-of-the-box automated large-scale data analysis. We use a multi-agent large language system comprised of an Instructor agent and Worker agents. Upon receiving the users' instructions, the Instructor agent retrieves the data from the cloud platform and produces instruction prompts to the Worker agents. The Worker agents then analyze the data and provide summaries. The summaries are finally input back into the Instructor agent, which then provides the final data analysis. We test this system's capability for data-based policy recommendation by assessing our Instructor-Worker LLM system's health recommendations based on air quality during the Los Angeles wildfires. 

---
# Human-AI Collaboration: Trade-offs Between Performance and Preferences 

**Authors**: Lukas William Mayer, Sheer Karny, Jackie Ayoub, Miao Song, Danyang Tian, Ehsan Moradi-Pari, Mark Steyvers  

**Link**: [PDF](https://arxiv.org/pdf/2503.00248)  

**Abstract**: Despite the growing interest in collaborative AI, designing systems that seamlessly integrate human input remains a major challenge. In this study, we developed a task to systematically examine human preferences for collaborative agents. We created and evaluated five collaborative AI agents with strategies that differ in the manner and degree they adapt to human actions. Participants interacted with a subset of these agents, evaluated their perceived traits, and selected their preferred agent. We used a Bayesian model to understand how agents' strategies influence the Human-AI team performance, AI's perceived traits, and the factors shaping human-preferences in pairwise agent comparisons. Our results show that agents who are more considerate of human actions are preferred over purely performance-maximizing agents. Moreover, we show that such human-centric design can improve the likability of AI collaborators without reducing performance. We find evidence for inequality-aversion effects being a driver of human choices, suggesting that people prefer collaborative agents which allow them to meaningfully contribute to the team. Taken together, these findings demonstrate how collaboration with AI can benefit from development efforts which include both subjective and objective metrics. 

---
# Agentic AI Needs a Systems Theory 

**Authors**: Erik Miehling, Karthikeyan Natesan Ramamurthy, Kush R. Varshney, Matthew Riemer, Djallel Bouneffouf, John T. Richards, Amit Dhurandhar, Elizabeth M. Daly, Michael Hind, Prasanna Sattigeri, Dennis Wei, Ambrish Rawat, Jasmina Gajcin, Werner Geyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00237)  

**Abstract**: The endowment of AI with reasoning capabilities and some degree of agency is widely viewed as a path toward more capable and generalizable systems. Our position is that the current development of agentic AI requires a more holistic, systems-theoretic perspective in order to fully understand their capabilities and mitigate any emergent risks. The primary motivation for our position is that AI development is currently overly focused on individual model capabilities, often ignoring broader emergent behavior, leading to a significant underestimation in the true capabilities and associated risks of agentic AI. We describe some fundamental mechanisms by which advanced capabilities can emerge from (comparably simpler) agents simply due to their interaction with the environment and other agents. Informed by an extensive amount of existing literature from various fields, we outline mechanisms for enhanced agent cognition, emergent causal reasoning ability, and metacognitive awareness. We conclude by presenting some key open challenges and guidance for the development of agentic AI. We emphasize that a systems-level perspective is essential for better understanding, and purposefully shaping, agentic AI systems. 

---
# Jailbreaking Safeguarded Text-to-Image Models via Large Language Models 

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01839)  

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks. 

---
# CrowdSelect: Synthetic Instruction Data Selection with Multi-LLM Wisdom 

**Authors**: Yisen Li, Lingfeng Yang, Wenxuan Shen, Pan Zhou, Yao Wan, Weiwei Lin, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01836)  

**Abstract**: Distilling advanced Large Language Models' instruction-following capabilities into smaller models using a selected subset has become a mainstream approach in model training. While existing synthetic instruction data selection strategies rely mainly on single-dimensional signals (i.e., reward scores, model perplexity), they fail to capture the complexity of instruction-following across diverse fields. Therefore, we investigate more diverse signals to capture comprehensive instruction-response pair characteristics and propose three foundational metrics that leverage Multi-LLM wisdom, informed by (1) diverse LLM responses and (2) reward model assessment. Building upon base metrics, we propose CrowdSelect, an integrated metric incorporating a clustering-based approach to maintain response diversity. Our comprehensive experiments demonstrate that our foundation metrics consistently improve performance across 4 base models on MT-bench and Arena-Hard. CrowdSelect, efficiently incorporating all metrics, achieves state-of-the-art performance in both Full and LoRA fine-tuning, showing improvements of 4.81% on Arena-Hard and 11.1% on MT-bench with Llama-3.2-3b-instruct. We hope our findings will bring valuable insights for future research in this direction. Code are available at this https URL. 

---
# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2503.01829)  

**Abstract**: Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Moreover, LLMs' susceptibility to persuasion raises concerns about alignment with ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasion through multi-agent interactions. Here, Persuader agents engage in multi-turn conversations with the Persuadee agents, allowing us to measure LLMs' persuasive effectiveness and their susceptibility to persuasion. We conduct comprehensive evaluations across diverse LLMs, ensuring each model is assessed against others in both subjective and misinformation contexts. We validate the efficacy of our framework through human evaluations and show alignment with prior work. PMIYC offers a scalable alternative to human annotation for studying persuasion in LLMs. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems. 

---
# Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry 

**Authors**: Sai Sumedh R. Hindupur, Ekdeep Singh Lubana, Thomas Fel, Demba Ba  

**Link**: [PDF](https://arxiv.org/pdf/2503.01822)  

**Abstract**: Sparse Autoencoders (SAEs) are widely used to interpret neural networks by identifying meaningful concepts from their representations. However, do SAEs truly uncover all concepts a model relies on, or are they inherently biased toward certain kinds of concepts? We introduce a unified framework that recasts SAEs as solutions to a bilevel optimization problem, revealing a fundamental challenge: each SAE imposes structural assumptions about how concepts are encoded in model representations, which in turn shapes what it can and cannot detect. This means different SAEs are not interchangeable -- switching architectures can expose entirely new concepts or obscure existing ones. To systematically probe this effect, we evaluate SAEs across a spectrum of settings: from controlled toy models that isolate key variables, to semi-synthetic experiments on real model activations and finally to large-scale, naturalistic datasets. Across this progression, we examine two fundamental properties that real-world concepts often exhibit: heterogeneity in intrinsic dimensionality (some concepts are inherently low-dimensional, others are not) and nonlinear separability. We show that SAEs fail to recover concepts when these properties are ignored, and we design a new SAE that explicitly incorporates both, enabling the discovery of previously hidden concepts and reinforcing our theoretical insights. Our findings challenge the idea of a universal SAE and underscores the need for architecture-specific choices in model interpretability. Overall, we argue an SAE does not just reveal concepts -- it determines what can be seen at all. 

---
# RSQ: Learning from Important Tokens Leads to Better Quantized LLMs 

**Authors**: Yi-Lin Sung, Prateek Yadav, Jialu Li, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.01820)  

**Abstract**: Layer-wise quantization is a key technique for efficiently compressing large models without expensive retraining. Previous methods typically quantize the weights of each layer by "uniformly" optimizing the layer reconstruction loss across all output tokens. However, in this paper, we demonstrate that better-quantized models can be obtained by prioritizing learning from important tokens (e.g. which have large attention scores). Building on this finding, we propose RSQ (Rotate, Scale, then Quantize), which (1) applies rotations (orthogonal transformation) to the model to mitigate outliers (those with exceptionally large magnitude), (2) scales the token feature based on its importance, and (3) quantizes the model using the GPTQ framework with the second-order statistics computed by scaled tokens. To compute token importance, we explore both heuristic and dynamic strategies. Based on a thorough analysis of all approaches, we adopt attention concentration, which uses attention scores of each token as its importance, as the best approach. We demonstrate that RSQ consistently outperforms baseline methods across multiple downstream tasks and three model families: LLaMA3, Mistral, and Qwen2.5. Additionally, models quantized with RSQ achieve superior performance on long-context tasks, further highlighting its effectiveness. Lastly, RSQ demonstrates generalizability across various setups, including different model sizes, calibration datasets, bit precisions, and quantization methods. 

---
# LLMInit: A Free Lunch from Large Language Models for Selective Initialization of Recommendation 

**Authors**: Weizhi Zhang, Liangwei Yang, Wooseong Yang, Henry Peng Zou, Yuqing Liu, Ke Xu, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01814)  

**Abstract**: Collaborative filtering models, particularly graph-based approaches, have demonstrated strong performance in capturing user-item interactions for recommendation systems. However, they continue to struggle in cold-start and data-sparse scenarios. The emergence of large language models (LLMs) like GPT and LLaMA presents new possibilities for enhancing recommendation performance, especially in cold-start settings. Despite their promise, LLMs pose challenges related to scalability and efficiency due to their high computational demands and limited ability to model complex user-item relationships effectively. In this work, we introduce a novel perspective on leveraging LLMs for CF model initialization. Through experiments, we uncover an embedding collapse issue when scaling CF models to larger embedding dimensions. To effectively harness large-scale LLM embeddings, we propose innovative selective initialization strategies utilizing random, uniform, and variance-based index sampling. Our comprehensive evaluation on multiple real-world datasets demonstrates significant performance gains across various CF models while maintaining a lower computational cost compared to existing LLM-based recommendation approaches. 

---
# AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses 

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2503.01811)  

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at this https URL. 

---
# Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers 

**Authors**: Gilad Yehudai, Clayton Sanford, Maya Bechler-Speicher, Orr Fischer, Ran Gilad-Bachrach, Amir Globerson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01805)  

**Abstract**: Transformers have revolutionized the field of machine learning. In particular, they can be used to solve complex algorithmic problems, including graph-based tasks. In such algorithmic tasks a key question is what is the minimal size of a transformer that can implement a task. Recent work has begun to explore this problem for graph-based tasks, showing that for sub-linear embedding dimension (i.e., model width) logarithmic depth suffices. However, an open question, which we address here, is what happens if width is allowed to grow linearly. Here we analyze this setting, and provide the surprising result that with linear width, constant depth suffices for solving a host of graph-based problems. This suggests that a moderate increase in width can allow much shallower models, which are advantageous in terms of inference time. For other problems, we show that quadratic width is required. Our results demonstrate the complex and intriguing landscape of transformer implementations of graph-based algorithms. We support our theoretical results with empirical evaluations. 

---
# $\texttt{SEM-CTRL}$: Semantically Controlled Decoding 

**Authors**: Mohammad Albinhassan, Pranava Madhyastha, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01804)  

**Abstract**: Ensuring both syntactic and semantic correctness in Large Language Model (LLM) outputs remains a significant challenge, despite being critical for real-world deployment. In this paper, we introduce $\texttt{SEM-CTRL}$, a unified approach that enforces rich context-sensitive constraints and task- and instance-specific semantics directly on an LLM decoder. Our approach integrates token-level MCTS, which is guided by specific syntactic and semantic constraints. The constraints over the desired outputs are expressed using Answer Set Grammars -- a logic-based formalism that generalizes context-sensitive grammars while incorporating background knowledge to represent task-specific semantics. We show that our approach guarantees correct completions for any off-the-shelf LLM without the need for fine-tuning. We evaluate $\texttt{SEM-CTRL}$ on a range of tasks, including synthetic grammar synthesis, combinatorial reasoning, and planning. Our results demonstrate that $\texttt{SEM-CTRL}$ allows small pre-trained LLMs to efficiently outperform larger variants and state-of-the-art reasoning models (e.g., o1-preview) while simultaneously guaranteeing solution correctness. 

---
# Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation 

**Authors**: Tiansheng Wen, Yifei Wang, Zequn Zeng, Zhong Peng, Yudi Su, Xinyang Liu, Bo Chen, Hongwei Liu, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2503.01776)  

**Abstract**: Many large-scale systems rely on high-quality deep representations (embeddings) to facilitate tasks like retrieval, search, and generative modeling. Matryoshka Representation Learning (MRL) recently emerged as a solution for adaptive embedding lengths, but it requires full model retraining and suffers from noticeable performance degradations at short lengths. In this paper, we show that sparse coding offers a compelling alternative for achieving adaptive representation with minimal overhead and higher fidelity. We propose Contrastive Sparse Representation (CSR), a method that sparsifies pre-trained embeddings into a high-dimensional but selectively activated feature space. By leveraging lightweight autoencoding and task-aware contrastive objectives, CSR preserves semantic quality while allowing flexible, cost-effective inference at different sparsity levels. Extensive experiments on image, text, and multimodal benchmarks demonstrate that CSR consistently outperforms MRL in terms of both accuracy and retrieval speed-often by large margins-while also cutting training time to a fraction of that required by MRL. Our results establish sparse coding as a powerful paradigm for adaptive representation learning in real-world applications where efficiency and fidelity are both paramount. Code is available at this https URL 

---
# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models 

**Authors**: Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01763)  

**Abstract**: Tool learning aims to augment large language models (LLMs) with diverse tools, enabling them to act as agents for solving practical tasks. Due to the limited context length of tool-using LLMs, adopting information retrieval (IR) models to select useful tools from large toolsets is a critical initial step. However, the performance of IR models in tool retrieval tasks remains underexplored and unclear. Most tool-use benchmarks simplify this step by manually pre-annotating a small set of relevant tools for each task, which is far from the real-world scenarios. In this paper, we propose ToolRet, a heterogeneous tool retrieval benchmark comprising 7.6k diverse retrieval tasks, and a corpus of 43k tools, collected from existing datasets. We benchmark six types of models on ToolRet. Surprisingly, even the models with strong performance in conventional IR benchmarks, exhibit poor performance on ToolRet. This low retrieval quality degrades the task pass rate of tool-use LLMs. As a further step, we contribute a large-scale training dataset with over 200k instances, which substantially optimizes the tool retrieval ability of IR models. 

---
# Zero-Trust Artificial Intelligence Model Security Based on Moving Target Defense and Content Disarm and Reconstruction 

**Authors**: Daniel Gilkarov, Ran Dubin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01758)  

**Abstract**: This paper examines the challenges in distributing AI models through model zoos and file transfer mechanisms. Despite advancements in security measures, vulnerabilities persist, necessitating a multi-layered approach to mitigate risks effectively. The physical security of model files is critical, requiring stringent access controls and attack prevention solutions. This paper proposes a novel solution architecture composed of two prevention approaches. The first is Content Disarm and Reconstruction (CDR), which focuses on disarming serialization attacks that enable attackers to run malicious code as soon as the model is loaded. The second is protecting the model architecture and weights from attacks by using Moving Target Defense (MTD), alerting the model structure, and providing verification steps to detect such attacks. The paper focuses on the highly exploitable Pickle and PyTorch file formats. It demonstrates a 100% disarm rate while validated against known AI model repositories and actual malware attacks from the HuggingFace model zoo. 

---
# Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs 

**Authors**: Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi-ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, Xiren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01743)  

**Abstract**: We introduce Phi-4-Mini and Phi-4-Multimodal, compact yet highly capable language and multimodal models. Phi-4-Mini is a 3.8-billion-parameter language model trained on high-quality web and synthetic data, significantly outperforming recent open-source models of similar size and matching the performance of models twice its size on math and coding tasks requiring complex reasoning. This achievement is driven by a carefully curated synthetic data recipe emphasizing high-quality math and coding datasets. Compared to its predecessor, Phi-3.5-Mini, Phi-4-Mini features an expanded vocabulary size of 200K tokens to better support multilingual applications, as well as group query attention for more efficient long-sequence generation. Phi-4-Multimodal is a multimodal model that integrates text, vision, and speech/audio input modalities into a single model. Its novel modality extension approach leverages LoRA adapters and modality-specific routers to allow multiple inference modes combining various modalities without interference. For example, it now ranks first in the OpenASR leaderboard to date, although the LoRA component of the speech/audio modality has just 460 million parameters. Phi-4-Multimodal supports scenarios involving (vision + language), (vision + speech), and (speech/audio) inputs, outperforming larger vision-language and speech-language models on a wide range of tasks. Additionally, we experiment to further train Phi-4-Mini to enhance its reasoning capabilities. Despite its compact 3.8-billion-parameter size, this experimental version achieves reasoning performance on par with or surpassing significantly larger models, including DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. 

---
# Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning 

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel  

**Link**: [PDF](https://arxiv.org/pdf/2503.01734)  

**Abstract**: Reinforcement learning (RL) offers powerful techniques for solving complex sequential decision-making tasks from experience. In this paper, we demonstrate how RL can be applied to adversarial machine learning (AML) to develop a new class of attacks that learn to generate adversarial examples: inputs designed to fool machine learning models. Unlike traditional AML methods that craft adversarial examples independently, our RL-based approach retains and exploits past attack experience to improve future attacks. We formulate adversarial example generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On CIFAR-10, our agent increases the success rate of adversarial examples by 19.4% and decreases the median number of victim model queries per adversarial example by 53.2% from the start to the end of training. In a head-to-head comparison with a state-of-the-art image attack, SquareAttack, our approach enables an adversary to generate adversarial examples with 13.1% more success after 5000 episodes of training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to attack ML models efficiently and at scale. 

---
# DISCOVER: Data-driven Identification of Sub-activities via Clustering and Visualization for Enhanced Activity Recognition in Smart Homes 

**Authors**: Alexander Karpekov, Sonia Chernova, Thomas Plötz  

**Link**: [PDF](https://arxiv.org/pdf/2503.01733)  

**Abstract**: Human Activity Recognition (HAR) using ambient sensors has great potential for practical applications, particularly in elder care and independent living. However, deploying HAR systems in real-world settings remains challenging due to the high cost of labeled data, the need for pre-segmented sensor streams, and the lack of flexibility in activity granularity. To address these limitations, we introduce DISCOVER, a method designed to discover fine-grained human sub-activities from unlabeled sensor data without relying on pre-segmentation. DISCOVER combines unsupervised feature extraction and clustering with a user-friendly visualization tool to streamline the labeling process. DISCOVER enables domain experts to efficiently annotate only a minimal set of representative cluster centroids, reducing the annotation workload to a small number of samples (0.05% of our dataset). We demonstrate DISCOVER's effectiveness through a re-annotation exercise on widely used HAR datasets, showing that it uncovers finer-grained activities and produces more nuanced annotations than traditional coarse labels. DISCOVER represents a step toward practical, deployable HAR systems that adapt to diverse real environments. 

---
# KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation 

**Authors**: Antoni Bigata, Michał Stypułkowski, Rodrigo Mira, Stella Bounareli, Konstantinos Vougioukas, Zoe Landgraf, Nikita Drobyshev, Maciej Zieba, Stavros Petridis, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2503.01715)  

**Abstract**: Current audio-driven facial animation methods achieve impressive results for short videos but suffer from error accumulation and identity drift when extended to longer durations. Existing methods attempt to mitigate this through external spatial control, increasing long-term consistency but compromising the naturalness of motion. We propose KeyFace, a novel two-stage diffusion-based framework, to address these issues. In the first stage, keyframes are generated at a low frame rate, conditioned on audio input and an identity frame, to capture essential facial expressions and movements over extended periods of time. In the second stage, an interpolation model fills in the gaps between keyframes, ensuring smooth transitions and temporal coherence. To further enhance realism, we incorporate continuous emotion representations and handle a wide range of non-speech vocalizations (NSVs), such as laughter and sighs. We also introduce two new evaluation metrics for assessing lip synchronization and NSV generation. Experimental results show that KeyFace outperforms state-of-the-art methods in generating natural, coherent facial animations over extended durations, successfully encompassing NSVs and continuous emotions. 

---
# Word Form Matters: LLMs' Semantic Reconstruction under Typoglycemia 

**Authors**: Chenxi Wang, Tianle Gu, Zhongyu Wei, Lang Gao, Zirui Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01714)  

**Abstract**: Human readers can efficiently comprehend scrambled words, a phenomenon known as Typoglycemia, primarily by relying on word form; if word form alone is insufficient, they further utilize contextual cues for interpretation. While advanced large language models (LLMs) exhibit similar abilities, the underlying mechanisms remain unclear. To investigate this, we conduct controlled experiments to analyze the roles of word form and contextual information in semantic reconstruction and examine LLM attention patterns. Specifically, we first propose SemRecScore, a reliable metric to quantify the degree of semantic reconstruction, and validate its effectiveness. Using this metric, we study how word form and contextual information influence LLMs' semantic reconstruction ability, identifying word form as the core factor in this process. Furthermore, we analyze how LLMs utilize word form and find that they rely on specialized attention heads to extract and process word form information, with this mechanism remaining stable across varying levels of word scrambling. This distinction between LLMs' fixed attention patterns primarily focused on word form and human readers' adaptive strategy in balancing word form and contextual information provides insights into enhancing LLM performance by incorporating human-like, context-aware mechanisms. 

---
# SAGE: A Framework of Precise Retrieval for RAG 

**Authors**: Jintao Zhang, Guoliang Li, Jinyang Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.01713)  

**Abstract**: Retrieval-augmented generation (RAG) has demonstrated significant proficiency in conducting question-answering (QA) tasks within a specified corpus. Nonetheless, numerous failure instances of RAG in QA still exist. These failures are not solely attributable to the limitations of Large Language Models (LLMs); instead, they predominantly arise from the retrieval of inaccurate information for LLMs due to two limitations: (1) Current RAG methods segment the corpus without considering semantics, making it difficult to find relevant context due to impaired correlation between questions and the segments. (2) There is a trade-off between missing essential context with fewer context retrieved and getting irrelevant context with more context retrieved.
In this paper, we introduce a RAG framework (SAGE), to overcome these limitations. First, to address the segmentation issue without considering semantics, we propose to train a semantic segmentation model. This model is trained to segment the corpus into semantically complete chunks. Second, to ensure that only the most relevant chunks are retrieved while the irrelevant ones are ignored, we design a chunk selection algorithm to dynamically select chunks based on the decreasing speed of the relevance score, leading to a more relevant selection. Third, to further ensure the precision of the retrieved chunks, we propose letting LLMs assess whether retrieved chunks are excessive or lacking and then adjust the amount of context accordingly. Experiments show that SAGE outperforms baselines by 61.25% in the quality of QA on average. Moreover, by avoiding retrieving noisy context, SAGE lowers the cost of the tokens consumed in LLM inference and achieves a 49.41% enhancement in cost efficiency on average. Additionally, our work offers valuable insights for boosting RAG. 

---
# Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens 

**Authors**: Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, Weizhen Bian, Zhen Ye, Sitong Cheng, Ruibin Yuan, Zhixian Zhao, Xinfa Zhu, Jiahao Pan, Liumeng Xue, Pengcheng Zhu, Yunlin Chen, Zhifei Li, Xie Chen, Lei Xie, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.01710)  

**Abstract**: Recent advancements in large language models (LLMs) have driven significant progress in zero-shot text-to-speech (TTS) synthesis. However, existing foundation models rely on multi-stage processing or complex architectures for predicting multiple codebooks, limiting efficiency and integration flexibility. To overcome these challenges, we introduce Spark-TTS, a novel system powered by BiCodec, a single-stream speech codec that decomposes speech into two complementary token types: low-bitrate semantic tokens for linguistic content and fixed-length global tokens for speaker attributes. This disentangled representation, combined with the Qwen2.5 LLM and a chain-of-thought (CoT) generation approach, enables both coarse-grained control (e.g., gender, speaking style) and fine-grained adjustments (e.g., precise pitch values, speaking rate). To facilitate research in controllable TTS, we introduce VoxBox, a meticulously curated 100,000-hour dataset with comprehensive attribute annotations. Extensive experiments demonstrate that Spark-TTS not only achieves state-of-the-art zero-shot voice cloning but also generates highly customizable voices that surpass the limitations of reference-based synthesis. Source code, pre-trained models, and audio samples are available at this https URL. 

---
# Relating Piecewise Linear Kolmogorov Arnold Networks to ReLU Networks 

**Authors**: Nandi Schoots, Mattia Jacopo Villani, Niels uit de Bos  

**Link**: [PDF](https://arxiv.org/pdf/2503.01702)  

**Abstract**: Kolmogorov-Arnold Networks are a new family of neural network architectures which holds promise for overcoming the curse of dimensionality and has interpretability benefits (arXiv:2404.19756). In this paper, we explore the connection between Kolmogorov Arnold Networks (KANs) with piecewise linear (univariate real) functions and ReLU networks. We provide completely explicit constructions to convert a piecewise linear KAN into a ReLU network and vice versa. 

---
# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

**Authors**: Yongchao Chen, Yilun Hao, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01700)  

**Abstract**: Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website this https URL for prompts, videos, and code. 

---
# Perceptual Motor Learning with Active Inference Framework for Robust Lateral Control 

**Authors**: Elahe Delavari, John Moore, Junho Hong, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.01676)  

**Abstract**: This paper presents a novel Perceptual Motor Learning (PML) framework integrated with Active Inference (AIF) to enhance lateral control in Highly Automated Vehicles (HAVs). PML, inspired by human motor learning, emphasizes the seamless integration of perception and action, enabling efficient decision-making in dynamic environments. Traditional autonomous driving approaches--including modular pipelines, imitation learning, and reinforcement learning--struggle with adaptability, generalization, and computational efficiency. In contrast, PML with AIF leverages a generative model to minimize prediction error ("surprise") and actively shape vehicle control based on learned perceptual-motor representations. Our approach unifies deep learning with active inference principles, allowing HAVs to perform lane-keeping maneuvers with minimal data and without extensive retraining across different environments. Extensive experiments in the CARLA simulator demonstrate that PML with AIF enhances adaptability without increasing computational overhead while achieving performance comparable to conventional methods. These findings highlight the potential of PML-driven active inference as a robust alternative for real-world autonomous driving applications. 

---
# Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the Lens of Summarization 

**Authors**: Siya Qi, Rui Cao, Yulan He, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01670)  

**Abstract**: With the rapid development of large language models (LLMs), LLM-as-a-judge has emerged as a widely adopted approach for text quality evaluation, including hallucination evaluation. While previous studies have focused exclusively on single-context evaluation (e.g., discourse faithfulness or world factuality), real-world hallucinations typically involve mixed contexts, which remains inadequately evaluated. In this study, we use summarization as a representative task to comprehensively evaluate LLMs' capability in detecting mixed-context hallucinations, specifically distinguishing between factual and non-factual hallucinations. Through extensive experiments across direct generation and retrieval-based models of varying scales, our main observations are: (1) LLMs' intrinsic knowledge introduces inherent biases in hallucination evaluation; (2) These biases particularly impact the detection of factual hallucinations, yielding a significant performance bottleneck; (3) The fundamental challenge lies in effective knowledge utilization, balancing between LLMs' intrinsic knowledge and external context for accurate mixed-context hallucination evaluation. 

---
# An Efficient Continual Learning Framework for Multivariate Time Series Prediction Tasks with Application to Vehicle State Estimation 

**Authors**: Arvin Hosseinzadeh, Ladan Khoshnevisan, Mohammad Pirani, Shojaeddin Chenouri, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2503.01669)  

**Abstract**: In continual time series analysis using neural networks, catastrophic forgetting (CF) of previously learned models when training on new data domains has always been a significant challenge. This problem is especially challenging in vehicle estimation and control, where new information is sequentially introduced to the model. Unfortunately, existing work on continual learning has not sufficiently addressed the adverse effects of catastrophic forgetting in time series analysis, particularly in multivariate output environments. In this paper, we present EM-ReSeleCT (Efficient Multivariate Representative Selection for Continual Learning in Time Series Tasks), an enhanced approach designed to handle continual learning in multivariate environments. Our approach strategically selects representative subsets from old and historical data and incorporates memory-based continual learning techniques with an improved optimization algorithm to adapt the pre-trained model on new information while preserving previously acquired information. Additionally, we develop a sequence-to-sequence transformer model (autoregressive model) specifically designed for vehicle state estimation. Moreover, we propose an uncertainty quantification framework using conformal prediction to assess the sensitivity of the memory size and to showcase the robustness of the proposed method. Experimental results from tests on an electric Equinox vehicle highlight the superiority of our method in continually learning new information while retaining prior knowledge, outperforming state-of-the-art continual learning methods. Furthermore, EM-ReSeleCT significantly reduces training time, a critical advantage in continual learning applications. 

---
# CoPL: Collaborative Preference Learning for Personalizing LLMs 

**Authors**: Youngbin Choi, Seunghyuk Cho, Minjong Lee, MoonJeong Park, Yesong Ko, Jungseul Ok, Dongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.01658)  

**Abstract**: Personalizing large language models (LLMs) is important for aligning outputs with diverse user preferences, yet existing methods struggle with flexibility and generalization. We propose CoPL (Collaborative Preference Learning), a graph-based collaborative filtering framework that models user-response relationships to enhance preference estimation, particularly in sparse annotation settings. By integrating a mixture of LoRA experts, CoPL efficiently fine-tunes LLMs while dynamically balancing shared and user-specific preferences. Additionally, an optimization-free adaptation strategy enables generalization to unseen users without fine-tuning. Experiments on UltraFeedback-P demonstrate that CoPL outperforms existing personalized reward models, effectively capturing both common and controversial preferences, making it a scalable solution for personalized LLM alignment. 

---
# Enhancing Object Detection Accuracy in Underwater Sonar Images through Deep Learning-based Denoising 

**Authors**: Ziyu Wang, Tao Xue, Yanbin Wang, Jingyuan Li, Haibin Zhang, Zhiqiang Xu, Gaofei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01655)  

**Abstract**: Sonar image object detection is crucial for underwater robotics and other applications. However, various types of noise in sonar images can affect the accuracy of object detection. Denoising, as a critical preprocessing step, aims to remove noise while retaining useful information to improve detection accuracy. Although deep learning-based denoising algorithms perform well on optical images, their application to underwater sonar images remains underexplored. This paper systematically evaluates the effectiveness of several deep learning-based denoising algorithms, originally designed for optical images, in the context of underwater sonar image object detection. We apply nine trained denoising models to images from five open-source sonar datasets, each processing different types of noise. We then test the denoised images using four object detection algorithms. The results show that different denoising models have varying effects on detection performance. By combining the strengths of multiple denoising models, the detection results can be optimized, thus more effectively suppressing noise. Additionally, we adopt a multi-frame denoising technique, using different outputs generated by multiple denoising models as multiple frames of the same scene for further processing to enhance detection accuracy. This method, originally designed for optical images, leverages complementary noise-reduction effects. Experimental results show that denoised sonar images improve the performance of object detection algorithms compared to the original sonar images. 

---
# Distilled Prompt Learning for Incomplete Multimodal Survival Prediction 

**Authors**: Yingxue Xu, Fengtao Zhou, Chenyu Zhao, Yihui Wang, Can Yang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01653)  

**Abstract**: The integration of multimodal data including pathology images and gene profiles is widely applied in precise survival prediction. Despite recent advances in multimodal survival models, collecting complete modalities for multimodal fusion still poses a significant challenge, hindering their application in clinical settings. Current approaches tackling incomplete modalities often fall short, as they typically compensate for only a limited part of the knowledge of missing modalities. To address this issue, we propose a Distilled Prompt Learning framework (DisPro) to utilize the strong robustness of Large Language Models (LLMs) to missing modalities, which employs two-stage prompting for compensation of comprehensive information for missing modalities. In the first stage, Unimodal Prompting (UniPro) distills the knowledge distribution of each modality, preparing for supplementing modality-specific knowledge of the missing modality in the subsequent stage. In the second stage, Multimodal Prompting (MultiPro) leverages available modalities as prompts for LLMs to infer the missing modality, which provides modality-common information. Simultaneously, the unimodal knowledge acquired in the first stage is injected into multimodal inference to compensate for the modality-specific knowledge of the missing modality. Extensive experiments covering various missing scenarios demonstrated the superiority of the proposed method. The code is available at this https URL. 

---
# OpenGS-SLAM: Open-Set Dense Semantic SLAM with 3D Gaussian Splatting for Object-Level Scene Understanding 

**Authors**: Dianyi Yang, Yu Gao, Xihan Wang, Yufeng Yue, Yi Yang, Mengyin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01646)  

**Abstract**: Recent advancements in 3D Gaussian Splatting have significantly improved the efficiency and quality of dense semantic SLAM. However, previous methods are generally constrained by limited-category pre-trained classifiers and implicit semantic representation, which hinder their performance in open-set scenarios and restrict 3D object-level scene understanding. To address these issues, we propose OpenGS-SLAM, an innovative framework that utilizes 3D Gaussian representation to perform dense semantic SLAM in open-set environments. Our system integrates explicit semantic labels derived from 2D foundational models into the 3D Gaussian framework, facilitating robust 3D object-level scene understanding. We introduce Gaussian Voting Splatting to enable fast 2D label map rendering and scene updating. Additionally, we propose a Confidence-based 2D Label Consensus method to ensure consistent labeling across multiple views. Furthermore, we employ a Segmentation Counter Pruning strategy to improve the accuracy of semantic scene representation. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of our method in scene understanding, tracking, and mapping, achieving 10 times faster semantic rendering and 2 times lower storage costs compared to existing methods. Project page: this https URL. 

---
# Machine Learners Should Acknowledge the Legal Implications of Large Language Models as Personal Data 

**Authors**: Henrik Nolte, Michèle Finck, Kristof Meding  

**Link**: [PDF](https://arxiv.org/pdf/2503.01630)  

**Abstract**: Does GPT know you? The answer depends on your level of public recognition; however, if your information was available on a website, the answer is probably yes. All Large Language Models (LLMs) memorize training data to some extent. If an LLM training corpus includes personal data, it also memorizes personal data. Developing an LLM typically involves processing personal data, which falls directly within the scope of data protection laws. If a person is identified or identifiable, the implications are far-reaching: the AI system is subject to EU General Data Protection Regulation requirements even after the training phase is concluded. To back our arguments: (1.) We reiterate that LLMs output training data at inference time, be it verbatim or in generalized form. (2.) We show that some LLMs can thus be considered personal data on their own. This triggers a cascade of data protection implications such as data subject rights, including rights to access, rectification, or erasure. These rights extend to the information embedded with-in the AI model. (3.) This paper argues that machine learning researchers must acknowledge the legal implications of LLMs as personal data throughout the full ML development lifecycle, from data collection and curation to model provision on, e.g., GitHub or Hugging Face. (4.) We propose different ways for the ML research community to deal with these legal implications. Our paper serves as a starting point for improving the alignment between data protection law and the technical capabilities of LLMs. Our findings underscore the need for more interaction between the legal domain and the ML community. 

---
# Advancing vision-language models in front-end development via data synthesis 

**Authors**: Tong Ge, Yashu Liu, Jieping Ye, Tianyi Li, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01619)  

**Abstract**: Modern front-end (FE) development, especially when leveraging the unique features of frameworks like React and Vue, presents distinctive challenges. These include managing modular architectures, ensuring synchronization between data and visual outputs for declarative rendering, and adapting reusable components to various scenarios. Such complexities make it particularly difficult for state-of-the-art large vision-language models (VLMs) to generate accurate and functional code directly from design images. To address these challenges, we propose a reflective agentic workflow that synthesizes high-quality image-text data to capture the diverse characteristics of FE development. This workflow automates the extraction of self-contained\footnote{A \textbf{self-contained} code snippet is one that encapsulates all necessary logic, styling, and dependencies, ensuring it functions independently without requiring external imports or context.} code snippets from real-world projects, renders the corresponding visual outputs, and generates detailed descriptions that link design elements to functional code. To further expand the scope and utility of the synthesis, we introduce three data synthesis strategies: Evolution-based synthesis, which enables scalable and diverse dataset expansion; Waterfall-Model-based synthesis, which generates logically coherent code derived from system requirements; and Additive Development synthesis, which iteratively increases the complexity of human-authored components. We build a large vision-language model, Flame, trained on the synthesized datasets and demonstrate its effectiveness in generating React code via the $\text{pass}@k$ metric. Our results suggest that a code VLM trained to interpret images before code generation may achieve better performance. 

---
# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering 

**Authors**: Zhanghao Hu, Hanqi Yan, Qingling Zhu, Zhenyi Shen, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2503.01606)  

**Abstract**: Large language models have recently pushed open domain question answering (ODQA) to new frontiers. However, prevailing retriever-reader pipelines often depend on multiple rounds of prompt level instructions, leading to high computational overhead, instability, and suboptimal retrieval coverage. In this paper, we propose EmbQA, an embedding-level framework that alleviates these shortcomings by enhancing both the retriever and the reader. Specifically, we refine query representations via lightweight linear layers under an unsupervised contrastive learning objective, thereby reordering retrieved passages to highlight those most likely to contain correct answers. Additionally, we introduce an exploratory embedding that broadens the model's latent semantic space to diversify candidate generation and employs an entropy-based selection mechanism to choose the most confident answer automatically. Extensive experiments across three open-source LLMs, three retrieval methods, and four ODQA benchmarks demonstrate that EmbQA substantially outperforms recent baselines in both accuracy and efficiency. 

---
# Triple-Stream Deep Feature Selection with Metaheuristic Optimization and Machine Learning for Multi-Stage Hypertensive Retinopathy Diagnosis 

**Authors**: Suleyman Burcin Suyun, Mustafa Yurdakul, Sakir Tasdemir, Serkan Bilic  

**Link**: [PDF](https://arxiv.org/pdf/2503.01603)  

**Abstract**: Hypertensive retinopathy (HR) is a severe eye disease that may cause permanent vision loss if not diagnosed early. Traditional diagnostic methods are time-consuming and subjective, highlighting the need for an automated, reliable system. Existing studies often use a single Deep Learning (DL) model, struggling to distinguish HR stages. This study introduces a three-stage approach to enhance HR diagnosis accuracy. Initially, 14 CNN models were tested, identifying DenseNet169, MobileNet, and ResNet152 as the most effective. DenseNet169 achieved 87.73% accuracy, 87.75% precision, 87.73% recall, 87.67% F1-score, and 0.8359 Cohen's Kappa. MobileNet followed with 86.40% accuracy, 86.60% precision, 86.40% recall, 86.31% F1-score, and 0.8180 Cohen's Kappa. ResNet152 ranked third with 85.87% accuracy, 86.01% precision, 85.87% recall, 85.83% F1-score, and 0.8188 Cohen's Kappa. In the second stage, deep features from these models were fused and classified using Machine Learning (ML) algorithms (SVM, RF, XGBoost). SVM (sigmoid kernel) performed best with 92.00% accuracy, 91.93% precision, 92.00% recall, 91.91% F1-score, and 0.8930 Cohen's Kappa. The third stage applied meta-heuristic optimization (GA, ABC, PSO, HHO) for feature selection. HHO yielded 94.66% accuracy, precision, and recall, 94.64% F1-score, and 0.9286 Cohen's Kappa. The proposed approach surpassed single CNN models and previous studies in HR diagnosis accuracy and generalization. 

---
# STAR: Stability-Inducing Weight Perturbation for Continual Learning 

**Authors**: Masih Eskandar, Tooba Imtiaz, Davin Hill, Zifeng Wang, Jennifer Dy  

**Link**: [PDF](https://arxiv.org/pdf/2503.01595)  

**Abstract**: Humans can naturally learn new and varying tasks in a sequential manner. Continual learning is a class of learning algorithms that updates its learned model as it sees new data (on potentially new tasks) in a sequence. A key challenge in continual learning is that as the model is updated to learn new tasks, it becomes susceptible to catastrophic forgetting, where knowledge of previously learned tasks is lost. A popular approach to mitigate forgetting during continual learning is to maintain a small buffer of previously-seen samples and to replay them during training. However, this approach is limited by the small buffer size, and while forgetting is reduced, it is still present. In this paper, we propose a novel loss function, STAR, that exploits the worst-case parameter perturbation that reduces the KL-divergence of model predictions with that of its local parameter neighborhood to promote stability and alleviate forgetting. STAR can be combined with almost any existing rehearsal-based method as a plug-and-play component. We empirically show that STAR consistently improves the performance of existing methods by up to 15% across varying baselines and achieves superior or competitive accuracy to that of state-of-the-art methods aimed at improving rehearsal-based continual learning. 

---
# An Efficient Approach to Detecting Lung Nodules Using Swin Transformer 

**Authors**: Saeed Shakuri, Alireza Rezvanian  

**Link**: [PDF](https://arxiv.org/pdf/2503.01592)  

**Abstract**: Lung cancer has the highest rate of cancer-caused deaths, and early-stage diagnosis could increase the survival rate. Lung nodules are common indicators of lung cancer, making their detection crucial. Various lung nodule detection models exist, but many lack efficiency. Hence, we propose a more efficient approach by leveraging 2D CT slices, reducing computational load and complexity in training and inference. We employ the tiny version of Swin Transformer to benefit from Vision Transformers (ViT) while maintaining low computational cost. A Feature Pyramid Network is added to enhance detection, particularly for small nodules. Additionally, Transfer Learning is used to accelerate training. Our experimental results show that the proposed model outperforms state-of-the-art methods, achieving higher mAP and mAR for small nodules by 1.3% and 1.6%, respectively. Overall, our model achieves the highest mAP of 94.7% and mAR of 94.9%. 

---
# EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection 

**Authors**: Yuhao Zhou, Sirui Song, Boyang Liu, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Zhihao Zhang, Wei Li, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01586)  

**Abstract**: Rotary Position Embedding (RoPE) enables each attention head to capture multi-frequency information along the sequence dimension and is widely applied in foundation models. However, the nonlinearity introduced by RoPE complicates optimization of the key state in the Key-Value (KV) cache for RoPE-based attention. Existing KV cache compression methods typically store key state before rotation and apply the transformation during decoding, introducing additional computational overhead. This paper introduces EliteKV, a flexible modification framework for RoPE-based models supporting variable KV cache compression ratios. EliteKV first identifies the intrinsic frequency preference of each head using RoPElite, selectively restoring linearity to certain dimensions of key within attention computation. Building on this, joint low-rank compression of key and value enables partial cache sharing. Experimental results show that with minimal uptraining on only $0.6\%$ of the original training data, RoPE-based models achieve a $75\%$ reduction in KV cache size while preserving performance within a negligible margin. Furthermore, EliteKV consistently performs well across models of different scales within the same family. 

---
# A Selective Learning Method for Temporal Graph Continual Learning 

**Authors**: Hanmo Liu, Shimin Di, Haoyang Li, Xun Jian, Yue Wang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01580)  

**Abstract**: Node classification is a key task in temporal graph learning (TGL). Real-life temporal graphs often introduce new node classes over time, but existing TGL methods assume a fixed set of classes. This assumption brings limitations, as updating models with full data is costly, while focusing only on new classes results in forgetting old ones. Graph continual learning (GCL) methods mitigate forgetting using old-class subsets but fail to account for their evolution. We define this novel problem as temporal graph continual learning (TGCL), which focuses on efficiently maintaining up-to-date knowledge of old classes. To tackle TGCL, we propose a selective learning framework that substitutes the old-class data with its subsets, Learning Towards the Future (LTF). We derive an upper bound on the error caused by such replacement and transform it into objectives for selecting and learning subsets that minimize classification error while preserving the distribution of the full old-class data. Experiments on three real-world datasets validate the effectiveness of LTF on TGCL. 

---
# MoCFL: Mobile Cluster Federated Learning Framework for Highly Dynamic Network 

**Authors**: Kai Fang, Jiangtao Deng, Chengzu Dong, Usman Naseem, Tongcun Liu, Hailin Feng, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01557)  

**Abstract**: Frequent fluctuations of client nodes in highly dynamic mobile clusters can lead to significant changes in feature space distribution and data drift, posing substantial challenges to the robustness of existing federated learning (FL) strategies. To address these issues, we proposed a mobile cluster federated learning framework (MoCFL). MoCFL enhances feature aggregation by introducing an affinity matrix that quantifies the similarity between local feature extractors from different clients, addressing dynamic data distribution changes caused by frequent client churn and topology changes. Additionally, MoCFL integrates historical and current feature information when training the global classifier, effectively mitigating the catastrophic forgetting problem frequently encountered in mobile scenarios. This synergistic combination ensures that MoCFL maintains high performance and stability in dynamically changing mobile environments. Experimental results on the UNSW-NB15 dataset show that MoCFL excels in dynamic environments, demonstrating superior robustness and accuracy while maintaining reasonable training costs. 

---
# Effective High-order Graph Representation Learning for Credit Card Fraud Detection 

**Authors**: Yao Zou, Dawei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01556)  

**Abstract**: Credit card fraud imposes significant costs on both cardholders and issuing banks. Fraudsters often disguise their crimes, such as using legitimate transactions through several benign users to bypass anti-fraud detection. Existing graph neural network (GNN) models struggle with learning features of camouflaged, indirect multi-hop transactions due to their inherent over-smoothing issues in deep multi-layer aggregation, presenting a major challenge in detecting disguised relationships. Therefore, in this paper, we propose a novel High-order Graph Representation Learning model (HOGRL) to avoid incorporating excessive noise during the multi-layer aggregation process. In particular, HOGRL learns different orders of \emph{pure} representations directly from high-order transaction graphs. We realize this goal by effectively constructing high-order transaction graphs first and then learning the \emph{pure} representations of each order so that the model could identify fraudsters' multi-hop indirect transactions via multi-layer \emph{pure} feature learning. In addition, we introduce a mixture-of-expert attention mechanism to automatically determine the importance of different orders for jointly optimizing fraud detection performance. We conduct extensive experiments in both the open source and real-world datasets, the result demonstrates the significant improvements of our proposed HOGRL compared with state-of-the-art fraud detection baselines. HOGRL's superior performance also proves its effectiveness in addressing high-order fraud camouflage criminals. 

---
# Compositional Reasoning with Transformers, RNNs, and Chain of Thought 

**Authors**: Gilad Yehudai, Noah Amsel, Joan Bruna  

**Link**: [PDF](https://arxiv.org/pdf/2503.01544)  

**Abstract**: We study and compare the expressive power of transformers, RNNs, and transformers with chain of thought tokens on a simple and natural class of problems we term Compositional Reasoning Questions (CRQ). This family captures problems like evaluating Boolean formulas and multi-step word problems. Assuming standard hardness assumptions from circuit complexity and communication complexity, we prove that none of these three architectures is capable of solving CRQs unless some hyperparameter (depth, embedding dimension, and number of chain of thought tokens, respectively) grows with the size of the input. We also provide a construction for each architecture that solves CRQs. For transformers, our construction uses depth that is logarithmic in the problem size. For RNNs, logarithmic embedding dimension is necessary and sufficient, so long as the inputs are provided in a certain order. (Otherwise, a linear dimension is necessary). For transformers with chain of thought, our construction uses $n$ CoT tokens. These results show that, while CRQs are inherently hard, there are several different ways for language models to overcome this hardness. Even for a single class of problems, each architecture has strengths and weaknesses, and none is strictly better than the others. 

---
# Revisiting Large Language Model Pruning using Neuron Semantic Attribution 

**Authors**: Yizhuo Ding, Xinwei Sun, Yanwei Fu, Guosheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01542)  

**Abstract**: Model pruning technique is vital for accelerating large language models by reducing their size and computational requirements. However, the generalizability of existing pruning methods across diverse datasets and tasks remains unclear. Thus, we conduct extensive evaluations on 24 datasets and 4 tasks using popular pruning methods. Based on these evaluations, we find and then investigate that calibration set greatly affect the performance of pruning methods. In addition, we surprisingly find a significant performance drop of existing pruning methods in sentiment classification tasks. To understand the link between performance drop and pruned neurons, we propose Neuron Semantic Attribution, which learns to associate each neuron with specific semantics. This method first makes the unpruned neurons of LLMs explainable. 

---
# Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language 

**Authors**: Xi Chen, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01539)  

**Abstract**: The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, and DeepSeek-v2.5 in identifying implicit toxic language, compared to both direct prompting and Chain-of-Thought. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors. 

---
# Entailment vs. Verification for Partial-assignment Satisfiability and Enumeration 

**Authors**: Roberto Sebastiani  

**Link**: [PDF](https://arxiv.org/pdf/2503.01536)  

**Abstract**: Many procedures for SAT-related problems, in particular for those requiring the complete enumeration of satisfying truth assignments, rely their efficiency and effectiveness on the detection of (possibly small) partial assignments satisfying an input formula. Surprisingly, there seems to be no unique universally-agreed definition of formula satisfaction by a partial assignment in the literature. In this paper we analyze in deep the issue of satisfaction by partial assignments, raising a flag about some ambiguities and subtleties of this concept, and investigating their practical consequences. We identify two alternative notions that are implicitly used in the literature, namely verification and entailment, which coincide if applied to CNF formulas but differ and present complementary properties if applied to non-CNF or to existentially-quantified formulas. We show that, although the former is easier to check and as such is implicitly used by most current search procedures, the latter has better theoretical properties, and can improve the efficiency and effectiveness of enumeration procedures. 

---
# Compare different SG-Schemes based on large least square problems 

**Authors**: Ramkrishna Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2503.01507)  

**Abstract**: This study reviews some of the popular stochastic gradient-based schemes based on large least-square problems. These schemes, often called optimizers in machine learning play a crucial role in finding better parameters of a model. Hence this study focuses on viewing such optimizers with different hyper-parameters and analyzing them based on least square problems. Codes that produced results in this work are available on this https URL. 

---
# Lossy Neural Compression for Geospatial Analytics: A Review 

**Authors**: Carlos Gomes, Isabelle Wittmann, Damien Robert, Johannes Jakubik, Tim Reichelt, Michele Martone, Stefano Maurogiovanni, Rikard Vinge, Jonas Hurst, Erik Scheurer, Rocco Sedona, Thomas Brunschwiler, Stefan Kesselheim, Matej Batic, Philip Stier, Jan Dirk Wegner, Gabriele Cavallaro, Edzer Pebesma, Michael Marszalek, Miguel A Belenguer-Plomer, Kennedy Adriko, Paolo Fraccaro, Romeo Kienzler, Rania Briq, Sabrina Benassou, Michele Lazzarini, Conrad M Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2503.01505)  

**Abstract**: Over the past decades, there has been an explosion in the amount of available Earth Observation (EO) data. The unprecedented coverage of the Earth's surface and atmosphere by satellite imagery has resulted in large volumes of data that must be transmitted to ground stations, stored in data centers, and distributed to end users. Modern Earth System Models (ESMs) face similar challenges, operating at high spatial and temporal resolutions, producing petabytes of data per simulated day.
Data compression has gained relevance over the past decade, with neural compression (NC) emerging from deep learning and information theory, making EO data and ESM outputs ideal candidates due to their abundance of unlabeled data.
In this review, we outline recent developments in NC applied to geospatial data. We introduce the fundamental concepts of NC including seminal works in its traditional applications to image and video compression domains with focus on lossy compression. We discuss the unique characteristics of EO and ESM data, contrasting them with "natural images", and explain the additional challenges and opportunities they present. Moreover, we review current applications of NC across various EO modalities and explore the limited efforts in ESM compression to date.
The advent of self-supervised learning (SSL) and foundation models (FM) has advanced methods to efficiently distill representations from vast unlabeled data. We connect these developments to NC for EO, highlighting the similarities between the two fields and elaborate on the potential of transferring compressed feature representations for machine--to--machine communication.
Based on insights drawn from this review, we devise future directions relevant to applications in EO and ESM. 

---
# Liger: Linearizing Large Language Models to Gated Recurrent Structures 

**Authors**: Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01496)  

**Abstract**: Transformers with linear recurrent modeling offer linear-time training and constant-memory inference. Despite their demonstrated efficiency and performance, pretraining such non-standard architectures from scratch remains costly and risky. The linearization of large language models (LLMs) transforms pretrained standard models into linear recurrent structures, enabling more efficient deployment. However, current linearization methods typically introduce additional feature map modules that require extensive fine-tuning and overlook the gating mechanisms used in state-of-the-art linear recurrent models. To address these issues, this paper presents Liger, short for Linearizing LLMs to gated recurrent structures. Liger is a novel approach for converting pretrained LLMs into gated linear recurrent models without adding extra parameters. It repurposes the pretrained key matrix weights to construct diverse gating mechanisms, facilitating the formation of various gated recurrent structures while avoiding the need to train additional components from scratch. Using lightweight fine-tuning with Low-Rank Adaptation (LoRA), Liger restores the performance of the linearized gated recurrent models to match that of the original LLMs. Additionally, we introduce Liger Attention, an intra-layer hybrid attention mechanism, which significantly recovers 93\% of the Transformer-based LLM at 0.02\% pre-training tokens during the linearization process, achieving competitive results across multiple benchmarks, as validated on models ranging from 1B to 8B parameters. Code is available at this https URL. 

---
# SePer: Measure Retrieval Utility Through The Lens Of Semantic Perplexity Reduction 

**Authors**: Lu Dai, Yijie Xu, Jinhui Ye, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01478)  

**Abstract**: Large Language Models (LLMs) have demonstrated improved generation performance by incorporating externally retrieved knowledge, a process known as retrieval-augmented generation (RAG). Despite the potential of this approach, existing studies evaluate RAG effectiveness by 1) assessing retrieval and generation components jointly, which obscures retrieval's distinct contribution, or 2) examining retrievers using traditional metrics such as NDCG, which creates a gap in understanding retrieval's true utility in the overall generation process. To address the above limitations, in this work, we introduce an automatic evaluation method that measures retrieval quality through the lens of information gain within the RAG framework. Specifically, we propose Semantic Perplexity (SePer), a metric that captures the LLM's internal belief about the correctness of the retrieved information. We quantify the utility of retrieval by the extent to which it reduces semantic perplexity post-retrieval. Extensive experiments demonstrate that SePer not only aligns closely with human preferences but also offers a more precise and efficient evaluation of retrieval utility across diverse RAG scenarios. 

---
# Position: Ensuring mutual privacy is necessary for effective external evaluation of proprietary AI systems 

**Authors**: Ben Bucknall, Robert F. Trager, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2503.01470)  

**Abstract**: The external evaluation of AI systems is increasingly recognised as a crucial approach for understanding their potential risks. However, facilitating external evaluation in practice faces significant challenges in balancing evaluators' need for system access with AI developers' privacy and security concerns. Additionally, evaluators have reason to protect their own privacy - for example, in order to maintain the integrity of held-out test sets. We refer to the challenge of ensuring both developers' and evaluators' privacy as one of providing mutual privacy. In this position paper, we argue that (i) addressing this mutual privacy challenge is essential for effective external evaluation of AI systems, and (ii) current methods for facilitating external evaluation inadequately address this challenge, particularly when it comes to preserving evaluators' privacy. In making these arguments, we formalise the mutual privacy problem; examine the privacy and access requirements of both model owners and evaluators; and explore potential solutions to this challenge, including through the application of cryptographic and hardware-based approaches. 

---
# Rethinking Data: Towards Better Performing Domain-Specific Small Language Models 

**Authors**: Boris Nazarov, Darya Frolova, Yackov Lubarsky, Alexei Gaissinski, Pavel Kisilev  

**Link**: [PDF](https://arxiv.org/pdf/2503.01464)  

**Abstract**: Fine-tuning of Large Language Models (LLMs) for downstream tasks, performed on domain-specific data has shown significant promise. However, commercial use of such LLMs is limited by the high computational cost required for their deployment at scale. On the other hand, small Language Models (LMs) are much more cost effective but have subpar performance in a similar setup. This paper presents our approach to finetuning a small LM, that reaches high accuracy in multiple choice question answering task. We achieve this by improving data quality at each stage of the LM training pipeline. In particular, we start with data structuring resulting in extraction of compact, semantically meaningful text chunks used by a retriever. This allows more efficient knowledge digestion by the LM. Further, we improve the retrieved context by training a lightweight Chunk Re-Ranker (CRR) that generates more accurate relative relevance chunk scores. Finally, we improve the model generalization ability by merging the models fine-tuned with different parameters on different data subsets. We present detailed procedure descriptions, and corresponding experimental findings that show the improvements of each one of the proposed techniques. 

---
# Towards Widening The Distillation Bottleneck for Reasoning Models 

**Authors**: Huifeng Yin, Yu Zhao, Minghao Wu, Xuanfan Ni, Bo Zeng, Hao Wang, Tianqi Shi, Liangying Shao, Chenyang Lyu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01461)  

**Abstract**: Large Reasoning Models(LRMs) such as OpenAI o1 and DeepSeek-R1 have shown remarkable reasoning capabilities by scaling test-time compute and generating long Chain-of-Thought(CoT). Distillation--post-training on LRMs-generated data--is a straightforward yet effective method to enhance the reasoning abilities of smaller models, but faces a critical bottleneck: we found that distilled long CoT data poses learning difficulty for small models and leads to the inheritance of biases (i.e. over-thinking) when using Supervised Fine-tuning(SFT) and Reinforcement Learning(RL) methods. To alleviate this bottleneck, we propose constructing tree-based CoT data from scratch via Monte Carlo Tree Search(MCTS). We then exploit a set of CoT-aware approaches, including Thoughts Length Balance, Fine-grained DPO, and Joint Post-training Objective, to enhance SFT and RL on the construted data. 

---
# Structural Deep Encoding for Table Question Answering 

**Authors**: Raphaël Mouravieff, Benjamin Piwowarski, Sylvain Lamprier  

**Link**: [PDF](https://arxiv.org/pdf/2503.01457)  

**Abstract**: Although Transformers-based architectures excel at processing textual information, their naive adaptation for tabular data often involves flattening the table structure. This simplification can lead to the loss of essential inter-dependencies between rows, columns, and cells, while also posing scalability challenges for large tables. To address these issues, prior works have explored special tokens, structured embeddings, and sparse attention patterns. In this paper, we conduct a comprehensive analysis of tabular encoding techniques, which highlights the crucial role of attention sparsity in preserving structural information of tables. We also introduce a set of novel sparse attention mask designs for tabular data, that not only enhance computational efficiency but also preserve structural integrity, leading to better overall performance. 

---
# AC-Lite : A Lightweight Image Captioning Model for Low-Resource Assamese Language 

**Authors**: Pankaj Choudhury, Yogesh Aggarwal, Prithwijit Guha, Sukumar Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01453)  

**Abstract**: Neural networks have significantly advanced AI applications, yet their real-world adoption remains constrained by high computational demands, hardware limitations, and accessibility challenges. In image captioning, many state-of-the-art models have achieved impressive performances while relying on resource-intensive architectures. This made them impractical for deployment on resource-constrained devices. This limitation is particularly noticeable for applications involving low-resource languages. We demonstrate the case of image captioning in Assamese language, where lack of effective, scalable systems can restrict the accessibility of AI-based solutions for native Assamese speakers. This work presents AC-Lite, a computationally efficient model for image captioning in low-resource Assamese language. AC-Lite reduces computational requirements by replacing computation-heavy visual feature extractors like FasterRCNN with lightweight ShuffleNetv2x1.5. Additionally, Gated Recurrent Units (GRUs) are used as the caption decoder to further reduce computational demands and model parameters. Furthermore, the integration of bilinear attention enhances the model's overall performance. AC-Lite can operate on edge devices, thereby eliminating the need for computation on remote servers. The proposed AC-Lite model achieves 82.3 CIDEr score on the COCO-AC dataset with 1.098 GFLOPs and 25.65M parameters. 

---
# POPGym Arcade: Parallel Pixelated POMDPs 

**Authors**: Zekang Wang, Zhe He, Edan Toledo, Steven Morad  

**Link**: [PDF](https://arxiv.org/pdf/2503.01450)  

**Abstract**: We introduce POPGym Arcade, a benchmark consisting of 7 pixel-based environments each with three difficulties, utilizing a single observation and action space. Each environment offers both fully observable and partially observable variants, enabling counterfactual studies on partial observability. POPGym Arcade utilizes JIT compilation on hardware accelerators to achieve substantial speedups over CPU-bound environments. Moreover, this enables Podracer-style architectures to further increase hardware utilization and training speed. We evaluate memory models on our environments using a Podracer variant of Q learning, and examine the results. Finally, we generate memory saliency maps, uncovering how memories propagate through policies. Our library is available at this https URL popgym_arcade. 

---
# Leveraging LLMs for Mental Health: Detection and Recommendations from Social Discussions 

**Authors**: Vaishali Aggarwal, Sachin Thukral, Krushil Patel, Arnab Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01442)  

**Abstract**: Textual data from social platforms captures various aspects of mental health through discussions around and across issues, while users reach out for help and others sympathize and offer support. We propose a comprehensive framework that leverages Natural Language Processing (NLP) and Generative AI techniques to identify and assess mental health disorders, detect their severity, and create recommendations for behavior change and therapeutic interventions based on users' posts on Reddit.
To classify the disorders, we use rule-based labeling methods as well as advanced pre-trained NLP models to extract nuanced semantic features from the data. We fine-tune domain-adapted and generic pre-trained NLP models based on predictions from specialized Large Language Models (LLMs) to improve classification accuracy. Our hybrid approach combines the generalization capabilities of pre-trained models with the domain-specific insights captured by LLMs, providing an improved understanding of mental health discourse. Our findings highlight the strengths and limitations of each model, offering valuable insights into their practical applicability.
This research potentially facilitates early detection and personalized care to aid practitioners and aims to facilitate timely interventions and improve overall well-being, thereby contributing to the broader field of mental health surveillance and digital health analytics. 

---
# Eau De $Q$-Network: Adaptive Distillation of Neural Networks in Deep Reinforcement Learning 

**Authors**: Théo Vincent, Tim Faust, Yogesh Tripathi, Jan Peters, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01437)  

**Abstract**: Recent works have successfully demonstrated that sparse deep reinforcement learning agents can be competitive against their dense counterparts. This opens up opportunities for reinforcement learning applications in fields where inference time and memory requirements are cost-sensitive or limited by hardware. Until now, dense-to-sparse methods have relied on hand-designed sparsity schedules that are not synchronized with the agent's learning pace. Crucially, the final sparsity level is chosen as a hyperparameter, which requires careful tuning as setting it too high might lead to poor performances. In this work, we address these shortcomings by crafting a dense-to-sparse algorithm that we name Eau De $Q$-Network (EauDeQN). To increase sparsity at the agent's learning pace, we consider multiple online networks with different sparsity levels, where each online network is trained from a shared target network. At each target update, the online network with the smallest loss is chosen as the next target network, while the other networks are replaced by a pruned version of the chosen network. We evaluate the proposed approach on the Atari $2600$ benchmark and the MuJoCo physics simulator, showing that EauDeQN reaches high sparsity levels while keeping performances high. 

---
# Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding 

**Authors**: Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01422)  

**Abstract**: Test-time scaling improves large language model performance by adding extra compute during decoding. Best-of-N (BoN) sampling serves as a common scaling technique, broadening the search space for finding better solutions from the model distribution. However, traditional BoN requires N full generations, leading to high GPU memory overhead and time latency. Moreover, some methods depend on reward models, adding computational cost and limiting domain generalization.
In this paper, we propose Self-Truncation Best-of-N (ST-BoN), a novel decoding method that avoids fully generating all samplings and eliminates the need for reward models. ST-BoN introduces early sampling consistency to estimate the most promising sample, truncating suboptimal ones to free memory and accelerate inference. This pushes the sampling-efficient test-time scaling. Compared to traditional BoN, ST-BoN can reduce dynamic GPU memory overhead by over 90% and time latency by 50%, while achieving comparable or even better performance across reasoning and open-ended domains. 

---
# Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace 

**Authors**: Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia, Dong-Hai Zhu, Xi-He Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01419)  

**Abstract**: Large language model (LLM) is considered a milestone towards achieving Artificial General Intelligence (AGI). With its advanced emergent capabilities, it adapt to a wide range of specific applications. Fine-tuning LLMs for various downstream tasks has become a new paradigm. Low-Rank Adaptation (LoRA) is well-known for its parameter efficiency. It can reduce the number of parameters needed to fine-tune LLMs by several orders of magnitude. However, LoRA-based approaches encounter a significant limitation due to the bottleneck imposed by rank one decomposition. As the parameters count in LLMs increase, even rank one decomposition might surpass the number of parameters truly necessary for handling more downstream tasks. In this paper, we propose a new method for Parameter-Efficient Fine-Tuning (PEFT) via deconvolution in subspace, dubbed as DCFT. We innovatively use deconvolution to complete details and enhance knowledge in subspace incremental matrices, and dynamically control parameters by adjusting the kernel size, unconstrained by rank-one decomposition. Extensive experiments are conducted to validate the effectiveness of DCFT. Results show that compared to LoRA, DCFT achieve an 8$\times$ reduction in parameters, and still achieves highly impressive performance. Our code is available here: this https URL. 

---
# Learning Actionable World Models for Industrial Process Control 

**Authors**: Peng Yan, Ahmed Abdulkadir, Gerrit A. Schatte, Giulia Anguzzi, Joonsu Gha, Nikola Pascher, Matthias Rosenthal, Yunlong Gao, Benjamin F. Grewe, Thilo Stadelmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.01411)  

**Abstract**: To go from (passive) process monitoring to active process control, an effective AI system must learn about the behavior of the complex system from very limited training data, forming an ad-hoc digital twin with respect to process in- and outputs that captures the consequences of actions on the process's world. We propose a novel methodology based on learning world models that disentangles process parameters in the learned latent representation, allowing for fine-grained control. Representation learning is driven by the latent factors that influence the processes through contrastive learning within a joint embedding predictive architecture. This makes changes in representations predictable from changes in inputs and vice versa, facilitating interpretability of key factors responsible for process variations, paving the way for effective control actions to keep the process within operational bounds. The effectiveness of our method is validated on the example of plastic injection molding, demonstrating practical relevance in proposing specific control actions for a notoriously unstable process. 

---
# Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification 

**Authors**: Gaozheng Pei, Shaojie Lyu, Gong Chen, Ke Ma, Qianqian Xu, Yingfei Sun, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01407)  

**Abstract**: Existing diffusion-based purification methods aim to disrupt adversarial perturbations by introducing a certain amount of noise through a forward diffusion process, followed by a reverse process to recover clean examples. However, this approach is fundamentally flawed: the uniform operation of the forward process across all pixels compromises normal pixels while attempting to combat adversarial perturbations, resulting in the target model producing incorrect predictions. Simply relying on low-intensity noise is insufficient for effective defense. To address this critical issue, we implement a heterogeneous purification strategy grounded in the interpretability of neural networks. Our method decisively applies higher-intensity noise to specific pixels that the target model focuses on while the remaining pixels are subjected to only low-intensity noise. This requirement motivates us to redesign the sampling process of the diffusion model, allowing for the effective removal of varying noise levels. Furthermore, to evaluate our method against strong adaptative attack, our proposed method sharply reduces time cost and memory usage through a single-step resampling. The empirical evidence from extensive experiments across three datasets demonstrates that our method outperforms most current adversarial training and purification techniques by a substantial margin. 

---
# Enhancing Social Media Rumor Detection: A Semantic and Graph Neural Network Approach for the 2024 Global Election 

**Authors**: Liu Yan, Liu Yunpeng, Zhao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01394)  

**Abstract**: The development of social media platforms has revolutionized the speed and manner in which information is disseminated, leading to both beneficial and detrimental effects on society. While these platforms facilitate rapid communication, they also accelerate the spread of rumors and extremist speech, impacting public perception and behavior significantly. This issue is particularly pronounced during election periods, where the influence of social media on election outcomes has become a matter of global concern. With the unprecedented number of elections in 2024, against this backdrop, the election ecosystem has encountered unprecedented challenges. This study addresses the urgent need for effective rumor detection on social media by proposing a novel method that combines semantic analysis with graph neural networks. We have meticulously collected a dataset from PolitiFact and Twitter, focusing on politically relevant rumors. Our approach involves semantic analysis using a fine-tuned BERT model to vectorize text content and construct a directed graph where tweets and comments are nodes, and interactions are edges. The core of our method is a graph neural network, SAGEWithEdgeAttention, which extends the GraphSAGE model by incorporating first-order differences as edge attributes and applying an attention mechanism to enhance feature aggregation. This innovative approach allows for the fine-grained analysis of the complex social network structure, improving rumor detection accuracy. The study concludes that our method significantly outperforms traditional content analysis and time-based models, offering a theoretically sound and practically efficient solution. 

---
# Geo-Semantic-Parsing: AI-powered geoparsing by traversing semantic knowledge graphs 

**Authors**: Leonardo Nizzoli, Marco Avvenuti, Maurizio Tesconi, Stefano Cresci  

**Link**: [PDF](https://arxiv.org/pdf/2503.01386)  

**Abstract**: Online social networks convey rich information about geospatial facets of reality. However in most cases, geographic information is not explicit and structured, thus preventing its exploitation in real-time applications. We address this limitation by introducing a novel geoparsing and geotagging technique called Geo-Semantic-Parsing (GSP). GSP identifies location references in free text and extracts the corresponding geographic coordinates. To reach this goal, we employ a semantic annotator to identify relevant portions of the input text and to link them to the corresponding entity in a knowledge graph. Then, we devise and experiment with several efficient strategies for traversing the knowledge graph, thus expanding the available set of information for the geoparsing task. Finally, we exploit all available information for learning a regression model that selects the best entity with which to geotag the input text. We evaluate GSP on a well-known reference dataset including almost 10k event-related tweets, achieving $F1=0.66$. We extensively compare our results with those of 2 baselines and 3 state-of-the-art geoparsing techniques, achieving the best performance. On the same dataset, competitors obtain $F1 \leq 0.55$. We conclude by providing in-depth analyses of our results, showing that the overall superior performance of GSP is mainly due to a large improvement in recall, with respect to existing techniques. 

---
# Combining Flow Matching and Transformers for Efficient Solution of Bayesian Inverse Problems 

**Authors**: Daniil Sherki, Ivan Oseledets, Ekaterina Muravleva  

**Link**: [PDF](https://arxiv.org/pdf/2503.01375)  

**Abstract**: Solving Bayesian inverse problems efficiently remains a significant challenge due to the complexity of posterior distributions and the computational cost of traditional sampling methods. Given a series of observations and the forward model, we want to recover the distribution of the parameters, conditioned on observed experimental data. We show, that combining Conditional Flow Mathching (CFM) with transformer-based architecture, we can efficiently sample from such kind of distribution, conditioned on variable number of observations. 

---
# SwiLTra-Bench: The Swiss Legal Translation Benchmark 

**Authors**: Joel Niklaus, Jakob Merane, Luka Nenadic, Sina Ahmadi, Yingqiang Gao, Cyrill A. H. Chevalley, Claude Humbel, Christophe Gösken, Lorenzo Tanzi, Thomas Lüthi, Stefan Palombo, Spencer Poff, Boling Yang, Nan Wu, Matthew Guillod, Robin Mamié, Daniel Brunner, Julio Pereyra, Niko Grupen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01372)  

**Abstract**: In Switzerland legal translation is uniquely important due to the country's four official languages and requirements for multilingual legal documentation. However, this process traditionally relies on professionals who must be both legal experts and skilled translators -- creating bottlenecks and impacting effective access to justice. To address this challenge, we introduce SwiLTra-Bench, a comprehensive multilingual benchmark of over 180K aligned Swiss legal translation pairs comprising laws, headnotes, and press releases across all Swiss languages along with English, designed to evaluate LLM-based translation systems. Our systematic evaluation reveals that frontier models achieve superior translation performance across all document types, while specialized translation systems excel specifically in laws but under-perform in headnotes. Through rigorous testing and human expert validation, we demonstrate that while fine-tuning open SLMs significantly improves their translation quality, they still lag behind the best zero-shot prompted frontier models such as Claude-3.5-Sonnet. Additionally, we present SwiLTra-Judge, a specialized LLM evaluation system that aligns best with human expert assessments. 

---
# Dendron: Enhancing Human Activity Recognition with On-Device TinyML Learning 

**Authors**: Hazem Hesham Yousef Shalby, Manuel Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2503.01353)  

**Abstract**: Human activity recognition (HAR) is a research field that employs Machine Learning (ML) techniques to identify user activities. Recent studies have prioritized the development of HAR solutions directly executed on wearable devices, enabling the on-device activity recognition. This approach is supported by the Tiny Machine Learning (TinyML) paradigm, which integrates ML within embedded devices with limited resources. However, existing approaches in the field lack in the capability for on-device learning of new HAR tasks, particularly when supervised data are scarce. To address this limitation, our paper introduces Dendron, a novel TinyML methodology designed to facilitate the on-device learning of new tasks for HAR, even in conditions of limited supervised data. Experimental results on two public-available datasets and an off-the-shelf device (STM32-NUCLEO-F401RE) show the effectiveness and efficiency of the proposed solution. 

---
# Same Question, Different Words: A Latent Adversarial Framework for Prompt Robustness 

**Authors**: Tingchen Fu, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2503.01345)  

**Abstract**: Insensitivity to semantically-preserving variations of prompts (paraphrases) is crucial for reliable behavior and real-world deployment of large language models. However, language models exhibit significant performance degradation when faced with semantically equivalent but differently phrased prompts, and existing solutions either depend on trial-and-error prompt engineering or require computationally expensive inference-time algorithms. In this study, built on the key insight that worst-case prompts exhibit a drift in embedding space, we present Latent Adversarial Paraphrasing (LAP), a dual-loop adversarial framework: the inner loop trains a learnable perturbation to serve as a "latent continuous paraphrase" while preserving semantics through Lagrangian regulation, and the outer loop optimizes the language model parameters on these perturbations. We conduct extensive experiments to demonstrate the effectiveness of LAP across multiple LLM architectures on the RobustAlpaca benchmark with a 0.5%-4% absolution improvement on worst-case win-rate compared with vanilla supervised fine-tuning. 

---
# Answer, Refuse, or Guess? Investigating Risk-Aware Decision Making in Language Models 

**Authors**: Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01332)  

**Abstract**: Knowing when to answer or refuse is crucial for safe and reliable decision-making language agents. Although prior work has introduced refusal strategies to boost LMs' reliability, how these models adapt their decisions to different risk levels remains underexplored. We formalize the task of risk-aware decision-making, expose critical weaknesses in existing LMs, and propose skill-decomposition solutions to mitigate them. Our findings show that even cutting-edge LMs--both regular and reasoning models--still require explicit prompt chaining to handle the task effectively, revealing the challenges that must be overcome to achieve truly autonomous decision-making agents. 

---
# Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning 

**Authors**: Anh Tong, Thanh Nguyen-Tang, Dongeun Lee, Duc Nguyen, Toan Tran, David Hall, Cheongwoong Kang, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01329)  

**Abstract**: Recent advancements in large language models (LLMs) based on transformer architectures have sparked significant interest in understanding their inner workings. In this paper, we introduce a novel approach to modeling transformer architectures using highly flexible non-autonomous neural ordinary differential equations (ODEs). Our proposed model parameterizes all weights of attention and feed-forward blocks through neural networks, expressing these weights as functions of a continuous layer index. Through spectral analysis of the model's dynamics, we uncover an increase in eigenvalue magnitude that challenges the weight-sharing assumption prevalent in existing theoretical studies. We also leverage the Lyapunov exponent to examine token-level sensitivity, enhancing model interpretability. Our neural ODE transformer demonstrates performance comparable to or better than vanilla transformers across various configurations and datasets, while offering flexible fine-tuning capabilities that can adapt to different architectural constraints. 

---
# PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization 

**Authors**: Xinyi Wan, Penghui Qi, Guangxing Huang, Jialin Li, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01328)  

**Abstract**: Pipeline parallelism (PP) is widely used for training large language models (LLMs), yet its scalability is often constrained by high activation memory consumption as the number of in-flight microbatches grows with the degree of PP. In this paper, we focus on addressing this challenge by leveraging the under-explored memory offload strategy in PP. With empirical study, we discover that in the majority of standard configurations, at least half, and potentially all, of the activations can be offloaded with negligible overhead. In the cases where full overload is not possible, we introduce a novel selective offload strategy that decreases peak activation memory in a better-than-linear manner. Furthermore, we integrate memory offload with other techniques to jointly consider overall throughput and memory limitation. Our experiments proves that the per-device activation memory effectively reduces with the total number of stages, making PP a stronger alternative than TP, offering up to a 19\% acceleration with even lower memory consumption. The implementation is open-sourced at \href{this https URL}{this url}. 

---
# CacheQuant: Comprehensively Accelerated Diffusion Models 

**Authors**: Xuewen Liu, Zhikai Li, Qingyi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01323)  

**Abstract**: Diffusion models have gradually gained prominence in the field of image synthesis, showcasing remarkable generative capabilities. Nevertheless, the slow inference and complex networks, resulting from redundancy at both temporal and structural levels, hinder their low-latency applications in real-world scenarios. Current acceleration methods for diffusion models focus separately on temporal and structural levels. However, independent optimization at each level to further push the acceleration limits results in significant performance degradation. On the other hand, integrating optimizations at both levels can compound the acceleration effects. Unfortunately, we find that the optimizations at these two levels are not entirely orthogonal. Performing separate optimizations and then simply integrating them results in unsatisfactory performance. To tackle this issue, we propose CacheQuant, a novel training-free paradigm that comprehensively accelerates diffusion models by jointly optimizing model caching and quantization techniques. Specifically, we employ a dynamic programming approach to determine the optimal cache schedule, in which the properties of caching and quantization are carefully considered to minimize errors. Additionally, we propose decoupled error correction to further mitigate the coupled and accumulated errors step by step. Experimental results show that CacheQuant achieves a 5.18 speedup and 4 compression for Stable Diffusion on MS-COCO, with only a 0.02 loss in CLIP score. Our code are open-sourced: this https URL . 

---
# Scaling Law Phenomena Across Regression Paradigms: Multiple and Kernel Approaches 

**Authors**: Yifang Chen, Xuyang Guo, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.01314)  

**Abstract**: Recently, Large Language Models (LLMs) have achieved remarkable success. A key factor behind this success is the scaling law observed by OpenAI. Specifically, for models with Transformer architecture, the test loss exhibits a power-law relationship with model size, dataset size, and the amount of computation used in training, demonstrating trends that span more than seven orders of magnitude. This scaling law challenges traditional machine learning wisdom, notably the Oscar Scissors principle, which suggests that an overparametrized algorithm will overfit the training datasets, resulting in poor test performance. Recent research has also identified the scaling law in simpler machine learning contexts, such as linear regression. However, fully explaining the scaling law in large practical models remains an elusive goal. In this work, we advance our understanding by demonstrating that the scaling law phenomenon extends to multiple regression and kernel regression settings, which are significantly more expressive and powerful than linear methods. Our analysis provides deeper insights into the scaling law, potentially enhancing our understanding of LLMs. 

---
# From Claims to Evidence: A Unified Framework and Critical Analysis of CNN vs. Transformer vs. Mamba in Medical Image Segmentation 

**Authors**: Pooya Mohammadi Kazaj, Giovanni Baj, Yazdan Salimi, Anselm W. Stark, Waldo Valenzuela, George CM. Siontis, Habib Zaidi, Mauricio Reyes, Christoph Graeni, Isaac Shiri  

**Link**: [PDF](https://arxiv.org/pdf/2503.01306)  

**Abstract**: While numerous architectures for medical image segmentation have been proposed, achieving competitive performance with state-of-the-art models networks such as nnUNet, still leave room for further innovation. In this work, we introduce nnUZoo, an open source benchmarking framework built upon nnUNet, which incorporates various deep learning architectures, including CNNs, Transformers, and Mamba-based models. Using this framework, we provide a fair comparison to demystify performance claims across different medical image segmentation tasks. Additionally, in an effort to enrich the benchmarking, we explored five new architectures based on Mamba and Transformers, collectively named X2Net, and integrated them into nnUZoo for further evaluation. The proposed models combine the features of conventional U2Net, nnUNet, CNN, Transformer, and Mamba layers and architectures, called X2Net (UNETR2Net (UNETR), SwT2Net (SwinTransformer), SS2D2Net (SwinUMamba), Alt1DM2Net (LightUMamba), and MambaND2Net (MambaND)). We extensively evaluate the performance of different models on six diverse medical image segmentation datasets, including microscopy, ultrasound, CT, MRI, and PET, covering various body parts, organs, and labels. We compare their performance, in terms of dice score and computational efficiency, against their baseline models, U2Net, and nnUNet. CNN models like nnUNet and U2Net demonstrated both speed and accuracy, making them effective choices for medical image segmentation tasks. Transformer-based models, while promising for certain imaging modalities, exhibited high computational costs. Proposed Mamba-based X2Net architecture (SS2D2Net) achieved competitive accuracy with no significantly difference from nnUNet and U2Net, while using fewer parameters. However, they required significantly longer training time, highlighting a trade-off between model efficiency and computational cost. 

---
# MINT: Multi-modal Chain of Thought in Unified Generative Models for Enhanced Image Generation 

**Authors**: Yi Wang, Mushui Liu, Wanggui He, Longxiang Zhang, Ziwei Huang, Guanghao Zhang, Fangxun Shu, Zhong Tao, Dong She, Zhelun Yu, Haoyuan Li, Weilong Dai, Mingli Song, Jie Song, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01298)  

**Abstract**: Unified generative models have demonstrated extraordinary performance in both text and image generation. However, they tend to underperform when generating intricate images with various interwoven conditions, which is hard to solely rely on straightforward text-to-image generation. In response to this challenge, we introduce MINT, an innovative unified generative model, empowered with native multimodal chain of thought (MCoT) for enhanced image generation for the first time. Firstly, we design Mixture of Transformer Experts (MTXpert), an expert-parallel structure that effectively supports both natural language generation (NLG) and visual capabilities, while avoiding potential modality conflicts that could hinder the full potential of each modality. Building on this, we propose an innovative MCoT training paradigm, a step-by-step approach to multimodal thinking, reasoning, and reflection specifically designed to enhance image generation. This paradigm equips MINT with nuanced, element-wise decoupled alignment and a comprehensive understanding of textual and visual components. Furthermore, it fosters advanced multimodal reasoning and self-reflection, enabling the construction of images that are firmly grounded in the logical relationships between these elements. Notably, MINT has been validated to exhibit superior performance across multiple benchmarks for text-to-image (T2I) and image-to-text (I2T) tasks. 

---
# Fine-Grained Controllable Apparel Showcase Image Generation via Garment-Centric Outpainting 

**Authors**: Rong Zhang, Jingnan Wang, Zhiwen Zuo, Jianfeng Dong, Wei Li, Chi Wang, Weiwei Xu, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01294)  

**Abstract**: In this paper, we propose a novel garment-centric outpainting (GCO) framework based on the latent diffusion model (LDM) for fine-grained controllable apparel showcase image generation. The proposed framework aims at customizing a fashion model wearing a given garment via text prompts and facial images. Different from existing methods, our framework takes a garment image segmented from a dressed mannequin or a person as the input, eliminating the need for learning cloth deformation and ensuring faithful preservation of garment details. The proposed framework consists of two stages. In the first stage, we introduce a garment-adaptive pose prediction model that generates diverse poses given the garment. Then, in the next stage, we generate apparel showcase images, conditioned on the garment and the predicted poses, along with specified text prompts and facial images. Notably, a multi-scale appearance customization module (MS-ACM) is designed to allow both overall and fine-grained text-based control over the generated model's appearance. Moreover, we leverage a lightweight feature fusion operation without introducing any extra encoders or modules to integrate multiple conditions, which is more efficient. Extensive experiments validate the superior performance of our framework compared to state-of-the-art methods. 

---
# ACTIVA: Amortized Causal Effect Estimation without Graphs via Transformer-based Variational Autoencoder 

**Authors**: Andreas Sauter, Saber Salehkaleybar, Aske Plaat, Erman Acar  

**Link**: [PDF](https://arxiv.org/pdf/2503.01290)  

**Abstract**: Predicting the distribution of outcomes under hypothetical interventions is crucial in domains like healthcare, economics, and policy-making. Current methods often rely on strong assumptions, such as known causal graphs or parametric models, and lack amortization across problem instances, limiting their practicality. We propose a novel transformer-based conditional variational autoencoder architecture, named ACTIVA, that extends causal transformer encoders to predict causal effects as mixtures of Gaussians. Our method requires no causal graph and predicts interventional distributions given only observational data and a queried intervention. By amortizing over many simulated instances, it enables zero-shot generalization to novel datasets without retraining. Experiments demonstrate accurate predictions for synthetic and semi-synthetic data, showcasing the effectiveness of our graph-free, amortized causal inference approach. 

---
# Robust Simulation-Based Inference under Missing Data via Neural Processes 

**Authors**: Yogesh Verma, Ayush Bharti, Vikas Garg  

**Link**: [PDF](https://arxiv.org/pdf/2503.01287)  

**Abstract**: Simulation-based inference (SBI) methods typically require fully observed data to infer parameters of models with intractable likelihood functions. However, datasets often contain missing values due to incomplete observations, data corruptions (common in astrophysics), or instrument limitations (e.g., in high-energy physics applications). In such scenarios, missing data must be imputed before applying any SBI method. We formalize the problem of missing data in SBI and demonstrate that naive imputation methods can introduce bias in the estimation of SBI posterior. We also introduce a novel amortized method that addresses this issue by jointly learning the imputation model and the inference network within a neural posterior estimation (NPE) framework. Extensive empirical results on SBI benchmarks show that our approach provides robust inference outcomes compared to standard baselines for varying levels of missing data. Moreover, we demonstrate the merits of our imputation model on two real-world bioactivity datasets (Adrenergic and Kinase assays). Code is available at this https URL. 

---
# Multi-Level Collaboration in Model Merging 

**Authors**: Qi Li, Runpeng Yu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01268)  

**Abstract**: Parameter-level model merging is an emerging paradigm in multi-task learning with significant promise. Previous research has explored its connections with prediction-level model ensembling-commonly viewed as the upper bound for merging-to reveal the potential of achieving performance consistency between the two. However, this observation relies on certain preconditions, such as being limited to two models, using ViT-based models, and all models are fine-tuned from the same pre-trained checkpoint. To further understand the intrinsic connections between model merging and model ensembling, this paper explores an interesting possibility: If these restrictions are removed, can performance consistency still be achieved between merging and ensembling? To answer this question, we first theoretically establish a performance correlation between merging and ensembling. We find that even when previous restrictions are not met, there is still a way for model merging to attain a near-identical and superior performance similar to that of ensembling. To verify whether our findings are practical, we introduce a validation framework termed Neural Ligand (NeuLig). The learning process of NeuLig is meticulously designed with a specialized loss function supported by theoretical foundations. Experimental results demonstrate the robust resilience of NeuLig in terms of both model scale and the number of collaborating models. For instance, for the case involving 5 CLIP-ViT-B/32 models, parameter-level merging achieves the same performance as prediction-level ensembling (merging: 95.44% vs. ensembling: 95.46%). 

---
# Voice Cloning for Dysarthric Speech Synthesis: Addressing Data Scarcity in Speech-Language Pathology 

**Authors**: Birger Moell, Fredrik Sand Aronsson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01266)  

**Abstract**: This study explores voice cloning to generate synthetic speech replicating the unique patterns of individuals with dysarthria. Using the TORGO dataset, we address data scarcity and privacy challenges in speech-language pathology. Our contributions include demonstrating that voice cloning preserves dysarthric speech characteristics, analyzing differences between real and synthetic data, and discussing implications for diagnostics, rehabilitation, and communication. We cloned voices from dysarthric and control speakers using a commercial platform, ensuring gender-matched synthetic voices. A licensed speech-language pathologist (SLP) evaluated a subset for dysarthria, speaker gender, and synthetic indicators. The SLP correctly identified dysarthria in all cases and speaker gender in 95% but misclassified 30% of synthetic samples as real, indicating high realism. Our results suggest synthetic speech effectively captures disordered characteristics and that voice cloning has advanced to produce high-quality data resembling real speech, even to trained professionals. This has critical implications for healthcare, where synthetic data can mitigate data scarcity, protect privacy, and enhance AI-driven diagnostics. By enabling the creation of diverse, high-quality speech datasets, voice cloning can improve generalizable models, personalize therapy, and advance assistive technologies for dysarthria.
We publicly release our synthetic dataset to foster further research and collaboration, aiming to develop robust models that improve patient outcomes in speech-language pathology. 

---
# A Taxonomy for Evaluating Generalist Robot Policies 

**Authors**: Jensen Gao, Suneel Belkhale, Sudeep Dasari, Ashwin Balakrishna, Dhruv Shah, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01238)  

**Abstract**: Machine learning for robotics promises to unlock generalization to novel tasks and environments. Guided by this promise, many recent works have focused on scaling up robot data collection and developing larger, more expressive policies to achieve this. But how do we measure progress towards this goal of policy generalization in practice? Evaluating and quantifying generalization is the Wild West of modern robotics, with each work proposing and measuring different types of generalization in their own, often difficult to reproduce, settings. In this work, our goal is (1) to outline the forms of generalization we believe are important in robot manipulation in a comprehensive and fine-grained manner, and (2) to provide reproducible guidelines for measuring these notions of generalization. We first propose STAR-Gen, a taxonomy of generalization for robot manipulation structured around visual, semantic, and behavioral generalization. We discuss how our taxonomy encompasses most prior notions of generalization in robotics. Next, we instantiate STAR-Gen with a concrete real-world benchmark based on the widely-used Bridge V2 dataset. We evaluate a variety of state-of-the-art models on this benchmark to demonstrate the utility of our taxonomy in practice. Our taxonomy of generalization can yield many interesting insights into existing models: for example, we observe that current vision-language-action models struggle with various types of semantic generalization, despite the promise of pre-training on internet-scale language datasets. We believe STAR-Gen and our guidelines can improve the dissemination and evaluation of progress towards generalization in robotics, which we hope will guide model design and future data collection efforts. We provide videos and demos at our website this http URL. 

---
# LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains 

**Authors**: Ling Xiao, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01236)  

**Abstract**: Multi-terrain cost-efficient path planning is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes travel costs. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments, where recharging or refueling is difficult. However, there is very limited research on this topic. In this paper, we develop a prompt-based approach, LLM-Advisor, which leverages large language models (LLMs) as effective advisors for path planning. The LLM-Advisor selectively provides suggestions, demonstrating its ability to recognize when no modifications are necessary. When suggestions are made, 70.59% of the paths suggested for the A* algorithm, 69.47% for the RRT* algorithm, and 78.70% for the LLM-A* algorithm achieve greater cost efficiency. Since LLM-Advisor may occasionally lack common sense in their suggestions, we propose two hallucination-mitigation strategies. Furthermore, we experimentally verified that GPT-4o performs poorly in zero-shot path planning, even when terrain descriptions are clearly provided, demonstrating its low spatial awareness. We also experimentally demonstrate that using an LLM as an advisor is more effective than directly integrating it into the path-planning loop. Since LLMs may generate hallucinations, using LLMs in the loop of a search-based method (such as A*) may lead to a higher number of failed paths, demonstrating that our proposed LLM-Advisor is a better choice. 

---
# Learning Covariance-Based Multi-Scale Representation of Neuroimaging Measures for Alzheimer Classification 

**Authors**: Seunghun Baek, Injun Choi, Mustafa Dere, Minjeong Kim, Guorong Wu, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.01232)  

**Abstract**: Stacking excessive layers in DNN results in highly underdetermined system when training samples are limited, which is very common in medical applications. In this regard, we present a framework capable of deriving an efficient high-dimensional space with reasonable increase in model size. This is done by utilizing a transform (i.e., convolution) that leverages scale-space theory with covariance structure. The overall model trains on this transform together with a downstream classifier (i.e., Fully Connected layer) to capture the optimal multi-scale representation of the original data which corresponds to task-specific components in a dual space. Experiments on neuroimaging measures from Alzheimer's Disease Neuroimaging Initiative (ADNI) study show that our model performs better and converges faster than conventional models even when the model size is significantly reduced. The trained model is made interpretable using gradient information over the multi-scale transform to delineate personalized AD-specific regions in the brain. 

---
# Tera-MIND: Tera-scale mouse brain simulation via spatial mRNA-guided diffusion 

**Authors**: Jiqing Wu, Ingrid Berg, Yawei Li, Ender Konukoglu, Viktor H. Koelzer  

**Link**: [PDF](https://arxiv.org/pdf/2503.01220)  

**Abstract**: Holistic 3D modeling of molecularly defined brain structures is crucial for understanding complex brain functions. Emerging tissue profiling technologies enable the construction of a comprehensive atlas of the mammalian brain with sub-cellular resolution and spatially resolved gene expression data. However, such tera-scale volumetric datasets present significant computational challenges in understanding complex brain functions within their native 3D spatial context. Here, we propose the novel generative approach $\textbf{Tera-MIND}$, which can simulate $\textbf{Tera}$-scale $\textbf{M}$ouse bra$\textbf{IN}s$ in 3D using a patch-based and boundary-aware $\textbf{D}$iffusion model. Taking spatial transcriptomic data as the conditional input, we generate virtual mouse brains with comprehensive cellular morphological detail at teravoxel scale. Through the lens of 3D $gene$-$gene$ self-attention, we identify spatial molecular interactions for key transcriptomic pathways in the murine brain, exemplified by glutamatergic and dopaminergic neuronal systems. Importantly, these $in$-$silico$ biological findings are consistent and reproducible across three tera-scale virtual mouse brains. Therefore, Tera-MIND showcases a promising path toward efficient and generative simulations of whole organ systems for biomedical research. Project website: $\href{this http URL}{https}$ 

---
# STGAN: Spatial-temporal Graph Autoregression Network for Pavement Distress Deterioration Prediction 

**Authors**: Shilin Tong, Difei Wu, Xiaona Liu, Le Zheng, Yuchuan Du, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01152)  

**Abstract**: Pavement distress significantly compromises road integrity and poses risks to drivers. Accurate prediction of pavement distress deterioration is essential for effective road management, cost reduction in maintenance, and improvement of traffic safety. However, real-world data on pavement distress is usually collected irregularly, resulting in uneven, asynchronous, and sparse spatial-temporal datasets. This hinders the application of existing spatial-temporal models, such as DCRNN, since they are only applicable to regularly and synchronously collected data. To overcome these challenges, we propose the Spatial-Temporal Graph Autoregression Network (STGAN), a novel graph neural network model designed for accurately predicting irregular pavement distress deterioration using complex spatial-temporal data. Specifically, STGAN integrates the temporal domain into the spatial domain, creating a larger graph where nodes are represented by spatial-temporal tuples and edges are formed based on a similarity-based connection mechanism. Furthermore, based on the constructed spatiotemporal graph, we formulate pavement distress deterioration prediction as a graph autoregression task, i.e., the graph size increases incrementally and the prediction is performed sequentially. This is accomplished by a novel spatial-temporal attention mechanism deployed by STGAN. Utilizing the ConTrack dataset, which contains pavement distress records collected from different locations in Shanghai, we demonstrate the superior performance of STGAN in capturing spatial-temporal correlations and addressing the aforementioned challenges. Experimental results further show that STGAN outperforms baseline models, and ablation studies confirm the effectiveness of its novel modules. Our findings contribute to promoting proactive road maintenance decision-making and ultimately enhancing road safety and resilience. 

---
# ReaderLM-v2: Small Language Model for HTML to Markdown and JSON 

**Authors**: Feng Wang, Zesheng Shi, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01151)  

**Abstract**: We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The model's effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20\% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements. 

---
# Dynamic spillovers and investment strategies across artificial intelligence ETFs, artificial intelligence tokens, and green markets 

**Authors**: Ying-Hui Shao, Yan-Hong Yang, Wei-Xing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01148)  

**Abstract**: This paper investigates the risk spillovers among AI ETFs, AI tokens, and green markets using the R2 decomposition method. We reveal several key insights. First, the overall transmission connectedness index (TCI) closely aligns with the contemporaneous TCI, while the lagged TCI is significantly lower. Second, AI ETFs and clean energy act as risk transmitters, whereas AI tokens and green bond function as risk receivers. Third, AI tokens are difficult to hedge and provide limited hedging ability compared to AI ETFs and green assets. However, multivariate portfolios effectively reduce AI tokens investment risk. Among them, the minimum correlation portfolio outperforms the minimum variance and minimum connectedness portfolios. 

---
# One-shot In-context Part Segmentation 

**Authors**: Zhenqi Dai, Ting Liu, Xingxing Zhang, Yunchao Wei, Yanning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01144)  

**Abstract**: In this paper, we present the One-shot In-context Part Segmentation (OIParts) framework, designed to tackle the challenges of part segmentation by leveraging visual foundation models (VFMs). Existing training-based one-shot part segmentation methods that utilize VFMs encounter difficulties when faced with scenarios where the one-shot image and test image exhibit significant variance in appearance and perspective, or when the object in the test image is partially visible. We argue that training on the one-shot example often leads to overfitting, thereby compromising the model's generalization capability. Our framework offers a novel approach to part segmentation that is training-free, flexible, and data-efficient, requiring only a single in-context example for precise segmentation with superior generalization ability. By thoroughly exploring the complementary strengths of VFMs, specifically DINOv2 and Stable Diffusion, we introduce an adaptive channel selection approach by minimizing the intra-class distance for better exploiting these two features, thereby enhancing the discriminatory power of the extracted features for the fine-grained parts. We have achieved remarkable segmentation performance across diverse object categories. The OIParts framework not only eliminates the need for extensive labeled data but also demonstrates superior generalization ability. Through comprehensive experimentation on three benchmark datasets, we have demonstrated the superiority of our proposed method over existing part segmentation approaches in one-shot settings. 

---
# How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach 

**Authors**: Ayeong Lee, Ethan Che, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01141)  

**Abstract**: Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability. 

---
# Statistical Tractability of Off-policy Evaluation of History-dependent Policies in POMDPs 

**Authors**: Yuheng Zhang, Nan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01134)  

**Abstract**: We investigate off-policy evaluation (OPE), a central and fundamental problem in reinforcement learning (RL), in the challenging setting of Partially Observable Markov Decision Processes (POMDPs) with large observation spaces. Recent works of Uehara et al. (2023a); Zhang & Jiang (2024) developed a model-free framework and identified important coverage assumptions (called belief and outcome coverage) that enable accurate OPE of memoryless policies with polynomial sample complexities, but handling more general target policies that depend on the entire observable history remained an open problem. In this work, we prove information-theoretic hardness for model-free OPE of history-dependent policies in several settings, characterized by additional assumptions imposed on the behavior policy (memoryless vs. history-dependent) and/or the state-revealing property of the POMDP (single-step vs. multi-step revealing). We further show that some hardness can be circumvented by a natural model-based algorithm -- whose analysis has surprisingly eluded the literature despite the algorithm's simplicity -- demonstrating provable separation between model-free and model-based OPE in POMDPs. 

---
# Beyond QA Pairs: Assessing Parameter-Efficient Fine-Tuning for Fact Embedding in LLMs 

**Authors**: Shivam Ratnakar, Abhiroop Talasila, Raghav Chamadiya, Nikhil Agarwal, Vinayak K Doifode  

**Link**: [PDF](https://arxiv.org/pdf/2503.01131)  

**Abstract**: This paper presents an extensive examination of Parameter-Efficient Fine-Tuning (PEFT) for embedding domain specific facts into Large Language Models (LLMs), focusing on improving the fine-tuning process by categorizing question-answer (QA) pairs into Factual and Conceptual classes using a BERT-based classifier. Two distinct Llama-2 models are fine-tuned based on these classifications and evaluated using larger models like GPT-3.5 Turbo and Gemini. Our results indicate that models trained on conceptual datasets outperform those trained on factual datasets. Additionally, we compare the efficiency of two synthetic fine-tuning dataset generation techniques, D-RAG and D-Naive, with D-Naive demonstrating superior performance. Although PEFT has shown effectiveness, our research indicates that it may not be the most optimal method for embedding facts into LLMs. However, it has demonstrated exceptional performance in instruction-based tasks. Our findings are reinforced by a 1000-sample dataset in the data center domain, where the fine-tuned Llama-2 7B model significantly outperforms the baseline model in generating product recommendations. Our study highlights the importance of QA pair categorization and synthetic dataset generation techniques in enhancing the performance of LLMs in specific domains. 

---
# FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion 

**Authors**: Yansong Xu, Junlin Li, Wei Zhang, Siyu Chen, Shengyong Zhang, Yuquan Leng, Weijia Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01109)  

**Abstract**: 3D gaussian splatting has advanced simultaneous localization and mapping (SLAM) technology by enabling real-time positioning and the construction of high-fidelity maps. However, the uncertainty in gaussian position and initialization parameters introduces challenges, often requiring extensive iterative convergence and resulting in redundant or insufficient gaussian representations. To address this, we introduce a novel adaptive densification method based on Fourier frequency domain analysis to establish gaussian priors for rapid convergence. Additionally, we propose constructing independent and unified sparse and dense maps, where a sparse map supports efficient tracking via Generalized Iterative Closest Point (GICP) and a dense map creates high-fidelity visual representations. This is the first SLAM system leveraging frequency domain analysis to achieve high-quality gaussian mapping in real-time. Experimental results demonstrate an average frame rate of 36 FPS on Replica and TUM RGB-D datasets, achieving competitive accuracy in both localization and mapping. 

---
# Ground contact and reaction force sensing for linear policy control of quadruped robot 

**Authors**: Harshita Mhaske, Aniket Mandhare, Jidong Huang, Yu Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01102)  

**Abstract**: Designing robots capable of traversing uneven terrain and overcoming physical obstacles has been a longstanding challenge in the field of robotics. Walking robots show promise in this regard due to their agility, redundant DOFs and intermittent ground contact of locomoting appendages. However, the complexity of walking robots and their numerous DOFs make controlling them extremely difficult and computation heavy. Linear policies trained with reinforcement learning have been shown to perform adequately to enable quadrupedal walking, while being computationally light weight. The goal of this research is to study the effect of augmentation of observation space of a linear policy with newer state variables on performance of the policy. Since ground contact and reaction forces are the primary means of robot-environment interaction, they are essential state variables on which the linear policy must be informed. Experimental results show that augmenting the observation space with ground contact and reaction force data trains policies with better survivability, better stability against external disturbances and higher adaptability to untrained conditions. 

---
# Fence Theorem: Preprocessing is Dual-Objective Semantic Structure Isolator in 3D Anomaly Detection 

**Authors**: Hanzhe Liang, Jie Zhou, Xuanxin Chen, Jinbao Wang, Can Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01100)  

**Abstract**: 3D anomaly detection (AD) is prominent but difficult due to lacking a unified theoretical foundation for preprocessing design. We establish the Fence Theorem, formalizing preprocessing as a dual-objective semantic isolator: (1) mitigating cross-semantic interference to the greatest extent feasible and (2) confining anomaly judgments to aligned semantic spaces wherever viable, thereby establishing intra-semantic comparability. Any preprocessing approach achieves this goal through a two-stage process of Emantic-Division and Spatial-Constraints stage. Through systematic deconstruction, we theoretically and experimentally subsume existing preprocessing methods under this theorem via tripartite evidence: qualitative analyses, quantitative studies, and mathematical proofs. Guided by the Fence Theorem, we implement Patch3D, consisting of Patch-Cutting and Patch-Matching modules, to segment semantic spaces and consolidate similar ones while independently modeling normal features within each space. Experiments on Anomaly-ShapeNet and Real3D-AD with different settings demonstrate that progressively finer-grained semantic alignment in preprocessing directly enhances point-level AD accuracy, providing inverse validation of the theorem's causal logic. 

---
# SolBench: A Dataset and Benchmark for Evaluating Functional Correctness in Solidity Code Completion and Repair 

**Authors**: Zaoyu Chen, Haoran Qin, Nuo Chen, Xiangyu Zhao, Lei Xue, Xiapu Luo, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01098)  

**Abstract**: Smart contracts are crucial programs on blockchains, and their immutability post-deployment makes functional correctness vital. Despite progress in code completion models, benchmarks for Solidity, the primary smart contract language, are lacking. Existing metrics like BLEU do not adequately assess the functional correctness of generated smart contracts. To fill this gap, we introduce SolBench, a benchmark for evaluating the functional correctness of Solidity smart contracts generated by code completion models. SolBench includes 4,178 functions from 1,155 Ethereum-deployed contracts. Testing advanced models revealed challenges in generating correct code without context, as Solidity functions rely on context-defined variables and interfaces. To address this, we propose a Retrieval-Augmented Code Repair framework. In this framework, an executor verifies functional correctness, and if necessary, an LLM repairs the code using retrieved snippets informed by executor traces. We conduct a comprehensive evaluation of both closed-source and open-source LLMs across various model sizes and series to assess their performance in smart contract completion. The results show that code repair and retrieval techniques effectively enhance the correctness of smart contract completion while reducing computational costs. 

---
# Depth-Adaptive Graph Neural Networks via Learnable Bakry-'Emery Curvature 

**Authors**: Asela Hevapathige, Ahad N. Zehmakan, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01079)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated strong representation learning capabilities for graph-based tasks. Recent advances on GNNs leverage geometric properties, such as curvature, to enhance its representation capabilities by modeling complex connectivity patterns and information flow within graphs. However, most existing approaches focus solely on discrete graph topology, overlooking diffusion dynamics and task-specific dependencies essential for effective learning. To address this, we propose integrating Bakry-Émery curvature, which captures both structural and task-driven aspects of information propagation. We develop an efficient, learnable approximation strategy, making curvature computation scalable for large graphs. Furthermore, we introduce an adaptive depth mechanism that dynamically adjusts message-passing layers per vertex based on its curvature, ensuring efficient propagation. Our theoretical analysis establishes a link between curvature and feature distinctiveness, showing that high-curvature vertices require fewer layers, while low-curvature ones benefit from deeper propagation. Extensive experiments on benchmark datasets validate the effectiveness of our approach, showing consistent performance improvements across diverse graph learning tasks. 

---
# Tackling Hallucination from Conditional Models for Medical Image Reconstruction with DynamicDPS 

**Authors**: Seunghoi Kim, Henry F. J. Tregidgo, Matteo Figini, Chen Jin, Sarang Joshi, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2503.01075)  

**Abstract**: Hallucinations are spurious structures not present in the ground truth, posing a critical challenge in medical image reconstruction, especially for data-driven conditional models. We hypothesize that combining an unconditional diffusion model with data consistency, trained on a diverse dataset, can reduce these hallucinations. Based on this, we propose DynamicDPS, a diffusion-based framework that integrates conditional and unconditional diffusion models to enhance low-quality medical images while systematically reducing hallucinations. Our approach first generates an initial reconstruction using a conditional model, then refines it with an adaptive diffusion-based inverse problem solver. DynamicDPS skips early stage in the reverse process by selecting an optimal starting time point per sample and applies Wolfe's line search for adaptive step sizes, improving both efficiency and image fidelity. Using diffusion priors and data consistency, our method effectively reduces hallucinations from any conditional model output. We validate its effectiveness in Image Quality Transfer for low-field MRI enhancement. Extensive evaluations on synthetic and real MR scans, including a downstream task for tissue volume estimation, show that DynamicDPS reduces hallucinations, improving relative volume estimation by over 15% for critical tissues while using only 5% of the sampling steps required by baseline diffusion models. As a model-agnostic and fine-tuning-free approach, DynamicDPS offers a robust solution for hallucination reduction in medical imaging. The code will be made publicly available upon publication. 

---
# Language-Guided Object Search in Agricultural Environments 

**Authors**: Advaith Balaji, Saket Pradhan, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01068)  

**Abstract**: Creating robots that can assist in farms and gardens can help reduce the mental and physical workload experienced by farm workers. We tackle the problem of object search in a farm environment, providing a method that allows a robot to semantically reason about the location of an unseen target object among a set of previously seen objects in the environment using a Large Language Model (LLM). We leverage object-to-object semantic relationships to plan a path through the environment that will allow us to accurately and efficiently locate our target object while also reducing the overall distance traveled, without needing high-level room or area-level semantic relationships. During our evaluations, we found that our method outperformed a current state-of-the-art baseline and our ablations. Our offline testing yielded an average path efficiency of 84%, reflecting how closely the predicted path aligns with the ideal path. Upon deploying our system on the Boston Dynamics Spot robot in a real-world farm environment, we found that our system had a success rate of 80%, with a success weighted by path length of 0.67, which demonstrates a reasonable trade-off between task success and path efficiency under real-world conditions. The project website can be viewed at this https URL 

---
# Scientific Reasoning: Assessment of Multimodal Generative LLMs 

**Authors**: Florian Dreyer, Ekaterina Kolos, Daria Matiash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01064)  

**Abstract**: Large language models (LLMs) can answer questions and reason about complex tasks, also from the scientific domain. We assess several multimodal LLMs (MLLMs) on ScienceQA and find that Gemini models show the highest accuracy with little context, and the highest textual similarity to human explanations with richer context. Adapter-tuning of smaller MLLMs did not lead to any reliable performance. Training from Gemini outputs consistently underperformed training from the original data. 

---
# SFO: Piloting VLM Feedback for Offline RL 

**Authors**: Jacob Beck  

**Link**: [PDF](https://arxiv.org/pdf/2503.01062)  

**Abstract**: While internet-scale image and textual data have enabled strong generalization in Vision-Language Models (VLMs), the absence of internet-scale control data has impeded the development of similar generalization in standard reinforcement learning (RL) agents. Although VLMs are fundamentally limited in their ability to solve control tasks due to their lack of action-conditioned training data, their capacity for image understanding allows them to provide valuable feedback in RL tasks by recognizing successful outcomes. A key challenge in Reinforcement Learning from AI Feedback (RLAIF) is determining how best to integrate VLM-derived signals into the learning process. We explore this question in the context of offline RL and introduce a class of methods called sub-trajectory filtered optimization. We identify three key insights. First, trajectory length plays a crucial role in offline RL, as full-trajectory preference learning exacerbates the stitching problem, necessitating the use of sub-trajectories. Second, even in Markovian environments, a non-Markovian reward signal from a sequence of images is required to assess trajectory improvement, as VLMs do not interpret control actions and must rely on visual cues over time. Third, a simple yet effective approach--filtered and weighted behavior cloning--consistently outperforms more complex reinforcement learning from human feedback-based methods. We propose sub-trajectory filtered behavior cloning, a method that leverages VLM feedback on sub-trajectories while incorporating a retrospective filtering mechanism that removes sub-trajectories preceding failures to improve robustness and prevent turbulence. This study is preliminary; we provide initial evidence through evaluations on a toy control domain. Please enjoy our airport puns. 

---
# MAPS: Multi-Fidelity AI-Augmented Photonic Simulation and Inverse Design Infrastructure 

**Authors**: Pingchuan Ma, Zhengqi Gao, Meng Zhang, Haoyu Yang, Mark Ren, Rena Huang, Duane S. Boning, Jiaqi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01046)  

**Abstract**: Inverse design has emerged as a transformative approach for photonic device optimization, enabling the exploration of high-dimensional, non-intuitive design spaces to create ultra-compact devices and advance photonic integrated circuits (PICs) in computing and interconnects. However, practical challenges, such as suboptimal device performance, limited manufacturability, high sensitivity to variations, computational inefficiency, and lack of interpretability, have hindered its adoption in commercial hardware. Recent advancements in AI-assisted photonic simulation and design offer transformative potential, accelerating simulations and design generation by orders of magnitude over traditional numerical methods. Despite these breakthroughs, the lack of an open-source, standardized infrastructure and evaluation benchmark limits accessibility and cross-disciplinary collaboration. To address this, we introduce MAPS, a multi-fidelity AI-augmented photonic simulation and inverse design infrastructure designed to bridge this gap. MAPS features three synergistic components: (1) MAPS-Data: A dataset acquisition framework for generating multi-fidelity, richly labeled devices, providing high-quality data for AI-for-optics research. (2) MAPS-Train: A flexible AI-for-photonics training framework offering a hierarchical data loading pipeline, customizable model construction, support for data- and physics-driven losses, and comprehensive evaluations. (3) MAPS-InvDes: An advanced adjoint inverse design toolkit that abstracts complex physics but exposes flexible optimization steps, integrates pre-trained AI models, and incorporates fabrication variation models. This infrastructure MAPS provides a unified, open-source platform for developing, benchmarking, and advancing AI-assisted photonic design workflows, accelerating innovation in photonic hardware optimization and scientific machine learning. 

---
# Language Models Predict Empathy Gaps Between Social In-groups and Out-groups 

**Authors**: Yu Hou, Hal Daumé III, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.01030)  

**Abstract**: Studies of human psychology have demonstrated that people are more motivated to extend empathy to in-group members than out-group members (Cikara et al., 2011). In this study, we investigate how this aspect of intergroup relations in humans is replicated by LLMs in an emotion intensity prediction task. In this task, the LLM is given a short description of an experience a person had that caused them to feel a particular emotion; the LLM is then prompted to predict the intensity of the emotion the person experienced on a numerical scale. By manipulating the group identities assigned to the LLM's persona (the "perceiver") and the person in the narrative (the "experiencer"), we measure how predicted emotion intensities differ between in-group and out-group settings. We observe that LLMs assign higher emotion intensity scores to in-group members than out-group members. This pattern holds across all three types of social groupings we tested: race/ethnicity, nationality, and religion. We perform an in-depth analysis on Llama-3.1-8B, the model which exhibited strongest intergroup bias among those tested. 

---
# LLM-Fusion: A Novel Multimodal Fusion Model for Accelerated Material Discovery 

**Authors**: Onur Boyar, Indra Priyadarsini, Seiji Takeda, Lisa Hamada  

**Link**: [PDF](https://arxiv.org/pdf/2503.01022)  

**Abstract**: Discovering materials with desirable properties in an efficient way remains a significant problem in materials science. Many studies have tackled this problem by using different sets of information available about the materials. Among them, multimodal approaches have been found to be promising because of their ability to combine different sources of information. However, fusion algorithms to date remain simple, lacking a mechanism to provide a rich representation of multiple modalities. This paper presents LLM-Fusion, a novel multimodal fusion model that leverages large language models (LLMs) to integrate diverse representations, such as SMILES, SELFIES, text descriptions, and molecular fingerprints, for accurate property prediction. Our approach introduces a flexible LLM-based architecture that supports multimodal input processing and enables material property prediction with higher accuracy than traditional methods. We validate our model on two datasets across five prediction tasks and demonstrate its effectiveness compared to unimodal and naive concatenation baselines. 

---
# MedUnifier: Unifying Vision-and-Language Pre-training on Medical Data with Vision Generation Task using Discrete Visual Representations 

**Authors**: Ziyang Zhang, Yang Yu, Yucheng Chen, Xulei Yang, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01019)  

**Abstract**: Despite significant progress in Vision-Language Pre-training (VLP), current approaches predominantly emphasize feature extraction and cross-modal comprehension, with limited attention to generating or transforming visual content. This gap hinders the model's ability to synthesize coherent and novel visual representations from textual prompts, thereby reducing the effectiveness of multi-modal learning. In this work, we propose MedUnifier, a unified VLP framework tailored for medical data. MedUnifier seamlessly integrates text-grounded image generation capabilities with multi-modal learning strategies, including image-text contrastive alignment, image-text matching and image-grounded text generation. Unlike traditional methods that reply on continuous visual representations, our approach employs visual vector quantization, which not only facilitates a more cohesive learning strategy for cross-modal understanding but also enhances multi-modal generation quality by effectively leveraging discrete representations. Our framework's effectiveness is evidenced by the experiments on established benchmarks, including uni-modal tasks (supervised fine-tuning), cross-modal tasks (image-text retrieval and zero-shot image classification), and multi-modal tasks (medical report generation, image synthesis), where it achieves state-of-the-art performance across various tasks. MedUnifier also offers a highly adaptable tool for a wide range of language and vision tasks in healthcare, marking advancement toward the development of a generalizable AI model for medical applications. 

---
# A Semantic Search Pipeline for Causality-driven Adhoc Information Retrieval 

**Authors**: Dhairya Dalal, Sharmi Dev Gupta, Bentolhoda Binaei  

**Link**: [PDF](https://arxiv.org/pdf/2503.01003)  

**Abstract**: We present a unsupervised semantic search pipeline for the Causality-driven Adhoc Information Retrieval (CAIR-2021) shared task. The CAIR shared task expands traditional information retrieval to support the retrieval of documents containing the likely causes of a query event. A successful system must be able to distinguish between topical documents and documents containing causal descriptions of events that are causally related to the query event. Our approach involves aggregating results from multiple query strategies over a semantic and lexical index. The proposed approach leads the CAIR-2021 leaderboard and outperformed both traditional IR and pure semantic embedding-based approaches. 

---
# Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs 

**Authors**: Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, Poulami Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.00979)  

**Abstract**: Autoregressive Transformers rely on Key-Value (KV) caching to accelerate inference. However, the linear growth of the KV cache with context length leads to excessive memory consumption and bandwidth constraints. This bottleneck is particularly problematic in real-time applications -- such as chatbots and interactive assistants -- where low latency and high memory efficiency are critical. Existing methods drop distant tokens or compress states in a lossy manner, sacrificing accuracy by discarding vital context or introducing bias.
We propose MorphKV, an inference-time technique that maintains a constant-sized KV cache while preserving accuracy. MorphKV balances long-range dependencies and local coherence during text generation. It eliminates early-token bias while retaining high-fidelity context by adaptively ranking tokens through correlation-aware selection. Unlike heuristic retention or lossy compression, MorphKV iteratively refines the KV cache via lightweight updates guided by attention patterns of recent tokens. This approach captures inter-token correlation with greater accuracy, crucial for tasks like content creation and code generation. Our studies on long-response tasks show 52.9$\%$ memory savings and 18.2$\%$ higher accuracy on average compared to state-of-the-art prior works, enabling efficient real-world deployment. 

---
# Using Synthetic Images to Augment Small Medical Image Datasets 

**Authors**: Minh H. Vu, Lorenzo Tronchin, Tufve Nyholm, Tommy Löfstedt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00962)  

**Abstract**: Recent years have witnessed a growing academic and industrial interest in deep learning (DL) for medical imaging. To perform well, DL models require very large labeled datasets. However, most medical imaging datasets are small, with a limited number of annotated samples. The reason they are small is usually because delineating medical images is time-consuming and demanding for oncologists. There are various techniques that can be used to augment a dataset, for example, to apply affine transformations or elastic transformations to available images, or to add synthetic images generated by a Generative Adversarial Network (GAN). In this work, we have developed a novel conditional variant of a current GAN method, the StyleGAN2, to generate multi-modal high-resolution medical images with the purpose to augment small medical imaging datasets with these synthetic images. We use the synthetic and real images from six datasets to train models for the downstream task of semantic segmentation. The quality of the generated medical images and the effect of this augmentation on the segmentation performance were evaluated afterward. Finally, the results indicate that the downstream segmentation models did not benefit from the generated images. Further work and analyses are required to establish how this augmentation affects the segmentation performance. 

---
# Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks 

**Authors**: Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00957)  

**Abstract**: As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at this https URL. 

---
# SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking 

**Authors**: Nam V. Nguyen, Dien X. Tran, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le  

**Link**: [PDF](https://arxiv.org/pdf/2503.00955)  

**Abstract**: The rise of misinformation, exacerbated by Large Language Models (LLMs) like GPT and Gemini, demands robust fact-checking solutions, especially for low-resource languages like Vietnamese. Existing methods struggle with semantic ambiguity, homonyms, and complex linguistic structures, often trading accuracy for efficiency. We introduce SemViQA, a novel Vietnamese fact-checking framework integrating Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC). Our approach balances precision and speed, achieving state-of-the-art results with 78.97\% strict accuracy on ISE-DSC01 and 80.82\% on ViWikiFC, securing 1st place in the UIT Data Science Challenge. Additionally, SemViQA Faster improves inference speed 7x while maintaining competitive accuracy. SemViQA sets a new benchmark for Vietnamese fact verification, advancing the fight against misinformation. The source code is available at: this https URL. 

---
# Cross Modality Medical Image Synthesis for Improving Liver Segmentation 

**Authors**: Muhammad Rafiq, Hazrat Ali, Ghulam Mujtaba, Zubair Shah, Shoaib Azmat  

**Link**: [PDF](https://arxiv.org/pdf/2503.00945)  

**Abstract**: Deep learning-based computer-aided diagnosis (CAD) of medical images requires large datasets. However, the lack of large publicly available labeled datasets limits the development of deep learning-based CAD systems. Generative Adversarial Networks (GANs), in particular, CycleGAN, can be used to generate new cross-domain images without paired training data. However, most CycleGAN-based synthesis methods lack the potential to overcome alignment and asymmetry between the input and generated data. We propose a two-stage technique for the synthesis of abdominal MRI using cross-modality translation of abdominal CT. We show that the synthetic data can help improve the performance of the liver segmentation network. We increase the number of abdominal MRI images through cross-modality image transformation of unpaired CT images using a CycleGAN inspired deformation invariant network called EssNet. Subsequently, we combine the synthetic MRI images with the original MRI images and use them to improve the accuracy of the U-Net on a liver segmentation task. We train the U-Net on real MRI images and then on real and synthetic MRI images. Consequently, by comparing both scenarios, we achieve an improvement in the performance of U-Net. In summary, the improvement achieved in the Intersection over Union (IoU) is 1.17%. The results show potential to address the data scarcity challenge in medical imaging. 

---
# Can AI Model the Complexities of Human Moral Decision-Making? A Qualitative Study of Kidney Allocation Decisions 

**Authors**: Vijay Keswani, Vincent Conitzer, Walter Sinnott-Armstrong, Breanna K. Nguyen, Hoda Heidari, Jana Schaich Borg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00940)  

**Abstract**: A growing body of work in Ethical AI attempts to capture human moral judgments through simple computational models. The key question we address in this work is whether such simple AI models capture {the critical} nuances of moral decision-making by focusing on the use case of kidney allocation. We conducted twenty interviews where participants explained their rationale for their judgments about who should receive a kidney. We observe participants: (a) value patients' morally-relevant attributes to different degrees; (b) use diverse decision-making processes, citing heuristics to reduce decision complexity; (c) can change their opinions; (d) sometimes lack confidence in their decisions (e.g., due to incomplete information); and (e) express enthusiasm and concern regarding AI assisting humans in kidney allocation decisions. Based on these findings, we discuss challenges of computationally modeling moral judgments {as a stand-in for human input}, highlight drawbacks of current approaches, and suggest future directions to address these issues. 

---
# Improving the Transferability of Adversarial Attacks by an Input Transpose 

**Authors**: Qing Wan, Shilong Deng, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00932)  

**Abstract**: Deep neural networks (DNNs) are highly susceptible to adversarial examples--subtle perturbations applied to inputs that are often imperceptible to humans yet lead to incorrect model predictions. In black-box scenarios, however, existing adversarial examples exhibit limited transferability and struggle to effectively compromise multiple unseen DNN models. Previous strategies enhance the cross-model generalization of adversarial examples by introducing versatility into adversarial perturbations, thereby improving transferability. However, further refining perturbation versatility often demands intricate algorithm development and substantial computation consumption. In this work, we propose an input transpose method that requires almost no additional labor and computation costs but can significantly improve the transferability of existing adversarial strategies. Even without adding adversarial perturbations, our method demonstrates considerable effectiveness in cross-model attacks. Our exploration finds that on specific datasets, a mere $1^\circ$ left or right rotation might be sufficient for most adversarial examples to deceive unseen models. Our further analysis suggests that this transferability improvement triggered by rotating only $1^\circ$ may stem from visible pattern shifts in the DNN's low-level feature maps. Moreover, this transferability exhibits optimal angles that, when identified under unrestricted query conditions, could potentially yield even greater performance. 

---
# Behavior Preference Regression for Offline Reinforcement Learning 

**Authors**: Padmanaba Srinivasan, William Knottenbelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00930)  

**Abstract**: Offline reinforcement learning (RL) methods aim to learn optimal policies with access only to trajectories in a fixed dataset. Policy constraint methods formulate policy learning as an optimization problem that balances maximizing reward with minimizing deviation from the behavior policy. Closed form solutions to this problem can be derived as weighted behavioral cloning objectives that, in theory, must compute an intractable partition function. Reinforcement learning has gained popularity in language modeling to align models with human preferences; some recent works consider paired completions that are ranked by a preference model following which the likelihood of the preferred completion is directly increased. We adapt this approach of paired comparison. By reformulating the paired-sample optimization problem, we fit the maximum-mode of the Q function while maximizing behavioral consistency of policy actions. This yields our algorithm, Behavior Preference Regression for offline RL (BPR). We empirically evaluate BPR on the widely used D4RL Locomotion and Antmaze datasets, as well as the more challenging V-D4RL suite, which operates in image-based state spaces. BPR demonstrates state-of-the-art performance over all domains. Our on-policy experiments suggest that BPR takes advantage of the stability of on-policy value functions with minimal perceptible performance degradation on Locomotion datasets. 

---
# Multimodal Distillation-Driven Ensemble Learning for Long-Tailed Histopathology Whole Slide Images Analysis 

**Authors**: Xitong Ling, Yifeng Ping, Jiawen Li, Jing Peng, Yuxuan Chen, Minxi Ouyang, Yizhi Wang, Yonghong He, Tian Guan, Xiaoping Liu, Lianghui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00915)  

**Abstract**: Multiple Instance Learning (MIL) plays a significant role in computational pathology, enabling weakly supervised analysis of Whole Slide Image (WSI) datasets. The field of WSI analysis is confronted with a severe long-tailed distribution problem, which significantly impacts the performance of classifiers. Long-tailed distributions lead to class imbalance, where some classes have sparse samples while others are abundant, making it difficult for classifiers to accurately identify minority class samples. To address this issue, we propose an ensemble learning method based on MIL, which employs expert decoders with shared aggregators and consistency constraints to learn diverse distributions and reduce the impact of class imbalance on classifier performance. Moreover, we introduce a multimodal distillation framework that leverages text encoders pre-trained on pathology-text pairs to distill knowledge and guide the MIL aggregator in capturing stronger semantic features relevant to class information. To ensure flexibility, we use learnable prompts to guide the distillation process of the pre-trained text encoder, avoiding limitations imposed by specific prompts. Our method, MDE-MIL, integrates multiple expert branches focusing on specific data distributions to address long-tailed issues. Consistency control ensures generalization across classes. Multimodal distillation enhances feature extraction. Experiments on Camelyon+-LT and PANDA-LT datasets show it outperforms state-of-the-art methods. 

---
# HiBench: Benchmarking LLMs Capability on Hierarchical Structure Reasoning 

**Authors**: Zhuohang Jiang, Pangjing Wu, Ziran Liang, Peter Q. Chen, Xu Yuan, Ye Jia, Jiancheng Tu, Chen Li, Peter H.F. Ng, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00912)  

**Abstract**: Structure reasoning is a fundamental capability of large language models (LLMs), enabling them to reason about structured commonsense and answer multi-hop questions. However, existing benchmarks for structure reasoning mainly focus on horizontal and coordinate structures (\emph{e.g.} graphs), overlooking the hierarchical relationships within them. Hierarchical structure reasoning is crucial for human cognition, particularly in memory organization and problem-solving. It also plays a key role in various real-world tasks, such as information extraction and decision-making. To address this gap, we propose HiBench, the first framework spanning from initial structure generation to final proficiency assessment, designed to benchmark the hierarchical reasoning capabilities of LLMs systematically. HiBench encompasses six representative scenarios, covering both fundamental and practical aspects, and consists of 30 tasks with varying hierarchical complexity, totaling 39,519 queries. To evaluate LLMs comprehensively, we develop five capability dimensions that depict different facets of hierarchical structure understanding. Through extensive evaluation of 20 LLMs from 10 model families, we reveal key insights into their capabilities and limitations: 1) existing LLMs show proficiency in basic hierarchical reasoning tasks; 2) they still struggle with more complex structures and implicit hierarchical representations, especially in structural modification and textual reasoning. Based on these findings, we create a small yet well-designed instruction dataset, which enhances LLMs' performance on HiBench by an average of 88.84\% (Llama-3.1-8B) and 31.38\% (Qwen2.5-7B) across all tasks. The HiBench dataset and toolkit are available here, this https URL, to encourage evaluation. 

---
# S4M: S4 for multivariate time series forecasting with Missing values 

**Authors**: Jing Peng, Meiqi Yang, Qiong Zhang, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00900)  

**Abstract**: Multivariate time series data play a pivotal role in a wide range of real-world applications. However, the presence of block missing data introduces significant challenges, often compromising the performance of predictive models. Traditional two-step approaches, which first impute missing values and then perform forecasting, are prone to error accumulation, particularly in complex multivariate settings characterized by high missing ratios and intricate dependency structures. In this work, we introduce S4M, an end-to-end time series forecasting framework that seamlessly integrates missing data handling into the Structured State Space Sequence (S4) model architecture. Unlike conventional methods that treat imputation as a separate preprocessing step, S4M leverages the latent space of S4 models to directly recognize and represent missing data patterns, thereby more effectively capturing the underlying temporal and multivariate dependencies. Our framework comprises two key components: the Adaptive Temporal Prototype Mapper (ATPM) and the Missing-Aware Dual Stream S4 (MDS-S4). The ATPM employs a prototype bank to derive robust and informative representations from historical data patterns, while the MDS-S4 processes these representations alongside missingness masks as dual input streams to enable accurate forecasting. Through extensive empirical evaluations on diverse real-world datasets, we demonstrate that S4M consistently achieves state-of-the-art performance. These results underscore the efficacy of our integrated approach in handling missing data, showcasing its robustness and superiority over traditional imputation-based methods. Our findings highlight the potential of S4M to advance reliable time series forecasting in practical applications, offering a promising direction for future research and deployment. Code is available at this https URL. 

---
# A Simple and Effective Reinforcement Learning Method for Text-to-Image Diffusion Fine-tuning 

**Authors**: Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, Satya Narayan Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2503.00897)  

**Abstract**: Reinforcement learning ( RL)-based fine-tuning has emerged as a powerful approach for aligning diffusion models with black-box objectives. Proximal policy optimization (PPO) is the most popular choice of method for policy optimization. While effective in terms of performance, PPO is highly sensitive to hyper-parameters and involves substantial computational overhead. REINFORCE, on the other hand, mitigates some computational complexities such as high memory overhead and sensitive hyper-parameter tuning, but has suboptimal performance due to high-variance and sample inefficiency. While the variance of the REINFORCE can be reduced by sampling multiple actions per input prompt and using a baseline correction term, it still suffers from sample inefficiency. To address these challenges, we systematically analyze the efficiency-effectiveness trade-off between REINFORCE and PPO, and propose leave-one-out PPO ( LOOP), a novel RL for diffusion fine-tuning method. LOOP combines variance reduction techniques from REINFORCE, such as sampling multiple actions per input prompt and a baseline correction term, with the robustness and sample efficiency of PPO via clipping and importance sampling. Our results demonstrate that LOOP effectively improves diffusion models on various black-box objectives, and achieves a better balance between computational efficiency and performance. 

---
# Estimating Blood Pressure with a Camera: An Exploratory Study of Ambulatory Patients with Cardiovascular Disease 

**Authors**: Theodore Curran, Chengqian Ma, Xin Liu, Daniel McDuff, Girish Narayanswamy, George Stergiou, Shwetak Patel, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00890)  

**Abstract**: Hypertension is a leading cause of morbidity and mortality worldwide. The ability to diagnose and treat hypertension in the ambulatory population is hindered by limited access and poor adherence to current methods of monitoring blood pressure (BP), specifically, cuff-based devices. Remote photoplethysmography (rPPG) evaluates an individual's pulse waveform through a standard camera without physical contact. Cameras are readily available to the majority of the global population via embedded technologies such as smartphones, thus rPPG is a scalable and promising non-invasive method of BP monitoring. The few studies investigating rPPG for BP measurement have excluded high-risk populations, including those with cardiovascular disease (CVD) or its risk factors, as well as subjects in active cardiac arrhythmia. The impact of arrhythmia, like atrial fibrillation, on the prediction of BP using rPPG is currently uncertain. We performed a study to better understand the relationship between rPPG and BP in a real-world sample of ambulatory patients from a cardiology clinic with established CVD or risk factors for CVD. We collected simultaneous rPPG, PPG, BP, ECG, and other vital signs data from 143 subjects while at rest, and used this data plus demographics to train a deep learning model to predict BP. We report that facial rPPG yields a signal that is comparable to finger PPG. Pulse wave analysis (PWA)-based BP estimates on this cohort performed comparably to studies on healthier subjects, and notably, the accuracy of BP prediction in subjects with atrial fibrillation was not inferior to subjects with normal sinus rhythm. In a binary classification task, the rPPG model identified subjects with systolic BP $\geq$ 130 mm Hg with a positive predictive value of 71% (baseline prevalence 48.3%), highlighting the potential of rPPG for hypertension monitoring. 

---
# Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization 

**Authors**: You Shen, Zhipeng Zhang, Xinyang Li, Yansong Qu, Yu Lin, Shengchuan Zhang, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00881)  

**Abstract**: Representing 3D scenes from multiview images is a core challenge in computer vision and graphics, which requires both precise rendering and accurate reconstruction. Recently, 3D Gaussian Splatting (3DGS) has garnered significant attention for its high-quality rendering and fast inference speed. Yet, due to the unstructured and irregular nature of Gaussian point clouds, ensuring accurate geometry reconstruction remains difficult. Existing methods primarily focus on geometry regularization, with common approaches including primitive-based and dual-model frameworks. However, the former suffers from inherent conflicts between rendering and reconstruction, while the latter is computationally and storage-intensive. To address these challenges, we propose CarGS, a unified model leveraging Contribution-adaptive regularization to achieve simultaneous, high-quality rendering and surface reconstruction. The essence of our framework is learning adaptive contribution for Gaussian primitives by squeezing the knowledge from geometry regularization into a compact MLP. Additionally, we introduce a geometry-guided densification strategy with clues from both normals and Signed Distance Fields (SDF) to improve the capability of capturing high-frequency details. Our design improves the mutual learning of the two tasks, meanwhile its unified structure does not require separate models as in dual-model based approaches, guaranteeing efficiency. Extensive experiments demonstrate the ability to achieve state-of-the-art (SOTA) results in both rendering fidelity and reconstruction accuracy while maintaining real-time speed and minimal storage size. 

---
# CyberCScope: Mining Skewed Tensor Streams and Online Anomaly Detection in Cybersecurity Systems 

**Authors**: Kota Nakamura, Koki Kawabata, Shungo Tanaka, Yasuko Matsubara, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2503.00871)  

**Abstract**: Cybersecurity systems are continuously producing a huge number of time-stamped events in the form of high-order tensors, such as {count; time, port, flow duration, packet size, . . . }, and so how can we detect anomalies/intrusions in real time? How can we identify multiple types of intrusions and capture their characteristic behaviors? The tensor data consists of categorical and continuous attributes and the data distributions of continuous attributes typically exhibit skew. These data properties require handling skewed infinite and finite dimensional spaces simultaneously. In this paper, we propose a novel streaming method, namely CyberCScope. The method effectively decomposes incoming tensors into major trends while explicitly distinguishing between categorical and skewed continuous attributes. To our knowledge, it is the first to compute hybrid skewed infinite and finite dimensional decomposition. Based on this decomposition, it streamingly finds distinct time-evolving patterns, enabling the detection of multiple types of anomalies. Extensive experiments on large-scale real datasets demonstrate that CyberCScope detects various intrusions with higher accuracy than state-of-the-art baselines while providing meaningful summaries for the intrusions that occur in practice. 

---
# Babel: Open Multilingual Large Language Models Serving Over 90% of Global Speakers 

**Authors**: Yiran Zhao, Chaoqun Liu, Yue Deng, Jiahao Ying, Mahani Aljunied, Zhaodonghui Li, Lidong Bing, Hou Pong Chan, Yu Rong, Deli Zhao, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00865)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing (NLP), yet open-source multilingual LLMs remain scarce, with existing models often limited in language coverage. Such models typically prioritize well-resourced languages, while widely spoken but under-resourced languages are often overlooked. To address this disparity, we introduce $\texttt{Babel}$, an open multilingual LLM that covers the top 25 languages by number of speakers, supports over 90% of the global population, and includes many languages neglected by other open multilingual LLMs. Unlike traditional continue pretraining approaches, Babel expands its parameter count through a layer extension technique that elevates Babel's performance ceiling. We introduce two variants: $\texttt{Babel-9B}$, designed for efficient inference and fine-tuning, and $\texttt{Babel-83B}$, which sets a new standard for open multilingual LLMs. Extensive evaluations on multilingual tasks demonstrate its superior performance compared to open LLMs of comparable size. In addition, using open-source supervised fine-tuning datasets, Babel achieves remarkable performance, with Babel-9B-Chat leading among 10B-sized LLMs and Babel-83B-Chat setting a new standard for multilingual tasks, reaching the same level of commercial models. 

---
# MTReD: 3D Reconstruction Dataset for Fly-over Videos of Maritime Domain 

**Authors**: Rui Yi Yong, Samuel Picosson, Arnold Wiliem  

**Link**: [PDF](https://arxiv.org/pdf/2503.00853)  

**Abstract**: This work tackles 3D scene reconstruction for a video fly-over perspective problem in the maritime domain, with a specific emphasis on geometrically and visually sound reconstructions. This will allow for downstream tasks such as segmentation, navigation, and localization. To our knowledge, there is no dataset available in this domain. As such, we propose a novel maritime 3D scene reconstruction benchmarking dataset, named as MTReD (Maritime Three-Dimensional Reconstruction Dataset). The MTReD comprises 19 fly-over videos curated from the Internet containing ships, islands, and coastlines. As the task is aimed towards geometrical consistency and visual completeness, the dataset uses two metrics: (1) Reprojection error; and (2) Perception based metrics. We find that existing perception-based metrics, such as Learned Perceptual Image Patch Similarity (LPIPS), do not appropriately measure the completeness of a reconstructed image. Thus, we propose a novel semantic similarity metric utilizing DINOv2 features coined DiFPS (DinoV2 Features Perception Similarity). We perform initial evaluation on two baselines: (1) Structured from Motion (SfM) through Colmap; and (2) the recent state-of-the-art MASt3R model. We find that the reconstructed scenes by MASt3R have higher reprojection errors, but superior perception based metric scores. To this end, some pre-processing methods are explored, and we find a pre-processing method which improves both the reprojection error and perception-based score. We envisage our proposed MTReD to stimulate further research in these directions. The dataset and all the code will be made available in this https URL. 

---
# Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners 

**Authors**: Miao Peng, Nuo Chen, Zongrui Suo, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00845)  

**Abstract**: Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs. 

---
# AI Agents for Ground-Based Gamma Astronomy 

**Authors**: D. Kostunin, V. Sotnikov, S. Golovachev, A. Strube  

**Link**: [PDF](https://arxiv.org/pdf/2503.00821)  

**Abstract**: Next-generation instruments for ground-based gamma-ray astronomy are marked by a substantial increase in complexity, featuring dozens of telescopes. This leap in scale introduces significant challenges in managing system operations and offline data analysis. Methods, which depend on advanced personnel training and sophisticated software, become increasingly strained as system complexity grows, making it more challenging to effectively support users in such a multifaceted environment. To address these challenges, we propose the development of AI agents based on instruction-finetuned large language models (LLMs). These agents align with specific documentation and codebases, understand the environmental context, operate with external APIs, and communicate with humans in natural language. Leveraging the advanced capabilities of modern LLMs, which can process and retain vast amounts of information, these AI agents offer a transformative approach to system management and data analysis by automating complex tasks and providing intelligent assistance. We present two prototypes that integrate with the Cherenkov Telescope Array Observatory pipelines for operations and offline data analysis. The first prototype automates data model implementation and maintenance for the Configuration Database of the Array Control and Data Acquisition (ACADA). The second prototype is an open-access code generation application tailored for data analysis based on the Gammapy framework. 

---
# DELST: Dual Entailment Learning for Hyperbolic Image-Gene Pretraining in Spatial Transcriptomics 

**Authors**: Xulin Chen, Junzhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00804)  

**Abstract**: Spatial transcriptomics (ST) maps gene expression within tissue at individual spots, making it a valuable resource for multimodal representation learning. Additionally, ST inherently contains rich hierarchical information both across and within modalities. For instance, different spots exhibit varying numbers of nonzero gene expressions, corresponding to different levels of cellular activity and semantic hierarchies. However, existing methods rely on contrastive alignment of image-gene pairs, failing to accurately capture the intricate hierarchical relationships in ST data. Here, we propose DELST, the first framework to embed hyperbolic representations while modeling hierarchy for image-gene pretraining at two levels: (1) Cross-modal entailment learning, which establishes an order relationship between genes and images to enhance image representation generalization; (2) Intra-modal entailment learning, which encodes gene expression patterns as hierarchical relationships, guiding hierarchical learning across different samples at a global scale and integrating biological insights into single-modal representations. Extensive experiments on ST benchmarks annotated by pathologists demonstrate the effectiveness of our framework, achieving improved predictive performance compared to existing methods. Our code and models are available at: this https URL. 

---
# Towards Reliable LLM-Driven Fuzz Testing: Vision and Road Ahead 

**Authors**: Yiran Cheng, Hong Jin Kang, Lwin Khin Shar, Chaopeng Dong, Zhiqiang Shi, Shichao Lv, Limin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00795)  

**Abstract**: Fuzz testing is a crucial component of software security assessment, yet its effectiveness heavily relies on valid fuzz drivers and diverse seed inputs. Recent advancements in Large Language Models (LLMs) offer transformative potential for automating fuzz testing (LLM4Fuzz), particularly in generating drivers and seeds. However, current LLM4Fuzz solutions face critical reliability challenges, including low driver validity rates and seed quality trade-offs, hindering their practical adoption.
This paper aims to examine the reliability bottlenecks of LLM-driven fuzzing and explores potential research directions to address these limitations. It begins with an overview of the current development of LLM4SE and emphasizes the necessity for developing reliable LLM4Fuzz solutions. Following this, the paper envisions a vision where reliable LLM4Fuzz transforms the landscape of software testing and security for industry, software development practitioners, and economic accessibility. It then outlines a road ahead for future research, identifying key challenges and offering specific suggestions for the researchers to consider. This work strives to spark innovation in the field, positioning reliable LLM4Fuzz as a fundamental component of modern software testing. 

---
# Bridging Spectral-wise and Multi-spectral Depth Estimation via Geometry-guided Contrastive Learning 

**Authors**: Ukcheol Shin, Kyunghyun Lee, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00793)  

**Abstract**: Deploying depth estimation networks in the real world requires high-level robustness against various adverse conditions to ensure safe and reliable autonomy. For this purpose, many autonomous vehicles employ multi-modal sensor systems, including an RGB camera, NIR camera, thermal camera, LiDAR, or Radar. They mainly adopt two strategies to use multiple sensors: modality-wise and multi-modal fused inference. The former method is flexible but memory-inefficient, unreliable, and vulnerable. Multi-modal fusion can provide high-level reliability, yet it needs a specialized architecture. In this paper, we propose an effective solution, named align-and-fuse strategy, for the depth estimation from multi-spectral images. In the align stage, we align embedding spaces between multiple spectrum bands to learn shareable representation across multi-spectral images by minimizing contrastive loss of global and spatially aligned local features with geometry cue. After that, in the fuse stage, we train an attachable feature fusion module that can selectively aggregate the multi-spectral features for reliable and robust prediction results. Based on the proposed method, a single-depth network can achieve both spectral-invariant and multi-spectral fused depth estimation while preserving reliability, memory efficiency, and flexibility. 

---
# Taming Infinity one Chunk at a Time: Concisely Represented Strategies in One-Counter MDPs 

**Authors**: Michal Ajdarów, James C. A. Main, Petr Novotný, Mickael Randour  

**Link**: [PDF](https://arxiv.org/pdf/2503.00788)  

**Abstract**: Markov decision processes (MDPs) are a canonical model to reason about decision making within a stochastic environment. We study a fundamental class of infinite MDPs: one-counter MDPs (OC-MDPs). They extend finite MDPs via an associated counter taking natural values, thus inducing an infinite MDP over the set of configurations (current state and counter value). We consider two characteristic objectives: reaching a target state (state-reachability), and reaching a target state with counter value zero (selective termination). The synthesis problem for the latter is not known to be decidable and connected to major open problems in number theory. Furthermore, even seemingly simple strategies (e.g., memoryless ones) in OC-MDPs might be impossible to build in practice (due to the underlying infinite configuration space): we need finite, and preferably small, representations.
To overcome these obstacles, we introduce two natural classes of concisely represented strategies based on a (possibly infinite) partition of counter values in intervals. For both classes, and both objectives, we study the verification problem (does a given strategy ensure a high enough probability for the objective?), and two synthesis problems (does there exist such a strategy?): one where the interval partition is fixed as input, and one where it is only parameterized. We develop a generic approach based on a compression of the induced infinite MDP that yields decidability in all cases, with all complexities within PSPACE. 

---
# Graph Attention Networks Unleashed: A Fast and Explainable Vulnerability Assessment Framework for Microgrids 

**Authors**: Wei Liu, Tao Zhang, Chenhui Lin, Kaiwen Li, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00786)  

**Abstract**: Independent microgrids are crucial for supplying electricity by combining distributed energy resources and loads in scenarios like isolated islands and field combat. Fast and accurate assessments of microgrid vulnerability against intentional attacks or natural disasters are essential for effective risk prevention and design optimization. However, conventional Monte Carlo simulation (MCS) methods are computationally expensive and time-consuming, while existing machine learning-based approaches often lack accuracy and explainability. To address these challenges, this study proposes a fast and explainable vulnerability assessment framework that integrates MCS with a graph attention network enhanced by self-attention pooling (GAT-S). MCS generates training data, while the GAT-S model learns the structural and electrical characteristics of the microgrid and further assesses its vulnerability intelligently. The GAT-S improves explainability and computational efficiency by dynamically assigning attention weights to critical nodes. Comprehensive experimental evaluations across various microgrid configurations demonstrate that the proposed framework provides accurate vulnerability assessments, achieving a mean squared error as low as 0.001, real-time responsiveness within 1 second, and delivering explainable results. 

---
# FLOAT Drone: A Fully-actuated Coaxial Aerial Robot for Close-Proximity Operations 

**Authors**: Junxiao Lin, Shuhang Ji, Yuze Wu, Tianyue Wu, Zhichao Han, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00785)  

**Abstract**: How to endow aerial robots with the ability to operate in close proximity remains an open problem. The core challenges lie in the propulsion system's dual-task requirement: generating manipulation forces while simultaneously counteracting gravity. These competing demands create dynamic coupling effects during physical interactions. Furthermore, rotor-induced airflow disturbances critically undermine operational reliability. Although fully-actuated unmanned aerial vehicles (UAVs) alleviate dynamic coupling effects via six-degree-of-freedom (6-DoF) force-torque decoupling, existing implementations fail to address the aerodynamic interference between drones and environments. They also suffer from oversized designs, which compromise maneuverability and limit their applications in various operational scenarios. To address these limitations, we present FLOAT Drone (FuLly-actuated cOaxial Aerial roboT), a novel fully-actuated UAV featuring two key structural innovations. By integrating control surfaces into fully-actuated systems for the first time, we significantly suppress lateral airflow disturbances during operations. Furthermore, a coaxial dual-rotor configuration enables a compact size while maintaining high hovering efficiency. Through dynamic modeling, we have developed hierarchical position and attitude controllers that support both fully-actuated and underactuated modes. Experimental validation through comprehensive real-world experiments confirms the system's functional capabilities in close-proximity operations. 

---
# Towards Efficient Educational Chatbots: Benchmarking RAG Frameworks 

**Authors**: Umar Ali Khan, Ekram Khan, Fiza Khan, Athar Ali Moinuddin  

**Link**: [PDF](https://arxiv.org/pdf/2503.00781)  

**Abstract**: Large Language Models (LLMs) have proven immensely beneficial in education by capturing vast amounts of literature-based information, allowing them to generate context without relying on external sources. In this paper, we propose a generative AI-powered GATE question-answering framework (GATE stands for Graduate Aptitude Test in Engineering) that leverages LLMs to explain GATE solutions and support students in their exam preparation. We conducted extensive benchmarking to select the optimal embedding model and LLM, evaluating our framework based on criteria such as latency, faithfulness, and relevance, with additional validation through human evaluation. Our chatbot integrates state-of-the-art embedding models and LLMs to deliver accurate, context-aware responses. Through rigorous experimentation, we identified configurations that balance performance and computational efficiency, ensuring a reliable chatbot to serve students' needs. Additionally, we discuss the challenges faced in data processing and modeling and implemented solutions. Our work explores the application of Retrieval-Augmented Generation (RAG) for GATE Q/A explanation tasks, and our findings demonstrate significant improvements in retrieval accuracy and response quality. This research offers practical insights for developing effective AI-driven educational tools while highlighting areas for future enhancement in usability and scalability. 

---
# Enhanced Multi-Class Classification of Gastrointestinal Endoscopic Images with Interpretable Deep Learning Model 

**Authors**: Astitva Kamble, Vani Bandodkar, Saakshi Dharmadhikary, Veena Anand, Pradyut Kumar Sanki, Mei X. Wu, Biswabandhu Jana  

**Link**: [PDF](https://arxiv.org/pdf/2503.00780)  

**Abstract**: Endoscopy serves as an essential procedure for evaluating the gastrointestinal (GI) tract and plays a pivotal role in identifying GI-related disorders. Recent advancements in deep learning have demonstrated substantial progress in detecting abnormalities through intricate models and data augmentation this http URL research introduces a novel approach to enhance classification accuracy using 8,000 labeled endoscopic images from the Kvasir dataset, categorized into eight distinct classes. Leveraging EfficientNetB3 as the backbone, the proposed architecture eliminates reliance on data augmentation while preserving moderate model complexity. The model achieves a test accuracy of 94.25%, alongside precision and recall of 94.29% and 94.24% respectively. Furthermore, Local Interpretable Model-agnostic Explanation (LIME) saliency maps are employed to enhance interpretability by defining critical regions in the images that influenced model predictions. Overall, this work highlights the importance of AI in advancing medical imaging by combining high classification accuracy with interpretability. 

---
# LLMs are everywhere: Ubiquitous Utilization of AI Models through Air Computing 

**Authors**: Baris Yamansavascilar, Atay Ozgovde, Cem Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00767)  

**Abstract**: We are witnessing a new era where problem-solving and cognitive tasks are being increasingly delegated to Large Language Models (LLMs) across diverse domains, ranging from code generation to holiday planning. This trend also creates a demand for the ubiquitous execution of LLM-powered applications in a wide variety of environments in which traditional terrestrial 2D networking infrastructures may prove insufficient. A promising solution in this context is to extend edge computing into a 3D setting to include aerial platforms organized in multiple layers, a paradigm we refer to as air computing, to augment local devices for running LLM and Generative AI (GenAI) applications. This approach alleviates the strain on existing infrastructure while enhancing service efficiency by offloading computational tasks to the corresponding air units such as UAVs. Furthermore, the coordinated deployment of various air units can significantly improve the Quality of Experience (QoE) by ensuring seamless, adaptive, and resilient task execution. In this study, we investigate the synergy between LLM-based applications and air computing, exploring their potential across various use cases. Additionally, we present a disaster response case study demonstrating how the collaborative utilization of LLMs and air computing can significantly improve outcomes in critical situations. 

---
# MR-EIT: Multi-Resolution Reconstruction for Electrical Impedance Tomography via Data-Driven and Unsupervised Dual-Mode Neural Networks 

**Authors**: Fangming Shi, Jinzhen Liu, Xiangqian Meng, Yapeng Zhou, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.00762)  

**Abstract**: This paper presents a multi-resolution reconstruction method for Electrical Impedance Tomography (EIT), referred to as MR-EIT, which is capable of operating in both supervised and unsupervised learning modes. MR-EIT integrates an ordered feature extraction module and an unordered coordinate feature expression module. The former achieves the mapping from voltage to two-dimensional conductivity features through pre-training, while the latter realizes multi-resolution reconstruction independent of the order and size of the input sequence by utilizing symmetric functions and local feature extraction mechanisms. In the data-driven mode, MR-EIT reconstructs high-resolution images from low-resolution data of finite element meshes through two stages of pre-training and joint training, and demonstrates excellent performance in simulation experiments. In the unsupervised learning mode, MR-EIT does not require pre-training data and performs iterative optimization solely based on measured voltages to rapidly achieve image reconstruction from low to high resolution. It shows robustness to noise and efficient super-resolution reconstruction capabilities in both simulation and real water tank experiments. Experimental results indicate that MR-EIT outperforms the comparison methods in terms of Structural Similarity (SSIM) and Relative Image Error (RIE), especially in the unsupervised learning mode, where it can significantly reduce the number of iterations and improve image reconstruction quality. 

---
# RAPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery 

**Authors**: Hongchao Gu, Dexun Li, Kuicai Dong, Hao Zhang, Hang Lv, Hao Wang, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00751)  

**Abstract**: Generating knowledge-intensive and comprehensive long texts, such as encyclopedia articles, remains significant challenges for Large Language Models. It requires not only the precise integration of facts but also the maintenance of thematic coherence throughout the article. Existing methods, such as direct generation and multi-agent discussion, often struggle with issues like hallucinations, topic incoherence, and significant latency. To address these challenges, we propose RAPID, an efficient retrieval-augmented long text generation framework. RAPID consists of three main modules: (1) Retrieval-augmented preliminary outline generation to reduce hallucinations, (2) Attribute-constrained search for efficient information discovery, (3) Plan-guided article generation for enhanced coherence. Extensive experiments on our newly compiled benchmark dataset, FreshWiki-2024, demonstrate that RAPID significantly outperforms state-of-the-art methods across a wide range of evaluation metrics (e.g. long-text generation, outline quality, latency, etc). Our work provides a robust and efficient solution to the challenges of automated long-text generation. 

---
# Edge Prompt Tuning for Graph Neural Networks 

**Authors**: Xingbo Fu, Yinhan He, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00750)  

**Abstract**: Pre-training powerful Graph Neural Networks (GNNs) with unlabeled graph data in a self-supervised manner has emerged as a prominent technique in recent years. However, inevitable objective gaps often exist between pre-training and downstream tasks. To bridge this gap, graph prompt tuning techniques design and learn graph prompts by manipulating input graphs or reframing downstream tasks as pre-training tasks without fine-tuning the pre-trained GNN models. While recent graph prompt tuning methods have proven effective in adapting pre-trained GNN models for downstream tasks, they overlook the crucial role of edges in graph prompt design, which can significantly affect the quality of graph representations for downstream tasks. In this study, we propose EdgePrompt, a simple yet effective graph prompt tuning method from the perspective of edges. Unlike previous studies that design prompt vectors on node features, EdgePrompt manipulates input graphs by learning additional prompt vectors for edges and incorporates the edge prompts through message passing in the pre-trained GNN models to better embed graph structural information for downstream tasks. Our method is compatible with prevalent GNN architectures pre-trained under various pre-training strategies and is universal for different downstream tasks. We provide comprehensive theoretical analyses of our method regarding its capability of handling node classification and graph classification as downstream tasks. Extensive experiments on ten graph datasets under four pre-training strategies demonstrate the superiority of our proposed method against six baselines. Our code is available at this https URL. 

---
# Confounder-Aware Medical Data Selection for Fine-Tuning Pretrained Vision Models 

**Authors**: Anyang Ji, Qingbo Kang, Wei Xu, Changfan Wang, Kang Li, Qicheng Lao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00744)  

**Abstract**: The emergence of large-scale pre-trained vision foundation models has greatly advanced the medical imaging field through the pre-training and fine-tuning paradigm. However, selecting appropriate medical data for downstream fine-tuning remains a significant challenge considering its annotation cost, privacy concerns, and the detrimental effects of confounding variables. In this work, we present a confounder-aware medical data selection approach for medical dataset curation aiming to select minimal representative data by strategically mitigating the undesirable impact of confounding variables while preserving the natural distribution of the dataset. Our approach first identifies confounding variables within data and then develops a distance-based data selection strategy for confounder-aware sampling with a constrained budget in the data size. We validate the superiority of our approach through extensive experiments across diverse medical imaging modalities, highlighting its effectiveness in addressing the substantial impact of confounding variables and enhancing the fine-tuning efficiency in the medical imaging domain, compared to other data selection approaches. 

---
# LADDER: Self-Improving LLMs Through Recursive Problem Decomposition 

**Authors**: Toby Simonds, Akira Yoshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2503.00735)  

**Abstract**: We introduce LADDER (Learning through Autonomous Difficulty-Driven Example Recursion), a framework enabling LLMs to autonomously improve their problem-solving capabilities through self-guided learning. By recursively generating and solving progressively simpler variants of complex problems, LADDER enables models to progressively learn through reinforcement learning how to solve harder problems. This self-improvement process is guided by verifiable reward signals, allowing the model to assess its solutions. Unlike prior approaches requiring curated datasets or human feedback, LADDER leverages the model's own capabilities to easier variants of sample questions. We demonstrate LADDER's effectiveness on mathematical integration tasks, where it improves a Llama 3B model's accuracy from 1\% to 82\% on undergraduate-level problems and enables a 7B parameter model to achieve state-of-the-art performance (70\%) on the MIT Integration Bee examination for it's model size. We also introduce TTRL (Test-Time Reinforcement Learning), a method that generates variants of test problems at inference time and applies reinforcement learning to further improve performance. By further creating and solving related problems during testing, TTRL enables the 7B model to achieve a score of 85\%, surpassing o1. These results showcase how strategic self-directed learning can achieve significant capability improvements without relying on architectural scaling or human supervision. 

---
# CLEA: Closed-Loop Embodied Agent for Enhancing Task Execution in Dynamic Environments 

**Authors**: Mingcong Lei, Ge Wang, Yiming Zhao, Zhixin Mai, Qing Zhao, Yao Guo, Zhen Li, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.00729)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in the hierarchical decomposition of complex tasks through semantic reasoning. However, their application in embodied systems faces challenges in ensuring reliable execution of subtask sequences and achieving one-shot success in long-term task completion. To address these limitations in dynamic environments, we propose Closed-Loop Embodied Agent (CLEA) -- a novel architecture incorporating four specialized open-source LLMs with functional decoupling for closed-loop task management. The framework features two core innovations: (1) Interactive task planner that dynamically generates executable subtasks based on the environmental memory, and (2) Multimodal execution critic employing an evaluation framework to conduct a probabilistic assessment of action feasibility, triggering hierarchical re-planning mechanisms when environmental perturbations exceed preset thresholds. To validate CLEA's effectiveness, we conduct experiments in a real environment with manipulable objects, using two heterogeneous robots for object search, manipulation, and search-manipulation integration tasks. Across 12 task trials, CLEA outperforms the baseline model, achieving a 67.3% improvement in success rate and a 52.8% increase in task completion rate. These results demonstrate that CLEA significantly enhances the robustness of task planning and execution in dynamic environments. 

---
# From Understanding the World to Intervening in It: A Unified Multi-Scale Framework for Embodied Cognition 

**Authors**: Maijunxian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00727)  

**Abstract**: In this paper, we propose AUKAI, an Adaptive Unified Knowledge-Action Intelligence for embodied cognition that seamlessly integrates perception, memory, and decision-making via multi-scale error feedback. Interpreting AUKAI as an embedded world model, our approach simultaneously predicts state transitions and evaluates intervention utility. The framework is underpinned by rigorous theoretical analysis drawn from convergence theory, optimal control, and Bayesian inference, which collectively establish conditions for convergence, stability, and near-optimal performance. Furthermore, we present a hybrid implementation that combines the strengths of neural networks with symbolic reasoning modules, thereby enhancing interpretability and robustness. Finally, we demonstrate the potential of AUKAI through a detailed application in robotic navigation and obstacle avoidance, and we outline comprehensive experimental plans to validate its effectiveness in both simulated and real-world environments. 

---
# Enhancing Monocular 3D Scene Completion with Diffusion Model 

**Authors**: Changlin Song, Jiaqi Wang, Liyun Zhu, He Weng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00726)  

**Abstract**: 3D scene reconstruction is essential for applications in virtual reality, robotics, and autonomous driving, enabling machines to understand and interact with complex environments. Traditional 3D Gaussian Splatting techniques rely on images captured from multiple viewpoints to achieve optimal performance, but this dependence limits their use in scenarios where only a single image is available. In this work, we introduce FlashDreamer, a novel approach for reconstructing a complete 3D scene from a single image, significantly reducing the need for multi-view inputs. Our approach leverages a pre-trained vision-language model to generate descriptive prompts for the scene, guiding a diffusion model to produce images from various perspectives, which are then fused to form a cohesive 3D reconstruction. Extensive experiments show that our method effectively and robustly expands single-image inputs into a comprehensive 3D scene, extending monocular 3D reconstruction capabilities without further training. Our code is available this https URL. 

---
# LLMDR: LLM-Driven Deadlock Detection and Resolution in Multi-Agent Pathfinding 

**Authors**: Seungbae Seo, Junghwan Kim, Minjeong Shin, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00717)  

**Abstract**: Multi-Agent Pathfinding (MAPF) is a core challenge in multi-agent systems. Existing learning-based MAPF methods often struggle with scalability, particularly when addressing complex scenarios that are prone to deadlocks. To address these challenges, we introduce LLMDR (LLM-Driven Deadlock Detection and Resolution), an approach designed to resolve deadlocks and improve the performance of learnt MAPF models. LLMDR integrates the inference capabilities of large language models (LLMs) with learnt MAPF models and prioritized planning, enabling it to detect deadlocks and provide customized resolution strategies. We evaluate LLMDR on standard MAPF benchmark maps with varying agent numbers, measuring its performance when combined with several base models. The results demonstrate that LLMDR improves the performance of learnt MAPF models, particularly in deadlock-prone scenarios, with notable improvements in success rates. These findings show the potential of integrating LLMs to improve the scalability of learning-based MAPF methods.
The source code for LLMDR is available at: this https URL 

---
# Speculative Ad-hoc Querying 

**Authors**: Haoyu Li, Srikanth Kandula, Maria Angels de Luis Balaguer, Aditya Akella, Venkat Arun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00714)  

**Abstract**: Analyzing large datasets requires responsive query execution, but executing SQL queries on massive datasets can be slow. This paper explores whether query execution can begin even before the user has finished typing, allowing results to appear almost instantly. We propose SpeQL, a system that leverages Large Language Models (LLMs) to predict likely queries based on the database schema, the user's past queries, and their incomplete query. Since exact query prediction is infeasible, SpeQL speculates on partial queries in two ways: 1) it predicts the query structure to compile and plan queries in advance, and 2) it precomputes smaller temporary tables that are much smaller than the original database, but are still predicted to contain all information necessary to answer the user's final query. Additionally, SpeQL continuously displays results for speculated queries and subqueries in real time, aiding exploratory analysis. A utility/user study showed that SpeQL improved task completion time, and participants reported that its speculative display of results helped them discover patterns in the data more quickly. In the study, SpeQL improves user's query latency by up to $289\times$ and kept the overhead reasonable, at $\$4$ per hour. 

---
# OpenECG: Benchmarking ECG Foundation Models with Public 1.2 Million Records 

**Authors**: Zhijiang Wan, Qianhao Yu, Jia Mao, Wenfeng Duan, Cheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.00711)  

**Abstract**: This study introduces OpenECG, a large-scale benchmark of 1.2 million 12-lead ECG recordings from nine centers, to evaluate ECG foundation models (ECG-FMs) trained on public datasets. We investigate three self-supervised learning methods (SimCLR, BYOL, MAE) with ResNet-50 and Vision Transformer architectures, assessing model generalization through leave-one-dataset-out experiments and data scaling analysis. Results show that pre-training on diverse datasets significantly improves generalization, with BYOL and MAE outperforming SimCLR, highlighting the efficacy of feature-consistency and generative learning over contrastive approaches. Data scaling experiments reveal that performance saturates at 60-70% of total data for BYOL and MAE, while SimCLR requires more data. These findings demonstrate that publicly available ECG data can match or surpass proprietary datasets in training robust ECG-FMs, paving the way for scalable, clinically meaningful AI-driven ECG analysis. 

---
# Parameter Expanded Stochastic Gradient Markov Chain Monte Carlo 

**Authors**: Hyunsu Kim, Giung Nam, Chulhee Yun, Hongseok Yang, Juho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.00699)  

**Abstract**: Bayesian Neural Networks (BNNs) provide a promising framework for modeling predictive uncertainty and enhancing out-of-distribution robustness (OOD) by estimating the posterior distribution of network parameters. Stochastic Gradient Markov Chain Monte Carlo (SGMCMC) is one of the most powerful methods for scalable posterior sampling in BNNs, achieving efficiency by combining stochastic gradient descent with second-order Langevin dynamics. However, SGMCMC often suffers from limited sample diversity in practice, which affects uncertainty estimation and model performance. We propose a simple yet effective approach to enhance sample diversity in SGMCMC without the need for tempering or running multiple chains. Our approach reparameterizes the neural network by decomposing each of its weight matrices into a product of matrices, resulting in a sampling trajectory that better explores the target parameter space. This approach produces a more diverse set of samples, allowing faster mixing within the same computational budget. Notably, our sampler achieves these improvements without increasing the inference cost compared to the standard SGMCMC. Extensive experiments on image classification tasks, including OOD robustness, diversity, loss surface analyses, and a comparative study with Hamiltonian Monte Carlo, demonstrate the superiority of the proposed approach. 

---
# CREATE-FFPE: Cross-Resolution Compensated and Multi-Frequency Enhanced FS-to-FFPE Stain Transfer for Intraoperative IHC Images 

**Authors**: Yiyang Lin, Danling Jiang, Xinyu Liu, Yun Miao, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00697)  

**Abstract**: In the immunohistochemical (IHC) analysis during surgery, frozen-section (FS) images are used to determine the benignity or malignancy of the tumor. However, FS image faces problems such as image contamination and poor nuclear detail, which may disturb the pathologist's diagnosis. In contrast, formalin-fixed and paraffin-embedded (FFPE) image has a higher staining quality, but it requires quite a long time to prepare and thus is not feasible during surgery. To help pathologists observe IHC images with high quality in surgery, this paper proposes a Cross-REsolution compensATed and multi-frequency Enhanced FS-to-FFPE (CREATE-FFPE) stain transfer framework, which is the first FS-to-FFPE method for the intraoperative IHC images. To solve the slide contamination and poor nuclear detail mentioned above, we propose the cross-resolution compensation module (CRCM) and the wavelet detail guidance module (WDGM). Specifically, CRCM compensates for information loss due to contamination by providing more tissue information across multiple resolutions, while WDGM produces the desirable details in a wavelet way, and the details can be used to guide the stain transfer to be more precise. Experiments show our method can beat all the competing methods on our dataset. In addition, the FID has decreased by 44.4%, and KID*100 has decreased by 71.2% by adding the proposed CRCM and WDGM in ablation studies, and the performance of a downstream microsatellite instability prediction task with public dataset can be greatly improved by performing our FS-to-FFPE stain transfer. 

---
# How Diversely Can Language Models Solve Problems? Exploring the Algorithmic Diversity of Model-Generated Code 

**Authors**: Seonghyeon Lee, Heejae Chon, Joonwon Jang, Dongha Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00691)  

**Abstract**: Language models (LMs) have exhibited impressive abilities in generating code from natural language requirements. In this work, we highlight the diversity of code generated by LMs as a critical criterion for evaluating their code generation capabilities. There is a lack of studies focused on assessing the diversity of generated code, which overlooks its importance in code LMs. Therefore, we propose a systematic approach to evaluate code diversity, introducing various metrics with inter-code similarity. Specifically, we introduce code clustering methods that leverages LMs' capabilities in code understanding and reasoning, resulting in a set of metrics that represent the number of algorithms in model-generated solutions. We extensively investigate the property of model-generated solutions by contrasting them with human-written ones and quantifying the impact of various factors on code diversity: model size, temperature, instruction tuning, and problem complexity. Our analysis demonstrates that model-generated solutions exhibit low algorithmic diversity, which was neglected by the research community. Moreover, we explore methods to increase code diversity by combining solutions from different models and increasing sampling temperatures. Our findings highlight that code diversity can be enhanced with the help of heterogeneous models and setting temperature beyond 1.0 that has not been fully explored due to the functional correctness degradation. To facilitate our research direction, we publicly share our code and datasets through open-source repositories. 

---
# GPIoT: Tailoring Small Language Models for IoT Program Synthesis and Development 

**Authors**: Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, Yuanqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00686)  

**Abstract**: Code Large Language Models (LLMs) enhance software development efficiency by automatically generating code and documentation in response to user requirements. However, code LLMs cannot synthesize specialized programs when tasked with IoT applications that require domain knowledge. While Retrieval-Augmented Generation (RAG) offers a promising solution by fetching relevant domain knowledge, it necessitates powerful cloud LLMs (e.g., GPT-4) to process user requirements and retrieved contents, which raises significant privacy concerns. This approach also suffers from unstable networks and prohibitive LLM query costs. Moreover, it is challenging to ensure the correctness and relevance of the fetched contents. To address these issues, we propose GPIoT, a code generation system for IoT applications by fine-tuning locally deployable Small Language Models (SLMs) on IoT-specialized datasets. SLMs have smaller model sizes, allowing efficient local deployment and execution to mitigate privacy concerns and network uncertainty. Furthermore, by fine-tuning the SLMs with our IoT-specialized datasets, the SLMs' ability to synthesize IoT-related programs can be substantially improved. To evaluate GPIoT's capability in synthesizing programs for IoT applications, we develop a benchmark, IoTBench. Extensive experiments and user trials demonstrate the effectiveness of GPIoT in generating IoT-specialized code, outperforming state-of-the-art code LLMs with an average task accuracy increment of 64.7% and significant improvements in user satisfaction. 

---
# Factorized Deep Q-Network for Cooperative Multi-Agent Reinforcement Learning in Victim Tagging 

**Authors**: Maria Ana Cardei, Afsaneh Doryab  

**Link**: [PDF](https://arxiv.org/pdf/2503.00684)  

**Abstract**: Mass casualty incidents (MCIs) are a growing concern, characterized by complexity and uncertainty that demand adaptive decision-making strategies. The victim tagging step in the emergency medical response must be completed quickly and is crucial for providing information to guide subsequent time-constrained response actions. In this paper, we present a mathematical formulation of multi-agent victim tagging to minimize the time it takes for responders to tag all victims. Five distributed heuristics are formulated and evaluated with simulation experiments. The heuristics considered are on-the go, practical solutions that represent varying levels of situational uncertainty in the form of global or local communication capabilities, showcasing practical constraints. We further investigate the performance of a multi-agent reinforcement learning (MARL) strategy, factorized deep Q-network (FDQN), to minimize victim tagging time as compared to baseline heuristics. Extensive simulations demonstrate that between the heuristics, methods with local communication are more efficient for adaptive victim tagging, specifically choosing the nearest victim with the option to replan. Analyzing all experiments, we find that our FDQN approach outperforms heuristics in smaller-scale scenarios, while heuristics excel in more complex scenarios. Our experiments contain diverse complexities that explore the upper limits of MARL capabilities for real-world applications and reveal key insights. 

---
# OrdRankBen: A Novel Ranking Benchmark for Ordinal Relevance in NLP 

**Authors**: Yan Wang, Lingfei Qian, Xueqing Peng, Jimin Huang, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00674)  

**Abstract**: The evaluation of ranking tasks remains a significant challenge in natural language processing (NLP), particularly due to the lack of direct labels for results in real-world scenarios. Benchmark datasets play a crucial role in providing standardized testbeds that ensure fair comparisons, enhance reproducibility, and enable progress tracking, facilitating rigorous assessment and continuous improvement of ranking models. Existing NLP ranking benchmarks typically use binary relevance labels or continuous relevance scores, neglecting ordinal relevance scores. However, binary labels oversimplify relevance distinctions, while continuous scores lack a clear ordinal structure, making it challenging to capture nuanced ranking differences effectively. To address these challenges, we introduce OrdRankBen, a novel benchmark designed to capture multi-granularity relevance distinctions. Unlike conventional benchmarks, OrdRankBen incorporates structured ordinal labels, enabling more precise ranking evaluations. Given the absence of suitable datasets for ordinal relevance ranking in NLP, we constructed two datasets with distinct ordinal label distributions. We further evaluate various models for three model types, ranking-based language models, general large language models, and ranking-focused large language models on these datasets. Experimental results show that ordinal relevance modeling provides a more precise evaluation of ranking models, improving their ability to distinguish multi-granularity differences among ranked items-crucial for tasks that demand fine-grained relevance differentiation. 

---
# Generative Artificial Intelligence for Academic Research: Evidence from Guidance Issued for Researchers by Higher Education Institutions in the United States 

**Authors**: Amrita Ganguly, Aditya Johri, Areej Ali, Nora McDonald  

**Link**: [PDF](https://arxiv.org/pdf/2503.00664)  

**Abstract**: The recent development and use of generative AI (GenAI) has signaled a significant shift in research activities such as brainstorming, proposal writing, dissemination, and even reviewing. This has raised questions about how to balance the seemingly productive uses of GenAI with ethical concerns such as authorship and copyright issues, use of biased training data, lack of transparency, and impact on user privacy. To address these concerns, many Higher Education Institutions (HEIs) have released institutional guidance for researchers. To better understand the guidance that is being provided we report findings from a thematic analysis of guidelines from thirty HEIs in the United States that are classified as R1 or 'very high research activity.' We found that guidance provided to researchers: (1) asks them to refer to external sources of information such as funding agencies and publishers to keep updated and use institutional resources for training and education; (2) asks them to understand and learn about specific GenAI attributes that shape research such as predictive modeling, knowledge cutoff date, data provenance, and model limitations, and educate themselves about ethical concerns such as authorship, attribution, privacy, and intellectual property issues; and (3) includes instructions on how to acknowledge sources and disclose the use of GenAI, how to communicate effectively about their GenAI use, and alerts researchers to long term implications such as over reliance on GenAI, legal consequences, and risks to their institutions from GenAI use. Overall, guidance places the onus of compliance on individual researchers making them accountable for any lapses, thereby increasing their responsibility. 

---
# Deep Change Monitoring: A Hyperbolic Representative Learning Framework and a Dataset for Long-term Fine-grained Tree Change Detection 

**Authors**: Yante Li, Hanwen Qi, Haoyu Chen, Xinlian Liang, Guoying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00643)  

**Abstract**: In environmental protection, tree monitoring plays an essential role in maintaining and improving ecosystem health. However, precise monitoring is challenging because existing datasets fail to capture continuous fine-grained changes in trees due to low-resolution images and high acquisition costs. In this paper, we introduce UAVTC, a large-scale, long-term, high-resolution dataset collected using UAVs equipped with cameras, specifically designed to detect individual Tree Changes (TCs). UAVTC includes rich annotations and statistics based on biological knowledge, offering a fine-grained view for tree monitoring. To address environmental influences and effectively model the hierarchical diversity of physiological TCs, we propose a novel Hyperbolic Siamese Network (HSN) for TC detection, enabling compact and hierarchical representations of dynamic tree changes.
Extensive experiments show that HSN can effectively capture complex hierarchical changes and provide a robust solution for fine-grained TC detection. In addition, HSN generalizes well to cross-domain face anti-spoofing task, highlighting its broader significance in AI. We believe our work, combining ecological insights and interdisciplinary expertise, will benefit the community by offering a new benchmark and innovative AI technologies. 

---
# Efficiently Editing Mixture-of-Experts Models with Compressed Experts 

**Authors**: Yifei He, Yang Liu, Chen Liang, Hany Hassan Awadalla  

**Link**: [PDF](https://arxiv.org/pdf/2503.00634)  

**Abstract**: Mixture-of-Experts (MoE) models have become a key approach for scaling large language models efficiently by activating only a subset of experts during training and inference. Typically, the number of activated experts presents a trade-off: fewer experts reduce computational costs, while more experts improve performance. Recent studies reveal that not all activated experts contribute equally to model performance, with some providing minimal utility, particularly when finetuning pretrained MoE models for specialized downstream tasks. The co-existence of significant and redundant parameters in experts provides us an opportunity to reduce the number of activated experts while maintaining model performance. In this work, we propose the concept of compressed experts, lightweight modules that serve as compact representations of full experts. Our approach preserves the most important experts while replacing other auxiliary activated experts with compressed experts. The reduction of active parameters significantly lowers inference costs while achieving comparable performance. Extensive experiments on models including Phi-MoE and OLMoE demonstrate that compressed experts recover over 90% of full expert performance across various tasks while reducing more than 30% active parameters and saving 20% in inference costs. This approach enables efficient deployment of MoE models in resource-constrained settings and facilitates scaling to larger models with manageable overhead. Our code is available at this https URL. 

---
# An evaluation of DeepSeek Models in Biomedical Natural Language Processing 

**Authors**: Zaifu Zhan, Shuang Zhou, Huixue Zhou, Jiawen Deng, Yu Hou, Jeremy Yeung, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00624)  

**Abstract**: The advancement of Large Language Models (LLMs) has significantly impacted biomedical Natural Language Processing (NLP), enhancing tasks such as named entity recognition, relation extraction, event extraction, and text classification. In this context, the DeepSeek series of models have shown promising potential in general NLP tasks, yet their capabilities in the biomedical domain remain underexplored. This study evaluates multiple DeepSeek models (Distilled-DeepSeek-R1 series and Deepseek-LLMs) across four key biomedical NLP tasks using 12 datasets, benchmarking them against state-of-the-art alternatives (Llama3-8B, Qwen2.5-7B, Mistral-7B, Phi-4-14B, Gemma-2-9B). Our results reveal that while DeepSeek models perform competitively in named entity recognition and text classification, challenges persist in event and relation extraction due to precision-recall trade-offs. We provide task-specific model recommendations and highlight future research directions. This evaluation underscores the strengths and limitations of DeepSeek models in biomedical NLP, guiding their future deployment and optimization. 

---
# PinLanding: Content-First Keyword Landing Page Generation via Multi-Modal AI for Web-Scale Discovery 

**Authors**: Faye Zhang, Jasmine Wan, Qianyu Cheng, Jinfeng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00619)  

**Abstract**: Online platforms like Pinterest hosting vast content collections traditionally rely on manual curation or user-generated search logs to create keyword landing pages (KLPs) -- topic-centered collection pages that serve as entry points for content discovery. While manual curation ensures quality, it doesn't scale to millions of collections, and search log approaches result in limited topic coverage and imprecise content matching. In this paper, we present PinLanding, a novel content-first architecture that transforms the way platforms create topical collections. Instead of deriving topics from user behavior, our system employs a multi-stage pipeline combining vision-language model (VLM) for attribute extraction, large language model (LLM) for topic generation, and a CLIP-based dual-encoder architecture for precise content matching. Our model achieves 99.7% Recall@10 on Fashion200K benchmark, demonstrating strong attribute understanding capabilities. In production deployment for search engine optimization with 4.2 million shopping landing pages, the system achieves a 4X increase in topic coverage and 14.29% improvement in collection attribute precision over the traditional search log-based approach via human evaluation. The architecture can be generalized beyond search traffic to power various user experiences, including content discovery and recommendations, providing a scalable solution to transform unstructured content into curated topical collections across any content domain. 

---
# Urban Safety Perception Through the Lens of Large Multimodal Models: A Persona-based Approach 

**Authors**: Ciro Beneduce, Bruno Lepri, Massimiliano Luca  

**Link**: [PDF](https://arxiv.org/pdf/2503.00610)  

**Abstract**: Understanding how urban environments are perceived in terms of safety is crucial for urban planning and policymaking. Traditional methods like surveys are limited by high cost, required time, and scalability issues. To overcome these challenges, this study introduces Large Multimodal Models (LMMs), specifically Llava 1.6 7B, as a novel approach to assess safety perceptions of urban spaces using street-view images. In addition, the research investigated how this task is affected by different socio-demographic perspectives, simulated by the model through Persona-based prompts. Without additional fine-tuning, the model achieved an average F1-score of 59.21% in classifying urban scenarios as safe or unsafe, identifying three key drivers of perceived unsafety: isolation, physical decay, and urban infrastructural challenges. Moreover, incorporating Persona-based prompts revealed significant variations in safety perceptions across the socio-demographic groups of age, gender, and nationality. Elder and female Personas consistently perceive higher levels of unsafety than younger or male Personas. Similarly, nationality-specific differences were evident in the proportion of unsafe classifications ranging from 19.71% in Singapore to 40.15% in Botswana. Notably, the model's default configuration aligned most closely with a middle-aged, male Persona. These findings highlight the potential of LMMs as a scalable and cost-effective alternative to traditional methods for urban safety perceptions. While the sensitivity of these models to socio-demographic factors underscores the need for thoughtful deployment, their ability to provide nuanced perspectives makes them a promising tool for AI-driven urban planning. 

---
# Semantic Integrity Constraints: Declarative Guardrails for AI-Augmented Data Processing Systems 

**Authors**: Alexander W. Lee, Justin Chan, Michael Fu, Nicolas Kim, Akshay Mehta, Deepti Raghavan, Ugur Cetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2503.00600)  

**Abstract**: The emergence of AI-augmented Data Processing Systems (DPSs) has introduced powerful semantic operators that extend traditional data management capabilities with LLM-based processing. However, these systems face fundamental reliability (a.k.a. trust) challenges, as LLMs can generate erroneous outputs, limiting their adoption in critical domains. Existing approaches to LLM constraints--ranging from user-defined functions to constrained decoding--are fragmented, imperative, and lack semantics-aware integration into query execution. To address this gap, we introduce Semantic Integrity Constraints (SICs), a novel declarative abstraction that extends traditional database integrity constraints to govern and optimize semantic operators within DPSs. SICs integrate seamlessly into the relational model, allowing users to specify common classes of constraints (e.g., grounding and soundness) while enabling query-aware enforcement and optimization strategies.
In this paper, we present the core design of SICs, describe their formal integration into query execution, and detail our conception of grounding constraints, a key SIC class that ensures factual consistency of generated outputs. In addition, we explore novel enforcement mechanisms, combining proactive (constrained decoding) and reactive (validation and recovery) techniques to optimize efficiency and reliability. Our work establishes SICs as a foundational framework for trustworthy, high-performance AI-augmented data processing, paving the way for future research in constraint-driven optimizations, adaptive enforcement, and enterprise-scale deployments. 

---
# Zero-Shot Keyphrase Generation: Investigating Specialized Instructions and Multi-Sample Aggregation on Large Language Models 

**Authors**: Jayanth Mohan, Jishnu Ray Chowdhury, Tomas Malik, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00597)  

**Abstract**: Keyphrases are the essential topical phrases that summarize a document. Keyphrase generation is a long-standing NLP task for automatically generating keyphrases for a given document. While the task has been comprehensively explored in the past via various models, only a few works perform some preliminary analysis of Large Language Models (LLMs) for the task. Given the impact of LLMs in the field of NLP, it is important to conduct a more thorough examination of their potential for keyphrase generation. In this paper, we attempt to meet this demand with our research agenda. Specifically, we focus on the zero-shot capabilities of open-source instruction-tuned LLMs (Phi-3, Llama-3) and the closed-source GPT-4o for this task. We systematically investigate the effect of providing task-relevant specialized instructions in the prompt. Moreover, we design task-specific counterparts to self-consistency-style strategies for LLMs and show significant benefits from our proposals over the baselines. 

---
# BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge 

**Authors**: Terry Tong, Fei Wang, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00596)  

**Abstract**: This paper proposes a novel backdoor threat attacking the LLM-as-a-Judge evaluation regime, where the adversary controls both the candidate and evaluator model. The backdoored evaluator victimizes benign users by unfairly assigning inflated scores to adversary. A trivial single token backdoor poisoning 1% of the evaluator training data triples the adversary's score with respect to their legitimate score. We systematically categorize levels of data access corresponding to three real-world settings, (1) web poisoning, (2) malicious annotator, and (3) weight poisoning. These regimes reflect a weak to strong escalation of data access that highly correlates with attack severity. Under the weakest assumptions - web poisoning (1), the adversary still induces a 20% score inflation. Likewise, in the (3) weight poisoning regime, the stronger assumptions enable the adversary to inflate their scores from 1.5/5 to 4.9/5. The backdoor threat generalizes across different evaluator architectures, trigger designs, evaluation tasks, and poisoning rates. By poisoning 10% of the evaluator training data, we control toxicity judges (Guardrails) to misclassify toxic prompts as non-toxic 89% of the time, and document reranker judges in RAG to rank the poisoned document first 97% of the time. LLM-as-a-Judge is uniquely positioned at the intersection of ethics and technology, where social implications of mislead model selection and evaluation constrain the available defensive tools. Amidst these challenges, model merging emerges as a principled tool to offset the backdoor, reducing ASR to near 0% whilst maintaining SOTA performance. Model merging's low computational cost and convenient integration into the current LLM Judge training pipeline position it as a promising avenue for backdoor mitigation in the LLM-as-a-Judge setting. 

---
# Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning 

**Authors**: Jingtao Tang, Zining Mao, Lufan Yang, Hang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.00583)  

**Abstract**: We address the Multi-Robot Motion Planning (MRMP) problem of computing collision-free trajectories for multiple robots in shared continuous environments. While existing frameworks effectively decompose MRMP into single-robot subproblems, spatiotemporal motion planning with dynamic obstacles remains challenging, particularly in cluttered or narrow-corridor settings. We propose Space-Time Graphs of Convex Sets (ST-GCS), a novel planner that systematically covers the collision-free space-time domain with convex sets instead of relying on random sampling. By extending Graphs of Convex Sets (GCS) into the time dimension, ST-GCS formulates time-optimal trajectories in a unified convex optimization that naturally accommodates velocity bounds and flexible arrival times. We also propose Exact Convex Decomposition (ECD) to "reserve" trajectories as spatiotemporal obstacles, maintaining a collision-free space-time graph of convex sets for subsequent planning. Integrated into two prioritized-planning frameworks, ST-GCS consistently achieves higher success rates and better solution quality than state-of-the-art sampling-based planners -- often at orders-of-magnitude faster runtimes -- underscoring its benefits for MRMP in challenging settings. 

---
# Brain Foundation Models: A Survey on Advancements in Neural Signal Processing and Brain Discovery 

**Authors**: Xinliang Zhou, Chenyu Liu, Zhisheng Chen, Kun Wang, Yi Ding, Ziyu Jia, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00580)  

**Abstract**: Brain foundation models (BFMs) have emerged as a transformative paradigm in computational neuroscience, offering a revolutionary framework for processing diverse neural signals across different brain-related tasks. These models leverage large-scale pre-training techniques, allowing them to generalize effectively across multiple scenarios, tasks, and modalities, thus overcoming the traditional limitations faced by conventional artificial intelligence (AI) approaches in understanding complex brain data. By tapping into the power of pretrained models, BFMs provide a means to process neural data in a more unified manner, enabling advanced analysis and discovery in the field of neuroscience. In this survey, we define BFMs for the first time, providing a clear and concise framework for constructing and utilizing these models in various applications. We also examine the key principles and methodologies for developing these models, shedding light on how they transform the landscape of neural signal processing. This survey presents a comprehensive review of the latest advancements in BFMs, covering the most recent methodological innovations, novel views of application areas, and challenges in the field. Notably, we highlight the future directions and key challenges that need to be addressed to fully realize the potential of BFMs. These challenges include improving the quality of brain data, optimizing model architecture for better generalization, increasing training efficiency, and enhancing the interpretability and robustness of BFMs in real-world applications. 

---
# LoR2C : Low-Rank Residual Connection Adaptation for Parameter-Efficient Fine-Tuning 

**Authors**: Jiancheng Zhao, Xingda Yu, Yuxiang Zhang, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00572)  

**Abstract**: In recent years, pretrained large language models have demonstrated outstanding performance across various natural language processing tasks. However, full-parameter fine-tuning methods require adjusting all model parameters, leading to immense computational resource demands. Although parameter-efficient fine-tuning methods like LoRA have significantly reduced the number of parameters, they still face challenges such as gradient vanishing and the potential for further parameter reduction. To address these issues, this paper proposes a novel parameter-efficient fine-tuning method called LoR2C (Low-Rank Residual Connection Adaptation). LoR2C introduces residual connections with low-rank matrices within the model layers, which not only reduces the number of fine-tuning parameters but also effectively alleviates the gradient vanishing problem. Additionally, this paper presents three optimization variants of LoR2C: ShareLoR2C, MergeLoR2C, and InjectLoR2C. These variants further improve parameter efficiency and model performance through parameter sharing, module merging, and injection mechanisms, respectively. Experimental results on multiple natural language understanding and natural language generation tasks demonstrate that LoR2C and its optimized variants significantly reduce parameter overhead while maintaining or even improving performance, outperforming existing mainstream parameter-efficient fine-tuning this http URL code is publicly available at this https URL. 

---
# A Guide to Failure in Machine Learning: Reliability and Robustness from Foundations to Practice 

**Authors**: Eric Heim, Oren Wright, David Shriver  

**Link**: [PDF](https://arxiv.org/pdf/2503.00563)  

**Abstract**: One of the main barriers to adoption of Machine Learning (ML) is that ML models can fail unexpectedly. In this work, we aim to provide practitioners a guide to better understand why ML models fail and equip them with techniques they can use to reason about failure. Specifically, we discuss failure as either being caused by lack of reliability or lack of robustness. Differentiating the causes of failure in this way allows us to formally define why models fail from first principles and tie these definitions to engineering concepts and real-world deployment settings. Throughout the document we provide 1) a summary of important theoretic concepts in reliability and robustness, 2) a sampling current techniques that practitioners can utilize to reason about ML model reliability and robustness, and 3) examples that show how these concepts and techniques can apply to real-world settings. 

---
# Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable 

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00555)  

**Abstract**: Safety alignment is an important procedure before the official deployment of a Large Language Model (LLM). While safety alignment has been extensively studied for LLM, there is still a large research gap for Large Reasoning Models (LRMs) that equip with improved reasoning capability. We in this paper systematically examine a simplified pipeline for producing safety aligned LRMs. With our evaluation of various LRMs, we deliver two main findings: i) Safety alignment can be done upon the LRM to restore its safety capability. ii) Safety alignment leads to a degradation of the reasoning capability of LRMs. The two findings show that there exists a trade-off between reasoning and safety capability with the sequential LRM production pipeline. The discovered trade-off, which we name Safety Tax, should shed light on future endeavors of safety research on LRMs. As a by-product, we curate a dataset called DirectRefusal, which might serve as an alternative dataset for safety alignment. Our source code is available at this https URL. 

---
# Distributionally Robust Reinforcement Learning with Human Feedback 

**Authors**: Debmalya Mandal, Paulius Sasnauskas, Goran Radanovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.00539)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has evolved to be one of the main methods for fine-tuning large language models (LLMs). However, existing RLHF methods are non-robust, and their performance deteriorates if the downstream task differs significantly from the preference dataset used in fine-tuning. In order to mitigate this problem, we introduce a distributionally robust RLHF for fine-tuning LLMs. In particular, our goal is to ensure that a fine-tuned model retains its performance even when the distribution of prompts significantly differs from the distribution encountered during fine-tuning. We formulate distributionally robust optimization (DRO) version of two popular fine-tuning methods -- (1) reward-based RLHF and (2) reward-free DPO (direct preference optimization). We propose a minibatch gradient descent based algorithms for both of them, and theoretically prove convergence guarantees for the algorithms. Subsequently, we evaluate our algorithms on an out-of-distribution (OOD) task by first training the model on the Unified-Feedback dataset and evaluating its performance on two different datasets. The experimental results show that our robust training improves the accuracy of the learned reward models on average, and markedly on some tasks, such as reasoning. Furthermore, we show that the robust versions of policy optimization methods, similarly improve performance on OOD tasks. 

---
# What Makes a Good Diffusion Planner for Decision Making? 

**Authors**: Haofei Lu, Dongqi Han, Yifei Shen, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00535)  

**Abstract**: Diffusion models have recently shown significant potential in solving decision-making problems, particularly in generating behavior plans -- also known as diffusion planning. While numerous studies have demonstrated the impressive performance of diffusion planning, the mechanisms behind the key components of a good diffusion planner remain unclear and the design choices are highly inconsistent in existing studies. In this work, we address this issue through systematic empirical experiments on diffusion planning in an offline reinforcement learning (RL) setting, providing practical insights into the essential components of diffusion planning. We trained and evaluated over 6,000 diffusion models, identifying the critical components such as guided sampling, network architecture, action generation and planning strategy. We revealed that some design choices opposite to the common practice in previous work in diffusion planning actually lead to better performance, e.g., unconditional sampling with selection can be better than guided sampling and Transformer outperforms U-Net as denoising network. Based on these insights, we suggest a simple yet strong diffusion planning baseline that achieves state-of-the-art results on standard offline RL benchmarks. 

---
# Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions 

**Authors**: Guanwen Xie, Jingzehua Xu, Yimian Ding, Zhi Zhang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00527)  

**Abstract**: The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers. 

---
# End-To-End Learning of Gaussian Mixture Priors for Diffusion Sampler 

**Authors**: Denis Blessing, Xiaogang Jia, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2503.00524)  

**Abstract**: Diffusion models optimized via variational inference (VI) have emerged as a promising tool for generating samples from unnormalized target densities. These models create samples by simulating a stochastic differential equation, starting from a simple, tractable prior, typically a Gaussian distribution. However, when the support of this prior differs greatly from that of the target distribution, diffusion models often struggle to explore effectively or suffer from large discretization errors. Moreover, learning the prior distribution can lead to mode-collapse, exacerbated by the mode-seeking nature of reverse Kullback-Leibler divergence commonly used in VI. To address these challenges, we propose end-to-end learnable Gaussian mixture priors (GMPs). GMPs offer improved control over exploration, adaptability to target support, and increased expressiveness to counteract mode collapse. We further leverage the structure of mixture models by proposing a strategy to iteratively refine the model by adding mixture components during training. Our experimental results demonstrate significant performance improvements across a diverse range of real-world and synthetic benchmark problems when using GMPs without requiring additional target evaluations. 

---
# Functional multi-armed bandit and the best function identification problems 

**Authors**: Yuriy Dorn, Aleksandr Katrutsa, Ilgam Latypov, Anastasiia Soboleva  

**Link**: [PDF](https://arxiv.org/pdf/2503.00509)  

**Abstract**: Bandit optimization usually refers to the class of online optimization problems with limited feedback, namely, a decision maker uses only the objective value at the current point to make a new decision and does not have access to the gradient of the objective function. While this name accurately captures the limitation in feedback, it is somehow misleading since it does not have any connection with the multi-armed bandits (MAB) problem class. We propose two new classes of problems: the functional multi-armed bandit problem (FMAB) and the best function identification problem. They are modifications of a multi-armed bandit problem and the best arm identification problem, respectively, where each arm represents an unknown black-box function. These problem classes are a surprisingly good fit for modeling real-world problems such as competitive LLM training. To solve the problems from these classes, we propose a new reduction scheme to construct UCB-type algorithms, namely, the F-LCB algorithm, based on algorithms for nonlinear optimization with known convergence rates. We provide the regret upper bounds for this reduction scheme based on the base algorithms' convergence rates. We add numerical experiments that demonstrate the performance of the proposed scheme. 

---
# Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture 

**Authors**: Xuanchen Li, Jianyu Wang, Yuhao Cheng, Yikun Zeng, Xingyu Ren, Wenhan Zhu, Weiming Zhao, Yichao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00495)  

**Abstract**: Significant progress has been made for speech-driven 3D face animation, but most works focus on learning the motion of mesh/geometry, ignoring the impact of dynamic texture. In this work, we reveal that dynamic texture plays a key role in rendering high-fidelity talking avatars, and introduce a high-resolution 4D dataset \textbf{TexTalk4D}, consisting of 100 minutes of audio-synced scan-level meshes with detailed 8K dynamic textures from 100 subjects. Based on the dataset, we explore the inherent correlation between motion and texture, and propose a diffusion-based framework \textbf{TexTalker} to simultaneously generate facial motions and dynamic textures from speech. Furthermore, we propose a novel pivot-based style injection strategy to capture the complicity of different texture and motion styles, which allows disentangled control. TexTalker, as the first method to generate audio-synced facial motion with dynamic texture, not only outperforms the prior arts in synthesising facial motions, but also produces realistic textures that are consistent with the underlying facial movements. Project page: this https URL. 

---
# LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement 

**Authors**: Boyi Kang, Xinfa Zhu, Zihan Zhang, Zhen Ye, Mingshuai Liu, Ziqian Wang, Yike Zhu, Guobin Ma, Jun Chen, Longshuai Xiao, Chao Weng, Wei Xue, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.00493)  

**Abstract**: Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area. 

---
# Embracing Diversity: A Multi-Perspective Approach with Soft Labels 

**Authors**: Benedetta Muscato, Praveen Bushipaka, Gizem Gezici, Lucia Passaro, Fosca Giannotti, Tommaso Cucinotta  

**Link**: [PDF](https://arxiv.org/pdf/2503.00489)  

**Abstract**: Prior studies show that adopting the annotation diversity shaped by different backgrounds and life experiences and incorporating them into the model learning, i.e. multi-perspective approach, contribute to the development of more responsible models. Thus, in this paper we propose a new framework for designing and further evaluating perspective-aware models on stance detection task,in which multiple annotators assign stances based on a controversial topic. We also share a new dataset established through obtaining both human and LLM annotations. Results show that the multi-perspective approach yields better classification performance (higher F1-scores), outperforming the traditional approaches that use a single ground-truth, while displaying lower model confidence scores, probably due to the high level of subjectivity of the stance detection task. 

---
# Interacting with AI Reasoning Models: Harnessing "Thoughts" for AI-Driven Software Engineering 

**Authors**: Christoph Treude, Raula Gaikovina Kula  

**Link**: [PDF](https://arxiv.org/pdf/2503.00483)  

**Abstract**: Recent advances in AI reasoning models provide unprecedented transparency into their decision-making processes, transforming them from traditional black-box systems into models that articulate step-by-step chains of thought rather than producing opaque outputs. This shift has the potential to improve software quality, explainability, and trust in AI-augmented development. However, software engineers rarely have the time or cognitive bandwidth to analyze, verify, and interpret every AI-generated thought in detail. Without an effective interface, this transparency could become a burden rather than a benefit.
In this paper, we propose a vision for structuring the interaction between AI reasoning models and software engineers to maximize trust, efficiency, and decision-making power. We argue that simply exposing AI's reasoning is not enough -- software engineers need tools and frameworks that selectively highlight critical insights, filter out noise, and facilitate rapid validation of key assumptions. To illustrate this challenge, we present motivating examples in which AI reasoning models state their assumptions when deciding which external library to use and produce divergent reasoning paths and recommendations about security vulnerabilities, highlighting the need for an interface that prioritizes actionable insights while managing uncertainty and resolving conflicts. We then outline a research roadmap for integrating automated summarization, assumption validation, and multi-model conflict resolution into software engineering workflows. Achieving this vision will unlock the full potential of AI reasoning models to enable software engineers to make faster, more informed decisions without being overwhelmed by unnecessary detail. 

---
# Challenges in Testing Large Language Model Based Software: A Faceted Taxonomy 

**Authors**: Felix Dobslaw, Robert Feldt, Juyeon Yoon, Shin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00481)  

**Abstract**: Large Language Models (LLMs) and Multi-Agent LLMs (MALLMs) introduce non-determinism unlike traditional or machine learning software, requiring new approaches to verifying correctness beyond simple output comparisons or statistical accuracy over test datasets.
This paper presents a taxonomy for LLM test case design, informed by both the research literature, our experience, and open-source tools that represent the state of practice. We identify key variation points that impact test correctness and highlight open challenges that the research, industry, and open-source communities must address as LLMs become integral to software systems.
Our taxonomy defines four facets of LLM test case design, addressing ambiguity in both inputs and outputs while establishing best practices. It distinguishes variability in goals, the system under test, and inputs, and introduces two key oracle types: atomic and aggregated. Our mapping indicates that current tools insufficiently account for these variability points, highlighting the need for closer collaboration between academia and practitioners to improve the reliability and reproducibility of LLM testing. 

---
# Leveraging Compute-in-Memory for Efficient Generative Model Inference in TPUs 

**Authors**: Zhantong Zhu, Hongou Li, Wenjie Ren, Meng Wu, Le Ye, Ru Huang, Tianyu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.00461)  

**Abstract**: With the rapid advent of generative models, efficiently deploying these models on specialized hardware has become critical. Tensor Processing Units (TPUs) are designed to accelerate AI workloads, but their high power consumption necessitates innovations for improving efficiency. Compute-in-memory (CIM) has emerged as a promising paradigm with superior area and energy efficiency. In this work, we present a TPU architecture that integrates digital CIM to replace conventional digital systolic arrays in matrix multiply units (MXUs). We first establish a CIM-based TPU architecture model and simulator to evaluate the benefits of CIM for diverse generative model inference. Building upon the observed design insights, we further explore various CIM-based TPU architectural design choices. Up to 44.2% and 33.8% performance improvement for large language model and diffusion transformer inference, and 27.3x reduction in MXU energy consumption can be achieved with different design choices, compared to the baseline TPUv4i architecture. 

---
# PodAgent: A Comprehensive Framework for Podcast Generation 

**Authors**: Yujia Xiao, Lei He, Haohan Guo, Fenglong Xie, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.00455)  

**Abstract**: Existing Existing automatic audio generation methods struggle to generate podcast-like audio programs effectively. The key challenges lie in in-depth content generation, appropriate and expressive voice production. This paper proposed PodAgent, a comprehensive framework for creating audio programs. PodAgent 1) generates informative topic-discussion content by designing a Host-Guest-Writer multi-agent collaboration system, 2) builds a voice pool for suitable voice-role matching and 3) utilizes LLM-enhanced speech synthesis method to generate expressive conversational speech. Given the absence of standardized evaluation criteria for podcast-like audio generation, we developed comprehensive assessment guidelines to effectively evaluate the model's performance. Experimental results demonstrate PodAgent's effectiveness, significantly surpassing direct GPT-4 generation in topic-discussion dialogue content, achieving an 87.4% voice-matching accuracy, and producing more expressive speech through LLM-guided synthesis. Demo page: this https URL. Source code: this https URL. 

---
# Rehearse With User: Personalized Opinion Summarization via Role-Playing based on Large Language Models 

**Authors**: Yanyue Zhang, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00449)  

**Abstract**: Personalized opinion summarization is crucial as it considers individual user interests while generating product summaries. Recent studies show that although large language models demonstrate powerful text summarization and evaluation capabilities without the need for training data, they face difficulties in personalized tasks involving long texts. To address this, \textbf{Rehearsal}, a personalized opinion summarization framework via LLMs-based role-playing is proposed. Having the model act as the user, the model can better understand the user's personalized needs. Additionally, a role-playing supervisor and practice process are introduced to improve the role-playing ability of the LLMs, leading to a better expression of user needs. Furthermore, through suggestions from virtual users, the summary generation is intervened, ensuring that the generated summary includes information of interest to the user, thus achieving personalized summary generation. Experiment results demonstrate that our method can effectively improve the level of personalization in large model-generated summaries. 

---
# HalCECE: A Framework for Explainable Hallucination Detection through Conceptual Counterfactuals in Image Captioning 

**Authors**: Maria Lymperaiou, Giorgos FIlandrianos, Angeliki Dimitriou, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00436)  

**Abstract**: In the dynamic landscape of artificial intelligence, the exploration of hallucinations within vision-language (VL) models emerges as a critical frontier. This work delves into the intricacies of hallucinatory phenomena exhibited by widely used image captioners, unraveling interesting patterns. Specifically, we step upon previously introduced techniques of conceptual counterfactual explanations to address VL hallucinations. The deterministic and efficient nature of the employed conceptual counterfactuals backbone is able to suggest semantically minimal edits driven by hierarchical knowledge, so that the transition from a hallucinated caption to a non-hallucinated one is performed in a black-box manner. HalCECE, our proposed hallucination detection framework is highly interpretable, by providing semantically meaningful edits apart from standalone numbers, while the hierarchical decomposition of hallucinated concepts leads to a thorough hallucination analysis. Another novelty tied to the current work is the investigation of role hallucinations, being one of the first works to involve interconnections between visual concepts in hallucination detection. Overall, HalCECE recommends an explainable direction to the crucial field of VL hallucination detection, thus fostering trustworthy evaluation of current and future VL systems. 

---
# Unveiling AI's Threats to Child Protection: Regulatory efforts to Criminalize AI-Generated CSAM and Emerging Children's Rights Violations 

**Authors**: Emmanouela Kokolaki, Paraskevi Fragopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00433)  

**Abstract**: This paper aims to present new alarming trends in the field of child sexual abuse through imagery, as part of SafeLine's research activities in the field of cybercrime, child sexual abuse material and the protection of children's rights to safe online experiences. It focuses primarily on the phenomenon of AI-generated CSAM, sophisticated ways employed for its production which are discussed in dark web forums and the crucial role that the open-source AI models play in the evolution of this overwhelming phenomenon. The paper's main contribution is a correlation analysis between the hotline's reports and domain names identified in dark web forums, where users' discussions focus on exchanging information specifically related to the generation of AI-CSAM. The objective was to reveal the close connection of clear net and dark web content, which was accomplished through the use of the ATLAS dataset of the Voyager system. Furthermore, through the analysis of a set of posts' content drilled from the above dataset, valuable conclusions on forum members' techniques employed for the production of AI-generated CSAM are also drawn, while users' views on this type of content and routes followed in order to overcome technological barriers set with the aim of preventing malicious purposes are also presented. As the ultimate contribution of this research, an overview of the current legislative developments in all country members of the INHOPE organization and the issues arising in the process of regulating the AI- CSAM is presented, shedding light in the legal challenges regarding the regulation and limitation of the phenomenon. 

---
# Language Model Mapping in Multimodal Music Learning: A Grand Challenge Proposal 

**Authors**: Daniel Chin, Gus Xia  

**Link**: [PDF](https://arxiv.org/pdf/2503.00427)  

**Abstract**: We have seen remarkable success in representation learning and language models (LMs) using deep neural networks. Many studies aim to build the underlying connections among different modalities via the alignment and mappings at the token or embedding level, but so far, most methods are very data-hungry, limiting their performance in domains such as music where paired data are less abundant. We argue that the embedding alignment is only at the surface level of multimodal alignment. In this paper, we propose a grand challenge of \textit{language model mapping} (LMM), i.e., how to map the essence implied in the LM of one domain to the LM of another domain under the assumption that LMs of different modalities are tracking the same underlying phenomena. We first introduce a basic setup of LMM, highlighting the goal to unveil a deeper aspect of cross-modal alignment as well as to achieve more sample-efficiency learning. We then discuss why music is an ideal domain in which to conduct LMM research. After that, we connect LMM in music with a more general and challenging scientific problem of \textit{learning to take actions based on both sensory input and abstract symbols}, and in the end, present an advanced version of the challenge problem setup. 

---
# Auto-encoding Molecules: Graph-Matching Capabilities Matter 

**Authors**: Magnus Cunow, Gerrit Großmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.00426)  

**Abstract**: Autoencoders are effective deep learning models that can function as generative models and learn latent representations for downstream tasks. The use of graph autoencoders - with both encoder and decoder implemented as message passing networks - is intriguing due to their ability to generate permutation-invariant graph representations. However, this approach faces difficulties because decoding a graph structure from a single vector is challenging, and comparing input and output graphs requires an effective permutation-invariant similarity measure. As a result, many studies rely on approximate methods.
In this work, we explore the effect of graph matching precision on the training behavior and generation capabilities of a Variational Autoencoder (VAE). Our contribution is two-fold: (1) we propose a transformer-based message passing graph decoder as an alternative to a graph neural network decoder, that is more robust and expressive by leveraging global attention mechanisms. (2) We show that the precision of graph matching has significant impact on training behavior and is essential for effective de novo (molecular) graph generation.
Code is available at this https URL 

---
# A physics-informed Bayesian optimization method for rapid development of electrical machines 

**Authors**: Pedram Asef, Christopher Vagg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00420)  

**Abstract**: Advanced slot and winding designs are imperative to create future high performance electrical machines (EM). As a result, the development of methods to design and improve slot filling factor (SFF) has attracted considerable research. Recent developments in manufacturing processes, such as additive manufacturing and alternative materials, has also highlighted a need for novel high-fidelity design techniques to develop high performance complex geometries and topologies. This study therefore introduces a novel physics-informed machine learning (PIML) design optimization process for improving SFF in traction electrical machines used in electric vehicles. A maximum entropy sampling algorithm (MESA) is used to seed a physics-informed Bayesian optimization (PIBO) algorithm, where the target function and its approximations are produced by Gaussian processes (GP)s. The proposed PIBO-MESA is coupled with a 2D finite element model (FEM) to perform a GP-based surrogate and provide the first demonstration of the optimal combination of complex design variables for an electrical machine. Significant computational gains were achieved using the new PIBO-MESA approach, which is 45% faster than existing stochastic methods, such as the non-dominated sorting genetic algorithm II (NSGA-II). The FEM results confirm that the new design optimization process and keystone shaped wires lead to a higher SFF (i.e. by 20%) and electromagnetic improvements (e.g. maximum torque by 12%) with similar resistivity. The newly developed PIBO-MESA design optimization process therefore presents significant benefits in the design of high-performance electric machines, with reduced development time and costs. 

---
# Breaking the Loop: Detecting and Mitigating Denial-of-Service Vulnerabilities in Large Language Models 

**Authors**: Junzhe Yu, Yi Liu, Huijia Sun, Ling Shi, Yuqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00416)  

**Abstract**: Large Language Models (LLMs) have significantly advanced text understanding and generation, becoming integral to applications across education, software development, healthcare, entertainment, and legal services. Despite considerable progress in improving model reliability, latency remains under-explored, particularly through recurrent generation, where models repeatedly produce similar or identical outputs, causing increased latency and potential Denial-of-Service (DoS) vulnerabilities.
We propose RecurrentGenerator, a black-box evolutionary algorithm that efficiently identifies recurrent generation scenarios in prominent LLMs like LLama-3 and GPT-4o. Additionally, we introduce RecurrentDetector, a lightweight real-time classifier trained on activation patterns, achieving 95.24% accuracy and an F1 score of 0.87 in detecting recurrent loops. Our methods provide practical solutions to mitigate latency-related vulnerabilities, and we publicly share our tools and data to support further research. 

---
# Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks 

**Authors**: Zongru Wu, Pengzhou Cheng, Zheng Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00401)  

**Abstract**: Perception-enhanced pre-training, particularly through grounding techniques, is widely adopted to enhance the performance of graphical user interface (GUI) agents. However, in resource-constrained scenarios, the format discrepancy between coordinate-oriented grounding and action-oriented reasoning limits the effectiveness of grounding for reasoning tasks. To address this challenge, we propose a query-oriented pivot approach called query inference, which serves as a bridge between GUI grounding and reasoning. By inferring potential user queries from a screenshot and its associated element coordinates, query inference improves the understanding of coordinates while aligning more closely with reasoning tasks. Experimental results show that query inference outperforms previous grounding techniques under the same training data scale. Notably, query inference achieves comparable or even better performance to large-scale grounding-enhanced OS-Atlas with less than 0.1% of training data. Furthermore, we explore the impact of reasoning formats and demonstrate that integrating additional semantic information into the input further boosts reasoning performance. The code is publicly available athttps://github.com/ZrW00/GUIPivot. 

---
# Reservoir Network with Structural Plasticity for Human Activity Recognition 

**Authors**: Abdullah M. Zyarah, Alaa M. Abdul-Hadi, Dhireesha Kudithipudi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00393)  

**Abstract**: The unprecedented dissemination of edge devices is accompanied by a growing demand for neuromorphic chips that can process time-series data natively without cloud support. Echo state network (ESN) is a class of recurrent neural networks that can be used to identify unique patterns in time-series data and predict future events. It is known for minimal computing resource requirements and fast training, owing to the use of linear optimization solely at the readout stage. In this work, a custom-design neuromorphic chip based on ESN targeting edge devices is proposed. The proposed system supports various learning mechanisms, including structural plasticity and synaptic plasticity, locally on-chip. This provides the network with an additional degree of freedom to continuously learn, adapt, and alter its structure and sparsity level, ensuring high performance and continuous stability. We demonstrate the performance of the proposed system as well as its robustness to noise against real-world time-series datasets while considering various topologies of data movement. An average accuracy of 95.95% and 85.24% are achieved on human activity recognition and prosthetic finger control, respectively. We also illustrate that the proposed system offers a throughput of 6x10^4 samples/sec with a power consumption of 47.7mW on a 65nm IBM process. 

---
# Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving 

**Authors**: Qihui Zhou, Peiqi Yin, Pengfei Zuo, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00392)  

**Abstract**: Processing long contexts has become a critical capability for modern large language models (LLMs). However, serving long-context LLMs comes with significant inference costs due to the high memory overhead of the key-value (KV) cache. Existing work leverages dynamic sparse attention algorithms (DSAes) to mitigate the KV cache overhead, but these algorithms rely on top-$k$ KV cache selection, which results in a trade-off between accuracy and efficiency. A larger $k$ improves accuracy but decreases efficiency, while a smaller $k$ boosts efficiency but compromises accuracy. To overcome this trade-off, this paper presents PSA, a $\underline{P}$rogressive $\underline{S}$parse $\underline{A}$ttention mechanism that integrates algorithmic innovations with system co-design to achieve both high inference accuracy and improved efficiency in LLM serving. The PSA algorithm adaptively adjusts the KV cache budget of different tokens and layers according to their real attention weight distributions, rather than relying on a fixed budget $k$. This enables high accuracy while minimizing KV cache usage. To further enhance execution efficiency, we introduce a pipelined iteration scheme that reduces CPU-GPU interleaving and synchronization overhead during PSA computation. Additionally, we implement unified GPU memory management that optimizes PSA's memory utilization by accounting for uneven memory requirements across different model layers. Extensive experimental results demonstrate that PSA reduces KV cache usage for attention computation by up to 2.4$\times$ and 8.8$\times$, and increases end-to-end serving throughput by up to 1.4$\times$ and 2.0$\times$, compared to state-of-the-art DSAes and systems without sparse attention, respectively. 

---
# BGM2Pose: Active 3D Human Pose Estimation with Non-Stationary Sounds 

**Authors**: Yuto Shibata, Yusuke Oumi, Go Irie, Akisato Kimura, Yoshimitsu Aoki, Mariko Isogawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.00389)  

**Abstract**: We propose BGM2Pose, a non-invasive 3D human pose estimation method using arbitrary music (e.g., background music) as active sensing signals. Unlike existing approaches that significantly limit practicality by employing intrusive chirp signals within the audible range, our method utilizes natural music that causes minimal discomfort to humans. Estimating human poses from standard music presents significant challenges. In contrast to sound sources specifically designed for measurement, regular music varies in both volume and pitch. These dynamic changes in signals caused by music are inevitably mixed with alterations in the sound field resulting from human motion, making it hard to extract reliable cues for pose estimation. To address these challenges, BGM2Pose introduces a Contrastive Pose Extraction Module that employs contrastive learning and hard negative sampling to eliminate musical components from the recorded data, isolating the pose information. Additionally, we propose a Frequency-wise Attention Module that enables the model to focus on subtle acoustic variations attributable to human movement by dynamically computing attention across frequency bands. Experiments suggest that our method outperforms the existing methods, demonstrating substantial potential for real-world applications. Our datasets and code will be made publicly available. 

---
# LNUCB-TA: Linear-nonlinear Hybrid Bandit Learning with Temporal Attention 

**Authors**: Hamed Khosravi, Mohammad Reza Shafie, Ahmed Shoyeb Raihan, Srinjoy Das, Imtiaz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00387)  

**Abstract**: Existing contextual multi-armed bandit (MAB) algorithms fail to effectively capture both long-term trends and local patterns across all arms, leading to suboptimal performance in environments with rapidly changing reward structures. They also rely on static exploration rates, which do not dynamically adjust to changing conditions. To overcome these limitations, we propose LNUCB-TA, a hybrid bandit model integrating a novel nonlinear component (adaptive k-Nearest Neighbors (k-NN)) for reducing time complexity, alongside a global-and-local attention-based exploration mechanism. Our approach uniquely combines linear and nonlinear estimation techniques, with the nonlinear module dynamically adjusting k based on reward variance to enhance spatiotemporal pattern recognition. This reduces the likelihood of selecting suboptimal arms while improving reward estimation accuracy and computational efficiency. The attention-based mechanism ranks arms by past performance and selection frequency, dynamically adjusting exploration and exploitation in real time without requiring manual tuning of exploration rates. By integrating global attention (assessing all arms collectively) and local attention (focusing on individual arms), LNUCB-TA efficiently adapts to temporal and spatial complexities. Empirical results show LNUCB-TA significantly outperforms state-of-the-art linear, nonlinear, and hybrid bandits in cumulative and mean reward, convergence, and robustness across different exploration rates. Theoretical analysis further confirms its reliability with a sub-linear regret bound. 

---
# A Survey of Adversarial Defenses in Vision-based Systems: Categorization, Methods and Challenges 

**Authors**: Nandish Chattopadhyay, Abdul Basit, Bassem Ouni, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2503.00384)  

**Abstract**: Adversarial attacks have emerged as a major challenge to the trustworthy deployment of machine learning models, particularly in computer vision applications. These attacks have a varied level of potency and can be implemented in both white box and black box approaches. Practical attacks include methods to manipulate the physical world and enforce adversarial behaviour by the corresponding target neural network models. Multiple different approaches to mitigate different kinds of such attacks are available in the literature, each with their own advantages and limitations. In this survey, we present a comprehensive systematization of knowledge on adversarial defenses, focusing on two key computer vision tasks: image classification and object detection. We review the state-of-the-art adversarial defense techniques and categorize them for easier comparison. In addition, we provide a schematic representation of these categories within the context of the overall machine learning pipeline, facilitating clearer understanding and benchmarking of defenses. Furthermore, we map these defenses to the types of adversarial attacks and datasets where they are most effective, offering practical insights for researchers and practitioners. This study is necessary for understanding the scope of how the available defenses are able to address the adversarial threats, and their shortcomings as well, which is necessary for driving the research in this area in the most appropriate direction, with the aim of building trustworthy AI systems for regular practical use-cases. 

---
# Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems 

**Authors**: Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Lingyu Duan, Alex C. Kot, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00383)  

**Abstract**: By locally encoding raw data into intermediate features, collaborative inference enables end users to leverage powerful deep learning models without exposure of sensitive raw data to cloud servers. However, recent studies have revealed that these intermediate features may not sufficiently preserve privacy, as information can be leaked and raw data can be reconstructed via model inversion attacks (MIAs). Obfuscation-based methods, such as noise corruption, adversarial representation learning, and information filters, enhance the inversion robustness by obfuscating the task-irrelevant redundancy empirically. However, methods for quantifying such redundancy remain elusive, and the explicit mathematical relation between this redundancy minimization and inversion robustness enhancement has not yet been established. To address that, this work first theoretically proves that the conditional entropy of inputs given intermediate features provides a guaranteed lower bound on the reconstruction mean square error (MSE) under any MIA. Then, we derive a differentiable and solvable measure for bounding this conditional entropy based on the Gaussian mixture estimation and propose a conditional entropy maximization (CEM) algorithm to enhance the inversion robustness. Experimental results on four datasets demonstrate the effectiveness and adaptability of our proposed CEM; without compromising feature utility and computing efficiency, plugging the proposed CEM into obfuscation-based defense mechanisms consistently boosts their inversion robustness, achieving average gains ranging from 12.9\% to 48.2\%. Code is available at \href{this https URL}{this https URL}. 

---
# Conditioning on Local Statistics for Scalable Heterogeneous Federated Learning 

**Authors**: Rickard Brännvall  

**Link**: [PDF](https://arxiv.org/pdf/2503.00378)  

**Abstract**: Federated learning is a distributed machine learning approach where multiple clients collaboratively train a model without sharing their local data, which contributes to preserving privacy. A challenge in federated learning is managing heterogeneous data distributions across clients, which can hinder model convergence and performance due to the need for the global model to generalize well across diverse local datasets. We propose to use local characteristic statistics, by which we mean some statistical properties calculated independently by each client using only their local training dataset. These statistics, such as means, covariances, and higher moments, are used to capture the characteristics of the local data distribution. They are not shared with other clients or a central node. During training, these local statistics help the model learn how to condition on the local data distribution, and during inference, they guide the client's predictions. Our experiments show that this approach allows for efficient handling of heterogeneous data across the federation, has favorable scaling compared to approaches that directly try to identify peer nodes that share distribution characteristics, and maintains privacy as no additional information needs to be communicated. 

---
# MIRROR: Multi-Modal Pathological Self-Supervised Representation Learning via Modality Alignment and Retention 

**Authors**: Tianyi Wang, Jianan Fan, Dingxin Zhang, Dongnan Liu, Yong Xia, Heng Huang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.00374)  

**Abstract**: Histopathology and transcriptomics are fundamental modalities in oncology, encapsulating the morphological and molecular aspects of the disease. Multi-modal self-supervised learning has demonstrated remarkable potential in learning pathological representations by integrating diverse data sources. Conventional multi-modal integration methods primarily emphasize modality alignment, while paying insufficient attention to retaining the modality-specific structures. However, unlike conventional scenarios where multi-modal inputs share highly overlapping features, histopathology and transcriptomics exhibit pronounced heterogeneity, offering orthogonal yet complementary insights. Histopathology provides morphological and spatial context, elucidating tissue architecture and cellular topology, whereas transcriptomics delineates molecular signatures through gene expression patterns. This inherent disparity introduces a major challenge in aligning them while maintaining modality-specific fidelity. To address these challenges, we present MIRROR, a novel multi-modal representation learning method designed to foster both modality alignment and retention. MIRROR employs dedicated encoders to extract comprehensive features for each modality, which is further complemented by a modality alignment module to achieve seamless integration between phenotype patterns and molecular profiles. Furthermore, a modality retention module safeguards unique attributes from each modality, while a style clustering module mitigates redundancy and enhances disease-relevant information by modeling and aligning consistent pathological signatures within a clustering space. Extensive evaluations on TCGA cohorts for cancer subtyping and survival analysis highlight MIRROR's superior performance, demonstrating its effectiveness in constructing comprehensive oncological feature representations and benefiting the cancer diagnosis. 

---
# Nucleolus Credit Assignment for Effective Coalitions in Multi-agent Reinforcement Learning 

**Authors**: Yugu Li, Zehong Cao, Jianglin Qiao, Siyi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00372)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), agents typically form a single grand coalition based on credit assignment to tackle a composite task, often resulting in suboptimal performance. This paper proposed a nucleolus-based credit assignment grounded in cooperative game theory, enabling the autonomous partitioning of agents into multiple small coalitions that can effectively identify and complete subtasks within a larger composite task. Specifically, our designed nucleolus Q-learning could assign fair credits to each agent, and the nucleolus Q-operator provides theoretical guarantees with interpretability for both learning convergence and the stability of the formed small coalitions. Through experiments on Predator-Prey and StarCraft scenarios across varying difficulty levels, our approach demonstrated the emergence of multiple effective coalitions during MARL training, leading to faster learning and superior performance in terms of win rate and cumulative rewards especially in hard and super-hard environments, compared to four baseline methods. Our nucleolus-based credit assignment showed the promise for complex composite tasks requiring effective subteams of agents. 

---
# AI-Augmented Thyroid Scintigraphy for Robust Classification 

**Authors**: Maziar Sabouri, Ghasem Hajianfar, Alireza Rafiei Sardouei, Milad Yazdani, Azin Asadzadeh, Soroush Bagheri, Mohsen Arabi, Seyed Rasoul Zakavi, Emran Askari, Atena Aghaee, Dena Shahriari, Habib Zaidi, Arman Rahmim  

**Link**: [PDF](https://arxiv.org/pdf/2503.00366)  

**Abstract**: Thyroid scintigraphy is a key imaging modality for diagnosing thyroid disorders. Deep learning models for thyroid scintigraphy classification often face challenges due to limited and imbalanced datasets, leading to suboptimal generalization. In this study, we investigate the effectiveness of different data augmentation techniques including Stable Diffusion (SD), Flow Matching (FM), and Conventional Augmentation (CA) to enhance the performance of a ResNet18 classifier for thyroid condition classification. Our results showed that FM-based augmentation consistently outperforms SD-based approaches, particularly when combined with original (O) data and CA (O+FM+CA), achieving both high accuracy and fair classification across Diffuse Goiter (DG), Nodular Goiter (NG), Normal (NL), and Thyroiditis (TI) cases. The Wilcoxon statistical analysis further validated the superiority of O+FM and its variants (O+FM+CA) over SD-based augmentations in most scenarios. These findings highlight the potential of FM-based augmentation as a superior approach for generating high-quality synthetic thyroid scintigraphy images and improving model generalization in medical image classification. 

---
# Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding 

**Authors**: Wei Suo, Lijun Zhang, Mengyang Sun, Lin Yuanbo Wu, Peng Wang, Yanning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00361)  

**Abstract**: Large Vision-Language Models (LVLMs) have obtained impressive performance in visual content understanding and multi-modal reasoning. Unfortunately, these large models suffer from serious hallucination problems and tend to generate fabricated responses. Recently, several Contrastive Decoding (CD) strategies have been proposed to alleviate hallucination by introducing disturbed inputs. Although great progress has been made, these CD strategies mostly apply a one-size-fits-all approach for all input conditions. In this paper, we revisit this process through extensive experiments. Related results show that hallucination causes are hybrid and each generative step faces a unique hallucination challenge. Leveraging these meaningful insights, we introduce a simple yet effective Octopus-like framework that enables the model to adaptively identify hallucination types and create a dynamic CD workflow. Our Octopus framework not only outperforms existing methods across four benchmarks but also demonstrates excellent deployability and expansibility. Code is available at this https URL. 

---
# CRUPL: A Semi-Supervised Cyber Attack Detection with Consistency Regularization and Uncertainty-aware Pseudo-Labeling in Smart Grid 

**Authors**: Smruti P. Dash, Kedar V. Khandeparkar, Nipun Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.00358)  

**Abstract**: The modern power grids are integrated with digital technologies and automation systems. The inclusion of digital technologies has made the smart grids vulnerable to cyber-attacks. Cyberattacks on smart grids can compromise data integrity and jeopardize the reliability of the power supply. Traditional intrusion detection systems often need help to effectively detect novel and sophisticated attacks due to their reliance on labeled training data, which may only encompass part of the spectrum of potential threats. This work proposes a semi-supervised method for cyber-attack detection in smart grids by leveraging the labeled and unlabeled measurement data. We implement consistency regularization and pseudo-labeling to identify deviations from expected behavior and predict the attack classes. We use a curriculum learning approach to improve pseudo-labeling performance, capturing the model uncertainty. We demonstrate the efficiency of the proposed method in detecting different types of cyberattacks, minimizing the false positives by implementing them on publicly available datasets. The method proposes a promising solution by improving the detection accuracy to 99% in the presence of unknown samples and significantly reducing false positives. 

---
# BERT-based model for Vietnamese Fact Verification Dataset 

**Authors**: Bao Tran, T. N. Khanh, Khang Nguyen Tuong, Thien Dang, Quang Nguyen, Nguyen T. Thinh, Vo T. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2503.00356)  

**Abstract**: The rapid advancement of information and communication technology has facilitated easier access to information. However, this progress has also necessitated more stringent verification measures to ensure the accuracy of information, particularly within the context of Vietnam. This paper introduces an approach to address the challenges of Fact Verification using the Vietnamese dataset by integrating both sentence selection and classification modules into a unified network architecture. The proposed approach leverages the power of large language models by utilizing pre-trained PhoBERT and XLM-RoBERTa as the backbone of the network. The proposed model was trained on a Vietnamese dataset, named ISE-DSC01, and demonstrated superior performance compared to the baseline model across all three metrics. Notably, we achieved a Strict Accuracy level of 75.11\%, indicating a remarkable 28.83\% improvement over the baseline model. 

---
# Structured Reasoning for Fairness: A Multi-Agent Approach to Bias Detection in Textual Data 

**Authors**: Tianyi Huang, Elsa Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00355)  

**Abstract**: From disinformation spread by AI chatbots to AI recommendations that inadvertently reinforce stereotypes, textual bias poses a significant challenge to the trustworthiness of large language models (LLMs). In this paper, we propose a multi-agent framework that systematically identifies biases by disentangling each statement as fact or opinion, assigning a bias intensity score, and providing concise, factual justifications. Evaluated on 1,500 samples from the WikiNPOV dataset, the framework achieves 84.9% accuracy$\unicode{x2014}$an improvement of 13.0% over the zero-shot baseline$\unicode{x2014}$demonstrating the efficacy of explicitly modeling fact versus opinion prior to quantifying bias intensity. By combining enhanced detection accuracy with interpretable explanations, this approach sets a foundation for promoting fairness and accountability in modern language models. 

---
# MCNet: Monotonic Calibration Networks for Expressive Uncertainty Calibration in Online Advertising 

**Authors**: Quanyu Dai, Jiaren Xiao, Zhaocheng Du, Jieming Zhu, Chengxiao Luo, Xiao-Ming Wu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.00334)  

**Abstract**: In online advertising, uncertainty calibration aims to adjust a ranking model's probability predictions to better approximate the true likelihood of an event, e.g., a click or a conversion. However, existing calibration approaches may lack the ability to effectively model complex nonlinear relations, consider context features, and achieve balanced performance across different data subsets. To tackle these challenges, we introduce a novel model called Monotonic Calibration Networks, featuring three key designs: a monotonic calibration function (MCF), an order-preserving regularizer, and a field-balance regularizer. The nonlinear MCF is capable of naturally modeling and universally approximating the intricate relations between uncalibrated predictions and the posterior probabilities, thus being much more expressive than existing methods. MCF can also integrate context features using a flexible model architecture, thereby achieving context awareness. The order-preserving and field-balance regularizers promote the monotonic relationship between adjacent bins and the balanced calibration performance on data subsets, respectively. Experimental results on both public and industrial datasets demonstrate the superior performance of our method in generating well-calibrated probability predictions. 

---
# More of the Same: Persistent Representational Harms Under Increased Representation 

**Authors**: Jennifer Mickel, Maria De-Arteaga, Leqi Liu, Kevin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.00333)  

**Abstract**: To recognize and mitigate the harms of generative AI systems, it is crucial to consider who is represented in the outputs of generative AI systems and how people are represented. A critical gap emerges when naively improving who is represented, as this does not imply bias mitigation efforts have been applied to address how people are represented. We critically examined this by investigating gender representation in occupation across state-of-the-art large language models. We first show evidence suggesting that over time there have been interventions to models altering the resulting gender distribution, and we find that women are more represented than men when models are prompted to generate biographies or personas. We then demonstrate that representational biases persist in how different genders are represented by examining statistically significant word differences across genders. This results in a proliferation of representational harms, stereotypes, and neoliberalism ideals that, despite existing interventions to increase female representation, reinforce existing systems of oppression. 

---
# Investigating the contribution of terrain-following coordinates and conservation schemes in AI-driven precipitation forecasts 

**Authors**: Yingkai Sha, John S. Schreck, William Chapman, David John Gagne II  

**Link**: [PDF](https://arxiv.org/pdf/2503.00332)  

**Abstract**: Artificial Intelligence (AI) weather prediction (AIWP) models often produce ``blurry'' precipitation forecasts that overestimate drizzle and underestimate extremes. This study provides a novel solution to tackle this problem -- integrating terrain-following coordinates with global mass and energy conservation schemes into AIWP models. Forecast experiments are conducted to evaluate the effectiveness of this solution using FuXi, an example AIWP model, adapted to 1.0$^\circ$ grid spacing data. Verification results show large performance gains. The conservation schemes are found to reduce drizzle bias, whereas using terrain-following coordinates improves the estimation of extreme events and precipitation intensity spectra. Furthermore, a case study reveals that terrain-following coordinates capture near-surface winds better over mountains, offering AIWP models more accurate information on understanding the dynamics of precipitation processes. The proposed solution of this study can benefit a wide range of AIWP models and bring insights into how atmospheric domain knowledge can support the development of AIWP models. 

---
# PINN-DT: Optimizing Energy Consumption in Smart Building Using Hybrid Physics-Informed Neural Networks and Digital Twin Framework with Blockchain Security 

**Authors**: Hajar Kazemi Naeini, Roya Shomali, Abolhassan Pishahang, Hamidreza Hasanzadeh, Mahdieh Mohammadi, Saeid Asadi, Ahmad Gholizadeh Lonbar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00331)  

**Abstract**: The advancement of smart grid technologies necessitates the integration of cutting-edge computational methods to enhance predictive energy optimization. This study proposes a multi-faceted approach by incorporating (1) Deep Reinforcement Learning (DRL) agents trained using data from Digital Twins (DTs) to optimize energy consumption in real time, (2) Physics-Informed Neural Networks (PINNs) to seamlessly embed physical laws within the optimization process, ensuring model accuracy and interpretability, and (3) Blockchain (BC) technology to facilitate secure and transparent communication across the smart grid infrastructure. The model was trained and validated using comprehensive datasets, including smart meter energy consumption data, renewable energy outputs, dynamic pricing, and user preferences collected from IoT devices. The proposed framework achieved superior predictive performance with a Mean Absolute Error (MAE) of 0.237 kWh, Root Mean Square Error (RMSE) of 0.298 kWh, and an R-squared (R2) value of 0.978, indicating a 97.8% explanation of data variance. Classification metrics further demonstrated the model's robustness, achieving 97.7% accuracy, 97.8% precision, 97.6% recall, and an F1 Score of 97.7%. Comparative analysis with traditional models like Linear Regression, Random Forest, SVM, LSTM, and XGBoost revealed the superior accuracy and real-time adaptability of the proposed method. In addition to enhancing energy efficiency, the model reduced energy costs by 35%, maintained a 96% user comfort index, and increased renewable energy utilization to 40%. This study demonstrates the transformative potential of integrating PINNs, DT, and Blockchain technologies to optimize energy consumption in smart grids, paving the way for sustainable, secure, and efficient energy management systems. 

---
# FLStore: Efficient Federated Learning Storage for non-training workloads 

**Authors**: Ahmad Faraz Khan, Samuel Fountain, Ahmed M. Abdelmoniem, Ali R. Butt, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00323)  

**Abstract**: Federated Learning (FL) is an approach for privacy-preserving Machine Learning (ML), enabling model training across multiple clients without centralized data collection. With an aggregator server coordinating training, aggregating model updates, and storing metadata across rounds. In addition to training, a substantial part of FL systems are the non-training workloads such as scheduling, personalization, clustering, debugging, and incentivization. Most existing systems rely on the aggregator to handle non-training workloads and use cloud services for data storage. This results in high latency and increased costs as non-training workloads rely on large volumes of metadata, including weight parameters from client updates, hyperparameters, and aggregated updates across rounds, making the situation even worse. We propose FLStore, a serverless framework for efficient FL non-training workloads and storage. FLStore unifies the data and compute planes on a serverless cache, enabling locality-aware execution via tailored caching policies to reduce latency and costs. Per our evaluations, compared to cloud object store based aggregator server FLStore reduces per request average latency by 71% and costs by 92.45%, with peak improvements of 99.7% and 98.8%, respectively. Compared to an in-memory cloud cache based aggregator server, FLStore reduces average latency by 64.6% and costs by 98.83%, with peak improvements of 98.8% and 99.6%, respectively. FLStore integrates seamlessly with existing FL frameworks with minimal modifications, while also being fault-tolerant and highly scalable. 

---
# T-REX: A 68-567 μs/token, 0.41-3.95 μJ/token Transformer Accelerator with Reduced External Memory Access and Enhanced Hardware Utilization in 16nm FinFET 

**Authors**: Seunghyun Moon, Mao Li, Gregory Chen, Phil Knag, Ram Krishnamurthy, Mingoo Seok  

**Link**: [PDF](https://arxiv.org/pdf/2503.00322)  

**Abstract**: This work introduces novel training and post-training compression schemes to reduce external memory access during transformer model inference. Additionally, a new control flow mechanism, called dynamic batching, and a novel buffer architecture, termed a two-direction accessible register file, further reduce external memory access while improving hardware utilization. 

---
# Shifting Power: Leveraging LLMs to Simulate Human Aversion in ABMs of Bilateral Financial Exchanges, A bond market study 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00320)  

**Abstract**: Bilateral markets, such as those for government bonds, involve decentralized and opaque transactions between market makers (MMs) and clients, posing significant challenges for traditional modeling approaches. To address these complexities, we introduce TRIBE an agent-based model augmented with a large language model (LLM) to simulate human-like decision-making in trading environments. TRIBE leverages publicly available data and stylized facts to capture realistic trading dynamics, integrating human biases like risk aversion and ambiguity sensitivity into the decision-making processes of agents. Our research yields three key contributions: first, we demonstrate that integrating LLMs into agent-based models to enhance client agency is feasible and enriches the simulation of agent behaviors in complex markets; second, we find that even slight trade aversion encoded within the LLM leads to a complete cessation of trading activity, highlighting the sensitivity of market dynamics to agents' risk profiles; third, we show that incorporating human-like variability shifts power dynamics towards clients and can disproportionately affect the entire system, often resulting in systemic agent collapse across simulations. These findings underscore the emergent properties that arise when introducing stochastic, human-like decision processes, revealing new system behaviors that enhance the realism and complexity of artificial societies. 

---
# Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM 

**Authors**: Yuxin Yang, Haoyang Wu, Tao Wang, Jia Yang, Hao Ma, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00309)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized natural language processing. However, these models face challenges in retrieving precise information from vast datasets. Retrieval-Augmented Generation (RAG) was developed to combining LLMs with external information retrieval systems to enhance the accuracy and context of responses. Despite improvements, RAG still struggles with comprehensive retrieval in high-volume, low-information-density databases and lacks relational awareness, leading to fragmented answers.
To address this, this paper introduces the Pseudo-Knowledge Graph (PKG) framework, designed to overcome these limitations by integrating Meta-path Retrieval, In-graph Text and Vector Retrieval into LLMs. By preserving natural language text and leveraging various retrieval techniques, the PKG offers a richer knowledge representation and improves accuracy in information retrieval. Extensive evaluations using Open Compass and MultiHop-RAG benchmarks demonstrate the framework's effectiveness in managing large volumes of data and complex relationships. 

---
# Hidden Convexity of Fair PCA and Fast Solver via Eigenvalue Optimization 

**Authors**: Junhui Shen, Aaron J. Davis, Ding Lu, Zhaojun Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.00299)  

**Abstract**: Principal Component Analysis (PCA) is a foundational technique in machine learning for dimensionality reduction of high-dimensional datasets. However, PCA could lead to biased outcomes that disadvantage certain subgroups of the underlying datasets. To address the bias issue, a Fair PCA (FPCA) model was introduced by Samadi et al. (2018) for equalizing the reconstruction loss between subgroups. The semidefinite relaxation (SDR) based approach proposed by Samadi et al. (2018) is computationally expensive even for suboptimal solutions. To improve efficiency, several alternative variants of the FPCA model have been developed. These variants often shift the focus away from equalizing the reconstruction loss. In this paper, we identify a hidden convexity in the FPCA model and introduce an algorithm for convex optimization via eigenvalue optimization. Our approach achieves the desired fairness in reconstruction loss without sacrificing performance. As demonstrated in real-world datasets, the proposed FPCA algorithm runs $8\times$ faster than the SDR-based algorithm, and only at most 85% slower than the standard PCA. 

---
# A Unified Framework for Heterogeneous Semi-supervised Learning 

**Authors**: Marzi Heidari, Abdullah Alchihabi, Hao Yan, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00286)  

**Abstract**: In this work, we introduce a novel problem setup termed as Heterogeneous Semi-Supervised Learning (HSSL), which presents unique challenges by bridging the semi-supervised learning (SSL) task and the unsupervised domain adaptation (UDA) task, and expanding standard semi-supervised learning to cope with heterogeneous training data. At its core, HSSL aims to learn a prediction model using a combination of labeled and unlabeled training data drawn separately from heterogeneous domains that share a common set of semantic categories; this model is intended to differentiate the semantic categories of test instances sampled from both the labeled and unlabeled domains. In particular, the labeled and unlabeled domains have dissimilar label distributions and class feature distributions. This heterogeneity, coupled with the assorted sources of the test data, introduces significant challenges to standard SSL and UDA methods. Therefore, we propose a novel method, Unified Framework for Heterogeneous Semi-supervised Learning (Uni-HSSL), to address HSSL by directly learning a fine-grained classifier from the heterogeneous data, which adaptively handles the inter-domain heterogeneity while leveraging both the unlabeled data and the inter-domain semantic class relationships for cross-domain knowledge transfer and adaptation. We conduct comprehensive experiments and the experimental results validate the efficacy and superior performance of the proposed Uni-HSSL over state-of-the-art semi-supervised learning and unsupervised domain adaptation methods. 

---
# Reducing Large Language Model Safety Risks in Women's Health using Semantic Entropy 

**Authors**: Jahan C. Penny-Dimri, Magdalena Bachmann, William R. Cooke, Sam Mathewlynn, Samuel Dockree, John Tolladay, Jannik Kossen, Lin Li, Yarin Gal, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2503.00269)  

**Abstract**: Large language models (LLMs) hold substantial promise for clinical decision support. However, their widespread adoption in medicine, particularly in healthcare, is hindered by their propensity to generate false or misleading outputs, known as hallucinations. In high-stakes domains such as women's health (obstetrics & gynaecology), where errors in clinical reasoning can have profound consequences for maternal and neonatal outcomes, ensuring the reliability of AI-generated responses is critical. Traditional methods for quantifying uncertainty, such as perplexity, fail to capture meaning-level inconsistencies that lead to misinformation. Here, we evaluate semantic entropy (SE), a novel uncertainty metric that assesses meaning-level variation, to detect hallucinations in AI-generated medical content. Using a clinically validated dataset derived from UK RCOG MRCOG examinations, we compared SE with perplexity in identifying uncertain responses. SE demonstrated superior performance, achieving an AUROC of 0.76 (95% CI: 0.75-0.78), compared to 0.62 (0.60-0.65) for perplexity. Clinical expert validation further confirmed its effectiveness, with SE achieving near-perfect uncertainty discrimination (AUROC: 0.97). While semantic clustering was successful in only 30% of cases, SE remains a valuable tool for improving AI safety in women's health. These findings suggest that SE could enable more reliable AI integration into clinical practice, particularly in resource-limited settings where LLMs could augment care. This study highlights the potential of SE as a key safeguard in the responsible deployment of AI-driven tools in women's health, leading to safer and more effective digital health interventions. 

---
# Input Specific Neural Networks 

**Authors**: Asghar A. Jadoon, D. Thomas Seidl, Reese E. Jones, Jan N. Fuhg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00268)  

**Abstract**: The black-box nature of neural networks limits the ability to encode or impose specific structural relationships between inputs and outputs. While various studies have introduced architectures that ensure the network's output adheres to a particular form in relation to certain inputs, the majority of these approaches impose constraints on only a single set of inputs. This paper introduces a novel neural network architecture, termed the Input Specific Neural Network (ISNN), which extends this concept by allowing scalar-valued outputs to be subject to multiple constraints. Specifically, the ISNN can enforce convexity in some inputs, non-decreasing monotonicity combined with convexity with respect to others, and simple non-decreasing monotonicity or arbitrary relationships with additional inputs. The paper presents two distinct ISNN architectures, along with equations for the first and second derivatives of the output with respect to the inputs. These networks are broadly applicable.
In this work, we restrict their usage to solving problems in computational mechanics. In particular, we show how they can be effectively applied to fitting data-driven constitutive models. We then embed our trained data-driven constitutive laws into a finite element solver where significant time savings can be achieved by using explicit manual differentiation using the derived equations as opposed to automatic differentiation. We also show how ISNNs can be used to learn structural relationships between inputs and outputs via a binary gating mechanism. Particularly, ISNNs are employed to model an anisotropic free energy potential to get the homogenized macroscopic response in a decoupled multiscale setting, where the network learns whether or not the potential should be modeled as polyconvex, and retains only the relevant layers while using the minimum number of inputs. 

---
# Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text 

**Authors**: Guangsheng Bao, Lihua Rong, Yanbin Zhao, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00258)  

**Abstract**: The wide usage of LLMs raises critical requirements on detecting AI participation in texts. Existing studies investigate these detections in scattered contexts, leaving a systematic and unified approach unexplored. In this paper, we present HART, a hierarchical framework of AI risk levels, each corresponding to a detection task. To address these tasks, we propose a novel 2D Detection Method, decoupling a text into content and language expression. Our findings show that content is resistant to surface-level changes, which can serve as a key feature for detection. Experiments demonstrate that 2D method significantly outperforms existing detectors, achieving an AUROC improvement from 0.705 to 0.849 for level-2 detection and from 0.807 to 0.886 for RAID. We release our data and code at this https URL. 

---
# 1-Lipschitz Network Initialization for Certifiably Robust Classification Applications: A Decay Problem 

**Authors**: Marius F. R. Juston, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00240)  

**Abstract**: This paper discusses the weight parametrization of two standard 1-Lipschitz network structure methodologies, the Almost-Orthogonal-Layers (AOL) and the SDP-based Lipschitz Layers (SLL), and derives their impact on the initialization for deep 1-Lipschitz feedforward networks in addition to discussing underlying issues surrounding this initialization. These networks are mainly used in certifiably robust classification applications to combat adversarial attacks by limiting the effects of perturbations on the output classification result. An exact and an upper bound for the parameterized weight variance was calculated assuming a standard Normal distribution initialization; additionally, an upper bound was computed assuming a Generalized Normal Distribution, generalizing the proof for Uniform, Laplace, and Normal distribution weight initializations. It is demonstrated that the weight variance holds no bearing on the output variance distribution and that only the dimension of the weight matrices matters. Additionally, this paper demonstrates that the weight initialization always causes deep 1-Lipschitz networks to decay to zero. 

---
# Towards Fairness for the Right Reasons: Using Saliency Maps to Evaluate Bias Removal in Neural Networks 

**Authors**: Lukasz Sztukiewicz, Ignacy Stępka, Michał Wiliński, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2503.00234)  

**Abstract**: The widespread adoption of machine learning systems has raised critical concerns about fairness and bias, making mitigating harmful biases essential for AI development. In this paper, we investigate the relationship between fairness improvement and the removal of harmful biases in neural networks applied to computer vision tasks. First, we introduce a set of novel XAI-based metrics that analyze saliency maps to assess shifts in a model's decision-making process. Then, we demonstrate that successful debiasing methods systematically redirect model focus away from protected attributes. Additionally, we show that techniques originally developed for artifact removal can be effectively repurposed for fairness. These findings underscore the importance of ensuring that models are fair for the right reasons, contributing to the development of more ethical and trustworthy AI systems. 

---
# Jawaher: A Multidialectal Dataset of Arabic Proverbs for LLM Benchmarking 

**Authors**: Samar M. Magdy, Sang Yun Kwon, Fakhraddin Alwajih, Safaa Abdelfadil, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00231)  

**Abstract**: Recent advancements in instruction fine-tuning, alignment methods such as reinforcement learning from human feedback (RLHF), and optimization techniques like direct preference optimization (DPO) have significantly enhanced the adaptability of large language models (LLMs) to user preferences. However, despite these innovations, many LLMs continue to exhibit biases toward Western, Anglo-centric, or American cultures, with performance on English data consistently surpassing that of other languages. This reveals a persistent cultural gap in LLMs, which complicates their ability to accurately process culturally rich and diverse figurative language such as proverbs. To address this, we introduce Jawaher, a benchmark designed to assess LLMs' capacity to comprehend and interpret Arabic proverbs. Jawaher includes proverbs from various Arabic dialects, along with idiomatic translations and explanations. Through extensive evaluations of both open- and closed-source models, we find that while LLMs can generate idiomatically accurate translations, they struggle with producing culturally nuanced and contextually relevant explanations. These findings highlight the need for ongoing model refinement and dataset expansion to bridge the cultural gap in figurative language processing. 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

---
# Foundation-Model-Boosted Multimodal Learning for fMRI-based Neuropathic Pain Drug Response Prediction 

**Authors**: Wenrui Fan, L. M. Riza Rizky, Jiayang Zhang, Chen Chen, Haiping Lu, Kevin Teh, Dinesh Selvarajah, Shuo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00210)  

**Abstract**: Neuropathic pain, affecting up to 10% of adults, remains difficult to treat due to limited therapeutic efficacy and tolerability. Although resting-state functional MRI (rs-fMRI) is a promising non-invasive measurement of brain biomarkers to predict drug response in therapeutic development, the complexity of fMRI demands machine learning models with substantial capacity. However, extreme data scarcity in neuropathic pain research limits the application of high-capacity models. To address the challenge of data scarcity, we propose FMM$_{TC}$, a Foundation-Model-boosted Multimodal learning framework for fMRI-based neuropathic pain drug response prediction, which leverages both internal multimodal information in pain-specific data and external knowledge from large pain-agnostic data. Specifically, to maximize the value of limited pain-specific data, FMM$_{TC}$ integrates complementary information from two rs-fMRI modalities: Time series and functional Connectivity. FMM$_{TC}$ is further boosted by an fMRI foundation model with its external knowledge from extensive pain-agnostic fMRI datasets enriching limited pain-specific information. Evaluations with an in-house dataset and a public dataset from OpenNeuro demonstrate FMM$_{TC}$'s superior representation ability, generalizability, and cross-dataset adaptability over existing unimodal fMRI models that only consider one of the rs-fMRI modalities. The ablation study validates the effectiveness of multimodal learning and foundation-model-powered external knowledge transfer in FMM$_{TC}$. An integrated gradient-based interpretation study explains how FMM$_{TC}$'s cross-dataset dynamic behaviors enhance its adaptability. In conclusion, FMM$_{TC}$ boosts clinical trials in neuropathic pain therapeutic development by accurately predicting drug responses to improve the participant stratification efficiency. 

---
# Quantifying First-Order Markov Violations in Noisy Reinforcement Learning: A Causal Discovery Approach 

**Authors**: Naveen Mysore  

**Link**: [PDF](https://arxiv.org/pdf/2503.00206)  

**Abstract**: Reinforcement learning (RL) methods frequently assume that each new observation completely reflects the environment's state, thereby guaranteeing Markovian (one-step) transitions. In practice, partial observability or sensor/actuator noise often invalidates this assumption. This paper proposes a systematic methodology for detecting such violations, combining a partial correlation-based causal discovery process (PCMCI) with a novel Markov Violation score (MVS). The MVS measures multi-step dependencies that emerge when noise or incomplete state information disrupts the Markov property.
Classic control tasks (CartPole, Pendulum, Acrobot) serve as examples to illustrate how targeted noise and dimension omissions affect both RL performance and measured Markov consistency. Surprisingly, even substantial observation noise sometimes fails to induce strong multi-lag dependencies in certain domains (e.g., Acrobot). In contrast, dimension-dropping investigations show that excluding some state variables (e.g., angular velocities in CartPole and Pendulum) significantly reduces returns and increases MVS, while removing other dimensions has minimal impact.
These findings emphasize the importance of locating and safeguarding the most causally essential dimensions in order to preserve effective single-step learning. By integrating partial correlation tests with RL performance outcomes, the proposed approach precisely identifies when and where the Markov assumption is violated. This framework offers a principled mechanism for developing robust policies, informing representation learning, and addressing partial observability in real-world RL scenarios. All code and experimental logs are accessible for reproducibility (this https URL). 

---
# PRISM: High-Resolution & Precise Counterfactual Medical Image Generation using Language-guided Stable Diffusion 

**Authors**: Amar Kumar, Anita Kriz, Mohammad Havaei, Tal Arbel  

**Link**: [PDF](https://arxiv.org/pdf/2503.00196)  

**Abstract**: Developing reliable and generalizable deep learning systems for medical imaging faces significant obstacles due to spurious correlations, data imbalances, and limited text annotations in datasets. Addressing these challenges requires architectures robust to the unique complexities posed by medical imaging data. The rapid advancements in vision-language foundation models within the natural image domain prompt the question of how they can be adapted for medical imaging tasks. In this work, we present PRISM, a framework that leverages foundation models to generate high-resolution, language-guided medical image counterfactuals using Stable Diffusion. Our approach demonstrates unprecedented precision in selectively modifying spurious correlations (the medical devices) and disease features, enabling the removal and addition of specific attributes while preserving other image characteristics. Through extensive evaluation, we show how PRISM advances counterfactual generation and enables the development of more robust downstream classifiers for clinically deployable solutions. To facilitate broader adoption and research, we make our code publicly available at this https URL. 

---
# Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees 

**Authors**: Xinhang Ma, Junlin Wu, Hussein Sibai, Yiannis Kantaros, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2503.00191)  

**Abstract**: Ensuring safety in autonomous systems with vision-based control remains a critical challenge due to the high dimensionality of image inputs and the fact that the relationship between true system state and its visual manifestation is unknown. Existing methods for learning-based control in such settings typically lack formal safety guarantees. To address this challenge, we introduce a novel semi-probabilistic verification framework that integrates reachability analysis with conditional generative adversarial networks and distribution-free tail bounds to enable efficient and scalable verification of vision-based neural network controllers. Next, we develop a gradient-based training approach that employs a novel safety loss function, safety-aware data-sampling strategy to efficiently select and store critical training examples, and curriculum learning, to efficiently synthesize safe controllers in the semi-probabilistic framework. Empirical evaluations in X-Plane 11 airplane landing simulation, CARLA-simulated autonomous lane following, and F1Tenth lane following in a physical visually-rich miniature environment demonstrate the effectiveness of our method in achieving formal safety guarantees while maintaining strong nominal performance. Our code is available at this https URL. 

---
# Zero-Shot and Efficient Clarification Need Prediction in Conversational Search 

**Authors**: Lili Lu, Chuan Meng, Federico Ravenda, Mohammad Aliannejadi, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00179)  

**Abstract**: Clarification need prediction (CNP) is a key task in conversational search, aiming to predict whether to ask a clarifying question or give an answer to the current user query. However, current research on CNP suffers from the issues of limited CNP training data and low efficiency. In this paper, we propose a zero-shot and efficient CNP framework (Zef-CNP), in which we first prompt large language models (LLMs) in a zero-shot manner to generate two sets of synthetic queries: ambiguous and specific (unambiguous) queries. We then use the generated queries to train efficient CNP models. Zef-CNP eliminates the need for human-annotated clarification-need labels during training and avoids the use of LLMs with high query latency at query time. To further improve the generation quality of synthetic queries, we devise a topic-, information-need-, and query-aware chain-of-thought (CoT) prompting strategy (TIQ-CoT). Moreover, we enhance TIQ-CoT with counterfactual query generation (CoQu), which guides LLMs first to generate a specific/ambiguous query and then sequentially generate its corresponding ambiguous/specific query. Experimental results show that Zef-CNP achieves superior CNP effectiveness and efficiency compared with zero- and few-shot LLM-based CNP predictors. 

---
# Steering Large Language Model Activations in Sparse Spaces 

**Authors**: Reza Bayat, Ali Rahimi-Kalahroudi, Mohammad Pezeshki, Sarath Chandar, Pascal Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2503.00177)  

**Abstract**: A key challenge in AI alignment is guiding large language models (LLMs) to follow desired behaviors at test time. Activation steering, which modifies internal model activations during inference, offers a potential solution. However, prior work in dense activation spaces struggles with superposition, wherein multiple features become entangled, limiting interpretability and precise control. In contrast, sparse representations provide an untapped opportunity for more interpretable behavior modulation. In this work, we introduce sparse activation steering (SAS), a method that leverages sparse autoencoders (SAEs) to steer LLM behavior in sparse spaces. By isolating behavior-specific features through a contrastive prompt-pairing approach, we define a set of features that can selectively reinforce or suppress behaviors. Experiments on Gemma 2 LLMs show that SAS vectors enable nuanced behavioral modulation and finer-grained control. Furthermore, scaling SAEs improves monosemanticity of SAS vectors, suggesting more reliable and interpretable interventions. 

---
# PaliGemma-CXR: A Multi-task Multimodal Model for TB Chest X-ray Interpretation 

**Authors**: Denis Musinguzi, Andrew Katumba, Sudi Murindanyi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00171)  

**Abstract**: Tuberculosis (TB) is a infectious global health challenge. Chest X-rays are a standard method for TB screening, yet many countries face a critical shortage of radiologists capable of interpreting these images. Machine learning offers an alternative, as it can automate tasks such as disease diagnosis, and report generation. However, traditional approaches rely on task-specific models, which cannot utilize the interdependence between tasks. Building a multi-task model capable of performing multiple tasks poses additional challenges such as scarcity of multimodal data, dataset imbalance, and negative transfer. To address these challenges, we propose PaliGemma-CXR, a multi-task multimodal model capable of performing TB diagnosis, object detection, segmentation, report generation, and VQA. Starting with a dataset of chest X-ray images annotated with TB diagnosis labels and segmentation masks, we curated a multimodal dataset to support additional tasks. By finetuning PaliGemma on this dataset and sampling data using ratios of the inverse of the size of task datasets, we achieved the following results across all tasks: 90.32% accuracy on TB diagnosis and 98.95% on close-ended VQA, 41.3 BLEU score on report generation, and a mAP of 19.4 and 16.0 on object detection and segmentation, respectively. These results demonstrate that PaliGemma-CXR effectively leverages the interdependence between multiple image interpretation tasks to enhance performance. 

---
# PreMind: Multi-Agent Video Understanding for Advanced Indexing of Presentation-style Videos 

**Authors**: Kangda Wei, Zhengyu Zhou, Bingqing Wang, Jun Araki, Lukas Lange, Ruihong Huang, Zhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00162)  

**Abstract**: In recent years, online lecture videos have become an increasingly popular resource for acquiring new knowledge. Systems capable of effectively understanding/indexing lecture videos are thus highly desirable, enabling downstream tasks like question answering to help users efficiently locate specific information within videos. This work proposes PreMind, a novel multi-agent multimodal framework that leverages various large models for advanced understanding/indexing of presentation-style videos. PreMind first segments videos into slide-presentation segments using a Vision-Language Model (VLM) to enhance modern shot-detection techniques. Each segment is then analyzed to generate multimodal indexes through three key steps: (1) extracting slide visual content, (2) transcribing speech narratives, and (3) consolidating these visual and speech contents into an integrated understanding. Three innovative mechanisms are also proposed to improve performance: leveraging prior lecture knowledge to refine visual understanding, detecting/correcting speech transcription errors using a VLM, and utilizing a critic agent for dynamic iterative self-reflection in vision analysis. Compared to traditional video indexing methods, PreMind captures rich, reliable multimodal information, allowing users to search for details like abbreviations shown only on slides. Systematic evaluations on the public LPM dataset and an internal enterprise dataset are conducted to validate PreMind's effectiveness, supported by detailed analyses. 

---
# EXACT-CT: EXplainable Analysis for Crohn's and Tuberculosis using CT 

**Authors**: Shashwat Gupta, Sarthak Gupta, Akshan Agrawal, Mahim Naaz, Rajanikanth Yadav, Priyanka Bagade  

**Link**: [PDF](https://arxiv.org/pdf/2503.00159)  

**Abstract**: Crohn's disease and intestinal tuberculosis share many overlapping features such as clinical, radiological, endoscopic, and histological features - particularly granulomas, making it challenging to clinically differentiate them. Our research leverages 3D CTE scans, computer vision, and machine learning to improve this differentiation to avoid harmful treatment mismanagement such as unnecessary anti-tuberculosis therapy for Crohn's disease or exacerbation of tuberculosis with immunosuppressants. Our study proposes a novel method to identify radiologist - identified biomarkers such as VF to SF ratio, necrosis, calcifications, comb sign and pulmonary TB to enhance accuracy. We demonstrate the effectiveness by using different ML techniques on the features extracted from these biomarkers, computing SHAP on XGBoost for understanding feature importance towards predictions, and comparing against SOTA methods such as pretrained ResNet and CTFoundation. 

---
# Fed-KAN: Federated Learning with Kolmogorov-Arnold Networks for Traffic Prediction 

**Authors**: Engin Zeydan, Cristian J. Vaca-Rubio, Luis Blanco, Roberto Pereira, Marius Caus, Kapal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2503.00154)  

**Abstract**: Non-Terrestrial Networks (NTNs) are becoming a critical component of modern communication infrastructures, especially with the advent of Low Earth Orbit (LEO) satellite systems. Traditional centralized learning approaches face major challenges in such networks due to high latency, intermittent connectivity and limited bandwidth. Federated Learning (FL) is a promising alternative as it enables decentralized training while maintaining data privacy. However, existing FL models, such as Federated Learning with Multi-Layer Perceptrons (Fed-MLP), can struggle with high computational complexity and poor adaptability to dynamic NTN environments. This paper provides a detailed analysis for Federated Learning with Kolmogorov-Arnold Networks (Fed-KAN), its implementation and performance improvements over traditional FL models in NTN environments for traffic forecasting. The proposed Fed-KAN is a novel approach that utilises the functional approximation capabilities of KANs in a FL framework. We evaluate Fed-KAN compared to Fed-MLP on a traffic dataset of real satellite operator and show a significant reduction in training and test loss. Our results show that Fed-KAN can achieve a 77.39% reduction in average test loss compared to Fed-MLP, highlighting its improved performance and better generalization ability. At the end of the paper, we also discuss some potential applications of Fed-KAN within O-RAN and Fed-KAN usage for split functionalities in NTN architecture. 

---
# Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs 

**Authors**: Fakhraddin Alwajih, Abdellah El Mekki, Samar Mohamed Magdy, Abdelrahim A. Elmadany, Omer Nacar, El Moatez Billah Nagoudi, Reem Abdel-Salam, Hanin Atwany, Youssef Nafea, Abdulfattah Mohammed Yahya, Rahaf Alhamouri, Hamzah A. Alsayadi, Hiba Zayed, Sara Shatnawi, Serry Sibaee, Yasir Ech-Chammakhy, Walid Al-Dhabyani, Marwa Mohamed Ali, Imen Jarraya, Ahmed Oumar El-Shangiti, Aisha Alraeesi, Mohammed Anwar Al-Ghrawi, Abdulrahman S. Al-Batati, Elgizouli Mohamed, Noha Taha Elgindi, Muhammed Saeed, Houdaifa Atou, Issam Ait Yahia, Abdelhak Bouayad, Mohammed Machrouh, Amal Makouar, Dania Alkawi, Mukhtar Mohamed, Safaa Taher Abdelfadil, Amine Ziad Ounnoughene, Rouabhia Anfel, Rwaa Assi, Ahmed Sorkatti, Mohamedou Cheikh Tourad, Anis Koubaa, Ismail Berrada, Mustafa Jarrar, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00151)  

**Abstract**: As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available. 

---
# Learner and Instructor Needs in AI-Supported Programming Learning Tools: Design Implications for Features and Adaptive Control 

**Authors**: Zihan Wu, Yicheng Tang, Barbara Ericson  

**Link**: [PDF](https://arxiv.org/pdf/2503.00144)  

**Abstract**: AI-supported tools can help learners overcome challenges in programming education by providing adaptive assistance. However, existing research often focuses on individual tools rather than deriving broader design recommendations. A key challenge in designing these systems is balancing learner control with system-driven guidance. To explore user preferences for AI-supported programming learning tools, we conducted a participatory design study with 15 undergraduate novice programmers and 10 instructors to gather insights on their desired help features and control preferences, as well as a follow-up survey with 172 introductory programming students.
Our qualitative findings show that learners prefer help that is encouraging, incorporates visual aids, and includes peer-related insights, whereas instructors prioritize scaffolding that reflects learners' progress and reinforces best practices. Both groups favor shared control, though learners generally prefer more autonomy, while instructors lean toward greater system guidance to prevent cognitive overload. Additionally, our interviews revealed individual differences in control preferences.
Based on our findings, we propose design guidelines for AI-supported programming tools, particularly regarding user-centered help features and adaptive control mechanisms. Our work contributes to the human-centered design of AI-supported learning environments by informing the development of systems that effectively balance autonomy and guidance, enhancing AI-supported educational tools for programming and beyond. 

---
# AnnoCaseLaw: A Richly-Annotated Dataset For Benchmarking Explainable Legal Judgment Prediction 

**Authors**: Magnus Sesodia, Alina Petrova, John Armour, Thomas Lukasiewicz, Oana-Maria Camburu, Puneet K. Dokania, Philip Torr, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00128)  

**Abstract**: Legal systems worldwide continue to struggle with overwhelming caseloads, limited judicial resources, and growing complexities in legal proceedings. Artificial intelligence (AI) offers a promising solution, with Legal Judgment Prediction (LJP) -- the practice of predicting a court's decision from the case facts -- emerging as a key research area. However, existing datasets often formulate the task of LJP unrealistically, not reflecting its true difficulty. They also lack high-quality annotation essential for legal reasoning and explainability. To address these shortcomings, we introduce AnnoCaseLaw, a first-of-its-kind dataset of 471 meticulously annotated U.S. Appeals Court negligence cases. Each case is enriched with comprehensive, expert-labeled annotations that highlight key components of judicial decision making, along with relevant legal concepts. Our dataset lays the groundwork for more human-aligned, explainable LJP models. We define three legally relevant tasks: (1) judgment prediction; (2) concept identification; and (3) automated case annotation, and establish a performance baseline using industry-leading large language models (LLMs). Our results demonstrate that LJP remains a formidable task, with application of legal precedent proving particularly difficult. Code and data are available at this https URL. 

---
# Evaluation of LLMs-based Hidden States as Author Representations for Psychological Human-Centered NLP Tasks 

**Authors**: Nikita Soni, Pranav Chitale, Khushboo Singh, Niranjan Balasubramanian, H. Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2503.00124)  

**Abstract**: Like most of NLP, models for human-centered NLP tasks -- tasks attempting to assess author-level information -- predominantly use representations derived from hidden states of Transformer-based LLMs. However, what component of the LM is used for the representation varies widely. Moreover, there is a need for Human Language Models (HuLMs) that implicitly model the author and provide a user-level hidden state. Here, we systematically evaluate different ways of representing documents and users using different LM and HuLM architectures to predict task outcomes as both dynamically changing states and averaged trait-like user-level attributes of valence, arousal, empathy, and distress. We find that representing documents as an average of the token hidden states performs the best generally. Further, while a user-level hidden state itself is rarely the best representation, we find its inclusion in the model strengthens token or document embeddings used to derive document- and user-level representations resulting in best performances. 

---
# BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology 

**Authors**: Ludovico Mitchener, Jon M Laurent, Benjamin Tenmann, Siddharth Narayanan, Geemi P Wellawatte, Andrew White, Lorenzo Sani, Samuel G Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2503.00096)  

**Abstract**: Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery. 

---
# Rethinking LLM Bias Probing Using Lessons from the Social Sciences 

**Authors**: Kirsten N. Morehouse, Siddharth Swaroop, Weiwei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00093)  

**Abstract**: The proliferation of LLM bias probes introduces three significant challenges: (1) we lack principled criteria for choosing appropriate probes, (2) we lack a system for reconciling conflicting results across probes, and (3) we lack formal frameworks for reasoning about when (and why) probe results will generalize to real user behavior. We address these challenges by systematizing LLM social bias probing using actionable insights from social sciences. We then introduce EcoLevels - a framework that helps (a) determine appropriate bias probes, (b) reconcile conflicting findings across probes, and (c) generate predictions about bias generalization. Overall, we ground our analysis in social science research because many LLM probes are direct applications of human probes, and these fields have faced similar challenges when studying social bias in humans. Based on our work, we suggest how the next generation of LLM bias probing can (and should) benefit from decades of social science research. 

---
# EdgeAIGuard: Agentic LLMs for Minor Protection in Digital Spaces 

**Authors**: Ghulam Mujtaba, Sunder Ali Khowaja, Kapal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2503.00092)  

**Abstract**: Social media has become integral to minors' daily lives and is used for various purposes, such as making friends, exploring shared interests, and engaging in educational activities. However, the increase in screen time has also led to heightened challenges, including cyberbullying, online grooming, and exploitations posed by malicious actors. Traditional content moderation techniques have proven ineffective against exploiters' evolving tactics. To address these growing challenges, we propose the EdgeAIGuard content moderation approach that is designed to protect minors from online grooming and various forms of digital exploitation. The proposed method comprises a multi-agent architecture deployed strategically at the network edge to enable rapid detection with low latency and prevent harmful content targeting minors. The experimental results show the proposed method is significantly more effective than the existing approaches. 

---
# Protein Structure Tokenization: Benchmarking and New Recipe 

**Authors**: Xinyu Yuan, Zichen Wang, Marcus Collins, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2503.00089)  

**Abstract**: Recent years have witnessed a surge in the development of protein structural tokenization methods, which chunk protein 3D structures into discrete or continuous representations. Structure tokenization enables the direct application of powerful techniques like language modeling for protein structures, and large multimodal models to integrate structures with protein sequences and functional texts. Despite the progress, the capabilities and limitations of these methods remain poorly understood due to the lack of a unified evaluation framework. We first introduce StructTokenBench, a framework that comprehensively evaluates the quality and efficiency of structure tokenizers, focusing on fine-grained local substructures rather than global structures, as typical in existing benchmarks. Our evaluations reveal that no single model dominates all benchmarking perspectives. Observations of codebook under-utilization led us to develop AminoAseed, a simple yet effective strategy that enhances codebook gradient updates and optimally balances codebook size and dimension for improved tokenizer utilization and quality. Compared to the leading model ESM3, our method achieves an average of 6.31% performance improvement across 24 supervised tasks, with sensitivity and utilization rates increased by 12.83% and 124.03%, respectively. 

---
# Generalization of CNNs on Relational Reasoning with Bar Charts 

**Authors**: Zhenxing Cui, Lu Chen, Yunhai Wang, Daniel Haehn, Yong Wang, Hanspeter Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2503.00086)  

**Abstract**: This paper presents a systematic study of the generalization of convolutional neural networks (CNNs) and humans on relational reasoning tasks with bar charts. We first revisit previous experiments on graphical perception and update the benchmark performance of CNNs. We then test the generalization performance of CNNs on a classic relational reasoning task: estimating bar length ratios in a bar chart, by progressively perturbing the standard visualizations. We further conduct a user study to compare the performance of CNNs and humans. Our results show that CNNs outperform humans only when the training and test data have the same visual encodings. Otherwise, they may perform worse. We also find that CNNs are sensitive to perturbations in various visual encodings, regardless of their relevance to the target bars. Yet, humans are mainly influenced by bar lengths. Our study suggests that robust relational reasoning with visualizations is challenging for CNNs. Improving CNNs' generalization performance may require training them to better recognize task-related visual properties. 

---
# InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation 

**Authors**: Chong Zhang, Yukun Ma, Qian Chen, Wen Wang, Shengkui Zhao, Zexu Pan, Hao Wang, Chongjia Ni, Trung Hieu Nguyen, Kun Zhou, Yidi Jiang, Chaohong Tan, Zhifu Gao, Zhihao Du, Bin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.00084)  

**Abstract**: We introduce InspireMusic, a framework integrated super resolution and large language model for high-fidelity long-form music generation. A unified framework generates high-fidelity music, songs, and audio, which incorporates an autoregressive transformer with a super-resolution flow-matching model. This framework enables the controllable generation of high-fidelity long-form music at a higher sampling rate from both text and audio prompts. Our model differs from previous approaches, as we utilize an audio tokenizer with one codebook that contains richer semantic information, thereby reducing training costs and enhancing efficiency. This combination enables us to achieve high-quality audio generation with long-form coherence of up to $8$ minutes. Then, an autoregressive transformer model based on Qwen 2.5 predicts audio tokens. Next, we employ a super-resolution flow-matching model to generate high-sampling rate audio with fine-grained details learned from an acoustic codec model. Comprehensive experiments show that the InspireMusic-1.5B-Long model has a comparable performance to recent top-tier open-source systems, including MusicGen and Stable Audio 2.0, on subjective and objective evaluations. The code and pre-trained models are released at this https URL. 

---
# Experiences with Content Development and Assessment Design in the Era of GenAI 

**Authors**: Aakanksha Sharma, Samar Shailendra, Rajan Kadel  

**Link**: [PDF](https://arxiv.org/pdf/2503.00081)  

**Abstract**: Generative Artificial Intelligence (GenAI) has the potential to transform higher education by generating human-like content. The advancement in GenAI has revolutionised several aspects of education, especially subject and assessment design. In this era, it is crucial to design assessments that challenge students and cannot be solved using GenAI tools. This makes it necessary to update the educational content with rapidly evolving technology. The assessment plays a significant role in ensuring the students learning, as it encourages students to engage actively, leading to the achievement of learning outcomes. The paper intends to determine how effectively GenAI can design a subject, including lectures, labs and assessments, using prompts and custom-based training. This paper aims to elucidate the direction to educators so they can leverage GenAI to create subject content. Additionally, we provided our experiential learning for educators to develop content, highlighting the importance of prompts and fine-tuning to ensure output quality. It has also been observed that expert evaluation is essential for assessing the quality of GenAI-generated materials throughout the content generation process. 

---
# AI Literacy in K-12 and Higher Education in the Wake of Generative AI: An Integrative Review 

**Authors**: Xingjian, Barbara J. Ericson  

**Link**: [PDF](https://arxiv.org/pdf/2503.00079)  

**Abstract**: Even though AI literacy has emerged as a prominent education topic in the wake of generative AI, its definition remains vague. There is little consensus among researchers and practitioners on how to discuss and design AI literacy interventions. The term has been used to describe both learning activities that train undergraduate students to use ChatGPT effectively and having kindergarten children interact with social robots. This paper applies an integrative review method to examine empirical and theoretical AI literacy studies published since 2020. In synthesizing the 124 reviewed studies, three ways to conceptualize literacy-functional, critical, and indirectly beneficial-and three perspectives on AI-technical detail, tool, and sociocultural-were identified, forming a framework that reflects the spectrum of how AI literacy is approached in practice. The framework highlights the need for more specialized terms within AI literacy discourse and indicates research gaps in certain AI literacy objectives. 

---
# Navigating the Edge with the State-of-the-Art Insights into Corner Case Identification and Generation for Enhanced Autonomous Vehicle Safety 

**Authors**: Gabriel Kenji Godoy Shimanuki, Alexandre Moreira Nascimento, Lucio Flavio Vismari, Joao Batista Camargo Junior, Jorge Rady de Almeida Junior, Paulo Sergio Cugnasca  

**Link**: [PDF](https://arxiv.org/pdf/2503.00077)  

**Abstract**: In recent years, there has been significant development of autonomous vehicle (AV) technologies. However, despite the notable achievements of some industry players, a strong and appealing body of evidence that demonstrate AVs are actually safe is lacky, which could foster public distrust in this technology and further compromise the entire development of this industry, as well as related social impacts. To improve the safety of AVs, several techniques are proposed that use synthetic data in virtual simulation. In particular, the highest risk data, known as corner cases (CCs), are the most valuable for developing and testing AV controls, as they can expose and improve the weaknesses of these autonomous systems. In this context, the present paper presents a systematic literature review aiming to comprehensively analyze methodologies for CC identifi cation and generation, also pointing out current gaps and further implications of synthetic data for AV safety and reliability. Based on a selection criteria, 110 studies were picked from an initial sample of 1673 papers. These selected paper were mapped into multiple categories to answer eight inter-linked research questions. It concludes with the recommendation of a more integrated approach focused on safe development among all stakeholders, with active collaboration between industry, academia and regulatory bodies. 

---
# Systematic Review of Cybersecurity in Banking: Evolution from Pre-Industry 4.0 to Post-Industry 4.0 in Artificial Intelligence, Blockchain, Policies and Practice 

**Authors**: Tue Nhi Tran  

**Link**: [PDF](https://arxiv.org/pdf/2503.00070)  

**Abstract**: Throughout the history from pre-industry 4.0 to post-industry 4.0, cybersecurity at banks has undergone significant changes. Pre-industry 4.0 cyber security at banks relied on individual security methods that were highly manual and had low accuracy. When moving to post-industry 4.0, cybersecurity at banks had a major turning point with security methods that combined different technologies such as Artificial Intelligence (AI), Blockchain, IoT, automating necessary processes and significantly increasing the defence layer for banks. However, along with the development of new technologies, the current challenge of cybersecurity at banks lies in scalability, high costs and resources in both money and time for R&D of defence methods along with the threat of high-tech cybercriminals growing and expanding. This report goes from introducing the importance of cybersecurity at banks, analyzing their management, operational and business objectives, evaluating pre-industry 4.0 technologies used for cybersecurity at banks to assessing post-industry 4.0 technologies focusing on Artificial Intelligence and Blockchain, discussing current policies and practices and ending with discussing key advantages and challenges for 4.0 technologies and recommendations for further developing cybersecurity at banks. 

---
# Societal Alignment Frameworks Can Improve LLM Alignment 

**Authors**: Karolina Stańczak, Nicholas Meade, Mehar Bhatia, Hattie Zhou, Konstantin Böttinger, Jeremy Barnes, Jason Stanley, Jessica Montgomery, Richard Zemel, Nicolas Papernot, Nicolas Chapados, Denis Therien, Timothy P. Lillicrap, Ana Marasović, Sylvie Delacroix, Gillian K. Hadfield, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00069)  

**Abstract**: Recent progress in large language models (LLMs) has focused on producing responses that meet human expectations and align with shared values - a process coined alignment. However, aligning LLMs remains challenging due to the inherent disconnect between the complexity of human values and the narrow nature of the technological approaches designed to address them. Current alignment methods often lead to misspecified objectives, reflecting the broader issue of incomplete contracts, the impracticality of specifying a contract between a model developer, and the model that accounts for every scenario in LLM alignment. In this paper, we argue that improving LLM alignment requires incorporating insights from societal alignment frameworks, including social, economic, and contractual alignment, and discuss potential solutions drawn from these domains. Given the role of uncertainty within societal alignment frameworks, we then investigate how it manifests in LLM alignment. We end our discussion by offering an alternative view on LLM alignment, framing the underspecified nature of its objectives as an opportunity rather than perfect their specification. Beyond technical improvements in LLM alignment, we discuss the need for participatory alignment interface designs. 

---
# SAC-ViT: Semantic-Aware Clustering Vision Transformer with Early Exit 

**Authors**: Youbing Hu, Yun Cheng, Anqi Lu, Dawei Wei, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00060)  

**Abstract**: The Vision Transformer (ViT) excels in global modeling but faces deployment challenges on resource-constrained devices due to the quadratic computational complexity of its attention mechanism. To address this, we propose the Semantic-Aware Clustering Vision Transformer (SAC-ViT), a non-iterative approach to enhance ViT's computational efficiency. SAC-ViT operates in two stages: Early Exit (EE) and Semantic-Aware Clustering (SAC). In the EE stage, downsampled input images are processed to extract global semantic information and generate initial inference results. If these results do not meet the EE termination criteria, the information is clustered into target and non-target tokens. In the SAC stage, target tokens are mapped back to the original image, cropped, and embedded. These target tokens are then combined with reused non-target tokens from the EE stage, and the attention mechanism is applied within each cluster. This two-stage design, with end-to-end optimization, reduces spatial redundancy and enhances computational efficiency, significantly boosting overall ViT performance. Extensive experiments demonstrate the efficacy of SAC-ViT, reducing 62% of the FLOPs of DeiT and achieving 1.98 times throughput without compromising performance. 

---
# African Gender Classification Using Clothing Identification Via Deep Learning 

**Authors**: Samuel Ozechi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00058)  

**Abstract**: Human attribute identification and classification are crucial in computer vision, driving the development of innovative recognition systems. Traditional gender classification methods primarily rely on facial recognition, which, while effective, struggles under non-ideal conditions such as blurriness, side views, or partial occlusions. This study explores an alternative approach by leveraging clothing identification, specifically focusing on African traditional attire, which carries culturally significant and gender-specific features.
We use the AFRIFASHION1600 dataset, a curated collection of 1,600 images of African traditional clothing labeled into two gender classes: male and female. A deep learning model, based on a modified VGG16 architecture and trained using transfer learning, was developed for classification. Data augmentation was applied to address the challenges posed by the relatively small dataset and to mitigate overfitting. The model achieved an accuracy of 87% on the test set, demonstrating strong predictive capability despite dataset imbalances favoring female samples.
These findings highlight the potential of clothing-based identification as a complementary technique to facial recognition for gender classification in African contexts. Future research should focus on expanding and balancing datasets to enhance classification robustness and improve the applicability of clothing-based gender recognition systems. 

---
# Deciphering the complaint aspects: Towards an aspect-based complaint identification model with video complaint dataset in finance 

**Authors**: Sarmistha Das, Basha Mujavarsheik, R E Zera Lyngkhoi, Sriparna Saha, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2503.00054)  

**Abstract**: In today's competitive marketing landscape, effective complaint management is crucial for customer service and business success. Video complaints, integrating text and image content, offer invaluable insights by addressing customer grievances and delineating product benefits and drawbacks. However, comprehending nuanced complaint aspects within vast daily multimodal financial data remains a formidable challenge. Addressing this gap, we have curated a proprietary multimodal video complaint dataset comprising 433 publicly accessible instances. Each instance is meticulously annotated at the utterance level, encompassing five distinct categories of financial aspects and their associated complaint labels. To support this endeavour, we introduce Solution 3.0, a model designed for multimodal aspect-based complaint identification task. Solution 3.0 is tailored to perform three key tasks: 1) handling multimodal features ( audio and video), 2) facilitating multilabel aspect classification, and 3) conducting multitasking for aspect classifications and complaint identification parallelly. Solution 3.0 utilizes a CLIP-based dual frozen encoder with an integrated image segment encoder for global feature fusion, enhanced by contextual attention (ISEC) to improve accuracy and efficiency. Our proposed framework surpasses current multimodal baselines, exhibiting superior performance across nearly all metrics by opening new ways to strengthen appropriate customer care initiatives and effectively assisting individuals in resolving their problems. 

---
# AI and Semantic Communication for Infrastructure Monitoring in 6G-Driven Drone Swarms 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.00053)  

**Abstract**: The adoption of unmanned aerial vehicles to monitor critical infrastructure is gaining momentum in various industrial domains. Organizational imperatives drive this progression to minimize expenses, accelerate processes, and mitigate hazards faced by inspection personnel. However, traditional infrastructure monitoring systems face critical bottlenecks-5G networks lack the latency and reliability for large-scale drone coordination, while manual inspections remain costly and slow. We propose a 6G-enabled drone swarm system that integrates ultra-reliable, low-latency communications, edge AI, and semantic communication to automate inspections. By adopting LLMs for structured output and report generation, our framework is hypothesized to reduce inspection costs and improve fault detection speed compared to existing methods. 

---
# RURA-Net: A general disease diagnosis method based on Zero-Shot Learning 

**Authors**: Yan Su, Qiulin Wu, Weizhen Li, Chengchang Pan, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00052)  

**Abstract**: The training of deep learning models relies on a large amount of labeled data. However, the high cost of medical labeling seriously hinders the development of deep learning in the medical field. Our study proposes a general disease diagnosis approach based on Zero-Shot Learning. The Siamese neural network is used to find similar diseases for the target diseases, and the U-Net segmentation model is used to accurately segment the key lesions of the disease. Finally, based on the ResNet-Agglomerative clustering algorithm, a clustering model is trained on a large number of sample data of similar diseases to obtain a approximate diagnosis of the target disease. Zero-Shot Learning of the target disease is then successfully achieved. To evaluate the validity of the model, we validated our method on a dataset of ophthalmic diseases in CFP modality. The external dataset was used to test its performance, and the accuracy=0.8395, precision=0.8094, recall=0.8463, F1 Score=0.8274, AUC=0.9226, which exceeded the indexes of most Few-Shot Learning and One-Shot Learning models. It proves that our method has great potential and reference value in the medical field, where annotation data is usually scarce and expensive to obtain. 

---
# Omni-SILA: Towards Omni-scene Driven Visual Sentiment Identifying, Locating and Attributing in Videos 

**Authors**: Jiamin Luo, Jingjing Wang, Junxiao Ma, Yujie Jin, Shoushan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00049)  

**Abstract**: Prior studies on Visual Sentiment Understanding (VSU) primarily rely on the explicit scene information (e.g., facial expression) to judge visual sentiments, which largely ignore implicit scene information (e.g., human action, objection relation and visual background), while such information is critical for precisely discovering visual sentiments. Motivated by this, this paper proposes a new Omni-scene driven visual Sentiment Identifying, Locating and Attributing in videos (Omni-SILA) task, aiming to interactively and precisely identify, locate and attribute visual sentiments through both explicit and implicit scene information. Furthermore, this paper believes that this Omni-SILA task faces two key challenges: modeling scene and highlighting implicit scene beyond explicit. To this end, this paper proposes an Implicit-enhanced Causal MoE (ICM) approach for addressing the Omni-SILA task. Specifically, a Scene-Balanced MoE (SBM) and an Implicit-Enhanced Causal (IEC) blocks are tailored to model scene information and highlight the implicit scene information beyond explicit, respectively. Extensive experimental results on our constructed explicit and implicit Omni-SILA datasets demonstrate the great advantage of the proposed ICM approach over advanced Video-LLMs. 

---
# Leveraging Large Models for Evaluating Novel Content: A Case Study on Advertisement Creativity 

**Authors**: Zhaoyi Joey Hou, Adriana Kovashka, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00046)  

**Abstract**: Evaluating creativity is challenging, even for humans, not only because of its subjectivity but also because it involves complex cognitive processes. Inspired by work in marketing, we attempt to break down visual advertisement creativity into atypicality and originality. With fine-grained human annotations on these dimensions, we propose a suit of tasks specifically for such a subjective problem. We also evaluate the alignment between state-of-the-art (SoTA) vision language models (VLM) and humans on our proposed benchmark, demonstrating both the promises and challenges of using VLMs for automatic creativity assessment. 

---
# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning 

**Authors**: Nilay Yilmaz, Maitreya Patel, Yiran Lawrence Luo, Tejas Gokhale, Chitta Baral, Suren Jayasuriya, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00043)  

**Abstract**: Multimodal Large Language Models (MLLMs) have become a powerful tool for integrating visual and textual information. Despite their exceptional performance on visual understanding benchmarks, measuring their ability to reason abstractly across multiple images remains a significant challenge. To address this, we introduce VOILA, a large-scale, open-ended, dynamic benchmark designed to evaluate MLLMs' perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices. Our experiments demonstrate that the analogical reasoning tasks in VOILA present a challenge to MLLMs. Through multi-step analysis, we reveal that current MLLMs struggle to comprehend inter-image relationships and exhibit limited capabilities in high-level relational reasoning. Notably, we observe that performance improves when following a multi-step strategy of least-to-most prompting. Comprehensive evaluations on open-source models and GPT-4o show that on text-based answers, the best accuracy for challenging scenarios is 13% (LLaMa 3.2) and even for simpler tasks is only 29% (GPT-4o), while human performance is significantly higher at 70% across both difficulty levels. 

---
# An Analysis of Segment Anything 2 

**Authors**: Clayton Bromley, Alexander Moore, Amar Saini, Doug Poland, Carmen Carrano  

**Link**: [PDF](https://arxiv.org/pdf/2503.00042)  

**Abstract**: Video object segmentation (VOS) is a critical task in the development of video perception and understanding. The Segment-Anything Model 2 (SAM 2), released by Meta AI, is the current state-of-the-art architecture for end-to-end VOS. SAM 2 performs very well on both clean video data and augmented data, and completely intelligent video perception requires an understanding of how this architecture is capable of achieving such quality results. To better understand how each step within the SAM 2 architecture permits high-quality video segmentation, we pass a variety of complex video transformations through the architecture and measure the impact at each stage of the process. We observe that each progressive stage enables the filtering of complex transformation noise and the emphasis of the object of interest. Our contributions include the creation of complex transformation video datasets, an analysis of how each stage of the SAM 2 architecture interprets these transformations, and visualizations of segmented objects through each stage. By better understanding how each model structure impacts overall video understanding, VOS development can work to improve real-world applicability and performance tracking, localizing, and segmenting objects despite complex cluttered scenes and obscurations. 

---
# from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors 

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Qi Li, Jiangyu Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.00038)  

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. 

---
# Zero-Shot Defense Against Toxic Images via Inherent Multimodal Alignment in LVLMs 

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00037)  

**Abstract**: Large Vision-Language Models (LVLMs) have made significant strides in multimodal comprehension, thanks to extensive pre-training and fine-tuning on large-scale visual datasets. However, despite their robust textual safety mechanisms, they remain vulnerable to harmful visual inputs. Existing safeguards-typically relying on pre-filtering or fine-tuning-incur high costs and diminish overall utility. To address this critical vulnerability, we introduce SafeCLIP, a lightweight method that leverages LVLMs inherent multimodal alignment for zero-shot toxic image detection. By projecting CLIPs discarded CLS token into its text space and matching it with toxic descriptors, SafeCLIP detects harmful content without any architectural changes-adding minimal latency and enabling dynamic safety corrections during inference and this http URL show that SafeCLIP achieves a 66.9% defense success rate with only 3.2% false positive rate and 7.2% overhead. In contrast, state-of-the-art methods achieve 52.9% success but have a 10.7% false positive rate and 210% overhead. Our work demonstrates that leveraging inherent multimodal alignment can yield efficient, low-cost LVLM safety. Code is available at this http URL. 

---
# A Novel Spatiotemporal Correlation Anomaly Detection Method Based on Time-Frequency-Domain Feature Fusion and a Dynamic Graph Neural Network in Wireless Sensor Network 

**Authors**: Miao Ye, Zhibang Jiang, Xingsi Xue, Xingwang Li, Peng Wen, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00036)  

**Abstract**: Attention-based transformers have played an important role in wireless sensor network (WSN) timing anomaly detection due to their ability to capture long-term dependencies. However, there are several issues that must be addressed, such as the fact that their ability to capture long-term dependencies is not completely reliable, their computational complexity levels are high, and the spatiotemporal features of WSN timing data are not sufficiently extracted for detecting the correlation anomalies of multinode WSN timing data. To address these limitations, this paper proposes a WSN anomaly detection method that integrates frequency-domain features with dynamic graph neural networks (GNN) under a designed self-encoder reconstruction framework. First, the discrete wavelet transform effectively decomposes trend and seasonal components of time series to solve the poor long-term reliability of transformers. Second, a frequency-domain attention mechanism is designed to make full use of the difference between the amplitude distributions of normal data and anomalous data in this domain. Finally, a multimodal fusion-based dynamic graph convolutional network (MFDGCN) is designed by combining an attention mechanism and a graph convolutional network (GCN) to adaptively extract spatial correlation features. A series of experiments conducted on public datasets and their results demonstrate that the anomaly detection method designed in this paper exhibits superior precision and recall than the existing methods do, with an F1 score of 93.5%, representing an improvement of 2.9% over that of the existing models. 

---
# Constraining Sequential Model Editing with Editing Anchor Compression 

**Authors**: Hao-Xiang Xu, Jun-Yu Ma, Zhen-Hua Ling, Ningyu Zhang, Jia-Chen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00035)  

**Abstract**: Large language models (LLMs) struggle with hallucinations due to false or outdated knowledge. Given the high resource demands of retraining these models, there is an increasing focus on developing model editing. However, the general abilities of LLMs across downstream tasks are prone to significant degradation during sequential editing. This paper statistically observes that the parameter matrix after editing exhibits a significant deviation compared to its previous state as the number of edits increases. This serious deviation affects the original knowledge associations within LLMs and leads to the degradation of their general abilities. To this end, a framework termed Editing Anchor Compression (EAC) is proposed to constrain the deviation of the parameter matrix during sequential editing. It compresses the editing information by selecting editing anchors that are important in encoding new relations without deviating too much from the original matrix, thereby preserving the general abilities. Experiments of applying EAC to two popular editing methods on three LLMs across four tasks are conducted. Evaluation results show that EAC effectively minimizes unreasonable deviations caused by model editing, preserving over 70% of the general abilities while better retaining the editing knowledge compared to the original counterpart methods. 

---
# MergeIT: From Selection to Merging for Efficient Instruction Tuning 

**Authors**: Hongyi Cai, Yuqian Fu, Hongming Fu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00034)  

**Abstract**: Instruction tuning is crucial for optimizing Large Language Models (LLMs), yet mainstream data selection methods heavily rely on LLMs as instruction quality scorers, leading to high computational costs and reduced data diversity. To address these limitations, we propose MergeIT, a novel LLM-based Merging strategy for better Instruction Tuning that shifts the focus from selection to synthesis. MergeIT operates in two stages: first, topic-aware filtering clusters and refines the dataset, preserving diversity while eliminating redundancy without relying on LLM-based scoring. Second, LLM-based merging synthesizes semantically similar instructions into more informative and compact training data, enhancing data richness while further reducing dataset size. Experimental results demonstrate that MergeIT enables efficient, diverse, and scalable instruction selection and synthesis, establishing LLM-based merging as a promising alternative to conventional scoring-based selection methods for instruction tuning. Our source code and datasets are now available at this https URL 

---
# Detecting LLM-Generated Korean Text through Linguistic Feature Analysis 

**Authors**: Shinwoo Park, Shubin Kim, Do-Kyung Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00032)  

**Abstract**: The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres.
By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. 

---
# Efficient Test-Time Scaling via Self-Calibration 

**Authors**: Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00031)  

**Abstract**: Increasing test-time computation is a straightforward approach to enhancing the quality of responses in Large Language Models (LLMs). While Best-of-N sampling and Self-Consistency with majority voting are simple and effective, they require a fixed number of sampling responses for each query, regardless of its complexity. This could result in wasted computation for simpler questions and insufficient exploration for more challenging ones. In this work, we argue that model confidence of responses can be used for improving the efficiency of test-time scaling. Unfortunately, LLMs are known to be overconfident and provide unreliable confidence estimation. To address this limitation, we introduce Self-Calibration by distilling Self-Consistency-derived confidence into the model itself. This enables reliable confidence estimation at test time with one forward pass. We then design confidence-based efficient test-time scaling methods to handle queries of various difficulty, such as Early-Stopping for Best-of-N and Self-Consistency with calibrated confidence. Experiments on three LLMs across six datasets demonstrate the effectiveness of our approach. Specifically, applying confidence-based Early Stopping to Best-of-N improves MathQA accuracy from 81.0 to 83.6 with a sample budget of 16 responses, indicating the efficacy of confidence-based sampling strategy at inference time. 

---
# Game-Theoretic Regularized Self-Play Alignment of Large Language Models 

**Authors**: Xiaohang Tang, Sangwoong Yoon, Seongho Son, Huizhuo Yuan, Quanquan Gu, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.00030)  

**Abstract**: Self-play alignment algorithms have been developed as effective methods for fine-tuning large language models (LLMs), formulating preference optimization as a two-player game. However, the regularization with respect to the reference policy, which is crucial for mitigating over-optimization, has been insufficiently investigated in self-play alignment. In this paper, we show that our regularization method can improve the unregularized self-play significantly. To study the impact of different regularizations in self-play alignment, we propose Regularized Self-Play Policy Optimization (RSPO). This generalized framework regularizes the self-play by simply adding a chosen regularization term into the loss while maintaining provable last-iterate convergence to the Nash Equilibrium of the corresponding regularized game. Surprisingly, empirical evaluations using the Mistral-7B-Instruct base model reveal that forward KL divergence regularization reduces response length in RSPO, whereas reverse KL divergence markedly improves raw win rates. RSPO with a linear combination of forward and reverse KL divergence regularization substantially increases the length-controlled win rate in AlpacaEval-2, elevating the unregularized self-play alignment method (SPPO) from $28.53\%$ to $35.44\%$. Finally, we show that RSPO also improves the response diversity. 

---
# Streaming Looking Ahead with Token-level Self-reward 

**Authors**: Hongming Zhang, Ruixin Hong, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00029)  

**Abstract**: Autoregressive decoding algorithms that use only past information often cannot guarantee the best performance. Recently, people discovered that looking-ahead algorithms such as Monte Carlo Tree Search (MCTS) with external reward models (RMs) can significantly improve models' output by allowing them to think ahead and leverage future outputs and associated rewards to guide the current generation. Such techniques can help the reinforcement fine-tuning phase by sampling better trajectories and the inference phase by selecting the better output. However, their high computational cost limits their applications, especially in streaming scenarios. To address this issue, we propose equipping the policy model with token-level self-reward modeling (TRM) capability to eliminate the need for external models and extra communication. We name the new architecture as Reward Transformer. In addition, we propose a streaming-looking-ahead (SLA) algorithm to further boost search efficiency with better parallelization. Experiments show that SLA achieves an overall win rate of 79.7\% against the baseline greedy decoding algorithm on three general-domain datasets with a frozen policy model while maintaining streaming efficiency. If we combine SLA with reinforcement fine-tuning techniques such as DPO, SLA achieves an overall win rate of 89.4\%. 

---
# Genetics-Driven Personalized Disease Progression Model 

**Authors**: Haoyu Yang, Sanjoy Dey, Pablo Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00028)  

**Abstract**: Modeling disease progression through multiple stages is critical for clinical decision-making for chronic diseases, e.g., cancer, diabetes, chronic kidney diseases, and so on. Existing approaches often model the disease progression as a uniform trajectory pattern at the population level. However, chronic diseases are highly heterogeneous and often have multiple progression patterns depending on a patient's individual genetics and environmental effects due to lifestyles. We propose a personalized disease progression model to jointly learn the heterogeneous progression patterns and groups of genetic profiles. In particular, an end-to-end pipeline is designed to simultaneously infer the characteristics of patients from genetic markers using a variational autoencoder and how it drives the disease progressions using an RNN-based state-space model based on clinical observations. Our proposed model shows improvement on real-world and synthetic clinical data. 

---
# Evaluating Large Language Models on the Spanish Medical Intern Resident (MIR) Examination 2024/2025:A Comparative Analysis of Clinical Reasoning and Knowledge Application 

**Authors**: Carlos Luengo Vera, Ignacio Ferro Picon, M. Teresa del Val Nunez, Jose Andres Gomez Gandia, Antonio de Lucas Ancillo, Victor Ramos Arroyo, Carlos Milan Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00025)  

**Abstract**: This study presents a comparative evaluation of 22 large language models LLMs on the Spanish Medical Intern Resident MIR examinations for 2024 and 2025 with a focus on clinical reasoning domain specific expertise and multimodal processing capabilities The MIR exam consisting of 210 multiple choice questions some requiring image interpretation serves as a stringent benchmark for assessing both factual recall and complex clinical problem solving skills Our investigation encompasses general purpose models such as GPT4 Claude LLaMA and Gemini as well as specialized fine tuned systems like Miri Pro which leverages proprietary Spanish healthcare data to excel in medical contexts
Recent market entries Deepseek and Grok have further enriched the evaluation landscape particularly for tasks that demand advanced visual and semantic analysis The findings indicate that while general purpose LLMs perform robustly overall fine tuned models consistently achieve superior accuracy especially in addressing nuanced domain specific challenges A modest performance decline observed between the two exam cycles appears attributable to the implementation of modified questions designed to mitigate reliance on memorization
The results underscore the transformative potential of domain specific fine tuning and multimodal integration in advancing medical AI applications They also highlight critical implications for the future integration of LLMs into medical education training and clinical decision making emphasizing the importance of balancing automated reasoning with ethical and context aware judgment 

---
# KVCrush: Key value cache size-reduction using similarity in head-behaviour 

**Authors**: Gopi Krishna Jha, Sameh Gobriel, Liubov Talamanova, Alexander Kozlov, Nilesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2503.00022)  

**Abstract**: Key-value (KV) caching has emerged as a crucial optimization technique for accelerating inference in large language models (LLMs). By allowing the attention operation to scale linearly rather than quadratically with the total sequence length, KV caching significantly enhances generation throughput. However, due to large context lengths in the modern LLMs, the memory footprint of the KV is a huge bottleneck for model deployment directly impacting the model's batch size, hindering its ability to deliver high-throughput. Existing research addresses this challenge using several techniques, such as discarding low-attention tokens, quantization, and matrix approximation which typically lead to a negative impact on the model accuracy.
In this paper, We propose KVCrush technology which can be combined with many KV compression technologies to improve the model accuracy at a much smaller memory. KVCrush provides an alternate representation scheme for key-value states, along with a low-overhead token pruning algorithm that accounts for the token distribution in the KV cache, which in turn allows for a a smaller footprint while maintaining the accuracy of the model. Based on our results, KVCrush reduces LongBench KV Cache size by 4x with less than 1% accuracy drop and achieves state-of-the-art average accuracy with minimal overhead, incurring less than 0.5% total inference latency. KVCrush not only outperforms the accuracy of state-of-the-art importance-based token retention schemes but is also compatible with typical practical LLM deployments using KV cache paging schemes such as vLLM and mixed precision quantization. 

---
# A Systematic Review of Open Datasets Used in Text-to-Image (T2I) Gen AI Model Safety 

**Authors**: Rakeen Rouf, Trupti Bavalatti, Osama Ahmed, Dhaval Potdar, Faraz Jawed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00020)  

**Abstract**: Novel research aimed at text-to-image (T2I) generative AI safety often relies on publicly available datasets for training and evaluation, making the quality and composition of these datasets crucial. This paper presents a comprehensive review of the key datasets used in the T2I research, detailing their collection methods, compositions, semantic and syntactic diversity of prompts and the quality, coverage, and distribution of harm types in the datasets. By highlighting the strengths and limitations of the datasets, this study enables researchers to find the most relevant datasets for a use case, critically assess the downstream impacts of their work given the dataset distribution, particularly regarding model safety and ethical considerations, and also identify the gaps in dataset coverage and quality that future research may address. 

---
# Eeyore: Realistic Depression Simulation via Supervised and Preference Optimization 

**Authors**: Siyang Liu, Bianca Brie, Wenda Li, Laura Biester, Andrew Lee, James Pennebaker, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00018)  

**Abstract**: Large Language Models (LLMs) have been previously explored for mental healthcare training and therapy client simulation, but they still fall short in authentically capturing diverse client traits and psychological conditions. We introduce \textbf{Eeyore}, an 8B model optimized for realistic depression simulation through a structured alignment framework, incorporating expert input at every stage. First, we systematically curate real-world depression-related conversations, extracting depressive traits to guide data filtering and psychological profile construction, and use this dataset to instruction-tune Eeyore for profile adherence. Next, to further enhance realism, Eeyore undergoes iterative preference optimization -- first leveraging model-generated preferences and then calibrating with a small set of expert-annotated preferences. Throughout the entire pipeline, we actively collaborate with domain experts, developing interactive interfaces to validate trait extraction and iteratively refine structured psychological profiles for clinically meaningful role-play customization. Despite its smaller model size, the Eeyore depression simulation outperforms GPT-4o with SOTA prompting strategies, both in linguistic authenticity and profile adherence. 

---
