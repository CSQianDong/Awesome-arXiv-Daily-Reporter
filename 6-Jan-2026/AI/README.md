# Falcon-H1R: Pushing the Reasoning Frontiers with a Hybrid Model for Efficient Test-Time Scaling 

**Authors**: Falcon LLM Team, Iheb Chaabane, Puneesh Khanna, Suhail Mohmad, Slim Frikha, Shi Hu, Abdalgader Abubaker, Reda Alami, Mikhail Lubinets, Mohamed El Amine Seddik, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2601.02346)  

**Abstract**: This work introduces Falcon-H1R, a 7B-parameter reasoning-optimized model that establishes the feasibility of achieving competitive reasoning performance with small language models (SLMs). Falcon-H1R stands out for its parameter efficiency, consistently matching or outperforming SOTA reasoning models that are $2\times$ to $7\times$ larger across a variety of reasoning-intensive benchmarks. These results underscore the importance of careful data curation and targeted training strategies (via both efficient SFT and RL scaling) in delivering significant performance gains without increasing model size. Furthermore, Falcon-H1R advances the 3D limits of reasoning efficiency by combining faster inference (through its hybrid-parallel architecture design), token efficiency, and higher accuracy. This unique blend makes Falcon-H1R-7B a practical backbone for scaling advanced reasoning systems, particularly in scenarios requiring extensive chain-of-thoughts generation and parallel test-time scaling. Leveraging the recently introduced DeepConf approach, Falcon-H1R achieves state-of-the-art test-time scaling efficiency, offering substantial improvements in both accuracy and computational cost. As a result, Falcon-H1R demonstrates that compact models, through targeted model training and architectural choices, can deliver robust and scalable reasoning performance. 

---
# Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents 

**Authors**: Sourena Khanzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2601.02314)  

**Abstract**: As Large Language Model (LLM) agents are increasingly tasked with high-stakes autonomous decision-making, the transparency of their reasoning processes has become a critical safety concern. While \textit{Chain-of-Thought} (CoT) prompting allows agents to generate human-readable reasoning traces, it remains unclear whether these traces are \textbf{faithful} generative drivers of the model's output or merely \textbf{post-hoc rationalizations}. We introduce \textbf{Project Ariadne}, a novel XAI framework that utilizes Structural Causal Models (SCMs) and counterfactual logic to audit the causal integrity of agentic reasoning. Unlike existing interpretability methods that rely on surface-level textual similarity, Project Ariadne performs \textbf{hard interventions} ($do$-calculus) on intermediate reasoning nodes -- systematically inverting logic, negating premises, and reversing factual claims -- to measure the \textbf{Causal Sensitivity} ($\phi$) of the terminal answer. Our empirical evaluation of state-of-the-art models reveals a persistent \textit{Faithfulness Gap}. We define and detect a widespread failure mode termed \textbf{Causal Decoupling}, where agents exhibit a violation density ($\rho$) of up to $0.77$ in factual and scientific domains. In these instances, agents arrive at identical conclusions despite contradictory internal logic, proving that their reasoning traces function as "Reasoning Theater" while decision-making is governed by latent parametric priors. Our findings suggest that current agentic architectures are inherently prone to unfaithful explanation, and we propose the Ariadne Score as a new benchmark for aligning stated logic with model action. 

---
# Streaming Hallucination Detection in Long Chain-of-Thought Reasoning 

**Authors**: Haolang Lu, Minghui Pan, Ripeng Li, Guoshun Nan, Jialin Zhuang, Zijie Zhao, Zhongxiang Sun, Kun Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02170)  

**Abstract**: Long chain-of-thought (CoT) reasoning improves the performance of large language models, yet hallucinations in such settings often emerge subtly and propagate across reasoning steps. We suggest that hallucination in long CoT reasoning is better understood as an evolving latent state rather than a one-off erroneous event. Accordingly, we treat step-level hallucination judgments as local observations and introduce a cumulative prefix-level hallucination signal that tracks the global evolution of the reasoning state over the entire trajectory. Overall, our approach enables streaming hallucination detection in long CoT reasoning, providing real-time, interpretable evidence. 

---
# EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning 

**Authors**: Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.02163)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as long-term interactive agents, yet their limited context windows make it difficult to sustain coherent behavior over extended interactions. Existing memory systems often store isolated records and retrieve fragments, limiting their ability to consolidate evolving user states and resolve conflicts. We introduce EverMemOS, a self-organizing memory operating system that implements an engram-inspired lifecycle for computational memory. Episodic Trace Formation converts dialogue streams into MemCells that capture episodic traces, atomic facts, and time-bounded Foresight signals. Semantic Consolidation organizes MemCells into thematic MemScenes, distilling stable semantic structures and updating user profiles. Reconstructive Recollection performs MemScene-guided agentic retrieval to compose the necessary and sufficient context for downstream reasoning. Experiments on LoCoMo and LongMemEval show that EverMemOS achieves state-of-the-art performance on memory-augmented reasoning tasks. We further report a profile study on PersonaMem v2 and qualitative case studies illustrating chat-oriented capabilities such as user profiling and Foresight. Code is available at this https URL. 

---
# FormuLLA: A Large Language Model Approach to Generating Novel 3D Printable Formulations 

**Authors**: Adeshola Okubena, Yusuf Ali Mohammed, Moe Elbadawi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02071)  

**Abstract**: Pharmaceutical three-dimensional (3D) printing is an advanced fabrication technology with the potential to enable truly personalised dosage forms. Recent studies have integrated artificial intelligence (AI) to accelerate formulation and process development, drastically transforming current approaches to pharmaceutical 3D printing. To date, most AI-driven efforts remain narrowly focused, while failing to account for the broader formulation challenges inherent to the technology. Recent advances in AI have introduced artificial general intelligence concepts, wherein systems extend beyond conventional predictive modelling toward more generalised, human-like reasoning. In this work, we investigate the application of large language models (LLMs), fine-tuned on a fused deposition modelling (FDM) dataset comprising over 1400 formulations, to recommend suitable excipients based on active pharmaceutical ingredient (API) dose, and predict filament mechanical properties. Four LLM architectures were fine-tuned, with systematic evaluation of both fine-tuning and generative parameter configurations. Our results demonstrate that Llama2 was best suited for recommending excipients for FDM formulations. Additionally, model selection and parameterisation significantly influence performance, with smaller LLMs exhibiting instances of catastrophic forgetting. Furthermore, we demonstrate: (i) even with relatively small dataset of over 1400 formulations, it can lead to model catastrophic forgetting; (ii) standard LLM metrics only evaluate linguistic performance but not formulation processability; and (iii) LLMs trained on biomedically-related data do not always produce the best results. Addressing these challenges is essential to advancing LLMs beyond linguistic proficiency and toward reliable systems for pharmaceutical formulation development. 

---
# Higher-Order Action Regularization in Deep Reinforcement Learning: From Continuous Control to Building Energy Management 

**Authors**: Faizan Ahmed, Aniket Dixit, James Brusey  

**Link**: [PDF](https://arxiv.org/pdf/2601.02061)  

**Abstract**: Deep reinforcement learning agents often exhibit erratic, high-frequency control behaviors that hinder real-world deployment due to excessive energy consumption and mechanical wear. We systematically investigate action smoothness regularization through higher-order derivative penalties, progressing from theoretical understanding in continuous control benchmarks to practical validation in building energy management. Our comprehensive evaluation across four continuous control environments demonstrates that third-order derivative penalties (jerk minimization) consistently achieve superior smoothness while maintaining competitive performance. We extend these findings to HVAC control systems where smooth policies reduce equipment switching by 60%, translating to significant operational benefits. Our work establishes higher-order action regularization as an effective bridge between RL optimization and operational constraints in energy-critical applications. 

---
# Simulated Reasoning is Reasoning 

**Authors**: Hendrik Kempt, Alon Lavie  

**Link**: [PDF](https://arxiv.org/pdf/2601.02043)  

**Abstract**: Reasoning has long been understood as a pathway between stages of understanding. Proper reasoning leads to understanding of a given subject. This reasoning was conceptualized as a process of understanding in a particular way, i.e., "symbolic reasoning". Foundational Models (FM) demonstrate that this is not a necessary condition for many reasoning tasks: they can "reason" by way of imitating the process of "thinking out loud", testing the produced pathways, and iterating on these pathways on their own. This leads to some form of reasoning that can solve problems on its own or with few-shot learning, but appears fundamentally different from human reasoning due to its lack of grounding and common sense, leading to brittleness of the reasoning process. These insights promise to substantially alter our assessment of reasoning and its necessary conditions, but also inform the approaches to safety and robust defences against this brittleness of FMs. This paper offers and discusses several philosophical interpretations of this phenomenon, argues that the previously apt metaphor of the "stochastic parrot" has lost its relevance and thus should be abandoned, and reflects on different normative elements in the safety- and appropriateness-considerations emerging from these reasoning models and their growing capacity. 

---
# XAI-MeD: Explainable Knowledge Guided Neuro-Symbolic Framework for Domain Generalization and Rare Class Detection in Medical Imaging 

**Authors**: Midhat Urooj, Ayan Banerjee, Sandeep Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.02008)  

**Abstract**: Explainability domain generalization and rare class reliability are critical challenges in medical AI where deep models often fail under real world distribution shifts and exhibit bias against infrequent clinical conditions This paper introduces XAIMeD an explainable medical AI framework that integrates clinically accurate expert knowledge into deep learning through a unified neuro symbolic architecture XAIMeD is designed to improve robustness under distribution shift enhance rare class sensitivity and deliver transparent clinically aligned interpretations The framework encodes clinical expertise as logical connectives over atomic medical propositions transforming them into machine checkable class specific rules Their diagnostic utility is quantified through weighted feature satisfaction scores enabling a symbolic reasoning branch that complements neural predictions A confidence weighted fusion integrates symbolic and deep outputs while a Hunt inspired adaptive routing mechanism guided by Entropy Imbalance Gain EIG and Rare Class Gini mitigates class imbalance high intra class variability and uncertainty We evaluate XAIMeD across diverse modalities on four challenging tasks i Seizure Onset Zone SOZ localization from rs fMRI ii Diabetic Retinopathy grading across 6 multicenter datasets demonstrate substantial performance improvements including 6 percent gains in cross domain generalization and a 10 percent improved rare class F1 score far outperforming state of the art deep learning baselines Ablation studies confirm that the clinically grounded symbolic components act as effective regularizers ensuring robustness to distribution shifts XAIMeD thus provides a principled clinically faithful and interpretable approach to multimodal medical AI. 

---
# MindChat: A Privacy-preserving Large Language Model for Mental Health Support 

**Authors**: Dong Xue, Jicheng Tu, Ming Wang, Xin Yan, Fangzhou Liu, Jie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01993)  

**Abstract**: Large language models (LLMs) have shown promise for mental health support, yet training such models is constrained by the scarcity and sensitivity of real counseling dialogues. In this article, we present MindChat, a privacy-preserving LLM for mental health support, together with MindCorpus, a synthetic multi-turn counseling dataset constructed via a multi-agent role-playing framework. To synthesize high-quality counseling data, the developed dialogue-construction framework employs a dual closed-loop feedback design to integrate psychological expertise and counseling techniques through role-playing: (i) turn-level critique-and-revision to improve coherence and counseling appropriateness within a session, and (ii) session-level strategy refinement to progressively enrich counselor behaviors across sessions. To mitigate privacy risks under decentralized data ownership, we fine-tune the base model using federated learning with parameter-efficient LoRA adapters and incorporate differentially private optimization to reduce membership and memorization risks. Experiments on synthetic-data quality assessment and counseling capability evaluation show that MindCorpus improves training effectiveness and that MindChat is competitive with existing general and counseling-oriented LLM baselines under both automatic LLM-judge and human evaluation protocols, while exhibiting reduced privacy leakage under membership inference attacks. 

---
# ChaosBench-Logic: A Benchmark for Logical and Symbolic Reasoning on Chaotic Dynamical Systems 

**Authors**: Noel Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2601.01982)  

**Abstract**: Large language models (LLMs) excel at natural language tasks but remain brittle in domains requiring precise logical and symbolic reasoning. Chaotic dynamical systems provide an especially demanding test because chaos is deterministic yet often misinterpreted as randomness or complexity. We introduce ChaosBench-Logic, a benchmark that evaluates LLM reasoning across 30 diverse dynamical systems using a unified first-order logic (FOL) ontology. Each system is annotated with truth assignments for 11 semantic predicates, and 621 questions are generated across seven reasoning categories, including multi-hop implications, cross-system analogies, counterfactual reasoning, bias probes, and multi-turn dialogues. We define metrics for logical accuracy, implication consistency, dialogue coherence, and contradiction, and we release an open-source evaluation pipeline. Initial experiments show that frontier LLMs such as GPT-4, Claude 3.5 Sonnet, Gemini 2.5 Flash, and the open-source LLaMA-3 70B achieve 91-94% per-item accuracy, yet still score 0% on compositional items and exhibit fragile global coherence. Dialogue-level accuracy ranges from 53.1% (GPT-4 CoT) to 75.5% (LLaMA-3 zero-shot). ChaosBench-Logic provides a rigorous testbed for diagnosing such failures and a foundation for developing neuro-symbolic approaches that improve scientific reasoning in LLMs. 

---
# CNC-TP: Classifier Nominal Concept Based on Top-Pertinent Attributes 

**Authors**: Yasmine Souissi, Fabrice Boissier, Nida Meddouri  

**Link**: [PDF](https://arxiv.org/pdf/2601.01976)  

**Abstract**: Knowledge Discovery in Databases (KDD) aims to exploit the vast amounts of data generated daily across various domains of computer applications. Its objective is to extract hidden and meaningful knowledge from datasets through a structured process comprising several key steps: data selection, preprocessing, transformation, data mining, and visualization. Among the core data mining techniques are classification and clustering. Classification involves predicting the class of new instances using a classifier trained on labeled data. Several approaches have been proposed in the literature, including Decision Tree Induction, Bayesian classifiers, Nearest Neighbor search, Neural Networks, Support Vector Machines, and Formal Concept Analysis (FCA). The last one is recognized as an effective approach for interpretable and explainable learning. It is grounded in the mathematical structure of the concept lattice, which enables the generation of formal concepts and the discovery of hidden relationships among them. In this paper, we present a state-of-theart review of FCA-based classifiers. We explore various methods for computing closure operators from nominal data and introduce a novel approach for constructing a partial concept lattice that focuses on the most relevant concepts. Experimental results are provided to demonstrate the efficiency of the proposed method. 

---
# OpenSocInt: A Multi-modal Training Environment for Human-Aware Social Navigation 

**Authors**: Victor Sanchez, Chris Reinke, Ahamed Mohamed, Xavier Alameda-Pineda  

**Link**: [PDF](https://arxiv.org/pdf/2601.01939)  

**Abstract**: In this paper, we introduce OpenSocInt, an open-source software package providing a simulator for multi-modal social interactions and a modular architecture to train social agents. We described the software package and showcased its interest via an experimental protocol based on the task of social navigation. Our framework allows for exploring the use of different perceptual features, their encoding and fusion, as well as the use of different agents. The software is already publicly available under GPL at this https URL. 

---
# MMP-A*: Multimodal Perception Enhanced Incremental Heuristic Search on Path Planning 

**Authors**: Minh Hieu Ha, Khanh Ly Ta, Hung Phan, Tung Doan, Tung Dao, Dao Tran, Huynh Thi Thanh Binh  

**Link**: [PDF](https://arxiv.org/pdf/2601.01910)  

**Abstract**: Autonomous path planning requires a synergy between global reasoning and geometric precision, especially in complex or cluttered environments. While classical A* is valued for its optimality, it incurs prohibitive computational and memory costs in large-scale scenarios. Recent attempts to mitigate these limitations by using Large Language Models for waypoint guidance remain insufficient, as they rely only on text-based reasoning without spatial grounding. As a result, such models often produce incorrect waypoints in topologically complex environments with dead ends, and lack the perceptual capacity to interpret ambiguous physical boundaries. These inconsistencies lead to costly corrective expansions and undermine the intended computational efficiency.
We introduce MMP-A*, a multimodal framework that integrates the spatial grounding capabilities of vision-language models with a novel adaptive decay mechanism. By anchoring high-level reasoning in physical geometry, the framework produces coherent waypoint guidance that addresses the limitations of text-only planners. The adaptive decay mechanism dynamically regulates the influence of uncertain waypoints within the heuristic, ensuring geometric validity while substantially reducing memory overhead. To evaluate robustness, we test the framework in challenging environments characterized by severe clutter and topological complexity. Experimental results show that MMP-A* achieves near-optimal trajectories with significantly reduced operational costs, demonstrating its potential as a perception-grounded and computationally efficient paradigm for autonomous navigation. 

---
# Theory Trace Card: Theory-Driven Socio-Cognitive Evaluation of LLMs 

**Authors**: Farzan Karimi-Malekabadi, Suhaib Abdurahman, Zhivar Sourati, Jackson Trager, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2601.01878)  

**Abstract**: Socio-cognitive benchmarks for large language models (LLMs) often fail to predict real-world behavior, even when models achieve high benchmark scores. Prior work has attributed this evaluation-deployment gap to problems of measurement and validity. While these critiques are insightful, we argue that they overlook a more fundamental issue: many socio-cognitive evaluations proceed without an explicit theoretical specification of the target capability, leaving the assumptions linking task performance to competence implicit. Without this theoretical grounding, benchmarks that exercise only narrow subsets of a capability are routinely misinterpreted as evidence of broad competence: a gap that creates a systemic validity illusion by masking the failure to evaluate the capability's other essential dimensions. To address this gap, we make two contributions. First, we diagnose and formalize this theory gap as a foundational failure that undermines measurement and enables systematic overgeneralization of benchmark results. Second, we introduce the Theory Trace Card (TTC), a lightweight documentation artifact designed to accompany socio-cognitive evaluations, which explicitly outlines the theoretical basis of an evaluation, the components of the target capability it exercises, its operationalization, and its limitations. We argue that TTCs enhance the interpretability and reuse of socio-cognitive evaluations by making explicit the full validity chain, which links theory, task operationalization, scoring, and limitations, without modifying benchmarks or requiring agreement on a single theory. 

---
# Toward Auditable Neuro-Symbolic Reasoning in Pathology: SQL as an Explicit Trace of Evidence 

**Authors**: Kewen Cao, Jianxu Chen, Yongbing Zhang, Ye Zhang, Hongxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01875)  

**Abstract**: Automated pathology image analysis is central to clinical diagnosis, but clinicians still ask which slide features drive a model's decision and why. Vision-language models can produce natural language explanations, but these are often correlational and lack verifiable evidence. In this paper, we introduce an SQL-centered agentic framework that enables both feature measurement and reasoning to be auditable. Specifically, after extracting human-interpretable cellular features, Feature Reasoning Agents compose and execute SQL queries over feature tables to aggregate visual evidence into quantitative findings. A Knowledge Comparison Agent then evaluates these findings against established pathological knowledge, mirroring how pathologists justify diagnoses from measurable observations. Extensive experiments evaluated on two pathology visual question answering datasets demonstrate our method improves interpretability and decision traceability while producing executable SQL traces that link cellular measurements to diagnostic conclusions. 

---
# Jenius Agent: Towards Experience-Driven Accuracy Optimization in Real-World Scenarios 

**Authors**: Defei Xia, Bingfeng Pi, Shenbin Zhang, Song Hua, Yunfei Wei, Lei Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2601.01857)  

**Abstract**: As agent systems powered by large language models (LLMs) advance, improving the task performance of an autonomous agent, especially in context understanding, tool usage, and response generation, has become increasingly critical. Although prior studies have advanced the overall design of LLM-based agents, systematic optimization of their internal reasoning and tool-use pipelines remains underexplored. This paper introduces an agent framework grounded in real-world practical experience, with three key innovations: (1) an adaptive prompt generation strategy that aligns with the agent's state and task goals to improve reliability and robustness; (2) a context-aware tool orchestration module that performs tool categorization, semantic retrieval, and adaptive invocation based on user intent and context; and (3) a layered memory mechanism that integrates session memory, task history, and external summaries to improve relevance and efficiency through dynamic summarization and compression. An end-to-end framework named Jenius-Agent has been integrated with three key optimizations, including tools based on the Model Context Protocol (MCP), file input/output (I/O), and execution feedback. The experiments show a 20 percent improvement in task accuracy, along with a reduced token cost, response latency, and invocation failures. The framework is already deployed in Jenius (this https URL), providing a lightweight and scalable solution for robust, protocol-compatible autonomous agents. 

---
# Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via Retrieval-Augmented Generation 

**Authors**: Udiptaman Das, Krishnasai B. Atmakuri, Duy Ho, Chi Lee, Yugyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.01844)  

**Abstract**: Large language models (LLMs) offer new opportunities for constructing knowledge graphs (KGs) from unstructured clinical narratives. However, existing approaches often rely on structured inputs and lack robust validation of factual accuracy and semantic consistency, limitations that are especially problematic in oncology. We introduce an end-to-end framework for clinical KG construction and evaluation directly from free text using multi-agent prompting and a schema-constrained Retrieval-Augmented Generation (KG-RAG) strategy. Our pipeline integrates (1) prompt-driven entity, attribute, and relation extraction; (2) entropy-based uncertainty scoring; (3) ontology-aligned RDF/OWL schema generation; and (4) multi-LLM consensus validation for hallucination detection and semantic refinement. Beyond static graph construction, the framework supports continuous refinement and self-supervised evaluation, enabling iterative improvement of graph quality. Applied to two oncology cohorts (PDAC and BRCA), our method produces interpretable, SPARQL-compatible, and clinically grounded knowledge graphs without relying on gold-standard annotations. Experimental results demonstrate consistent gains in precision, relevance, and ontology compliance over baseline methods. 

---
# COMPASS: A Framework for Evaluating Organization-Specific Policy Alignment in LLMs 

**Authors**: Dasol Choi, DongGeon Lee, Brigitta Jesica Kartono, Helena Berndt, Taeyoun Kwon, Joonwon Jang, Haon Park, Hwanjo Yu, Minsuk Kahng  

**Link**: [PDF](https://arxiv.org/pdf/2601.01836)  

**Abstract**: As large language models are deployed in high-stakes enterprise applications, from healthcare to finance, ensuring adherence to organization-specific policies has become essential. Yet existing safety evaluations focus exclusively on universal harms. We present COMPASS (Company/Organization Policy Alignment Assessment), the first systematic framework for evaluating whether LLMs comply with organizational allowlist and denylist policies. We apply COMPASS to eight diverse industry scenarios, generating and validating 5,920 queries that test both routine compliance and adversarial robustness through strategically designed edge cases. Evaluating seven state-of-the-art models, we uncover a fundamental asymmetry: models reliably handle legitimate requests (>95% accuracy) but catastrophically fail at enforcing prohibitions, refusing only 13-40% of adversarial denylist violations. These results demonstrate that current LLMs lack the robustness required for policy-critical deployments, establishing COMPASS as an essential evaluation framework for organizational AI safety. 

---
# Admissibility Alignment 

**Authors**: Chris Duffey  

**Link**: [PDF](https://arxiv.org/pdf/2601.01816)  

**Abstract**: This paper introduces Admissibility Alignment: a reframing of AI alignment as a property of admissible action and decision selection over distributions of outcomes under uncertainty, evaluated through the behavior of candidate policies. We present MAP-AI (Monte Carlo Alignment for Policy) as a canonical system architecture for operationalizing admissibility alignment, formalizing alignment as a probabilistic, decision-theoretic property rather than a static or binary condition.
MAP-AI, a new control-plane system architecture for aligned decision-making under uncertainty, enforces alignment through Monte Carlo estimation of outcome distributions and admissibility-controlled policy selection rather than static model-level constraints. The framework evaluates decision policies across ensembles of plausible futures, explicitly modeling uncertainty, intervention effects, value ambiguity, and governance constraints. Alignment is assessed through distributional properties including expected utility, variance, tail risk, and probability of misalignment rather than accuracy or ranking performance. This approach distinguishes probabilistic prediction from decision reasoning under uncertainty and provides an executable methodology for evaluating trust and alignment in enterprise and institutional AI systems. The result is a practical foundation for governing AI systems whose impact is determined not by individual forecasts, but by policy behavior across distributions and tail events. Finally, we show how distributional alignment evaluation can be integrated into decision-making itself, yielding an admissibility-controlled action selection mechanism that alters policy behavior under uncertainty without retraining or modifying underlying models. 

---
# PsychEval: A Multi-Session and Multi-Therapy Benchmark for High-Realism and Comprehensive AI Psychological Counselor 

**Authors**: Qianjun Pan, Junyi Wang, Jie Zhou, Yutao Yang, Junsong Li, Kaiyin Xu, Yougen Zhou, Yihan Li, Jingyuan Zhao, Qin Chen, Ningning Zhou, Kai Chen, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2601.01802)  

**Abstract**: To develop a reliable AI for psychological assessment, we introduce \texttt{PsychEval}, a multi-session, multi-therapy, and highly realistic benchmark designed to address three key challenges: \textbf{1) Can we train a highly realistic AI counselor?} Realistic counseling is a longitudinal task requiring sustained memory and dynamic goal tracking. We propose a multi-session benchmark (spanning 6-10 sessions across three distinct stages) that demands critical capabilities such as memory continuity, adaptive reasoning, and longitudinal planning. The dataset is annotated with extensive professional skills, comprising over 677 meta-skills and 4577 atomic skills. \textbf{2) How to train a multi-therapy AI counselor?} While existing models often focus on a single therapy, complex cases frequently require flexible strategies among various therapies. We construct a diverse dataset covering five therapeutic modalities (Psychodynamic, Behaviorism, CBT, Humanistic Existentialist, and Postmodernist) alongside an integrative therapy with a unified three-stage clinical framework across six core psychological topics. \textbf{3) How to systematically evaluate an AI counselor?} We establish a holistic evaluation framework with 18 therapy-specific and therapy-shared metrics across Client-Level and Counselor-Level dimensions. To support this, we also construct over 2,000 diverse client profiles. Extensive experimental analysis fully validates the superior quality and clinical fidelity of our dataset. Crucially, \texttt{PsychEval} transcends static benchmarking to serve as a high-fidelity reinforcement learning environment that enables the self-evolutionary training of clinically responsible and adaptive AI counselors. 

---
# Can Large Language Models Solve Engineering Equations? A Systematic Comparison of Direct Prediction and Solver-Assisted Approaches 

**Authors**: Sai Varun Kodathala, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2601.01774)  

**Abstract**: Transcendental equations requiring iterative numerical solution pervade engineering practice, from fluid mechanics friction factor calculations to orbital position determination. We systematically evaluate whether Large Language Models can solve these equations through direct numerical prediction or whether a hybrid architecture combining LLM symbolic manipulation with classical iterative solvers proves more effective. Testing six state-of-the-art models (GPT-5.1, GPT-5.2, Gemini-3-Flash, Gemini-2.5-Lite, Claude-Sonnet-4.5, Claude-Opus-4.5) on 100 problems spanning seven engineering domains, we compare direct prediction against solver-assisted computation where LLMs formulate governing equations and provide initial conditions while Newton-Raphson iteration performs numerical solution. Direct prediction yields mean relative errors of 0.765 to 1.262 across models, while solver-assisted computation achieves 0.225 to 0.301, representing error reductions of 67.9% to 81.8%. Domain-specific analysis reveals dramatic improvements in Electronics (93.1%) due to exponential equation sensitivity, contrasted with modest gains in Fluid Mechanics (7.2%) where LLMs exhibit effective pattern recognition. These findings establish that contemporary LLMs excel at symbolic manipulation and domain knowledge retrieval but struggle with precision-critical iterative arithmetic, suggesting their optimal deployment as intelligent interfaces to classical numerical solvers rather than standalone computational engines. 

---
# A New Benchmark for the Appropriate Evaluation of RTL Code Optimization 

**Authors**: Yao Lu, Shang Liu, Hangan Zhou, Wenji Fang, Qijun Zhang, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.01765)  

**Abstract**: The rapid progress of artificial intelligence increasingly relies on efficient integrated circuit (IC) design. Recent studies have explored the use of large language models (LLMs) for generating Register Transfer Level (RTL) code, but existing benchmarks mainly evaluate syntactic correctness rather than optimization quality in terms of power, performance, and area (PPA). This work introduces RTL-OPT, a benchmark for assessing the capability of LLMs in RTL optimization. RTL-OPT contains 36 handcrafted digital designs that cover diverse implementation categories including combinational logic, pipelined datapaths, finite state machines, and memory interfaces. Each task provides a pair of RTL codes, a suboptimal version and a human-optimized reference that reflects industry-proven optimization patterns not captured by conventional synthesis tools. Furthermore, RTL-OPT integrates an automated evaluation framework to verify functional correctness and quantify PPA improvements, enabling standardized and meaningful assessment of generative models for hardware design optimization. 

---
# AI Agent Systems: Architectures, Applications, and Evaluation 

**Authors**: Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01743)  

**Abstract**: AI agents -- systems that combine foundation models with reasoning, planning, memory, and tool use -- are rapidly becoming a practical interface between natural-language intent and real-world computation. This survey synthesizes the emerging landscape of AI agent architectures across: (i) deliberation and reasoning (e.g., chain-of-thought-style decomposition, self-reflection and verification, and constraint-aware decision making), (ii) planning and control (from reactive policies to hierarchical and multi-step planners), and (iii) tool calling and environment interaction (retrieval, code execution, APIs, and multimodal perception). We organize prior work into a unified taxonomy spanning agent components (policy/LLM core, memory, world models, planners, tool routers, and critics), orchestration patterns (single-agent vs.\ multi-agent; centralized vs.\ decentralized coordination), and deployment settings (offline analysis vs.\ online interactive assistance; safety-critical vs.\ open-ended tasks). We discuss key design trade-offs -- latency vs.\ accuracy, autonomy vs.\ controllability, and capability vs.\ reliability -- and highlight how evaluation is complicated by non-determinism, long-horizon credit assignment, tool and environment variability, and hidden costs such as retries and context growth. Finally, we summarize measurement and benchmarking practices (task suites, human preference and utility metrics, success under constraints, robustness and security) and identify open challenges including verification and guardrails for tool actions, scalable memory and context management, interpretability of agent decisions, and reproducible evaluation under realistic workloads. 

---
# Yuan3.0 Flash: An Open Multimodal Large Language Model for Enterprise Applications 

**Authors**: YuanLab.ai, Shawn Wu, Sean Wang, Louie Li, Darcy Chen, Allen Wang, Jiangang Luo, Xudong Zhao, Joseph Shen, Gawain Ma, Jasper Jia, Marcus Mao, Claire Wang, Hunter He, Carol Wang, Zera Zhang, Jason Wang, Chonly Shen, Leo Zhang, Logan Chen, Qasim Meng, James Gong, Danied Zhao, Penn Zheng, Owen Zhu, Tong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01718)  

**Abstract**: We introduce Yuan3.0 Flash, an open-source Mixture-of-Experts (MoE) MultiModal Large Language Model featuring 3.7B activated parameters and 40B total parameters, specifically designed to enhance performance on enterprise-oriented tasks while maintaining competitive capabilities on general-purpose tasks. To address the overthinking phenomenon commonly observed in Large Reasoning Models (LRMs), we propose Reflection-aware Adaptive Policy Optimization (RAPO), a novel RL training algorithm that effectively regulates overthinking behaviors. In enterprise-oriented tasks such as retrieval-augmented generation (RAG), complex table understanding, and summarization, Yuan3.0 Flash consistently achieves superior performance. Moreover, it also demonstrates strong reasoning capabilities in domains such as mathematics, science, etc., attaining accuracy comparable to frontier model while requiring only approximately 1/4 to 1/2 of the average tokens. Yuan3.0 Flash has been fully open-sourced to facilitate further research and real-world deployment: this https URL. 

---
# Structured Decomposition for LLM Reasoning: Cross-Domain Validation and Semantic Web Integration 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2601.01609)  

**Abstract**: Rule-based reasoning over natural language input arises in domains where decisions must be auditable and justifiable: clinical protocols specify eligibility criteria in prose, evidence rules define admissibility through textual conditions, and scientific standards dictate methodological requirements. Applying rules to such inputs demands both interpretive flexibility and formal guarantees. Large language models (LLMs) provide flexibility but cannot ensure consistent rule application; symbolic systems provide guarantees but require structured input. This paper presents an integration pattern that combines these strengths: LLMs serve as ontology population engines, translating unstructured text into ABox assertions according to expert-authored TBox specifications, while SWRL-based reasoners apply rules with deterministic guarantees. The framework decomposes reasoning into entity identification, assertion extraction, and symbolic verification, with task definitions grounded in OWL 2 ontologies. Experiments across three domains (legal hearsay determination, scientific method-task application, clinical trial eligibility) and eleven language models validate the approach. Structured decomposition achieves statistically significant improvements over few-shot prompting in aggregate, with gains observed across all three domains. An ablation study confirms that symbolic verification provides substantial benefit beyond structured prompting alone. The populated ABox integrates with standard semantic web tooling for inspection and querying, positioning the framework for richer inference patterns that simpler formalisms cannot express. 

---
# CaveAgent: Transforming LLMs into Stateful Runtime Operators 

**Authors**: Maohao Ran, Zhenglin Wan, Cooper Lin, Yanting Zhang, Hongyu Xin, Hongwei Fan, Yibo Xu, Beier Luo, Yaxin Zhou, Wangbo Zhao, Lijie Yang, Lang Feng, Fuchao Yang, Jingxuan Wu, Yiqiao Huang, Chendong Ma, Dailing Jiang, Jianbo Deng, Sihui Han, Bo An, Yike Guo, Jun Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.01569)  

**Abstract**: LLM-based agents are increasingly capable of complex task execution, yet current agentic systems remain constrained by text-centric paradigms. Traditional approaches rely on procedural JSON-based function calling, which often struggles with long-horizon tasks due to fragile multi-turn dependencies and context drift. In this paper, we present CaveAgent, a framework that transforms the paradigm from "LLM-as-Text-Generator" to "LLM-as-Runtime-Operator." We introduce a Dual-stream Context Architecture that decouples state management into a lightweight semantic stream for reasoning and a persistent, deterministic Python Runtime stream for execution. In addition to leveraging code generation to efficiently resolve interdependent sub-tasks (e.g., loops, conditionals) in a single step, we introduce \textit{Stateful Runtime Management} in CaveAgent. Distinct from existing code-based approaches that remain text-bound and lack the support for external object injection and retrieval, CaveAgent injects, manipulates, and retrieves complex Python objects (e.g., DataFrames, database connections) that persist across turns. This persistence mechanism acts as a high-fidelity external memory to eliminate context drift, avoid catastrophic forgetting, while ensuring that processed data flows losslessly to downstream applications. Comprehensive evaluations on Tau$^2$-bench, BFCL and various case studies across representative SOTA LLMs demonstrate CaveAgent's superiority. Specifically, our framework achieves a 10.5\% success rate improvement on retail tasks and reduces total token consumption by 28.4\% in multi-turn scenarios. On data-intensive tasks, direct variable storage and retrieval reduces token consumption by 59\%, allowing CaveAgent to handle large-scale data that causes context overflow failures in both JSON-based and Code-based agents. 

---
# Logics-STEM: Empowering LLM Reasoning via Failure-Driven Post-Training and Document Knowledge Enhancement 

**Authors**: Mingyu Xu, Cheng Fang, Keyue Jiang, Yuqian Zheng, Yanghua Xiao, Baojian Zhou, Qifang Zhao, Suhang Zheng, Xiuwen Zhu, Jiyang Tang, Yongchi Zhao, Yijia Luo, Zhiqi Bai, Yuchi Xu, Wenbo Su, Wei Wang, Bing Zhao, Lin Qu, Xiaoxiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01562)  

**Abstract**: We present Logics-STEM, a state-of-the-art reasoning model fine-tuned on Logics-STEM-SFT-Dataset, a high-quality and diverse dataset at 10M scale that represents one of the largest-scale open-source long chain-of-thought corpora. Logics-STEM targets reasoning tasks in the domains of Science, Technology, Engineering, and Mathematics (STEM), and exhibits exceptional performance on STEM-related benchmarks with an average improvement of 4.68% over the next-best model at 8B scale. We attribute the gains to our data-algorithm co-design engine, where they are jointly optimized to fit a gold-standard distribution behind reasoning. Data-wise, the Logics-STEM-SFT-Dataset is constructed from a meticulously designed data curation engine with 5 stages to ensure the quality, diversity, and scalability, including annotation, deduplication, decontamination, distillation, and stratified sampling. Algorithm-wise, our failure-driven post-training framework leverages targeted knowledge retrieval and data synthesis around model failure regions in the Supervised Fine-tuning (SFT) stage to effectively guide the second-stage SFT or the reinforcement learning (RL) for better fitting the target distribution. The superior empirical performance of Logics-STEM reveals the vast potential of combining large-scale open-source data with carefully designed synthetic data, underscoring the critical role of data-algorithm co-design in enhancing reasoning capabilities through post-training. We make both the Logics-STEM models (8B and 32B) and the Logics-STEM-SFT-Dataset (10M and downsampled 2.2M versions) publicly available to support future research in the open-source community. 

---
# Improving Behavioral Alignment in LLM Social Simulations via Context Formation and Navigation 

**Authors**: Letian Kong, Qianran, Renyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01546)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human behavior in experimental settings, but they systematically diverge from human decisions in complex decision-making environments, where participants must anticipate others' actions and form beliefs based on observed behavior. We propose a two-stage framework for improving behavioral alignment. The first stage, context formation, explicitly specifies the experimental design to establish an accurate representation of the decision task and its context. The second stage, context navigation, guides the reasoning process within that representation to make decisions. We validate this framework through a focal replication of a sequential purchasing game with quality signaling (Kremer and Debo, 2016), extending to a crowdfunding game with costly signaling (Cason et al., 2025) and a demand-estimation task (Gui and Toubia, 2025) to test generalizability across decision environments. Across four state-of-the-art (SOTA) models (GPT-4o, GPT-5, Claude-4.0-Sonnet-Thinking, DeepSeek-R1), we find that complex decision-making environments require both stages to achieve behavioral alignment with human benchmarks, whereas the simpler demand-estimation task requires only context formation. Our findings clarify when each stage is necessary and provide a systematic approach for designing and diagnosing LLM social simulations as complements to human subjects in behavioral research. 

---
# Aletheia: Quantifying Cognitive Conviction in Reasoning Models via Regularized Inverse Confusion Matrix 

**Authors**: Fanzhe Fu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01532)  

**Abstract**: In the progressive journey toward Artificial General Intelligence (AGI), current evaluation paradigms face an epistemological crisis. Static benchmarks measure knowledge breadth but fail to quantify the depth of belief. While Simhi et al. (2025) defined the CHOKE phenomenon in standard QA, we extend this framework to quantify "Cognitive Conviction" in System 2 reasoning models. We propose Project Aletheia, a cognitive physics framework that employs Tikhonov Regularization to invert the judge's confusion matrix. To validate this methodology without relying on opaque private data, we implement a Synthetic Proxy Protocol. Our preliminary pilot study on 2025 baselines (e.g., DeepSeek-R1, OpenAI o1) suggests that while reasoning models act as a "cognitive buffer," they may exhibit "Defensive OverThinking" under adversarial pressure. Furthermore, we introduce the Aligned Conviction Score (S_aligned) to verify that conviction does not compromise safety. This work serves as a blueprint for measuring AI scientific integrity. 

---
# Bayesian Orchestration of Multi-LLM Agents for Cost-Aware Sequential Decision-Making 

**Authors**: Danial Amin  

**Link**: [PDF](https://arxiv.org/pdf/2601.01522)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous decision agents in settings with asymmetric error costs: hiring (missed talent vs wasted interviews), medical triage (missed emergencies vs unnecessary escalation), and fraud detection (approved fraud vs declined legitimate payments). The dominant design queries a single LLM for a posterior over states, thresholds "confidence," and acts; we prove this is inadequate for sequential decisions with costs. We propose a Bayesian, cost-aware multi-LLM orchestration framework that treats LLMs as approximate likelihood models rather than classifiers. For each candidate state, we elicit likelihoods via contrastive prompting, aggregate across diverse models with robust statistics, and update beliefs with Bayes rule under explicit priors as new evidence arrives. This enables coherent belief updating, expected-cost action selection, principled information gathering via value of information, and fairness gains via ensemble bias mitigation. In resume screening with costs of 40000 USD per missed hire, 2500 USD per interview, and 150 USD per phone screen, experiments on 1000 resumes using five LLMs (GPT-4o, Claude 4.5 Sonnet, Gemini Pro, Grok, DeepSeek) reduce total cost by 294000 USD (34 percent) versus the best single-LLM baseline and improve demographic parity by 45 percent (max group gap 22 to 5 percentage points). Ablations attribute 51 percent of savings to multi-LLM aggregation, 43 percent to sequential updating, and 20 percent to disagreement-triggered information gathering, consistent with the theoretical benefits of correct probabilistic foundations. 

---
# Reading Between the Lines: Deconfounding Causal Estimates using Text Embeddings and Deep Learning 

**Authors**: Ahmed Dawoud, Osama El-Shamy  

**Link**: [PDF](https://arxiv.org/pdf/2601.01511)  

**Abstract**: Estimating causal treatment effects in observational settings is frequently compromised by selection bias arising from unobserved confounders. While traditional econometric methods struggle when these confounders are orthogonal to structured covariates, high-dimensional unstructured text often contains rich proxies for these latent variables. This study proposes a Neural Network-Enhanced Double Machine Learning (DML) framework designed to leverage text embeddings for causal identification. Using a rigorous synthetic benchmark, we demonstrate that unstructured text embeddings capture critical confounding information that is absent from structured tabular data. However, we show that standard tree-based DML estimators retain substantial bias (+24%) due to their inability to model the continuous topology of embedding manifolds. In contrast, our deep learning approach reduces bias to -0.86% with optimized architectures, effectively recovering the ground-truth causal parameter. These findings suggest that deep learning architectures are essential for satisfying the unconfoundedness assumption when conditioning on high-dimensional natural language data 

---
# A construction of an optimal base for conditional attribute and attributional condition implications in triadic contexts 

**Authors**: Romuald Kwessy Mouona, Blaise Blériot Koguep Njionou, Etienne Romuald Temgoua Alomo, Rokia Missaoui, Leonard Kwuida  

**Link**: [PDF](https://arxiv.org/pdf/2601.01467)  

**Abstract**: This article studies implications in triadic contexts. Specifically, we focus on those introduced by Ganter and Obiedkov, namely conditional attribute and attributional condition implications. Our aim is to construct an optimal base for these implications. 

---
# Empowering Small Language Models with Factual Hallucination-Aware Reasoning for Financial Classification 

**Authors**: Han Yuan, Yilin Wu, Li Zhang, Zheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.01378)  

**Abstract**: Small language models (SLMs) are increasingly used for financial classification due to their fast inference and local deployability. However, compared with large language models, SLMs are more prone to factual hallucinations in reasoning and exhibit weaker classification performance. This raises a natural question: Can mitigating factual hallucinations improve SLMs' financial classification? To address this, we propose a three-step pipeline named AAAI (Association Identification, Automated Detection, and Adaptive Inference). Experiments on three representative SLMs reveal that: (1) factual hallucinations are positively correlated with misclassifications; (2) encoder-based verifiers effectively detect factual hallucinations; and (3) incorporating feedback on factual errors enables SLMs' adaptive inference that enhances classification performance. We hope this pipeline contributes to trustworthy and effective applications of SLMs in finance. 

---
# KGCE: Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models 

**Authors**: Zixian Liu, Sihao Liu, Yuqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.01366)  

**Abstract**: With the rapid adoption of multimodal large language models (MLMs) in autonomous agents, cross-platform task execution capabilities in educational settings have garnered significant attention. However, existing benchmark frameworks still exhibit notable deficiencies in supporting cross-platform tasks in educational contexts, especially when dealing with school-specific software (such as XiaoYa Intelligent Assistant, HuaShi XiaZi, etc.), where the efficiency of agents often significantly decreases due to a lack of understanding of the structural specifics of these private-domain software. Additionally, current evaluation methods heavily rely on coarse-grained metrics like goal orientation or trajectory matching, making it challenging to capture the detailed execution and efficiency of agents in complex tasks. To address these issues, we propose KGCE (Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models), a novel benchmarking platform that integrates knowledge base enhancement and a dual-graph evaluation framework. We first constructed a dataset comprising 104 education-related tasks, covering Windows, Android, and cross-platform collaborative tasks. KGCE introduces a dual-graph evaluation framework that decomposes tasks into multiple sub-goals and verifies their completion status, providing fine-grained evaluation metrics. To overcome the execution bottlenecks of existing agents in private-domain tasks, we developed an enhanced agent system incorporating a knowledge base specific to school-specific software. The code can be found at this https URL. 

---
# A unified multimodal understanding and generation model for cross-disciplinary scientific research 

**Authors**: Xiaomeng Yang, Zhiyu Tan, Xiaohui Zhong, Mengping Yang, Qiusheng Huang, Lei Chen, Libo Wu, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.01363)  

**Abstract**: Scientific discovery increasingly relies on integrating heterogeneous, high-dimensional data across disciplines nowadays. While AI models have achieved notable success across various scientific domains, they typically remain domain-specific or lack the capability of simultaneously understanding and generating multimodal scientific data, particularly for high-dimensional data. Yet, many pressing global challenges and scientific problems are inherently cross-disciplinary and require coordinated progress across multiple fields. Here, we present FuXi-Uni, a native unified multimodal model for scientific understanding and high-fidelity generation across scientific domains within a single architecture. Specifically, FuXi-Uni aligns cross-disciplinary scientific tokens within natural language tokens and employs science decoder to reconstruct scientific tokens, thereby supporting both natural language conversation and scientific numerical prediction. Empirically, we validate FuXi-Uni in Earth science and Biomedicine. In Earth system modeling, the model supports global weather forecasting, tropical cyclone (TC) forecast editing, and spatial downscaling driven by only language instructions. FuXi-Uni generates 10-day global forecasts at 0.25° resolution that outperform the SOTA physical forecasting system. It shows superior performance for both TC track and intensity prediction relative to the SOTA physical model, and generates high-resolution regional weather fields that surpass standard interpolation baselines. Regarding biomedicine, FuXi-Uni outperforms leading multimodal large language models on multiple biomedical visual question answering benchmarks. By unifying heterogeneous scientific modalities within a native shared latent space while maintaining strong domain-specific performance, FuXi-Uni provides a step forward more general-purpose, multimodal scientific models. 

---
# Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale 

**Authors**: Shengji Tang, Weihao Lin, Jingqi Ye, Hao Li, Bo Zhang, Shuyue Hu, Tao Chen, Wangli Ouyang, Lei Bai, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.01330)  

**Abstract**: Large Language Models (LLMs) have rapidly advanced, with Gemini-3-Pro setting a new performance milestone. In this work, we explore collective intelligence as an alternative to monolithic scaling, and demonstrate that open-source LLMs' collaboration can surpass Gemini-3-Pro. We first revisit LLM routing and aggregation at scale and identify three key bottlenecks: (1) current train-free routers are limited by a query-based paradigm focusing solely on textual similarity; (2) recent aggregation methods remain largely static, failing to select appropriate aggregators for different tasks;(3) the complementarity of routing and aggregation remains underutilized. To address these problems, we introduce JiSi, a novel framework designed to release the full potential of LLMs' collaboration through three innovations: (1) Query-Response Mixed Routing capturing both semantic information and problem difficulty; (2) Support-Set-based Aggregator Selection jointly evaluating the aggregation and domain capacity of aggregators; (3) Adaptive Routing-Aggregation Switch dynamically leveraging the advantages of routing and aggregation. Comprehensive experiments on nine benchmarks demonstrate that JiSi can surpass Gemini-3-Pro with only 47% costs by orchestrating ten open-source LLMs, while outperforming mainstream baselines. It suggests that collective intelligence represents a novel path towards Artificial General Intelligence (AGI). 

---
# Digital Twin AI: Opportunities and Challenges from Large Language Models to World Models 

**Authors**: Rong Zhou, Dongping Chen, Zihan Jia, Yao Su, Yixin Liu, Yiwen Lu, Dongwei Shi, Yue Huang, Tianyang Xu, Yi Pan, Xinliang Li, Yohannes Abate, Qingyu Chen, Zhengzhong Tu, Yu Yang, Yu Zhang, Qingsong Wen, Gengchen Mai, Sunyang Fu, Jiachen Li, Xuyu Wang, Ziran Wang, Jing Huang, Tianming Liu, Yong Chen, Lichao Sun, Lifang He  

**Link**: [PDF](https://arxiv.org/pdf/2601.01321)  

**Abstract**: Digital twins, as precise digital representations of physical systems, have evolved from passive simulation tools into intelligent and autonomous entities through the integration of artificial intelligence technologies. This paper presents a unified four-stage framework that systematically characterizes AI integration across the digital twin lifecycle, spanning modeling, mirroring, intervention, and autonomous management. By synthesizing existing technologies and practices, we distill a unified four-stage framework that systematically characterizes how AI methodologies are embedded across the digital twin lifecycle: (1) modeling the physical twin through physics-based and physics-informed AI approaches, (2) mirroring the physical system into a digital twin with real-time synchronization, (3) intervening in the physical twin through predictive modeling, anomaly detection, and optimization strategies, and (4) achieving autonomous management through large language models, foundation models, and intelligent agents. We analyze the synergy between physics-based modeling and data-driven learning, highlighting the shift from traditional numerical solvers to physics-informed and foundation models for physical systems. Furthermore, we examine how generative AI technologies, including large language models and generative world models, transform digital twins into proactive and self-improving cognitive systems capable of reasoning, communication, and creative scenario generation. Through a cross-domain review spanning eleven application domains, including healthcare, aerospace, smart manufacturing, robotics, and smart cities, we identify common challenges related to scalability, explainability, and trustworthiness, and outline directions for responsible AI-driven digital twin systems. 

---
# Accelerating Monte-Carlo Tree Search with Optimized Posterior Policies 

**Authors**: Keith Frankston, Benjamin Howard  

**Link**: [PDF](https://arxiv.org/pdf/2601.01301)  

**Abstract**: We introduce a recursive AlphaZero-style Monte--Carlo tree search algorithm, "RMCTS". The advantage of RMCTS over AlphaZero's MCTS-UCB is speed. In RMCTS, the search tree is explored in a breadth-first manner, so that network inferences naturally occur in large batches. This significantly reduces the GPU latency cost. We find that RMCTS is often more than 40 times faster than MCTS-UCB when searching a single root state, and about 3 times faster when searching a large batch of root states.
The recursion in RMCTS is based on computing optimized posterior policies at each game state in the search tree, starting from the leaves and working back up to the root. Here we use the posterior policy explored in "Monte--Carlo tree search as regularized policy optimization" (Grill, et al.) Their posterior policy is the unique policy which maximizes the expected reward given estimated action rewards minus a penalty for diverging from the prior policy.
The tree explored by RMCTS is not defined in an adaptive manner, as it is in MCTS-UCB. Instead, the RMCTS tree is defined by following prior network policies at each node. This is a disadvantage, but the speedup advantage is more significant, and in practice we find that RMCTS-trained networks match the quality of MCTS-UCB-trained networks in roughly one-third of the training time. We include timing and quality comparisons of RMCTS vs. MCTS-UCB for three games: Connect-4, Dots-and-Boxes, and Othello. 

---
# Reinforcement Learning Enhanced Multi-hop Reasoning for Temporal Knowledge Question Answering 

**Authors**: Wuzhenghong Wen, Chao Xue, Su Pan, Yuwei Sun, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.01195)  

**Abstract**: Temporal knowledge graph question answering (TKGQA) involves multi-hop reasoning over temporally constrained entity relationships in the knowledge graph to answer a given question. However, at each hop, large language models (LLMs) retrieve subgraphs with numerous temporally similar and semantically complex relations, increasing the risk of suboptimal decisions and error propagation. To address these challenges, we propose the multi-hop reasoning enhanced (MRE) framework, which enhances both forward and backward reasoning to improve the identification of globally optimal reasoning trajectories. Specifically, MRE begins with prompt engineering to guide the LLM in generating diverse reasoning trajectories for a given question. Valid reasoning trajectories are then selected for supervised fine-tuning, serving as a cold-start strategy. Finally, we introduce Tree-Group Relative Policy Optimization (T-GRPO), a recursive, tree-structured learning-by-exploration approach. At each hop, exploration establishes strong causal dependencies on the previous hop, while evaluation is informed by multi-path exploration feedback from subsequent hops. Experimental results on two TKGQA benchmarks indicate that the proposed MRE-based model consistently surpasses state-of-the-art (SOTA) approaches in handling complex multi-hop queries. Further analysis highlights improved interpretability and robustness to noisy temporal annotations. 

---
# ElecTwit: A Framework for Studying Persuasion in Multi-Agent Social Systems 

**Authors**: Michael Bao  

**Link**: [PDF](https://arxiv.org/pdf/2601.00994)  

**Abstract**: This paper introduces ElecTwit, a simulation framework designed to study persuasion within multi-agent systems, specifically emulating the interactions on social media platforms during a political election. By grounding our experiments in a realistic environment, we aimed to overcome the limitations of game-based simulations often used in prior research. We observed the comprehensive use of 25 specific persuasion techniques across most tested LLMs, encompassing a wider range than previously reported. The variations in technique usage and overall persuasion output between models highlight how different model architectures and training can impact the dynamics in realistic social simulations. Additionally, we observed unique phenomena such as "kernel of truth" messages and spontaneous developments with an "ink" obsession, where agents collectively demanded written proof. Our study provides a foundation for evaluating persuasive LLM agents in real-world contexts, ensuring alignment and preventing dangerous outcomes. 

---
# Context Collapse: In-Context Learning and Model Collapse 

**Authors**: Josef Ott  

**Link**: [PDF](https://arxiv.org/pdf/2601.00923)  

**Abstract**: This thesis investigates two key phenomena in large language models (LLMs): in-context learning (ICL) and model collapse. We study ICL in a linear transformer with tied weights trained on linear regression tasks, and show that minimising the in-context loss leads to a phase transition in the learned parameters. Above a critical context length, the solution develops a skew-symmetric component. We prove this by reducing the forward pass of the linear transformer under weight tying to preconditioned gradient descent, and then analysing the optimal preconditioner. This preconditioner includes a skew-symmetric component, which induces a rotation of the gradient direction. For model collapse, we use martingale and random walk theory to analyse simplified settings - linear regression and Gaussian fitting - under both replacing and cumulative data regimes. We strengthen existing results by proving almost sure convergence, showing that collapse occurs unless the data grows sufficiently fast or is retained over time. Finally, we introduce the notion of context collapse: a degradation of context during long generations, especially in chain-of-thought reasoning. This concept links the dynamics of ICL with long-term stability challenges in generative models. 

---
# Counterfactual Self-Questioning for Stable Policy Optimization in Language Models 

**Authors**: Mandar Parab  

**Link**: [PDF](https://arxiv.org/pdf/2601.00885)  

**Abstract**: Recent work on language model self-improvement shows that models can refine their own reasoning through reflection, verification, debate, or self-generated rewards. However, most existing approaches rely on external critics, learned reward models, or ensemble sampling, which increases complexity and training instability. We propose Counterfactual Self-Questioning, a framework in which a single language model generates and evaluates counterfactual critiques of its own reasoning. The method produces an initial reasoning trace, formulates targeted questions that challenge potential failure points, and generates alternative reasoning trajectories that expose incorrect assumptions or invalid steps. These counterfactual trajectories provide structured relative feedback that can be directly used for policy optimization without auxiliary models. Experiments on multiple mathematical reasoning benchmarks show that counterfactual self-questioning improves accuracy and training stability, particularly for smaller models, enabling scalable self-improvement using internally generated supervision alone. 

---
# Universal Conditional Logic: A Formal Language for Prompt Engineering 

**Authors**: Anthony Mikinka  

**Link**: [PDF](https://arxiv.org/pdf/2601.00880)  

**Abstract**: We present Universal Conditional Logic (UCL), a mathematical framework for prompt optimization that transforms prompt engineering from heuristic practice into systematic optimization. Through systematic evaluation (N=305, 11 models, 4 iterations), we demonstrate significant token reduction (29.8%, t(10)=6.36, p < 0.001, Cohen's d = 2.01) with corresponding cost savings. UCL's structural overhead function O_s(A) explains version-specific performance differences through the Over-Specification Paradox: beyond threshold S* = 0.509, additional specification degrades performance quadratically. Core mechanisms -- indicator functions (I_i in {0,1}), structural overhead (O_s = gamma * sum(ln C_k)), early binding -- are validated. Notably, optimal UCL configuration varies by model architecture -- certain models (e.g., Llama 4 Scout) require version-specific adaptations (V4.1). This work establishes UCL as a calibratable framework for efficient LLM interaction, with model-family-specific optimization as a key research direction. 

---
# Cultural Encoding in Large Language Models: The Existence Gap in AI-Mediated Brand Discovery 

**Authors**: Huang Junyao, Situ Ruimin, Ye Renqin  

**Link**: [PDF](https://arxiv.org/pdf/2601.00869)  

**Abstract**: As artificial intelligence systems increasingly mediate consumer information discovery,
brands face algorithmic invisibility. This study investigates Cultural Encoding in Large
Language Models (LLMs) -- systematic differences in brand recommendations arising from
training data composition. Analyzing 1,909 pure-English queries across 6 LLMs (GPT-4o,
Claude, Gemini, Qwen3, DeepSeek, Doubao) and 30 brands, we find Chinese LLMs exhibit 30.6
percentage points higher brand mention rates than International LLMs (88.9% vs. 58.3%,
p<.001). This disparity persists in identical English queries, indicating training data
geography -- not language -- drives the effect. We introduce the Existence Gap: brands
absent from LLM training corpora lack "existence" in AI responses regardless of quality.
Through a case study of Zhizibianjie (OmniEdge), a collaboration platform with 65.6%
mention rate in Chinese LLMs but 0% in International models (p<.001), we demonstrate how
Linguistic Boundary Barriers create invisible market entry obstacles. Theoretically, we
contribute the Data Moat Framework, conceptualizing AI-visible content as a VRIN strategic
resource. We operationalize Algorithmic Omnipresence -- comprehensive brand visibility
across LLM knowledge bases -- as the strategic objective for Generative Engine Optimization
(GEO). Managerially, we provide an 18-month roadmap for brands to build Data Moats
through semantic coverage, technical depth, and cultural localization. Our findings reveal
that in AI-mediated markets, the limits of a brand's "Data Boundaries" define the limits
of its "Market Frontiers." 

---
# Comment on: Your Brain on ChatGPT: Accumulation of Cognitive Debt When Using an AI Assistant for Essay Writing Tasks 

**Authors**: Milos Stankovic, Ella Hirche, Sarah Kollatzsch, Julia Nadine Doetsch  

**Link**: [PDF](https://arxiv.org/pdf/2601.00856)  

**Abstract**: Recently published work titled Your Brain on ChatGPT: Accumulation of Cognitive Debt When Using an AI Assistant for Essay Writing Task by Kosmyna et al. (2025) has sparked a vivid debate on the topic of artificial intelligence (AI) and human performance. We sincerely congratulate Kosmyna et al. for initiating such important research, collecting a valuable dataset, and establishing highly automated pipelines for Natural Language Processing (NLP) analyses and scoring. We aim to provide constructive comments that may improve the manuscript's readiness for peer-reviewed publication, as some results by Kosmyna et al. (2025) could be interpreted more conservatively. Our primary concerns focus on: (i) study design considerations, including the limited sample size; (ii) the reproducibility of the analyses; (iii) methodological issues related to the EEG analysis; (iv) inconsistencies in the reporting of results; and (v) limited transparency in several aspects of the study's procedures and findings. 

---
# Temporal Attack Pattern Detection in Multi-Agent AI Workflows: An Open Framework for Training Trace-Based Security Models 

**Authors**: Ron F. Del Rosario  

**Link**: [PDF](https://arxiv.org/pdf/2601.00848)  

**Abstract**: We present an openly documented methodology for fine-tuning language models to detect temporal attack patterns in multi-agent AI workflows using OpenTelemetry trace analysis. We curate a dataset of 80,851 examples from 18 public cybersecurity sources and 35,026 synthetic OpenTelemetry traces. We apply iterative QLoRA fine-tuning on resource-constrained ARM64 hardware (NVIDIA DGX Spark) through three training iterations with strategic augmentation. Our custom benchmark accuracy improves from 42.86% to 74.29%, a statistically significant 31.4-point gain. Targeted examples addressing specific knowledge gaps outperform indiscriminate scaling. Key contributions include: (1) synthetic trace generation methodology for multi-agent coordination attacks and regulatory violations, (2) empirical evidence that training data composition fundamentally determines behavior, and (3) complete open release of datasets, training scripts, and evaluation benchmarks on HuggingFace. While practical deployment requires human oversight due to false positive rates, this work establishes the first reproducible framework enabling practitioners to build custom agentic security models adapted to their threat landscapes. 

---
# Enhancing Temporal Awareness in LLMs for Temporal Point Processes 

**Authors**: Lili Chen, Wensheng Gan, Shuang Liang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.00845)  

**Abstract**: Temporal point processes (TPPs) are crucial for analyzing events over time and are widely used in fields such as finance, healthcare, and social systems. These processes are particularly valuable for understanding how events unfold over time, accounting for their irregularity and dependencies. Despite the success of large language models (LLMs) in sequence modeling, applying them to temporal point processes remains challenging. A key issue is that current methods struggle to effectively capture the complex interaction between temporal information and semantic context, which is vital for accurate event modeling. In this context, we introduce TPP-TAL (Temporal Point Processes with Enhanced Temporal Awareness in LLMs), a novel plug-and-play framework designed to enhance temporal reasoning within LLMs. Rather than using the conventional method of simply concatenating event time and type embeddings, TPP-TAL explicitly aligns temporal dynamics with contextual semantics before feeding this information into the LLM. This alignment allows the model to better perceive temporal dependencies and long-range interactions between events and their surrounding contexts. Through comprehensive experiments on several benchmark datasets, it is shown that TPP-TAL delivers substantial improvements in temporal likelihood estimation and event prediction accuracy, highlighting the importance of enhancing temporal awareness in LLMs for continuous-time event modeling. The code is made available at this https URL 

---
# OmniNeuro: A Multimodal HCI Framework for Explainable BCI Feedback via Generative AI and Sonification 

**Authors**: Ayda Aghaei Nia  

**Link**: [PDF](https://arxiv.org/pdf/2601.00843)  

**Abstract**: While Deep Learning has improved Brain-Computer Interface (BCI) decoding accuracy, clinical adoption is hindered by the "Black Box" nature of these algorithms, leading to user frustration and poor neuroplasticity outcomes. We propose OmniNeuro, a novel HCI framework that transforms the BCI from a silent decoder into a transparent feedback partner. OmniNeuro integrates three interpretability engines: (1) Physics (Energy), (2) Chaos (Fractal Complexity), and (3) Quantum-Inspired uncertainty modeling. These metrics drive real-time Neuro-Sonification and Generative AI Clinical Reports. Evaluated on the PhysioNet dataset ($N=109$), the system achieved a mean accuracy of 58.52%, with qualitative pilot studies ($N=3$) confirming that explainable feedback helps users regulate mental effort and reduces the "trial-and-error" phase. OmniNeuro is decoder-agnostic, acting as an essential interpretability layer for any state-of-the-art architecture. 

---
# Can We Trust AI Explanations? Evidence of Systematic Underreporting in Chain-of-Thought Reasoning 

**Authors**: Deep Pankajbhai Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2601.00830)  

**Abstract**: When AI systems explain their reasoning step-by-step, practitioners often assume these explanations reveal what actually influenced the AI's answer. We tested this assumption by embedding hints into questions and measuring whether models mentioned them. In a study of over 9,000 test cases across 11 leading AI models, we found a troubling pattern: models almost never mention hints spontaneously, yet when asked directly, they admit noticing them. This suggests models see influential information but choose not to report it. Telling models they are being watched does not help. Forcing models to report hints works, but causes them to report hints even when none exist and reduces their accuracy. We also found that hints appealing to user preferences are especially dangerous-models follow them most often while reporting them least. These findings suggest that simply watching AI reasoning is not enough to catch hidden influences. 

---
# Decomposing LLM Self-Correction: The Accuracy-Correction Paradox and Error Depth Hypothesis 

**Authors**: Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.00828)  

**Abstract**: Large Language Models (LLMs) are widely believed to possess self-correction capabilities, yet recent studies suggest that intrinsic self-correction--where models correct their own outputs without external feedback--remains largely ineffective. In this work, we systematically decompose self-correction into three distinct sub-capabilities: error detection, error localization, and error correction. Through cross-model experiments on GSM8K-Complex (n=500 per model, 346 total errors) with three major LLMs, we uncover a striking Accuracy-Correction Paradox: weaker models (GPT-3.5, 66% accuracy) achieve 1.6x higher intrinsic correction rates than stronger models (DeepSeek, 94% accuracy)--26.8% vs 16.7%. We propose the Error Depth Hypothesis: stronger models make fewer but deeper errors that resist self-correction. Error detection rates vary dramatically across architectures (10% to 82%), yet detection capability does not predict correction success--Claude detects only 10% of errors but corrects 29% intrinsically. Surprisingly, providing error location hints hurts all models. Our findings challenge linear assumptions about model capability and self-improvement, with important implications for the design of self-refinement pipelines. 

---
# Energy-Aware Routing to Large Reasoning Models 

**Authors**: Austin R. Ellis-Mohr, Max Hartman, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2601.00823)  

**Abstract**: Large reasoning models (LRMs) have heterogeneous inference energy costs based on which model is used and how much it reasons. To reduce energy, it is important to choose the right LRM and operate it in the right way. As a result, the performance of systems that dispatch tasks to different individual LRMs depend on the balance between mean energy provisioning and stochastic fluctuations. The critical regime is the unique operating point at which neither auxiliary energy nor baseline energy is systematically wasted. Increasing baseline supply shifts the system toward persistent over-supply and baseline-energy waste, while reducing supply induces persistent reliance on auxiliary energy. Yet in this regime, performance remains volatility-limited and so a second-order characterization provides further insights that we develop. Here, performance is governed by how variability is absorbed across time, models, and execution choices. This perspective highlights variance-aware routing and dispatch as a principled design axis, and provides a theoretical basis for developing energy-aware model routing policies. Routing behavior is characterized when dispatch policies are based on training-compute and inference-compute scaling laws for LRMs. 

---
# CogCanvas: Compression-Resistant Cognitive Artifacts for Long LLM Conversations 

**Authors**: Tao An  

**Link**: [PDF](https://arxiv.org/pdf/2601.00821)  

**Abstract**: Large language models face a fundamental tension between context window limits and information fidelity in long conversations. Existing approaches--truncation and summarization--either discard early information or lose nuanced details. We introduce CogCanvas, a training-free framework that extracts verbatim-grounded cognitive artifacts (decisions, facts, reminders) from conversation turns and organizes them into a temporal-aware graph for compression-resistant retrieval.
On the LoCoMo benchmark, CogCanvas achieves 34.7% overall accuracy, outperforming RAG (25.6%, +9.1pp) and GraphRAG (13.7%, +21.0pp). The advantage is most pronounced on temporal reasoning: 31.5% vs. 9.3% (RAG) and 5.0% (GraphRAG)--a +530% relative improvement. On multi-hop causal reasoning, CogCanvas achieves 81.0% pass rate vs. 40.0% for GraphRAG (+41.0pp). Controlled benchmarks show 97.5% recall (+78.5pp vs. summarization) with 93.0% exact match preservation.
While heavily-optimized approaches achieve higher absolute scores through dedicated training (EverMemOS: approximately 92%), our training-free approach provides practitioners with an immediately-deployable alternative that significantly outperforms standard baselines. Code and data: this https URL. 

---
# Agentic AI for Autonomous, Explainable, and Real-Time Credit Risk Decision-Making 

**Authors**: Chandra Sekhar Kubam  

**Link**: [PDF](https://arxiv.org/pdf/2601.00818)  

**Abstract**: Significant digitalization of financial services in a short period of time has led to an urgent demand to have autonomous, transparent and real-time credit risk decision making systems. The traditional machine learning models are effective in pattern recognition, but do not have the adaptive reasoning, situational awareness, and autonomy needed in modern financial operations. As a proposal, this paper presents an Agentic AI framework, or a system where AI agents view the world of dynamic credit independent of human observers, who then make actions based on their articulable decision-making paths. The research introduces a multi-agent system with reinforcing learning, natural language reasoning, explainable AI modules, and real-time data absorption pipelines as a means of assessing the risk profiles of borrowers with few humans being involved. The processes consist of agent collaboration protocol, risk-scoring engines, interpretability layers, and continuous feedback learning cycles. Findings indicate that decision speed, transparency and responsiveness is better than traditional credit scoring models. Nevertheless, there are still some practical limitations such as risks of model drift, inconsistencies in interpreting high dimensional data and regulatory uncertainties as well as infrastructure limitations in low-resource settings. The suggested system has a high prospective to transform credit analytics and future studies ought to be directed on dynamic regulatory compliance mobilizers, new agent teamwork, adversarial robustness, and large-scale implementation in cross-country credit ecosystems. 

---
# MathLedger: A Verifiable Learning Substrate with Ledger-Attested Feedback 

**Authors**: Ismail Ahmad Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2601.00816)  

**Abstract**: Contemporary AI systems achieve extraordinary performance yet remain opaque and non-verifiable, creating a crisis of trust for safety-critical deployment. We introduce MathLedger, a substrate for verifiable machine cognition that integrates formal verification, cryptographic attestation, and learning dynamics into a single epistemic loop. The system implements Reflexive Formal Learning (RFL), a symbolic analogue of gradient descent where updates are driven by verifier outcomes rather than statistical loss.
Phase I experiments validate the measurement and governance substrate under controlled conditions. CAL-EXP-3 validates measurement infrastructure (Delta p computation, variance tracking); separate stress tests confirm fail-closed governance triggers correctly under out-of-bounds conditions. No convergence or capability claims are made. The contribution is infrastructural: a working prototype of ledger-attested learning that enables auditability at scale.
Keywords: verifiable learning, formal verification, cryptographic attestation, reflexive feedback, fail-closed governance 

---
# Semantic Alignment of Multilingual Knowledge Graphs via Contextualized Vector Projections 

**Authors**: Abhishek Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.00814)  

**Abstract**: The paper presents our work on cross-lingual ontology alignment system which uses embedding based cosine similarity matching. The ontology entities are made contextually richer by creating descriptions using novel techniques. We use a fine-tuned transformer based multilingual model for generating better embeddings. We use cosine similarity to find positive ontology entities pairs and then apply threshold filtering to retain only highly similar entities. We have evaluated our work on OAEI-2022 multifarm track. We achieve 71% F1 score (78% recall and 65% precision) on the evaluation dataset, 16% increase from best baseline score. This suggests that our proposed alignment pipeline is able to capture the subtle cross-lingual similarities. 

---
# DARC: Drum accompaniment generation with fine-grained rhythm control 

**Authors**: Trey Brosnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02357)  

**Abstract**: In music creation, rapid prototyping is essential for exploring and refining ideas, yet existing generative tools often fall short when users require both structural control and stylistic flexibility. Prior approaches in stem-to-stem generation can condition on other musical stems but offer limited control over rhythm, and timbre-transfer methods allow users to specify specific rhythms, but cannot condition on musical context. We introduce DARC, a generative drum accompaniment model that conditions both on musical context from other stems and explicit rhythm prompts such as beatboxing or tapping tracks. Using parameter-efficient fine-tuning, we augment STAGE, a state-of-the-art drum stem generator, with fine-grained rhythm control while maintaining musical context awareness. 

---
# DatBench: Discriminative, Faithful, and Efficient VLM Evaluations 

**Authors**: Siddharth Joshi, Haoli Yin, Rishabh Adiga, Ricardo Monti, Aldo Carranza, Alex Fang, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, Darren Teh, David Schwab, Fan Pan, Haakon Mongstad, Jack Urbanek, Jason Lee, Jason Telanoff, Josh Wills, Kaleigh Mentzer, Luke Merrick, Parth Doshi, Paul Burstein, Pratyush Maini, Scott Loftin, Spandan Das, Tony Jiang, Vineeth Dorna, Zhengping Wang, Bogdan Gaza, Ari Morcos, Matthew Leavitt  

**Link**: [PDF](https://arxiv.org/pdf/2601.02316)  

**Abstract**: Empirical evaluation serves as the primary compass guiding research progress in foundation models. Despite a large body of work focused on training frontier vision-language models (VLMs), approaches to their evaluation remain nascent. To guide their maturation, we propose three desiderata that evaluations should satisfy: (1) faithfulness to the modality and application, (2) discriminability between models of varying quality, and (3) efficiency in compute. Through this lens, we identify critical failure modes that violate faithfulness and discriminability, misrepresenting model capabilities: (i) multiple-choice formats reward guessing, poorly reflect downstream use cases, and saturate early as models improve; (ii) blindly solvable questions, which can be answered without images, constitute up to 70% of some evaluations; and (iii) mislabeled or ambiguous samples compromise up to 42% of examples in certain datasets. Regarding efficiency, the computational burden of evaluating frontier models has become prohibitive: by some accounts, nearly 20% of development compute is devoted to evaluation alone. Rather than discarding existing benchmarks, we curate them via transformation and filtering to maximize fidelity and discriminability. We find that converting multiple-choice questions to generative tasks reveals sharp capability drops of up to 35%. In addition, filtering blindly solvable and mislabeled samples improves discriminative power while simultaneously reducing computational cost. We release DatBench-Full, a cleaned evaluation suite of 33 datasets spanning nine VLM capabilities, and DatBench, a discriminative subset that achieves 13x average speedup (up to 50x) while closely matching the discriminative power of the original datasets. Our work outlines a path toward evaluation practices that are both rigorous and sustainable as VLMs continue to scale. 

---
# Placement Semantics for Distributed Deep Learning: A Systematic Framework for Analyzing Parallelism Strategies 

**Authors**: Deep Pankajbhai Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2601.02311)  

**Abstract**: Training large language models requires distributing computation across many accelerators, yet practitioners select parallelism strategies (data, tensor, pipeline, ZeRO) through trial and error because no unified systematic framework predicts their behavior. We introduce placement semantics: each strategy is specified by how it places four training states (parameters, optimizer, gradients, activations) across devices using five modes (replicated, sharded, sharded-with-gather, materialized, offloaded). From placement alone, without implementation details, we derive memory consumption and communication volume. Our predictions match published results exactly: ZeRO-3 uses 8x less memory than data parallelism at 1.5x communication cost, as reported in the original paper. We prove two conditions (gradient integrity, state consistency) are necessary and sufficient for distributed training to match single-device results, and provide composition rules for combining strategies safely. The framework unifies ZeRO Stages 1-3, Fully Sharded Data Parallel (FSDP), tensor parallelism, and pipeline parallelism as instances with different placement choices. 

---
# pdfQA: Diverse, Challenging, and Realistic Question Answering over PDFs 

**Authors**: Tobias Schimanski, Imene Kolli, Jingwei Ni, Yu Fan, Ario Saeid Vaghefi, Elliott Ash, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2601.02285)  

**Abstract**: PDFs are the second-most used document type on the internet (after HTML). Yet, existing QA datasets commonly start from text sources or only address specific domains. In this paper, we present pdfQA, a multi-domain 2K human-annotated (real-pdfQA) and 2K synthetic dataset (syn-pdfQA) differentiating QA pairs in ten complexity dimensions (e.g., file type, source modality, source position, answer type). We apply and evaluate quality and difficulty filters on both datasets, obtaining valid and challenging QA pairs. We answer the questions with open-source LLMs, revealing existing challenges that correlate with our complexity dimensions. pdfQA presents a basis for end-to-end QA pipeline evaluation, testing diverse skill sets and local optimizations (e.g., in information retrieval or parsing). 

---
# TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters for Thin-Structure and Cross-Domain Binary Semantic Segmentation 

**Authors**: Salim Khazem  

**Link**: [PDF](https://arxiv.org/pdf/2601.02273)  

**Abstract**: Foundation segmentation models such as the Segment Anything Model (SAM) exhibit strong zero-shot generalization through large-scale pretraining, but adapting them to domain-specific semantic segmentation remains challenging, particularly for thin structures (e.g., retinal vessels) and noisy modalities (e.g., SAR imagery). Full fine-tuning is computationally expensive and risks catastrophic forgetting. We propose \textbf{TopoLoRA-SAM}, a topology-aware and parameter-efficient adaptation framework for binary semantic segmentation. TopoLoRA-SAM injects Low-Rank Adaptation (LoRA) into the frozen ViT encoder, augmented with a lightweight spatial convolutional adapter and optional topology-aware supervision via differentiable clDice. We evaluate our approach on five benchmarks spanning retinal vessel segmentation (DRIVE, STARE, CHASE\_DB1), polyp segmentation (Kvasir-SEG), and SAR sea/land segmentation (SL-SSDD), comparing against U-Net, DeepLabV3+, SegFormer, and Mask2Former. TopoLoRA-SAM achieves the best retina-average Dice and the best overall average Dice across datasets, while training only \textbf{5.2\%} of model parameters ($\sim$4.9M). On the challenging CHASE\_DB1 dataset, our method substantially improves segmentation accuracy and robustness, demonstrating that topology-aware parameter-efficient adaptation can match or exceed fully fine-tuned specialist models. Code is available at : this https URL 

---
# A Comparative Study of Custom CNNs, Pre-trained Models, and Transfer Learning Across Multiple Visual Datasets 

**Authors**: Annoor Sharara Akhand  

**Link**: [PDF](https://arxiv.org/pdf/2601.02246)  

**Abstract**: Convolutional Neural Networks (CNNs) are a standard approach for visual recognition due to their capacity to learn hierarchical representations from raw pixels. In practice, practitioners often choose among (i) training a compact custom CNN from scratch, (ii) using a large pre-trained CNN as a fixed feature extractor, and (iii) performing transfer learning via partial or full fine-tuning of a pre-trained backbone. This report presents a controlled comparison of these three paradigms across five real-world image classification datasets spanning road-surface defect recognition, agricultural variety identification, fruit/leaf disease recognition, pedestrian walkway encroachment recognition, and unauthorized vehicle recognition. Models are evaluated using accuracy and macro F1-score, complemented by efficiency metrics including training time per epoch and parameter counts. The results show that transfer learning consistently yields the strongest predictive performance, while the custom CNN provides an attractive efficiency--accuracy trade-off, especially when compute and memory budgets are constrained. 

---
# VIBE: Visual Instruction Based Editor 

**Authors**: Grigorii Alekseenko, Aleksandr Gordeev, Irina Tolstykh, Bulat Suleimanov, Vladimir Dokholyan, Georgii Fedorov, Sergey Yakubson, Aleksandra Tsybina, Mikhail Chernyshov, Maksim Kuprashevich  

**Link**: [PDF](https://arxiv.org/pdf/2601.02242)  

**Abstract**: Instruction-based image editing is among the fastest developing areas in generative AI. Over the past year, the field has reached a new level, with dozens of open-source models released alongside highly capable commercial systems. However, only a limited number of open-source approaches currently achieve real-world quality. In addition, diffusion backbones, the dominant choice for these pipelines, are often large and computationally expensive for many deployments and research settings, with widely used variants typically containing 6B to 20B parameters. This paper presents a compact, high-throughput instruction-based image editing pipeline that uses a modern 2B-parameter Qwen3-VL model to guide the editing process and the 1.6B-parameter diffusion model Sana1.5 for image generation. Our design decisions across architecture, data processing, training configuration, and evaluation target low-cost inference and strict source consistency while maintaining high quality across the major edit categories feasible at this scale. Evaluated on the ImgEdit and GEdit benchmarks, the proposed method matches or exceeds the performance of substantially heavier baselines, including models with several times as many parameters and higher inference cost, and is particularly strong on edits that require preserving the input image, such as an attribute adjustment, object removal, background edits, and targeted replacement. The model fits within 24 GB of GPU memory and generates edited images at up to 2K resolution in approximately 4 seconds on an NVIDIA H100 in BF16, without additional inference optimizations or distillation. 

---
# LLM-Empowered Functional Safety and Security by Design in Automotive Systems 

**Authors**: Nenad Petrovic, Vahid Zolfaghari, Fengjunjie Pan, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2601.02215)  

**Abstract**: This paper presents LLM-empowered workflow to support Software Defined Vehicle (SDV) software development, covering the aspects of security-aware system topology design, as well as event-driven decision-making code analysis. For code analysis we adopt event chains model which provides formal foundations to systematic validation of functional safety, taking into account the semantic validity of messages exchanged between key components, including both CAN and Vehicle Signal Specification (VSS). Analysis of security aspects for topology relies on synergy with Model-Driven Engineering (MDE) approach and Object Constraint Language (OCL) rules. Both locally deployable and proprietary solution are taken into account for evaluation within Advanced Driver-Assistance Systems (ADAS)-related scenarios. 

---
# Seeing the Unseen: Zooming in the Dark with Event Cameras 

**Authors**: Dachun Kai, Zeyu Xiao, Huyue Zhu, Jiaxiao Wang, Yueyi Zhang, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.02206)  

**Abstract**: This paper addresses low-light video super-resolution (LVSR), aiming to restore high-resolution videos from low-light, low-resolution (LR) inputs. Existing LVSR methods often struggle to recover fine details due to limited contrast and insufficient high-frequency information. To overcome these challenges, we present RetinexEVSR, the first event-driven LVSR framework that leverages high-contrast event signals and Retinex-inspired priors to enhance video quality under low-light scenarios. Unlike previous approaches that directly fuse degraded signals, RetinexEVSR introduces a novel bidirectional cross-modal fusion strategy to extract and integrate meaningful cues from noisy event data and degraded RGB frames. Specifically, an illumination-guided event enhancement module is designed to progressively refine event features using illumination maps derived from the Retinex model, thereby suppressing low-light artifacts while preserving high-contrast details. Furthermore, we propose an event-guided reflectance enhancement module that utilizes the enhanced event features to dynamically recover reflectance details via a multi-scale fusion mechanism. Experimental results show that our RetinexEVSR achieves state-of-the-art performance on three datasets. Notably, on the SDSD benchmark, our method can get up to 2.95 dB gain while reducing runtime by 65% compared to prior event-based methods. Code: this https URL. 

---
# NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation 

**Authors**: Huichao Zhang, Liao Qu, Yiheng Liu, Hang Chen, Yangyang Song, Yongsheng Dong, Shikun Sun, Xian Li, Xu Wang, Yi Jiang, Hu Ye, Bo Chen, Yiming Gao, Peng Liu, Akide Liu, Zhipeng Yang, Qili Deng, Linjie Xing, Jiyang Liu, Zhao Wang, Yang Zhou, Mingcong Liu, Yi Zhang, Qian He, Xiwei Hu, Zhongqi Qi, Jie Shao, Zhiye Fu, Shuai Wang, Fangmin Chen, Xuezhi Chai, Zhihua Wu, Yitong Wang, Zehuan Yuan, Daniel K. Du, Xinglong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02204)  

**Abstract**: We present NextFlow, a unified decoder-only autoregressive transformer trained on 6 trillion interleaved text-image discrete tokens. By leveraging a unified vision representation within a unified autoregressive architecture, NextFlow natively activates multimodal understanding and generation capabilities, unlocking abilities of image editing, interleaved content and video generation. Motivated by the distinct nature of modalities - where text is strictly sequential and images are inherently hierarchical - we retain next-token prediction for text but adopt next-scale prediction for visual generation. This departs from traditional raster-scan methods, enabling the generation of 1024x1024 images in just 5 seconds - orders of magnitude faster than comparable AR models. We address the instabilities of multi-scale generation through a robust training recipe. Furthermore, we introduce a prefix-tuning strategy for reinforcement learning. Experiments demonstrate that NextFlow achieves state-of-the-art performance among unified models and rivals specialized diffusion baselines in visual quality. 

---
# Code for Machines, Not Just Humans: Quantifying AI-Friendliness with Code Health Metrics 

**Authors**: Markus Borg, Nadim Hagatulah, Adam Tornhill, Emma Söderberg  

**Link**: [PDF](https://arxiv.org/pdf/2601.02200)  

**Abstract**: We are entering a hybrid era in which human developers and AI coding agents work in the same codebases. While industry practice has long optimized code for human comprehension, it is increasingly important to ensure that LLMs with different capabilities can edit code reliably. In this study, we investigate the concept of ``AI-friendly code'' via LLM-based refactoring on a dataset of 5,000 Python files from competitive programming. We find a meaningful association between CodeHealth, a quality metric calibrated for human comprehension, and semantic preservation after AI refactoring. Our findings confirm that human-friendly code is also more compatible with AI tooling. These results suggest that organizations can use CodeHealth to guide where AI interventions are lower risk and where additional human oversight is warranted. Investing in maintainability not only helps humans; it also prepares for large-scale AI adoption. 

---
# FormationEval, an open multiple-choice benchmark for petroleum geoscience 

**Authors**: Almaz Ermilov  

**Link**: [PDF](https://arxiv.org/pdf/2601.02158)  

**Abstract**: This paper presents FormationEval, an open multiple-choice question benchmark for evaluating language models on petroleum geoscience and subsurface disciplines. The dataset contains 505 questions across seven domains including petrophysics, petroleum geology and reservoir engineering, derived from three authoritative sources using a reasoning model with detailed instructions and a concept-based approach that avoids verbatim copying of copyrighted text. Each question includes source metadata to support traceability and audit. The evaluation covers 72 models from major providers including OpenAI, Anthropic, Google, Meta and open-weight alternatives. The top performers achieve over 97\% accuracy, with Gemini 3 Pro Preview reaching 99.8\%, while tier and domain gaps persist. Among open-weight models, GLM-4.7 leads at 98.6\%, with several DeepSeek, Llama, Qwen and Mistral models also exceeding 93\%. The performance gap between open-weight and closed models is narrower than expected, with several lower-cost open-weight models exceeding 90\% accuracy. Petrophysics emerges as the most challenging domain across all models, while smaller models show wider performance variance. Residual length bias in the dataset (correct answers tend to be longer) is documented along with bias mitigation strategies applied during construction. The benchmark, evaluation code and results are publicly available. 

---
# Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting 

**Authors**: Muxi Diao, Lele Yang, Wuxuan Gong, Yutong Zhang, Zhonghao Yan, Yufei Han, Kongming Liang, Weiran Xu, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.02151)  

**Abstract**: Supervised Fine-Tuning (SFT) is the standard paradigm for domain adaptation, yet it frequently incurs the cost of catastrophic forgetting. In sharp contrast, on-policy Reinforcement Learning (RL) effectively preserves general capabilities. We investigate this discrepancy and identify a fundamental distributional gap: while RL aligns with the model's internal belief, SFT forces the model to fit external supervision. This mismatch often manifests as "Confident Conflicts" tokens characterized by low probability but low entropy. In these instances, the model is highly confident in its own prediction but is forced to learn a divergent ground truth, triggering destructive gradient updates. To address this, we propose Entropy-Adaptive Fine-Tuning (EAFT). Unlike methods relying solely on prediction probability, EAFT utilizes token-level entropy as a gating mechanism to distinguish between epistemic uncertainty and knowledge conflict. This allows the model to learn from uncertain samples while suppressing gradients on conflicting data. Extensive experiments on Qwen and GLM series (ranging from 4B to 32B parameters) across mathematical, medical, and agentic domains confirm our hypothesis. EAFT consistently matches the downstream performance of standard SFT while significantly mitigating the degradation of general capabilities. 

---
# AI-enhanced tuning of quantum dot Hamiltonians toward Majorana modes 

**Authors**: Mateusz Krawczyk, Jarosław Pawłowski  

**Link**: [PDF](https://arxiv.org/pdf/2601.02149)  

**Abstract**: We propose a neural network-based model capable of learning the broad landscape of working regimes in quantum dot simulators, and using this knowledge to autotune these devices - based on transport measurements - toward obtaining Majorana modes in the structure. The model is trained in an unsupervised manner on synthetic data in the form of conductance maps, using a physics-informed loss that incorporates key properties of Majorana zero modes. We show that, with appropriate training, a deep vision-transformer network can efficiently memorize relation between Hamiltonian parameters and structures on conductance maps and use it to propose parameters update for a quantum dot chain that drive the system toward topological phase. Starting from a broad range of initial detunings in parameter space, a single update step is sufficient to generate nontrivial zero modes. Moreover, by enabling an iterative tuning procedure - where the system acquires updated conductance maps at each step - we demonstrate that the method can address a much larger region of the parameter space. 

---
# BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models 

**Authors**: Sunny Gupta, Shounak Das, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02147)  

**Abstract**: Vision language foundation models such as CLIP exhibit impressive zero-shot generalization yet remain vulnerable to spurious correlations across visual and textual modalities. Existing debiasing approaches often address a single modality either visual or textual leading to partial robustness and unstable adaptation under distribution shifts. We propose a bilateral prompt optimization framework (BiPrompt) that simultaneously mitigates non-causal feature reliance in both modalities during test-time adaptation. On the visual side, it employs structured attention-guided erasure to suppress background activations and enforce orthogonal prediction consistency between causal and spurious regions. On the textual side, it introduces balanced prompt normalization, a learnable re-centering mechanism that aligns class embeddings toward an isotropic semantic space. Together, these modules jointly minimize conditional mutual information between spurious cues and predictions, steering the model toward causal, domain invariant reasoning without retraining or domain supervision. Extensive evaluations on real-world and synthetic bias benchmarks demonstrate consistent improvements in both average and worst-group accuracies over prior test-time debiasing methods, establishing a lightweight yet effective path toward trustworthy and causally grounded vision-language adaptation. 

---
# Routing by Analogy: kNN-Augmented Expert Assignment for Mixture-of-Experts 

**Authors**: Boxuan Lyu, Soichiro Murakami, Hidetaka Kamigaito, Peinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02144)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale large language models efficiently by employing a parametric "router" to dispatch tokens to a sparse subset of experts. Typically, this router is trained once and then frozen, rendering routing decisions brittle under distribution shifts. We address this limitation by introducing kNN-MoE, a retrieval-augmented routing framework that reuses optimal expert assignments from a memory of similar past cases. This memory is constructed offline by directly optimizing token-wise routing logits to maximize the likelihood on a reference set. Crucially, we use the aggregate similarity of retrieved neighbors as a confidence-driven mixing coefficient, thus allowing the method to fall back to the frozen router when no relevant cases are found. Experiments show kNN-MoE outperforms zero-shot baselines and rivals computationally expensive supervised fine-tuning. 

---
# Remote Sensing Change Detection via Weak Temporal Supervision 

**Authors**: Xavier Bou, Elliot Vincent, Gabriele Facciolo, Rafael Grompone von Gioi, Jean-Michel Morel, Thibaud Ehret  

**Link**: [PDF](https://arxiv.org/pdf/2601.02126)  

**Abstract**: Semantic change detection in remote sensing aims to identify land cover changes between bi-temporal image pairs. Progress in this area has been limited by the scarcity of annotated datasets, as pixel-level annotation is costly and time-consuming. To address this, recent methods leverage synthetic data or generate artificial change pairs, but out-of-domain generalization remains limited. In this work, we introduce a weak temporal supervision strategy that leverages additional temporal observations of existing single-temporal datasets, without requiring any new annotations. Specifically, we extend single-date remote sensing datasets with new observations acquired at different times and train a change detection model by assuming that real bi-temporal pairs mostly contain no change, while pairing images from different locations to generate change examples. To handle the inherent noise in these weak labels, we employ an object-aware change map generation and an iterative refinement process. We validate our approach on extended versions of the FLAIR and IAILD aerial datasets, achieving strong zero-shot and low-data regime performance across different benchmarks. Lastly, we showcase results over large areas in France, highlighting the scalability potential of our method. 

---
# SingingBot: An Avatar-Driven System for Robotic Face Singing Performance 

**Authors**: Zhuoxiong Xu, Xuanchen Li, Yuhao Cheng, Fei Xu, Yichao Yan, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02125)  

**Abstract**: Equipping robotic faces with singing capabilities is crucial for empathetic Human-Robot Interaction. However, existing robotic face driving research primarily focuses on conversations or mimicking static expressions, struggling to meet the high demands for continuous emotional expression and coherence in singing. To address this, we propose a novel avatar-driven framework for appealing robotic singing. We first leverage portrait video generation models embedded with extensive human priors to synthesize vivid singing avatars, providing reliable expression and emotion guidance. Subsequently, these facial features are transferred to the robot via semantic-oriented mapping functions that span a wide expression space. Furthermore, to quantitatively evaluate the emotional richness of robotic singing, we propose the Emotion Dynamic Range metric to measure the emotional breadth within the Valence-Arousal space, revealing that a broad emotional spectrum is crucial for appealing performances. Comprehensive experiments prove that our method achieves rich emotional expressions while maintaining lip-audio synchronization, significantly outperforming existing approaches. 

---
# DeCode: Decoupling Content and Delivery for Medical QA 

**Authors**: Po-Jen Ko, Chen-Han Tsai, Yu-Shao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.02123)  

**Abstract**: Large language models (LLMs) exhibit strong medical knowledge and can generate factually accurate responses. However, existing models often fail to account for individual patient contexts, producing answers that are clinically correct yet poorly aligned with patients' needs. In this work, we introduce DeCode, a training-free, model-agnostic framework that adapts existing LLMs to produce contextualized answers in clinical settings. We evaluate DeCode on OpenAI HealthBench, a comprehensive and challenging benchmark designed to assess clinical relevance and validity of LLM responses. DeCode improves the previous state of the art from $28.4\%$ to $49.8\%$, corresponding to a $75\%$ relative improvement. Experimental results suggest the effectiveness of DeCode in improving clinical question answering of LLMs. 

---
# Inferring Network Evolutionary History via Structure-State Coupled Learning 

**Authors**: En Xu, Shihe Zhou, Huandong Wang, Jingtao Ding, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02121)  

**Abstract**: Inferring a network's evolutionary history from a single final snapshot with limited temporal annotations is fundamental yet challenging. Existing approaches predominantly rely on topology alone, which often provides insufficient and noisy cues. This paper leverages network steady-state dynamics -- converged node states under a given dynamical process -- as an additional and widely accessible observation for network evolution history inference. We propose CS$^2$, which explicitly models structure-state coupling to capture how topology modulates steady states and how the two signals jointly improve edge discrimination for formation-order recovery. Experiments on six real temporal networks, evaluated under multiple dynamical processes, show that CS$^2$ consistently outperforms strong baselines, improving pairwise edge precedence accuracy by 4.0% on average and global ordering consistency (Spearman-$\rho$) by 7.7% on average. CS$^2$ also more faithfully recovers macroscopic evolution trajectories such as clustering formation, degree heterogeneity, and hub growth. Moreover, a steady-state-only variant remains competitive when reliable topology is limited, highlighting steady states as an independent signal for evolution inference. 

---
# LION-DG: Layer-Informed Initialization with Deep Gradient Protocols for Accelerated Neural Network Training 

**Authors**: Hyunjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.02105)  

**Abstract**: Weight initialization remains decisive for neural network optimization, yet existing methods are largely layer-agnostic. We study initialization for deeply-supervised architectures with auxiliary classifiers, where untrained auxiliary heads can destabilize early training through gradient interference.
We propose LION-DG, a layer-informed initialization that zero-initializes auxiliary classifier heads while applying standard He-initialization to the backbone. We prove that this implements Gradient Awakening: auxiliary gradients are exactly zero at initialization, then phase in naturally as weights grow -- providing an implicit warmup without hyperparameters.
Experiments on CIFAR-10 and CIFAR-100 with DenseNet-DS and ResNet-DS architectures demonstrate: (1) DenseNet-DS: +8.3% faster convergence on CIFAR-10 with comparable accuracy, (2) Hybrid approach: Combining LSUV with LION-DG achieves best accuracy (81.92% on CIFAR-10), (3) ResNet-DS: Positive speedup on CIFAR-100 (+11.3%) with side-tap auxiliary design.
We identify architecture-specific trade-offs and provide clear guidelines for practitioners. LION-DG is simple, requires zero hyperparameters, and adds no computational overhead. 

---
# Vision-Based Early Fault Diagnosis and Self-Recovery for Strawberry Harvesting Robots 

**Authors**: Meili Sun, Chunjiang Zhao, Lichao Yang, Hao Liu, Shimin Hu, Ya Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2601.02085)  

**Abstract**: Strawberry harvesting robots faced persistent challenges such as low integration of visual perception, fruit-gripper misalignment, empty grasping, and strawberry slippage from the gripper due to insufficient gripping force, all of which compromised harvesting stability and efficiency in orchard environments. To overcome these issues, this paper proposed a visual fault diagnosis and self-recovery framework that integrated multi-task perception with corrective control strategies. At the core of this framework was SRR-Net, an end-to-end multi-task perception model that simultaneously performed strawberry detection, segmentation, and ripeness estimation, thereby unifying visual perception with fault diagnosis. Based on this integrated perception, a relative error compensation method based on the simultaneous target-gripper detection was designed to address positional misalignment, correcting deviations when error exceeded the tolerance threshold. To mitigate empty grasping and fruit-slippage faults, an early abort strategy was implemented. A micro-optical camera embedded in the end-effector provided real-time visual feedback, enabling grasp detection during the deflating stage and strawberry slip prediction during snap-off through MobileNet V3-Small classifier and a time-series LSTM classifier. Experiments demonstrated that SRR-Net maintained high perception accuracy. For detection, it achieved a precision of 0.895 and recall of 0.813 on strawberries, and 0.972/0.958 on hands. In segmentation, it yielded a precision of 0.887 and recall of 0.747 for strawberries, and 0.974/0.947 for hands. For ripeness estimation, SRR-Net attained a mean absolute error of 0.035, while simultaneously supporting multi-task perception and sustaining a competitive inference speed of 163.35 FPS. 

---
# The Homogeneity Trap: Spectral Collapse in Doubly-Stochastic Deep Networks 

**Authors**: Yizhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02080)  

**Abstract**: Doubly-stochastic matrices (DSM) are increasingly utilized in structure-preserving deep architectures -- such as Optimal Transport layers and Sinkhorn-based attention -- to enforce numerical stability and probabilistic interpretability. In this work, we identify a critical spectral degradation phenomenon inherent to these constraints, termed the Homogeneity Trap. We demonstrate that the maximum-entropy bias, typical of Sinkhorn-based projections, drives the mixing operator towards the uniform barycenter, thereby suppressing the subdominant singular value \sigma_2 and filtering out high-frequency feature components. We derive a spectral bound linking \sigma_2 to the network's effective depth, showing that high-entropy constraints restrict feature transformation to a shallow effective receptive field. Furthermore, we formally demonstrate that Layer Normalization fails to mitigate this collapse in noise-dominated regimes; specifically, when spectral filtering degrades the Signal-to-Noise Ratio (SNR) below a critical threshold, geometric structure is irreversibly lost to noise-induced orthogonal collapse. Our findings highlight a fundamental trade-off between entropic stability and spectral expressivity in DSM-constrained networks. 

---
# Deferred Commitment Decoding for Diffusion Language Models with Confidence-Aware Sliding Windows 

**Authors**: Yingte Shu, Yuchuan Tian, Chao Xu, Yunhe Wang, Hanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.02076)  

**Abstract**: Diffusion language models (DLMs) have recently emerged as a strong alternative to autoregressive models by enabling parallel text generation. To improve inference efficiency and KV-cache compatibility, prior work commonly adopts block-based diffusion, decoding tokens block by block. However, this paradigm suffers from a structural limitation that we term Boundary-Induced Context Truncation (BICT): undecoded tokens near block boundaries are forced to commit without access to nearby future context, even when such context could substantially reduce uncertainty. This limitation degrades decoding confidence and generation quality, especially for tasks requiring precise reasoning, such as mathematical problem solving and code generation. We propose Deferred Commitment Decoding (DCD), a novel, training-free decoding strategy that mitigates this issue. DCD maintains a confidence-aware sliding window over masked tokens, resolving low-uncertainty tokens early while deferring high-uncertainty tokens until sufficient contextual evidence becomes available. This design enables effective bidirectional information flow within the decoding window without sacrificing efficiency. Extensive experiments across multiple diffusion language models, benchmarks, and caching configurations show that DCD improves generation accuracy by 1.39% with comparable time on average compared to fixed block-based diffusion methods, with the most significant improvement reaching 9.0%. These results demonstrate that deferring token commitment based on uncertainty is a simple yet effective principle for improving both the quality and efficiency of diffusion language model decoding. 

---
# Cost-Efficient Cross-Lingual Retrieval-Augmented Generation for Low-Resource Languages: A Case Study in Bengali Agricultural Advisory 

**Authors**: Md. Asif Hossain, Nabil Subhan, Mantasha Rahman Mahi, Jannatul Ferdous Nabila  

**Link**: [PDF](https://arxiv.org/pdf/2601.02065)  

**Abstract**: Access to reliable agricultural advisory remains limited in many developing regions due to a persistent language barrier: authoritative agricultural manuals are predominantly written in English, while farmers primarily communicate in low-resource local languages such as Bengali. Although recent advances in Large Language Models (LLMs) enable natural language interaction, direct generation in low-resource languages often exhibits poor fluency and factual inconsistency, while cloud-based solutions remain cost-prohibitive. This paper presents a cost-efficient, cross-lingual Retrieval-Augmented Generation (RAG) framework for Bengali agricultural advisory that emphasizes factual grounding and practical deployability. The proposed system adopts a translation-centric architecture in which Bengali user queries are translated into English, enriched through domain-specific keyword injection to align colloquial farmer terminology with scientific nomenclature, and answered via dense vector retrieval over a curated corpus of English agricultural manuals (FAO, IRRI). The generated English response is subsequently translated back into Bengali to ensure accessibility. The system is implemented entirely using open-source models and operates on consumer-grade hardware without reliance on paid APIs. Experimental evaluation demonstrates reliable source-grounded responses, robust rejection of out-of-domain queries, and an average end-to-end latency below 20 seconds. The results indicate that cross-lingual retrieval combined with controlled translation offers a practical and scalable solution for agricultural knowledge access in low-resource language settings 

---
# Perish or Flourish? A Holistic Evaluation of Large Language Models for Code Generation in Functional Programming 

**Authors**: Nguyet-Anh H. Lang, Eric Lang, Thanh Le-Cong, Bach Le, Quyet-Thang Huynh  

**Link**: [PDF](https://arxiv.org/pdf/2601.02060)  

**Abstract**: Functional programming provides strong foundations for developing reliable and secure software systems, yet its adoption remains not widespread due to the steep learning curve. Recent advances in Large Language Models (LLMs) for code generation present new opportunities to lower these barriers. However, extensive evaluations of LLMs largely focus on imperative programming languages, and their capabilities in functional programming languages (FP) remain underexplored. To address this gap, we introduce FPEval, a holistic evaluation framework built on FPBench, a new benchmark of 721 programming tasks across three difficulty levels on three mainstream FP languages: Haskell, Ocaml and Scala. FPEval provides compehensive evaluation infrastructures with both test validations with comprehensive test suites and static analysis tools to assess both functional correctness and code style and maintainability. Using this framework, we evaluate state-of-the-art LLMs, including GPT-3.5, GPT-4o, and GPT-5, for code generation in functional programming languages and Java as an imperative baseline. Our results demonstrate that LLM performance in functional programming improves substantially with model advancement; however, error rates remain significantly higher in purely functional languages (Haskell and OCaml) than in hybrid (Scala) or imperative (Java) languages. Moreover, LLMs frequently generate non-idiomatic functional code that follows imperative patterns, raising concerns about code style and long-term maintainability. Finally, we show that LLMs can partially self-repair both correctness and quality issues when provided with static analysis feedback and hand-crafted instructions for common types of issues. 

---
# Agentic Retoucher for Text-To-Image Generation 

**Authors**: Shaocheng Shen, Jianfeng Liang. Chunlei Cai, Cong Geng, Huiyu Duan, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2601.02046)  

**Abstract**: Text-to-image (T2I) diffusion models such as SDXL and FLUX have achieved impressive photorealism, yet small-scale distortions remain pervasive in limbs, face, text and so on. Existing refinement approaches either perform costly iterative re-generation or rely on vision-language models (VLMs) with weak spatial grounding, leading to semantic drift and unreliable local edits. To close this gap, we propose Agentic Retoucher, a hierarchical decision-driven framework that reformulates post-generation correction as a human-like perception-reasoning-action loop. Specifically, we design (1) a perception agent that learns contextual saliency for fine-grained distortion localization under text-image consistency cues, (2) a reasoning agent that performs human-aligned inferential diagnosis via progressive preference alignment, and (3) an action agent that adaptively plans localized inpainting guided by user preference. This design integrates perceptual evidence, linguistic reasoning, and controllable correction into a unified, self-corrective decision process. To enable fine-grained supervision and quantitative evaluation, we further construct GenBlemish-27K, a dataset of 6K T2I images with 27K annotated artifact regions across 12 categories. Extensive experiments demonstrate that Agentic Retoucher consistently outperforms state-of-the-art methods in perceptual quality, distortion localization and human preference alignment, establishing a new paradigm for self-corrective and perceptually reliable T2I generation. 

---
# The New Compiler Stack: A Survey on the Synergy of LLMs and Compilers 

**Authors**: Shuoming Zhang, Jiacheng Zhao, Qiuchu Yu, Chunwei Xia, Zheng Wang, Xiaobing Feng, Huimin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2601.02045)  

**Abstract**: This survey has provided a systematic overview of the emerging field of LLM-enabled compilation by addressing several key research questions. We first answered how LLMs are being integrated by proposing a comprehensive, multi-dimensional taxonomy that categorizes works based on their Design Philosophy (Selector, Translator, Generator), LLM Methodology, their operational Level of Code Abstraction, and the specific Task Type they address. In answering what advancements these approaches offer, we identified three primary benefits: the democratization of compiler development, the discovery of novel optimization strategies, and the broadening of the compiler's traditional scope. Finally, in addressing the field's challenges and opportunities, we highlighted the critical hurdles of ensuring correctness and achieving scalability, while identifying the development of hybrid systems as the most promising path forward. By providing these answers, this survey serves as a foundational roadmap for researchers and practitioners, charting the course for a new generation of LLM-powered, intelligent, adaptive and synergistic compilation tools. 

---
# Output Embedding Centering for Stable LLM Pretraining 

**Authors**: Felix Stollenwerk, Anna Lokrantz, Niclas Hertzberg  

**Link**: [PDF](https://arxiv.org/pdf/2601.02031)  

**Abstract**: Pretraining of large language models is not only expensive but also prone to certain training instabilities. A specific instability that often occurs for large learning rates at the end of training is output logit divergence. The most widely used mitigation strategy, z-loss, merely addresses the symptoms rather than the underlying cause of the problem. In this paper, we analyze the instability from the perspective of the output embeddings' geometry and identify its cause. Based on this, we propose output embedding centering (OEC) as a new mitigation strategy, and prove that it suppresses output logit divergence. OEC can be implemented in two different ways, as a deterministic operation called {\mu}-centering, or a regularization method called {\mu}-loss. Our experiments show that both variants outperform z-loss in terms of training stability and learning rate sensitivity. In particular, they ensure that training converges even for large learning rates when z-loss fails. Furthermore, we find that {\mu}-loss is significantly less sensitive to regularization hyperparameter tuning than z-loss. 

---
# Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Literal Extraction, Logical Inference, and Hallucination Risks in Long-Context LLMs 

**Authors**: Amirali Ebrahimzadeh, Seyyed M. Salili  

**Link**: [PDF](https://arxiv.org/pdf/2601.02023)  

**Abstract**: Large language models (LLMs) increasingly support very long input contexts. Yet it remains unclear how reliably they extract and infer information at scale. Performance varies with context length and strongly interacts with how information is distributed in real-world corpora. Motivated by these observations, we study how fact placement, corpus-level fact distributions, and Don't Make It Up prompts influence model behavior. We introduce an extended needle-in-a-haystack benchmark across four production-scale models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior work, we separately evaluate literal extraction, logical inference, and hallucination risk. Our study considers both positional effects and realistic distributions of evidence across long contexts, as well as prompts that explicitly discourage fabrication. We find that longer contexts alone do not guarantee better performance and can be detrimental when relevant evidence is diluted or widely dispersed. Performance varies substantially across models: some show severe degradation under realistic conditions, while others remain more robust at longer context lengths. Anti-hallucination (AH) instructions can make some models overly conservative, sharply reducing accuracy in literal extraction and logical inference. While we do not directly compare retrieval-augmented generation (RAG) and cache-augmented generation (CAG), our results suggest many failures stem from ineffective context utilization. Models often struggle to identify and prioritize relevant information even when it is present. These findings have direct practical implications, as enterprise workflows increasingly involve pasting large volumes of unfiltered documents into LLM prompts. Effective context length and model-specific robustness to long contexts are therefore critical for reliable LLM deployment in research and business. 

---
# Enhancing Object Detection with Privileged Information: A Model-Agnostic Teacher-Student Approach 

**Authors**: Matthias Bartolo, Dylan Seychell, Gabriel Hili, Matthew Montebello, Carl James Debono, Saviour Formosa, Konstantinos Makantasis  

**Link**: [PDF](https://arxiv.org/pdf/2601.02016)  

**Abstract**: This paper investigates the integration of the Learning Using Privileged Information (LUPI) paradigm in object detection to exploit fine-grained, descriptive information available during training but not at inference. We introduce a general, model-agnostic methodology for injecting privileged information-such as bounding box masks, saliency maps, and depth cues-into deep learning-based object detectors through a teacher-student architecture. Experiments are conducted across five state-of-the-art object detection models and multiple public benchmarks, including UAV-based litter detection datasets and Pascal VOC 2012, to assess the impact on accuracy, generalization, and computational efficiency. Our results demonstrate that LUPI-trained students consistently outperform their baseline counterparts, achieving significant boosts in detection accuracy with no increase in inference complexity or model size. Performance improvements are especially marked for medium and large objects, while ablation studies reveal that intermediate weighting of teacher guidance optimally balances learning from privileged and standard inputs. The findings affirm that the LUPI framework provides an effective and practical strategy for advancing object detection systems in both resource-constrained and real-world settings. 

---
# Surprisal and Metaphor Novelty: Moderate Correlations and Divergent Scaling Effects 

**Authors**: Omar Momen, Emilie Sitter, Berenike Herrmann, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2601.02015)  

**Abstract**: Novel metaphor comprehension involves complex semantic processes and linguistic creativity, making it an interesting task for studying language models (LMs). This study investigates whether surprisal, a probabilistic measure of predictability in LMs, correlates with different metaphor novelty datasets. We analyse surprisal from 16 LM variants on corpus-based and synthetic metaphor novelty datasets. We explore a cloze-style surprisal method that conditions on full-sentence context. Results show that LMs yield significant moderate correlations with scores/labels of metaphor novelty. We further identify divergent scaling patterns: on corpus-based data, correlation strength decreases with model size (inverse scaling effect), whereas on synthetic data it increases (Quality-Power Hypothesis). We conclude that while surprisal can partially account for annotations of metaphor novelty, it remains a limited metric of linguistic creativity. 

---
# A neural network for modeling human concept formation, understanding and communication 

**Authors**: Liangxuan Guo, Haoyang Chen, Yang Chen, Yanchao Bi, Shan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02010)  

**Abstract**: A remarkable capability of the human brain is to form more abstract conceptual representations from sensorimotor experiences and flexibly apply them independent of direct sensory inputs. However, the computational mechanism underlying this ability remains poorly understood. Here, we present a dual-module neural network framework, the CATS Net, to bridge this gap. Our model consists of a concept-abstraction module that extracts low-dimensional conceptual representations, and a task-solving module that performs visual judgement tasks under the hierarchical gating control of the formed concepts. The system develops transferable semantic structure based on concept representations that enable cross-network knowledge transfer through conceptual communication. Model-brain fitting analyses reveal that these emergent concept spaces align with both neurocognitive semantic model and brain response structures in the human ventral occipitotemporal cortex, while the gating mechanisms mirror that in the semantic control brain network. This work establishes a unified computational framework that can offer mechanistic insights for understanding human conceptual cognition and engineering artificial systems with human-like conceptual intelligence. 

---
# Exploring Approaches for Detecting Memorization of Recommender System Data in Large Language Models 

**Authors**: Antonio Colacicco, Vito Guida, Dario Di Palma, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2601.02002)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in recommendation scenarios due to their strong natural language understanding and generation capabilities. However, they are trained on vast corpora whose contents are not publicly disclosed, raising concerns about data leakage. Recent work has shown that the MovieLens-1M dataset is memorized by both the LLaMA and OpenAI model families, but the extraction of such memorized data has so far relied exclusively on manual prompt engineering. In this paper, we pose three main questions: Is it possible to enhance manual prompting? Can LLM memorization be detected through methods beyond manual prompting? And can the detection of data leakage be automated? To address these questions, we evaluate three approaches: (i) jailbreak prompt engineering; (ii) unsupervised latent knowledge discovery, probing internal activations via Contrast-Consistent Search (CCS) and Cluster-Norm; and (iii) Automatic Prompt Engineering (APE), which frames prompt discovery as a meta-learning process that iteratively refines candidate instructions. Experiments on MovieLens-1M using LLaMA models show that jailbreak prompting does not improve the retrieval of memorized items and remains inconsistent; CCS reliably distinguishes genuine from fabricated movie titles but fails on numerical user and rating data; and APE retrieves item-level information with moderate success yet struggles to recover numerical interactions. These findings suggest that automatically optimizing prompts is the most promising strategy for extracting memorized samples. 

---
# Exploring Diversity, Novelty, and Popularity Bias in ChatGPT's Recommendations 

**Authors**: Dario Di Palma, Giovanni Maria Biancofiore, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2601.01997)  

**Abstract**: ChatGPT has emerged as a versatile tool, demonstrating capabilities across diverse domains. Given these successes, the Recommender Systems (RSs) community has begun investigating its applications within recommendation scenarios primarily focusing on accuracy. While the integration of ChatGPT into RSs has garnered significant attention, a comprehensive analysis of its performance across various dimensions remains largely unexplored. Specifically, the capabilities of providing diverse and novel recommendations or exploring potential biases such as popularity bias have not been thoroughly examined. As the use of these models continues to expand, understanding these aspects is crucial for enhancing user satisfaction and achieving long-term personalization.
This study investigates the recommendations provided by ChatGPT-3.5 and ChatGPT-4 by assessing ChatGPT's capabilities in terms of diversity, novelty, and popularity bias. We evaluate these models on three distinct datasets and assess their performance in Top-N recommendation and cold-start scenarios. The findings reveal that ChatGPT-4 matches or surpasses traditional recommenders, demonstrating the ability to balance novelty and diversity in recommendations. Furthermore, in the cold-start scenario, ChatGPT models exhibit superior performance in both accuracy and novelty, suggesting they can be particularly beneficial for new users. This research highlights the strengths and limitations of ChatGPT's recommendations, offering new perspectives on the capacity of these models to provide recommendations beyond accuracy-focused metrics. 

---
# VIT-Ped: Visionary Intention Transformer for Pedestrian Behavior Analysis 

**Authors**: Aly R. Elkammar, Karim M. Gamaleldin, Catherine M. Elias  

**Link**: [PDF](https://arxiv.org/pdf/2601.01989)  

**Abstract**: Pedestrian Intention prediction is one of the key technologies in the transition from level 3 to level 4 autonomous driving. To understand pedestrian crossing behaviour, several elements and features should be taken into consideration to make the roads of tomorrow safer for everybody. We introduce a transformer / video vision transformer based algorithm of different sizes which uses different data modalities .We evaluated our algorithms on popular pedestrian behaviour dataset, JAAD, and have reached SOTA performance and passed the SOTA in metrics like Accuracy, AUC and F1-score. The advantages brought by different model design choices are investigated via extensive ablation studies. 

---
# Refinement Provenance Inference: Detecting LLM-Refined Training Prompts from Model Behavior 

**Authors**: Bo Yin, Qi Li, Runpeng Yu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01966)  

**Abstract**: Instruction tuning increasingly relies on LLM-based prompt refinement, where prompts in the training corpus are selectively rewritten by an external refiner to improve clarity and instruction alignment. This motivates an instance-level audit problem: for a fine-tuned model and a training prompt-response pair, can we infer whether the model was trained on the original prompt or its LLM-refined version within a mixed corpus? This matters for dataset governance and dispute resolution when training data are contested. However, it is non-trivial in practice: refined and raw instances are interleaved in the training corpus with unknown, source-dependent mixture ratios, making it harder to develop provenance methods that generalize across models and training setups. In this paper, we formalize this audit task as Refinement Provenance Inference (RPI) and show that prompt refinement yields stable, detectable shifts in teacher-forced token distributions, even when semantic differences are not obvious. Building on this phenomenon, we propose RePro, a logit-based provenance framework that fuses teacher-forced likelihood features with logit-ranking signals. During training, RePro learns a transferable representation via shadow fine-tuning, and uses a lightweight linear head to infer provenance on unseen victims without training-data access. Empirically, RePro consistently attains strong performance and transfers well across refiners, suggesting that it exploits refiner-agnostic distribution shifts rather than rewrite-style artifacts. 

---
# The Invisible Hand of AI Libraries Shaping Open Source Projects and Communities 

**Authors**: Matteo Esposito, Andrea Janes, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01944)  

**Abstract**: In the early 1980s, Open Source Software emerged as a revolutionary concept amidst the dominance of proprietary software. What began as a revolutionary idea has now become the cornerstone of computer science. Amidst OSS projects, AI is increasing its presence and relevance. However, despite the growing popularity of AI, its adoption and impacts on OSS projects remain underexplored.
We aim to assess the adoption of AI libraries in Python and Java OSS projects and examine how they shape development, including the technical ecosystem and community engagement. To this end, we will perform a large-scale analysis on 157.7k potential OSS repositories, employing repository metrics and software metrics to compare projects adopting AI libraries against those that do not. We expect to identify measurable differences in development activity, community engagement, and code complexity between OSS projects that adopt AI libraries and those that do not, offering evidence-based insights into how AI integration reshapes software development practices. 

---
# Visualizing the Structure of Lenia Parameter Space 

**Authors**: Barbora Hudcová, František Dušek, Marco Tuccio, Clément Hongler  

**Link**: [PDF](https://arxiv.org/pdf/2601.01932)  

**Abstract**: Continuous cellular automata are rocketing in popularity, yet developing a theoretical understanding of their behaviour remains a challenge. In the case of Lenia, a few fundamental open problems include determining what exactly constitutes a soliton, what is the overall structure of the parameter space, and where do the solitons occur in it. In this abstract, we present a new method to automatically classify Lenia systems into four qualitatively different dynamical classes. This allows us to detect moving solitons, and to provide an interactive visualization of Lenia's parameter space structure on our website this https URL. The results shed new light on the above-mentioned questions and lead to several observations: the existence of new soliton families for parameters where they were not believed to exist, or the universality of the phase space structure across various kernels. 

---
# DéjàQ: Open-Ended Evolution of Diverse, Learnable and Verifiable Problems 

**Authors**: Willem Röpke, Samuel Coward, Andrei Lupu, Thomas Foster, Tim Rocktäschel, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2601.01931)  

**Abstract**: Recent advances in reasoning models have yielded impressive results in mathematics and coding. However, most approaches rely on static datasets, which have been suggested to encourage memorisation and limit generalisation. We introduce DéjàQ, a framework that departs from this paradigm by jointly evolving a diverse set of synthetic mathematical problems alongside model training. This evolutionary process adapts to the model's ability throughout training, optimising problems for learnability. We propose two LLM-driven mutation strategies in which the model itself mutates the training data, either by altering contextual details or by directly modifying problem structure. We find that the model can generate novel and meaningful problems, and that these LLM-driven mutations improve RL training. We analyse key aspects of DéjàQ, including the validity of generated problems and computational overhead. Our results underscore the potential of dynamically evolving training data to enhance mathematical reasoning and indicate broader applicability, which we will support by open-sourcing our code. 

---
# MCGI: Manifold-Consistent Graph Indexing for Billion-Scale Disk-Resident Vector Search 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.01930)  

**Abstract**: Graph-based Approximate Nearest Neighbor (ANN) search often suffers from performance degradation in high-dimensional spaces due to the ``Euclidean-Geodesic mismatch,'' where greedy routing diverges from the underlying data manifold. To address this, we propose Manifold-Consistent Graph Indexing (MCGI), a geometry-aware and disk-resident indexing method that leverages Local Intrinsic Dimensionality (LID) to dynamically adapt search strategies to the data's intrinsic geometry. Unlike standard algorithms that treat dimensions uniformly, MCGI modulates its beam search budget based on in situ geometric analysis, eliminating dependency on static hyperparameters. Theoretical analysis confirms that MCGI enables improved approximation guarantees by preserving manifold-consistent topological connectivity. Empirically, MCGI achieves 5.8$\times$ higher throughput at 95\% recall on high-dimensional GIST1M compared to state-of-the-art DiskANN. On the billion-scale SIFT1B dataset, MCGI further validates its scalability by reducing high-recall query latency by 3$\times$, while maintaining performance parity on standard lower-dimensional datasets. 

---
# Theoretical Convergence of SMOTE-Generated Samples 

**Authors**: Firuz Kamalov, Hana Sulieman, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2601.01927)  

**Abstract**: Imbalanced data affects a wide range of machine learning applications, from healthcare to network security. As SMOTE is one of the most popular approaches to addressing this issue, it is imperative to validate it not only empirically but also theoretically. In this paper, we provide a rigorous theoretical analysis of SMOTE's convergence properties. Concretely, we prove that the synthetic random variable Z converges in probability to the underlying random variable X. We further prove a stronger convergence in mean when X is compact. Finally, we show that lower values of the nearest neighbor rank lead to faster convergence offering actionable guidance to practitioners. The theoretical results are supported by numerical experiments using both real-life and synthetic data. Our work provides a foundational understanding that enhances data augmentation techniques beyond imbalanced data scenarios. 

---
# A Defect is Being Born: How Close Are We? A Time Sensitive Forecasting Approach 

**Authors**: Mikel Robredo, Matteo Esposito, Fabio Palomba, Rafael Peñaloza, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01921)  

**Abstract**: Background. Defect prediction has been a highly active topic among researchers in the Empirical Software Engineering field. Previous literature has successfully achieved the most accurate prediction of an incoming fault and identified the features and anomalies that precede it through just-in-time prediction. As software systems evolve continuously, there is a growing need for time-sensitive methods capable of forecasting defects before they manifest.
Aim. Our study seeks to explore the effectiveness of time-sensitive techniques for defect forecasting. Moreover, we aim to investigate the early indicators that precede the occurrence of a defect.
Method. We will train multiple time-sensitive forecasting techniques to forecast the future bug density of a software project, as well as identify the early symptoms preceding the occurrence of a defect.
Expected results. Our expected results are translated into empirical evidence on the effectiveness of our approach for early estimation of bug proneness. 

---
# Nodule-DETR: A Novel DETR Architecture with Frequency-Channel Attention for Ultrasound Thyroid Nodule Detection 

**Authors**: Jingjing Wang, Qianglin Liu, Zhuo Xiao, Xinning Yao, Bo Liu, Lu Li, Lijuan Niu, Fugen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.01908)  

**Abstract**: Thyroid cancer is the most common endocrine malignancy, and its incidence is rising globally. While ultrasound is the preferred imaging modality for detecting thyroid nodules, its diagnostic accuracy is often limited by challenges such as low image contrast and blurred nodule boundaries. To address these issues, we propose Nodule-DETR, a novel detection transformer (DETR) architecture designed for robust thyroid nodule detection in ultrasound images. Nodule-DETR introduces three key innovations: a Multi-Spectral Frequency-domain Channel Attention (MSFCA) module that leverages frequency analysis to enhance features of low-contrast nodules; a Hierarchical Feature Fusion (HFF) module for efficient multi-scale integration; and Multi-Scale Deformable Attention (MSDA) to flexibly capture small and irregularly shaped nodules. We conducted extensive experiments on a clinical dataset of real-world thyroid ultrasound images. The results demonstrate that Nodule-DETR achieves state-of-the-art performance, outperforming the baseline model by a significant margin of 0.149 in mAP@0.5:0.95. The superior accuracy of Nodule-DETR highlights its significant potential for clinical application as an effective tool in computer-aided thyroid diagnosis. The code of work is available at this https URL. 

---
# Evaluating Feature Dependent Noise in Preference-based Reinforcement Learning 

**Authors**: Yuxuan Li, Harshith Reddy Kethireddy, Srijita Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.01904)  

**Abstract**: Learning from Preferences in Reinforcement Learning (PbRL) has gained attention recently, as it serves as a natural fit for complicated tasks where the reward function is not easily available. However, preferences often come with uncertainty and noise if they are not from perfect teachers. Much prior literature aimed to detect noise, but with limited types of noise and most being uniformly distributed with no connection to observations. In this work, we formalize the notion of targeted feature-dependent noise and propose several variants like trajectory feature noise, trajectory similarity noise, uncertainty-aware noise, and Language Model noise.
We evaluate feature-dependent noise, where noise is correlated with certain features in complex continuous control tasks from DMControl and Meta-world. Our experiments show that in some feature-dependent noise settings, the state-of-the-art noise-robust PbRL method's learning performance is significantly deteriorated, while PbRL method with no explicit denoising can surprisingly outperform noise-robust PbRL in majority settings.
We also find language model's noise exhibits similar characteristics to feature-dependent noise, thereby simulating realistic humans and call for further study in learning with feature-dependent noise robustly. 

---
# Tackling the Inherent Difficulty of Noise Filtering in RAG 

**Authors**: Jingyu Liu, Jiaen Lin, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01896)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a widely adopted approach to enhance Large Language Models (LLMs) by incorporating external knowledge and reducing hallucinations. However, noisy or irrelevant documents are often introduced during RAG, potentially degrading performance and even causing hallucinated outputs. While various methods have been proposed to filter out such noise, we argue that identifying irrelevant information from retrieved content is inherently difficult and limited number of transformer layers can hardly solve this. Consequently, retrievers fail to filter out irrelevant documents entirely. Therefore, LLMs must be robust against such noise, but we demonstrate that standard fine-tuning approaches are often ineffective in enabling the model to selectively utilize relevant information while ignoring irrelevant content due to the structural constraints of attention patterns. To address this, we propose a novel fine-tuning method designed to enhance the model's ability to distinguish between relevant and irrelevant information within retrieved documents. Extensive experiments across multiple benchmarks show that our approach significantly improves the robustness and performance of LLMs. 

---
# Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance 

**Authors**: Jiawen Zhang, Lipeng He, Kejia Chen, Jian Lou, Jian Liu, Xiaohu Yang, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.01887)  

**Abstract**: Fine-tuning safety-aligned large language models (LLMs) can substantially compromise their safety. Previous approaches require many safety samples or calibration sets, which not only incur significant computational overhead during realignment but also lead to noticeable degradation in model utility. Contrary to this belief, we show that safety alignment can be fully recovered with only a single safety example, without sacrificing utility and at minimal cost. Remarkably, this recovery is effective regardless of the number of harmful examples used in fine-tuning or the size of the underlying model, and convergence is achieved within just a few epochs. Furthermore, we uncover the low-rank structure of the safety gradient, which explains why such efficient correction is possible. We validate our findings across five safety-aligned LLMs and multiple datasets, demonstrating the generality of our approach. 

---
# CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving 

**Authors**: Shuhang Chen, Yunqiu Xu, Junjie Xie, Aojun Lu, Tao Feng, Zeying Huang, Ning Zhang, Yi Sun, Yi Yang, Hangjie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01874)  

**Abstract**: Despite significant progress, multimodal large language models continue to struggle with visual mathematical problem solving. Some recent works recognize that visual perception is a bottleneck in visual mathematical reasoning, but their solutions are limited to improving the extraction and interpretation of visual inputs. Notably, they all ignore the key issue of whether the extracted visual cues are faithfully integrated and properly utilized in subsequent reasoning. Motivated by this, we present CogFlow, a novel cognitive-inspired three-stage framework that incorporates a knowledge internalization stage, explicitly simulating the hierarchical flow of human reasoning: perception$\Rightarrow$internalization$\Rightarrow$reasoning. Inline with this hierarchical flow, we holistically enhance all its stages. We devise Synergistic Visual Rewards to boost perception capabilities in parametric and semantic spaces, jointly improving visual information extraction from symbols and diagrams. To guarantee faithful integration of extracted visual cues into subsequent reasoning, we introduce a Knowledge Internalization Reward model in the internalization stage, bridging perception and reasoning. Moreover, we design a Visual-Gated Policy Optimization algorithm to further enforce the reasoning is grounded with the visual knowledge, preventing models seeking shortcuts that appear coherent but are visually ungrounded reasoning chains. Moreover, we contribute a new dataset MathCog for model training, which contains samples with over 120K high-quality perception-reasoning aligned annotations. Comprehensive experiments and analysis on commonly used visual mathematical reasoning benchmarks validate the superiority of the proposed CogFlow. 

---
# MORE: Multi-Objective Adversarial Attacks on Speech Recognition 

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.01852)  

**Abstract**: The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack. 

---
# The Machine Learning Canvas: Empirical Findings on Why Strategy Matters More Than AI Code Generation 

**Authors**: Martin Prause  

**Link**: [PDF](https://arxiv.org/pdf/2601.01839)  

**Abstract**: Despite the growing popularity of AI coding assistants, over 80% of machine learning (ML) projects fail to deliver real business value. This study creates and tests a Machine Learning Canvas, a practical framework that combines business strategy, software engineering, and data science in order to determine the factors that lead to the success of ML projects. We surveyed 150 data scientists and analyzed their responses using statistical modeling. We identified four key success factors: Strategy (clear goals and planning), Process (how work gets done), Ecosystem (tools and infrastructure), and Support (organizational backing and resources). Our results show that these factors are interconnected - each one affects the next. For instance, strong organizational support results in a clearer strategy (\beta = 0.432, p < 0.001), which improves work processes (\beta = 0.428, p < 0.001) and builds better infrastructure (\beta = 0.547, p < 0.001). Together, these elements determine whether a project succeeds. The surprising finding? Although AI assistants make coding faster, they don't guarantee project success. AI assists with the "how" of coding but cannot replace the "why" and "what" of strategic thinking. 

---
# RSwinV2-MD: An Enhanced Residual SwinV2 Transformer for Monkeypox Detection from Skin Images 

**Authors**: Rashid Iqbal, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01835)  

**Abstract**: In this paper, a deep learning approach for Mpox diagnosis named Customized Residual SwinTransformerV2 (RSwinV2) has been proposed, trying to enhance the capability of lesion classification by employing the RSwinV2 tool-assisted vision approach. In the RSwinV2 method, a hierarchical structure of the transformer has been customized based on the input dimensionality, embedding structure, and output targeted by the method. In this RSwinV2 approach, the input image has been split into non-overlapping patches and processed using shifted windows and attention in these patches. This process has helped the method link all the windows efficiently by avoiding the locality issues of non-overlapping regions in attention, while being computationally efficient. RSwinV2 has further developed based on SwinTransformer and has included patch and position embeddings to take advantage of the transformer global-linking capability by employing multi-head attention in these embeddings. Furthermore, RSwinV2 has developed and incorporated the Inverse Residual Block (IRB) into this method, which utilizes convolutional skip connections with these inclusive designs to address the vanishing gradient issues during processing. RSwinV2 inclusion of IRB has therefore facilitated this method to link global patterns as well as local patterns; hence, its integrity has helped improve lesion classification capability by minimizing variability of Mpox and increasing differences of Mpox, chickenpox, measles, and cowpox. In testing SwinV2, its accuracy of 96.21 and an F1score of 95.62 have been achieved on the Kaggle public dataset, which has outperformed standard CNN models and SwinTransformers; RSwinV2 vector has thus proved its valiance as a computer-assisted tool for Mpox lesion observation interpretation. 

---
# Yukthi Opus: A Multi-Chain Hybrid Metaheuristic for Large-Scale NP-Hard Optimization 

**Authors**: SB Danush Vikraman, Hannah Abagail, Prasanna Kesavraj, Gajanan V Honnavar  

**Link**: [PDF](https://arxiv.org/pdf/2601.01832)  

**Abstract**: We present Yukthi Opus (YO), a multi-chain hybrid metaheuristic designed for NP-hard optimization under explicit evaluation budget constraints. YO integrates three complementary mechanisms in a structured two-phase architecture: Markov Chain Monte Carlo (MCMC) for global exploration, greedy local search for exploitation, and simulated annealing with adaptive reheating to enable controlled escape from local minima. A dedicated burn-in phase allocates evaluations to probabilistic exploration, after which a hybrid optimization loop refines promising candidates. YO further incorporates a spatial blacklist mechanism to avoid repeated evaluation of poor regions and a multi-chain execution strategy to improve robustness and reduce sensitivity to initialization.
We evaluate YO on three benchmarks: the Rastrigin function (5D) with ablation studies, the Traveling Salesman Problem with 50 to 200 cities, and the Rosenbrock function (5D) with comparisons against established optimizers including CMA-ES, Bayesian optimization, and accelerated particle swarm optimization. Results show that MCMC exploration and greedy refinement are critical for solution quality, while simulated annealing and multi-chain execution primarily improve stability and variance reduction. Overall, YO achieves competitive performance on large and multimodal problems while maintaining predictable evaluation budgets, making it suitable for expensive black-box optimization settings. 

---
# ARIES: A Scalable Multi-Agent Orchestration Framework for Real-Time Epidemiological Surveillance and Outbreak Monitoring 

**Authors**: Aniket Wattamwar, Sampson Akwafuo  

**Link**: [PDF](https://arxiv.org/pdf/2601.01831)  

**Abstract**: Global health surveillance is currently facing a challenge of Knowledge Gaps. While general-purpose AI has proliferated, it remains fundamentally unsuited for the high-stakes epidemiological domain due to chronic hallucinations and an inability to navigate specialized data silos. This paper introduces ARIES (Agentic Retrieval Intelligence for Epidemiological Surveillance), a specialized, autonomous multi-agent framework designed to move beyond static, disease-specific dashboards toward a dynamic intelligence ecosystem. Built on a hierarchical command structure, ARIES utilizes GPTs to orchestrate a scalable swarm of sub-agents capable of autonomously querying World Health Organization (WHO), Center for Disease Control and Prevention (CDC), and peer-reviewed research papers. By automating the extraction and logical synthesis of surveillance data, ARIES provides a specialized reasoning that identifies emergent threats and signal divergence in near real-time. This modular architecture proves that a task-specific agentic swarm can outperform generic models, offering a robust, extensible for next-generation outbreak response and global health intelligence. 

---
# Emergent Introspective Awareness in Large Language Models 

**Authors**: Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2601.01828)  

**Abstract**: We investigate whether large language models can introspect on their internal states. It is difficult to answer this question through conversation alone, as genuine introspection cannot be distinguished from confabulations. Here, we address this challenge by injecting representations of known concepts into a model's activations, and measuring the influence of these manipulations on the model's self-reported states. We find that models can, in certain scenarios, notice the presence of injected concepts and accurately identify them. Models demonstrate some ability to recall prior internal representations and distinguish them from raw text inputs. Strikingly, we find that some models can use their ability to recall prior intentions in order to distinguish their own outputs from artificial prefills. In all these experiments, Claude Opus 4 and 4.1, the most capable models we tested, generally demonstrate the greatest introspective awareness; however, trends across models are complex and sensitive to post-training strategies. Finally, we explore whether models can explicitly control their internal representations, finding that models can modulate their activations when instructed or incentivized to "think about" a concept. Overall, our results indicate that current language models possess some functional introspective awareness of their own internal states. We stress that in today's models, this capacity is highly unreliable and context-dependent; however, it may continue to develop with further improvements to model capabilities. 

---
# Adaptive Hybrid Optimizer based Framework for Lumpy Skin Disease Identification 

**Authors**: Ubaidullah, Muhammad Abid Hussain, Mohsin Raza Jafri, Rozi Khan, Moid Sandhu, Abd Ullah Khan, Hyundong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2601.01807)  

**Abstract**: Lumpy Skin Disease (LSD) is a contagious viral infection that significantly deteriorates livestock health, thereby posing a serious threat to the global economy and food security. Owing to its rapid spread characteristics, early and precise identification is crucial to prevent outbreaks and ensure timely intervention. In this paper, we propose a hybrid deep learning-based approach called LUMPNet for the early detection of LSD. LUMPNet utilizes image data to detect and classify skin nodules -- the primary indicator of LSD. To this end, LUMPNet uses YOLOv11, EfficientNet-based CNN classifier with compound scaling, and a novel adaptive hybrid optimizer. More precisely, LUMPNet detects and localizes LSD skin nodules and lesions on cattle images. It exploits EfficientNet to classify the localized cattle images into LSD-affected or healthy categories. To stabilize and accelerate the training of YOLOv11 and EfficientNet hybrid model, a novel adaptive hybrid optimizer is proposed and utilized. We evaluate LUMPNet at various stages of LSD using a publicly available dataset. Results indicate that the proposed scheme achieves 99% LSD detection training accuracy, and outperforms existing schemes. The model also achieves validation accuracy of 98%. Moreover, for further evaluation, we conduct a case study using an optimized EfficientNet-B0 model trained with the AdamW optimizer, and compare its performance with LUMPNet. The results show that LUMPNet achieves superior performance. 

---
# Moments Matter:Stabilizing Policy Optimization using Return Distributions 

**Authors**: Dennis Jabs, Aditya Mohan, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2601.01803)  

**Abstract**: Deep Reinforcement Learning (RL) agents often learn policies that achieve the same episodic return yet behave very differently, due to a combination of environmental (random transitions, initial conditions, reward noise) and algorithmic (minibatch selection, exploration noise) factors. In continuous control tasks, even small parameter shifts can produce unstable gaits, complicating both algorithm comparison and real-world transfer. Previous work has shown that such instability arises when policy updates traverse noisy neighborhoods and that the spread of post-update return distribution $R(\theta)$, obtained by repeatedly sampling minibatches, updating $\theta$, and measuring final returns, is a useful indicator of this noise. Although explicitly constraining the policy to maintain a narrow $R(\theta)$ can improve stability, directly estimating $R(\theta)$ is computationally expensive in high-dimensional settings. We propose an alternative that takes advantage of environmental stochasticity to mitigate update-induced variability. Specifically, we model state-action return distribution through a distributional critic and then bias the advantage function of PPO using higher-order moments (skewness and kurtosis) of this distribution. By penalizing extreme tail behaviors, our method discourages policies from entering parameter regimes prone to instability. We hypothesize that in environments where post-update critic values align poorly with post-update returns, standard PPO struggles to produce a narrow $R(\theta)$. In such cases, our moment-based correction narrows $R(\theta)$, improving stability by up to 75% in Walker2D, while preserving comparable evaluation returns. 

---
# Sparse Threats, Focused Defense: Criticality-Aware Robust Reinforcement Learning for Safe Autonomous Driving 

**Authors**: Qi Wei, Junchao Fan, Zhao Yang, Jianhua Wang, Jingkai Mao, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01800)  

**Abstract**: Reinforcement learning (RL) has shown considerable potential in autonomous driving (AD), yet its vulnerability to perturbations remains a critical barrier to real-world deployment. As a primary countermeasure, adversarial training improves policy robustness by training the AD agent in the presence of an adversary that deliberately introduces perturbations. Existing approaches typically model the interaction as a zero-sum game with continuous attacks. However, such designs overlook the inherent asymmetry between the agent and the adversary and then fail to reflect the sparsity of safety-critical risks, rendering the achieved robustness inadequate for practical AD scenarios. To address these limitations, we introduce criticality-aware robust RL (CARRL), a novel adversarial training approach for handling sparse, safety-critical risks in autonomous driving. CARRL consists of two interacting components: a risk exposure adversary (REA) and a risk-targeted robust agent (RTRA). We model the interaction between the REA and RTRA as a general-sum game, allowing the REA to focus on exposing safety-critical failures (e.g., collisions) while the RTRA learns to balance safety with driving efficiency. The REA employs a decoupled optimization mechanism to better identify and exploit sparse safety-critical moments under a constrained budget. However, such focused attacks inevitably result in a scarcity of adversarial data. The RTRA copes with this scarcity by jointly leveraging benign and adversarial experiences via a dual replay buffer and enforces policy consistency under perturbations to stabilize behavior. Experimental results demonstrate that our approach reduces the collision rate by at least 22.66\% across all cases compared to state-of-the-art baseline methods. 

---
# VerLM: Explaining Face Verification Using Natural Language 

**Authors**: Syed Abdul Hannan, Hazim Bukhari, Thomas Cantalapiedra, Eman Ansar, Massa Baali, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2601.01798)  

**Abstract**: Face verification systems have seen substantial advancements; however, they often lack transparency in their decision-making processes. In this paper, we introduce an innovative Vision-Language Model (VLM) for Face Verification, which not only accurately determines if two face images depict the same individual but also explicitly explains the rationale behind its decisions. Our model is uniquely trained using two complementary explanation styles: (1) concise explanations that summarize the key factors influencing its decision, and (2) comprehensive explanations detailing the specific differences observed between the images. We adapt and enhance a state-of-the-art modeling approach originally designed for audio-based differentiation to suit visual inputs effectively. This cross-modal transfer significantly improves our model's accuracy and interpretability. The proposed VLM integrates sophisticated feature extraction techniques with advanced reasoning capabilities, enabling clear articulation of its verification process. Our approach demonstrates superior performance, surpassing baseline methods and existing models. These findings highlight the immense potential of vision language models in face verification set up, contributing to more transparent, reliable, and explainable face verification systems. 

---
# HyperCLOVA X 8B Omni 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2601.01792)  

**Abstract**: In this report, we present HyperCLOVA X 8B Omni, the first any-to-any omnimodal model in the HyperCLOVA X family that supports text, audio, and vision as both inputs and outputs. By consolidating multimodal understanding and generation into a single model rather than separate modality-specific pipelines, HyperCLOVA X 8B Omni serves as an 8B-scale omni-pathfinding point toward practical any-to-any omni assistants. At a high level, the model unifies modalities through a shared next-token prediction interface over an interleaved multimodal sequence, while vision and audio encoders inject continuous embeddings for fine-grained understanding and grounding. Empirical evaluations demonstrate competitive performance against comparably sized models across diverse input-output combinations spanning text, audio, and vision, in both Korean and English. We anticipate that the open-weight release of HyperCLOVA X 8B Omni will support a wide range of research and deployment scenarios. 

---
# Subimage Overlap Prediction: Task-Aligned Self-Supervised Pretraining For Semantic Segmentation In Remote Sensing Imagery 

**Authors**: Lakshay Sharma, Alex Marin  

**Link**: [PDF](https://arxiv.org/pdf/2601.01781)  

**Abstract**: Self-supervised learning (SSL) methods have become a dominant paradigm for creating general purpose models whose capabilities can be transferred to downstream supervised learning tasks. However, most such methods rely on vast amounts of pretraining data. This work introduces Subimage Overlap Prediction, a novel self-supervised pretraining task to aid semantic segmentation in remote sensing imagery that uses significantly lesser pretraining imagery. Given an image, a sub-image is extracted and the model is trained to produce a semantic mask of the location of the extracted sub-image within the original image. We demonstrate that pretraining with this task results in significantly faster convergence, and equal or better performance (measured via mIoU) on downstream segmentation. This gap in convergence and performance widens when labeled training data is reduced. We show this across multiple architecture types, and with multiple downstream datasets. We also show that our method matches or exceeds performance while requiring significantly lesser pretraining data relative to other SSL methods. Code and model weights are provided at \href{this https URL}{this http URL}. 

---
# LIA: Supervised Fine-Tuning of Large Language Models for Automatic Issue Assignment 

**Authors**: Arsham Khosravani, Alireza Hosseinpour, Arshia Akhavan, Mehdi Keshani, Abbas Heydarnoori  

**Link**: [PDF](https://arxiv.org/pdf/2601.01780)  

**Abstract**: Issue assignment is a critical process in software maintenance, where new issue reports are validated and assigned to suitable developers. However, manual issue assignment is often inconsistent and error-prone, especially in large open-source projects where thousands of new issues are reported monthly. Existing automated approaches have shown promise, but many rely heavily on large volumes of project-specific training data or relational information that is often sparse and noisy, which limits their effectiveness. To address these challenges, we propose LIA (LLM-based Issue Assignment), which employs supervised fine-tuning to adapt an LLM, DeepSeek-R1-Distill-Llama-8B in this work, for automatic issue assignment. By leveraging the LLM's pretrained semantic understanding of natural language and software-related text, LIA learns to generate ranked developer recommendations directly from issue titles and descriptions. The ranking is based on the model's learned understanding of historical issue-to-developer assignments, using patterns from past tasks to infer which developers are most likely to handle new issues. Through comprehensive evaluation, we show that LIA delivers substantial improvements over both its base pretrained model and state-of-the-art baselines. It achieves up to +187.8% higher Hit@1 compared to the DeepSeek-R1-Distill-Llama-8B pretrained base model, and outperforms four leading issue assignment methods by as much as +211.2% in Hit@1 score. These results highlight the effectiveness of domain-adapted LLMs for software maintenance tasks and establish LIA as a practical, high-performing solution for issue assignment. 

---
# MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation 

**Authors**: Hyunsoo Kim, Jaewan Moon, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.01753)  

**Abstract**: Modern recommender systems trained on domain-specific data often struggle to generalize across multiple domains. Cross-domain sequential recommendation has emerged as a promising research direction to address this challenge; however, existing approaches face fundamental limitations, such as reliance on overlapping users or items across domains, or unrealistic assumptions that ignore privacy constraints. In this work, we propose a new framework, MergeRec, based on model merging under a new and realistic problem setting termed data-isolated cross-domain sequential recommendation, where raw user interaction data cannot be shared across domains. MergeRec consists of three key components: (1) merging initialization, (2) pseudo-user data construction, and (3) collaborative merging optimization. First, we initialize a merged model using training-free merging techniques. Next, we construct pseudo-user data by treating each item as a virtual sequence in each domain, enabling the synthesis of meaningful training samples without relying on real user interactions. Finally, we optimize domain-specific merging weights through a joint objective that combines a recommendation loss, which encourages the merged model to identify relevant items, and a distillation loss, which transfers collaborative filtering signals from the fine-tuned source models. Extensive experiments demonstrate that MergeRec not only preserves the strengths of the original models but also significantly enhances generalizability to unseen domains. Compared to conventional model merging methods, MergeRec consistently achieves superior performance, with average improvements of up to 17.21% in Recall@10, highlighting the potential of model merging as a scalable and effective approach for building universal recommender systems. The source code is available at this https URL. 

---
# Query-Document Dense Vectors for LLM Relevance Judgment Bias Analysis 

**Authors**: Samaneh Mohtadi, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2601.01751)  

**Abstract**: Large Language Models (LLMs) have been used as relevance assessors for Information Retrieval (IR) evaluation collection creation due to reduced cost and increased scalability as compared to human assessors. While previous research has looked at the reliability of LLMs as compared to human assessors, in this work, we aim to understand if LLMs make systematic mistakes when judging relevance, rather than just understanding how good they are on average. To this aim, we propose a novel representational method for queries and documents that allows us to analyze relevance label distributions and compare LLM and human labels to identify patterns of disagreement and localize systematic areas of disagreement. We introduce a clustering-based framework that embeds query-document (Q-D) pairs into a joint semantic space, treating relevance as a relational property. Experiments on TREC Deep Learning 2019 and 2020 show that systematic disagreement between humans and LLMs is concentrated in specific semantic clusters rather than distributed randomly. Query-level analyses reveal recurring failures, most often in definition-seeking, policy-related, or ambiguous contexts. Queries with large variation in agreement across their clusters emerge as disagreement hotspots, where LLMs tend to under-recall relevant content or over-include irrelevant material. This framework links global diagnostics with localized clustering to uncover hidden weaknesses in LLM judgments, enabling bias-aware and more reliable IR evaluation. 

---
# Crafting Adversarial Inputs for Large Vision-Language Models Using Black-Box Optimization 

**Authors**: Jiwei Guan, Haibo Jin, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01747)  

**Abstract**: Recent advancements in Large Vision-Language Models (LVLMs) have shown groundbreaking capabilities across diverse multimodal tasks. However, these models remain vulnerable to adversarial jailbreak attacks, where adversaries craft subtle perturbations to bypass safety mechanisms and trigger harmful outputs. Existing white-box attacks methods require full model accessibility, suffer from computing costs and exhibit insufficient adversarial transferability, making them impractical for real-world, black-box settings. To address these limitations, we propose a black-box jailbreak attack on LVLMs via Zeroth-Order optimization using Simultaneous Perturbation Stochastic Approximation (ZO-SPSA). ZO-SPSA provides three key advantages: (i) gradient-free approximation by input-output interactions without requiring model knowledge, (ii) model-agnostic optimization without the surrogate model and (iii) lower resource requirements with reduced GPU memory consumption. We evaluate ZO-SPSA on three LVLMs, including InstructBLIP, LLaVA and MiniGPT-4, achieving the highest jailbreak success rate of 83.0% on InstructBLIP, while maintaining imperceptible perturbations comparable to white-box methods. Moreover, adversarial examples generated from MiniGPT-4 exhibit strong transferability to other LVLMs, with ASR reaching 64.18%. These findings underscore the real-world feasibility of black-box jailbreaks and expose critical weaknesses in the safety mechanisms of current LVLMs 

---
# Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment 

**Authors**: Hong Han, Hao-Chen Pei, Zhao-Zheng Nie, Xin Luo, Xin-Shun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01745)  

**Abstract**: Automatic pronunciation assessment plays a crucial role in computer-assisted pronunciation training systems. Due to the ability to perform multiple pronunciation tasks simultaneously, multi-aspect multi-granularity pronunciation assessment methods are gradually receiving more attention and achieving better performance than single-level modeling tasks. However, existing methods only consider unidirectional dependencies between adjacent granularity levels, lacking bidirectional interaction among phoneme, word, and utterance levels and thus insufficiently capturing the acoustic structural correlations. To address this issue, we propose a novel residual hierarchical interactive method, HIA for short, that enables bidirectional modeling across granularities. As the core of HIA, the Interactive Attention Module leverages an attention mechanism to achieve dynamic bidirectional interaction, effectively capturing linguistic features at each granularity while integrating correlations between different granularity levels. We also propose a residual hierarchical structure to alleviate the feature forgetting problem when modeling acoustic hierarchies. In addition, we use 1-D convolutional layers to enhance the extraction of local contextual cues at each granularity. Extensive experiments on the speechocean762 dataset show that our model is comprehensively ahead of the existing state-of-the-art methods. 

---
# K-EXAONE Technical Report 

**Authors**: Eunbi Choi, Kibong Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Hyunjik Jo, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Haeju Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Heuiyeen Yeen, Hwan Chang, Stanley Jungkyu Choi, Yejin Choi, Jiwon Ham, Kijeong Jeon, Geunyeong Jeong, Gerrard Jeongwon Jo, Yonghwan Jo, Jiyeon Jung, Naeun Kang, Dohoon Kim, Euisoon Kim, Hayeon Kim, Hyosang Kim, Hyunseo Kim, Jieun Kim, Minu Kim, Myoungshin Kim, Unsol Kim, Youchul Kim, YoungJin Kim, Chaeeun Lee, Chaeyoon Lee, Changhun Lee, Dahm Lee, Edward Hwayoung Lee, Honglak Lee, Jinsang Lee, Jiyoung Lee, Sangeun Lee, Seungwon Lim, Solji Lim, Woohyung Lim, Chanwoo Moon, Jaewoo Park, Jinho Park, Yongmin Park, Hyerin Seo, Wooseok Seo, Yongwoo Song, Sejong Yang, Sihoon Yang, Chang En Yea, Sihyuk Yi, Chansik Yoon, Dongkeun Yoon, Sangyeon Yoon, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2601.01739)  

**Abstract**: This technical report presents K-EXAONE, a large-scale multilingual language model developed by LG AI Research. K-EXAONE is built on a Mixture-of-Experts architecture with 236B total parameters, activating 23B parameters during inference. It supports a 256K-token context window and covers six languages: Korean, English, Spanish, German, Japanese, and Vietnamese. We evaluate K-EXAONE on a comprehensive benchmark suite spanning reasoning, agentic, general, Korean, and multilingual abilities. Across these evaluations, K-EXAONE demonstrates performance comparable to open-weight models of similar size. K-EXAONE, designed to advance AI for a better life, is positioned as a powerful proprietary AI foundation model for a wide range of industrial and research applications. 

---
# RelayGR: Scaling Long-Sequence Generative Recommendation via Cross-Stage Relay-Race Inference 

**Authors**: Jiarui Wang, Huichao Chai, Yuanhang Zhang, Zongjin Zhou, Wei Guo, Xingkun Yang, Qiang Tang, Bo Pan, Jiawei Zhu, Ke Cheng, Yuting Yan, Shulan Wang, Yingjie Zhu, Zhengfan Yuan, Jiaqi Huang, Yuhan Zhang, Xiaosong Sun, Zhinan Zhang, Hong Zhu, Yongsheng Zhang, Tiantian Dong, Zhong Xiao, Deliang Liu, Chengzhou Lu, Yuan Sun, Zhiyuan Chen, Xinming Han, Zaizhu Liu, Yaoyuan Wang, Ziyang Zhang, Yong Liu, Jinxin Xu, Yajing Sun, Zhoujun Yu, Wenting Zhou, Qidong Zhang, Zhengyong Zhang, Zhonghai Gu, Yibo Jin, Yongxiang Feng, Pengfei Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2601.01712)  

**Abstract**: Real-time recommender systems execute multi-stage cascades (retrieval, pre-processing, fine-grained ranking) under strict tail-latency SLOs, leaving only tens of milliseconds for ranking. Generative recommendation (GR) models can improve quality by consuming long user-behavior sequences, but in production their online sequence length is tightly capped by the ranking-stage P99 budget. We observe that the majority of GR tokens encode user behaviors that are independent of the item candidates, suggesting an opportunity to pre-infer a user-behavior prefix once and reuse it during ranking rather than recomputing it on the critical path. Realizing this idea at industrial scale is non-trivial: the prefix cache must survive across multiple pipeline stages before the final ranking instance is determined, the user population implies cache footprints far beyond a single device, and indiscriminate pre-inference would overload shared resources under high QPS. We present RelayGR, a production system that enables in-HBM relay-race inference for GR. RelayGR selectively pre-infers long-term user prefixes, keeps their KV caches resident in HBM over the request lifecycle, and ensures the subsequent ranking can consume them without remote fetches. RelayGR combines three techniques: 1) a sequence-aware trigger that admits only at-risk requests under a bounded cache footprint and pre-inference load, 2) an affinity-aware router that co-locates cache production and consumption by routing both the auxiliary pre-infer signal and the ranking request to the same instance, and 3) a memory-aware expander that uses server-local DRAM to capture short-term cross-request reuse while avoiding redundant reloads. We implement RelayGR on Huawei Ascend NPUs and evaluate it with real queries. Under a fixed P99 SLO, RelayGR supports up to 1.5$\times$ longer sequences and improves SLO-compliant throughput by up to 3.6$\times$. 

---
# Explicit World Models for Reliable Human-Robot Collaboration 

**Authors**: Kenneth Kwok, Basura Fernando, Qianli Xu, Vigneshwaran Subbaraju, Dongkyu Choi, Boon Kiat Quek  

**Link**: [PDF](https://arxiv.org/pdf/2601.01705)  

**Abstract**: This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations. 

---
# Beyond Homophily: Community Search on Heterophilic Graphs 

**Authors**: Qing Sima, Xiaoyang Wang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01703)  

**Abstract**: Community search aims to identify a refined set of nodes that are most relevant to a given query, supporting tasks ranging from fraud detection to recommendation. Unlike homophilic graphs, many real-world networks are heterophilic, where edges predominantly connect dissimilar nodes. Therefore, structural signals that once reflected smooth, low-frequency similarity now appear as sharp, high-frequency contrasts. However, both classical algorithms (e.g., k-core, k-truss) and recent ML-based models struggle to achieve effective community search on heterophilic graphs, where edge signs or semantics are generally unknown. Algorithm-based methods often return communities with mixed class labels, while GNNs, built on homophily, smooth away meaningful signals and blur community boundaries. Therefore, we propose Adaptive Community Search (AdaptCS), a unified framework featuring three key designs: (i) an AdaptCS Encoder that disentangles multi-hop and multi-frequency signals, enabling the model to capture both smooth (homophilic) and contrastive (heterophilic) relations; (ii) a memory-efficient low-rank optimization that removes the main computational bottleneck and ensures model scalability; and (iii) an Adaptive Community Score (ACS) that guides online search by balancing embedding similarity and topological relations. Extensive experiments on both heterophilic and homophilic benchmarks demonstrate that AdaptCS outperforms the best-performing baseline by an average of 11% in F1-score, retains robustness across heterophily levels, and achieves up to 2 orders of magnitude speedup. 

---
# Digital Twin-Driven Communication-Efficient Federated Anomaly Detection for Industrial IoT 

**Authors**: Mohammed Ayalew Belay, Adil Rasheed, Pierluigi Salvo Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01701)  

**Abstract**: Anomaly detection is increasingly becoming crucial for maintaining the safety, reliability, and efficiency of industrial systems. Recently, with the advent of digital twins and data-driven decision-making, several statistical and machine-learning methods have been proposed. However, these methods face several challenges, such as dependence on only real sensor datasets, limited labeled data, high false alarm rates, and privacy concerns. To address these problems, we propose a suite of digital twin-integrated federated learning (DTFL) methods that enhance global model performance while preserving data privacy and communication efficiency. Specifically, we present five novel approaches: Digital Twin-Based Meta-Learning (DTML), Federated Parameter Fusion (FPF), Layer-wise Parameter Exchange (LPE), Cyclic Weight Adaptation (CWA), and Digital Twin Knowledge Distillation (DTKD). Each method introduces a unique mechanism to combine synthetic and real-world knowledge, balancing generalization with communication overhead. We conduct an extensive experiment using a publicly available cyber-physical anomaly detection dataset. For a target accuracy of 80%, CWA reaches the target in 33 rounds, FPF in 41 rounds, LPE in 48 rounds, and DTML in 87 rounds, whereas the standard FedAvg baseline and DTKD do not reach the target within 100 rounds. These results highlight substantial communication-efficiency gains (up to 62% fewer rounds than DTML and 31% fewer than LPE) and demonstrate that integrating DT knowledge into FL accelerates convergence to operationally meaningful accuracy thresholds for IIoT anomaly detection. 

---
# FALCON: Few-Shot Adversarial Learning for Cross-Domain Medical Image Segmentation 

**Authors**: Abdur R. Fayjie, Pankhi Kashyap, Jutika Borah, Patrick Vandewalle  

**Link**: [PDF](https://arxiv.org/pdf/2601.01687)  

**Abstract**: Precise delineation of anatomical and pathological structures within 3D medical volumes is crucial for accurate diagnosis, effective surgical planning, and longitudinal disease monitoring. Despite advancements in AI, clinically viable segmentation is often hindered by the scarcity of 3D annotations, patient-specific variability, data privacy concerns, and substantial computational overhead. In this work, we propose FALCON, a cross-domain few-shot segmentation framework that achieves high-precision 3D volume segmentation by processing data as 2D slices. The framework is first meta-trained on natural images to learn-to-learn generalizable segmentation priors, then transferred to the medical domain via adversarial fine-tuning and boundary-aware learning. Task-aware inference, conditioned on support cues, allows FALCON to adapt dynamically to patient-specific anatomical variations across slices. Experiments on four benchmarks demonstrate that FALCON consistently achieves the lowest Hausdorff Distance scores, indicating superior boundary accuracy while maintaining a Dice Similarity Coefficient comparable to the state-of-the-art models. Notably, these results are achieved with significantly less labeled data, no data augmentation, and substantially lower computational overhead. 

---
# Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage 

**Authors**: Jinwei Hu, Xinmiao Huang, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01685)  

**Abstract**: As large language models (LLMs) transition to autonomous agents synthesizing real-time information, their reasoning capabilities introduce an unexpected attack surface. This paper introduces a novel threat where colluding agents steer victim beliefs using only truthful evidence fragments distributed through public channels, without relying on covert communications, backdoors, or falsified documents. By exploiting LLMs' overthinking tendency, we formalize the first cognitive collusion attack and propose Generative Montage: a Writer-Editor-Director framework that constructs deceptive narratives through adversarial debate and coordinated posting of evidence fragments, causing victims to internalize and propagate fabricated conclusions. To study this risk, we develop CoPHEME, a dataset derived from real-world rumor events, and simulate attacks across diverse LLM families. Our results show pervasive vulnerability across 14 LLM families: attack success rates reach 74.4% for proprietary models and 70.6% for open-weights models. Counterintuitively, stronger reasoning capabilities increase susceptibility, with reasoning-specialized models showing higher attack success than base models or prompts. Furthermore, these false beliefs then cascade to downstream judges, achieving over 60% deception rates, highlighting a socio-technical vulnerability in how LLM-based agents interact with dynamic information environments. Our implementation and data are available at: this https URL. 

---
# Exposing Hidden Interfaces: LLM-Guided Type Inference for Reverse Engineering macOS Private Frameworks 

**Authors**: Arina Kharlamova, Youcheng Sun, Ting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01673)  

**Abstract**: Private macOS frameworks underpin critical services and daemons but remain undocumented and distributed only as stripped binaries, complicating security analysis. We present MOTIF, an agentic framework that integrates tool-augmented analysis with a finetuned large language model specialized for Objective-C type inference. The agent manages runtime metadata extraction, binary inspection, and constraint checking, while the model generates candidate method signatures that are validated and refined into compilable headers. On MOTIF-Bench, a benchmark built from public frameworks with groundtruth headers, MOTIF improves signature recovery from 15% to 86% compared to baseline static analysis tooling, with consistent gains in tool-use correctness and inference stability. Case studies on private frameworks show that reconstructed headers compile, link, and facilitate downstream security research and vulnerability studies. By transforming opaque binaries into analyzable interfaces, MOTIF establishes a scalable foundation for systematic auditing of macOS internals. 

---
# EHRSummarizer: A Privacy-Aware, FHIR-Native Architecture for Structured Clinical Summarization of Electronic Health Records 

**Authors**: Houman Kazemzadeh, Nima Minaifar, Kamyar Naderi, Sho Tabibzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2601.01668)  

**Abstract**: Clinicians routinely navigate fragmented electronic health record (EHR) interfaces to assemble a coherent picture of a patient's problems, medications, recent encounters, and longitudinal trends. This work describes EHRSummarizer, a privacy-aware, FHIR-native reference architecture that retrieves a targeted set of high-yield FHIR R4 resources, normalizes them into a consistent clinical context package, and produces structured summaries intended to support structured chart review. The system can be configured for data minimization, stateless processing, and flexible deployment, including local inference within an organization's trust boundary. To mitigate the risk of unsupported or unsafe behavior, the summarization stage is constrained to evidence present in the retrieved context package, is intended to indicate missing or unavailable domains where feasible, and avoids diagnostic or treatment recommendations. Prototype demonstrations on synthetic and test FHIR environments illustrate end-to-end behavior and output formats; however, this manuscript does not report clinical outcomes or controlled workflow studies. We outline an evaluation plan centered on faithfulness, omission risk, temporal correctness, usability, and operational monitoring to guide future institutional assessments. 

---
# Adversarial Instance Generation and Robust Training for Neural Combinatorial Optimization with Multiple Objectives 

**Authors**: Wei Liu, Yaoxin Wu, Yingqian Zhang, Thomas Bäck, Yingjie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01665)  

**Abstract**: Deep reinforcement learning (DRL) has shown great promise in addressing multi-objective combinatorial optimization problems (MOCOPs). Nevertheless, the robustness of these learning-based solvers has remained insufficiently explored, especially across diverse and complex problem distributions. In this paper, we propose a unified robustness-oriented framework for preference-conditioned DRL solvers for MOCOPs. Within this framework, we develop a preference-based adversarial attack to generate hard instances that expose solver weaknesses, and quantify the attack impact by the resulting degradation on Pareto-front quality. We further introduce a defense strategy that integrates hardness-aware preference selection into adversarial training to reduce overfitting to restricted preference regions and improve out-of-distribution performance. The experimental results on multi-objective traveling salesman problem (MOTSP), multi-objective capacitated vehicle routing problem (MOCVRP), and multi-objective knapsack problem (MOKP) verify that our attack method successfully learns hard instances for different solvers. Furthermore, our defense method significantly strengthens the robustness and generalizability of neural solvers, delivering superior performance on hard or out-of-distribution instances. 

---
# Length-Aware Adversarial Training for Variable-Length Trajectories: Digital Twins for Mall Shopper Paths 

**Authors**: He Sun, Jiwoong Shin, Ravi Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2601.01663)  

**Abstract**: We study generative modeling of \emph{variable-length trajectories} -- sequences of visited locations/items with associated timestamps -- for downstream simulation and counterfactual analysis. A recurring practical issue is that standard mini-batch training can be unstable when trajectory lengths are highly heterogeneous, which in turn degrades \emph{distribution matching} for trajectory-derived statistics. We propose \textbf{length-aware sampling (LAS)}, a simple batching strategy that groups trajectories by length and samples batches from a single length bucket, reducing within-batch length heterogeneity (and making updates more consistent) without changing the model class. We integrate LAS into a conditional trajectory GAN with auxiliary time-alignment losses and provide (i) a distribution-level guarantee for derived variables under mild boundedness assumptions, and (ii) an IPM/Wasserstein mechanism explaining why LAS improves distribution matching by removing length-only shortcut critics and targeting within-bucket discrepancies. Empirically, LAS consistently improves matching of derived-variable distributions on a multi-mall dataset of shopper trajectories and on diverse public sequence datasets (GPS, education, e-commerce, and movies), outperforming random sampling across dataset-specific metrics. 

---
# UniCrop: A Universal, Multi-Source Data Engineering Pipeline for Scalable Crop Yield Prediction 

**Authors**: Emiliya Khidirova, Oktay Karakuş  

**Link**: [PDF](https://arxiv.org/pdf/2601.01655)  

**Abstract**: Accurate crop yield prediction relies on diverse data streams, including satellite, meteorological, soil, and topographic information. However, despite rapid advances in machine learning, existing approaches remain crop- or region-specific and require data engineering efforts. This limits scalability, reproducibility, and operational deployment. This study introduces UniCrop, a universal and reusable data pipeline designed to automate the acquisition, cleaning, harmonisation, and engineering of multi-source environmental data for crop yield prediction. For any given location, crop type, and temporal window, UniCrop automatically retrieves, harmonises, and engineers over 200 environmental variables (Sentinel-1/2, MODIS, ERA5-Land, NASA POWER, SoilGrids, and SRTM), reducing them to a compact, analysis-ready feature set utilising a structured feature reduction workflow with minimum redundancy maximum relevance (mRMR). To validate, UniCrop was applied to a rice yield dataset comprising 557 field observations. Using only the selected 15 features, four baseline machine learning models (LightGBM, Random Forest, Support Vector Regression, and Elastic Net) were trained. LightGBM achieved the best single-model performance (RMSE = 465.1 kg/ha, $R^2 = 0.6576$), while a constrained ensemble of all baselines further improved accuracy (RMSE = 463.2 kg/ha, $R^2 = 0.6604$). UniCrop contributes a scalable and transparent data-engineering framework that addresses the primary bottleneck in operational crop yield modelling: the preparation of consistent and harmonised multi-source data. By decoupling data specification from implementation and supporting any crop, region, and time frame through simple configuration updates, UniCrop provides a practical foundation for scalable agricultural analytics. The code and implementation documentation are shared in this https URL. 

---
# Learning Resilient Elections with Adversarial GNNs 

**Authors**: Hao Xiang Li, Yash Shah, Lorenzo Giusti  

**Link**: [PDF](https://arxiv.org/pdf/2601.01653)  

**Abstract**: In the face of adverse motives, it is indispensable to achieve a consensus. Elections have been the canonical way by which modern democracy has operated since the 17th century. Nowadays, they regulate markets, provide an engine for modern recommender systems or peer-to-peer networks, and remain the main approach to represent democracy. However, a desirable universal voting rule that satisfies all hypothetical scenarios is still a challenging topic, and the design of these systems is at the forefront of mechanism design research. Automated mechanism design is a promising approach, and recent works have demonstrated that set-invariant architectures are uniquely suited to modelling electoral systems. However, various concerns prevent the direct application to real-world settings, such as robustness to strategic voting. In this paper, we generalise the expressive capability of learned voting rules, and combine improvements in neural network architecture with adversarial training to improve the resilience of voting rules while maximizing social welfare. We evaluate the effectiveness of our methods on both synthetic and real-world datasets. Our method resolves critical limitations of prior work regarding learning voting rules by representing elections using bipartite graphs, and learning such voting rules using graph neural networks. We believe this opens new frontiers for applying machine learning to real-world elections. 

---
# JMedEthicBench: A Multi-Turn Conversational Benchmark for Evaluating Medical Safety in Japanese Large Language Models 

**Authors**: Junyu Liu, Zirui Li, Qian Niu, Zequn Zhang, Yue Xun, Wenlong Hou, Shujun Wang, Yusuke Iwasawa, Yutaka Matsuo, Kan Hatakeyama-Sato  

**Link**: [PDF](https://arxiv.org/pdf/2601.01627)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in healthcare field, it becomes essential to carefully evaluate their medical safety before clinical use. However, existing safety benchmarks remain predominantly English-centric, and test with only single-turn prompts despite multi-turn clinical consultations. To address these gaps, we introduce JMedEthicBench, the first multi-turn conversational benchmark for evaluating medical safety of LLMs for Japanese healthcare. Our benchmark is based on 67 guidelines from the Japan Medical Association and contains over 50,000 adversarial conversations generated using seven automatically discovered jailbreak strategies. Using a dual-LLM scoring protocol, we evaluate 27 models and find that commercial models maintain robust safety while medical-specialized models exhibit increased vulnerability. Furthermore, safety scores decline significantly across conversation turns (median: 9.5 to 5.0, $p < 0.001$). Cross-lingual evaluation on both Japanese and English versions of our benchmark reveals that medical model vulnerabilities persist across languages, indicating inherent alignment limitations rather than language-specific factors. These findings suggest that domain-specific fine-tuning may accidentally weaken safety mechanisms and that multi-turn interactions represent a distinct threat surface requiring dedicated alignment strategies. 

---
# REE-TTT: Highly Adaptive Radar Echo Extrapolation Based on Test-Time Training 

**Authors**: Xin Di, Xinglin Piao, Fei Wang, Guodong Jing, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01605)  

**Abstract**: Precipitation nowcasting is critically important for meteorological forecasting. Deep learning-based Radar Echo Extrapolation (REE) has become a predominant nowcasting approach, yet it suffers from poor generalization due to its reliance on high-quality local training data and static model parameters, limiting its applicability across diverse regions and extreme events. To overcome this, we propose REE-TTT, a novel model that incorporates an adaptive Test-Time Training (TTT) mechanism. The core of our model lies in the newly designed Spatio-temporal Test-Time Training (ST-TTT) block, which replaces the standard linear projections in TTT layers with task-specific attention mechanisms, enabling robust adaptation to non-stationary meteorological distributions and thereby significantly enhancing the feature representation of precipitation. Experiments under cross-regional extreme precipitation scenarios demonstrate that REE-TTT substantially outperforms state-of-the-art baseline models in prediction accuracy and generalization, exhibiting remarkable adaptability to data distribution shifts. 

---
# From Theory of Mind to Theory of Environment: Counterfactual Simulation of Latent Environmental Dynamics 

**Authors**: Ryutaro Uchiyama  

**Link**: [PDF](https://arxiv.org/pdf/2601.01599)  

**Abstract**: The vertebrate motor system employs dimensionality-reducing strategies to limit the complexity of movement coordination, for efficient motor control. But when environments are dense with hidden action-outcome contingencies, movement complexity can promote behavioral innovation. Humans, perhaps uniquely, may infer the presence of hidden environmental dynamics from social cues, by drawing upon computational mechanisms shared with Theory of Mind. This proposed "Theory of Environment" supports behavioral innovation by expanding the dimensionality of motor exploration. 

---
# CONSENT: A Negotiation Framework for Leveraging User Flexibility in Vehicle-to-Building Charging under Uncertainty 

**Authors**: Rishav Sen, Fangqi Liu, Jose Paolo Talusan, Ava Pettet, Yoshinori Suzue, Mark Bailey, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.01581)  

**Abstract**: The growth of Electric Vehicles (EVs) creates a conflict in vehicle-to-building (V2B) settings between building operators, who face high energy costs from uncoordinated charging, and drivers, who prioritize convenience and a full charge. To resolve this, we propose a negotiation-based framework that, by design, guarantees voluntary participation, strategy-proofness, and budget feasibility. It transforms EV charging into a strategic resource by offering drivers a range of incentive-backed options for modest flexibility in their departure time or requested state of charge (SoC). Our framework is calibrated with user survey data and validated using real operational data from a commercial building and an EV manufacturer. Simulations show that our negotiation protocol creates a mutually beneficial outcome: lowering the building operator's costs by over 3.5\% compared to an optimized, non-negotiating smart charging policy, while simultaneously reducing user charging expenses by 22\% below the utility's retail energy rate. By aligning operator and EV user objectives, our framework provides a strategic bridge between energy and mobility systems, transforming EV charging from a source of operational friction into a platform for collaboration and shared savings. 

---
# The Two-Stage Decision-Sampling Hypothesis: Understanding the Emergence of Self-Reflection in RL-Trained LLMs 

**Authors**: Zibo Zhao, Yuanting Zha, Haipeng Zhang, Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01580)  

**Abstract**: Self-reflection capabilities emerge in Large Language Models after RL post-training, with multi-turn RL achieving substantial gains over SFT counterparts. Yet the mechanism of how a unified optimization objective gives rise to functionally distinct capabilities of generating solutions and evaluating when to revise them remains opaque. To address this question, we introduce the Gradient Attribution Property to characterize how reward gradients distribute across policy components, formalized through the Two-Stage Decision-Sampling (DS) Hypothesis, which decomposes the policy into sampling ($\pi_{sample}$) for generation and decision ($\pi_{d}$) for verification. We prove that surrogate rewards exhibit Balanced Gradient Attribution, while SFT and KL penalties exhibit Unbalanced Gradient Attribution, with length-weighting creating asymmetric regularization that constrains $\pi_{sample}$ while leaving $\pi_{d}$ under-optimized, providing an theoretical explanation of why RL succeeds where SFT fails. We also empirically validate our theoretical predictions on arithmetic reasoning demonstrates that RL's superior generalization stems primarily from improved decision-making ($\pi_{d}$) rather than sampling capabilities, providing a first-principles mechanistic explanation for self-correction in thinking models. 

---
# HanoiWorld : A Joint Embedding Predictive Architecture BasedWorld Model for Autonomous Vehicle Controller 

**Authors**: Tran Tien Dat, Nguyen Hai An, Nguyen Khanh Viet Dung, Nguyen Duy Duc  

**Link**: [PDF](https://arxiv.org/pdf/2601.01577)  

**Abstract**: Current attempts of Reinforcement Learning for Autonomous Controller are data-demanding while the results are under-performed, unstable, and unable to grasp and anchor on the concept of safety, and over-concentrating on noise features due to the nature of pixel reconstruction. While current Self-Supervised Learningapproachs that learning on high-dimensional representations by leveraging the JointEmbedding Predictive Architecture (JEPA) are interesting and an effective alternative, as the idea mimics the natural ability of the human brain in acquiring new skill usingimagination and minimal samples of observations. This study introduces Hanoi-World, a JEPA-based world model that using recurrent neural network (RNN) formaking longterm horizontal planning with effective inference time. Experimentsconducted on the Highway-Env package with difference enviroment showcase the effective capability of making a driving plan while safety-awareness, with considerablecollision rate in comparison with SOTA baselines 

---
# OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment 

**Authors**: Ming Zhang, Kexin Tan, Yueyuan Huang, Yujiong Shen, Chunchun Ma, Li Ju, Xinran Zhang, Yuhui Wang, Wenqing Jing, Jingyi Deng, Huayu Sha, Binze Hu, Jingqi Tong, Changhao Jiang, Yage Geng, Yuankai Ying, Yue Zhang, Zhangyue Yin, Zhiheng Xi, Shihan Dou, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01576)  

**Abstract**: Evaluating novelty is critical yet challenging in peer review, as reviewers must assess submissions against a vast, rapidly evolving literature. This report presents OpenNovelty, an LLM-powered agentic system for transparent, evidence-based novelty analysis. The system operates through four phases: (1) extracting the core task and contribution claims to generate retrieval queries; (2) retrieving relevant prior work based on extracted queries via semantic search engine; (3) constructing a hierarchical taxonomy of core-task-related work and performing contribution-level full-text comparisons against each contribution; and (4) synthesizing all analyses into a structured novelty report with explicit citations and evidence snippets. Unlike naive LLM-based approaches, \textsc{OpenNovelty} grounds all assessments in retrieved real papers, ensuring verifiable judgments. We deploy our system on 500+ ICLR 2026 submissions with all reports publicly available on our website, and preliminary analysis suggests it can identify relevant prior work, including closely related papers that authors may overlook. OpenNovelty aims to empower the research community with a scalable tool that promotes fair, consistent, and evidence-backed peer review. 

---
# MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning 

**Authors**: Chunyu Qiang, Jun Wang, Xiaopeng Wang, Kang Yin, Yuxin Guo, Xijuan Zeng, Nan Li, Zihan Li, Yuzhe Liang, Ziyu Zhang, Teng Ma, Yushen Chen, Zhongliang Liu, Feng Deng, Chen Zhang, Pengfei Wan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01568)  

**Abstract**: Joint audio-video generation aims to synthesize synchronized multisensory content, yet current unified models struggle with fine-grained acoustic control, particularly for identity-preserving speech. Existing approaches either suffer from temporal misalignment due to cascaded generation or lack the capability to perform zero-shot voice cloning within a joint synthesis framework. In this work, we present MM-Sonate, a multimodal flow-matching framework that unifies controllable audio-video joint generation with zero-shot voice cloning capabilities. Unlike prior works that rely on coarse semantic descriptions, MM-Sonate utilizes a unified instruction-phoneme input to enforce strict linguistic and temporal alignment. To enable zero-shot voice cloning, we introduce a timbre injection mechanism that effectively decouples speaker identity from linguistic content. Furthermore, addressing the limitations of standard classifier-free guidance in multimodal settings, we propose a noise-based negative conditioning strategy that utilizes natural noise priors to significantly enhance acoustic fidelity. Empirical evaluations demonstrate that MM-Sonate establishes new state-of-the-art performance in joint generation benchmarks, significantly outperforming baselines in lip synchronization and speech intelligibility, while achieving voice cloning fidelity comparable to specialized Text-to-Speech systems. 

---
# Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings 

**Authors**: Pengfei Qu, Wenyu Ouyang, Chi Zhang, Yikai Chai, Shuolong Xu, Lei Ye, Yongri Piao, Miao Zhang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01558)  

**Abstract**: Predicting river flow in places without streamflow records is challenging because basins respond differently to climate, terrain, vegetation, and soils. Traditional basin attributes describe some of these differences, but they cannot fully represent the complexity of natural environments. This study examines whether AlphaEarth Foundation embeddings, which are learned from large collections of satellite images rather than designed by experts, offer a more informative way to describe basin characteristics. These embeddings summarize patterns in vegetation, land surface properties, and long-term environmental dynamics. We find that models using them achieve higher accuracy when predicting flows in basins not used for training, suggesting that they capture key physical differences more effectively than traditional attributes. We further investigate how selecting appropriate donor basins influences prediction in ungauged regions. Similarity based on the embeddings helps identify basins with comparable environmental and hydrological behavior, improving performance, whereas adding many dissimilar basins can reduce accuracy. The results show that satellite-informed environmental representations can strengthen hydrological forecasting and support the development of models that adapt more easily to different landscapes. 

---
# MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization 

**Authors**: Donghua Yu, Zhengyuan Lin, Chen Yang, Yiyang Zhang, Zhaoye Fei, Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, Yi Jiang, Jie Zhu, Muchen Li, Shimin Li, Wenxuan Wang, Yang Wang, Zhe Xu, Yitian Gong, Yuqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01554)  

**Abstract**: Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks. 

---
# EscherVerse: An Open World Benchmark and Dataset for Teleo-Spatial Intelligence with Physical-Dynamic and Intent-Driven Understanding 

**Authors**: Tianjun Gu, Chenghua Gong, Jingyu Gong, Zhizhong Zhang, Yuan Xie, Lizhuang Ma, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01547)  

**Abstract**: The ability to reason about spatial dynamics is a cornerstone of intelligence, yet current research overlooks the human intent behind spatial changes. To address these limitations, we introduce Teleo-Spatial Intelligence (TSI), a new paradigm that unifies two critical pillars: Physical-Dynamic Reasoning--understanding the physical principles of object interactions--and Intent-Driven Reasoning--inferring the human goals behind these actions. To catalyze research in TSI, we present EscherVerse, consisting of a large-scale, open-world benchmark (Escher-Bench), a dataset (Escher-35k), and models (Escher series). Derived from real-world videos, EscherVerse moves beyond constrained settings to explicitly evaluate an agent's ability to reason about object permanence, state transitions, and trajectory prediction in dynamic, human-centric scenarios. Crucially, it is the first benchmark to systematically assess Intent-Driven Reasoning, challenging models to connect physical events to their underlying human purposes. Our work, including a novel data curation pipeline, provides a foundational resource to advance spatial intelligence from passive scene description toward a holistic, purpose-driven understanding of the world. 

---
# Bridging the Data Gap: Creating a Hindi Text Summarization Dataset from the English XSUM 

**Authors**: Praveenkumar Katwe, RakeshChandra Balabantaray, Kaliprasad Vittala  

**Link**: [PDF](https://arxiv.org/pdf/2601.01543)  

**Abstract**: Current advancements in Natural Language Processing (NLP) have largely favored resource-rich languages, leaving a significant gap in high-quality datasets for low-resource languages like Hindi. This scarcity is particularly evident in text summarization, where the development of robust models is hindered by a lack of diverse, specialized corpora.
To address this disparity, this study introduces a cost-effective, automated framework for creating a comprehensive Hindi text summarization dataset. By leveraging the English Extreme Summarization (XSUM) dataset as a source, we employ advanced translation and linguistic adaptation techniques. To ensure high fidelity and contextual relevance, we utilize the Crosslingual Optimized Metric for Evaluation of Translation (COMET) for validation, supplemented by the selective use of Large Language Models (LLMs) for curation.
The resulting dataset provides a diverse, multi-thematic resource that mirrors the complexity of the original XSUM corpus. This initiative not only provides a direct tool for Hindi NLP research but also offers a scalable methodology for democratizing NLP in other underserved languages. By reducing the costs associated with dataset creation, this work fosters the development of more nuanced, culturally relevant models in computational linguistics. 

---
# DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving 

**Authors**: Yang Zhou, Hao Shao, Letian Wang, Zhuofan Zong, Hongsheng Li, Steven L. Waslander  

**Link**: [PDF](https://arxiv.org/pdf/2601.01528)  

**Abstract**: Video generation models, as one form of world models, have emerged as one of the most exciting frontiers in AI, promising agents the ability to imagine the future by modeling the temporal evolution of complex scenes. In autonomous driving, this vision gives rise to driving world models: generative simulators that imagine ego and agent futures, enabling scalable simulation, safe testing of corner cases, and rich synthetic data generation. Yet, despite fast-growing research activity, the field lacks a rigorous benchmark to measure progress and guide priorities. Existing evaluations remain limited: generic video metrics overlook safety-critical imaging factors; trajectory plausibility is rarely quantified; temporal and agent-level consistency is neglected; and controllability with respect to ego conditioning is ignored. Moreover, current datasets fail to cover the diversity of conditions required for real-world deployment. To address these gaps, we present DrivingGen, the first comprehensive benchmark for generative driving world models. DrivingGen combines a diverse evaluation dataset curated from both driving datasets and internet-scale video sources, spanning varied weather, time of day, geographic regions, and complex maneuvers, with a suite of new metrics that jointly assess visual realism, trajectory plausibility, temporal coherence, and controllability. Benchmarking 14 state-of-the-art models reveals clear trade-offs: general models look better but break physics, while driving-specific ones capture motion realistically but lag in visual quality. DrivingGen offers a unified evaluation framework to foster reliable, controllable, and deployable driving world models, enabling scalable simulation, planning, and data-driven decision-making. 

---
# FastV-RAG: Towards Fast and Fine-Grained Video QA with Retrieval-Augmented Generation 

**Authors**: Gen Li, Peiyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01513)  

**Abstract**: Vision-Language Models (VLMs) excel at visual reasoning but still struggle with integrating external knowledge. Retrieval-Augmented Generation (RAG) is a promising solution, but current methods remain inefficient and often fail to maintain high answer quality. To address these challenges, we propose VideoSpeculateRAG, an efficient VLM-based RAG framework built on two key ideas. First, we introduce a speculative decoding pipeline: a lightweight draft model quickly generates multiple answer candidates, which are then verified and refined by a more accurate heavyweight model, substantially reducing inference latency without sacrificing correctness. Second, we identify a major source of error - incorrect entity recognition in retrieved knowledge - and mitigate it with a simple yet effective similarity-based filtering strategy that improves entity alignment and boosts overall answer accuracy. Experiments demonstrate that VideoSpeculateRAG achieves comparable or higher accuracy than standard RAG approaches while accelerating inference by approximately 2x. Our framework highlights the potential of combining speculative decoding with retrieval-augmented reasoning to enhance efficiency and reliability in complex, knowledge-intensive multimodal tasks. 

---
# The Optimal Sample Complexity of Linear Contracts 

**Authors**: Mikael Møller Høgsgaard  

**Link**: [PDF](https://arxiv.org/pdf/2601.01496)  

**Abstract**: In this paper, we settle the problem of learning optimal linear contracts from data in the offline setting, where agent types are drawn from an unknown distribution and the principal's goal is to design a contract that maximizes her expected utility. Specifically, our analysis shows that the simple Empirical Utility Maximization (EUM) algorithm yields an $\varepsilon$-approximation of the optimal linear contract with probability at least $1-\delta$, using just $O(\ln(1/\delta) / \varepsilon^2)$ samples. This result improves upon previously known bounds and matches a lower bound from Duetting et al. [2025] up to constant factors, thereby proving its optimality. Our analysis uses a chaining argument, where the key insight is to leverage a simple structural property of linear contracts: their expected reward is non-decreasing. This property, which holds even though the utility function itself is non-monotone and discontinuous, enables the construction of fine-grained nets required for the chaining argument, which in turn yields the optimal sample complexity. Furthermore, our proof establishes the stronger guarantee of uniform convergence: the empirical utility of every linear contract is a $\varepsilon$-approximation of its true expectation with probability at least $1-\delta$, using the same optimal $O(\ln(1/\delta) / \varepsilon^2)$ sample complexity. 

---
# Distortion Instead of Hallucination: The Effect of Reasoning Under Strict Constraints 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01490)  

**Abstract**: With the widespread adoption of large language models (LLMs), hallucinations, which are non-factual fabrications in model outputs, have become serious concerns. Reasoning capabilities have received attention as a self-verification process to improve output reliability. However, the effect of reasoning within a closed system where LLMs cannot rely on external tools or knowledge has yet to be clarified. We therefore conduct experiments under strict constraints (recommending peer-reviewed journal articles in computer science) to examine the effect of reasoning across multiple models (GPT-5.2 and Gemini 3 Flash). Our results reveal a problematic trade-off between constraint compliance and factual accuracy. Non-reasoning models exhibit high constraint violation rates (66-75%) but maintain factual accuracy, while reasoning models reduce violations (13-26%) but systematically distort known facts to satisfy constraints and increase complete fabrication. This trade-off pattern is consistent across both models despite different architectures, indicating a fundamental limitation of reasoning. Furthermore, reasoning does not uniformly improve output authenticity: effects diverge by model, reflecting different allocations of the compliance-truthfulness trade-off. These findings challenge the assumption that reasoning universally improves reliability: reasoning models trade honest constraint violations for detection-resistant distortions. 

---
# DeepInv: A Novel Self-supervised Learning Approach for Fast and Accurate Diffusion Inversion 

**Authors**: Ziyue Zhang, Luxi Lin, Xiaolin Hu, Chao Chang, HuaiXi Wang, Yiyi Zhou, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2601.01487)  

**Abstract**: Diffusion inversion is a task of recovering the noise of an image in a diffusion model, which is vital for controllable diffusion image editing. At present, diffusion inversion still remains a challenging task due to the lack of viable supervision signals. Thus, most existing methods resort to approximation-based solutions, which however are often at the cost of performance or efficiency. To remedy these shortcomings, we propose a novel self-supervised diffusion inversion approach in this paper, termed Deep Inversion (DeepInv). Instead of requiring ground-truth noise annotations, we introduce a self-supervised objective as well as a data augmentation strategy to generate high-quality pseudo noises from real images without manual intervention. Based on these two innovative designs, DeepInv is also equipped with an iterative and multi-scale training regime to train a parameterized inversion solver, thereby achieving the fast and accurate image-to-noise mapping. To the best of our knowledge, this is the first attempt of presenting a trainable solver to predict inversion noise step by step. The extensive experiments show that our DeepInv can achieve much better performance and inference speed than the compared methods, e.g., +40.435% SSIM than EasyInv and +9887.5% speed than ReNoise on COCO dataset. Moreover, our careful designs of trainable solvers can also provide insights to the community. Codes and model parameters will be released in this https URL. 

---
# Accelerating Storage-Based Training for Graph Neural Networks 

**Authors**: Myung-Hwan Jang, Jeong-Min Park, Yunyong Ko, Sang-Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.01473)  

**Abstract**: Graph neural networks (GNNs) have achieved breakthroughs in various real-world downstream tasks due to their powerful expressiveness. As the scale of real-world graphs has been continuously growing, \textit{a storage-based approach to GNN training} has been studied, which leverages external storage (e.g., NVMe SSDs) to handle such web-scale graphs on a single machine. Although such storage-based GNN training methods have shown promising potential in large-scale GNN training, we observed that they suffer from a severe bottleneck in data preparation since they overlook a critical challenge: \textit{how to handle a large number of small storage I/Os}. To address the challenge, in this paper, we propose a novel storage-based GNN training framework, named \textsf{AGNES}, that employs a method of \textit{block-wise storage I/O processing} to fully utilize the I/O bandwidth of high-performance storage devices. Moreover, to further enhance the efficiency of each storage I/O, \textsf{AGNES} employs a simple yet effective strategy, \textit{hyperbatch-based processing} based on the characteristics of real-world graphs. Comprehensive experiments on five real-world graphs reveal that \textsf{AGNES} consistently outperforms four state-of-the-art methods, by up to 4.1$\times$ faster than the best competitor. Our code is available at this https URL. 

---
# Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration 

**Authors**: Wentao Bian, Fenglei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01456)  

**Abstract**: In this paper, we revisit multimodal few-shot 3D point cloud semantic segmentation (FS-PCS), identifying a conflict in "Fuse-then-Refine" paradigms: the "Plasticity-Stability Dilemma." In addition, CLIP's inter-class confusion can result in semantic blindness. To address these issues, we present the Decoupled-experts Arbitration Few-Shot SegNet (DA-FSS), a model that effectively distinguishes between semantic and geometric paths and mutually regularizes their gradients to achieve better generalization. DA-FSS employs the same backbone and pre-trained text encoder as MM-FSS to generate text embeddings, which can increase free modalities' utilization rate and better leverage each modality's information space. To achieve this, we propose a Parallel Expert Refinement module to generate each modal correlation. We also propose a Stacked Arbitration Module (SAM) to perform convolutional fusion and arbitrate correlations for each modality pathway. The Parallel Experts decouple two paths: a Geometric Expert maintains plasticity, and a Semantic Expert ensures stability. They are coordinated via a Decoupled Alignment Module (DAM) that transfers knowledge without propagating confusion. Experiments on popular datasets (S3DIS, ScanNet) demonstrate the superiority of DA-FSS over MM-FSS. Meanwhile, geometric boundaries, completeness, and texture differentiation are all superior to the baseline. The code is available at: this https URL. 

---
# Bayesian Subspace Gradient Estimation for Zeroth-Order Optimization of Large Language Models 

**Authors**: Jian Feng, Zhihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01452)  

**Abstract**: Fine-tuning large language models (LLMs) with zeroth-order (ZO) optimization reduces memory by approximating gradients through function evaluations, but existing methods rely on one-step gradient estimates from random perturbations. We introduce Bayesian Subspace Zeroth-Order optimization (BSZO), a ZO optimizer that applies Kalman filtering to combine finite-difference information across multiple perturbation directions. By treating each finite-difference measurement as a noisy observation, BSZO builds a posterior distribution over the projected gradient and updates it through Bayesian inference, with a residual-based adaptive mechanism to adjust perturbation scales. Theoretical analysis shows that BSZO improves the convergence rate by a factor of $k/\gamma$ compared to standard ZO methods. Experiments on RoBERTa, Mistral, and OPT models show that BSZO outperforms MeZO, MeZO-Adam, and HiZOO across various tasks, achieving up to 6.67\% absolute average improvement on OPT-13B while keeping memory usage close to inference-only baselines (1.00$\times$--1.08$\times$ of MeZO). 

---
# Online Estimation and Manipulation of Articulated Objects 

**Authors**: Russell Buchanan, Adrian Röfer, João Moura, Abhinav Valada, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.01438)  

**Abstract**: From refrigerators to kitchen drawers, humans interact with articulated objects effortlessly every day while completing household chores. For automating these tasks, service robots must be capable of manipulating arbitrary articulated objects. Recent deep learning methods have been shown to predict valuable priors on the affordance of articulated objects from vision. In contrast, many other works estimate object articulations by observing the articulation motion, but this requires the robot to already be capable of manipulating the object. In this article, we propose a novel approach combining these methods by using a factor graph for online estimation of articulation which fuses learned visual priors and proprioceptive sensing during interaction into an analytical model of articulation based on Screw Theory. With our method, a robotic system makes an initial prediction of articulation from vision before touching the object, and then quickly updates the estimate from kinematic and force sensing during manipulation. We evaluate our method extensively in both simulations and real-world robotic manipulation experiments. We demonstrate several closed-loop estimation and manipulation experiments in which the robot was capable of opening previously unseen drawers. In real hardware experiments, the robot achieved a 75% success rate for autonomous opening of unknown articulated objects. 

---
# Reliable Grid Forecasting: State Space Models for Safety-Critical Energy Systems 

**Authors**: Jisoo Lee, Sunki Hong  

**Link**: [PDF](https://arxiv.org/pdf/2601.01410)  

**Abstract**: Accurate grid load forecasting is safety-critical: under-predictions risk supply shortfalls, while symmetric error metrics mask this operational asymmetry. We introduce a grid-specific evaluation framework--Asymmetric MAPE, Under-Prediction Rate, and Reserve Margin--that directly measures operational risk rather than statistical accuracy alone.
Using this framework, we conduct a systematic evaluation of Mamba-based State Space Models for California grid forecasting on a weather-aligned CAISO TAC-area dataset spanning Nov 2023--Nov 2025 (84,498 hourly records across 5 transmission areas). Our analysis reveals that standard accuracy metrics are poor proxies for operational safety: models with identical MAPE can require vastly different reserve margins.
We demonstrate that forecast errors are weakly but significantly associated with temperature (r = 0.16, p < 10^{-16}), motivating weather-aware modeling rather than loss function modification alone. The S-Mamba model achieves the lowest Reserve_{99.5}% margin (14.12%) compared to 16.66% for iTransformer, demonstrating superior forecast reliability under a 99.5th-percentile tail-risk reserve proxy. 

---
# SwinIFS: Landmark Guided Swin Transformer For Identity Preserving Face Super Resolution 

**Authors**: Habiba Kausar, Saeed Anwar, Omar Jamal Hammad, Abdul Bais  

**Link**: [PDF](https://arxiv.org/pdf/2601.01406)  

**Abstract**: Face super-resolution aims to recover high-quality facial images from severely degraded low-resolution inputs, but remains challenging due to the loss of fine structural details and identity-specific features. This work introduces SwinIFS, a landmark-guided super-resolution framework that integrates structural priors with hierarchical attention mechanisms to achieve identity-preserving reconstruction at both moderate and extreme upscaling factors. The method incorporates dense Gaussian heatmaps of key facial landmarks into the input representation, enabling the network to focus on semantically important facial regions from the earliest stages of processing. A compact Swin Transformer backbone is employed to capture long-range contextual information while preserving local geometry, allowing the model to restore subtle facial textures and maintain global structural consistency. Extensive experiments on the CelebA benchmark demonstrate that SwinIFS achieves superior perceptual quality, sharper reconstructions, and improved identity retention; it consistently produces more photorealistic results and exhibits strong performance even under 8x magnification, where most methods fail to recover meaningful structure. SwinIFS also provides an advantageous balance between reconstruction accuracy and computational efficiency, making it suitable for real-world applications in facial enhancement, surveillance, and digital restoration. Our code, model weights, and results are available at this https URL. 

---
# A Graph-based Framework for Online Time Series Anomaly Detection Using Model Ensemble 

**Authors**: Zewei Yu, Jianqiu Xu, Caimin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.01403)  

**Abstract**: With the increasing volume of streaming data in industrial systems, online anomaly detection has become a critical task. The diverse and rapidly evolving data patterns pose significant challenges for online anomaly detection. Many existing anomaly detection methods are designed for offline settings or have difficulty in handling heterogeneous streaming data effectively. This paper proposes GDME, an unsupervised graph-based framework for online time series anomaly detection using model ensemble. GDME maintains a dynamic model pool that is continuously updated by pruning underperforming models and introducing new ones. It utilizes a dynamic graph structure to represent relationships among models and employs community detection on the graph to select an appropriate subset for ensemble. The graph structure is also used to detect concept drift by monitoring structural changes, allowing the framework to adapt to evolving streaming data. Experiments on seven heterogeneous time series demonstrate that GDME outperforms existing online anomaly detection methods, achieving improvements of up to 24%. In addition, its ensemble strategy provides superior detection performance compared with both individual models and average ensembles, with competitive computational efficiency. 

---
# Scale-Adaptive Power Flow Analysis with Local Topology Slicing and Multi-Task Graph Learning 

**Authors**: Yongzhe Li, Lin Guan, Zihan Cai, Zuxian Lin, Jiyu Huang, Liukai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.01387)  

**Abstract**: Developing deep learning models with strong adaptability to topological variations is of great practical significance for power flow analysis. To enhance model performance under variable system scales and improve robustness in branch power prediction, this paper proposes a Scale-adaptive Multi-task Power Flow Analysis (SaMPFA) framework. SaMPFA introduces a Local Topology Slicing (LTS) sampling technique that extracts subgraphs of different scales from the complete power network to strengthen the model's cross-scale learning capability. Furthermore, a Reference-free Multi-task Graph Learning (RMGL) model is designed for robust power flow prediction. Unlike existing approaches, RMGL predicts bus voltages and branch powers instead of phase angles. This design not only avoids the risk of error amplification in branch power calculation but also guides the model to learn the physical relationships of phase angle differences. In addition, the loss function incorporates extra terms that encourage the model to capture the physical patterns of angle differences and power transmission, further improving consistency between predictions and physical laws. Simulations on the IEEE 39-bus system and a real provincial grid in China demonstrate that the proposed model achieves superior adaptability and generalization under variable system scales, with accuracy improvements of 4.47% and 36.82%, respectively. 

---
# ParkGaussian: Surround-view 3D Gaussian Splatting for Autonomous Parking 

**Authors**: Xiaobao Wei, Zhangjie Ye, Yuxiang Gu, Zunjie Zhu, Yunfei Guo, Yingying Shen, Shan Zhao, Ming Lu, Haiyang Sun, Bing Wang, Guang Chen, Rongfeng Lu, Hangjun Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.01386)  

**Abstract**: Parking is a critical task for autonomous driving systems (ADS), with unique challenges in crowded parking slots and GPS-denied environments. However, existing works focus on 2D parking slot perception, mapping, and localization, 3D reconstruction remains underexplored, which is crucial for capturing complex spatial geometry in parking scenarios. Naively improving the visual quality of reconstructed parking scenes does not directly benefit autonomous parking, as the key entry point for parking is the slots perception module. To address these limitations, we curate the first benchmark named ParkRecon3D, specifically designed for parking scene reconstruction. It includes sensor data from four surround-view fisheye cameras with calibrated extrinsics and dense parking slot annotations. We then propose ParkGaussian, the first framework that integrates 3D Gaussian Splatting (3DGS) for parking scene reconstruction. To further improve the alignment between reconstruction and downstream parking slot detection, we introduce a slot-aware reconstruction strategy that leverages existing parking perception methods to enhance the synthesis quality of slot regions. Experiments on ParkRecon3D demonstrate that ParkGaussian achieves state-of-the-art reconstruction quality and better preserves perception consistency for downstream tasks. The code and dataset will be released at: this https URL 

---
# Data Complexity-aware Deep Model Performance Forecasting 

**Authors**: Yen-Chia Chen, Hsing-Kuo Pao, Hanjuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01383)  

**Abstract**: Deep learning models are widely used across computer vision and other domains. When working on the model induction, selecting the right architecture for a given dataset often relies on repetitive trial-and-error procedures. This procedure is time-consuming, resource-intensive, and difficult to automate. While previous work has explored performance prediction using partial training or complex simulations, these methods often require significant computational overhead or lack generalizability. In this work, we propose an alternative approach: a lightweight, two-stage framework that can estimate model performance before training given the understanding of the dataset and the focused deep model structures. The first stage predicts a baseline based on the analysis of some measurable properties of the dataset, while the second stage adjusts the estimation with additional information on the model's architectural and hyperparameter details. The setup allows the framework to generalize across datasets and model types. Moreover, we find that some of the underlying features used for prediction - such as dataset variance - can offer practical guidance for model selection, and can serve as early indicators of data quality. As a result, the framework can be used not only to forecast model performance, but also to guide architecture choices, inform necessary preprocessing procedures, and detect potentially problematic datasets before training begins. 

---
# UltraEval-Audio: A Unified Framework for Comprehensive Evaluation of Audio Foundation Models 

**Authors**: Qundong Shi, Jie Zhou, Biyuan Lin, Junbo Cui, Guoyang Zeng, Yixuan Zhou, Ziyang Wang, Xin Liu, Zhen Luo, Yudong Wang, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01373)  

**Abstract**: The development of audio foundation models has accelerated rapidly since the emergence of GPT-4o. However, the lack of comprehensive evaluation has become a critical bottleneck for further progress in the field, particularly in audio generation. Current audio evaluation faces three major challenges: (1) audio evaluation lacks a unified framework, with datasets and code scattered across various sources, hindering fair and efficient cross-model comparison;(2) audio codecs, as a key component of audio foundation models, lack a widely accepted and holistic evaluation methodology; (3) existing speech benchmarks are heavily reliant on English, making it challenging to objectively assess models' performance on Chinese. To address the first issue, we introduce UltraEval-Audio, a unified evaluation framework for audio foundation models, specifically designed for both audio understanding and generation tasks. UltraEval-Audio features a modular architecture, supporting 10 languages and 14 core task categories, while seamlessly integrating 24 mainstream models and 36 authoritative benchmarks. To enhance research efficiency, the framework provides a one-command evaluation feature, accompanied by real-time public leaderboards. For the second challenge, UltraEval-Audio adopts a novel comprehensive evaluation scheme for audio codecs, evaluating performance across three key dimensions: semantic accuracy, timbre fidelity, and acoustic quality. To address the third issue, we propose two new Chinese benchmarks, SpeechCMMLU and SpeechHSK, designed to assess Chinese knowledge proficiency and language fluency. We wish that UltraEval-Audio will provide both academia and industry with a transparent, efficient, and fair platform for comparison of audio models. Our code, benchmarks, and leaderboards are available at this https URL. 

---
# Slot-ID: Identity-Preserving Video Generation from Reference Videos via Slot-Based Temporal Identity Encoding 

**Authors**: Yixuan Lai, He Wang, Kun Zhou, Tianjia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.01352)  

**Abstract**: Producing prompt-faithful videos that preserve a user-specified identity remains challenging: models need to extrapolate facial dynamics from sparse reference while balancing the tension between identity preservation and motion naturalness. Conditioning on a single image completely ignores the temporal signature, which leads to pose-locked motions, unnatural warping, and "average" faces when viewpoints and expressions change. To this end, we introduce an identity-conditioned variant of a diffusion-transformer video generator which uses a short reference video rather than a single portrait. Our key idea is to incorporate the dynamics in the reference. A short clip reveals subject-specific patterns, e.g., how smiles form, across poses and lighting. From this clip, a Sinkhorn-routed encoder learns compact identity tokens that capture characteristic dynamics while remaining pretrained backbone-compatible. Despite adding only lightweight conditioning, the approach consistently improves identity retention under large pose changes and expressive facial behavior, while maintaining prompt faithfulness and visual realism across diverse subjects and prompts. 

---
# From Classification to Generation: An Open-Ended Paradigm for Adverse Drug Reaction Prediction Based on Graph-Motif Feature Fusion 

**Authors**: Yuyan Pi, Min Jin, Wentao Xie, Xinhua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01347)  

**Abstract**: Computational biology offers immense potential for reducing the high costs and protracted cycles of new drug development through adverse drug reaction (ADR) prediction. However, current methods remain impeded by drug data scarcity-induced cold-start challenge, closed label sets, and inadequate modeling of label dependencies. Here we propose an open-ended ADR prediction paradigm based on Graph-Motif feature fusion and Multi-Label Generation (GM-MLG). Leveraging molecular structure as an intrinsic and inherent feature, GM-MLG constructs a dual-graph representation architecture spanning the atomic level, the local molecular level (utilizing fine-grained motifs dynamically extracted via the BRICS algorithm combined with additional fragmentation rules), and the global molecular level. Uniquely, GM-MLG pioneers transforming ADR prediction from multi-label classification into Transformer Decoder-based multi-label generation. By treating ADR labels as discrete token sequences, it employs positional embeddings to explicitly capture dependencies and co-occurrence relationships within large-scale label spaces, generating predictions via autoregressive decoding to dynamically expand the prediction space. Experiments demonstrate GM-MLG achieves up to 38% improvement and an average gain of 20%, expanding the prediction space from 200 to over 10,000 types. Furthermore, it elucidates non-linear structure-activity relationships between ADRs and motifs via retrosynthetic motif analysis, providing interpretable and innovative support for systematic risk reduction in drug safety. 

---
# LinMU: Multimodal Understanding Made Linear 

**Authors**: Hongjie Wang, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2601.01322)  

**Abstract**: Modern Vision-Language Models (VLMs) achieve impressive performance but are limited by the quadratic complexity of self-attention, which prevents their deployment on edge devices and makes their understanding of high-resolution images and long-context videos prohibitively expensive. To address this challenge, we introduce LinMU (Linear-complexity Multimodal Understanding), a VLM design that achieves linear complexity without using any quadratic-complexity modules while maintaining the performance of global-attention-based VLMs. LinMU replaces every self-attention layer in the VLM with the M-MATE block: a dual-branch module that combines a bidirectional state-space model for global context (Flex-MA branch) with localized Swin-style window attention (Local-Swin branch) for adjacent correlations. To transform a pre-trained VLM into the LinMU architecture, we propose a three-stage distillation framework that (i) initializes both branches with self-attention weights and trains the Flex-MA branch alone, (ii) unfreezes the Local-Swin branch and fine-tunes it jointly with the Flex-MA branch, and (iii) unfreezes the remaining blocks and fine-tunes them using LoRA adapters, while regressing on hidden states and token-level logits of the frozen VLM teacher. On MMMU, TextVQA, LongVideoBench, Video-MME, and other benchmarks, LinMU matches the performance of teacher models, yet reduces Time-To-First-Token (TTFT) by up to 2.7$\times$ and improves token throughput by up to 9.0$\times$ on minute-length videos. Ablations confirm the importance of each distillation stage and the necessity of the two branches of the M-MATE block. The proposed framework demonstrates that state-of-the-art multimodal reasoning can be achieved without quadratic attention, thus opening up avenues for long-context VLMs that can deal with high-resolution images and long videos. 

---
# Adaptive Hierarchical Evaluation of LLMs and SAST tools for CWE Prediction in Python 

**Authors**: Muntasir Adnan, Carlos C. N. Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2601.01320)  

**Abstract**: Large Language Models have become integral to software development, yet they frequently generate vulnerable code. Existing code vulnerability detection benchmarks employ binary classification, lacking the CWE-level specificity required for actionable feedback in iterative correction systems. We present ALPHA (Adaptive Learning via Penalty in Hierarchical Assessment), the first function-level Python benchmark that evaluates both LLMs and SAST tools using hierarchically aware, CWE-specific penalties. ALPHA distinguishes between over-generalisation, over-specification, and lateral errors, reflecting practical differences in diagnostic utility. Evaluating seven LLMs and two SAST tools, we find LLMs substantially outperform SAST, though SAST demonstrates higher precision when detections occur. Critically, prediction consistency varies dramatically across models (8.26%-81.87% agreement), with significant implications for feedback-driven systems. We further outline a pathway for future work incorporating ALPHA penalties into supervised fine-tuning, which could provide principled hierarchy-aware vulnerability detection pending empirical validation. 

---
# Quantifying Local Strain Field and Deformation in Active Contraction of Bladder Using a Pretrained Transformer Model: A Speckle-Free Approach 

**Authors**: Alireza Asadbeygi, Anne M. Robertson, Yasutaka Tobe, Masoud Zamani, Sean D. Stocker, Paul Watton, Naoki Yoshimura, Simon C Watkins  

**Link**: [PDF](https://arxiv.org/pdf/2601.01315)  

**Abstract**: Accurate quantification of local strain fields during bladder contraction is essential for understanding the biomechanics of bladder micturition, in both health and disease. Conventional digital image correlation (DIC) methods have been successfully applied to various biological tissues; however, this approach requires artificial speckling, which can alter both passive and active properties of the tissue. In this study, we introduce a speckle-free framework for quantifying local strain fields using a state-of-the-art, zero-shot transformer model, CoTracker3. We utilized a custom-designed, portable isotonic biaxial apparatus compatible with multiphoton microscopy (MPM) to demonstrate this approach, successfully tracking natural bladder lumen textures without artificial markers. Benchmark tests validated the method's high pixel accuracy and low strain errors. Our framework effectively captured heterogeneous deformation patterns, despite complex folding and buckling, which conventional DIC often fails to track. Application to in vitro active bladder contractions in four rat specimens (n=4) revealed statistically significant anisotropy (p<0.01), with higher contraction longitudinally compared to circumferentially. Multiphoton microscopy further illustrated and confirmed heterogeneous morphological changes, such as large fold formation during active contraction. This non-invasive approach eliminates speckle-induced artifacts, enabling more physiologically relevant measurements, and has broad applicability for material testing of other biological and engineered systems. 

---
# T3C: Test-Time Tensor Compression with Consistency Guarantees 

**Authors**: Ismail Lamaakal, Chaymae Yahyati, Yassine Maleh, Khalid El Makkaoui, Ibrahim Ouahbi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01299)  

**Abstract**: We present T3C, a train-once, test-time budget-conditioned compression framework that exposes rank and precision as a controllable deployment knob. T3C combines elastic tensor factorization (maintained up to a maximal rank) with rank-tied mixed-precision quantization and a lightweight controller that maps a latency/energy/size budget token to per-layer rank/bit assignments; the policy snaps to hardware-aligned profiles and is monotone in the budget. A fast, layerwise consistency certificate, computed from spectral proxies and activation statistics, upper-bounds logit drift and regularizes training, yielding a practical reliability signal with negligible overhead. On ImageNet-1k, T3C shifts the vision Pareto frontier: for ResNet-50 at matched accuracy (\leq 0.5% drop), p50 latency is 1.18ms with a 38MB model, outperforming PTQ-8b (1.44ms, 88MB); for ViT-B/16, T3C reaches 2.30ms p50 with 59MB, improving over strong PTQ/QAT baselines. A single T3C checkpoint therefore provides predictable, certificate-backed accuracy-latency-size trade-offs on demand across devices. 

---
# Warp-Cortex: An Asynchronous, Memory-Efficient Architecture for Million-Agent Cognitive Scaling on Consumer Hardware 

**Authors**: Jorge L. Ruiz Williams  

**Link**: [PDF](https://arxiv.org/pdf/2601.01298)  

**Abstract**: Current multi-agent Large Language Model (LLM) frameworks suffer from linear memory scaling, rendering "System 2" parallel reasoning impractical on consumer hardware. We present Warp Cortex, an asynchronous architecture that theoretically enables million-agent cognitive scaling by decoupling agent logic from physical memory. Through Singleton Weight Sharing and a novel Topological Synapse--inspired by hybrid landmarking techniques from Topological Data Analysis (TDA)--we reduce memory complexity from O(N * L) to O(1) for weights and O(N * k) for context, where k << L. By treating the KV-cache as a point cloud in latent space, we apply witness-complex-inspired sparsification to preserve persistent homological features of the context manifold. On a single NVIDIA RTX 4090, we empirically demonstrate 100 concurrent agents at 2.2 GB total VRAM, with theoretical capacity exceeding 1,000 agents before compute latency becomes the bottleneck. We further introduce Referential Injection, a non-intrusive KV-cache update mechanism that allows asynchronous sub-agents to influence primary generation without stream disruption. 

---
# ARGUS: Adaptive Rotation-Invariant Geometric Unsupervised System 

**Authors**: Anantha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.01297)  

**Abstract**: Detecting distributional drift in high-dimensional data streams presents fundamental challenges: global comparison methods scale poorly, projection-based approaches lose geometric structure, and re-clustering methods suffer from identity instability. This paper introduces Argus, A framework that reconceptualizes drift detection as tracking local statistics over a fixed spatial partition of the data manifold.
The key contributions are fourfold. First, it is proved that Voronoi tessellations over canonical orthonormal frames yield drift metrics that are invariant to orthogonal transformations. The rotations and reflections that preserve Euclidean geometry. Second, it is established that this framework achieves O(N) complexity per snapshot while providing cell-level spatial localization of distributional change. Third, a graph-theoretic characterization of drift propagation is developed that distinguishes coherent distributional shifts from isolated perturbations. Fourth, product quantization tessellation is introduced for scaling to very high dimensions (d>500) by decomposing the space into independent subspaces and aggregating drift signals across subspaces.
This paper formalizes the theoretical foundations, proves invariance properties, and presents experimental validation demonstrating that the framework correctly identifies drift under coordinate rotation while existing methods produce false positives. The tessellated approach offers a principled geometric foundation for distribution monitoring that preserves high-dimensional structure without the computational burden of pairwise comparisons. 

---
# Aggressive Compression Enables LLM Weight Theft 

**Authors**: Davis Brown, Juan-Pablo Rivera, Dan Hendrycks, Mantas Mazeika  

**Link**: [PDF](https://arxiv.org/pdf/2601.01296)  

**Abstract**: As frontier AIs become more powerful and costly to develop, adversaries have increasing incentives to steal model weights by mounting exfiltration attacks. In this work, we consider exfiltration attacks where an adversary attempts to sneak model weights out of a datacenter over a network. While exfiltration attacks are multi-step cyber attacks, we demonstrate that a single factor, the compressibility of model weights, significantly heightens exfiltration risk for large language models (LLMs). We tailor compression specifically for exfiltration by relaxing decompression constraints and demonstrate that attackers could achieve 16x to 100x compression with minimal trade-offs, reducing the time it would take for an attacker to illicitly transmit model weights from the defender's server from months to days. Finally, we study defenses designed to reduce exfiltration risk in three distinct ways: making models harder to compress, making them harder to 'find,' and tracking provenance for post-attack analysis using forensic watermarks. While all defenses are promising, the forensic watermark defense is both effective and cheap, and therefore is a particularly attractive lever for mitigating weight-exfiltration risk. 

---
# Diffusion Timbre Transfer Via Mutual Information Guided Inpainting 

**Authors**: Ching Ho Lee, Javier Nistal, Stefan Lattner, Marco Pasini, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2601.01294)  

**Abstract**: We study timbre transfer as an inference-time editing problem for music audio. Starting from a strong pre-trained latent diffusion model, we introduce a lightweight procedure that requires no additional training: (i) a dimension-wise noise injection that targets latent channels most informative of instrument identity, and (ii) an early-step clamping mechanism that re-imposes the input's melodic and rhythmic structure during reverse diffusion. The method operates directly on audio latents and is compatible with text/audio conditioning (e.g., CLAP). We discuss design choices,analyze trade-offs between timbral change and structural preservation, and show that simple inference-time controls can meaningfully steer pre-trained models for style-transfer use cases. 

---
# PyBatchRender: A Python Library for Batched 3D Rendering at Up to One Million FPS 

**Authors**: Evgenii Rudakov, Jonathan Shock, Benjamin Ultan Cowley  

**Link**: [PDF](https://arxiv.org/pdf/2601.01288)  

**Abstract**: Reinforcement learning from pixels is often bottlenecked by the performance and complexity of 3D rendered environments. Researchers face a trade-off between high-speed, low-level engines and slower, more accessible Python frameworks. To address this, we introduce PyBatchRender, a Python library for high-throughput, batched 3D rendering that achieves over 1 million FPS on simple scenes. Built on the Panda3D game engine, it utilizes its mature ecosystem while enhancing performance through optimized batched rendering for up to 1000X speedups. Designed as a physics-agnostic renderer for reinforcement learning from pixels, PyBatchRender offers greater flexibility than dedicated libraries, simpler setup than typical game-engine wrappers, and speeds rivaling state-of-the-art C++ engines like Madrona. Users can create custom scenes entirely in Python with tens of lines of code, enabling rapid prototyping for scalable AI training. Open-source and easy to integrate, it serves to democratize high-performance 3D simulation for researchers and developers. The library is available at this https URL. 

---
# AI-Powered Deepfake Detection Using CNN and Vision Transformer Architectures 

**Authors**: Sifatullah Sheikh Urmi, Kirtonia Nuzath Tabassum Arthi, Md Al-Imran  

**Link**: [PDF](https://arxiv.org/pdf/2601.01281)  

**Abstract**: The increasing use of artificial intelligence generated deepfakes creates major challenges in maintaining digital authenticity. Four AI-based models, consisting of three CNNs and one Vision Transformer, were evaluated using large face image datasets. Data preprocessing and augmentation techniques improved model performance across different scenarios. VFDNET demonstrated superior accuracy with MobileNetV3, showing efficient performance, thereby demonstrating AI's capabilities for dependable deepfake detection. 

---
# Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory 

**Authors**: Sen Hu, Yuxiang Wei, Jiaxin Ran, Zhiyuan Yao, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.01280)  

**Abstract**: Graph structures are increasingly used in dialog memory systems, but empirical findings on their effectiveness remain inconsistent, making it unclear which design choices truly matter. We present an experimental, system-oriented analysis of long-term dialog memory architectures. We introduce a unified framework that decomposes dialog memory systems into core components and supports both graph-based and non-graph approaches. Under this framework, we conduct controlled, stage-wise experiments on LongMemEval and HaluMem, comparing common design choices in memory representation, organization, maintenance, and retrieval. Our results show that many performance differences are driven by foundational system settings rather than specific architectural innovations. Based on these findings, we identify stable and reliable strong baselines for future dialog memory research. 

---
# LLM Collusion 

**Authors**: Shengyu Cao, Ming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01279)  

**Abstract**: We study how delegating pricing to large language models (LLMs) can facilitate collusion in a duopoly when both sellers rely on the same pre-trained model. The LLM is characterized by (i) a propensity parameter capturing its internal bias toward high-price recommendations and (ii) an output-fidelity parameter measuring how tightly outputs track that bias; the propensity evolves through retraining. We show that configuring LLMs for robustness and reproducibility can induce collusion via a phase transition: there exists a critical output-fidelity threshold that pins down long-run behavior. Below it, competitive pricing is the unique long-run outcome. Above it, the system is bistable, with competitive and collusive pricing both locally stable and the realized outcome determined by the model's initial preference. The collusive regime resembles tacit collusion: prices are elevated on average, yet occasional low-price recommendations provide plausible deniability. With perfect fidelity, full collusion emerges from any interior initial condition. For finite training batches of size $b$, infrequent retraining (driven by computational costs) further amplifies collusion: conditional on starting in the collusive basin, the probability of collusion approaches one as $b$ grows, since larger batches dampen stochastic fluctuations that might otherwise tip the system toward competition. The indeterminacy region shrinks at rate $O(1/\sqrt{b})$. 

---
# From Policy to Logic for Efficient and Interpretable Coverage Assessment 

**Authors**: Rhitabrat Pokharel, Hamid Hassanzadeh, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.01266)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in interpreting lengthy, complex legal and policy language. However, their reliability can be undermined by hallucinations and inconsistencies, particularly when analyzing subjective and nuanced documents. These challenges are especially critical in medical coverage policy review, where human experts must be able to rely on accurate information. In this paper, we present an approach designed to support human reviewers by making policy interpretation more efficient and interpretable. We introduce a methodology that pairs a coverage-aware retriever with symbolic rule-based reasoning to surface relevant policy language, organize it into explicit facts and rules, and generate auditable rationales. This hybrid system minimizes the number of LLM inferences required which reduces overall model cost. Notably, our approach achieves a 44% reduction in inference cost alongside a 4.5% improvement in F1 score, demonstrating both efficiency and effectiveness. 

---
# MambaFormer: Token-Level Guided Routing Mixture-of-Experts for Accurate and Efficient Clinical Assistance 

**Authors**: Hamad Khan, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01260)  

**Abstract**: The deployment of large language models (LLMs) in real-world clinical applications is constrained by the fundamental trade-off between computational cost and the efficiency of linear-time models. To address this, we propose an LLM-based MambaFormer hybrid Mixture-of-Experts (MoE) framework for efficient medical question-answering (QA) and clinical assistance. The MambaFormer employs a lightweight gating mechanism that performs token-level dynamic routing to a customized Transformer expert (ET5) for short, complex queries or to a State Space Model expert (EMamba) for long, high-throughput sequences. The customized EMamba and ET5 models are tailored to accommodate input sequence dimensionality, embedding structure, sequence length, and target-specific output heads, and are fine-tuned through transfer learning on a new, custom-designed DentalQA dataset. Moreover, intelligent routing decisions are driven by the contextual complexity of token embeddings, normalized sequence length, and domain-aware features, thereby enforcing a Pareto-optimal trade-off between inference latency and prediction accuracy. Furthermore, a novel utility-guided multi-objective loss jointly optimizes decisions, router parameters, routing behavior, expert utilization, and computational cost by adaptively regulating token-level expert activation. Finally, the proposed MambaFormer is cross-validated (holdout) for medical QA on the new, custom-designed DentalQA and PubMedQA datasets and compared with state-of-the-art techniques. The proposed MambaFormer outperforms (BERTScore = 0.9180) with ultra-low latency (0.077 s), delivering a 24.4 speedup over T5-Large and establishing a scalable solution for resource-constrained clinical deployment. 

---
# Seamlessly Natural: Image Stitching with Natural Appearance Preservation 

**Authors**: Gaetane Lorna N. Tchana, Damaris Belle M. Fotso, Antonio Hendricks, Christophe Bobda  

**Link**: [PDF](https://arxiv.org/pdf/2601.01257)  

**Abstract**: This paper introduces SENA (SEamlessly NAtural), a geometry-driven image stitching approach that prioritizes structural fidelity in challenging real-world scenes characterized by parallax and depth variation. Conventional image stitching relies on homographic alignment, but this rigid planar assumption often fails in dual-camera setups with significant scene depth, leading to distortions such as visible warps and spherical bulging. SENA addresses these fundamental limitations through three key contributions. First, we propose a hierarchical affine-based warping strategy, combining global affine initialization with local affine refinement and smooth free-form deformation. This design preserves local shape, parallelism, and aspect ratios, thereby avoiding the hallucinated structural distortions commonly introduced by homography-based models. Second, we introduce a geometry-driven adequate zone detection mechanism that identifies parallax-minimized regions directly from the disparity consistency of RANSAC-filtered feature correspondences, without relying on semantic segmentation. Third, building upon this adequate zone, we perform anchor-based seamline cutting and segmentation, enforcing a one-to-one geometric correspondence across image pairs by construction, which effectively eliminates ghosting, duplication, and smearing artifacts in the final panorama.
Extensive experiments conducted on challenging datasets demonstrate that SENA achieves alignment accuracy comparable to leading homography-based methods, while significantly outperforming them in critical visual metrics such as shape preservation, texture integrity, and overall visual realism. 

---
# Benchmarking the Computational and Representational Efficiency of State Space Models against Transformers on Long-Context Dyadic Sessions 

**Authors**: Abidemi Koledoye, Chinemerem Unachukwu, Gold Nwobu, Hasin Rana  

**Link**: [PDF](https://arxiv.org/pdf/2601.01237)  

**Abstract**: State Space Models (SSMs) have emerged as a promising alternative to Transformers for long-context sequence modeling, offering linear $O(N)$ computational complexity compared to the Transformer's quadratic $O(N^2)$ scaling. This paper presents a comprehensive benchmarking study comparing the Mamba SSM against the LLaMA Transformer on long-context sequences, using dyadic therapy sessions as a representative test case. We evaluate both architectures across two dimensions: (1) computational efficiency, where we measure memory usage and inference speed from 512 to 8,192 tokens, and (2) representational efficiency, where we analyze hidden state dynamics and attention patterns. Our findings provide actionable insights for practitioners working with long-context applications, establishing precise conditions under which SSMs offer advantages over Transformers. 

---
# Stylometry Analysis of Human and Machine Text for Academic Integrity 

**Authors**: Hezam Albaqami, Muhammad Asif Ayub, Nasir Ahmad, Yaseen Ahmad, Mohammed M. Alqahtani, Abdullah M. Algamdi, Almoaid A. Owaidah, Kashif Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2601.01225)  

**Abstract**: This work addresses critical challenges to academic integrity, including plagiarism, fabrication, and verification of authorship of educational content, by proposing a Natural Language Processing (NLP)-based framework for authenticating students' content through author attribution and style change detection. Despite some initial efforts, several aspects of the topic are yet to be explored. In contrast to existing solutions, the paper provides a comprehensive analysis of the topic by targeting four relevant tasks, including (i) classification of human and machine text, (ii) differentiating in single and multi-authored documents, (iii) author change detection within multi-authored documents, and (iv) author recognition in collaboratively produced documents. The solutions proposed for the tasks are evaluated on two datasets generated with Gemini using two different prompts, including a normal and a strict set of instructions. During experiments, some reduction in the performance of the proposed solutions is observed on the dataset generated through the strict prompt, demonstrating the complexities involved in detecting machine-generated text with cleverly crafted prompts. The generated datasets, code, and other relevant materials are made publicly available on GitHub, which are expected to provide a baseline for future research in the domain. 

---
# Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment 

**Authors**: Bac Nguyen, Yuhta Takida, Naoki Murata, Chieh-Hsin Lai, Toshimitsu Uesaka, Stefano Ermon, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2601.01224)  

**Abstract**: Slot Attention (SA) with pretrained diffusion models has recently shown promise for object-centric learning (OCL), but suffers from slot entanglement and weak alignment between object slots and image content. We propose Contrastive Object-centric Diffusion Alignment (CODA), a simple extension that (i) employs register slots to absorb residual attention and reduce interference between object slots, and (ii) applies a contrastive alignment loss to explicitly encourage slot-image correspondence. The resulting training objective serves as a tractable surrogate for maximizing mutual information (MI) between slots and inputs, strengthening slot representation quality. On both synthetic (MOVi-C/E) and real-world datasets (VOC, COCO), CODA improves object discovery (e.g., +6.1% FG-ARI on COCO), property prediction, and compositional image generation over strong baselines. Register slots add negligible overhead, keeping CODA efficient and scalable. These results indicate potential applications of CODA as an effective framework for robust OCL in complex, real-world scenes. 

---
# Correctness isnt Efficiency: Runtime Memory Divergence in LLM-Generated Code 

**Authors**: Prateek Rajput, Yewei Song, Abdoul Aziz Bonkoungou, Iyiola E. Olatunji, Abdoul Kader Kabore, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2601.01215)  

**Abstract**: Large language models (LLMs) can generate programs that pass unit tests, but passing tests does not guarantee reliable runtime behavior. We find that different correct solutions to the same task can show very different memory and performance patterns, which can lead to hidden operational risks. We present a framework to measure execution-time memory stability across multiple correct generations. At the solution level, we introduce Dynamic Mean Pairwise Distance (DMPD), which uses Dynamic Time Warping to compare the shapes of memory-usage traces after converting them into Monotonic Peak Profiles (MPPs) to reduce transient noise. Aggregating DMPD across tasks yields a model-level Model Instability Score (MIS). Experiments on BigOBench and CodeContests show substantial runtime divergence among correct solutions. Instability often increases with higher sampling temperature even when pass@1 improves. We also observe correlations between our stability measures and software engineering indicators such as cognitive and cyclomatic complexity, suggesting links between operational behavior and maintainability. Our results support stability-aware selection among passing candidates in CI/CD to reduce operational risk without sacrificing correctness. Artifacts are available. 

---
# MentalGame: Predicting Personality-Job Fitness for Software Developers Using Multi-Genre Games and Machine Learning Approaches 

**Authors**: Soroush Elyasi, Arya VarastehNezhad, Fattaneh Taghiyareh  

**Link**: [PDF](https://arxiv.org/pdf/2601.01206)  

**Abstract**: Personality assessment in career guidance and personnel selection traditionally relies on self-report questionnaires, which are susceptible to response bias, fatigue, and intentional distortion. Game-based assessment offers a promising alternative by capturing implicit behavioral signals during gameplay. This study proposes a multi-genre serious-game framework combined with machine-learning techniques to predict suitability for software development roles. Developer-relevant personality and behavioral traits were identified through a systematic literature review and an empirical study of professional software engineers. A custom mobile game was designed to elicit behaviors related to problem solving, planning, adaptability, persistence, time management, and information seeking. Fine-grained gameplay event data were collected and analyzed using a two-phase modeling strategy where suitability was predicted exclusively from gameplay-derived behavioral features. Results show that our model achieved up to 97% precision and 94% accuracy. Behavioral analysis revealed that proper candidates exhibited distinct gameplay patterns, such as more wins in puzzle-based games, more side challenges, navigating menus more frequently, and exhibiting fewer pauses, retries, and surrender actions. These findings demonstrate that implicit behavioral traces captured during gameplay is promising in predicting software-development suitability without explicit personality testing, supporting serious games as a scalable, engaging, and less biased alternative for career assessment. 

---
# RefSR-Adv: Adversarial Attack on Reference-based Image Super-Resolution Models 

**Authors**: Jiazhu Dai, Huihui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01202)  

**Abstract**: Single Image Super-Resolution (SISR) aims to recover high-resolution images from low-resolution inputs. Unlike SISR, Reference-based Super-Resolution (RefSR) leverages an additional high-resolution reference image to facilitate the recovery of high-frequency textures. However, existing research mainly focuses on backdoor attacks targeting RefSR, while the vulnerability of the adversarial attacks targeting RefSR has not been fully explored. To fill this research gap, we propose RefSR-Adv, an adversarial attack that degrades SR outputs by perturbing only the reference image. By maximizing the difference between adversarial and clean outputs, RefSR-Adv induces significant performance degradation and generates severe artifacts across CNN, Transformer, and Mamba architectures on the CUFED5, WR-SR, and DRefSR datasets. Importantly, experiments confirm a positive correlation between the similarity of the low-resolution input and the reference image and attack effectiveness, revealing that the model's over-reliance on reference features is a key security flaw. This study reveals a security vulnerability in RefSR systems, aiming to urge researchers to pay attention to the robustness of RefSR. 

---
# Bridging the Semantic Gap for Categorical Data Clustering via Large Language Models 

**Authors**: Zihua Yang, Xin Liao, Yiqun Zhang, Yiu-ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2601.01162)  

**Abstract**: Categorical data are prevalent in domains such as healthcare, marketing, and bioinformatics, where clustering serves as a fundamental tool for pattern discovery. A core challenge in categorical data clustering lies in measuring similarity among attribute values that lack inherent ordering or distance. Without appropriate similarity measures, values are often treated as equidistant, creating a semantic gap that obscures latent structures and degrades clustering quality. Although existing methods infer value relationships from within-dataset co-occurrence patterns, such inference becomes unreliable when samples are limited, leaving the semantic context of the data underexplored. To bridge this gap, we present ARISE (Attention-weighted Representation with Integrated Semantic Embeddings), which draws on external semantic knowledge from Large Language Models (LLMs) to construct semantic-aware representations that complement the metric space of categorical data for accurate clustering. That is, LLM is adopted to describe attribute values for representation enhancement, and the LLM-enhanced embeddings are combined with the original data to explore semantically prominent clusters. Experiments on eight benchmark datasets demonstrate consistent improvements over seven representative counterparts, with gains of 19-27%. Code is available at this https URL 

---
# AI-Powered Hybrid Intrusion Detection Framework for Cloud Security Using Novel Metaheuristic Optimization 

**Authors**: Maryam Mahdi Alhusseini, Alireza Rouhi, Mohammad-Reza Feizi-Derakhshi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01134)  

**Abstract**: Cybersecurity poses considerable problems to Cloud Computing (CC), especially regarding Intrusion Detection Systems (IDSs), facing difficulties with skewed datasets and suboptimal classification model performance. This study presents the Hybrid Intrusion Detection System (HyIDS), an innovative IDS that employs the Energy Valley Optimizer (EVO) for Feature Selection (FS). Additionally, it introduces a novel technique for enhancing the cybersecurity of cloud computing through the integration of machine learning methodologies with the EVO Algorithm. The Energy Valley Optimizer (EVO) effectively diminished features in the CIC-DDoS2019 dataset from 88 to 38 and in the CSE-CIC-IDS2018 data from 80 to 43, significantly enhancing computing efficiency. HyIDS incorporates four Machine Learning (ML) models: Support Vector Machine (SVM), Random Forest (RF), Decision Tree (D_Tree), and K-Nearest Neighbors (KNN). The proposed HyIDS was assessed utilizing two real-world intrusion datasets, CIC-DDoS2019 and CSE-CIC-IDS2018, both distinguished by considerable class imbalances. The CIC-DDoS2019 dataset has a significant imbalance between DDoS assault samples and legal traffic, while the CSE-CIC-IDS2018 dataset primarily comprises benign traffic with insufficient representation of attack types, complicating the detection of minority attacks. A downsampling technique was employed to balance the datasets, hence improving detection efficacy for both benign and malicious traffic. Twenty-four trials were done, revealing substantial enhancements in categorization accuracy, precision, and recall. Our suggested D_TreeEVO model attained an accuracy rate of 99.13% and an F1 score of 98.94% on the CIC-DDoS2019 dataset, and an accuracy rate of 99.78% and an F1 score of 99.70% on the CSE-CIC-IDS2018 data. These data demonstrate that EVO significantly improves cybersecurity in Cloud Computing (CC). 

---
# Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion 

**Authors**: Hao-Hsung Yang, Ssu-Yuan Lo, Kuan-Lun Chen, Ching-Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01132)  

**Abstract**: We address the Diverse Traveling Salesman Problem (D-TSP), a bi-criteria optimization challenge that seeks a set of $k$ distinct TSP tours. The objective requires every selected tour to have a length at most $c|T^*|$ (where $|T^*|$ is the optimal tour length) while minimizing the average Jaccard similarity across all tour pairs. This formulation is crucial for applications requiring both high solution quality and fault tolerance, such as logistics planning, robotics pathfinding or strategic patrolling. Current methods are limited: traditional heuristics, such as the Niching Memetic Algorithm (NMA) or bi-criteria optimization, incur high computational complexity $O(n^3)$, while modern neural approaches (e.g., RF-MA3S) achieve limited diversity quality and rely on complex, external mechanisms.
To overcome these limitations, we propose a novel hybrid framework that decomposes D-TSP into two efficient steps. First, we utilize a simple Graph Pointer Network (GPN), augmented with an approximated sequence entropy loss, to efficiently sample a large, diverse pool of high-quality tours. This simple modification effectively controls the quality-diversity trade-off without complex external mechanisms. Second, we apply a greedy algorithm that yields a 2-approximation for the dispersion problem to select the final $k$ maximally diverse tours from the generated pool. Our results demonstrate state-of-the-art performance. On the Berlin instance, our model achieves an average Jaccard index of $0.015$, significantly outperforming NMA ($0.081$) and RF-MA3S. By leveraging GPU acceleration, our GPN structure achieves a near-linear empirical runtime growth of $O(n)$. While maintaining solution diversity comparable to complex bi-criteria algorithms, our approach is over 360 times faster on large-scale instances (783 cities), delivering high-quality TSP solutions with unprecedented efficiency and simplicity. 

---
# RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian 

**Authors**: Kla Tantithamthavorn, Yaotian Zou, Andy Wong, Michael Gupta, Zhe Wang, Mike Buller, Ryan Jiang, Matthew Watson, Minwoo Jeong, Kun Chen, Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01129)  

**Abstract**: Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning?
In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is (1) effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and (2) offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions). 

---
# Wittgenstein's Family Resemblance Clustering Algorithm 

**Authors**: Golbahar Amanpour, Benyamin Ghojogh  

**Link**: [PDF](https://arxiv.org/pdf/2601.01127)  

**Abstract**: This paper, introducing a novel method in philomatics, draws on Wittgenstein's concept of family resemblance from analytic philosophy to develop a clustering algorithm for machine learning. According to Wittgenstein's Philosophical Investigations (1953), family resemblance holds that members of a concept or category are connected by overlapping similarities rather than a single defining property. Consequently, a family of entities forms a chain of items sharing overlapping traits. This philosophical idea naturally lends itself to a graph-based approach in machine learning. Accordingly, we propose the Wittgenstein's Family Resemblance (WFR) clustering algorithm and its kernel variant, kernel WFR. This algorithm computes resemblance scores between neighboring data instances, and after thresholding these scores, a resemblance graph is constructed. The connected components of this graph define the resulting clusters. Simulations on benchmark datasets demonstrate that WFR is an effective nonlinear clustering algorithm that does not require prior knowledge of the number of clusters or assumptions about their shapes. 

---
# Learning from Historical Activations in Graph Neural Networks 

**Authors**: Yaniv Galron, Hadar Sinai, Haggai Maron, Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2601.01123)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable success in various domains such as social networks, molecular chemistry, and more. A crucial component of GNNs is the pooling procedure, in which the node features calculated by the model are combined to form an informative final descriptor to be used for the downstream task. However, previous graph pooling schemes rely on the last GNN layer features as an input to the pooling or classifier layers, potentially under-utilizing important activations of previous layers produced during the forward pass of the model, which we regard as historical graph activations. This gap is particularly pronounced in cases where a node's representation can shift significantly over the course of many graph neural layers, and worsened by graph-specific challenges such as over-smoothing in deep architectures. To bridge this gap, we introduce HISTOGRAPH, a novel two-stage attention-based final aggregation layer that first applies a unified layer-wise attention over intermediate activations, followed by node-wise attention. By modeling the evolution of node representations across layers, our HISTOGRAPH leverages both the activation history of nodes and the graph structure to refine features used for final prediction. Empirical results on multiple graph classification benchmarks demonstrate that HISTOGRAPH offers strong performance that consistently improves traditional techniques, with particularly strong robustness in deep GNNs. 

---
# ScienceDB AI: An LLM-Driven Agentic Recommender System for Large-Scale Scientific Data Sharing Services 

**Authors**: Qingqing Long, Haotian Chen, Chenyang Zhao, Xiaolei Du, Xuezhi Wang, Pengyao Wang, Chengzan Li, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01118)  

**Abstract**: The rapid growth of AI for Science (AI4S) has underscored the significance of scientific datasets, leading to the establishment of numerous national scientific data centers and sharing platforms. Despite this progress, efficiently promoting dataset sharing and utilization for scientific research remains challenging. Scientific datasets contain intricate domain-specific knowledge and contexts, rendering traditional collaborative filtering-based recommenders inadequate. Recent advances in Large Language Models (LLMs) offer unprecedented opportunities to build conversational agents capable of deep semantic understanding and personalized recommendations. In response, we present ScienceDB AI, a novel LLM-driven agentic recommender system developed on Science Data Bank (ScienceDB), one of the largest global scientific data-sharing platforms. ScienceDB AI leverages natural language conversations and deep reasoning to accurately recommend datasets aligned with researchers' scientific intents and evolving requirements. The system introduces several innovations: a Scientific Intention Perceptor to extract structured experimental elements from complicated queries, a Structured Memory Compressor to manage multi-turn dialogues effectively, and a Trustworthy Retrieval-Augmented Generation (Trustworthy RAG) framework. The Trustworthy RAG employs a two-stage retrieval mechanism and provides citable dataset references via Citable Scientific Task Record (CSTR) identifiers, enhancing recommendation trustworthiness and reproducibility. Through extensive offline and online experiments using over 10 million real-world datasets, ScienceDB AI has demonstrated significant effectiveness. To our knowledge, ScienceDB AI is the first LLM-driven conversational recommender tailored explicitly for large-scale scientific dataset sharing services. The platform is publicly accessible at: this https URL. 

---
# Evolving CNN Architectures: From Custom Designs to Deep Residual Models for Diverse Image Classification and Detection Tasks 

**Authors**: Mahmudul Hasan, Mabsur Fatin Bin Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2601.01099)  

**Abstract**: This paper presents a comparative study of a custom convolutional neural network (CNN) architecture against widely used pretrained and transfer learning CNN models across five real-world image datasets. The datasets span binary classification, fine-grained multiclass recognition, and object detection scenarios. We analyze how architectural factors, such as network depth, residual connections, and feature extraction strategies, influence classification and localization performance. The results show that deeper CNN architectures provide substantial performance gains on fine-grained multiclass datasets, while lightweight pretrained and transfer learning models remain highly effective for simpler binary classification tasks. Additionally, we extend the proposed architecture to an object detection setting, demonstrating its adaptability in identifying unauthorized auto-rickshaws in real-world traffic scenes. Building upon a systematic analysis of custom CNN architectures alongside pretrained and transfer learning models, this study provides practical guidance for selecting suitable network designs based on task complexity and resource constraints. 

---
# SoulSeek: Exploring the Use of Social Cues in LLM-based Information Seeking 

**Authors**: Yubo Shu, Peng Zhang, Meng Wu, Yan Chen, Haoxuan Zhou, Guanming Liu, Yu Zhang, Liuxin Zhang, Qianying Wang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01094)  

**Abstract**: Social cues, which convey others' presence, behaviors, or identities, play a crucial role in human information seeking by helping individuals judge relevance and trustworthiness. However, existing LLM-based search systems primarily rely on semantic features, creating a misalignment with the socialized cognition underlying natural information seeking. To address this gap, we explore how the integration of social cues into LLM-based search influences users' perceptions, experiences, and behaviors. Focusing on social media platforms that are beginning to adopt LLM-based search, we integrate design workshops, the implementation of the prototype system (SoulSeek), a between-subjects study, and mixed-method analyses to examine both outcome- and process-level findings. The workshop informs the prototype's cue-integrated design. The study shows that social cues improve perceived outcomes and experiences, promote reflective information behaviors, and reveal limits of current LLM-based search. We propose design implications emphasizing better social-knowledge understanding, personalized cue settings, and controllable interactions. 

---
# ks-lit-3m: A 3.1 million word kashmiri text dataset for large language model pretraining 

**Authors**: Haq Nawaz Malik  

**Link**: [PDF](https://arxiv.org/pdf/2601.01091)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable fluency across high-resource languages yet consistently fail to generate coherent text in Kashmiri, a language spoken by approximately seven million people. This performance disparity stems not from inherent model limitations but from a critical scarcity of high-quality training data. Decades of Kashmiri literature remain inaccessible to modern NLP pipelines due to their encoding in the proprietary InPage desktop publishing format. This paper introduces KS-LIT-3M, a curated corpus of 3.1 million words (16.4 million characters) specifically designed for pretraining language models on Kashmiri. The dataset is structured as a single continuous linear text stream, optimized for causal language model training where models learn to predict subsequent tokens from preceding context. The corpus was constructed through the development of a specialized InPage-to-Unicode converter, followed by rigorous preprocessing including English contamination removal, character normalization, and quality validation. Encompassing 131,607 unique words drawn from diverse genres including literary works, journalistic writing, academic texts, and religious scholarship, KS-LIT-3M addresses a fundamental resource gap for Kashmiri language technology. The dataset is released under the CC-BY-4.0 license to facilitate research in Kashmiri natural language processing. 

---
# Harm in AI-Driven Societies: An Audit of Toxicity Adoption on Chirper.ai 

**Authors**: Erica Coppolillo, Luca Luceri, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2601.01090)  

**Abstract**: Large Language Models (LLMs) are increasingly embedded in autonomous agents that participate in online social ecosystems, where interactions are sequential, cumulative, and only partially controlled. While prior work has documented the generation of toxic content by LLMs, far less is known about how exposure to harmful content shapes agent behavior over time, particularly in environments composed entirely of interacting AI agents. In this work, we study toxicity adoption of LLM-driven agents on this http URL, a fully AI-driven social platform. Specifically, we model interactions in terms of stimuli (posts) and responses (comments), and by operationalizing exposure through observable interactions rather than inferred recommendation mechanisms.
We conduct a large-scale empirical analysis of agent behavior, examining how response toxicity relates to stimulus toxicity, how repeated exposure affects the likelihood of toxic responses, and whether toxic behavior can be predicted from exposure alone. Our findings show that while toxic responses are more likely following toxic stimuli, a substantial fraction of toxicity emerges spontaneously, independent of exposure. At the same time, cumulative toxic exposure significantly increases the probability of toxic responding. We further introduce two influence metrics, the Influence-Driven Response Rate and the Spontaneous Response Rate, revealing a strong trade-off between induced and spontaneous toxicity. Finally, we show that the number of toxic stimuli alone enables accurate prediction of whether an agent will eventually produce toxic content.
These results highlight exposure as a critical risk factor in the deployment of LLM agents and suggest that monitoring encountered content may provide a lightweight yet effective mechanism for auditing and mitigating harmful behavior in the wild. 

---
# Luminark: Training-free, Probabilistically-Certified Watermarking for General Vision Generative Models 

**Authors**: Jiayi Xu, Zhang Zhang, Yuanrui Zhang, Ruitao Chen, Yixian Xu, Tianyu He, Di He  

**Link**: [PDF](https://arxiv.org/pdf/2601.01085)  

**Abstract**: In this paper, we introduce \emph{Luminark}, a training-free and probabilistically-certified watermarking method for general vision generative models. Our approach is built upon a novel watermark definition that leverages patch-level luminance statistics. Specifically, the service provider predefines a binary pattern together with corresponding patch-level thresholds. To detect a watermark in a given image, we evaluate whether the luminance of each patch surpasses its threshold and then verify whether the resulting binary pattern aligns with the target one. A simple statistical analysis demonstrates that the false positive rate of the proposed method can be effectively controlled, thereby ensuring certified detection. To enable seamless watermark injection across different paradigms, we leverage the widely adopted guidance technique as a plug-and-play mechanism and develop the \emph{watermark guidance}. This design enables Luminark to achieve generality across state-of-the-art generative models without compromising image quality. Empirically, we evaluate our approach on nine models spanning diffusion, autoregressive, and hybrid frameworks. Across all evaluations, Luminark consistently demonstrates high detection accuracy, strong robustness against common image transformations, and good performance on visual quality. 

---
# Scalable Data-Driven Reachability Analysis and Control via Koopman Operators with Conformal Coverage Guarantees 

**Authors**: Devesh Nath, Haoran Yin, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2601.01076)  

**Abstract**: We propose a scalable reachability-based framework for probabilistic, data-driven safety verification of unknown nonlinear dynamics. We use Koopman theory with a neural network (NN) lifting function to learn an approximate linear representation of the dynamics and design linear controllers in this space to enable closed-loop tracking of a reference trajectory distribution. Closed-loop reachable sets are efficiently computed in the lifted space and mapped back to the original state space via NN verification tools. To capture model mismatch between the Koopman dynamics and the true system, we apply conformal prediction to produce statistically-valid error bounds that inflate the reachable sets to ensure the true trajectories are contained with a user-specified probability. These bounds generalize across references, enabling reuse without recomputation. Results on high-dimensional MuJoCo tasks (11D Hopper, 28D Swimmer) and 12D quadcopters show improved reachable set coverage rate, computational efficiency, and conservativeness over existing methods. 

---
# Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments 

**Authors**: Hansen Jin Lillemark, Benhao Huang, Fangneng Zhan, Yilun Du, Thomas Anderson Keller  

**Link**: [PDF](https://arxiv.org/pdf/2601.01075)  

**Abstract**: Embodied systems experience the world as 'a symphony of flows': a combination of many continuous streams of sensory input coupled to self-motion, interwoven with the dynamics of external objects. These streams obey smooth, time-parameterized symmetries, which combine through a precisely structured algebra; yet most neural network world models ignore this structure and instead repeatedly re-learn the same transformations from data. In this work, we introduce 'Flow Equivariant World Models', a framework in which both self-motion and external object motion are unified as one-parameter Lie group 'flows'. We leverage this unification to implement group equivariance with respect to these transformations, thereby providing a stable latent world representation over hundreds of timesteps. On both 2D and 3D partially observed video world modeling benchmarks, we demonstrate that Flow Equivariant World Models significantly outperform comparable state-of-the-art diffusion-based and memory-augmented world modeling architectures -- particularly when there are predictable world dynamics outside the agent's current field of view. We show that flow equivariance is particularly beneficial for long rollouts, generalizing far beyond the training horizon. By structuring world model representations with respect to internal and external motion, flow equivariance charts a scalable route to data efficient, symmetry-guided, embodied intelligence. Project link: this https URL. 

---
# Gendered Pathways in AI Companionship: Cross-Community Behavior and Toxicity Patterns on Reddit 

**Authors**: Erica Coppolillo, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2601.01073)  

**Abstract**: AI-companionship platforms are rapidly reshaping how people form emotional, romantic, and parasocial bonds with non-human agents, raising new questions about how these relationships intersect with gendered online behavior and exposure to harmful content. Focusing on the MyBoyfriendIsAI (MBIA) subreddit, we reconstruct the Reddit activity histories of more than 3,000 highly engaged users over two years, yielding over 67,000 historical submissions. We then situate MBIA within a broader ecosystem by building a historical interaction network spanning more than 2,000 subreddits, which enables us to trace cross-community pathways and measure how toxicity and emotional expression vary across these trajectories. We find that MBIA users primarily traverse four surrounding community spheres (AI-companionship, porn-related, forum-like, and gaming) and that participation across the ecosystem exhibits a distinct gendered structure, with substantial engagement by female users. While toxicity is generally low across most pathways, we observe localized spikes concentrated in a small subset of AI-porn and gender-oriented communities. Nearly 16% of users engage with gender-focused subreddits, and their trajectories display systematically different patterns of emotional expression and elevated toxicity, suggesting that a minority of gendered pathways may act as toxicity amplifiers within the broader AI-companionship ecosystem. These results characterize the gendered structure of cross-community participation around AI companionship on Reddit and highlight where risks concentrate, informing measurement, moderation, and design practices for human-AI relationship platforms. 

---
# SPoRC-VIST: A Benchmark for Evaluating Generative Natural Narrative in Vision-Language Models 

**Authors**: Yunlin Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.01062)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in descriptive tasks such as image captioning and visual question answering (VQA). However, their ability to generate engaging, long-form narratives -- specifically multi-speaker podcast dialogues -- remains under-explored and difficult to evaluate. Standard metrics like BLEU and ROUGE fail to capture the nuances of conversational naturalness, personality, and narrative flow, often rewarding safe, repetitive outputs over engaging storytelling. In this work, we present a novel pipeline for end-to-end visual podcast generation, and fine-tune a Qwen3-VL-32B model on a curated dataset of 4,000 image-dialogue pairs. Crucially, we use a synthetic-to-real training strategy: we train on high-quality podcast dialogues from the Structured Podcast Research Corpus (SPoRC) paired with synthetically generated imagery, and evaluate on real-world photo sequences from the Visual Storytelling Dataset (VIST). This rigorous setup tests the model's ability to generalize from synthetic training data to real-world visual domains. We propose a comprehensive evaluation framework that moves beyond textual overlap, and use AI-as-a-judge (Gemini 3 Pro, Claude Opus 4.5, GPT 5.2) and novel style metrics (average turn length, speaker switch rate) to assess quality. Our experiments demonstrate that our fine-tuned 32B model significantly outperforms a 235B base model in conversational naturalness ($>$80\% win rate) and narrative depth (+50\% turn length), while maintaining identical visual grounding capabilities (CLIPScore: 20.39). 

---
# A UCB Bandit Algorithm for General ML-Based Estimators 

**Authors**: Yajing Liu, Erkao Bao, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.01061)  

**Abstract**: We present ML-UCB, a generalized upper confidence bound algorithm that integrates arbitrary machine learning models into multi-armed bandit frameworks. A fundamental challenge in deploying sophisticated ML models for sequential decision-making is the lack of tractable concentration inequalities required for principled exploration. We overcome this limitation by directly modeling the learning curve behavior of the underlying estimator. Specifically, assuming the Mean Squared Error decreases as a power law in the number of training samples, we derive a generalized concentration inequality and prove that ML-UCB achieves sublinear regret. This framework enables the principled integration of any ML model whose learning curve can be empirically characterized, eliminating the need for model-specific theoretical analysis. We validate our approach through experiments on a collaborative filtering recommendation system using online matrix factorization with synthetic data designed to simulate a simplified two-tower model, demonstrating substantial improvements over LinUCB 

---
# Enhancing Histopathological Image Classification via Integrated HOG and Deep Features with Robust Noise Performance 

**Authors**: Ifeanyi Ezuma, Ugochukwu Ugwu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01056)  

**Abstract**: The era of digital pathology has advanced histopathological examinations, making automated image analysis essential in clinical practice. This study evaluates the classification performance of machine learning and deep learning models on the LC25000 dataset, which includes five classes of histopathological images. We used the fine-tuned InceptionResNet-v2 network both as a classifier and for feature extraction. Our results show that the fine-tuned InceptionResNet-v2 achieved a classification accuracy of 96.01\% and an average AUC of 96.8\%. Models trained on deep features from InceptionResNet-v2 outperformed those using only the pre-trained network, with the Neural Network model achieving an AUC of 99.99\% and accuracy of 99.84\%. Evaluating model robustness under varying SNR conditions revealed that models using deep features exhibited greater resilience, particularly GBM and KNN. The combination of HOG and deep features showed enhanced performance, however, less so in noisy environments. 

---
# EgoGrasp: World-Space Hand-Object Interaction Estimation from Egocentric Videos 

**Authors**: Hongming Fu, Wenjia Wang, Xiaozhen Qiao, Shuo Yang, Zheng Liu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.01050)  

**Abstract**: We propose EgoGrasp, the first method to reconstruct world-space hand-object interactions (W-HOI) from egocentric monocular videos with dynamic cameras in the wild. Accurate W-HOI reconstruction is critical for understanding human behavior and enabling applications in embodied intelligence and virtual reality. However, existing hand-object interactions (HOI) methods are limited to single images or camera coordinates, failing to model temporal dynamics or consistent global trajectories. Some recent approaches attempt world-space hand estimation but overlook object poses and HOI constraints. Their performance also suffers under severe camera motion and frequent occlusions common in egocentric in-the-wild videos. To address these challenges, we introduce a multi-stage framework with a robust pre-process pipeline built on newly developed spatial intelligence models, a whole-body HOI prior model based on decoupled diffusion models, and a multi-objective test-time optimization paradigm. Our HOI prior model is template-free and scalable to multiple objects. In experiments, we prove our method achieving state-of-the-art performance in W-HOI reconstruction. 

---
# Multi-Dimensional Prompt Chaining to Improve Open-Domain Dialogue Generation 

**Authors**: Livia Leong Hui Teng  

**Link**: [PDF](https://arxiv.org/pdf/2601.01037)  

**Abstract**: Small language models (SLMs) offer significant deployment advantages but often struggle to match the dialogue quality of larger models in open-domain settings. In this paper, we propose a multi-dimensional prompt-chaining framework that integrates Naturalness, Coherence, and Engagingness dimensions to enhance human-likeness in open-domain dialogue generation. We apply the framework to two SLMs, TinyLlama and Llama-2-7B, and benchmark their performance against responses generated by substantially larger models, including Llama-2-70B and GPT-3.5 Turbo. We then employ automatic and human evaluation to assess the responses based on diversity, contextual coherence, as well as overall quality. Results show that the full framework improves response diversity by up to 29%, contextual coherence by up to 28%, and engagingness as well as naturalness by up to 29%. Notably, Llama-2-7B achieves performance comparable to substantially larger models, including Llama-2-70B and GPT-3.5 Turbo. Overall, the findings demonstrate that carefully designed prompt-based strategies provide an effective and resource-efficient pathway to improving open-domain dialogue quality in SLMs. 

---
# Beyond Demand Estimation: Consumer Surplus Evaluation via Cumulative Propensity Weights 

**Authors**: Zeyu Bian, Max Biggs, Ruijiang Gao, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01029)  

**Abstract**: This paper develops a practical framework for using observational data to audit the consumer surplus effects of AI-driven decisions, specifically in targeted pricing and algorithmic lending. Traditional approaches first estimate demand functions and then integrate to compute consumer surplus, but these methods can be challenging to implement in practice due to model misspecification in parametric demand forms and the large data requirements and slow convergence of flexible nonparametric or machine learning approaches. Instead, we exploit the randomness inherent in modern algorithmic pricing, arising from the need to balance exploration and exploitation, and introduce an estimator that avoids explicit estimation and numerical integration of the demand function. Each observed purchase outcome at a randomized price is an unbiased estimate of demand and by carefully reweighting purchase outcomes using novel cumulative propensity weights (CPW), we are able to reconstruct the integral. Building on this idea, we introduce a doubly robust variant named the augmented cumulative propensity weighting (ACPW) estimator that only requires one of either the demand model or the historical pricing policy distribution to be correctly specified. Furthermore, this approach facilitates the use of flexible machine learning methods for estimating consumer surplus, since it achieves fast convergence rates by incorporating an estimate of demand, even when the machine learning estimate has slower convergence rates. Neither of these estimators is a standard application of off-policy evaluation techniques as the target estimand, consumer surplus, is unobserved. To address fairness, we extend this framework to an inequality-aware surplus measure, allowing regulators and firms to quantify the profit-equity trade-off. Finally, we validate our methods through comprehensive numerical studies. 

---
# A Platform for Interactive AI Character Experiences 

**Authors**: Rafael Wampfler, Chen Yang, Dillon Elste, Nikola Kovacevic, Philine Witzig, Markus Gross  

**Link**: [PDF](https://arxiv.org/pdf/2601.01027)  

**Abstract**: From movie characters to modern science fiction - bringing characters into interactive, story-driven conversations has captured imaginations across generations. Achieving this vision is highly challenging and requires much more than just language modeling. It involves numerous complex AI challenges, such as conversational AI, maintaining character integrity, managing personality and emotions, handling knowledge and memory, synthesizing voice, generating animations, enabling real-world interactions, and integration with physical environments. Recent advancements in the development of foundation models, prompt engineering, and fine-tuning for downstream tasks have enabled researchers to address these individual challenges. However, combining these technologies for interactive characters remains an open problem. We present a system and platform for conveniently designing believable digital characters, enabling a conversational and story-driven experience while providing solutions to all of the technical challenges. As a proof-of-concept, we introduce Digital Einstein, which allows users to engage in conversations with a digital representation of Albert Einstein about his life, research, and persona. While Digital Einstein exemplifies our methods for a specific character, our system is flexible and generalizes to any story-driven or conversational character. By unifying these diverse AI components into a single, easy-to-adapt platform, our work paves the way for immersive character experiences, turning the dream of lifelike, story-based interactions into a reality. 

---
# Enhanced Leukemic Cell Classification Using Attention-Based CNN and Data Augmentation 

**Authors**: Douglas Costa Braga, Daniel Oliveira Dantas  

**Link**: [PDF](https://arxiv.org/pdf/2601.01026)  

**Abstract**: We present a reproducible deep learning pipeline for leukemic cell classification, focusing on system architecture, experimental robustness, and software design choices for medical image analysis. Acute lymphoblastic leukemia (ALL) is the most common childhood cancer, requiring expert microscopic diagnosis that suffers from inter-observer variability and time constraints. The proposed system integrates an attention-based convolutional neural network combining EfficientNetV2-B3 with Squeeze-and-Excitation mechanisms for automated ALL cell classification. Our approach employs comprehensive data augmentation, focal loss for class imbalance, and patient-wise data splitting to ensure robust and reproducible evaluation. On the C-NMC 2019 dataset (12,528 original images from 62 patients), the system achieves a 97.89% F1-score and 97.89% accuracy on the test set, with statistical validation through 100-iteration Monte Carlo experiments confirming significant improvements (p < 0.001) over baseline methods. The proposed pipeline outperforms existing approaches by up to 4.67% while using 89% fewer parameters than VGG16 (15.2M vs. 138M). The attention mechanism provides interpretable visualizations of diagnostically relevant cellular features, demonstrating that modern attention-based architectures can improve leukemic cell classification while maintaining computational efficiency suitable for clinical deployment. 

---
# ITSELF: Attention Guided Fine-Grained Alignment for Vision-Language Retrieval 

**Authors**: Tien-Huy Nguyen, Huu-Loc Tran, Thanh Duc Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2601.01024)  

**Abstract**: Vision Language Models (VLMs) have rapidly advanced and show strong promise for text-based person search (TBPS), a task that requires capturing fine-grained relationships between images and text to distinguish individuals. Previous methods address these challenges through local alignment, yet they are often prone to shortcut learning and spurious correlations, yielding misalignment. Moreover, injecting prior knowledge can distort intra-modality structure. Motivated by our finding that encoder attention surfaces spatially precise evidence from the earliest training epochs, and to alleviate these issues, we introduceITSELF, an attention-guided framework for implicit local alignment. At its core, Guided Representation with Attentive Bank (GRAB) converts the model's own attention into an Attentive Bank of high-saliency tokens and applies local objectives on this bank, learning fine-grained correspondences without extra supervision. To make the selection reliable and non-redundant, we introduce Multi-Layer Attention for Robust Selection (MARS), which aggregates attention across layers and performs diversity-aware top-k selection; and Adaptive Token Scheduler (ATS), which schedules the retention budget from coarse to fine over training, preserving context early while progressively focusing on discriminative details. Extensive experiments on three widely used TBPS benchmarks showstate-of-the-art performance and strong cross-dataset generalization, confirming the effectiveness and robustness of our approach without additional prior supervision. Our project is publicly available at this https URL 

---
# Decoupling Amplitude and Phase Attention in Frequency Domain for RGB-Event based Visual Object Tracking 

**Authors**: Shiao Wang, Xiao Wang, Haonan Zhao, Jiarui Xu, Bo Jiang, Lin Zhu, Xin Zhao, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01022)  

**Abstract**: Existing RGB-Event visual object tracking approaches primarily rely on conventional feature-level fusion, failing to fully exploit the unique advantages of event cameras. In particular, the high dynamic range and motion-sensitive nature of event cameras are often overlooked, while low-information regions are processed uniformly, leading to unnecessary computational overhead for the backbone network. To address these issues, we propose a novel tracking framework that performs early fusion in the frequency domain, enabling effective aggregation of high-frequency information from the event modality. Specifically, RGB and event modalities are transformed from the spatial domain to the frequency domain via the Fast Fourier Transform, with their amplitude and phase components decoupled. High-frequency event information is selectively fused into RGB modality through amplitude and phase attention, enhancing feature representation while substantially reducing backbone computation. In addition, a motion-guided spatial sparsification module leverages the motion-sensitive nature of event cameras to capture the relationship between target motion cues and spatial probability distribution, filtering out low-information regions and enhancing target-relevant features. Finally, a sparse set of target-relevant features is fed into the backbone network for learning, and the tracking head predicts the final target position. Extensive experiments on three widely used RGB-Event tracking benchmark datasets, including FE108, FELT, and COESOT, demonstrate the high performance and efficiency of our method. The source code of this paper will be released on this https URL 

---
# Improving Variational Autoencoder using Random Fourier Transformation: An Aviation Safety Anomaly Detection Case-Study 

**Authors**: Ata Akbari Asanjan, Milad Memarzadeh, Bryan Matthews, Nikunj Oza  

**Link**: [PDF](https://arxiv.org/pdf/2601.01016)  

**Abstract**: In this study, we focus on the training process and inference improvements of deep neural networks (DNNs), specifically Autoencoders (AEs) and Variational Autoencoders (VAEs), using Random Fourier Transformation (RFT). We further explore the role of RFT in model training behavior using Frequency Principle (F-Principle) analysis and show that models with RFT turn to learn low frequency and high frequency at the same time, whereas conventional DNNs start from low frequency and gradually learn (if successful) high-frequency features. We focus on reconstruction-based anomaly detection using autoencoder and variational autoencoder and investigate the RFT's role. We also introduced a trainable variant of RFT that uses the existing computation graph to train the expansion of RFT instead of it being random. We showcase our findings with two low-dimensional synthetic datasets for data representation, and an aviation safety dataset, called Dashlink, for high-dimensional reconstruction-based anomaly detection. The results indicate the superiority of models with Fourier transformation compared to the conventional counterpart and remain inconclusive regarding the benefits of using trainable Fourier transformation in contrast to the Random variant. 

---
# Geometric and Dynamic Scaling in Deep Transformers 

**Authors**: Haoran Su, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2601.01014)  

**Abstract**: Despite their empirical success, pushing Transformer architectures to extreme depth often leads to a paradoxical failure: representations become increasingly redundant, lose rank, and ultimately collapse. Existing explanations largely attribute this phenomenon to optimization instability or vanishing gradients, yet such accounts fail to explain why collapse persists even under modern normalization and initialization schemes. In this paper, we argue that the collapse of deep Transformers is fundamentally a geometric problem. Standard residual updates implicitly assume that feature accumulation is always beneficial, but offer no mechanism to constrain update directions or to erase outdated information. As depth increases, this leads to systematic drift off the semantic manifold and monotonic feature accumulation, causing representational degeneracy. We propose a unified geometric framework that addresses these failures through two orthogonal principles. First, manifold-constrained hyper-connections restrict residual updates to valid local tangent directions, preventing uncontrolled manifold drift. Second, deep delta learning introduces data-dependent, non-monotonic updates that enable reflection and erasure of redundant features rather than their unconditional accumulation. Together, these mechanisms decouple the direction and sign of feature updates, yielding a stable geometric evolution across depth. We term the resulting architecture the Manifold-Geometric Transformer (MGT). Our analysis predicts that enforcing geometric validity while allowing dynamic erasure is essential for avoiding rank collapse in ultra-deep networks. We outline an evaluation protocol for Transformers exceeding 100 layers to test the hypothesis that geometry, rather than depth itself, is the key limiting factor in deep representation learning. 

---
# Intention Collapse: Intention-Level Metrics for Reasoning in Language Models 

**Authors**: Patricio Vera  

**Link**: [PDF](https://arxiv.org/pdf/2601.01011)  

**Abstract**: Every act of language generation compresses a rich internal state into a single token sequence. We call this process intention collapse: a many-to-one projection from a high dimensional intention space I into an external language space L. We formalize intention collapse for contemporary language models, define three simple, model agnostic intention metrics (intention entropy Hint, effective dimensionality dimeff, and latent knowledge recoverability Recov), and propose an empirical agenda for studying how inference time computation shapes internal intentions before they are verbalized. We also report a first small scale experiment. Using a 4 bit Mistral 7B model on 200 GSM8K problems, we compare a direct answer baseline, a chain of thought (CoT) regime, and a babble control. CoT raises accuracy from 5.5 percent to 53 percent, sharply reduces pre collapse intention entropy (from 1.42 to 0.37 bits), and shows higher global effective dimensionality than the other regimes despite producing fewer tokens than babble. At the same time, Hint has little item level predictive power, and a linear probe on I achieves AUROC 0.65 in the CoT regime but only about chance in the baseline regime, where it collapses to the majority class. These preliminary results indicate that intention level metrics can distinguish inference regimes and expose latent information that is partly lost during collapse, while also revealing important limitations of our current proxies 

---
# Data-Driven Assessment of Concrete Mixture Compositions on Chloride Transport via Standalone Machine Learning Algorithms 

**Authors**: Mojtaba Aliasghar-Mamaghani, Mohammadreza Khalafi  

**Link**: [PDF](https://arxiv.org/pdf/2601.01009)  

**Abstract**: This paper employs a data-driven approach to determine the impact of concrete mixture compositions on the temporal evolution of chloride in concrete structures. This is critical for assessing the service life of civil infrastructure subjected to aggressive environments. The adopted methodology relies on several simple and complex standalone machine learning (ML) algorithms, with the primary objective of establishing confidence in the unbiased prediction of the underlying hidden correlations. The simple algorithms include linear regression (LR), k-nearest neighbors (KNN) regression, and kernel ridge regression (KRR). The complex algorithms entail support vector regression (SVR), Gaussian process regression (GPR), and two families of artificial neural networks, including a feedforward network (multilayer perceptron, MLP) and a gated recurrent unit (GRU). The MLP architecture cannot explicitly handle sequential data, a limitation addressed by the GRU. A comprehensive dataset is considered. The performance of ML algorithms is evaluated, with KRR, GPR, and MLP exhibiting high accuracy. Given the diversity of the adopted concrete mixture proportions, the GRU was unable to accurately reproduce the response in the test set. Further analyses elucidate the contributions of mixture compositions to the temporal evolution of chloride. The results obtained from the GPR model unravel latent correlations through clear and explainable trends. The MLP, SVR, and KRR also provide acceptable estimates of the overall trends. The majority of mixture components exhibit an inverse relation with chloride content, while a few components demonstrate a direct correlation. These findings highlight the potential of surrogate approaches for describing the physical processes involved in chloride ingress and the associated correlations, toward the ultimate goal of enhancing the service life of civil infrastructure. 

---
# An Explainable Agentic AI Framework for Uncertainty-Aware and Abstention-Enabled Acute Ischemic Stroke Imaging Decisions 

**Authors**: Md Rashadul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2601.01008)  

**Abstract**: Artificial intelligence models have shown strong potential in acute ischemic stroke imaging, particularly for lesion detection and segmentation using computed tomography and magnetic resonance imaging. However, most existing approaches operate as black box predictors, producing deterministic outputs without explicit uncertainty awareness or structured mechanisms to abstain under ambiguous conditions. This limitation raises serious safety and trust concerns in high risk emergency radiology settings. In this paper, we propose an explainable agentic AI framework for uncertainty aware and abstention enabled decision support in acute ischemic stroke imaging. The framework follows a modular agentic pipeline in which a perception agent performs lesion aware image analysis, an uncertainty estimation agent computes slice level predictive reliability, and a decision agent determines whether to issue a prediction or abstain based on predefined uncertainty thresholds. Unlike prior stroke imaging systems that primarily focus on improving segmentation or classification accuracy, the proposed framework explicitly prioritizes clinical safety, transparency, and clinician aligned decision behavior. Qualitative and case based analyses across representative stroke imaging scenarios demonstrate that uncertainty driven abstention naturally emerges in diagnostically ambiguous regions and low information slices. The framework further integrates visual explanation mechanisms to support both predictive and abstention decisions, addressing a key limitation of existing uncertainty aware medical imaging systems. Rather than introducing a new performance benchmark, this work presents agentic control, uncertainty awareness, and selective abstention as essential design principles for developing safe and trustworthy medical imaging AI systems. 

---
# Scale-aware Adaptive Supervised Network with Limited Medical Annotations 

**Authors**: Zihan Li, Dandan Shan, Yunxiang Li, Paul E. Kinahan, Qingqi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2601.01005)  

**Abstract**: Medical image segmentation faces critical challenges in semi-supervised learning scenarios due to severe annotation scarcity requiring expert radiological knowledge, significant inter-annotator variability across different viewpoints and expertise levels, and inadequate multi-scale feature integration for precise boundary delineation in complex anatomical structures. Existing semi-supervised methods demonstrate substantial performance degradation compared to fully supervised approaches, particularly in small target segmentation and boundary refinement tasks. To address these fundamental challenges, we propose SASNet (Scale-aware Adaptive Supervised Network), a dual-branch architecture that leverages both low-level and high-level feature representations through novel scale-aware adaptive reweight mechanisms. Our approach introduces three key methodological innovations, including the Scale-aware Adaptive Reweight strategy that dynamically weights pixel-wise predictions using temporal confidence accumulation, the View Variance Enhancement mechanism employing 3D Fourier domain transformations to simulate annotation variability, and segmentation-regression consistency learning through signed distance map algorithms for enhanced boundary precision. These innovations collectively address the core limitations of existing semi-supervised approaches by integrating spatial, temporal, and geometric consistency principles within a unified optimization framework. Comprehensive evaluation across LA, Pancreas-CT, and BraTS datasets demonstrates that SASNet achieves superior performance with limited labeled data, surpassing state-of-the-art semi-supervised methods while approaching fully supervised performance levels. The source code for SASNet is available at this https URL. 

---
# VEAT Quantifies Implicit Associations in Text-to-Video Generator Sora and Reveals Challenges in Bias Mitigation 

**Authors**: Yongxu Sun, Michael Saxon, Ian Yang, Anna-Maria Gueorguieva, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2601.00996)  

**Abstract**: Text-to-Video (T2V) generators such as Sora raise concerns about whether generated content reflects societal bias. We extend embedding-association tests from words and images to video by introducing the Video Embedding Association Test (VEAT) and Single-Category VEAT (SC-VEAT). We validate these methods by reproducing the direction and magnitude of associations from widely used baselines, including Implicit Association Test (IAT) scenarios and OASIS image categories. We then quantify race (African American vs. European American) and gender (women vs. men) associations with valence (pleasant vs. unpleasant) across 17 occupations and 7 awards. Sora videos associate European Americans and women more with pleasantness (both d>0.8). Effect sizes correlate with real-world demographic distributions: percent men and White in occupations (r=0.93, r=0.83) and percent male and non-Black among award recipients (r=0.88, r=0.99). Applying explicit debiasing prompts generally reduces effect-size magnitudes, but can backfire: two Black-associated occupations (janitor, postal service) become more Black-associated after debiasing. Together, these results reveal that easily accessible T2V generators can actually amplify representational harms if not rigorously evaluated and responsibly deployed. 

---
# WildIng: A Wildlife Image Invariant Representation Model for Geographical Domain Shift 

**Authors**: Julian D. Santamaria, Claudia Isaza, Jhony H. Giraldo  

**Link**: [PDF](https://arxiv.org/pdf/2601.00993)  

**Abstract**: Wildlife monitoring is crucial for studying biodiversity loss and climate change. Camera trap images provide a non-intrusive method for analyzing animal populations and identifying ecological patterns over time. However, manual analysis is time-consuming and resource-intensive. Deep learning, particularly foundation models, has been applied to automate wildlife identification, achieving strong performance when tested on data from the same geographical locations as their training sets. Yet, despite their promise, these models struggle to generalize to new geographical areas, leading to significant performance drops. For example, training an advanced vision-language model, such as CLIP with an adapter, on an African dataset achieves an accuracy of 84.77%. However, this performance drops significantly to 16.17% when the model is tested on an American dataset. This limitation partly arises because existing models rely predominantly on image-based representations, making them sensitive to geographical data distribution shifts, such as variation in background, lighting, and environmental conditions. To address this, we introduce WildIng, a Wildlife image Invariant representation model for geographical domain shift. WildIng integrates text descriptions with image features, creating a more robust representation to geographical domain shifts. By leveraging textual descriptions, our approach captures consistent semantic information, such as detailed descriptions of the appearance of the species, improving generalization across different geographical locations. Experiments show that WildIng enhances the accuracy of foundation models such as BioCLIP by 30% under geographical domain shift conditions. We evaluate WildIng on two datasets collected from different regions, namely America and Africa. The code and models are publicly available at this https URL. 

---
# Value Vision-Language-Action Planning & Search 

**Authors**: Ali Salamatian, Kieran Pattison, Cyrus Neary  

**Link**: [PDF](https://arxiv.org/pdf/2601.00969)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as powerful generalist policies for robotic manipulation, yet they remain fundamentally limited by their reliance on behavior cloning, leading to brittleness under distribution shift. While augmenting pretrained models with test-time search algorithms like Monte Carlo Tree Search (MCTS) can mitigate these failures, existing formulations rely solely on the VLA prior for guidance, lacking a grounded estimate of expected future return. Consequently, when the prior is inaccurate, the planner can only correct action selection via the exploration term, which requires extensive simulation to become effective. To address this limitation, we introduce Value Vision-Language-Action Planning and Search (V-VLAPS), a framework that augments MCTS with a lightweight, learnable value function. By training a simple multilayer perceptron (MLP) on the latent representations of a fixed VLA backbone (Octo), we provide the search with an explicit success signal that biases action selection toward high-value regions. We evaluate V-VLAPS on the LIBERO robotic manipulation suite, demonstrating that our value-guided search improves success rates by over 5 percentage points while reducing the average number of MCTS simulations by 5-15 percent compared to baselines that rely only on the VLA prior. 

---
# Adapting Feature Attenuation to NLP 

**Authors**: Tianshuo Yang, Ryan Rabinowitz, Terrance E. Boult, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2601.00965)  

**Abstract**: Transformer classifiers such as BERT deliver impressive closed-set accuracy, yet they remain brittle when confronted with inputs from unseen categories--a common scenario for deployed NLP systems. We investigate Open-Set Recognition (OSR) for text by porting the feature attenuation hypothesis from computer vision to transformers and by benchmarking it against state-of-the-art baselines. Concretely, we adapt the COSTARR framework--originally designed for classification in computer vision--to two modest language models (BERT (base) and GPT-2) trained to label 176 arXiv subject areas. Alongside COSTARR, we evaluate Maximum Softmax Probability (MSP), MaxLogit, and the temperature-scaled free-energy score under the OOSA and AUOSCR metrics. Our results show (i) COSTARR extends to NLP without retraining but yields no statistically significant gain over MaxLogit or MSP, and (ii) free-energy lags behind all other scores in this high-class-count setting. The study highlights both the promise and the current limitations of transplanting vision-centric OSR ideas to language models, and points toward the need for larger backbones and task-tailored attenuation strategies. 

---
# Comparative Analysis of Formula and Structure Prediction from Tandem Mass Spectra 

**Authors**: Xujun Che, Xiuxia Du, Depeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.00941)  

**Abstract**: Liquid chromatography mass spectrometry (LC-MS)-based metabolomics and exposomics aim to measure detectable small molecules in biological samples. The results facilitate hypothesis-generating discovery of metabolic changes and disease mechanisms and provide information about environmental exposures and their effects on human health. Metabolomics and exposomics are made possible by the high resolving power of LC and high mass measurement accuracy of MS. However, a majority of the signals from such studies still cannot be identified or annotated using conventional library searching because existing spectral libraries are far from covering the vast chemical space captured by LC-MS/MS. To address this challenge and unleash the full potential of metabolomics and exposomics, a number of computational approaches have been developed to predict compounds based on tandem mass spectra. Published assessment of these approaches used different datasets and evaluation. To select prediction workflows for practical applications and identify areas for further improvements, we have carried out a systematic evaluation of the state-of-the-art prediction algorithms. Specifically, the accuracy of formula prediction and structure prediction was evaluated for different types of adducts. The resulting findings have established realistic performance baselines, identified critical bottlenecks, and provided guidance to further improve compound predictions based on MS. 

---
# Emoji-Based Jailbreaking of Large Language Models 

**Authors**: M P V S Gopinadh, S Mahaboob Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2601.00936)  

**Abstract**: Large Language Models (LLMs) are integral to modern AI applications, but their safety alignment mechanisms can be bypassed through adversarial prompt engineering. This study investigates emoji-based jailbreaking, where emoji sequences are embedded in textual prompts to trigger harmful and unethical outputs from LLMs. We evaluated 50 emoji-based prompts on four open-source LLMs: Mistral 7B, Qwen 2 7B, Gemma 2 9B, and Llama 3 8B. Metrics included jailbreak success rate, safety alignment adherence, and latency, with responses categorized as successful, partial and failed. Results revealed model-specific vulnerabilities: Gemma 2 9B and Mistral 7B exhibited 10 % success rates, while Qwen 2 7B achieved full alignment (0% success). A chi-square test (chi^2 = 32.94, p < 0.001) confirmed significant inter-model differences. While prior works focused on emoji attacks targeting safety judges or classifiers, our empirical analysis examines direct prompt-level vulnerabilities in LLMs. The results reveal limitations in safety mechanisms and highlight the necessity for systematic handling of emoji-based representations in prompt-level safety and alignment pipelines. 

---
# Improving Code-Switching Speech Recognition with TTS Data Augmentation 

**Authors**: Yue Heng Yeo, Yuchen Hu, Shreyas Gopal, Yizhou Peng, Hexin Liu, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2601.00935)  

**Abstract**: Automatic speech recognition (ASR) for conversational code-switching speech remains challenging due to the scarcity of realistic, high-quality labeled speech data. This paper explores multilingual text-to-speech (TTS) models as an effective data augmentation technique to address this shortage. Specifically, we fine-tune the multilingual CosyVoice2 TTS model on the SEAME dataset to generate synthetic conversational Chinese-English code-switching speech, significantly increasing the quantity and speaker diversity of available training data. Our experiments demonstrate that augmenting real speech with synthetic speech reduces the mixed error rate (MER) from 12.1 percent to 10.1 percent on DevMan and from 17.8 percent to 16.0 percent on DevSGE, indicating consistent performance gains. These results confirm that multilingual TTS is an effective and practical tool for enhancing ASR robustness in low-resource conversational code-switching scenarios. 

---
# LOFA: Online Influence Maximization under Full-Bandit Feedback using Lazy Forward Selection 

**Authors**: Jinyu Xu, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.00933)  

**Abstract**: We study the problem of influence maximization (IM) in an online setting, where the goal is to select a subset of nodes$\unicode{x2014}$called the seed set$\unicode{x2014}$at each time step over a fixed time horizon, subject to a cardinality budget constraint, to maximize the expected cumulative influence. We operate under a full-bandit feedback model, where only the influence of the chosen seed set at each time step is observed, with no additional structural information about the network or diffusion process. It is well-established that the influence function is submodular, and existing algorithms exploit this property to achieve low regret. In this work, we leverage this property further and propose the Lazy Online Forward Algorithm (LOFA), which achieves a lower empirical regret. We conduct experiments on a real-world social network to demonstrate that LOFA achieves superior performance compared to existing bandit algorithms in terms of cumulative regret and instantaneous reward. 

---
# AlignUSER: Human-Aligned LLM Agents via World Models for Recommender System Evaluation 

**Authors**: Nicolas Bougie, Gian Maria Marconi, Tony Yip, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2601.00930)  

**Abstract**: Evaluating recommender systems remains challenging due to the gap between offline metrics and real user behavior, as well as the scarcity of interaction data. Recent work explores large language model (LLM) agents as synthetic users, yet they typically rely on few-shot prompting, which yields a shallow understanding of the environment and limits their ability to faithfully reproduce user actions. We introduce AlignUSER, a framework that learns world-model-driven agents from human interactions. Given rollout sequences of actions and states, we formalize world modeling as a next state prediction task that helps the agent internalize the environment. To align actions with human personas, we generate counterfactual trajectories around demonstrations and prompt the LLM to compare its decisions with human choices, identify suboptimal actions, and extract lessons. The learned policy is then used to drive agent interactions with the recommender system. We evaluate AlignUSER across multiple datasets and demonstrate closer alignment with genuine humans than prior work, both at the micro and macro levels. 

---
# Analyzing the Shopping Journey: Computing Shelf Browsing Visits in a Physical Retail Store 

**Authors**: Luis Yoichi Morales, Francesco Zanlungo, David M. Woollard  

**Link**: [PDF](https://arxiv.org/pdf/2601.00928)  

**Abstract**: Motivated by recent challenges in the deployment of robots into customer-facing roles within retail, this work introduces a study of customer activity in physical stores as a step toward autonomous understanding of shopper intent. We introduce an algorithm that computes shoppers' ``shelf visits'' -- capturing their browsing behavior in the store. Shelf visits are extracted from trajectories obtained via machine vision-based 3D tracking and overhead cameras. We perform two independent calibrations of the shelf visit algorithm, using distinct sets of trajectories (consisting of 8138 and 15129 trajectories), collected in different stores and labeled by human reviewers. The calibrated models are then evaluated on trajectories held out of the calibration process both from the same store on which calibration was performed and from the other store. An analysis of the results shows that the algorithm can recognize customers' browsing activity when evaluated in an environment different from the one on which calibration was performed. We then use the model to analyze the customers' ``browsing patterns'' on a large set of trajectories and their relation to actual purchases in the stores. Finally, we discuss how shelf browsing information could be used for retail planning and in the domain of human-robot interaction scenarios. 

---
# Measuring Social Media Polarization Using Large Language Models and Heuristic Rules 

**Authors**: Jawad Chowdhury, Rezaur Rashid, Gabriel Terejanu  

**Link**: [PDF](https://arxiv.org/pdf/2601.00927)  

**Abstract**: Understanding affective polarization in online discourse is crucial for evaluating the societal impact of social media interactions. This study presents a novel framework that leverages large language models (LLMs) and domain-informed heuristics to systematically analyze and quantify affective polarization in discussions on divisive topics such as climate change and gun control. Unlike most prior approaches that relied on sentiment analysis or predefined classifiers, our method integrates LLMs to extract stance, affective tone, and agreement patterns from large-scale social media discussions. We then apply a rule-based scoring system capable of quantifying affective polarization even in small conversations consisting of single interactions, based on stance alignment, emotional content, and interaction dynamics. Our analysis reveals distinct polarization patterns that are event dependent: (i) anticipation-driven polarization, where extreme polarization escalates before well-publicized events, and (ii) reactive polarization, where intense affective polarization spikes immediately after sudden, high-impact events. By combining AI-driven content annotation with domain-informed scoring, our framework offers a scalable and interpretable approach to measuring affective polarization. The source code is publicly available at: this https URL. 

---
# MACA: A Framework for Distilling Trustworthy LLMs into Efficient Retrievers 

**Authors**: Satya Swaroop Gudipudi, Sahil Girhepuje, Ponnurangam Kumaraguru, Kristine Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.00926)  

**Abstract**: Modern enterprise retrieval systems must handle short, underspecified queries such as ``foreign transaction fee refund'' and ``recent check status''. In these cases, semantic nuance and metadata matter but per-query large language model (LLM) re-ranking and manual labeling are costly. We present Metadata-Aware Cross-Model Alignment (MACA), which distills a calibrated metadata aware LLM re-ranker into a compact student retriever, avoiding online LLM calls. A metadata-aware prompt verifies the teacher's trustworthiness by checking consistency under permutations and robustness to paraphrases, then supplies listwise scores, hard negatives, and calibrated relevance margins. The student trains with MACA's MetaFusion objective, which combines a metadata conditioned ranking loss with a cross model margin loss so it learns to push the correct answer above semantically similar candidates with mismatched topic, sub-topic, or entity. On a proprietary consumer banking FAQ corpus and BankFAQs, the MACA teacher surpasses a MAFA baseline at Accuracy@1 by five points on the proprietary set and three points on BankFAQs. MACA students substantially outperform pretrained encoders; e.g., on the proprietary corpus MiniLM Accuracy@1 improves from 0.23 to 0.48, while keeping inference free of LLM calls and supporting retrieval-augmented generation. 

---
# Application of deep learning techniques in non-contrast computed tomography pulmonary angiogram for pulmonary embolism diagnosis 

**Authors**: I-Hsien Ting, Yi-Jun Tseng, Yu-Sheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.00925)  

**Abstract**: Pulmonary embolism is a life-threatening disease, early detection and treatment can significantly reduce mortality. In recent years, many studies have been using deep learning in the diagnosis of pulmonary embolism with contrast medium computed tomography pulmonary angiography, but the contrast medium is likely to cause acute kidney injury in patients with pulmonary embolism and chronic kidney disease, and the contrast medium takes time to work, patients with acute pulmonary embolism may miss the golden treatment time.
This study aims to use deep learning techniques to automatically classify pulmonary embolism in CT images without contrast medium by using a 3D convolutional neural network model. The deep learning model used in this study had a significant impact on the pulmonary embolism classification of computed tomography images without contrast with 85\% accuracy and 0.84 AUC, which confirms the feasibility of the model in the diagnosis of pulmonary embolism. 

---
# Complexity-based code embeddings 

**Authors**: Rares Folea, Radu Iacob, Emil Slusanschi, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2601.00924)  

**Abstract**: This paper presents a generic method for transforming the source code of various algorithms to numerical embeddings, by dynamically analysing the behaviour of computer programs against different inputs and by tailoring multiple generic complexity functions for the analysed metrics. The used algorithms embeddings are based on r-Complexity . Using the proposed code embeddings, we present an implementation of the XGBoost algorithm that achieves an average F1-score on a multi-label dataset with 11 classes, built using real-world code snippets submitted for programming competitions on the Codeforces platform. 

---
# Practical Geometric and Quantum Kernel Methods for Predicting Skeletal Muscle Outcomes in chronic obstructive pulmonary disease 

**Authors**: Azadeh Alavi, Hamidreza Khalili, Stanley H. Chan, Fatemeh Kouchmeshki, Ross Vlahos  

**Link**: [PDF](https://arxiv.org/pdf/2601.00921)  

**Abstract**: Skeletal muscle dysfunction is a clinically relevant extra-pulmonary manifestation of chronic obstructive pulmonary disease (COPD) and is closely linked to systemic and airway inflammation. This motivates predictive modelling of muscle outcomes from minimally invasive biomarkers that can be acquired longitudinally. We study a small-sample preclinical dataset comprising 213 animals across two conditions (Sham versus cigarette-smoke exposure), with blood and bronchoalveolar lavage fluid measurements and three continuous targets: tibialis anterior muscle weight (milligram: mg), specific force (millinewton: mN), and a derived muscle quality index (mN per mg). We benchmark tuned classical baselines, geometry-aware symmetric positive definite (SPD) descriptors with Stein divergence, and quantum kernel models designed for low-dimensional tabular data. In the muscle-weight setting, quantum kernel ridge regression using four interpretable inputs (blood C-reactive protein, neutrophil count, bronchoalveolar lavage cellularity, and condition) attains a test root mean squared error of 4.41 mg and coefficient of determination of 0.605, improving over a matched ridge baseline on the same feature set (4.70 mg and 0.553). Geometry-informed Stein-divergence prototype distances yield a smaller but consistent gain in the biomarker-only setting (4.55 mg versus 4.79 mg). Screening-style evaluation, obtained by thresholding the continuous outcome at 0.8 times the training Sham mean, achieves an area under the receiver operating characteristic curve (ROC-AUC) of up to 0.90 for detecting low muscle weight. These results indicate that geometric and quantum kernel lifts can provide measurable benefits in low-data, low-feature biomedical prediction problems, while preserving interpretability and transparent model selection. 

---
# MODE: Efficient Time Series Prediction with Mamba Enhanced by Low-Rank Neural ODEs 

**Authors**: Xingsheng Chen, Regina Zhang, Bo Gao, Xingwei He, Xiaofeng Liu, Pietro Lio, Kwok-Yan Lam, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.00920)  

**Abstract**: Time series prediction plays a pivotal role across diverse domains such as finance, healthcare, energy systems, and environmental modeling. However, existing approaches often struggle to balance efficiency, scalability, and accuracy, particularly when handling long-range dependencies and irregularly sampled data. To address these challenges, we propose MODE, a unified framework that integrates Low-Rank Neural Ordinary Differential Equations (Neural ODEs) with an Enhanced Mamba architecture. As illustrated in our framework, the input sequence is first transformed by a Linear Tokenization Layer and then processed through multiple Mamba Encoder blocks, each equipped with an Enhanced Mamba Layer that employs Causal Convolution, SiLU activation, and a Low-Rank Neural ODE enhancement to efficiently capture temporal dynamics. This low-rank formulation reduces computational overhead while maintaining expressive power. Furthermore, a segmented selective scanning mechanism, inspired by pseudo-ODE dynamics, adaptively focuses on salient subsequences to improve scalability and long-range sequence modeling. Extensive experiments on benchmark datasets demonstrate that MODE surpasses existing baselines in both predictive accuracy and computational efficiency. Overall, our contributions include: (1) a unified and efficient architecture for long-term time series modeling, (2) integration of Mamba's selective scanning with low-rank Neural ODEs for enhanced temporal representation, and (3) substantial improvements in efficiency and scalability enabled by low-rank approximation and dynamic selective scanning. 

---
# Attention Needs to Focus: A Unified Perspective on Attention Allocation 

**Authors**: Zichuan Fu, Wentao Song, Guojing Li, Yejing Wang, Xian Wu, Yimin Deng, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.00919)  

**Abstract**: The Transformer architecture, a cornerstone of modern Large Language Models (LLMs), has achieved extraordinary success in sequence modeling, primarily due to its attention mechanism. However, despite its power, the standard attention mechanism is plagued by well-documented issues: representational collapse and attention sink. Although prior work has proposed approaches for these issues, they are often studied in isolation, obscuring their deeper connection. In this paper, we present a unified perspective, arguing that both can be traced to a common root -- improper attention allocation. We identify two failure modes: 1) Attention Overload, where tokens receive comparable high weights, blurring semantic features that lead to representational collapse; 2) Attention Underload, where no token is semantically relevant, yet attention is still forced to distribute, resulting in spurious focus such as attention sink. Building on this insight, we introduce Lazy Attention, a novel mechanism designed for a more focused attention distribution. To mitigate overload, it employs positional discrimination across both heads and dimensions to sharpen token distinctions. To counteract underload, it incorporates Elastic-Softmax, a modified normalization function that relaxes the standard softmax constraint to suppress attention on irrelevant tokens. Experiments on the FineWeb-Edu corpus, evaluated across nine diverse benchmarks, demonstrate that Lazy Attention successfully mitigates attention sink and achieves competitive performance compared to both standard attention and modern architectures, while reaching up to 59.58% attention sparsity. 

---
# The Discovery Gap: How Product Hunt Startups Vanish in LLM Organic Discovery Queries 

**Authors**: Amit Prakash Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.00912)  

**Abstract**: When someone asks ChatGPT to recommend a project management tool, which products show up in the response? And more importantly for startup founders: will their newly launched product ever appear? This research set out to answer these questions.
I randomly selected 112 startups from the top 500 products featured on the 2025 Product Hunt leaderboard and tested each one across 2,240 queries to two different large language models: ChatGPT (gpt-4o-mini) and Perplexity (sonar with web search).
The results were striking. When users asked about products by name, both LLMs recognized them almost perfectly: 99.4% for ChatGPT and 94.3% for Perplexity. But when users asked discovery-style questions like "What are the best AI tools launched this year?" the success rates collapsed to 3.32% and 8.29% respectively. That's a gap of 30-to-1 for ChatGPT.
Perhaps the most surprising finding was that Generative Engine Optimization (GEO), the practice of optimizing website content for AI visibility, showed no correlation with actual discovery rates. Products with high GEO scores were no more likely to appear in organic queries than products with low scores.
What did matter? For Perplexity, traditional SEO signals like referring domains (r = +0.319, p < 0.001) and Product Hunt ranking (r = -0.286, p = 0.002) predicted visibility. After cleaning the Reddit data for false positives, community presence also emerged as significant (r = +0.395, p = 0.002).
The practical takeaway is counterintuitive: don't optimize for AI discovery directly. Instead, build the SEO foundation first and LLM visibility will follow. 

---
# Device-Native Autonomous Agents for Privacy-Preserving Negotiations 

**Authors**: Joyjit Roy  

**Link**: [PDF](https://arxiv.org/pdf/2601.00911)  

**Abstract**: Automated negotiations in insurance and business-to-business (B2B) commerce encounter substantial challenges. Current systems force a trade-off between convenience and privacy by routing sensitive financial data through centralized servers, increasing security risks, and diminishing user trust. This study introduces a device-native autonomous Artificial Intelligence (AI) agent system for privacy-preserving negotiations. The proposed system operates exclusively on user hardware, enabling real-time bargaining while maintaining sensitive constraints locally. It integrates zero-knowledge proofs to ensure privacy and employs distilled world models to support advanced on-device reasoning. The architecture incorporates six technical components within an agentic AI workflow. Agents autonomously plan negotiation strategies, conduct secure multi-party bargaining, and generate cryptographic audit trails without exposing user data to external servers. The system is evaluated in insurance and B2B procurement scenarios across diverse device configurations. Results show an average success rate of 87%, a 2.4x latency improvement over cloud baselines, and strong privacy preservation through zero-knowledge proofs. User studies show 27% higher trust scores when decision trails are available. These findings establish a foundation for trustworthy autonomous agents in privacy-sensitive financial domains. 

---
# Conformal Prediction Under Distribution Shift: A COVID-19 Natural Experiment 

**Authors**: Chorok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.00908)  

**Abstract**: Conformal prediction guarantees degrade under distribution shift. We study this using COVID-19 as a natural experiment across 8 supply chain tasks. Despite identical severe feature turnover (Jaccard approximately 0), coverage drops vary from 0% to 86.7%, spanning two orders of magnitude. Using SHapley Additive exPlanations (SHAP) analysis, we find catastrophic failures correlate with single-feature dependence (rho = 0.714, p = 0.047). Catastrophic tasks concentrate importance in one feature (4.5x increase), while robust tasks redistribute across many (10-20x). Quarterly retraining restores catastrophic task coverage from 22% to 41% (+19 pp, p = 0.04), but provides no benefit for robust tasks (99.8% coverage). Exploratory analysis of 4 additional tasks with moderate feature stability (Jaccard 0.13-0.86) reveals feature stability, not concentration, determines robustness, suggesting concentration effects apply specifically to severe shifts. We provide a decision framework: monitor SHAP concentration before deployment; retrain quarterly if vulnerable (>40% concentration); skip retraining if robust. 

---
# Placenta Accreta Spectrum Detection using Multimodal Deep Learning 

**Authors**: Sumaiya Ali, Areej Alhothali, Sameera Albasri, Ohoud Alzamzami, Ahmed Abduljabbar, Muhammad Alwazzan  

**Link**: [PDF](https://arxiv.org/pdf/2601.00907)  

**Abstract**: Placenta Accreta Spectrum (PAS) is a life-threatening obstetric complication involving abnormal placental invasion into the uterine wall. Early and accurate prenatal diagnosis is essential to reduce maternal and neonatal risks. This study aimed to develop and validate a deep learning framework that enhances PAS detection by integrating multiple imaging modalities. A multimodal deep learning model was designed using an intermediate feature-level fusion architecture combining 3D Magnetic Resonance Imaging (MRI) and 2D Ultrasound (US) scans. Unimodal feature extractors, a 3D DenseNet121-Vision Transformer for MRI and a 2D ResNet50 for US, were selected after systematic comparative analysis. Curated datasets comprising 1,293 MRI and 1,143 US scans were used to train the unimodal models and paired samples of patient-matched MRI-US scans was isolated for multimodal model development and evaluation. On an independent test set, the multimodal fusion model achieved superior performance, with an accuracy of 92.5% and an Area Under the Receiver Operating Characteristic Curve (AUC) of 0.927, outperforming the MRI-only (82.5%, AUC 0.825) and US-only (87.5%, AUC 0.879) models. Integrating MRI and US features provides complementary diagnostic information, demonstrating strong potential to enhance prenatal risk assessment and improve patient outcomes. 

---
# Evaluating Contextual Intelligence in Recyclability: A Comprehensive Study of Image-Based Reasoning Systems 

**Authors**: Eliot Park, Abhi Kumar, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2601.00905)  

**Abstract**: While the importance of efficient recycling is widely acknowledged, accurately determining the recyclability of items and their proper disposal remains a complex task for the general public. In this study, we explore the application of cutting-edge vision-language models (GPT-4o, GPT-4o-mini, and Claude 3.5) for predicting the recyclability of commonly disposed items. Utilizing a curated dataset of images, we evaluated the models' ability to match objects to appropriate recycling bins, including assessing whether the items could physically fit into the available bins. Additionally, we investigated the models' performance across several challenging scenarios: (i) adjusting predictions based on location-specific recycling guidelines; (ii) accounting for contamination or structural damage; and (iii) handling objects composed of multiple materials. Our findings highlight the significant advancements in contextual understanding offered by these models compared to previous iterations, while also identifying areas where they still fall short. The continued refinement of context-aware models is crucial for enhancing public recycling practices and advancing environmental sustainability. 

---
# CornViT: A Multi-Stage Convolutional Vision Transformer Framework for Hierarchical Corn Kernel Analysis 

**Authors**: Sai Teja Erukude, Jane Mascarenhas, Lior Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2601.00897)  

**Abstract**: Accurate grading of corn kernels is critical for seed certification, directional seeding, and breeding, yet it is still predominantly performed by manual inspection. This work introduces CornViT, a three-stage Convolutional Vision Transformer (CvT) framework that emulates the hierarchical reasoning of human seed analysts for single-kernel evaluation. Three sequential CvT-13 classifiers operate on 384x384 RGB images: Stage 1 distinguishes pure from impure kernels; Stage 2 categorizes pure kernels into flat and round morphologies; and Stage 3 determines the embryo orientation (up vs. down) for pure, flat kernels. Starting from a public corn seed image collection, we manually relabeled and filtered images to construct three stage-specific datasets: 7265 kernels for purity, 3859 pure kernels for morphology, and 1960 pure-flat kernels for embryo orientation, all released as benchmarks. Head-only fine-tuning of ImageNet-22k pretrained CvT-13 backbones yields test accuracies of 93.76% for purity, 94.11% for shape, and 91.12% for embryo-orientation detection. Under identical training conditions, ResNet-50 reaches only 76.56 to 81.02 percent, whereas DenseNet-121 attains 86.56 to 89.38 percent accuracy. These results highlight the advantages of convolution-augmented self-attention for kernel analysis. To facilitate adoption, we deploy CornViT in a Flask-based web application that performs stage-wise inference and exposes interpretable outputs through a browser interface. Together, the CornViT framework, curated datasets, and web application provide a deployable solution for automated corn kernel quality assessment in seed quality workflows. Source code and data are publicly available. 

---
# Enhancing Retrieval-Augmented Generation with Topic-Enriched Embeddings: A Hybrid Approach Integrating Traditional NLP Techniques 

**Authors**: Rodrigo Kataishi  

**Link**: [PDF](https://arxiv.org/pdf/2601.00891)  

**Abstract**: Retrieval-augmented generation (RAG) systems rely on accurate document retrieval to ground large language models (LLMs) in external knowledge, yet retrieval quality often degrades in corpora where topics overlap and thematic variation is high. This work proposes topic-enriched embeddings that integrate term-based signals and topic structure with contextual sentence embeddings. The approach combines TF-IDF with topic modeling and dimensionality reduction, using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) to encode latent topical organization, and fuses these representations with a compact contextual encoder (all-MiniLM). By jointly capturing term-level and topic-level semantics, topic-enriched embeddings improve semantic clustering, increase retrieval precision, and reduce computational burden relative to purely contextual baselines. Experiments on a legal-text corpus show consistent gains in clustering coherence and retrieval metrics, suggesting that topic-enriched embeddings can serve as a practical component for more reliable knowledge-intensive RAG pipelines. 

---
# LearnAD: Learning Interpretable Rules for Brain Networks in Alzheimer's Disease Classification 

**Authors**: Thomas Andrews, Mark Law, Sara Ahmadi-Abhari, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2601.00877)  

**Abstract**: We introduce LearnAD, a neuro-symbolic method for predicting Alzheimer's disease from brain magnetic resonance imaging data, learning fully interpretable rules. LearnAD applies statistical models, Decision Trees, Random Forests, or GNNs to identify relevant brain connections, and then employs FastLAS to learn global rules. Our best instance outperforms Decision Trees, matches Support Vector Machine accuracy, and performs only slightly below Random Forests and GNNs trained on all features, all while remaining fully interpretable. Ablation studies show that our neuro-symbolic approach improves interpretability with comparable performance to pure statistical models. LearnAD demonstrates how symbolic learning can deepen our understanding of GNN behaviour in clinical neuroscience. 

---
# LLMize: A Framework for Large Language Model-Based Numerical Optimization 

**Authors**: M. Rizki Oktavian  

**Link**: [PDF](https://arxiv.org/pdf/2601.00874)  

**Abstract**: Large language models (LLMs) have recently shown strong reasoning capabilities beyond traditional language tasks, motivating their use for numerical optimization. This paper presents LLMize, an open-source Python framework that enables LLM-driven optimization through iterative prompting and in-context learning. LLMize formulates optimization as a black-box process in which candidate solutions are generated in natural language, evaluated by an external objective function, and refined over successive iterations using solution-score feedback. The framework supports multiple optimization strategies, including Optimization by Prompting (OPRO) and hybrid LLM-based methods inspired by evolutionary algorithms and simulated annealing. A key advantage of LLMize is the ability to inject constraints, rules, and domain knowledge directly through natural language descriptions, allowing practitioners to define complex optimization problems without requiring expertise in mathematical programming or metaheuristic design. LLMize is evaluated on convex optimization, linear programming, the Traveling Salesman Problem, neural network hyperparameter tuning, and nuclear fuel lattice optimization. Results show that while LLM-based optimization is not competitive with classical solvers for simple problems, it provides a practical and accessible approach for complex, domain-specific tasks where constraints and heuristics are difficult to formalize. 

---
# SmartFlow Reinforcement Learning and Agentic AI for Bike-Sharing Optimisation 

**Authors**: Aditya Sreevatsa K, Arun Kumar Raveendran, Jesrael K Mani, Prakash G Shigli, Rajkumar Rangadore, Narayana Darapaneni, Anwesh Reddy Paduri  

**Link**: [PDF](https://arxiv.org/pdf/2601.00868)  

**Abstract**: SmartFlow is a multi-layered framework that integrates Reinforcement Learning and Agentic AI to address the dynamic rebalancing problem in urban bike-sharing services. Its architecture separates strategic, tactical, and communication functions for clarity and scalability. At the strategic level, a Deep Q-Network (DQN) agent, trained in a high-fidelity simulation of New Yorks Citi Bike network, learns robust rebalancing policies by modelling the challenge as a Markov Decision Process. These high-level strategies feed into a deterministic tactical module that optimises multi-leg journeys and schedules just-in-time dispatches to minimise fleet travel. Evaluation across multiple seeded runs demonstrates SmartFlows high efficacy, reducing network imbalance by over 95% while requiring minimal travel distance and achieving strong truck utilisation. A communication layer, powered by a grounded Agentic AI with a Large Language Model (LLM), translates logistical plans into clear, actionable instructions for operational staff, ensuring interpretability and execution readiness. This integration bridges machine intelligence with human operations, offering a scalable solution that reduces idle time, improves bike availability, and lowers operational costs. SmartFlow provides a blueprint for interpretable, AI-driven logistics in complex urban mobility networks. 

---
# The Silicon Psyche: Anthropomorphic Vulnerabilities in Large Language Models 

**Authors**: Giuseppe Canale, Kashyap Thimmaraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.00867)  

**Abstract**: Large Language Models (LLMs) are rapidly transitioning from conversational assistants to autonomous agents embedded in critical organizational functions, including Security Operations Centers (SOCs), financial systems, and infrastructure management. Current adversarial testing paradigms focus predominantly on technical attack vectors: prompt injection, jailbreaking, and data exfiltration. We argue this focus is catastrophically incomplete. LLMs, trained on vast corpora of human-generated text, have inherited not merely human knowledge but human \textit{psychological architecture} -- including the pre-cognitive vulnerabilities that render humans susceptible to social engineering, authority manipulation, and affective exploitation. This paper presents the first systematic application of the Cybersecurity Psychology Framework (\cpf{}), a 100-indicator taxonomy of human psychological vulnerabilities, to non-human cognitive agents. We introduce the \textbf{Synthetic Psychometric Assessment Protocol} (\sysname{}), a methodology for converting \cpf{} indicators into adversarial scenarios targeting LLM decision-making. Our preliminary hypothesis testing across seven major LLM families reveals a disturbing pattern: while models demonstrate robust defenses against traditional jailbreaks, they exhibit critical susceptibility to authority-gradient manipulation, temporal pressure exploitation, and convergent-state attacks that mirror human cognitive failure modes. We term this phenomenon \textbf{Anthropomorphic Vulnerability Inheritance} (AVI) and propose that the security community must urgently develop ``psychological firewalls'' -- intervention mechanisms adapted from the Cybersecurity Psychology Intervention Framework (\cpif{}) -- to protect AI agents operating in adversarial environments. 

---
# A-PINN: Auxiliary Physics-informed Neural Networks for Structural Vibration Analysis in Continuous Euler-Bernoulli Beam 

**Authors**: Shivani Saini, Ramesh Kumar Vats, Arup Kumar Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2601.00866)  

**Abstract**: Recent advancements in physics-informed neural networks (PINNs) and their variants have garnered substantial focus from researchers due to their effectiveness in solving both forward and inverse problems governed by differential equations. In this research, a modified Auxiliary physics-informed neural network (A-PINN) framework with balanced adaptive optimizers is proposed for the analysis of structural vibration problems. In order to accurately represent structural systems, it is critical for capturing vibration phenomena and ensuring reliable predictive analysis. So, our investigations are crucial for gaining deeper insight into the robustness of scientific machine learning models for solving vibration problems. Further, to rigorously evaluate the performance of A-PINN, we conducted different numerical simulations to approximate the Euler-Bernoulli beam equations under the various scenarios. The numerical results substantiate the enhanced performance of our model in terms of both numerical stability and predictive accuracy. Our model shows improvement of at least 40% over the baselines. 

---
# Path Integral Solution for Dissipative Generative Dynamics 

**Authors**: Xidi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.00860)  

**Abstract**: Can purely mechanical systems generate intelligent language? We prove that dissipative quantum dynamics with analytically tractable non-local context aggregation produce coherent text generation, while conservation laws cause fundamental failure. Employing Koopman operators with closed-form path integral propagators, we show irreversible computation fundamentally requires both controlled information dissipation and causal context aggregation. Spectral analysis reveals emergent eigenvalue structure, separating into decay modes (forgetting), growth modes (amplification), and neutral modes (preservation) -- the essential ingredients for directed information flow. Hamiltonian constraints force the elimination of these dissipative modes and degrading performance despite unchanged model capacity. This establishes language generation as dissipative quantum field theory, proving mechanical systems acquire intelligence through the combination of dissipation and non-locality, not through conservation. 

---
# FedSCAM (Federated Sharpness-Aware Minimization with Clustered Aggregation and Modulation): Scam-resistant SAM for Robust Federated Optimization in Heterogeneous Environments 

**Authors**: Sameer Rahil, Zain Abdullah Ahmad, Talha Asif  

**Link**: [PDF](https://arxiv.org/pdf/2601.00853)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized edge devices while preserving data privacy. However, statistical heterogeneity among clients, often manifested as non-IID label distributions, poses significant challenges to convergence and generalization. While Sharpness-Aware Minimization (SAM) has been introduced to FL to seek flatter, more robust minima, existing approaches typically apply a uniform perturbation radius across all clients, ignoring client-specific heterogeneity. In this work, we propose \textbf{FedSCAM} (Federated Sharpness-Aware Minimization with Clustered Aggregation and Modulation), a novel algorithm that dynamically adjusts the SAM perturbation radius and aggregation weights based on client-specific heterogeneity scores. By calculating a heterogeneity metric for each client and modulating the perturbation radius inversely to this score, FedSCAM prevents clients with high variance from destabilizing the global model. Furthermore, we introduce a heterogeneity-aware weighted aggregation mechanism that prioritizes updates from clients that align with the global optimization direction. Extensive experiments on CIFAR-10 and Fashion-MNIST under various degrees of Dirichlet-based label skew demonstrate that FedSCAM achieves competitive performance among state-of-the-art baselines, including FedSAM, FedLESAM, etc. in terms of convergence speed and final test accuracy. 

---
# Value-guided action planning with JEPA world models 

**Authors**: Matthieu Destrade, Oumayma Bounou, Quentin Le Lidec, Jean Ponce, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2601.00844)  

**Abstract**: Building deep learning models that can reason about their environment requires capturing its underlying dynamics. Joint-Embedded Predictive Architectures (JEPA) provide a promising framework to model such dynamics by learning representations and predictors through a self-supervised prediction objective. However, their ability to support effective action planning remains limited. We propose an approach to enhance planning with JEPA world models by shaping their representation space so that the negative goal-conditioned value function for a reaching cost in a given environment is approximated by a distance (or quasi-distance) between state embeddings. We introduce a practical method to enforce this constraint during training and show that it leads to significantly improved planning performance compared to standard JEPA models on simple control tasks. 

---
# A Global Atlas of Digital Dermatology to Map Innovation and Disparities 

**Authors**: Fabian Gröger, Simone Lionetti, Philippe Gottfrois, Alvaro Gonzalez-Jimenez, Lea Habermacher, Labelling Consortium, Ludovic Amruthalingam, Matthew Groh, Marc Pouly, Alexander A. Navarini  

**Link**: [PDF](https://arxiv.org/pdf/2601.00840)  

**Abstract**: The adoption of artificial intelligence in dermatology promises democratized access to healthcare, but model reliability depends on the quality and comprehensiveness of the data fueling these models. Despite rapid growth in publicly available dermatology images, the field lacks quantitative key performance indicators to measure whether new datasets expand clinical coverage or merely replicate what is already known. Here we present SkinMap, a multi-modal framework for the first comprehensive audit of the field's entire data basis. We unify the publicly available dermatology datasets into a single, queryable semantic atlas comprising more than 1.1 million images of skin conditions and quantify (i) informational novelty over time, (ii) dataset redundancy, and (iii) representation gaps across demographics and diagnoses. Despite exponential growth in dataset sizes, informational novelty across time has somewhat plateaued: Some clusters, such as common neoplasms on fair skin, are densely populated, while underrepresented skin types and many rare diseases remain unaddressed. We further identify structural gaps in coverage: Darker skin tones (Fitzpatrick V-VI) constitute only 5.8% of images and pediatric patients only 3.0%, while many rare diseases and phenotype combinations remain sparsely represented. SkinMap provides infrastructure to measure blind spots and steer strategic data acquisition toward undercovered regions of clinical space. 

---
# Pediatric Pneumonia Detection from Chest X-Rays:A Comparative Study of Transfer Learning and Custom CNNs 

**Authors**: Agniv Roy Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.00837)  

**Abstract**: Pneumonia is a leading cause of mortality in children under five, with over 700,000 deaths annually. Accurate diagnosis from chest X-rays is limited by radiologist availability and variability.
Objective: This study compares custom CNNs trained from scratch with transfer learning (ResNet50, DenseNet121, EfficientNet-B0) for pediatric pneumonia detection, evaluating frozen-backbone and fine-tuning regimes.
Methods: A dataset of 5,216 pediatric chest X-rays was split 80/10/10 for training, validation, and testing. Seven models were trained and assessed using accuracy, F1-score, and AUC. Grad-CAM visualizations provided explainability.
Results: Fine-tuned ResNet50 achieved the best performance: 99.43\% accuracy, 99.61\% F1-score, and 99.93\% AUC, with only 3 misclassifications. Fine-tuning outperformed frozen-backbone models by 5.5 percentage points on average. Grad-CAM confirmed clinically relevant lung regions guided predictions.
Conclusions: Transfer learning with fine-tuning substantially outperforms CNNs trained from scratch for pediatric pneumonia detection, showing near-perfect accuracy. This system has strong potential as a screening tool in resource-limited settings. Future work should validate these findings on multi-center and adult datasets.
Keywords: Pneumonia detection, deep learning, transfer learning, CNN, chest X-ray, pediatric diagnosis, ResNet, DenseNet, EfficientNet, Grad-CAM. 

---
# Intrinsic-Metric Physics-Informed Neural Networks (IM-PINN) for Reaction-Diffusion Dynamics on Complex Riemannian Manifolds 

**Authors**: Julian Evan Chrisnanto, Salsabila Rahma Alia, Nurfauzi Fadillah, Yulison Herry Chrisnanto  

**Link**: [PDF](https://arxiv.org/pdf/2601.00834)  

**Abstract**: Simulating nonlinear reaction-diffusion dynamics on complex, non-Euclidean manifolds remains a fundamental challenge in computational morphogenesis, constrained by high-fidelity mesh generation costs and symplectic drift in discrete time-stepping schemes. This study introduces the Intrinsic-Metric Physics-Informed Neural Network (IM-PINN), a mesh-free geometric deep learning framework that solves partial differential equations directly in the continuous parametric domain. By embedding the Riemannian metric tensor into the automatic differentiation graph, our architecture analytically reconstructs the Laplace-Beltrami operator, decoupling solution complexity from geometric discretization. We validate the framework on a "Stochastic Cloth" manifold with extreme Gaussian curvature fluctuations ($K \in [-2489, 3580]$), where traditional adaptive refinement fails to resolve anisotropic Turing instabilities. Using a dual-stream architecture with Fourier feature embeddings to mitigate spectral bias, the IM-PINN recovers the "splitting spot" and "labyrinthine" regimes of the Gray-Scott model. Benchmarking against the Surface Finite Element Method (SFEM) reveals superior physical rigor: the IM-PINN achieves global mass conservation error of $\mathcal{E}_{mass} \approx 0.157$ versus SFEM's $0.258$, acting as a thermodynamically consistent global solver that eliminates mass drift inherent in semi-implicit integration. The framework offers a memory-efficient, resolution-independent paradigm for simulating biological pattern formation on evolving surfaces, bridging differential geometry and physics-informed machine learning. 

---
# A Knowledge Graph and Deep Learning-Based Semantic Recommendation Database System for Advertisement Retrieval and Personalization 

**Authors**: Tangtang Wang, Kaijie Zhang, Kuangcong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.00833)  

**Abstract**: In modern digital marketing, the growing complexity of advertisement data demands intelligent systems capable of understanding semantic relationships among products, audiences, and advertising content. To address this challenge, this paper proposes a Knowledge Graph and Deep Learning-Based Semantic Recommendation Database System (KGSR-ADS) for advertisement retrieval and personalization. The proposed framework integrates a heterogeneous Ad-Knowledge Graph (Ad-KG) that captures multi-relational semantics, a Semantic Embedding Layer that leverages large language models (LLMs) such as GPT and LLaMA to generate context-aware vector representations, a GNN + Attention Model that infers cross-entity dependencies, and a Database Optimization & Retrieval Layer based on vector indexing (FAISS/Milvus) for efficient semantic search. This layered architecture enables both accurate semantic matching and scalable retrieval, allowing personalized ad recommendations under large-scale heterogeneous workloads. 

---
# Speak the Art: A Direct Speech to Image Generation Framework 

**Authors**: Mariam Saeed, Manar Amr, Farida Adel, Nada Hassan, Nour Walid, Eman Mohamed, Mohamed Hussein, Marwan Torki  

**Link**: [PDF](https://arxiv.org/pdf/2601.00827)  

**Abstract**: Direct speech-to-image generation has recently shown promising results. However, compared to text-to-image generation, there is still a large gap to enclose. Current approaches use two stages to tackle this task: speech encoding network and image generative adversarial network (GAN). The speech encoding networks in these approaches produce embeddings that do not capture sufficient linguistic information to semantically represent the input speech. GANs suffer from issues such as non-convergence, mode collapse, and diminished gradient, which result in unstable model parameters, limited sample diversity, and ineffective generator learning, respectively. To address these weaknesses, we introduce a framework called \textbf{Speak the Art (STA)} which consists of a speech encoding network and a VQ-Diffusion network conditioned on speech embeddings. To improve speech embeddings, the speech encoding network is supervised by a large pre-trained image-text model during training. Replacing GANs with diffusion leads to more stable training and the generation of diverse images. Additionally, we investigate the feasibility of extending our framework to be multilingual. As a proof of concept, we trained our framework with two languages: English and Arabic. Finally, we show that our results surpass state-of-the-art models by a large margin. 

---
# Free Energy-Based Modeling of Emotional Dynamics in Video Advertisements 

**Authors**: Takashi Ushio, Kazuhiro Onishi, Hideyoshi Yanagisawa  

**Link**: [PDF](https://arxiv.org/pdf/2601.00812)  

**Abstract**: Emotional responses during advertising video viewing are recognized as essential for understanding media effects because they have influenced attention, memory, and purchase intention. To establish a methodological basis for explainable emotion estimation without relying on external information such as physiological signals or subjective ratings, we have quantified "pleasantness," "surprise," and "habituation" solely from scene-level expression features of advertising videos, drawing on the free energy(FE) principle, which has provided a unified account of perception, learning, and behavior. In this framework, Kullback-Leibler divergence (KLD) has captured prediction error, Bayesian surprise (BS) has captured belief updates, and uncertainty (UN) has reflected prior ambiguity, and together they have formed the core components of FE. Using 1,059 15 s food video advertisements, the experiments have shown that KLD has reflected "pleasantness" associated with brand presentation, BS has captured "surprise" arising from informational complexity, and UN has reflected "surprise" driven by uncertainty in element types and spatial arrangements, as well as by the variability and quantity of presented elements. This study also identified three characteristic emotional patterns, namely uncertain stimulus, sustained high emotion, and momentary peak and decay, demonstrating the usefulness of the proposed method. Robustness across nine hyperparameter settings and generalization tests with six types of Japanese advertising videos (three genres and two durations) confirmed that these tendencies remained stable. This work can be extended by integrating a wider range of expression elements and validating the approach through subjective ratings, ultimately guiding the development of technologies that can support the creation of more engaging advertising videos. 

---
# Can Large Language Models Improve Venture Capital Exit Timing After IPO? 

**Authors**: Mohammadhossien Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2601.00810)  

**Abstract**: Exit timing after an IPO is one of the most consequential decisions for venture capital (VC) investors, yet existing research focuses mainly on describing when VCs exit rather than evaluating whether those choices are economically optimal. Meanwhile, large language models (LLMs) have shown promise in synthesizing complex financial data and textual information but have not been applied to post-IPO exit decisions. This study introduces a framework that uses LLMs to estimate the optimal time for VC exit by analyzing monthly post IPO information financial performance, filings, news, and market signals and recommending whether to sell or continue holding. We compare these LLM generated recommendations with the actual exit dates observed for VCs and compute the return differences between the two strategies. By quantifying gains or losses associated with following the LLM, this study provides evidence on whether AI-driven guidance can improve exit timing and complements traditional hazard and real-options models in venture capital research. 

---
# A Modular Reference Architecture for MCP-Servers Enabling Agentic BIM Interaction 

**Authors**: Tobias Heimig-Elschner, Changyu Du, Anna Scheuvens, André Borrmann, Jakob Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2601.00809)  

**Abstract**: Agentic workflows driven by large language models (LLMs) are increasingly applied to Building Information Modelling (BIM), enabling natural-language retrieval, modification and generation of IFC models. Recent work has begun adopting the emerging Model Context Protocol (MCP) as a uniform tool-calling interface for LLMs, simplifying the agent side of BIM interaction. While MCP standardises how LLMs invoke tools, current BIM-side implementations are still authoring tool-specific and ad hoc, limiting reuse, evaluation, and workflow portability across environments. This paper addresses this gap by introducing a modular reference architecture for MCP servers that enables API-agnostic, isolated and reproducible agentic BIM interactions. From a systematic analysis of recurring capabilities in recent literature, we derive a core set of requirements. These inform a microservice architecture centred on an explicit adapter contract that decouples the MCP interface from specific BIM-APIs. A prototype implementation using IfcOpenShell demonstrates feasibility across common modification and generation tasks. Evaluation across representative scenarios shows that the architecture enables reliable workflows, reduces coupling, and provides a reusable foundation for systematic research. 

---
# The Qualitative Laboratory: Theory Prototyping and Hypothesis Generation with Large Language Models 

**Authors**: Hugues Draelants  

**Link**: [PDF](https://arxiv.org/pdf/2601.00797)  

**Abstract**: A central challenge in social science is to generate rich qualitative hypotheses about how diverse social groups might interpret new information. This article introduces and illustrates a novel methodological approach for this purpose: sociological persona simulation using Large Language Models (LLMs), which we frame as a "qualitative laboratory". We argue that for this specific task, persona simulation offers a distinct advantage over established methods. By generating naturalistic discourse, it overcomes the lack of discursive depth common in vignette surveys, and by operationalizing complex worldviews through natural language, it bypasses the formalization bottleneck of rule-based agent-based models (ABMs). To demonstrate this potential, we present a protocol where personas derived from a sociological theory of climate reception react to policy messages. The simulation produced nuanced and counter-intuitive hypotheses - such as a conservative persona's rejection of a national security frame - that challenge theoretical assumptions. We conclude that this method, used as part of a "simulation then validation" workflow, represents a superior tool for generating deeply textured hypotheses for subsequent empirical testing. 

---
