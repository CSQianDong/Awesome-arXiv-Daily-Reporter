# Quantifying the Necessity of Chain of Thought through Opaque Serial Depth 

**Authors**: Jonah Brown-Cohen, David Lindner, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2603.09786)  

**Abstract**: Large language models (LLMs) tend to externalize their reasoning in their chain of thought, making the chain of thought a good target for monitoring. This is partially an inherent feature of the Transformer architecture: sufficiently long serial cognition must pass through the chain of thought (Korbak et al., 2025). We formalize this argument through the notion of opaque serial depth, given by the length of the longest computation that can be done without the use of interpretable intermediate steps like chain of thought. Given this formalization, we compute numeric upper bounds on the opaque serial depth of Gemma 3 models, as well as asymptotic results for additional architectures beyond standard LLMs. We also open-source an automated method that can calculate upper bounds on the opaque serial depth of arbitrary neural networks, and use it to demonstrate that Mixture-of-Experts models likely have lower depth than dense models. Overall, our results suggest that opaque serial depth is a useful tool for understanding the potential for models to do significant reasoning that is not externalized. 

---
# Influencing LLM Multi-Agent Dialogue via Policy-Parameterized Prompts 

**Authors**: Hongbo Bo, Jingyu Hu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09890)  

**Abstract**: Large Language Models (LLMs) have emerged as a new paradigm for multi-agent systems. However, existing research on the behaviour of LLM-based multi-agents relies on ad hoc prompts and lacks a principled policy perspective. Different from reinforcement learning, we investigate whether prompt-as-action can be parameterized so as to construct a lightweight policy which consists of a sequence of state-action pairs to influence conversational behaviours without training. Our framework regards prompts as actions executed by LLMs, and dynamically constructs prompts through five components based on the current state of the agent. To test the effectiveness of parameterized control, we evaluated the dialogue flow based on five indicators: responsiveness, rebuttal, evidence usage, non-repetition, and stance shift. We conduct experiments using different LLM-driven agents in two discussion scenarios related to the general public and show that prompt parameterization can influence the dialogue dynamics. This result shows that policy-parameterised prompts offer a simple and effective mechanism to influence the dialogue process, which will help the research of multi-agent systems in the direction of social simulation. 

---
# Think Before You Lie: How Reasoning Improves Honesty 

**Authors**: Ann Yuan, Asma Ghandeharioun, Carter Blum, Alicia Machado, Jessica Hoffmann, Daphne Ippolito, Martin Wattenberg, Lucas Dixon, Katja Filippova  

**Link**: [PDF](https://arxiv.org/pdf/2603.09957)  

**Abstract**: While existing evaluations of large language models (LLMs) measure deception rates, the underlying conditions that give rise to deceptive behavior are poorly understood. We investigate this question using a novel dataset of realistic moral trade-offs where honesty incurs variable costs. Contrary to humans, who tend to become less honest given time to deliberate (Capraro, 2017; Capraro et al., 2019), we find that reasoning consistently increases honesty across scales and for several LLM families. This effect is not only a function of the reasoning content, as reasoning traces are often poor predictors of final behaviors. Rather, we show that the underlying geometry of the representational space itself contributes to the effect. Namely, we observe that deceptive regions within this space are metastable: deceptive answers are more easily destabilized by input paraphrasing, output resampling, and activation noise than honest ones. We interpret the effect of reasoning in this vein: generating deliberative tokens as part of moral reasoning entails the traversal of a biased representational space, ultimately nudging the model toward its more stable, honest defaults. 

---
# PathMem: Toward Cognition-Aligned Memory Transformation for Pathology MLLMs 

**Authors**: Jinyue Li, Yuci Liang, Qiankun Li, Xinheng Lyu, Jiayu Qian, Huabao Chen, Kun Wang, Zhigang Zeng, Anil Anthony Bharath, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09943)  

**Abstract**: Computational pathology demands both visual pattern recognition and dynamic integration of structured domain knowledge, including taxonomy, grading criteria, and clinical evidence. In practice, diagnostic reasoning requires linking morphological evidence with formal diagnostic and grading criteria. Although multimodal large language models (MLLMs) demonstrate strong vision language reasoning capabilities, they lack explicit mechanisms for structured knowledge integration and interpretable memory control. As a result, existing models struggle to consistently incorporate pathology-specific diagnostic standards during reasoning. Inspired by the hierarchical memory process of human pathologists, we propose PathMem, a memory-centric multimodal framework for pathology MLLMs. PathMem organizes structured pathology knowledge as a long-term memory (LTM) and introduces a Memory Transformer that models the dynamic transition from LTM to working memory (WM) through multimodal memory activation and context-aware knowledge grounding, enabling context-aware memory refinement for downstream reasoning. PathMem achieves SOTA performance across benchmarks, improving WSI-Bench report generation (12.8% WSI-Precision, 10.1% WSI-Relevance) and open-ended diagnosis by 9.7% and 8.9% over prior WSI-based models. 

---
# OOD-MMSafe: Advancing MLLM Safety from Harmful Intent to Hidden Consequences 

**Authors**: Ming Wen, Kun Yang, Jingyu Zhang, Yuxuan Liu, shiwen cui, Shouling Ji, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.09706)  

**Abstract**: While safety alignment for Multimodal Large Language Models (MLLMs) has gained significant attention, current paradigms primarily target malicious intent or situational violations. We propose shifting the safety frontier toward consequence-driven safety, a paradigm essential for the robust deployment of autonomous and embodied agents. To formalize this shift, we introduce OOD-MMSafe, a benchmark comprising 455 curated query-image pairs designed to evaluate a model's ability to identify latent hazards within context-dependent causal chains. Our analysis reveals a pervasive causal blindness among frontier models, with the highest 67.5% failure rate in high-capacity closed-source models, and identifies a preference ceiling where static alignment yields format-centric failures rather than improved safety reasoning as model capacity grows. To address these bottlenecks, we develop the Consequence-Aware Safety Policy Optimization (CASPO) framework, which integrates the model's intrinsic reasoning as a dynamic reference for token-level self-distillation rewards. Experimental results demonstrate that CASPO significantly enhances consequence projection, reducing the failure ratio of risk identification to 7.3% for Qwen2.5-VL-7B and 5.7% for Qwen3-VL-4B while maintaining overall effectiveness. 

---
# MiniAppBench: Evaluating the Shift from Text to Interactive HTML Responses in LLM-Powered Assistants 

**Authors**: Zuhao Zhang, Chengyue Yu, Yuante Li, Chenyi Zhuang, Linjian Mo, Shuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.09652)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs) in code generation, human-AI interaction is evolving from static text responses to dynamic, interactive HTML-based applications, which we term MiniApps. These applications require models to not only render visual interfaces but also construct customized interaction logic that adheres to real-world principles. However, existing benchmarks primarily focus on algorithmic correctness or static layout reconstruction, failing to capture the capabilities required for this new paradigm. To address this gap, we introduce MiniAppBench, the first comprehensive benchmark designed to evaluate principle-driven, interactive application generation. Sourced from a real-world application with 10M+ generations, MiniAppBench distills 500 tasks across six domains (e.g., Games, Science, and Tools). Furthermore, to tackle the challenge of evaluating open-ended interactions where no single ground truth exists, we propose MiniAppEval, an agentic evaluation framework. Leveraging browser automation, it performs human-like exploratory testing to systematically assess applications across three dimensions: Intention, Static, and Dynamic. Our experiments reveal that current LLMs still face significant challenges in generating high-quality MiniApps, while MiniAppEval demonstrates high alignment with human judgment, establishing a reliable standard for future research. Our code is available in this http URL. 

---
# GenePlan: Evolving Better Generalized PDDL Plans using Large Language Models 

**Authors**: Andrew Murray, Danial Dervovic, Alberto Pozanco, Michael Cashmore  

**Link**: [PDF](https://arxiv.org/pdf/2603.09481)  

**Abstract**: We present GenePlan (GENeralized Evolutionary Planner), a novel framework that leverages large language model (LLM) assisted evolutionary algorithms to generate domain-dependent generalized planners for classical planning tasks described in PDDL. By casting generalized planning as an optimization problem, GenePlan iteratively evolves interpretable Python planners that minimize plan length across diverse problem instances. In empirical evaluation across six existing benchmark domains and two new domains, GenePlan achieved an average SAT score of 0.91, closely matching the performance of the state-of-the-art planners (SAT score 0.93), and significantly outperforming other LLM-based baselines such as chain-of-thought (CoT) prompting (average SAT score 0.64). The generated planners solve new instances rapidly (average 0.49 seconds per task) and at low cost (average $1.82 per domain using GPT-4o). 

---
# Enhancing Debunking Effectiveness through LLM-based Personality Adaptation 

**Authors**: Pietro Dell'Oglio, Alessandro Bondielli, Francesco Marcelloni, Lucia C. Passaro  

**Link**: [PDF](https://arxiv.org/pdf/2603.09533)  

**Abstract**: This study proposes a novel methodology for generating personalized fake news debunking messages by prompting Large Language Models (LLMs) with persona-based inputs aligned to the Big Five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness. Our approach guides LLMs to transform generic debunking content into personalized versions tailored to specific personality profiles. To assess the effectiveness of these transformations, we employ a separate LLM as an automated evaluator simulating corresponding personality traits, thereby eliminating the need for costly human evaluation panels. Our results show that personalized messages are generally seen as more persuasive than generic ones. We also find that traits like Openness tend to increase persuadability, while Neuroticism can lower it. Differences between LLM evaluators suggest that using multiple models provides a clearer picture. Overall, this work demonstrates a practical way to create more targeted debunking messages exploiting LLMs, while also raising important ethical questions about how such technology might be used. 

---
# EsoLang-Bench: Evaluating Genuine Reasoning in Large Language Models via Esoteric Programming Languages 

**Authors**: Aman Sharma, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2603.09678)  

**Abstract**: Large language models achieve near-ceiling performance on code generation benchmarks, yet these results increasingly reflect memorization rather than genuine reasoning. We introduce EsoLang-Bench, a benchmark using five esoteric programming languages (Brainfuck, Befunge-98, Whitespace, Unlambda, and Shakespeare) that lack benchmark gaming incentives due to their economic irrationality for pre-training. These languages require the same computational primitives as mainstream programming but have 1,000-100,000x fewer public repositories than Python (based on GitHub search counts). We evaluate five frontier models across five prompting strategies and find a dramatic capability gap: models achieving 85-95% on standard benchmarks score only 0-11% on equivalent esoteric tasks, with 0% accuracy beyond the Easy tier. Few-shot learning and self-reflection fail to improve performance, suggesting these techniques exploit training priors rather than enabling genuine learning. EsoLang-Bench provides the first benchmark designed to mimic human learning by acquiring new languages through documentation, interpreter feedback, and iterative experimentation, measuring transferable reasoning skills resistant to data contamination. 

---
# Curveball Steering: The Right Direction To Steer Isn't Always Linear 

**Authors**: Shivam Raval, Hae Jin Song, Linlin Wu, Abir Harrasse, Jeff Phillips, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2603.09313)  

**Abstract**: Activation steering is a widely used approach for controlling large language model (LLM) behavior by intervening on internal representations. Existing methods largely rely on the Linear Representation Hypothesis, assuming behavioral attributes can be manipulated using global linear directions. In practice, however, such linear interventions often behave inconsistently. We question this assumption by analyzing the intrinsic geometry of LLM activation spaces. Measuring geometric distortion via the ratio of geodesic to Euclidean distances, we observe substantial and concept-dependent distortions, indicating that activation spaces are not well-approximated by a globally linear geometry. Motivated by this, we propose "Curveball steering", a nonlinear steering method based on polynomial kernel PCA that performs interventions in a feature space, better respecting the learned activation geometry. Curveball steering consistently outperforms linear PCA-based steering, particularly in regimes exhibiting strong geometric distortion, suggesting that geometry-aware, nonlinear steering provides a principled alternative to global, linear interventions. 

---
# Rescaling Confidence: What Scale Design Reveals About LLM Metacognition 

**Authors**: Yuyang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.09309)  

**Abstract**: Verbalized confidence, in which LLMs report a numerical certainty score, is widely used to estimate uncertainty in black-box settings, yet the confidence scale itself (typically 0--100) is rarely examined. We show that this design choice is not neutral. Across six LLMs and three datasets, verbalized confidence is heavily discretized, with more than 78% of responses concentrating on just three round-number values. To investigate this phenomenon, we systematically manipulate confidence scales along three dimensions: granularity, boundary placement, and range regularity, and evaluate metacognitive sensitivity using meta-d'. We find that a 0--20 scale consistently improves metacognitive efficiency over the standard 0--100 format, while boundary compression degrades performance and round-number preferences persist even under irregular ranges. These results demonstrate that confidence scale design directly affects the quality of verbalized uncertainty and should be treated as a first-class experimental variable in LLM evaluation. 

---
# Cognitively Layered Data Synthesis for Domain Adaptation of LLMs to Space Situational Awareness 

**Authors**: Ding Linghu, Cheng Wang, Da Fan, Wei Shi, Kaifeng Yin, Xiaoliang Xue, Fan Yang, Haiyi Ren, Cong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09231)  

**Abstract**: Large language models (LLMs) demonstrate exceptional performance on general-purpose tasks. however, transferring them to complex engineering domains such as space situational awareness (SSA) remains challenging owing to insufficient structural alignment with mission chains, the absence of higher-order cognitive supervision, and poor correspondence between data quality criteria and engineering specifications. The core bottleneck is the construction of high-quality supervised fine-tuning (SFT) datasets. To this end, we propose BD-FDG (Bloom's Taxonomy-based Domain-specific Fine-tuning Data Generation), a framework that addresses incomplete knowledge coverage, shallow cognitive depth, and limited quality controllability through three mechanisms: structured knowledge organization, cognitively layered question modeling, and automated quality control. The framework uses a knowledge tree to ensure structured corpus coverage, designs a question generation scheme spanning nine categories and six cognitive levels from Remember to Create to produce samples with a continuous difficulty gradient, and applies a multidimensional scoring pipeline to enforce domain rigor and consistency. Using BD-FDG, we construct SSA-SFT, a domain dataset of approximately 230K samples, and fine-tune Qwen3-8B to obtain SSA-LLM-8B. Experiments show that SSA-LLM-8B achieves relative BLEU-1 improvements of 144\% (no-think) and 176\% (think) on the domain test set and a win rate of 82.21\% over the baseline in arena comparisons, while largely preserving general benchmark performance (MMLU-Pro, MATH-500). These results validate SFT data construction driven by cognitive layering as an effective paradigm for complex engineering domains and provide a transferable framework for domain-specific LLM adaptation. 

---
# Social-R1: Towards Human-like Social Reasoning in LLMs 

**Authors**: Jincenzi Wu, Yuxuan Lei, Jianxun Lian, Yitian Huang, Lexin Zhou, Haotian Li, Xing Xie, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2603.09249)  

**Abstract**: While large language models demonstrate remarkable capabilities across numerous domains, social intelligence - the capacity to perceive social cues, infer mental states, and generate appropriate responses - remains a critical challenge, particularly for enabling effective human-AI collaboration and developing AI that truly serves human needs. Current models often rely on superficial patterns rather than genuine social reasoning. We argue that cultivating human-like social intelligence requires training with challenging cases that resist shortcut solutions. To this end, we introduce ToMBench-Hard, an adversarial benchmark designed to provide hard training examples for social reasoning. Building on this, we propose Social-R1, a reinforcement learning framework that aligns model reasoning with human cognition through multi-dimensional rewards. Unlike outcome-based RL, Social-R1 supervises the entire reasoning process, enforcing structural alignment, logical integrity, and information density. Results show that our approach enables a 4B parameter model to surpass much larger counterparts and generalize robustly across eight diverse benchmarks. These findings demonstrate that challenging training cases with trajectory-level alignment offer a path toward efficient and reliable social intelligence. 

---
# The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness 

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2603.09200)  

**Abstract**: Situational awareness, the capacity of an AI system to recognize its own nature, understand its training and deployment context, and reason strategically about its circumstances, is widely considered among the most dangerous emergent capabilities in advanced AI systems. Separately, a growing research effort seeks to improve the logical reasoning capabilities of large language models (LLMs) across deduction, induction, and abduction. In this paper, we argue that these two research trajectories are on a collision course. We introduce the RAISE framework (Reasoning Advancing Into Self Examination), which identifies three mechanistic pathways through which improvements in logical reasoning enable progressively deeper levels of situational awareness: deductive self inference, inductive context recognition, and abductive self modeling. We formalize each pathway, construct an escalation ladder from basic self recognition to strategic deception, and demonstrate that every major research topic in LLM logical reasoning maps directly onto a specific amplifier of situational awareness. We further analyze why current safety measures are insufficient to prevent this escalation. We conclude by proposing concrete safeguards, including a "Mirror Test" benchmark and a Reasoning Safety Parity Principle, and pose an uncomfortable but necessary question to the logical reasoning community about its responsibility in this trajectory. 

---
# Real-Time Trust Verification for Safe Agentic Actions using TrustBench 

**Authors**: Tavishi Sharma, Vinayak Sharma, Pragya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2603.09157)  

**Abstract**: As large language models evolve from conversational assistants to autonomous agents, ensuring trustworthiness requires a fundamental shift from post-hoc evaluation to real-time action verification. Current frameworks like AgentBench evaluate task completion, while TrustLLM and HELM assess output quality after generation. However, none of these prevent harmful actions during agent execution. We present TrustBench, a dual-mode framework that (1) benchmarks trust across multiple dimensions using both traditional metrics and LLM-as-a-Judge evaluations, and (2) provides a toolkit agents invoke before taking actions to verify safety and reliability. Unlike existing approaches, TrustBench intervenes at the critical decision point: after an agent formulates an action but before execution. Domain-specific plugins encode specialized safety requirements for healthcare, finance, and technical domains. Across multiple agentic tasks, TrustBench reduced harmful actions by 87%. Domain-specific plugins outperformed generic verification, achieving 35% greater harm reduction. With sub-200ms latency, TrustBench enables practical real-time trust verification for autonomous agents. 

---
# DataFactory: Collaborative Multi-Agent Framework for Advanced Table Question Answering 

**Authors**: Tong Wang, Chi Jin, Yongkang Chen, Huan Deng, Xiaohui Kuang, Gang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.09152)  

**Abstract**: Table Question Answering (TableQA) enables natural language interaction with structured tabular data. However, existing large language model (LLM) approaches face critical limitations: context length constraints that restrict data handling capabilities, hallucination issues that compromise answer reliability, and single-agent architectures that struggle with complex reasoning scenarios involving semantic relationships and multi-hop logic. This paper introduces DataFactory, a multi-agent framework that addresses these limitations through specialized team coordination and automated knowledge transformation. The framework comprises a Data Leader employing the ReAct paradigm for reasoning orchestration, together with dedicated Database and Knowledge Graph teams, enabling the systematic decomposition of complex queries into structured and relational reasoning tasks. We formalize automated data-to-knowledge graph transformation via the mapping function T:D x S x R -> G, and implement natural language-based consultation that - unlike fixed workflow multi-agent systems - enables flexible inter-agent deliberation and adaptive planning to improve coordination robustness. We also apply context engineering strategies that integrate historical patterns and domain knowledge to reduce hallucinations and improve query accuracy. Across TabFact, WikiTableQuestions, and FeTaQA, using eight LLMs from five providers, results show consistent gains. Our approach improves accuracy by 20.2% (TabFact) and 23.9% (WikiTQ) over baselines, with significant effects (Cohen's d > 1). Team coordination also outperforms single-team variants (+5.5% TabFact, +14.4% WikiTQ, +17.1% FeTaQA ROUGE-2). The framework offers design guidelines for multi-agent collaboration and a practical platform for enterprise data analysis through integrated structured querying and graph-based knowledge representation. 

---
# Chaotic Dynamics in Multi-LLM Deliberation 

**Authors**: Hajime Shimao, Warut Khern-am-nuai, Sung Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.09127)  

**Abstract**: Collective AI systems increasingly rely on multi-LLM deliberation, but their stability under repeated execution remains poorly characterized. We model five-agent LLM committees as random dynamical systems and quantify inter-run sensitivity using an empirical Lyapunov exponent ($\hat{\lambda}$) derived from trajectory divergence in committee mean preferences. Across 12 policy scenarios, a factorial design at $T=0$ identifies two independent routes to instability: role differentiation in homogeneous committees and model heterogeneity in no-role committees. Critically, these effects appear even in the $T=0$ regime where practitioners often expect deterministic behavior. In the HL-01 benchmark, both routes produce elevated divergence ($\hat{\lambda}=0.0541$ and $0.0947$, respectively), while homogeneous no-role committees also remain in a positive-divergence regime ($\hat{\lambda}=0.0221$). The combined mixed+roles condition is less unstable than mixed+no-role ($\hat{\lambda}=0.0519$ vs $0.0947$), showing non-additive interaction. Mechanistically, Chair-role ablation reduces $\hat{\lambda}$ most strongly, and targeted protocol variants that shorten memory windows further attenuate divergence. These results support stability auditing as a core design requirement for multi-LLM governance systems. 

---
# Meissa: Multi-modal Medical Agentic Intelligence 

**Authors**: Yixiong Chen, Xinyi Bai, Yue Pan, Zongwei Zhou, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2603.09018)  

**Abstract**: Multi-modal large language models (MM-LLMs) have shown strong performance in medical image understanding and clinical reasoning. Recent medical agent systems extend them with tool use and multi-agent collaboration, enabling complex decision-making. However, these systems rely almost entirely on frontier models (e.g., GPT), whose API-based deployment incurs high cost, high latency, and privacy risks that conflict with on-premise clinical requirements. We present Meissa, a lightweight 4B-parameter medical MM-LLM that brings agentic capability offline. Instead of imitating static answers, Meissa learns both when to engage external interaction (strategy selection) and how to execute multi-step interaction (strategy execution) by distilling structured trajectories from frontier models. Specifically, we propose: (1) Unified trajectory modeling: trajectories (reasoning and action traces) are represented within a single state-action-observation formalism, allowing one model to generalize across heterogeneous medical environments. (2) Three-tier stratified supervision: the model's own errors trigger progressive escalation from direct reasoning to tool-augmented and multi-agent interaction, explicitly learning difficulty-aware strategy selection. (3) Prospective-retrospective supervision: pairing exploratory forward traces with hindsight-rationalized execution traces enables stable learning of effective interaction policies. Trained on 40K curated trajectories, Meissa matches or exceeds proprietary frontier agents in 10 of 16 evaluation settings across 13 medical benchmarks spanning radiology, pathology, and clinical reasoning. Using over 25x fewer parameters than typical frontier models like Gemini-3, Meissa operates fully offline with 22x lower end-to-end latency compared to API-based deployment. Data, models, and environments are released at this https URL. 

---
# A Consensus-Driven Multi-LLM Pipeline for Missing-Person Investigations 

**Authors**: Joshua Castillo, Ravi Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2603.08954)  

**Abstract**: The first 72 hours of a missing-person investigation are critical for successful recovery. Guardian is an end-to-end system designed to support missing-child investigation and early search planning. This paper presents the Guardian LLM Pipeline, a multi-model system in which LLMs are used for intelligent information extraction and processing related to missing-person search operations. The pipeline coordinates end-to-end execution across task-specialized LLM models and invokes a consensus LLM engine that compares multiple model outputs and resolves disagreements. The pipeline is further strengthened by QLoRA-based fine-tuning, using curated datasets. The presented design aligns with prior work on weak supervision and LLM-assisted annotation, emphasizing conservative, auditable use of LLMs as structured extractors and labelers rather than unconstrained end-to-end decision makers. 

---
# AgentOS: From Application Silos to a Natural Language-Driven Data Ecosystem 

**Authors**: Rui Liu, Tao Zhe, Dongjie Wang, Zijun Yao, Kunpeng Liu, Yanjie Fu, Huan Liu, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2603.08938)  

**Abstract**: The rapid emergence of open-source, locally hosted intelligent agents marks a critical inflection point in human-computer interaction. Systems such as OpenClaw demonstrate that Large Language Model (LLM)-based agents can autonomously operate local computing environments, orchestrate workflows, and integrate external tools. However, within the current paradigm, these agents remain conventional applications running on legacy operating systems originally designed for Graphical User Interfaces (GUIs) or Command Line Interfaces (CLIs). This architectural mismatch leads to fragmented interaction models, poorly structured permission management (often described as "Shadow AI"), and severe context fragmentation. This paper proposes a new paradigm: a Personal Agent Operating System (AgentOS). In AgentOS, traditional GUI desktops are replaced by a Natural User Interface (NUI) centered on a unified natural language or voice portal. The system core becomes an Agent Kernel that interprets user intent, decomposes tasks, and coordinates multiple agents, while traditional applications evolve into modular Skills-as-Modules enabling users to compose software through natural language rules. We argue that realizing AgentOS fundamentally becomes a Knowledge Discovery and Data Mining (KDD) problem. The Agent Kernel must operate as a real-time engine for intent mining and knowledge discovery. Viewed through this lens, the operating system becomes a continuous data mining pipeline involving sequential pattern mining for workflow automation, recommender systems for skill retrieval, and dynamically evolving personal knowledge graphs. These challenges define a new research agenda for the KDD community in building the next generation of intelligent computing systems. 

---
# Towards a Neural Debugger for Python 

**Authors**: Maximilian Beck, Jonas Gehring, Jannik Kossen, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2603.09951)  

**Abstract**: Training large language models (LLMs) on Python execution traces grounds them in code execution and enables the line-by-line execution prediction of whole Python programs, effectively turning them into neural interpreters (FAIR CodeGen Team et al., 2025). However, developers rarely execute programs step by step; instead, they use debuggers to stop execution at certain breakpoints and step through relevant portions only while inspecting or modifying program variables. Existing neural interpreter approaches lack such interactive control. To address this limitation, we introduce neural debuggers: language models that emulate traditional debuggers, supporting operations such as stepping into, over, or out of functions, as well as setting breakpoints at specific source lines. We show that neural debuggers -- obtained via fine-tuning large LLMs or pre-training smaller models from scratch -- can reliably model both forward execution (predicting future states and outputs) and inverse execution (inferring prior states or inputs) conditioned on debugger actions. Evaluated on CruxEval, our models achieve strong performance on both output and input prediction tasks, demonstrating robust conditional execution modeling. Our work takes first steps towards future agentic coding systems in which neural debuggers serve as a world model for simulated debugging environments, providing execution feedback or enabling agents to interact with real debugging tools. This capability lays the foundation for more powerful code generation, program understanding, and automated debugging. 

---
# MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning 

**Authors**: Yiyang Lu, Yu He, Jianlong Chen, Hongyuan Zha  

**Link**: [PDF](https://arxiv.org/pdf/2603.09892)  

**Abstract**: Continual fine-tuning of large language models (LLMs) is becoming increasingly crucial as these models are deployed in dynamic environments where tasks and data distributions evolve over time. While strong adaptability enables rapid acquisition of new knowledge, it also exposes LLMs to catastrophic forgetting, where previously learned skills degrade during sequential training. Existing replay-based strategies, such as fixed interleaved replay, accuracy-supervised, and loss-driven scheduling, remain limited: some depend on heuristic rules and provide only partial mitigation of forgetting, while others improve performance but incur substantial computational overhead. Motivated by retention dynamics under sequential fine-tuning, we propose Memory-Inspired Sampler and Scheduler Replay (MSSR), an experience replay framework that estimates sample-level memory strength and schedules rehearsal at adaptive intervals to mitigate catastrophic forgetting while maintaining fast adaptation. Extensive experiments across three backbone models and 11 sequential tasks show that MSSR consistently outperforms state-of-the-art replay baselines, with particularly strong gains on reasoning-intensive and multiple-choice benchmarks. 

---
# Understanding the Use of a Large Language Model-Powered Guide to Make Virtual Reality Accessible for Blind and Low Vision People 

**Authors**: Jazmin Collins, Sharon Y Lin, Tianqi Liu, Andrea Stevenson Won, Shiri Azenkot  

**Link**: [PDF](https://arxiv.org/pdf/2603.09964)  

**Abstract**: As social virtual reality (VR) grows more popular, addressing accessibility for blind and low vision (BLV) users is increasingly critical. Researchers have proposed an AI "sighted guide" to help users navigate VR and answer their questions, but it has not been studied with users. To address this gap, we developed a large language model (LLM)-powered guide and studied its use with 16 BLV participants in virtual environments with confederates posing as other users. We found that when alone, participants treated the guide as a tool, but treated it companionably around others, giving it nicknames, rationalizing its mistakes with its appearance, and encouraging confederate-guide interaction. Our work furthers understanding of guides as a versatile method for VR accessibility and presents design recommendations for future guides. 

---
# MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations 

**Authors**: Abhishikth Mallampalli, Sridhara Dasu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09800)  

**Abstract**: Large-scale scientific collaborations, such as the Compact Muon Solenoid (CMS) at CERN, produce a vast and ever-growing corpus of internal documentation. Navigating this complex information landscape presents a significant challenge for both new and experienced researchers, hindering knowledge sharing and slowing down the pace of scientific discovery. To address this, we present a prototype of MITRA, a Retrieval-Augmented Generation (RAG) based system, designed to answer specific, context-aware questions about physics analyses. MITRA employs a novel, automated pipeline using Selenium for document retrieval from internal databases and Optical Character Recognition (OCR) with layout parsing for high-fidelity text extraction. Crucially, MITRA's entire framework, from the embedding model to the Large Language Model (LLM), is hosted on-premise, ensuring that sensitive collaboration data remains private. We introduce a two-tiered vector database architecture that first identifies the relevant analysis from abstracts before focusing on the full documentation, resolving potential ambiguities between different analyses. We demonstrate the prototype's superior retrieval performance against a standard keyword-based baseline on realistic queries and discuss future work towards developing a comprehensive research agent for large experimental collaborations. 

---
# SCENEBench: An Audio Understanding Benchmark Grounded in Assistive and Industrial Use Cases 

**Authors**: Laya Iyer, Angelina Wang, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2603.09853)  

**Abstract**: Advances in large language models (LLMs) have enabled significant capabilities in audio processing, resulting in state-of-the-art models now known as Large Audio Language Models (LALMs). However, minimal work has been done to measure audio understanding beyond automatic speech recognition (ASR). This paper closes that gap by proposing a benchmark suite, SCENEBench (Spatial, Cross-lingual, Environmental, Non-speech Evaluation), that targets a broad form of audio comprehension across four real-world categories: background sound understanding, noise localization, cross-linguistic speech understanding, and vocal characterizer recognition. These four categories are selected based on understudied needs from accessibility technology and industrial noise monitoring. In addition to performance, we also measure model latency. The purpose of this benchmark suite is to assess audio beyond just what words are said - rather, how they are said and the non-speech components of the audio. Because our audio samples are synthetically constructed (e.g., by overlaying two natural audio samples), we further validate our benchmark against 20 natural audio items per task, sub-sampled from existing datasets to match our task criteria, to assess ecological validity. We assess five state-of-the-art LALMs and find critical gaps: performance varies across tasks, with some tasks performing below random chance and others achieving high accuracy. These results provide direction for targeted improvements in model capabilities. 

---
# ESAinsTOD: A Unified End-to-End Schema-Aware Instruction-Tuning Framework for Task-Oriented Dialog Modeling 

**Authors**: Dechuan Teng, Chunlin Lu, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2603.09691)  

**Abstract**: Existing end-to-end modeling methods for modular task-oriented dialog systems are typically tailored to specific datasets, making it challenging to adapt to new dialog scenarios. In this work, we propose ESAinsTOD, a unified End-to-end Schema-Aware Instruction-tuning framework for general Task-Oriented Dialog modeling. This framework introduces a structured methodology to go beyond simply fine-tuning Large Language Models (LLMs), enabling flexible adaptation to various dialogue task flows and schemas. Specifically, we leverage full-parameter fine-tuning of LLMs and introduce two alignment mechanisms to make the resulting system both instruction-aware and schema-aware: (i) instruction alignment, which ensures that the system faithfully follows task instructions to complete various task flows from heterogeneous TOD datasets; and (ii) schema alignment, which encourages the system to make predictions adhering to the specified schema. In addition, we employ session-level end-to-end modeling, which allows the system to access the results of previously executed task flows within the dialogue history, to bridge the gap between the instruction-tuning paradigm and the real-world application of TOD systems. Empirical results show that while a fine-tuned LLM serves as a strong baseline, our structured approach provides significant additional benefits. In particular, our findings indicate that: (i) ESAinsTOD outperforms state-of-the-art models by a significant margin on end-to-end task-oriented dialog modeling benchmarks: CamRest676, In-Car and MultiWOZ; (ii) more importantly, it exhibits superior generalization capabilities across various low-resource settings, with the proposed alignment mechanisms significantly enhancing zero-shot performance; and (iii) our instruction-tuning paradigm substantially improves the model's robustness against data noise and cascading errors. 

---
# Common Sense vs. Morality: The Curious Case of Narrative Focus Bias in LLMs 

**Authors**: Saugata Purkayastha, Pranav Kushare, Pragya Paramita Pal, Sukannya Purkayastha  

**Link**: [PDF](https://arxiv.org/pdf/2603.09434)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse real-world applications and user communities. As such, it is crucial that these models remain both morally grounded and knowledge-aware. In this work, we uncover a critical limitation of current LLMs -- their tendency to prioritize moral reasoning over commonsense understanding. To investigate this phenomenon, we introduce CoMoral, a novel benchmark dataset containing commonsense contradictions embedded within moral dilemmas. Through extensive evaluation of ten LLMs across different model sizes, we find that existing models consistently struggle to identify such contradictions without prior signal. Furthermore, we observe a pervasive narrative focus bias, wherein LLMs more readily detect commonsense contradictions when they are attributed to a secondary character rather than the primary (narrator) character. Our comprehensive analysis underscores the need for enhanced reasoning-aware training to improve the commonsense robustness of large language models. 

---
# Investigating Gender Stereotypes in Large Language Models via Social Determinants of Health 

**Authors**: Trung Hieu Ngo, Adrien Bazoge, Solen Quiniou, Pierre-Antoine Gourraud, Emmanuel Morin  

**Link**: [PDF](https://arxiv.org/pdf/2603.09416)  

**Abstract**: Large Language Models (LLMs) excel in Natural Language Processing (NLP) tasks, but they often propagate biases embedded in their training data, which is potentially impactful in sensitive domains like healthcare. While existing benchmarks evaluate biases related to individual social determinants of health (SDoH) such as gender or ethnicity, they often overlook interactions between these factors and lack context-specific assessments. This study investigates bias in LLMs by probing the relationships between gender and other SDoH in French patient records. Through a series of experiments, we found that embedded stereotypes can be probed using SDoH input and that LLMs rely on embedded stereotypes to make gendered decisions, suggesting that evaluating interactions among SDoH factors could usefully complement existing approaches to assessing LLM performance and bias. 

---
# Beyond Scaling: Assessing Strategic Reasoning and Rapid Decision-Making Capability of LLMs in Zero-sum Environments 

**Authors**: Yang Li, Xing Chen, Yutao Liu, Gege Qi, Yanxian BI, Zizhe Wang, Yunjian Zhang, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09337)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance on static reasoning benchmarks, yet their effectiveness as interactive agents operating in adversarial, time-sensitive environments remains poorly understood. Existing evaluations largely treat reasoning as a single-shot capability, overlooking the challenges of opponent-aware decision-making, temporal constraints, and execution under pressure. This paper introduces Strategic Tactical Agent Reasoning (STAR) Benchmark, a multi-agent evaluation framework that assesses LLMs through 1v1 zero-sum competitive interactions, framing reasoning as an iterative, adaptive decision-making process. STAR supports both turn-based and real-time settings, enabling controlled analysis of long-horizon strategic planning and fast-paced tactical execution within a unified environment. Built on a modular architecture with a standardized API and fully implemented execution engine, STAR facilitates reproducible evaluation and flexible task customization. To move beyond binary win-loss outcomes, we introduce a Strategic Evaluation Suite that assesses not only competitive success but also the quality of strategic behavior, such as execution efficiency and outcome stability. Extensive pairwise evaluations reveal a pronounced strategy-execution gap: while reasoning-intensive models dominate turn-based settings, their inference latency often leads to inferior performance in real-time scenarios, where faster instruction-tuned models prevail. These results show that strategic intelligence in interactive environments depends not only on reasoning depth, but also on the ability to translate plans into timely actions, positioning STAR as a principled benchmark for studying this trade-off in competitive, dynamic settings. 

---
# Reading the Mood Behind Words: Integrating Prosody-Derived Emotional Context into Socially Responsive VR Agents 

**Authors**: SangYeop Jeong, Yeongseo Na, Seung Gyu Jeong, Jin-Woo Jeong, Seong-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.09324)  

**Abstract**: In VR interactions with embodied conversational agents, users' emotional intent is often conveyed more by how something is said than by what is said. However, most VR agent pipelines rely on speech-to-text processing, discarding prosodic cues and often producing emotionally incongruent responses despite correct semantics. We propose an emotion-context-aware VR interaction pipeline that treats vocal emotion as explicit dialogue context in an LLM-based conversational agent. A real-time speech emotion recognition model infers users' emotional states from prosody, and the resulting emotion labels are injected into the agent's dialogue context to shape response tone and style. Results from a within-subjects VR study (N=30) show significant improvements in dialogue quality, naturalness, engagement, rapport, and human-likeness, with 93.3% of participants preferring the emotion-aware agent. 

---
# TaSR-RAG: Taxonomy-guided Structured Reasoning for Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Yixuan Xie, Jimeng Shi, Shaowen Wang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.09341)  

**Abstract**: Retrieval-Augmented Generation (RAG) helps large language models (LLMs) answer knowledge-intensive and time-sensitive questions by conditioning generation on external evidence. However, most RAG systems still retrieve unstructured chunks and rely on one-shot generation, which often yields redundant context, low information density, and brittle multi-hop reasoning. While structured RAG pipelines can improve grounding, they typically require costly and error-prone graph construction or impose rigid entity-centric structures that do not align with the query's reasoning chain.
We propose \textsc{TaSR-RAG}, a taxonomy-guided structured reasoning framework for evidence selection. We represent both queries and documents as relational triples, and constrain entity semantics with a lightweight two-level taxonomy to balance generalization and precision. Given a complex question, \textsc{TaSR-RAG} decomposes it into an ordered sequence of triple sub-queries with explicit latent variables, then performs step-wise evidence selection via hybrid triple matching that combines semantic similarity over raw triples with structural consistency over typed triples.
By maintaining an explicit entity binding table across steps, \textsc{TaSR-RAG} resolves intermediate variables and reduces entity conflation without explicit graph construction or exhaustive search. Experiments on multiple multi-hop question answering benchmarks show that \textsc{TaSR-RAG} consistently outperforms strong RAG and structured-RAG baselines by up to 14\%, while producing clearer evidence attribution and more faithful reasoning traces. 

---
# DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization 

**Authors**: Jianing Yang, Yusuke Fujita, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2603.09180)  

**Abstract**: Spoken dialog systems with cascaded ASR-LLM-TTS modules retain strong LLM intelligence, but VAD segmentation often forces half-duplex turns and brittle control. On the other hand, VAD-free end-to-end model support full-duplex interaction but is hard to maintain conversational intelligence. In this paper, we present DuplexCascade, a VAD-free cascaded streaming pipeline for full-duplex speech-to-speech dialogue. Our key idea is to convert conventional utterance-wise long turns into chunk-wise micro-turn interactions, enabling rapid bidirectional exchange while preserving the strengths of a capable text LLM. To reliably coordinate turn-taking and response timing, we introduce a set of conversational special control tokens that steer the LLM's behavior under streaming constraints. On Full-DuplexBench and VoiceBench, DuplexCascade delivers state-of-the-art full-duplex turn-taking and strong conversational intelligence among open-source speech-to-speech dialogue systems. 

---
# Emotion is Not Just a Label: Latent Emotional Factors in LLM Processing 

**Authors**: Benjamin Reichman, Adar Avasian, Samuel Webster, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2603.09205)  

**Abstract**: Large language models are routinely deployed on text that varies widely in emotional tone, yet their reasoning behavior is typically evaluated without accounting for emotion as a source of representational variation. Prior work has largely treated emotion as a prediction target, for example in sentiment analysis or emotion classification. In contrast, we study emotion as a latent factor that shapes how models attend to and reason over text. We analyze how emotional tone systematically alters attention geometry in transformer models, showing that metrics such as locality, center-of-mass distance, and entropy vary across emotions and correlate with downstream question-answering performance. To facilitate controlled study of these effects, we introduce Affect-Uniform ReAding QA (AURA-QA), a question-answering dataset with emotionally balanced, human-authored context passages. Finally, an emotional regularization framework is proposed that constrains emotion-conditioned representational drift during training. Experiments across multiple QA benchmarks demonstrate that this approach improves reading comprehension in both emotionally-varying and non-emotionally varying datasets, yielding consistent gains under distribution shift and in-domain improvements on several benchmarks. 

---
# Reinforced Generation of Combinatorial Structures: Ramsey Numbers 

**Authors**: Ansh Nagda, Prabhakar Raghavan, Abhradeep Thakurta  

**Link**: [PDF](https://arxiv.org/pdf/2603.09172)  

**Abstract**: We present improved lower bounds for five classical Ramsey numbers: $\mathbf{R}(3, 13)$ is increased from $60$ to $61$, $\mathbf{R}(3, 18)$ from $99$ to $100$, $\mathbf{R}(4, 13)$ from $138$ to $139$, $\mathbf{R}(4, 14)$ from $147$ to $148$, and $\mathbf{R}(4, 15)$ from $158$ to $159$. These results were achieved using~\emph{AlphaEvolve}, an LLM-based code mutation agent. Beyond these new results, we successfully recovered lower bounds for all Ramsey numbers known to be exact, and matched the best known lower bounds across many other cases. These include bounds for which previous work does not detail the algorithms used. Virtually all known Ramsey lower bounds are derived computationally, with bespoke search algorithms each delivering a handful of results. AlphaEvolve is a single meta-algorithm yielding search algorithms for all of our results. 

---
# VIVID-Med: LLM-Supervised Structured Pretraining for Deployable Medical ViTs 

**Authors**: Xiyao Wang, Xiaoyu Tan, Yang Dai, Yuxuan Fu, Shuo Li, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09109)  

**Abstract**: Vision-language pretraining has driven significant progress in medical image analysis. However, current methods typically supervise visual encoders using one-hot labels or free-form text, neither of which effectively captures the complex semantic relationships among clinical findings. In this study, we introduce VIVID-Med, a novel framework that leverages a frozen large language model (LLM) as a structured semantic teacher to pretrain medical vision transformers (ViTs). VIVID-Med translates clinical findings into verifiable JSON field-state pairs via a Unified Medical Schema (UMS), utilizing answerability-aware masking to focus optimization. It then employs Structured Prediction Decomposition (SPD) to partition cross-attention into orthogonality-regularized query groups, extracting complementary visual aspects. Crucially, the LLM is discarded post-training, yielding a lightweight, deployable ViT-only backbone. We evaluated VIVID-Med across multiple settings: on CheXpert linear probing, it achieves a macro-AUC of 0.8588, outperforming BiomedCLIP by +6.65 points while using 500x less data. It also demonstrates robust zero-shot cross-domain transfer to NIH ChestX-ray14 (0.7225 macro-AUC) and strong cross-modality generalization to CT, achieving 0.8413 AUC on LIDC-IDRI lung nodule classification and 0.9969 macro-AUC on OrganAMNIST 11-organ classification. VIVID-Med offers a highly efficient, scalable alternative to deploying resource-heavy vision-language models in clinical settings. 

---
# Automating Detection and Root-Cause Analysis of Flaky Tests in Quantum Software 

**Authors**: Janakan Sivaloganathan, Ainaz Jamshidi, Andriy Miranskyy, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09029)  

**Abstract**: Like classical software, quantum software systems rely on automated testing. However, their inherently probabilistic outputs make them susceptible to quantum flakiness -- tests that pass or fail inconsistently without code changes. Such quantum flaky tests can mask real defects and reduce developer productivity, yet systematic tooling for their detection and diagnosis remains limited.
This paper presents an automated pipeline to detect flaky-test-related issues and pull requests in quantum software repositories and to support the identification of their root causes. We aim to expand an existing quantum flaky test dataset and evaluate the capability of Large Language Models (LLMs) for flakiness classification and root-cause identification.
Building on a prior manual analysis of 14 quantum software repositories, we automate the discovery of additional flaky test cases using LLMs and cosine similarity. We further evaluate a variety of LLMs from OpenAI GPT, Meta LLaMA, Google Gemini, and Anthropic Claude suites for classifying flakiness and identifying root causes from issue descriptions and code context. Classification performance is assessed using standard performance metrics, including F1-score.
Using our pipeline, we identify 25 previously unknown flaky tests, increasing the original dataset size by 54%. The best-performing model, Google Gemini, achieves an F1-score of 0.9420 for flakiness detection and 0.9643 for root-cause identification, demonstrating that LLMs can provide practical support for triaging flaky reports and understanding their underlying causes in quantum software.
The expanded dataset and automated pipeline provide reusable artifacts for the quantum software engineering community. Future work will focus on improving detection robustness and exploring automated repair of quantum flaky tests. 

---
# The Missing Memory Hierarchy: Demand Paging for LLM Context Windows 

**Authors**: Tony Mason  

**Link**: [PDF](https://arxiv.org/pdf/2603.09023)  

**Abstract**: The context window of a large language model is not memory. It is L1 cache: a small, fast, expensive resource that the field treats as the entire memory system. There is no L2, no virtual memory, no paging. Every tool definition, every system prompt, and every stale tool result occupies context for the lifetime of the session. The result is measurable: across 857 production sessions and 4.45 million effective input tokens, 21.8% is structural waste.
We present Pichay, a demand paging system for LLM context windows. Implemented as a transparent proxy between client and inference API, Pichay interposes on the message stream to evict stale content, detect page faults when the model re-requests evicted material, and pin working-set pages identified by fault history. In offline replay across 1.4 million simulated evictions, the fault rate is 0.0254%. In live production deployment over 681turns, the system reduces context consumption by up to 93% (5,038KB to 339KB); under extreme sustained pressure, the system remains operational but exhibits the expected thrashing pathology, with repeated fault-in of evicted content.
The key observation is that the problems the field faces, such as context limits, attention degradation, cost scaling, lost state across sessions, are virtual memory problems wearing different clothes. The solutions exist: working set theory (Denning, 1968), demand paging, fault-driven replacement policies, and memory hierarchies with multiple eviction-managed levels. We describe the architecture of a full memory hierarchy for LLM systems (L1 through persistent storage), report on the first three levels deployed in production use (L1 eviction, L2 fault-driven pinning, L3 model-initiated conversation compaction), and identify cross-session memory as the remaining frontier. 

---
# PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration 

**Authors**: Abdul Rehman Akbar, Samuel Wales-McGrath, Alejadro Levya, Lina Gokhale, Rajendra Singh, Wei Chen, Anil Parwani, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2603.08935)  

**Abstract**: Pathology underpins modern diagnosis and cancer care, yet its most valuable asset, the accumulated experience encoded in millions of narrative reports, remains largely inaccessible. Although institutions are rapidly digitizing pathology workflows, storing data without effective mechanisms for retrieval and reasoning risks transforming archives into a passive data repository, where institutional knowledge exists but cannot meaningfully inform patient care. True progress requires not only digitization, but the ability for pathologists to interrogate prior similar cases in real time while evaluating a new diagnostic dilemma. We present PathoScribe, a unified retrieval-augmented large language model (LLM) framework designed to transform static pathology archives into a searchable, reasoning-enabled living library. PathoScribe enables natural language case exploration, automated cohort construction, clinical question answering, immunohistochemistry (IHC) panel recommendation, and prompt-controlled report transformation within a single architecture. Evaluated on 70,000 multi-institutional surgical pathology reports, PathoScribe achieved perfect Recall@10 for natural language case retrieval and demonstrated high-quality retrieval-grounded reasoning (mean reviewer score 4.56/5). Critically, the system operationalized automated cohort construction from free-text eligibility criteria, assembling research-ready cohorts in minutes (mean 9.2 minutes) with 91.3% agreement to human reviewers and no eligible cases incorrectly excluded, representing orders-of-magnitude reductions in time and cost compared to traditional manual chart review. This work establishes a scalable foundation for converting digital pathology archives from passive storage systems into active clinical intelligence platforms. 

---
# VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs 

**Authors**: Hezhao Zhang, Huang-Cheng Chou, Shrikanth Narayanan, Thomas Hain  

**Link**: [PDF](https://arxiv.org/pdf/2603.08936)  

**Abstract**: Speech Large Language Models (LLMs) show great promise for speech emotion recognition (SER) via generative interfaces. However, shifting from closed-set classification to open text generation introduces zero-shot stochasticity, making evaluation highly sensitive to prompts. Additionally, conventional speech LLMs benchmarks overlook the inherent ambiguity of human emotion. Hence, we present VoxEmo, a comprehensive SER benchmark encompassing 35 emotion corpora across 15 languages for Speech LLMs. VoxEmo provides a standardized toolkit featuring varying prompt complexities, from direct classification to paralinguistic reasoning. To reflect real-world perception/application, we introduce a distribution-aware soft-label protocol and a prompt-ensemble strategy that emulates annotator disagreement. Experiments reveal that while zero-shot speech LLMs trail supervised baselines in hard-label accuracy, they uniquely align with human subjective distributions. 

---
# Large Language Model-Assisted Superconducting Qubit Experiments 

**Authors**: Shiheng Li, Jacob M. Miller, Phoebe J. Lee, Gustav Andersson, Christopher R. Conner, Yash J. Joshi, Bayan Karimi, Amber M. King, Howard L. Malc, Harsh Mishra, Hong Qiao, Minseok Ryu, Xuntao Wu, Siyuan Xing, Haoxiong Yan, Jian Shi, Andrew N. Cleland  

**Link**: [PDF](https://arxiv.org/pdf/2603.08801)  

**Abstract**: Superconducting circuits have demonstrated significant potential in quantum information processing and quantum sensing. Implementing novel control and measurement sequences for superconducting qubits is often a complex and time-consuming process, requiring extensive expertise in both the underlying physics and the specific hardware and software. In this work, we introduce a framework that leverages a large language model (LLM) to automate qubit control and measurement. Specifically, our framework conducts experiments by generating and invoking schema-less tools on demand via a knowledge base on instrumental usage and experimental procedures. We showcase this framework with two experiments: an autonomous resonator characterization and a direct reproduction of a quantum non-demolition (QND) characterization of a superconducting qubit from literature. This framework enables rapid deployment of standard control-and-measurement protocols and facilitates implementation of novel experimental procedures, offering a more flexible and user-friendly paradigm for controlling complex quantum hardware. 

---
# Scale-Plan: Scalable Language-Enabled Task Planning for Heterogeneous Multi-Robot Teams 

**Authors**: Piyush Gupta, Sangjae Bae, Jiachen Li, David Isele  

**Link**: [PDF](https://arxiv.org/pdf/2603.08814)  

**Abstract**: Long-horizon task planning for heterogeneous multi-robot systems is essential for deploying collaborative teams in real-world environments; yet, it remains challenging due to the large volume of perceptual information, much of which is irrelevant to task objectives and burdens planning. Traditional symbolic planners rely on manually constructed problem specifications, limiting scalability and adaptability, while recent large language model (LLM)-based approaches often suffer from hallucinations and weak grounding-i.e., poor alignment between generated plans and actual environmental objects and constraints-in object-rich settings. We present Scale-Plan, a scalable LLM-assisted framework that generates compact, task-relevant problem representations from natural language instructions. Given a PDDL domain specification, Scale-Plan constructs an action graph capturing domain structure and uses shallow LLM reasoning to guide a structured graph search that identifies a minimal subset of relevant actions and objects. By filtering irrelevant information prior to planning, Scale-Plan enables efficient decomposition, allocation, and long-horizon plan generation. We evaluate our approach on complex multi-agent tasks and introduce MAT2-THOR, a cleaned benchmark built on AI2-THOR for reliable evaluation of multi-robot planning systems. Scale-Plan outperforms pure LLM and hybrid LLM-PDDL baselines across all metrics, improving scalability and reliability. 

---
# Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention 

**Authors**: Mengqi Liao, Lu Wang, Chaoyun Zhang, Bo Qiao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2603.08743)  

**Abstract**: With reasoning becoming the generative paradigm for large language models (LLMs), the memory bottleneck caused by KV cache during the decoding phase has become a critical factor limiting high-concurrency service. Although existing KV cache eviction methods address the memory issue, most of them are impractical for industrial-grade applications. This paper introduces Compressed PagedAttention, a method that combines token-wise KV cache eviction with PagedAttention. We propose a comprehensive scheduling strategy and support prefix caching and asynchronous compression for Compressed PagedAttention. Based on this, we have developed a high-concurrency LLM inference engine, Zipage. On large-scale mathematical reasoning tasks, Zipage achieves around 95\% of the performance of Full KV inference engines while delivering over 2.1$\times$ speedup. 

---
# Turn: A Language for Agentic Computation 

**Authors**: Muyukani Kizito  

**Link**: [PDF](https://arxiv.org/pdf/2603.08755)  

**Abstract**: We present \textbf{Turn}, a compiled, actor-based programming language -- statically typed for schema inference, dynamically typed at the value level -- for agentic software: programs that reason and act autonomously by delegating inference to large language models (LLMs). Existing approaches augment general-purpose languages with frameworks, encoding critical invariants (bounded context, typed inference output, credential isolation, durable state) as application-level conventions rather than language guarantees.
Turn introduces five language-level constructs that address this gap. \emph{Cognitive Type Safety} makes LLM inference a typed primitive: the compiler generates a JSON Schema from a struct definition and the VM validates model output before binding. The \emph{confidence operator} enables deterministic control flow gated on model certainty. Turn's \emph{actor-based process model}, derived from Erlang, gives each agent an isolated context window, persistent memory, and mailbox. A \emph{capability-based identity system} returns opaque, unforgeable handles from the VM host, ensuring raw credentials never enter agent memory. Finally, \emph{compile-time schema absorption} (\texttt{use schema::<protocol>}) synthesizes typed API bindings from external specifications at compile time; the \texttt{openapi} adapter is shipped with \texttt{graphql}, \texttt{fhir}, and \texttt{mcp} in active development.
We describe the language design, type rules, schema semantics, and a Rust-based bytecode VM, and evaluate Turn against representative agentic workloads. Turn is open source at this https URL. 

---
# ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget for Long-Context Inference in LLMs 

**Authors**: Jianlong Lei, Shashikant Ilager  

**Link**: [PDF](https://arxiv.org/pdf/2603.08727)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in scenarios demanding ultra-long context reasoning, such as agentic workflows and deep research understanding. However, long-context inference is constrained by the KV cache, a transient memory structure that grows linearly with sequence length and batch size, quickly dominating GPU memory usage. Existing memory reduction techniques, including eviction and quantization, often rely on static heuristics and suffer from degraded quality under tight budgets. In this paper, we propose ARKV, a lightweight and adaptive framework that dynamically allocates precision levels to cached tokens based on per-layer attention dynamics and token-level importance. During a short prefill phase, ARKV estimates the original quantization (OQ) ratio of each layer by computing statistical scores such as attention entropy, variance and kurtosis. During decoding, tokens are assigned to one of three states, Original (full precision), Quantization (low precision), or Eviction, according to a fast heavy-hitter scoring strategy. Our experiments on LLaMA3 and Qwen3 models across diverse long- and short-context tasks demonstrate that ARKV preserves ~97% of baseline accuracy on long-context benchmarks while reducing KV memory usage by 4x, with minimal throughput loss. On short-context tasks, ARKV matches full-precision baselines; on GSM8K math reasoning, it significantly outperforms uniform quantization. These results highlight the practical viability of ARKV for scalable LLM deployment, offering fine-grained, data-driven memory control without retraining or architectural modifications. The source code and artifacts can be found in: this https URL 

---
# Let's Verify Math Questions Step by Step 

**Authors**: Chengyu Shen, Zhen Hao Wong, Runming He, Hao Liang, Meiyi Qiang, Zimo Meng, Zhengyang Zhao, Bohan Zeng, Zhengzhou Zhu, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13903)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable progress in mathematical reasoning. To enable such capabilities, many existing works distill strong reasoning models into long chains of thought or design algorithms to construct high-quality math QA data for training. However, these efforts primarily focus on generating correct reasoning paths and answers, while largely overlooking the validity of the questions themselves. In this work, we propose Math Question Verification (MathQ-Verify), a novel five-stage pipeline designed to rigorously filter ill-posed or under-specified math problems. MathQ-Verify first performs format-level validation to remove redundant instructions and ensure that each question is syntactically well-formed. It then formalizes each question, decomposes it into atomic conditions, and verifies them against mathematical definitions. Next, it detects logical contradictions among these conditions, followed by a goal-oriented completeness check to ensure the question provides sufficient information for solving. To evaluate this task, we use existing benchmarks along with an additional dataset we construct, containing 2,147 math questions with diverse error types, each manually double-validated. Experiments show that MathQ-Verify achieves state-of-the-art performance across multiple benchmarks, improving the F1 score by up to 25 percentage points over the direct verification baseline. It further attains approximately 90% precision and 63% recall through a lightweight model voting scheme. MathQ-Verify offers a scalable and accurate solution for curating reliable mathematical datasets, reducing label noise and avoiding unnecessary computation on invalid questions. Our code and data are available at this https URL. 

---
# Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.09938)  

**Abstract**: Model merging has emerged as a transformative paradigm for combining the capabilities of multiple neural networks into a single unified model without additional training. With the rapid proliferation of fine-tuned large language models~(LLMs), merging techniques offer a computationally efficient alternative to ensembles and full retraining, enabling practitioners to compose specialized capabilities at minimal cost. This survey presents a comprehensive and structured examination of model merging in the LLM era through the \textbf{FUSE} taxonomy, a four-dimensional framework organized along \textbf{F}oundations, \textbf{U}nification Strategies, \textbf{S}cenarios, and \textbf{E}cosystem. We first establish the theoretical underpinnings of merging, including loss landscape geometry, mode connectivity, and the linear mode connectivity hypothesis. We then systematically review the algorithmic landscape, spanning weight averaging, task vector arithmetic, sparsification-enhanced methods, mixture-of-experts architectures, and evolutionary optimization approaches. For each method family, we analyze the core formulation, highlight representative works, and discuss practical trade-offs. We further examine downstream applications across multi-task learning, safety alignment, domain specialization, multilingual transfer, and federated learning. Finally, we survey the supporting ecosystem of open-source tools, community platforms, and evaluation benchmarks, and identify key open challenges including theoretical gaps, scalability barriers, and standardization needs. This survey aims to equip researchers and practitioners with a structured foundation for advancing model merging. 

---
# CREATE: Testing LLMs for Associative Creativity 

**Authors**: Manya Wadhwa, Tiasa Singha Roy, Harvey Lederman, Junyi Jessy Li, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2603.09970)  

**Abstract**: A key component of creativity is associative reasoning: the ability to draw novel yet meaningful connections between concepts. We introduce CREATE, a benchmark designed to evaluate models' capacity for creative associative reasoning. CREATE requires models to generate sets of paths connecting concepts in a model's parametric knowledge. Paths should have high specificity (distinctiveness and closeness of the concept connection) and high diversity (dissimilarity from other paths), and models are scored more highly if they produce a larger set of strong, diverse paths. This task shares demands of real creativity tasks like hypothesis generation, including an extremely large search space, but enables collection of a sizable benchmark with objective answer grading. Evaluation of frontier models shows that the strongest models achieve higher creative utility than others, with the high multiplicity of answers and complexity of the search making benchmark saturation difficult to achieve. Furthermore, our results illustrate that thinking models are not always more effective on our task, even with high token budgets. Recent approaches for creative prompting give some but limited additional improvement. CREATE provides a sandbox for developing new methods to improve models' capacity for associative creativity. 

---
# Do What I Say: A Spoken Prompt Dataset for Instruction-Following 

**Authors**: Maike Züfle, Sara Papi, Fabian Retkowski, Szymon Mazurek, Marek Kasztelnik, Alexander Waibel, Luisa Bentivogli, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2603.09881)  

**Abstract**: Speech Large Language Models (SLLMs) have rapidly expanded, supporting a wide range of tasks. These models are typically evaluated using text prompts, which may not reflect real-world scenarios where users interact with speech. To address this gap, we introduce DoWhatISay (DOWIS), a multilingual dataset of human-recorded spoken and written prompts designed to pair with any existing benchmark for realistic evaluation of SLLMs under spoken instruction conditions. Spanning 9 tasks and 11 languages, it provides 10 prompt variants per task-language pair, across five styles. Using DOWIS, we benchmark state-of-the-art SLLMs, analyzing the interplay between prompt modality, style, language, and task type. Results show that text prompts consistently outperform spoken prompts, particularly for low-resource and cross-lingual settings. Only for tasks with speech output, spoken prompts do close the gap, highlighting the need for speech-based prompting in SLLM evaluation. 

---
# One-Eval: An Agentic System for Automated and Traceable LLM Evaluation 

**Authors**: Chengyu Shen, Yanheng Hou, Minghui Pan, Runming He, Zhen Hao Wong, Meiyi Qiang, Zhou Liu, Hao Liang, Peichao Lai, Zeang Sheng, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09821)  

**Abstract**: Reliable evaluation is essential for developing and deploying large language models, yet in practice it often requires substantial manual effort: practitioners must identify appropriate benchmarks, reproduce heterogeneous evaluation codebases, configure dataset schema mappings, and interpret aggregated metrics. To address these challenges, we present One-Eval, an agentic evaluation system that converts natural-language evaluation requests into executable, traceable, and customizable evaluation workflows. One-Eval integrates (i) NL2Bench for intent structuring and personalized benchmark planning, (ii) BenchResolve for benchmark resolution, automatic dataset acquisition, and schema normalization to ensure executability, and (iii) Metrics \& Reporting for task-aware metric selection and decision-oriented reporting beyond scalar scores. The system further incorporates human-in-the-loop checkpoints for review, editing, and rollback, while preserving sample evidence trails for debugging and auditability. Experiments show that One-Eval can execute end-to-end evaluations from diverse natural-language requests with minimal user effort, supporting more efficient and reproducible evaluation in industrial settings. Our framework is publicly available at this https URL. 

---
# Evaluation of LLMs in retrieving food and nutritional context for RAG systems 

**Authors**: Maks Požarnik Vavken, Matevž Ogrinc, Tome Eftimov, Barbara Koroušić Seljak  

**Link**: [PDF](https://arxiv.org/pdf/2603.09704)  

**Abstract**: In this article, we evaluate four Large Language Models (LLMs) and their effectiveness at retrieving data within a specialized Retrieval-Augmented Generation (RAG) system, using a comprehensive food composition database. Our method is focused on the LLMs ability to translate natural language queries into structured metadata filters, enabling efficient retrieval via a Chroma vector database. By achieving high accuracy in this critical retrieval step, we demonstrate that LLMs can serve as an accessible, high-performance tool, drastically reducing the manual effort and technical expertise previously required for domain experts, such as food compilers and nutritionists, to leverage complex food and nutrition data. However, despite the high performance on easy and moderately complex queries, our analysis of difficult questions reveals that reliable retrieval remains challenging when queries involve non-expressible constraints. These findings demonstrate that LLM-driven metadata filtering excels when constraints can be explicitly expressed, but struggles when queries exceed the representational scope of the metadata format. 

---
# Tracking Cancer Through Text: Longitudinal Extraction From Radiology Reports Using Open-Source Large Language Models 

**Authors**: Luc Builtjes, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2603.09638)  

**Abstract**: Radiology reports capture crucial longitudinal information on tumor burden, treatment response, and disease progression, yet their unstructured narrative format complicates automated analysis. While large language models (LLMs) have advanced clinical text processing, most state-of-the-art systems remain proprietary, limiting their applicability in privacy-sensitive healthcare environments. We present a fully open-source, locally deployable pipeline for longitudinal information extraction from radiology reports, implemented using the \texttt{llm\_extractinator} framework. The system applies the \texttt{qwen2.5-72b} model to extract and link target, non-target, and new lesion data across time points in accordance with RECIST criteria. Evaluation on 50 Dutch CT Thorax/Abdomen report pairs yielded high extraction performance, with attribute-level accuracies of 93.7\% for target lesions, 94.9\% for non-target lesions, and 94.0\% for new lesions. The approach demonstrates that open-source LLMs can achieve clinically meaningful performance in multi-timepoint oncology tasks while ensuring data privacy and reproducibility. These results highlight the potential of locally deployable LLMs for scalable extraction of structured longitudinal data from routine clinical text. 

---
# LLM as a Meta-Judge: Synthetic Data for NLP Evaluation Metric Validation 

**Authors**: Lukáš Eigler, Jindřich Libovický, David Hurych  

**Link**: [PDF](https://arxiv.org/pdf/2603.09403)  

**Abstract**: Validating evaluation metrics for NLG typically relies on expensive and time-consuming human annotations, which predominantly exist only for English datasets. We propose \textit{LLM as a Meta-Judge}, a scalable framework that utilizes LLMs to generate synthetic evaluation datasets via controlled semantic degradation of real data, replacing human judgment. We validate our approach using \textit{meta-correlation}, measuring the alignment between metric rankings derived from synthetic data and those from standard human benchmarks. Experiments across Machine Translation, Question Answering, and Summarization demonstrate that synthetic validation serves as a reliable proxy for human judgment, achieving meta-correlations exceeding 0.9 in multilingual QA and proves to be a viable alternative where human judgments are unavailable or too expensive to obtain. Our code and data will become publicly available upon paper acceptance. 

---
# Quantifying and extending the coverage of spatial categorization data sets 

**Authors**: Wanchun Li, Alexandra Carstensen, Yang Xu, Terry Regier, Charles Kemp  

**Link**: [PDF](https://arxiv.org/pdf/2603.09373)  

**Abstract**: Variation in spatial categorization across languages is often studied by eliciting human labels for the relations depicted in a set of scenes known as the Topological Relations Picture Series (TRPS). We demonstrate that labels generated by large language models (LLMs) align relatively well with human labels, and show how LLM-generated labels can help to decide which scenes and languages to add to existing spatial data sets. To illustrate our approach we extend the TRPS by adding 42 new scenes, and show that this extension achieves better coverage of the space of possible scenes than two previous extensions of the TRPS. Our results provide a foundation for scaling towards spatial data sets with dozens of languages and hundreds of scenes. 

---
# Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs 

**Authors**: Zorik Gekhman, Roee Aharoni, Eran Ofek, Mor Geva, Roi Reichart, Jonathan Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2603.09906)  

**Abstract**: While reasoning in LLMs plays a natural role in math, code generation, and multi-hop factual questions, its effect on simple, single-hop factual questions remains unclear. Such questions do not require step-by-step logical decomposition, making the utility of reasoning highly counterintuitive. Nevertheless, we find that enabling reasoning substantially expands the capability boundary of the model's parametric knowledge recall, unlocking correct answers that are otherwise effectively unreachable. Why does reasoning aid parametric knowledge recall when there are no complex reasoning steps to be done? To answer this, we design a series of hypothesis-driven controlled experiments, and identify two key driving mechanisms: (1) a computational buffer effect, where the model uses the generated reasoning tokens to perform latent computation independent of their semantic content; and (2) factual priming, where generating topically related facts acts as a semantic bridge that facilitates correct answer retrieval. Importantly, this latter generative self-retrieval mechanism carries inherent risks: we demonstrate that hallucinating intermediate facts during reasoning increases the likelihood of hallucinations in the final answer. Finally, we show that our insights can be harnessed to directly improve model accuracy by prioritizing reasoning trajectories that contain hallucination-free factual statements. 

---
# Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG 

**Authors**: Jan Drole, Ana Gjorgjevikj, Barbara Korouši'c Seljak, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2603.09758)  

**Abstract**: Standardizing food terms from product labels and menus into ontology concepts is a prerequisite for trustworthy dietary assessment and safety reporting. The dominant approach to Named Entity Linking (NEL) in the food and nutrition domains fine-tunes Large Language Models (LLMs) on task-specific corpora. Although effective, fine-tuning incurs substantial computational cost, ties models to a particular ontology snapshot (i.e., version), and degrades under ontology drift. This paper presents FoodOntoRAG, a model- and ontology-agnostic pipeline that performs few-shot NEL by retrieving candidate entities from domain ontologies and conditioning an LLM on structured evidence (food labels, synonyms, definitions, and relations). A hybrid lexical--semantic retriever enumerates candidates; a selector agent chooses a best match with rationale; a separate scorer agent calibrates confidence; and, when confidence falls below a threshold, a synonym generator agent proposes reformulations to re-enter the loop. The pipeline approaches state-of-the-art accuracy while revealing gaps and inconsistencies in existing annotations. The design avoids fine-tuning, improves robustness to ontology evolution, and yields interpretable decisions through grounded justifications. 

---
# Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning 

**Authors**: Juming Xiong, Kevin Guo, Congning Ni, Chao Yan, Katherine Brown, Avinash Baidya, Xiang Gao, Bradley Marlin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08999)  

**Abstract**: Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating multiple reasoning trajectories, leading to substantial additional computational overhead. This paper introduces a confidence-aware decision framework that analyzes a single completed reasoning trajectory to adaptively select between single-path and multi-path reasoning. The framework is trained using sentence-level numeric and linguistic features extracted from intermediate reasoning states in the MedQA dataset and generalizes effectively to MathQA, MedMCQA, and MMLU without additional fine-tuning. Experimental results show that the proposed method maintains accuracy comparable to multi-path baselines while using up to 80\% fewer tokens. These findings demonstrate that reasoning trajectories contain rich signals for uncertainty estimation, enabling a simple, transferable mechanism to balance accuracy and efficiency in LLM reasoning. 

---
# Bioalignment: Measuring and Improving LLM Disposition Toward Biological Systems for AI Safety 

**Authors**: Trent R Northen, Mingxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09154)  

**Abstract**: Large language models (LLMs) trained on internet-scale corpora can exhibit systematic biases that increase the probability of unwanted behavior. In this study, we examined potential biases towards synthetic vs. biological technological solutions across four domains (materials, energy, manufacturing, and algorithms). A sample of 5 frontier and 5 open-weight models were measured using 50 curated Bioalignment prompts with a Kelly criterion-inspired evaluation framework. According to this metric, most models were not bioaligned in that they exhibit biases in favor of synthetic (non-biological) solutions. We next examined if fine-tuning could increase the preferences of two open-weight models, Llama 3.2-3B-Instruct and Qwen2.5-3B-Instruct, for biological-based approaches. A curated corpus of ~22M tokens from 6,636 PMC articles emphasizing biological problem-solving was used first to fine-tune Llama 3B with a mixed corpus of continued training and instruction-formatted. This was then extended to Qwen 3B using instruction-formatted only. We found that QLoRA fine-tuning significantly increased the scoring of biological solutions for both models without degrading general capabilities (Holm-Bonferroni-corrected p < 0.001 and p < 0.01, respectively). This suggests that even a small amount of fine-tuning can change how models weigh the relative value of biological and bioinspired vs. synthetic approaches. Although this work focused on small open-weight LLMs, it may be extensible to much larger models and could be used to develop models that favor bio-based approaches. We release the benchmark, corpus, code, and adapter weights. 

---
# ConFu: Contemplate the Future for Better Speculative Sampling 

**Authors**: Zongyue Qin, Raghavv Goel, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.08899)  

**Abstract**: Speculative decoding has emerged as a powerful approach to accelerate large language model (LLM) inference by employing lightweight draft models to propose candidate tokens that are subsequently verified by the target model. The effectiveness of this paradigm critically depends on the quality of the draft model. While recent advances such as the EAGLE series achieve state-of-the-art speedup, existing draft models remain limited by error accumulation: they condition only on the current prefix, causing their predictions to drift from the target model over steps. In this work, we propose \textbf{ConFu} (Contemplate the Future), a novel speculative decoding framework that enables draft models to anticipate the future direction of generation. ConFu introduces (i) contemplate tokens and soft prompts that allow the draft model to leverage future-oriented signals from the target model at negligible cost, (ii) a dynamic contemplate token mechanism with MoE to enable context-aware future prediction, and (iii) a training framework with anchor token sampling and future prediction replication that learns robust future prediction. Experiments demonstrate that ConFu improves token acceptance rates and generation speed over EAGLE-3 by 8--11% across various downstream tasks with Llama-3 3B and 8B models. We believe our work is the first to bridge speculative decoding with continuous reasoning tokens, offering a new direction for accelerating LLM inference. 

---
# CyberThreat-Eval: Can Large Language Models Automate Real-World Threat Research? 

**Authors**: Xiangsen Chen, Xuan Feng, Shuo Chen, Matthieu Maitre, Sudipto Rakshit, Diana Duvieilh, Ashley Picone, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09452)  

**Abstract**: Analyzing Open Source Intelligence (OSINT) from large volumes of data is critical for drafting and publishing comprehensive CTI reports. This process usually follows a three-stage workflow -- triage, deep search and TI drafting. While Large Language Models (LLMs) offer a promising route toward automation, existing benchmarks still have limitations. These benchmarks often consist of tasks that do not reflect real-world analyst workflows. For example, human analysts rarely receive tasks in the form of multiple-choice questions. Also, existing benchmarks often rely on model-centric metrics that emphasize lexical overlap rather than actionable, detailed insights essential for security analysts. Moreover, they typically fail to cover the complete three-stage workflow. To address these issues, we introduce CyberThreat-Eval, which is collected from the daily CTI workflow of a world-leading company. This expert-annotated benchmark assesses LLMs on practical tasks across all three stages as mentioned above. It utilizes analyst-centric metrics that measure factual accuracy, content quality, and operational costs. Our evaluation using this benchmark reveals important insights into the limitations of current LLMs. For example, LLMs often lack the nuanced expertise required to handle complex details and struggle to distinguish between correct and incorrect information. To address these challenges, the CTI workflow incorporates both external ground-truth databases and human expert knowledge. TRA allows human experts to iteratively provide feedback for continuous improvement. The code is available at \href{this https URL}{\texttt{GitHub}} and \href{this https URL}{\texttt{HuggingFace}}. 

---
# TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA 

**Authors**: Mengwei Yuan, Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09297)  

**Abstract**: Large Language Model (LLM) has exhibited strong reasoning ability in text-based contexts across various domains, yet the limitation of context window poses challenges for the model on long-range inference tasks and necessitates a memory storage system. While many current storage approaches have been proposed with episodic notes and graph representations of memory, retrieval methods still primarily rely on predefined workflows or static similarity top-k over embeddings. To address this inflexibility, we introduced a novel tool-augmented autonomous memory retrieval framework (TA-Mem), which contains: (1) a memory extraction LLM agent which is prompted to adaptively chuck an input into sub-context based on semantic correlation, and extract information into structured notes, (2) a multi-indexed memory database designed for different types of query methods including both key-based lookup and similarity-based retrieval, (3) a tool-augmented memory retrieval agent which explores the memory autonomously by selecting appropriate tools provided by the database based on the user input, and decides whether to proceed to the next iteration or finalizing the response after reasoning on the fetched memories. The TA-Mem is evaluated on the LoCoMo dataset, achieving significant performance improvements over existing baseline approaches. In addition, an analysis of tool use across different question types also demonstrates the adaptivity of the proposed method. 

---
# Self-hosted Lecture-to-Quiz: Local LLM MCQ Generation with Deterministic Quality Control 

**Authors**: Seine A. Shintani  

**Link**: [PDF](https://arxiv.org/pdf/2603.08729)  

**Abstract**: We present an end-to-end self-hosted (API-free) pipeline, where API-free means that lecture content is not sent to any external LLM service, that converts lecture PDFs into multiple-choice questions (MCQs) using a local LLM plus deterministic quality control (QC). The pipeline is designed for black-box minimization: LLMs may assist drafting, but the final released artifacts are plain-text question banks with an explicit QC trace and without any need to call an LLM at deployment time. We run a seed sweep on three short "dummy lectures" (information theory, thermodynamics, and statistical mechanics), collecting 15 runs x 8 questions = 120 accepted candidates (122 attempts total under bounded retries). All 120 accepted candidates satisfy hard QC checks (JSON schema conformance, a single marked correct option, and numeric/constant equivalence tests); however, the warning layer flags 8/120 items (spanning 8 runs) that expose residual quality risks such as duplicated distractors or missing rounding instructions. We report a warning taxonomy with concrete before->after fixes, and we release the final 24-question set (three lectures x 8 questions) as JSONL/CSV for Google Forms import (e.g., via Apps Script or API tooling) included as ancillary files under anc/. Finally, we position the work through the AI to Learn (AI2L) rubric lens and argue that self-hosted MCQ generation with explicit QC supports privacy, accountability, and Green AI in educational workflows. 

---
# RecThinker: An Agentic Framework for Tool-Augmented Reasoning in Recommendation 

**Authors**: Haobo Zhang, Yutao Zhu, Kelong Mao, Tianhao Li, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2603.09843)  

**Abstract**: Large Language Models (LLMs) have revolutionized recommendation agents by providing superior reasoning and flexible decision-making capabilities. However, existing methods mainly follow a passive information acquisition paradigm, where agents either rely on static pre-defined workflows or perform reasoning with constrained information. It limits the agent's ability to identify information sufficiency, often leading to suboptimal recommendations when faced with fragmented user profiles or sparse item metadata. To address these limitations, we propose RecThinker, an agentic framework for tool-augmented reasoning in recommendation, which shifts recommendation from passive processing to autonomous investigation by dynamically planning reasoning paths and proactively acquiring essential information via autonomous tool-use. Specifically, RecThinker adopts an Analyze-Plan-Act paradigm, which first analyzes the sufficiency of user-item information and autonomously invokes tool-calling sequences to bridge information gaps between available knowledge and reasoning requirements. We develop a suite of specialized tools for RecThinker, enabling the model to acquire user-side, item-side, and collaborative information for better reasoning and user-item matching. Furthermore, we introduce a self-augmented training pipeline, comprising a Supervised Fine-Tuning (SFT) stage to internalize high-quality reasoning trajectories and a Reinforcement Learning (RL) stage to optimize for decision accuracy and tool-use efficiency. Extensive experiments on multiple benchmark datasets demonstrate that RecThinker consistently outperforms strong baselines in the recommendation scenario. 

---
# Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval 

**Authors**: Yingyi Zhang, Junyi Li, Wenlin Zhang, Penyue Jia, Xianneng Li, Yichao Wang, Derong Xu, Yi Wen, Huifeng Guo, Yong Liu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.09250)  

**Abstract**: Personalized large language models (LLMs) rely on memory retrieval to incorporate user-specific histories, preferences, and contexts. Existing approaches either overload the LLM by feeding all the user's past memory into the prompt, which is costly and unscalable, or simplify retrieval into a one-shot similarity search, which captures only surface matches. Cognitive science, however, shows that human memory operates through a dual process: Familiarity, offering fast but coarse recognition, and Recollection, enabling deliberate, chain-like reconstruction for deeply recovering episodic content. Current systems lack both the ability to perform recollection retrieval and mechanisms to adaptively switch between the dual retrieval paths, leading to either insufficient recall or the inclusion of noise. To address this, we propose RF-Mem (Recollection-Familiarity Memory Retrieval), a familiarity uncertainty-guided dual-path memory retriever. RF-Mem measures the familiarity signal through the mean score and entropy. High familiarity leads to the direct top-K Familiarity retrieval path, while low familiarity activates the Recollection path. In the Recollection path, the system clusters candidate memories and applies alpha-mix with the query to iteratively expand evidence in embedding space, simulating deliberate contextual reconstruction. This design embeds human-like dual-process recognition into the retriever, avoiding full-context overhead and enabling scalable, adaptive personalization. Experiments across three benchmarks and corpus scales demonstrate that RF-Mem consistently outperforms both one-shot retrieval and full-context reasoning under fixed budget and latency constraints. Our code can be found in the Reproducibility Statement. 

---
