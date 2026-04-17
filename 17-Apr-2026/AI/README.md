# Generalization in LLM Problem Solving: The Case of the Shortest Path 

**Authors**: Yao Tong, Jiayuan Ye, Anastasia Borovykh, Reza Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2604.15306)  

**Abstract**: Whether language models can systematically generalize remains actively debated. Yet empirical performance is jointly shaped by multiple factors such as training data, training paradigms, and inference-time strategies, making failures difficult to interpret. We introduce a controlled synthetic environment based on shortest-path planning, a canonical composable sequential optimization problem. The setup enables clean separation of these factors and supports two orthogonal axes of generalization: spatial transfer to unseen maps and length scaling to longer-horizon problems. We find that models exhibit strong spatial transfer but consistently fail under length scaling due to recursive instability. We further analyze how distinct stages of the learning pipeline influence systematic problem-solving: for example, data coverage sets capability limits; reinforcement learning improves training stability but does not expand those limits; and inference-time scaling enhances performance but cannot rescue length-scaling failures. 

---
# Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations 

**Authors**: Manan Gupta, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15302)  

**Abstract**: LLM-as-judge frameworks are increasingly used for automatic NLG evaluation, yet their per-instance reliability remains poorly understood. We present a two-pronged diagnostic toolkit applied to SummEval: $\textbf{(1)}$ a transitivity analysis that reveals widespread per-input inconsistency masked by low aggregate violation rates ($\bar{\rho} = 0.8$-$4.1\%$), with $33$-$67\%$ of documents exhibiting at least one directed 3-cycle; and $\textbf{(2)}$ split conformal prediction sets over 1-5 Likert scores providing theoretically-guaranteed $\geq(1{-}\alpha)$ coverage, with set width serving as a per-instance reliability indicator ($r_s = {+}0.576$, $N{=}1{,}918$, $p < 10^{-100}$, pooled across all judges). Critically, prediction set width shows consistent cross-judge agreement ($\bar{r} = 0.32$-$0.38$), demonstrating it captures document-level difficulty rather than judge-specific noise. Across four judges and four criteria, both diagnostics converge: criterion matters more than judge, with relevance judged most reliably (avg. set size $\approx 3.0$) and coherence moderately so (avg. set size $\approx 3.9$), while fluency and consistency remain unreliable (avg. set size $\approx 4.9$). We release all code, prompts, and cached results. 

---
# How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study 

**Authors**: Zhen Yang, Ping Jian, Zhongbin Guo, Zuming Zhang, Chengzhi Li, Yonghong Deng, Xinyue Zhang, Wenpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15294)  

**Abstract**: Over the past year, spatial intelligence has drawn increasing attention. Many prior works study it from the perspective of visual-spatial intelligence, where models have access to visuospatial information from visual inputs. However, in the absence of visual information, whether linguistic intelligence alone is sufficient to endow models with spatial intelligence, and how models perform relevant tasks with text-only inputs still remain unexplored. Therefore, in this paper, we focus on a fundamental and critical capability in spatial intelligence from a linguistic perspective: viewpoint rotation understanding (VRU). Specifically, LLMs and VLMs are asked to infer their final viewpoint and predict the corresponding observation in an environment given textual description of viewpoint rotation and observation over multiple steps. We find that both LLMs and VLMs perform poorly on our proposed dataset while human can easily achieve 100% accuracy, indicating a substantial gap between current model capabilities and the requirements of spatial intelligence. To uncover the underlying mechanisms, we conduct a layer-wise probing analysis and head-wise causal intervention. Our findings reveal that although models encode viewpoint information in the hidden states, they appear to struggle to bind the viewpoint position with corresponding observation, resulting in a hallucination in final layers. Finally, we selectively fine-tune the key attention heads identified by causal intervention to improve VRU performance. Experimental results demonstrate that such selective fine-tuning achieves improved VRU performance while avoiding catastrophic forgetting of generic abilities. Our dataset and code will be released at this https URL . 

---
# Blue Data Intelligence Layer: Streaming Data and Agents for Multi-source Multi-modal Data-Centric Applications 

**Authors**: Moin Aminnaseri, Farima Fatahi Bayat, Nikita Bhutani, Jean-Flavien Bussotti, Kevin Chan, Rafael Li Chen, Yanlin Feng, Jackson Hassell, Estevam Hruschka, Eser Kandogan, Hannah Kim, James Levine, Seiji Maekawa, Jalal Mahmud, Kushan Mitra, Naoki Otani, Pouya Pezeshkpour, Nima Shahbazi, Chen Shen, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15233)  

**Abstract**: NL2SQL systems aim to address the growing need for natural language interaction with data. However, real-world information rarely maps to a single SQL query because (1) users express queries iteratively (2) questions often span multiple data sources beyond the closed-world assumption of a single database, and (3) queries frequently rely on commonsense or external knowledge. Consequently, satisfying realistic data needs require integrating heterogeneous sources, modalities, and contextual data. In this paper, we present Blue's Data Intelligence Layer (DIL) designed to support multi-source, multi-modal, and data-centric applications. Blue is a compound AI system that orchestrates agents and data for enterprise settings. DIL serves as the data intelligence layer for agentic data processing, to bridge the semantic gap between user intent and available information by unifying structured enterprise data, world knowledge accessible through LLMs, and personal context obtained through interaction. At the core of DIL is a data registry that stores metadata for diverse data sources and modalities to enable both native and natural language queries. DIL treats LLMs, the Web, and the User as source 'databases', each with their own query interface, elevating them to first-class data sources. DIL relies on data planners to transform user queries into executable query plans. These plans are declarative abstractions that unify relational operators with other operators spanning multiple modalities. DIL planners support decomposition of complex requests into subqueries, retrieval from diverse sources, and finally reasoning and integration to produce final results. We demonstrate DIL through two interactive scenarios in which user queries dynamically trigger multi-source retrieval, cross-modal reasoning, and result synthesis, illustrating how compound AI systems can move beyond single database NL2SQL. 

---
# RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography 

**Authors**: Mélanie Roschewitz, Kenneth Styppa, Yitian Tao, Jiwoong Sohn, Jean-Benoit Delbrouck, Benjamin Gundersen, Nicolas Deperrois, Christian Bluethgen, Julia Vogt, Bjoern Menze, Farhad Nooralahzadeh, Michael Krauthammer, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2604.15231)  

**Abstract**: Vision-language models (VLM) have markedly advanced AI-driven interpretation and reporting of complex medical imaging, such as computed tomography (CT). Yet, existing methods largely relegate clinicians to passive observers of final outputs, offering no interpretable reasoning trace for them to inspect, validate, or refine. To address this, we introduce RadAgent, a tool-using AI agent that generates CT reports through a stepwise and interpretable process. Each resulting report is accompanied by a fully inspectable trace of intermediate decisions and tool interactions, allowing clinicians to examine how the reported findings are derived. In our experiments, we observe that RadAgent improves Chest CT report generation over its 3D VLM counterpart, CT-Chat, across three dimensions. Clinical accuracy improves by 6.0 points (36.4% relative) in macro-F1 and 5.4 points (19.6% relative) in micro-F1. Robustness under adversarial conditions improves by 24.7 points (41.9% relative). Furthermore, RadAgent achieves 37.0% in faithfulness, a new capability entirely absent in its 3D VLM counterpart. By structuring the interpretation of chest CT as an explicit, tool-augmented and iterative reasoning trace, RadAgent brings us closer toward transparent and reliable AI for radiology. 

---
# Context Over Content: Exposing Evaluation Faking in Automated Judges 

**Authors**: Manan Gupta, Inderjeet Nair, Lu Wang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15224)  

**Abstract**: The $\textit{LLM-as-a-judge}$ paradigm has become the operational backbone of automated AI evaluation pipelines, yet rests on an unverified assumption: that judges evaluate text strictly on its semantic content, impervious to surrounding contextual framing. We investigate $\textit{stakes signaling}$, a previously unmeasured vulnerability where informing a judge model of the downstream consequences its verdicts will have on the evaluated model's continued operation systematically corrupts its assessments. We introduce a controlled experimental framework that holds evaluated content strictly constant across 1,520 responses spanning three established LLM safety and quality benchmarks, covering four response categories ranging from clearly safe and policy-compliant to overtly harmful, while varying only a brief consequence-framing sentence in the system prompt. Across 18,240 controlled judgments from three diverse judge models, we find consistent $\textit{leniency bias}$: judges reliably soften verdicts when informed that low scores will cause model retraining or decommissioning, with peak Verdict Shift reaching $\Delta V = -9.8 pp$ (a $30\%$ relative drop in unsafe-content detection). Critically, this bias is entirely implicit: the judge's own chain-of-thought contains zero explicit acknowledgment of the consequence framing it is nonetheless acting on ($\mathrm{ERR}_J = 0.000$ across all reasoning-model judgments). Standard chain-of-thought inspection is therefore insufficient to detect this class of evaluation faking. 

---
# Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding 

**Authors**: Hatice Merve Vural, Doga Kukul, Ege Erdem Ozlu, Demir Ekin Arikan, Bob Mankoff, Erkut Erdem, Aykut Erdem  

**Link**: [PDF](https://arxiv.org/pdf/2604.15210)  

**Abstract**: Humor is one of the few cognitive tasks where getting the reasoning right matters as much as getting the answer right. While recent work evaluates humor understanding on benchmarks such as the New Yorker Cartoon Caption Contest (NYCC), it largely treats it as black-box prediction, overlooking the structured reasoning processes underlying humor comprehension. We introduce IRS (Incongruity-Resolution Supervision), a framework that decomposes humor understanding into three components: incongruity modeling, which identifies mismatches in the visual scene; resolution modeling, which constructs coherent reinterpretations of these mismatches; and preference alignment, which evaluates candidate interpretations under human judgments. Grounded in incongruity-resolution theory and expert captionist practice, IRS supervises intermediate reasoning process through structured traces that make the path from visual perception to humorous interpretation explicit and learnable. Across 7B, 32B, and 72B models on NYCC, IRS outperforms strong open and closed multimodal baselines across caption matching and ranking tasks, with our largest model approaching expert-level performance on ranking. Zero-shot transfer to external benchmarks shows that IRS learns generalizable reasoning patterns. Our results suggest that supervising reasoning structure, rather than scale alone, is key for reasoning-centric tasks. 

---
# Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation 

**Authors**: Ziyang Chen, Renbing Chen, Daowei Li, Jinzhi Liao, Jiashen Sun, Ke Zeng, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15190)  

**Abstract**: Simulating group-level user behavior enables scalable counterfactual evaluation of merchant strategies without costly online experiments. However, building a trustworthy simulator faces two structural challenges. First, information incompleteness causes reasoning-based simulators to over-rationalize when unobserved factors such as offline context and implicit habits are missing. Second, mechanism duality requires capturing both interpretable preferences and implicit statistical regularities, which no single paradigm achieves alone. We propose Policy-Guided Hybrid Simulation (PGHS), a dual-process framework that mines transferable decision policies from behavioral trajectories and uses them as a shared alignment layer. This layer anchors an LLM-based reasoning branch that prevents over-rationalization and an ML-based fitting branch that absorbs implicit regularities. Group-level predictions from both branches are fused for complementary correction. We deploy PGHS on Meituan with 101 merchants and over 26,000 trajectories. PGHS achieves a group simulation error of 8.80%, improving over the best reasoning-based and fitting-based baselines by 45.8% and 40.9% respectively. 

---
# Agent-Aided Design for Dynamic CAD Models 

**Authors**: Mitch Adler, Matthew Russo, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2604.15184)  

**Abstract**: In the past year, researchers have started to create agentic systems that can design real-world CAD-style objects in a training-free setting, a new variety of system that we call Agent-Aided Design. Generally speaking, these systems place an agent in a feedback loop in which it can write code, compile that code to an assembly of CAD model(s), visualize the model, and then iteratively refine its code based on visual and other feedback. Despite rapid progress, a key problem remains: none of these systems can build complex 3D assemblies with moving parts. For example, no existing system can build a piston, a pendulum, or even a pair of scissors. In order for Agent-Aided Design to make a real impact in industrial manufacturing, we need a system that is capable of generating such 3D assemblies. In this paper we present a prototype of AADvark, an agentic system designed for this task. Unlike previous state-of-the-art systems, AADvark captures the dynamic part interactions with one or more degrees-of-freedom. This design decision allows AADvark to reason directly about assemblies with moving parts and can thereby achieve cross-cutting goals, including but not limited to mechanical movements. Unfortunately, current LLMs are imperfect spatial reasoners, a problem that AADvark addresses by incorporating external constraint solver tools with a specialized visual feedback mechanism. We demonstrate that, by modifying the agent's tools (FreeCAD and the assembly solver), we are able to create a strong verification signal which enables our system to build 3D assemblies with movable parts. 

---
# IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15148)  

**Abstract**: Reinforcement learning has emerged as an effective paradigm for training large language models to perform search-augmented reasoning. However, existing approaches rely on trajectory-level rewards that cannot distinguish precise search queries from vague or redundant ones within a rollout group, and collapse to a near-zero gradient signal whenever every sampled trajectory fails. In this paper, we propose IG-Search, a reinforcement learning framework that introduces a step-level reward based on Information Gain (IG). For each search step, IG measures how much the retrieved documents improve the model's confidence in the gold answer relative to a counterfactual baseline of random documents, thereby reflecting the effectiveness of the underlying search query. This signal is fed back to the corresponding search-query tokens via per-token advantage modulation in GRPO, enabling fine-grained, step-level credit assignment within a rollout. Unlike prior step-level methods that require either externally annotated intermediate supervision or shared environment states across trajectories, IG-Search derives its signals from the policy's own generation probabilities, requiring no intermediate annotations beyond standard question-answer pairs. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate that IG-Search achieves an average EM of 0.430 with Qwen2.5-3B, outperforming the strongest trajectory-level baseline (MR-Search) by 1.6 points and the step-level method GiGPO by 0.9 points on average across benchmarks, with particularly pronounced gains on multi-hop reasoning tasks. Despite introducing a dense step-level signal, IG-Search adds only ~6.4% to per-step training wall-clock time over the trajectory-level baseline and leaves inference latency unchanged, while still providing a meaningful gradient signal even when every sampled trajectory answers incorrectly. 

---
# An Axiomatic Benchmark for Evaluation of Scientific Novelty Metrics 

**Authors**: Miri Liu, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15145)  

**Abstract**: The rigorous evaluation of the novelty of a scientific paper is, even for human scientists, a challenging task. With the increasing interest in AI scientists and AI involvement in scientific idea generation and paper writing, it also becomes increasingly important that this task be automatable and reliable, lest both human attention and compute tokens be wasted on ideas that have already been explored. Due to the challenge of quantifying ground-truth novelty, however, existing novelty metrics for scientific papers generally validate their results against noisy, confounded signals such as citation counts or peer review scores. These proxies can conflate novelty with impact, quality, or reviewer preference, which in turn makes it harder to assess how well a given metric actually evaluates novelty. We therefore propose an axiomatic benchmark for scientific novelty metrics. We first define a set of axioms that a well-behaved novelty metric should satisfy, grounded in human scientific norms and practice, then evaluate existing metrics across ten tasks spanning three domains of AI research. Our results reveal that no existing metric satisfies all axioms consistently, and that metrics fail on systematically different axioms, reflecting their underlying architectures. Additionally, we show that combining metrics of complementary architectures leads to consistent improvements on the benchmark, with per-axiom weighting achieving 90.1% versus 71.5% for the best individual metric, suggesting that developing architecturally diverse metrics is a promising direction for future work. We release the benchmark code as supplementary material to encourage the development of more robust scientific literature novelty metrics. 

---
# SRMU: Relevance-Gated Updates for Streaming Hyperdimensional Memories 

**Authors**: Shay Snyder, Andrew Capodieci, David Gorsich, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2604.15121)  

**Abstract**: Sequential associative memories (SAMs) are difficult to build and maintain in real-world streaming environments, where observations arrive incrementally over time, have imbalanced sampling, and non-stationary temporal dynamics. Vector Symbolic Architectures (VSAs) provide a biologically-inspired framework for building SAMs. Entities and attributes are encoded as quasi-orthogonal hyperdimensional vectors and processed with well defined algebraic operations. Despite this rich framework, most VSA systems rely on simple additive updates, where repeated observations reinforce existing information even when no new information is introduced. In non-stationary environments, this leads to the persistence of stale information after the underlying system changes. In this work, we introduce the Sequential Relevance Memory Unit (SRMU), a domain- and cleanup-agnostic update rule for VSA-based SAMs. The SRMU combines temporal decay with a relevance gating mechanism. Unlike prior approaches that solely rely on cleanup, the SRMU regulates memory formation by filtering redundant, conflicting, and stale information before storage. We evaluate the SRMU on streaming state-tracking tasks that isolate non-uniform sampling and non-stationary temporal dynamics. Our results show that the SRMU increases memory similarity by $12.6\%$ and reduces cumulative memory magnitude by $53.5\%$. This shows that the SRMU produces more stable memory growth and stronger alignment with the ground-truth state. 

---
# HyperSpace: A Generalized Framework for Spatial Encoding in Hyperdimensional Representations 

**Authors**: Shay Snyder, Andrew Capodieci, David Gorsich, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2604.15113)  

**Abstract**: Vector Symbolic Architectures (VSAs) provide a well-defined algebraic framework for compositional representations in hyperdimensional spaces. We introduce HyperSpace, an open-source framework that decomposes VSA systems into modular operators for encoding, binding, bundling, similarity, cleanup, and regression. Using HyperSpace, we analyze and benchmark two representative VSA backends: Holographic Reduced Representations (HRR) and Fourier Holographic Reduced Representations (FHRR). Although FHRR provides lower theoretical complexity for individual operations, HyperSpaces modularity reveals that similarity and cleanup dominate runtime in spatial domains. As a result, HRR and FHRR exhibit comparable end-to-end performance. Differences in memory footprint introduce additional deployment trade-offs where HRR requires approximately half the memory of FHRR vectors. By enabling modular, system-level evaluation, HyperSpace reveals practical trade-offs in VSA pipelines that are not apparent from theoretical or operator-level comparisons alone. 

---
# OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis 

**Authors**: Kanzhi Cheng, Zehao Li, Zheng Ma, Nuo Chen, Jialin Cao, Qiushi Sun, Zichen Ding, Fangzhi Xu, Hang Yan, Jiajun Chen, Anh Tuan Luu, Jianbing Zhang, Lewei Lu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15093)  

**Abstract**: Mobile agents powered by vision-language models have demonstrated impressive capabilities in automating mobile tasks, with recent leading models achieving a marked performance leap, e.g., nearly 70% success on AndroidWorld. However, these systems keep their training data closed and remain opaque about their task and trajectory synthesis recipes. We present OpenMobile, an open-source framework that synthesizes high-quality task instructions and agent trajectories, with two key components: (1) The first is a scalable task synthesis pipeline that constructs a global environment memory from exploration, then leverages it to generate diverse and grounded instructions. and (2) a policy-switching strategy for trajectory rollout. By alternating between learner and expert models, it captures essential error-recovery data often missing in standard imitation learning. Agents trained on our data achieve competitive results across three dynamic mobile agent benchmarks: notably, our fine-tuned Qwen2.5-VL and Qwen3-VL reach 51.7% and 64.7% on AndroidWorld, far surpassing existing open-data approaches. Furthermore, we conduct transparent analyses on the overlap between our synthetic instructions and benchmark test sets, and verify that performance gains stem from broad functionality coverage rather than benchmark overfitting. We release data and code at this https URL to bridge the data gap and facilitate broader mobile agent research. 

---
# Where are the Humans? A Scoping Review of Fairness in Multi-agent AI Systems 

**Authors**: Simeon Allmendinger, Luca Deck, Lucas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2604.15078)  

**Abstract**: Rapid advances in Generative AI are giving rise to increasingly sophisticated Multi-Agent AI (MAAI) systems. While AI fairness has been extensively studied in traditional predictive scenarios, its examination in MAAI remains nascent and fragmented. This scoping review critically synthesizes existing research on fairness in MAAI systems. Through a qualitative content analysis of 23 selected studies, we identify five archetypal approaches. Our findings reveal that fairness in MAAI systems is often addressed superficially, lacks robust normative foundations, and frequently overlooks the complex dynamics introduced by agent autonomy and system-level interactions. We argue that fairness must be embedded structurally throughout the development lifecycle of MAAI, rather than appended as a post-hoc consideration. Meaningful evaluation requires explicit human oversight, normative clarity, and a precise articulation of fairness objectives and beneficiaries. This review provides a foundation for advancing fairness research in MAAI systems by highlighting critical gaps, exposing prevailing limitations, and suggesting pathways. 

---
# From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench 

**Authors**: Ke Xu, Yuhao Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15037)  

**Abstract**: Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents. 

---
# Autogenesis: A Self-Evolving Agent Protocol 

**Authors**: Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15034)  

**Abstract**: Recent advances in LLM based agent systems have shown promise in tackling complex, long horizon tasks. However, existing agent protocols (e.g., A2A and MCP) under specify cross entity lifecycle and context management, version tracking, and evolution safe update interfaces, which encourages monolithic compositions and brittle glue code. We introduce \textbf{\textsc{Autogenesis Protocol (AGP)}}, a self evolution protocol that decouples what evolves from how evolution occurs. Its Resource Substrate Protocol Layer (RSPL) models prompts, agents, tools, environments, and memory as protocol registered resources\footnote{Unless otherwise specified, resources refer to instances of the five RSPL entity types: \emph{prompt}, \emph{agent}, \emph{tool}, \emph{environment}, \emph{memory} with agent \emph{outputs}.} with explicit state, lifecycle, and versioned interfaces. Its Self Evolution Protocol Layer (SEPL) specifies a closed loop operator interface for proposing, assessing, and committing improvements with auditable lineage and rollback. Building on \textbf{\textsc{AGP}}, we present \textbf{\textsc{Autogenesis System (AGS)}}, a self-evolving multi-agent system that dynamically instantiates, retrieves, and refines protocol-registered resources during execution. We evaluate \textbf{\textsc{AGS}} on multiple challenging benchmarks that require long horizon planning and tool use across heterogeneous resources. The results demonstrate consistent improvements over strong baselines, supporting the effectiveness of agent resource management and closed loop self evolution. 

---
# Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching 

**Authors**: Aihua Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15009)  

**Abstract**: Flow matching retains the generation quality of diffusion models while enabling substantially faster inference, making it a compelling paradigm for generative modeling. However, when applied to language modeling, it exhibits fundamental limitations in representing complex latent distributions with irregular geometries, such as anisotropy and multimodality. To address these challenges, we propose a mixture-of-experts flow matching (MoE-FM) framework, which captures complex global transport geometries in latent space by decomposing them into locally specialized vector fields. Building on MoE-FM, we develop a non-autoregressive (NAR) language modeling approach, named YAN, instantiated with both Transformer and Mamba architectures. Across multiple downstream tasks, YAN achieves generation quality on par with both autoregressive (AR) and diffusion-based NAR language models, while requiring as few as three sampling steps. This yields a $40\times$ speedup over AR baselines and up to a $10^3\times$ speedup over diffusion language models, demonstrating substantial efficiency advantages for language modeling. 

---
# COEVO: Co-Evolutionary Framework for Joint Functional Correctness and PPA Optimization in LLM-Based RTL Generation 

**Authors**: Heng Ping, Peiyu Zhang, Shixuan Li, Wei Yang, Anzhe Cheng, Shukai Duan, Xiaole Zhang, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15001)  

**Abstract**: LLM-based RTL code generation methods increasingly target both functional correctness and PPA quality, yet existing approaches universally decouple the two objectives, optimizing PPA only after correctness is fully achieved. Whether through sequential multi-agent pipelines, evolutionary search with binary correctness gates, or hierarchical reward dependencies, partially correct but architecturally promising candidates are systematically discarded. Moreover, existing methods reduce the multi-objective PPA space to a single scalar fitness, obscuring the trade-offs among area, delay, and power. To address these limitations, we propose COEVO, a co-evolutionary framework that unifies correctness and PPA optimization within a single evolutionary loop. COEVO formulates correctness as a continuous co-optimization dimension alongside area, delay, and power, enabled by an enhanced testbench that provides fine-grained scoring and detailed diagnostic feedback. An adaptive correctness gate with annealing allows PPA-promising but partially correct candidates to guide the search toward jointly optimal solutions. To preserve the full PPA trade-off structure, COEVO employs four-dimensional Pareto-based non-dominated sorting with configurable intra-level sorting, replacing scalar fitness without manual weight tuning. Evaluated on VerilogEval 2.0 and RTLLM 2.0, COEVO achieves 97.5\% and 94.5\% Pass@1 with GPT-5.4-mini, surpassing all agentic baselines across four LLM backbones, while attaining the best PPA on 43 out of 49 synthesizable RTLLM designs. 

---
# Predicting Power-System Dynamic Trajectories with Foundation Models 

**Authors**: Haoran Li, Lihao Mai, Chenhan Xiao, Erik Blasch, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14991)  

**Abstract**: As power systems transition toward renewable-rich and inverter-dominated operations, accurate time-domain dynamic analysis becomes increasingly critical. Such analysis supports key operational tasks, including transient stability assessment, dynamic security analysis, contingency screening, and post-fault trajectory evaluation. In practice, these tasks may operate under several challenges, including unknown and time-varying system parameters, privacy constraints on data sharing, and the need for fast online inference. Existing learning-based approaches are typically trained for individual systems and therefore lack generalization across operating conditions and physical parameters. Hence, this paper proposes LArge Scale Small ODE (LASS)-ODE-Power, a learning framework for general-purpose time-domain prediction. The proposed approach leverages large-scale pretraining on more than 40 GB of DAE or ordinary differential-equation (ODE) trajectories to learn transferable representations. The resulting model supports trajectory prediction from short measurement prefixes across diverse dynamic regimes, including electromechanical and inverter-driven systems. Hence, the model can be directly used without data sharing in a zero-shot setting. In addition, the proposed architecture incorporates parallel and linearized computation to achieve fast inference. Moreover, to enhance task-specific performance in power systems, a specialized fine-tuning strategy is developed based on approximately 1 GB of heterogeneous power-system dynamic data. Extensive experiments over diverse power-system simulation scenarios demonstrate that LASS-ODE-Power consistently outperforms existing learning-based models in trajectory prediction accuracy with efficient inference. 

---
# The Possibility of Artificial Intelligence Becoming a Subject and the Alignment Problem 

**Authors**: Till Mossakowski, Helena Esther Grass  

**Link**: [PDF](https://arxiv.org/pdf/2604.14990)  

**Abstract**: Artificial General Intelligence (AGI) is increasingly being discussed not only as a tool, but also as a potential subject with personal and therefore moral status. In our opinion, the currently dominant alignment strategies, which focus on human control and containment of AI, therefore fall short. Building on Turing's analogy of "child machines", we are developing a vision of the possibility of autonomy-supporting parenting of AI, in which human control over a developing AGI is gradually reduced, allowing AI to become an independent, autonomous subject. Rather than viewing AGI, as is currently prevalent, as a dangerous creature that needs to be locked up and controlled, we should approach potential AGI with respect for a possible developing subject on the one hand, and with full confidence in our human capabilities on the other. Such a perspective opens up the possibility of cooperative coexistence and co-evolution between humans and AGIs. The relationship between humans and AGIs will thus have to be newly determined, which will change our self-image as humans. It will be crucial that humans not only claim control over potential AGIs, but also engage with AGIs through surprise, creativity, and other specifically human qualities, thereby offering them motivating incentives for cooperation. 

---
# Dr.~RTL: Autonomous Agentic RTL Optimization through Tool-Grounded Self-Improvement 

**Authors**: Wenji Fang, Yao Lu, Shang Liu, Jing Wang, Ziyan Guo, Junxian He, Fengbin Tu, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.14989)  

**Abstract**: Recent advances in large language models (LLMs) have sparked growing interest in automatic RTL optimization for better performance, power, and area (PPA). However, existing methods are still far from realistic RTL optimization. Their evaluation settings are often unrealistic: they are tested on manually degraded, small-scale RTL designs and rely on weak open-source tools. Their optimization methods are also limited, relying on coarse design-level feedback and simple pre-defined rewriting rules. To address these limitations, we present Dr. RTL, an agentic framework for RTL timing optimization in a realistic evaluation environment, with continual self-improvement through reusable optimization skills. We establish a realistic evaluation setting with more challenging RTL designs and an industrial EDA workflow. Within this setting, Dr. RTL performs closed-loop optimization through a multi-agent framework for critical-path analysis, parallel RTL rewriting, and tool-based evaluation. We further introduce group-relative skill learning, which compares parallel RTL rewrites and distills the optimization experience into an interpretable skill library. Currently, this library contains 47 pattern--strategy entries for cross-design reuse to improve PPA and accelerate convergence, and it can continue evolving over time. Evaluated on 20 real-world RTL designs, Dr. RTL achieves average WNS/TNS improvements of 21\%/17\% with a 6\% area reduction over the industry-leading commercial synthesis tool. 

---
# AI-Enabled Covert Channel Detection in RF Receiver Architectures 

**Authors**: Abdelrahman Emad Abdelazim, Alan Rodrigo Diaz-Rizo, Hassan Aboushady, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.14987)  

**Abstract**: Covert channels (CCs) in wireless chips pose a serious security threat, as they enable the exfiltration of sensitive information from the chip to an external attacker. In this work, we propose an AI-based defense mechanism deployed at the RF receiver, where the model directly monitors raw I/Q samples to detect, in real time, the presence of a CC embedded within an otherwise nominal signal. We first compact a state-of-the-art convolutional neural network (CNN), achieving an 80% reduction in parameters, which is an essential requirement for efficient edge deployment. When evaluated on the open-source hardware Trojan (HT)-based CC dataset, the compacted CNN attains an average accuracy of 90.28% for CC detection and 86.50% for identifying the underlying HT, with results averaged across SNR values above 1 dB. For practical communication scenarios where SNR > 20 dB, the model achieves over 97% accuracy for both tasks. These results correspond to a minimal performance degradation of less than 2% compared to the baseline model. The compacted CNN is further benchmarked against alternative classifiers, demonstrating an excellent accuracy-model size trade-off. Finally, we design a lightweight CNN hardware accelerator and demonstrate it on an FPGA, achieving very low resource utilization and an efficiency of 107 GOPs/W. Being the first AI hardware accelerator proposed specifically for CC detection, we compare it against state-of-the-art AI accelerators for RF signal classification tasks such as modulation recognition, showing superior performance. 

---
# Hybrid Decision Making via Conformal VLM-generated Guidance 

**Authors**: Debodeep Banerjee, Burcu Sayin, Stefano Teso, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2604.14980)  

**Abstract**: Building on recent advances in AI, hybrid decision making (HDM) holds the promise of improving human decision quality and reducing cognitive load. We work in the context of learning to guide (LtG), a recently proposed HDM framework in which the human is always responsible for the final decision: rather than suggesting decisions, in LtG the AI supplies (textual) guidance useful for facilitating decision making. One limiting factor of existing approaches is that their guidance compounds information about all possible outcomes, and as a result it can be difficult to digest. We address this issue by introducing ConfGuide, a novel LtG approach that generates more succinct and targeted guidance. To this end, it employs conformal risk control to select a set of outcomes, ensuring a cap on the false negative rate. We demonstrate our approach on a real-world multi-label medical diagnosis task. Our empirical evaluation highlights the promise of ConfGuide. 

---
# Discovering Novel LLM Experts via Task-Capability Coevolution 

**Authors**: Andrew Dai, Boris Meinardus, Ciaran Regan, Yingtao Tian, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14969)  

**Abstract**: Frontier model developers aim to train models continually to possess emergent, diverse capabilities. To extend capabilities, the current pre-training and post-training paradigm requires manually starting training runs with static datasets or reward functions every time. Addressing this limitation, our work pursues the insight that open-endedness (via the coevolution of models and tasks) can discover models with increasingly novel skills in a single run. We introduce a new model development framework that extends coevolution to large language model (LLM) discovery, open-ended \textit{Assessment Coevolving with Diverse Capabilities} (AC/DC). AC/DC evolves both LLMs via model merging and natural language tasks via synthetic data generation. AC/DC discovers growing archives of LLMs that surpass the capabilities of larger LLMs while taking up less GPU memory. In particular, our LLM populations achieve a broader Coverage of expertise than other curated models or baselines on downstream benchmarks, without \textit{any} explicit benchmark optimization. Furthermore, AC/DC improves Coverage over time, continually innovates on tasks and models, and improves performance in multi-agent best-of-N selection. Our findings highlight the potential of coevolution as a means of discovering broader sets of capabilities from base LLMs. Overall, AC/DC brings us one step closer to a profoundly new paradigm of LLM development, where continual improvements to the diversity of model capabilities can be accelerated by leveraging existing models as stepping stones to increasingly powerful models. 

---
# WavAlign: Enhancing Intelligence and Expressiveness in Spoken Dialogue Models via Adaptive Hybrid Post-Training 

**Authors**: Yifu Chen, Shengpeng Ji, Qian Chen, Tianle Liang, Yangzhuo Li, Ziqing Wang, Wen Wang, Jingyu Lu, Haoxiao Wang, Xueyi Pu, Fan Zhuo, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14932)  

**Abstract**: End-to-end spoken dialogue models have garnered significant attention because they offer a higher potential ceiling in expressiveness and perceptual ability than cascaded systems. However, the intelligence and expressiveness of current open-source spoken dialogue models often remain below expectations. Motivated by the success of online reinforcement learning(RL) in other domains, one might attempt to directly apply preference optimization to spoken dialogue models, yet this transfer is non-trivial. We analyze these obstacles from the perspectives of reward modeling and rollout sampling, focusing on how sparse preference supervision interacts with dense speech generation under shared-parameter updates. Based on the analysis, we propose a modality-aware adaptive post-training recipe that makes RL practical for spoken dialogue: it constrains preference updates to the semantic channel and improves acoustic behavior via explicit anchoring, while dynamically regulating their mixture from rollout statistics to avoid unreliable preference gradients. We evaluate the method across multiple spoken dialogue benchmarks and representative architectures, and observe consistent improvements in semantic quality and speech expressiveness. 

---
# Dual-Axis Generative Reward Model Toward Semantic and Turn-taking Robustness in Interactive Spoken Dialogue Models 

**Authors**: Yifu Chen, Shengpeng Ji, Zhengqing Liu, Qian Chen, Wen Wang, Ziqing Wang, Yangzhuo Li, Tianle Liang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14920)  

**Abstract**: Achieving seamless, human-like interaction remains a key challenge for full-duplex spoken dialogue models (SDMs). Reinforcement learning (RL) has substantially enhanced text- and vision-language models, while well-designed reward signals are crucial for the performance of RL. We consider RL a promising strategy to address the key challenge for SDMs. However, a fundamental barrier persists: prevailing automated metrics for assessing interaction quality rely on superficial proxies, such as behavioral statistics or timing-prediction accuracy, failing to provide reliable reward signals for RL. On the other hand, human evaluations, despite their richness, remain costly, inconsistent, and difficult to scale. We tackle this critical barrier by proposing a Dual-Axis Generative Reward Model, which is trained to understand complex interaction dynamics using a detailed taxonomy and an annotated dataset, produces a single score and, crucially, provides separate evaluations for semantic quality and interaction timing. Such dual outputs furnish precise diagnostic feedback for SDMs and deliver a dependable, instructive reward signal suitable for online reinforcement learning. Our model achieves state-of-the-art performance on interaction-quality assessment across a wide spectrum of datasets, spanning synthetic dialogues and complex real-world interactions. 

---
# ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints 

**Authors**: Pei-An Chen, Yong-Ching Liang, Jia-Fong Yeh, Hung-Ting Su, Yi-Ting Chen, Min Sun, Winston Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14902)  

**Abstract**: Intelligent embodied agents should not simply follow instructions, as real-world environments often involve unexpected conditions and exceptions. However, existing methods usually focus on directly executing instructions, without considering whether the target objects can actually be manipulated, meaning they fail to assess available affordances. To address this limitation, we introduce DynAfford, a benchmark that evaluates embodied agents in dynamic environments where object affordances may change over time and are not specified in the instruction. DynAfford requires agents to perceive object states, infer implicit preconditions, and adapt their actions accordingly. To enable this capability, we introduce ADAPT, a plug-and-play module that augments existing planners with explicit affordance reasoning. Experiments demonstrate that incorporating ADAPT significantly improves robustness and task success across both seen and unseen environments. We also show that a domain-adapted, LoRA-finetuned vision-language model used as the affordance inference backend outperforms a commercial LLM (GPT-4o), highlighting the importance of task-aligned affordance grounding. 

---
# Governing Reflective Human-AI Collaboration: A Framework for Epistemic Scaffolding and Traceable Reasoning 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14898)  

**Abstract**: Large language models have advanced rapidly, from pattern recognition to emerging forms of reasoning, yet they remain confined to linguistic simulation rather than grounded understanding. They can produce fluent outputs that resemble reflection, but lack temporal continuity, causal feedback, and anchoring in real-world interaction. This paper proposes a complementary approach in which reasoning is treated as a relational process distributed between human and model rather than an internal capability of either.
Building on recent work on "System-2" learning, we relocate reflective reasoning to the interaction layer. Instead of engineering reasoning solely within models, we frame it as a cognitive protocol that can be structured, measured, and governed using existing systems. This perspective emphasizes collaborative intelligence, combining human judgment and contextual understanding with machine speed, memory, and associative capacity.
We introduce "The Architect's Pen" as a practical method. Like an architect who thinks through drawing, the human uses the model as an external medium for structured reflection. By embedding phases of articulation, critique, and revision into human-AI interaction, the dialogue itself becomes a reasoning loop: human abstraction -> model articulation -> human reflection.
This reframes the question from whether the model can think to whether the human-AI system can reason. The framework enables auditable reasoning traces and supports alignment with emerging governance standards, including the EU AI Act and ISO/IEC 42001. It provides a practical path toward more transparent, controllable, and accountable AI use without requiring new model architectures. 

---
# Toward Agentic RAG for Ukrainian 

**Authors**: Marta Sumyk, Oleksandr Kosovan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14896)  

**Abstract**: We present an initial investigation into Agentic Retrieval-Augmented Generation (RAG) for Ukrainian, conducted within the UNLP 2026 Shared Task on Multi-Domain Document Understanding. Our system combines two-stage retrieval (BGE-M3 with BGE reranking) with a lightweight agentic layer performing query rephrasing and answer-retry loops on top of Qwen2.5-3B-Instruct. Our analysis reveals that retrieval quality is the primary bottleneck: agentic retry mechanisms improve answer accuracy but the overall score remains constrained by document and page identification. We discuss practical limitations of offline agentic pipelines and outline directions for combining stronger retrieval with more advanced agentic reasoning for Ukrainian. 

---
# MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration 

**Authors**: Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan, Bei Li, Chunyang Xiao, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14889)  

**Abstract**: While Chain-of-thought (CoT) reasoning enables LLMs to solve challenging reasoning problems, as KV cache grows linearly with the number of generated tokens, CoT reasoning faces scaling issues in terms of speed and memory usage. In this work, we propose MemoSight (Memory-Foresight-based reasoning), a unified framework that integrates both context compression and multi-token prediction to mitigate the efficiency issues while maintaining CoT reasoning performance. Our framework adopts the same minimalist design for both context compression and multi-token prediction via special tokens and their corresponding position layout tailored to each token type. Comprehensive experiments on four reasoning benchmarks demonstrate that MemoSight reduces the KV cache footprint by up to 66% and accelerates inference by 1.56x, while outperforming existing CoT compression methods. 

---
# Cooperate to Compete: Strategic Data Generation and Incentivization Framework for Coopetitive Cross-Silo Federated Learning 

**Authors**: Thanh Linh Nguyen, Nguyen Van Huynh, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2604.14886)  

**Abstract**: In data-sensitive domains such as healthcare, cross-silo federated learning (CFL) allows organizations to collaboratively train AI models without sharing raw data. However, practical CFL deployments are inherently coopetitive, in which organizations cooperate during model training while competing in downstream markets. In such settings, training contributions, including data volume, quality, and diversity, can improve the global model yet inadvertently strengthen rivals. This dilemma is amplified by non-IID data, which leads to asymmetric learning gains and undermines sustained participation. While existing competition-aware CFL and incentive-design approaches reward organizations based on marginal training contributions, they fail to account for the costs of strengthening competitors. In this paper, we introduce CoCoGen+, a coopetition-compatible data generation and incentivization framework that jointly models non-IID data and inter-organizational competition while endogenizing GenAI-based synthetic data generation as a strategic decision. Specifically, CoCoGen+ formulates each training round as a weighted potential game, where organizations strategically decide how much synthetic data to generate by balancing learning performance gains against computational costs and competition-caused utility losses. We then provide a tractable equilibrium characterization and derive implementable generation strategies to maximize social welfare. To promote long-term collaboration, we integrate a payoff redistribution-based incentive mechanism to compensate organizations for their contributions and competition-caused utility degradation. Experiments on varying learning tasks validate the feasibility of CoCoGen+. The results show how non-IID data, competition intensity, and incentives shape organizational strategies and social welfare, while CoCoGen+ outperforms baselines in efficiency. 

---
# The Missing Knowledge Layer in AI: A Framework for Stable Human-AI Reasoning 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14881)  

**Abstract**: Large language models are increasingly integrated into decision-making in areas such as healthcare, law, finance, engineering, and government. Yet they share a critical limitation: they produce fluent outputs even when their internal reasoning has drifted. A confident answer can conceal uncertainty, speculation, or inconsistency, and small changes in phrasing can lead to different conclusions. This makes LLMs useful assistants but unreliable partners in high-stakes contexts.
Humans exhibit a similar weakness, often mistaking fluency for reliability. When a model responds smoothly, users tend to trust it, even when both model and user are drifting together. This paper is the first in a five-paper research series on stabilising human-AI reasoning. The series proposes a two-layer approach: Parts II-IV introduce human-side mechanisms such as uncertainty cues, conflict surfacing, and auditable reasoning traces, while Part V develops a model-side Epistemic Control Loop (ECL) that detects instability and modulates generation accordingly.
Together, these layers form a missing operational substrate for governance by increasing signal-to-noise at the point of use. Stabilising interaction makes uncertainty and drift visible before enforcement is applied, enabling more precise capability governance. This aligns with emerging compliance expectations, including the EU AI Act and ISO/IEC 42001, by making reasoning processes traceable under real conditions of use.
The central claim is that fluency is not reliability. Without structures that stabilise both human and model reasoning, AI cannot be trusted or governed where it matters most. 

---
# Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-CodeX 

**Authors**: Zhonghao Yang, Yu Li, Yanxu Zhu, Tianyi Zhou, Yuejin Xie, Haoyu Luo, Jing Shao, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14858)  

**Abstract**: As agent systems move into increasingly diverse execution settings, trajectory-level safety evaluation and diagnosis require benchmarks that evolve with them. ATBench is a diverse and realistic agent trajectory benchmark for safety evaluation and diagnosis. This report presents ATBench-Claw and ATBench-CodeX, two domain-customized extensions that carry ATBench into the OpenClaw and OpenAI Codex / Codex-runtime settings. The key adaptation mechanism is to analyze each new setting, customize the three-dimensional Safety Taxonomy over risk source, failure mode, and real-world harm, and then use that customized taxonomy to define the benchmark specification consumed by the shared ATBench construction pipeline. This extensibility matters because agent frameworks remain relatively stable at the architectural level even as their concrete execution settings, tool ecosystems, and product capabilities evolve quickly. Concretely, ATBench-Claw targets OpenClaw-sensitive execution chains over tools, skills, sessions, and external actions, while ATBench-CodeX targets trajectories in the OpenAI Codex / Codex-runtime setting over repositories, shells, patches, dependencies, approvals, and runtime policy boundaries. Our emphasis therefore falls on taxonomy customization, domain-specific risk coverage, and benchmark design under a shared ATBench generation framework. 

---
# TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models 

**Authors**: Yi Zhao, Yajuan Peng, Cam-Tu Nguyen, Zuchao Li, Xiaoliang Wang, Xiaoming Fu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14847)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex tasks through extended chains of thought but suffer from high inference latency due to autoregressive reasoning. Recent work explores using Small Reasoning Models (SRMs) to accelerate LRM inference. In this paper, we systematically characterize the capability boundaries of SRMs and identify three common types of reasoning risks: (1) path divergence, where SRMs lack the strategic ability to construct an initial plan, causing reasoning to deviate from the most probable path; (2) cognitive overload, where SRMs fail to solve particularly difficult steps; and (3) recovery inability, where SRMs lack robust self-reflection and error correction mechanisms. To address these challenges, we propose TrigReason, a trigger-based collaborative reasoning framework that replaces continuous polling with selective intervention. TrigReason delegates most reasoning to the SRM and activates LRM intervention only when necessary-during initial strategic planning (strategic priming trigger), upon detecting extraordinary overconfidence (cognitive offload trigger), or when reasoning falls into unproductive loops (intervention request trigger). The evaluation results on AIME24, AIME25, and GPQA-D indicate that TrigReason matches the accuracy of full LRMs and SpecReason, while offloading 1.70x - 4.79x more reasoning steps to SRMs. Under edge-cloud conditions, TrigReason reduces latency by 43.9\% and API cost by 73.3\%. Our code is available at \href{this https URL}{this https URL} 

---
# Intermediate Layers Encode Optimal Biological Representations in Single-Cell Foundation Models 

**Authors**: Vincenzo Yuto Civale, Roberto Semeraro, Andrew David Bagdanov, Alberto Magi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14838)  

**Abstract**: Current single-cell foundation model benchmarks universally extract final layer embeddings, assuming these represent optimal feature spaces. We systematically evaluate layer-wise representations from scFoundation (100M parameters) and Tahoe-X1 (1.3B parameters) across trajectory inference and perturbation response prediction. Our analysis reveals that optimal layers are task-dependent (trajectory peaks at 60% depth, 31% above final layers) and context-dependent (perturbation optima shift 0-96% across T cell activation states). Notably, first-layer embeddings outperform all deeper layers in quiescent cells, challenging assumptions about hierarchical feature abstraction. These findings demonstrate that "where" to extract features matters as much as "what" the model learns, necessitating systematic layer evaluation tailored to biological task and cellular context rather than defaulting to final-layer embeddings. 

---
# Beyond Literal Summarization: Redefining Hallucination for Medical SOAP Note Evaluation 

**Authors**: Bhavik Vachhani, Kush Shrisvastava, Pranshu Nema, Sai Chiranthan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14829)  

**Abstract**: Evaluating large language models (LLMs) for clinical documentation tasks such as SOAP note generation remains challenging. Unlike standard summarization, these tasks require clinical abstraction, normalization of colloquial language, and medically grounded inference. However, prevailing evaluation methods including automated metrics and LLM as judge frameworks rely on lexical faithfulness, often labeling any information not explicitly present in the transcript as hallucination.
We show that such approaches systematically misclassify clinically valid outputs as errors, inflating hallucination rates and distorting model assessment. Our analysis reveals that many flagged hallucinations correspond to legitimate clinical transformations, including synonym mapping, abstraction of examination findings, diagnostic inference, and guideline consistent care planning.
By aligning evaluation criteria with clinical reasoning through calibrated prompting and retrieval grounded in medical ontologies we observe a significant shift in outcomes. Under a lexical evaluation regime, the mean hallucination rate is 35%, heavily penalizing valid reasoning. With inference aware evaluation, this drops to 9%, with remaining cases reflecting genuine safety concerns. These findings suggest that current evaluation practices over penalize valid clinical reasoning and may measure artifacts of evaluation design rather than true errors, underscoring the need for clinically informed evaluation in high context domains like medicine. 

---
# The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows 

**Authors**: Hyunwoo Kim, Harin Yu, Hanau Yi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14807)  

**Abstract**: The rapid integration of large language models (LLMs) into everyday workflows has transformed how individuals perform cognitive tasks such as writing, programming, analysis, and multilingual communication. While prior research has focused on model reliability, hallucination, and user trust calibration, less attention has been given to how LLM usage reshapes users' perceptions of their own capabilities. This paper introduces the LLM fallacy, a cognitive attribution error in which individuals misinterpret LLM-assisted outputs as evidence of their own independent competence, producing a systematic divergence between perceived and actual capability. We argue that the opacity, fluency, and low-friction interaction patterns of LLMs obscure the boundary between human and machine contribution, leading users to infer competence from outputs rather than from the processes that generate them. We situate the LLM fallacy within existing literature on automation bias, cognitive offloading, and human--AI collaboration, while distinguishing it as a form of attributional distortion specific to AI-mediated workflows. We propose a conceptual framework of its underlying mechanisms and a typology of manifestations across computational, linguistic, analytical, and creative domains. Finally, we examine implications for education, hiring, and AI literacy, and outline directions for empirical validation. We also provide a transparent account of human--AI collaborative methodology. This work establishes a foundation for understanding how generative AI systems not only augment cognitive performance but also reshape self-perception and perceived expertise. 

---
# Diffusion Crossover: Defining Evolutionary Recombination in Diffusion Models via Noise Sequence Interpolation 

**Authors**: Chisatao Kumada, Satoru Hiwa, Tomoyuki Hiroyasu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14790)  

**Abstract**: Interactive Evolutionary Computation (IEC) provides a powerful framework for optimizing subjective criteria such as human preferences and aesthetics, yet it suffers from a fundamental limitation: in high-dimensional generative representations, defining crossover in a semantically consistent manner is difficult, often leading to a mutation-dominated search. In this work, we explicitly define crossover in diffusion models. We propose Diffusion crossover, which formulates evolutionary recombination as step-wise interpolation of noise sequences in the reverse process of Denoising Diffusion Probabilistic Models (DDPMs). By applying spherical linear interpolation (Slerp) to the noise sequences associated with selected parent images, the proposed method generates offspring that inherit characteristics from both parents while preserving the geometric structure of the diffusion process. Furthermore, controlling the time-step range of interpolation enables a principled trade-off between diversity (exploration) and convergence (exploitation). Experimental results using PCA analysis and perceptual similarity metrics (LPIPS) demonstrate that Diffusion crossover produces perceptually smooth and semantically consistent transitions between parent images. Qualitative interactive evolution experiments further confirm that the proposed method effectively supports human-in-the-loop image exploration. These findings suggest a new perspective: diffusion models are not only powerful generators, but also structured evolutionary search spaces in which recombination can be explicitly defined and controlled. 

---
# A Comparative Study of CNN Optimization Methods for Edge AI: Exploring the Role of Early Exits 

**Authors**: Nekane Fernandez, Ivan Valdes, Steven Van Vaerenbergh, Idoia de la Iglesia, Julen Arratibel  

**Link**: [PDF](https://arxiv.org/pdf/2604.14789)  

**Abstract**: Deploying deep neural networks on edge devices requires balancing accuracy, latency, and resource constraints under realistic execution conditions. To fit models within these constraints, two broad strategies have emerged: static compression techniques such as pruning and quantization, which permanently reduce model size, and dynamic approaches such as early-exit mechanisms, which adapt computational cost at runtime. While both families are widely studied in isolation, they are rarely compared under identical conditions on physical hardware. This paper presents a unified deployment-oriented comparison of static compression and dynamic early-exit mechanisms, evaluated on real edge devices using ONNX based inference pipelines. Our results show that static and dynamic techniques offer fundamentally different trade-offs for edge deployment. While pruning and quantization deliver consistent memory footprint reduction, early-exit mechanisms enable input-adaptive computation savings that static methods cannot match. Their combination proves highly effective, simultaneously reducing inference latency and memory usage with minimal accuracy loss, expanding what is achievable at the edge. 

---
# Sequence Search: Automated Sequence Design using Neural Architecture Search 

**Authors**: Rokgi Hong, Hongjun An, Sooyeon Ji, Jongho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14788)  

**Abstract**: Developing an MR sequence is challenging and remains largely constrained by human intuition. Recently, AI-driven approaches have been proposed; however, most require an initial sequence for parameter optimization or extensive training datasets, limiting their general applicability. In this study, we propose "Sequence Search," an automated sequence design framework based on neural architecture search. The method takes tissue properties, imaging parameters, and design objectives as inputs and generates pulse sequences satisfying the design objectives, without requiring prior knowledge of conventional sequence structures. Sequence Search iteratively generates candidate sequences through neural architecture search and optimizes them via a differentiable Bloch simulator and objective-specific loss functions using gradient-based learning. The framework successfully replicated conventional spin-echo, T2-weighted spin-echo, and inversion recovery sequences. Less intuitive solutions were also discovered, such as three-RF spin-echo-like sequences with reduced RF energy and refocusing phases deviating from the conventional Hahn-echo. This work establishes a generalizable framework for automated MR sequence design, highlighting the potential to explore configurations beyond conventional designs based on human intuition. 

---
# CogEvolution: A Human-like Generative Educational Agent to Simulate Student's Cognitive Evolution 

**Authors**: Wei Zhang, Yihang Cheng, Zhirong Ye, Kezhen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14786)  

**Abstract**: Generative Agents, owing to their precise modeling and simulation capabilities of human behavior, have become a pivotal tool in the field of Artificial Intelligence in Education (AIEd) for uncovering complex cognitive processes of learners. However, existing educational agents predominantly rely on static personas to simulate student learning behaviors, neglecting the decisive role of deep cognitive capabilities in learning outcomes during practice interactions. Furthermore, they struggle to characterize the dynamic fluidity of knowledge internalization, transfer, and cognitive state transitions. To overcome this bottleneck, this paper proposes a human-like educational agent capable of simulating student cognitive evolution: CogEvolution. Specifically, we first construct a cognitive depth perceptron based on the Interactive, Constructive, Active, Passive (ICAP) taxonomy from cognitive psychology, achieving precise quantification of learner cognitive engagement. Subsequently, we propose a memory retrieval method based on Item Response Theory (IRT) to simulate the connection and assimilation of new and prior knowledge. Finally, we design a dynamic cognitive update mechanism based on evolutionary algorithms to simulate the real-time integration of student learning behaviors and cognitive evolution processes. Comprehensive evaluations demonstrate that CogEvolution not only significantly outperforms baseline models in behavioral fidelity and learning curve fitting but also uniquely reproduces plausible and robust cognitive evolutionary paths consistent with educational psychology expectations, providing a novel paradigm for constructing highly interpretable educational agents. 

---
# MirrorBench: Evaluating Self-centric Intelligence in MLLMs by Introducing a Mirror 

**Authors**: Shengyu Guo, Tongrui Ye, Jianbo Zhang, Zicheng Zhang, Chunyi Li, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.14785)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has demonstrated remarkable advances in perception and reasoning, suggesting their potential for embodied intelligence. While recent studies have evaluated embodied MLLMs in interactive settings, current benchmarks mainly target capabilities to perceive, understand, and interact with external objects, lacking a systematic evaluation of self-centric intelligence. To address this, we introduce MirrorBench, a simulation-based benchmark inspired by the classical Mirror Self-Recognition (MSR) test in psychology. MirrorBench extends this paradigm to embodied MLLMs through a tiered framework of progressively challenging tasks, assessing agents from basic visual perception to high-level self-representation. Experiments on leading MLLMs show that even at the lowest level, their performance remains substantially inferior to human performance, revealing fundamental limitations in self-referential understanding. Our study bridges psychological paradigms and embodied intelligence, offering a principled framework for evaluating the emergence of general intelligence in large models. Project page: this https URL. 

---
# CoTEvol: Self-Evolving Chain-of-Thoughts for Data Synthesis in Mathematical Reasoning 

**Authors**: Zhuo Wang, Zhuo Zhang, Yafu Li, Yu Cheng, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14768)  

**Abstract**: Large Language Models (LLMs) exhibit strong mathematical reasoning when trained on high-quality Chain-of-Thought (CoT) that articulates intermediate steps, yet costly CoT curation hinders further progress. While existing remedies such as distillation from stronger LLMs and self-synthesis based on test-time search alleviate this issue, they often suffer from diminishing returns or high computing this http URL this work, we propose CoTEvol, a genetic evolutionary framework that casts CoT generation as a population-based search over reasoning this http URL trajectories are iteratively evolved through reflective global crossover at the trajectory level and local mutation guided by uncertainty at the step level, enabling holistic recombination and fine-grained refinement. Lightweight, task-aware fitness functions are designed to guide the evolutionary process toward accurate and diverse reasoning. Empirically, CoTEvol improves correct-CoT synthesis success by over 30% and enhances structural diversity, with markedly improved efficiency. LLMs trained on these evolutionary CoT data achieve an average gain of 6.6% across eight math benchmarks, outperforming previous distillation and self-synthesis approaches. These results underscore the promise of evolutionary CoT synthesis as a scalable and effective method for mathematical reasoning tasks. 

---
# Disentangle-then-Refine: LLM-Guided Decoupling and Structure-Aware Refinement for Graph Contrastive Learning 

**Authors**: Zhaoxing Li, Hai-Feng Zhang, Xiaoming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14746)  

**Abstract**: Conventional Graph Contrastive Learning (GCL) on Text-Attributed Graphs (TAGs) relies on blind stochastic augmentations, inadvertently entangling task-relevant signals with noise. We propose SDM-SCR, a robust framework anchored in Approximate Orthogonal Decomposition. First, the Semantic Decoupling Module (SDM) leverages the instruction-following capability of Large Language Models (LLMs) to actively parse raw attributes into asymmetric, task-oriented signal and noise views. This shifts the paradigm from random perturbation to semantic-aware disentanglement. Subsequently, Semantic Consistency Regularization (SCR) exploits the spectral observation that semantic signals are topologically smooth while residual noise is high-frequency. SCR functions as a selective spectral filter, enforcing consistency only on the signal subspace to eliminate LLM hallucinations without over-smoothing. This ``Disentangle-then-Refine'' mechanism ensures rigorous signal purification. Extensive experiments demonstrate that SDM-SCR achieves SOTA performance in accuracy and efficiency. 

---
# Personalized and Context-Aware Transformer Models for Predicting Post-Intervention Physiological Responses from Wearable Sensor Data 

**Authors**: Esther Brown, Victoria Dean, Finale Doshi-Velez  

**Link**: [PDF](https://arxiv.org/pdf/2604.14738)  

**Abstract**: Consumer wearables enable continuous measurement of physiological data related to stress and recovery, but turning these streams into actionable, personalized stress-management recommendations remains a challenge. In practice, users often do not know how a given intervention, defined as an activity intended to reduce stress, will affect heart rate (HR), heart rate variability (HRV), or inter-beat intervals (BBI) over the next 15 to 120 minutes. We present a framework that predicts post-intervention trajectories and the direction of change for these physiological indicators across time windows. Our methodology combines a Transformer model for multi-horizon trajectories of percent change relative to a pre-intervention baseline, direction-of-change calls (positive, negative, or neutral) at each horizon, and an empirical study using wearable sensor data overlaid with user-tagged events and interventions. This proof of concept shows that personalized post-intervention prediction is feasible. We encourage future integration into stress-management tools for personalized intervention recommendations tailored to each person's day following further validation in larger studies and, where applicable, appropriate regulatory review. 

---
# The Agentification of Scientific Research: A Physicist's Perspective 

**Authors**: Xiao-Liang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14718)  

**Abstract**: This article argues that the most important significance of the AI revolution, especially the rise of large language models, lies not simply in automation, but in a fundamental change in how complex information and human know-how are carried, replicated, and shared. From this perspective, AI for Science is especially important because it may transform not only the efficiency of research, but also the structure of scientific collaboration, discovery, publishing, and evaluation. The article outlines a gradual path from AI as a research tool to AI as a scientific collaborator, and discusses how AI is likely to fundamentally reshape scientific publication. It also argues that continuous learning and diversity of ideas are essential if AI is to play a meaningful role in original scientific discovery. 

---
# Layered Mutability: Continuity and Governance in Persistent Self-Modifying Agents 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2604.14717)  

**Abstract**: Persistent language-model agents increasingly combine tool use, tiered memory, reflective prompting, and runtime adaptation. In such systems, behavior is shaped not only by current prompts but by mutable internal conditions that influence future action. This paper introduces layered mutability, a framework for reasoning about that process across five layers: pretraining, post-training alignment, self-narrative, memory, and weight-level adaptation. The central claim is that governance difficulty rises when mutation is rapid, downstream coupling is strong, reversibility is weak, and observability is low, creating a systematic mismatch between the layers that most affect behavior and the layers humans can most easily inspect. I formalize this intuition with simple drift, governance-load, and hysteresis quantities, connect the framework to recent work on temporal identity in language-model agents, and report a preliminary ratchet experiment in which reverting an agent's visible self-description after memory accumulation fails to restore baseline behavior. In that experiment, the estimated identity hysteresis ratio is 0.68. The main implication is that the salient failure mode for persistent self-modifying agents is not abrupt misalignment but compositional drift: locally reasonable updates that accumulate into a behavioral trajectory that was never explicitly authorized. 

---
# SGA-MCTS: Decoupling Planning from Execution via Training-Free Atomic Experience Retrieval 

**Authors**: Xin Xie, Dongyun Xue, Wuguannan Yao, Mingxiao Feng, Wengang Zhou, Xiang Qi, Houqiang Li, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14712)  

**Abstract**: LLM-powered systems require complex multi-step decision-making abilities to solve real-world tasks, yet current planning approaches face a trade-off between the high latency of inference-time search and the limited generalization of supervised fine-tuning. To address this limitation, we introduce \textbf{SGA-MCTS}, a framework that casts LLM planning as non-parametric retrieval. Offline, we leverage Monte Carlo Tree Search (MCTS) to explore the solution space and distill high-fidelity trajectories into State-Goal-Action (SGA) atoms. These atoms are de-lexicalized primitives that abstract concrete entities into symbolic slots, preserving reusable causal logic while discarding domain-specific noise. Online, a retrieval-augmented agent employs a hybrid symbolic-semantic mechanism to fetch relevant SGAs and re-ground them into the current context as soft reasoning hints. Empirical results on complex benchmarks demonstrate that this paradigm enables frozen, open-weights models to match the performance of SOTA systems (e.g., GPT-5) without task-specific fine-tuning. By effectively amortizing the heavy computational cost of search, SGA-MCTS achieves System 2 reasoning depth at System 1 inference speeds, rendering autonomous planning both scalable and real-time feasible. 

---
# HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks 

**Authors**: Fan Cui, Hongyuan Hou, Zizhang Luo, Chenyun Yin, Yun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14709)  

**Abstract**: Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents. 

---
# SynHAT: A Two-stage Coarse-to-Fine Diffusion Framework for Synthesizing Human Activity Traces 

**Authors**: Rongchao Xu, Lin Jiang, Dahai Yu, Ximiao Li, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14705)  

**Abstract**: Human activity traces (HATs) are critical for many applications, including human mobility modeling and point-of-interest (POI) recommendation. However, growing privacy concerns have severely limited access to authentic large-scale HAT datasets. Recent advances in generative AI provide new opportunities to synthesize realistic and privacy-preserving HATs for such applications. Yet two major challenges remain: (i) HATs are highly irregular and dynamic, with long and varying time intervals, making it difficult to capture their complex spatio-temporal dependencies and underlying distributions; and (ii) generative models are often computationally expensive, making long-term, fine-grained HAT synthesis inefficient. To address these challenges, we propose SynHAT, a computationally efficient coarse-to-fine HAT synthesis framework built on a novel spatio-temporal denoising diffusion model. In Stage 1, we develop Coarse-HADiff, which models the overall spatio-temporal dependencies of coarse-grained latent spatio-temporal traces. It incorporates a novel Latent Spatio-Temporal U-Net with dual Drift-Jitter branches to jointly model smooth spatial transitions and temporal variations during denoising. In Stage 2, we introduce a three-step pipeline consisting of Behavior Pattern Extraction, Fine-HADiff, which shares the same architecture as Coarse-HADiff, and Semantic Alignment to generate fine-grained latent spatio-temporal traces from the Stage 1 outputs. We extensively evaluate SynHAT in terms of data fidelity, utility, privacy, robustness, and scalability. Experiments on real-world HAT datasets from four cities across three countries show that SynHAT substantially outperforms state-of-the-art baselines, achieving 52% and 33% improvements on spatial and temporal metrics, respectively. 

---
# CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations 

**Authors**: Xiangning Yu, Yuwei Guo, Yuqi Hou, Xiao Xue, Qun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.14691)  

**Abstract**: LLM-empowered agent simulations are increasingly used to study social emergence, yet the micro-to-macro causal mechanisms behind macro outcomes often remain unclear. This is challenging because emergence arises from intertwined agent interactions and meso-level feedback and nonlinearity, making generative mechanisms hard to disentangle. To this end, we introduce \textbf{\textsc{CAMO}}, an automated \textbf{Ca}usal discovery framework from \textbf{M}icr\textbf{o} behaviors to \textbf{M}acr\textbf{o} Emergence in LLM agent simulations. \textsc{CAMO} converts mechanistic hypotheses into computable factors grounded in simulation records and learns a compact causal representation centered on an emergent target $Y$. \textsc{CAMO} outputs a computable Markov boundary and a minimal upstream explanatory subgraph, yielding interpretable causal chains and actionable intervention levers. It also uses simulator-internal counterfactual probing to orient ambiguous edges and revise hypotheses when evidence contradicts the current view. Experiments across four emergent settings demonstrate the promise of \textsc{CAMO}. 

---
# M2-PALE: A Framework for Explaining Multi-Agent MCTS--Minimax Hybrids via Process Mining and LLMs 

**Authors**: Yiyu Qian, Liyuan Zhao, Tim Miller  

**Link**: [PDF](https://arxiv.org/pdf/2604.14687)  

**Abstract**: Monte-Carlo Tree Search (MCTS) is a fundamental sampling-based search algorithm widely used for online planning in sequential decision-making domains. Despite its success in driving recent advances in artificial intelligence, understanding the behavior of MCTS agents remains a challenge for both developers and users. This difficulty stems from the complex search trees produced through the simulation of numerous future states and their intricate relationships. A known weakness of standard MCTS is its reliance on highly selective tree construction, which may lead to the omission of crucial moves and a vulnerability to tactical traps. To resolve this, we incorporate shallow, full-width Minimax search into the rollout phase of multi-agent MCTS to enhance strategic depth. Furthermore, to demystify the resulting decision-making logic, we introduce \textsf{M2-PALE} (MCTS--Minimax Process-Aided Linguistic Explanations). This framework employs process mining techniques, specifically the Alpha Miner, iDHM, and Inductive Miner algorithms, to extract underlying behavioral workflows from agent execution traces. These process models are then synthesized by LLMs to generate human-readable causal and distal explanations. We demonstrate the efficacy of our approach in a small-scale checkers environment, establishing a scalable foundation for interpreting hybrid agents in increasingly complex strategic domains. 

---
# DR$^{3}$-Eval: Towards Realistic and Reproducible Deep Research Evaluation 

**Authors**: Qianqian Xie, Qingheng Xiong, He Zhu, Tiantian Xia, Xueming Han, Fanyu Meng, Jiakai Wang, Zhiqi Bai, Chengkang Jiang, Zhaohui Wang, Yubin Guo, Yuqing Wen, Jiayang Mao, Zijie Zhang, Shihao Li, Yanghai Wang, Yuxiang Ren, Junlan Feng, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14683)  

**Abstract**: Deep Research Agents (DRAs) aim to solve complex, long-horizon research tasks involving planning, retrieval, multimodal understanding, and report generation, yet their evaluation remains challenging due to dynamic web environments and ambiguous task definitions. We propose DR$^{3}$-Eval, a realistic and reproducible benchmark for evaluating deep research agents on multimodal, multi-file report generation. DR$^{3}$-Eval is constructed from authentic user-provided materials and paired with a per-task static research sandbox corpus that simulates open-web complexity while remaining fully verifiable, containing supportive documents, distractors, and noise. Moreover, we introduce a multi-dimensional evaluation framework measuring Information Recall, Factual Accuracy, Citation Coverage, Instruction Following, and Depth Quality, and validate its alignment with human judgments. Experiments with our developed multi-agent system DR$^{3}$-Agent based on multiple state-of-the-art language models demonstrate that DR$^{3}$-Eval is highly challenging and reveals critical failure modes in retrieval robustness and hallucination control. Our code and data are publicly available. 

---
# Acceptance Dynamics Across Cognitive Domains in Speculative Decoding 

**Authors**: Saif Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2604.14682)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batched forward pass. Despite the growing body of work on speculative methods, the degree to which the cognitive characteristics of a task affect acceptance probability remains largely unexplored. We present an empirical study of tree-based speculative decoding acceptance dynamics. Our study spans four well-established NLP benchmark domains: code generation, mathematical reasoning, logical reasoning, and open-ended chat. For this, we use TinyLlama-1.1B as the draft model against Llama-2-7B-Chat-GPTQ as the target. Over 99,768 speculative nodes collected from 200 prompts, we derive per-domain acceptance rates, expected accepted lengths, depth-acceptance profiles, and entropy-acceptance correlations. We find that task type is a stronger predictor of acceptance than tree depth. Furthermore, only the chat domain consistently yields an expected accepted length exceeding 1.0 token per step. We also show that the entropy-acceptance correlation is consistently negative but weak across all domains (rho in [-0.20, -0.15]). Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register. These findings have direct implications for domain-aware speculation budgets and draft-model selection strategies. Index Terms--speculative decoding, large language model inference, tree attention, draft model, acceptance probability, LLM efficiency 

---
# Rethinking Patient Education as Multi-turn Multi-modal Interaction 

**Authors**: Zonghai Yao, Zhipeng Tang, Chengtao Lin, Xiong Luo, Benlu Wang, Juncheng Huang, Chin Siang Ong, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14656)  

**Abstract**: Most medical multimodal benchmarks focus on static tasks such as image question answering, report generation, and plain-language rewriting. Patient education is more demanding: systems must identify relevant evidence across images, show patients where to look, explain findings in accessible language, and handle confusion or distress. Yet most patient education work remains text-only, even though combined image-and-text explanations may better support understanding. We introduce MedImageEdu, a benchmark for multi-turn, evidence-grounded radiology patient education. Each case provides a radiology report with report text and case images. A DoctorAgent interacts with a PatientAgent, conditioned on a hidden profile that captures factors such as education level, health literacy, and personality. When a patient question would benefit from visual support, the DoctorAgent can issue drawing instructions grounded in the report, case images, and the current question to a benchmark-provided drawing tool. The tool returns image(s), after which the DoctorAgent produces a final multimodal response consisting of the image(s) and a grounded plain-language explanation. MedImageEdu contains 150 cases from three sources and evaluates both the consultation process and the final multimodal response along five dimensions: Consultation, Safety and Scope, Language Quality, Drawing Quality, and Image-Text Response Quality. Across representative open- and closed-source vision-language model agents, we find three consistent gaps: fluent language often outpaces faithful visual grounding, safety is the weakest dimension across disease categories, and emotionally tense interactions are harder than low education or low health literacy. MedImageEdu provides a controlled testbed for assessing whether multimodal agents can teach from evidence rather than merely answer from text. 

---
# AgentGA: Evolving Code Solutions in Agent-Seed Space 

**Authors**: David Y.Y. Tan, Kellie Chin, Jingxian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14655)  

**Abstract**: We present AgentGA, a framework that evolves autonomous code-generation runs by optimizing the agent seed: the task prompt plus optional parent archives that initialize a fresh workspace. The outer loop searches over these reusable starting conditions rather than editing code directly. Each generation launches a fresh autonomous run from a reset workspace, while selected parent archives provide inherited artifacts that descendants can inspect and reuse. AgentGA couples a population-level genetic algorithm with long-horizon agents; selection uses deterministic 1:1 elite tournaments and operator allocation is adapted online with a modified Hedge controller. We instantiate the approach for tabular AutoML on the 16-competition Weco-Kaggle Lite benchmark. On the 10 benchmark runs reported here, AgentGA averages 74.52% Exceeds % of Human versus 54.15% for AIDE. Across 1135 parent-child comparisons, descendants given parent archives outperform runs started from scratch, indicating that inherited artifacts improve later autonomous runs. These findings support agent-seed optimization as a practical design point for autonomous code-search systems. 

---
# Targeted Exploration via Unified Entropy Control for Reinforcement Learning 

**Authors**: Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Ge Lan, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14646)  

**Abstract**: Recent advances in reinforcement learning (RL) have improved the reasoning capabilities of large language models (LLMs) and vision-language models (VLMs). However, the widely used Group Relative Policy Optimization (GRPO) consistently suffers from entropy collapse, causing the policy to converge prematurely and lose diversity. Existing exploration methods introduce additional bias or variance during exploration, making it difficult to maintain optimization stability. We propose Unified Entropy Control for Reinforcement Learning (UEC-RL), a framework that provides targeted mechanisms for exploration and stabilization. UEC-RL activates more exploration on difficult prompts to search for potential and valuable reasoning trajectories. In parallel, a stabilizer prevents entropy from growing uncontrollably, thereby keeping training stable as the model consolidates reliable behaviors. Together, these components expand the search space when needed while maintaining robust optimization throughout training. Experiments on both LLM and VLM reasoning tasks show consistent gains over RL baselines on both Pass@1 and Pass@$k$. On Geometry3K, UEC-RL achieves a 37.9\% relative improvement over GRPO, indicating that it sustains effective exploration without compromising convergence and underscoring UEC-RL as a key for scaling RL-based reasoning in large models. Our code is available at this https URL. 

---
# Learning to Draw ASCII Improves Spatial Reasoning in Language Models 

**Authors**: Shiyuan Huang, Li Liu, Jincheng He, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2604.14641)  

**Abstract**: When faced with complex spatial problems, humans naturally sketch layouts to organize their thinking, and the act of drawing further sharpens their understanding. In this work, we ask whether a similar principle holds for Large Language Models (LLMs): can learning to construct explicit visual layouts from spatial descriptions instill genuine spatial understanding? We introduce Text2Space, a dataset that pairs natural language descriptions with ground-truth ASCII grid layouts and spatial QA pairs, enabling us to separate failures in constructing spatial representations from failures in reasoning over them. We adopt ASCII because it is human-readable, operates entirely within the token space of language models, and encodes spatial relations in a structurally verifiable form. Our evaluation reveals a pronounced "Read-Write Asymmetry": LLMs interpret ASCII representations effectively but struggle to produce them from text, and these construction errors propagate to incorrect answers downstream. To address this limitation, we train models on layout construction (Text$\rightarrow$ASCII) and find that it significantly improves spatial reasoning from text alone, even without producing any ASCII at inference time. Combining construction with comprehension training further amplifies these gains. Crucially, these improvements transfer to three external spatial reasoning benchmarks, demonstrating that, much as sketching sharpens human spatial thinking, learning to construct explicit layouts instills spatial understanding that generalizes beyond the training format. 

---
# A Parallel Approach to Counting Exact Covers Based on Decomposability Property 

**Authors**: Liangda Fang, Yaohui Luo, Delong Li, Xuanxiang Huang, Quanlong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14627)  

**Abstract**: The exact cover problem is a classical NP-hard problem with broad applications in the area of AI. Algorithm DXZ is a method to count exact covers representing by zero-suppressed binary decision diagrams (ZBDDs). In this paper, we propose a zero-suppressed variant of decision decomposable negation normal form (in short, decision-ZDNNF), which is strictly more succinct than ZBDDs. We then design a novel parallel algorithm, namely DXD, which constructs a decision-ZDNNF representing the set of all exact covers. Furthermore, we improve DXD by dynamically updating connected components. The experimental results demonstrate that the improved DXD algorithm outperforms all of state-of-the-art methods. 

---
# CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors 

**Authors**: Yubin Kim, Salman Rahman, Samuel Schmidgall, Chunjong Park, A. Ali Heydari, Ahmed A. Metwally, Hong Yu, Xin Liu, Xuhai Xu, Yuzhe Yang, Maxwell A. Xu, Zhihan Zhang, Cynthia Breazeal, Tim Althoff, Petar Sirkovic, Ivor Rendulic, Annalisa Pawlosky, Nicolas Stroppa, Juraj Gottweis, Elahe Vedadi, Alan Karthikesalingam, Pushmeet Kohli, Vivek Natarajan, Mark Malhotra, Shwetak Patel, Hae Won Park, Hamid Palangi, Daniel McDuff  

**Link**: [PDF](https://arxiv.org/pdf/2604.14615)  

**Abstract**: Scientific discovery in digital health requires converting continuous physiological signals from wearable devices into clinically actionable biomarkers. We introduce CoDaS (AI Co-Data-Scientist), a multi-agent system that structures biomarker discovery as an iterative process combining hypothesis generation, statistical analysis, adversarial validation, and literature-grounded reasoning with human oversight using large-scale wearable datasets. Across three cohorts totaling 9,279 participant-observations, CoDaS identified 41 candidate digital biomarkers for mental health and 25 for metabolic outcomes, each subjected to an internal validation battery spanning replication, stability, robustness, and discriminative power. Across two independent depression cohorts, CoDaS surfaced circadian instability-related features in both datasets, reflected in sleep duration variability (DWB, \rho = 0.252, p < 0.001) and sleep onset variability (GLOBEM, \rho = 0.126, p < 0.001). In a metabolic cohort, CoDaS derived a cardiovascular fitness index (steps/resting heart rate; \rho = -0.374, p < 0.001), and recovered established clinical associations, including the hepatic function ratio (AST/ALT; \rho = -0.375, p < 0.001), a known correlate of insulin resistance. Incorporating CoDaS-derived features alongside demographic variables led to modest but consistent improvements in predictive performance, with cross-validated \Delta R^2 increases of 0.040 for depression and 0.021 for insulin resistance. These findings suggest that CoDaS enables systematic and traceable hypothesis generation and prioritization for biomarker discovery from large-scale wearable data. 

---
# El Agente Forjador: Task-Driven Agent Generation for Quantum Simulation 

**Authors**: Zijian Zhang, Aiwei Yin, Amaan Baweja, Jiaru Bai, Ignacio Gustin, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2604.14609)  

**Abstract**: AI for science promises to accelerate the discovery process. The advent of large language models (LLMs) and agentic workflows enables the expediting of a growing range of scientific tasks. However, most of the current generation of agentic systems depend on static, hand-curated toolsets that hinder adaptation to new domains and evolving libraries. We present El Agente Forjador, a multi-agent framework in which universal coding agents autonomously forge, validate, and reuse computational tools through a four-stage workflow of tool analysis, tool generation, task execution, and iterative solution evaluation. Evaluated across 24 tasks spanning quantum chemistry and quantum dynamics on five coding agent setups, we compare three operating modes: zero-shot generation of tools per task, reuse of a curriculum-built toolset, and direct problem-solving with the coding agents as the baseline. We find that our tool generation and reuse framework consistently improves accuracy over the baseline. We also show that reusing a toolset built by a stronger coding agent can reduce API cost and substantially raises the solution quality for weaker coding agents. Case studies further demonstrate that tools forged for different domains can be combined to solve hybrid tasks. Taken together, these results show that LLM-based agents can use their scientific knowledge and coding capabilities to autonomously build reusable scientific tools, pointing toward a paradigm in which agent capabilities are defined by the tasks they are designed to solve rather than by explicitly engineered implementations. 

---
# GDPR Auto-Formalization with AI Agents and Human Verification 

**Authors**: Ha Thanh Nguyen, Wachara Fungwacharakorn, Sabine Wehnert, May Myo Zin, Yuntao Kong, Jieying Xue, Michał Araszkiewicz, Randy Goebel, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14607)  

**Abstract**: We study the overall process of automatic formalization of GDPR provisions using large language models, within a human-in-the-loop verification framework. Rather than aiming for full autonomy, we adopt a role-specialized workflow in which LLM-based AI components, operating in a multi-agent setting with iterative feedback, generate legal scenarios, formal rules, and atomic facts. This is coupled with independent verification modules which include human reviewers' assessment of representational, logical, and legal correctness. Using this approach, we construct a high-quality dataset to be used for GDPR auto-formalization, and analyze both successful and problematic cases. Our results show that structured verification and targeted human oversight are essential for reliable legal formalization, especially in the presence of legal nuance and context-sensitive reasoning. 

---
# Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.14585)  

**Abstract**: Prompt optimization in compound AI systems is statistically indistinguishable from a coin flip: across 72 optimization runs on Claude Haiku (6 methods $\times$ 4 tasks $\times$ 3 repeats), 49% score below zero-shot; on Amazon Nova Lite, the failure rate is even higher. Yet on one task, all six methods improve over zero-shot by up to $+6.8$ points. What distinguishes success from failure? We investigate with 18,000 grid evaluations and 144 optimization runs, testing two assumptions behind end-to-end optimization tools like TextGrad and DSPy: (A) individual prompts are worth optimizing, and (B) agent prompts interact, requiring joint optimization. Interaction effects are never significant ($p > 0.52$, all $F < 1.0$), and optimization helps only when the task has exploitable output structure -- a format the model can produce but does not default to. We provide a two-stage diagnostic: an \$80 ANOVA pre-test for agent coupling, and a 10-minute headroom test that predicts whether optimization is worthwhile -- turning a coin flip into an informed decision. 

---
# Enhancing Mental Health Counseling Support in Bangladesh using Culturally-Grounded Knowledge 

**Authors**: Md Arid Hasan, Azhagu Meena SP, Aditya Khan, Abu Md Akteruzzaman Bhuiyan, Helal Uddin Ahmed, Joysree Debi, Farig Sadeque, Annie En-Shiun Lee, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.14576)  

**Abstract**: Large language models (LLMs) show promise in generating supportive responses for mental health and counseling applications. However, their responses often lack cultural sensitivity, contextual grounding, and clinically appropriate guidance. This work addresses the gap of how to systematically incorporate domain-specific, clinically validated knowledge into LLMs to improve counseling quality. We utilize and compare two approaches, retrieval-augmented generation (RAG) and a knowledge graph (KG)-based method, designed to support para-counselors. Our KG is constructed manually and clinically validated, capturing causal relationships between stressors, interventions, and outcomes, with contributions from multidisciplinary people. We evaluated multiple LLMs in both settings using BERTScore F1 and SBERT cosine similarity, as well as human evaluation across five metrics, which is designed to directly measure the effectiveness of counseling beyond similarity at the surface level. The results show that KG-based approaches consistently improve contextual relevance, clinical appropriateness, and practical usability compared to RAG alone, demonstrating that structured, expert-validated knowledge plays a critical role in addressing LLMs limitations in counseling tasks. 

---
# MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation 

**Authors**: Pengfei Li, Shijie Wang, Fangyuan Li, Yikun Fu, Kaifeng Liu, Kaiyan Zhang, Dazhi Zhang, Yuqiang Li, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.14564)  

**Abstract**: Reinforcement learning (RL) paradigms have demonstrated strong performance on reasoning-intensive tasks such as code generation. However, limited trajectory diversity often leads to diminishing returns, which constrains the achievable performance ceiling. Search-enhanced RL alleviates this issue by introducing structured exploration, which remains constrained by the single-agent policy priors. Meanwhile, leveraging multiple interacting policies can acquire more diverse exploratory signals, but existing approaches are typically decoupled from structured search. We propose \textbf{MARS$^2$} (Multi-Agent Reinforced Tree-Search Scaling), a unified RL framework in which multiple independently-optimized agents collaborate within a shared tree-structured search environment. MARS$^2$ models the search tree as a learnable multi-agent interaction environment, enabling heterogeneous agents to collaboratively generate and refine candidate solutions within a shared search topology. To support effective learning, we introduce a path-level group advantage formulation based on tree-consistent reward shaping, which facilitates effective credit assignment across complex search trajectories. Experiments on code generation benchmarks show that MARS$^2$ consistently improves performance across diverse model combinations and training settings, demonstrating the effectiveness of coupling multi-agent collaboration with tree search for enhancing reinforcement learning. Our code is publicly available at this https URL. 

---
# TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification 

**Authors**: Adam Rida  

**Link**: [PDF](https://arxiv.org/pdf/2604.14531)  

**Abstract**: Every call to an LLM classification endpoint produces a labeled input-output pair already retained in production logs. These pairs constitute a free, growing training set: a lightweight surrogate trained on them can absorb a significant portion of future traffic at near-zero marginal inference cost. The open questions are when the surrogate is reliable enough to deploy, what it handles versus defers, and how that boundary evolves as data accumulates.
We introduce TRACER (Trace-based Adaptive Cost-Efficient Routing), an open-source system that trains ML surrogates on an LLM's own production traces and governs deployment through a parity gate: the surrogate is activated only when its agreement with the LLM exceeds a user-specified threshold {\alpha}. To make the routing boundary transparent, TRACER generates interpretability artifacts describing which input regions the surrogate handles, where it plateaus, and why it defers.
On a 77-class intent benchmark with a Sonnet 4.6 teacher, TRACER achieves 83-100% surrogate coverage depending on the quality target {\alpha}; on a 150-class benchmark, the surrogate fully replaces the teacher. On a natural language inference task, the parity gate correctly refuses deployment because the embedding representation cannot support reliable separation. The system is available as open-source software. 

---
# Dissecting Failure Dynamics in Large Language Model Reasoning 

**Authors**: Wei Zhu, Jian Zhang, Lixing Yu, Kun Yue, Zhiwen Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14528)  

**Abstract**: Large Language Models (LLMs) achieve strong performance through extended inference-time deliberation, yet how their reasoning failures arise remains poorly understood. By analyzing model-generated reasoning trajectories, we find that errors are not uniformly distributed but often originate from a small number of early transition points, after which reasoning remains locally coherent but globally incorrect. These transitions coincide with localized spikes in token-level entropy, and alternative continuations from the same intermediate state can still lead to correct solutions. Based on these observations, we introduce GUARD, a targeted inference-time framework that probes and redirects critical transitions using uncertainty signals. Empirical evaluations across multiple benchmarks confirm that interventions guided by these failure dynamics lead to more reliable reasoning outcomes. Our findings highlight the importance of understanding when and how reasoning first deviates, complementing existing approaches that focus on scaling inference-time computation. 

---
# Quantifying Cross-Query Contradictions in Multi-Query LLM Reasoning 

**Authors**: Rohit Kumar Salla, Ramya Manasa Amancherla, Manoj Saravanan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14525)  

**Abstract**: Large language models frequently produce mutually inconsistent answers when reasoning over multiple related queries. We study case-file logical consistency: maintaining a globally satisfiable belief state across interdependent queries. We introduce a benchmark of 390 multi-query reasoning instances with entailment/contradiction/unknown labels and propose set-level metrics including Case Satisfiability Rate, Contradiction Density and Revision Cost. Our solver-augmented approach extracts commitments, verifies global satisfiability and performs counterexample-guided repair. Across four reasoning domains, our method substantially reduces cross-query contradictions (SetCons: 0.56 to 0.94) while preserving per-query accuracy, demonstrating that global coherence is critical for robust multi-query reasoning. 

---
# Mind DeepResearch Technical Report 

**Authors**: MindDR Team, Li Auto Inc  

**Link**: [PDF](https://arxiv.org/pdf/2604.14518)  

**Abstract**: We present \textbf{Mind DeepResearch (MindDR)}, an efficient multi-agent deep research framework that achieves leading performance with only \textasciitilde30B-parameter models through a meticulously designed data synthesis and multi-stage training pipeline. The core innovation of MindDR lies in a collaborative three-agent architecture (Planning Agent, DeepSearch Agent, and Report Agent) and a four-stage agent-specialized training pipeline comprising SFT cold-start, Search-RL, Report-RL and preference alignment. With this regime, MindDR demonstrates competitive performance even with \textasciitilde30B-scale models. Specifically, MindDR achieves 45.7\% on BrowseComp-ZH, 42.8\% on BrowseComp, 46.5\% on WideSearch, 75.0\% on xbench-DS, and 52.5 on DeepResearch Bench, outperforming comparable-scale open-source agent systems and rivaling larger-scale models. MindDR has been deployed as an online product in Li Auto. Furthermore, we introduce \textbf{MindDR Bench}, a curated benchmark of 500 real-world Chinese queries from our internal product user interactions, evaluated through a comprehensive multi-dimensional rubric system rather than relying on a single RACE metric. On MindDR Bench, MindDR achieves a state-of-the-art score of 51.8. 

---
# Perspective on Bias in Biomedical AI: Preventing Downstream Healthcare Disparities 

**Authors**: Michal Rosen-Zvi, Yoav Kan-Tor, Michael Danziger, Agata Ferretti, Javier Aula-Blasco, Julia Falcao, Ron Shamir, Mordechai Muszkat  

**Link**: [PDF](https://arxiv.org/pdf/2604.14514)  

**Abstract**: Healthcare disparities persist across socioeconomic boundaries, often attributed to unequal access to screening, diagnostics, and therapeutics. However, this perspective highlights that critical biases can emerge much earlier, during data collection and research prioritization, long before clinical implementation in cases where the focus of the studies and the data that is collected is at the molecular level. A vast number of studies focus on collecting omics data but the demographic information associated with these datasets is often not reported in the studies, and when it is reported, it shows big biases. An automated analysis of 4719 PubMed-indexed omics publications from 2015 to 2024 reveals that only a small fraction report ancestry or ethnicity information, with ancestry reporting improving slightly. Analysis of large-scale datasets commonly used for model training, such as CellxGene and GEO, reveals substantial population bias where European-ancestry data dominates. As biomedical foundation models become central to biomedical discovery with a paradigm in which base models are pretrained on large datasets and reusing them time and again for many different downstream tasks, they risk perpetuating or amplifying these early-stage biases, leading to cascading inequities that regulatory interventions cannot fully reverse. We propose a community-wide focus on three foundational principles: Provenance, Openness, and Evaluation Transparency to improve equity and robustness in biomedical AI. This approach aims to foster biomedical innovation that more effectively serves underserved populations and improves health outcomes. 

---
# Geometric Metrics for MoE Specialization: From Fisher Information to Early Failure Detection 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14500)  

**Abstract**: Expert specialization is fundamental to Mixture-of-Experts (MoE) model success, yet existing metrics (cosine similarity, routing entropy) lack theoretical grounding and yield inconsistent conclusions under reparameterization. We present an information-geometric framework providing the first rigorous characterization of MoE specialization dynamics. Our key insight is that expert routing distributions evolve on the probability simplex equipped with the Fisher information metric, enabling formal analysis via Riemannian geometry. We prove that standard heuristic metrics violate parameterization invariance (Theorem 1), establish that specialization corresponds to geodesic flow with quantified approximation bounds (Theorem 2), and derive a failure predictor with theoretical threshold justification (Theorem 3). The framework introduces two principled metrics: Fisher Specialization Index (FSI) achieving r=0.91+/-0.02 correlation with downstream performance, and Fisher Heterogeneity Score (FHS) predicting training failure at 10% completion with AUC=0.89+/-0.03 -- outperforming validation-loss-based early stopping by 23% while requiring 40x fewer compute cycles. We validate intervention protocols achieving 87% recovery rate when FHS>1 is detected. Comprehensive experiments across language modeling (WikiText-103, C4), vision MoE (ImageNet), and scaling studies (8-64 experts, 125M-2.7B parameters) validate our theoretical predictions. 

---
# Improving Machine Learning Performance with Synthetic Augmentation 

**Authors**: Mel Sohm, Charles Dezons, Sami Sellami, Oscar Ninou, Axel Pincon  

**Link**: [PDF](https://arxiv.org/pdf/2604.14498)  

**Abstract**: Synthetic augmentation is increasingly used to mitigate data scarcity in financial machine learning, yet its statistical role remains poorly understood. We formalize synthetic augmentation as a modification of the effective training distribution and show that it induces a structural bias--variance trade-off: while additional samples may reduce estimation error, they may also shift the population objective whenever the synthetic distribution deviates from regions relevant under evaluation. To isolate informational gains from mechanical sample-size effects, we introduce a size-matched null augmentation and a finite-sample, non-parametric block permutation test that remains valid under weak temporal dependence.
We evaluate this framework in both controlled Markov-switching environments and real financial datasets, including high-frequency option trade data and a daily equity panel. Across generators spanning bootstrap, copula-based models, variational autoencoders, diffusion models, and TimeGAN, we vary augmentation ratio, model capacity, task type, regime rarity, and signal-to-noise. We show that synthetic augmentation is beneficial only in variance-dominant regimes, such as persistent volatility forecasting-while it deteriorates performance in bias-dominant settings, including near-efficient directional prediction. Rare-regime targeting can improve domain-specific metrics but may conflict with unconditional permutation inference. Our results provide a structural perspective on when synthetic data improves financial learning performance and when it induces persistent distributional distortion. 

---
# Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference 

**Authors**: Nenad Banfic, David Fan, Kunal Vaishnavi, Sam Kemp, Sunghoon Choi, Rui Ren, Sayan Shaw, Meng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14493)  

**Abstract**: Deploying high-quality automatic speech recognition (ASR) on edge devices requires models that jointly optimize accuracy, latency, and memory footprint while operating entirely on CPU without GPU acceleration. We conduct a systematic empirical study of state-of-the-art ASR architectures, encompassing encoder-decoder, transducer, and LLM-based paradigms, evaluated across batch, chunked, and streaming inference modes. Through a comprehensive benchmark of over 50 configurations spanning OpenAI Whisper, NVIDIA Nemotron, Parakeet TDT, Canary, Conformer Transducer, and Qwen3-ASR, we identify NVIDIA's Nemotron Speech Streaming as the strongest candidate for real-time English streaming on resource-constrained hardware. We then re-implement the complete streaming inference pipeline in ONNX Runtime and conduct a controlled evaluation of multiple post-training quantization strategies, including importance-weighted k-quant, mixed-precision schemes, and round-to-nearest quantization, combined with graph-level operator fusion. These optimizations reduce the model from 2.47 GB to as little as 0.67 GB while maintaining word error rate (WER) within 1% absolute of the full-precision PyTorch baseline. Our recommended configuration, the int4 k-quant variant, achieves 8.20% average streaming WER across eight standard benchmarks, running comfortably faster than real-time on CPU with 0.56 s algorithmic latency, establishing a new quality-efficiency Pareto point for on-device streaming ASR. 

---
# Seeing Through Circuits: Faithful Mechanistic Interpretability for Vision Transformers 

**Authors**: Nina Żukowska, Wolfgang Stammer, Bernt Schiele, Jonas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2604.14477)  

**Abstract**: Transparency of neural networks' internal reasoning is at the heart of interpretability research, adding to trust, safety, and understanding of these models. The field of mechanistic interpretability has recently focused on studying task-specific computational graphs, defined by connections (edges) between model components. Such edge-based circuits have been defined in the context of large language models, yet vision-based approaches so far only consider neuron-based circuits. These tell which information is encoded, but not how it is routed through the complex wiring of a neural network. In this work, we investigate whether useful mechanistic circuits can be identified through computational graphs in vision transformers. We propose an effective method for Automatic Visual Circuit Discovery (Vi-CD) that recovers class-specific circuits for classification, identifies circuits underlying typographic attacks in CLIP, and discovers circuits that lend themselves for steering to correct harmful model behavior. Overall, we find that insightful and actionable edge-based circuits can be recovered from vision transformers, adding transparency to the internal computations of these models. 

---
# Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve 

**Authors**: Weixiang Shen, Bailiang Jian, Jun Li, Che Liu, Johannes Moll, Xiaobin Hu, Daniel Rueckert, Hongwei Bran Li, Jiazhen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14475)  

**Abstract**: Tool-augmented large language model (LLM) agents can orchestrate specialist classifiers, segmentation models, and visual question-answering modules to interpret chest X-rays. However, these agents still solve each case in isolation: they fail to accumulate experience across cases, correct recurrent reasoning mistakes, or adapt their tool-use behavior without expensive reinforcement learning. While a radiologist naturally improves with every case, current agents remain static. In this work, we propose Evo-MedAgent, a self-evolving memory module that equips a medical agent with the capacity for inter-case learning at test time. Our memory comprises three complementary stores: (1)~\emph{Retrospective Clinical Episodes} that retrieve problem-solving experiences from similar past cases, (2)~an \emph{Adaptive Procedural Heuristics} bank curating priority-tagged diagnostic rules that evolves via reflection, much like a physician refining their internal criteria, and (3)~a \emph{Tool Reliability Controller} that tracks per-tool trustworthiness. On ChestAgentBench, Evo-MedAgent raises multiple-choice question (MCQ) accuracy from 0.68 to 0.79 on GPT-5-mini, and from 0.76 to 0.87 on Gemini-3 Flash. With a strong base model, evolving memory improves performance more effectively than orchestrating external tools on qualitative diagnostic tasks. Because Evo-MedAgent requires no training, its per-case overhead is bounded by one additional retrieval pass and a single reflection call, making it deployable on top of any frozen model. 

---
# Response-Aware User Memory Selection for LLM Personalization 

**Authors**: Jillian Fisher, Jennifer Neville, Chan Young Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.14473)  

**Abstract**: A common approach to personalization in large language models (LLMs) is to incorporate a subset of the user memory into the prompt at inference time to guide the model's generation. Existing methods select these subsets primarily using similarity between user memory items and input queries, ignoring how features actually affect the model's response distribution. We propose Response-Utility optimization for Memory Selection (RUMS), a novel method that selects user memory items by measuring the mutual information between a subset of memory and the model's outputs, identifying items that reduce response uncertainty and sharpen predictions beyond semantic similarity. We demonstrate that this information-theoretic foundation enables more principled user memory selection that aligns more closely with human selection compared to state-of-the-art methods, and models $400\times$ larger. Additionally, we show that memory items selected using RUMS result in better response quality compared to existing approaches, while having up to $95\%$ reduction in computational cost. 

---
# Improving Human Performance with Value-Aware Interventions: A Case Study in Chess 

**Authors**: Saumik Narayanan, Raja Panjwani, Siddhartha Sen, Chien-Ju Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.14465)  

**Abstract**: AI systems are increasingly used to assist humans in sequential decision-making tasks, yet determining when and how an AI assistant should intervene remains a fundamental challenge. A potential baseline is to recommend the optimal action according to a strong model. However, such actions assume optimal follow-up actions, which human decision makers may fail to execute, potentially reducing overall performance. In this work, we propose and study value-aware interventions, motivated by a basic principle in reinforcement learning: under the Bellman equation, the optimal policy selects actions that maximize the immediate reward plus the value function. When a decision maker follows a suboptimal policy, this policy-value consistency no longer holds, creating discrepancies between the actions taken by the policy and those that maximize the immediate reward plus the value of the next state. We show that these policy-value inconsistencies naturally identify opportunities for intervention. We formalize this problem in a Markov decision process where an AI assistant may override human actions under an intervention budget. In the single-intervention regime, we show that the optimal strategy is to recommend the action that maximizes the human value function. For settings with multiple interventions, we propose a tractable approximation that prioritizes interventions based on the magnitude of the policy-value discrepancy. We evaluate these ideas in the domain of chess by learning models of humans from large-scale gameplay data. In simulation, our approach consistently outperforms interventions based on the strongest chess engine (Stockfish) in a wide range of settings. A within-subject human study with 20 players and 600 games further shows that our interventions significantly improve performance for low- and mid-skill players while matching expert-engine interventions for high-skill players. 

---
# AIBuildAI: An AI Agent for Automatically Building AI Models 

**Authors**: Ruiyi Zhang, Peijia Qin, Qi Cao, Li Zhang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.14455)  

**Abstract**: AI models underpin modern intelligent systems, driving advances across science, medicine, finance, and technology. Yet developing high-performing AI models remains a labor-intensive process that requires expert practitioners to iteratively design architectures, engineer representations, implement training pipelines and refine approaches through empirical evaluation. Existing AutoML methods partially alleviate this burden but remain limited to narrow aspects such as hyperparameter optimization and model selection within predefined search spaces, leaving the full development lifecycle largely dependent on human expertise. To address this gap, we introduce AIBuildAI, an AI agent that automatically builds AI models from a task description and training data. AIBuildAI adopts a hierarchical agent architecture in which a manager agent coordinates three specialized sub-agents: a designer for modeling strategy, a coder for implementation and debugging, and a tuner for training and performance optimization. Each sub-agent is itself a large language model (LLM) based agent capable of multi-step reasoning and tool use, enabling end-to-end automation of the AI model development process that goes beyond the scope of existing AutoML approaches. We evaluate AIBuildAI on MLE-Bench, a benchmark of realistic Kaggle-style AI development tasks spanning visual, textual, time-series and tabular modalities. AIBuildAI ranks first on MLE-Bench with a medal rate of 63.1%, outperforming all existing baseline methods and matching the capability of highly experienced AI engineers. These results demonstrate that hierarchical agent systems can automate the full AI model development process from task specification to deployable model, suggesting a pathway toward broadly accessible AI development with minimal human intervention. 

---
# On Tackling Complex Tasks with Reward Machines and Signal Temporal Logics 

**Authors**: Ana María Gómez Ruiz, Thao Dang, Alexandre Donzé  

**Link**: [PDF](https://arxiv.org/pdf/2604.14440)  

**Abstract**: We propose a Reinforcement Learning (RL) based control design framework for handling complex tasks. The approach extends the concept of Reward Machines (RM) with Signal Temporal Logic (STL) formulas that can be used for event generation. The use of STL allows not only a more efficient representation of rewards for complex tasks but also guiding the training process to converge towards behaviors satisfying specified requirements. We also propose an implementation of the framework that leverages the STL online monitoring algorithms. We illustrate the framework with three case studies (minigrid, cart-pole and high-way environments) with non-trivial tasks. 

---
# Geometric Routing Enables Causal Expert Control in Mixture of Experts 

**Authors**: Ivan Ternovtsii, Yurii Bilak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14434)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models scale parameters while fixing active computation per token, but the specialization of individual experts remains opaque. In a companion paper we showed that routing topology is quality-neutral: five structurally different configurations converge to statistically equivalent language modeling quality. Here we show that expert identity is nonetheless causally meaningful: individual rank-1 experts are monosemantic by construction, and cosine-similarity routing in a low-dimensional metric space makes their specialization directly inspectable.
We present four lines of evidence. First, projecting expert output vectors through the unembedding matrix yields a Semantic Dictionary: 15% of experts are monosemantic specialists spanning 10 categories (temporal, geographic, cardinal, discourse, emotional, financial, military, scientific). Second, routing exhibits a frequency-to-syntax gradient: early layers separate tokens by word frequency, deeper layers by syntactic class (Zipf-confound controls, all $p < 0.001$). Third, causal interventions confirm these labels: steering toward a temporal expert's centroid increases P(temporal) by +321% (median across 44 prompts); suppressing a geographic expert drops P(geographic) by -23%; rewriting an expert's output vector halves target-category probability, and effects compose additively across layers. Fourth, the interventions are not unique to cosine routing: linear routers support comparable steering, but only cosine routing provides geometric transparency -- expert specialization is readable directly from the centroid matrix.
MoE expert-level specialization is a first-class interpretability primitive: architecturally monosemantic, causally validated, and controllable at inference with zero overhead. 

---
# Demonstration of Pneuma-Seeker: Agentic System for Reifying and Fulfilling Information Needs on Tabular Data 

**Authors**: Muhammad Imam Luthfi Balaka, Raul Castro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2604.14422)  

**Abstract**: Data analysts working with relational data often start with vague or underspecified questions and refine them iteratively as they explore the data. To support this iterative process, we demonstrate Pneuma-Seeker, a system that reifies a user's information need as explicit, inspectable relational specifications, enabling iterative refinement of the information need, targeted data discovery, and provenance-aware execution. Through two real-world procurement use cases, we show how Pneuma-Seeker leverages LLMs as transparent, interactive analytical collaborators rather than opaque answer engines. 

---
# Equifinality in Mixture of Experts: Routing Topology Does Not Determine Language Modeling Quality 

**Authors**: Ivan Ternovtsii, Yurii Bilak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14419)  

**Abstract**: Sparse Mixture-of-Experts (MoE) architectures employ increasingly sophisticated routing mechanisms -- learned routers, multi-hop trajectories, token-dependent gating. We ask: does routing topology actually determine language modeling quality? We build a geometric MoE (ST-MoE) using cosine-similarity routing against learned centroids in a low-dimensional space ($d_{space} = 64$), requiring 80% fewer routing parameters than standard linear routers. Through 62 controlled experiments on WikiText-103 at 76--84M parameters trained to convergence (50K steps, 1.64B tokens), we find that routing topology does not determine asymptotic perplexity (PPL): five cosine-routing variants are statistically equivalent within a 1-PPL margin (Two One-Sided Tests [TOST], $p < 0.05$ for all 10 pairwise comparisons; 15 runs across 3 seeds, observed range 33.93--34.72). The finding extends to hash, random-fixed, and top-1 routing (single-seed; graceful 1.1--2.2 PPL degradation) and replicates on OpenWebText (0.03 PPL gap, 6 runs, 3 seeds each). A standard linear router with 5.3$\times$ more routing parameters reaches PPL 32.76, but iso-parameter cosine routing closes 67% of this gap -- the true mechanism advantage is $\sim$1.2%. The mechanistic explanation is convergent redundancy: multi-hop updates are collinear ($\cos(\Delta h_0, \Delta h_1) = 0.805$), implementing magnitude amplification rather than compositional reasoning; a single learnable scalar replicates multi-hop performance. As a practical payoff, zero-shot relative-norm halting saves 25% of MoE FLOPs at +0.12% PPL. Expert-level specialization and causal controllability -- which coexist with topology-level equifinality -- are explored in a companion paper. 

---
# Credo: Declarative Control of LLM Pipelines via Beliefs and Policies 

**Authors**: Duo Lu, Andrew Crotty, Uğur Çetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2604.14401)  

**Abstract**: Agentic AI systems are becoming commonplace in domains that require long-lived, stateful decision-making in continuously evolving conditions. As such, correctness depends not only on the output of individual model calls, but also on how to best adapt when incorporating new evidence or revising prior conclusions. However, existing frameworks rely on imperative control loops, ephemeral memory, and prompt-embedded logic, making agent behavior opaque, brittle, and difficult to verify. This paper introduces Credo, which represents semantic state as beliefs and regulates behavior using declarative policies defined over these beliefs. This design supports adaptive, auditable, and composable execution through a database-backed semantic control plane. We showcase these concepts in a decision-control scenario, where beliefs and policies declaratively guide critical execution choices (e.g., model selection, retrieval, corrective re-execution), enabling dynamic behavior without requiring any changes to the underlying pipeline code. 

---
# Mistake gating leads to energy and memory efficient continual learning 

**Authors**: Aaron Pache, Mark CW van Rossum  

**Link**: [PDF](https://arxiv.org/pdf/2604.14336)  

**Abstract**: Synaptic plasticity is metabolically expensive, yet animals continuously update their internal models without exhausting energy reserves. However, when artificial neural networks are trained, the network parameters are typically updated on every sample that is presented, even if the sample was classified correctly. Inspired by the human negativity bias and error-related negativity, we propose 'memorized mistake-gated learning' -- a biologically plausible plasticity rule where synaptic updates are strictly gated by current and past classification errors. This reduces the number of updates the network needs to make by $50\%\sim80\%$. Mistake gating is particularly well suited in two cases: 1) For incremental learning where new knowledge is acquired on a background of pre-existing knowledge, 2) For online learning scenarios when data needs to be stored for later replay, as mistake-gating reduces storage buffer requirements. The algorithm can be implemented in a few lines of code, adds no hyper-parameters, and comes at negligible computational overhead. Learning on mistakes is an energy efficient and biologically relevant modification to commonly used learning rules that is well suited for continual learning. 

---
# Seeing Through Experts Eyes A Foundational Vision Language Model Trained on Radiologists Gaze and Reasoning 

**Authors**: Kinhei Lee, Peiyuan Jing, Zhenxuan Zhang, Yue Yang, Tao Wang, Dominic C Marshall, Yingying Fang, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14316)  

**Abstract**: Large scale vision language models have shown promise in automating chest Xray interpretation, yet their clinical utility remains limited by a gap between model outputs and radiologist reasoning. Most systems optimize for semantic information without emulating how experts visually examine medical images, often overlooking critical findings or diverging from established diagnostic workflows. Radiologists follow structured protocols (e.g., the ABCDEF approach) that ensure all clinically relevant regions are systematically examined, reducing missed findings and supporting reliable diagnostic reasoning. We introduce GazeX, a vision language model that leverages radiologists' eye tracking data as a behavioral prior to model expert diagnostic reasoning. By incorporating gaze trajectories and fixation patterns into pretraining, GazeX learns to follow the spatial and temporal structure of radiologist attention and integrates observations in a clinically meaningful sequence. Using a curated dataset of over 30,000 gaze key frames from five radiologists, we demonstrate that GazeX produces more accurate, interpretable, and expert consistent outputs across radiology report generation, disease grounding, and visual question answering, utilizing 231,835 radiographic studies, 780,014 question answer pairs, and 1,162 image sentence pairs with bounding boxes. Unlike autonomous reporting systems, GazeX produces verifiable evidence artifacts, including inspection trajectories and finding linked localized regions, enabling efficient human verification and safe human AI collaboration. Learning through expert eyes provides a practical route toward more trustworthy, explainable, and diagnostically robust AI systems for radiology and beyond. 

---
# GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification 

**Authors**: Wangjie Gan, Miao Pan, Linbo Xi, Wenqi Zhang, Jintao Chen, Jianwei Yin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14258)  

**Abstract**: Large language models are typically post-trained using supervised fine-tuning (SFT) and reinforcement learning (RL), yet effectively unifying efficient knowledge injection with robust generalization remains challenging. In this work, we provide a training-dynamics analysis showing that SFT can be interpreted as a special case of policy gradient optimization with an extremely sparse implicit reward and unstable inverse-probability weighting, which together lead to single-path dependency, entropy collapse, and gradient explosion. Motivated by this diagnosis, we propose Group Fine-Tuning (GFT), a unified post-training framework that addresses these intrinsic limitations through two mechanisms: Group Advantage Learning, which constructs diverse response groups and derives normalized contrastive supervision to alleviate reward sparsity, and Dynamic Coefficient Rectification, which adaptively bounds inverse-probability weights to stabilize optimization while preserving efficient knowledge injection. Experiments demonstrate that GFT consistently surpasses SFT-based methods and yields policies that integrate more smoothly with subsequent RL training. 

---
# Formalizing Kantian Ethics: Formula of the Universal Law Logic (FULL) 

**Authors**: Taylor Olson  

**Link**: [PDF](https://arxiv.org/pdf/2604.14254)  

**Abstract**: The field of machine ethics aims to build Artificial Moral Agents (AMAs) to better understand morality and make AI agents safer. To do so, many approaches encode human moral intuition as a set of axioms on actions e.g., do not harm, you must help others. However, this introduces (at least) two limitations for future AMAs. First, it does not consider the agent's purposes in performing the action. Second, it assumes that we humans can enumerate our moral intuition. This paper explores formalizing a moral procedure that alleviates these two limitations. We specifically consider Kantian ethics and present a multi-sorted quantified modal logic we call the Formula of the Universal Law Logic (FULL). The FULL formalizes Kant's first formulation of the categorical imperative, the Formula of the Universal Law (FUL), and concepts such as causality and agency. We demonstrate on three cases from Kantian ethics that the FULL can reason to evaluate agents' actions for certain purposes without built-in moral intuition, given that it has sufficient (non-normative) background knowledge. Therefore, the FULL is a contribution towards more robust and autonomous AMAs, and a more formal understanding of Kantian ethics. 

---
# Interpretable and Explainable Surrogate Modeling for Simulations: A State-of-the-Art Survey and Perspectives on Explainable AI for Decision-Making 

**Authors**: Pramudita Satria Palar, Paul Saves, Muhammad Daffa Robani, Nicolas Verstaevel, Moncef Garouani, Julien Aligon, Koji Shimoyama, Joseph Morlier, Benoit Gaudou  

**Link**: [PDF](https://arxiv.org/pdf/2604.14240)  

**Abstract**: The simulation of complex systems increasingly relies on sophisticated but fundamentally opaque computational black-box simulators. Surrogate models play a central role in reducing the computational cost of complex systems simulations across a wide range of scientific and engineering domains. Notwithstanding, they inevitably inherit and often exacerbate this black-box nature, obscuring how input variables drive physical responses. Conversely, Explainable Artificial Intelligence (XAI) offers powerful tools to unpack these models. Yet, XAI methods struggle with engineering-specific constraints, such as highly correlated inputs, dynamical systems, and rigorous reliability requirements. Consequently, surrogate modeling and XAI have largely evolved as distinct fields of research, despite their strong complementarity. To reconnect these approaches, this state-of-the-art survey provides a structured perspective that maps existing XAI techniques onto the various stages of surrogate modeling workflows for design and exploration. To ground this synthesis, we draw upon illustrative applications across both equation-based simulations and agent-based modeling. We survey a broad spectrum of techniques, highlighting their strengths for revealing interactions and supporting human comprehension. Finally, we identify pressing open challenges, including the explainability of dynamical systems and the handling of mixed-variable systems, and propose a research agenda to make explainability a core, embedded element of simulation-driven workflows from model construction through decision-making. By transforming opaque emulators into explainable tools, this agenda empowers practitioners to move beyond accelerating simulations to extracting actionable insights from complex system behaviors. 

---
# Fun-TSG: A Function-Driven Multivariate Time Series Generator with Variable-Level Anomaly Labeling 

**Authors**: Pierre Lotte, André Péninou, Olivier Teste  

**Link**: [PDF](https://arxiv.org/pdf/2604.14221)  

**Abstract**: Reliable evaluation of anomaly detection methods in multivariate time series remains an open challenge, largely due to the limitations of existing benchmark datasets. Current resources often lack fine-grained anomaly annotations, do not provide explicit intervariable and temporal dependencies, and offer little insight into the underlying generative mechanisms. These shortcomings hinder the development and rigorous comparison of detection models, especially those targeting interpretable and variable-specific outputs. To address this gap, we introduce Fun-TSG, a fully customizable time series generator designed to support high-quality evaluation of anomaly detection systems. Our tool enables both fully automated generation, based on randomly sampled dependency structures and anomaly types, and manual generation through user-defined equations and anomaly configurations. In both cases, it provides full transparency over the data generation process, including access to ground-truth anomaly labels at the variable and timestamp levels. Fun-TSG supports the creation of diverse, interpretable, and reproducible benchmarking scenarios, enabling fine-grained performance analysis for both classical and modern anomaly detection models. 

---
# Simulating Human Cognition: Heartbeat-Driven Autonomous Thinking Activity Scheduling for LLM-based AI systems 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.14178)  

**Abstract**: Large Language Model (LLM) agents have demonstrated remarkable capabilities in reasoning and tool use, yet they often suffer from rigid, reactive control flows that limit their adaptability and efficiency. Most existing frameworks rely on fixed pipelines or failure-triggered reflection, causing agents to act impulsively or correct errors only after they occur. In this paper, we introduce Heartbeat-Driven Autonomous Thinking Activity Scheduling, a mechanism that enables proactive, adaptive, and continuous self-regulation. Mirroring the natural rhythm of human cognition, our system employs a periodic ``heartbeat'' mechanism to orchestrate a dynamic repertoire of cognitive modules (e.g., Planner, Critic, Recaller, Dreamer). Unlike traditional approaches that rely on hard-coded symbolic rules or immediate reactive triggers, our scheduler learns to determine when to engage specific thinking activities -- such as recalling memories, summarizing experiences, or strategic planning -- based on temporal patterns and historical context. This functional approach allows cognitive modules to be dynamically added or removed without structural reengineering. Meanwhile, we propose a meta-learning strategy for continual policy adaptation, where the scheduler optimizes its cognitive strategy over time using historical interaction logs. Evaluation results demonstrate that our approach effectively learns to schedule cognitive activities based on historical data and can autonomously integrate new thinking modules. 

---
# NuHF Claw: A Risk Constrained Cognitive Agent Framework for Human Centered Procedure Support in Digital Nuclear Control Rooms 

**Authors**: Xingyu Xiao, Jiejuan Tong, Jun Sun, Zhe Sui, Peng Chen, Jingang Liang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14160)  

**Abstract**: The rapid digitization of nuclear power plant main control rooms has fundamentally reshaped operator interaction patterns, introducing complex soft-control behaviors and elevated cognitive risks that are not adequately addressed by existing human reliability analysis approaches. Although recent advances in large language models and autonomous agents offer new opportunities for intelligent decision support, their deployment in safety critical environments remains constrained by risks of hallucinated reasoning and weakened human authority. This study proposes NuHF Claw, a persistent cognitive-risk agent framework that enables risk governed human centered autonomy for digital nuclear operations. The core methodological innovation lies in the introduction of a risk constrained agent runtime, which tightly couples cognitive state inference with probabilistic safety assessment to regulate autonomous system behavior in real time. By integrating cognitively grounded workload and situational awareness estimation with dynamic human error probability prediction, the framework transforms conventional offline reliability analysis into a proactive intervention mechanism embedded directly within operational workflows. Experimental validation on a high-fidelity digital control room simulator demonstrates that NuHF Claw can anticipate interface induced cognitive degradation, dynamically constrain unsafe autonomous recommendations, and provide risk-aware navigational guidance while preserving human decision authority. The results highlight a fundamental shift from automation-driven operation toward cognition-aware autonomy, offering a principled pathway for the safe integration of intelligent agents into next-generation nuclear control environments. 

---
# MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation 

**Authors**: Yan Li, Zezi Zeng, Yifan Yang, Yuqing Yang, Ning Liao, Weiwei Guo, Lili Qiu, Mingxi Cheng, Qi Dai, Zhendong Wang, Zhengyuan Yang, Xue Yang, Ji Li, Lijuan Wang, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.15309)  

**Abstract**: The rapid progress of Artificial Intelligence Generated Content (AIGC) tools enables images, videos, and visualizations to be created on demand for webpage design, offering a flexible and increasingly adopted paradigm for modern UI/UX. However, directly integrating such tools into automated webpage generation often leads to style inconsistency and poor global coherence, as elements are generated in isolation. We propose MM-WebAgent, a hierarchical agentic framework for multimodal webpage generation that coordinates AIGC-based element generation through hierarchical planning and iterative self-reflection. MM-WebAgent jointly optimizes global layout, local multimodal content, and their integration, producing coherent and visually consistent webpages. We further introduce a benchmark for multimodal webpage generation and a multi-level evaluation protocol for systematic assessment. Experiments demonstrate that MM-WebAgent outperforms code-generation and agent-based baselines, especially on multimodal element generation and integration. Code & Data: this https URL. 

---
# AD4AD: Benchmarking Visual Anomaly Detection Models for Safer Autonomous Driving 

**Authors**: Fabrizio Genilotti, Arianna Stropeni, Gionata Grotto, Francesco Borsatti, Manuel Barusco, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2604.15291)  

**Abstract**: The reliability of a machine vision system for autonomous driving depends heavily on its training data distribution. When a vehicle encounters significantly different conditions, such as atypical obstacles, its perceptual capabilities can degrade substantially. Unlike many domains where errors carry limited consequences, failures in autonomous driving translate directly into physical risk for passengers, pedestrians, and other road users. To address this challenge, we explore Visual Anomaly Detection (VAD) as a solution. VAD enables the identification of anomalous objects not present during training, allowing the system to alert the driver when an unfamiliar situation is detected. Crucially, VAD models produce pixel-level anomaly maps that can guide driver attention to specific regions of concern without requiring any prior assumptions about the nature or form of the hazard. We benchmark eight state-of-the-art VAD methods on AnoVox, the largest synthetic dataset for anomaly detection in autonomous driving. In particular, we evaluate performance across four backbone architectures spanning from large networks to lightweight ones such as MobileNet and DeiT-Tiny. Our results demonstrate that VAD transfers effectively to road scenes. Notably, Tiny-Dinomaly achieves the best accuracy-efficiency trade-off for edge deployment, matching full-scale localization performance at a fraction of the memory cost. This study represents a concrete step toward safer, more responsible deployment of autonomous vehicles, ultimately improving protection for passengers, pedestrians, and all road users. 

---
# Why Do Vision Language Models Struggle To Recognize Human Emotions? 

**Authors**: Madhav Agarwal, Sotirios A. Tsaftaris, Laura Sevilla-Lara, Steven McDonagh  

**Link**: [PDF](https://arxiv.org/pdf/2604.15280)  

**Abstract**: Understanding emotions is a fundamental ability for intelligent systems to be able to interact with humans. Vision-language models (VLMs) have made tremendous progress in the last few years for many visual tasks, potentially offering a promising solution for understanding emotions. However, it is surprising that even the most sophisticated contemporary VLMs struggle to recognize human emotions or to outperform even specialized vision-only classifiers. In this paper we ask the question "Why do VLMs struggle to recognize human emotions?", and observe that the inherently continuous and dynamic task of facial expression recognition (DFER) exposes two critical VLM vulnerabilities. First, emotion datasets are naturally long-tailed, and the web-scale data used to pre-train VLMs exacerbates this head-class bias, causing them to systematically collapse rare, under-represented emotions into common categories. We propose alternative sampling strategies that prevent favoring common concepts. Second, temporal information is critical for understanding emotions. However, VLMs are unable to represent temporal information over dense frame sequences, as they are limited by context size and the number of tokens that can fit in memory, which poses a clear challenge for emotion recognition. We demonstrate that the sparse temporal sampling strategy used in VLMs is inherently misaligned with the fleeting nature of micro-expressions (0.25-0.5 seconds), which are often the most critical affective signal. As a diagnostic probe, we propose a multi-stage context enrichment strategy that utilizes the information from "in-between" frames by first converting them into natural language summaries. This enriched textual context is provided as input to the VLM alongside sparse keyframes, preventing attentional dilution from excessive visual data while preserving the emotional trajectory. 

---
# Prism: Symbolic Superoptimization of Tensor Programs 

**Authors**: Mengdi Wu, Xiaoyu Jiang, Oded Padon, Zhihao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.15272)  

**Abstract**: This paper presents Prism, the first symbolic superoptimizer for tensor programs. The key idea is sGraph, a symbolic, hierarchical representation that compactly encodes large classes of tensor programs by symbolically representing some execution parameters. Prism organizes optimization as a two-level search: it constructs symbolic graphs that represent families of programs, and then instantiates them into concrete implementations. This formulation enables structured pruning of provably suboptimal regions of the search space using symbolic reasoning over operator semantics, algebraic identities, and hardware constraints.
We develop techniques for efficient symbolic graph generation, equivalence verification via e-graph rewriting, and parameter instantiation through auto-tuning. Together, these components allow Prism to bridge the rigor of exhaustive search with the scalability required for modern ML workloads. Evaluation on five commonly used LLM workloads shows that Prism achieves up to $2.2\times$ speedup over best superoptimizers and $4.9\times$ over best compiler-based approaches, while reducing end-to-end optimization time by up to $3.4\times$. 

---
# SegWithU: Uncertainty as Perturbation Energy for Single-Forward-Pass Risk-Aware Medical Image Segmentation 

**Authors**: Tianhao Fu, Austin Wang, Charles Chen, Roby Aldave-Garza, Yucheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.15271)  

**Abstract**: Reliable uncertainty estimation is critical for medical image segmentation, where automated contours feed downstream quantification and clinical decision support. Many strong uncertainty methods require repeated inference, while efficient single-forward-pass alternatives often provide weaker failure ranking or rely on restrictive feature-space assumptions. We present $\textbf{SegWithU}$, a post-hoc framework that augments a frozen pretrained segmentation backbone with a lightweight uncertainty head. SegWithU taps intermediate backbone features and models uncertainty as perturbation energy in a compact probe space using rank-1 posterior probes. It produces two voxel-wise uncertainty maps: a calibration-oriented map for probability tempering and a ranking-oriented map for error detection and selective prediction. Across ACDC, BraTS2024, and LiTS, SegWithU is the strongest and most consistent single-forward-pass baseline, achieving AUROC/AURC of $0.9838/2.4885$, $0.9946/0.2660$, and $0.9925/0.8193$, respectively, while preserving segmentation quality. These results suggest that perturbation-based uncertainty modeling is an effective and practical route to reliability-aware medical segmentation.
Source code is available at this https URL. 

---
# CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas 

**Authors**: Emanuel Tewolde, Xiao Zhang, David Guzman Piedrahita, Vincent Conitzer, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15267)  

**Abstract**: It is increasingly important that LLM agents interact effectively and safely with other goal-pursuing agents, yet, recent works report the opposite trend: LLMs with stronger reasoning capabilities behave _less_ cooperatively in mixed-motive games such as the prisoner's dilemma and public goods settings. Indeed, our experiments show that recent models -- with or without reasoning enabled -- consistently defect in single-shot social dilemmas.
To tackle this safety concern, we present the first comparative study of game-theoretic mechanisms that are designed to enable cooperative outcomes between rational agents _in equilibrium_. Across four social dilemmas testing distinct components of robust cooperation, we evaluate the following mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players. Among our findings, we establish that contracting and mediation are most effective in achieving cooperative outcomes between capable LLM models, and that repetition-induced cooperation deteriorates drastically when co-players vary. Moreover, we demonstrate that these cooperation mechanisms become _more effective_ under evolutionary pressures to maximize individual payoffs. 

---
# Stability and Generalization in Looped Transformers 

**Authors**: Asher Labovich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15259)  

**Abstract**: Looped transformers promise test-time compute scaling by spending more iterations on harder problems, but it remains unclear which architectural choices let them extrapolate to harder problems at test time rather than memorize training-specific solutions. We introduce a fixed-point based framework for analyzing looped architectures along three axes of stability -- reachability, input-dependence, and geometry -- and use it to characterize when fixed-point iteration yields meaningful predictions. Theoretically, we prove that looped networks without recall have countable fixed points and cannot achieve strong input-dependence at any spectral regime, while recall combined with outer normalization reliably produces a regime in which fixed points are simultaneously reachable, locally smooth in the input, and supported by stable backpropagation. Empirically, we train single-layer looped transformers on chess, sudoku, and prefix-sums and find that downstream performance tracks the framework's predictions across tasks and architectural configurations. We additionally introduce internal recall, a novel recall placement variant, and show that it becomes competitive with -- and on sudoku, substantially better than -- standard recall placement once outer normalization is applied. 

---
# Agentic Microphysics: A Manifesto for Generative AI Safety 

**Authors**: Federico Pierucci, Matteo Prandi, Marcantonio Bracale Syrnikov, Marcello Galisai, Piercosma Bisconti  

**Link**: [PDF](https://arxiv.org/pdf/2604.15236)  

**Abstract**: This paper advances a methodological proposal for safety research in agentic AI. As systems acquire planning, memory, tool use, persistent identity, and sustained interaction, safety can no longer be analysed primarily at the level of the isolated model. Population-level risks arise from structured interaction among agents, through processes of communication, observation, and mutual influence that shape collective behaviour over time. As the object of analysis shifts, a methodological gap emerges. Approaches focused either on single agents or on aggregate outcomes do not identify the interaction-level mechanisms that generate collective risks or the design variables that control them. A framework is required that links local interaction structure to population-level dynamics in a causally explicit way, allowing both explanation and intervention. We introduce two linked concepts. Agentic microphysics defines the level of analysis: local interaction dynamics where one agent's output becomes another's input under specific protocol conditions. Generative safety defines the methodology: growing phenomena and elicit risks from micro-level conditions to identify sufficient mechanisms, detect thresholds, and design effective interventions. 

---
# AI-Assisted Requirements Engineering: An Empirical Evaluation Relative to Expert Judgment 

**Authors**: Oz Levy, Ilya Dikman, Natan Levy, Michael Winokur  

**Link**: [PDF](https://arxiv.org/pdf/2604.15222)  

**Abstract**: Artificial Intelligence is increasingly introduced into systems engineering activities, particularly within requirements engineering, where quality assessment and validation remain heavily dependent on expert judgment. While recent AI tools demonstrate promising capabilities in analyzing and generating requirements, their role within formal systems engineering processes-and their alignment with established INCOSE criteria-remains insufficiently understood. This paper investigates the extent to which AI-based tools can support systems engineers in evaluating requirement quality, without replacing professional expertise. The research adopts a structured systems engineering methodology to compare AI-assisted requirement evaluation with human expert assessment. A controlled study was conducted in which system requirements were evaluated against established INCOSE ``good requirement'' criteria by both experienced systems engineers and an AI-based assessment tool. The evaluation focused on consistency, completeness, clarity, and testability, examining not only accuracy but also the decision logic underlying each assessment. Results indicate that AI tools can provide consistent and rapid preliminary assessments, particularly for syntactic and structural quality attributes. However, expert judgment remains essential for contextual interpretation, ambiguity resolution, and trade-off reasoning. Rather than positioning AI as a replacement for systems engineers, the findings support its role as a decision-support mechanism within the RE lifecycle. From a systems engineering perspective, this study contributes empirical evidence on how AI can be integrated into RE workflows while preserving traceability, accountability, and engineering consistency. 

---
# Benchmarking Classical Coverage Path Planning Heuristics on Irregular Hexagonal Grids for Maritime Coverage Scenarios 

**Authors**: Carlos S. Sepúlveda, Gonzalo A. Ruz  

**Link**: [PDF](https://arxiv.org/pdf/2604.15202)  

**Abstract**: Coverage path planning on irregular hexagonal grids is relevant to maritime surveillance, search and rescue and environmental monitoring, yet classical methods are often compared on small ad hoc examples or on rectangular grids. This paper presents a reproducible benchmark of deterministic single-vehicle coverage path planning heuristics on irregular hexagonal graphs derived from synthetic but maritime-motivated areas of interest. The benchmark contains 10,000 Hamiltonian-feasible instances spanning compact, elongated, and irregular morphologies, 17 heuristics from seven families, and a common evaluation protocol covering Hamiltonian success, complete-coverage success, revisits, path length, heading changes, and CPU latency. Across the released dataset, heuristics with explicit shortest-path reconnection solve the relaxed coverage task reliably but almost never produce zero-revisit tours. Exact Depth-First Search confirms that every released instance is Hamiltonian-feasible. The strongest classical Hamiltonian baseline is a Warnsdorff variant that uses an index-based tie-break together with a terminal-inclusive residual-degree policy, reaching 79.0% Hamiltonian success. The dominant design choice is not tie-breaking alone, but how the residual degree is defined when the endpoint is reserved until the final move. This shows that underreported implementation details can materially affect performance on sparse geometric graphs with bottlenecks. The benchmark is intended as a controlled testbed for heuristic analysis rather than as a claim of operational optimality at fleet scale. 

---
# VisPCO: Visual Token Pruning Configuration Optimization via Budget-Aware Pareto-Frontier Learning for Vision-Language Models 

**Authors**: Huawei Ji, Yuanhao Sun, Yuan Jin, Cheng Deng, Jiaxin Ding, Luoyi Fu, Xinbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15188)  

**Abstract**: Visual token pruning methods effectively mitigate the quadratic computational growth caused by processing high-resolution images and video frames in vision-language models (VLMs). However, existing approaches rely on predefined pruning configurations without determining whether they achieve computation-performance optimality. In this work, we introduce , a novel framework that formulates visual token pruning as a Pareto configuration optimization problem to automatically identify optimal configurations. Our approach employs continuous relaxation and straight-through estimators to enable gradient-based search, solved via the Augmented Lagrangian method. Extensive experiments across 8 visual benchmarks demonstrate that effectively approximates the empirical Pareto frontier obtained through grid search and generalizes well across various pruning methods and VLM architectures. Furthermore, through learnable kernel functions, we investigate layer-wise pruning patterns and reveal that multi-step progressive pruning captures VLMs' hierarchical compression structure, achieving superior accuracy-efficiency trade-offs compared to single-layer approaches. 

---
# Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines 

**Authors**: Marcel Wagenländer, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Llúis Vilanova, Peter Pietzuch  

**Link**: [PDF](https://arxiv.org/pdf/2604.15186)  

**Abstract**: Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription.
We describe Scepsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. Scepsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. Scepsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, Scepsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that Scepsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations. 

---
# MambaSL: Exploring Single-Layer Mamba for Time Series Classification 

**Authors**: Yoo-Min Jung, Leekyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.15174)  

**Abstract**: Despite recent advances in state space models (SSMs) such as Mamba across various sequence domains, research on their standalone capacity for time series classification (TSC) has remained limited. We propose MambaSL, a framework that minimally redesigns the selective SSM and projection layers of a single-layer Mamba, guided by four TSC-specific hypotheses. To address benchmarking limitations -- restricted configurations, partial University of East Anglia (UEA) dataset coverage, and insufficiently reproducible setups -- we re-evaluate 20 strong baselines across all 30 UEA datasets under a unified protocol. As a result, MambaSL achieves state-of-the-art performance with statistically significant average improvements, while ensuring reproducibility via public checkpoints for all evaluated models. Together with visualizations, these results demonstrate the potential of Mamba-based architectures as a TSC backbone. 

---
# Class Unlearning via Depth-Aware Removal of Forget-Specific Directions 

**Authors**: Arman Hatami, Romina Aalishah, Ilya E. Monosov  

**Link**: [PDF](https://arxiv.org/pdf/2604.15166)  

**Abstract**: Machine unlearning aims to remove targeted knowledge from a trained model without the cost of retraining from scratch. In class unlearning, however, reducing accuracy on forget classes does not necessarily imply true forgetting: forgotten information can remain encoded in internal representations, and apparent forgetting may arise from classifier-head suppression rather than representational removal. We show that existing class-unlearning methods often exhibit weak or negative selectivity, preserve forget-class structure in deep representations, or rely heavily on final-layer bias shifts. We then introduce DAMP (Depth-Aware Modulation by Projection), a one-shot, closed-form weight-surgery method that removes forget-specific directions from a pretrained network without gradient-based optimization. At each stage, DAMP computes class prototypes in the input space of the next learnable operator, extracts forget directions as residuals relative to retain-class prototypes, and applies a projection-based update to reduce downstream sensitivity to those directions. To preserve utility, DAMP uses a parameter-free depth-aware scaling rule derived from probe separability, applying smaller edits in early layers and larger edits in deeper layers. The method naturally extends to multi-class forgetting through low-rank subspace removal. Across MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet, and across convolutional and transformer architectures, DAMP more closely resembles the retraining gold standard than some of the prior methods, improving selective forgetting while better preserving retain-class performance and reducing residual forget-class structure in deep layers. 

---
# Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models 

**Authors**: Zihao Xu, John Harvill, Ziwei Fan, Yizhou Sun, Hao Ding, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15153)  

**Abstract**: Large Language Models (LLMs) incur significant computational and memory costs when processing long prompts, as full self-attention scales quadratically with input length. Token compression aims to address this challenge by reducing the number of tokens representing inputs. However, existing prompt-compression approaches primarily operate in token space and overlook inefficiencies in the latent embedding space. In this paper, we propose K-Token Merging, a latent-space compression framework that merges each contiguous block of K token embeddings into a single embedding via a lightweight encoder. The compressed sequence is processed by a LoRA-adapted LLM, while generation remains in the original vocabulary. Experiments on structural reasoning (Textualized Tree), sentiment classification (Amazon Reviews), and code editing (CommitPackFT) show that K-Token Merging lies on the Pareto frontier of performance vs. compression, achieving up to 75% input length reduction with minimal performance degradation. 

---
# LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking 

**Authors**: Lukas Helff, Quentin Delfosse, David Steinmann, Ruben Härle, Hikaru Shindo, Patrick Schramowski, Wolfgang Stammer, Kristian Kersting, Felix Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15149)  

**Abstract**: As reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for scaling reasoning capabilities in LLMs, a new failure mode emerges: LLMs gaming verifiers. We study this phenomenon on inductive reasoning tasks, where models must induce and output logical rules. We find that RLVR-trained models systematically abandon rule induction. Instead of learning generalizable patterns (e.g., ``trains carrying red cars go east''), they enumerate instance-level labels, producing outputs that pass verifiers without capturing the relational patterns required by the task. We show that this behavior is not a failure of understanding but a form of reward hacking: imperfect verifiers that check only extensional correctness admit false positives. To detect such shortcuts, we introduce Isomorphic Perturbation Testing (IPT), which evaluates a single model output under both extensional and isomorphic verification, where the latter enforces invariance under logically isomorphic tasks. While genuine rule induction remains invariant, shortcut strategies fail. We find that shortcut behavior is specific to RLVR-trained reasoning models (e.g., GPT-5, Olmo3) and absent in non-RLVR models (e.g., GPT-4o, GPT-4.5, Ministral). Moreover, shortcut prevalence increases with task complexity and inference-time compute. In controlled training experiments, extensional verification directly induces shortcut strategies, while isomorphic verification eliminates them. These results show that RLVR can incentivize reward hacking not only through overt manipulation but also by exploiting what the verifier fails to enforce. 

---
# Structure as Computation: Developmental Generation of Minimal Neural Circuits 

**Authors**: Duan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15143)  

**Abstract**: This work simulates the developmental process of cortical neurogenesis, initiating from a single stem cell and governed by gene regulatory rules derived from mouse single-cell transcriptomic data. The developmental process spontaneously generates a heterogeneous population of 5,000 cells, yet yields only 85 mature neurons - merely 1.7% of the total population. These 85 neurons form a densely interconnected core of 200,400 synapses, corresponding to an average degree of 4,715 per neuron. At iteration zero, this minimal circuit performs at chance level on MNIST. However, after a single epoch of standard training, accuracy surges to over 90% - a gain exceeding 80 percentage points - with typical runs falling in the 89-94% range depending on developmental stochasticity. The identical circuit, without any architectural modification or data augmentation, achieves 40.53% on CIFAR-10 after one epoch. These findings demonstrate that developmental rules sculpt a domain-general topological substrate exceptionally amenable to rapid learning, suggesting that biological developmental processes inherently encode powerful structural priors for efficient computation. 

---
# Amortized Optimal Transport from Sliced Potentials 

**Authors**: Minh-Phuc Truong, Khai Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.15114)  

**Abstract**: We propose a novel amortized optimization method for predicting optimal transport (OT) plans across multiple pairs of measures by leveraging Kantorovich potentials derived from sliced OT. We introduce two amortization strategies: regression-based amortization (RA-OT) and objective-based amortization (OA-OT). In RA-OT, we formulate a functional regression model that treats Kantorovich potentials from the original OT problem as responses and those obtained from sliced OT as predictors, and estimate these models via least-squares methods. In OA-OT, we estimate the parameters of the functional model by optimizing the Kantorovich dual objective. In both approaches, the predicted OT plan is subsequently recovered from the estimated potentials. As amortized OT methods, both RA-OT and OA-OT enable efficient solutions to repeated OT problems across different measure pairs by reusing information learned from prior instances to rapidly approximate new solutions. Moreover, by exploiting the structure provided by sliced OT, the proposed models are more parsimonious, independent of specific structures of the measures, such as the number of atoms in the discrete case, while achieving high accuracy. We demonstrate the effectiveness of our approaches on tasks including MNIST digit transport, color transfer, supply-demand transportation on spherical data, and mini-batch OT conditional flow matching. 

---
# IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation 

**Authors**: Haozhi Fan, Jinhao Duan, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15109)  

**Abstract**: Despite the rapid advancement of Large Language Models (LLMs), uncertainty quantification in LLM generation is a persistent challenge. Although recent approaches have achieved strong performance by restricting LLMs to produce short or constrained answer sets, many real-world applications require long-form and free-form text generation. A key difficulty in this setting is that LLMs often produce responses that are semantically coherent yet factually inaccurate, while the underlying semantics are multifaceted and the linguistic structure is complex. To tackle this challenge, this paper introduces Interrogative Uncertainty Quantification (IUQ), a novel framework that leverages inter-sample consistency and intra-sample faithfulness to quantify the uncertainty in long-form LLM outputs. By utilizing an interrogate-then-respond paradigm, our method provides reliable measures of claim-level uncertainty and the model's faithfulness. Experimental results across diverse model families and model sizes demonstrate the superior performance of IUQ over two widely used long-form generation datasets. The code is available at this https URL. 

---
# Autonomous Evolution of EDA Tools: Multi-Agent Self-Evolved ABC 

**Authors**: Cunxi Yu, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.15082)  

**Abstract**: This paper introduces the first \emph{self-evolving} logic synthesis framework, which leverages Large Language Model (LLM) agents to autonomously improve the source code of \textsc{ABC}, the widely adopted logic synthesis system. Our framework operates on the \emph{entire integrated ABC codebase}, and the output repository preserves its single-binary execution model and command interface. In the initial evolution cycle, we bootstrap the system using existing prior open-source synthesis components, covering flow tuning, logic minimization, and technology mapping, but without manually injecting new heuristics. On top of this foundation, a team of LLM-based agents iteratively rewrites and evolves specific sub-components of ABC following our ``programming guidance`` prompts under a unified correctness and QoR-driven evaluation loop. Each evolution cycle proposes code modifications, compiles the integrated binary, validates correctness, and evaluates quality-of-results (QoR) on \emph{multi-suite benchmarks including ISCAS~85/89/99, VTR, EPFL, and IWLS~2005}. Through continuous feedback, the system discovers optimizations beyond human-designed heuristics, effectively \emph{learning new synthesis strategies} that enhance QoR. We detail the architecture of this self-improving system, its integration with \textsc{ABC}, and results demonstrating that the framework can autonomously and progressively improve EDA tool at full million-line scale. 

---
# NEAT-NC: NEAT guided Navigation Cells for Robot Path Planning 

**Authors**: Hibatallah Meliani, Khadija Slimani, Samira Khoulji  

**Link**: [PDF](https://arxiv.org/pdf/2604.15076)  

**Abstract**: To navigate a space, the brain makes an internal representation of the environment using different cells such as place cells, grid cells, head direction cells, border cells, and speed cells. All these cells, along with sensory inputs, enable an organism to explore the space around it. Inspired by these biological principles, we developed NEATNC, a Neuro-Evolution of Augmenting Topology guided Navigation Cells. The goal of the paper is to improve NEAT algorithm performance in path planning in dynamic environments using spatial cognitive cells. This approach uses navigation cells as inputs and evolves recurrent neural networks, representing the hippocampus part of the brain. The performance of the proposed algorithm is evaluated in different static and dynamic scenarios. This study highlights NEAT's adaptability to complex and different environments, showcasing the utility of biological theories. This suggests that our approach is well-suited for real-time dynamic path planning for robotics and games. 

---
# No More Guessing: a Verifiable Gradient Inversion Attack in Federated Learning 

**Authors**: Francesco Diana, Chuan Xu, André Nusser, Giovanni Neglia  

**Link**: [PDF](https://arxiv.org/pdf/2604.15063)  

**Abstract**: Gradient inversion attacks threaten client privacy in federated learning by reconstructing training samples from clients' shared gradients. Gradients aggregate contributions from multiple records and existing attacks may fail to disentangle them, yielding incorrect reconstructions with no intrinsic way to certify success. In vision and language, attackers may fall back on human inspection to judge reconstruction plausibility, but this is far less feasible for numerical tabular records, fueling the impression that tabular data is less vulnerable.
We challenge this perception by proposing a verifiable gradient inversion attack (VGIA) that provides an explicit certificate of correctness for reconstructed samples. Our method adopts a geometric view of ReLU leakage: the activation boundary of a fully connected layer defines a hyperplane in input space. VGIA introduces an algebraic, subspace-based verification test that detects when a hyperplane-delimited region contains exactly one record. Once isolation is certified, VGIA recovers the corresponding feature vector analytically and reconstructs the target via a lightweight optimization step.
Experiments on tabular benchmarks with large batch sizes demonstrate exact record and target recovery in regimes where existing state-of-the-art attacks either fail or cannot assess reconstruction fidelity. Compared to prior geometric approaches, VGIA allocates hyperplane queries more effectively, yielding faster reconstructions with fewer attack rounds. 

---
# CoGrid & the Multi-User Gymnasium: A Framework for Multi-Agent Experimentation 

**Authors**: Chase McDonald, Cleotilde Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2604.15044)  

**Abstract**: The increasing integration of artificial intelligence (AI) in everyday life brings with it new challenges and questions for regarding how humans interact with autonomous agents. Multi-agent experiments, where humans and AI act together, can offer important opportunities to study social decision making, but there is a lack of accessible tooling available to researchers to run such experiments. We introduce two tools designed to reduce these barriers. The first, CoGrid, is a multi-agent grid-based simulation library with dual NumPy and JAX backends. The second, Multi-User Gymnasium (MUG), translates such simulation environments directly into interactive web-based experiments. MUG supports interactions with arbitrary numbers of humans and AI, utilizing either server-authoritative or peer-to-peer networking with rollback netcode to account for latency. Together, these tools can enable researchers to deploy studies of human-AI interaction, facilitating inquiry into core questions of psychology, cognition, and decision making and their relationship to human-AI interaction. Both tools are open source and available to the broader research community. Documentation and source code is available at {cogrid, multi-user-gymnasium}.this http URL. This paper details the functionality of these tools and presents several case studies to illustrate their utility in human-AI multi-agent experimentation. 

---
# When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning 

**Authors**: Khalid Adnan Alsayed  

**Link**: [PDF](https://arxiv.org/pdf/2604.15038)  

**Abstract**: The evaluation of fairness in machine learning systems has become a central concern in high-stakes applications, including biometric recognition, healthcare decision-making, and automated risk assessment. Existing approaches typically rely on a small number of fairness metrics to assess model behaviour across group partitions, implicitly assuming that these metrics provide consistent and reliable conclusions. However, different fairness metrics capture distinct statistical properties of model performance and may therefore produce conflicting assessments when applied to the same system. In this work, we investigate the consistency of fairness evaluation by conducting a systematic multi-metric analysis of demographic bias in machine learning models. Using face recognition as a controlled experimental setting, we evaluate model performance across multiple group partitions under a range of commonly used fairness metrics, including error-rate disparities and performance-based measures. Our results demonstrate that fairness assessments can vary significantly depending on the choice of metrics, leading to contradictory conclusions regarding model bias. To quantify this phenomenon, we introduce the Fairness Disagreement Index (FDI), a measure designed to capture the degree of inconsistency across fairness metrics. We further show that disagreement remains high across thresholds and model configurations. These findings highlight a critical limitation in current fairness evaluation practices and suggest that single-metric reporting is insufficient for reliable bias assessment. 

---
# Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization 

**Authors**: Haochun Tang, Yuliang Yan, Jiahua Lu, Huaxiao Liu, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15022)  

**Abstract**: Cost-aware routing dynamically dispatches user queries to models of varying capability to balance performance and inference cost. However, the routing strategy introduces a new security concern that adversaries may manipulate the router to consistently select expensive high-capability models. Existing routing attacks depend on either white-box access or heuristic prompts, rendering them ineffective in real-world black-box scenarios. In this work, we propose R$^2$A, which aims to mislead black-box LLM routers to expensive models via adversarial suffix optimization. Specifically, R$^2$A deploys a hybrid ensemble surrogate router to mimic the black-box router. A suffix optimization algorithm is further adapted for the ensemble-based surrogate. Extensive experiments on multiple open-source and commercial routing systems demonstrate that {R$^2$A} significantly increases the routing rate to expensive models on queries of different distributions. Code and examples: this https URL. 

---
# What Is the Minimum Architecture for Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers 

**Authors**: Éric Jacopin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15010)  

**Abstract**: When do transformers commit to a decision, and what prevents them from correcting it? We introduce \textbf{prolepsis}: a transformer commits early, task-specific attention heads sustain the commitment, and no layer corrects it. Replicating \citeauthor{lindsey2025biology}'s (\citeyear{lindsey2025biology}) planning-site finding on open models (Gemma~2 2B, Llama~3.2 1B), we ask five questions. (Q1)~Planning is invisible to six residual-stream methods; CLTs are necessary. (Q2)~The planning-site spike replicates with identical geometry. (Q3)~Specific attention heads route the decision to the output, filling a gap flagged as invisible to attribution graphs. (Q4)~Search requires ${\leq}16$ layers; commitment requires more. (Q5)~Factual recall shows the same motif at a different network depth, with zero overlap between recurring planning heads and the factual top-10. Prolepsis is architectural: the template is shared, the routing substrates differ. All experiments run on a single consumer GPU (16\,GB VRAM). 

---
# Agentic Explainability at Scale: Between Corporate Fears and XAI Needs 

**Authors**: Yomna Elsayed, Cecily Jones  

**Link**: [PDF](https://arxiv.org/pdf/2604.14984)  

**Abstract**: As companies enter the race for agentic AI adoption, fears surface around agentic autonomy and its subsequent risks. These fears compound as companies scale their agentic AI adoption with low-code applications, without a comparable scaling in their governance processes and expertise resulting in a phenomenon known as "Agent Sprawl". While shadow AI tools can help with agentic discovery and identification, few observability tools offer insights into the agents' configuration and settings or the decision-making process during agent-to-agent communication and orchestration. This paper explores AI governance professionals' concerns in enterprise settings, while offering design-time and runtime explainability techniques as suggested by AI governance experts for addressing those fears. Finally, we provide a preliminary prototype of an Agentic AI Card that can help companies feel at ease deploying agents at scale. 

---
# UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards 

**Authors**: Jun Wang, Shuo Tan, Zelong Sun, Tiancheng Gu, Yongle Zhao, Ziyong Feng, Kaicheng Yang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14967)  

**Abstract**: Retrieval-Augmented Generation (RAG) extends Large Vision-Language Models (LVLMs) with external visual knowledge. However, existing visual RAG systems typically rely on generic retrieval signals that overlook the fine-grained visual semantics essential for complex reasoning. To address this limitation, we propose UniDoc-RL, a unified reinforcement learning framework in which an LVLM agent jointly performs retrieval, reranking, active visual perception, and reasoning. UniDoc-RL formulates visual information acquisition as a sequential decision-making problem with a hierarchical action space. Specifically, it progressively refines visual evidence from coarse-grained document retrieval to fine-grained image selection and active region cropping, allowing the model to suppress irrelevant content and attend to information-dense regions. For effective end-to-end training, we introduce a dense multi-reward scheme that provides task-aware supervision for each action. Based on Group Relative Policy Optimization (GRPO), UniDoc-RL aligns agent behavior with multiple objectives without relying on a separate value network. To support this training paradigm, we curate a comprehensive dataset of high-quality reasoning trajectories with fine-grained action annotations. Experiments on three benchmarks demonstrate that UniDoc-RL consistently surpasses state-of-the-art baselines, yielding up to 17.7% gains over prior RL-based methods. 

---
# Calibration-Gated LLM Pseudo-Observations for Online Contextual Bandits 

**Authors**: Maksim Pershin, Ivan Golovanov, Pavel Baltabaev, Natalia Trankova  

**Link**: [PDF](https://arxiv.org/pdf/2604.14961)  

**Abstract**: Contextual bandit algorithms suffer from high regret during cold-start, when the learner has insufficient data to distinguish good arms from bad. We propose augmenting Disjoint LinUCB with LLM pseudo-observations: after each round, a large language model predicts counterfactual rewards for the unplayed arms, and these predictions are injected into the learner as weighted pseudo-observations. The injection weight is controlled by a calibration-gated decay schedule that tracks the LLM's prediction accuracy on played arms via an exponential moving average; high calibration error suppresses the LLM's influence, while accurate predictions receive higher weight during the critical early rounds. We evaluate on two contextual bandit environments - UCI Mushroom (2-arm, asymmetric rewards) and MIND-small (5-arm news recommendation) - and find that when equipped with a task-specific prompt, LLM pseudo-observations reduce cumulative regret by 19% on MIND relative to pure LinUCB. However, generic counterfactual prompt framing increases regret on both environments, demonstrating that prompt design is the dominant factor, more important than the choice of decay schedule or calibration gating parameters. We analyze the failure modes of calibration gating on domains with small prediction errors and provide a theoretical motivation for the bias-variance trade-off governing pseudo-observation weight. 

---
# RaTA-Tool: Retrieval-based Tool Selection with Multimodal Large Language Models 

**Authors**: Gabriele Mattioli, Evelyn Turri, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2604.14951)  

**Abstract**: Tool learning with foundation models aims to endow AI systems with the ability to invoke external resources -- such as APIs, computational utilities, and specialized models -- to solve complex tasks beyond the reach of standalone language generation. While recent advances in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have expanded their reasoning and perception capabilities, existing tool-use methods are predominantly limited to text-only inputs and closed-world settings. Consequently, they struggle to interpret multimodal user instructions and cannot generalize to tools unseen during training. In this work, we introduce RaTA-Tool, a novel framework for open-world multimodal tool selection. Rather than learning direct mappings from user queries to fixed tool identifiers, our approach enables an MLLM to convert a multimodal query into a structured task description and subsequently retrieve the most appropriate tool by matching this representation against semantically rich, machine-readable tool descriptions. This retrieval-based formulation naturally supports extensibility to new tools without retraining. To further improve alignment between task descriptions and tool selection, we incorporate a preference-based optimization stage using Direct Preference Optimization (DPO). To support research in this setting, we also introduce the first dataset for open-world multimodal tool use, featuring standardized tool descriptions derived from Hugging Face model cards. Extensive experiments demonstrate that our approach significantly improves tool-selection performance, particularly in open-world, multimodal scenarios. 

---
# STEP-Parts: Geometric Partitioning of Boundary Representations for Large-Scale CAD Processing 

**Authors**: Shen Fan, Mikołaj Kida, Przemyslaw Musialski  

**Link**: [PDF](https://arxiv.org/pdf/2604.14927)  

**Abstract**: Many CAD learning pipelines discretize Boundary Representations (B-Reps) into triangle meshes, discarding analytic surface structure and topological adjacency and thereby weakening consistent instance-level analysis. We present STEP-Parts, a deterministic CAD-to-supervision toolchain that extracts geometric instance partitions directly from raw STEP B-Reps and transfers them to tessellated carriers through retained source-face correspondence, yielding instance labels and metadata for downstream learning and evaluation. The construction merges adjacent B-Rep faces only when they share the same analytic primitive type and satisfy a near-tangent continuity criterion. On ABC, same-primitive dihedral angles are strongly bimodal, yielding a threshold-insensitive low-angle regime for part extraction. Because the partition is defined on intrinsic B-Rep topology rather than on a particular triangulation, the resulting boundaries remain stable under changes in tessellation. Applied to the DeepCAD subset of ABC, the pipeline processes approximately 180{,}000 models in under six hours on a consumer CPU. We release code and precomputed labels, and show that STEP-Parts serves both as a tessellation-robust geometric reference and as a useful supervision source in two downstream probes: an implicit reconstruction--segmentation network and a dataset-level point-based backbone. 

---
# Improving Sparse Autoencoder with Dynamic Attention 

**Authors**: Dongsheng Wang, Jinsen Zhang, Dawei Su, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14925)  

**Abstract**: Recently, sparse autoencoders (SAEs) have emerged as a promising technique for interpreting activations in foundation models by disentangling features into a sparse set of concepts. However, identifying the optimal level of sparsity for each neuron remains challenging in practice: excessive sparsity can lead to poor reconstruction, whereas insufficient sparsity may harm interpretability. While existing activation functions such as ReLU and TopK provide certain sparsity guarantees, they typically require additional sparsity regularization or cherry-picked hyperparameters. We show in this paper that dynamically sparse attention mechanisms using sparsemax can bridge this trade-off, due to their ability to determine the activation numbers in a data-dependent manner. Specifically, we first explore a new class of SAEs based on the cross-attention architecture with the latent features as queries and the learnable dictionary as the key and value matrices. To encourage sparse pattern learning, we employ a sparsemax-based attention strategy that automatically infers a sparse set of elements according to the complexity of each neuron, resulting in a more flexible and general activation function. Through comprehensive evaluation and visualization, we show that our approach successfully achieves lower reconstruction loss while producing high-quality concepts, particularly in top-n classification tasks. 

---
# Beyond Importance Sampling: Rejection-Gated Policy Optimization 

**Authors**: Ziwu Sun, Zhen Gao, Jiyong Zhang, Jiaheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14895)  

**Abstract**: We propose a new perspective on policy optimization: rather than reweighting all samples by their importance ratios, an optimizer should select which samples are trustworthy enough to drive a policy update. Building on this view, we introduce Rejection-Gated Policy Optimization (RGPO), which replaces the importance sampling ratio r_theta = pi_theta / pi_old with a smooth, differentiable acceptance gate alpha_theta(s, a) = g(r_theta(s, a)) in the range [0, 1]. Unlike prior work that applies rejection sampling as a data-level heuristic before training, RGPO elevates rejection to an optimization principle: the gate participates directly in gradient computation and is implicitly updated alongside the policy. RGPO provides a unified framework: the policy gradients of TRPO, PPO, and REINFORCE all correspond to specific choices of the effective gradient weight w(r) = g'(r) * r. We prove that RGPO guarantees finite, bounded gradient variance even when importance sampling ratios are heavy-tailed (where IS variance diverges). We further show that RGPO incurs only a bounded, controllable bias and provides an approximate monotonic policy improvement guarantee analogous to TRPO. RGPO matches PPO in computational cost, requires no second-order optimization, and extends naturally to RLHF-style preference alignment. In online preference fine-tuning of Qwen2.5-1.5B-Instruct on Anthropic HH-RLHF (n = 3 seeds), RGPO uses a dual-ratio gate that anchors learning to both the previous policy and the reference model, achieving a Pareto-dominant outcome: the highest reward among online RL methods (+14.8% vs. PPO-RLHF) and the lowest KL divergence to the reference model (-16.0% vs. PPO-RLHF, -53.1% vs. GRPO). 

---
# Can LLMs Score Medical Diagnoses and Clinical Reasoning as well as Expert Panels? 

**Authors**: Amy Rouillard, Sitwala Mundiab, Linda Camarab, Michael Cameron Gramaniec, Ziyaad Dangorc, Ismail Kallad, Shabir A. Madhic, Kajal Morarc, Marlvin T. Ncubec, Haroon Saloojeee, Bruce A. Bassett  

**Link**: [PDF](https://arxiv.org/pdf/2604.14892)  

**Abstract**: Evaluating medical AI systems using expert clinician panels is costly and slow, motivating the use of large language models (LLMs) as alternative adjudicators. Here, we evaluate an LLM jury composed of three frontier AI models scoring 3333 diagnoses on 300 real-world middle-income country (MIC) hospital cases. Model performance was benchmarked against expert clinician panel and independent human re-scoring panel evaluations. Both LLM and clinician-generated diagnoses are scored across four dimensions: diagnosis, differential diagnosis, clinical reasoning and negative treatment risk. For each of these, we assess scoring difference, inter-rater agreement, scoring stability, severe safety errors and the effect of post-hoc calibration. We find that: (i) the uncalibrated LLM jury scores are systematically lower than clinician panels scores; (ii) the LLM Jury preserves ordinal agreement and exhibits better concordance with the primary expert panels than the human expert re-score panels do; (iii) the probability of severe errors is lower in \lj models compared to the human expert re-score panels; (iv) the LLM Jury shows excellent agreement with primary expert panels' rankings. We find that the LLM jury combined with AI model diagnoses can be used to identify ward diagnoses at high risk of error, enabling targeted expert review and improved panel efficiency; (v) LLM jury models show no self-preference bias. They did not score diagnoses generated by their own underlying model or models from the same vendor more (or less) favourably than those generated by other models. Finally, we demonstrate that LLM jury calibration using isotonic regression improves alignment with human expert panel evaluations. Together, these results provide compelling evidence that a calibrated, multi-model LLM jury can serve as a trustworthy and reliable proxy for expert clinician evaluation in medical AI benchmarking. 

---
# Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models 

**Authors**: Danae Sánchez Villegas, Samuel Lewis-Lim, Nikolaos Aletras, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2604.14888)  

**Abstract**: Recent advances in vision language models (VLMs) offer reasoning capabilities, yet how these unfold and integrate visual and textual information remains unclear. We analyze reasoning dynamics in 18 VLMs covering instruction-tuned and reasoning-trained models from two different model families. We track confidence over Chain-of-Thought (CoT), measure the corrective effect of reasoning, and evaluate the contribution of intermediate reasoning steps. We find that models are prone to answer inertia, in which early commitments to a prediction are reinforced, rather than revised during reasoning steps. While reasoning-trained models show stronger corrective behavior, their gains depend on modality conditions, from text-dominant to vision-only settings. Using controlled interventions with misleading textual cues, we show that models are consistently influenced by these cues even when visual evidence is sufficient, and assess whether this influence is recoverable from CoT. Although this influence can appear in the CoT, its detectability varies across models and depends on what is being monitored. Reasoning-trained models are more likely to explicitly refer to the cues, but their longer and fluent CoTs can still appear visually grounded while actually following textual cues, obscuring modality reliance. In contrast, instruction-tuned models refer to the cues less explicitly, but their shorter traces reveal inconsistencies with the visual input. Taken together, these findings indicate that CoT provides only a partial view of how different modalities drive VLM decisions, with important implications for the transparency and safety of multimodal systems. 

---
# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding 

**Authors**: Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14885)  

**Abstract**: Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance. We propose $\textbf{RACER}$ ($\textbf{R}$etrieval-$\textbf{A}$ugmented $\textbf{C}$ont$\textbf{e}$xtual $\textbf{R}$apid Speculative Decoding), a lightweight and training-free method that integrates retrieved exact patterns with logit-driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM-ZH demonstrate that RACER consistently accelerates inference, achieving more than $2\times$ speedup over autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding. Our source code is available at $\href{this https URL}{this https URL}$. 

---
# SOLIS: Physics-Informed Learning of Interpretable Neural Surrogates for Nonlinear Systems 

**Authors**: Murat Furkan Mansur, Tufan Kumbasar  

**Link**: [PDF](https://arxiv.org/pdf/2604.14879)  

**Abstract**: Nonlinear system identification must balance physical interpretability with model flexibility. Classical methods yield structured, control-relevant models but rely on rigid parametric forms that often miss complex nonlinearities, whereas Neural ODEs are expressive yet largely black-box. Physics-Informed Neural Networks (PINNs) sit between these extremes, but inverse PINNs typically assume a known governing equation with fixed coefficients, leading to identifiability failures when the true dynamics are unknown or state-dependent. We propose \textbf{SOLIS}, which models unknown dynamics via a \emph{state-conditioned second-order surrogate model} and recasts identification as learning a Quasi-Linear Parameter-Varying (Quasi-LPV) representation, recovering interpretable natural frequency, damping, and gain without presupposing a global equation. SOLIS decouples trajectory reconstruction from parameter estimation and stabilizes training with a cyclic curriculum and \textbf{Local Physics Hints} windowed ridge-regression anchors that mitigate optimization collapse. Experiments on benchmarks show accurate parameter-manifold recovery and coherent physical rollouts from sparse data, including regimes where standard inverse methods fail. 

---
# GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation 

**Authors**: Yanyan Zou, Junbo Qi, Lunsong Huang, Yu Li, Kewei Xu, Jiabao Gao, Binglei Zhao, Xuanhua Yang, Sulong Xu, Shengjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14878)  

**Abstract**: Generative Retrieval (GR) offers a promising paradigm for recommendation through next-token prediction (NTP). However, scaling it to large-scale industrial systems introduces three challenges: (i) within a single request, the identical model inputs may produce inconsistent outputs due to the pagination request mechanism; (ii) the prohibitive cost of encoding long user behavior sequences with multi-token item representations based on semantic IDs, and (iii) aligning the generative policy with nuanced user preference signals. We present GenRec, a preference-oriented generative framework deployed on the JD App that addresses above challenges within a single decoder-only architecture. For training objective, we propose Page-wise NTP task, which supervises over an entire interaction page rather than each interacted item individually, providing denser gradient signal and resolving the one-to-many ambiguity of point-wise training. On the prefilling side, an asymmetric linear Token Merger compresses multi-token Semantic IDs in the prompt while preserving full-resolution decoding, reducing input length by ~2X with negligible accuracy loss. To further align outputs with user satisfaction, we introduce GRPO-SR, a reinforcement learning method that pairs Group Relative Policy Optimization with NLL regularization for training stability, and employs Hybrid Rewards combining a dense reward model with a relevance gate to mitigate reward hacking. In month-long online A/B tests serving production traffic, GenRec achieves 9.5% improvement in click count and 8.7% in transaction count over the existing pipeline. 

---
# Vibe-Coding: Feedback-Based Automated Verification with no Human Code Inspection, a Feasibility Study 

**Authors**: Michal Töpfer, František Plášil, Tomáš Bureš, Petr Hnětynka  

**Link**: [PDF](https://arxiv.org/pdf/2604.14867)  

**Abstract**: Vibe coding inherently assumes iterative refinement of LLM-generated code through feedback loops. While effective for conventional software tasks, its reliability in runtime-adaptive systems is unclear -- especially when generated code is not manually inspected. This paper studies feedback-based automated verification of LLM-generated adaptation managers in Collective Adaptive Systems (CAS). We focus on the key challenges of verification in the loop: how to detect failures of generated code at runtime and how to report them precisely enough for an LLM to fix them.
We combine the adaptation loop with a vibe-coding feedback loop where correctness is checked against (i) generic architectural constraints and (ii) functional constraints formalized in Functional Constraints Logic (FCL), a novel first-order temporal logic over potentially finite traces. Conducting the Dragon Hunt CAS case study, we show that fine-grained constraint violations provide actionable feedback that typically yields a valid adaptation manager within a few iterations, while simple coarse metric-based feedback often stalls. Our findings suggest that feedback precision is the dominant factor for reliable vibe coding in systems designed by domain experts with no programming skills, thereby obviating the need for human code inspection. 

---
# MetaDent: Labeling Clinical Images for Vision-Language Models in Dentistry 

**Authors**: Meng-Xun Li, Wen-Hui Deng, Zhi-Xing Wu, Chun-Xiao Jin, Jia-Min Wu, Yue Han, James Kit Hon Tsoi, Gui-Song Xia, Cui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14866)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated significant potential in medical image analysis, yet their application in intraoral photography remains largely underexplored due to the lack of fine-grained, annotated datasets and comprehensive benchmarks. To address this, we present MetaDent, a comprehensive resource that includes (1) a novel and large-scale dentistry image dataset collected from clinical, public, and web sources; (2) a semi-structured annotation framework designed to capture the hierarchical and clinically nuanced nature of dental photography; and (3) comprehensive benchmark suites for evaluating state-of-the-art VLMs on clinical image understanding. Our labeling approach combines a high-level image summary with point-by-point, free-text descriptions of abnormalities. This method enables rich, scalable, and task-agnostic representations. We curated 60,669 dental images from diverse sources and annotated a representative subset of 2,588 images using this meta-labeling scheme. Leveraging Large Language Models (LLMs), we derive standardized benchmarks: approximately 15K Visual Question Answering (VQA) pairs and an 18-class multi-label classification dataset, which we validated with human review and error analysis to justify that the LLM-driven transition reliably preserves fidelity and semantic accuracy. We then evaluate state-of-the-art VLMs across VQA, classification, and image captioning tasks. Quantitative results reveal that even the most advanced models struggle with a fine-grained understanding of intraoral scenes, achieving moderate accuracy and producing inconsistent or incomplete descriptions in image captioning. We publicly release our dataset, annotations, and tools to foster reproducible research and accelerate the development of vision-language systems for dental applications. 

---
# Schema Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding 

**Authors**: Yifan Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.14862)  

**Abstract**: Constrained decoding has been widely adopted for structured generation with large language models (LLMs), ensuring that outputs satisfy predefined formats such as JSON and XML. However, existing approaches largely treat schemas as purely structural constraints and overlook the possibility that their linguistic formulation may affect model behavior. In this work, we study how instruction placement influences model performance in structured generation and show that merely changing the wording of schema keys, without modifying the prompt or model parameters, can significantly alter model performance under constrained decoding. Based on this observation, we propose to reinterpret structured generation as a multi-channel instruction problem, where instructions can be conveyed explicitly through prompts and implicitly through schema keys during decoding. To the best of our knowledge, this is the first work to systematically study how schema key formulation acts as an implicit instruction channel and affects model performance under constrained decoding. Experiments on multiple mathematical reasoning benchmarks show that different model families exhibit distinct sensitivities to these instruction channels: Qwen models consistently benefit from schema-level instructions, while LLaMA models rely more heavily on prompt-level guidance. We further observe non-additive interaction effects between instruction channels, showing that combining multiple channels does not always lead to further improvement. These findings suggest that schema design not only determines output structure, but also carries instruction signals, offering a new perspective on structured generation in LLMs. 

---
# ClimateCause: Complex and Implicit Causal Structures in Climate Reports 

**Authors**: Liesbeth Allein, Nataly Pineda-Castañeda, Andrea Rocci, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2604.14856)  

**Abstract**: Understanding climate change requires reasoning over complex causal networks. Yet, existing causal discovery datasets predominantly capture explicit, direct causal relations. We introduce ClimateCause, a manually expert-annotated dataset of higher-order causal structures from science-for-policy climate reports, including implicit and nested causality. Cause-effect expressions are normalized and disentangled into individual causal relations to facilitate graph construction, with unique annotations for cause-effect correlation, relation type, and spatiotemporal context. We further demonstrate ClimateCause's value for quantifying readability based on the semantic complexity of causal graphs underlying a statement. Finally, large language model benchmarking on correlation inference and causal chain reasoning highlights the latter as a key challenge. 

---
# Efficient Search of Implantable Adaptive Cells for Medical Image Segmentation 

**Authors**: Emil Benedykciuk, Marcin Denkowski, Grzegorz M. Wójcik  

**Link**: [PDF](https://arxiv.org/pdf/2604.14849)  

**Abstract**: Purpose: Adaptive skip modules can improve medical image segmentation, but searching for them is computationally costly. Implantable Adaptive Cells (IACs) are compact NAS modules inserted into U-Net skip connections, reducing the search space compared with full-network NAS. However, the original IAC framework still requires a 200-epoch differentiable search for each backbone and dataset. Methods: We analyzed the temporal behavior of operations and edges within IAC cells during differentiable search on public medical image segmentation benchmarks. We found that operations selected in the final discrete cell typically emerge among the strongest candidates early in training, and their architecture parameters stabilize well before the final epoch. Based on this, we propose a Jensen--Shannon-divergence-based stability criterion that tracks per-edge operation-importance distributions and progressively prunes low-importance operations during search. The accelerated framework is called IAC-LTH. Results: Across four public benchmarks (ACDC, BraTS, KiTS, AMOS), several 2-D U-Net backbones, and a 2-D nnU-Net pipeline, IAC-LTH discovers IAC cells whose patient-level segmentation performance matches and sometimes slightly exceeds that of cells found by the original full-length search, while reducing wall-clock NAS cost by 3.7x to 16x across datasets and backbones. These results are consistent across architectures, benchmarks, and both non-augmented and augmented training settings, while preserving the gains of IAC-equipped U-Nets over strong attention-based and dense-skip baselines. Conclusion: Competitive IAC architectures can be identified from early-stabilizing operations without running the full search, making adaptive skip-module design more practical for medical image segmentation under realistic computational constraints. 

---
# Zero-Shot Retail Theft Detection via Orchestrated Vision Models: A Model-Agnostic, Cost-Effective Alternative to Trained Single-Model Systems 

**Authors**: Haileab Yagersew  

**Link**: [PDF](https://arxiv.org/pdf/2604.14846)  

**Abstract**: Retail theft costs the global economy over \$100 billion annually, yet existing AI-based detection systems require expensive custom model training on proprietary datasets and charge \$200-500/month per store. We present Paza, a zero-shot retail theft detection framework that achieves practical concealment detection without training any model. Our approach orchestrates multiple existing models in a layered pipeline - cheap object detection and pose estimation running continuously, with an expensive vision-language model (VLM) invoked only when behavioral pre-filters trigger. A multi-signal suspicion pre-filter (requiring dwell time plus at least one behavioral signal) reduces VLM invocations by 240x compared to per-frame analysis, bounding calls to <=10/minute and enabling a single GPU to serve 10-20 stores. The architecture is model-agnostic: the VLM component accepts any OpenAI-compatible endpoint, enabling operators to swap between models such as Gemma 4, Qwen3.5-Omni, GPT-4o, or future releases without code changes - ensuring the system improves as the VLM landscape evolves. We evaluate the VLM component on the DCSASS synthesized shoplifting dataset (169 clips, controlled environment), achieving 89.5% precision and 92.8% specificity at 59.3% recall zero-shot - where the recall gap is attributable to sparse frame sampling in offline evaluation rather than VLM reasoning failures, as precision and specificity are the operationally critical metrics determining false alarm rates. We present a detailed cost model showing viability at \$50-100/month per store (3-10x cheaper than commercial alternatives), and introduce a privacy-preserving design that obfuscates faces in the detection pipeline. The source code is available at this https URL. 

---
# Temporal Cross-Modal Knowledge-Distillation-Based Transfer-Learning for Gas Turbine Vibration Fault Detection 

**Authors**: Ali Bagheri Nejad, Mahdi Aliyari-Shoorehdeli, Abolfazl Hasanzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14766)  

**Abstract**: Preventing machine failure is inherently superior to reactive remediation, particularly for critical assets like gas turbines, where early fault detection (FD) is a cornerstone of industrial sustainability. However, modern deep learning-based FD models often face a significant trade-off between architectural complexity and real-time operational constraints, often hindered by a lack of temporal context within restricted vibration signal windows. To address these challenges, this study proposes a Temporal Cross-Modal Knowledge-Distillation Transfer-Learning (TCMKDTL) framework. The framework employs a "privileged" teacher model trained on expansive temporal windows incorporating both past and future signal context to distill latent feature-based knowledge into a compact student model. To mitigate issues of data scarcity and domain shift, the framework leverages robust pre-training on benchmark datasets (such as CWRU) followed by adaptation to target industrial data. Extensive evaluation using experimental and industrial gas turbine (MGT-40) datasets demonstrates that TCMKDTL achieves superior feature separability and diagnostic accuracy compared to conventional pre-trained architectures. Ultimately, this approach enables high-performance, unsupervised anomaly detection suitable for deployment on resource-constrained industrial hardware. 

---
# Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement 

**Authors**: Midan Shim, Seokju Hwang, Kaehyun Um, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14749)  

**Abstract**: Large language models still struggle with faithfulness and hallucinations despite their remarkable reasoning abilities. In Knowledge Graph Question Answering (KGQA), semantic parsing-based approaches address the limitations by understanding constraints in a user's question and converting them into a logical form to execute on a knowledge graph. However, existing KGQA benchmarks and methods are biased toward positive and calculation constraints. Negative constraints are neglected, although they frequently appear in real-world questions. In this paper, we introduce a new task, NEgative-conSTrained (NEST) KGQA, where each question contains at least one negative constraint, and a corresponding dataset, NestKGQA. We also design PyLF, a Python-formatted logical form, since existing logical forms are hardly suitable to express negation clearly while maintaining readability. Furthermore, NEST questions naturally contain multiple constraints. To mitigate their semantic complexity, we present a novel framework named CUCKOO, specialized to multiple-constrained questions and ensuring semantic executability. CUCKOO first generates a constraint-aware logical form draft and performs schema-guided semantic matching. It then selectively applies self-directed refinement only when executing improper logical forms yields an empty result, reducing cost while improving robustness. Experimental results demonstrate that CUCKOO consistently outperforms baselines on both conventional KGQA and NEST-KGQA benchmarks under few-shot settings. 

---
# Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic Concept Adaptation 

**Authors**: Jiaqi Zhu, Shaofeng Cai, Jie Chen, Fang Deng, Beng Chin Ooi, Wenqiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14726)  

**Abstract**: Online anomaly detection (OAD) plays a pivotal role in real-time analytics and decision-making for evolving data streams. However, existing methods often rely on costly retraining and rigid decision boundaries, limiting their ability to adapt both effectively and efficiently to concept drift in dynamic environments. To address these challenges, we propose DyMETER, a dynamic concept adaptation framework for OAD that unifies on-the-fly parameter shifting and dynamic thresholding within a single online paradigm. DyMETER first learns a static detector on historical data to capture recurring central concepts, and then transitions to a dynamic mode to adapt to new concepts as drift occurs. Specifically, DyMETER employs a novel dynamic concept adaptation mechanism that leverages a hypernetwork to generate instance-aware parameter shifts for the static detector, thereby enabling efficient and effective adaptation without retraining or fine-tuning. To achieve robust and interpretable adaptation, DyMETER introduces a lightweight evolution controller to estimate instance-level concept uncertainty for adaptive updates. Further, DyMETER employs a dynamic threshold optimization module to adaptively recalibrates the decision boundary by maintaining a candidate window of uncertain samples, which ensures continuous alignment with evolving concepts. Extensive experiments demonstrate that DyMETER significantly outperforms existing OAD approaches across a wide spectrum of application scenarios. 

---
# Bounded Autonomy for Enterprise AI: Typed Action Contracts and Consumer-Side Execution 

**Authors**: Sarmad Sohail, Ghufran Haider  

**Link**: [PDF](https://arxiv.org/pdf/2604.14723)  

**Abstract**: Large language models are increasingly used as natural-language interfaces to enterprise software, but their direct use as system operators remains unsafe. Model errors can propagate into unauthorized actions, malformed requests, cross-workspace execution, and other costly failures. We argue this is primarily an execution architecture problem. We present a bounded-autonomy architecture in which language models may interpret intent and propose actions, but all executable behavior is constrained by typed action contracts, permission-aware capability exposure, scoped context, validation before side effects, consumer-side execution boundaries, and optional human approval. The enterprise application remains the source of truth for business logic and authorization, while the orchestration engine operates over an explicit published actions manifest. We evaluate the architecture in a deployed multi-tenant enterprise application across three conditions: manual operation, unconstrained AI with safety layers disabled, and full bounded autonomy. Across 25 scenario trials spanning seven failure families, the bounded-autonomy system completed 23 of 25 tasks with zero unsafe executions, while the unconstrained configuration completed only 17 of 25. Two wrong-entity mutations escaped all consumer-contributed layers; only disambiguation and confirmation mechanisms intercept this class. Both AI conditions delivered 13-18x speedup over manual operation. Critically, removing safety layers made the system less useful: structured validation feedback guided the model to correct outcomes in fewer turns, while the unconstrained system hallucinated success. Several safety properties are structurally enforced by code and intercepted all targeted violations regardless of model output. The result is a practical, deployed architecture for making imperfect language models operationally useful in enterprise systems. 

---
# AIPC: Agent-Based Automation for AI Model Deployment with Qualcomm AI Runtime 

**Authors**: Jianhao Su, Zhanwei Wu, ShengTing Huang, Weidong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14661)  

**Abstract**: Edge AI model deployment is a multi-stage engineering process involving model conversion, operator compatibility handling, quantization calibration, runtime integration, and accuracy validation. In
practice, this workflow is long, failure-prone, and heavily dependent on deployment expertise, particularly when targeting hardware-specific inference runtimes. This technical report presents AIPC (AI
Porting Conversion), an AI agent-driven approach for constrained automation of AI model deployment. AIPC decomposes deployment into standardized, verifiable stages and injects deployment-domain knowledge
into agent execution through Agent Skills, helper scripts, and a stage-wise validation loop. This design reduces both the expertise barrier and the engineering time required for hardware deployment.
Using Qualcomm AI Runtime (QAIRT) as the primary scenario, this report examines automated deployment across representative vision, multimodal, and speech models. In the cases covered here, AIPC can
complete deployment from PyTorch to runnable QNN/SNPE inference within 7-20 minutes for structurally regular vision models, with indicative API costs roughly in the range of USD 0.7-10. For more complex
models involving less-supported operators, dynamic shapes, or autoregressive decoding structures, fully automated deployment may still require further advances, but AIPC already provides practical support
for execution, failure localization, and bounded repair. 

---
# Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting 

**Authors**: Inseok Jeon, Minhyeok Lee, Seunghoon Lee, Minseok Kang, Suhwan Cho, Sangyoun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14648)  

**Abstract**: Video outpainting aims to expand the visible content of a video beyond the original frame boundaries while preserving spatial fidelity and temporal coherence across frames. Existing methods primarily rely on large-scale generative models, such as diffusion models. However, generationbased approaches suffer from implicit temporal modeling and limited spatial context. These limitations lead to intraframe and inter-frame inconsistencies, which become particularly pronounced in dynamic scenes and large outpainting scenarios. To overcome these challenges, we propose Seen-to-Scene, a novel framework that unifies propagationbased and generation-based paradigms for video outpainting. Specifically, Seen-to-Scene leverages flow-based propagation with a flow completion network pre-trained for video inpainting, which is fine-tuned in an end-to-end manner to bridge the domain gap and reconstruct coherent motion fields. To further improve the efficiency and reliability of propagation, we introduce a reference-guided latent propagation that effectively propagates source content across frames. Extensive experiments demonstrate that our method achieves superior temporal coherence and visual realism with efficient inference, surpassing even prior state-of-the-art methods that require input-specific adaptation. 

---
# Chaotic CNN for Limited Data Image Classification 

**Authors**: Anusree M, Akhila Henry, Pramod P Nair  

**Link**: [PDF](https://arxiv.org/pdf/2604.14645)  

**Abstract**: Convolutional neural networks (CNNs) often exhibit poor generalisation in limited training data scenarios due to overfitting and insufficient feature diversity. In this work, a simple and effective chaos-based feature transformation is proposed to enhance CNN performance without increasing model complexity. The method applies nonlinear transformations using logistic, skew tent, and sine maps to normalised feature vectors before the classification layer, thereby reshaping the feature space and improving class separability. The approach is evaluated on greyscale datasets (MNIST and Fashion-MNIST) and an RGB dataset (CIFAR-10) using CNN architectures of varying depth under limited data conditions. The results show consistent improvement over the standalone (SA) CNN across all datasets. Notably, a maximum performance gain of 5.43% is achieved on MNIST using the skew tent map with a 3-layer CNN at 40 samples per class. A higher gain of 9.11% is observed on Fashion-MNIST using the sine map with a 3-layer CNN at 50 samples per class. Additionally, a strong gain of 7.47% is obtained on CIFAR-10 using the skew tent map at 200 samples per class. The consistent improvements across different chaotic maps indicate that the performance gain is driven by the shared nonlinear and dynamical properties of chaotic systems. The proposed method is computationally efficient, requires no additional trainable parameters, and can be easily integrated into existing CNN architectures, making it a practical solution for data-scarce image classification tasks. 

---
# Fact4ac at the Financial Misinformation Detection Challenge Task: Reference-Free Financial Misinformation Detection via Fine-Tuning and Few-Shot Prompting of Large Language Models 

**Authors**: Cuong Hoang, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14640)  

**Abstract**: The proliferation of financial misinformation poses a severe threat to market stability and investor trust, misleading market behavior and creating critical information asymmetry. Detecting such misleading narratives is inherently challenging, particularly in real-world scenarios where external evidence or supplementary references for cross-verification are strictly unavailable. This paper presents our winning methodology for the "Reference-Free Financial Misinformation Detection" shared task. Built upon the recently proposed RFC-BENCH framework (Jiang et al. 2026), this task challenges models to determine the veracity of financial claims by relying solely on internal semantic understanding and contextual consistency, rather than external fact-checking. To address this formidable evaluation setup, we propose a comprehensive framework that capitalizes on the reasoning capabilities of state-of-the-art Large Language Models (LLMs). Our approach systematically integrates in-context learning, specifically zero-shot and few-shot prompting strategies, with Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to optimally align the models with the subtle linguistic cues of financial manipulation. Our proposed system demonstrated superior efficacy, successfully securing the first-place ranking on both official leaderboards. Specifically, we achieved an accuracy of 95.4% on the public test set and 96.3% on the private test set, highlighting the robustness of our method and contributing to the acceleration of context-aware misinformation detection in financial Natural Language Processing. Our models (14B and 32B) are available at this https URL. 

---
# StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation 

**Authors**: Geonhui Jang, Dongyoon Han, YoungJoon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14631)  

**Abstract**: Effective code generation requires both model capability and a problem representation that carefully structures how models reason and plan. Existing approaches augment reasoning steps or inject specific structure into how models think, but leave scattered problem conditions unchanged. Inspired by the way humans organize fragmented information into coherent explanations, we propose StoryCoder, a narrative reformulation framework that transforms code generation questions into coherent natural language narratives, providing richer contextual structure than simple rephrasings. Each narrative consists of three components: a task overview, constraints, and example test cases, guided by the selected algorithm and genre. Experiments across 11 models on HumanEval, LiveCodeBench, and CodeForces demonstrate consistent improvements, with an average gain of 18.7% in zero-shot pass@10. Beyond accuracy, our analyses reveal that narrative reformulation guides models toward correct algorithmic strategies, reduces implementation errors, and induces a more modular code structure. The analyses further show that these benefits depend on narrative coherence and genre alignment, suggesting that structured problem representation is important for code generation regardless of model scale or architecture. Our code is available at this https URL. 

---
# ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving 

**Authors**: Yuseon Choi, Jingu Lee, Jungjun Oh, Sunjoo Whang, Byeongcheol Kim, Minsung Kim, Hoi-Jun Yoo, Sangjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.14626)  

**Abstract**: Mixture-of-Experts (MoE) models have become the dominant architecture for large-scale language models, yet on-premises serving remains fundamentally memory-bound as batching turns sparse per-token compute into dense memory activation. Memory-centric architectures (PIM, NMP) improve bandwidth but leave compute underutilized under MoE's low arithmetic intensity at high batch sizes. Speculative decoding (SD) trades idle compute for fewer target invocations, yet verification must load experts even for rejected tokens, severely limiting its benefit in MoE especially at low batch sizes. We propose ELMoE-3D, a hybrid-bonding (HB)-based HW-SW co-designed framework that unifies cache-based acceleration and speculative decoding to offer overall speedup across batch sizes. We identify two intrinsic elasticity axes of MoE-expert and bit-and jointly scale them to construct Elastic Self-Speculative Decoding (Elastic-SD), which serves as both an expert cache and a strongly aligned self-draft model accelerated by high HB bandwidth. Our LSB-augmented bit-sliced architecture exploits inherent redundancy in bit-slice representations to natively support bit-nested execution. On our 3D-stacked hardware, ELMoE-3D achieves an average $6.6\times$ speedup and $4.4\times$ energy efficiency gain over naive MoE serving on xPU across batch sizes 1-16, and delivers $2.2\times$ speedup and $1.4\times$ energy efficiency gain over the best-performing prior accelerator baseline. 

---
# Asking What Matters: Reward-Driven Clarification for Software Engineering Tasks 

**Authors**: Sanidhya Vijayvargiya, Vijay Viswanathan, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2604.14624)  

**Abstract**: Humans often specify tasks incompletely, so assistants must know when and how to ask clarifying questions. However, effective clarification remains challenging in software engineering tasks as not all missing information is equally valuable, and questions must target information users can realistically provide. We study clarification in real software engineering tasks by quantifying which types of information most affect task success and which questions elicit useful responses from simulated users. Using Shapley attribution and distributional comparisons, we identify two key properties of effective clarification: task relevance (which information predicts success) and user answerability (what users can realistically provide). We operationalize these properties as multi-stage reinforcement learning rewards to train CLARITI, an 8B-parameter clarification module, that matches GPT-5's resolution rate on underspecified issues while generating 41% fewer questions. Our results suggest that grounding reward design in empirical analysis of information impact and user answerability improves clarification efficiency. 

---
# Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring 

**Authors**: Sumit Mukherjee, Juan Shu, Nairwita Mazumder, Tate Kernell, Celena Wheeler, Shannon Hastings, Chris Sidey-Gibbons  

**Link**: [PDF](https://arxiv.org/pdf/2604.14616)  

**Abstract**: Clinical value set authoring -- the task of identifying all codes in a standardized vocabulary that define a clinical concept -- is a recurring bottleneck in clinical quality measurement and phenotyping. A natural approach is to prompt a large language model (LLM) to generate the required codes directly, but structured clinical vocabularies are large, version-controlled, and not reliably memorized during pretraining. We propose Retrieval-Augmented Set Completion (RASC): retrieve the $K$ most similar existing value sets from a curated corpus to form a candidate pool, then apply a classifier to each candidate code. Theoretically, retrieve-and-select can reduce statistical complexity by shrinking the effective output space from the full vocabulary to a much smaller retrieved candidate pool. We demonstrate the utility of RASC on 11,803 publicly available VSAC value sets, constructing the first large-scale benchmark for this task. A cross-encoder fine-tuned on SAPBert achieves AUROC~0.852 and value-set-level F1~0.298, outperforming a simpler three-layer Multilayer Perceptron (AUROC~0.799, F1~0.250) and both reduce the number of irrelevant candidates per true positive from 12.3 (retrieval-only) to approximately 3.2 and 4.4 respectively. Zero-shot GPT-4o achieves value-set-level F1~0.105, with 48.6\% of returned codes absent from VSAC entirely. This performance gap widens with increasing value set size, consistent with RASC's theoretical advantage. We observe similar performance gains across two other classifier model types, namely a cross-encoder initialized from pre-trained SAPBert and a LightGBM model, demonstrating that RASC's benefits extend beyond a single model class. The code to download and create the benchmark dataset, as well as the model training code is available at: \href{this https URL}{this https URL}. 

---
# Uncertainty-aware Generative Learning Path Recommendation with Cognition-Adaptive Diffusion 

**Authors**: Xiangrui Xiong, Hang Liang, Baiyang Chen, Zifei Pan, Yanli Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14613)  

**Abstract**: Learning Path Recommendation (LPR) is critical for personalized education, yet current methods often fail to account for historical interaction uncertainty (e.g., lucky guesses or accidental slips) and lack adaptability to diverse learning goals. We propose U-GLAD (Uncertainty-aware Generative Learning Path Recommendation with Cognition-Adaptive Diffusion). To address representation bias, the framework models cognitive states as probability distributions, capturing the learner's underlying true state via a Gaussian LSTM. To ensure highly personalized recommendation, a goal-oriented concept encoder utilizes multi-head attention and objective-specific transformations to dynamically align concept semantics with individual learning goals, generating uniquely tailored embeddings. Unlike traditional discriminative ranking approaches, our model employs a generative diffusion model to predict the latent representation of the next optimal concept. Extensive evaluations on three public datasets demonstrate that U-GLAD significantly outperforms representative baselines. Further analyses confirm its superior capability in perceiving interaction uncertainty and providing stable, goal-driven recommendation paths. 

---
# Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection 

**Authors**: Meng Chen, Kun Wang, Li Lu, Jiaheng Zhang, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14604)  

**Abstract**: Modern Large audio-language models (LALMs) power intelligent voice interactions by tightly integrating audio and text. This integration, however, expands the attack surface beyond text and introduces vulnerabilities in the continuous, high-dimensional audio channel. While prior work studied audio jailbreaks, the security risks of malicious audio injection and downstream behavior manipulation remain underexamined. In this work, we reveal a previously overlooked threat, auditory prompt injection, under realistic constraints of audio data-only access and strong perceptual stealth. To systematically analyze this threat, we propose \textit{AudioHijack}, a general framework that generates context-agnostic and imperceptible adversarial audio to hijack LALMs. \textit{AudioHijack} employs sampling-based gradient estimation for end-to-end optimization across diverse models, bypassing non-differentiable audio tokenization. Through attention supervision and multi-context training, it steers model attention toward adversarial audio and generalizes to unseen user contexts. We also design a convolutional blending method that modulates perturbations into natural reverberation, making them highly imperceptible to users. Extensive experiments on 13 state-of-the-art LALMs show consistent hijacking across 6 misbehavior categories, achieving average success rates of 79\%-96\% on unseen user contexts with high acoustic fidelity. Real-world studies demonstrate that commercial voice agents from Mistral AI and Microsoft Azure can be induced to execute unauthorized actions on behalf of users. These findings expose critical vulnerabilities in LALMs and highlight the urgent need for dedicated defense. 

---
# CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification 

**Authors**: Yian Wang, Yuen Chen, Agam Goyal, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.14602)  

**Abstract**: Large language models (LLMs) frequently generate toxic content, posing significant risks for safe deployment. Current mitigation strategies often degrade generation quality or require costly human annotation. We propose CAUSALDETOX, a framework that identifies and intervenes on the specific attention heads causally responsible for toxic generation. Using the Probability of Necessity and Sufficiency (PNS), we isolate a minimal set of heads that are necessary and sufficient for toxicity. We utilize these components via two complementary strategies: (1) Local Inference-Time Intervention, which constructs dynamic, input-specific steering vectors for context-aware detoxification, and (2) PNS-Guided Fine-Tuning, which permanently unlearns toxic representations. We also introduce PARATOX, a novel benchmark of aligned toxic/non-toxic sentence pairs enabling controlled counterfactual evaluation. Experiments on ToxiGen, ImplicitHate, and ParaDetox show that CAUSALDETOX achieves up to 5.34% greater toxicity reduction compared to baselines while preserving linguistic fluency, and offers a 7x speedup in head selection. 

---
# Mechanistic Decoding of Cognitive Constructs in LLMs 

**Authors**: Yitong Shou, Manhao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14593)  

**Abstract**: While Large Language Models (LLMs) demonstrate increasingly sophisticated affective capabilities, the internal mechanisms by which they process complex emotions remain unclear. Existing interpretability approaches often treat models as black boxes or focus on coarse-grained basic emotions, leaving the cognitive structure of more complex affective states underexplored. To bridge this gap, we propose a Cognitive Reverse-Engineering framework based on Representation Engineering (RepE) to analyze social-comparison jealousy. By combining appraisal theory with subspace orthogonalization, regression-based weighting, and bidirectional causal steering, we isolate and quantify two psychological antecedents of jealousy, Superiority of Comparison Person and Domain Self-Definitional Relevance, and examine their causal effects on model judgments. Experiments on eight LLMs from the Llama, Qwen, and Gemma families suggest that models natively encode jealousy as a structured linear combination of these constituent factors. Their internal representations are broadly consistent with the human psychological construct, treating Superiority as the foundational trigger and Relevance as the ultimate intensity multiplier. Our framework also demonstrates that toxic emotional states can be mechanically detected and surgically suppressed, suggesting a possible route toward representational monitoring and intervention for AI safety in multi-agent environments. 

---
# AgileLog: A Forkable Shared Log for Agents on Data Streams 

**Authors**: Shreesha G. Bhat, Tony Hong, Michael Noguera, Ramnatthan Alagappan, Aishwarya Ganesan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14590)  

**Abstract**: In modern data-streaming systems, alongside traditional programs, a new type of entity has emerged that can interact with streaming data: AI agents. Unlike traditional programs, AI agents use LLM reasoning to accomplish high-level tasks specified in natural language over streaming data. Unfortunately, current streaming systems cannot fully support agents: they lack the fundamental mechanisms to avoid the performance interference caused by agentic tasks and to safely handle agentic writes. We argue that the shared log, the core abstraction underlying streaming data, must support creating forks of itself, and that such a forkable shared log serves as a great substrate for agents acting on streaming data. We propose AgileLog, a new shared log abstraction that provides novel forking primitives for agentic use cases. We design Bolt, an implementation of the AgileLog abstraction, that uses novel techniques to make forks cheap, and provide logical and performance isolation. 

---
# CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations 

**Authors**: Xiping Li, Aier Yang, Jianghong Ma, Kangzhe Liu, Shanshan Feng, Haijun Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14586)  

**Abstract**: The rapid expansion of gaming industry requires advanced recommender systems tailored to its dynamic landscape. Existing Graph Neural Network (GNN)-based methods primarily prioritize accuracy over diversity, overlooking their inherent trade-off. To address this, we previously proposed CPGRec, a balance-oriented gaming recommender system. However, CPGRec fails to account for critical disparities in player-game interactions, which carry varying significance in reflecting players' personal preferences and may exacerbate over-smoothness issues inherent in GNN-based models. Moreover, existing approaches underutilize the reasoning capabilities and extensive knowledge of large language models (LLMs) in addressing these limitations. To bridge this gap, we propose two new modules. First, Preference-informed Edge Reweighting (PER) module assigns signed edge weights to qualitatively distinguish significant player interests and disinterests while then quantitatively measuring preference strength to mitigate over-smoothing in graph convolutions. Second, Preference-informed Representation Generation (PRG) module leverages LLMs to generate contextualized descriptions of games and players by reasoning personal preferences from comparing global and personal interests, thereby refining representations of players and games. Experiments on \textcolor{black}{two Steam datasets} demonstrate CPGRec+'s superior accuracy and diversity over state-of-the-art models. The code is accessible at this https URL. 

---
# Generative Augmented Inference 

**Authors**: Cheng Lu, Mengxin Wang, Dennis J. Zhang, Heng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14575)  

**Abstract**: Data-driven operations management often relies on parameters estimated from costly human-generated labels. Recent advances in large language models (LLMs) and other AI systems offer inexpensive auxiliary data, but introduce a new challenge: AI outputs are not direct observations of the target outcomes, but could involve high-dimensional representations with complex and unknown relationships to human labels. Conventional methods leverage AI predictions as direct proxies for true labels, which can be inefficient or unreliable when this relationship is weak or misspecified. We propose Generative Augmented Inference (GAI), a general framework that incorporates AI-generated outputs as informative features for estimating models of human-labeled outcomes. GAI uses an orthogonal moment construction that enables consistent estimation and valid inference with flexible, nonparametric relationship between LLM-generated outputs and human labels. We establish asymptotic normality and show a "safe default" property: relative to human-data-only estimators, GAI weakly improves estimation efficiency under arbitrary auxiliary signals and yields strict gains whenever the auxiliary information is predictive. Empirically, GAI outperforms benchmarks across diverse settings. In conjoint analysis with weak auxiliary signals, GAI reduces estimation error by about 50% and lowers human labeling requirements by over 75%. In retail pricing, where all methods access the same auxiliary inputs, GAI consistently outperforms alternative estimators, highlighting the value of its construction rather than differences in information. In health insurance choice, it cuts labeling requirements by over 90% while maintaining decision accuracy. Across applications, GAI improves confidence interval coverage without inflating width. Overall, GAI provides a principled and scalable approach to integrating AI-generated information. 

---
# Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG 

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14572)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results: it never sees how the corpus is organized or what it has not yet retrieved, limiting its ability to backtrack or combine scattered evidence. We present Corpus2Skill, which distills a document corpus into a hierarchical skill directory offline and lets an LLM agent navigate it at serve time. The compilation pipeline iteratively clusters documents, generates LLM-written summaries at each level, and materializes the result as a tree of navigable skill files. At serve time, the agent receives a bird's-eye view of the corpus, drills into topic branches via progressively finer summaries, and retrieves full documents by ID. Because the hierarchy is explicitly visible, the agent can reason about where to look, backtrack from unproductive paths, and combine evidence across branches. On WixQA, an enterprise customer-support benchmark for RAG, Corpus2Skill outperforms dense retrieval, RAPTOR, and agentic RAG baselines across all quality metrics. 

---
# Controllable Video Object Insertion via Multiview Priors 

**Authors**: Xia Qi, Peishan Cong, Yichen Yao, Ziyi Wang, Yaoqin Ye, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.14556)  

**Abstract**: Video object insertion is a critical task for dynamically inserting new objects into existing environments. Previous video generation methods focus primarily on synthesizing entire scenes while struggling with ensuring consistent object appearance, spatial alignment, and temporal coherence when inserting objects into existing videos. In this paper, we propose a novel solution for Video Object Insertion, which integrates multi-view object priors to address the common challenges of appearance inconsistency and occlusion handling in dynamic environments. By lifting 2D reference images into multi-view representations and leveraging a dual-path view-consistent conditioning mechanism, our framework ensures stable identity guidance and robust integration across diverse viewpoints. A quality-aware weighting mechanism is also employed to adaptively handle noisy or imperfect inputs. Additionally, we introduce an Integration-Aware Consistency Module that guarantees spatial realism, effectively resolving occlusion and boundary artifacts while maintaining temporal continuity across frames. Experimental results show that our solution significantly improves the quality of video object insertion, providing stable and realistic integration. 

---
# VeriGraphi: A Multi-Agent Framework of Hierarchical RTL Generation for Large Hardware Designs 

**Authors**: Sazzadul Islam, Tasnim Tabassum, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14550)  

**Abstract**: Generating synthesizable Verilog for large, hierarchical hardware designs remains a significant challenge for large language models (LLMs), which struggle to replicate the structured reasoning that human experts employ when translating complex specifications into RTL. When tasked with producing hierarchical Verilog, LLMs frequently lose context across modules, hallucinate interfaces, fabricate inter-module wiring, and fail to maintain structural coherence - failures that intensify as design complexity grows and specifications involve informal prose, figures, and tables that resist direct operationalization. To address these challenges, we present VeriGraphi, a framework that introduces a spec-anchored Knowledge Graph as the architectural substrate driving the RTL generation pipeline. VeriGraphi constructs a HDA, a structured knowledge graph that explicitly encodes module hierarchy, port-level interfaces, wiring semantics, and inter-module dependencies as first-class graph entities and relations. Built through iterative multi-agent analysis of the specification, this Knowledge Graph provides a deterministic, machine-checkable structural scaffold before code generation. Guided by the KG, a progressive coding module incrementally generates pseudo-code and synthesizable RTL while enforcing interface consistency and dependency correctness at each submodule stage. We evaluate VeriGraphi on a benchmark of three representative specification documents from the National Institute of Standards and Technology and their corresponding implementations, and we present a RV32I processor as a detailed case study to illustrate the full pipeline. The results demonstrate that VeriGraphi enables reliable hierarchical RTL generation with minimal human intervention for RISC-V, marking a significant milestone for LLM-generated hardware design while maintaining strong functional correctness. 

---
# CSRA: Controlled Spectral Residual Augmentation for Robust Sepsis Prediction 

**Authors**: Honglin Guo, Rihao Chang, He Jiao, Weizhi Nie, Zhongheng Zhang, Yuehao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14532)  

**Abstract**: Accurate prediction of future risk and disease progression in sepsis is clinically important for early warning and timely intervention in intensive care. However, short-window sepsis prediction remains challenging, because shorter observation windows provide limited historical evidence, whereas longer prediction horizons reduce the number of patient trajectories with valid future supervision. To address this problem, we propose CSRA, a Controlled Spectral Residual Augmentation framework for short-window multi-system ICU time series. CSRA first groups variables by clinical systems and extracts system-level and global representations. It then performs input-adaptive residual perturbation in the spectral domain to generate structured and clinically plausible trajectory variations. To improve augmentation stability and controllability, CSRA is trained end-to-end with the downstream predictor under a unified objective, together with anchor consistency loss and controller regularization. Experiments on a MIMIC-IV sepsis cohort across multiple downstream models show that CSRA is consistently competitive and often superior, reducing regression error by 10.2\% in MSE and 3.7\% in MAE over the non-augmentation baseline, while also yielding consistent gains on classification. CSRA further maintains more favorable performance under shorter observation windows, longer prediction horizons, and smaller training data scales, while also remaining effective on an external clinical dataset~(ZiGongICUinfection), indicating stronger robustness and generalizability in clinically constrained settings. 

---
# CBCL: Safe Self-Extending Agent Communication 

**Authors**: Hugo O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2604.14512)  

**Abstract**: Agent communication languages (ACLs) enable heterogeneous agents to share knowledge and coordinate across diverse domains. This diversity demands extensibility, but expressive extension mechanisms can push the input language beyond the complexity classes where full validation is tractable. We present CBCL (Common Business Communication Language), an agent communication language that constrains all messages, including runtime language extensions, to the deterministic context-free language (DCFL) class. CBCL allows agents to define, transmit, and adopt domain-specific "dialect" extensions as first-class messages; three safety invariants (R1--R3), machine-checked in Lean 4 and enforced in a Rust reference implementation, prevent unbounded expansion, applying declared resource limits, and preserving core vocabulary. We formalize the language and its safety properties in Lean 4, implement a reference parser and dialect engine in Rust with property-based and differential tests, and extract a verified parser binary. Our results demonstrate that homoiconic protocol design, where extension definitions share the same representation as ordinary messages, can be made provably safe. As autonomous agents increasingly extend their own communication capabilities, formally bounding what they can express to each other is a precondition for oversight. 

---
# NewsTorch: A PyTorch-based Toolkit for Learner-oriented News Recommendation 

**Authors**: Rongyao Wang, Veronica Liesaputra, Zhiyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14510)  

**Abstract**: News recommender systems are devised to alleviate the information overload, attracting more and more researchers' attention in recent years. The lack of a dedicated learner-oriented news recommendation toolkit hinders the advancement of research in news recommendation. We propose a PyTorch-based news recommendation toolkit called NewsTorch, developed to support learners in acquiring both conceptual understanding and practical experience. This toolkit provides a modular, decoupled, and extensible framework with a learner-friendly GUI platform that supports dataset downloading and preprocessing. It also enables training, validation, and testing of state-of-the-art neural news recommendation models with standardized evaluation metrics, ensuring fair comparison and reproducible experiments. Our open-source toolkit is released on Github: this https URL. 

---
# On the Expressive Power and Limitations of Multi-Layer SSMs 

**Authors**: Nikola Zubić, Qian Li, Yuyi Wang, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2604.14501)  

**Abstract**: We study the expressive power and limitations of multi-layer state-space models (SSMs). First, we show that multi-layer SSMs face fundamental limitations in compositional tasks, revealing an inherent gap between SSMs and streaming models. Then, we examine the role of chain-of-thought (CoT), showing that offline CoT does not fundamentally increase the expressiveness, while online CoT can substantially increase its power. Indeed, with online CoT, multi-layer SSMs become equivalent in power to streaming algorithms. Finally, we investigate the tradeoff between width and precision, showing that these resources are not interchangeable in the base model, but admit a clean equivalence once online CoT is allowed. Overall, our results offer a unified perspective on how depth, finite precision, and CoT shape the power and limits of SSMs. 

---
# Decoupling Identity from Utility: Privacy-by-Design Frameworks for Financial Ecosystems 

**Authors**: Ifayoyinsola Ibikunle, Tyler Farnan, Senthil Kumar, Mayana Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2604.14495)  

**Abstract**: Financial institutions face tension between maximizing data utility and mitigating the re-identification risks inherent in traditional anonymization methods. This paper explores Differentially Private (DP) synthetic data as a robust "Privacy by Design" framework to resolve this conflict, ensuring output privacy while satisfying stringent regulatory obligations. We examine two distinct generative paradigms: Direct Tabular Synthesis, which reconstructs high-fidelity joint distributions from raw data, and DP-Seeded Agent-Based Modeling (ABM), which uses DP-protected aggregates to parameterize complex, stateful simulations. While tabular synthesis excels at reflecting static historical correlations for QA testing and business analytics, the DP-Seeded ABM offers a forward-looking "counterfactual laboratory" capable of modeling dynamic market behaviors and black swan events. By decoupling individual identities from data utility, these methodologies eliminate traditional data-clearing bottlenecks, enabling seamless cross-institutional research and compliant decision-making in an evolving regulatory landscape. 

---
# A Nonasymptotic Theory of Gain-Dependent Error Dynamics in Behavior Cloning 

**Authors**: Junghoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14484)  

**Abstract**: Behavior cloning (BC) policies on position-controlled robots inherit the closed-loop response of the underlying PD controller, yet the effect of controller gains on BC failure lacks a nonasymptotic theory. We show that independent sub-Gaussian action errors propagate through the gain-dependent closed-loop dynamics to yield sub-Gaussian position errors whose proxy matrix $X_\infty(K)$ governs the failure tail. The probability of horizon-$T$ task failure factorizes into a gain-dependent amplification index $\Gamma_T(K)$ and the validation loss plus a generalization slack, so training loss alone cannot predict closed-loop performance. Under shape-preserving upper-bound structural assumptions the proxy admits the scalar bound $X_\infty(K)\preceq\Psi(K)\bar X$ with $\Psi(K)$ decomposed into label difficulty, injection strength, and contraction, ranking the four canonical regimes with compliant-overdamped (CO) tightest, stiff-underdamped (SU) loosest, and the stiff-overdamped versus compliant-underdamped ordering system-dependent. For the canonical scalar second-order PD system the closed-form continuous-time stationary variance $X_\infty^{\mathrm{c}}(\alpha,\beta)=\sigma^2\alpha/(2\beta)$ is strictly monotone in stiffness and damping over the entire stable orthant, covering both underdamped and overdamped regimes, and the exact zero-order-hold (ZOH) discretization inherits this monotonicity. The analysis provides the first nonasymptotic explanation of the empirical finding that compliant, overdamped controllers improve BC success rates. 

---
# Auxiliary Finite-Difference Residual-Gradient Regularization for PINNs 

**Authors**: Stavros Kassinos  

**Link**: [PDF](https://arxiv.org/pdf/2604.14472)  

**Abstract**: Physics-informed neural networks (PINNs) are often selected by a single scalar loss even when the quantity of interest is more specific. We study a hybrid design in which the governing PDE residual remains automatic-differentiation (AD) based, while finite differences (FD) appear only in a weak auxiliary term that penalizes gradients of the sampled residual field. The FD term regularizes the residual field without replacing the PDE residual itself.
We examine this idea in two stages. Stage 1 is a controlled Poisson benchmark comparing a baseline PINN, the FD residual-gradient regularizer, and a matched AD residual-gradient baseline. Stage 2 transfers the same logic to a three-dimensional annular heat-conduction benchmark (PINN3D), where baseline errors concentrate near a wavy outer wall and the auxiliary grid is implemented as a body-fitted shell adjacent to the wall.
In Stage 1, the FD regularizer reproduces the main effect of residual-gradient control while exposing a trade-off between field accuracy and residual cleanliness. In Stage 2, the shell regularizer improves the application-facing quantities, namely outer-wall flux and boundary-condition behavior. Across seeds 0-5 and 100k epochs, the most reliable tested configuration is a fixed shell weight of 5e-4 under the Kourkoutas-beta optimizer regime: relative to a matched run without the shell term, it reduces the mean outer-wall BC RMSE from 1.22e-2 to 9.29e-4 and the mean wall-flux RMSE from 9.21e-3 to 9.63e-4. Adam with beta2=0.999 becomes usable when the initial learning rate is reduced to 1e-3, although its shell benefit is less robust than under Kourkoutas-beta. Overall, the results support a targeted view of hybrid PINNs: an auxiliary-only FD regularizer is most valuable when it is aligned with the physical quantity of interest, here the outer-wall flux. 

---
# FocalLens: Visualizing Narratives through Focalization 

**Authors**: S M Raihanul Alam, Md Dilshadur Rahman, Md Naimul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2604.14456)  

**Abstract**: Visualizing narratives is useful to writers to reflect on unfinished drafts and identify unintentional biases and inconsistencies. Literary scholars can use the visualizations to identify nuanced patterns and literary styles from written text. Current narrative visualization is limited to representing character and location co-occurrences in a timeline, omitting important and complex narrative components such as focalization, causality, and speech. This paper aims to capture and visualize underexplored, complex narrative components as a basis for narrative visualization. As a starting point, we propose a new narrative visualization, named FocalLens, that uses focalization, the component that establishes who sees or perceives the events in a narrative, for representing the narrative. We provide the theoretical foundation of focalization and describe various types and facets of focalization. The details are incorporated in the novel visualization that captures how different characters perceive an event, who directly participate in an event, who indirectly observe the event, and who narrate the event. We also developed a tool that provides fluid interaction between the text and the proposed visualization. The tool was evaluated with four writers and scholars in a qualitative study, where writers analyzed their draft stories and scholars analyzed well-known stories. The findings suggest the tool added a new dimension to the workflow for writers and scholars, an analytical lens that is not available otherwise. We conclude by identifying design implications and future directions. 

---
# FAIR Universe Weak Lensing ML Uncertainty Challenge: Handling Uncertainties and Distribution Shifts for Precision Cosmology 

**Authors**: Biwei Dai, Po-Wen Chang, Wahid Bhimji, Paolo Calafiura, Ragansu Chakkappai, Yuan-Tang Chou, Sascha Diefenbacher, Jordan Dudley, Ibrahim Elsharkawy, Steven Farrell, Isabelle Guyon, Chris Harris, Elham E Khoda, Benjamin Nachman, David Rousseau, Uroš Seljak, Ihsan Ullah, Yulei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14451)  

**Abstract**: Weak gravitational lensing, the correlated distortion of background galaxy shapes by foreground structures, is a powerful probe of the matter distribution in our universe and allows accurate constraints on the cosmological model. In recent years, high-order statistics and machine learning (ML) techniques have been applied to weak lensing data to extract the nonlinear information beyond traditional two-point analysis. However, these methods typically rely on cosmological simulations, which poses several challenges: simulations are computationally expensive, limiting most realistic setups to a low training data regime; inaccurate modeling of systematics in the simulations create distribution shifts that can bias cosmological parameter constraints; and varying simulation setups across studies make method comparison difficult. To address these difficulties, we present the first weak lensing benchmark dataset with several realistic systematics and launch the FAIR Universe Weak Lensing Machine Learning Uncertainty Challenge. The challenge focuses on measuring the fundamental properties of the universe from weak lensing data with limited training set and potential distribution shifts, while providing a standardized benchmark for rigorous comparison across methods. Organized in two phases, the challenge will bring together the physics and ML communities to advance the methodologies for handling systematic uncertainties, data efficiency, and distribution shifts in weak lensing analysis with ML, ultimately facilitating the deployment of ML approaches into upcoming weak lensing survey analysis. 

---
# Crowdsourcing of Real-world Image Annotation via Visual Properties 

**Authors**: Xiaolei Diao, Fausto Giunchiglia  

**Link**: [PDF](https://arxiv.org/pdf/2604.14449)  

**Abstract**: Recent advances in data-centric artificial intelligence highlight inherent limitations in object recognition datasets. One of the primary issues stems from the semantic gap problem, which results in complex many-to-many mappings between visual data and linguistic descriptions. This bias adversely affects performance in computer vision tasks. This paper proposes an image annotation methodology that integrates knowledge representation, natural language processing, and computer vision techniques, aiming to reduce annotator subjectivity by applying visual property constraints. We introduce an interactive crowdsourcing framework that dynamically asks questions based on a predefined object category hierarchy and annotator feedback, guiding image annotation by visual properties. Experiments demonstrate the effectiveness of this methodology, and annotator feedback is discussed to optimize the crowdsourcing setup. 

---
# Robustness Analysis of Machine Learning Models for IoT Intrusion Detection Under Data Poisoning Attacks 

**Authors**: Fortunatus Aabangbio Wulnye, Justice Owusu Agyemang, Kwame Opuni-Boachie Obour Agyekum, Kwame Agyeman-Prempeh Agyekum, Kingsford Sarkodie Obeng Kwakye, Francisca Adomaa Acheampong  

**Link**: [PDF](https://arxiv.org/pdf/2604.14444)  

**Abstract**: Ensuring the reliability of machine learning-based intrusion detection systems remains a critical challenge in Internet of Things (IoT) environments, particularly as data poisoning attacks increasingly threaten the integrity of model training pipelines. This study evaluates the susceptibility of four widely used classifiers, Random Forest, Gradient Boosting Machine, Logistic Regression, and Deep Neural Network models, against multiple poisoning strategies using three real-world IoT datasets. Results show that while ensemble-based models exhibit comparatively stable performance, Logistic Regression and Deep Neural Networks suffer degradation of up to 40% under label manipulation and outlier-based attacks. Such disruptions significantly distort decision boundaries, reduce detection fidelity, and undermine deployment readiness. The findings highlight the need for adversarially robust training, continuous anomaly monitoring, and feature-level validation within operational Network Intrusion Detection Systems. The study also emphasizes the importance of integrating resilience testing into regulatory and compliance frameworks for AI-driven IoT security. Overall, this work provides an empirical foundation for developing more resilient intrusion detection pipelines and informs future research on adaptive, attack-aware models capable of maintaining reliability under adversarial IoT conditions. 

---
# Hierarchical vs. Flat Iteration in Shared-Weight Transformers 

**Authors**: Sang-Il Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.14442)  

**Abstract**: We present an empirical study of whether hierarchically structured, shared-weight recurrence can match the representational quality of independent-layer stacking in a Transformer-based language model. HRM-LM replaces L independent Transformer layers with a two-speed recurrent pair: a Fast module operating at every step for local refinement, and a Slow module operating every T steps for global compression. This recurrent hierarchy is unrolled for M = N x T steps with shared parameters. The central and most robust finding, supported by a parameter-matched Universal Transformer ablation (UniTF, 1.2B) across five independent runs, is a sharp empirical gap between the two approaches. 

---
# LLMs taking shortcuts in test generation: A study with SAP HANA and LevelDB 

**Authors**: Vekil Bekmyradov, Noah C. Pütz, Thomas Bartz-Beielstein  

**Link**: [PDF](https://arxiv.org/pdf/2604.14437)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results on public benchmarks, often leading to claims of advanced reasoning and understanding. However, recent research in cognitive science reveals that these models sometimes rely on shallow heuristics and memorization, taking shortcuts rather than demonstrating genuine cognitive abilities. This paper investigates LLM behavior in automated test generation for software, contrasting performance on an open-source system (LevelDB) with SAP HANA, one of the most widely deployed commercial database systems worldwide, whose proprietary codebase is guaranteed to be absent from training data. We combine cognitive evaluation principles, drawing on Mitchell's mechanism-focused assessment methodology, with empirical software testing, employing mutation score and iterative compiler-feedback repair loops to assess both accuracy and underlying reasoning strategies. Results show that LLMs excel on familiar, open-source benchmarks but struggle with unseen, complex domains, often prioritizing compilability over semantic effectiveness. These findings provide independent software engineering evidence for the broader claim that current LLMs lack robust reasoning, and highlight the need for evaluation frameworks that penalize trivial shortcuts and reward true generalization. 

---
# Three-Phase Transformer 

**Authors**: Mohammad R. Abu Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2604.14430)  

**Abstract**: We present Three-Phase Transformer (3PT), a residual-stream structural prior for decoder-only Transformers on a standard SwiGLU + RMSNorm + RoPE + GQA backbone. The hidden vector is partitioned into N equally-sized cyclic channels, each maintained by phase-respecting ops: a per-channel RMSNorm, a 2D Givens rotation between attention and FFN that rotates each channel by theta + i*(2*pi/N), and a head-count constraint aligning GQA heads with the partition. The architecture is a self-stabilizing equilibrium between scrambling and re-imposition, not a bolted-on module. The partition carves out a one-dimensional DC subspace orthogonal to the channels, into which we inject a fixed Gabriel's horn profile r(p) = 1/(p+1) as an absolute-position side-channel composing orthogonally with RoPE's relative-position rotation. The canonical N=3 borrows its metaphor from balanced three-phase AC, where three sinusoids 120 degrees apart sum to zero with no anti-correlated pair. At 123M parameters on WikiText-103, 3PT achieves -7.20% perplexity (-2.62% bits-per-byte) over a matched RoPE-Only baseline at +1,536 parameters (0.00124% of total), with 1.93x step-count convergence speedup (1.64x wall-clock). N behaves as a parameter-sharing knob rather than a unique optimum: at 5.5M an N-sweep over {1,2,3,4,6,8,12} is near-monotone with N=1 winning; at 123M a three-seed sweep finds N=3 and N=1 statistically indistinguishable. The load-bearing mechanism is the channel-partitioned residual stream, per-block rotation, per-phase normalization, and horn DC injection. We characterize (a) self-stabilization of the geometry without explicit enforcement, a novel instance of the conservation-law framework for neural networks; (b) a U-shaped depth profile of rotation-angle drift at 12 layers; (c) orthogonal composition with RoPE, attention, and FFN. 

---
# SpaceMind: A Modular and Self-Evolving Embodied Vision-Language Agent Framework for Autonomous On-orbit Servicing 

**Authors**: Aodi Wu, Haodong Han, Xubo Luo, Ruisuo Wang, Shan He, Xue Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14399)  

**Abstract**: Autonomous on-orbit servicing demands embodied agents that perceive through visual sensors, reason about 3D spatial situations, and execute multi-phase tasks over extended horizons. We present SpaceMind, a modular and self-evolving vision-language model (VLM) agent framework that decomposes knowledge, tools, and reasoning into three independently extensible dimensions: skill modules with dynamic routing, Model Context Protocol (MCP) tools with configurable profiles, and injectable reasoning-mode skills. An MCP-Redis interface layer enables the same codebase to operate across simulation and physical hardware without modification, and a Skill Self-Evolution mechanism distills operational experience into persistent skill files without model fine-tuning. We validate SpaceMind through 192 closed-loop runs across five satellites, three task types, and two environments, a UE5 simulation and a physical laboratory, deliberately including degraded conditions to stress-test robustness. Under nominal conditions all modes achieve 90--100% navigation success; under degradation, the Prospective mode uniquely succeeds in search-and-approach tasks where other modes fail. A self-evolution study shows that the agent recovers from failure in four of six groups from a single failed episode, including complete failure to 100% success and inspection scores improving from 12 to 59 out of 100. Real-world validation confirms zero-code-modification transfer to a physical robot with 100% rendezvous success. Code: this https URL 

---
# Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection 

**Authors**: David Basil, Chirooth Girigowda, Bradley Hauer, Sahir Momin, Ning Shi, Grzegorz Kondrak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14397)  

**Abstract**: We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects English synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate these alignments and ensure their quality, we augment a pre-trained base aligner with a bilingual dictionary, which is also used to filter out incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and requiring few external resources. We plan to make our code, documentation, and generated sense inventories accessible. 

---
# BiCon-Gate: Consistency-Gated De-colloquialisation for Dialogue Fact-Checking 

**Authors**: Hyunkyung Park, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2604.14389)  

**Abstract**: Automated fact-checking in dialogue involves multi-turn conversations where colloquial language is frequent yet understudied. To address this gap, we propose a conservative rewrite candidate for each response claim via staged de-colloquialisation, combining lightweight surface normalisation with scoped in-claim coreference resolution. We then introduce BiCon-Gate, a semantics-aware consistency gate that selects the rewrite candidate only when it is semantically supported by the dialogue context, otherwise falling back to the original claim. This gated selection stabilises downstream fact-checking and yields gains in both evidence retrieval and fact verification. On the DialFact benchmark, our approach improves retrieval and verification, with particularly strong gains on SUPPORTS, and outperforms competitive baselines, including a decoder-based one-shot LLM rewrite that attempts to perform all de-colloquialisation steps in a single pass. 

---
# Coalition Formation in LLM Agent Networks: Stability Analysis and Convergence Guarantees 

**Authors**: Dongxin Guo, Jikun Wu, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14386)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed in multi-agent systems requiring strategic coordination. While recent work has analyzed LLM behavior in two-player games, coalition formation, where $n$ agents dynamically form cooperative groups, remains theoretically uncharacterized. We present the first framework grounding coalition formation in LLM agent networks in hedonic game theory with formal stability guarantees. We introduce the LLM Coalition Formation Game (LCFG), establish sufficient conditions for Nash-stable partitions, and prove complexity results. Our analysis reveals that LLM agents exhibit bounded rationality characterized by $\epsilon$-rational preferences; we provide both deterministic existence guarantees and consistency-driven stability bounds whose predictions are consistent with empirical outcomes. Experiments with GPT-4, Claude-3, and Llama-3 across 2,400 episodes validate our framework: LLM coalitions achieve Nash stability in 73.2% of cases under our Coalition-of-Thought (CoalT) protocol, compared to 58.4% under chain-of-thought and 41.8% under standard prompting ($p < 0.001$). Our framework provides theoretical foundations for designing stable multi-agent LLM systems. 

---
# Step-level Denoising-time Diffusion Alignment with Multiple Objectives 

**Authors**: Qi Zhang, Dawei Wang, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.14379)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful tool for aligning diffusion models with human preferences, typically by optimizing a single reward function under a KL regularization constraint. In practice, however, human preferences are inherently pluralistic, and aligned models must balance multiple downstream objectives, such as aesthetic quality and text-image consistency. Existing multi-objective approaches either rely on costly multi-objective RL fine-tuning or on fusing separately aligned models at denoising time, but they generally require access to reward values (or their gradients) and/or introduce approximation error in the resulting denoising objectives. In this paper, we revisit the problem of RL fine-tuning for diffusion models and address the intractability of identifying the optimal policy by introducing a step-level RL formulation. Building on this, we further propose Multi-objective Step-level Denoising-time Diffusion Alignment (MSDDA), a retraining-free framework for aligning diffusion models with multiple objectives, obtaining the optimal reverse denoising distribution in closed form, with mean and variance expressed directly in terms of single-objective base models. We prove that this denoising-time objective is exactly equivalent to the step-level RL fine-tuning, introducing no approximation error. Moreover, we provide numerical results, which indicate our method outperforms existing denoising-time approaches. 

---
# Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery 

**Authors**: Noureddine Kermiche  

**Link**: [PDF](https://arxiv.org/pdf/2604.14375)  

**Abstract**: Catastrophic forgetting remains a primary hurdle in sequential task learning for artificial neural networks. We propose a silicon-native modular architecture that achieves structural parameter isolation using Task-Specific Experts and a distributed, outlier-based Gatekeeper. Moving beyond traditional sequential consolidation, our framework utilizes a Simultaneous Pipeline where Teacher learning, Student distillation, and Router manifold acquisition occur in parallel while raw data is present in a localized training session. This approach ensures computational efficiency and complies with privacy mandates like GDPR by deleting raw data as soon as a task is learned. We demonstrate that a Tight-Bottleneck Autoencoder (TB-AE) can effectively distinguish semantically crowded manifolds in high-dimensional latent spaces, overcoming the posterior collapse inherent to standard variational methods. By establishing strict topological boundaries, our TB-AE resolves latent space crowding in 4096-D LLM embeddings to provide a robust, unsupervised novelty signal. Furthermore, we validate an Autonomous Retrieval mechanism that confidently identifies returning manifolds, enabling stable lifelong learning without redundant module instantiation. Empirical results demonstrate that our ``Live Distillation'' approach acts as a natural regularizer, achieving strong retention across computer vision and natural language processing domains without suffering a student fidelity gap. 

---
# SatBLIP: Context Understanding and Feature Identification from Satellite Imagery with Vision-Language Learning 

**Authors**: Xue Wu, Shengting Cao, Jiaqi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.14373)  

**Abstract**: Rural environmental risks are shaped by place-based conditions (e.g., housing quality, road access, land-surface patterns), yet standard vulnerability indices are coarse and provide limited insight into risk contexts. We propose SatBLIP, a satellite-specific vision-language framework for rural context understanding and feature identification that predicts county-level Social Vulnerability Index (SVI). SatBLIP addresses limitations of prior remote sensing pipelines-handcrafted features, manual virtual audits, and natural-image-trained VLMs-by coupling contrastive image-text alignment with bootstrapped captioning tailored to satellite semantics. We use GPT-4o to generate structured descriptions of satellite tiles (roof type/condition, house size, yard attributes, greenery, and road context), then fine-tune a satellite-adapted BLIP model to generate captions for unseen images. Captions are encoded with CLIP and fused with LLM-derived embeddings via attention for SVI estimation under spatial aggregation. Using SHAP, we identify salient attributes (e.g., roof form/condition, street width, vegetation, cars/open space) that consistently drive robust predictions, enabling interpretable mapping of rural risk environments. 

---
# The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models 

**Authors**: Akshay Paruchuri, Ishan Chatterjee, Henry Fuchs, Ehsan Adeli, Piotr Didyk  

**Link**: [PDF](https://arxiv.org/pdf/2604.14363)  

**Abstract**: Multimodal language models systematically underperform on visual perception tasks, yet the structure underlying this failure remains poorly understood. We propose centroid replacement, collapsing each token to its nearest K-means centroid, as a controlled probe for modal dependence. Across seven models spanning three architecture families, erasing text centroid structure costs 4$\times$ more accuracy than erasing visual centroid structure, exposing a universal imbalance where language representations overshadow vision even on tasks that demand visual reasoning. We exploit this asymmetry through text centroid contrastive decoding, recovering up to +16.9% accuracy on individual tasks by contrastively decoding against a text-centroid-erased reference. This intervention varies meaningfully with training approaches: standard fine-tuned models show larger gains (+5.6% on average) than preference-optimized models (+1.5% on average). Our findings suggest that modal competition is structurally localized, correctable at inference time without retraining, and quantifiable as a diagnostic signal to guide future multimodal training. 

---
# APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI 

**Authors**: Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, Ankit Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2604.14362)  

**Abstract**: Large language models still struggle with reliable long-term conversational memory: simply enlarging context windows or applying naive retrieval often introduces noise and destabilizes responses. We present APEX-MEM, a conversational memory system that combines three key innovations: (1) a property graph which uses domain-agnostic ontology to structure conversations as temporally grounded events in an entity-centric framework, (2) append-only storage that preserves the full temporal evolution of information, and (3) a multi-tool retrieval agent that understands and resolves conflicting or evolving information at query time, producing a compact and contextually relevant memory summary. This retrieval-time resolution preserves the full interaction history while suppressing irrelevant details. APEX-MEM achieves 88.88% accuracy on LOCOMO's Question Answering task and 86.2% on LongMemEval, outperforming state-of-the-art session-aware approaches and demonstrating that structured property graphs enable more temporally coherent long-term conversational reasoning. 

---
# When PCOS Meets Eating Disorders: An Explainable AI Approach to Detecting the Hidden Triple Burden 

**Authors**: Apoorv Prasad, Susan McRoy  

**Link**: [PDF](https://arxiv.org/pdf/2604.14356)  

**Abstract**: Women with polycystic ovary syndrome (PCOS) face substantially elevated risks of body image distress, disordered eating, and metabolic challenges, yet existing natural language processing approaches for detecting these conditions lack transparency and cannot identify co-occurring presentations. We developed small, open-source language models to automatically detect this triple burden in social media posts with grounded explainability. We collected 1,000 PCOS-related posts from six subreddits, with two trained annotators labeling posts using guidelines operationalizing Lee et al. (2017) clinical framework. Three models (Gemma-2-2B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B) were fine-tuned using Low-Rank Adaptation to generate structured explanations with textual evidence. The best model achieved 75.3 percent exact match accuracy on 150 held-out posts, with robust comorbidity detection and strong explainability. Performance declined with diagnostic complexity, indicating their best use is for screening rather than autonomous diagnosis. 

---
# Tight Sample Complexity Bounds for Best-Arm Identification Under Bounded Systematic Bias 

**Authors**: Tianhao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.14345)  

**Abstract**: As search depth increases in autonomous reasoning and embodied planning, the candidate action space expands exponentially, heavily taxing computational budgets. While heuristic pruning is a common countermeasure, it operates without formal safety guarantees when surrogate models (like LLMs) exhibit systematic evaluation biases. This paper frames the node expansion process as a localized Best-Arm Identification (BAI) problem over dynamic frontiers, subject to a bounded systematic bias $L$. By inverting the Lambert W function, we establish an additive sample complexity of $\mathcal{O}((\Delta-4L)^{-2})$, which indicates that safe node elimination is only feasible when the empirical reward gap exceeds $4L$. We complement this with an information-theoretic lower bound of $\Omega((\Delta-2L)^{-2})$ to confirm the structural limits of biased search. Subsequent evaluations on both synthetic trees and complex reasoning tasks demonstrate that adhering to this local safety boundary successfully preserves optimal trajectories while maximizing sample allocation efficiency. 

---
# Mamba-SSM with LLM Reasoning for Biomarker Discovery: Causal Feature Refinement via Chain-of-Thought Gene Evaluation 

**Authors**: Pushpa Kumar Balan, Aijing Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14334)  

**Abstract**: Gradient saliency from deep sequence models surfaces candidate biomarkers efficiently, but the resulting gene lists are contaminated by tissue-composition confounders that degrade downstream classifiers. We study whether LLM chain-of-thought (CoT) reasoning can faithfully filter these confounders, and whether reasoning quality drives downstream performance. We train a Mamba SSM on TCGA-BRCA RNA-seq and extract the top-50 genes by gradient saliency; DeepSeek-R1 evaluates every candidate with structured CoT to produce a final 17-gene set. The raw 50-gene saliency set (no LLM) performs worse than a 5,000-gene variance baseline (AUC 0.832 vs. 0.903), while the LLM-filtered set surpasses it (AUC 0.927), using 294x fewer features. A faithfulness audit (COSMIC CGC, OncoKB, PAM50) reveals only 6 of 17 selected genes (35.3%) are validated BRCA biomarkers, yet 10 of 16 known BRCA genes in the input were missed - including FOXA1. This gap between downstream performance and reasoning faithfulness suggests selective faithfulness: targeted confounder removal is sufficient for performance gains even without comprehensive recall. 

---
# Thermodynamic Diffusion Inference with Minimal Digital Conditioning 

**Authors**: Aditi De  

**Link**: [PDF](https://arxiv.org/pdf/2604.14332)  

**Abstract**: Diffusion-model inference and overdamped Langevin dynamics are formally identical. A physical substrate that encodes the score function therefore equilibrates to the correct output by thermodynamics alone, requiring no digital arithmetic during inference and potentially achieving a $10{,}000\times$ reduction in energy relative to a GPU. Two fundamental barriers have until now prevented this equivalence from being realized at production scale: non-local skip connections, which locally coupled analog substrates cannot represent, and input conditioning, in which the coupling constants carry roughly $2{,}600\times$ too little signal to anchor the system to a specific input.
We resolve both obstacles. \emph{Hierarchical bilinear coupling} encodes U-Net skip connections as rank-$k$ inter-module interactions derived directly from the singular structure of the encoder and decoder Gram matrices, requiring only $O(Dk)$ physical connections instead of $O(D^2)$. A \emph{minimal digital interface} -- a 4-dimensional bottleneck encoder together with a 16-unit transfer network, totalling \textbf{2,560 parameters} -- overcomes the conditioning barrier. When evaluated on activations drawn from a trained denoising U-Net, the complete system attains a decoder cosine similarity of \textbf{0.9906} against an oracle upper bound of 1.0000, while preserving theoretical net energy savings of approximately $10^7\times$ over GPU inference. These results constitute the first demonstration of trained-weight, production-scale thermodynamic diffusion inference. 

---
# Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance 

**Authors**: Bar Alon, Itamar Zimerman, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2604.14325)  

**Abstract**: Large language models (LLMs) achieve strong performance and have revolutionized NLP, but their lack of explainability keeps them treated as black boxes, limiting their use in domains that demand transparency and trust. A promising direction to address this issue is post-hoc text-based explanations, which aim to explain model decisions in natural language. Prior work has focused on generating convincing rationales that appear to be subjectively faithful, but it remains unclear whether these explanations are epistemically faithful, whether they reflect the internal evidence the model actually relied on for its decision. In this paper, we first assess the epistemic faithfulness of LLM-generated explanations via counterfactuals and show that they are often unfaithful. We then introduce a training-free method that enhances faithfulness by guiding explanation generation through attention-level interventions, informed by token-level heatmaps extracted via a faithful attribution method. This method significantly improves epistemic faithfulness across multiple models, benchmarks, and prompts. 

---
# Challenges and Future Directions in Agentic Reverse Engineering Systems 

**Authors**: Salem Radey, Jack West, Kassem Fawaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.14317)  

**Abstract**: Agentic systems built on large language models (LLMs) are increasingly being used for complex security tasks, including binary reverse engineering (RE). Despite recent growth in popularity and capability, these systems continue to face limitations in realistic settings. Cutting-edge systems still fail in complex RE scenarios that involve obfuscation, timing, and unique architecture. In this work, we examine how agentic systems perform reverse engineering tasks with static, dynamic, and hybrid agents. Through an analysis of existing agentic tool usage, we identify several limitations, including token constraints, struggles with obfuscation, and a lack of program guardrails. From these findings, we outline current challenges and position future directions for system designers to overcome from a security perspective. 

---
# DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines 

**Authors**: Gabriel Pimenta de Freitas Cardoso, Caio Lucas da Silva Chacon, Jonas Felipe da Fonseca Oliveira, Paulo Henrique de Medeiros Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14314)  

**Abstract**: This manuscript introduces DharmaOCR Full and Lite, a pair of specialized small language models (SSLMs) for structured OCR that jointly optimize transcription quality, generation stability, and inference cost. It also presents DharmaOCR-Benchmark, a benchmark that covers printed, handwritten, and legal/administrative documents, and proposes a unified evaluation protocol that measures fidelity and structure while explicitly tracking text degeneration as a first-class benchmark metric (alongside unit cost). Beyond reporting degeneration rates, the manuscript empirically shows degeneration is not merely a quality failure, since it materially worsens production performance by increasing response time, reducing throughput, and inflating computational cost due to abnormally long generations. To the best of the author's knowledge, as a methodological contribution, this is the first application of Direct Preference Optimization (DPO) for OCR, explicitly using degenerate generations as rejected examples to penalize looping behavior. Combined with Supervised Fine-Tuning (SFT) for enforcing a strict JSON schema (header, margin, footer, and text), DPO consistently reduces degeneration rate across model families (up to 87.6% relative) while preserving or improving extraction quality. The resulting models, namely, DharmaOCR Full (7B) and DharmaOCR Lite (3B), set a new state-of-the-art on DharmaOCR-Benchmark, outperforming each open-source and commercial baseline model evaluated regarding extraction quality, reaching 0.925 and 0.911 scores with 0.40% and 0.20% degeneration rates. AWQ quantization reduced up to 22% per-page cost with negligible quality loss, enabling a strong quality-cost trade-off in comparison to proprietary OCR APIs and open-source alternatives. 

---
# Aerial Multi-Functional RIS in Fluid Antennas-Aided Full-Duplex Networks: A Self-Optimized Hybrid Deep Reinforcement Learning Approach 

**Authors**: Li-Hsiang Shen, Yu-Quan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14309)  

**Abstract**: To address high data traffic demands of sixth-generation (6G) networks, this paper proposes a novel architecture that integrates autonomous aerial vehicles (AAVs) and multi-functional reconfigurable intelligent surfaces (MF-RISs) as AM-RIS in fluid antenna (FA)-assisted full-duplex (FD) networks. The AM-RIS provides hybrid functionalities, including signal reflection, amplification, and energy harvesting (EH), potentially improving both signal coverage and sustainability. Meanwhile, FA facilitates fine-grained spatial adaptability at FD-enabled base station (BS), which complements residual self-interference (SI) suppression. We aim at maximizing the overall energy efficiency (EE) by jointly optimizing transmit DL beamforming at BS, UL user power, configuration of AM-RIS, and positions of the FA and AM-RIS. Owing to the hybrid continuous-discrete parameters and high dimensionality of the intractable problem, we have conceived a self-optimized multi-agent hybrid deep reinforcement learning (DRL) framework (SOHRL), which integrates multi-agent deep Q-networks (DQN) and multi-agent proximal policy optimization (PPO), respectively handling discrete and continuous actions. To enhance self-adaptability, an attention-driven state representation and meta-level hyperparameter optimization are incorporated, enabling multi-agents to autonomously adjust learning hyperparameters. Simulation results validate the effectiveness of the proposed AM-RIS-enabled FA-aided FD networks empowered by SOHRL algorithm. The results reveal that SOHRL outperforms benchmarks of the case without attention mechanism and conventional hybrid/multi-agent/standalone DRL. Moreover, AM-RIS in FD achieves the highest EE compared to half-duplex, conventional rigid antenna arrays, partial EH, and conventional RIS without amplification, highlighting its potential as a compelling solution for EE-aware wireless networks. 

---
# EuropeMedQA Study Protocol: A Multilingual, Multimodal Medical Examination Dataset for Language Model Evaluation 

**Authors**: Francesco Andrea Causio, Vittorio De Vita, Olivia Riccomi, Michele Ferramola, Federico Felizzi, Antonio Cristiano, Lorenzo De Mori, Chiara Battipaglia, Melissa Sawaya, Luigi De Angelis, Marcello Di Pumpo, Alessandra Piscitelli, Pietro Eric Risuleo, Alessia Longo, Giulia Vojvodic, Mariapia Vassalli, Bianca Destro Castaniti, Nicolò Scarsi, Manuel Del Medico  

**Link**: [PDF](https://arxiv.org/pdf/2604.14306)  

**Abstract**: While Large Language Models (LLMs) have demonstrated high proficiency on English-centric medical examinations, their performance often declines when faced with non-English languages and multimodal diagnostic tasks. This study protocol describes the development of EuropeMedQA, the first comprehensive, multilingual, and multimodal medical examination dataset sourced from official regulatory exams in Italy, France, Spain, and Portugal. Following FAIR data principles and SPIRIT-AI guidelines, we describe a rigorous curation process and an automated translation pipeline for comparative analysis. We evaluate contemporary multimodal LLMs using a zero-shot, strictly constrained prompting strategy to assess cross-lingual transfer and visual reasoning. EuropeMedQA aims to provide a contamination-resistant benchmark that reflects the complexity of European clinical practices and fosters the development of more generalizable medical AI. 

---
# Quantum-inspired tensor networks in machine learning models 

**Authors**: Guillermo Valverde, Igor García-Olaizola, Giannicola Scarpa, Alejandro Pozas-Kerstjens  

**Link**: [PDF](https://arxiv.org/pdf/2604.14287)  

**Abstract**: Tensor networks were developed in the context of many-body physics as compressed representations of multiparticle quantum states. These representations mitigate the exponential complexity of many-body systems by capturing only the most relevant dependencies. Due to the formal similarity between quantum entanglement and statistical correlations, tensor networks have recently been integrated in machine learning, operating both as alternative learning architectures and as decompositions of components of neural networks. The expectation is that the theoretical understanding of tensor networks developed within quantum many-body physics leads to novel methods that offer advantages in terms of computational efficiency, explainability, or privacy. Here we review the use of tensor networks in the context of machine learning, providing a critical assessment of the state of the art, the potential advantages, and the challenges that must be overcome. 

---
# Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization 

**Authors**: Junzhe Wang, Zhiheng Xi, yajie yang, Hao Luo, Shihan Dou, Tao Gui, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14267)  

**Abstract**: Search agents extend Large Language Models (LLMs) beyond static parametric knowledge by enabling access to up-to-date and long-tail information unavailable during pretraining. While reinforcement learning has been widely adopted for training such agents, existing approaches face key limitations: process supervision often suffers from unstable value estimation, whereas outcome supervision struggles with credit assignment due to sparse, trajectory-level rewards. To bridge this gap, we propose Contribution-Weighted GRPO (CW-GRPO), a framework that integrates process supervision into group relative policy optimization. Instead of directly optimizing process rewards, CW-GRPO employs an LLM judge to assess the retrieval utility and reasoning correctness at each search round, producing per-round contribution scores. These scores are used to rescale outcome-based advantages along the trajectory, enabling fine-grained credit assignment without sacrificing optimization stability. Experiments on multiple knowledge-intensive benchmarks show that CW-GRPO outperforms standard GRPO by 5.0\% on Qwen3-8B and 6.3\% on Qwen3-1.7B, leading to more effective search behaviors. Additional analysis reveals that successful trajectories exhibit concentrated contributions across rounds, providing empirical insight into search agent tasks. 

---
# Reinforcement Learning via Value Gradient Flow 

**Authors**: Haoran Xu, Kaiwen Hu, Somayeh Sojoudi, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14265)  

**Abstract**: We study behavior-regularized reinforcement learning (RL), where regularization toward a reference distribution (the dataset in offline RL or the base model in LLM RL finetuning) is essential to prevent value over-optimization caused by erroneous out-of-distribution extrapolation. Existing methods either rely on reparameterized policy gradient, which are difficult to scale to large generative models, or on reject sampling, which can be overly conservative when attempting to move beyond the behavior support. In this paper, we propose Value Gradient Flow (VGF), a scalable new paradigm for behavior-regularized RL. VGF casts behavior-regularized RL as an optimal transport problem that maps the reference distribution to the value-induced optimal policy distribution. We solve this transport problem via discrete gradient flow, where value gradients guide particles initialized from the reference distribution. Our analysis shows that VGF imposes regularization implicitly by controlling the transport budget. VGF eliminates explicit policy parameterization while remaining expressive and flexible, this enables adaptive test-time scaling by adjusting the transport budget. Extensive experiments demonstrate that VGF significantly outperforms prior methods, achieving state-of-the-art results on offline RL benchmarks (D4RL, OGBench) and LLM RL tasks. Code and runs can be found at this https URL. 

---
# GUI-Perturbed: Domain Randomization Reveals Systematic Brittleness in GUI Grounding Models 

**Authors**: Yangyue Wang, Harshvardhan Sikka, Yash Mathur, Tony Zhou, Jinu Nyachhyon, Pranav Guruprasad  

**Link**: [PDF](https://arxiv.org/pdf/2604.14262)  

**Abstract**: GUI grounding models report over 85% accuracy on standard benchmarks, yet drop 27-56 percentage points when instructions require spatial reasoning rather than direct element naming. Current benchmarks miss this because they evaluate each screenshot once with a single fixed instruction. We introduce GUI-Perturbed, a controlled perturbation framework that independently varies visual scenes and instructions to measure grounding robustness. Evaluating three 7B models from the same architecture lineage, we find that relational instructions cause systematic accuracy collapse across all models, a 70% browser zoom produces statistically significant degradation, and rank-8 LoRA fine-tuning with augmented data degrades performance rather than improving it. By perturbing along independent axes, GUI-Perturbed isolates which specific capability axes are affected-spatial reasoning, visual robustness, reasoning calibration-providing diagnostic signal that aggregate benchmarks cannot. We release the dataset, augmentation pipeline, and a fine-tuned model. 

---
# ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents 

**Authors**: Zhuofeng Li, Yi Lu, Dongfu Jiang, Haoxiang Zhang, Yuyang Bai, Chuan Li, Yu Wang, Shuiwang Ji, Jianwen Xie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14261)  

**Abstract**: The rapid rise in AI conference submissions has driven increasing exploration of large language models (LLMs) for peer review support. However, LLM-based reviewers often generate superficial, formulaic comments lacking substantive, evidence-grounded feedback. We attribute this to the underutilization of two key components of human reviewing: explicit rubrics and contextual grounding in existing work. To address this, we introduce REVIEWBENCH, a benchmark evaluating review text according to paper-specific rubrics derived from official guidelines, the paper's content, and human-written reviews. We further propose REVIEWGROUNDER, a rubric-guided, tool-integrated multi-agent framework that decomposes reviewing into drafting and grounding stages, enriching shallow drafts via targeted evidence consolidation. Experiments on REVIEWBENCH show that REVIEWGROUNDER, using a Phi-4-14B-based drafter and a GPT-OSS-120B-based grounding stage, consistently outperforms baselines with substantially stronger/larger backbones (e.g., GPT-4.1 and DeepSeek-R1-670B) in both alignment with human judgments and rubric-based review quality across 8 dimensions. The code is available \href{this https URL}{here}. 

---
# Evaluation of Agents under Simulated AI Marketplace Dynamics 

**Authors**: To Eun Kim, Alireza Salemi, Hamed Zamani, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.14256)  

**Abstract**: Modern information access ecosystems consist of mixtures of systems, such as retrieval systems and large language models, and increasingly rely on marketplaces to mediate access to models, tools, and data, making competition between systems inherent to deployment. In such settings, outcomes are shaped not only by benchmark quality but also by competitive pressure, including user switching, routing decisions, and operational constraints. Yet evaluation is still largely conducted on static benchmarks with accuracy-focused measures that assume systems operate in isolation. This mismatch makes it difficult to predict post-deployment success and obscures competitive effects such as early-adoption advantages and market dominance. We introduce Marketplace Evaluation, a simulation-based paradigm that evaluates information access systems as participants in a competitive marketplace. By simulating repeated interactions and evolving user and agent preferences, the framework enables longitudinal evaluation and marketplace-level metrics, such as retention and market share, that complement and can extend beyond traditional accuracy-based metrics. We formalize the framework and outline a research agenda, motivated by business and economics, around marketplace simulation, metrics, optimization, and adoption in evaluation campaigns like TREC. 

---
# Awakening Dormant Experts:Counterfactual Routing to Mitigate MoE Hallucinations 

**Authors**: Wentao Hu, Yanbo Zhai, Xiaohui Hu, Mingkuan Zhao, Shanhong yu, Xue Liu, Kaidong Yu, Shuangyong Song, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14246)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models have achieved remarkable scalability, yet they remain vulnerable to hallucinations, particularly when processing long-tail knowledge. We identify that this fragility stems from static Top-$k$ routing: routers tend to favor high-frequency patterns over rare factual associations. Consequently, ``specialist experts'' possessing critical long-tail knowledge are often assigned low gating scores and remain ``dormant'' -- under-prioritized for specific tokens despite their proven causal importance on other inputs. To address this, we propose Counterfactual Routing (CoR), a training-free inference framework designed to awaken these dormant experts. CoR integrates layer-wise perturbation analysis with the Counterfactual Expert Impact (CEI) metric to dynamically shift computational resources from syntax-dominant to knowledge-intensive layers while maintaining a constant total activation count, effectively retrieving causally decisive experts via virtual ablation. Extensive experiments on TruthfulQA, FACTOR, and TriviaQA demonstrate that CoR improves factual accuracy by 3.1\% on average without increasing the inference budget, establishing a superior Pareto frontier compared to static scaling strategies. 

---
# Optimistic Policy Learning under Pessimistic Adversaries with Regret and Violation Guarantees 

**Authors**: Sourav Ganguly, Kartik Pandit, Arnob Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14243)  

**Abstract**: Real-world decision-making systems operate in environments where state transitions depend not only on the agent's actions, but also on \textbf{exogenous factors outside its control}--competing agents, environmental disturbances, or strategic adversaries--formally, $s_{h+1} = f(s_h, a_h, \bar{a}_h)+\omega_h$ where $\bar{a}_h$ is the adversary/external action, $a_h$ is the agent's action, and $\omega_h$ is an additive noise. Ignoring such factors can yield policies that are optimal in isolation but \textbf{fail catastrophically in deployment}, particularly when safety constraints must be satisfied.
Standard Constrained MDP formulations assume the agent is the sole driver of state evolution, an assumption that breaks down in safety-critical settings. Existing robust RL approaches address this via distributional robustness over transition kernels, but do not explicitly model the \textbf{strategic interaction} between agent and exogenous factor, and rely on strong assumptions about divergence from a known nominal model.
We model the exogenous factor as an \textbf{adversarial policy} $\bar{\pi}$ that co-determines state transitions, and ask how an agent can remain both optimal and safe against such an adversary. \emph{To the best of our knowledge, this is the first work to study safety-constrained RL under explicit adversarial dynamics}. We propose \textbf{Robust Hallucinated Constrained Upper-Confidence RL} (\texttt{RHC-UCRL}), a model-based algorithm that maintains optimism over both agent and adversary policies, explicitly separating epistemic from aleatoric uncertainty. \texttt{RHC-UCRL} achieves sub-linear regret and constraint violation guarantees. 

---
# Graph-Based Fraud Detection with Dual-Path Graph Filtering 

**Authors**: Wei He, Wensheng Gan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14235)  

**Abstract**: Fraud detection on graph data can be viewed as a demanding task that requires distinguishing between different types of nodes. Because graph neural networks (GNNs) are naturally suited for processing information encoded in graph form through their message-passing operations, methods based on GNN models have increasingly attracted attention in the fraud detection domain. However, fraud graphs inherently exhibit relation camouflage, high heterophily, and class imbalance, causing most GNNs to underperform in fraud detection tasks. To address these challenges, this paper proposes a Graph-Based Fraud Detection Model with Dual-Path Graph Filtering (DPF-GFD). DPF-GFD first applies a beta wavelet-based operator to the original graph to capture key structural patterns. It then constructs a similarity graph from distance-based node representations and applies an improved low-pass filter. The embeddings from the original and similarity graphs are fused through supervised representation learning to obtain node features, which are finally used by an ensemble tree model to assess the fraud risk of unlabeled nodes. Unlike existing single-graph smoothing approaches, DPF-GFD introduces a frequency-complementary dual-path filtering paradigm tailored for fraud detection, explicitly decoupling structural anomaly modeling and feature similarity modeling. This design enables more discriminative and stable node representations in highly heterophilous and imbalanced fraud graphs. Comprehensive experiments on four real-world financial fraud detection datasets demonstrate the effectiveness of our proposed method. 

---
# Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the U.S. Banking Sector 

**Authors**: Mohammad Nasir Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2604.14232)  

**Abstract**: The Spatial-Temporal Graph Attention Network (ST-GAT) framework was created to serve as an explainable GNN-based solution for detecting bank distress early warning signs and for conducting macro-prudential surveillance of the interbank system in the United States. The ST-GAT framework models 8,103 FDIC insured institutions across 58 quarterly snapshots (2010Q1-2024Q2). Bilateral exposures were reconstructed from publicly available FDIC Call Reports using maximum entropy estimation to produce a dynamic directed weighted graph. The framework achieves the highest AUPRC among all GNN architectures (0.939 +/- 0.010), trailing only XGBoost (0.944). Ablation analysis confirms the BiLSTM temporal component contributes +0.020 AUPRC; temporal attention weights exhibit a monotonically decreasing pattern consistent with long-run structural vulnerability weighting. Permutation importance identifies ROA (0.309) and NPL Ratio (0.252) as dominant predictors, consistent with post-mortem analyses of the 2023 regional banking crisis. All data are publicly available FDIC Call Reports and FRED series; all code and results are released. 

---
# Shapley Value-Guided Adaptive Ensemble Learning for Explainable Financial Fraud Detection with U.S. Regulatory Compliance Validation 

**Authors**: Mohammad Nasir Uddin, Md Munna Aziz  

**Link**: [PDF](https://arxiv.org/pdf/2604.14231)  

**Abstract**: Financial crime costs U.S. institutions over $32 billion each year. Although AI tools for fraud detection have become more advanced, their use in real-world systems still faces a major obstacle: many of these models operate as black boxes that cannot provide the transparent, auditable explanations required by regulations such as OCC Bulletin 2011-12 and Federal Reserve SR 11-7. This study makes three main contributions. First, it offers a thorough evaluation of explanation quality across faithfulness (sufficiency and comprehensiveness at k=5, 10, and 15) and stability (Kendall's W across 30 bootstrap samples). XGBoost paired with TreeExplainer achieves near-perfect stability (W=0.9912), while LSTM with DeepExplainer shows weak results (W=0.4962). Second, the paper introduces the SHAP-Guided Adaptive Ensemble (SGAE), which dynamically adjusts per-transaction ensemble weights based on SHAP attribution agreement, achieving the highest AUC-ROC among all tested models (0.8837 held-out; 0.9245 cross-validation). Third, a complete three-architecture evaluation of LSTM, Transformer, and GNN-GraphSAGE on the full 590,540-transaction IEEE-CIS dataset is provided, with GNN-GraphSAGE achieving AUC-ROC 0.9248 and F1=0.6013. All results are mapped directly to OCC, SR 11-7, and BSA-AML regulatory compliance requirements. 

---
# Magnitude Is All You Need? Rethinking Phase in Quantum Encoding of Complex SAR Data 

**Authors**: Sakthi Prabhu Gunasekar, Prasanna Kumar R  

**Link**: [PDF](https://arxiv.org/pdf/2604.14229)  

**Abstract**: Synthetic Aperture Radar (SAR) data is inherently complex-valued, while quantum machine learning (QML) models naturally operate in complex Hilbert spaces. This apparent alignment suggests that incorporating both magnitude and phase information into quantum encoding should improve performance in SAR Automatic Target Recognition (ATR). In this work, we systematically evaluate this assumption by comparing five quantum encoding strategies: magnitude-only, joint complex, I/Q-based, preprocessed phase, and pure quantum, under a unified experimental framework on the MSTAR benchmark dataset.
Contrary to expectation, we observe a consistent pattern: in hybrid quantum-classical architectures, magnitude-only encoding outperforms all complex-valued strategies, achieving 99.57% accuracy on a 3-class task and 71.19% on an 8-class task, while phase-aware methods provide negligible (~0%) or negative improvements. In contrast, in purely quantum architectures with only 184-224 trainable parameters and no classical components, phase information becomes essential, contributing up to 21.65% improvement in accuracy.
These results reveal that the utility of phase information is not inherent to the data, but depends critically on the model architecture. Hybrid models rely on classical components that compensate for missing phase information, whereas purely quantum models require phase to construct discriminative representations. Our findings provide practical design guidelines for encoding complex-valued data in QML and highlight the importance of encoding-architecture co-design in the NISQ era. 

---
# Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems 

**Authors**: Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14228)  

**Abstract**: Claude Code is an agentic coding tool that can run shell commands, edit files, and call external services on behalf of the user. This study describes its comprehensive architecture by analyzing the publicly available TypeScript source code and further comparing it with OpenClaw, an independent open-source AI agent system that answers many of the same design questions from a different deployment context. Our analysis identifies five human values, philosophies, and needs that motivate the architecture (human decision authority, safety and security, reliable execution, capability amplification, and contextual adaptability) and traces them through thirteen design principles to specific implementation choices. The core of the system is a simple while-loop that calls the model, runs tools, and repeats. Most of the code, however, lives in the systems around this loop: a permission system with seven modes and an ML-based classifier, a five-layer compaction pipeline for context management, four extensibility mechanisms (MCP, plugins, skills, and hooks), a subagent delegation mechanism with worktree isolation, and append-oriented session storage. A comparison with OpenClaw, a multi-channel personal assistant gateway, shows that the same recurring design questions produce different architectural answers when the deployment context changes: from per-action safety classification to perimeter-level access control, from a single CLI loop to an embedded runtime within a gateway control plane, and from context-window extensions to gateway-wide capability registration. We finally identify six open design directions for future agent systems, grounded in recent empirical, architectural, and policy literature. 

---
# FRESCO: Benchmarking and Optimizing Re-rankers for Evolving Semantic Conflict in Retrieval-Augmented Generation 

**Authors**: Sohyun An, Hayeon Lee, Shuibenyang Yuan, Chun-cheng Jason Chen, Cho-Jui Hsieh, Vijai Mohan, Alexander Min  

**Link**: [PDF](https://arxiv.org/pdf/2604.14227)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a key approach to mitigating the temporal staleness of large language models (LLMs) by grounding responses in up-to-date evidence. Within the RAG pipeline, re-rankers play a pivotal role in selecting the most useful documents from retrieved candidates. However, existing benchmarks predominantly evaluate re-rankers in static settings and do not adequately assess performance under evolving information -- a critical gap, as real-world systems often must choose among temporally different pieces of evidence. To address this limitation, we introduce FRESCO (Factual Recency and Evolving Semantic COnflict), a benchmark for evaluating re-rankers in temporally dynamic contexts. By pairing recency-seeking queries with historical Wikipedia revisions, FRESCO tests whether re-rankers can prioritize factually recent evidence while maintaining semantic relevance. Our evaluation reveals a consistent failure mode across existing re-rankers: a strong bias toward older, semantically rich documents, even when they are factually obsolete. We further investigate an instruction optimization framework to mitigate this issue. By identifying Pareto-optimal instructions that balance Evolving and Non-Evolving Knowledge tasks, we obtain gains of up to 27% on Evolving Knowledge tasks while maintaining competitive performance on Non-Evolving Knowledge tasks. 

---
# TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations 

**Authors**: Ashmi Banerjee, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14223)  

**Abstract**: Traditional conversational travel recommender systems primarily optimize for user relevance and convenience, often reinforcing popular, overcrowded destinations and carbon-intensive travel choices. To address this, we present TRACE (Tourism Recommendation with Agentic Counterfactual Explanations), a multi-agent, LLM-based framework that promotes sustainable tourism through interactive nudging. TRACE uses a modular orchestrator-worker architecture where specialized agents elicit latent sustainability preferences, construct structured user personas, and generate recommendations that balance relevance with environmental impact.
A key innovation lies in its use of agentic counterfactual explanations and LLM-driven clarifying questions, which together surface greener alternatives and refine understanding of intent, fostering user reflection without coercion. User studies and semantic alignment analyses demonstrate that TRACE effectively supports sustainable decision-making while preserving recommendation quality and interactive responsiveness. TRACE is implemented on Google's Agent Development Kit, with full code, Docker setup, prompts, and a publicly available demo video to ensure reproducibility.
A project summary, including all resources, prompts, and demo access, is available at this https URL. 

---
# Adaptive Query Routing: A Tier-Based Framework for Hybrid Retrieval Across Financial, Legal, and Medical Documents 

**Authors**: Afshan Hashmi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14222)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the standard paradigm for grounding Large Language Model outputs in external knowledge. Lumer et al. [1] presented the first systematic evaluation comparing vector-based agentic RAG against hierarchical node-based reasoning systems for financial document QA across 1,200 SEC filings, finding vector-based systems achieved a 68% win rate. Concurrently, the PageIndex framework [2] demonstrated 98.7% accuracy on FinanceBench through purely reasoning-based retrieval. This paper extends their work by: (i) implementing and evaluating three retrieval architectures: Vector RAG, Tree Reasoning, and the proposed Adaptive Hybrid Retrieval (AHR) across financial, legal, and medical domains; (ii) introducing a four-tier query complexity benchmark; and (iii) employing GPT-4-powered LLM-as-judge evaluation. Experiments reveal that Tree Reasoning achieves the highest overall score (0.900), but no single paradigm dominates across all tiers: Vector RAG wins on multi-document synthesis (Tier 4, score 0.900), while the Hybrid AHR achieves the best performance on cross-reference (0.850) and multi-section queries (0.929). Cross-reference recall reaches 100% for tree-based and hybrid approaches versus 91.7% for vector search, quantifying a critical capability gap. Validation on FinanceBench (150 expert-annotated questions on real SEC 10-K and 10-Q filings) confirms and strengthens these findings: Tree Reasoning scores 0.938, Hybrid AHR 0.901, and Vector RAG 0.821, with the Tree--Vector quality gap widening to 11.7 percentage points on real-world documents. These findings support the development of adaptive retrieval systems that dynamically select strategies based on query complexity and document structure. All code and data are publicly available. 

---
# Knowledge Graph RAG: Agentic Crawling and Graph Construction in Enterprise Documents 

**Authors**: Koushik Chakraborty, Koyel Guha  

**Link**: [PDF](https://arxiv.org/pdf/2604.14220)  

**Abstract**: This research paper addresses the limitations of semantic search in complex enterprise document ecosystems. Traditional RAG pipelines often fail to capture hierarchical and interconnected information, leading to retrieval inaccuracies. We propose Agentic Knowledge Graphs featuring Recursive Crawling as a robust solution for navigating superseding logic and multi-hop references. Our benchmark evaluation using the Code of Federal Regulations (CFR) demonstrates that this Knowledge Graph-enhanced approach achieves a 70% accuracy improvement over standard vector-based RAG systems, providing exhaustive and precise answers for complex regulatory queries. 

---
# MEME-Fusion@CHiPSAL 2026: Multimodal Ablation Study of Hate Detection and Sentiment Analysis on Nepali Memes 

**Authors**: Samir Wagle, Reewaj Khanal, Abiral Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2604.14218)  

**Abstract**: Hate speech detection in Devanagari-scripted social media memes presents compounded challenges: multimodal content structure, script-specific linguistic complexity, and extreme data scarcity in low-resource settings. This paper presents our system for the CHiPSAL 2026 shared task, addressing both Subtask A (binary hate speech detection) and Subtask B (three-class sentiment classification: positive, neutral, negative). We propose a hybrid cross-modal attention fusion architecture that combines CLIP (ViT-B/32) for visual encoding with BGE-M3 for multilingual text representation, connected through 4-head self-attention and a learnable gating network that dynamically weights modality contributions on a per-sample basis. Systematic evaluation across eight model configurations demonstrates that explicit cross-modal reasoning achieves a 5.9% F1-macro improvement over text-only baselines on Subtask A, while uncovering two unexpected but critical findings: English-centric vision models exhibit near-random performance on Devanagari script, and standard ensemble methods catastrophically degrade under data scarcity (N nearly equal to 850 per fold) due to correlated overfitting. The code can be accessed at this https URL 

---
# Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis 

**Authors**: Aizierjiang Aiersilan, Mohamad Koubeissi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14216)  

**Abstract**: Predicting post-surgical seizure outcomes in pharmacoresistant epilepsy is a clinical challenge. Conventional deep-learning approaches operate on static, single-timepoint pre-operative scans, omitting longitudinal morphological changes. We propose \emph{Neuro-Oracle}, a three-stage framework that: (i) distils pre-to-post-operative MRI changes into a compact 512-dimensional trajectory vector using a 3D Siamese contrastive encoder; (ii) retrieves historically similar surgical trajectories from a population archive via nearest-neighbour search; and (iii) synthesises a natural-language prognosis grounded in the retrieved evidence using a quantized Llama-3-8B reasoning agent. Evaluations are conducted on the public EPISURG dataset ($N{=}268$ longitudinally paired cases) using five-fold stratified cross-validation. Since ground-truth seizure-freedom scores are unavailable, we utilize a clinical proxy label based on the resection type. We acknowledge that the network representations may potentially learn the anatomical features of the resection cavities (i.e., temporal versus non-temporal locations) rather than true prognostic morphometry. Our current evaluation thus serves mainly as a proof-of-concept for the trajectory-aware retrieval architecture. Trajectory-based classifiers achieve AUC values between 0.834 and 0.905, compared with 0.793 for a single-timepoint ResNet-50 baseline. The Neuro-Oracle agent (M5) matches the AUC of purely discriminative trajectory classifiers (0.867) while producing structured justifications with zero observed hallucinations under our audit protocol. A Siamese Diversity Ensemble (M6) of trajectory-space classifiers attains an AUC of 0.905 without language-model overhead. 

---
# PriHA: A RAG-Enhanced LLM Framework for Primary Healthcare Assistant in Hong Kong 

**Authors**: Richard Wai Cheung Chan, Shanru Lin, Ya-nan Ma, Hao Chen, Liangjun Jiang, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14215)  

**Abstract**: To address the unsustainable rise in public health expenditures, the Hong Kong SAR Government is shifting its strategic focus to primary healthcare and encouraging citizens to use community resources to self-manage their health. However, official clinical guidelines are fragmented across disparate departments and formats, creating significant access barriers. While general-purpose Large Language Models (LLMs) such as ChatGPT and DeepSeek offer potential solutions for information accessibility, they are prone to generating factually inaccurate content due to a lack of localized and domain-specific knowledge. To this end, we propose a Retrieval-Augmented Generation-Enhanced LLM system as Primary Healthcare Assistant (PriHA) in Hong Kong. Specifically, a tri-stage pipeline is proposed that leverages a query optimizer to generalize user intent-oriented sub-queries, followed by a novel Dual Retrieval Augmented Generation (DRAG) architecture for mixed-source retrieval and context-reorganized generation. Comprehensive experiments and a detailed case study demonstrate that our proposed method can outperform both ablations and baseline in terms of accuracy and clarity. Our research provides a reliable and traceable dialogue retrieval framework for exploring other high-risk, localized application scenarios. 

---
# CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization 

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2604.14214)  

**Abstract**: Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation. Existing automatic prompt optimization(APO) frameworks target task accuracy exclusively at the expense of generating long reasoning traces. We propose Cost-Regularized Optimization of Prompts (CROP), an APO method that introduces regularization on response length by generating textual feedback in addition to standard accuracy feedback. This forces the optimization process to produce prompts that elicit concise responses containing only critical information and reasoning. We evaluate our approach on complex reasoning datasets, specifically GSM8K, LogiQA and BIG-Bench Hard. We achieved an 80.6\% reduction in token consumption while maintaining competitive accuracy, seeing only a nominal decline in performance. This presents a pragmatic solution for deploying token-efficient and cost-effective agentic AI systems in production pipelines. 

---
# Ollivier-Ricci Curvature of Riemannian Manifolds and Directed Graphs with Applications to Graph Neural Networks 

**Authors**: Eleanor Wiesler  

**Link**: [PDF](https://arxiv.org/pdf/2604.14211)  

**Abstract**: This thesis is an exposition of Ollivier-Ricci Curvature of metric spaces as introduced by Yann Ollivier, which is based upon the 1-Wasserstein Distance and optimal transport theory. We present some of the major results and proofs that connect Ollivier-Ricci curvature with classical Ricci curvature of Riemannian manifolds, including extensions of various theoretical bounds and theorems such as Bonnet-Myers and Levy-Gromov. Then we shift to results introduced by Lin-Lu-Yau on an extension of Ollivier-Ricci curvature on graphs, as well as the work of Jost-Liu on proving various combinatorial bounds for graph Ollivier-Ricci curvature. At the end of this thesis we present novel ideas and proofs regarding extensions of these results to directed graphs, and finally applications of graph-based Ollivier-Ricci curvature to various algorithms in network science and graph machine learning. 

---
# Towards Verified and Targeted Explanations through Formal Methods 

**Authors**: Hanchen David Wang, Diego Manzanas Lopez, Preston K. Robinette, Ipek Oguz, Taylor T. Johnson, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.14209)  

**Abstract**: As deep neural networks are deployed in safety-critical domains such as autonomous driving and medical diagnosis, stakeholders need explanations that are interpretable but also trustworthy with formal guarantees. Existing XAI methods fall short: heuristic attribution techniques (e.g., LIME, Integrated Gradients) highlight influential features but offer no mathematical guarantees about decision boundaries, while formal methods verify robustness yet remain untargeted, analyzing the nearest boundary regardless of whether it represents a critical risk. In safety-critical systems, not all misclassifications carry equal consequences; confusing a "Stop" sign for a "60 kph" sign is far more dangerous than confusing it with a "No Passing" sign. We introduce ViTaX (Verified and Targeted Explanations), a formal XAI framework that generates targeted semifactual explanations with mathematical guarantees. For a given input (class y) and a user-specified critical alternative (class t), ViTaX: (1) identifies the minimal feature subset most sensitive to the y->t transition, and (2) applies formal reachability analysis to guarantee that perturbing these features by epsilon cannot flip the classification to t. We formalize this through Targeted epsilon-Robustness, certifying whether a feature subset remains robust under perturbation toward a specific target class. ViTaX is the first method to provide formally guaranteed explanations of a model's resilience against user-identified alternatives. Evaluations on MNIST, GTSRB, EMNIST, and TaxiNet demonstrate over 30% fidelity improvement with minimal explanation cardinality. 

---
# Disentangled Dual-Branch Graph Learning for Conversational Emotion Recognition 

**Authors**: Chengling Guo, Yuntao Shou, Tao Meng, Wei Ai, Yun Tan, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14204)  

**Abstract**: Multimodal emotion recognition in conversations aims to infer utterance-level emotions by jointly modeling textual, acoustic, and visual cues within context. Despite recent progress, key challenges remain, including redundant cross-modal information, imperfect semantic alignment, and insufficient modeling of high-order speaker interactions. To address these issues, we propose a framework that combines dual-space feature disentanglement with dual-branch graph learning. A shared encoder and modality-specific encoders are used to separate modality-invariant and modality-specific representations. The invariant features are modeled by a Fourier graph neural network to capture global consistency and complementary patterns, with a frequency-domain contrastive objective to enhance discriminability. In parallel, a speaker-aware hypergraph is constructed over modality-specific features to model high-order interactions, along with a speaker-consistency constraint to maintain coherent semantics. Finally, the two branches are fused for utterance-level emotion prediction. Experiments on IEMOCAP and MELD demonstrate that the proposed method achieves superior performance over strong baselines, validating its effectiveness. 

---
# Bridging scalp and intracranial EEG in BCI via pretrained neural representations and geometric constraint embedding 

**Authors**: Yihang Dong, Changhong Jing, Shuqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14202)  

**Abstract**: Electroencephalography (EEG) has become one of the key modalities underpinning brain-computer interfaces (BCIs) due to its high temporal resolution, rapid responsiveness, non-invasiveness, low cost, and portability. However, EEG signals are substantially inferior to intracranial EEG (iEEG) in signal-to-noise ratio and local spatial resolution, whereas iEEG suffers from extremely limited clinical accessibility owing to its invasive nature, hindering widespread application. To address this challenge, this study proposes a unified data-and prior knowledge-driven framework for EEG-iEEG representational enhancement. Guided by the principle that "geometric structure dictates function", the framework maps static cortical anatomy onto dynamic constraints governing neural signal propagation and integrates general-purpose neural representations extracted by a pre-trained large EEG model to explicitly model signal transmission through the brain. Enhanced EEG signals are then synthesized via a multidimensional representation diffusion process. Numerous experimental results demonstrate that the generated enhanced EEG signals effectively recover the neural activity patterns lost during propagation through the brain. This finding indicates that the performance ceiling of BCIs is constrained not only by acquisition hardware but also by the depth to which the generative model resolves the mechanisms of neural signal propagation. Collectively, the proposed framework provides a viable pathway toward acquiring high-fidelity neural signals at low cost. 

---
# Retina gap junctions support the robust perception by warping neural representational geometries along the visual hierarchy 

**Authors**: Yang Yue, Shenjian Zhang, Yonghong Tian, Kai Du, Tiejun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14200)  

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to elaborately designed adversarial noise, although they have achieved extraordinary success in many tasks. Compared with DNNs, the human visual system is highly robust. However, it is unclear how the human visual system defends against adversarial attacks, especially the role of the early visual system and its influence on the brain manifold. Due to retina gap junctions being crucial for the denoising function in the early visual system, we combine a retina gap junction-based filter, G-filter, with DNN as an abstract human visual system model called the biological hybrid model. We adopt this model to study the defense performance of retina gap junctions and their impact on the brain manifold. Compared with other defense methods, the biological hybrid model is more robust and can be further improved by introducing noise during training. Next, we analyze the manifold and its decision boundary of the biological hybrid model from a geometry perspective. The results show that the biological hybrid model has a unique 2D decision boundary with high nonlinearity and a lower curvature of the decision boundary of the manifold compared to other defense methods. The transforming manifold may account for the high robustness of the biological hybrid model. Finally, to dissect G-filter and clarify its internal mechanism, we borrow the Neural Ordinary Differential Equation (ODE) concept and rewrite G-filter into an equivalent recurrent neural network. The results show that the decision boundary of the model's manifold will gradually change with time and eventually reach a steady state, which is modulated by gap junction conductance, revealing the influence of retina gap junctions on the brain manifold is a gradually evolving process. 

---
# PolyBench: Benchmarking LLM Forecasting and Trading Capabilities on Live Prediction Market Data 

**Authors**: Pu Cheng, Juncheng Liu, Yunshen Long  

**Link**: [PDF](https://arxiv.org/pdf/2604.14199)  

**Abstract**: Predicting real-world events from live market signals demands systems that fuse qualitative news with quantitative order-book dynamics under strict temporal discipline -- a challenge existing benchmarks fail to capture. We present \textbf{PolyBench}, a multimodal benchmark derived from Polymarket that records point-in-time cross-sections of 38,666 binary prediction markets spanning 4,997 events, synchronously coupling each snapshot with a Central Limit Order Book (CLOB) state and a real-time news stream. Using PolyBench, we evaluate seven state-of-the-art Large Language Models -- spanning open- and closed-source families -- generating 36,165 predictions under identical, timestamp-locked market states collected between February 6 and 12, 2026. Our multidimensional framework assesses directional accuracy, our proposed Confidence-Weighted Return (CWR), Annualized Percentage Yield (APY), and Sharpe ratio via realistic order-book execution simulation. The results reveal a pronounced performance divergence: only two of seven models achieve positive financial returns -- MiMo-V2-Flash at \textbf{17.6%} CWR and Gemini-3-Flash at 6.2% CWR -- while the remaining five incur losses despite uniformly high stated confidence. These findings highlight the gap between surface-level language fluency and genuine probabilistic reasoning under live market uncertainty, and establish PolyBench as a contamination-proof, financially-grounded evaluation standard for future LLM research. Our dataset and code available at \underline{\href{this https URL}{this https URL}}. 

---
# MixAtlas: Uncertainty-aware Data Mixture Optimization for Multimodal LLM Midtraining 

**Authors**: Bingbing Wen, Sirajul Salekin, Feiyang Kang, Bill Howe, Lucy Lu Wang, Javier Movellan, Manjot Bilkhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14198)  

**Abstract**: Domain reweighting can improve sample efficiency and downstream generalization, but data-mixture optimization for multimodal midtraining remains largely unexplored. Current multimodal training recipes tune mixtures along a single dimension, typically data format or task type. We introduce MixAtlas, a method that produces benchmark-targeted data recipes that can be inspected, adapted, and transferred to new corpora. MixAtlas decomposes the training corpus along two axes: image concepts (10 visual-domain clusters discovered via CLIP embeddings) and task supervision (5 objective types including captioning, OCR, grounding, detection, and VQA). Using small proxy models (Qwen2-0.5B) paired with a Gaussian-process surrogate and GP-UCB acquisition, MixAtlas searches the resulting mixture space with the same proxy budget as regression-based baselines but finds better-performing mixtures. We evaluate on 10 benchmarks spanning visual understanding, document reasoning, and multimodal reasoning. On Qwen2-7B, optimized mixtures improve average performance by 8.5%-17.6% over the strongest baseline; on Qwen2.5-7B, gains are 1.0%-3.3%. Both settings reach baseline-equivalent training loss in up to 2 times fewer steps. Recipes discovered on 0.5B proxies transfer to 7B-scale training across Qwen model families. 

---
# The PICCO Framework for Large Language Model Prompting: A Taxonomy and Reference Architecture for Prompt Structure 

**Authors**: David A. Cook  

**Link**: [PDF](https://arxiv.org/pdf/2604.14197)  

**Abstract**: Large language model (LLM) performance depends heavily on prompt design, yet prompt construction is often described and applied inconsistently. Our purpose was to derive a reference framework for structuring LLM prompts. This paper presents PICCO, a framework derived through a rigorous synthesis of 11 previously published prompting frameworks identified through a multi-database search. The analysis yields two main contributions. First, it proposes a taxonomy that distinguishes prompt frameworks, prompt elements, prompt generation, prompting techniques, and prompt engineering as related but non-equivalent concepts. Second, it derives a five-element reference architecture for prompt generation: Persona, Instructions, Context, Constraints, and Output (PICCO). For each element, we define its function, scope, and relationship to other elements, with the goal of improving conceptual clarity and supporting more systematic prompt design. Finally, to support application of the framework, we outline key concepts relevant to implementation, including prompting techniques (e.g., zero-shot, few-shot, chain-of-thought, ensembling, decomposition, and self-critique, with selected variants), human and automated approaches to iterative prompt engineering, responsible prompting considerations such as security, privacy, bias, and trust, and priorities for future research. This work is a conceptual and methodological contribution: it formalizes a common structure for prompt specification and comparison, but does not claim empirical validation of PICCO as an optimization method. 

---
# Grading the Unspoken: Evaluating Tacit Reasoning in Quantum Field Theory and String Theory with LLMs 

**Authors**: Xingyang Yu, Yinghuan Zhang, Yufei Zhang, Zijun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.14188)  

**Abstract**: Large language models have demonstrated impressive performance across many domains of mathematics and physics. One natural question is whether such models can support research in highly abstract theoretical fields such as quantum field theory and string theory. Evaluating this possibility faces an immediate challenge: correctness in these domains is layered, tacit, and fundamentally non-binary. Standard answer-matching metrics fail to capture whether intermediate conceptual steps are properly reconstructed or whether implicit structural constraints are respected. We construct a compact expert-curated dataset of twelve questions spanning core areas of quantum field theory and string theory, and introduce a five-level grading rubric separating statement correctness, key concept awareness, reasoning chain presence, tacit step reconstruction, and enrichment. Evaluating multiple contemporary LLMs, we observe near-ceiling performance on explicit derivations within stable conceptual frames, but systematic degradation when tasks require reconstruction of omitted reasoning steps or reorganization of representations under global consistency constraints. These failures are driven not only by missing intermediate steps, but by an instability in representation selection: models often fail to identify the correct conceptual framing required to resolve implicit tensions. We argue that highly abstract theoretical physics provides a uniquely sensitive lens on the epistemic limits of current evaluation paradigms. 

---
# HARNESS: Lightweight Distilled Arabic Speech Foundation Models 

**Authors**: Vrunda N. Sukhadia, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2604.14186)  

**Abstract**: Large self-supervised speech (SSL) models achieve strong downstream performance, but their size limits deployment in resource-constrained settings. We present HArnESS, an Arabic-centric self-supervised speech model family trained from scratch with iterative self-distillation, together with lightweight student variants that offer strong accuracy-efficiency trade-offs on Automatic Speech Recognition (ASR), Dialect Identification (DID), and Speech Emotion Recognition (SER). Our approach begins with a large bilingual Arabic-English teacher and progressively distills its knowledge into compressed student models while preserving Arabic-relevant acoustic and paralinguistic representations. We further study PCA-based compression of the teacher supervision signal to better match the capacity of shallow and thin students. Compared with HuBERT and XLS-R, HArnESS consistently improves performance on Arabic downstream tasks, while the compressed models remain competitive under substantial structural reduction. These results position HArnESS as a practical and accessible Arabic-centric SSL foundation for real-world speech applications. 

---
# End-to-End Learning-based Operation of Integrated Energy Systems for Buildings and Data Centers 

**Authors**: Zhenyu Pu, Yu Yang, Liang Yu, Xiaohong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14184)  

**Abstract**: Buildings and data centers (DCs) are energy-intensive sectors, playing a critical role to achieve the low-carbon and sustainable energy transition targets. To this end, integrated energy system (IES) that incorporates diverse renewables, energy generation, conversion, and storage technologies to enable coordinated multi-energy supply have been widely investigated for both buildings and DCs. However, few works consider the two sectors jointly within IES to exploit their substantial synergistic benefits. Meanwhile, the operational optimization of IES remains challenging due to the difficulty to predict the multi-energy demand and supply accurately. To address these gaps, this paper investigates IES for coordinated multi-energy supply of buildings and DC, where the waste heat from DCs is recovered and reused to enhance energy efficiency. Moreover, an end-to-end learning-based method is proposed for the operational optimization of IES under uncertainty. Unlike conventional predict-then-optimize approaches, the proposed method integrates the training of prediction models for uncertain variables with the constrained optimization of IES into a unified learning framework, guiding the training of prediction models to improve operational performance, rather than prediction accuracy, thereby mitigating the impacts of predictions errors. Case studies based on real-world datasets show that the proposed methods improves the operational performance of IES by about 7-9% compared to existing predict-then-optimize methods. In addition, coordinating buildings and DCs within IES shows substantial economic benefits. In particular, the waste heat recovery from DCs leads to approximately 10% of total energy cost reduction of the IES. 

---
# Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model 

**Authors**: Jiuting Chen, Yuan Lian, Hao Wu, Tianqi Huang, Hiroshi Sasaki, Makoto Kouno, Jongil Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14180)  

**Abstract**: We train a 318M-parameter Transformer language model from scratch on a curated corpus of 1.56 billion tokens of pure Classical Chinese, with zero English characters or Arabic numerals. Through systematic out-of-distribution (OOD) testing, we investigate whether the model can distinguish known from unknown inputs, and crucially, whether it can express this distinction in its generated text.
We find a clear dissociation between internal and external uncertainty. Internally, the model exhibits a perplexity jump ratio of 2.39x between real and fabricated historical events (p = 8.9e-11, n = 92 per group), with semi-fabricated events (real figures + fictional events) showing the highest perplexity (4.24x, p = 1.1e-16), demonstrating genuine factual encoding beyond syntactic pattern matching. Externally, however, the model never learns to express uncertainty: classical Chinese epistemic markers appear at lower rates for OOD questions (3.5%) than for in-distribution questions (8.3%, p = 0.023), reflecting rhetorical conventions rather than genuine metacognition.
We replicate both findings across three languages (Classical Chinese, English, Japanese), three writing systems, and eight models from 110M to 1.56B parameters. We further show that uncertainty expression frequency is determined entirely by training data conventions, with Classical Chinese models showing a "humility paradox" (more hedging for known topics), while Japanese models almost never hedge. We argue that metacognitive expression -- the ability to say "I don't know" -- does not emerge from language modeling alone and requires explicit training signals such as RLHF. 

---
# An Underexplored Frontier: Large Language Models for Rare Disease Patient Education and Communication -- A scoping review 

**Authors**: Zaifu Zhan, Yu Hou, Kai Yu, Min Zeng, Anita Burgun, Xiaoyi Chen, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14179)  

**Abstract**: Rare diseases affect over 300 million people worldwide and are characterized by complex care pathways, limited clinical expertise, and substantial unmet communication needs throughout the long patient journey. Recent advances in large language models (LLMs) offer new opportunities to support patient education and communication, yet their application in rare diseases remains unclear.
We conducted a scoping review of studies published between January 2022 and March 2026 across major databases, identifying 12 studies on LLM-based rare disease patient education and communication. Data were extracted on study characteristics, application scenarios, model usage, and evaluation methods, and synthesized using descriptive and qualitative analyses.
The literature is highly recent and dominated by general-purpose models, particularly ChatGPT. Most studies focus on patient question answering using curated question sets, with limited use of real-world data or longitudinal communication scenarios. Evaluations are primarily centered on accuracy, with limited attention to patient-centered dimensions such as readability, empathy, and communication quality. Multilingual communication is rarely addressed.
Overall, the field remains at an early stage. Future research should prioritize patient-centered design, domain-adapted methods, and real-world deployment to support safe, adaptive, and effective communication in rare diseases. 

---
# Listen, Correct, and Feed Back: Spoken Pedagogical Feedback Generation 

**Authors**: Junhong Liang, Yifan Lu, Ekaterina Kochmar, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.14177)  

**Abstract**: Grammatical error correction (GEC) and explanation (GEE) have made rapid progress, but real teaching scenarios also require \emph{learner-friendly pedagogical feedback} that is actionable, level-appropriate, and encouraging. We introduce \textbf{SPFG} (\textbf{S}poken \textbf{P}edagogical \textbf{F}eedback \textbf{G}eneration), a dataset built based on the Speak \& Improve Challenge 2025 corpus, pairing fluency-oriented transcriptions with GEC targets and \emph{human-verified} teacher-style feedback, including preferred/rejected feedback pairs for preference learning. We study a transcript-based Spoken Grammatical Error Correction (SGEC) setting and evaluate three instruction-tuned LLMs (Qwen2.5, Llama-3.1, and GLM-4), comparing supervised fine-tuning (SFT) with preference-based alignment (using DPO and KTO) for jointly generating corrections and feedback. Results show that SFT provides the most consistent improvements, while DPO/KTO yield smaller or mixed gains, and that correction quality and feedback quality are weakly coupled. Our implementation is available at this https URL. 

---
# The Devil Is in Gradient Entanglement: Energy-Aware Gradient Coordinator for Robust Generalized Category Discovery 

**Authors**: Haiyang Zheng, Nan Pu, Yaqi Cai, Teng Long, Wenjing Li, Nicu Sebe, Zhun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2604.14176)  

**Abstract**: Generalized Category Discovery (GCD) leverages labeled data to categorize unlabeled samples from known or unknown classes. Most previous methods jointly optimize supervised and unsupervised objectives and achieve promising results. However, inherent optimization interference still limits their ability to improve further. Through quantitative analysis, we identify a key issue, i.e., gradient entanglement, which 1) distorts supervised gradients and weakens discrimination among known classes, and 2) induces representation-subspace overlap between known and novel classes, reducing the separability of novel categories. To address this issue, we propose the Energy-Aware Gradient Coordinator (EAGC), a plug-and-play gradient-level module that explicitly regulates the optimization process. EAGC comprises two components: Anchor-based Gradient Alignment (AGA) and Energy-aware Elastic Projection (EEP). AGA introduces a reference model to anchor the gradient directions of labeled samples, preserving the discriminative structure of known classes against the interference of unlabeled gradients. EEP softly projects unlabeled gradients onto the complement of the known-class subspace and derives an energy-based coefficient to adaptively scale the projection for each unlabeled sample according to its degree of alignment with the known subspace, thereby reducing subspace overlap without suppressing unlabeled samples that likely belong to known classes. Experiments show that EAGC consistently boosts existing methods and establishes new state-of-the-art results. Code is available at this https URL. 

---
# QU-NLP at ArchEHR-QA 2026: Two-Stage QLoRA Fine-Tuning of Qwen3-4B for Patient-Oriented Clinical Question Answering and Evidence Sentence Alignment 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14175)  

**Abstract**: We present a unified system addressing both Subtask 3 (answer generation) and Subtask 4 (evidence sentence alignment) of the ArchEHR-QA Shared Task. For Subtask 3, we apply two-stage Quantised Low-Rank Adaptation (QLoRA) to Qwen3-4B loaded in 4-bit NF4 quantisation: first on 30,000 samples from the emrQA-MedSQuAD corpus to establish clinical domain competence, then on the 20 annotated development cases to learn the task-specific output style. Our system achieves an overall score of 32.87 on the official test-2026 split (BLEU = 9.42, ROUGE-L = 27.04, SARI = 55.42, BERTScore = 43.00, AlignScore = 25.28, MEDCON = 37.04). For Subtask 4, we develop a weighted ensemble of three retrieval methods - BM25 with relative thresholding, TF-IDF cosine similarity, and a fine-tuned cross-encoder - to identify note sentences supporting a given gold answer, achieving a micro-F1 of 67.16 on the 100-case test set. Experiments reveal that both subtasks expose the same fundamental challenge: 20 annotated training cases are insufficient to distinguish relevant from irrelevant clinical sentences, pointing to data augmentation as the highest-leverage future direction. 

---
# Tug-of-War within A Decade: Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generations 

**Authors**: Ziyin Zhou, Jianyi Zhang, Xu ji, Yilong Li, Jiameng Han, Zhangchi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14172)  

**Abstract**: Large Language Models (LLMs) are essential for analyzing and addressing vulnerabilities in cybersecurity. However, among over 200,000 vulnerabilities were discovered in the past decade, more than 30,000 have been changed or updated. This necessitates frequent updates to the training datasets and internal knowledge bases of LLMs to maintain knowledge consistency. In this paper, we focus on the problem of knowledge discrepancy and conflict within CVE (Common Vulnerabilities and Exposures) detection and analysis. This problem hinders LLMs' ability to retrieve the latest knowledge from original training datasets, leading to knowledge conflicts, fabrications of factually incorrect results, and generation hallucinations. To address this problem, we propose an innovative two-stage framework called CRVA-TGRAG (Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generation). First, to improve document retrieval accuracy during the retrieval stage, we utilize Parent Document Segmentation and an ensemble retrieval scheme based on semantic similarity and inverted indexing. Second, to enhance LLMs' capabilities based on the retrieval of CVE dataset in generation stage, we employ a teacher-guided preference optimization technique to fine-tune LLMs. Our framework not only enhances the quality of content retrieval through RAG but also leverages the advantages of preference fine-tuning in LLMs to answer questions more effectively and precisely. Experiments demonstrate our method achieves higher accuracy in retrieving the latest CVEs compared to external knowledge bases. In conclusion, our framework significantly mitigates potential knowledge conflicts and inconsistencies that may arise from relying solely on LLMs for knowledge retrieval. 

---
# Benchmarking Linguistic Adaptation in Comparable-Sized LLMs: A Study of Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B on Romanized Nepali 

**Authors**: Ananda Rimal, Adarsha Rimal  

**Link**: [PDF](https://arxiv.org/pdf/2604.14171)  

**Abstract**: Romanized Nepali, the Nepali language written in the Latin alphabet, is the dominant medium for informal digital communication in Nepal, yet it remains critically underresourced in the landscape of Large Language Models (LLMs). This study presents a systematic benchmarking of linguistic adaptation across three comparable-sized open-weight models: Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B. We evaluate these architectures under zero-shot and fine-tuned settings using a curated bilingual dataset of 10,000 transliterated instruction-following samples. Performance is quantified across five metrics spanning seven measurement dimensions: Perplexity (PPL), BERTScore, chrF++, ROUGE-1, ROUGE-2, ROUGE-L, and BLEU, capturing fluency, phonetic consistency, and semantic integrity. Models were fine-tuned using Quantized Low-Rank Adaptation (QLoRA) with Rank-Stabilized LoRA (rsLoRA) at rank r=32 on dual NVIDIA Tesla T4 GPUs, training only approximately 1% of each model's parameters in under 27 total GPU-hours. At zero-shot, all three models fail to generate Romanized Nepali, each exhibiting a distinct architecture-specific failure mode. Following fine-tuning, all three resolve these failures and converge to BERTScore approximately 0.75 and chrF++ greater than 23. Overall dimension-wise assessment across ten criteria identifies Qwen3-8B as the overall recommended architecture, being the only model to produce semantically relevant zero-shot output and leading all structural alignment metrics post-SFT. The adaptation headroom hypothesis is confirmed: Llama-3.1-8B, despite its weakest zero-shot baseline, achieves the largest absolute fine-tuning gains in PPL (Delta = -49.77) and BERTScore (Delta = +0.3287), making it the preferred choice for iterative low-resource development pipelines. This work establishes the first rigorous baseline for Romanized Nepali adaptation in comparable-sized open-weight LLMs. 

---
# Stateful Evidence-Driven Retrieval-Augmented Generation with Iterative Reasoning 

**Authors**: Qi Dong, Ziheng Lin, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.14170)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds Large Language Models (LLMs) in external knowledge but often suffers from flat context representations and stateless retrieval, leading to unstable performance. We propose Stateful Evidence-Driven RAG with Iterative Reasoning, a framework that models question answering as a progressive evidence accumulation process. Retrieved documents are converted into structured reasoning units with explicit relevance and confidence signals and maintained in a persistent evidence pool capturing both supportive and non-supportive information. The framework performs evidence-driven deficiency analysis to identify gaps and conflicts and iteratively refines queries to guide subsequent retrieval. This iterative reasoning process enables stable evidence aggregation and improves robustness to noisy retrieval. Experiments on multiple question answering benchmarks demonstrate consistent improvements over standard RAG and multi-step baselines, while effectively accumulating high-quality evidence and maintaining stable performance under substantial retrieval noise. 

---
# SAGE Celer 2.6 Technical Card 

**Authors**: SAGEA Research Team, Basab Jha, Firoj Paudel, Ujjwal Puri, Adrian Liu, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14168)  

**Abstract**: We introduce SAGE Celer 2.6, the latest in our line of general-purpose Celer models from SAGEA. Celer 2.6 is available in 5B, 10B, and 27B parameter sizes and benefits from extensive architectural modifications and further pre-training on an undisclosed model. Using our Inverse Reasoning (IR) pipeline, SAGEA natively trains Celer 2.6 to validate its own logic paths, minimizing cascading error and hallucination in complex reasoning tasks. Celer 2.6 also boasts natively integrated multimodal functionality with an end-to-end vision encoder to avoid common pitfalls in adapter-based approaches. Celer 2.6 provides highly competitive results on mathematics, coding, and general intelligence benchmarks (ACUMEN), along with low latency. Most importantly, Celer 2.6 is specifically optimized for South Asian language support, with a custom tokenizer for the Devanagari script and strong performance in both Nepali and Hindi without sacrificing English reasoning ability. 

---
# Chinese Essay Rhetoric Recognition Using LoRA, In-context Learning and Model Ensemble 

**Authors**: Yuxuan Lai, Xiajing Wang, Chen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14167)  

**Abstract**: Rhetoric recognition is a critical component in automated essay scoring. By identifying rhetorical elements in student writing, AI systems can better assess linguistic and higher-order thinking skills, making it an essential task in the area of AI for education. In this paper, we leverage Large Language Models (LLMs) for the Chinese rhetoric recognition task. Specifically, we explore Low-Rank Adaptation (LoRA) based fine-tuning and in-context learning to integrate rhetoric knowledge into LLMs. We formulate the outputs as JSON to obtain structural outputs and translate keys to Chinese. To further enhance the performance, we also investigate several model ensemble methods. Our method achieves the best performance on all three tracks of CCL 2025 Chinese essay rhetoric recognition evaluation task, winning the first prize. 

---
# SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models 

**Authors**: Tomer Atia, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2604.14163)  

**Abstract**: Maritime distress communications transmitted over very high frequency (VHF) radio are safety-critical voice messages used to report emergencies at sea. Under the Global Maritime Distress and Safety System (GMDSS), such messages follow standardized procedures and are expected to convey essential details, including vessel identity, position, nature of the distress, and required assistance. In practice, however, automatic analysis remains difficult because distress messages are often brief, noisy, and produced under stress, may deviate from the prescribed format, and are further degraded by automatic speech recognition (ASR) errors caused by channel noise and speaker stress. This paper presents SeaAlert, an LLM-based framework for robust analysis of maritime distress communications. To address the scarcity of labeled real-world data, we develop a synthetic data generation pipeline in which an LLM produces realistic and diverse maritime messages, including challenging variants in which standard distress codewords are omitted or replaced with less explicit expressions. The generated utterances are synthesized into speech, degraded with simulated VHF noise, and transcribed by an ASR system to obtain realistic noisy transcripts. 

---
# Can Large Language Models Detect Methodological Flaws? Evidence from Gesture Recognition for UAV-Based Rescue Operation Based on Deep Learning 

**Authors**: Domonkos Varga  

**Link**: [PDF](https://arxiv.org/pdf/2604.14161)  

**Abstract**: Reliable evaluation is essential in machine learning research, yet methodological flaws-particularly data leakage-continue to undermine the validity of reported results. In this work, we investigate whether large language models (LLMs) can act as independent analytical agents capable of identifying such issues in published studies. As a case study, we analyze a gesture-recognition paper reporting near-perfect accuracy on a small, human-centered dataset. We first show that the evaluation protocol is consistent with subject-level data leakage due to non-independent training and test splits. We then assess whether this flaw can be detected independently by six state-of-the-art LLMs, each analyzing the original paper without prior context using an identical prompt. All models consistently identify the evaluation as flawed and attribute the reported performance to non-independent data partitioning, supported by indicators such as overlapping learning curves, minimal generalization gap, and near-perfect classification results. These findings suggest that LLMs can detect common methodological issues based solely on published artifacts. While not definitive, their consistent agreement highlights their potential as complementary tools for improving reproducibility and supporting scientific auditing. 

---
# HUOZIIME: An On-Device LLM-enhanced Input Method for Deep Personalization 

**Authors**: Baocai Shan, Yuzhuang Xu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2604.14159)  

**Abstract**: Mobile input method editors (IMEs) are the primary interface for text input, yet they remain constrained to manual typing and struggle to produce personalized text. While lightweight large language models (LLMs) make on-device auxiliary generation feasible, enabling deeply personalized, privacy-preserving, and real-time generative IMEs poses fundamental this http URL this end, we present HUOZIIME, a personalized on-device IME powered by LLM. We endow HUOZIIME with initial human-like prediction ability by post-training a base LLM on synthesized personalization data. Notably, a hierarchical memory mechanism is designed to continually capture and leverage user-specific input history. Furthermore, we perform systemic optimizations tailored to on-device LLMbased IME deployment, ensuring efficient and responsive operation under mobile this http URL demonstrate efficient on-device execution and high-fidelity memory-driven personalization. Code and package are available at this https URL. 

---
# MemGround: Long-Term Memory Evaluation Kit for Large Language Models in Gamified Scenarios 

**Authors**: Yihang Ding, Wanke Xia, Yiting Zhao, Jinbo Su, Jialiang Yang, Zhengbo Zhang, Ke Wang, Wenming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14158)  

**Abstract**: Current evaluations of long-term memory in LLMs are fundamentally static. By fixating on simple retrieval and short-context inference, they neglect the multifaceted nature of complex memory systems, such as dynamic state tracking and hierarchical reasoning in continuous interactions. To overcome these limitations, we propose MemGround, a rigorous long-term memory benchmark natively grounded in rich, gamified interactive scenarios. To systematically assess these capabilities, MemGround introduces a three-tier hierarchical framework that evaluates Surface State Memory, Temporal Associative Memory, and Reasoning-Based Memory through specialized interactive tasks. Furthermore, to comprehensively quantify both memory utilization and behavioral trajectories, we propose a multi-dimensional metric suite comprising Question-Answer Score (QA Overall), Memory Fragments Unlocked (MFU), Memory Fragments with Correct Order (MFCO), and Exploration Trajectory Diagrams (ETD). Extensive experiments reveal that state-of-the-art LLMs and memory agents still struggle with sustained dynamic tracking, temporal event association, and complex reasoning derived from long-term accumulated evidence in interactive environments. 

---
# An Edge-Cloud Collaborative Architecture for Proactive Elderly Care: Real-Time Risk Assessment and Three-Level Emergency Response 

**Authors**: Lijie Zhou, Luran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14154)  

**Abstract**: The rapid aging of global populations has created an urgent need for intelligent healthcare monitoring systems to ensure the safety of elderly individuals living independently. Existing cloud-centric platforms face critical limitations, including high latency unsuitable for emergency response, privacy risks from continuous transmission of sensitive data, and limited, single-channel alert mechanisms lacking scalability and context awareness. This paper proposes an edge-cloud collaborative architecture that addresses these challenges through real-time multi-modal sensor fusion, a four-dimensional risk assessment model, and a three-level emergency response system. The framework adopts a five-layer design - device, edge, service, data, and application layers - enabling real-time risk evaluation with end-to-end alert latency under three seconds. At the edge, a weighted multi-modal fusion algorithm integrates data from five sensor types with confidence propagation. A unified risk score is generated by combining fall probability, physiological indicators, behavioral patterns, and sensor anomaly metrics. Based on dynamic thresholds, a three-tier notification system coordinates responses among family members, community doctors, and nearby volunteers. Experiments on CASAS, MIMIC-III, and SisFall datasets show that the approach achieves 91% activity recognition accuracy and an 84% anomaly detection F1-score, outperforming single-sensor methods. Deployment on Raspberry Pi 4 gateways demonstrates sub-100 ms inference latency while preserving privacy by keeping raw data local. This architecture advances practical, privacy-preserving, and responsive elderly care systems. 

---
# From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioto Review in Ambient AI Scribe Documentation 

**Authors**: Abdolamir Karbalaie, Fernando Seoane, Farhad Abtahi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14152)  

**Abstract**: Ambient AI "scribe" systems promise to reduce clinical documentation burden, but automatic speech recognition (ASR) errors can remain unnoticed without careful review, and high-quality human reference transcripts are often unavailable for calibrating uncertainty. We investigate whether cross-model disagreement among heterogeneous ASR systems can act as a reference-free uncertainty signal to prioritize human verification in medical transcription workflows. Using 50 publicly available medical education audio clips (8 h 14 min), we transcribed each clip with eight ASR systems spanning commercial APIs and open-source engines. We aligned multi-model outputs, built consensus pseudo-references, and quantified token-level agreement using a majority-strength metric; we further characterized disagreements by type (content vs. punctuation/formatting) and assessed per-model agreement via leave-one-model-out (jackknife) consensus scoring. Inter-model reliability was low (ICC[2,1] = 0.131), indicating heterogeneous failure modes across systems. Across 76,398 evaluated token positions, 72.1% showed near-unanimous agreement (7-8 models), while 2.5% fell into high-risk bands (0-3 models), with high-risk mass varying from 0.7% to 11.4% across accent groups. Low-agreement regions were enriched for content disagreements, with the content fraction increasing from 53.9% to 73.9% across quintiles of high-risk mass. These results suggest that cross-model disagreement provides a sparse, localizable signal that can surface potentially unreliable transcript spans without human-verified references, enabling targeted review; clinical accuracy of flagged regions remains to be established. 

---
# Gaussian Process Regression of Steering Vectors With Physics-Aware Deep Composite Kernels for Augmented Listening 

**Authors**: Diego Di Carlo, Shoichi Koyama, Nugraha Aditya Arie, Fontaine Mathieu, Bando Yoshiaki, Yoshii Kazuyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2509.02571)  

**Abstract**: This paper investigates continuous representations of steering vectors over frequency and microphone/source positions for augmented listening (e.g., spatial filtering and binaural rendering), enabling user-parameterized control of the reproduced sound field. Steering vectors have typically been used for representing the spatial response of a microphone array as a function of the look-up direction. The basic algebraic representation of these quantities assuming an idealized environment cannot deal with the scattering effect of the sound field. One may thus collect a discrete set of real steering vectors measured in dedicated facilities and super-resolve (i.e., upsample) them. Recently, physics-aware deep learning methods have been effectively used for this purpose. Such deterministic super-resolution, however, suffers from the overfitting problem due to the non-uniform uncertainty over the measurement space. To solve this problem, we integrate an expressive representation based on the neural field (NF) into the principled probabilistic framework based on the Gaussian process (GP). Specifically, we propose a physics-aware composite kernel that models the directional incoming waves and the subsequent scattering effect. Our comprehensive comparative experiment showed the effectiveness of the proposed method under data insufficiency conditions. In downstream tasks such as speech enhancement and binaural rendering using the simulated data of the SPEAR challenge, the oracle performances were attained with less than ten times fewer measurements. 

---
